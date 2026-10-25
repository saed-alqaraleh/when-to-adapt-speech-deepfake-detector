"""
XLSR-Mamba (Xiao & Das, IEEE Signal Processing Letters 2025) in plain PyTorch.

The authors' code (github.com/swagshaw/XLSR-Mamba, commit 10ec7381) needs fairseq and the
compiled mamba-ssm 1.1.4 kernels, neither of which installs reliably on current Colab images.
This module reproduces the same computation without them:

* front end: XLS-R 300M in Hugging Face form, loaded from the checkpoint's fairseq weights with
  the renaming already verified for SSL-AASIST (pga_g.aasist);
* Mamba: mamba_ssm 1.1.4 `Mamba` (slow path) with the selective scan of `selective_scan_ref`,
  computed step by step without materialising the (B, D, L, N) tensors;
* norms: mamba_ssm `RMSNorm` / `rms_norm_fn` semantics (fused add + RMS norm, residual in fp32);
* blocks and head: `MixerModel` of the authors' mamba_blocks.py, including its final fused norm,
  which adds the model *input* (not the last block output) to the residual stream.

Parameter names equal the authors' state dict, so their checkpoint loads strictly.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aasist import build_hf_wav2vec2, convert_official_state_dict


def rms_norm(x, weight, eps, residual=None, prenorm=False, residual_in_fp32=False):
    """mamba_ssm.ops.triton.layernorm.rms_norm_fn (no bias), computed in fp32."""
    dtype = x.dtype
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    res_out = xf if residual_in_fp32 else xf.to(dtype)
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    y = (xf * rstd * weight.float()).to(dtype)
    return (y, res_out) if prenorm else y


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.register_parameter("bias", None)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def selective_scan(u, delta, A, B, C, D, z, delta_bias):
    """selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True) with
    variable B, C of shape (b, n, l); the state is updated step by step."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    delta = F.softplus(delta + delta_bias[..., None].float())
    B, C = B.float(), C.float()
    b, d, L = u.shape
    x = u.new_zeros((b, d, A.shape[1]))
    du = delta * u
    ys = []
    for i in range(L):
        x = torch.exp(delta[:, :, i, None] * A) * x + du[:, :, i, None] * B[:, None, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y + u * D[:, None]
    out = out * F.silu(z.float())
    return out.to(dtype_in)


class Mamba(nn.Module):
    """mamba_ssm 1.1.4 Mamba, inference path without the fused kernels."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, conv_bias=True, bias=False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                            .repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, h):
        L = h.shape[1]
        xz = self.in_proj(h).transpose(1, 2)                       # (b, 2*d_inner, L)
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :L])
        x_dbl = self.x_proj(x.transpose(1, 2))                     # (b, L, r + 2n)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ self.dt_proj.weight.t()).transpose(1, 2)        # (b, d_inner, L), bias added in the scan
        y = selective_scan(x, dt, A, Bm.transpose(1, 2), Cm.transpose(1, 2), self.D.float(), z,
                           self.dt_proj.bias.float())
        return self.out_proj(y.transpose(1, 2))


class Block(nn.Module):
    """mamba_ssm Block with fused_add_norm=True, RMSNorm, residual_in_fp32=True."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.mixer = Mamba(d_model)
        self.norm = RMSNorm(d_model, eps)

    def forward(self, h, residual=None):
        h, residual = rms_norm(h, self.norm.weight, self.norm.eps, residual=residual, prenorm=True,
                               residual_in_fp32=True)
        return self.mixer(h), residual


class MixerModel(nn.Module):
    """The authors' bidirectional MixerModel (mamba_blocks.py), inference path."""

    def __init__(self, d_model=144, n_layer=6, eps=1e-5):
        super().__init__()
        self.forward_layers = nn.ModuleList([Block(d_model, eps) for _ in range(n_layer)])
        self.backward_layers = nn.ModuleList([Block(d_model, eps) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model, eps)
        self.f_attention_pool = nn.Linear(d_model, 1)
        self.b_attention_pool = nn.Linear(d_model, 1)
        self.LL = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        f_h, f_res = x, None
        b_h, b_res = x.flip([1]), None
        for layer in self.forward_layers:
            f_h, f_res = layer(f_h, f_res)
        for layer in self.backward_layers:
            b_h, b_res = layer(b_h, b_res)
        # authors' fused final norm: the model input x (not f_h / b_h) is added to the residual
        f_h = rms_norm(x, self.norm_f.weight, self.norm_f.eps, residual=f_res, residual_in_fp32=True)
        b_h = rms_norm(x, self.norm_f.weight, self.norm_f.eps, residual=b_res, residual_in_fp32=True)
        f_h = torch.matmul(F.softmax(self.f_attention_pool(f_h), dim=1).transpose(-1, -2), f_h).squeeze(-2)
        b_h = torch.matmul(F.softmax(self.b_attention_pool(b_h), dim=1).transpose(-1, -2), b_h).squeeze(-2)
        emb = self.LL(torch.cat((f_h, b_h), dim=1))
        return self.classifier(emb), emb


class XLSRMamba(nn.Module):
    """forward(wav) -> logits (genuine = index 1), 144-d pre-classifier embedding, time-mean
    of the 1,024-d XLS-R features."""

    def __init__(self, attn: str = "sdpa", emb_size: int = 144, num_encoders: int = 12):
        super().__init__()
        self.ssl = build_hf_wav2vec2(attn)
        self.LL = nn.Linear(1024, emb_size)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MixerModel(d_model=emb_size, n_layer=num_encoders // 2)

    def forward(self, wav):
        feats = self.ssl(wav).last_hidden_state
        x = self.LL(feats).unsqueeze(1)
        x = self.selu(self.first_bn(x)).squeeze(1)
        logits, emb = self.conformer(x)
        return logits, emb, feats.mean(dim=1)


def load_xlsr_mamba(sd: Dict[str, torch.Tensor], attn: str = "sdpa"):
    """Build XLSRMamba and load the authors' state dict strictly (fairseq front end renamed)."""
    model = XLSRMamba(attn=attn)
    model_sd = model.state_dict()
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.endswith("LL_hidden.weight") and not k.endswith("LL_hidden.bias")}
    new_sd, report = convert_official_state_dict(sd, model_sd.keys())
    if report["missing_in_checkpoint"] or report["unexpected_in_checkpoint"]:
        raise RuntimeError("XLSR-Mamba checkpoint does not match the architecture: "
                           f"missing={report['missing_in_checkpoint'][:8]} "
                           f"unexpected={report['unexpected_in_checkpoint'][:8]}")
    bad = [k for k, v in new_sd.items() if tuple(v.shape) != tuple(model_sd[k].shape)]
    if bad:
        raise RuntimeError(f"XLSR-Mamba shape mismatch for {bad[:8]}")
    model.load_state_dict(new_sd, strict=True)
    return model.eval(), report
