"""
Frozen detectors for the generalisation test (v3). Every detector maps a batch of 64,600-sample
waveforms (16 kHz, official crop/tile) to genuine/spoof logits (index 1 = genuine) and named
feature arrays; the score is logit_genuine - logit_spoof, as for SSL-AASIST.

  aasist  official AASIST (Jung et al., ICASSP 2022), ASVspoof 2019 LA weights from
          github.com/clovaai/aasist commit a04c9863 (MIT); checkpoint pinned by SHA-256,
          obtained separately from upstream and not redistributed with this package.
          Features: emb160 = `last_hidden`, the 160-d readout before the output layer.
  mamba   XLSR-Mamba (Xiao & Das, IEEE SPL 2025), authors' checkpoint trained on ASVspoof 2019 LA
          (huggingface.co/AustinXiao/XLSR-Mamba-DF, revision pinned at first use).
          Features: ssl1024 = time-mean of the fine-tuned XLS-R output; emb144 = pre-classifier
          embedding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AASIST_WEIGHTS = HERE / "weights" / "AASIST.pth"
AASIST_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
AASIST_CONFIG = {"architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
                 "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]], "gat_dims": [64, 32],
                 "pool_ratios": [0.5, 0.7, 0.5, 0.5], "temperatures": [2.0, 2.0, 100.0, 100.0]}
MAMBA_REPO, MAMBA_FILE = "AustinXiao/XLSR-Mamba-DF", "model.safetensors"
MAMBA_PUBLISHED_DF21 = ("https://raw.githubusercontent.com/swagshaw/XLSR-Mamba/"
                        "10ec73810d24091028a2aa5454c8dfd1a239e441/Scores/DF/Bmamba3_LA_WCE_1e-06_ES144_NE12.txt")

# feature name -> (dimension, storage dtype)
FEATURES = {"aasist": {"emb160": (160, "float32")},
            "mamba": {"ssl1024": (1024, "float16"), "emb144": (144, "float32")},
            "sslaasist": {"ssl1024": (1024, "float16"), "emb160": (160, "float32")}}
PRIMARY = {"aasist": "emb160", "mamba": "ssl1024", "sslaasist": "ssl1024"}
SECONDARY = {"aasist": None, "mamba": "emb144", "sslaasist": "emb160"}
NAMES = {"sslaasist": "SSL-AASIST", "aasist": "AASIST", "mamba": "XLSR-Mamba"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# Input length of each detector's official evaluation recipe (official crop/tile, 16 kHz):
# AASIST and SSL-AASIST use 64,600 samples; the XLSR-Mamba authors evaluate ASVspoof 2021 with
# 66,800 samples (data_utils.Dataset_eval; 208 XLS-R frames). The first 64,600 samples of the
# 66,800-sample crop/tile equal the 64,600-sample crop/tile, so one decoded array serves both.
CUTS = {"aasist": 64600, "sslaasist": 64600, "mamba": 66800}


class Detector:
    """name, model (eval mode), cut, forward(wav) -> (logits float32 (B,2), {feature: tensor});
    forward receives the first `cut` samples of each waveform."""

    def __init__(self, name, model, run, info):
        self.name, self.model, self._run, self.info = name, model, run, info
        self.cut = CUTS[name]

    def forward(self, wav):
        return self._run(wav[:, : self.cut])


def build_aasist(device):
    import torch
    try:
        from .third_party.aasist_official import Model
    except ImportError as exc:
        raise ImportError("Obtain the official AASIST model from its upstream repository before running detector inference; model code and weights are not distributed here.") from exc
    if _sha256(AASIST_WEIGHTS) != AASIST_SHA256:
        raise SystemExit(f"{AASIST_WEIGHTS} does not match the official AASIST weights")
    model = Model(AASIST_CONFIG)
    sd = torch.load(AASIST_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    def run(wav):
        last_hidden, out = model(wav)
        return out, {"emb160": last_hidden}
    return Detector("aasist", model, run, {"source": "github.com/clovaai/aasist@a04c9863 models/weights/AASIST.pth",
                                           "sha256": AASIST_SHA256, "trained_on": "ASVspoof 2019 LA"})


def mamba_revision(token=None) -> str:
    from huggingface_hub import HfApi
    return HfApi().model_info(MAMBA_REPO, token=token).sha


def fetch_mamba(local_dir: Path, revision: str, token=None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(MAMBA_REPO, MAMBA_FILE, revision=revision, local_dir=str(local_dir), token=token))


def build_mamba(device, ckpt: Path, attn: str = "sdpa"):
    from safetensors.torch import load_file
    from .mamba_pt import load_xlsr_mamba
    model, report = load_xlsr_mamba(load_file(str(ckpt)), attn=attn)
    model = model.to(device).eval()

    def run(wav):
        logits, emb, sslm = model(wav)
        return logits, {"ssl1024": sslm, "emb144": emb}
    return Detector("mamba", model, run, {"source": f"huggingface.co/{MAMBA_REPO}", "file": str(ckpt),
                                          "sha256": _sha256(Path(ckpt)), "load_report": {
                                              k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()},
                                          "trained_on": "ASVspoof 2019 LA", "input_samples": CUTS["mamba"]})


def make_forward(det: Detector, device, precision: str):
    """Inference wrapper; returns float32 outputs. precision: fp32 (strict), tf32 (matmul + cuDNN TF32),
    cudnn_tf32 (PyTorch's default GPU setting: TF32 cuDNN convolutions, fp32 matmul), bf16, fp16."""
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    def fwd(wav):
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        torch.backends.cudnn.allow_tf32 = precision in ("tf32", "cudnn_tf32")
        with torch.inference_mode():
            if dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    lg, feats = det.forward(wav)
            else:
                lg, feats = det.forward(wav)
        return lg.float(), {k: v.float() for k, v in feats.items()}
    return fwd


def decide_precision(det: Detector, device, wavs: np.ndarray, requested: str, tol: float) -> dict:
    """Fastest of bf16/fp16 -> TF32 -> fp32 whose genuine-minus-spoof logit stays within tol of
    strict fp32 on the given clips (the same rule as the SSL-AASIST runs)."""
    import torch
    if device.type != "cuda":
        return {"precision": "fp32", "reason": "cpu"}
    if requested != "auto":
        return {"precision": requested, "reason": "requested"}

    def scores(prec):
        f = make_forward(det, device, prec)
        out = []
        for a in range(0, len(wavs), 16):
            lg = f(torch.from_numpy(wavs[a:a + 16]).to(device))[0]
            out.append((lg[:, 1] - lg[:, 0]).cpu())
        return torch.cat(out)
    ref = scores("fp32")
    diffs = {}
    for prec in ["bf16" if torch.cuda.is_bf16_supported() else "fp16", "tf32"]:
        diffs[prec] = float((scores(prec) - ref).abs().max())
        if diffs[prec] <= tol:
            return {"precision": prec, "max_abs_score_diff": diffs, "tol": tol, "n_check": int(len(wavs)),
                    "reason": "within tolerance"}
    return {"precision": "fp32", "max_abs_score_diff": diffs, "tol": tol, "n_check": int(len(wavs)),
            "reason": "no faster precision within tolerance; strict fp32"}


def info_json(dets) -> str:
    return json.dumps({d.name: d.info for d in dets}, indent=2, default=str)
