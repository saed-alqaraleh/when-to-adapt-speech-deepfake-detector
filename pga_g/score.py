"""
Score a corpus with the frozen SSL-AASIST checkpoint and save, per clip:
logit_spoof, logit_bona, score = logit_bona - logit_spoof (the convention of Zhou & Wang
2026), the 160-d AASIST readout and the time-mean 1024-d XLS-R features.

Speed: parallel audio decoding (DataLoader workers, pinned memory, prefetch), SDPA
attention, bf16/fp16 autocast when a measured check shows it does not move the scores,
OOM-adaptive micro-batching, and a background thread that writes shards to Drive while
the GPU keeps working.

Restartability: work is split into fixed shards written atomically to the run directory;
a rerun skips every valid shard and the whole stage when _STAGE_SUCCESS.json matches the
configuration hash. Nothing already written is overwritten.

    python -u -m pga_g.score --dataset itw --data-root DATA_ROOT/itw \
        --ckpt WORKSPACE/checkpoints/LA_model.pth \
        --run-dir WORKSPACE/runs/score_v1/itw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (CUT, ITW_EXPECTED, LA19_EVAL_EXPECTED, AudioDataset, build_itw_manifest,
                   build_la19_eval_manifest, check_expected, collate, write_manifest)

BUILDERS = {"itw": (build_itw_manifest, ITW_EXPECTED),
            "la19eval": (build_la19_eval_manifest, LA19_EVAL_EXPECTED)}
SUCCESS = "_STAGE_SUCCESS.json"
# Bump only when the scoring computation changes; packaging/path fixes must not force rescoring.
SCORING_LOGIC_VERSION = "1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    cache = Path(str(path) + ".sha256")
    st = path.stat()
    if cache.exists():
        rec = json.loads(cache.read_text())
        if rec.get("size") == st.st_size and rec.get("mtime") == int(st.st_mtime):
            return rec["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    digest = h.hexdigest()
    try:
        cache.write_text(json.dumps({"size": st.st_size, "mtime": int(st.st_mtime), "sha256": digest}))
    except OSError:
        pass
    return digest


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def atomic_npy(path: Path, arr) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


# ------------------------------------------------------------------------- shards
def shard_ranges(n: int, size: int):
    return [(i, s, min(s + size, n)) for i, s in enumerate(range(0, n, size))]


def shard_path(run_dir: Path, i: int) -> Path:
    return run_dir / "shards" / f"shard_{i:05d}.npz"


def shard_valid(path: Path, start: int, end: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            idx = z["idx"]
            ok = (idx.shape[0] == end - start and np.array_equal(idx, np.arange(start, end))
                  and np.isfinite(z["logits"]).all() and np.isfinite(z["emb160"]).all())
        return bool(ok)
    except Exception:
        return False


# ------------------------------------------------------------------------- model / precision
def load_model(ckpt: Path, device, attn: str):
    from .aasist import load_official_checkpoint
    model, report = load_official_checkpoint(ckpt, attn=attn, map_location="cpu")
    return model.to(device).eval(), report


def make_forward(model, device, precision: str):
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    def fwd(wav):
        with torch.inference_mode():
            if dtype is not None:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits, emb, sslm = model(wav)
            else:
                logits, emb, sslm = model(wav)
        return logits.float(), emb.float(), sslm.float()
    return fwd


def decide_precision(model, device, ds: AudioDataset, requested: str, tol: float, n_check: int) -> dict:
    """'auto': compare a reduced-precision run against strict fp32 on the first clips and
    keep it only if the largest score difference is within tol (logit units)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type != "cuda":
        return {"precision": "fp32", "reason": "cpu"}
    if requested != "auto":
        return {"precision": requested, "reason": "requested"}
    cand = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    k = min(n_check, len(ds))
    f32, fc = make_forward(model, device, "fp32"), make_forward(model, device, cand)
    s32, sc = [], []
    for a in range(0, k, 16):                      # small chunks: fp32 activations are large
        wav = torch.from_numpy(np.stack([ds[i][0] for i in range(a, min(a + 16, k))])).to(device)
        s32.append(f32(wav)[0])
        sc.append(fc(wav)[0])
    s32, sc = torch.cat(s32), torch.cat(sc)
    d32 = (s32[:, 1] - s32[:, 0])
    dc = (sc[:, 1] - sc[:, 0])
    max_diff = float((d32 - dc).abs().max())
    spread = float(d32.std())
    keep = max_diff <= tol
    return {"precision": cand if keep else "fp32", "candidate": cand, "n_check": k,
            "max_abs_score_diff": max_diff, "score_std_fp32": spread, "tol": tol,
            "reason": "within tolerance" if keep else "difference above tolerance; strict fp32"}


# ------------------------------------------------------------------------- main stage
def run(args) -> int:
    import torch

    run_dir = Path(args.run_dir)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    ckpt = Path(args.ckpt)

    builder, expected = BUILDERS[args.dataset]
    log(f"building manifest for {args.dataset} from {data_root}")
    man = builder(data_root)
    if args.limit:
        man = man.iloc[: args.limit].copy()
    man_csv = man.to_csv(index=False).encode()
    man_sha = hashlib.sha256(man_csv).hexdigest()
    man_path = run_dir / "manifest.csv"
    if man_path.exists():
        old_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()
        if old_sha != man_sha:
            raise SystemExit(f"manifest differs from the one already in {run_dir}; "
                             "use a new --run-dir rather than mixing runs")
    else:
        write_manifest(man, man_path)
    counts = check_expected(man, expected)
    log(f"manifest: {counts['got']} (published: {expected}) -> "
        f"{'MATCH' if counts['match'] else 'DIFFERENT, check before trusting G0'}")

    ckpt_sha = sha256_file(ckpt)
    config = {"dataset": args.dataset, "n": len(man), "manifest_sha256": man_sha,
              "ckpt_sha256": ckpt_sha, "cut": CUT, "precision_request": args.precision,
              "shard_size": args.shard_size, "scoring_logic_version": SCORING_LOGIC_VERSION}
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    succ = run_dir / SUCCESS
    if succ.exists() and json.loads(succ.read_text()).get("config_hash") == cfg_hash:
        log(f"stage already complete ({succ}); nothing to do")
        return 0

    shards = shard_ranges(len(man), args.shard_size)
    todo = [(i, s, e) for i, s, e in shards if not shard_valid(shard_path(run_dir, i), s, e)]
    log(f"{len(shards)} shards, {len(shards) - len(todo)} already valid, {len(todo)} to do")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    model, report = load_model(ckpt, device, args.attn)
    log(f"checkpoint loaded strictly ({report['n_loaded']} tensors; ignored {len(report['ignored'])}"
        f" pre-training-only keys) on {device}"
        + (f" [{torch.cuda.get_device_name(0)}]" if device.type == "cuda" else ""))

    ds_all = AudioDataset(data_root, man.relpath.values, man.idx.values)
    prec_path = run_dir / "precision.json"
    if prec_path.exists():
        prec = json.loads(prec_path.read_text())
        log(f"reusing precision decision: {prec['precision']}")
    else:
        prec = decide_precision(model, device, ds_all, args.precision, args.precision_tol, args.precision_check_n)
        atomic_json(prec_path, prec)
        log(f"precision decision: {prec}")
    fwd = make_forward(model, device, prec["precision"])

    t0 = time.time()
    if todo:
        order = np.concatenate([np.arange(s, e) for _, s, e in todo])
        ds = AudioDataset(data_root, man.relpath.values[order], man.idx.values[order])
        workers = args.workers if args.workers >= 0 else min(16, os.cpu_count() or 2)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=workers,
            collate_fn=collate, pin_memory=device.type == "cuda",
            persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
            drop_last=False)
        log(f"scoring {len(order)} clips | batch {args.batch_size} | {workers} decode workers | "
            f"precision {prec['precision']} | attention {args.attn}")

        writer = ThreadPoolExecutor(max_workers=2)
        futures = []
        shard_of = {}
        for i, s, e in todo:
            shard_of.update({j: (i, s, e) for j in range(s, e)})
        buffers = {i: {} for i, _, _ in todo}
        micro = args.batch_size
        done = 0
        last = time.time()

        def flush(i, s, e):
            buf = buffers.pop(i)
            idx = np.arange(s, e)
            arr = lambda key: np.stack([buf[j][key] for j in idx])  # noqa: E731
            futures.append(writer.submit(
                atomic_npz, shard_path(run_dir, i), idx=idx, logits=arr("logits").astype(np.float32),
                emb160=arr("emb").astype(np.float32), ssl_mean=arr("sslm").astype(np.float16),
                sr=arr("sr").astype(np.int32), n_orig=arr("n").astype(np.int64)))

        for wav, idx, sr, n in loader:
            wav = wav.to(device, non_blocking=True)
            outs = []
            k = 0
            while k < wav.shape[0]:
                try:
                    outs.append(fwd(wav[k:k + micro]))
                    k += micro
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
                    log(f"CUDA OOM -> micro-batch {micro}")
            logits = torch.cat([o[0] for o in outs]).cpu().numpy()
            emb = torch.cat([o[1] for o in outs]).cpu().numpy()
            sslm = torch.cat([o[2] for o in outs]).cpu().numpy()
            for r, j in enumerate(idx.tolist()):
                i, s, e = shard_of[j]
                buffers[i][j] = {"logits": logits[r], "emb": emb[r], "sslm": sslm[r],
                                 "sr": int(sr[r]), "n": int(n[r])}
                if len(buffers[i]) == e - s:
                    flush(i, s, e)
            done += len(idx)
            if time.time() - last > args.log_every or done == len(order):
                rate = done / (time.time() - t0)
                eta = (len(order) - done) / max(rate, 1e-9)
                log(f"{done}/{len(order)} clips ({100 * done / len(order):.1f}%) | "
                    f"{rate:.0f} clips/s | ETA {eta / 60:.1f} min")
                last = time.time()
        for f in futures:
            f.result()
        writer.shutdown()
        if buffers:
            raise RuntimeError(f"incomplete shards left in memory: {list(buffers)}")

    # ---------------------------------------------------------------- assemble + validate
    bad = [i for i, s, e in shards if not shard_valid(shard_path(run_dir, i), s, e)]
    if bad:
        raise RuntimeError(f"invalid shards after scoring: {bad}")
    parts = [np.load(shard_path(run_dir, i)) for i, _, _ in shards]
    logits = np.concatenate([p["logits"] for p in parts])
    emb = np.concatenate([p["emb160"] for p in parts])
    sslm = np.concatenate([p["ssl_mean"] for p in parts])
    sr = np.concatenate([p["sr"] for p in parts])
    n_orig = np.concatenate([p["n_orig"] for p in parts])
    idx = np.concatenate([p["idx"] for p in parts])
    assert np.array_equal(idx, man.idx.values), "shard order does not match the manifest"

    out = man.copy()
    out["logit_spoof"] = logits[:, 0]
    out["logit_bona"] = logits[:, 1]
    out["score"] = logits[:, 1] - logits[:, 0]
    out["score_official"] = logits[:, 1]
    out["orig_sr"] = sr
    out["orig_len"] = n_orig
    tmp = run_dir / "scores.csv.tmp"
    out.to_csv(tmp, index=False)
    tmp.replace(run_dir / "scores.csv")
    atomic_npy(run_dir / "emb160.npy", emb)
    atomic_npy(run_dir / "ssl_mean1024.npy", sslm)

    from .metrics import eer_and_threshold
    eer, tau = eer_and_threshold(out.score[out.label == 1].values, out.score[out.label == 0].values)
    marker = {
        "stage": f"score_{args.dataset}", "config": config, "config_hash": cfg_hash,
        "counts_check": counts, "precision": prec, "checkpoint_report": {
            k: (v if k != "ignored" else len(v)) for k, v in report.items()},
        "eer_on_this_set": eer, "eer_threshold_on_this_set": tau,
        "outputs": ["manifest.csv", "scores.csv", "emb160.npy", "ssl_mean1024.npy", "shards/"],
        "runtime_s_this_session": round(time.time() - t0, 1),
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_json(succ, marker)
    log(f"done: EER on this set {100 * eer:.2f}% | outputs in {run_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=-1, help="-1 = min(16, CPU count)")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    ap.add_argument("--precision-tol", type=float, default=0.05,
                    help="max |score difference| vs fp32 allowed for reduced precision")
    ap.add_argument("--precision-check-n", type=int, default=128)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--log-every", type=float, default=20.0, help="seconds between progress lines")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N clips only")
    ap.add_argument("--cpu", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
