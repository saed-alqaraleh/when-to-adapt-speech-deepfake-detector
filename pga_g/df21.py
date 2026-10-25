"""
ASVspoof 2021 DF scoring across several Colab runtimes at once.

Data: huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF, 80 parquet
shards (34.4 GB, ~612k clips; FLAC bytes + label + JSON notes with speaker, codec, subset,
attack, vocoder, source). The repository revision is pinned in the run's config.json by the
first worker, so every worker scores the same data.

Parallelism
  across runtimes : worker k of N owns shards s with s % N == k; when its own shards are done
                    it takes over shards whose claim heartbeat is older than --stale-min
                    (a worker that died or was disconnected). Scoring is deterministic, so a
                    rare double-claim only repeats work; files are written atomically.
  inside a runtime: the next shard downloads in a background thread while the current one is
                    scored; audio is decoded by DataLoader worker processes from in-memory
                    FLAC bytes; GPU inference uses SDPA attention, checked mixed precision and
                    OOM-adaptive micro-batches; part files are written by a background thread.

Resume: one part file per shard on Drive (runs/df21_v1/parts/part_index.npz). A restarted worker
skips every valid part. `merge` assembles the corpus once all 80 parts are valid.

Silence-trimmed variant (--trim-db 40): leading/trailing silence is removed with
librosa.effects.trim (top_db 40, as Mueller et al. 2021) before the official crop/tile; the run
lives in runs/df21trim40_v1, reuses the base run's pinned revision and shard list, and stores the
leading-silence length of every clip.

    python -u -m pga_g.df21 work   --workspace W --worker-id 0 --n-workers 3
    python -u -m pga_g.df21 status --workspace W
    python -u -m pga_g.df21 merge  --workspace W [--wait]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BytesAudioDataset, collate

REPO = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF"
BASE_RUN = "df21_v1"
RUN = BASE_RUN
TRIM_DB = 0.0


def configure(trim_db: float = 0.0) -> None:
    """Select the base run or the silence-trimmed run (module-level, set once by main())."""
    global RUN, TRIM_DB
    TRIM_DB = float(trim_db or 0.0)
    RUN = BASE_RUN if not TRIM_DB else f"df21trim{int(TRIM_DB)}_v1"


def dataset_name() -> str:
    return RUN.rsplit("_v", 1)[0]          # df21 | df21trim40


def inputs_dir(ws: Path) -> Path:
    return ws / "results" / f"{dataset_name()}_inputs"
TEXT_FIELDS = ["utt_id", "speaker", "codec", "subset", "attack", "vocoder", "source"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


# ---------------------------------------------------------------- remote access (patchable)
def _list_shards(token=None):
    from huggingface_hub import HfApi
    api = HfApi()
    sha = api.dataset_info(REPO, token=token).sha
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", revision=sha, token=token)
                   if f.startswith("data/test-") and f.endswith(".parquet"))
    return sha, files


def _download(fname: str, sha: str, local_dir: Path, token=None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, fname, repo_type="dataset", revision=sha,
                                local_dir=str(local_dir), token=token))


# ---------------------------------------------------------------- run configuration
def run_dir(ws: Path) -> Path:
    return ws / "runs" / RUN


def ensure_config(ws: Path, ckpt_sha: str | None = None) -> dict:
    """First worker pins the repository revision and shard list; later workers reuse it."""
    rd = run_dir(ws)
    cfg_path = rd / "config.json"
    base_cfg = ws / "runs" / BASE_RUN / "config.json"
    if not cfg_path.exists() and RUN != BASE_RUN and base_cfg.exists():
        base = json.loads(base_cfg.read_text())           # same data as the confirmatory run
        _atomic_json(cfg_path, {**base, "trim_db": TRIM_DB, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "created_by": socket.gethostname(), "pinned_from": BASE_RUN})
    if not cfg_path.exists():
        sha, files = _list_shards(os.environ.get("HF_TOKEN") or None)
        if not files:
            raise RuntimeError(f"no parquet shards found in {REPO}")
        _atomic_json(cfg_path, {"repo": REPO, "revision": sha, "shards": files, "n_shards": len(files),
                                "ckpt_sha256": ckpt_sha, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "created_by": socket.gethostname()})
        time.sleep(2)                       # let a concurrent first write settle, then re-read
    cfg = json.loads(cfg_path.read_text())
    if ckpt_sha and cfg.get("ckpt_sha256") not in (None, ckpt_sha):
        raise SystemExit("this run was started with a different checkpoint")
    return cfg


def part_path(ws: Path, i: int) -> Path:
    return run_dir(ws) / "parts" / f"part_{i:03d}.npz"


def claim_path(ws: Path, i: int) -> Path:
    return run_dir(ws) / "claims" / f"shard_{i:03d}.json"


def part_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            n = int(json.loads(str(z["meta"]))["rows"])
            return (z["logits"].shape == (n, 2) and z["emb160"].shape == (n, 160)
                    and z["utt_id"].shape == (n,) and np.isfinite(z["logits"]).all())
    except Exception:
        return False


def read_claim(ws: Path, i: int) -> dict | None:
    try:
        return json.loads(claim_path(ws, i).read_text())
    except Exception:
        return None


# ---------------------------------------------------------------- shard decoding
def _our_label(v) -> int:
    """Mirror coding 0 = bonafide, 1 = spoof (ClassLabel) -> ours 1 = bona fide, 0 = spoof."""
    if v in (0, "bonafide"):
        return 1
    if v in (1, "spoof"):
        return 0
    return -1

def read_shard(path: Path):
    """Return (blobs, metadata DataFrame) from one parquet shard."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["path", "audio", "label", "notes"])
    audio = t.column("audio").combine_chunks()
    blobs = audio.field("bytes").to_pylist() if hasattr(audio, "field") else [a["bytes"] for a in audio.to_pylist()]
    labels = t.column("label").to_pylist()
    notes = [json.loads(n) if n else {} for n in t.column("notes").to_pylist()]
    paths = t.column("path").to_pylist()
    meta = pd.DataFrame({
        "utt_id": [n.get("utterance_id") or Path(p).stem for n, p in zip(notes, paths)],
        "speaker": [n.get("speaker_id", "unknown") for n in notes],
        "codec": [n.get("codec", "-") for n in notes],
        "subset": [n.get("subset", "-") for n in notes],
        "attack": [n.get("attack_id", "-") for n in notes],
        "vocoder": [n.get("vocoder", "-") for n in notes],
        "source": [n.get("source", "-") for n in notes],
        "label": [_our_label(l) for l in labels],
    })
    if (meta.label < 0).any() or any(b is None for b in blobs):
        raise ValueError(f"{path.name}: unexpected label values or missing audio bytes")
    return blobs, meta


# ---------------------------------------------------------------- worker
class Heartbeat:
    def __init__(self, ws: Path, i: int, worker: str, every: float = 60.0):
        self.ws, self.i, self.worker, self.every = ws, i, worker, every
        self.stop = threading.Event()
        self.started = time.time()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _write(self):
        _atomic_json(claim_path(self.ws, self.i), {"worker": self.worker, "started": self.started,
                                                  "heartbeat": time.time()})

    def _run(self):
        while not self.stop.wait(self.every):
            try:
                self._write()
            except OSError:
                pass

    def __enter__(self):
        self._write()
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


def score_shard(blobs, fwd, device, batch_size: int, workers: int, state: dict, trim_db: float = 0.0):
    import torch
    loader = torch.utils.data.DataLoader(
        BytesAudioDataset(blobs, trim_db or None), batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, pin_memory=device.type == "cuda", prefetch_factor=4 if workers else None,
        persistent_workers=False)
    n = len(blobs)
    logits = np.zeros((n, 2), np.float32)
    emb = np.zeros((n, 160), np.float32)
    sslm = np.zeros((n, 1024), np.float16)
    sr_a = np.zeros(n, np.int32)
    len_a = np.zeros(n, np.int64)
    lead_a = np.full(n, -1, np.int64)
    kept_a = np.full(n, -1, np.int64)
    for batch in loader:
        wav, idx, sr, ln = batch[:4]
        wav = wav.to(device, non_blocking=True)
        outs, k = [], 0
        while k < wav.shape[0]:
            try:
                outs.append(fwd(wav[k:k + state["micro"]]))
                k += state["micro"]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if state["micro"] == 1:
                    raise
                state["micro"] = max(1, state["micro"] // 2)
                log(f"CUDA OOM -> micro-batch {state['micro']}")
        ii = idx.numpy()
        logits[ii] = torch.cat([o[0] for o in outs]).cpu().numpy()
        emb[ii] = torch.cat([o[1] for o in outs]).cpu().numpy()
        sslm[ii] = torch.cat([o[2] for o in outs]).cpu().numpy().astype(np.float16)
        sr_a[ii] = sr.numpy()
        len_a[ii] = ln.numpy()
        if len(batch) > 4:
            lead_a[ii] = batch[4].numpy()
            kept_a[ii] = batch[5].numpy()
    return logits, emb, sslm, sr_a, len_a, lead_a, kept_a


def write_part(path: Path, meta: pd.DataFrame, arrays, info: dict) -> None:
    logits, emb, sslm, sr_a, len_a, lead_a, kept_a = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, logits=logits, emb160=emb, ssl_mean=sslm, sr=sr_a, n_orig=len_a,
             trim_lead=lead_a, trim_kept=kept_a,
             label=meta.label.to_numpy(np.int8),
             **{f: np.asarray(meta[f].astype(str).tolist(), dtype=np.str_) for f in TEXT_FIELDS},
             meta=np.array(json.dumps({**info, "rows": int(len(meta))})))
    tmp.replace(path)


def work(args) -> int:
    import torch
    from .score import decide_precision, load_model, make_forward, sha256_file

    ws = Path(args.workspace)
    if not (ws / "results" / "PREREGISTRATION_PGA.json").exists():
        from .pga import prereg      # the hypotheses go to Drive before the first DF21 score
        prereg(ws)
    local = Path(args.local) / "df21"
    local.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    cfg = ensure_config(ws, sha256_file(ckpt))
    shards, sha = cfg["shards"], cfg["revision"]
    worker = f"w{args.worker_id}@{socket.gethostname()}"
    token = os.environ.get("HF_TOKEN") or None
    log(f"{worker}: {RUN}: {len(shards)} shards in {REPO} @ {sha[:10]}; I own every shard with index % "
        f"{args.n_workers} == {args.worker_id}" + (f"; silence trimmed at {TRIM_DB:g} dB" if TRIM_DB else ""))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    model, _ = load_model(ckpt, device, args.attn)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    workers = args.workers if args.workers >= 0 else min(16, os.cpu_count() or 2)
    prec_file = run_dir(ws) / "precision" / f"{socket.gethostname()}_w{args.worker_id}.json"
    prec = json.loads(prec_file.read_text()) if prec_file.exists() else None
    fwd = None
    state = {"micro": args.batch_size}
    downloader = ThreadPoolExecutor(max_workers=1)
    writer = ThreadPoolExecutor(max_workers=1)
    prefetch = {}

    def fetch(i):
        return _download(shards[i], sha, local, token)

    def next_shard():
        todo = [i for i in range(len(shards)) if not part_valid(part_path(ws, i))]
        mine = [i for i in todo if i % args.n_workers == args.worker_id and i not in busy]
        if mine:
            return mine[0], mine[1:2]
        now = time.time()
        steal = []
        for i in todo:
            if i in busy:
                continue
            c = read_claim(ws, i)
            if c is None or now - float(c.get("heartbeat", 0)) > args.stale_min * 60:
                steal.append(i)
        steal.sort(reverse=True)            # owners work upwards, so take over from the end
        return (steal[0], steal[1:2]) if steal else (None, [])

    busy: set = set()
    pending = []                      # (future, shard, local parquet, clips, start time)
    t0, n_done, clips = time.time(), 0, 0

    def finalize(block: bool):
        nonlocal n_done, clips
        for item in list(pending):
            fut, i, path, n, ts = item
            if not (block or fut.done()):
                continue
            fut.result()
            if not part_valid(part_path(ws, i)):
                raise RuntimeError(f"part {i} failed validation after writing")
            try:
                path.unlink()                         # free local disk only once the part is safe
            except OSError:
                pass
            busy.discard(i)
            pending.remove(item)
            n_done += 1
            clips += n
            left = sum(1 for k in range(len(shards)) if not part_valid(part_path(ws, k)))
            log(f"{worker}: shard {i:03d} saved ({n} clips, {time.time() - ts:.0f}s) | {n_done} shards "
                f"here, {clips / (time.time() - t0):.0f} clips/s | {left} shards left overall")

    while True:
        finalize(block=False)
        i, ahead = next_shard()
        if i is None:
            if pending:
                finalize(block=True)
                continue
            break
        busy.add(i)
        for j in ahead:                                          # overlap download with scoring
            if j not in prefetch:
                prefetch[j] = downloader.submit(fetch, j)
        with Heartbeat(ws, i, worker):
            ts = time.time()
            path = (prefetch.pop(i) if i in prefetch else downloader.submit(fetch, i)).result()
            blobs, meta = read_shard(path)
            if fwd is None:
                if prec is None:
                    from .data import official_pad, load_audio_bytes

                    class _Probe:
                        def __len__(self):
                            return min(128, len(blobs))

                        def __getitem__(self, k):
                            return (official_pad(load_audio_bytes(blobs[k])[0]), k, 0, 0)
                    prec = decide_precision(model, device, _Probe(), args.precision, args.precision_tol, 128)
                    _atomic_json(prec_file, {**prec, "gpu": gpu})
                log(f"{worker}: precision {prec['precision']} ({prec.get('reason')}) on {gpu}")
                fwd = make_forward(model, device, prec["precision"])
            arrays = score_shard(blobs, fwd, device, args.batch_size, workers, state, TRIM_DB)
            del blobs
            info = {"shard": i, "file": shards[i], "revision": sha, "worker": worker, "gpu": gpu, "trim_db": TRIM_DB,
                    "precision": prec["precision"], "seconds": round(time.time() - ts, 1),
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            # the part is written to Drive in the background while the next shard is scored
            pending.append((writer.submit(write_part, part_path(ws, i), meta, arrays, info),
                            i, path, len(meta), ts))
    downloader.shutdown()
    writer.shutdown()
    left = [k for k in range(len(shards)) if not part_valid(part_path(ws, k))]
    if left:
        log(f"{worker}: nothing left that I may take; {len(left)} shards are being scored by other "
            f"live workers ({left[:10]}{'...' if len(left) > 10 else ''}). Run `merge --wait` anywhere.")
    else:
        log(f"{worker}: all {len(shards)} shards are scored")
    return 0


# ---------------------------------------------------------------- status / merge
def status(args) -> int:
    ws = Path(args.workspace)
    cfg = json.loads((run_dir(ws) / "config.json").read_text())
    now, done, live, stale, free = time.time(), [], [], [], []
    for i in range(cfg["n_shards"]):
        if part_valid(part_path(ws, i)):
            done.append(i)
            continue
        c = read_claim(ws, i)
        if c is None:
            free.append(i)
        elif now - float(c["heartbeat"]) <= args.stale_min * 60:
            live.append((i, c["worker"]))
        else:
            stale.append(i)
    log(f"{len(done)}/{cfg['n_shards']} shards scored | in progress: {live} | stale claims: {stale} | "
        f"unclaimed: {len(free)}")
    return 0


def merge(args) -> int:
    ws = Path(args.workspace)
    rd = run_dir(ws)
    marker = rd / "_STAGE_SUCCESS.json"
    while True:
        cfg = json.loads((rd / "config.json").read_text())
        left = [i for i in range(cfg["n_shards"]) if not part_valid(part_path(ws, i))]
        if not left:
            break
        if not args.wait:
            log(f"{len(left)} shards not scored yet: {left[:20]}")
            return 1
        log(f"waiting for {len(left)} shards ({left[:10]}{'...' if len(left) > 10 else ''}); next check in 2 min")
        time.sleep(120)
    if marker.exists() and json.loads(marker.read_text()).get("revision") == cfg["revision"]:
        log(f"already merged: {marker}")
        return 0
    cols = {f: [] for f in TEXT_FIELDS}
    logits, emb, sslm, label, sr, nlen, shard, metas, lead, kept = [], [], [], [], [], [], [], [], [], []
    for i in range(cfg["n_shards"]):
        with np.load(part_path(ws, i)) as z:
            for f in TEXT_FIELDS:
                cols[f].append(z[f])
            logits.append(z["logits"]); emb.append(z["emb160"]); sslm.append(z["ssl_mean"])
            label.append(z["label"]); sr.append(z["sr"]); nlen.append(z["n_orig"])
            shard.append(np.full(len(z["label"]), i, np.int16))
            n_i = len(z["label"])
            lead.append(z["trim_lead"] if "trim_lead" in z.files else np.full(n_i, -1, np.int64))
            kept.append(z["trim_kept"] if "trim_kept" in z.files else np.full(n_i, -1, np.int64))
            metas.append(json.loads(str(z["meta"])))
    logits = np.concatenate(logits)
    df = pd.DataFrame({f: np.concatenate(v) for f, v in cols.items()})
    df["label"] = np.concatenate(label).astype(int)
    df["logit_spoof"], df["logit_bona"] = logits[:, 0], logits[:, 1]
    df["score"] = logits[:, 1] - logits[:, 0]
    df["score_official"] = logits[:, 1]
    df["orig_sr"], df["orig_len"], df["shard"] = np.concatenate(sr), np.concatenate(nlen), np.concatenate(shard)
    if TRIM_DB:
        df["lead_silence_s"] = np.concatenate(lead) / 16000.0
        df["kept_s"] = np.concatenate(kept) / 16000.0
    if df.utt_id.duplicated().any():
        raise RuntimeError("duplicate utterance ids across parts")
    tmp = rd / "scores.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(rd / "scores.csv")
    for name, arr in [("emb160.npy", np.concatenate(emb)), ("ssl_mean1024.npy", np.concatenate(sslm))]:
        t = rd / f"{name}.tmp.npy"
        np.save(t, arr)
        t.replace(rd / name)
    inp = inputs_dir(ws)
    inp.mkdir(parents=True, exist_ok=True)
    df[["utt_id", "score", "label", "speaker", "codec", "subset"]].rename(columns={"speaker": "group"}) \
        .to_csv(inp / "pilot.csv", index=False)

    from .metrics import eer_and_threshold, hter
    b, s = df.score[df.label == 1].values, df.score[df.label == 0].values
    eer, _ = eer_and_threshold(b, s)
    summary = {"stage": f"{dataset_name()}_merge", "run": RUN, "trim_db": TRIM_DB, "revision": cfg["revision"], "n": int(len(df)),
               "n_bona": int(b.size), "n_spoof": int(s.size), "spoof_prior": float(s.size / len(df)),
               "eer": eer, "precision_by_part": sorted({m["precision"] for m in metas}),
               "gpus": sorted({m["gpu"] for m in metas}), "workers": sorted({m["worker"] for m in metas}),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    g0 = ws / "results" / "G0_reproduction.json"
    if g0.exists():
        tau = json.loads(g0.read_text())["tau_src"]
        summary["at_tau_src"] = hter(b, s, tau)
        summary["reference_zhou_wang_2026_subset"] = {"eer": 0.0792, "hter": 0.1705, "note": "4 of 80 shards"}
    by = {}
    for key in ["subset", "codec", "source"]:
        for v, g in df.groupby(key):
            gb, gs = g.score[g.label == 1].values, g.score[g.label == 0].values
            if gb.size and gs.size:
                by[f"{key}={v}"] = {"n": int(len(g)), "eer": eer_and_threshold(gb, gs)[0]}
    summary["eer_by"] = by
    _atomic_json(marker, summary)
    log(f"merged {len(df)} clips ({b.size} bona fide, {s.size} spoof; spoof share {s.size / len(df):.3f}); "
        f"EER {100 * eer:.2f}% -> {rd}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["work", "status", "merge"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--local", default="LOCAL_CACHE")
    ap.add_argument("--ckpt", default="LOCAL_CACHE/checkpoints/LA_model.pth")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--stale-min", type=float, default=20.0,
                    help="a claim whose heartbeat is older than this may be taken over")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    ap.add_argument("--precision-tol", type=float, default=0.05)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--wait", action="store_true", help="merge: poll until every shard is scored")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--trim-db", type=float, default=0.0,
                    help="0 = confirmatory run; 40 = silence-trimmed run (librosa.effects.trim, top_db 40)")
    a = ap.parse_args(argv)
    configure(a.trim_db)
    if a.cmd == "work" and not (0 <= a.worker_id < a.n_workers):
        raise SystemExit("--worker-id must be in [0, n-workers)")
    return {"work": work, "status": status, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
