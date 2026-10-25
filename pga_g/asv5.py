"""
ASVspoof 5 evaluation set (track 1) scoring across several Colab runtimes at once.

Data: huggingface.co/datasets/jungjee/asvspoof5 (public mirror of Zenodo record 14498691,
ODC-By 1.0): ten evaluation tar files flac_E_aa..aj (~8.5 GB each, ~681k FLAC files at 16 kHz)
and ASVspoof5_protocols.tar. If the Hugging Face download fails, the same files are fetched
from Zenodo. The repository revision is pinned in runs/asv5_v1/config.json by the first worker.

Units and parallelism
  tar level  : worker k of N owns tars t with t % N == k; then it takes over tars that nobody
               claimed or whose claim heartbeat is older than --stale-min.
  chunk level: each tar is scored in chunks of --chunk files (one part file per chunk on Drive),
               so a restarted worker loses at most one chunk. A worker with nothing left to claim
               helps on a tar another worker is still scoring, from its last chunk backwards
               (the owner works forwards; each checks for a finished part before every chunk).
  in runtime : the next tar downloads in a background thread; chunk bytes are read from the tar
               by a reader thread; FLAC decoding runs in DataLoader worker processes; parts are
               written to Drive by a writer thread; precision is the fastest of bf16/fp16 -> TF32
               -> fp32 whose scores stay within --precision-tol logits of strict fp32.

    python -u -m pga_g.asv5 work   --workspace W --worker-id 0 --n-workers 2
    python -u -m pga_g.asv5 status --workspace W
    python -u -m pga_g.asv5 merge  --workspace W [--wait]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import shutil
import socket
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .df21 import Heartbeat as _DF21Heartbeat, _atomic_json, score_shard

REPO = "jungjee/asvspoof5"
ZENODO = "https://zenodo.org/records/14498691/files/{name}?download=1"
RUN = "asv5_v1"
EVAL_TARS = [f"flac_E_a{c}.tar" for c in "abcdefghij"]
PROTO_FILES = ("ASVspoof5_protocols.tar", "ASVspoof5_protocols.tar.gz")
PROTO_NAME = "ASVspoof5.eval.track_1.tsv"
PROTO_COLS = ["speaker", "utt_id", "gender", "codec", "codec_q", "codec_seed", "attack_tag", "attack",
              "key", "tmp"]
CHUNK = 4096


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- remote access (patchable)
def _repo_sha(token=None) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(REPO, token=token).sha


def _download_hf(fname: str, sha: str, local_dir: Path, token=None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, fname, repo_type="dataset", revision=sha,
                                local_dir=str(local_dir), token=token))


def _download_zenodo(fname: str, local_dir: Path) -> Path:
    import requests
    name = fname if fname != "ASVspoof5_protocols.tar" else "ASVspoof5_protocols.tar.gz"
    dst = Path(local_dir) / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with requests.get(ZENODO.format(name=name), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(1 << 24):
                fh.write(block)
    tmp.replace(dst)
    return dst


def fetch(fname: str, sha: str, local_dir: Path, token=None, source: str = "auto") -> Path:
    if source in ("auto", "hf"):
        try:
            return _download_hf(fname, sha, local_dir, token)
        except Exception as e:                       # noqa: BLE001
            if source == "hf":
                raise
            log(f"Hugging Face download of {fname} failed ({type(e).__name__}: {e}); trying Zenodo")
    return _download_zenodo(fname, local_dir)


# ---------------------------------------------------------------- paths and configuration
def run_dir(ws: Path) -> Path:
    return ws / "runs" / RUN


def part_path(ws: Path, t: int, c: int) -> Path:
    return run_dir(ws) / "parts" / f"part_{t:02d}_{c:03d}.npz"


def manifest_path(ws: Path, t: int) -> Path:
    return run_dir(ws) / "manifests" / f"tar_{t:02d}.json"


def claim_path(ws: Path, t: int) -> Path:
    return run_dir(ws) / "claims" / f"tar_{t:02d}.json"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                # noqa: BLE001
        return None


def ensure_config(ws: Path, ckpt_sha: str | None = None, source: str = "auto") -> dict:
    cfg_path = run_dir(ws) / "config.json"
    if not cfg_path.exists():
        try:
            sha = _repo_sha(os.environ.get("HF_TOKEN") or None)
        except Exception as e:                       # noqa: BLE001
            if source == "hf":
                raise
            log(f"cannot reach Hugging Face ({e}); pinning to the Zenodo record instead")
            sha = "zenodo-14498691"
        _atomic_json(cfg_path, {"repo": REPO, "revision": sha, "tars": EVAL_TARS, "chunk": CHUNK,
                                "ckpt_sha256": ckpt_sha, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "created_by": socket.gethostname()})
        time.sleep(2)
    cfg = json.loads(cfg_path.read_text())
    if ckpt_sha and cfg.get("ckpt_sha256") not in (None, ckpt_sha):
        raise SystemExit("this run was started with a different checkpoint")
    return cfg


def part_valid(path: Path, n_expected: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            n = int(json.loads(str(z["meta"]))["rows"])
            ok = (z["logits"].shape == (n, 2) and z["emb160"].shape == (n, 160)
                  and z["ssl_mean"].shape == (n, 1024) and z["utt_id"].shape == (n,)
                  and np.isfinite(z["logits"]).all())
            return bool(ok and (n_expected is None or n == n_expected))
    except Exception:                                # noqa: BLE001
        return False


def chunk_sizes(n: int, chunk: int) -> list[int]:
    return [min(chunk, n - i) for i in range(0, n, chunk)]


def tar_state(ws: Path, t: int):
    """(manifest or None, list of missing chunk indices or None if the manifest is unknown)."""
    man = read_json(manifest_path(ws, t))
    if man is None:
        return None, None
    sizes = chunk_sizes(man["n"], man["chunk"])
    missing = [c for c, k in enumerate(sizes) if not part_valid(part_path(ws, t, c), k)]
    return man, missing


# ---------------------------------------------------------------- protocol
def load_protocol(ws: Path, local: Path, sha: str, token, source: str) -> pd.DataFrame:
    dst = run_dir(ws) / "protocol_eval_track1.tsv"
    if not dst.exists():
        proto = None
        for name in PROTO_FILES:
            try:
                proto = fetch(name, sha, local, token, source) if name.endswith(".tar") else _download_zenodo(name, local)
                break
            except Exception as e:                   # noqa: BLE001
                log(f"protocol file {name}: {e}")
        if proto is None:
            raise SystemExit("could not download the ASVspoof 5 protocol archive")
        with tarfile.open(proto, "r:*") as tf:
            m = [x for x in tf.getmembers() if x.isfile() and Path(x.name).name == PROTO_NAME]
            if not m:
                raise SystemExit(f"{PROTO_NAME} not found in {proto.name}")
            data = tf.extractfile(m[0]).read()
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(dst)
    return parse_protocol(dst)


def parse_protocol(path: Path) -> pd.DataFrame:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    if rows and rows[0][:2] == ["SPEAKER_ID", "FLAC_FILE_NAME"]:            # optional header line
        rows = rows[1:]
    rows = [r + ["-"] * (len(PROTO_COLS) - len(r)) if len(r) < len(PROTO_COLS) else r[: len(PROTO_COLS)] for r in rows]
    df = pd.DataFrame(rows, columns=PROTO_COLS)
    bad = ~df.key.isin(["bonafide", "spoof"])
    if bad.any():
        raise SystemExit(f"{bad.sum()} evaluation rows have no bonafide/spoof key (e.g. {df.key[bad].iloc[0]!r}); "
                         "this release does not contain evaluation keys")
    if df.utt_id.duplicated().any():
        raise SystemExit("duplicate file names in the evaluation protocol")
    return df


# ---------------------------------------------------------------- precision ladder
def make_forward(model, device, precision: str):
    import torch
    from .score import make_forward as _mf
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return _mf(model, device, "fp32" if tf32 else precision)


def decide_precision(model, device, wavs: np.ndarray, requested: str, tol: float) -> dict:
    """Fastest precision whose genuine-minus-spoof logit stays within tol of strict fp32."""
    import torch
    if device.type != "cuda":
        return {"precision": "fp32", "reason": "cpu"}
    if requested != "auto":
        return {"precision": requested, "reason": "requested"}

    def scores(prec):
        f = make_forward(model, device, prec)
        out = []
        for a in range(0, len(wavs), 16):
            lg = f(torch.from_numpy(wavs[a:a + 16]).to(device))[0]
            out.append((lg[:, 1] - lg[:, 0]).cpu())
        return torch.cat(out)
    ref = scores("fp32")
    tried = {}
    for cand in (["bf16"] if torch.cuda.is_bf16_supported() else ["fp16"]) + ["tf32"]:
        d = float((scores(cand) - ref).abs().max())
        tried[cand] = d
        if d <= tol:
            make_forward(model, device, cand)
            return {"precision": cand, "max_abs_score_diff": tried, "tol": tol, "n_check": len(wavs),
                    "reason": f"{cand} within tolerance"}
    make_forward(model, device, "fp32")
    return {"precision": "fp32", "max_abs_score_diff": tried, "tol": tol, "n_check": len(wavs),
            "reason": "no faster precision within tolerance; strict fp32"}


# ---------------------------------------------------------------- tar handling
def open_tar(path: Path):
    tf = tarfile.open(path, "r:")
    members = sorted((m for m in tf.getmembers() if m.isfile() and m.name.endswith(".flac")
                      and not Path(m.name).name.startswith("._")),        # skip macOS metadata entries
                     key=lambda m: m.name)
    return tf, members


def ensure_manifest(ws: Path, t: int, members, chunk: int) -> dict:
    man = read_json(manifest_path(ws, t))
    names = [Path(m.name).stem for m in members]
    if man is None:
        man = {"tar": EVAL_TARS[t], "n": len(names), "chunk": chunk, "names": names}
        _atomic_json(manifest_path(ws, t), man)
    elif man["names"] != names:
        raise RuntimeError(f"{EVAL_TARS[t]}: file list differs from the manifest on Drive")
    return man


class ClaimBeat(_DF21Heartbeat):
    """Heartbeat on a tar-level claim file."""

    def __init__(self, ws: Path, t: int, worker: str, every: float = 60.0):
        super().__init__(ws, t, worker, every)

    def _write(self):
        _atomic_json(claim_path(self.ws, self.i), {"worker": self.worker, "started": self.started,
                                                  "heartbeat": time.time()})


def write_part(path: Path, names, arrays, info: dict) -> None:
    logits, emb, sslm, sr_a, len_a, _, _ = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, logits=logits, emb160=emb, ssl_mean=sslm, sr=sr_a, n_orig=len_a,
             utt_id=np.asarray(names, dtype=np.str_), meta=np.array(json.dumps({**info, "rows": len(names)})))
    tmp.replace(path)


# ---------------------------------------------------------------- worker
def work(args) -> int:
    import torch
    from .score import load_model, sha256_file

    ws = Path(args.workspace)
    from .confirm5 import prereg
    prereg(ws)                                   # the hypotheses go to Drive before the first score
    local = Path(args.local) / "asv5"
    local.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    token = os.environ.get("HF_TOKEN") or None
    cfg = ensure_config(ws, sha256_file(ckpt), args.source)
    sha, chunk = cfg["revision"], int(cfg["chunk"])
    source = "zenodo" if sha.startswith("zenodo") else args.source
    proto = load_protocol(ws, local, sha, token, source)
    worker = f"w{args.worker_id}@{socket.gethostname()}"
    n_t = len(EVAL_TARS)
    log(f"{worker}: {len(proto)} evaluation trials in {n_t} tars ({REPO} @ {sha[:10]}); I own tars with "
        f"index % {args.n_workers} == {args.worker_id}; chunks of {chunk} files")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    model, _ = load_model(ckpt, device, args.attn)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    workers = args.workers if args.workers >= 0 else min(16, os.cpu_count() or 2)
    prec_file = run_dir(ws) / "precision" / f"{socket.gethostname()}_w{args.worker_id}.json"
    prec = read_json(prec_file)
    fwd = None
    state = {"micro": args.batch_size}
    downloader = ThreadPoolExecutor(max_workers=1)
    writer = ThreadPoolExecutor(max_workers=2)
    prefetch: dict = {}
    t0, done_clips, done_chunks = time.time(), 0, 0
    pending_writes: list = []

    def get_tar(t):
        if t in prefetch:
            return prefetch.pop(t).result()
        return fetch(EVAL_TARS[t], sha, local, token, source)

    def stale(t, now):
        c = read_json(claim_path(ws, t))
        return c is None or now - float(c.get("heartbeat", 0)) > args.stale_min * 60

    def tar_done(t):
        man, missing = tar_state(ws, t)
        return man is not None and not missing

    def plan():
        """Next unit: (tar, 'own'|'takeover'|'help', prefetch candidate)."""
        todo = [t for t in range(n_t) if not tar_done(t)]
        now = time.time()
        mine = f"w{args.worker_id}@"                 # a restarted tab resumes its own tars at once
        own = [t for t in todo if t % args.n_workers == args.worker_id and (stale(t, now) or
               str((read_json(claim_path(ws, t)) or {}).get("worker", "")).startswith(mine))]
        if own:
            return own[0], "own", (own[1:2] or [None])[0]
        free = sorted((t for t in todo if stale(t, now)), reverse=True)
        if free:
            return free[0], "takeover", (free[1:2] or [None])[0]
        best, best_left = None, 0
        for t in todo:                               # help where most chunks remain
            man, missing = tar_state(ws, t)
            left = len(missing) if missing is not None else 0
            if man is not None and left > max(best_left, args.help_min):
                best, best_left = t, left
        return (best, "help", None) if best is not None else (None, None, None)

    def flush_writes(block):
        nonlocal done_clips, done_chunks
        for item in list(pending_writes):
            fut, t, c, n = item
            if block or fut.done():
                fut.result()
                pending_writes.remove(item)
                done_clips += n
                done_chunks += 1

    def score_chunk(blobs, members, t, c, man):
        nonlocal fwd, prec
        mem = members[c * man["chunk"]: min((c + 1) * man["chunk"], man["n"])]
        if fwd is None:
            if prec is None:
                from .data import load_audio_bytes, official_pad
                probe = np.stack([official_pad(load_audio_bytes(b)[0]) for b in blobs[:128]]).astype(np.float32)
                prec = {**decide_precision(model, device, probe, args.precision, args.precision_tol), "gpu": gpu}
                _atomic_json(prec_file, prec)
            log(f"{worker}: precision {prec['precision']} ({prec.get('reason')}; differences {prec.get('max_abs_score_diff')}) on {gpu}")
            fwd = make_forward(model, device, prec["precision"])
        ts = time.time()
        arrays = score_shard(blobs, fwd, device, args.batch_size, workers, state)
        names = [Path(m.name).stem for m in mem]
        info = {"tar": EVAL_TARS[t], "chunk": c, "revision": sha, "worker": worker, "gpu": gpu,
                "precision": prec["precision"], "seconds": round(time.time() - ts, 1),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        pending_writes.append((writer.submit(write_part, part_path(ws, t, c), names, arrays, info), t, c, len(names)))
        flush_writes(block=False)
        rate = (done_clips + len(names)) / max(time.time() - t0, 1e-9)
        log(f"{worker}: {EVAL_TARS[t]} chunk {c + 1}/{len(chunk_sizes(man['n'], man['chunk']))} "
            f"({len(names)} files, {time.time() - ts:.0f}s) | {rate:.0f} clips/s here")

    while True:
        flush_writes(block=False)
        t, kind, nxt = plan()
        if t is None:
            break
        if nxt is not None and nxt not in prefetch:
            prefetch[nxt] = downloader.submit(fetch, EVAL_TARS[nxt], sha, local, token, source)
        path = get_tar(t)
        beat = ClaimBeat(ws, t, worker) if kind in ("own", "takeover") else None
        if beat:
            beat.__enter__()
        try:
            tf, members = open_tar(path)
            man = ensure_manifest(ws, t, members, chunk)
            sizes = chunk_sizes(man["n"], man["chunk"])
            order = iter(range(len(sizes)) if kind != "help" else reversed(range(len(sizes))))
            log(f"{worker}: {kind} {EVAL_TARS[t]} ({man['n']} files, {len(sizes)} chunks)")

            started = [False]

            def next_needed():
                for c in order:
                    if part_valid(part_path(ws, t, c), sizes[c]) or any(p[1] == t and p[2] == c for p in pending_writes):
                        if kind == "help" and started[0]:
                            return None              # met the owner coming forwards
                        continue                     # finished earlier (e.g. an earlier help round)
                    started[0] = True
                    return c
                return None

            def read(c):                             # only this thread touches the tar file
                return [tf.extractfile(m).read() for m in members[c * man["chunk"]: min((c + 1) * man["chunk"], man["n"])]]

            reader = ThreadPoolExecutor(max_workers=1)
            n_before = done_chunks + len(pending_writes)
            c = next_needed()
            fut = reader.submit(read, c) if c is not None else None
            while c is not None:
                blobs = fut.result()
                c_next = next_needed()               # read the next chunk while this one is scored
                fut = reader.submit(read, c_next) if c_next is not None else None
                score_chunk(blobs, members, t, c, man)
                del blobs
                c = c_next
            reader.shutdown()
            tf.close()
            idle = done_chunks + len(pending_writes) == n_before
        finally:
            if beat:
                beat.__exit__()
        flush_writes(block=True)
        if kind == "help" and idle:                  # nothing left to take here right now: do not spin
            time.sleep(60)
        if tar_done(t):
            try:
                path.unlink()
            except OSError:
                pass
    flush_writes(block=True)
    downloader.shutdown()
    writer.shutdown()
    for f in local.glob("flac_E_*.tar"):              # prefetched tars that were not needed
        try:
            f.unlink()
        except OSError:
            pass
    left = [EVAL_TARS[t] for t in range(n_t) if not tar_done(t)]
    log(f"{worker}: finished; {done_chunks} chunks / {done_clips} files scored here"
        + (f"; still being scored elsewhere: {left}" if left else "; all tars are scored"))
    return 0


# ---------------------------------------------------------------- status / merge
def status(args) -> int:
    ws = Path(args.workspace)
    now, lines, total, done = time.time(), [], 0, 0
    for t, name in enumerate(EVAL_TARS):
        man, missing = tar_state(ws, t)
        c = read_json(claim_path(ws, t))
        who = (f"{c['worker']} ({(now - float(c['heartbeat'])) / 60:.0f} min ago)" if c else "unclaimed")
        if man is None:
            lines.append(f"  {name}: not started | {who}")
            continue
        n_c = len(chunk_sizes(man["n"], man["chunk"]))
        total += n_c
        done += n_c - len(missing)
        lines.append(f"  {name}: {n_c - len(missing)}/{n_c} chunks | {who}")
    log(f"ASVspoof 5 eval scoring: {done}/{total or '?'} chunks of the started tars\n" + "\n".join(lines))
    return 0


def merge(args) -> int:
    ws = Path(args.workspace)
    rd = run_dir(ws)
    marker = rd / "_STAGE_SUCCESS.json"
    while True:
        states = [tar_state(ws, t) for t in range(len(EVAL_TARS))]
        left = [EVAL_TARS[t] for t, (m, miss) in enumerate(states) if m is None or miss]
        if not left:
            break
        if not args.wait:
            log(f"not complete: {left}")
            return 1
        log(f"waiting for {len(left)} tars ({left[:4]}{'...' if len(left) > 4 else ''}); next check in 2 min")
        time.sleep(120)
    cfg = json.loads((rd / "config.json").read_text())
    if marker.exists() and json.loads(marker.read_text()).get("revision") == cfg["revision"]:
        log(f"already merged: {marker}")
        return 0
    proto = parse_protocol(rd / "protocol_eval_track1.tsv")
    logits, emb, sslm, utt, sr, nlen, metas = [], [], [], [], [], [], []
    for t, (man, _) in enumerate(states):
        for c in range(len(chunk_sizes(man["n"], man["chunk"]))):
            with np.load(part_path(ws, t, c)) as z:
                logits.append(z["logits"]); emb.append(z["emb160"]); sslm.append(z["ssl_mean"])
                utt.append(z["utt_id"]); sr.append(z["sr"]); nlen.append(z["n_orig"])
                metas.append(json.loads(str(z["meta"])))
    utt = np.concatenate(utt)
    if len(set(utt)) != utt.size:
        raise RuntimeError("a file was scored twice across parts")
    pos = pd.Series(np.arange(utt.size), index=utt)
    missing = set(proto.utt_id) - set(utt)
    if missing:
        raise RuntimeError(f"{len(missing)} protocol files were not found in the tars (e.g. {sorted(missing)[:3]})")
    idx = pos.loc[proto.utt_id].to_numpy()           # protocol order
    logits = np.concatenate(logits)[idx]
    df = proto.drop(columns=["tmp"]).copy()
    df["label"] = (df.key == "bonafide").astype(int)
    df["src_group"] = np.where(df.codec_seed != "-", df.codec_seed, df.utt_id)
    df["logit_spoof"], df["logit_bona"] = logits[:, 0], logits[:, 1]
    df["score"] = logits[:, 1] - logits[:, 0]
    df["orig_sr"], df["orig_len"] = np.concatenate(sr)[idx], np.concatenate(nlen)[idx]
    extra = len(utt) - len(proto)
    tmp = rd / "scores.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(rd / "scores.csv")
    for name, arr in [("emb160.npy", np.concatenate(emb)[idx]), ("ssl_mean1024.npy", np.concatenate(sslm)[idx])]:
        t_ = rd / f"{name}.tmp.npy"
        np.save(t_, arr)
        t_.replace(rd / name)
    inp = ws / "results" / "asv5_inputs"
    inp.mkdir(parents=True, exist_ok=True)
    df[["utt_id", "score", "label", "speaker", "codec", "attack", "src_group"]].rename(columns={"speaker": "group"}) \
        .to_csv(inp / "pilot.csv", index=False)

    from .metrics import eer_and_threshold, hter
    b, s = df.score[df.label == 1].values, df.score[df.label == 0].values
    eer, _ = eer_and_threshold(b, s)
    summary = {"stage": "asv5_merge", "revision": cfg["revision"], "n": int(len(df)), "n_bona": int(b.size),
               "n_spoof": int(s.size), "spoof_share": float(s.size / len(df)), "eer": eer,
               "files_in_tars_not_in_protocol": int(extra),
               "n_speakers_genuine": int(df.speaker[df.label == 1].nunique()),
               "n_source_groups": int(df.src_group.nunique()),
               "precision_by_part": sorted({m["precision"] for m in metas}),
               "gpus": sorted({m["gpu"] for m in metas}), "workers": sorted({m["worker"] for m in metas}),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    g0 = read_json(ws / "results" / "G0_reproduction.json")
    if g0:
        summary["at_tau_src"] = hter(b, s, g0["tau_src"])
    by = {}
    for key in ["codec", "attack"]:
        for v, g in df.groupby(key):
            gb = g.score[g.label == 1].values if key == "codec" else b
            gs = g.score[g.label == 0].values
            if gb.size and gs.size:
                by[f"{key}={v}"] = {"n": int(len(g)), "eer": eer_and_threshold(gb, gs)[0]}
    summary["eer_by"] = by
    _atomic_json(marker, summary)
    log(f"merged {len(df)} trials ({b.size} genuine, {s.size} spoof; spoof share {s.size / len(df):.3f}); "
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
    ap.add_argument("--stale-min", type=float, default=20.0)
    ap.add_argument("--help-min", type=int, default=3, help="help on another tar only if more chunks remain")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--precision", choices=["auto", "fp32", "tf32", "bf16", "fp16"], default="auto")
    ap.add_argument("--precision-tol", type=float, default=0.05)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--source", choices=["auto", "hf", "zenodo"], default="auto")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "work" and not (0 <= a.worker_id < a.n_workers):
        raise SystemExit("--worker-id must be in [0, n-workers)")
    return {"work": work, "status": status, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
