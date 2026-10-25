"""
Generalisation test (v3): score ASVspoof 2019 LA eval, ASVspoof 2021 DF eval and ASVspoof 5 eval
with two more frozen detectors (AASIST and XLSR-Mamba, pga_g.detectors) in ONE pass over the audio:
every batch is decoded once and scored by both detectors.

Units of work (99): 9 ASVspoof 2019 LA parquet shards, 80 ASVspoof 2021 DF parquet shards and
10 ASVspoof 5 tars, at the revisions already pinned by the earlier runs (runs/df21_v1,
runs/asv5_v1; the LA mirror revision is pinned here by the first worker). Each unit is scored in
chunks of 4,096 files, one part file per chunk on Drive.

Parallelism (same scheme as pga_g.asv5)
  across runtimes : worker k of N owns units u with u % N == k, then takes over units whose claim
                    heartbeat is older than --stale-min, then helps on another worker's unit from
                    its last chunk backwards (finished chunks are skipped, so a restart or a lost
                    chunk costs at most one chunk).
  inside a runtime: the next unit downloads in a background thread, chunk bytes are read by a
                    reader thread, FLAC decoding runs in DataLoader worker processes, both detectors
                    run on each GPU batch, parts are written to Drive by a writer thread; precision is
                    the fastest of bf16/fp16 -> TF32 -> fp32 within --precision-tol of strict fp32,
                    decided per detector.

    python -u -m pga_g.multi work   --workspace W --worker-id 0 --n-workers 2
    python -u -m pga_g.multi status --workspace W
    python -u -m pga_g.multi merge  --workspace W
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import asv5 as A5
from . import df21 as D21
from . import detectors as DT
from .df21 import Heartbeat as _Heartbeat, _atomic_json

RUN = "v3"
CORPORA = ("la19", "df21", "asv5")
DETS = ("aasist", "mamba")
CHUNK = 4096
LA19_REPO = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
LA19_N_SHARDS = 9
AASIST_PUBLISHED_LA19_EER = 0.0083            # github.com/clovaai/aasist README


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                # noqa: BLE001
        return None


# ---------------------------------------------------------------- remote access (patchable)
def _list_la19(token=None):
    from huggingface_hub import HfApi
    api = HfApi()
    sha = api.dataset_info(LA19_REPO, token=token).sha
    files = sorted(f for f in api.list_repo_files(LA19_REPO, repo_type="dataset", revision=sha, token=token)
                   if f.startswith("data/test-") and f.endswith(".parquet"))
    return sha, files


def _download_parquet(repo: str, fname: str, sha: str, local_dir: Path, token=None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, fname, repo_type="dataset", revision=sha, local_dir=str(local_dir),
                                token=token))


def _mamba_revision(token=None) -> str:
    return DT.mamba_revision(token)


def _fetch_mamba(local_dir: Path, revision: str, token=None) -> Path:
    return DT.fetch_mamba(local_dir, revision, token)


def _published_mamba_df21() -> pd.DataFrame:
    import urllib.request
    with urllib.request.urlopen(DT.MAMBA_PUBLISHED_DF21, timeout=120) as r:
        text = r.read().decode()
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    return pd.DataFrame({"utt_id": [r[0] for r in rows], "published": [float(r[1]) for r in rows]})


# ---------------------------------------------------------------- paths and configuration
def run_dir(ws: Path) -> Path:
    return Path(ws) / "runs" / RUN


def part_path(ws, corpus, i, c) -> Path:
    return run_dir(ws) / "parts" / corpus / f"{i:03d}_{c:03d}.npz"


def manifest_path(ws, corpus, i) -> Path:
    return run_dir(ws) / "manifests" / f"{corpus}_{i:03d}.json"


def claim_path(ws, corpus, i) -> Path:
    return run_dir(ws) / "claims" / f"{corpus}_{i:03d}.json"


def ensure_config(ws: Path, token=None) -> dict:
    """First worker pins every input; later workers reuse the file."""
    ws = Path(ws)
    path = run_dir(ws) / "config.json"
    if not path.exists():
        d21 = read_json(ws / "runs" / "df21_v1" / "config.json")
        a5 = read_json(ws / "runs" / "asv5_v1" / "config.json")
        if d21 is None or a5 is None:
            raise SystemExit("runs/df21_v1/config.json and runs/asv5_v1/config.json must exist (earlier runs)")
        la_sha, la_files = _list_la19(token)
        if len(la_files) != LA19_N_SHARDS:
            raise SystemExit(f"expected {LA19_N_SHARDS} ASVspoof 2019 LA eval shards, found {len(la_files)}")
        units = ([{"corpus": "la19", "repo": LA19_REPO, "revision": la_sha, "file": f} for f in la_files]
                 + [{"corpus": "df21", "repo": d21["repo"], "revision": d21["revision"], "file": f}
                    for f in d21["shards"]]
                 + [{"corpus": "asv5", "repo": A5.REPO, "revision": a5["revision"], "file": f} for f in A5.EVAL_TARS])
        _atomic_json(path, {"units": units, "chunk": CHUNK, "detectors": list(DETS),
                            "mamba_repo": DT.MAMBA_REPO, "mamba_revision": _mamba_revision(token),
                            "aasist_sha256": DT.AASIST_SHA256, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "created_by": socket.gethostname()})
        time.sleep(2)
    return json.loads(path.read_text())


def unit_index(cfg) -> list[tuple[str, int, dict]]:
    """[(corpus, index within corpus, unit)] in global order."""
    out, count = [], {}
    for u in cfg["units"]:
        k = count.get(u["corpus"], 0)
        out.append((u["corpus"], k, u))
        count[u["corpus"]] = k + 1
    return out


def chunk_sizes(n, chunk):
    return [min(chunk, n - i) for i in range(0, n, chunk)]


def part_valid(path: Path, n_expected=None) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            n = int(json.loads(str(z["meta"]))["rows"])
            ok = z["utt_id"].shape == (n,)
            for d in DETS:
                ok &= z[f"{d}__logits"].shape == (n, 2) and bool(np.isfinite(z[f"{d}__logits"]).all())
                for f, (dim, _) in DT.FEATURES[d].items():
                    ok &= z[f"{d}__{f}"].shape == (n, dim)
            return bool(ok and (n_expected is None or n == n_expected))
    except Exception:                                # noqa: BLE001
        return False


def unit_state(ws, corpus, i):
    man = read_json(manifest_path(ws, corpus, i))
    if man is None:
        return None, None
    sizes = chunk_sizes(man["n"], man["chunk"])
    return man, [c for c, k in enumerate(sizes) if not part_valid(part_path(ws, corpus, i, c), k)]


class ClaimBeat(_Heartbeat):
    def __init__(self, ws, corpus, i, worker, every=60.0):
        super().__init__(ws, i, worker, every)
        self.corpus = corpus

    def _write(self):
        _atomic_json(claim_path(self.ws, self.corpus, self.i),
                     {"worker": self.worker, "started": self.started, "heartbeat": time.time()})


# ---------------------------------------------------------------- unit reading
class UnitReader:
    """Gives the sorted file names of a unit and the encoded bytes of any chunk."""

    def __init__(self, path: Path, kind: str):
        self.kind = kind
        if kind == "parquet":
            blobs, meta = D21.read_shard(path)
            self.names = meta.utt_id.astype(str).tolist()
            self.labels = meta.label.to_numpy(np.int8)
            self.meta = meta
            self._blobs = blobs
        else:
            self.tf, self.members = A5.open_tar(path)
            self.names = [Path(m.name).stem for m in self.members]
            self.labels = np.full(len(self.names), -1, np.int8)

    def read(self, a, b):
        if self.kind == "parquet":
            return self._blobs[a:b]
        return [self.tf.extractfile(m).read() for m in self.members[a:b]]

    def close(self):
        if self.kind != "parquet":
            self.tf.close()
        self._blobs = None


def ensure_manifest(ws, corpus, i, names, chunk):
    man = read_json(manifest_path(ws, corpus, i))
    if man is None:
        man = {"n": len(names), "chunk": chunk, "names": names}
        _atomic_json(manifest_path(ws, corpus, i), man)
    elif man["names"] != names:
        raise RuntimeError(f"{corpus} unit {i}: file list differs from the manifest on Drive")
    return man


# ---------------------------------------------------------------- scoring
def score_blobs(blobs, fwds: dict, device, batch_size: int, workers: int, state: dict):
    """Decode once, run every detector; returns ({det: (logits, {feature: array})},
    {"sr": original sample rates, "n_orig": original lengths})."""
    import torch
    from .data import BytesAudioDataset, collate
    cut = max(DT.CUTS[d] for d in fwds)              # decode once at the longest crop; each detector takes its prefix
    loader = torch.utils.data.DataLoader(
        BytesAudioDataset(blobs, cut=cut), batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, pin_memory=device.type == "cuda", prefetch_factor=4 if workers else None)
    n = len(blobs)
    out = {d: (np.zeros((n, 2), np.float32),
               {f: np.zeros((n, dim), dt) for f, (dim, dt) in DT.FEATURES[d].items()}) for d in fwds}
    audio = {"sr": np.zeros(n, np.int32), "n_orig": np.zeros(n, np.int64)}
    for wav, idx, sr, ln in loader:
        wav = wav.to(device, non_blocking=True)
        ii = idx.numpy()
        audio["sr"][ii], audio["n_orig"][ii] = sr.numpy(), ln.numpy()
        for d, fwd in fwds.items():
            parts, k = [], 0
            while k < wav.shape[0]:
                try:
                    parts.append(fwd(wav[k:k + state["micro"]]))
                    k += state["micro"]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if state["micro"] == 1:
                        raise
                    state["micro"] = max(1, state["micro"] // 2)
                    log(f"CUDA OOM -> micro-batch {state['micro']}")
            lg, feats = out[d]
            lg[ii] = torch.cat([p[0] for p in parts]).cpu().numpy()
            for f in feats:
                feats[f][ii] = torch.cat([p[1][f] for p in parts]).cpu().numpy().astype(feats[f].dtype)
    return out, audio


def write_part(path: Path, names, labels, out: dict, info: dict, audio: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"utt_id": np.asarray(names, dtype=np.str_), "label": np.asarray(labels, np.int8),
              "meta": np.array(json.dumps({**info, "rows": len(names)})), **(audio or {})}
    for d, (lg, feats) in out.items():
        arrays[f"{d}__logits"] = lg
        for f, v in feats.items():
            arrays[f"{d}__{f}"] = v
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def work(args) -> int:
    import torch
    from .general import prereg
    ws = Path(args.workspace)
    prereg(ws)                                   # hypotheses on Drive before the first v3 score
    record_deviations(ws)
    token = os.environ.get("HF_TOKEN") or None
    cfg = ensure_config(ws, token)
    local = Path(args.local) / "v3"
    local.mkdir(parents=True, exist_ok=True)
    units = unit_index(cfg)
    worker = f"w{args.worker_id}@{socket.gethostname()}"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    dets = {"aasist": DT.build_aasist(device)}
    if "mamba" in DETS:
        ck = _fetch_mamba(Path(args.local) / "checkpoints", cfg["mamba_revision"], token)
        dets["mamba"] = DT.build_mamba(device, ck, args.attn)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    workers = args.workers if args.workers >= 0 else min(16, os.cpu_count() or 2)
    prec_file = run_dir(ws) / "precision" / f"{socket.gethostname()}_w{args.worker_id}.json"
    prec = read_json(prec_file)
    fwds = None
    state = {"micro": args.batch_size}
    _atomic_json(run_dir(ws) / "detectors" / f"{socket.gethostname()}_w{args.worker_id}.json",
                 {d: x.info for d, x in dets.items()})
    log(f"{worker}: {len(units)} units ({', '.join(f'{c}: {sum(u[0] == c for u in units)}' for c in CORPORA)}); "
        f"I own units with index % {args.n_workers} == {args.worker_id}; detectors {list(dets)} on {gpu}")
    downloader, writer = ThreadPoolExecutor(max_workers=1), ThreadPoolExecutor(max_workers=2)
    prefetch, pending = {}, []
    t0, done_files, done_chunks = time.time(), 0, 0

    def fetch(g):
        corpus, i, u = units[g]
        if corpus == "asv5":
            return A5.fetch(u["file"], u["revision"], local, token, "hf")
        return _download_parquet(u["repo"], u["file"], u["revision"], local, token)

    def get(g):
        return prefetch.pop(g).result() if g in prefetch else fetch(g)

    def stale(g, now):
        corpus, i, _ = units[g]
        c = read_json(claim_path(ws, corpus, i))
        return c is None or now - float(c.get("heartbeat", 0)) > args.stale_min * 60

    def done(g):
        corpus, i, _ = units[g]
        man, miss = unit_state(ws, corpus, i)
        return man is not None and not miss

    def plan():
        todo = [g for g in range(len(units)) if not done(g)]
        now, mine = time.time(), f"w{args.worker_id}@"
        own = [g for g in todo if g % args.n_workers == args.worker_id and (
            stale(g, now) or str((read_json(claim_path(ws, units[g][0], units[g][1])) or {}).get("worker", "")).startswith(mine))]
        if own:
            return own[0], "own", (own[1:2] or [None])[0]
        free = sorted((g for g in todo if stale(g, now)), reverse=True)
        if free:
            return free[0], "takeover", (free[1:2] or [None])[0]
        best, left_best = None, 0
        for g in todo:
            man, miss = unit_state(ws, units[g][0], units[g][1])
            if man is not None and len(miss) > max(left_best, args.help_min):
                best, left_best = g, len(miss)
        return (best, "help", None) if best is not None else (None, None, None)

    def flush(block):
        nonlocal done_files, done_chunks
        for item in list(pending):
            fut, g, c, n = item
            if block or fut.done():
                fut.result()
                pending.remove(item)
                done_files += n
                done_chunks += 1

    def ensure_fwds(blobs):
        nonlocal fwds, prec
        if fwds is not None:
            return
        if prec is None:
            from .data import load_audio_bytes, official_pad
            cut = max(DT.CUTS[d] for d in dets)
            probe = np.stack([official_pad(load_audio_bytes(b)[0], cut) for b in blobs[:128]]).astype(np.float32)
            prec = {d: {**DT.decide_precision(x, device, probe, args.precision, args.precision_tol), "gpu": gpu}
                    for d, x in dets.items()}
            _atomic_json(prec_file, prec)
        for d in dets:
            log(f"{worker}: {d} precision {prec[d]['precision']} ({prec[d].get('reason')}; "
                f"differences {prec[d].get('max_abs_score_diff')})")
        fwds = {d: DT.make_forward(x, device, prec[d]["precision"]) for d, x in dets.items()}

    while True:
        flush(False)
        g, kind, nxt = plan()
        if g is None:
            break
        if nxt is not None and nxt not in prefetch:
            prefetch[nxt] = downloader.submit(fetch, nxt)
        corpus, i, u = units[g]
        path = get(g)
        beat = ClaimBeat(ws, corpus, i, worker) if kind in ("own", "takeover") else None
        if beat:
            beat.__enter__()
        try:
            rd = UnitReader(path, "tar" if corpus == "asv5" else "parquet")
            man = ensure_manifest(ws, corpus, i, rd.names, cfg["chunk"])
            sizes = chunk_sizes(man["n"], man["chunk"])
            order = iter(range(len(sizes)) if kind != "help" else reversed(range(len(sizes))))
            started = [False]
            log(f"{worker}: {kind} {corpus} unit {i} ({man['n']} files, {len(sizes)} chunks)")

            def next_needed():
                for c in order:
                    if part_valid(part_path(ws, corpus, i, c), sizes[c]) or any(
                            p[1] == g and p[2] == c for p in pending):
                        if kind == "help" and started[0]:
                            return None              # met the owner
                        continue
                    started[0] = True
                    return c
                return None

            reader = ThreadPoolExecutor(max_workers=1)
            n_before = done_chunks + len(pending)
            c = next_needed()
            fut = reader.submit(rd.read, c * man["chunk"], c * man["chunk"] + sizes[c]) if c is not None else None
            while c is not None:
                blobs = fut.result()
                c_next = next_needed()
                fut = (reader.submit(rd.read, c_next * man["chunk"], c_next * man["chunk"] + sizes[c_next])
                       if c_next is not None else None)
                ensure_fwds(blobs)
                ts = time.time()
                out, audio = score_blobs(blobs, fwds, device, args.batch_size, workers, state)
                a = c * man["chunk"]
                info = {"corpus": corpus, "unit": i, "chunk": c, "file": u["file"], "revision": u["revision"],
                        "worker": worker, "gpu": gpu, "precision": {d: prec[d]["precision"] for d in dets},
                        "seconds": round(time.time() - ts, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                pending.append((writer.submit(write_part, part_path(ws, corpus, i, c), rd.names[a:a + sizes[c]],
                                              rd.labels[a:a + sizes[c]], out, info, audio), g, c, sizes[c]))
                flush(False)
                rate = (done_files + sizes[c]) / max(time.time() - t0, 1e-9)
                log(f"{worker}: {corpus} unit {i} chunk {c + 1}/{len(sizes)} ({sizes[c]} files, "
                    f"{time.time() - ts:.0f}s) | {rate:.0f} files/s here")
                del blobs
                c = c_next
            reader.shutdown()
            rd.close()
            idle = done_chunks + len(pending) == n_before
        finally:
            if beat:
                beat.__exit__()
        flush(True)
        if done(g):
            try:
                Path(path).unlink()
            except OSError:
                pass
        if kind == "help" and idle:
            time.sleep(60)
    flush(True)
    downloader.shutdown()
    writer.shutdown()
    for f in list(local.rglob("*.tar")) + list(local.rglob("*.parquet")):
        try:
            f.unlink()
        except OSError:
            pass
    left = [f"{units[g][0]}:{units[g][1]}" for g in range(len(units)) if not done(g)]
    log(f"{worker}: finished; {done_chunks} chunks / {done_files} files scored here"
        + (f"; still being scored elsewhere: {left[:6]}" if left else "; all units are scored"))
    return 0


# ---------------------------------------------------------------- status / merge
def status(args) -> int:
    ws = Path(args.workspace)
    cfg = read_json(run_dir(ws) / "config.json")
    if cfg is None:
        log("v3 scoring has not started")
        return 0
    for corpus in CORPORA:
        tot = dn = started = 0
        n_units = sum(u["corpus"] == corpus for u in cfg["units"])
        for i in range(n_units):
            man, miss = unit_state(ws, corpus, i)
            if man is None:
                continue
            started += 1
            k = len(chunk_sizes(man["n"], man["chunk"]))
            tot += k
            dn += k - len(miss)
        log(f"{corpus}: {started}/{n_units} units started, {dn}/{tot or '?'} chunks of the started units done")
    return 0


def _eer_tau(b, s):
    from .metrics import eer_and_threshold
    return eer_and_threshold(b, s)


def merge_corpus(ws: Path, corpus: str) -> dict | None:
    rd = run_dir(ws)
    marker = rd / corpus / "_STAGE_SUCCESS.json"
    if marker.exists():
        return read_json(marker)
    cfg = read_json(rd / "config.json")
    n_units = sum(u["corpus"] == corpus for u in cfg["units"])
    states = [unit_state(ws, corpus, i) for i in range(n_units)]
    left = [i for i, (m, miss) in enumerate(states) if m is None or miss]
    if left:
        log(f"{corpus}: {len(left)} units not complete ({left[:6]})")
        return None
    names, labels, data = [], [], {d: {"logits": []} | {f: [] for f in DT.FEATURES[d]} for d in DETS}
    for i, (man, _) in enumerate(states):
        for c, k in enumerate(chunk_sizes(man["n"], man["chunk"])):
            with np.load(part_path(ws, corpus, i, c)) as z:
                names.append(z["utt_id"])
                labels.append(z["label"])
                for d in DETS:
                    data[d]["logits"].append(z[f"{d}__logits"])
                    for f in DT.FEATURES[d]:
                        data[d][f].append(z[f"{d}__{f}"])
    names = np.concatenate(names).astype(str)
    labels = np.concatenate(labels)
    if len(set(names)) != names.size:
        raise RuntimeError(f"{corpus}: a file was scored twice")
    pos = pd.Series(np.arange(names.size), index=names)
    if corpus == "la19":
        order, lab = np.arange(names.size), labels.astype(int)
        base = None
    else:
        base = pd.read_csv(ws / "runs" / ("df21_v1" if corpus == "df21" else "asv5_v1") / "scores.csv",
                           usecols=["utt_id", "label"], dtype={"utt_id": str})
        miss = set(base.utt_id) - set(names)
        if miss:
            raise RuntimeError(f"{corpus}: {len(miss)} files of the earlier run were not scored (e.g. {sorted(miss)[:3]})")
        order = pos.loc[base.utt_id].to_numpy()
        lab = base.label.to_numpy()
        if corpus == "df21" and (labels[order] != lab).any():
            raise RuntimeError("df21: labels differ from the earlier run")
    la_sum = read_json(rd / "la19" / "_STAGE_SUCCESS.json") if corpus != "la19" else None
    summary = {"corpus": corpus, "n": int(order.size), "n_bona": int((lab == 1).sum()), "n_spoof": int((lab == 0).sum()),
               "files_not_in_base": int(names.size - order.size), "detectors": {}}
    for d in DETS:
        out = rd / corpus / d
        out.mkdir(parents=True, exist_ok=True)
        lg = np.concatenate(data[d]["logits"])[order]
        sc = lg[:, 1] - lg[:, 0]
        pd.DataFrame({"utt_id": names[order], "label": lab, "logit_spoof": lg[:, 0], "logit_bona": lg[:, 1],
                      "score": sc}).to_csv(out / "scores.csv.tmp", index=False)
        (out / "scores.csv.tmp").replace(out / "scores.csv")
        for f in DT.FEATURES[d]:
            arr = np.concatenate(data[d][f])[order]
            np.save(out / f"{f}.tmp.npy", arr)
            (out / f"{f}.tmp.npy").replace(out / f"{f}.npy")
        eer, tau = _eer_tau(sc[lab == 1], sc[lab == 0])
        s = {"eer": float(eer), "eer_threshold": float(tau)}
        if corpus == "la19":
            s["tau_src"] = float(tau)
            if d == "aasist":
                s["published_eer"] = AASIST_PUBLISHED_LA19_EER
        elif la_sum:
            from .metrics import hter
            s["at_tau_src"] = hter(sc[lab == 1], sc[lab == 0], la_sum["detectors"][d]["tau_src"])
        summary["detectors"][d] = s
    if corpus == "df21" and "mamba" in DETS:
        try:
            pub = _published_mamba_df21().set_index("utt_id").published
            ours = pd.Series(np.concatenate(data["mamba"]["logits"])[:, 1], index=names)
            common = pub.index.intersection(ours.index)
            a, b = ours.loc[common].to_numpy(), pub.loc[common].to_numpy()
            lab_c = pd.Series(lab, index=names[order]).loc[common].to_numpy()
            dd = np.abs(a - b)
            try:
                bm = pd.read_csv(ws / "runs" / "df21_v1" / "scores.csv", usecols=["utt_id", "codec", "subset"],
                                 dtype=str).set_index("utt_id").reindex(common)
                big = dd > 0.05
                by = {k: bm.loc[big, k].value_counts().head(12).to_dict() for k in ("codec", "subset")}
                rate = {k: (pd.Series(big, index=common).groupby(bm[k].to_numpy()).mean().round(4).to_dict())
                        for k in ("codec", "subset")}
            except Exception as e:                   # noqa: BLE001
                by, rate = {"error": str(e)}, {}
            summary["mamba_published_check"] = {
                "n_common": int(common.size), "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                "share_within_0.05": float(np.mean(dd <= 0.05)), "disagreeing_by": by, "disagreement_rate_by": rate,
                "max_abs_diff": float(np.abs(a - b).max()), "median_abs_diff": float(np.median(np.abs(a - b))),
                "eer_ours_logit_bona": float(_eer_tau(a[lab_c == 1], a[lab_c == 0])[0]),
                "eer_published": float(_eer_tau(b[lab_c == 1], b[lab_c == 0])[0])}
        except Exception as e:                       # noqa: BLE001
            summary["mamba_published_check"] = {"error": f"{type(e).__name__}: {e}"}
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_json(marker, summary)
    log(f"{corpus}: merged {summary['n']} trials; " + "; ".join(
        f"{d} EER {100 * v['eer']:.2f}%" for d, v in summary["detectors"].items()))
    return summary


def merge(args) -> int:
    ws = Path(args.workspace)
    while True:
        res = {}
        for corpus in CORPORA:
            res[corpus] = merge_corpus(ws, corpus)
            if res[corpus] is None:
                break
        if all(res.get(c) for c in CORPORA):
            log(json.dumps({c: res[c]["detectors"] for c in CORPORA}, indent=1, default=float))
            if "mamba_published_check" in res["df21"]:
                log(f"XLSR-Mamba vs published DF21 scores: {res['df21']['mamba_published_check']}")
            return 0
        if not args.wait:
            return 1
        log("waiting 2 min for the remaining units")
        time.sleep(120)


DEVIATIONS = {
    "D1_xlsr_mamba_input_length": {
        "preregistered": "score on the first 64,600 samples (official crop/tile)",
        "applied": "XLSR-Mamba is scored on the first 66,800 samples (crop/tile), its authors' evaluation recipe for "
                   "ASVspoof data (github.com/swagshaw/XLSR-Mamba data_utils.Dataset_eval); AASIST keeps 64,600",
        "reason": "the preflight check (no score stored) found that the 64,600-sample input does not reproduce the "
                  "authors' published ASVspoof 2021 DF scores (Pearson r 0.980, max difference 3.9 logits on 256 "
                  "trials); the authors' recipe is the faithful use of their detector",
        "decided": "22 Oct 2026, before any v3 score or analysis result was stored",
        "affects": "XLSR-Mamba scores only; hypotheses, method, thresholds and analyses unchanged",
    },
}


def record_deviations(ws: Path) -> None:
    """Write-once deviation record, created before the first v3 part (refused afterwards)."""
    path = Path(ws) / "results" / "PREREGISTRATION_V3_DEVIATIONS.json"
    body = {"deviations": DEVIATIONS}
    if path.exists():
        if json.loads(path.read_text()).get("deviations") != DEVIATIONS:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        return
    parts = run_dir(ws) / "parts"
    if parts.exists() and any(parts.rglob("*.npz")):
        raise SystemExit("v3 parts exist without a deviation record; refusing to back-date one")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, {**body, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    log(f"deviation record written: {path}")


def _audio_stats(blob: bytes) -> dict:
    """Digital-silence statistics of one decoded file (first 66,800 samples after crop/tile)."""
    from .data import load_audio_bytes, official_pad
    x = official_pad(load_audio_bytes(blob)[0], DT.CUTS["mamba"])
    z = np.r_[False, x == 0, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(z))                 # starts and ends of zero runs, alternating
    best = int((edges[1::2] - edges[0::2]).max()) if edges.size else 0
    return {"zero_share": float((x == 0).mean()), "longest_zero_run": best, "peak": float(np.abs(x).max())}


def _agreement(ours, ref, tol=0.05) -> dict:
    d = np.abs(ours - ref)
    return {"pearson_r": float(np.corrcoef(ours, ref)[0, 1]), "median_abs_diff": float(np.median(d)),
            "max_abs_diff": float(d.max()), "share_within_tol": float(np.mean(d <= tol)), "tol": tol}


def check(args) -> int:
    """Preflight (nothing is written to the run): both detectors load and XLSR-Mamba reproduces the
    authors' published ASVspoof 2021 DF scores. XLSR-Mamba is run twice on the same files: strict fp32
    and PyTorch's default GPU setting (TF32 cuDNN convolutions, used by the authors' torch 2.2 run);
    every file's digital-silence statistics, length, codec and subset are recorded, so that trials
    that disagree can be explained."""
    import torch
    ws = Path(args.workspace)
    token = os.environ.get("HF_TOKEN") or None
    cfg = ensure_config(ws, token)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    det_a = DT.build_aasist(device)
    ck = _fetch_mamba(Path(args.local) / "checkpoints", cfg["mamba_revision"], token)
    det_m = DT.build_mamba(device, ck, args.attn)
    u = next(x for x in cfg["units"] if x["corpus"] == "df21")
    path = _download_parquet(u["repo"], u["file"], u["revision"], Path(args.local) / "v3_check", token)
    rd = UnitReader(path, "parquet")
    n = min(args.n_check, len(rd.names))
    blobs, names = rd.read(0, n), rd.names[:n]
    bs = min(args.batch_size, 32)
    runs = {}
    for prec in ("fp32", "cudnn_tf32"):
        out, audio = score_blobs(blobs, {"mamba": DT.make_forward(det_m, device, prec)}, device, bs, 2, {"micro": bs})
        runs[prec] = out["mamba"][0][:, 1].astype(np.float64)
    out_a, _ = score_blobs(blobs, {"aasist": DT.make_forward(det_a, device, "fp32")}, device, bs, 2, {"micro": bs})
    pub = _published_mamba_df21().set_index("utt_id").published
    ref = pub.reindex(names).to_numpy()
    ok = np.isfinite(ref)
    stats = [_audio_stats(b) for b in blobs]
    meta = rd.meta.iloc[:n].reset_index(drop=True)
    res = {"n": int(n), "matched_in_published": int(ok.sum()), "mamba_input_samples": DT.CUTS["mamba"],
           "aasist_finite": bool(np.isfinite(out_a["aasist"][0]).all()),
           "mamba_load": det_m.info.get("load_report"),
           "agreement": {p: _agreement(v[ok], ref[ok]) for p, v in runs.items()}}
    best = min(res["agreement"], key=lambda p: res["agreement"][p]["median_abs_diff"])
    res["best_setting"] = best
    diff = np.abs(runs[best] - ref)
    big = ok & (diff > 0.05)
    zs = np.array([s["zero_share"] for s in stats])
    lz = np.array([s["longest_zero_run"] for s in stats])
    res["disagreeing_trials"] = {
        "n": int(big.sum()),
        "median_zero_share_disagreeing": float(np.median(zs[big])) if big.any() else None,
        "median_zero_share_agreeing": float(np.median(zs[ok & ~big])) if (ok & ~big).any() else None,
        "median_longest_zero_run_disagreeing": float(np.median(lz[big])) if big.any() else None,
        "median_longest_zero_run_agreeing": float(np.median(lz[ok & ~big])) if (ok & ~big).any() else None,
        "share_with_zero_run_over_400_samples": {
            "disagreeing": float(np.mean(lz[big] > 400)) if big.any() else None,
            "agreeing": float(np.mean(lz[ok & ~big] > 400)) if (ok & ~big).any() else None},
        "by_codec": meta.loc[big, "codec"].value_counts().to_dict() if big.any() else {},
        "by_subset": meta.loc[big, "subset"].value_counts().to_dict() if big.any() else {},
        "fp32_vs_cudnn_tf32_median_abs_diff": float(np.median(np.abs(runs["fp32"] - runs["cudnn_tf32"]))),
        "fp32_vs_cudnn_tf32_max_abs_diff": float(np.max(np.abs(runs["fp32"] - runs["cudnn_tf32"])))}
    res["per_trial"] = [{"utt_id": names[i], "published": float(ref[i]), "fp32": float(runs["fp32"][i]),
                         "cudnn_tf32": float(runs["cudnn_tf32"][i]), "label": int(rd.labels[i]),
                         "codec": str(meta.codec[i]), "subset": str(meta.subset[i]), "n_orig": int(audio["n_orig"][i]),
                         **stats[i]} for i in range(n)]
    _atomic_json(run_dir(ws) / "preflight" / f"{socket.gethostname()}.json", res)
    log(f"preflight agreement with the authors' published scores: {json.dumps(res['agreement'])}")
    log(f"disagreeing trials (> 0.05 logit, {best}): {json.dumps(res['disagreeing_trials'], default=str)}")
    for t in sorted(res["per_trial"], key=lambda t: -abs(t[best] - t["published"]))[:6]:
        log(f"  {t['utt_id']}: published {t['published']:.3f}, fp32 {t['fp32']:.3f}, cudnn_tf32 {t['cudnn_tf32']:.3f}, "
            f"zero share {t['zero_share']:.3f}, longest zero run {t['longest_zero_run']}, {t['codec']}/{t['subset']}")
    rd.close()
    try:
        Path(path).unlink()
    except OSError:
        pass
    a = res["agreement"][best]
    # Model identity: a wrong checkpoint, layer or input length moves every trial (median >> 0.01).
    if not (ok.sum() >= n // 2 and a["median_abs_diff"] < args.check_tol and a["share_within_tol"] >= 0.75):
        log("PREFLIGHT FAILED: XLSR-Mamba does not reproduce the authors' published scores; do not start scoring")
        return 1
    log(f"preflight passed: XLSR-Mamba reproduces the authors' published scores (median difference "
        f"{a['median_abs_diff']:.1e} logit, {100 * a['share_within_tol']:.0f}% of trials within 0.05, {best}); "
        "AASIST loads and scores")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["work", "status", "merge", "check"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--local", default="LOCAL_CACHE")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--stale-min", type=float, default=20.0)
    ap.add_argument("--help-min", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--precision", choices=["auto", "fp32", "tf32", "bf16", "fp16"], default="auto")
    ap.add_argument("--precision-tol", type=float, default=0.05)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--check-tol", type=float, default=0.01,
                    help="largest allowed median per-trial logit difference against the authors' published scores")
    a = ap.parse_args(argv)
    return {"work": work, "status": status, "merge": merge, "check": check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
