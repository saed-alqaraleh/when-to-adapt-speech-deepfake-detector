"""
Fetch the checkpoint and corpora in parallel, cache them on Drive, and extract them to the
fast local disk of the Colab VM.

Sources (verified 21 Oct 2026):
  checkpoint  huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST  LA_model.pth (1.27 GB)
              = the authors' ASVspoof 2019 LA model (also on the authors' Google Drive)
  itw         huggingface.co/datasets/mueller91/In-The-Wild  release_in_the_wild.zip (8.16 GB)
  la19        huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA (4.39 GB):
              the 71,237-trial LA eval set as 9 parquet shards of the original FLAC bytes plus
              the official eval protocol (the parquet mirror used by Zhou & Wang 2026).
              It is materialised to the standard LA/ASVspoof2019_LA_eval/flac layout.
              Edinburgh DataShare (LA.zip, 7.12 GB) refuses scripted downloads (HTTP 403,
              21 Oct 2026); if you download LA.zip in a browser and put it in
              <workspace>/archives/, it is used instead of the mirror.

Every step is idempotent: an existing valid file or an _EXTRACTED.json marker is reused.
One failing source does not stop the others; the stage exits non-zero listing failures.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCES = {
    "checkpoint": {"kind": "hf", "repo": "SpeechAntiSpoofingBenchmarks/W2V2-AASIST",
                   "repo_type": "model", "file": "LA_model.pth", "min_bytes": 1_200_000_000},
    "itw": {"kind": "hf", "repo": "mueller91/In-The-Wild", "repo_type": "dataset",
            "file": "release_in_the_wild.zip", "min_bytes": 7_500_000_000,
            "must_contain": "meta.csv"},
    "la19": {"kind": "url", "file": "LA.zip", "min_bytes": 6_500_000_000,
             "url": "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download",
             "must_contain": "ASVspoof2019.LA.cm.eval.trl.txt",
             "members": ["LA/ASVspoof2019_LA_eval/*", "LA/ASVspoof2019_LA_cm_protocols/*"]},
}
LA19_MIRROR = {"repo": "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", "n_shards": 9,
               "expected_rows": 71237,
               "patterns": ["data/test-*.parquet", "protocols/ASVspoof2019.LA.cm.eval.trl.txt"]}
PROTO_NAME = "ASVspoof2019.LA.cm.eval.trl.txt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ok_size(p: Path, min_bytes: int) -> bool:
    return p.exists() and p.stat().st_size >= min_bytes


def _zip_ok(p: Path, must_contain: str | None) -> bool:
    try:
        with zipfile.ZipFile(p) as z:          # reads the central directory: fails if truncated
            names = z.namelist()
        return must_contain is None or any(n.endswith(must_contain) for n in names)
    except Exception:
        return False


def _valid(p: Path, src: dict) -> bool:
    if not _ok_size(p, src["min_bytes"]):
        return False
    return _zip_ok(p, src.get("must_contain")) if p.suffix == ".zip" else True


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=64 << 20)
    tmp.replace(dst)


def _download_hf(src: dict, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=src["repo"], filename=src["file"], repo_type=src["repo_type"],
                        local_dir=str(dest_dir), token=os.environ.get("HF_TOKEN") or None)
    return Path(p)


def _download_url(src: dict, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / src["file"]
    if shutil.which("aria2c"):
        cmd = ["aria2c", "-x", "16", "-s", "16", "-k", "16M", "--continue=true",
               "--file-allocation=none", "--summary-interval=30", "--console-log-level=warn",
               "-d", str(dest_dir), "-o", src["file"], src["url"]]
        subprocess.run(cmd, check=True)
        return out
    import requests                             # fallback: single stream with resume
    have = out.stat().st_size if out.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(src["url"], stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(out, mode) as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    return out


def ensure(name: str, local_dir: Path, drive_dir: Path, cache_on_drive: bool) -> Path:
    """Return a valid local copy; reuse local, then Drive cache, else download."""
    src = SOURCES[name]
    local = local_dir / src["file"]
    cached = drive_dir / src["file"]
    if _valid(local, src):
        log(f"{name}: local copy present")
    elif _valid(cached, src):
        log(f"{name}: copying from Drive cache")
        _copy(cached, local)
    else:
        log(f"{name}: downloading {src['file']}")
        t0 = time.time()
        got = _download_hf(src, local_dir) if src["kind"] == "hf" else _download_url(src, local_dir)
        if got != local:
            shutil.move(str(got), str(local))
        if not _valid(local, src):
            raise RuntimeError(f"{name}: downloaded file failed validation ({local})")
        log(f"{name}: downloaded {local.stat().st_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")
    if cache_on_drive and not _valid(cached, src):
        log(f"{name}: caching on Drive -> {cached}")
        _copy(local, cached)
    return local


def extract(name: str, archive: Path, dest: Path) -> Path:
    src = SOURCES[name]
    marker = dest / "_EXTRACTED.json"
    if marker.exists() and json.loads(marker.read_text()).get("archive_size") == archive.stat().st_size:
        log(f"{name}: already extracted in {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    members = src.get("members", [])
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", "-o", str(archive), *members, "-d", str(dest)], check=True)
    else:
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist()
                     if not members or any(n.startswith(m.rstrip("*")) for m in members)]
            z.extractall(dest, members=names)
    marker.write_text(json.dumps({"archive": str(archive), "archive_size": archive.stat().st_size,
                                  "members": members, "seconds": round(time.time() - t0, 1)}))
    log(f"{name}: extracted in {(time.time() - t0) / 60:.1f} min")
    return dest


# ------------------------------------------------------------------ ASVspoof 2019 LA mirror
def download_la19_mirror(dest: Path) -> tuple[Path, str]:
    """Download the parquet shards + protocol; returns (dir, repo commit sha)."""
    from huggingface_hub import HfApi, snapshot_download
    token = os.environ.get("HF_TOKEN") or None
    sha = HfApi().dataset_info(LA19_MIRROR["repo"], token=token).sha
    t0 = time.time()
    snapshot_download(repo_id=LA19_MIRROR["repo"], repo_type="dataset", revision=sha,
                      allow_patterns=LA19_MIRROR["patterns"], local_dir=str(dest), token=token,
                      max_workers=16)
    shards = sorted((dest / "data").glob("test-*.parquet"))
    if len(shards) != LA19_MIRROR["n_shards"] or not (dest / "protocols" / PROTO_NAME).exists():
        raise RuntimeError(f"mirror incomplete in {dest}: {len(shards)} shards")
    log(f"la19: mirror downloaded ({sum(p.stat().st_size for p in shards) / 1e9:.2f} GB, "
        f"revision {sha[:10]}) in {(time.time() - t0) / 60:.1f} min")
    return dest, sha


def _audio_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return b
        raise ValueError(f"audio cell without bytes (path={cell.get('path')})")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unexpected audio cell type {type(cell)}")


def _write_shard(pq_path: Path, flac_dir: Path) -> dict:
    """Write each row's audio bytes to flac_dir/<utt>.flac. Original FLAC bytes are copied
    unchanged; any other container is decoded and re-encoded losslessly as FLAC (counted)."""
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    n = reenc = 0
    labels = {}
    pf = pq.ParquetFile(pq_path)
    for batch in pf.iter_batches(batch_size=256, columns=["path", "audio", "label", "notes"]):
        d = batch.to_pydict()
        for path, audio, label, notes in zip(d["path"], d["audio"], d["label"], d["notes"]):
            utt = json.loads(notes).get("utterance_id") if notes else None
            utt = utt or Path(path).stem
            out = flac_dir / f"{utt}.flac"
            data = _audio_bytes(audio)
            if not out.exists() or out.stat().st_size == 0:
                tmp = out.with_name(out.name + ".part")
                if data[:4] == b"fLaC":
                    tmp.write_bytes(data)
                else:
                    x, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
                    sf.write(tmp, x, sr, format="FLAC", subtype="PCM_16")
                    reenc += 1
                tmp.replace(out)
            labels[utt] = int(label) if not isinstance(label, str) else (0 if label == "bonafide" else 1)
            n += 1
    return {"rows": n, "reencoded": reenc, "labels": labels}


def materialise_la19(mirror_dir: Path, sha: str, dest: Path, workers: int = 9) -> Path:
    """Build dest/LA/ASVspoof2019_LA_eval/flac + the protocol, cross-checking every label."""
    marker = dest / "_EXTRACTED.json"
    if marker.exists() and json.loads(marker.read_text()).get("source_revision") == sha:
        log(f"la19: already materialised in {dest}")
        return dest
    t0 = time.time()
    flac_dir = dest / "LA" / "ASVspoof2019_LA_eval" / "flac"
    proto_dir = dest / "LA" / "ASVspoof2019_LA_cm_protocols"
    flac_dir.mkdir(parents=True, exist_ok=True)
    proto_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mirror_dir / "protocols" / PROTO_NAME, proto_dir / PROTO_NAME)
    shards = sorted((mirror_dir / "data").glob("test-*.parquet"))
    with ThreadPoolExecutor(max_workers=workers) as ex:      # one thread per shard
        parts = list(ex.map(lambda p: _write_shard(p, flac_dir), shards))
    rows = sum(p["rows"] for p in parts)
    labels = {k: v for p in parts for k, v in p["labels"].items()}
    proto = {}
    for line in (proto_dir / PROTO_NAME).read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            proto[f[1]] = 0 if f[4] == "bonafide" else 1        # mirror coding: 0 = bonafide
    missing = [u for u in proto if not (flac_dir / f"{u}.flac").exists()]
    mismatch = [u for u in proto if u in labels and labels[u] != proto[u]]
    if rows != LA19_MIRROR["expected_rows"] or len(proto) != rows or missing or mismatch:
        raise RuntimeError(f"la19 mirror check failed: rows={rows}, protocol={len(proto)}, "
                           f"missing files={len(missing)}, label mismatches={len(mismatch)}")
    marker.write_text(json.dumps({"source": f"hf:{LA19_MIRROR['repo']}", "source_revision": sha,
                                  "rows": rows, "reencoded": sum(p["reencoded"] for p in parts),
                                  "labels_checked_against_protocol": len(proto),
                                  "seconds": round(time.time() - t0, 1)}, indent=2))
    log(f"la19: materialised {rows} flac files in {(time.time() - t0) / 60:.1f} min "
        f"(all labels agree with the official protocol)")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, help="Drive workspace, e.g. WORKSPACE")
    ap.add_argument("--local", default="LOCAL_CACHE", help="fast local disk")
    ap.add_argument("--what", nargs="+", default=["checkpoint", "itw", "la19"], choices=sorted(SOURCES))
    ap.add_argument("--cache-datasets-on-drive", action="store_true",
                    help="also keep the 8.2 GB + 7.1 GB archives on Drive (needs Drive space)")
    a = ap.parse_args(argv)
    ws, loc = Path(a.workspace), Path(a.local)
    t0 = time.time()

    def job(name):
        try:
            if name == "la19":
                zip_local, zip_drive = loc / "archives" / "LA.zip", ws / "archives" / "LA.zip"
                if _valid(zip_local, SOURCES["la19"]) or _valid(zip_drive, SOURCES["la19"]):
                    path = ensure("la19", loc / "archives", ws / "archives", cache_on_drive=False)
                    extract("la19", path, loc / "data" / "la19")
                    return name, f"ok: {path}"
                mdir, sha = download_la19_mirror(loc / "archives" / "la19_hf")
                materialise_la19(mdir, sha, loc / "data" / "la19")
                return name, f"ok: hf mirror @ {sha[:10]}"
            is_ckpt = name == "checkpoint"
            ldir = loc / ("checkpoints" if is_ckpt else "archives")
            ddir = ws / ("checkpoints" if is_ckpt else "archives")
            path = ensure(name, ldir, ddir, cache_on_drive=is_ckpt or a.cache_datasets_on_drive)
            if not is_ckpt:
                extract(name, path, loc / "data" / name)
            return name, f"ok: {path}"
        except Exception as e:                   # isolate failures: other sources keep going
            log(f"{name}: FAILED: {type(e).__name__}: {e}")
            return name, f"FAILED: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=len(a.what)) as ex:      # downloads/extractions in parallel
        results = dict(ex.map(job, a.what))
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    (ws / "logs" / "fetch_last.json").write_text(json.dumps(
        {"results": results, "seconds": round(time.time() - t0, 1),
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    failed = {k: v for k, v in results.items() if v.startswith("FAILED")}
    for k, v in results.items():
        log(f"  {k:10s} {v}")
    if failed:
        log(f"fetch finished with failures in {sorted(failed)}; completed sources are kept")
        return 1
    log(f"fetch complete in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
