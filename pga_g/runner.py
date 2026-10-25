"""
Run a stage as an unbuffered subprocess: stream its output live, emit a heartbeat when it
is silent, tee everything to a persistent log on Drive, and propagate the exit code.
Also provides the pilot stages (Stage A, Stage B) with success markers.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def run_stage(cmd: list[str], log_path: Path, heartbeat_s: float = 60.0, cwd: str | None = None,
              env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    t0 = time.time()
    last = [time.time()]
    with open(log_path, "a", buffering=1) as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=cwd, env=env)
        stop = threading.Event()

        def beat():
            while not stop.wait(5):
                if time.time() - last[0] > heartbeat_s:
                    msg = f"... still running ({(time.time() - t0) / 60:.1f} min elapsed)"
                    print(msg, flush=True)
                    lf.write(msg + "\n")
                    last[0] = time.time()

        th = threading.Thread(target=beat, daemon=True)
        th.start()
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            last[0] = time.time()
        code = proc.wait()
        stop.set()
        lf.write(f"===== exit {code} after {(time.time() - t0) / 60:.1f} min\n")
    if code != 0:
        raise RuntimeError(f"stage failed with exit code {code}; see {log_path}")
    print(f"[done in {(time.time() - t0) / 60:.1f} min] log: {log_path}", flush=True)


def _hash_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        st = Path(p).stat()
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def pilot_stage(name: str, ws: Path, code_dir: Path, args: list[str], inputs: list[Path]) -> None:
    """Run pga_g.pilot into results/<name>/ unless a matching success marker exists."""
    out = ws / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "_STAGE_SUCCESS.json"
    key = _hash_files(inputs) + "|" + " ".join(args)
    if marker.exists() and json.loads(marker.read_text()).get("key") == key:
        print(f"{name}: already complete; summary:\n")
        print((out / "summary.txt").read_text())
        return
    cmd = [sys.executable, "-u", "-m", "pga_g.pilot", *args, "--out", str(out / "results.csv")]
    run_stage(cmd, out / "summary.txt.log", cwd=str(code_dir))
    log_text = (out / "summary.txt.log").read_text()
    (out / "summary.txt").write_text(log_text.split("$ ", )[-1])
    if not (out / "results.csv").exists():
        raise RuntimeError(f"{name}: results.csv missing")
    marker.write_text(json.dumps({"key": key, "args": args, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
