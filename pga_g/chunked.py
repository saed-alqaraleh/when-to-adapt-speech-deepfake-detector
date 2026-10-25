"""
Resumable, process-parallel task engine for the CPU analyses.

* Every task has a stable key. Finished task rows are flushed to Drive in small chunk files
  (atomic writes) every `flush_every` tasks or `flush_s` seconds, so an interruption loses
  at most the tasks in flight; a rerun loads the chunks and runs only the missing keys.
* `part` / `n_parts` split the task list deterministically so that several Colab runtimes
  can share one analysis (each writes its own chunks into the same directory). A runtime that
  finishes its own part then takes the other parts' unfinished tasks from the end of their
  lists, re-reading Drive at every flush and skipping tasks finished elsewhere, so fast and slow
  runtimes finish together (a task done twice is identical; one copy is kept). The final
  results.csv is assembled by whichever run finds every key present.
* Results do not depend on scheduling: each task derives its own seed from its key.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_key(task: tuple) -> str:
    return "|".join(str(t) for t in task)


def load_done(out_dir: Path) -> tuple[pd.DataFrame, set]:
    chunks = sorted((out_dir / "chunks").glob("chunk_*.csv"))
    if not chunks:
        return pd.DataFrame(), set()
    df = pd.concat([pd.read_csv(c).assign(_chunk=c.name) for c in chunks], ignore_index=True)
    df["task_key"] = df["task_key"].astype(str)
    first = df.groupby("task_key")["_chunk"].transform("min")      # a key re-run elsewhere: keep one copy
    df = df[df["_chunk"] == first].drop(columns="_chunk").reset_index(drop=True)
    return df, set(df["task_key"])


def done_keys(out_dir: Path) -> set:
    keys = set()
    for c in (out_dir / "chunks").glob("chunk_*.csv"):
        keys.update(pd.read_csv(c, usecols=["task_key"]).task_key.astype(str))
    return keys


def _flush(out_dir: Path, rows: list, tag: str) -> None:
    if not rows:
        return
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    name = f"chunk_{time.strftime('%Y%m%d%H%M%S')}_{tag}_{os.getpid()}_{time.time_ns() % 10**9:09d}.csv"
    tmp = out_dir / "chunks" / (name + ".tmp")
    with open(tmp, "w") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(out_dir / "chunks" / name)


def _call(fn, task):
    rows = fn(*task)
    key = task_key(task)
    for r in rows:
        r["task_key"] = key
    return rows


def run_chunked(fn: Callable, tasks: Iterable[tuple], out_dir: Path, init: Callable, init_args: tuple,
                jobs: int = 0, part: int = 0, n_parts: int = 1, flush_every: int = 20,
                flush_s: float = 60.0, label: str = "tasks", config: dict | None = None,
                max_tasks: int | None = None, steal: bool = True) -> pd.DataFrame | None:
    """Run every task whose key is not yet on disk; return the full results when complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(tasks)
    all_keys = [task_key(t) for t in tasks]
    cfg = {"n_tasks": len(tasks), "keys_sha": hashlib.sha256("\n".join(all_keys).encode()).hexdigest()[:16],
           **(config or {})}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old.get("keys_sha") != cfg["keys_sha"] or {k: v for k, v in old.items() if k != "created"} != cfg:
            raise SystemExit(f"{out_dir} holds a different analysis configuration; use a new output directory")
    else:
        tmp = cfg_path.with_name("config.json.tmp")
        tmp.write_text(json.dumps({**cfg, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        tmp.replace(cfg_path)

    _, done = load_done(out_dir)
    own = [t for i, t in enumerate(tasks) if i % n_parts == part and task_key(t) not in done]
    others = [t for i, t in enumerate(tasks) if i % n_parts != part and task_key(t) not in done][::-1]
    queue = own + (others if steal else [])
    if max_tasks is not None:
        queue = queue[:max_tasks]
    jobs = jobs if jobs > 0 else (os.cpu_count() or 1)
    by_part: dict = {}
    for c in (out_dir / "chunks").glob("chunk_*.csv"):
        tag = c.name.split("_")[2] if c.name.count("_") >= 4 else "?"
        by_part[tag] = by_part.get(tag, 0) + 1
    log(f"[{label}] chunk files on Drive by writing part: {dict(sorted(by_part.items())) or 'none'}")
    log(f"[{label}] {len(tasks)} tasks, {len(done)} done on Drive, {len(own)} of my part to run here, then "
        f"{len(others) if steal else 0} left by other parts (taken from the end; part {part + 1}/{n_parts}, "
        f"{jobs} processes)")
    if (os.cpu_count() or 1) < 8:
        log(f"[{label}] NOTE: this runtime has only {os.cpu_count()} CPU cores; a runtime with more cores "
            "finishes proportionally faster")
    t0, buf, last_flush, n_done, n_skip = time.time(), [], time.time(), 0, 0
    seen = set(done)

    def refresh():
        """Re-read which tasks other runtimes have finished (their chunks reach Drive in batches)."""
        nonlocal seen
        try:
            seen = done_keys(out_dir) | seen
        except Exception:
            pass

    def progress():
        rate = n_done / max(time.time() - t0, 1e-9)
        left = sum(1 for t in queue if task_key(t) not in seen)
        log(f"[{label}] {n_done} done here, {n_skip} skipped (done elsewhere) | {rate * 60:.1f} tasks/min | "
            f"~{left} tasks left overall | ETA here {left / max(rate, 1e-9) / 60:.1f} min")

    def flush():
        nonlocal buf, last_flush
        if buf:
            seen.update(r["task_key"] for r in buf)
            _flush(out_dir, buf, f"p{part}")
        buf, last_flush = [], time.time()
        refresh()
        progress()

    if queue:
        if jobs <= 1:
            init(*init_args)
            for t in queue:
                if task_key(t) in seen:
                    n_skip += 1
                    continue
                buf.extend(_call(fn, t))
                n_done += 1
                if n_done % flush_every == 0 or time.time() - last_flush > flush_s:
                    flush()
        else:
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init,
                                     initargs=init_args) as ex:
                it, pending = iter(queue), set()

                def refill():
                    nonlocal n_skip
                    while len(pending) < 2 * jobs:            # bounded: skip tasks finished elsewhere
                        t = next(it, None)
                        if t is None:
                            return
                        if task_key(t) in seen:
                            n_skip += 1
                            continue
                        pending.add(ex.submit(_call, fn, t))
                refill()
                while pending:
                    finished, rest = wait(pending, timeout=flush_s, return_when=FIRST_COMPLETED)
                    pending.clear()
                    pending.update(rest)
                    for f in finished:
                        buf.extend(f.result())
                        n_done += 1
                    if buf and (len(buf) >= flush_every or time.time() - last_flush > flush_s):
                        flush()
                    refill()
        flush()

    df, done = load_done(out_dir)
    missing = [k for k in all_keys if k not in done]
    if missing:
        log(f"[{label}] {len(missing)} tasks still missing (other parts or an interrupted run); "
            "rerun this cell, or the other part, to complete")
        return None
    order = {k: i for i, k in enumerate(all_keys)}
    df = df.assign(_o=df["task_key"].map(order)).sort_values(["_o"], kind="mergesort").drop(columns="_o")
    tmp = out_dir / "results.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(out_dir / "results.csv")
    (out_dir / "_STAGE_SUCCESS.json").write_text(json.dumps(
        {**cfg, "rows": int(len(df)), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    log(f"[{label}] complete: {len(df)} rows -> {out_dir / 'results.csv'}")
    return df
