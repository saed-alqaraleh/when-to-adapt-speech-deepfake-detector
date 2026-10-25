"""
v4: threat model analyses, a detector-agnostic guard and the cost of one adaptation decision
(CPU only; uses the scores and features of the earlier runs). Pre-registered before any v4 result.

Analyses (results/v4/<kind>_<corpus>_<detector>/):
  poison    the genuine reference contains 0-20% deepfakes (random, or all from the campaign's own
            synthesis system); gate, guard, EER, and acceptance of the planted system's spoofs
  edge      an attacker holds the spoof share just below / above the gate (0.70-0.85), matched and
            speaker-disjoint references: does PGA-G ever do worse than the frozen detector?
  agnostic  the guard computed from a self-supervised probe (SSL-AASIST features; XLSR-Mamba features
            for SSL-AASIST itself) instead of the adaptation features; same draws and folds as v3 main
  timing    wall-clock time of one adaptation decision (results/v4/timing.csv)

    python -m pga_g.v4 prereg --workspace W
    python -m pga_g.v4 all    --workspace W --part 0 --n-parts 2 --local LOCAL_CACHE
    python -m pga_g.v4 timing --workspace W --local LOCAL_CACHE
    python -m pga_g.v4 summary --workspace W
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pga_g import pilot as P

from . import general as G
from . import robust as R
from .chunked import log, run_chunked
from .detectors import NAMES, PRIMARY

DETECTORS, CORPORA = ("sslaasist", "aasist", "mamba"), ("df21", "asv5")
POISON_RATES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
POISON_SHARES = (0.05, 0.50, 0.90)
SCENARIOS = ("random", "targeted")
EDGE_SHARES = (0.70, 0.75, 0.78, 0.82, 0.85)
PROBE = {"aasist": "sslaasist", "mamba": "sslaasist", "sslaasist": "mamba"}
KINDS = ("poison", "edge", "agnostic")
TIMING_N, TIMING_DIMS, TIMING_REPS = (1000, 5000, 12000), (160, 1024), 5

PREREG = {
    "title": "Threat-model analyses, detector-agnostic guard and decision cost (v4)",
    "fixed_on": "2026-10-22, before any v4 result (results/v4) existed; all inputs are the existing v3 and earlier scores",
    "status": "pre-specified exploratory analyses; every result is reported, none is a confirmatory decision",
    "method": "unchanged PGA-G (alpha 0.05, 5-fold PU logistic regression C 0.1, gate 0.80, guard 0.5) unless stated",
    "detectors_and_corpora": "SSL-AASIST, XLSR-Mamba, AASIST on ASVspoof 2021 DF and ASVspoof 5 (complete evaluation sets)",
    "poison": ("matched 1,000-recording reference in which a fraction r of 0, 0.01, 0.02, 0.05, 0.10, 0.20 is replaced by "
               "spoofed trials, before the random train/calibration halving; scenario 'random': spoofs of any attack; "
               "scenario 'targeted': all planted spoofs from one attack drawn at random per repetition (the attacker "
               "enrols deepfakes of the system used in the campaign); traffic spoof shares 0.05, 0.50, 0.90; source groups "
               "of planted spoofs excluded from the traffic; 20 repetitions; reported: gate and adaptation rates, guard "
               "statistic, EER, anchor FRR/FAR/HTER of the frozen detector and of PGA-G (both thresholded with the "
               "poisoned reference), and for 'targeted' the EER and anchor FAR on the planted attack's traffic trials"),
    "edge": ("spoof shares 0.70, 0.75, 0.78, 0.82, 0.85 (an attacker diluting the campaign to stay near the gate), matched "
             "and speaker-disjoint references, 20 repetitions; reported: EER, anchor FRR and HTER of PGA-G relative to the "
             "frozen detector; together with all v3 main conditions this measures the fail-safe property (PGA-G not worse "
             "than the frozen detector for any traffic composition)"),
    "agnostic": ("the guard statistic computed from a PU classifier on a self-supervised probe instead of the adaptation "
                 "features: probe = SSL-AASIST time-mean XLS-R features for AASIST and XLSR-Mamba, XLSR-Mamba's for "
                 "SSL-AASIST; adaptation, gate and anchor unchanged; same references, traffic and folds as the v3 main "
                 "analysis (the PGA-G rows with the original guard are reproduced exactly); hypotheses H2-H5 of v3 "
                 "evaluated for both guards"),
    "timing": ("wall-clock time of one decision (gate estimate, 5-fold cross-fitted classifier, guard estimate, fold-wise "
               "anchors) on traffic of 1,000, 5,000, 12,000 trials with 160- and 1,024-dimensional features, one thread "
               "and all threads, 5 repetitions, on the Colab CPU (model recorded)"),
}


# ---------------------------------------------------------------- pre-registration
def prereg(ws: Path) -> str:
    ws = Path(ws)
    path = ws / "results" / "PREREGISTRATION_V4.json"
    body = json.dumps(PREREG, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        if json.loads(path.read_text()).get("sha256") != digest:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        return digest
    res = ws / "results" / "v4"
    if res.exists() and (any(res.rglob("chunk_*.csv")) or (res / "timing.csv").exists()):
        raise SystemExit("v4 analyses have already started: a pre-registration written now would not be one")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"preregistration": PREREG, "sha256": digest,
                               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    tmp.replace(path)
    log(f"pre-registration written: {path} (sha256 {digest[:16]})")
    return digest


# ---------------------------------------------------------------- inputs
def _init(meta, score, feats, corpus, det, probe):
    G._init(meta, score, feats, corpus, det)
    G._G["probe"] = probe
    G._G["attack"] = meta.attack.astype(str).to_numpy()


def _load(ws, corpus, det, local):
    meta, score, feats = G.load(ws, corpus, det, local)
    probe = G.load(ws, corpus, PROBE[det], local)[2]["ssl1024"]
    return meta, score, feats, probe


# ---------------------------------------------------------------- cells
def poison_cell(pi, rate, scenario, rep, seed):
    rng = np.random.default_rng([seed, 71, G.CID[G._G["corpus"]], int(round(pi * 1000)), int(round(rate * 1000)),
                                 0 if scenario == "random" else 1, rep])
    lab, att, fz = G._G["lab"], G._G["attack"], G._G["score"]
    bona = G._G["bona"]
    spoof = np.flatnonzero(lab == 0)
    n_p = int(round(rate * G.N_REF))
    target = None
    if scenario == "targeted":
        names, counts = np.unique(att[spoof], return_counts=True)
        target = str(rng.choice(names[counts >= 2000]))
        cand = spoof[att[spoof] == target]
    else:
        cand = spoof
    planted = rng.choice(cand, n_p, replace=False) if n_p else np.array([], int)
    ref = np.r_[rng.choice(bona, G.N_REF - n_p, replace=False), planted].astype(int)
    keep = np.ones(lab.size, bool)
    keep[ref] = False
    pool = G._exclude(keep, ref)
    b, s = G.draw_traffic(pool, pi, rng)
    traffic, nb = np.r_[b, s], b.size
    ref = rng.permutation(ref)
    tr, cal = ref[: ref.size // 2], ref[ref.size // 2:]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= G.GATE)
    base = {"pi": pi, "rate": rate, "scenario": scenario, "rep": rep, "target": target, "n_planted": n_p,
            "pi_hat_frozen": 1 - b_fr, "gate_open": gate_open}

    def targeted_metrics(s_cal, s_u, folds=None):
        if target is None:
            return {}
        m = att[s] == target
        if m.sum() < 20:
            return {"n_target": int(m.sum())}
        sb, ss = s_u[:nb], s_u[nb:][m]
        if folds is None:
            tau = P.conformal_bona_threshold(s_cal, P.ALPHA)
            far = float(np.mean(ss >= tau))
        else:
            op = P.foldwise_anchor_op(s_cal, folds[0], sb, folds[1][:nb], ss, folds[1][nb:][m])
            far = op.far
        return {"n_target": int(m.sum()), "eer_target": P.eer(sb, ss), "far_target": far}

    frozen = {**G._evaluate(fz[cal], fz[traffic], nb), **targeted_metrics(fz[cal], fz[traffic])}
    rows = [{**base, "variant": "frozen", "adapted": False, **frozen}]
    if gate_open:                                     # the classifier is only fitted when PGA-G would use it
        X = G._G["feats"][PRIMARY[G._G["det"]]]
        Xtr, Xcal, Xu = G._feat(X, tr), G._feat(X, cal), G._feat(X, traffic)
        a = R.crossfit(Xtr, Xcal, Xu, rng)
        ev = {**G._evaluate(a[0], a[2], nb, (a[1], a[3])), **targeted_metrics(a[0], a[2], (a[1], a[3]))}
        b_ad = R.bona_share(a[0], a[2])
        ratio = b_ad / max(b_fr, 1e-9) if np.isfinite(b_ad) else float("nan")
        guard = bool(np.isfinite(ratio) and ratio >= G.RHO)
        rows.append({**base, "variant": "adapt_primary", "adapted": True, "guard_ratio": ratio, **ev})
        rows.append({**base, "variant": "pgag_primary", "adapted": guard, "guard_ratio": ratio,
                     **(ev if guard else frozen)})
    else:
        rows.append({**base, "variant": "pgag_primary", "adapted": False, "guard_ratio": float("nan"), **frozen})
    return rows


def edge_cell(mode, pi, rep, seed):
    rng = np.random.default_rng([seed, 73, G.CID[G._G["corpus"]], 0 if mode == "overlap" else 1,
                                 int(round(pi * 1000)), rep])
    ref, pool, n_spk = G.split(mode, rng, G.N_REF)
    b, s = G.draw_traffic(pool, pi, rng)
    ref = rng.permutation(ref)
    tr, cal, traffic, nb = ref[: ref.size // 2], ref[ref.size // 2:], np.r_[b, s], b.size
    fz = G._G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= G.GATE)
    base = {"mode": mode, "pi": pi, "rep": rep, "n_traffic": int(traffic.size), "gate_open": gate_open,
            "pi_hat_frozen": 1 - b_fr}
    frozen = G._evaluate(fz[cal], fz[traffic], nb)
    ev, ratio, guard, b_ad = G._adapt(fz, tr, cal, traffic, nb, rng, G._G["feats"][PRIMARY[G._G["det"]]], b_fr, gate_open)
    return [{**base, "variant": "frozen", "adapted": False, **frozen}] + \
        G._rows3(base, "primary", frozen, ev, ratio, guard, gate_open, b_ad)


def agnostic_cell(mode, pi, rep, seed):
    """Exactly the v3 main cell's draws and primary adaptation, then the probe guard."""
    rng, tr, cal, traffic, nb, n_spk = G._draw_main(mode, pi, rep, seed)
    fz = G._G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= G.GATE)
    base = {"mode": mode, "pi": pi, "rep": rep, "n_ref_speakers": n_spk, "gate_open": gate_open}
    frozen = G._evaluate(fz[cal], fz[traffic], nb)
    ev, ratio_own, guard_own, _ = G._adapt(fz, tr, cal, traffic, nb, rng, G._G["feats"][PRIMARY[G._G["det"]]],
                                           b_fr, gate_open)
    Xp = G._G["probe"]
    a = R.crossfit(G._feat(Xp, tr), G._feat(Xp, cal), G._feat(Xp, traffic), rng)
    b_p = R.bona_share(a[0], a[2])
    ratio_p = b_p / max(b_fr, 1e-9) if np.isfinite(b_p) else float("nan")
    guard_p = bool(np.isfinite(ratio_p) and ratio_p >= G.RHO)
    return [{**base, "variant": "frozen", "adapted": False, **frozen},
            {**base, "variant": "pgag_own", "adapted": gate_open and guard_own, "guard_ratio": ratio_own,
             **(ev if gate_open and guard_own else frozen)},
            {**base, "variant": "pgag_probe", "adapted": gate_open and guard_p, "guard_ratio": ratio_p,
             **(ev if gate_open and guard_p else frozen)},
            {**base, "variant": "pgag_both", "adapted": gate_open and guard_p and guard_own,
             "guard_ratio": min(ratio_p, ratio_own) if np.isfinite(ratio_p) and np.isfinite(ratio_own) else float("nan"),
             **(ev if gate_open and guard_p and guard_own else frozen)}]


def tasks_for(kind, corpus, seed):
    if kind == "poison":
        return poison_cell, [(p, r, sc, rep, seed) for p in POISON_SHARES for r in POISON_RATES for sc in SCENARIOS
                             for rep in range(G.REPS)], 20
    if kind == "edge":
        return edge_cell, [(m, p, rep, seed) for m in ("overlap", "disjoint") for p in EDGE_SHARES
                           for rep in range(G.REPS)], 10
    if kind == "agnostic":
        shares = sorted(set(G.SHARES[corpus]) | {G.NATURAL[corpus]})
        return agnostic_cell, [(m, p, rep, seed) for m in ("overlap", "disjoint") for p in shares
                               for rep in range(G.REPS)], 10
    raise ValueError(kind)


def out_dir(ws, kind, corpus, det) -> Path:
    return Path(ws) / "results" / "v4" / f"{kind}_{corpus}_{det}"


def _run(ws, kind, corpus, det, part, n_parts, jobs, seed, local, max_tasks=None):
    digest = prereg(ws)
    meta, score, feats, probe = _load(ws, corpus, det, local)
    fn, tasks, flush = tasks_for(kind, corpus, seed)
    return run_chunked(fn, tasks, out_dir(ws, kind, corpus, det), _init, (meta, score, feats, corpus, det, probe),
                       jobs=jobs, part=part, n_parts=n_parts, flush_every=flush,
                       label=f"v4 {kind} {corpus} {NAMES[det]}", max_tasks=max_tasks,
                       config={"prereg_sha256": digest, "kind": kind, "corpus": corpus, "detector": det, "seed": seed})


# ---------------------------------------------------------------- timing
def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def timing(ws: Path, local=None, reps: int = TIMING_REPS) -> int:
    from threadpoolctl import threadpool_limits
    ws = Path(ws)
    prereg(ws)
    meta, score, feats = G.load(ws, "asv5", "sslaasist", local)
    G._init(meta, score, feats, "asv5", "sslaasist")
    rows = []
    for threads in ("1", "all"):
        ctx = threadpool_limits(1) if threads == "1" else threadpool_limits(None)
        with ctx:
            for n in TIMING_N:
                for d in TIMING_DIMS:
                    for rep in range(reps):
                        rng = np.random.default_rng([2026, 79, n, d, rep])
                        ref, pool, _ = G.split("overlap", rng, G.N_REF)
                        b, s = G.draw_traffic(pool, 0.9, rng, n_max=n)
                        ref = rng.permutation(ref)
                        tr, cal, traffic, nb = ref[: ref.size // 2], ref[ref.size // 2:], np.r_[b, s], b.size
                        X = feats["ssl1024"] if d == 1024 else feats["emb160"]
                        Xtr, Xcal, Xu = G._feat(X, tr), G._feat(X, cal), G._feat(X, traffic)
                        fz = score
                        t0 = time.perf_counter()
                        b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
                        t1 = time.perf_counter()
                        a = R.crossfit(Xtr, Xcal, Xu, rng)
                        t2 = time.perf_counter()
                        R.bona_share(a[0], a[2])
                        t3 = time.perf_counter()
                        P.foldwise_anchor_op(a[0], a[1], a[2][:nb], a[3][:nb], a[2][nb:], a[3][nb:])
                        t4 = time.perf_counter()
                        rows.append({"threads": threads, "n_traffic": int(traffic.size), "dim": d, "rep": rep,
                                     "gate_s": t1 - t0, "classifier_s": t2 - t1, "guard_s": t3 - t2, "anchor_s": t4 - t3,
                                     "total_s": t4 - t0, "cpu": _cpu_model(), "n_cpus": os.cpu_count()})
                        log(f"timing threads={threads} n={traffic.size} d={d} rep={rep}: {t4 - t0:.2f}s")
    out = ws / "results" / "v4" / "timing.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out.with_name("timing.csv.tmp"), index=False)
    out.with_name("timing.csv.tmp").replace(out)
    log(f"timing written: {out}")
    return 0


# ---------------------------------------------------------------- summary
def _read(ws, kind, corpus, det):
    p = out_dir(ws, kind, corpus, det) / "results.csv"
    return pd.read_csv(p) if p.exists() else None


def summary(ws: Path) -> int:
    ws = Path(ws)
    rng = np.random.default_rng(0)
    S = {"prereg_sha256": prereg(ws), "poison": {}, "edge": {}, "agnostic": {}, "failsafe": {}, "timing": None}
    lines = ["v4 summary"]
    for det in DETECTORS:
        for c in CORPORA:
            key = f"{det}|{c}"
            p = _read(ws, "poison", c, det)
            if p is not None:
                t = p.groupby(["scenario", "pi", "rate", "variant"]).agg(
                    eer=("eer", "mean"), frr=("frr", "mean"), far=("far", "mean"), hter=("hter", "mean"),
                    adapted=("adapted", "mean"), gate=("gate_open", "mean"),
                    eer_target=("eer_target", "mean") if "eer_target" in p else ("eer", "size"),
                    far_target=("far_target", "mean") if "far_target" in p else ("eer", "size")).reset_index()
                S["poison"][key] = t.round(5).to_dict("records")
            e = _read(ws, "edge", c, det)
            m3 = G._read(ws, "main", c, det)
            frames = [x for x in (e, m3) if x is not None]
            if frames:
                allm = pd.concat([f[f.variant.isin(["frozen", "pgag_primary"])][["mode", "pi", "rep", "variant", "eer",
                                                                                  "frr", "hter"]] for f in frames])
                g = allm.groupby(["mode", "pi", "variant"])[["eer", "frr", "hter"]].mean()
                r = g.xs("pgag_primary", level="variant") / g.xs("frozen", level="variant")
                d_frr = g.xs("pgag_primary", level="variant").frr - g.xs("frozen", level="variant").frr
                S["failsafe"][key] = {"n_conditions": int(len(r)), "max_eer_ratio": float(r.eer.max()),
                                      "max_hter_ratio": float(r.hter.max()), "max_frr_increase": float(d_frr.max()),
                                      "argmax_eer": str(r.eer.idxmax())}
                if e is not None:
                    S["edge"][key] = e[e.variant.isin(["frozen", "pgag_primary"])].groupby(
                        ["mode", "pi", "variant"])[["eer", "frr", "hter", "adapted", "gate_open"]].mean().round(5) \
                        .reset_index().to_dict("records")
                lines.append(f"{NAMES[det]:<11} {c}: fail-safe over {len(r)} conditions: max EER ratio "
                             f"{r.eer.max():.3f}, max HTER ratio {r.hter.max():.3f}, max FRR increase {100 * d_frr.max():.1f} pt")
            a = _read(ws, "agnostic", c, det)
            if a is not None:
                hyp = {}
                for v in ("pgag_own", "pgag_probe", "pgag_both"):
                    h = G.hypotheses(a.rename(columns={}).assign(), c, v, rng)
                    hyp[v] = {k: h[k] for k in ("H2_no_harm_low_share", "H3_no_harm_disjoint", "H4_genuine_rejection",
                                                "H5_shortcut", "H1_gain")}
                    hyp[v]["safety_pass"] = h["safety_pass"]
                    hyp[v]["all_pass"] = h["all_pass"]
                pr = a[a.variant == "pgag_probe"]
                sep = pr[pr["mode"] == "overlap"].guard_ratio.min() / pr[pr["mode"] == "disjoint"].guard_ratio.max()
                S["agnostic"][key] = {"hypotheses": hyp, "probe_separation": float(sep),
                                      "probe_overlap_min": float(pr[pr["mode"] == "overlap"].guard_ratio.min()),
                                      "probe_disjoint_max": float(pr[pr["mode"] == "disjoint"].guard_ratio.max())}
                lines.append(f"{NAMES[det]:<11} {c}: agnostic guard safety (H2-H5) "
                             f"{'pass' if hyp['pgag_probe']['safety_pass'] else 'fail'}, all H1-H5 "
                             f"{'pass' if hyp['pgag_probe']['all_pass'] else 'fail'}; own guard safety "
                             f"{'pass' if hyp['pgag_own']['safety_pass'] else 'fail'}; probe separation {sep:.2f}")
    tp = ws / "results" / "v4" / "timing.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        S["timing"] = t.groupby(["threads", "n_traffic", "dim"])[["gate_s", "classifier_s", "guard_s", "anchor_s",
                                                                   "total_s"]].median().round(4).reset_index() \
            .to_dict("records")
        S["timing_cpu"] = {"cpu": str(t.cpu.iloc[0]), "n_cpus": int(t.n_cpus.iloc[0])}
        big = t[(t.n_traffic == t.n_traffic.max()) & (t.dim == 1024)].groupby("threads").total_s.median()
        lines.append(f"timing: one decision on {t.n_traffic.max()} trials x 1,024 dims: "
                     + ", ".join(f"{k} thread(s) {v:.2f} s" for k, v in big.items()))
    out = ws / "results" / "V4_RESULTS.json"
    out.with_name(out.name + ".tmp").write_text(json.dumps(S, indent=1, default=float))
    out.with_name(out.name + ".tmp").replace(out)
    (ws / "results" / "V4_SUMMARY.txt").write_text("\n".join(lines) + "\n")
    for ln in lines:
        log(ln)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prereg", "all", "timing", "summary"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--detectors", default="sslaasist,aasist,mamba")
    ap.add_argument("--corpora", default="df21,asv5")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--local", default=None)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--timing-reps", type=int, default=TIMING_REPS)
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    if a.cmd == "prereg":
        prereg(ws)
        return 0
    if a.cmd == "timing":
        return timing(ws, a.local, a.timing_reps)
    if a.cmd == "summary":
        return summary(ws)
    complete = True
    for kind in a.kinds.split(","):
        for det in a.detectors.split(","):
            for corpus in a.corpora.split(","):
                if not (G.available(ws, corpus, det) and G.available(ws, corpus, PROBE[det])):
                    log(f"{NAMES[det]} on {corpus}: inputs missing; skipped")
                    complete = False
                    continue
                complete &= _run(ws, kind, corpus, det, a.part, a.n_parts, a.jobs, a.seed, a.local,
                                 a.max_tasks) is not None
    if complete:
        return summary(ws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
