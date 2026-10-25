"""
Route 2: prevalence-gated adaptation (PGA), tested on data not used so far.

Rule (fixed before any ASVspoof 2021 DF score existed):
  1. Estimate the spoof share of the unlabelled traffic from the FROZEN score and the genuine
     reference (best-bin mixture-proportion estimate; accurate on In-the-Wild).
  2. If the estimate >= GATE (0.80): score traffic with the PU-adapted classifier (reference
     genuine vs traffic, cross-fitted logistic regression on the time-mean XLS-R features);
     otherwise keep the frozen score.
  3. Set the threshold with the fold-matched conformal genuine anchor (alpha = 5%).

Confirmatory endpoints (speaker-overlap reference, ASVspoof 2021 DF, 20 repetitions/cell):
  H1 gain     at the natural spoof share and at 0.90: mean EER(PGA) <= 0.85 x mean EER(frozen)
              and mean anchor HTER(PGA) < mean anchor HTER(frozen)
  H2 safety   at 0.05, 0.20, 0.50: mean EER(PGA) <= 1.02 x mean EER(frozen)
  H3 FRR      mean anchor FRR(PGA) <= 6% in every cell
  H4 shortcut mean shortcut AUC(PGA) <= 0.60 in every cell
  All pass -> method confirmed; otherwise -> route 1 (short paper, DF21 as second corpus).
The 160-d readout variant, the always-adapt variant and the speaker-disjoint mode are
reported as secondary. In-the-Wild results are discovery data, never confirmatory.

    python -m pga_g.pga prereg  --workspace W
    python -m pga_g.pga run     --workspace W --dataset df21 [--part 0 --n-parts 2]
    python -m pga_g.pga stagea  --workspace W --dataset df21 [--part 0 --n-parts 2]
    python -m pga_g.pga verdict --workspace W --dataset df21
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pga_g import pilot as P

from .chunked import log, run_chunked

PREREG = {
    "title": "Prevalence-gated adaptation (PGA) on ASVspoof 2021 DF",
    "fixed_on": "2026-10-21, before any ASVspoof 2021 DF score was computed",
    "discovery_data": "In-the-Wild (kill test, 21 Oct 2026): adaptation helped only at a 96% spoof share",
    "confirmatory_data": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF, all 80 shards, revision pinned in runs/df21_v1/config.json",
    "detector": "authors' SSL-AASIST LA checkpoint, frozen; score = bona - spoof logit",
    "gate": 0.80,
    "gate_estimator": "pilot_pga_g.estimate_bona_share on the frozen score (calibration half of the reference vs traffic)",
    "adapted_feature_primary": "ssl1024 (time-mean XLS-R features)",
    "adapted_feature_secondary": "emb160 (AASIST readout)",
    "classifier": "StandardScaler + LogisticRegression(C=0.1, class_weight=balanced), 5-fold cross-fitting, fold-matched calibration",
    "anchor": "fold-matched conformal genuine anchor, alpha = 0.05",
    "reference": "1000 genuine clips: 500 train, 500 calibration; speaker-overlap mode primary, speaker-disjoint secondary",
    "spoof_shares": "0.05, 0.20, 0.50, 0.80, 0.90 and the natural share of the corpus",
    "traffic_size": "min(12000, available)",
    "repetitions": 20,
    "H1": "gain: at natural share and 0.90, mean EER(PGA) <= 0.85 x mean EER(frozen) and mean anchor HTER(PGA) < frozen",
    "H2": "safety: at 0.05, 0.20, 0.50, mean EER(PGA) <= 1.02 x mean EER(frozen)",
    "H3": "FRR: mean anchor FRR(PGA) <= 0.06 in every cell",
    "H4": "shortcut: mean shortcut AUC(PGA) <= 0.60 in every cell",
    "decision": "H1-H4 all pass -> method confirmed; otherwise route 1",
}
GATE = 0.80
SHARES = (0.05, 0.20, 0.50, 0.80, 0.90)
N_REF, REPS = 1000, 20
_G: dict = {}


# ---------------------------------------------------------------- inputs
def load_inputs(ws: Path, dataset: str):
    if dataset.startswith("df21"):
        rd = ws / "runs" / f"{dataset}_v1"
        if not (rd / "_STAGE_SUCCESS.json").exists():
            raise SystemExit(f"{dataset} is not merged yet: run `python -m pga_g.df21 merge --wait`")
        df = pd.read_csv(ws / "results" / f"{dataset}_inputs" / "pilot.csv")
        ssl = np.load(rd / "ssl_mean1024.npy", mmap_mode="r")
        emb = np.load(rd / "emb160.npy", mmap_mode="r")
    elif dataset == "itw":
        pin = ws / "results" / "pilot_inputs"
        df = pd.read_csv(pin / "itw_pilot.csv")
        ssl = np.load(pin / "itw_ssl1024.npy", mmap_mode="r")
        emb = np.load(pin / "itw_emb160.npy", mmap_mode="r")
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    df = df.reset_index(drop=True)
    if len(df) != ssl.shape[0] or len(df) != emb.shape[0]:
        raise SystemExit("scores and features are not row-aligned")
    return df, ssl, emb


def natural_share(df) -> float:
    return round(float((df.label == 0).mean()), 3)


def _init(df, ssl, emb):
    _G.update(df=df, ssl=ssl, emb=emb)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


# ---------------------------------------------------------------- one cell
def _split(df, mode, rng, n_ref):
    bona = df.index[df.label == 1].to_numpy()
    if mode == "overlap":
        ref = rng.choice(bona, n_ref, replace=False)
        return ref, df.index.difference(ref).to_numpy()
    groups = df.loc[bona, "group"].unique()
    rng.shuffle(groups)
    chosen, ref = [], np.array([], int)
    for g in groups:
        chosen.append(g)
        ref = df.index[(df.label == 1) & df.group.isin(chosen)].to_numpy()
        if ref.size >= n_ref:
            break
    ref = rng.choice(ref, min(n_ref, ref.size), replace=False)
    return ref, df.index[~df.group.isin(chosen)].to_numpy()


def cell(mode: str, pi: float, rep: int, seed: int):
    df, ssl, emb = _G["df"], _G["ssl"], _G["emb"]
    rng = np.random.default_rng([seed, 23, 0 if mode == "overlap" else 1, int(round(pi * 1000)), rep])
    ref, pool = _split(df, mode, rng, N_REF)
    lab = df.label.to_numpy()
    b_pool, s_pool = pool[lab[pool] == 1], pool[lab[pool] == 0]
    n = min(12000, int(min(b_pool.size / max(1 - pi, 1e-9), s_pool.size / max(pi, 1e-9))))
    b_idx = rng.choice(b_pool, int(round(n * (1 - pi))), replace=False)
    s_idx = rng.choice(s_pool, int(round(n * pi)), replace=False)
    traffic = np.r_[b_idx, s_idx]
    nb = b_idx.size
    ref = rng.permutation(ref)
    half = ref.size // 2
    ref_tr, ref_cal = ref[:half], ref[half:]
    frozen = df.score.to_numpy()

    # gate: frozen-score prevalence estimate
    pi_hat = 1 - P.estimate_bona_share(frozen[ref_cal], frozen[traffic])
    gate_open = bool(pi_hat >= GATE)
    rows = []

    def record(variant, sc_ref, sb, ss, fold_info=None, extra=None):
        if fold_info is None:
            op = P.op_point(P.conformal_bona_threshold(sc_ref, P.ALPHA), sb, ss)
        else:
            fc, fu = fold_info
            op = P.foldwise_anchor_op(sc_ref, fc, sb, fu[:nb], ss, fu[nb:])
        rows.append({"mode": mode, "pi": pi, "rep": rep, "variant": variant, "n_traffic": int(n),
                     "eer": P.eer(sb, ss), "frr": op.frr, "far": op.far, "hter": op.hter,
                     "shortcut_auc": P.auc(sc_ref, sb), "pi_hat_frozen": pi_hat,
                     "gate_open": gate_open, **(extra or {})})

    record("frozen", frozen[ref_cal], frozen[b_idx], frozen[s_idx])
    for feat, X in (("ssl1024", ssl), ("emb160", emb)):
        Xt, Xc, Xu = (np.asarray(X[idx], dtype=np.float32) for idx in (ref_tr, ref_cal, traffic))
        s_cal, fold_c, s_unl, fold_u = P.pu_adapted_scores(Xt, Xc, Xu, rng)
        pi_hat_ad = 1 - P.estimate_bona_share(s_cal, s_unl)
        record(f"adapt_{feat}", s_cal, s_unl[:nb], s_unl[nb:], (fold_c, fold_u), {"pi_hat_adapted": pi_hat_ad})
        if gate_open:
            record(f"pga_{feat}", s_cal, s_unl[:nb], s_unl[nb:], (fold_c, fold_u))
        else:
            record(f"pga_{feat}", frozen[ref_cal], frozen[b_idx], frozen[s_idx])
    return rows


# ---------------------------------------------------------------- commands
def prereg(ws: Path) -> int:
    path = ws / "results" / "PREREGISTRATION_PGA.json"
    body = json.dumps(PREREG, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("sha256") != digest:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        log(f"pre-registration already on Drive (sha256 {digest[:16]}, written {old['written']})")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ws / "runs" / "df21_v1" / "parts"
    if parts.exists() and any(parts.glob("part_*.npz")):
        raise SystemExit("DF21 scoring has already started: a pre-registration written now would not be one")
    tmp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"preregistration": PREREG, "sha256": digest,
                               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    tmp.replace(path)
    log(f"pre-registration written: {path} (sha256 {digest[:16]})")
    return 0


def run(ws: Path, dataset: str, part: int, n_parts: int, jobs: int, seed: int, max_tasks=None) -> int:
    df, ssl, emb = load_inputs(ws, dataset)
    shares = sorted(set(SHARES) | {natural_share(df)})
    tasks = [(m, pi, r, seed) for m in ("overlap", "disjoint") for pi in shares for r in range(REPS)]
    out = ws / "results" / f"pga_{dataset}"
    res = run_chunked(cell, tasks, out, _init, (df, ssl, emb), jobs=jobs, part=part, n_parts=n_parts,
                      flush_every=10, label=f"PGA {dataset}", max_tasks=max_tasks,
                      config={"dataset": dataset, "shares": shares, "reps": REPS, "n_ref": N_REF,
                              "gate": GATE, "seed": seed, "n_rows": int(len(df))})
    if res is not None:
        summary = res.groupby(["mode", "pi", "variant"])[["eer", "frr", "far", "hter", "shortcut_auc",
                                                          "gate_open"]].mean()
        txt = (summary * 100).round(2).to_string()
        (out / "summary.txt").write_text(txt)
        print(txt)
    return 0


def stagea(ws: Path, dataset: str, part: int, n_parts: int, jobs: int, seed: int, reps: int) -> int:
    df, _, _ = load_inputs(ws, dataset)
    tau = json.loads((ws / "results" / "G0_reproduction.json").read_text())["tau_src"]
    tasks = [(n_ref, pi, r, seed, m) for m in ("overlap", "disjoint") for n_ref in P.REF_SIZES
             for pi in P.PRIORS for r in range(reps)]
    out = ws / "results" / f"stageA_{dataset}"
    res = run_chunked(P._cell_a, tasks, out, P._init_worker, (df[["score", "label", "group"]], None, tau),
                      jobs=jobs, part=part, n_parts=n_parts, flush_every=25, label=f"Stage A {dataset}",
                      config={"dataset": dataset, "reps": reps, "tau_src": tau, "seed": seed,
                              "pilot_version": P.__version__})
    if res is not None:
        txt = P.summarise(res.drop(columns=["task_key", "tau"]))
        (out / "summary.txt").write_text(txt)
        print(txt)
    return 0


def _ci_ratio(a: np.ndarray, b: np.ndarray, rng, n_boot=4000):
    """Percentile bootstrap (over paired repetitions) for mean(a) / mean(b)."""
    idx = rng.integers(0, a.size, (n_boot, a.size))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def verdict(ws: Path, dataset: str) -> int:
    out = ws / "results" / f"pga_{dataset}"
    res = pd.read_csv(out / "results.csv")
    rng = np.random.default_rng(0)
    shares = json.loads((out / "config.json").read_text())["shares"]
    extra = [x for x in shares if x not in SHARES]
    nat = extra[0] if extra else max(shares)
    ov = res[res["mode"] == "overlap"]

    def arr(pi, variant, col):
        return ov[(ov.pi == pi) & (ov.variant == variant)].sort_values("rep")[col].to_numpy()

    role = {"df21": "confirmatory", "itw": "discovery"}.get(dataset, "exploratory (robustness R1)")
    report = {"dataset": dataset, "natural_share": float(nat), "role": role}
    for feat, role in (("ssl1024", "primary"), ("emb160", "secondary")):
        v = f"pga_{feat}"
        h1 = {}
        for pi in sorted({nat, 0.90} if dataset.startswith("df21") else {0.90}):
            e_p, e_f = arr(pi, v, "eer"), arr(pi, "frozen", "eer")
            h_p, h_f = arr(pi, v, "hter"), arr(pi, "frozen", "hter")
            h1[str(pi)] = {"eer_ratio": float(e_p.mean() / e_f.mean()), "eer_ratio_ci95": _ci_ratio(e_p, e_f, rng),
                           "hter_pga": float(h_p.mean()), "hter_frozen": float(h_f.mean()),
                           "gate_open_rate": float(arr(pi, v, "gate_open").mean()),
                           "pass": bool(e_p.mean() <= 0.85 * e_f.mean() and h_p.mean() < h_f.mean())}
        h2 = {}
        for pi in (0.05, 0.20, 0.50):
            e_p, e_f = arr(pi, v, "eer"), arr(pi, "frozen", "eer")
            h2[str(pi)] = {"eer_ratio": float(e_p.mean() / e_f.mean()),
                           "gate_open_rate": float(arr(pi, v, "gate_open").mean()),
                           "pass": bool(e_p.mean() <= 1.02 * e_f.mean())}
        g = ov[ov.variant == v].groupby("pi")[["frr", "shortcut_auc"]].mean()
        h3 = {"max_mean_frr": float(g.frr.max()), "pass": bool((g.frr <= 0.06).all())}
        h4 = {"max_mean_shortcut_auc": float(g.shortcut_auc.max()), "pass": bool((g.shortcut_auc <= 0.60).all())}
        allp = all(x["pass"] for x in h1.values()) and all(x["pass"] for x in h2.values()) and h3["pass"] and h4["pass"]
        report[v] = {"role": role, "H1": h1, "H2": h2, "H3": h3, "H4": h4, "all_pass": allp}
    dj = res[res["mode"] == "disjoint"].groupby(["pi", "variant"])[["eer", "frr", "shortcut_auc"]].mean()
    report["disjoint_stress"] = {f"{pi}|{var}": {k: float(x) for k, x in row.items()} for (pi, var), row in dj.iterrows()}
    if dataset == "df21":
        report["decision"] = ("METHOD CONFIRMED (H1-H4 pass on the primary variant)"
                              if report["pga_ssl1024"]["all_pass"] else
                              "ROUTE 1 (pre-registered PGA hypothesis not confirmed; use DF21 as second corpus)")
    else:
        report["decision"] = f"{role}: no decision" + (f" (H1-H4 all pass: {report['pga_ssl1024']['all_pass']})")
    (ws / "results" / f"PGA_VERDICT_{dataset}.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    print(f"\nDECISION ({dataset}): {report['decision']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prereg", "run", "stagea", "verdict"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--dataset", choices=["df21", "itw", "df21trim40"], default="df21")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reps", type=int, default=50, help="Stage A repetitions per cell")
    ap.add_argument("--max-tasks", type=int, default=None, help="testing: stop after N tasks")
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    if a.cmd == "prereg":
        return prereg(ws)
    if a.cmd == "run":
        return run(ws, a.dataset, a.part, a.n_parts, a.jobs, a.seed, a.max_tasks)
    if a.cmd == "stagea":
        return stagea(ws, a.dataset, a.part, a.n_parts, a.jobs, a.seed, a.reps)
    return verdict(ws, a.dataset)


if __name__ == "__main__":
    sys.exit(main())
