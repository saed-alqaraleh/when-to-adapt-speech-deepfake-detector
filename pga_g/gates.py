"""
Kill-test gates, exactly as pre-registered in the proposal (21 Oct 2026).

G0 reproduction : ITW EER within 11.18% +/- 1.0 pt and ITW HTER at the transferred LA
                  EER threshold in [35%, 45%] (Zhou & Wang 2026: 11.18% and 39.49%).
G1 anchor       : mean FRR of PGA-P <= 6% at every prior and reference size, and its
                  HTER <= the reference-free GMM2 baseline + 0.5 pt at every prior and size
                  (tolerance fixed on 21 Oct 2026, before any real score was seen, so that
                  exact ties do not fail the gate).
G2 ranking      : speaker-overlap mode, PU-adapted EER <= 0.85 x frozen EER at priors
                  0.20 and 0.50 (primary embedding: the 160-d AASIST readout).
G3 shortcut     : shortcut AUC <= 0.60 in both reference modes (PU-adapted score).
G4 visibility   : |pi_hat - pi| smaller on the adapted than on the frozen score at 0.50 and
                  0.96 (overlap mode); informative, does not block.

    python -m pga_g.gates g0 --workspace WORKSPACE
    python -m pga_g.gates verdict --workspace WORKSPACE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import eer_and_threshold, hter

PUBLISHED = {"itw_eer": 0.1118, "itw_hter": 0.3949, "la_eer": 0.0021, "tau_la": 4.97}


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


def _scores(run_dir: Path) -> pd.DataFrame:
    if not (run_dir / "_STAGE_SUCCESS.json").exists():
        raise SystemExit(f"scoring not complete: {run_dir}/_STAGE_SUCCESS.json missing")
    return pd.read_csv(run_dir / "scores.csv")


def g0(ws: Path, score_ver: str, tau_override: float | None) -> int:
    runs = ws / "runs" / score_ver
    res = ws / "results"
    itw = _scores(runs / "itw")
    report = {"published": PUBLISHED}
    if tau_override is None:
        la = _scores(runs / "la19eval")
        la_eer, tau = eer_and_threshold(la.score[la.label == 1].values, la.score[la.label == 0].values)
        la_eer_off, _ = eer_and_threshold(la.score_official[la.label == 1].values,
                                          la.score_official[la.label == 0].values)
        report.update({"la_eer": la_eer, "la_eer_official_score": la_eer_off, "tau_src": tau,
                       "tau_source": "LA19 eval EER point (this run)"})
    else:
        tau = float(tau_override)
        report.update({"tau_src": tau, "tau_source": "override (no LA scores)"})
    b, s = itw.score[itw.label == 1].values, itw.score[itw.label == 0].values
    itw_eer, _ = eer_and_threshold(b, s)
    op = hter(b, s, tau)
    op_pub = hter(b, s, PUBLISHED["tau_la"])
    report.update({"itw_eer": itw_eer, "itw_at_tau_src": op, "itw_at_published_tau_4.97": op_pub,
                   "itw_eer_official_score": eer_and_threshold(
                       itw.score_official[itw.label == 1].values,
                       itw.score_official[itw.label == 0].values)[0],
                   "n_itw": int(len(itw)), "n_itw_bona": int(b.size), "n_itw_spoof": int(s.size)})
    c_eer = abs(itw_eer - PUBLISHED["itw_eer"]) <= 0.010
    c_hter = 0.35 <= op["hter"] <= 0.45
    report["G0"] = {"eer_within_1pt": bool(c_eer), "hter_in_35_45": bool(c_hter),
                    "pass": bool(c_eer and c_hter)}
    _dump(res / "G0_reproduction.json", report)

    inputs = res / "pilot_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    pilot = pd.DataFrame({"utt_id": itw.utt_id, "score": itw.score, "label": itw.label,
                          "group": itw.speaker})
    pilot.to_csv(inputs / "itw_pilot.csv", index=False)
    for src, dst in [("emb160.npy", "itw_emb160.npy"), ("ssl_mean1024.npy", "itw_ssl1024.npy")]:
        np.save(inputs / dst, np.load(runs / "itw" / src).astype(np.float32))
    _dump(inputs / "tau_src.json", {"tau_src": tau, "source": report["tau_source"]})

    pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    print("G0 reproduction check")
    if "la_eer" in report:
        print(f"  LA19 eval EER           {pct(report['la_eer'])}   (published 0.21-0.22%)   tau_src = {tau:.3f}")
    print(f"  ITW EER                 {pct(itw_eer)}   (published 11.18%)")
    print(f"  ITW HTER at tau_src     {pct(op['hter'])}   FRR {pct(op['frr'])}  FAR {pct(op['far'])}   (published 39.49%, FRR 78.7%)")
    print(f"  ITW HTER at tau = 4.97  {pct(op_pub['hter'])}")
    print(f"  G0: {'PASS' if report['G0']['pass'] else 'FAIL'}  -> {res / 'G0_reproduction.json'}")
    return 0


def verdict(ws: Path) -> int:
    res = ws / "results"
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    g0r = json.loads((res / "G0_reproduction.json").read_text())
    out["G0"] = g0r["G0"]

    fa = res / "stageA_v2" / "results.csv"
    a = pd.read_csv(fa if fa.exists() else res / "stageA" / "results.csv")
    if "ref_mode" not in a.columns:
        a["ref_mode"] = "disjoint"                     # v2.0.0 Stage A ran the disjoint mode only

    def g1_for(df):
        m = df.groupby(["n_ref", "pi", "method"])[["frr", "hter"]].mean().unstack("method")
        gap = m[("hter", "PGA-P")] - m[("hter", "GMM2")]
        frr_ok = bool((m[("frr", "PGA-P")] <= 0.06).all())
        vs = bool((gap <= 0.005).all())
        return {"max_mean_frr": float(m[("frr", "PGA-P")].max()), "frr_ok": frr_ok,
                "max_hter_gap_vs_gmm2": float(gap.max()), "not_worse_than_gmm2_plus_0.5pt": vs,
                "reps_per_cell": int(df.groupby(["n_ref", "pi", "method"]).size().min()),
                "pass": frr_ok and vs}

    if "overlap" in set(a.ref_mode):
        out["G1"] = {**g1_for(a[a.ref_mode == "overlap"]), "reference_mode": "overlap (primary, pre-registered)"}
    else:
        out["G1"] = {**g1_for(a), "reference_mode": "disjoint only: primary overlap mode not run yet",
                     "pass": False, "frr_ok": False}
    if "disjoint" in set(a.ref_mode):
        out["G1_stress_disjoint"] = g1_for(a[a.ref_mode == "disjoint"])

    for emb in ["emb160", "ssl1024"]:
        f = res / f"stageB_{emb}" / "results.csv"
        if not f.exists():
            continue
        b = pd.read_csv(f)
        mb = b.groupby(["ref_mode", "pi", "score"])[["eer", "pi_hat", "shortcut_auc"]].mean()
        ov = mb.loc["overlap"]
        ratios = {p: float(ov.loc[(p, "pu_adapted"), "eer"] / ov.loc[(p, "frozen"), "eer"]) for p in (0.2, 0.5)}
        g2 = all(r <= 0.85 for r in ratios.values())
        auc_max = float(mb.xs("pu_adapted", level="score")["shortcut_auc"].max())
        g3 = auc_max <= 0.60
        err = {p: {s: float(abs(ov.loc[(p, s), "pi_hat"] - p)) for s in ("frozen", "pu_adapted")} for p in (0.5, 0.96)}
        g4 = all(err[p]["pu_adapted"] < err[p]["frozen"] for p in err)
        out[f"stageB_{emb}"] = {"G2": {"eer_ratio_adapted_over_frozen": ratios, "pass": g2},
                                "G3": {"max_shortcut_auc": auc_max, "pass": g3},
                                "G4": {"prior_abs_error": err, "pass": g4},
                                "role": "primary (pre-registered)" if emb == "emb160" else "secondary (exploratory)"}

    prim = out.get("stageB_emb160")
    if not out["G0"]["pass"]:
        decision = "FIX PIPELINE (G0 failed: nothing else counts yet)"
    elif prim is None:
        decision = "INCOMPLETE (Stage B on emb160 missing)"
    elif not out["G1"]["frr_ok"]:
        decision = "INVESTIGATE (anchor FRR above 6%: the bound is distribution-free, so suspect a bug or too few repetitions)"
    elif not out["G1"]["pass"]:
        decision = "REVIEW (anchor FRR holds but HTER exceeds GMM2 + 0.5 pt at some prior; report it, then read G2/G3)"
    elif prim["G2"]["pass"] and prim["G3"]["pass"]:
        decision = "GO"
    else:
        decision = "PIVOT (anchor holds; ranking repair or shortcut gate failed)"
    out["decision"] = decision
    _dump(res / "GATES.json", out)
    print(json.dumps(out, indent=2, default=float))
    print(f"\nDECISION: {decision}\n-> {res / 'GATES.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["g0", "verdict"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--score-version", default="score_v1")
    ap.add_argument("--tau-src", type=float, default=None,
                    help="use this source threshold instead of computing it from LA19 scores")
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    return g0(ws, a.score_version, a.tau_src) if a.cmd == "g0" else verdict(ws)


if __name__ == "__main__":
    sys.exit(main())
