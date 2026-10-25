"""Recompute the source-grouped evaluation summary from frozen per-repetition records.

This script does not regenerate manuscript tables or figures.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DETECTORS = ("sslaasist", "mamba", "aasist")
CORPORA = ("df21", "asv5")
H1 = {"df21": (0.9, 0.963), "asv5": (0.9, 0.96)}

def pct(x):
    return 100 * float(x)

def calculate():
    S = json.loads((RESULTS / "V5_RESULTS.json").read_text())
    PRE = json.loads((ROOT / "preregistration/PREREGISTRATION_V5.json").read_text())
    N = {"prereg": {"sha256": PRE["sha256"], "written": PRE["written"]},
         "siblings_df21": S["siblings_df21"], "main": {}}

    for d in DETECTORS:
        for c in CORPORA:
            key = f"{d}|{c}"
            m = S["main"][key]
            r = pd.read_csv(RESULTS / "v5" / f"main_{c}_{d}" / "results.csv")
            ov = r[r["mode"] == "overlap"]
            g = ov.groupby(["pi", "design", "method"])[["eer", "hter", "frr", "far", "h_eer", "h_frr", "adapted"]].mean()
            h1 = {}
            for p in H1[c]:
                h1[f"{p:g}"] = {des: {
                    "eer": pct(g.loc[(p, des, "pgag"), "eer"]),
                    "hter": pct(g.loc[(p, des, "pgag"), "hter"]),
                    "frr": pct(g.loc[(p, des, "pgag"), "frr"]),
                    "adapted": float(g.loc[(p, des, "pgag"), "adapted"]),
                    "h_eer": pct(g.loc[(p, des, "pgag"), "h_eer"]),
                    "h_eer_frozen": pct(g.loc[(p, des, "frozen"), "h_eer"]),
                } for des in ("A", "B", "C")}
                for des in ("A", "B", "C"):
                    h1[f"{p:g}"][des]["h_ratio"] = (
                        h1[f"{p:g}"][des]["h_eer"] /
                        h1[f"{p:g}"][des]["h_eer_frozen"]
                    )
                h1[f"{p:g}"]["dEER_B_A"] = m["paired"][f"overlap|{p:g}"]["B-A"]["eer"]
            hfrr_c = ov[(ov.design == "C") & (ov.method == "pgag")].groupby("pi").h_frr.mean()
            expo = r[r.design == "A"].drop_duplicates(["mode", "pi", "rep"]).groupby("mode")[
                ["x_cal_with_train_version", "x_traffic_bona_with_ref_version", "x_heldout_bona_with_version"]
            ].mean()
            N["main"][key] = {
                "outcomes": m["outcomes"], "robust": m["robust_to_source_leakage"],
                "worst_B_minus_A_pt": {a: pct(b) for a, b in m["worst_B_minus_A"].items()},
                "reproduces_v3": m["reproduces_v3"], "h1": h1,
                "heldout_frr_C_matched": {f"{p:g}": pct(v) for p, v in hfrr_c.items()},
                "exposure_A": {mode: {kk: float(vv) for kk, vv in row.items()}
                               for mode, row in expo.iterrows()},
            }

    ssl = [f"{d}|{c}" for d in ("sslaasist", "mamba") for c in CORPORA]
    allc = [v for k in ssl for v in N["main"][k]["heldout_frr_C_matched"].values()]
    N["ssl_summary"] = {
        "h1_dEER_B_A_range_pt": [
            min(pct(N["main"][k]["h1"][p]["dEER_B_A"]["diff"])
                for k in ssl for p in N["main"][k]["h1"]),
            max(pct(N["main"][k]["h1"][p]["dEER_B_A"]["diff"])
                for k in ssl for p in N["main"][k]["h1"]),
        ],
        "max_abs_dFRR_pt": max(N["main"][k]["worst_B_minus_A_pt"]["frr"] for k in ssl),
        "heldout_frr_C_matched_range": [min(allc), max(allc)],
        "heldout_frr_C_matched_mean": float(np.mean(allc)),
        "outcomes_unchanged": all(
            N["main"][k]["outcomes"]["A"] == N["main"][k]["outcomes"][des]
            for k in ssl for des in ("B", "C")),
    }

    ratios = {}
    for d in DETECTORS:
        for c in CORPORA:
            r = pd.read_csv(RESULTS / "v5" / f"main_{c}_{d}" / "results.csv")
            ov = r[r["mode"] == "overlap"]
            g = ov.groupby(["pi", "design", "method"])[["eer", "h_eer", "adapted"]].mean()
            for p in H1[c]:
                q = {}
                for des in ("A", "B"):
                    q[des] = {
                        "traffic": float(g.loc[(p, des, "pgag"), "eer"] / g.loc[(p, des, "frozen"), "eer"]),
                        "heldout": float(g.loc[(p, des, "pgag"), "h_eer"] / g.loc[(p, des, "frozen"), "h_eer"]),
                        "h_eer_frozen": pct(g.loc[(p, des, "frozen"), "h_eer"]),
                        "h_eer_pgag": pct(g.loc[(p, des, "pgag"), "h_eer"]),
                        "adapted": float(g.loc[(p, des, "pgag"), "adapted"]),
                    }
                ratios[f"{d}|{c}|{p:g}"] = q
    sslr = {k: v for k, v in ratios.items() if not k.startswith("aasist")}
    N["heldout_vs_traffic"] = {
        "cells": ratios,
        "ssl_max_heldout_minus_traffic_B": max(v["B"]["heldout"] - v["B"]["traffic"] for v in sslr.values()),
        "ssl_min_heldout_minus_traffic_B": min(v["B"]["heldout"] - v["B"]["traffic"] for v in sslr.values()),
    }
    return N

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, help="write regenerated JSON to a separate file")
    args = ap.parse_args()
    body = json.dumps(calculate(), indent=1, default=float).encode()
    archived = (RESULTS / "verification/numbers_v5.json").read_bytes()
    if body != archived:
        raise SystemExit("Frozen summary regeneration does not match stored bytes")
    print(f"Source-grouped summary: byte-identical ({len(body)} bytes)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(body)

if __name__ == "__main__":
    main()
