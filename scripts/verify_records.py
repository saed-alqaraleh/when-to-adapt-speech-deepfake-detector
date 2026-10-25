"""Recompute every confirmatory verdict from the per-repetition result files and compare with the stored
verdict files (independent of the verdict code paths: means and ratios are recomputed here).

usage: python scripts/verify_records.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R1 = A5 = V3 = Path(__file__).resolve().parents[1]
out, worst = [], 0.0


def cmp(name, mine, stored, tol=1e-9):
    global worst
    d = abs(float(mine) - float(stored))
    worst = max(worst, d)
    out.append((name, float(mine), float(stored), d <= tol))


def cell(df, mode, pi, var):
    x = df[(df["mode"] == mode) & np.isclose(df.pi, pi) & (df.variant == var)].sort_values("rep")
    assert len(x) == 20, (mode, pi, var, len(x))
    return x


# ---- ASVspoof 2021 DF and In-the-Wild, pre-registered PGA (XLS-R features): PGA = adapt where the gate opened
for ds, nat in (("df21", 0.963), ("itw", 0.372)):
    r = pd.read_csv(R1 / "results" / f"robust_{ds}" / "results.csv").rename(columns={"method": "variant"})
    fz = r[r.variant == "frozen"]
    ad = r[r.variant == "adapt"].merge(fz[["mode", "pi", "rep", "gate_open"]], on=["mode", "pi", "rep"],
                                       suffixes=("", "_fz"))
    pga = fz.copy()
    pga = pga.set_index(["mode", "pi", "rep"])
    a = ad.set_index(["mode", "pi", "rep"])
    for c in ("eer", "frr", "far", "hter", "shortcut_auc"):
        pga.loc[a.index[a.gate_open_fz.values], c] = a.loc[a.gate_open_fz.values, c]
    pga = pga.reset_index().assign(variant="pga")
    allr = pd.concat([fz, pga])
    V = json.loads((R1 / "results" / f"PGA_VERDICT_{ds}.json").read_text())["pga_ssl1024"]
    if ds == "df21":
        for p in (0.9, 0.963):
            g, f = cell(allr, "overlap", p, "pga"), cell(allr, "overlap", p, "frozen")
            cmp(f"df21 H1 EER ratio {p}", g.eer.mean() / f.eer.mean(), V["H1"][f"{p:g}"]["eer_ratio"])
            cmp(f"df21 H1 HTER PGA {p}", g.hter.mean(), V["H1"][f"{p:g}"]["hter_pga"])
            cmp(f"df21 H1 HTER frozen {p}", f.hter.mean(), V["H1"][f"{p:g}"]["hter_frozen"])
        for p in (0.05, 0.2, 0.5):
            cmp(f"df21 H2 EER ratio {p}", cell(allr, "overlap", p, "pga").eer.mean()
                / cell(allr, "overlap", p, "frozen").eer.mean(), V["H2"][f"{p:g}"]["eer_ratio"])
        mf = max(cell(allr, "overlap", p, "pga").frr.mean() for p in sorted(allr.pi.unique()))
        cmp("df21 H3 max mean FRR", mf, V["H3"]["max_mean_frr"])
        ma = max(cell(allr, "overlap", p, "pga").shortcut_auc.mean() for p in sorted(allr.pi.unique()))
        cmp("df21 H4 max mean shortcut AUC", ma, V["H4"]["max_mean_shortcut_auc"])
    else:
        g, f = cell(allr, "overlap", 0.9, "pga"), cell(allr, "overlap", 0.9, "frozen")
        cmp("itw PGA EER 0.90 (rounded percentage)", round(100 * g.eer.mean(), 1), 5.2)
        cmp("itw frozen EER 0.90 (rounded percentage)", round(100 * f.eer.mean(), 1), 11.3)

# ---- ASVspoof 5, pre-registered PGA-G
m = pd.read_csv(A5 / "results" / "asv5_main" / "results.csv")
V = json.loads((A5 / "results" / "ASV5_VERDICT.json").read_text())["primary_pgag_ssl1024"]
var = "pgag_ssl1024" if "pgag_ssl1024" in set(m.variant) else [v for v in m.variant.unique() if v.startswith("pgag")][0]
for p in (0.9, 0.96):
    g, f = cell(m, "overlap", p, var), cell(m, "overlap", p, "frozen")
    c = V["H1_gain"]["cells"][f"{p:g}"]
    cmp(f"asv5 H1 EER ratio {p}", g.eer.mean() / f.eer.mean(), c["eer_ratio"])
    cmp(f"asv5 H1 HTER {p}", g.hter.mean(), c["hter"])
for p in sorted(m.pi.unique()):
    g, f = cell(m, "disjoint", p, var), cell(m, "disjoint", p, "frozen")
    c = V["H3_no_harm_disjoint"]["cells"][f"{p:g}"]
    cmp(f"asv5 H3 FRR disjoint {p}", g.frr.mean(), c["frr"])
cmp("asv5 H4 max FRR", max(cell(m, "overlap", p, var).frr.mean() for p in sorted(m.pi.unique())),
    V["H4_genuine_rejection"]["max_mean_frr"])

# ---- third test: every detector and corpus
VV = json.loads((V3 / "results" / "V3_VERDICT.json").read_text())["main"]
for det in ("sslaasist", "mamba", "aasist"):
    for c in ("df21", "asv5"):
        m = pd.read_csv(V3 / "results" / "v3" / f"main_{c}_{det}" / "results.csv")
        h = VV[f"{det}|{c}"]["pgag_primary"]
        for p, cc in h["H1_gain"]["cells"].items():
            g, f = cell(m, "overlap", float(p), "pgag_primary"), cell(m, "overlap", float(p), "frozen")
            cmp(f"v3 {det} {c} H1 EER ratio {p}", g.eer.mean() / f.eer.mean(), cc["eer_ratio"])
        for p, cc in h["H3_no_harm_disjoint"]["cells"].items():
            g = cell(m, "disjoint", float(p), "pgag_primary")
            cmp(f"v3 {det} {c} H3 FRR {p}", g.frr.mean(), cc["frr"])
        cmp(f"v3 {det} {c} H4 max FRR", max(cell(m, "overlap", p, "pgag_primary").frr.mean()
                                             for p in sorted(m.pi.unique())), h["H4_genuine_rejection"]["max_mean_frr"])


# ---- operating-point study (Operating-point record) and first confirmatory runs, when their per-repetition files are present
sa = R1 / "results" / "stageA_df21" / "results.csv"
if sa.exists():
    a = pd.read_csv(sa)
    a = a[a.n_ref == 1000].groupby(["ref_mode", "method", "pi"])[["hter", "frr"]].mean()
    table1 = {("overlap", "SRC"): [(16.7, 33.2)] * 4, ("overlap", "GMM2"): [(19.3, 38.6), (18.9, 37.8), (17.8, 35.4), (7.9, 2.2)],
              ("overlap", "BONA"): [(7.6, 5.0), (7.7, 5.0), (7.8, 5.1), (7.6, 4.9)],
              ("overlap", "PGA-P"): [(7.6, 5.0), (7.8, 4.8), (7.8, 4.6), (7.6, 4.3)],
              ("overlap", "ORACLE"): [(8.0, None), (8.3, None), (8.3, None), (8.2, None)],
              ("disjoint", "SRC"): [(16.6, 33.1), (16.7, 33.3), (16.6, 33.1), (16.8, 33.4)],
              ("disjoint", "GMM2"): [(19.3, 38.5), (19.0, 38.0), (17.7, 35.4), (8.1, 2.4)],
              ("disjoint", "BONA"): [(8.0, 5.8), (8.0, 6.1), (8.0, 6.2), (8.1, 5.9)],
              ("disjoint", "PGA-P"): [(9.4, 5.1), (8.4, 5.6), (8.0, 5.6), (7.9, 4.8)],
              ("disjoint", "ORACLE"): [(8.3, None), (8.3, None), (8.3, None), (8.4, None)]}
    for (mode, meth), vals in table1.items():
        for p_, (h, f) in zip((0.05, 0.2, 0.5, 0.96), vals):
            r = a.loc[(mode, meth, p_)]
            out.append((f"Operating-point record {mode} {meth} {p_} HTER", round(100 * r.hter, 1), h, abs(round(100 * r.hter, 1) - h) < 1e-9))
            if f is not None:
                out.append((f"Operating-point record {mode} {meth} {p_} FRR", round(100 * r.frr, 1), f, abs(round(100 * r.frr, 1) - f) < 1e-9))
pg = R1 / "results" / "pga_df21" / "results.csv"
if pg.exists():
    d = pd.read_csv(pg)
    V = json.loads((R1 / "results" / "PGA_VERDICT_df21.json").read_text())
    for var in ("pga_ssl1024", "pga_emb160"):
        for p_ in (0.9, 0.963):
            g, f = cell(d, "overlap", p_, var), cell(d, "overlap", p_, "frozen")
            cmp(f"pga_df21 {var} H1 EER ratio {p_}", g.eer.mean() / f.eer.mean(), V[var]["H1"][f"{p_:g}"]["eer_ratio"])
        cmp(f"pga_df21 {var} H3 max FRR", max(cell(d, "overlap", p_, var).frr.mean() for p_ in sorted(d.pi.unique())),
            V[var]["H3"]["max_mean_frr"])

n_ok = sum(o[3] for o in out)
for o in out:
    print(("OK  " if o[3] else "DIFF"), f"{o[0]:<45} recomputed {o[1]:.6f}  stored {o[2]:.6f}")
print(f"\n{n_ok}/{len(out)} agree; largest absolute difference {worst:.2e}")

if n_ok != len(out):
    raise SystemExit(f"Numerical verification failed: {n_ok}/{len(out)}")
