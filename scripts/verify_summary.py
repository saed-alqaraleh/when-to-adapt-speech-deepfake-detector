"""Verify stored summary metrics against frozen per-repetition records."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1] / "results"
out = []
def chk(name, value, shown, tol):
    ok = abs(value - shown) <= tol
    out.append((ok, name, value, shown))
# operating-point study (Table II): anchor FRR ranges, matched and disjoint, all shares, n_ref 1000
a = pd.read_csv(R / "stageA_df21" / "results.csv")
a = a[(a.n_ref == 1000) & (a.method == "BONA")].groupby(["ref_mode", "pi"]).frr.mean() * 100
chk("anchor FRR matched min", a["overlap"].min(), 4.9, 0.05); chk("anchor FRR matched max", a["overlap"].max(), 5.1, 0.05)
chk("anchor FRR disjoint min", a["disjoint"].min(), 5.7, 0.05); chk("anchor FRR disjoint max", a["disjoint"].max(), 6.2, 0.05)
# pre-registered tests: H1 reductions (SSL), H2 ratios, matched max mean FRR
V3 = json.load(open(R / "V3_VERDICT.json"))["main"]; A5 = json.load(open(R / "ASV5_VERDICT.json"))["primary_pgag_ssl1024"]
P1 = json.load(open(R / "PGA_VERDICT_df21.json"))["pga_ssl1024"]
h1 = [c["eer_ratio"] for k in ("sslaasist|df21", "sslaasist|asv5", "mamba|df21", "mamba|asv5")
      for c in V3[k]["pgag_primary"]["H1_gain"]["cells"].values()]
h1 += [c["eer_ratio"] for c in A5["H1_gain"]["cells"].values()] + [c["eer_ratio"] for c in P1["H1"].values()]
chk("EER reduction min % (SSL, H1)", 100 * (1 - max(h1)), 43, 0.5); chk("EER reduction max % (SSL, H1)", 100 * (1 - min(h1)), 68, 0.5)
h2 = [c["eer_ratio"] for k in V3 for c in V3[k]["pgag_primary"]["H2_no_harm_low_share"]["cells"].values()]
h2 += [c["eer_ratio"] for c in A5["H2_no_harm_low_share"]["cells"].values()] + [c["eer_ratio"] for c in P1["H2"].values()]
chk("H2 EER ratio max (all tests, all detectors)", max(h2), 1.0, 1e-9); chk("H2 EER ratio min", min(h2), 1.0, 1e-9)
fr = [V3[k]["pgag_primary"]["H4_genuine_rejection"]["max_mean_frr"] for k in V3] + \
     [A5["H4_genuine_rejection"]["max_mean_frr"], P1["H3"]["max_mean_frr"]]
chk("max mean FRR matched, three tests (%)", 100 * max(fr), 5.6, 0.05)
chk("corpus trials total (millions)", (611829 + 680774) / 1e6, 1.29, 0.005)
# v5: held-out vs traffic, design C held-out FRR
N = json.load(open(Path(__file__).resolve().parents[1] / "results" / "verification" / "numbers_v5.json"))
chk("held-out minus traffic min (B, SSL)", N["heldout_vs_traffic"]["ssl_min_heldout_minus_traffic_B"], -0.020, 0.0005)
chk("held-out minus traffic max (B, SSL)", N["heldout_vs_traffic"]["ssl_max_heldout_minus_traffic_B"], 0.004, 0.0005)
for k, s in (("sslaasist|asv5|0.9", 0.554), ("sslaasist|asv5|0.96", 0.449), ("mamba|asv5|0.9", 0.401), ("mamba|asv5|0.96", 0.321)):
    chk(f"held-out ratio B {k}", N["heldout_vs_traffic"]["cells"][k]["B"]["heldout"], s, 0.0005)
asv5c = [v for d in ("sslaasist", "mamba") for v in N["main"][f"{d}|asv5"]["heldout_frr_C_matched"].values()]
allc = [v for d in ("sslaasist", "mamba") for c in ("df21", "asv5") for v in N["main"][f"{d}|{c}"]["heldout_frr_C_matched"].values()]
chk("held-out FRR C mean, ASVspoof 5 (abstract)", float(np.mean(asv5c)), 4.7, 0.05)
chk("held-out FRR C mean, both corpora (V-H, conclusion)", float(np.mean(allc)), 4.7, 0.05)
chk("split bound 12/251 (%)", 100 * 12 / 251, 4.8, 0.05)
# Additional stored effect checks: unconditional adaptation at 5% spoofs; speaker-mismatch FRR (reported values)
r = pd.read_csv(R / "robust_df21" / "results.csv")
g = r[(r["mode"] == "overlap") & np.isclose(r.pi, 0.05)].groupby("method").eer.mean() * 100
chk("unconditional EER at 5% spoofs / frozen (ratio > 2)", g["adapt"] / g["frozen"], 2.18, 0.02)
for ok, n, v, s in out:
    print(("OK  " if ok else "DIFF"), f"{n:<52} computed {v:.4f}  shown {s}")
print(f"{sum(o[0] for o in out)}/{len(out)} agree")

if not all(o[0] for o in out):
    raise SystemExit("Numerical summary verification failed")
