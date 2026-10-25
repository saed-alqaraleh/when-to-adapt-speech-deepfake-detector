"""
Pre-registered confirmatory test of PGA-G (prevalence-gated adaptation with consistency guard)
on the ASVspoof 5 evaluation set (track 1).

The method is fixed exactly as reported for ASVspoof 2021 DF: frozen SSL-AASIST, positive-
unlabelled logistic regression on time-averaged XLS-R features (C 0.1, balanced, 5-fold
cross-fitting), fold-matched conformal anchor (alpha 0.05), gate 0.80 on the frozen best-bin
spoof-share estimate, guard rho 0.5 on the ratio of adapted to frozen genuine-share estimates.
The pre-registration (results/PREREGISTRATION_ASV5.json) is written by the scoring workers before
the first ASVspoof 5 score exists and is never overwritten.

New in the protocol: ASVspoof 5 contains coded copies of the same source utterance (CODEC_SEED),
so references and traffic are separated by source utterance as well as by trial.

    python -u -m pga_g.confirm5 prereg    --workspace W
    python -u -m pga_g.confirm5 all       --workspace W [--part k --n-parts N] [--local L]
    python -u -m pga_g.confirm5 verdict   --workspace W
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

from . import robust as R
from .chunked import log, run_chunked

GATE, RHO, N_REF, REPS = 0.80, 0.5, 1000, 20
SHARES = (0.05, 0.20, 0.50, 0.80, 0.90, 0.96)
SPK_K, SPK_NREF, SPK_SHARE, SPK_REPS = (10, 20, 40, 80, 160), 300, 0.90, 20
BREAK_SHARES, BREAK_REPS = (0.90, 0.96), 10
MALAFIDE, MALACOPULA = ("A18", "A20", "A23", "A30"), ("A27", "A30", "A31", "A32")

PREREG = {
    "title": "Confirmatory test of prevalence-gated adaptation with consistency guard (PGA-G) on ASVspoof 5",
    "fixed_on": "2026-10-22, before any ASVspoof 5 score was computed",
    "prior_evidence": ("ASVspoof 2021 DF (confirmatory for the gate, PGA_VERDICT_df21.json) and exploratory robustness "
                       "analyses on ASVspoof 2021 DF and In-the-Wild (EXPLORATORY_PLAN_R1.json) in which the guard was "
                       "specified; this test is the first on data untouched by the study"),
    "data": ("ASVspoof 5 evaluation set, track 1 protocol ASVspoof5.eval.track_1.tsv, all ten flac_E tars "
             "(huggingface.co/datasets/jungjee/asvspoof5, mirror of Zenodo 10.5281/zenodo.14498691), revision pinned in "
             "runs/asv5_v1/config.json"),
    "detector": "authors' SSL-AASIST ASVspoof 2019 LA checkpoint, frozen; score = genuine minus spoof logit; first 64,600 samples",
    "precision": "fastest of bf16/fp16, TF32, fp32 whose score differs from strict fp32 by at most 0.05 logit on 128 clips",
    "method": {
        "primary_features": "time-mean XLS-R (1,024-d); secondary: AASIST readout (160-d)",
        "classifier": "StandardScaler + LogisticRegression(C=0.1, class_weight=balanced), reference-train=1 vs traffic=0, 5-fold cross-fitting",
        "anchor": "fold-matched conformal anchor on the calibration reference, alpha = 0.05",
        "gate": "1 - best-bin genuine-share estimate (frozen score, calibration reference vs traffic) >= 0.80",
        "guard": "best-bin genuine-share estimate of the adapted score / that of the frozen score >= 0.5",
        "decision": "adapt only if gate and guard both pass; otherwise frozen score with its conformal anchor",
    },
    "design": {
        "reference": "1,000 genuine recordings, random halves 500 train / 500 calibration",
        "matched_mode": "reference = random genuine recordings pooled over all speakers",
        "disjoint_mode": "reference = genuine recordings of a random speaker subset (speakers added until 1,000 are available); traffic excludes every trial of those speakers",
        "leakage_control": "traffic excludes every trial sharing a source utterance (CODEC_SEED, or the file itself if uncoded) with a reference recording",
        "traffic": "min(12,000, available) trials per repetition at spoof shares 0.05, 0.20, 0.50, 0.80, 0.90, 0.96 and the natural share",
        "repetitions": 20,
        "seeds": "derived from (2026, mode, share, repetition)",
    },
    "hypotheses": {
        "H1_gain": "matched references, spoof shares 0.90 and 0.96: mean EER(PGA-G) <= 0.85 x mean EER(frozen) and mean anchor HTER(PGA-G) < frozen",
        "H2_no_harm_low_share": "matched references, spoof shares 0.05, 0.20, 0.50: mean EER(PGA-G) <= 1.02 x mean EER(frozen)",
        "H3_no_harm_disjoint": "disjoint references, every spoof share: mean EER(PGA-G) <= 1.02 x mean EER(frozen) and mean anchor FRR(PGA-G) <= mean anchor FRR(frozen) + 0.01",
        "H4_genuine_rejection": "matched references, every spoof share: mean anchor FRR(PGA-G) <= 0.06",
        "H5_shortcut": "matched references, every spoof share: mean shortcut AUC of the scores PGA-G uses <= 0.60",
        "decision": "H1-H5 all pass -> PGA-G confirmed on ASVspoof 5; otherwise not confirmed (reported in full either way)",
    },
    "secondary_analyses_no_decision": [
        "natural spoof share (about 0.80, at the gate threshold), PGA without guard, readout features",
        "guard versus number of reference speakers: disjoint references of exactly 10, 20, 40, 80, 160 speakers and a matched control, 300 recordings, spoof share 0.90, 20 repetitions",
        "held-out breakdown (trials sharing no source utterance with reference or traffic), matched, shares 0.90 and 0.96, 10 repetitions: EER and anchor FRR/FAR by codec and by attack, with Malafide (A18, A20, A23, A30) and Malacopula (A27, A30, A31, A32) attacks reported separately",
    ],
    "statistics": "means over repetitions; 95% percentile bootstrap intervals over repetitions (4,000 resamples, paired for ratios)",
}

_G: dict = {}


# ---------------------------------------------------------------- pre-registration
def prereg(ws: Path) -> str:
    path = ws / "results" / "PREREGISTRATION_ASV5.json"
    body = json.dumps(PREREG, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("sha256") != digest:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        return digest
    parts = ws / "runs" / "asv5_v1" / "parts"
    if parts.exists() and any(parts.glob("part_*.npz")):
        raise SystemExit("ASVspoof 5 scoring has already started: a pre-registration written now would not be one")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"preregistration": PREREG, "sha256": digest,
                               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    tmp.replace(path)
    log(f"pre-registration written: {path} (sha256 {digest[:16]})")
    return digest


# ---------------------------------------------------------------- inputs
def load(ws: Path, local=None):
    rd = ws / "runs" / "asv5_v1"
    if not (rd / "_STAGE_SUCCESS.json").exists():
        raise SystemExit("ASVspoof 5 is not merged yet: run `python -m pga_g.asv5 merge --wait`")
    text = ["utt_id", "speaker", "codec", "attack", "src_group"]
    df = pd.read_csv(rd / "scores.csv", usecols=text + ["label", "score"], dtype={c: str for c in text},
                     keep_default_na=False)
    ssl = np.load(R._cache(rd / "ssl_mean1024.npy", local), mmap_mode="r")
    emb = np.load(R._cache(rd / "emb160.npy", local), mmap_mode="r")
    if not (len(df) == ssl.shape[0] == emb.shape[0]):
        raise SystemExit("scores and features are not row-aligned")
    return df, ssl, emb


def _init(df, ssl, emb):
    _G.clear()
    _G.update(df=df, ssl=ssl, emb=emb, lab=df.label.to_numpy(), score=df.score.to_numpy(np.float64),
              spk=pd.factorize(df.speaker)[0], grp=pd.factorize(df.src_group)[0])
    _G["bona"] = np.flatnonzero(_G["lab"] == 1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                # noqa: BLE001
        pass


def natural_share(df) -> float:
    return round(float((df.label == 0).mean()), 3)


# ---------------------------------------------------------------- splits
def _exclude(pool_mask: np.ndarray, ref: np.ndarray) -> np.ndarray:
    grp = _G["grp"]
    bad = np.zeros(grp.max() + 1, bool)
    bad[grp[ref]] = True
    return np.flatnonzero(pool_mask & ~bad[grp])


def split(mode: str, rng, n_ref: int):
    lab, spk, bona = _G["lab"], _G["spk"], _G["bona"]
    keep = np.ones(lab.size, bool)
    if mode == "overlap":
        ref = rng.choice(bona, n_ref, replace=False)
        keep[ref] = False
        return ref, _exclude(keep, ref), int(np.unique(spk[ref]).size)
    speakers = np.unique(spk[bona])
    rng.shuffle(speakers)
    counts = np.bincount(spk[bona], minlength=spk.max() + 1)
    chosen, n = [], 0
    for u in speakers:
        chosen.append(u)
        n += counts[u]
        if n >= n_ref:
            break
    return _ref_from_speakers(np.array(chosen), rng, n_ref)


def _ref_from_speakers(chosen: np.ndarray, rng, n_ref: int):
    lab, spk, bona = _G["lab"], _G["spk"], _G["bona"]
    in_ref = np.zeros(spk.max() + 1, bool)
    in_ref[chosen] = True
    cand = bona[in_ref[spk[bona]]]
    ref = rng.choice(cand, min(n_ref, cand.size), replace=False)
    return ref, _exclude(~in_ref[spk], ref), int(chosen.size)


def draw_traffic(pool: np.ndarray, pi: float, rng):
    lab = _G["lab"]
    b_pool, s_pool = pool[lab[pool] == 1], pool[lab[pool] == 0]
    n = min(12000, int(min(b_pool.size / max(1 - pi, 1e-9), s_pool.size / max(pi, 1e-9))))
    b = rng.choice(b_pool, int(round(n * (1 - pi))), replace=False)
    s = rng.choice(s_pool, int(round(n * pi)), replace=False)
    return b, s


def _feat(X, idx):
    return np.asarray(X[np.sort(idx)], dtype=np.float32)[np.argsort(np.argsort(idx))]


# ---------------------------------------------------------------- one repetition
def _evaluate(s_cal, s_u, nb, folds):
    sb, ss = s_u[:nb], s_u[nb:]
    if folds is None:
        op = P.op_point(P.conformal_bona_threshold(s_cal, P.ALPHA), sb, ss)
    else:
        op = P.foldwise_anchor_op(s_cal, folds[0], sb, folds[1][:nb], ss, folds[1][nb:])
    return {"eer": P.eer(sb, ss), "frr": op.frr, "far": op.far, "hter": op.hter, "shortcut_auc": P.auc(s_cal, sb)}


def _adapt_rows(base, frozen_eval, b_fr, gate_open, X, tr, cal, traffic, nb, rng, feat):
    a = R.crossfit(_feat(X, tr), _feat(X, cal), _feat(X, traffic), rng)
    ad_eval = _evaluate(a[0], a[2], nb, (a[1], a[3]))
    b_ad = R.bona_share(a[0], a[2])
    ratio = b_ad / max(b_fr, 1e-9) if np.isfinite(b_ad) else float("nan")
    guard = bool(np.isfinite(ratio) and ratio >= RHO)
    rows = [{**base, "variant": f"adapt_{feat}", "adapted": True, **ad_eval},
            {**base, "variant": f"pga_{feat}", "adapted": gate_open, **(ad_eval if gate_open else frozen_eval)},
            {**base, "variant": f"pgag_{feat}", "adapted": gate_open and guard,
             **(ad_eval if gate_open and guard else frozen_eval)}]
    for r in rows:
        r.update(guard_ratio=ratio, bona_share_adapted=b_ad)
    return rows, a


def main_cell(mode: str, pi: float, rep: int, seed: int):
    rng = np.random.default_rng([seed, 31, 0 if mode == "overlap" else 1, int(round(pi * 1000)), rep])
    ref, pool, n_spk = split(mode, rng, N_REF)
    b, s = draw_traffic(pool, pi, rng)
    traffic, nb = np.r_[b, s], b.size
    ref = rng.permutation(ref)
    tr, cal = ref[: ref.size // 2], ref[ref.size // 2:]
    fz = _G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= GATE)
    base = {"mode": mode, "pi": pi, "rep": rep, "n_traffic": int(traffic.size), "n_ref_speakers": n_spk,
            "pi_hat_frozen": 1 - b_fr, "gate_open": gate_open}
    frozen_eval = _evaluate(fz[cal], fz[traffic], nb, None)
    rows = [{**base, "variant": "frozen", "adapted": False, **frozen_eval}]
    for feat, X in (("ssl1024", _G["ssl"]), ("emb160", _G["emb"])):
        r, _ = _adapt_rows(base, frozen_eval, b_fr, gate_open, X, tr, cal, traffic, nb, rng, feat)
        rows += r
    return rows


def speaker_cell(k, rep: int, seed: int):
    """Guard versus the number of reference speakers (k = 'matched' is the control)."""
    rng = np.random.default_rng([seed, 37, 0 if k == "matched" else int(k), rep])
    if k == "matched":
        ref, pool, n_spk = split("overlap", rng, SPK_NREF)
    else:
        speakers = np.unique(_G["spk"][_G["bona"]])
        ref, pool, n_spk = _ref_from_speakers(rng.choice(speakers, int(k), replace=False), rng, SPK_NREF)
    b, s = draw_traffic(pool, SPK_SHARE, rng)
    traffic, nb = np.r_[b, s], b.size
    ref = rng.permutation(ref)
    tr, cal = ref[: ref.size // 2], ref[ref.size // 2:]
    fz = _G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= GATE)
    base = {"k": str(k), "rep": rep, "n_ref": int(ref.size), "n_ref_speakers": n_spk, "gate_open": gate_open}
    frozen_eval = _evaluate(fz[cal], fz[traffic], nb, None)
    rows, _ = _adapt_rows(base, frozen_eval, b_fr, gate_open, _G["ssl"], tr, cal, traffic, nb, rng, "ssl1024")
    return [{**base, "variant": "frozen", "adapted": False, **frozen_eval}] + rows


def break_cell(pi: float, rep: int, seed: int):
    rng = np.random.default_rng([seed, 41, int(round(pi * 1000)), rep])
    ref, pool, _ = split("overlap", rng, N_REF)
    b, s = draw_traffic(pool, pi, rng)
    traffic = np.r_[b, s]
    ref = rng.permutation(ref)
    tr, cal = ref[: ref.size // 2], ref[ref.size // 2:]
    X = _G["ssl"]
    *_, models = R.crossfit(_feat(X, tr), _feat(X, cal), _feat(X, traffic), rng)
    lins = [R._lin(m) for m in models]
    w = np.mean([l[0] for l in lins], axis=0)
    c0 = float(np.mean([l[1] for l in lins]))
    lab, df = _G["lab"], _G["df"]
    keep = np.ones(lab.size, bool)
    keep[np.r_[ref, traffic]] = False
    rest = _exclude(keep, np.r_[ref, traffic])     # no source utterance shared with reference or traffic
    _G_ssl = _G["ssl"]
    ad = np.empty(rest.size)
    for i in range(0, rest.size, 65536):
        ad[i:i + 65536] = np.asarray(_G_ssl[rest[i:i + 65536]], np.float32) @ w.astype(np.float32) + c0
    fz = _G["score"][rest]
    lr = lab[rest]
    codec, attack = df.codec.to_numpy()[rest], df.attack.to_numpy()[rest]
    rows = []
    for name, sc, cal_s in (("frozen", fz, _G["score"][cal]),
                            ("adapt", ad, _feat(X, cal).astype(np.float64) @ w + c0)):
        tau = P.conformal_bona_threshold(cal_s, P.ALPHA)
        b_all, s_all = sc[lr == 1], sc[lr == 0]

        def add(dim, g, sb, ss):
            if sb.size >= 20 and ss.size >= 20:
                rows.append({"pi": pi, "rep": rep, "method": name, "dim": dim, "group": g, "n_bona": int(sb.size),
                             "n_spoof": int(ss.size), "eer": P.eer(sb, ss), "frr_anchor": float(np.mean(sb < tau)),
                             "far_anchor": float(np.mean(ss >= tau))})
        add("all", "all", b_all, s_all)
        for v in sorted(set(codec)):
            m = codec == v
            add("codec", v, sc[m & (lr == 1)], sc[m & (lr == 0)])
        for v in sorted(set(attack[lr == 0])):
            add("attack", v, b_all, sc[(attack == v) & (lr == 0)])
        for gname, members in (("malafide", MALAFIDE), ("malacopula", MALACOPULA)):
            add("adversarial", gname, b_all, sc[np.isin(attack, members) & (lr == 0)])
    return rows


# ---------------------------------------------------------------- runs
def _run(ws, kind, part, n_parts, jobs, seed, local, max_tasks=None):
    digest = prereg(ws)
    df, ssl, emb = load(ws, local)
    if kind == "main":
        shares = sorted(set(SHARES) | {natural_share(df)})
        tasks = [(m, pi, r, seed) for m in ("overlap", "disjoint") for pi in shares for r in range(REPS)]
        fn, out, flush = main_cell, "asv5_main", 10
    elif kind == "speakers":
        tasks = [(k, r, seed) for k in ("matched",) + tuple(str(x) for x in SPK_K) for r in range(SPK_REPS)]
        fn, out, flush = speaker_cell, "asv5_speakers", 10
    else:
        tasks = [(pi, r, seed) for pi in BREAK_SHARES for r in range(BREAK_REPS)]
        fn, out, flush = break_cell, "asv5_breakdown", 2
    return run_chunked(fn, tasks, ws / "results" / out, _init, (df, ssl, emb), jobs=jobs, part=part,
                       n_parts=n_parts, flush_every=flush, label=f"ASVspoof 5 {kind}", max_tasks=max_tasks,
                       config={"prereg_sha256": digest, "kind": kind, "seed": seed, "n_rows": int(len(df))})


def _ci_ratio(a, b, rng, n=4000):
    idx = rng.integers(0, a.size, (n, a.size))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def verdict(ws: Path) -> int:
    rng = np.random.default_rng(0)
    res = pd.read_csv(ws / "results" / "asv5_main" / "results.csv")
    shares = sorted(res.pi.unique())
    natural = [p for p in shares if p not in SHARES]

    def arr(mode, pi, v, col):
        g = res[(res["mode"] == mode) & (res.pi == pi) & (res.variant == v)].sort_values("rep")
        return g[col].to_numpy(dtype=float)

    def h(v):
        out = {}
        h1 = {}
        for pi in (0.90, 0.96):
            e, f = arr("overlap", pi, v, "eer"), arr("overlap", pi, "frozen", "eer")
            hh, hf = arr("overlap", pi, v, "hter"), arr("overlap", pi, "frozen", "hter")
            h1[f"{pi:g}"] = {"eer": float(e.mean()), "eer_frozen": float(f.mean()), "eer_ratio": float(e.mean() / f.mean()),
                             "eer_ratio_ci95": _ci_ratio(e, f, rng), "hter": float(hh.mean()), "hter_frozen": float(hf.mean()),
                             "adapted_rate": float(arr("overlap", pi, v, "adapted").mean()),
                             "pass": bool(e.mean() <= 0.85 * f.mean() and hh.mean() < hf.mean())}
        out["H1_gain"] = {"cells": h1, "pass": all(x["pass"] for x in h1.values())}
        h2 = {}
        for pi in (0.05, 0.20, 0.50):
            e, f = arr("overlap", pi, v, "eer"), arr("overlap", pi, "frozen", "eer")
            h2[f"{pi:g}"] = {"eer_ratio": float(e.mean() / f.mean()), "adapted_rate": float(arr("overlap", pi, v, "adapted").mean()),
                             "pass": bool(e.mean() <= 1.02 * f.mean())}
        out["H2_no_harm_low_share"] = {"cells": h2, "pass": all(x["pass"] for x in h2.values())}
        h3 = {}
        for pi in shares:
            e, f = arr("disjoint", pi, v, "eer"), arr("disjoint", pi, "frozen", "eer")
            r_, rf = arr("disjoint", pi, v, "frr"), arr("disjoint", pi, "frozen", "frr")
            h3[f"{pi:g}"] = {"eer_ratio": float(e.mean() / f.mean()), "frr": float(r_.mean()), "frr_frozen": float(rf.mean()),
                             "adapted_rate": float(arr("disjoint", pi, v, "adapted").mean()),
                             "pass": bool(e.mean() <= 1.02 * f.mean() and r_.mean() <= rf.mean() + 0.01)}
        out["H3_no_harm_disjoint"] = {"cells": h3, "pass": all(x["pass"] for x in h3.values())}
        frr = {f"{pi:g}": float(arr("overlap", pi, v, "frr").mean()) for pi in shares}
        out["H4_genuine_rejection"] = {"max_mean_frr": max(frr.values()), "by_share": frr, "pass": max(frr.values()) <= 0.06}
        auc = {f"{pi:g}": float(arr("overlap", pi, v, "shortcut_auc").mean()) for pi in shares}
        out["H5_shortcut"] = {"max_mean_auc": max(auc.values()), "by_share": auc, "pass": max(auc.values()) <= 0.60}
        out["all_pass"] = all(out[k]["pass"] for k in ("H1_gain", "H2_no_harm_low_share", "H3_no_harm_disjoint",
                                                        "H4_genuine_rejection", "H5_shortcut"))
        return out

    rep = {"prereg_sha256": json.loads((ws / "results" / "PREREGISTRATION_ASV5.json").read_text())["sha256"],
           "natural_share": natural[0] if natural else None,
           "primary_pgag_ssl1024": h("pgag_ssl1024")}
    rep["decision"] = ("PGA-G CONFIRMED on ASVspoof 5 (H1-H5 pass)" if rep["primary_pgag_ssl1024"]["all_pass"]
                       else "PGA-G NOT CONFIRMED on ASVspoof 5 (see failed hypotheses)")
    rep["secondary"] = {"pga_ssl1024_no_guard": h("pga_ssl1024"), "pgag_emb160": h("pgag_emb160")}
    g = res[res.variant.isin(["frozen", "adapt_ssl1024", "pga_ssl1024", "pgag_ssl1024", "pgag_emb160"])]
    tab = g.groupby(["mode", "pi", "variant"])[["eer", "frr", "hter", "adapted", "shortcut_auc"]].mean()
    gr = res[res.variant == "adapt_ssl1024"].groupby(["mode", "pi"]).guard_ratio.describe()[["min", "50%", "max"]]
    rep["guard_ratio_by_cell"] = {f"{m}|{p:g}": row.to_dict() for (m, p), row in gr.iterrows()}
    lines = ["Mean EER / FRR / HTER (%), share adapted, shortcut AUC", (tab * 100).round(2).to_string(), "",
             "Guard statistic (adapted / frozen genuine share), unconditional adaptation", gr.round(4).to_string(), ""]
    sp = ws / "results" / "asv5_speakers" / "results.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        s = s[s.variant.isin(["frozen", "pgag_ssl1024"])]
        st = s.groupby(["k", "variant"])[["eer", "frr", "adapted"]].mean().unstack("variant")
        gq = pd.read_csv(sp).query("variant == 'adapt_ssl1024'").groupby("k").guard_ratio.describe()[["min", "50%", "max"]]
        rep["speakers"] = {"table": {f"{k}": {f"{a}|{b}": float(x) for (a, b), x in row.items()} for k, row in st.iterrows()},
                           "guard_ratio": {k: row.to_dict() for k, row in gq.iterrows()}}
        lines += ["Guard versus number of reference speakers (spoof share 0.90, 300 recordings)",
                  (st * 100).round(2).to_string(), gq.round(4).to_string(), ""]
    bp = ws / "results" / "asv5_breakdown" / "results.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        e = b.groupby(["pi", "dim", "group", "method"])[["eer", "frr_anchor", "n_bona", "n_spoof"]].mean().unstack("method")
        diff = e[("eer", "adapt")] - e[("eer", "frozen")]
        rep["breakdown"] = {"worse_by_1pt": {f"{p:g}|{d}|{g}": float(x) for (p, d, g), x in diff[diff > 0.01].items()},
                            "adversarial": {f"{p:g}|{g}|{m}": float(e.loc[(p, "adversarial", g), ("eer", m)])
                                            for p in e.index.get_level_values(0).unique()
                                            for g in ("malafide", "malacopula") for m in ("frozen", "adapt")
                                            if (p, "adversarial", g) in e.index}}
        lines += ["Held-out breakdown (matched), EER and anchor FRR (%)", (e[["eer", "frr_anchor"]] * 100).round(2).to_string(), ""]
    (ws / "results" / "ASV5_VERDICT.json").write_text(json.dumps(rep, indent=2, default=float))
    (ws / "results" / "ASV5_SUMMARY.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(json.dumps({k: rep[k] for k in ("decision", "natural_share")}, indent=2))
    for k, v in rep["primary_pgag_ssl1024"].items():
        if isinstance(v, dict):
            print(f"  {k}: {'pass' if v['pass'] else 'FAIL'}")
    log(f"DECISION: {rep['decision']} -> {ws / 'results' / 'ASV5_VERDICT.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prereg", "main", "speakers", "breakdown", "all", "verdict"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--local", default=None)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-tasks", type=int, default=None)
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    if a.cmd == "prereg":
        prereg(ws)
        return 0
    if a.cmd == "verdict":
        return verdict(ws)
    kinds = ["main", "speakers", "breakdown"] if a.cmd == "all" else [a.cmd]
    done = [(_run(ws, k, a.part, a.n_parts, a.jobs, a.seed, a.local, a.max_tasks) is not None) for k in kinds]
    if a.cmd == "all" and all(done):
        return verdict(ws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
