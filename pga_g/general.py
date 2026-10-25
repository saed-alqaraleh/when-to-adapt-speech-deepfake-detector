"""
v3: generalisation across detectors + deployment analyses (pre-registered before any v3 score
or analysis result exists).

Detectors: sslaasist (the study's detector, existing scores), aasist and mamba (pga_g.multi).
Corpora:   df21 (ASVspoof 2021 DF eval) and asv5 (ASVspoof 5 eval, track 1).

Analyses (results/v3/<kind>_<corpus>_<detector>/):
  main     the confirmatory design of the ASVspoof 5 test, on both corpora and every detector
  size     traffic of 500 to 5,000 trials (deployment analysis)
  stream   a traffic stream in which a spoofing campaign starts and stops (deployment analysis)
  refsize  references of 200 to 2,000 recordings, with two small-reference variants
  tta      test-time-adaptation baselines: information maximisation and pseudo-label self-training

Every draw of references and traffic depends only on (seed, analysis, corpus, condition,
repetition), never on the detector, so the three detectors see identical data in every repetition.

    python -m pga_g.general prereg  --workspace W
    python -m pga_g.general deploy  --workspace W --detectors sslaasist --part 0 --n-parts 2
    python -m pga_g.general all     --workspace W --detectors sslaasist,aasist,mamba --part 0 --n-parts 2
    python -m pga_g.general verdict --workspace W
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
from .detectors import FEATURES, NAMES, PRIMARY, SECONDARY

GATE, RHO = 0.80, 0.5
N_REF, REPS = 1000, 20
SHARES = {"df21": (0.05, 0.20, 0.50, 0.80, 0.90), "asv5": (0.05, 0.20, 0.50, 0.80, 0.90, 0.96)}
NATURAL = {"df21": 0.963, "asv5": 0.796}
H1_SHARES = {"df21": (0.90, 0.963), "asv5": (0.90, 0.96)}
SIZE_N, SIZE_SHARES = (500, 1000, 2000, 5000), (0.05, 0.90)
STREAM_W, STREAM_T, STREAM_REPS = 1000, 30, 10
REFSIZES, REFSIZE_SHARE = (200, 300, 500, 1000, 2000), 0.90
ST_CONF = float(np.log(0.9 / 0.1))                    # pseudo-label threshold: posterior 0.9 / 0.1
EM_STEPS, EM_LR, EM_L2 = 200, 0.01, 1e-3
DETECTORS, CORPORA = ("sslaasist", "aasist", "mamba"), ("df21", "asv5")
CID = {"df21": 0, "asv5": 1}
KINDS = ("main", "size", "stream", "refsize", "tta")

PREREG = {
    "title": "Generalisation of PGA-G across detectors, and deployment analyses",
    "fixed_on": "2026-10-22, before any v3 score (runs/v3) or v3 analysis result (results/v3) existed",
    "prior_evidence": ("PGA confirmed on ASVspoof 2021 DF (PREREGISTRATION_PGA.json); PGA-G confirmed on ASVspoof 5 "
                       "(PREREGISTRATION_ASV5.json); both with the SSL-AASIST detector only"),
    "detectors": {
        "aasist": "official AASIST trained on ASVspoof 2019 LA (github.com/clovaai/aasist@a04c9863, weights sha256 51d2d9cf...); "
                  "no self-supervised front end; adaptation features: its 160-d readout",
        "mamba": "XLSR-Mamba trained on ASVspoof 2019 LA (huggingface.co/AustinXiao/XLSR-Mamba-DF, revision pinned in "
                 "runs/v3/config.json); adaptation features: time-mean of its fine-tuned XLS-R output (1,024-d); "
                 "secondary: its 144-d pre-classifier embedding",
        "sslaasist": "the study's SSL-AASIST, existing scores, reference rows (no decision)",
    },
    "scoring": ("score = genuine minus spoof logit on the first 64,600 samples (official crop/tile); precision: fastest of "
                "bf16/fp16, TF32, fp32 within 0.05 logit of strict fp32; corpora at the revisions pinned by the earlier runs; "
                "ASVspoof 2019 LA eval scored to obtain each detector's source threshold (EER threshold)"),
    "reproduction_checks": ("AASIST ASVspoof 2019 LA EER against the published 0.83%; XLSR-Mamba ASVspoof 2021 DF per-trial "
                            "genuine logits against the authors' published score file; reported, not part of the decision"),
    "method": "unchanged: alpha 0.05 fold-matched conformal anchor, 5-fold cross-fitted PU logistic regression (C 0.1, "
              "balanced), gate 0.80 on the frozen best-bin estimate, guard 0.5 on the adapted / frozen estimate ratio",
    "design_main": ("as PREREGISTRATION_ASV5.json on both corpora: 1,000-recording reference (500/500), matched and "
                    "speaker-disjoint modes, traffic min(12,000, available), 20 repetitions; spoof shares ASVspoof 2021 DF "
                    "0.05, 0.20, 0.50, 0.80, 0.90, natural 0.963; ASVspoof 5 0.05, 0.20, 0.50, 0.80, 0.90, 0.96, natural "
                    "0.796; source-recording groups never split (ASVspoof 5 CODEC_SEED; ASVspoof 2021 DF file)"),
    "hypotheses_per_detector_and_corpus": {
        "H1_gain": "matched, shares 0.90 and 0.963 (ASVspoof 2021 DF) / 0.90 and 0.96 (ASVspoof 5): mean EER ratio <= 0.85 "
                   "and mean anchor HTER below the frozen detector's",
        "H2_no_harm_low_share": "matched, shares 0.05, 0.20, 0.50: mean EER ratio <= 1.02",
        "H3_no_harm_disjoint": "disjoint, every share: mean EER ratio <= 1.02 and mean anchor FRR <= frozen + 0.01",
        "H4_genuine_rejection": "matched, every share: mean anchor FRR <= 0.06",
        "H5_shortcut": "matched, every share: mean shortcut AUC <= 0.60",
    },
    "decision": {
        "G_full_method_on_an_ssl_detector": "XLSR-Mamba: H1-H5 hold on both corpora -> PGA-G generalises to a second "
                                            "self-supervised detector",
        "S_safety_on_a_non_ssl_detector": "AASIST: H2-H5 hold on both corpora -> the safety of PGA-G generalises to a "
                                          "detector without self-supervised features; AASIST H1 is reported as a "
                                          "secondary result because the adaptation step is specified on self-supervised "
                                          "features and AASIST has none",
        "reporting": "every hypothesis for every detector and corpus is reported, whatever the outcome",
    },
    "deployment_analyses_pre_specified_exploratory": {
        "size": "traffic of 500, 1,000, 2,000, 5,000 trials (12,000 from main), shares 0.05 and 0.90, both modes, 20 "
                "repetitions: EER, anchor FRR, gate and adaptation rates of PGA-G versus the frozen detector",
        "stream": "matched 1,000-recording reference; 30 consecutive windows of 1,000 trials drawn without replacement; "
                  "profile 'burst': spoof share 0.05 (windows 1-10), 0.90 (11-20), 0.05 (21-30); profile 'ramp': share "
                  "rising linearly from 0.05 to 0.95; 10 repetitions; PGA-G decides per window",
        "refsize": "matched references of 200, 300, 500, 1,000, 2,000 recordings (half for training), share 0.90, 20 "
                   "repetitions; variants with identical folds: base (C 0.1, all dimensions); 'pca': standardise and "
                   "project on min(64, n_train/4) principal components fitted on the unlabelled traffic; 'strongC': C 0.01",
        "tta": "matched, every main share, the same references and traffic as main: (i) information maximisation "
               "(SHOT-IM): linear head on standardised features initialised by ridge regression on the frozen score, "
               "200 Adam steps (lr 0.01) minimising mean prediction entropy minus marginal entropy plus 0.001 x squared "
               "distance to the initial weights; (ii) pseudo-label self-training: traffic with frozen posterior >= 0.9 "
               "labelled genuine and <= 0.1 spoof, cross-fitted logistic regression (C 0.1, balanced), two rounds; "
               "fewer than 20 pseudo-labels in a class -> frozen detector; both thresholded with the conformal anchor",
    },
    "statistics": "means over repetitions; 95% percentile bootstrap intervals (4,000 resamples, paired by repetition)",
}

_G: dict = {}


# ---------------------------------------------------------------- pre-registration
def prereg(ws: Path) -> str:
    ws = Path(ws)
    path = ws / "results" / "PREREGISTRATION_V3.json"
    body = json.dumps(PREREG, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        if json.loads(path.read_text()).get("sha256") != digest:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        return digest
    parts = ws / "runs" / "v3" / "parts"
    if parts.exists() and any(parts.rglob("*.npz")):
        raise SystemExit("v3 scoring has already started: a pre-registration written now would not be one")
    res = ws / "results" / "v3"
    if res.exists() and any(res.rglob("chunk_*.csv")):
        raise SystemExit("v3 analyses have already started: a pre-registration written now would not be one")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"preregistration": PREREG, "sha256": digest,
                               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    tmp.replace(path)
    log(f"pre-registration written: {path} (sha256 {digest[:16]})")
    return digest


# ---------------------------------------------------------------- inputs
def available(ws: Path, corpus: str, det: str) -> bool:
    ws = Path(ws)
    if det == "sslaasist":
        return (ws / "runs" / ("df21_v1" if corpus == "df21" else "asv5_v1") / "_STAGE_SUCCESS.json").exists()
    return (ws / "runs" / "v3" / corpus / "_STAGE_SUCCESS.json").exists()


def load(ws: Path, corpus: str, det: str, local=None):
    ws = Path(ws)
    base = ws / "runs" / ("df21_v1" if corpus == "df21" else "asv5_v1")
    cols = ["utt_id", "speaker", "codec", "attack", "label", "score"] + (["src_group"] if corpus == "asv5" else [])
    meta = pd.read_csv(base / "scores.csv", usecols=cols, dtype={c: str for c in cols if c not in ("label", "score")},
                       keep_default_na=False)
    if corpus == "df21":
        meta["src_group"] = meta["utt_id"]
    if det == "sslaasist":
        score = meta.score.to_numpy(np.float64)
        feats = {"ssl1024": np.load(R._cache(base / "ssl_mean1024.npy", local), mmap_mode="r"),
                 "emb160": np.load(R._cache(base / "emb160.npy", local), mmap_mode="r")}
    else:
        d = ws / "runs" / "v3" / corpus / det
        s = pd.read_csv(d / "scores.csv", usecols=["utt_id", "score"], dtype={"utt_id": str})
        if not (s.utt_id.to_numpy() == meta.utt_id.to_numpy()).all():
            raise SystemExit(f"{corpus}/{det}: scores are not row-aligned with the earlier run")
        score = s.score.to_numpy(np.float64)
        feats = {f: np.load(R._cache(d / f"{f}.npy", local), mmap_mode="r") for f in FEATURES[det]}
    for f, a in feats.items():
        if a.shape[0] != len(meta):
            raise SystemExit(f"{corpus}/{det}/{f}: features are not row-aligned")
    return meta.drop(columns=["score"]), score, feats


def _init(meta, score, feats, corpus, det):
    _G.clear()
    _G.update(lab=meta.label.to_numpy(), score=score, feats=feats, corpus=corpus, det=det,
              spk=pd.factorize(meta.speaker)[0], grp=pd.factorize(meta.src_group)[0])
    _G["bona"] = np.flatnonzero(_G["lab"] == 1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                # noqa: BLE001
        pass


# ---------------------------------------------------------------- draws (detector-independent)
def _exclude(pool_mask, ref):
    grp = _G["grp"]
    bad = np.zeros(grp.max() + 1, bool)
    bad[grp[ref]] = True
    return np.flatnonzero(pool_mask & ~bad[grp])


def split(mode, rng, n_ref):
    lab, spk, bona = _G["lab"], _G["spk"], _G["bona"]
    if mode == "overlap":
        ref = rng.choice(bona, n_ref, replace=False)
        keep = np.ones(lab.size, bool)
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
    in_ref = np.zeros(spk.max() + 1, bool)
    in_ref[chosen] = True
    cand = bona[in_ref[spk[bona]]]
    ref = rng.choice(cand, min(n_ref, cand.size), replace=False)
    return ref, _exclude(~in_ref[spk], ref), len(chosen)


def draw_traffic(pool, pi, rng, n_max=12000):
    lab = _G["lab"]
    b_pool, s_pool = pool[lab[pool] == 1], pool[lab[pool] == 0]
    n = min(n_max, int(min(b_pool.size / max(1 - pi, 1e-9), s_pool.size / max(pi, 1e-9))))
    return (rng.choice(b_pool, int(round(n * (1 - pi))), replace=False),
            rng.choice(s_pool, int(round(n * pi)), replace=False))


def _draw_main(mode, pi, rep, seed):
    rng = np.random.default_rng([seed, 51, CID[_G["corpus"]], 0 if mode == "overlap" else 1, int(round(pi * 1000)), rep])
    ref, pool, n_spk = split(mode, rng, N_REF)
    b, s = draw_traffic(pool, pi, rng)
    ref = rng.permutation(ref)
    return rng, ref[: ref.size // 2], ref[ref.size // 2:], np.r_[b, s], b.size, n_spk


def _feat(X, idx):
    idx = np.asarray(idx)
    o = np.argsort(idx)
    out = np.empty((idx.size, X.shape[1]), np.float32)
    out[o] = np.asarray(X[idx[o]], dtype=np.float32)
    return out


# ---------------------------------------------------------------- evaluation pieces
def _evaluate(s_cal, s_u, nb, folds=None):
    sb, ss = s_u[:nb], s_u[nb:]
    if folds is None:
        op = P.op_point(P.conformal_bona_threshold(s_cal, P.ALPHA), sb, ss)
    else:
        op = P.foldwise_anchor_op(s_cal, folds[0], sb, folds[1][:nb], ss, folds[1][nb:])
    return {"eer": P.eer(sb, ss), "frr": op.frr, "far": op.far, "hter": op.hter, "shortcut_auc": P.auc(s_cal, sb)}


def _fit_c(C):
    def fit(X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)).fit(X, y)
    return fit


def _pca(Xtr, Xcal, Xu, k):
    from sklearn.decomposition import PCA
    mu, sd = Xu.mean(0), Xu.std(0) + 1e-6
    p = PCA(n_components=k, random_state=0).fit((Xu - mu) / sd)
    return tuple(p.transform((X - mu) / sd).astype(np.float32) for X in (Xtr, Xcal, Xu))


def _adapt(fz, tr, cal, traffic, nb, rng, X, b_fr, gate_open, fit=None, pca_k=None):
    Xtr, Xcal, Xu = _feat(X, tr), _feat(X, cal), _feat(X, traffic)
    if pca_k:
        Xtr, Xcal, Xu = _pca(Xtr, Xcal, Xu, pca_k)
    a = R.crossfit(Xtr, Xcal, Xu, rng, fit=fit)
    ev = _evaluate(a[0], a[2], nb, (a[1], a[3]))
    b_ad = R.bona_share(a[0], a[2])
    ratio = b_ad / max(b_fr, 1e-9) if np.isfinite(b_ad) else float("nan")
    guard = bool(np.isfinite(ratio) and ratio >= RHO)
    return ev, ratio, guard, b_ad


def _rows3(base, tag, frozen_eval, ev, ratio, guard, gate_open, b_ad):
    rows = [{**base, "variant": f"adapt_{tag}", "adapted": True, **ev},
            {**base, "variant": f"pga_{tag}", "adapted": gate_open, **(ev if gate_open else frozen_eval)},
            {**base, "variant": f"pgag_{tag}", "adapted": gate_open and guard,
             **(ev if gate_open and guard else frozen_eval)}]
    for r in rows:
        r.update(guard_ratio=ratio, bona_share_adapted=b_ad)
    return rows


# ---------------------------------------------------------------- analysis cells
def main_cell(mode, pi, rep, seed):
    rng, tr, cal, traffic, nb, n_spk = _draw_main(mode, pi, rep, seed)
    fz = _G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= GATE)
    base = {"mode": mode, "pi": pi, "rep": rep, "n_traffic": int(traffic.size), "n_ref_speakers": n_spk,
            "pi_hat_frozen": 1 - b_fr, "gate_open": gate_open}
    frozen = _evaluate(fz[cal], fz[traffic], nb)
    rows = [{**base, "variant": "frozen", "adapted": False, **frozen}]
    det = _G["det"]
    for tag, f in (("primary", PRIMARY[det]), ("secondary", SECONDARY[det] if det == "mamba" else None)):
        if f is None:
            continue
        ev, ratio, guard, b_ad = _adapt(fz, tr, cal, traffic, nb, rng, _G["feats"][f], b_fr, gate_open)
        rows += _rows3(base, tag, frozen, ev, ratio, guard, gate_open, b_ad)
    return rows


def size_cell(mode, pi, n, rep, seed):
    rng = np.random.default_rng([seed, 53, CID[_G["corpus"]], 0 if mode == "overlap" else 1, int(round(pi * 1000)), n, rep])
    ref, pool, n_spk = split(mode, rng, N_REF)
    b, s = draw_traffic(pool, pi, rng, n_max=n)
    ref = rng.permutation(ref)
    tr, cal, traffic, nb = ref[: ref.size // 2], ref[ref.size // 2:], np.r_[b, s], b.size
    fz = _G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= GATE)
    base = {"mode": mode, "pi": pi, "n": n, "rep": rep, "n_traffic": int(traffic.size), "gate_open": gate_open}
    frozen = _evaluate(fz[cal], fz[traffic], nb)
    ev, ratio, guard, b_ad = _adapt(fz, tr, cal, traffic, nb, rng, _G["feats"][PRIMARY[_G["det"]]], b_fr, gate_open)
    return [{**base, "variant": "frozen", "adapted": False, **frozen}] + \
        [r for r in _rows3(base, "primary", frozen, ev, ratio, guard, gate_open, b_ad) if r["variant"] != "pga_primary"]


def stream_shares(profile):
    if profile == "burst":
        return [0.05] * 10 + [0.90] * 10 + [0.05] * 10
    return list(np.round(np.linspace(0.05, 0.95, STREAM_T), 4))


def stream_cell(profile, rep, seed):
    rng = np.random.default_rng([seed, 57, CID[_G["corpus"]], 0 if profile == "burst" else 1, rep])
    lab, fz = _G["lab"], _G["score"]
    ref, pool, _ = split("overlap", rng, N_REF)
    ref = rng.permutation(ref)
    tr, cal = ref[: ref.size // 2], ref[ref.size // 2:]
    b_pool, s_pool = rng.permutation(pool[lab[pool] == 1]), rng.permutation(pool[lab[pool] == 0])
    ib = is_ = 0
    rows = []
    X = _G["feats"][PRIMARY[_G["det"]]]
    for w, pi in enumerate(stream_shares(profile), start=1):
        nb = int(round(STREAM_W * (1 - pi)))
        b, s = b_pool[ib: ib + nb], s_pool[is_: is_ + STREAM_W - nb]
        ib, is_ = ib + nb, is_ + STREAM_W - nb
        if b.size < nb or s.size < STREAM_W - nb:
            raise RuntimeError(f"stream pool exhausted at window {w}")
        traffic = np.r_[b, s]
        b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
        gate_open = bool(1 - b_fr >= GATE)
        frozen = _evaluate(fz[cal], fz[traffic], nb)
        if gate_open:
            ev, ratio, guard, _ = _adapt(fz, tr, cal, traffic, nb, rng, X, b_fr, gate_open)
        else:
            ev, ratio, guard = frozen, float("nan"), False
        used = ev if gate_open and guard else frozen
        rows.append({"profile": profile, "rep": rep, "window": w, "pi": pi, "pi_hat_frozen": 1 - b_fr,
                     "gate_open": gate_open, "adapted": gate_open and guard, "guard_ratio": ratio,
                     "eer_frozen": frozen["eer"], "frr_frozen": frozen["frr"], "hter_frozen": frozen["hter"],
                     "eer_pgag": used["eer"], "frr_pgag": used["frr"], "hter_pgag": used["hter"]})
    return rows


def refsize_cell(n_ref, rep, seed):
    rng = np.random.default_rng([seed, 59, CID[_G["corpus"]], n_ref, rep])
    ref, pool, _ = split("overlap", rng, n_ref)
    b, s = draw_traffic(pool, REFSIZE_SHARE, rng)
    ref = rng.permutation(ref)
    tr, cal, traffic, nb = ref[: ref.size // 2], ref[ref.size // 2:], np.r_[b, s], b.size
    fz = _G["score"]
    b_fr = P.estimate_bona_share(fz[cal], fz[traffic])
    gate_open = bool(1 - b_fr >= GATE)
    base = {"n_ref": n_ref, "rep": rep, "gate_open": gate_open}
    frozen = _evaluate(fz[cal], fz[traffic], nb)
    rows = [{**base, "variant": "frozen", "adapted": False, **frozen}]
    X = _G["feats"][PRIMARY[_G["det"]]]
    k = int(min(64, tr.size // 4))
    for tag, fit, pk in (("base", None, None), ("pca", None, k), ("strongC", _fit_c(0.01), None)):
        vr = np.random.default_rng([seed, 59, CID[_G["corpus"]], n_ref, rep, 7])   # identical folds per variant
        ev, ratio, guard, b_ad = _adapt(fz, tr, cal, traffic, nb, vr, X, b_fr, gate_open, fit=fit, pca_k=pk)
        rows += [r for r in _rows3({**base, "pca_k": k if tag == "pca" else None}, tag, frozen, ev, ratio, guard,
                                   gate_open, b_ad) if r["variant"] != f"pga_{tag}"]
    return rows


def _em_scores(Xu, Xc, s):
    """Information maximisation (SHOT-IM) on a linear head over standardised features."""
    mu, sd = Xu.mean(0), Xu.std(0) + 1e-6
    Z, Zc = (Xu - mu) / sd, (Xc - mu) / sd
    n, d = Z.shape
    A = Z.T @ Z + 1.0 * np.eye(d)
    w0 = np.linalg.solve(A, Z.T @ (s - s.mean()))
    b0 = float(s.mean())
    w, b = w0.copy(), b0
    m = np.zeros(d + 1)
    v = np.zeros(d + 1)
    for t in range(1, EM_STEPS + 1):
        z = Z @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        q = p * (1 - p)
        pbar = float(np.clip(p.mean(), 1e-6, 1 - 1e-6))
        g = (-z * q - np.log((1 - pbar) / pbar) * q) / n           # d/dz [mean H(p_i) - H(mean p)]
        grad = np.r_[Z.T @ g + 2 * EM_L2 * (w - w0), g.sum()]
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        step = EM_LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        w, b = w - step[:d], b - step[d]
    return Z @ w + b, Zc @ w + b


def _st_scores(Xu, Xc, s, rng, folds=5):
    """Two rounds of cross-fitted pseudo-label self-training; None if a class has < 20 labels."""
    fit = _fit_c(0.1)
    cur = s
    fold_u = fold_c = None
    for _ in range(2):
        y = np.where(cur >= ST_CONF, 1, np.where(cur <= -ST_CONF, 0, -1))
        if (y == 1).sum() < 20 or (y == 0).sum() < 20:
            return None
        fold_u = rng.integers(0, folds, Xu.shape[0])
        fold_c = np.arange(Xc.shape[0]) % folds
        rng.shuffle(fold_c)
        new_u, s_c = np.empty(Xu.shape[0]), np.empty(Xc.shape[0])
        for k in range(folds):
            m = (fold_u != k) & (y >= 0)
            if len(set(y[m])) < 2:
                return None
            clf = fit(Xu[m], y[m])
            new_u[fold_u == k] = clf.decision_function(Xu[fold_u == k])
            s_c[fold_c == k] = clf.decision_function(Xc[fold_c == k])
        cur = new_u
    return cur, s_c, fold_c, fold_u


def tta_cell(pi, rep, seed):
    rng, tr, cal, traffic, nb, _ = _draw_main("overlap", pi, rep, seed)     # the main cell's exact data
    fz = _G["score"]
    X = _G["feats"][PRIMARY[_G["det"]]]
    Xu, Xc = _feat(X, traffic).astype(np.float64), _feat(X, cal).astype(np.float64)
    base = {"mode": "overlap", "pi": pi, "rep": rep}
    zu, zc = _em_scores(Xu, Xc, fz[traffic])
    rows = [{**base, "variant": "tta_em", "fallback": False, **_evaluate(zc, zu, nb)}]
    st = _st_scores(Xu, Xc, fz[traffic], np.random.default_rng([seed, 61, CID[_G["corpus"]], int(round(pi * 1000)), rep]))
    if st is None:
        rows.append({**base, "variant": "tta_st", "fallback": True, **_evaluate(fz[cal], fz[traffic], nb)})
    else:
        su, sc, fc, fu = st
        rows.append({**base, "variant": "tta_st", "fallback": False, **_evaluate(sc, su, nb, (fc, fu))})
    return rows


# ---------------------------------------------------------------- runs
def tasks_for(kind, corpus, seed):
    shares = sorted(set(SHARES[corpus]) | {NATURAL[corpus]})
    if kind == "main":
        return main_cell, [(m, p, r, seed) for m in ("overlap", "disjoint") for p in shares for r in range(REPS)], 10
    if kind == "size":
        return size_cell, [(m, p, n, r, seed) for m in ("overlap", "disjoint") for p in SIZE_SHARES for n in SIZE_N
                           for r in range(REPS)], 20
    if kind == "stream":
        return stream_cell, [(pr, r, seed) for pr in ("burst", "ramp") for r in range(STREAM_REPS)], 2
    if kind == "refsize":
        return refsize_cell, [(n, r, seed) for n in REFSIZES for r in range(REPS)], 5
    if kind == "tta":
        return tta_cell, [(p, r, seed) for p in shares for r in range(REPS)], 10
    raise ValueError(kind)


def out_dir(ws, kind, corpus, det) -> Path:
    return Path(ws) / "results" / "v3" / f"{kind}_{corpus}_{det}"


def _run(ws, kind, corpus, det, part, n_parts, jobs, seed, local, max_tasks=None):
    digest = prereg(ws)
    meta, score, feats = load(ws, corpus, det, local)
    fn, tasks, flush = tasks_for(kind, corpus, seed)
    return run_chunked(fn, tasks, out_dir(ws, kind, corpus, det), _init, (meta, score, feats, corpus, det), jobs=jobs,
                       part=part, n_parts=n_parts, flush_every=flush, label=f"v3 {kind} {corpus} {NAMES[det]}",
                       max_tasks=max_tasks, config={"prereg_sha256": digest, "kind": kind, "corpus": corpus,
                                                    "detector": det, "seed": seed, "n_rows": int(len(meta))})


# ---------------------------------------------------------------- verdict
def _boot_ratio(a, b, rng, n=4000):
    idx = rng.integers(0, a.size, (n, a.size))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def hypotheses(res: pd.DataFrame, corpus: str, variant: str, rng) -> dict:
    def cell(mode, pi, v):
        return res[(res["mode"] == mode) & np.isclose(res.pi, pi) & (res.variant == v)].sort_values("rep")
    out, shares = {}, sorted(res.pi.unique())
    h1 = {}
    for p in H1_SHARES[corpus]:
        g, f = cell("overlap", p, variant), cell("overlap", p, "frozen")
        ratio = g.eer.mean() / f.eer.mean()
        h1[f"{p:g}"] = {"eer": g.eer.mean(), "eer_frozen": f.eer.mean(), "eer_ratio": ratio,
                        "eer_ratio_ci95": _boot_ratio(g.eer.to_numpy(), f.eer.to_numpy(), rng),
                        "hter": g.hter.mean(), "hter_frozen": f.hter.mean(), "adapted_rate": g.adapted.mean(),
                        "pass": bool(ratio <= 0.85 and g.hter.mean() < f.hter.mean())}
    out["H1_gain"] = {"cells": h1, "pass": all(c["pass"] for c in h1.values())}
    h2 = {f"{p:g}": {"eer_ratio": cell("overlap", p, variant).eer.mean() / cell("overlap", p, "frozen").eer.mean(),
                     "adapted_rate": cell("overlap", p, variant).adapted.mean()} for p in (0.05, 0.20, 0.50)}
    for c in h2.values():
        c["pass"] = bool(c["eer_ratio"] <= 1.02)
    out["H2_no_harm_low_share"] = {"cells": h2, "pass": all(c["pass"] for c in h2.values())}
    h3 = {}
    for p in shares:
        g, f = cell("disjoint", p, variant), cell("disjoint", p, "frozen")
        h3[f"{p:g}"] = {"eer_ratio": g.eer.mean() / f.eer.mean(), "frr": g.frr.mean(), "frr_frozen": f.frr.mean(),
                        "adapted_rate": g.adapted.mean()}
        h3[f"{p:g}"]["pass"] = bool(h3[f"{p:g}"]["eer_ratio"] <= 1.02 and g.frr.mean() <= f.frr.mean() + 0.01)
    out["H3_no_harm_disjoint"] = {"cells": h3, "pass": all(c["pass"] for c in h3.values())}
    frr = {f"{p:g}": cell("overlap", p, variant).frr.mean() for p in shares}
    out["H4_genuine_rejection"] = {"by_share": frr, "max_mean_frr": max(frr.values()), "pass": max(frr.values()) <= 0.06}
    auc = {f"{p:g}": cell("overlap", p, variant).shortcut_auc.mean() for p in shares}
    out["H5_shortcut"] = {"by_share": auc, "max_mean_auc": max(auc.values()), "pass": max(auc.values()) <= 0.60}
    out["all_pass"] = all(out[h]["pass"] for h in out)
    out["safety_pass"] = all(out[h]["pass"] for h in ("H2_no_harm_low_share", "H3_no_harm_disjoint",
                                                      "H4_genuine_rejection", "H5_shortcut"))
    return out


def _read(ws, kind, corpus, det):
    p = out_dir(ws, kind, corpus, det) / "results.csv"
    return pd.read_csv(p) if p.exists() else None


def verdict(ws: Path) -> int:
    ws = Path(ws)
    rng = np.random.default_rng(0)
    V = {"prereg_sha256": prereg(ws), "main": {}, "deployment": {}}
    lines = []
    for det in DETECTORS:
        for corpus in CORPORA:
            res = _read(ws, "main", corpus, det)
            if res is None:
                continue
            h = hypotheses(res, corpus, "pgag_primary", rng)
            V["main"][f"{det}|{corpus}"] = {"pgag_primary": h}
            if det == "mamba":
                V["main"][f"{det}|{corpus}"]["pgag_secondary"] = hypotheses(res, corpus, "pgag_secondary", rng)
                V["main"][f"{det}|{corpus}"]["pga_primary_no_guard"] = hypotheses(res, corpus, "pga_primary", rng)
            lines.append(f"{NAMES[det]:<11} {corpus}: " + " ".join(
                f"{k.split('_')[0]}={'pass' if h[k]['pass'] else 'fail'}" for k in
                ("H1_gain", "H2_no_harm_low_share", "H3_no_harm_disjoint", "H4_genuine_rejection", "H5_shortcut")))
    m = V["main"]
    if all(f"mamba|{c}" in m for c in CORPORA):
        V["decision_G"] = ("PGA-G GENERALISES to XLSR-Mamba (H1-H5 on both corpora)"
                           if all(m[f"mamba|{c}"]["pgag_primary"]["all_pass"] for c in CORPORA)
                           else "PGA-G does not fully generalise to XLSR-Mamba (see per-hypothesis results)")
    if all(f"aasist|{c}" in m for c in CORPORA):
        V["decision_S"] = ("PGA-G SAFETY GENERALISES to AASIST (H2-H5 on both corpora)"
                           if all(m[f"aasist|{c}"]["pgag_primary"]["safety_pass"] for c in CORPORA)
                           else "PGA-G safety does not fully generalise to AASIST (see per-hypothesis results)")
    tables = {}
    for det in DETECTORS:
        for corpus in CORPORA:
            key = f"{det}|{corpus}"
            d = {}
            s = _read(ws, "size", corpus, det)
            if s is not None:
                d["size"] = s.groupby(["mode", "pi", "n", "variant"])[["eer", "frr", "hter", "adapted", "gate_open"]] \
                    .mean().round(4).reset_index().to_dict("records")
            st = _read(ws, "stream", corpus, det)
            if st is not None:
                g = st.groupby(["profile", "window"]).agg(pi=("pi", "first"), gate=("gate_open", "mean"),
                                                          adapted=("adapted", "mean"), eer_frozen=("eer_frozen", "mean"),
                                                          eer_pgag=("eer_pgag", "mean"), frr_pgag=("frr_pgag", "mean"),
                                                          hter_frozen=("hter_frozen", "mean"),
                                                          hter_pgag=("hter_pgag", "mean")).round(4).reset_index()
                b = st[st.profile == "burst"]
                d["stream"] = {"per_window": g.to_dict("records"),
                               "burst_background_adapted_rate": float(b[b.pi < 0.5].adapted.mean()),
                               "burst_campaign_adapted_rate": float(b[b.pi >= 0.5].adapted.mean()),
                               "burst_campaign_eer_frozen": float(b[b.pi >= 0.5].eer_frozen.mean()),
                               "burst_campaign_eer_pgag": float(b[b.pi >= 0.5].eer_pgag.mean()),
                               "burst_background_eer_ratio": float(b[b.pi < 0.5].eer_pgag.mean() / b[b.pi < 0.5].eer_frozen.mean()),
                               "ramp_first_adapted_share_mean": float(st[(st.profile == "ramp") & st.adapted]
                                                                      .groupby("rep").pi.min().mean())
                               if (st[(st.profile == "ramp")].adapted.any()) else None}
            rs = _read(ws, "refsize", corpus, det)
            if rs is not None:
                fz = rs[rs.variant == "frozen"].groupby("n_ref").eer.mean()
                t = rs.groupby(["n_ref", "variant"])[["eer", "frr", "hter", "adapted"]].mean().reset_index()
                t["eer_ratio"] = t.eer / t.n_ref.map(fz)
                d["refsize"] = t.round(4).to_dict("records")
            tt = _read(ws, "tta", corpus, det)
            mres = _read(ws, "main", corpus, det)
            if tt is not None and mres is not None:
                mm = mres[(mres["mode"] == "overlap") & mres.variant.isin(["frozen", "pgag_primary", "adapt_primary"])]
                comb = pd.concat([mm[["pi", "rep", "variant", "eer", "frr", "hter"]],
                                  tt[["pi", "rep", "variant", "eer", "frr", "hter"]]])
                t = comb.groupby(["pi", "variant"])[["eer", "frr", "hter"]].mean().reset_index()
                d["tta"] = {"table": t.round(4).to_dict("records"),
                            "st_fallback_rate": float(tt[tt.variant == "tta_st"].fallback.mean())}
            if d:
                tables[key] = d
    V["deployment"] = tables
    out = ws / "results" / "V3_VERDICT.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(V, indent=2, default=float))
    tmp.replace(out)
    txt = ["v3 verdict", *lines, V.get("decision_G", "XLSR-Mamba: incomplete"), V.get("decision_S", "AASIST: incomplete")]
    for key, d in tables.items():
        if "stream" in d:
            s = d["stream"]
            txt.append(f"{key} stream: adapted in {100 * s['burst_campaign_adapted_rate']:.0f}% of campaign windows and "
                       f"{100 * s['burst_background_adapted_rate']:.0f}% of background windows; campaign EER "
                       f"{100 * s['burst_campaign_eer_frozen']:.1f}% -> {100 * s['burst_campaign_eer_pgag']:.1f}%")
    (ws / "results" / "V3_SUMMARY.txt").write_text("\n".join(txt) + "\n")
    for t in txt:
        log(t)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prereg", "main", "deploy", "all", "verdict"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--detectors", default="sslaasist,aasist,mamba")
    ap.add_argument("--corpora", default="df21,asv5")
    ap.add_argument("--kinds", default=None, help="override the analyses to run, e.g. size,stream")
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
    kinds = (a.kinds.split(",") if a.kinds else
             {"main": ["main"], "deploy": ["size", "stream", "refsize", "tta"], "all": list(KINDS)}[a.cmd])
    complete = True
    for det in a.detectors.split(","):
        for corpus in a.corpora.split(","):
            if not available(ws, corpus, det):
                log(f"{NAMES[det]} on {corpus}: scores not merged yet; skipped")
                complete = False
                continue
            for kind in kinds:
                complete &= _run(ws, kind, corpus, det, a.part, a.n_parts, a.jobs, a.seed, a.local, a.max_tasks) is not None
    if a.cmd == "all" and complete:
        return verdict(ws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
