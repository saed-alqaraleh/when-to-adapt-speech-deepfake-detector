"""
Exploratory robustness package R1 for prevalence-gated adaptation (route 2).

The confirmatory test is finished (results/PREREGISTRATION_PGA.json -> PGA_VERDICT_df21.json).
Everything here is EXPLORATORY: it was planned after the confirmatory verdict was known. To limit
analytic freedom, every parameter and every success criterion is written once to
results/EXPLORATORY_PLAN_R1.json (with its sha256) before the first robustness task runs, and the
report applies exactly those criteria.

  Q1 guard     label-free consistency guard: adapt only if the gate is open AND
               bona_share(adapted score) >= rho * bona_share(frozen score), rho = 0.5.
  Q2 fixes     speaker-grouped reference split (calibration speakers unseen by the classifier);
               nuisance attribute projection (NAP) of the reference's between-speaker directions.
               PGA-R = NAP + grouped split + gate + guard.
  Q3 baselines nnPU (linear, Kiryo et al. 2017); CORAL and diagonal alignment of the frozen AASIST
               readout (Sun & Saenko 2016; AdaBN-style); supervised in-domain upper bound.
  Q4 cues      EER and anchor FRR/FAR by codec, source, subset, attack, vocoder, duration and
               leading-silence quintile on every clip outside each repetition's reference and
               traffic; AUC of duration / leading silence alone; rank correlation with duration.
  Q5 silence   the confirmatory protocol on silence-trimmed audio (pga run --dataset df21trim40).

The main cells reproduce the confirmatory cells exactly (same seeds, reference/traffic draws,
fold assignments): the report checks that `adapt` here equals `adapt_ssl1024` there.

    python -u -m pga_g.robust plan      --workspace W
    python -u -m pga_g.robust main      --workspace W --dataset df21 [--part k --n-parts N] [--local L]
    python -u -m pga_g.robust breakdown --workspace W --dataset df21 [--part k --n-parts N] [--local L]
    python -u -m pga_g.robust report    --workspace W --dataset df21
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pga_g import pilot as P

from . import pga as PG
from .chunked import log, run_chunked

RHO = 0.5
RHOS = (0.25, 0.5, 0.75, 0.9)
NAP_K = 32
BREAK_REPS = 10
NNPU = {"steps": 300, "lr": 1e-3, "weight_decay": 1e-4}
ADAPT_METHODS = ("adapt", "adapt_grp", "adapt_nap_grp", "nnpu", "coral", "diag")
MATCHED_DIMS = ("codec", "source", "subset", "dur_q", "lead_q")
SPOOF_DIMS = ("attack", "vocoder")
METRICS = ["eer", "frr", "far", "hter", "shortcut_auc"]

PLAN = {
    "title": "Exploratory robustness analyses R1 for prevalence-gated adaptation (PGA)",
    "status": ("exploratory: planned on 2026-10-21 after the confirmatory ASVspoof 2021 DF verdict "
               "(METHOD CONFIRMED) was known, before any robustness result was computed"),
    "splits": ("identical to the confirmatory cells (same seeds, reference/traffic draws and fold "
               "assignments); added methods draw from an independent random stream"),
    "gate": "estimated spoof share (frozen score) >= 0.80, as pre-registered",
    "guard": ("adapt only if the gate is open AND bona_share(adapted calibration reference vs traffic) "
              ">= rho x bona_share(frozen); rho = 0.5 primary; rho in 0.25, 0.5, 0.75, 0.9 reported"),
    "grouped_split": "reference split into train / calibration halves by speaker",
    "nap": ("remove the top k = min(32, n_speakers - 1) principal directions of the train-reference "
            "speaker means (speakers with >= 2 clips) from all features before scaling"),
    "pga_r": "PGA-R = NAP + grouped split + gate + guard (rho 0.5), ssl1024 features",
    "nnpu": ("linear nnPU (beta 0, gamma 1), sigmoid loss, class prior = frozen bona-share estimate "
             "clipped to [0.01, 0.99], standardised features, full-batch Adam lr 1e-3, 300 steps, "
             "weight decay 1e-4, same 5-fold cross-fitting as the confirmatory method"),
    "coral": ("frozen AASIST readout (score = w.emb160 + c, recovered by least squares from the stored "
              "scores and checked to 1e-2) on traffic-aligned emb160: CORAL (full covariance, "
              "eps = 1e-3 x mean variance) and diagonal mean/std alignment, both to ASVspoof 2019 LA "
              "eval statistics; alignment estimated on the unlabelled traffic only"),
    "oracle": ("supervised upper bound: logistic regression (C 0.1, balanced) on labelled traffic + "
               "train reference, 5-fold cross-fitted, ssl1024"),
    "breakdown": ("speaker-overlap mode, spoof shares 0.90 and natural, 10 repetitions; frozen vs adapted "
                  "(fold-averaged confirmatory model) on every clip outside the repetition's reference and "
                  "traffic; anchor threshold from the calibration reference; subgroups with >= 20 bona fide "
                  "and >= 20 spoof clips"),
    "criteria": {
        "fix_works": ("PGA-R, speaker-disjoint mode, at every spoof share: mean EER <= 1.02 x frozen and mean "
                      "anchor FRR <= frozen FRR + 0.01 (never worse than falling back to the frozen detector; "
                      "the frozen disjoint FRR itself exceeds 6% at some shares)"),
        "gain_kept": ("PGA-R, speaker-overlap mode: mean EER <= 0.85 x frozen at the natural share and 0.90; "
                      "<= 1.02 x frozen at 0.05, 0.20, 0.50; mean anchor FRR <= 0.06 everywhere"),
        "subgroup_flag": "adapted EER exceeds frozen EER by more than 1 point in a subgroup",
        "cue_flag": "a single cue (duration, leading silence) separates the classes with AUC > 0.70 (either direction)",
        "silence": "the gain survives trimming if the trimmed PGA (ssl1024) EER ratio at the natural share is <= 0.85",
    },
}

_G: dict = {}


# ---------------------------------------------------------------- plan (write-once)
def plan(ws: Path) -> str:
    path = ws / "results" / "EXPLORATORY_PLAN_R1.json"
    body = json.dumps(PLAN, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("sha256") != digest:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"plan": PLAN, "sha256": digest,
                               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    tmp.replace(path)
    log(f"exploratory plan written: {path} (sha256 {digest[:16]})")
    return digest


# ---------------------------------------------------------------- inputs
def _cache(src: Path, local: Path | None) -> Path:
    """Copy a large array from Drive to local disk once (fast, page-cached reads by every process)."""
    if local is None:
        return src
    dst = Path(local) / "cache" / src.parent.name / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    log(f"caching {src.name} ({src.stat().st_size / 1e9:.2f} GB) on local disk")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)
    return dst


def load(ws: Path, dataset: str, local: Path | None = None):
    if dataset.startswith("df21"):
        rd = ws / "runs" / f"{dataset}_v1"
        if not (rd / "_STAGE_SUCCESS.json").exists():
            raise SystemExit(f"{dataset} is not merged yet: run `python -m pga_g.df21 merge --wait`")
        text = ["utt_id", "speaker", "codec", "subset", "attack", "vocoder", "source"]
        df = pd.read_csv(rd / "scores.csv", usecols=text + ["label", "score", "orig_sr", "orig_len"],
                         dtype={c: str for c in text}, keep_default_na=False)
        df = df.rename(columns={"speaker": "group"})
        df["dur_s"] = df.orig_len / df.orig_sr
        trim = ws / "runs" / "df21trim40_v1"
        if dataset == "df21" and (trim / "_STAGE_SUCCESS.json").exists():
            t = pd.read_csv(trim / "scores.csv", usecols=["utt_id", "lead_silence_s"], dtype={"utt_id": str})
            df = df.merge(t, on="utt_id", how="left", validate="one_to_one")
        ssl = np.load(_cache(rd / "ssl_mean1024.npy", local), mmap_mode="r")
        emb = np.load(_cache(rd / "emb160.npy", local), mmap_mode="r")
    elif dataset == "itw":
        pin = ws / "results" / "pilot_inputs"
        df = pd.read_csv(pin / "itw_pilot.csv")
        ssl = np.load(_cache(pin / "itw_ssl1024.npy", local), mmap_mode="r")
        emb = np.load(_cache(pin / "itw_emb160.npy", local), mmap_mode="r")
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    df = df.reset_index(drop=True)
    if len(df) != ssl.shape[0] or len(df) != emb.shape[0]:
        raise SystemExit("scores and features are not row-aligned")
    return df, ssl, emb


def fit_readout(emb, score: np.ndarray, rng, n: int = 50000):
    """The AASIST output layer is linear in emb160: recover score = w.emb + c and check it."""
    idx = np.sort(rng.choice(score.size, min(n, score.size), replace=False))
    X = np.c_[np.asarray(emb[idx], np.float64), np.ones(idx.size)]
    coef, *_ = np.linalg.lstsq(X, score[idx].astype(np.float64), rcond=None)
    chk = np.sort(rng.choice(score.size, min(200000, score.size), replace=False))
    resid = float(np.abs(np.asarray(emb[chk], np.float64) @ coef[:-1] + coef[-1] - score[chk]).max())
    return coef[:-1], float(coef[-1]), resid


def source_stats(ws: Path):
    p = ws / "runs" / "score_v1" / "la19eval" / "emb160.npy"
    if not p.exists():
        return None
    E = np.asarray(np.load(p, mmap_mode="r"), np.float64)
    return {"mu": E.mean(0), "cov": np.cov(E, rowvar=False), "sd": E.std(0) + 1e-6, "n": int(E.shape[0])}


def _quintile_labels(x: np.ndarray) -> np.ndarray:
    out = np.full(x.size, "", dtype=object)
    ok = np.isfinite(x)
    if ok.sum() < 100:
        return out
    edges = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, x[ok], side="right")
    out[ok] = np.array([f"Q{k + 1}" for k in range(5)], dtype=object)[q]
    return out


def _init(df, ssl, emb, extras):
    _G.clear()
    _G.update(df=df, ssl=ssl, emb=emb, **extras)
    _G["lab"] = df.label.to_numpy()
    _G["score"] = df.score.to_numpy(np.float64)
    _G["gcode"] = pd.factorize(df.group.astype(str))[0]
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _extras(ws: Path, df: pd.DataFrame, emb) -> dict:
    rng = np.random.default_rng(7)
    w, c, resid = fit_readout(emb, df.score.to_numpy(np.float64), rng)
    ex = {"readout": (w, c), "readout_resid": resid, "readout_ok": resid <= 1e-2, "src": source_stats(ws)}
    cats = {}
    for d in ("codec", "source", "subset", "attack", "vocoder"):
        if d in df:
            cats[d] = df[d].astype(str).to_numpy(dtype=object)
    if "dur_s" in df:
        cats["dur_q"] = _quintile_labels(df.dur_s.to_numpy(np.float64))
    if "lead_silence_s" in df:
        cats["lead_q"] = _quintile_labels(df.lead_silence_s.to_numpy(np.float64))
    ex["cats"] = cats
    log(f"readout recovered from stored scores: max |residual| {resid:.2e} "
        f"({'ok' if ex['readout_ok'] else 'too large: CORAL/diag skipped'}); source statistics: "
        f"{'ASVspoof 2019 LA eval, n=' + str(ex['src']['n']) if ex['src'] else 'missing: CORAL/diag skipped'}")
    return ex


# ---------------------------------------------------------------- learners
class _Linear:
    def __init__(self, w, b):
        self.w, self.b = w, b

    def decision_function(self, X):
        return np.asarray(X, np.float64) @ self.w + self.b


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    clf.fit(X, y)
    return clf


def _lin(clf):
    """Effective (w, b) of a fitted model's decision function."""
    if isinstance(clf, _Linear):
        return clf.w, clf.b
    sc, lr = clf[0], clf[-1]
    w = lr.coef_[0] / sc.scale_
    return w, float(lr.intercept_[0] - np.dot(sc.mean_, w))


def _fit_nnpu(prior: float, steps: int = NNPU["steps"], lr: float = NNPU["lr"], wd: float = NNPU["weight_decay"]):
    """Linear non-negative PU learner (Kiryo et al. 2017), sigmoid loss, full-batch Adam."""
    def fit(X, y):
        X = np.asarray(X, np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        Pz, Uz = Z[y == 1], Z[y == 0]
        th = np.zeros(Z.shape[1] + 1)
        m, v = np.zeros_like(th), np.zeros_like(th)

        def sig(z):
            return 0.5 * (1 + np.tanh(0.5 * z))
        for t in range(1, steps + 1):
            w, b = th[:-1], th[-1]
            fp, fu = Pz @ w + b, Uz @ w + b
            sp, su = sig(fp), sig(fu)
            # risks with sigmoid loss l(z) = sig(-z): positive part pi*E_P[l(f)], negative part
            # E_U[l(-f)] - pi*E_P[l(-f)];  d sig(-f)/df = -sp(1-sp), d sig(f)/df = s(1-s)
            dpos = -sp * (1 - sp)
            g_pos = prior * np.r_[Pz.T @ dpos / len(Pz), dpos.mean()]
            du, dpn = su * (1 - su), sp * (1 - sp)
            g_neg = np.r_[Uz.T @ du / len(Uz) - prior * Pz.T @ dpn / len(Pz), du.mean() - prior * dpn.mean()]
            r_neg = su.mean() - prior * sp.mean()
            g = g_pos + g_neg if r_neg >= 0 else -g_neg
            g[:-1] += wd * th[:-1]
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            th -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        w = th[:-1] / sd
        return _Linear(w, float(th[-1] - np.dot(mu, w)))
    return fit


def crossfit(X_tr, X_cal, X_unl, rng, folds: int = 5, y_unl=None, fit=None):
    """Fold scheme and random-number use identical to pilot_pga_g.pu_adapted_scores (so the
    confirmatory scores are reproduced exactly). y_unl given: supervised (oracle) labels for the
    traffic. Returns (cal scores, cal folds, traffic scores, traffic folds, fold models)."""
    fit = fit or _fit_lr
    n_u, n_c = X_unl.shape[0], X_cal.shape[0]
    fold_u = rng.integers(0, folds, n_u)
    fold_c = np.arange(n_c) % folds
    rng.shuffle(fold_c)
    s_unl, s_cal, models = np.empty(n_u), np.empty(n_c), []
    for k in range(folds):
        m = fold_u != k
        X = np.vstack([X_tr, X_unl[m]])
        y = np.r_[np.ones(X_tr.shape[0]), np.zeros(int(m.sum())) if y_unl is None else y_unl[m]]
        clf = fit(X, y)
        s_unl[~m] = clf.decision_function(X_unl[~m])
        if (fold_c == k).any():                      # a tiny grouped calibration half can leave a fold empty
            s_cal[fold_c == k] = clf.decision_function(X_cal[fold_c == k])
        models.append(clf)
    return s_cal, fold_c, s_unl, fold_u, models


MIN_HALF = 30          # = pilot_pga_g.MIN_REF_TAIL: the smallest calibration set the estimators accept


def grouped_halves(ref: np.ndarray, gcode: np.ndarray, rng):
    """Split the reference into train / calibration by speaker (about half the clips each).
    Returns (train, calibration, how) with how in {grouped, rebalanced, random}.

    Deviation D1 (21 Oct 2026, after a crash on In-the-Wild, speaker-disjoint mode): when one
    speaker holds almost the whole reference, the first pass leaves fewer than MIN_HALF clips on
    one side. Only then are speakers re-drawn with a cap that keeps MIN_HALF clips on both sides
    ('rebalanced'), and if no speaker split can do that, random halves are used ('random').
    Every split the first pass could already use is unchanged, so finished tasks stay valid."""
    g = gcode[ref]
    uniq = np.unique(g)
    rng.shuffle(uniq)
    chosen, n = [], 0
    for u in uniq:
        if n >= ref.size / 2:
            break
        chosen.append(u)
        n += int((g == u).sum())
    tr = np.isin(g, chosen)
    if tr.all() or not tr.any():                     # a single speaker: fall back to random halves
        perm = rng.permutation(ref.size)
        tr = np.zeros(ref.size, bool)
        tr[perm[: ref.size // 2]] = True
        return ref[tr], ref[~tr], "random"
    if min(int(tr.sum()), int((~tr).sum())) >= MIN_HALF:
        return ref[tr], ref[~tr], "grouped"
    sizes = {u: int((g == u).sum()) for u in uniq}
    chosen, n = [], 0
    for u in rng.permutation(uniq):
        if n + sizes[u] <= ref.size - MIN_HALF:
            chosen.append(u)
            n += sizes[u]
            if n >= ref.size / 2:
                break
    tr = np.isin(g, chosen)
    if min(int(tr.sum()), int((~tr).sum())) >= MIN_HALF:
        return ref[tr], ref[~tr], "rebalanced"
    perm = rng.permutation(ref.size)
    tr = np.zeros(ref.size, bool)
    tr[perm[: ref.size // 2]] = True
    return ref[tr], ref[~tr], "random"


def nap_basis(X: np.ndarray, g: np.ndarray, k_max: int = NAP_K):
    uniq, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = np.flatnonzero(cnt >= 2)
    if keep.size < 2:
        return None
    means = np.vstack([X[inv == j].mean(0) for j in keep]).astype(np.float64)
    means -= means.mean(0)
    k = int(min(k_max, keep.size - 1))
    _, _, vt = np.linalg.svd(means, full_matrices=False)
    return vt[:k].T


def _project(X, U):
    return X - (X @ U) @ U.T


def _sqrtm(C, inv=False):
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * vals ** (-0.5 if inv else 0.5)) @ vecs.T


def coral_map(Eu: np.ndarray, src: dict):
    d = Eu.shape[1]
    mu_t, C_t = Eu.mean(0), np.cov(Eu, rowvar=False)
    eps_t = 1e-3 * np.trace(C_t) / d
    eps_s = 1e-3 * np.trace(src["cov"]) / d
    A = _sqrtm(C_t + eps_t * np.eye(d), inv=True) @ _sqrtm(src["cov"] + eps_s * np.eye(d))
    return lambda E: (E - mu_t) @ A + src["mu"]


def diag_map(Eu: np.ndarray, src: dict):
    mu_t, sd_t = Eu.mean(0), Eu.std(0) + 1e-6
    return lambda E: (E - mu_t) / sd_t * src["sd"] + src["mu"]


# ---------------------------------------------------------------- one repetition
def bona_share(s_cal: np.ndarray, s_u: np.ndarray) -> float:
    """Genuine-share estimate, or NaN when the calibration reference is too small for the
    best-bin estimator (a speaker-grouped split of a reference dominated by one speaker, as in
    In-the-Wild). NaN makes the guard fail, so that cell falls back to the frozen detector."""
    if s_cal.size <= P.MIN_REF_TAIL:
        return float("nan")
    return P.estimate_bona_share(s_cal, s_u)


def _draw(mode: str, pi: float, rng):
    """Exactly the reference/traffic draw of pga_g.pga.cell."""
    df, lab = _G["df"], _G["lab"]
    ref, pool = PG._split(df, mode, rng, PG.N_REF)
    b_pool, s_pool = pool[lab[pool] == 1], pool[lab[pool] == 0]
    n = min(12000, int(min(b_pool.size / max(1 - pi, 1e-9), s_pool.size / max(pi, 1e-9))))
    b_idx = rng.choice(b_pool, int(round(n * (1 - pi))), replace=False)
    s_idx = rng.choice(s_pool, int(round(n * pi)), replace=False)
    ref = rng.permutation(ref)
    half = ref.size // 2
    return ref, b_idx, s_idx, ref[:half], ref[half:]


def _rngs(mode, pi, rep, seed):
    mid = 0 if mode == "overlap" else 1
    return (np.random.default_rng([seed, 23, mid, int(round(pi * 1000)), rep]),
            np.random.default_rng([seed, 24, mid, int(round(pi * 1000)), rep]))


def _ssl(idx):
    return np.asarray(_G["ssl"][idx], dtype=np.float32)


def main_cell(mode: str, pi: float, rep: int, seed: int):
    rng, rng2 = _rngs(mode, pi, rep, seed)
    ref, b_idx, s_idx, ref_tr, ref_cal = _draw(mode, pi, rng)
    traffic, nb = np.r_[b_idx, s_idx], b_idx.size
    frozen, gcode = _G["score"], _G["gcode"]
    b_fr = P.estimate_bona_share(frozen[ref_cal], frozen[traffic])
    pi_hat = 1 - b_fr
    rows = []

    def record(method, s_cal, s_u, folds=None, extra=None):
        sb, ss = s_u[:nb], s_u[nb:]
        if folds is None:
            op = P.op_point(P.conformal_bona_threshold(s_cal, P.ALPHA), sb, ss)
        else:
            op = P.foldwise_anchor_op(s_cal, folds[0], sb, folds[1][:nb], ss, folds[1][nb:])
        b_m = bona_share(s_cal, s_u)
        rows.append({"mode": mode, "pi": pi, "rep": rep, "method": method, "n_traffic": int(traffic.size),
                     "n_cal": int(s_cal.size),
                     "eer": P.eer(sb, ss), "frr": op.frr, "far": op.far, "hter": op.hter,
                     "shortcut_auc": P.auc(s_cal, sb), "pi_hat_frozen": pi_hat,
                     "gate_open": bool(pi_hat >= PG.GATE), "bona_share": b_m,
                     "guard_ratio": b_m / max(b_fr, 1e-9), **(extra or {})})

    record("frozen", frozen[ref_cal], frozen[traffic])
    Xt, Xc, Xu = _ssl(ref_tr), _ssl(ref_cal), _ssl(traffic)
    a = crossfit(Xt, Xc, Xu, rng)                         # == confirmatory adapt_ssl1024
    record("adapt", a[0], a[2], (a[1], a[3]))

    gtr, gcal, how = grouped_halves(np.r_[ref_tr, ref_cal], gcode, rng2)
    Gt, Gc = _ssl(gtr), _ssl(gcal)
    n_spk = int(np.unique(gcode[gtr]).size)
    a = crossfit(Gt, Gc, Xu, rng2)
    gx = {"n_ref_spk_train": n_spk, "grp_split": how, "n_cal_grp": int(gcal.size)}
    record("adapt_grp", a[0], a[2], (a[1], a[3]), gx)
    U = nap_basis(Gt, gcode[gtr])
    if U is not None:
        a = crossfit(_project(Gt, U), _project(Gc, U), _project(Xu, U), rng2)
    else:
        a = crossfit(Gt, Gc, Xu, rng2)
    record("adapt_nap_grp", a[0], a[2], (a[1], a[3]),
           {**gx, "nap_k": 0 if U is None else int(U.shape[1])})

    a = crossfit(Xt, Xc, Xu, rng2, fit=_fit_nnpu(float(np.clip(b_fr, 0.01, 0.99))))
    record("nnpu", a[0], a[2], (a[1], a[3]))

    if _G.get("readout_ok") and _G.get("src") is not None:
        w, c = _G["readout"]
        Eu, Ec = (np.asarray(_G["emb"][i], np.float64) for i in (traffic, ref_cal))
        for name, mk in (("coral", coral_map), ("diag", diag_map)):
            f = mk(Eu, _G["src"])
            record(name, f(Ec) @ w + c, f(Eu) @ w + c)

    y = np.r_[np.ones(nb), np.zeros(traffic.size - nb)]
    a = crossfit(Xt, Xc, Xu, rng2, y_unl=y)
    record("oracle_sup", a[0], a[2], (a[1], a[3]))
    return rows


def _score_rows(idx: np.ndarray, w: np.ndarray, b: float, block: int = 65536) -> np.ndarray:
    out = np.empty(idx.size)
    w32 = w.astype(np.float32)
    for i in range(0, idx.size, block):
        out[i:i + block] = np.asarray(_G["ssl"][idx[i:i + block]], np.float32) @ w32 + b
    return out


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def break_cell(pi: float, rep: int, seed: int):
    mode = "overlap"
    rng, _ = _rngs(mode, pi, rep, seed)
    ref, b_idx, s_idx, ref_tr, ref_cal = _draw(mode, pi, rng)
    traffic = np.r_[b_idx, s_idx]
    Xc = _ssl(ref_cal)
    _, _, _, _, models = crossfit(_ssl(ref_tr), Xc, _ssl(traffic), rng)
    lins = [_lin(m) for m in models]
    w_bar = np.mean([l[0] for l in lins], axis=0)
    b_bar = float(np.mean([l[1] for l in lins]))
    lab, frozen, cats = _G["lab"], _G["score"], _G["cats"]
    rest = np.setdiff1d(np.arange(lab.size), np.r_[ref, traffic])
    lr = lab[rest]
    rows = []

    def add(method, dim, group, sb, ss, tau):
        if sb.size < 20 or ss.size < 20:
            return
        rows.append({"pi": pi, "rep": rep, "method": method, "dim": dim, "group": group, "metric": "eer",
                     "n_bona": int(sb.size), "n_spoof": int(ss.size), "value": P.eer(sb, ss),
                     "frr_anchor": float(np.mean(sb < tau)), "far_anchor": float(np.mean(ss >= tau))})

    def stat(method, dim, metric, value):
        rows.append({"pi": pi, "rep": rep, "method": method, "dim": dim, "group": "all", "metric": metric,
                     "value": value})

    dur = _G["df"].dur_s.to_numpy(np.float64)[rest] if "dur_s" in _G["df"] else None
    for method, sr, sc in (("frozen", frozen[rest], frozen[ref_cal]),
                           ("adapt", _score_rows(rest, w_bar, b_bar), Xc.astype(np.float64) @ w_bar + b_bar)):
        tau = P.conformal_bona_threshold(sc, P.ALPHA)
        b_all, s_all = sr[lr == 1], sr[lr == 0]
        add(method, "all", "all", b_all, s_all, tau)
        for dim in MATCHED_DIMS:
            if dim not in cats:
                continue
            col = cats[dim][rest]
            for v in sorted(set(col[col != ""])):
                m = col == v
                add(method, dim, v, sr[m & (lr == 1)], sr[m & (lr == 0)], tau)
        for dim in SPOOF_DIMS:
            if dim not in cats:
                continue
            col = cats[dim][rest]
            for v in sorted(set(col[lr == 0])):
                add(method, dim, v, b_all, sr[(col == v) & (lr == 0)], tau)
        if dur is not None:
            stat(method, "dur_s", "spearman_bona", _spearman(sr[lr == 1], dur[lr == 1]))
            stat(method, "dur_s", "spearman_spoof", _spearman(sr[lr == 0], dur[lr == 0]))
    if dur is not None:
        stat("cue", "dur_s", "auc_bona_vs_spoof", P.auc(dur[lr == 1], dur[lr == 0]))
    if "lead_silence_s" in _G["df"]:
        ls = _G["df"].lead_silence_s.to_numpy(np.float64)[rest]
        ok = np.isfinite(ls)
        if ok.sum() > 100:
            stat("cue", "lead_silence_s", "auc_bona_vs_spoof", P.auc(ls[ok & (lr == 1)], ls[ok & (lr == 0)]))
    return rows


# ---------------------------------------------------------------- commands
def _shares(df):
    return sorted(set(PG.SHARES) | {PG.natural_share(df)})


def run_main(ws, dataset, part, n_parts, jobs, seed, local=None, max_tasks=None) -> int:
    digest = plan(ws)
    df, ssl, emb = load(ws, dataset, local)
    ex = _extras(ws, df, emb)
    tasks = [(m, pi, r, seed) for m in ("overlap", "disjoint") for pi in _shares(df) for r in range(PG.REPS)]
    run_chunked(main_cell, tasks, ws / "results" / f"robust_{dataset}", _init, (df, ssl, emb, ex),
                jobs=jobs, part=part, n_parts=n_parts, flush_every=10, label=f"robust main {dataset}",
                max_tasks=max_tasks,
                config={"dataset": dataset, "plan_sha256": digest, "reps": PG.REPS, "n_ref": PG.N_REF,
                        "gate": PG.GATE, "seed": seed, "n_rows": int(len(df))})
    return 0


def run_breakdown(ws, dataset, part, n_parts, jobs, seed, local=None, max_tasks=None) -> int:
    if not dataset.startswith("df21"):
        raise SystemExit("the breakdown needs codec/source/attack labels (ASVspoof 2021 DF)")
    digest = plan(ws)
    df, ssl, emb = load(ws, dataset, local)
    ex = _extras(ws, df, emb)
    shares = sorted({0.90, PG.natural_share(df)})
    tasks = [(pi, r, seed) for pi in shares for r in range(BREAK_REPS)]
    run_chunked(break_cell, tasks, ws / "results" / f"robust_break_{dataset}", _init, (df, ssl, emb, ex),
                jobs=jobs, part=part, n_parts=n_parts, flush_every=2, label=f"robust breakdown {dataset}",
                max_tasks=max_tasks,
                config={"dataset": dataset, "plan_sha256": digest, "reps": BREAK_REPS, "n_ref": PG.N_REF,
                        "seed": seed, "n_rows": int(len(df)), "has_lead_silence": "lead_silence_s" in df})
    return 0


def variants(res: pd.DataFrame) -> pd.DataFrame:
    """Gated / guarded variants assembled from per-method rows (frozen where not adapted)."""
    key = ["mode", "pi", "rep"]
    fr = res[res.method == "frozen"].set_index(key)
    specs = [("frozen", "frozen", False, None)]
    for m in ADAPT_METHODS:
        specs += [(f"{m} (always)", m, False, None), (f"{m} gated", m, True, None)]
        specs += [(f"{m} gated+guard rho={r:g}", m, True, r) for r in RHOS]
    specs += [("oracle_sup", "oracle_sup", False, None)]
    out = []
    for name, m, gated, rho in specs:
        a = res[res.method == m].set_index(key)
        if a.empty:
            continue
        a = a.join(fr[METRICS], rsuffix="_fr", how="inner")
        use = np.ones(len(a), bool) if m != "frozen" else np.zeros(len(a), bool)
        if gated:
            use &= a.gate_open.astype(bool).to_numpy()
        if rho is not None:
            use &= (a.guard_ratio >= rho).to_numpy()
        v = pd.DataFrame({k: np.where(use, a[k], a[f"{k}_fr"]) if m != "frozen" else a[k] for k in METRICS},
                         index=a.index)
        v["adapted"] = use
        v["variant"] = name
        out.append(v.reset_index())
    return pd.concat(out, ignore_index=True)


ALIASES = {"PGA (confirmatory)": "adapt gated", "PGA+guard": f"adapt gated+guard rho={RHO:g}",
           "PGA-grp+guard": f"adapt_grp gated+guard rho={RHO:g}", "PGA-R": f"adapt_nap_grp gated+guard rho={RHO:g}",
           "nnPU gated": "nnpu gated", "CORAL gated": "coral gated", "diag gated": "diag gated"}


def _ratio_table(v: pd.DataFrame, variant: str) -> pd.DataFrame:
    g = v.groupby(["variant", "mode", "pi"])[["eer", "frr", "hter"]].mean()
    f = g.xs("frozen", level="variant")
    x = g.xs(variant, level="variant")
    t = x.join(f, rsuffix="_frozen")
    t["eer_ratio"] = t.eer / t.eer_frozen
    return t


def report(ws: Path, dataset: str) -> int:
    rdir = ws / "results" / f"robust_{dataset}"
    if not (rdir / "results.csv").exists():
        raise SystemExit(f"{rdir}/results.csv missing: run `robust main` (all parts) first")
    res = pd.read_csv(rdir / "results.csv")
    v = variants(res)
    nat = max(res.pi.unique(), key=lambda p: (p not in PG.SHARES, p))
    rep: dict = {"dataset": dataset, "status": "exploratory (EXPLORATORY_PLAN_R1.json)",
                 "deviations": {"D1": ("speaker-grouped split: when one speaker held almost the whole reference "
                                       "(fewer than 30 calibration clips), speakers were re-drawn with a cap, else "
                                       "random halves were used; added 21 Oct 2026 after a crash, before any "
                                       "robustness result was read")},
                 "plan_sha256": json.loads((rdir / "config.json").read_text()).get("plan_sha256"),
                 "natural_share": float(nat)}
    lines = []
    g = res[res.method == "adapt_grp"]
    if "grp_split" in g:
        rep["grp_split_counts"] = {str(k): int(x) for k, x in
                                   g.grp_split.fillna("grouped or random (before D1)").value_counts().items()}

    # consistency with the confirmatory run
    conf = ws / "results" / f"pga_{dataset}" / "results.csv"
    if conf.exists():
        c = pd.read_csv(conf)
        c = c[c.variant == "adapt_ssl1024"].set_index(["mode", "pi", "rep"]).eer
        a = res[res.method == "adapt"].set_index(["mode", "pi", "rep"]).eer
        j = a.to_frame("robust").join(c.to_frame("confirmatory"), how="inner")
        rep["consistency"] = {"cells": int(len(j)), "max_abs_eer_diff": float((j.robust - j.confirmatory).abs().max())}

    # main table
    names = ["frozen", "adapt (always)"] + list(ALIASES.values()) + ["oracle_sup"]
    names = [n for n in names if n in set(v.variant)]
    tab = v[v.variant.isin(names)].groupby(["mode", "pi", "variant"])[["eer", "frr", "hter", "adapted"]].mean()
    inv = {b: a for a, b in ALIASES.items()}
    tab = tab.rename(index=lambda n: inv.get(n, n), level="variant")
    lines += ["Mean EER / anchor FRR / HTER (%) and share of repetitions adapted", (tab * 100).round(2).to_string(), ""]

    # criteria
    crit = {}
    for alias in ("PGA-R", "PGA+guard", "PGA-grp+guard", "PGA (confirmatory)"):
        name = ALIASES[alias]
        if name not in set(v.variant):
            continue
        t = _ratio_table(v, name)
        dj, ov = t.xs("disjoint", level="mode"), t.xs("overlap", level="mode")
        fix = bool((dj.frr <= dj.frr_frozen + 0.01).all() and (dj.eer_ratio <= 1.02).all())
        gain = all(ov.loc[p, "eer_ratio"] <= 0.85 for p in (nat, 0.90) if p in ov.index)
        safe = all(ov.loc[p, "eer_ratio"] <= 1.02 for p in (0.05, 0.20, 0.50) if p in ov.index)
        kept = bool(gain and safe and (ov.frr <= 0.06).all())
        crit[alias] = {"fix_works_disjoint": fix, "gain_kept_overlap": kept,
                       "disjoint_max_frr": float(dj.frr.max()),
                       "disjoint_max_frr_excess_over_frozen": float((dj.frr - dj.frr_frozen).max()),
                       "disjoint_max_eer_ratio": float(dj.eer_ratio.max()),
                       "overlap_eer_ratio_natural": float(ov.loc[nat, "eer_ratio"]) if nat in ov.index else None,
                       "overlap_max_frr": float(ov.frr.max())}
    rep["criteria"] = crit
    na = res[res.guard_ratio.isna()].groupby(["method", "mode"]).size()
    rep["guard_unavailable_cells"] = {f"{m}|{mode}": int(k) for (m, mode), k in na.items()}
    if "n_cal" in res:
        rep["min_calibration_size"] = {m: int(x) for m, x in res.groupby("method").n_cal.min().dropna().items()}

    # guard sweep
    sweep = {}
    for m in ("adapt", "adapt_nap_grp"):
        for r in RHOS:
            name = f"{m} gated+guard rho={r:g}"
            if name in set(v.variant):
                t = _ratio_table(v, name)
                sweep[name] = {"max_eer_ratio_disjoint": float(t.xs("disjoint", level="mode").eer_ratio.max()),
                               "max_frr_disjoint": float(t.xs("disjoint", level="mode").frr.max()),
                               "eer_ratio_overlap_natural": float(t.xs("overlap", level="mode").eer_ratio.get(nat, np.nan))}
    rep["guard_sweep"] = sweep
    lines += ["Guard sweep", pd.DataFrame(sweep).T.round(3).to_string(), ""]

    # baselines at the natural share and 0.90 (overlap) for the paper table
    base = {}
    for alias in ("adapt (always)", "nnpu (always)", "coral (always)", "diag (always)", "oracle_sup"):
        if alias in set(v.variant):
            t = _ratio_table(v, alias)
            base[alias] = {f"{mode}|{pi:g}": float(t.loc[(mode, pi), "eer_ratio"]) for mode, pi in t.index}
    rep["eer_ratio_vs_frozen"] = base

    # breakdown
    bpath = ws / "results" / f"robust_break_{dataset}" / "results.csv"
    if bpath.exists():
        b = pd.read_csv(bpath)
        e = b[b.metric == "eer"].groupby(["pi", "dim", "group", "method"])[
            ["value", "frr_anchor", "far_anchor", "n_bona", "n_spoof"]].mean().unstack("method")
        diff = e[("value", "adapt")] - e[("value", "frozen")]
        flagged = diff[diff > 0.01]
        rep["breakdown"] = {
            "subgroups": int(len(e)), "flagged_adapted_worse_by_1pt": {f"{pi:g}|{d}|{g}": float(x)
                                                                      for (pi, d, g), x in flagged.items()},
            "max_anchor_frr_adapted": float(e[("frr_anchor", "adapt")].max()),
            "max_anchor_frr_frozen": float(e[("frr_anchor", "frozen")].max())}
        st = b[b.metric != "eer"].groupby(["method", "dim", "metric"]).value.mean()
        rep["cues"] = {f"{m}|{d}|{k}": float(x) for (m, d, k), x in st.items()}
        rep["cue_flags"] = {f"{d}": float(x) for (m, d, k), x in st.items()
                            if m == "cue" and max(x, 1 - x) > 0.70}
        tbl = e[[("value", "frozen"), ("value", "adapt"), ("frr_anchor", "frozen"), ("frr_anchor", "adapt"),
                 ("n_bona", "adapt"), ("n_spoof", "adapt")]].copy()
        tbl.columns = ["eer_frozen", "eer_adapt", "frr_frozen", "frr_adapt", "n_bona", "n_spoof"]
        for k in ["eer_frozen", "eer_adapt", "frr_frozen", "frr_adapt"]:
            tbl[k] = (tbl[k] * 100).round(2)
        tbl[["n_bona", "n_spoof"]] = tbl[["n_bona", "n_spoof"]].round(0)
        lines += ["Breakdown on clips outside reference and traffic (overlap mode, %)", tbl.to_string(), "",
                  "Cues and rank correlations", st.round(3).to_string(), ""]

    # silence trimming
    tpath = ws / "results" / "pga_df21trim40" / "results.csv"
    if dataset == "df21" and tpath.exists():
        t = pd.read_csv(tpath)
        t = t[t["mode"] == "overlap"].groupby(["pi", "variant"]).eer.mean().unstack()
        u = pd.read_csv(ws / "results" / "pga_df21" / "results.csv") if conf.exists() else None
        sil = {}
        for p in t.index:
            sil[f"{p:g}"] = {"trimmed_frozen_eer": float(t.loc[p, "frozen"]),
                             "trimmed_pga_eer": float(t.loc[p, "pga_ssl1024"]),
                             "trimmed_ratio": float(t.loc[p, "pga_ssl1024"] / t.loc[p, "frozen"])}
            if u is not None:
                uu = u[(u["mode"] == "overlap") & (u.pi == p)].groupby("variant").eer.mean()
                sil[f"{p:g}"]["untrimmed_ratio"] = float(uu["pga_ssl1024"] / uu["frozen"])
        key = f"{nat:g}"
        rep["silence"] = {"by_share": sil,
                          "gain_survives_trimming": bool(key in sil and sil[key]["trimmed_ratio"] <= 0.85)}
        lines += ["Silence-trimmed confirmatory protocol (overlap, EER %)", (t * 100).round(2).to_string(), ""]

    (ws / "results" / f"ROBUST_R1_{dataset}.json").write_text(json.dumps(rep, indent=2, default=float))
    (rdir / "REPORT.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(json.dumps({k: rep[k] for k in ("consistency", "criteria") if k in rep}, indent=2, default=float))
    log(f"-> {ws / 'results' / f'ROBUST_R1_{dataset}.json'} and {rdir / 'REPORT.txt'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "main", "breakdown", "report"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--dataset", choices=["df21", "itw", "df21trim40"], default="df21")
    ap.add_argument("--local", default=None, help="local disk for a cached copy of the feature arrays")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-tasks", type=int, default=None)
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    if a.cmd == "plan":
        plan(ws)
        return 0
    if a.cmd == "main":
        return run_main(ws, a.dataset, a.part, a.n_parts, a.jobs, a.seed, a.local, a.max_tasks)
    if a.cmd == "breakdown":
        return run_breakdown(ws, a.dataset, a.part, a.n_parts, a.jobs, a.seed, a.local, a.max_tasks)
    return report(ws, a.dataset)


if __name__ == "__main__":
    sys.exit(main())
