#!/usr/bin/env python3
"""
Pilot for bona-fide-anchored, prior-robust operating-point transfer (working name: PGA_G)
for speech deepfake countermeasures (CMs).

v2.0.0: cells run in parallel worker processes (--jobs) with per-cell seeds, so results are
identical for any number of workers (they differ from v1, which used one random stream).

Modes
-----
1) Synthetic sanity check (no data needed):
       python -m pga_g.pilot --synthetic [--overlap] [--undetectable 0.1]
2) Stage A kill test on real scores from a frozen CM:
       python -m pga_g.pilot --scores itw_scores.csv --source-threshold 4.97
   CSV columns: score (higher = more bona fide), label (1 = bona fide, 0 = spoof),
                group (speaker ID; used for speaker-disjoint reference sampling).
3) Stage B kill test (target-adapted PU score in embedding space):
       python -m pga_g.pilot --scores itw_scores.csv --embeddings itw_emb.npy --stage-b
   itw_emb.npy: float array, one row per CSV row (e.g. the CM's penultimate readout).
   Synthetic smoke test of this path:  python -m pga_g.pilot --synthetic --stage-b

Stage A methods, compared at every (target spoof prior, reference size, repetition):
  SRC    : threshold transferred from the labelled source domain (deployment default)
  GMM2   : fully unsupervised 2-component Gaussian mixture on target scores, equal-posterior
           threshold (Brummer & Garcia-Romero-style unsupervised calibration; no reference)
  BONA   : conformal (1-alpha) bona fide quantile from the target reference -> FRR <= alpha
  PGA_G   : BONA + score-space mixture-proportion estimate + point-estimate HTER minimiser
           (ablation; known to be unstable at low spoof prevalence)
  PGA-P : BONA + bootstrap-pessimistic refinement; may only LOWER the threshold, and only
           when the pessimistic estimate of added spoof acceptance is smaller than the
           genuine trials recovered -> FRR <= alpha is preserved
  ORACLE : EER threshold on the labelled target (not deployable; reference only)

Stage B (the research question): train a probabilistic classifier "reference bona fide vs
unlabelled traffic" on embeddings with cross-fitting (Elkan & Noto non-traditional
classifier; its output is monotone in the bona fide posterior under SCAR), then
  - EER of the adapted score vs the frozen score       (ranking repair)
  - spoof-prevalence error of score-space MPE on each   (label-free monitoring)
  - shortcut AUC: held-out reference bona fide vs target bona fide on the adapted score
    (should be ~0.5; >>0.5 means the classifier learned reference speakers/channel)

Labels on the target are used ONLY to score the methods afterwards.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

__version__ = "2.1.0"   # 2.1: Stage A speaker-overlap mode (primary); 2.0: per-cell seeds + parallel cells

PRIORS = (0.01, 0.05, 0.20, 0.37, 0.50, 0.80, 0.96)   # target spoof prior
REF_SIZES = (200, 1000)
ALPHA = 0.05            # FRR budget for BONA, also the FRR cap applied inside PGA_G
DELTA = 0.05            # confidence level used in the best-bin search
MIN_REF_TAIL = 30       # min. reference points above a candidate bin
PI_FLOOR = 0.02         # below this estimated spoof prior, PGA_G returns the BONA threshold


# --------------------------------------------------------------------------- metrics
@dataclass
class OpPoint:
    frr: float
    far: float

    @property
    def hter(self) -> float:
        return 0.5 * (self.frr + self.far)


def op_point(tau: float, s_bona: np.ndarray, s_spoof: np.ndarray) -> OpPoint:
    frr = float(np.mean(s_bona < tau)) if s_bona.size else float("nan")
    far = float(np.mean(s_spoof >= tau)) if s_spoof.size else float("nan")
    return OpPoint(frr, far)


def eer_threshold(s_bona: np.ndarray, s_spoof: np.ndarray) -> float:
    grid = np.unique(np.concatenate([s_bona, s_spoof]))
    frr = np.searchsorted(np.sort(s_bona), grid, side="left") / s_bona.size
    far = 1.0 - np.searchsorted(np.sort(s_spoof), grid, side="left") / s_spoof.size
    return float(grid[np.argmin(np.abs(frr - far))])


# --------------------------------------------------------------------------- methods
def conformal_bona_threshold(ref: np.ndarray, alpha: float) -> float:
    """Largest tau with E[FRR] <= alpha for an exchangeable new bona fide trial."""
    n = ref.size
    k = math.floor(alpha * (n + 1))          # number of reference points allowed below tau
    if k < 1:
        return -np.inf                       # reference too small for this alpha: accept all
    return float(np.sort(ref)[k - 1])        # reject s < tau  ->  at most k-1 ref points below


def upper_tail(x_sorted: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 1.0 - np.searchsorted(x_sorted, t, side="left") / x_sorted.size


def estimate_bona_share(ref: np.ndarray, unl: np.ndarray, delta: float = DELTA) -> float:
    """
    1-D best-bin mixture-proportion estimate of kappa = P(bona fide) in the unlabelled pool.
    For every t:  q_U(t) = kappa*q_B(t) + (1-kappa)*q_S(t) >= kappa*q_B(t),
    so q_U(t)/q_B(t) is an upper bound on kappa, tight where spoofs are absent.
    The bin is chosen by minimising a DKW-style upper confidence bound (Garg et al., 2021).
    """
    rs, us = np.sort(ref), np.sort(unl)
    cand = np.quantile(rs, np.linspace(0.0, 1.0 - MIN_REF_TAIL / rs.size, 200))
    qb, qu = upper_tail(rs, cand), upper_tail(us, cand)
    eps_b = math.sqrt(math.log(4 / delta) / (2 * rs.size))
    eps_u = math.sqrt(math.log(4 / delta) / (2 * us.size))
    ok = qb - eps_b > 0
    if not ok.any():
        return 1.0
    ucb = (qu[ok] + eps_u) / (qb[ok] - eps_b)
    j = int(np.argmin(ucb))
    return float(np.clip(qu[ok][j] / qb[ok][j], 0.0, 1.0))


def pga_g_threshold(ref: np.ndarray, unl: np.ndarray, alpha_cap: float = ALPHA):
    """Returns (tau, estimated spoof prior, estimated FAR at tau)."""
    kappa = estimate_bona_share(ref, unl)
    pi_hat = 1.0 - kappa
    tau_bona = conformal_bona_threshold(ref, alpha_cap)
    if pi_hat < PI_FLOOR:
        return tau_bona, pi_hat, float("nan")
    rs, us = np.sort(ref), np.sort(unl)
    grid = np.unique(np.concatenate([rs, us]))
    frr_hat = np.searchsorted(rs, grid, side="left") / rs.size
    far_hat = (upper_tail(us, grid) - kappa * upper_tail(rs, grid)) / pi_hat
    far_hat = np.clip(far_hat, 0.0, 1.0)
    far_hat = np.maximum.accumulate(far_hat[::-1])[::-1]       # must be non-increasing in tau
    feasible = grid <= tau_bona                                # respect the FRR cap
    if not feasible.any():
        return tau_bona, pi_hat, float("nan")
    hter_hat = np.where(feasible, 0.5 * (frr_hat + far_hat), np.inf)
    j = int(np.argmin(hter_hat))
    return float(grid[j]), pi_hat, float(far_hat[j])


def pga_g_pessimistic_threshold(ref: np.ndarray, unl: np.ndarray, rng: np.random.Generator,
                               alpha_cap: float = ALPHA, alpha_max: float = ALPHA,
                               n_boot: int = 40, q: float = 0.9):
    """
    Uncertainty-aware PGA_G: the spoof acceptance curve is replaced by its bootstrap
    q-quantile (pessimistic FAR), and the threshold may leave the bona fide anchor only
    when that pessimistic curve still predicts a lower HTER. At low spoof prevalence the
    FAR curve is poorly identified, the pessimistic curve is high, and the rule stays on
    the anchor; at moderate/high prevalence it moves toward the HTER-optimal threshold.
    """
    tau_bona = conformal_bona_threshold(ref, alpha_cap)
    tau_max = conformal_bona_threshold(ref, alpha_max)   # operator's FRR budget
    rs = np.sort(ref)
    grid = np.unique(np.concatenate([rs, np.sort(unl), [tau_bona]]))
    grid = grid[grid <= tau_max]
    if grid.size == 0:
        return tau_bona, float("nan")
    k0 = int(np.searchsorted(grid, tau_bona))           # index of the anchor
    # added spoof acceptance when moving the threshold from the anchor to grid[k]
    # (negative = spoofs removed when the threshold is raised)
    added = np.empty((n_boot, grid.size))
    pis = np.empty(n_boot)
    for b in range(n_boot):
        rb = rng.choice(ref, ref.size, replace=True)
        ub = rng.choice(unl, unl.size, replace=True)
        kappa = estimate_bona_share(rb, ub)
        pis[b] = 1.0 - kappa
        if pis[b] < PI_FLOOR:
            # spoof acceptance is not identifiable when (almost) no spoofs are detected:
            # be pessimistic: lowering adds every spoof, raising removes none
            added[b] = np.where(np.arange(grid.size) < k0, 1.0, 0.0)
            continue
        f = (upper_tail(np.sort(ub), grid) - kappa * upper_tail(np.sort(rb), grid)) / pis[b]
        f = np.clip(f, 0.0, 1.0)
        f = np.maximum.accumulate(f[::-1])[::-1]
        added[b] = f - f[k0]
    added_pess = np.quantile(added, q, axis=0)
    frr_hat = np.searchsorted(rs, grid, side="left") / rs.size
    frr_saved = frr_hat[k0] - frr_hat               # >0 when lowering tau, <0 when raising it
    gain = 0.5 * (frr_saved - added_pess)           # pessimistic HTER reduction vs the anchor
    j = int(np.argmax(gain))
    tau = float(grid[j]) if gain[j] > 0 else tau_bona
    return tau, float(np.median(pis))


def gmm2_threshold(unl: np.ndarray, seed: int = 0) -> float:
    g = GaussianMixture(2, random_state=seed).fit(unl.reshape(-1, 1))
    hi = int(np.argmax(g.means_.ravel()))                      # bona fide = higher mean
    grid = np.linspace(unl.min(), unl.max(), 2000).reshape(-1, 1)
    post = g.predict_proba(grid)[:, hi]
    lo_mean, hi_mean = np.sort(g.means_.ravel())
    inside = (grid.ravel() > lo_mean) & (grid.ravel() < hi_mean)
    if not inside.any():
        return float(np.median(unl))
    j = np.argmin(np.abs(post[inside] - 0.5))
    return float(grid.ravel()[inside][j])


# --------------------------------------------------------------------------- data
def synthetic_scores(rng: np.random.Generator, n_bona=20000, n_spoof=20000, undetectable=0.0,
                     hard_mean=-1.0, hard_frac=0.35):
    """
    Logit-difference scores mimicking a source-trained CM on an out-of-domain target:
    the target bona fide mode has drifted far below the source threshold (4.97),
    spoofs are a mixture of an easy and a hard (overlapping) attack family.
    'undetectable' is the share of spoofs drawn from the bona fide score distribution.
    """
    bona = rng.normal(1.5, 2.2, n_bona)
    n_und = int(round(undetectable * n_spoof))
    n_hard = int(hard_frac * (n_spoof - n_und))
    n_easy = n_spoof - n_und - n_hard
    spoof = np.concatenate([
        rng.normal(-6.0, 1.5, n_easy),
        rng.normal(hard_mean, 2.0, n_hard),
        rng.normal(1.5, 2.2, n_und),
    ])
    groups_b = rng.integers(0, 60, n_bona)
    groups_s = rng.integers(0, 60, n_spoof)
    return pd.DataFrame({
        "score": np.concatenate([bona, spoof]),
        "label": np.concatenate([np.ones(n_bona, int), np.zeros(n_spoof, int)]),
        "group": np.concatenate([groups_b, groups_s]),
    })


def speaker_disjoint_split(df: pd.DataFrame, n_ref: int, rng: np.random.Generator):
    """Reference = bona fide from a random speaker subset; pool = all clips of other speakers."""
    groups = df["group"].unique()
    rng.shuffle(groups)
    ref_groups, ref = [], []
    bona = df[df["label"] == 1]
    for g in groups:
        ref_groups.append(g)
        ref = bona[bona["group"].isin(ref_groups)]["score"].to_numpy()
        if ref.size >= n_ref:
            break
    ref = rng.choice(ref, size=min(n_ref, ref.size), replace=False)
    pool = df[~df["group"].isin(ref_groups)]
    return ref, pool


def speaker_overlap_split(df: pd.DataFrame, n_ref: int, rng: np.random.Generator):
    """Reference = random bona fide clips from all speakers (enrolled-user deployment: the
    reference is exchangeable with the genuine traffic); pool = every other clip."""
    bona_idx = df.index[df["label"] == 1].to_numpy()
    ref_idx = rng.choice(bona_idx, size=min(n_ref, bona_idx.size), replace=False)
    return df.loc[ref_idx, "score"].to_numpy(), df.drop(index=ref_idx)


def resample_prior(pool: pd.DataFrame, pi_spoof: float, rng: np.random.Generator, n_max=12000):
    b = pool[pool["label"] == 1]["score"].to_numpy()
    s = pool[pool["label"] == 0]["score"].to_numpy()
    n = min(n_max, int(min(b.size / max(1 - pi_spoof, 1e-9), s.size / max(pi_spoof, 1e-9))))
    nb, ns = int(round(n * (1 - pi_spoof))), int(round(n * pi_spoof))
    nb, ns = max(nb, 1), max(ns, 1)
    return rng.choice(b, nb, replace=False), rng.choice(s, ns, replace=False)


# --------------------------------------------------------------------------- protocol
# v2: every cell (n_ref, prior, repetition) has its own seed derived from (seed, cell),
# so results do not depend on the number of worker processes or their scheduling.
_G = {}


def _init_worker(df, emb=None, source_tau=None):
    _G["df"], _G["emb"], _G["tau"] = df, emb, source_tau
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)                      # one BLAS thread per worker process
    except Exception:
        pass


def _cell_seed(seed, *key):
    return np.random.default_rng([int(seed)] + [int(round(k * 1000)) if isinstance(k, float) else int(k)
                                                for k in key])


def _cell_a(n_ref, pi, r, seed, ref_mode="disjoint"):
    df, source_tau = _G["df"], _G["tau"]
    if ref_mode == "disjoint":                      # v2.0.0 seeds kept for this mode
        rng = _cell_seed(seed, n_ref, pi, r)
        ref, pool = speaker_disjoint_split(df, n_ref, rng)
    else:
        rng = _cell_seed(seed, 11, n_ref, pi, r)
        ref, pool = speaker_overlap_split(df, n_ref, rng)
    sb, ss = resample_prior(pool, pi, rng)
    unl = np.concatenate([sb, ss])
    taus = {
        "SRC": source_tau,
        "GMM2": gmm2_threshold(unl, seed=r),
        "BONA": conformal_bona_threshold(ref, ALPHA),
        "ORACLE": eer_threshold(sb, ss),
    }
    tau_b, pi_hat, _ = pga_g_threshold(ref, unl)
    taus["PGA_G"] = tau_b
    taus["PGA-P"], _ = pga_g_pessimistic_threshold(ref, unl, rng)
    rows = []
    for m, tau in taus.items():
        op = op_point(tau, sb, ss)
        rows.append(dict(ref_mode=ref_mode, n_ref=n_ref, pi=pi, rep=r, method=m, tau=tau, frr=op.frr,
                         far=op.far, hter=op.hter, pi_hat=pi_hat if m == "PGA_G" else np.nan))
    return rows


def _parallel(fn, tasks, jobs, init_args, label):
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, out, n = time.time(), [], len(tasks)
    step = max(1, n // 20)
    if jobs <= 1:
        _init_worker(*init_args)
        for k, t in enumerate(tasks, 1):
            out.append(fn(*t))
            if k % step == 0 or k == n:
                print(f"  [{label}] {k}/{n} cells, {time.time() - t0:.0f}s", flush=True)
        return out
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=init_args) as ex:
        futs = [ex.submit(fn, *t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            if k % step == 0 or k == n:
                print(f"  [{label}] {k}/{n} cells, {time.time() - t0:.0f}s", flush=True)
        out = [f.result() for f in futs]          # submission order -> deterministic output
    return out


def run(df: pd.DataFrame, source_tau: float, reps: int, seed: int, jobs: int = 1,
        modes=("disjoint",)) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    tasks = [(n_ref, pi, r, seed, m) for m in modes for n_ref in REF_SIZES for pi in PRIORS
             for r in range(reps)]
    res = _parallel(_cell_a, tasks, jobs, (df, None, source_tau), "stage A")
    return pd.DataFrame([row for cell in res for row in cell])


def eer(s_bona: np.ndarray, s_spoof: np.ndarray) -> float:
    tau = eer_threshold(s_bona, s_spoof)
    return op_point(tau, s_bona, s_spoof).hter


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def pu_adapted_scores(emb_ref_tr, emb_ref_cal, emb_unl, rng, folds=5):
    """
    Cross-fitted non-traditional classifier (reference bona fide = 1, unlabelled = 0).
    Fold-matched: model k is trained on ref_tr + unlabelled points outside fold k, and it
    scores ONLY the unlabelled points of fold k and the calibration-reference points
    assigned to fold k. Within a fold, calibration and traffic points are both unseen by
    the model, which is what the conformal FRR guarantee needs.
    Returns (cal scores, cal fold ids, unlabelled scores, unlabelled fold ids).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n_u, n_c = emb_unl.shape[0], emb_ref_cal.shape[0]
    fold_u = rng.integers(0, folds, n_u)
    fold_c = np.arange(n_c) % folds
    rng.shuffle(fold_c)
    s_unl, s_cal = np.empty(n_u), np.empty(n_c)
    for k in range(folds):
        tr_u = emb_unl[fold_u != k]
        X = np.vstack([emb_ref_tr, tr_u])
        y = np.r_[np.ones(emb_ref_tr.shape[0]), np.zeros(tr_u.shape[0])]
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000))
        clf.fit(X, y)
        s_unl[fold_u == k] = clf.decision_function(emb_unl[fold_u == k])
        s_cal[fold_c == k] = clf.decision_function(emb_ref_cal[fold_c == k])
    return s_cal, fold_c, s_unl, fold_u


def foldwise_anchor_op(s_cal, fold_c, s_b, fold_b, s_s, fold_s, alpha=ALPHA):
    """Per-fold conformal anchors; FRR/FAR pooled over folds."""
    rej_b = np.zeros(s_b.size, bool)
    acc_s = np.zeros(s_s.size, bool)
    for k in np.unique(fold_c):
        tau = conformal_bona_threshold(s_cal[fold_c == k], alpha)
        rej_b[fold_b == k] = s_b[fold_b == k] < tau
        acc_s[fold_s == k] = s_s[fold_s == k] >= tau
    return OpPoint(float(rej_b.mean()), float(acc_s.mean()))


def _cell_b(pi, r, seed, ref_mode, n_ref):
    """
    ref_mode="overlap": reference = random genuine clips; the same speakers also appear in
        the traffic (enrolled-user deployment; the SCAR assumption of PU learning holds).
    ref_mode="disjoint": reference speakers never appear in the traffic (stress test; a
        classifier can then separate reference from traffic by speaker identity, which the
        shortcut_auc column is there to catch).
    """
    df, emb = _G["df"], _G["emb"]
    rng = _cell_seed(seed, 7, 0 if ref_mode == "overlap" else 1, pi, r)
    bona = df[df["label"] == 1]
    if ref_mode == "overlap":
        idx_ref = rng.choice(bona.index.to_numpy(), n_ref, replace=False)
        pool = df.drop(index=idx_ref)
    else:
        groups = df["group"].unique()
        rng.shuffle(groups)
        chosen, idx_ref = [], np.array([], int)
        for g in groups:
            chosen.append(g)
            idx_ref = bona.index[bona["group"].isin(chosen)].to_numpy()
            if idx_ref.size >= n_ref:
                break
        idx_ref = rng.choice(idx_ref, min(n_ref, idx_ref.size), replace=False)
        pool = df[~df["group"].isin(chosen)]
    b_idx = pool.index[pool["label"] == 1].to_numpy()
    s_idx = pool.index[pool["label"] == 0].to_numpy()
    n = min(12000, int(min(b_idx.size / (1 - pi), s_idx.size / pi)))
    b_idx = rng.choice(b_idx, int(round(n * (1 - pi))), replace=False)
    s_idx = rng.choice(s_idx, int(round(n * pi)), replace=False)
    unl_idx = np.r_[b_idx, s_idx]
    half = idx_ref.size // 2
    ref_tr, ref_cal = idx_ref[:half], idx_ref[half:]
    s_cal, fold_c, s_unl, fold_u = pu_adapted_scores(emb[ref_tr], emb[ref_cal], emb[unl_idx], rng)
    nb = b_idx.size
    frozen = df["score"].to_numpy()
    rows = []
    for name, sc_ref, sc_b, sc_s in [
        ("frozen", frozen[ref_cal], frozen[b_idx], frozen[s_idx]),
        ("pu_adapted", s_cal, s_unl[:nb], s_unl[nb:]),
    ]:
        pi_hat = 1 - estimate_bona_share(sc_ref, np.r_[sc_b, sc_s])
        if name == "pu_adapted":
            op = foldwise_anchor_op(sc_ref, fold_c, sc_b, fold_u[:nb], sc_s, fold_u[nb:])
        else:
            op = op_point(conformal_bona_threshold(sc_ref, ALPHA), sc_b, sc_s)
        rows.append(dict(ref_mode=ref_mode, pi=pi, rep=r, score=name, eer=eer(sc_b, sc_s),
                         pi_hat=pi_hat, frr_at_anchor=op.frr, far_at_anchor=op.far,
                         shortcut_auc=auc(sc_ref, sc_b), n_ref=int(idx_ref.size), n_traffic=int(n)))
    return rows


def run_stage_b(df: pd.DataFrame, emb: np.ndarray, reps: int, seed: int, modes=("overlap", "disjoint"),
                priors=(0.20, 0.50, 0.96), n_ref=1000, jobs: int = 1) -> pd.DataFrame:
    tasks = [(pi, r, seed, m, n_ref) for m in modes for pi in priors for r in range(reps)]
    res = _parallel(_cell_b, tasks, jobs, (df, emb, None), "stage B")
    return pd.DataFrame([row for cell in res for row in cell])


def synthetic_embeddings(df: pd.DataFrame, rng: np.random.Generator, dim=16):
    """Smoke-test embeddings: dim 0 carries the frozen score, dim 1 separates the hard
    attack family that the frozen score confuses with bona fide, the rest is noise."""
    e = rng.normal(0, 1, (len(df), dim))
    e[:, 0] = df["score"].to_numpy() / 2.0
    hard = (df["label"].to_numpy() == 0) & (df["score"].to_numpy() > -3.0)
    e[hard, 1] += 2.5
    return e


def summarise(res: pd.DataFrame) -> str:
    if "ref_mode" in res.columns and res["ref_mode"].nunique() > 1:
        return "\n".join(f"\n===== reference mode: {m} =====" + summarise(g.drop(columns="ref_mode"))
                         for m, g in res.groupby("ref_mode", sort=False))
    out = []
    piv = (res.groupby(["n_ref", "pi", "method"])[["frr", "far", "hter"]].mean() * 100).round(1)
    for n_ref in REF_SIZES:
        cols = ["SRC", "GMM2", "BONA", "PGA_G", "PGA-P", "ORACLE"]
        t = piv.loc[n_ref]["hter"].unstack("method")[cols]
        out.append(f"\nMean HTER (%) by target spoof prior, reference size n_ref={n_ref}\n{t}")
        f = piv.loc[n_ref]["frr"].unstack("method")[cols[:-1]]
        out.append(f"\nMean FRR (%) (bona fide rejected), n_ref={n_ref}\n{f}")
    est = res[res.method == "PGA_G"].groupby(["n_ref", "pi"])["pi_hat"].agg(["mean", "std"]).round(3)
    out.append(f"\nEstimated spoof prior (PGA_G) vs true prior\n{est}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--undetectable", type=float, default=0.0,
                    help="synthetic only: share of spoofs scoring like bona fide")
    ap.add_argument("--overlap", action="store_true",
                    help="synthetic only: harder attack family overlapping bona fide")
    ap.add_argument("--stage-b", action="store_true",
                    help="run the embedding-space PU adaptation kill test")
    ap.add_argument("--embeddings", type=str, help=".npy array aligned with --scores rows")
    ap.add_argument("--ref-mode", choices=["overlap", "disjoint", "both"], default="both",
                    help="reference speakers shared with traffic (primary), disjoint (stress test), or both")
    ap.add_argument("--scores", type=str)
    ap.add_argument("--source-threshold", type=float, default=4.97,
                    help="threshold fixed on the labelled source eval set (LA EER point)")
    ap.add_argument("--reps", type=int, default=50, help="Stage A repetitions per cell")
    ap.add_argument("--reps-b", type=int, default=10, help="Stage B repetitions per cell")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all CPU cores)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="pga_g_pilot_results.csv")
    a = ap.parse_args()
    import os
    jobs = a.jobs if a.jobs > 0 else (os.cpu_count() or 1)
    print(f"pga_g.pilot v{__version__} | {jobs} worker process(es)", flush=True)

    if a.synthetic:
        kw = dict(hard_mean=0.0, hard_frac=0.5) if a.overlap else {}
        df = synthetic_scores(np.random.default_rng(a.seed), undetectable=a.undetectable, **kw)
    elif a.scores:
        df = pd.read_csv(a.scores)
        need = {"score", "label", "group"}
        if not need.issubset(df.columns):
            raise SystemExit(f"CSV must contain columns {sorted(need)}")
    else:
        raise SystemExit("use --synthetic or --scores FILE")

    if a.stage_b:
        df = df.reset_index(drop=True)
        if a.synthetic:
            emb = synthetic_embeddings(df, np.random.default_rng(a.seed + 1))
        elif a.embeddings:
            emb = np.load(a.embeddings)
            if emb.shape[0] != len(df):
                raise SystemExit("embeddings must have one row per CSV row")
        else:
            raise SystemExit("--stage-b needs --embeddings (or --synthetic)")
        modes = ["overlap", "disjoint"] if a.ref_mode == "both" else [a.ref_mode]
        res_b = run_stage_b(df, np.asarray(emb, dtype=np.float32), reps=a.reps_b, seed=a.seed,
                            modes=modes, jobs=jobs)
        res_b.to_csv(a.out, index=False)
        summ = res_b.groupby(["ref_mode", "pi", "score"])[
            ["eer", "pi_hat", "frr_at_anchor", "far_at_anchor", "shortcut_auc"]].mean()
        print((summ * 100).round(1).to_string())
        print("\n(shortcut_auc is x100; ~50 means no reference-specific shortcut)")
        print(f"Per-repetition results written to {a.out}")
        return

    modes = ["overlap", "disjoint"] if a.ref_mode == "both" else [a.ref_mode]
    res = run(df, a.source_threshold, a.reps, a.seed, jobs=jobs, modes=modes)
    res.to_csv(a.out, index=False)
    print(summarise(res))
    print(f"\nPer-repetition results written to {a.out}")


if __name__ == "__main__":
    main()
