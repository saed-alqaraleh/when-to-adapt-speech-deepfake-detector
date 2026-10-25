"""
v5: split audit and group-aware re-analysis (CPU only; existing scores and features). Pre-specified
before any v5 result exists; the original analyses and their verdicts are preserved unchanged.

  siblings  ASVspoof 2021 DF ships no source-recording link, so codec versions of one recording are
            identified from the speech duration of the 40 dB trimmed copy and the frozen XLS-R features
            (results/v5/groups_df21.csv, results/v5/SIBLINGS_df21.json). ASVspoof 5 uses its official
            source groups (codec_seed).
  main      the main design of the third test (both corpora, three detectors, both reference modes,
            every share, 20 repetitions) on the same references, under three split designs:
              A original   exactly the third test (reproduces its rows)
              B grouped    reference halves keep every source on one side; the traffic loses the
                           sources of the reference (spoofs thinned to keep the spoof share); held-out
                           trials exclude the sources of the reference and the traffic
              C grouped +  the calibration half is split by source into a selection half (gate and
                independent guard) and a final-calibration half (anchors only), so the complete
                calibration pipeline carries the conformal bound on held-out genuine trials
            Every design is evaluated on its traffic and on a held-out sample.

    python -m pga_g.v5 prereg   --workspace W
    python -m pga_g.v5 siblings --workspace W --local LOCAL_CACHE
    python -m pga_g.v5 all      --workspace W --part 0 --n-parts 2 --local LOCAL_CACHE
    python -m pga_g.v5 summary  --workspace W
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

from . import general as G
from . import robust as R
from .chunked import log, run_chunked
from .detectors import NAMES, PRIMARY

DETECTORS, CORPORA = ("sslaasist", "aasist", "mamba"), ("df21", "asv5")
KEPT_TOL, NULL_GAP, NULL_Q = 0.10, 0.50, 0.999
KEEP_MIN = 0.5
NULL_PER_ITEM, NULL_MAX = 10, 2_000_000
HELD_BONA, HELD_SPOOF = 5000, 20000
MATERIAL_EER, MATERIAL_FRR = 0.005, 0.01
LA19_EVAL_BONA = 7355                      # bona fide utterances of the ASVspoof 2019 LA evaluation set
DESIGNS = ("A", "B", "C")
WAIT_S, WAIT_MAX_S = 30, 4 * 3600

PREREG = {
    "title": "Split audit and group-aware re-analysis (v5)",
    "fixed_on": "2026-10-24, before any v5 result (results/v5) existed; inputs are the existing scores and features",
    "status": ("pre-specified validation of the split and calibration design; the original analyses and verdicts "
               "are preserved and reported unchanged; v5 adds evidence and changes no earlier verdict"),
    "motivation": ("ASVspoof 2021 DF provides no source-recording link and the published counts imply several codec "
                   "versions per bona fide source; earlier DF analyses treated each file as its own source; in both "
                   "corpora the reference halves were drawn without source grouping; the gate and the guard reuse the "
                   "calibration half that sets the anchor"),
    "siblings_df21": ("blocks of equal label, speaker, attack and source corpus; candidate pairs whose speech duration "
                      "after 40 dB trimming (runs/df21trim40_v1) differs by at most 0.10 s; linked if the cosine "
                      "similarity of SSL-AASIST time-mean XLS-R features is at least theta, theta = 99.9th percentile "
                      "of the cosine over same-block pairs whose trimmed durations differ by at least 0.50 s (up to 10 "
                      "random pairs per trial, at most 2,000,000, fixed seed); sources = connected components; "
                      "reported: theta, cosine distributions, component sizes, linked shares by label, codec and "
                      "source corpus, and versions per source for ASVspoof-2019-derived bona fide trials"),
    "groups_asv5": "official source groups (protocol column codec_seed), as in every earlier ASVspoof 5 analysis",
    "designs": {
        "A_original": "the third test's main cell, primary features (identical references, traffic, halves and folds)",
        "B_grouped": ("same reference; halves assigned by source (random source order, filled to 500); traffic = "
                      "original traffic without trials whose source is in the reference, spoofs removed at random to "
                      "keep the spoof share; crossfit, gate, guard and anchors as specified"),
        "C_grouped_independent_calibration": ("as B; calibration half split by source into selection (250, gate and "
                                              "guard) and final calibration (250, anchors only)"),
    },
    "held_out": ("per repetition a random sample of up to 5,000 genuine and 20,000 spoofed trials outside the "
                 "reference and the traffic (A: by the groups of the original analysis; B, C: by source); "
                 "scored with the fold-averaged adapted model or the frozen score; anchor from the calibration "
                 "(A, B) or final-calibration (C) recordings scored the same way"),
    "metrics": "EER, anchor FRR (genuine rejection), FAR (spoof acceptance), HTER, shortcut AUC; traffic and held-out",
    "design_conditions": "main conditions of the third test: both modes, all shares, 20 repetitions, three detectors",
    "decision": {
        "robust_to_source_leakage": ("per detector and corpus: the PGA-G outcome of each of H1-H5 under B equals the "
                                     "outcome under A, and in every main condition |mean EER(B) - mean EER(A)| <= 0.5 "
                                     "point and |mean FRR(B) - mean FRR(A)| <= 1 point for PGA-G (traffic)"),
        "independent_calibration": ("reported without decision: H1-H5 under C, held-out FRR of PGA-G under C (formal "
                                    "bound alpha = 0.05 under exchangeability), paired differences C - B"),
        "reporting": "every result is reported, whatever the outcome",
    },
    "statistics": "means over repetitions; paired differences with 95% percentile bootstrap intervals (4,000 resamples)",
}

_V: dict = {}


# ---------------------------------------------------------------- pre-registration (write-once)
def prereg(ws: Path) -> str:
    ws = Path(ws)
    path = ws / "results" / "PREREGISTRATION_V5.json"
    body = json.dumps(PREREG, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        if json.loads(path.read_text()).get("sha256") != digest:
            raise SystemExit(f"{path} exists with different content; it is never overwritten")
        return digest
    res = ws / "results" / "v5"
    if res.exists() and (any(res.rglob("chunk_*.csv")) or (res / "groups_df21.csv").exists()):
        raise SystemExit("v5 analyses have already started: a pre-registration written now would not be one")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"preregistration": PREREG, "sha256": digest,
                               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    tmp.replace(path)
    log(f"pre-registration written: {path} (sha256 {digest[:16]})")
    return digest


# ---------------------------------------------------------------- source groups
def _uf_components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    for i, j in zip(a.tolist(), b.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def groups_path(ws: Path) -> Path:
    return Path(ws) / "results" / "v5" / "groups_df21.csv"


def siblings(ws: Path, local=None) -> int:
    ws = Path(ws)
    prereg(ws)
    out, rep = groups_path(ws), ws / "results" / "v5" / "SIBLINGS_df21.json"
    if out.exists() and rep.exists():
        log(f"source groups exist: {out}")
        return 0
    base = ws / "runs" / "df21_v1"
    head = pd.read_csv(base / "scores.csv", nrows=0).columns
    cols = [c for c in ("utt_id", "speaker", "attack", "label", "codec", "source") if c in head]
    meta = pd.read_csv(base / "scores.csv", usecols=cols, dtype={c: str for c in cols if c != "label"},
                       keep_default_na=False)
    trim = pd.read_csv(ws / "runs" / "df21trim40_v1" / "scores.csv", usecols=["utt_id", "kept_s"],
                       dtype={"utt_id": str})
    m = meta[["utt_id"]].merge(trim, on="utt_id", how="left", validate="one_to_one")
    if m.kept_s.isna().any() or (m.kept_s < 0).any():
        raise SystemExit(f"trimmed speech duration missing for {int(m.kept_s.isna().sum() + (m.kept_s < 0).sum())} trials")
    kept = m.kept_s.to_numpy(np.float64)
    X = np.load(R._cache(base / "ssl_mean1024.npy", local), mmap_mode="r")
    if X.shape[0] != len(meta):
        raise SystemExit("features are not row-aligned with the scores")
    if "source" not in meta:
        meta["source"] = "-"
    key = meta.label.astype(str) + "|" + meta.speaker + "|" + meta.attack + "|" + meta.source
    blocks = pd.factorize(key)[0]
    order = np.lexsort((kept, blocks))
    bounds = np.flatnonzero(np.diff(blocks[order])) + 1
    rng = np.random.default_rng([2026, 95])
    cand_i, cand_j, cand_c, null_c, sample_c, n_cand = [], [], [], [], [], 0
    t0 = time.time()
    for bi, idx in enumerate(np.split(order, bounds)):
        n = idx.size
        if n < 2:
            continue
        rows = np.sort(idx)
        F = _unit(X[rows])
        loc = np.searchsorted(rows, idx)                # position of each kept-sorted trial in F
        k_s = kept[idx]
        ends = np.searchsorted(k_s, k_s + KEPT_TOL, side="right")
        ii = np.repeat(np.arange(n), ends - np.arange(n) - 1)
        if ii.size:
            jj = np.concatenate([np.arange(a + 1, e) for a, e in enumerate(ends) if e > a + 1])
            for s in range(0, ii.size, 2_000_000):
                a, b = ii[s:s + 2_000_000], jj[s:s + 2_000_000]
                cs = np.einsum("ij,ij->i", F[loc[a]], F[loc[b]])
                if cs.size > 2000:
                    sample_c.append(rng.choice(cs, 2000, replace=False))
                else:
                    sample_c.append(cs)
                keep = cs >= KEEP_MIN                   # pairs far below any plausible theta are not stored
                cand_c.append(cs[keep]); cand_i.append(idx[a][keep]); cand_j.append(idx[b][keep])
                n_cand += int(cs.size)
        m_null = NULL_PER_ITEM * n
        a, b = rng.integers(0, n, m_null), rng.integers(0, n, m_null)
        ok = np.abs(k_s[a] - k_s[b]) >= NULL_GAP
        if ok.any():
            null_c.append(np.einsum("ij,ij->i", F[loc[a[ok]]], F[loc[b[ok]]]))
        if bi % 500 == 0:
            log(f"siblings: block {bi} of {len(bounds) + 1} ({time.time() - t0:.0f}s)")
    cand_i = np.concatenate(cand_i) if cand_i else np.array([], int)
    cand_j = np.concatenate(cand_j) if cand_j else np.array([], int)
    cand_c = np.concatenate(cand_c) if cand_c else np.array([])
    null_c = np.concatenate(null_c) if null_c else np.array([])
    if null_c.size > NULL_MAX:
        null_c = rng.choice(null_c, NULL_MAX, replace=False)
    theta = float(np.quantile(null_c, NULL_Q)) if null_c.size else 1.0
    if theta < KEEP_MIN:
        raise SystemExit(f"theta {theta:.3f} below the storage bound {KEEP_MIN}; lower KEEP_MIN")
    sample_c = np.concatenate(sample_c) if sample_c else np.array([])
    link = cand_c >= theta
    comp = _uf_components(len(meta), cand_i[link], cand_j[link])
    gid = pd.factorize(comp)[0]
    size = np.bincount(gid)[gid]
    grp = pd.DataFrame({"utt_id": meta.utt_id, "src_group_v5": [f"G{g:07d}" for g in gid]})
    lab = meta.label.astype(int).to_numpy()
    q = lambda x, p: float(np.quantile(x, p)) if x.size else float("nan")
    la_bona = (lab == 1) & meta.speaker.str.startswith("LA_").to_numpy() & (meta.source.isin(["asvspoof", "-"]).to_numpy())
    rep_d = {
        "theta": theta, "kept_tol_s": KEPT_TOL, "null_gap_s": NULL_GAP, "null_quantile": NULL_Q,
        "n_null_pairs": int(null_c.size), "n_candidate_pairs": int(n_cand), "n_links": int(link.sum()),
        "null_cosine_quantiles": {p: q(null_c, float(p)) for p in ("0.5", "0.9", "0.99", "0.999")},
        "candidate_cosine_quantiles_sample": {p: q(sample_c, float(p)) for p in ("0.01", "0.1", "0.5", "0.9")},
        "linked_candidate_cosine_quantiles": {p: q(cand_c[link], float(p)) for p in ("0.01", "0.5")},
        "n_trials": int(len(meta)), "n_sources": int(gid.max() + 1),
        "component_size_counts": {int(k): int(v) for k, v in zip(*np.unique(np.bincount(gid), return_counts=True))},
        "share_with_version_by_label": {("genuine" if l_ == 1 else "spoof"): float((size[lab == l_] > 1).mean())
                                        for l_ in (1, 0)},
        "share_with_version_by_codec": {str(c): float((size[(meta.codec == c).to_numpy()] > 1).mean())
                                        for c in sorted(meta.codec.unique())},
        "share_with_version_by_source": {str(c): float((size[(meta.source == c).to_numpy()] > 1).mean())
                                         for c in sorted(meta.source.unique())},
        "la19_bona": {"trials": int(la_bona.sum()), "sources": int(np.unique(gid[la_bona]).size),
                      "versions_per_source": float(la_bona.sum() / max(np.unique(gid[la_bona]).size, 1)),
                      "la19_eval_bona_utterances": LA19_EVAL_BONA},
        "largest_component": int(np.bincount(gid).max()), "seconds": round(time.time() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(out.with_name(out.name + ".tmp"), index=False)
    out.with_name(out.name + ".tmp").replace(out)
    rep.with_name(rep.name + ".tmp").write_text(json.dumps(rep_d, indent=1))
    rep.with_name(rep.name + ".tmp").replace(rep)
    log(f"siblings: theta {theta:.4f}, {int(link.sum())} links, {rep_d['n_sources']} sources for {len(meta)} trials; "
        f"genuine with another version {100 * rep_d['share_with_version_by_label']['genuine']:.1f}%; "
        f"ASVspoof 2019 bona fide versions per source {rep_d['la19_bona']['versions_per_source']:.2f}")
    return 0


def _wait_groups(ws: Path):
    t = 0
    while not groups_path(ws).exists():
        if t >= WAIT_MAX_S:
            raise SystemExit("ASVspoof 2021 DF source groups were not written; run 'siblings' first")
        log("waiting for the ASVspoof 2021 DF source groups (computed by part 0)")
        time.sleep(WAIT_S)
        t += WAIT_S


def _grp5(ws: Path, meta: pd.DataFrame, corpus: str) -> np.ndarray:
    if corpus == "asv5":
        return pd.factorize(meta.src_group)[0]
    g = pd.read_csv(groups_path(ws), dtype=str)
    m = meta[["utt_id"]].merge(g, on="utt_id", how="left", validate="one_to_one")
    if m.src_group_v5.isna().any():
        raise SystemExit("source groups do not cover every ASVspoof 2021 DF trial")
    return pd.factorize(m.src_group_v5)[0]


# ---------------------------------------------------------------- inputs
def _load(ws, corpus, det, local):
    meta, score, feats = G.load(ws, corpus, det, local)
    return meta, score, feats, _grp5(ws, meta, corpus)


def _init(meta, score, feats, corpus, det, grp5):
    G._init(meta, score, feats, corpus, det)
    _V.clear()
    _V["grp5"] = np.asarray(grp5)


# ---------------------------------------------------------------- split designs
def grouped_halves(idx: np.ndarray, g: np.ndarray, rng, n_first: int):
    """Assign whole sources to the first part (random source order) until it holds n_first trials."""
    idx = np.asarray(idx)
    codes, inv = np.unique(g[idx], return_inverse=True)
    first = np.zeros(idx.size, bool)
    filled = 0
    for c in rng.permutation(codes.size):
        mem = inv == c
        k = int(mem.sum())
        if filled + k <= n_first:
            first[mem] = True
            filled += k
        if filled >= n_first:
            break
    return idx[first], idx[~first]


def _without_sources(items: np.ndarray, of: np.ndarray, g: np.ndarray) -> np.ndarray:
    bad = np.zeros(g.max() + 1, bool)
    bad[g[of]] = True
    return items[~bad[g[items]]]


def _held_out(order: np.ndarray, excl: np.ndarray, g: np.ndarray, lab: np.ndarray):
    bad = np.zeros(g.max() + 1, bool)
    bad[g[excl]] = True
    ok = order[~bad[g[order]]]
    b, s = ok[lab[ok] == 1][:HELD_BONA], ok[lab[ok] == 0][:HELD_SPOOF]
    return b, s


def _avg_scores(models, X):
    return np.mean([m.decision_function(X) for m in models], axis=0)


def _held_metrics(s_b, s_s, tau):
    op = P.op_point(tau, s_b, s_s)
    return {"h_eer": P.eer(s_b, s_s), "h_frr": op.frr, "h_far": op.far, "h_hter": op.hter,
            "h_n_bona": int(s_b.size), "h_n_spoof": int(s_s.size)}


def _design_rows(base, design, fz, X, cal_gate, cal_anchor, traffic, nb, a, hb, hs, fold_anchor_pos=None,
                 gate_pos=None):
    """Frozen and PGA-G rows of one design. a = crossfit output on (training half, calibration half, traffic) or None."""
    b_fr = P.estimate_bona_share(fz[cal_gate], fz[traffic])
    gate_open = bool(1 - b_fr >= G.GATE)
    frozen = G._evaluate(fz[cal_anchor], fz[traffic], nb)
    tau_fz = P.conformal_bona_threshold(fz[cal_anchor], P.ALPHA)
    h_fz = _held_metrics(fz[hb], fz[hs], tau_fz)
    row_f = {**base, "design": design, "method": "frozen", "adapted": False, "gate_open": gate_open,
             "pi_hat_frozen": 1 - b_fr, "n_traffic": int(traffic.size), "n_traffic_bona": int(nb), **frozen, **h_fz}
    adapted, ratio, ev, h_ad = False, float("nan"), None, None
    if gate_open and a is not None:
        s_cal, fold_c, s_u, fold_u, models = a
        g_pos = slice(None) if gate_pos is None else gate_pos
        f_pos = slice(None) if fold_anchor_pos is None else fold_anchor_pos
        b_ad = R.bona_share(s_cal[g_pos], s_u)
        ratio = b_ad / max(b_fr, 1e-9) if np.isfinite(b_ad) else float("nan")
        adapted = bool(np.isfinite(ratio) and ratio >= G.RHO)
        if adapted:
            ev = G._evaluate(s_cal[f_pos], s_u, nb, (fold_c[f_pos], fold_u))
            Xc = G._feat(X, cal_anchor)
            tau_ad = P.conformal_bona_threshold(_avg_scores(models, Xc), P.ALPHA)
            h_ad = _held_metrics(_avg_scores(models, G._feat(X, hb)), _avg_scores(models, G._feat(X, hs)), tau_ad)
    row_g = {**row_f, "method": "pgag", "adapted": adapted, "guard_ratio": ratio,
             **(ev if adapted else {}), **(h_ad if adapted else {})}
    return [row_f, row_g]


def main_cell(mode, pi, rep, seed):
    rng, tr, cal, traffic, nb, n_spk = G._draw_main(mode, pi, rep, seed)
    corpus, det = G._G["corpus"], G._G["det"]
    fz, lab, g0, g5 = G._G["score"], G._G["lab"], G._G["grp"], _V["grp5"]
    X = G._G["feats"][PRIMARY[det]]
    ref = np.r_[tr, cal]
    key = [seed, CID_V5, G.CID[corpus], 0 if mode == "overlap" else 1, int(round(pi * 1000)), rep]
    rng_b, rng_c, rng_h = (np.random.default_rng(key + [k]) for k in (1, 2, 3))
    order = rng_h.permutation(lab.size)
    base = {"mode": mode, "pi": pi, "rep": rep, "n_ref_speakers": n_spk}

    # exposure of the original design to source versions
    in_tr = np.zeros(g5.max() + 1, bool)
    in_tr[g5[tr]] = True
    in_ref = np.zeros(g5.max() + 1, bool)
    in_ref[g5[ref]] = True
    hbA, hsA = _held_out(order, np.r_[ref, traffic], g0, lab)
    in_rt = in_ref.copy()
    in_rt[g5[traffic]] = True
    base.update(x_cal_with_train_version=int(in_tr[g5[cal]].sum()),
                x_traffic_bona_with_ref_version=int(in_ref[g5[traffic[:nb]]].sum()),
                x_heldout_bona_with_version=int(in_rt[g5[hbA]].sum()))

    # A: original design (the third test's draws, halves, folds and random-number use)
    b_frA = P.estimate_bona_share(fz[cal], fz[traffic])
    aA = R.crossfit(G._feat(X, tr), G._feat(X, cal), G._feat(X, traffic), rng) if 1 - b_frA >= G.GATE else None
    rows = _design_rows(base, "A", fz, X, cal, cal, traffic, nb, aA, hbA, hsA)

    # B: source-grouped halves and traffic
    trB, calB = grouped_halves(ref, g5, rng_b, ref.size // 2)
    keepB = _without_sources(traffic[:nb], ref, g5)
    spB = _without_sources(traffic[nb:], ref, g5)
    n_s = int(round(keepB.size * (traffic.size - nb) / max(nb, 1)))
    if spB.size > n_s:
        spB = np.sort(rng_b.choice(spB, n_s, replace=False))
    trafB, nbB = np.r_[keepB, spB], int(keepB.size)
    hbB, hsB = _held_out(order, np.r_[ref, trafB], g5, lab)
    # C: selection and final-calibration halves of the grouped calibration half
    sel, fin = grouped_halves(calB, g5, rng_c, calB.size // 2)
    calBC = np.r_[sel, fin]                           # calibration order: selection first, then final
    pos_sel, pos_fin = np.arange(sel.size), np.arange(sel.size, calBC.size)
    gate_B = 1 - P.estimate_bona_share(fz[calBC], fz[trafB]) >= G.GATE
    gate_C = 1 - P.estimate_bona_share(fz[sel], fz[trafB]) >= G.GATE
    aB = R.crossfit(G._feat(X, trB), G._feat(X, calBC), G._feat(X, trafB), rng_b) if (gate_B or gate_C) else None
    rows += _design_rows(base, "B", fz, X, calBC, calBC, trafB, nbB, aB, hbB, hsB)
    rows += _design_rows(base, "C", fz, X, sel, fin, trafB, nbB, aB, hbB, hsB, fold_anchor_pos=pos_fin,
                         gate_pos=pos_sel)
    for r in rows:
        r["n_ref_train"], r["n_cal"] = (int(tr.size), int(cal.size)) if r["design"] == "A" else (int(trB.size), int(calB.size))
    return rows


CID_V5 = 97


def tasks_for(corpus, seed):
    shares = sorted(set(G.SHARES[corpus]) | {G.NATURAL[corpus]})
    return [(m, p, r, seed) for m in ("overlap", "disjoint") for p in shares for r in range(G.REPS)]


def out_dir(ws, corpus, det) -> Path:
    return Path(ws) / "results" / "v5" / f"main_{corpus}_{det}"


def _run(ws, corpus, det, part, n_parts, jobs, seed, local, max_tasks=None):
    digest = prereg(ws)
    if corpus == "df21":
        if part == 0 and not groups_path(ws).exists():
            siblings(ws, local)
        _wait_groups(ws)
    meta, score, feats, grp5 = _load(ws, corpus, det, local)
    return run_chunked(main_cell, tasks_for(corpus, seed), out_dir(ws, corpus, det), _init,
                       (meta, score, feats, corpus, det, grp5), jobs=jobs, part=part, n_parts=n_parts, flush_every=10,
                       label=f"v5 main {corpus} {NAMES[det]}", max_tasks=max_tasks,
                       config={"prereg_sha256": digest, "corpus": corpus, "detector": det, "seed": seed,
                               "groups_sha256": hashlib.sha256(groups_path(ws).read_bytes()).hexdigest()[:16]
                               if corpus == "df21" else "codec_seed"})


# ---------------------------------------------------------------- summary
def _boot_diff(a, b, rng, n=4000):
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, d.size, (n, d.size))
    m = d[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def _as_v3(res, design):
    x = res[res.design == design].copy()
    x["variant"] = np.where(x.method == "frozen", "frozen", "pgag")
    return x


def summary(ws: Path) -> int:
    ws = Path(ws)
    rng = np.random.default_rng(0)
    S = {"prereg_sha256": prereg(ws), "siblings_df21": None, "main": {}}
    sp = ws / "results" / "v5" / "SIBLINGS_df21.json"
    lines = ["v5 summary"]
    if sp.exists():
        S["siblings_df21"] = json.loads(sp.read_text())
        s = S["siblings_df21"]
        lines.append(f"ASVspoof 2021 DF sources: theta {s['theta']:.4f}; {s['n_sources']} sources for {s['n_trials']} trials; "
                     f"genuine trials with another version {100 * s['share_with_version_by_label']['genuine']:.1f}%, "
                     f"spoofed {100 * s['share_with_version_by_label']['spoof']:.1f}%; ASVspoof-2019 bona fide: "
                     f"{s['la19_bona']['versions_per_source']:.2f} versions per source "
                     f"({s['la19_bona']['sources']} sources; {LA19_EVAL_BONA} LA 2019 eval utterances); "
                     f"largest source {s['largest_component']}")
    for det in DETECTORS:
        for c in CORPORA:
            p = out_dir(ws, c, det) / "results.csv"
            if not p.exists():
                continue
            res = pd.read_csv(p)
            hyp = {d: G.hypotheses(_as_v3(res, d), c, "pgag", rng) for d in DESIGNS}
            outc = {d: {h: bool(hyp[d][h]["pass"]) for h in ("H1_gain", "H2_no_harm_low_share", "H3_no_harm_disjoint",
                                                             "H4_genuine_rejection", "H5_shortcut")} for d in DESIGNS}
            diffs, worst = {}, {"eer": 0.0, "frr": 0.0}
            for (mode, pi), g in res.groupby(["mode", "pi"]):
                cellk = f"{mode}|{pi:g}"
                diffs[cellk] = {}
                piv = {d: g[(g.design == d) & (g.method == "pgag")].sort_values("rep").reset_index(drop=True)
                       for d in DESIGNS}
                fzv = {d: g[(g.design == d) & (g.method == "frozen")].sort_values("rep").reset_index(drop=True)
                       for d in DESIGNS}
                for d in ("B", "C"):
                    e = {}
                    for mtr in ("eer", "frr", "far", "hter", "h_eer", "h_frr", "h_far", "h_hter"):
                        a_, b_ = piv[d][mtr].to_numpy(float), piv["A"][mtr].to_numpy(float)
                        e[mtr] = {"A": float(np.nanmean(b_)), d: float(np.nanmean(a_)),
                                  "diff": float(np.nanmean(a_ - b_)), "ci95": _boot_diff(a_, b_, rng)}
                    e["adapted"] = {"A": float(piv["A"].adapted.mean()), d: float(piv[d].adapted.mean())}
                    e["frozen_eer"] = {"A": float(fzv["A"].eer.mean()), d: float(fzv[d].eer.mean())}
                    diffs[cellk][f"{d}-A"] = e
                    if d == "B":
                        worst["eer"] = max(worst["eer"], abs(e["eer"]["diff"]))
                        worst["frr"] = max(worst["frr"], abs(e["frr"]["diff"]))
            exposure = {k: float(res[res.design == "A"].drop_duplicates(["mode", "pi", "rep"])[k].mean())
                        for k in ("x_cal_with_train_version", "x_traffic_bona_with_ref_version",
                                  "x_heldout_bona_with_version")}
            heldC = res[(res.design == "C") & (res.method == "pgag") & (res["mode"] == "overlap")].groupby("pi").h_frr.mean()
            robust = bool(outc["B"] == outc["A"] and worst["eer"] <= MATERIAL_EER and worst["frr"] <= MATERIAL_FRR)
            reprod = None
            v3p = G.out_dir(ws, "main", c, det) / "results.csv"
            if v3p.exists():
                v3 = pd.read_csv(v3p)
                v3 = v3[v3.variant == "pgag_primary"][["mode", "pi", "rep", "eer", "adapted"]]
                j = res[(res.design == "A") & (res.method == "pgag")].merge(v3, on=["mode", "pi", "rep"], suffixes=("", "_v3"))
                reprod = {"n": int(len(j)), "decision_mismatch": int((j.adapted != j.adapted_v3).sum()),
                          "max_abs_eer_diff": float(np.abs(j.eer - j.eer_v3).max())}
            S["main"][f"{det}|{c}"] = {"outcomes": outc, "hypotheses": hyp, "paired": diffs, "worst_B_minus_A": worst,
                                       "robust_to_source_leakage": robust, "exposure_A": exposure,
                                       "heldout_frr_C_matched": {f"{k:g}": float(v) for k, v in heldC.items()},
                                       "reproduces_v3": reprod}
            lines.append(f"{NAMES[det]:<11} {c}: robust to source leakage: {'yes' if robust else 'NO'} "
                         f"(max |dEER| {100 * worst['eer']:.2f} pt, max |dFRR| {100 * worst['frr']:.2f} pt; outcomes "
                         f"A {sum(outc['A'].values())}/5, B {sum(outc['B'].values())}/5, C {sum(outc['C'].values())}/5); "
                         f"exposure A: calibration with training version {exposure['x_cal_with_train_version']:.1f}, "
                         f"genuine traffic with reference version {exposure['x_traffic_bona_with_ref_version']:.1f}; "
                         f"held-out FRR C matched max {100 * float(heldC.max()) if len(heldC) else float('nan'):.2f}%"
                         + (f"; A reproduces v3: {reprod['decision_mismatch']} decision mismatches, max |dEER| "
                            f"{100 * reprod['max_abs_eer_diff']:.3f} pt" if reprod else ""))
    out = ws / "results" / "V5_RESULTS.json"
    out.with_name(out.name + ".tmp").write_text(json.dumps(S, indent=1, default=float))
    out.with_name(out.name + ".tmp").replace(out)
    (ws / "results" / "V5_SUMMARY.txt").write_text("\n".join(lines) + "\n")
    for ln in lines:
        log(ln)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prereg", "siblings", "all", "summary"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--corpora", default=",".join(CORPORA))
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
    if a.cmd == "siblings":
        return siblings(ws, a.local)
    if a.cmd == "summary":
        return summary(ws)
    complete = True
    for corpus in a.corpora.split(","):
        for det in a.detectors.split(","):
            if not G.available(ws, corpus, det):
                log(f"{NAMES[det]} on {corpus}: inputs missing; skipped")
                complete = False
                continue
            complete &= _run(ws, corpus, det, a.part, a.n_parts, a.jobs, a.seed, a.local, a.max_tasks) is not None
    if complete:
        return summary(ws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
