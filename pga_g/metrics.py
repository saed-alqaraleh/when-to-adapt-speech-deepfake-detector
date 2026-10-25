"""Threshold metrics. Convention: higher score = more bona fide; reject when score < tau."""
from __future__ import annotations

import numpy as np


def rates(scores_bona: np.ndarray, scores_spoof: np.ndarray, tau: float) -> tuple[float, float]:
    frr = float(np.mean(scores_bona < tau))
    far = float(np.mean(scores_spoof >= tau))
    return frr, far


def eer_and_threshold(scores_bona: np.ndarray, scores_spoof: np.ndarray) -> tuple[float, float]:
    """EER on the observed-score grid; tau = grid point minimising |FRR - FAR|
    (the convention of Zhou & Wang 2026). EER is reported as (FRR + FAR) / 2 there."""
    b = np.sort(np.asarray(scores_bona, dtype=np.float64))
    s = np.sort(np.asarray(scores_spoof, dtype=np.float64))
    grid = np.unique(np.concatenate([b, s]))
    frr = np.searchsorted(b, grid, side="left") / b.size
    far = 1.0 - np.searchsorted(s, grid, side="left") / s.size
    j = int(np.argmin(np.abs(frr - far)))
    return float(0.5 * (frr[j] + far[j])), float(grid[j])


def hter(scores_bona, scores_spoof, tau) -> dict:
    frr, far = rates(np.asarray(scores_bona), np.asarray(scores_spoof), tau)
    return {"tau": float(tau), "frr": frr, "far": far, "hter": 0.5 * (frr + far)}
