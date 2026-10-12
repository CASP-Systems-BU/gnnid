"""Shared artifact helpers: run manifest + benign-val quantile normalization.

Each detector owns its bundle layout (see detectors/flash.py, detectors/ppt/);
what they share is the manifest format and the score-normalization scheme —
per node type, store the sorted benign-val raw scores and normalize at
inference by empirical quantile rank against them.
"""
from __future__ import annotations

import subprocess

import numpy as np

from .config import Config


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                              capture_output=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_manifest(cfg: Config, splits: dict) -> dict:
    return {"git_sha": _git_sha(), "splits": splits, "config": dict(cfg)}


def fit_norm_thresholds(scores_by_type: dict[str, list[float]],
                        threshold_q: float) -> tuple[dict, dict]:
    """Benign-val raw scores per node type -> (quantile-norm tables,
    per-type alert thresholds).

    Thresholds live in NORMALIZED (quantile-rank) space — the same space as
    score_norm (see `normalize`) — because scoring alerts via
    `score_norm >= threshold` (score.apply_thresholds). The alert cut is the
    quantile level itself: a node fires when its benign-val rank exceeds
    threshold_q (e.g. the top 0.5% for its type at q=0.995). Storing the raw
    score quantile value here instead mixes units (rank vs raw) and destroys
    calibration — it drove ~100% benign alert rates on narrow-support types and
    0% on pods.
    """
    norm = quantile_norm_tables(scores_by_type)
    thresholds = {t: float(threshold_q) for t in norm}
    return norm, thresholds


def quantile_norm_tables(scores_by_type: dict[str, list[float]]) -> dict:
    """Per node-type: store the sorted benign-val score array; normalization at
    inference is the empirical quantile rank against it."""
    return {t: sorted(float(s) for s in arr)
            for t, arr in scores_by_type.items() if arr}


def normalize(score: float, sorted_arr: list[float]) -> float:
    if not sorted_arr:
        return score
    lo = np.searchsorted(sorted_arr, score, side="left")
    hi = np.searchsorted(sorted_arr, score, side="right")
    return (lo + hi) / (2.0 * len(sorted_arr))
