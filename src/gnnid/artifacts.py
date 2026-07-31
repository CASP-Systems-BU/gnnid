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
    per-type alert thresholds at the given quantile)."""
    norm = quantile_norm_tables(scores_by_type)
    thresholds = {t: float(np.quantile(arr, threshold_q))
                  for t, arr in scores_by_type.items() if arr}
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
