"""Detector seam: one class per GNN-IDS method = strategy + trained bundle.

A Detector owns its graph construction, training procedure (any number of
stages), artifact bundle, and scoring function. Everything else is shared —
ingest, windowing, temporal splits, label vocab, quantile normalization,
per-pod aggregation, and the eval harness — and speaks the SCORE_COLUMNS
record schema below.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from ..config import Config

# The shared per-node-window score record. score_raw and score_norm are in
# [0, 1]; score_norm is the empirical quantile of score_raw against the
# detector's benign-val tables (artifacts.quantile_norm_tables / normalize).
SCORE_COLUMNS = ("run_id", "w_idx", "entity_id", "node_type",
                 "true_label", "pred_label", "p_true", "score_raw",
                 "score_norm")


class Detector(abc.ABC):
    name: ClassVar[str]     # registry key == the config `detector` value
    manifest: dict          # must contain "splits": {train, val, test}
    thresholds: dict        # node_type -> normalized-score alert threshold

    @classmethod
    @abc.abstractmethod
    def train(cls, cfg: Config, parquet_dir: Path,
              splits: dict[str, list[str]]) -> "Detector":
        """Full training procedure — all stages happen inside. cfg is the
        merged detector view and the caller has already set the seed. Must
        fit thresholds on benign val at cfg scoring.threshold_q and set
        manifest = artifacts.make_manifest(cfg, splits)."""

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path, cfg: Config) -> "Detector": ...

    @abc.abstractmethod
    def score_run(self, parquet_dir: str | Path, run_id: str, cfg: Config,
                  strip_verdict: bool | None = None) -> pd.DataFrame:
        """SCORE_COLUMNS frame for one ingested run. strip_verdict is the
        weak-signal honesty knob (suppress verdict-derived features); a
        detector without such features may ignore it — the with/strip AUCs
        coming out equal is then itself the answer."""

    @abc.abstractmethod
    def score_events(self, events: pd.DataFrame, entities: pd.DataFrame,
                     cfg: Config, only_entity: str | None = None
                     ) -> pd.DataFrame:
        """SCORE_COLUMNS frame for an in-memory (possibly perturbed) events
        table — the detector-agnostic perturbation-eval seam. Rows for
        only_entity must be complete for every window it appears in; rows for
        other entities may be omitted (speedup hint). Detectors with window
        memory must still walk all windows in temporal order."""

    def ablations(self, cfg: Config) -> dict[str, list[tuple[str, Any]]]:
        """Eval-time config forks: name -> [(dotted_key, value), ...].
        Forks re-SCORE with a modified config; they never retrain."""
        return {}
