"""Scoring: per-node anomaly scores -> per-pod alerts.

The detector produces SCORE_COLUMNS records (raw score = 1 - p(true label),
normalized against its per-node-type benign-val CDF); this module applies the
per-type thresholds and aggregates per pod: a pod alerts when its per-window
max normalized score crosses the node-type threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset
from .config import Config, detector_view
from .detectors import load_detector
from .schema import POD


def apply_thresholds(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add threshold/is_alert columns to a SCORE_COLUMNS frame."""
    if not df.empty:
        df["threshold"] = df["node_type"].map(lambda t: thresholds.get(t, 1.0))
        df["is_alert"] = df["score_norm"] >= df["threshold"]
    return df


def pod_alerts(scores: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate per-pod across windows: max + top3-mean; alert if max crosses
    the node-type threshold."""
    if scores.empty:
        return scores
    pods = scores[scores.node_type == POD]
    out = []
    for (run_id, eid), grp in pods.groupby(["run_id", "entity_id"]):
        s = grp["score_norm"].to_numpy()
        top3 = np.sort(s)[::-1][:3]
        out.append({
            "run_id": run_id, "entity_id": eid,
            "windows": len(grp),
            "max_score": float(s.max()),
            "top3_mean": float(top3.mean()),
            "threshold": float(grp["threshold"].iloc[0]),
            "is_alert": bool((grp["is_alert"]).any()),
            "true_label": grp["true_label"].iloc[0],
        })
    res = pd.DataFrame(out).sort_values("max_score", ascending=False)
    return res.reset_index(drop=True)


def run_scoring(cfg: Config, repo_root: str | Path = ".",
                run_ids: list[str] | None = None) -> pd.DataFrame:
    dcfg = detector_view(cfg)
    parquet_dir = Path(repo_root) / dcfg.data.parquet_dir
    det = load_detector(dcfg, repo_root)
    if run_ids is None:
        run_ids = det.manifest.get("splits", {}).get("test") \
            or dataset.list_run_ids(parquet_dir, dcfg.data.benchmarks)
    all_scores = []
    for run_id in run_ids:
        all_scores.append(apply_thresholds(
            det.score_run(parquet_dir, run_id, dcfg), det.thresholds))
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    out = Path(repo_root) / str(dcfg.dotted_get("results_dir", "results_eval"))
    out.mkdir(parents=True, exist_ok=True)
    if not scores.empty:
        scores.to_parquet(out / "scores.parquet", index=False)
        alerts = pod_alerts(scores, dcfg)
        alerts.to_json(out / "alerts.json", orient="records", indent=2)
        n_alert = int(alerts["is_alert"].sum())
        print(f"[score] {len(run_ids)} runs, {len(scores)} node-windows, "
              f"{n_alert}/{len(alerts)} pods alerted")
    return scores
