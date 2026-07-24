"""Scoring: per-node anomaly scores -> per-pod alerts.

Score source is configurable (xgb | gnn | w2v) for ablations. Raw score =
1 - p(true label). Normalized against per-node-type benign-val CDF; a pod
alerts when its per-window max normalized score crosses the node-type threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import artifacts, dataset
from .config import Config
from .schema import ENTITY_TYPES, POD

_TYPE_NAME = list(ENTITY_TYPES)


def _probs(graph, model, booster, source: str, booster_w2v=None) -> np.ndarray:
    """Per-node class-probability matrix from the chosen source."""
    with torch.no_grad():
        logits, emb = model(graph.data.x, graph.data.edge_index)
    if source == "gnn":
        return F.softmax(logits, dim=-1).numpy()
    import xgboost as xgb
    if source == "w2v" and booster_w2v is not None:
        # ablation: XGBoost on the Word2Vec half only (FLASH Table 8)
        Xw = graph.data.x.numpy()
        p = booster_w2v.predict(xgb.DMatrix(Xw))
        return p if p.ndim == 2 else p.reshape(len(Xw), -1)
    if source == "xgb" and booster is not None:
        X = np.concatenate([graph.data.x.numpy(), emb.numpy()], axis=1)
        p = booster.predict(xgb.DMatrix(X))
        return p if p.ndim == 2 else p.reshape(len(X), -1)
    return F.softmax(logits, dim=-1).numpy()


def score_nodes(graph, model, booster, vocab, cfg: Config,
                norm: dict | None, booster_w2v=None) -> list[dict]:
    """One window graph -> per-node score records."""
    source = cfg.dotted_get("scoring.source", "xgb")
    probs = _probs(graph, model, booster, source, booster_w2v)
    y = graph.data.y.numpy()
    ntypes = graph.data.node_type.numpy()
    recs = []
    for i, eid in enumerate(graph.entity_ids):
        ntype = _TYPE_NAME[ntypes[i]]
        if y[i] == vocab.other_idx:
            p_true, pred, raw = 0.0, int(probs[i].argmax()), 1.0
        else:
            p_true = float(probs[i, y[i]]) if y[i] < probs.shape[1] else 0.0
            pred = int(probs[i].argmax())
            raw = 1.0 - p_true
        rec = {
            "run_id": graph.run_id, "w_idx": graph.w_idx, "entity_id": eid,
            "node_type": ntype, "true_label": vocab.classes[y[i]],
            "pred_label": vocab.classes[pred] if pred < len(vocab.classes) else "?",
            "p_true": p_true, "score_raw": raw,
        }
        if norm is not None:
            rec["score_norm"] = artifacts.normalize(raw, norm.get(ntype, []))
        recs.append(rec)
    return recs


def score_run(bundle: artifacts.Bundle, parquet_dir, run_id: str, cfg: Config,
              strip_verdict: bool | None = None) -> pd.DataFrame:
    graphs = dataset.build_graphs(parquet_dir, [run_id], bundle.embedder,
                                  bundle.vocab, cfg, strip_verdict)
    rows = []
    for g in graphs:
        rows.extend(score_nodes(g, bundle.gnn, bundle.xgb, bundle.vocab, cfg,
                                bundle.norm, bundle.xgb_w2v))
    df = pd.DataFrame(rows)
    if not df.empty:
        thr = bundle.thresholds
        df["threshold"] = df["node_type"].map(lambda t: thr.get(t, 1.0))
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
    parquet_dir = Path(repo_root) / cfg.data.parquet_dir
    bundle = artifacts.Bundle.load(Path(repo_root) / cfg.artifacts_dir, cfg)
    if run_ids is None:
        run_ids = bundle.manifest.get("splits", {}).get("test") \
            or dataset.list_run_ids(parquet_dir, cfg.data.benchmarks)
    all_scores = []
    for run_id in run_ids:
        all_scores.append(score_run(bundle, parquet_dir, run_id, cfg))
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    out = Path(repo_root) / "results_eval"
    out.mkdir(parents=True, exist_ok=True)
    if not scores.empty:
        scores.to_parquet(out / "scores.parquet", index=False)
        alerts = pod_alerts(scores, cfg)
        alerts.to_json(out / "alerts.json", orient="records", indent=2)
        n_alert = int(alerts["is_alert"].sum())
        print(f"[score] {len(run_ids)} runs, {len(scores)} node-windows, "
              f"{n_alert}/{len(alerts)} pods alerted")
    return scores
