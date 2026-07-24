"""Evaluation without attack data.

(a) benign FP proxy    misclassification + alert rate on held-out benign test
(b) weak-signal        ROC-AUC: pod-windows with DROPPED/POLICY_DENIED flows
                       should score higher (with and without verdict tokens)
(c) perturbations      per-perturbation AUC of perturbed vs original scores
(d) ablations          w2v-only vs gnn-only vs concat (xgb)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts, dataset
from .config import Config, deep_copy
from .schema import L4_FLOW, POD


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via Mann-Whitney U (P(score_pos > score_neg))."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def benign_fp(scores: pd.DataFrame) -> dict:
    if scores.empty:
        return {}
    pods = scores[scores.node_type == POD]
    labeled = pods[pods.true_label != "<other>"]
    misclf = float((labeled.true_label != labeled.pred_label).mean()) \
        if not labeled.empty else float("nan")
    return {"node_windows": int(len(pods)),
            "misclassification_rate": misclf,
            "alert_rate": float(pods["is_alert"].mean())}


def _dropped_pods_per_window(parquet_dir, run_id: str, cfg: Config) -> set:
    """(run_id, w_idx, entity_id) with >=1 DROPPED/denied L4 flow."""
    rd = dataset.load_run(parquet_dir, run_id)
    flagged = set()
    dropped = rd.events[(rd.events.event_type == L4_FLOW) &
                        (rd.events.verdict != "FORWARDED")]
    if dropped.empty:
        return flagged
    for win in dataset.run_windows(rd, cfg):
        d = dropped[(dropped.ts >= win.t0) & (dropped.ts < win.t1)]
        for eid in set(d["src_id"]) | set(d["dst_id"]):
            flagged.add((run_id, win.w_idx, eid))
    return flagged


def weak_signal(bundle, parquet_dir, run_ids, cfg: Config) -> dict:
    from .score import score_run
    out = {}
    for strip in (False, True):
        pos, neg = [], []
        for run_id in run_ids:
            flagged = _dropped_pods_per_window(parquet_dir, run_id, cfg)
            df = score_run(bundle, parquet_dir, run_id, cfg, strip_verdict=strip)
            df = df[df.node_type == POD]
            for r in df.itertuples(index=False):
                key = (r.run_id, r.w_idx, r.entity_id)
                (pos if key in flagged else neg).append(r.score_norm)
        out["strip_verdict" if strip else "with_verdict"] = {
            "auc": _auc(np.array(pos), np.array(neg)),
            "n_pos": len(pos), "n_neg": len(neg)}
    return out


def perturbation_eval(bundle, parquet_dir, run_ids, cfg: Config,
                      seed: int = 17) -> dict:
    from .perturb import PERTURBATIONS
    rng = np.random.default_rng(seed)
    results = {}
    for name, fn in PERTURBATIONS.items():
        orig_scores, pert_scores = [], []
        detected = 0
        total = 0
        for run_id in run_ids:
            rd = dataset.load_run(parquet_dir, run_id)
            meta = dataset.entity_meta_map(rd.entities)
            pert_events, victim = fn(rd.events, rng, rd.entities)
            if not victim:
                continue
            base = _score_events(bundle, rd.events, meta, cfg, victim)
            pert = _score_events(bundle, pert_events, meta, cfg, victim)
            orig_scores.extend(base)
            pert_scores.extend(pert)
            if pert and max(pert) >= bundle.thresholds.get("Pod", 1.0):
                detected += 1
            total += 1
        results[name] = {
            "auc": _auc(np.array(pert_scores), np.array(orig_scores)),
            "detection_rate": detected / total if total else float("nan"),
            "runs": total}
    return results


def _score_events(bundle, events, meta, cfg: Config, victim: str) -> list[float]:
    """Score just the victim pod's window-normalized scores for the given events."""
    from .graph import build_window_graph
    from .score import score_nodes
    from .windows import iter_windows
    rd_run = events["run_id"].iloc[0] if not events.empty else "x"
    scores = []
    for win in iter_windows(events.sort_values("ts"), float(cfg.windows.width_s),
                            float(cfg.windows.stride_s),
                            float(cfg.dotted_get("windows.min_tail_s", cfg.windows.width_s)),
                            rd_run):
        # only windows the victim participates in matter (big speedup)
        we = win.events
        if not ((we.src_id == victim) | (we.dst_id == victim)).any():
            continue
        from .sentences import build_sentences
        sents = build_sentences(win.events, set(meta), cfg)
        embs = bundle.embedder.encode_many(sents)
        g = build_window_graph(win, meta, embs, bundle.vocab, cfg)
        if g is None:
            continue
        for r in score_nodes(g, bundle.gnn, bundle.xgb, bundle.vocab, cfg,
                             bundle.norm, bundle.xgb_w2v):
            if r["entity_id"] == victim:
                scores.append(r["score_norm"])
    return scores


def ablation(cfg: Config, repo_root, parquet_dir, bundle, test_ids) -> dict:
    from .score import score_run
    out = {}
    for source in ("w2v", "gnn", "xgb"):
        c = deep_copy(cfg)
        c.dotted_set("scoring.source", source)
        frames = [score_run(bundle, parquet_dir, r, c) for r in test_ids]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out[source] = benign_fp(df)
    return out


def run_eval(cfg: Config, repo_root: str | Path = ".") -> dict:
    parquet_dir = Path(repo_root) / cfg.data.parquet_dir
    bundle = artifacts.Bundle.load(Path(repo_root) / cfg.artifacts_dir, cfg)
    splits = bundle.manifest.get("splits", {})
    test_ids = splits.get("test") or dataset.list_run_ids(parquet_dir, cfg.data.benchmarks)

    from .score import score_run
    frames = [score_run(bundle, parquet_dir, r, cfg) for r in test_ids]
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    report = {
        "test_runs": test_ids,
        "benign_fp": benign_fp(scores),
        "weak_signal": weak_signal(bundle, parquet_dir, test_ids, cfg),
        "perturbations": perturbation_eval(bundle, parquet_dir, test_ids, cfg,
                                           int(cfg.dotted_get("seed", 17))),
        "ablation": ablation(cfg, repo_root, parquet_dir, bundle, test_ids),
    }
    out = Path(repo_root) / "results_eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report
