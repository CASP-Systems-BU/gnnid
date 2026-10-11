"""Evaluation without attack data — detector-agnostic via the Detector API.

(a) benign FP proxy    misclassification + alert rate on held-out benign test
(b) weak-signal        ROC-AUC: pod-windows with DROPPED/POLICY_DENIED flows
                       should score higher (with and without verdict features)
(c) perturbations      per-perturbation AUC of perturbed vs original scores
(d) ablations          detector-declared scoring forks (flash: w2v/gnn/xgb)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset
from .config import Config, deep_copy, detector_view
from .detectors import Detector, load_detector
from .labels import OTHER
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


def weak_signal(det: Detector, parquet_dir, run_ids, cfg: Config) -> dict:
    out = {}
    for strip in (False, True):
        pos, neg = [], []
        for run_id in run_ids:
            flagged = _dropped_pods_per_window(parquet_dir, run_id, cfg)
            df = det.score_run(parquet_dir, run_id, cfg, strip_verdict=strip)
            if df.empty:
                continue
            df = df[df.node_type == POD]
            for r in df.itertuples(index=False):
                key = (r.run_id, r.w_idx, r.entity_id)
                (pos if key in flagged else neg).append(r.score_norm)
        out["strip_verdict" if strip else "with_verdict"] = {
            "auc": _auc(np.array(pos), np.array(neg)),
            "n_pos": len(pos), "n_neg": len(neg)}
    return out


def _victim_scores(det: Detector, events, entities, cfg: Config,
                   victim: str) -> list[float]:
    df = det.score_events(events, entities, cfg, only_entity=victim)
    if df.empty:
        return []
    return df[df.entity_id == victim]["score_norm"].tolist()


def perturbation_eval(det: Detector, parquet_dir, run_ids, cfg: Config,
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
            pert_events, victim = fn(rd.events, rng, rd.entities)
            if not victim:
                continue
            base = _victim_scores(det, rd.events, rd.entities, cfg, victim)
            pert = _victim_scores(det, pert_events, rd.entities, cfg, victim)
            orig_scores.extend(base)
            pert_scores.extend(pert)
            if pert and max(pert) >= det.thresholds.get("Pod", 1.0):
                detected += 1
            total += 1
        results[name] = {
            "auc": _auc(np.array(pert_scores), np.array(orig_scores)),
            "detection_rate": detected / total if total else float("nan"),
            "runs": total}
    return results


def ablation(det: Detector, parquet_dir, cfg: Config, test_ids) -> dict:
    from .score import apply_thresholds
    out = {}
    for name, mods in det.ablations(cfg).items():
        c = deep_copy(cfg)
        for key, value in mods:
            c.dotted_set(key, value)
        frames = [apply_thresholds(det.score_run(parquet_dir, r, c),
                                   det.thresholds) for r in test_ids]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out[name] = benign_fp(df)
    return out


def _malicious_node_windows(parquet_dir, run_id: str, cfg: Config):
    """(w_idx, entity_id) with >=1 incident malicious event -> the attack-touched
    node-windows of a labeled run. None if the run carries no labels."""
    rd = dataset.load_run(parquet_dir, run_id)
    if "label" not in rd.events.columns or rd.events["label"].isna().all():
        return None
    mal = rd.events[rd.events["label"] == "malicious"]
    touched = set()
    if mal.empty:
        return touched
    for win in dataset.run_windows(rd, cfg):
        d = mal[(mal.ts >= win.t0) & (mal.ts < win.t1)]
        for eid in set(d["src_id"]) | set(d["dst_id"]):
            touched.add((win.w_idx, eid))
    return touched


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec and rec and not (np.isnan(prec) or np.isnan(rec)) else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _ap(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision (PR-AUC) from positive/negative score arrays."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return float("nan")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(average_precision_score(y, s))


def labeled_detection(det: Detector, parquet_dir, run_ids, cfg: Config) -> dict:
    """Real supervised detection metrics on labeled attack runs.

    Node-window truth = any incident malicious event. Reported for two
    populations: `all_nodes`, and `known_role_nodes` (excludes the <other> class,
    i.e. the attacker pod and infra — whose role-mismatch score is 1.0 by
    construction and would trivially inflate the numbers). `attacker_nodes` is the
    sanity recall on the attacker's own <other> node-windows."""
    from .score import apply_thresholds
    pops = {"all_nodes": [], "known_role_nodes": []}  # lists of (score, y, is_alert)
    atk_n = atk_alert = 0
    scored_runs = []
    for run_id in run_ids:
        touched = _malicious_node_windows(parquet_dir, run_id, cfg)
        if touched is None:
            continue
        df = apply_thresholds(det.score_run(parquet_dir, run_id, cfg), det.thresholds)
        if df.empty:
            continue
        scored_runs.append(run_id)
        for r in df.itertuples(index=False):
            y = 1 if (r.w_idx, r.entity_id) in touched else 0
            row = (float(r.score_norm), y, bool(r.is_alert))
            pops["all_nodes"].append(row)
            if r.true_label != OTHER:
                pops["known_role_nodes"].append(row)
            elif y:  # attacker/other, positive -> sanity recall
                atk_n += 1
                atk_alert += int(r.is_alert)

    out = {"runs": scored_runs}
    for name, rows in pops.items():
        if not rows:
            out[name] = {"n_pos": 0, "n_neg": 0}
            continue
        s = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        a = np.array([r[2] for r in rows], dtype=int)
        pos, neg = s[y == 1], s[y == 0]
        out[name] = {
            "roc_auc": _auc(pos, neg),
            "pr_auc": _ap(pos, neg),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            **_prf(y, a),
        }
    out["attacker_nodes"] = {
        "n_windows": atk_n,
        "alert_recall": (atk_alert / atk_n) if atk_n else float("nan")}
    return out


def run_eval(cfg: Config, repo_root: str | Path = ".") -> dict:
    from .score import apply_thresholds
    dcfg = detector_view(cfg)
    parquet_dir = Path(repo_root) / dcfg.data.parquet_dir
    det = load_detector(dcfg, repo_root)
    splits = det.manifest.get("splits", {})
    test_ids = splits.get("test") or dataset.list_run_ids(parquet_dir,
                                                          dcfg.data.benchmarks)

    # Benign proxies run over benign test runs only; labeled detection runs over
    # the attack runs (held in test by dataset.split_runs).
    attack_ids = [r for r in test_ids if dataset.is_attack_run(parquet_dir, r)]
    benign_ids = [r for r in test_ids if r not in attack_ids] or test_ids

    frames = [apply_thresholds(det.score_run(parquet_dir, r, dcfg),
                               det.thresholds) for r in benign_ids]
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    report = {
        "test_runs": test_ids,
        "attack_runs": attack_ids,
        "benign_fp": benign_fp(scores),
        "weak_signal": weak_signal(det, parquet_dir, benign_ids, dcfg),
        "perturbations": perturbation_eval(det, parquet_dir, benign_ids, dcfg,
                                           int(dcfg.dotted_get("seed", 17))),
        "ablation": ablation(det, parquet_dir, dcfg, benign_ids),
    }
    if attack_ids:
        report["labeled_detection"] = labeled_detection(
            det, parquet_dir, attack_ids, dcfg)
    out = Path(repo_root) / str(dcfg.dotted_get("results_dir", "results_eval"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report
