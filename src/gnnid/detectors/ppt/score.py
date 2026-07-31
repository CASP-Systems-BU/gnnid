"""PPT scoring: newest-window entity copies -> SCORE_COLUMNS records.

Exact mirror of the FLASH record semantics (score.py lineage): raw score =
1 - p(true role), OTHER forced to 1.0, normalized against the per-node-type
benign-val quantile tables. Only score_mask (newest-window) copies emit
records — each entity-window is scored exactly once across a run.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ... import artifacts
from ...config import Config
from ...labels import LabelVocab
from ...schema import ENTITY_TYPES
from .graph import MemoryGraph

_TYPE_NAME = list(ENTITY_TYPES)


def score_memory_graph(g: MemoryGraph, model, vocab: LabelVocab,
                       norm: dict | None) -> list[dict]:
    model.eval()
    with torch.no_grad():
        logits, _ = model(g.data)
        probs = F.softmax(logits, dim=-1).numpy()
    ent = g.data["entity"]
    y = ent.y.numpy()
    ntypes = ent.node_type.numpy()
    mask = ent.score_mask.numpy()
    recs = []
    for i, eid in enumerate(g.entity_ids):
        if not mask[i]:
            continue
        ntype = _TYPE_NAME[ntypes[i]]
        if y[i] == vocab.other_idx:
            p_true, pred, raw = 0.0, int(probs[i].argmax()), 1.0
        else:
            p_true = float(probs[i, y[i]]) if y[i] < probs.shape[1] else 0.0
            pred = int(probs[i].argmax())
            raw = 1.0 - p_true
        rec = {
            "run_id": g.run_id, "w_idx": g.w_idx, "entity_id": eid,
            "node_type": ntype, "true_label": vocab.classes[y[i]],
            "pred_label": vocab.classes[pred] if pred < len(vocab.classes) else "?",
            "p_true": p_true, "score_raw": raw,
        }
        if norm is not None:
            rec["score_norm"] = artifacts.normalize(raw, norm.get(ntype, []))
        recs.append(rec)
    return recs


def benign_val_norm(model, val_graphs: list[MemoryGraph], cfg: Config,
                    vocab: LabelVocab) -> tuple[dict, dict]:
    """Per-node-type benign-val scores -> normalization tables + thresholds."""
    by_type: dict[str, list[float]] = {t: [] for t in ENTITY_TYPES}
    for g in val_graphs:
        for r in score_memory_graph(g, model, vocab, norm=None):
            by_type[r["node_type"]].append(r["score_raw"])
    q = float(cfg.dotted_get("scoring.threshold_q", 0.995))
    return artifacts.fit_norm_thresholds(by_type, q)
