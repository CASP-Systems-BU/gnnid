"""Objective seam — the config swap point for future detectors.

`flash_cls` (v1): weighted CE role classification; anomaly = 1 - p(true label).
`masked_recon` (future GraphMAE/MAGIC): registered stub, not implemented.
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class Objective(Protocol):
    def loss(self, logits, batch) -> torch.Tensor: ...
    def node_scores(self, logits, batch) -> torch.Tensor: ...


class FlashClassification:
    """Weighted CE over role classes; OTHER (index 0) excluded from loss."""

    def __init__(self, class_weights: torch.Tensor | None, other_idx: int = 0):
        self.class_weights = class_weights
        self.other_idx = other_idx

    def loss(self, logits, batch) -> torch.Tensor:
        y = batch.y
        mask = y != self.other_idx
        if mask.sum() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], y[mask], weight=self.class_weights)

    def node_scores(self, logits, batch) -> torch.Tensor:
        """Anomaly score per node = 1 - p(true label). OTHER nodes -> 1.0
        (an unknown workload is itself an alert)."""
        p = F.softmax(logits, dim=-1)
        y = batch.y
        p_true = p.gather(1, y.clamp(min=0).unsqueeze(1)).squeeze(1)
        scores = 1.0 - p_true
        scores = torch.where(y == self.other_idx,
                             torch.ones_like(scores), scores)
        return scores


_REGISTRY = {"flash_cls": FlashClassification}


def build_objective(name: str, **kw) -> Objective:
    if name == "masked_recon":
        raise NotImplementedError(
            "masked_recon objective is a registered stub for a future config "
            "swap; v1 uses flash_cls.")
    if name not in _REGISTRY:
        raise ValueError(f"unknown objective {name!r}; have {list(_REGISTRY)}")
    return _REGISTRY[name](**kw)
