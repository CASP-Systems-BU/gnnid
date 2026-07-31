"""PPT-GNN two-stage training.

Stage 1 (pretrain): self-supervised link prediction over all seven relations
of the benign train memory graphs. Per batch, edges are split 70/30 into
message-passing and supervision sets so the encoder cannot read a supervision
edge it must predict. The split is per LINK, not per relation: the four
spatial relations come in mirrored forward/reverse pairs (graph.py builds
them in lockstep), so a reverse edge is held out exactly when its forward
twin is — otherwise ~70% of held-out edges would stay visible to the encoder
through their mirror and the task degenerates to adjacency recall. Negatives
come from bipartite-aware negative_sampling and the per-relation BCE losses
are averaged with EQUAL weight — the dense spatial relations must not drown
the sparse recurrence one.

Stage 2 (finetune): the pre-trained encoder + a fresh classifier head, trained
on the role-classification pretext task with class weights and OTHER masking,
restricted to each memory graph's newest-window entity copies (ppt_cls).
Early stopping mirrors the FLASH loop (benign-val loss, min-delta 1e-4).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from ...config import Config
from ...labels import LabelVocab
from ...models.objectives import build_objective
from .features import PPTFeatureSpec
from .graph import ALL_EDGE_TYPES, MemoryGraph
from .model import PPTEncoder, PPTLinkPredictor, PPTModel


# ---------------------------------------------------------------- pre-training
_MIRROR = {("entity", "src_of", "event"): ("event", "rev_src_of", "entity"),
           ("entity", "dst_of", "event"): ("event", "rev_dst_of", "entity")}
_MIRRORED = set(_MIRROR.values())


def _cap(pos: torch.Tensor, max_pos: int) -> torch.Tensor:
    return pos[:, :max_pos] if pos.size(1) > max_pos else pos


def _link_loss(model: PPTLinkPredictor, batch, neg_ratio: float,
               holdout_frac: float, max_pos: int) -> torch.Tensor | None:
    msg_ei, sup = {}, {}
    for et in ALL_EDGE_TYPES:
        if et in _MIRRORED:
            continue                    # split together with the forward twin
        ei = batch[et].edge_index
        met = _MIRROR.get(et)
        if ei.size(1) == 0:
            msg_ei[et] = ei
            if met is not None:
                msg_ei[met] = batch[met].edge_index
            continue
        perm = torch.randperm(ei.size(1))
        k = int((1.0 - holdout_frac) * ei.size(1))
        msg_ei[et] = ei[:, perm[:k]]
        held = ei[:, perm[k:]]          # full held-out set, pre-cap
        if held.size(1):
            sup[et] = _cap(held, max_pos)
        if met is None:
            continue
        # reverse twin: a reverse edge goes to supervision iff its flipped
        # pair was held out of the forward relation (link-level holdout)
        rev = batch[met].edge_index
        if rev.size(1) == 0:
            msg_ei[met] = rev
            continue
        n_dst = batch[et[2]].num_nodes  # encode pairs in forward orientation
        held_ids = held[0] * n_dst + held[1]
        rev_ids = rev[1] * n_dst + rev[0]
        held_mask = torch.isin(rev_ids, held_ids)
        msg_ei[met] = rev[:, ~held_mask]
        rpos = rev[:, held_mask]
        if rpos.size(1):
            sup[met] = _cap(rpos, max_pos)
    if not sup:
        return None
    h = model.encoder(batch.x_dict, msg_ei)
    losses = []
    for et, pos in sup.items():
        neg = negative_sampling(
            batch[et].edge_index,
            num_nodes=(batch[et[0]].num_nodes, batch[et[2]].num_nodes),
            num_neg_samples=max(int(neg_ratio * pos.size(1)), 1))
        cand = torch.cat([pos, neg], dim=1)
        labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        logits = model.decode(h, et, cand)
        losses.append(F.binary_cross_entropy_with_logits(logits, labels))
    return torch.stack(losses).mean()


def _eval_link_loss(model, loader, neg_ratio, holdout_frac, max_pos,
                    seed: int) -> float:
    """Validation link loss with fixed RNG so the edge splits and negatives
    are identical across epochs (comparable early-stop signal). PyG's
    negative_sampling draws from STDLIB random (and numpy), which
    torch.random.fork_rng does not cover — pin and restore those too."""
    model.eval()
    tot, n = 0.0, 0
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(seed)
            for batch in loader:
                loss = _link_loss(model, batch, neg_ratio, holdout_frac,
                                  max_pos)
                if loss is not None:
                    tot += float(loss)
                    n += 1
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return tot / max(n, 1)


def pretrain(train_graphs: list[MemoryGraph], val_graphs: list[MemoryGraph],
             spec: PPTFeatureSpec, cfg: Config) -> dict | None:
    """Returns the best encoder state_dict, or None when disabled/no data."""
    if not bool(cfg.dotted_get("train.pretrain.enabled", True)):
        return None
    if not train_graphs:
        return None
    model = PPTLinkPredictor(PPTEncoder(
        spec.entity_dim, spec.event_dim, int(cfg.model.hidden),
        int(cfg.model.layers), float(cfg.model.dropout)))
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(cfg.dotted_get("train.pretrain.lr", 1e-4)))
    neg_ratio = float(cfg.dotted_get("train.pretrain.neg_ratio", 1.0))
    holdout = float(cfg.dotted_get("train.pretrain.holdout_frac", 0.3))
    max_pos = int(cfg.dotted_get("train.pretrain.max_pos_per_type", 2000))
    bs = int(cfg.dotted_get("train.pretrain.batch_graphs", 4))
    seed = int(cfg.dotted_get("seed", 17))
    train_loader = DataLoader([g.data for g in train_graphs], batch_size=bs,
                              shuffle=True)
    val_loader = DataLoader([g.data for g in val_graphs], batch_size=bs) \
        if val_graphs else train_loader

    best_val, best_state, patience = float("inf"), None, 0
    pat = int(cfg.dotted_get("train.pretrain.patience", 8))
    for ep in range(int(cfg.dotted_get("train.pretrain.epochs", 40))):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = _link_loss(model, batch, neg_ratio, holdout, max_pos)
            if loss is None:
                continue
            loss.backward()
            opt.step()
        vloss = _eval_link_loss(model, val_loader, neg_ratio, holdout,
                                max_pos, seed)
        if vloss < best_val - 1e-4:
            best_val, patience = vloss, 0
            best_state = {k: v.clone()
                          for k, v in model.encoder.state_dict().items()}
        else:
            patience += 1
            if patience >= pat:
                break
    print(f"[train] ppt pretrain: best val link loss {best_val:.4f}")
    return best_state if best_state is not None else \
        {k: v.clone() for k, v in model.encoder.state_dict().items()}


# ----------------------------------------------------------------- fine-tuning
def _class_weights_masked(graphs: list[MemoryGraph], num_classes: int,
                          other_idx: int) -> torch.Tensor:
    """flash._class_weights formula over newest-window entity copies only
    (each entity-window counts exactly once across a run's memory graphs)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for g in graphs:
        ent = g.data["entity"]
        y = ent.y[ent.score_mask].numpy()
        for c in y[y != other_idx]:
            counts[c] += 1
    counts[other_idx] = 0
    n_present = (counts > 0).sum() or 1
    total = counts.sum() or 1
    w = np.zeros(num_classes, dtype=np.float32)
    nz = counts > 0
    w[nz] = total / (n_present * counts[nz])
    return torch.from_numpy(w)


def _eval_cls_loss(model, objective, loader) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch)
            tot += float(objective.loss(logits, batch))
            n += 1
    return tot / max(n, 1)


def finetune(enc_state: dict | None, train_graphs: list[MemoryGraph],
             val_graphs: list[MemoryGraph], vocab: LabelVocab,
             spec: PPTFeatureSpec, cfg: Config) -> PPTModel:
    model = PPTModel(spec.entity_dim, spec.event_dim, int(cfg.model.hidden),
                     int(cfg.model.layers), vocab.num_classes,
                     float(cfg.model.dropout))
    if enc_state is not None:
        model.encoder.load_state_dict(enc_state)   # head stays random
        lr = float(cfg.dotted_get("train.finetune.lr", 1e-2))
    else:
        lr = float(cfg.dotted_get("train.scratch_lr", 1e-3))
    weights = _class_weights_masked(train_graphs, vocab.num_classes,
                                    vocab.other_idx)
    objective = build_objective(
        cfg.dotted_get("model.objective", "ppt_cls"),
        class_weights=weights, other_idx=vocab.other_idx)
    opt = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=float(cfg.dotted_get("train.finetune.weight_decay", 5e-4)))
    bs = int(cfg.dotted_get("train.finetune.batch_graphs", 4))
    train_loader = DataLoader([g.data for g in train_graphs], batch_size=bs,
                              shuffle=True)
    val_loader = DataLoader([g.data for g in val_graphs], batch_size=bs) \
        if val_graphs else None

    best_val, best_state, patience = float("inf"), None, 0
    pat = int(cfg.dotted_get("train.finetune.patience", 15))
    for ep in range(int(cfg.dotted_get("train.finetune.max_epochs", 100))):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits, _ = model(batch)
            loss = objective.loss(logits, batch)
            loss.backward()
            opt.step()
        vloss = _eval_cls_loss(model, objective, val_loader) if val_loader \
            else _eval_cls_loss(model, objective, train_loader)
        if vloss < best_val - 1e-4:
            best_val, patience = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= pat:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
