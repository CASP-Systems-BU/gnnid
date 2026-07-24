"""Staged training: Word2Vec -> GraphSAGE (weighted CE) -> XGBoost.

All benign. Temporal split by run. Produces a scoring Bundle.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from . import artifacts, dataset
from .config import Config
from .embed import SentenceEmbedder
from .models.objectives import build_objective
from .models.sage import FlashSAGE
from .schema import ENTITY_TYPES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_weights(graphs, num_classes: int, other_idx: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for g in graphs:
        y = g.data.y.numpy()
        for c in y[y != other_idx]:
            counts[c] += 1
    counts[other_idx] = 0
    n_present = (counts > 0).sum() or 1
    total = counts.sum() or 1
    w = np.zeros(num_classes, dtype=np.float32)
    nz = counts > 0
    # sklearn "balanced": total / (n_classes * count)
    w[nz] = total / (n_present * counts[nz])
    return torch.from_numpy(w)


def _eval_loss(model, objective, loader) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch.x, batch.edge_index)
            tot += float(objective.loss(logits, batch))
            n += 1
    return tot / max(n, 1)


def train_gnn(train_graphs, val_graphs, num_classes: int, cfg: Config,
              other_idx: int = 0) -> FlashSAGE:
    dim = int(cfg.dotted_get("w2v.dim", 64))
    model = FlashSAGE(dim, int(cfg.model.hidden), int(cfg.model.embed),
                      num_classes, float(cfg.model.dropout))
    weights = _class_weights(train_graphs, num_classes, other_idx)
    objective = build_objective(cfg.model.objective, class_weights=weights,
                                other_idx=other_idx)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr),
                           weight_decay=float(cfg.train.weight_decay))
    bs = int(cfg.dotted_get("train.batch_windows", 8))
    train_loader = DataLoader([g.data for g in train_graphs], batch_size=bs,
                              shuffle=True)
    val_loader = DataLoader([g.data for g in val_graphs], batch_size=bs) \
        if val_graphs else None

    best_val, best_state, patience = float("inf"), None, 0
    max_ep = int(cfg.train.max_epochs)
    pat = int(cfg.train.patience)
    for ep in range(max_ep):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits, _ = model(batch.x, batch.edge_index)
            loss = objective.loss(logits, batch)
            loss.backward()
            opt.step()
        vloss = _eval_loss(model, objective, val_loader) if val_loader \
            else _eval_loss(model, objective, train_loader)
        if vloss < best_val - 1e-4:
            best_val, best_state, patience = vloss, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= pat:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _node_features(model, graphs):
    """concat(w2v, gnn-embed) per node, with labels and node types."""
    xs, ys, nts = [], [], []
    model.eval()
    with torch.no_grad():
        for g in graphs:
            _, emb = model(g.data.x, g.data.edge_index)
            xs.append(np.concatenate(
                [g.data.x.numpy(), emb.numpy()], axis=1))
            ys.append(g.data.y.numpy())
            nts.append(g.data.node_type.numpy())
    if not xs:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(nts)


def _train_booster(Xtr, ytr, Xv, yv, num_classes, cfg, other_idx):
    import xgboost as xgb
    mask = ytr != other_idx
    Xtr, ytr = Xtr[mask], ytr[mask]
    if len(np.unique(ytr)) < 2:
        return None
    classes, counts = np.unique(ytr, return_counts=True)
    cw = {c: len(ytr) / (len(classes) * n) for c, n in zip(classes, counts)}
    wtr = np.array([cw[c] for c in ytr], dtype=np.float32)
    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    evals = [(dtrain, "train")]
    if len(Xv):
        vm = yv != other_idx
        if vm.sum():
            evals.append((xgb.DMatrix(Xv[vm], label=yv[vm]), "val"))
    params = {"objective": "multi:softprob", "num_class": num_classes,
              "max_depth": int(cfg.dotted_get("train.xgb.max_depth", 6)),
              "eta": float(cfg.dotted_get("train.xgb.learning_rate", 0.1)),
              "eval_metric": "mlogloss", "seed": int(cfg.dotted_get("seed", 17))}
    return xgb.train(
        params, dtrain,
        num_boost_round=int(cfg.dotted_get("train.xgb.n_estimators", 300)),
        evals=evals,
        early_stopping_rounds=int(cfg.dotted_get("train.xgb.early_stopping_rounds", 20)),
        verbose_eval=False)


def train_xgb(model, train_graphs, val_graphs, num_classes: int, cfg: Config,
              other_idx: int = 0):
    """Two boosters: `full` on concat(w2v, gnn-embed) — the v1 detector — and
    `w2v` on the w2v half only, for the FLASH-Table-8 ablation (scoring.source)."""
    dim = int(cfg.dotted_get("w2v.dim", 64))
    Xtr, ytr, _ = _node_features(model, train_graphs)
    Xv, yv, _ = _node_features(model, val_graphs)
    full = _train_booster(Xtr, ytr, Xv, yv, num_classes, cfg, other_idx)
    w2v = _train_booster(Xtr[:, :dim], ytr,
                         Xv[:, :dim] if len(Xv) else Xv, yv,
                         num_classes, cfg, other_idx)
    return full, w2v


def _benign_val_norm(model, booster, val_graphs, cfg, vocab):
    """Per-node-type benign-val scores -> normalization tables + thresholds."""
    from .score import score_nodes
    by_type: dict[str, list[float]] = {t: [] for t in ENTITY_TYPES}
    for g in val_graphs:
        recs = score_nodes(g, model, booster, vocab, cfg, norm=None)
        for r in recs:
            by_type[r["node_type"]].append(r["score_raw"])
    norm = artifacts.quantile_norm_tables(by_type)
    q = float(cfg.dotted_get("scoring.threshold_q", 0.995))
    thresholds = {t: float(np.quantile(arr, q)) for t, arr in by_type.items() if arr}
    return norm, thresholds


def run_training(cfg: Config, repo_root: str | Path = ".") -> artifacts.Bundle:
    set_seed(int(cfg.dotted_get("seed", 17)))
    parquet_dir = Path(repo_root) / cfg.data.parquet_dir
    run_ids = dataset.list_run_ids(parquet_dir, cfg.data.benchmarks)
    if not run_ids:
        raise SystemExit(f"no ingested runs under {parquet_dir}; run `gnnid ingest` first")
    train_ids, val_ids, test_ids = dataset.temporal_split(
        run_ids, float(cfg.data.val_frac), float(cfg.data.test_frac))
    print(f"[train] runs: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test")

    # 1. Word2Vec on train sentences
    corpus = dataset.collect_sentences(parquet_dir, train_ids, cfg)
    print(f"[train] w2v corpus: {len(corpus)} sentences")
    embedder = SentenceEmbedder.train(corpus, cfg)

    # 2. label vocab + graphs
    vocab = dataset.fit_vocab(parquet_dir, train_ids)
    print(f"[train] label classes ({vocab.num_classes}): {vocab.classes}")
    train_graphs = dataset.build_graphs(parquet_dir, train_ids, embedder, vocab, cfg)
    val_graphs = dataset.build_graphs(parquet_dir, val_ids, embedder, vocab, cfg)
    print(f"[train] graphs: {len(train_graphs)} train / {len(val_graphs)} val windows")

    # 3. GNN role classifier
    gnn = train_gnn(train_graphs, val_graphs, vocab.num_classes, cfg,
                    vocab.other_idx)

    # 4. XGBoost on concat(w2v, gnn-embed) + a w2v-only ablation booster
    booster, booster_w2v = train_xgb(gnn, train_graphs, val_graphs,
                                     vocab.num_classes, cfg, vocab.other_idx)
    print(f"[train] xgb: {'trained' if booster is not None else 'skipped (few classes)'}")

    # 5. normalization + thresholds from benign val
    norm, thresholds = _benign_val_norm(gnn, booster, val_graphs, cfg, vocab)

    manifest = artifacts.make_manifest(
        cfg, {"train": train_ids, "val": val_ids, "test": test_ids})
    bundle = artifacts.Bundle(cfg, embedder, vocab, gnn, booster, booster_w2v,
                              norm, thresholds, manifest)
    out = Path(repo_root) / cfg.artifacts_dir
    bundle.save(out)
    print(f"[train] artifacts -> {out}")
    return bundle
