"""FLASH detector: Word2Vec sentences -> GraphSAGE role classifier -> XGBoost.

The v1 pipeline (previously spread over train.py / score.py / eval.py /
artifacts.Bundle) behind the Detector interface. Code is moved verbatim:
behavior, artifact filenames, and the RNG stream are unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .. import artifacts, dataset
from ..config import Config
from ..embed import SentenceEmbedder
from ..labels import LabelVocab
from ..models.objectives import build_objective
from ..models.sage import FlashSAGE
from ..schema import ENTITY_TYPES
from . import register
from .base import Detector

_TYPE_NAME = list(ENTITY_TYPES)


# ------------------------------------------------------------------ training
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


def _benign_val_norm(model, booster, val_graphs, cfg, vocab, booster_w2v=None):
    """Per-node-type benign-val scores -> normalization tables + thresholds."""
    by_type: dict[str, list[float]] = {t: [] for t in ENTITY_TYPES}
    for g in val_graphs:
        recs = score_nodes(g, model, booster, vocab, cfg, norm=None,
                           booster_w2v=booster_w2v)
        for r in recs:
            by_type[r["node_type"]].append(r["score_raw"])
    q = float(cfg.dotted_get("scoring.threshold_q", 0.995))
    return artifacts.fit_norm_thresholds(by_type, q)


# ------------------------------------------------------------------- scoring
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


# ------------------------------------------------------------------ detector
@register
@dataclass
class FlashDetector(Detector):
    cfg: Config
    embedder: SentenceEmbedder
    vocab: LabelVocab
    gnn: FlashSAGE
    xgb: object | None                 # xgboost.Booster (concat features) or None
    xgb_w2v: object | None             # xgboost.Booster (w2v-half only), ablation
    norm: dict                         # node_type -> sorted score array (list)
    thresholds: dict                   # node_type -> float
    manifest: dict

    name: ClassVar[str] = "flash"

    @classmethod
    def train(cls, cfg: Config, parquet_dir: Path,
              splits: dict[str, list[str]]) -> "FlashDetector":
        train_ids, val_ids = splits["train"], splits["val"]

        # 1. Word2Vec on train sentences
        corpus = dataset.collect_sentences(parquet_dir, train_ids, cfg)
        print(f"[train] w2v corpus: {len(corpus)} sentences")
        embedder = SentenceEmbedder.train(corpus, cfg)

        # 2. label vocab + graphs
        vocab = dataset.fit_vocab(parquet_dir, train_ids)
        print(f"[train] label classes ({vocab.num_classes}): {vocab.classes}")
        train_graphs = dataset.build_graphs(parquet_dir, train_ids, embedder,
                                            vocab, cfg)
        val_graphs = dataset.build_graphs(parquet_dir, val_ids, embedder,
                                          vocab, cfg)
        print(f"[train] graphs: {len(train_graphs)} train / {len(val_graphs)} "
              f"val windows")

        # 3. GNN role classifier
        gnn = train_gnn(train_graphs, val_graphs, vocab.num_classes, cfg,
                        vocab.other_idx)

        # 4. XGBoost on concat(w2v, gnn-embed) + a w2v-only ablation booster
        booster, booster_w2v = train_xgb(gnn, train_graphs, val_graphs,
                                         vocab.num_classes, cfg,
                                         vocab.other_idx)
        print(f"[train] xgb: {'trained' if booster is not None else 'skipped (few classes)'}")

        # 5. normalization + thresholds from benign val
        norm, thresholds = _benign_val_norm(gnn, booster, val_graphs, cfg, vocab)

        manifest = artifacts.make_manifest(cfg, splits)
        return cls(cfg, embedder, vocab, gnn, booster, booster_w2v, norm,
                   thresholds, manifest)

    # --------------------------------------------------------------- save/load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.embedder.save(path / "w2v.kv")
        self.vocab.save(path / "label_vocab.json")
        torch.save({"state_dict": self.gnn.state_dict(),
                    "arch": {"in_dim": self.gnn.conv1.in_channels,
                             "hidden": self.gnn.conv1.out_channels,
                             "embed": self.gnn.conv2.out_channels,
                             "num_classes": self.gnn.head.out_features,
                             "dropout": self.gnn.dropout}},
                   path / "gnn.pt")
        if self.xgb is not None:
            self.xgb.save_model(str(path / "xgb.json"))
        if self.xgb_w2v is not None:
            self.xgb_w2v.save_model(str(path / "xgb_w2v.json"))
        with open(path / "norm.json", "w") as f:
            json.dump(self.norm, f)
        with open(path / "thresholds.json", "w") as f:
            json.dump(self.thresholds, f, indent=2)
        with open(path / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, cfg: Config) -> "FlashDetector":
        import xgboost as xgb
        path = Path(path)
        dim = int(cfg.dotted_get("w2v.dim", 64))
        embedder = SentenceEmbedder.load(path / "w2v.kv", dim)
        vocab = LabelVocab.load(path / "label_vocab.json")
        ck = torch.load(path / "gnn.pt", weights_only=False)
        gnn = FlashSAGE(**ck["arch"])
        gnn.load_state_dict(ck["state_dict"])
        gnn.eval()
        booster = None
        if (path / "xgb.json").exists():
            booster = xgb.Booster()
            booster.load_model(str(path / "xgb.json"))
        booster_w2v = None
        if (path / "xgb_w2v.json").exists():
            booster_w2v = xgb.Booster()
            booster_w2v.load_model(str(path / "xgb_w2v.json"))
        with open(path / "norm.json") as f:
            norm = json.load(f)
        with open(path / "thresholds.json") as f:
            thresholds = json.load(f)
        manifest = {}
        if (path / "manifest.json").exists():
            with open(path / "manifest.json") as f:
                manifest = json.load(f)
        return cls(cfg, embedder, vocab, gnn, booster, booster_w2v, norm,
                   thresholds, manifest)

    # ----------------------------------------------------------------- scoring
    def score_run(self, parquet_dir: str | Path, run_id: str, cfg: Config,
                  strip_verdict: bool | None = None) -> pd.DataFrame:
        graphs = dataset.build_graphs(parquet_dir, [run_id], self.embedder,
                                      self.vocab, cfg, strip_verdict)
        rows = []
        for g in graphs:
            rows.extend(score_nodes(g, self.gnn, self.xgb, self.vocab, cfg,
                                    self.norm, self.xgb_w2v))
        return pd.DataFrame(rows)

    def score_events(self, events: pd.DataFrame, entities: pd.DataFrame,
                     cfg: Config, only_entity: str | None = None
                     ) -> pd.DataFrame:
        from ..graph import build_window_graph
        from ..sentences import build_sentences
        from ..windows import iter_windows
        meta = dataset.entity_meta_map(entities)
        run_id = events["run_id"].iloc[0] if not events.empty else "x"
        rows = []
        for win in iter_windows(events.sort_values("ts"),
                                float(cfg.windows.width_s),
                                float(cfg.windows.stride_s),
                                float(cfg.dotted_get("windows.min_tail_s",
                                                     cfg.windows.width_s)),
                                run_id):
            # only windows the target participates in matter (big speedup)
            if only_entity is not None:
                we = win.events
                if not ((we.src_id == only_entity) |
                        (we.dst_id == only_entity)).any():
                    continue
            sents = build_sentences(win.events, set(meta), cfg)
            embs = self.embedder.encode_many(sents)
            g = build_window_graph(win, meta, embs, self.vocab, cfg)
            if g is None:
                continue
            rows.extend(score_nodes(g, self.gnn, self.xgb, self.vocab, cfg,
                                    self.norm, self.xgb_w2v))
        return pd.DataFrame(rows)

    def ablations(self, cfg: Config) -> dict[str, list[tuple[str, Any]]]:
        return {"w2v": [("scoring.source", "w2v")],
                "gnn": [("scoring.source", "gnn")],
                "xgb": [("scoring.source", "xgb")]}
