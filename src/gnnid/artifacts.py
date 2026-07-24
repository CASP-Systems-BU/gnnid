"""Artifact bundle: everything needed to score a new run.

<artifacts_dir>/
    w2v.kv            Word2Vec KeyedVectors
    label_vocab.json  role/type classes
    gnn.pt            FlashSAGE state_dict + arch config
    xgb.json          XGBoost booster
    norm.json         per-node-type benign-val score CDFs (for normalization)
    thresholds.json   per-node-type alert thresholds
    manifest.json     config + split run_ids + git sha
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .embed import SentenceEmbedder
from .labels import LabelVocab
from .models.sage import FlashSAGE


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                              capture_output=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class Bundle:
    cfg: Config
    embedder: SentenceEmbedder
    vocab: LabelVocab
    gnn: FlashSAGE
    xgb: object | None                 # xgboost.Booster (concat features) or None
    xgb_w2v: object | None             # xgboost.Booster (w2v-half only), ablation
    norm: dict                         # node_type -> sorted score array (list)
    thresholds: dict                   # node_type -> float
    manifest: dict

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
    def load(cls, path: str | Path, cfg: Config) -> "Bundle":
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


def make_manifest(cfg: Config, splits: dict) -> dict:
    return {"git_sha": _git_sha(), "splits": splits, "config": dict(cfg)}


def quantile_norm_tables(scores_by_type: dict[str, list[float]]) -> dict:
    """Per node-type: store the sorted benign-val score array; normalization at
    inference is the empirical quantile rank against it."""
    return {t: sorted(float(s) for s in arr)
            for t, arr in scores_by_type.items() if arr}


def normalize(score: float, sorted_arr: list[float]) -> float:
    if not sorted_arr:
        return score
    lo = np.searchsorted(sorted_arr, score, side="left")
    hi = np.searchsorted(sorted_arr, score, side="right")
    return (lo + hi) / (2.0 * len(sorted_arr))
