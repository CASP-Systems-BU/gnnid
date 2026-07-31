"""PPT-GNN detector (Van Langendonck et al. 2024) ported to K8s telemetry.

Spatio-temporal heterogeneous GNN over sliding-window memory graphs: telemetry
events become typed "event" nodes (line-graph style) with identity-free
features, canonical entities the "entity" endpoints; self-supervised
link-prediction pre-training on benign runs, then fine-tuning on the
role-classification pretext task. Anomaly = 1 - p(true role), the same
scoring/threshold scheme as FLASH, so norm/thresholds/pod-aggregation/eval
are shared unchanged.

Artifact bundle:
    label_vocab.json    role/type classes (shared LabelVocab format)
    feature_spec.json   vocabs + scalers + encoding dims (PPTFeatureSpec)
    ppt.pt              PPTModel state_dict + arch
    pretrained.pt       encoder-only state_dict from stage 1 (optional)
    norm.json, thresholds.json, manifest.json   same formats as flash
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch

from ... import artifacts, dataset
from ...config import Config
from ...labels import LabelVocab
from .. import register
from ..base import Detector
from .features import PPTFeatureSpec
from .graph import build_memory_graphs, build_run_memory_graphs
from .model import PPTModel
from .score import benign_val_norm, score_memory_graph
from .train import finetune, pretrain


@register
@dataclass
class PPTDetector(Detector):
    cfg: Config
    vocab: LabelVocab
    spec: PPTFeatureSpec
    model: PPTModel
    enc_state: dict | None             # stage-1 encoder weights (reproducibility)
    norm: dict                         # node_type -> sorted score array (list)
    thresholds: dict                   # node_type -> float
    manifest: dict

    name: ClassVar[str] = "ppt"

    @classmethod
    def train(cls, cfg: Config, parquet_dir: Path,
              splits: dict[str, list[str]]) -> "PPTDetector":
        train_ids, val_ids = splits["train"], splits["val"]

        vocab = dataset.fit_vocab(parquet_dir, train_ids)
        print(f"[train] label classes ({vocab.num_classes}): {vocab.classes}")
        spec = PPTFeatureSpec.fit(
            [dataset.load_run(parquet_dir, r).events for r in train_ids], cfg)

        train_graphs = build_run_memory_graphs(parquet_dir, train_ids, vocab,
                                               spec, cfg)
        val_graphs = build_run_memory_graphs(parquet_dir, val_ids, vocab,
                                             spec, cfg)
        print(f"[train] memory graphs: {len(train_graphs)} train / "
              f"{len(val_graphs)} val")

        enc_state = pretrain(train_graphs, val_graphs, spec, cfg)
        model = finetune(enc_state, train_graphs, val_graphs, vocab, spec, cfg)

        norm, thresholds = benign_val_norm(model, val_graphs, cfg, vocab)
        manifest = artifacts.make_manifest(cfg, splits)
        return cls(cfg, vocab, spec, model, enc_state, norm, thresholds,
                   manifest)

    # --------------------------------------------------------------- save/load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vocab.save(path / "label_vocab.json")
        self.spec.save(path / "feature_spec.json")
        torch.save({"state_dict": self.model.state_dict(),
                    "arch": self.model.arch}, path / "ppt.pt")
        if self.enc_state is not None:
            torch.save(self.enc_state, path / "pretrained.pt")
        with open(path / "norm.json", "w") as f:
            json.dump(self.norm, f)
        with open(path / "thresholds.json", "w") as f:
            json.dump(self.thresholds, f, indent=2)
        with open(path / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, cfg: Config) -> "PPTDetector":
        path = Path(path)
        vocab = LabelVocab.load(path / "label_vocab.json")
        spec = PPTFeatureSpec.load(path / "feature_spec.json")
        ck = torch.load(path / "ppt.pt", weights_only=False)
        model = PPTModel(**ck["arch"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        enc_state = None
        if (path / "pretrained.pt").exists():
            enc_state = torch.load(path / "pretrained.pt", weights_only=False)
        with open(path / "norm.json") as f:
            norm = json.load(f)
        with open(path / "thresholds.json") as f:
            thresholds = json.load(f)
        manifest = {}
        if (path / "manifest.json").exists():
            with open(path / "manifest.json") as f:
                manifest = json.load(f)
        return cls(cfg, vocab, spec, model, enc_state, norm, thresholds,
                   manifest)

    # ----------------------------------------------------------------- scoring
    def score_run(self, parquet_dir: str | Path, run_id: str, cfg: Config,
                  strip_verdict: bool | None = None) -> pd.DataFrame:
        rd = dataset.load_run(parquet_dir, run_id)
        return self._score_frame(rd.events, rd.entities, cfg,
                                 strip_verdict=bool(strip_verdict),
                                 only_entity=None, run_id=run_id)

    def score_events(self, events: pd.DataFrame, entities: pd.DataFrame,
                     cfg: Config, only_entity: str | None = None
                     ) -> pd.DataFrame:
        run_id = events["run_id"].iloc[0] if not events.empty else "x"
        return self._score_frame(events, entities, cfg, strip_verdict=False,
                                 only_entity=only_entity, run_id=run_id)

    def _score_frame(self, events, entities, cfg, strip_verdict, only_entity,
                     run_id) -> pd.DataFrame:
        meta = dataset.entity_meta_map(entities)
        rows = []
        for g in build_memory_graphs(events.sort_values("ts", kind="stable"),
                                     meta, self.vocab, self.spec, cfg, run_id,
                                     strip_verdict):
            if only_entity is not None:
                # every window is still walked (memory stays intact); the
                # forward pass is skipped when the target has no newest copy
                mask = g.data["entity"].score_mask.tolist()
                if not any(eid == only_entity and m
                           for eid, m in zip(g.entity_ids, mask)):
                    continue
            rows.extend(score_memory_graph(g, self.model, self.vocab,
                                           self.norm))
        return pd.DataFrame(rows)
