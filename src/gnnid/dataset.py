"""Dataset assembly: parquet runs -> windows -> sentences -> window graphs.

Centralizes the ingest->graph flow so train.py / score.py / eval.py share one
code path. Word2Vec must be trained on TRAIN sentences before any graph can be
built, so callers: (1) collect_sentences(train) -> train embedder,
(2) build_graphs(split, embedder, vocab).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .embed import SentenceEmbedder
from .graph import WindowGraph, build_window_graph
from .labels import LabelVocab
from .sentences import build_sentences
from .windows import iter_windows


@dataclass
class RunData:
    run_id: str
    events: pd.DataFrame
    entities: pd.DataFrame
    meta: dict


def load_run(parquet_dir: str | Path, run_id: str) -> RunData:
    d = Path(parquet_dir) / run_id
    events = pd.read_parquet(d / "events.parquet")
    entities = pd.read_parquet(d / "entities.parquet")
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return RunData(run_id, events, entities, meta)


def list_run_ids(parquet_dir: str | Path, benchmarks: list[str] | None = None
                 ) -> list[str]:
    parquet_dir = Path(parquet_dir)
    ids = []
    for d in parquet_dir.iterdir() if parquet_dir.exists() else []:
        if not (d / "events.parquet").exists():
            continue
        if benchmarks and d.name.split("-", 1)[0] not in benchmarks:
            continue
        ids.append(d.name)
    return sorted(ids, key=lambda r: r.rsplit("_", 1)[-1])  # temporal


def temporal_split(run_ids: list[str], val_frac: float, test_frac: float
                   ) -> tuple[list[str], list[str], list[str]]:
    """Split BY RUN in temporal order (never random — overlapping windows leak).
    Latest runs are val/test."""
    n = len(run_ids)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n >= 3 else (1 if n == 2 else 0)
    n_train = n - n_val - n_test
    return run_ids[:n_train], run_ids[n_train:n_train + n_val], \
        run_ids[n_train + n_val:]


def entity_meta_map(entities: pd.DataFrame) -> dict:
    return {row.entity_id: (row.entity_type, row.canonical_service, row.namespace)
            for row in entities.itertuples(index=False)}


def run_windows(rd: RunData, cfg: Config):
    return iter_windows(
        rd.events,
        width_s=float(cfg.windows.width_s), stride_s=float(cfg.windows.stride_s),
        min_tail_s=float(cfg.dotted_get("windows.min_tail_s", cfg.windows.width_s)),
        run_id=rd.run_id)


def collect_sentences(parquet_dir, run_ids: list[str], cfg: Config,
                      strip_verdict: bool | None = None) -> list[list[str]]:
    """All window sentences across the given runs (w2v training corpus)."""
    out = []
    for run_id in run_ids:
        rd = load_run(parquet_dir, run_id)
        meta = entity_meta_map(rd.entities)
        for win in run_windows(rd, cfg):
            sents = build_sentences(win.events, set(meta), cfg, strip_verdict)
            out.extend(toks for toks in sents.values() if toks)
    return out


def build_graphs(parquet_dir, run_ids: list[str], embedder: SentenceEmbedder,
                 vocab: LabelVocab, cfg: Config,
                 strip_verdict: bool | None = None) -> list[WindowGraph]:
    graphs: list[WindowGraph] = []
    for run_id in run_ids:
        rd = load_run(parquet_dir, run_id)
        meta = entity_meta_map(rd.entities)
        for win in run_windows(rd, cfg):
            sents = build_sentences(win.events, set(meta), cfg, strip_verdict)
            embs = embedder.encode_many(sents)
            g = build_window_graph(win, meta, embs, vocab, cfg)
            if g is not None:
                graphs.append(g)
    return graphs


def fit_vocab(parquet_dir, run_ids: list[str], app_namespace: str = "default"
              ) -> LabelVocab:
    frames = [load_run(parquet_dir, r).entities for r in run_ids]
    allent = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return LabelVocab.fit(allent, app_namespace)
