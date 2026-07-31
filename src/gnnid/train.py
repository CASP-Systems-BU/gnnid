"""Training entry point: seed, temporal split by run, then hand off to the
configured detector (all training stages happen inside Detector.train)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from . import dataset
from .config import Config, detector_view
from .detectors import Detector, get_detector, save_detector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(cfg: Config, repo_root: str | Path = ".") -> Detector:
    dcfg = detector_view(cfg)
    set_seed(int(dcfg.dotted_get("seed", 17)))
    parquet_dir = Path(repo_root) / dcfg.data.parquet_dir
    run_ids = dataset.list_run_ids(parquet_dir, dcfg.data.benchmarks)
    if not run_ids:
        raise SystemExit(f"no ingested runs under {parquet_dir}; run `gnnid ingest` first")
    train_ids, val_ids, test_ids = dataset.temporal_split(
        run_ids, float(dcfg.data.val_frac), float(dcfg.data.test_frac))
    name = dcfg.dotted_get("detector", "flash")
    print(f"[train] detector={name} windows={dict(dcfg.windows)}")
    print(f"[train] runs: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test")

    det = get_detector(name).train(
        dcfg, parquet_dir,
        {"train": train_ids, "val": val_ids, "test": test_ids})

    out = Path(repo_root) / dcfg.artifacts_dir
    save_detector(det, out)
    print(f"[train] artifacts -> {out}")
    return det
