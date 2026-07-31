"""Detector-interface acceptance test, parametrized over the registry.

Every registered detector must train on the 3-run fixture, emit SCORE_COLUMNS
records, survive a save/load round trip, honor strip_verdict, and score
perturbed in-memory events for a victim entity. Adding a detector means
making this file pass for it.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from gnnid import detectors
from gnnid.config import detector_view
from gnnid.dataset import load_run
from gnnid.detectors.base import SCORE_COLUMNS
from gnnid.perturb import dns_exfil
from gnnid.train import set_seed

# per-detector speed overrides (dotted keys set on the BASE config; detector-
# scoped ones go under detectors.<name>. so they survive the overlay merge)
SPEED = {
    "flash": [
        ("w2v.epochs", 2), ("w2v.min_count", 1), ("w2v.workers", 1),
        ("train.max_epochs", 3), ("train.patience", 2),
        ("train.xgb.n_estimators", 5), ("train.xgb.early_stopping_rounds", 2),
    ],
    "ppt": [
        ("detectors.ppt.train.pretrain.epochs", 2),
        ("detectors.ppt.train.pretrain.patience", 1),
        ("detectors.ppt.train.finetune.max_epochs", 2),
        ("detectors.ppt.train.finetune.patience", 1),
    ],
}


def _train(name, parquet_runs, cfg):
    parquet_dir, run_ids = parquet_runs
    cfg.dotted_set("detector", name)
    for k, v in SPEED.get(name, []):
        cfg.dotted_set(k, v)
    dcfg = detector_view(cfg)
    set_seed(17)
    splits = {"train": run_ids[:1], "val": run_ids[1:2], "test": run_ids[2:]}
    try:
        det = detectors.get_detector(name).train(dcfg, parquet_dir, splits)
    except NotImplementedError:
        pytest.skip(f"{name} detector not implemented yet")
    return det, dcfg, parquet_dir, run_ids, splits


@pytest.mark.parametrize("name", sorted(detectors._REGISTRY))
def test_detector_contract(name, parquet_runs, cfg, tmp_path):
    det, dcfg, parquet_dir, run_ids, splits = _train(name, parquet_runs, cfg)

    assert det.manifest["splits"] == splits
    assert isinstance(det.thresholds, dict)

    # score_run emits the shared record schema with sane ranges
    df = det.score_run(parquet_dir, run_ids[2], dcfg)
    assert not df.empty
    assert set(SCORE_COLUMNS) <= set(df.columns)
    assert df["score_raw"].between(0, 1).all()
    assert df["score_norm"].between(0, 1).all()
    assert (df["run_id"] == run_ids[2]).all()

    # strip_verdict (weak-signal honesty knob) must not crash, and for ppt it
    # must actually flow through to the features: the fixture run contains a
    # DROPPED flow, so zeroing the verdict blocks must change the scores
    # (flash's xgb trees may legitimately ignore the changed dims, so the
    # difference is asserted only for ppt)
    df_strip = det.score_run(parquet_dir, run_ids[2], dcfg, strip_verdict=True)
    if name == "ppt":
        assert not df_strip["p_true"].equals(df["p_true"])

    # score_events: perturbed in-memory frame, victim rows complete
    rd = load_run(parquet_dir, run_ids[2])
    pert, victim = dns_exfil(rd.events, np.random.default_rng(0), rd.entities)
    assert victim
    out = det.score_events(pert, rd.entities, dcfg, only_entity=victim)
    assert not out.empty
    assert set(SCORE_COLUMNS) <= set(out.columns)
    assert (out.entity_id == victim).any()

    # save/load round trip preserves scoring exactly
    detectors.save_detector(det, tmp_path / "art")
    det2 = detectors.get_detector(name).load(tmp_path / "art", dcfg)
    pd.testing.assert_frame_equal(
        df, det2.score_run(parquet_dir, run_ids[2], dcfg))


def test_flash_roundtrip_state(parquet_runs, cfg, tmp_path):
    det, dcfg, *_ = _train("flash", parquet_runs, cfg)
    detectors.save_detector(det, tmp_path / "art")
    det2 = detectors.get_detector("flash").load(tmp_path / "art", dcfg)
    for k, v in det.gnn.state_dict().items():
        assert torch.equal(v, det2.gnn.state_dict()[k]), k
    assert det.vocab.classes == det2.vocab.classes
    assert det.norm == det2.norm
    assert det.thresholds == det2.thresholds
    assert det.manifest == det2.manifest
