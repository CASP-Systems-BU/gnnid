"""Detector registry: dispatch, artifact tagging, back-compat, mismatch."""
import json

import pytest

from gnnid.config import Config
from gnnid.detectors import (_REGISTRY, Detector, get_detector, load_detector,
                             save_detector)


class _Dummy(Detector):
    name = "dummy"

    def __init__(self):
        self.manifest, self.thresholds = {}, {}

    @classmethod
    def train(cls, cfg, parquet_dir, splits):
        return cls()

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "dummy.txt").write_text("x")

    @classmethod
    def load(cls, path, cfg):
        assert (path / "dummy.txt").exists()
        return cls()

    def score_run(self, parquet_dir, run_id, cfg, strip_verdict=None):
        raise NotImplementedError

    def score_events(self, events, entities, cfg, only_entity=None):
        raise NotImplementedError


@pytest.fixture
def dummy_registered(monkeypatch):
    monkeypatch.setitem(_REGISTRY, "dummy", _Dummy)
    return _Dummy


def test_get_detector_dispatch():
    assert get_detector("flash").name == "flash"


def test_get_detector_unknown_lists_available():
    with pytest.raises(KeyError, match="unknown detector 'nope'.*flash"):
        get_detector("nope")


def test_save_writes_tag_and_load_dispatches(dummy_registered, tmp_path):
    det = _Dummy()
    save_detector(det, tmp_path / "art")
    tag = json.loads((tmp_path / "art" / "detector.json").read_text())
    assert tag == {"detector": "dummy"}
    cfg = Config({"detector": "dummy", "artifacts_dir": str(tmp_path / "art")})
    assert isinstance(load_detector(cfg), _Dummy)


def test_load_mismatch_exits(dummy_registered, tmp_path):
    save_detector(_Dummy(), tmp_path / "art")
    cfg = Config({"detector": "flash", "artifacts_dir": str(tmp_path / "art")})
    with pytest.raises(SystemExit, match="trained by detector 'dummy'"):
        load_detector(cfg)


def test_load_missing_tag_defaults_to_flash(dummy_registered, tmp_path,
                                            monkeypatch):
    # pre-seam artifact dirs have no detector.json -> treated as flash
    (tmp_path / "art").mkdir()
    seen = {}

    @classmethod
    def fake_load(cls, path, cfg):
        seen["cls"] = cls
        return object.__new__(cls)

    flash_cls = get_detector("flash")
    monkeypatch.setattr(flash_cls, "load", fake_load)
    cfg = Config({"artifacts_dir": str(tmp_path / "art")})
    load_detector(cfg)
    assert seen["cls"] is flash_cls
