"""Detector registry: `detector: <name>` in the config selects the class."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from .base import SCORE_COLUMNS, Detector  # noqa: F401  (re-export)

_REGISTRY: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    _REGISTRY[cls.name] = cls
    return cls


def get_detector(name: str) -> type[Detector]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def save_detector(det: Detector, path: str | Path) -> None:
    path = Path(path)
    det.save(path)
    with open(path / "detector.json", "w") as f:
        json.dump({"detector": det.name}, f)


def load_detector(cfg: Config, repo_root: str | Path = ".") -> Detector:
    """Load the trained detector from cfg.artifacts_dir (merged view)."""
    path = Path(repo_root) / str(cfg.dotted_get("artifacts_dir",
                                                "artifacts/default"))
    tag = "flash"  # artifact dirs from before the seam carry no tag
    if (path / "detector.json").exists():
        with open(path / "detector.json") as f:
            tag = json.load(f)["detector"]
    want = cfg.dotted_get("detector", "flash")
    if tag != want:
        raise SystemExit(
            f"artifacts at {path} were trained by detector {tag!r} but the "
            f"config selects {want!r}; retrain or point artifacts_dir at the "
            f"right bundle")
    return get_detector(tag).load(path, cfg)


from . import flash  # noqa: E402,F401  (imports register the detectors)
