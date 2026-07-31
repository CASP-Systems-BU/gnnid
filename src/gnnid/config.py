"""Config loading: one YAML file drives everything; CLI --set overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access and dotted-path get/set."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def dotted_get(self, path: str, default: Any = None) -> Any:
        cur: Any = self
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def dotted_set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        cur: dict = self
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value


def _coerce(s: str) -> Any:
    """CLI override values arrive as strings; YAML-parse them so numbers,
    bools and lists round-trip ('0.5' -> 0.5, '[a,b]' -> list)."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(path) as f:
        cfg = Config(yaml.safe_load(f))
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        cfg.dotted_set(key.strip(), _coerce(val.strip()))
    return cfg


def deep_copy(cfg: Config) -> Config:
    return Config(copy.deepcopy(dict(cfg)))


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursive dict merge, overlay wins. Dicts merge key-by-key; scalars and
    lists REPLACE (a detector overriding a list must restate it whole)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def detector_view(cfg: Config) -> Config:
    """Per-detector merged config: `detectors.<cfg.detector>` overlaid on a
    deep copy of the base. The base sections ARE the flash defaults, so
    flash's overlay is empty and its view equals the base config."""
    view = deep_copy(cfg)
    name = cfg.dotted_get("detector", "flash")
    overlay = cfg.dotted_get(f"detectors.{name}") or {}
    _deep_merge(view, overlay)
    return view
