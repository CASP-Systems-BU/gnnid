"""Role/type label vocabulary — the pretext-task targets.

Pods and Services are labeled by canonical service (frontend, cart, ...);
DNSName/ExternalEndpoint/KubeAPI by entity type. Vocab is fit on benign TRAIN
runs; entities mapping to no vocab entry get the reserved OTHER class (excluded
from training loss, scored 1.0 at inference — an unknown workload is itself an
alert).

No leakage risk here: the label is a *target*, never a feature. sentences.py
guarantees the label's source strings never enter the node's own sentence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import DNSNAME, EXTERNAL, KUBEAPI, POD, SERVICE, WORKLOAD

OTHER = "<other>"
_TYPE_LABEL = {DNSNAME: "dnsname", EXTERNAL: "external", KUBEAPI: "kubeapi",
               WORKLOAD: "workload"}


def entity_label(entity_type: str, canonical_service: str | None,
                 namespace: str | None, app_namespace: str) -> str | None:
    """Raw label for an entity (pre-vocab). Only benchmark-namespace pods and
    services get a role; everything else is typed. None => not labelable."""
    if entity_type in (POD, SERVICE):
        if namespace == app_namespace and canonical_service:
            return canonical_service
        return None  # cross-namespace infra pod/svc -> OTHER after vocab fit
    return _TYPE_LABEL.get(entity_type)


class LabelVocab:
    def __init__(self, classes: list[str], app_namespace: str = "default"):
        # OTHER is always index 0 (excluded from training loss)
        self.classes = [OTHER] + [c for c in classes if c != OTHER]
        self.app_namespace = app_namespace
        self.to_idx = {c: i for i, c in enumerate(self.classes)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def other_idx(self) -> int:
        return 0

    def index_of(self, entity_type: str, canonical_service: str | None,
                 namespace: str | None) -> int:
        raw = entity_label(entity_type, canonical_service, namespace,
                           self.app_namespace)
        return self.to_idx.get(raw, self.other_idx) if raw else self.other_idx

    @classmethod
    def fit(cls, entities: pd.DataFrame, app_namespace: str = "default") -> "LabelVocab":
        seen: set[str] = set()
        for row in entities.itertuples(index=False):
            raw = entity_label(row.entity_type, row.canonical_service,
                               row.namespace, app_namespace)
            if raw:
                seen.add(raw)
        return cls(sorted(seen), app_namespace)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"classes": self.classes,
                       "app_namespace": self.app_namespace}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LabelVocab":
        with open(path) as f:
            d = json.load(f)
        v = cls([], d["app_namespace"])
        v.classes = d["classes"]
        v.to_idx = {c: i for i, c in enumerate(v.classes)}
        return v
