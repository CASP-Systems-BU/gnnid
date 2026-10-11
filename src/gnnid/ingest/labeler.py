"""Attack ground-truth labeling: fill Event.label / label_source from attack.json.

A captured attack run carries `<run>/attack/attack.json` written by the ubench
red-team orchestrator (scripts/cloudlab/redteam.sh): the attacker IDENTITY
(namespace + pod IP[s]) and the attack session window. This module stamps every
parsed event benign/malicious from that ground truth, filling the `label` /
`label_source` columns the schema already reserves (schema.Event, DATA.md).

Labeling is keyed on identity matched against resolved entities, not on raw IPs
in the event rows: an event is malicious when either of its endpoint entities is
the attacker pod. The network plane (Hubble L4/DNS) is the reliable signal — it
carries pod identity from eBPF, so it is unaffected by the pod->apiserver source
NAT that masquerades a cross-node API call to its node IP (which would otherwise
mis-attribute the attacker's audit calls to `ext:host`). A session-window guard
keeps any pre/post-window stragglers benign.

Pure post-processing over the parsed frames; benign runs (no attack.json) are
never touched, so their labels stay null (= "no ground truth").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import POD

BENIGN = "benign"
MALICIOUS = "malicious"


def load_attack(run_dir: str | Path) -> dict | None:
    """Return the attack ground truth for a run dir, or None if it is benign."""
    p = Path(run_dir) / "attack" / "attack.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def attacker_entity_ids(attack: dict, entities_df: pd.DataFrame) -> set[str]:
    """Resolve the attacker pod(s) to canonical entity IDs by IP or namespace."""
    ips = set(attack.get("attacker_ips") or [])
    namespaces = set(attack.get("attacker_namespaces") or [])
    if attack.get("attacker_namespace"):
        namespaces.add(attack["attacker_namespace"])
    ids: set[str] = set()
    if entities_df.empty:
        return ids
    for row in entities_df.itertuples(index=False):
        if getattr(row, "entity_type", None) != POD:
            continue
        if getattr(row, "pod_ip", None) in ips or getattr(row, "namespace", None) in namespaces:
            ids.add(row.entity_id)
    return ids


def apply_labels(events_df: pd.DataFrame, entities_df: pd.DataFrame,
                 attack: dict) -> dict:
    """Set `label` / `label_source` on events_df in place; return a summary dict.

    An event is malicious iff one of its endpoints is an attacker entity AND its
    timestamp is within the attack session window (when the window is known).
    """
    summary = {"total": int(len(events_df)), "malicious": 0, "benign": int(len(events_df)),
               "attacker_entities": 0, "attacker_entity_ids": []}
    if events_df.empty:
        return summary

    atk = attacker_entity_ids(attack, entities_df)
    summary["attacker_entities"] = len(atk)
    summary["attacker_entity_ids"] = sorted(atk)

    src_in = events_df["src_id"].isin(atk)
    dst_in = events_df["dst_id"].isin(atk)
    incident = src_in | dst_in

    in_window = pd.Series(True, index=events_df.index)
    w0, w1 = attack.get("session_start_epoch"), attack.get("session_end_epoch")
    if w0 is not None and w1 is not None:
        in_window = (events_df["ts"] >= float(w0)) & (events_df["ts"] <= float(w1))

    mal = incident & in_window
    events_df["label"] = np.where(mal, MALICIOUS, BENIGN)
    events_df["label_source"] = np.where(
        mal & src_in & dst_in, "attacker-both",
        np.where(mal & src_in, "attacker-src",
                 np.where(mal & dst_in, "attacker-dst", None)))

    n_mal = int(mal.sum())
    summary["malicious"] = n_mal
    summary["benign"] = int(len(events_df) - n_mal)
    return summary
