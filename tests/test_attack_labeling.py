"""Attack ground-truth labeling + labeled detection eval (the supervised seam)."""
import json
from pathlib import Path

import pandas as pd

from gnnid import dataset
from gnnid.eval import labeled_detection
from gnnid.ingest import labeler
from gnnid.ingest.run_dir import ingest_run
from gnnid.labels import OTHER
from gnnid.schema import (
    K8S_API_CALL, L4_FLOW, POD, SERVICE, Entity, Event,
    entities_to_frame, events_to_frame,
)


# --- unit: attacker identity resolution --------------------------------------
def _entities():
    return entities_to_frame([
        Entity("r", "pod:redteam/attacker", POD, name="attacker",
               namespace="redteam", pod_ip="10.244.9.9", canonical_service="attacker"),
        Entity("r", "pod:default/frontend-abc", POD, name="frontend-abc",
               namespace="default", pod_ip="10.244.1.10", canonical_service="frontend"),
        Entity("r", "pod:default/cart-xyz", POD, name="cart-xyz",
               namespace="default", pod_ip="10.244.2.20", canonical_service="cart"),
        Entity("r", "svc:default/frontend", SERVICE, name="frontend", namespace="default"),
    ])


def test_attacker_entity_ids_by_ip():
    atk = labeler.attacker_entity_ids({"attacker_ips": ["10.244.9.9"]}, _entities())
    assert atk == {"pod:redteam/attacker"}


def test_attacker_entity_ids_by_namespace():
    atk = labeler.attacker_entity_ids({"attacker_namespace": "redteam"}, _entities())
    assert atk == {"pod:redteam/attacker"}


def test_apply_labels_incidence_and_window():
    ents = _entities()
    events = events_to_frame([
        # attacker -> frontend inside window: malicious
        Event("r:f:1", "r", 100.0, L4_FLOW, "pod:redteam/attacker", POD,
              "pod:default/frontend-abc", POD),
        # frontend -> cart inside window: benign (no attacker endpoint)
        Event("r:f:2", "r", 100.1, L4_FLOW, "pod:default/frontend-abc", POD,
              "pod:default/cart-xyz", POD),
        # attacker API call OUTSIDE the window: benign (window guard)
        Event("r:a:3", "r", 999.0, K8S_API_CALL, "pod:redteam/attacker", POD,
              "kubeapi:cluster", "KubeAPI"),
    ])
    summary = labeler.apply_labels(events, ents, {
        "attacker_ips": ["10.244.9.9"],
        "session_start_epoch": 90, "session_end_epoch": 110})
    labels = dict(zip(events["event_id"], events["label"]))
    assert labels["r:f:1"] == "malicious"
    assert labels["r:f:2"] == "benign"
    assert labels["r:a:3"] == "benign"          # outside the session window
    assert summary["malicious"] == 1
    assert summary["attacker_entities"] == 1
    srcs = dict(zip(events["event_id"], events["label_source"]))
    assert srcs["r:f:1"] == "attacker-src"


# --- integration: ingest marks + labels an attack run ------------------------
def _write_attack(run_dir: Path, **attack):
    (run_dir / "attack").mkdir(exist_ok=True)
    (run_dir / "attack" / "attack.json").write_text(json.dumps(attack))


def test_ingest_run_sets_is_attack_and_labels(run_dir, cfg):
    rd = run_dir
    # Treat the cart pod's IP as the attacker foothold (from conftest snapshot).
    _write_attack(rd, is_attack=True, attacker_ips=["10.244.2.20"],
                  attacker_namespace="default",
                  session_start_epoch=1784702584, session_end_epoch=1784702614)

    ev, _, meta = ingest_run(rd, cfg, apply_labels=True)
    assert meta["is_attack"] is True
    assert meta["labels_summary"]["malicious"] > 0
    assert set(ev["label"].dropna().unique()) <= {"malicious", "benign"}
    assert (ev["label"] == "malicious").any()

    # Without --labels: still flagged is_attack, but labels stay null.
    ev2, _, meta2 = ingest_run(rd, cfg, apply_labels=False)
    assert meta2["is_attack"] is True
    assert "labels_summary" not in meta2
    assert ev2["label"].isna().all()


# --- integration: attack runs are held out of benign train/val ---------------
def test_split_runs_forces_attack_into_test(parquet_runs, cfg):
    parquet_dir, run_ids = parquet_runs
    # Mark the latest ingested run as an attack run (what ingest does when
    # attack/attack.json is present).
    attack_id = run_ids[-1]
    meta_path = parquet_dir / attack_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["is_attack"] = True
    meta_path.write_text(json.dumps(meta))

    assert dataset.is_attack_run(parquet_dir, attack_id)
    assert not dataset.is_attack_run(parquet_dir, run_ids[0])
    tr, va, te = dataset.split_runs(parquet_dir, run_ids, cfg)
    assert attack_id in te
    assert attack_id not in tr and attack_id not in va


# --- integration: labeled_detection separates attacker-touched known roles ---
class _StubDetector:
    """Minimal Detector surface for labeled_detection: fixed per-node scores."""
    thresholds = {"Pod": 0.5}

    def __init__(self, scores):
        self._scores = scores  # entity_id -> (true_label, score_norm)

    def score_run(self, parquet_dir, run_id, cfg, strip_verdict=None):
        from gnnid.detectors.base import SCORE_COLUMNS
        rows = []
        for eid, (true_label, s) in self._scores.items():
            rows.append((run_id, 0, eid, "Pod", true_label, true_label,
                         1.0 - s, s, s))
        return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def test_labeled_detection_separates_touched_known_role(tmp_path, cfg):
    parquet_dir = tmp_path / "parquet"
    run_id = "boutique-mix_20260722-064900"
    out = parquet_dir / run_id
    out.mkdir(parents=True)

    ents = entities_to_frame([
        Entity(run_id, "pod:redteam/attacker", POD, name="attacker",
               namespace="redteam", pod_ip="10.244.9.9", canonical_service="attacker"),
        Entity(run_id, "pod:default/frontend-abc", POD, name="frontend-abc",
               namespace="default", pod_ip="10.244.1.10", canonical_service="frontend"),
        Entity(run_id, "pod:default/cart-xyz", POD, name="cart-xyz",
               namespace="default", pod_ip="10.244.2.20", canonical_service="cart"),
    ])
    events = events_to_frame([
        Event(f"{run_id}:f:1", run_id, 100.0, L4_FLOW, "pod:redteam/attacker", POD,
              "pod:default/frontend-abc", POD),         # attacker -> frontend (malicious)
        Event(f"{run_id}:f:2", run_id, 100.1, L4_FLOW, "pod:default/frontend-abc", POD,
              "pod:default/cart-xyz", POD),             # benign backend call
    ])
    labeler.apply_labels(events, ents, {"attacker_ips": ["10.244.9.9"]})
    events.to_parquet(out / "events.parquet", index=False)
    ents.to_parquet(out / "entities.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"is_attack": True}))

    # frontend (attacker-touched known role) scores high; cart (benign) low;
    # attacker is <other> and scores 1.0 by construction.
    det = _StubDetector({
        "pod:redteam/attacker": (OTHER, 1.0),
        "pod:default/frontend-abc": ("frontend", 0.9),
        "pod:default/cart-xyz": ("cart", 0.1),
    })
    rep = labeled_detection(det, parquet_dir, [run_id], cfg)

    kr = rep["known_role_nodes"]
    assert kr["n_pos"] == 1 and kr["n_neg"] == 1       # frontend pos, cart neg
    assert kr["roc_auc"] == 1.0
    assert kr["recall"] == 1.0 and kr["precision"] == 1.0
    assert rep["attacker_nodes"]["n_windows"] == 1     # the <other> attacker pod
    assert rep["attacker_nodes"]["alert_recall"] == 1.0
