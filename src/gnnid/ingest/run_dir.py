"""Run-dir ingestion: the only module that knows the ubench results layout.

<run_dir>/
    meta.json                    run window + params
    k8s_snapshot/objects*.json   entity snapshot (resolver input)
    istio/access_logs/<pod>.log  Envoy RPC views (JSON or text)
    cilium/flows.jsonl           Hubble L4 + DNS flows
    audit/audit.jsonl            K8s API audit events (optional)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config
from ..resolve import EntityResolver
from ..schema import entities_to_frame, events_to_frame
from . import audit as audit_mod
from . import envoy, hubble


def load_meta(run_dir: Path) -> dict:
    with open(run_dir / "meta.json") as f:
        return json.load(f)


def ingest_run(run_dir: str | Path, cfg: Config,
               namespace: str = "default") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One run dir -> (events_df, entities_df, meta)."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    run_id = f"{meta['benchmark']}-{meta['request']}_{meta['run_id']}"
    resolver = EntityResolver.from_snapshot(run_id, run_dir / "k8s_snapshot")

    events = []

    # Envoy RPC views: parse every pod log, then merge x-request-id pairs
    # GLOBALLY (outbound and inbound views live in different pods' files).
    views = []
    log_dir = run_dir / "istio" / "access_logs"
    if log_dir.is_dir():
        for log_file in sorted(log_dir.glob("*.log")):
            pod_name = log_file.stem
            pod_eid = resolver.pod_by_name(namespace, pod_name)
            if pod_eid is None:
                # pod churned between snapshot and log collection
                from ..resolve import strip_pod_hash
                from ..schema import POD, Entity
                pod_eid = f"pod:{namespace}/{pod_name}"
                resolver._add(Entity(run_id, pod_eid, POD, name=pod_name,
                                     namespace=namespace,
                                     canonical_service=strip_pod_hash(pod_name)))
            views.extend(envoy.parse_pod_log(
                log_file.read_text(errors="replace"), pod_eid,
                f"istio/access_logs/{log_file.name}", run_id, resolver))
        events.extend(envoy.merge_views(run_id, views, resolver))

    flows_path = run_dir / "cilium" / "flows.jsonl"
    if flows_path.exists():
        events.extend(hubble.parse_flows(
            flows_path.read_text(errors="replace"), run_id, resolver,
            keep_rules=list(cfg.dotted_get("events.l4_keep", ["non_forwarded"]))))

    audit_path = run_dir / "audit" / "audit.jsonl"
    if audit_path.exists():
        events.extend(audit_mod.parse_audit(
            audit_path.read_text(errors="replace"), run_id, resolver,
            drop_control_plane=bool(
                cfg.dotted_get("events.drop_control_plane_audit", True))))

    events_df = events_to_frame(events)
    entities_df = entities_to_frame(list(resolver.entities.values()))
    return events_df, entities_df, meta


def ingest_all(cfg: Config, repo_root: str | Path = ".") -> list[str]:
    """Ingest every matching run dir -> data/parquet/<run_id>/{events,entities}.parquet.
    Returns the ingested run_ids (sorted by run timestamp = temporal order)."""
    repo_root = Path(repo_root)
    out_root = repo_root / cfg.data.parquet_dir
    benchmarks = set(cfg.data.benchmarks or [])
    run_ids = []
    for run_dir in sorted(repo_root.glob(cfg.data.runs_glob)):
        if not (run_dir / "meta.json").exists():
            continue
        meta = load_meta(run_dir)
        if benchmarks and meta.get("benchmark") not in benchmarks:
            continue
        if meta.get("run_status", 1) != 0:
            print(f"[ingest] skip {run_dir.name}: run_status != 0")
            continue
        events_df, entities_df, meta = ingest_run(run_dir, cfg)
        run_id = f"{meta['benchmark']}-{meta['request']}_{meta['run_id']}"
        out = out_root / run_id
        out.mkdir(parents=True, exist_ok=True)
        events_df.to_parquet(out / "events.parquet", index=False)
        entities_df.to_parquet(out / "entities.parquet", index=False)
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        run_ids.append(run_id)
        print(f"[ingest] {run_id}: {len(events_df)} events, "
              f"{len(entities_df)} entities")
    return sorted(run_ids, key=lambda r: r.rsplit("_", 1)[-1])
