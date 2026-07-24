"""Event / Entity schemas — the contract between ingest and everything else.

Events live in ONE wide nullable table (per-type payload columns are null for
other types). Entities are one row per canonical entity per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd

# Event types (the telemetry planes)
RPC_CALL = "RPC_CALL"        # Envoy access logs (L7 HTTP/gRPC)
L4_FLOW = "L4_FLOW"          # Hubble flows (kept by the selection rule)
DNS_QUERY = "DNS_QUERY"      # Hubble DNS
K8S_API_CALL = "K8S_API_CALL"  # kube-apiserver audit events
EVENT_TYPES = (RPC_CALL, L4_FLOW, DNS_QUERY, K8S_API_CALL)

# Entity types (graph node types)
POD = "Pod"
SERVICE = "Service"
WORKLOAD = "Workload"
DNSNAME = "DNSName"
EXTERNAL = "ExternalEndpoint"
KUBEAPI = "KubeAPI"
ENTITY_TYPES = (POD, SERVICE, WORKLOAD, DNSNAME, EXTERNAL, KUBEAPI)


@dataclass
class Event:
    """One telemetry event. src/dst are canonical entity IDs from resolve.py."""
    event_id: str            # deterministic: "<run_id>:<file>:<lineno>"
    run_id: str
    ts: float                # epoch seconds
    event_type: str          # one of EVENT_TYPES
    src_id: str
    src_type: str
    dst_id: str
    dst_type: str
    # RPC only: the Service in front of dst (graph adds a src->Service edge too;
    # dst_id is the most specific endpoint known — server pod if resolved).
    dst_svc_id: str | None = None
    # --- RPC_CALL ---
    protocol: str | None = None       # http | grpc
    method: str | None = None
    path: str | None = None           # raw path (templating happens in sentences)
    status_code: int | None = None
    grpc_status: int | None = None
    response_flags: str | None = None  # Envoy flags, '-' if none
    duration_ms: float | None = None
    request_bytes: int | None = None
    response_bytes: int | None = None
    reporter_views: int | None = None  # 1|2 after x-request-id outbound/inbound merge
    # --- L4_FLOW ---
    l4_proto: str | None = None       # tcp | udp | icmp
    dst_port: int | None = None
    verdict: str | None = None        # FORWARDED | DROPPED | ...
    drop_reason: str | None = None
    is_reply: bool | None = None
    # --- DNS_QUERY ---
    dns_query: str | None = None      # normalized fqdn (lowercase, no trailing dot)
    dns_qtypes: str | None = None     # comma-joined
    dns_rcode: str | None = None
    # --- K8S_API_CALL ---
    k8s_verb: str | None = None
    k8s_resource: str | None = None
    k8s_subresource: str | None = None
    k8s_user_type: str | None = None  # sa | node | user
    k8s_status_code: int | None = None
    # --- labels (attack ground truth, filled later by `ingest --labels`) ---
    label: str | None = None
    label_source: str | None = None


@dataclass
class Entity:
    """One canonical entity per run (graph node candidate)."""
    run_id: str
    entity_id: str           # e.g. "pod:default/frontend-abc123-xyz"
    entity_type: str         # one of ENTITY_TYPES
    name: str | None = None
    namespace: str | None = None
    canonical_service: str | None = None  # role label source for Pod/Service
    workload: str | None = None
    node_name: str | None = None
    pod_ip: str | None = None
    service_account: str | None = None
    uid: str | None = None
    has_sidecar: bool | None = None
    extra: dict = field(default_factory=dict)


EVENT_COLUMNS = [f.name for f in fields(Event)]
ENTITY_COLUMNS = [f.name for f in fields(Entity) if f.name != "extra"]


def events_to_frame(events: list[Event]) -> pd.DataFrame:
    df = pd.DataFrame([vars(e) for e in events], columns=EVENT_COLUMNS)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def entities_to_frame(entities: list[Entity]) -> pd.DataFrame:
    rows = []
    for e in entities:
        d = {k: v for k, v in vars(e).items() if k != "extra"}
        rows.append(d)
    return pd.DataFrame(rows, columns=ENTITY_COLUMNS)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
