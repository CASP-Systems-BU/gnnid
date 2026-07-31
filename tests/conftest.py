"""Shared fixtures: a tiny hand-crafted run dir mirroring real ubench formats
(pinned against actual smoke-run lines). Small + deterministic for unit tests.
"""
import json
from pathlib import Path

import pytest

from gnnid.config import load_config

# --- k8s snapshot: 2 app pods (frontend, cart) + client + 2 services ---------
_OBJECTS = {
    "items": [
        {"kind": "Pod", "metadata": {
            "name": "frontend-5976767489-mljz4", "namespace": "default",
            "uid": "u-frontend", "ownerReferences": [{"kind": "ReplicaSet",
                "name": "frontend-5976767489"}]},
         "spec": {"nodeName": "node-1", "serviceAccountName": "default",
                  "containers": [{"name": "server"}, {"name": "istio-proxy"}]},
         "status": {"podIP": "10.244.1.10"}},
        {"kind": "Pod", "metadata": {
            "name": "cart-649c7444bc-2zqqw", "namespace": "default",
            "uid": "u-cart", "ownerReferences": [{"kind": "ReplicaSet",
                "name": "cart-649c7444bc"}]},
         "spec": {"nodeName": "node-2", "serviceAccountName": "default",
                  "containers": [{"name": "server"}, {"name": "istio-proxy"}]},
         "status": {"podIP": "10.244.2.20"}},
        {"kind": "Pod", "metadata": {
            "name": "ubuntu-client", "namespace": "default", "uid": "u-client"},
         "spec": {"nodeName": "node-1", "containers": [{"name": "client"}]},
         "status": {"podIP": "10.244.4.221"}},
        {"kind": "ReplicaSet", "metadata": {
            "name": "frontend-5976767489", "namespace": "default",
            "ownerReferences": [{"kind": "Deployment", "name": "frontend"}]}},
        {"kind": "ReplicaSet", "metadata": {
            "name": "cart-649c7444bc", "namespace": "default",
            "ownerReferences": [{"kind": "Deployment", "name": "cart"}]}},
        {"kind": "Service", "metadata": {"name": "frontend", "namespace": "default"},
         "spec": {"clusterIP": "10.96.0.10"}},
        {"kind": "Service", "metadata": {"name": "cart", "namespace": "default"},
         "spec": {"clusterIP": "10.96.0.20"}},
    ]
}
_NODES = {"items": [
    {"metadata": {"name": "node-1"},
     "status": {"addresses": [{"type": "InternalIP", "address": "10.0.0.102"}]}},
    {"metadata": {"name": "node-2"},
     "status": {"addresses": [{"type": "InternalIP", "address": "10.0.0.103"}]}},
]}

# --- Envoy access logs (JSON). client->frontend appears ONLY inbound at
#     frontend (client is sidecar-less); frontend->cart appears as an
#     outbound view at frontend AND an inbound view at cart (same request_id) --
_FRONTEND_LOG = [
    # inbound from the sidecar-less client (downstream_remote_address = client IP)
    {"start_time": "2026-07-22T06:43:04.100Z", "method": "GET", "path": "/cart",
     "protocol": "HTTP/1.1", "response_code": 200, "response_flags": "-",
     "bytes_received": 0, "bytes_sent": 500, "duration": 5, "user_agent": "wrk",
     "request_id": "req-A", "upstream_cluster": "inbound|8080||",
     "downstream_remote_address": "10.244.4.221:33848",
     "upstream_host": "10.244.1.10:8080"},
    # outbound frontend->cart (gRPC)
    {"start_time": "2026-07-22T06:43:04.120Z", "method": "POST",
     "path": "/hipstershop.CartService/GetCart", "protocol": "HTTP/2",
     "response_code": 200, "response_flags": "-", "bytes_received": 10,
     "bytes_sent": 40, "duration": 3, "user_agent": "grpc-go/1.0",
     "request_id": "req-B",
     "upstream_cluster": "outbound|7070||cart.default.svc.cluster.local",
     "downstream_remote_address": "10.244.1.10:40000",
     "upstream_host": "10.244.2.20:7070"},
]
_CART_LOG = [
    # inbound view of frontend->cart (same request_id req-B)
    {"start_time": "2026-07-22T06:43:04.121Z", "method": "POST",
     "path": "/hipstershop.CartService/GetCart", "protocol": "HTTP/2",
     "response_code": 200, "response_flags": "-", "bytes_received": 10,
     "bytes_sent": 40, "duration": 2, "user_agent": "grpc-go/1.0",
     "request_id": "req-B", "upstream_cluster": "inbound|7070||",
     "downstream_remote_address": "10.244.1.10:40000",
     "upstream_host": "10.244.2.20:7070"},
]

# --- Hubble flows: a kept SYN (no ACK), a DROPPED policy flow, a DNS pair -----
_FLOWS = [
    {"flow": {"time": "2026-07-22T06:43:04.10Z", "verdict": "FORWARDED",
              "l4": {"TCP": {"destination_port": 7070, "flags": {"SYN": True}}},
              "source": {"pod_name": "frontend-5976767489-mljz4",
                         "namespace": "default"},
              "destination": {"pod_name": "cart-649c7444bc-2zqqw",
                              "namespace": "default"}, "is_reply": False}},
    {"flow": {"time": "2026-07-22T06:43:04.30Z", "verdict": "DROPPED",
              "drop_reason_desc": "POLICY_DENIED",
              "l4": {"TCP": {"destination_port": 9999, "flags": {"SYN": True}}},
              "source": {"pod_name": "cart-649c7444bc-2zqqw", "namespace": "default"},
              "destination": {"labels": ["reserved:world"]}, "is_reply": False}},
    # a mid-stream ACK that the selection rule must DROP
    {"flow": {"time": "2026-07-22T06:43:04.40Z", "verdict": "FORWARDED",
              "l4": {"TCP": {"destination_port": 7070,
                             "flags": {"ACK": True}}},
              "source": {"pod_name": "frontend-5976767489-mljz4", "namespace": "default"},
              "destination": {"pod_name": "cart-649c7444bc-2zqqw", "namespace": "default"},
              "is_reply": False}},
    # DNS response (querier is the response's destination endpoint)
    {"flow": {"time": "2026-07-22T06:43:04.05Z", "verdict": "FORWARDED",
              "l7": {"type": "RESPONSE", "dns": {"query": "cart.default.svc.cluster.local.",
                                                 "qtypes": ["A"], "rcode": 0}},
              "source": {"labels": ["reserved:world"]},
              "destination": {"pod_name": "frontend-5976767489-mljz4",
                              "namespace": "default"}}},
]

# --- audit: cart pod's SA lists configmaps (from its pod IP) ------------------
_AUDIT = [
    {"kind": "Event", "level": "Metadata", "verb": "list",
     "requestReceivedTimestamp": "2026-07-22T06:43:04.200Z",
     "user": {"username": "system:serviceaccount:default:default"},
     "sourceIPs": ["10.244.2.20"],
     "objectRef": {"resource": "configmaps", "namespace": "default"},
     "responseStatus": {"code": 200}},
    # a control-plane event that must be dropped (host source IP)
    {"kind": "Event", "level": "Metadata", "verb": "get",
     "requestReceivedTimestamp": "2026-07-22T06:43:04.210Z",
     "user": {"username": "system:node:node-1"}, "sourceIPs": ["10.0.0.102"],
     "objectRef": {"resource": "nodes"}, "responseStatus": {"code": 200}},
]


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def make_run_dir(base: Path, run_ts: str = "20260722-064121") -> Path:
    d = base / f"boutique-mix_{run_ts}"
    (d / "istio" / "access_logs").mkdir(parents=True)
    (d / "cilium").mkdir(parents=True)
    (d / "audit").mkdir(parents=True)
    (d / "k8s_snapshot").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "benchmark": "boutique", "request": "mix", "threads": 4,
        "connections": 16, "duration_s": 30, "run_id": run_ts,
        "start_epoch": 1784702584, "end_epoch": 1784702614,
        "start_iso": "2026-07-22T06:43:04Z", "istio_enabled": True,
        "cilium_enabled": True, "audit_enabled": True, "run_status": 0}))
    (d / "k8s_snapshot" / "objects.json").write_text(json.dumps(_OBJECTS))
    (d / "k8s_snapshot" / "nodes.json").write_text(json.dumps(_NODES))
    _write_jsonl(d / "istio" / "access_logs" / "frontend-5976767489-mljz4.log",
                 _FRONTEND_LOG)
    _write_jsonl(d / "istio" / "access_logs" / "cart-649c7444bc-2zqqw.log", _CART_LOG)
    _write_jsonl(d / "cilium" / "flows.jsonl", _FLOWS)
    _write_jsonl(d / "audit" / "audit.jsonl", _AUDIT)
    return d


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return make_run_dir(tmp_path)


@pytest.fixture
def parquet_runs(tmp_path, cfg) -> tuple[Path, list[str]]:
    """Three synthetic runs with distinct timestamps, ingested to parquet —
    temporal_split yields 1 train / 1 val / 1 test."""
    from gnnid.ingest.run_dir import ingest_run
    parquet_dir = tmp_path / "parquet"
    run_ids = []
    for ts in ("20260722-064121", "20260722-064500", "20260722-064900"):
        rd = make_run_dir(tmp_path, ts)
        events_df, entities_df, meta = ingest_run(rd, cfg)
        run_id = f"{meta['benchmark']}-{meta['request']}_{meta['run_id']}"
        out = parquet_dir / run_id
        out.mkdir(parents=True)
        events_df.to_parquet(out / "events.parquet", index=False)
        entities_df.to_parquet(out / "entities.parquet", index=False)
        (out / "meta.json").write_text(json.dumps(meta))
        run_ids.append(run_id)
    return parquet_dir, run_ids


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")
