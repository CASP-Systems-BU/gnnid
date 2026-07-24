"""K8s audit logs (audit/audit.jsonl) -> K8S_API_CALL events.

src = the calling pod, resolved from sourceIPs[0]; dst = the KubeAPI
singleton. Events whose source cannot be attributed to a pod (kubelet,
control-plane components on host IPs) are dropped when
events.drop_control_plane_audit is true — they are cluster noise the
per-pod detector cannot act on.
"""
from __future__ import annotations

import datetime
import json

from ..resolve import KUBEAPI_ID, EntityResolver
from ..schema import K8S_API_CALL, KUBEAPI, POD, Event


def _iso_epoch(ts: str | None) -> float | None:
    if not isinstance(ts, str):
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def user_type(username: str | None) -> str:
    if not username:
        return "user"
    if username.startswith("system:serviceaccount:"):
        return "sa"
    if username.startswith("system:node:"):
        return "node"
    return "user"


def parse_audit(text: str, run_id: str, resolver: EntityResolver,
                drop_control_plane: bool = True,
                source_file: str = "audit/audit.jsonl") -> list[Event]:
    events: list[Event] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _iso_epoch(ev.get("requestReceivedTimestamp") or ev.get("stageTimestamp"))
        if ts is None:
            continue
        src_ip = (ev.get("sourceIPs") or [None])[0]
        src = resolver.resolve_ip(src_ip)
        if src is None or (drop_control_plane and src[1] != POD):
            continue
        ref = ev.get("objectRef") or {}
        status = (ev.get("responseStatus") or {}).get("code")
        events.append(Event(
            event_id=f"{run_id}:{source_file}:{lineno}", run_id=run_id, ts=ts,
            event_type=K8S_API_CALL,
            src_id=src[0], src_type=src[1], dst_id=KUBEAPI_ID, dst_type=KUBEAPI,
            k8s_verb=(ev.get("verb") or "").lower() or None,
            k8s_resource=ref.get("resource"),
            k8s_subresource=ref.get("subresource"),
            k8s_user_type=user_type((ev.get("user") or {}).get("username")),
            k8s_status_code=int(status) if status is not None else None))
    return events
