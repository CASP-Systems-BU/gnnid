"""Envoy access logs -> RPC_CALL events.

Input: istio/access_logs/<pod>.log — one line per request, JSON encoding
(accessLogEncoding=JSON) primary, Istio default TEXT format fallback.
The reporting pod is the log filename. Mesh-internal RPCs appear twice
(client's outbound view + server's inbound view) and are merged on
x-request-id with the server view authoritative (reporter_views=2).
"""
from __future__ import annotations

import datetime
import json
import re

from ..resolve import (EntityResolver, parse_upstream_cluster,
                       service_from_cluster_fqdn)
from ..schema import EXTERNAL, POD, RPC_CALL, Event

# Istio default TEXT format (fallback for pre-JSON runs):
# [START_TIME] "METHOD PATH PROTOCOL" CODE FLAGS DETAILS TERM "UP_FAIL"
# BYTES_RECV BYTES_SENT DURATION UP_SVC_TIME "XFF" "UA" "REQ_ID" "AUTHORITY"
# "UPSTREAM_HOST" UPSTREAM_CLUSTER UP_LOCAL DOWN_LOCAL DOWN_REMOTE SNI ROUTE
_TEXT_RE = re.compile(
    r'^\[(?P<start_time>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" '
    r'(?P<response_code>\d+) (?P<response_flags>\S+) \S+ \S+ "[^"]*" '
    r'(?P<bytes_received>\d+) (?P<bytes_sent>\d+) (?P<duration>\d+) \S+ '
    r'"[^"]*" "(?P<user_agent>[^"]*)" "(?P<request_id>[^"]*)" "[^"]*" '
    r'"(?P<upstream_host>[^"]*)" (?P<upstream_cluster>\S+) \S+ \S+ '
    r'(?P<downstream_remote_address>\S+)')


def _iso_epoch(ts: str | None) -> float | None:
    if not isinstance(ts, str):
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def parse_line(line: str) -> dict | None:
    """One access-log line -> normalized record dict, or None."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return None
        if "start_time" not in rec or "method" not in rec:
            return None
        return rec
    m = _TEXT_RE.match(line)
    if not m:
        return None
    rec = m.groupdict()
    for k in ("response_code", "bytes_received", "bytes_sent", "duration"):
        rec[k] = int(rec[k])
    return rec


def _protocol(rec: dict) -> str:
    ua = (rec.get("user_agent") or "") or ""
    if ua.startswith("grpc"):
        return "grpc"
    path = rec.get("path") or ""
    seg = path.lstrip("/").split("/", 1)[0]
    if (rec.get("protocol") or "").startswith("HTTP/2") and "." in seg:
        return "grpc"  # /package.Service/Method
    return "http"


def _split_hostport(addr: str | None) -> str | None:
    if not addr:
        return None
    return addr.rsplit(":", 1)[0] if ":" in addr else addr


def parse_pod_log(text: str, reporting_pod_eid: str, source_file: str,
                  run_id: str, resolver: EntityResolver) -> list[dict]:
    """All lines of one pod's access log -> list of view dicts (pre-merge)."""
    views = []
    for lineno, line in enumerate(text.splitlines(), 1):
        rec = parse_line(line)
        if rec is None:
            continue
        ts = _iso_epoch(rec.get("start_time"))
        if ts is None:
            continue
        direction, svc_fqdn = parse_upstream_cluster(rec.get("upstream_cluster") or "")
        views.append({
            "rec": rec, "ts": ts, "direction": direction, "svc_fqdn": svc_fqdn,
            "reporting_pod": reporting_pod_eid,
            "event_id": f"{run_id}:{source_file}:{lineno}",
            "request_id": rec.get("request_id") or None,
        })
    return views


def _svc_entity(resolver: EntityResolver, svc_fqdn: str | None) -> str | None:
    parsed = service_from_cluster_fqdn(svc_fqdn)
    if parsed:
        ns, name = parsed
        return resolver.service_entity(ns, name)
    return None


def _mk_event(run_id: str, v: dict, src: tuple[str, str], dst: tuple[str, str],
              dst_svc_id: str | None, reporter_views: int) -> Event:
    rec = v["rec"]
    return Event(
        event_id=v["event_id"], run_id=run_id, ts=v["ts"], event_type=RPC_CALL,
        src_id=src[0], src_type=src[1], dst_id=dst[0], dst_type=dst[1],
        dst_svc_id=dst_svc_id,
        protocol=_protocol(rec), method=rec.get("method"), path=rec.get("path"),
        status_code=int(rec.get("response_code") or 0),
        grpc_status=(int(rec["grpc_status"]) if rec.get("grpc_status")
                     not in (None, "", "-") else None),
        response_flags=rec.get("response_flags") or "-",
        duration_ms=float(rec.get("duration") or 0),
        request_bytes=int(rec.get("bytes_received") or 0),
        response_bytes=int(rec.get("bytes_sent") or 0),
        reporter_views=reporter_views)


def merge_views(run_id: str, views: list[dict],
                resolver: EntityResolver) -> list[Event]:
    """Merge outbound/inbound view pairs on x-request-id (server view
    authoritative), turn unmatched views into single-view events."""
    outbound: dict[str, dict] = {}
    inbound: dict[str, dict] = {}
    unmatched: list[dict] = []
    for v in views:
        rid = v["request_id"]
        book = outbound if v["direction"] == "outbound" else \
            inbound if v["direction"] == "inbound" else None
        if book is None or not rid or rid in book:
            unmatched.append(v)
        else:
            book[rid] = v

    events: list[Event] = []
    for rid, ob in outbound.items():
        ib = inbound.pop(rid, None)
        svc_id = _svc_entity(resolver, ob["svc_fqdn"])
        if ib is not None:
            # matched pair: client pod -> server pod, timing/status from server
            src = (ob["reporting_pod"], POD)
            dst = (ib["reporting_pod"], POD)
            events.append(_mk_event(run_id, ib | {"event_id": ib["event_id"]},
                                    src, dst, svc_id, reporter_views=2))
        else:
            # outbound only: client pod -> Service (server view lost/absent)
            dst = (svc_id, "Service") if svc_id else \
                (resolver.ext_entity("world"), EXTERNAL)
            events.append(_mk_event(run_id, ob, (ob["reporting_pod"], POD),
                                    dst, svc_id, reporter_views=1))
    for ib in list(inbound.values()) + unmatched:
        # inbound only (e.g. the sidecar-less load client): src from the
        # downstream remote IP, dst = the reporting (server) pod
        ip = _split_hostport(ib["rec"].get("downstream_remote_address"))
        src = resolver.resolve_ip(ip) or (resolver.ext_entity("world"), EXTERNAL)
        if ib["direction"] == "inbound":
            dst = (ib["reporting_pod"], POD)
            events.append(_mk_event(run_id, ib, src, dst, None, reporter_views=1))
        else:
            svc_id = _svc_entity(resolver, ib["svc_fqdn"])
            dst = (svc_id, "Service") if svc_id else \
                (resolver.ext_entity("world"), EXTERNAL)
            events.append(_mk_event(run_id, ib, (ib["reporting_pod"], POD),
                                    dst, svc_id, reporter_views=1))
    return events
