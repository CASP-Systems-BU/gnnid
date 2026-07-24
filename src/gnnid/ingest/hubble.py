"""Hubble flows.jsonl -> L4_FLOW + DNS_QUERY events.

Hubble emits many observations per connection; the L4 selection rule keeps
only informative ones:
  * verdict != FORWARDED           (drops/denials — always keep)
  * TCP SYN without ACK            (connection open)
  * UDP/ICMP with is_reply=false   (first packet of an exchange)
DNS: one DNS_QUERY per l7.dns RESPONSE line (it carries the rcode; every
request gets a response, NXDOMAIN included). Runs without L7 DNS visibility
simply produce no DNS events.
"""
from __future__ import annotations

import datetime
import json

from ..resolve import EntityResolver, normalize_dns
from ..schema import DNS_QUERY, L4_FLOW, Event


def _iso_epoch(ts: str | None) -> float | None:
    if not isinstance(ts, str):
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def _l4(flow: dict) -> tuple[str | None, int | None, dict]:
    l4 = flow.get("l4") or {}
    for proto in ("TCP", "UDP", "ICMPv4", "ICMPv6"):
        if proto in l4:
            p = l4[proto] or {}
            name = "icmp" if proto.startswith("ICMP") else proto.lower()
            return name, p.get("destination_port"), p.get("flags") or {}
    return None, None, {}


def keep_l4(flow: dict, keep_rules: list[str]) -> bool:
    verdict = flow.get("verdict", "UNKNOWN")
    proto, _, flags = _l4(flow)
    if "non_forwarded" in keep_rules and verdict != "FORWARDED":
        return True
    if ("syn_no_ack" in keep_rules and proto == "tcp"
            and flags.get("SYN") and not flags.get("ACK")):
        return True
    if ("udp_first" in keep_rules and proto in ("udp", "icmp")
            and not flow.get("is_reply", False)):
        return True
    return False


def parse_flows(text: str, run_id: str, resolver: EntityResolver,
                keep_rules: list[str], source_file: str = "cilium/flows.jsonl",
                ) -> list[Event]:
    events: list[Event] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        flow = obj.get("flow", obj)
        ts = _iso_epoch(flow.get("time"))
        if ts is None:
            continue
        eid = f"{run_id}:{source_file}:{lineno}"
        dns = (flow.get("l7") or {}).get("dns") or {}

        if dns.get("query"):
            # DNS_QUERY: querier pod -> DNSName. RESPONSE carries the rcode and
            # the querier is the *destination* endpoint of the response flow.
            l7type = (flow.get("l7") or {}).get("type", "")
            if l7type != "RESPONSE":
                continue
            querier = resolver.resolve_hubble_endpoint(flow.get("destination"))
            if querier is None:
                continue
            fqdn = normalize_dns(dns["query"])
            events.append(Event(
                event_id=eid, run_id=run_id, ts=ts, event_type=DNS_QUERY,
                src_id=querier[0], src_type=querier[1],
                dst_id=resolver.dns_entity(fqdn), dst_type="DNSName",
                dns_query=fqdn,
                dns_qtypes=",".join(dns.get("qtypes") or []),
                dns_rcode=str(dns.get("rcode", "")) or None))
            continue

        if not keep_l4(flow, keep_rules):
            continue
        src = resolver.resolve_hubble_endpoint(flow.get("source"))
        dst = resolver.resolve_hubble_endpoint(flow.get("destination"))
        if src is None or dst is None:
            continue
        proto, dport, _ = _l4(flow)
        events.append(Event(
            event_id=eid, run_id=run_id, ts=ts, event_type=L4_FLOW,
            src_id=src[0], src_type=src[1], dst_id=dst[0], dst_type=dst[1],
            l4_proto=proto, dst_port=dport,
            verdict=flow.get("verdict", "UNKNOWN"),
            drop_reason=flow.get("drop_reason_desc"),
            is_reply=bool(flow.get("is_reply", False))))
    return events
