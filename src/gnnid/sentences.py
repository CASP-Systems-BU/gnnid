"""Entity x window -> ordered token "sentence" (FLASH's node sentences).

Leakage is enforced STRUCTURALLY: token builders receive an EventView that
physically lacks every identity field (names, namespaces, IPs, labels), so a
node's own role can never appear in its own sentence. Tokens are colon-
namespaced (`p:/cart/*`, `dst:cluster`) so exact-match leakage tests are
meaningful.

Grammar (per event, appended to BOTH endpoints, direction-flipped):
  RPC   rpc:out|in proto:<http|grpc> m:<method> p:<tpl> st:<class> [fl:<f>...]
  L4    flow:out|in proto:<tcp|udp|icmp> port:<class> v:<forwarded|dropped|denied>
  DNS   dns:q|in qt:<type> dst:<cluster|ext:<etld1>> rc:<rcode>
  API   api:out|in verb:<verb> res:<resource>[/<sub>] ut:<sa|node|user> st:<class>
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import DNS_QUERY, K8S_API_CALL, L4_FLOW, RPC_CALL

# ---------------------------------------------------------------- normalizers
_WELL_KNOWN_PORTS = {
    53: "dns", 80: "http", 443: "https", 3000: "http-alt", 8080: "http-alt",
    5000: "grpc", 7000: "grpc", 7070: "grpc", 8000: "http-alt", 6379: "redis",
    3306: "mysql", 5432: "postgres", 9092: "kafka", 27017: "mongo",
    11211: "memcached", 6443: "kubeapi", 9090: "prom", 15021: "istio-health",
    15090: "istio-metrics", 15012: "istio-xds", 15017: "istio-webhook",
}
_HEXISH = re.compile(r"^[0-9a-f]{8,}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def port_class(port: int | float | None) -> str:
    if port is None or (isinstance(port, float) and port != port):
        return "na"
    port = int(port)
    if port in _WELL_KNOWN_PORTS:
        return _WELL_KNOWN_PORTS[port]
    if port < 1024:
        return "system"
    if port < 49152:
        return "registered"
    return "ephemeral"


def status_class(code: int | float | None) -> str:
    if not code or (isinstance(code, float) and code != code):
        return "na"
    return f"{int(code) // 100}xx"


def _id_like(seg: str) -> bool:
    if not seg:
        return False
    if seg.isdigit() or _UUID.match(seg) or _HEXISH.match(seg):
        return True
    digits = sum(c.isdigit() for c in seg)
    # long alphanumeric token containing a digit -> SKU/ID (e.g. oljcespc7z),
    # or a segment that is mostly digits
    if len(seg) >= 8 and seg.isalnum() and digits >= 1:
        return True
    return len(seg) >= 6 and digits / len(seg) > 0.5


def template_path(path: str | None, protocol: str | None) -> str:
    """Normalize an RPC path. gRPC `/Package.Service/Method` keeps only the
    method (the package.Service segment is the callee's identity, so dropping
    it prevents a node's inbound calls from encoding its own role); HTTP paths
    keep up to 4 segments with ID-like segments templated to `*`."""
    if not path:
        return "/"
    path = path.split("?", 1)[0].split("#", 1)[0].lower()
    if protocol == "grpc":
        parts = [p for p in path.split("/") if p]
        if len(parts) == 2 and "." in parts[0]:
            return f"/*/{parts[1]}"
    segs = path.split("/")
    out = []
    for seg in segs[1:5]:
        out.append("*" if _id_like(seg) else seg)
    return "/" + "/".join(out) if out else "/"


_INTERNAL_DNS = (".svc.cluster.local", ".cluster.local", ".svc", ".local",
                 ".in-addr.arpa", ".cluster")


def dns_dst_token(fqdn: str | None) -> str:
    if not fqdn:
        return "dst:na"
    f = fqdn.lower().rstrip(".")
    if any(f.endswith(s) for s in _INTERNAL_DNS) or "." not in f:
        return "dst:cluster"
    labels = f.split(".")
    etld1 = ".".join(labels[-2:]) if len(labels) >= 2 else f
    return f"dst:ext:{etld1}"


# ------------------------------------------------------------------ EventView
@dataclass
class EventView:
    """Behavior-only projection of an event from ONE endpoint's perspective.
    Deliberately carries NO identity fields — that is the leakage firewall."""
    event_type: str
    perspective: str        # "out" | "in"
    ts: float
    # union of behavior fields (unused ones stay None)
    protocol: str | None = None
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    response_flags: str | None = None
    l4_proto: str | None = None
    dst_port: int | None = None
    verdict: str | None = None
    dns_qtypes: str | None = None
    dns_query: str | None = None
    dns_rcode: str | None = None
    k8s_verb: str | None = None
    k8s_resource: str | None = None
    k8s_subresource: str | None = None
    k8s_user_type: str | None = None
    k8s_status_code: int | None = None
    # behavior metrics (used by PPT features, not by sentence tokens)
    grpc_status: int | None = None
    duration_ms: float | None = None
    request_bytes: int | None = None
    response_bytes: int | None = None
    reporter_views: int | None = None
    drop_reason: str | None = None
    is_reply: bool | None = None


_VERDICT_TOK = {"FORWARDED": "forwarded", "DROPPED": "dropped",
                "ERROR": "error", "AUDIT": "audit"}


def event_tokens(v: EventView, strip_verdict: bool = False) -> list[str]:
    """One EventView -> its behavior tokens (identity-free by construction)."""
    d = v.perspective
    if v.event_type == RPC_CALL:
        toks = [f"rpc:{d}", f"proto:{v.protocol or 'http'}",
                f"m:{(v.method or 'na').lower()}",
                f"p:{template_path(v.path, v.protocol)}",
                f"st:{status_class(v.status_code)}"]
        for fl in (v.response_flags or "-").split(","):
            fl = fl.strip()
            if fl and fl != "-":
                toks.append(f"fl:{fl.lower()}")
        return toks
    if v.event_type == L4_FLOW:
        verdict = _VERDICT_TOK.get(v.verdict or "", (v.verdict or "na").lower())
        toks = [f"flow:{d}", f"proto:{v.l4_proto or 'na'}",
                f"port:{port_class(v.dst_port)}"]
        if not strip_verdict:
            toks.append(f"v:{verdict}")
        return toks
    if v.event_type == DNS_QUERY:
        pref = "dns:q" if d == "out" else "dns:in"
        qt = (v.dns_qtypes or "").split(",")[0].strip().lower() or "na"
        toks = [pref, f"qt:{qt}"]
        if d == "out":
            toks.append(dns_dst_token(v.dns_query))
        toks.append(f"rc:{(v.dns_rcode or 'na').lower()}")
        return toks
    if v.event_type == K8S_API_CALL:
        res = (v.k8s_resource or "na").lower()
        if v.k8s_subresource:
            res += f"/{v.k8s_subresource.lower()}"
        return [f"api:{d}", f"verb:{(v.k8s_verb or 'na').lower()}",
                f"res:{res}", f"ut:{v.k8s_user_type or 'na'}",
                f"st:{status_class(v.k8s_status_code)}"]
    return []


# ------------------------------------------------------------ sentence builder
def _clean(x):
    """Sparse string/number columns round-trip through pandas as NaN, not None;
    normalize both to None so token builders never see a float where a str is
    expected."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and x != x:  # NaN
            return None
    except TypeError:
        pass
    return x


def _view_from_row(row, perspective: str) -> EventView:
    g = lambda f: _clean(getattr(row, f))  # noqa: E731
    return EventView(
        event_type=row.event_type, perspective=perspective, ts=row.ts,
        protocol=g("protocol"), method=g("method"), path=g("path"),
        status_code=g("status_code"), response_flags=g("response_flags"),
        l4_proto=g("l4_proto"), dst_port=g("dst_port"), verdict=g("verdict"),
        dns_qtypes=g("dns_qtypes"), dns_query=g("dns_query"),
        dns_rcode=g("dns_rcode"), k8s_verb=g("k8s_verb"),
        k8s_resource=g("k8s_resource"), k8s_subresource=g("k8s_subresource"),
        k8s_user_type=g("k8s_user_type"), k8s_status_code=g("k8s_status_code"),
        grpc_status=g("grpc_status"), duration_ms=g("duration_ms"),
        request_bytes=g("request_bytes"), response_bytes=g("response_bytes"),
        reporter_views=g("reporter_views"), drop_reason=g("drop_reason"),
        is_reply=g("is_reply"))


def _dedup_and_cap(items: list, max_repeats: int, max_tokens: int) -> list[str]:
    """items = list[(ts, tuple_of_tokens)]. Keep <= max_repeats occurrences per
    unique token signature (first/middle/last to preserve spread), then flatten
    ts-ordered and stride-downsample to max_tokens."""
    from collections import defaultdict
    by_sig: dict[tuple, list] = defaultdict(list)
    for ts, toks in items:
        by_sig[toks].append((ts, toks))
    kept = []
    for occ in by_sig.values():
        if len(occ) <= max_repeats:
            kept.extend(occ)
        else:
            idxs = sorted({0, len(occ) // 2, len(occ) - 1} |
                          set(range(min(max_repeats, len(occ)))))
            kept.extend(occ[i] for i in idxs[:max_repeats])
    kept.sort(key=lambda x: x[0])
    flat = [t for _, toks in kept for t in toks]
    if len(flat) > max_tokens:
        step = len(flat) / max_tokens
        flat = [flat[int(i * step)] for i in range(max_tokens)]
    return flat


def build_sentences(win_events, entity_ids: set[str], cfg,
                    strip_verdict: bool | None = None) -> dict[str, list[str]]:
    """One window's events -> {entity_id: token list}. Every entity in
    entity_ids gets a sentence (possibly empty)."""
    if strip_verdict is None:
        strip_verdict = bool(cfg.dotted_get("sentences.strip_verdict_tokens", False))
    max_repeats = int(cfg.dotted_get("sentences.max_repeats_per_signature", 3))
    max_tokens = int(cfg.dotted_get("sentences.max_tokens", 256))

    per_entity: dict[str, list] = {eid: [] for eid in entity_ids}
    for row in win_events.itertuples(index=False):
        # src -> "out" perspective, dst -> "in" perspective
        if row.src_id in per_entity:
            toks = tuple(event_tokens(_view_from_row(row, "out"), strip_verdict))
            if toks:
                per_entity[row.src_id].append((row.ts, toks))
        if row.dst_id in per_entity:
            toks = tuple(event_tokens(_view_from_row(row, "in"), strip_verdict))
            if toks:
                per_entity[row.dst_id].append((row.ts, toks))
        # RPC also gives the fronting Service an inbound view (role aggregation)
        svc = getattr(row, "dst_svc_id", None)
        if svc and svc != row.dst_id and svc in per_entity:
            toks = tuple(event_tokens(_view_from_row(row, "in"), strip_verdict))
            if toks:
                per_entity[svc].append((row.ts, toks))

    return {eid: _dedup_and_cap(items, max_repeats, max_tokens)
            for eid, items in per_entity.items()}
