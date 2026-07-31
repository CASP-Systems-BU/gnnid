"""PPT-GNN featurization: identity-free event and entity vectors.

Event features are computed EXCLUSIVELY from sentences.EventView — the same
structural leakage firewall the FLASH sentences use (the type physically lacks
every identity field) — through the shared normalizers port_class /
status_class / template_path / dns_dst_token. A vector is a shared block
(event-type one-hot + order-in-window sinusoidal PE + relative time) plus one
active per-type block; the other type blocks stay zero. Entity nodes carry
only a bias 1.0 and a window-age PE — no type, name, namespace, or IP (the
type IS the label for DNSName/External/KubeAPI nodes; see graph.py:2-6).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...schema import DNS_QUERY, EVENT_TYPES, K8S_API_CALL, L4_FLOW, RPC_CALL
from ...sentences import (_WELL_KNOWN_PORTS, EventView, _view_from_row,
                          dns_dst_token, port_class, status_class,
                          template_path)

OOV = "<oov>"

# fixed categorical slots (order is part of the feature contract)
_PROTOCOLS = ("http", "grpc", "na")
_METHODS = ("get", "post", "put", "delete", "patch", "head", "options",
            "connect", "other", "na")
_STATUS = ("na", "1xx", "2xx", "3xx", "4xx", "5xx")
_GRPC_STATUS = ("na", "ok", "err")
_RESP_FLAGS = ("uh", "uf", "uo", "nr", "urx", "nc", "dc", "lr", "ut", "rl",
               "other")
_L4_PROTO = ("tcp", "udp", "icmp", "na")
_PORT_CLASSES = tuple(sorted(set(_WELL_KNOWN_PORTS.values()))) + \
    ("system", "registered", "ephemeral", "na")
_VERDICTS = ("forwarded", "dropped", "error", "audit", "other", "na")
_DROP_REASONS = ("na", "policy_denied", "other")
_TRISTATE = ("true", "false", "na")
_QTYPES = ("a", "aaaa", "cname", "srv", "ptr", "txt", "mx", "ns", "other",
           "na")
_DNS_DST = ("cluster", "ext", "na")
_RCODES = ("noerror", "nxdomain", "servfail", "refused", "other", "na")
_RCODE_NUM = {"0": "noerror", "2": "servfail", "3": "nxdomain", "5": "refused"}
_VERBS = ("get", "list", "watch", "create", "update", "patch", "delete",
          "deletecollection", "other", "na")
_SUBRES = ("na", "status", "exec", "log", "attach", "portforward", "other")
_USER_TYPES = ("sa", "node", "user", "na")

_SCALED_FIELDS = ("duration_ms", "request_bytes", "response_bytes")


def _slot(slots: tuple[str, ...], value: str | None) -> int:
    v = (value or "na").lower()
    if v in slots:
        return slots.index(v)
    return slots.index("other") if "other" in slots else slots.index("na")


def _pe_row(pos: int, dim: int) -> np.ndarray:
    """One row of the standard sinusoidal PE (matches embed._positional_encoding)."""
    i = np.arange(dim)
    angle = pos / np.power(10000.0, (2 * (i // 2)) / dim)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle)).astype(np.float32)


def event_view(row) -> EventView:
    """Events-frame row (itertuples) -> identity-free EventView. Direction
    lives in the graph's edge types, so one 'out' perspective suffices."""
    return _view_from_row(row, "out")


@dataclass
class PPTFeatureSpec:
    """Train-fit vocabularies + scalers; fully determines the feature dims."""
    path_vocab: list[str]              # index 0 = OOV
    dns_vocab: list[str]               # external eTLD+1, index 0 = OOV
    api_res_vocab: list[str]           # index 0 = OOV
    scalers: dict[str, tuple[float, float]]   # field -> (mean, std) of log1p
    time_enc_dim: int = 16
    path_dim: int = 64                 # one-hot block widths (>= len(vocab))
    dns_dim: int = 32
    api_dim: int = 32

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, event_frames: list[pd.DataFrame], cfg) -> "PPTFeatureSpec":
        time_enc = int(cfg.dotted_get("features.time_enc_dim", 16))
        k_path = int(cfg.dotted_get("features.path_vocab", 64))
        k_dns = int(cfg.dotted_get("features.dns_vocab", 32))
        k_api = int(cfg.dotted_get("features.api_resource_vocab", 32))
        paths: Counter = Counter()
        dnss: Counter = Counter()
        apis: Counter = Counter()
        scaled: dict[str, list[float]] = {f: [] for f in _SCALED_FIELDS}
        for events in event_frames:
            for row in events.itertuples(index=False):
                v = event_view(row)
                if v.event_type == RPC_CALL:
                    paths[template_path(v.path, v.protocol)] += 1
                    for f in _SCALED_FIELDS:
                        x = getattr(v, f)
                        if x is not None:
                            scaled[f].append(math.log1p(max(float(x), 0.0)))
                elif v.event_type == DNS_QUERY:
                    tok = dns_dst_token(v.dns_query)
                    if tok.startswith("dst:ext:"):
                        dnss[tok[len("dst:ext:"):]] += 1
                elif v.event_type == K8S_API_CALL:
                    apis[(v.k8s_resource or "na").lower()] += 1

        def top(counter: Counter, k: int) -> list[str]:
            return [OOV] + [w for w, _ in counter.most_common(max(k - 1, 0))]

        scalers = {}
        for f, xs in scaled.items():
            mu = float(np.mean(xs)) if xs else 0.0
            sd = float(np.std(xs)) if xs else 0.0
            scalers[f] = (mu, sd if sd > 0 else 1.0)
        return cls(top(paths, k_path), top(dnss, k_dns), top(apis, k_api),
                   scalers, time_enc, k_path, k_dns, k_api)

    # ------------------------------------------------------------------ dims
    @property
    def shared_dim(self) -> int:
        return len(EVENT_TYPES) + self.time_enc_dim + 1

    @property
    def rpc_dim(self) -> int:
        return (len(_PROTOCOLS) + len(_METHODS) + self.path_dim + len(_STATUS)
                + len(_GRPC_STATUS) + len(_RESP_FLAGS) + len(_SCALED_FIELDS)
                + 1)  # + reporter_views flag

    @property
    def l4_dim(self) -> int:
        return (len(_L4_PROTO) + len(_PORT_CLASSES) + len(_VERDICTS)
                + len(_DROP_REASONS) + len(_TRISTATE))

    @property
    def dns_dim_total(self) -> int:
        return len(_QTYPES) + len(_DNS_DST) + self.dns_dim + len(_RCODES)

    @property
    def api_dim_total(self) -> int:
        return (len(_VERBS) + self.api_dim + len(_SUBRES) + len(_USER_TYPES)
                + len(_STATUS))

    @property
    def event_dim(self) -> int:
        return (self.shared_dim + self.rpc_dim + self.l4_dim
                + self.dns_dim_total + self.api_dim_total)

    @property
    def entity_dim(self) -> int:
        return 1 + self.time_enc_dim

    def _scale(self, f: str, x) -> float:
        if x is None:
            return 0.0
        mu, sd = self.scalers[f]
        z = (math.log1p(max(float(x), 0.0)) - mu) / sd
        return float(np.clip(z, -5.0, 5.0))

    # ---------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PPTFeatureSpec":
        with open(path) as f:
            d = json.load(f)
        d["scalers"] = {k: tuple(v) for k, v in d["scalers"].items()}
        return cls(**d)


def event_features(view: EventView, order_idx: int, rel_time: float,
                   spec: PPTFeatureSpec,
                   strip_verdict: bool = False) -> np.ndarray:
    """One event node's feature vector. Accepts ONLY EventView — the
    structural guarantee that identity fields are unreachable."""
    if not isinstance(view, EventView):
        raise TypeError("event_features accepts sentences.EventView only")
    x = np.zeros(spec.event_dim, dtype=np.float32)

    # shared block: event-type one-hot + order PE + relative time
    if view.event_type in EVENT_TYPES:
        x[EVENT_TYPES.index(view.event_type)] = 1.0
    o = len(EVENT_TYPES)
    x[o:o + spec.time_enc_dim] = _pe_row(order_idx, spec.time_enc_dim)
    x[o + spec.time_enc_dim] = float(rel_time)

    off = spec.shared_dim
    if view.event_type == RPC_CALL:
        x[off + _slot(_PROTOCOLS, view.protocol)] = 1.0
        off += len(_PROTOCOLS)
        x[off + _slot(_METHODS, view.method)] = 1.0
        off += len(_METHODS)
        tpl = template_path(view.path, view.protocol)
        x[off + (spec.path_vocab.index(tpl) if tpl in spec.path_vocab else 0)] = 1.0
        off += spec.path_dim
        x[off + _slot(_STATUS, status_class(view.status_code))] = 1.0
        off += len(_STATUS)
        gs = "na" if view.grpc_status is None else \
            ("ok" if int(view.grpc_status) == 0 else "err")
        x[off + _GRPC_STATUS.index(gs)] = 1.0
        off += len(_GRPC_STATUS)
        for fl in (view.response_flags or "-").split(","):
            fl = fl.strip().lower()
            if fl and fl != "-":
                x[off + _slot(_RESP_FLAGS, fl)] = 1.0
        off += len(_RESP_FLAGS)
        for f in _SCALED_FIELDS:
            x[off] = spec._scale(f, getattr(view, f))
            off += 1
        x[off] = 1.0 if view.reporter_views == 2 else 0.0
        return x

    off += spec.rpc_dim
    if view.event_type == L4_FLOW:
        x[off + _slot(_L4_PROTO, view.l4_proto)] = 1.0
        off += len(_L4_PROTO)
        x[off + _PORT_CLASSES.index(port_class(view.dst_port))] = 1.0
        off += len(_PORT_CLASSES)
        if not strip_verdict:
            # verdict + drop_reason are both verdict-derived; the weak-signal
            # honesty check (strip_verdict=True) zeroes them together
            x[off + _slot(_VERDICTS, view.verdict)] = 1.0
            dr = "na" if not view.drop_reason else \
                ("policy_denied" if view.drop_reason.upper() == "POLICY_DENIED"
                 else "other")
            x[off + len(_VERDICTS) + _DROP_REASONS.index(dr)] = 1.0
        off += len(_VERDICTS) + len(_DROP_REASONS)
        rep = "na" if view.is_reply is None else str(bool(view.is_reply)).lower()
        x[off + _TRISTATE.index(rep)] = 1.0
        return x

    off += spec.l4_dim
    if view.event_type == DNS_QUERY:
        qt = (view.dns_qtypes or "").split(",")[0].strip().lower() or "na"
        x[off + _slot(_QTYPES, qt)] = 1.0
        off += len(_QTYPES)
        tok = dns_dst_token(view.dns_query)
        if tok == "dst:cluster":
            x[off + _DNS_DST.index("cluster")] = 1.0
        elif tok.startswith("dst:ext:"):
            x[off + _DNS_DST.index("ext")] = 1.0
            etld1 = tok[len("dst:ext:"):]
            x[off + len(_DNS_DST)
              + (spec.dns_vocab.index(etld1) if etld1 in spec.dns_vocab else 0)] = 1.0
        else:
            x[off + _DNS_DST.index("na")] = 1.0
        off += len(_DNS_DST) + spec.dns_dim
        rc = (view.dns_rcode or "na").strip().lower()
        rc = _RCODE_NUM.get(rc, rc)
        x[off + _slot(_RCODES, rc)] = 1.0
        return x

    off += spec.dns_dim_total
    if view.event_type == K8S_API_CALL:
        x[off + _slot(_VERBS, view.k8s_verb)] = 1.0
        off += len(_VERBS)
        res = (view.k8s_resource or "na").lower()
        x[off + (spec.api_res_vocab.index(res)
                 if res in spec.api_res_vocab else 0)] = 1.0
        off += spec.api_dim
        x[off + _slot(_SUBRES, view.k8s_subresource)] = 1.0
        off += len(_SUBRES)
        x[off + _slot(_USER_TYPES, view.k8s_user_type)] = 1.0
        off += len(_USER_TYPES)
        x[off + _slot(_STATUS, status_class(view.k8s_status_code))] = 1.0
    return x


def entity_features(age: int, spec: PPTFeatureSpec) -> np.ndarray:
    """Entity copies get a bias + window-age PE (0 = newest) and nothing else."""
    return np.concatenate([np.ones(1, dtype=np.float32),
                           _pe_row(age, spec.time_enc_dim)])
