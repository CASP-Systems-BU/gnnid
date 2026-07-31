"""PPT-GNN memory graphs: sliding windows -> heterogeneous line graphs.

One training/scoring example is a "memory graph": the newest window plus up to
`windows.memory - 1` preceding windows of the same run. Telemetry events are
"event" nodes carrying identity-free features; canonical entities are "entity"
endpoint nodes, one COPY per window they appear in. Only the newest window's
entity copies carry loss and scores (`score_mask`) — every window is newest
exactly once, so training matches online inference.

Edge types (all instantiated in every graph, empty ones as [2, 0] tensors, so
HeteroConv output and Batch collation are deterministic):
  spatial   (entity src_of event), (event rev_src_of entity),
            (entity dst_of event), (event rev_dst_of entity)
            [+ the RPC fronting-Service dst pair, mirroring graph.py:74-77]
            DNS_QUERY events get NO rev_dst_of edge: their features carry the
            queried name's class/eTLD+1 (legitimate querier-side behavior,
            FLASH's "out" sentence), and the dst IS the named DNSName node —
            messaging it would let its own name enter its own role prediction
            (the identity gate FLASH applies by omitting dst tokens from the
            queried node's "in" sentence).
  temporal  (event follows_src event)  same-src chain within a window,
            (event follows_dst event)  same-dst chain within a window,
              both past->future, capped at graph.flow_memory predecessors
            (entity recurs entity)     all-pairs past->future among an
              entity's window copies (all-pairs, not chains: 2 GNN layers
              must reach the whole memory)

Window positions are derived from t0 — never from w_idx, which skips empty
windows (windows.py:37-41); a gap simply contributes no copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ... import dataset
from ...config import Config
from ...labels import LabelVocab
from ...schema import DNS_QUERY, ENTITY_TYPES, L4_FLOW, RPC_CALL, WORKLOAD
from ...windows import Window, iter_windows
from .features import PPTFeatureSpec, entity_features, event_features, event_view

SPATIAL_EDGE_TYPES = (
    ("entity", "src_of", "event"),
    ("event", "rev_src_of", "entity"),
    ("entity", "dst_of", "event"),
    ("event", "rev_dst_of", "entity"),
)
TEMPORAL_EDGE_TYPES = (
    ("event", "follows_src", "event"),
    ("event", "follows_dst", "event"),
    ("entity", "recurs", "entity"),
)
ALL_EDGE_TYPES = SPATIAL_EDGE_TYPES + TEMPORAL_EDGE_TYPES


@dataclass
class MemoryGraph:
    data: HeteroData
    entity_ids: list[str]   # entity node index -> canonical id (per copy)
    run_id: str
    w_idx: int              # iter_windows index of the NEWEST window


def _cap_events(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Bound a window's events; non-FORWARDED L4 flows are always kept (they
    are the weak-signal positives), the rest is stride-downsampled."""
    if len(df) <= cap:
        return df.sort_values(["ts", "event_id"], kind="stable")
    must_mask = ((df.event_type == L4_FLOW) & df.verdict.notna()
                 & (df.verdict != "FORWARDED"))
    must, rest = df[must_mask], df[~must_mask]
    n_rest = max(cap - len(must), 0)
    if n_rest and len(rest):
        idx = np.unique(np.linspace(0, len(rest) - 1, num=n_rest, dtype=int))
        rest = rest.iloc[idx]
    else:
        rest = rest.iloc[0:0]
    out = pd.concat([must, rest])
    return out.sort_values(["ts", "event_id"], kind="stable")


def _incident(row, entity_meta: dict, include_workload: bool) -> list[str]:
    """Entities an event connects to, mirroring graph.py's selection: present
    in entity_meta, Workload excluded unless configured in."""
    ids = [row.src_id, row.dst_id]
    if row.event_type == RPC_CALL and isinstance(row.dst_svc_id, str) \
            and row.dst_svc_id and row.dst_svc_id != row.dst_id:
        ids.append(row.dst_svc_id)
    out = []
    for eid in ids:
        meta = entity_meta.get(eid)
        if meta is None:
            continue
        if meta[0] == WORKLOAD and not include_workload:
            continue
        out.append(eid)
    return out


def build_memory_graphs(events: pd.DataFrame, entity_meta: dict,
                        vocab: LabelVocab, spec: PPTFeatureSpec, cfg: Config,
                        run_id: str, strip_verdict: bool = False
                        ) -> Iterator[MemoryGraph]:
    """One MemoryGraph per (non-empty) window, in temporal order."""
    if events.empty:
        return
    events = events.sort_values("ts", kind="stable")
    width = float(cfg.windows.width_s)
    stride = float(cfg.windows.stride_s)
    min_tail = float(cfg.dotted_get("windows.min_tail_s", cfg.windows.width_s))
    memory = int(cfg.dotted_get("windows.memory", 5))
    cap = int(cfg.dotted_get("graph.max_events_per_window", 1000))

    wins = list(iter_windows(events, width, stride, min_tail, run_id))
    if not wins:
        return
    first_t0 = wins[0].t0
    positioned = [(int(round((w.t0 - first_t0) / stride)), w) for w in wins]
    capped = [_cap_events(w.events, cap) for _, w in positioned]

    for k, (pos_k, win_k) in enumerate(positioned):
        mem = [(p, w, capped[i])
               for i, (p, w) in enumerate(positioned[:k + 1])
               if p > pos_k - memory]
        g = _build_one(mem, pos_k, win_k, entity_meta, vocab, spec, cfg,
                       run_id, strip_verdict)
        if g is not None:
            yield g


def _build_one(mem: list[tuple[int, Window, pd.DataFrame]], newest_pos: int,
               newest_win: Window, entity_meta: dict, vocab: LabelVocab,
               spec: PPTFeatureSpec, cfg: Config, run_id: str,
               strip_verdict: bool) -> MemoryGraph | None:
    include_workload = bool(cfg.dotted_get("graph.include_workload_nodes", False))
    flow_memory = int(cfg.dotted_get("graph.flow_memory", 20))
    width = float(cfg.windows.width_s)

    # ---- nodes -------------------------------------------------------------
    # entity copies: (pos, entity_id), sorted for determinism
    copies: set[tuple[int, str]] = set()
    per_window_rows: list[tuple[int, Window, list]] = []
    for pos, win, wev in mem:
        rows = list(wev.itertuples(index=False))
        per_window_rows.append((pos, win, rows))
        for row in rows:
            for eid in _incident(row, entity_meta, include_workload):
                copies.add((pos, eid))
    ent_keys = sorted(copies)
    ent_idx = {key: i for i, key in enumerate(ent_keys)}

    n_events = sum(len(rows) for _, _, rows in per_window_rows)
    if not ent_keys and n_events == 0:
        return None

    ent_x, ent_y, ent_nt, ent_mask, entity_ids = [], [], [], [], []
    for pos, eid in ent_keys:
        etype, csvc, ns = entity_meta[eid]
        ent_x.append(entity_features(newest_pos - pos, spec))
        ent_y.append(vocab.index_of(etype, csvc, ns))
        ent_nt.append(ENTITY_TYPES.index(etype))
        ent_mask.append(pos == newest_pos)
        entity_ids.append(eid)

    # event nodes, ordered (pos, ts, event_id); per-window ts order for PE/chains
    ev_x = []
    ev_nodes: list[tuple[int, object]] = []   # (pos, row) aligned with node idx
    for pos, win, rows in per_window_rows:
        for order_idx, row in enumerate(rows):
            rel_time = (float(row.ts) - win.t0) / width
            ev_x.append(event_features(event_view(row), order_idx, rel_time,
                                       spec, strip_verdict))
            ev_nodes.append((pos, row))

    # ---- edges -------------------------------------------------------------
    edges: dict[tuple, tuple[list[int], list[int]]] = \
        {et: ([], []) for et in ALL_EDGE_TYPES}

    def _add(et, s, d):
        edges[et][0].append(s)
        edges[et][1].append(d)

    for j, (pos, row) in enumerate(ev_nodes):
        src_key, dst_key = (pos, row.src_id), (pos, row.dst_id)
        if src_key in ent_idx:
            _add(("entity", "src_of", "event"), ent_idx[src_key], j)
            _add(("event", "rev_src_of", "entity"), j, ent_idx[src_key])
        if dst_key in ent_idx:
            _add(("entity", "dst_of", "event"), ent_idx[dst_key], j)
            # DNS: the dst IS the queried name node; see module docstring
            if row.event_type != DNS_QUERY:
                _add(("event", "rev_dst_of", "entity"), j, ent_idx[dst_key])
        if row.event_type == RPC_CALL and isinstance(row.dst_svc_id, str) \
                and row.dst_svc_id and row.dst_svc_id != row.dst_id:
            svc_key = (pos, row.dst_svc_id)
            if svc_key in ent_idx:
                _add(("entity", "dst_of", "event"), ent_idx[svc_key], j)
                _add(("event", "rev_dst_of", "entity"), j, ent_idx[svc_key])

    # intra-window temporal chains: same src (resp. dst), past -> future,
    # each event linked to at most flow_memory predecessors
    for rel, keyfn in ((("event", "follows_src", "event"),
                        lambda r: r.src_id),
                       (("event", "follows_dst", "event"),
                        lambda r: r.dst_id)):
        groups: dict[tuple[int, str], list[int]] = {}
        for j, (pos, row) in enumerate(ev_nodes):
            groups.setdefault((pos, keyfn(row)), []).append(j)
        for members in groups.values():   # already in (ts, event_id) order
            for jj in range(1, len(members)):
                for ii in range(max(0, jj - flow_memory), jj):
                    _add(rel, members[ii], members[jj])

    # inter-window recurrence: all-pairs forward among an entity's copies
    by_eid: dict[str, list[tuple[int, int]]] = {}
    for i, (pos, eid) in enumerate(ent_keys):
        by_eid.setdefault(eid, []).append((pos, i))
    for occ in by_eid.values():
        for a in range(len(occ)):
            for b in range(a + 1, len(occ)):
                _add(("entity", "recurs", "entity"), occ[a][1], occ[b][1])

    # ---- assemble ------------------------------------------------------------
    data = HeteroData()
    data["entity"].x = torch.from_numpy(
        np.stack(ent_x).astype(np.float32)) if ent_x else \
        torch.zeros((0, spec.entity_dim), dtype=torch.float32)
    data["entity"].y = torch.tensor(ent_y, dtype=torch.long)
    data["entity"].node_type = torch.tensor(ent_nt, dtype=torch.long)
    data["entity"].score_mask = torch.tensor(ent_mask, dtype=torch.bool)
    data["event"].x = torch.from_numpy(
        np.stack(ev_x).astype(np.float32)) if ev_x else \
        torch.zeros((0, spec.event_dim), dtype=torch.float32)
    data["event"].num_nodes = len(ev_x)
    data["entity"].num_nodes = len(ent_x)
    for et, (src, dst) in edges.items():
        data[et].edge_index = torch.tensor([src, dst], dtype=torch.long) \
            if src else torch.zeros((2, 0), dtype=torch.long)
    return MemoryGraph(data, entity_ids, run_id, newest_win.w_idx)


def build_run_memory_graphs(parquet_dir, run_ids: list[str], vocab: LabelVocab,
                            spec: PPTFeatureSpec, cfg: Config,
                            strip_verdict: bool = False) -> list[MemoryGraph]:
    graphs: list[MemoryGraph] = []
    for run_id in run_ids:
        rd = dataset.load_run(parquet_dir, run_id)
        meta = dataset.entity_meta_map(rd.entities)
        graphs.extend(build_memory_graphs(rd.events, meta, vocab, spec, cfg,
                                          run_id, strip_verdict))
    return graphs
