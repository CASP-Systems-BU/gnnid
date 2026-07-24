"""Window -> homogeneous PyG Data.

Nodes = entities incident to >=1 event in the window. Node features x = the
Word2Vec sentence vector ONLY (no node-type one-hot: the label is type-derived,
so appending type would let the classifier bypass behavior and gut the
mismatch-based detector). Edges = deduped directed (src,dst,event_type) pairs
plus reverse edges (SAGE aggregates along edge direction; symmetric adjacency
lets servers see clients). Direction survives in the rpc:out/in sentence tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from .labels import LabelVocab
from .schema import ENTITY_TYPES, RPC_CALL, WORKLOAD

_TYPE_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}


@dataclass
class WindowGraph:
    data: Data                 # x, edge_index, y, node_type
    entity_ids: list[str]      # node index -> entity_id
    run_id: str
    w_idx: int


def incident_entities(win_events, include_workload: bool) -> set[str]:
    ids: set[str] = set()
    for row in win_events.itertuples(index=False):
        ids.add(row.src_id)
        ids.add(row.dst_id)
        svc = getattr(row, "dst_svc_id", None)
        if svc:
            ids.add(svc)
    return ids


def build_window_graph(win, entity_meta: dict, embeddings: dict[str, np.ndarray],
                       vocab: LabelVocab, cfg) -> WindowGraph | None:
    """win: windows.Window; entity_meta: eid -> (type, canonical_service, ns).
    embeddings: eid -> w2v vector. Returns None for empty windows."""
    include_workload = bool(cfg.dotted_get("graph.include_workload_nodes", False))
    reverse = bool(cfg.dotted_get("graph.reverse_edges", True))
    dim = int(cfg.dotted_get("w2v.dim", 64))

    node_ids = sorted(eid for eid in incident_entities(win.events, include_workload)
                      if eid in entity_meta
                      and not (entity_meta[eid][0] == WORKLOAD and not include_workload))
    if not node_ids:
        return None
    idx = {eid: i for i, eid in enumerate(node_ids)}

    x = np.zeros((len(node_ids), dim), dtype=np.float32)
    y = np.zeros(len(node_ids), dtype=np.int64)
    ntype = np.zeros(len(node_ids), dtype=np.int64)
    for eid, i in idx.items():
        etype, canonical, ns = entity_meta[eid]
        vec = embeddings.get(eid)
        if vec is not None:
            x[i] = vec
        y[i] = vocab.index_of(etype, canonical, ns)
        ntype[i] = _TYPE_IDX.get(etype, 0)

    # deduped directed edges: (src, dst, event_type). RPC also -> fronting svc.
    edge_set: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, str]] = set()
    for row in win.events.itertuples(index=False):
        pairs = [(row.src_id, row.dst_id)]
        svc = getattr(row, "dst_svc_id", None)
        if row.event_type == RPC_CALL and svc and svc != row.dst_id:
            pairs.append((row.src_id, svc))
        for s, d in pairs:
            if s not in idx or d not in idx or s == d:
                continue
            key = (idx[s], idx[d], row.event_type)
            if key in seen:
                continue
            seen.add(key)
            edge_set.add((idx[s], idx[d]))
            if reverse:
                edge_set.add((idx[d], idx[s]))

    if edge_set:
        ei = np.array(sorted(edge_set), dtype=np.int64).T
    else:
        ei = np.zeros((2, 0), dtype=np.int64)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(ei),
        y=torch.from_numpy(y),
        node_type=torch.from_numpy(ntype))
    data.num_nodes = len(node_ids)
    return WindowGraph(data, node_ids, win.run_id, win.w_idx)
