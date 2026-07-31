"""PPT memory-graph builder: edge-type correctness, memory, chains, masks."""
import torch

from gnnid.config import detector_view
from gnnid.dataset import entity_meta_map
from gnnid.detectors.ppt.features import PPTFeatureSpec
from gnnid.detectors.ppt.graph import (ALL_EDGE_TYPES, build_memory_graphs)
from gnnid.ingest.run_dir import ingest_run
from gnnid.labels import LabelVocab
from gnnid.schema import (DNS_QUERY, RPC_CALL, WORKLOAD, Entity, Event,
                          entities_to_frame, events_to_frame)

SRC_OF = ("entity", "src_of", "event")
DST_OF = ("entity", "dst_of", "event")
FOLLOWS_SRC = ("event", "follows_src", "event")
RECURS = ("entity", "recurs", "entity")


def _pcfg(cfg):
    cfg.dotted_set("detector", "ppt")
    return detector_view(cfg)


def _synthetic(n_or_ts, src="pod:default/a", dst="pod:default/b"):
    ts_list = n_or_ts if isinstance(n_or_ts, list) else \
        [i * 0.1 for i in range(n_or_ts)]
    events = [Event(event_id=f"e{i:03d}", run_id="r", ts=ts,
                    event_type=RPC_CALL, src_id=src, src_type="Pod",
                    dst_id=dst, dst_type="Pod", protocol="http",
                    method="GET", path="/x", status_code=200)
              for i, ts in enumerate(ts_list)]
    ents = [Entity(run_id="r", entity_id="pod:default/a", entity_type="Pod",
                   name="a", namespace="default", canonical_service="a"),
            Entity(run_id="r", entity_id="pod:default/b", entity_type="Pod",
                   name="b", namespace="default", canonical_service="b")]
    ev, ent = events_to_frame(events), entities_to_frame(ents)
    return ev, entity_meta_map(ent), LabelVocab.fit(ent)


def test_fixture_run_single_window(run_dir, cfg):
    pcfg = _pcfg(cfg)
    ev, ent, _ = ingest_run(run_dir, pcfg)
    meta = entity_meta_map(ent)
    vocab = LabelVocab.fit(ent)
    spec = PPTFeatureSpec.fit([ev], pcfg)
    graphs = list(build_memory_graphs(ev, meta, vocab, spec, pcfg, "r"))
    assert len(graphs) == 1                      # fixture spans ~0.2s < 5s
    d = graphs[0].data
    for et in ALL_EDGE_TYPES:                    # every relation instantiated
        assert d[et].edge_index.shape[0] == 2
    # spatial counts derived from the events themselves
    exp_src, exp_dst, n_dns_dst = 0, 0, 0
    for r in ev.itertuples(index=False):
        if r.src_id in meta and meta[r.src_id][0] != WORKLOAD:
            exp_src += 1
        if r.dst_id in meta and meta[r.dst_id][0] != WORKLOAD:
            exp_dst += 1
            if r.event_type == DNS_QUERY:
                n_dns_dst += 1                   # gets no rev_dst_of (leak gate)
        if r.event_type == RPC_CALL and isinstance(r.dst_svc_id, str) \
                and r.dst_svc_id and r.dst_svc_id != r.dst_id \
                and r.dst_svc_id in meta:
            exp_dst += 1                         # fronting-Service pair
    assert d[SRC_OF].edge_index.shape[1] == exp_src
    assert d[DST_OF].edge_index.shape[1] == exp_dst
    assert d[("event", "rev_src_of", "entity")].edge_index.shape[1] == exp_src
    assert d[("event", "rev_dst_of", "entity")].edge_index.shape[1] == \
        exp_dst - n_dns_dst
    assert n_dns_dst > 0                         # the DNS gate actually fired
    assert exp_dst > exp_src                     # the svc pair actually fired
    # single window: no recurrence, everything scored
    assert d[RECURS].edge_index.shape[1] == 0
    assert bool(d["entity"].score_mask.all())
    assert d[FOLLOWS_SRC].edge_index.shape[1] > 0
    assert d["event"].x.shape == (len(ev), spec.event_dim)
    assert d["entity"].x.shape[1] == spec.entity_dim
    assert torch.isfinite(d["event"].x).all()


def test_memory_and_recurs_across_gap(run_dir, cfg):
    pcfg = _pcfg(cfg)                            # width 5, stride 5, memory 5
    # last event 3s into its window so the trailing window passes min_tail_s
    ev, meta, vocab = _synthetic([0.0, 1.0, 20.0, 23.0])
    spec = PPTFeatureSpec.fit([ev], pcfg)
    graphs = list(build_memory_graphs(ev, meta, vocab, spec, pcfg, "r"))
    assert [g.w_idx for g in graphs] == [0, 1]   # empty windows skipped
    g1 = graphs[1].data
    # gap: window t0=20 is position 4, still inside the 5-window memory of pos 0
    assert g1["entity"].x.shape[0] == 4          # a,b copies in both windows
    ei = g1[RECURS].edge_index
    assert ei.shape[1] == 2                      # one forward edge per entity
    assert (ei[0] < ei[1]).all()                 # past -> future only
    # only the newest window's copies are scored
    assert g1["entity"].score_mask.tolist() == [False, False, True, True]
    assert graphs[1].entity_ids[2:] == ["pod:default/a", "pod:default/b"]


def test_memory_horizon_excludes_old_windows(run_dir, cfg):
    pcfg = _pcfg(cfg)
    pcfg.dotted_set("windows.memory", 3)
    ev, meta, vocab = _synthetic([0.0, 10.0, 20.0, 23.0])  # positions 0, 2, 4
    spec = PPTFeatureSpec.fit([ev], pcfg)
    graphs = list(build_memory_graphs(ev, meta, vocab, spec, pcfg, "r"))
    g_last = graphs[-1].data
    # memory=3 at pos 4 keeps pos > 1: windows at pos 2 and 4 only
    assert g_last["entity"].x.shape[0] == 4
    assert g_last[RECURS].edge_index.shape[1] == 2


def test_flow_memory_caps_chains(run_dir, cfg):
    pcfg = _pcfg(cfg)                            # flow_memory 20
    ev, meta, vocab = _synthetic(25)             # one window, same src+dst
    spec = PPTFeatureSpec.fit([ev], pcfg)
    (g,) = build_memory_graphs(ev, meta, vocab, spec, pcfg, "r")
    ei = g.data[FOLLOWS_SRC].edge_index
    assert (ei[0] < ei[1]).all()                 # strictly past -> future
    in_deg = torch.bincount(ei[1], minlength=25)
    assert int(in_deg[24]) == 20                 # newest event: exactly 20
    assert int(in_deg[0]) == 0
    total = sum(min(j, 20) for j in range(25))
    assert ei.shape[1] == total


def test_dns_event_never_messages_its_own_name_node(run_dir, cfg):
    """Anti-leakage: the DNS event node carries the queried name's eTLD+1, so
    it must have NO rev_dst_of edge into the queried DNSName entity — the
    name node's role prediction must stay blind to its own name."""
    pcfg = _pcfg(cfg)
    ents = [Entity(run_id="r", entity_id="pod:default/a", entity_type="Pod",
                   name="a", namespace="default", canonical_service="a"),
            Entity(run_id="r", entity_id="dns:evil.example",
                   entity_type="DNSName", name="evil.example")]
    ent = entities_to_frame(ents)
    ev = events_to_frame([Event(
        event_id="d0", run_id="r", ts=0.0, event_type=DNS_QUERY,
        src_id="pod:default/a", src_type="Pod", dst_id="dns:evil.example",
        dst_type="DNSName", dns_query="evil.example", dns_qtypes="A",
        dns_rcode="0")])
    meta = entity_meta_map(ent)
    spec = PPTFeatureSpec.fit([ev], pcfg)
    (g,) = build_memory_graphs(ev, meta, LabelVocab.fit(ent), spec, pcfg, "r")
    d = g.data
    # querier side intact: src pair + entity->event dst edge exist
    assert d[SRC_OF].edge_index.shape[1] == 1
    assert d[("event", "rev_src_of", "entity")].edge_index.shape[1] == 1
    assert d[DST_OF].edge_index.shape[1] == 1
    # the leak edge is absent: nothing flows event -> DNSName
    assert d[("event", "rev_dst_of", "entity")].edge_index.shape[1] == 0


def test_event_cap_keeps_dropped_l4(run_dir, cfg):
    import pandas as pd

    from gnnid.schema import L4_FLOW
    pcfg = _pcfg(cfg)
    pcfg.dotted_set("graph.max_events_per_window", 10)
    ev, meta, vocab = _synthetic(30)
    dropped = Event(event_id="zzz-dropped", run_id="r", ts=1.55,
                    event_type=L4_FLOW, src_id="pod:default/a",
                    src_type="Pod", dst_id="pod:default/b", dst_type="Pod",
                    l4_proto="tcp", dst_port=9999, verdict="DROPPED",
                    drop_reason="POLICY_DENIED", is_reply=False)
    ev = pd.concat([ev, events_to_frame([dropped])], ignore_index=True) \
        .sort_values("ts", kind="stable")
    spec = PPTFeatureSpec.fit([ev], pcfg)
    (g,) = build_memory_graphs(ev, meta, vocab, spec, pcfg, "r")
    assert g.data["event"].x.shape[0] <= 10
    # the non-FORWARDED L4 event survives the cap: exactly one event node
    # carries the DROPPED verdict slot
    l4_off = spec.shared_dim + spec.rpc_dim
    verdict_block = g.data["event"].x[:, l4_off + 4 + 21:l4_off + 4 + 21 + 6]
    assert int((verdict_block[:, 1] == 1.0).sum()) == 1   # 'dropped' slot
