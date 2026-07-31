"""PPT scoring records: schema, OTHER handling, newest-copy-only emission."""
import torch

from gnnid.config import detector_view
from gnnid.dataset import entity_meta_map
from gnnid.detectors.base import SCORE_COLUMNS
from gnnid.detectors.ppt.features import PPTFeatureSpec
from gnnid.detectors.ppt.graph import build_memory_graphs
from gnnid.detectors.ppt.model import PPTModel
from gnnid.detectors.ppt.score import score_memory_graph
from gnnid.labels import LabelVocab
from gnnid.schema import RPC_CALL, Entity, Event, entities_to_frame, \
    events_to_frame


def _setup(cfg):
    cfg.dotted_set("detector", "ppt")
    pcfg = detector_view(cfg)
    # pod "a" is in the app namespace (labeled), pod "x" is cross-namespace
    # -> entity_label None -> OTHER
    ents = [Entity(run_id="r", entity_id="pod:default/a", entity_type="Pod",
                   name="a", namespace="default", canonical_service="a"),
            Entity(run_id="r", entity_id="pod:kube-system/x",
                   entity_type="Pod", name="x", namespace="kube-system",
                   canonical_service="x")]
    events = [Event(event_id=f"e{i}", run_id="r", ts=float(i),
                    event_type=RPC_CALL, src_id="pod:default/a",
                    src_type="Pod", dst_id="pod:kube-system/x",
                    dst_type="Pod", protocol="http", method="GET", path="/x",
                    status_code=200) for i in range(3)]
    ent = entities_to_frame(ents)
    ev = events_to_frame(events)
    vocab = LabelVocab.fit(ent)
    spec = PPTFeatureSpec.fit([ev], pcfg)
    graphs = list(build_memory_graphs(ev, entity_meta_map(ent), vocab, spec,
                                      pcfg, "r"))
    torch.manual_seed(0)
    model = PPTModel(spec.entity_dim, spec.event_dim, 32, 2,
                     vocab.num_classes, 0.0)
    return graphs, model, vocab


def test_records_schema_and_other_forced(cfg):
    graphs, model, vocab = _setup(cfg)
    recs = score_memory_graph(graphs[0], model, vocab, norm={})
    assert recs
    for r in recs:
        assert set(SCORE_COLUMNS) <= set(r)
        assert 0.0 <= r["score_raw"] <= 1.0
    other = [r for r in recs if r["true_label"] == "<other>"]
    assert other and all(r["score_raw"] == 1.0 and r["p_true"] == 0.0
                         for r in other)


def test_only_newest_copies_emit(cfg):
    graphs, model, vocab = _setup(cfg)
    for g in graphs:
        recs = score_memory_graph(g, model, vocab, norm={})
        n_newest = int(g.data["entity"].score_mask.sum())
        assert len(recs) == n_newest
        assert all(r["w_idx"] == g.w_idx for r in recs)


def test_score_events_only_entity_complete_across_windows(cfg):
    """Detector contract: rows for only_entity must be complete for every
    window it appears in — including after a window it is absent from — and
    identical to a full pass (window memory must stay intact)."""
    import pandas as pd

    from gnnid.detectors.ppt import PPTDetector

    cfg.dotted_set("detector", "ppt")
    pcfg = detector_view(cfg)
    ents = [Entity(run_id="r", entity_id=f"pod:default/{n}", entity_type="Pod",
                   name=n, namespace="default", canonical_service=n)
            for n in ("a", "b", "c")]
    ent = entities_to_frame(ents)

    def _rpc(i, ts, src, dst):
        return Event(event_id=f"e{i}", run_id="r", ts=ts, event_type=RPC_CALL,
                     src_id=f"pod:default/{src}", src_type="Pod",
                     dst_id=f"pod:default/{dst}", dst_type="Pod",
                     protocol="http", method="GET", path="/x",
                     status_code=200)

    # a active in windows 0 and 2 ([0,5) and [10,15)), absent from window 1
    ev = events_to_frame([_rpc(0, 0.0, "a", "b"), _rpc(1, 3.0, "a", "b"),
                          _rpc(2, 5.0, "b", "c"), _rpc(3, 8.0, "b", "c"),
                          _rpc(4, 10.0, "a", "b"), _rpc(5, 13.0, "a", "b")])
    vocab = LabelVocab.fit(ent)
    spec = PPTFeatureSpec.fit([ev], pcfg)
    torch.manual_seed(0)
    model = PPTModel(spec.entity_dim, spec.event_dim, 32, 2,
                     vocab.num_classes, 0.0)
    det = PPTDetector(cfg=pcfg, vocab=vocab, spec=spec, model=model,
                      enc_state=None, norm={}, thresholds={}, manifest={})

    full = det.score_events(ev, ent, pcfg)
    only = det.score_events(ev, ent, pcfg, only_entity="pod:default/a")
    a_full = full[full.entity_id == "pod:default/a"].reset_index(drop=True)
    a_only = only[only.entity_id == "pod:default/a"].reset_index(drop=True)
    assert set(a_full.w_idx) == {0, 2}           # the fixture is multi-window
    assert set(a_only.w_idx) == set(a_full.w_idx)
    pd.testing.assert_frame_equal(a_only, a_full)
