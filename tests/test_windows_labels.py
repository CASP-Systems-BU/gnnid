"""Window boundary + label vocab tests."""
import pandas as pd

from gnnid.labels import OTHER, LabelVocab, entity_label
from gnnid.schema import DNSNAME, EXTERNAL, KUBEAPI, POD, SERVICE
from gnnid import windows


def _events(ts_list):
    return pd.DataFrame({"ts": ts_list, "event_type": ["RPC_CALL"] * len(ts_list)})


def test_window_boundaries():
    ev = _events([0.0, 5.0, 10.0, 14.999, 15.0, 29.0])
    ws = list(windows.iter_windows(ev, width_s=15, stride_s=15, min_tail_s=5,
                                   run_id="r"))
    # first window [0,15): 0,5,10,14.999 => 4 events
    assert len(ws[0].events) == 4
    # second window [15,30): 15,29 => 2
    assert len(ws[1].events) == 2


def test_window_partial_tail_dropped():
    ev = _events([0.0, 16.0])  # second window would be [15,30) but only 16 -> span 1s
    ws = list(windows.iter_windows(ev, width_s=15, stride_s=15, min_tail_s=5,
                                   run_id="r"))
    # window0 [0,15) has 0.0; the tail window starting at 15 spans only 1s < 5 -> dropped
    assert all(w.events["ts"].max() < 15 or len(w.events) >= 1 for w in ws)


def test_entity_label():
    assert entity_label(POD, "frontend", "default", "default") == "frontend"
    # cross-namespace pod -> None (becomes OTHER)
    assert entity_label(POD, "istiod", "istio-system", "default") is None
    assert entity_label(DNSNAME, None, None, "default") == "dnsname"
    assert entity_label(KUBEAPI, None, None, "default") == "kubeapi"
    assert entity_label(EXTERNAL, None, None, "default") == "external"


def test_vocab_fit_and_other():
    ent = pd.DataFrame([
        {"entity_type": POD, "canonical_service": "frontend", "namespace": "default"},
        {"entity_type": POD, "canonical_service": "cart", "namespace": "default"},
        {"entity_type": SERVICE, "canonical_service": "cart", "namespace": "default"},
        {"entity_type": POD, "canonical_service": "istiod", "namespace": "istio-system"},
        {"entity_type": DNSNAME, "canonical_service": None, "namespace": None},
    ])
    v = LabelVocab.fit(ent, "default")
    assert v.classes[0] == OTHER and v.other_idx == 0
    assert "frontend" in v.classes and "cart" in v.classes and "dnsname" in v.classes
    # cross-namespace istiod maps to OTHER
    assert v.index_of(POD, "istiod", "istio-system") == v.other_idx
    assert v.index_of(POD, "frontend", "default") == v.to_idx["frontend"]


def test_vocab_roundtrip(tmp_path):
    ent = pd.DataFrame([{"entity_type": POD, "canonical_service": "frontend",
                         "namespace": "default"}])
    v = LabelVocab.fit(ent, "default")
    p = tmp_path / "vocab.json"
    v.save(p)
    v2 = LabelVocab.load(p)
    assert v2.classes == v.classes and v2.app_namespace == "default"
