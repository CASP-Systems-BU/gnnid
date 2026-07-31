"""PPT features: identity fields unreachable, OOV/na slots, spec round trip."""
import numpy as np
import pytest

from gnnid.config import detector_view
from gnnid.detectors.ppt.features import (PPTFeatureSpec, entity_features,
                                          event_features, event_view)
from gnnid.ingest.run_dir import ingest_run
from gnnid.schema import (DNS_QUERY, L4_FLOW, RPC_CALL, Event,
                          events_to_frame)


def _pcfg(cfg):
    cfg.dotted_set("detector", "ppt")
    return detector_view(cfg)


def _rpc(src, dst, svc):
    return Event(event_id="e1", run_id="r", ts=1.0, event_type=RPC_CALL,
                 src_id=src, src_type="Pod", dst_id=dst, dst_type="Pod",
                 dst_svc_id=svc, protocol="grpc", method="POST",
                 path="/hipstershop.CartService/GetCart", status_code=200,
                 grpc_status=0, response_flags="UT,DC", duration_ms=3.0,
                 request_bytes=10, response_bytes=40, reporter_views=2)


def _fit_fixture_spec(run_dir, cfg):
    pcfg = _pcfg(cfg)
    ev, _, _ = ingest_run(run_dir, pcfg)
    return PPTFeatureSpec.fit([ev], pcfg), ev, pcfg


def test_dims(run_dir, cfg):
    spec, _, _ = _fit_fixture_spec(run_dir, cfg)
    assert spec.event_dim == 269
    assert spec.entity_dim == 17
    assert spec.path_vocab[0] == "<oov>"
    assert "/*/getcart" in spec.path_vocab  # gRPC callee identity dropped


def test_identity_fields_unreachable(run_dir, cfg):
    spec, _, _ = _fit_fixture_spec(run_dir, cfg)
    ev = events_to_frame([_rpc("pod:a/x", "pod:a/y", "svc:a/s"),
                          _rpc("pod:b/q", "pod:b/w", "svc:b/z")])
    rows = list(ev.itertuples(index=False))
    fa = event_features(event_view(rows[0]), 3, 0.4, spec)
    fb = event_features(event_view(rows[1]), 3, 0.4, spec)
    np.testing.assert_array_equal(fa, fb)  # endpoints can never influence x
    # raw rows (which carry src_id/dst_id) are rejected outright
    with pytest.raises(TypeError):
        event_features(rows[0], 3, 0.4, spec)


def test_oov_and_na_slots(run_dir, cfg):
    spec, _, _ = _fit_fixture_spec(run_dir, cfg)
    dns = Event(event_id="e2", run_id="r", ts=2.0, event_type=DNS_QUERY,
                src_id="pod:a/x", src_type="Pod", dst_id="dns:x",
                dst_type="DNSName", dns_query="never-seen.evil.example",
                dns_qtypes="A", dns_rcode="3")
    l4 = Event(event_id="e3", run_id="r", ts=3.0, event_type=L4_FLOW,
               src_id="pod:a/x", src_type="Pod", dst_id="ext:world",
               dst_type="ExternalEndpoint", l4_proto="tcp")  # dst_port null
    rows = list(events_to_frame([dns, l4]).itertuples(index=False))
    fd = event_features(event_view(rows[0]), 0, 0.0, spec)
    fl = event_features(event_view(rows[1]), 1, 0.5, spec)
    assert fd.sum() > 0 and np.isfinite(fd).all()
    assert fl.sum() > 0 and np.isfinite(fl).all()
    # dns: ext class set, unseen eTLD+1 lands on the OOV slot (index 0)
    dns_off = spec.shared_dim + spec.rpc_dim + spec.l4_dim
    ext_slot = dns_off + 10 + 1                     # _QTYPES(10) then ext idx 1
    assert fd[ext_slot] == 1.0
    assert fd[dns_off + 10 + 3] == 1.0              # eTLD+1 vocab OOV slot
    # nxdomain rcode (numeric "3" normalized)
    rc_off = dns_off + 10 + 3 + spec.dns_dim
    assert fd[rc_off + 1] == 1.0                    # _RCODES[1] == nxdomain
    # l4: null dst_port -> port_class 'na' slot active
    l4_off = spec.shared_dim + spec.rpc_dim
    port_block = fl[l4_off + 4:l4_off + 4 + 21]
    assert port_block.sum() == 1.0 and port_block[-1] == 1.0


def test_strip_verdict_zeroes_verdict_blocks(run_dir, cfg):
    spec, _, _ = _fit_fixture_spec(run_dir, cfg)
    l4 = Event(event_id="e4", run_id="r", ts=3.0, event_type=L4_FLOW,
               src_id="pod:a/x", src_type="Pod", dst_id="ext:world",
               dst_type="ExternalEndpoint", l4_proto="tcp", dst_port=9999,
               verdict="DROPPED", drop_reason="POLICY_DENIED", is_reply=False)
    row = next(events_to_frame([l4]).itertuples(index=False))
    keep = event_features(event_view(row), 0, 0.0, spec)
    strip = event_features(event_view(row), 0, 0.0, spec, strip_verdict=True)
    l4_off = spec.shared_dim + spec.rpc_dim
    vd_off = l4_off + 4 + 21
    assert keep[vd_off:vd_off + 9].sum() == 2.0     # dropped + policy_denied
    assert strip[vd_off:vd_off + 9].sum() == 0.0
    # everything outside the verdict blocks is identical
    mask = np.ones_like(keep, dtype=bool)
    mask[vd_off:vd_off + 9] = False
    np.testing.assert_array_equal(keep[mask], strip[mask])


def test_scaler_and_spec_roundtrip(run_dir, cfg, tmp_path):
    spec, _, _ = _fit_fixture_spec(run_dir, cfg)
    assert spec._scale("duration_ms", None) == 0.0
    assert -5.0 <= spec._scale("duration_ms", 1e12) <= 5.0
    spec.save(tmp_path / "spec.json")
    spec2 = PPTFeatureSpec.load(tmp_path / "spec.json")
    assert spec == spec2
    assert spec2.event_dim == spec.event_dim


def test_entity_features_age_only(run_dir, cfg):
    spec, _, _ = _fit_fixture_spec(run_dir, cfg)
    e0, e1 = entity_features(0, spec), entity_features(1, spec)
    assert e0.shape == (17,) and e0[0] == 1.0
    assert not np.array_equal(e0, e1)               # age is encoded
