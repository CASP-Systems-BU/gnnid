"""Parser + entity-resolution + merge/selection-rule tests."""
from gnnid.ingest.run_dir import ingest_run
from gnnid.resolve import (EntityResolver, parse_upstream_cluster,
                           service_from_cluster_fqdn, strip_pod_hash)
from gnnid.schema import DNS_QUERY, K8S_API_CALL, L4_FLOW, RPC_CALL


def test_strip_pod_hash():
    assert strip_pod_hash("frontend-5976767489-mljz4") == "frontend"
    assert strip_pod_hash("cart-649c7444bc-2zqqw") == "cart"
    assert strip_pod_hash("ubuntu-client") == "ubuntu-client"


def test_parse_upstream_cluster():
    assert parse_upstream_cluster(
        "outbound|7070||cart.default.svc.cluster.local") == (
        "outbound", "cart.default.svc.cluster.local")
    assert parse_upstream_cluster("inbound|8080||") == ("inbound", None)
    assert service_from_cluster_fqdn("cart.default.svc.cluster.local") == (
        "default", "cart")


def test_resolver_from_snapshot(run_dir):
    r = EntityResolver.from_snapshot("run", run_dir / "k8s_snapshot")
    fe = r.pod_by_name("default", "frontend-5976767489-mljz4")
    assert r.entities[fe].canonical_service == "frontend"
    # pod IP resolves to the pod
    assert r.resolve_ip("10.244.2.20") == ("pod:default/cart-649c7444bc-2zqqw", "Pod")
    # sidecar-less client IP resolves too
    assert r.resolve_ip("10.244.4.221")[1] == "Pod"


def test_ingest_event_counts(run_dir, cfg):
    ev, ent, meta = ingest_run(run_dir, cfg)
    by = ev["event_type"].value_counts().to_dict()
    # RPC: client->frontend (inbound-only) + frontend->cart (merged pair) = 2
    assert by.get(RPC_CALL) == 2
    # merged pair has reporter_views==2
    rpc = ev[ev.event_type == RPC_CALL]
    assert set(rpc["reporter_views"].dropna()) == {1.0, 2.0}
    # L4: SYN kept, mid-stream ACK dropped, DROPPED kept => 2
    assert by.get(L4_FLOW) == 2
    # DNS: one response
    assert by.get(DNS_QUERY) == 1
    # audit: cart SA list configmaps kept, node control-plane dropped => 1
    assert by.get(K8S_API_CALL) == 1


def test_rpc_merge_direction(run_dir, cfg):
    ev, _, _ = ingest_run(run_dir, cfg)
    merged = ev[(ev.event_type == RPC_CALL) & (ev.reporter_views == 2)].iloc[0]
    assert merged.src_id.endswith("frontend-5976767489-mljz4")
    assert merged.dst_id.endswith("cart-649c7444bc-2zqqw")


def test_l4_selection_drops_midstream_ack(run_dir, cfg):
    ev, _, _ = ingest_run(run_dir, cfg)
    l4 = ev[ev.event_type == L4_FLOW]
    # the only kept FORWARDED flow is the SYN (port 7070); the ACK is dropped
    fwd = l4[l4.verdict == "FORWARDED"]
    assert len(fwd) == 1 and int(fwd.iloc[0].dst_port) == 7070
    assert (l4.verdict == "DROPPED").sum() == 1


def test_audit_drops_control_plane(run_dir, cfg):
    ev, _, _ = ingest_run(run_dir, cfg)
    api = ev[ev.event_type == K8S_API_CALL]
    assert len(api) == 1
    assert api.iloc[0].k8s_resource == "configmaps"
    assert api.iloc[0].src_id.endswith("cart-649c7444bc-2zqqw")
