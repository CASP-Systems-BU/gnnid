"""Sentence grammar, normalization, dedup/caps, and the LEAKAGE firewall."""
import re as _re

from gnnid import windows
from gnnid.dataset import entity_meta_map
from gnnid.ingest.run_dir import ingest_run
from gnnid.sentences import (build_sentences, dns_dst_token, port_class,
                             status_class, template_path)


def test_path_templating():
    assert template_path("/product/OLJCESPC7Z", "http") == "/product/*"
    assert template_path("/cart?foo=1", "http") == "/cart"
    assert template_path("/api/v1/12345/detail", "http") == "/api/v1/*/detail"
    # gRPC: package.Service dropped, method kept (prevents callee-identity leak)
    assert template_path("/hipstershop.CartService/GetCart", "grpc") == "/*/getcart"


def test_port_and_status_classes():
    assert port_class(7070) == "grpc"
    assert port_class(443) == "https"
    assert port_class(9999) == "registered"
    assert port_class(50000) == "ephemeral"
    assert port_class(None) == "na"
    assert status_class(200) == "2xx"
    assert status_class(503.0) == "5xx"   # float from parquet round-trip
    assert status_class(None) == "na"


def test_dns_dst_token():
    assert dns_dst_token("cart.default.svc.cluster.local") == "dst:cluster"
    assert dns_dst_token("api.googleapis.com") == "dst:ext:googleapis.com"
    assert dns_dst_token("exfil.evil-domain.example") == "dst:ext:evil-domain.example"


def _win_and_meta(run_dir, cfg):
    ev, ent, _ = ingest_run(run_dir, cfg)
    meta = entity_meta_map(ent)
    wins = list(windows.iter_windows(ev, 30, 15, 15, "run"))
    return wins[0], meta, ent


def test_sentences_built_and_ordered(run_dir, cfg):
    win, meta, _ = _win_and_meta(run_dir, cfg)
    sents = build_sentences(win.events, set(meta), cfg)
    fe = next(k for k in sents if "frontend" in k and k.startswith("pod:"))
    assert sents[fe], "frontend should have a non-empty sentence"
    # tokens are namespaced
    assert all(":" in t for t in sents[fe])


# Identity strings that must never appear as a standalone token COMPONENT of a
# node's own sentence. Component match (not substring): a served method name
# like `getcart` is behavior and legitimately contains 'cart' — only the exact
# label/name/hash components are forbidden (the plan's spec).
LEAK_TERMS = {
    "pod:default/frontend-5976767489-mljz4": ["frontend", "5976767489", "mljz4"],
    "pod:default/cart-649c7444bc-2zqqw": ["cart", "649c7444bc", "2zqqw"],
}


def _components(tok: str) -> set[str]:
    return set(p for p in _re.split(r"[^a-z0-9.]+", tok.lower()) if p)


def test_no_identity_leakage(run_dir, cfg):
    """A node's own name/workload/service/hash must never appear as a token
    component (exact match, not substring)."""
    win, meta, _ = _win_and_meta(run_dir, cfg)
    sents = build_sentences(win.events, set(meta), cfg)
    for eid, terms in LEAK_TERMS.items():
        toks = sents.get(eid, [])
        comps = set().union(*(_components(t) for t in toks)) if toks else set()
        for term in terms:
            assert term not in comps, \
                f"LEAK: {eid} sentence has identity component {term!r}: {toks}"


def test_dedup_caps_repeats(cfg):
    from gnnid.sentences import _dedup_and_cap
    items = [(float(i), ("rpc:out", "m:get")) for i in range(100)]
    out = _dedup_and_cap(items, max_repeats=3, max_tokens=256)
    # 100 identical events collapse to <=3 occurrences (each 2 tokens) => <=6
    assert len(out) <= 6


def test_max_tokens_cap(cfg):
    from gnnid.sentences import _dedup_and_cap
    items = [(float(i), (f"p:/x{i}",)) for i in range(1000)]  # all unique
    out = _dedup_and_cap(items, max_repeats=3, max_tokens=256)
    assert len(out) == 256


def test_strip_verdict_removes_v_tokens(run_dir, cfg):
    win, meta, _ = _win_and_meta(run_dir, cfg)
    with_v = build_sentences(win.events, set(meta), cfg, strip_verdict=False)
    without = build_sentences(win.events, set(meta), cfg, strip_verdict=True)
    all_with = [t for toks in with_v.values() for t in toks]
    all_without = [t for toks in without.values() for t in toks]
    assert any(t.startswith("v:") for t in all_with)
    assert not any(t.startswith("v:") for t in all_without)
