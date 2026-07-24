"""End-to-end schema-contract smoke + overfit-one-window."""
import pytest
import torch

from gnnid import windows
from gnnid.dataset import entity_meta_map
from gnnid.embed import SentenceEmbedder
from gnnid.graph import build_window_graph
from gnnid.ingest.run_dir import ingest_run
from gnnid.labels import LabelVocab
from gnnid.models.objectives import build_objective
from gnnid.models.sage import FlashSAGE
from gnnid.sentences import build_sentences


def _build_graph(run_dir, cfg, epochs=2):
    ev, ent, _ = ingest_run(run_dir, cfg)
    meta = entity_meta_map(ent)
    win = list(windows.iter_windows(ev, 30, 15, 15, "run"))[0]
    sents = build_sentences(win.events, set(meta), cfg)
    cfg.dotted_set("w2v.epochs", epochs)
    cfg.dotted_set("w2v.min_count", 1)
    emb = SentenceEmbedder.train([v for v in sents.values() if v], cfg)
    vocab = LabelVocab.fit(ent)
    g = build_window_graph(win, meta, emb.encode_many(sents), vocab, cfg)
    return g, vocab


def test_e2e_smoke(run_dir, cfg):
    g, vocab = _build_graph(run_dir, cfg)
    assert g is not None
    assert g.data.num_nodes > 0
    assert g.data.x.shape[1] == cfg.w2v.dim
    assert torch.isfinite(g.data.x).all()
    # model forward produces finite logits + embedding
    model = FlashSAGE(cfg.w2v.dim, cfg.model.hidden, cfg.model.embed,
                      vocab.num_classes, cfg.model.dropout)
    model.eval()
    logits, emb = model(g.data.x, g.data.edge_index)
    assert logits.shape == (g.data.num_nodes, vocab.num_classes)
    assert torch.isfinite(logits).all() and torch.isfinite(emb).all()


def test_scores_finite(run_dir, cfg):
    g, vocab = _build_graph(run_dir, cfg)
    model = FlashSAGE(cfg.w2v.dim, cfg.model.hidden, cfg.model.embed,
                      vocab.num_classes, cfg.model.dropout)
    obj = build_objective("flash_cls", class_weights=None,
                          other_idx=vocab.other_idx)
    logits, _ = model(g.data.x, g.data.edge_index)
    scores = obj.node_scores(logits, g.data)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()


@pytest.mark.slow
def test_overfit_one_window(run_dir, cfg):
    g, vocab = _build_graph(run_dir, cfg, epochs=5)
    torch.manual_seed(0)
    model = FlashSAGE(cfg.w2v.dim, 64, 32, vocab.num_classes, dropout=0.0)
    obj = build_objective("flash_cls", class_weights=None,
                          other_idx=vocab.other_idx)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    y = g.data.y
    mask = y != vocab.other_idx
    if mask.sum() < 2:
        pytest.skip("fixture window has too few labeled nodes to overfit")
    for _ in range(300):
        model.train()
        opt.zero_grad()
        logits, _ = model(g.data.x, g.data.edge_index)
        loss = obj.loss(logits, g.data)
        loss.backward()
        opt.step()
    model.eval()
    logits, _ = model(g.data.x, g.data.edge_index)
    pred = logits.argmax(1)
    acc = (pred[mask] == y[mask]).float().mean().item()
    assert acc >= 0.99, f"failed to overfit one window (acc={acc})"
