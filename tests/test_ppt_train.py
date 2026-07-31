"""PPT model + training: forward shapes, pretrain->finetune, overfit gate."""
import pytest
import torch

from gnnid.config import detector_view
from gnnid.dataset import entity_meta_map
from gnnid.detectors.ppt.features import PPTFeatureSpec
from gnnid.detectors.ppt.graph import build_memory_graphs
from gnnid.detectors.ppt.model import PPTEncoder, PPTModel
from gnnid.detectors.ppt.train import finetune, pretrain
from gnnid.ingest.run_dir import ingest_run
from gnnid.labels import LabelVocab
from gnnid.models.objectives import build_objective


def _fixture_graphs(run_dir, cfg):
    cfg.dotted_set("detector", "ppt")
    pcfg = detector_view(cfg)
    ev, ent, _ = ingest_run(run_dir, pcfg)
    meta = entity_meta_map(ent)
    vocab = LabelVocab.fit(ent)
    spec = PPTFeatureSpec.fit([ev], pcfg)
    graphs = list(build_memory_graphs(ev, meta, vocab, spec, pcfg, "r"))
    return graphs, vocab, spec, pcfg


def test_forward_shapes_and_scores(run_dir, cfg):
    graphs, vocab, spec, _ = _fixture_graphs(run_dir, cfg)
    g = graphs[0].data
    model = PPTModel(spec.entity_dim, spec.event_dim, 64, 2,
                     vocab.num_classes, 0.2)
    model.eval()
    logits, emb = model(g)
    n_ent = g["entity"].x.shape[0]
    assert logits.shape == (n_ent, vocab.num_classes)
    assert emb.shape == (n_ent, 64)
    assert torch.isfinite(logits).all()
    obj = build_objective("ppt_cls", class_weights=None,
                          other_idx=vocab.other_idx)
    assert torch.isfinite(obj.loss(logits, g))
    scores = obj.node_scores(logits, g)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_pretrain_returns_encoder_state(run_dir, cfg):
    torch.manual_seed(0)
    graphs, vocab, spec, pcfg = _fixture_graphs(run_dir, cfg)
    pcfg.dotted_set("train.pretrain.epochs", 2)
    pcfg.dotted_set("train.pretrain.patience", 2)
    state = pretrain(graphs, graphs, spec, pcfg)
    fresh = PPTEncoder(spec.entity_dim, spec.event_dim,
                       int(pcfg.model.hidden), int(pcfg.model.layers),
                       float(pcfg.model.dropout))
    assert set(state) == set(fresh.state_dict())
    assert all(torch.isfinite(v).all() for v in state.values())


def test_pretrain_disabled_returns_none(run_dir, cfg):
    graphs, vocab, spec, pcfg = _fixture_graphs(run_dir, cfg)
    pcfg.dotted_set("train.pretrain.enabled", False)
    assert pretrain(graphs, graphs, spec, pcfg) is None


def test_encoder_transfer_into_finetune(run_dir, cfg):
    torch.manual_seed(0)
    graphs, vocab, spec, pcfg = _fixture_graphs(run_dir, cfg)
    pcfg.dotted_set("train.pretrain.epochs", 1)
    state = pretrain(graphs, graphs, spec, pcfg)
    pcfg.dotted_set("train.finetune.max_epochs", 0)   # transfer only
    model = finetune(state, graphs, graphs, vocab, spec, pcfg)
    for k, v in state.items():
        assert torch.equal(v, model.encoder.state_dict()[k]), k


def test_finetune_smoke(run_dir, cfg):
    torch.manual_seed(0)
    graphs, vocab, spec, pcfg = _fixture_graphs(run_dir, cfg)
    pcfg.dotted_set("train.finetune.max_epochs", 2)
    pcfg.dotted_set("train.finetune.patience", 1)
    model = finetune(None, graphs, graphs, vocab, spec, pcfg)
    logits, _ = model(graphs[0].data)
    assert torch.isfinite(logits).all()


@pytest.mark.slow
def test_overfit_one_memory_graph(run_dir, cfg):
    graphs, vocab, spec, _ = _fixture_graphs(run_dir, cfg)
    g = graphs[0].data
    ent = g["entity"]
    mask = (ent.y != vocab.other_idx) & ent.score_mask
    if mask.sum() < 2:
        pytest.skip("fixture memory graph has too few labeled entities")
    torch.manual_seed(0)
    model = PPTModel(spec.entity_dim, spec.event_dim, 64, 2,
                     vocab.num_classes, dropout=0.0)
    obj = build_objective("ppt_cls", class_weights=None,
                          other_idx=vocab.other_idx)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        model.train()
        opt.zero_grad()
        logits, _ = model(g)
        loss = obj.loss(logits, g)
        loss.backward()
        opt.step()
    model.eval()
    logits, _ = model(g)
    acc = (logits.argmax(1)[mask] == ent.y[mask]).float().mean().item()
    assert acc >= 0.99, f"failed to overfit one memory graph (acc={acc})"
