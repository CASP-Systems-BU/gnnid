"""The detector config seam: detectors.<name> overlays deep-merged over base."""
from gnnid.config import Config, detector_view, load_config


def test_flash_view_equals_base(cfg):
    view = detector_view(cfg)
    assert dict(view) == dict(cfg)


def test_ppt_overlay_wins_base_inherited(cfg):
    cfg.dotted_set("detector", "ppt")
    view = detector_view(cfg)
    # overlay wins
    assert view.windows.width_s == 5
    assert view.windows.memory == 5
    assert view.artifacts_dir == "artifacts/ppt"
    assert view.results_dir == "results_eval/ppt"
    assert view.model.objective == "ppt_cls"
    # dicts merge: untouched keys inherit from base
    assert view.graph.reverse_edges is True
    assert view.graph.flow_memory == 20
    assert view.w2v.dim == cfg.w2v.dim
    assert view.scoring.threshold_q == 0.995
    # base config is not mutated
    assert cfg.windows.width_s == 30
    assert cfg.artifacts_dir == "artifacts/default"


def test_set_overrides_flow_through():
    cfg = load_config("configs/default.yaml",
                      ["detector=ppt", "detectors.ppt.train.pretrain.epochs=5",
                       "train.lr=0.0005"])
    view = detector_view(cfg)
    assert view.train.pretrain.epochs == 5
    assert view.train.lr == 0.0005        # base key visible through the view


def test_deep_merge_replaces_lists_and_scalars():
    cfg = Config({"detector": "d",
                  "events": {"l4_keep": ["a", "b"], "flag": True},
                  "detectors": {"d": {"events": {"l4_keep": ["c"]}}}})
    view = detector_view(cfg)
    assert view.events.l4_keep == ["c"]   # lists replace, never concat
    assert view.events.flag is True       # sibling scalar inherited
    # mutating the view never leaks into the base overlay
    view.dotted_set("events.l4_keep", ["x"])
    assert cfg["detectors"]["d"]["events"]["l4_keep"] == ["c"]


def test_missing_detector_defaults_to_flash():
    cfg = Config({"windows": {"width_s": 30}})
    view = detector_view(cfg)             # no detector/detectors keys at all
    assert dict(view) == dict(cfg)
