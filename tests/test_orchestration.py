"""Characterization test: the full run_training -> run_scoring -> run_eval path.

Written BEFORE the detector-seam refactor; must pass unmodified at every
migration step (the orchestration layer's regression net).
"""
from gnnid.eval import run_eval
from gnnid.score import run_scoring
from gnnid.train import run_training

SPEED = [
    ("w2v.epochs", 2), ("w2v.min_count", 1), ("w2v.workers", 1),
    ("train.max_epochs", 3), ("train.patience", 2),
    ("train.xgb.n_estimators", 5), ("train.xgb.early_stopping_rounds", 2),
]

SCORE_COLUMNS = {"run_id", "w_idx", "entity_id", "node_type", "true_label",
                 "pred_label", "p_true", "score_raw", "score_norm"}


def test_train_score_eval_e2e(parquet_runs, cfg, tmp_path):
    parquet_dir, run_ids = parquet_runs
    for k, v in SPEED:
        cfg.dotted_set(k, v)
    cfg.dotted_set("data.parquet_dir", str(parquet_dir))
    cfg.dotted_set("artifacts_dir", str(tmp_path / "artifacts"))

    trained = run_training(cfg, repo_root=tmp_path)
    adir = tmp_path / "artifacts"
    for f in ("w2v.kv", "label_vocab.json", "gnn.pt", "norm.json",
              "thresholds.json", "manifest.json", "xgb.json"):
        assert (adir / f).exists(), f
    assert trained.manifest["splits"]["train"] == run_ids[:1]
    assert trained.manifest["splits"]["test"] == run_ids[2:]

    scores = run_scoring(cfg, repo_root=tmp_path)
    assert not scores.empty
    assert SCORE_COLUMNS | {"threshold", "is_alert"} <= set(scores.columns)
    assert scores["score_norm"].between(0, 1).all()
    assert set(scores["run_id"]) == set(run_ids[2:])
    results = tmp_path / "results_eval"
    assert (results / "scores.parquet").exists()
    assert (results / "alerts.json").exists()

    report = run_eval(cfg, repo_root=tmp_path)
    assert (results / "eval_report.json").exists()
    assert report["test_runs"] == run_ids[2:]
    assert set(report["benign_fp"]) == {"node_windows", "misclassification_rate",
                                        "alert_rate"}
    assert set(report["weak_signal"]) == {"with_verdict", "strip_verdict"}
    for section in report["weak_signal"].values():
        assert {"auc", "n_pos", "n_neg"} <= set(section)
    assert set(report["perturbations"]) == {"rewire", "dns_exfil",
                                            "sentence_swap", "api_burst"}
    for section in report["perturbations"].values():
        assert {"auc", "detection_rate", "runs"} <= set(section)
    assert set(report["ablation"]) == {"w2v", "gnn", "xgb"}


def test_train_score_eval_e2e_ppt(parquet_runs, cfg, tmp_path):
    parquet_dir, run_ids = parquet_runs
    cfg.dotted_set("detector", "ppt")
    for k, v in [("detectors.ppt.train.pretrain.epochs", 2),
                 ("detectors.ppt.train.pretrain.patience", 1),
                 ("detectors.ppt.train.finetune.max_epochs", 2),
                 ("detectors.ppt.train.finetune.patience", 1)]:
        cfg.dotted_set(k, v)
    cfg.dotted_set("data.parquet_dir", str(parquet_dir))

    trained = run_training(cfg, repo_root=tmp_path)
    adir = tmp_path / "artifacts" / "ppt"          # ppt overlay dir, no clobber
    for f in ("label_vocab.json", "feature_spec.json", "ppt.pt",
              "pretrained.pt", "norm.json", "thresholds.json",
              "manifest.json", "detector.json"):
        assert (adir / f).exists(), f
    assert trained.manifest["splits"]["test"] == run_ids[2:]

    scores = run_scoring(cfg, repo_root=tmp_path)
    assert not scores.empty
    assert SCORE_COLUMNS | {"threshold", "is_alert"} <= set(scores.columns)
    assert scores["score_norm"].between(0, 1).all()
    results = tmp_path / "results_eval" / "ppt"    # ppt overlay results dir
    assert (results / "scores.parquet").exists()
    assert (results / "alerts.json").exists()

    report = run_eval(cfg, repo_root=tmp_path)
    assert (results / "eval_report.json").exists()
    assert set(report["weak_signal"]) == {"with_verdict", "strip_verdict"}
    assert set(report["perturbations"]) == {"rewire", "dns_exfil",
                                            "sentence_swap", "api_burst"}
    assert report["ablation"] == {}                # ppt declares no score forks
