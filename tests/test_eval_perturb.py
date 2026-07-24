"""Fast unit tests for eval helpers + perturbation generators (no trained model)."""
import numpy as np

from gnnid.eval import _auc, benign_fp
from gnnid.ingest.run_dir import ingest_run
from gnnid.perturb import PERTURBATIONS
from gnnid.schema import DNS_QUERY, K8S_API_CALL, RPC_CALL
import pandas as pd


def test_auc_perfect_separation():
    assert _auc(np.array([1.0, 1.0, 0.9]), np.array([0.1, 0.2, 0.0])) == 1.0
    assert _auc(np.array([0.0]), np.array([1.0])) == 0.0
    assert abs(_auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) - 0.5) < 1e-9


def test_benign_fp_empty():
    assert benign_fp(pd.DataFrame()) == {}


def test_perturbations_mutate_events(run_dir, cfg):
    ev, ent, _ = ingest_run(run_dir, cfg)
    rng = np.random.default_rng(0)

    pert, victim = PERTURBATIONS["dns_exfil"](ev, rng, ent)
    assert victim and len(pert) > len(ev)
    added = pert[pert.event_id.astype(str).str.startswith("perturb:dns")]
    assert (added.event_type == DNS_QUERY).all()
    assert added.iloc[0].dst_id.startswith("dns:")

    pert, victim = PERTURBATIONS["api_burst"](ev, rng, ent)
    if victim:  # only if a non-API pod exists
        added = pert[pert.event_id.astype(str).str.startswith("perturb:api")]
        assert (added.event_type == K8S_API_CALL).all()
        assert set(added.k8s_resource) & {"secrets", "pods"}


def test_rewire_changes_destination(run_dir, cfg):
    ev, ent, _ = ingest_run(run_dir, cfg)
    rng = np.random.default_rng(1)
    pert, victim = PERTURBATIONS["rewire"](ev, rng, ent)
    if victim:
        orig_dsts = set(ev[(ev.event_type == RPC_CALL) & (ev.src_id == victim)]["dst_id"])
        new_dsts = set(pert[(pert.event_type == RPC_CALL) & (pert.src_id == victim)]["dst_id"])
        assert new_dsts != orig_dsts or not orig_dsts
