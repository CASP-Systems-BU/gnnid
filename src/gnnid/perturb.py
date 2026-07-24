"""Synthetic attack perturbations for eval (no real attack data in v1).

Each perturbation mutates a run's EVENTS (not features), so the full
sentences->graph->score pipeline reruns — the test stays end-to-end honest.
Perturbations target a specific pod's windows; eval compares perturbed vs
original scores for that pod.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DNS_QUERY, K8S_API_CALL, POD, RPC_CALL


def _pods_in(events: pd.DataFrame) -> list[str]:
    src = events[events.src_type == POD]["src_id"]
    return sorted(src.unique().tolist())


def rewire(events: pd.DataFrame, rng: np.random.Generator,
           entities: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Retarget a pod's outbound RPCs to a service it never calls."""
    ev = events.copy()
    rpc = ev[(ev.event_type == RPC_CALL) & (ev.src_type == POD)]
    if rpc.empty:
        return ev, ""
    victim = rng.choice(rpc["src_id"].unique())
    svc_ids = sorted(entities[entities.entity_type == "Service"]["entity_id"])
    used = set(rpc[rpc.src_id == victim]["dst_id"])
    candidates = [s for s in svc_ids if s not in used]
    if not candidates:
        return ev, victim
    new_dst = rng.choice(candidates)
    m = (ev.event_type == RPC_CALL) & (ev.src_id == victim)
    ev.loc[m, "dst_id"] = new_dst
    ev.loc[m, "dst_type"] = "Service"
    ev.loc[m, "dst_svc_id"] = new_dst
    return ev, victim


def dns_exfil(events: pd.DataFrame, rng: np.random.Generator,
              entities: pd.DataFrame, k: int = 20,
              domain: str = "exfil-c2-4f9a2.evil-domain.example") -> tuple[pd.DataFrame, str]:
    """Inject DNS queries from a pod to a never-seen external domain."""
    ev = events.copy()
    pods = _pods_in(ev)
    if not pods:
        return ev, ""
    victim = rng.choice(pods)
    base = ev[ev.src_id == victim]
    t0 = float(base["ts"].min()) if not base.empty else float(ev["ts"].min())
    rows = []
    for i in range(k):
        rows.append({
            "event_id": f"perturb:dns:{i}", "run_id": ev["run_id"].iloc[0],
            "ts": t0 + i * 0.1, "event_type": DNS_QUERY,
            "src_id": victim, "src_type": POD,
            "dst_id": f"dns:{domain}", "dst_type": "DNSName",
            "dns_query": domain, "dns_qtypes": "A", "dns_rcode": "0"})
    return pd.concat([ev, pd.DataFrame(rows)], ignore_index=True), victim


def sentence_swap(events: pd.DataFrame, rng: np.random.Generator,
                  entities: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Swap the event sets of two pods (the direct role-mismatch probe)."""
    ev = events.copy()
    pods = _pods_in(ev)
    if len(pods) < 2:
        return ev, ""
    a, b = rng.choice(pods, size=2, replace=False)
    a_mask = ev.src_id == a
    b_mask = ev.src_id == b
    ev.loc[a_mask, "src_id"] = "__TMP__"
    ev.loc[b_mask, "src_id"] = a
    ev.loc[ev.src_id == "__TMP__", "src_id"] = b
    return ev, a


def api_burst(events: pd.DataFrame, rng: np.random.Generator,
              entities: pd.DataFrame, k: int = 15) -> tuple[pd.DataFrame, str]:
    """Inject sensitive K8s API calls (list secrets / create pods) from a pod
    that never touches the API server."""
    ev = events.copy()
    api_pods = set(ev[ev.event_type == K8S_API_CALL]["src_id"])
    pods = [p for p in _pods_in(ev) if p not in api_pods]
    if not pods:
        return ev, ""
    victim = rng.choice(pods)
    from .resolve import KUBEAPI_ID
    t0 = float(ev[ev.src_id == victim]["ts"].min()) if (ev.src_id == victim).any() \
        else float(ev["ts"].min())
    calls = [("list", "secrets", None), ("create", "pods", None),
             ("get", "secrets", None), ("create", "pods", "exec")]
    rows = []
    for i in range(k):
        verb, res, sub = calls[i % len(calls)]
        rows.append({
            "event_id": f"perturb:api:{i}", "run_id": ev["run_id"].iloc[0],
            "ts": t0 + i * 0.1, "event_type": K8S_API_CALL,
            "src_id": victim, "src_type": POD,
            "dst_id": KUBEAPI_ID, "dst_type": "KubeAPI",
            "k8s_verb": verb, "k8s_resource": res, "k8s_subresource": sub,
            "k8s_user_type": "sa", "k8s_status_code": 200})
    return pd.concat([ev, pd.DataFrame(rows)], ignore_index=True), victim


PERTURBATIONS = {
    "rewire": rewire,
    "dns_exfil": dns_exfil,
    "sentence_swap": sentence_swap,
    "api_burst": api_burst,
}
