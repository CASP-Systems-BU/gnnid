"""EntityResolver: raw telemetry identifiers -> canonical entity IDs.

Built per run from the k8s snapshot (objects.json + objects_end.json merged,
nodes.json). Canonical IDs:
    pod:<ns>/<name>      svc:<ns>/<name>      workload:<ns>/<kind>/<name>
    dns:<fqdn>           ext:<class>          kubeapi:cluster
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import DNSNAME, EXTERNAL, KUBEAPI, POD, SERVICE, Entity

KUBEAPI_ID = "kubeapi:cluster"


def _hashish(seg: str) -> bool:
    """A k8s pod-template-hash / ReplicaSet suffix: alnum (no uppercase),
    4-11 chars, containing at least one digit. The digit requirement
    distinguishes a real hash ('649c7444bc', 'mljz4', all-digit '5976767489')
    from a word segment ('client', 'catalog'). Note str.islower() is False for
    all-digit strings, so we compare against .lower() instead."""
    return (4 <= len(seg) <= 11 and seg.isalnum() and seg == seg.lower()
            and any(c.isdigit() for c in seg))


def strip_pod_hash(pod_name: str) -> str:
    """frontend-5976767489-mljz4 -> frontend (ReplicaSet + pod hash suffixes);
    cilium-abcd1 -> cilium (single DaemonSet-style hash); ubuntu-client kept
    (no hash segments)."""
    parts = pod_name.split("-")
    if len(parts) >= 3 and _hashish(parts[-1]) and _hashish(parts[-2]):
        return "-".join(parts[:-2])
    if len(parts) >= 2 and _hashish(parts[-1]):
        return "-".join(parts[:-1])
    return pod_name


def normalize_dns(name: str) -> str:
    return name.lower().rstrip(".")


def parse_upstream_cluster(uc: str) -> tuple[str, str | None]:
    """Envoy UPSTREAM_CLUSTER -> (direction, service_fqdn|None).
    'outbound|7070||cartservice.default.svc.cluster.local' -> ('outbound', fqdn)
    'inbound|8080||' -> ('inbound', None)"""
    parts = (uc or "").split("|")
    if len(parts) >= 4 and parts[0] in ("outbound", "inbound"):
        return parts[0], normalize_dns(parts[3]) if parts[3] else None
    return "unknown", None


def service_from_cluster_fqdn(fqdn: str | None) -> tuple[str, str] | None:
    """'cartservice.default.svc.cluster.local' -> ('default', 'cartservice')."""
    if not fqdn:
        return None
    parts = fqdn.split(".")
    if len(parts) >= 3 and parts[2] == "svc":
        return parts[1], parts[0]
    return None


class EntityResolver:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.entities: dict[str, Entity] = {}
        self.ip_to_pod: dict[str, str] = {}      # pod IP -> pod entity_id
        self.node_ips: dict[str, str] = {}       # node InternalIP -> node name
        self.svc_cluster_ips: dict[str, str] = {}  # clusterIP -> svc entity_id
        self._pod_owner: dict[str, str] = {}     # pod name -> owner (RS/STS/DS) name
        self._rs_owner: dict[str, str] = {}      # RS name -> Deployment name
        self._add(Entity(run_id, KUBEAPI_ID, KUBEAPI, name="kube-apiserver"))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_snapshot(cls, run_id: str, snapshot_dir: str | Path) -> "EntityResolver":
        r = cls(run_id)
        snapshot_dir = Path(snapshot_dir)
        items: list[dict] = []
        for fname in ("objects.json", "objects_end.json"):
            p = snapshot_dir / fname
            if p.exists():
                with open(p) as f:
                    items.extend(json.load(f).get("items", []))
        nodes_p = snapshot_dir / "nodes.json"
        if nodes_p.exists():
            with open(nodes_p) as f:
                for node in json.load(f).get("items", []):
                    name = node["metadata"]["name"]
                    for addr in node.get("status", {}).get("addresses", []):
                        if addr.get("type") == "InternalIP":
                            r.node_ips[addr["address"]] = name
        # two passes: owners first (RS -> Deployment chain), then pods/services
        for it in items:
            kind = it.get("kind")
            meta = it.get("metadata", {})
            if kind == "ReplicaSet":
                for ref in meta.get("ownerReferences", []) or []:
                    if ref.get("kind") == "Deployment":
                        r._rs_owner[meta["name"]] = ref["name"]
        for it in items:
            r._ingest_object(it)
        return r

    def _ingest_object(self, it: dict) -> None:
        kind = it.get("kind")
        meta = it.get("metadata", {})
        ns, name = meta.get("namespace", ""), meta.get("name", "")
        if kind == "Pod":
            eid = f"pod:{ns}/{name}"
            owner = None
            for ref in meta.get("ownerReferences", []) or []:
                owner = self._rs_owner.get(ref.get("name", ""), ref.get("name"))
                break
            labels = meta.get("labels", {}) or {}
            canonical = (labels.get("service.istio.io/canonical-name")
                         or (owner and strip_pod_hash(owner))
                         or strip_pod_hash(name))
            spec, status = it.get("spec", {}), it.get("status", {})
            containers = [c.get("name") for c in spec.get("containers", [])]
            ent = Entity(
                self.run_id, eid, POD, name=name, namespace=ns,
                canonical_service=canonical, workload=owner,
                node_name=spec.get("nodeName"), pod_ip=status.get("podIP"),
                service_account=spec.get("serviceAccountName"),
                uid=meta.get("uid"), has_sidecar="istio-proxy" in containers)
            self._add(ent)
            if status.get("podIP"):
                self.ip_to_pod[status["podIP"]] = eid
        elif kind == "Service":
            eid = f"svc:{ns}/{name}"
            self._add(Entity(self.run_id, eid, SERVICE, name=name, namespace=ns,
                             canonical_service=name, uid=meta.get("uid")))
            cip = it.get("spec", {}).get("clusterIP")
            if cip and cip not in ("None", ""):
                self.svc_cluster_ips[cip] = eid

    def _add(self, ent: Entity) -> None:
        self.entities.setdefault(ent.entity_id, ent)

    # ---------------------------------------------------------------- resolve
    def resolve_ip(self, ip: str | None) -> tuple[str, str] | None:
        """IP -> (entity_id, entity_type). Pods first, then service ClusterIPs,
        then node IPs (host-network -> external class), else external."""
        if not ip:
            return None
        if ip in self.ip_to_pod:
            return self.ip_to_pod[ip], POD
        if ip in self.svc_cluster_ips:
            return self.svc_cluster_ips[ip], SERVICE
        if ip in self.node_ips:
            return self.ext_entity("host"), EXTERNAL
        return self.ext_entity("world"), EXTERNAL

    def pod_by_name(self, ns: str, name: str) -> str | None:
        eid = f"pod:{ns}/{name}"
        return eid if eid in self.entities else None

    def service_entity(self, ns: str, name: str) -> str:
        eid = f"svc:{ns}/{name}"
        if eid not in self.entities:
            self._add(Entity(self.run_id, eid, SERVICE, name=name, namespace=ns,
                             canonical_service=name))
        return eid

    def dns_entity(self, fqdn: str) -> str:
        fqdn = normalize_dns(fqdn)
        eid = f"dns:{fqdn}"
        if eid not in self.entities:
            self._add(Entity(self.run_id, eid, DNSNAME, name=fqdn))
        return eid

    def ext_entity(self, cls: str) -> str:
        eid = f"ext:{cls}"
        if eid not in self.entities:
            self._add(Entity(self.run_id, eid, EXTERNAL, name=cls))
        return eid

    def resolve_hubble_endpoint(self, ep: dict | None) -> tuple[str, str] | None:
        """Hubble flow endpoint dict -> (entity_id, entity_type)."""
        if not isinstance(ep, dict):
            return None
        pod, ns = ep.get("pod_name"), ep.get("namespace")
        if pod and ns:
            eid = self.pod_by_name(ns, pod)
            if eid:
                return eid, POD
            # pod not in snapshot (churned) -> register from the flow itself
            wls = ep.get("workloads") or []
            canonical = (wls[0].get("name") if wls and wls[0].get("name")
                         else strip_pod_hash(pod))
            eid = f"pod:{ns}/{pod}"
            self._add(Entity(self.run_id, eid, POD, name=pod, namespace=ns,
                             canonical_service=canonical))
            return eid, POD
        for lbl in ep.get("labels") or []:
            if lbl.startswith("reserved:"):
                cls = lbl.split(":", 1)[1]
                if cls in ("host", "remote-node", "kube-apiserver"):
                    return self.ext_entity(cls), EXTERNAL
                return self.ext_entity("world"), EXTERNAL
        return self.resolve_ip(ep.get("ip") or (ep.get("IP") or {}).get("source"))

    def entity_type(self, eid: str) -> str | None:
        e = self.entities.get(eid)
        return e.entity_type if e else None
