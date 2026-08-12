# Data reference: what we collect, where it comes from, what it means

This documents the telemetry the pipeline collects from a live cluster and the
two tables gnnid compressed it into (`events.parquet`, `entities.parquet`).
Flow: **ubench run → run directory (raw telemetry) → `gnnid ingest` →
`data/parquet/<run_id>/{events,entities}.parquet`**.

- One **run directory** = one time segment of a (possibly much longer) continuous load: e.g. a 1-hour experiment with `segment_s: 600` produces six directories, each a contiguous 10-minute slice.
- One **events.parquet** = every telemetry event observed in that segment, from all sources, in a single wide table
- One **entities.parquet** = every entity (pod, service, …) that existed in the segment: the graph-node candidates.

The authoritative schema is [src/gnnid/schema.py](src/gnnid/schema.py); the
run-dir layout contract is [src/gnnid/ingest/run_dir.py](src/gnnid/ingest/run_dir.py).

---

## 1. The run directory (raw telemetry)

Produced on the control node by `ubench/scripts/run_and_collect.sh`, copied
back to `ubench/results/<benchmark>-<request>_<UTC-timestamp>/`.

| File | Source | What it is | Ingested? |
|---|---|---|---|
| `meta.json` | run_and_collect.sh | Run parameters + time window (below) | ✔ (keys: `benchmark`, `request`, `run_id`, `run_status`) |
| `k8s_snapshot/objects.json` | `kubectl get pods,services,endpoints,deployments,replicasets,statefulsets,daemonsets -A` at segment start | Entity inventory: pod IPs, labels, ownerReferences chains, service ClusterIPs, node placement | ✔ (resolver input) |
| `k8s_snapshot/objects_end.json` | same query at collection time | Catches pods that churned mid-segment | ✔ (merged with the above) |
| `k8s_snapshot/nodes.json` | `kubectl get nodes` | Node objects (InternalIPs → node-name mapping) | ✔ |
| `istio/access_logs/<pod>.log` | Envoy sidecar access logs, read from the rotated CRI files on each worker (`kubectl logs` alone loses rotated data), window-filtered | One line per HTTP/gRPC request seen by that pod's sidecar (JSON encoding) | ✔ → `RPC_CALL` |
| `cilium/flows.jsonl` | `hubble observe -f` streamed live for the whole experiment, sliced per segment by flow timestamp | Every L3/L4 network flow observation (one JSON object per line) | ✔ → `L4_FLOW` (+ `DNS_QUERY` if L7 DNS visibility is on) |
| `audit/audit.jsonl` | kube-apiserver audit log (`Metadata` level policy), window-filtered | Every Kubernetes API request (who asked the API server for what) | ✔ → `K8S_API_CALL` |
| `resources.csv` | `kubectl top pods` sampled every 5s (metrics-server) | Per-pod CPU/mem time series: `epoch,pod,cpu,mem` | ✘ (ops/debugging) |
| `run.log` | tee of the run scripts | Setup output + full wrk2 output | ✘ |
| `wrk.txt` | sliced from run.log | wrk2 stats block: throughput + coordinated-omission-corrected HdrHistogram latency spectrum (whole-experiment stats, duplicated into every segment) | ✘ |
| `istio/requests_total.json`, `request_duration_ms.json`, `request_bytes.json`, `response_bytes.json`, `tcp_sent_bytes.json`, `tcp_received_bytes.json` | in-cluster Prometheus (`istio_*` metrics), 15s-step range queries over the segment window, `reporter="destination"` | Aggregated per-service L7 counters/histograms | ✘ (aggregates; the per-request truth is the access logs) |
| `istio/edges.json`, `istio/summary.json` | Prometheus | L7 call graph (src→dst request counts) and per-service rate/error/latency summary | ✘ |
| `cilium/edges.json`, `cilium/dns.jsonl`, `cilium/hubble_metrics.prom` | Hubble | L4 call graph with verdicts/ports; DNS flows sliced out; Hubble Prometheus snapshot | ✘ |
| `experiment.json` | copy of the renderer's `resolved.json` | Full resolved experiment spec: workers, replica map, placement, rate, seed, provenance for this segment | ✘ (provenance) |

### meta.json keys

| Key | Meaning |
|---|---|
| `benchmark`, `request` | workload (`boutique`, …) and request mix (`mix` = the Lua script name) |
| `threads`, `connections` | wrk2 `-t` / `-c` |
| `duration_s` | **this segment's** length (seconds) |
| `run_id` | segment start timestamp `YYYYMMDD-HHMMSS` (UTC): the temporal sort key |
| `start_epoch` / `end_epoch`, `start_iso` / `end_iso` | segment window; windows of consecutive segments tile exactly (`end_epoch(k) == start_epoch(k+1)`) |
| `istio_enabled`, `cilium_enabled`, `audit_enabled` | which telemetry planes were live |
| `run_status` | 0 = valid; anything else ⇒ `gnnid ingest` skips the segment |
| `generator` | `"wrk2"` |
| `rate_rps` / `achieved_rps` | offered vs measured request rate; a gap means the rate exceeded the cluster's saturation knee and the run measured backlog, not services |
| `experiment` | the spec name (`experiments/<name>.yaml`) |
| `workers` | worker-node count the placement spanned |
| `segment_index` / `segments_total` / `total_duration_s` | where this slice sits in the continuous experiment |

---

## 2. events.parquet

One row per telemetry event. **Identity/context columns** are always present;
each `event_type` fills only its own payload columns (the rest stay null).

### Identity & context (all rows)

| Column | Meaning |
|---|---|
| `event_id` | Deterministic provenance ID: `<run_id>:<file>:<lineno>`: you can always open the raw line that produced a row |
| `run_id` | `<benchmark>-<request>_<timestamp>` (the run-dir name) |
| `ts` | Event time, epoch seconds (float) |
| `event_type` | `RPC_CALL` \| `L4_FLOW` \| `DNS_QUERY` \| `K8S_API_CALL` |
| `src_id`, `src_type` | Canonical source entity (see §3 for the ID grammar) and its type |
| `dst_id`, `dst_type` | Canonical destination entity: the most specific endpoint known (server pod if resolvable) |
| `dst_svc_id` | RPC only: the Service in front of `dst` (the graph also adds a src→Service edge); null otherwise |
| `label`, `label_source` | Reserved for attack ground truth (filled by `ingest --labels`); null on benign runs |

### RPC_CALL: from `istio/access_logs/<pod>.log` (Envoy, L7)

One row per application request/response observed by the mesh. A
mesh-internal call appears in **two** sidecars' logs (client outbound view +
server inbound view); ingest merges the pair on `x-request-id` with the
server view authoritative, recorded as `reporter_views=2`. `reporter_views=1`
means only one side saw it (e.g. the un-meshed load generator calling in, or
a pod calling out of the mesh).

| Column | Meaning |
|---|---|
| `protocol` | `http` \| `grpc` |
| `method`, `path` | HTTP verb and raw request path (path templating happens later, in sentence building) |
| `status_code` | HTTP response code |
| `grpc_status` | gRPC status if the call was gRPC (null for plain HTTP) |
| `response_flags` | Envoy's response flags (`-` = none; e.g. `UT` upstream timeout, `UF` upstream connect failure, `NR` no route) — how the request *failed*, when it did |
| `duration_ms` | End-to-end request duration as measured by the reporting sidecar |
| `request_bytes`, `response_bytes` | Body sizes |
| `reporter_views` | 1 or 2 (see merge rule above) |

Read a row as: *`src_id` sent `method path` to `dst_id` (behind `dst_svc_id`),
which answered `status_code` in `duration_ms` ms with `response_bytes` bytes.*

### L4_FLOW: from `cilium/flows.jsonl` (Hubble, L3/L4)

The network-plane counterpart: socket-level flows that no sidecar can see,
including traffic from/to un-meshed pods and **denied/dropped packets**.
Hubble emits many observations per connection, so ingest keeps only
informative ones (config `events.l4_keep`, default all three rules):

- `non_forwarded` — any verdict ≠ `FORWARDED` (drops, policy denials) — always kept;
- `syn_no_ack` — TCP SYN without ACK = a connection being opened;
- `udp_first` — first packet of a UDP/ICMP exchange (`is_reply=false`).

So one row ≈ one connection attempt or one anomaly, not one packet.

| Column | Meaning |
|---|---|
| `l4_proto` | `tcp` \| `udp` \| `icmp` |
| `dst_port` | Destination port (the service being contacted) |
| `verdict` | `FORWARDED`, `DROPPED`, `POLICY_DENIED`, … — non-FORWARDED rows are the strongest weak-label signal we have (eval uses them as a proxy) |
| `drop_reason` | Cilium's reason when dropped, else null |
| `is_reply` | Direction hint: was this packet part of the reply path? |

### K8S_API_CALL: from `audit/audit.jsonl` (kube-apiserver audit)

The control-plane view: every request a pod made to the Kubernetes API
(watches, gets, creates…). `src` is the calling pod resolved from the audit
event's `sourceIPs[0]`; `dst` is always the `kubeapi:cluster` singleton.
Events not attributable to a pod (kubelet, control-plane components on host
IPs) are dropped by default (`events.drop_control_plane_audit: true`) — noise
a per-pod detector cannot act on. The audit policy is `Metadata` level:
we record *who did what to which resource*, never request bodies.

| Column | Meaning |
|---|---|
| `k8s_verb` | `get`, `list`, `watch`, `create`, `update`, `patch`, `delete`, … |
| `k8s_resource`, `k8s_subresource` | e.g. `pods` + `exec` — the pair is what makes an action sensitive (a pod doing `pods/exec` is a very different story from `configmaps get`) |
| `k8s_user_type` | `sa` (ServiceAccount) \| `node` \| `user` — derived from the audit username |
| `k8s_status_code` | HTTP status the API server returned (403s = something probing permissions) |

### DNS_QUERY: from `cilium/flows.jsonl` L7 DNS records

One row per DNS **response** (the response carries the rcode; every query
gets one, NXDOMAIN included).

| Column | Meaning |
|---|---|
| `dns_query` | Normalized FQDN (lowercase, no trailing dot) |
| `dns_qtypes` | Query types, comma-joined (`A,AAAA`) |
| `dns_rcode` | `NOERROR`, `NXDOMAIN`, … — bursts of NXDOMAIN are a classic exfiltration/DGA tell |

> **Status: not currently collected.** Hubble only emits L7 DNS payloads with
> an explicit L7 visibility policy on the DNS port, which we keep disabled for
> performance. The ingest path is implemented and exercised synthetically: the
> `dns_exfil` perturbation in `gnnid eval` injects DNS_QUERY events to check
> the detector reacts (it does — which shows the *plumbing* works, not that
> the model would catch real exfiltration; treat that result accordingly).

### Worked example (one RPC_CALL row)

*The cart pod sent `GET /heartbeat` to the currency pod, which answered
200 OK in ~0 ms with a 10-byte response; only one sidecar reported it.*

```
event_id        boutique-mix_20260729-144632:istio/access_logs/currency-65df7f8478-qxg6t.log:1
run_id          boutique-mix_20260729-144632
ts              1785336488.104
event_type      RPC_CALL
src_id          pod:default/cart-649c7444bc-fhm2q       src_type  Pod
dst_id          pod:default/currency-65df7f8478-qxg6t   dst_type  Pod
dst_svc_id      NaN
protocol http | method GET | path /heartbeat | status_code 200 | grpc_status None
response_flags - | duration_ms 0.0 | request_bytes 0 | response_bytes 10 | reporter_views 1
l4_proto NaN | dst_port NaN | verdict NaN | drop_reason None | is_reply None       <- L4_FLOW only
dns_query None | dns_qtypes None | dns_rcode None                                  <- DNS_QUERY only
k8s_verb NaN | k8s_resource NaN | k8s_subresource NaN | k8s_user_type NaN | ...    <- K8S_API_CALL only
label None | label_source None                                                     <- attack ground truth (reserved)
```

---

## 3. entities.parquet

One row per canonical entity per run — the candidates for graph nodes.
Entity IDs follow a fixed grammar (from `src/gnnid/resolve.py`):

```
pod:<ns>/<name>       svc:<ns>/<name>       workload:<ns>/<kind>/<name>
dns:<fqdn>            ext:<class>           kubeapi:cluster
```

| entity_type | ID example | What it is |
|---|---|---|
| `Pod` | `pod:default/cart-649c7444bc-fhm2q` | A running pod — the unit the detector scores |
| `Service` | `svc:default/cart` | A ClusterIP service (RPCs addressed to it resolve here when the backend pod can't be pinned) |
| `Workload` | `workload:default/Deployment/cart` | The owning controller (excluded from graphs by default, `graph.include_workload_nodes: false`) |
| `DNSName` | `dns:productcatalog.default.svc.cluster.local` | A name that appeared in DNS traffic |
| `ExternalEndpoint` | `ext:host`, `ext:world` | Traffic classes outside the pod network: `host` = a cluster node's own IP, `world` = anything beyond |
| `KubeAPI` | `kubeapi:cluster` | The API server singleton — destination of every `K8S_API_CALL` |

### Columns

Only Pods carry the full attribute set; other types fill what applies.

| Column | Meaning | Populated for |
|---|---|---|
| `run_id` | Owning run | all |
| `entity_id`, `entity_type` | See grammar above | all |
| `name` | Object name (pod name, service name, FQDN, …) | all |
| `namespace` | K8s namespace | Pod, Service, Workload |
| `canonical_service` | **The role-label source**: the service this entity "is" (`cart`, `frontend`, …). For pods: the Istio `service.istio.io/canonical-name` label, else the pod's `app` label, else the hash-stripped owner name. Replica-suffixed Deployments (`frontend-2`) still map to `frontend` via the app label | Pod, Service |
| `workload` | Owning controller name (Deployment via the ReplicaSet chain) | Pod |
| `node_name` | Node the pod ran on | Pod |
| `pod_ip` | Pod IP (how flows/audit events get attributed to the pod) | Pod |
| `service_account` | ServiceAccount the pod runs as | Pod |
| `uid` | K8s object UID | Pod, Service |
| `has_sidecar` | Whether an `istio-proxy` container was present (no sidecar ⇒ the pod produces no RPC_CALL rows of its own; it still appears in flows/audit) | Pod |

**Identity firewall note:** identity fields (`name`, `canonical_service`,
`workload`, IPs, …) exist in this table for graph construction and for the
*label* side of training, but they are structurally excluded from a node's
own feature sentence (the `EventView` firewall in `src/gnnid/sentences.py`).
A pod is judged by what it *does*; `canonical_service` is what the model must
*predict*, never an input about itself.

---

## 4. How downstream consumes these tables

- Runs are sorted by the `run_id` timestamp; the temporal train/val/test
  split happens across run dirs. Since segments of one experiment are
  contiguous time slices, the split doubles as a within-experiment time split.
- Each run is cut into sliding windows (`windows.width_s: 30`,
  `stride_s: 15`); each window becomes one graph: entities are nodes, the
  events between them are the edges/features.
- Both detectors (`flash`, `ppt`) train **only on benign runs** to predict
  each node's role (`canonical_service`); at scoring time, a pod whose
  behavior no longer matches its role scores high. The `label` columns stay
  null until attack traffic with ground truth exists.
