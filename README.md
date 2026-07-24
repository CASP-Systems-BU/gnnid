# gnnid

GNN-based intrusion detection for Kubernetes. It ingests telemetry collected from a live cluster by [ubench](https://github.com/CASP-Systems-BU/ubench) (K8s audit logs, Envoy access logs, Hubble L4 and DNS flows) and builds a graph per time window: entities are nodes (pods, services, DNS names, external endpoints, the K8s API) and observed events are typed edges. A model is trained on benign runs only to predict each node's service role; at scoring time a node whose behavior no longer matches its role is reported as an anomaly, so no attack data is needed for training.

Flow: ubench runs → parquet → per-window sentences → graphs → GraphSAGE + XGBoost → per-pod scores and alerts.

The detector follows the FLASH design (Word2Vec-encoded event sentences feeding a GraphSAGE classifier), moved from host provenance graphs to cluster telemetry. A node's own identity — name, workload, service, labels, IPs — is kept out of its own sentence, so a pod is judged on behavior rather than on what it is called.

## Setup

```bash
git submodule update --init      # ubench collector (pinned)
uv sync && uv pip install -e .   # CPU torch + PyG + gensim + xgboost
uv run pytest                    # 27 tests
```

## Running

```bash
uv run gnnid ingest   # ubench/results/*/ -> data/parquet/
uv run gnnid train    # Word2Vec -> GraphSAGE -> XGBoost -> artifacts/default/
uv run gnnid score    # per-pod scores and alerts for the test runs
uv run gnnid eval     # evaluation report (see below)
```

[configs/default.yaml](configs/default.yaml) holds all settings; override per run with `--set key=value` (e.g. `--set windows.width_s=60`, `--set scoring.source=gnn`).

Input runs come from `ingest.runs_glob`, by default `ubench/results/*/` — the runs sitting in the ubench submodule's working tree. ubench gitignores `results/`, so collected data is never committed by either repo; point `runs_glob` at another directory to ingest from elsewhere.

## Evaluation

There is no labelled attack data, so `gnnid eval` reports proxies:

- false-positive rate on held-out benign runs
- ROC-AUC on a weak signal — pod-windows containing Hubble `DROPPED`/`POLICY_DENIED` flows should score higher
- detection of four synthetic perturbations of benign events: `rewire`, `dns_exfil`, `sentence_swap`, `api_burst`
- ablations: Word2Vec only, GNN only, and the two concatenated

## Data collection

ubench deploys the benchmarks on a live cluster and collects the telemetry — `./bootstrap.sh addons` enables the audit-log and Istio gates, then `deploy.sh --run` runs a workload and writes a run directory. It is vendored as a submodule at [ubench/](ubench/), pinned to `main` (`b0b1f54`).

## Extending

This is the base pipeline. Ingest, graph construction, scoring, and evaluation are shared infrastructure, and the detector sits behind a config seam (`model.objective`, currently `flash_cls`), so further GNN intrusion-detection methods can be added alongside it without touching ingest or graph building.
