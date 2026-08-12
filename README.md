# gnnid

GNN-based intrusion detection for Kubernetes. It ingests telemetry collected from a live cluster by [ubench](https://github.com/CASP-Systems-BU/ubench) (K8s audit logs, Envoy access logs, Hubble L4 and DNS flows) and builds a graph per time window: entities are nodes (pods, services, DNS names, external endpoints, the K8s API) and observed events connect them. Models are trained on benign runs only to predict each node's service role; at scoring time a node whose behavior no longer matches its role is reported as an anomaly, so no attack data is needed for training.

Flow: ubench runs → parquet → per-window graphs → detector → per-pod scores and alerts.

The data specification lives in [DATA.md](DATA.md), including what each run directory contains and which source file every metric comes from

Two detectors are implemented behind a shared pipeline (`detector:` in the config):

- **flash** (default) — the FLASH design (Word2Vec-encoded event sentences feeding a GraphSAGE classifier + XGBoost), moved from host provenance graphs to cluster telemetry.
- **ppt** — PPT-GNN ([Van Langendonck et al. 2024](https://arxiv.org/abs/2406.13365)), a spatio-temporal heterogeneous GNN over seconds-scale sliding-window "memory graphs": events become feature-carrying nodes (line-graph style) linked to their endpoint entities by typed spatial edges, with same-source/same-destination temporal chains inside a window and entity-recurrence edges across the last N windows. It is pre-trained self-supervised (link prediction — no labels needed) on the benign runs, then fine-tuned on the same role-classification task.

In both, a node's own identity — name, workload, service, labels, IPs — is kept out of its own features (structurally, via the shared `EventView` firewall), so a pod is judged on behavior rather than on what it is called.

## Setup

```bash
git submodule update --init      # ubench collector (pinned)
uv sync && uv pip install -e .   # CPU torch + PyG + gensim + xgboost
uv run pytest                    # 61 tests
```

## Running

```bash
uv run gnnid ingest   # ubench/results/*/ -> data/parquet/  (shared by all detectors)
uv run gnnid train    # flash: Word2Vec -> GraphSAGE -> XGBoost -> artifacts/default/
uv run gnnid score    # per-pod scores and alerts for the test runs
uv run gnnid eval     # evaluation report (see below)

uv run gnnid train --detector ppt   # PPT-GNN: pretrain -> finetune -> artifacts/ppt/
uv run gnnid score --detector ppt   # results under results_eval/ppt/
uv run gnnid eval  --detector ppt
```

[configs/default.yaml](configs/default.yaml) holds all settings; override per run with `--set key=value` (e.g. `--set windows.width_s=60`, `--set scoring.source=gnn`). Base sections are the flash defaults; per-detector overlays live under `detectors.<name>` and win after the merge, so detector-scoped overrides go under that prefix (e.g. `--set detectors.ppt.windows.memory=8`). To compare PPT-GNN pretrained vs from-scratch, retrain with `--set detectors.ppt.train.pretrain.enabled=false`.

Input runs come from `ingest.runs_glob`, by default `ubench/results/*/` — the runs sitting in the ubench submodule's working tree. ubench gitignores `results/`, so collected data is never committed by either repo; point `runs_glob` at another directory to ingest from elsewhere.

## Evaluation

There is no labelled attack data, so `gnnid eval` reports proxies:

- false-positive rate on held-out benign runs
- ROC-AUC on a weak signal — pod-windows containing Hubble `DROPPED`/`POLICY_DENIED` flows should score higher
- detection of four synthetic perturbations of benign events: `rewire`, `dns_exfil`, `sentence_swap`, `api_burst`
- ablations: Word2Vec only, GNN only, and the two concatenated

## Data collection

ubench deploys the benchmarks on a live cluster and collects the telemetry. An experiment is a reusable YAML spec in `ubench/experiments/`:

```yaml
benchmark: boutique          # k8s/<benchmark>/ in ubench
request: mix                 # request mix (client/lua/<request>.lua)
workers: 4                   # placement spans this many worker nodes
replicas: {default: 1, overrides: {frontend: 2}}   # per-service pod counts
load: {threads: 4, conns: 16, rate: 1000}          # wrk2; rate = offered req/s
total_duration_s: 3600       # one continuous load...
segment_s: 600               # ...harvested into one run dir per segment
```

and runs with one command (env vars override spec values, e.g. `RATE=500`):

```bash
cd ubench/scripts/cloudlab
python3 register_cluster.py <node-0 hostname> <node-1> ...   # once per cluster
./bootstrap.sh                                               # once per cluster
./deploy.sh --experiment ../../experiments/boutique-mix.yaml --run
```

The load generator is wrk2 (open-loop, fixed `-R` rate); a long experiment is **one continuous run harvested into time-segment run directories**, so each run dir here is a contiguous slice of the same load and the temporal train/val/test split doubles as a within-experiment time split. Segments land in `ubench/results/` ready for `gnnid ingest`. Full spec schema and knee-sweep guidance: [ubench/experiments/README.md](ubench/experiments/README.md); cluster/bootstrap details: [ubench/scripts/cloudlab/README.md](ubench/scripts/cloudlab/README.md). ubench is vendored as a submodule at [ubench/](ubench/).

Runs collected with the old closed-loop `wrk` generator (`meta.json` has no `generator` key) and new wrk2 runs (`"generator": "wrk2"`) come from different load regimes — don't mix them in one trained model; retrain on a consistent set.

## Extending

Ingest, windowing, temporal splits, role labels, quantile normalization, per-pod aggregation, and the whole eval harness are shared infrastructure. A detector is one class implementing the `Detector` interface ([src/gnnid/detectors/base.py](src/gnnid/detectors/base.py)) — it owns its graph construction, training procedure (any number of stages), artifact bundle, and scoring — registered in [src/gnnid/detectors/](src/gnnid/detectors/) and selected by the `detector:` config key with a `detectors.<name>` config overlay. The parametrized acceptance test in [tests/test_detector_contract.py](tests/test_detector_contract.py) defines what a new detector must satisfy; `flash` and `ppt` are the two references.
