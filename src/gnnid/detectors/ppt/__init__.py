"""PPT-GNN detector (Van Langendonck et al. 2024) ported to K8s telemetry.

Spatio-temporal heterogeneous GNN over sliding-window memory graphs: telemetry
events become typed "event" nodes (line-graph style) with identity-free
features, canonical entities the "entity" endpoints; self-supervised
link-prediction pre-training on benign runs, then fine-tuning on the
role-classification pretext task. Anomaly = 1 - p(true role), the same
scoring/threshold scheme as FLASH, so norm/thresholds/pod-aggregation/eval
are shared unchanged.

The package lands bottom-up: features -> graph -> model -> train -> score;
the PPTDetector class arrives with the scoring layer and registers itself
in gnnid.detectors.
"""
