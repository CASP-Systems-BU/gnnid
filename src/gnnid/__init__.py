"""gnnid — FLASH-style GNN intrusion detection for Kubernetes telemetry.

Pipeline: ubench run dirs -> events/entities parquet -> windowed sentences
-> Word2Vec features -> homogeneous PyG graphs -> GraphSAGE role classifier
-> XGBoost on concat(w2v, gnn) -> misclassification/confidence anomaly scores.
"""

__version__ = "0.1.0"
