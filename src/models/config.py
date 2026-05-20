import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
@dataclass
class V5Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    k_core: int = 10  # k=10 gives cleaner, denser interactions → better model quality
    adaptive_k_core: bool = True
    test_split: float = 0.2
    min_train_interactions: int = 5

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    embedding_device: Optional[str] = None
    embedding_dim: int = 384  # fallback when model unavailable

    # ── CF (ALS + BPR) ────────────────────────────────────────────────────────
    cf_models: List[str] = field(default_factory=lambda: ["als", "bpr"])
    cf_factors: int = 256
    cf_iterations: int = 70
    cf_regularization: float = 0.01
    cf_learning_rate: float = 0.05
    cf_alpha: float = 40.0

    # ── LightGCN ──────────────────────────────────────────────────────────────
    use_lightgcn: bool = True
    gcn_dim: int = 256
    gcn_layers: int = 3
    gcn_epochs: int = 50
    gcn_lr: float = 1e-3
    gcn_reg: float = 1e-4
    gcn_batch_size: int = 2048
    gcn_max_users: int = 50_000   # subsample if dataset too large

    # ── SASRec ────────────────────────────────────────────────────────────────
    use_sasrec: bool = True
    sas_dim: int = 256
    sas_heads: int = 8
    sas_layers: int = 4
    sas_maxlen: int = 50
    sas_dropout: float = 0.2
    sas_epochs: int = 100
    sas_lr: float = 1e-3
    sas_batch_size: int = 512
    sas_max_users: int = 50_000

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_m: int = 32
    faiss_ef_construction: int = 200
    faiss_ef_search: int = 64

    # ── Retrieval ─────────────────────────────────────────────────────────────
    candidate_pool_size: int = 500
    router_cold_threshold: int = 5
    router_warm_threshold: int = 20

    # ── LTR ───────────────────────────────────────────────────────────────────
    use_ltr: bool = False
    ltr_epochs: int = 100          # reduced from 150 → prevents overfitting
    ltr_learning_rate: float = 0.03  # lower LR → slower, more stable convergence
    ltr_hard_neg_cf: int = 8       # doubled hard negatives from CF → harder training signal
    ltr_hard_neg_gcn: int = 4      # doubled hard negatives from GCN → better discrimination
    ltr_random_neg: int = 8        # doubled random negatives → more diverse training
    ltr_sample_users: int = 15000   # more users → better generalization

    # ── IPS Debiasing ─────────────────────────────────────────────────────────
    ips_alpha: float = 0.5         # propensity smoothing exponent
    ips_clip_max: float = 10.0     # max IPS weight
    ips_clip_min: float = 0.1      # min IPS weight

    # ── MMR ───────────────────────────────────────────────────────────────────
    use_mmr: bool = False
    mmr_lambda: float = 0.6        # relevance vs diversity trade-off

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    eval_max_users: int = 2000

    # ── System ────────────────────────────────────────────────────────────────
    seed: int = 42
    artifact_dir: str = "./artifacts_v5"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "V5Config":
        with open(path) as f:
            return cls(**json.load(f))


# ==============================================================================
# LAYER 1: DATA LAYER
# ==============================================================================

