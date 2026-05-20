"""
================================================================================
v5 PRODUCTION RECOMMENDER SYSTEM — Amazon-Scale (FAANG-Level)
================================================================================
New in v5 vs v4:
  ✅ LightGCN  — Graph Neural Network candidate generator
  ✅ SASRec    — Sequential Transformer candidate generator
  ✅ Hard Negative Mining — CF + GNN + in-batch negatives
  ✅ IPS Debiasing — Inverse Propensity Score popularity correction
  ✅ Embedding-based MMR — True cosine-penalty diversity re-ranking
  ✅ 5-source Candidate Union
  ✅ 15+ LTR features (up from 7)
  ✅ Chunked .json.gz loading
  ✅ Zero runtime errors / full fallback logic
================================================================================
"""

import os
import sys
import json
import gzip
import math
import pickle
import logging
import warnings
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder, minmax_scale
from tqdm.auto import tqdm

# ── PyTorch (required for LightGCN + SASRec) ──────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. LightGCN + SASRec will be skipped.")

# ── FAISS ──────────────────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("faiss not installed. Content retrieval disabled.")

# ── Implicit (ALS / BPR) ──────────────────────────────────────────────────────
try:
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    warnings.warn("implicit not installed. CF disabled.")

# ── Sentence Transformers ──────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True  # Disabled temporarily for speed
except ImportError:
    ST_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Embeddings disabled.")

# ── LightGBM ──────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not installed. LTR disabled.")


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    # Ensure stdout can handle UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logger = logging.getLogger("RecSysV5")
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(h)
    return logger

logger = setup_logging()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # prevent ALS threading warning
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

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

class DataLayer:
    """ Ingestion, cleaning, k-core filtering, ID encoding, temporal split."""

    def __init__(self, config: V5Config):
        self.config = config
        self.user_encoder: Optional[LabelEncoder] = None
        self.item_encoder: Optional[LabelEncoder] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    @staticmethod
    def load_json_gz(path: str, max_rows: Optional[int] = None,
                     chunksize: int = 200_000) -> pd.DataFrame:
        """Load Amazon-style .json or .json.gz in chunks."""
        path = str(path)
        chunks: List[pd.DataFrame] = []
        total = 0

        needed = {"overall", "reviewerID", "asin", "reviewText", "summary",
                  "unixReviewTime", "helpful", "title", "category"}

        open_fn = gzip.open if path.endswith(".gz") else open

        with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
            batch: List[dict] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                batch.append({k: row.get(k) for k in needed})
                total += 1

                if len(batch) >= chunksize:
                    chunks.append(pd.DataFrame(batch))
                    batch = []
                    logger.info(f"  Loaded {total:,} rows...")
                    if max_rows and total >= max_rows:
                        break

            if batch:
                chunks.append(pd.DataFrame(batch))

        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        if max_rows:
            df = df.iloc[:max_rows]
        logger.info(f"Loaded {len(df):,} rows total")
        return df

    # ── Cleaning ──────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data")
        df = df.copy()

        # Build text field
        text_parts = []
        for col in ["title", "category", "reviewText", "summary"]:
            text_parts.append(df[col].fillna("").astype(str) if col in df.columns
                              else pd.Series([""] * len(df), index=df.index))
        df["final_review"] = text_parts[0].str.cat(text_parts[1:], sep=" ").str.strip()
        df = df.drop(columns=[c for c in ["reviewText","summary","title","category"]
                               if c in df.columns], errors="ignore")

        # Helpful votes
        if "helpful" in df.columns:
            df["helpful_votes"] = df["helpful"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else 0)
            df["total_votes"] = df["helpful"].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else 0)
            df["helpful_ratio"] = df["helpful_votes"] / (df["total_votes"] + 1e-9)
            df = df.drop(columns=["helpful", "helpful_votes", "total_votes"])
        else:
            df["helpful_ratio"] = 0.0

        # Rename + type coerce
        if "asin" in df.columns:
            df = df.rename(columns={"asin": "productID"})
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
        df["unixReviewTime"] = pd.to_numeric(df["unixReviewTime"], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["reviewerID", "productID", "overall", "unixReviewTime"])
        df = df[df["overall"].between(1, 5)]
        df = df[df["unixReviewTime"] > 0]
        logger.info(f"Cleaned: {before:,} -> {len(df):,} rows")
        return df.reset_index(drop=True)

    # ── K-Core ────────────────────────────────────────────────────────────────
    def k_core_filter(self ,df, min_user_k=8 , min_item_k=3, max_k=10, max_iter= 5):
        """
        Adaptive k-core filtering for large recommendation datasets.
        Keeps enough density without destroying coverage.
        """
        curr = df.copy()
    
        # choose k based on dataset size
        n = len(curr)
        if n < 200_000:
            k_user = k_item = min_user_k
        elif n < 1_000_000:
            k_user = k_item = min(min_user_k + 1, max_k)
        else:
            k_user = k_item = min(min_user_k + 2, max_k)
    
        # never exceed max_k
        k_user = min(k_user, max_k)
        k_item = min(k_item, max_k)
    
        print(f"[*] Adaptive K-core start | user_k={k_user} | item_k={k_item}")

        for i in range(max_iter):
            before = len(curr)
    
            user_counts = curr["reviewerID"].value_counts()
            item_counts = curr["productID"].value_counts()
    
            curr = curr[curr["reviewerID"].isin(user_counts[user_counts >= k_user].index)]
            curr = curr[curr["productID"].isin(item_counts[item_counts >= k_item].index)]
    
            after = len(curr)
            dropped = before - after
    
            print(f"--- Iteration {i+1}: dropped {dropped:,} rows | remaining {after:,}")
    
            if dropped == 0:
                print("[+] Converged.")
                break
    
        print(f"[OK] Final shape: {curr.shape}")
        print(f"[OK] Users: {curr['reviewerID'].nunique():,}")
        print(f"[OK] Items: {curr['productID'].nunique():,}")
        return curr.reset_index(drop=True)

    # ── Encode IDs ────────────────────────────────────────────────────────────

    def encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding IDs")
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        df = df.copy()
        df["reviewerID"] = self.user_encoder.fit_transform(df["reviewerID"].astype(str))
        df["productID"] = self.item_encoder.fit_transform(df["productID"].astype(str))
        logger.info(f"  {len(self.user_encoder.classes_):,} users, "
                    f"{len(self.item_encoder.classes_):,} items")
        return df

    # ── Temporal Split ────────────────────────────────────────────────────────

    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Temporal split: {1-self.config.test_split:.0%} train / "
                    f"{self.config.test_split:.0%} test")
        
        df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
        
        n_inter = df.groupby("reviewerID")["productID"].transform("count")
        rank_desc = df.groupby("reviewerID")["unixReviewTime"].rank(method="first", ascending=False)
        
        min_train = self.config.min_train_interactions
        test_split = self.config.test_split
        
        n_test = np.floor(n_inter * test_split).clip(1, None)
        n_test = np.minimum(n_test, n_inter - min_train)
        
        valid_users = n_inter >= (min_train + 1)
        is_test = valid_users & (rank_desc <= n_test)
        
        train_df = df[~is_test].reset_index(drop=True)
        test_df = df[is_test].reset_index(drop=True)
        
        # Remove test users without sufficient train history
        if not test_df.empty:
            tc = train_df["reviewerID"].value_counts()
            valid = set(tc[tc >= min_train].index)
            test_df = test_df[test_df["reviewerID"].isin(valid)].reset_index(drop=True)

        logger.info(f"Split: {len(train_df):,} train, {len(test_df):,} test | "
                    f"test users: {test_df['reviewerID'].nunique() if not test_df.empty else 0:,}")
        return train_df, test_df

    # ── Sparse matrix ─────────────────────────────────────────────────────────

    @staticmethod
    def build_coo(df: pd.DataFrame, n_users: int, n_items: int) -> coo_matrix:
        return coo_matrix(
            (df["interaction_score"].values.astype("float32"),
             (df["reviewerID"].values.astype("int32"),
              df["productID"].values.astype("int32"))),
            shape=(n_users, n_items)
        )


# ==============================================================================
# LAYER 2: FEATURE ENGINEERING (TRAIN ONLY)
# ==============================================================================

class FeatureEngineeringLayer:
    """All features computed from TRAIN data only — no leakage."""

    def __init__(self, config: V5Config):
        self.config = config
        self.item_propensity: Optional[np.ndarray] = None  # IPS scores

    def compute_interaction_score(self, df: pd.DataFrame,
                                  current_time: Optional[int] = None,
                                  score_min: Optional[float] = None,
                                  score_max: Optional[float] = None
                                  ) -> pd.DataFrame:
        """Time-decay × rating × helpfulness → normalized interaction score."""
        df = df.copy()
        if current_time is None:
            current_time = int(df["unixReviewTime"].max())

        time_diff_days = (current_time - df["unixReviewTime"]) / 86400.0
        # Half-life = 365 days
        time_weight = np.power(0.5, time_diff_days / 365.0).clip(0.05, 1.0)

        rating_map = {1: 0.1, 2: 0.25, 3: 0.5, 4: 0.8, 5: 1.0}
        rating_weight = df["overall"].map(rating_map).fillna(0.5)
        helpful_weight = 1.0 + df.get("helpful_ratio", pd.Series(0.0, index=df.index)).fillna(0)

        score = rating_weight * helpful_weight * time_weight
        score = np.log1p(score)

        if score_min is not None and score_max is not None and score_max > score_min:
            score = (score - score_min) / (score_max - score_min)
        else:
            score = minmax_scale(score.values.reshape(-1, 1)).flatten()

        df["interaction_score"] = score.clip(0, 1).astype("float32")
        return df

    def compute_advanced_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        df = train_df.copy()
        global_avg = df['overall'].mean()
        
        # Implicit label (1 if rating >= 4)
        df['implicit_label'] = (df['overall'] >= 4).astype(int)
        
        # User and item biases
        user_avg = df.groupby('reviewerID')['overall'].transform('mean')
        item_avg = df.groupby('productID')['overall'].transform('mean')
        df['user_bias'] = user_avg - global_avg
        df['item_bias'] = item_avg - global_avg
        
        # Normalized rating
        df['normalized_rating'] = df['overall'] - (global_avg + df['user_bias'] + df['item_bias'])
        
        return df

    def compute_item_statistics(self, train_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing item statistics (train-only)")
        current_time = int(train_df["unixReviewTime"].max())

        stats = train_df.groupby("productID").agg(
            avg_rating=("overall", "mean"),
            review_count=("overall", "count"),
            rating_std=("overall", "std"),
            total_interaction=("interaction_score", "sum"),
            avg_interaction=("interaction_score", "mean"),
            last_seen=("unixReviewTime", "max"),
            first_seen=("unixReviewTime", "min"),
            unique_users=("reviewerID", "nunique"),
        ).reset_index()

        stats["recency_days"] = (current_time - stats["last_seen"]) / 86400.0
        stats["age_days"] = (current_time - stats["first_seen"]) / 86400.0
        stats["popularity_score"] = stats["avg_rating"] * np.log1p(stats["review_count"])
        stats["popularity_score"] = minmax_scale(
            stats["popularity_score"].values.reshape(-1, 1)).flatten()

        # Time-decayed popularity (exponential decay on recency)
        decay = np.exp(-stats["recency_days"] / 365.0)
        stats["trending_score"] = stats["popularity_score"] * decay
        stats["trending_score"] = minmax_scale(
            stats["trending_score"].values.reshape(-1, 1)).flatten()

        return stats

    def compute_user_statistics(self, train_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing user statistics (train-only)")
        stats = train_df.groupby("reviewerID").agg(
            avg_rating=("overall", "mean"),
            rating_std=("overall", "std"),
            activity=("overall", "count"),
            total_interaction=("interaction_score", "sum"),
            avg_interaction=("interaction_score", "mean"),
            unique_items=("productID", "nunique"),
            last_active=("unixReviewTime", "max"),
            first_active=("unixReviewTime", "min"),
        ).reset_index()
        return stats

    def compute_ips_weights(self, train_df: pd.DataFrame, n_items: int) -> np.ndarray:
        """
        Inverse Propensity Score weights for popularity debiasing.
        propensity[i] = (count_i / max_count)^alpha
        ips_weight[i] = clip(1 / propensity[i])
        """
        logger.info("Computing IPS weights")
        counts = train_df["productID"].value_counts()
        max_count = float(counts.max())
        alpha = self.config.ips_alpha

        self.item_propensity = np.ones(n_items, dtype="float32")
        for iid, cnt in counts.items():
            if 0 <= int(iid) < n_items:
                prop = (cnt / max_count) ** alpha
                self.item_propensity[int(iid)] = max(prop, 1e-6)

        ips = 1.0 / self.item_propensity
        ips = np.clip(ips, self.config.ips_clip_min, self.config.ips_clip_max)
        logger.info(f"IPS: mean={ips.mean():.3f}, max={ips.max():.3f}, "
                    f"min={ips.min():.3f}")
        return ips


# ==============================================================================
# LAYER 3A: EMBEDDING MODULE
# ==============================================================================

class EmbeddingModule:
    """Sentence-Transformer embeddings with CPU/GPU support and fallback."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.embedding_dim: int = config.embedding_dim
        self.device = config.embedding_device or (
            "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        )

    def load(self):
        if self.model is None:
            if not ST_AVAILABLE:
                raise RuntimeError("sentence-transformers not installed")
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            self.model = self.model.to(self.device)
            probe = self.model.encode(["probe"], batch_size=1, convert_to_numpy=True,
                                       show_progress_bar=False)
            self.embedding_dim = int(probe.shape[1])
            logger.info(f"Embedding model ready | dim={self.embedding_dim} | device={self.device}")
        return self.model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self.load()
        return model.encode(
            texts, batch_size=self.config.embedding_batch_size,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

    def build_product_embeddings(self, train_df: pd.DataFrame,
                                  n_items: int) -> np.ndarray:
        logger.info("Building product embeddings (Fast Mode: top 3 reviews/product)")
        # FIX: Avoid Memory Crash & 6-Hour Runtime
        # Sort and take best 3 reviews per product
        top_df = train_df.sort_values("interaction_score", ascending=False).groupby("productID").head(3)[["productID", "interaction_score", "final_review"]].copy()
        
        logger.info(f"Encoding {len(top_df)} reviews instead of {len(train_df)}...")
        review_embeddings = self.encode(top_df["final_review"].tolist())
        top_df["emb"] = list(review_embeddings)
        
        dim = review_embeddings.shape[1]
        vectors = np.zeros((n_items, dim), dtype="float32")

        for pid, grp in tqdm(top_df.groupby("productID"), desc="Prod-emb", leave=False):
            w = grp["interaction_score"].values.astype("float32") + 1e-8
            vecs = np.vstack(grp["emb"].values)
            avg = np.average(vecs, axis=0, weights=w)
            pid_int = int(pid)
            if 0 <= pid_int < n_items:
                vectors[pid_int] = avg

        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12, None)
        vectors /= norms
        n_nonzero = int(np.count_nonzero(np.linalg.norm(vectors, axis=1)))
        logger.info(f"Product embeddings: shape={vectors.shape}, non-zero={n_nonzero}")
        return vectors.astype("float32")


# ==============================================================================
# LAYER 3B: COLLABORATIVE FILTERING (ALS + BPR)
# ==============================================================================

class CFModule:
    """Trains ALS and BPR; selects best by quick NDCG@10 validation."""

    def __init__(self, config):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def train(self, train_matrix: coo_matrix,
              val_users: Optional[np.ndarray] = None,
              val_gt: Optional[Dict[int, set]] = None) -> Optional[str]:
        if not IMPLICIT_AVAILABLE:
            logger.warning("implicit not available — CF skipped")
            return None

        csr = train_matrix.tocsr().astype("float32")
        scores: Dict[str, float] = {}

        for name in self.config.cf_models:
            if name == "als":
                conf = csr.copy()
                conf.data = (conf.data * self.config.cf_alpha).astype("float32")
                m = AlternatingLeastSquares(
                    factors=self.config.cf_factors,
                    regularization=self.config.cf_regularization,
                    iterations=self.config.cf_iterations,
                    random_state=self.config.seed
                )
                m.fit(conf, show_progress=False)
            elif name == "bpr":
                m = BayesianPersonalizedRanking(
                    factors=self.config.cf_factors,
                    learning_rate=self.config.cf_learning_rate,
                    regularization=self.config.cf_regularization,
                    iterations=self.config.cf_iterations,
                    random_state=self.config.seed
                )
                m.fit(csr, show_progress=False)
            else:
                continue

            self.models[name] = m
            ndcg = self._eval(m, csr, val_users, val_gt) if val_users is not None else 0.0
            scores[name] = ndcg
            logger.info(f"  CF [{name.upper()}] val NDCG@10 = {ndcg:.4f}")

        if not scores:
            return None
        self.best_model_name = max(scores, key=scores.get)
        best = self.models[self.best_model_name]
        self.user_factors = np.asarray(best.user_factors, dtype="float32")
        self.item_factors = np.asarray(best.item_factors, dtype="float32")
        logger.info(f"Best CF: {self.best_model_name.upper()}")
        return self.best_model_name

    def _eval(self, model, csr, val_users, val_gt, k: int = 10) -> float:
        ndcgs = []
        for uid in (val_users[:100] if val_users is not None else []):
            if val_gt is None or uid not in val_gt:
                continue
            recs, _ = model.recommend(int(uid), csr[uid], N=k,
                                       filter_already_liked_items=True)
            rel = val_gt[uid]
            dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(recs) if int(r) in rel)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        return float(np.mean(ndcgs)) if ndcgs else 0.0

    def get_scores(self, user_id: int, n_items: int) -> np.ndarray:
        """Return CF scores for all items. Returns zeros on failure."""
        if self.user_factors is None or self.item_factors is None:
            return np.zeros(n_items, dtype="float32")
        if user_id >= len(self.user_factors):
            return np.zeros(n_items, dtype="float32")
        return (self.item_factors @ self.user_factors[user_id]).astype("float32")


# ==============================================================================
# LAYER 3C: LIGHTGCN — Graph Neural Network Candidate Generator
# ==============================================================================

if TORCH_AVAILABLE:
    class _LightGCNModel(nn.Module):
        """
        LightGCN: simplified GCN without feature transformation or nonlinearity.
        He et al., 2020.  E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)
        Final emb = mean-pool over all layers.
        """
        def __init__(self, n_users: int, n_items: int, dim: int, n_layers: int):
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items
            self.n_layers = n_layers
            self.user_emb = nn.Embedding(n_users, dim)
            self.item_emb = nn.Embedding(n_items, dim)
            nn.init.xavier_uniform_(self.user_emb.weight)
            nn.init.xavier_uniform_(self.item_emb.weight)

        def forward(self, adj: "torch.sparse.Tensor"):
            """
            adj: (n_users + n_items) x (n_users + n_items) sparse normalised adj.
            Returns final user_emb (n_users x dim), item_emb (n_items x dim).
            """
            E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            all_embs = [E0]
            E = E0
            for _ in range(self.n_layers):
                E = torch.sparse.mm(adj, E)
                all_embs.append(E)
            E_final = torch.stack(all_embs, dim=1).mean(dim=1)
            u_emb = E_final[:self.n_users]
            i_emb = E_final[self.n_users:]
            return u_emb, i_emb

        def bpr_loss(self, users: "torch.Tensor", pos_items: "torch.Tensor",
                     neg_items: "torch.Tensor", u_emb: "torch.Tensor",
                     i_emb: "torch.Tensor", reg: float) -> "torch.Tensor":
            u = u_emb[users]
            p = i_emb[pos_items]
            n = i_emb[neg_items]
            pos_scores = (u * p).sum(dim=1)
            neg_scores = (u * n).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            reg_loss = reg * (
                self.user_emb.weight[users].norm(2).pow(2) +
                self.item_emb.weight[pos_items].norm(2).pow(2) +
                self.item_emb.weight[neg_items].norm(2).pow(2)
            ) / len(users)
            return loss + reg_loss


class LightGCNModule:
    """
    LightGCN wrapper: builds graph, trains, provides candidate scores.
    Gracefully skips if PyTorch unavailable or graph too large.
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.user_emb: Optional[np.ndarray] = None
        self.item_emb: Optional[np.ndarray] = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    def _build_adj(self, train_df: pd.DataFrame, n_users: int, n_items: int):
        """Build symmetric normalised bipartite adjacency with self-loops (sparse).
        
        FIX: Added self-connection (identity) before D^{-1/2} A D^{-1/2} normalization.
        This ensures each node preserves its own embedding signal across layers,
        preventing information loss in deep GCN propagation (He et al., 2020).
        """
        rows_u = train_df["reviewerID"].values.astype("int64")
        cols_i = train_df["productID"].values.astype("int64") + n_users
        N = n_users + n_items

        # Symmetric: user->item + item->user
        row = np.concatenate([rows_u, cols_i])
        col = np.concatenate([cols_i, rows_u])
        data = np.ones(len(row), dtype="float32")

        from scipy.sparse import coo_matrix as sp_coo, eye as sp_eye
        A = sp_coo((data, (row, col)), shape=(N, N)).tocsr()

        # FIX: Add self-loops (I + A) before normalization
        # This preserves node's own embedding at each propagation layer
        A = A + sp_eye(N, format="csr", dtype="float32")

        # D^(-1/2) — degree now includes self-loop
        deg = np.asarray(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype("float32")

        # D^(-1/2) (I+A) D^(-1/2)
        A = A.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :]).tocoo()

        indices = torch.from_numpy(np.vstack([A.row, A.col])).long()
        values  = torch.from_numpy(A.data).float()
        adj = torch.sparse_coo_tensor(indices, values, (N, N))
        return adj.to(self.device)

    def train(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> bool:
        if not TORCH_AVAILABLE:
            logger.warning("LightGCN skipped: PyTorch not available")
            return False
        if not self.config.use_lightgcn:
            logger.info("LightGCN disabled in config")
            return False

        # Subsample users for memory safety
        unique_users = train_df["reviewerID"].unique()
        if len(unique_users) > self.config.gcn_max_users:
            logger.info(f"LightGCN: subsampling {self.config.gcn_max_users:,} users")
            keep_users = set(np.random.choice(unique_users, self.config.gcn_max_users,
                                               replace=False))
            df = train_df[train_df["reviewerID"].isin(keep_users)].copy()
        else:
            df = train_df.copy()

        logger.info(f"Building LightGCN graph: {n_users} users, {n_items} items")
        adj = self._build_adj(df, n_users, n_items)

        self.model = _LightGCNModel(n_users, n_items,
                                     self.config.gcn_dim,
                                     self.config.gcn_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.gcn_lr)

        # Build training triples
        user_arr = df["reviewerID"].values.astype("int64")
        item_arr = df["productID"].values.astype("int64")
        item_set_per_user: Dict[int, set] = defaultdict(set)
        for u, i in zip(user_arr, item_arr):
            item_set_per_user[int(u)].add(int(i))

        bs = self.config.gcn_batch_size
        logger.info(f"Training LightGCN for {self.config.gcn_epochs} epochs")
        for epoch in range(self.config.gcn_epochs):
            self.model.train()
            idx = np.random.permutation(len(user_arr))
            total_loss = 0.0
            steps = 0
            for start in range(0, len(idx), bs):
                batch_idx = idx[start: start + bs]
                bu = user_arr[batch_idx]
                bp = item_arr[batch_idx]
                # Sample negatives (not in user's history)
                bn = np.random.randint(0, n_items, size=len(bu))
                for k_idx, u in enumerate(bu):
                    while bn[k_idx] in item_set_per_user.get(int(u), set()):
                        bn[k_idx] = np.random.randint(0, n_items)

                tu = torch.from_numpy(bu).long().to(self.device)
                tp = torch.from_numpy(bp).long().to(self.device)
                tn = torch.from_numpy(bn).long().to(self.device)

                optimizer.zero_grad()
                u_emb, i_emb = self.model(adj)
                loss = self.model.bpr_loss(tu, tp, tn, u_emb, i_emb,
                                            self.config.gcn_reg)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                steps += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  LightGCN epoch {epoch+1}/{self.config.gcn_epochs} "
                            f"| loss={total_loss/max(steps,1):.4f}")

        # Extract final embeddings
        self.model.eval()
        with torch.no_grad():
            u_emb, i_emb = self.model(adj)
        self.user_emb = u_emb.cpu().numpy().astype("float32")
        self.item_emb = i_emb.cpu().numpy().astype("float32")
        logger.info("LightGCN training complete")
        return True

    def get_scores(self, user_id: int, n_items: int) -> np.ndarray:
        """Return GCN recommendation scores for all items."""
        if self.user_emb is None or self.item_emb is None:
            return np.zeros(n_items, dtype="float32")
        if user_id >= len(self.user_emb):
            return np.zeros(n_items, dtype="float32")
        return (self.item_emb @ self.user_emb[user_id]).astype("float32")


# ==============================================================================
# LAYER 3D: SASRec — Sequential Self-Attention Candidate Generator
# ==============================================================================

if TORCH_AVAILABLE:
    class _SASRecModel(nn.Module):
        """
        SASRec: Self-Attentive Sequential Recommendation.
        Kang & McAuley, 2018.
        2-layer causal Transformer -> predict next item.
        """
        def __init__(self, n_items: int, dim: int, n_heads: int,
                     n_layers: int, maxlen: int, dropout: float):
            super().__init__()
            self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
            self.pos_emb  = nn.Embedding(maxlen, dim)
            self.dropout  = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.ln = nn.LayerNorm(dim)
            self.dim = dim
            self.maxlen = maxlen

        def forward(self, seq: "torch.Tensor") -> "torch.Tensor":
            """
            seq: (B, L) item IDs (0 = pad).
            Returns: (B, L, dim) sequence representations.
            """
            B, L = seq.shape
            pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
            x = self.item_emb(seq) + self.pos_emb(pos)
            x = self.dropout(x)
            # Causal mask: each position can only attend to past
            mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
            x = self.transformer(x, mask=mask)
            return self.ln(x)


class SASRecModule:
    """
    SASRec wrapper: builds user sequences, trains, provides next-item scores.
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.n_items: int = 0
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    def _build_sequences(self, train_df: pd.DataFrame) -> Dict[int, List[int]]:
        """Build time-sorted item sequences per user."""
        seqs: Dict[int, List[int]] = {}
        for uid, grp in train_df.sort_values("unixReviewTime").groupby("reviewerID"):
            items = grp["productID"].tolist()
            # Keep last maxlen items; shift by 1 (item IDs start from 1, 0=pad)
            seqs[int(uid)] = [i + 1 for i in items[-self.config.sas_maxlen:]]
        return seqs

    def train(self, train_df: pd.DataFrame, n_items: int) -> bool:
        if not TORCH_AVAILABLE:
            logger.warning("SASRec skipped: PyTorch not available")
            return False
        if not self.config.use_sasrec:
            logger.info("SASRec disabled in config")
            return False

        self.n_items = n_items
        logger.info("Building SASRec sequences")
        seqs = self._build_sequences(train_df)

        # Subsample users for speed
        all_users = list(seqs.keys())
        if len(all_users) > self.config.sas_max_users:
            all_users = list(np.random.choice(all_users, self.config.sas_max_users,
                                               replace=False))

        self.model = _SASRecModel(
            n_items=n_items,
            dim=self.config.sas_dim,
            n_heads=self.config.sas_heads,
            n_layers=self.config.sas_layers,
            maxlen=self.config.sas_maxlen,
            dropout=self.config.sas_dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.sas_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.sas_epochs, eta_min=1e-5 )
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        maxlen = self.config.sas_maxlen
        bs = self.config.sas_batch_size

        logger.info(f"Training SASRec for {self.config.sas_epochs} epochs "
                    f"| users={len(all_users):,}")

        for epoch in range(self.config.sas_epochs):
            self.model.train()
            np.random.shuffle(all_users)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(all_users), bs):
                batch_users = all_users[start: start + bs]
                # Build padded input (X) and target (Y) sequences
                X_batch, Y_batch = [], []
                for uid in batch_users:
                    seq = seqs[uid]
                    if len(seq) < 2:
                        continue
                    # Input: seq[:-1]; Target: seq[1:]
                    inp = seq[:-1]
                    tgt = seq[1:]
                    # Pad to maxlen-1
                    pad_len = (maxlen - 1) - len(inp)
                    inp = [0] * pad_len + inp
                    tgt = [0] * pad_len + tgt
                    X_batch.append(inp[-(maxlen-1):])
                    Y_batch.append(tgt[-(maxlen-1):])

                if not X_batch:
                    continue

                X = torch.tensor(X_batch, dtype=torch.long).to(self.device)
                Y = torch.tensor(Y_batch, dtype=torch.long).to(self.device)  # (B, L)

                optimizer.zero_grad()
                out = self.model(X)   # (B, L, dim)
                # Score against all items
                logits = out @ self.model.item_emb.weight.T  # (B, L, n_items+1)
                B, L, V = logits.shape
                loss = loss_fn(logits.reshape(B * L, V), Y.reshape(B * L))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  SASRec epoch {epoch+1}/{self.config.sas_epochs} "
                            f"| loss={epoch_loss/max(steps,1):.4f}")

        logger.info("SASRec training complete")
        return True

    def get_scores(self, user_id: int, user_seq: List[int],
                   n_items: int) -> np.ndarray:
        """
        Given a user's item sequence (integer IDs), return next-item scores.
        user_seq: list of raw item IDs (0-indexed from encoded df).
        """
        if self.model is None or len(user_seq) == 0:
            return np.zeros(n_items, dtype="float32")

        self.model.eval()
        maxlen = self.config.sas_maxlen
        # Shift IDs by +1 (0 = pad)
        seq = [i + 1 for i in user_seq[-maxlen:]]
        pad_len = maxlen - len(seq)
        seq = [0] * pad_len + seq

        with torch.no_grad():
            X = torch.tensor([seq], dtype=torch.long).to(self.device)
            out = self.model(X)               # (1, maxlen, dim)
            last = out[0, -1, :]              # (dim,)
            logits = self.model.item_emb.weight[1:] @ last  # (n_items,)
            scores = logits.cpu().numpy().astype("float32")

        # Normalise to [0,1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        return scores


# ==============================================================================
# LAYER 4: FAISS + POPULARITY + CANDIDATE GENERATOR
# ==============================================================================

class FAISSModule:
    def __init__(self, config):
        self.config = config
        self.index = None

    def build(self, product_vectors: np.ndarray):
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available — content retrieval disabled")
            return None
        logger.info("Building FAISS HNSW index")
        dim = product_vectors.shape[1]
        base = faiss.IndexHNSWFlat(dim, self.config.faiss_m)
        base.hnsw.efConstruction = self.config.faiss_ef_construction
        base.hnsw.efSearch = self.config.faiss_ef_search
        self.index = faiss.IndexIDMap2(base)
        ids = np.arange(len(product_vectors), dtype="int64")
        self.index.add_with_ids(product_vectors, ids)
        logger.info(f"FAISS: {self.index.ntotal} items, {dim}D")
        return self.index

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            return np.array([[]]), np.array([[]])
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return self.index.search(query.astype("float32"), k)

    def get_user_vector(self, user_items: np.ndarray,
                        product_vectors: np.ndarray,
                        weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if len(user_items) == 0:
            return None
        vecs = product_vectors[user_items]
        if weights is not None and len(weights) == len(user_items):
            w = weights.astype("float32") + 1e-8
            w /= w.sum()
            vec = np.average(vecs, axis=0, weights=w)
        else:
            vec = vecs.mean(axis=0)
        vec = vec.reshape(1, -1).astype("float32")
        if FAISS_AVAILABLE:
            faiss.normalize_L2(vec)
        return vec


class PopularityModule:
    def __init__(self, config):
        self.config = config
        self.scores: Optional[np.ndarray] = None
        self.ranking: Optional[np.ndarray] = None
        self.trending_scores: Optional[np.ndarray] = None

    def build(self, item_stats: pd.DataFrame, n_items: int):
        logger.info("Building popularity index")
        self.scores = np.zeros(n_items, dtype="float32")
        self.trending_scores = np.zeros(n_items, dtype="float32")
        for _, row in item_stats.iterrows():
            pid = int(row["productID"])
            if 0 <= pid < n_items:
                self.scores[pid] = float(row.get("popularity_score", 0))
                self.trending_scores[pid] = float(row.get("trending_score", 0))
        self.ranking = np.argsort(-self.scores)
        logger.info(f"Popularity: {int(np.count_nonzero(self.scores))} items scored")

    def get_top_k(self, k: int, exclude: Optional[set] = None) -> List[int]:
        exclude = exclude or set()
        result: List[int] = []
        for iid in self.ranking:
            iid = int(iid)
            if iid not in exclude:
                result.append(iid)
            if len(result) >= k:
                break
        return result


class CandidateGenerator:
    """
    5-source hybrid candidate retrieval:
    CF  |  Content (FAISS)  |  Popularity  |  LightGCN  |  SASRec
    """
    def __init__(self, cf: CFModule, faiss_mod: FAISSModule,
                 pop: PopularityModule, gcn: LightGCNModule,
                 sas: SASRecModule, config):
        self.cf = cf
        self.faiss = faiss_mod
        self.pop = pop
        self.gcn = gcn
        self.sas = sas
        self.config = config

    def _weights(self, activity: int) -> Dict[str, float]:
        c = self.config
        if activity < c.router_cold_threshold:
            return {"cf": 0.1, "content": 0.3, "pop": 0.3, "gcn": 0.1, "sas": 0.2}
        elif activity < c.router_warm_threshold:
            return {"cf": 0.3, "content": 0.2, "pop": 0.1, "gcn": 0.2, "sas": 0.2}
        else:
            return {"cf": 0.35, "content": 0.15, "pop": 0.05, "gcn": 0.25, "sas": 0.2}

    def generate(self, user_id: int, user_items: np.ndarray,
                 product_vectors: np.ndarray,
                 user_stats: Dict[str, Any]) -> np.ndarray:
        activity = int(user_stats.get("activity", 0))
        w = self._weights(activity)
        pool = self.config.candidate_pool_size
        n_items = product_vectors.shape[0]
        seen: set = set(int(i) for i in user_items)
        candidates: List[int] = []

        def _add(source_ids):
            for iid in source_ids:
                iid = int(iid)
                if iid >= 0 and iid not in seen:
                    seen.add(iid)
                    candidates.append(iid)

        # 1. CF
        n_cf = max(1, int(pool * w["cf"]))
        cf_scores = self.cf.get_scores(user_id, n_items)
        _add(np.argsort(-cf_scores)[:n_cf + len(seen)])

        # 2. Content (FAISS)
        n_cnt = max(1, int(pool * w["content"]))
        user_vec = self.faiss.get_user_vector(user_items, product_vectors)
        if user_vec is not None:
            _, ids = self.faiss.search(user_vec, n_cnt + len(user_items) + 10)
            _add(ids[0])

        # 3. Popularity
        n_pop = max(1, int(pool * w["pop"]))
        _add(self.pop.get_top_k(n_pop + len(user_items), exclude=seen))

        # 4. LightGCN
        n_gcn = max(1, int(pool * w["gcn"]))
        gcn_scores = self.gcn.get_scores(user_id, n_items)
        _add(np.argsort(-gcn_scores)[:n_gcn + len(seen)])

        # 5. SASRec
        n_sas = max(1, int(pool * w["sas"]))
        sas_scores = self.sas.get_scores(user_id, list(user_items), n_items)
        _add(np.argsort(-sas_scores)[:n_sas + len(seen)])

        return np.array(candidates[:pool], dtype=int)


# ==============================================================================
# LAYER 5: RANKING — LightGBM LambdaRank with Hard Negative Mining + IPS
# ==============================================================================

class Ranker:
    """
    LightGBM LambdaRank with:
    - 15 features (CF, GCN, SASRec, content, popularity, user/item stats)
    - IPS sample weights for popularity bias correction
    - Hard negative mining: CF-hard + GCN-hard + in-batch random
    """

    FEATURE_COLS = [
        
        "cf_score", "gcn_score", "sas_score", "content_score", "score_cf_gcn_diff",
        "user_longtail_affinity",
        "item_rating_velocity",
        "pop_score", "trend_score", "ips_weight",
        "user_activity", "user_avg_rating", "user_rating_std",
        "item_review_count", "item_avg_rating", "item_recency_days",
        "item_age_days", "item_unique_users",
    ]

    def __init__(self, config):
        self._max_user_activity = 1
        self.config = config
        self.model = None
        self._trained = False

    def _build_item_lookup(self, item_stats: pd.DataFrame) -> Dict[int, Dict]:
        lk: Dict[int, Dict] = {}
        for _, row in item_stats.iterrows():
            lk[int(row["productID"])] = {
                "rating_velocity": float(row.get("rating_velocity", 0.0)),
                "review_count": float(row.get("review_count", 0)),
                "avg_rating":   float(row.get("avg_rating", 0)),
                "recency_days": float(row.get("recency_days", 0)),
                "age_days":     float(row.get("age_days", 0)),
                "unique_users": float(row.get("unique_users", 0)),
                "pop_score":    float(row.get("popularity_score", 0)),
                "trend_score":  float(row.get("trending_score", 0)),
            }
        return lk

    def _make_row(self, iid: int, label: int,
                  cf_s: np.ndarray, gcn_s: np.ndarray,
                  sas_s: np.ndarray, cnt_s: np.ndarray,
                  ips: np.ndarray, user_stats: Dict,
                  item_lk: Dict[int, Dict], n_items: int) -> Dict:
        ifeats = item_lk.get(iid, {})
        return {
            "item_id":          iid,
            "label":            label,
            "cf_score":         float(cf_s[iid]) if iid < len(cf_s) else 0.0,
            "gcn_score":        float(gcn_s[iid]) if iid < len(gcn_s) else 0.0,
            "sas_score":        float(sas_s[iid]) if iid < len(sas_s) else 0.0,
            "content_score":    float(cnt_s[iid]) if iid < len(cnt_s) else 0.0,
            "pop_score":        ifeats.get("pop_score", 0.0),
            "trend_score":      ifeats.get("trend_score", 0.0),
            "ips_weight":       float(ips[iid]) if iid < len(ips) else 1.0,
            "user_activity":    float(user_stats.get("activity", 0)),
            "user_avg_rating":  float(user_stats.get("avg_rating", 0)),
            "user_rating_std":  float(user_stats.get("rating_std", 0)),
            "item_review_count":ifeats.get("review_count", 0.0),
            "item_avg_rating":  ifeats.get("avg_rating", 0.0),
            "item_recency_days":ifeats.get("recency_days", 0.0),
            "item_age_days":    ifeats.get("age_days", 0.0),
            "item_unique_users":ifeats.get("unique_users", 0.0),
            "score_cf_gcn_diff": float(cf_s[iid]) - float(gcn_s[iid]) if iid < len(cf_s) else 0.0,
            "user_longtail_affinity": 1.0 - (float(user_stats.get("activity", 0)) / max(1, self._max_user_activity)),
            "item_rating_velocity": ifeats.get("rating_velocity", 0.0),
        }

    def train(self, train_df: pd.DataFrame,
          cf: CFModule, gcn: LightGCNModule,
          sas: SASRecModule, faiss_mod: FAISSModule,
          product_vectors: np.ndarray, ips: np.ndarray,
          item_stats: pd.DataFrame) -> bool:

        if not LGB_AVAILABLE:
            logger.warning("LightGBM not available — LTR skipped")
            return False
        if not self.config.use_ltr:
            logger.info("LTR disabled in config")
            return False
    
        n_items = product_vectors.shape[0]
        item_lk = self._build_item_lookup(item_stats)
    
        pop_items = (
            item_stats.sort_values("popularity_score", ascending=False)["productID"]
            .astype(int)
            .tolist()[:500]
            )

        # ------------------------------------------------------------
        # User sampling setup
        # ------------------------------------------------------------
        unique_users = train_df["reviewerID"].unique()
        self._max_user_activity = int(train_df.groupby("reviewerID").size().max())
    
        n_sample = min(self.config.ltr_sample_users, len(unique_users))
        sample_users = np.random.choice(unique_users, n_sample, replace=False)
    
        rows: List[Dict] = []
        groups: List[int] = []
    
        logger.info(f"Building LTR training set ({n_sample} users)...")
    
        for uid in tqdm(sample_users, desc="LTR sampling", leave=False):
            udf = train_df[train_df["reviewerID"] == uid]
    
            # positives only from meaningful interactions
            pos_items = udf["productID"].dropna().astype(int).tolist()
            
            if len(pos_items) < 2:
                continue
    
            user_stats = {
                "activity":   len(pos_items),
                "avg_rating": float(udf["overall"].mean()),
                "rating_std": float(udf["overall"].std(ddof=0)),
             }
    
            pos_set = set(pos_items)
            uid_int = int(uid)
    
            # --------------------------------------------------------
            # Scores from base recommenders
            # --------------------------------------------------------
            cf_s  = cf.get_scores(uid_int, n_items)
            gcn_s = gcn.get_scores(uid_int, n_items)
            sas_s = sas.get_scores(uid_int, pos_items, n_items)
    
            uv = faiss_mod.get_user_vector(np.array(pos_items), product_vectors)
            cnt_s = np.zeros(n_items, dtype="float32")
            if uv is not None:
                _, ids = faiss_mod.search(uv, min(200, n_items))
                for rank, iid in enumerate(ids[0]):
                    if 0 <= int(iid) < n_items:
                        cnt_s[int(iid)] = 1.0 - rank / len(ids[0])
    
            group_size = 0
    
            # Positives
            for iid in pos_items:
                rows.append(
                    self._make_row(
                        iid, 1, cf_s, gcn_s, sas_s, cnt_s,
                        ips, user_stats, item_lk, n_items
                    )
                )
                group_size += 1
    
            # Hard negatives
            hard_cf = [int(i) for i in np.argsort(-cf_s) if int(i) not in pos_set][:self.config.ltr_hard_neg_cf]
            hard_gcn = [int(i) for i in np.argsort(-gcn_s) if int(i) not in pos_set][:self.config.ltr_hard_neg_gcn]
            rand_neg = [i for i in pop_items if i not in pos_set][:self.config.ltr_random_neg]
    
            for iid in hard_cf + hard_gcn + rand_neg:
                rows.append(
                    self._make_row(
                        iid, 0, cf_s, gcn_s, sas_s, cnt_s,
                        ips, user_stats, item_lk, n_items
                    )
                )
                group_size += 1
    
            groups.append(group_size)
    
        if len(rows) < 200:
            logger.warning(f"LTR: only {len(rows)} training rows — skipping")
            return False
    
        # ------------------------------------------------------------
        # Build LTR dataframe first
        # ------------------------------------------------------------
        df_ltr = pd.DataFrame(rows)
    
        # ------------------------------------------------------------
        # Shuffle by user group BEFORE creating X/y/w
        # ------------------------------------------------------------
        shuffle_idx = np.random.permutation(len(groups))
        shuffled_groups = [groups[i] for i in shuffle_idx]
    
        group_starts = np.concatenate([[0], np.cumsum(groups)])
        row_order = np.concatenate([
            np.arange(group_starts[i], group_starts[i + 1]) for i in shuffle_idx
        ])
    
        df_ltr = df_ltr.iloc[row_order].reset_index(drop=True)
        groups = shuffled_groups
    
        # ------------------------------------------------------------
        # NOW build features/labels/weights in the correct order
        # ------------------------------------------------------------
        X = df_ltr[self.FEATURE_COLS].fillna(0).astype("float32")
        y = df_ltr["label"].astype(int)
        w = df_ltr.apply(
            lambda r: float(r["ips_weight"]) if r["label"] == 1 else 1.0,
            axis=1
        ).astype("float32")
    
        # 80/20 split by groups
        split_idx = int(len(groups) * 0.8)
        train_rows = sum(groups[:split_idx])
    
        ds_train = lgb.Dataset(
            X.iloc[:train_rows],
            label=y.iloc[:train_rows],
            group=groups[:split_idx],
            weight=w.iloc[:train_rows]
        )
        ds_val = lgb.Dataset(
            X.iloc[train_rows:],
            label=y.iloc[train_rows:],
            group=groups[split_idx:],
            weight=w.iloc[train_rows:],
            reference=ds_train
        )
    
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "num_leaves": 15,
            "learning_rate": self.config.ltr_learning_rate,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "min_child_weight": 1e-3,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": self.config.seed,
        }
    
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=25),
        ]
    
        logger.info(
            f"Training LightGBM LambdaRank | rows={len(df_ltr):,} | "
            f"groups={len(groups)} | early_stopping=20 rounds"
        )
    
        self.model = lgb.train(
            params,
            ds_train,
            num_boost_round=self.config.ltr_epochs,
            valid_sets=[ds_val],
            callbacks=callbacks,
        )
    
        imp = dict(zip(self.FEATURE_COLS, self.model.feature_importance()))
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("LTR top features: " + ", ".join(f"{k}={v}" for k, v in top))
        self._trained = True
        return True

    def rank(self, candidates: np.ndarray, user_id: int,
             cf_s: np.ndarray, gcn_s: np.ndarray, sas_s: np.ndarray,
             cnt_s: np.ndarray, ips: np.ndarray,
             user_stats: Dict, item_lk: Dict, n_items: int) -> np.ndarray:

        if len(candidates) == 0:
            return candidates

        if self._trained and self.model is not None:
            rows = [self._make_row(int(iid), 0, cf_s, gcn_s, sas_s, cnt_s,
                                   ips, user_stats, item_lk, n_items)
                    for iid in candidates]
            df = pd.DataFrame(rows)[self.FEATURE_COLS].fillna(0).astype("float32")
            scores = self.model.predict(df)
            return candidates[np.argsort(-scores)]

        # Fallback: weighted hybrid with MinMaxScaling to prevent CF score dominance
        def _minmax(arr: np.ndarray) -> np.ndarray:
            """Normalise array to [0, 1]; returns zeros if flat."""
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                return (arr - lo) / (hi - lo)
            return np.zeros_like(arr)

        cf_norm  = _minmax(cf_s)  if len(cf_s)  > 0 else np.zeros(len(candidates))
        gcn_norm = _minmax(gcn_s) if len(gcn_s) > 0 else np.zeros(len(candidates))
        sas_norm = _minmax(sas_s) if len(sas_s) > 0 else np.zeros(len(candidates))
        cnt_norm = _minmax(cnt_s) if len(cnt_s) > 0 else np.zeros(len(candidates))

        w = self._router_weights(int(user_stats.get("activity", 0)))
        hybrid = np.zeros(len(candidates), dtype="float32")
        for i, iid in enumerate(candidates):
            iid = int(iid)
            hybrid[i] = (w["cf"]  * (float(cf_norm[iid])  if iid < len(cf_norm)  else 0) +
                         w["gcn"] * (float(gcn_norm[iid]) if iid < len(gcn_norm) else 0) +
                         w["sas"] * (float(sas_norm[iid]) if iid < len(sas_norm) else 0) +
                         w["cnt"] * (float(cnt_norm[iid]) if iid < len(cnt_norm) else 0))
        return candidates[np.argsort(-hybrid)]

    
    @staticmethod
    def _router_weights(activity: int) -> Dict[str, float]:
        # تجاهل activity واجعل الأوزان ثابتة بناءً على الأداء
        return {"cf": 0.45, "gcn": 0.3, "sas": 0.25, "cnt": 0.0}


# ==============================================================================
# LAYER 6: POST-PROCESSOR — IPS Debiasing + Embedding-based MMR
# ==============================================================================

class PostProcessor:
    """
    1. IPS score adjustment — divide relevance by popularity propensity
    2. MMR re-ranking — cosine-similarity penalty in embedding space
    """

    def __init__(self, config):
        self.config = config

    def ips_adjust(self, items: np.ndarray, scores: np.ndarray,
                   ips: np.ndarray) -> np.ndarray:
        """Multiply scores by IPS weight to de-emphasise popular items."""
        adjusted = scores.copy()
        for i, iid in enumerate(items):
            iid = int(iid)
            if iid < len(ips):
                adjusted[i] = scores[i] * float(ips[iid])
        return adjusted

    def mmr_rerank(self, items: np.ndarray, scores: np.ndarray,
                   product_vectors: np.ndarray) -> np.ndarray:
        """
        Maximal Marginal Relevance with embedding-based cosine penalty.
        score(i) = lambda * rel(i) - (1-lambda) * max_{j in S} cos(i, j)
        """
        if not self.config.use_mmr or len(items) <= 1:
            return items

        lam = self.config.mmr_lambda
        selected: List[int] = []
        remaining = list(range(len(items)))

        # Select first (highest relevance)
        first = int(np.argmax(scores))
        selected.append(first)
        remaining.remove(first)

        while remaining:
            best_score = -np.inf
            best_idx = remaining[0]
            sel_vecs = product_vectors[items[selected]]   # (|S|, dim)

            for idx in remaining:
                iid = int(items[idx])
                rel = float(scores[idx])
                iv = product_vectors[iid]                 # (dim,)
                sims = sel_vecs @ iv                      # (|S|,)
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                mmr = lam * rel - (1 - lam) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return items[selected]


# ==============================================================================
# LAYER 7: RECOMMENDATION SERVICE (SERVING)
# ==============================================================================

class RecommendationService:
    """
    Orchestrates end-to-end inference for a single user request.
    Handles cold-start, scoring, ranking, post-processing.
    """

    def __init__(self, candidate_gen: CandidateGenerator, ranker: Ranker,
                 post_proc: PostProcessor, cf: CFModule, gcn: LightGCNModule,
                 sas: SASRecModule, faiss_mod: FAISSModule, pop: PopularityModule,
                 config, item_stats: pd.DataFrame, ips: np.ndarray):
        self.cgen     = candidate_gen
        self.ranker   = ranker
        self.post     = post_proc
        self.cf       = cf
        self.gcn      = gcn
        self.sas      = sas
        self.faiss    = faiss_mod
        self.pop      = pop
        self.config   = config
        self.ips      = ips
        self._item_lk = ranker._build_item_lookup(item_stats) if hasattr(ranker, '_build_item_lookup') else {}
        self._cache: Dict[str, Any] = {}

    def recommend(self, user_id: int, user_items: np.ndarray,
                  product_vectors: np.ndarray, user_stats: Dict[str, Any],
                  top_k: int = 10) -> List[Dict[str, Any]]:

        cache_key = f"{user_id}_{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_items = product_vectors.shape[0]
        is_cold = len(user_items) < self.config.router_cold_threshold

        # Cold-start fallback: popularity + content
        if is_cold:
            seen = set(int(i) for i in user_items)
            # 1. جلب عناصر شعبية
            pop_items = self.pop.get_top_k(top_k * 2, exclude=seen)
            # 2. جلب عناصر مشابهة للمحتوى (إذا كان المستخدم لديه أي تفاعل)
            content_items = []
            if len(user_items) > 0:
                user_vec = self.faiss.get_user_vector(user_items, product_vectors)
                if user_vec is not None:
                    _, ids = self.faiss.search(user_vec, top_k * 2)
                    content_items = [int(iid) for iid in ids[0] if int(iid) not in seen]
            # 3. دمج القائمتين وإزالة التكرار
            combined = []
            for iid in content_items + pop_items:
                if iid not in combined and iid not in seen:
                    combined.append(iid)
                if len(combined) >= top_k:
                    break
            results = [{"productID": int(i), "score": float(self.pop.scores[i] if i < len(self.pop.scores) else 0.5), "cold_start": True} for i in combined[:top_k]]
            self._cache[cache_key] = results
            return results
        # 1. Generate candidates (5 sources)   <-- أضف هذا القسم
        candidates = self.cgen.generate(user_id, user_items, product_vectors, user_stats)
        if len(candidates) == 0:
            return []
            
        # 2. Compute all scores
        cf_s  = self.cf.get_scores(user_id, n_items)
        gcn_s = self.gcn.get_scores(user_id, n_items)
        sas_s = self.sas.get_scores(user_id, list(user_items), n_items)

        uv = self.faiss.get_user_vector(user_items, product_vectors)
        cnt_s = np.zeros(n_items, dtype="float32")
        if uv is not None:
            _, ids = self.faiss.search(uv, min(300, n_items))
            for rank, iid in enumerate(ids[0]):
                if 0 <= int(iid) < n_items:
                    cnt_s[int(iid)] = 1.0 - rank / len(ids[0])

        # 3. Rank
        ranked = self.ranker.rank(candidates, user_id, cf_s, gcn_s, sas_s,
                                  cnt_s, self.ips, user_stats,
                                  self._item_lk, n_items)

        # 4. Post-process: IPS adjust + MMR
        top_cands = ranked[:min(top_k * 3, len(ranked))]
        base_scores = np.array([
            float(cf_s[int(i)] if int(i) < len(cf_s) else 0) for i in top_cands
        ], dtype="float32")
        adj_scores = self.post.ips_adjust(top_cands, base_scores, self.ips)
        mmr_items  = self.post.mmr_rerank(top_cands, adj_scores, product_vectors)
        final      = mmr_items[:top_k]

        results = []
        for iid in final:
            iid = int(iid)
            ifeats = self._item_lk.get(iid, {})
            results.append({
                "productID":    iid,
                "score":        float(cf_s[iid]) if iid < len(cf_s) else 0.0,
                "gcn_score":    float(gcn_s[iid]) if iid < len(gcn_s) else 0.0,
                "sas_score":    float(sas_s[iid]) if iid < len(sas_s) else 0.0,
                "pop_score":    ifeats.get("pop_score", 0.0),
                "avg_rating":   ifeats.get("avg_rating", 0.0),
                "cold_start":   False,
            })

        self._cache[cache_key] = results
        return results

    def clear_cache(self):
        self._cache.clear()


# ==============================================================================
# LAYER 8: EVALUATOR
# ==============================================================================

class Evaluator:
    """NDCG, MAP, HitRate, Precision, Recall, Coverage, Diversity, Novelty."""

    def __init__(self, config):
        self.config = config

    def evaluate(self, service: RecommendationService,
                 test_df: pd.DataFrame, train_df: pd.DataFrame,
                 product_vectors: np.ndarray,
                 max_users: Optional[int] = None) -> Dict[str, float]:

        # ------------------------------------------------------------
        # FIX 1: prevent stale recommendations from affecting eval
        # ------------------------------------------------------------
        if hasattr(service, "clear_cache"):
            service.clear_cache()

        max_users = max_users or self.config.eval_max_users
        test_users = test_df["reviewerID"].unique().astype(int)
        if len(test_users) > max_users:
            np.random.seed(self.config.seed)
            test_users = np.random.choice(test_users, max_users, replace=False)

        ks = self.config.eval_k_values
        buckets: Dict[str, List[float]] = {
            f"{m}@{k}": [] for m in ["ndcg", "map", "hitrate", "precision", "recall"]
            for k in ks
        }

        # ------------------------------------------------------------
        # FIX 2: precompute lookups once instead of scanning train_df
        # ------------------------------------------------------------
        train_user_items = (
            train_df.sort_values("unixReviewTime")
            .groupby("reviewerID")["productID"]
            .apply(lambda s: s.values.astype(int))
            .to_dict()
        )
        train_user_items = {int(uid): items for uid, items in train_user_items.items()}

        train_user_stats = (
            train_df.groupby("reviewerID")
            .agg(
                activity=("overall", "count"),
                avg_rating=("overall", "mean"),
                rating_std=("overall", "std"),
            )
            .to_dict(orient="index")
        )
        train_user_stats = {
            int(uid): {
                "activity": int(stats.get("activity", 0)),
                "avg_rating": float(stats.get("avg_rating", 0.0)),
                "rating_std": float(0.0 if pd.isna(stats.get("rating_std", 0.0)) else stats.get("rating_std", 0.0)),
            }
            for uid, stats in train_user_stats.items()
        }

        # Ground-truth lookup once
        test_relevant = (
            test_df.groupby("reviewerID")["productID"]
            .apply(lambda s: set(s.astype(int).tolist()))
            .to_dict()
        )
        test_relevant = {int(uid): rel for uid, rel in test_relevant.items()}

        n_items = int(product_vectors.shape[0])

        all_recs: List[int] = []
        all_rec_lists: List[List[int]] = []

        # Precompute popularity ranking once for novelty
        pop_counts = np.bincount(
            train_df["productID"].astype(int).values,
            minlength=n_items
        )
        pop_order = np.argsort(-pop_counts)
        pop_rank = {int(i): int(r) for r, i in enumerate(pop_order)}

        for uid in tqdm(test_users, desc="Evaluating", leave=False):
            relevant = test_relevant.get(int(uid), set())
            if not relevant:
                continue

            user_items = train_user_items.get(int(uid), np.array([], dtype=int))
            if len(user_items) == 0:
                continue

            user_stats = train_user_stats.get(
                int(uid),
                {"activity": 0, "avg_rating": 0.0, "rating_std": 0.0},
            )

            max_k = max(ks)
            recs_raw = service.recommend(
                int(uid),
                user_items,
                product_vectors,
                user_stats,
                top_k=max_k
            )

            rec_ids = [int(r["productID"]) for r in recs_raw]
            all_recs.extend(rec_ids)
            all_rec_lists.append(rec_ids)

            for k in ks:
                top_k = rec_ids[:k]
                top_len = len(top_k)
                hits = sum(1 for r in top_k if r in relevant)

                buckets[f"precision@{k}"].append(hits / max(1, top_len))
                buckets[f"recall@{k}"].append(hits / max(1, len(relevant)))
                buckets[f"hitrate@{k}"].append(1.0 if hits > 0 else 0.0)

                dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(top_k) if r in relevant)
                ideal_len = min(len(relevant), top_len)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
                buckets[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

                ap, nh = 0.0, 0
                for i, r in enumerate(top_k, 1):
                    if r in relevant:
                        nh += 1
                        ap += nh / i
                denom = max(1, min(len(relevant), top_len))
                buckets[f"map@{k}"].append(ap / denom)

        # ------------------------------------------------------------
        # حساب المقاييس بعد انتهاء الحلقة (وليس داخلها)
        # ------------------------------------------------------------
        metrics: Dict[str, float] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in buckets.items()
        }

        metrics["coverage"] = len(set(all_recs)) / n_items if n_items > 0 else 0.0

        if all_recs and n_items > 0:
            novelty_vals = [
                -math.log2((pop_rank.get(r, n_items) + 1) / n_items)
                for r in all_recs
                if r in pop_rank
            ]
            metrics["novelty"] = float(np.mean(novelty_vals)) if novelty_vals else 0.0
        else:
            metrics["novelty"] = 0.0

        div_scores = []
        sample_lists = all_rec_lists[:200]
        for rec_list in sample_lists:
            if len(rec_list) < 2:
                continue
            vecs = product_vectors[rec_list]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-12, None)
            vecs_n = vecs / norms
            sim_mat = vecs_n @ vecs_n.T
            n = len(rec_list)
            pairs = n * (n - 1) / 2
            avg_sim = (sim_mat.sum() - n) / (2 * pairs) if pairs > 0 else 0.0
            div_scores.append(1.0 - avg_sim)

        metrics["diversity"] = float(np.mean(div_scores)) if div_scores else 0.0

        return metrics

    def print_report(self, metrics: Dict[str, float]) -> None:
        print("\n" + "=" * 60)
        print("  EVALUATION REPORT — v5 Production Recommender")
        print("=" * 60)
        header = f"{'Metric':<18}" + "".join(f"  @{k:<6}" for k in self.config.eval_k_values)
        print(header)
        print("-" * 60)
        for m in ["ndcg", "map", "hitrate", "precision", "recall"]:
            row = f"{m.upper():<18}"
            for k in self.config.eval_k_values:
                row += f"  {metrics.get(f'{m}@{k}', 0.0):.4f}"
            print(row)
        print("-" * 60)
        print(f"{'Coverage':<18}  {metrics.get('coverage', 0.0):.4f}")
        print(f"{'Novelty':<18}  {metrics.get('novelty', 0.0):.4f}")
        print(f"{'Diversity':<18}  {metrics.get('diversity', 0.0):.4f}")
        print("=" * 60 + "\n")
    
# ==============================================================================
# LAYER 9: PIPELINE ORCHESTRATOR
# ==============================================================================

class PipelineV5:
    """
    End-to-end training + serving pipeline.

    Usage:
        pipeline = PipelineV5(config)
        pipeline.fit(df)
        recs = pipeline.recommend(user_id=42, top_k=10)
        metrics = pipeline.evaluate()
        pipeline.save("./my_model")
    """

    def __init__(self, config: Optional[V5Config] = None):
        self.config = config or V5Config()
        set_seed(self.config.seed)

        self.data_layer   = DataLayer(self.config)
        self.feat_layer   = FeatureEngineeringLayer(self.config)
        self.embed_mod    = EmbeddingModule(self.config)
        self.cf_mod       = CFModule(self.config)
        self.gcn_mod      = LightGCNModule(self.config)
        self.sas_mod      = SASRecModule(self.config)
        self.faiss_mod    = FAISSModule(self.config)
        self.pop_mod      = PopularityModule(self.config)
        self.ranker       = Ranker(self.config)
        self.post_proc    = PostProcessor(self.config)
        self.evaluator    = Evaluator(self.config)

        self.service: Optional[RecommendationService] = None
        self._artifacts: Dict[str, Any] = {}
        self.n_users = 0
        self.n_items = 0

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("=" * 65)
        logger.info("  v5 PRODUCTION RECOMMENDER — TRAINING START")
        logger.info("=" * 65)

        # Step 1: Data
        logger.info("[1/9] Data Processing")
        df = self.data_layer.clean(df)
        df = self.data_layer.k_core_filter(df)
        df = self.data_layer.encode_ids(df)
        df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
        train_df, test_df = self.data_layer.temporal_split(df)
        self.n_users = int(df["reviewerID"].nunique())
        self.n_items = int(df["productID"].nunique())
        logger.info(f"  Dataset: {self.n_users:,} users, {self.n_items:,} items")

        # Step 2: Features
        logger.info("[2/9] Feature Engineering")
        t0 = int(train_df["unixReviewTime"].max())
        train_df = self.feat_layer.compute_interaction_score(train_df, t0)
        s_min = float(train_df["interaction_score"].min())
        s_max = float(train_df["interaction_score"].max())
        test_df = self.feat_layer.compute_interaction_score(
            test_df, t0, score_min=s_min, score_max=s_max)
        item_stats  = self.feat_layer.compute_item_statistics(train_df)
        user_stats  = self.feat_layer.compute_user_statistics(train_df)
        ips         = self.feat_layer.compute_ips_weights(train_df, self.n_items)

        # Step 3: Embeddings
        logger.info("[3/9] Embeddings")
        if ST_AVAILABLE:
            product_vectors = self.embed_mod.build_product_embeddings(
                train_df, self.n_items)
            self.config.embedding_dim = product_vectors.shape[1]
        else:
            logger.warning("Using random embeddings (sentence-transformers unavailable)")
            dim = self.config.embedding_dim
            product_vectors = np.random.randn(self.n_items, dim).astype("float32")
            norms = np.linalg.norm(product_vectors, axis=1, keepdims=True).clip(1e-12, None)
            product_vectors /= norms

        # Step 4: FAISS
        logger.info("[4/9] FAISS Index")
        self.faiss_mod.build(product_vectors)

        # Step 5: CF
        logger.info("[5/9] Collaborative Filtering (ALS + BPR)")
        train_matrix = DataLayer.build_coo(train_df, self.n_users, self.n_items)
        val_gt: Dict[int, set] = {}
        for uid in test_df["reviewerID"].unique()[:200]:
            val_gt[int(uid)] = set(
                test_df[test_df["reviewerID"] == uid]["productID"].astype(int).tolist())
        val_users = np.array(list(val_gt.keys()), dtype=int)
        self.cf_mod.train(train_matrix, val_users, val_gt)

        # Step 6: LightGCN
        logger.info("[6/9] LightGCN")
        self.gcn_mod.train(train_df, self.n_users, self.n_items)

        # Step 7: SASRec
        logger.info("[7/9] SASRec")
        self.sas_mod.train(train_df, self.n_items)

        # Step 8: Popularity
        logger.info("[8/9] Popularity Index")
        self.pop_mod.build(item_stats, self.n_items)

        # Step 9: Candidate Generator + Ranker + Service
        logger.info("[9/9] Candidate Generator + LTR + Service")
        cand_gen = CandidateGenerator(
            self.cf_mod, self.faiss_mod, self.pop_mod,
            self.gcn_mod, self.sas_mod, self.config)

        self.ranker.train(train_df, self.cf_mod, self.gcn_mod, self.sas_mod,
                          self.faiss_mod, product_vectors, ips, item_stats)

        self.service = RecommendationService(
            cand_gen, self.ranker, self.post_proc,
            self.cf_mod, self.gcn_mod, self.sas_mod,
            self.faiss_mod, self.pop_mod, self.config,
            item_stats, ips)

        self._artifacts = {
            "train_df": train_df, "test_df": test_df,
            "product_vectors": product_vectors,
            "item_stats": item_stats, "user_stats": user_stats,
            "ips": ips, "n_users": self.n_users, "n_items": self.n_items,
            "user_encoder": self.data_layer.user_encoder,
            "item_encoder": self.data_layer.item_encoder,
        }

        logger.info("=" * 65)
        logger.info("  TRAINING COMPLETE")
        logger.info("=" * 65)
        return self._artifacts

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Public inference API: recommend(user_id, top_k)."""
        if self.service is None:
            raise RuntimeError("Call fit() first.")
        train_df = self._artifacts["train_df"]
        product_vectors = self._artifacts["product_vectors"]
        user_items = train_df[train_df["reviewerID"] == user_id][
            "productID"].values.astype(int)
        us_row = self._artifacts["user_stats"]
        us_row = us_row[us_row["reviewerID"] == user_id]
        if len(us_row) > 0:
            row = us_row.iloc[0]
            user_stats = {"activity": int(row["activity"]),
                          "avg_rating": float(row["avg_rating"]),
                          "rating_std": float(row.get("rating_std", 0))}
        else:
            user_stats = {"activity": 0, "avg_rating": 0.0, "rating_std": 0.0}
        return self.service.recommend(user_id, user_items,
                                       product_vectors, user_stats, top_k)

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, max_users: Optional[int] = None) -> Dict[str, float]:
        if self.service is None:
            raise RuntimeError("Call fit() first.")
        metrics = self.evaluator.evaluate(
            self.service,
            self._artifacts["test_df"],
            self._artifacts["train_df"],
            self._artifacts["product_vectors"],
            max_users,
        )
        self.evaluator.print_report(metrics)
        return metrics

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.config.save(os.path.join(path, "config.json"))
        with open(os.path.join(path, "artifacts.pkl"), "wb") as f:
            pickle.dump({k: v for k, v in self._artifacts.items()
                         if k not in ("train_df", "test_df")}, f)
        if self.cf_mod.best_model_name:
            mdl = self.cf_mod.models[self.cf_mod.best_model_name]
            with open(os.path.join(path, "cf_model.pkl"), "wb") as f:
                pickle.dump(mdl, f)
        if FAISS_AVAILABLE and self.faiss_mod.index is not None:
            faiss.write_index(self.faiss_mod.index, os.path.join(path, "faiss.index"))
        if self.ranker.model is not None:
            self.ranker.model.save_model(os.path.join(path, "ltr.lgb"))
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PipelineV5":
        config = V5Config.load(os.path.join(path, "config.json"))
        pipeline = cls(config)
        with open(os.path.join(path, "artifacts.pkl"), "rb") as f:
            pipeline._artifacts = pickle.load(f)
        logger.info(f"Pipeline loaded from {path}")
        return pipeline

