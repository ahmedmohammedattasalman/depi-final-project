import os
import sys
import pickle
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# Re-export all subcomponents to preserve API compatibility
from .config import V5Config
from .data_layer import DataLayer, FeatureEngineeringLayer
from .components import EmbeddingModule, CFModule, LightGCNModule, SASRecModule, FAISSModule, PopularityModule
from .ranking import CandidateGenerator, Ranker, PostProcessor
from .service import RecommendationService, Evaluator

def setup_logging(level: int = logging.INFO) -> logging.Logger:
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
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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

