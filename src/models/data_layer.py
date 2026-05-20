import json
import gzip
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder, minmax_scale

from .config import V5Config

logger = logging.getLogger("RecSysV5")
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

