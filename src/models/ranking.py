import logging
import warnings
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not installed. LTR disabled.")

from .config import V5Config
from .components import CFModule, LightGCNModule, SASRecModule, FAISSModule, PopularityModule

logger = logging.getLogger("RecSysV5")
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

