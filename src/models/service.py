import math
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from .config import V5Config
from .ranking import CandidateGenerator, Ranker, PostProcessor
from .components import CFModule, LightGCNModule, SASRecModule, FAISSModule, PopularityModule

logger = logging.getLogger("RecSysV5")
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

