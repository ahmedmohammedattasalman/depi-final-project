import pickle
import json
import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb
from pathlib import Path
from pipeline import PipelineV5, V5Config

class RecSysInference:
    def __init__(self, model_dir='.'):
        self.model_dir = Path(model_dir)
        self.config = V5Config.load(self.model_dir / 'config.json')
        self.pipeline = PipelineV5(self.config)

        # 1. Load base artifacts (has n_users, n_items, user_stats, etc.)
        with open(self.model_dir / 'artifacts.pkl', 'rb') as f:
            self.pipeline._artifacts = pickle.load(f)

        # 2. Set n_users / n_items from artifacts FIRST
        self.pipeline.n_users = self.pipeline._artifacts.get('n_users', 0)
        self.pipeline.n_items = self.pipeline._artifacts.get('n_items', 0)
        n_items = self.pipeline.n_items

        # 3. Load FAISS index
        self.pipeline.faiss_mod.index = faiss.read_index(str(self.model_dir / 'faiss.index'))

        # 4. Load LTR model (optional)
        ltr_path = self.model_dir / 'ltr.lgb'
        if ltr_path.exists():
            self.pipeline.ranker.model = lgb.Booster(model_file=str(ltr_path))
            self.pipeline.ranker._trained = True

        # 5. Load CF model
        with open(self.model_dir / 'cf_model.pkl', 'rb') as f:
            cf_model = pickle.load(f)
        self.pipeline.cf_mod.models[self.config.cf_models[0]] = cf_model
        self.pipeline.cf_mod.best_model_name = self.config.cf_models[0]
        self.pipeline.cf_mod.user_factors = np.asarray(cf_model.user_factors, dtype='float32')
        self.pipeline.cf_mod.item_factors = np.asarray(cf_model.item_factors, dtype='float32')

        # 6. Load product vectors
        self.pipeline._artifacts['product_vectors'] = self._load_product_vectors()

        # 7. Load IPS weights (now n_items is correct)
        self.pipeline._artifacts['ips'] = np.ones(n_items, dtype='float32')

        # 8. Load item metadata & encoders
        self.pipeline._artifacts['item_stats'] = pd.read_parquet(self.model_dir / 'item_metadata.parquet')
        self.pipeline._artifacts['user_encoder'] = pickle.load(open(self.model_dir / 'user_mapping.pkl', 'rb'))
        self.pipeline._artifacts['item_encoder'] = pickle.load(open(self.model_dir / 'item_mapping.pkl', 'rb'))

        # 9. Build PopularityModule from item_stats
        self.pipeline.pop_mod.build(self.pipeline._artifacts['item_stats'], n_items)

        # 10. Rebuild the recommendation service
        self._rebuild_service()

    def _load_product_vectors(self):
        vec_path = self.model_dir / 'product_vectors.npy'
        if vec_path.exists():
            return np.load(vec_path)
        else:
            return np.random.randn(self.pipeline.n_items, self.config.embedding_dim).astype('float32')

    def _rebuild_service(self):
        from pipeline import CandidateGenerator, RecommendationService, PostProcessor
        cand_gen = CandidateGenerator(
            self.pipeline.cf_mod, self.pipeline.faiss_mod, self.pipeline.pop_mod,
            self.pipeline.gcn_mod, self.pipeline.sas_mod, self.config
        )
        self.pipeline.service = RecommendationService(
            cand_gen, self.pipeline.ranker, self.pipeline.post_proc,
            self.pipeline.cf_mod, self.pipeline.gcn_mod, self.pipeline.sas_mod,
            self.pipeline.faiss_mod, self.pipeline.pop_mod, self.config,
            self.pipeline._artifacts['item_stats'], self.pipeline._artifacts['ips']
        )

    def recommend(self, user_id, top_k=10):
        try:
            uid_enc = self.pipeline._artifacts['user_encoder'].transform([str(user_id)])[0]
        except Exception:
            raise ValueError(f"User {user_id} not found in mapping")

        product_vectors = self.pipeline._artifacts['product_vectors']
        n_items = product_vectors.shape[0]

        # Reconstruct approximate user_items from CF model scores
        cf_scores = self.pipeline.cf_mod.get_scores(uid_enc, n_items)
        # Top items the CF model associates with this user (proxy for history)
        user_items = np.argsort(-cf_scores)[:50].astype(int)

        # Get user stats from artifacts
        user_stats_df = self.pipeline._artifacts.get('user_stats')
        if user_stats_df is not None and uid_enc in user_stats_df['reviewerID'].values:
            row = user_stats_df[user_stats_df['reviewerID'] == uid_enc].iloc[0]
            user_stats = {
                'activity': int(row.get('activity', 0)),
                'avg_rating': float(row.get('avg_rating', 0.0)),
                'rating_std': float(row.get('rating_std', 0.0)),
            }
        else:
            user_stats = {'activity': 0, 'avg_rating': 0.0, 'rating_std': 0.0}

        return self.pipeline.service.recommend(
            uid_enc, user_items, product_vectors, user_stats, top_k
        )
