import math
import logging
import warnings
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.auto import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. LightGCN + SASRec will be skipped.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("faiss not installed. Content retrieval disabled.")

try:
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    warnings.warn("implicit not installed. CF disabled.")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Embeddings disabled.")

from .config import V5Config
from .data_layer import DataLayer

logger = logging.getLogger("RecSysV5")
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


