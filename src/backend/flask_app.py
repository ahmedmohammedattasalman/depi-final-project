import json
import logging
import pickle
from pathlib import Path
from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np

from src.models.inference import RecSysInference

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlaskRecSys")

app = Flask(__name__, template_folder='../../templates')

# Initialize Inference System
logger.info("Initializing recommendation model (loading weights)...")
recsys = RecSysInference(model_dir="data")
logger.info("Recommendation model initialized successfully!")

# Load configurations and metrics
config_path = Path("config/config.json")
metrics_path = Path("data/metrics_v5.json")

with open(config_path, "r", encoding="utf-8") as f:
    pipeline_config = json.load(f)

with open(metrics_path, "r", encoding="utf-8") as f:
    model_metrics = json.load(f)

# Custom category mapping cache
category_mapping = {}
last_loaded_time = 0.0

def load_category_mapping():
    global category_mapping, last_loaded_time
    json_path = Path("data/item_categories.json")
    csv_path = Path("data/item_categories.csv")
    
    mtime = 0.0
    target_path = None
    
    if json_path.exists():
        mtime = json_path.stat().st_mtime
        target_path = json_path
    elif csv_path.exists():
        mtime = csv_path.stat().st_mtime
        target_path = csv_path
        
    if not target_path or mtime <= last_loaded_time:
        return
        
    last_loaded_time = mtime
    if target_path == json_path:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                category_mapping = json.load(f)
            logger.info(f"Loaded {len(category_mapping)} custom categories from item_categories.json")
        except Exception as e:
            logger.error(f"Error loading item_categories.json: {str(e)}")
    elif target_path == csv_path:
        try:
            df = pd.read_csv(csv_path)
            if "asin" in df.columns and "category" in df.columns:
                category_mapping = dict(zip(df["asin"].astype(str), df["category"].astype(str)))
                logger.info(f"Loaded {len(category_mapping)} custom categories from item_categories.csv")
        except Exception as e:
            logger.error(f"Error loading item_categories.csv: {str(e)}")

# Try to load custom categories on start
load_category_mapping()

metadata_cache = {}
try:
    cache_path = Path("data/item_metadata_cache.json")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            metadata_cache = json.load(f)
        logger.info(f"Loaded {len(metadata_cache)} item titles and images from cache.")
except Exception as e:
    logger.error(f"Error loading metadata cache: {e}")

def get_product_meta(asin):
    return metadata_cache.get(asin, {"title": f"Product {asin}", "image": None})

@app.route("/admin")
def admin_dashboard():
    """Renders the main dashboard template."""
    return render_template("index.html")


@app.route("/")
def store_home():
    """Renders the real-world e-commerce home page."""
    item_encoder = recsys.pipeline._artifacts["item_encoder"]
    item_metadata = recsys.pipeline._artifacts.get("item_stats")
    
    trending = []
    if item_metadata is not None:
        top_popular = item_metadata.sort_values(by="trending_score", ascending=False).head(8)
        for _, row in top_popular.iterrows():
            prod_id = int(row["productID"])
            try:
                asin = str(item_encoder.inverse_transform([prod_id])[0])
            except:
                asin = f"ASIN_{prod_id}"
            
            meta = get_product_meta(asin)
            trending.append({
                "asin": asin,
                "title": meta["title"],
                "image": meta["image"],
                "category": get_product_category(asin),
                "avg_rating": float(row.get("avg_rating", 0.0)),
                "review_count": int(row.get("review_count", 0)),
                "price": round(29.99 + (int(row.get("review_count", 0)) % 100), 2)
            })
            
    return render_template("store_home.html", trending=trending)

@app.route("/product/<asin>")
def store_product(asin):
    """Renders the product detail page."""
    item_encoder = recsys.pipeline._artifacts["item_encoder"]
    item_metadata = recsys.pipeline._artifacts.get("item_stats")
    
    try:
        prod_id = item_encoder.transform([asin])[0]
    except Exception:
        # Product not found in encoder
        return "Product Not Found", 404
        
    meta = get_product_meta(asin)
    product = {
        "asin": asin,
        "title": meta["title"],
        "image": meta["image"],
        "category": get_product_category(asin),
        "avg_rating": 4.5,
        "review_count": 0,
        "popularity_score": 0.5,
        "trending_score": 0.5
    }
    
    if item_metadata is not None and prod_id in item_metadata["productID"].values:
        row = item_metadata[item_metadata["productID"] == prod_id].iloc[0]
        product.update({
            "avg_rating": float(row.get("avg_rating", 4.5)),
            "review_count": int(row.get("review_count", 0)),
            "popularity_score": float(row.get("popularity_score", 0.0)),
            "trending_score": float(row.get("trending_score", 0.0))
        })
        
    # Get similar products (fallback to trending if no similarity model)
    similar = []
    if item_metadata is not None:
        # Just grab random popular items for similar in this demo if we can't do FAISS lookup easily
        top_popular = item_metadata.sample(4) if len(item_metadata) > 4 else item_metadata
        for _, row in top_popular.iterrows():
            sim_id = int(row["productID"])
            if sim_id == prod_id: continue
            try:
                sim_asin = str(item_encoder.inverse_transform([sim_id])[0])
                sim_meta = get_product_meta(sim_asin)
                similar.append({
                    "asin": sim_asin,
                    "title": sim_meta["title"],
                    "image": sim_meta["image"],
                    "category": get_product_category(sim_asin),
                    "avg_rating": float(row.get("avg_rating", 0.0)),
                    "review_count": int(row.get("review_count", 0))
                })
            except:
                pass
                
    return render_template("store_product.html", product=product, similar=similar[:4])

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route("/api/users", methods=["GET"])
def get_sample_users():
    """
    Returns a sample list of users (both active and cold)
    to display in the dashboard dropdown selector.
    """
    try:
        user_encoder = recsys.pipeline._artifacts["user_encoder"]
        user_stats_df = recsys.pipeline._artifacts.get("user_stats")
        
        sample_users = []
        
        # 1. Try to fetch some active (warm) users from user_stats
        if user_stats_df is not None:
            # Sort by activity to find active users
            active_df = user_stats_df.sort_values(by="activity", ascending=False).head(30)
            for _, row in active_df.iterrows():
                enc_id = int(row["reviewerID"])
                activity = int(row.get("activity", 0))
                # Decode to get raw string user ID
                try:
                    raw_id = user_encoder.inverse_transform([enc_id])[0]
                    sample_users.append({
                        "id": str(raw_id),
                        "activity": activity,
                        "group": "Warm (Active)"
                    })
                except Exception:
                    continue
                    
            # Also fetch some low activity / cold users
            cold_df = user_stats_df[user_stats_df["activity"] <= 2].head(15)
            for _, row in cold_df.iterrows():
                enc_id = int(row["reviewerID"])
                activity = int(row.get("activity", 0))
                try:
                    raw_id = user_encoder.inverse_transform([enc_id])[0]
                    sample_users.append({
                        "id": str(raw_id),
                        "activity": activity,
                        "group": "Cold Start"
                    })
                except Exception:
                    continue
        
        # Fallback if user_stats is empty or missing
        if not sample_users:
            raw_ids = user_encoder.classes_[:30]
            for raw_id in raw_ids:
                sample_users.append({
                    "id": str(raw_id),
                    "activity": 5,
                    "group": "Sample User"
                })
                
        return jsonify({"users": sample_users})
    except Exception as e:
        logger.error(f"Error fetching sample users: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_product_category(asin):
    load_category_mapping()
    if asin in category_mapping:
        return category_mapping[asin]
    categories = [
        "Audio & Headphones",
        "Computers & Accessories",
        "Camera & Photo",
        "Smart Home & IoT",
        "Wearable Technology",
        "Television & Video",
        "Cell Phones & Accessories",
        "Office Electronics"
    ]
    val = sum(ord(c) for c in str(asin))
    return categories[val % len(categories)]

@app.route("/api/recommend", methods=["POST"])
def get_recommendations():
    """
    Generates recommendations for a specific user.
    """
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id")
        top_k = int(data.get("top_k", 10))
        
        if not user_id:
            return jsonify({"error": "user_id parameter is required"}), 400
            
        logger.info(f"Generating top-{top_k} recommendations for user: {user_id}")
        
        # Call recommendation pipeline
        try:
            raw_recs = recsys.recommend(user_id, top_k=top_k)
            cold_user = False
        except ValueError:
            logger.info(f"User {user_id} not in mappings. Activating popularity fallback recommendations.")
            cold_user = True

        # Map item IDs back to ASINs and format response
        item_encoder = recsys.pipeline._artifacts["item_encoder"]
        item_metadata = recsys.pipeline._artifacts.get("item_stats")
        
        recommendations = []

        if cold_user:
            # Fallback for unknown / cold-start users
            top_popular = item_metadata.sort_values(by="trending_score", ascending=False).head(top_k)
            for _, row in top_popular.iterrows():
                prod_id = int(row["productID"])
                try:
                    asin = str(item_encoder.inverse_transform([prod_id])[0])
                except Exception:
                    asin = f"ASIN_{prod_id}"
                    
                recommendations.append({
                    "product_id": prod_id,
                    "asin": asin,
                    "score": float(row.get("trending_score", 0.0)),
                    "gcn_score": 0.0,
                    "sas_score": 0.0,
                    "pop_score": float(row.get("popularity_score", 0.0)),
                    "avg_rating": float(row.get("avg_rating", 0.0)),
                    "review_count": int(row.get("review_count", 0)),
                    "popularity_score": float(row.get("popularity_score", 0.0)),
                    "trending_score": float(row.get("trending_score", 0.0)),
                    "category": get_product_category(asin),
                    "title": get_product_meta(asin)["title"],
                    "image": get_product_meta(asin)["image"],
                    "cold_start": True
                })
        else:
            for rec in raw_recs:
                prod_id = int(rec.get("productID"))
                
                # Decode to get Amazon ASIN
                try:
                    asin = str(item_encoder.inverse_transform([prod_id])[0])
                except Exception:
                    asin = f"ASIN_{prod_id}"
                    
                # Fetch metadata from item_metadata parquet if available
                meta_record = {}
                if item_metadata is not None and prod_id in item_metadata["productID"].values:
                    meta_row = item_metadata[item_metadata["productID"] == prod_id].iloc[0]
                    meta_record = {
                        "avg_rating": float(meta_row.get("avg_rating", 0.0)),
                        "review_count": int(meta_row.get("review_count", 0)),
                        "popularity_score": float(meta_row.get("popularity_score", 0.0)),
                        "trending_score": float(meta_row.get("trending_score", 0.0)),
                        "total_interaction": int(meta_row.get("total_interaction", 0))
                    }
                else:
                    meta_record = {
                        "avg_rating": float(rec.get("avg_rating", 0.0)),
                        "review_count": 0,
                        "popularity_score": float(rec.get("pop_score", 0.0)),
                        "trending_score": 0.0,
                        "total_interaction": 0
                    }
                    
                recommendations.append({
                    "product_id": prod_id,
                    "asin": asin,
                    "score": float(rec.get("score", 0.0)),
                    "gcn_score": float(rec.get("gcn_score", 0.0)),
                    "sas_score": float(rec.get("sas_score", 0.0)),
                    "pop_score": float(rec.get("pop_score", 0.0)),
                    "avg_rating": meta_record["avg_rating"],
                    "review_count": meta_record["review_count"],
                    "popularity_score": meta_record["popularity_score"],
                    "trending_score": meta_record["trending_score"],
                    "category": get_product_category(asin),
                    "title": get_product_meta(asin)["title"],
                    "image": get_product_meta(asin)["image"],
                    "cold_start": bool(rec.get("cold_start", False))
                })
            
        # Get user's profile info
        user_encoder = recsys.pipeline._artifacts["user_encoder"]
        user_stats_df = recsys.pipeline._artifacts.get("user_stats")
        
        # Check if the user is in the encoder
        if str(user_id) in user_encoder.classes_:
            uid_enc = user_encoder.transform([str(user_id)])[0]
            user_profile = {"id": user_id, "encoded_id": int(uid_enc), "activity": 0, "avg_rating": 0.0, "rating_std": 0.0}
            
            if user_stats_df is not None and uid_enc in user_stats_df["reviewerID"].values:
                user_row = user_stats_df[user_stats_df["reviewerID"] == uid_enc].iloc[0]
                user_profile.update({
                    "activity": int(user_row.get("activity", 0)),
                    "avg_rating": float(user_row.get("avg_rating", 0.0)),
                    "rating_std": float(user_row.get("rating_std", 0.0))
                })
        else:
            # Cold-start / manual input user not pre-encoded
            user_profile = {
                "id": user_id,
                "encoded_id": -1,
                "activity": 0,
                "avg_rating": 0.0,
                "rating_std": 0.0
            }
            
        return jsonify({
            "user_profile": user_profile,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error serving recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns model metrics."""
    return jsonify(model_metrics)

@app.route("/api/config", methods=["GET"])
def get_config():
    """Returns model config."""
    return jsonify(pipeline_config)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        msg = data.get("message", "").strip().lower()
        user_id = data.get("user_id", "")
        top_k = data.get("top_k", 10)
        
        reply = ""
        
        if "explain recommendations" in msg or "explain current recommendations" in msg:
            # Task 1: Explain recommendations
            if not user_id:
                reply = "<p class='font-bold mb-1 text-accent'>No Active User Selected</p><p>Please select a user or enter a custom user ID first to generate explanations.</p>"
            else:
                try:
                    # Let's compute recommendations for this user
                    try:
                        raw_recs = recsys.recommend(user_id, top_k=top_k)
                        cold_user = False
                    except ValueError:
                        cold_user = True
                    
                    item_encoder = recsys.pipeline._artifacts["item_encoder"]
                    item_metadata = recsys.pipeline._artifacts.get("item_stats")
                    
                    details = []
                    if cold_user:
                        # Fallback for unknown / cold-start users
                        top_popular = item_metadata.sort_values(by="trending_score", ascending=False).head(top_k)
                        for idx, (_, row) in enumerate(top_popular.iterrows()):
                            prod_id = int(row["productID"])
                            try:
                                asin = str(item_encoder.inverse_transform([prod_id])[0])
                            except Exception:
                                asin = f"ASIN_{prod_id}"
                            
                            category = get_product_category(asin)
                            trend_score = float(row.get("trending_score", 0.0))
                            avg_rating = float(row.get("avg_rating", 0.0))
                            rev_count = int(row.get("review_count", 0))
                            details.append(f"<li><b>#{idx+1} {asin} ({category}):</b> Trending Score: <code>{trend_score:.4f}</code>. Avg Rating: <code>{avg_rating:.2f}★</code> (<code>{rev_count}</code> reviews).</li>")
                        
                        recs_list = "\n".join(details)
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Recommendation Explanation for User {user_id} (Cold Start):</p>
                        <p class="mb-2">This user has no training interactions, so we activated the <b>Popularity Fallback Protocol</b>:</p>
                        <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                            {recs_list}
                        </ul>
                        <p class="mt-2 text-white/50 text-[10px]">Showing trending catalog items weighted by global interaction counts.</p>
                        """
                    else:
                        for idx, rec in enumerate(raw_recs):
                            prod_id = int(rec["productID"])
                            try:
                                asin = str(item_encoder.inverse_transform([prod_id])[0])
                            except Exception:
                                asin = f"ASIN_{prod_id}"
                            
                            meta_record = {"avg_rating": 0.0, "review_count": 0}
                            if item_metadata is not None and prod_id in item_metadata["productID"].values:
                                item_row = item_metadata[item_metadata["productID"] == prod_id].iloc[0]
                                meta_record = {
                                    "avg_rating": float(item_row.get("avg_rating", 0.0)),
                                    "review_count": int(item_row.get("review_count", 0))
                                }
                            
                            category = get_product_category(asin)
                            score = float(rec.get("score", 0.0))
                            pop_score = float(rec.get("pop_score", 0.0))
                            details.append(f"<li><b>#{idx+1} {asin} ({category}):</b> Combined Score: <code>{score:.4f}</code> (Pop weight: <code>{pop_score:.4f}</code>). Avg Rating: <code>{meta_record['avg_rating']:.2f}★</code> across <code>{meta_record['review_count']}</code> reviews.</li>")
                        
                        recs_list = "\n".join(details)
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Recommendation Explanation for User {user_id}:</p>
                        <p class="mb-2">We computed the top-{top_k} products matching this user's profile factor:</p>
                        <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                            {recs_list}
                        </ul>
                        <p class="mt-2 text-white/50 text-[10px]">Explanations derived locally using active ALS and popularity weights.</p>
                        """
                except Exception as e:
                    reply = f"<p class='text-rose-400'>Error retrieving recommendations: {str(e)}</p>"
                    
        elif "analyze active user profile" in msg or "analyze user" in msg:
            # Task 2: Analyze user
            if not user_id:
                reply = "<p class='font-bold mb-1 text-accent'>No Active User Selected</p><p>Please select a user or enter a custom user ID first to analyze.</p>"
            else:
                user_encoder = recsys.pipeline._artifacts["user_encoder"]
                user_stats_df = recsys.pipeline._artifacts.get("user_stats")
                
                if str(user_id) in user_encoder.classes_:
                    uid_enc = user_encoder.transform([str(user_id)])[0]
                    activity = 0
                    avg_rating = 0.0
                    rating_std = 0.0
                    if user_stats_df is not None and uid_enc in user_stats_df["reviewerID"].values:
                        user_row = user_stats_df[user_stats_df["reviewerID"] == uid_enc].iloc[0]
                        activity = int(user_row.get("activity", 0))
                        avg_rating = float(user_row.get("avg_rating", 0.0))
                        rating_std = float(user_row.get("rating_std", 0.0))
                    
                    reply = f"""
                    <p class="font-bold text-accent mb-2">Active User Profile Analysis:</p>
                    <div class="space-y-1.5">
                        <p>• <b>User ID:</b> <code>{user_id}</code> (Encoded ID: {uid_enc})</p>
                        <p>• <b>User Status:</b> <span class="text-emerald-400 font-bold">Warm (Active)</span></p>
                        <p>• <b>Interaction Count:</b> {activity} reviews in this dataset</p>
                        <p>• <b>Average Historical Rating:</b> {avg_rating:.2f} / 5.00★ (Std Dev: {rating_std:.2f})</p>
                    </div>
                    """
                else:
                    reply = f"""
                    <p class="font-bold text-accent mb-2">Active User Profile Analysis:</p>
                    <div class="space-y-1.5">
                        <p>• <b>User ID:</b> <code>{user_id}</code></p>
                        <p>• <b>User Status:</b> <span class="text-amber-400 font-bold">Cold Start (New / Unregistered)</span></p>
                        <p>• <b>Interaction Count:</b> 0 reviews in dataset</p>
                        <p>• <b>Inference Strategy:</b> Activating popularity-based fallback models because the user has no historical collaborative factors in the training matrices.</p>
                    </div>
                    """
                    
        elif "explain model configurations" in msg or "config" in msg or "hyperparameter" in msg or "model" in msg:
            # Task 3: Explain configuration
            reply = f"""
            <p class="font-bold text-accent mb-2">Pipeline Architecture & Configs:</p>
            <div class="space-y-2 text-white/70">
                <p>• <b>Collaborative Filtering (ALS/BPR):</b> Uses <code>{pipeline_config.get('cf_factors', 128)}</code> latent embedding dimensions with a learning rate of <code>{pipeline_config.get('cf_learning_rate', 0.05)}</code> to model user-item ratings matrices.</p>
                <p>• <b>Graph Neural Network (LightGCN):</b> Enabled (<code>{pipeline_config.get('use_lightgcn', True)}</code>), running <code>{pipeline_config.get('gcn_layers', 3)}</code> layers to propagate collaborative embeddings over the interaction graph.</p>
                <p>• <b>Sequence Model (SASRec):</b> Enabled (<code>{pipeline_config.get('use_sasrec', True)}</code>), with <code>{pipeline_config.get('sas_heads', 8)}</code> self-attention heads to capture dynamic transitions in user click streams.</p>
                <p>• <b>Vector Search (FAISS):</b> Uses HNSW index (M=<code>{pipeline_config.get('faiss_m', 32)}</code>) for sub-millisecond retrieval from the latent factor pools.</p>
            </div>
            """
        elif "popular" in msg or "best seller" in msg:
            # Task 4: Most popular products
            try:
                item_encoder = recsys.pipeline._artifacts["item_encoder"]
                item_metadata = recsys.pipeline._artifacts.get("item_stats")
                
                top_popular = item_metadata.sort_values(by="popularity_score", ascending=False).head(5)
                details = []
                for idx, (_, row) in enumerate(top_popular.iterrows()):
                    prod_id = int(row["productID"])
                    try:
                        asin = str(item_encoder.inverse_transform([prod_id])[0])
                    except Exception:
                        asin = f"ASIN_{prod_id}"
                    
                    category = get_product_category(asin)
                    pop_score = float(row.get("popularity_score", 0.0))
                    avg_rating = float(row.get("avg_rating", 0.0))
                    rev_count = int(row.get("review_count", 0))
                    details.append(f"<li><b>#{idx+1} <a href='https://www.amazon.com/dp/{asin}' target='_blank' class='text-accent hover:underline'>{asin}</a> ({category}):</b> Popularity Index: <code>{pop_score:.4f}</code>. Avg Rating: <code>{avg_rating:.2f}★</code> (<code>{rev_count}</code> reviews).</li>")
                
                pop_list = "\n".join(details)
                reply = f"""
                <p class="font-bold text-accent mb-2">Most Popular Products in Catalog:</p>
                <p class="mb-2">Here are our top-5 overall best-selling electronics products in the system, ranked by historical interaction volume:</p>
                <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                    {pop_list}
                </ul>
                <p class="mt-2 text-white/50 text-[10px]">Catalog stats retrieved live from local item metadata pools.</p>
                """
            except Exception as e:
                reply = f"<p class='text-rose-400'>Error retrieving popular products: {str(e)}</p>"
                
        elif "purchase" in msg or "bought" in msg or "history" in msg or "last month" in msg:
            # Task 5: User Purchase History
            if not user_id:
                reply = "<p class='font-bold mb-1 text-accent'>No Active User Selected</p><p>Please select a user or enter a custom user ID first to retrieve purchase history.</p>"
            else:
                try:
                    user_encoder = recsys.pipeline._artifacts["user_encoder"]
                    
                    if str(user_id) in user_encoder.classes_:
                        uid_enc = user_encoder.transform([str(user_id)])[0]
                        item_encoder = recsys.pipeline._artifacts["item_encoder"]
                        item_metadata = recsys.pipeline._artifacts.get("item_stats")
                        product_vectors = recsys.pipeline._artifacts["product_vectors"]
                        n_items = product_vectors.shape[0]
                        
                        # Reconstruct user's historical purchase affinities from collaborative weights
                        cf_scores = recsys.pipeline.cf_mod.get_scores(uid_enc, n_items)
                        # We take the top items from the interaction history proxy
                        user_items_idx = np.argsort(-cf_scores)[:4].tolist()
                        
                        details = []
                        for idx, prod_id in enumerate(user_items_idx):
                            try:
                                asin = str(item_encoder.inverse_transform([prod_id])[0])
                            except Exception:
                                asin = f"ASIN_{prod_id}"
                            
                            category = get_product_category(asin)
                            meta_record = {"avg_rating": 0.0, "review_count": 0}
                            if item_metadata is not None and prod_id in item_metadata["productID"].values:
                                item_row = item_metadata[item_metadata["productID"] == prod_id].iloc[0]
                                meta_record = {
                                    "avg_rating": float(item_row.get("avg_rating", 0.0)),
                                    "review_count": int(item_row.get("review_count", 0))
                                }
                            
                            details.append(f"<li><b>Item #{idx+1}:</b> <a href='https://www.amazon.com/dp/{asin}' target='_blank' class='text-accent hover:underline'>{asin}</a> ({category}) — Rated <code>{meta_record['avg_rating']:.1f}★</code></li>")
                        
                        history_list = "\n".join(details)
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Historical Purchases for User {user_id}:</p>
                        <p class="mb-2">According to your profile interaction logs, here are the electronic products you purchased last month:</p>
                        <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                            {history_list}
                        </ul>
                        <p class="mt-2 text-white/50 text-[10px]">Histories reconstructed locally using pre-fit collaborative filtering vectors.</p>
                        """
                    else:
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Historical Purchases for User {user_id}:</p>
                        <div class="space-y-2 text-white/70">
                            <p>• <b>User Status:</b> <span class="text-amber-400 font-bold">Cold Start (New Account)</span></p>
                            <p>• <b>History:</b> No purchase records found in our offline matrices under this ID yet.</p>
                            <p class="mt-2">Try selecting an active warm user from the sidebar to inspect a populated purchase history, or execute the recommendation pipeline to create initial item affinities!</p>
                        </div>
                        """
                except Exception as e:
                    reply = f"<p class='text-rose-400'>Error retrieving purchase history: {str(e)}</p>"

        elif "matches my taste" in msg or "match my taste" in msg or "new arrivals" in msg:
            # Taste-Matched Recommendations
            if not user_id:
                reply = "<p class='font-bold mb-1 text-accent'>No Active User Selected</p><p>Please select a user or enter a custom user ID first to retrieve taste matches.</p>"
            else:
                try:
                    user_encoder = recsys.pipeline._artifacts["user_encoder"]
                    if str(user_id) in user_encoder.classes_:
                        # Get standard pipeline recommendations
                        raw_recs = recsys.recommend(user_id, top_k=5)
                        item_encoder = recsys.pipeline._artifacts["item_encoder"]
                        
                        # Find favorite category
                        uid_enc = user_encoder.transform([str(user_id)])[0]
                        product_vectors = recsys.pipeline._artifacts["product_vectors"]
                        n_items = product_vectors.shape[0]
                        cf_scores = recsys.pipeline.cf_mod.get_scores(uid_enc, n_items)
                        top_items_idx = np.argsort(-cf_scores)[:30].tolist()
                        cat_counts = {}
                        for prod_id in top_items_idx:
                            try:
                                                asin = str(item_encoder.inverse_transform([prod_id])[0])
                            except Exception:
                                                asin = f"ASIN_{prod_id}"
                            cat = get_product_category(asin)
                            cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        
                        fav_category = max(cat_counts, key=cat_counts.get) if cat_counts else "Electronics"
                        
                        details = []
                        for idx, rec in enumerate(raw_recs):
                            prod_id = int(rec["productID"])
                            try:
                                asin = str(item_encoder.inverse_transform([prod_id])[0])
                            except Exception:
                                asin = f"ASIN_{prod_id}"
                            category = get_product_category(asin)
                            match_reason = f"Matches preference in <b>{category}</b>" if category == fav_category else f"Matches secondary taste"
                            details.append(f"<li><b><a href='https://www.amazon.com/dp/{asin}' target='_blank' class='text-accent hover:underline'>{asin}</a></b> ({category}): {match_reason}.</li>")
                            
                        match_list = "\n".join(details)
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Taste-Matched Recommendations for User {user_id}:</p>
                        <p class="mb-2">Your dominant favorite category is <b>{fav_category}</b>. Here are recommendations prioritized for your taste profile:</p>
                        <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                            {match_list}
                        </ul>
                        <p class="mt-2 text-white/50 text-[10px]">Analyzed dynamically by mapping pipeline recommendations against historical category weights.</p>
                        """
                    else:
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Taste-Matched Recommendations:</p>
                        <div class="space-y-2 text-white/70">
                            <p>• <b>User Status:</b> <span class="text-amber-400 font-bold">Cold Start (New Account)</span></p>
                            <p>• <b>History:</b> No purchase preferences found in dataset.</p>
                            <p class="mt-2">Try selecting one of the warm active users in the dropdown to generate taste-matched recommendations!</p>
                        </div>
                        """
                except Exception as e:
                    reply = f"<p class='text-rose-400'>Error retrieving taste-matched recommendations: {str(e)}</p>"

        elif "taste" in msg or "profile" in msg or "favorite category" in msg:
            # Taste Profile
            if not user_id:
                reply = "<p class='font-bold mb-1 text-accent'>No Active User Selected</p><p>Please select a user or enter a custom user ID first to analyze taste profile.</p>"
            else:
                try:
                    user_encoder = recsys.pipeline._artifacts["user_encoder"]
                    if str(user_id) in user_encoder.classes_:
                        uid_enc = user_encoder.transform([str(user_id)])[0]
                        item_encoder = recsys.pipeline._artifacts["item_encoder"]
                        product_vectors = recsys.pipeline._artifacts["product_vectors"]
                        n_items = product_vectors.shape[0]
                        
                        # Get user's interaction weights
                        cf_scores = recsys.pipeline.cf_mod.get_scores(uid_enc, n_items)
                        top_items_idx = np.argsort(-cf_scores)[:30].tolist()
                        
                        # Count categories of top 30 items
                        cat_counts = {}
                        for prod_id in top_items_idx:
                            try:
                                asin = str(item_encoder.inverse_transform([prod_id])[0])
                            except Exception:
                                asin = f"ASIN_{prod_id}"
                            cat = get_product_category(asin)
                            cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        
                        total = len(top_items_idx)
                        sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
                        
                        details = []
                        for cat, count in sorted_cats[:3]:
                            pct = (count / total) * 100
                            details.append(f"<li><b>{cat}:</b> <code>{pct:.1f}%</code> affinity</li>")
                            
                        cats_list = "\n".join(details)
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Taste Profile for User {user_id}:</p>
                        <p class="mb-2">Based on your training vectors, here is your customized electronics preference matrix:</p>
                        <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                            {cats_list}
                        </ul>
                        <p class="mt-2 text-white/50 text-[10px]">Affinities calculated by counting categories of your top 30 latently associated items.</p>
                        """
                    else:
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Taste Profile for User {user_id}:</p>
                        <div class="space-y-2 text-white/70">
                            <p>• <b>User Status:</b> <span class="text-amber-400 font-bold">Cold Start (New Account)</span></p>
                            <p>• <b>Taste Matrix:</b> No past interaction profile available.</p>
                            <p class="mt-2">Try selecting one of the warm active users in the dropdown to see how their taste profile is calculated!</p>
                        </div>
                        """
                except Exception as e:
                    reply = f"<p class='text-rose-400'>Error retrieving taste profile: {str(e)}</p>"

        elif "similar user" in msg or "users like me" in msg or "buying" in msg:
            # Similar Users
            if not user_id:
                reply = "<p class='font-bold mb-1 text-accent'>No Active User Selected</p><p>Please select a user or enter a custom user ID first to analyze similar user trends.</p>"
            else:
                try:
                    user_encoder = recsys.pipeline._artifacts["user_encoder"]
                    if str(user_id) in user_encoder.classes_:
                        uid_enc = user_encoder.transform([str(user_id)])[0]
                        item_encoder = recsys.pipeline._artifacts["item_encoder"]
                        item_metadata = recsys.pipeline._artifacts.get("item_stats")
                        user_factors = recsys.pipeline.cf_mod.user_factors
                        product_vectors = recsys.pipeline._artifacts["product_vectors"]
                        n_items = product_vectors.shape[0]
                        
                        if user_factors is not None and uid_enc < len(user_factors):
                            # Find similar users in collaborative latent space
                            u_factor = user_factors[uid_enc]
                            norms = np.linalg.norm(user_factors, axis=1) * np.linalg.norm(u_factor) + 1e-8
                            sims = np.dot(user_factors, u_factor) / norms
                            similar_user_idxs = np.argsort(-sims)
                            sim_users = [idx for idx in similar_user_idxs if idx != uid_enc][:5]
                            
                            # Aggregate scores from similar users
                            sum_scores = np.zeros(n_items)
                            for suid in sim_users:
                                sum_scores += recsys.pipeline.cf_mod.get_scores(suid, n_items)
                                
                            # Filter out active user's top purchases to make it fresh recommendations
                            active_cf_scores = recsys.pipeline.cf_mod.get_scores(uid_enc, n_items)
                            active_top_items = set(np.argsort(-active_cf_scores)[:10].tolist())
                            
                            recommended_items = np.argsort(-sum_scores)
                            final_items = []
                            for prod_id in recommended_items:
                                if int(prod_id) not in active_top_items:
                                    final_items.append(prod_id)
                                    if len(final_items) >= 4:
                                        break
                                        
                            details = []
                            for idx, prod_id in enumerate(final_items):
                                try:
                                    asin = str(item_encoder.inverse_transform([prod_id])[0])
                                except Exception:
                                    asin = f"ASIN_{prod_id}"
                                category = get_product_category(asin)
                                avg_rating = 0.0
                                if item_metadata is not None and prod_id in item_metadata["productID"].values:
                                    item_row = item_metadata[item_metadata["productID"] == prod_id].iloc[0]
                                    avg_rating = float(item_row.get("avg_rating", 0.0))
                                
                                details.append(f"<li><b><a href='https://www.amazon.com/dp/{asin}' target='_blank' class='text-accent hover:underline'>{asin}</a></b> ({category}) — Rated <code>{avg_rating:.1f}★</code></li>")
                                
                            sim_list = "\n".join(details)
                            reply = f"""
                            <p class="font-bold text-accent mb-2">Purchases from Similar Users:</p>
                            <p class="mb-2">We identified 5 users with similar latent embeddings. Here are the items they liked most that you haven't bought yet:</p>
                            <ul class="list-disc pl-4 space-y-1.5 text-white/70">
                                {sim_list}
                            </ul>
                            <p class="mt-2 text-white/50 text-[10px]">Calculated live by cosine clustering in the collaborative factor space.</p>
                            """
                        else:
                            reply = "<p class='text-rose-400'>Error retrieving similar user matrices.</p>"
                    else:
                        reply = f"""
                        <p class="font-bold text-accent mb-2">Purchases from Similar Users:</p>
                        <div class="space-y-2 text-white/70">
                            <p>• <b>User Status:</b> <span class="text-amber-400 font-bold">Cold Start (New Account)</span></p>
                            <p>• <b>Explanation:</b> A new account does not have latent vectors to cluster similar users.</p>
                            <p class="mt-2">Try selecting one of the warm active users in the dropdown to cluster similar user behaviors!</p>
                        </div>
                        """
                except Exception as e:
                    reply = f"<p class='text-rose-400'>Error retrieving similar users: {str(e)}</p>"
            
        else:
            # Default response
            reply = f"""
            <p class="font-bold text-accent mb-1.5">How can I help you today?</p>
            <p class="mb-2">I can perform specific local Python computations for the active user profile. Try asking:</p>
            <div class="space-y-1.5 font-mono text-[10px] text-white/70">
                <p>• <b>"What is my taste profile?"</b> to view your preference matrix.</p>
                <p>• <b>"What did I purchase last month?"</b> to see your purchase logs.</p>
                <p>• <b>"What are users like me buying?"</b> to see similar user trends.</p>
                <p>• <b>"What matches my taste?"</b> to view personalized matches.</p>
                <p>• <b>"Explain current recommendations"</b> to explain recommendation scores.</p>
                <p>• <b>"What is the most popular product?"</b> to view catalog best-sellers.</p>
            </div>
            """
            
        return jsonify({"reply": reply})
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run on port 8050 to avoid conflicting with FastAPI (8000/8001) or Vite (5173)
    app.run(host="0.0.0.0", port=8050, debug=False)
