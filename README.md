# Amazon Electronics Recommendation & E-Commerce Platform

Welcome to the **Amazon Electronics Recommendation System & E-Commerce Platform**. This project is a production-grade, local-first recommender system paired with a realistic, highly dynamic e-commerce storefront and a specialized analytics dashboard.

It loads pre-trained multi-layered collaborative filtering and graph neural network embeddings, leverages approximate nearest neighbors (FAISS) for sub-millisecond retrieval, and serves real-time personalized predictions via a Flask backend.

---

## 🚀 Key Features

*   **Real-World Storefront UI (`/`)**: A fully functional mock e-commerce storefront featuring dynamic carousels, personalized "Recommended for You" sections, and product detail cross-sells, all powered by a beautiful Dark Glassmorphic TailwindCSS design.
*   **Real Amazon Metadata**: Automatically maps ASINs to their real Amazon product titles and high-resolution images via a lightning-fast local JSON cache system, streaming from HuggingFace dataset backups.
*   **Multi-Model Hybrid Pipeline**: Recommends items using collaborative filtering (implicit ALS), Graph Neural Networks (LightGCN graph embedding propagation), and sequence clickstreams (SASRec self-attention).
*   **Developer Analytics Dashboard (`/admin`)**: Designed with a sleek, clinical boutique layout. Features an interactive **AI Chatbot Assistant** running customized local Python computations for taste profiling and dataset querying.
*   **Approximate Nearest Neighbors**: Uses an HNSW vector index (`faiss.index`) to fetch matches from high-dimensional space in less than a millisecond.
*   **Warm & Cold Start Safeguards**: Gracefully falls back to overall best sellers and popular category items if an unknown or new User ID is selected.

---

## 🛠️ Machine Learning Pipeline Architecture

The recommendation engine has been heavily refactored into a clean, modular structure (`src/models/`) acting as a 9-layer pipeline to capture both static collaborative preferences and sequential user intentions:

1.  **Dataset Layer**: Parses historical user-item interactions, tracks ratings, and maps raw Amazon ASINs and User IDs into continuous integer indexes.
2.  **Embedding Layer**: Initializes 128-dimensional dense latent vectors representing users and products.
3.  **Collaborative Filtering (ALS/BPR)**: Pre-calculates user-item affinity factors.
4.  **Graph Neural Network (LightGCN)**: Propagates collaborative embedding vectors across user-item bipartite interaction graphs.
5.  **Sequence Modeling (SASRec)**: Utilizes a self-attention transformer network to capture temporal, short-term trends.
6.  **Vector Search (FAISS)**: Serializes product embedding coordinates into a Hierarchical Navigable Small World (HNSW) index for fast approximate nearest neighbor lookups.
7.  **Scoring & Ranking**: Blends collaborative vectors, sequence factors, and catalog popularity indexes to yield a final sorted recommendation array.
8.  **Metadata Resolution**: Intercepts item index arrays to resolve titles, average ratings, high-resolution image URLs, and categorizations from parquet metadata databases and JSON cache files.
9.  **API Gateway**: Exposes routes enabling web clients to request recommendations, inspect configurations, retrieve metrics, and converse with the chatbot.

---

## 📂 Project Directory Structure Explained

Here is a guide to what each file and folder does in this repository:

```bash
deploy_bundle/
│
├── run.py                     # Entrypoint script to launch the recommendation server.
│
├── config/
│   └── config.json            # Model parameters (factor sizes, layer sizes, and FAISS options).
│
├── data/
│   ├── cf_model.pkl           # Trained implicit ALS/BPR model weights.
│   ├── product_vectors.npy    # Serialized item latent factor embeddings.
│   ├── faiss.index            # FAISS index loaded into RAM for vector searches.
│   ├── artifacts.pkl          # Pickled pipeline metadata, user/item encoders, and parameters.
│   ├── item_mapping.pkl       # Map dictionary converting product IDs to original Amazon ASINs.
│   ├── user_mapping.pkl       # Map dictionary converting user indices to original User IDs.
│   ├── item_metadata_cache.json # High-speed cache for real Amazon Product Titles and Image URLs.
│   ├── item_metadata.parquet  # Parquet data sheet detailing item average ratings and review counts.
│   └── item_categories.json   # 490k+ item-to-category dictionary mappings.
│
├── src/
│   ├── features/
│   │   ├── preprocess.py      # Initial data cleaning script.
│   │   └── extract_categories.py # Utility to parse category names from raw datasets.
│   │
│   ├── models/
│   │   ├── config.py          # Configuration settings class (V5Config).
│   │   ├── data_layer.py      # Data ingestion & FeatureEngineeringLayer.
│   │   ├── components.py      # Model subcomponents (ALS/BPR, LightGCN, SASRec, FAISS).
│   │   ├── ranking.py         # CandidateGenerator, Ranker (LambdaMART LTR), PostProcessor.
│   │   ├── service.py         # RecommendationService and Evaluator.
│   │   ├── pipeline.py        # PipelineV5 coordinator & sub-module facade.
│   │   └── inference.py       # RecSysInference helper for backend API endpoints.
│   │
│   └── backend/
│       ├── flask_app.py       # Main Flask server code. Serves endpoints, storefront routes, and chatbot logic.
│       └── app.py             # FastAPI backend (alternative API framework).
│
├── templates/
│   ├── store_base.html        # Base Jinja layout for the storefront.
│   ├── store_home.html        # Main e-commerce landing page with personalized carousels.
│   ├── store_product.html     # Product Detail Page (PDP) with similarity cross-sells.
│   └── index.html             # The developer analytics dashboard UI template (Served at /admin).
│
├── venv/                      # Python virtual environment containing libraries and dependencies.
└── requirements.txt           # File containing lists of required python packages.
```

---

## 🚀 Getting Started

Follow these steps to run the interactive storefront and dashboard locally:

### 1. Set up the Environment
Open a terminal (e.g., PowerShell on Windows) in the project directory:

```powershell
# Create virtual environment (if not already present)
python -m venv venv

# Activate the virtual environment
venv\Scripts\Activate
```

### 2. Install Dependencies
Install all required libraries:

```powershell
pip install -r requirements.txt
pip install flask requests pandas
```

### 3. Launch the Server
Start the application backend via the main `run.py` script. The server will automatically load the machine learning vector indexes and cache real Amazon product names:

```powershell
python run.py
```

### 4. Explore the Interfaces
Open your web browser and navigate to:

*   👉 **Storefront UI**: [http://localhost:8050/](http://localhost:8050/)
    *   Test personalized recommendations by clicking the "Sign In" dropdown at the top right and selecting an active user profile.
*   👉 **Analytics & AI Dashboard**: [http://localhost:8050/admin](http://localhost:8050/admin)
    *   Analyze precision/recall metrics, view interactive distribution charts, and talk to the personalized AI Chatbot assistant.
