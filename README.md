# Amazon Electronics Recommendation & Analytics Dashboard

Welcome to the **Amazon Electronics Recommendation & Interactive Analytics Dashboard**. This project is a production-grade, local-first recommender system paired with a cyberpunk glassmorphic web dashboard (styled under the *Organic Tech / Clinical Boutique* design system). 

It loads pre-trained multi-layered collaborative filtering and graph neural network embeddings, leverages approximate nearest neighbors (FAISS) for sub-millisecond retrieval, serves predictions via a local Flask backend, and features an interactive **AI Chatbot Assistant** running customized local Python computations.

---

## 🚀 Key Features

*   **Multi-Model Hybrid Pipeline**: Recommends items using collaborative filtering (implicit ALS), Graph Neural Networks (LightGCN graph embedding propagation), and sequence clickstreams (SASRec self-attention).
*   **Approximate Nearest Neighbors**: Uses an HNSW vector index (`faiss.index`) to fetch matches from high-dimensional space in less than a millisecond.
*   **Clinical Boutique Dashboard UI**: Designed with a sleek, dark glassmorphic layout, using custom CSS and SVG noise overlays, tailored typography (Plus Jakarta Sans + Outfit), and dynamic micro-animations.
*   **Interactive AI Assistant**: A local chatbot that computes user taste profiles, cosign-clusters similar user trends, highlights taste match rationale, traces recommended scores, and handles cold starts.
*   **Warm & Cold Start Safeguards**: Gracefully falls back to overall best sellers and popular category items if an unknown or new User ID is selected.
*   **Dynamic SVG Charting**: Renders live Precision-Recall and model metrics curves using responsive inline SVGs.

---

## 🛠️ Machine Learning Pipeline Architecture

The recommendation engine is built as a 9-layer pipeline structured to capture both static collaborative preferences and sequential user intentions:

1.  **Dataset Layer**: Parses historical user-item interactions, tracks ratings, and maps raw Amazon ASINs and User IDs into continuous integer indexes.
2.  **Embedding Layer**: Initializes 128-dimensional dense latent vectors representing users and products.
3.  **Collaborative Filtering (ALS/BPR)**: Leverages Alternating Least Squares (ALS) and Bayesian Personalized Ranking (BPR) matrices to pre-calculate user-item affinity factors.
4.  **Graph Neural Network (LightGCN)**: Propagates collaborative embedding vectors across user-item bipartite interaction graphs, capturing high-order neighborhood structures.
5.  **Sequence Modeling (SASRec)**: Utilizes a self-attention transformer network to capture temporal, short-term trends from user click sequences.
6.  **Vector Search (FAISS)**: Serializes product embedding coordinates into a Hierarchical Navigable Small World (HNSW) index for fast approximate nearest neighbor lookups.
7.  **Scoring & Ranking**: Blends collaborative vectors, sequence factors, and catalog popularity indexes to yield a final sorted recommendation array.
8.  **Metadata Resolution**: Intercepts item index arrays to resolve titles, average ratings, review counts, and categorizations from parquet metadata databases.
9.  **API Gateway**: Exposes routes enabling web clients to request recommendations, inspect configurations, retrieve metrics, and converse with the chatbot.

---

## 📂 Project Directory Structure Explained

Here is a guide to what each file and folder does in this repository:

```bash
deploy_bundle/
│
├── flask_app.py               # The main Flask backend server. Handles routing and chatbot computations.
├── pipeline.py                # Definition of the multi-model RecSys pipeline structures & ranking.
├── inference.py               # RecSysInference helper for loading models, encoder mapping, and recommending.
│
├── templates/
│   └── index.html             # The frontend dashboard interface (HTML5, Tailwind CSS, Lucide icons, JS).
│
├── venv/                      # Python virtual environment containing libraries and dependencies.
│
├── requirements.txt           # File containing lists of required python packages.
│
├── Data Assets (Pre-trained & Pre-computed):
│   ├── cf_model.pkl           # Trained implicit ALS/BPR model weights.
│   ├── product_vectors.npy    # Serialized item latent factor embeddings.
│   ├── faiss.index            # FAISS index loaded into RAM for vector searches.
│   ├── artifacts.pkl          # Pickled pipeline metadata, user/item encoders, and parameters.
│   ├── config.json            # Model parameters (factor sizes, layer sizes, and FAISS options).
│   ├── metrics_v5.json        # Pre-calculated test evaluation performance metrics.
│   ├── item_mapping.pkl       # Map dictionary converting product IDs to original Amazon ASINs.
│   ├── user_mapping.pkl       # Map dictionary converting user indices to original User IDs.
│   ├── item_metadata.parquet  # Parquet data sheet detailing item average ratings and review counts.
│   └── item_categories.json   # 490k+ item-to-category dictionary mappings extracted from Amazon datasets.
│
└── Developer/Scratch Scripts:
    ├── app.py                 # Alternate entry point script.
    ├── preprocess.py          # Initial data cleaning script.
    ├── extract_categories.py  # Utility to parse category names from raw datasets.
    └── read.ipynb             # Jupyter Notebook for inspecting pickle weights.
```

---

## 💬 The Offline AI Chatbot Helper

At the bottom-right corner of the dashboard, you will find an **AI Chatbot Assistant**. Unlike generic chatbots, it executes local python computations on the active model caching state to output actual analytical insights for the selected user:

*   **Taste Profile (`"What is my taste profile?"`)**: Scans the user's top 30 latent product affinities, computes category distribution percentages, and renders a tailored preference matrix.
*   **Similar Users (`"What are users like me buying?"`)**: Applies cosine similarity across the user factors matrix to locate the top 5 similar profiles, aggregates their purchased items, filters out products the active user already bought, and lists fresh recommendations.
*   **Taste Matches (`"What matches my taste?"`)**: Cross-references the active recommendation list against the user's primary category to output custom reason codes.
*   **Purchase History (`"What did I purchase last month?"`)**: Extracts past user interactions from pre-fit collaborative metrics.
*   **Hyperparameter Config (`"Explain model configurations"`)**: Explains GNN layers, ALS embedding size, SASRec self-attention heads, and FAISS HNSW configuration factors.
*   **Catalog Best Sellers (`"What is the most popular product?"`)**: Sorts and lists top products by review counts and ratings.

---

## 🚀 Getting Started

Follow these steps to run the interactive dashboard locally:

### 1. Set up the Environment
Open a terminal (e.g., PowerShell on Windows) in the project directory:

```powershell
# Create virtual environment (if not already present)
python -m venv venv

# Activate the virtual environment
venv\Scripts\Activate
```

### 2. Install Dependencies
Install all required libraries including PyTorch, FAISS, Implicit, and Flask:

```powershell
pip install -r requirements.txt
pip install flask
```

### 3. Launch the Server
Start the Flask application backend. The server will construct the popularity metrics, load vectors, parse categories, and bind to port `8050`:

```powershell
python flask_app.py
```

Log outputs should indicate:
```text
INFO:FlaskRecSys:Recommendation model initialized successfully!
INFO:FlaskRecSys:Loaded 498196 custom categories from item_categories.json
 * Running on http://127.0.0.1:8050
```

### 4. Open the Interface
Open your web browser and navigate to:
👉 **[http://localhost:8050](http://localhost:8050)**

Choose an active user ID from the sidebar dropdown (e.g., `ADLVFFE4VBT8`), explore the metrics tabs, or type queries into the AI Chatbot to see live personalization responses in action!
