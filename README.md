<div align="center">
  
# 🛡️ BitcoinGuard AI
**Enterprise-Grade Bitcoin Fraud & AML Detection Platform**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bitcoguard-gsukuu5ghamedqjgfxysqr.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**BitcoinGuard AI** is an advanced, offline-first compliance and transaction monitoring dashboard. It is engineered to detect anomalous behavioral topologies—such as Ponzi schemes, Darknet market clustering, Ransomware routing, and CoinJoin mixers—across complex Bitcoin network structures.

---

## 🌟 Live Cloud Deployment
**[Access the Live Production Platform Here](https://bitcoguard-gsukuu5ghamedqjgfxysqr.streamlit.app/)**

*(Note: The live cloud deployment showcases the complete Machine Learning Ensemble, Network Topologies, and SHAP Explainability logic. However, because the native **AI Investigator** relies on a massive local LLM instance via Ollama to preserve data privacy, the RAG chat feature shows a warning in the public cloud. All other core analytical engines are fully functional online!)*

---

## ✨ Core Platform Architecture

BitcoinGuard AI is broken down into four massive engines working in tandem to expose anomalous activity.

### 1. 🖥️ The Fintech UI
A modern, immersive dark-mode interface built on **Streamlit** (v1.36+). It utilizes Streamlit's native multi-page routing and custom injected Glassmorphism CSS components, forcing a pristine `#e0e6ed` typography across `#0e1117` panels for maximum data legibility.

### 2. 🧠 Ensemble Machine Learning Engine
Real-time cross-validation scoring combining the inference of multiple gradient boosting frameworks.
- **XGBoost, LightGBM, and CatBoost**: Tuned symmetrically with `.json`, `.txt`, and `.cbm` parameter boundaries dynamically loaded at runtime via `@st.cache_resource`. 
- **Weighted Consensus**: A soft-voting layer resolves internal model disagreements to predict the absolute risk threshold of a transaction (Minimal, Low, Medium, High, Critical).

### 3. 🔍 Cognitive SHAP Explainer
Breaking down mathematical inference weights into absolute logic.
- Analyzes all 166 evaluation nodes dynamically. 
- Distinguishes visually between absolute local features (fees/timing logic) vs 1-hop aggregated structural graphs, mapping out why the ensemble triggered an alert in plain visual layers.

### 4. 🕵️ RAG Copilot Investigator
An embedded Agentic LLM pipeline built directly into the UI.
- **ChromaDB Vector Local Search**: Loads FinCEN SAR rules, FATF regulations, and Elliptic typologies into a `SentenceTransformer` local vector store.
- **Ollama Offline Engine**: Passes topological graphs and ensemble metric telemetry securely to a local `llama3.2` model to autonomously draft Suspicious Activity Reports (SAR) without ever exposing data to a public API.

---

## 📊 The Elliptic Dataset Intelligence

The models backing BitcoinGuard AI are trained natively on the renowned **[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)**. The underlying mechanics of the dashboard natively understand the dataset's constraints:

- **Graph Structure**: 203,000 distinct network nodes mapped chronologically over 49 (2-week) time-step boundaries.
- **166 Feature Variables**: Separated cleanly into **94 local variables** (transaction-specific factors) and **71 aggregated variables** (1-hop neighborhood statistical proxies).
- **Imbalance Engineering**: Handles the extreme illicit/licit rarity (1:10) environments utilizing **SMOTE** (Synthetic Minority Over-sampling Technique). This creates a balanced synthetic landscape optimized forcefully for **AUC-PR** (Precision-Recall) thresholds instead of standard ROC curves to brutally minimize False Positives.

---

## 🛠️ Local Installation & Development

To run the complete platform natively (including the AI Investigator LLM Pipeline), follow these steps locally:

### 1. Clone the Repository
```bash
git clone https://github.com/karthikllos/Bitco_Guard.git
cd Bitco_Guard
```

### 2. Configure the Python Environment
It is highly recommended to use Conda. The environment strictly requires `torchvision`, `chromadb`, and `streamlit>=1.36.0`.
```bash
conda create -n bitco python=3.10
conda activate bitco
pip install -r requirements.txt
```

### 3. Initialize the AI Investigator Engine (Ollama)
The investigator LLM works strictly offline (running locally on your hardware):
1. Install **[Ollama](https://ollama.com/)** for your OS.
2. Spin up the specific model parameter locally in a separate terminal:
```bash
ollama run llama3.2
```

### 4. Boot the AML Dashboard
Launch the dashboard via Streamlit on port `8501`.
```bash
streamlit run app.py
```
*(Note: Ensure you are running this from the exact project root (`/Bitco_Guard/`) so the `artifacts/` folder correctly maps its local weights into the Streamlit cache).*

---

## 🚀 Repository Structure Overview

- **`app.py`**: The Main Streamlit entry-point defining structural CSS formatting and `st.navigation` rules.
- **`pages_ui/`**: Contains the modular dashboard components:
  - `overview.py`: Front-facing executive telemetry and model KPIs.
  - `score_transaction.py`: Active sandboxed inference mapping.
  - `shap_explainability.py`: Tree-explainer visualizations.
  - `unknown_risk_scores.py`: The cached Global Ledger Search parameter DB.
  - `ai_investigator.py`: The LLM Copilot Chat interface mapping to ChromaDB context.
- **`utils/`**: Core backend logic for caching payloads and embedding text (`rag_engine.py`, `model_loader.py`, `scorer.py`, `data_models.py`).
- **`db/`**: The dynamically un-packed local Vector database (managed inherently at runtime by `rag_engine.py`).
- **`artifacts/`**: Deep storage for `.pkl` meta-learners, `xgboost.json` boundaries, local testing `.npy` clusters, and `.parquet` metric datasets required for dynamic rendering.

---

## 🛡️ Roadmap

- [x] Consolidate Multi-model Machine Learning pipelines.
- [x] Overhaul UI to robust FinTech architecture (`streamlit>=1.36.0`).
- [x] Complete Offline RAG Copilot Engine for Privacy-centric inference.
- [ ] Connect Live Bitcoin Node RPC endpoint for real-time validation tracking limit testing.
- [ ] Expand Graph Neural Network (GAT/GraphSAGE) visualizations directly within the UI.
