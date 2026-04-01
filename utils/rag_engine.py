import os
import json
import requests
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# Setup persistent chroma client
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db")
chroma_client = chromadb.PersistentClient(path=DB_DIR)

@st.cache_resource
def load_embed_model():
    """Loads a fast, efficient model for local embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    collection = chroma_client.get_or_create_collection(name="bitcoinguard_knowledge")
    
    if collection.count() == 0:
        build_knowledge_base(collection)
        
    return collection

def build_knowledge_base(collection):
    """Populates the knowledge base with structural domain documents."""
    docs = [
        "Darknet Transaction Pattern: High degree of hopping and mixing often signifies darknet market deposits. Typically, illicit actors will route UTXOs through at least 3 intermediatory hops before consolidating.",
        "Ransomware Tactics: Immediate cashing out at KYC exchanges is rare. They prefer unregulated OTC brokers or mixing services like Tornado Cash or Samurai Whirlpool to obfuscate trails.",
        "Ponzi Scheme Red Flags: High volume of small inbound transactions from unique addresses followed by batched outbound transfers to a few addresses. Lack of economic sense in transaction loops.",
        "Mixing Services (CoinJoin): Equal output amounts across multiple recipients simultaneously. Used to break deterministic links between sender and receiver.",
        "XGBoost Behavior: XGBoost heavily weights transaction velocity and recent USD equivalent volume. If feat_34 > 500, XGBoost risk dramatically spikes.",
        "Model Disagreement: Graph models emphasize network loops whereas tree models (XGB/LGBM) care about the isolated transaction features.",
        "Elliptic Dataset Origin: The system is trained on the Elliptic Bitcoin dataset which contains 203,000 node traces modeled across 49 consecutive 2-week time step intervals.",
        "Elliptic Feature Groupings: Features feat_1 through feat_94 are absolute local properties (timeline differences, exact fees). Features feat_95 through feat_165 are neighborhood aggregated structural graphs (1-hop max distance).",
        "SMOTE Class Imbalance: Out of the 203,000 nodes, illicit signals are outmatched 10 to 1 by Licit nodes. We utilize Synthetic Minority Over-sampling Technique (SMOTE) strictly on the training bounds to create a balanced synthetic landscape, aiming purely for AUC-PR score.",
        "FATF Regulation Recs: Virtual Asset Service Providers (VASPs) must enforce AML/CFT controls, verifying the identities of the originators.",
        "FinCEN Rules: $10,000 threshold reporting requirement. Structuring below this amount regularly is a direct indicator of AML circumvention.",
        "SAR Rules: A Suspicious Activity Report (SAR) must document Who, What, When, Where, and Why the transaction is deemed suspicious."
    ]
    
    embed_model = load_embed_model()
    embeddings = embed_model.encode(docs).tolist()
    
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )
    
def query_llm(prompt: str):
    """Yields chunks of text from the local Ollama LLM API."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                yield data["response"]
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ **Error**: Failed to connect to Ollama. Ensure it is running (`ollama run llama3.2`)."
    except Exception as e:
        yield f"\n\n⚠️ **GenAI Error**: {str(e)}"

def retrieve_context(query_str: str, top_k: int = 3) -> str:
    try:
        collection = get_collection()
        embed_model = load_embed_model()
        query_emb = embed_model.encode([query_str]).tolist()
        
        results = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )
        
        docs = results.get("documents", [[]])[0]
        return "\n".join([f"- {d}" for d in docs])
    except Exception as e:
        st.error(f"Failed to retrieve context: {e}")
        return ""
