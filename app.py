import streamlit as st
import requests
from utils.model_loader import load_models

# Must be the first Streamlit command
st.set_page_config(
    page_title="BitcoinGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS for Card structures (Theme colors handled by config.toml)
st.markdown("""
<style>
.stCard {
    background-color: rgba(26, 43, 60, 0.4);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.stCard:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)


def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            if any("llama3.2" in m for m in models):
                return True, "✅ Local LLM ready"
            return False, "⚠️ llama3.2 not found"
        return False, "⚠️ Ollama error"
    except Exception:
        return False, "⚠️ Start Ollama (Port 11434)"

def main():
    with st.sidebar:
        st.markdown("## 🛡️ BitcoinGuard AI")
        st.caption("v2.0 Fintech Enterprise | Fraud Detection")
        st.markdown("---")
        
        st.subheader("System Status")
        # Check Model Status
        _, _, status = load_models()
        if all(status.values()):
            st.success("✅ Models Loaded")
        else:
            st.error("❌ Models Missing")
            
        # Check AI Status
        ai_status, ai_msg = check_ollama()
        if ai_status:
            st.success(ai_msg)
        else:
            st.warning(ai_msg)

    # Define Navigation Structure using native st.navigation
    pg = st.navigation({
        "Overview": [
            st.Page("pages_ui/overview.py", title="Dashboard Overview", icon="📊"),
            st.Page("pages_ui/model_performance.py", title="Model Performance", icon="📈")
        ],
        "Analysis Tools": [
            st.Page("pages_ui/score_transaction.py", title="Score Transaction", icon="🎯"),
            st.Page("pages_ui/shap_explainability.py", title="Feature Explainability", icon="🧠"),
            st.Page("pages_ui/unknown_risk_scores.py", title="Risk Database", icon="📋")
        ],
        "AI Agent": [
            st.Page("pages_ui/ai_investigator.py", title="Investigator LLM", icon="🤖")
        ]
    })
    
    pg.run()

if __name__ == "__main__":
    main()
