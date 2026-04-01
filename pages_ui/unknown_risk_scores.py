import streamlit as st
import pandas as pd
from utils.model_loader import load_data_artifacts

def display_risk_db():
    st.markdown("## 📋 Intelligence Database")
    st.caption("Investigate isolated high-risk topologies and cross-reference unknown transactions.")
    
    data = load_data_artifacts()
    df_unknown = data.get("unknown_risk_scores")
    df_high = data.get("high_risk_unknowns")
    
    if df_unknown is None and df_high is None:
        st.warning("Risk score datasets missing. Cannot display interactive tables.")
        return
        
    t1, t2 = st.tabs(["🔥 Critical Isolations", "🌐 Global Ledger Search"])
    
    with t1:
        st.markdown("""
        <div class="stCard" style="background-color: rgba(231, 76, 60, 0.05); border: 1px solid rgba(231, 76, 60, 0.2);">
            <h4 style="color:#e0e6ed;">Accelerated Threat Profiles (Elliptic Dataset)</h4>
            <p style="color:#a0b2c6;">Profiles that exceeded safety thresholds in recent offline scans across the 203,000 node graph structure.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if df_high is not None:
            # Map a gradient if probability columns exist
            score_cols = [c for c in df_high.columns if any(k in c.lower() for k in ['score', 'proba', 'pred'])]
            if score_cols:
                st.dataframe(
                    df_high.style.background_gradient(cmap='Reds', subset=score_cols),
                    width="stretch",
                    height=500
                )
            else:
                st.dataframe(df_high, width="stretch", height=500)
        else:
            st.info("high_risk_unknowns missing.")
            
    with t2:
        st.markdown("### Scalpel Edge Scan")
        if df_unknown is not None:
            search = st.text_input("🔍 Node or Path Identifier (Supports Regex):", "")
            if search:
                try:
                    mask = pd.Series(False, index=df_unknown.index)
                    for col in df_unknown.select_dtypes(include=['object', 'string', 'category']).columns:
                        mask |= df_unknown[col].astype(str).str.contains(search, case=False, na=False)
                    
                    filtered = df_unknown[mask]
                    st.success(f"Discovered {len(filtered)} matching vectors.")
                    st.dataframe(filtered, width="stretch", height=500)
                except Exception as e:
                    st.error(f"Search format error: {str(e)}")
            else:
                st.dataframe(df_unknown.head(500), width="stretch", height=500)
        else:
            st.info("unknown_risk_scores missing.")

display_risk_db()
