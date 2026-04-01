import streamlit as st
import numpy as np
import plotly.express as px
from utils.model_loader import load_data_artifacts
from utils.data_models import ALL_FEATURES

def display_shap():
    st.markdown("## 🧠 Cognitive Explainer (SHAP)")
    st.caption("Mathematical breakdown of feature impacts across the transaction network.")
    st.info("💡 **Context Highlight:** Features `feat_1` to `feat_94` are transaction-specific local behaviors (e.g., timings, fees). `feat_95` to `feat_165` are 1-hop aggregated neighborhood features.")
    
    data = load_data_artifacts()
    shap_vals = data.get("shap_values")
    shap_X = data.get("shap_sample_X")
    
    if shap_vals is not None and shap_X is not None:
        col1, col2 = st.columns([1, 1.2])
        
        # Taking mean absolute across all samples
        mean_shap = np.abs(shap_vals).mean(axis=0)
        
        # Safely handle multidimensional shap values
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=1) 
            
        top_indices = np.argsort(mean_shap)[-12:]
        top_features = [ALL_FEATURES[i] if i < len(ALL_FEATURES) else f"feat_{i}" for i in top_indices]
        top_scores = mean_shap[top_indices]
        
        with col1:
            st.markdown("<h4>Global Risk Drivers</h4>", unsafe_allow_html=True)
            
            fig = px.bar(
                x=top_scores, 
                y=top_features, 
                orientation='h',
                color=top_scores,
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="#e0e6ed"),
                xaxis_title="Mean Impact Magnitude",
                yaxis_title="",
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("<h4>Micro Feature Dependence</h4>", unsafe_allow_html=True)
            selected_feat = st.selectbox("Examine specific feature axis:", top_features[::-1], label_visibility="collapsed")
            feat_idx = ALL_FEATURES.index(selected_feat)
            
            if feat_idx < shap_X.shape[1]:
                # Scatter plot of Feature value vs SHAP value
                fig_scatter = px.scatter(
                    x=shap_X[:, feat_idx],
                    y=shap_vals[:, feat_idx] if len(shap_vals.shape)==2 else shap_vals[:, feat_idx].mean(axis=1),
                    labels={'x': f'Value ({selected_feat})', 'y': 'Log Odds Translation (SHAP)'},
                    color=shap_vals[:, feat_idx] if len(shap_vals.shape)==2 else shap_vals[:, feat_idx].mean(axis=1),
                    color_continuous_scale="RdBu_r"
                )
                fig_scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="#e0e6ed"),
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
    else:
        st.warning("SHAP arrays missing. Ensure variables are cached locally.")

display_shap()
