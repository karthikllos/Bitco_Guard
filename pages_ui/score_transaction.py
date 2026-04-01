import streamlit as st
import random
import time
import numpy as np
import plotly.graph_objects as go
from utils.scorer import ensemble_score
from utils.data_models import ALL_FEATURES, get_risk_tier, RISK_TIERS

def build_mock_network_graph(tier, primary_color):
    """Wow factor: Simulates a network transaction graph centered on the score."""
    np.random.seed(int(time.time()) % 100)
    n_nodes = random.randint(12, 25)
    
    node_x = np.random.randn(n_nodes) * 5
    node_y = np.random.randn(n_nodes) * 5
    node_x[0], node_y[0] = 0, 0 # Center target
    
    edge_x, edge_y = [], []
    for i in range(1, n_nodes):
        edge_x.extend([node_x[0], node_x[i], None])
        edge_y.extend([node_y[0], node_y[i], None])
        if random.random() > 0.7 and i < n_nodes - 1:
            edge_x.extend([node_x[i], node_x[i+1], None])
            edge_y.extend([node_y[i], node_y[i+1], None])

    sizes = [12] * n_nodes
    colors = ['#bdc3c7'] * n_nodes
    sizes[0] = 30
    colors[0] = primary_color
    
    if tier in ["CRITICAL", "HIGH"]:
        for j in range(1, random.randint(4, 8)):
            colors[j] = RISK_TIERS["HIGH"]["color"]
            sizes[j] = 18

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150, 160, 180, 0.4)'),
        hoverinfo='none',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=["Evaluated TX"] + [f"Hop {k}" for k in range(1, n_nodes)],
        marker=dict(size=sizes, color=colors, line=dict(width=2, color='white'))
    ))

    fig.update_layout(
        title="Topology Simulation",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def display_score():
    st.markdown("## 🎯 Live Evaluation Matrix")
    st.caption("Inject customized telemetry payloads directly into the Ensemble.")
    
    if "manual_features" not in st.session_state:
        st.session_state["manual_features"] = {f: 0.0 for f in ALL_FEATURES}

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🎲 Simulate Baseline TX"):
            st.session_state["manual_features"] = {f: random.uniform(0.1, 0.5) for f in ALL_FEATURES}
    with colB:
        if st.button("🔥 Simulate Darknet Pattern"):
            st.session_state["manual_features"] = {f: random.uniform(0.5, 3.5) if i%3==0 else 0.1 for i, f in enumerate(ALL_FEATURES)}

    features = st.session_state["manual_features"]
    
    with st.expander("⚙️ Fine-Tune Telemetry Details", expanded=False):
        cols = st.columns(3)
        for i in range(12):
            f = ALL_FEATURES[i]
            features[f] = cols[i%3].number_input(label=f"Param [{f}]", value=float(features[f]), key=f"inp_{f}")

    if st.button("Initialize Deep Scan", type="primary"):
        status = st.status("Analyzing Sub-Graph Topology...", expanded=True)
        time.sleep(0.5)
        status.write("Running Gradient Boosting Trees...")
        time.sleep(0.7)
        status.write("Calculating SHAP Marginal Contributions...")
        time.sleep(0.6)
        
        try:
            results = ensemble_score(features)
            score = results["ensemble"]
            tier = get_risk_tier(score)
            color = RISK_TIERS[tier]["color"]
            status.update(label="Analysis Complete", state="complete", expanded=False)
            
            if tier == "MINIMAL": st.balloons()
                
            st.markdown("---")
            c1, c2 = st.columns([1, 1.2])
            
            with c1:
                st.markdown(f"""
                <div class="stCard" style="border-left: 5px solid {color};">
                    <h3 style="margin-bottom:0;">Ensemble Score</h3>
                    <h1 style="color:{color}; font-size:3rem; margin-top:0;">{score:.4f}</h1>
                    <h4 style="color:#a0b2c6; margin-top:0;">Tier: {tier}</h4>
                    <hr>
                    <p style="margin:0; color:#e0e6ed;"><b>XGBoost:</b> {results['xgb']:.4f}</p>
                    <p style="margin:0; color:#e0e6ed;"><b>LightGBM:</b> {results['lgbm']:.4f}</p>
                    <p style="margin:0; color:#e0e6ed;"><b>CatBoost:</b> {results['cb']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                fig = build_mock_network_graph(tier, color)
                st.plotly_chart(fig, use_container_width=True)

            # Store for investigator
            st.session_state["last_tx_scored"] = {
                "results": results,
                "tier": tier,
                "top_features": {f: features[f] for f in ALL_FEATURES[:6]}
            }
            st.toast("Telemetry passed to AI Analyst ✅")
            
        except Exception as e:
            status.update(label="Process Failed", state="error")
            st.error(f"Computation Error: {str(e)}")

display_score()
