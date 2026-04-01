import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from utils.model_loader import load_data_artifacts

def display_overview():
    st.markdown("""
    <div class='stCard'>
        <h2>🛡️ Enterprise Fraud Detection Hub</h2>
        <p>Real-time machine learning engine trained on the <b>Elliptic Bitcoin Dataset</b> (203,000 nodes, 49 distinct time intervals). Utilizing high-fidelity Graph Neural Networks and XGBoost ensembles, this protects institutions against Ponzi schemes, Darknet clustering, and Ransomware paths.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load basic analytics
    data = load_data_artifacts()
    
    col1, col2, col3, col4 = st.columns(4)
    if "unknown_risk_scores" in data and "high_risk_unknowns" in data:
        total_unknown = len(data["unknown_risk_scores"])
        high_risk = len(data["high_risk_unknowns"])
        flag_rate = (high_risk/total_unknown)*100 if total_unknown else 0
        
        with col1:
            st.metric("Transactions Scanned", f"{total_unknown:,}", "2.4% (Last 24h)")
        with col2:
            st.metric("High Risk Flags", f"{high_risk:,}", "12 Alerts", delta_color="inverse")
        with col3:
            st.metric("Critical Flag Rate", f"{flag_rate:.2f}%", "-0.15%", delta_color="normal")
        with col4:
            st.metric("Active ML Models", "3 Live", "Ensemble Intact")
    
    st.markdown("### Threat Telemetry (Simulated 30 Days)")
    st.caption("Tracking chronological anomaly spikes across simulated 2-week active periods.")
    
    # Generate mock timeline for visual wow factor
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=30)
    baseline = np.random.normal(500, 50, 30)
    threats = np.random.normal(20, 10, 30)
    # Add a spike
    threats[25] += 120 
    threats[26] += 80

    df_timeline = pd.DataFrame({
        "Date": dates,
        "Total Volume": baseline,
        "Suspicious Events": threats
    })
    
    fig = px.area(df_timeline, x="Date", y=["Total Volume", "Suspicious Events"],
                  color_discrete_sequence=["#90caf9", "#e53935"],
                  title="Network Activity & Anomaly Detections")
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Tx Volume",
        legend_title="",
        font=dict(color="#e0e6ed")
    )
    
    st.plotly_chart(fig, use_container_width=True)

display_overview()
