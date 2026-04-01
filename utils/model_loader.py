import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from utils.data_models import ARTIFACT_DIR

@st.cache_resource
def load_models():
    """Loads XGBoost, LightGBM, CatBoost and their metrics from artifacts."""
    models_status = {"xgb": False, "lgbm": False, "cb": False}
    models = {}
    metrics = {}

    # XGBoost
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(ARTIFACT_DIR, "xgb_bitcoinguard_tuned.json"))
        models['xgb'] = xgb_model
        
        with open(os.path.join(ARTIFACT_DIR, "xgb_metrics.json"), 'r') as f:
            metrics['xgb'] = json.load(f)
            
        models_status["xgb"] = True
    except Exception as e:
        st.error(f"Failed to load XGBoost model or metrics: {e}")

    # LightGBM
    try:
        lgbm_model = lgb.Booster(model_file=os.path.join(ARTIFACT_DIR, "lgb_bitcoinguard.txt"))
        models['lgbm'] = lgbm_model
        
        with open(os.path.join(ARTIFACT_DIR, "lgb_metrics.json"), 'r') as f:
            metrics['lgbm'] = json.load(f)
            
        models_status["lgbm"] = True
    except Exception as e:
        st.error(f"Failed to load LightGBM model or metrics: {e}")

    # CatBoost
    try:
        cb_model = CatBoostClassifier()
        cb_model.load_model(os.path.join(ARTIFACT_DIR, "catboost_bitcoinguard.cbm"))
        models['cb'] = cb_model
        
        with open(os.path.join(ARTIFACT_DIR, "cat_metrics.json"), 'r') as f:
            metrics['cb'] = json.load(f)
            
        models_status["cb"] = True
    except Exception as e:
        st.error(f"Failed to load CatBoost model or metrics: {e}")

    return models, metrics, models_status

@st.cache_resource
def load_transformers():
    """Loads the scaler and meta learner."""
    scaler, meta_learner = None, None
    try:
        with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")

    try:
        with open(os.path.join(ARTIFACT_DIR, "meta_learner.pkl"), "rb") as f:
            meta_learner = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load meta learner: {e}")
        
    return scaler, meta_learner

@st.cache_resource
def load_data_artifacts():
    """Loads pandas and numpy data artifacts for analysis and explainability."""
    data = {}
    try:
        data["unknown_risk_scores"] = pd.read_parquet(os.path.join(ARTIFACT_DIR, "unknown_risk_scores.parquet"))
    except Exception as e:
        st.error(f"Failed to load unknown_risk_scores.parquet: {e}")

    try:
        data["high_risk_unknowns"] = pd.read_csv(os.path.join(ARTIFACT_DIR, "high_risk_unknowns.csv"))
    except Exception as e:
        st.error(f"Failed to load high_risk_unknowns.csv: {e}")

    try:
        data["shap_feature_ranking"] = pd.read_parquet(os.path.join(ARTIFACT_DIR, "shap_feature_ranking.parquet"))
    except Exception as e:
        st.error(f"Failed to load shap_feature_ranking.parquet: {e}")

    try:
        data["ensemble_with_disagreement"] = pd.read_parquet(os.path.join(ARTIFACT_DIR, "ensemble_with_disagreement.parquet"))
    except Exception as e:
        st.error(f"Failed to load ensemble_with_disagreement.parquet: {e}")

    try:
        data["shap_values"] = np.load(os.path.join(ARTIFACT_DIR, "shap_values.npy"))
        data["shap_sample_X"] = np.load(os.path.join(ARTIFACT_DIR, "shap_sample_X.npy"))
        data["shap_sample_y"] = np.load(os.path.join(ARTIFACT_DIR, "shap_sample_y.npy"))
    except Exception as e:
        st.error(f"Failed to load SHAP numpy arrays: {e}")

    try:
        with open(os.path.join(ARTIFACT_DIR, "bitcoinguard_final_metrics.json"), 'r') as f:
            data["final_metrics"] = json.load(f)
    except Exception as e:
        st.error(f"Failed to load bitcoinguard_final_metrics.json: {e}")
        
    return data
