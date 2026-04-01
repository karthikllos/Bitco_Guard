import numpy as np
from typing import Dict
from utils.data_models import ALL_FEATURES
from utils.model_loader import load_models, load_transformers

def ensemble_score(features: Dict[str, float]) -> dict:
    """
    Scores the given transaction features using the XGB, LGBM, and CB models.
    `features` should be a dictionary with keys matching ALL_FEATURES.
    """
    models, metrics, status = load_models()
    scaler, _ = load_transformers()
    
    if not status.get("xgb") or not status.get("lgbm") or not status.get("cb") or scaler is None:
        raise ValueError("Cannot score transaction: One or more models/scaler failed to load.")
        
    xgb_model = models["xgb"]
    lgbm_model = models["lgbm"]
    cb_model = models["cb"]
    
    # Prepare feature array
    feature_arr = [features.get(f, 0.0) for f in ALL_FEATURES]
    scaled = scaler.transform([feature_arr])
    
    # XGBoost
    try:
        xgb_score = xgb_model.predict_proba(scaled)[0][1]
    except AttributeError:
        # Fallback if xgboost acts like a Booster
        xgb_score = xgb_model.predict(scaled)[0]
        if isinstance(xgb_score, (list, np.ndarray)) and len(xgb_score) > 1:
            xgb_score = xgb_score[1]

    # LightGBM (Booster .txt uses .predict())
    lgbm_pred = lgbm_model.predict(scaled)
    lgbm_score = lgbm_pred[0] if len(lgbm_pred.shape) == 1 else lgbm_pred[0][1]
    
    # CatBoost
    cb_score = cb_model.predict_proba(scaled)[0][1]
    
    # Ensemble Weights
    xgb_w = metrics["xgb"].get("auc_pr", 0.33)
    lgbm_w = metrics["lgbm"].get("auc_pr", 0.33)
    cb_w = metrics["cb"].get("auc_pr", 0.33)
    
    total = xgb_w + lgbm_w + cb_w
    ensemble = (xgb_w * xgb_score + lgbm_w * lgbm_score + cb_w * cb_score) / total
    
    return {
        "xgb": float(xgb_score),
        "lgbm": float(lgbm_score),
        "cb": float(cb_score),
        "ensemble": float(ensemble)
    }
