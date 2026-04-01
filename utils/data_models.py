import os

ALL_FEATURES = [f"feat_{i}" for i in range(1, 166)]

RISK_TIERS = {
    "CRITICAL": {"threshold": 0.9, "color": "#c0392b"},
    "HIGH": {"threshold": 0.75, "color": "#e74c3c"},
    "MEDIUM": {"threshold": 0.5, "color": "#e67e22"},
    "LOW": {"threshold": 0.25, "color": "#f1c40f"},
    "MINIMAL": {"threshold": 0.0, "color": "#95a5a6"}
}

def get_risk_tier(score: float) -> str:
    """Returns the risk tier given an ensemble score."""
    if score >= RISK_TIERS["CRITICAL"]["threshold"]:
        return "CRITICAL"
    elif score >= RISK_TIERS["HIGH"]["threshold"]:
        return "HIGH"
    elif score >= RISK_TIERS["MEDIUM"]["threshold"]:
        return "MEDIUM"
    elif score >= RISK_TIERS["LOW"]["threshold"]:
        return "LOW"
    else:
        return "MINIMAL"

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
