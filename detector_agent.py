# agents/detector_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Detector
#
# Responsibility:
#   • Computes the 5 engineered features from the transaction + profile
#   • Runs the Isolation Forest model to get a risk score 0-100
#   • Computes SHAP values to explain which features drove the score
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import shap
import pandas as pd
import numpy as np

from state import FraudState, FEATURES

# ── Load model once at import time ───────────────────────────────────────────

_model  = joblib.load("isolation_forest.pkl")
_scaler = joblib.load("risk_scaler.pkl")

# SHAP TreeExplainer for Isolation Forest
_explainer = shap.Explainer(_model)

print("[Detector] Isolation Forest + SHAP explainer loaded.")


# ── Feature engineering ──────────────────────────────────────────────────────

def _compute_features(transaction: dict, profile: dict) -> dict:

    amount_ratio = transaction["amount"] / max(profile["avg_amount"], 1)

    country_changed = int(
        transaction["country"] != profile["usual_country"]
    )

    device_changed = int(
        transaction["device"] != profile["usual_device"]
    )

    outside_hours = int(
        not (
            profile["active_start"]
            <= transaction["hour"]
            <= profile["active_end"]
        )
    )

    tx_last_hour = transaction.get("tx_last_hour", 1)

    return {
        "amount_ratio":    round(amount_ratio, 4),
        "country_changed": country_changed,
        "device_changed":  device_changed,
        "outside_hours":   outside_hours,
        "tx_last_hour":    tx_last_hour,
    }


# ── Risk scoring ─────────────────────────────────────────────────────────────

def _get_risk_score(features: dict) -> float:

    X = pd.DataFrame([features])[FEATURES]

    raw_score  = _model.score_samples(X)[0]
    risk_score = _scaler.transform([[-raw_score]])[0][0]

    return round(float(np.clip(risk_score, 0, 100)), 2)


# ── SHAP explanation ─────────────────────────────────────────────────────────

def _get_shap_values(features: dict) -> dict:
    """
    Returns a dict mapping each feature name to its SHAP contribution.
    Positive = pushed the score UP (more anomalous).
    """
    X = pd.DataFrame([features])[FEATURES]

    shap_matrix = _explainer.shap_values(X)   # shape (1, n_features)
    raw_shap    = shap_matrix[0]               # array of length n_features

    # Negate: isolation forest shap_values use the raw anomaly score convention
    # (more negative = more anomalous), so we flip for readability.
    contributions = {
        feat: round(float(-val), 4)
        for feat, val in zip(FEATURES, raw_shap)
    }

    # Sort descending so the biggest driver comes first
    contributions = dict(
        sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    )

    return contributions


# ── Agent node ───────────────────────────────────────────────────────────────

def detector_agent(state: FraudState) -> FraudState:

    transaction = state["transaction"]
    profile     = state["profile"]

    print(f"\n[Detector] Computing features...")

    features   = _compute_features(transaction, profile)
    ml_score   = _get_risk_score(features)
    shap_vals  = _get_shap_values(features)

    print(f"[Detector] Features    : {features}")
    print(f"[Detector] Risk Score  : {ml_score}/100")
    print(f"[Detector] SHAP        : {shap_vals}")

    return {
        **state,
        "features":   features,
        "ml_score":   ml_score,
        "shap_values": shap_vals,
    }
 