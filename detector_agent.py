# agents/detector_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Detector
#
# Responsibility:
#   • Computes 8 engineered features from the transaction + profile
#   • Maintains a short per-customer transaction history (in-memory)
#     to compute the 3 new card-testing features
#   • Runs the Isolation Forest model → risk score 0-100
#   • Computes SHAP values to explain which features drove the score
#
# Feature set (8) — must match generate_data.py and ML_model.py:
#   Original : amount_ratio, country_changed, device_changed,
#              outside_hours, tx_last_hour
#   New      : low_amount_probe, amount_escalating, small_tx_count
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import shap
import pandas as pd
import numpy as np
from collections import defaultdict

from state import FraudState
from ML_model import FEATURES, get_risk_score

# ── Load model + SHAP explainer once at import time ──────────────────────────

_model  = joblib.load("isolation_forest.pkl")
_scaler = joblib.load("risk_scaler.pkl")

try:
    _explainer = shap.TreeExplainer(_model)
    _shap_available = True
    print("[Detector] Isolation Forest + SHAP explainer loaded.")
except Exception as e:
    _explainer = None
    _shap_available = False
    print(f"[Detector] SHAP unavailable ({e}). Continuing without SHAP.")

# ── In-memory per-customer recent transaction history ─────────────────────────
# Stores the last 3 transaction amounts per customer.
# Used to compute amount_escalating and small_tx_count.
# Note: resets when the process restarts — acceptable for a PFA demo.

_customer_history: dict[int, list[float]] = defaultdict(list)


# ── Feature engineering ───────────────────────────────────────────────────────

def _compute_features(transaction: dict, profile: dict) -> dict:

    amount    = transaction["amount"]
    avg       = max(profile["avg_amount"], 1)
    cid       = transaction["customer_id"]

    # ── Original 5 ───────────────────────────────────────────────────────────

    amount_ratio = round(amount / avg, 4)

    country_changed = int(
        transaction["country"] != profile["usual_country"]
    )
    device_changed = int(
        transaction["device"] != profile["usual_device"]
    )
    outside_hours = int(
        not (profile["active_start"]
             <= transaction["hour"]
             <= profile["active_end"])
    )
    tx_last_hour = transaction.get("tx_last_hour", 1)

    # ── New feature 1: low_amount_probe ──────────────────────────────────────
    # Signals a possible card-testing probe:
    #   → Amount is abnormally LOW (< 40% of customer average)
    #   → AND there has been recent burst activity (tx_last_hour >= 2)
    # Fraudsters send small amounts first to verify the card is active
    # before escalating to large withdrawals.

    low_amount_probe = int(amount_ratio < 0.4 and tx_last_hour >= 2)

    # ── New feature 2: amount_escalating ─────────────────────────────────────
    # Signals a card-testing escalation pattern:
    #   → Last 3 transactions for this customer are strictly increasing
    # Classic attack: 50 → 200 → 500 → (large theft)

    history = _customer_history[cid]

    if len(history) >= 2:
        last_three = history[-2:] + [amount]
        amount_escalating = int(
            all(last_three[i] < last_three[i + 1]
                for i in range(len(last_three) - 1))
        )
    else:
        amount_escalating = 0

    # ── New feature 3: small_tx_count ────────────────────────────────────────
    # Number of recent transactions (last 3) that are micro-transactions:
    # below 15% of the customer's average amount.
    # Multiple micro-transactions = repeated probing.

    micro_threshold = avg * 0.15
    small_tx_count  = sum(1 for a in history[-3:] if a < micro_threshold)

    # ── Update history ────────────────────────────────────────────────────────

    history.append(amount)
    if len(history) > 3:
        history.pop(0)

    return {
        "amount_ratio":      amount_ratio,
        "country_changed":   country_changed,
        "device_changed":    device_changed,
        "outside_hours":     outside_hours,
        "tx_last_hour":      tx_last_hour,
        "low_amount_probe":  low_amount_probe,
        "amount_escalating": amount_escalating,
        "small_tx_count":    small_tx_count,
    }


# ── SHAP explanation ──────────────────────────────────────────────────────────

def _get_shap_values(features: dict) -> dict:
    """
    Returns per-feature SHAP contributions (positive = more anomalous).
    Falls back to empty dict if SHAP is unavailable (Windows DLL issue).
    """
    if not _shap_available:
        return {}

    try:
        X           = pd.DataFrame([{f: features[f] for f in FEATURES}])
        shap_matrix = _explainer.shap_values(X)
        raw_shap    = shap_matrix[0]

        contributions = {
            feat: round(float(-val), 4)
            for feat, val in zip(FEATURES, raw_shap)
        }
        # Sort descending — biggest driver first
        return dict(sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        ))

    except Exception as e:
        print(f"[Detector] SHAP computation failed: {e}")
        return {}


# ── Agent node ────────────────────────────────────────────────────────────────

def detector_agent(state: FraudState) -> FraudState:

    transaction = state["transaction"]
    profile     = state["profile"]

    print(f"\n[Detector] Computing features...")

    features  = _compute_features(transaction, profile)
    ml_score  = get_risk_score(features)
    shap_vals = _get_shap_values(features)

    # ── Flag card-testing signals in output ───────────────────────────────────

    card_testing_signals = []
    if features["low_amount_probe"]:
        card_testing_signals.append("low_amount_probe")
    if features["amount_escalating"]:
        card_testing_signals.append("amount_escalating")
    if features["small_tx_count"] >= 2:
        card_testing_signals.append(
            f"small_tx_count={features['small_tx_count']}"
        )

    print(f"[Detector] Features    : {features}")
    if card_testing_signals:
        print(f"[Detector] ⚠️  Card-testing signals: {card_testing_signals}")
    print(f"[Detector] Risk Score  : {ml_score}/100")
    if shap_vals:
        print(f"[Detector] SHAP        : {shap_vals}")

    return {
        **state,
        "features":    features,
        "ml_score":    ml_score,
        "shap_values": shap_vals,
    }