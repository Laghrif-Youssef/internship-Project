# ML_model.py

import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# CONFIG
# =====================================================

FEATURES = [
    "amount_ratio",
    "country_changed",
    "device_changed",
    "outside_hours",
    "tx_last_hour"
]

MODEL_FILE = "isolation_forest.pkl"
SCALER_FILE = "risk_scaler.pkl"

# =====================================================
# GLOBALS
# =====================================================

model = None
scaler = None

# =====================================================
# TRAINING
# =====================================================

def train_model(csv_file="normal_transactions.csv"):

    print("Loading training dataset...")

    df = pd.read_csv(csv_file)

    X = df[FEATURES]

    print(f"Training on {len(X)} transactions...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )

    model.fit(X)

    # Used later to normalize anomaly scores
    scores = model.score_samples(X)

    scaler = MinMaxScaler(
        feature_range=(0, 100)
    )

    scaler.fit(
        (-scores).reshape(-1, 1)
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("Model saved:")
    print(f" - {MODEL_FILE}")
    print(f" - {SCALER_FILE}")

# =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    global model
    global scaler

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    print("Model loaded successfully.")

# =====================================================
# RISK SCORING
# =====================================================

def get_risk_score(features):

    global model
    global scaler

    if model is None or scaler is None:
        raise Exception(
            "Model not loaded. Call load_model() first."
        )

    X = pd.DataFrame([features])

    raw_score = model.score_samples(X)[0]

    risk_score = scaler.transform(
        [[-raw_score]]
    )[0][0]

    risk_score = max(
        0,
        min(100, risk_score)
    )

    return round(risk_score, 2)

# =====================================================
# DECISION ENGINE
# =====================================================

def decision_from_score(score):

    if score >= 80:
        return "BLOCK"

    elif score >= 40:
        return "REVIEW"

    return "APPROVE"

# =====================================================
# TRAIN MODEL
# =====================================================

if __name__ == "__main__":

    train_model()