# ML_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains and serves the Isolation Forest anomaly detection model.
#
# Feature set (8) — must match generate_data.py and detector_agent.py exactly:
#   amount_ratio, country_changed, device_changed, outside_hours,
#   tx_last_hour, low_amount_probe, amount_escalating, small_tx_count
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ── Feature list — single source of truth ────────────────────────────────────
# Import this list in detector_agent.py so both always stay in sync.

FEATURES = [
    # Original 5
    "amount_ratio",
    "country_changed",
    "device_changed",
    "outside_hours",
    "tx_last_hour",
    # New 3 — card testing / probing patterns
    "low_amount_probe",    # small amount + recent burst = possible probe
    "amount_escalating",   # last 3 amounts strictly increasing
    "small_tx_count",      # number of micro-transactions in recent history
]

MODEL_FILE  = "isolation_forest.pkl"
SCALER_FILE = "risk_scaler.pkl"

# ── Globals (loaded once per process) ────────────────────────────────────────

model  = None
scaler = None

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(csv_file="normal_transactions.csv"):

    print("Loading training dataset...")
    df = pd.read_csv(csv_file)

    # Verify all expected features are present
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing features: {missing}\n"
            "Re-run generate_data.py before training."
        )

    X = df[FEATURES]
    print(f"Training on {len(X)} transactions with {len(FEATURES)} features...")
    print(f"Features: {FEATURES}")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    # contamination=0.01: tells the model to expect ~1% anomalies in training
    # data (which is realistic — our normal data has occasional edge cases).
    # n_estimators=200: more trees = more stable scores.

    _model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,        # use all CPU cores
    )
    _model.fit(X)

    # ── Score normalizer ──────────────────────────────────────────────────────
    # score_samples() returns negative values (more negative = more anomalous).
    # We negate and scale to [0, 100] so 100 = most anomalous.

    raw_scores = _model.score_samples(X)

    _scaler = MinMaxScaler(feature_range=(0, 100))
    _scaler.fit((-raw_scores).reshape(-1, 1))

    # ── Validation: check score distribution ─────────────────────────────────

    normalized = _scaler.transform((-raw_scores).reshape(-1, 1)).flatten()
    print(f"\nScore distribution on training data:")
    print(f"  Mean  : {normalized.mean():.2f}")
    print(f"  Median: {np.median(normalized):.2f}")
    print(f"  P95   : {np.percentile(normalized, 95):.2f}")
    print(f"  P99   : {np.percentile(normalized, 99):.2f}")
    print(f"  Max   : {normalized.max():.2f}")
    print(f"  Scores >= 80 (would BLOCK): {(normalized >= 80).sum()} "
          f"({(normalized >= 80).mean()*100:.2f}%)")
    print(f"  Scores >= 40 (would REVIEW): {(normalized >= 40).sum()} "
          f"({(normalized >= 40).mean()*100:.2f}%)")

    joblib.dump(_model,  MODEL_FILE)
    joblib.dump(_scaler, SCALER_FILE)

    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")

# ── Load ──────────────────────────────────────────────────────────────────────

def load_model():
    global model, scaler
    model  = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print(f"Model loaded — {len(FEATURES)} features: {FEATURES}")

# ── Inference ─────────────────────────────────────────────────────────────────

def get_risk_score(features: dict) -> float:
    """
    Score a single feature dict.
    Returns a float 0-100 (higher = more anomalous = higher fraud risk).
    """
    if model is None or scaler is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Only pass the expected features in the correct order
    X = pd.DataFrame([{f: features[f] for f in FEATURES}])

    raw_score  = model.score_samples(X)[0]
    risk_score = scaler.transform([[-raw_score]])[0][0]

    return round(float(np.clip(risk_score, 0, 100)), 2)


def decision_from_score(score: float) -> str:
    if score >= 80:
        return "BLOCK"
    if score >= 40:
        return "REVIEW"
    return "APPROVE"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()