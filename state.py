# state.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared LangGraph state.  Every agent reads from and writes to this dict.
# ─────────────────────────────────────────────────────────────────────────────

from typing import TypedDict, Optional


class FraudState(TypedDict):

    # ── Ingestor Agent ────────────────────────────────────────────────────────
    transaction:     dict               # raw message from Kafka
    profile:         dict               # customer profile from Profile API

    # ── Detector Agent ───────────────────────────────────────────────────────
    features:        dict               # engineered feature vector
    ml_score:        float              # Isolation Forest risk score 0-100
    shap_values:     dict               # per-feature SHAP contributions

    # ── LLM Reasoner Agent ───────────────────────────────────────────────────
    llm_reasoning:   str                # LLM explanation paragraph
    top_signals:     list               # ["amount 73x avg", "outside hours", …]
    final_decision:  str                # BLOCK | REVIEW | APPROVE

    # ── Action Agent ─────────────────────────────────────────────────────────
    action_taken:    str                # human-readable confirmation
    audit_id:        Optional[str]      # DB row id if persisted

    # ── Error handling ───────────────────────────────────────────────────────
    error:           Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────────────

KAFKA_BROKER        = "localhost:9092"
TRANSACTION_TOPIC   = "transactions"
PROFILE_API         = "http://localhost:8000"
OLLAMA_MODEL        = "qwen3:8b"          # change to qwen3:8b if available

# Thresholds
BLOCK_THRESHOLD     = 80
REVIEW_THRESHOLD    = 40

# Feature list (must match ML_model.py)
FEATURES = [
    "amount_ratio",
    "country_changed",
    "device_changed",
    "outside_hours",
    "tx_last_hour",
]
