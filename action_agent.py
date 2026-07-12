# agents/action_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Action Agent
#
# Responsibility:
#   • Executes the final decision (BLOCK / REVIEW / APPROVE)
#   • Logs every processed transaction to the audit trail (JSON file)
#   • In a production system this would also:
#       - Call a card-blocking API for BLOCK decisions
#       - Push to an analyst queue for REVIEW decisions
#       - Send a notification to the customer
#
# The audit trail is written to audit_log.jsonl (one JSON object per line).
# This format is easy to query, stream, and later load into PostgreSQL.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
from datetime import datetime, timezone

from state import FraudState
from database import get_connection

AUDIT_FILE = "audit_log.jsonl"


# ── Action executors ──────────────────────────────────────────────────────────

def _action_block(transaction: dict, reasoning: str) -> str:
    customer_id = transaction["customer_id"]
    amount      = transaction["amount"]
    print(f"[Action] 🔴 BLOCK  — customer {customer_id} | amount {amount}")
    print(f"[Action]    Reason: {reasoning}")
    # TODO: call card-blocking API
    return f"Transaction BLOCKED for customer {customer_id} (amount {amount})"


def _action_review(transaction: dict, reasoning: str) -> str:
    customer_id = transaction["customer_id"]
    amount      = transaction["amount"]
    print(f"[Action] 🟠 REVIEW — customer {customer_id} | amount {amount}")
    print(f"[Action]    Reason: {reasoning}")
    # TODO: push to analyst review queue
    return f"Transaction queued for REVIEW — customer {customer_id} (amount {amount})"


def _action_approve(transaction: dict) -> str:
    customer_id = transaction["customer_id"]
    amount      = transaction["amount"]
    print(f"[Action] 🟢 APPROVE — customer {customer_id} | amount {amount}")
    return f"Transaction APPROVED for customer {customer_id} (amount {amount})"


# ── Audit logger ──────────────────────────────────────────────────────────────

def _write_audit(state: FraudState, audit_id: str) -> None:
    import json
    tx   = state["transaction"]
    feat = state["features"]

    sql = """
        INSERT INTO fraud_decisions (
            audit_id, customer_id,
            amount, country, device, hour, tx_last_hour,
            amount_ratio, country_changed, device_changed, outside_hours,
            ml_score, shap_values,
            llm_reasoning, top_signals, final_decision, action_taken
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
    """
    values = (
        audit_id,
        tx["customer_id"],
        tx["amount"], tx["country"], tx["device"],
        tx["hour"], tx.get("tx_last_hour", 1),
        feat["amount_ratio"], bool(feat["country_changed"]),
        bool(feat["device_changed"]), bool(feat["outside_hours"]),
        state["ml_score"],
        json.dumps(state["shap_values"]),
        state["llm_reasoning"],
        json.dumps(state["top_signals"]),
        state["final_decision"],
        state.get("action_taken", ""),
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, values)
        conn.commit()

# ── Agent node ────────────────────────────────────────────────────────────────

def action_agent(state: FraudState) -> FraudState:

    decision    = state["final_decision"]
    transaction = state["transaction"]
    reasoning   = state.get("llm_reasoning", "")

    print(f"\n[Action] Executing decision: {decision}")

    # Execute the right action
    if decision == "BLOCK":
        action_taken = _action_block(transaction, reasoning)

    elif decision == "REVIEW":
        action_taken = _action_review(transaction, reasoning)

    else:
        action_taken = _action_approve(transaction)

    # Write audit record
    audit_id = str(uuid.uuid4())[:8]
    _write_audit({**state, "action_taken": action_taken}, audit_id)

    print(f"[Action] Audit record written — ID: {audit_id}")

    return {
        **state,
        "action_taken": action_taken,
        "audit_id":     audit_id,
    }
