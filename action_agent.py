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
    """
    Append one line to audit_log.jsonl.
    Each line is a self-contained JSON record — easy to tail, grep, or import.
    """
    record = {
        "audit_id":       audit_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "customer_id":    state["transaction"]["customer_id"],
        "transaction":    state["transaction"],
        "profile":        state["profile"],
        "features":       state["features"],
        "ml_score":       state["ml_score"],
        "shap_values":    state["shap_values"],
        "llm_reasoning":  state["llm_reasoning"],
        "top_signals":    state["top_signals"],
        "final_decision": state["final_decision"],
        "action_taken":   state.get("action_taken", ""),
    }

    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
