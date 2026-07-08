# agents/ingestor_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Ingestor
#
# Responsibility:
#   • Receives a raw transaction dict (already polled from Kafka by the runner)
#   • Fetches the customer profile from the Profile API
#   • Returns an updated state with transaction + profile filled in
# ─────────────────────────────────────────────────────────────────────────────

import requests
from state import FraudState, PROFILE_API


def ingestor_agent(state: FraudState) -> FraudState:

    transaction = state["transaction"]
    customer_id = transaction["customer_id"]

    print(f"\n[Ingestor] Processing transaction for customer {customer_id}")
    print(f"           Amount: {transaction['amount']}  |  "
          f"Country: {transaction['country']}  |  "
          f"Device: {transaction['device']}  |  "
          f"Hour: {transaction['hour']}  |  "
          f"tx_last_hour: {transaction.get('tx_last_hour', 1)}")

    try:
        response = requests.get(
            f"{PROFILE_API}/profile/{customer_id}",
            timeout=5
        )
        response.raise_for_status()
        profile = response.json()

    except requests.RequestException as e:
        print(f"[Ingestor] ERROR: could not fetch profile — {e}")
        return {**state, "error": f"Profile fetch failed: {e}"}

    print(f"[Ingestor] Profile loaded — "
          f"avg_amount: {profile['avg_amount']}  |  "
          f"usual_country: {profile['usual_country']}  |  "
          f"usual_device: {profile['usual_device']}")

    return {
        **state,
        "profile": profile,
        "error":   None,
    }
