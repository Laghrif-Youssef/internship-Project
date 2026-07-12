# agents/llm_reasoner_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — LLM Reasoner
#
# Changes vs previous version:
#   • Prompt now includes card-testing fraud pattern context
#   • Explicitly tells the LLM that LOW amounts can be MORE suspicious
#   • New features (low_amount_probe, amount_escalating, small_tx_count)
#     are included in the prompt with human-readable explanations
#   • Override guard: LLM cannot escalate a score < 25 without at least
#     2 strong signals — reduces false positives on clean transactions
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import ollama

from state import (
    FraudState,
    OLLAMA_MODEL,
    BLOCK_THRESHOLD,
    REVIEW_THRESHOLD,
)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(state: FraudState) -> str:

    tx      = state["transaction"]
    profile = state["profile"]
    feat    = state["features"]
    score   = state["ml_score"]
    shap    = state["shap_values"]

    # ML suggestion from thresholds
    ml_suggestion = (
        "BLOCK"   if score >= BLOCK_THRESHOLD  else
        "REVIEW"  if score >= REVIEW_THRESHOLD else
        "APPROVE"
    )

    # SHAP top 3 drivers (if available)
    if shap:
        top3 = list(shap.items())[:3]
        shap_section = "TOP SHAP DRIVERS (positive = pushes score up):\n" + "\n".join(
            f"  - {name}: {val:+.4f}" for name, val in top3
        )
    else:
        shap_section = "SHAP: not available for this transaction."

    # Card testing signals summary
    card_signals = []
    if feat.get("low_amount_probe"):
        card_signals.append(
            f"low_amount_probe=1  "
            f"(amount is {feat['amount_ratio']:.2f}x avg = "
            f"ABNORMALLY LOW, AND {feat['tx_last_hour']} recent transactions "
            f"→ possible card-testing probe)"
        )
    if feat.get("amount_escalating"):
        card_signals.append(
            "amount_escalating=1  "
            "(last 3 transaction amounts are strictly increasing "
            "→ classic escalation pattern before a large fraud)"
        )
    if feat.get("small_tx_count", 0) >= 2:
        card_signals.append(
            f"small_tx_count={feat['small_tx_count']}  "
            f"({feat['small_tx_count']} recent micro-transactions below 15% of avg "
            f"→ repeated probing behavior)"
        )

    card_section = (
        "CARD-TESTING SIGNALS DETECTED:\n" +
        "\n".join(f"  ⚠️  {s}" for s in card_signals)
    ) if card_signals else "No card-testing signals detected."

    return f"""You are an expert fraud analyst at a bank.
A transaction has been scored by an Isolation Forest ML model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL FRAUD KNOWLEDGE — READ BEFORE ANALYSING:

1. LOW AMOUNTS CAN BE MORE SUSPICIOUS THAN HIGH AMOUNTS.
   Card testing / probing attacks: fraudsters send SMALL transactions
   first to verify the card is active, then escalate to large thefts.
   A transaction that is 30% of the customer's average is NOT safe —
   it may be a deliberate probe. Treat low_amount_probe=1 seriously.

2. ESCALATING AMOUNTS = HIGH RISK.
   A pattern of 3 recent transactions with strictly increasing amounts
   (amount_escalating=1) is a textbook card-testing sequence.
   Even if the current amount looks normal, the pattern is suspicious.

3. MULTIPLE MICRO-TRANSACTIONS = HIGH RISK.
   Several recent transactions below 15% of the customer's average
   (small_tx_count >= 2) indicate repeated probing behavior.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRANSACTION:
  Amount        : {tx['amount']} ({feat['amount_ratio']:.2f}x customer average of {profile['avg_amount']})
  Country       : {tx['country']} (usual: {profile['usual_country']}, changed: {'YES' if feat['country_changed'] else 'NO'})
  Device        : {tx['device']} (usual: {profile['usual_device']}, changed: {'YES' if feat['device_changed'] else 'NO'})
  Hour          : {tx['hour']}h (active window: {profile['active_start']}h-{profile['active_end']}h, outside: {'YES' if feat['outside_hours'] else 'NO'})
  tx_last_hour  : {feat['tx_last_hour']}

{card_section}

{shap_section}

ML SCORE  : {score}/100
ML SUGGEST: {ml_suggestion}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERRIDE RULES:
  - You MAY escalate APPROVE → REVIEW if you see at least 2 genuine risk signals.
  - You MAY escalate REVIEW → BLOCK if the card-testing pattern is clear.
  - You MAY downgrade REVIEW → APPROVE ONLY if ML score < 25 AND all 8
    features look genuinely normal AND no card-testing signals are present.
  - NEVER downgrade a score >= 80.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write a 2-3 sentence explanation of why this transaction is or is not
suspicious. Mention card-testing explicitly if relevant.

Return ONLY valid JSON:
{{
  "reasoning": "...",
  "top_signals": ["signal 1", "signal 2", "signal 3"],
  "decision": "BLOCK" | "REVIEW" | "APPROVE"
}}"""


# ── JSON parser with fallback ──────────────────────────────────────────────────

def _parse_llm_response(raw: str, fallback_score: float) -> tuple[str, list, str]:

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Strip Qwen3 <think>...</think> block if present
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()

    try:
        data     = json.loads(clean)
        reasoning = data.get("reasoning", "No explanation provided.")
        signals   = data.get("top_signals", [])
        decision  = data.get("decision", "").upper().strip()

        if decision not in ("BLOCK", "REVIEW", "APPROVE"):
            raise ValueError(f"Invalid decision value: '{decision}'")

        return reasoning, signals, decision

    except Exception as e:
        print(f"[LLM Reasoner] JSON parse failed ({e}), using fallback.")
        decision = (
            "BLOCK"   if fallback_score >= BLOCK_THRESHOLD  else
            "REVIEW"  if fallback_score >= REVIEW_THRESHOLD else
            "APPROVE"
        )
        return (
            f"LLM response could not be parsed. ML score: {fallback_score}/100.",
            [],
            decision,
        )


# ── Agent node ────────────────────────────────────────────────────────────────

def llm_reasoner_agent(state: FraudState) -> FraudState:

    print(f"\n[LLM Reasoner] Calling Ollama ({OLLAMA_MODEL})...")

    prompt = _build_prompt(state)

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "temperature": 0,
                "num_ctx":     4096,
            },
        )
        raw = response["message"]["content"]

    except Exception as e:
        print(f"[LLM Reasoner] Ollama call failed: {e}")
        fallback = (
            "BLOCK"   if state["ml_score"] >= BLOCK_THRESHOLD  else
            "REVIEW"  if state["ml_score"] >= REVIEW_THRESHOLD else
            "APPROVE"
        )
        return {
            **state,
            "llm_reasoning":  f"LLM unavailable: {e}",
            "top_signals":    [],
            "final_decision": fallback,
        }

    reasoning, signals, decision = _parse_llm_response(raw, state["ml_score"])

    print(f"[LLM Reasoner] Decision  : {decision}")
    print(f"[LLM Reasoner] Signals   : {signals}")
    print(f"[LLM Reasoner] Reasoning : {reasoning}")

    return {
        **state,
        "llm_reasoning":  reasoning,
        "top_signals":    signals,
        "final_decision": decision,
    }