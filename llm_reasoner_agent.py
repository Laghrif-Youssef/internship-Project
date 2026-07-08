# agents/llm_reasoner_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — LLM Reasoner
#
# Responsibility:
#   • Receives the ML score + SHAP values
#   • Sends a structured prompt to Ollama (qwen2.5:3b or qwen3:8b)
#   • Gets back: explanation paragraph, top risk signals, final decision
#   • The LLM can OVERRIDE the ML decision when context justifies it
#     (e.g. amount ratio is high but all other signals are clean → REVIEW not BLOCK)
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

    # Identify the top 3 SHAP drivers
    top3 = list(shap.items())[:3]
    shap_lines = "\n".join(
        f"  - {feat_name}: contribution = {val:+.4f}"
        for feat_name, val in top3
    )

    # Human-readable feature summary
    feat_summary = (
        f"  - Amount: {tx['amount']} "
        f"({feat['amount_ratio']:.1f}x the customer's average of {profile['avg_amount']})\n"
        f"  - Country: {tx['country']} "
        f"(usual: {profile['usual_country']}, changed: {'YES' if feat['country_changed'] else 'NO'})\n"
        f"  - Device: {tx['device']} "
        f"(usual: {profile['usual_device']}, changed: {'YES' if feat['device_changed'] else 'NO'})\n"
        f"  - Hour: {tx['hour']}h "
        f"(active window: {profile['active_start']}h–{profile['active_end']}h, "
        f"outside: {'YES' if feat['outside_hours'] else 'NO'})\n"
        f"  - Transactions in last hour: {feat['tx_last_hour']}"
    )

    ml_suggestion = (
        "BLOCK"   if score >= BLOCK_THRESHOLD  else
        "REVIEW"  if score >= REVIEW_THRESHOLD else
        "APPROVE"
    )

    return f"""You are an expert fraud analyst at a bank.
A transaction has been scored by an Isolation Forest ML model with a risk score of {score}/100.

TRANSACTION DETAILS:
{feat_summary}

TOP RISK SIGNALS (SHAP feature contributions, higher = more anomalous):
{shap_lines}

ML MODEL SUGGESTION: {ml_suggestion} (score {score}/100)

YOUR TASK:
1. Analyse ALL signals holistically.
2. Write a concise explanation (2-3 sentences) of why this transaction is or is not suspicious.
3. Identify the 2-3 most important risk signals as short phrases.
4. Confirm or override the ML decision based on your reasoning.
   - Override is allowed when context clearly justifies it (e.g. single high-amount but no other signals → downgrade to REVIEW).
   - Never upgrade APPROVE to BLOCK without at least 3 strong signals.

Return ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "reasoning": "...",
  "top_signals": ["signal 1", "signal 2", "signal 3"],
  "decision": "BLOCK" | "REVIEW" | "APPROVE"
}}"""


# ── JSON extraction helper ────────────────────────────────────────────────────

def _parse_llm_response(raw: str, fallback_score: float) -> tuple[str, list, str]:
    """
    Parse the LLM JSON response.
    Falls back gracefully if the model returns malformed output.
    """
    # Strip markdown fences if the model added them despite instructions
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data      = json.loads(clean)
        reasoning = data.get("reasoning", "No explanation provided.")
        signals   = data.get("top_signals", [])
        decision  = data.get("decision", "").upper()

        if decision not in ("BLOCK", "REVIEW", "APPROVE"):
            raise ValueError(f"Invalid decision: {decision}")

        return reasoning, signals, decision

    except Exception as e:
        print(f"[LLM Reasoner] JSON parse failed ({e}), using fallback.")
        # Fallback: use ML thresholds
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
        # Fallback to pure ML decision
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
