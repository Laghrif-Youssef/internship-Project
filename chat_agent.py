# agents/chat_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — Chat Agent
#
# Responsibility:
#   • Standalone agent — NOT part of the fraud pipeline graph
#   • Lets a human analyst ask questions in natural language about:
#       - Past decisions ("why was customer 110 blocked?")
#       - Recent BLOCK/REVIEW patterns ("how many transactions blocked today?")
#       - A specific transaction ("explain audit_id abc123")
#   • Reads from audit_log.jsonl and answers via Ollama
#
# Usage (interactive):
#   python agents/chat_agent.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import ollama
from datetime import datetime, timezone

#from state import OLLAMA_MODEL

AUDIT_FILE = "audit_log.jsonl"


# ── Audit loader ──────────────────────────────────────────────────────────────

def _load_recent_audit(n: int = 30) -> list[dict]:
    """Load the last N records from the audit log."""
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = [json.loads(l) for l in lines if l.strip()]
        return records[-n:]
    except FileNotFoundError:
        return []


def _summarize_audit(records: list[dict]) -> str:
    """
    Produce a compact text summary of recent audit records
    to fit inside the LLM context window.
    """
    if not records:
        return "No audit records available yet."

    lines = []
    for r in records:
        lines.append(
            f"[{r['audit_id']}] "
            f"customer={r['customer_id']} "
            f"amount={r['transaction'].get('amount')} "
            f"country={r['transaction'].get('country')} "
            f"device={r['transaction'].get('device')} "
            f"hour={r['transaction'].get('hour')} "
            f"ml_score={r['ml_score']} "
            f"decision={r['final_decision']} "
            f"top_signals={r['top_signals']} "
            f"reasoning=\"{r['llm_reasoning'][:120]}...\""
        )
    return "\n".join(lines)


# ── Chat function ─────────────────────────────────────────────────────────────

def chat(question: str) -> str:
    """
    Answer an analyst's question about recent fraud decisions.
    """
    records = _load_recent_audit(n=50)
    audit_summary = _summarize_audit(records)

    stats = {
        "total": len(records),
        "block":   sum(1 for r in records if r["final_decision"] == "BLOCK"),
        "review":  sum(1 for r in records if r["final_decision"] == "REVIEW"),
        "approve": sum(1 for r in records if r["final_decision"] == "APPROVE"),
    }

    prompt = f"""You are a fraud operations assistant at a bank.
You have access to the last {stats['total']} processed transactions.

SUMMARY STATISTICS:
- BLOCK:   {stats['block']}
- REVIEW:  {stats['review']}
- APPROVE: {stats['approve']}

RECENT AUDIT RECORDS (most recent last):
{audit_summary}

ANALYST QUESTION:
{question}

Answer clearly and concisely. Reference specific audit_ids, customer_ids, 
or risk signals when relevant. If the question cannot be answered from the 
available data, say so clearly.
"""

    try:
        response = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_ctx": 6144},
        )
        return response["message"]["content"]

    except Exception as e:
        return f"LLM unavailable: {e}"


# ── Interactive CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("Fraud Chat Agent — ask questions about recent transactions")
    print("Type 'exit' to quit")
    print("=" * 70)

    while True:
        try:
            question = input("\nAnalyst: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in ("exit", "quit", "q"):
            break

        if not question:
            continue

        print("\nAgent: ", end="", flush=True)
        answer = chat(question)
        print(answer)
