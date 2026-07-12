# agents/chat_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — Chat Agent (Text-to-SQL version)
#
# Flow:
#   1. Analyst types a question in natural language
#   2. LLM reads the DB schema and generates a PostgreSQL query
#   3. Query is validated (read-only guard) and executed against the DB
#   4. LLM receives the real query results and formats a human answer
#
# This means ALL answers come from real database queries — no guessing,
# no limited 50-record window, no hallucinated counts.
# ─────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import re
import ollama
import psycopg2

from database import get_connection, get_cursor

OLLAMA_MODEL = "qwen3:8b"

# ─────────────────────────────────────────────────────────────────────────────
# Database schema — given to the LLM so it knows what tables/columns exist
# ─────────────────────────────────────────────────────────────────────────────

DB_SCHEMA = """
PostgreSQL database with two tables:

TABLE: customers
  customer_id   INTEGER PRIMARY KEY
  avg_amount    NUMERIC       -- customer's average transaction amount
  usual_country VARCHAR(5)    -- e.g. 'MA', 'FR', 'ES', 'DE'
  usual_device  VARCHAR(20)   -- e.g. 'Android', 'iPhone', 'Web'
  active_start  INTEGER       -- hour of day (0-23) when customer typically starts transacting, e.g. 7 means 7:00 AM
  active_end    INTEGER       -- hour of day (0-23) when customer typically stops transacting, e.g. 20 means 8:00 PM

TABLE: fraud_decisions
  audit_id        VARCHAR(8)  PRIMARY KEY
  created_at      TIMESTAMPTZ -- when the decision was made
  customer_id     INTEGER     REFERENCES customers(customer_id)
  amount          NUMERIC     -- transaction amount
  country         VARCHAR(5)  -- country of the transaction
  device          VARCHAR(20) -- device used
  hour            INTEGER     -- hour of the transaction (0-23)
  tx_last_hour    INTEGER     -- number of transactions in the last hour
  amount_ratio    NUMERIC     -- amount / customer avg_amount
  country_changed BOOLEAN     -- true if country != usual_country
  device_changed  BOOLEAN     -- true if device != usual_device
  outside_hours   BOOLEAN     -- true if transaction outside active window
  ml_score        NUMERIC     -- Isolation Forest risk score (0-100)
  shap_values     JSONB       -- per-feature SHAP contributions
  llm_reasoning   TEXT        -- LLM explanation of the decision
  top_signals     JSONB       -- list of top risk signals
  final_decision  VARCHAR(10) -- 'BLOCK', 'REVIEW', or 'APPROVE'
  action_taken    TEXT        -- human-readable description of action

NOTES:
  - final_decision values are exactly: 'BLOCK', 'REVIEW', 'APPROVE' (uppercase)
  - shap_values and top_signals are JSONB arrays/objects
  - Use fd as alias for fraud_decisions, c as alias for customers
  - For date filtering: created_at is TIMESTAMPTZ, use NOW() and INTERVAL
  - To get customer profile alongside a decision: JOIN customers c USING (customer_id)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate SQL from the analyst's question
# ─────────────────────────────────────────────────────────────────────────────

SQL_PROMPT_TEMPLATE = """You are a PostgreSQL expert working for a bank fraud detection team.

DATABASE SCHEMA:
{schema}

ANALYST QUESTION:
{question}

Generate a single valid PostgreSQL SELECT query to answer this question.

RULES:
- Only SELECT queries allowed — never INSERT, UPDATE, DELETE, DROP, or TRUNCATE
- Use table aliases: fd for fraud_decisions, c for customers
- Limit results to 100 rows maximum unless the question asks for counts/aggregates
- For "recent" or "latest" without a specific time: use the last 24 hours
- For questions about a specific customer: filter by customer_id
- For questions about blocked/reviewed/approved: filter final_decision
- Return ONLY the SQL query — no explanation, no markdown, no backticks
"""

def _generate_sql(question: str) -> str:
    """Ask the LLM to write a SQL query for the analyst's question."""

    prompt = SQL_PROMPT_TEMPLATE.format(
        schema=DB_SCHEMA,
        question=question,
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 4096},
    )

    raw = response["message"]["content"]

    # Strip think tags (Qwen3), markdown fences, leading/trailing whitespace
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:sql)?|```", "", raw)
    raw = raw.strip()

    # Take only the first statement if the model generated multiple
    if ";" in raw:
        raw = raw.split(";")[0].strip()

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Validate and execute the SQL query
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def _execute_query(sql: str) -> tuple[list[str], list[tuple]]:
    """
    Execute a read-only SQL query.
    Returns (column_names, rows).
    Raises ValueError if the query is not a SELECT or contains forbidden keywords.
    """
    # Safety guard: only allow SELECT queries
    clean = sql.strip().upper()
    if not clean.startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are allowed. Got: {sql[:60]}")

    if FORBIDDEN.search(sql):
        raise ValueError(f"Query contains forbidden keyword: {sql[:60]}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows    = cur.fetchall()

    return columns, rows


def _format_results(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as a readable text table for the LLM."""
    if not rows:
        return "Query returned 0 rows."

    # Convert rows to list of dicts
    records = [dict(zip(columns, row)) for row in rows]

    # For large result sets, summarize instead of listing everything
    if len(records) > 20:
        return (
            f"Query returned {len(records)} rows. First 20:\n" +
            json.dumps(records[:20], indent=2, default=str)
        )

    return json.dumps(records, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Format the result into a human answer
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_PROMPT_TEMPLATE = """You are a fraud operations assistant at a bank.

IMPORTANT FIELD DEFINITIONS:
- active_start and active_end are HOURS OF THE DAY (0-23), not months.
  Example: active_start=7, active_end=20 means the customer is active from 7:00 AM to 8:00 PM.
- amount and avg_amount are transaction amounts in the local currency.
- ml_score is a risk score from 0 to 100 (higher = more suspicious).

The analyst asked:
"{question}"

You ran this SQL query:
{sql}

The query returned:
{results}

Write a clear, concise answer to the analyst's question based on the query results.
- Reference specific values from the results (customer IDs, amounts, decisions, etc.)
- If the result is a count or aggregate, state it directly
- If the result is empty, say so clearly
- Keep the answer focused — do not add unsolicited information
"""

def _format_answer(question: str, sql: str, results_text: str) -> str:
    """Ask the LLM to turn raw query results into a human-readable answer."""

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        sql=sql,
        results=results_text,
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_ctx": 4096},
    )

    raw = response["message"]["content"]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Main chat function — orchestrates all 3 steps
# ─────────────────────────────────────────────────────────────────────────────

def chat(question: str) -> str:
    """
    Answer an analyst's question using Text-to-SQL:
      1. Generate SQL from the question
      2. Execute it against PostgreSQL
      3. Format the result into a human answer
    """

    # ── Step 1: Generate SQL ─────────────────────────────────────────────────
    try:
        sql = _generate_sql(question)
        print(f"\n[Chat] Generated SQL:\n  {sql}")
    except Exception as e:
        return f"Failed to generate SQL query: {e}"

    # ── Step 2: Execute ──────────────────────────────────────────────────────
    try:
        columns, rows = _execute_query(sql)
        results_text  = _format_results(columns, rows)
        print(f"[Chat] Query returned {len(rows)} row(s)")
    except ValueError as e:
        return f"Query rejected (security): {e}"
    except Exception as e:
        # If the SQL has a syntax error, tell the analyst and show the query
        return (
            f"Database error: {e}\n\n"
            f"Generated query was:\n{sql}\n\n"
            "Please rephrase your question."
        )

    # ── Step 3: Format answer ─────────────────────────────────────────────────
    try:
        answer = _format_answer(question, sql, results_text)
        return answer
    except Exception as e:
        # If formatting fails, return raw results as fallback
        return f"Query succeeded but formatting failed ({e}).\n\nRaw results:\n{results_text}"


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("Fraud Chat Agent — Text-to-SQL")
    print("Ask questions about transactions and fraud decisions.")
    print("Type 'exit' to quit.")
    print("=" * 70)
    print("\nExample questions:")
    print("  - How many transactions were blocked today?")
    print("  - What is the profile of customer 3528?")
    print("  - Show me the last 5 blocked transactions")
    print("  - Which customer had the highest risk score?")
    print("  - How many transactions had country_changed = true?")
    print("  - What was the average ml_score for BLOCK decisions?")

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

        answer = chat(question)
        print(f"\nAgent: {answer}")