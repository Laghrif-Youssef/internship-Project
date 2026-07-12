# generate_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Generates 5 000 customer profiles and 100 000 NORMAL training transactions.
#
# Feature set (8 features — must match detector_agent.py and ML_model.py):
#
#   ORIGINAL (5):
#     amount_ratio      — transaction amount / customer average
#     country_changed   — 1 if country != usual_country
#     device_changed    — 1 if device != usual_device
#     outside_hours     — 1 if hour outside active window
#     tx_last_hour      — number of transactions in the last hour
#
#   NEW (3) — card-testing / probing patterns:
#     low_amount_probe  — 1 if amount_ratio < 0.4 AND tx_last_hour >= 2
#                         (small amount + recent activity = possible probe)
#     amount_escalating — 1 if the last 3 transactions for this customer
#                         show a strictly increasing amount pattern
#     small_tx_count    — number of recent transactions below 15% of avg_amount
#                         (repeated micro-transactions = card testing)
#
# For NORMAL training data:
#   - low_amount_probe  : rare (2% of normal txns)
#   - amount_escalating : rare (5% of normal txns — can happen legitimately)
#   - small_tx_count    : low Poisson draw (lambda=0.3)
#
# This teaches the Isolation Forest what "normal" looks like for these
# features, so it can flag abnormally high values at inference time.
# ─────────────────────────────────────────────────────────────────────────────

import random
import json
import pandas as pd
import numpy as np
from database import get_connection

# ── Parameters ────────────────────────────────────────────────────────────────

N_CUSTOMERS    = 5000
N_TRANSACTIONS = 100000

COUNTRIES = ["MA", "FR", "ES", "DE"]
DEVICES   = ["Android", "iPhone", "Web"]

random.seed(42)
np.random.seed(42)

# ── Customer generation ───────────────────────────────────────────────────────

print("Generating customer profiles...")

customers = {}

for customer_id in range(1, N_CUSTOMERS + 1):
    customers[customer_id] = {
        "avg_amount":    random.randint(100, 10000),
        "usual_country": random.choice(COUNTRIES),
        "usual_device":  random.choice(DEVICES),
        "active_start":  random.randint(6, 10),
        "active_end":    random.randint(20, 23),
    }

conn = get_connection()
cur  = conn.cursor()

cur.execute("TRUNCATE customers RESTART IDENTITY CASCADE")

insert_sql = """
    INSERT INTO customers
        (customer_id, avg_amount, usual_country, usual_device, active_start, active_end)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

rows = [
    (cid, p["avg_amount"], p["usual_country"], p["usual_device"],
     p["active_start"], p["active_end"])
    for cid, p in customers.items()
]

cur.executemany(insert_sql, rows)
conn.commit()
cur.close()
conn.close()
print(f"Inserted {len(rows)} customers into PostgreSQL.")

# ── Transaction generation ────────────────────────────────────────────────────

print(f"Generating {N_TRANSACTIONS} normal training transactions...")

# Keep a short recent-transaction history per customer to compute
# amount_escalating and small_tx_count (last 3 amounts per customer)
customer_history: dict[int, list[float]] = {cid: [] for cid in customers}

rows = []

for _ in range(N_TRANSACTIONS):

    customer_id = random.randint(1, N_CUSTOMERS)
    profile     = customers[customer_id]
    avg_amount  = profile["avg_amount"]

    # ── Normal transaction values ─────────────────────────────────────────────

    amount = max(
        1.0,
        random.normalvariate(avg_amount, avg_amount * 0.2)
    )
    amount = round(amount, 2)

    # ── Original 5 features ──────────────────────────────────────────────────

    amount_ratio = amount / avg_amount

    # In normal data: occasional country/device/hour deviations happen
    # legitimately (travel, new phone, night shifts) but are rare.
    country_changed = int(np.random.choice([0, 1], p=[0.90, 0.10]))
    device_changed  = int(np.random.choice([0, 1], p=[0.85, 0.15]))
    outside_hours   = int(np.random.choice([0, 1], p=[0.90, 0.10]))

    # Normal velocity: Poisson(2) → most customers do 1-3 tx/hour
    tx_last_hour = int(np.random.poisson(2))

    # ── New feature 1: low_amount_probe ──────────────────────────────────────
    # For normal data: this combo is very rare (travel + small purchase etc.)
    # We simulate ~2% occurrence.
    # The actual value is computed from the real features for consistency.
    is_low_ratio    = amount_ratio < 0.4
    is_recent_burst = tx_last_hour >= 2

    if is_low_ratio and is_recent_burst:
        # Can happen legitimately ~2% of the time even for normal customers
        low_amount_probe = int(np.random.choice([0, 1], p=[0.98, 0.02]))
    else:
        low_amount_probe = 0

    # ── New feature 2: amount_escalating ─────────────────────────────────────
    # Look at the last 3 amounts for this customer.
    # Strictly increasing = possible escalation pattern.
    # For normal customers this happens ~5% of the time by chance.
    history = customer_history[customer_id]

    if len(history) >= 2:
        last_three = (history[-2:] + [amount])
        amount_escalating = int(
            all(last_three[i] < last_three[i + 1]
                for i in range(len(last_three) - 1))
        )
        # In normal data: reduce accidental escalation signal
        # by occasionally overriding with 0 (it's usually coincidence)
        if amount_escalating == 1 and random.random() < 0.95:
            amount_escalating = 0
    else:
        amount_escalating = 0

    # ── New feature 3: small_tx_count ────────────────────────────────────────
    # Count how many of the last 3 transactions were micro-transactions
    # (below 15% of customer average).
    # For normal customers: Poisson(0.3) → almost always 0
    micro_threshold = avg_amount * 0.15

    recent_micros = sum(1 for a in history[-3:] if a < micro_threshold)

    # Add a small Poisson noise to represent normal occasional micro-payments
    noise         = int(np.random.poisson(0.3))
    small_tx_count = min(recent_micros + noise, 3)   # cap at 3

    # ── Update history ────────────────────────────────────────────────────────
    history.append(amount)
    if len(history) > 3:
        history.pop(0)

    # ── Append row ────────────────────────────────────────────────────────────
    rows.append({
        "customer_id":      customer_id,
        "amount_ratio":     round(amount_ratio, 4),
        "country_changed":  country_changed,
        "device_changed":   device_changed,
        "outside_hours":    outside_hours,
        "tx_last_hour":     tx_last_hour,
        "low_amount_probe": low_amount_probe,
        "amount_escalating":amount_escalating,
        "small_tx_count":   small_tx_count,
    })

# ── Save ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)

print("\nFeature statistics for generated data:")
print(df[[
    "amount_ratio", "tx_last_hour",
    "low_amount_probe", "amount_escalating", "small_tx_count"
]].describe().round(4))

print(f"\nlow_amount_probe  == 1: {df['low_amount_probe'].sum()} "
      f"({df['low_amount_probe'].mean()*100:.2f}%)")
print(f"amount_escalating == 1: {df['amount_escalating'].sum()} "
      f"({df['amount_escalating'].mean()*100:.2f}%)")
print(f"small_tx_count    >= 1: {(df['small_tx_count']>=1).sum()} "
      f"({(df['small_tx_count']>=1).mean()*100:.2f}%)")

df.to_csv("normal_transactions.csv", index=False)
print(f"\n{len(df)} transactions saved to normal_transactions.csv")