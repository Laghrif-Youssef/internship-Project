import random
import pandas as pd
import numpy as np
import json

# =====================================================
# PARAMETERS
# =====================================================

N_CUSTOMERS = 5000
N_TRANSACTIONS = 100000

COUNTRIES = ["MA", "FR", "ES", "DE"]
DEVICES = ["Android", "iPhone", "Web"]

# =====================================================
# CUSTOMER GENERATION
# =====================================================

customers = {}

for customer_id in range(1, N_CUSTOMERS + 1):

    customers[customer_id] = {

        "avg_amount": random.randint(100, 10000),

        "usual_country": random.choice(COUNTRIES),

        "usual_device": random.choice(DEVICES),

        "active_start": random.randint(6, 10),

        "active_end": random.randint(20, 23)
    }
with open("customers.json", "w") as f:
    json.dump(customers, f, indent=4)

# =====================================================
# DATASET GENERATION
# =====================================================

rows = []

for _ in range(N_TRANSACTIONS):

    customer_id = random.randint(1, N_CUSTOMERS)

    profile = customers[customer_id]

    # ---------------------------------------------
    # Generate a NORMAL transaction
    # ---------------------------------------------

    amount = max(
        1,
        random.normalvariate(
            profile["avg_amount"],
            profile["avg_amount"] * 0.2
        )
    )

    country = profile["usual_country"]

    device = profile["usual_device"]

    hour = random.randint(
        profile["active_start"],
        profile["active_end"]
    )

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    amount_ratio = amount / profile["avg_amount"]

    country_changed = np.random.choice(
    [0,1],
    p=[0.9,0.1]
)

    device_changed = np.random.choice(
        [0,1],
        p=[0.85,0.15]
    )

    outside_hours = np.random.choice(
        [0,1],
        p=[0.9,0.1]
    )

    tx_last_hour = np.random.poisson(2)

    rows.append({

        "customer_id": customer_id,

        "amount_ratio": amount_ratio,

        "country_changed": country_changed,

        "device_changed": device_changed,

        "outside_hours": outside_hours,

        "tx_last_hour": tx_last_hour

    })

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame(rows)

print(df.head())

df.to_csv(
    "normal_transactions.csv",
    index=False
)

print(f"{len(df)} transactions generated.")