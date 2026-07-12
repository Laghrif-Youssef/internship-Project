# simulator_server.py

from fastapi import FastAPI, HTTPException
from threading import Thread
from kafka import KafkaProducer
import json
import random
import time
from database import get_connection, get_cursor

app = FastAPI()

# =====================================================
# CUSTOMER PROFILES
# =====================================================

with open("customers.json", "r") as f:
    customers = json.load(f)
    
customers = {
    int(k): v
    for k, v in customers.items()
}

# =====================================================
# KAFKA PRODUCER
# =====================================================

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

TOPIC = "transactions"

# =====================================================
# PROFILE API
# =====================================================

@app.get("/profile/{customer_id}")
def get_profile(customer_id: int):
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT * FROM customers WHERE customer_id = %s",
                (customer_id,)
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return dict(row)

# =====================================================
# TRANSACTION GENERATION
# =====================================================

COUNTRIES = [
    "MA",
    "FR",
    "ES",
    "DE",
    "US",
    "RU"
]

DEVICES = [
    "Android",
    "iPhone",
    "Web"
]


def generate_normal_transaction(customer_id, profile):

    amount = round(
        random.normalvariate(
            profile["avg_amount"],
            profile["avg_amount"] * 0.2
        ),
        2
    )

    hour = random.randint(
        profile["active_start"],
        profile["active_end"]
    )

    tx_last_hour = max(
        0,
        int(random.normalvariate(2, 1))
    )

    return {
        "customer_id": customer_id,
        "amount": max(amount, 1),
        "country": profile["usual_country"],
        "device": profile["usual_device"],
        "hour": hour,
        "tx_last_hour": tx_last_hour
    }


def generate_anomalous_transaction(customer_id, profile):

    amount = round(
        profile["avg_amount"]
        * random.randint(20, 100),
        2
    )

    country = random.choice(
        [c for c in COUNTRIES
         if c != profile["usual_country"]]
    )

    device = random.choice(
        [d for d in DEVICES
         if d != profile["usual_device"]]
    )

    hour = random.randint(0, 5)

    tx_last_hour = random.randint(15, 50)

    return {
        "customer_id": customer_id,
        "amount": amount,
        "country": country,
        "device": device,
        "hour": hour,
        "tx_last_hour": tx_last_hour
    }


def transaction_loop():

    while True:

        customer_id = random.choice(
            list(customers.keys())
        )

        profile = customers[customer_id]

        # 95% normal
        # 5% anomalous

        if random.random() < 0.95:

            transaction = generate_normal_transaction(
                customer_id,
                profile
            )

        else:

            transaction = generate_anomalous_transaction(
                customer_id,
                profile
            )

        producer.send(
            TOPIC,
            transaction
        )

        producer.flush()

        print(
            f"Sent transaction: {transaction}"
        )

        time.sleep(8)

# =====================================================
# STARTUP
# =====================================================

@app.on_event("startup")
def startup_event():

    thread = Thread(
        target=transaction_loop,
        daemon=True
    )

    thread.start()

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )