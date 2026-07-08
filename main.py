# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point — Kafka consumer loop
#
# How to run:
#   python main.py
#
# Prerequisites:
#   1. Kafka running:       docker-compose up -d
#   2. Simulator running:   python simulator_server.py
#   3. Models present:      isolation_forest.pkl + risk_scaler.pkl
#                           (train with: python ML_model.py)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import signal
from kafka import KafkaConsumer

from state import KAFKA_BROKER, TRANSACTION_TOPIC
from orchestrator import process_transaction

import json

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_running = True

def _shutdown(signum, frame):
    global _running
    print("\n[Main] Shutting down gracefully...")
    _running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Kafka consumer ────────────────────────────────────────────────────────────

def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TRANSACTION_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="latest",   # only process new transactions
        enable_auto_commit=True,
        group_id="fraud-agent-v2",
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():

    print("=" * 70)
    print("  Fraud Detection — Multi-Agent System (LangGraph + Ollama + SHAP)")
    print("=" * 70)

    consumer = build_consumer()
    print(f"[Main] Connected to Kafka at {KAFKA_BROKER}")
    print(f"[Main] Listening on topic: {TRANSACTION_TOPIC}")
    print("[Main] Waiting for transactions...\n")

    processed = 0
    errors     = 0

    while _running:

        records = consumer.poll(timeout_ms=1000)

        for tp, messages in records.items():
            for message in messages:

                if not _running:
                    break

                transaction = message.value

                print("\n" + "=" * 70)
                print(f"[Main] New transaction received (#{processed + 1})")

                try:
                    final_state = process_transaction(transaction)
                    processed += 1

                    print(f"\n[Main] ✅ Pipeline complete")
                    print(f"       Decision   : {final_state['final_decision']}")
                    print(f"       ML Score   : {final_state['ml_score']}/100")
                    print(f"       Top signals: {final_state['top_signals']}")
                    print(f"       Audit ID   : {final_state['audit_id']}")

                except Exception as e:
                    errors += 1
                    print(f"[Main] ❌ Unhandled pipeline error: {e}")

    consumer.close()
    print(f"\n[Main] Stopped. Processed: {processed} | Errors: {errors}")


if __name__ == "__main__":
    main()
