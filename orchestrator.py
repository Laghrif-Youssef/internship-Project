# orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Orchestrator
#
# Graph topology:
#
#   [ingestor] ──► [detector] ──► [llm_reasoner] ──► [action]
#                                                         │
#                                                    (audit log)
#
# Routing:
#   • If ingestor sets error → skip to END (don't process broken transactions)
#   • Otherwise linear: ingestor → detector → llm_reasoner → action → END
#
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Literal

from langgraph.graph import StateGraph, END

from state import FraudState
from agents.ingestor_agent   import ingestor_agent
from agents.detector_agent   import detector_agent
from agents.llm_reasoner_agent import llm_reasoner_agent
from agents.action_agent     import action_agent


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_ingestor(
    state: FraudState,
) -> Literal["detector", "__end__"]:
    """
    If the ingestor failed to fetch the profile, abort the pipeline.
    This prevents downstream agents from crashing on missing data.
    """
    if state.get("error"):
        print(f"[Orchestrator] Aborting — ingestor error: {state['error']}")
        return END
    return "detector"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:

    graph = StateGraph(FraudState)

    # Register nodes
    graph.add_node("ingestor",     ingestor_agent)
    graph.add_node("detector",     detector_agent)
    graph.add_node("llm_reasoner", llm_reasoner_agent)
    graph.add_node("action",       action_agent)

    # Entry point
    graph.set_entry_point("ingestor")

    # Conditional edge after ingestor
    graph.add_conditional_edges(
        "ingestor",
        route_after_ingestor,
        {
            "detector": "detector",
            END:        END,
        },
    )

    # Linear edges for the rest
    graph.add_edge("detector",     "llm_reasoner")
    graph.add_edge("llm_reasoner", "action")
    graph.add_edge("action",       END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

# Compiled graph — import this in main.py
fraud_graph = build_graph()


def process_transaction(transaction: dict) -> FraudState:
    """
    Run one transaction through the full agent pipeline.
    Returns the final state after all agents have executed.
    """
    initial_state: FraudState = {
        "transaction":    transaction,
        "profile":        {},
        "features":       {},
        "ml_score":       0.0,
        "shap_values":    {},
        "llm_reasoning":  "",
        "top_signals":    [],
        "final_decision": "",
        "action_taken":   "",
        "audit_id":       None,
        "error":          None,
    }

    return fraud_graph.invoke(initial_state)
