"""
CAG graph assembly and execution entrypoints.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from cag.graph.nodes import (
    entry_node,
    exit_node,
    reason_node,
    refine_node,
    retrieve_node,
    route_after_refine,
    route_after_validate,
    validate_node,
)
from cag.graph.state import CAGState

logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the CAG graph."""

    builder = StateGraph(CAGState)

    builder.add_node("entry", entry_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("refine", refine_node)
    builder.add_node("reason", reason_node)
    builder.add_node("validate", validate_node)
    builder.add_node("exit", exit_node)

    builder.add_edge(START, "entry")
    builder.add_edge("entry", "retrieve")
    builder.add_edge("retrieve", "refine")
    builder.add_edge("reason", "validate")
    builder.add_edge("exit", END)

    builder.add_conditional_edges(
        "refine",
        route_after_refine,
        {"reason": "reason", "validate": "validate"},
    )
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"exit": "exit", "reason": "reason"},
    )

    graph = builder.compile()
    logger.info("CAG graph compiled successfully")
    return graph


_graph = None


def get_graph():
    """Return the compiled graph singleton."""

    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, conversation_history: list | None = None) -> dict:
    """Run a query through the CAG graph and return the final state."""

    graph = get_graph()

    initial_state: CAGState = {
        "query": query,
        "question_scope": "domain",
        "retrieval_strategy": "semantic",
        "chunks": [],
        "ranked_chunks": [],
        "gaps": [],
        "relevance_score": 0.0,
        "answer": "",
        "confidence": 0.0,
        "citations": [],
        "hallucination_risk": 0.0,
        "query_type": "GENERAL",
        "should_escalate": False,
        "reason_retries": 0,
        "error_message": "",
        "node_trace": [],
        "conversation_history": conversation_history or [],
    }

    logger.info("=== CAG Query: '%s' ===", query[:80])
    final_state = graph.invoke(initial_state)
    logger.info(
        "=== CAG Done: trace=%s ===",
        " -> ".join(final_state.get("node_trace", [])),
    )
    return final_state
