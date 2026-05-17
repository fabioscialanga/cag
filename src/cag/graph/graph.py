"""
CAG graph assembly and execution entrypoints.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langchain_core.documents import Document

from cag.config import settings
from cag.graph.nodes import (
    contextualize_query_node,
    entry_node,
    exit_node,
    post_grounding_node,
    reason_node,
    select_context_node,
    retrieve_node,
    route_after_select_context,
    route_after_validate,
    validate_node,
)
from cag.graph.runtime import RuntimeConfig, resolve_runtime_config
from cag.graph.state import CAGState
from cag.ingestion.embedder import similarity_search as default_similarity_search

logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the CAG graph."""

    builder = StateGraph(CAGState)

    builder.add_node("entry", entry_node)
    builder.add_node("contextualize_query", contextualize_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("select_context", select_context_node)
    builder.add_node("reason", reason_node)
    builder.add_node("post_grounding", post_grounding_node)
    builder.add_node("validate", validate_node)
    builder.add_node("exit", exit_node)

    builder.add_edge(START, "entry")
    builder.add_edge("entry", "contextualize_query")
    builder.add_edge("contextualize_query", "retrieve")
    builder.add_edge("retrieve", "select_context")
    builder.add_edge("reason", "post_grounding")
    builder.add_edge("post_grounding", "validate")
    builder.add_edge("exit", END)

    builder.add_conditional_edges(
        "select_context",
        route_after_select_context,
        {"reason": "reason", "validate": "validate"},
    )
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"exit": "exit", "reason": "reason", "retrieve": "retrieve"},
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


def run_query(
    query: str,
    conversation_history: list | None = None,
    runtime_config: RuntimeConfig | None = None,
    search_fn: Callable[[str, int | None], list[Document]] | None = None,
    access_filter: dict | None = None,
    original_query: str | None = None,
) -> dict:
    """Run a query through the CAG graph and return the final state."""

    graph = get_graph()
    resolved_runtime = resolve_runtime_config(runtime_config)
    active_search_fn = search_fn or default_similarity_search

    initial_state: CAGState = {
        "query": query,
        "original_query": original_query or query,
        "modified_query": query,
        "question_scope": "domain",
        "retrieval_strategy": "semantic",
        "intent": {},
        "retrieval_plan": {},
        "access_filter": access_filter or {},
        "chunks": [],
        "ranked_chunks": [],
        "document_candidates": [],
        "gaps": [],
        "relevance_score": 0.0,
        "answer": "",
        "confidence": 0.0,
        "citations": [],
        "hallucination_risk": 0.0,
        "query_type": "GENERAL",
        "response_language": "en",
        "should_escalate": False,
        "should_retry_reason": False,
        "should_retry_retrieval": False,
        "retrieval_retry_used": False,
        "reason_retries": 0,
        "error_message": "",
        "retry_guidance": "",
        "fallback_used": False,
        "fallback_reason": "",
        "grounding_checks": [],
        "unsupported_claims": [],
        "post_grounding_status": "pending",
        "suggested_actions": [],
        "node_trace": [],
        "conversation_history": conversation_history or [],
        "relevance_threshold": resolved_runtime.relevance_threshold,
        "confidence_threshold": resolved_runtime.confidence_threshold,
        "hallucination_threshold": resolved_runtime.hallucination_threshold,
        "retrieval_top_k": resolved_runtime.retrieval_top_k,
        "search_fn": active_search_fn,
    }

    logger.info("=== CAG Query: '%s' ===", query[:80])
    final_state = graph.invoke(initial_state)
    logger.info(
        "=== CAG Done: trace=%s ===",
        " -> ".join(final_state.get("node_trace", [])),
    )
    return final_state
