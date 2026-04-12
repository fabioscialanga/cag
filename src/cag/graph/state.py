"""
CAG graph state definition.
"""
from __future__ import annotations

from typing import Annotated, Callable, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class CAGState(TypedDict):
    """Shared state passed across all CAG graph nodes."""

    query: str
    question_scope: str
    retrieval_strategy: str

    chunks: list[dict]
    ranked_chunks: list[dict]
    gaps: list[str]
    relevance_score: float

    answer: str
    confidence: float
    citations: list[dict]
    hallucination_risk: float
    query_type: str
    response_language: str

    should_escalate: bool
    should_retry_reason: bool
    reason_retries: int
    error_message: str
    retry_guidance: str
    fallback_used: bool
    fallback_reason: str

    conversation_history: Annotated[list, add_messages]
    node_trace: list[str]
    relevance_threshold: float
    confidence_threshold: float
    hallucination_threshold: float
    search_fn: Callable[[str, int | None], list[Document]]
