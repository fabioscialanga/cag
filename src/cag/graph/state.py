"""
CAG graph state definition.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


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

    should_escalate: bool
    reason_retries: int
    error_message: str

    conversation_history: Annotated[list, add_messages]
    node_trace: list[str]
