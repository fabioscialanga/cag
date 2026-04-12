from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Callable

from agno.agent import Agent
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from cag.config import settings
from cag.eval.models import CitationRecord, SystemOutput
from cag.graph.graph import run_query
from cag.graph.runtime import resolve_runtime_config
from cag.graph.nodes import (
    entry_node,
    exit_node,
    reason_node,
    select_context_node,
    retrieve_node,
    route_after_select_context,
    route_after_validate,
    validate_node,
)
from cag.llm_factory import get_agno_model

logger = logging.getLogger(__name__)
REASONING_CONTEXT_LIMIT = 6
SearchFn = Callable[[str, int | None], list[Document]]


class BaselineGeneration(BaseModel):
    answer: str
    query_type: str = "GENERAL"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: list[CitationRecord] = Field(default_factory=list)
    hallucination_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    should_escalate: bool = False
    insufficiency_reason: str = ""


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def estimate_cost_units(prompt_text: str, output_text: str, multiplier: float = 1.0) -> float:
    total_tokens = estimate_tokens(prompt_text) + estimate_tokens(output_text)
    return round((total_tokens / 1000) * multiplier, 3)


def _chunk_to_dict(doc) -> dict:
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("filename", doc.metadata.get("source", "N/A")),
        "domain_module": doc.metadata.get("domain_module", "general"),
        "chunk_index": doc.metadata.get("chunk_index", 0),
    }


def _chunk_identity(chunk: dict) -> tuple[str, int, str]:
    return (
        str(chunk.get("source", "N/A")),
        int(chunk.get("chunk_index", 0)),
        str(chunk.get("content", ""))[:160],
    )


def _selected_context_from_state(state: dict) -> list[dict]:
    node_trace = [str(node) for node in state.get("node_trace", [])]
    if not any(node.startswith("REASON") for node in node_trace):
        return []
    return list(state.get("ranked_chunks", []))[:REASONING_CONTEXT_LIMIT]


def _selected_context_sources(selected_context: list[dict]) -> list[str]:
    return [str(chunk.get("source", "")) for chunk in selected_context if chunk.get("source")]


def _build_initial_cag_state(query: str, conversation_history: list | None = None) -> dict:
    runtime_config = resolve_runtime_config()
    return {
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
        "response_language": "en",
        "should_escalate": False,
        "should_retry_reason": False,
        "reason_retries": 0,
        "error_message": "",
        "retry_guidance": "",
        "fallback_used": False,
        "fallback_reason": "",
        "node_trace": [],
        "conversation_history": conversation_history or [],
        "relevance_threshold": runtime_config.relevance_threshold,
        "confidence_threshold": runtime_config.confidence_threshold,
        "hallucination_threshold": runtime_config.hallucination_threshold,
        "search_fn": None,
    }


def _restore_raw_chunk_order(raw_chunks: list[dict], ranked_chunks: list[dict]) -> list[dict]:
    ranked_by_identity = {_chunk_identity(chunk): chunk for chunk in ranked_chunks}
    ordered_chunks: list[dict] = []

    for raw_chunk in raw_chunks:
        ranked_match = ranked_by_identity.get(_chunk_identity(raw_chunk), {})
        ordered_chunks.append(
            {
                "content": raw_chunk.get("content", ""),
                "source": raw_chunk.get("source", "N/A"),
                "domain_module": raw_chunk.get("domain_module", "general"),
                "chunk_index": raw_chunk.get("chunk_index", 0),
                "cluster_id": ranked_match.get("cluster_id", "cluster_1"),
                "selection_category": ranked_match.get("selection_category", "general"),
                "relevance_score": float(ranked_match.get("relevance_score", 0.0)),
                "relevance_reason": ranked_match.get(
                    "relevance_reason",
                    "Raw retrieval order preserved for cag_no_selection baseline.",
                ),
            }
        )

    return ordered_chunks


def _run_cag_no_selection_query(
    query: str,
    conversation_history: list | None = None,
    search_fn: SearchFn | None = None,
) -> dict:
    state = _build_initial_cag_state(query, conversation_history)
    if search_fn is not None:
        state["search_fn"] = search_fn
    state.update(entry_node(state))
    state.update(retrieve_node(state))

    raw_chunks = list(state.get("chunks", []))
    state.update(select_context_node(state))

    if route_after_select_context(state) == "reason":
        state["ranked_chunks"] = _restore_raw_chunk_order(raw_chunks, state.get("ranked_chunks", []))
        while True:
            state.update(reason_node(state))
            state.update(validate_node(state))
            if route_after_validate(state) != "reason":
                break
    else:
        state.update(validate_node(state))

    state.update(exit_node(state))
    return state


def _citations_from_raw(raw_citations: list[dict] | list[CitationRecord]) -> list[CitationRecord]:
    citations: list[CitationRecord] = []
    for citation in raw_citations:
        if isinstance(citation, CitationRecord):
            citations.append(citation)
        else:
            citations.append(CitationRecord(**citation))
    return citations


def create_baseline_agent(system: str) -> Agent:
    use_retrieval = system == "rag_baseline"
    instructions = [
        "You are a benchmark baseline for grounded question answering.",
        "Answer only using the supplied evidence.",
        "If the evidence is insufficient, set should_escalate=true and explain why briefly.",
        "Do not invent values, paths, procedures, or sources.",
        "Return only valid JSON.",
    ]
    if use_retrieval:
        instructions.append("This is a standard retrieval-plus-generation baseline with no reranking or validation loop.")
    else:
        instructions.append("This is a direct LLM baseline with no retrieval context.")

    return Agent(
        name=f"{system}_agent",
        model=get_agno_model(),
        role="Produces a one-shot grounded answer for benchmark evaluation.",
        instructions=instructions,
        structured_outputs=True,
        output_schema=BaselineGeneration,
    )


def run_cag_system(
    question_id: str,
    question: str,
    search_fn: SearchFn,
    top_k: int,
    query_runner: Callable[..., dict] = run_query,
) -> SystemOutput:
    retrieved_docs = search_fn(question, top_k)
    retrieval_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    started = time.perf_counter()
    result = query_runner(question, conversation_history=[], search_fn=search_fn)
    latency_ms = (time.perf_counter() - started) * 1000

    selected_context = _selected_context_from_state(result)
    final_chunks = result.get("chunks") or [_chunk_to_dict(doc) for doc in retrieved_docs]
    reason_calls = max(1, sum(1 for node in result.get("node_trace", []) if str(node).startswith("REASON")))
    prompt_estimate = f"{question}\n\n{retrieval_context}"
    cost_estimate = estimate_cost_units(prompt_estimate, result.get("answer", ""), multiplier=reason_calls + 1)

    return SystemOutput(
        question_id=question_id,
        question=question,
        system="cag",
        answer=result.get("answer", ""),
        citations=_citations_from_raw(result.get("citations", [])),
        query_type=result.get("query_type", "GENERAL"),
        confidence=float(result.get("confidence", 0.0)),
        hallucination_risk=float(result.get("hallucination_risk", 1.0)),
        should_escalate=bool(result.get("should_escalate", False)),
        fallback_used=bool(result.get("fallback_used", False)),
        fallback_reason=result.get("fallback_reason") or None,
        selected_context_sources=_selected_context_sources(selected_context),
        retrieved_chunk_count=len(final_chunks),
        selected_chunk_count=len(selected_context),
        latency_ms=round(latency_ms, 2),
        cost_estimate=cost_estimate,
        node_trace=[str(node) for node in result.get("node_trace", [])],
    )


def run_cag_no_selection(
    question_id: str,
    question: str,
    search_fn: SearchFn,
    top_k: int,
    query_runner: Callable[..., dict] = _run_cag_no_selection_query,
) -> SystemOutput:
    retrieved_docs = search_fn(question, top_k)
    retrieval_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    started = time.perf_counter()
    result = query_runner(question, conversation_history=[], search_fn=search_fn)
    latency_ms = (time.perf_counter() - started) * 1000

    selected_context = _selected_context_from_state(result)
    final_chunks = result.get("chunks") or [_chunk_to_dict(doc) for doc in retrieved_docs]
    reason_calls = max(1, sum(1 for node in result.get("node_trace", []) if str(node).startswith("REASON")))
    prompt_estimate = f"{question}\n\n{retrieval_context}"
    cost_estimate = estimate_cost_units(prompt_estimate, result.get("answer", ""), multiplier=reason_calls + 1)

    return SystemOutput(
        question_id=question_id,
        question=question,
        system="cag_no_selection",
        answer=result.get("answer", ""),
        citations=_citations_from_raw(result.get("citations", [])),
        query_type=result.get("query_type", "GENERAL"),
        confidence=float(result.get("confidence", 0.0)),
        hallucination_risk=float(result.get("hallucination_risk", 1.0)),
        should_escalate=bool(result.get("should_escalate", False)),
        fallback_used=bool(result.get("fallback_used", False)),
        fallback_reason=result.get("fallback_reason") or None,
        selected_context_sources=_selected_context_sources(selected_context),
        retrieved_chunk_count=len(final_chunks),
        selected_chunk_count=len(selected_context),
        latency_ms=round(latency_ms, 2),
        cost_estimate=cost_estimate,
        node_trace=[str(node) for node in result.get("node_trace", [])],
    )


def run_rag_baseline(
    question_id: str,
    question: str,
    search_fn: Callable[[str, int], list],
    top_k: int,
    agent: Agent | None = None,
) -> SystemOutput:
    agent = agent or create_baseline_agent("rag_baseline")
    retrieved_docs = search_fn(question, top_k)
    chunks = [_chunk_to_dict(doc) for doc in retrieved_docs]
    context = "\n\n".join(
        f"[SOURCE {index + 1}: {chunk['source']}]\n{chunk['content']}"
        for index, chunk in enumerate(chunks)
    )
    prompt = f"""QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Return the one-shot JSON answer now."""

    started = time.perf_counter()
    try:
        response = agent.run(prompt)
    except Exception as exc:
        logger.error("RAG baseline agent error: %s", exc)
        response = None
    latency_ms = (time.perf_counter() - started) * 1000

    payload = BaselineGeneration(
        answer="Unable to generate a baseline answer.",
        confidence=0.0,
        hallucination_risk=1.0,
        should_escalate=True,
    )

    if response is not None:
        if isinstance(response.content, BaselineGeneration):
            payload = response.content
        else:
            content = response.content if isinstance(response.content, str) else str(response.content)
            try:
                payload = BaselineGeneration(**json.loads(_extract_json(content)))
            except (json.JSONDecodeError, Exception) as exc:
                logger.error("RAG baseline JSON parse error: %s | raw: %s", exc, content[:200])

    return SystemOutput(
        question_id=question_id,
        question=question,
        system="rag_baseline",
        answer=payload.answer,
        citations=_citations_from_raw(payload.citations),
        selected_context_sources=[chunk["source"] for chunk in chunks if chunk.get("source")],
        query_type=payload.query_type,
        confidence=payload.confidence,
        hallucination_risk=payload.hallucination_risk,
        should_escalate=payload.should_escalate,
        retrieved_chunk_count=len(chunks),
        selected_chunk_count=len(chunks),
        latency_ms=round(latency_ms, 2),
        cost_estimate=estimate_cost_units(prompt, payload.answer),
        node_trace=["RETRIEVE", "GENERATE"],
    )


def run_direct_baseline(
    question_id: str,
    question: str,
    agent: Agent | None = None,
) -> SystemOutput:
    agent = agent or create_baseline_agent("direct_baseline")
    prompt = f"""QUESTION:
{question}

No retrieval context is available. If the question cannot be answered safely, escalate.
Return the one-shot JSON answer now."""

    started = time.perf_counter()
    try:
        response = agent.run(prompt)
    except Exception as exc:
        logger.error("Direct baseline agent error: %s", exc)
        response = None
    latency_ms = (time.perf_counter() - started) * 1000

    payload = BaselineGeneration(
        answer="Unable to generate a baseline answer.",
        confidence=0.0,
        hallucination_risk=1.0,
        should_escalate=True,
    )

    if response is not None:
        if isinstance(response.content, BaselineGeneration):
            payload = response.content
        else:
            content = response.content if isinstance(response.content, str) else str(response.content)
            try:
                payload = BaselineGeneration(**json.loads(_extract_json(content)))
            except (json.JSONDecodeError, Exception) as exc:
                logger.error("Direct baseline JSON parse error: %s | raw: %s", exc, content[:200])

    return SystemOutput(
        question_id=question_id,
        question=question,
        system="direct_baseline",
        answer=payload.answer,
        citations=_citations_from_raw(payload.citations),
        query_type=payload.query_type,
        confidence=payload.confidence,
        hallucination_risk=payload.hallucination_risk,
        should_escalate=payload.should_escalate,
        latency_ms=round(latency_ms, 2),
        cost_estimate=estimate_cost_units(prompt, payload.answer),
        node_trace=["GENERATE"],
    )


def run_system(
    system: str,
    question_id: str,
    question: str,
    search_fn: Callable[[str, int], list],
    top_k: int | None = None,
    runtime=None,
) -> SystemOutput:
    effective_top_k = top_k or settings.retrieval_top_k
    if system == "cag":
        return run_cag_system(question_id, question, search_fn, effective_top_k)
    if system == "cag_no_selection":
        return run_cag_no_selection(question_id, question, search_fn, effective_top_k)
    if system == "rag_baseline":
        return run_rag_baseline(question_id, question, search_fn, effective_top_k)
    if system == "direct_baseline":
        return run_direct_baseline(question_id, question)
    if system == "lightrag_baseline":
        if runtime is None:
            raise ValueError("LightRAG baseline requires a prepared runtime.")
        return runtime.query(question_id, question)
    raise ValueError(f"Unsupported evaluation system: {system}")
