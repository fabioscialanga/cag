from __future__ import annotations

import json
import logging
import math
import re
import time
from contextlib import contextmanager
from typing import Callable
from unittest.mock import patch

from agno.agent import Agent
from pydantic import BaseModel, Field

from cag.config import settings
from cag.eval.models import CitationRecord, SystemOutput
from cag.graph.graph import run_query
from cag.llm_factory import get_agno_model

logger = logging.getLogger(__name__)


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


@contextmanager
def patched_similarity_search(search_fn: Callable[[str, int], list], top_k: int):
    def _side_effect(query: str, k: int | None = None):
        return search_fn(query, k or top_k)

    with patch("cag.graph.nodes.similarity_search", side_effect=_side_effect):
        yield


def run_cag_system(
    question_id: str,
    question: str,
    search_fn: Callable[[str, int], list],
    top_k: int,
    query_runner: Callable[[str, list | None], dict] = run_query,
) -> SystemOutput:
    retrieved_docs = search_fn(question, top_k)
    retrieval_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    started = time.perf_counter()
    with patched_similarity_search(search_fn, top_k):
        result = query_runner(question, conversation_history=[])
    latency_ms = (time.perf_counter() - started) * 1000

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
        query_type=payload.query_type,
        confidence=payload.confidence,
        hallucination_risk=payload.hallucination_risk,
        should_escalate=payload.should_escalate,
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
    if system == "rag_baseline":
        return run_rag_baseline(question_id, question, search_fn, effective_top_k)
    if system == "direct_baseline":
        return run_direct_baseline(question_id, question)
    if system == "lightrag_baseline":
        if runtime is None:
            raise ValueError("LightRAG baseline requires a prepared runtime.")
        return runtime.query(question_id, question)
    raise ValueError(f"Unsupported evaluation system: {system}")
