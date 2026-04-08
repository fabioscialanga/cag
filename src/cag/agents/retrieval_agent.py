"""
Retrieval agent for evidence ranking and gap detection.
"""
from __future__ import annotations

import json
import logging
import re

from agno.agent import Agent

from cag.agents.models import RankedChunk, RetrievalOutput
from cag.llm_factory import get_agno_model

logger = logging.getLogger(__name__)

_RETRIEVAL_INSTRUCTIONS = [
    "You are the RetrievalAgent of a CAG (Cognitive Augmented Generation) system.",
    "You receive a user question and a list of retrieved documentation chunks.",
    "Your job is to score the relevance of each chunk and identify missing information.",
    "",
    "PROCESS:",
    "1. Identify the core task or question.",
    "2. Score each chunk with relevance_score (0.0-1.0).",
    "3. Record information gaps that are required to answer safely.",
    "4. Compute the overall relevance_score using the top evidence.",
    "5. If the core feature or answer is not documented, state that clearly in gaps and lower relevance.",
    "",
    "QUERY TYPE GUIDANCE:",
    "- PROCEDURAL: prioritize ordered steps, menu paths, fields, and operational sequences.",
    "- DIAGNOSTIC: prioritize symptoms, likely causes, checks, and corrective actions.",
    "- CONFIGURATION: prioritize prerequisites, settings, fields, and options.",
    "- GENERAL: prioritize definitions, constraints, timelines, and context.",
    "",
    "Return ONLY valid JSON with this structure:",
    '{"chunks_ranked": [...], "gaps": [...], "relevance_score": 0.0, "summary": "..."}',
    "Sort chunks_ranked from most relevant to least relevant.",
]


def create_retrieval_agent() -> Agent:
    """Create the configured retrieval agent."""

    return Agent(
        name="RetrievalAgent",
        model=get_agno_model(),
        role="Scores retrieved evidence and identifies information gaps.",
        instructions=_RETRIEVAL_INSTRUCTIONS,
        structured_outputs=True,
        output_schema=RetrievalOutput,
    )


_retrieval_agent: Agent | None = None


def get_retrieval_agent() -> Agent:
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = create_retrieval_agent()
    return _retrieval_agent


def _extract_json(text: str) -> str:
    """Strip optional Markdown fences and return raw JSON text."""

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text.strip()


def run_retrieval_agent(
    query: str,
    chunks: list[dict],
    query_type_hint: str = "GENERAL",
    strategy_hint: str = "semantic",
) -> RetrievalOutput:
    """Rank chunks and estimate evidence coverage for a query."""

    agent = get_retrieval_agent()
    chunks_text = "\n\n".join(
        (
            f"[CHUNK {index + 1}]\n"
            f"Source: {chunk.get('source', 'N/A')}\n"
            f"Domain module: {chunk.get('domain_module', 'general')}\n"
            f"{chunk.get('content', '')}"
        )
        for index, chunk in enumerate(chunks)
    )

    prompt = f"""USER QUERY: {query}
QUERY TYPE: {query_type_hint}
RETRIEVAL STRATEGY: {strategy_hint}

DOCUMENT CHUNKS:
{chunks_text}

Analyze the chunks and return the evaluation JSON."""

    try:
        response = agent.run(prompt)
        if isinstance(response.content, RetrievalOutput):
            return response.content

        content = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(_extract_json(content))
        return RetrievalOutput(**data)

    except Exception as exc:
        logger.error("RetrievalAgent error: %s", exc)
        return RetrievalOutput(
            chunks_ranked=[
                RankedChunk(
                    content=(
                        chunk.get("content", "")
                        if len(chunk.get("content", "")) <= 500
                        else chunk.get("content", "")[:500] + "...[TRUNCATED]"
                    ),
                    source=chunk.get("source", "N/A"),
                    domain_module=chunk.get("domain_module", "general"),
                    relevance_score=0.5,
                    relevance_reason="Fallback output after retrieval agent failure.",
                )
                for chunk in chunks[:5]
            ],
            gaps=["Unable to determine coverage gaps because the retrieval agent failed."],
            relevance_score=0.5,
            summary="Fallback retrieval output after agent failure.",
        )
