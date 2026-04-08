"""
Reasoning agent for grounded answer generation.
"""
from __future__ import annotations

import json
import logging
import re

from agno.agent import Agent

from cag.agents.models import ReasoningOutput
from cag.llm_factory import get_agno_model

logger = logging.getLogger(__name__)

_REASONING_INSTRUCTIONS = [
    "You are the ReasoningAgent of a CAG (Cognitive Augmented Generation) system.",
    "You receive a user question, ranked documentation chunks, and explicit information gaps.",
    "Your job is to generate a structured, grounded, cited answer.",
    "",
    "QUERY TYPES AND RESPONSE SHAPE:",
    "- DIAGNOSTIC: structure as Cause -> Check -> Resolution.",
    "- PROCEDURAL: write a short numbered sequence of steps.",
    "- CONFIGURATION: focus on prerequisites, fields, parameters, and options.",
    "- GENERAL: provide a concise factual explanation with relevant context.",
    "",
    "CORE RULES:",
    "1. Every claim must be supported by the provided chunks.",
    "2. Never invent steps, settings, values, paths, or decisions that are not in the evidence.",
    "3. If the chunks do not cover the request, say so explicitly.",
    "4. If the core concept is unsupported, do not answer by analogy.",
    "5. For insufficient coverage set confidence <= 0.35 and hallucination_risk >= 0.80.",
    "6. For partial coverage, answer only the supported portion and state what is missing.",
    "7. Always include citations.",
    "8. Answer the core request first and avoid unrelated prerequisites or side topics.",
    "9. Reuse field names, step names, and labels exactly as they appear in the evidence when possible.",
    "10. If the chunks already contain a numbered procedure or cause/solution wording, preserve that phrasing as much as possible.",
    "11. Avoid generic closing filler or meta commentary unless the question requires it.",
    "",
    "Return ONLY valid JSON with this structure:",
    '{"answer": "...", "query_type": "DIAGNOSTIC|PROCEDURAL|CONFIGURATION|GENERAL",'
    ' "confidence": 0.0, "citations": [...], "hallucination_risk": 0.0, "hallucination_reason": "..."}',
]

MODE_INSTRUCTIONS = {
    "DIAGNOSTIC": (
        "Structure the answer as Cause and Resolution. Reuse the verbs and nouns from the source chunks "
        "when they already describe checks or corrective actions."
    ),
    "PROCEDURAL": (
        "Write short numbered steps. If the evidence already contains ordered steps, preserve the same "
        "sequence and wording as much as possible. Do not add related processes that were not requested."
    ),
    "CONFIGURATION": (
        "Highlight only the prerequisites, fields, parameters, and options required by the question. "
        "Do not recommend undocumented settings."
    ),
    "GENERAL": (
        "Answer briefly but stay anchored to the evidence. If the question asks 'what' or 'which', "
        "prefer a factual list over a step-by-step procedure."
    ),
}


_reasoning_agent: Agent | None = None


def get_reasoning_agent() -> Agent:
    """Return the configured reasoning agent singleton."""

    global _reasoning_agent
    if _reasoning_agent is None:
        _reasoning_agent = Agent(
            name="ReasoningAgent",
            model=get_agno_model(),
            role="Builds grounded, structured answers with citations and confidence scores.",
            instructions=_REASONING_INSTRUCTIONS,
            structured_outputs=True,
            output_schema=ReasoningOutput,
        )
    return _reasoning_agent


def _extract_json(text: str) -> str:
    """Strip optional Markdown fences and return raw JSON text."""

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text.strip()


def run_reasoning_agent(
    query: str,
    ranked_chunks: list[dict],
    gaps: list[str],
    query_type_hint: str = "GENERAL",
) -> ReasoningOutput:
    """Generate a grounded answer from ranked evidence."""

    agent = get_reasoning_agent()

    ranked_serialized = []
    for chunk in ranked_chunks[:6]:
        if hasattr(chunk, "model_dump"):
            try:
                ranked_serialized.append(chunk.model_dump())
            except Exception:
                ranked_serialized.append(chunk)
        else:
            ranked_serialized.append(chunk)

    context = "\n\n".join(
        (
            f"[SOURCE {index + 1}: {chunk.get('source', 'N/A')} | "
            f"Relevance: {chunk.get('relevance_score', 0):.2f}]\n"
            f"{chunk.get('content', '')}"
        )
        for index, chunk in enumerate(ranked_serialized)
    )

    gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "No explicit gaps identified."
    mode_instruction = MODE_INSTRUCTIONS.get(query_type_hint, MODE_INSTRUCTIONS["GENERAL"])

    prompt = f"""USER QUERY: {query}
SUGGESTED QUERY TYPE: {query_type_hint}
MODE INSTRUCTIONS: {mode_instruction}

DOCUMENT CONTEXT:
{context}

IDENTIFIED GAPS:
{gaps_text}

Build the structured answer and return JSON only.
Focus on the user's explicit request and avoid lateral context unless it is necessary to answer.
If the evidence already includes numbered steps or an explicit cause/solution wording, preserve it.
If the documentation does not cover the core of the request, say so clearly and answer conservatively."""

    try:
        response = agent.run(prompt)
        if isinstance(response.content, ReasoningOutput):
            return response.content

        content = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(_extract_json(content))
        return ReasoningOutput(**data)

    except Exception as exc:
        logger.error("ReasoningAgent error: %s", exc)
        return ReasoningOutput(
            answer=(
                "An internal error occurred while generating the answer. "
                "Please try again or provide additional source material."
            ),
            query_type="GENERAL",
            confidence=0.0,
            citations=[],
            hallucination_risk=1.0,
            hallucination_reason=f"Internal reasoning agent error: {str(exc)[:200]}",
        )
