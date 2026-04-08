"""
CAG agent output models.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RankedChunk(BaseModel):
    """A retrieved chunk scored for relevance."""

    content: str = Field(description="Chunk text")
    source: str = Field(description="Source file or page")
    domain_module: str = Field(description="Derived domain/module label", default="general")
    relevance_score: float = Field(description="Relevance score from 0 to 1", ge=0.0, le=1.0)
    relevance_reason: str = Field(description="Short explanation for the score")


class RetrievalOutput(BaseModel):
    """Structured output returned by the retrieval agent."""

    chunks_ranked: list[RankedChunk] = Field(description="Chunks ordered by relevance")
    gaps: list[str] = Field(
        description="Information gaps not covered by the current evidence",
        default=[],
    )
    relevance_score: float = Field(
        description="Weighted average relevance score for the retrieved evidence",
        ge=0.0,
        le=1.0,
    )
    summary: str = Field(description="Short summary of the retrieved evidence")


class Citation(BaseModel):
    """A citation attached to the generated answer."""

    text: str = Field(description="Quoted or extracted supporting text")
    source: str = Field(description="Source filename")
    domain_module: str = Field(description="Derived domain/module label")


class ReasoningOutput(BaseModel):
    """Structured output returned by the reasoning agent."""

    answer: str = Field(description="Grounded answer generated from the evidence")
    query_type: str = Field(
        description="Query type: DIAGNOSTIC | PROCEDURAL | CONFIGURATION | GENERAL",
        pattern="^(DIAGNOSTIC|PROCEDURAL|CONFIGURATION|GENERAL)$",
    )
    confidence: float = Field(description="Confidence score from 0 to 1", ge=0.0, le=1.0)
    citations: list[Citation] = Field(description="Supporting citations used in the answer")
    hallucination_risk: float = Field(
        description="Estimated hallucination risk from 0 to 1",
        ge=0.0,
        le=1.0,
    )
    hallucination_reason: str = Field(
        description="Explanation for the hallucination risk estimate",
        default="",
    )
