from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


QueryType = Literal["GENERAL", "PROCEDURAL", "DIAGNOSTIC", "CONFIGURATION"]
SystemName = Literal["cag", "rag_baseline", "direct_baseline", "lightrag_baseline"]
JudgeMode = Literal["auto", "off", "required"]


class BenchmarkItem(BaseModel):
    id: str
    question: str
    query_type: QueryType
    gold_answer_points: list[str] = Field(default_factory=list)
    gold_sources: list[str] = Field(default_factory=list)
    answerable: bool
    notes: str = ""


class CitationRecord(BaseModel):
    text: str = ""
    source: str = ""
    domain_module: str = "general"


class SystemOutput(BaseModel):
    question_id: str
    question: str
    system: SystemName
    answer: str
    citations: list[CitationRecord] = Field(default_factory=list)
    query_type: str = "GENERAL"
    confidence: float = 0.0
    hallucination_risk: float = 1.0
    should_escalate: bool = False
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    node_trace: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0, default=0.0)
    completeness: float = Field(ge=0.0, le=1.0, default=0.0)
    grounding: float = Field(ge=0.0, le=1.0, default=0.0)
    unsupported_claims: float = Field(ge=0.0, le=1.0, default=1.0)
    summary: str = ""


class ScoredResult(SystemOutput):
    expected_query_type: QueryType
    answerable: bool
    gold_answer_points: list[str] = Field(default_factory=list)
    gold_sources: list[str] = Field(default_factory=list)
    notes: str = ""
    point_coverage: float = 0.0
    source_grounding: float = 0.0
    unsupported_claim_score: float = 1.0
    grounded_answer_score: float = 0.0
    task_success: bool = False
    escalation_correct: bool = False
    hallucination_flag: bool = False
    query_type_match: bool = False
    judge_used: bool = False
    judge_summary: str = ""


class AggregateMetrics(BaseModel):
    grounded_answer_score: float = 0.0
    point_coverage: float = 0.0
    source_grounding: float = 0.0
    hallucination_rate: float = 0.0
    escalation_precision: float | None = None
    false_escalation_rate: float = 0.0
    task_success_rate: float = 0.0
    query_type_match_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_estimate: float = 0.0
    answerable_count: int = 0
    unanswerable_count: int = 0
    total_count: int = 0


class RunManifest(BaseModel):
    run_id: str
    system: SystemName
    dataset_path: str
    data_dir: str
    judge_mode: JudgeMode
    judge_active: bool
    benchmark_sources: list[str]
    model_id: str
    embedding_model: str
    top_k: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: AggregateMetrics
    by_query_type: dict[str, AggregateMetrics] = Field(default_factory=dict)
    run_index: int | None = None


class MetricStats(BaseModel):
    """Mean and standard deviation for a single metric across multiple runs."""

    mean: float
    std: float
    min: float
    max: float
    n: int


class MultiRunManifest(BaseModel):
    """Aggregate manifest for a multi-run benchmark."""

    run_id: str
    system: SystemName
    dataset_path: str
    data_dir: str
    judge_mode: JudgeMode
    judge_active: bool
    benchmark_sources: list[str]
    model_id: str
    embedding_model: str
    top_k: int
    total_runs: int
    aggregate_metrics: dict[str, MetricStats]
    by_query_type: dict[str, dict[str, MetricStats]] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
