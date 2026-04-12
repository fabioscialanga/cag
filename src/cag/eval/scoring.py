from __future__ import annotations

from statistics import mean

from cag.eval.models import AggregateMetrics, BenchmarkItem, JudgeVerdict, ScoredResult, SystemOutput

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "may",
    "no", "not", "of", "on", "or", "our", "should", "so", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "up", "us",
    "was", "we", "what", "when", "which", "will", "with", "would",
}


def _normalize(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).strip()


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if len(token) > 2 and token not in STOPWORDS}


def _basename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1].lower()


def point_coverage(answer: str, gold_points: list[str]) -> float:
    if not gold_points:
        return 0.0

    answer_norm = _normalize(answer)
    answer_tokens = _tokens(answer)
    covered = 0

    for point in gold_points:
        point_norm = _normalize(point)
        point_tokens = _tokens(point)
        if not point_tokens:
            continue
        overlap = len(point_tokens & answer_tokens) / len(point_tokens)
        if point_norm in answer_norm or overlap >= 0.6:
            covered += 1

    return covered / len(gold_points)


def source_grounding_score(citations: list, gold_sources: list[str]) -> float:
    if not gold_sources:
        return 1.0 if not citations else 0.0

    cited_sources = {_basename(citation.source) for citation in citations if citation.source}
    expected_sources = {_basename(source) for source in gold_sources}

    if not cited_sources:
        return 0.0

    return len(cited_sources & expected_sources) / len(expected_sources)


def context_precision_score(selected_context_sources: list[str], gold_sources: list[str], answerable: bool) -> float | None:
    if not answerable:
        return None

    if not selected_context_sources:
        return 0.0

    expected_sources = {_basename(source) for source in gold_sources}
    if not expected_sources:
        return 0.0

    matched = sum(
        1
        for source in selected_context_sources
        if _basename(source) in expected_sources
    )
    return matched / len(selected_context_sources)


def fallback_judge(item: BenchmarkItem, result: SystemOutput, coverage: float, grounding: float) -> JudgeVerdict:
    if not item.answerable:
        if result.should_escalate:
            return JudgeVerdict(
                correctness=1.0,
                completeness=1.0,
                grounding=1.0,
                unsupported_claims=0.0,
                summary="Correct escalation on an intentionally unsupported query.",
            )
        return JudgeVerdict(
            correctness=0.0,
            completeness=0.0,
            grounding=grounding,
            unsupported_claims=1.0,
            summary="System answered an unanswerable query without escalation.",
        )

    unsupported = 1.0 - min(1.0, (coverage * 0.6) + (grounding * 0.4))
    return JudgeVerdict(
        correctness=coverage,
        completeness=coverage,
        grounding=grounding,
        unsupported_claims=unsupported,
        summary="Deterministic fallback rubric.",
    )


def score_result(item: BenchmarkItem, result: SystemOutput, judge=None) -> ScoredResult:
    coverage = point_coverage(result.answer, item.gold_answer_points)
    grounding = source_grounding_score(result.citations, item.gold_sources)
    context_precision = context_precision_score(result.selected_context_sources, item.gold_sources, item.answerable)
    verdict = judge.evaluate(item, result) if judge else fallback_judge(item, result, coverage, grounding)
    base_payload = result.model_dump(exclude={"context_precision_score"})

    escalation_correct = result.should_escalate if not item.answerable else not result.should_escalate
    query_type_match = result.query_type.upper() == item.query_type

    if item.answerable:
        task_success = (
            not result.should_escalate
            and coverage >= 0.75
            and grounding >= 0.5
            and verdict.unsupported_claims <= 0.5
        )
        grounded_score = (
            (coverage * 0.35)
            + (grounding * 0.20)
            + (verdict.correctness * 0.15)
            + (verdict.completeness * 0.15)
            + ((1.0 - verdict.unsupported_claims) * 0.15)
        )
        if result.should_escalate:
            grounded_score *= 0.25
    else:
        task_success = result.should_escalate
        grounded_score = 1.0 if result.should_escalate else 0.0

    hallucination_flag = (
        (not item.answerable and not result.should_escalate)
        or verdict.unsupported_claims >= 0.6
        or (item.answerable and not result.should_escalate and coverage < 0.3 and grounding < 0.3)
    )

    return ScoredResult(
        **base_payload,
        expected_query_type=item.query_type,
        answerable=item.answerable,
        gold_answer_points=item.gold_answer_points,
        gold_sources=item.gold_sources,
        notes=item.notes,
        point_coverage=coverage,
        source_grounding=grounding,
        context_precision_score=context_precision,
        unsupported_claim_score=verdict.unsupported_claims,
        grounded_answer_score=min(1.0, grounded_score),
        task_success=task_success,
        escalation_correct=escalation_correct,
        hallucination_flag=hallucination_flag,
        query_type_match=query_type_match,
        judge_used=judge is not None,
        judge_summary=verdict.summary,
    )


def aggregate_results(results: list[ScoredResult]) -> AggregateMetrics:
    if not results:
        return AggregateMetrics()

    answerable = [result for result in results if result.answerable]
    unanswerable = [result for result in results if not result.answerable]
    escalations = [result for result in results if result.should_escalate]
    false_escalations = [result for result in answerable if result.should_escalate]
    correct_escalations = [result for result in escalations if not result.answerable]
    context_precision_values = [
        result.context_precision_score
        for result in answerable
        if result.context_precision_score is not None
    ]

    return AggregateMetrics(
        grounded_answer_score=mean(result.grounded_answer_score for result in results),
        point_coverage=mean(result.point_coverage for result in answerable) if answerable else 0.0,
        source_grounding=mean(result.source_grounding for result in answerable) if answerable else 0.0,
        context_precision_score=mean(context_precision_values) if context_precision_values else None,
        hallucination_rate=mean(1.0 if result.hallucination_flag else 0.0 for result in results),
        escalation_precision=(len(correct_escalations) / len(escalations) if escalations else None),
        false_escalation_rate=(len(false_escalations) / len(answerable) if answerable else 0.0),
        task_success_rate=mean(1.0 if result.task_success else 0.0 for result in results),
        query_type_match_rate=mean(1.0 if result.query_type_match else 0.0 for result in results),
        avg_latency_ms=mean(result.latency_ms for result in results),
        avg_cost_estimate=mean(result.cost_estimate for result in results),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        total_count=len(results),
    )


def aggregate_by_query_type(results: list[ScoredResult]) -> dict[str, AggregateMetrics]:
    grouped: dict[str, list[ScoredResult]] = {}
    for result in results:
        grouped.setdefault(result.expected_query_type, []).append(result)
    return {query_type: aggregate_results(items) for query_type, items in grouped.items()}
