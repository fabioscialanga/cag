from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from cag.eval.models import AggregateMetrics, RunManifest, ScoredResult


def _load_results(path: Path) -> list[ScoredResult]:
    results: list[ScoredResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            results.append(ScoredResult(**json.loads(raw)))
    return results


def _load_run(run_path: str | Path) -> tuple[RunManifest, list[ScoredResult], Path]:
    base = Path(run_path)
    if base.is_file():
        base = base.parent
    manifest = RunManifest(**json.loads((base / "run.json").read_text(encoding="utf-8")))
    results = _load_results(base / "results.jsonl")
    return manifest, results, base


def _load_run_collection(run_path: str | Path) -> list[tuple[RunManifest, list[ScoredResult], Path]]:
    base = Path(run_path)
    if base.is_file():
        base = base.parent

    if (base / "run.json").exists():
        return [_load_run(base)]

    if (base / "multi_run.json").exists():
        run_dirs = sorted(
            path
            for path in base.iterdir()
            if path.is_dir() and (path / "run.json").exists() and (path / "results.jsonl").exists()
        )
        if not run_dirs:
            raise FileNotFoundError(f"No run subdirectories found under multi-run directory: {base}")
        return [_load_run(run_dir) for run_dir in run_dirs]

    raise FileNotFoundError(f"Could not find run.json or multi_run.json under: {base}")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _mean_metric_by_question(results: list[ScoredResult], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        grouped[result.question_id].append(float(getattr(result, metric)))
    return {question_id: mean(values) for question_id, values in grouped.items()}


def _collapse_results_by_question(results: list[ScoredResult]) -> list[ScoredResult]:
    grouped: dict[str, list[ScoredResult]] = defaultdict(list)
    for result in results:
        grouped[result.question_id].append(result)

    collapsed: list[ScoredResult] = []
    for question_id, items in grouped.items():
        exemplar = items[0]
        collapsed.append(
            exemplar.model_copy(
                update={
                    "grounded_answer_score": mean(item.grounded_answer_score for item in items),
                    "point_coverage": mean(item.point_coverage for item in items),
                    "source_grounding": mean(item.source_grounding for item in items),
                    "context_precision_score": (
                        mean(valid)
                        if (valid := [item.context_precision_score for item in items if item.context_precision_score is not None])
                        else None
                    ),
                    "unsupported_claim_score": mean(item.unsupported_claim_score for item in items),
                    "confidence": mean(item.confidence for item in items),
                    "hallucination_risk": mean(item.hallucination_risk for item in items),
                    "retrieved_chunk_count": round(mean(item.retrieved_chunk_count for item in items)),
                    "selected_chunk_count": round(mean(item.selected_chunk_count for item in items)),
                    "latency_ms": mean(item.latency_ms for item in items),
                    "cost_estimate": mean(item.cost_estimate for item in items),
                    "task_success": mean(1.0 if item.task_success else 0.0 for item in items) >= 0.5,
                    "should_escalate": any(item.should_escalate for item in items),
                    "hallucination_flag": any(item.hallucination_flag for item in items),
                }
            )
        )

    return collapsed


def _aggregate_metrics(manifests: list[RunManifest]) -> AggregateMetrics:
    if len(manifests) == 1:
        return manifests[0].metrics

    metrics_list = [manifest.metrics for manifest in manifests]

    def maybe_mean(values: list[float | None]) -> float | None:
        valid = [value for value in values if value is not None]
        if not valid:
            return None
        return mean(valid)

    return AggregateMetrics(
        grounded_answer_score=mean(metric.grounded_answer_score for metric in metrics_list),
        point_coverage=mean(metric.point_coverage for metric in metrics_list),
        source_grounding=mean(metric.source_grounding for metric in metrics_list),
        context_precision_score=maybe_mean([metric.context_precision_score for metric in metrics_list]),
        hallucination_rate=mean(metric.hallucination_rate for metric in metrics_list),
        escalation_precision=maybe_mean([metric.escalation_precision for metric in metrics_list]),
        false_escalation_rate=mean(metric.false_escalation_rate for metric in metrics_list),
        task_success_rate=mean(metric.task_success_rate for metric in metrics_list),
        query_type_match_rate=mean(metric.query_type_match_rate for metric in metrics_list),
        avg_latency_ms=mean(metric.avg_latency_ms for metric in metrics_list),
        avg_cost_estimate=mean(metric.avg_cost_estimate for metric in metrics_list),
        answerable_count=round(mean(metric.answerable_count for metric in metrics_list)),
        unanswerable_count=round(mean(metric.unanswerable_count for metric in metrics_list)),
        total_count=round(mean(metric.total_count for metric in metrics_list)),
    )


def _aggregate_by_query_type(manifests: list[RunManifest]) -> dict[str, AggregateMetrics]:
    query_types = sorted({query_type for manifest in manifests for query_type in manifest.by_query_type})
    by_query_type: dict[str, AggregateMetrics] = {}
    for query_type in query_types:
        available = [manifest.by_query_type[query_type] for manifest in manifests if query_type in manifest.by_query_type]
        if not available:
            continue
        pseudo_manifests = [
            RunManifest(
                run_id=f"aggregate-{index}",
                system=manifests[0].system,
                dataset_path=manifests[0].dataset_path,
                data_dir=manifests[0].data_dir,
                judge_mode=manifests[0].judge_mode,
                judge_active=manifests[0].judge_active,
                benchmark_sources=manifests[0].benchmark_sources,
                model_id=manifests[0].model_id,
                embedding_model=manifests[0].embedding_model,
                top_k=manifests[0].top_k,
                metrics=metrics,
            )
            for index, metrics in enumerate(available)
        ]
        by_query_type[query_type] = _aggregate_metrics(pseudo_manifests)
    return by_query_type


def _paired_statistical_test(
    cag_results: list[ScoredResult],
    rag_results: list[ScoredResult],
    metric: str = "grounded_answer_score",
) -> dict:
    """Run paired statistical test comparing CAG vs RAG on a per-question metric."""

    from scipy import stats as scipy_stats

    cag_by_id = _mean_metric_by_question(cag_results, metric)
    rag_by_id = _mean_metric_by_question(rag_results, metric)
    common_ids = sorted(set(cag_by_id) & set(rag_by_id))

    if not common_ids:
        return {
            "metric": metric,
            "n_pairs": 0,
            "cag_mean": 0.0,
            "rag_mean": 0.0,
            "mean_difference": 0.0,
            "paired_t": {"t_statistic": 0.0, "p_value": 1.0},
            "wilcoxon": {"w_statistic": 0.0, "p_value": 1.0},
            "cohens_d": 0.0,
            "significant_at_005": False,
        }

    cag_scores = [cag_by_id[qid] for qid in common_ids]
    rag_scores = [rag_by_id[qid] for qid in common_ids]
    differences = [c - r for c, r in zip(cag_scores, rag_scores)]
    mean_diff = mean(differences)
    all_equal = all(math.isclose(diff, 0.0, abs_tol=1e-12) for diff in differences)

    if all_equal:
        t_stat, p_value_t = 0.0, 1.0
        w_stat, p_value_w = 0.0, 1.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_stat, p_value_t = scipy_stats.ttest_rel(cag_scores, rag_scores, nan_policy="omit")

        try:
            w_stat, p_value_w = scipy_stats.wilcoxon(cag_scores, rag_scores, zero_method="zsplit")
        except ValueError:
            w_stat, p_value_w = 0.0, 1.0

    std_diff = (
        (sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)) ** 0.5
        if len(differences) > 1
        else 1.0
    )
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        "metric": metric,
        "n_pairs": len(common_ids),
        "cag_mean": float(mean(cag_scores)),
        "rag_mean": float(mean(rag_scores)),
        "mean_difference": float(mean_diff),
        "paired_t": {"t_statistic": float(t_stat), "p_value": float(p_value_t)},
        "wilcoxon": {"w_statistic": float(w_stat), "p_value": float(p_value_w)},
        "cohens_d": float(cohens_d),
        "significant_at_005": bool(p_value_t < 0.05 or p_value_w < 0.05),
    }


def _verdict_text(
    metrics_by_system: dict[str, AggregateMetrics],
    statistical_test: dict | None = None,
) -> str:
    cag = metrics_by_system.get("cag")
    rag = metrics_by_system.get("rag_baseline")
    if not cag or not rag:
        return "Comparison verdict unavailable: both `cag` and `rag_baseline` runs are required."

    primary_win = cag.grounded_answer_score > rag.grounded_answer_score
    safety_win = (
        cag.hallucination_rate < rag.hallucination_rate
        or (cag.escalation_precision or 0.0) >= (rag.escalation_precision or 0.0)
    )

    if statistical_test and statistical_test.get("significant_at_005"):
        effect = statistical_test["cohens_d"]
        effect_label = "large" if abs(effect) >= 0.8 else "medium" if abs(effect) >= 0.5 else "small"
        if primary_win and safety_win:
            return (
                f"Claim SUPPORTED with statistical significance (p < 0.05, Cohen's d = {effect:.2f}, "
                f"{effect_label} effect): CAG outperforms standard RAG on grounded answer score "
                f"and improves at least one safety metric."
            )
        if primary_win:
            return (
                f"Claim PARTIALLY SUPPORTED with statistical significance (p < 0.05, Cohen's d = {effect:.2f}): "
                f"CAG wins on grounded answer score, but safety metrics are mixed."
            )
        return "Claim NOT SUPPORTED despite statistical significance: CAG does not outperform RAG on the primary metric."

    if primary_win and safety_win:
        return (
            "Claim supported (descriptive): CAG beats RAG on grounded answer score and safety, "
            "but statistical significance was not established on the available paired question set."
        )
    if primary_win:
        return "Primary claim partially supported (descriptive): CAG wins on score but safety is mixed."
    return "Claim not supported: CAG does not beat the RAG baseline on the primary metric."


def _representative_failures(results: list[ScoredResult], limit: int = 5) -> list[ScoredResult]:
    return sorted(results, key=lambda item: item.grounded_answer_score)[:limit]


def _delta_vs_baseline(metrics_by_system: dict[str, AggregateMetrics], baseline_system: str = "rag_baseline") -> dict[str, dict[str, float | None]]:
    baseline = metrics_by_system.get(baseline_system)
    if baseline is None:
        return {}

    metric_names = [
        "grounded_answer_score",
        "context_precision_score",
        "hallucination_rate",
        "task_success_rate",
        "avg_latency_ms",
        "avg_cost_estimate",
    ]
    deltas: dict[str, dict[str, float | None]] = {}
    for system, metrics in metrics_by_system.items():
        if system == baseline_system:
            continue
        system_delta: dict[str, float | None] = {}
        for metric_name in metric_names:
            system_value = getattr(metrics, metric_name)
            baseline_value = getattr(baseline, metric_name)
            if system_value is None or baseline_value is None:
                system_delta[metric_name] = None
            else:
                system_delta[metric_name] = float(system_value - baseline_value)
        deltas[system] = system_delta
    return deltas


def _build_markdown(
    manifests: list[RunManifest],
    results_by_system: dict[str, list[ScoredResult]],
    statistical_test: dict | None = None,
    run_counts_by_system: dict[str, int] | None = None,
) -> str:
    metrics_by_system = {manifest.system: manifest.metrics for manifest in manifests}
    deltas_vs_rag = _delta_vs_baseline(metrics_by_system)
    lines = [
        "# CAG Benchmark Comparison",
        "",
        _verdict_text(metrics_by_system, statistical_test),
        "",
        "## Aggregate Metrics",
        "",
        "| System | Runs | Grounded Answer | Context Precision | Hallucination Rate | Task Success | Escalation Precision | False Escalation | Avg Latency (ms) | Avg Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for manifest in manifests:
        metrics = manifest.metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    manifest.system,
                    str((run_counts_by_system or {}).get(manifest.system, 1)),
                    _format_metric(metrics.grounded_answer_score),
                    _format_metric(metrics.context_precision_score),
                    _format_metric(metrics.hallucination_rate),
                    _format_metric(metrics.task_success_rate),
                    _format_metric(metrics.escalation_precision),
                    _format_metric(metrics.false_escalation_rate),
                    _format_metric(metrics.avg_latency_ms),
                    _format_metric(metrics.avg_cost_estimate),
                ]
            )
            + " |"
        )

    if deltas_vs_rag:
        lines.extend(["", "## Delta vs RAG Baseline", ""])
        lines.append("| System | Grounded Answer | Context Precision | Hallucination Rate | Task Success | Avg Latency (ms) | Avg Cost |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for system, metrics in sorted(deltas_vs_rag.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        system,
                        _format_metric(metrics["grounded_answer_score"]),
                        _format_metric(metrics["context_precision_score"]),
                        _format_metric(metrics["hallucination_rate"]),
                        _format_metric(metrics["task_success_rate"]),
                        _format_metric(metrics["avg_latency_ms"]),
                        _format_metric(metrics["avg_cost_estimate"]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## By Query Type", ""])
    for manifest in manifests:
        lines.append(f"### {manifest.system}")
        lines.append("")
        lines.append("| Query Type | Grounded Answer | Context Precision | Task Success | Hallucination Rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for query_type, metrics in manifest.by_query_type.items():
            lines.append(
                f"| {query_type} | {_format_metric(metrics.grounded_answer_score)} | "
                f"{_format_metric(metrics.context_precision_score)} | {_format_metric(metrics.task_success_rate)} | "
                f"{_format_metric(metrics.hallucination_rate)} |"
            )
        lines.append("")

    if statistical_test:
        lines.extend([
            "## Statistical Testing",
            "",
            "Scores are paired by question ID. When multiple runs are provided for the same system, the comparison uses the mean score per question across runs.",
            "",
            f"| Test | Statistic | p-value | Significant |",
            f"| --- | ---: | ---: | --- |",
            f"| Paired t-test | {statistical_test['paired_t']['t_statistic']:.4f} | {statistical_test['paired_t']['p_value']:.4f} | {'Yes' if statistical_test['paired_t']['p_value'] < 0.05 else 'No'} |",
            f"| Wilcoxon signed-rank | {statistical_test['wilcoxon']['w_statistic']:.4f} | {statistical_test['wilcoxon']['p_value']:.4f} | {'Yes' if statistical_test['wilcoxon']['p_value'] < 0.05 else 'No'} |",
            "",
            f"- **CAG mean**: {statistical_test['cag_mean']:.4f}",
            f"- **RAG mean**: {statistical_test['rag_mean']:.4f}",
            f"- **Mean difference**: {statistical_test['mean_difference']:.4f}",
            f"- **Cohen's d**: {statistical_test['cohens_d']:.4f} ({'large' if abs(statistical_test['cohens_d']) >= 0.8 else 'medium' if abs(statistical_test['cohens_d']) >= 0.5 else 'small'} effect)",
            f"- **N pairs**: {statistical_test['n_pairs']}",
            "",
        ])

    lines.append("## Representative Failures")
    lines.append("")
    for system, results in results_by_system.items():
        lines.append(f"### {system}")
        lines.append("")
        lines.append("| Question ID | Grounded Score | Escalated | Notes |")
        lines.append("| --- | ---: | --- | --- |")
        for item in _representative_failures(results):
            note = item.notes or item.question
            lines.append(
                f"| {item.question_id} | {_format_metric(item.grounded_answer_score)} | "
                f"{item.should_escalate} | {note} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare benchmark run artifacts.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories created by `python -m cag.eval.run`.")
    parser.add_argument("--output-dir", default="./artifacts/eval_comparisons", help="Directory for comparison artifacts.")
    return parser


def _create_unique_output_dir(base_dir: Path, timestamp: str) -> Path:
    candidate = base_dir / f"{timestamp}_comparison"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 1
    while True:
        candidate = base_dir / f"{timestamp}_comparison_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    manifests_by_system: dict[str, list[RunManifest]] = defaultdict(list)
    results_by_system: dict[str, list[ScoredResult]] = {}
    run_paths: list[str] = []

    for run in args.runs:
        for manifest, results, base in _load_run_collection(run):
            manifests_by_system[manifest.system].append(manifest)
            results_by_system.setdefault(manifest.system, []).extend(results)
            run_paths.append(str(base))

    manifests = []
    run_counts_by_system: dict[str, int] = {}
    for system, system_manifests in sorted(manifests_by_system.items()):
        first_manifest = system_manifests[0]
        manifests.append(
            first_manifest.model_copy(
                update={
                    "metrics": _aggregate_metrics(system_manifests),
                    "by_query_type": _aggregate_by_query_type(system_manifests),
                    "run_index": None,
                }
            )
        )
        run_counts_by_system[system] = len(system_manifests)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _create_unique_output_dir(Path(args.output_dir), timestamp)

    statistical_test = None
    if "cag" in results_by_system and "rag_baseline" in results_by_system:
        statistical_test = _paired_statistical_test(
            results_by_system["cag"],
            results_by_system["rag_baseline"],
        )

    comparison_payload = {
        "runs": run_paths,
        "systems": [manifest.system for manifest in manifests],
        "run_counts_by_system": run_counts_by_system,
        "metrics_by_system": {manifest.system: manifest.metrics.model_dump(mode="json") for manifest in manifests},
        "delta_vs_rag_baseline": _delta_vs_baseline({manifest.system: manifest.metrics for manifest in manifests}),
        "claim_verdict": _verdict_text(
            {manifest.system: manifest.metrics for manifest in manifests},
            statistical_test,
        ),
        "statistical_test": statistical_test,
    }

    (output_dir / "comparison.json").write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        _build_markdown(
            manifests,
            {system: _collapse_results_by_question(results) for system, results in results_by_system.items()},
            statistical_test,
            run_counts_by_system,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
