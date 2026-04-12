from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from cag.config import settings
from cag.eval.corpus import BenchmarkVectorIndex, collect_benchmark_sources, load_benchmark_dataset
from cag.eval.judge import build_judge
from cag.eval.lightrag_adapter import LightRAGRuntime
from cag.eval.models import (
    AggregateMetrics,
    MetricStats,
    MultiRunManifest,
    RunManifest,
    ScoredResult,
)
from cag.eval.scoring import aggregate_by_query_type, aggregate_results, score_result
from cag.eval.systems import run_system

logger = logging.getLogger(__name__)


def _write_results(path: Path, results: list[ScoredResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CAG scientific benchmark.")
    parser.add_argument(
        "--system",
        choices=["cag", "cag_no_selection", "rag_baseline", "direct_baseline", "lightrag_baseline"],
        required=True,
    )
    parser.add_argument("--dataset", default=None, help="Path to the benchmark JSONL dataset.")
    parser.add_argument("--data-dir", default="./data/benchmark_corpus", help="Directory containing source documents. Default: benchmark corpus.")
    parser.add_argument("--judge-mode", choices=["auto", "off", "required"], default="auto")
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N benchmark items.")
    parser.add_argument("--runs", type=int, default=1, help="Run the benchmark N times for stochastic reproducibility.")
    parser.add_argument("--output-dir", default="./artifacts/eval_runs", help="Base directory for run artifacts.")
    return parser


def _run_single_pass(
    system: str,
    benchmark_items: list,
    dataset_path: str,
    data_dir: str,
    top_k: int,
    judge,
    judge_mode: str,
    run_index: int | None,
    run_dir: Path,
) -> tuple[list[ScoredResult], RunManifest]:
    """Execute a single benchmark pass and write artifacts."""

    results: list[ScoredResult] = []
    if system == "lightrag_baseline":
        with LightRAGRuntime(data_dir, benchmark_items, top_k).build() as runtime:
            for item in benchmark_items:
                logger.info("Benchmarking %s on %s", item.id, system)
                raw_result = run_system(
                    system=system,
                    question_id=item.id,
                    question=item.question,
                    search_fn=lambda *_args, **_kwargs: [],
                    top_k=top_k,
                    runtime=runtime,
                )
                results.append(score_result(item, raw_result, judge=judge))
    else:
        with BenchmarkVectorIndex(data_dir, benchmark_items).build() as index:
            for item in benchmark_items:
                logger.info("Benchmarking %s on %s", item.id, system)
                raw_result = run_system(
                    system=system,
                    question_id=item.id,
                    question=item.question,
                    search_fn=index.similarity_search,
                    top_k=top_k,
                )
                results.append(score_result(item, raw_result, judge=judge))

    manifest = RunManifest(
        run_id=run_dir.name,
        system=system,
        dataset_path=dataset_path,
        data_dir=str(data_dir),
        judge_mode=judge_mode,
        judge_active=judge is not None,
        benchmark_sources=collect_benchmark_sources(benchmark_items),
        model_id=settings.active_model_id,
        embedding_model=settings.embedding_model,
        top_k=top_k,
        metrics=aggregate_results(results),
        by_query_type=aggregate_by_query_type(results),
        run_index=run_index,
    )

    _write_results(run_dir / "results.jsonl", results)
    (run_dir / "run.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return results, manifest


def _compute_multi_run_stats(all_runs: list[list[ScoredResult]]) -> dict[str, MetricStats]:
    """Compute mean/std/min/max across multiple runs for each metric."""

    per_run_metrics = [aggregate_results(run) for run in all_runs]
    metric_names = [
        "grounded_answer_score", "point_coverage", "source_grounding", "context_precision_score",
        "hallucination_rate", "task_success_rate", "false_escalation_rate",
        "avg_latency_ms", "avg_cost_estimate",
    ]

    stats: dict[str, MetricStats] = {}
    for name in metric_names:
        values = [value for value in (getattr(m, name) for m in per_run_metrics) if value is not None]
        if not values:
            continue
        stats[name] = MetricStats(
            mean=mean(values),
            std=stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            max=max(values),
            n=len(values),
        )

    return stats


def _compute_multi_run_by_type(
    all_runs: list[list[ScoredResult]],
) -> dict[str, dict[str, MetricStats]]:
    """Compute per-query-type multi-run stats."""

    query_types: set[str] = set()
    for run in all_runs:
        for result in run:
            query_types.add(result.expected_query_type)

    by_type: dict[str, dict[str, MetricStats]] = {}
    for qt in sorted(query_types):
        per_run_for_type = []
        for run in all_runs:
            type_results = [r for r in run if r.expected_query_type == qt]
            if type_results:
                per_run_for_type.append(aggregate_results(type_results))

        if len(per_run_for_type) < 2:
            continue

        metric_names = [
            "grounded_answer_score", "point_coverage", "source_grounding", "context_precision_score",
            "hallucination_rate", "task_success_rate",
        ]
        type_stats: dict[str, MetricStats] = {}
        for name in metric_names:
            values = [value for value in (getattr(m, name) for m in per_run_for_type) if value is not None]
            if not values:
                continue
            type_stats[name] = MetricStats(
                mean=mean(values),
                std=stdev(values) if len(values) > 1 else 0.0,
                min=min(values),
                max=max(values),
                n=len(values),
            )
        by_type[qt] = type_stats

    return by_type


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    benchmark_items = load_benchmark_dataset(args.dataset)
    dataset_path = str(args.dataset or "package://cag.eval/benchmark_dataset.jsonl")
    if args.limit:
        benchmark_items = benchmark_items[: args.limit]

    judge = build_judge(args.judge_mode)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_run_id = f"{timestamp}_{args.system}"
    base_dir = Path(args.output_dir) / base_run_id

    all_runs: list[list[ScoredResult]] = []
    manifests: list[RunManifest] = []

    for run_index in range(1, args.runs + 1):
        if args.runs > 1:
            run_dir = base_dir / f"run_{run_index:03d}"
        else:
            run_dir = base_dir

        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run %d/%d starting", run_index, args.runs)

        results, manifest = _run_single_pass(
            system=args.system,
            benchmark_items=benchmark_items,
            dataset_path=dataset_path,
            data_dir=args.data_dir,
            top_k=args.top_k,
            judge=judge,
            judge_mode=args.judge_mode,
            run_index=run_index if args.runs > 1 else None,
            run_dir=run_dir,
        )
        all_runs.append(results)
        manifests.append(manifest)
        logger.info("Run %d/%d complete", run_index, args.runs)

    if args.runs > 1:
        aggregate_metrics = _compute_multi_run_stats(all_runs)
        by_query_type = _compute_multi_run_by_type(all_runs)

        multi_manifest = MultiRunManifest(
            run_id=base_run_id,
            system=args.system,
            dataset_path=dataset_path,
            data_dir=str(args.data_dir),
            judge_mode=args.judge_mode,
            judge_active=judge is not None,
            benchmark_sources=collect_benchmark_sources(benchmark_items),
            model_id=settings.active_model_id,
            embedding_model=settings.embedding_model,
            top_k=args.top_k,
            total_runs=args.runs,
            aggregate_metrics=aggregate_metrics,
            by_query_type=by_query_type,
        )
        (base_dir / "multi_run.json").write_text(
            json.dumps(multi_manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Multi-run aggregate saved to %s/multi_run.json", base_dir)

    logger.info("Benchmark complete. Artifacts in %s", base_dir)


if __name__ == "__main__":
    main()
