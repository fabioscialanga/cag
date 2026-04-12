from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cag.eval.corpus import get_default_dataset_path, load_benchmark_dataset
from cag.eval.models import BenchmarkItem
from cag.ingestion.loader import SUPPORTED_EXTENSIONS


class QueryTypeAudit(BaseModel):
    total_count: int
    answerable_count: int
    unanswerable_count: int


class DatasetAudit(BaseModel):
    dataset_path: str
    data_dir: str | None = None
    readiness: Literal["ready", "needs_expansion", "invalid"]
    total_count: int
    answerable_count: int
    unanswerable_count: int
    by_query_type: dict[str, QueryTypeAudit] = Field(default_factory=dict)
    gold_source_usage: dict[str, int] = Field(default_factory=dict)
    corpus_sources: list[str] = Field(default_factory=list)
    uncovered_corpus_sources: list[str] = Field(default_factory=list)
    duplicate_ids: list[str] = Field(default_factory=list)
    duplicate_questions: list[str] = Field(default_factory=list)
    answerable_without_gold_sources: list[str] = Field(default_factory=list)
    answerable_without_gold_answer_points: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


def _collect_corpus_sources(data_dir: str | Path | None) -> list[str]:
    if data_dir is None:
        return []

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    return sorted(
        file_path.name
        for file_path in data_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_dataset_audit(
    items: list[BenchmarkItem],
    dataset_path: str | Path,
    data_dir: str | Path | None = None,
    *,
    min_total: int = 100,
    min_per_query_type: int = 20,
) -> DatasetAudit:
    id_counts = Counter(item.id for item in items)
    question_counts = Counter(item.question.strip().lower() for item in items)
    by_query_type: dict[str, QueryTypeAudit] = {}
    gold_source_usage = Counter(source for item in items for source in item.gold_sources)

    for query_type in sorted({item.query_type for item in items}):
        matching = [item for item in items if item.query_type == query_type]
        answerable_count = sum(1 for item in matching if item.answerable)
        by_query_type[query_type] = QueryTypeAudit(
            total_count=len(matching),
            answerable_count=answerable_count,
            unanswerable_count=len(matching) - answerable_count,
        )

    duplicate_ids = sorted(item_id for item_id, count in id_counts.items() if count > 1)
    duplicate_questions = sorted(question for question, count in question_counts.items() if count > 1)
    answerable_without_gold_sources = sorted(
        item.id for item in items if item.answerable and not item.gold_sources
    )
    answerable_without_gold_answer_points = sorted(
        item.id for item in items if item.answerable and not item.gold_answer_points
    )
    corpus_sources = _collect_corpus_sources(data_dir)
    uncovered_corpus_sources = sorted(set(corpus_sources) - set(gold_source_usage))

    issues: list[str] = []
    recommendations: list[str] = []
    readiness: Literal["ready", "needs_expansion", "invalid"] = "ready"

    if duplicate_ids:
        readiness = "invalid"
        issues.append(f"Duplicate benchmark IDs: {', '.join(duplicate_ids)}")
    if duplicate_questions:
        readiness = "invalid"
        issues.append(f"Duplicate benchmark questions: {', '.join(duplicate_questions[:5])}")
    if answerable_without_gold_sources:
        readiness = "invalid"
        issues.append(
            "Answerable items without gold sources: "
            + ", ".join(answerable_without_gold_sources[:8])
        )
    if answerable_without_gold_answer_points:
        readiness = "invalid"
        issues.append(
            "Answerable items without gold answer points: "
            + ", ".join(answerable_without_gold_answer_points[:8])
        )

    low_query_types = {
        query_type: stats.total_count
        for query_type, stats in by_query_type.items()
        if stats.total_count < min_per_query_type
    }
    if len(items) < min_total:
        if readiness == "ready":
            readiness = "needs_expansion"
        issues.append(
            f"Dataset has {len(items)} items; recommended minimum for stronger claims is {min_total}."
        )
    if low_query_types:
        if readiness == "ready":
            readiness = "needs_expansion"
        issues.append(
            "Query types below target coverage: "
            + ", ".join(f"{query_type}={count}" for query_type, count in sorted(low_query_types.items()))
        )

    if uncovered_corpus_sources:
        recommendations.append(
            "Add gold-labeled questions for uncovered corpus files: "
            + ", ".join(uncovered_corpus_sources[:8])
        )
    if readiness != "ready":
        recommendations.append(
            "Increase benchmark size while keeping query types balanced and preserving answerable/unanswerable coverage."
        )
    if readiness == "ready":
        recommendations.append("Benchmark coverage looks healthy enough for repeated-run and multi-corpus comparison.")

    return DatasetAudit(
        dataset_path=str(dataset_path),
        data_dir=str(data_dir) if data_dir is not None else None,
        readiness=readiness,
        total_count=len(items),
        answerable_count=sum(1 for item in items if item.answerable),
        unanswerable_count=sum(1 for item in items if not item.answerable),
        by_query_type=by_query_type,
        gold_source_usage=dict(sorted(gold_source_usage.items())),
        corpus_sources=corpus_sources,
        uncovered_corpus_sources=uncovered_corpus_sources,
        duplicate_ids=duplicate_ids,
        duplicate_questions=duplicate_questions,
        answerable_without_gold_sources=answerable_without_gold_sources,
        answerable_without_gold_answer_points=answerable_without_gold_answer_points,
        issues=issues,
        recommendations=recommendations,
    )


def _build_markdown(audit: DatasetAudit) -> str:
    lines = [
        "# Benchmark Dataset Audit",
        "",
        f"- Dataset: `{audit.dataset_path}`",
        f"- Data dir: `{audit.data_dir or 'n/a'}`",
        f"- Readiness: `{audit.readiness}`",
        f"- Total items: `{audit.total_count}`",
        f"- Answerable / Unanswerable: `{audit.answerable_count}` / `{audit.unanswerable_count}`",
        "",
        "## Query Type Coverage",
        "",
        "| Query Type | Total | Answerable | Unanswerable |",
        "| --- | ---: | ---: | ---: |",
    ]

    for query_type, stats in audit.by_query_type.items():
        lines.append(
            f"| {query_type} | {stats.total_count} | {stats.answerable_count} | {stats.unanswerable_count} |"
        )

    lines.extend(["", "## Source Coverage", "", "| Source | Gold-Labeled Questions |", "| --- | ---: |"])
    for source in audit.corpus_sources:
        lines.append(f"| {source} | {audit.gold_source_usage.get(source, 0)} |")

    if audit.issues:
        lines.extend(["", "## Issues", ""])
        for issue in audit.issues:
            lines.append(f"- {issue}")

    if audit.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for recommendation in audit.recommendations:
            lines.append(f"- {recommendation}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit benchmark dataset readiness.")
    parser.add_argument("--dataset", default=None, help="Path to the benchmark JSONL dataset.")
    parser.add_argument("--data-dir", default="./data/benchmark_corpus", help="Directory containing source documents.")
    parser.add_argument("--min-total", type=int, default=100, help="Recommended minimum benchmark size.")
    parser.add_argument(
        "--min-per-query-type",
        type=int,
        default=20,
        help="Recommended minimum items per query type.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional output file path.")
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset) if args.dataset else get_default_dataset_path()
    items = load_benchmark_dataset(dataset_path)
    audit = build_dataset_audit(
        items,
        dataset_path,
        args.data_dir,
        min_total=args.min_total,
        min_per_query_type=args.min_per_query_type,
    )
    rendered = (
        json.dumps(audit.model_dump(mode="json"), ensure_ascii=False, indent=2)
        if args.format == "json"
        else _build_markdown(audit)
    )

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
