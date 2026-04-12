"""
CAG command-line interface.

Provides the ``cag`` console command with subcommands:

    cag ingest   -- run the ingestion pipeline
    cag query    -- run a single query against the CAG graph
    cag eval     -- run a benchmark evaluation
    cag eval-audit -- inspect benchmark dataset readiness
    cag compare  -- compare benchmark run artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging() -> None:
    from cag.config import settings

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    _setup_logging()
    from cag.ingestion.embedder import main as ingest_main

    ingest_argv: list[str] = []
    if args.data_dir:
        ingest_argv.extend(["--data-dir", args.data_dir])
    if args.reset:
        ingest_argv.append("--reset")
    ingest_main(ingest_argv or None)


def _cmd_query(args: argparse.Namespace) -> None:
    _setup_logging()
    from cag.graph.graph import run_query

    result = run_query(query=args.query, conversation_history=[])
    output = {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "query_type": result.get("query_type", "GENERAL"),
        "citations": result.get("citations", []),
        "should_escalate": result.get("should_escalate", False),
        "node_trace": result.get("node_trace", []),
    }
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(output["answer"])
        if output["citations"]:
            sources = ", ".join(c.get("source", "") for c in output["citations"])
            print(f"\nSources: {sources}")
        if output["should_escalate"]:
            print("\n[ESCALATION RECOMMENDED]")


def _cmd_eval(args: argparse.Namespace) -> None:
    _setup_logging()
    from cag.eval.run import main as eval_main

    eval_argv: list[str] = []
    if args.system:
        eval_argv.extend(["--system", args.system])
    if args.dataset:
        eval_argv.extend(["--dataset", args.dataset])
    if args.data_dir:
        eval_argv.extend(["--data-dir", args.data_dir])
    if args.judge_mode:
        eval_argv.extend(["--judge-mode", args.judge_mode])
    if args.top_k is not None:
        eval_argv.extend(["--top-k", str(args.top_k)])
    if args.limit is not None:
        eval_argv.extend(["--limit", str(args.limit)])
    if args.runs is not None:
        eval_argv.extend(["--runs", str(args.runs)])
    if args.output_dir:
        eval_argv.extend(["--output-dir", args.output_dir])
    eval_main(eval_argv or None)


def _cmd_compare(args: argparse.Namespace) -> None:
    _setup_logging()
    from cag.eval.compare import main as compare_main

    compare_argv: list[str] = []
    compare_argv.extend(["--runs"] + args.runs)
    if args.output_dir:
        compare_argv.extend(["--output-dir", args.output_dir])
    compare_main(compare_argv)


def _cmd_eval_audit(args: argparse.Namespace) -> None:
    _setup_logging()
    from cag.eval.audit import main as audit_main

    audit_argv: list[str] = []
    if args.dataset:
        audit_argv.extend(["--dataset", args.dataset])
    if args.data_dir:
        audit_argv.extend(["--data-dir", args.data_dir])
    if args.min_total is not None:
        audit_argv.extend(["--min-total", str(args.min_total)])
    if args.min_per_query_type is not None:
        audit_argv.extend(["--min-per-query-type", str(args.min_per_query_type)])
    if args.format:
        audit_argv.extend(["--format", args.format])
    if args.output:
        audit_argv.extend(["--output", args.output])
    audit_main(audit_argv or None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cag",
        description="CAG - Cognitive Augmented Generation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- cag ingest ---
    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline")
    ingest_parser.add_argument("--data-dir", default="./data/raw", help="Directory containing source documents")
    ingest_parser.add_argument("--reset", action="store_true", help="Reset the vector store before ingestion")
    ingest_parser.set_defaults(func=_cmd_ingest)

    # --- cag query ---
    query_parser = subparsers.add_parser("query", help="Run a query through the CAG graph")
    query_parser.add_argument("query", help="The question to ask")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")
    query_parser.set_defaults(func=_cmd_query)

    # --- cag eval ---
    eval_parser = subparsers.add_parser("eval", help="Run a benchmark evaluation")
    eval_parser.add_argument(
        "--system",
        choices=["cag", "cag_no_selection", "rag_baseline", "direct_baseline", "lightrag_baseline"],
        required=True,
    )
    eval_parser.add_argument("--dataset", default=None, help="Path to benchmark JSONL dataset")
    eval_parser.add_argument("--data-dir", default="./data/benchmark_corpus")
    eval_parser.add_argument("--judge-mode", choices=["auto", "off", "required"], default="auto")
    eval_parser.add_argument("--top-k", type=int, default=10)
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--runs", type=int, default=1)
    eval_parser.add_argument("--output-dir", default="./artifacts/eval_runs")
    eval_parser.set_defaults(func=_cmd_eval)

    # --- cag eval-audit ---
    audit_parser = subparsers.add_parser("eval-audit", help="Inspect benchmark dataset readiness")
    audit_parser.add_argument("--dataset", default=None, help="Path to benchmark JSONL dataset")
    audit_parser.add_argument("--data-dir", default="./data/benchmark_corpus")
    audit_parser.add_argument("--min-total", type=int, default=100)
    audit_parser.add_argument("--min-per-query-type", type=int, default=20)
    audit_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    audit_parser.add_argument("--output", default=None, help="Optional output file path")
    audit_parser.set_defaults(func=_cmd_eval_audit)

    # --- cag compare ---
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark run artifacts")
    compare_parser.add_argument("--runs", nargs="+", required=True, help="Run directories to compare")
    compare_parser.add_argument("--output-dir", default="./artifacts/eval_comparisons")
    compare_parser.set_defaults(func=_cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
