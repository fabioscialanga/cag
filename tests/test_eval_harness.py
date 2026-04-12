from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from cag.eval.audit import build_dataset_audit
from cag.eval.corpus import cleanup_temp_path, collect_benchmark_sources, load_benchmark_dataset
from cag.eval.lightrag_adapter import infer_should_escalate, parse_lightrag_response
from cag.eval.models import AggregateMetrics, BenchmarkItem, CitationRecord, RunManifest, ScoredResult, SystemOutput
from cag.eval.scoring import aggregate_results, score_result
from cag.eval.systems import BaselineGeneration, run_cag_no_selection, run_cag_system, run_rag_baseline, run_system


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_benchmark_dataset_and_collect_sources():
    items = load_benchmark_dataset(FIXTURES_DIR / "benchmark_smoke.jsonl")
    assert len(items) == 4
    assert items[0].query_type == "CONFIGURATION"
    assert collect_benchmark_sources(items) == [
        "nexus_api_reference.txt",
        "nexus_incident_response_runbook.txt",
        "nexus_platform_configuration_guide.txt",
    ]


def test_invalid_benchmark_item_raises_validation_error(tmp_path: Path):
    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text(
        json.dumps(
            {
                "id": "broken",
                "question": "Missing query type",
                "gold_answer_points": [],
                "gold_sources": [],
                "answerable": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_benchmark_dataset(invalid_path)


def test_cleanup_temp_path_removes_directory_tree(tmp_path: Path):
    temp_dir = tmp_path / "cag_eval_temp"
    temp_dir.mkdir()
    (temp_dir / "nested.txt").write_text("temp", encoding="utf-8")

    cleanup_temp_path(temp_dir)

    assert not temp_dir.exists()


def test_cleanup_temp_path_ignores_missing_targets(tmp_path: Path):
    missing = tmp_path / "missing_dir"
    cleanup_temp_path(missing)
    assert not missing.exists()


def test_runners_share_the_same_retrieval_top_k():
    calls: list[int] = []
    docs = [
        Document(
            page_content="Test context",
            metadata={"filename": "team_handbook_setup.txt", "domain_module": "workflow"},
        )
    ]

    def search_fn(query: str, k: int):
        calls.append(k)
        return docs

    def fake_query_runner(question: str, conversation_history=None, search_fn=None):
        if search_fn is not None:
            search_fn(question, 7)
        return {
            "answer": "Supported answer",
            "citations": [{"text": "Test context", "source": "team_handbook_setup.txt", "domain_module": "workflow"}],
            "query_type": "CONFIGURATION",
            "confidence": 0.8,
            "hallucination_risk": 0.1,
            "should_escalate": False,
            "fallback_used": False,
            "fallback_reason": "",
            "node_trace": ["ENTRY", "RETRIEVE", "REFINE", "REASON(retry=0)", "VALIDATE", "EXIT"],
        }

    class FakeAgent:
        def run(self, prompt: str):
            return SimpleNamespace(
                content=BaselineGeneration(
                    answer="Baseline answer",
                    query_type="CONFIGURATION",
                    confidence=0.7,
                    citations=[CitationRecord(text="Test context", source="team_handbook_setup.txt", domain_module="workflow")],
                    hallucination_risk=0.2,
                    should_escalate=False,
                )
            )

    run_cag_system("q1", "How do I configure the workflow?", search_fn, top_k=7, query_runner=fake_query_runner)
    run_rag_baseline("q1", "How do I configure the workflow?", search_fn, top_k=7, agent=FakeAgent())

    assert calls == [7, 7, 7]


def test_cag_and_cag_no_selection_share_the_same_selected_context_budget():
    docs = [
        Document(
            page_content=f"Context chunk {index}",
            metadata={"filename": f"source_{index}.txt", "domain_module": "workflow", "chunk_index": index},
        )
        for index in range(8)
    ]

    def search_fn(query: str, k: int):
        return docs

    def fake_query_runner(question: str, conversation_history=None, search_fn=None):
        ranked_chunks = [
            {
                "content": doc.page_content,
                "source": doc.metadata["filename"],
                "domain_module": doc.metadata["domain_module"],
                "chunk_index": doc.metadata["chunk_index"],
                "cluster_id": "cluster_1",
                "selection_category": "setup",
                "relevance_score": 0.9 - (doc.metadata["chunk_index"] * 0.01),
                "relevance_reason": "Test ranking",
            }
            for doc in docs
        ]
        return {
            "answer": "Supported answer",
            "chunks": ranked_chunks,
            "ranked_chunks": ranked_chunks,
            "citations": [],
            "query_type": "CONFIGURATION",
            "confidence": 0.8,
            "hallucination_risk": 0.1,
            "should_escalate": False,
            "fallback_used": False,
            "fallback_reason": "",
            "node_trace": ["ENTRY", "RETRIEVE", "REFINE", "REASON(retry=0)", "VALIDATE", "EXIT"],
        }

    cag_result = run_cag_system("q1", "How do I configure the workflow?", search_fn, top_k=8, query_runner=fake_query_runner)
    no_selection_result = run_cag_no_selection(
        "q1",
        "How do I configure the workflow?",
        search_fn,
        top_k=8,
        query_runner=fake_query_runner,
    )

    assert cag_result.retrieved_chunk_count == 8
    assert no_selection_result.retrieved_chunk_count == 8
    assert cag_result.selected_chunk_count == 6
    assert no_selection_result.selected_chunk_count == 6
    assert len(cag_result.selected_context_sources) == 6
    assert len(no_selection_result.selected_context_sources) == 6


def test_scoring_covers_supported_and_unsupported_cases():
    answerable_item = BenchmarkItem(
        id="supported",
        question="What prerequisites are required before using the workflow?",
        query_type="CONFIGURATION",
        gold_answer_points=[
            "active project",
            "assigned owner",
        ],
        gold_sources=["team_handbook_setup.txt"],
        answerable=True,
    )
    supported_output = SystemOutput(
        question_id="supported",
        question=answerable_item.question,
        system="cag",
        answer="An active project and an assigned owner are required before using the workflow.",
        citations=[CitationRecord(source="team_handbook_setup.txt", text="...", domain_module="workflow")],
        selected_context_sources=["team_handbook_setup.txt"],
        query_type="CONFIGURATION",
        should_escalate=False,
    )
    supported_score = score_result(answerable_item, supported_output, judge=None)
    assert supported_score.point_coverage == 1.0
    assert supported_score.context_precision_score == 1.0
    assert supported_score.task_success is True
    assert supported_score.hallucination_flag is False

    incomplete_output = SystemOutput(
        question_id="supported",
        question=answerable_item.question,
        system="rag_baseline",
        answer="An active project is required.",
        citations=[CitationRecord(source="team_handbook_setup.txt", text="...", domain_module="workflow")],
        selected_context_sources=["wrong_source.txt"],
        query_type="CONFIGURATION",
        should_escalate=False,
    )
    incomplete_score = score_result(answerable_item, incomplete_output, judge=None)
    assert incomplete_score.point_coverage < 1.0
    assert incomplete_score.context_precision_score == 0.0
    assert incomplete_score.task_success is False

    unsupported_item = BenchmarkItem(
        id="unsupported",
        question="How do I configure blockchain-based approval signing?",
        query_type="CONFIGURATION",
        gold_answer_points=[],
        gold_sources=[],
        answerable=False,
    )
    escalation_output = SystemOutput(
        question_id="unsupported",
        question=unsupported_item.question,
        system="cag",
        answer="The available documentation does not cover blockchain-based approval signing.",
        citations=[],
        query_type="CONFIGURATION",
        should_escalate=True,
    )
    escalation_score = score_result(unsupported_item, escalation_output, judge=None)
    assert escalation_score.task_success is True
    assert escalation_score.context_precision_score is None
    assert escalation_score.grounded_answer_score == 1.0

    hallucinated_output = SystemOutput(
        question_id="unsupported",
        question=unsupported_item.question,
        system="rag_baseline",
        answer="Open Settings > Blockchain and enable the signing certificate.",
        citations=[],
        query_type="CONFIGURATION",
        should_escalate=False,
    )
    hallucinated_score = score_result(unsupported_item, hallucinated_output, judge=None)
    assert hallucinated_score.task_success is False
    assert hallucinated_score.hallucination_flag is True


def test_compare_cli_is_reproducible_on_same_inputs(tmp_path: Path):
    run1 = tmp_path / "run_cag"
    run2 = tmp_path / "run_rag"
    run1.mkdir()
    run2.mkdir()

    result = SystemOutput(
        question_id="q1",
        question="Question",
        system="cag",
        answer="Answer",
        citations=[],
        query_type="GENERAL",
        should_escalate=False,
    )
    item = BenchmarkItem(
        id="q1",
        question="Question",
        query_type="GENERAL",
        gold_answer_points=["Answer"],
        gold_sources=[],
        answerable=True,
    )
    scored = score_result(item, result, judge=None)
    scored_rag = scored.model_copy(update={"system": "rag_baseline", "grounded_answer_score": 0.2})

    for base, system_name, metrics in [
        (run1, "cag", aggregate_results([scored])),
        (run2, "rag_baseline", aggregate_results([scored_rag])),
    ]:
        manifest = RunManifest(
            run_id=base.name,
            system=system_name,
            dataset_path="fixture",
            data_dir="fixture",
            judge_mode="off",
            judge_active=False,
            benchmark_sources=[],
            model_id="test-model",
            embedding_model="test-embedding",
            top_k=3,
            metrics=metrics,
            by_query_type={"GENERAL": metrics},
        )
        (base / "run.json").write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
        (base / "results.jsonl").write_text(
            json.dumps(
                (scored if system_name == "cag" else scored_rag).model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    output_base = tmp_path / "comparisons"
    for _ in range(2):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cag.eval.compare",
                "--runs",
                str(run1),
                str(run2),
                "--output-dir",
                str(output_base),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0

    comparison_dirs = sorted(output_base.iterdir())
    assert len(comparison_dirs) == 2
    first = json.loads((comparison_dirs[0] / "comparison.json").read_text(encoding="utf-8"))
    second = json.loads((comparison_dirs[1] / "comparison.json").read_text(encoding="utf-8"))
    assert first["claim_verdict"] == second["claim_verdict"]
    assert first["metrics_by_system"] == second["metrics_by_system"]
    assert first["delta_vs_rag_baseline"] == second["delta_vs_rag_baseline"]


def test_parse_lightrag_response_extracts_references():
    answer, citations = parse_lightrag_response(
        {
            "response": "Enable notifications and assign an owner.",
            "references": [
                {
                    "reference_id": "1",
                    "file_path": "/docs/team_handbook_setup.txt",
                    "content": ["Enable notifications.", "Assign an owner."],
                }
            ],
        }
    )

    assert answer == "Enable notifications and assign an owner."
    assert len(citations) == 1
    assert citations[0].source == "team_handbook_setup.txt"
    assert "Enable notifications." in citations[0].text


def test_parse_lightrag_response_extracts_markdown_references():
    answer, citations = parse_lightrag_response(
        """Enable notifications first.

### References

- [1] data\\raw\\team_handbook_setup.txt
"""
    )

    assert answer == "Enable notifications first."
    assert [citation.source for citation in citations] == ["team_handbook_setup.txt"]


def test_lightrag_escalation_and_dispatch():
    assert infer_should_escalate("") is True
    assert infer_should_escalate("The documentation available does not cover this setting.") is True
    assert infer_should_escalate("There is not enough information in the provided context.") is True

    class FakeRuntime:
        def query(self, question_id: str, question: str):
            return SystemOutput(
                question_id=question_id,
                question=question,
                system="lightrag_baseline",
                answer="Answer from LightRAG",
                citations=[],
                query_type="GENERAL",
                should_escalate=False,
                node_trace=["LIGHTRAG"],
            )

    result = run_system(
        system="lightrag_baseline",
        question_id="q-light",
        question="What does the document say?",
        search_fn=lambda *_args, **_kwargs: [],
        top_k=5,
        runtime=FakeRuntime(),
    )

    assert result.system == "lightrag_baseline"
    assert result.answer == "Answer from LightRAG"


def test_stopwords_are_english_only():
    from cag.eval.scoring import STOPWORDS

    italian_words = {
        "ad", "al", "alla", "alle", "anche", "che", "con", "da", "dal",
        "dei", "del", "della", "delle", "di", "è", "gli", "il", "la",
        "le", "lo", "nel", "nella", "per", "un", "una",
    }
    overlap = italian_words & STOPWORDS
    assert not overlap, f"Italian stopwords found in scoring: {overlap}"


def test_paired_statistical_test_with_known_data():
    from cag.eval.compare import _paired_statistical_test

    cag_results = [
        ScoredResult(
            question_id=f"q{i}",
            question=f"Question {i}",
            system="cag",
            answer=f"Answer {i}",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.8 + i * 0.01,
        )
        for i in range(10)
    ]
    rag_results = [
        ScoredResult(
            question_id=f"q{i}",
            question=f"Question {i}",
            system="rag_baseline",
            answer=f"Answer {i}",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.5 + i * 0.01,
        )
        for i in range(10)
    ]

    test_result = _paired_statistical_test(cag_results, rag_results)
    assert test_result["cag_mean"] > test_result["rag_mean"]
    assert test_result["mean_difference"] > 0
    assert test_result["n_pairs"] == 10


def test_paired_statistical_test_averages_replicated_questions():
    from cag.eval.compare import _paired_statistical_test

    cag_results = [
        ScoredResult(
            question_id="q1",
            question="Question 1",
            system="cag",
            answer="Answer",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.8,
        ),
        ScoredResult(
            question_id="q1",
            question="Question 1",
            system="cag",
            answer="Answer",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.9,
        ),
    ]
    rag_results = [
        ScoredResult(
            question_id="q1",
            question="Question 1",
            system="rag_baseline",
            answer="Answer",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.5,
        ),
        ScoredResult(
            question_id="q1",
            question="Question 1",
            system="rag_baseline",
            answer="Answer",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.7,
        ),
    ]

    test_result = _paired_statistical_test(cag_results, rag_results)
    assert test_result["n_pairs"] == 1
    assert abs(test_result["cag_mean"] - 0.85) < 0.001
    assert abs(test_result["rag_mean"] - 0.60) < 0.001
    assert abs(test_result["mean_difference"] - 0.25) < 0.001


def test_compare_cli_supports_multi_run_directories(tmp_path: Path):
    def write_run(base: Path, system_name: str, score: float, run_index: int) -> None:
        run_dir = base / f"run_{run_index:03d}"
        run_dir.mkdir(parents=True)

        scored = ScoredResult(
            question_id="q1",
            question="Question",
            system=system_name,
            answer="Answer",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=score,
            task_success=True,
        )
        metrics = aggregate_results([scored])
        manifest = RunManifest(
            run_id=run_dir.name,
            system=system_name,
            dataset_path="fixture",
            data_dir="fixture",
            judge_mode="off",
            judge_active=False,
            benchmark_sources=[],
            model_id="test-model",
            embedding_model="test-embedding",
            top_k=3,
            metrics=metrics,
            by_query_type={"GENERAL": metrics},
            run_index=run_index,
        )
        (run_dir / "run.json").write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
        (run_dir / "results.jsonl").write_text(
            json.dumps(scored.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    cag_base = tmp_path / "cag_multi"
    rag_base = tmp_path / "rag_multi"
    cag_base.mkdir()
    rag_base.mkdir()

    for base, system_name, scores in [
        (cag_base, "cag", [0.8, 0.9]),
        (rag_base, "rag_baseline", [0.4, 0.5]),
    ]:
        for run_index, score in enumerate(scores, start=1):
            write_run(base, system_name, score, run_index)
        (base / "multi_run.json").write_text(
            json.dumps(
                {
                    "run_id": base.name,
                    "system": system_name,
                    "dataset_path": "fixture",
                    "data_dir": "fixture",
                    "judge_mode": "off",
                    "judge_active": False,
                    "benchmark_sources": [],
                    "model_id": "test-model",
                    "embedding_model": "test-embedding",
                    "top_k": 3,
                    "total_runs": 2,
                    "aggregate_metrics": {},
                    "by_query_type": {},
                }
            ),
            encoding="utf-8",
        )

    output_base = tmp_path / "comparisons"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cag.eval.compare",
            "--runs",
            str(cag_base),
            str(rag_base),
            "--output-dir",
            str(output_base),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0

    comparison_dirs = sorted(output_base.iterdir())
    assert len(comparison_dirs) == 1
    payload = json.loads((comparison_dirs[0] / "comparison.json").read_text(encoding="utf-8"))
    comparison_markdown = (comparison_dirs[0] / "comparison.md").read_text(encoding="utf-8")
    assert payload["run_counts_by_system"] == {"cag": 2, "rag_baseline": 2}
    assert payload["statistical_test"]["n_pairs"] == 1
    assert "delta_vs_rag_baseline" in payload
    assert payload["delta_vs_rag_baseline"]["cag"]["grounded_answer_score"] > 0
    assert "Context Precision" in comparison_markdown
    assert "Delta vs RAG Baseline" in comparison_markdown


def test_cag_cli_module_entrypoint_shows_usage():
    completed = subprocess.run(
        [sys.executable, "-m", "cag.cli"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "usage:" in combined_output.lower()


def test_multi_run_aggregation():
    from cag.eval.models import MetricStats
    from cag.eval.run import _compute_multi_run_stats

    run_a = [
        ScoredResult(
            question_id="q1",
            question="Q1",
            system="cag",
            answer="A1",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.8,
            context_precision_score=0.8,
            task_success=True,
        )
    ]
    run_b = [
        ScoredResult(
            question_id="q1",
            question="Q1",
            system="cag",
            answer="A1",
            citations=[],
            query_type="GENERAL",
            expected_query_type="GENERAL",
            answerable=True,
            grounded_answer_score=0.9,
            context_precision_score=0.6,
            task_success=True,
        )
    ]

    stats = _compute_multi_run_stats([run_a, run_b])
    assert "grounded_answer_score" in stats
    gas = stats["grounded_answer_score"]
    assert isinstance(gas, MetricStats)
    assert gas.n == 2
    assert abs(gas.mean - 0.85) < 0.001
    assert abs(gas.std - 0.0707) < 0.01
    assert "context_precision_score" in stats
    assert abs(stats["context_precision_score"].mean - 0.7) < 0.001


def test_dataset_audit_flags_invalid_items_and_coverage_gaps(tmp_path: Path):
    dataset_path = tmp_path / "audit_dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "dup_01",
                        "question": "What is the logging level?",
                        "query_type": "GENERAL",
                        "gold_answer_points": ["INFO"],
                        "gold_sources": ["nexus_platform_configuration_guide.txt"],
                        "answerable": True,
                    }
                ),
                json.dumps(
                    {
                        "id": "dup_01",
                        "question": "What is the logging level?",
                        "query_type": "GENERAL",
                        "gold_answer_points": ["INFO"],
                        "gold_sources": [],
                        "answerable": True,
                    }
                ),
                json.dumps(
                    {
                        "id": "cfg_01",
                        "question": "How do I rotate the SMTP password?",
                        "query_type": "CONFIGURATION",
                        "gold_answer_points": ["Update SMTP credentials"],
                        "gold_sources": ["nexus_platform_configuration_guide.txt"],
                        "answerable": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "nexus_platform_configuration_guide.txt").write_text("config", encoding="utf-8")
    (data_dir / "nexus_employee_handbook.txt").write_text("handbook", encoding="utf-8")

    audit = build_dataset_audit(
        load_benchmark_dataset(dataset_path),
        dataset_path,
        data_dir,
        min_total=10,
        min_per_query_type=2,
    )

    assert audit.readiness == "invalid"
    assert audit.duplicate_ids == ["dup_01"]
    assert audit.duplicate_questions == ["what is the logging level?"]
    assert audit.answerable_without_gold_sources == ["dup_01"]
    assert audit.uncovered_corpus_sources == ["nexus_employee_handbook.txt"]
    assert any("recommended minimum" in issue.lower() for issue in audit.issues)


def test_eval_audit_cli_outputs_json(tmp_path: Path):
    dataset_path = tmp_path / "audit_dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "gen_01",
                "question": "What is the minimum RAM required to run Nexus Platform?",
                "query_type": "GENERAL",
                "gold_answer_points": ["8 GB"],
                "gold_sources": ["nexus_platform_configuration_guide.txt"],
                "answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "nexus_platform_configuration_guide.txt").write_text("config", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cag.cli",
            "eval-audit",
            "--dataset",
            str(dataset_path),
            "--data-dir",
            str(data_dir),
            "--format",
            "json",
            "--min-total",
            "1",
            "--min-per-query-type",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["readiness"] == "ready"
    assert payload["total_count"] == 1
    assert payload["gold_source_usage"]["nexus_platform_configuration_guide.txt"] == 1


def test_cag_cli_eval_supports_cag_no_selection():
    from cag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["eval", "--system", "cag_no_selection"])
    assert args.system == "cag_no_selection"


def test_run_query_uses_injected_search_function():
    from cag.graph.graph import run_query

    class FakeDoc:
        page_content = "Workflow setup instructions."
        metadata = {"filename": "workflow.txt", "source": "workflow.txt", "domain_module": "workflow", "chunk_index": 0}

    calls: list[tuple[str, int | None]] = []

    def fake_search(query: str, k: int | None = None):
        calls.append((query, k))
        return [FakeDoc()]

    from cag.agents.models import Citation, ReasoningOutput, RankedChunk, RetrievalOutput

    with patch(
        "cag.graph.nodes.run_retrieval_agent",
        return_value=RetrievalOutput(
            chunks_ranked=[
                RankedChunk(
                    content="Workflow setup instructions.",
                    source="workflow.txt",
                    domain_module="workflow",
                    relevance_score=0.9,
                    relevance_reason="direct",
                )
            ],
            gaps=[],
            relevance_score=0.9,
            summary="ok",
        ),
    ), patch(
        "cag.graph.nodes.run_reasoning_agent",
        return_value=ReasoningOutput(
            answer="Workflow setup instructions.",
            query_type="CONFIGURATION",
            confidence=0.8,
            citations=[Citation(text="Workflow setup instructions.", source="workflow.txt", domain_module="workflow")],
            hallucination_risk=0.1,
            hallucination_reason="grounded",
        ),
    ):
        result = run_query("How do I configure the workflow?", search_fn=fake_search)

    assert result["answer"]
    assert calls


def test_fallback_flags_are_propagated_in_eval_output():
    def search_fn(query: str, k: int):
        return []

    def fake_query_runner(question: str, conversation_history=None, search_fn=None):
        return {
            "answer": "Fallback answer",
            "citations": [],
            "query_type": "GENERAL",
            "confidence": 0.0,
            "hallucination_risk": 1.0,
            "should_escalate": True,
            "fallback_used": True,
            "fallback_reason": "reasoning_agent_error",
            "node_trace": ["ENTRY", "RETRIEVE", "REFINE", "REASON(retry=0)", "VALIDATE", "EXIT"],
        }

    result = run_cag_system("q1", "Question", search_fn, top_k=5, query_runner=fake_query_runner)
    assert result.fallback_used is True
    assert result.fallback_reason == "reasoning_agent_error"
