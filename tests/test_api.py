from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from cag.api.upload import ConversationRoute, _empty_ingest_status, _ingest_dir, _ingest_status, _route_conversation_turn, app
from cag.config import settings


client = TestClient(app)


def test_query_requires_api_key_when_configured():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post("/query", json={"query": "Hello"})
        assert response.status_code == 401
    finally:
        settings.cag_api_key = original_key


def test_query_rejects_invalid_api_key():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post("/query", headers={"X-API-Key": "wrong"}, json={"query": "Hello"})
        assert response.status_code == 403
    finally:
        settings.cag_api_key = original_key


def test_query_accepts_valid_api_key_and_does_not_mutate_settings():
    original_api_key = settings.cag_api_key
    original_relevance = settings.relevance_threshold
    original_confidence = settings.confidence_threshold
    original_hallucination = settings.hallucination_threshold
    settings.cag_api_key = "secret-key"
    try:
        with patch("cag.api.upload.run_query") as mock_run_query:
            mock_run_query.return_value = {"answer": "ok", "fallback_used": False, "fallback_reason": ""}
            response = client.post(
                "/query",
                headers={"X-API-Key": "secret-key"},
                json={
                    "query": "Hello",
                    "relevance_threshold": 0.8,
                    "confidence_threshold": 0.7,
                    "hallucination_threshold": 0.2,
                },
            )

        assert response.status_code == 200
        kwargs = mock_run_query.call_args.kwargs
        runtime_config = kwargs["runtime_config"]
        assert runtime_config.relevance_threshold == 0.8
        assert runtime_config.confidence_threshold == 0.7
        assert runtime_config.hallucination_threshold == 0.2
        assert settings.relevance_threshold == original_relevance
        assert settings.confidence_threshold == original_confidence
        assert settings.hallucination_threshold == original_hallucination
    finally:
        settings.cag_api_key = original_api_key


def test_query_response_ignores_internal_non_serializable_state():
    with patch("cag.api.upload.run_query") as mock_run_query:
        mock_run_query.return_value = {
            "answer": "ok",
            "confidence": 0.8,
            "fallback_used": False,
            "fallback_reason": "",
            "search_fn": lambda *_args, **_kwargs: [],
        }
        response = client.post(
            "/query",
            headers={"Origin": "http://localhost:5175"},
            json={"query": "Hello"},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5175"
    assert response.json()["answer"] == "ok"
    assert "search_fn" not in response.json()


def test_query_handles_conversation_transform_without_retrieval():
    history = [
        {"role": "user", "content": "Summarize the latest uploaded document."},
        {"role": "assistant", "content": "TeamSystem provides HR and ERP software."},
    ]
    with (
        patch("cag.api.upload._route_conversation_turn", return_value=ConversationRoute(action="answer_transform")),
        patch("cag.api.upload.run_query") as mock_run_query,
        patch("cag.api.upload._transform_previous_answer", return_value="TeamSystem fornisce software HR ed ERP.") as mock_transform,
    ):
        response = client.post(
            "/query",
            json={"query": "me lo puoi tradurre in italiano?", "conversation_history": history},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "TeamSystem fornisce software HR ed ERP."
    assert payload["query_type"] == "CONVERSATION_TRANSFORM"
    assert payload["should_escalate"] is False
    assert payload["node_trace"] == ["CONVERSATION_TRANSFORM"]
    mock_transform.assert_called_once()
    mock_run_query.assert_not_called()


def test_query_handles_general_previous_answer_rewrite_without_retrieval():
    history = [
        {"role": "assistant", "content": "CAG selects evidence before reasoning."},
    ]
    with (
        patch("cag.api.upload._route_conversation_turn", return_value=ConversationRoute(action="answer_transform")),
        patch("cag.api.upload.run_query") as mock_run_query,
        patch("cag.api.upload._transform_previous_answer", return_value="- CAG selects evidence first.") as mock_transform,
    ):
        response = client.post(
            "/query",
            json={"query": "puoi metterlo in elenco puntato?", "conversation_history": history},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "- CAG selects evidence first."
    assert payload["query_type"] == "CONVERSATION_TRANSFORM"
    assert payload["should_escalate"] is False
    mock_transform.assert_called_once()
    mock_run_query.assert_not_called()


def test_query_rewrites_elaboration_followup_for_retrieval():
    history = [
        {"role": "user", "content": "ciao di cosa vi occupate?"},
        {
            "role": "assistant",
            "content": (
                "La documentazione indica servizi IT, soluzioni TeamSystem, "
                "tecnologie innovative e infrastrutture IT."
            ),
        },
    ]
    with (
        patch(
            "cag.api.upload._route_conversation_turn",
            return_value=ConversationRoute(
                action="rewrite_for_retrieval",
                rewritten_query=(
                    "Quali servizi IT, soluzioni TeamSystem, tecnologie innovative "
                    "e infrastrutture IT offre l'azienda?"
                ),
            ),
        ) as mock_route,
        patch("cag.api.upload._rewrite_query_with_history") as mock_rewrite,
        patch("cag.api.upload.run_query") as mock_run_query,
    ):
        mock_run_query.return_value = {
            "answer": "Studio 81 offre servizi IT e soluzioni TeamSystem.",
            "confidence": 0.8,
            "fallback_used": False,
            "fallback_reason": "",
        }
        response = client.post(
            "/query",
            json={"query": "possiamo approfondire", "conversation_history": history},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Studio 81 offre servizi IT e soluzioni TeamSystem."
    mock_route.assert_called_once()
    mock_rewrite.assert_not_called()
    assert "Quali servizi IT" in mock_run_query.call_args.kwargs["query"]


def test_conversation_route_uses_deterministic_rewrite_without_llm():
    history = [
        {"role": "user", "content": "ciao di cosa vi occupate?"},
        {"role": "assistant", "content": "Studio 81 offre servizi IT, TeamSystem, cloud e sicurezza."},
    ]

    with patch("agno.agent.Agent") as mock_agent:
        route = _route_conversation_turn("spiegami meglio come potete esserci di aiuto", history)

    assert route.action == "rewrite_for_retrieval"
    assert "servizi" in route.rewritten_query.lower()
    assert "teamsystem" in route.rewritten_query.lower()
    mock_agent.assert_not_called()


def test_query_rewrites_contextual_followup_before_retrieval():
    history = [
        {"role": "user", "content": "What does the handbook say about remote work?"},
        {"role": "assistant", "content": "Remote work is allowed up to three days per week."},
    ]
    with (
        patch(
            "cag.api.upload._route_conversation_turn",
            return_value=ConversationRoute(
                action="rewrite_for_retrieval",
                rewritten_query="Does the remote work policy also require manager approval?",
            ),
        ) as mock_route,
        patch("cag.api.upload._rewrite_query_with_history") as mock_rewrite,
        patch("cag.api.upload.run_query") as mock_run_query,
    ):
        mock_run_query.return_value = {
            "answer": "Yes.",
            "confidence": 0.8,
            "fallback_used": False,
            "fallback_reason": "",
        }
        response = client.post(
            "/query",
            json={"query": "e serve anche approvazione?", "conversation_history": history},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Yes."
    mock_route.assert_called_once()
    mock_rewrite.assert_not_called()
    assert mock_run_query.call_args.kwargs["query"] == "Does the remote work policy also require manager approval?"


def test_query_falls_back_to_rewrite_when_model_router_fails_on_elaboration():
    history = [
        {"role": "user", "content": "ciao di cosa vi occupate?"},
        {"role": "assistant", "content": "Studio 81 offre servizi IT e soluzioni TeamSystem."},
    ]
    with (
        patch("cag.api.upload._route_conversation_turn", side_effect=RuntimeError("model offline")),
        patch("cag.api.upload._rewrite_query_with_history", side_effect=RuntimeError("rewrite offline")) as mock_rewrite,
        patch("cag.api.upload.run_query") as mock_run_query,
    ):
        mock_run_query.return_value = {
            "answer": "Approfondimento dai documenti.",
            "confidence": 0.8,
            "fallback_used": False,
            "fallback_reason": "",
        }
        response = client.post(
            "/query",
            json={"query": "possiamo approfondire", "conversation_history": history},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Approfondimento dai documenti."
    mock_rewrite.assert_not_called()
    rewritten_query = mock_run_query.call_args.kwargs["query"]
    assert "ciao di cosa vi occupate" in rewritten_query
    assert "servizi" in rewritten_query.lower()


def test_query_ignores_unusable_model_rewrite_output():
    history = [
        {"role": "user", "content": "ciao di cosa vi occupate?"},
        {"role": "assistant", "content": "Studio 81 offre servizi IT e soluzioni TeamSystem."},
    ]
    with (
        patch(
            "cag.api.upload._route_conversation_turn",
            return_value=ConversationRoute(action="rewrite_for_retrieval"),
        ),
        patch("cag.api.upload._rewrite_query_with_history", return_value="Connection error.") as mock_rewrite,
        patch("cag.api.upload.run_query") as mock_run_query,
    ):
        mock_run_query.return_value = {
            "answer": "Approfondimento dai documenti.",
            "confidence": 0.8,
            "fallback_used": False,
            "fallback_reason": "",
        }
        response = client.post(
            "/query",
            json={"query": "possiamo approfondire?", "conversation_history": history},
        )

    assert response.status_code == 200
    mock_rewrite.assert_called_once()
    rewritten_query = mock_run_query.call_args.kwargs["query"]
    assert rewritten_query != "Connection error."
    assert "servizi" in rewritten_query.lower()


def test_query_rejects_out_of_range_thresholds():
    response = client.post("/query", json={"query": "Hello", "relevance_threshold": 1.2})
    assert response.status_code == 422


def test_query_rejects_blank_query():
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422


def test_retrieval_diagnostics_returns_selected_context_shape():
    with (
        patch("cag.api.upload.retrieve_node") as mock_retrieve,
        patch("cag.api.upload.select_context_node") as mock_select,
    ):
        mock_retrieve.return_value = {
            "document_candidates": [
                {
                    "filename": "workflow.txt",
                    "score": 3.0,
                    "match_reason": "Matched profile terms: workflow",
                    "generator": "llm",
                }
            ],
            "chunks": [
                {
                    "content": "Workflow setup instructions.",
                    "source": "workflow.txt",
                    "domain_module": "workflow",
                    "chunk_index": 0,
                }
            ],
            "node_trace": ["ENTRY", "RETRIEVE"],
        }
        mock_select.return_value = {
            "ranked_chunks": [
                {
                    "content": "Workflow setup instructions.",
                    "source": "workflow.txt",
                    "domain_module": "workflow",
                    "chunk_index": 0,
                    "relevance_score": 0.9,
                }
            ],
            "gaps": [],
            "relevance_score": 0.9,
            "node_trace": ["ENTRY", "RETRIEVE", "SELECT_CONTEXT"],
        }

        response = client.post("/diagnostics/retrieval", json={"query": "What is the workflow?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What is the workflow?"
    assert payload["query_type"] == "GENERAL"
    assert payload["retrieval_strategy"] == "semantic"
    assert payload["document_candidates"][0]["filename"] == "workflow.txt"
    assert len(payload["chunks"]) == 1
    assert len(payload["ranked_chunks"]) == 1
    assert payload["relevance_score"] == 0.9
    assert payload["node_trace"] == ["ENTRY", "RETRIEVE", "SELECT_CONTEXT"]


def test_upload_requires_api_key_when_configured():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post(
            "/upload",
            files={"files": ("demo.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 401
    finally:
        settings.cag_api_key = original_key


def test_upload_rejects_invalid_extension():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post(
            "/upload",
            headers={"X-API-Key": "secret-key"},
            files={"files": ("malware.exe", BytesIO(b"boom"), "application/octet-stream")},
        )
        assert response.status_code == 400
    finally:
        settings.cag_api_key = original_key


def test_upload_rejects_large_file():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post(
            "/upload",
            headers={"X-API-Key": "secret-key"},
            files={"files": ("large.txt", BytesIO(b"a" * (10 * 1024 * 1024 + 1)), "text/plain")},
        )
        assert response.status_code == 413
    finally:
        settings.cag_api_key = original_key


def test_upload_response_returns_filenames_not_absolute_paths(tmp_path: Path):
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        with patch("cag.api.upload._ensure_raw_dir", return_value=tmp_path):
            response = client.post(
                "/upload?ingest=false",
                headers={"X-API-Key": "secret-key"},
                files={"files": ("demo.txt", BytesIO(b"hello"), "text/plain")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["saved"] == ["demo.txt"]
        assert not Path(payload["saved"][0]).is_absolute()
        assert (tmp_path / "demo.txt").read_bytes() == b"hello"
    finally:
        settings.cag_api_key = original_key


def test_files_endpoint_lists_uploaded_documents(tmp_path: Path):
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "ignored.exe").write_text("ignored", encoding="utf-8")
    try:
        with patch("cag.api.upload._ensure_raw_dir", return_value=tmp_path):
            response = client.get("/files", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["files"][0]["name"] == "alpha.txt"
        assert payload["files"][0]["size_bytes"] == 5
        assert "modified" in payload["files"][0]
    finally:
        settings.cag_api_key = original_key


def test_document_profiles_endpoint_returns_compiled_profiles():
    with patch("cag.api.upload.list_document_profiles") as mock_profiles:
        mock_profiles.return_value = [
            {
                "profile_id": "profile-1",
                "source_version_id": "version-1",
                "filename": "handbook.txt",
                "source": "data/raw/handbook.txt",
                "version": 1,
                "summary": "Company overview and handbook.",
                "keywords": ["company", "overview"],
                "topics": ["overview"],
                "entities": ["Company"],
                "covered_intents": ["Questions about the company"],
                "status": "active",
                "generator": "llm",
                "created_at": "2026-05-10 20:00:00",
                "chunk_count": 3,
            }
        ]

        response = client.get("/document-profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["profiles"][0]["filename"] == "handbook.txt"
    assert payload["profiles"][0]["generator"] == "llm"
    assert payload["profiles"][0]["chunk_count"] == 3


def test_knowledge_graph_endpoint_returns_nodes_and_edges():
    graph = {
        "nodes": [
            {"id": "source:version-1", "type": "source", "label": "handbook.txt", "properties": {}},
            {"id": "claim:claim-1", "type": "claim", "label": "A supported claim.", "properties": {}},
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "source:version-1",
                "target": "claim:claim-1",
                "relation": "contains_claim",
                "confidence": 0.8,
                "evidence_chunk_id": "chunk-1",
            }
        ],
    }
    with (
        patch("cag.api.upload.connect") as mock_connect,
        patch("cag.api.upload.initialize") as mock_initialize,
        patch("cag.api.upload.list_knowledge_graph", return_value=graph) as mock_graph,
    ):
        mock_connection = mock_connect.return_value
        response = client.get("/knowledge-graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_nodes"] == 2
    assert payload["total_edges"] == 1
    assert payload["edges"][0]["relation"] == "contains_claim"
    mock_initialize.assert_called_once_with(mock_connection)
    mock_graph.assert_called_once_with(mock_connection)
    mock_connection.close.assert_called_once()


def test_ingest_status_endpoint_returns_latest_status():
    original_status = dict(_ingest_status)
    try:
        _ingest_status.clear()
        _ingest_status.update(_empty_ingest_status())
        _ingest_status.update({"status": "failed", "stage": "load", "message": "boom", "chunks_indexed": 0})
        response = client.get("/ingest/status")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "failed"
        assert payload["stage"] == "load"
        assert payload["message"] == "boom"
        assert payload["chunks_indexed"] == 0
        assert payload["steps"]
    finally:
        _ingest_status.clear()
        _ingest_status.update(original_status)


def test_upload_records_ingest_failure(tmp_path: Path):
    original_status = dict(_ingest_status)
    try:
        with (
            patch("cag.api.upload.load_documents", side_effect=RuntimeError("tokenizer download failed")),
            patch("cag.api.upload._ensure_raw_dir", return_value=tmp_path),
        ):
            response = client.post(
                "/upload?ingest=true",
                files={"files": ("demo.txt", BytesIO(b"hello"), "text/plain")},
            )

        assert response.status_code == 200
        assert _ingest_status["status"] == "failed"
        assert _ingest_status["stage"] == "load"
        assert "tokenizer download failed" in _ingest_status["message"]
        assert any(step["status"] == "failed" for step in _ingest_status["steps"])
    finally:
        _ingest_status.clear()
        _ingest_status.update(original_status)


def test_ingest_dir_reports_pipeline_counts(tmp_path: Path):
    original_status = dict(_ingest_status)
    source = tmp_path / "demo.txt"
    source.write_text("hello", encoding="utf-8")
    document = Document(page_content="hello world", metadata={"filename": "demo.txt", "source": str(source)})
    chunk = Document(page_content="hello world", metadata={"filename": "demo.txt", "source": str(source)})

    try:
        with (
            patch("cag.api.upload.load_documents", return_value=[document]),
            patch("cag.api.upload.chunk_documents", return_value=[chunk]),
            patch(
                "cag.api.upload.compile_chunks",
                return_value={"sources": 1, "chunks": 1, "claims": 2, "document_profiles": 1},
            ),
            patch("cag.api.upload.get_embeddings"),
            patch("cag.api.upload.get_vector_store"),
            patch("cag.api.upload.upsert_chunks"),
        ):
            _ingest_dir(tmp_path)

        assert _ingest_status["status"] == "ready"
        assert _ingest_status["progress"] == 1.0
        assert _ingest_status["files_total"] == 1
        assert _ingest_status["documents_loaded"] == 1
        assert _ingest_status["chunks_created"] == 1
        assert _ingest_status["claims_created"] == 2
        assert _ingest_status["profiles_created"] == 1
        assert _ingest_status["vectors_indexed"] == 1
        assert all(step["status"] == "done" for step in _ingest_status["steps"])
    finally:
        _ingest_status.clear()
        _ingest_status.update(original_status)


def test_demo_reset_replaces_raw_documents_and_schedules_ingest(tmp_path: Path):
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    raw_dir = tmp_path / "raw"
    demo_dir = tmp_path / "demo"
    raw_dir.mkdir()
    demo_dir.mkdir()
    (raw_dir / "old.txt").write_text("old", encoding="utf-8")
    (raw_dir / "ignored.bin").write_text("keep", encoding="utf-8")
    (demo_dir / "nexus_demo.txt").write_text("demo", encoding="utf-8")
    try:
        with (
            patch("cag.api.upload._ensure_raw_dir", return_value=raw_dir),
            patch("cag.api.upload._benchmark_corpus_dir", return_value=demo_dir),
            patch("cag.api.upload._ingest_dir") as mock_ingest,
        ):
            response = client.post("/demo/reset", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "copied": ["nexus_demo.txt"], "ingest_started": True}
        assert not (raw_dir / "old.txt").exists()
        assert (raw_dir / "ignored.bin").exists()
        assert (raw_dir / "nexus_demo.txt").read_text(encoding="utf-8") == "demo"
        mock_ingest.assert_called_once_with(str(raw_dir))
    finally:
        settings.cag_api_key = original_key


def test_demo_reset_can_skip_ingest(tmp_path: Path):
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    raw_dir = tmp_path / "raw"
    demo_dir = tmp_path / "demo"
    raw_dir.mkdir()
    demo_dir.mkdir()
    (demo_dir / "nexus_demo.md").write_text("# demo", encoding="utf-8")
    try:
        with (
            patch("cag.api.upload._ensure_raw_dir", return_value=raw_dir),
            patch("cag.api.upload._benchmark_corpus_dir", return_value=demo_dir),
            patch("cag.api.upload._ingest_dir") as mock_ingest,
        ):
            response = client.post("/demo/reset?ingest=false", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        assert response.json()["ingest_started"] is False
        mock_ingest.assert_not_called()
    finally:
        settings.cag_api_key = original_key


def test_reset_all_deletes_documents_knowledge_and_vector_index(tmp_path: Path):
    original_key = settings.cag_api_key
    original_knowledge_path = settings.knowledge_db_path
    settings.cag_api_key = "secret-key"
    settings.knowledge_db_path = tmp_path / "knowledge.db"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / ".gitkeep").write_text("", encoding="utf-8")
    (raw_dir / "alpha.pdf").write_text("alpha", encoding="utf-8")
    (raw_dir / "notes.md").write_text("notes", encoding="utf-8")
    settings.knowledge_db_path.write_text("db", encoding="utf-8")
    try:
        with (
            patch("cag.api.upload._ensure_raw_dir", return_value=raw_dir),
            patch("cag.api.upload._reset_vector_store", return_value=True) as mock_vector_reset,
        ):
            response = client.delete("/reset/all", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["deleted_files"] == ["alpha.pdf", "notes.md"]
        assert payload["knowledge_deleted"] is True
        assert payload["vector_reset"] is True
        assert (raw_dir / ".gitkeep").exists()
        assert not (raw_dir / "alpha.pdf").exists()
        assert not settings.knowledge_db_path.exists()
        mock_vector_reset.assert_called_once()
        assert _ingest_status["status"] == "idle"
        assert _ingest_status["stage"] == "idle"
        assert _ingest_status["chunks_indexed"] == 0
        assert _ingest_status["progress"] == 0.0
        assert _ingest_status["steps"]
    finally:
        settings.cag_api_key = original_key
        settings.knowledge_db_path = original_knowledge_path


def test_delete_file_endpoint_removes_document_and_schedules_reindex(tmp_path: Path):
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    target = tmp_path / "alpha.txt"
    target.write_text("alpha", encoding="utf-8")
    try:
        with (
            patch("cag.api.upload._ensure_raw_dir", return_value=tmp_path),
            patch("cag.api.upload._reindex_after_delete") as mock_reindex,
        ):
            response = client.delete("/files/alpha.txt", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "deleted": "alpha.txt", "reindex_started": True}
        assert not target.exists()
        mock_reindex.assert_called_once_with(str(tmp_path))
    finally:
        settings.cag_api_key = original_key


def test_delete_file_endpoint_rejects_missing_document(tmp_path: Path):
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        with patch("cag.api.upload._ensure_raw_dir", return_value=tmp_path):
            response = client.delete("/files/missing.txt", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 404
    finally:
        settings.cag_api_key = original_key


def test_root_frontend_serves_built_index_and_assets(tmp_path: Path):
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>CAG</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('cag')", encoding="utf-8")

    with (
        patch("cag.api.upload._frontend_dist_dir", return_value=dist_dir),
        patch("cag.api.upload._frontend_assets_dir", return_value=assets_dir),
    ):
        response = client.get("/")
        asset_response = client.get("/assets/app.js")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert asset_response.status_code == 200


def test_root_frontend_redirects_when_build_is_missing(tmp_path: Path):
    missing_dist = tmp_path / "missing_dist"
    missing_assets = missing_dist / "assets"

    with (
        patch("cag.api.upload._frontend_dist_dir", return_value=missing_dist),
        patch("cag.api.upload._frontend_assets_dir", return_value=missing_assets),
    ):
        response = client.get("/", follow_redirects=False)
        asset_response = client.get("/assets/app.js")

    assert response.status_code == 307
    assert response.headers["location"] == "http://localhost:5174/"
    assert asset_response.status_code == 404
