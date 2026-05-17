from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document

from cag.knowledge.document_map import (
    DocumentProfileOutput,
    build_document_profile,
    build_hypothetical_questions,
    list_document_profiles,
    search_chunks_for_document_candidates,
    search_document_profiles,
)
from cag.ingestion.chunker import add_contextual_headers
from cag.knowledge.compiler import compile_chunks, compiled_search, extract_claims
from cag.knowledge.lint import lint_knowledge
from cag.knowledge.store import connect, initialize, load_retrieval_term_stats, rows, store_claim, store_topic
from cag.knowledge.store import list_knowledge_graph


def test_knowledge_schema_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "knowledge.db"
    connection = connect(db_path)
    try:
        initialize(connection)
        initialize(connection)
        versions = rows(connection, "SELECT version FROM schema_migrations")
    finally:
        connection.close()

    assert [row["version"] for row in versions] == [4]


def test_compile_chunks_stores_sources_claims_and_evidence(tmp_path):
    source = tmp_path / "support_faq.md"
    source.write_text(
        "Customers can reset an API token from Account Settings. "
        "Only workspace owners can update billing contacts.",
        encoding="utf-8",
    )
    chunks = [
        Document(
            page_content=source.read_text(encoding="utf-8"),
            metadata={
                "source": str(source),
                "filename": source.name,
                "domain_module": "support",
                "chunk_index": 0,
            },
        )
    ]

    summary = compile_chunks(chunks, tmp_path / "knowledge.db")

    assert summary == {"sources": 1, "chunks": 1, "claims": 2, "document_profiles": 1}
    connection = connect(tmp_path / "knowledge.db")
    try:
        assert rows(connection, "SELECT COUNT(*) AS count FROM sources")[0]["count"] == 1
        assert rows(connection, "SELECT COUNT(*) AS count FROM chunks")[0]["count"] == 1
        assert rows(connection, "SELECT COUNT(*) AS count FROM claims")[0]["count"] == 2
        assert rows(connection, "SELECT COUNT(*) AS count FROM claim_evidence")[0]["count"] == 2
        assert rows(connection, "SELECT COUNT(*) AS count FROM document_profiles")[0]["count"] == 1
    finally:
        connection.close()


def test_compile_chunks_persists_retrieval_term_statistics(tmp_path):
    source = tmp_path / "retrieval_stats.md"
    source.write_text(
        "SAML certificate rotation requires owner approval. "
        "Rotation windows should be scheduled during maintenance.",
        encoding="utf-8",
    )
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "security", "chunk_index": 0},
    )

    compile_chunks([chunk], tmp_path / "knowledge.db")
    connection = connect(tmp_path / "knowledge.db")
    try:
        doc_frequencies, corpus_size = load_retrieval_term_stats(connection)
    finally:
        connection.close()

    assert corpus_size == 1
    assert doc_frequencies["rotation"] >= 1
    assert doc_frequencies["certificate"] >= 1


def test_compile_chunks_is_idempotent_for_same_source_version(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("Admin access must be reviewed every quarter.", encoding="utf-8")
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "policy", "chunk_index": 0},
    )

    compile_chunks([chunk], tmp_path / "knowledge.db")
    compile_chunks([chunk], tmp_path / "knowledge.db")

    connection = connect(tmp_path / "knowledge.db")
    try:
        assert rows(connection, "SELECT COUNT(*) AS count FROM sources")[0]["count"] == 1
        assert rows(connection, "SELECT COUNT(*) AS count FROM source_versions")[0]["count"] == 1
        assert rows(connection, "SELECT COUNT(*) AS count FROM chunks")[0]["count"] == 1
        assert rows(connection, "SELECT COUNT(*) AS count FROM document_profiles")[0]["count"] == 1
    finally:
        connection.close()


def test_compile_chunks_preserves_multiple_loaded_sections_with_same_metadata_index(tmp_path):
    source = tmp_path / "multipage.pdf"
    source.write_text("placeholder bytes", encoding="utf-8")
    chunks = [
        Document(
            page_content="First page company overview.",
            metadata={"source": str(source), "filename": source.name, "domain_module": "profile", "chunk_index": 0},
        ),
        Document(
            page_content="Second page services and products.",
            metadata={"source": str(source), "filename": source.name, "domain_module": "profile", "chunk_index": 0},
        ),
    ]

    compile_chunks(chunks, tmp_path / "knowledge.db")

    connection = connect(tmp_path / "knowledge.db")
    try:
        stored_chunks = rows(connection, "SELECT chunk_index, content FROM chunks ORDER BY chunk_index")
    finally:
        connection.close()

    assert [row["chunk_index"] for row in stored_chunks] == [0, 1]
    assert "First page" in stored_chunks[0]["content"]
    assert "Second page" in stored_chunks[1]["content"]


def test_build_document_profile_uses_llm_when_available():
    profile = DocumentProfileOutput(
        summary="Nexus is a notification platform.",
        keywords=["notification", "platform"],
        topics=["overview"],
        entities=["Nexus"],
        covered_intents=["Questions about what Nexus does"],
    )

    with patch("cag.knowledge.document_map.build_llm_document_profile", return_value=profile):
        result, generator = build_document_profile("nexus.txt", "Nexus document")

    assert result.summary == "Nexus is a notification platform."
    assert generator == "llm"


def test_build_document_profile_falls_back_when_llm_fails():
    with patch("cag.knowledge.document_map.build_llm_document_profile", side_effect=RuntimeError("offline")):
        result, generator = build_document_profile(
            "nexus_handbook.txt",
            "Nexus Technologies Inc. is a technology company focused on real-time communication.",
        )

    assert "Nexus Technologies" in result.summary
    assert generator == "local_fallback"


def test_contextual_headers_preserve_original_content_for_display():
    chunk = Document(
        page_content="API keys are generated in Settings.",
        metadata={
            "filename": "api.md",
            "domain_module": "api",
            "chunk_index": 0,
            "total_chunks": 2,
            "document_summary": "API authentication reference.",
            "document_topics": ["api", "authentication"],
        },
    )

    enriched = add_contextual_headers([chunk])[0]

    assert "Document: api.md" in enriched.page_content
    assert "Topics: api, authentication" in enriched.page_content
    assert enriched.metadata["original_content"] == "API keys are generated in Settings."


def test_hype_lite_questions_are_stored_as_covered_intents(tmp_path):
    source = tmp_path / "api_reference.md"
    source.write_text("API keys are generated from Settings > Integrations.", encoding="utf-8")
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "api", "chunk_index": 0},
    )

    with patch("cag.knowledge.compiler.build_document_profile") as mock_profile:
        mock_profile.return_value = (
            DocumentProfileOutput(
                summary="API authentication reference.",
                keywords=["api keys", "credentials"],
                topics=["api", "authentication"],
                entities=["API Keys"],
                covered_intents=["Questions about API authentication"],
            ),
            "llm",
        )
        compile_chunks([chunk], tmp_path / "knowledge.db")

    profiles = list_document_profiles(db_path=tmp_path / "knowledge.db")
    intents = profiles[0]["covered_intents"]

    assert "What does api reference say about api?" in intents
    assert "What does the document say about API Keys?" in intents
    assert chunk.metadata["document_summary"] == "API authentication reference."


def test_build_hypothetical_questions_is_deterministic():
    profile = DocumentProfileOutput(
        summary="API authentication reference.",
        keywords=["api keys"],
        topics=["api"],
        entities=["API Keys"],
        covered_intents=["Questions about API authentication"],
    )

    questions = build_hypothetical_questions("api_reference.md", profile)

    assert questions[:3] == [
        "What is covered in api reference?",
        "Summarize api reference.",
        "What does api reference say about api?",
    ]


def test_document_profile_search_finds_file_before_chunks(tmp_path):
    handbook = tmp_path / "nexus_employee_handbook.txt"
    handbook.write_text(
        "Nexus Technologies Inc. is a technology company focused on real-time communication. "
        "Mission Statement: Building infrastructure that helps teams respond faster and smarter.",
        encoding="utf-8",
    )
    api = tmp_path / "nexus_api_reference.txt"
    api.write_text("API keys are generated in the Nexus dashboard.", encoding="utf-8")
    chunks = [
        Document(
            page_content=handbook.read_text(encoding="utf-8"),
            metadata={"source": str(handbook), "filename": handbook.name, "domain_module": "nexus", "chunk_index": 0},
        ),
        Document(
            page_content=api.read_text(encoding="utf-8"),
            metadata={"source": str(api), "filename": api.name, "domain_module": "api", "chunk_index": 0},
        ),
    ]

    with patch("cag.knowledge.compiler.build_document_profile") as mock_profile:
        mock_profile.side_effect = [
            (
                DocumentProfileOutput(
                    summary="Company overview and mission for Nexus Technologies.",
                    keywords=["company", "overview", "mission", "communication"],
                    topics=["overview"],
                    entities=["Nexus Technologies"],
                    covered_intents=["Questions about what Nexus does"],
                ),
                "llm",
            ),
            (
                DocumentProfileOutput(
                    summary="API authentication reference.",
                    keywords=["api", "authentication", "keys"],
                    topics=["api"],
                    entities=["Nexus API"],
                    covered_intents=["Questions about API keys"],
                ),
                "llm",
            ),
        ]
        compile_chunks(chunks, tmp_path / "knowledge.db")

    candidates = search_document_profiles("Ciao di cosa vi occupate?", db_path=tmp_path / "knowledge.db")
    docs = search_chunks_for_document_candidates(
        "Ciao di cosa vi occupate?",
        candidates,
        db_path=tmp_path / "knowledge.db",
    )

    assert candidates[0]["filename"] == "nexus_employee_handbook.txt"
    assert all(doc.metadata["filename"] == "nexus_employee_handbook.txt" for doc in docs[:1])


def test_document_chunk_search_can_include_procedural_neighbors(tmp_path):
    source = tmp_path / "token_runbook.md"
    source.write_text("placeholder", encoding="utf-8")
    chunks = [
        Document(
            page_content="Open Account Settings and choose Integrations.",
            metadata={"source": str(source), "filename": source.name, "domain_module": "runbook", "chunk_index": 0},
        ),
        Document(
            page_content="Rotate the API token from the credentials panel.",
            metadata={"source": str(source), "filename": source.name, "domain_module": "runbook", "chunk_index": 1},
        ),
        Document(
            page_content="Save the new token and update dependent services.",
            metadata={"source": str(source), "filename": source.name, "domain_module": "runbook", "chunk_index": 2},
        ),
    ]
    compile_chunks(chunks, tmp_path / "knowledge.db")
    connection = connect(tmp_path / "knowledge.db")
    try:
        source_version_id = rows(connection, "SELECT id FROM source_versions")[0]["id"]
    finally:
        connection.close()

    candidate = {
        "profile_id": "profile-1",
        "source_version_id": source_version_id,
        "filename": source.name,
        "source": str(source),
        "score": 1.0,
        "match_reason": "Matched runbook",
        "generator": "test",
    }

    docs = search_chunks_for_document_candidates(
        "How do I rotate the API token?",
        [candidate],
        k=1,
        include_neighbors=True,
        db_path=tmp_path / "knowledge.db",
    )

    assert [doc.metadata["chunk_index"] for doc in docs] == [0, 1, 2]
    assert docs[0].metadata["procedural_neighbor"] is True
    assert docs[1].metadata["procedural_neighbor"] is False
    assert docs[2].metadata["procedural_anchor_index"] == 1


def test_document_profile_search_matches_gestionale_to_product_profiles(tmp_path):
    source = tmp_path / "teamsystem_products.txt"
    source.write_text(
        "Studio 81 offers TeamSystem products including Alyante, ACG Enterprise, and HR All-In-One.",
        encoding="utf-8",
    )
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "products", "chunk_index": 0},
    )

    with patch("cag.knowledge.compiler.build_document_profile") as mock_profile:
        mock_profile.return_value = (
            DocumentProfileOutput(
                summary="Overview of TeamSystem management software and ERP products.",
                keywords=["TeamSystem", "gestionali Alyante", "gestionali ACG", "ERP", "HR All-In-One"],
                topics=["product", "services"],
                entities=["TeamSystem", "Alyante", "ACG Enterprise", "HR All-In-One"],
                covered_intents=["Questions about available management software products"],
            ),
            "llm",
        )
        compile_chunks([chunk], tmp_path / "knowledge.db")

    candidates = search_document_profiles("voi avete qualche gestionale?", db_path=tmp_path / "knowledge.db")

    assert candidates
    assert candidates[0]["filename"] == "teamsystem_products.txt"


def test_document_profile_search_matches_typoed_application_development_query(tmp_path):
    source = tmp_path / "software_services.txt"
    source.write_text(
        "Studio 81 develops custom software applications, CRM integrations, websites, "
        "data integration pipelines, and digital marketing solutions.",
        encoding="utf-8",
    )
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "services", "chunk_index": 0},
    )

    with patch("cag.knowledge.compiler.build_document_profile") as mock_profile:
        mock_profile.return_value = (
            DocumentProfileOutput(
                summary="Software development, applications, CRM, websites, and data integration services.",
                keywords=["software", "applicazioni", "sviluppo software", "CRM", "siti web"],
                topics=["services", "software development"],
                entities=["Studio 81"],
                covered_intents=["Questions about custom application development"],
            ),
            "llm",
        )
        compile_chunks([chunk], tmp_path / "knowledge.db")

    candidates = search_document_profiles("svilupppate applicazioni?", db_path=tmp_path / "knowledge.db")

    assert candidates
    assert candidates[0]["filename"] == "software_services.txt"


def test_list_document_profiles_returns_dashboard_shape(tmp_path):
    source = tmp_path / "profile.md"
    source.write_text("Studio 81 provides technology services and product support.", encoding="utf-8")
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "profile", "chunk_index": 0},
    )

    compile_chunks([chunk], tmp_path / "knowledge.db")

    profiles = list_document_profiles(db_path=tmp_path / "knowledge.db")

    assert len(profiles) == 1
    assert profiles[0]["filename"] == "profile.md"
    assert profiles[0]["chunk_count"] == 1
    assert profiles[0]["summary"]
    assert isinstance(profiles[0]["keywords"], list)


def test_compiled_search_returns_claim_documents_with_provenance(tmp_path):
    source = tmp_path / "runbook.md"
    source.write_text("Checkout latency is high when p95 exceeds 1200 ms.", encoding="utf-8")
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "runbook", "chunk_index": 0},
    )
    compile_chunks([chunk], tmp_path / "knowledge.db")

    results = compiled_search("checkout p95 latency", k=3, db_path=tmp_path / "knowledge.db")

    assert len(results) == 1
    assert "Checkout latency" in results[0].page_content
    assert results[0].metadata["filename"] == "runbook.md"
    assert results[0].metadata["compiled_knowledge"] is True
    assert results[0].metadata["claim_id"]
    assert results[0].metadata["chunk_id"]


def test_compiled_search_matches_concept_aliases(tmp_path):
    source = tmp_path / "api.md"
    source.write_text("API keys are generated from Settings > Integrations.", encoding="utf-8")
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "api", "chunk_index": 0},
    )
    compile_chunks([chunk], tmp_path / "knowledge.db")

    results = compiled_search(
        "How can I mint a fresh dashboard credential for programmatic access?",
        k=3,
        db_path=tmp_path / "knowledge.db",
    )

    assert len(results) == 1
    assert "API keys" in results[0].page_content


def test_compile_chunks_populates_knowledge_graph(tmp_path):
    source = tmp_path / "nexus.md"
    source.write_text(
        "NexusFlow helps Support Teams triage incidents. "
        "NexusFlow coordinates customer communication during outages.",
        encoding="utf-8",
    )
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "runbook", "chunk_index": 0},
    )
    compile_chunks([chunk], tmp_path / "knowledge.db")

    connection = connect(tmp_path / "knowledge.db")
    try:
        graph = list_knowledge_graph(connection)
    finally:
        connection.close()

    node_types = {node["type"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}
    assert {"source", "claim", "topic", "entity"} <= node_types
    assert "contains_claim" in relations
    assert "mentions" in relations


def test_extract_claims_filters_short_fragments():
    assert extract_claims("Short. This longer sentence should become a claim.") == [
        "This longer sentence should become a claim."
    ]


def test_lint_detects_missing_evidence_stale_claims_and_orphan_topics(tmp_path):
    db_path = tmp_path / "knowledge.db"
    connection = connect(db_path)
    try:
        initialize(connection)
        connection.execute(
            """
            INSERT INTO claims(id, claim_text, claim_type, confidence, status)
            VALUES ('claim-missing', 'A claim without evidence.', 'general', 0.5, 'active')
            """
        )
        connection.execute(
            """
            INSERT INTO claims(id, claim_text, claim_type, confidence, status)
            VALUES ('claim-stale', 'A stale claim with no evidence.', 'general', 0.5, 'stale')
            """
        )
        store_topic(connection, slug="orphan", title="Orphan Topic")
        connection.commit()
    finally:
        connection.close()

    report = lint_knowledge(db_path)

    assert "claim-missing" in report.missing_evidence_claims
    assert "claim-stale" in report.stale_claims
    assert len(report.orphan_topics) == 1
    assert report.issue_count >= 3


def test_lint_records_simple_contradictions(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("API tokens can be rotated by admins.", encoding="utf-8")
    chunk = Document(
        page_content=source.read_text(encoding="utf-8"),
        metadata={"source": str(source), "filename": source.name, "domain_module": "policy", "chunk_index": 0},
    )
    compile_chunks([chunk], tmp_path / "knowledge.db")

    connection = connect(tmp_path / "knowledge.db")
    try:
        chunk_id = rows(connection, "SELECT id FROM chunks LIMIT 1")[0]["id"]
        store_claim(
            connection,
            claim_text="API tokens can be rotated by admins.",
            chunk_id=chunk_id,
            claim_type="policy",
        )
        store_claim(
            connection,
            claim_text="API tokens cannot be rotated by admins.",
            chunk_id=chunk_id,
            claim_type="policy",
        )
        connection.commit()
    finally:
        connection.close()

    report = lint_knowledge(tmp_path / "knowledge.db")

    assert len(report.contradictions) == 1
    connection = connect(tmp_path / "knowledge.db")
    try:
        assert rows(connection, "SELECT COUNT(*) AS count FROM contradictions")[0]["count"] == 1
        assert rows(connection, "SELECT COUNT(*) AS count FROM knowledge_log WHERE event_type = 'lint'")[0]["count"] == 1
    finally:
        connection.close()
