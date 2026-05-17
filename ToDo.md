# CAG ToDo

This file is the working roadmap for turning CAG from a promising preview into a repo that many developers can clone, trust, run, and integrate.

Rule for checkboxes:

- `[x]` means implemented and verified with the test or check listed on the same item.
- `[ ]` means not done yet.
- `[~]` means started, but not fully implemented or not fully tested.

## Completed And Verified

- [x] Rename the core evidence stage from vague refinement language to explicit `SELECT_CONTEXT`.
  Verified by existing graph, README, architecture docs, and pipeline tests.

- [x] Add diversity-aware context selection over retrieved chunks.
  Verified by `tests/test_cag_pipeline.py`.

- [x] Add `cag_no_selection` ablation baseline to isolate the value of context selection.
  Verified by `tests/test_eval_harness.py`.

- [x] Add dataset audit command for benchmark readiness.
  Verified by `tests/test_eval_harness.py`.

- [x] Add per-request runtime thresholds for relevance, confidence, and hallucination risk.
  Verified by graph/eval tests.

- [x] Harden preview API upload/query path with API key support and upload validation.
  Verified by `tests/test_api.py`.

- [x] Add discriminative lexical candidate scoring after multi-query retrieval.
  Verified by:

  ```powershell
  .venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py -q
  ```

  Result: `44 passed`.

- [x] Add provider-failure resilience for local retrieval and evidence-only fallback.
  Verified by `.venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_api.py tests\test_eval_harness.py -q` (`105 passed`), `npm.cmd run build`, and HTTP smoke query `POST http://127.0.0.1:8000/query` with `Ciao di cosa vi occupate?` returning cited Nexus evidence instead of escalation.

- [x] Add Document Map before chunk retrieval.
  Verified by `document_profiles` schema, LLM-first/local-fallback profile compiler, document-first graph retrieval, API/frontend diagnostics, `.venv\Scripts\python.exe -m pytest -q` (`119 passed`), and `npm.cmd run build`.

- [x] Add procedural neighbor retrieval for document-map searches.
  Verified by adjacent chunk expansion in `search_chunks_for_document_candidates`, procedural `RETRIEVE` routing, and `.venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_knowledge_compiler.py tests\test_eval_harness.py -q` (`121 passed`).

- [x] Add graph-level conversational query contextualization before retrieval.
  Verified by `CONTEXTUALIZE_QUERY` graph node, vague follow-up rewrite tests, API/pipeline/eval suite, and `.venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_eval_harness.py tests\test_api.py -q` (`133 passed`).

- [x] Add fast deterministic context selection for high-confidence evidence.
  Verified by RetrievalAgent fast-path tests, runtime config entries, and `.venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_eval_harness.py tests\test_api.py -q` (`134 passed`).

- [x] Add typo-tolerant application/software development retrieval aliases.
  Verified by `svilupppate applicazioni?` lexical/document-profile tests and `.venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_knowledge_compiler.py tests\test_eval_harness.py tests\test_api.py -q` (`156 passed`).

- [x] Apply quality-oriented retrieval/orchestration defaults.
  Includes `text-embedding-3-large`/3072 dimensions, enabled Knowledge Compiler, `RETRIEVAL_TOP_K=20`, smaller configurable chunks (`1200`/`180`), dynamic selected-context budgets (`8` general, `10` complex), RetrievalAgent and ReasoningAgent few-shot guidance, semantic context ordering, and adaptive retrieval retry.
  Verified by `.venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_knowledge_compiler.py tests\test_eval_harness.py tests\test_api.py -q` (`158 passed`).

- [x] Add Document Intelligence Dashboard.
  Verified by `/document-profiles`, frontend document profile cards, `.venv\Scripts\python.exe -m pytest -q` (`121 passed`), and `npm.cmd run build`.

## Immediate Release Prep

- [x] Fix eval `--top-k` so it controls retrieval inside the CAG graph, not only the outer benchmark runner.
  Verified by `test_run_query_uses_injected_search_function` and `test_cag_eval_runner_passes_top_k_to_runtime_config`.

- [x] Short-circuit retrieval-agent ranking when no chunks are retrieved.
  Verified by `test_retrieval_agent_short_circuits_empty_chunks`.

- [x] Validate eval CLI numeric arguments:
  - [x] reject or define `--runs 0`
  - [x] reject or define `--limit 0`
  - [x] reject or define `--top-k 0`
  Verified by `test_eval_run_parser_rejects_non_positive_numbers`.

- [x] Harden `/query` input validation for empty/whitespace queries.
  Verified by `test_query_rejects_blank_query`.

- [x] Preserve original vector-search rank as a secondary signal in discriminative candidate reranking.
  Verified by `test_deduped_results_preserve_vector_order_as_secondary_signal`.

- [x] Re-run the focused backend test suite after the current retrieval changes:

  ```powershell
  .venv\Scripts\python.exe -m pytest tests\test_cag_pipeline.py tests\test_eval_harness.py tests\test_api.py -q
  ```

  Result: `81 passed`.

- [ ] Run a smoke benchmark with judge off:

  ```powershell
  .venv\Scripts\python.exe -m cag.eval.run --system cag --limit 5 --judge-mode off
  ```

  Attempted locally on 2026-05-10. Blocked by SSL certificate verification while `tiktoken`
  tried to download `cl100k_base.tiktoken` from `openaipublic.blob.core.windows.net`, including
  after retrying outside the sandbox.

- [x] Update `CHANGELOG.md` with the retrieval discriminative scoring change.
  Verified by `[Unreleased]` changelog entries and focused backend tests.

- [ ] Commit the current retrieval improvement and this roadmap.

- [ ] Push the improved version to GitHub.

- [ ] Create a GitHub release for the current usable version.

## GitHub Adoption Work

- [x] Rewrite the top of `README.md` around an adoption-first promise:
  "A safer RAG pipeline that selects evidence, validates answers, and refuses when docs are insufficient."
  Verified by README opening promise, `Start Here`, and `5 Minute Quickstart`.

- [x] Move deeper research/benchmark narrative lower in `README.md` so the first screen is quickstart and value.
  Verified by README opening quickstart before architecture and benchmark narrative.

- [x] Add a "5 minute quickstart" using the bundled benchmark corpus, not an empty `data/raw` folder.
  Verified by `docs/quickstart.md` and README `Start Here`.

- [x] Add expected output snippets for:
  - [x] `cag ingest`
  - [x] `cag query`
  - [x] `cag eval --limit 3`
  Verified by `docs/quickstart.md`.

- [x] Add a minimal Python API example:

  ```python
  from cag.graph.graph import run_query

  result = run_query("What is the minimum RAM required?")
  print(result["answer"])
  ```
  Verified by `docs/api.md`.

- [x] Add `examples/` with at least:
  - [x] `examples/basic_cli/`
  - [x] `examples/customer_support/`
  - [x] `examples/policy_qa/`
  - [x] `examples/incident_runbook/`
  Verified by `rg --files examples`.

- [x] Add `docs/quickstart.md`.

- [x] Add `docs/examples.md`.

- [x] Add `docs/api.md`.

- [x] Add `docs/evaluation.md`.

- [x] Add `docs/configuration.md`.

- [x] Add issue templates or public issue seeds for roadmap items.
  Verified by `.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`, and `config.yml`.

- [x] Enable or document GitHub Discussions for questions and showcase posts.
  Verified by `docs/community.md` and README link.

## Developer Experience

- [x] Add Dockerfile for API usage.
  Verified by `Dockerfile`.

- [x] Add Docker Compose for API plus frontend.
  Verified by `docker-compose.yml`.

- [x] Add `.env.example` entries for all required local modes.
  Verified by `.env.example` entries for providers, vector DB, API key, thresholds, logging, and Knowledge Compiler settings.

- [x] Add `MODERATE_RELEVANCE_THRESHOLD` to `.env.example`.
  Verified by `test_settings_accept_lowercase_log_level_and_moderate_threshold`.

- [x] Make lowercase `LOG_LEVEL=info` either accepted or clearly documented as invalid.
  Verified by `test_settings_accept_lowercase_log_level_and_moderate_threshold`.

- [x] Add a single command local demo path.
  Verified by `test_cag_cli_demo_defaults_to_bundled_corpus`, `test_cag_cli_demo_runs_ingest_and_query`, and quickstart docs.

- [x] Add clearer install modes in README/docs:
  - [x] local editable install
  - [x] API mode
  - [x] frontend mode
  - [x] evaluation mode
  Verified by `docs/configuration.md`.

- [x] Add a public package plan:
  - [x] confirm package name availability
  - [x] decide first PyPI release version
  - [x] document release command
  Verified by `docs/release.md`; PyPI name `cag` is already occupied, so the package plan uses a future distinct distribution name.

## Retrieval Improvements

- [x] Add candidate-level discriminative lexical scoring after vector retrieval.
  Verified by `test_deduped_results_prefer_discriminative_terms`.

- [x] Move lexical scoring helpers into a dedicated retrieval module if they grow beyond graph-node utility scope.
  Verified by `src/cag/retrieval/lexical.py` and focused pipeline/eval tests.

- [ ] Persist corpus-level document frequency statistics during ingestion.

- [ ] Add optional BM25/hybrid lexical retriever over local chunks.

- [ ] Add query expansion terms validated against corpus statistics.

- [ ] Add benchmark comparison for:
  - [ ] current vector retrieval
  - [ ] discriminative candidate scoring
  - [ ] hybrid lexical/vector retrieval

- [ ] Record retrieval diagnostics in outputs:
  - [ ] query variants
  - [ ] selected lexical terms
  - [ ] retrieval score explanation

## DB-First Knowledge Compiler

Goal: adapt Karpathy's "LLM Wiki" idea to CAG as a database-backed compiled knowledge layer rather than markdown files.

- [x] Design schema for immutable sources:
  - [x] `sources`
  - [x] `source_versions`
  - [x] `chunks`
  Verified by `docs/knowledge-compiler.md`.

- [x] Design schema for compiled knowledge:
  - [x] `claims`
  - [x] `claim_evidence`
  - [x] `entities`
  - [x] `topics`
  - [x] `topic_claims`
  - [x] `contradictions`
  - [x] `knowledge_log`
  - [x] `saved_answers`
  Verified by `docs/knowledge-compiler.md`.

- [x] Choose first DB target:
  - [x] SQLite for local-first quickstart
  - [x] Postgres as production-compatible follow-up
  Verified by `docs/knowledge-compiler.md`.

- [x] Add migration mechanism.
  Verified by `test_knowledge_schema_migration_is_idempotent`.

- [x] Add claim extraction during ingest.
  Verified by optional `ENABLE_KNOWLEDGE_COMPILER` ingest integration and `test_compile_chunks_stores_sources_claims_and_evidence`.

- [x] Add provenance-preserving citation from claim back to chunk/source.
  Verified by `test_compiled_search_returns_claim_documents_with_provenance`.

- [x] Add query path that searches compiled claims/topics before raw chunks.
  Verified by `test_cag_compiled_uses_compiled_search_before_raw_search`.

- [x] Add fallback path from compiled knowledge to raw chunk retrieval.
  Verified by `cag_compiled` hybrid search implementation.

- [x] Add lint command:
  - [x] stale claims
  - [x] contradictions
  - [x] orphan topics
  - [x] missing evidence
  Verified by `lint_knowledge()` and `tests/test_knowledge_compiler.py`.

- [~] Add tests for:
  - [x] source hash/version handling
  - [x] claim insertion
  - [x] evidence provenance
  - [x] contradiction recording
  - [x] compiled-knowledge retrieval
  Verified by `tests/test_knowledge_compiler.py`.

## API And Product Surface

- [~] Return evidence diagnostics from `/query`.
  Started: `/diagnostics/retrieval` exposes retrieval/context diagnostics without generation. `/query` response still needs a tighter public evidence field.

- [x] Return selected context and gaps in a stable response shape.
  Verified by `QueryResponse` fields and `/diagnostics/retrieval` response model.

- [x] Add endpoint for benchmark/demo corpus reset.
  Verified by `test_demo_reset_replaces_raw_documents_and_schedules_ingest` and `test_demo_reset_can_skip_ingest`.

- [x] Add endpoint for listing uploaded documents.
  Verified by `test_files_endpoint_lists_uploaded_documents`.

- [x] Add endpoint for deleting a document safely.
  Verified by `test_delete_file_endpoint_removes_document_and_schedules_reindex` and `test_delete_file_endpoint_rejects_missing_document`.

- [x] Add endpoint for retrieval diagnostics without generation.
  Verified by `test_retrieval_diagnostics_returns_selected_context_shape`.

- [x] Add typed response models for public API docs.
  Verified by FastAPI `response_model` declarations and `tests/test_api.py`.

- [x] Avoid returning absolute local filesystem paths from `/upload` responses.
  Verified by `test_upload_response_returns_filenames_not_absolute_paths`.

## Frontend Demo

- [x] Make the first screen a usable document QA workspace, not a preview shell.
  Verified by frontend Evidence Workbench, upload/file management, chat, and `npm run build`.

- [x] Show retrieved chunks vs selected chunks.
  Verified by `EvidencePanel` and `npm run build`.

- [x] Show gaps and escalation reason.
  Verified by `EvidencePanel`, response status, and `npm run build`.

- [x] Show confidence/hallucination risk in a non-alarming but inspectable way.
  Verified by `IntelligencePanel`, chat meta strip, and `npm run build`.

- [x] Add upload state, ingest state, and query state.
  Verified by upload panel, mini-stats, sending state, and `npm run build`.

- [x] Add demo corpus loader.
  Verified by `/demo/reset` API tests and `npm.cmd run build`.

- [ ] Verify desktop and mobile layout.
  Started with `npm.cmd run build`; full viewport screenshot verification still needs a browser automation tool or manual browser pass.

## Evaluation And Trust

- [x] Add a retrieval-specific benchmark metric for discriminative retrieval.
  Verified by `context_recall_score` in eval scoring, aggregation, comparison output, and `test_context_recall_scores_multi_source_retrieval_coverage`.

- [x] Add unsupported-question tests where refusal is the correct behavior.
  Verified by `test_default_benchmark_preserves_unsupported_question_coverage`.

- [x] Add synonym/alias benchmark questions where the wording differs from the document.
  Verified by `test_default_benchmark_includes_synonym_alias_questions`.

- [x] Add repeated-run comparison in CI or nightly workflow.
  Verified by `.github/workflows/benchmark-nightly.yml` and `docs/evaluation.md`.

- [x] Add benchmark artifact examples to docs.
  Verified by `docs/evaluation.md` and `docs/quickstart.md`.

- [x] Add a "Known limits" section that is frank and practical.
  Verified by `docs/evaluation.md` and focused eval harness tests.

- [x] Add `cag_compiled`, `compiled_only`, and `compiled_plus_raw` eval systems once DB-first compiled knowledge exists.
  - [x] `cag_compiled`
  - [x] `compiled_only`
  - [x] `compiled_plus_raw`
  Verified by `test_cag_cli_eval_supports_cag_compiled`, `test_run_system_dispatches_cag_compiled`, `test_cag_compiled_uses_compiled_search_before_raw_search`, `test_compiled_only_does_not_call_raw_search`, and `test_compiled_plus_raw_merges_compiled_and_raw_results`.

- [ ] Add metrics for:
  - [x] selected compiled-section precision
    Initial support via `compiled_chunk_count`; full precision scoring remains future work.
  - [~] claim support precision
    Started via claim/chunk provenance and `compiled_chunk_count`; precision scoring remains future work.
  - [~] contradiction detection rate
    Started with deterministic lint contradiction detection; benchmark-rate metric remains future work.
  - [~] stale answer avoidance
    Started with stale claim linting; graph-level stale answer avoidance remains future work.
  - [ ] latency and token reduction vs raw retrieval

## Repository Hygiene

- [x] Fix mojibake characters in README where apostrophes/dashes rendered incorrectly.
  Verified by checks for non-ASCII dash/arrow characters and corrupted UTF-8 markers in `README.md` and `CHANGELOG.md` returning no matches.

- [x] Normalize stale `v0.1` package docstrings to current version wording.
  Verified by `rg "v0\.1" -n src tests README.md CHANGELOG.md`; only historical changelog text remains.

- [x] Decide whether `experiments/autoresearch_external/` belongs in the public package or should move to docs/experiments.
  Verified by `MANIFEST.in`: experiments remain in repo but are excluded from public packages.

- [x] Review `.gitignore` and package data before release.
  Verified by `.gitignore`, `MANIFEST.in`, and CI package-build step. Local isolated build is blocked by SSL/package download issues in this environment.

- [x] Ensure generated artifacts are not accidentally committed.
  Verified by `.gitignore` entries for `artifacts/` and `*.log`, plus `git ls-files backend.log artifacts` returning no tracked files.

- [x] Decide whether tracked `backend.log` should be removed from the repo and ignored.
  Verified: `backend.log` is ignored by `*.log` and is not tracked.

- [x] Make frontend-serving tests independent of a prebuilt `frontend/dist`.
  Verified by `test_root_frontend_serves_built_index_and_assets` and `test_root_frontend_redirects_when_build_is_missing`.

## Subagent Review Queue

- [x] DB-first Knowledge Compiler review completed and integrated into this roadmap.

- [x] Open-source adoption/GitHub developer experience review completed and integrated into this roadmap.

- [x] Technical improvement review completed and integrated into this roadmap.
