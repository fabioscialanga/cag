# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

- Added discriminative lexical candidate scoring after multi-query retrieval, with vector-order preservation as a secondary signal.
- Added per-request `retrieval_top_k` runtime control so evaluation `--top-k` reaches the CAG graph retrieval path.
- Added deterministic empty-retrieval handling in the retrieval agent to avoid unnecessary LLM calls when no chunks were recovered.
- Added `MODERATE_RELEVANCE_THRESHOLD` to `.env.example`.
- Added `ToDo.md` as the working roadmap for release prep, adoption work, retrieval improvements, and the DB-first Knowledge Compiler direction.
- Added Quickstart, API, configuration, evaluation, examples, release, and Knowledge Compiler docs.
- Added community guidance for GitHub Discussions, issues, and showcase posts.
- Added example corpora for basic CLI, customer support, policy QA, and incident runbook flows.
- Added `cag demo` as a single-command local demo path over the bundled benchmark corpus.
- Added Dockerfile and Docker Compose preview setup.
- Added a first SQLite-backed Knowledge Compiler slice with schema migrations, deterministic claim extraction, provenance links, and compiled claim search.
- Added typed FastAPI response models for query, upload, file listing, and delete responses.
- Added API tests for listing and deleting uploaded documents.
- Added `/demo/reset` to load the bundled benchmark corpus into `data/raw/` and optionally trigger ingestion.
- Added `/diagnostics/retrieval` to inspect retrieval and context selection without answer generation.
- Added `cag_compiled` evaluation system using compiled-knowledge search with raw retrieval fallback.
- Added `compiled_only` and `compiled_plus_raw` evaluation systems to isolate compiled-knowledge coverage and hybrid retrieval behavior.
- Added explicit alias/synonym benchmark questions and tests that preserve unsupported-question coverage across query types.
- Added `context_recall_score` as a retrieval-specific benchmark metric for gold-source coverage in selected context.
- Added a scheduled/manual benchmark workflow for repeated-run CAG vs RAG comparison when `OPENAI_API_KEY` is configured.
- Evaluation outputs now include `compiled_chunk_count` and retrieval preserves compiled-knowledge provenance metadata.
- Added Knowledge Compiler lint checks for stale claims, missing evidence, orphan topics, and simple contradiction detection.
- Frontend demo now exposes retrieved chunks, selected context, evidence gaps, and compiled-knowledge markers in an Evidence Workbench panel.
- Added practical Known Limits documentation for benchmark and compiled-knowledge evaluation paths.

### Changed

- Normalized lowercase `LOG_LEVEL` values such as `info` to the configured uppercase logging level.
- `/upload` responses now return saved filenames instead of absolute local filesystem paths.
- Frontend-serving API tests no longer depend on a prebuilt `frontend/dist` directory.
- Cleaned stale package docstrings that still referenced `v0.1`.
- Normalized README and changelog dash/arrow characters to ASCII-safe rendering.
- Moved lexical retrieval scoring helpers into `cag.retrieval.lexical`.
- Excluded research experiments, generated artifacts, and local Chroma data from package manifests.
- CI now runs API tests, uses `npm ci`, and checks Python package builds.

### Fixed

- `cag eval` now rejects non-positive `--top-k`, `--limit`, and `--runs` values.
- `/query` now rejects empty or whitespace-only query strings.
- `langdetect` is seeded for deterministic language detection in tests and runtime.

---

## [0.2.0] - 2026-04-12

### Changed

- **SELECT_CONTEXT replaces REFINE throughout the codebase and documentation.**
  The graph node previously named `refine_node` has been renamed to `select_context_node`.
  The routing function `route_after_refine` is now `route_after_select_context`.
  The LangGraph node key changes from `"refine"` to `"select_context"`.
  The `node_trace` field in query outputs now records `"SELECT_CONTEXT"` instead of `"REFINE"`.
  This is a **breaking change** for any consumer that inspects `node_trace` strings or imports `refine_node` or `route_after_refine` directly.

- **`eval/systems.py` updated.** The `_run_cag_no_selection_query` function now calls `select_context_node` and `route_after_select_context` instead of the old names.

- **Documentation updated.**
  - `docs/cag-architecture.md`: all diagrams, tables, routing descriptions, and worked examples now use `SELECT_CONTEXT`.
  - `README.md`: rewritten in full to reflect the updated architecture, with a new "What CAG Does" section, a dedicated "The SELECT_CONTEXT Step in Detail" section, a "What changed in 0.2" entry, and an updated CAG vs RAG comparison table.

- **Version bumped** in `pyproject.toml`, `src/cag/__init__.py`, `src/cag/graph/__init__.py`, and `src/cag/ui/app.py`.

### Why this change

The previous name REFINE was architecturally misleading. The node does not refine retrieved chunks in a simple sense -- it scores them, clusters them, assigns semantic categories, and reorders them using a diversity-aware algorithm to construct the optimal context set for the reasoning agent. SELECT_CONTEXT reflects this responsibility accurately and removes the ambiguity noted in the README itself (*"this selection logic currently lives inside the REFINE stage, but conceptually it is a SELECT_CONTEXT step"*).

### Upgrade notes

If you import `refine_node` or `route_after_refine` directly, rename them to `select_context_node` and `route_after_select_context`.

If you parse `node_trace` output (e.g. in benchmark scripts or log analysis), update `"REFINE"` to `"SELECT_CONTEXT"`.

Existing benchmark artifacts generated under v0.1.x will contain `"REFINE"` in their `node_trace` fields. These are historical and remain valid for comparison, but will not match v0.2 artifacts directly on the trace field.

---

## [0.1.0] - 2026-04-05

### Added

- Initial preview release.
- Graph-based CAG runtime: `ENTRY -> RETRIEVE -> REFINE -> REASON -> VALIDATE -> EXIT`.
- Query classification (GENERAL, PROCEDURAL, DIAGNOSTIC, CONFIGURATION).
- Strategy-aware retrieval (semantic, hierarchical, multi_evidence) with multi-variant queries.
- Context selection with diversity-aware chunk reordering (`_reorder_for_context_selection`).
- Automatic language detection (55+ languages via `langdetect`) with localized error messages.
- Answer validation with confidence and hallucination risk thresholds.
- Structured escalation in place of low-confidence generation.
- Retry loop (up to `max_reason_retries`) on high hallucination risk.
- Per-request runtime threshold overrides via `RuntimeConfig`.
- Benchmark harness with `cag`, `cag_no_selection`, `rag_baseline`, `direct_baseline`, and `lightrag_baseline` systems.
- `context_precision_score`, `retrieved_chunk_count`, and `selected_chunk_count` metrics.
- `cag eval-audit` for benchmark dataset coverage checks.
- `cag compare` for structured run comparison.
- React + FastAPI preview path.
- Streamlit development UI.
- CLI: `cag ingest`, `cag query`, `cag eval`, `cag eval-audit`, `cag compare`.
- Preview API protection via `CAG_API_KEY` / `X-API-Key`.
- Upload validation (extension, size) on `/upload`.
- `fallback_used` and `fallback_reason` fields in query and benchmark output.
