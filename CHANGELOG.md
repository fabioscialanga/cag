# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-04-12

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

The previous name REFINE was architecturally misleading. The node does not refine retrieved chunks in a simple sense — it scores them, clusters them, assigns semantic categories, and reorders them using a diversity-aware algorithm to construct the optimal context set for the reasoning agent. SELECT_CONTEXT reflects this responsibility accurately and removes the ambiguity noted in the README itself (*"this selection logic currently lives inside the REFINE stage, but conceptually it is a SELECT_CONTEXT step"*).

### Upgrade notes

If you import `refine_node` or `route_after_refine` directly, rename them to `select_context_node` and `route_after_select_context`.

If you parse `node_trace` output (e.g. in benchmark scripts or log analysis), update `"REFINE"` to `"SELECT_CONTEXT"`.

Existing benchmark artifacts generated under v0.1.x will contain `"REFINE"` in their `node_trace` fields. These are historical and remain valid for comparison, but will not match v0.2 artifacts directly on the trace field.

---

## [0.1.0] — 2026-04-05

### Added

- Initial preview release.
- Graph-based CAG runtime: `ENTRY → RETRIEVE → REFINE → REASON → VALIDATE → EXIT`.
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
