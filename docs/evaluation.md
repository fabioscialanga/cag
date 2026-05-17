# Evaluation

CAG includes a benchmark harness for comparing the full graph against ablations and baselines.

## Systems

- `cag`: full graph with context selection and validation
- `cag_compiled`: CAG with compiled-knowledge search first and raw retrieval fallback
- `compiled_only`: CAG over compiled claims only, useful for isolating Knowledge Compiler coverage
- `compiled_plus_raw`: CAG over compiled claims plus raw chunks, useful for measuring hybrid retrieval
- `cag_no_selection`: same graph, but preserves raw retrieval order after selection
- `rag_baseline`: one-shot retrieval plus generation
- `direct_baseline`: generation without retrieval context
- `lightrag_baseline`: optional LightRAG comparison

## Smoke Run

```bash
cag eval --system cag --limit 3 --judge-mode off
```

Artifacts are written to:

```text
artifacts/eval_runs/<timestamp>_cag/
  run.json
  results.jsonl
```

## Compare Runs

```bash
cag eval --system rag_baseline --limit 3 --judge-mode off
cag eval --system cag_compiled --limit 3 --judge-mode off
cag eval --system compiled_only --limit 3 --judge-mode off
cag eval --system compiled_plus_raw --limit 3 --judge-mode off
cag eval --system cag --limit 3 --judge-mode off
cag compare --runs ./artifacts/eval_runs/<rag_run> ./artifacts/eval_runs/<cag_run>
```

## Multi-Run Checks

```bash
cag eval --system cag --runs 3 --judge-mode off
cag eval --system rag_baseline --runs 3 --judge-mode off
cag compare --runs ./artifacts/eval_runs/<cag_multi_run> ./artifacts/eval_runs/<rag_multi_run>
```

The repository includes `.github/workflows/benchmark-nightly.yml` for scheduled or manual repeated-run comparison. It runs only when `OPENAI_API_KEY` is configured as a repository secret.

## Dataset Audit

```bash
cag eval-audit --format markdown
```

The audit checks dataset size, query-type balance, duplicate IDs/questions, gold source coverage, and unsupported-item quality.

The default benchmark also includes:

- intentionally unsupported questions across all query types, where escalation is the correct behavior
- explicit alias/synonym questions marked in `notes` so retrieval can be tested when user wording differs from document wording

## Current Claim

The benchmark is intended to make the context-selection claim inspectable:

CAG should select better evidence than raw top-k retrieval, and that should improve task success while reducing unsupported answers.

Retrieval-specific fields to inspect:

- `context_precision_score`: how much of the selected context came from expected gold sources
- `context_recall_score`: how many expected gold sources were represented in the selected context
- `retrieved_chunk_count` and `selected_chunk_count`: how much evidence entered retrieval and how much reached reasoning

## Latest Reproducible Snapshot (May 17, 2026)

Source artifacts:
- `artifacts/eval_runs/20260517T092730Z_rag_baseline/run.json`
- `artifacts/eval_runs/20260517T095338Z_cag_no_selection/run.json`
- `artifacts/eval_runs/20260517T104227Z_cag/run.json`
- `artifacts/eval_comparisons/20260517T114033Z_comparison/comparison.md`

Single-run snapshot (`--runs 1`, `judge-mode off`, 104 questions):

| System | Grounded Answer | Context Precision | Context Recall | Hallucination Rate | Task Success | Avg Latency (ms) | Avg Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rag_baseline` | 0.906 | 0.502 | 1.000 | 0.029 | 0.817 | 14721.881 | 5.319 |
| `cag` | 0.686 | 0.497 | 0.725 | 0.250 | 0.615 | 33176.651 | 11.204 |
| `cag_no_selection` | 0.587 | 0.463 | 0.713 | 0.365 | 0.548 | 27774.001 | 10.137 |

Interpretation:
- In this latest single-run configuration, `rag_baseline` outperforms both `cag` and `cag_no_selection` on primary quality metrics.
- `cag` still improves over `cag_no_selection`, indicating the context-selection stage adds value relative to the no-selection ablation.
- Latency/cost overhead remains significant for CAG orchestration and must be justified by quality gains in target domains.

Important methodology note:
- This is a single-run snapshot and can be sensitive to model/version/runtime variance.
- Treat it as an operational checkpoint, not a definitive final claim.
- For release claims, run repeated evaluations (e.g., `--runs 3` or `--runs 5`) and compare aggregated metrics.

## Known Limits

- The default benchmark corpus is useful for smoke tests and regression checks, but it is not broad enough to prove domain-general superiority.
- `judge-mode off` avoids external LLM judging, so it only scores deterministic coverage, grounding, escalation, and context metrics.
- `cost_estimate` is a relative token-style estimate for comparing runs, not a billing source of truth.
- `compiled_only` is expected to fail questions whose facts were not extracted into claims; use it to measure compiler coverage, not as the default product path.
- `compiled_plus_raw` and `cag_compiled` still depend on raw retrieval quality when compiled claims are sparse.
- LightRAG comparison requires its optional dependencies and provider credentials, so it is not part of the fastest local validation path.
