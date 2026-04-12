# Benchmark Results: CAG vs RAG Baseline

This file documents the original preview-stage benchmark snapshot.

The evaluation harness has since evolved to support a stronger context-selection claim with:

- `context_precision_score`
- `retrieved_chunk_count` and `selected_chunk_count`
- `cag_no_selection` as an ablation between `cag` and `rag_baseline`

Use the newer triplet comparison flow when you want to measure whether context selection itself improves answer quality:

```powershell
.\run_eval_triplet.ps1
```

Date: 2026-04-08
Model: gpt-4o
Embedding: text-embedding-3-small
Runs per system: 5
Questions: 50

## Methodology

### Corpus

Four synthetic documents in `data/benchmark_corpus/`:

| Document | Content |
| --- | --- |
| nexus_platform_configuration_guide.txt | Platform settings, modules, permissions, deployment |
| nexus_incident_response_runbook.txt | Incident classification, escalation, resolution procedures |
| nexus_employee_handbook.txt | Onboarding, time off, expenses, code of conduct |
| nexus_api_reference.txt | REST endpoints, authentication, rate limits, error codes |

### Dataset

50 questions distributed across four query types, with 8 intentionally unanswerable questions (16%):

| Query Type | Count | Unanswerable |
| --- | ---: | ---: |
| GENERAL | 12 | 2 |
| PROCEDURAL | 13 | 2 |
| DIAGNOSTIC | 13 | 2 |
| CONFIGURATION | 12 | 2 |

Each question has gold answer points and gold sources. Unanswerable questions have no gold points and no gold sources.

### Systems Compared

- **CAG**: The full graph pipeline (ENTRY -> RETRIEVE -> REFINE -> REASON -> VALIDATE -> EXIT) with query typing, evidence refinement, and validation with escalation.
- **RAG Baseline**: Standard retrieve-then-generate. Same embedding, same retrieval top-k, same generator model. No query typing, no evidence refinement, no validation loop.

Both systems use the same embedding model (`text-embedding-3-small`), the same generator (`gpt-4o`), and the same retrieval top-k (10).

### Scoring

Each answer is scored on:

- **Point coverage**: Fraction of gold answer points found in the response (token overlap, stopword-filtered).
- **Source grounding**: Whether cited sources match gold sources.
- **Grounded answer score**: Composite of point coverage and source grounding.
- **Hallucination flag**: Set when the system answers an unanswerable question without escalating.
- **Task success**: Correct behavior for both answerable (good answer) and unanswerable (escalation) questions.

No LLM judge was used. All scoring is deterministic and token-based.

### Statistical Testing

Five independent runs per system. Scores are averaged per question across runs, then compared using:

- **Paired t-test** (parametric): tests whether the mean difference between paired observations is zero.
- **Wilcoxon signed-rank test** (non-parametric): tests the same hypothesis without assuming normality.
- **Cohen's d**: standardized effect size (small: 0.2, medium: 0.5, large: 0.8).

Significance threshold: p < 0.05.

## Results

### Aggregate Metrics

Mean across 5 runs:

| Metric | CAG | RAG Baseline | Difference |
| --- | ---: | ---: | ---: |
| Grounded Answer Score | 0.825 | 0.762 | +0.063 |
| Point Coverage | 0.807 | 0.741 | +0.066 |
| Source Grounding | 0.819 | 0.795 | +0.024 |
| Hallucination Rate | 0.116 | 0.176 | -0.060 |
| Task Success Rate | 0.816 | 0.736 | +0.080 |
| False Escalation Rate | 0.119 | 0.157 | -0.038 |
| Avg Latency (ms) | 11,747 | 3,350 | +8,397 |
| Avg Cost Estimate | 10.11 | 4.71 | +5.39 |

CAG wins on every quality metric. RAG is faster and cheaper by roughly 3.5x.

### Per Query Type

Grounded answer score, mean across 5 runs:

| Query Type | CAG | RAG Baseline | Difference |
| --- | ---: | ---: | ---: |
| GENERAL | 0.948 | 0.886 | +0.062 |
| CONFIGURATION | 0.884 | 0.843 | +0.041 |
| DIAGNOSTIC | 0.813 | 0.719 | +0.094 |
| PROCEDURAL | 0.667 | 0.617 | +0.050 |

Both systems struggle most with PROCEDURAL queries. CAG's strongest advantage is on DIAGNOSTIC questions (+0.094), where evidence refinement and validation appear to help most.

### Hallucination Rate by Query Type

| Query Type | CAG | RAG Baseline |
| --- | ---: | ---: |
| GENERAL | 0.017 | 0.067 |
| CONFIGURATION | 0.050 | 0.083 |
| DIAGNOSTIC | 0.092 | 0.200 |
| PROCEDURAL | 0.292 | 0.338 |

CAG shows lower hallucination rates across all query types. The gap is largest for DIAGNOSTIC queries, where RAG hallucinates at roughly twice the rate of CAG.

### Stability Across Runs

Grounded answer score per run:

| Run | CAG | RAG Baseline |
| --- | ---: | ---: |
| 1 | 0.805 | 0.703 |
| 2 | 0.839 | 0.790 |
| 3 | 0.822 | 0.749 |
| 4 | 0.805 | 0.762 |
| 5 | 0.856 | 0.808 |
| **Std** | **0.021** | **0.035** |

CAG is more stable across runs (std=0.021 vs 0.035).

## Statistical Testing

Scores were averaged per question across 5 runs before testing. 50 paired observations.

| Test | Statistic | p-value | Significant |
| --- | ---: | ---: | --- |
| Paired t-test | t = 2.196 | p = 0.033 | Yes |
| Wilcoxon signed-rank | W = 448.0 | p = 0.064 | No |

- **Mean difference**: 0.062
- **Cohen's d**: 0.31 (small effect)
- **N pairs**: 50

The paired t-test reaches significance (p = 0.033). The Wilcoxon test does not (p = 0.064), though it is close to the threshold. The effect size is small.

## Interpretation

### What the numbers say

CAG produces measurably better grounded answers than a standard RAG baseline on this benchmark. The advantage is consistent across runs and across query types, and the paired t-test supports significance at p < 0.05.

The effect is modest. Cohen's d of 0.31 means CAG does not dominate RAG; it does better on average, but there is substantial overlap in per-question performance.

The Wilcoxon test does not reach significance. This is worth noting honestly. The t-test assumes normality of differences; the Wilcoxon does not. When the two tests disagree, the truth is likely somewhere in between.

### Where CAG adds value

- **DIAGNOSTIC queries**: The largest quality gap (+0.094). Questions that require reasoning about symptoms, root causes, and evidence quality benefit most from the CAG pipeline.
- **Escalation behavior**: CAG hallucinates less on unanswerable questions. Its meta-cognitive validation loop catches unsupported answers more often than the baseline.
- **Stability**: CAG's scores vary less across runs, suggesting the orchestration layer smooths out some of the generator's stochastic variance.

### Where CAG pays a cost

- **Latency**: CAG takes roughly 3.5x longer per question (11.7s vs 3.4s). This is because it runs multiple graph nodes (classify, retrieve, refine, reason, validate) rather than a single retrieve-and-generate pass.
- **Cost**: CAG uses roughly 2.1x more tokens. The reason node can be called multiple times on retries, and the validation loop adds its own generation step.
- **PROCEDURAL queries**: Both systems score below 0.7. CAG improves over RAG but neither handles step-by-step procedural questions well. This is likely a retrieval problem: procedural answers require precise sequential chunk retrieval, which top-k similarity search does not always provide.

### What this does not prove

- These results apply to this specific benchmark (50 questions, 4 synthetic documents, gpt-4o). They do not generalize to all domains, all corpora, or all models.
- The benchmark is small. 50 questions is enough to detect a small effect with some statistical support, but a larger benchmark (200+ questions) would produce more reliable significance estimates.
- The RAG baseline is deliberately simple (retrieve-then-generate, no reranking, no query decomposition). Advanced RAG systems with reranking or query decomposition may close some of the gap.
- No LLM judge was used. The scoring is deterministic and token-based, which is reproducible but may miss semantic nuances that a judge would capture.

## Claim

On this benchmark, CAG produces better grounded answers than a standard RAG baseline with statistical support from a paired t-test (p = 0.033). The effect is small but consistent. CAG also shows lower hallucination rates and higher task success, at the cost of higher latency and token usage.

This is a preview-stage result. A larger corpus, more diverse question types, and comparison against advanced RAG baselines would strengthen the claim.
