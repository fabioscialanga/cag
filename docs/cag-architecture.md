# CAG Architecture

## What CAG Is

CAG stands for **Cognitive Augmented Generation**. It is a retrieval system that does not just fetch chunks and hand them to a language model. It decides how to search, how to reason about the evidence it found, and -- when the evidence is too weak -- whether to answer at all.

The system is organized as a directed graph where each node performs a distinct cognitive function: classifying the question, retrieving documents, evaluating evidence quality, generating a grounded answer, and validating that answer before it reaches the user. Routing between nodes is conditional: the graph can skip generation entirely when evidence is insufficient, or it can retry generation when the first attempt looks unreliable.

The runtime implementation lives under:

- `src/cag/graph/` -- graph assembly, node functions, conditional routing, shared state
- `src/cag/agents/` -- retrieval agent (evidence ranking and context selection) and reasoning agent (answer generation)
- `src/cag/eval/` -- benchmark harness for comparing CAG against RAG baselines


## Why Cognitive?

CAG uses the word "cognitive" in the specific sense of **meta-cognition**: the system evaluates its own ability to produce a reliable answer at multiple points during processing. This is not a cosmetic label. Each node in the graph serves a distinct meta-cognitive function:

| Node | Meta-cognitive role | What it evaluates |
|---|---|---|
| **ENTRY** | Context comprehension | What kind of question is this? What scope does it have? What retrieval strategy fits best? What language does the user speak? |
| **RETRIEVE + SELECT_CONTEXT** | Self-assessment of evidence quality and context selection | Is what I found good enough to answer? Which chunks are most useful together? Are there gaps? Should I even proceed? |
| **REASON** | Conditioned generation | Given the evidence I judged sufficient, what answer can I safely construct? |
| **VALIDATE** | Meta-validation | Did I produce a grounded answer? Is the hallucination risk acceptable? Should I retry or escalate? |

The key insight is that the system can **refuse to answer**. When evidence quality is below threshold, CAG either retries with adjusted reasoning or escalates to a human rather than fabricating a plausible-sounding response. This awareness of its own limits -- and the architectural discipline to act on that awareness -- is what separates CAG from a pipeline that always produces output regardless of input quality.

This is different from "RAG with extra steps." In a standard RAG pipeline, retrieval is a preprocessing stage and the model always generates. In CAG, the orchestration layer itself is where the reasoning happens: classification informs retrieval strategy, evidence evaluation gates generation, and validation gates delivery.


## Core Difference from RAG

Standard RAG is usually:

```
retrieve -> generate
```

CAG replaces this with a control loop:

```
ENTRY -> RETRIEVE -> SELECT_CONTEXT (rank + select context) -> REASON -> VALIDATE -> EXIT
```

This is worth examining in more detail, because advanced RAG implementations exist that add reranking, guardrails, and output filtering. The distinction is not about feature count. It is about **where the intelligence lives**.

In standard RAG -- even advanced variants -- retrieval is a preprocessing step. The query comes in, chunks come out, and the language model generates. Post-hoc filters may reject unsafe outputs, but the core flow is linear: always retrieve, always generate, maybe block.

In CAG, the orchestration is not a thin wrapper around vector search. It is part of the system's reasoning:

1. **Classification before retrieval.** The ENTRY node inspects the query and decides what type it is (GENERAL, PROCEDURAL, DIAGNOSTIC, CONFIGURATION) and what scope it has (domain, personal, consultative). This classification directly controls the retrieval strategy, the number of chunks fetched, and how query variants are constructed.

2. **Evidence quality and context selection gate generation.** After retrieval, the SELECT_CONTEXT node scores each chunk, clusters them, assigns semantic categories, and selects the best context through diversity-aware reordering. Then the system explicitly asks: is this evidence good enough to answer? If not, it skips REASON entirely and goes straight to VALIDATE with no answer, which triggers escalation. Generation is conditioned on both evidence quality and context selection, not performed unconditionally.

3. **Retry with awareness.** When REASON produces an answer but VALIDATE detects high hallucination risk, the system can send the query back through REASON (up to a configured maximum of 2 retries). This is not blind repetition -- each retry increments a counter that eventually forces escalation.

4. **Structured escalation over fabrication.** When the system cannot produce a safe answer, it says so explicitly and tells the user what went wrong. It does not lower its thresholds and hope for the best.


## Graph Flow

The graph is assembled in `src/cag/graph/graph.py` using LangGraph's `StateGraph`. The shared state is defined in `src/cag/graph/state.py` as a `TypedDict` with fields for query metadata, chunks, scores, answer, citations, and control flags.

### ASCII Diagram

```
START
  |
  v
 ENTRY -- classify query type, scope, strategy, detect language
  |
  v
 RETRIEVE -- fetch chunks with strategy-aware k and query variants
  |
  v
 SELECT_CONTEXT -- rank chunks, select best context (diversity + category), compute relevance, identify gaps
  |
  +-- reasonable evidence --> REASON  <-- receives top-6 context-selected chunks
  |                            |
  |                            v
  |                          VALIDATE
  |                            |
  +-- weak evidence ----> VALIDATE  +-- acceptable --------------> EXIT
                               |
                               +-- high hallucination risk,   -> REASON (retry)
                               |   retries <= max
                               |
                               +-- max retries reached or      -> EXIT (escalation)
                                   fatal quality problem
```

### Routing Functions

Two conditional edges control the flow:

**`route_after_select_context`** (SELECT_CONTEXT -> REASON or VALIDATE)

After SELECT_CONTEXT ranks chunks, performs context selection, and scores the evidence, the system checks `_has_reasonable_evidence()`. If evidence is reasonable, the graph proceeds to REASON with the top-6 context-selected chunks. If not, it skips REASON entirely and goes directly to VALIDATE with an empty answer -- which will trigger escalation.

**`route_after_validate`** (VALIDATE -> EXIT or REASON)

After VALIDATE checks the answer, two outcomes are possible:
- If `should_escalate` is `True`, the graph goes to EXIT with an escalation message.
- If hallucination risk exceeds the threshold (0.3) and the retry count has not exceeded the maximum (2), the graph routes back to REASON for another attempt.
- Otherwise, the graph goes to EXIT with the generated answer.

### Evidence Quality Criteria

`_has_reasonable_evidence()` returns `True` if any of these conditions hold:

| Condition | Rationale |
|---|---|
| Top chunk score >= `relevance_threshold` (0.7) | At least one chunk is strongly relevant |
| Top chunk score >= 0.55 AND query type is PROCEDURAL or DIAGNOSTIC | Troubleshooting and how-to queries tolerate slightly weaker individual chunks because they often require assembling multiple partial sources |
| At least 2 chunks with score >= 0.55 AND overall relevance >= 0.55 | Multiple moderately relevant chunks provide adequate coverage even without a single strong hit |

If none of these hold, evidence is considered weak and REASON is skipped.

### VALIDATE Escalation Conditions

VALIDATE checks five escalation conditions, evaluated in order. If any is true, `should_escalate` is set to `True` and an error message is attached:

1. **No answer AND no reasonable evidence.** REASON was skipped (weak evidence path) and no answer exists. The system cannot fabricate something from nothing.

2. **Insufficient answer text AND (no reasonable evidence OR low confidence OR high hallucination risk).** REASON ran but its own output signals that the documentation was inadequate. This is detected by pattern matching against signals like "documentation is insufficient," "not documented," "cannot determine," and similar phrases.

3. **Low relevance AND no reasonable evidence.** Overall relevance is below threshold and the per-chunk check also fails. The evidence base is weak from every angle.

4. **High hallucination risk AND max retries reached.** The system tried to generate multiple times but each attempt produced high hallucination risk. Further retries are unlikely to help.

5. **Low confidence AND max retries reached.** Confidence remains below threshold after all retry attempts are exhausted.

### Retry Condition

The only condition that triggers a retry (as opposed to an escalation or normal exit) is:

- `hallucination_risk > hallucination_threshold` (0.3 by default)
- `reason_retries <= max_reason_retries` (2 by default)

When this condition holds, the graph routes back to REASON. The retry counter increments on each REASON invocation. Once the counter exceeds the maximum, any subsequent high hallucination risk triggers escalation instead.


## Language Detection

The ENTRY node detects the language of each query using `langdetect` (`_infer_response_language` in `src/cag/graph/nodes.py`). The detected language is stored in the `response_language` field of the shared state as an ISO 639-1 code (e.g., `en`, `it`, `fr`, `de`, `es`).

This language code propagates through the entire pipeline:

1. **ReasoningAgent** receives `RESPONSE LANGUAGE: {code}` in its prompt, which instructs the LLM to answer in the detected language.
2. **VALIDATE** and **EXIT** nodes use `_localized_message()` to produce error messages and escalation notices in the user's language. Messages are available for English, Italian, French, German, Spanish, and Portuguese, with English as the fallback.
3. **Error fallback** in the ReasoningAgent also produces localized error messages based on the detected language.

If `langdetect` cannot determine the language (e.g., very short queries), the system defaults to English.


## Runtime Configuration

CAG supports per-request runtime overrides via `RuntimeConfig` (`src/cag/graph/runtime.py`). This allows callers to adjust thresholds without modifying global settings:

| Parameter | Default | Description |
|---|---|---|
| `relevance_threshold` | 0.7 | Minimum relevance score to consider evidence reasonable |
| `confidence_threshold` | 0.6 | Minimum confidence for a valid answer |
| `hallucination_threshold` | 0.3 | Maximum acceptable hallucination risk |

Runtime config is passed to `run_query()` and stored in the shared state. All node functions read thresholds from the state rather than from global settings, ensuring that per-request overrides are respected throughout the graph execution.


## Query Classification

### Query Types

The ENTRY node classifies each query into one of four types. This classification controls retrieval strategy, chunk count, and query variant construction.

| Type | Retrieval Strategy | Effective k | Typical Questions |
|---|---|---|---|
| **GENERAL** | `semantic` | default (10) | Factual lookups, definitions, "what is" questions |
| **PROCEDURAL** | `hierarchical` | default + 4 (14) | Step-by-step guides, "how do I" questions |
| **DIAGNOSTIC** | `multi_evidence` | default + 4 (14) | Troubleshooting, error analysis, "why does" questions |
| **CONFIGURATION** | `semantic` | default (10) | Settings, parameters, prerequisites, "which fields" questions |

Classification is pattern-based. The query string is matched against keyword lists defined in `_infer_query_type()`. Diagnostic patterns include error-specific terms ("error", "404", "not working", "rejected"). Procedural patterns include action phrases ("how do i", "step by step", "how to"). Configuration patterns include setup terms ("configure", "settings", "prerequisite"). General patterns include definitional phrases ("what is", "how does"). The fallback is GENERAL.

### Question Scopes

In addition to query type, ENTRY classifies the question's scope:

| Scope | Meaning | Retrieval Impact |
|---|---|---|
| **domain** | Document-specific questions about content | Normal retrieval flow |
| **personal** | Conversational questions ("how are you", "who are you") | Not a document query; the system still processes normally but may find no relevant chunks |
| **consultative** | Advisory questions ("what should I", "recommend") | Forces `multi_evidence` retrieval strategy regardless of query type |

Scope is determined by `_classify_question_scope()`, which checks for personal markers, consultative markers, and domain markers in the query. If personal markers are found without domain markers, the scope is `personal`. If consultative markers are found, the scope is `consultative`. Otherwise, the default is `domain`.

### Query Variants

RETRIEVE does not use a single query string. It constructs up to 3 query variants via `_build_query_variants()`:

1. **Original query** -- the user's exact input, stripped of whitespace.
2. **Keyword-focused variant** -- the query is normalized (lowercased, punctuation replaced with spaces), stopwords are removed, and the top 8 keywords are joined. Common action words are rewritten to their noun forms (e.g., "configure" becomes "configuration", "solve" becomes "resolution").
3. **Compact variant** (PROCEDURAL, DIAGNOSTIC, CONFIGURATION only) -- question prefixes like "how do I", "how can I", "why" are stripped, leaving only the substantive terms.

Each variant is run against the vector store independently. The first variant uses the full `per_query_k`; subsequent variants use half that value (minimum 4). Results are deduplicated by (filename, chunk_index, content prefix) and sorted by keyword overlap with the original query.


## Context Selection

After the retrieval agent ranks chunks by relevance, CAG performs a **context selection** step (`_reorder_for_context_selection` in `src/cag/agents/retrieval_agent.py`). This is not a simple top-k truncation. It reorders the ranked chunks to maximize the information value of the limited context window that gets passed to the ReasoningAgent.

### Why Context Selection Exists

A naive approach would pass the top-N chunks by relevance score directly to the reasoning agent. But this creates two problems:

1. **Redundancy.** The top chunks often overlap heavily -- they cover the same information from the same source. Passing redundant chunks wastes context window budget without adding information.
2. **Category imbalance.** For a procedural query, the most relevant chunks might all be about prerequisites, while the actual step-by-step procedure is ranked slightly lower. Without category awareness, the answer will miss critical information.

Context selection solves both problems by choosing which chunks to include and in what order, respecting a limit of `SELECTION_CONTEXT_LIMIT` (6 chunks by default).

### How It Works

The selection algorithm uses a greedy scoring function that evaluates each remaining chunk against the chunks already selected:

**Score = relevance + category_priority + diversity + overlap_bonus + type_bonus - penalties**

| Factor | What it does |
|---|---|
| **Relevance score** | Base signal: `chunk.relevance_score * 100`. Higher relevance chunks are preferred. |
| **Category priority** | Each query type has a prioritized list of semantic categories. For PROCEDURAL queries, `ordered_steps` and `navigation` get the highest priority. For DIAGNOSTIC, `symptoms`, `error_causes`, `checks`, and `resolution` are prioritized. |
| **Cluster diversity** | A bonus (+6.0 for focused queries, +2.5 for GENERAL) for selecting a chunk from a cluster that is not yet represented in the selected set. Repeated clusters are penalized (-4.0 per occurrence). |
| **Category diversity** | A bonus for selecting a chunk with a category not yet in the selected set. Repeated categories are penalized. |
| **Source penalty** | Slight penalty (-1.5 for focused queries, -0.5 for GENERAL) for repeating the same source document, to encourage cross-source evidence. |
| **Query overlap bonus** | `+2.0 * keyword_overlap(query, chunk)`. Chunks that share keywords with the original query get a boost. |
| **Near-duplicate penalty** | If a candidate chunk has keyword overlap >= 0.72 with any already-selected chunk, it receives a penalty of `-5.0 * similarity`. This prevents passing near-identical chunks to the reasoning agent. |
| **Type-specific bonuses** | PROCEDURAL queries get +6.0 for `ordered_steps`/`navigation` categories and a bonus for earlier chunk indices. DIAGNOSTIC and CONFIGURATION queries get +4.0 for their respective priority categories. |
| **Weaker new source penalty** | For GENERAL queries, introducing a new source that has lower relevance than the best already-selected chunk is penalized heavily (`-50.0 * relevance_gap`), preventing low-quality diversity. |

The algorithm iterates greedily: at each step, it picks the chunk with the highest combined score from the remaining pool and adds it to the selected set. This produces an ordered list that balances relevance, diversity, and query-type-specific information needs.

### Chunk Clustering

Before ranking, chunks are grouped into automatic clusters (`_cluster_chunks`). Two chunks are assigned to the same cluster if they share at least 2 keywords or share a source document and at least 1 keyword. These clusters are used by the diversity scoring during context selection.

### Semantic Categories

The retrieval agent assigns each chunk a `selection_category` -- a short semantic label like `ordered_steps`, `prerequisites`, `error_causes`, `settings`, etc. These categories are normalized through `CATEGORY_ALIASES` (e.g., "steps", "step", "procedure" all map to `ordered_steps`) and used to prioritize chunks that match the query type's information needs.

### Context Selection Budget

The ReasoningAgent receives at most the top `SELECTION_CONTEXT_LIMIT` (6) chunks from the context-selected list. The limit is enforced in `run_reasoning_agent` (`ranked_chunks[:6]`). This budget ensures the reasoning prompt stays focused and avoids diluting the signal with marginally relevant evidence.

### cag_no_selection Baseline

The evaluation harness includes a `cag_no_selection` system (`_run_cag_no_selection_query` in `src/cag/eval/systems.py`) that runs the full CAG pipeline but bypasses context selection. After REFINE, the ranked chunks are reordered back to their raw retrieval order via `_restore_raw_chunk_order`. This baseline exists to isolate the contribution of context selection to answer quality in benchmark comparisons. When comparing `cag` vs `cag_no_selection`, the difference in scores measures the specific impact of the context selection step.


## Worked Examples

### Example 1: Success Path

**Query:** "How do I configure the notification module?"

```
ENTRY
  query_type  = CONFIGURATION  (matches "configur" pattern)
  scope       = domain         (no personal or consultative markers)
  strategy    = semantic       (default for CONFIGURATION)

RETRIEVE
  variants    = ["How do I configure the notification module?",
                 "notification module configuration",
                 "notification module"]
  k           = 10 per variant (semantic strategy, default k)
  result      = 10 deduplicated chunks about notification configuration

SELECT_CONTEXT
  top chunk score  = 0.88  (strong match on notification module configuration)
  gaps             = []    (no missing information detected)
  relevance_score  = 0.82  (overall evidence quality is high)
  routing          -> REASON  (top score 0.88 >= threshold 0.7)

REASON
  confidence         = 0.85
  hallucination_risk = 0.10
  citations          = 3 chunks cited
  answer             = structured response with prerequisites, required fields,
                       and configuration steps, grounded in the retrieved chunks

VALIDATE
  top score >= threshold          -> reasonable evidence: YES
  answer is substantive           -> insufficient answer: NO
  hallucination_risk 0.10 < 0.3   -> within limits
  confidence 0.85 >= 0.6          -> above threshold
  should_escalate                 -> FALSE
  routing                         -> EXIT (normal)

EXIT
  Delivers the generated answer with 3 citations.
  Node trace: ENTRY -> RETRIEVE -> SELECT_CONTEXT -> REASON(retry=0) -> VALIDATE -> EXIT
```

### Example 2: Escalation Path

**Query:** "What is the pricing for the enterprise plan?"

```
ENTRY
  query_type  = GENERAL        (matches "what is" pattern)
  scope       = domain
  strategy    = semantic       (default for GENERAL)

RETRIEVE
  variants    = ["What is the pricing for the enterprise plan?",
                 "pricing enterprise plan"]
  k           = 10 per variant
  result      = 10 chunks, none about pricing (documentation does not cover this topic)

SELECT_CONTEXT
  top chunk score  = 0.22  (best match is tangential, perhaps mentioning "enterprise" in
                            an unrelated context)
  gaps             = ["pricing information", "enterprise plan details"]
  relevance_score  = 0.18  (overall evidence quality is very low)
  routing          -> VALIDATE  (top score 0.22 < threshold 0.7;
                                  no 0.55+ chunks; no moderate-coverage case)

  Note: REASON is skipped entirely. The system refuses to generate from this evidence.

VALIDATE
  answer is empty                  -> no answer
  _has_reasonable_evidence()       -> FALSE (top score 0.22, no moderate chunks)
  Escalation condition 1 triggers  -> should_escalate = TRUE
  error_message = "The retrieved documentation does not cover this request
                   reliably. Human review or additional source material is required."
  routing                          -> EXIT (escalation)

EXIT
  Delivers escalation message:
    "Support escalation recommended.
     The retrieved documentation does not cover this request reliably.
     Human review or additional source material is required.
     Please route this question to a human reviewer or provide additional
     supporting documents."
  Node trace: ENTRY -> RETRIEVE -> SELECT_CONTEXT -> VALIDATE -> EXIT
```

### Example 3: Retry Path

**Query:** "How do I fix the rejected invoice error?"

```
ENTRY
  query_type  = DIAGNOSTIC     (matches "fix" + "rejected" + "error" patterns)
  scope       = domain
  strategy    = multi_evidence (DIAGNOSTIC forces this strategy)

RETRIEVE
  k           = 14 per variant (multi_evidence strategy, default + 4)
  result      = 14 chunks about invoice errors and rejection handling

SELECT_CONTEXT
  top chunk score  = 0.71
  relevance_score  = 0.65
  routing          -> REASON  (top score 0.71 >= threshold 0.7)

REASON (attempt 1)
  confidence         = 0.55
  hallucination_risk = 0.45   (high -- the answer strays from the evidence)
  answer             = partially grounded response

VALIDATE
  hallucination_risk 0.45 > 0.3   -> above threshold
  reason_retries = 1 (first attempt)
  1 <= max_reason_retries (2)     -> retry allowed
  should_escalate                  -> FALSE (not checked, retry takes priority)
  routing                          -> REASON (retry)

REASON (attempt 2)
  confidence         = 0.78
  hallucination_risk = 0.15   (improved -- the model stays closer to the evidence)
  answer             = well-grounded response with corrective steps

VALIDATE
  hallucination_risk 0.15 < 0.3  -> within limits
  should_escalate                 -> FALSE
  routing                         -> EXIT (normal)

EXIT
  Delivers the second answer with citations.
  Node trace: ENTRY -> RETRIEVE -> SELECT_CONTEXT -> REASON(retry=0) -> VALIDATE
              -> REASON(retry=1) -> VALIDATE -> EXIT
```


## What This Repository Tries to Prove

This repository is not claiming that every possible CAG system is better than every RAG system.

It is making one narrower claim testable:

**For this implementation, orchestration can improve grounded answer quality over a simpler RAG baseline.**

That is why the benchmark harness is a first-class part of the repository. The evaluation framework under `src/cag/eval/` supports running the same question set through both CAG and a standard RAG pipeline, scoring answers on relevance, groundedness, and safety, and producing a structured comparison.

The goal is not to prove a general theorem. It is to provide a reproducible experiment where the hypothesis -- that meta-cognitive orchestration improves answer quality -- can be confirmed or falsified against a concrete baseline.


## Preview Scope

**What is already solid:**

- Graph-driven orchestration with conditional routing and retry logic
- Context selection with diversity-aware chunk ordering and category prioritization
- Automatic language detection (55+ languages via langdetect) with localized messages
- Runtime configuration for per-request threshold overrides
- Evaluation harness with `cag_no_selection` baseline for isolating context selection impact
- Generic document-oriented positioning (no domain-specific assumptions)
- React frontend and FastAPI backend for interactive testing
- Configurable thresholds (relevance, confidence, hallucination, max retries)

**What is still preview-stage:**

- Benchmark size and corpus diversity
- Prompt stability across different LLM providers
- Broader community-facing examples and documentation
- Performance optimization for large document collections
