# CAG

![Status](https://img.shields.io/badge/status-preview-yellow)
![Version](https://img.shields.io/badge/version-0.2-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Cognitive Augmented Generation — graph-driven document reasoning with explicit context selection**

CAG is a retrieval reasoning system for AI builders who want something more deliberate than plain `retrieve top-k -> generate answer`.

In one sentence: **CAG is a retrieval system that decides how to search, how to select evidence, how to reason, and when not to answer.**

If standard RAG treats retrieval as context assembly, CAG treats retrieval as a reasoning problem.

## What CAG Does

CAG is built around a **directed reasoning graph** where every step serves a distinct cognitive function. Before generating an answer, the system:

1. **Classifies the query** — what type of question is this? (factual, procedural, diagnostic, configuration). The classification directly controls how retrieval is performed.
2. **Detects the user's language** — 55+ languages via langdetect. The entire pipeline responds in the user's language.
3. **Retrieves with strategy awareness** — different query types use different retrieval strategies and different chunk budgets. Multi-variant queries are constructed so retrieval covers more surface area.
4. **Selects the best context** — after retrieving candidate chunks, the system clusters them, assigns semantic categories (e.g. `ordered_steps`, `error_causes`, `prerequisites`), and reorders them using a diversity-aware scoring function. Only the most informative set of chunks reaches the reasoning agent.
5. **Reasons over selected evidence** — the reasoning agent generates an answer grounded in the selected context, not the entire raw retrieval pool.
6. **Validates the answer** — confidence, hallucination risk, and evidence sufficiency are all checked before the answer is returned. If any check fails, the system either retries generation or escalates to a human instead of guessing.

The control loop is:

```
ENTRY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> VALIDATE -> EXIT
```

Each node is aware of the others. The system can backtrack, retry, or refuse — not because a guardrail caught a bad output, but because the orchestration layer determined the evidence was inadequate.

---

## TL;DR

- **What it is**: a graph-based retrieval reasoning system for grounded document QA.
- **What makes it different**: it adds query typing, strategy-aware retrieval, diversity-aware context selection, language detection, validation, and escalation to the retrieval loop.
- **Why it matters**: the hardest part of grounded QA is usually not generation itself — it is deciding what to retrieve, what to keep, whether the evidence is sufficient, and when to refuse.

---

## RAG vs CAG

```text
Standard RAG
  question
    -> retrieve
    -> generate
    -> answer

CAG
  question
    -> classify the request (type + scope + language)
    -> retrieve candidate evidence (strategy-aware, multi-variant)
    -> select the best context (cluster, categorize, diversity-reorder)
    -> reason over selected evidence
    -> validate (confidence + hallucination risk + evidence sufficiency)
    -> answer / retry / escalate
```

The key architectural difference is the **SELECT_CONTEXT** step. Standard RAG passes all retrieved chunks directly to generation. CAG first asks: *which of these chunks actually deserve to be in the answer prompt?*

Context selection matters because:

- The top chunks by relevance score often overlap — they cover the same information from the same source. Passing them all wastes context window budget without adding information.
- For a procedural query, the most relevant chunks might all cover prerequisites, while the actual step-by-step procedure is ranked slightly lower. Without category awareness, the answer misses critical information.

The SELECT_CONTEXT node resolves both problems by choosing which chunks to include and in what order, respecting a budget of 6 chunks and balancing relevance, category diversity, source diversity, and query-type-specific priorities.

---

## CAG vs RAG at a Glance

| Area | Standard RAG | CAG |
| --- | --- | --- |
| Paradigm | retrieval-enhanced generation | self-aware retrieval reasoning |
| Core flow | `retrieve -> generate` | `ENTRY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> VALIDATE -> EXIT` |
| Query handling | one generic path | query-typed path selection |
| Retrieval behavior | static top-k | strategy-aware, multi-variant |
| Context passed to LLM | all retrieved chunks | diversity-selected subset (max 6) |
| Grounding control | prompt-only | explicit validation and escalation |
| Failure mode | answers even with weak support | can refuse or escalate unsupported requests |
| Language support | typically English-only | 55+ languages, auto-detected |
| Benchmark story | usually ad hoc | built-in evaluation harness with ablation baselines |

---

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
cag ingest --data-dir ./data/raw
cag eval --system cag --limit 3
```

---

## Installable Package

CAG is installed as a local editable package and used through the `cag` CLI.

```bash
pip install -e .
```

Useful subcommands:

```bash
cag ingest   --data-dir ./data/raw
cag query    "How is this workflow configured?"
cag eval     --system cag --limit 3
cag eval-audit
cag compare  --runs ./artifacts/eval_runs/<run_a> ./artifacts/eval_runs/<run_b>
```

Windows triplet benchmark launcher:

```powershell
.\run_eval_triplet.ps1
```

---

## What This Repo Gives You

- a graph-based CAG runtime with explicit context selection
- a fair benchmark harness against multiple baselines
- a generic document-oriented stack rather than a domain-specific demo
- a React + FastAPI preview path for local experimentation

---

## API Preview Notes

- `/query` accepts per-request validation thresholds without mutating global process settings
- `/query` and `/upload` support preview protection via `X-API-Key` when `CAG_API_KEY` is configured
- `/upload` validates filename, extension, and upload size before writing to `data/raw/`
- query and benchmark outputs expose `fallback_used` and `fallback_reason` so degraded agent paths are visible instead of silent
- the eval harness injects retrieval search explicitly instead of patching global graph functions

How preview API protection works:

- `CAG_API_KEY` is the shared secret stored on the backend in `.env`
- `X-API-Key` is the HTTP header that the client sends on each request to `/query` and `/upload`
- the backend compares the incoming `X-API-Key` value with `CAG_API_KEY`
- if they match, the request is accepted
- if `CAG_API_KEY` is set and the header is missing or wrong, the API returns `401` or `403`
- if `CAG_API_KEY` is not set, the preview API stays open locally and logs a warning instead of blocking requests

Example `.env`:

```env
CAG_API_KEY=my-preview-key
```

Example preview request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CAG_API_KEY" \
  -d '{
    "query": "What is the minimum RAM required to run Nexus Platform?",
    "relevance_threshold": 0.7,
    "confidence_threshold": 0.6,
    "hallucination_threshold": 0.3
  }'
```

---

## Benchmark Snapshot

This repo is built to make the orchestration claim inspectable, not just assertive.

- compare `cag` against `cag_no_selection`, `rag_baseline`, `direct_baseline`, and `lightrag_baseline`
- inspect machine-readable artifacts in `artifacts/eval_runs/` and `artifacts/eval_comparisons/`
- rerun the same benchmark with `--runs N` when you want repeated-run statistics instead of a single descriptive comparison

The current benchmark design tracks context selection quality directly:

- `context_precision_score`: fraction of selected context chunks whose sources match gold sources
- `retrieved_chunk_count` and `selected_chunk_count`: minimal counters to interpret context quality and budget
- `cag_no_selection`: ablation that keeps the same retrieval and validation flow but disables context selection — this isolates the contribution of SELECT_CONTEXT to answer quality
- `cag eval-audit`: benchmark coverage check for dataset size, balance, duplicate questions, and corpus-source coverage

Latest 100-question triplet snapshot:

| System | Grounded Answer | Context Precision | Hallucination Rate | Task Success |
| --- | ---: | ---: | ---: | ---: |
| `cag` | 0.848 | 0.878 | 0.100 | 0.820 |
| `cag_no_selection` | 0.799 | 0.645 | 0.150 | 0.750 |
| `rag_baseline` | 0.824 | 0.531 | 0.110 | 0.760 |

The strongest current claim is not yet "CAG is universally superior on every final metric." It is:

**CAG selects evidence much better than both `cag_no_selection` and `rag_baseline`, and that better context selection improves downstream task success while slightly reducing hallucination.**

The gap between `cag` and `cag_no_selection` isolates the contribution of the SELECT_CONTEXT step: same retrieval, same reasoning agent, same validation — only context selection disabled. The 23-point context precision gap (0.878 vs 0.645) and 7-point task success gap (0.820 vs 0.750) show the measurable impact of deliberate evidence selection.

---

## Project Status

**Version:** 0.2  
**Status:** Preview

**Audience:** AI builders and practitioners exploring retrieval, agentic orchestration, and grounded QA

**What changed in 0.2:**

- The SELECT_CONTEXT stage is now architecturally explicit in the code, graph, and documentation. Previously named REFINE, it has been renamed to reflect its actual responsibility: evidence scoring, clustering, semantic categorization, and diversity-aware context selection. The graph node is `select_context`, the routing function is `route_after_select_context`, and the node trace now records `SELECT_CONTEXT`.
- Documentation updated throughout (`cag-architecture.md`, `README.md`) to use the correct terminology.

**Current non-goals:**
- production SLA
- hosted multi-tenant deployment
- broad domain benchmark coverage
- turnkey package distribution

---

## Why CAG Over Standard RAG?

To be clear: advanced RAG systems already exist. Production RAG pipelines routinely include reranking, query decomposition, guardrails, and other improvements over naive retrieval. CAG is not claiming that all RAG is simplistic.

The distinction is architectural, not incremental. In standard RAG -- even advanced RAG -- retrieval is a preprocessing step: documents are fetched and then passed to a generator. The retrieval layer does not reason about what it is doing. Guardrails and rerankers are usually bolted on as separate stages rather than integrated into a single reasoning loop.

CAG uses "Cognitive" in the specific sense of meta-cognition: the system reasons about its own reasoning. The orchestration layer is not a passive pipeline -- it is part of the system's intelligence. It:

- **classifies the query type** before deciding how to search
- **detects the user's language** and responds in kind (55+ languages)
- **selects and adapts the retrieval strategy** based on what kind of question is being asked
- **selects the best context** from retrieved evidence before generation begins — this is the SELECT_CONTEXT step
- **evaluates evidence quality** after selection, surfacing gaps before generation begins
- **validates the generated answer** against the evidence and can refuse to answer when support is insufficient

### The SELECT_CONTEXT Step in Detail

After retrieval, the system has a pool of candidate chunks. Before passing anything to the reasoning agent, SELECT_CONTEXT:

1. **Clusters** the chunks by keyword and source proximity
2. **Assigns semantic categories** (e.g. `ordered_steps`, `prerequisites`, `error_causes`, `settings`) using an LLM-assisted categorization
3. **Scores each chunk** for relevance on a 0–1 scale
4. **Reorders** the pool using a greedy diversity-aware algorithm that balances:
   - relevance score
   - category diversity (avoiding redundant category overrepresentation)
   - cluster diversity (avoiding near-duplicate evidence)
   - query-type-specific priorities (e.g. PROCEDURAL queries get a bonus for `ordered_steps` chunks)
   - near-duplicate penalty (chunks with >72% keyword similarity to already-selected chunks are penalized)
5. **Passes** at most 6 chunks to the reasoning agent

This means the reasoning agent always receives a small, intentional context set — not a raw dump of whatever happened to score highest.

### The Difference in Plain Language

RAG answers a question by retrieving relevant chunks and asking the model to answer from them.

CAG first asks a higher-level question:

`What kind of problem is this, what kind of evidence would justify an answer, which specific chunks are most informative together, and is the current evidence strong enough to answer at all?`

That difference matters because the hardest part of grounded QA is often not generation itself. It is deciding:
- what to retrieve
- what to keep
- whether the retrieved material is enough
- how much confidence is justified
- when the system should stop and refuse instead of guessing

### Practical Advantages of CAG

- **Better control over failure modes**: CAG can refuse or escalate unsupported requests instead of answering with weak confidence.
- **More appropriate retrieval behavior**: procedural, diagnostic, and factual questions do not share the same retrieval path.
- **Stronger grounding discipline**: evidence is selected, refined, and validated before the answer is treated as trustworthy.
- **More interpretable orchestration**: the graph makes it easier to inspect how the system reached an answer or why it declined.
- **More meaningful evaluation**: the benchmark harness lets you compare orchestration strategies, not only model outputs.

---

## When To Use CAG

Use CAG when:
- you want the system to adapt retrieval behavior to different query types
- you care about grounded answers more than raw response speed
- you want explicit refusal or escalation behavior on weakly supported requests
- you want to benchmark orchestration quality, not just model quality

Do not use CAG when:
- you only need simple semantic search plus one-shot answering
- latency and minimal complexity matter more than control
- you do not need benchmarkable retrieval behavior
- a plain RAG stack already solves your use case cleanly

---

## Current Maturity

Solid today:
- graph-based CAG runtime with explicit SELECT_CONTEXT stage
- automatic language detection (55+ languages via langdetect)
- benchmark harness with multiple systems including `cag_no_selection` ablation
- React + FastAPI preview path
- passing test suite and frontend production build

Still experimental:
- benchmark size and corpus diversity
- prompt-level stability across providers
- LightRAG comparison path depending on external credentials/runtime

Still needs work before a stable release:
- broader evaluation corpus
- richer CI and release automation over time
- more polished onboarding assets and demo material
- stronger contributor workflow and long-term governance

---

## Quick Proof

The fastest way to see the project's shape is:

1. run tests
2. run a small benchmark
3. inspect the generated artifacts

Smoke benchmark example:

```bash
cag eval --system cag --limit 3
```

Comparison example:

```bash
cag eval --system rag_baseline --limit 3 --judge-mode off
cag eval --system cag_no_selection --limit 3 --judge-mode off
cag eval --system cag --limit 3 --judge-mode off
cag compare --runs ./artifacts/eval_runs/<rag_run> ./artifacts/eval_runs/<cag_no_selection_run> ./artifacts/eval_runs/<cag_run>
```

Expected artifact locations:
- `artifacts/eval_runs/<timestamp>_cag/run.json`
- `artifacts/eval_runs/<timestamp>_cag/results.jsonl`
- `artifacts/eval_comparisons/<timestamp>_comparison/comparison.md`

Example of what a successful run leaves behind:

```text
artifacts/
  eval_runs/
    20260407T120000Z_cag/
      run.json
      results.jsonl
```

---

## Canonical Local Validation Flow

This is the recommended preview path.

### 1. Install CAG

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the core package:

```bash
pip install -e .
```

Optional dependency groups:

```bash
pip install -e ".[dev]"         # pytest and local dev tooling
pip install -e ".[eval]"        # benchmark evaluation
pip install -e ".[lightrag]"    # LightRAG baseline comparison
pip install -e ".[pinecone]"    # Pinecone vector store
pip install -e ".[ui]"          # Streamlit UI
pip install -e ".[api]"         # FastAPI server
pip install -e ".[all]"         # everything
```

### 2. Configure environment

Windows:

```bash
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Then set at least:
- `LLM_PROVIDER`
- `OPENAI_API_KEY` if you use OpenAI-backed flows
- `CAG_API_KEY` if you want preview protection for `/query` and `/upload`

If you set:

```env
CAG_API_KEY=my-preview-key
```

then your client must send:

```http
X-API-Key: my-preview-key
```

on requests to `/query` and `/upload`.

### 3. Use your own corpus

Place your own documents (`.pdf`, `.txt`, `.md`) in `data/raw/`. The repository does not ship with sample documents.

### 4. Ingest documents

```bash
cag ingest --data-dir ./data/raw
```

### 5. Run tests

```bash
pip install -e ".[dev,eval,api]"
pytest
```

### 6. Run a smoke benchmark

```bash
cag eval --system cag --limit 3
```

### 7. Optionally run the API/UI

Recommended preview API + UI:

```bash
python -m uvicorn cag.api.upload:app --reload --port 8000
cd frontend
npm install
npm run dev
```

Secondary/dev-facing UI:

```bash
streamlit run src/cag/ui/app.py
```

---

## Benchmarking

Benchmarking is a first-class feature of this preview.

Available systems:
- `cag`
- `cag_no_selection`
- `rag_baseline`
- `direct_baseline`
- `lightrag_baseline`

Method note:
- same benchmark corpus
- same embedding stack
- same retrieval corpus boundaries
- same or comparable generator family where applicable
- the orchestration path is the main variable under comparison

Typical commands:

```bash
cag eval --system cag
cag eval --system cag_no_selection
cag eval --system rag_baseline
cag eval --system direct_baseline
cag eval --system lightrag_baseline
cag compare --runs ./artifacts/eval_runs/<run_a> ./artifacts/eval_runs/<run_b>
```

For repeated-run significance checks:

```bash
cag eval --system cag --runs 3
cag eval --system rag_baseline --runs 3
cag compare --runs ./artifacts/eval_runs/<cag_multi_run> ./artifacts/eval_runs/<rag_multi_run>
```

`compare` accepts both single-run directories and multi-run directories. When multiple runs are provided for the same system, the statistical comparison uses the mean score per question across runs.

Known benchmark limitations:
- the benchmark is now stronger, but still single-corpus by default
- `cost_estimate` is a relative estimate, not billing truth
- results currently support a stronger context-selection claim than a universal paradigm-superiority claim

---

## First-Run Troubleshooting

Missing API key:
- verify `.env` exists
- verify the provider-specific key is set
- if `/query` or `/upload` returns `401` or `403`, verify `CAG_API_KEY` and the `X-API-Key` header
- LightRAG currently expects an OpenAI-backed path

No documents found:
- add files under `data/raw/`
- confirm supported extensions: `.pdf`, `.txt`, `.md`

Upload rejected:
- check the file extension is one of `.pdf`, `.txt`, `.md`
- check the file is below `10 MiB`
- check the total request size is below `25 MiB`

Vector DB path issues:
- check `CHROMA_PERSIST_DIR`
- delete and rebuild local preview indexes only if you intentionally want a fresh local state

LightRAG issues:
- make sure `OPENAI_API_KEY` is set
- use the generic benchmark first before trying the LightRAG path

Frontend/API mismatch:
- start FastAPI on `http://localhost:8000`
- start the React preview from `frontend/`

---

## Repository Structure

```text
src/cag/
  agents/        # Retrieval agent (evidence ranking + context selection) and reasoning agent
  api/           # FastAPI upload and query endpoints
  eval/          # Benchmark runner, scoring, comparison, datasets
  graph/         # LangGraph state, nodes (ENTRY, RETRIEVE, SELECT_CONTEXT, REASON, VALIDATE, EXIT), routing
  ingestion/     # Loader, chunker, embeddings, vector store integration
  ui/            # Streamlit interface
docs/            # Public architecture and project docs
frontend/        # React preview UI
tests/           # Unit and harness tests
data/raw/        # Source documents for ingestion
artifacts/       # Generated benchmark artifacts
data/chroma_db/  # Local Chroma runtime data
```

---

## Convenience Scripts

The `.bat` files (Windows) and `.sh` files (macOS/Linux) in the repo are convenience launchers for local use. They are **not** the primary documented install or validation path for this preview.

---

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

## License

This repository includes a top-level [LICENSE](LICENSE).
