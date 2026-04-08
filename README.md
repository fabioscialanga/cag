# CAG

![Status](https://img.shields.io/badge/status-preview-yellow)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Cognitive Augmented Generation for grounded document intelligence**

CAG is a graph-driven document reasoning system for AI builders who want something more deliberate than plain `retrieve top-k -> generate answer`.

In one sentence: **CAG is a retrieval system that decides how to search, how to reason, and when not to answer.**

If standard RAG treats retrieval as context assembly, CAG treats retrieval as a reasoning problem.

It combines:
- query typing
- retrieval strategy selection
- evidence refinement
- grounded answer generation
- answer validation and escalation

The goal of this repository is a **GitHub preview**, not a finished product launch. The main hook is the combination of **architecture + benchmarkability**.

Start here if you want the shortest architectural explanation:
- [docs/cag-architecture.md](docs/cag-architecture.md)

## TL;DR

- **What it is**: a graph-based retrieval reasoning system for grounded document QA.
- **Why it matters**: it adds query typing, evidence refinement, validation, and escalation to the retrieval loop.
- **Why it is different from RAG**: RAG mainly retrieves context for generation; CAG reasons about how to retrieve, whether evidence is enough, and when not to answer.

## RAG vs CAG

```text
Standard RAG
question
  -> retrieve
  -> generate
  -> answer

CAG
question
  -> classify the request
  -> retrieve
  -> refine evidence
  -> reason
  -> validate
  -> answer / retry / escalate
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
cag ingest --data-dir ./data/raw
cag eval --system cag --limit 3
```

If you want the shortest `proof of life` path, that is the one.

## Installable Package

CAG can now be installed as a local editable package and used through the `cag` CLI.

Core install:

```bash
pip install -e .
```

Useful subcommands:

```bash
cag ingest   --data-dir ./data/raw
cag query    "How is this workflow configured?"
cag eval     --system cag --limit 3
cag compare  --runs ./artifacts/eval_runs/<run_a> ./artifacts/eval_runs/<run_b>
```

## What This Repo Gives You

- a graph-based CAG runtime
- a fair benchmark harness against multiple baselines
- a generic document-oriented stack rather than a domain-specific demo
- a React + FastAPI preview path for local experimentation

## Benchmark Snapshot

This repo is built to make the orchestration claim inspectable, not just assertive.

- compare `cag` against `rag_baseline`, `direct_baseline`, and `lightrag_baseline`
- inspect machine-readable artifacts in `artifacts/eval_runs/` and `artifacts/eval_comparisons/`
- rerun the same benchmark with `--runs N` when you want repeated-run statistics instead of a single descriptive comparison

## Project Status

**Status:** Preview

**Audience:** AI builders and practitioners exploring retrieval, agentic orchestration, and grounded QA

**Current non-goals:**
- production SLA
- hosted multi-tenant deployment
- broad domain benchmark coverage
- turnkey package distribution

## Why CAG Over Standard RAG?

To be clear: advanced RAG systems already exist. Production RAG pipelines routinely include reranking, query decomposition, guardrails, and other improvements over naive retrieval. CAG is not claiming that all RAG is simplistic.

The distinction is architectural, not incremental. CAG is not "RAG with extra steps." It is a different paradigm.

In standard RAG -- even advanced RAG -- retrieval is a preprocessing step: documents are fetched and then passed to a generator. The retrieval layer does not reason about what it is doing. Guardrails and rerankers are usually bolted on as separate stages rather than integrated into a single reasoning loop.

CAG uses "Cognitive" in the specific sense of meta-cognition: the system reasons about its own reasoning. The orchestration layer is not a passive pipeline -- it is part of the system's intelligence. It:

- **classifies the query type** before deciding how to search
- **selects and adapts the retrieval strategy** based on what kind of question is being asked
- **evaluates evidence quality** after retrieval, surfacing gaps before generation begins
- **validates the generated answer** against the evidence and can refuse to answer when support is insufficient

The control loop is:

`ENTRY -> RETRIEVE -> REFINE -> REASON -> VALIDATE -> EXIT`

Each stage is aware of the others. The system can backtrack, escalate, or refuse -- not because a guardrail caught a bad output, but because the orchestration layer determined the evidence was inadequate.

If you care about grounded answers, explicit failure modes, and benchmarkable retrieval orchestration, CAG is the interesting part. If you only need a minimal QA layer over a vector store, standard RAG may be enough.

### The Difference in Plain Language

RAG answers a question by retrieving relevant chunks and asking the model to answer from them.

CAG first asks a higher-level question:

`What kind of problem is this, what kind of evidence would justify an answer, and is the current evidence strong enough to answer at all?`

That difference matters because the hardest part of grounded QA is often not generation itself. It is deciding:
- what to retrieve
- whether the retrieved material is enough
- how much confidence is justified
- when the system should stop and refuse instead of guessing

RAG improves generation with retrieved context.

CAG tries to improve the decision process around retrieval, reasoning, and validation.

### CAG vs RAG at a Glance

| Area | Standard RAG | CAG |
| --- | --- | --- |
| Paradigm | retrieval-enhanced generation | self-aware retrieval reasoning |
| Core flow | `retrieve -> generate` | `ENTRY -> RETRIEVE -> REFINE -> REASON -> VALIDATE -> EXIT` |
| Query handling | one generic path | query-typed path selection |
| Retrieval behavior | static top-k retrieval | strategy-aware retrieval and refinement |
| Grounding control | prompt-only | explicit validation and escalation |
| Failure mode | can answer with weak support | can refuse or escalate unsupported requests |
| Advanced RAG | reranking, query decomposition, and guardrails exist but are usually bolted on as separate stages | these capabilities are architecturally integrated into the reasoning loop |
| Benchmark story | usually ad hoc | built-in evaluation harness |

### Practical Advantages of CAG

- **Better control over failure modes**: CAG can refuse or escalate unsupported requests instead of answering with weak confidence.
- **More appropriate retrieval behavior**: procedural, diagnostic, and factual questions do not have to share the same retrieval path.
- **Stronger grounding discipline**: evidence is refined and validated before the answer is treated as trustworthy.
- **More interpretable orchestration**: the graph makes it easier to inspect how the system reached an answer or why it declined.
- **More meaningful evaluation**: the benchmark harness lets you compare orchestration strategies, not only model outputs.

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

## Current Maturity

Solid today:
- graph-based CAG runtime
- generic English-first public API
- benchmark harness with multiple systems
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
cag eval --system rag_baseline --limit 3
cag compare --runs ./artifacts/eval_runs/<cag_run> ./artifacts/eval_runs/<rag_run>
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

### 3. Use your own corpus

Place your own documents (`.pdf`, `.txt`, `.md`) in `data/raw/`. The repository does not ship with sample documents.

### 4. Ingest documents

```bash
cag ingest --data-dir ./data/raw
```

### 5. Run tests

```bash
pip install -e ".[dev,eval]"
pytest
```

### 6. Run a smoke benchmark

```bash
cag eval --system cag --limit 3
```

### 7. Optionally run the UI

Recommended preview UI:

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

## Benchmarking

Benchmarking is a first-class feature of this preview.

Available systems:
- `cag`
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
- the default benchmark is still small
- `cost_estimate` is a relative estimate, not billing truth
- results support preview-stage claims about this implementation, not universal claims about all CAG systems

## First-Run Troubleshooting

Missing API key:
- verify `.env` exists
- verify the provider-specific key is set
- LightRAG currently expects an OpenAI-backed path

No documents found:
- add files under `data/raw/`
- confirm supported extensions: `.pdf`, `.txt`, `.md`

Vector DB path issues:
- check `CHROMA_PERSIST_DIR`
- delete and rebuild local preview indexes only if you intentionally want a fresh local state

LightRAG issues:
- make sure `OPENAI_API_KEY` is set
- use the generic benchmark first before trying the LightRAG path

Frontend/API mismatch:
- start FastAPI on `http://localhost:8000`
- start the React preview from `frontend/`

## Repository Structure

```text
src/cag/
  agents/        # Retrieval and reasoning agents
  api/           # FastAPI upload and query endpoints
  eval/          # Benchmark runner, scoring, comparison, datasets
  graph/         # LangGraph state, nodes, and execution
  ingestion/     # Loader, chunker, embeddings, vector store integration
  ui/            # Streamlit interface
docs/            # Public architecture and project docs
frontend/        # React preview UI
tests/           # Unit and harness tests
data/raw/        # Source documents for ingestion
artifacts/       # Generated benchmark artifacts
data/chroma_db/  # Local Chroma runtime data
```

## Convenience Scripts

The `.bat` files (Windows) and `.sh` files (macOS/Linux) in the repo are convenience launchers for local use. They are **not** the primary documented install or validation path for this preview.

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

## License

This repository includes a top-level [LICENSE](LICENSE).
