# Quickstart

This path uses the bundled benchmark corpus, so a first run does not depend on you preparing documents.

## 1. Install

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install CAG with the dev/eval/API extras:

```bash
pip install -e ".[dev,eval,api]"
```

## 2. Configure

Copy the environment template:

```bash
cp .env.example .env
```

Windows:

```powershell
copy .env.example .env
```

Set the provider key for your chosen model. For OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
```

## 3. Run The One-Command Demo

```bash
cag demo --reset --json
```

Expected result: the command ingests the bundled demo corpus, asks the default Nexus Platform RAM question, and prints a JSON answer with `confidence`, `citations`, `should_escalate`, and `node_trace`.

If you already ingested the corpus, reuse the existing index:

```bash
cag demo --skip-ingest --json
```

## 4. Ingest The Demo Corpus Manually

```bash
cag ingest --data-dir ./data/benchmark_corpus --reset
```

Expected result: logs ending with an indexed chunk count.

## 5. Ask A Question

```bash
cag query "What is the minimum RAM required to run Nexus Platform?" --json
```

Expected result: a JSON object with `answer`, `confidence`, `citations`, `should_escalate`, and `node_trace`.

## 6. Run A Smoke Benchmark

```bash
cag eval --system cag --limit 3 --judge-mode off
```

Expected artifacts:

```text
artifacts/eval_runs/<timestamp>_cag/
  run.json
  results.jsonl
```

## 7. Run The API

```bash
python -m uvicorn cag.api.upload:app --reload --port 8000
```

Open:

```text
http://localhost:8000/
```

If the React frontend has not been built, the API redirects to the Vite dev server.
