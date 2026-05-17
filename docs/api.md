# API

CAG exposes a FastAPI preview app at `cag.api.upload:app`.

## Python API

For local Python usage, call the graph directly:

```python
from cag.graph.graph import run_query

result = run_query("What is the minimum RAM required?")
print(result["answer"])
```

The returned dictionary is the final graph state, including answer, citations, selected chunks, gaps, validation fields, and node trace.

Run it locally:

```bash
python -m uvicorn cag.api.upload:app --reload --port 8000
```

## Authentication

If `CAG_API_KEY` is empty, the preview API is open for local use.

If `CAG_API_KEY` is set, send:

```http
X-API-Key: your-key
```

Missing keys return `401`. Invalid keys return `403`.

## POST /query

Request:

```json
{
  "query": "What is the minimum RAM required?",
  "conversation_history": [],
  "relevance_threshold": 0.7,
  "confidence_threshold": 0.6,
  "hallucination_threshold": 0.3
}
```

Only `query` is required. Empty or whitespace-only queries return `422`.

Response includes the final graph state. Important fields:

- `answer`
- `confidence`
- `citations`
- `query_type`
- `ranked_chunks`
- `gaps`
- `hallucination_risk`
- `should_escalate`
- `fallback_used`
- `fallback_reason`
- `node_trace`

## POST /upload

Uploads `.pdf`, `.txt`, or `.md` files into `data/raw/`.

Example:

```bash
curl -X POST http://localhost:8000/upload?ingest=false \
  -H "X-API-Key: $CAG_API_KEY" \
  -F "files=@./my-doc.txt"
```

Response:

```json
{
  "status": "ok",
  "saved": ["my-doc.txt"],
  "ingest_started": false
}
```

Limits:

- max single file size: 10 MiB
- max request size: 25 MiB

## POST /demo/reset

Replaces supported files in `data/raw/` with the bundled benchmark corpus and optionally starts background ingestion. Unsupported files in `data/raw/` are left untouched.

Example:

```bash
curl -X POST "http://localhost:8000/demo/reset?ingest=true" \
  -H "X-API-Key: $CAG_API_KEY"
```

Response:

```json
{
  "status": "ok",
  "copied": ["nexus_api_reference.txt"],
  "ingest_started": true
}
```

## GET /files

Lists supported documents currently in `data/raw/`.

## DELETE /files/{filename}

Deletes a document from `data/raw/` and schedules a background re-index of the remaining files.

## POST /diagnostics/retrieval

Runs only the retrieval side of the graph:

```text
ENTRY -> RETRIEVE -> SELECT_CONTEXT
```

It does not call the reasoning agent and does not generate an answer.

Request:

```json
{
  "query": "What is the minimum RAM required?"
}
```

Response includes:

- `query`
- `query_type`
- `retrieval_strategy`
- `chunks`
- `ranked_chunks`
- `gaps`
- `relevance_score`
- `node_trace`
