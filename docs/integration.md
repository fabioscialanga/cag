# Integration Guide

This guide shows the fastest ways to integrate CAG into existing applications.

## 1) Python In-Process Integration

Use CAG directly from Python when your app already runs in the same environment.

```python
from cag.graph.graph import run_query

result = run_query(
    query="What is the minimum RAM required?",
    conversation_history=[],
)

print(result["answer"])
print(result["citations"])
```

Good fit:
- backend services in Python
- internal tools and prototypes
- low-latency local orchestration

## 2) HTTP API Integration

Run the API:

```bash
python -m uvicorn cag.api.upload:app --host 0.0.0.0 --port 8000
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the minimum RAM required?\"}"
```

If `CAG_API_KEY` is configured, add:

```http
X-API-Key: your-key
```

Good fit:
- polyglot stacks (Node, Java, .NET, Go, etc.)
- frontend + backend architectures
- containerized deployments

## 3) Node.js (Fetch) Example

```js
const response = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    // "X-API-Key": process.env.CAG_API_KEY,
  },
  body: JSON.stringify({
    query: "How is this workflow configured?",
  }),
});

const data = await response.json();
console.log(data.answer);
```

## 4) Docker Compose Local Stack

Use the repository `docker-compose.yml` for API + frontend local preview.

```bash
docker compose up --build
```

## 5) Integration Checklist

- Keep source docs local in `data/raw/` (do not commit private docs).
- Configure `.env` from `.env.example`.
- Set thresholds (`RELEVANCE_THRESHOLD`, `CONFIDENCE_THRESHOLD`, `HALLUCINATION_THRESHOLD`) per domain risk.
- Track `should_escalate`, `fallback_used`, and `node_trace` in your observability stack.
- Run small benchmark smoke checks before release (`cag eval --system cag --limit 3 --judge-mode off`).

## 6) Attribution Request

CAG is MIT-licensed. If you use it publicly, attribution is appreciated:

`Built with CAG by Fabio Scialanga`

See [NOTICE](../NOTICE) for details.
