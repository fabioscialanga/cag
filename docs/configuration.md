# Configuration

CAG reads configuration from environment variables, with `.env` support through Pydantic settings.

Start from:

```bash
cp .env.example .env
```

## Install Modes

| Flow | Install |
| --- | --- |
| CLI/query only | `pip install -e .` |
| Tests | `pip install -e ".[dev]"` |
| API | `pip install -e ".[api]"` |
| Evaluation | `pip install -e ".[eval]"` |
| Streamlit UI | `pip install -e ".[ui]"` |
| LightRAG baseline | `pip install -e ".[lightrag]"` |
| Everything | `pip install -e ".[all]"` |

## Core Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `groq`, or `ollama` |
| `OPENAI_API_KEY` | empty | Required for OpenAI flows |
| `OPENAI_MODEL` | `gpt-4o` | Generation model |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `EMBEDDING_DIM` | `3072` | Embedding vector dimension |
| `VECTOR_DB` | `chroma` | `chroma` or `pinecone` |
| `KNOWLEDGE_DB_PATH` | `./data/knowledge.db` | SQLite path for compiled knowledge |
| `ENABLE_KNOWLEDGE_COMPILER` | `true` | Write compiled knowledge during ingest |
| `RETRIEVAL_TOP_K` | `20` | Default retrieval budget |
| `RELEVANCE_THRESHOLD` | `0.7` | Evidence sufficiency threshold |
| `MODERATE_RELEVANCE_THRESHOLD` | `0.55` | Secondary support threshold |
| `CONFIDENCE_THRESHOLD` | `0.6` | Minimum accepted answer confidence |
| `HALLUCINATION_THRESHOLD` | `0.3` | Maximum accepted hallucination risk |
| `MAX_REASON_RETRIES` | `2` | Reasoning retries before escalation |
| `FAST_CONTEXT_SELECTION` | `true` | Skip the RetrievalAgent LLM when deterministic evidence scoring is strong |
| `FAST_CONTEXT_MIN_SCORE` | `4.0` | Minimum lexical evidence score required for fast context selection |
| `ENABLE_CONVERSATION_ROUTER_LLM` | `false` | Use an LLM to route ambiguous conversational turns; deterministic routing is faster |
| `CHUNK_SIZE` | `1200` | Ingest chunk size in characters |
| `CHUNK_OVERLAP` | `180` | Ingest chunk overlap in characters |
| `CONTEXT_SELECTION_LIMIT` | `8` | Max selected context chunks for general queries |
| `COMPLEX_CONTEXT_SELECTION_LIMIT` | `10` | Max selected context chunks for procedural, diagnostic, and configuration queries |
| `ADAPTIVE_RETRIEVAL_RETRY` | `true` | Retry retrieval once with expanded query/top-k before retrying generation |
| `ADAPTIVE_RETRY_TOP_K_BOOST` | `5` | Top-k increase for adaptive retrieval retry |
| `HYBRID_LEXICAL_RETRIEVAL` | `true` | Enable hybrid retrieval (vector + lexical local candidates) when Document Map candidates are not found |
| `HYBRID_LEXICAL_TOP_K` | `8` | Max lexical candidates merged into retrieval before dedupe/rerank |
| `STRICT_FAST_PROFILE` | `false` | Apply a speed-focused profile while keeping validation/escalation enabled |
| `LOG_LEVEL` | `INFO` | Lowercase values are normalized |
| `CAG_API_KEY` | empty | Optional API protection |

Notes:
- Runtime defaults come from `src/cag/config.py`.
- `.env.example` is an opinionated starter profile and may choose a different model value than code defaults.

## STRICT_FAST Profile

Set:

```env
STRICT_FAST_PROFILE=true
```

When enabled, CAG applies runtime speed overrides:
- `RETRIEVAL_TOP_K` capped to `10`
- `CONTEXT_SELECTION_LIMIT` capped to `6`
- `COMPLEX_CONTEXT_SELECTION_LIMIT` capped to `8`
- `ADAPTIVE_RETRIEVAL_RETRY=false`
- `HYBRID_LEXICAL_RETRIEVAL=false`
- fast deterministic context path kept enabled

Rigor is preserved:
- `VALIDATE` remains active
- confidence/hallucination thresholds still gate final answers
- escalation remains enabled when evidence is weak

## Local-First Option

For local generation without a hosted LLM provider, configure Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Embeddings still need a configured embedding backend unless you adapt the embedding provider.
