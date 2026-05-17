# DB-First Knowledge Compiler

This is the design target for adapting the "LLM Wiki" idea to CAG without making markdown files the source of truth.

The core principle:

Raw documents remain immutable evidence. The database stores a compiled, queryable knowledge layer with provenance back to the raw evidence.

## First Database Target

Start with SQLite.

Why:

- local-first
- easy quickstart
- no service dependency
- works in CI
- can later map cleanly to Postgres

Postgres should be the production target after the schema and query flow stabilize.

## Schema

### sources

One row per original document or uploaded source.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `filename` | text | display name |
| `source_uri` | text | local path, upload ID, or URL |
| `mime_type` | text | detected content type |
| `sha256` | text | immutable content hash |
| `created_at` | text | ISO timestamp |
| `ingest_status` | text | `pending`, `indexed`, `failed` |

### source_versions

Tracks changes when the same logical source is replaced.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `source_id` | text | references `sources.id` |
| `version` | integer | monotonic version |
| `sha256` | text | version hash |
| `created_at` | text | ISO timestamp |

### chunks

Raw chunk evidence.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `source_version_id` | text | references `source_versions.id` |
| `chunk_index` | integer | source-local order |
| `content` | text | raw chunk text |
| `domain_module` | text | existing CAG metadata |
| `start_offset` | integer | optional source offset |
| `end_offset` | integer | optional source offset |
| `vector_id` | text | optional Chroma/Pinecone ID |

### claims

Atomic compiled facts.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `claim_text` | text | one fact, procedure step, constraint, or rule |
| `claim_type` | text | `definition`, `procedure`, `configuration`, `diagnostic`, `policy`, `constraint` |
| `confidence` | real | compiler confidence |
| `status` | text | `active`, `stale`, `contradicted`, `retracted` |
| `created_at` | text | ISO timestamp |
| `updated_at` | text | ISO timestamp |

### claim_evidence

Provenance from compiled knowledge back to raw chunks.

| Column | Type | Notes |
| --- | --- | --- |
| `claim_id` | text | references `claims.id` |
| `chunk_id` | text | references `chunks.id` |
| `support_type` | text | `supports`, `contradicts`, `mentions` |
| `evidence_quote` | text | short quote or span |

### entities

Named concepts extracted from the corpus.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `name` | text | canonical name |
| `entity_type` | text | `product`, `module`, `setting`, `role`, `error`, `api`, `concept` |
| `aliases` | text | JSON array |

### topics

Compiled wiki-like pages stored as rows.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `slug` | text unique | stable topic slug |
| `title` | text | display title |
| `summary` | text | concise synthesized overview |
| `status` | text | `active`, `stale`, `needs_review` |
| `updated_at` | text | ISO timestamp |

### topic_claims

Maps topic pages to claims.

| Column | Type | Notes |
| --- | --- | --- |
| `topic_id` | text | references `topics.id` |
| `claim_id` | text | references `claims.id` |
| `rank` | integer | order within topic |

### contradictions

Explicit contradiction records.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `left_claim_id` | text | references `claims.id` |
| `right_claim_id` | text | references `claims.id` |
| `reason` | text | short explanation |
| `status` | text | `open`, `resolved`, `accepted` |
| `created_at` | text | ISO timestamp |

### knowledge_log

Append-only audit log.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `event_type` | text | `ingest`, `compile`, `lint`, `query`, `update` |
| `payload` | text | JSON payload |
| `created_at` | text | ISO timestamp |

### saved_answers

Optional saved answer records for review and regression tests.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | text primary key | stable UUID |
| `query` | text | user question |
| `answer` | text | generated answer |
| `claim_ids` | text | JSON array |
| `citation_chunk_ids` | text | JSON array |
| `created_at` | text | ISO timestamp |

## Query Flow

1. Classify the query with the existing `ENTRY` node.
2. Search compiled topics, entities, and claims with SQL full-text search.
3. Search raw chunks as fallback or supporting evidence.
4. Merge compiled candidates and raw chunks into the existing `SELECT_CONTEXT` stage.
5. Prefer compiled claims only when they have evidence links.
6. Return citations to raw chunks, not only compiled summaries.
7. Escalate when claims are stale, contradicted, or unsupported.

## First Implementation Slice

1. Add SQLite connection and migrations.
2. Store `sources`, `source_versions`, and `chunks` during ingest.
3. Add a deterministic claim extractor interface with a stub implementation.
4. Add `claims` and `claim_evidence`.
5. Add a `compiled_search` function returning LangChain-compatible `Document` objects.
6. Add `cag_compiled` to the eval harness.

## Lint Checks

The first lint API is available as:

```python
from cag.knowledge.lint import lint_knowledge

report = lint_knowledge("./data/knowledge.db")
print(report.model_dump())
```

Current checks:

- stale claims
- claims with missing evidence
- orphan topics
- simple opposite-polarity contradictions

Contradictions detected by lint are recorded in the `contradictions` table and the run is logged in `knowledge_log`.

## Non-Negotiables

- Every compiled claim needs provenance.
- Raw chunk retrieval remains available as fallback.
- Contradictions are modeled explicitly.
- Compiled knowledge can be stale; validation must know that.
