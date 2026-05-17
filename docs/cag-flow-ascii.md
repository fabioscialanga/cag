# CAG ASCII Diagrams

## 1) End-to-End Runtime Graph

```text
+------------------+
|      ENTRY       |
| classify, lang   |
+---------+--------+
          |
          v
+------------------+
| CONTEXTUALIZE    |
| rewrite followup |
+---------+--------+
          |
          v
+------------------+
|     RETRIEVE     |
| profiles/vector  |
| lexical hybrid   |
+---------+--------+
          |
          v
+------------------+
|  SELECT_CONTEXT  |
| rank + diversify |
+---------+--------+
          |
          v
+------------------+
|      REASON      |
| grounded answer  |
+---------+--------+
          |
          v
+------------------+
|      REVIEW      |
| tighten output   |
+---------+--------+
          |
          v
+------------------+
|  POST_GROUNDING  |
| support checks   |
+---------+--------+
          |
          v
+------------------+
|     VALIDATE     |
| decide route     |
+----+--------+----+
     |        |
     |        +-------------------------+
     |                                  |
     v                                  v
+-----------+                   +----------------+
|   EXIT    |<------------------| RETRIEVE RETRY |
| answer or |                   | (adaptive)     |
| escalate  |                   +----------------+
+-----------+
     ^
     |
+----------------+
| REASON RETRY   |
| (constrained)  |
+----------------+
```

## 2) Validate Routing Logic

```text
if should_escalate:
    route -> EXIT
elif should_retry_retrieval:
    route -> RETRIEVE
elif should_retry_reason:
    route -> REASON
else:
    route -> EXIT
```

## 3) Retrieval Decision Ladder

```text
1. Document Map (profiles + candidate chunks)
2. Global semantic search (query variants)
3. Optional lexical hybrid merge
4. Dedupe + discriminative rerank
5. Context selection budget
```

## 4) Safety Contract

```text
- Answer only from evidence
- Cite used sources
- Surface gaps
- Escalate when support is weak
```
