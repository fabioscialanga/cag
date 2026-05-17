# Basic CLI Example

This example uses the bundled benchmark corpus.

## Run

```bash
pip install -e ".[dev,eval]"
cag ingest --data-dir ./data/benchmark_corpus --reset
cag query "What is the minimum RAM required to run Nexus Platform?" --json
```

## Expected Shape

The query output is JSON with:

- `answer`
- `confidence`
- `query_type`
- `citations`
- `should_escalate`
- `node_trace`

