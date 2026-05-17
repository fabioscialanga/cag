# Incident Runbook Example

Use this folder as a tiny runbook corpus.

```bash
cag ingest --data-dir ./examples/incident_runbook --reset
cag query "What should I check when checkout latency is high?" --json
cag query "How do I recover the analytics warehouse?" --json
```

The analytics warehouse question is intentionally unsupported by the sample document.

