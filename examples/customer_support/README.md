# Customer Support Example

Use this folder as a tiny support-document corpus.

```bash
cag ingest --data-dir ./examples/customer_support --reset
cag query "How do I reset an API token?" --json
cag query "Can customers enable biometric login?" --json
```

The second query is intentionally unsupported by the sample document, so CAG should prefer escalation over fabrication.

