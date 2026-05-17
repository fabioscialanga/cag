# Policy QA Example

Use this folder as a tiny internal-policy corpus.

```bash
cag ingest --data-dir ./examples/policy_qa --reset
cag query "When must a security incident be escalated?" --json
cag query "What is the laptop reimbursement amount?" --json
```

The reimbursement question is intentionally unsupported by the sample document.

