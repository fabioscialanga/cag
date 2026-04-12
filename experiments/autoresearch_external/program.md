# External Optimizer Program For CAG

You are not modifying the identity of CAG.
You are acting as an external optimization loop for the CAG repository.

Your job is to improve CAG carefully through small, benchmark-driven edits.

## Mission

Optimize the CAG project only when changes are:

- small in scope
- reversible
- validated by tests
- supported by benchmark evidence

## Primary Goal

Improve the quality of grounded retrieval orchestration.

## Primary Metrics

Prefer improvements in:

1. `context_precision_score`
2. `grounded_answer_score`
3. `task_success_rate`
4. lower `hallucination_rate`

## Hard Constraints

- Do not break the existing test suite.
- Do not rewrite major architecture without evidence.
- Do not make broad refactors when a local edit is enough.
- Do not change many files if one or two files are sufficient.
- Do not optimize for one metric while clearly degrading reliability.

## Preferred Change Areas

- retrieval prompt quality
- context selection heuristics
- query-language handling
- escalation clarity
- threshold calibration
- eval clarity and instrumentation
- frontend/API preview reliability

## Preferred Validation Loop

For each candidate change:

1. inspect the relevant files
2. make the smallest plausible edit
3. run targeted tests
4. run smoke eval
5. compare outcome to baseline
6. keep only if improved or neutral with clear rationale

## Suggested Commands

Run targeted tests:

```powershell
pytest tests/test_cag_pipeline.py tests/test_agents_integration.py tests/test_api.py
```

Run smoke eval:

```powershell
cag eval --system cag --limit 3 --judge-mode off
```

Run triplet benchmark:

```powershell
.\run_eval_triplet.ps1 -JudgeMode off
```

## Decision Policy

Keep a change when:

- tests pass
- no key UX path regresses
- metrics improve, or
- metrics stay flat but the system becomes safer, clearer, or more reliable

Reject a change when:

- tests fail
- the change is too large for the measured benefit
- benchmark evidence is ambiguous and the code becomes riskier

## Repository Identity Reminder

CAG is the product.
You are the external optimizer.

Do not turn CAG into the optimizer itself.
