# Checkout Latency Runbook

## Symptoms

Checkout latency is high when p95 checkout requests exceed 1200 ms for more than 10 minutes.

## First Checks

Check payment provider status first.

Check database connection pool saturation.

Check recent deployments touching checkout, pricing, or payment authorization.

## Escalation

Escalate to the incident commander if p95 remains above 1200 ms for 15 minutes after rollback or mitigation.

