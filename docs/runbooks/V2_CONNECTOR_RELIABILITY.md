# Runbook: V2 Connector Reliability

## Scope

Operational runbook for connector incidents in V2 release lanes.

## Immediate Triage

1. identify connector and impacted app profiles.
2. inspect retry counters and timeout distribution.
3. verify circuit-breaker state.
4. check dead-letter queue growth.

## Mitigation

- transient spike: reduce concurrency and allow retries to drain.
- sustained auth failures: rotate credentials and revalidate policy scopes.
- schema drift: disable connector lane and route to fallback ingestion path.
- downstream outage: open circuit breaker and activate degraded mode.

## Recovery Validation

- p95 latency returns to SLO target window.
- retry recovery rate above threshold.
- no idempotency violations in post-incident audit.
- dead-letter backlog drained and replayed with audit evidence.

## Escalation

- Severity 1: SRE on-call + Security Council (if auth/data boundary impact).
- Severity 2: App Reliability Team + owning app team.
