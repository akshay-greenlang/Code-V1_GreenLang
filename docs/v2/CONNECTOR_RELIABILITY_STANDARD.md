# GreenLang V2 Connector Reliability Standard

## Reliability Baseline

Every production connector must declare:

- retry policy (`max_attempts`, `backoff`, retryable codes)
- timeout budget (connect/read/overall)
- idempotency strategy
- circuit-breaker thresholds
- dead-letter handling strategy

## SLO Targets

- availability: >= 99.9%
- p95 connector latency: <= 2s (interactive), <= 10s (batch)
- retry recovery success: >= 95% for transient failures
- duplicate-write rate: 0 for idempotent operations

## Failure Classes

- transient (retryable)
- throttling/rate-limit (retry with bounded backoff)
- permanent configuration error (fail fast)
- authz/authn failure (security escalation)
- downstream schema mismatch (contract escalation)

## Required Tests

1. timeout and retry behavior test.
2. idempotency test for write operations.
3. circuit-breaker open/half-open behavior test.
4. dead-letter routing for terminal failures.

## Release Gate

Connector changes are blocked from release trains unless reliability checks and SLO assertions pass.
