# Factors API Load Test Report (DEP12)

**Run date:** _YYYY-MM-DD UTC_
**Run by:** _name / oncall rotation_
**Target env:** _staging (must be staging)_
**Git SHA on target:** _git rev-parse HEAD_
**Image tag:** _ghcr.io/greenlang/factors-api:<tag>_

## Scenarios executed

| Scenario | Tag | Duration | Users | Spawn rate | Target rps |
|----------|-----|----------|-------|------------|-----------:|
| Single resolve | `resolve` | 10 min | 1000 | 50/s | 1000 |
| Batch 10k rows | `batch`   | 10 min | 100  | 10/s | 100  |

## Results

### `/v1/resolve` (single row)

| Metric | Observed | Threshold | Pass/Fail |
|--------|---------:|----------:|:---------:|
| RPS sustained | _fill_ | >= 1000 | _fill_ |
| p50 latency (ms) | _fill_ | <= 80 | _fill_ |
| p95 latency (ms) | _fill_ | <= 500 | _fill_ |
| p99 latency (ms) | _fill_ | <= 1500 | _fill_ |
| Error rate (5xx) | _fill_ | <= 0.1% | _fill_ |
| 429 rate (rate-limit) | _fill_ | informational | _fill_ |

### `/v1/batch/resolve` (10k rows)

| Metric | Observed | Threshold | Pass/Fail |
|--------|---------:|----------:|:---------:|
| RPS sustained | _fill_ | >= 100 | _fill_ |
| p50 latency (ms) | _fill_ | <= 1500 | _fill_ |
| p95 latency (ms) | _fill_ | <= 4000 | _fill_ |
| p99 latency (ms) | _fill_ | <= 6500 | _fill_ |
| Error rate (5xx) | _fill_ | <= 0.5% | _fill_ |

## Infra observations

- HPA max replicas reached: _yes/no_.
- CPU utilization at peak: _%_.
- Memory utilization at peak: _%_.
- Postgres connections: _count_ / _max_.
- Redis hit rate during batch burst: _%_.
- Any AnalysisTemplate would have aborted rollouts? _yes/no_.

## Follow-ups

- [ ] _Capture action items from observed thresholds (HPA tuning, query
  optimization, etc.)._

## Sign-off

| Role | Name | Date |
|------|------|------|
| SRE owner | | |
| Eng Mgr | | |
