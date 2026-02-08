# QA Test Harness Regression Detected

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `QARegressionDetected` | Critical | Regression detected (output hash differs from baseline) |
| `QAGoldenFileMismatch` | Warning | Golden file mismatch (output differs from approved snapshot, >3 in 15m) |
| `QAZeroHallucinationFailure` | Critical | Agent output contains hallucinated data not traceable to inputs |
| `QADeterminismFailure` | Critical | Agent produces different outputs for identical inputs |
| `QAPerformanceThresholdBreach` | Warning | Agent operation exceeds performance baseline threshold (>3 in 15m) |

**Thresholds:**

```promql
# QARegressionDetected
# Any regression detected in agent output
increase(gl_qa_test_harness_regressions_detected_total[5m]) > 0

# QAGoldenFileMismatch
# More than 3 golden file mismatches in 15 minutes
increase(gl_qa_test_harness_golden_file_mismatches_total[15m]) > 3
# sustained for 10 minutes

# QAZeroHallucinationFailure
# Any zero-hallucination test failure
increase(gl_qa_test_harness_test_runs_total{category="zero_hallucination", status="failed"}[5m]) > 0

# QADeterminismFailure
# Any determinism test failure
increase(gl_qa_test_harness_test_runs_total{category="determinism", status="failed"}[5m]) > 0

# QAPerformanceThresholdBreach
# More than 3 performance threshold breaches in 15 minutes
increase(gl_qa_test_harness_performance_breaches_total[15m]) > 3
# sustained for 10 minutes
```

---

## Description

These alerts fire when the QA Test Harness (AGENT-FOUND-009) detects quality issues in agent outputs, indicating that one or more GreenLang agents may be producing incorrect, non-deterministic, or degraded results. The QA Test Harness is a critical quality gate for all GreenLang Climate OS calculations, required for regulatory compliance under CSRD, CBAM, ISO 14064, and SOC 2.

### How Regression Detection Works

The QA Test Harness maintains a multi-layered quality verification system:

1. **Regression Baselines** -- For each agent, the expected output hash is recorded for known inputs. When a test produces a different output hash for the same input, a regression is detected. This catches any change in agent behavior, whether intentional or accidental.

2. **Golden File Comparison** -- Approved output snapshots (golden files) are stored as reference. Agent outputs are compared against these snapshots with configurable tolerance. Mismatches may indicate intentional changes requiring golden file updates, or unintended regressions.

3. **Zero-Hallucination Verification** -- Every value in an agent's output is traced back to input data or known reference sources. If any output value cannot be traced, it is flagged as hallucinated data -- a critical data integrity violation for climate calculations.

4. **Determinism Testing** -- The same agent is executed N times with identical inputs. All outputs must be bit-for-bit identical. If outputs differ, the agent has non-deterministic behavior, which may be caused by:
   - `random_state`: Unseeded or improperly seeded random number generators
   - `parallel_execution`: Thread/process scheduling differences
   - `floating_point`: Floating-point accumulation order differences
   - `system_time`: System clock dependencies
   - `hash_ordering`: Dictionary/set ordering differences

5. **Performance Benchmarking** -- Agent operations are benchmarked against established baselines (min, max, mean, median, p95, p99). Breaches indicate performance regression from recent changes.

### Regulatory Requirements

- **CSRD**: Requires verifiable, auditable calculation methodology
- **CBAM**: Requires consistent emission factor application across reporting periods
- **ISO 14064**: Requires documented and reproducible GHG quantification
- **SOC 2**: Requires data integrity controls and change detection

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Agent outputs may be incorrect or inconsistent; reports may show wrong values |
| **Data Impact** | Critical | Output data integrity is compromised for affected agents |
| **SLA Impact** | High | Quality verification SLA targets violated |
| **Revenue Impact** | High | Compliance-sensitive customers require verified, accurate calculations |
| **Compliance Impact** | Critical | Regulatory requirements for accurate, traceable calculations violated |
| **Downstream Impact** | High | All agents consuming regressed outputs may produce cascading errors |

---

## Symptoms

- `gl_qa_test_harness_regressions_detected_total` counter is incrementing
- `gl_qa_test_harness_golden_file_mismatches_total` counter is incrementing
- `gl_qa_test_harness_test_runs_total{category="zero_hallucination", status="failed"}` is elevated
- `gl_qa_test_harness_test_runs_total{category="determinism", status="failed"}` is elevated
- `gl_qa_test_harness_performance_breaches_total` counter is incrementing
- Grafana QA Test Harness dashboard shows elevated failure rates
- Test runs are failing with status "failed" at elevated rates
- CI/CD pipelines are failing at the QA verification gate
- Agent outputs produce different values when re-run with identical inputs

---

## Diagnostic Steps

### Step 1: Identify the Scope of the Regression

```bash
# Check recent regression detections via API
curl -s http://localhost:8080/v1/regressions?limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check which agents are affected
curl -s http://localhost:8080/v1/runs?status=failed&limit=20 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check regression metrics by agent type
kubectl port-forward -n greenlang svc/qa-test-harness-service 8080:8080
curl -s http://localhost:8080/metrics | grep gl_qa_test_harness_regressions
```

### Step 2: Identify the Failure Category

```bash
# Check which test categories are failing
curl -s http://localhost:8080/metrics | grep gl_qa_test_harness_test_runs_total

# Check if this is a zero-hallucination failure
curl -s "http://localhost:8080/v1/runs?category=zero_hallucination&status=failed&limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check if this is a determinism failure
curl -s "http://localhost:8080/v1/runs?category=determinism&status=failed&limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check if this is a golden file mismatch
curl -s "http://localhost:8080/v1/runs?category=golden_file&status=failed&limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 3: Examine Failing Test Details

```bash
# Get detailed failure information for a specific test run
curl -s "http://localhost:8080/v1/runs/<run_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get assertion details for the failing test run
curl -s "http://localhost:8080/v1/runs/<run_id>/assertions" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Compare input/output hashes with baseline
curl -s "http://localhost:8080/v1/regression/baselines?agent_type=<agent_type>&is_active=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Check Golden File Diffs

```bash
# Get the golden file comparison details
curl -s "http://localhost:8080/v1/golden-files?agent_type=<agent_type>&is_active=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get the diff report for a specific golden file mismatch
curl -s "http://localhost:8080/v1/golden-files/<file_id>/diff?run_id=<run_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 5: Check Performance Baselines

```bash
# Check performance baselines for the affected agent
curl -s "http://localhost:8080/v1/performance/baselines?agent_type=<agent_type>&is_active=true" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check if performance breaches are occurring
curl -s http://localhost:8080/metrics | grep gl_qa_test_harness_performance
```

### Step 6: Check Recent Deployments

```bash
# Check recent deployment history for the affected agent
kubectl rollout history deployment/<agent-deployment> -n greenlang

# Check if the regression correlates with a recent deployment
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep -i "deploy\|image\|rolling" | tail -20
```

---

## Resolution Steps

### Scenario 1: Regression After Agent Deployment

**Symptoms:** Regression detected immediately after a new agent version was deployed. Output hash differs from baseline.

**Resolution:**

1. Identify the affected deployment:
```bash
kubectl rollout history deployment/<agent-deployment> -n greenlang
```

2. Rollback the affected agent to the previous version:
```bash
kubectl rollout undo deployment/<agent-deployment> -n greenlang
kubectl rollout status deployment/<agent-deployment> -n greenlang
```

3. Verify the regression is resolved:
```bash
curl -s -X POST http://localhost:8080/v1/suites/<suite_id>/run \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  | python3 -m json.tool
```

4. Investigate the root cause in the reverted code before re-deploying.

### Scenario 2: Golden File Update Required (Intentional Change)

**Symptoms:** Golden file mismatch after an intentional change to agent behavior (updated emission factors, new calculation methodology, etc.).

**Resolution:**

1. Review the golden file diff to confirm the change is expected:
```bash
curl -s "http://localhost:8080/v1/golden-files/<file_id>/diff?run_id=<run_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Update the golden file with the new expected output:
```bash
curl -s -X POST "http://localhost:8080/v1/golden-files/<file_id>/update" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"run_id": "<run_id>", "description": "Updated after <change_reason>"}' \
  | python3 -m json.tool
```

3. Update the regression baseline:
```bash
curl -s -X POST "http://localhost:8080/v1/regression/baselines" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "<agent_type>",
    "input_hash": "<input_hash>",
    "output_hash": "<new_output_hash>"
  }' | python3 -m json.tool
```

### Scenario 3: Zero-Hallucination Failure

**Symptoms:** Agent output contains values that cannot be traced to input data or known reference sources.

**Resolution:**

1. **Immediately quarantine affected outputs.** Flag all outputs from the affected agent as "unverified" in downstream systems.

2. Identify the hallucinated values:
```bash
curl -s "http://localhost:8080/v1/runs/<run_id>/assertions" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for a in data.get('items', data if isinstance(data, list) else []):
    if not a.get('passed'):
        print(f\"Assertion: {a.get('name')}\")
        print(f\"  Expected: {a.get('expected')}\")
        print(f\"  Actual: {a.get('actual')}\")
        print(f\"  Message: {a.get('message')}\")
"
```

3. Investigate the agent's data processing logic for the source of hallucinated values.

4. Fix the agent and re-run the zero-hallucination test suite before redeploying.

### Scenario 4: Determinism Failure

**Symptoms:** Agent produces different outputs when executed with identical inputs.

**Resolution:**

1. Check the determinism test details:
```bash
curl -s "http://localhost:8080/v1/runs/<run_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Common causes and fixes:
   - **Unseeded RNG**: Add explicit seed configuration with `global_seed=42`
   - **Parallel execution**: Ensure deterministic ordering in parallel operations
   - **Floating-point accumulation**: Use Kahan summation or sort before accumulation
   - **Dictionary ordering**: Use OrderedDict or sorted keys
   - **System time**: Remove timestamp dependencies from calculation logic

3. Fix the non-determinism source and verify with repeated test execution.

### Scenario 5: Performance Regression

**Symptoms:** Agent operations exceed their performance baselines.

**Resolution:**

1. Check the performance baseline comparison:
```bash
curl -s "http://localhost:8080/v1/performance/baselines?agent_type=<agent_type>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. If the performance change is expected (e.g., increased data volume):
   - Update the performance baselines with new thresholds
   - Document the reason for the baseline update

3. If the performance change is unexpected:
   - Profile the agent to identify the bottleneck
   - Check for new database queries, cache misses, or algorithmic changes
   - Optimize or rollback the causing change

---

## Post-Incident Steps

### Step 1: Verify Regression Is Resolved

```bash
# Check that no new regressions are being detected
curl -s http://localhost:8080/metrics | grep gl_qa_test_harness_regressions_detected_total

# Run the full test suite for the affected agent
curl -s -X POST http://localhost:8080/v1/suites/<suite_id>/run \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  | python3 -m json.tool
```

### Step 2: Update Baselines If Needed

```bash
# Deactivate old baselines that are no longer valid
curl -s -X PATCH "http://localhost:8080/v1/regression/baselines/<baseline_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}' \
  | python3 -m json.tool

# Create new baselines with updated expected output
curl -s -X POST "http://localhost:8080/v1/regression/baselines" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "<agent_type>",
    "input_hash": "<input_hash>",
    "output_hash": "<new_expected_output_hash>"
  }' | python3 -m json.tool
```

### Step 3: Verify Audit Trail Integrity

```promql
# Verify audit events are being recorded
increase(gl_qa_test_harness_audit_events_total[5m])

# Check for audit gaps
(
  sum(rate(gl_qa_test_harness_test_runs_total[5m]))
  - sum(rate(gl_qa_test_harness_audit_events_total[5m]))
)
```

---

## Interim Mitigation

While the regression is being investigated:

1. **Affected agent outputs should be flagged.** Any outputs produced by the regressed agent should be marked as "pending verification" in downstream reports and dashboards.

2. **Golden files can be temporarily relaxed.** If the mismatch is known to be caused by an intentional change, temporarily increase the comparison tolerance while new golden files are approved.

3. **CI/CD pipelines can skip the affected test.** If the regression is in a non-critical agent and a fix is in progress, the specific test can be temporarily marked as `skip: true` in the test case definition.

4. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#compliance-ops` -- compliance impact
   - `#data-pipeline-ops` -- affected pipeline notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Golden file mismatch or performance breach, investigation in progress | On-call engineer | 15 minutes |
| L2 | Regression or determinism failure detected, agent outputs affected | Platform team lead + #platform-foundation | Immediate (<5 min) |
| L3 | Zero-hallucination failure detected, data integrity compromised | Platform team + compliance team + CTO notification | Immediate |
| L4 | Systemic regression affecting multiple agents, all outputs suspect | All-hands engineering + incident commander + executive notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** QA Test Harness Health (`/d/qa-test-harness-service`)
- **Alerts:** `QARegressionDetected`, `QAGoldenFileMismatch`, `QAZeroHallucinationFailure`, `QADeterminismFailure`
- **Key metrics to watch:**
  - `gl_qa_test_harness_regressions_detected_total` (should be 0)
  - `gl_qa_test_harness_golden_file_mismatches_total` (should be 0 or near-zero)
  - `gl_qa_test_harness_test_runs_total{category="zero_hallucination", status="failed"}` (should be 0)
  - `gl_qa_test_harness_test_runs_total{category="determinism", status="failed"}` (should be 0)
  - `gl_qa_test_harness_performance_breaches_total` (should be 0 or near-zero)
  - `gl_qa_test_harness_test_runs_total{status="failed"}` rate (should be < 15%)
  - `gl_qa_test_harness_coverage_percent` (should be > 80% for all agents)

### Best Practices

1. **Run the QA test suite** before every agent deployment as a CI/CD gate
2. **Maintain golden files** and update them with every intentional change
3. **Set regression baselines** for all agent input/output combinations
4. **Run determinism tests** with at least 5 iterations for every agent
5. **Enforce zero-hallucination** checks for all agents producing climate calculation data
6. **Track coverage trends** and ensure coverage never drops below 80%
7. **Benchmark performance** baselines after every significant agent change
8. **Review test failure trends** weekly to catch gradual quality degradation

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `QARegressionDetected` | Critical | Regression detected (output hash differs from baseline) |
| `QAGoldenFileMismatch` | Warning | Golden file mismatch (>3 in 15m) |
| `QAZeroHallucinationFailure` | Critical | Agent output contains hallucinated data |
| `QADeterminismFailure` | Critical | Agent produces non-deterministic output |
| `QAPerformanceThresholdBreach` | Warning | Performance baseline exceeded (>3 in 15m) |
| `QAHighTestFailureRateWarning` | Warning | >15% test failure rate |
| `QAHighTestFailureRateCritical` | Critical | >30% test failure rate |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 regression detection incident
- **Related runbooks:** [QA Test Harness Service Down](./qa-test-harness-service-down.md)
