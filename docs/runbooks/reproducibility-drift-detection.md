# Reproducibility Drift Detection

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `ReproducibilityDriftCritical` | Critical | Critical drift detected (>5% deviation from baseline) |
| `ReproducibilityDriftModerate` | Warning | Moderate drift detected (1-5% deviation, >3 in 15m) |
| `ReproducibilityHashMismatch` | Critical | Artifact hash mismatch (exact output difference) |
| `ReproducibilityNonDeterminismDetected` | Warning | Non-deterministic behavior detected in execution pipeline |
| `ReproducibilityEnvironmentMismatch` | Warning | Environment fingerprint differences between executions |

**Thresholds:**

```promql
# ReproducibilityDriftCritical
# Any critical severity drift detection
increase(gl_reproducibility_drift_detections_total{severity="critical"}[5m]) > 0

# ReproducibilityDriftModerate
# More than 3 moderate drift detections in 15 minutes
increase(gl_reproducibility_drift_detections_total{severity="moderate"}[15m]) > 3
# sustained for 10 minutes

# ReproducibilityHashMismatch
# Any artifact hash mismatch
increase(gl_reproducibility_hash_mismatches_total[5m]) > 0

# ReproducibilityNonDeterminismDetected
# Non-determinism rate > 0.5/s
sum(rate(gl_reproducibility_non_determinism_total[5m])) > 0.5
# sustained for 15 minutes

# ReproducibilityEnvironmentMismatch
# Environment mismatch rate > 0.1/s
sum(rate(gl_reproducibility_environment_mismatches_total[5m])) > 0.1
# sustained for 15 minutes
```

---

## Description

These alerts fire when the Reproducibility Agent (AGENT-FOUND-008) detects deviations in execution outputs, indicating that climate calculations may not be producing consistent, deterministic results. The drift detection system is a critical component of GreenLang's data integrity model, required for regulatory compliance under CSRD, CBAM, ISO 14064, and SOC 2.

### How Drift Detection Works

The Reproducibility Agent maintains a multi-layered drift detection system:

1. **Artifact Hashing** -- Every input and output artifact is hashed using SHA-256 (default) with optional floating-point normalization. Identical inputs should always produce identical hashes. A mismatch indicates non-deterministic behavior.

2. **Drift Baselines** -- Reference output snapshots are established for known-good executions. Subsequent runs are compared against these baselines with configurable thresholds:
   - **None** (0-0.1%): Output is within expected tolerance
   - **Minor** (0.1-1%): Small deviation, typically caused by floating-point precision differences
   - **Moderate** (1-5%): Significant deviation, may indicate updated factors or model changes
   - **Critical** (>5%): Major deviation, calculation results may be unreliable

3. **Non-Determinism Source Detection** -- When non-determinism is detected, the agent identifies the likely source:
   - `floating_point`: Floating-point accumulation order differences
   - `random_state`: Unseeded or improperly seeded random number generators
   - `parallel_execution`: Thread/process scheduling differences affecting output order
   - `system_time`: System clock dependencies in calculations
   - `external_api`: External API responses varying between calls
   - `file_system`: File system ordering differences
   - `network`: Network call timing or response differences
   - `memory_allocation`: Memory allocation patterns affecting output
   - `gpu_computation`: GPU floating-point non-determinism
   - `hash_ordering`: Dictionary/set ordering differences

4. **Environment Fingerprinting** -- Environment differences (Python version, dependency versions, platform, hardware) are detected and flagged as potential sources of non-reproducibility.

5. **Version Manifest Comparison** -- Agent versions, model versions, emission factor versions, and data source versions are compared between executions to detect version-caused differences.

### Regulatory Requirements

- **CSRD**: Requires verifiable, auditable calculation methodology
- **CBAM**: Requires consistent emission factor application across reporting periods
- **ISO 14064**: Requires documented and reproducible GHG quantification
- **SOC 2**: Requires data integrity controls and change detection

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Calculation results may be inconsistent across runs; reports may show different values |
| **Data Impact** | Critical | Output data integrity is compromised; historical comparisons may be unreliable |
| **SLA Impact** | High | Reproducibility SLA targets violated |
| **Revenue Impact** | High | Compliance-sensitive customers require deterministic calculations |
| **Compliance Impact** | Critical | Regulatory requirements for reproducible calculations violated |
| **Downstream Impact** | High | All agents consuming drifted outputs may produce cascading inaccuracies |

---

## Symptoms

- `gl_reproducibility_drift_detections_total{severity="critical"}` gauge is elevated
- `gl_reproducibility_hash_mismatches_total` counter is incrementing
- `gl_reproducibility_non_determinism_total` counter is incrementing
- Grafana Reproducibility dashboard shows drift percentage above thresholds
- Verification runs are failing with status "fail" at elevated rates
- Replay sessions show `output_match=false`
- Environment fingerprint comparisons show mismatches
- Version manifest comparisons show different versions between runs
- Calculation reports show different values when re-run with identical inputs

---

## Diagnostic Steps

### Step 1: Identify the Scope of Drift

```bash
# Check recent drift detections via API
curl -s http://localhost:8080/v1/drift/detections?severity=critical&limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check which baselines are triggering
curl -s http://localhost:8080/v1/drift/baselines?is_active=true \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check drift metrics by severity
kubectl port-forward -n greenlang svc/reproducibility-service 8080:8080
curl -s http://localhost:8080/metrics | grep gl_reproducibility_drift
```

### Step 2: Identify Non-Determinism Sources

```bash
# Check non-determinism source breakdown
curl -s http://localhost:8080/metrics | grep gl_reproducibility_non_determinism

# Check recent verification failures for non-determinism details
curl -s http://localhost:8080/v1/verifications?status=fail&limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Look for specific non-determinism sources in verification details
curl -s "http://localhost:8080/v1/verifications?status=fail&limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for v in data.get('items', data if isinstance(data, list) else []):
    print(f\"Verification {v.get('verification_id', 'unknown')}:\")
    print(f\"  Sources: {v.get('non_determinism_sources', [])}\")
    print(f\"  Details: {json.dumps(v.get('non_determinism_details', {}), indent=4)}\")
"
```

### Step 3: Compare Environment Fingerprints

```bash
# Check recent environment mismatches
curl -s http://localhost:8080/metrics | grep gl_reproducibility_environment_mismatch

# Compare two environment fingerprints
curl -s http://localhost:8080/v1/environments/<fingerprint_id_1> \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

curl -s http://localhost:8080/v1/environments/<fingerprint_id_2> \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Compare Version Manifests

```bash
# Check recent version manifest differences
curl -s http://localhost:8080/v1/manifests?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Compare two version manifests to find version differences
curl -s "http://localhost:8080/v1/manifests/compare?manifest_a=<id1>&manifest_b=<id2>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 5: Check Seed Configurations

```bash
# Verify seed configurations are consistent
curl -s http://localhost:8080/v1/seeds?limit=10 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check if seed hashes match between executions
curl -s http://localhost:8080/metrics | grep gl_reproducibility_seed
```

### Step 6: Run a Replay to Confirm

```bash
# Trigger a replay of the affected execution
curl -s -X POST http://localhost:8080/v1/replay \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"original_execution_id": "<execution_id>"}' \
  | python3 -m json.tool

# Check replay result
curl -s http://localhost:8080/v1/replay/<replay_id> \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

---

## Resolution Steps

### Scenario 1: Floating-Point Accumulation Drift

**Symptoms:** Non-determinism source is `floating_point`, drift is minor (<1%), affects numerical outputs.

**Resolution:**

1. Verify the drift is within acceptable tolerance:
```bash
# Check current tolerance settings
kubectl get configmap reproducibility-service-config -n greenlang -o yaml \
  | grep -i tolerance
```

2. If the drift is within the configured tolerance range, update the baseline:
```bash
curl -s -X POST http://localhost:8080/v1/drift/baselines \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "<baseline_name>",
    "baseline_data": <new_reference_data>,
    "drift_threshold": 0.01
  }' | python3 -m json.tool
```

3. If the drift exceeds acceptable tolerance, investigate the calculation pipeline for accumulation order changes.

### Scenario 2: Version Change Drift

**Symptoms:** Version manifest comparison shows different agent/model/factor versions between runs. Drift correlates with a recent deployment.

**Resolution:**

1. Identify which versions changed:
```bash
curl -s "http://localhost:8080/v1/manifests/compare?manifest_a=<old>&manifest_b=<new>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. If the drift is expected (intentional version update):
   - Update the drift baselines to reflect the new expected output
   - Document the version change and its impact on outputs
   - Close the drift alert

3. If the drift is unexpected (unintended version change):
   - Rollback the affected component to the previous version
   - Investigate how the version change occurred

### Scenario 3: Environment Drift

**Symptoms:** Environment fingerprint mismatch detected, different Python version or dependency versions between runs.

**Resolution:**

1. Identify the environment difference:
```bash
# Compare fingerprints
curl -s "http://localhost:8080/v1/environments/compare?fp_a=<id1>&fp_b=<id2>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. If pods were rescheduled to nodes with different configurations:
   - Verify all nodes have identical base images and Python versions
   - Check for dependency version drift in container images

3. If a dependency was updated:
   - Pin the dependency version in `requirements.txt`
   - Rebuild and redeploy the container image

### Scenario 4: Unseeded Random State

**Symptoms:** Non-determinism source is `random_state`, outputs vary between runs with identical inputs.

**Resolution:**

1. Check seed configurations:
```bash
curl -s http://localhost:8080/v1/seeds?execution_id=<execution_id> \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. If seeds are not configured for the affected execution:
   - Add explicit seed configuration with `global_seed=42`
   - Ensure NumPy, PyTorch, and Python hash seeds are set

3. If seeds are configured but not being applied:
   - Check agent logs for seed initialization errors
   - Verify the agent reads and applies the seed configuration

### Scenario 5: Critical Drift -- Immediate Baseline Review

**Symptoms:** Critical drift (>5%) detected, calculations may be producing unreliable results.

**Resolution:**

1. **Immediately assess impact:**
```bash
# Check which executions are affected
curl -s "http://localhost:8080/v1/drift/detections?severity=critical&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. **Quarantine affected outputs:** Flag affected calculation results as "unverified" in downstream systems.

3. **Investigate root cause** using the diagnostic steps above.

4. **Re-run affected calculations** after resolving the root cause to produce verified results.

5. **Update baselines** if the drift is due to an intentional change (new emission factors, model updates).

---

## Post-Incident Steps

### Step 1: Verify Drift Is Resolved

```bash
# Check that no new critical drift detections are occurring
curl -s http://localhost:8080/metrics | grep 'gl_reproducibility_drift_detections_total{severity="critical"}'

# Run a verification on the previously-drifting execution
curl -s -X POST http://localhost:8080/v1/verify \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"execution_id": "<execution_id>"}' \
  | python3 -m json.tool
```

### Step 2: Update Baselines If Needed

```bash
# Deactivate old baselines that are no longer valid
curl -s -X PATCH http://localhost:8080/v1/drift/baselines/<baseline_id> \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}' \
  | python3 -m json.tool

# Create new baselines with updated reference data
curl -s -X POST http://localhost:8080/v1/drift/baselines \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "<baseline_name>",
    "description": "Updated after <incident_reason>",
    "baseline_data": <new_reference_data>,
    "drift_threshold": 0.01
  }' | python3 -m json.tool
```

### Step 3: Verify Audit Trail Integrity

```promql
# Verify audit events are being recorded
increase(gl_reproducibility_audit_events_total[5m])

# Check for audit gaps
(
  sum(rate(gl_reproducibility_verifications_total[5m]))
  - sum(rate(gl_reproducibility_audit_events_total[5m]))
)
```

---

## Interim Mitigation

While drift is being investigated:

1. **Affected calculations should be flagged.** Any calculation results produced during the drift period should be marked as "pending verification" in downstream reports and dashboards.

2. **Baselines can be temporarily widened.** If the drift is known to be caused by an intentional change (e.g., new emission factors), increase the drift threshold temporarily while new baselines are established.

3. **Non-determinism can be mitigated.** If the source is identified (e.g., unseeded RNG), apply the fix and re-run affected calculations before updating baselines.

4. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#compliance-ops` -- compliance impact
   - `#data-pipeline-ops` -- affected pipeline notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Moderate drift detected, investigation in progress | On-call engineer | 15 minutes |
| L2 | Critical drift detected, calculation outputs affected | Platform team lead + #platform-foundation | Immediate (<5 min) |
| L3 | Critical drift > 30 minutes, compliance impact, customer-facing reports affected | Platform team + compliance team + CTO notification | Immediate |
| L4 | Systemic non-determinism affecting multiple agents and calculations | All-hands engineering + incident commander + executive notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Reproducibility Agent Health (`/d/reproducibility-service`)
- **Alerts:** `ReproducibilityDriftCritical`, `ReproducibilityDriftModerate`, `ReproducibilityHashMismatch`
- **Key metrics to watch:**
  - `gl_reproducibility_drift_detections_total` by severity (critical should be 0)
  - `gl_reproducibility_hash_mismatches_total` (should be 0)
  - `gl_reproducibility_non_determinism_total` by source (should be near-zero)
  - `gl_reproducibility_environment_mismatches_total` (should be 0)
  - `gl_reproducibility_drift_percentage` (should stay below drift threshold)
  - `gl_reproducibility_verifications_total{status="fail"}` rate (should be < 10%)

### Best Practices

1. **Always set seeds** for any calculation involving random number generation
2. **Pin all dependency versions** in container images to prevent environment drift
3. **Use consistent hardware** across execution environments (same CPU architecture)
4. **Normalize floating-point values** before hashing to account for accumulation order
5. **Update baselines proactively** when emission factors or models are intentionally updated
6. **Run replay verification** after any agent deployment to catch regressions early
7. **Monitor drift trends** to catch gradual drift before it becomes critical

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ReproducibilityDriftCritical` | Critical | Critical severity drift (>5% deviation) |
| `ReproducibilityDriftModerate` | Warning | Moderate severity drift (1-5% deviation, >3 in 15m) |
| `ReproducibilityHashMismatch` | Critical | Artifact hash mismatch (exact output difference) |
| `ReproducibilityNonDeterminismDetected` | Warning | Non-deterministic behavior detected |
| `ReproducibilityEnvironmentMismatch` | Warning | Environment fingerprint differences |
| `ReproducibilityHighFailureRateWarning` | Warning | >10% verification failures |
| `ReproducibilityReplayFailure` | Warning | >20% replay session failures |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 drift detection incident
- **Related runbooks:** [Reproducibility Service Down](./reproducibility-service-down.md)
