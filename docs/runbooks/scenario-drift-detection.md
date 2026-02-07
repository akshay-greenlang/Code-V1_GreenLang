# Scenario Drift Detection

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `AssumptionsScenarioDrift` | Warning | Scenario override values diverging >30% from baseline for 15 minutes |
| `AssumptionsStaleScenario` | Warning | Active scenario with overrides not updated in >90 days |
| `AssumptionsSensitivityAnomaly` | Warning | Sensitivity analysis reveals >50% variance in dependent calculations |

**Thresholds:**

```promql
# AssumptionsScenarioDrift
# Scenario override values diverging significantly from baseline
# Detected via periodic drift check job
increase(gl_assumptions_scenario_drift_detected_total[15m]) > 0
# sustained for 15 minutes

# AssumptionsStaleScenario
# Active scenarios with overrides that have not been reviewed/updated
gl_assumptions_scenario_staleness_days > 90

# AssumptionsSensitivityAnomaly
# Sensitivity analysis reveals large variance in dependent calculations
gl_assumptions_sensitivity_max_variance > 0.5
```

---

## Description

These alerts fire when the scenario management subsystem of the Assumptions Registry service (AGENT-FOUND-004) detects inconsistencies between scenario override values and the baseline assumptions, or when scenarios become stale and unreliable for what-if analysis.

### How Scenario Management Works

The Assumptions Registry supports multiple named scenarios, each representing a different set of assumption values for what-if analysis:

1. **Baseline Scenario** -- The primary scenario containing the officially approved assumption values used for standard compliance calculations. This is the "source of truth" for regulatory submissions.

2. **Optimistic Scenario** -- Overrides specific assumption values with more favorable estimates (e.g., lower emission factors, higher efficiency rates). Used for best-case planning.

3. **Conservative Scenario** -- Overrides specific assumption values with less favorable estimates (e.g., higher emission factors, lower efficiency rates). Used for worst-case planning and risk assessment.

4. **Custom Scenarios** -- User-defined scenarios for specific analysis purposes (e.g., "new_regulation_2027", "supplier_switch", "technology_upgrade").

Each scenario stores only the **overrides** -- assumption values that differ from the baseline. When a calculation queries an assumption under a specific scenario:

1. The registry first checks if the scenario has an override for that assumption key
2. If an override exists, the override value is returned
3. If no override exists, the baseline value is returned
4. The provenance record tracks which scenario was used and whether the value came from an override or baseline fallback

### What Scenario Drift Means

1. **Baseline Has Changed But Overrides Have Not**: The most common cause of drift. When a baseline assumption value is updated (e.g., a new emission factor from IPCC), existing scenario overrides may no longer represent a meaningful deviation from the new baseline. For example, if the baseline CO2 emission factor for coal was 2.4 and the conservative override was 2.8, but the baseline was updated to 2.9, the "conservative" override is now actually more optimistic than the baseline.

2. **Stale Overrides From Previous Reporting Periods**: Scenarios created for a past reporting period may still be active with outdated override values. These stale scenarios can produce misleading what-if analysis results if used with current baseline values.

3. **Orphaned Overrides**: Overrides referencing assumption keys that have been deprecated, renamed, or removed from the baseline registry. These overrides will never be used because the baseline key no longer exists.

4. **Diverging Scenario Intent**: Over time, as individual overrides are updated independently, a scenario may lose its coherent "story" (e.g., a "conservative" scenario may have some values that are actually more optimistic than baseline due to incremental updates).

5. **Cross-Scenario Inconsistency**: Multiple scenarios may contain contradictory assumptions that are internally inconsistent, leading to confusing comparison results.

### Normal vs. Abnormal Drift Levels

| Condition | Status | Typical Cause |
|-----------|--------|---------------|
| <10% deviation from baseline on most overrides | Normal | Scenarios represent small adjustments |
| 10-30% deviation on specific overrides | Normal | Scenarios represent significant what-if variations |
| >30% deviation on multiple overrides | Warning | Overrides may be stale or baseline has shifted significantly |
| Overrides older than 90 days without review | Warning | Scenario may not reflect current conditions |
| Override value on wrong side of baseline (e.g., "conservative" < baseline) | Alert | Baseline shift has inverted the scenario's intent |
| Sensitivity variance >50% on dependent calculations | Alert | Assumption change has outsized impact on calculations |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Users relying on scenario comparisons may receive misleading what-if results; scenario reports may misrepresent risk |
| **Data Impact** | Medium | Calculations using stale scenarios produce results that do not reflect current assumptions; comparison analyses are unreliable |
| **SLA Impact** | Low | Service remains available and functional; scenarios still return values, but the values may be misleading |
| **Revenue Impact** | Medium | Incorrect scenario analysis may lead to poor business decisions; customers relying on what-if analysis may lose confidence |
| **Compliance Impact** | Medium-High | Regulatory scenarios (e.g., conservative estimates for CBAM) must accurately reflect the intended deviation from baseline; stale scenarios may not satisfy auditor requirements |
| **Downstream Impact** | Medium | Calculation agents using scenarios for sensitivity analysis will produce unreliable results; reporting agents may generate misleading comparison charts |

---

## Symptoms

### Scenario Override Drift

- `AssumptionsScenarioDrift` alert firing
- `gl_assumptions_scenario_drift_detected_total` counter incrementing
- Users reporting "unexpected" results when comparing scenarios to baseline
- Scenario comparison API returning overrides that are on the wrong side of the baseline
- Audit reports showing scenario values that contradict the scenario's stated intent

### Stale Scenarios

- `AssumptionsStaleScenario` alert firing
- `gl_assumptions_scenario_staleness_days` gauge exceeding 90 for active scenarios
- Scenario overrides with `last_updated` timestamps from previous reporting periods
- Users questioning why scenario results seem "off" or do not reflect recent changes

### Sensitivity Anomalies

- `AssumptionsSensitivityAnomaly` alert firing
- `gl_assumptions_sensitivity_max_variance` exceeding 0.5 (50%)
- Sensitivity analysis reports showing a single assumption driving >50% of output variance
- Downstream calculation agents reporting "high uncertainty" flags

---

## Diagnostic Steps

### Step 1: Check Scenario Drift Metrics

```bash
# Port-forward to the assumptions service
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080

# Get current scenario metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Active scenarios: {data.get(\"scenarios_active_count\", 0)}')
print(f'Total overrides: {data.get(\"scenario_overrides_total\", 0)}')
print(f'Drift detections: {data.get(\"scenario_drift_detected_total\", 0)}')
print(f'Max staleness (days): {data.get(\"scenario_max_staleness_days\", 0)}')
print(f'Sensitivity max variance: {data.get(\"sensitivity_max_variance\", 0):.3f}')
"
```

```promql
# Drift detection rate
rate(gl_assumptions_scenario_drift_detected_total[5m])

# Staleness by scenario
gl_assumptions_scenario_staleness_days

# Scenario query rate (which scenarios are actually being used)
sum by (scenario_name) (rate(gl_assumptions_scenario_queries_total[1h]))

# Override count by scenario
gl_assumptions_scenario_override_count

# Sensitivity variance trend
gl_assumptions_sensitivity_max_variance
```

### Step 2: Compare Scenario Overrides to Baseline Values

```bash
# List all scenarios and their override counts
curl -s http://localhost:8080/v1/assumptions/scenarios \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('scenarios', []):
    print(f\"Scenario: {s['name']}\")
    print(f\"  Active: {s.get('is_active', False)}\")
    print(f\"  Overrides: {s.get('override_count', 0)}\")
    print(f\"  Created: {s.get('created_at', 'N/A')}\")
    print(f\"  Last updated: {s.get('updated_at', 'N/A')}\")
    print()
"

# Get drift analysis for a specific scenario
curl -s http://localhost:8080/v1/assumptions/scenarios/<scenario_name>/drift-analysis \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Scenario: {data.get('scenario_name')}\")
print(f\"Total overrides: {data.get('total_overrides', 0)}\")
print(f\"Drifted overrides: {data.get('drifted_count', 0)}\")
print(f\"Inverted overrides: {data.get('inverted_count', 0)}\")
print()
if data.get('drifted_overrides'):
    print('Drifted overrides (>30% deviation from baseline):')
    for d in data['drifted_overrides']:
        print(f\"  {d['key']}: baseline={d['baseline_value']}, override={d['override_value']}, deviation={d['deviation_pct']:.1f}%\")
print()
if data.get('inverted_overrides'):
    print('INVERTED overrides (wrong side of baseline for scenario intent):')
    for i in data['inverted_overrides']:
        print(f\"  {i['key']}: baseline={i['baseline_value']}, override={i['override_value']}, expected_direction={i['expected_direction']}\")
"
```

### Step 3: Identify Stale Overrides

```bash
# Query overrides that have not been updated in the current reporting period
kubectl run pg-stale --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT so.scenario_id, s.name as scenario_name, so.assumption_key,
          so.override_value, so.updated_at,
          EXTRACT(DAY FROM NOW() - so.updated_at) as days_stale,
          ar.value as current_baseline, ar.updated_at as baseline_updated
   FROM assumptions_scenario_overrides so
   JOIN assumptions_scenarios s ON so.scenario_id = s.id
   JOIN assumptions_registry ar ON so.assumption_key = ar.key
   WHERE s.is_active = true
     AND so.updated_at < NOW() - INTERVAL '90 days'
   ORDER BY days_stale DESC
   LIMIT 30;"

# Check for overrides where the baseline was updated more recently than the override
kubectl run pg-baseline-newer --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT so.assumption_key, s.name as scenario_name,
          so.override_value, so.updated_at as override_updated,
          ar.value as baseline_value, ar.updated_at as baseline_updated,
          EXTRACT(DAY FROM ar.updated_at - so.updated_at) as days_behind
   FROM assumptions_scenario_overrides so
   JOIN assumptions_scenarios s ON so.scenario_id = s.id
   JOIN assumptions_registry ar ON so.assumption_key = ar.key
   WHERE s.is_active = true
     AND ar.updated_at > so.updated_at
   ORDER BY days_behind DESC
   LIMIT 30;"
```

### Step 4: Check for Orphaned Overrides

```bash
# Find overrides referencing assumption keys that no longer exist in the baseline
kubectl run pg-orphaned --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT so.scenario_id, s.name as scenario_name, so.assumption_key,
          so.override_value, so.updated_at
   FROM assumptions_scenario_overrides so
   JOIN assumptions_scenarios s ON so.scenario_id = s.id
   LEFT JOIN assumptions_registry ar ON so.assumption_key = ar.key
   WHERE s.is_active = true
     AND (ar.key IS NULL OR ar.is_active = false)
   ORDER BY so.updated_at DESC;"
```

### Step 5: Run Sensitivity Analysis

```bash
# Run sensitivity analysis for a specific scenario
curl -s -X POST http://localhost:8080/v1/assumptions/sensitivity \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "scenario_name": "<scenario_name>",
    "target_calculations": ["emissions_total", "carbon_intensity"],
    "perturbation_pct": 10,
    "top_n": 10
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Scenario: {data.get('scenario_name')}\")
print(f\"Target calculations: {', '.join(data.get('target_calculations', []))}\")
print(f\"Assumptions analyzed: {data.get('assumptions_analyzed', 0)}\")
print()
for result in data.get('sensitivity_results', []):
    print(f\"Assumption: {result['key']}\")
    print(f\"  Current value: {result['value']} {result.get('unit', '')}\")
    print(f\"  Sensitivity index: {result['sensitivity_index']:.4f}\")
    print(f\"  Impact on emissions_total: {result.get('impact_pct', 0):.2f}%\")
    print(f\"  Category: {result.get('category', 'N/A')}\")
    print()
"
```

### Step 6: Check Dependency Impact

```bash
# Analyze the dependency impact of drifted overrides
curl -s http://localhost:8080/v1/assumptions/dependencies/<assumption_key>/impact \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Assumption: {data.get('key')}\")
print(f\"Direct dependents: {data.get('direct_dependents', 0)}\")
print(f\"Total affected calculations: {data.get('total_affected', 0)}\")
print()
if data.get('affected_calculations'):
    print('Affected calculations:')
    for calc in data['affected_calculations']:
        print(f\"  {calc['name']} (impact: {calc.get('estimated_impact', 'unknown')})\")
"

# Check the full dependency graph for a scenario
curl -s http://localhost:8080/v1/assumptions/scenarios/<scenario_name>/dependency-graph \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Scenario: {data.get('scenario_name')}\")
print(f\"Overridden assumptions: {data.get('overridden_count', 0)}\")
print(f\"Total dependencies: {data.get('total_dependencies', 0)}\")
print(f\"Max depth: {data.get('max_depth', 0)}\")
print(f\"Circular dependencies: {data.get('circular_count', 0)}\")
"
```

### Step 7: Review Override Change History

```bash
# Review the change history for scenario overrides
kubectl run pg-history --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT cl.assumption_key, cl.old_value, cl.new_value, cl.change_reason,
          cl.user_id, cl.created_at, s.name as scenario_name
   FROM assumptions_change_log cl
   JOIN assumptions_scenarios s ON cl.scenario_id = s.id
   WHERE cl.scenario_id IS NOT NULL
     AND cl.created_at > NOW() - INTERVAL '30 days'
   ORDER BY cl.created_at DESC
   LIMIT 30;"
```

---

## Resolution Steps

### Option 1: Update Stale Overrides to Reflect Current Baseline

When overrides have not been reviewed after baseline changes, update them to maintain the intended deviation from the new baseline.

```bash
# Step 1: Generate a drift report
curl -s http://localhost:8080/v1/assumptions/scenarios/<scenario_name>/drift-analysis \
  -H "Authorization: Bearer $ACCESS_TOKEN" > /tmp/drift_report.json

# Step 2: Review the drift report and determine the correct override values
# For each drifted override, decide:
# a) Update the override to maintain the intended percentage deviation
# b) Update the override to a new absolute value
# c) Remove the override (use baseline value)

# Step 3: Update individual overrides
curl -X PUT http://localhost:8080/v1/assumptions/scenarios/<scenario_name>/overrides/<assumption_key> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "override_value": 3.2,
    "change_reason": "Updated conservative override to maintain +15% deviation from new baseline (2.78)"
  }' | python3 -m json.tool

# Step 4: Batch-update multiple overrides
curl -X PATCH http://localhost:8080/v1/assumptions/scenarios/<scenario_name>/overrides/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "updates": [
      {
        "assumption_key": "emission_factor_coal",
        "override_value": 3.2,
        "change_reason": "Updated to +15% above new baseline"
      },
      {
        "assumption_key": "emission_factor_gas",
        "override_value": 2.1,
        "change_reason": "Updated to +10% above new baseline"
      }
    ]
  }' | python3 -m json.tool
```

### Option 2: Deactivate Outdated Scenarios

If a scenario is no longer relevant (e.g., from a previous reporting period), deactivate it to prevent misleading analysis.

```bash
# Deactivate a scenario
curl -X PATCH http://localhost:8080/v1/assumptions/scenarios/<scenario_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "is_active": false,
    "change_reason": "Deactivated stale scenario from Q3-2025 reporting period. Replaced by 2026_conservative."
  }' | python3 -m json.tool

# Verify deactivation
curl -s http://localhost:8080/v1/assumptions/scenarios \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('scenarios', []):
    status = 'ACTIVE' if s.get('is_active') else 'INACTIVE'
    print(f\"[{status}] {s['name']} (overrides: {s.get('override_count', 0)}, updated: {s.get('updated_at', 'N/A')})\")"
```

### Option 3: Create New Scenarios with Corrected Overrides

If the existing scenario has drifted too far, create a new scenario from the current baseline with intentional deviations.

```bash
# Create a new scenario
curl -X POST http://localhost:8080/v1/assumptions/scenarios \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "name": "conservative_2026_q1",
    "description": "Conservative scenario for Q1 2026 compliance reporting. All emission factors +15% above baseline.",
    "base_scenario": "baseline",
    "constraints": {
      "override_must_be_within_pct_of_baseline": 50,
      "allow_negative_values": false
    },
    "overrides": [
      {
        "assumption_key": "emission_factor_coal_combustion",
        "override_value": 3.22,
        "reason": "+15% above baseline (2.80)"
      },
      {
        "assumption_key": "emission_factor_natural_gas",
        "override_value": 2.19,
        "reason": "+15% above baseline (1.90)"
      },
      {
        "assumption_key": "grid_emission_factor",
        "override_value": 0.52,
        "reason": "+15% above baseline (0.45)"
      }
    ],
    "change_reason": "Created new conservative scenario for Q1 2026 reporting period"
  }' | python3 -m json.tool
```

### Option 4: Remove Orphaned Overrides

Clean up overrides that reference assumption keys no longer in the baseline registry.

```bash
# Remove orphaned overrides via API
curl -X DELETE http://localhost:8080/v1/assumptions/scenarios/<scenario_name>/overrides/orphaned \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "X-Change-Reason: Cleanup of orphaned overrides referencing deprecated assumption keys" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Orphaned overrides removed: {data.get('removed_count', 0)}\")
for r in data.get('removed', []):
    print(f\"  Removed: {r['assumption_key']} (was: {r['override_value']})\")
"

# Or remove directly in the database if API is not available
kubectl run pg-cleanup --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "DELETE FROM assumptions_scenario_overrides so
   USING assumptions_scenarios s
   WHERE so.scenario_id = s.id
     AND s.is_active = true
     AND NOT EXISTS (
       SELECT 1 FROM assumptions_registry ar
       WHERE ar.key = so.assumption_key AND ar.is_active = true
     )
   RETURNING so.assumption_key, s.name as scenario_name;"
```

### Option 5: Trigger Recalculations with Updated Scenarios

After correcting scenario overrides, trigger recalculations for any reports or analyses that used the drifted scenario values.

```bash
# Identify calculations that used the drifted scenario in the affected period
kubectl run pg-affected --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT DISTINCT calculation_id, calculation_type, scenario_name,
          created_at, status
   FROM assumptions_usage_log
   WHERE scenario_name = '<scenario_name>'
     AND created_at > NOW() - INTERVAL '30 days'
   ORDER BY created_at DESC
   LIMIT 50;"

# Queue recalculations for affected reports
curl -X POST http://localhost:8080/v1/assumptions/admin/trigger-recalculation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "scenario_name": "<scenario_name>",
    "affected_period_start": "2026-01-01",
    "affected_period_end": "2026-02-07",
    "dry_run": true
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Affected calculations: {data.get('affected_count', 0)}\")
print(f\"Dry run: {data.get('dry_run', True)}\")
for c in data.get('affected', []):
    print(f\"  {c['calculation_id']}: {c['calculation_type']} ({c['status']})\")
"
```

---

## Post-Resolution Verification

```promql
# 1. Drift detection rate should drop to 0
rate(gl_assumptions_scenario_drift_detected_total[5m]) == 0

# 2. Scenario staleness should be below threshold
gl_assumptions_scenario_staleness_days < 90

# 3. Sensitivity variance should be within acceptable range
gl_assumptions_sensitivity_max_variance < 0.5

# 4. Scenario query rate should remain stable (scenarios still being used)
sum by (scenario_name) (rate(gl_assumptions_scenario_queries_total[5m]))

# 5. Override count should reflect cleanup (orphaned removed)
gl_assumptions_scenario_override_count
```

```bash
# 6. Run drift analysis on corrected scenarios
for scenario in "conservative" "optimistic" "conservative_2026_q1"; do
  result=$(curl -s http://localhost:8080/v1/assumptions/scenarios/$scenario/drift-analysis \
    -H "Authorization: Bearer $ACCESS_TOKEN" 2>/dev/null)
  if [ $? -eq 0 ]; then
    drifted=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drifted_count', 'ERROR'))" 2>/dev/null)
    inverted=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('inverted_count', 'ERROR'))" 2>/dev/null)
    echo "$scenario: drifted=$drifted, inverted=$inverted"
  fi
done

# 7. Verify provenance records for scenario updates
kubectl run pg-prov --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT assumption_key, scenario_id, change_reason, provenance_hash, created_at
   FROM assumptions_change_log
   WHERE scenario_id IS NOT NULL
     AND created_at > NOW() - INTERVAL '1 hour'
   ORDER BY created_at DESC
   LIMIT 10;"
```

---

## Scenario Management Best Practices

### Scenario Lifecycle

1. **Create** scenarios at the beginning of each reporting period with clearly defined intent and constraints
2. **Review** all active scenarios when baseline assumptions are updated to check for drift
3. **Update** overrides promptly when baseline changes to maintain intended deviations
4. **Deactivate** scenarios at the end of each reporting period (archive, do not delete)
5. **Audit** scenario usage to ensure calculations reference the correct scenario

### Naming Conventions

| Pattern | Example | Use Case |
|---------|---------|----------|
| `baseline` | `baseline` | Official values for compliance |
| `conservative_YYYY_QN` | `conservative_2026_q1` | Period-specific conservative estimates |
| `optimistic_YYYY_QN` | `optimistic_2026_q1` | Period-specific optimistic estimates |
| `analysis_<purpose>` | `analysis_supplier_switch` | Ad-hoc what-if analysis |
| `regulatory_<framework>` | `regulatory_cbam_2026` | Framework-specific regulatory scenario |

### Drift Prevention

1. **Automate drift checks** -- Run the drift analysis endpoint daily as a scheduled job and alert on deviations
2. **Link scenarios to reporting periods** -- Each scenario should have `effective_start` and `effective_end` dates
3. **Require review on baseline updates** -- When a baseline assumption changes, automatically flag all active scenarios with overrides for that key as "needs review"
4. **Set staleness thresholds per scenario type** -- Conservative and regulatory scenarios should be reviewed more frequently than exploratory scenarios
5. **Document override rationale** -- Every override should include a `change_reason` explaining the intended deviation (e.g., "+15% for risk buffer per policy XYZ")

---

## Prevention

### Monitoring

- **Dashboard:** Assumptions Registry Health (`/d/assumptions-service-health`) -- scenario panels
- **Dashboard:** Assumptions Operations Overview (`/d/assumptions-operations-overview`)
- **Alert:** `AssumptionsScenarioDrift` (this alert)
- **Alert:** `AssumptionsStaleScenario` (this alert)
- **Alert:** `AssumptionsSensitivityAnomaly` (this alert)
- **Key metrics to watch:**
  - `gl_assumptions_scenario_drift_detected_total` rate (should be 0 in steady state)
  - `gl_assumptions_scenario_staleness_days` per scenario (should be <90)
  - `gl_assumptions_sensitivity_max_variance` (should be <0.5)
  - `gl_assumptions_scenario_queries_total` by scenario (tracks scenario usage patterns)
  - `gl_assumptions_scenario_override_count` (should be stable; sudden changes indicate bulk updates)

### Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Scenario drift detected, limited to a single non-regulatory scenario | On-call engineer | Within 30 minutes |
| L2 | Drift in conservative or regulatory scenario, or staleness >180 days | Platform team lead + compliance team | 15 minutes |
| L3 | Sensitivity variance >50% affecting compliance calculations, regulatory deadline at risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Multiple scenarios drifted, calculations across multiple pipelines affected | All-hands engineering + incident commander | Immediate |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team + Compliance Team
- **Review cadence:** Quarterly or after any scenario-related incident
- **Related alerts:** `AssumptionsServiceDown`, `AssumptionsValidationFailureSpike`, `AssumptionsAuditGap`
- **Related dashboards:** Assumptions Registry Health, Assumptions Operations Overview
- **Related runbooks:** [Assumptions Service Down](./assumptions-service-down.md), [Assumption Validation Failures](./assumption-validation-failures.md), [Assumptions Audit Compliance](./assumptions-audit-compliance.md)
