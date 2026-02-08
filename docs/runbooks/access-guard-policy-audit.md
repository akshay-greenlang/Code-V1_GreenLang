# Access Guard Policy Audit

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `AccessGuardAuditGap` | Critical | Missing audit entries detected for access decisions (gap > 30 minutes with active decisions) |
| `AccessGuardTenantViolation` | Critical | Cross-tenant access attempt detected and blocked |
| `AccessGuardClassificationBreach` | Critical | Unauthorized access to restricted/top_secret classified resource |
| `AccessGuardPolicyConflict` | Warning | Multiple rules with conflicting effects matching the same request |
| `AccessGuardSimulationDrift` | Warning | Simulation mode denial rate diverging >10% from enforcement mode |

**Thresholds:**

```promql
# AccessGuardAuditGap
# Gap detection: decisions occurring but no corresponding audit entries
(
  sum(rate(gl_access_guard_decisions_total[5m]))
  - sum(rate(gl_access_guard_audit_events_total[5m]))
) > 1
# sustained for 30 minutes

# AccessGuardTenantViolation
# Any cross-tenant access attempt
increase(gl_access_guard_tenant_violations_total[5m]) > 0

# AccessGuardClassificationBreach
# Any unauthorized access to restricted/top_secret
increase(gl_access_guard_classification_breach_total{level=~"restricted|top_secret"}[5m]) > 0

# AccessGuardPolicyConflict
# Multiple rules with conflicting effects
sum(rate(gl_access_guard_policy_conflicts_total[5m])) > 1

# AccessGuardSimulationDrift
# Simulation vs enforcement denial rate divergence > 10%
abs(
  sum(rate(gl_access_guard_simulation_decisions_total{decision="deny"}[15m]))
  / sum(rate(gl_access_guard_simulation_decisions_total[15m]))
  -
  sum(rate(gl_access_guard_decisions_total{decision="deny"}[15m]))
  / sum(rate(gl_access_guard_decisions_total[15m]))
) > 0.10
```

---

## Description

These alerts fire when the audit and compliance subsystem of the Access & Policy Guard Agent service (AGENT-FOUND-006) detects integrity issues in the access decision audit trail, tenant isolation enforcement, data classification access control, or policy consistency. The audit system is a critical component of the zero-trust security model and is required for SOC 2, ISO 27001, CSRD, CBAM, and other regulatory compliance frameworks.

### How the Audit System Works

The Access & Policy Guard Agent maintains a multi-layered audit trail:

1. **Access Decision History** -- Every access decision (allow, deny, rate_limited, challenge, defer) is recorded in the `access_decisions` hypertable with:
   - Timestamp (UTC, microsecond precision)
   - Request ID, principal ID/type/tenant, resource ID/type/tenant
   - Action, decision, allowed boolean
   - Matching rule IDs, deny reasons
   - Evaluation time in milliseconds
   - SHA-256 decision hash for tamper detection

2. **Comprehensive Audit Events** -- Every access control operation is recorded in the `audit_events` hypertable with:
   - Event type (access_request, access_granted, access_denied, policy_created, rate_limit_hit, tenant_violation, etc.)
   - Principal, resource, action, decision
   - Details JSONB with full context
   - Source IP and user agent
   - Configurable retention period

3. **Tenant Isolation Verification** -- Cross-tenant access attempts are detected, blocked, and logged as `tenant_violation` audit events with immediate alerting

4. **Classification Breach Detection** -- Unauthorized access to classified resources is detected, blocked, and logged as `classification_breach` audit events

5. **Policy Conflict Detection** -- When multiple rules with conflicting effects (allow + deny) match the same request, the conflict is logged while the deny-wins model resolves correctly

### Regulatory Requirements

| Framework | Requirement | How Access Guard Satisfies |
|-----------|-------------|--------------------------|
| **SOC 2** | CC6.1-6.8 Logical access controls; audit trail for authorization decisions | Complete decision history with principal, resource, action, result, matching rules, and decision hash |
| **ISO 27001** | A.9.2 User access management; A.9.4 Access control; A.12.4 Logging | Policy-based access control with deny-wins model; comprehensive audit events with source IP |
| **CSRD** | Data quality and assurance over sustainability reporting | Classification-based access control ensuring only authorized analysts access emission data |
| **CBAM** | Verifiable access controls for emission data and calculation parameters | Tenant isolation preventing cross-company data access; classification enforcement on restricted data |
| **GDPR** | Article 30 Records of processing activities; Article 32 Security of processing | PII classified as restricted; access decisions audited with decision hash; retention policies enforced |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Users may not notice audit gaps immediately; violations are detected asynchronously |
| **Data Impact** | Critical | Audit trail integrity compromised; access decisions may not be traceable; tenant isolation breach possible |
| **SLA Impact** | Medium | Service remains available for access decisions, but audit operations may fail |
| **Revenue Impact** | High | Customers depend on auditable access control for regulatory submissions |
| **Compliance Impact** | Critical | SOC 2, ISO 27001, CSRD, CBAM, GDPR compliance directly impacted; audit reports may be rejected |
| **Downstream Impact** | High | Compliance reporting agents cannot generate auditor-ready reports; regulatory submission workflows blocked |

---

## Symptoms

### Audit Gap

- `AccessGuardAuditGap` alert firing
- `gl_access_guard_audit_events_total` not incrementing despite active access decisions
- Compliance reports showing missing audit entries
- SOC 2 auditor flagging gaps in access decision records

### Tenant Violation

- `AccessGuardTenantViolation` alert firing
- `gl_access_guard_tenant_violations_total` incrementing
- Audit events showing `tenant_violation` or `cross_tenant_attempt` entries
- Security team receiving violation notifications

### Classification Breach

- `AccessGuardClassificationBreach` alert firing
- `gl_access_guard_classification_breach_total` incrementing for restricted/top_secret levels
- Audit events showing unauthorized classification access attempts
- Data protection officer receiving breach notifications

### Policy Conflict

- `AccessGuardPolicyConflict` alert firing
- Multiple rules matching the same request with conflicting effects
- Evaluation latency increasing due to conflict resolution overhead
- Policy administrators notified of overlapping rule definitions

---

## Diagnostic Steps

### Step 1: Identify the Alert Type

```bash
# Check which access guard alerts are firing
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'ALERTS{alertname=~"AccessGuard.*", alertstate="firing"}'

# Check current decision rates
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'sum(rate(gl_access_guard_decisions_total[5m])) by (decision)'
```

### Step 2: Review Audit Event Flow

```bash
# Check audit event write rate
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'sum(rate(gl_access_guard_audit_events_total[5m])) by (event_type)'

# Check for audit write errors in logs
kubectl logs -n greenlang -l app=access-guard-service --tail=500 \
  | grep -i "audit\|event\|write.*fail\|insert.*error"
```

### Step 3: Check Tenant Violation Details

```bash
# Query recent tenant violations from the database
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT event_id, principal_id, resource_id, action, source_ip, details, timestamp
   FROM access_guard_service.audit_events
   WHERE event_type IN ('tenant_violation', 'cross_tenant_attempt')
   ORDER BY timestamp DESC
   LIMIT 20;"
```

### Step 4: Check Classification Breach Details

```bash
# Query recent classification breaches
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT decision_id, principal_id, resource_id, action, deny_reasons, timestamp
   FROM access_guard_service.access_decisions
   WHERE 'classification_breach' = ANY(deny_reasons)
   ORDER BY timestamp DESC
   LIMIT 20;"
```

### Step 5: Check Policy Conflicts

```bash
# Query decisions with multiple matching rules
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT decision_id, principal_id, resource_id, action, matching_rules, decision, timestamp
   FROM access_guard_service.access_decisions
   WHERE array_length(matching_rules, 1) > 1
   ORDER BY timestamp DESC
   LIMIT 20;"
```

---

## Resolution Steps

### Scenario 1: Audit Gap -- Database Write Failures

**Symptoms:** Audit events not being recorded despite active decisions.

**Resolution:**

1. Check database connectivity and pool status:
```bash
kubectl logs -n greenlang -l app=access-guard-service --tail=200 \
  | grep -i "pool\|connection\|database\|timeout"
```

2. Check TimescaleDB hypertable health:
```bash
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT hypertable_name, num_chunks, total_bytes
   FROM timescaledb_information.hypertable_detailed_size
   WHERE hypertable_schema = 'access_guard_service';"
```

3. Restart the service to reset connection pools:
```bash
kubectl rollout restart deployment/access-guard-service -n greenlang
kubectl rollout status deployment/access-guard-service -n greenlang
```

### Scenario 2: Tenant Violation -- Misconfigured Service Account

**Symptoms:** Legitimate internal service triggering cross-tenant violations.

**Resolution:**

1. Identify the violating principal:
```bash
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT principal_id, COUNT(*) as violations, MIN(timestamp), MAX(timestamp)
   FROM access_guard_service.audit_events
   WHERE event_type = 'tenant_violation'
   AND timestamp > NOW() - INTERVAL '1 hour'
   GROUP BY principal_id
   ORDER BY violations DESC;"
```

2. If it is a legitimate internal service, add an admin-scoped cross-tenant rule or update the service account tenant configuration.

3. If it is unauthorized, escalate to the security team immediately.

### Scenario 3: Policy Conflict -- Overlapping Rules

**Symptoms:** Multiple rules with conflicting effects matching the same requests.

**Resolution:**

1. Identify the conflicting rules:
```bash
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT r.rule_id, r.name, r.effect, r.priority, r.actions, r.resources, r.principals, p.name as policy_name
   FROM access_guard_service.policy_rules r
   JOIN access_guard_service.policies p ON r.policy_id = p.policy_id
   WHERE r.enabled = true
   ORDER BY r.priority, r.effect;"
```

2. Refine the conflicting rules by narrowing their scope (actions, resources, principals) or adjusting priority.

3. Use simulation mode to validate changes before deploying to enforcement.

---

## Post-Incident Steps

### Step 1: Verify Audit Trail Integrity

```bash
# Check audit event counts for the last 24 hours
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT event_type, COUNT(*) as count
   FROM access_guard_service.audit_events
   WHERE timestamp > NOW() - INTERVAL '24 hours'
   GROUP BY event_type
   ORDER BY count DESC;"
```

### Step 2: Verify Decision Hash Chain

```bash
# Run decision hash integrity verification
curl -s -X POST http://localhost:8080/v1/admin/verify-decision-hashes \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -m json.tool
```

### Step 3: Generate Compliance Report

```bash
# Trigger an ad-hoc compliance report covering the incident period
curl -s -X POST http://localhost:8080/v1/compliance/reports \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "period_start": "2026-02-07T00:00:00Z",
    "period_end": "2026-02-08T00:00:00Z",
    "include_violations": true,
    "include_classification_breaches": true
  }' | python3 -m json.tool
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Audit gap or policy conflict, no active security breach | On-call engineer | 15 minutes |
| L2 | Tenant violation detected, source identified as internal misconfiguration | Platform team lead + #security-oncall | 15 minutes |
| L3 | Tenant violation or classification breach from unknown source | Platform team + security team + CTO notification | Immediate |
| L4 | Multiple tenant violations or sustained classification breaches indicating active attack | All-hands security + incident commander + legal notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Access & Policy Guard Agent Health (`/d/access-guard-service`)
- **Alerts:** `AccessGuardAuditGap`, `AccessGuardTenantViolation`, `AccessGuardClassificationBreach`, `AccessGuardPolicyConflict`, `AccessGuardSimulationDrift`
- **Key metrics to watch:**
  - `gl_access_guard_audit_events_total` rate (should track decision rate)
  - `gl_access_guard_tenant_violations_total` (should always be 0)
  - `gl_access_guard_classification_breach_total` (should always be 0)
  - `gl_access_guard_policy_conflicts_total` rate (should be minimal)
  - `gl_access_guard_decisions_total` by decision type (deny rate should be < 20%)

### Best Practices

1. **Always use simulation mode** before deploying new policies to enforcement
2. **Review policy conflicts weekly** and consolidate overlapping rules
3. **Rotate service account credentials** quarterly
4. **Audit classification assignments** monthly to ensure resources are correctly classified
5. **Generate compliance reports weekly** and review denial patterns

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `AccessGuardServiceDown` | Critical | No access guard service pods running |
| `AccessGuardAuditGap` | Critical | Missing audit entries for access decisions |
| `AccessGuardTenantViolation` | Critical | Cross-tenant access attempt detected |
| `AccessGuardClassificationBreach` | Critical | Unauthorized access to restricted/top_secret |
| `AccessGuardPolicyConflict` | Warning | Multiple rules with conflicting effects matching |
| `AccessGuardSimulationDrift` | Warning | Simulation vs enforcement diverging >10% |
| `AccessGuardOPAEvaluationFailure` | Warning | OPA Rego policy evaluation errors |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Security Team
- **Review cadence:** Quarterly or after any P1 access guard incident
- **Related runbooks:** [Access Guard Service Down](./access-guard-service-down.md)
