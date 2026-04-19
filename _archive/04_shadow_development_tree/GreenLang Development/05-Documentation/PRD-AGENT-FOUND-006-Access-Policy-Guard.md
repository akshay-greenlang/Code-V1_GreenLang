# PRD: AGENT-FOUND-006 - GreenLang Access & Policy Guard

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-006 |
| **Agent ID** | GL-FOUND-X-006 |
| **Component** | Access & Policy Guard Agent |
| **Category** | Foundations Agent |
| **Priority** | P1 - High (authorization backbone for all agent operations) |
| **Status** | Layer 1 Complete (~1,715 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS processes sensitive sustainability data across multiple tenants,
jurisdictions, and regulatory frameworks. Every agent operation, data access, and export
must be authorized against a deterministic policy framework that ensures:

- **Multi-tenant isolation**: Tenant A cannot access Tenant B's emission data
- **Data classification enforcement**: Restricted/top-secret data requires elevated clearance
- **Regulatory compliance**: CSRD, CBAM, EUDR each have specific data handling requirements
- **Agent execution control**: Not all agents can run with all data
- **Export/retention policies**: Data export and retention governed by classification + geography
- **Rate limiting**: Per-tenant, per-user, per-role request throttling
- **OPA Rego integration**: Policy-as-code with version control and simulation
- **Complete audit trail**: Every access decision logged with SHA-256 provenance

Without a production-grade access guard:
- Cross-tenant data leakage risks in multi-tenant deployments
- No way to enforce data classification policies programmatically
- Manual policy management is error-prone and unauditable
- OPA Rego policies cannot be versioned or simulated safely
- Compliance reports for SOC 2 / ISO 27001 cannot be generated automatically

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/policy_guard.py` (1,715 lines)
- `PolicyGuardAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-006)
- `AccessDecision` enum: ALLOW, DENY, CONDITIONAL
- `PolicyType` enum: DATA_ACCESS, AGENT_EXECUTION, EXPORT, RETENTION, GEOGRAPHIC, RATE_LIMIT (6 types)
- `DataClassification` enum: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED, TOP_SECRET (5 levels)
- `RoleType` enum: VIEWER, ANALYST, EDITOR, ADMIN, SUPER_ADMIN, SERVICE_ACCOUNT (6 roles)
- `AuditEventType` enum: ACCESS_GRANTED, ACCESS_DENIED, POLICY_EVALUATED, RATE_LIMIT_EXCEEDED, TENANT_BOUNDARY_VIOLATION, CLASSIFICATION_CHECK, EXPORT_APPROVED, EXPORT_DENIED, POLICY_UPDATED, SIMULATION_RUN (10 types)
- Pydantic models: Principal, Resource, AccessRequest, PolicyRule, Policy, AccessDecisionResult, AuditEvent, RateLimitConfig, PolicyGuardConfig, ComplianceReport, PolicySimulationResult
- `PolicyEngine` class with RBAC/ABAC evaluation, rule matching, pattern matching, OPA Rego stubs
- `RateLimiter` class with token bucket per-minute/hour/day + role overrides
- 6 agent operations: check_access, add_policy, remove_policy, classify_data, generate_report, simulate_policies
- Classification hierarchy, default role permissions, tenant isolation, decision caching
- In-memory storage (no database persistence)

### 3.2 Layer 1 Tests
**File**: `tests/agents/foundation/test_policy_guard.py`

### 3.3 Related SEC Components (Already Built)
- **SEC-002 RBAC**: 10 roles, 61 permissions, role hierarchy (authorization layer for HTTP routes)
- **SEC-005 Audit Logging**: 70+ event types, async queue, PostgreSQL+Loki+Redis
- Note: AGENT-FOUND-006 is the **foundations-layer** policy guard for agent-to-agent and data-to-agent authorization, complementing (not replacing) SEC-002's HTTP-level RBAC

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/access_guard/` package providing a clean SDK for other agents/services.

### Gap 2: No Prometheus Metrics
No `greenlang/access_guard/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_access_guard(app)` / `get_access_guard(app)` pattern.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 has in-memory storage; doesn't delegate to persistent integration module.

### Gap 5: No REST API Router
No `greenlang/access_guard/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/access-guard-service/` manifests.

### Gap 7: No Database Migration
No `V026__access_guard_service.sql` for persistent policy storage and audit trails.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/access-guard-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for access guard operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/access_guard/
  __init__.py           # Public API exports
  config.py             # AccessGuardConfig with GL_ACCESS_GUARD_ env prefix
  models.py             # Pydantic v2 models (re-export + enhance from foundation agent)
  policy_engine.py      # PolicyEngine: add/remove/get/evaluate policies, rule matching
  rate_limiter.py       # RateLimiter: token bucket, role overrides, quota tracking
  classifier.py         # DataClassifier: data classification, sensitivity detection
  audit_logger.py       # AuditLogger: event logging, filtering, retention
  opa_integration.py    # OPAClient: Rego policy management, evaluation bridge
  provenance.py         # ProvenanceTracker: SHA-256 hash chain for policies/decisions
  metrics.py            # 12 Prometheus metrics
  setup.py              # AccessGuardService facade, configure/get
  api/
    __init__.py
    router.py           # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V026)
```sql
CREATE SCHEMA access_guard_service;
-- policies (policy definitions with version history)
-- policy_rules (individual rules linked to policies)
-- access_decisions (hypertable - decision audit trail)
-- audit_events (hypertable - comprehensive audit log)
-- rate_limit_state (current rate limit counters)
-- data_classifications (resource classification registry)
-- rego_policies (OPA Rego policy source storage)
-- compliance_reports (generated compliance reports)
```

### 5.3 Prometheus Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_access_guard_decisions_total` | Counter | Total decisions by action, result |
| 2 | `gl_access_guard_decision_duration_seconds` | Histogram | Decision evaluation latency |
| 3 | `gl_access_guard_denials_total` | Counter | Denials by reason |
| 4 | `gl_access_guard_rate_limits_total` | Counter | Rate limit hits by tenant |
| 5 | `gl_access_guard_policy_evaluations_total` | Counter | Policy evaluations by policy_id |
| 6 | `gl_access_guard_tenant_violations_total` | Counter | Tenant boundary violations |
| 7 | `gl_access_guard_classification_checks_total` | Counter | Classification checks by level |
| 8 | `gl_access_guard_policies_total` | Gauge | Total loaded policies |
| 9 | `gl_access_guard_rules_total` | Gauge | Total active rules |
| 10 | `gl_access_guard_cache_hits_total` | Counter | Decision cache hits |
| 11 | `gl_access_guard_cache_misses_total` | Counter | Decision cache misses |
| 12 | `gl_access_guard_audit_events_total` | Gauge | Total audit events |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/access/check` | Check access (main authorization endpoint) |
| POST | `/v1/access/check-export` | Check export authorization |
| POST | `/v1/access/classify` | Classify a resource |
| POST | `/v1/policies` | Create/add a policy |
| GET | `/v1/policies` | List policies (with filters) |
| GET | `/v1/policies/{id}` | Get policy by ID |
| PUT | `/v1/policies/{id}` | Update policy |
| DELETE | `/v1/policies/{id}` | Delete policy |
| POST | `/v1/policies/simulate` | Simulate policy evaluation |
| POST | `/v1/opa/policies` | Add OPA Rego policy |
| GET | `/v1/opa/policies` | List OPA Rego policies |
| DELETE | `/v1/opa/policies/{id}` | Delete OPA Rego policy |
| GET | `/v1/audit/events` | Get audit events (with filters) |
| GET | `/v1/audit/events/{id}` | Get audit event by ID |
| POST | `/v1/reports/compliance` | Generate compliance report |
| GET | `/v1/rate-limits/{tenant_id}/{principal_id}` | Get remaining quota |
| POST | `/v1/rate-limits/reset` | Reset rate limit counters |
| GET | `/v1/classifications` | List data classifications |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### 5.5 Key Design Principles
1. **Deterministic decisions**: No ML/probabilistic access control - all rule-based
2. **Deny-by-default**: Strict mode denies unless explicitly allowed
3. **Tenant isolation**: Cross-tenant access is impossible unless explicitly configured
4. **First-match-wins**: Rules sorted by priority, first matching rule determines outcome
5. **Policy-as-code**: OPA Rego policies version-controlled with SHA-256 hashes
6. **Simulation mode**: Test policy changes without enforcing them
7. **Complete audit trail**: Every decision logged with provenance hash

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/access_guard/__init__.py` - Public API exports
2. Create `greenlang/access_guard/config.py` - AccessGuardConfig with GL_ACCESS_GUARD_ env prefix
3. Create `greenlang/access_guard/models.py` - Pydantic v2 models
4. Create `greenlang/access_guard/policy_engine.py` - PolicyEngine with RBAC/ABAC evaluation
5. Create `greenlang/access_guard/rate_limiter.py` - RateLimiter with token bucket
6. Create `greenlang/access_guard/classifier.py` - DataClassifier for sensitivity detection
7. Create `greenlang/access_guard/audit_logger.py` - AuditLogger with filtering/retention
8. Create `greenlang/access_guard/opa_integration.py` - OPA Rego policy management
9. Create `greenlang/access_guard/provenance.py` - ProvenanceTracker
10. Create `greenlang/access_guard/metrics.py` - 12 Prometheus metrics
11. Create `greenlang/access_guard/api/router.py` - FastAPI router with 20 endpoints
12. Create `greenlang/access_guard/setup.py` - AccessGuardService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V026__access_guard_service.sql`
2. Create K8s manifests in `deployment/kubernetes/access-guard-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-14. Create unit, integration, and load tests

## 7. Success Criteria
- Integration module provides clean SDK for all policy operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V026 database migration for persistent policy storage
- 20 REST API endpoints operational
- 500+ tests passing
- Complete audit trail for every access decision
- OPA Rego policy simulation without enforcement side effects
- Compliance report generation for SOC 2 / ISO 27001

## 8. Integration Points

### 8.1 Upstream Dependencies
- **SEC-002 RBAC**: HTTP-level role/permission definitions
- **SEC-005 Audit Logging**: Event forwarding to centralized audit

### 8.2 Downstream Consumers
- **All foundation agents (001-005)**: Must check access before data operations
- **All calculation agents**: Must verify data access before reading emission factors
- **API Gateway (Kong)**: Policy guard as authorization backend
- **Evidence packages**: Access decisions become audit evidence

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent policy and audit storage (V026 migration)
- **Redis**: Decision caching, rate limit state
- **Prometheus**: 12 observability metrics
- **Grafana**: Access guard dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
