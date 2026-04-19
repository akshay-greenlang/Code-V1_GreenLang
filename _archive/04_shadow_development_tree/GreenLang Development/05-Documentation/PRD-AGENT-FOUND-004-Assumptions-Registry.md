# PRD: AGENT-FOUND-004 - GreenLang Assumptions Registry

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-004 |
| **Agent ID** | GL-FOUND-X-004 |
| **Component** | Assumptions Registry Agent |
| **Category** | Foundations Agent |
| **Priority** | P0 - Critical (blocks downstream calculations) |
| **Status** | Core Complete (~90%), Integration Gap-Fill In Progress |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires a version-controlled, audit-ready registry for managing
all assumptions used in zero-hallucination compliance calculations. Climate/sustainability
calculations depend on hundreds of assumptions (emission factors, conversion rates,
economic parameters, regulatory thresholds) that must be:

- Explicitly defined (never inferred or hallucinated)
- Version-controlled with immutable history
- Scenario-aware for what-if analysis
- Fully auditable for regulatory compliance
- Validated against allowed ranges
- Traceable to authoritative sources (EPA, IPCC, IEA, DEFRA)

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/assumptions_registry.py` (1,638 lines)
- `AssumptionsRegistryAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-004)
- `AssumptionDataType` enum: FLOAT, INTEGER, STRING, BOOLEAN, PERCENTAGE, RATIO, DATE, LIST_FLOAT, LIST_STRING, DICT
- `AssumptionCategory` enum: EMISSION_FACTOR, CONVERSION_FACTOR, ECONOMIC, OPERATIONAL, REGULATORY, CLIMATE, ENERGY, TRANSPORT, WASTE, WATER, CUSTOM
- `ScenarioType` enum: BASELINE, OPTIMISTIC, CONSERVATIVE, BEST_CASE, WORST_CASE, REGULATORY, CUSTOM
- `ChangeType` enum: CREATE, UPDATE, DELETE, SCENARIO_OVERRIDE, INHERIT, REVERT
- Assumption CRUD operations (create/get/update/delete/list)
- Scenario management (create/get/update/delete/list with overrides)
- Value operations with inheritance and scenario override chains
- Complete change audit trail with SHA-256 provenance hashes
- Validation engine with min/max/allowed/regex/custom validators
- Dependency graph tracking (upstream/downstream/calculation links)
- Sensitivity analysis with scenario comparison and range calculation
- Export/import functionality with integrity hashing
- 3 default scenarios created on init (Baseline, Conservative, Optimistic)

### 3.2 Layer 1 Tests
**File**: `tests/agents/foundation/test_assumptions_registry.py` (1,316 lines)
- 40+ tests across 9 test classes
- Covers: CRUD, version control, scenario management, change logging, validation, dependency tracking, sensitivity analysis, export/import, edge cases, integration workflows

### 3.3 Additional Files
- `GreenLang Development/01-Core-Platform/agents/foundation/assumptions_registry.py` - Copy of foundation agent
- `GreenLang Development/07-Testing/agents/foundation/test_assumptions_registry.py` - Copy of tests
- `cbam-pack-mvp/test_output/audit/assumptions.json` - Sample assumptions output
- `cbam-pack-mvp/output/audit/assumptions.json` - Production assumptions output

## 4. Identified Gaps

### Gap 1: No Integration Module in Main Codebase
No `greenlang/assumptions/` package that provides a clean SDK for other services to
interact with the assumptions registry using standard patterns (config, metrics, setup facade).

### Gap 2: No Prometheus Metrics (Standard Pattern)
No `greenlang/assumptions/metrics.py` following the standard GreenLang Prometheus pattern
used by other services (orchestrator, schema, normalizer).

### Gap 3: No Service Setup Facade
No `configure_assumptions_service(app)` / `get_assumptions_service(app)` pattern matching
other GreenLang services.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 agent has its own in-memory storage and doesn't delegate to a comprehensive
integration module that could provide persistent storage, caching, and shared state.

### Gap 5: No Standard REST API Router
No `greenlang/assumptions/api/router.py` with FastAPI endpoints following the standard
GreenLang API pattern used by all other services.

### Gap 6: No Standard Deployment Manifests
No K8s manifests in `deployment/kubernetes/assumptions-service/` following the standard
pattern.

### Gap 7: No Database Migration
No `V024__assumptions_service.sql` in the standard migration directory for persistent
assumption storage, audit logs, and scenario metadata.

### Gap 8: No Standard Monitoring
No dashboard/alerts in `deployment/monitoring/` following standard patterns.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/assumptions-ci.yml` following the standard GreenLang CI pattern.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for assumptions registry operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/assumptions/
  __init__.py           # Public API exports
  config.py             # AssumptionsConfig with GL_ASSUMPTIONS_ env prefix
  models.py             # Pydantic v2 models (re-export + enhance from foundation agent)
  registry.py           # AssumptionRegistry: create/get/update/delete/list assumptions
  scenarios.py          # ScenarioManager: create/get/update/delete/list scenarios
  validator.py          # AssumptionValidator: validate values, rules, custom validators
  provenance.py         # ProvenanceTracker: SHA-256 hash chain, audit trail
  dependencies.py       # DependencyTracker: upstream/downstream graph, impact analysis
  metrics.py            # 12 Prometheus metrics
  setup.py              # AssumptionsService facade, configure/get
  api/
    __init__.py
    router.py           # FastAPI router (18 endpoints)
```

### 5.2 Database Schema (V024)
```sql
CREATE SCHEMA assumptions_service;
-- assumptions (main registry)
-- assumption_versions (hypertable - immutable version history)
-- scenarios (scenario definitions)
-- scenario_overrides (scenario-specific value overrides)
-- assumption_change_log (hypertable - full audit trail)
-- assumption_dependencies (dependency graph edges)
```

### 5.3 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| `gl_assumptions_operations_total` | Counter | Total operations by type, result |
| `gl_assumptions_operation_duration_seconds` | Histogram | Operation latency |
| `gl_assumptions_validations_total` | Counter | Validations by result |
| `gl_assumptions_validation_failures_total` | Counter | Validation failures by rule |
| `gl_assumptions_scenario_accesses_total` | Counter | Scenario value accesses |
| `gl_assumptions_version_creates_total` | Counter | New versions created |
| `gl_assumptions_change_log_entries` | Gauge | Total change log entries |
| `gl_assumptions_total` | Gauge | Total registered assumptions |
| `gl_assumptions_scenarios_total` | Gauge | Total scenarios |
| `gl_assumptions_cache_hits_total` | Counter | Assumption cache hits |
| `gl_assumptions_cache_misses_total` | Counter | Assumption cache misses |
| `gl_assumptions_dependency_depth` | Histogram | Dependency chain depths |

### 5.4 API Endpoints (18)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/assumptions` | Create assumption |
| GET | `/v1/assumptions` | List assumptions (with filters) |
| GET | `/v1/assumptions/{id}` | Get assumption by ID |
| PUT | `/v1/assumptions/{id}` | Update assumption |
| DELETE | `/v1/assumptions/{id}` | Delete assumption |
| GET | `/v1/assumptions/{id}/versions` | Get version history |
| GET | `/v1/assumptions/{id}/value` | Get value (with optional scenario) |
| PUT | `/v1/assumptions/{id}/value` | Set value |
| POST | `/v1/assumptions/validate` | Validate assumption value |
| POST | `/v1/scenarios` | Create scenario |
| GET | `/v1/scenarios` | List scenarios |
| GET | `/v1/scenarios/{id}` | Get scenario |
| PUT | `/v1/scenarios/{id}` | Update scenario |
| DELETE | `/v1/scenarios/{id}` | Delete scenario |
| GET | `/v1/assumptions/{id}/dependencies` | Get dependency graph |
| GET | `/v1/assumptions/{id}/sensitivity` | Sensitivity analysis |
| POST | `/v1/assumptions/export` | Export all assumptions |
| POST | `/v1/assumptions/import` | Import assumptions |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/assumptions/__init__.py` - Public API exports (50+ symbols)
2. Create `greenlang/assumptions/config.py` - AssumptionsConfig with GL_ASSUMPTIONS_ env prefix
3. Create `greenlang/assumptions/models.py` - Pydantic v2 models: Assumption, AssumptionVersion, Scenario, ChangeLogEntry, ValidationResult, DependencyNode, etc.
4. Create `greenlang/assumptions/registry.py` - AssumptionRegistry wrapping foundation agent with persistent-ready interface
5. Create `greenlang/assumptions/scenarios.py` - ScenarioManager for scenario CRUD + override resolution
6. Create `greenlang/assumptions/validator.py` - AssumptionValidator for rule evaluation + custom validators
7. Create `greenlang/assumptions/provenance.py` - ProvenanceTracker with SHA-256 hash chain
8. Create `greenlang/assumptions/dependencies.py` - DependencyTracker for upstream/downstream graph
9. Create `greenlang/assumptions/metrics.py` - 12 Prometheus metrics
10. Create `greenlang/assumptions/api/router.py` - FastAPI router with 18 endpoints
11. Create `greenlang/assumptions/setup.py` - AssumptionsService facade
12. Update `greenlang/agents/foundation/assumptions_registry.py` - Add ASSUMPTIONS_SDK_AVAILABLE flag, delegate to SDK when available

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V024__assumptions_service.sql`
2. Create K8s manifests in `deployment/kubernetes/assumptions-service/`
3. Create `deployment/monitoring/dashboards/assumptions-service.json`
4. Create `deployment/monitoring/alerts/assumptions-service-alerts.yaml`
5. Create `.github/workflows/assumptions-ci.yml`

### Phase 3: Tests (Test Engineer)
1-14. Create unit, integration, and load tests in `tests/*/assumptions_service/`

### Phase 4: Documentation (Tech Writer)
1-4. Create operational runbooks

## 7. Success Criteria
- Integration module provides clean SDK for all assumption operations
- Foundation agent delegates to integration module
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V024 database migration for persistent assumption storage
- 4 operational runbooks
- 200+ new tests passing
- Zero-hallucination guarantees maintained (no implicit defaults)
