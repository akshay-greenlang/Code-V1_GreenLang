# PRD: AGENT-FOUND-002 - GreenLang Schema Compiler & Validator

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-002 |
| **Agent ID** | GL-FOUND-X-002 |
| **Component** | Schema Compiler & Validator |
| **Category** | Foundations Agent |
| **Priority** | P0 - Critical |
| **Status** | Core Complete (~95%), Infrastructure Gap-Fill In Progress |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires a production-grade schema compilation and validation
engine that can:

- Parse and compile JSON Schema (Draft 2020-12) with GreenLang-specific extensions
- Validate climate data payloads with zero-hallucination guarantees
- Provide actionable error hints with machine-applicable fix suggestions (JSON Patch RFC 6902)
- Support unit consistency checking for climate-specific units (kgCO2e, tCO2e, MWh, etc.)
- Handle type coercion safely with full audit trail
- Detect and prevent ReDoS attacks in schema regex patterns
- Serve as both an embeddable library and a standalone microservice

## 3. Existing Implementation (Pre-Gap-Fill)

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/schema_compiler.py` (~1,788 lines)
- `SchemaCompilerAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-002)
- Built-in `SchemaRegistry` with 3 domain schemas
- `TypeCoercionEngine` with string-to-number/boolean coercion
- `UnitConsistencyChecker` with 10 unit families
- `FixSuggestionGenerator` for machine-fixable suggestions
- JSON Schema Draft-07 validation via `jsonschema` library
- Provenance hashing (SHA-256)

### 3.2 Layer 2: Comprehensive Schema Module
**Root**: `greenlang/schema/` (68+ Python files)

| Subpackage | Files | Purpose |
|-----------|-------|---------|
| `compiler/` | 7 | Parser, AST, IR, Compiler, $ref Resolver, Regex Analyzer, Schema Self-Validator |
| `validator/` | 6 | Structural, Constraints, Units, Cross-field Rules, Core Pipeline, Linter |
| `normalizer/` | 5 | Type Coercion, Key Canonicalization, Unit Canonicalization, Engine |
| `suggestions/` | 6 | JSON Patch (RFC 6902), Safety Classification, Heuristics, Fix Engine |
| `registry/` | 4 | Schema Registry Interface, Git Backend, IR Cache + Warmup |
| `units/` | 6 | Unit Catalog, Dimensions, Conversions, SI/Climate/Finance Packs |
| `models/` | 6 | SchemaRef, ValidationOptions, Finding, Report, Patch, Config |
| `api/` | 4 | FastAPI Routes, Request/Response Models, Dependencies |
| `cli/` | 9 | Commands (validate, compile, lint, migrate), Formatters (JSON, pretty, table, text, SARIF) |
| `sdk.py` | 1 | High-level SDK: validate(), validate_batch(), compile_schema(), apply_fixes() |

### 3.3 Test Coverage
**Location**: `tests/schema/` + `tests/agents/foundation/` + `tests/unit/`

| Test Type | Files | Coverage |
|-----------|-------|----------|
| Unit tests | 20+ | Compiler, validator, normalizer, suggestions, registry, SDK |
| Integration tests | 5+ | End-to-end validation flows |
| Property-based tests | 4 | Normalization idempotency, patch monotonicity, validation determinism, coercion reversibility |
| Security tests | 5 | YAML bombs, input limits, path traversal, ReDoS, schema bombs |
| Golden tests | 5+ | Regression test schemas and expected outputs |
| Foundation agent tests | 1 | Registry, coercion, unit checking, validation, determinism |
| IR compilation tests | 1 | CompiledPattern, constraints, units, rules, hashing |

## 4. Identified Gaps

### Gap 1: Compiler Exports (FIXED)
The `compiler/__init__.py` had IR and Compiler imports commented out as TODO.
**Resolution**: Uncommented - now exports SchemaIR, PropertyIR, CompiledPattern, SchemaCompiler, CompilationResult.

### Gap 2: Prometheus Metrics Module
No standalone Prometheus metrics module for schema service self-monitoring. The API
`dependencies.py` has an inline `MetricsCollector` but no `prometheus_client` integration.

**Required**: `greenlang/schema/metrics.py` with counters, histograms, and gauges matching
the pattern used by OBS-005 and AGENT-FOUND-001.

### Gap 3: Service Setup Facade
No `setup.py` facade for FastAPI lifespan integration (unlike SLO service, alerting service,
and orchestrator which all have `configure_X(app)` / `get_X(app)` patterns).

**Required**: `greenlang/schema/setup.py` with `SchemaService` facade, `configure_schema_service(app)`,
`get_schema_service(app)`.

### Gap 4: Foundation Agent Integration
Layer 1 (`schema_compiler.py`) has its own `SchemaRegistry`, `TypeCoercionEngine`, and
`FixSuggestionGenerator` that do NOT delegate to Layer 2's more sophisticated equivalents.

**Required**: Refactor foundation agent to delegate validation to `greenlang.schema.sdk.validate()`,
keeping the BaseAgent interface intact.

### Gap 5: Kubernetes Deployment
No K8s manifests for the schema service.

**Required**: Standard GreenLang service deployment (deployment, service, configmap, HPA, PDB,
network policy, service monitor, kustomization).

### Gap 6: Database Migration
No persistent storage for schema registry metadata. Git backend exists for schema content,
but validation audit logs and cache metadata are ephemeral.

**Required**: `V022__schema_service.sql` with tables for schema_registry, validation_audit_log,
schema_cache_metadata.

### Gap 7: Monitoring & Alerting
No Grafana dashboard or Prometheus alert rules for the schema service.

**Required**: Dashboard with validation throughput, error rates, compilation latency, cache hit
rates. Alert rules for service health, high error rates, slow compilation.

### Gap 8: CI/CD Pipeline
No dedicated CI/CD pipeline for the schema service.

**Required**: `.github/workflows/schema-ci.yml` with lint, type-check, unit-test,
integration-test, security-test, coverage, build, deploy jobs.

### Gap 9: Operational Runbooks
No runbooks for schema service operations.

**Required**: Standard set (service-down, high-validation-errors, cache-corruption,
compilation-timeout).

## 5. Architecture

### 5.1 Module Structure (Final State)

```
greenlang/schema/
  __init__.py           # Public SDK exports (EXISTING)
  sdk.py                # High-level Python SDK (EXISTING)
  version.py            # Version info (EXISTING)
  constants.py          # Configuration constants (EXISTING)
  errors.py             # Error code definitions (EXISTING)
  metrics.py            # Prometheus metrics (NEW)
  setup.py              # Service facade + lifespan (NEW)
  compiler/             # AST, IR, Compilation (EXISTING - exports FIXED)
  validator/            # 7-phase validation pipeline (EXISTING)
  normalizer/           # Type coercion, canonicalization (EXISTING)
  suggestions/          # Fix engine, JSON Patch (EXISTING)
  registry/             # Git backend, IR cache (EXISTING)
  units/                # Unit catalog, SI/Climate/Finance (EXISTING)
  models/               # Pydantic v2 data models (EXISTING)
  api/                  # FastAPI REST endpoints (EXISTING)
  cli/                  # CLI commands + formatters (EXISTING)
```

### 5.2 Key Technical Decisions

- **JSON Schema Draft 2020-12** with backward compatibility for Draft-07
- **GreenLang Extensions**: `$unit`, `$dimension`, `$rules`, `$aliases`, `$deprecated`
- **7-Phase Validation Pipeline**: Parse -> Compile -> Structural -> Constraints -> Units -> Rules -> Lint
- **AST -> IR Compilation**: O(1) property lookup, precompiled regex, constraint indexing
- **ReDoS Prevention**: Regex analysis before compilation, dangerous pattern detection
- **Fix Suggestions**: RFC 6902 JSON Patch with safety classification (safe/needs_review/unsafe)
- **Git-Backed Registry**: Schema versioning with SemVer, IR caching with warmup scheduler

### 5.3 Database Schema (V022)

```sql
-- Schema registry metadata
CREATE TABLE schema_service.schema_registry (
    schema_id VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    schema_hash CHAR(64) NOT NULL,
    content_type VARCHAR(20) DEFAULT 'json',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (schema_id, version)
);

-- Validation audit log (hypertable)
CREATE TABLE schema_service.validation_audit_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    schema_id VARCHAR(255) NOT NULL,
    schema_version VARCHAR(50) NOT NULL,
    schema_hash CHAR(64) NOT NULL,
    valid BOOLEAN NOT NULL,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    payload_hash CHAR(64),
    duration_ms DOUBLE PRECISION,
    tenant_id VARCHAR(100) DEFAULT 'default',
    PRIMARY KEY (id, timestamp)
);

-- Schema cache metadata
CREATE TABLE schema_service.schema_cache_metadata (
    cache_key VARCHAR(512) PRIMARY KEY,
    schema_id VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    ir_hash CHAR(64) NOT NULL,
    compiled_at TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count BIGINT DEFAULT 0
);
```

### 5.4 Prometheus Metrics (12)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_schema_validations_total` | Counter | Total validations by schema_id, valid/invalid |
| `gl_schema_validation_duration_seconds` | Histogram | Validation latency |
| `gl_schema_compilation_duration_seconds` | Histogram | Compilation latency |
| `gl_schema_errors_total` | Counter | Validation errors by error_code |
| `gl_schema_warnings_total` | Counter | Validation warnings by warning_code |
| `gl_schema_fixes_applied_total` | Counter | Fix suggestions applied by safety level |
| `gl_schema_cache_hits_total` | Counter | IR cache hits |
| `gl_schema_cache_misses_total` | Counter | IR cache misses |
| `gl_schema_batch_size` | Histogram | Batch validation sizes |
| `gl_schema_payload_bytes` | Histogram | Payload sizes in bytes |
| `gl_schema_active_validations` | Gauge | Currently running validations |
| `gl_schema_registered_schemas` | Gauge | Number of registered schemas |

### 5.5 API Endpoints (7 - EXISTING)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/schema/validate` | Validate single payload |
| POST | `/v1/schema/validate/batch` | Batch validation |
| POST | `/v1/schema/compile` | Compile schema to IR |
| GET | `/v1/schema/{id}/versions` | List schema versions |
| GET | `/v1/schema/{id}/{version}` | Get schema details |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## 6. Completion Requirements

### Phase 1: Core Completion (Backend Developer)
1. Create `greenlang/schema/metrics.py` - 12 Prometheus metrics with graceful fallback
2. Create `greenlang/schema/setup.py` - SchemaService facade, configure_schema_service(app), get_schema_service(app)
3. Update `greenlang/agents/foundation/schema_compiler.py` - Delegate to Layer 2 SDK
4. Update `greenlang/schema/__init__.py` - Export metrics and setup
5. Update `greenlang/schema/version.py` - Bump to 1.0.0

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V022__schema_service.sql`
2. Create K8s manifests in `deployment/kubernetes/schema-service/`
3. Create `deployment/monitoring/dashboards/schema-service.json`
4. Create `deployment/monitoring/alerts/schema-service-alerts.yaml`
5. Create `.github/workflows/schema-ci.yml`

### Phase 3: Tests (Test Engineer)
1. Create `tests/unit/schema_service/test_metrics.py`
2. Create `tests/unit/schema_service/test_setup.py`
3. Create `tests/integration/schema_service/test_foundation_delegation.py`
4. Create `tests/integration/schema_service/test_api_integration.py`
5. Verify all existing tests still pass after changes

### Phase 4: Documentation (Tech Writer)
1. Create `docs/runbooks/schema-service-down.md`
2. Create `docs/runbooks/high-validation-errors.md`
3. Create `docs/runbooks/schema-cache-corruption.md`
4. Create `docs/runbooks/compilation-timeout.md`

## 7. Success Criteria

- All compiler exports working (IR, Compiler accessible from `greenlang.schema.compiler`)
- Foundation agent delegates to Layer 2 SDK
- 12 Prometheus metrics instrumented
- Service facade pattern matching other GreenLang services
- K8s deployment manifests with HPA, PDB, network policies
- Grafana dashboard with 20+ panels
- 12+ alert rules
- CI/CD pipeline with 7+ jobs
- 4 operational runbooks
- All existing tests pass (500+)
- New tests pass (50+)
- Overall coverage >= 85%

## 8. Dependencies

- **INFRA-002**: PostgreSQL + TimescaleDB (for V022 migration)
- **INFRA-003**: Redis (for IR caching)
- **INFRA-006**: Kong API Gateway (for rate limiting)
- **OBS-001**: Prometheus (for metrics collection)
- **OBS-002**: Grafana (for dashboards)
- **SEC-001**: JWT Auth (for API authentication)
- **SEC-002**: RBAC (for authorization)
