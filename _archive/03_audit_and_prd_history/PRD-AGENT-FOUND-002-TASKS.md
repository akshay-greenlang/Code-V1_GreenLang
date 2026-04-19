# AGENT-FOUND-002: Schema Compiler & Validator - Gap-Fill Tasks

## Phase 0: Immediate Fix (DONE)
- [x] Uncomment IR/Compiler imports in `greenlang/schema/compiler/__init__.py` (lines 131-140)
- [x] Uncomment IR/Compiler exports in `greenlang/schema/compiler/__init__.py` (lines 222-227)

## Phase 1: Core Completion (Backend Developer)

- [ ] Create `greenlang/schema/metrics.py` - 12 Prometheus metrics (counters, histograms, gauges) with prometheus_client graceful fallback, matching OBS-005/AGENT-FOUND-001 patterns
- [ ] Create `greenlang/schema/setup.py` - SchemaService facade class, configure_schema_service(app), get_schema_service(app), lifespan management, registry initialization
- [ ] Update `greenlang/agents/foundation/schema_compiler.py` - Refactor `_validate_against_schema()` to delegate to `greenlang.schema.sdk.validate()`, keep BaseAgent interface, add `SCHEMA_SDK_AVAILABLE` flag with graceful fallback
- [ ] Update `greenlang/schema/__init__.py` - Add metrics and setup exports
- [ ] Update `greenlang/schema/version.py` - Bump to 1.0.0

## Phase 2: Infrastructure (DevOps Engineer)

- [ ] Create `deployment/database/migrations/sql/V022__schema_service.sql` - 3 tables (schema_registry, validation_audit_log, schema_cache_metadata) + hypertable on validation_audit_log + indexes + RLS + continuous aggregate + seed data
- [ ] Create `deployment/kubernetes/schema-service/deployment.yaml` - 2 replicas, resource limits, probes, init containers
- [ ] Create `deployment/kubernetes/schema-service/service.yaml` - ClusterIP port 8080
- [ ] Create `deployment/kubernetes/schema-service/configmap.yaml` - Full configuration with GL_SCHEMA_ prefix
- [ ] Create `deployment/kubernetes/schema-service/hpa.yaml` - HPA min=2 max=6 + PDB
- [ ] Create `deployment/kubernetes/schema-service/networkpolicy.yaml` - Default deny + specific rules
- [ ] Create `deployment/kubernetes/schema-service/servicemonitor.yaml` - Prometheus scrape config
- [ ] Create `deployment/kubernetes/schema-service/kustomization.yaml` - Kustomize base
- [ ] Create `deployment/monitoring/dashboards/schema-service.json` - Grafana dashboard (20+ panels)
- [ ] Create `deployment/monitoring/alerts/schema-service-alerts.yaml` - Alert rules (12+ alerts)
- [ ] Create `.github/workflows/schema-ci.yml` - 7+ job CI/CD pipeline

## Phase 3: Tests (Test Engineer)

- [ ] Create `tests/unit/schema_service/conftest.py` - Shared fixtures
- [ ] Create `tests/unit/schema_service/test_metrics.py` - All 12 Prometheus metrics
- [ ] Create `tests/unit/schema_service/test_setup.py` - Facade, configure, get, lifespan
- [ ] Create `tests/integration/schema_service/conftest.py` - Integration fixtures
- [ ] Create `tests/integration/schema_service/test_foundation_delegation.py` - Foundation agent delegates to Layer 2
- [ ] Create `tests/integration/schema_service/test_api_integration.py` - Full API with TestClient
- [ ] Create `tests/integration/schema_service/test_end_to_end.py` - SDK validate, compile, batch, fixes
- [ ] Create `tests/load/schema_service/test_schema_load.py` - Concurrent validations, large payloads, batch throughput
- [ ] Verify all existing tests still pass after changes

## Phase 4: Documentation (Tech Writer)

- [ ] Create `docs/runbooks/schema-service-down.md` - Detection, diagnosis, recovery
- [ ] Create `docs/runbooks/high-validation-errors.md` - Error spike diagnosis, schema debugging
- [ ] Create `docs/runbooks/schema-cache-corruption.md` - Cache inspection, warmup, rebuild
- [ ] Create `docs/runbooks/compilation-timeout.md` - Schema complexity diagnosis, ReDoS investigation
