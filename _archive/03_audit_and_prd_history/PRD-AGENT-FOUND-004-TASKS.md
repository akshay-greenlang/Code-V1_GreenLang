# AGENT-FOUND-004: Assumptions Registry - Gap-Fill Tasks

## Phase 1: Core Integration (Backend Developer)

- [ ] Create `greenlang/assumptions/__init__.py` - Public API exports (50+ symbols): AssumptionRegistry, ScenarioManager, AssumptionValidator, DependencyTracker, ProvenanceTracker, AssumptionsConfig, all models, configure_assumptions_service, get_assumptions_service
- [ ] Create `greenlang/assumptions/config.py` - AssumptionsConfig with `GL_ASSUMPTIONS_` env prefix, from_env(), thread-safe singleton (get_config/set_config/reset_config), defaults for max_versions, validation, caching, scenario settings
- [ ] Create `greenlang/assumptions/models.py` - Pydantic v2 models: AssumptionDataType, AssumptionCategory, ScenarioType, ChangeType, ValidationSeverity enums; Assumption, AssumptionVersion, Scenario, ScenarioOverride, ChangeLogEntry, ValidationRule, ValidationResult, DependencyNode, AssumptionMetadata, SensitivityResult
- [ ] Create `greenlang/assumptions/registry.py` - AssumptionRegistry class: create_assumption(), get_assumption(), update_assumption(), delete_assumption(), list_assumptions(), get_value(), set_value(); wraps foundation agent logic with config, metrics, provenance
- [ ] Create `greenlang/assumptions/scenarios.py` - ScenarioManager class: create_scenario(), get_scenario(), update_scenario(), delete_scenario(), list_scenarios(), resolve_value(), get_overrides(); scenario inheritance chain resolution
- [ ] Create `greenlang/assumptions/validator.py` - AssumptionValidator class: validate(), validate_value(), register_custom_validator(), check_rules(), validate_data_type(); min/max/allowed/regex/custom validation
- [ ] Create `greenlang/assumptions/provenance.py` - ProvenanceTracker: record_change(), build_chain_hash(), get_audit_trail(), verify_chain(), export_json(); SHA-256 hash chain linking all operations
- [ ] Create `greenlang/assumptions/dependencies.py` - DependencyTracker: register_dependency(), get_upstream(), get_downstream(), get_impact(), detect_cycles(), get_calculation_assumptions(); DAG-based dependency graph
- [ ] Create `greenlang/assumptions/metrics.py` - 12 Prometheus metrics with graceful fallback, matching pattern from greenlang/normalizer/metrics.py
- [ ] Create `greenlang/assumptions/api/__init__.py` - API package init
- [ ] Create `greenlang/assumptions/api/router.py` - FastAPI router with 18 endpoints: CRUD for assumptions, CRUD for scenarios, value get/set, validate, versions, dependencies, sensitivity, export, import, health, metrics
- [ ] Create `greenlang/assumptions/setup.py` - AssumptionsService facade, configure_assumptions_service(app), get_assumptions_service(app), lifespan management
- [ ] Update `greenlang/agents/foundation/assumptions_registry.py` - Add ASSUMPTIONS_SDK_AVAILABLE flag, delegate core operations to greenlang.assumptions.registry when available

## Phase 2: Infrastructure (DevOps Engineer)

- [ ] Create `deployment/database/migrations/sql/V024__assumptions_service.sql` - 6 tables (assumptions, assumption_versions, scenarios, scenario_overrides, assumption_change_log, assumption_dependencies) + hypertables on versions and change_log + indexes + RLS + continuous aggregates + seed data (common emission factors, default scenarios)
- [ ] Create `deployment/kubernetes/assumptions-service/deployment.yaml` - 2 replicas, resource limits, probes, init containers
- [ ] Create `deployment/kubernetes/assumptions-service/service.yaml` - ClusterIP port 8080
- [ ] Create `deployment/kubernetes/assumptions-service/configmap.yaml` - All GL_ASSUMPTIONS_ vars
- [ ] Create `deployment/kubernetes/assumptions-service/hpa.yaml` - HPA min=2 max=6 + PDB
- [ ] Create `deployment/kubernetes/assumptions-service/networkpolicy.yaml` - Default deny + specific rules
- [ ] Create `deployment/kubernetes/assumptions-service/servicemonitor.yaml` - Prometheus scrape config
- [ ] Create `deployment/kubernetes/assumptions-service/kustomization.yaml` - Kustomize base
- [ ] Create `deployment/monitoring/dashboards/assumptions-service.json` - Grafana dashboard (20+ panels)
- [ ] Create `deployment/monitoring/alerts/assumptions-service-alerts.yaml` - Alert rules (12+ alerts)
- [ ] Create `.github/workflows/assumptions-ci.yml` - CI/CD pipeline (8+ jobs)

## Phase 3: Tests (Test Engineer)

- [ ] Create `tests/unit/assumptions_service/conftest.py` - Shared fixtures (sample assumptions, scenarios, validation rules)
- [ ] Create `tests/unit/assumptions_service/test_config.py` - Config, env overrides, defaults, validation
- [ ] Create `tests/unit/assumptions_service/test_models.py` - Model serialization, validation, enums
- [ ] Create `tests/unit/assumptions_service/test_registry.py` - CRUD operations, value get/set, version history, filters
- [ ] Create `tests/unit/assumptions_service/test_scenarios.py` - Scenario CRUD, override resolution, inheritance
- [ ] Create `tests/unit/assumptions_service/test_validator.py` - Min/max/allowed/regex/custom validators, data type checks
- [ ] Create `tests/unit/assumptions_service/test_provenance.py` - Hash chain, audit trail, verify, export
- [ ] Create `tests/unit/assumptions_service/test_dependencies.py` - Upstream/downstream, impact analysis, cycle detection
- [ ] Create `tests/unit/assumptions_service/test_metrics.py` - All 12 Prometheus metrics
- [ ] Create `tests/unit/assumptions_service/test_setup.py` - Facade, configure, get, lifespan
- [ ] Create `tests/unit/assumptions_service/test_api_router.py` - All 18 API endpoints
- [ ] Create `tests/integration/assumptions_service/test_end_to_end.py` - Full assumption lifecycle
- [ ] Create `tests/integration/assumptions_service/test_scenario_workflows.py` - Multi-scenario workflows
- [ ] Create `tests/integration/assumptions_service/test_api_integration.py` - API with TestClient
- [ ] Create `tests/load/assumptions_service/test_assumptions_load.py` - 50 concurrent operations, large registry, throughput

## Phase 4: Documentation (Tech Writer)

- [ ] Create `docs/runbooks/assumptions-service-down.md` - Detection, diagnosis, recovery
- [ ] Create `docs/runbooks/assumption-validation-failures.md` - Rule debugging, threshold tuning
- [ ] Create `docs/runbooks/scenario-drift-detection.md` - Scenario comparison, override audit
- [ ] Create `docs/runbooks/assumptions-audit-compliance.md` - Audit trail verification, export procedures
