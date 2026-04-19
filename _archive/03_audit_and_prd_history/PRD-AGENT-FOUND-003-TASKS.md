# AGENT-FOUND-003: Unit & Reference Normalizer - Gap-Fill Tasks

## Phase 1: Core Integration (Backend Developer)

- [ ] Create `greenlang/normalizer/__init__.py` - Public API exports (50+ symbols): UnitConverter, EntityResolver, DimensionalAnalyzer, NormalizerConfig, ConversionResult, NormalizationResult, EntityMatch, configure_normalizer_service, get_normalizer_service, NormalizerService
- [ ] Create `greenlang/normalizer/config.py` - NormalizerConfig with `GL_NORMALIZER_` env prefix, from_env(), thread-safe singleton (get_config/set_config/reset_config), defaults for precision, GWP version, cache settings
- [ ] Create `greenlang/normalizer/models.py` - Pydantic v2 models: ConversionResult, BatchConversionResult, NormalizationResult, EntityMatch, EntityResolutionResult, DimensionInfo, GWPInfo, VocabularyEntry, ConversionProvenance, UnitInfo
- [ ] Create `greenlang/normalizer/converter.py` - UnitConverter class: convert(), batch_convert(), convert_ghg(), supported_units(), supported_dimensions(), get_conversion_factor(); wraps foundation agent logic with Decimal precision, dimensional validation, GWP conversion (AR5/AR6), provenance tracking
- [ ] Create `greenlang/normalizer/entity_resolver.py` - EntityResolver class: resolve_fuel(), resolve_material(), resolve_process(), batch_resolve(), search_vocabulary(); confidence scoring, exact/alias/fuzzy matching, provenance tracking
- [ ] Create `greenlang/normalizer/dimensional.py` - DimensionalAnalyzer: check_compatibility(), get_dimension(), get_base_unit(), list_dimensions(), is_valid_unit(); prevent invalid cross-dimension conversions
- [ ] Create `greenlang/normalizer/provenance.py` - ConversionProvenanceTracker: record_conversion(), record_resolution(), build_chain_hash(), get_audit_trail(), export_json(); SHA-256 hash chain linking all operations
- [ ] Create `greenlang/normalizer/metrics.py` - 12 Prometheus metrics with graceful fallback, matching pattern from greenlang/schema/metrics.py
- [ ] Create `greenlang/normalizer/api/__init__.py` - API package init
- [ ] Create `greenlang/normalizer/api/router.py` - FastAPI router with 15 endpoints: convert, convert/batch, resolve/fuel, resolve/material, resolve/process, resolve/batch, units, units/{dimension}, dimensions, gwp, vocabulary/{type}, vocabulary/{type}/versions, validate/dimension, health, metrics
- [ ] Create `greenlang/normalizer/setup.py` - NormalizerService facade, configure_normalizer_service(app), get_normalizer_service(app), lifespan management
- [ ] Update `greenlang/agents/foundation/unit_normalizer.py` - Add NORMALIZER_SDK_AVAILABLE flag, delegate core conversions to greenlang.normalizer.converter when available

## Phase 2: Infrastructure (DevOps Engineer)

- [ ] Create `deployment/database/migrations/sql/V023__normalizer_service.sql` - 5 tables (conversion_audit_log, entity_resolution_log, vocabulary_versions, canonical_units, custom_conversion_factors) + hypertables on audit logs + indexes + RLS + continuous aggregate + seed data (200+ canonical units, 50+ fuels, 40+ materials)
- [ ] Create `deployment/kubernetes/normalizer-service/deployment.yaml` - 2 replicas, resource limits, probes, init containers
- [ ] Create `deployment/kubernetes/normalizer-service/service.yaml` - ClusterIP port 8080
- [ ] Create `deployment/kubernetes/normalizer-service/configmap.yaml` - All GL_NORMALIZER_ vars
- [ ] Create `deployment/kubernetes/normalizer-service/hpa.yaml` - HPA min=2 max=6 + PDB
- [ ] Create `deployment/kubernetes/normalizer-service/networkpolicy.yaml` - Default deny + specific rules
- [ ] Create `deployment/kubernetes/normalizer-service/servicemonitor.yaml` - Prometheus scrape config
- [ ] Create `deployment/kubernetes/normalizer-service/kustomization.yaml` - Kustomize base
- [ ] Create `deployment/monitoring/dashboards/normalizer-service.json` - Grafana dashboard (20+ panels)
- [ ] Create `deployment/monitoring/alerts/normalizer-service-alerts.yaml` - Alert rules (12+ alerts)
- [ ] Create `.github/workflows/normalizer-ci.yml` - CI/CD pipeline (8+ jobs)

## Phase 3: Tests (Test Engineer)

- [ ] Create `tests/unit/normalizer_service/conftest.py` - Shared fixtures (sample conversions, mock vocabularies, GWP values)
- [ ] Create `tests/unit/normalizer_service/test_config.py` - Config, env overrides, defaults, validation
- [ ] Create `tests/unit/normalizer_service/test_models.py` - Model serialization, validation
- [ ] Create `tests/unit/normalizer_service/test_converter.py` - Unit conversion: mass, energy, volume, emissions, GWP, Decimal precision, dimensional errors
- [ ] Create `tests/unit/normalizer_service/test_entity_resolver.py` - Fuel/material/process resolution, confidence scoring, fuzzy matching
- [ ] Create `tests/unit/normalizer_service/test_dimensional.py` - Dimension compatibility, invalid conversions, base units
- [ ] Create `tests/unit/normalizer_service/test_provenance.py` - Hash chain, audit trail, export
- [ ] Create `tests/unit/normalizer_service/test_metrics.py` - All 12 Prometheus metrics
- [ ] Create `tests/unit/normalizer_service/test_setup.py` - Facade, configure, get, lifespan
- [ ] Create `tests/unit/normalizer_service/test_api_router.py` - All 15 API endpoints
- [ ] Create `tests/integration/normalizer_service/test_end_to_end.py` - Full conversion pipelines
- [ ] Create `tests/integration/normalizer_service/test_entity_resolution.py` - Full entity resolution flows
- [ ] Create `tests/integration/normalizer_service/test_api_integration.py` - API with TestClient
- [ ] Create `tests/load/normalizer_service/test_normalizer_load.py` - 50 concurrent conversions, large batch, throughput

## Phase 4: Documentation (Tech Writer)

- [ ] Create `docs/runbooks/normalizer-service-down.md` - Detection, diagnosis, recovery
- [ ] Create `docs/runbooks/conversion-accuracy-drift.md` - Factor verification, GWP version audit
- [ ] Create `docs/runbooks/entity-resolution-low-confidence.md` - Vocabulary curation, threshold tuning
- [ ] Create `docs/runbooks/normalizer-high-latency.md` - Cache diagnosis, vocabulary optimization
