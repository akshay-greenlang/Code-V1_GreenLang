# PRD: AGENT-FOUND-003 - GreenLang Unit & Reference Normalizer

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-003 |
| **Agent ID** | GL-FOUND-X-003 |
| **Component** | Unit & Reference Normalizer |
| **Category** | Foundations Agent |
| **Priority** | P0 - Critical (blocks downstream calculations) |
| **Status** | Core Complete (~90%), Integration Gap-Fill In Progress |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires a deterministic, audit-ready normalization authority for
units, conversions, and reference data across all sectors/domains. Green/sustainability
datasets frequently arrive with messy unit strings, mixed units, non-standard emissions
expressions, and inconsistent entity naming that must be standardized before calculations.

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/unit_normalizer.py` (1,910 lines)
- `UnitNormalizerAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-003)
- `UnitDimension` enum: MASS, ENERGY, VOLUME, AREA, DISTANCE, EMISSIONS, CURRENCY, TIME
- `GHGType` enum with IPCC AR6 GWP values (CO2=1, CH4=29.8, N2O=273)
- Unit conversion across 8 dimensions with Decimal precision
- Fuel name standardization (50+ mappings)
- Material name standardization (40+ mappings)
- Dimensional analysis to prevent invalid conversions
- Currency conversion with date-specific rates
- Complete provenance hash (SHA-256) for every conversion
- Custom conversion factors per tenant

### 3.2 Layer 2: Data Utility
**File**: `greenlang/data/unit_normalizer.py` (440 lines)
- `UnitNormalizer` class for case-insensitive unit normalization
- 200+ canonical unit mappings (energy, mass, emissions, volume, area, etc.)
- Fixes kWh vs kwh case sensitivity issues
- Sources: SI, EPA, IPCC, IEA

### 3.3 Layer 3: Standalone Monorepo
**Root**: `greenlang-normalizer/` (100+ files)

| Package | Purpose |
|---------|---------|
| `gl-normalizer-core` | Parser (Pint integration, AST, preprocessor, locale), Converter (engine, factors, validators, contexts), Dimensions (analyzer, compatibility, constants), Resolution (pipeline, matchers, scorers, thresholds, models), Vocabulary (manager, registry, validators, loader, models), Policy (engine, loader, compliance, defaults, models), Audit (chain, builder, serializer, schema), Errors (codes, exceptions, factory, response) |
| `gl-normalizer-service` | FastAPI microservice: API routes, models, deps, middleware (auth, rate_limit, audit), audit (storage, publisher, outbox, chain, models, retention), admin, jobs |
| `gl-normalizer-sdk` | Python SDK: sync/async clients, models, exceptions, vocab provider, caching |
| `gl-normalizer-cli` | CLI: normalize, batch, vocab, config commands |
| `review-console` | Admin UI (React frontend + Python backend) for reviewing low-confidence mappings |

**Infrastructure** (in `greenlang-normalizer/infrastructure/`):
- Docker: 3 Dockerfiles + 2 docker-compose
- K8s: base manifests (7) + 3 env overlays (dev/staging/prod)
- Terraform: 4 modules (EKS, RDS, Kafka, S3) + 2 envs (dev, prod)
- Monitoring: Prometheus rules + Grafana dashboard

**Vocabulary** (`greenlang-normalizer/vocab/`):
- `units/unit_registry.yaml` - Unit definitions
- `fuels/common_fuels.yaml` - Fuel reference data
- `materials/common_materials.yaml` - Material reference data
- `processes/common_processes.yaml` - Process reference data
- `schemas/vocabulary_entry.yaml` - Schema for vocab entries

**Tests** (`greenlang-normalizer/tests/`):
- Golden tests: unit conversion, entity resolution, full pipeline
- Property tests: conversion, roundtrip, dimension properties
- Cross-validation: unit consistency
- Integration: API integration

**Config** (`greenlang-normalizer/config/`):
- `canonical_units.yaml` - Canonical unit definitions
- `confidence_thresholds.yaml` - Resolution confidence settings
- `policy_defaults.yaml` - Default conversion policies

### 3.4 Additional Files
- `tests/agents/foundation/test_unit_normalizer.py` (1,006 lines)
- `greenlang/utilities/utils/unit_conversion.py` - Utility conversion functions
- `greenlang/utilities/utils/unit_converter.py` - Utility converter class
- `greenlang/utils/unit_converter.py` - Legacy converter
- `greenlang/utils/unit_conversion.py` - Legacy conversion functions
- `greenlang/agents/calculation/emissions/unit_converter.py` - Emissions-specific converter
- `greenlang/agents/intelligence/runtime/units.py` - Runtime unit helpers
- `greenlang/schema/units/` - Schema-level unit validation (part of AGENT-FOUND-002)
- `GreenLang_Agents_PRD_402/PRD_GL-FOUND-X-003_Agent.md` - Existing detailed PRD
- `GreenLang_Agents_PRD_402/GL-FOUND-X-003_Implementation_Plan.md` - Existing impl plan

## 4. Identified Gaps

### Gap 1: No Integration Module in Main Codebase
No `greenlang/normalizer/` package that integrates the monorepo's core library into the
main GreenLang codebase with standard patterns (metrics, setup facade, config).

### Gap 2: No Prometheus Metrics (Standard Pattern)
The monorepo has its own monitoring, but no `greenlang/normalizer/metrics.py` following the
standard GreenLang Prometheus pattern used by other services.

### Gap 3: No Service Setup Facade
No `configure_normalizer_service(app)` / `get_normalizer_service(app)` pattern matching
other GreenLang services.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 agent has its own conversion logic and doesn't delegate to the comprehensive
monorepo's core library.

### Gap 5: No Standard Deployment Manifests
The main `deployment/` directory has no normalizer-service manifests following the standard
K8s pattern used by all other services.

### Gap 6: No Database Migration
No `V023__normalizer_service.sql` in the standard migration directory for conversion
audit logs and vocabulary metadata persistence.

### Gap 7: No Standard Monitoring
No dashboard/alerts in `deployment/monitoring/` following standard patterns.

### Gap 8: No CI/CD Pipeline
No `.github/workflows/normalizer-ci.yml` following the standard GreenLang CI pattern.

### Gap 9: No Operational Runbooks
No `docs/runbooks/` for normalizer operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/normalizer/
  __init__.py           # Public API exports
  config.py             # NormalizerConfig with GL_NORMALIZER_ env prefix
  models.py             # ConversionResult, NormalizationResult, EntityMatch, etc.
  converter.py          # UnitConverter: convert(), batch_convert(), supported_units()
  entity_resolver.py    # EntityResolver: resolve_fuel(), resolve_material(), resolve_process()
  dimensional.py        # DimensionalAnalyzer: check_compatibility(), get_dimension()
  provenance.py         # ConversionProvenance: hash chain, audit trail
  metrics.py            # 12 Prometheus metrics
  setup.py              # NormalizerService facade, configure/get
  api/
    __init__.py
    router.py           # FastAPI router (15 endpoints)
```

### 5.2 Database Schema (V023)
```sql
CREATE SCHEMA normalizer_service;
-- conversion_audit_log (hypertable)
-- entity_resolution_log (hypertable)
-- vocabulary_versions
-- canonical_units
-- custom_conversion_factors
```

### 5.3 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| `gl_normalizer_conversions_total` | Counter | Total conversions by dimension, result |
| `gl_normalizer_conversion_duration_seconds` | Histogram | Conversion latency |
| `gl_normalizer_entity_resolutions_total` | Counter | Entity resolutions by type, confidence |
| `gl_normalizer_resolution_duration_seconds` | Histogram | Resolution latency |
| `gl_normalizer_dimension_errors_total` | Counter | Dimensional mismatch errors |
| `gl_normalizer_gwp_conversions_total` | Counter | GWP conversions by gas type |
| `gl_normalizer_batch_size` | Histogram | Batch conversion sizes |
| `gl_normalizer_vocabulary_entries` | Gauge | Vocabulary entries by type |
| `gl_normalizer_cache_hits_total` | Counter | Conversion cache hits |
| `gl_normalizer_cache_misses_total` | Counter | Conversion cache misses |
| `gl_normalizer_active_conversions` | Gauge | Currently running conversions |
| `gl_normalizer_custom_factors` | Gauge | Custom conversion factors by tenant |

### 5.4 API Endpoints (15)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/normalize/convert` | Convert single value |
| POST | `/v1/normalize/convert/batch` | Batch conversion |
| POST | `/v1/normalize/resolve/fuel` | Resolve fuel name |
| POST | `/v1/normalize/resolve/material` | Resolve material name |
| POST | `/v1/normalize/resolve/process` | Resolve process name |
| POST | `/v1/normalize/resolve/batch` | Batch entity resolution |
| GET | `/v1/normalize/units` | List supported units |
| GET | `/v1/normalize/units/{dimension}` | List units by dimension |
| GET | `/v1/normalize/dimensions` | List supported dimensions |
| GET | `/v1/normalize/gwp` | List GWP values |
| GET | `/v1/normalize/vocabulary/{type}` | Get vocabulary entries |
| GET | `/v1/normalize/vocabulary/{type}/versions` | Vocabulary version history |
| POST | `/v1/normalize/validate/dimension` | Check dimension compatibility |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/normalizer/__init__.py` - Public API exports (50+ symbols)
2. Create `greenlang/normalizer/config.py` - NormalizerConfig with GL_NORMALIZER_ env prefix
3. Create `greenlang/normalizer/models.py` - ConversionResult, NormalizationResult, EntityMatch, DimensionInfo, GWPInfo, VocabularyEntry, ConversionProvenance
4. Create `greenlang/normalizer/converter.py` - UnitConverter wrapping foundation agent + monorepo core
5. Create `greenlang/normalizer/entity_resolver.py` - EntityResolver for fuel/material/process resolution
6. Create `greenlang/normalizer/dimensional.py` - DimensionalAnalyzer for compatibility checks
7. Create `greenlang/normalizer/provenance.py` - Provenance tracking with SHA-256 hash chain
8. Create `greenlang/normalizer/metrics.py` - 12 Prometheus metrics
9. Create `greenlang/normalizer/api/router.py` - FastAPI router with 15 endpoints
10. Create `greenlang/normalizer/setup.py` - NormalizerService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V023__normalizer_service.sql`
2. Create K8s manifests in `deployment/kubernetes/normalizer-service/`
3. Create `deployment/monitoring/dashboards/normalizer-service.json`
4. Create `deployment/monitoring/alerts/normalizer-service-alerts.yaml`
5. Create `.github/workflows/normalizer-ci.yml`

### Phase 3: Tests (Test Engineer)
1-8. Create unit, integration, and load tests in `tests/*/normalizer_service/`

### Phase 4: Documentation (Tech Writer)
1-4. Create operational runbooks

## 7. Success Criteria
- Integration module provides clean SDK for all conversion/resolution needs
- Foundation agent delegates to integration module
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V023 database migration for audit persistence
- 4 operational runbooks
- 100+ new tests passing
- 100% conversion accuracy for supported dimensions
