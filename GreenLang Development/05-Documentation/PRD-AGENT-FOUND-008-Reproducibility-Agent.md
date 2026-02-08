# PRD: AGENT-FOUND-008 - Reproducibility Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-008 |
| **Agent ID** | GL-FOUND-X-008 |
| **Component** | Run Reproducibility Agent (Artifact Hashing, Deterministic Replay) |
| **Category** | Foundations Agent |
| **Priority** | P1 - High (deterministic execution backbone for regulatory compliance) |
| **Status** | Layer 1 Complete (~1,613 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS performs emission calculations, regulatory compliance assessments,
and sustainability reporting that must be **provably deterministic** and **fully
reproducible**. Every calculation pipeline must produce identical results when given
identical inputs, regardless of when or where it runs. Without a production-grade
reproducibility service:

- **No artifact hashing**: Calculation outputs cannot be verified for tamper-evidence
- **No deterministic replay**: Failed or questioned calculations cannot be re-executed
- **No drift detection**: Gradual divergence from baselines goes undetected
- **No environment fingerprinting**: Platform differences silently corrupt results
- **No seed management**: Random operations produce unpredictable outcomes
- **No version pinning**: Agent/model/factor version changes silently alter results
- **No non-determinism tracking**: Sources of randomness are unidentified
- **No audit trail**: Regulators cannot verify calculation reproducibility

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/reproducibility_agent.py` (1,613 lines)
- `ReproducibilityAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-008)
- 3 enums: VerificationStatus(4), DriftSeverity(4), NonDeterminismSource(10)
- 10 Pydantic models: EnvironmentFingerprint, SeedConfiguration, VersionPin, VersionManifest, VerificationCheck, DriftDetection, ReplayConfiguration, ReproducibilityInput, ReproducibilityOutput, ReproducibilityReport
- Deterministic hashing: SHA-256 with float normalization (`_normalize_value` handles Decimal, float, dict, list)
- Input/output hash verification: `_verify_input_hash`, `_verify_output_hash`
- Environment fingerprinting: Python version, platform, architecture, dependencies, environment_hash
- Seed management: global_seed, numpy_seed, torch_seed, custom_seeds with `_apply_replay_seeds`
- Version pinning: agent, model, factor, data version pins with manifest hashing
- Drift detection: soft/hard thresholds (1%/5%), field-level drift analysis, severity classification
- Non-determinism tracking: 10 known sources (timestamp, random_seed, external_api, floating_point, dict_ordering, file_ordering, thread_scheduling, network_latency, environment_variable, dependency_version)
- Replay mode: full environment+seed+version replay configuration
- Report generation: comprehensive reports with recommendations
- Constants: DEFAULT_ABSOLUTE_TOLERANCE=1e-9, DEFAULT_RELATIVE_TOLERANCE=1e-6
- Uses: `greenlang.utilities.determinism.clock.DeterministicClock`, `greenlang.utilities.determinism.random.DeterministicRandom`, `greenlang.utilities.determinism.uuid.content_hash, deterministic_id`
- In-memory storage (no database persistence)

### 3.2 Layer 1 Tests
None found.

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/reproducibility/` package providing a clean SDK for other agents/services.

### Gap 2: No Prometheus Metrics
No `greenlang/reproducibility/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_reproducibility(app)` / `get_reproducibility(app)` pattern.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 has in-memory storage; doesn't delegate to persistent integration module.

### Gap 5: No REST API Router
No `greenlang/reproducibility/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/reproducibility-service/` manifests.

### Gap 7: No Database Migration
No `V028__reproducibility_service.sql` for persistent verification storage.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/reproducibility-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for reproducibility operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/reproducibility/
  __init__.py              # Public API exports
  config.py                # ReproducibilityConfig with GL_REPRODUCIBILITY_ env prefix
  models.py                # Pydantic v2 models (re-export + enhance from foundation agent)
  artifact_hasher.py       # ArtifactHasher: SHA-256 deterministic hashing with normalization
  determinism_verifier.py  # DeterminismVerifier: input/output hash verification engine
  drift_detector.py        # DriftDetector: baseline comparison, field-level analysis
  replay_engine.py         # ReplayEngine: environment+seed+version replay orchestration
  environment_capture.py   # EnvironmentCapture: platform fingerprinting, dependency scanning
  seed_manager.py          # SeedManager: global/numpy/torch/custom seed management
  version_pinner.py        # VersionPinner: agent/model/factor/data version manifest
  provenance.py            # ProvenanceTracker: SHA-256 hash chain for verification mutations
  metrics.py               # 12 Prometheus metrics
  setup.py                 # ReproducibilityService facade, configure/get
  api/
    __init__.py
    router.py              # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V028)
```sql
CREATE SCHEMA reproducibility_service;
-- verification_runs (execution verification records with hashes)
-- artifact_hashes (computed artifact hashes with provenance)
-- environment_fingerprints (captured environment snapshots)
-- seed_configurations (seed state per execution)
-- version_manifests (pinned version manifests)
-- drift_baselines (baseline results for drift detection)
-- replay_sessions (hypertable - replay execution records)
-- reproducibility_audit_log (hypertable - verification audit trail)
```

### 5.3 Prometheus Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_reproducibility_verifications_total` | Counter | Total verification runs by status |
| 2 | `gl_reproducibility_verification_duration_seconds` | Histogram | Verification latency |
| 3 | `gl_reproducibility_hash_computations_total` | Counter | Total hash computations by type |
| 4 | `gl_reproducibility_hash_mismatches_total` | Counter | Hash mismatches detected |
| 5 | `gl_reproducibility_drift_detections_total` | Counter | Drift detections by severity |
| 6 | `gl_reproducibility_drift_percentage` | Gauge | Current drift percentage |
| 7 | `gl_reproducibility_replays_total` | Counter | Replay executions by result |
| 8 | `gl_reproducibility_replay_duration_seconds` | Histogram | Replay execution latency |
| 9 | `gl_reproducibility_non_determinism_sources_total` | Counter | Non-determinism sources detected |
| 10 | `gl_reproducibility_environment_mismatches_total` | Counter | Environment mismatches |
| 11 | `gl_reproducibility_cache_hits_total` | Counter | Hash cache hits |
| 12 | `gl_reproducibility_cache_misses_total` | Counter | Hash cache misses |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/verify` | Run full reproducibility verification |
| POST | `/v1/verify/input` | Verify input hash only |
| POST | `/v1/verify/output` | Verify output hash only |
| GET | `/v1/verifications` | List verification runs (with filters) |
| GET | `/v1/verifications/{verification_id}` | Get verification details |
| POST | `/v1/hash` | Compute deterministic hash for data |
| GET | `/v1/hashes/{artifact_id}` | Get artifact hash history |
| POST | `/v1/drift/detect` | Run drift detection against baseline |
| GET | `/v1/drift/baselines` | List drift baselines |
| POST | `/v1/drift/baselines` | Create/update drift baseline |
| GET | `/v1/drift/baselines/{baseline_id}` | Get specific baseline |
| POST | `/v1/replay` | Execute replay verification |
| GET | `/v1/replays/{replay_id}` | Get replay session details |
| GET | `/v1/environment` | Capture current environment fingerprint |
| GET | `/v1/environment/{fingerprint_id}` | Get stored fingerprint |
| POST | `/v1/versions/pin` | Pin component versions |
| GET | `/v1/versions/manifest/{manifest_id}` | Get version manifest |
| POST | `/v1/report` | Generate reproducibility report |
| GET | `/v1/statistics` | Get verification statistics |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Deterministic hashing**: SHA-256 with float normalization (Decimal precision, sorted keys)
2. **Tolerance-aware**: Configurable absolute (1e-9) and relative (1e-6) tolerances
3. **Environment-aware**: Full platform fingerprinting for cross-environment verification
4. **Seed-controlled**: Global/numpy/torch/custom seed management for complete reproducibility
5. **Version-pinned**: Agent, model, factor, and data version manifests
6. **Drift-sensitive**: Soft (1%) and hard (5%) thresholds with field-level analysis
7. **Replay-capable**: Full environment+seed+version replay for debugging and audit
8. **Non-determinism tracking**: 10 known sources identified and flagged
9. **Complete audit trail**: Every verification logged with SHA-256 provenance chain
10. **Zero-hallucination**: All verification uses deterministic comparisons, no probabilistic methods

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/reproducibility/__init__.py` - Public API exports
2. Create `greenlang/reproducibility/config.py` - ReproducibilityConfig with GL_REPRODUCIBILITY_ env prefix
3. Create `greenlang/reproducibility/models.py` - Pydantic v2 models
4. Create `greenlang/reproducibility/artifact_hasher.py` - ArtifactHasher with SHA-256 deterministic hashing
5. Create `greenlang/reproducibility/determinism_verifier.py` - DeterminismVerifier engine
6. Create `greenlang/reproducibility/drift_detector.py` - DriftDetector with baseline comparison
7. Create `greenlang/reproducibility/replay_engine.py` - ReplayEngine for full replay
8. Create `greenlang/reproducibility/environment_capture.py` - EnvironmentCapture fingerprinting
9. Create `greenlang/reproducibility/seed_manager.py` - SeedManager for all seed types
10. Create `greenlang/reproducibility/version_pinner.py` - VersionPinner with manifest management
11. Create `greenlang/reproducibility/provenance.py` - ProvenanceTracker
12. Create `greenlang/reproducibility/metrics.py` - 12 Prometheus metrics
13. Create `greenlang/reproducibility/api/router.py` - FastAPI router with 20 endpoints
14. Create `greenlang/reproducibility/setup.py` - ReproducibilityService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V028__reproducibility_service.sql`
2. Create K8s manifests in `deployment/kubernetes/reproducibility-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-14. Create unit, integration, and load tests (500+ tests target)

## 7. Success Criteria
- Integration module provides clean SDK for all reproducibility operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V028 database migration for persistent verification storage
- 20 REST API endpoints operational
- 500+ tests passing
- Deterministic hashing with SHA-256 and float normalization
- Drift detection with configurable soft/hard thresholds
- Replay mode for full environment+seed+version replay
- Non-determinism source tracking (10 known sources)
- Complete audit trail for every verification

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-001 Orchestrator**: Calls reproducibility verification before/after DAG execution
- **AGENT-FOUND-003 Normalizer**: Hash verification for unit conversions
- **AGENT-FOUND-004 Assumptions**: Version pinning for assumption data
- **AGENT-FOUND-005 Citations**: Hash verification for evidence packaging
- **AGENT-FOUND-006 Access Guard**: Authorization for verification operations
- **AGENT-FOUND-007 Agent Registry**: Registry of reproducibility agent versions

### 8.2 Downstream Consumers
- **All calculation agents**: Must verify determinism before/after execution
- **Orchestrator DAG execution**: Automatic verification hooks
- **Compliance reporting**: Reproducibility certificates for regulators
- **Audit workflows**: Replay capability for auditor verification

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent verification records and audit storage (V028 migration)
- **Redis**: Hash caching, verification result caching
- **Prometheus**: 12 observability metrics
- **Grafana**: Reproducibility dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
- **Determinism Utilities**: `greenlang.utilities.determinism.clock`, `random`, `uuid`
