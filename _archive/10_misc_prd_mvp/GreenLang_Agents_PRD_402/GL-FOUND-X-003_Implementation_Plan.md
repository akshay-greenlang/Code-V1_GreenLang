# GL-FOUND-X-003 Implementation Plan
## GreenLang Unit & Reference Normalizer

**Version:** 1.0
**Created:** 2026-01-30
**Timeline:** 9-12 months to GA
**Based on:** Interview Summary & PRD Decisions

---

## Executive Summary

This document provides a comprehensive step-by-step implementation guide for building the GL-FOUND-X-003 Unit & Reference Normalizer agent. The implementation is organized into 6 phases across 9-12 months, with clear milestones, deliverables, and dependencies.

---

## Phase 0: Foundation Setup (Weeks 1-4)

### 0.1 Project Structure

```
greenlang-normalizer/
├── packages/
│   ├── gl-normalizer-core/           # Embedded library (pure, deterministic)
│   │   ├── src/
│   │   │   ├── parser/               # Unit parser (Pint + extensions)
│   │   │   ├── converter/            # Conversion engine
│   │   │   ├── resolver/             # Entity resolution
│   │   │   ├── dimension/            # Dimensional analysis
│   │   │   ├── policy/               # Policy engine
│   │   │   └── audit/                # Audit event generation
│   │   ├── registry/                 # Unit & dimension registry
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   ├── gl-normalizer-service/        # API service
│   │   ├── src/
│   │   │   ├── api/                  # FastAPI routes
│   │   │   ├── vocab/                # Vocabulary management
│   │   │   ├── audit/                # Audit persistence (outbox + Kafka)
│   │   │   ├── jobs/                 # Async job processing
│   │   │   └── admin/                # Admin endpoints
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   ├── gl-normalizer-sdk/            # Python SDK
│   │   ├── src/
│   │   │   ├── client.py
│   │   │   ├── vocab_provider.py
│   │   │   └── cache.py
│   │   └── pyproject.toml
│   │
│   └── gl-normalizer-cli/            # CLI tool
│       ├── src/
│       │   ├── commands/
│       │   └── main.py
│       └── pyproject.toml
│
├── vocab/                            # Vocabulary repository
│   ├── fuels/
│   ├── materials/
│   ├── processes/
│   ├── units/
│   └── schemas/
│
├── review-console/                   # Review Console UI
│   ├── web/                          # React web app
│   └── api/                          # Backend for UI
│
├── config/
│   ├── canonical_units.yaml          # GL Canonical Units v1
│   ├── compliance_profiles/          # GHG Protocol, EU CSRD, etc.
│   ├── confidence_thresholds.yaml
│   └── policy_defaults.yaml
│
├── tests/
│   ├── golden/                       # Golden test suite
│   ├── property/                     # Property-based tests
│   └── cross_validation/             # Pint cross-validation
│
├── docs/
│   ├── api/
│   ├── developer/
│   ├── governance/
│   └── compliance/
│
└── infrastructure/
    ├── terraform/
    ├── kubernetes/
    └── docker/
```

### 0.2 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Core Library | Python 3.11+ | GreenLang standard, Pint compatibility |
| Unit Parser | Pint + custom extensions | Interview decision |
| API Service | FastAPI | High performance, OpenAPI support |
| Database | PostgreSQL | Vocab store, review workflow state |
| Audit Stream | Kafka | Interview decision: event-sourced audit |
| Audit Cold Storage | S3/GCS (Parquet) | Long-term retention |
| Cache | Redis | Vocab snapshot caching |
| Web UI | React + TypeScript | Modern, accessible |
| CLI | Click/Typer | Python CLI framework |

### 0.3 Initial Setup Tasks

- [ ] Initialize monorepo structure
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Configure linting, formatting, type checking
- [ ] Set up pre-commit hooks
- [ ] Create development environment (Docker Compose)
- [ ] Set up test infrastructure
- [ ] Create initial documentation structure

---

## Phase 1: Core Library - Unit Parsing & Conversion (Weeks 5-12)

### 1.1 GL Canonical Units Registry

**File:** `config/canonical_units.yaml`

```yaml
# GL Canonical Units v1 (Interview Decision)
version: "1.0.0"
canonical_units:
  energy:
    canonical: MJ
    common_inputs: [kWh, MWh, GJ, MMBtu, therm, kcal, BTU]
    conversion_exact:
      kWh: 3.6  # 1 kWh = 3.6 MJ (exact)

  mass:
    canonical: kg
    common_inputs: [g, t, lb, oz, tonne]

  emissions_mass:
    canonical: kgCO2e
    common_inputs: [tCO2e, lbCO2e, gCO2e]
    requires_metadata: [gwp_profile_id]

  volume:
    canonical: m3
    common_inputs: [L, bbl, gal_us, gal_uk, ft3]

  volume_standard:
    canonical: Nm3
    common_inputs: [scf, Sm3]
    requires_metadata: [reference_conditions]

  pressure:
    canonical: kPa_abs
    common_inputs: [bar, bara, barg, psi, psia, psig, Pa, MPa]

  temperature:
    canonical: degC
    common_inputs: [degF, K]
    conversion_type: affine

  temperature_delta:
    canonical: K
    common_inputs: [delta_degC, delta_degF]

  power:
    canonical: kW
    common_inputs: [W, MW, hp]

  time:
    canonical: s
    common_inputs: [h, min, day]

  mass_flow:
    canonical: kg_per_s
    common_inputs: [kg/h, t/h, lb/h]

  volume_flow:
    canonical: m3_per_s
    common_inputs: [m3/h, L/min, gpm]

  dimensionless:
    canonical: "1"
    common_inputs: ["%", "ppm", "ppb"]
```

### 1.2 Unit Parser Implementation

**Tasks:**

1. **Pint Integration Layer** (`gl-normalizer-core/src/parser/pint_wrapper.py`)
   - Initialize Pint UnitRegistry with GreenLang extensions
   - Add custom units (Nm3, CO2e, emissions-specific)
   - Configure SI prefixes support
   - Implement parse error handling with suggestions

2. **Unit String Preprocessor** (`gl-normalizer-core/src/parser/preprocessor.py`)
   - Unicode normalization (NFC)
   - Whitespace/casing normalization
   - Separator canonicalization (per, /, ·, *)
   - Synonym expansion (lbs→lb, litre→L, kilograms→kg)
   - Exponent notation handling (m2, m^2, m²)

3. **Unit AST Generator** (`gl-normalizer-core/src/parser/ast.py`)
   - Parse to structured AST: numerator_terms, denominator_terms, exponents, prefixes
   - Compute dimension signature
   - Validate against unit registry

4. **Locale Handler** (`gl-normalizer-core/src/parser/locale.py`)
   - Locale profile resolution (request → dataset → org)
   - Ambiguous unit resolution (gallon, ton, pint)
   - Numeric string parsing (decimal/thousands separators)

### 1.3 Conversion Engine

**Tasks:**

1. **Scalar Converter** (`gl-normalizer-core/src/converter/scalar.py`)
   - Linear conversions with factor lookup
   - Affine conversions (temperature)
   - Multi-step conversion tracing
   - Precision rules per dimension

2. **Basis Converter** (`gl-normalizer-core/src/converter/basis.py`)
   - Reference conditions handler (T, P for Nm3/scf)
   - Pressure mode inference (barg/psig vs bara/psia)
   - Gauge-to-absolute conversion with atmospheric reference

3. **Compound Unit Converter** (`gl-normalizer-core/src/converter/compound.py`)
   - Parse compound units (kg/m3, MJ/kg)
   - Component-wise conversion
   - Dimension compatibility check

4. **GWP Handler** (`gl-normalizer-core/src/converter/gwp.py`)
   - GWP profile resolution (request → org → fail)
   - Version tracking (AR5, AR6)
   - No-mixing enforcement within run

5. **Energy Basis Handler** (`gl-normalizer-core/src/converter/energy_basis.py`)
   - LHV/HHV explicit requirement (STRICT)
   - Org default fallback (LENIENT)
   - Audit metadata generation

### 1.4 Dimensional Analysis

**Tasks:**

1. **Dimension Registry** (`gl-normalizer-core/src/dimension/registry.py`)
   - Dimension signature definitions
   - Domain-specific dimensions (emissions_mass, volume_flow)
   - Versioned registry with hash

2. **Dimension Validator** (`gl-normalizer-core/src/dimension/validator.py`)
   - Expected vs computed dimension comparison
   - STRICT/LENIENT mode behavior
   - Error generation (GLNORM-E200)

---

## Phase 2: Entity Resolution & Vocabulary (Weeks 13-20)

### 2.1 Vocabulary Data Model

**Schema:** `vocab/schemas/vocabulary_entry.yaml`

```yaml
VocabularyEntry:
  type: object
  required: [reference_id, canonical_name, entity_type, status]
  properties:
    reference_id:
      type: string
      pattern: "^GL-(FUEL|MAT|PROC)-[A-Z0-9_]+$"
    canonical_name:
      type: string
    entity_type:
      enum: [fuel, material, process]
    status:
      enum: [active, deprecated]
    replaced_by:
      type: string
      nullable: true
    aliases:
      type: array
      items:
        type: object
        properties:
          alias: { type: string }
          locale: { type: string, nullable: true }
          source: { type: string }  # vendor, iso, common
          priority: { type: integer, default: 100 }
    attributes:
      type: object
      additionalProperties: true  # LHV, HHV, grade, etc.
    effective_from:
      type: string
      format: date
    effective_to:
      type: string
      format: date
      nullable: true
```

### 2.2 Entity Resolver

**Tasks:**

1. **Resolution Pipeline** (`gl-normalizer-core/src/resolver/pipeline.py`)
   - Priority order implementation:
     1. Exact Reference ID match
     2. Exact canonical name match
     3. Exact alias match
     4. Rule-based normalization match
     5. Fuzzy match candidate retrieval
     6. LLM candidate suggestion (feature-flagged)

2. **Deterministic Matchers** (`gl-normalizer-core/src/resolver/deterministic.py`)
   - ID matcher
   - Name matcher (case-insensitive, normalized)
   - Alias matcher (type-scoped)
   - Rule matcher (tokenization, punctuation, abbreviation)

3. **Fuzzy Matcher** (`gl-normalizer-core/src/resolver/fuzzy.py`)
   - Token overlap score
   - Normalized edit distance
   - Domain hint boosting (sector, region)
   - Configurable threshold per entity type:
     - Fuel: ≥0.95
     - Material: ≥0.90
     - Process: ≥0.85

4. **Confidence Calculator** (`gl-normalizer-core/src/resolver/confidence.py`)
   - Score calculation
   - Margin rule (top1 - top2 < 0.07 → needs_review)
   - Constraint validation (type, domain)

5. **LLM Candidate Generator** (`gl-normalizer-core/src/resolver/llm.py`)
   - Feature flag control (OFF by default)
   - Sandbox isolation
   - Data minimization (only entity name + type + hints)
   - Always `needs_review=true`

### 2.3 Vocabulary Manager

**Tasks:**

1. **Snapshot Loader** (`gl-normalizer-core/src/vocab/loader.py`)
   - Version-pinned loading
   - Signature verification (ed25519)
   - Hash validation
   - Cache integration

2. **Deprecation Handler** (`gl-normalizer-core/src/vocab/deprecation.py`)
   - `replaced_by` resolution
   - Warning generation (GLNORM-E402)
   - Audit trail preservation

3. **Alias Collision Detector** (`gl-normalizer-core/src/vocab/collision.py`)
   - Type-scoped uniqueness check
   - CI integration for publish-time validation
   - Runtime defense (GLNORM-E406)

---

## Phase 3: Audit & Policy System (Weeks 21-28)

### 3.1 Audit Event Generation

**Tasks:**

1. **Event Schema** (`gl-normalizer-core/src/audit/schema.py`)
   ```python
   @dataclass
   class NormalizationEvent:
       event_id: str
       event_ts: datetime
       prev_event_hash: Optional[str]
       request_id: str
       source_record_id: str
       org_id: str
       policy_mode: PolicyMode
       status: EventStatus

       # Version tracking (determinism)
       vocab_version: str
       policy_version: str
       unit_registry_version: str
       validator_version: str
       api_revision: str

       # Payload
       measurements: List[MeasurementAudit]
       entities: List[EntityAudit]

       # Integrity
       payload_hash: str
       event_hash: str
   ```

2. **Hash Chain Generator** (`gl-normalizer-core/src/audit/chain.py`)
   - Per-run hash chaining
   - `prev_event_hash` linking
   - Tamper-evident verification

3. **Audit Payload Builder** (`gl-normalizer-core/src/audit/builder.py`)
   - Measurement audit (raw → canonical with steps)
   - Entity audit (resolution trace)
   - Complete version metadata

### 3.2 Audit Persistence (Service)

**Tasks:**

1. **Durable Outbox** (`gl-normalizer-service/src/audit/outbox.py`)
   - PostgreSQL outbox table
   - Transactional write with normalization
   - Delivery status tracking

2. **Kafka Publisher** (`gl-normalizer-service/src/audit/publisher.py`)
   - Async background worker
   - Exponential backoff retry
   - Idempotency (event_id dedup)
   - Partition by org_id + run_id

3. **Cold Storage Sink** (`gl-normalizer-service/src/audit/sink.py`)
   - Kafka Connect / Flink sink config
   - S3/GCS Parquet output
   - Partition path: `org_id/yyyy/mm/dd/event_type`

4. **Backpressure Controller** (`gl-normalizer-service/src/audit/backpressure.py`)
   - Max outbox rows check
   - Max publish lag check
   - Policy-based failure (STRICT vs LENIENT)

### 3.3 Policy Engine

**Tasks:**

1. **Policy Configuration** (`gl-normalizer-core/src/policy/config.py`)
   ```yaml
   policy:
     mode: STRICT  # STRICT | LENIENT

     normalization_on_invalid:
       mode: STRICT
       lenient:
         normalize_only_valid_fields: true
         emit_review_suggestions_for_invalid_refs: true

     batch_policy:
       default_mode: PARTIAL  # PARTIAL | FAIL_FAST | THRESHOLD
       max_error_rate: 0.10

     audit_delivery:
       mode: STRICT
       commit_requirement: OUTBOX
       backpressure:
         max_outbox_rows: 5000000
         max_publish_lag_seconds: 3600

     locale_policy:
       resolution_precedence: [request, dataset, org]
       require_locale_for_ambiguous_units: true

     energy_basis:
       require_explicit_in_strict: true
       lenient_default: LHV

     pressure_mode:
       infer_from_suffix: true
       require_explicit_for_ambiguous: true
   ```

2. **Compliance Profiles** (`gl-normalizer-core/src/policy/compliance.py`)
   - Profile loader
   - Metadata requirements per profile
   - Validation rules

   Profiles:
   - `GHP.CORP.v1` (GHG Protocol Corporate)
   - `EU.ESRS_E1.v1` (EU CSRD)
   - `IFRS.S2.v1` (ISSB)
   - `IN.BRSR.v1` (India)
   - `CA.SB253.v1` (California)
   - `US.SEC.CLIMATE.v1` (volatile)

---

## Phase 4: API Service & SDK (Weeks 29-36)

### 4.1 API Endpoints

**Tasks:**

1. **Normalize API** (`gl-normalizer-service/src/api/normalize.py`)
   - `POST /v1/normalize` (sync, ≤10K records)
   - Request validation
   - Version locking
   - Response with all version fields

2. **Batch/Job API** (`gl-normalizer-service/src/api/jobs.py`)
   - `POST /v1/normalize/jobs` (async, 100K+)
   - `GET /v1/normalize/jobs/{job_id}`
   - `GET /v1/normalize/jobs/{job_id}/results`
   - `GET /v1/normalize/jobs/{job_id}/errors`
   - `POST /v1/normalize/jobs/{job_id}:cancel`

3. **Entity Resolution API** (`gl-normalizer-service/src/api/resolve.py`)
   - `POST /v1/resolve`
   - Candidate retrieval with scores

4. **Vocabulary API** (`gl-normalizer-service/src/api/vocab.py`)
   - `GET /v1/vocab/snapshots`
   - `GET /v1/vocab/snapshots/{version}`
   - Signature in response

5. **Admin API** (`gl-normalizer-service/src/api/admin.py`)
   - `POST /admin/vocabulary/reload`
   - `POST /admin/cache/invalidate`
   - `POST /admin/vocabulary/rollback`
   - `POST /admin/rate-limits/override`

6. **Health API** (`gl-normalizer-service/src/api/health.py`)
   - `GET /health`
   - `GET /ready`
   - `GET /metrics`

### 4.2 Python SDK

**Tasks:**

1. **Client** (`gl-normalizer-sdk/src/client.py`)
   ```python
   class NormalizerClient:
       def __init__(self, vocab_provider: VocabProvider):
           ...

       def normalize(
           self,
           records: List[NormalizationRequest],
           vocab_version: Optional[str] = None,
           policy_mode: PolicyMode = PolicyMode.STRICT,
       ) -> NormalizationResult:
           ...

       def normalize_batch(
           self,
           records: List[NormalizationRequest],
           batch_policy: BatchPolicy = BatchPolicy.PARTIAL,
       ) -> BatchResult:
           ...

       def resolve_entity(
           self,
           raw_name: str,
           entity_type: EntityType,
           hints: Optional[ResolutionHints] = None,
       ) -> ResolutionResult:
           ...
   ```

2. **Vocab Provider** (`gl-normalizer-sdk/src/vocab_provider.py`)
   - `CachedRemoteVocabProvider` (auto-fetch + TTL cache)
   - `FileVocabProvider` (offline mode)
   - Signature verification
   - ETag-based refresh

3. **Cache Manager** (`gl-normalizer-sdk/src/cache.py`)
   - Local cache directory management
   - TTL enforcement
   - Stale-while-revalidate support

### 4.3 CLI Tool

**Tasks:**

1. **Normalize Command** (`gl-normalizer-cli/src/commands/normalize.py`)
   ```bash
   greenlang normalize \
     --input <file> \
     --output <file> \
     --policy <STRICT|LENIENT> \
     --vocab-version <version> \
     --format <json|yaml|parquet>
   ```

2. **Resolve Command** (`gl-normalizer-cli/src/commands/resolve.py`)
   ```bash
   greenlang resolve-entity \
     --name <name> \
     --type <fuel|material|process> \
     --return-candidates
   ```

3. **Review Commands** (`gl-normalizer-cli/src/commands/review.py`)
   ```bash
   greenlang review list [--filters]
   greenlang review show <id>
   greenlang review resolve <id> --entity <GL_ID>
   greenlang review bulk-resolve --match <pattern> --entity <GL_ID>
   greenlang review export --format <csv|jsonl>
   greenlang review propose-alias <id>
   ```

4. **Vocab Commands** (`gl-normalizer-cli/src/commands/vocab.py`)
   ```bash
   greenlang vocab list
   greenlang vocab diff --v1 <ver1> --v2 <ver2>
   greenlang vocab validate <file>
   ```

---

## Phase 5: Review Console & Integration (Weeks 37-44)

### 5.1 Review Console Backend

**Tasks:**

1. **Review Items API** (`review-console/api/src/routes/review.py`)
   - `POST /review-items` (ingest from runs)
   - `GET /review-items` (with filters)
   - `GET /review-items/{id}`
   - `POST /review-items/{id}/resolve`
   - `POST /review-items/{id}/propose-alias`
   - `POST /review-items/{id}/propose-entity`

2. **Resolution Actions** (`review-console/api/src/actions/`)
   - Org-local override creation
   - Git PR generation for global vocab
   - Audit event emission

3. **Workflow State** (`review-console/api/src/models/`)
   - Review item model (PostgreSQL)
   - Resolution history
   - Assignment/ownership

### 5.2 Review Console Web UI

**Tasks:**

1. **Queue View** (`review-console/web/src/pages/Queue.tsx`)
   - Filterable list (org, entity_type, confidence, source)
   - Pagination
   - Bulk selection

2. **Item Detail** (`review-console/web/src/pages/ItemDetail.tsx`)
   - Raw value + normalized tokens
   - Candidates with scores
   - "Why flagged" explanation
   - Resolution trace

3. **Resolution Actions** (`review-console/web/src/components/ResolutionPanel.tsx`)
   - Approve candidate
   - Pick different entity
   - Create new entity proposal
   - Mark unresolved
   - Bulk actions

4. **Audit Trail** (`review-console/web/src/components/AuditTrail.tsx`)
   - Per-item resolution history
   - Who, when, what

### 5.3 X-002 Integration

**Tasks:**

1. **Validator Contract** (`gl-normalizer-core/src/integration/validator.py`)
   - Field-level validity map consumption
   - STRICT/LENIENT behavior implementation

2. **Schema Context** (`gl-normalizer-core/src/integration/schema.py`)
   - Expected dimension extraction
   - Required metadata constraints

---

## Phase 6: Testing, Docs & Production (Weeks 45-52)

### 6.1 Testing

**Tasks:**

1. **Golden Test Suite** (`tests/golden/`)
   - Core conversions (energy, mass, pressure, temperature)
   - Edge cases (negative, zero, huge/small magnitudes)
   - Offset units (temperature)
   - Entity resolution (exact, alias, fuzzy)
   - SME verification workflow

2. **Property-Based Tests** (`tests/property/`)
   - Round-trip invariant
   - Idempotence: `normalize(normalize(x)) == normalize(x)`
   - Monotonicity (linear conversions)
   - Dimensional safety

3. **Cross-Validation** (`tests/cross_validation/`)
   - Pint comparison for all supported units
   - Independent factor table validation
   - Custom unit verification

4. **Integration Tests** (`tests/integration/`)
   - API endpoint tests
   - Batch/job workflow tests
   - Audit delivery tests
   - X-002 integration tests

5. **Performance Tests** (`tests/performance/`)
   - Latency benchmarks (p50, p95, p99)
   - Throughput tests (records/sec)
   - Memory profiling
   - Cache hit rate verification

### 6.2 Documentation

**Tasks:**

1. **API Reference** (`docs/api/`)
   - OpenAPI spec
   - Endpoint documentation
   - Error codes reference

2. **Developer Guide** (`docs/developer/`)
   - Quickstart
   - SDK usage
   - CLI reference
   - Migration guide

3. **Governance Guide** (`docs/governance/`)
   - Vocabulary management
   - Review workflow
   - Version pinning best practices

4. **Compliance Mapping** (`docs/compliance/`)
   - GHG Protocol alignment
   - EU CSRD/ESRS mapping
   - Audit requirements

5. **Operational Runbook** (`docs/operations/`)
   - Failure modes
   - On-call actions
   - DR procedures

### 6.3 Infrastructure & Deployment

**Tasks:**

1. **Docker Images**
   - `gl-normalizer-service`
   - `gl-normalizer-worker` (job processor)
   - `review-console`

2. **Kubernetes Manifests**
   - Deployment configs
   - Service definitions
   - Ingress rules
   - HPA configs

3. **Terraform**
   - Kafka cluster
   - PostgreSQL (RDS)
   - S3/GCS buckets
   - Redis cluster
   - KMS keys

4. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - AlertManager rules

---

## Milestone Summary

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| M0 | 4 | Project setup, CI/CD, dev environment |
| M1 | 12 | Unit parser + conversion engine + golden tests |
| M2 | 20 | Entity resolution + vocabulary management |
| M3 | 28 | Audit system + policy engine + compliance profiles |
| M4 | 36 | API service + SDK + CLI |
| M5 | 44 | Review Console + X-002 integration |
| M6 | 52 | Production deployment + GA |

---

## Risk Mitigations (from Interview)

### 1. Determinism & Replayability
- Hard version pinning in ALL outputs
- Snapshot locking for async jobs
- Golden corpus tests across releases
- Numeric determinism policy (explicit rounding rules)

### 2. Vocabulary Governance Bottlenecks
- Git as source of truth + thin UI
- Two-tier mappings (org-local + global)
- CI collision enforcement
- Review Console to clear needs_review queue

### 3. Integration Complexity
- Explicit X-002 contract (field-level validity)
- Policy-driven behavior (STRICT/LENIENT)
- Two-lane API (sync + async)
- SDK with auto-fetch + cache

---

## Additional Requirements (from Interview)

### Performance Benchmarks
- Single-record P95: < 50ms
- Batch (100 records) P95: < 500ms
- Throughput: > 10,000 records/sec/node
- Vocab cache hit rate: > 90%

### Disaster Recovery
- Vocab snapshots: versioned, signed, replicated
- Audit outbox: PostgreSQL with PITR
- Kafka: multi-AZ replication
- Cold storage: cross-region replication

### Internationalization
- Error messages: i18n-ready (en, de, ja, zh)
- Unit names: locale-aware aliases in registry
- Review Console: multi-language support

---

## Next Steps

1. **Approve this implementation plan**
2. **Create detailed task breakdown in project tracker**
3. **Begin Phase 0: Foundation Setup**
4. **Weekly progress reviews against milestones**
