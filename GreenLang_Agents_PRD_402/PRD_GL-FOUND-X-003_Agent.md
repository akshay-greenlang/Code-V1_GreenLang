# PRD: GreenLang Unit & Reference Normalizer (GL-FOUND-X-003)

**Agent family:** NormalizationFamily
**Layer:** Foundation & Governance
**Primary domains:** Unit normalization, entity resolution, reference data governance (cross-cutting)
**Priority:** P0 (cross-cutting; blocks downstream calculations)
**Doc version:** 1.0
**Last updated:** 2026-01-29 (Asia/Kolkata)
**Owner:** (TBD)
**Reviewers:** Data Engineering, ESG Analytics, Governance/Audit, Domain SMEs, Platform Eng

---

## 1. Executive Summary

GreenLang needs a **deterministic, audit-ready normalization authority** for units, conversions, and reference data across sectors/domains. The **Unit & Reference Normalizer (GL-FOUND-X-003)** is a foundation agent that:

1. **Normalizes unit strings** into a canonical representation.
2. **Converts measurements** into GreenLang canonical units (per measurement dimension).
3. **Standardizes naming** and resolves raw names/codes for **fuels, processes, and materials** to stable **reference IDs** using **controlled vocabularies**.
4. Produces a **conversion audit log** that provides **full lineage** from raw input → canonical output.

This agent is part of the **NormalizationFamily** and is considered **critical infrastructure**: downstream calculations must be able to trust its output without re-validating units or entity naming.

---

## 2. Context and Problem Statement

Green/sustainability datasets frequently arrive with:

- Messy unit strings (`"KG"`, `"k.g"`, `"kg "`, `"kilograms"`)
- Mixed units within a dataset (`kWh`, `GJ`, `MMBtu` in the same column)
- Domain-specific units and bases (`Nm3`, `scf`, `bbl`, `therm`)
- Non-standard emissions expressions (`tCO2`, `kg CO2e`, `lbs CO2 eq`)
- Inconsistent entity naming and identifiers:
  - Fuel names: `"Nat Gas"`, `"Natural Gas"`, `"天然ガス"`, `"Natural-gas"`
  - Process names: `"EAF"`, `"electric arc furnace"`
  - Material names: `"Portland cement"`, `"OPC"`, `"CEM I"`

Without standardization:

- Calculations become inconsistent and hard to reproduce.
- Governance teams cannot trace how numbers were derived.
- Reference drift creates “multiple truths” across pipelines and products.

---

## 3. Goals, Non-Goals, and Success Metrics

### 3.1 Goals
- **G1 - Canonical measurements:** Convert supported measurements into canonical units with correct dimensionality checks.
- **G2 - Stable reference IDs:** Resolve fuels/materials/processes to controlled vocabulary IDs with clear confidence + method.
- **G3 - Governance-grade audit:** Emit an immutable audit log with enough detail to replay normalization.
- **G4 - Deterministic behavior:** Same input + same versions → same output.
- **G5 - Extensibility:** Add new units/entities via versioned reference tables without code changes (where possible).

### 3.2 Non-Goals (v1)
- Arbitrary “best guess” conversions when dimensions are unknown or metadata is missing.
- Master Data Management (MDM) UI (agent provides APIs + logs; UI is separate product).
- Automatic creation of new reference IDs without governance approval (agent may propose candidates only).
- Emissions factor selection and application (handled by separate factor/enrichment agents).

### 3.3 Success Metrics (KPIs)
- **Unit parse success rate:** ≥ 99.9% for top units in production within 30 days of rollout.
- **Canonical conversion correctness:** 100% for supported conversions (within numeric tolerance).
- **Reference resolution coverage:** ≥ 95% of raw fuel/material/process values resolved deterministically (exact/alias/rule) in initial target datasets.
- **Ambiguity rate:** ≤ 2% records flagged `needs_review` after vocabulary stabilization.
- **Audit completeness:** 100% normalized outputs have associated audit events.

---

## 4. Users and Key Use Cases

### 4.1 Personas
- **Pipeline Engineer:** wants consistent units before calculations; wants failures to be explicit and debuggable.
- **ESG Analyst:** wants consistent naming and ID mapping to compare across facilities and time.
- **Auditor / Governance Lead:** needs a replayable audit log with conversion steps and reference versions.
- **Domain SME:** reviews low-confidence mappings and curates vocabulary changes.

### 4.2 Core Use Cases
1. Normalize incoming meter readings (`kWh`, `MWh`) to canonical energy unit (`MJ` or `kWh` depending on canonical policy).
2. Normalize fuel names and vendor codes to stable fuel IDs.
3. Convert emissions quantities to canonical `kgCO2e` while preserving GWP metadata.
4. Batch-normalize historical datasets and reproduce results after months via pinned vocabulary versions.
5. Detect and block dimension mismatches (e.g., `m3` value provided where schema expects `kg`).

---

## 5. Scope

### 5.1 In Scope
- Unit parsing and canonicalization
- Dimensional analysis and validation
- Conversion to canonical units (including compound units)
- Entity resolution for fuels/materials/processes
- Controlled vocabulary consumption (read-only) + version pinning
- Audit log emission and persistence
- Policy modes (`STRICT`, `LENIENT`) and structured error codes

### 5.2 Out of Scope
- Currency conversion / inflation
- Geospatial normalization
- OCR / extraction from documents
- Emissions factor selection and application
- Human workflow tooling (ticketing/MDM UI) beyond emitting review artifacts

### 5.3 Definitions and Glossary

| Term | Definition |
|------|------------|
| **Canonical Unit** | The schema-specified standard unit for a dimension (e.g., kWh for energy, kg for mass). |
| **Dimension Signature** | Unique identifier for a physical quantity type (e.g., `energy`, `mass`, `volume_flow`). |
| **Reference ID** | Stable, immutable identifier for a controlled vocabulary entity (e.g., `GL-FUEL-NATGAS`). |
| **Controlled Vocabulary** | Versioned registry of allowed entities (fuels, materials, processes) with aliases. |
| **Entity Resolution** | Process of mapping raw input strings/codes to controlled vocabulary Reference IDs. |
| **Match Method** | Resolution technique used: `exact`, `alias`, `rule`, `fuzzy`, or `llm_candidate`. |
| **Confidence Score** | Numeric value (0.0-1.0) indicating certainty of entity resolution match. |
| **Policy Mode** | Validation strictness level: `STRICT` (fail fast) or `LENIENT` (warn and continue). |
| **Vocabulary Version** | Semantic version of the controlled vocabulary used for resolution (e.g., `2026.01.0`). |
| **Deprecation Workflow** | Process for marking vocabulary entities as deprecated with `replaced_by` references. |
| **Unit AST** | Abstract Syntax Tree representing parsed unit structure (numerator, denominator, prefixes). |
| **Normalization Event** | Immutable audit record capturing all transformations applied to a single input. |
| **Audit Payload** | Detailed JSON structure within an audit event containing conversion steps and metadata. |
| **Fuzzy Matching** | Non-deterministic resolution using token overlap, edit distance, and similarity scoring. |
| **LLM Candidate** | Entity match suggested by language model; requires human review before acceptance. |
| **Reference Conditions** | Standard temperature/pressure values for volume-based unit conversions (e.g., Nm³). |
| **GWP Version** | Global Warming Potential assessment report version (AR5, AR6) affecting CO2e conversions. |

---

## 6. Inputs, Outputs, and Contracts

### 6.1 Required Inputs
- **Raw measurement(s)**: numeric value(s)
- **Raw unit string**: may be messy
- **Schema context** (from Schema Validator):
  - expected dimension (e.g., `energy`, `mass`, `volume_flow`)
  - constraints (e.g., required metadata like reference conditions)
- **Entity inputs** (optional per record):
  - raw name, alias, vendor code, taxonomy hints
- **Reference tables**:
  - unit metadata registry (if needed)
  - controlled vocabulary (fuels/materials/processes) + aliases

### 6.2 Outputs
- Canonical measurement objects (value + canonical unit + dimension)
- Normalized entity objects (reference_id + canonical_name + match_method + confidence)
- Conversion audit event ID + audit payload (or pointer)

### 6.3 Input Schema (conceptual)
```json
{
  "source_record_id": "string",
  "policy_mode": "STRICT|LENIENT",
  "measurements": [
    {
      "field": "string",
      "value": 0.0,
      "unit": "string",
      "expected_dimension": "string",
      "metadata": {
        "locale": "string",
        "reference_conditions": {
          "temperature_C": 0,
          "pressure_kPa": 101.325
        },
        "gwp_version": "AR5|AR6",
        "notes": "string"
      }
    }
  ],
  "entities": [
    {
      "field": "string",
      "entity_type": "fuel|material|process",
      "raw_name": "string",
      "raw_code": "string",
      "hints": {
        "region": "string",
        "sector": "string"
      }
    }
  ]
}
```

### 6.4 Output Schema (conceptual)
```json
{
  "source_record_id": "string",
  "canonical_measurements": [
    {
      "field": "string",
      "dimension": "string",
      "value": 0.0,
      "unit": "string",
      "raw_value": 0.0,
      "raw_unit": "string",
      "precision_applied": {},
      "warnings": []
    }
  ],
  "normalized_entities": [
    {
      "field": "string",
      "entity_type": "fuel|material|process",
      "raw_name": "string",
      "reference_id": "string",
      "canonical_name": "string",
      "vocabulary_version": "string",
      "match_method": "exact|alias|rule|fuzzy|llm_candidate",
      "confidence": 0.0,
      "needs_review": false,
      "warnings": []
    }
  ],
  "audit": {
    "normalization_event_id": "string",
    "status": "success|warning|failed"
  }
}
```

---

## 7. Canonical Units and Unit System Design

### 7.1 Canonical Unit Policy
GreenLang must define a **canonical unit per dimension**. Canonical policy must be:

- **Explicit** (documented and versioned)
- **Stable** (changes require migration plan)
- **Practical** for ESG reporting and computation

Suggested defaults (final choices are an open question until governance signs off):

| Dimension | Canonical Unit | Common Inputs | Notes |
|---|---|---|---|
| energy | MJ | kWh, MWh, GJ, MMBtu, therm | canonical energy should minimize rounding error and match reporting |
| mass | kg | g, t, lb | |
| volume | m3 | L, bbl, scf | scf requires basis |
| time | h | s, day | choose per schema |
| power | kW | W, MW | |
| emissions | kgCO2e | tCO2e, lbCO2e | requires GWP version |

### 7.2 Supported Unit Features (v1)
- Prefixes (`k`, `M`, `G`, `m`, `µ`)
- Exponents (`m2`, `m^2`)
- Compound units (`kg/m3`, `MJ/kg`, `kgCO2e/kWh`)
- Affine units (temperature conversions)
- Domain tokens (`CO2`, `CO2e`) **with explicit schema dimension** (`emissions_mass`)

### 7.3 Unit Parsing Strategy
A unit parsing subsystem must:
- Normalize strings (trim, collapse whitespace, unicode normalization, casefold)
- Convert separators (“per”, “/”, “·”) to a canonical grammar
- Produce an AST or equivalent structure:
  - numerator terms
  - denominator terms
  - exponents
  - prefixes

The parser must return structured parse errors with suggestions (e.g., “did you mean `kWh`?”).

### 7.4 Dimensional Analysis
- Each unit expression maps to a **dimension signature**.
- Compare with `expected_dimension` from Schema Validator.
- If mismatch → fail or flag depending on policy mode.
- Dimension signatures must be deterministic and versioned alongside unit registry.

---

## 8. Controlled Vocabularies and Reference ID Governance

### 8.1 Vocabulary Entities
Each vocabulary entry must contain:

- `reference_id` (stable)
- `canonical_name`
- `entity_type` (`fuel|material|process`)
- `status` (`active|deprecated`)
- `replaced_by` (if deprecated)
- `aliases[]` (synonyms, vendor terms, localized names)
- optional `attributes` (LHV/HHV basis, grade, composition) - used by downstream agents

### 8.2 Reference ID Principles
- IDs should be **stable** and **non-semantic** where possible to avoid churn when names change.
- IDs must be globally unique within GreenLang.
- IDs must be immutable once published (deprecate + replace; never reuse).

### 8.3 Versioning Model
- Vocabulary releases are versioned (`semver` recommended).
- Every normalization output includes:
  - `vocabulary_version`
  - `vocabulary_hash` (optional)
- Pipelines may pin to a version for reproducibility.

### 8.4 Deprecation Workflow
- Deprecate an entity by marking `status=deprecated` and setting `replaced_by`.
- Normalizer behavior:
  - If input matches deprecated entity: return `replaced_by` but include warning and audit detail.
  - Keep the deprecated ID in audit to support historical replay.

### 8.5 Governance Workflow (v1 minimum)
1. SMEs propose vocabulary changes (new entity, alias, deprecation)
2. Automated checks:
   - duplicate detection
   - alias collisions
   - regression tests
3. Approval + publish new version
4. Rollout plan:
   - update pinned versions gradually
   - monitor changes in ambiguity rate

---

## 9. Functional Requirements

> **Priority legend:** P0 = must-have (MVP/GA critical), P1 = should-have, P2 = nice-to-have.

### 9.1 Unit Parsing & Normalization

#### Intake and Preprocessing
- **FR-001 (P0):** Accept raw unit strings via API/CLI with support for Unicode normalization.
- **FR-002 (P0):** Normalize whitespace, casing, and punctuation before parsing.
- **FR-003 (P0):** Handle common unit synonyms (`"lbs"` → `lb`, `"litre"` → `L`, `"kilograms"` → `kg`).
- **FR-004 (P0):** Support SI prefixes (`k`, `M`, `G`, `m`, `µ`, `n`).
- **FR-005 (P0):** Support exponent notation (`m2`, `m^2`, `m²`).

#### Unit Parsing
- **FR-006 (P0):** Parse unit strings into structured AST with numerator/denominator terms.
- **FR-007 (P0):** Support compound units (`kg/m3`, `MJ/kg`, `kgCO2e/kWh`).
- **FR-008 (P0):** Support separator variations (`per`, `/`, `·`, `*`).
- **FR-009 (P0):** Compute dimension signature from parsed unit.
- **FR-010 (P0):** Emit `GLNORM-E100` (UNIT_PARSE_FAILED) with suggestions on parse failure.
- **FR-011 (P1):** Provide "did you mean?" suggestions for close-match units.
- **FR-012 (P1):** Support locale-aware parsing (decimal separators, unit names).

### 9.2 Dimensional Validation

- **FR-020 (P0):** Compare computed dimension signature with schema's `expected_dimension`.
- **FR-021 (P0):** Emit `GLNORM-E200` (DIMENSION_MISMATCH) with expected vs actual on mismatch.
- **FR-022 (P0):** In STRICT mode, fail record on dimension mismatch (no conversion attempt).
- **FR-023 (P0):** In LENIENT mode, emit warning and mark measurement `needs_review`.
- **FR-024 (P0):** Maintain versioned dimension registry with all supported dimensions.
- **FR-025 (P1):** Support domain-specific dimensions (`emissions_mass`, `volume_flow`, `energy_intensity`).
- **FR-026 (P1):** Detect dimensionless quantities and handle appropriately.

### 9.3 Canonical Conversion Engine

- **FR-030 (P0):** Convert scalar values to canonical units with full conversion trace.
- **FR-031 (P0):** Support array/batch conversion with per-element audit.
- **FR-032 (P0):** Support multi-step conversions (`MWh` → `kWh` → `MJ`) with step-by-step trace.
- **FR-033 (P0):** Support temperature affine conversions (°C ↔ °F ↔ K).
- **FR-034 (P0):** Require reference conditions (T, P) for basis-dependent conversions (`Nm3`, `scf`).
- **FR-035 (P0):** Emit `GLNORM-E201` (MISSING_REFERENCE_CONDITIONS) when required metadata absent.
- **FR-036 (P0):** Apply configurable precision rules per dimension/field.
- **FR-037 (P0):** Ensure numerical tolerance bounds (default `1e-9` relative).
- **FR-038 (P0):** Version and log all conversion factors used.
- **FR-039 (P1):** Support GWP version metadata (AR5/AR6) for CO2e conversions.
- **FR-040 (P1):** Support energy basis conversions (LHV/HHV) with explicit metadata.
- **FR-041 (P2):** Support custom conversion factor overrides with governance approval.

### 9.4 Entity Resolution (Fuels, Materials, Processes)

#### Resolution Pipeline
- **FR-050 (P0):** Resolve entities using deterministic priority order:
  1. Exact Reference ID match
  2. Exact canonical name match
  3. Exact alias match
  4. Rule-based normalization match
  5. Fuzzy match candidate retrieval
  6. LLM candidate suggestion (requires review)
- **FR-051 (P0):** Return best match with confidence score (0.0-1.0) and match method.
- **FR-052 (P0):** Emit `GLNORM-E400` (REFERENCE_NOT_FOUND) when no match found.
- **FR-053 (P0):** Emit `GLNORM-E401` (REFERENCE_AMBIGUOUS) when multiple high-confidence matches.
- **FR-054 (P0):** Flag low-confidence matches as `needs_review` (configurable threshold).
- **FR-055 (P0):** In STRICT mode, reject non-deterministic matches (fuzzy/LLM).

#### Scoring and Matching
- **FR-056 (P0):** Compute token overlap score for fuzzy matching.
- **FR-057 (P0):** Compute normalized edit distance for candidate ranking.
- **FR-058 (P1):** Apply alias weight boosting for preferred synonyms.
- **FR-059 (P1):** Use domain hints (sector/region) to disambiguate matches.
- **FR-060 (P1):** Return top-N candidates for review workflows.
- **FR-061 (P2):** Support LLM-assisted candidate generation with sandbox isolation.

#### Vocabulary Management
- **FR-062 (P0):** Load vocabularies by version with pinning support.
- **FR-063 (P0):** Handle deprecated entities: return `replaced_by` ID with warning.
- **FR-064 (P0):** Emit `GLNORM-E402` (ENTITY_DEPRECATED) when deprecated entity matched.
- **FR-065 (P0):** Include vocabulary_version in all resolution outputs.
- **FR-066 (P1):** Support vocabulary version rollback for historical replay.
- **FR-067 (P1):** Detect alias collisions across entity types with governance alerts.

### 9.5 Conversion Audit Log

- **FR-070 (P0):** Emit one audit event per normalization request.
- **FR-071 (P0):** Ensure audit log is append-only (immutable).
- **FR-072 (P0):** Include in audit: raw inputs, parsed AST, dimension signature.
- **FR-073 (P0):** Include in audit: all conversion steps with factors and intermediate values.
- **FR-074 (P0):** Include in audit: entity resolution method, confidence, vocabulary version.
- **FR-075 (P0):** Include in audit: agent version, policy mode, any overrides.
- **FR-076 (P0):** Generate stable `normalization_event_id` for correlation.
- **FR-077 (P0):** Ensure 100% audit coverage (every output has corresponding audit event).
- **FR-078 (P1):** Support tamper-evident hashing of audit events.
- **FR-079 (P1):** Enable audit replay to reproduce identical outputs with pinned versions.
- **FR-080 (P2):** Support audit event streaming to external systems (Kafka, etc.).

### 9.6 Policy Modes and Error Strategy

- **FR-081 (P0):** Support STRICT mode: fail fast, reject ambiguous outputs.
- **FR-082 (P0):** Support LENIENT mode: emit warnings, output `needs_review` flags.
- **FR-083 (P0):** Classify errors as hard (blocking) or soft (warning).
- **FR-084 (P0):** Use stable error codes (GLNORM-Exxx) for all errors/warnings.
- **FR-085 (P0):** Include structured payload in all error responses.
- **FR-086 (P1):** Support per-field policy overrides via configuration.
- **FR-087 (P1):** Allow configurable confidence thresholds for `needs_review` flagging.

### 9.7 Interfaces

#### Synchronous API
- **FR-090 (P0):** `POST /v1/normalize` endpoint accepting single normalization request.
- **FR-091 (P0):** Return canonical measurements, normalized entities, audit event ID.
- **FR-092 (P0):** Return structured errors with GLNORM-Exxx codes on failure.

#### Batch API
- **FR-093 (P0):** `POST /v1/normalize/batch` endpoint accepting JSONL input.
- **FR-094 (P1):** Support Parquet input/output for large datasets.
- **FR-095 (P1):** Return per-record results with partial success handling.
- **FR-096 (P1):** Output canonical dataset + separate audit dataset.

#### Library API
- **FR-097 (P1):** Provide in-process SDK for Python pipelines.
- **FR-098 (P1):** SDK functions: `normalize()`, `normalize_batch()`, `resolve_entity()`.
- **FR-099 (P2):** TypeScript SDK for frontend/Node.js integration.

#### CLI
- **FR-100 (P1):** `greenlang normalize --input <file> --output <file> --policy <mode>`.
- **FR-101 (P1):** `greenlang resolve-entity --name <name> --type <fuel|material|process>`.
- **FR-102 (P2):** `greenlang vocab diff --v1 <ver1> --v2 <ver2>` for vocabulary comparison.

---

## 10. Audit Event Schema (Proposed)

### 10.1 Event Envelope
```json
{
  "normalization_event_id": "norm-evt-...",
  "processed_at": "2026-01-28T00:00:00Z",
  "agent": {
    "agent_id": "GL-FOUND-X-003",
    "agent_version": "1.0.0"
  },
  "source": {
    "source_record_id": "rec-001",
    "pipeline_id": "ingest-xyz",
    "schema_version": "..."
  },
  "policy_mode": "STRICT",
  "status": "success|warning|failed",
  "errors": [],
  "warnings": [],
  "payload": {
    "measurements": [],
    "entities": []
  }
}
```

### 10.2 Measurement Audit Payload (example)
```json
{
  "field": "energy_used",
  "raw_value": 1200,
  "raw_unit": "kWh",
  "expected_dimension": "energy",
  "parsed_unit": {
    "normalized_unit_string": "kWh",
    "ast": "..."
  },
  "conversion": {
    "canonical_unit": "MJ",
    "canonical_value": 4320.0,
    "steps": [
      {
        "from": "kWh",
        "to": "MJ",
        "factor": 3.6,
        "method": "multiply"
      }
    ],
    "precision": {
      "rule": "dimension_default",
      "digits": 6
    }
  }
}
```

### 10.3 Entity Audit Payload (example)
```json
{
  "field": "fuel_type",
  "entity_type": "fuel",
  "raw_name": "Nat Gas",
  "resolution": {
    "reference_id": "GL-FUEL-NATGAS",
    "canonical_name": "Natural gas",
    "match_method": "alias",
    "confidence": 1.0,
    "vocabulary_version": "2026.01.0"
  }
}
```

---

## 11. Architecture

### 11.1 Logical Components
- **Unit Parser** → AST + normalized string
- **Dimension Engine** → dimension signature + compatibility checks
- **Conversion Engine** → canonical value + step trace
- **Reference Resolver** → candidate retrieval + scoring + selection
- **Vocabulary Manager** → loads versioned vocabularies, supports pinning
- **Audit Logger** → writes append-only events + indexes for search

### 11.2 Deployment Options
- **Service mode** (recommended): stateless API + external stores for vocab/audit
- **Embedded mode**: run as library inside ingestion jobs
- **Hybrid**: library for parsing/conversion + service for vocabulary + audit logging

### 11.3 Storage and Data Stores
- Reference vocabulary store: Postgres or dedicated Reference Data Service
- Audit store: append-only DB table and/or event stream (Kafka) with sink
- Cache: Redis (optional)

### 11.4 Determinism Guarantees
- All unit conversion factors and vocabulary versions must be explicit inputs to the normalization run (directly or via pinned configuration).
- LLM-assisted steps must not directly determine final output unless governance explicitly enables it (recommended default: suggestions only).

### 11.5 Data Model (Core Entities)

| Entity | Description | Key Fields |
|--------|-------------|------------|
| **NormalizationRequest** | Incoming normalization request | request_id, source_record_id, policy_mode, measurements[], entities[] |
| **CanonicalMeasurement** | Normalized measurement output | field, dimension, value, unit, raw_value, raw_unit, conversion_trace |
| **NormalizedEntity** | Resolved entity output | field, entity_type, reference_id, canonical_name, match_method, confidence |
| **ConversionStep** | Single step in conversion trace | from_unit, to_unit, factor, method, factor_version |
| **AuditEvent** | Immutable audit record | event_id, timestamp, request_id, status, payload, prev_event_hash |
| **VocabularyEntry** | Controlled vocabulary entity | reference_id, canonical_name, entity_type, status, aliases[], attributes{} |
| **UnitDefinition** | Unit registry entry | unit_symbol, canonical_symbol, dimension, conversion_factor, prefixes[] |
| **ResolutionCandidate** | Entity resolution candidate | reference_id, canonical_name, score, match_method |

### 11.6 Normalization State Machine

```
Request → PARSING → VALIDATING → CONVERTING → RESOLVING → AUDITING → COMPLETE
                        ↓              ↓           ↓           ↓
                     FAILED         FAILED      FAILED      FAILED
```

**States:**
- `PARSING`: Parsing input request, validating structure
- `VALIDATING`: Validating units against registry, checking dimensions
- `CONVERTING`: Converting measurements to canonical units
- `RESOLVING`: Resolving entities to vocabulary IDs
- `AUDITING`: Writing audit event to persistent store
- `COMPLETE`: Returning successful response
- `FAILED`: Error occurred; partial results may be available

**Transitions:**
- `PARSING` → `VALIDATING`: Input validated and parsed
- `VALIDATING` → `CONVERTING`: All units recognized, dimensions valid
- `VALIDATING` → `FAILED`: Unit parse failure or dimension mismatch (STRICT)
- `CONVERTING` → `RESOLVING`: All conversions complete
- `CONVERTING` → `FAILED`: Missing reference conditions or unsupported conversion
- `RESOLVING` → `AUDITING`: Entity resolution complete (or flagged needs_review)
- `RESOLVING` → `FAILED`: Required entity not found (STRICT)
- `AUDITING` → `COMPLETE`: Audit event persisted
- `AUDITING` → `FAILED`: Audit store unavailable

---

## 12. Dependency: Schema Validator

### 12.1 Contract
Schema Validator must provide:
- typed numeric values
- expected dimension per measurement
- required metadata constraints (e.g., GWP version required for CO2e)

### 12.2 Failure Handling
- If Schema Validator marks record invalid:
  - STRICT: do not normalize; emit audit failure event
  - LENIENT: attempt safe normalization only for fields marked safe

---

## 13. Non-Functional Requirements

> **Priority legend:** P0 = must-have (MVP/GA critical), P1 = should-have, P2 = nice-to-have.

### 13.1 Accuracy and Correctness

- **NFR-001 (P0):** Conversion factors must be curated, reviewed, and versioned.
- **NFR-002 (P0):** Conversion correctness for all supported units must be provable via golden tests.
- **NFR-003 (P0):** Normalization must be deterministic (same input + versions → same output).
- **NFR-004 (P0):** No silent data loss or transformation without audit trail.
- **NFR-005 (P0):** Entity resolution determinism: same input + vocabulary version → same reference ID.
- **NFR-006 (P1):** Numerical precision: maintain at least 15 significant digits during conversion.
- **NFR-007 (P1):** Confidence scores must be calibrated (predicted vs actual accuracy alignment).

### 13.2 Performance and Scale

- **NFR-010 (P0):** P95 latency for single-record normalization: < 50 ms.
- **NFR-011 (P0):** P95 latency for batch normalization (100 records): < 500 ms.
- **NFR-012 (P0):** P99 latency for single-record normalization: < 150 ms.
- **NFR-013 (P0):** Support horizontal scaling of normalization service.
- **NFR-014 (P1):** Vocabulary cache hit rate > 90% under steady traffic.
- **NFR-015 (P1):** Unit registry cache hit rate > 95%.
- **NFR-016 (P1):** Batch throughput: > 10,000 records/second per node.
- **NFR-017 (P2):** Entity resolution latency: < 10 ms for exact/alias matches.

### 13.3 Reliability

- **NFR-020 (P0):** Service availability target: 99.9% monthly.
- **NFR-021 (P0):** Graceful degradation if vocabulary store unavailable (use cached pinned versions).
- **NFR-022 (P0):** No data corruption on service crash or restart.
- **NFR-023 (P0):** Audit events must be durably persisted before returning success.
- **NFR-024 (P1):** Mean time to recovery (MTTR): < 5 minutes.
- **NFR-025 (P1):** Automatic failover for vocabulary store connections.
- **NFR-026 (P2):** Support read replicas for vocabulary queries.

### 13.4 Security and Compliance

- **NFR-030 (P0):** Encryption in transit (TLS 1.2+) for all communications.
- **NFR-031 (P0):** Encryption at rest for audit logs and vocabulary data.
- **NFR-032 (P0):** RBAC for audit log access (read/write separation).
- **NFR-033 (P0):** Audit immutability: append-only storage, no update/delete.
- **NFR-034 (P0):** No sensitive data in error messages or logs.
- **NFR-035 (P1):** Tamper-evident audit hashing with hash chaining.
- **NFR-036 (P1):** Data retention configurable by environment/namespace.
- **NFR-037 (P1):** Support for compliance hold (prevent deletion during legal hold).
- **NFR-038 (P2):** Audit log signing with platform key.

### 13.5 Observability

- **NFR-040 (P0):** Structured logs with `source_record_id`, `normalization_event_id`, `request_id`.
- **NFR-041 (P0):** Metrics: `parse_success_rate`, `unknown_unit_rate`, `conversion_failure_rate`.
- **NFR-042 (P0):** Metrics: `ambiguous_reference_rate`, `needs_review_rate`, `vocabulary_cache_hit_rate`.
- **NFR-043 (P0):** Metrics: latency histograms (p50, p95, p99) for normalize and resolve operations.
- **NFR-044 (P0):** Tracing spans: parse, convert, resolve, audit (OpenTelemetry-compatible).
- **NFR-045 (P1):** Alert hooks for: unknown unit spikes, ambiguous reference surges, audit failures.
- **NFR-046 (P1):** Alert hooks for: vocabulary version mismatch, conversion factor drift.
- **NFR-047 (P2):** Dashboard templates for Grafana/similar.

### 13.6 Operability

- **NFR-050 (P0):** Health endpoints: `/health`, `/ready`, `/live`.
- **NFR-051 (P0):** Clear error codes (GLNORM-Exxx) with documented remediation actions.
- **NFR-052 (P0):** Vocabulary reload without service restart.
- **NFR-053 (P1):** Cache invalidation API for vocabulary and unit registry.
- **NFR-054 (P1):** Graceful shutdown with request draining.
- **NFR-055 (P1):** Configuration hot-reload for policy thresholds.
- **NFR-056 (P2):** Blue-green deployment support for vocabulary updates.

### 13.7 SLO Targets

- **NFR-060 (P0):** Availability SLO: 99.9% monthly.
- **NFR-061 (P0):** Latency SLO: 95% of requests < 50ms.
- **NFR-062 (P1):** Error budget policy: escalation after 0.1% error rate.

---

## 14. Testing and QA

### 14.1 Test Categories
- Unit parsing tests (golden strings)
- Conversion golden tests (known factors)
- Dimensional mismatch tests (must fail)
- Entity resolution tests (exact, alias, fuzzy)
- Deprecation tests (replaced_by behavior)
- Regression tests with real production samples

### 14.2 Quality Gates for Release
- 0 critical conversion errors
- No regression in parse success KPI
- Audit schema compatibility tests passing
- Vocabulary diff report reviewed and approved

---

## 15. Rollout Plan (Suggested Milestones)

| Milestone | Deliverable |
|---|---|
| M0 | Canonical unit policy signed off + initial unit registry |
| M1 | Unit parsing + conversion engine + audit schema |
| M2 | Controlled vocabulary v1 + entity resolver + deprecation handling |
| M3 | Integration with Schema Validator + contract tests |
| M4 | Observability dashboards + load testing |
| M5 | Production rollout + monitoring + review workflows |

---

## 16. Acceptance Criteria (P0)

The following criteria must be satisfied for GA readiness:

1. **Determinism:** Given the same input + vocabulary version + unit registry version, normalization output hash is identical across runs.

2. **Unit Parsing Coverage:** ≥ 99.9% parse success rate for the top 200 units in production datasets, verified via golden tests.

3. **Conversion Accuracy:** 100% correctness for all supported unit conversions (within defined numerical tolerance of 1e-9 relative), verified via conversion matrix tests.

4. **Reference Resolution Rate:** ≥ 95% of raw entity values resolved deterministically (exact/alias/rule match) in initial target datasets.

5. **Audit Completeness:** 100% of normalized outputs have corresponding audit events with full conversion trace.

6. **Ambiguity Rate:** ≤ 2% of records flagged as `needs_review` after vocabulary stabilization period.

7. **Performance:** P95 latency < 50ms for single-record normalization with warm caches.

8. **Reproducibility:** Audit replay with pinned versions produces byte-identical outputs.

9. **Vocabulary Pinning:** Can pin vocabulary version and reproduce historical results within 0.001% tolerance.

10. **Error Detection:** All GLNORM-Exxx error codes have test coverage; golden tests for each error category passing.

---

## 17. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Canonical unit policy changes late | downstream churn | governance sign-off early; migration plan |
| Hidden basis assumptions (Nm3/scf) | wrong conversions | require metadata; strict failures |
| Alias collisions across entity types | wrong mapping | type-scoped alias tables + regression suite |
| Vocabulary drift across pipelines | inconsistent results | version pinning + controlled releases |
| Performance hits from fuzzy matching | latency | candidate index + caching + configurable matching levels |

---

## 18. Open Questions
1. Final canonical unit per dimension (energy/emissions especially).
2. Required support for `GWP` versioning (AR5/AR6) in output schema and audit.
3. Where will vocabularies be authored and published (repo vs service)?
4. What confidence thresholds define `needs_review` per entity type?
5. What are the first production datasets and their top units/entities to prioritize?

---

## 19. Security Model

### 19.1 Input Sanitization
- All unit strings and entity names are sanitized before processing.
- Unicode normalization (NFC) applied to prevent homoglyph attacks.
- Input length limits enforced: max 500 characters for unit strings, max 1000 characters for entity names.
- Special characters and control characters stripped or escaped.

### 19.2 Vocabulary Access Control
- **Read access:** All authenticated services can query vocabularies.
- **Write access:** Only governance-approved roles can modify vocabularies.
- **Version publishing:** Requires dual approval (data steward + domain SME).
- **Alias management:** Changes to alias mappings require audit trail entry.

### 19.3 Audit Log Security
- Audit logs are append-only; no update or delete operations permitted.
- Audit store access requires explicit RBAC permission.
- Write access limited to normalization service identity.
- Read access tiered: full access for auditors, filtered access for operators.
- Tamper-evident hash chaining: each event includes hash of previous event.
- Optional cryptographic signing of audit batches with platform key.

### 19.4 LLM Candidate Sandbox
- LLM-assisted entity resolution runs in isolated sandbox.
- No network access from LLM evaluation context.
- Timeout limits: max 5 seconds per LLM query.
- LLM suggestions are never auto-accepted; always flagged as `needs_review`.
- LLM prompts contain no sensitive data beyond the entity name being resolved.

### 19.5 Secrets Handling
- Vocabulary store credentials managed via secrets manager (Vault/KMS).
- API keys for external services never stored in configuration files.
- Credentials rotated automatically per environment policy.
- No secrets in error messages, logs, or audit payloads.
- Service-to-service authentication via mTLS or short-lived tokens.

### 19.6 Data Residency
- Vocabulary data location configurable per namespace.
- Audit logs stored in region-compliant storage.
- Cross-region vocabulary replication requires explicit policy approval.
- Data classification tags respected for vocabulary entities.

### 19.7 Rate Limiting and Abuse Prevention
- Per-client rate limits for API endpoints.
- Batch size limits: max 10,000 records per batch request.
- Vocabulary query rate limits to prevent enumeration attacks.
- Automatic blocking of clients exceeding error rate thresholds.

---

## 20. Operational Runbook

### 20.1 Common Failure Modes

| Failure | Symptoms | Remediation |
|---------|----------|-------------|
| Unit parse failure spike | GLNORM-E100 errors surge, alerts fire | Check for new data sources with non-standard units; add to unit registry |
| Vocabulary store outage | GLNORM-E501 errors, increased latency | Verify cached versions active; check store connectivity; failover to replica |
| Ambiguous reference surge | GLNORM-E401 errors spike, needs_review rate increases | Review new vocabulary entries; check for alias collisions |
| Audit log write failures | GLNORM-E600 errors, requests timeout | Check audit store capacity; verify network connectivity; scale audit writers |
| Conversion factor drift | Regression tests failing, output changes | Audit conversion factor changes; rollback if unapproved; notify governance |
| Cache invalidation lag | Stale vocabulary data served | Force cache invalidation; check TTL settings; verify cache replication |

### 20.2 On-Call Actions

- **Vocabulary Reload:** `POST /admin/vocabulary/reload?version=<ver>`
- **Cache Invalidation:** `POST /admin/cache/invalidate?scope=vocabulary|units|all`
- **Emergency Vocabulary Rollback:** `POST /admin/vocabulary/rollback?to_version=<ver>`
- **Rate Limit Override:** `POST /admin/rate-limits/override?client_id=<id>&limit=<n>`
- **Fuzzy Matching Disable:** Set feature flag `fuzzy_matching_enabled=false` in config.
- **LLM Resolution Disable:** Set feature flag `llm_candidates_enabled=false` in config.

### 20.3 Health Check Endpoints

- `GET /health` - Basic liveness check (returns 200 if service running)
- `GET /ready` - Readiness including:
  - Vocabulary store connectivity
  - Unit registry loaded
  - Audit store connectivity
  - Cache warm status
- `GET /metrics` - Prometheus-format metrics endpoint

### 20.4 Monitoring Dashboards

Recommended dashboard panels:
- Unit parse success rate (target: > 99.9%)
- Reference resolution success rate (target: > 95%)
- Needs review rate (target: < 2%)
- Normalization latency percentiles (p50, p95, p99)
- Vocabulary cache hit rate (target: > 90%)
- Audit event write latency
- Error rate by error code

### 20.5 Incident Response Playbook

1. **P1 - Service Down:** Escalate immediately; check health endpoints; verify dependencies.
2. **P2 - High Error Rate:** Check error distribution by code; identify root cause; apply mitigation.
3. **P3 - Performance Degradation:** Check cache hit rates; review recent vocabulary changes; scale if needed.
4. **P4 - Data Quality Issues:** Investigate affected records; notify downstream consumers; prepare correction plan.

---

## 21. Detailed API Examples

### 21.1 Normalize Request (Single Record)
```json
{
  "source_record_id": "meter-2026-001",
  "policy_mode": "STRICT",
  "vocabulary_version": "2026.01.0",
  "measurements": [
    {
      "field": "energy_consumption",
      "value": 1500,
      "unit": "kWh",
      "expected_dimension": "energy"
    },
    {
      "field": "fuel_volume",
      "value": 250,
      "unit": "Nm3",
      "expected_dimension": "volume",
      "metadata": {
        "reference_conditions": {
          "temperature_C": 0,
          "pressure_kPa": 101.325
        }
      }
    }
  ],
  "entities": [
    {
      "field": "fuel_type",
      "entity_type": "fuel",
      "raw_name": "Nat Gas",
      "hints": {
        "region": "EU",
        "sector": "energy"
      }
    }
  ]
}
```

### 21.2 Normalize Response (Success)
```json
{
  "source_record_id": "meter-2026-001",
  "status": "success",
  "canonical_measurements": [
    {
      "field": "energy_consumption",
      "dimension": "energy",
      "value": 5400.0,
      "unit": "MJ",
      "raw_value": 1500,
      "raw_unit": "kWh",
      "conversion_trace": {
        "steps": [
          {"from": "kWh", "to": "MJ", "factor": 3.6, "method": "multiply"}
        ],
        "factor_version": "2026.01.0"
      },
      "warnings": []
    },
    {
      "field": "fuel_volume",
      "dimension": "volume",
      "value": 250.0,
      "unit": "m3",
      "raw_value": 250,
      "raw_unit": "Nm3",
      "conversion_trace": {
        "steps": [
          {"from": "Nm3", "to": "m3", "factor": 1.0, "method": "basis_conversion", "reference_conditions": {"T": 273.15, "P": 101.325}}
        ],
        "factor_version": "2026.01.0"
      },
      "warnings": []
    }
  ],
  "normalized_entities": [
    {
      "field": "fuel_type",
      "entity_type": "fuel",
      "raw_name": "Nat Gas",
      "reference_id": "GL-FUEL-NATGAS",
      "canonical_name": "Natural gas",
      "vocabulary_version": "2026.01.0",
      "match_method": "alias",
      "confidence": 1.0,
      "needs_review": false,
      "warnings": []
    }
  ],
  "audit": {
    "normalization_event_id": "norm-evt-abc123",
    "status": "success"
  }
}
```

### 21.3 Normalize Response (Failure - Dimension Mismatch)
```json
{
  "source_record_id": "meter-2026-002",
  "status": "failed",
  "canonical_measurements": [],
  "normalized_entities": [],
  "errors": [
    {
      "code": "GLNORM-E200",
      "severity": "error",
      "path": "/measurements/0",
      "message": "Dimension mismatch: expected 'energy', got 'mass'",
      "expected": {"dimension": "energy"},
      "actual": {"dimension": "mass", "unit": "kg", "value": 100},
      "hint": {
        "suggestion": "Use energy units like kWh, MJ, or GJ",
        "docs": "gl://docs/units#energy"
      }
    }
  ],
  "audit": {
    "normalization_event_id": "norm-evt-def456",
    "status": "failed"
  }
}
```

### 21.4 Batch Normalize Request
```json
{
  "policy_mode": "LENIENT",
  "vocabulary_version": "2026.01.0",
  "records": [
    {
      "source_record_id": "batch-001",
      "measurements": [
        {"field": "energy", "value": 100, "unit": "kWh", "expected_dimension": "energy"}
      ],
      "entities": [
        {"field": "fuel", "entity_type": "fuel", "raw_name": "Diesel"}
      ]
    },
    {
      "source_record_id": "batch-002",
      "measurements": [
        {"field": "energy", "value": 200, "unit": "MWh", "expected_dimension": "energy"}
      ],
      "entities": [
        {"field": "fuel", "entity_type": "fuel", "raw_name": "Heavy Fuel Oil"}
      ]
    }
  ]
}
```

### 21.5 Batch Normalize Response
```json
{
  "summary": {
    "total": 2,
    "success": 2,
    "failed": 0,
    "warnings": 1
  },
  "results": [
    {
      "source_record_id": "batch-001",
      "status": "success",
      "canonical_measurements": [
        {"field": "energy", "value": 360.0, "unit": "MJ", "dimension": "energy"}
      ],
      "normalized_entities": [
        {"field": "fuel", "reference_id": "GL-FUEL-DIESEL", "confidence": 1.0, "match_method": "exact"}
      ],
      "audit": {"normalization_event_id": "norm-evt-batch-001"}
    },
    {
      "source_record_id": "batch-002",
      "status": "warning",
      "canonical_measurements": [
        {"field": "energy", "value": 720000.0, "unit": "MJ", "dimension": "energy"}
      ],
      "normalized_entities": [
        {
          "field": "fuel",
          "reference_id": "GL-FUEL-HFO",
          "confidence": 0.85,
          "match_method": "fuzzy",
          "needs_review": true
        }
      ],
      "warnings": [
        {
          "code": "GLNORM-E403",
          "severity": "warning",
          "path": "/entities/0",
          "message": "Low confidence match (0.85) for entity resolution"
        }
      ],
      "audit": {"normalization_event_id": "norm-evt-batch-002"}
    }
  ]
}
```

### 21.6 Entity Resolution Request
```json
{
  "entity_type": "fuel",
  "raw_name": "Natural-gas",
  "vocabulary_version": "2026.01.0",
  "hints": {
    "region": "NA",
    "sector": "utilities"
  },
  "options": {
    "return_candidates": true,
    "max_candidates": 5
  }
}
```

### 21.7 Entity Resolution Response
```json
{
  "best_match": {
    "reference_id": "GL-FUEL-NATGAS",
    "canonical_name": "Natural gas",
    "match_method": "rule",
    "confidence": 0.98,
    "needs_review": false
  },
  "candidates": [
    {"reference_id": "GL-FUEL-NATGAS", "canonical_name": "Natural gas", "score": 0.98},
    {"reference_id": "GL-FUEL-LNG", "canonical_name": "Liquefied natural gas", "score": 0.65},
    {"reference_id": "GL-FUEL-CNG", "canonical_name": "Compressed natural gas", "score": 0.60}
  ],
  "resolution_trace": {
    "input_normalized": "natural gas",
    "methods_tried": ["exact_id", "exact_name", "alias", "rule"],
    "matched_at": "rule",
    "rule_applied": "hyphen_removal + case_normalization"
  }
}
```

---

## Appendix A: Reference ID Pattern

- Fuels: `GL-FUEL-<TOKEN>`
- Materials: `GL-MAT-<TOKEN>`
- Processes: `GL-PROC-<TOKEN>`

`<TOKEN>` should be stable and not tied to display name changes.

---

## Appendix B: Error Code Taxonomy (GLNORM-Exxx)

> **Error code format:** `GLNORM-Exxx` where the first digit indicates category.

### B.1 Unit Parsing Errors (E1xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E100` | UNIT_PARSE_FAILED | Unit string could not be parsed | no |
| `GLNORM-E101` | UNKNOWN_UNIT | Unit not found in unit registry | no |
| `GLNORM-E102` | INVALID_PREFIX | Unrecognized SI prefix | no |
| `GLNORM-E103` | INVALID_EXPONENT | Malformed exponent notation | no |
| `GLNORM-E104` | AMBIGUOUS_UNIT | Unit string matches multiple units | no |
| `GLNORM-E105` | UNSUPPORTED_COMPOUND | Compound unit structure not supported | no |
| `GLNORM-E106` | LOCALE_PARSE_ERROR | Locale-specific parsing failed | no |

### B.2 Dimension Errors (E2xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E200` | DIMENSION_MISMATCH | Computed dimension doesn't match expected | no |
| `GLNORM-E201` | DIMENSION_UNKNOWN | Dimension not recognized | no |
| `GLNORM-E202` | DIMENSION_INCOMPATIBLE | Cannot convert between incompatible dimensions | no |
| `GLNORM-E203` | DIMENSIONLESS_EXPECTED | Expected dimensionless but got dimension | no |
| `GLNORM-E204` | DIMENSION_EXPECTED | Expected dimension but got dimensionless | no |

### B.3 Conversion Errors (E3xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E300` | CONVERSION_NOT_SUPPORTED | Conversion path not available | no |
| `GLNORM-E301` | MISSING_REFERENCE_CONDITIONS | Temperature/pressure conditions required but missing | no |
| `GLNORM-E302` | INVALID_REFERENCE_CONDITIONS | Reference conditions out of valid range | no |
| `GLNORM-E303` | CONVERSION_FACTOR_MISSING | Required conversion factor not found | no |
| `GLNORM-E304` | PRECISION_OVERFLOW | Result exceeds precision bounds | no |
| `GLNORM-E305` | GWP_VERSION_MISSING | GWP version required for CO2e conversion | no |
| `GLNORM-E306` | BASIS_MISSING | Energy basis (LHV/HHV) required but not provided | no |
| `GLNORM-E307` | CONVERSION_FACTOR_DEPRECATED | Using deprecated conversion factor (warning) | n/a |

### B.4 Entity Resolution Errors (E4xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E400` | REFERENCE_NOT_FOUND | No matching entity in vocabulary | no |
| `GLNORM-E401` | REFERENCE_AMBIGUOUS | Multiple high-confidence matches | no |
| `GLNORM-E402` | ENTITY_DEPRECATED | Matched entity is deprecated (warning) | n/a |
| `GLNORM-E403` | LOW_CONFIDENCE_MATCH | Match confidence below threshold | no |
| `GLNORM-E404` | VOCABULARY_NOT_FOUND | Requested vocabulary version not available | no |
| `GLNORM-E405` | ENTITY_TYPE_MISMATCH | Entity found but wrong type (fuel vs material) | no |
| `GLNORM-E406` | ALIAS_COLLISION | Alias maps to multiple entities | no |
| `GLNORM-E407` | LLM_CANDIDATE_ONLY | Only LLM-suggested match available (needs review) | no |

### B.5 Vocabulary Errors (E5xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E500` | VOCABULARY_VERSION_MISMATCH | Requested version incompatible | no |
| `GLNORM-E501` | VOCABULARY_LOAD_FAILED | Failed to load vocabulary from store | yes |
| `GLNORM-E502` | VOCABULARY_CORRUPTED | Vocabulary data integrity check failed | no |
| `GLNORM-E503` | VOCABULARY_EXPIRED | Pinned vocabulary version no longer available | no |
| `GLNORM-E504` | GOVERNANCE_REQUIRED | Operation requires governance approval | no |

### B.6 Audit Errors (E6xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E600` | AUDIT_WRITE_FAILED | Failed to write audit event | yes |
| `GLNORM-E601` | AUDIT_STORE_UNAVAILABLE | Audit store not reachable | yes |
| `GLNORM-E602` | AUDIT_INTEGRITY_VIOLATION | Audit chain integrity check failed | no |
| `GLNORM-E603` | REPLAY_MISMATCH | Audit replay produced different result | no |

### B.7 System/Limit Errors (E9xx)

| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLNORM-E900` | LIMIT_EXCEEDED | Input size/complexity limit exceeded | no |
| `GLNORM-E901` | TIMEOUT | Operation timed out | yes (bounded) |
| `GLNORM-E902` | RESOURCE_EXHAUSTED | Memory/CPU quota exceeded | yes (bounded) |
| `GLNORM-E903` | SERVICE_UNAVAILABLE | Normalizer service unavailable | yes |
| `GLNORM-E904` | INTERNAL_ERROR | Unexpected internal error | no |

### B.8 Error Response Structure

Every error response MUST include:
```json
{
  "code": "GLNORM-E200",
  "severity": "error",
  "path": "/measurements/0",
  "message": "Dimension mismatch: expected 'energy', got 'mass'",
  "expected": {"dimension": "energy"},
  "actual": {"dimension": "mass", "unit": "kg"},
  "hint": {
    "suggestion": "Use energy units like kWh, MJ, or GJ",
    "docs": "gl://docs/units#energy"
  }
}
