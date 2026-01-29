# Product Requirements Document (PRD)
## GreenLang - Unit & Reference Normalizer (GL-FOUND-X-003)

**Agent Name:** Unit & Reference Normalizer  
**Agent ID:** GL-FOUND-X-003  
**Family:** NormalizationFamily  
**Layer:** Foundation & Governance  
**Priority:** High  
**Status:** Draft v0.2  
**Date:** January 28, 2026  
**Upstream Dependency:** Schema Validator  
**Downstream Consumers (typical):** Emissions calculators, factor selectors, aggregators, reporting services

---

## 0. Document Control

| Field | Value |
|---|---|
| Owner | GreenLang Foundation & Governance |
| Reviewers | Data Engineering, ESG Analytics, Governance/Audit, Domain SMEs |
| Change policy | Any change to canonical unit set or vocabulary resolution rules requires version bump + regression tests |
| Target release | v1 (production-ready) after M5 milestone |

### 0.1 Revision History

| Version | Date | Notes |
|---|---|---|
| v0.1 | January 28, 2026 | Initial draft |
| v0.2 | January 28, 2026 | Expanded “in-depth” technical requirements, governance workflow, and audit schema |

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

## 9. Functional Requirements (Detailed)

### FR-1 Unit Parsing & Normalization
**Goal:** Convert raw unit strings into normalized, machine-readable units.

**Requirements**
- Handle casing/whitespace/punctuation variations.
- Handle common synonyms (`"lbs"` → `lb`, `"litre"` → `L`).
- Output:
  - `normalized_unit_string`
  - `parsed_unit_ast`
  - `dimension_signature`
- Provide `UNIT_PARSE_FAILED` with suggestions on failure.

**Acceptance Criteria**
- Top unit list coverage hits KPI (≥ 99.9% parse success).
- Parser behavior is deterministic.

---

### FR-2 Dimensional Validation
**Goal:** Ensure units match the schema’s expected dimension.

**Requirements**
- Compare computed dimension signature with `expected_dimension`.
- If mismatch:
  - STRICT: fail record (no conversion)
  - LENIENT: emit warning and mark measurement `needs_review`
- Emit `DIMENSION_MISMATCH` with expected vs actual.

**Acceptance Criteria**
- No silent conversions across incompatible dimensions.

---

### FR-3 Canonical Conversion Engine
**Goal:** Convert values to canonical units, with full trace.

**Requirements**
- Support scalar and array conversion.
- Support multi-step conversions (e.g., `MWh` → `kWh` → `MJ`), but audit must reflect steps.
- Support temperature affine conversions.
- Basis conversions (e.g., `Nm3`) only with required metadata:
  - reference temperature/pressure conditions
- Apply consistent rounding rules:
  - configurable precision per dimension/field
  - no implicit rounding beyond policy

**Acceptance Criteria**
- Conversion factors used are versioned and logged.
- Numerical tolerance bounds are defined (e.g., `1e-9` relative).

---

### FR-4 Entity Resolution (Fuels, Materials, Processes)
**Goal:** Map raw strings/codes → controlled vocabulary IDs.

**Resolution pipeline**
1. Exact ID match
2. Exact canonical name match
3. Exact alias match
4. Rule normalization (tokenization, punctuation removal, known abbreviation expansions)
5. Fuzzy match candidate retrieval + scoring
6. Optional LLM candidate generation (suggest only)

**Scoring features (minimum)**
- token overlap score
- normalized edit distance
- alias weight
- domain hints (sector/region) if provided

**Outputs**
- best match + confidence
- top-N candidates (optional) for review workflows
- `REFERENCE_NOT_FOUND` or `REFERENCE_AMBIGUOUS` errors as needed

**Acceptance Criteria**
- Deterministic match methods have priority over fuzzy/LLM methods.
- Low-confidence outputs are flagged `needs_review` and never auto-committed as final IDs in STRICT mode.

---

### FR-5 Conversion Audit Log
**Goal:** Provide governance traceability and replay.

**Requirements**
- Emit one audit event per record (or per measurement + entity; configurable).
- Audit log must be **append-only** (immutable).
- Audit must include:
  - raw inputs
  - parsed unit AST and dimension signature
  - conversion steps and factors
  - reference resolution method + confidence
  - versions: agent, vocabulary, factor set
  - policy mode and any overrides

**Acceptance Criteria**
- 100% coverage: every normalized output has an audit event.
- Replays produce identical outputs when pinned to the same versions.

---

### FR-6 Policy Modes and Error Strategy
**Policy Modes**
- STRICT: fail fast, no ambiguous outputs
- LENIENT: output `needs_review` + warnings where safe

**Error Types**
- Hard errors: block output (e.g., unit parse failure in STRICT)
- Soft warnings: allow output but flag (e.g., deprecated ID replaced)

**Acceptance Criteria**
- Each error/warning uses a stable code and structured payload.

---

### FR-7 Interfaces
**Synchronous API**
- `POST /normalize` returns canonical measurements, normalized entities, and audit event id.

**Batch API**
- Accept JSONL/Parquet; output canonical dataset + audit dataset.

**Library API**
- In-process SDK for pipelines.

**Acceptance Criteria**
- Backward-compatible schema evolution (additive changes) for v1.

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

### 13.1 Accuracy
- Factor sets are curated, reviewed, versioned.
- Conversion correctness for supported units must be provable via tests.

### 13.2 Performance
- p95 latency target (service mode): < 100 ms for a typical single record
- Must support horizontal scaling

### 13.3 Reliability
- 99.9% service availability target
- Degraded mode if vocabulary store unavailable (use cached pinned versions)

### 13.4 Security and Compliance
- Encryption in transit + at rest
- RBAC for audit log access
- Audit immutability (append-only), optional tamper-evident hashing
- Data retention configurable by environment

### 13.5 Observability
- Metrics:
  - parse_success_rate
  - unknown_unit_rate
  - conversion_failure_rate
  - ambiguous_reference_rate
  - needs_review_rate
  - latency histograms
- Tracing: `source_record_id`, `normalization_event_id`
- Alerts:
  - spikes in unknown units
  - spikes in ambiguous references
  - vocabulary version mismatch events

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

## 16. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Canonical unit policy changes late | downstream churn | governance sign-off early; migration plan |
| Hidden basis assumptions (Nm3/scf) | wrong conversions | require metadata; strict failures |
| Alias collisions across entity types | wrong mapping | type-scoped alias tables + regression suite |
| Vocabulary drift across pipelines | inconsistent results | version pinning + controlled releases |
| Performance hits from fuzzy matching | latency | candidate index + caching + configurable matching levels |

---

## 17. Open Questions
1. Final canonical unit per dimension (energy/emissions especially).
2. Required support for `GWP` versioning (AR5/AR6) in output schema and audit.
3. Where will vocabularies be authored and published (repo vs service)?
4. What confidence thresholds define `needs_review` per entity type?
5. What are the first production datasets and their top units/entities to prioritize?

---

## Appendix A: Draft Reference ID Pattern

- Fuels: `GL-FUEL-<TOKEN>`
- Materials: `GL-MAT-<TOKEN>`
- Processes: `GL-PROC-<TOKEN>`

`<TOKEN>` should be stable and not tied to display name changes.

---

## Appendix B: Error Codes (Draft)

- `UNIT_PARSE_FAILED`
- `DIMENSION_MISMATCH`
- `UNKNOWN_UNIT`
- `CONVERSION_NOT_SUPPORTED`
- `MISSING_REQUIRED_METADATA`
- `REFERENCE_NOT_FOUND`
- `REFERENCE_AMBIGUOUS`
- `REFERENCE_DEPRECATED_REPLACED`
