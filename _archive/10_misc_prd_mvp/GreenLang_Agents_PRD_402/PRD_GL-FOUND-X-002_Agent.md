# PRD: GreenLang Schema Compiler & Validator (GL-FOUND-X-002)

**Agent family:** SchemaFamily
**Layer:** Foundation & Governance
**Primary domains:** Schema validation, payload normalization (cross-cutting)
**Priority:** P0 (cross-cutting; blocks downstream agents)
**Doc version:** 1.0
**Last updated:** 2026-01-29 (Asia/Kolkata)
**Owner:** (TBD)
**Reviewers:** Platform Eng, Schema Governance, Domain Leads, Security

---

## 1. Executive Summary

**GreenLang Schema Compiler & Validator (GL-FOUND-X-002)** is the **foundation & governance layer agent** responsible for **validating, normalizing, and providing machine-fixable feedback** for YAML/JSON payloads against GreenLang schemas. It compiles schemas into an efficient Intermediate Representation (IR), validates payloads with deterministic error reporting, and produces actionable fix suggestions.

This agent is the "validation backbone" for GreenLang payloads:
- Compiles GreenLang schemas into a **fast, cached IR** for repeated validation.
- Validates payloads against schemas with **precise, deterministic diagnostics**.
- Normalizes payloads to **canonical form** (types, units, defaults).
- Generates **machine-fixable patches** (JSON Patch) with safety classifications.
- Enforces **unit compatibility** and **cross-field governance rules**.

---

## 2. Problem Statement

GreenLang needs a **single, authoritative** foundation-layer agent that can:

- Validate incoming **YAML/JSON payloads** against a specified **GreenLang schema version**
- Detect and explain:
  - Missing/extra fields
  - Type mismatches
  - Unit inconsistencies (incompatible units, missing units, wrong canonical unit)
  - Invalid ranges/constraints
  - Rule violations (cross-field, conditional, domain packs)
- Produce outputs that are useful both to humans **and** machines:
  - A structured validation report
  - A normalized payload (canonical form)
  - **Machine-fixable** suggestions (patches + hints)

Without this agent, every downstream consumer re-implements validation differently, creating drift, inconsistent governance, and fragile integrations.

---

## 3. Goals / Objectives

### 3.1 Primary goals (MVP -> v1)
1. **Deterministic schema validation** for JSON/YAML payloads against a schema reference (id + version).
2. **High-signal diagnostics**:
   - Precise field paths
   - Expected vs actual values/types
   - Unit/range specifics
3. **Normalization** to canonical representation (types, units, defaults, key canonicalization).
4. **Machine-fixable hints**: emit patches (JSON Patch preferred) with safety constraints and confidence.

### 3.2 Secondary goals (v1+)
- Schema compilation to an intermediate representation (IR) for fast repeated validations
- Rule-pack system (pluggable governance rules across domains)
- Linting and deprecation guidance (schema evolution governance)
- Compatibility mode for schema migrations (suggest updated field names, etc.)

### 3.3 Non-goals
- Persisting payloads or owning a database of customer data
- Being the source of truth for schema authoring UI (it can validate schemas, but does not replace schema editor tooling)
- Automated execution of fixes in upstream systems (it only suggests)

---

## 4. Stakeholders and Users

### 4.1 Primary Stakeholders
- **Schema Governance Team:** owns schema standards, versioning, and evolution policies.
- **Platform Engineering:** owns runtime, validation service, performance, and scaling.
- **Security Team:** YAML/JSON parsing safety, input validation, ReDoS protection.
- **Domain Leads:** require consistent validation across domain-specific schemas.
- **Developer Experience (DX):** SDK usability, error message clarity, documentation.

### 4.2 Primary User Personas
1. **Schema Author (Governance):** Defines schemas and validation rules; needs strict, explainable failures and schema self-validation.
2. **Platform Engineer:** Integrates validation into APIs, pipelines, and agents; needs high performance and reliable error codes.
3. **Domain Agent Developer:** Builds agents that consume validated payloads; needs consistent, predictable validation behavior.
4. **Data Producer (Submitter):** Sends payloads to GreenLang; wants actionable errors, auto-fix suggestions, and clear migration guidance.
5. **Auditor / Compliance Analyst:** Reviews validation reports for compliance; needs deterministic, reproducible results.

---

## 5. Assumptions & Definitions

### 5.1 GreenLang schema (assumed baseline)
A GreenLang schema is represented in YAML/JSON and supports:
- Primitive and composite types: string/number/integer/boolean/object/array
- Required vs optional fields
- Defaults
- Constraints: min/max, pattern, enum, minItems/maxItems, etc.
- Units metadata for measurable quantities (e.g., `unit: kWh`, `dimension: energy`)
- References and reuse: `$ref`-like or `imports` semantics
- Versioning: semver-ish (`1.2.3`) or date-based (`2026-01`)

> Note: This PRD defines interfaces that work whether GreenLang is a superset of JSON Schema or a custom schema language. Implementation details can adapt.

### 5.2 Definitions and Glossary

| Term | Definition |
|------|------------|
| **SchemaRef** | `{schema_id, version}` (and optionally `variant/profile`) identifying a schema. |
| **Schema IR** | Intermediate Representation: compiled, validated schema for fast repeated validation. |
| **Finding** | A validation result entry with code, severity, path, message, and hint. |
| **Normalization** | Transforming payload into canonical form without changing semantics. |
| **Fix Suggestion** | A machine-applyable patch (JSON Patch) + rationale + preconditions. |
| **Coercion** | Type conversion from input form to schema-expected type (e.g., "42" → 42). |
| **Rule Pack** | Versioned set of cross-field validation rules (e.g., `climate/v1@2026-01`). |
| **Unit Dimension** | Physical quantity category (energy, mass, volume, temperature). |
| **Canonical Unit** | Schema-specified standard unit for normalization (e.g., kWh for energy). |
| **JSON Pointer** | RFC 6901 path notation for referencing JSON values (e.g., `/foo/bar/0`). |
| **JSON Patch** | RFC 6902 format for describing changes to JSON documents. |
| **Validation Profile** | Strictness level: `strict`, `standard`, or `permissive`. |
| **Schema Bomb** | Malicious schema with deeply nested $refs designed to exhaust resources. |
| **ReDoS** | Regular Expression Denial of Service via catastrophic backtracking. |

---

## 6. High-Level Workflow

1. **Ingest**: parse YAML/JSON safely; enforce size/complexity limits.
2. **Resolve Schema**: locate schema by SchemaRef (registry/packaging layer).
3. **Compile Schema**: produce validated IR (cached).
4. **Validate**:
   - Structural (shape, types, required)
   - Constraint (ranges, patterns, enums)
   - Units (compatibility + conversions)
   - Cross-field rules (rule engine)
   - Linting (style/deprecation)
5. **Normalize** (optional, configurable): defaults, canonical types, unit canonicalization.
6. **Suggest Fixes** (optional): patches for safe, unambiguous repairs.
7. **Emit Outputs**: report + normalized payload + fix suggestions.

---

## 7. Functional Requirements

> **Priority legend:** P0 = must-have (MVP/GA critical), P1 = should-have, P2 = nice-to-have.

### 7.1 Inputs
The agent MUST accept:
- `payload`: YAML or JSON (string or object)
- `schema_ref`: `schema_id`, `version` (required)
- `validation_profile`: one of:
  - `strict` (unknown fields = error; coercions off by default)
  - `standard` (unknown fields = warning; safe coercions on)
  - `permissive` (unknown fields allowed; best-effort normalization)
- `rules`: list of rule pack identifiers and/or inline rules
- `options` (all optional):
  - `normalize`: boolean (default true in standard/permissive)
  - `emit_patches`: boolean (default true)
  - `max_errors`: int (default 100)
  - `fail_fast`: boolean
  - `unit_system`: `{canonical: SI|custom, allow_conversions: true/false}`
  - `timezone`, `locale` (for date parsing rules)
  - `unknown_field_policy`: error|warn|ignore
  - `coercion_policy`: off|safe|aggressive

### 7.2 Outputs
The agent MUST produce:

1. **Validation report** (machine-readable JSON):
   - Summary: valid/invalid, error counts, warnings, schema used, timings
   - Findings list with stable error codes and field paths
2. **Normalized payload** (if enabled):
   - Canonical keys, types, units, defaults applied, metadata preserved per policy
3. **Fix suggestions** (if enabled):
   - Ordered list of patches (JSON Patch operations) with preconditions

### 7.3 Schema validation capabilities
The validator MUST support:
- Required field checks (including nested)
- Type checks and coercions (configurable)
- Constraint checks:
  - numeric: min/max, exclusive bounds
  - string: pattern, minLength/maxLength
  - arrays: minItems/maxItems, uniqueItems
  - enums
- Object constraints:
  - additionalProperties policy
  - property dependencies
- Reference resolution (`$ref`-like) with cycle detection
- Schema evolution signals:
  - deprecated fields
  - renamed fields (emit migration hint)
  - removed fields

### 7.4 Unit validation & conversion
The validator MUST:
- Detect missing units when schema requires units
- Detect incompatible units (e.g., kg vs kWh)
- Convert compatible units to canonical units during normalization (if allowed)
- Handle unit annotations in common shapes:
  - `{ "value": 10, "unit": "kWh" }`
  - `"10 kWh"` (string parsing as optional feature)
  - separate unit field (`energy_value`, `energy_unit`) if schema declares pairing

### 7.5 Cross-field rule engine
The agent MUST support governance rules such as:
- Conditional requirements: if `fuel_type=gas` then `methane_slip` required
- Ranges dependent on context: `temperature` range depends on `sensor_model`
- Consistency: sum of components equals total within tolerance
- Domain packs: e.g., `climate/v1`, `finance/v2` (extensible)

Rule format options (implementation choice):
- JSONLogic-like expressions
- CEL (Common Expression Language)-style
- Custom DSL that compiles to an AST

### 7.6 Linting
Linting is non-blocking by default (warnings) and includes:
- Unknown fields / suspicious keys (typos, close matches)
- Deprecated fields usage
- Non-canonical casing/keys
- Unit formatting
- Schema anti-patterns (if validating schemas too)

### 7.7 Schema compilation (compiler) requirements
The agent is explicitly a **Schema Compiler & Validator**. "Compiler" here means: convert a schema source document into a fast, validated, portable **Intermediate Representation (IR)** that the validator executes.

The compiler MUST:
- Parse schema YAML/JSON to a strongly typed **Schema AST**
- Validate the schema itself (see 6.8)
- Resolve `$ref`/imports into a fully linked graph
- Emit a compact IR with:
  - Flattened property maps for O(1) property lookup
  - Precompiled numeric constraints and string constraints
  - Precompiled regex with guardrails (timeout/size limits)
  - Unit/dimension metadata normalized to a standard form
  - Precomputed "required paths" and conditional requirement indexes
  - Rule pack bindings (resolved and type-checked)

Compiler outputs (internal or exposed via tooling):
- `schema_ir`: binary or JSON (implementation choice)
- `schema_hash`: stable hash over canonical schema representation
- `compile_warnings`: deprecations, anti-patterns, ambiguous units, etc.

Optional but recommended:
- `greenlang compile-schema --schema <ref|file> --out schema.ir`

### 7.8 Schema self-validation (governance)
Before any payload validation, the agent MUST be able to validate schema documents for governance.

Schema self-validation checks include:
- Reference resolution (no missing refs; cycle detection with clear trace)
- No duplicate property keys after canonicalization/alias resolution
- Deprecated/renamed field metadata is well-formed (must specify target field + version window)
- Unit metadata is consistent:
  - unit exists in allowed unit catalog
  - dimension is specified for unit-bearing fields
  - canonical unit exists and is convertible
- Constraints are internally consistent (min <= max; enums match type, etc.)
- Rule expressions are type-checkable (where possible) and cannot reference missing paths

### 7.9 Validation profile policy matrix
| Behavior | strict | standard | permissive |
|---|---:|---:|---:|
| Unknown fields | error | warn | ignore (but record) |
| Safe type coercions | off | on | on |
| Defaults applied | optional | on | on |
| Unit conversions | optional | on | on |
| Drop empty optional objects | off | optional | on |
| Lint severity | warn/info | warn/info | warn/info |
| Fail-fast | optional | optional | optional |

> "Safe coercions" are those that are reversible and unambiguous (e.g., `"42"` -> `42` for integer).

### 7.10 Limits and guardrails (must be configurable)
The service MUST enforce configurable limits to avoid abuse and pathological inputs:
- Max payload bytes (default 1 MB)
- Max schema bytes (default 2 MB)
- Max object depth (default 50)
- Max array items per array (default 10,000)
- Max total nodes (default 200,000)
- Max `$ref` expansions (default 10,000)
- Max regex length and complexity; avoid catastrophic backtracking
- Max findings (`max_errors`, default 100)

When a limit is exceeded, emit a single deterministic error:
- `GLSCHEMA-E900` with details about which limit was hit.

### 7.11 Numbered Functional Requirements (Summary)

#### Intake and Parsing
- **FR-001 (P0):** Accept payload via API/CLI as YAML or JSON; parse safely with size limits.
- **FR-002 (P0):** Accept schema_ref (schema_id + version) and resolve from registry.
- **FR-003 (P0):** Accept validation_profile (strict/standard/permissive) with documented defaults.
- **FR-004 (P1):** Accept optional rule pack references and inline rules.
- **FR-005 (P1):** Accept optional processing options (normalize, emit_patches, max_errors, etc.).

#### Schema Compilation
- **FR-010 (P0):** Compile schema YAML/JSON to typed Schema AST.
- **FR-011 (P0):** Validate schema itself before use (ref resolution, consistency checks).
- **FR-012 (P0):** Resolve $ref/imports into fully linked graph with cycle detection.
- **FR-013 (P0):** Produce compact IR with O(1) property lookup.
- **FR-014 (P0):** Precompile regex patterns with ReDoS protection.
- **FR-015 (P0):** Generate stable schema_hash for caching and provenance.
- **FR-016 (P1):** Cache compiled IR by (schema_id, version, profile) tuple.
- **FR-017 (P1):** CLI command: `greenlang compile-schema --schema <ref> --out schema.ir`.

#### Structural Validation
- **FR-020 (P0):** Validate required fields (including nested) with precise paths.
- **FR-021 (P0):** Validate types (string/number/integer/boolean/object/array/null).
- **FR-022 (P0):** Validate additionalProperties policy per profile.
- **FR-023 (P0):** Validate property dependencies (dependentRequired).
- **FR-024 (P1):** Detect and report unknown fields with close-match suggestions.

#### Constraint Validation
- **FR-030 (P0):** Validate numeric constraints (min/max, exclusiveMin/Max, multipleOf).
- **FR-031 (P0):** Validate string constraints (pattern, minLength/maxLength, format).
- **FR-032 (P0):** Validate array constraints (minItems/maxItems, uniqueItems, contains).
- **FR-033 (P0):** Validate enum values.
- **FR-034 (P0):** Validate const values.
- **FR-035 (P1):** Validate format types (email, uri, uuid, date, date-time, ipv4, ipv6).
- **FR-036 (P1):** Validate propertyNames pattern constraints.

#### Unit Validation
- **FR-040 (P0):** Detect missing units when schema requires x-gl-unit.
- **FR-041 (P0):** Detect incompatible units (dimension mismatch).
- **FR-042 (P0):** Validate units against allowed unit catalog.
- **FR-043 (P1):** Convert compatible units to canonical form during normalization.
- **FR-044 (P1):** Parse unit strings in `{value, unit}` or `"10 kWh"` formats.

#### Rule Engine
- **FR-050 (P0):** Evaluate cross-field rules (x-gl-rules) with when/check clauses.
- **FR-051 (P0):** Support conditional requirements (if X then Y required).
- **FR-052 (P0):** Support consistency checks (sum validation with tolerance).
- **FR-053 (P1):** Support rule packs (versioned, addressable, with dependencies).
- **FR-054 (P1):** Type-check rule expressions at compile time.

#### Composition Validation
- **FR-060 (P0):** Validate oneOf constraints (exactly one schema match).
- **FR-061 (P0):** Validate anyOf constraints (at least one schema match).
- **FR-062 (P1):** Validate allOf constraints (all schemas match).
- **FR-063 (P1):** Validate not constraints (no schema match).

#### Normalization
- **FR-070 (P0):** Apply schema defaults for missing optional fields.
- **FR-071 (P0):** Canonicalize types (safe coercions when enabled).
- **FR-072 (P1):** Canonicalize units to schema-specified canonical form.
- **FR-073 (P1):** Canonicalize keys (resolve aliases, normalize casing).
- **FR-074 (P0):** Ensure normalization is idempotent.

#### Fix Suggestions
- **FR-080 (P1):** Generate JSON Patch suggestions for safe fixes.
- **FR-081 (P1):** Include preconditions (test operations) for each patch.
- **FR-082 (P1):** Classify fixes as safe/needs_review/unsafe.
- **FR-083 (P2):** Suggest renamed field migrations when schema declares renames.
- **FR-084 (P2):** Suggest close-match corrections for unknown fields.

#### Outputs
- **FR-090 (P0):** Produce validation report with summary and findings list.
- **FR-091 (P0):** Include stable error codes and JSON Pointer paths in findings.
- **FR-092 (P0):** Sort findings deterministically (severity, path, code).
- **FR-093 (P1):** Produce normalized payload with _meta provenance.
- **FR-094 (P1):** Support batch validation with per-item results.
- **FR-095 (P1):** Support YAML and SARIF output formats.

#### API and CLI
- **FR-100 (P0):** POST /v1/validate endpoint with request/response schema.
- **FR-101 (P0):** CLI: `greenlang validate --schema <ref> --input <file>`.
- **FR-102 (P1):** SDK functions: validate(), validate_batch(), compile_schema().
- **FR-103 (P1):** Support dry-run mode (validate without normalization side effects).

---

## 8. Error Taxonomy & Machine-Fixable Hints

### 8.1 Finding structure
Every finding MUST include:
- `code`: stable identifier (e.g., `GLV-MISSING-REQUIRED`)
- `severity`: `error|warning|info`
- `path`: JSON Pointer (RFC 6901) to offending location
- `message`: human-readable explanation
- `expected`: structured expectation (type/range/unit/etc.)
- `actual`: observed value/metadata
- `hint`: optional machine hint

### 8.2 Suggested patch format
Fix suggestions SHOULD use **JSON Patch (RFC 6902)**:
- `op`: add|remove|replace|move|copy|test
- `path`: JSON Pointer
- `value`: new value (for add/replace)
- `from`: source path (for move/copy)

Additionally:
- `preconditions`: list of `test` operations that must pass
- `confidence`: 0.0-1.0
- `safety`: `safe|needs_review|unsafe`
- `rationale`: short text

### 8.3 Error codes (canonical taxonomy)

> **Error code format:** `GLSCHEMA-Exxx` where the first digit indicates category.

#### Structural Errors (E1xx)
| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLSCHEMA-E100` | MISSING_REQUIRED | Required field is missing | no |
| `GLSCHEMA-E101` | UNKNOWN_FIELD | Unknown field in strict mode | no |
| `GLSCHEMA-E102` | TYPE_MISMATCH | Value type doesn't match schema | no |
| `GLSCHEMA-E103` | INVALID_NULL | Null value not allowed | no |
| `GLSCHEMA-E104` | CONTAINER_TYPE_MISMATCH | Expected object/array, got different type | no |
| `GLSCHEMA-E105` | PROPERTY_COUNT_VIOLATION | Object has too many/few properties | no |
| `GLSCHEMA-E106` | REQUIRED_PROPERTIES_MISSING | Multiple required properties missing | no |
| `GLSCHEMA-E107` | DUPLICATE_KEY | Duplicate key after canonicalization | no |

#### Constraint Errors (E2xx)
| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLSCHEMA-E200` | RANGE_VIOLATION | Numeric value out of min/max bounds | no |
| `GLSCHEMA-E201` | PATTERN_MISMATCH | String doesn't match regex pattern | no |
| `GLSCHEMA-E202` | ENUM_VIOLATION | Value not in allowed enum set | no |
| `GLSCHEMA-E203` | LENGTH_VIOLATION | String/array length out of bounds | no |
| `GLSCHEMA-E204` | UNIQUE_VIOLATION | Array contains duplicate items | no |
| `GLSCHEMA-E205` | MULTIPLE_OF_VIOLATION | Number not multiple of constraint | no |
| `GLSCHEMA-E206` | FORMAT_VIOLATION | String doesn't match format (email, uri, etc.) | no |
| `GLSCHEMA-E207` | CONST_VIOLATION | Value doesn't match const constraint | no |
| `GLSCHEMA-E208` | CONTAINS_VIOLATION | Array doesn't contain required item | no |
| `GLSCHEMA-E209` | PROPERTY_NAME_VIOLATION | Property name doesn't match pattern | no |

#### Unit Errors (E3xx)
| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLSCHEMA-E300` | UNIT_MISSING | Required unit not provided | no |
| `GLSCHEMA-E301` | UNIT_INCOMPATIBLE | Unit dimension mismatch (e.g., kg vs kWh) | no |
| `GLSCHEMA-E302` | UNIT_NONCANONICAL | Unit not in canonical form (warning) | n/a |
| `GLSCHEMA-E303` | UNIT_UNKNOWN | Unit not in allowed catalog | no |

#### Rule Errors (E4xx)
| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLSCHEMA-E400` | RULE_VIOLATION | Cross-field rule failed | no |
| `GLSCHEMA-E401` | CONDITIONAL_REQUIRED | Conditional requirement not met | no |
| `GLSCHEMA-E402` | CONSISTENCY_ERROR | Sum/consistency check failed | no |
| `GLSCHEMA-E403` | DEPENDENCY_VIOLATION | Property dependency not satisfied | no |
| `GLSCHEMA-E405` | ONE_OF_VIOLATION | Payload doesn't match exactly one oneOf schema | no |
| `GLSCHEMA-E406` | ANY_OF_VIOLATION | Payload doesn't match any anyOf schema | no |

#### Schema Errors (E5xx)
| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLSCHEMA-E500` | REF_RESOLUTION_FAILED | $ref target not found | no |
| `GLSCHEMA-E501` | SCHEMA_DEPRECATED_FIELD | Using deprecated field (warning) | n/a |
| `GLSCHEMA-E502` | SCHEMA_RENAMED_FIELD | Field was renamed (warning with migration hint) | n/a |
| `GLSCHEMA-E503` | CYCLE_DETECTED | Circular reference in schema | no |

#### Limit Errors (E9xx)
| Code | Name | Description | Default Retry? |
|------|------|-------------|----------------|
| `GLSCHEMA-E900` | LIMIT_EXCEEDED | Input size/complexity limit exceeded | no |
| `GLSCHEMA-E901` | TIMEOUT | Validation timeout (regex, rule) | yes (bounded) |

### 8.4 Fix safety levels
Fix suggestions MUST be labeled:

- **safe**: mechanically correct with high confidence and strong preconditions (e.g., add missing optional field with schema default)
- **needs_review**: likely correct but depends on meaning or context (e.g., unit conversion from user-provided unit string)
- **unsafe**: speculative; only produce if explicitly requested (e.g., infer a missing required value)

Examples:
- Missing required field with **no default** -> `unsafe` unless rule pack provides a deterministic derivation.
- Renamed field where old->new mapping is declared in schema -> `safe` (with `test` precondition that old exists and new missing).
- Type mismatch `"123"` for integer -> `safe` only if `coercion_policy` allows and parsing is exact.

### 8.5 Finding ordering (determinism)
To ensure stable diffs and reproducibility, findings MUST be sorted by:
1. `severity` (error > warning > info)
2. `path` (lexicographic JSON Pointer)
3. `code`
4. stable tie-breaker (e.g., schema node id)

---

## 9. Normalization Specification

Normalization MUST be deterministic and configurable:
- **Defaults**: apply schema defaults for missing optional fields (configurable)
- **Type canonicalization**:
  - `"42"` -> `42` for integer fields if `coercion_policy=safe`
  - `"true"` -> `true` for boolean fields if safe
- **Unit canonicalization**:
  - convert to canonical unit (e.g., `Wh` -> `kWh`) if allowed
  - store original unit optionally in metadata (`_meta.original_unit`)
- **Key canonicalization**:
  - resolve known aliases (schema `aliases` map)
  - normalize casing if schema demands (e.g., snake_case)
- **Ordering**: stable key ordering for reproducible output (where applicable)

Normalization MUST NOT:
- invent values for required fields (except where schema provides explicit defaults and policy allows)
- silently drop fields in strict mode (must at least warn)

### 9.1 Canonical payload representation
Normalized payloads MUST follow a canonical representation to minimize downstream friction:

- All object keys follow schema canonical keys (after alias resolution).
- Unit-bearing values use one consistent structure per schema (prefer object form `{value, unit}`).
- Optional metadata is stored under a reserved key, e.g. `_meta`, and MUST NOT collide with schema domain keys.

Recommended `_meta` keys:
- `_meta.schema_ref`: schema_id + version used for normalization
- `_meta.original`: optional snapshot of original raw input (disabled by default for privacy)
- `_meta.conversions`: list of unit conversions applied

---

## 10. Interfaces

### 13.1 CLI (developer tooling)
Example:
```bash
greenlang validate \
  --schema gl://schemas/emissions/activity@1.3.0 \
  --profile strict \
  --input payload.yaml \
  --output report.json \
  --normalized normalized.json \
  --patches fixes.json
```

### 13.2 API (service mode)
`POST /v1/validate`

Request:
```json
{
  "schema_ref": {"schema_id":"...", "version":"1.3.0"},
  "payload": { "...": "..." },
  "validation_profile": "standard",
  "rules": ["governance/base", "climate/v1"],
  "options": {"normalize": true, "emit_patches": true, "max_errors": 100}
}
```

Response:
```json
{
  "valid": false,
  "schema_ref": {"schema_id":"...", "version":"1.3.0"},
  "summary": {"errors": 2, "warnings": 1, "timings_ms": {"total": 12}},
  "findings": [ ... ],
  "normalized_payload": { ... },
  "fix_suggestions": [ ... ]
}
```

### 11.3 SDK
- `validate(payload, schemaRef, options) -> {report, normalized, fixes}`
- Language targets: TypeScript, Python (initial); others later.

### 11.4 Output formats
In addition to JSON responses, the agent SHOULD support:
- YAML report output (human-friendly)
- SARIF report output (for CI/IDE integration)
- Text summary (CLI) with colorized output and compact diff-like hints

### 11.5 Batch validation
The agent SHOULD support validating an array of payloads in a single request with:
- Per-item findings and patches
- Shared schema compilation/caching
- Partial failures without aborting the entire batch (unless `fail_fast=true`)

---

## 11. Architecture (Reference)

### 13.1 Components
- **Parser**: safe YAML/JSON parsing with limits
- **Schema Resolver**: loads schema by SchemaRef (registry adapter)
- **Schema Compiler**: parses schema -> AST -> IR; validates schema itself
- **Validator Core**: structural + constraint validation
- **Unit Engine**: unit parsing, dimensional analysis, conversion
- **Rule Engine**: evaluates cross-field rules against normalized view
- **Linter**: non-fatal best-practice checks
- **Fix Suggestion Engine**: generates patches and preconditions
- **Renderer**: produces report formats (JSON, SARIF optional)

### 13.2 Caching strategy
- Cache compiled schema IR by `(schema_id, version, profile, rulepack_hash)`
- LRU with size and TTL; metrics for hit rate

### 11.3 Determinism & reproducibility
- Stable ordering of findings
- Stable normalization output
- Same input -> same output given same schema + options

### 11.4 Schema resolver abstraction
The validator MUST not hardcode schema storage. Define a simple resolver interface:

- `resolve(schema_id, version) -> {schema_text, content_type, etag/hash}`
- `list_versions(schema_id) -> [versions]` (optional)
- `get_latest_compatible(schema_id, constraint) -> version` (optional)

Resolver backends can include:
- Local filesystem (dev)
- Git-backed registry
- HTTP registry service

### 11.5 Rule pack loading
Rule packs MUST be versioned and addressable:
- `governance/base@1.0.0`
- `climate/v1@2026-01`

Rule packs MUST declare:
- Supported schema ids/versions (constraints)
- Rule severity (error/warn/info)
- Dependencies on other packs
- Optional unit catalogs/extensions

### 11.6 Data Model (Core Entities)

| Entity | Description | Key Fields |
|--------|-------------|------------|
| **SchemaDefinition** | Immutable schema source (YAML/JSON) + metadata | schema_id, version, content_hash, source_text |
| **SchemaIR** | Compiled intermediate representation | ir_version, schema_hash, properties_map, constraints, rules |
| **ValidationRequest** | Incoming validation request | request_id, payload, schema_ref, profile, options |
| **ValidationReport** | Validation result | request_id, valid, summary, findings[], timings |
| **Finding** | Single validation error/warning | code, severity, path, message, expected, actual, hint |
| **NormalizedPayload** | Canonicalized output payload | payload, _meta (schema_ref, conversions) |
| **FixSuggestion** | Machine-applyable repair | patch[], preconditions[], confidence, safety, rationale |
| **RulePack** | Versioned rule set | pack_id, version, rules[], dependencies[] |

### 11.7 Validation State Machine

```
Request → PARSING → RESOLVING → COMPILING → VALIDATING → NORMALIZING → COMPLETE
                                    ↓              ↓             ↓
                                 FAILED         FAILED        FAILED
```

**States:**
- `PARSING`: Parsing input YAML/JSON safely
- `RESOLVING`: Resolving schema from registry
- `COMPILING`: Compiling schema to IR (may be cached)
- `VALIDATING`: Running structural, constraint, unit, and rule validation
- `NORMALIZING`: Applying normalization transformations
- `COMPLETE`: Emitting final report
- `FAILED`: Error occurred; partial findings may be available

---

## 12. Non-Functional Requirements

> **Priority legend:** P0 = must-have (MVP/GA critical), P1 = should-have, P2 = nice-to-have.

### 12.1 Reliability and Correctness
- **NFR-001 (P0):** No panics/uncaught exceptions on malformed input.
- **NFR-002 (P0):** Graceful degradation: partial findings up to `max_errors`.
- **NFR-003 (P0):** Validation must be deterministic (same input → same output).
- **NFR-004 (P0):** Normalization must be idempotent (normalize(normalize(x)) == normalize(x)).

### 12.2 Performance and Scale
- **NFR-010 (P0):** P95 validation latency for small payloads (<50KB): < 25 ms.
- **NFR-011 (P0):** P95 validation latency for medium payloads (<500KB): < 150 ms.
- **NFR-012 (P1):** P99 latency for <50KB payloads: < 60 ms.
- **NFR-013 (P0):** Schema IR cache hit rate > 80% under steady traffic.
- **NFR-014 (P1):** Support batch validation with predictable memory use.
- **NFR-015 (P1):** Schema compilation latency < 500 ms for typical schemas.

### 12.3 Security
- **NFR-020 (P0):** Safe YAML parsing (no object instantiation, no arbitrary code execution).
- **NFR-021 (P0):** Input size/structure limits enforced (depth, array length, ref expansions).
- **NFR-022 (P0):** ReDoS protection: regex patterns with timeout and complexity limits.
- **NFR-023 (P0):** Schema bomb protection: limit $ref expansion depth and count.
- **NFR-024 (P0):** Rules are declarative only; never execute user-provided code.
- **NFR-025 (P1):** Path traversal protection in schema resolver.
- **NFR-026 (P1):** No sensitive data in logs or error messages.

### 12.4 Observability
- **NFR-030 (P0):** Structured logs with request_id, schema_ref, timings.
- **NFR-031 (P0):** Metrics: validations_total, validations_failed_total, findings_by_code.
- **NFR-032 (P0):** Metrics: schema_cache_hit_rate, unit_conversion_count.
- **NFR-033 (P1):** Tracing spans: parse, resolve, compile, validate, normalize, suggest.
- **NFR-034 (P1):** Alert hooks for sustained validation failures.

### 12.5 Operability
- **NFR-040 (P0):** Health endpoints: /health, /ready, /live.
- **NFR-041 (P0):** Clear error codes with recommended remediation actions.
- **NFR-042 (P1):** Schema cache invalidation API.
- **NFR-043 (P1):** Graceful shutdown with request draining.

### 12.6 SLOs
Service mode SLO targets:
- **NFR-050 (P0):** Availability: 99.9% monthly (if run as shared service).
- **NFR-051 (P1):** Error budget policy: (TBD based on deployment model).
- **NFR-052 (P1):** Mean time to recovery (MTTR): < 5 minutes.

### 12.7 Resource Constraints
- **NFR-060 (P0):** Memory: validation should be streaming-friendly; avoid full deep copies.
- **NFR-061 (P0):** CPU: regex and rule evaluation must have quotas.
- **NFR-062 (P0):** Concurrency: thread-safe schema cache; no global mutable state.

---

## 13. Testing Strategy

- **Unit tests**:
  - schema parser/compiler
  - validator for each constraint type
  - unit conversion matrix tests
  - rule engine semantics tests
- **Golden tests**:
  - input payload + schema -> expected report/normalized/patches
- **Fuzzing**:
  - payload fuzzer for parser + validator
  - schema fuzzer for compiler + resolver
- **Property-based tests**:
  - normalization is idempotent (normalize(normalize(x)) == normalize(x))
  - patch application leads to fewer errors (monotonic improvement) for safe patches

### 13.1 Conformance suite (highly recommended)
Create a shared test corpus:
- `schemas/` (representative schema set + edge cases)
- `payloads/valid/` and `payloads/invalid/`
- `expected/` reports + normalized payloads + patches

This suite becomes the contract for:
- SDKs
- alternative implementations
- future optimizations (must not change outputs unexpectedly)

### 13.2 Security testing
- YAML parser hardening tests (anchors, aliases, billion laughs)
- ReDoS tests for regex patterns
- Schema bomb tests for deep `$ref` chains
- Fuzz tests for rule engine input

---

## 14. Rollout Plan

1. **MVP (v0.1)**: structural validation + required/type/range, report format, CLI.
2. **v0.2**: schema IR compilation + caching; basic normalization.
3. **v0.3**: unit engine + unit validation + conversions.
4. **v0.4**: rule engine + rule packs; linting.
5. **v1.0**: fix suggestions with JSON Patch, SDKs, production hardening.

---

## 15. Success Metrics

- Adoption: % of GreenLang ingestion points using GL-FOUND-X-002
- Defect reduction: decrease in downstream validation bugs/incidents
- Developer experience:
  - median time-to-fix invalid payloads
  - user-rated clarity of diagnostics
- Performance:
  - cache hit rate > 80% under steady traffic
  - P95 latency within targets

---

## 16. Acceptance Criteria (P0)

The following criteria must be satisfied for GA readiness:

1. **Determinism:** Given the same payload + schema + options, validation report hash is identical across runs.
2. **Error Detection:** All GLSCHEMA-E1xx and E2xx error codes have test coverage with golden tests passing.
3. **Unit Validation:** Validates unit presence, compatibility, and catalog membership (E3xx errors implemented).
4. **Rule Engine:** Cross-field rules evaluate correctly with when/check clauses (E4xx errors implemented).
5. **Normalization Idempotency:** normalize(normalize(x)) == normalize(x) for all valid payloads.
6. **Fix Suggestions:** Safe fixes have preconditions and confidence scores; can be applied via JSON Patch.
7. **Performance:** P95 latency < 25ms for payloads < 50KB with warm schema cache.
8. **Security:** No YAML deserialization vulnerabilities; ReDoS protection verified via fuzz testing.

---

## 17. Risks & Mitigations

- **Ambiguous fixes** -> label as `needs_review`; require strong preconditions.
- **Schema evolution churn** -> enforce semver/compat rules; deprecation windows.
- **Unit library correctness** -> curated unit set, extensive tests, explicit dimensions.
- **Rule complexity** -> cap expression depth/size; provide linting for rules.

---

## 18. Security Model

### 18.1 Input Sanitization
- All YAML/JSON parsing uses safe parsers (no object instantiation, no arbitrary code execution).
- Input size limits enforced before parsing begins.
- Character encoding validated (UTF-8 only).

### 18.2 Schema Bomb Protection
- Maximum $ref expansion depth: 50 (configurable).
- Maximum total $ref expansions: 10,000.
- Cycle detection with clear error traces.

### 18.3 ReDoS Protection
- Regex patterns analyzed for catastrophic backtracking potential.
- Pattern execution timeout: 100ms default.
- Pattern complexity limits enforced at compile time.
- Dangerous patterns flagged with GLSCHEMA-E901.

### 18.4 Rule Execution Sandbox
- Rules are declarative expressions only (JSONLogic/CEL-style).
- No user-provided code execution.
- Expression depth limits enforced.
- No external network calls from rule evaluation.

### 18.5 Secrets Handling
- Schema resolver may require authentication for private registries.
- Credentials passed via environment or secrets manager.
- No secrets logged or included in error messages.

---

## 19. Operational Runbook

### 19.1 Common Failure Modes

| Failure | Symptoms | Remediation |
|---------|----------|-------------|
| Schema cache miss spike | Increased latency, high CPU | Check registry availability, increase cache size |
| ReDoS timeout surge | GLSCHEMA-E901 errors, blocked threads | Identify offending patterns, add to blocklist |
| Schema bomb detected | GLSCHEMA-E503 errors, memory pressure | Reject malicious schema, alert security |
| Validation backlog | Queue growth, increased p99 latency | Scale workers, enable rate limiting |

### 19.2 On-Call Actions
- **Cache Invalidation:** `POST /admin/cache/invalidate?schema_id=<id>&version=<ver>`
- **Emergency Pattern Blocklist:** Add pattern to `config/blocked_patterns.yaml`, reload.
- **Rate Limiting:** Enable via `config/rate_limits.yaml` for specific schema_refs.
- **Circuit Breaker:** Disable problematic rule pack via feature flag.

### 19.3 Health Check Endpoints
- `GET /health` - Basic liveness check
- `GET /ready` - Readiness including cache warm status
- `GET /metrics` - Prometheus-format metrics

---

## 20. Detailed API Examples

### 20.1 Validate Request
```json
{
  "schema_ref": {
    "schema_id": "gl://schemas/emissions/activity",
    "version": "1.3.0"
  },
  "payload": {
    "activity_id": "ACT-2026-001",
    "energy_consumption": {"value": 1500, "unit": "kWh"},
    "reporting_period": "2026-Q1"
  },
  "validation_profile": "standard",
  "options": {
    "normalize": true,
    "emit_patches": true,
    "max_errors": 50
  }
}
```

### 20.2 Validate Response (Success)
```json
{
  "valid": true,
  "schema_ref": {"schema_id": "gl://schemas/emissions/activity", "version": "1.3.0"},
  "summary": {
    "errors": 0,
    "warnings": 1,
    "timings_ms": {"parse": 2, "resolve": 5, "compile": 0, "validate": 8, "normalize": 3, "total": 18}
  },
  "findings": [
    {
      "code": "GLSCHEMA-E501",
      "severity": "warning",
      "path": "/reporting_period",
      "message": "Field 'reporting_period' is deprecated; use 'period' instead"
    }
  ],
  "normalized_payload": {
    "activity_id": "ACT-2026-001",
    "energy_consumption": {"value": 1500, "unit": "kWh"},
    "period": "2026-Q1",
    "_meta": {"schema_ref": "gl://schemas/emissions/activity@1.3.0"}
  },
  "fix_suggestions": []
}
```

### 20.3 Validate Response (Failure)
```json
{
  "valid": false,
  "schema_ref": {"schema_id": "gl://schemas/emissions/activity", "version": "1.3.0"},
  "summary": {"errors": 2, "warnings": 0, "timings_ms": {"total": 12}},
  "findings": [
    {
      "code": "GLSCHEMA-E100",
      "severity": "error",
      "path": "/activity_id",
      "message": "Required field 'activity_id' is missing",
      "expected": {"type": "string", "required": true},
      "actual": null
    },
    {
      "code": "GLSCHEMA-E301",
      "severity": "error",
      "path": "/energy_consumption",
      "message": "Unit 'kg' is incompatible with dimension 'energy'",
      "expected": {"dimension": "energy", "allowed": ["kWh", "MWh", "GJ"]},
      "actual": {"value": 100, "unit": "kg"}
    }
  ],
  "fix_suggestions": [
    {
      "patch": [
        {"op": "add", "path": "/activity_id", "value": ""}
      ],
      "confidence": 0.0,
      "safety": "unsafe",
      "rationale": "Required field has no default; value must be provided by user"
    }
  ]
}
```

### 20.4 Batch Validate Request
```json
{
  "schema_ref": {"schema_id": "gl://schemas/emissions/activity", "version": "1.3.0"},
  "payloads": [
    {"activity_id": "ACT-001", "energy_consumption": {"value": 100, "unit": "kWh"}},
    {"activity_id": "ACT-002", "energy_consumption": {"value": 200, "unit": "kWh"}}
  ],
  "validation_profile": "standard"
}
```

---

## 21. Open Questions

1. What is the authoritative GreenLang schema grammar today (JSON Schema-compatible or custom)?
2. Where are schemas stored (registry design) and what are auth requirements?
3. Canonical unit system: strict SI, domain-specific, or per-schema configurable?
4. Preferred rule DSL (CEL vs JSONLogic vs custom)?
5. Do we need SARIF output for IDE/tooling integration?

---

## Appendix A: Example Finding + Patch

Example finding:
```json
{
  "code": "GLSCHEMA-E301",
  "severity": "error",
  "path": "/energy_consumption",
  "message": "Unit 'kg' is incompatible with expected dimension 'energy'.",
  "expected": {"dimension":"energy","unit":"kWh"},
  "actual": {"value": 10, "unit": "kg"},
  "hint": {
    "category": "unit",
    "suggested_units": ["kWh", "MWh"],
    "docs": "gl://docs/units#energy"
  }
}
```

Example fix suggestion (needs_review):
```json
{
  "patch": [
    {"op":"test","path":"/energy_consumption/unit","value":"Wh"},
    {"op":"replace","path":"/energy_consumption/unit","value":"kWh"},
    {"op":"replace","path":"/energy_consumption/value","value":0.01}
  ],
  "confidence": 0.72,
  "safety": "needs_review",
  "rationale": "Convert Wh to canonical kWh."
}
```

---

## Appendix B: Minimal schema example (illustrative)

```yaml
schema_id: emissions/activity
version: 1.3.0
type: object
required: [activity_id, energy_consumption]
properties:
  activity_id:
    type: string
    pattern: "^[A-Z0-9_-]{8,32}$"
  energy_consumption:
    type: object
    required: [value, unit]
    properties:
      value:
        type: number
        minimum: 0
      unit:
        type: string
        enum: [Wh, kWh, MWh]
    unit:
      dimension: energy
      canonical: kWh
```

---

## Appendix C: Patch generation heuristics (initial)

1. **Rename field** when schema provides `renamed_from` mapping.
2. **Add optional defaults** when default exists and field missing.
3. **Coerce safe primitives** (string->number/bool) with exact parsing only.
4. **Unit conversion** only when unit is in allowed catalog and dimension matches.
5. **Close-match unknown keys** (edit distance) only as `needs_review` unless schema declares aliases.

---

## Appendix D: Example rule pack snippet (illustrative)

```yaml
pack: climate/v1
version: 2026-01
rules:
  - id: CLIM-001
    severity: error
    when: { eq: [ { var: "/fuel_type" }, "gas" ] }
    then:
      require: ["/methane_slip"]
      message: "methane_slip is required when fuel_type=gas"
  - id: CLIM-002
    severity: warn
    check:
      lte: [ { var: "/temperature" }, 120 ]
    message: "temperature unusually high; check sensor calibration"
```
