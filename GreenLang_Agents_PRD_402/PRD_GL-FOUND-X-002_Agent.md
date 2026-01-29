# PRD: GreenLang Schema Compiler & Validator (GL-FOUND-X-002)

**Agent name:** GreenLang Schema Compiler & Validator  
**Agent ID:** GL-FOUND-X-002  
**Family:** SchemaFamily (Foundation & Governance Layer)  
**Priority:** P0 (cross-cutting; blocks downstream agents)  
**Status:** Draft PRD  
**Last updated:** 2026-01-28  
**Owner:** (TBD)  
**Reviewers:** Platform Eng, Schema Governance, Domain Leads, Security

---

## 1. Context / Problem Statement

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

## 2. Goals / Objectives

### 2.1 Primary goals (MVP -> v1)
1. **Deterministic schema validation** for JSON/YAML payloads against a schema reference (id + version).
2. **High-signal diagnostics**:
   - Precise field paths
   - Expected vs actual values/types
   - Unit/range specifics
3. **Normalization** to canonical representation (types, units, defaults, key canonicalization).
4. **Machine-fixable hints**: emit patches (JSON Patch preferred) with safety constraints and confidence.

### 2.2 Secondary goals (v1+)
- Schema compilation to an intermediate representation (IR) for fast repeated validations
- Rule-pack system (pluggable governance rules across domains)
- Linting and deprecation guidance (schema evolution governance)
- Compatibility mode for schema migrations (suggest updated field names, etc.)

### 2.3 Non-goals
- Persisting payloads or owning a database of customer data
- Being the source of truth for schema authoring UI (it can validate schemas, but does not replace schema editor tooling)
- Automated execution of fixes in upstream systems (it only suggests)

---

## 3. Users, Personas, and Stakeholders

### Personas
- **Schema Authors / Governance**: define schemas/contracts; need strict, explainable failures.
- **Platform Engineers**: integrate validation into APIs, pipelines, agents.
- **Domain Teams**: build domain agents that depend on consistent payloads.
- **Data Producers**: send payloads; want actionable errors and auto-fix suggestions.

### Stakeholders
- SchemaFamily owners, platform runtime, security, domain leads, DX team.

---

## 4. Assumptions & Definitions

### 4.1 GreenLang schema (assumed baseline)
A GreenLang schema is represented in YAML/JSON and supports:
- Primitive and composite types: string/number/integer/boolean/object/array
- Required vs optional fields
- Defaults
- Constraints: min/max, pattern, enum, minItems/maxItems, etc.
- Units metadata for measurable quantities (e.g., `unit: kWh`, `dimension: energy`)
- References and reuse: `$ref`-like or `imports` semantics
- Versioning: semver-ish (`1.2.3`) or date-based (`2026-01`)

> Note: This PRD defines interfaces that work whether GreenLang is a superset of JSON Schema or a custom schema language. Implementation details can adapt.

### 4.2 Key terms
- **SchemaRef**: `{schema_id, version}` (and optionally `variant/profile`)
- **Normalization**: Transforming payload into canonical form without changing semantics
- **Fix suggestion**: A machine-applyable patch + rationale + preconditions

---

## 5. High-Level Workflow

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

## 6. Functional Requirements

### 6.1 Inputs
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

### 6.2 Outputs
The agent MUST produce:

1. **Validation report** (machine-readable JSON):
   - Summary: valid/invalid, error counts, warnings, schema used, timings
   - Findings list with stable error codes and field paths
2. **Normalized payload** (if enabled):
   - Canonical keys, types, units, defaults applied, metadata preserved per policy
3. **Fix suggestions** (if enabled):
   - Ordered list of patches (JSON Patch operations) with preconditions

### 6.3 Schema validation capabilities
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

### 6.4 Unit validation & conversion
The validator MUST:
- Detect missing units when schema requires units
- Detect incompatible units (e.g., kg vs kWh)
- Convert compatible units to canonical units during normalization (if allowed)
- Handle unit annotations in common shapes:
  - `{ "value": 10, "unit": "kWh" }`
  - `"10 kWh"` (string parsing as optional feature)
  - separate unit field (`energy_value`, `energy_unit`) if schema declares pairing

### 6.5 Cross-field rule engine
The agent MUST support governance rules such as:
- Conditional requirements: if `fuel_type=gas` then `methane_slip` required
- Ranges dependent on context: `temperature` range depends on `sensor_model`
- Consistency: sum of components equals total within tolerance
- Domain packs: e.g., `climate/v1`, `finance/v2` (extensible)

Rule format options (implementation choice):
- JSONLogic-like expressions
- CEL (Common Expression Language)-style
- Custom DSL that compiles to an AST

### 6.6 Linting
Linting is non-blocking by default (warnings) and includes:
- Unknown fields / suspicious keys (typos, close matches)
- Deprecated fields usage
- Non-canonical casing/keys
- Unit formatting
- Schema anti-patterns (if validating schemas too)

### 6.7 Schema compilation (compiler) requirements
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

### 6.8 Schema self-validation (governance)
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

### 6.9 Validation profile policy matrix
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

### 6.10 Limits and guardrails (must be configurable)
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
- `GLV-LIMIT-EXCEEDED` with details about which limit was hit.

---

## 7. Error Taxonomy & Machine-Fixable Hints

### 7.1 Finding structure
Every finding MUST include:
- `code`: stable identifier (e.g., `GLV-MISSING-REQUIRED`)
- `severity`: `error|warning|info`
- `path`: JSON Pointer (RFC 6901) to offending location
- `message`: human-readable explanation
- `expected`: structured expectation (type/range/unit/etc.)
- `actual`: observed value/metadata
- `hint`: optional machine hint

### 7.2 Suggested patch format
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

### 7.3 Error codes (initial set)
- `GLV-MISSING-REQUIRED`
- `GLV-UNKNOWN-FIELD`
- `GLV-TYPE-MISMATCH`
- `GLV-RANGE-VIOLATION`
- `GLV-PATTERN-MISMATCH`
- `GLV-ENUM-VIOLATION`
- `GLV-UNIT-MISSING`
- `GLV-UNIT-INCOMPATIBLE`
- `GLV-UNIT-NONCANONICAL`
- `GLV-RULE-VIOLATION`
- `GLV-REF-RESOLUTION-FAILED`
- `GLV-SCHEMA-DEPRECATED-FIELD`
- `GLV-SCHEMA-RENAMED-FIELD`

### 7.4 Fix safety levels
Fix suggestions MUST be labeled:

- **safe**: mechanically correct with high confidence and strong preconditions (e.g., add missing optional field with schema default)
- **needs_review**: likely correct but depends on meaning or context (e.g., unit conversion from user-provided unit string)
- **unsafe**: speculative; only produce if explicitly requested (e.g., infer a missing required value)

Examples:
- Missing required field with **no default** -> `unsafe` unless rule pack provides a deterministic derivation.
- Renamed field where old->new mapping is declared in schema -> `safe` (with `test` precondition that old exists and new missing).
- Type mismatch `"123"` for integer -> `safe` only if `coercion_policy` allows and parsing is exact.

### 7.5 Finding ordering (determinism)
To ensure stable diffs and reproducibility, findings MUST be sorted by:
1. `severity` (error > warning > info)
2. `path` (lexicographic JSON Pointer)
3. `code`
4. stable tie-breaker (e.g., schema node id)

---

## 8. Normalization Specification

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

### 8.1 Canonical payload representation
Normalized payloads MUST follow a canonical representation to minimize downstream friction:

- All object keys follow schema canonical keys (after alias resolution).
- Unit-bearing values use one consistent structure per schema (prefer object form `{value, unit}`).
- Optional metadata is stored under a reserved key, e.g. `_meta`, and MUST NOT collide with schema domain keys.

Recommended `_meta` keys:
- `_meta.schema_ref`: schema_id + version used for normalization
- `_meta.original`: optional snapshot of original raw input (disabled by default for privacy)
- `_meta.conversions`: list of unit conversions applied

---

## 9. Interfaces

### 9.1 CLI (developer tooling)
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

### 9.2 API (service mode)
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

### 9.3 SDK
- `validate(payload, schemaRef, options) -> {report, normalized, fixes}`
- Language targets: TypeScript, Python (initial); others later.

### 9.4 Output formats
In addition to JSON responses, the agent SHOULD support:
- YAML report output (human-friendly)
- SARIF report output (for CI/IDE integration)
- Text summary (CLI) with colorized output and compact diff-like hints

### 9.5 Batch validation
The agent SHOULD support validating an array of payloads in a single request with:
- Per-item findings and patches
- Shared schema compilation/caching
- Partial failures without aborting the entire batch (unless `fail_fast=true`)

---

## 10. Architecture (Reference)

### 10.1 Components
- **Parser**: safe YAML/JSON parsing with limits
- **Schema Resolver**: loads schema by SchemaRef (registry adapter)
- **Schema Compiler**: parses schema -> AST -> IR; validates schema itself
- **Validator Core**: structural + constraint validation
- **Unit Engine**: unit parsing, dimensional analysis, conversion
- **Rule Engine**: evaluates cross-field rules against normalized view
- **Linter**: non-fatal best-practice checks
- **Fix Suggestion Engine**: generates patches and preconditions
- **Renderer**: produces report formats (JSON, SARIF optional)

### 10.2 Caching strategy
- Cache compiled schema IR by `(schema_id, version, profile, rulepack_hash)`
- LRU with size and TTL; metrics for hit rate

### 10.3 Determinism & reproducibility
- Stable ordering of findings
- Stable normalization output
- Same input -> same output given same schema + options

### 10.4 Schema resolver abstraction
The validator MUST not hardcode schema storage. Define a simple resolver interface:

- `resolve(schema_id, version) -> {schema_text, content_type, etag/hash}`
- `list_versions(schema_id) -> [versions]` (optional)
- `get_latest_compatible(schema_id, constraint) -> version` (optional)

Resolver backends can include:
- Local filesystem (dev)
- Git-backed registry
- HTTP registry service

### 10.5 Rule pack loading
Rule packs MUST be versioned and addressable:
- `governance/base@1.0.0`
- `climate/v1@2026-01`

Rule packs MUST declare:
- Supported schema ids/versions (constraints)
- Rule severity (error/warn/info)
- Dependencies on other packs
- Optional unit catalogs/extensions

---

## 11. Non-Functional Requirements

### Performance
- P95 validation latency targets (service mode):
  - Small payload (<50KB): < 25 ms
  - Medium payload (<500KB): < 150 ms
- Must support batch validation (optional) with predictable memory use.

### Reliability
- No panics/uncaught exceptions on malformed input
- Graceful degradation: partial findings up to `max_errors`

### Security
- Safe YAML parsing (no object instantiation)
- Input size/structure limits (depth, array length)
- Protect against schema bombs (deep refs), regex DoS, and path traversal in schema resolver
- Never execute user-provided code in rules; rules are declarative

### Observability
- Structured logs with request_id, schema_ref, timings
- Metrics:
  - validations_total, validations_failed_total
  - findings_by_code
  - schema_cache_hit_rate
  - unit_conversion_count
- Tracing spans: parse, resolve, compile, validate, normalize, suggest

### 11.1 SLOs
Service mode SLO targets:
- Availability: 99.9% monthly (if run as shared service)
- Error budget policy: (TBD)
- P99 latency for <50KB payloads: < 60 ms

### 11.2 Resource constraints
- Memory: validation should be streaming-friendly; avoid full deep copies when possible
- CPU: regex and rule evaluation must have quotas
- Concurrency: thread-safe schema cache; no global mutable state in validators

---

## 12. Testing Strategy

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

### 12.1 Conformance suite (highly recommended)
Create a shared test corpus:
- `schemas/` (representative schema set + edge cases)
- `payloads/valid/` and `payloads/invalid/`
- `expected/` reports + normalized payloads + patches

This suite becomes the contract for:
- SDKs
- alternative implementations
- future optimizations (must not change outputs unexpectedly)

### 12.2 Security testing
- YAML parser hardening tests (anchors, aliases, billion laughs)
- ReDoS tests for regex patterns
- Schema bomb tests for deep `$ref` chains
- Fuzz tests for rule engine input

---

## 13. Rollout Plan

1. **MVP (v0.1)**: structural validation + required/type/range, report format, CLI.
2. **v0.2**: schema IR compilation + caching; basic normalization.
3. **v0.3**: unit engine + unit validation + conversions.
4. **v0.4**: rule engine + rule packs; linting.
5. **v1.0**: fix suggestions with JSON Patch, SDKs, production hardening.

---

## 14. Success Metrics

- Adoption: % of GreenLang ingestion points using GL-FOUND-X-002
- Defect reduction: decrease in downstream validation bugs/incidents
- Developer experience:
  - median time-to-fix invalid payloads
  - user-rated clarity of diagnostics
- Performance:
  - cache hit rate > 80% under steady traffic
  - P95 latency within targets

---

## 15. Risks & Mitigations

- **Ambiguous fixes** -> label as `needs_review`; require strong preconditions.
- **Schema evolution churn** -> enforce semver/compat rules; deprecation windows.
- **Unit library correctness** -> curated unit set, extensive tests, explicit dimensions.
- **Rule complexity** -> cap expression depth/size; provide linting for rules.

---

## 16. Open Questions

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
  "code": "GLV-UNIT-INCOMPATIBLE",
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
