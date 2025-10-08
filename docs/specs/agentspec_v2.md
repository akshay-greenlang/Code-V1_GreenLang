# GreenLang AgentSpec v2 - Comprehensive Documentation

**Version:** 2.0.0
**Status:** Production-Ready
**Last Updated:** October 2025
**Specification:** FRMW-201

---

## Table of Contents

1. [Overview](#overview)
2. [Schema Sections](#schema-sections)
3. [Field Reference](#field-reference)
4. [Validation Rules](#validation-rules)
5. [Error Codes Reference](#error-codes-reference)
6. [Examples](#examples)
7. [CLI Usage](#cli-usage)
8. [Python API](#python-api)
9. [Migration Guide](#migration-guide)
10. [Best Practices](#best-practices)
11. [FAQ](#faq)
12. [References](#references)

---

## 1. Overview

### What is AgentSpec v2?

AgentSpec v2 is the authoritative schema for defining GreenLang agent packs. It provides a declarative, type-safe manifest format for specifying:

- **Computational logic**: Entrypoints, inputs/outputs, emission factors
- **AI capabilities**: LLM configuration, tools, RAG, budget constraints
- **Realtime data**: Replay/live modes, connector configuration
- **Provenance**: Factor pinning, reproducibility, audit trails
- **Security**: Network egress control, safe tool enforcement
- **Testing**: Golden tests and property-based invariants

AgentSpec v2 manifests are written in YAML or JSON and validated against a comprehensive Pydantic schema.

### Why was it created?

AgentSpec v2 addresses critical requirements for production climate accounting:

1. **AI-Native**: First-class support for LLM-powered agents with budget controls, tool calling, and RAG
2. **Deterministic**: Reproducible calculations with factor pinning and provenance tracking
3. **Climate-Specific**: Built-in validation for emissions units (kgCO2e, tCO2e), GWP sets (AR6, AR5), and emission factor URIs
4. **Type-Safe**: Comprehensive Pydantic models with 104 test cases covering all validation scenarios
5. **Auditable**: Full provenance tracking for compliance with GHG Protocol and ISO 14064-1

### Key Design Principles

- **Pydantic as Source of Truth**: JSON Schema is generated from Pydantic models, not vice versa
- **Strict Validation**: `extra='forbid'` catches typos and unsupported fields early
- **Determinism by Default**: `compute.deterministic=true` enforces reproducibility
- **Unit-Aware**: All physical quantities validated against climate units whitelist
- **Security-Conscious**: Safe tool enforcement via AST analysis, URI validation, network egress control

### Who Should Use This?

AgentSpec v2 is designed for:

- **Climate Scientists**: Defining emission factor calculations with scientific rigor
- **Software Engineers**: Building production-grade climate accounting systems
- **Auditors**: Verifying reproducibility and compliance with GHG Protocol
- **DevOps Teams**: Deploying agents with predictable resource constraints
- **AI Engineers**: Integrating LLMs safely into climate calculations

---

## 2. Schema Sections

### 2.1 Metadata Section

The metadata section provides human-readable identification and categorization.

#### Purpose
- Unique identification of agent packs
- Semantic versioning for dependency management
- Search and discovery via tags
- License and ownership tracking

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | Yes | MUST be `"2.0.0"` (literal) |
| `id` | string | Yes | Agent ID slug (e.g., `"buildings/boiler_ng_v1"`) |
| `name` | string | Yes | Human-readable name (min 3 chars) |
| `version` | string | Yes | Semantic version (e.g., `"2.1.3"`) |
| `summary` | string | No | Short description of agent purpose |
| `tags` | list[string] | No | Tags for categorization (default: `[]`) |
| `owners` | list[string] | No | Owner identifiers (e.g., `["@gl/climate-science"]`) |
| `license` | string | No | SPDX license identifier (e.g., `"Apache-2.0"`) |

#### Validation Rules

- **schema_version**: MUST be exactly `"2.0.0"` (enforced via Pydantic `Literal`)
- **id**: MUST match slug pattern `^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$`
  - Lowercase alphanumeric only
  - Separators: `/`, `-`, `_`
  - Must contain at least one `/` (hierarchical)
  - Example: `"buildings/boiler_ng_v1"`, `"transport/ev_charging/level2"`
- **version**: MUST conform to Semantic Versioning 2.0.0
  - Format: `MAJOR.MINOR.PATCH[-prerelease][+build]`
  - Example: `"2.1.3"`, `"1.0.0-rc.1"`, `"3.2.1+20241006"`
- **tags**: MUST be unique (no duplicates)
- **name**: MUST be at least 3 characters

#### Example

```yaml
schema_version: "2.0.0"
id: "buildings/boiler_ng_v1"
name: "Boiler – Natural Gas (LHV)"
version: "2.1.3"
summary: "Computes CO2e emissions from natural gas boiler fuel consumption using Lower Heating Value (LHV) method."
tags:
  - "buildings"
  - "combustion"
  - "scope1"
  - "natural-gas"
owners:
  - "@gl/industry-buildings"
  - "@gl/climate-science"
license: "Apache-2.0"
```

---

### 2.2 Compute Section

The compute section defines the core computational logic of the agent.

#### Purpose
- Specify Python entrypoint function
- Define input/output parameters with types and units
- Reference emission factors
- Declare determinism guarantees
- Set performance constraints

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `entrypoint` | string | Yes | - | Python URI (e.g., `"python://module:function"`) |
| `deterministic` | bool | No | `true` | Whether same inputs → same outputs |
| `inputs` | dict[str, IOField] | Yes | - | Input parameter specifications |
| `outputs` | dict[str, OutputField] | Yes | - | Output parameter specifications |
| `factors` | dict[str, FactorRef] | No | `{}` | Emission factor references |
| `dependencies` | list[string] | No | `null` | Python dependencies (e.g., `["pandas==2.1.4"]`) |
| `python_version` | string | No | `null` | Required Python version (e.g., `"3.11"`) |
| `timeout_s` | int | No | `null` | Max execution time (1-3600 seconds) |
| `memory_limit_mb` | int | No | `null` | Max memory usage (128-16384 MB) |

#### Validation Rules

- **entrypoint**: MUST match `python://module.path:function_name` pattern
  - No path traversal (`..`)
  - No absolute paths (`/`)
  - Valid Python identifiers only
- **inputs**: MUST be non-empty
- **outputs**: MUST be non-empty
- **timeout_s**: Range 1-3600 (1 second to 1 hour)
- **memory_limit_mb**: Range 128-16384 (128 MB to 16 GB)

#### Example

```yaml
compute:
  entrypoint: "python://gl.agents.boiler.ng:compute"
  deterministic: true
  dependencies:
    - "pandas==2.1.4"
    - "numpy==1.26.0"
    - "pint==0.23"
  python_version: "3.11"
  timeout_s: 30
  memory_limit_mb: 512

  inputs:
    fuel_volume:
      dtype: "float64"
      unit: "m^3"
      required: true
      ge: 0
      description: "Volume of natural gas consumed (cubic meters)"

    efficiency:
      dtype: "float64"
      unit: "1"
      required: true
      gt: 0
      le: 1
      description: "Boiler thermal efficiency (fraction, 0-1)"

  outputs:
    co2e_kg:
      dtype: "float64"
      unit: "kgCO2e"
      description: "Total CO2 equivalent emissions (kilograms)"

  factors:
    co2e_factor:
      ref: "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
      gwp_set: "AR6GWP100"
      description: "IPCC AR6 natural gas combustion emission factor"
```

---

### 2.3 AI Section

The AI section configures LLM capabilities, tools, and budget constraints.

#### Purpose
- Define LLM system prompt and JSON mode
- Specify function calling tools with schemas
- Configure RAG document retrieval
- Enforce budget limits to prevent runaway costs

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `json_mode` | bool | No | `true` | Use structured JSON output |
| `system_prompt` | string | No | `null` | Instructions for LLM |
| `budget` | AIBudget | No | `null` | Cost and token constraints |
| `rag_collections` | list[string] | No | `[]` | RAG collection names |
| `tools` | list[AITool] | No | `[]` | Function calling tools |

#### AIBudget Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `max_cost_usd` | float | No | `null` | Max USD per agent run (≥0) |
| `max_input_tokens` | int | No | `null` | Max cumulative input tokens (≥0) |
| `max_output_tokens` | int | No | `null` | Max cumulative output tokens (≥0) |
| `max_retries` | int | No | `3` | Max retries for failed calls (0-10) |

#### AITool Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Tool name (valid Python identifier) |
| `description` | string | No | `null` | Human-readable description |
| `schema_in` | dict | Yes | - | JSON Schema for input (draft-2020-12) |
| `schema_out` | dict | Yes | - | JSON Schema for output (draft-2020-12) |
| `impl` | string | Yes | - | Python URI for implementation |
| `safe` | bool | No | `true` | Whether tool is pure/safe (enforced) |

#### Validation Rules

- **tools**: Tool names MUST be unique
- **tools[].name**: MUST match `^[a-zA-Z_][a-zA-Z0-9_]*$` (valid Python identifier)
- **tools[].schema_in**: MUST be valid JSON Schema draft-2020-12
- **tools[].schema_out**: MUST be valid JSON Schema draft-2020-12
- **tools[].impl**: MUST be valid `python://` URI
- **tools[].safe=true**: Tool implementation undergoes AST analysis (future: enforced)
- **budget.max_retries**: Range 0-10

#### Example

```yaml
ai:
  json_mode: true
  system_prompt: |
    You are a climate advisor specializing in industrial building emissions.

    Guidelines:
    - Use tools to calculate emissions; never guess numbers.
    - Always cite emission factors with proper URIs (ef://).
    - Validate all inputs for physical plausibility.

  budget:
    max_cost_usd: 1.00
    max_input_tokens: 15000
    max_output_tokens: 2000
    max_retries: 3

  rag_collections:
    - "ghg_protocol_corp"
    - "ipcc_ar6"
    - "gl_docs"

  tools:
    - name: "select_emission_factor"
      description: "Select appropriate emission factor based on fuel type, region, and year"
      schema_in:
        type: object
        properties:
          fuel_type:
            type: string
            enum: ["natural_gas", "coal", "oil", "biomass"]
          region:
            type: string
          year:
            type: integer
            minimum: 1990
            maximum: 2100
        required: ["fuel_type", "region", "year"]
      schema_out:
        type: object
        properties:
          ef_uri:
            type: string
            pattern: "^ef://"
          value:
            type: number
          unit:
            type: string
        required: ["ef_uri", "value", "unit"]
      impl: "python://gl.ai.tools.ef:select"
      safe: true
```

---

### 2.4 Realtime Section

The realtime section controls how agents handle external data streams.

#### Purpose
- Enable deterministic replay mode (cached data) vs live mode (fresh data)
- Configure realtime connectors for grid intensity, weather, commodity prices
- Specify data aggregation windows and TTL

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `default_mode` | string | No | `"replay"` | `"replay"` or `"live"` |
| `snapshot_path` | string | No | `null` | Path to cached snapshot for replay |
| `connectors` | list[ConnectorRef] | No | `[]` | Realtime connector configs |

#### ConnectorRef Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Connector name |
| `topic` | string | Yes | - | Data topic/stream identifier |
| `window` | string | No | `null` | Aggregation window (e.g., `"1h"`, `"15min"`) |
| `ttl` | string | No | `null` | Cache time-to-live (e.g., `"6h"`, `"1d"`) |
| `required` | bool | No | `false` | Whether agent fails if unavailable |

#### Validation Rules

- **default_mode**: MUST be `"replay"` or `"live"`
- If `default_mode="live"`: MUST have at least one connector
- **connectors**: Connector names MUST be unique

#### Example

```yaml
realtime:
  default_mode: "replay"
  snapshot_path: "snapshots/2024-10-06_boiler_data.json"

  connectors:
    - name: "grid_intensity"
      topic: "region_hourly_ci"
      window: "1h"
      ttl: "6h"
      required: false

    - name: "weather"
      topic: "outdoor_temperature"
      window: "15min"
      ttl: "1h"
      required: false
```

---

### 2.5 Provenance Section

The provenance section ensures reproducibility and auditability.

#### Purpose
- Pin emission factor versions for immutability
- Specify GWP set for CH4/N2O conversions
- Define audit trail fields for provenance hash

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `pin_ef` | bool | No | `true` | Pin emission factor versions |
| `gwp_set` | string | No | `"AR6GWP100"` | GWP set (AR6GWP100, AR5GWP100, SAR, AR4) |
| `record` | list[string] | Yes | - | Fields to include in provenance |

#### Provenance Record Fields

Common values for `record`:
- `"inputs"`: All input values
- `"outputs"`: All output values
- `"factors"`: Emission factor metadata
- `"ef_uri"`: Emission factor URIs
- `"ef_cid"`: Emission factor content IDs (hash)
- `"unit_system"`: Unit system used
- `"code_sha"`: Git SHA of agent code
- `"env"`: Environment metadata (Python version, OS)
- `"inputs_hash"`: Hash of inputs (deduplication)
- `"seed"`: Random seed (deterministic AI)
- `"timestamp"`: ISO 8601 timestamp
- `"user"`: User who ran the agent

#### Validation Rules

- **record**: MUST NOT contain duplicates
- **CRITICAL**: If `pin_ef=true`, MUST have at least one emission factor in `compute.factors`

#### Example

```yaml
provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"

  record:
    - "inputs"
    - "outputs"
    - "factors"
    - "ef_uri"
    - "ef_cid"
    - "unit_system"
    - "code_sha"
    - "env"
    - "inputs_hash"
    - "seed"
    - "timestamp"
    - "user"
```

---

### 2.6 Security Section (Optional)

The security section provides defense-in-depth for network access.

#### Purpose
- Allowlist hosts for network egress
- Block/log violations based on policy

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `allowlist_hosts` | list[string] | No | `[]` | Allowed hostnames for network access |
| `block_on_violation` | bool | No | `true` | Hard fail vs log warning |

#### Example

```yaml
security:
  allowlist_hosts:
    - "api.ipcc.ch"
    - "api.epa.gov"
    - "api.eia.gov"
  block_on_violation: true
```

---

### 2.7 Tests Section (Optional)

The tests section defines golden tests and property-based invariants.

#### Purpose
- Regression testing with known input/output pairs
- Property-based validation of invariants (monotonicity, bounds)

#### Golden Test Format

```yaml
tests:
  golden:
    - name: "baseline"
      description: "Standard boiler operation"
      input:
        fuel_volume: 100.0
        lhv: 38.0
        efficiency: 0.92
      expect:
        co2e_kg:
          value: 197.6
          tol: 0.01  # 1% tolerance
```

#### Property-Based Test Format

```yaml
tests:
  properties:
    - name: "monotone_fuel_volume"
      rule: "output.co2e_kg is nondecreasing in input.fuel_volume"
      description: "More fuel → more emissions"

    - name: "non_negative_emissions"
      rule: "output.co2e_kg >= 0"
      description: "Emissions cannot be negative"
```

---

## 3. Field Reference

### 3.1 IOField (Input Parameter)

Input field specification with type, unit, constraints.

| Field | Type | Required | Default | Description | Validation |
|-------|------|----------|---------|-------------|------------|
| `dtype` | string | Yes | - | Data type | One of: `float32`, `float64`, `int32`, `int64`, `string`, `bool` |
| `unit` | string | Yes | - | Physical unit | MUST be in climate units whitelist; use `"1"` for dimensionless |
| `required` | bool | No | `true` | Whether required | - |
| `default` | any | No | `null` | Default value | Only valid if `required=false` |
| `description` | string | No | `null` | Human description | - |
| `ge` | float | No | `null` | Greater or equal (≥) | Numeric only |
| `gt` | float | No | `null` | Greater than (>) | Numeric only |
| `le` | float | No | `null` | Less or equal (≤) | Numeric only |
| `lt` | float | No | `null` | Less than (<) | Numeric only |
| `enum` | list | No | `null` | Allowed values | - |

**Special Validation:**
- If `dtype` is `"string"` or `"bool"`, `unit` MUST be `"1"` (dimensionless)
- If `default` is set, `required` MUST be `false`

**Example:**

```yaml
efficiency:
  dtype: "float64"
  unit: "1"
  required: true
  gt: 0
  le: 1
  description: "Boiler thermal efficiency (0-1)"
```

---

### 3.2 OutputField (Output Parameter)

Output field specification (simpler than IOField).

| Field | Type | Required | Default | Description | Validation |
|-------|------|----------|---------|-------------|------------|
| `dtype` | string | Yes | - | Data type | One of: `float32`, `float64`, `int32`, `int64`, `string`, `bool` |
| `unit` | string | Yes | - | Physical unit | MUST be in climate units whitelist |
| `description` | string | No | `null` | Human description | - |

**Example:**

```yaml
co2e_kg:
  dtype: "float64"
  unit: "kgCO2e"
  description: "Total CO2 equivalent emissions"
```

---

### 3.3 FactorRef (Emission Factor Reference)

Emission factor reference with URI and optional GWP set.

| Field | Type | Required | Default | Description | Validation |
|-------|------|----------|---------|-------------|------------|
| `ref` | string | Yes | - | Emission factor URI | MUST match `ef://authority/path` pattern |
| `gwp_set` | string | No | `null` | GWP set | One of: `AR6GWP100`, `AR5GWP100`, `SAR`, `AR4` |
| `description` | string | No | `null` | Human description | - |

**Example:**

```yaml
co2e_factor:
  ref: "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
  gwp_set: "AR6GWP100"
  description: "IPCC AR6 natural gas combustion factor"
```

---

### 3.4 Climate Units Whitelist

AgentSpec v2 validates all units against a comprehensive climate units whitelist. Only approved units are allowed to prevent typos and ensure consistency.

**Dimensionless:**
- `1` (preferred)
- `` (empty string, some systems)

**GHG Emissions:**
- `kgCO2e`, `tCO2e`, `MtCO2e`, `GtCO2e`
- `kgCO2`, `tCO2`
- `kgCH4`, `tCH4`
- `kgN2O`, `tN2O`

**Energy:**
- `J`, `kJ`, `MJ`, `GJ`, `TJ`
- `Wh`, `kWh`, `MWh`, `GWh`, `TWh`
- `BTU`, `kBTU`, `MMBTU`, `therm`

**Power:**
- `W`, `kW`, `MW`, `GW`

**Mass:**
- `g`, `kg`, `t`, `Mt`, `Gt`
- `lb`, `ton` (US), `tonne` (metric)

**Volume:**
- `L`, `kL`, `ML`
- `m3`, `m^3`
- `gal`, `ft3`, `ft^3`

**Area:**
- `m2`, `m^2`, `km2`, `km^2`, `ha`
- `ft2`, `ft^2`

**Distance:**
- `m`, `km`, `mi`, `ft`

**Temperature:**
- `K`, `degC`, `degF`

**Pressure:**
- `Pa`, `kPa`, `MPa`, `bar`, `psi`

**Time:**
- `s`, `min`, `h`, `hr`, `d`, `day`, `yr`, `year`

**Monetary:**
- `USD`, `EUR`, `GBP`

**Intensity Units:**
- `kgCO2e/kWh`, `tCO2e/MWh`, `kgCO2e/km`, `tCO2e/t`, `kgCO2e/m2`, `kgCO2e/USD`
- `gCO2e/MJ`, `kgCO2e/MJ`, `kgCO2e/GJ`, `tCO2e/TJ`

**Heating Values:**
- `MJ/kg`, `MJ/L`, `MJ/m3`, `MJ/m^3`, `kJ/kg`, `BTU/lb`

**Grid Intensity:**
- `gCO2e/kWh`, `kgCO2e/MWh`, `lbCO2e/MWh`

**Note:** If you need a unit not in this list, contact the GreenLang team.

---

## 4. Validation Rules

AgentSpec v2 performs comprehensive validation across multiple dimensions.

### 4.1 Schema Version Validation

- **Rule**: `schema_version` MUST be exactly `"2.0.0"`
- **Enforced by**: Pydantic `Literal["2.0.0"]`
- **Error**: `GLValidationError.CONSTRAINT` if wrong value

### 4.2 Agent ID Validation

- **Rule**: `id` MUST match slug pattern
- **Pattern**: `^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$`
- **Requirements**:
  - Lowercase alphanumeric only
  - Separators: `/`, `-`, `_`
  - At least one `/` (hierarchical namespace)
- **Valid**: `"buildings/boiler_ng_v1"`, `"transport/ev_charging/level-2"`
- **Invalid**: `"Buildings/Boiler"` (uppercase), `"boiler"` (no `/`), `"boiler/ng v1"` (space)
- **Error**: `GLValidationError.INVALID_SLUG`

### 4.3 Semantic Versioning Validation

- **Rule**: `version` MUST conform to SemVer 2.0.0
- **Pattern**: `MAJOR.MINOR.PATCH[-prerelease][+build]`
- **Valid**: `"2.1.3"`, `"1.0.0-rc.1"`, `"3.2.1+20241006.gitsha"`
- **Invalid**: `"2.1"` (missing patch), `"v2.1.3"` (prefix), `"2.1.3.4"` (too many segments)
- **Error**: `GLValidationError.INVALID_SEMVER`

### 4.4 Python URI Validation

- **Rule**: Python URIs MUST match `python://module.path:function_name`
- **Pattern**: `^python://([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*):([a-z_][a-z0-9_]*)$`
- **Security**:
  - No path traversal (`..`)
  - No absolute paths (`/`)
  - Valid Python identifiers only
- **Valid**: `"python://gl.agents.boiler.ng:compute"`
- **Invalid**: `"python://../../../etc/passwd:read"` (traversal)
- **Error**: `GLValidationError.INVALID_URI`

### 4.5 Emission Factor URI Validation

- **Rule**: EF URIs MUST match `ef://authority/path/to/factor`
- **Pattern**: `^ef://[a-z0-9_/-]+$`
- **Valid**: `"ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"`
- **Invalid**: `"ef://IPCC_AR6/..."` (uppercase), `"ipcc_ar6/..."` (missing scheme)
- **Error**: `GLValidationError.INVALID_URI`

### 4.6 Unit Validation

- **Rule**: All units MUST be in climate units whitelist
- **Enforcement**: String/bool inputs MUST use dimensionless unit `"1"`
- **Valid**: `"kgCO2e"`, `"MWh"`, `"m^3"`, `"1"`
- **Invalid**: `"kgg"` (typo), `"kilogram"` (use `kg`), `"meters"` (use `m`)
- **Error**: `GLValidationError.UNIT_SYNTAX` or `GLValidationError.UNIT_FORBIDDEN`

**Example Violation:**

```yaml
# INVALID: string input with non-dimensionless unit
region:
  dtype: "string"
  unit: "m"  # ERROR! Must be "1"
```

### 4.7 Duplicate Name Validation

- **Rule**: No duplicate names across ALL namespaces
- **Checked namespaces**:
  - `compute.inputs`
  - `compute.outputs`
  - `compute.factors`
  - `ai.tools`
  - `realtime.connectors`
- **Error**: `GLValidationError.DUPLICATE_NAME`

**Example Violation:**

```yaml
compute:
  inputs:
    volume:  # Name "volume"
      dtype: "float64"
      unit: "m^3"
  outputs:
    volume:  # ERROR! Duplicate name
      dtype: "float64"
      unit: "m^3"
```

### 4.8 Cross-Field Validation

#### provenance.pin_ef Requires Factors

- **Rule**: If `provenance.pin_ef=true`, MUST have at least one factor in `compute.factors`
- **Rationale**: Cannot pin factors if none exist
- **Error**: `GLValidationError.PROVENANCE_INVALID`

**Example Violation:**

```yaml
provenance:
  pin_ef: true  # ERROR! No factors defined
  record: ["inputs"]

compute:
  factors: {}  # Empty!
```

#### default Requires required=false

- **Rule**: If `input.default` is set, `input.required` MUST be `false`
- **Error**: `GLValidationError.CONSTRAINT`

**Example Violation:**

```yaml
region:
  dtype: "string"
  unit: "1"
  required: true   # ERROR! Cannot be true if default is set
  default: "US"
```

#### live Mode Requires Connectors

- **Rule**: If `realtime.default_mode="live"`, MUST have at least one connector
- **Error**: `GLValidationError.MODE_INVALID`

### 4.9 JSON Schema Validation

- **Rule**: AI tool `schema_in` and `schema_out` MUST be valid JSON Schema draft-2020-12
- **Enforced by**: `jsonschema.Draft202012Validator.check_schema()`
- **Error**: `GLValidationError.AI_SCHEMA_INVALID`

### 4.10 Safe Tool Enforcement

- **Rule**: If `ai.tools[].safe=true`, tool implementation undergoes AST analysis
- **Checks** (future implementation):
  - Pure function (no side effects)
  - No network access
  - No filesystem writes
  - No subprocess execution
  - No `eval`/`exec`
- **Error**: `GLValidationError.CONSTRAINT`
- **Status**: Placeholder in v2.0.0; full enforcement in v2.1.0

---

## 5. Error Codes Reference

All validation errors map to stable error codes for automation.

### GLValidationError Error Codes

| Code | Name | Description | Example |
|------|------|-------------|---------|
| `GLValidationError.MISSING_FIELD` | Missing field | Required field is missing | `compute` section not present |
| `GLValidationError.UNKNOWN_FIELD` | Unknown field | Extra field detected (typo) | `compute.entrypont` (typo) |
| `GLValidationError.INVALID_SEMVER` | Invalid semver | Version not SemVer 2.0.0 | `version: "2.1"` (missing patch) |
| `GLValidationError.INVALID_SLUG` | Invalid slug | Agent ID invalid format | `id: "Boiler NG"` (uppercase, space) |
| `GLValidationError.INVALID_URI` | Invalid URI | URI scheme malformed | `entrypoint: "gl.agents:compute"` (missing `python://`) |
| `GLValidationError.DUPLICATE_NAME` | Duplicate name | Name collision detected | Input and output both named `"volume"` |
| `GLValidationError.DUPLICATE_ID` | Duplicate ID | Agent ID already exists | Two agents with `id: "buildings/boiler_ng_v1"` |
| `GLValidationError.UNIT_SYNTAX` | Unit syntax error | Unit not in whitelist | `unit: "kgg"` (typo for `kg`) |
| `GLValidationError.UNIT_FORBIDDEN` | Unit forbidden | Non-dimensionless for string/bool | `dtype: "string", unit: "kg"` |
| `GLValidationError.CONSTRAINT` | Constraint violation | Value violates constraint | `efficiency: 1.5` (violates `le: 1`) |
| `GLValidationError.FACTOR_UNRESOLVED` | Factor unresolved | EF URI not found in registry | `ref: "ef://unknown/factor"` |
| `GLValidationError.AI_SCHEMA_INVALID` | AI schema invalid | Tool schema not valid JSON Schema | `schema_in: {typ: "object"}` (typo) |
| `GLValidationError.BUDGET_INVALID` | Budget invalid | AI budget constraint invalid | `max_cost_usd: -1.0` (negative) |
| `GLValidationError.MODE_INVALID` | Mode invalid | Realtime mode invalid | `default_mode: "live"` with no connectors |
| `GLValidationError.CONNECTOR_INVALID` | Connector invalid | Connector config invalid | (Future use) |
| `GLValidationError.PROVENANCE_INVALID` | Provenance invalid | Provenance config invalid | `pin_ef: true` with no factors |

### Error Code Usage

#### In Python

```python
try:
    spec = from_yaml("pack.yaml")
except GLValidationError as e:
    print(f"Code: {e.code}")
    print(f"Path: {'/'.join(e.path)}")
    print(f"Message: {e.message}")

    # Automated handling
    if e.code == GLVErr.INVALID_SEMVER:
        # Suggest semver fix
        pass
    elif e.code == GLVErr.UNIT_SYNTAX:
        # Suggest valid units
        pass
```

#### In CI/CD

```bash
# Exit code mapping
gl spec validate pack.yaml
if [ $? -eq 1 ]; then
  echo "Validation failed"
fi
```

---

## 6. Examples

### 6.1 Minimal Valid Spec

The absolute minimum fields required for a valid AgentSpec v2.

```yaml
schema_version: "2.0.0"
id: "minimal/agent_v1"
name: "Minimal Agent"
version: "1.0.0"

compute:
  entrypoint: "python://minimal:compute"
  inputs:
    x:
      dtype: "float64"
      unit: "kg"
  outputs:
    y:
      dtype: "float64"
      unit: "kg"

ai:
  # Empty AI section (defaults apply)

realtime:
  # Empty realtime section (defaults to replay mode)

provenance:
  pin_ef: false  # No factors, so must be false
  record: ["inputs"]
```

### 6.2 Full-Featured Spec

Complete example with all sections enabled.

See `examples/agentspec_v2/pack.yaml` for the comprehensive boiler agent example (397 lines with all features).

### 6.3 Loading and Validating

#### From YAML

```python
from greenlang.specs.agentspec_v2 import from_yaml, GLValidationError

try:
    spec = from_yaml("pack.yaml")
    print(f"Loaded: {spec.name} v{spec.version}")
    print(f"Entrypoint: {spec.compute.entrypoint}")
except GLValidationError as e:
    print(f"Validation error: {e}")
except FileNotFoundError:
    print("File not found")
except yaml.YAMLError as e:
    print(f"YAML syntax error: {e}")
```

#### From JSON

```python
from greenlang.specs.agentspec_v2 import from_json

spec = from_json("pack.json")
```

#### From Dictionary

```python
from greenlang.specs.agentspec_v2 import validate_spec

data = {
    "schema_version": "2.0.0",
    "id": "test/agent",
    "name": "Test Agent",
    "version": "1.0.0",
    # ...
}

spec = validate_spec(data)
```

### 6.4 Common Patterns

#### Deterministic Agent (No AI)

```yaml
schema_version: "2.0.0"
id: "deterministic/calculator_v1"
name: "Emission Calculator"
version: "1.0.0"

compute:
  entrypoint: "python://calc:compute"
  deterministic: true  # Guaranteed reproducibility
  inputs:
    energy_kwh:
      dtype: "float64"
      unit: "kWh"
      ge: 0
  outputs:
    co2e_kg:
      dtype: "float64"
      unit: "kgCO2e"
  factors:
    grid_ef:
      ref: "ef://epa/egrid/us_average/co2e_kg_per_kwh"

ai:
  # No AI tools, pure deterministic calculation

realtime:
  default_mode: "replay"

provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"
  record: ["inputs", "outputs", "ef_uri", "ef_cid"]
```

#### AI-Powered Agent

```yaml
schema_version: "2.0.0"
id: "ai/advisor_v1"
name: "Climate Advisor Agent"
version: "1.0.0"

compute:
  entrypoint: "python://advisor:compute"
  deterministic: false  # LLM introduces non-determinism
  inputs:
    query:
      dtype: "string"
      unit: "1"
  outputs:
    recommendation:
      dtype: "string"
      unit: "1"

ai:
  json_mode: true
  system_prompt: "You are a climate advisor. Use tools; never guess."
  budget:
    max_cost_usd: 5.00
    max_input_tokens: 50000
    max_output_tokens: 5000
  tools:
    - name: "search_factors"
      description: "Search emission factors"
      schema_in: {...}
      schema_out: {...}
      impl: "python://tools:search_factors"
      safe: true

realtime:
  default_mode: "live"
  connectors:
    - name: "grid_intensity"
      topic: "realtime_ci"
      window: "1h"
      required: true

provenance:
  pin_ef: false  # No factors in this example
  record: ["inputs", "outputs", "seed", "timestamp"]
```

#### Realtime Agent with Snapshot

```yaml
schema_version: "2.0.0"
id: "realtime/grid_monitor_v1"
name: "Grid Intensity Monitor"
version: "1.0.0"

compute:
  entrypoint: "python://monitor:compute"
  inputs:
    region:
      dtype: "string"
      unit: "1"
      enum: ["US", "EU", "UK"]
  outputs:
    current_ci:
      dtype: "float64"
      unit: "kgCO2e/MWh"

ai: {}

realtime:
  default_mode: "replay"
  snapshot_path: "snapshots/2024-10-06_grid_ci.json"
  connectors:
    - name: "grid_intensity"
      topic: "region_hourly_ci"
      window: "1h"
      ttl: "6h"
      required: true

provenance:
  pin_ef: false
  record: ["inputs", "outputs", "timestamp", "snapshot_path"]
```

---

## 7. CLI Usage

AgentSpec v2 provides CLI commands for validation and introspection.

### 7.1 Validate Spec

Validate a spec file and report errors.

```bash
gl spec validate pack.yaml
```

**Output (success):**
```
✓ pack.yaml is valid
  Agent: buildings/boiler_ng_v1 v2.1.3
  Inputs: 4 parameters
  Outputs: 3 parameters
  Factors: 2 emission factors
```

**Output (error):**
```
✗ pack.yaml validation failed

GLValidationError.INVALID_SEMVER: version: Invalid semantic version: '2.1'
  Expected format: MAJOR.MINOR.PATCH (e.g., '2.1.3')
  See https://semver.org/

GLValidationError.DUPLICATE_NAME: compute/outputs: Duplicate names: ['volume']
```

### 7.2 Show Spec Details

Display comprehensive spec information.

```bash
gl spec show pack.yaml
```

**Output:**
```
Agent: buildings/boiler_ng_v1 v2.1.3
Name: Boiler – Natural Gas (LHV)
Summary: Computes CO2e emissions from natural gas boiler fuel consumption

Sections:
  compute:
    entrypoint: python://gl.agents.boiler.ng:compute
    deterministic: true
    inputs: 4 (fuel_volume, lhv, efficiency, region)
    outputs: 3 (co2e_kg, energy_mj, intensity)
    factors: 2 (co2e_factor, grid_factor)

  ai:
    json_mode: true
    budget: $1.00 max, 15000 in / 2000 out tokens
    tools: 2 (select_emission_factor, validate_input)
    rag: 3 collections

  realtime:
    mode: replay
    snapshot: snapshots/2024-10-06_boiler_data.json
    connectors: 2 (grid_intensity, weather)

  provenance:
    pin_ef: true
    gwp_set: AR6GWP100
    record: 13 fields
```

### 7.3 Export to JSON

Convert YAML spec to JSON format.

```bash
gl spec export pack.yaml --format json > pack.json
```

### 7.4 Export JSON Schema

Generate JSON Schema for external tooling.

```bash
gl spec schema > agentspec_v2.schema.json
```

This generates the draft-2020-12 JSON Schema for use with:
- VS Code (schema validation in YAML editor)
- CI/CD (schema-based linting)
- Documentation generators

### 7.5 Generate Code from Spec (Future)

```bash
# Future feature: Code generation
gl generate pack.yaml --output gen/
```

This will generate:
- Python stubs for entrypoint function
- Type-safe input/output dataclasses
- Test harness with golden tests

---

## 8. Python API

### 8.1 Loading Specs

#### from_yaml

```python
from greenlang.specs.agentspec_v2 import from_yaml
from pathlib import Path

# Load from path string
spec = from_yaml("pack.yaml")

# Load from Path object
spec = from_yaml(Path("examples/agentspec_v2/pack.yaml"))

# Raises:
#   - GLValidationError: If spec is invalid
#   - FileNotFoundError: If file doesn't exist
#   - yaml.YAMLError: If YAML is malformed
```

#### from_json

```python
from greenlang.specs.agentspec_v2 import from_json

spec = from_json("pack.json")

# Raises:
#   - GLValidationError: If spec is invalid
#   - FileNotFoundError: If file doesn't exist
#   - json.JSONDecodeError: If JSON is malformed
```

#### validate_spec

```python
from greenlang.specs.agentspec_v2 import validate_spec

data = {
    "schema_version": "2.0.0",
    "id": "test/agent",
    "name": "Test",
    "version": "1.0.0",
    # ...
}

spec = validate_spec(data)

# Raises:
#   - GLValidationError: If spec is invalid
```

### 8.2 Accessing Spec Fields

```python
spec = from_yaml("pack.yaml")

# Metadata
print(spec.id)               # "buildings/boiler_ng_v1"
print(spec.name)             # "Boiler – Natural Gas (LHV)"
print(spec.version)          # "2.1.3"
print(spec.tags)             # ["buildings", "combustion", ...]

# Compute
print(spec.compute.entrypoint)      # "python://gl.agents.boiler.ng:compute"
print(spec.compute.deterministic)   # True
print(len(spec.compute.inputs))     # 4
print(len(spec.compute.outputs))    # 3
print(len(spec.compute.factors))    # 2

# Inputs
fuel_input = spec.compute.inputs["fuel_volume"]
print(fuel_input.dtype)        # "float64"
print(fuel_input.unit)         # "m^3"
print(fuel_input.required)     # True
print(fuel_input.ge)           # 0

# Outputs
co2e_output = spec.compute.outputs["co2e_kg"]
print(co2e_output.dtype)       # "float64"
print(co2e_output.unit)        # "kgCO2e"

# Factors
factor = spec.compute.factors["co2e_factor"]
print(factor.ref)              # "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
print(factor.gwp_set)          # "AR6GWP100"

# AI
print(spec.ai.json_mode)       # True
print(spec.ai.budget.max_cost_usd)      # 1.00
print(len(spec.ai.tools))      # 2
print(spec.ai.rag_collections) # ["ghg_protocol_corp", "ipcc_ar6", "gl_docs"]

# Realtime
print(spec.realtime.default_mode)     # "replay"
print(spec.realtime.snapshot_path)    # "snapshots/2024-10-06_boiler_data.json"
print(len(spec.realtime.connectors))  # 2

# Provenance
print(spec.provenance.pin_ef)    # True
print(spec.provenance.gwp_set)   # "AR6GWP100"
print(len(spec.provenance.record)) # 13
```

### 8.3 Error Handling

```python
from greenlang.specs.agentspec_v2 import from_yaml, GLValidationError, GLVErr

try:
    spec = from_yaml("pack.yaml")
except GLValidationError as e:
    # Structured error handling
    print(f"Error code: {e.code}")
    print(f"Field path: {'/'.join(e.path)}")
    print(f"Message: {e.message}")

    # Serialize to dict for logging
    error_dict = e.to_dict()
    # {"code": "GLValidationError.INVALID_SEMVER", "message": "...", "path": ["version"], "context": None}

    # Conditional handling
    if e.code == GLVErr.INVALID_SEMVER:
        print("Fix version format: MAJOR.MINOR.PATCH")
    elif e.code == GLVErr.UNIT_SYNTAX:
        print("Use approved climate units: kgCO2e, tCO2e, kWh, etc.")
    elif e.code == GLVErr.DUPLICATE_NAME:
        print("Ensure unique names across inputs/outputs/factors/tools/connectors")
```

### 8.4 Exporting JSON Schema

```python
from greenlang.specs.agentspec_v2 import to_json_schema
import json

# Generate JSON Schema (draft-2020-12)
schema = to_json_schema()

# Save to file
with open("agentspec_v2.schema.json", "w") as f:
    json.dump(schema, f, indent=2)

# Schema metadata
print(schema["$schema"])  # "https://json-schema.org/draft/2020-12/schema"
print(schema["$id"])      # "https://greenlang.io/specs/agentspec_v2.json"
print(schema["title"])    # "GreenLang AgentSpec v2"
```

### 8.5 Programmatic Spec Construction

```python
from greenlang.specs.agentspec_v2 import (
    AgentSpecV2,
    ComputeSpec,
    IOField,
    OutputField,
    FactorRef,
    AISpec,
    AIBudget,
    RealtimeSpec,
    ProvenanceSpec,
)

spec = AgentSpecV2(
    schema_version="2.0.0",
    id="programmatic/agent_v1",
    name="Programmatic Agent",
    version="1.0.0",
    summary="Created programmatically",
    tags=["test", "programmatic"],

    compute=ComputeSpec(
        entrypoint="python://test:compute",
        deterministic=True,
        inputs={
            "x": IOField(dtype="float64", unit="kg", required=True, ge=0),
        },
        outputs={
            "y": OutputField(dtype="float64", unit="kgCO2e"),
        },
        factors={
            "ef": FactorRef(ref="ef://test/factor"),
        },
    ),

    ai=AISpec(
        json_mode=True,
        budget=AIBudget(max_cost_usd=1.00),
    ),

    realtime=RealtimeSpec(
        default_mode="replay",
    ),

    provenance=ProvenanceSpec(
        pin_ef=True,
        gwp_set="AR6GWP100",
        record=["inputs", "outputs", "ef_uri"],
    ),
)

# Serialize to dict
spec_dict = spec.model_dump()

# Serialize to JSON
spec_json = spec.model_dump_json(indent=2)
```

---

## 9. Migration Guide

### 9.1 Migrating from AgentSpec v1

AgentSpec v2 is a major revision with breaking changes.

#### Key Differences

| v1 | v2 | Notes |
|----|----|----|
| `schema_version: "1.0.0"` | `schema_version: "2.0.0"` | Required literal change |
| No `ai` section | `ai` section required | Even if empty: `ai: {}` |
| No `realtime` section | `realtime` section required | Even if empty: `realtime: {}` |
| No `provenance` section | `provenance` section required | Must specify `record` |
| Loose unit validation | Strict climate units whitelist | Only approved units allowed |
| Factors optional | If `pin_ef=true`, factors required | Cross-field validation |
| No duplicate checks | Global duplicate name checks | Across all namespaces |

#### Migration Checklist

1. **Update schema_version**
   ```yaml
   # v1
   schema_version: "1.0.0"

   # v2
   schema_version: "2.0.0"
   ```

2. **Add required sections**
   ```yaml
   # Add if missing:
   ai: {}

   realtime:
     default_mode: "replay"

   provenance:
     pin_ef: false  # Or true if you have factors
     record: ["inputs", "outputs"]
   ```

3. **Validate units**
   - Check all `unit` fields against climate units whitelist
   - Common migrations:
     - `"meters"` → `"m"`
     - `"kilograms"` → `"kg"`
     - `"kgCO2eq"` → `"kgCO2e"`

4. **Fix string/bool units**
   ```yaml
   # v1 (allowed)
   region:
     dtype: "string"
     unit: "country"  # Any unit

   # v2 (required)
   region:
     dtype: "string"
     unit: "1"  # MUST be dimensionless
   ```

5. **Check duplicate names**
   - Ensure no name appears in multiple namespaces
   - Rename conflicting fields

6. **Validate with CLI**
   ```bash
   gl spec validate pack.yaml
   ```

### 9.2 Breaking Changes

- **schema_version**: MUST be `"2.0.0"` (literal)
- **ai, realtime, provenance**: Now required sections
- **Unit validation**: Strict whitelist enforcement
- **String/bool units**: MUST use `"1"`
- **Duplicate names**: Globally disallowed
- **provenance.pin_ef=true**: Requires at least one factor

### 9.3 New Required Fields

- `ai` section (can be empty: `ai: {}`)
- `realtime` section (can be empty: `realtime: {}`)
- `provenance.record` (list of fields to include in provenance)

### 9.4 Deprecated Patterns

- **Non-climate units**: Custom units like `"units"`, `"items"` no longer allowed
- **Loose factor references**: All `ef://` URIs must match strict pattern
- **Duplicate names**: Previously allowed across namespaces, now forbidden

---

## 10. Best Practices

### 10.1 Determinism by Default

Always prefer `deterministic: true` unless you have a specific reason for non-determinism.

**Good:**
```yaml
compute:
  deterministic: true
  # Pure calculation, no LLM, no external data
```

**Acceptable (with justification):**
```yaml
compute:
  deterministic: false
  # Using LLM tools, result varies by prompt/temperature
```

### 10.2 Unit-Aware Outputs

Always specify units for all outputs, even if dimensionless.

**Good:**
```yaml
outputs:
  efficiency_ratio:
    dtype: "float64"
    unit: "1"  # Explicitly dimensionless
```

**Bad:**
```yaml
outputs:
  efficiency_ratio:
    dtype: "float64"
    unit: ""  # Unclear intent
```

### 10.3 Emission Factor Pinning

Enable `pin_ef: true` for production agents to ensure reproducibility.

**Good (auditable):**
```yaml
provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"

compute:
  factors:
    ef: FactorRef(ref="ef://ipcc_ar6/...")
```

**Bad (non-reproducible):**
```yaml
provenance:
  pin_ef: false  # Latest factors fetched at runtime
```

### 10.4 Budget Constraints for AI

Always set budget limits for LLM-powered agents.

**Good:**
```yaml
ai:
  budget:
    max_cost_usd: 1.00
    max_input_tokens: 15000
    max_output_tokens: 2000
```

**Bad (runaway costs):**
```yaml
ai:
  budget: null  # No limits!
```

### 10.5 Snapshot Paths for Replay Mode

Specify `snapshot_path` for deterministic replay.

**Good:**
```yaml
realtime:
  default_mode: "replay"
  snapshot_path: "snapshots/2024-10-06_data.json"
```

**Bad:**
```yaml
realtime:
  default_mode: "replay"
  snapshot_path: null  # Where's the data?
```

### 10.6 Golden Tests for Validation

Include golden tests for critical calculations.

**Good:**
```yaml
tests:
  golden:
    - name: "baseline"
      input: {fuel_volume: 100, efficiency: 0.92}
      expect:
        co2e_kg: {value: 197.6, tol: 0.01}
```

**Bad:**
```yaml
tests: null  # No tests!
```

### 10.7 Property-Based Invariants

Define invariants that MUST hold for all inputs.

**Good:**
```yaml
tests:
  properties:
    - name: "non_negative_emissions"
      rule: "output.co2e_kg >= 0"
    - name: "monotone_fuel"
      rule: "output.co2e_kg is nondecreasing in input.fuel_volume"
```

### 10.8 Descriptive Field Names

Use clear, unambiguous names for inputs/outputs.

**Good:**
```yaml
inputs:
  fuel_volume_m3:
    dtype: "float64"
    unit: "m^3"
    description: "Natural gas volume consumed"
```

**Bad:**
```yaml
inputs:
  x:  # Unclear what x represents
    dtype: "float64"
    unit: "m^3"
```

### 10.9 Input Validation Constraints

Use `ge`, `gt`, `le`, `lt` to enforce physical constraints.

**Good:**
```yaml
inputs:
  efficiency:
    dtype: "float64"
    unit: "1"
    gt: 0    # Must be positive
    le: 1    # Cannot exceed 100%
```

**Bad:**
```yaml
inputs:
  efficiency:
    dtype: "float64"
    unit: "1"
    # No constraints, allows efficiency > 1 or < 0
```

### 10.10 Comprehensive Provenance Records

Include all relevant fields in provenance for full auditability.

**Good:**
```yaml
provenance:
  record:
    - "inputs"
    - "outputs"
    - "factors"
    - "ef_uri"
    - "ef_cid"
    - "code_sha"
    - "seed"
    - "timestamp"
    - "user"
```

**Minimal (but acceptable):**
```yaml
provenance:
  record:
    - "inputs"
    - "outputs"
```

---

## 11. FAQ

### Q: Why do I need the `schema_version` field?

**A:** The `schema_version` field enables forward/backward compatibility. When we release AgentSpec v3, tools can detect which version to validate against. It must be `"2.0.0"` for v2 specs.

---

### Q: Can I use custom units?

**A:** No. AgentSpec v2 validates against a curated climate units whitelist to prevent typos and ensure consistency. If you need a unit that's not in the whitelist, contact the GreenLang team to request it be added.

Common approved units:
- Emissions: `kgCO2e`, `tCO2e`, `MtCO2e`
- Energy: `kWh`, `MWh`, `GJ`, `MJ`
- Volume: `m3`, `m^3`, `L`, `gal`
- Mass: `kg`, `t`, `lb`
- Dimensionless: `1`

---

### Q: How do I add a new emission factor?

**A:** Emission factors are defined in the factor registry (separate from AgentSpec). In your spec, reference them via `ef://` URIs:

```yaml
compute:
  factors:
    my_factor:
      ref: "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
      gwp_set: "AR6GWP100"
```

To add a NEW factor to the registry, contact the GreenLang climate science team.

---

### Q: What's the difference between replay and live mode?

**A:**

- **replay mode** (default): Uses cached snapshots for external data. Deterministic and auditable. Best for production.
  ```yaml
  realtime:
    default_mode: "replay"
    snapshot_path: "snapshots/2024-10-06_data.json"
  ```

- **live mode**: Fetches fresh data from connectors in real-time. Non-deterministic. Best for monitoring dashboards.
  ```yaml
  realtime:
    default_mode: "live"
    connectors:
      - name: "grid_intensity"
        topic: "realtime_ci"
  ```

---

### Q: How do safe tools work?

**A:** Tools marked `safe: true` undergo AST (Abstract Syntax Tree) analysis to ensure they're pure functions with no side effects:

- No network access (`requests`, `urllib`, `httpx`)
- No filesystem writes (`open(..., 'w')`, `pathlib.Path.write_text()`)
- No subprocess execution (`subprocess`, `os.system`)
- No `eval`/`exec`

This is enforced statically at spec validation time (future: runtime sandboxing).

---

### Q: Can I have multiple entrypoints?

**A:** No. Each AgentSpec defines exactly ONE entrypoint function. If you need multiple behaviors, create multiple agent specs with different IDs.

**Good:**
- `buildings/boiler_ng_v1` → `python://gl.agents.boiler.ng:compute`
- `buildings/boiler_coal_v1` → `python://gl.agents.boiler.coal:compute`

**Bad:**
- One spec with multiple entrypoints (not supported)

---

### Q: What happens if I violate a constraint?

**A:** AgentSpec validation raises `GLValidationError` with a specific error code. For example:

```python
# Spec with efficiency > 1
efficiency:
  dtype: "float64"
  unit: "1"
  le: 1  # Constraint: must be ≤ 1

# Runtime error:
GLValidationError.CONSTRAINT: compute/inputs/efficiency: Input violates constraint (le=1)
```

Fix the spec and re-validate.

---

### Q: How do I test my spec?

**A:**

1. **CLI validation:**
   ```bash
   gl spec validate pack.yaml
   ```

2. **Python validation:**
   ```python
   from greenlang.specs.agentspec_v2 import from_yaml
   spec = from_yaml("pack.yaml")  # Raises GLValidationError if invalid
   ```

3. **Golden tests:**
   ```yaml
   tests:
     golden:
       - name: "test1"
         input: {x: 100}
         expect: {y: {value: 42, tol: 0.01}}
   ```

---

### Q: Can I use environment variables in specs?

**A:** No. AgentSpec manifests are static YAML/JSON files. Use environment variables in the RUNTIME, not the spec itself.

**Bad:**
```yaml
entrypoint: "${MODULE_PATH}:compute"  # Not supported
```

**Good:**
```yaml
entrypoint: "python://gl.agents.boiler.ng:compute"  # Static
```

Then configure module paths via `PYTHONPATH` or virtualenv at runtime.

---

### Q: What's the difference between `ge` and `gt`?

**A:**

- `ge`: Greater than or equal (≥). Example: `ge: 0` allows `0, 0.1, 1, ...`
- `gt`: Greater than (>). Example: `gt: 0` allows `0.1, 1, ...` but NOT `0`

Use `gt: 0` for efficiency (must be positive), `ge: 0` for fuel volume (zero is valid).

---

## 12. References

### Standards and Protocols

- **GHG Protocol Corporate Standard**
  https://ghgprotocol.org/corporate-standard
  Foundation for scope 1/2/3 emissions accounting

- **IPCC AR6 Guidelines**
  https://www.ipcc.ch/report/ar6/
  Climate science basis, GWP values, emission factors

- **ISO 14064-1:2018**
  https://www.iso.org/standard/66453.html
  Specification for quantifying and reporting GHG emissions

- **Semantic Versioning 2.0.0**
  https://semver.org/
  Version numbering convention (MAJOR.MINOR.PATCH)

- **JSON Schema draft-2020-12**
  https://json-schema.org/draft/2020-12/schema
  Schema specification for AI tool validation

### GreenLang Documentation

- **Pydantic v2 Documentation**
  https://docs.pydantic.dev/2.0/
  Model validation and serialization

- **GreenLang SDK Documentation**
  (Internal link: `/docs/getting-started.md`)

- **Emission Factor Registry**
  (Internal link: `/docs/emissions-factors.md`)

### Related Specs

- **AgentSpec v1** (deprecated)
  `/docs/PACK_SCHEMA_V1.md`

- **Pipeline Spec v1**
  `/docs/GL_PIPELINE_SPEC_V1.md`

### Implementation

- **Source Code**
  `/greenlang/specs/agentspec_v2.py` (Pydantic models)
  `/greenlang/specs/errors.py` (Error codes)

- **Tests**
  `/tests/specs/test_agentspec_ok.py` (104 passing tests)
  `/tests/specs/test_agentspec_errors.py` (Error code tests)

- **Examples**
  `/examples/agentspec_v2/pack.yaml` (Full boiler agent example)

---

## Appendix A: Complete Example

See `examples/agentspec_v2/pack.yaml` for a production-ready example with all features:

- 4 inputs with constraints (fuel_volume, lhv, efficiency, region)
- 3 outputs (co2e_kg, energy_mj, intensity)
- 2 emission factors (co2e_factor, grid_factor)
- AI configuration with 2 tools, budget, RAG
- Realtime with 2 connectors, replay mode, snapshot
- Provenance with factor pinning, AR6GWP100, 13 audit fields
- Security with 3 allowlisted hosts
- Tests with 3 golden tests, 5 property-based tests

Total: 397 lines of production-ready YAML.

---

## Appendix B: Validation Flowchart

```
AgentSpec YAML
     ↓
Parse YAML → YAMLError?
     ↓ (no)
Pydantic Validation
     ↓
┌────────────────────────────────┐
│ Field-Level Validation         │
│ - schema_version = "2.0.0"     │
│ - id slug format               │
│ - version semver               │
│ - units in whitelist           │
│ - URIs valid format            │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│ Model Validation               │
│ - String/bool → unit="1"       │
│ - Default → required=false     │
│ - Live mode → has connectors   │
│ - Tool schemas valid JSON      │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│ Cross-Field Validation         │
│ - pin_ef=true → has factors    │
│ - No duplicate names (global)  │
│ - Unique tags/tools/connectors │
└────────────────────────────────┘
     ↓
Valid AgentSpecV2 ✓
```

---

**End of Documentation**

For questions or issues, contact the GreenLang team or open an issue at:
https://github.com/greenlang/greenlang/issues

Version: 2.0.0
Last Updated: October 2025
License: Apache-2.0
