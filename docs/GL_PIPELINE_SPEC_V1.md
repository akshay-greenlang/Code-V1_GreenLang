# GL Pipeline Specification v1.0

**Status:** Stable (v1.0)
**Scope:** Applies to all GreenLang pipeline definitions (`gl.yaml` files)
**SemVer:** Breaking changes bump MAJOR; additive fields bump MINOR; fixes bump PATCH

## Table of Contents

- [Overview](#overview)
- [Pipeline Schema](#pipeline-schema)
  - [Top-Level Fields](#top-level-fields)
  - [Step Object Specification](#step-object-specification)
- [Reference Syntax](#reference-syntax)
- [Error Handling & Control Flow](#error-handling--control-flow)
- [Determinism & Execution Rules](#determinism--execution-rules)
- [Security & Policy Requirements](#security--policy-requirements)
- [Examples](#examples)
  - [Minimal Pipeline](#minimal-pipeline)
  - [Full-Featured Pipeline](#full-featured-pipeline)
- [Validation Rules](#validation-rules)
- [CLI Usage](#cli-usage)
- [Migration Guide](#migration-guide)

## Overview

GL Pipeline Specification v1.0 defines the structure and semantics of GreenLang pipeline configuration files. These files describe deterministic, auditable workflows for carbon accounting, energy analysis, and sustainability calculations.

**Core Principles:**
- **Deterministic execution**: Same inputs always produce same outputs
- **Reproducible**: Full provenance and audit trails
- **Composable**: Steps can be combined and reused
- **Policy-aware**: Network and data governance constraints

## Pipeline Schema

### Top-Level Fields

#### `name` (required, string)
- **Format:** Slug format: `[a-z0-9][a-z0-9-]{1,62}[a-z0-9]`
- **Description:** Unique identifier for the pipeline
- **Example:** `"boiler-solar-analysis"`, `"portfolio-emissions-v2"`

#### `version` (optional, integer)
- **Format:** Positive integer
- **Description:** Pipeline version number
- **Default:** `1`
- **Example:** `1`, `2`, `42`

#### `vars` (optional, object)
- **Description:** Constants and configuration values available to all steps
- **Scope:** Available via `${vars.key}` syntax
- **Example:**
  ```yaml
  vars:
    region: "IN-North"
    grid_intensity: 0.82
    reporting_year: 2024
  ```

#### `inputs` (optional, object)
- **Description:** External input schema for pipeline parameterization
- **Format:** JSON Schema-like object with type definitions
- **Scope:** Available via `${inputs.key}` syntax
- **Example:**
  ```yaml
  inputs:
    building_size_sqft:
      type: number
      required: true
      minimum: 100
    location:
      type: string
      enum: ["IN-North", "IN-South", "IN-West", "IN-East"]
      default: "IN-North"
  ```

#### `artifacts_dir` (optional, string)
- **Description:** Directory for output artifacts and intermediate files
- **Default:** `"out/"`
- **Format:** Relative path ending with `/`
- **Example:** `"artifacts/"`, `"reports/output/"`

#### `steps` (required, array)
- **Description:** Ordered list of pipeline steps
- **Constraints:** Must contain at least one step
- **Details:** See [Step Object Specification](#step-object-specification)

#### `outputs` (optional, object)
- **Description:** Named outputs from the pipeline execution
- **Format:** Map of output names to step references or literal values
- **Example:**
  ```yaml
  outputs:
    total_emissions_tons: "${steps.carbon.emissions_total}"
    report_path: "${steps.report.output_file}"
    execution_metadata:
      pipeline_version: 1
      completed_at: "${system.timestamp}"
  ```

### Step Object Specification

Each step in the `steps` array is an object with the following fields:

#### `id` (required, string)
- **Format:** Unique identifier within pipeline: `[a-z0-9][a-z0-9_-]{0,62}`
- **Description:** Step identifier used for referencing outputs
- **Example:** `"validate_inputs"`, `"calc_emissions"`, `"generate_report"`

#### `agent` (required, string)
- **Format:** One of:
  - **Local module:** `pkg.module:ClassName`
  - **File path:** `agents/analyzer.py:BoilerAnalyzer`
  - **Pack reference:** `@emissions-core/FuelEmissionAgent`
- **Description:** Agent class responsible for executing the step
- **Examples:**
  ```yaml
  # Local Python module
  agent: "greenlang.agents.carbon:CarbonCalculator"

  # File path
  agent: "agents/boiler_analyzer.py:BoilerAnalyzer"

  # Pack reference
  agent: "@emissions-core/EmissionFactorAgent"
  ```

#### `action` (optional, string)
- **Description:** Method name to call on the agent
- **Default:** `"run"`
- **Example:** `"calculate_emissions"`, `"validate_inputs"`, `"generate_report"`

#### Input Specification (mutually exclusive)

##### `in` (object)
- **Description:** Direct input parameters passed to the agent
- **Format:** Key-value pairs with reference expressions or literals
- **Example:**
  ```yaml
  in:
    building_size: 50000
    location: "${vars.region}"
    previous_output: "${steps.validate.emissions_data}"
  ```

##### `inputsRef` (string)
- **Description:** Reference to another step's complete output
- **Format:** Step reference expression
- **Example:** `"${steps.data_parser.outputs}"`

> **Note:** `in` and `inputsRef` are mutually exclusive. Use `in` for granular control, `inputsRef` for passing complete step outputs.

#### `with` (optional, object)
- **Description:** Agent configuration and parameters (distinct from runtime inputs)
- **Usage:** Static configuration, connection strings, model parameters
- **Example:**
  ```yaml
  with:
    model_version: "v2.1"
    precision: "high"
    cache_enabled: true
    timeout_seconds: 300
  ```

#### `when` (optional, string|boolean)
- **Description:** Conditional execution expression
- **Format:** JavaScript-like expression or boolean literal
- **Default:** `true`
- **Examples:**
  ```yaml
  # Boolean literal
  when: true

  # Simple condition
  when: "${vars.enable_detailed_analysis}"

  # Complex expression
  when: "${steps.validate.success} && ${vars.reporting_year} >= 2024"
  ```

#### `retry` (optional, object)
- **Description:** Retry configuration for failed steps
- **Fields:**
  - `max` (integer): Maximum retry attempts (default: 0)
  - `backoff_seconds` (number): Delay between retries (default: 1)
- **Example:**
  ```yaml
  retry:
    max: 3
    backoff_seconds: 2.5
  ```

#### `on_error` (optional, string|object)
- **Description:** Error handling strategy
- **String values:** `"stop"` (default), `"skip"`, `"continue"`, `"retry"`
- **Object format:**
  ```yaml
  on_error:
    strategy: "continue"
    log_level: "warning"
    default_output:
      status: "failed"
      reason: "${error.message}"
  ```

#### `outputs` (optional, object)
- **Description:** Named outputs produced by this step
- **Format:** Map of output names to their expected types or schemas
- **Example:**
  ```yaml
  outputs:
    emissions_kg: number
    breakdown: object
    confidence_score:
      type: number
      minimum: 0
      maximum: 1
  ```

#### `tags` (optional, array of strings)
- **Description:** Metadata tags for policy enforcement and execution hints
- **Common tags:**
  - `"network:*"` - Requires network access
  - `"network:era5"` - Specific endpoint access
  - `"compute:intensive"` - High CPU/memory requirements
  - `"external:api"` - Calls external APIs
  - `"deterministic"` - Guaranteed deterministic execution
- **Example:**
  ```yaml
  tags:
    - "network:era5"
    - "compute:intensive"
    - "external:weather-api"
  ```

## Reference Syntax

GL Pipelines support a flexible reference syntax for dynamic value resolution:

### Pipeline Inputs
```yaml
# Reference pipeline input
building_size: "${inputs.building_size_sqft}"

# With default fallback
location: "${inputs.location || vars.default_region}"
```

### Constants (vars)
```yaml
# Simple variable reference
grid_factor: "${vars.grid_intensity}"

# Nested object access
region_config: "${vars.regions.north.grid_factor}"
```

### Step Outputs
```yaml
# Reference specific output from a step
emissions_data: "${steps.carbon_calc.emissions_kg}"

# Reference entire step output
all_data: "${steps.validate.outputs}"

# Chain references
report_input: "${steps.carbon_calc.breakdown.by_fuel}"
```

### Environment Variables
```yaml
# Access environment variables
api_key: "${env.WEATHER_API_KEY}"
region: "${env.GL_REGION}"

# With defaults
timeout: "${env.REQUEST_TIMEOUT || 30}"
```

### System Values
```yaml
# Built-in system values
timestamp: "${system.timestamp}"
execution_id: "${system.execution_id}"
pipeline_version: "${system.pipeline_version}"
```

### Expression Examples
```yaml
# Arithmetic
total_area: "${inputs.length} * ${inputs.width}"

# String concatenation
report_title: "Emissions Report - ${vars.reporting_year}"

# Conditional expressions
grid_factor: "${vars.region} == 'IN-North' ? 0.82 : 0.75"
```

## Error Handling & Control Flow

### Step Dependencies
Steps execute in definition order with implicit dependencies through output references:

```yaml
steps:
  - id: step_a
    agent: AgentA

  - id: step_b
    agent: AgentB
    in:
      data: "${steps.step_a.result}"  # Creates implicit dependency
```

### Error Strategies

#### `stop` (default)
Pipeline execution halts immediately on step failure.

#### `skip`
Skip failed step, continue with remaining steps. Step outputs are `null`.

#### `continue`
Continue execution, but mark pipeline as partially failed.

#### `retry`
Retry the step according to `retry` configuration, then apply fallback strategy.

### Conditional Execution
Use `when` conditions to create branching logic:

```yaml
- id: detailed_analysis
  agent: DetailedAnalyzer
  when: "${inputs.analysis_level} == 'detailed'"

- id: quick_analysis
  agent: QuickAnalyzer
  when: "${inputs.analysis_level} == 'quick'"
```

## Determinism & Execution Rules

### Deterministic Execution
All GL pipelines must be deterministic by default:

1. **Same inputs → Same outputs**: Identical input parameters must always produce identical results
2. **No random seeds**: Random number generation requires explicit seeds from `vars` or `inputs`
3. **Timestamp handling**: Use `vars.analysis_date` instead of `system.timestamp` for calculations
4. **External data**: Cache external API responses; use versioned datasets

### Non-Deterministic Operations
Mark steps as non-deterministic using tags:

```yaml
- id: fetch_weather
  agent: WeatherAgent
  tags: ["non-deterministic", "network:weather-api"]
  with:
    cache_duration: 3600  # Cache for reproducibility
```

### Execution Environment
- **Isolation**: Each step runs in isolated context
- **State persistence**: Only declared outputs persist between steps
- **Resource limits**: Memory and CPU constraints enforced per `policy` configuration

## Security & Policy Requirements

### Network Access Control
Steps requiring network access must be explicitly tagged:

```yaml
- id: fetch_emission_factors
  agent: EmissionFactorAgent
  tags: ["network:emission-database"]
  in:
    endpoint: "https://api.emissions-db.org/v1/factors"
```

### Data Residency
Enforce data residency through pack-level policies:

```yaml
# In pack.yaml
policy:
  network: ["emission-database:*", "era5:*"]
  data_residency: ["IN", "EU"]
```

### Audit Trail
All pipeline executions generate audit logs with:
- Input parameters and their sources
- Step execution order and timing
- Output provenance and lineage
- Network requests and data sources
- Agent versions and configurations

## Examples

### Minimal Pipeline

```yaml
name: simple-emissions
version: 1

steps:
  - id: calculate
    agent: "greenlang.agents.carbon:BasicCalculator"
    in:
      fuel_consumption: 1000
      fuel_type: "natural_gas"
```

### Full-Featured Pipeline

```yaml
name: comprehensive-building-analysis
version: 2

vars:
  grid_intensity: 0.82
  reporting_year: 2024
  analysis_precision: "high"

inputs:
  building_data:
    type: object
    required: true
    schema:
      properties:
        size_sqft: {type: number, minimum: 100}
        location: {type: string}
        fuel_usage: {type: object}

artifacts_dir: "analysis_output/"

steps:
  - id: validate_inputs
    agent: "@validation-pack/BuildingDataValidator"
    action: "validate_comprehensive"
    in:
      building_data: "${inputs.building_data}"
      validation_rules: "${vars.analysis_precision}"
    outputs:
      validated_data: object
      validation_report: object
    on_error: "stop"

  - id: fetch_emission_factors
    agent: "agents/emission_factors.py:EmissionFactorService"
    action: "get_factors_for_location"
    in:
      location: "${steps.validate_inputs.validated_data.location}"
      vintage_year: "${vars.reporting_year}"
    with:
      cache_duration: 86400
      precision: "${vars.analysis_precision}"
    retry:
      max: 3
      backoff_seconds: 2
    tags: ["network:emission-database", "deterministic"]
    outputs:
      emission_factors: object
      data_vintage: string

  - id: calculate_electricity_emissions
    agent: "greenlang.agents.carbon:ElectricityEmissionCalculator"
    in:
      consumption_kwh: "${steps.validate_inputs.validated_data.fuel_usage.electricity}"
      grid_factor: "${steps.fetch_emission_factors.emission_factors.electricity}"
      region_factor: "${vars.grid_intensity}"
    outputs:
      emissions_kg: number
      methodology: string

  - id: calculate_gas_emissions
    agent: "greenlang.agents.carbon:FuelEmissionCalculator"
    when: "${steps.validate_inputs.validated_data.fuel_usage.natural_gas} > 0"
    in:
      consumption: "${steps.validate_inputs.validated_data.fuel_usage.natural_gas}"
      emission_factor: "${steps.fetch_emission_factors.emission_factors.natural_gas}"
    outputs:
      emissions_kg: number
      heating_value_used: number

  - id: aggregate_emissions
    agent: "greenlang.agents.carbon:EmissionAggregator"
    in:
      electricity_emissions: "${steps.calculate_electricity_emissions.emissions_kg}"
      gas_emissions: "${steps.calculate_gas_emissions.emissions_kg || 0}"
      building_area: "${steps.validate_inputs.validated_data.size_sqft}"
    outputs:
      total_emissions_kg: number
      emissions_intensity: number
      breakdown: object

  - id: benchmark_performance
    agent: "@benchmarking-pack/BuildingBenchmark"
    in:
      emissions_intensity: "${steps.aggregate_emissions.emissions_intensity}"
      building_type: "${steps.validate_inputs.validated_data.building_type}"
      location: "${steps.validate_inputs.validated_data.location}"
    outputs:
      benchmark_score: number
      percentile_ranking: number
      peer_comparison: object

  - id: generate_report
    agent: "greenlang.agents.reporting:ComprehensiveReportGenerator"
    action: "generate_cfo_brief"
    in:
      emissions_data: "${steps.aggregate_emissions}"
      benchmark_data: "${steps.benchmark_performance}"
      building_info: "${steps.validate_inputs.validated_data}"
      analysis_metadata:
        pipeline_version: "${system.pipeline_version}"
        execution_date: "${vars.reporting_year}-12-31"
        data_sources: "${steps.fetch_emission_factors.data_vintage}"
    with:
      template: "reports/executive_summary.html.j2"
      include_charts: true
      format: ["html", "pdf"]
    outputs:
      report_html: string
      report_pdf: string
      summary_data: object

outputs:
  total_co2e_kg: "${steps.aggregate_emissions.total_emissions_kg}"
  intensity_kg_per_sqft: "${steps.aggregate_emissions.emissions_intensity}"
  benchmark_percentile: "${steps.benchmark_performance.percentile_ranking}"
  executive_report: "${steps.generate_report.report_pdf}"
  full_breakdown: "${steps.aggregate_emissions.breakdown}"
  data_provenance:
    emission_factors: "${steps.fetch_emission_factors.data_vintage}"
    validation_report: "${steps.validate_inputs.validation_report}"
    execution_metadata:
      pipeline_version: 2
      completed_at: "${system.timestamp}"
```

## Validation Rules

### Required Validation (MUST PASS)
1. **Schema compliance**: Pipeline structure matches specification
2. **Unique step IDs**: All step `id` fields must be unique
3. **Valid references**: All `${...}` references must resolve
4. **Agent resolution**: All `agent` specifications must be resolvable
5. **Mutual exclusivity**: Steps cannot have both `in` and `inputsRef`
6. **Circular dependencies**: No circular references in step outputs

### Recommended Validation (WARNINGS)
1. **Unused inputs**: Warn about `inputs` not referenced in steps
2. **Missing outputs**: Warn if steps don't declare expected outputs
3. **Determinism**: Warn about potentially non-deterministic operations
4. **Network tags**: Warn if network operations lack appropriate tags
5. **Error handling**: Recommend explicit `on_error` strategies for critical steps

### Runtime Validation
1. **Input validation**: Validate `inputs` against declared schemas
2. **Output validation**: Verify step outputs match declared types
3. **Policy compliance**: Enforce network and data residency policies
4. **Resource limits**: Monitor memory and CPU usage per policy

## CLI Usage

### Pipeline Validation
```bash
# Validate pipeline syntax and references
gl pipeline validate gl.yaml

# Validate with strict mode (warnings as errors)
gl pipeline validate --strict gl.yaml

# Validate with external input schema
gl pipeline validate gl.yaml --inputs inputs.json
```

### Pipeline Execution
```bash
# Run pipeline with default inputs
gl run gl.yaml

# Run with external inputs
gl run gl.yaml --inputs building_data.json

# Run with variable overrides
gl run gl.yaml --vars grid_intensity=0.85,reporting_year=2025

# Run with debug output
gl run gl.yaml --debug --artifacts-dir debug_output/
```

### Pipeline Inspection
```bash
# Show pipeline structure
gl pipeline inspect gl.yaml

# Show dependency graph
gl pipeline graph gl.yaml

# Estimate execution resources
gl pipeline estimate gl.yaml --inputs inputs.json
```

## Migration Guide

### From Pre-v1 Pipelines

#### Breaking Changes
1. **Required `id` field**: All steps must have unique `id` instead of `name`
2. **Agent specification**: Use explicit agent paths instead of `agent_id`
3. **Input format**: Use `in` object instead of `inputs` array
4. **Reference syntax**: Update to `${...}` format from `$var` format

#### Migration Steps

1. **Update step identifiers:**
   ```yaml
   # OLD (Pre-v1)
   - name: calculate_emissions
     agent_id: carbon

   # NEW (v1.0)
   - id: calculate_emissions
     agent: "greenlang.agents.carbon:CarbonCalculator"
   ```

2. **Update input format:**
   ```yaml
   # OLD
   - name: step1
     inputs:
       - building_size: 5000
       - location: "IN-North"

   # NEW
   - id: step1
     in:
       building_size: 5000
       location: "IN-North"
   ```

3. **Update reference syntax:**
   ```yaml
   # OLD
   previous_data: "$steps.step1.outputs"

   # NEW
   previous_data: "${steps.step1.outputs}"
   ```

### Automated Migration
Use the provided migration script:

```bash
# Migrate pipeline file
python scripts/migrate_pipeline_v1.py gl.yaml

# Preview changes without modification
python scripts/migrate_pipeline_v1.py gl.yaml --dry-run
```

## JSON Schema Reference

The formal JSON Schema for pipeline validation is available at:
- **Local:** `schemas/pipeline.schema.v1.json` (when available)
- **URL:** `https://greenlang.io/schema/pipeline.v1.json` (future)

## Backward Compatibility

### Reading Legacy Pipelines
The GL runtime will attempt to read pre-v1 pipelines with these adaptations:
- Map `name` to `id` in steps
- Convert `agent_id` references to full agent specifications
- Transform `inputs` arrays to `in` objects
- Update reference syntax automatically

### Forward Compatibility
New fields added in v1.x will be:
- **Optional**: Won't break existing v1.0 pipelines
- **Preserved**: Unknown fields maintained during processing
- **Validated**: New fields follow specification constraints

## References

- [Pack Schema v1.0](./PACK_SCHEMA_V1.md)
- [GreenLang CLI Documentation](./CLI.md)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [JSON Schema Draft 2020-12](https://json-schema.org/)
- [YAML Specification 1.2](https://yaml.org/spec/1.2/)