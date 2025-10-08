# GreenLang AgentSpec v2 - Example Pack

This directory contains a complete, production-ready example of a GreenLang Agent Pack using **AgentSpec v2**.

## Overview

**Agent:** Natural Gas Boiler Emissions Calculator
**ID:** `buildings/boiler_ng_v1`
**Version:** `2.1.3`
**Purpose:** Calculates CO2 equivalent emissions from natural gas boiler fuel consumption using the Lower Heating Value (LHV) method.

## File Structure

```
examples/agentspec_v2/
├── README.md              # This file
├── pack.yaml              # Complete AgentSpec v2 manifest
└── test_example.py        # Script to validate the example
```

## AgentSpec v2 Sections

This example demonstrates **all sections** of AgentSpec v2:

### 1. **Metadata**
- `schema_version`: `"2.0.0"` (required)
- `id`: Unique agent identifier (`buildings/boiler_ng_v1`)
- `name`: Human-readable name
- `version`: Semantic versioning (`2.1.3`)
- `summary`: Brief description
- `tags`: Categorization (`buildings`, `combustion`, `scope1`)
- `owners`: Team ownership (`@gl/industry-buildings`)
- `license`: SPDX license identifier (`Apache-2.0`)

### 2. **Compute Section**
Defines the computational logic:
- **Entrypoint**: `python://gl.agents.boiler.ng:compute`
- **Deterministic**: `true` (reproducible)
- **Dependencies**: Python packages with versions (e.g., `pandas==2.1.4`)
- **Python Version**: `3.11`
- **Performance**: `timeout_s: 30`, `memory_limit_mb: 512`
- **Inputs**: Fuel volume, LHV, efficiency, region
- **Outputs**: CO2e emissions, energy, intensity
- **Factors**: IPCC AR6 natural gas combustion factor

### 3. **AI Section**
LLM configuration and tools:
- **JSON Mode**: `true` (structured output)
- **System Prompt**: Climate advisor instructions
- **Budget**: Max cost `$1.00`, max tokens `15K/2K`
- **RAG Collections**: GHG Protocol, IPCC AR6, GreenLang docs
- **Tools**: `select_emission_factor`, `validate_input`

### 4. **Realtime Section**
Data streaming configuration:
- **Default Mode**: `replay` (deterministic)
- **Snapshot Path**: Cached data for replay mode
- **Connectors**: Grid intensity, weather data

### 5. **Provenance Section**
Reproducibility and audit:
- **Pin EF**: `true` (immutable factors)
- **GWP Set**: `AR6GWP100` (IPCC AR6 100-year GWP)
- **Record Fields**: Inputs, outputs, factors, code SHA, timestamp

### 6. **Security Section** (P1 enhancement)
Network egress control:
- **Allowlist Hosts**: IPCC, EPA, EIA APIs
- **Block on Violation**: `true` (hard fail)

### 7. **Tests Section** (optional)
Validation tests:
- **Golden Tests**: 3 scenarios (baseline, high efficiency, low efficiency)
- **Property Tests**: 5 invariants (monotonicity, non-negativity, energy balance)

## Usage

### Validate the Example

```python
from greenlang.specs import from_yaml

# Load and validate
spec = from_yaml("examples/agentspec_v2/pack.yaml")
print(f"✓ Loaded: {spec.name} v{spec.version}")
print(f"✓ Entrypoint: {spec.compute.entrypoint}")
print(f"✓ Inputs: {list(spec.compute.inputs.keys())}")
print(f"✓ Outputs: {list(spec.compute.outputs.keys())}")
```

### Export to JSON Schema

```python
from greenlang.specs import to_json_schema
import json

schema = to_json_schema()
print(json.dumps(schema, indent=2))
```

### CLI Validation

```bash
# Validate spec
gl spec validate examples/agentspec_v2/pack.yaml

# Show spec details
gl spec show examples/agentspec_v2/pack.yaml

# Export to JSON
gl spec export examples/agentspec_v2/pack.yaml --format json
```

## Key Features Demonstrated

### ✅ Climate-Specific Units
All units are validated against the climate units whitelist:
- `m^3` (cubic meters)
- `MJ/m^3` (megajoules per cubic meter)
- `kgCO2e` (kilograms CO2 equivalent)
- `kgCO2e/MJ` (emission intensity)

### ✅ Reproducibility
- Deterministic compute: `deterministic: true`
- Pinned dependencies: `pandas==2.1.4`, `numpy==1.26.0`
- Pinned emission factors: `pin_ef: true`
- Pinned GWP set: `gwp_set: "AR6GWP100"`
- Snapshot data: `snapshot_path` for replay mode

### ✅ Security
- Safe tools: `safe: true` with AST validation
- URI validation: `python://` and `ef://` schemes
- Network egress control: Allowlisted hosts only
- No raw numerics: All outputs unit-aware

### ✅ AI Integration
- Structured output: `json_mode: true`
- Cost controls: Budget constraints
- RAG retrieval: Document citations
- Function calling: 2 tools with JSON Schema validation

### ✅ Validation
- 3 golden tests with tolerance checks
- 5 property-based invariants
- All constraints enforced (ge, gt, le, lt, enum)

## Testing

Run the example validation:

```bash
# Python
python examples/agentspec_v2/test_example.py

# Pytest
pytest examples/agentspec_v2/test_example.py -v
```

## Compliance

This example follows:
- **GHG Protocol Corporate Standard** (Scope 1 emissions)
- **IPCC AR6 Guidelines** (emission factors, GWP100)
- **ISO 14064-1:2018** (GHG quantification and reporting)
- **Semantic Versioning 2.0.0** (version management)
- **JSON Schema draft-2020-12** (schema validation)

## References

- [GreenLang AgentSpec v2 Documentation](../../docs/specs/agentspec_v2.md)
- [IPCC AR6 Guidelines](https://www.ipcc.ch/report/ar6/)
- [GHG Protocol Corporate Standard](https://ghgprotocol.org/corporate-standard)
- [Semantic Versioning](https://semver.org/)

## License

Apache-2.0 (as specified in `pack.yaml`)
