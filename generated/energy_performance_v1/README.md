# Building Energy Performance Calculator

> Calculates Energy Use Intensity (EUI) and checks compliance with Building Performance Standards for urban decarbonization.


**Version:** 1.0.0
**License:** Apache-2.0
**ID:** buildings/energy_performance_v1

## Overview

This agent was generated from an AgentSpec YAML specification using the GreenLang Agent Generator.

## Installation

```bash
pip install -e .
```

## Usage

```python
from energy_performance_v1 import BuildingEnergyPerformanceCalculatorAgent

# Initialize agent
agent = BuildingEnergyPerformanceCalculatorAgent()

# Run agent
result = await agent.run({
    # Your input data here
})

print(result.output)
print(result.provenance)
```

## Tools

This agent includes the following tools:

### calculate_eui

Calculate Energy Use Intensity (kWh per sqm per year)

**Parameters:**
- `energy_consumption_kwh`: number (required)
- `floor_area_sqm`: number (required)

### lookup_bps_threshold

Look up BPS threshold for building type

**Parameters:**
- `building_type`: string (required)
- `climate_zone`: string

### check_bps_compliance

Check if building meets BPS threshold

**Parameters:**
- `actual_eui`: number (required)
- `threshold_eui`: number (required)


## Tests

Run tests with:

```bash
pytest tests/
```

## Provenance

This agent tracks complete provenance for audit trails including:
- Input/output hashes (SHA-256)
- Tool call records
- Emission factor citations
- Timestamp tracking

---

Generated with GreenLang Agent Generator v1.0.0
