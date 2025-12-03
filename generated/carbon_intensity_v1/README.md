# CBAM Carbon Intensity Calculator

> Calculates carbon intensity for CBAM-regulated goods (steel, cement, aluminum, fertilizers) and determines CBAM certificate requirements for EU imports.


**Version:** 1.0.0
**License:** Apache-2.0
**ID:** cbam/carbon_intensity_v1

## Overview

This agent was generated from an AgentSpec YAML specification using the GreenLang Agent Generator.

## Installation

```bash
pip install -e .
```

## Usage

```python
from carbon_intensity_v1 import CbamCarbonIntensityCalculatorAgent

# Initialize agent
agent = CbamCarbonIntensityCalculatorAgent()

# Run agent
result = await agent.run({
    # Your input data here
})

print(result.output)
print(result.provenance)
```

## Tools

This agent includes the following tools:

### lookup_cbam_benchmark

Look up CBAM default benchmark values

**Parameters:**
- `product_type`: string (required)

### calculate_carbon_intensity

Calculate emissions per tonne of product

**Parameters:**
- `total_emissions`: number (required)
- `production_quantity`: number (required)


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
