# EUDR Deforestation Compliance Agent

> Validates supply chain compliance with the EU Deforestation Regulation (EU) 2023/1115. Covers 7 regulated commodities: cattle, cocoa, coffee, palm oil, rubber, soya, and wood. Ensures products are deforestation-free (cutoff date: December 31, 2020) and legally produced. Generates EU Due Diligence Statements (DDS) for regulatory submission.


**Version:** 1.0.0
**License:** Apache-2.0
**ID:** regulatory/eudr_compliance_v1

## Overview

This agent was generated from an AgentSpec YAML specification using the GreenLang Agent Generator.

## Installation

```bash
pip install -e .
```

## Usage

```python
from eudr_compliance_v1 import EudrDeforestationComplianceAgentAgent

# Initialize agent
agent = EudrDeforestationComplianceAgentAgent()

# Run agent
result = await agent.run({
    # Your input data here
})

print(result.output)
print(result.provenance)
```

## Tools

This agent includes the following tools:

### validate_geolocation

Validates GPS coordinates or polygon data for EUDR compliance. Checks that coordinates are within valid bounds, not in known protected forest areas, and can be traced to a specific production plot. This is a DETERMINISTIC validation - same inputs always return same validation results.


**Parameters:**
- `coordinates`: array (required)
- `coordinate_type`: string
- `country_code`: string (required)
- `precision_meters`: number

### classify_commodity

Classifies commodities and derived products under EUDR using EU Combined Nomenclature (CN) codes. Identifies which of the 7 regulated commodities the product falls under and its specific CN code classification. DETERMINISTIC classification based on EU CN code database.


**Parameters:**
- `cn_code`: string (required)
- `product_description`: string
- `quantity_kg`: number

### assess_country_risk

Assesses deforestation risk for a specific country and region based on EC benchmarking system, FAO forest data, and Global Forest Watch. Returns risk level (low/standard/high) and determines due diligence requirements. DETERMINISTIC assessment based on official risk databases.


**Parameters:**
- `country_code`: string (required)
- `region`: string
- `commodity_type`: string (required)
- `production_year`: integer

### trace_supply_chain

Traces commodity supply chain from production plot to final product. Calculates traceability score and identifies gaps in documentation. Used for generating traceability maps required by EUDR.


**Parameters:**
- `shipment_id`: string (required)
- `supply_chain_nodes`: array (required)
- `commodity_type`: string (required)

### generate_dds_report

Generates EU Due Diligence Statement (DDS) for submission to the EU Information System. Validates all required fields and produces compliant JSON/XML output.


**Parameters:**
- `operator_info`: object (required)
- `commodity_data`: object (required)
- `geolocation_data`: object (required)
- `risk_assessment`: object (required)
- `traceability_data`: object


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
