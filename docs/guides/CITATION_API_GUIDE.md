# Citation API Guide

This guide explains how to access and use citation data from GreenLang AI agents.

## Overview

All 11 GreenLang AI agents now include citation tracking to provide complete transparency and auditability for all calculations and data sources.

## Citation Types

### 1. Emission Factor Citations

Track emission factors used in calculations:

```python
{
    "source": "EPA GHG Inventory",
    "factor_name": "Natural Gas Combustion",
    "value": 53.06,
    "unit": "kg CO2e/MMBtu",
    "ef_cid": "ef_428b1c64829dc8f5",  # Emission Factor Content ID
    "version": "2023",
    "confidence": "high",
    "region": "US",
    "gwp_set": "AR5"
}
```

### 2. Calculation Citations

Track intermediate calculations:

```python
{
    "step_name": "calculate_emissions",
    "formula": "Emissions = Amount × EF",
    "inputs": {"amount": 1000, "ef": 53.06},
    "output": {"emissions_kg": 53060},
    "timestamp": "2025-10-26T10:30:00",
    "tool_call_id": "calc_1"
}
```

### 3. Data Source Citations

Track external data sources:

```python
{
    "source_name": "Grid Intensity Database",
    "source_type": "database",
    "query": {"region": "US"},
    "timestamp": "2025-10-26T10:30:00"
}
```

## Agent Coverage

All 11 AI agents include citation tracking:

| Agent | Citation Types | Example Use Case |
|-------|---------------|-----------------|
| FuelAgentAI | Emission Factors | Natural gas combustion EF |
| CarbonAgentAI | EF + Calculations | Aggregated emissions with formula |
| GridFactorAgentAI | EF + Data Sources | Grid intensity lookup |
| BoilerReplacementAgent_AI | Calculations | Efficiency degradation formula |
| IndustrialProcessHeatAgent_AI | Calculations | Heat demand analysis |
| DecarbonizationRoadmapAgentAI | Calculations | Roadmap modeling |
| IndustrialHeatPumpAgent_AI | Calculations | COP calculation |
| RecommendationAgentAI | Calculations | ROI analysis |
| ReportAgentAI | Calculations | YoY trend analysis |
| IsolationForestAnomalyAgent | Calculations | ML anomaly scoring |
| SARIMAForecastAgent | Calculations | Model evaluation (RMSE, MAE, MAPE) |

## Accessing Citations

### Python API

```python
from greenlang.agents.fuel_agent_ai import FuelAgentAI

# Create agent
agent = FuelAgentAI()

# Run calculation
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
    "country": "US"
})

# Access citations
if result.success:
    citations = result.data.get("citations", [])

    # For FuelAgentAI: list of emission factor citations
    for citation in citations:
        print(f"Source: {citation['source']}")
        print(f"EF CID: {citation['ef_cid']}")
        print(f"Value: {citation['value']} {citation['unit']}")
```

### For Agents with Multiple Citation Types

```python
from greenlang.agents.carbon_agent_ai import CarbonAgentAI

agent = CarbonAgentAI()
result = agent.run({...})

if result.success:
    citations = result.data.get("citations", {})

    # Emission factor citations
    ef_citations = citations.get("emission_factors", [])
    for ef in ef_citations:
        print(f"EF: {ef['factor_name']} = {ef['value']} {ef['unit']}")

    # Calculation citations
    calc_citations = citations.get("calculations", [])
    for calc in calc_citations:
        print(f"Step: {calc['step_name']}")
        print(f"Formula: {calc['formula']}")
        print(f"Inputs: {calc['inputs']}")
        print(f"Output: {calc['output']}")
```

## EF CID (Emission Factor Content ID)

The EF CID is a deterministic hash that uniquely identifies an emission factor:

### Format
```
ef_<16-character-hex>
```

### Examples
```
ef_428b1c64829dc8f5
ef_7a2b9c1d3e4f5678
```

### Properties
- **Deterministic**: Same inputs always produce the same CID
- **Unique**: Different emission factors have different CIDs
- **Verifiable**: Can be used to verify data integrity

### Generation

```python
from greenlang.agents.citations import create_emission_factor_citation

citation = create_emission_factor_citation(
    source="EPA",
    factor_name="Natural Gas",
    value=53.06,
    unit="kg CO2e/MMBtu",
    version="2023",
    region="US"
)

# EF CID is automatically generated
print(citation.ef_cid)  # e.g., "ef_428b1c64829dc8f5"
```

## Citation Output Format

### Agents with Only Emission Factor Citations
```python
{
    "result": {...},
    "citations": [
        {
            "source": "...",
            "factor_name": "...",
            "value": 53.06,
            "unit": "...",
            "ef_cid": "ef_...",
            ...
        }
    ]
}
```

### Agents with Multiple Citation Types
```python
{
    "result": {...},
    "citations": {
        "emission_factors": [
            {...}
        ],
        "calculations": [
            {
                "step_name": "...",
                "formula": "...",
                "inputs": {...},
                "output": {...},
                ...
            }
        ],
        "data_sources": [
            {...}
        ]
    }
}
```

## Schema Validation

All agent outputs support optional citation fields using TypedDict `NotRequired`:

```python
from greenlang.agents.types import FuelOutput

class FuelOutput(TypedDict):
    co2e_emissions_kg: float
    ...
    citations: NotRequired[list]  # Optional citations field
```

This ensures backward compatibility while enabling citation tracking.

## Use Cases

### 1. Audit Trail
```python
# Track all emission factors used in a calculation
result = fuel_agent.run({...})
for citation in result.data["citations"]:
    audit_log.append({
        "timestamp": datetime.now(),
        "ef_cid": citation["ef_cid"],
        "source": citation["source"],
        "value": citation["value"]
    })
```

### 2. Verification
```python
# Verify emission factor hasn't changed
expected_cid = "ef_428b1c64829dc8f5"
actual_cid = result.data["citations"][0]["ef_cid"]

if expected_cid != actual_cid:
    raise ValueError("Emission factor has changed!")
```

### 3. Reproducibility
```python
# Store citations for reproducibility
calculation_record = {
    "timestamp": datetime.now(),
    "input": input_data,
    "result": result.data,
    "citations": result.data["citations"]
}

# Later: verify same inputs produce same results
# by comparing EF CIDs
```

### 4. Reporting
```python
# Generate citation section for reports
citations = result.data["citations"]["emission_factors"]

report = f"""
## Data Sources

This calculation used the following emission factors:

{chr(10).join([
    f"- {c['factor_name']}: {c['value']} {c['unit']} "
    f"(Source: {c['source']}, Version: {c['version']}, "
    f"EF CID: {c['ef_cid']})"
    for c in citations
])}
"""
```

## Best Practices

### 1. Always Check for Citations
```python
citations = result.data.get("citations", [])
if not citations:
    logger.warning("No citations available")
```

### 2. Store Citations with Results
```python
# Don't just store the result - store citations too
database.save({
    "calculation_id": uuid.uuid4(),
    "result": result.data,
    "citations": result.data.get("citations", []),
    "timestamp": datetime.now()
})
```

### 3. Use EF CIDs for Version Control
```python
# Track when emission factors change
if current_ef_cid != previous_ef_cid:
    notify_stakeholders(
        "Emission factor updated",
        old=previous_ef_cid,
        new=current_ef_cid
    )
```

### 4. Include Citations in API Responses
```python
# REST API example
@app.post("/calculate-emissions")
def calculate_emissions(request: EmissionRequest):
    result = fuel_agent.run(request.dict())

    return {
        "emissions_kg": result.data["co2e_emissions_kg"],
        "citations": result.data.get("citations", []),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0"
        }
    }
```

## Migration Guide

### For Existing Code

Citations are backward compatible - existing code will continue to work:

```python
# Old code (still works)
result = agent.run({...})
emissions = result.data["co2e_emissions_kg"]

# New code (with citations)
result = agent.run({...})
emissions = result.data["co2e_emissions_kg"]
citations = result.data.get("citations", [])  # Optional
```

### Adding Citation Support

```python
# Before: No citations
def calculate_emissions(fuel, amount):
    result = fuel_agent.run({"fuel_type": fuel, "amount": amount})
    return result.data["co2e_emissions_kg"]

# After: With citations
def calculate_emissions_with_provenance(fuel, amount):
    result = fuel_agent.run({"fuel_type": fuel, "amount": amount})
    return {
        "emissions": result.data["co2e_emissions_kg"],
        "citations": result.data.get("citations", []),
        "ef_cids": [c["ef_cid"] for c in result.data.get("citations", [])]
    }
```

## Testing

### Verify Citations Exist

```python
def test_agent_includes_citations():
    agent = FuelAgentAI()
    result = agent.run({
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US"
    })

    assert "citations" in result.data
    assert len(result.data["citations"]) > 0

    # Verify EF CID format
    ef_cid = result.data["citations"][0]["ef_cid"]
    assert ef_cid.startswith("ef_")
    assert len(ef_cid) == 19  # "ef_" + 16 hex chars
```

### Verify EF CID Determinism

```python
def test_ef_cid_determinism():
    from greenlang.agents.citations import create_emission_factor_citation

    # Same inputs
    cit1 = create_emission_factor_citation(
        source="EPA", factor_name="NG", value=53.06,
        unit="kg CO2e/MMBtu", version="2023", region="US"
    )

    cit2 = create_emission_factor_citation(
        source="EPA", factor_name="NG", value=53.06,
        unit="kg CO2e/MMBtu", version="2023", region="US"
    )

    # Should produce same EF CID
    assert cit1.ef_cid == cit2.ef_cid
```

## Support

For questions or issues with citations:
- Check tests: `tests/agents/test_citations.py`
- View implementation: `greenlang/agents/citations.py`
- Review integration examples in agent source files

## Version History

- **v0.3.0** (2025-10-26): Citation integration complete for all 11 agents
- **v0.1.0** (2025-10-26): Initial citation infrastructure

---

**Last Updated**: 2025-10-26
**Status**: Production Ready
**Coverage**: 11/11 AI Agents (100%)
