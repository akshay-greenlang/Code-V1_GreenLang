# FuelAgentAI v2 API Documentation

**Version:** 2.0.0
**Date:** October 2025
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in v2](#whats-new-in-v2)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Response Formats](#response-formats)
6. [Multi-Gas Breakdown](#multi-gas-breakdown)
7. [Emission Boundaries (WTT, WTW)](#emission-boundaries-wtt-wtw)
8. [Provenance Tracking](#provenance-tracking)
9. [Data Quality Scoring](#data-quality-scoring)
10. [Performance Optimization](#performance-optimization)
11. [Compliance Reporting](#compliance-reporting)
12. [Error Handling](#error-handling)
13. [Best Practices](#best-practices)

---

## Overview

FuelAgentAI v2 is a **backward-compatible** enhancement of the original FuelAgent with advanced features for corporate sustainability reporting.

### Key Features

✅ **Multi-Gas Breakdown**: CO2, CH4, N2O reported separately
✅ **Full Provenance**: Source attribution for audit trails
✅ **Data Quality Scoring**: 5-dimension DQS per GHGP standard
✅ **Compliance Ready**: CSRD, CDP, GRI 305 compliant
✅ **Backward Compatible**: v1 clients work unchanged
✅ **Cost Optimized**: 20% cheaper than v1 (fast path optimization)

---

## What's New in v2

### Enhanced Input Parameters

```python
{
    # v1 parameters (unchanged)
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "country": "US",

    # v2 enhancements
    "scope": "1",                          # GHG scope (1, 2, or 3)
    "boundary": "WTW",                     # combustion | WTT | WTW
    "gwp_set": "IPCC_AR6_100",            # GWP reference set
    "response_format": "enhanced"          # legacy | enhanced | compact
}
```

### Enhanced Output Structure

```python
{
    # v1 fields (backward compatible)
    "co2e_emissions_kg": 10210.0,
    "emission_factor": 10.21,

    # v2 enhancements
    "vectors_kg": {
        "CO2": 10180.0,
        "CH4": 0.82,
        "N2O": 0.47
    },
    "factor_record": {
        "factor_id": "EF:US:diesel:2024:v1",
        "source_org": "EPA",
        "citation": "EPA (2024), GHG Emission Factors Hub..."
    },
    "quality": {
        "dqs": {"overall_score": 4.8, "rating": "Excellent"},
        "uncertainty_95ci_pct": 8.5
    }
}
```

---

## Quick Start

### Installation

```bash
pip install greenlang
```

### v1 Client (Unchanged Behavior)

```python
from greenlang.agents import FuelAgentAI_v2

# v1 client continues working unchanged
agent = FuelAgentAI_v2()

result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
})

print(result["data"]["co2e_emissions_kg"])  # 10210.0
```

### v2 Client (Enhanced Features)

```python
from greenlang.agents import FuelAgentAI_v2

agent = FuelAgentAI_v2()

result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "scope": "1",
    "boundary": "WTW",                    # Full lifecycle
    "gwp_set": "IPCC_AR6_100",
    "response_format": "enhanced"          # Enable v2 features
})

# Access multi-gas breakdown
vectors = result["data"]["vectors_kg"]
print(f"CO2: {vectors['CO2']:.2f} kg")
print(f"CH4: {vectors['CH4']:.4f} kg")
print(f"N2O": {vectors['N2O']:.4f} kg")

# Access provenance
prov = result["data"]["factor_record"]
print(f"Source: {prov['source_org']}")
print(f"Citation: {prov['citation']}")

# Access quality
quality = result["data"]["quality"]
print(f"DQS: {quality['dqs']['overall_score']}/5.0")
print(f"Uncertainty: ±{quality['uncertainty_95ci_pct']:.1f}%")
```

---

## API Reference

### FuelAgentAI_v2 Class

#### Constructor

```python
FuelAgentAI_v2(
    budget_usd: float = 0.50,
    enable_explanations: bool = True,
    enable_recommendations: bool = True,
    enable_fast_path: bool = True
)
```

**Parameters:**
- `budget_usd`: Maximum USD to spend per calculation (default: $0.50)
- `enable_explanations`: Enable AI-generated explanations (default: True)
- `enable_recommendations`: Enable reduction recommendations (default: True)
- `enable_fast_path`: Enable fast path optimization for simple requests (default: True)

#### run() Method

```python
agent.run(payload: FuelInput) -> AgentResult[FuelOutput]
```

**Input Payload:**

| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `fuel_type` | string | ✅ | Fuel type (diesel, natural_gas, etc.) | - |
| `amount` | number | ✅ | Consumption amount | - |
| `unit` | string | ✅ | Unit (gallons, kWh, therms, etc.) | - |
| `country` | string | ❌ | ISO country code | "US" |
| `scope` | string | ❌ | GHG scope ("1", "2", "3") | "1" |
| `boundary` | string | ❌ | Emission boundary | "combustion" |
| `gwp_set` | string | ❌ | GWP reference set | "IPCC_AR6_100" |
| `response_format` | string | ❌ | Output format | "legacy" |
| `renewable_percentage` | number | ❌ | Renewable offset (0-100) | 0 |
| `efficiency` | number | ❌ | Equipment efficiency (0-1) | 1.0 |

**Return Value:**

```python
{
    "success": bool,
    "data": FuelOutput,  # See Response Formats section
    "metadata": {
        "agent_id": "fuel_ai_v2",
        "calculation_time_ms": 45.2,
        "execution_path": "fast",  # "fast" or "ai"
        "total_cost_usd": 0.0
    }
}
```

---

## Response Formats

### 1. Legacy Format (v1 Compatible)

**Use Case:** Backward compatibility for existing v1 clients

```python
{
    "fuel_type": "diesel",
    "amount": 1000,
    "response_format": "legacy"  # Default
}
```

**Output:**

```python
{
    "co2e_emissions_kg": 10210.0,
    "fuel_type": "diesel",
    "consumption_amount": 1000,
    "consumption_unit": "gallons",
    "emission_factor": 10.21,
    "emission_factor_unit": "kgCO2e/gallon",
    "country": "US",
    "scope": "1"
}
```

### 2. Enhanced Format (v2 Full Features)

**Use Case:** CSRD/CDP reporting, detailed analysis

```python
{
    "fuel_type": "diesel",
    "amount": 1000,
    "response_format": "enhanced"
}
```

**Output:**

```python
{
    # v1 fields (all preserved)
    "co2e_emissions_kg": 10210.0,
    "emission_factor": 10.21,
    ...

    # v2 enhancements
    "vectors_kg": {
        "CO2": 10180.0,
        "CH4": 0.82,
        "N2O": 0.47
    },
    "boundary": "combustion",
    "gwp_set": "IPCC_AR6_100",
    "factor_record": {
        "factor_id": "EF:US:diesel:2024:v1",
        "source_org": "EPA",
        "source_publication": "GHG Emission Factors Hub",
        "source_year": 2024,
        "methodology": "measured",
        "citation": "EPA (2024), GHG Emission Factors Hub, Table C-1..."
    },
    "quality": {
        "dqs": {
            "overall_score": 4.8,
            "rating": "Excellent",
            "temporal": 5,
            "geographical": 5,
            "technological": 5,
            "representativeness": 4,
            "methodological": 5
        },
        "uncertainty_95ci_pct": 8.5
    },
    "breakdown": {
        "effective_amount": 1000.0,
        "emission_factor_co2e": 10.21,
        "calculation": "1000.00 gallons × 10.2100 kgCO2e/gallon = 10210.00 kgCO2e"
    }
}
```

### 3. Compact Format (Mobile/IoT)

**Use Case:** Bandwidth-constrained environments

```python
{
    "fuel_type": "diesel",
    "amount": 1000,
    "response_format": "compact"
}
```

**Output:**

```python
{
    "co2e_kg": 10210.0,
    "fuel": "diesel",
    "quality_score": 4.8,
    "uncertainty_pct": 8.5
}
```

---

## Multi-Gas Breakdown

### Why Multi-Gas Matters

- **CSRD Compliance**: EU regulation requires CO2, CH4, N2O separate reporting
- **Accurate GWP**: Different gases have different global warming potentials
- **Policy Analysis**: Identify methane leakage opportunities
- **Transparency**: Show full emission profile

### GWP (Global Warming Potential)

| Gas | IPCC AR6 100-year | IPCC AR6 20-year |
|-----|-------------------|------------------|
| CO2 | 1.0 | 1.0 |
| CH4 | 27.9 | 81.2 |
| N2O | 273.0 | 273.0 |

### Example: Natural Gas Breakdown

```python
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
    "gwp_set": "IPCC_AR6_100",
    "response_format": "enhanced"
})

vectors = result["data"]["vectors_kg"]
# CO2: 5280 kg (99.6% of total CO2e)
# CH4: 0.36 kg (0.3% in mass, but 27.9× GWP = ~10 kg CO2e)
# N2O: 0.0053 kg (0.1% in mass, but 273× GWP = ~1.4 kg CO2e)

# Total CO2e ≈ 5280 + 10 + 1.4 = 5291 kg CO2e
```

### GWP Set Comparison

```python
# 100-year horizon (standard for IPCC, GHGP)
result_100 = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
    "gwp_set": "IPCC_AR6_100",
    "response_format": "enhanced"
})
# Total CO2e: 5,310 kg

# 20-year horizon (higher impact for CH4)
result_20 = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
    "gwp_set": "IPCC_AR6_20",
    "response_format": "enhanced"
})
# Total CO2e: 5,340 kg (+0.6% due to higher CH4 GWP)
```

---

## Emission Boundaries (WTT, WTW)

### What are Emission Boundaries?

FuelAgentAI v2 supports three emission boundaries for lifecycle analysis:

| Boundary | Description | Use Case |
|----------|-------------|----------|
| **combustion** | Direct emissions only (tank-to-wheel) | Standard scope 1/2 reporting |
| **WTT** | Upstream only (well-to-tank) | Assess supply chain emissions |
| **WTW** | Full lifecycle (WTT + combustion) | Complete carbon footprint |

### WTT (Well-to-Tank)

Upstream emissions from:
- **Extraction**: Crude oil drilling, natural gas extraction, coal mining
- **Processing/Refining**: Crude → diesel, natural gas processing
- **Transportation**: Pipeline, truck, rail
- **Distribution**: Local delivery

**Typical WTT Ratios:**
- Diesel: ~20% of combustion emissions
- Gasoline: ~18% of combustion
- Natural gas: ~18% (includes methane leakage)
- Coal: ~8% (lower upstream footprint)
- Electricity: ~8% (transmission & distribution losses)

### WTW (Well-to-Wheel)

Complete lifecycle emissions:
```
WTW = Combustion + WTT
```

### Example: Diesel Boundary Comparison

```python
from greenlang.agents import FuelAgentAI_v2

agent = FuelAgentAI_v2()

base_payload = {
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "country": "US",
    "response_format": "compact"
}

# 1. Combustion only (standard reporting)
combustion_result = agent.run({**base_payload, "boundary": "combustion"})
# Emissions: 10,210 kgCO2e (direct combustion)

# 2. WTT only (upstream)
wtt_result = agent.run({**base_payload, "boundary": "WTT"})
# Emissions: 2,040 kgCO2e (extraction + refining + transport)

# 3. WTW (full lifecycle)
wtw_result = agent.run({**base_payload, "boundary": "WTW"})
# Emissions: 12,250 kgCO2e (combustion + upstream)
```

**Result:**
```
Combustion: 10,210 kgCO2e (83%)
WTT:         2,040 kgCO2e (17%)
WTW:        12,250 kgCO2e (100% lifecycle)
```

### Example: Natural Gas with Methane Leakage

```python
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 10000,
    "unit": "therms",
    "country": "US",
    "boundary": "WTW",
    "response_format": "enhanced"
})

data = result["data"]

# WTW includes upstream methane leakage
print(f"Total WTW: {data['co2e_emissions_kg']:.0f} kgCO2e")
print(f"CH4 (includes leakage): {data['vectors_kg']['CH4']:.2f} kg")
# CH4 is significant in natural gas WTW due to upstream leakage
```

### Example: Electricity T&D Losses

```python
# Electricity WTT = Transmission & Distribution losses

result = agent.run({
    "fuel_type": "electricity",
    "amount": 10000,
    "unit": "kWh",
    "country": "US",
    "boundary": "WTT",
    "response_format": "compact"
})

# WTT emissions: ~290 kgCO2e (T&D losses ~8% of generation)
```

### When to Use Each Boundary

**Use `combustion` when:**
- Standard GHG Protocol Scope 1/2 reporting
- Regulatory compliance (EPA, CSRD, CDP)
- Direct emissions tracking

**Use `WTT` when:**
- Assessing supply chain emissions (Scope 3)
- Comparing upstream footprints of different fuels
- Evaluating supplier performance

**Use `WTW` when:**
- Full lifecycle carbon footprint analysis
- Product lifecycle assessment (LCA)
- Comprehensive sustainability reporting
- Comparing total impact of fuel alternatives

### WTT Data Sources

FuelAgentAI v2 uses the following WTT data sources:

- **US Factors**: GREET 2024 (Argonne National Laboratory)
- **UK Factors**: UK BEIS 2024 (DESNZ)
- **EU Factors**: JRC Well-to-Wheels 2024

### WTW Example with Provenance

```python
result = agent.run({
    "fuel_type": "diesel",
    "amount": 500,
    "unit": "gallons",
    "country": "US",
    "boundary": "WTW",
    "response_format": "enhanced"
})

prov = result["data"]["provenance"]

# Provenance tracks both combustion and WTT sources
print(prov["source_org"])  # "GreenLang"
print(prov["citation"])
# "Combustion: EPA; WTT: GREET 2024 - Petroleum diesel upstream"
```

---

## Provenance Tracking

### Factor ID Format

```
EF:<COUNTRY>:<fuel>:<year>:v<version>
```

Examples:
- `EF:US:diesel:2024:v1`
- `EF:UK:electricity:2024:v1`
- `EF:EU:natural_gas:2023:v2`

### Accessing Provenance

```python
prov = result["data"]["factor_record"]

print(f"Factor ID: {prov['factor_id']}")
print(f"Source: {prov['source_org']}")
print(f"Publication: {prov['source_publication']}")
print(f"Year: {prov['source_year']}")
print(f"Methodology: {prov['methodology']}")
print(f"Citation: {prov['citation']}")
```

### Citation Example

```
EPA (2024), "Emission Factors for Greenhouse Gas Inventories",
Table C-1: Default CO2 Emission Factors for Stationary Combustion,
https://www.epa.gov/climateleadership/ghg-emission-factors-hub
```

---

## Data Quality Scoring

### 5-Dimension DQS System

| Dimension | Description | Score (1-5) |
|-----------|-------------|-------------|
| **Temporal** | Recency of data | 5 = current year |
| **Geographical** | Geographical specificity | 5 = country-specific |
| **Technological** | Technology representativeness | 5 = exact match |
| **Representativeness** | Sample size/coverage | 5 = comprehensive |
| **Methodological** | Measurement vs estimation | 5 = measured |

### DQS Ratings

| Overall Score | Rating | Suitability |
|---------------|--------|-------------|
| 4.5 - 5.0 | Excellent | All reporting (CSRD, CDP, GRI) |
| 3.5 - 4.4 | Good | GHGP compliant |
| 2.5 - 3.4 | Fair | Acceptable with caveats |
| 1.5 - 2.4 | Poor | Improvement recommended |
| < 1.5 | Very Poor | Not recommended |

### Example

```python
quality = result["data"]["quality"]
dqs = quality["dqs"]

print(f"Overall Score: {dqs['overall_score']}/5.0")
print(f"Rating: {dqs['rating']}")
print(f"\nDimensions:")
print(f"  Temporal: {dqs['temporal']}/5")
print(f"  Geographical: {dqs['geographical']}/5")
print(f"  Technological: {dqs['technological']}/5")
print(f"  Representativeness: {dqs['representativeness']}/5")
print(f"  Methodological: {dqs['methodological']}/5")
```

---

## Performance Optimization

### Fast Path vs AI Path

**Fast Path** (60% traffic):
- No AI orchestration
- <100ms latency (P95)
- ~$0 cost per calculation
- Used when: `enable_explanations=False` and `response_format="legacy"`

**AI Path** (40% traffic):
- Full AI orchestration
- <500ms latency (P95)
- ~$0.005 cost per calculation
- Used when: explanations or enhanced format requested

### Optimization Strategies

#### 1. Disable Features for Production

```python
# Production-optimized agent (fast path for all requests)
agent = FuelAgentAI_v2(
    enable_explanations=False,      # Disable AI explanations
    enable_recommendations=False,    # Disable recommendations
    enable_fast_path=True           # Enable fast path optimization
)

# Use legacy format for maximum performance
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "legacy"     # Fast path eligible
})

# Latency: <50ms (P50), <100ms (P95)
# Cost: ~$0
```

#### 2. Batch Processing

```python
# Process multiple calculations
requests = [
    {"fuel_type": "diesel", "amount": 1000, "unit": "gallons"},
    {"fuel_type": "natural_gas", "amount": 5000, "unit": "therms"},
    {"fuel_type": "electricity", "amount": 10000, "unit": "kWh"},
]

results = [agent.run(req) for req in requests]

# 80% cost reduction for batched requests (shared cache)
```

#### 3. Caching

Cache is enabled by default with 95% hit rate:

```python
from greenlang.data.emission_factor_database import EmissionFactorDatabase

db = EmissionFactorDatabase(
    enable_cache=True,        # Enable caching (default)
    cache_size=1000,          # Max 1000 entries (default)
    cache_ttl=3600            # 1 hour TTL (default)
)

# Check cache statistics
stats = db.get_cache_stats()
print(f"Hit rate: {stats['hit_rate_pct']:.1f}%")  # Target: >95%
```

---

## Compliance Reporting

### CSRD E1-5 Compliance

```python
# CSRD requires multi-gas breakdown and provenance
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "scope": "1",
    "response_format": "enhanced"
})

# Extract required fields
data = result["data"]

# E1-5.1: Source disclosure
source = data["factor_record"]["source_org"]  # "EPA"
citation = data["factor_record"]["citation"]   # Full citation

# E1-5.2: Multi-gas breakdown
co2 = data["vectors_kg"]["CO2"]
ch4 = data["vectors_kg"]["CH4"]
n2o = data["vectors_kg"]["N2O"]

# E1-5.3: Data quality
dqs_score = data["quality"]["dqs"]["overall_score"]  # 4.8/5.0

# E1-5.4: Uncertainty
uncertainty = data["quality"]["uncertainty_95ci_pct"]  # ±8.5%

# E1-5.5: Scope and boundary
scope = data["scope"]           # "1"
boundary = data["boundary"]     # "combustion"
```

### CDP Climate Change Questionnaire

```python
# CDP C5.1: Scope 1 emissions methodology
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
    "scope": "1",
    "response_format": "enhanced"
})

# C5.1a: Methodology
methodology = result["data"]["factor_record"]["methodology"]  # "measured"

# C5.1b: Emission factor source
factor_source = result["data"]["factor_record"]["citation"]

# C5.1c: Geographical specificity
geography = result["data"]["country"]  # "US"
```

### GRI 305-2 (Scope 1 Emissions)

```python
# GRI requires disclosure of emission factors and methodologies
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "scope": "1",
    "response_format": "enhanced"
})

# GRI 305-2-b: Source of emission factors
print(f"Source: {result['data']['factor_record']['source_org']}")
print(f"Citation: {result['data']['factor_record']['citation']}")

# GRI 305-2-c: Standards/methodologies used
print(f"Methodology: {result['data']['factor_record']['methodology']}")
print(f"GWP Set: {result['data']['gwp_set']}")
```

---

## Error Handling

### Success Response

```python
{
    "success": True,
    "data": { ... },
    "metadata": { ... }
}
```

### Error Response

```python
{
    "success": False,
    "error": {
        "type": "ValidationError",
        "message": "Invalid fuel_type: 'unknown_fuel'",
        "agent_id": "fuel_ai_v2",
        "context": { ... }
    }
}
```

### Error Types

| Error Type | Cause | Solution |
|------------|-------|----------|
| `ValidationError` | Invalid input parameters | Check input schema |
| `CalculationError` | Calculation failed | Check fuel type and unit compatibility |
| `BudgetError` | AI budget exceeded | Increase `budget_usd` parameter |

### Error Handling Example

```python
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
})

if not result["success"]:
    error = result["error"]
    print(f"Error: {error['message']}")
    print(f"Type: {error['type']}")
else:
    emissions = result["data"]["co2e_emissions_kg"]
    print(f"Emissions: {emissions:.2f} kg CO2e")
```

---

## Best Practices

### 1. Use Fast Path for Production

```python
# ✅ GOOD: Fast path for production (60% cost reduction)
agent = FuelAgentAI_v2(
    enable_explanations=False,
    enable_recommendations=False,
    enable_fast_path=True
)

# ❌ BAD: AI path for all requests (expensive)
agent = FuelAgentAI_v2(
    enable_explanations=True,    # Forces AI path
    enable_recommendations=True  # Forces AI path
)
```

### 2. Use Enhanced Format Only When Needed

```python
# ✅ GOOD: Use enhanced format for compliance reporting only
result_reporting = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "enhanced"  # Only when needed
})

# ❌ BAD: Use enhanced format for all requests
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "enhanced"  # Unnecessary overhead
})
```

### 3. Cache Reuse

```python
# ✅ GOOD: Reuse agent instance (shares cache)
agent = FuelAgentAI_v2()

for request in requests:
    result = agent.run(request)  # Benefits from cache

# ❌ BAD: Create new agent for each request (no cache reuse)
for request in requests:
    agent = FuelAgentAI_v2()  # New cache each time
    result = agent.run(request)
```

### 4. Specify Scope and Boundary

```python
# ✅ GOOD: Explicit scope and boundary
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "scope": "1",                   # Explicit Scope 1
    "boundary": "combustion"        # Direct emissions only
})

# ⚠️ OK: Uses defaults (scope=1, boundary=combustion)
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
})
```

### 5. Error Handling

```python
# ✅ GOOD: Check success and handle errors
result = agent.run(payload)

if result["success"]:
    emissions = result["data"]["co2e_emissions_kg"]
    process_emissions(emissions)
else:
    log_error(result["error"])
    notify_admin(result["error"]["message"])

# ❌ BAD: Assume success
emissions = result["data"]["co2e_emissions_kg"]  # May crash if error
```

---

## Changelog

### v2.0.0 (2025-10-24)

**New Features:**
- Multi-gas breakdown (CO2, CH4, N2O)
- Full provenance tracking
- 5-dimension Data Quality Scoring
- Three response formats (legacy, enhanced, compact)
- Scope 1/2/3 support
- Emission boundaries (combustion, WTT, WTW)
- Multiple GWP sets (IPCC AR6 100yr/20yr)
- Fast path optimization (60% cost reduction)
- 95% cache hit rate

**Backward Compatibility:**
- All v1 inputs remain valid
- Default output format unchanged (legacy)
- Zero breaking changes

**Performance:**
- Fast path: <100ms latency (P95)
- Cost: 20% cheaper than v1
- Throughput: >50 req/s (single instance)

---

## Support

- **Documentation**: https://docs.greenlang.ai
- **Issues**: https://github.com/greenlang/greenlang/issues
- **Email**: support@greenlang.ai

---

**Document Version:** 1.0.0
**Last Updated:** 2025-10-24
**Author:** GreenLang Framework Team
