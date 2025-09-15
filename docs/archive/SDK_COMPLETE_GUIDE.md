# GreenLang SDK - Complete Feature Guide

## Installation & Setup

```python
from greenlang.sdk import GreenLangClient

# Initialize client (default: US region)
client = GreenLangClient()

# Initialize with specific region
client = GreenLangClient(region="IN")  # India
client = GreenLangClient(region="EU")  # Europe
client = GreenLangClient(region="CN")  # China
```

## 🔥 ALL SDK FEATURES

### 1. **Basic Emissions Calculation**
```python
# Calculate emissions for any fuel type
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh",
    region="US"  # Optional, overrides default
)

# Result structure
{
    "success": True,
    "data": {
        "fuel_type": "electricity",
        "consumption": 1000,
        "unit": "kWh",
        "emission_factor": 0.385,
        "co2e_emissions_kg": 385.0,
        "co2e_emissions_tons": 0.385,
        "region": "US"
    }
}
```

### 2. **Boiler Emissions Calculation**
```python
# Calculate boiler/thermal system emissions
result = client.calculate_boiler_emissions(
    fuel_type="natural_gas",
    thermal_output=1000,
    output_unit="kWh",
    efficiency=0.85,  # 85% efficiency
    boiler_type="condensing",  # condensing/standard/old
    region="US"
)

# Result includes fuel consumption and emissions
{
    "success": True,
    "data": {
        "co2e_emissions_kg": 212.75,
        "fuel_consumption_value": 40.14,
        "fuel_consumption_unit": "therms",
        "efficiency": 0.85,
        "thermal_efficiency_percent": 85.0
    }
}
```

### 3. **Building Analysis (Comprehensive)**
```python
# Analyze entire building with all metrics
building_data = {
    "metadata": {
        "building_type": "hospital",  # hospital/office/retail/data_center
        "area": 100000,
        "area_unit": "sqft",
        "location": {
            "country": "IN",
            "city": "Mumbai"
        },
        "occupancy": 500,
        "floor_count": 5,
        "building_age": 10,
        "climate_zone": "tropical"  # optional
    },
    "energy_consumption": {
        "electricity": {"value": 3500000, "unit": "kWh"},
        "natural_gas": {"value": 10000, "unit": "therms"},
        "diesel": {"value": 50000, "unit": "liters"},
        "solar_pv_generation": {"value": 100000, "unit": "kWh"}  # optional
    }
}

result = client.analyze_building(building_data)

# Result includes:
# - Total emissions
# - Emissions breakdown by source
# - Intensity metrics (per sqft, per person)
# - Benchmark rating
# - Performance category
# - Recommendations
```

### 4. **Emissions Aggregation**
```python
# Aggregate multiple emission sources
emissions_list = [
    {"fuel_type": "electricity", "co2e_emissions_kg": 1925.0},
    {"fuel_type": "natural_gas", "co2e_emissions_kg": 1060.0},
    {"fuel_type": "diesel", "co2e_emissions_kg": 510.5}
]

result = client.aggregate_emissions(emissions_list)

# Result
{
    "success": True,
    "data": {
        "total_co2e_kg": 3495.5,
        "total_co2e_tons": 3.495,
        "emissions_breakdown": [
            {"source": "electricity", "co2e_kg": 1925.0, "co2e_tons": 1.925, "percentage": 55.1},
            {"source": "natural_gas", "co2e_kg": 1060.0, "co2e_tons": 1.060, "percentage": 30.3},
            {"source": "diesel", "co2e_kg": 510.5, "co2e_tons": 0.510, "percentage": 14.6}
        ]
    }
}
```

### 5. **Benchmarking**
```python
# Compare emissions against industry benchmarks
result = client.benchmark_emissions(
    emissions_kg=50000,
    area=10000,
    building_type="commercial_office",
    period_months=12
)

# Result
{
    "success": True,
    "data": {
        "rating": "Good",  # Excellent/Good/Average/Poor
        "carbon_intensity": 5.0,  # kgCO2e/sqft/year
        "benchmark_value": 6.5,   # Industry average
        "performance_category": "Above Average",
        "comparison_message": "15% better than industry average"
    }
}
```

### 6. **Get Recommendations**
```python
# Get optimization recommendations
result = client.get_recommendations(
    building_type="hospital",
    emissions_by_source={
        "electricity": 2500000,
        "natural_gas": 500000,
        "diesel": 150000
    },
    country="IN"
)

# Result includes quick wins and long-term strategies
{
    "success": True,
    "data": {
        "quick_wins": [
            {"action": "LED lighting upgrade", "impact": "High", "payback": "1-2 years"},
            {"action": "HVAC scheduling optimization", "impact": "Medium", "payback": "< 1 year"}
        ],
        "long_term": [...],
        "potential_emissions_reduction": {"percentage_range": "15-25%"}
    }
}
```

### 7. **Calculate Intensity Metrics**
```python
# Calculate emission intensity
result = client.calculate_intensity(
    total_emissions_kg=100000,
    area=50000,
    area_unit="sqft",
    occupancy=200,
    floor_count=10
)

# Result
{
    "success": True,
    "data": {
        "intensities": {
            "per_sqft_year": 2.0,
            "per_sqft_month": 0.167,
            "per_person_year": 500.0,
            "per_floor_year": 10000.0
        },
        "performance_rating": "Good"
    }
}
```

### 8. **Get Grid Emission Factors**
```python
# Get country-specific emission factors
result = client.get_emission_factor(
    fuel_type="electricity",
    country="IN",
    unit="kWh"
)

# Result
{
    "success": True,
    "data": {
        "emission_factor": 0.71,
        "unit": "kgCO2e/kWh",
        "country": "IN",
        "source": "India Central Electricity Authority",
        "grid_mix": {
            "renewable": 0.23,
            "fossil": 0.77
        }
    }
}
```

### 9. **Generate Reports**
```python
# Generate formatted report
analysis_results = client.analyze_building(building_data)

report = client.generate_report(
    analysis_results=analysis_results,
    format="markdown"  # or "json", "text"
)

# Saves report with emissions summary, charts, recommendations
```

### 10. **Portfolio Analysis**
```python
# Analyze multiple buildings
buildings = [
    building1_data,
    building2_data,
    building3_data
]

result = client.analyze_portfolio(buildings)

# Result includes portfolio-level metrics
{
    "success": True,
    "data": {
        "buildings": [...],  # Individual building results
        "portfolio_metrics": {
            "total_buildings": 3,
            "total_emissions_kg": 500000,
            "total_emissions_tons": 500,
            "total_area_sqft": 300000,
            "average_intensity": 1.67,
            "best_performer": "building_1",
            "worst_performer": "building_3"
        }
    }
}
```

### 11. **Workflow Execution**
```python
# Create and execute custom workflows
workflow = client.create_workflow("custom_analysis", "My custom workflow")
workflow.add_agent("validator", {...})
workflow.add_agent("fuel", {...})
workflow.add_agent("report", {...})

result = client.execute_workflow("custom_analysis", input_data)
```

### 12. **Validation**
```python
# Validate building data before processing
is_valid = client.validate_building_data(building_data)

if is_valid["success"]:
    # Data is valid, proceed
    result = client.analyze_building(building_data)
else:
    print(f"Validation error: {is_valid['error']}")
```

### 13. **Utility Methods**
```python
# List all available agents
agents = client.list_agents()
# Returns: ['validator', 'fuel', 'carbon', 'report', 'benchmark', 
#           'grid_factor', 'building_profile', 'intensity', 
#           'recommendation', 'boiler']

# List registered workflows
workflows = client.list_workflows()

# Get supported countries
countries = client.get_supported_countries()
# Returns: ['US', 'IN', 'CN', 'EU', 'JP', 'BR', 'KR', ...]

# Get supported fuel types for a country
fuel_types = client.get_supported_fuel_types("IN")
# Returns: ['electricity', 'natural_gas', 'diesel', 'coal', ...]
```

## 📊 Complete Working Example

```python
from greenlang.sdk import GreenLangClient

# Initialize client for India
client = GreenLangClient(region="IN")

# 1. Calculate simple emissions
electricity = client.calculate_emissions("electricity", 5000, "kWh")
print(f"Electricity: {electricity['data']['co2e_emissions_kg']} kg CO2e")

# 2. Calculate boiler emissions
boiler = client.calculate_boiler_emissions(
    fuel_type="natural_gas",
    thermal_output=1000,
    output_unit="kWh",
    efficiency=0.85
)
print(f"Boiler: {boiler['data']['co2e_emissions_kg']} kg CO2e")

# 3. Analyze a complete building
building = {
    "metadata": {
        "building_type": "hospital",
        "area": 100000,
        "area_unit": "sqft",
        "location": {"country": "IN"},
        "occupancy": 500,
        "floor_count": 5,
        "building_age": 10
    },
    "energy_consumption": {
        "electricity": {"value": 3500000, "unit": "kWh"},
        "diesel": {"value": 50000, "unit": "liters"}
    }
}

analysis = client.analyze_building(building)

# 4. Display results
if analysis["success"]:
    data = analysis["data"]
    print(f"\nBuilding Analysis:")
    print(f"Total Emissions: {data['emissions']['total_co2e_tons']:.2f} tons/year")
    print(f"Intensity: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
    print(f"Rating: {data['benchmark']['rating']}")
    
    # Show recommendations
    for rec in data['recommendations']['quick_wins'][:3]:
        print(f"- {rec['action']} (Impact: {rec['impact']})")
```

## 🌍 Multi-Country Support

```python
# Compare emissions across countries
countries = ["US", "IN", "CN", "EU", "JP", "BR"]

for country in countries:
    client = GreenLangClient(region=country)
    result = client.calculate_emissions("electricity", 1000, "kWh")
    print(f"{country}: {result['data']['co2e_emissions_kg']} kg CO2e/MWh")

# Output:
# US: 385 kg CO2e/MWh
# IN: 710 kg CO2e/MWh
# CN: 650 kg CO2e/MWh
# EU: 230 kg CO2e/MWh
# JP: 450 kg CO2e/MWh
# BR: 120 kg CO2e/MWh
```

## 🚀 Advanced Features

### Async Operations
```python
import asyncio

async def analyze_buildings_async(buildings):
    tasks = []
    for building in buildings:
        task = asyncio.create_task(
            client.analyze_building_async(building)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### Caching
```python
# Results are automatically cached for performance
# Cache TTL: 1 hour for emission factors
# Clear cache if needed
client.orchestrator.clear_cache()
```

### Error Handling
```python
try:
    result = client.calculate_emissions("electricity", 1000, "kWh")
    if result["success"]:
        print(f"Emissions: {result['data']['co2e_emissions_kg']} kg")
    else:
        print(f"Error: {result['error']}")
except Exception as e:
    print(f"Exception: {e}")
```

## 📚 SDK Methods Summary

| Method | Description | Returns |
|--------|-------------|---------|
| `calculate_emissions()` | Calculate emissions for fuel | Emissions data |
| `calculate_boiler_emissions()` | Boiler/thermal emissions | Boiler emissions |
| `analyze_building()` | Complete building analysis | Full analysis |
| `aggregate_emissions()` | Combine multiple sources | Total emissions |
| `benchmark_emissions()` | Compare to standards | Rating & metrics |
| `get_recommendations()` | Get optimization tips | Recommendations |
| `calculate_intensity()` | Intensity metrics | Per sqft/person |
| `get_emission_factor()` | Country factors | Emission factors |
| `generate_report()` | Create reports | Formatted report |
| `analyze_portfolio()` | Multiple buildings | Portfolio metrics |
| `validate_building_data()` | Validate input | Validation result |
| `create_workflow()` | Custom workflows | Workflow builder |

---

**The SDK is fully featured with 13+ methods for comprehensive emissions analysis!**