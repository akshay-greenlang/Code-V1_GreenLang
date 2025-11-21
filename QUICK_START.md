# GreenLang Quick Start Guide

**Get productive in 5 minutes** - Start calculating emissions, running agents, and building climate applications.

---

## 1. Installation (1 minute)

### Install via pip

```bash
# Core platform (recommended to start)
pip install greenlang-cli

# Verify installation
gl --version
# Output: GreenLang CLI v0.3.0
```

### Optional: Install with Full Features

```bash
# Includes analytics, LLM integration, and advanced features
pip install greenlang-cli[full]
```

### System Requirements

- Python 3.10 or higher
- pip (latest version recommended)
- Internet connection (for emission factor library)

---

## 2. First Calculation (2 minutes)

### Simple Emission Calculation

Copy and run this code to calculate emissions from natural gas consumption:

```python
from greenlang.calculation import CalculationEngine
from greenlang.emission_factors import EmissionFactorLibrary

# Initialize emission factor library and calculation engine
ef_library = EmissionFactorLibrary()
engine = CalculationEngine(ef_library)

# Calculate emissions for natural gas
result = engine.calculate(
    activity_type="fuel_combustion",
    fuel_type="natural_gas",
    quantity=1000,  # kWh
    unit="kWh"
)

# Display results
print(f"CO2e Emissions: {result.emissions_co2e} kg")
print(f"Emission Factor: {result.factor_name}")
print(f"Source: {result.factor_source}")
print(f"Calculation ID: {result.calculation_id}")
```

**Expected Output:**
```
CO2e Emissions: 184.0 kg
Emission Factor: Natural Gas - Grid Average
Source: DEFRA 2024
Calculation ID: calc_abc123def456
```

### Multi-Fuel Building Emissions

Calculate total emissions for a building with multiple energy sources:

```python
from greenlang.calculation import CalculationEngine
from greenlang.emission_factors import EmissionFactorLibrary

# Initialize
ef_library = EmissionFactorLibrary()
engine = CalculationEngine(ef_library)

# Define energy consumption data
energy_sources = [
    {
        "activity_type": "fuel_combustion",
        "fuel_type": "natural_gas",
        "quantity": 1000,
        "unit": "kWh",
        "description": "Heating"
    },
    {
        "activity_type": "electricity",
        "fuel_type": "grid_electricity",
        "quantity": 5000,
        "unit": "kWh",
        "region": "US-CA",  # California grid
        "description": "Building operations"
    },
    {
        "activity_type": "fuel_combustion",
        "fuel_type": "diesel",
        "quantity": 200,
        "unit": "liters",
        "description": "Backup generator"
    }
]

# Calculate emissions for each source
total_emissions = 0
breakdown = []

for source in energy_sources:
    result = engine.calculate(**source)

    if result.success:
        total_emissions += result.emissions_co2e
        breakdown.append({
            "description": source["description"],
            "fuel_type": source["fuel_type"],
            "emissions_kg": result.emissions_co2e,
            "factor_source": result.factor_source
        })
        print(f"✓ {source['description']}: {result.emissions_co2e:.2f} kg CO2e")
    else:
        print(f"✗ {source['description']}: Calculation failed - {result.error}")

# Display summary
print(f"\n{'='*50}")
print(f"Total Annual Emissions: {total_emissions:.2f} kg CO2e")
print(f"Total Emissions (metric tons): {total_emissions/1000:.3f} tCO2e")
print(f"\nEmissions Breakdown:")
for item in breakdown:
    percentage = (item['emissions_kg'] / total_emissions) * 100
    print(f"  {item['description']:20s}: {item['emissions_kg']:8.2f} kg ({percentage:5.1f}%)")
```

**Expected Output:**
```
✓ Heating: 184.00 kg CO2e
✓ Building operations: 1250.00 kg CO2e
✓ Backup generator: 536.00 kg CO2e

==================================================
Total Annual Emissions: 1970.00 kg CO2e
Total Emissions (metric tons): 1.970 tCO2e

Emissions Breakdown:
  Heating             :   184.00 kg ( 9.3%)
  Building operations :  1250.00 kg (63.5%)
  Backup generator    :   536.00 kg (27.2%)
```

### Query Emission Factors

Find and use specific emission factors from the library:

```python
from greenlang.emission_factors import EmissionFactorLibrary

# Initialize library
ef_library = EmissionFactorLibrary()

# Search for emission factors
diesel_factors = ef_library.search(
    fuel_type="diesel",
    region="global",
    scope="scope1"
)

print(f"Found {len(diesel_factors)} diesel emission factors:\n")

for factor in diesel_factors[:3]:  # Show first 3
    print(f"Name: {factor.name}")
    print(f"Value: {factor.value} {factor.unit}")
    print(f"Source: {factor.source} ({factor.year})")
    print(f"Scope: {factor.scope}")
    print(f"GHGs Included: {', '.join(factor.ghg_coverage)}")
    print("-" * 50)

# Get specific factor by ID
natural_gas_factor = ef_library.get_factor(
    fuel_type="natural_gas",
    region="US",
    preferred_source="EPA"
)

print(f"\nNatural Gas Factor (EPA):")
print(f"  Value: {natural_gas_factor.value} {natural_gas_factor.unit}")
print(f"  CO2: {natural_gas_factor.co2_factor} kg/kWh")
print(f"  CH4: {natural_gas_factor.ch4_factor} kg/kWh")
print(f"  N2O: {natural_gas_factor.n2o_factor} kg/kWh")
```

---

## 3. First Agent (2 minutes)

GreenLang agents are pre-built, modular calculation engines. Run your first agent to analyze fuel emissions:

### Using the Fuel Emissions Agent

```python
from greenlang.sdk import Agent, Pipeline
from greenlang.packs import load_pack

# Load the emissions-core pack
pack = load_pack("emissions-core")

# Get the Fuel Agent
fuel_agent = pack.get_agent("fuel-emissions")

# Prepare input data
input_data = {
    "fuel_type": "natural_gas",
    "consumption": 1000,
    "unit": "kWh",
    "period": "monthly",
    "region": "US-CA"
}

# Run the agent
result = fuel_agent.run(input_data)

# Display results
if result.success:
    print(f"✓ Calculation successful!")
    print(f"  Emissions: {result.data['emissions_co2e']} kg CO2e")
    print(f"  Emission Factor: {result.data['factor_used']}")
    print(f"  Calculation Method: {result.data['methodology']}")
    print(f"  Confidence: {result.metadata.confidence}")
    print(f"  Sources: {', '.join(result.metadata.sources)}")
else:
    print(f"✗ Calculation failed: {result.error}")
```

### Using Pre-Built Packs

GreenLang packs bundle agents, data, and calculations for specific use cases:

```bash
# List available packs
gl pack list

# Install a pack
gl pack install emissions-core

# Run a pack agent via CLI
gl run emissions-core/fuel-emissions --input fuel_data.json
```

### Running a Complete Pipeline

Agents can be chained together into pipelines:

```python
from greenlang.sdk import Pipeline
from greenlang.packs import load_pack

# Load packs
emissions_pack = load_pack("emissions-core")
carbon_pack = load_pack("carbon_ai")

# Create a pipeline
pipeline = Pipeline("building-carbon-analysis")

# Add agents to pipeline
pipeline.add_step(
    agent=emissions_pack.get_agent("fuel-emissions"),
    name="calculate_fuel_emissions"
)

pipeline.add_step(
    agent=carbon_pack.get_agent("carbon-aggregator"),
    name="aggregate_emissions",
    depends_on=["calculate_fuel_emissions"]
)

pipeline.add_step(
    agent=carbon_pack.get_agent("benchmark-analyzer"),
    name="benchmark_performance",
    depends_on=["aggregate_emissions"]
)

# Run the pipeline
input_data = {
    "building_id": "bldg-001",
    "fuel_consumption": [
        {"fuel_type": "natural_gas", "consumption": 1000, "unit": "kWh"},
        {"fuel_type": "electricity", "consumption": 5000, "unit": "kWh"}
    ],
    "building_type": "commercial_office",
    "area_sqft": 10000
}

result = pipeline.run(input_data)

# Display pipeline results
if result.success:
    print(f"Pipeline completed successfully!")
    print(f"\nStep Results:")
    for step_name, step_result in result.steps.items():
        print(f"  {step_name}: {step_result.status}")

    print(f"\nFinal Output:")
    print(f"  Total Emissions: {result.output['total_emissions_tons']} tCO2e")
    print(f"  Intensity: {result.output['intensity_per_sqft']} kgCO2e/sqft")
    print(f"  Benchmark Rating: {result.output['benchmark']['rating']}")
    print(f"  Percentile: {result.output['benchmark']['percentile']}th")
else:
    print(f"Pipeline failed: {result.error}")
```

### Using Pre-Built Agent Examples

Run one of the included examples:

```bash
# Navigate to examples directory
cd examples/quickstart

# Run hello world example (building emissions)
python hello-world.py

# Run data processing example (portfolio analysis)
python process-data.py
```

---

## 4. Next Steps

### Learn More

**Core Concepts:**
- [Agent Architecture Guide](C:\Users\aksha\Code-V1_GreenLang\AGENT_PATTERNS_GUIDE.md) - Understand agent design patterns
- [Emission Factors Library](C:\Users\aksha\Code-V1_GreenLang\EMISSION_FACTORS_LIBRARY_STATUS.md) - Browse 1,000+ emission factors
- [API Documentation](C:\Users\aksha\Code-V1_GreenLang\API_DOCUMENTATION.md) - Complete API reference

**Tutorials:**
- [Building Emissions Calculator](C:\Users\aksha\Code-V1_GreenLang\examples\global_building_example.py) - Building carbon footprinting
- [Portfolio Analysis](C:\Users\aksha\Code-V1_GreenLang\examples\quickstart\process-data.py) - Analyze multiple buildings
- [Custom Agent Development](C:\Users\aksha\Code-V1_GreenLang\AGENT_USAGE_GUIDE.md) - Build your own agents

**Production Applications:**
- [CBAM Compliance](C:\Users\aksha\Code-V1_GreenLang\GL-CBAM-APP\README.md) - EU Carbon Border Adjustment Mechanism
- [CSRD Reporting](C:\Users\aksha\Code-V1_GreenLang\GL-CSRD-APP\README.md) - EU Corporate Sustainability Reporting Directive
- [VCCI Scope 3](C:\Users\aksha\Code-V1_GreenLang\GL-VCCI-Carbon-APP\README.md) - Value Chain Carbon Inventory

### Available Packs

GreenLang includes pre-built packs for common use cases:

| Pack | Description | Use Cases |
|------|-------------|-----------|
| `emissions-core` | Fuel and electricity emissions | Scope 1 & 2 calculations |
| `carbon_ai` | Carbon aggregation and analysis | Portfolio management |
| `boiler_replacement_ai` | Boiler replacement optimization | Industrial decarbonization |
| `forecast_sarima_ai` | Emissions forecasting (SARIMA) | Trend analysis |
| `recommendation_ai` | Decarbonization recommendations | Strategy planning |
| `report_ai` | Automated reporting | Compliance documentation |

Browse all packs:
```bash
gl pack list --all
```

### Development Resources

**Get Help:**
- Documentation: [C:\Users\aksha\Code-V1_GreenLang\docs](C:\Users\aksha\Code-V1_GreenLang\docs)
- Examples: [C:\Users\aksha\Code-V1_GreenLang\examples](C:\Users\aksha\Code-V1_GreenLang\examples)
- Contributing: [CONTRIBUTING.md](C:\Users\aksha\Code-V1_GreenLang\CONTRIBUTING.md)

**Testing Your Code:**
```bash
# Run tests
pytest tests/

# Check code quality
gl verify --coverage

# Run specific example
python examples/fuel_agent_integration.py
```

**Build Custom Applications:**
1. Start with an example from `examples/`
2. Modify input data for your use case
3. Add custom agents using `AGENT_USAGE_GUIDE.md`
4. Deploy using Docker ([DOCKER_RELEASE_GUIDE.md](C:\Users\aksha\Code-V1_GreenLang\DOCKER_RELEASE_GUIDE.md))

---

## Troubleshooting

**Installation Issues:**

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -v greenlang-cli

# Check Python version
python --version  # Must be 3.10+
```

**Import Errors:**

```python
# Check installation
import greenlang
print(greenlang.__version__)  # Should show 0.3.0

# Verify emission factors library
from greenlang.emission_factors import EmissionFactorLibrary
ef_lib = EmissionFactorLibrary()
print(f"Loaded {len(ef_lib.factors)} emission factors")
```

**Calculation Errors:**

- Ensure internet connection (for emission factor updates)
- Verify input data format matches agent schema
- Check fuel type names (use `ef_library.list_fuel_types()`)
- Review error messages for specific guidance

**Enable Debug Mode:**

```bash
# Set debug environment variable
export GL_DEBUG=1  # Linux/Mac
set GL_DEBUG=1     # Windows

# Run with verbose logging
python your_script.py
```

---

## Quick Reference

### Common Fuel Types

```python
# Supported fuel types (partial list)
fuel_types = [
    "natural_gas",
    "diesel",
    "gasoline",
    "coal",
    "propane",
    "electricity",
    "grid_electricity",
    "fuel_oil",
    "lpg",
    "biomass"
]
```

### Common Units

```python
# Energy units
energy_units = ["kWh", "MWh", "GJ", "MMBTU", "therms"]

# Volume units
volume_units = ["liters", "gallons", "cubic_meters", "cubic_feet"]

# Mass units
mass_units = ["kg", "tonnes", "pounds", "short_tons"]
```

### Common Regions

```python
# Regional emission factors
regions = [
    "global",      # Global average
    "US",          # United States average
    "US-CA",       # California
    "US-TX",       # Texas
    "EU",          # European Union average
    "GB",          # United Kingdom
    "DE",          # Germany
    "FR",          # France
    "JP",          # Japan
    "CN",          # China
    "IN"           # India
]
```

---

**You're now ready to build climate-intelligent applications with GreenLang!**

For more advanced features, see the [Complete Documentation](C:\Users\aksha\Code-V1_GreenLang\GREENLANG_DOCUMENTATION.md).
