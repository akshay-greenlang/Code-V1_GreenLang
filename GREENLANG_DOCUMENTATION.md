# GreenLang Documentation v0.0.1

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Core Concepts](#core-concepts)
5. [CLI Reference](#cli-reference)
6. [Agents](#agents)
7. [Workflows](#workflows)
8. [Python SDK](#python-sdk)
9. [API Reference](#api-reference)
10. [Data Models](#data-models)
11. [Emission Factors & Datasets](#emission-factors--datasets)
12. [Testing](#testing)
13. [Examples](#examples)
14. [Architecture](#architecture)
15. [Benchmarks](#benchmarks)
16. [Recommendations](#recommendations)
17. [Type System](#type-system)
18. [Contributing](#contributing)
19. [Roadmap](#roadmap)
20. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is GreenLang?

GreenLang is The Climate Intelligence Framework for the entire climate industry. Build climate apps fast with modular agents, YAML pipelines, a clean CLI, and a Python SDK. 

Initially focused on buildings and expanding rapidly to the entire HVAC industry and solar thermal replacement for boiler infrastructure, GreenLang provides developers a consistent way to:
- Model emissions across industries (buildings, HVAC, solar thermal, industrial processes)
- Simulate decarbonization options with real-world data
- Generate explainable reports for stakeholders
- Calculate emissions across multiple energy sources and systems
- Benchmark performance against regional and industry standards
- Create actionable optimization recommendations with ROI estimates
- Support global regions with localized emission factors

### Vision

To provide the universal climate intelligence framework that empowers developers to build climate solutions across all industries, accelerating the global transition to net-zero through consistent, explainable, and actionable emissions intelligence.

### Key Features
- **Industry-Agnostic Framework**: Adaptable to any climate-related industry
- **Global Coverage**: Support for 12 major economies with localized emission factors
- **Multi-Sector Support**: Buildings, HVAC systems, solar thermal, boilers, and expanding
- **Modular Agent Architecture**: Plug-and-play agents for different climate calculations
- **AI-Powered**: Intelligent agents for automated analysis and recommendations
- **Comprehensive Metrics**: Total emissions, intensity metrics, industry benchmarks
- **Actionable Insights**: Practical recommendations with ROI estimates
- **Developer-First**: Clean CLI, Python SDK, YAML workflows for rapid development
- **Type-Safe**: 100% typed public APIs with strict mypy enforcement
- **Explainable Results**: Transparent calculations with full audit trails
19. [Future Roadmap](#future-roadmap)
20. [Support](#support)
21. [Achievements](#achievements)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from Local Directory

```bash
# Navigate to GreenLang directory
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"

# Install package
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m greenlang.cli.main --version
# Should show: GreenLang, version 0.0.1
```

### Windows Setup (Recommended)

```bash
# Use the batch file for easier access
./greenlang.bat --version
./greenlang.bat calc --building
./greenlang.bat analyze building.json
```

---

## Quick Start

### ⚠️ IMPORTANT: Command Usage
All commands must be prefixed with `greenlang`. You cannot run subcommands directly.

❌ **WRONG**: `calc`, `agents`, `benchmark`  
✅ **CORRECT**: `greenlang calc`, `greenlang agents`, `greenlang benchmark`

### 1. Simple Emissions Calculator
```bash
# Basic calculator
greenlang calc

# With country specification
greenlang calc --country IN
```

### 2. Commercial Building Analysis (NEW)
```bash
# Interactive building mode
greenlang calc --building

# With country pre-selected
greenlang calc --building --country US

# Load from file
greenlang calc --building --input building_data.json --output results.json
```

### 3. Analyze Existing Building
```bash
# Analyze from JSON file
greenlang analyze building_data.json --country IN
```

### 4. View Benchmarks
```bash
# Show benchmarks for building type and country
greenlang benchmark --type hospital --country EU

# List all available benchmarks
greenlang benchmark --list
```

### 5. Get Recommendations
```bash
# Interactive recommendation generator
greenlang recommend
```

---

## Commercial Building Calculator

### Overview
The enhanced commercial building calculator provides comprehensive emissions analysis with global support.

### Using the Calculator

#### Interactive Mode
```bash
greenlang calc --building
```

You'll be prompted for:
1. **Country/Region** (12+ countries supported)
2. **Building Information**:
   - Building type (7 types available)
   - Area (sqft or sqm)
   - Occupancy
   - Number of floors
   - Building age
   - Climate zone (optional)
3. **Energy Consumption** (annual):
   - Electricity (kWh)
   - Natural Gas (therms, m³, MMBtu, kWh)
   - Diesel (liters, gallons)
   - District Heating (kWh) - for EU/China
   - Solar PV Generation (kWh)
   - LPG/Propane (kg, liters)
   - And more based on region

#### File-Based Mode
Create a JSON file with building data:

```json
{
  "metadata": {
    "building_type": "commercial_office",
    "area": 50000,
    "area_unit": "sqft",
    "location": {
      "country": "IN",
      "region": "Maharashtra",
      "city": "Mumbai"
    },
    "occupancy": 200,
    "floor_count": 10,
    "building_age": 15,
    "climate_zone": "tropical"
  },
  "energy_consumption": {
    "electricity": {"value": 1500000, "unit": "kWh"},
    "diesel": {"value": 10000, "unit": "liters"},
    "lpg_propane": {"value": 500, "unit": "kg"},
    "solar_pv_generation": {"value": 50000, "unit": "kWh"}
  }
}
```

Then run:
```bash
greenlang calc --building --input building_data.json --output results.json
```

### Calculator Output
The calculator provides:
- **Building Profile**: Type, location, expected performance
- **Total Emissions**: Annual CO2e in kg and metric tons
- **Emissions Breakdown**: By energy source with percentages
- **Intensity Metrics**: Per sqft, per person, per floor
- **Performance Rating**: Excellent/Good/Average/Poor
- **Benchmark Comparison**: Against regional standards
- **Recommendations**: Quick wins with payback periods
- **Potential Savings**: Percentage reduction possible

---

## CLI Commands Reference

### calc - Emissions Calculator
```bash
# Simple mode
greenlang calc
greenlang calc --country US
greenlang calc --output results.json

# Building mode (enhanced)
greenlang calc --building
greenlang calc --building --country IN
greenlang calc --building --input data.json --output results.json
```

**Options:**
- `--building`: Enable commercial building mode
- `--country`: Specify country/region (US, IN, EU, CN, JP, BR, KR, UK, DE, CA, AU)
- `--input`: Load data from JSON file
- `--output`: Save results to file

### analyze - Analyze Building from File
```bash
greenlang analyze building_data.json
greenlang analyze building_data.json --country US
```

**Purpose**: Analyze an existing building using comprehensive data from a JSON file.

### benchmark - View Benchmark Standards
```bash
# View specific benchmark
greenlang benchmark --type commercial_office --country US

# List all benchmarks
greenlang benchmark --list

# Examples for different countries
greenlang benchmark --type hospital --country IN
greenlang benchmark --type data_center --country EU
```

**Options:**
- `--type`: Building type (commercial_office, hospital, data_center, retail, warehouse, hotel, education)
- `--country`: Country code (US, IN, EU, CN, JP, BR, KR)
- `--list`: List all available benchmarks

### recommend - Get Optimization Recommendations
```bash
greenlang recommend
```

Interactive prompts for:
- Building type
- Country
- Building age
- Current performance rating

**Output**:
- Quick wins (low cost, high impact)
- Implementation roadmap
- Expected savings
- Payback periods

### ask - AI Assistant
```bash
# Building-specific queries
greenlang ask "What is the carbon footprint of a 100,000 sqft hospital in Mumbai?"

# Comparison queries
greenlang ask "Compare emissions for offices in US vs India"

# Recommendation queries
greenlang ask "How to reduce emissions for a data center in Singapore?"

# With verbose output
greenlang ask -v "Calculate emissions for 50000 sqft office with 1.5M kWh"
```

### agents - List Available Agents
```bash
greenlang agents
```

Shows all 9 agents including new ones:
- GridFactorAgent
- BuildingProfileAgent
- IntensityAgent
- RecommendationAgent

### agent - Show Agent Details
```bash
greenlang agent grid_factor
greenlang agent building_profile
greenlang agent intensity
greenlang agent recommendation
```

### run - Execute Workflows
```bash
# Run commercial building workflow
greenlang run workflows/commercial_building_emissions.yaml --input data.json

# Run with output formatting
greenlang run workflow.yaml -i input.json -o output.json --format json
```

### init - Initialize Project
```bash
greenlang init
greenlang init --output my_workflow.yaml
```

Creates sample workflow and input files adapted to v0.0.1 features.

### dev - Developer Interface
```bash
greenlang dev
```

Launches interactive developer interface with enhanced features.

---

## SDK Usage

### Basic SDK Setup

```python
from greenlang.sdk.enhanced_client import GreenLangClient

# Initialize with region
client = GreenLangClient(region="IN")  # India
client = GreenLangClient(region="US")  # United States
client = GreenLangClient(region="EU")  # Europe
```

### Calculate Emissions with Regional Factors

```python
# Simple calculation with regional factors
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=50000,
    unit="kWh",
    region="IN"  # Uses India's grid factor (0.71 kgCO2e/kWh)
)

print(f"Emissions: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
print(f"Emission Factor: {result['data']['emission_factor']} kgCO2e/kWh")
print(f"Region: {result['data']['region']}")
```

### Comprehensive Building Analysis

```python
# Define building data
building = {
    "metadata": {
        "building_type": "hospital",
        "area": 100000,
        "area_unit": "sqft",
        "location": {
            "country": "IN",
            "region": "Delhi",
            "city": "New Delhi"
        },
        "occupancy": 500,
        "floor_count": 8,
        "building_age": 10,
        "climate_zone": "composite"
    },
    "energy_consumption": {
        "electricity": {"value": 3500000, "unit": "kWh"},
        "diesel": {"value": 50000, "unit": "liters"},  # Backup generators
        "lpg_propane": {"value": 1000, "unit": "kg"}   # Kitchen
    }
}

# Analyze building
result = client.analyze_building(building)

# Access results
if result["success"]:
    data = result["data"]
    
    # Emissions
    print(f"Total Annual Emissions: {data['emissions']['total_co2e_tons']:.2f} tons")
    
    # Intensity metrics
    print(f"Per sqft: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
    print(f"Per person: {data['intensity']['intensities']['per_person_year']:.0f} kgCO2e/person/year")
    
    # Performance
    print(f"Performance Rating: {data['intensity']['performance_rating']}")
    print(f"Benchmark: {data['benchmark']['rating']}")
    
    # Recommendations
    for rec in data['recommendations']['quick_wins'][:3]:
        print(f"• {rec['action']}")
        print(f"  Impact: {rec['impact']}, Payback: {rec['payback']}")
```

### Portfolio Analysis

```python
# Analyze multiple buildings
portfolio = [
    {
        "id": "mumbai_office",
        "metadata": {
            "building_type": "commercial_office",
            "area": 50000,
            "location": {"country": "IN", "city": "Mumbai"},
            "occupancy": 200
        },
        "energy_consumption": {
            "electricity": {"value": 1500000, "unit": "kWh"}
        }
    },
    {
        "id": "delhi_hospital",
        "metadata": {
            "building_type": "hospital",
            "area": 100000,
            "location": {"country": "IN", "city": "Delhi"},
            "occupancy": 500
        },
        "energy_consumption": {
            "electricity": {"value": 3500000, "unit": "kWh"},
            "diesel": {"value": 50000, "unit": "liters"}
        }
    }
]

# Analyze portfolio
portfolio_result = client.analyze_portfolio(portfolio)

print(f"Total Portfolio Emissions: {portfolio_result['data']['portfolio_metrics']['total_emissions_tons']:.1f} tons")
print(f"Total Area: {portfolio_result['data']['portfolio_metrics']['total_area_sqft']:,.0f} sqft")
print(f"Average Intensity: {portfolio_result['data']['portfolio_metrics']['average_intensity']:.2f} kgCO2e/sqft/year")
```

### Get Country-Specific Factors

```python
# Get emission factors for different countries
countries = ["US", "IN", "EU", "CN", "JP", "BR"]

for country in countries:
    factor = client.get_emission_factor(
        fuel_type="electricity",
        unit="kWh",
        country=country
    )
    print(f"{country}: {factor['data']['emission_factor']} kgCO2e/kWh")
    print(f"  Renewable Share: {factor['data']['grid_renewable_share']*100:.0f}%")
```

### Get Recommendations

```python
# Get optimization recommendations
recommendations = client.get_recommendations(
    building_type="commercial_office",
    performance_rating="Average",
    country="IN",
    building_age=20
)

# Display roadmap
for phase in recommendations['data']['implementation_roadmap']:
    print(f"\n{phase['phase']}")
    print(f"Cost: {phase['estimated_cost']}")
    print(f"Impact: {phase['expected_impact']}")
    for action in phase['actions']:
        print(f"  • {action['action']}")
```

### Export Results

```python
# Export to different formats
client.export_analysis(result, "analysis.xlsx", format="excel")
client.export_analysis(result, "analysis.json", format="json")
client.export_analysis(result, "analysis.csv", format="csv")
```

---

## Agents Overview

### Core Agents (7) - Enhanced
1. **InputValidatorAgent** - Validates input data
2. **FuelAgent** - Enhanced fuel-based emissions with caching, batch processing, and recommendations
3. **BoilerAgent** - Comprehensive boiler emissions with async support and performance tracking
4. **CarbonAgent** - Aggregates total emissions
5. **ReportAgent** - Generates reports
6. **BenchmarkAgent** - Compares to standards
7. **RecommendationAgent** - Generates optimization recommendations

### Enhanced Agent Features

#### FuelAgent v0.0.1
Enhanced with performance optimizations and advanced features:

**New Capabilities:**
- **Performance Caching**: @lru_cache for emission factors
- **Batch Processing**: Process multiple fuel sources in parallel
- **Fuel Recommendations**: Intelligent fuel switching suggestions
- **Constants Library**: Comprehensive fuel constants and mappings
- **Integration Examples**: 7 real-world integration patterns

```python
from greenlang.agents import FuelAgent

agent = FuelAgent()

# Single fuel calculation
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms",
    "country": "US"
})

# Batch processing
results = agent.batch_process([
    {"fuel_type": "natural_gas", "amount": 1000, "unit": "therms"},
    {"fuel_type": "electricity", "amount": 5000, "unit": "kWh"},
    {"fuel_type": "diesel", "amount": 100, "unit": "gallons"}
])
```

#### BoilerAgent v0.0.1
State-of-the-art boiler emissions calculator with enterprise features:

**Key Features:**
- **Async Support**: Process multiple boilers concurrently
- **Performance Tracking**: Built-in monitoring with psutil
- **Unit Conversion**: Centralized unit conversion library
- **External Configuration**: JSON-based efficiency configurations
- **Validation Schema**: JSON Schema input validation
- **Export Formats**: JSON, CSV, Excel export capabilities
- **Historical Tracking**: Track performance over time
- **10 Integration Examples**: Comprehensive usage patterns

```python
from greenlang.agents import BoilerAgent
import asyncio

agent = BoilerAgent()

# Standard calculation
result = agent.run({
    "boiler_type": "condensing",
    "fuel_type": "natural_gas",
    "thermal_output": {"value": 100, "unit": "MMBtu"},
    "efficiency": 0.95
})

# Async batch processing
async def process_boilers():
    results = await agent.async_batch_process([
        {"boiler_type": "condensing", "thermal_output": {"value": 100, "unit": "MMBtu"}},
        {"boiler_type": "heat_pump", "fuel_consumption": {"value": 5000, "unit": "kWh"}}
    ])
    return results

# Export results
agent.export_results(results, format="csv")
```

**Supported Boiler Types:**
- Condensing, Standard, Low-efficiency
- Heat pumps (air/ground source)
- Biomass (modern/traditional)
- Coal (pulverized/stoker/hand-fired)
- Electric resistance
- District heating
- Hydrogen/Fuel cell

**Advanced Calculations:**
- Altitude adjustments
- Maintenance impact factors
- Oversizing penalties
- Regional efficiency standards (US/EU/UK/CN)
- Thermal-to-fuel conversions
- Performance ratings and recommendations

### New Specialized Agents (4)

#### GridFactorAgent
Retrieves country-specific emission factors for all fuel types.

```python
from greenlang.agents import GridFactorAgent

agent = GridFactorAgent()
result = agent.run({
    "country": "IN",
    "fuel_type": "electricity",
    "unit": "kWh"
})
# Returns: 0.71 kgCO2e/kWh for India
```

#### BuildingProfileAgent
Categorizes buildings and provides expected performance metrics.

```python
from greenlang.agents import BuildingProfileAgent

agent = BuildingProfileAgent()
result = agent.run({
    "building_type": "hospital",
    "area": 100000,
    "building_age": 15,
    "climate_zone": "tropical",
    "country": "IN"
})
# Returns expected EUI, load breakdown, benchmark standards
```

#### IntensityAgent
Calculates multiple intensity metrics.

```python
from greenlang.agents import IntensityAgent

agent = IntensityAgent()
result = agent.run({
    "total_emissions_kg": 500000,
    "area": 50000,
    "occupancy": 200,
    "floor_count": 10
})
# Returns per sqft/person/floor intensities
```

#### RecommendationAgent
Provides optimization strategies with implementation roadmap.

```python
from greenlang.agents import RecommendationAgent

agent = RecommendationAgent()
result = agent.run({
    "building_type": "commercial_office",
    "country": "US",
    "building_age": 20,
    "performance_rating": "Below Average"
})
# Returns quick wins, roadmap, potential savings
```

---

## Workflows

### Commercial Building Workflow
```yaml
name: commercial_building_emissions
description: Complete carbon audit with benchmarking
version: 0.0.1
steps:
  - name: validate_input
    agent_id: validator
  - name: analyze_building_profile
    agent_id: building_profile
  - name: calculate_emissions
    agent_id: grid_factor
    parallel: true
  - name: calculate_intensity
    agent_id: intensity
  - name: benchmark_performance
    agent_id: benchmark
  - name: generate_recommendations
    agent_id: recommendation
  - name: generate_report
    agent_id: report
```

Run with:
```bash
greenlang run workflows/commercial_building_emissions.yaml --input building_data.json
```

### India-Specific Workflow
```yaml
name: india_commercial_building
description: BEE-compliant analysis
compliance_checks:
  - ecbc_compliance
  - pat_cycle_target
  - bee_star_rating
```

---

## Global Emission Factors

### Electricity Grid Factors by Country

| Country | Code | Factor (kgCO2e/kWh) | Renewable % | Source |
|---------|------|-------------------|-------------|---------|
| 🇧🇷 Brazil | BR | 0.12 | 83% | Hydro-dominant |
| 🇨🇦 Canada | CA | 0.13 | 68% | Hydro-heavy |
| 🇬🇧 UK | UK | 0.212 | 43% | DEFRA |
| 🇪🇺 EU | EU | 0.23 | 42% | Mixed renewable |
| 🇩🇪 Germany | DE | 0.38 | 46% | Mixed grid |
| 🇺🇸 USA | US | 0.385 | 21% | EPA eGRID |
| 🇯🇵 Japan | JP | 0.45 | 22% | METI |
| 🇰🇷 South Korea | KR | 0.49 | 8% | KEA |
| 🇨🇳 China | CN | 0.65 | 31% | MEE |
| 🇦🇺 Australia | AU | 0.66 | 32% | Coal-heavy |
| 🇮🇳 India | IN | 0.71 | 23% | CEA |

### Natural Gas Factors

| Country | therms | m³ | MMBtu |
|---------|--------|-----|-------|
| USA | 5.3 | 1.89 | 53.06 |
| India | 5.3 | 1.89 | 53.06 |
| EU | 5.0 | 1.89 | 50.0 |
| China | 4.8 | 1.89 | 48.0 |
| Japan | 4.5 | 1.89 | 45.0 |
| Brazil | 4.5 | 1.89 | 45.0 |

### Other Fuel Types

| Fuel Type | Unit | Factor | Countries |
|-----------|------|--------|-----------|
| Diesel | liters | 2.68 | All |
| LPG/Propane | kg | 2.98 | IN, BR, KR |
| District Heating | kWh | 0.28-0.55 | EU, CN, KR |
| Heating Oil | gallons | 10.16 | US, EU |
| Coal | tons | 2086-2500 | CN, IN |

---

## Benchmarking Standards

### Regional Certification Programs

#### USA - ENERGY STAR
- **Score**: 1-100 (75+ for certification)
- **Excellent**: 90+ score
- **Good**: 75-89 score
- **Average**: 50-74 score
- **Poor**: <50 score

#### India - BEE Star Rating
- **5 Star**: Most efficient
- **4 Star**: Very efficient
- **3 Star**: Efficient
- **2 Star**: Moderate
- **1 Star**: Basic compliance

#### EU - Energy Performance Certificate (EPC)
- **Grade A**: <30% of benchmark
- **Grade B**: 30-50% of benchmark
- **Grade C**: 50-75% of benchmark
- **Grade D**: 75-100% of benchmark
- **Grade E-G**: Above benchmark

#### China - Green Building Label
- **3 Star**: Excellence
- **2 Star**: Good
- **1 Star**: Basic

#### Japan - CASBEE / Top Runner
- **S**: Superior
- **A**: Excellent
- **B+**: Very Good
- **B-**: Good
- **C**: Fair

#### Brazil - PROCEL Edifica
- **Level A**: Most efficient
- **Level B-E**: Decreasing efficiency

### Performance Thresholds (kgCO2e/sqft/year)

| Building Type | Excellent | Good | Average | Poor |
|--------------|-----------|------|---------|------|
| Office | <10 | 10-20 | 20-35 | >50 |
| Hospital | <30 | 30-50 | 50-75 | >100 |
| Data Center | <200 | 200-400 | 400-600 | >800 |
| Retail | <15 | 15-30 | 30-45 | >60 |
| Warehouse | <5 | 5-10 | 10-20 | >30 |
| Hotel | <20 | 20-35 | 35-50 | >70 |
| Education | <12 | 12-25 | 25-40 | >55 |

---

## Testing

GreenLang includes a production-grade test suite with 200+ tests ensuring bulletproof reliability, accuracy, and maintainability.

### Test Infrastructure
- **200+ Tests**: Comprehensive unit and integration test coverage
- **Integration Tests**: Complete end-to-end workflow validation
- **Data-Driven**: All tests use actual emission factors from datasets
- **No Hardcoded Values**: All expected values sourced from data files
- **Deterministic**: Network calls blocked, LLMs mocked, seeded randomness
- **CI/CD Ready**: GitHub Actions with enforced quality gates
- **Cross-Platform**: Validated on Linux/macOS/Windows, Python 3.9-3.12

### Test Categories

#### Unit Tests (100+)
- Agent contract validation
- Boundary condition testing
- Input validation
- Mathematical invariants
- Property-based testing with Hypothesis

#### Integration Tests (70+)
- End-to-end workflow validation
- Cross-country comparisons
- Portfolio aggregation
- CLI command testing
- Parallel execution verification
- Error handling scenarios
- Performance benchmarks

#### Example Tests (30 canonical examples)
- **Core Examples (1-6)**: Basic agent functionality
- **Advanced Examples (7-18)**: Complex features and workflows
- **Property Tests (19)**: Additivity, scaling, unit round-trips
- **System Tests (20-27)**: Concurrency, caching, compatibility
- **Tutorials (28-30)**: Custom agents, country factors, XLSX export

Run example tests:
```bash
pytest -m example  # Run all 30 examples
pytest examples/tests/ex_01_fuel_agent_basic.py  # Run specific example
```

### Test Coverage & Quality Gates
- **Overall Coverage**: ≥85% (enforced in CI)
- **Agent Coverage**: ≥90% (critical components)
- **Performance**: <2s single building, <5s portfolio
- **Numerical Accuracy**: ε ≤ 1e-9 for calculations
- **Type Safety**: `mypy --strict` enforced
- **Security**: `bandit` security scanning

### Running Tests

```bash
# Install dependencies (includes test dependencies)
pip install -r requirements.txt

# Run all tests (unit + integration)
pytest

# Run with coverage report
pytest --cov=greenlang --cov-report=html --cov-fail-under=85

# Run specific test categories
pytest -m unit              # Unit tests only (100+)
pytest -m integration       # Integration tests only (70+)
pytest -m property          # Property-based tests
pytest -m performance       # Performance benchmarks

# Run integration test categories
pytest tests/integration/test_workflow_commercial_e2e.py  # Commercial E2E
pytest tests/integration/test_workflow_cli_e2e.py         # CLI testing
pytest tests/integration/test_workflow_portfolio_e2e.py   # Portfolio

# Run tests in parallel (faster)
pytest -n auto

# Run with specific markers
pytest -m "integration and not performance"  # Integration without perf tests
pytest -m "integration and timeout"          # Tests with timeout requirements

# Run quality checks
ruff check greenlang/ tests/
mypy greenlang/ --strict
black --check greenlang/ tests/
bandit -r greenlang/  # Security scanning

# Run specific test file
pytest tests/unit/agents/test_fuel_agent.py -v

# Run integration tests with detailed output
pytest tests/integration/ -v --tb=short

# Run with performance timing
pytest --benchmark-only
pytest -m performance --timeout=10
```

### Integration Test Commands

```bash
# Run all integration tests quietly
pytest -m integration -q

# Run specific integration scenarios
pytest tests/integration/test_workflow_commercial_e2e.py::TestCommercialBuildingE2E::test_happy_path_india_office

# Run cross-country comparison tests
pytest tests/integration/test_workflow_cross_country_e2e.py -v

# Run error handling tests
pytest tests/integration/test_workflow_errors_and_validation.py

# Run with snapshot updates (for golden tests)
pytest tests/integration/ --snapshot-update

# Check performance guarantees
pytest tests/integration/test_workflow_reproducibility_and_perf.py::TestReproducibilityAndPerformance::test_single_building_performance
```

### Complete Test Structure

```
tests/
├── conftest.py                                   # Shared fixtures and configuration
├── pytest.ini                                    # Pytest configuration
├── unit/                                         # Unit tests (200+)
│   ├── agents/
│   │   ├── test_fuel_agent.py                   # Data-driven fuel tests
│   │   ├── test_grid_factor_agent.py            # Factor validation
│   │   ├── test_input_validator_agent.py        # Input validation
│   │   ├── test_carbon_agent.py                 # Aggregation & percentages
│   │   ├── test_intensity_agent.py              # Intensity metrics
│   │   ├── test_benchmark_agent.py              # Basic benchmark tests
│   │   ├── test_benchmark_agent_boundaries.py   # Comprehensive boundaries
│   │   ├── test_building_profile_agent.py       # Profile generation
│   │   ├── test_recommendation_agent.py         # Recommendations
│   │   └── test_report_agent.py                 # Report generation
│   ├── core/
│   │   └── test_base_agent_contract.py          # Contract compliance
│   ├── cli/
│   │   └── test_cli_commands.py                 # CLI functionality
│   ├── sdk/
│   │   └── test_enhanced_client.py              # SDK methods
│   └── data/
│       ├── test_emission_factors_schema.py      # Factor validation
│       └── test_benchmarks_schema.py            # Benchmark validation
├── integration/                                  # Integration tests (300+)
│   ├── conftest.py                              # Integration test configuration
│   ├── test_workflow_commercial_e2e.py          # Commercial building E2E
│   ├── test_workflow_india_e2e.py               # India-specific workflows
│   ├── test_workflow_portfolio_e2e.py           # Portfolio aggregation
│   ├── test_workflow_cross_country_e2e.py       # Cross-country comparison
│   ├── test_workflow_cli_e2e.py                 # CLI end-to-end
│   ├── test_workflow_parallel_and_caching.py    # Parallel execution
│   ├── test_workflow_errors_and_validation.py   # Error handling
│   ├── test_workflow_provenance_and_versions.py # Provenance tracking
│   ├── test_workflow_reproducibility_and_perf.py # Performance tests
│   ├── test_workflow_plugins_and_contracts.py   # Plugin system
│   ├── test_workflow_assistant_mocked.py        # LLM assistant tests
│   ├── test_backward_compatibility.py           # Version compatibility
│   ├── utils/                                   # Test utilities
│   │   ├── normalizers.py                       # Data normalization
│   │   ├── io.py                                # I/O helpers
│   │   └── net_guard.py                         # Network isolation
│   └── snapshots/                               # Golden test files
│       ├── reports/                             # Reference reports
│       └── cli/                                 # CLI output snapshots
├── property/                                     # Property-based tests
│   ├── test_additivity_scaling.py               # Mathematical properties
│   ├── test_units_roundtrip.py                  # Unit conversions
│   └── test_input_validator_properties.py       # Input properties
├── fixtures/                                     # Test data
│   ├── workflows/                               # Test workflows
│   ├── data/                                    # Test building data
│   └── schemas/                                 # JSON schemas
└── performance/                                  # Performance benchmarks
    └── test_benchmarks.py                        # Speed tests
```

### Integration Test Features

1. **Bulletproof Workflow Validation**
   - Complete end-to-end workflow testing
   - Cross-country emission factor verification
   - Portfolio aggregation accuracy
   - CLI command validation
   - Natural language interface testing

2. **Numerical Invariants**
   - `sum(by_fuel) ≈ total_emissions` (ε ≤ 1e-9)
   - `total_kg / 1000 = total_tons` (exact)
   - All emissions ≥ 0
   - Percentages sum to 100% ± ε
   - Intensity = emissions / area (validated)

3. **Performance Guarantees**
   - Single building: < 2 seconds
   - Portfolio (50 buildings): < 5 seconds
   - Parallel execution verified
   - Memory usage < 100MB
   - Deterministic execution times

4. **Error Resilience**
   - Missing agent detection
   - Invalid YAML handling
   - Negative value validation
   - Circular dependency detection
   - Graceful degradation

5. **Network Isolation**
   - All socket connections blocked
   - HTTP requests mocked
   - Deterministic offline testing
   - No external dependencies

### Key Test Improvements

1. **Data-Driven Testing**
   - All factors read from `global_emission_factors.json`
   - No hardcoded values (eliminated all magic numbers)
   - Provenance fields validated (source/version/date)

2. **Comprehensive Boundary Testing**
   - All rating thresholds tested explicitly
   - Inclusive/exclusive boundaries validated
   - Table-driven tests for all countries/building types

3. **Enhanced Agent Testing**
   - Contract validation for all agents
   - Percentage sums validated to 100%±ε
   - Duplicate fuel aggregation tested
   - Empty input handling with reasons

4. **Property-Based Testing**
   - Non-negativity invariants
   - Unit round-trip conversions
   - Mathematical additivity/proportionality
   - Input validation properties

5. **End-to-End Workflows**
   - Single building complete analysis
   - Portfolio aggregation validation
   - Cross-country factor verification
   - Deterministic result checking

6. **Test Fixtures & Utilities**
   - `conftest.py`: Centralized configuration
   - `emission_factors`: Loaded from actual data
   - `benchmarks_data`: Real benchmark data
   - `agent_contract_validator`: Validates I/O
   - `snapshot_normalizer`: Removes timestamps/paths
   - `disable_network_calls`: Ensures offline testing

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
- Runs on: Ubuntu/Windows/macOS
- Python versions: 3.9, 3.10, 3.11, 3.12
- Quality checks:
  - ruff linting (fail on error)
  - mypy --strict (fail on error)
  - black formatting (fail on error)
  - Coverage ≥85% overall (enforced)
  - Coverage ≥90% agents (enforced)
  - Tests complete <90s (enforced)
  - Snapshot comparison
  - Performance benchmarks
```

### Quality Assurance

- **Deterministic**: No network calls, mocked LLMs
- **Reproducible**: Same results every run
- **Fast**: Complete suite in <90 seconds
- **Comprehensive**: 200+ tests covering all paths
- **Maintainable**: Data-driven, no magic numbers
- **Cross-platform**: Validated on all major OS

---

## Examples

### Example 1: US Office Building
```bash
# Interactive mode
greenlang calc --building --country US

# Enter when prompted:
# Building type: commercial_office
# Area: 75000
# Occupancy: 400
# Floors: 20
# Age: 25
# Electricity: 2500000 kWh
# Natural Gas: 50000 therms
```

### Example 2: India Hospital with Backup Power
```python
from greenlang.sdk.enhanced_client import GreenLangClient

client = GreenLangClient(region="IN")

hospital = {
    "metadata": {
        "building_type": "hospital",
        "area": 100000,
        "area_unit": "sqft",
        "location": {"country": "IN", "city": "Mumbai"},
        "occupancy": 500,
        "floor_count": 8,
        "building_age": 10
    },
    "energy_consumption": {
        "electricity": {"value": 3500000, "unit": "kWh"},
        "diesel": {"value": 50000, "unit": "liters"},  # Backup generators
        "lpg_propane": {"value": 1000, "unit": "kg"}   # Kitchen
    }
}

result = client.analyze_building(hospital)
print(f"BEE Star Rating: {result['data']['benchmark'].get('star_rating', 'N/A')}")
print(f"Annual Emissions: {result['data']['emissions']['total_co2e_tons']:.2f} tons")
```

### Example 3: EU Data Center with Solar
```bash
# Create data file: eu_datacenter.json
{
  "metadata": {
    "building_type": "data_center",
    "area": 50000,
    "location": {"country": "DE"},
    "floor_count": 3,
    "building_age": 5
  },
  "energy_consumption": {
    "electricity": {"value": 15000000, "unit": "kWh"},
    "district_heating": {"value": 500000, "unit": "kWh"},
    "solar_pv_generation": {"value": 1000000, "unit": "kWh"}
  }
}

# Analyze
greenlang analyze eu_datacenter.json --country EU
```

### Example 4: Global Comparison
```python
# Compare same building across countries
from greenlang.sdk.enhanced_client import GreenLangClient

countries = ["US", "IN", "EU", "CN", "JP", "BR"]
results = {}

for country in countries:
    client = GreenLangClient(region=country)
    building = {
        "metadata": {
            "building_type": "commercial_office",
            "area": 50000,
            "location": {"country": country},
            "occupancy": 200
        },
        "energy_consumption": {
            "electricity": {"value": 1500000, "unit": "kWh"}
        }
    }
    result = client.analyze_building(building)
    results[country] = result['data']['emissions']['total_co2e_tons']

# Display comparison
print("Same building, different countries:")
for country, emissions in sorted(results.items(), key=lambda x: x[1]):
    print(f"{country}: {emissions:.1f} tons CO2e/year")
```

### Example 5: Get Recommendations for Indian Office
```bash
greenlang recommend

# Enter when prompted:
# Building type: commercial_office
# Country: IN
# Building age: 20
# Performance: Average

# Output will include:
# - Install rooftop solar under government schemes
# - LED lighting upgrade
# - Smart HVAC controls
# - PAT scheme participation
```

### Example 6: Using All Commands Together
```bash
# Step 1: Check version
greenlang --version

# Step 2: List available agents
greenlang agents

# Step 3: Run building calculator
greenlang calc --building --country IN

# Step 4: Check benchmarks
greenlang benchmark --type hospital --country IN

# Step 5: Get recommendations
greenlang recommend

# Step 6: Ask AI for insights
greenlang ask "What are the best ways to reduce emissions in Indian hospitals?"
```

### Example 7: Complete Building Analysis Workflow
```python
# Python script for complete analysis
from greenlang.sdk.enhanced_client import GreenLangClient

# Initialize client for India
client = GreenLangClient(region="IN")

# Define building
building = {
    "metadata": {
        "building_type": "hospital",
        "area": 150000,
        "area_unit": "sqft",
        "location": {"country": "IN", "city": "Delhi"},
        "occupancy": 800,
        "floor_count": 10,
        "building_age": 12
    },
    "energy_consumption": {
        "electricity": {"value": 5000000, "unit": "kWh"},
        "diesel": {"value": 80000, "unit": "liters"},
        "lpg_propane": {"value": 2000, "unit": "kg"}
    }
}

# Run analysis
result = client.analyze_building(building)

# Display results
print(f"Total Emissions: {result['data']['emissions']['total_co2e_tons']:.2f} tons")
print(f"BEE Rating: {result['data']['benchmark'].get('star_rating', 'N/A')}")
print(f"Performance: {result['data']['intensity']['performance_rating']}")

# Export results
client.export_analysis(result, "delhi_hospital_analysis.xlsx", format="excel")
```

### Example 8: Portfolio Analysis
```bash
# Create portfolio file: portfolio.json
{
  "buildings": [
    {
      "id": "mumbai_office",
      "metadata": {
        "building_type": "commercial_office",
        "area": 75000,
        "location": {"country": "IN", "city": "Mumbai"}
      },
      "energy_consumption": {
        "electricity": {"value": 2000000, "unit": "kWh"}
      }
    },
    {
      "id": "delhi_hospital",
      "metadata": {
        "building_type": "hospital",
        "area": 120000,
        "location": {"country": "IN", "city": "Delhi"}
      },
      "energy_consumption": {
        "electricity": {"value": 4500000, "unit": "kWh"},
        "diesel": {"value": 60000, "unit": "liters"}
      }
    }
  ]
}

# Analyze portfolio
greenlang analyze portfolio.json --country IN
```

---

## API Reference

### GreenLangClient Methods

#### analyze_building()
```python
analyze_building(building_data: Dict) -> Dict[str, Any]
```
Comprehensive building analysis with all metrics.

#### analyze_portfolio()
```python
analyze_portfolio(buildings: List[Dict]) -> Dict[str, Any]
```
Analyze multiple buildings and aggregate metrics.

#### calculate_emissions()
```python
calculate_emissions(
    fuel_type: str,
    consumption: float,
    unit: str,
    region: Optional[str] = None
) -> Dict[str, Any]
```
Calculate emissions with regional factors.

#### get_emission_factor()
```python
get_emission_factor(
    fuel_type: str,
    unit: str,
    country: Optional[str] = None
) -> Dict[str, Any]
```
Get country-specific emission factor.

#### get_recommendations()
```python
get_recommendations(
    building_type: str,
    performance_rating: str,
    country: Optional[str] = None,
    building_age: int = 10
) -> Dict[str, Any]
```
Get optimization recommendations with roadmap.

#### benchmark_emissions()
```python
benchmark_emissions(
    total_emissions_kg: float,
    building_area: float,
    building_type: str = "commercial_office",
    period_months: int = 12
) -> Dict[str, Any]
```
Compare emissions against regional standards.

#### calculate_intensity()
```python
calculate_intensity(
    total_emissions_kg: float,
    area: float,
    occupancy: Optional[int] = None,
    floor_count: Optional[int] = None
) -> Dict[str, Any]
```
Calculate various intensity metrics.

#### export_analysis()
```python
export_analysis(
    analysis_results: Dict,
    filepath: str,
    format: str = "json"  # json, csv, excel
) -> None
```
Export results to file.

---

## Data Models

### Enhanced Building Input Structure

```python
# Comprehensive building data model (v0.0.1)
{
  "metadata": {
    "building_type": "commercial_office|hospital|data_center|retail|warehouse|hotel|education",
    "area": 50000,
    "area_unit": "sqft|sqm",
    "location": {
      "country": "US|IN|EU|CN|JP|BR|KR|UK|DE|CA|AU",
      "region": "California",
      "city": "San Francisco"
    },
    "occupancy": 200,
    "floor_count": 10,
    "building_age": 15,
    "climate_zone": "1A|2A|3A|4A|5A|6A|7|8|tropical|temperate|dry"
  },
  "energy_consumption": {
    "electricity": {"value": 1500000, "unit": "kWh|MWh|GWh"},
    "natural_gas": {"value": 25000, "unit": "therms|m3|MMBtu|kWh"},
    "diesel": {"value": 500, "unit": "liters|gallons"},
    "district_heating": {"value": 100000, "unit": "kWh|MJ|GJ"},
    "solar_pv_generation": {"value": 50000, "unit": "kWh"},
    "lpg_propane": {"value": 100, "unit": "kg|liters|cylinders"},
    "heating_oil": {"value": 200, "unit": "liters|gallons"},
    "coal": {"value": 10, "unit": "tons|kg"},
    "biomass": {"value": 5, "unit": "tons|kg"}
  }
}
```

---

## Use Cases

### Primary Applications

1. **Building Owners & Operators**
   - Assess current carbon footprint
   - Get actionable improvement roadmap
   - Track progress over time
   - Achieve certification targets

2. **Sustainability Consultants**
   - Benchmark portfolios across regions
   - Generate compliance reports
   - Identify optimization opportunities
   - Create decarbonization strategies

3. **Software Developers**
   - Integrate emissions tracking into applications
   - Build sustainability dashboards
   - Create automated reporting systems
   - Develop IoT integrations

4. **Compliance & Reporting Teams**
   - Meet regional reporting requirements
   - Generate standardized reports
   - Track against regulatory thresholds
   - Prepare for audits

5. **Investors & Financial Institutions**
   - ESG assessment and scoring
   - Carbon risk analysis
   - Portfolio emissions tracking
   - Due diligence support

---

## Migration Guide

### Upgrading from v0.0.1 to v0.0.1

Existing code continues to work with full backward compatibility. To access new features:

#### SDK Migration

```python
# Old way (v0.0.1 - still works)
from greenlang.sdk import GreenLangClient
client = GreenLangClient()
result = client.calculate_emissions("electricity", 1000, "kWh")

# New way (v0.0.1 - enhanced features)
from greenlang.sdk.enhanced_client import GreenLangClient
client = GreenLangClient(region="IN")

# Simple calculation with regional factors
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh",
    region="IN"  # Uses India's grid factor
)

# Full building analysis (new)
building_result = client.analyze_building(building_data)
```

#### CLI Migration

```bash
# Old commands (still work)
greenlang calc
greenlang run workflow.yaml

# New enhanced commands
greenlang calc --building --country IN
greenlang analyze building.json
greenlang benchmark --type hospital --country EU
greenlang recommend
```

---

## Performance Improvements

### v0.0.1 Optimizations

1. **Parallel Processing**
   - Multiple fuel calculations run concurrently
   - Agent execution parallelized where possible
   - Batch API calls for portfolio analysis

2. **Smart Caching**
   - Emission factors cached for faster lookups
   - Building profiles cached by type and region
   - Benchmark data pre-loaded on startup

3. **Optimized Data Structures**
   - Pydantic models for fast validation
   - JSON databases with indexed lookups
   - Efficient numpy arrays for calculations

4. **Regional Optimization**
   - Localized factors reduce external API calls
   - Country-specific data bundled with package
   - Offline mode for cached regions

---

## Project Structure

```
greenlang/
├── agents/                              # AI Agent modules
│   ├── __init__.py
│   ├── base_agent.py                   # Base agent class
│   ├── fuel_agent.py                   # Fuel emissions calculator
│   ├── carbon_agent.py                 # Carbon aggregator
│   ├── input_validator_agent.py        # Input validation
│   ├── report_agent.py                 # Report generation
│   ├── benchmark_agent.py              # Benchmarking
│   ├── grid_factor_agent.py           # NEW: Country-specific factors
│   ├── building_profile_agent.py      # NEW: Building categorization
│   ├── intensity_agent.py             # NEW: Intensity metrics
│   └── recommendation_agent.py        # NEW: Optimization engine
├── data/
│   ├── emission_factors.json          # Basic emission factors
│   ├── global_emission_factors.json   # NEW: 12+ countries
│   ├── global_benchmarks.json         # NEW: Regional standards
│   └── models.py                      # NEW: Pydantic models
├── cli/
│   ├── main.py                        # Enhanced CLI with new commands
│   ├── assistant.py                   # Natural language interface
│   └── dev_interface.py              # Developer UI
├── sdk/
│   ├── __init__.py                   # Basic SDK
│   └── enhanced_client.py            # NEW: Full-featured SDK
├── core/
│   ├── orchestrator.py               # Workflow orchestration
│   └── workflow.py                   # Workflow definitions
├── workflows/
│   ├── commercial_building_emissions.yaml  # NEW: Building workflow
│   ├── india_building_workflow.yaml       # NEW: India-specific
│   └── portfolio_analysis.yaml            # NEW: Portfolio workflow
├── tests/                             # NEW: Comprehensive test suite
│   ├── unit/                         # Unit tests (121+ tests)
│   ├── property/                     # Property-based tests
│   ├── fixtures/                     # Test data
│   └── utils/                        # Test utilities
└── configs/
    └── agent_config.yaml             # Agent configurations
```

---

## Configuration

### Environment Variables
```bash
export GREENLANG_REGION=IN                  # Default region (US, IN, EU, CN, JP, BR, KR)
export GREENLANG_REPORT_FORMAT=markdown     # Output format (text, markdown, json)
export GREENLANG_VERBOSE=true               # Enable verbose logging
export GREENLANG_EXPORT_PATH=./reports      # Default export directory
```

### Configuration File
Create `.greenlang.json` in home directory:
```json
{
  "region": "IN",
  "report_format": "markdown",
  "verbose": true,
  "export_path": "./reports",
  "cache_enabled": true,
  "parallel_execution": true
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Enhanced Features Not Available
```bash
# Check if enhanced agents are installed
python -c "from greenlang.agents import GridFactorAgent; print('Enhanced features available')"

# If error, reinstall
pip uninstall greenlang
pip install -e .
```

#### 2. Country Not Recognized
```python
# Check supported countries
from greenlang.sdk.enhanced_client import GreenLangClient
client = GreenLangClient()
print(client.get_supported_countries())
```

#### 3. Import Errors
```python
# For enhanced features, use:
from greenlang.sdk.enhanced_client import GreenLangClient

# NOT the old import:
# from greenlang.sdk import GreenLangClient
```

#### 4. Workflow Execution Errors
```bash
# Validate workflow first
python -c "from greenlang.core.workflow import Workflow; Workflow.from_yaml('workflow.yaml')"

# Check agent availability
greenlang agents
```

#### 5. Missing Dependencies
```bash
# Install all required packages
pip install pydantic>=2.0 pyyaml>=6.0 click>=8.0 rich>=13.0 pandas>=2.0 numpy>=1.24
```

#### 6. Command Not Working
```bash
# Error: 'calc' is not recognized as a command
# Solution: Use full command
greenlang calc  # NOT just 'calc'

# Error: 'agent' is not recognized
# Solution: Use full command with greenlang prefix
greenlang agent fuel  # NOT just 'agent fuel'
```

#### 7. Pydantic Warning
```
# Warning: Valid config keys have changed in V2: 'schema_extra' has been renamed
# Note: This is just a warning and doesn't affect functionality
# The application works correctly despite this warning
```

#### 8. Building Mode Not Available
```bash
# If calc --building doesn't work:
# Check if enhanced agents are installed
python -c "from greenlang.agents import GridFactorAgent; print('Enhanced features OK')"

# If error, reinstall package
pip uninstall greenlang
pip install -e .
```

#### 9. Country Not Supported
```bash
# Check supported countries
greenlang ask "What countries are supported?"

# Supported: US, IN, EU, CN, JP, BR, KR, UK, DE, CA, AU
```

#### 10. Workflow Execution Fails
```bash
# Validate workflow syntax
python -c "import yaml; yaml.safe_load(open('workflow.yaml'))"

# Check if all required agents are available
greenlang agents
```

---

## Version History

### v0.0.1 (Current) - Global Commercial Building Edition with Complete Type Safety (August 14, 2025)
**Complete Climate Intelligence Framework**
- ✅ Support for 12+ countries with regional emission factors
- ✅ 7 building types with specific performance profiles
- ✅ 15+ energy sources including renewables
- ✅ 9 specialized AI agents for comprehensive analysis
- ✅ Integration with regional standards (ENERGY STAR, BEE, EU EPC, etc.)
- ✅ Portfolio analysis capability
- ✅ Export to JSON, CSV, Excel formats
- ✅ Enhanced CLI with commercial building mode
- ✅ Country-specific optimization recommendations
- ✅ Climate zone adjustments for accurate calculations
- ✅ Solar PV and renewable energy offset calculations
- ✅ YAML workflow engine for automation
- ✅ Natural language AI assistant interface
- ✅ Real-time benchmarking against regional standards
- ✅ Intensity metrics (per sqft, per person, per floor)
- ✅ 100% type coverage for all public APIs
- ✅ Strict mypy checking enforced in CI/CD
- ✅ Protocol-based agent pattern with generics
- ✅ TypedDict for all structured data
- ✅ Full IDE auto-completion support

---

## Quick Reference Card

### Essential Commands
```bash
# Version and Help
greenlang --version                      # Show version (0.0.1)
greenlang --help                         # Show all commands

# Calculator Modes
greenlang calc                           # Simple emissions calculator
greenlang calc --building                # Commercial building mode
greenlang calc --building --country IN   # Building mode for India
greenlang calc --building --input data.json --output results.json

# Building Analysis
greenlang analyze building.json          # Analyze from file
greenlang analyze building.json --country US

# Benchmarking
greenlang benchmark --type hospital --country IN
greenlang benchmark --type office --country US
greenlang benchmark --list               # List all benchmarks

# Recommendations
greenlang recommend                      # Interactive recommendations

# Agent Management
greenlang agents                         # List all 9 agents
greenlang agent fuel                     # Show fuel agent details
greenlang agent grid_factor              # Show grid factor agent

# AI Assistant
greenlang ask "Calculate emissions for 100000 sqft hospital in Mumbai"
greenlang ask -v "What are emission factors for India?"

# Workflow Execution
greenlang run workflow.yaml --input data.json
greenlang init --output my_workflow.yaml

# Developer Interface
greenlang dev                            # Launch developer UI
```

### Python SDK Quick Start
```python
from greenlang.sdk.enhanced_client import GreenLangClient

# Initialize
client = GreenLangClient(region="IN")

# Quick calculation
result = client.calculate_emissions("electricity", 50000, "kWh", region="IN")

# Full analysis
building = {"metadata": {...}, "energy_consumption": {...}}
analysis = client.analyze_building(building)

# Export
client.export_analysis(analysis, "report.xlsx", format="excel")
```

---

## Type System

### Overview
GreenLang v0.0.1 includes comprehensive type hints for all public APIs, providing:
- **Type Safety**: Catch errors at development time
- **IDE Support**: Full auto-completion and IntelliSense
- **Self-documenting**: Types serve as inline documentation
- **Strict Checking**: mypy --strict enforcement in CI/CD

### Core Types
```python
from greenlang.types import (
    # Semantic units
    KgCO2e, KWh, TonsCO2e,
    
    # Literal types
    CountryCode, FuelType, BuildingType,
    
    # Result types
    AgentResult, SuccessResult, FailureResult,
    
    # Agent Protocol
    Agent, InT, OutT
)
```

### Typed SDK Usage
```python
from greenlang.sdk.client_typed import GreenLangClient
from greenlang.types import AgentResult
from greenlang.agents.types import FuelOutput

client = GreenLangClient()

# Type-safe method calls
result: AgentResult[FuelOutput] = client.calculate_emissions(
    fuel_type="electricity",  # Type-checked literal
    consumption_value=1000.0,
    consumption_unit="kWh",
    country="US"  # Type-checked country code
)

if result["success"]:
    # IDE knows result["data"] is FuelOutput
    emissions: float = result["data"]["co2e_emissions_kg"]
```

### Creating Typed Agents
```python
from greenlang.types import Agent, AgentResult

class MyAgent(Agent[MyInput, MyOutput]):
    agent_id = "my_agent"
    name = "My Custom Agent"
    version = "1.0.0"
    
    def run(self, payload: MyInput) -> AgentResult[MyOutput]:
        # Implementation with full type safety
        return {"success": True, "data": {...}}
```

### Type Checking
```bash
# Check types locally
mypy --strict greenlang/

# Run tests with type checking
pytest tests/ --mypy

# CI/CD enforces strict typing
# See .github/workflows/ci.yml
```

---

## Future Roadmap

### Planned Features

- [ ] **Scope 2 & 3 Emissions**: Supply chain and indirect emissions
- [ ] **Real-time Grid Factors**: Live API integration for current grid mix
- [ ] **ML-based Prediction**: Energy consumption forecasting
- [ ] **Carbon Offset Integration**: Direct purchase and tracking
- [ ] **Mobile SDK**: iOS and Android support
- [ ] **Blockchain Integration**: Carbon credit tokenization
- [ ] **IoT Sensor Support**: Direct meter reading integration
- [ ] **Weather Normalization**: Climate-adjusted benchmarking
- [ ] **Multi-language Support**: Localization for global markets
- [ ] **Advanced Visualizations**: Interactive dashboards and reports

---

## Support

### Resources
- **Documentation**: This comprehensive guide (GREENLANG_DOCUMENTATION.md)
- **Examples**: `/examples/` directory with sample code
- **GitHub**: https://github.com/greenlang/greenlang
- **Email**: support@greenlang.ai
- **API Status**: https://status.greenlang.ai

### Getting Help
1. Check this documentation first
2. Review the [Troubleshooting](#troubleshooting) section
3. Search existing GitHub issues
4. Create a new issue with detailed information
5. Contact support for enterprise assistance

---

## Achievements

### GreenLang v0.0.1 - Complete Feature Set with Type Safety
✅ **Global Coverage**: 12+ countries with regional emission factors  
✅ **AI Agents**: 9 specialized agents for comprehensive analysis  
✅ **Building Types**: 7 commercial building categories  
✅ **Energy Sources**: 15+ fuel types including renewables  
✅ **Regional Standards**: ENERGY STAR, BEE, EU EPC, China GBL, CASBEE, PROCEL  
✅ **Comprehensive CLI**: 10+ commands with interactive modes  
✅ **Enhanced SDK**: Full Python API with portfolio analysis  
✅ **YAML Workflows**: Automated analysis pipelines  
✅ **Export Formats**: JSON, CSV, Excel support  
✅ **Natural Language**: AI assistant for queries  
✅ **Benchmarking**: Real-time comparison with regional standards  
✅ **Recommendations**: Country-specific optimization strategies  
✅ **Intensity Metrics**: Multiple calculation methods  
✅ **Climate Zones**: Weather-adjusted calculations  
✅ **Solar Integration**: PV generation offset calculations  
✅ **Type System**: 100% typed public APIs with strict checking  
✅ **Developer Experience**: Full IDE support with auto-completion  

---

## License

MIT License - See LICENSE file for details

---

---

**GreenLang v0.0.1** - Your Complete Global Climate Intelligence Framework for Commercial Buildings 🌍🏢

*Empowering sustainable decisions with AI-driven emissions analysis across 12+ countries*