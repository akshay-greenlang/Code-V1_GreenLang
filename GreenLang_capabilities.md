# GreenLang v0.0.1 Capability Matrix

## Overview
GreenLang is a Global Climate Intelligence Framework for Commercial Buildings that provides comprehensive tools for calculating, analyzing, and optimizing building emissions.

## Version Information
- **Version**: 0.0.1
- **Status**: Production-Ready
- **Edition**: Global Commercial Building Edition (Enhanced Mode)
- **Last Updated**: August 14, 2025
- **Type Coverage**: 100% for all public APIs

## Core Capabilities

### 1. Command-Line Interface (CLI)

| Command | Purpose | Inputs | Outputs | Status |
|---------|---------|--------|---------|--------|
| `greenlang --version` | Display version | None | Version string | ✅ Working |
| `greenlang --help` | Show all commands | None | Command list | ✅ Working |
| `greenlang calc` | Interactive emissions calculator | User prompts | Emissions results | ✅ Working |
| `greenlang calc --building` | Commercial building mode | Building data | Comprehensive analysis | ✅ Working |
| `greenlang calc --input FILE` | Calculate from file | JSON file | Emissions results | ✅ Working |
| `greenlang analyze FILE` | Analyze building from JSON | JSON file | Full analysis report | ✅ Working |
| `greenlang benchmark` | Show benchmark data | Building type, country | Benchmark thresholds | ✅ Working |
| `greenlang recommend` | Generate recommendations | Building metadata | Optimization suggestions | ✅ Working |
| `greenlang agents` | List available agents | None | Agent table | ✅ Working |
| `greenlang agent ID` | Show agent details | Agent ID | Agent configuration | ✅ Working |
| `greenlang run WORKFLOW` | Execute workflow | Workflow file, input data | Workflow results | ✅ Working |
| `greenlang ask` | AI assistant interface | Natural language query | Analysis results | ✅ Working |
| `greenlang dev` | Developer interface | None | Interactive UI | ✅ Working |
| `greenlang init` | Create sample workflow | Output path | Sample files | ✅ Working |

### 2. Intelligent Agents

| Agent ID | Class | Purpose | Inputs | Outputs | Status |
|----------|-------|---------|--------|---------|--------|
| `validator` | InputValidatorAgent | Validates input data | Raw input data | Validation results | ✅ Working |
| `fuel` | FuelAgent | Calculates fuel emissions | Fuel type, consumption, unit | CO2e emissions | ✅ Working |
| `carbon` | CarbonAgent | Aggregates emissions | Emissions list | Total CO2e, breakdown | ✅ Working |
| `report` | ReportAgent | Generates reports | Carbon data, format | Formatted report | ✅ Working |
| `benchmark` | BenchmarkAgent | Compares to standards | Emissions, area, type | Rating, intensity | ✅ Working |
| `grid_factor` | GridFactorAgent | Regional emission factors | Country/region | Grid factors | ✅ Working |
| `building_profile` | BuildingProfileAgent | Building categorization | Building metadata | Profile, expected EUI | ✅ Working |
| `intensity` | IntensityAgent | Intensity metrics | Emissions, metadata | Per sqft/person metrics | ✅ Working |
| `recommendation` | RecommendationAgent | Optimization advice | Building data, rating | Quick wins, roadmap | ✅ Working |

### 3. Python SDK (Fully Typed)

| Method | Purpose | Parameters | Returns | Status |
|--------|---------|------------|---------|--------|
| `GreenLangClient()` | Initialize client | Optional: region | Client instance | ✅ Working + Typed |
| `calculate_emissions()` | Calculate single fuel | fuel_type, consumption, unit, region | Emissions data | ✅ Working |
| `aggregate_emissions()` | Combine emissions | emissions_list | Total emissions | ✅ Working |
| `benchmark_emissions()` | Benchmark performance | emissions, area, type | Rating, intensity | ✅ Working |
| `analyze_building()` | Full building analysis | building_data | Complete analysis | ✅ Working |
| `calculate_carbon_footprint()` | Complete workflow | fuels, building_info | Full report | ✅ Working |
| `register_agent()` | Add custom agent | agent_id, agent | Success status | ✅ Working |
| `execute_workflow()` | Run workflow | workflow_id, input_data | Workflow results | ✅ Working |

### 4. Enhanced SDK (GreenLangClient - Enhanced Mode with Full Type Safety)

| Method | Purpose | Parameters | Returns | Status |
|--------|---------|------------|---------|--------|
| `analyze_building()` | Comprehensive analysis | building_data dict | Full analysis with all agents | ✅ Working |
| `quick_calculate()` | Fast calculation | fuels list | Basic emissions | ✅ Working |
| `get_grid_factor()` | Get regional factors | country | Emission factors | ✅ Working |
| `get_recommendations()` | Get suggestions | building_type, performance | Recommendations | ✅ Working |

### 5. Workflow Engine

| Feature | Description | Status |
|---------|-------------|--------|
| YAML Workflows | Define workflows in YAML | ✅ Working |
| JSON Workflows | Define workflows in JSON | ✅ Working |
| Sequential Execution | Execute steps in order | ✅ Working |
| Error Handling | Handle agent failures | ✅ Working |
| Data Passing | Pass data between steps | ✅ Working |
| Custom Workflows | Create custom workflows | ✅ Working |

### 6. Data & Calculation Features

| Feature | Description | Supported Values | Status |
|---------|-------------|------------------|--------|
| **Fuel Types** | Supported energy sources | electricity, natural_gas, diesel, propane, fuel_oil, coal, biomass, solar_pv_generation, district_heating | ✅ Working |
| **Units** | Measurement units | kWh, therms, gallons, liters, kg, m3, MMBtu | ✅ Working |
| **Regions** | Country/region support | US, IN, EU, CN, JP, BR, KR, UK, DE, CA, AU | ✅ Working |
| **Building Types** | Building categories | commercial_office, hospital, data_center, retail, warehouse, hotel, education, restaurant, industrial | ✅ Working |
| **Climate Zones** | ASHRAE zones | 1A-8, tropical, temperate, dry | ✅ Working |
| **Metrics** | Calculated metrics | Total CO2e, intensity per sqft, intensity per person, EUI | ✅ Working |
| **Benchmarks** | Performance ratings | Excellent, Good, Average, Below Average, Poor | ✅ Working |

### 7. Reporting & Output

| Format | Description | Features | Status |
|--------|-------------|----------|--------|
| Text | Console output | Formatted tables, colors | ✅ Working |
| JSON | Structured data | Complete data export | ✅ Working |
| Markdown | Documentation format | Tables, formatting | ✅ Working |
| Interactive | Rich console UI | Progress bars, panels | ✅ Working |

### 8. Testing Infrastructure (Production-Quality)

| Component | Description | Implementation | Status |
|-----------|-------------|----------------|--------|
| **Unit Tests** | 200+ tests for all components | Data-driven, no hardcoded values | ✅ Complete |
| **Boundary Tests** | All rating thresholds | Comprehensive table-driven tests | ✅ Complete |
| **Property Tests** | Mathematical invariants | Hypothesis-based, round-trips | ✅ Complete |
| **End-to-End Tests** | Complete workflows | Single building, portfolio, cross-country | ✅ Complete |
| **Contract Tests** | Agent I/O validation | All agents follow same contract | ✅ Complete |
| **Data Validation** | Schema tests | Factors, benchmarks from actual data | ✅ Complete |
| **Example Tests** | 30 canonical examples | Teaching-focused, dataset-driven | ✅ Complete |
| **Tutorial Tests** | 3 extension tutorials | Custom agent, country factors, XLSX | ✅ Complete |
| **Deterministic** | No external dependencies | Network disabled, LLMs mocked | ✅ Complete |
| **CI/CD Pipeline** | GitHub Actions | Multi-OS, Python 3.9-3.12, quality gates | ✅ Complete |
| **Coverage Gates** | Enforced minimums | ≥85% overall, ≥90% agents (enforced) | ✅ Complete |
| **Quality Checks** | Code standards | mypy --strict, ruff, black (enforced) | ✅ Complete |
| **Type Coverage** | Type hints everywhere | 100% public APIs, strict mypy | ✅ Complete |
| **Performance** | Execution time | <90 seconds (enforced in CI) | ✅ Complete |

### 10. Developer Experience

| Feature | Description | Status |
|---------|-------------|--------|
| **Type Hints** | Complete type coverage | ✅ 100% public APIs |
| **Auto-completion** | IDE support via types | ✅ Full IntelliSense |
| **Type Checking** | mypy --strict enforcement | ✅ CI/CD enforced |
| **Protocol-based** | Agent Protocol pattern | ✅ All agents typed |
| **Generic Types** | Result[T] pattern | ✅ Type-safe results |
| **TypedDict** | Structured data types | ✅ All I/O typed |
| **Documentation** | Types as documentation | ✅ Self-documenting |
| **Example Tests** | 30 typed examples | ✅ Complete |

### 11. Web Interface

| Feature | Description | Status |
|---------|-------------|--------|
| Web App | Flask-based web interface | ✅ Working |
| API Documentation | Interactive API docs | ✅ Working |
| Calculator UI | Web-based calculator | ✅ Working |

### 9. AI Assistant Features

| Feature | Description | Status |
|---------|-------------|--------|
| Natural Language | Process queries in plain English | ✅ Working |
| Agent Orchestration | Automatically select and run agents | ✅ Working |
| Context Understanding | Understand building context | ✅ Working |
| Verbose Mode | Show detailed execution | ✅ Working |

## Emission Factors Database

### Coverage
- **Countries**: 12 major economies
- **Grid Factors**: Regional electricity emission factors
- **Fuel Factors**: Standard emission factors for all fuel types
- **Update Frequency**: Static (v0.0.1)
- **Sources**: EPA, IEA, national databases

### Regional Grid Factors (kgCO2e/kWh)
- US: 0.385
- EU: 0.23
- IN: 0.71
- CN: 0.65
- JP: 0.45
- BR: 0.12
- KR: 0.49
- UK: 0.212
- DE: 0.38
- CA: 0.13
- AU: 0.66

## Benchmark Database

### Building Type Coverage
- Commercial Office
- Hospital
- Data Center
- Retail
- Warehouse
- Hotel
- Education
- Restaurant
- Industrial

### Regional Benchmarks
- Available for all 12 supported economies
- Performance categories: Excellent to Poor
- Based on kgCO2e/sqft/year metrics

## Quick Start Examples

### 1. Simple Calculation
```bash
greenlang calc
# Follow prompts for electricity, gas, diesel
```

### 2. Building Analysis
```bash
greenlang calc --building --country US
# Enter building details and energy consumption
```

### 3. File-based Analysis
```bash
greenlang analyze building_data.json --country IN
```

### 4. SDK Usage
```python
from greenlang.sdk.enhanced_client import GreenLangClient

client = GreenLangClient(region="US")
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh",
    region="US"
)
```

### 5. Workflow Execution
```bash
greenlang init
greenlang run workflow.yaml --input workflow_input.json
```

## Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| setup.py | Package configuration | Root |
| requirements.txt | Dependencies | Root |
| emission_factors.json | Emission factors data | greenlang/data/ |
| global_emission_factors.json | Regional factors | greenlang/data/ |
| global_benchmarks.json | Benchmark data | greenlang/data/ |

## Dependencies

### Core Dependencies
- Python >= 3.8
- pydantic >= 2.0
- pyyaml >= 6.0
- click >= 8.0
- rich >= 13.0

### AI Features
- openai >= 1.0
- langchain >= 0.1.0
- langchain-openai >= 0.0.5

### Data Processing
- pandas >= 2.0
- numpy >= 1.24

## Current Limitations

1. **Static Data**: Emission factors are hardcoded, not fetched from live sources
2. **Limited Coverage**: Only 12 countries/regions supported currently
3. **No Authentication**: No user authentication or multi-tenancy
4. **No Database**: No persistent storage, all calculations in-memory
5. **Basic Recommendations**: Recommendations are template-based, not ML-driven
6. **No Real-time Data**: No integration with IoT or building management systems
7. **Limited Visualizations**: No charts or graphs (text output only)

## Testing Coverage (Production-Quality)

| Component | Test Files | Tests | Implementation | Status |
|-----------|------------|-------|----------------|--------|
| **Agent Tests** | tests/unit/agents/*.py | 150+ tests | Data-driven from datasets | ✅ Complete |
| FuelAgent | test_fuel_agent.py | 20 tests | Uses factors from data | ✅ Complete |
| GridFactorAgent | test_grid_factor_agent.py | 15 tests | Validates provenance | ✅ Complete |
| InputValidatorAgent | test_input_validator_agent.py | 18 tests | Property tests added | ✅ Complete |
| CarbonAgent | test_carbon_agent.py | 12 tests | Percentages sum to 100%±ε | ✅ Complete |
| IntensityAgent | test_intensity_agent.py | 13 tests | Division by zero handled | ✅ Complete |
| BenchmarkAgent | test_benchmark_agent.py | 11 tests | Basic tests | ✅ Complete |
| BenchmarkAgent Boundaries | test_benchmark_agent_boundaries.py | 25 tests | All thresholds tested | ✅ Complete |
| BuildingProfileAgent | test_building_profile_agent.py | 10 tests | Deterministic profiles | ✅ Complete |
| RecommendationAgent | test_recommendation_agent.py | 12 tests | Stable ordering | ✅ Complete |
| ReportAgent | test_report_agent.py | 12 tests | Schema validation | ✅ Complete |
| **Framework Tests** | tests/unit/core/*.py | 12 tests | Contract compliance | ✅ Complete |
| **End-to-End Tests** | test_end_to_end.py | 8 tests | Complete workflows | ✅ Complete |
| **CLI Tests** | tests/unit/cli/*.py | 15 tests | All commands | ✅ Complete |
| **SDK Tests** | tests/unit/sdk/*.py | 12 tests | All methods | ✅ Complete |
| **Data Tests** | tests/unit/data/*.py | 24 tests | Schema validation | ✅ Complete |
| **Property Tests** | tests/property/*.py | 15 tests | Mathematical invariants | ✅ Complete |
| **Test Infrastructure** | conftest.py | N/A | Fixtures, validators, mocks | ✅ Complete |
| **CI/CD Pipeline** | .github/workflows/test.yml | N/A | Quality gates enforced | ✅ Complete |
| **Total Coverage** | All test files | 200+ tests | ≥85% overall, ≥90% agents | ✅ Enforced |

## Error Handling

- Input validation for all data types
- Graceful fallbacks for missing features
- Detailed error messages
- Agent failure recovery in workflows
- Type checking with Pydantic models

## Performance Characteristics

- **Calculation Speed**: < 1 second for single building
- **Workflow Execution**: < 5 seconds for full analysis
- **Memory Usage**: < 100MB for typical operation
- **Startup Time**: < 2 seconds

## Future Roadmap (Not Implemented)

- [ ] Live emission factor updates
- [ ] Database integration
- [ ] User authentication
- [ ] REST API endpoints
- [ ] GraphQL API
- [ ] Real-time monitoring
- [ ] ML-based recommendations
- [ ] Data visualizations
- [ ] Mobile app
- [ ] Cloud deployment
- [ ] Multi-language support
- [ ] Compliance reporting (LEED, Energy Star)
- [ ] Carbon offset integration
- [ ] Supply chain emissions
- [ ] Scope 3 emissions

## Support & Documentation

- **Version**: 0.0.1 (Production-Ready)
- **License**: MIT
- **Repository**: https://github.com/greenlang/greenlang
- **Documentation**: This file and inline code documentation
- **Support**: GitHub Issues