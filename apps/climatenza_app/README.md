# Climatenza AI - Solar Thermal Feasibility Analysis

## Overview
Climatenza AI is a comprehensive solar thermal feasibility analysis application built on the GreenLang Climate Intelligence Framework. It demonstrates the power of GreenLang's modular agent architecture for complex climate and energy calculations.

## Architecture

### Agents (in `greenlang/agents/`)
- **SiteInputAgent**: Loads and validates site configuration from YAML
- **SolarResourceAgent**: Fetches TMY solar radiation data
- **LoadProfileAgent**: Calculates hourly thermal energy demand
- **FieldLayoutAgent**: Sizes the solar collector field
- **EnergyBalanceAgent**: Performs 8760-hour energy balance simulation

### Schemas (in `climatenza_app/schemas/`)
- **FeasibilityInput**: Comprehensive Pydantic models for data validation
  - Site configuration
  - Process demand specifications
  - Boiler baseline
  - Solar system configuration
  - Financial parameters
  - Technical assumptions

### Workflows (in `climatenza_app/gl_workflows/`)
- **feasibility_base.yaml**: Orchestrates the complete feasibility analysis

### Examples (in `climatenza_app/examples/`)
- **dairy_hotwater_site.yaml**: Sample configuration for a dairy plant in India
- **data/**: Historical load profiles (2022-2024)

## Running the Application

### Using GreenLang CLI
```bash
gl run climatenza_app/gl_workflows/feasibility_base.yaml
```

### Using Test Script
```bash
cd climatenza_app
python test_workflow.py
```

## Key Features

1. **Data Validation**: Robust Pydantic schemas ensure data quality
2. **Modular Architecture**: Reusable agents for different calculations
3. **Hourly Simulation**: 8760-hour energy balance for accurate results
4. **Workflow Orchestration**: YAML-based workflow definitions
5. **Extensible Design**: Easy to add new agents and workflows

## Results

The feasibility analysis provides:
- **Solar Fraction**: Percentage of thermal demand met by solar
- **Total Solar Yield**: Annual solar energy generation (GWh)
- **Required Aperture Area**: Total collector area needed (m²)
- **Number of Collectors**: Count of solar collectors required
- **Required Land Area**: Total land needed for solar field (m²)
- **Total Annual Demand**: Baseline thermal energy demand (GWh)

## Week 1 & 2 Accomplishments

### Week 1: Foundation
✅ Project structure and configuration
✅ Comprehensive Pydantic schemas
✅ Core agent development (SiteInputAgent, SolarResourceAgent)
✅ Sample data generation

### Week 2: Energy Balance & Workflow
✅ LoadProfileAgent for demand calculation
✅ FieldLayoutAgent for system sizing
✅ EnergyBalanceAgent for hourly simulation
✅ Complete workflow orchestration
✅ Successful end-to-end execution

## Next Steps (Week 3+)
- Economic analysis agents
- Report generation with visualizations
- API endpoints for web integration
- Advanced optimization algorithms
- Real TMY data integration
- Database persistence

## Technical Notes

- Built on GreenLang v0.0.1
- Python 3.8+ required
- Uses pandas for time-series analysis
- Implements industry-standard solar thermal calculations
- Supports both hot water and steam applications

## Contributing

Climatenza AI is a showcase application for the GreenLang framework. It demonstrates best practices for building climate intelligence applications with modular, testable, and scalable architecture.