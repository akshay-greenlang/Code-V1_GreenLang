# GreenLang - The Climate Intelligence Framework

[![PyPI Version](https://img.shields.io/pypi/v/greenlang-cli.svg)](https://pypi.org/project/greenlang-cli/)
[![Python Support](https://img.shields.io/pypi/pyversions/greenlang-cli.svg)](https://pypi.org/project/greenlang-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Build climate-intelligent applications with the power of AI-driven orchestration. The LangChain of Climate Intelligence is here.**

## What is GreenLang?

GreenLang is a developer-first Climate Intelligence Framework that brings LangChain-style modularity to sustainable computing. It provides a comprehensive toolkit for building climate-aware applications across industries - from smart buildings and HVAC systems to industrial processes and renewable energy optimization.

### Key Features

- **AI-Powered Climate Intelligence**: Intelligent agents for automated emissions analysis and optimization recommendations
- **Modular Architecture**: Composable packs and pipelines for rapid development
- **Multi-Industry Support**: Buildings, HVAC, solar thermal, cement, and expanding
- **Global Coverage**: Localized emission factors for 12+ major economies
- **Developer-First Design**: Clean CLI, Python SDK, and YAML workflows
- **Type-Safe APIs**: 100% typed public interfaces with strict validation
- **Explainable Results**: Transparent calculations with full audit trails
- **Real-World Data**: Integration with industry-standard datasets and benchmarks

## Installation

```bash
# Basic installation
pip install greenlang-cli

# With analytics capabilities
pip install greenlang-cli[analytics]

# Full feature set
pip install greenlang-cli[full]

# Development environment
pip install greenlang-cli[dev]
```

## Quick Start

### CLI Usage

```bash
# Initialize a new GreenLang project
gl init my-climate-app

# Create a new pack for emissions calculation
gl pack new building-emissions

# Run emissions analysis
gl calc --building office_complex.json

# Analyze with recommendations
gl analyze results.json --format detailed

# Execute a pipeline
gl pipeline run decarbonization.yaml
```

### Python SDK

```python
from greenlang import GreenLang
from greenlang.models import Building, EmissionFactors
from greenlang.agents import BuildingAgent, HVACOptimizer

# Initialize GreenLang
gl = GreenLang()

# Create a building model
building = Building(
    name="Tech Campus A",
    area_m2=50000,
    location="San Francisco",
    building_type="office"
)

# Calculate emissions
agent = BuildingAgent()
results = agent.calculate_emissions(
    building=building,
    energy_data=energy_consumption,
    emission_factors=EmissionFactors.get_region("US-CA")
)

# Get optimization recommendations
optimizer = HVACOptimizer()
recommendations = optimizer.optimize(
    building=building,
    current_emissions=results.total_emissions,
    target_reduction=0.30  # 30% reduction target
)

print(f"Current emissions: {results.total_emissions} tCO2e/year")
print(f"Potential savings: ${recommendations.estimated_savings:,.2f}")
```

### YAML Pipelines

```yaml
# decarbonization_pipeline.yaml
version: "1.0"
name: "Building Decarbonization Analysis"

stages:
  - name: data_collection
    type: ingestion
    sources:
      - type: energy_bills
        format: csv
      - type: occupancy_sensors
        format: json

  - name: emissions_calculation
    type: calculation
    agent: BuildingAgent
    parameters:
      include_scope3: true
      use_regional_factors: true

  - name: optimization
    type: analysis
    agent: DecarbonizationAgent
    parameters:
      target_reduction: 0.40
      max_payback_years: 5

  - name: reporting
    type: output
    format: pdf
    template: executive_summary
```

## Core Concepts

### Packs
Modular, reusable components that encapsulate climate intelligence logic:
- **Calculation Packs**: Emissions calculations for specific industries
- **Optimization Packs**: Decarbonization strategies and recommendations
- **Integration Packs**: Connect to external data sources and APIs
- **Reporting Packs**: Generate customized sustainability reports

### Agents
AI-powered components that provide intelligent climate analysis:
- **BuildingAgent**: Comprehensive building emissions analysis
- **HVACOptimizer**: HVAC system optimization recommendations
- **SolarThermalAgent**: Solar thermal replacement calculations
- **PolicyAgent**: Climate policy compliance checking
- **BenchmarkAgent**: Industry and regional benchmarking

### Pipelines
Orchestrate complex climate intelligence workflows:
- Chain multiple agents and packs together
- Define conditional logic and branching
- Integrate with external systems
- Schedule recurring analyses
- Generate automated reports

## Real-World Applications

### Smart Buildings
- Real-time emissions monitoring and alerting
- Predictive maintenance for HVAC systems
- Occupancy-based energy optimization
- Automated sustainability reporting

### Industrial Decarbonization
- Process emissions calculation
- Energy efficiency recommendations
- Alternative fuel analysis
- Supply chain emissions tracking

### Renewable Energy Planning
- Solar thermal viability assessment
- Boiler replacement analysis
- Grid carbon intensity integration
- ROI calculations for green investments

## Documentation

- [Full Documentation](https://greenlang.io/docs)
- [API Reference](https://greenlang.io/api)
- [Examples Gallery](https://greenlang.io/examples)
- [Best Practices Guide](https://greenlang.io/best-practices)
- [Contributing Guide](CONTRIBUTING.md)

## Community & Support

- **Discord**: [Join our community](https://discord.gg/greenlang)
- **GitHub Issues**: [Report bugs or request features](https://github.com/greenlang/greenlang/issues)
- **Stack Overflow**: Tag questions with `greenlang`
- **Twitter**: [@GreenLangAI](https://twitter.com/GreenLangAI)

## Why GreenLang?

### For Developers
- **Rapid Development**: Build climate apps in hours, not months
- **Best Practices Built-in**: Industry standards and methodologies included
- **Extensible**: Easy to add custom agents and packs
- **Well-Documented**: Comprehensive docs with real examples

### For Organizations
- **Reduce Emissions**: Data-driven insights for real reduction
- **Save Costs**: Identify efficiency opportunities and ROI
- **Ensure Compliance**: Meet regulatory requirements
- **Transparent Reporting**: Explainable, auditable calculations

### For the Planet
- **Accelerate Net-Zero**: Enable faster climate action
- **Democratize Climate Intelligence**: Make tools accessible to all
- **Drive Innovation**: Foster new climate solutions
- **Scale Impact**: From single buildings to entire cities

## Roadmap

### Current Release (v0.2.x)
- Core CLI and SDK functionality
- Building and HVAC agents
- Basic pipeline orchestration
- Regional emission factors

### Next Release (v0.3.0)
- Kubernetes operator for cloud deployment
- Real-time grid carbon integration
- ML-powered prediction models
- Advanced visualization dashboard

### Future (v1.0.0)
- Complete industry coverage
- Global emission factor database
- Blockchain-verified carbon credits
- Enterprise governance features

## Contributing

We welcome contributions from the community! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up development environment
- Code style and standards
- Testing requirements
- Submission process

## License

GreenLang is released under the MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

GreenLang is built on the shoulders of giants:
- Climate science community for methodologies
- Open source community for inspiration
- Early adopters for invaluable feedback
- Contributors who make this possible

---

**Join us in building the climate-intelligent future. Every line of code counts.**

*Code Green. Deploy Clean. Save Tomorrow.*