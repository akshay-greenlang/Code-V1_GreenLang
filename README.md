# GreenLang - The Climate Intelligence Platform

[![PyPI Version](https://img.shields.io/pypi/v/greenlang-cli.svg)](https://pypi.org/project/greenlang-cli/)
[![Python Support](https://img.shields.io/pypi/pyversions/greenlang-cli.svg)](https://pypi.org/project/greenlang-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Enterprise-grade climate intelligence platform for building, deploying, and managing climate-aware applications. Infrastructure-first with a powerful SDK.**

## What is GreenLang?

GreenLang is the Climate Intelligence Platform that provides managed runtime primitives, governance, and distribution for climate-aware applications. Built infrastructure-first with a comprehensive SDK, GreenLang enables organizations to deploy, manage, and scale climate intelligence across their operations - from smart buildings and HVAC systems to industrial processes and renewable energy optimization.

### Platform Capabilities

**Infrastructure & Runtime:**
- **Managed Runtime**: Deploy packs with versioning, autoscaling, and isolation
- **Policy Governance**: RBAC, capability-based security, and audit logging
- **Pack Registry**: Signed, versioned components with SBOM and dependencies
- **Multi-Backend Support**: Local, Docker, and Kubernetes deployment options
- **Observability**: Built-in metrics, tracing, and performance monitoring

**Developer SDK & Framework:**
- **AI-Powered Agents**: 15+ specialized climate intelligence components
- **Composable Packs**: Modular, reusable building blocks for rapid development
- **YAML Pipelines**: Declarative workflows with conditional logic
- **Type-Safe Python SDK**: 100% typed interfaces with strict validation
- **Global Coverage**: Localized emission factors for 12+ major economies

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

## Platform Metrics & Status

![Coverage](https://img.shields.io/badge/coverage-9.43%25-red)
![Security](https://img.shields.io/badge/security-baseline-yellow)
![Performance](https://img.shields.io/badge/P95-<5ms-green)
![Uptime](https://img.shields.io/badge/uptime-alpha-orange)

## Documentation

- [Platform Documentation](https://greenlang.io/platform)
- [SDK & API Reference](https://greenlang.io/sdk)
- [Pack Development Guide](https://greenlang.io/packs)
- [Deployment Guide](https://greenlang.io/deploy)
- [Contributing Guide](CONTRIBUTING.md)

## Community & Support

- **Discord**: [Join our community](https://discord.gg/greenlang)
- **GitHub Issues**: [Report bugs or request features](https://github.com/greenlang/greenlang/issues)
- **Stack Overflow**: Tag questions with `greenlang`
- **Twitter**: [@GreenLangAI](https://twitter.com/GreenLangAI)

## Why GreenLang Platform?

### Enterprise Infrastructure
- **Production-Ready**: Managed runtime with SLOs, versioning, and rollback
- **Governance & Security**: RBAC, audit trails, signed artifacts with SBOM
- **Scale & Performance**: Autoscaling, P95 < 5ms response times
- **Multi-Tenancy**: Org isolation, resource quotas, usage analytics

### Developer Experience
- **10x Faster Development**: Pre-built climate components and SDK
- **Platform + Framework**: Infrastructure for ops, SDK for developers
- **Best Practices Built-in**: Industry standards and methodologies included
- **Comprehensive Tooling**: CLI, Python SDK, YAML workflows, debugging tools

### Climate Impact
- **Reduce Emissions**: Data-driven insights with real reduction strategies
- **Ensure Compliance**: Meet regulatory requirements with audit trails
- **Transparent Reporting**: Explainable, verifiable calculations
- **Scale Impact**: From single buildings to entire enterprise portfolios

## Platform Roadmap

### Current Release (v0.3.0) - Foundation
- ✅ Core platform architecture with pack system
- ✅ CLI and Python SDK for developers
- ✅ 15+ climate intelligence agents
- ✅ SBOM generation and security framework
- ✅ Local and Docker runtime support

### Q1 2025 - Platform Services
- [ ] Managed runtime (beta): autoscaling, versioned deploys, org isolation
- [ ] Durable state: run history, checkpoints, replay capabilities
- [ ] Pack registry (alpha): semver, signing, install analytics
- [ ] Enhanced observability: OTel integration, cost dashboards

### Q2 2025 - Enterprise Features
- [ ] Full Kubernetes operator with CRDs
- [ ] Multi-tenancy with resource isolation
- [ ] Advanced policy engine with OPA
- [ ] SLA guarantees and status page

### v1.0.0 - Production Platform
- [ ] 50+ official packs in registry
- [ ] Global emission factor service
- [ ] ML-powered optimization engine
- [ ] Enterprise support and SLAs

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