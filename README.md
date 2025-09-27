# GreenLang - The Climate Intelligence Platform

[![PyPI Version](https://img.shields.io/pypi/v/greenlang-cli.svg)](https://pypi.org/project/greenlang-cli/)
[![Python Support](https://img.shields.io/pypi/pyversions/greenlang-cli.svg)](https://pypi.org/project/greenlang-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/greenlang/greenlang/ci.yml?branch=master)](https://github.com/greenlang/greenlang/actions)
[![Test Coverage](https://img.shields.io/badge/coverage-9.43%25-red)](https://github.com/greenlang/greenlang/actions)
[![Latest Release](https://img.shields.io/github/v/release/greenlang/greenlang?include_prereleases)](https://github.com/greenlang/greenlang/releases)
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

## 🚀 Quick Start - Get Running in 2 Minutes

Choose your preferred installation method and start calculating carbon emissions immediately:

### Option A: PyPI Installation (Recommended)

```bash
# Install the latest version
pip install greenlang-cli==0.3.0

# Verify installation
gl version

# Run your first calculation
python3 -c "
from greenlang.sdk import GreenLangClient
client = GreenLangClient()
result = client.calculate_carbon_footprint([
    {'fuel_type': 'electricity', 'consumption': 1000, 'unit': 'kWh'},
    {'fuel_type': 'natural_gas', 'consumption': 50, 'unit': 'therms'}
])
print(f'Total emissions: {result[\"data\"][\"total_emissions_tons\"]:.2f} metric tons CO2e')
"
```

### Option B: Docker Installation

```bash
# Pull and run with Docker
docker run --rm ghcr.io/greenlang/greenlang:0.3.0 version

# Calculate emissions using Docker
echo '{"fuels":[{"fuel_type":"electricity","consumption":1000,"unit":"kWh"}]}' | \
docker run --rm -i ghcr.io/greenlang/greenlang:0.3.0 calc --input-format json
```

**Next Steps:**
- 📖 Follow the [10-minute quickstart guide](docs/quickstart.md)
- 🎯 Try the [ready-to-run examples](examples/quickstart/)
- 📚 Read the [full documentation](https://greenlang.io/docs)

## Installation Options

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

### System Requirements
- Python 3.10 or higher
- 2GB RAM minimum (4GB recommended)
- Internet connection for emission factor updates
- Docker (optional, for containerized deployments)

## Feature Highlights

### ✨ Core Capabilities
- **🧮 Instant Calculations**: Calculate carbon footprints in milliseconds
- **🏢 Building Intelligence**: 15+ specialized agents for building optimization
- **📊 Global Coverage**: Emission factors for 12+ major economies
- **🔄 Pipeline Orchestration**: YAML-based workflows with conditional logic
- **🐳 Multi-Deployment**: Local, Docker, and Kubernetes support
- **🔒 Enterprise Security**: RBAC, audit trails, and signed artifacts

### 🛠️ Developer Experience
- **Type-Safe SDK**: 100% typed Python interfaces
- **Modular Packs**: Reusable components for rapid development
- **Rich CLI**: Intuitive command-line interface with `gl` command
- **Hot Reload**: Real-time development with instant feedback
- **Comprehensive Testing**: Built-in validation and testing framework

### Python SDK Example

```python
from greenlang.sdk import GreenLangClient

# Initialize client
client = GreenLangClient()

# Calculate building emissions
result = client.calculate_building_emissions({
    "area_m2": 5000,
    "building_type": "office",
    "electricity_kwh": 50000,
    "gas_therms": 1000,
    "location": "San Francisco"
})

print(f"Annual emissions: {result.total_emissions_tons:.1f} tCO2e")
print(f"Intensity: {result.intensity_per_sqft:.2f} kgCO2e/sqft")
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

## 📚 Documentation & Resources

### Getting Started
- **[10-Minute Quickstart](docs/quickstart.md)** - Get running immediately
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Ready-to-Run Examples](examples/quickstart/)** - Copy-paste examples

### Developer Resources
- **[Platform Documentation](https://greenlang.io/platform)** - Platform architecture and features
- **[SDK & API Reference](https://greenlang.io/sdk)** - Complete API documentation
- **[Pack Development Guide](https://greenlang.io/packs)** - Build custom components
- **[Deployment Guide](https://greenlang.io/deploy)** - Production deployment

### Advanced Topics
- **[Pipeline Specification](docs/GL_PIPELINE_SPEC_V1.md)** - YAML workflow syntax
- **[Security Model](docs/SECURITY_MODEL.md)** - Security and governance
- **[Performance Guide](docs/PERFORMANCE.md)** - Optimization best practices

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

## 🤝 Contributing

We welcome contributions from the climate tech community! Whether you're fixing bugs, adding features, or improving documentation, every contribution helps accelerate climate action.

### Quick Start for Contributors
```bash
# Clone and setup development environment
git clone https://github.com/greenlang/greenlang.git
cd greenlang
pip install -e ".[dev]"

# Run tests
pytest

# Submit your first PR
# See CONTRIBUTING.md for detailed guidelines
```

### Ways to Contribute
- **🐛 Bug Reports**: Found an issue? [Open a GitHub issue](https://github.com/greenlang/greenlang/issues)
- **💡 Feature Requests**: Have an idea? [Start a discussion](https://github.com/greenlang/greenlang/discussions)
- **📖 Documentation**: Improve guides and examples
- **🧪 Testing**: Add test coverage or performance benchmarks
- **🌍 Emission Factors**: Contribute localized data for your region

**Read our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.**

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