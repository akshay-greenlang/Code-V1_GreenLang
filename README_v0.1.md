# GreenLang v0.1: Infrastructure for Climate Intelligence

[![Version](https://img.shields.io/badge/version-0.1.0-green)](https://github.com/greenlang/greenlang)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)

## 🚀 What's New in v0.1

GreenLang has evolved into **pure infrastructure**. Domain logic now lives in **packs**.

### Architecture Change
- **Before (v0.0.1)**: Monolithic framework with built-in agents
- **Now (v0.1.0)**: Infrastructure platform + Domain packs

```
Platform = SDK/CLI/Runtime + Hub + Policy/Provenance
Success = Developer Love + Trust + Distribution
```

## 🏗️ Core Infrastructure

### Pack System
```bash
# Initialize a new pack
gl init --name emissions-tracker --type domain

# Install packs from registry
gl pack add emissions-core
gl pack add building-analysis

# List installed packs
gl pack list

# Verify pack integrity
gl pack verify emissions-core
```

### Unified CLI (`gl`)
```bash
# Run pipelines from packs
gl run emissions-core.calculate --input data.json

# Enforce policies
gl policy check runtime --file policies/resource-limits.rego

# Generate and verify provenance
gl verify artifact.json --sig artifact.json.sig

# Health check
gl doctor
```

### SDK Base Abstractions
```python
from greenlang import Agent, Pipeline, Dataset, Report

class MyAgent(Agent):
    def validate(self, input_data):
        # Validation logic
        return True
    
    def process(self, input_data):
        # Processing logic
        return output_data
```

## 📦 Pack Structure

```
my-pack/
├── pack.yaml           # Pack manifest
├── agents/            # Domain-specific agents  
├── pipelines/         # YAML workflow definitions
├── data/              # Datasets with cards
├── policies/          # OPA policy files
├── tests/             # Pack tests
└── README.md          # Pack documentation
```

### Example pack.yaml
```yaml
name: emissions-core
version: 1.0.0
type: domain
description: Core emissions calculation agents

dependencies:
  - name: greenlang-sdk
    version: ">=0.1.0"

exports:
  agents:
    - name: FuelEmissions
      class_path: agents.fuel:FuelAgent
      description: Calculate fuel-based emissions
  
  pipelines:
    - name: building-analysis
      file: pipelines/building.yaml
      description: Complete building emissions analysis

policy:
  install: policies/install.rego
  runtime: policies/runtime.rego

provenance:
  sbom: true
  signing: true
```

## 🔒 Policy & Provenance

### Policy Enforcement (OPA)
- **Install-time**: Validate pack sources and signatures
- **Runtime**: Enforce resource limits and access control
- **Data access**: Ensure compliance and residency

### Provenance by Default
- **SBOM Generation**: Track all dependencies
- **Artifact Signing**: Cryptographic signatures
- **Run Ledger**: Immutable execution history

## 🚄 Runtime Profiles

### Local Execution
```bash
gl run pipeline --profile local
```

### Kubernetes Execution
```bash
gl run pipeline --profile k8s --resource-limit memory=4Gi
```

### Cloud Functions
```bash
gl run pipeline --profile cloud --region us-west-2
```

## 🔄 Migration from v0.0.1

### For Users
1. Install v0.1: `pip install greenlang==0.1.0`
2. Install domain packs: `gl pack add emissions-core`
3. Update imports in code (see compatibility layer)

### For Pack Developers
1. Create pack structure: `gl init --name my-pack`
2. Move domain agents to `packs/my-pack/agents/`
3. Create pack.yaml manifest
4. Publish: `gl pack publish my-pack`

### Compatibility Layer
```python
# Old code (v0.0.1) - still works
from greenlang import FuelAgent

# New code (v0.1.0) - recommended
from greenlang import PackLoader
loader = PackLoader()
pack = loader.load("emissions-core")
FuelAgent = pack.agents["FuelEmissions"]
```

## 📊 Pack Ecosystem

### Official Packs
- **emissions-core**: Fuel, electricity, carbon calculations
- **building-analysis**: Commercial building assessments
- **boiler-solar**: Thermal systems and solar analysis
- **hvac-measures**: HVAC optimization measures
- **climatenza-solar**: Solar thermal feasibility

### Community Packs
Browse and install from [hub.greenlang.io](https://hub.greenlang.io)

## 🛠️ Development

### Create a Pack
```bash
# Initialize pack
gl init --name awesome-pack --type domain

# Develop locally
cd awesome-pack
pip install -e .

# Run tests
pytest tests/

# Build and publish
gl pack publish .
```

### Pack Development Tools
```python
from greenlang import Agent, Pipeline, Dataset

# Create reusable agents
class CustomAgent(Agent):
    pass

# Define pipelines
pipeline = Pipeline()
pipeline.add_agent(CustomAgent())

# Package datasets with cards
dataset = Dataset(
    path="data/emissions.csv",
    card="cards/emissions.md"
)
```

## 📈 What Was Accomplished (v0.0.1 → v0.1.0)

### Previously (v0.0.1)
✅ 11 domain agents hardcoded in framework
✅ Basic CLI with emissions calculator
✅ YAML workflow orchestration
✅ Global emissions database
✅ 200+ tests with 85% coverage

### Now (v0.1.0) 
✅ **Pack System**: Modular architecture with manifest, registry, loader
✅ **Unified CLI**: `gl` command with pack, policy, and runtime commands
✅ **Core SDK**: Domain-agnostic base abstractions
✅ **Policy Engine**: OPA-style policy enforcement
✅ **Provenance**: SBOM generation and artifact signing
✅ **Runtime Profiles**: Local, K8s, and cloud execution
✅ **Pack Registry**: Discovery and distribution mechanism
✅ **Backward Compatibility**: Transitional layer for v0.0.1 code

## 🌍 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md)

### Priority Areas
1. Pack development (new domains)
2. Runtime executors (K8s, cloud)
3. Policy templates
4. Hub/marketplace features
5. Documentation and examples

## 📖 Documentation

- [Pack Development Guide](docs/pack-development.md)
- [Policy Writing Guide](docs/policy-guide.md)
- [Runtime Profiles](docs/runtime-profiles.md)
- [API Reference](docs/api-reference.md)
- [Migration Guide](docs/migration-v0.1.md)

## 🎯 Roadmap

### v0.2.0 (Q2 2025)
- [ ] Hub/marketplace launch
- [ ] K8s operator for pack deployment
- [ ] Real OPA integration
- [ ] Sigstore/cosign integration

### v0.3.0 (Q3 2025)
- [ ] GraphQL API
- [ ] Pack dependency resolution
- [ ] Cloud-native runtime
- [ ] Multi-tenancy support

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🤝 Support

- Documentation: [docs.greenlang.io](https://docs.greenlang.io)
- Issues: [GitHub Issues](https://github.com/greenlang/greenlang/issues)
- Discussions: [GitHub Discussions](https://github.com/greenlang/greenlang/discussions)
- Email: team@greenlang.io

---

**GreenLang v0.1**: Infrastructure for Climate Intelligence
*Domain logic lives in packs. Platform enables distribution and trust.*