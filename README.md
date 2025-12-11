# GreenLang

**Climate Operating System** - Enterprise infrastructure for building climate-intelligent applications

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.1--intelligence-orange.svg)](https://github.com/greenlang/greenlang/releases)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/greenlang/greenlang)
[![Test Coverage](https://img.shields.io/badge/coverage-~70%25-yellow.svg)](https://github.com/greenlang/greenlang)

---

## What is GreenLang?

GreenLang is a **Climate Operating System** that provides the infrastructure, calculation engines, and regulatory frameworks needed to build production-ready climate intelligence applications. Think of it as the foundation layer for climate-aware software.

Instead of building carbon accounting, emissions tracking, and regulatory compliance from scratch, GreenLang provides:

- **Calculation Engine**: 1,000+ emission factors from authoritative sources (IPCC, DEFRA, EPA, GHG Protocol)
- **Regulatory Frameworks**: Pre-built compliance engines for CBAM, CSRD, EU Taxonomy, SB253, EUDR, and VCCI
- **Agent Architecture**: Modular, composable agents for data intake, validation, calculations, forecasting, and reporting
- **Production Infrastructure**: Authentication, encryption, monitoring, testing, and deployment tools out of the box

GreenLang is **not** a SaaS product. It's an open-source platform and SDK for developers building climate applications. We provide the building blocks; you build the experience.

**Current Status**: Beta (v0.3.1-intelligence) - Core infrastructure stable, 3 production-ready applications (CSRD 100%, CBAM 100%, VCCI 55%), 9,282 Python files, 2,313 test files. Targeting v1.0.0 GA in Q2 2026.

---

## Quick Start (5 minutes)

### Prerequisites

- Python 3.10 or higher
- pip (latest version recommended)
- (Optional) Docker for containerized deployment

### Installation

**Option 1: Minimal Install (Core Platform - No ML, 250MB smaller)**

```bash
pip install greenlang-cli
```

This installs the core platform without ML/vector database dependencies, saving 250MB of disk space.

**Option 2: Install with ML Capabilities**

```bash
# ML features (PyTorch, transformers, sentence-transformers)
pip install greenlang-cli[ml]

# Vector databases (Weaviate, ChromaDB, Pinecone, Qdrant, FAISS)
pip install greenlang-cli[vector-db]

# Full AI capabilities (LLM + ML + Vector DBs)
pip install greenlang-cli[ai-full]
```

**Option 3: Install with Full Features**

```bash
pip install greenlang-cli[full]
```

This includes analytics, LLM integration, server components, and security features (but excludes heavy ML dependencies by default).

**Option 4: Everything (Full + AI)**

```bash
pip install greenlang-cli[all]
```

**Option 5: Install from Source**

```bash
git clone https://github.com/greenlang/greenlang.git
cd greenlang
pip install -e .

# For ML/AI features:
pip install -e ".[ai-full]"
```

### Verify Installation

```bash
gl --version
# Output: GreenLang CLI v0.3.0
```

### Run Your First Calculation

```python
from greenlang.calculation import CalculationEngine
from greenlang.emission_factors import EmissionFactorLibrary

# Load emission factor library
ef_library = EmissionFactorLibrary()

# Initialize calculation engine
engine = CalculationEngine(ef_library)

# Calculate emissions for natural gas consumption
result = engine.calculate(
    activity_type="fuel_combustion",
    fuel_type="natural_gas",
    quantity=1000,  # kWh
    unit="kWh"
)

print(f"CO2e Emissions: {result.emissions_co2e} kg")
print(f"Emission Factor: {result.factor_name} ({result.factor_source})")
```

**Output:**
```
CO2e Emissions: 184.0 kg
Emission Factor: Natural Gas - Grid Average (DEFRA 2024)
```

For more examples, see [docs/QUICK_START.md](docs/QUICK_START.md).

---

## Architecture

GreenLang is built as a **modular, agent-based platform**:

```
┌─────────────────────────────────────────────────────────────┐
│                     Applications Layer                       │
│   (VCCI, CBAM, CSRD, SB253, EUDR, EU Taxonomy, ...)         │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Agent Framework                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │  Intake    │ │ Calculator │ │  Reporter  │  ...         │
│  │  Agent     │ │   Agent    │ │   Agent    │              │
│  └────────────┘ └────────────┘ └────────────┘              │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Core Infrastructure                        │
│  • Calculation Engine     • Emission Factor Library          │
│  • Data Validation        • Authentication & Security        │
│  • Monitoring & Logging   • Testing & QA                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Concepts:**

- **Agents**: Self-contained, composable modules for specific tasks (intake, validation, calculation, forecasting, reporting)
- **Calculation Engine**: Unified interface for emissions calculations across all scopes (1, 2, 3)
- **Emission Factor Library**: 1,000+ factors from authoritative sources, versioned and auditable
- **Packs**: Reusable agent configurations for common scenarios (e.g., boiler replacement, fleet electrification)

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Production-Ready Applications

GreenLang currently powers **3 production-ready climate compliance applications**:

### 1. VCCI Scope 3 Platform (GL-VCCI-Carbon-APP)

**Status**: Production-ready (95% maturity)

**Purpose**: Calculate and report **Scope 3 emissions** for value chain carbon intensity (VCCI) reporting.

**Features**:
- 15 Scope 3 categories (purchased goods, transportation, waste, etc.)
- Spend-based, activity-based, and supplier-specific calculation methods
- Monte Carlo uncertainty quantification
- API and Excel intake

**Use Cases**:
- Corporate Scope 3 emissions inventory
- Supply chain carbon intensity analysis
- Vendor emissions benchmarking

**Getting Started**: See [GL-VCCI-Carbon-APP/README.md](GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/README.md)

---

### 2. CBAM Importer Copilot (GL-CBAM-APP)

**Status**: Production-ready (95% maturity)

**Purpose**: **EU Carbon Border Adjustment Mechanism (CBAM)** compliance for importers of carbon-intensive goods.

**Features**:
- Automated CBAM reporting for steel, aluminum, cement, fertilizers, electricity, hydrogen
- Embedded emissions calculation (direct + indirect)
- CBAM Registry XML export (EU CBAM Transitional Registry format)
- Default emission factor fallbacks (EU Commission Implementing Regulation)

**Use Cases**:
- EU importers of CBAM-regulated goods
- Quarterly CBAM reports submission
- Carbon price estimation for imported goods

**Getting Started**: See [GL-CBAM-APP/README.md](GL-CBAM-APP/CBAM-Importer-Copilot/README.md)

---

### 3. CSRD Reporting Platform (GL-CSRD-APP)

**Status**: Production-ready (95% maturity)

**Purpose**: **Corporate Sustainability Reporting Directive (CSRD)** compliance and **ESRS (European Sustainability Reporting Standards)** reporting.

**Features**:
- Double materiality assessment (impact + financial materiality)
- ESRS E1 (Climate), E2 (Pollution), E3 (Water), S1 (Workforce) reporting
- Automated audit trail and data lineage
- XBRL/iXBRL export for ESEF compliance

**Use Cases**:
- EU companies subject to CSRD (10,000+ employees or 40M+ revenue)
- Sustainability report generation (ESRS-compliant)
- Audit-ready documentation for external assurance

**Getting Started**: See [GL-CSRD-APP/README.md](GL-CSRD-APP/CSRD-Reporting-Platform/README.md)

---

**Roadmap Applications** (In Development):
- **GL-SB253-APP**: California SB 253 (Climate Corporate Data Accountability Act) compliance
- **GL-EUDR-APP**: EU Deforestation Regulation (EUDR) due diligence and reporting
- **GL-Taxonomy-APP**: EU Taxonomy alignment assessment (green investment classification)

---

## Key Features

### Calculation Engine
- **1,000+ Emission Factors**: Authoritative sources (IPCC, DEFRA, EPA, GHG Protocol, ecoinvent)
- **Multi-scope Coverage**: Scope 1 (direct), Scope 2 (electricity), Scope 3 (value chain)
- **Versioned Factors**: Automatic updates, backwards compatibility, audit trails
- **Uncertainty Quantification**: Monte Carlo simulation for emissions estimates

### Agent Framework
- **Modular Design**: Mix and match agents for custom workflows
- **Composable**: Chain agents together (intake → validate → calculate → forecast → report)
- **Extensible**: Build your own agents using the Agent SDK
- **Production-Ready**: Built-in error handling, logging, monitoring, testing

### Regulatory Compliance
- **CBAM**: Carbon Border Adjustment Mechanism (EU)
- **CSRD**: Corporate Sustainability Reporting Directive (EU)
- **EU Taxonomy**: Sustainable investment classification
- **SB253**: California Climate Corporate Data Accountability Act
- **EUDR**: EU Deforestation Regulation
- **GHG Protocol**: Corporate, Scope 3, Product standards

### Security & Authentication
- **JWT Authentication**: OAuth2-compliant token-based auth
- **Role-Based Access Control (RBAC)**: Fine-grained permissions
- **AES-256 Encryption**: Data at rest and in transit (TLS 1.3)
- **Audit Logging**: Full data lineage and access logs

### Infrastructure
- **Docker**: Multi-stage builds, optimized images (<500MB)
- **Kubernetes**: Helm charts for production deployment
- **Monitoring**: Prometheus metrics, Grafana dashboards, structured logging
- **CI/CD**: GitHub Actions, automated testing (95%+ coverage)

---

## Installation Options

### Core Platform (Minimal)

```bash
pip install greenlang-cli
```

Includes:
- CLI tools (`gl` command)
- Calculation engine
- Emission factor library
- Basic data validation

### Full Platform (Recommended)

```bash
pip install greenlang-cli[full]
```

Includes:
- All core features
- Analytics (pandas, numpy)
- LLM integration (OpenAI, Anthropic, LangChain)
- Server components (FastAPI, Celery, Redis)
- Security features (JWT, encryption)

### Development Install

```bash
pip install greenlang-cli[dev]
```

Includes:
- All full platform features
- Testing tools (pytest, coverage)
- Code quality tools (mypy, ruff, black, bandit)
- Pre-commit hooks
- Jupyter notebooks

### Docker Deployment

**Core Platform:**
```bash
docker pull greenlang/greenlang:0.3.0
docker run -it greenlang/greenlang:0.3.0 gl --version
```

**Full Platform (with server):**
```bash
docker pull greenlang/greenlang:0.3.0-full
docker run -p 8000:8000 greenlang/greenlang:0.3.0-full
```

**Docker Compose (Production):**
```bash
git clone https://github.com/greenlang/greenlang.git
cd greenlang
docker-compose up -d
```

See [docs/PRODUCTION_DEPLOYMENT_GUIDE.md](docs/PRODUCTION_DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

---

## Documentation

### Getting Started
- [Quick Start Guide](docs/QUICK_START.md) - Get running in 5 minutes
- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and concepts

### Core Concepts
- [Calculation Engine](docs/CALCULATION_ENGINE.md) - Emissions calculation API
- [Emission Factors](docs/emission_factors/README.md) - Factor library and data sources
- [Agent Framework](AGENT_PATTERNS_GUIDE.md) - Building and composing agents
- [Data Pipeline Guide](docs/DATA_PIPELINE_GUIDE.md) - Intake, validation, processing

### Application Guides
- [VCCI Scope 3 Platform](GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/README.md)
- [CBAM Importer Copilot](GL-CBAM-APP/CBAM-Importer-Copilot/README.md)
- [CSRD Reporting Platform](GL-CSRD-APP/CSRD-Reporting-Platform/README.md)

### Deployment & Operations
- [Production Deployment Guide](docs/PRODUCTION_DEPLOYMENT_GUIDE.md) - Deploy to production
- [Performance Optimization](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md) - Tune for scale
- [Monitoring & Observability](docs/observability/README.md) - Prometheus, Grafana, logging
- [Security Best Practices](docs/security/SECURITY.md) - Secure your deployment

### Development
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Developer Onboarding](DEVELOPER_ONBOARDING.md) - Set up development environment
- [API Reference](docs/API_REFERENCE_COMPLETE.md) - Complete API documentation

### Migration & Compliance
- [Emission Factor Migration Guide](docs/EMISSION_FACTOR_MIGRATION_GUIDE.md) - Update emission factors
- [V1 to V2 Migration Guide](docs/V1_TO_V2_MIGRATION_GUIDE.md) - Upgrade from v1.x
- [CBAM Migration ROI](docs/CBAM_MIGRATION_ROI.md) - Business case for CBAM compliance

---

## Use Cases

### Corporate Carbon Accounting
Calculate Scope 1, 2, and 3 emissions for GHG Protocol reporting:

```python
from greenlang.applications.vcci import VCCICalculator

calculator = VCCICalculator()
result = calculator.calculate_scope3(
    category="purchased_goods",
    spend=1000000,  # USD
    industry_sector="electronics"
)

print(f"Scope 3 Emissions: {result.emissions_co2e_tonnes} tonnes CO2e")
```

### CBAM Compliance (EU Importers)
Generate quarterly CBAM reports:

```python
from greenlang.applications.cbam import CBAMReporter

reporter = CBAMReporter()
report = reporter.generate_quarterly_report(
    goods=[
        {"type": "steel", "quantity": 1000, "country": "China"},
        {"type": "aluminum", "quantity": 500, "country": "India"}
    ],
    quarter="Q1",
    year=2025
)

report.export_xml("cbam_q1_2025.xml")  # Submit to EU CBAM Registry
```

### CSRD Sustainability Reporting
Automate ESRS-compliant sustainability reports:

```python
from greenlang.applications.csrd import CSRDReporter

reporter = CSRDReporter()
report = reporter.generate_esrs_report(
    standards=["E1", "E2", "S1"],
    reporting_year=2024,
    materiality_threshold="high"
)

report.export_xbrl("csrd_2024.xbrl")  # ESEF-compliant format
```

---

## Contributing

We welcome contributions from the community. Whether you're fixing bugs, adding features, improving documentation, or building new applications, your help is appreciated.

### How to Contribute

1. **Read the [Contributing Guide](CONTRIBUTING.md)** - Understand our process
2. **Set up development environment** - Follow [Developer Onboarding](DEVELOPER_ONBOARDING.md)
3. **Pick an issue** - Check [GitHub Issues](https://github.com/greenlang/greenlang/issues) or create a new one
4. **Submit a Pull Request** - Follow our PR template and code review process

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Check code quality
ruff check .
mypy greenlang/
```

### Code Standards

- **Python 3.10+**: Modern Python features
- **Type Hints**: All public APIs must have type annotations
- **Tests**: 95%+ code coverage required
- **Documentation**: Docstrings for all public functions/classes
- **Code Style**: Black formatting, isort imports, ruff linting

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

GreenLang is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

**Key Points:**
- **Free to use**: Personal, commercial, or government use
- **Free to modify**: Create derivative works
- **Free to distribute**: Share the software
- **Patent grant**: Contributors grant patent licenses
- **No trademark license**: "GreenLang" trademark not included

**Attribution Required**: If you use GreenLang, please include the Apache 2.0 license and copyright notice in your project.

---

## Support

### Community Support

- **Documentation**: [https://greenlang.io/docs](docs/README.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/greenlang/greenlang/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/greenlang/greenlang/discussions)
- **Discord**: [Join our community](https://discord.gg/greenlang) (coming soon)

### Commercial Support

For enterprise deployments, custom development, or consulting services:

- **Email**: support@greenlang.io
- **Website**: https://greenlang.io

### Security Issues

To report security vulnerabilities, please email: **security@greenlang.io**

Do NOT open public GitHub issues for security vulnerabilities.

See [SECURITY.md](docs/SECURITY.md) for our security policy and disclosure process.

---

## Roadmap

### Current Focus (v0.3.x - Beta)

- Expand emission factor library to 10,000+ factors
- Add 3 new applications (SB253, EUDR, EU Taxonomy)
- Improve agent composability and reusability
- Production hardening (performance, monitoring, reliability)

### Next Release (v0.4.0 - Q2 2025)

- **Scope 3 Category 11**: Use of sold products (product lifetime emissions)
- **AI Agents**: Anomaly detection, forecast explanation, intelligent routing
- **Multi-tenancy**: SaaS deployment support with tenant isolation
- **GraphQL API**: Alternative to REST for complex queries

### Long-term Vision (v1.0 - 2026)

- **100+ Regulatory Frameworks**: Global coverage (US, EU, Asia, LATAM)
- **10,000+ Emission Factors**: Comprehensive, localized, industry-specific
- **Marketplace**: Community-contributed agents, packs, and applications
- **Certification Program**: Verified agents and applications for compliance

See [GL_5_YEAR_PLAN.md](GL_5_YEAR_PLAN.md) for detailed roadmap.

---

## Frequently Asked Questions

**Q: Is GreenLang a SaaS product?**

A: No. GreenLang is an **open-source platform and SDK** for developers. You host it yourself (on-premises or cloud). We provide the tools, you build the application.

**Q: Do I need to be a climate expert to use GreenLang?**

A: No. GreenLang provides the **emission factors, calculation methodologies, and regulatory frameworks** out of the box. You focus on your application logic; we handle the climate science.

**Q: How accurate are the emission factors?**

A: We source factors from **authoritative bodies** (IPCC, DEFRA, EPA, GHG Protocol, ecoinvent). Each factor includes source, version, geography, and uncertainty. You can review the [Emission Factor Library](docs/emission_factors/README.md) for details.

**Q: Can I use my own emission factors?**

A: Yes. You can extend the emission factor library with custom factors. See [Emission Factor SDK](docs/EMISSION_FACTOR_SDK.md).

**Q: Is GreenLang production-ready?**

A: **Partially**. The core platform (v0.3.0) is **beta**. Three applications (VCCI, CBAM, CSRD) are **production-ready** (95% maturity). We recommend thorough testing before production deployment.

**Q: What's the difference between GreenLang and [other carbon accounting tools]?**

A: GreenLang is **infrastructure**, not an end-user application. It's like AWS for climate software. Other tools are built on top of platforms like GreenLang (or should be).

**Q: How do I stay updated?**

A: Watch this repository on GitHub, join our [Discord](https://discord.gg/greenlang), or subscribe to our newsletter at [greenlang.io](https://greenlang.io).

---

## Acknowledgments

GreenLang is built on the work of climate scientists, regulators, and open-source contributors worldwide. We're grateful to:

- **IPCC**: Emission factor methodologies (AR5, AR6)
- **DEFRA**: UK Government greenhouse gas conversion factors
- **EPA**: US Environmental Protection Agency emission factors
- **GHG Protocol**: Corporate and Scope 3 accounting standards
- **ecoinvent**: Life cycle assessment database
- **European Commission**: CBAM, CSRD, EU Taxonomy regulations

Special thanks to the open-source community for libraries we depend on: FastAPI, Pydantic, SQLAlchemy, LangChain, NumPy, Pandas, and many more.

---

## Project Status

- **Version**: 0.3.1-intelligence (Beta)
- **License**: Apache 2.0
- **Python**: 3.10+
- **Production Applications**: 3 (CSRD 100%, CBAM 100%, VCCI 55%)
- **Codebase**: 9,282 Python files, 2,313 test files
- **Operational Agents**: 47-59
- **Emission Factors**: 1,000+ (expanding to 10,000+)
- **Test Coverage**: ~70% (target: 85%)
- **Last Updated**: December 2025

**Build Status**:
- Core Platform: Beta (Stable)
- GL-CSRD-APP: Production-Ready (100%)
- GL-CBAM-APP: Production-Ready (100%)
- GL-VCCI-APP: In Development (55%)
- GL-EUDR-APP: Planned (Q1 2026)
- GL-SB253-APP: Planned (Q2 2026)

---

**Ready to build climate-intelligent applications?**

```bash
pip install greenlang-cli[full]
gl --version
```

Start with the [Quick Start Guide](docs/QUICK_START.md) or explore the [Applications](GL-VCCI-Carbon-APP/).

**Questions?** Open an [issue](https://github.com/greenlang/greenlang/issues) or join our [Discord](https://discord.gg/greenlang).

---

**GreenLang** - Measure what matters. Act on what you measure.
