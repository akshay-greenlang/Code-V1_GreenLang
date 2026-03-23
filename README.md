# GreenLang

**Enterprise Climate Intelligence Platform** - 100+ AI agents powering carbon accounting, ESG compliance, and regulatory reporting

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/akshay-greenlang/Code-V1_GreenLang/releases)
[![Agents](https://img.shields.io/badge/AI%20agents-100%2B-brightgreen.svg)](#what-weve-built)
[![Tests](https://img.shields.io/badge/tests-21%2C931%2B%20passing-brightgreen.svg)](#platform-statistics)
[![Migrations](https://img.shields.io/badge/DB%20migrations-128-blue.svg)](#database--migrations)

---

## About Us

**GreenLang is a pre-seed climate technology company building the world's most comprehensive open-source climate intelligence platform.**

We believe that accurate, auditable, and automated climate accounting is the foundation for meaningful climate action. Today, enterprises face a tsunami of climate regulations - CSRD, EUDR, CBAM, GHG Protocol, ISO 14064, CDP, TCFD, SBTi, EU Taxonomy, California SB 253 - each with complex, overlapping requirements. Most organizations are stuck with spreadsheets, consultants, and fragmented tools that can't scale.

**GreenLang changes this.** We're building a deterministic execution engine and agent-based platform where every calculation is reproducible, every number is traceable, and every output is audit-ready. Our platform is not a black box - it's an open, inspectable infrastructure that enterprises can trust for regulatory compliance.

### Our Vision

> **Make climate compliance as automated and reliable as financial accounting.**

We envision a world where:
- Every enterprise can accurately measure, report, and reduce their emissions across all scopes
- Regulatory compliance is automated, not manual - adapting in real-time as regulations evolve
- Supply chain transparency is the default, not the exception
- Climate data is deterministic, reproducible, and audit-ready by design

### Why We're Building This

The climate compliance market is exploding. The EU alone is rolling out CSRD (50,000+ companies), EUDR (every importer of 7 commodities), CBAM (all carbon-intensive imports), and EU Taxonomy (all financial institutions). In the US, California's SB 253 mandates Scope 1-3 reporting for 5,000+ companies. These regulations have hard deadlines, steep penalties, and no room for error.

Yet the tooling landscape is fragmented: point solutions that cover one regulation, require manual data entry, and produce non-auditable outputs. GreenLang is the horizontal platform that powers compliance across all of these regulations from a single, unified data and calculation layer.

---

## What We've Built

In 6 months of intensive development, we've built a production-grade platform with **100+ specialized AI agents**, **10 compliance applications**, **128 database migrations**, and **21,931+ automated tests**. Here's what exists today:

### MVP Quick Start (CBAM Flagship Loop)

Use one of these supported paths:

```bash
# Option A: Monorepo canonical CLI path
python -m greenlang.cli.main run cbam "cbam-pack-mvp/examples/sample_config.yaml" "cbam-pack-mvp/examples/sample_imports.csv" out

# Option B: Standalone CBAM CLI
cd cbam-pack-mvp
pip install -e ".[web,dev]"
gl-cbam run cbam --config examples/sample_config.yaml --imports examples/sample_imports.csv --out ./out

# Optional: launch web UI
gl-cbam web --host 127.0.0.1 --port 8000
```

CBAM outputs are generated in the selected output directory:
- `cbam_report.xml`
- `report_summary.xlsx`
- `audit/*` (claims, lineage, assumptions, policy validation, run manifest, checksums)
- `evidence/*` (immutable copies of inputs)

### Infrastructure Layer (10 Components - All Production Ready)

The foundation that everything runs on:

| Component | Technology | What It Does |
|-----------|-----------|-------------|
| **Container Orchestration** | Kubernetes on EKS | Auto-scaling agent deployment with Helm charts |
| **Primary Database** | PostgreSQL + TimescaleDB | Time-series emissions data with hypertable partitioning |
| **Caching Layer** | Redis Cluster | Sub-millisecond emission factor lookups, session management |
| **Object Storage** | S3-compatible | Document storage for PDFs, satellite imagery, reports |
| **Vector Database** | pgvector (HNSW) | Semantic search across 100K+ emission factors (384d/768d/1536d embeddings) |
| **API Gateway** | Kong | Rate limiting, auth, request routing for all 100+ agents |
| **CI/CD** | GitHub Actions | Automated testing, security scanning, deployment pipelines |
| **Feature Flags** | Custom engine | Gradual rollout of new calculation methodologies |
| **Log Aggregation** | Loki + Promtail | Centralized logging across all agent instances |
| **Agent Factory** | Custom framework | Dynamic agent instantiation, lifecycle management, health checks |

### Security Layer (11 Components - All Complete)

Enterprise-grade security built from day one:

| Component | Implementation | Details |
|-----------|---------------|---------|
| **Authentication** | JWT with RS256 | Token-based auth with refresh rotation |
| **Authorization** | RBAC | 10 roles, 61 permissions, hierarchical access control |
| **Encryption at Rest** | AES-256-GCM | All sensitive data encrypted with key rotation |
| **Transport Security** | TLS 1.3 | End-to-end encryption, certificate management |
| **Audit Logging** | Custom engine | 70+ auditable event types with tamper-proof storage |
| **Secrets Management** | HashiCorp Vault | Dynamic secrets, lease management, auto-rotation |
| **Security Scanning** | SAST/DAST/SCA pipeline | Automated vulnerability detection in CI/CD |
| **Security Policies** | 18 policy documents | Incident response, data classification, access control |
| **SOC 2 Preparation** | Type II readiness | Controls mapping, evidence collection, gap analysis |
| **Security Operations** | Automated response | Threat detection, incident triage, automated remediation |
| **PII Protection** | Detection & Redaction | Pattern-based PII detection across all data pipelines |

### Observability Layer (5 Components - All Complete)

Full visibility into platform operations:

| Component | Technology | Capabilities |
|-----------|-----------|-------------|
| **Metrics** | Prometheus HA + Thanos | Long-term metrics storage, cross-cluster federation |
| **Dashboards** | Grafana 11.4 | Pre-built dashboards for every agent and application |
| **Distributed Tracing** | OpenTelemetry + Tempo | Request tracing across agent pipelines |
| **Alerting** | Multi-channel | Slack, email, PagerDuty, webhooks, Teams, SMS |
| **SLO/SLI** | Error budget tracking | Burn-rate alerts, availability targets per service |

### Foundation Agents (10 Agents - All Complete)

The cross-cutting agents that every other agent depends on:

| Agent | Purpose |
|-------|---------|
| **Orchestrator** | DAG-based pipeline execution, dependency resolution, parallel processing |
| **Schema Compiler** | Validates all data against GreenLang schemas before processing |
| **Unit Normalizer** | Converts between 1000+ unit combinations (kg, tonnes, MWh, GJ, etc.) |
| **Assumptions Registry** | Tracks every assumption made in calculations with justification |
| **Citations Engine** | Links every output to its authoritative source (IPCC, DEFRA, EPA) |
| **Access Policy Guard** | Enforces RBAC at the agent level, data-level access control |
| **Agent Registry** | Service catalog, health monitoring, version management for all agents |
| **Reproducibility Agent** | Ensures byte-identical results across runs with same inputs |
| **QA Test Harness** | Automated validation of agent outputs against known baselines |
| **Telemetry Agent** | Per-agent metrics, traces, and logging integration |

### Data Agents (20 Agents - All Complete)

Everything needed to ingest, clean, validate, and transform enterprise data:

**Data Intake (7 agents):**
- PDF & Invoice Extractor - Extracts emissions data from invoices, utility bills, transport docs
- Excel/CSV Normalizer - Handles 50+ column naming conventions across industries
- ERP/Finance Connector - SAP, Oracle, Workday integration for spend and activity data
- API Gateway Agent - RESTful data ingestion with schema validation
- EUDR Traceability Connector - Supply chain data from EUDR-specific sources
- GIS/Mapping Connector - Geospatial data for land use and deforestation monitoring
- Satellite Imagery Connector - Sentinel-2, Landsat integration for forest cover analysis

**Data Quality (12 agents):**
- Supplier Questionnaire Processor - Parses and validates supplier sustainability surveys
- Spend Data Categorizer - Maps procurement data to GHG Protocol categories
- Data Quality Profiler - Statistical profiling, completeness scoring, anomaly flagging
- Duplicate Detection - Fuzzy matching to prevent double-counting emissions
- Missing Value Imputer - Statistical imputation with uncertainty propagation
- Outlier Detection - Z-score, IQR, isolation forest methods for emissions anomalies
- Time Series Gap Filler - Interpolation for missing monthly/quarterly data
- Cross-Source Reconciliation - Validates data consistency across multiple sources
- Data Freshness Monitor - Alerts when data sources go stale
- Schema Migration Agent - Handles evolving data schemas without data loss
- Data Lineage Tracker - Full provenance from raw input to final output
- Validation Rule Engine - 500+ configurable validation rules

**Geospatial (1 agent):**
- Climate Hazard Connector - Physical risk data from climate models

### MRV Agents (30 Agents - All Complete)

Full GHG Protocol coverage across all scopes and categories:

**Scope 1 - Direct Emissions (8 agents):**

| Agent | GHG Category | Methodologies |
|-------|-------------|---------------|
| Stationary Combustion | Boilers, furnaces, generators | IPCC Tier 1-3, EPA Part 98 |
| Mobile Combustion | Fleet vehicles, equipment | DEFRA, EPA, distance/fuel-based |
| Process Emissions | Chemical/industrial processes | Sector-specific factors |
| Fugitive Emissions | Leaks from equipment, pipelines | EPA Method 21, LDAR |
| Refrigerants & F-Gas | HFC, PFC, SF6 releases | IPCC AR6 GWP values |
| Land Use Change | LULUCF activities | IPCC Land Use guidance |
| Waste Treatment | On-site waste processing | IPCC Waste sector methods |
| Agricultural Emissions | Enteric fermentation, manure, soil | IPCC Agriculture guidance |

**Scope 2 - Indirect Energy (5 agents):**

| Agent | Approach | Standards |
|-------|----------|-----------|
| Location-Based | Grid average factors | IEA, EPA eGRID, national grids |
| Market-Based | Contractual instruments | RE-DISS, GOs, RECs, PPAs |
| Steam/Heat Purchase | District heating, CHP | IPCC stationary combustion |
| Cooling Purchase | District cooling | Efficiency-adjusted factors |
| Dual Reporting Reconciliation | Location vs market comparison | GHG Protocol Scope 2 Guidance |

**Scope 3 - Value Chain (15 agents):**

| Cat | Agent | Coverage |
|-----|-------|----------|
| 1 | Purchased Goods & Services | Spend-based, hybrid, supplier-specific methods |
| 2 | Capital Goods | Asset-level lifecycle emissions |
| 3 | Fuel & Energy Activities | T&D losses, WTT factors |
| 4 | Upstream Transportation | tonne-km, vehicle-km methods |
| 5 | Waste Generated in Operations | Landfill, incineration, recycling, composting |
| 6 | Business Travel | Air, rail, hotel, car rental emissions |
| 7 | Employee Commuting | Survey-based, distance-based, national averages |
| 8 | Upstream Leased Assets | Asset-specific, area-based methods |
| 9 | Downstream Transportation | Customer distribution emissions |
| 10 | Processing of Sold Products | Intermediate product transformation |
| 11 | Use of Sold Products | Direct/indirect use-phase emissions |
| 12 | End-of-Life Treatment | Disposal pathway modeling |
| 13 | Downstream Leased Assets | Tenant emission allocation |
| 14 | Franchises | Franchise-level Scope 1+2 rollup |
| 15 | Investments | Equity, debt, project finance methods |

**Cross-Cutting (2 agents):**
- Scope 3 Category Mapper - Auto-categorizes activities to the correct Scope 3 category
- Audit Trail & Lineage - Complete calculation provenance for auditor review

### EUDR Agents (40 Agents - All Complete)

The most comprehensive EU Deforestation Regulation compliance suite available:

**Supply Chain Traceability (15 agents):**
GPS Coordinate Validator, Plot Boundary Mapper, Geolocation Verification, Forest Cover Analysis, Satellite Monitoring, Land Use Change Detector, Supply Chain Mapper, Chain of Custody Tracker, Mass Balance Calculator, Segregation Verifier, Multi-Tier Supplier Manager, QR Code Generator, Reference Number Generator, Blockchain Integration, Mobile Data Collector

**Risk Assessment (5 agents):**
Country Risk Evaluator, Commodity Risk Analyzer, Corruption Index Monitor, Deforestation Alert System, Supplier Risk Scorer

**Due Diligence (6 agents):**
Risk Assessment Engine, Information Gathering Coordinator, Due Diligence Statement Creator, Legal Compliance Verifier, Protected Area Validator, Document Authentication

**Support & Workflow (14 agents):**
Risk Mitigation Advisor, Improvement Plan Creator, Mitigation Measure Designer, Stakeholder Engagement, Indigenous Rights Checker, Grievance Mechanism Manager, Third-Party Audit Manager, Customs Declaration Support, EU Information System Interface, Annual Review Scheduler, Continuous Monitoring, Documentation Generator, Authority Communication Manager, Due Diligence Orchestrator

### Compliance Applications (10 Apps - All Built)

Production-ready applications built on the agent platform:

| Application | Regulation | Status | Key Features |
|------------|-----------|--------|-------------|
| **GL-CSRD-APP** v1.1 | EU Corporate Sustainability Reporting Directive | Production | ESRS standards, double materiality, XBRL export |
| **GL-CBAM-APP** v1.1 | EU Carbon Border Adjustment Mechanism | Production | Quarterly reporting, embedded emissions, XML export |
| **GL-VCCI-APP** v1.1 | Value Chain Carbon Intelligence | Production | Scope 3 hotspot analysis, supplier engagement |
| **GL-EUDR-APP** v1.0 | EU Deforestation Regulation | Built | Due diligence statements, geolocation verification |
| **GL-GHG-APP** v1.0 | GHG Protocol Corporate Standard | Built | Scope 1-3 inventory, organizational boundaries |
| **GL-ISO14064-APP** v1.0 | ISO 14064 Verification | Built | Verification-ready reports, uncertainty analysis |
| **GL-CDP-APP** v1.0 | CDP Climate Disclosure | Beta | Questionnaire automation, scoring optimization |
| **GL-TCFD-APP** v1.0 | Task Force on Climate Disclosures | Beta | Scenario analysis, physical/transition risk |
| **GL-SBTi-APP** v1.0 | Science Based Targets Initiative | Beta | Target setting, progress tracking, SDA/ACA methods |
| **GL-Taxonomy-APP** v1.0 | EU Taxonomy Alignment | Alpha | DNSH assessment, substantial contribution, GAR/BTAR |

### Database & Migrations

**128 versioned SQL migrations** covering the entire platform:

| Range | Coverage |
|-------|----------|
| V001-V006 | Core schema, tenants, organizations |
| V007-V008 | Feature flags, agent factory |
| V009-V018 | Security (auth, RBAC, encryption, audit, vault) |
| V019-V020 | Observability (metrics, dashboards) |
| V021-V030 | Foundation agents (orchestrator, schema, registry) |
| V031-V050 | Data agents (intake, quality, lineage) |
| V051-V081 | MRV agents (Scope 1, 2, 3 - all 30 agents) |
| V082-V088 | Applications (EUDR, GHG, ISO14064, CDP, TCFD, SBTi, Taxonomy) |
| V089-V128 | EUDR agents (40 agents - traceability, risk, due diligence) |

---

## Platform Statistics

| Metric | Value |
|--------|-------|
| **Total AI Agents** | 100+ (10 foundation + 20 data + 30 MRV + 40 EUDR) |
| **Python Files** | 28,000+ |
| **Core Library Files** | 4,400+ |
| **Test Files** | 2,500+ |
| **Automated Tests** | 21,931+ passing |
| **Database Migrations** | 128 |
| **Compliance Applications** | 10 |
| **Infrastructure Components** | 10 (all production ready) |
| **Security Components** | 11 (all complete) |
| **Observability Components** | 5 (all complete) |
| **EUDR Agent Files** | 964 |
| **Emission Factors** | 1,000+ (IPCC, DEFRA, EPA, GHG Protocol) |
| **Supported Regulations** | 10 (CSRD, EUDR, CBAM, GHG, ISO 14064, CDP, TCFD, SBTi, Taxonomy, SB253) |
| **License** | Apache 2.0 |

---

## Architecture

```
+=====================================================================+
|                        APPLICATIONS (10)                             |
|  CSRD | CBAM | VCCI | EUDR | GHG | ISO14064 | CDP | TCFD | SBTi | Tax |
+=====================================================================+
                                |
+=====================================================================+
|                     AGENT LAYERS (100+ Agents)                       |
|  +------------------+  +------------------+  +------------------+   |
|  | EUDR Agents (40) |  | MRV Agents (30)  |  | Data Agents (20) |   |
|  | Traceability     |  | Scope 1 (8)      |  | Intake (7)       |   |
|  | Risk Assessment  |  | Scope 2 (5)      |  | Quality (12)     |   |
|  | Due Diligence    |  | Scope 3 (15)     |  | Geospatial (1)   |   |
|  | Workflow (14)    |  | Cross-Cut (2)    |  |                  |   |
|  +------------------+  +------------------+  +------------------+   |
|                    Foundation Agents (10)                             |
|  Orchestrator | Schema | Units | Assumptions | Citations | RBAC     |
|  Registry | Reproducibility | QA Harness | Telemetry                |
+=====================================================================+
                                |
+=====================================================================+
|                     INFRASTRUCTURE                                    |
|  +------------+  +------------+  +------------+  +------------+     |
|  | Security   |  |Observabil- |  | Data       |  | Deployment |     |
|  | (11 comp.) |  |ity (5)     |  | (Postgres, |  | (K8s, EKS, |    |
|  | JWT, RBAC, |  | Prometheus,|  |  Redis,    |  |  Helm,     |     |
|  | AES-256,   |  | Grafana,   |  |  pgvector, |  |  Terraform)|    |
|  | Vault, PII |  | OTel, SLO  |  |  S3)       |  |            |     |
|  +------------+  +------------+  +------------+  +------------+     |
+=====================================================================+
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.10, 3.11, 3.12 |
| **Framework** | FastAPI, Pydantic v2 |
| **Database** | PostgreSQL 16 + TimescaleDB + pgvector |
| **Cache** | Redis 7 Cluster |
| **Search** | pgvector HNSW (cosine similarity) |
| **Orchestration** | Kubernetes (EKS), Helm, Kustomize |
| **IaC** | Terraform |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus, Grafana 11.4, OpenTelemetry, Tempo, Loki |
| **Security** | JWT (RS256), RBAC, AES-256-GCM, TLS 1.3, HashiCorp Vault |
| **Testing** | pytest, pytest-asyncio, hypothesis |
| **AI/ML** | LLM integration (OpenAI, Anthropic), satellite ML models |

---

## 2026 Roadmap

### Q1 2026 (Completed)
- [x] EUDR Agent Suite (40 agents) - supply chain traceability, risk assessment, due diligence
- [x] MRV Agent Suite (30 agents) - full GHG Protocol Scope 1, 2, 3 coverage
- [x] 10 compliance applications (CSRD, CBAM, VCCI, EUDR, GHG, ISO14064, CDP, TCFD, SBTi, Taxonomy)
- [x] Complete security stack (11 components, SOC 2 readiness)
- [x] Full observability platform (Prometheus, Grafana, OTel, alerting, SLO/SLI)
- [x] 128 database migrations deployed

### Q2 2026 (In Progress)
- [ ] GL-SB253-APP - California SB 253 climate disclosure (deadline: June 30, 2026)
- [ ] GL-BuildingBPS-APP - Building Performance Standards compliance
- [ ] Production deployment and beta testing with pilot customers
- [ ] Pack Hub marketplace launch - reusable agent bundles
- [ ] Performance optimization and load testing at scale

### Q3-Q4 2026 (Planned)
- [ ] GL-CSDDD-APP - EU Corporate Sustainability Due Diligence (deadline: July 2027)
- [ ] GL-GreenClaims-APP - Greenwashing compliance (deadline: Sept 2026)
- [ ] GL-ProductPCF-APP - Product Carbon Footprint & Digital Product Passports
- [ ] Multi-tenant SaaS deployment for enterprise customers
- [ ] Advanced ML models for satellite-based deforestation detection
- [ ] Real-time supply chain monitoring and alerting
- [ ] Integration marketplace (SAP, Oracle, Workday connectors)

---

## Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip (latest version)
- (Optional) Docker for containerized deployment

### Installation

```bash
# Install from PyPI
pip install greenlang-cli

# Or install from source
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang
pip install -e ".[full]"

# Verify
gl --version
```

### Run Your First Calculation

```python
from greenlang.agents.calculation.emissions import EmissionsCalculator

calculator = EmissionsCalculator()
result = calculator.calculate(
    activity_type="fuel_combustion",
    fuel_type="natural_gas",
    quantity=1000,  # kWh
    unit="kWh"
)

print(f"CO2e Emissions: {result.emissions_co2e} kg")
print(f"Source: {result.factor_source}")
# Output: CO2e Emissions: 184.0 kg
# Output: Source: DEFRA 2024
```

---

## Development

```bash
# Setup
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -e ".[dev]"

# Run tests
pytest tests/                 # All tests
pytest tests/ -x --tb=short  # Stop on first failure
pytest --cov=greenlang       # With coverage

# Code quality
ruff check .                 # Linting
black greenlang/             # Formatting
mypy greenlang/              # Type checking
```

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/akshay-greenlang/Code-V1_GreenLang/issues)
- **Security**: security@greenlang.io

---

**Version**: 0.3.0 | **Last Updated**: March 2026 | **Stage**: Pre-Seed

**GreenLang** - Measure what matters. Act on what you measure.
