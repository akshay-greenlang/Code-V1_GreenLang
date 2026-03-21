# PACK-024: Carbon Neutral Pack

**Comprehensive carbon neutrality management powered by GreenLang AI agents**

## Overview

PACK-024 provides a complete, standalone carbon neutrality management solution
covering the full journey from GHG footprint quantification through carbon credit
procurement and retirement to verified neutrality claims. It implements the
requirements of ISO 14068-1:2023 (Carbon neutrality) and PAS 2060:2014
(Demonstrating carbon neutrality), with credit quality scoring per ICVCM Core
Carbon Principles, claims validation per VCMI Claims Code of Practice, and
verification package assembly per ISAE 3410.

| Metric | Value |
|--------|-------|
| Engines | 10 |
| Workflows | 8 |
| Templates | 10 |
| Integrations | 12 |
| Presets | 8 |
| Total files | 78 |
| Lines of code | ~47K |
| Tests | 693 |
| MRV agents | 30 (all Scope 1/2/3) |
| DATA agents | 20 (intake + quality) |
| Foundation agents | 10 (platform services) |
| Registries | 6 (Verra, Gold Standard, ACR, CAR, Puro.earth, Isometric) |
| Neutrality types | 8 (Corporate, SME, Event, Product, Building, Service, Project, Portfolio) |

## Architecture

```
+-----------------------------------------------------------------------+
|                     PACK-024 Carbon Neutral Pack                      |
+-----------------------------------------------------------------------+
|                                                                       |
|  +-------------------+   +-------------------+   +------------------+ |
|  | Footprint         |   | Carbon Mgmt Plan  |   | Credit Quality   | |
|  | Quantification    |-->| Engine            |-->| Engine           | |
|  | Engine            |   | (MACC/Reduction)  |   | (ICVCM 12-dim)  | |
|  +-------------------+   +-------------------+   +------------------+ |
|          |                        |                       |           |
|          v                        v                       v           |
|  +-------------------+   +-------------------+   +------------------+ |
|  | Portfolio          |   | Registry          |   | Neutralization   | |
|  | Optimization      |-->| Retirement        |-->| Balance          | |
|  | Engine            |   | Engine (6 regs)   |   | Engine           | |
|  +-------------------+   +-------------------+   +------------------+ |
|          |                        |                       |           |
|          v                        v                       v           |
|  +-------------------+   +-------------------+   +------------------+ |
|  | Claims             |   | Verification      |   | Annual Cycle     | |
|  | Substantiation    |-->| Package           |-->| Engine           | |
|  | Engine            |   | Engine (ISAE3410) |   |                  | |
|  +-------------------+   +-------------------+   +------------------+ |
|                                                          |           |
|                                        +------------------+          |
|                                        | Permanence Risk  |          |
|                                        | Engine           |          |
|                                        +------------------+          |
|                                                                       |
+-----+---------+---------+---------+--------+--------+--------+-------+
      |         |         |         |        |        |        |
      v         v         v         v        v        v        v
  30 MRV    20 DATA   10 FOUND  GL-GHG   PACK   PACK   PACK    6
  Agents    Agents    Agents    APP      -021   -022   -023  Registries
```

## Quick Start

```python
from packs.net_zero.PACK_024_carbon_neutral import (
    CarbonNeutralConfig,
    load_preset,
)

# 1. Load a neutrality-type preset
config = load_preset("corporate_neutrality")

# 2. Validate configuration
warnings = validate_config(config)

# 3. List enabled engines
engines = config.pack.get_enabled_engines()
print(f"Enabled engines: {len(engines)}")

# 4. Compute provenance hash for reproducibility
config_hash = config.get_config_hash()
print(f"Config hash: {config_hash[:16]}...")
```

## Installation

### Prerequisites

- Python >= 3.11
- PostgreSQL >= 16 with pgvector and TimescaleDB extensions
- Redis >= 7
- GreenLang Platform >= 2.0.0

### Install Steps

```bash
# 1. Verify Python version
python --version  # Requires 3.11+

# 2. Install Python dependencies
pip install pydantic>=2.0 pyyaml>=6.0 pandas>=2.0 numpy>=1.24 \
    httpx>=0.24 psycopg[binary]>=3.1 psycopg_pool>=3.1 redis>=5.0 \
    jinja2>=3.1 openpyxl>=3.1 cryptography>=41.0

# 3. Apply database migrations
#    Inherited: V001-V006, V007-V008, V009-V010, V019-V020,
#               V021-V030, V031-V050, V051-V081, V082-V088
#    New: V084-PACK024-001 through V084-PACK024-010

# 4. Seed reference data
#    - IPCC AR6 GWP-100 values
#    - Emission factor databases (DEFRA, EPA, ecoinvent)
#    - ICVCM CCP 12-dimension definitions
#    - Credit project type configurations
#    - Registry configurations (6 registries)
#    - VCMI Claims Code criteria
#    - Permanence risk parameters

# 5. Run health check
python -c "
from packs.net_zero.PACK_024_carbon_neutral.integrations import CarbonNeutralHealthCheck
hc = CarbonNeutralHealthCheck()
result = hc.run()
print(f'Health: {result.overall_status}')
"
```

### Infrastructure Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU cores | 4 | 8 |
| Memory (GB) | 8 | 16 |
| Storage (GB) | 50 | 200 |
| DB connections | 20 | 60 |

## Component Overview

### Engines (10)

| # | Engine | Description | Standard |
|---|--------|-------------|----------|
| 1 | Footprint Quantification | Scope 1/2/3 GHG quantification with DQIS scoring | ISO 14064-1:2018 |
| 2 | Carbon Management Plan | Reduction-first strategy with MACC curve analysis | ISO 14068-1, PAS 2060 |
| 3 | Credit Quality | 12-dimension ICVCM CCP scoring (A+ to F rating) | ICVCM CCP 2023 |
| 4 | Portfolio Optimization | Markowitz-inspired avoidance/removal allocation | Oxford Principles |
| 5 | Registry Retirement | 6-registry retirement with serial number tracking | Registry standards |
| 6 | Neutralization Balance | Footprint vs. credits balance calculation | ISO 14068-1, PAS 2060 |
| 7 | Claims Substantiation | Claim validation with VCMI tier assessment | VCMI Claims Code |
| 8 | Verification Package | ISAE 3410 evidence assembly with SHA-256 hashing | ISAE 3410 |
| 9 | Annual Cycle | Multi-year renewal with milestone tracking | ISO 14068-1 |
| 10 | Permanence Risk | Buffer pool calculation with reversal monitoring | Oxford Principles |

### Workflows (8)

| # | Workflow | Phases | Schedule | Duration |
|---|----------|--------|----------|----------|
| 1 | Full Annual Cycle | 10 | Annual | 120 min |
| 2 | Footprint Assessment | 4 | Annual | 60 min |
| 3 | Carbon Mgmt Plan | 5 | Annual | 45 min |
| 4 | Credit Procurement | 4 | On demand | 30 min |
| 5 | Retirement | 3 | On demand | 20 min |
| 6 | Neutralization | 5 | Annual | 30 min |
| 7 | Claims Validation | 4 | Annual | 30 min |
| 8 | Verification | 4 | Annual | 30 min |

### Templates (10)

| # | Template | Format | Description |
|---|----------|--------|-------------|
| 1 | Footprint Report | PDF | Scope 1/2/3 breakdown with data quality |
| 2 | Carbon Mgmt Plan Report | PDF | Reduction trajectory with MACC visualization |
| 3 | Credit Portfolio Report | PDF | ICVCM quality assessment per credit |
| 4 | Retirement Certificate | PDF | Registry retirement with serial numbers |
| 5 | Neutralization Statement | PDF | Balance statement per ISO 14068-1 |
| 6 | Claims Disclosure | PDF | Public claims per PAS 2060 Section 9 |
| 7 | Verification Package | PDF | ISAE 3410 evidence index with SHA-256 |
| 8 | Annual Progress Report | PDF | Year-over-year trend analysis |
| 9 | Permanence Risk Report | PDF | Credit-level and portfolio-level risk |
| 10 | Public Disclosure | PDF | Public-facing carbon neutrality statement |

### Integrations (12)

| # | Integration | Description |
|---|-------------|-------------|
| 1 | Pack Orchestrator | 10-phase DAG pipeline with retry and provenance |
| 2 | MRV Bridge | Routes to all 30 MRV agents (Scope 1/2/3) |
| 3 | GHG App Bridge | Connects to GL-GHG-APP v1.0 inventory |
| 4 | DECARB Bridge | Routes to DECARB agents for reduction planning |
| 5 | Data Bridge | Connects to all 20 DATA agents for intake/quality |
| 6 | Registry Bridge | API integration with 6 carbon credit registries |
| 7 | Credit Marketplace | Price discovery and procurement automation |
| 8 | Verification Body | ISAE 3410 engagement and opinion management |
| 9 | PACK-021 Bridge | Optional baseline and gap analysis (graceful degradation) |
| 10 | PACK-023 Bridge | Optional SBTi target alignment (graceful degradation) |
| 11 | Health Check | 20-category system verification |
| 12 | Setup Wizard | 6-step guided configuration |

### Presets (8)

| Preset | Neutrality Type | Scope | Standard | Use Case |
|--------|-----------------|-------|----------|----------|
| `corporate_neutrality` | CORPORATE | 1+2+3 | ISO 14068-1 + PAS 2060 | Large enterprises |
| `sme_neutrality` | SME | 1+2 | PAS 2060 | Small/medium businesses |
| `event_neutrality` | EVENT | Event-specific | PAS 2060 | Conferences, exhibitions |
| `product_neutrality` | PRODUCT | LCA boundary | ISO 14067 + PAS 2060 | Consumer goods |
| `building_neutrality` | BUILDING | Building ops | CRREM + PAS 2060 | Real estate |
| `service_neutrality` | SERVICE | Office + cloud | PAS 2060 | Consulting, tech |
| `project_neutrality` | PROJECT | A1-D modules | ISO 14064-2 + PAS 2080 | Construction |
| `portfolio_neutrality` | PORTFOLIO | Multi-entity | PCAF + ISO 14068-1 | Investment groups |

## Usage Examples

### Corporate Carbon Neutrality

```python
from packs.net_zero.PACK_024_carbon_neutral.config import load_preset
from packs.net_zero.PACK_024_carbon_neutral.engines import (
    FootprintQuantificationEngine,
    CarbonMgmtPlanEngine,
    NeutralizationBalanceEngine,
)

# Load corporate preset
config = load_preset("corporate_neutrality")

# Step 1: Quantify footprint
footprint_engine = FootprintQuantificationEngine(config)
footprint = footprint_engine.calculate(
    organization_id="org-001",
    reporting_year=2025,
    scopes=["scope_1", "scope_2", "scope_3"],
)
print(f"Total footprint: {footprint.total_tco2e:,.0f} tCO2e")

# Step 2: Generate management plan
plan_engine = CarbonMgmtPlanEngine(config)
plan = plan_engine.generate(
    footprint=footprint,
    reduction_target_pct=4.2,  # Annual reduction per SBTi
    planning_horizon_years=5,
)
print(f"Residual emissions: {plan.residual_tco2e:,.0f} tCO2e")

# Step 3: Calculate neutralization balance
balance_engine = NeutralizationBalanceEngine(config)
balance = balance_engine.calculate(
    footprint=footprint,
    credits_retired=credits,
    buffer_pct=0.10,
)
print(f"Balance status: {balance.status}")  # SURPLUS or DEFICIT
```

### Event Carbon Neutrality

```python
from packs.net_zero.PACK_024_carbon_neutral.config import load_preset

config = load_preset("event_neutrality")
# Event-specific: venue energy, attendee travel, accommodation,
# catering, materials. Up to 10,000 attendees.
```

### Running the Full Workflow

```python
from packs.net_zero.PACK_024_carbon_neutral.integrations import (
    CarbonNeutralOrchestrator,
)

orchestrator = CarbonNeutralOrchestrator(config)
result = orchestrator.run_pipeline(
    organization_id="org-001",
    reporting_year=2025,
)
print(f"Pipeline status: {result.status}")
print(f"Phases completed: {result.phases_completed}/{result.phases_total}")
```

## Configuration Guide

### Configuration Hierarchy

Configuration is resolved in order of increasing priority:

1. **Base `pack.yaml`** -- Pack manifest with component definitions
2. **Preset YAML** -- Neutrality-type preset (e.g., `corporate_neutrality.yaml`)
3. **Environment variables** -- Override with `CARBON_NEUTRAL_*` prefix
4. **Runtime overrides** -- Explicit programmatic overrides

### Environment Variable Overrides

```bash
# Organization
export CARBON_NEUTRAL_ORG_NAME="Acme Corp"
export CARBON_NEUTRAL_ORG_SECTOR="manufacturing"

# Footprint
export CARBON_NEUTRAL_BASE_YEAR=2020
export CARBON_NEUTRAL_REPORTING_YEAR=2025

# Credit quality
export CARBON_NEUTRAL_MIN_QUALITY_SCORE=70

# Portfolio
export CARBON_NEUTRAL_MIN_REMOVAL_PCT=20
export CARBON_NEUTRAL_MAX_NATURE_PCT=40

# Buffer
export CARBON_NEUTRAL_BUFFER_PCT=10
```

### Custom Configuration

```python
from packs.net_zero.PACK_024_carbon_neutral.config import (
    CarbonNeutralConfig,
    NeutralityType,
)

config = CarbonNeutralConfig(
    neutrality_type=NeutralityType.CORPORATE,
    base_year=2020,
    reporting_year=2025,
    scopes_included=["scope_1", "scope_2", "scope_3"],
    min_credit_quality_score=70,
    min_removal_pct=0.20,
    max_nature_based_pct=0.40,
    buffer_pool_pct=0.10,
    assurance_level="reasonable",
)
```

## Standards Compliance

### Primary Standards

| Standard | Version | Coverage |
|----------|---------|----------|
| ISO 14068-1 | 2023 | Full (all clauses) |
| PAS 2060 | 2014 | Full (all sections) |

### Secondary Standards

| Standard | Coverage |
|----------|----------|
| ISO 14064-1:2018 | Organization-level GHG quantification |
| ISO 14064-2:2019 | Project-level GHG quantification |
| ISO 14067:2018 | Product carbon footprint |
| GHG Protocol Corporate Standard | Scope 1/2/3 accounting |
| GHG Protocol Scope 2 Guidance | Location-based and market-based |
| GHG Protocol Scope 3 Standard | All 15 categories |
| ICVCM Core Carbon Principles | 12-dimension credit quality |
| VCMI Claims Code | Platinum/Gold/Silver tiers |
| ISAE 3410 | Assurance engagements on GHG statements |
| IPCC AR6 | GWP-100 values |
| Oxford Principles | Net-zero aligned offsetting |
| CRREM | Carbon risk real estate |
| PCAF | Financed emissions |

## Testing

### Run All Tests

```bash
# From pack root directory
cd packs/net-zero/PACK-024-carbon-neutral

# Run all 693 tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Run specific test suites
python -m pytest tests/engines/ -v          # Engine tests
python -m pytest tests/workflows/ -v        # Workflow tests
python -m pytest tests/templates/ -v        # Template tests
python -m pytest tests/integrations/ -v     # Integration tests
python -m pytest tests/test_config.py -v    # Config tests
python -m pytest tests/test_presets.py -v   # Preset tests
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Engine tests | 10 modules | Unit tests for all 10 calculation engines |
| Workflow tests | 1 module | Orchestration and phase sequencing |
| Template tests | 1 module | Report generation and formatting |
| Integration tests | 1 module | Bridge connectivity and data routing |
| Config tests | 1 module | Configuration loading and validation |
| Preset tests | 1 module | All 8 preset configurations |

## Pack Structure

```
PACK-024-carbon-neutral/
  __init__.py                          # Pack-level exports
  pack.yaml                           # Pack manifest (1681 lines)
  README.md                           # This file
  ARCHITECTURE.md                     # Technical architecture
  CHANGELOG.md                        # Version history
  VALIDATION_REPORT.md                # Final validation results
  config/
    __init__.py                        # Config module exports
    runtime_config.py                  # Pydantic v2 configuration models
    demo/
      __init__.py
      demo_config.yaml                 # Demo configuration
    presets/
      corporate_neutrality.yaml        # Corporate Scope 1+2+3
      sme_neutrality.yaml              # SME Scope 1+2
      event_neutrality.yaml            # Event neutrality
      product_neutrality.yaml          # Product LCA
      building_neutrality.yaml         # Building operations
      service_neutrality.yaml          # Office/cloud services
      project_neutrality.yaml          # Construction projects
      portfolio_neutrality.yaml        # Multi-entity portfolio
  engines/
    __init__.py                        # Engine exports
    footprint_quantification_engine.py # ISO 14064-1 quantification
    carbon_mgmt_plan_engine.py         # Reduction-first planning
    credit_quality_engine.py           # ICVCM CCP 12-dimension scoring
    portfolio_optimization_engine.py   # Markowitz allocation
    registry_retirement_engine.py      # 6-registry retirement
    neutralization_balance_engine.py   # ISO 14068-1 balance
    claims_substantiation_engine.py    # VCMI claims validation
    verification_package_engine.py     # ISAE 3410 evidence
    annual_cycle_engine.py             # Multi-year cycle
    permanence_risk_engine.py          # Buffer pool risk
  workflows/
    __init__.py                        # Workflow exports
    full_annual_cycle_workflow.py       # 10-phase end-to-end
    footprint_assessment_workflow.py    # 4-phase quantification
    carbon_mgmt_plan_workflow.py       # 5-phase management plan
    credit_procurement_workflow.py     # 4-phase procurement
    retirement_workflow.py             # 3-phase retirement
    neutralization_workflow.py         # 5-phase balance
    claims_validation_workflow.py      # 4-phase claims
    verification_workflow.py           # 4-phase verification
  templates/
    __init__.py                        # Template exports
    footprint_report.py                # Scope 1/2/3 breakdown
    carbon_mgmt_plan_report.py         # MACC visualization
    credit_portfolio_report.py         # Quality assessment
    registry_retirement_report.py      # Retirement certificates
    neutralization_statement_report.py # Balance statement
    claims_substantiation_report.py    # Public claims
    verification_package_report.py     # ISAE 3410 evidence index
    annual_report.py                   # Year-over-year progress
    permanence_assessment_report.py    # Permanence risk
    public_disclosure_report.py        # Public disclosure
  integrations/
    __init__.py                        # Integration exports
    pack_orchestrator.py               # 10-phase DAG pipeline
    mrv_bridge.py                      # 30 MRV agents
    ghg_app_bridge.py                  # GL-GHG-APP v1.0
    decarb_bridge.py                   # DECARB agents
    data_bridge.py                     # 20 DATA agents
    registry_bridge.py                 # 6-registry API
    credit_marketplace_bridge.py       # Marketplace integration
    verification_body_bridge.py        # ISAE 3410 bodies
    pack021_bridge.py                  # PACK-021 (optional)
    pack023_bridge.py                  # PACK-023 (optional)
    health_check.py                    # 20-category verification
    setup_wizard.py                    # 6-step wizard
  tests/
    __init__.py
    conftest.py                        # Shared test infrastructure
    test_config.py                     # Configuration tests
    test_presets.py                    # Preset validation tests
    engines/
      __init__.py
      test_footprint_quantification.py
      test_carbon_mgmt_plan.py
      test_credit_quality.py
      test_portfolio_optimization.py
      test_registry_retirement.py
      test_neutralization_balance.py
      test_claims_substantiation.py
      test_verification_package.py
      test_annual_cycle.py
      test_permanence_risk.py
    workflows/
      __init__.py
      test_workflows.py
    templates/
      __init__.py
      test_templates.py
    integrations/
      __init__.py
      test_integrations.py
```

## Security

- **Authentication**: JWT (RS256)
- **Authorization**: RBAC with facility-level and credit-level access control
- **Encryption at rest**: AES-256-GCM
- **Encryption in transit**: TLS 1.3
- **Audit logging**: All 10 engines produce audit events
- **PII redaction**: Automatic PII detection and redaction
- **Data classification**: CONFIDENTIAL, RESTRICTED, INTERNAL, PUBLIC

### Required Roles

| Role | Access Level |
|------|-------------|
| `carbon_neutral_admin` | Full read/write on all pack resources |
| `sustainability_manager` | Manage footprints, plans, and claims |
| `climate_analyst` | Read/write on calculations and reports |
| `credit_portfolio_manager` | Manage credits, registries, and retirements |
| `external_auditor` | Read-only access to verification packages |
| `viewer` | Read-only access to reports and dashboards |

## License

Proprietary - GreenLang Platform Team

## Support

- **Documentation**: https://docs.greenlang.io/packs/carbon-neutral
- **Changelog**: https://docs.greenlang.io/packs/carbon-neutral/changelog
- **Support tier**: Enterprise
