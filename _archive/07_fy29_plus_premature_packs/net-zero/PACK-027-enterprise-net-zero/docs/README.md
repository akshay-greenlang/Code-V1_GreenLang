# PACK-027: Enterprise Net Zero Pack

**Pack ID:** PACK-027-enterprise-net-zero
**Category:** Net Zero Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Production Ready
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quick Start Guide](#quick-start-guide)
3. [Architecture Overview](#architecture-overview)
4. [Components](#components)
5. [Installation Guide](#installation-guide)
6. [Usage Examples](#usage-examples)
7. [Configuration Guide](#configuration-guide)
8. [ERP Integration](#erp-integration)
9. [Security Model](#security-model)
10. [Troubleshooting](#troubleshooting)
11. [Frequently Asked Questions](#frequently-asked-questions)
12. [Related Documentation](#related-documentation)

---

## Executive Summary

### What is PACK-027?

PACK-027 is the **Enterprise Net Zero Pack** -- the flagship enterprise-grade net-zero solution within the GreenLang platform. It provides financial-grade GHG accounting, full SBTi Corporate Standard compliance, multi-entity consolidation, advanced analytics, and external assurance readiness across the complete net-zero lifecycle.

### Who is it for?

Large enterprises with more than 250 employees and more than $50 million in annual revenue that operate complex corporate structures with multiple subsidiaries, joint ventures, and associates across multiple countries.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **Financial-Grade GHG Accounting** | +/-3% accuracy target across all 15 Scope 3 categories with activity-based data |
| **Full SBTi Corporate Standard** | 42-criteria validation (C1-C28 near-term + NZ-C1 to NZ-C14 net-zero) with ACA/SDA/FLAG pathways |
| **Multi-Entity Consolidation** | 100+ subsidiaries with intercompany elimination across financial control, operational control, and equity share approaches |
| **Monte Carlo Scenario Modeling** | 10,000+ simulation runs across 1.5C, 2C, and BAU scenarios with sensitivity analysis |
| **Internal Carbon Pricing** | $50-$200/tCO2e shadow pricing with carbon-adjusted P&L, NPV, and CBAM exposure |
| **Scope 4 Avoided Emissions** | WBCSD-aligned avoided emissions quantification with conservative principles |
| **Supply Chain Mapping** | Multi-tier mapping (Tier 1-5) for 100,000+ suppliers with engagement tracking |
| **External Assurance Readiness** | ISO 14064-3 workpapers, Big 4 audit format, management assertion templates |
| **Multi-Framework Compliance** | GHG Protocol, SBTi, CDP, TCFD/ISSB S2, SEC, CSRD/ESRS E1, ISO 14064, CA SB 253 |
| **Enterprise ERP Integration** | SAP S/4HANA, Oracle ERP Cloud, Workday HCM bidirectional connectors |

### How PACK-027 Differs from Other Net Zero Packs

| Dimension | PACK-026 (SME) | PACK-022 (Acceleration) | **PACK-027 (Enterprise)** |
|-----------|----------------|------------------------|--------------------------|
| Target organization | <250 employees | 250-5,000 employees | >250 employees, >$50M revenue |
| Scope 3 coverage | 3 categories | All 15 (spend-based + activity) | All 15 (activity-based, supplier-specific) |
| Data quality target | +/-20-40% | +/-10-15% | **+/-3% (financial-grade)** |
| SBTi pathway | SME simplified | ACA + SDA | **Full Corporate Standard + Net-Zero Standard** |
| Entity consolidation | Single entity | Up to 50 subsidiaries | **100+ with intercompany elimination** |
| ERP integration | Xero/QuickBooks | Basic ERP | **SAP S/4HANA, Oracle ERP Cloud, Workday HCM** |
| Scenario modeling | None | 3-scenario Monte Carlo | **Full Monte Carlo (10,000 runs) with carbon pricing** |
| External assurance | Not applicable | SHA-256 provenance | **ISO 14064-3 ready, Big 4 workpapers** |
| Implementation timeline | 2 hours | 2-4 weeks | **6-12 weeks (phased rollout)** |
| Max suppliers | 500 | 50,000 | **100,000+** |
| Max facilities | 50 | 2,000 | **5,000+** |

### Target Users

| Persona | Role | Key PACK-027 Value |
|---------|------|-------------------|
| Chief Sustainability Officer | C-suite climate strategy | Board-ready dashboards, regulatory compliance, scenario analysis |
| Head of Sustainability | Program execution lead | Automated data collection, multi-entity consolidation, supplier portal |
| Chief Financial Officer | Climate-financial integration | Carbon-adjusted P&L, CBAM exposure, internal carbon pricing |
| Board Member | Climate governance | Executive dashboard, quarterly climate reports, assurance readiness |
| Sustainability Analyst | Technical data specialist | 30 MRV agents, DQ scoring, SHA-256 provenance, audit trails |
| Supply Chain Director | Scope 3 reduction | Supplier tiering, engagement tracking, CDP Supply Chain integration |
| External Auditor | Assurance provider | Pre-formatted workpapers, calculation traces, evidence packages |

### Success Metrics

| Metric | Target |
|--------|--------|
| Time to complete enterprise GHG baseline | <6 weeks (vs. 6-12 months manual) |
| GHG calculation accuracy | +/-3% or better |
| Scope 3 category coverage | 15/15 categories |
| Multi-entity consolidation accuracy | 100% match with financial consolidation |
| SBTi 42-criteria first-pass approval | 95%+ |
| Assurance engagement efficiency | <80 hours auditor time (vs. 200-400 hours) |
| Regulatory framework coverage | 8+ simultaneous frameworks |
| Scenario modeling throughput | 10,000 Monte Carlo runs in <30 minutes |
| Board report generation | <15 minutes from data refresh |

---

## Quick Start Guide

### Prerequisites

- GreenLang platform v1.0+ deployed (see `DEPLOYMENT_GUIDE.md`)
- PostgreSQL 16 with TimescaleDB extension
- Redis 7+ cache cluster
- Python 3.11+ runtime
- Platform migrations V001-V128 applied
- PACK-027 migrations V083-PACK027-001 through V083-PACK027-015 applied

### Step 1: Install the Pack

```bash
# From the GreenLang root directory
cd packs/net-zero/PACK-027-enterprise-net-zero

# Verify pack structure
python -c "from config.pack_config import PackConfig; print('Pack config loaded successfully')"
```

### Step 2: Run the Health Check

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run()
print(f"Health Score: {result.overall_score}/100")
print(f"Status: {result.status}")

# Expected output:
# Health Score: 100/100
# Status: HEALTHY
```

### Step 3: Configure Your Organization

```python
from config.pack_config import PackConfig, EnterpriseSector, ConsolidationApproach, SBTiPathway

config = PackConfig(
    sector=EnterpriseSector.MANUFACTURING,
    consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
    sbti_pathway=SBTiPathway.ACA_15C,
    reporting_year=2025,
    base_year=2023,
    currency="USD",
    entity_count=120,
    total_employees=85000,
    total_revenue=25_000_000_000,
    countries_of_operation=["US", "DE", "CN", "JP", "GB", "FR", "IN", "BR"],
    # ... engine-specific configurations
)
```

### Step 4: Use the Setup Wizard

```python
from integrations.setup_wizard import SetupWizard

wizard = SetupWizard()
result = wizard.run(
    step=1,  # Start with corporate profile
    profile={
        "legal_name": "GlobalMfg Corp",
        "sector": "manufacturing",
        "annual_revenue_usd": 25_000_000_000,
        "total_employees": 85_000,
        "headquarters_country": "US",
    }
)
```

The 10-step setup wizard guides you through:

1. Corporate profile (legal entity, sector, revenue, employees)
2. Organizational boundary definition (consolidation approach, entity hierarchy)
3. Reporting year and base year selection
4. Entity registration (subsidiaries, JVs, associates)
5. ERP system connection (SAP/Oracle/Workday)
6. Data source mapping per entity
7. Scope 3 materiality screening
8. SBTi target configuration
9. Carbon pricing setup
10. Go-live health check and demo validation

### Step 5: Calculate Your First Baseline

```python
from engines.enterprise_baseline_engine import EnterpriseBaselineEngine
from workflows.comprehensive_baseline_workflow import ComprehensiveBaselineWorkflow

# Use the workflow for full orchestration
workflow = ComprehensiveBaselineWorkflow(config=config)
result = workflow.execute()

# Access results
print(f"Total Scope 1: {result.scope1_total_tco2e:,.0f} tCO2e")
print(f"Total Scope 2 (location): {result.scope2_location_tco2e:,.0f} tCO2e")
print(f"Total Scope 2 (market): {result.scope2_market_tco2e:,.0f} tCO2e")
print(f"Total Scope 3: {result.scope3_total_tco2e:,.0f} tCO2e")
print(f"Grand Total: {result.grand_total_tco2e:,.0f} tCO2e")
print(f"Data Quality Score: {result.weighted_dq_score:.1f}")
print(f"Provenance Hash: {result.provenance_hash}")
```

### Step 6: Generate Reports

```python
from templates.ghg_inventory_report import GHGInventoryReport
from templates.executive_dashboard import ExecutiveDashboard

# Full GHG inventory report
report = GHGInventoryReport()
output = report.render(result, format="html")

# Board-level executive dashboard
dashboard = ExecutiveDashboard()
output = dashboard.render(result, format="html")
```

---

## Architecture Overview

### Three-Tier Architecture

```
+=============================================================================+
|                          PRESENTATION TIER                                   |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | Executive         |  | Analyst           |  | API Endpoints     |        |
|  | Dashboard (HTML)  |  | Workbench (HTML)  |  | (REST JSON)       |        |
|  +-------------------+  +-------------------+  +-------------------+        |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | Board Reports     |  | Supplier Portal   |  | Auditor Portal    |        |
|  | (PDF/HTML)        |  | (Web)             |  | (Read-Only)       |        |
|  +-------------------+  +-------------------+  +-------------------+        |
+=============================================================================+
|                          APPLICATION TIER                                     |
|  +-----------------------------------------------------------------------+  |
|  |                    Pack Orchestrator (DAG Engine)                       |  |
|  |  +---------+ +---------+ +---------+ +---------+ +---------+         |  |
|  |  | Baseline| | SBTi    | | Annual  | | Scenario| | Supply  |         |  |
|  |  | Wkflow  | | Submit  | | Inv     | | Anlysis | | Chain   |         |  |
|  |  +---------+ +---------+ +---------+ +---------+ +---------+         |  |
|  |  +---------+ +---------+ +---------+                                 |  |
|  |  | Carbon  | | Multi-  | | Ext     |                                 |  |
|  |  | Pricing | | Entity  | | Assur   |                                 |  |
|  |  +---------+ +---------+ +---------+                                 |  |
|  +-----------------------------------------------------------------------+  |
|  +-----------------------------------------------------------------------+  |
|  |                         8 Engines                                      |  |
|  |  +-----------+ +-----------+ +-----------+ +-----------+              |  |
|  |  | Enterprise| | SBTi      | | Scenario  | | Carbon    |              |  |
|  |  | Baseline  | | Target    | | Modeling  | | Pricing   |              |  |
|  |  +-----------+ +-----------+ +-----------+ +-----------+              |  |
|  |  +-----------+ +-----------+ +-----------+ +-----------+              |  |
|  |  | Scope 4   | | Supply    | | Multi-    | | Financial |              |  |
|  |  | Avoided   | | Chain Map | | Entity    | | Integrat  |              |  |
|  |  +-----------+ +-----------+ +-----------+ +-----------+              |  |
|  +-----------------------------------------------------------------------+  |
+=============================================================================+
|                           DATA TIER                                          |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | PostgreSQL 16     |  | Redis 7           |  | Emission Factor   |        |
|  | TimescaleDB       |  | Cache + Sessions  |  | Database          |        |
|  | 15 pack tables    |  |                   |  | (DEFRA/EPA/IEA)   |        |
|  +-------------------+  +-------------------+  +-------------------+        |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | ERP Systems       |  | CDP/SBTi APIs     |  | Carbon Markets    |        |
|  | SAP/Oracle/Workday|  | Questionnaire/Tgt |  | Verra/GS/ACR      |        |
|  +-------------------+  +-------------------+  +-------------------+        |
+=============================================================================+
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Horizontal scaling** via Kubernetes | Enterprise workloads require elastic compute for Monte Carlo and batch processing |
| **Deterministic calculations** (zero-hallucination) | No LLM in any calculation path; all emission factors are constants |
| **Financial-grade data quality** (+/-3%) | Enterprise disclosures subject to external assurance; regulatory fines for errors |
| **SHA-256 provenance hashing** | Every calculation output is cryptographically hashed for audit integrity |
| **Multi-entity consolidation** | GHG Protocol requires consistent consolidation across corporate structures |
| **Bidirectional ERP integration** | Pull activity data from ERP; push carbon allocation back into financial reporting |
| **8-role RBAC** | Enterprise governance requires segregation of duties with entity-level access |
| **Async batch processing** | Enterprise baselines with 100+ entities require background job orchestration |

### Data Flow

```
ERP Systems (SAP/Oracle/Workday)
    |
    v
DATA-003 ERP Connector --> Data Extraction (daily/weekly/monthly)
    |
    v
Data Transformation (unit normalization, currency conversion)
    |
    v
Data Quality Validation (DATA-010 Profiler, DATA-013 Outlier)
    |
    v
GreenLang Carbon Data Store (PostgreSQL + TimescaleDB)
    |
    +---> Enterprise Baseline Engine (all 30 MRV agents)
    |         |
    |         v
    |     Multi-Entity Consolidation Engine
    |         |
    |         v
    |     Report Templates (GHG, CDP, TCFD, CSRD, SEC, SBTi)
    |
    +---> Scenario Modeling Engine (Monte Carlo)
    |         |
    |         v
    |     Strategy Reports (fan charts, sensitivity, investment)
    |
    +---> Carbon Pricing Engine
    |         |
    |         v
    |     Carbon-Adjusted Financial Reports (P&L, NPV, CBAM)
    |
    +---> Supply Chain Mapping Engine
    |         |
    |         v
    |     Supplier Scorecards & Heatmaps
    |
    +---> Financial Integration Engine
              |
              v
          Reverse Integration --> ERP GL Carbon Allocation
```

---

## Components

### Engines (8)

| # | Engine | File | Purpose |
|---|--------|------|---------|
| 1 | Enterprise Baseline Engine | `enterprise_baseline_engine.py` | Financial-grade Scope 1+2+3 calculation across all 30 MRV agents with 5-level data quality scoring |
| 2 | SBTi Target Engine | `sbti_target_engine.py` | Full SBTi Corporate Standard with 42-criteria validation (C1-C28 + NZ-C1 to NZ-C14) |
| 3 | Scenario Modeling Engine | `scenario_modeling_engine.py` | Monte Carlo pathway analysis (10,000+ runs) across 1.5C, 2C, and BAU scenarios |
| 4 | Carbon Pricing Engine | `carbon_pricing_engine.py` | Internal carbon price management ($50-$200/tCO2e) with carbon-adjusted financials |
| 5 | Scope 4 Avoided Emissions Engine | `scope4_avoided_emissions_engine.py` | WBCSD-aligned avoided emissions quantification |
| 6 | Supply Chain Mapping Engine | `supply_chain_mapping_engine.py` | Multi-tier supplier mapping (Tier 1-5) for 100,000+ suppliers |
| 7 | Multi-Entity Consolidation Engine | `multi_entity_consolidation_engine.py` | 100+ entity consolidation with intercompany elimination |
| 8 | Financial Integration Engine | `financial_integration_engine.py` | Carbon-adjusted P&L, EBITDA, balance sheet, and ESRS E1-8/E1-9 |

### Workflows (8)

| # | Workflow | File | Phases | Purpose |
|---|----------|------|--------|---------|
| 1 | Comprehensive Baseline | `comprehensive_baseline_workflow.py` | 6 | Full GHG inventory across all entities and 15 Scope 3 categories |
| 2 | SBTi Submission | `sbti_submission_workflow.py` | 5 | Complete SBTi target submission preparation |
| 3 | Annual Inventory | `annual_inventory_workflow.py` | 5 | Annual recalculation with base year adjustment review |
| 4 | Scenario Analysis | `scenario_analysis_workflow.py` | 5 | Full scenario analysis for board-level strategic planning |
| 5 | Supply Chain Engagement | `supply_chain_engagement_workflow.py` | 5 | End-to-end supplier engagement program |
| 6 | Internal Carbon Pricing | `internal_carbon_pricing_workflow.py` | 4 | Implement and report on internal carbon pricing |
| 7 | Multi-Entity Rollup | `multi_entity_rollup_workflow.py` | 5 | Consolidate 100+ entities with intercompany elimination |
| 8 | External Assurance | `external_assurance_workflow.py` | 5 | Prepare for ISO 14064-3 / ISAE 3410 external assurance |

### Templates (10)

| # | Template | File | Formats | Purpose |
|---|----------|------|---------|---------|
| 1 | GHG Inventory Report | `ghg_inventory_report.py` | MD, HTML, JSON, XLSX | Full GHG Protocol Corporate Standard report |
| 2 | SBTi Target Submission | `sbti_target_submission.py` | MD, HTML, JSON, PDF | SBTi submission package with 42-criteria matrix |
| 3 | CDP Climate Response | `cdp_climate_response.py` | MD, HTML, JSON | Full CDP questionnaire (C0-C15) auto-populated |
| 4 | TCFD Report | `tcfd_report.py` | MD, HTML, JSON, PDF | TCFD/ISSB S2 disclosure (4 pillars) |
| 5 | Executive Dashboard | `executive_dashboard.py` | MD, HTML, JSON | Board-level climate dashboard with 15-20 KPIs |
| 6 | Supply Chain Heatmap | `supply_chain_heatmap.py` | MD, HTML, JSON | Supplier emissions heatmap by geography and engagement |
| 7 | Scenario Comparison | `scenario_comparison.py` | MD, HTML, JSON | 1.5C vs. 2C vs. BAU comparison with fan charts |
| 8 | Assurance Statement | `assurance_statement.py` | MD, HTML, JSON, PDF | ISO 14064-3 assurance statement template |
| 9 | Board Climate Report | `board_climate_report.py` | MD, HTML, JSON, PDF | Quarterly board climate report (5-10 pages) |
| 10 | Regulatory Filings | `regulatory_filings.py` | MD, HTML, JSON, PDF, XLSX | Multi-framework filings (SEC, CSRD, SB 253, ISO, CDP) |

### Integrations (13)

| # | Integration | File | Purpose |
|---|-------------|------|---------|
| 1 | SAP Connector | `sap_connector.py` | SAP S/4HANA (MM, FI, CO, SD, PM, HCM, TM) |
| 2 | Oracle Connector | `oracle_connector.py` | Oracle ERP Cloud (Procurement, Financial, SCM, HCM) |
| 3 | Workday Connector | `workday_connector.py` | Workday HCM (headcount, travel, expenses) |
| 4 | CDP Bridge | `cdp_bridge.py` | CDP Climate Change questionnaire (C0-C15) |
| 5 | SBTi Bridge | `sbti_bridge.py` | SBTi target submission and lifecycle |
| 6 | Assurance Provider Bridge | `assurance_provider_bridge.py` | Big 4 workpaper generation |
| 7 | Multi-Entity Orchestrator | `multi_entity_orchestrator.py` | 100+ entity hierarchy management |
| 8 | Carbon Marketplace Bridge | `carbon_marketplace_bridge.py` | Verra, Gold Standard, ACR, Puro.earth |
| 9 | Supply Chain Portal | `supply_chain_portal.py` | Supplier questionnaire distribution and tracking |
| 10 | Financial System Bridge | `financial_system_bridge.py` | GL carbon cost allocation |
| 11 | Data Quality Guardian | `data_quality_guardian.py` | Continuous DQ monitoring against +/-3% target |
| 12 | Setup Wizard | `setup_wizard.py` | 10-step enterprise onboarding |
| 13 | Health Check | `health_check.py` | 25-category system health verification |

### Presets (8)

| # | Preset | File | Target Sector |
|---|--------|------|---------------|
| 1 | Manufacturing Enterprise | `manufacturing_enterprise.yaml` | Siemens, BASF, Honeywell-type industrials |
| 2 | Energy & Utilities | `energy_utilities.yaml` | Shell, Enel, NextEra-type energy companies |
| 3 | Financial Services | `financial_services.yaml` | JPMorgan, HSBC, BlackRock-type financials |
| 4 | Technology | `technology.yaml` | Microsoft, Google, SAP-type technology companies |
| 5 | Consumer Goods | `consumer_goods.yaml` | Unilever, P&G, Nestle-type FMCG |
| 6 | Transport & Logistics | `transport_logistics.yaml` | Maersk, DHL, Delta-type transport companies |
| 7 | Real Estate | `real_estate.yaml` | Prologis, Vonovia-type REITs |
| 8 | Healthcare & Pharma | `healthcare_pharma.yaml` | Novartis, Roche, Pfizer-type pharma |

---

## Installation Guide

### System Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| CPU | 4 vCPU | 8+ vCPU | Monte Carlo may burst to 16 vCPU |
| RAM | 8 GB | 16 GB | 4 GB per pack instance |
| Storage | 2 GB | 10 GB | Emission factor databases + reference data |
| Database | PostgreSQL 16 + TimescaleDB | Same | 15 pack-specific tables |
| Cache | Redis 7+ | Same | Emission factor caching, intermediate results |
| Network | Outbound HTTPS | Same | ERP APIs, CDP API, SBTi portal |
| Python | 3.11+ | 3.12 | Pydantic v2 required |

### Installation Steps

#### 1. Verify Platform Prerequisites

```bash
# Verify Python version
python --version
# Expected: Python 3.11.x or higher

# Verify PostgreSQL connection
psql -h $DB_HOST -U $DB_USER -d greenlang -c "SELECT version();"

# Verify Redis connection
redis-cli -h $REDIS_HOST ping
# Expected: PONG

# Verify platform migrations
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"
# Expected: V128 or higher
```

#### 2. Apply Pack Migrations

```bash
# Apply PACK-027 specific migrations
for i in $(seq -w 1 15); do
  psql -h $DB_HOST -U $DB_USER -d greenlang -f \
    deployment/migrations/V083-PACK027-${i}.sql
done
```

#### 3. Configure Environment Variables

```bash
# Required environment variables
export ENT_NET_ZERO_DB_HOST="localhost"
export ENT_NET_ZERO_DB_PORT="5432"
export ENT_NET_ZERO_DB_NAME="greenlang"
export ENT_NET_ZERO_REDIS_HOST="localhost"
export ENT_NET_ZERO_REDIS_PORT="6379"
export ENT_NET_ZERO_LOG_LEVEL="INFO"
export ENT_NET_ZERO_PROVENANCE="true"
export ENT_NET_ZERO_MONTE_CARLO_WORKERS="4"

# Optional ERP connector variables
export ENT_NET_ZERO_SAP_HOST="sap.example.com"
export ENT_NET_ZERO_SAP_CLIENT="100"
export ENT_NET_ZERO_ORACLE_HOST="oracle.example.com"
export ENT_NET_ZERO_WORKDAY_TENANT="tenant-id"
```

#### 4. Run Health Check

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run()

# Verify all 25 categories pass
for category in result.categories:
    print(f"  [{category.status}] {category.name}: {category.score}/100")

assert result.overall_score >= 90, f"Health check score too low: {result.overall_score}"
```

#### 5. Load a Sector Preset

```python
from config.pack_config import PackConfig

# Load the manufacturing preset
config = PackConfig.from_preset("manufacturing_enterprise")

# Or load financial services preset
config = PackConfig.from_preset("financial_services")
```

---

## Usage Examples

### Example 1: Calculate Enterprise GHG Baseline

```python
from engines.enterprise_baseline_engine import EnterpriseBaselineEngine

engine = EnterpriseBaselineEngine(config=config)

# Prepare input data for a single entity
entity_input = {
    "entity_id": "subco-alpha",
    "entity_name": "SubCo Alpha GmbH",
    "country": "DE",
    "energy": {
        "electricity_kwh": 15_000_000,
        "natural_gas_kwh": 8_000_000,
        "grid_region": "DE",
        "ppa_percentage": 30,  # 30% from PPA
    },
    "fleet": {
        "diesel_litres": 250_000,
        "petrol_litres": 50_000,
        "ev_kwh": 100_000,
    },
    "refrigerants": {
        "r410a_kg_charged": 500,
        "r410a_leak_rate": 0.05,
    },
    "procurement": {
        "total_spend_usd": 500_000_000,
        "supplier_specific_data": [
            {"supplier_id": "S001", "emissions_tco2e": 12_500, "dq_level": 1},
            {"supplier_id": "S002", "emissions_tco2e": 8_200, "dq_level": 2},
        ],
        "spend_categories": [
            {"category": "raw_materials", "spend_usd": 200_000_000},
            {"category": "components", "spend_usd": 150_000_000},
            {"category": "services", "spend_usd": 100_000_000},
            {"category": "logistics", "spend_usd": 50_000_000},
        ],
    },
    "travel": {
        "short_haul_km": 500_000,
        "long_haul_km": 2_000_000,
        "rail_km": 300_000,
        "hotel_nights": 5_000,
    },
    "employees": {
        "headcount": 5_000,
        "remote_percentage": 25,
        "average_commute_km": 20,
    },
    "waste": {
        "general_waste_tonnes": 500,
        "recycling_tonnes": 300,
        "hazardous_waste_tonnes": 50,
    },
}

result = engine.calculate(entity_input)
print(f"Entity: {result.entity_name}")
print(f"Scope 1: {result.scope1_tco2e:,.0f} tCO2e")
print(f"Scope 2 (location): {result.scope2_location_tco2e:,.0f} tCO2e")
print(f"Scope 2 (market): {result.scope2_market_tco2e:,.0f} tCO2e")
print(f"Scope 3: {result.scope3_total_tco2e:,.0f} tCO2e")
for cat in result.scope3_by_category:
    print(f"  Cat {cat.number}: {cat.name} = {cat.tco2e:,.0f} tCO2e (DQ: {cat.dq_level})")
```

### Example 2: Set SBTi Corporate Standard Targets

```python
from engines.sbti_target_engine import SBTiTargetEngine

engine = SBTiTargetEngine(config=config)

result = engine.set_targets(
    baseline=baseline_result,
    pathway="aca_15c",  # 4.2%/yr absolute contraction for 1.5C
    base_year=2023,
    target_year=2030,  # Near-term
    long_term_year=2050,
)

# Check 42-criteria validation
print(f"Criteria Passed: {result.criteria_passed}/42")
print(f"Submission Ready: {result.is_submission_ready}")

for criterion in result.criteria_results:
    status = "PASS" if criterion.passed else "FAIL"
    print(f"  [{status}] {criterion.id}: {criterion.description}")
    if not criterion.passed:
        print(f"         Remediation: {criterion.remediation}")
```

### Example 3: Run Monte Carlo Scenario Analysis

```python
from engines.scenario_modeling_engine import ScenarioModelingEngine

engine = ScenarioModelingEngine(config=config)

result = engine.run_scenarios(
    baseline=baseline_result,
    scenarios=["1.5C", "2C", "BAU"],
    monte_carlo_runs=10_000,
    confidence_intervals=[0.10, 0.25, 0.50, 0.75, 0.90],
)

# Access scenario results
for scenario in result.scenarios:
    print(f"\n{scenario.name} Scenario:")
    print(f"  P10 emissions 2030: {scenario.p10_2030:,.0f} tCO2e")
    print(f"  P50 emissions 2030: {scenario.p50_2030:,.0f} tCO2e")
    print(f"  P90 emissions 2030: {scenario.p90_2030:,.0f} tCO2e")
    print(f"  Probability of SBTi target: {scenario.target_probability:.1%}")
    print(f"  Required investment (P50): ${scenario.investment_p50:,.0f}")

# Top sensitivity drivers
print("\nTop 10 Sensitivity Drivers:")
for driver in result.sensitivity[:10]:
    print(f"  {driver.parameter}: Sobol index = {driver.sobol_total:.3f}")
```

### Example 4: Multi-Entity Consolidation

```python
from engines.multi_entity_consolidation_engine import MultiEntityConsolidationEngine

engine = MultiEntityConsolidationEngine(config=config)

result = engine.consolidate(
    entity_results=[alpha_result, beta_result, gamma_result],
    hierarchy=entity_hierarchy,
    approach="financial_control",
    intercompany_transactions=[
        {
            "from_entity": "jv-gamma",
            "to_entity": "subco-alpha",
            "type": "electricity_supply",
            "tco2e": 1_000,
            "description": "Internal electricity supply from JV Gamma to SubCo Alpha",
        }
    ],
)

print(f"Consolidated Scope 1: {result.scope1_consolidated:,.0f} tCO2e")
print(f"Consolidated Scope 2: {result.scope2_consolidated:,.0f} tCO2e")
print(f"Intercompany eliminations: {result.eliminations_total:,.0f} tCO2e")
print(f"Entity count: {result.entity_count}")
```

### Example 5: Internal Carbon Pricing

```python
from engines.carbon_pricing_engine import CarbonPricingEngine

engine = CarbonPricingEngine(config=config)

result = engine.calculate(
    baseline=baseline_result,
    carbon_price_usd=100,  # $100/tCO2e
    price_escalation_pct=5.0,  # 5% annual escalation
    allocation="scope1_scope2",
)

# Carbon-adjusted P&L
print(f"Total Carbon Charge: ${result.total_charge:,.0f}")
for bu in result.business_unit_charges:
    print(f"  {bu.name}: ${bu.charge:,.0f} ({bu.pct_of_revenue:.1%} of revenue)")

# Investment appraisal
for project in result.investment_appraisals:
    print(f"\n{project.name}:")
    print(f"  Standard NPV: ${project.standard_npv:,.0f}")
    print(f"  Carbon-Adjusted NPV: ${project.carbon_adjusted_npv:,.0f}")
    print(f"  Decision change: {project.decision_changed}")
```

### Example 6: Generate CDP Climate Response

```python
from templates.cdp_climate_response import CDPClimateResponse

template = CDPClimateResponse()
response = template.render(
    baseline=baseline_result,
    targets=target_result,
    scenarios=scenario_result,
    format="html",
)

# Auto-populated modules
# C0: Introduction
# C1: Governance
# C2: Risks and opportunities
# C3: Business strategy
# C4: Targets and performance
# C5: Emissions methodology
# C6: Emissions data (Scope 1, 2, 3)
# C7: Emissions breakdown
# C8: Energy
# C9-C12: Additional modules
# C15: Biodiversity
```

---

## Configuration Guide

### Configuration Hierarchy

Configuration is resolved in the following order (later overrides earlier):

1. **Base `pack.yaml` manifest** -- default values for all settings
2. **Sector preset YAML** -- sector-specific overrides (e.g., `manufacturing_enterprise.yaml`)
3. **Environment variables** -- `ENT_NET_ZERO_*` prefix overrides
4. **Runtime overrides** -- explicit parameters passed at execution time

### Sector Presets

Each preset configures the pack for a specific industry sector:

#### Manufacturing Enterprise

```yaml
# manufacturing_enterprise.yaml
sector: manufacturing
sbti_pathway: mixed  # ACA for general, SDA for cement/chemicals/steel
baseline:
  scope1_agents: [MRV-001, MRV-002, MRV-003, MRV-004, MRV-005]
  scope3_priority_categories: [1, 2, 3, 4, 11, 12]
carbon_pricing:
  cbam_enabled: true
  ets_enabled: true
scenarios:
  technology_focus: [electrification, hydrogen, ccs, energy_efficiency]
supply_chain:
  tier_depth: 4
```

#### Financial Services

```yaml
# financial_services.yaml
sector: financial_services
sbti_pathway: aca_15c
finz_enabled: true
baseline:
  scope1_agents: [MRV-001, MRV-002]
  scope3_priority_categories: [15, 1, 6, 7]
  pcaf_enabled: true
  pcaf_asset_classes: [listed_equity, corporate_bonds, business_loans, mortgages]
scenarios:
  portfolio_temperature_scoring: true
financial_integration:
  green_bond_screening: true
  taxonomy_alignment: true
```

#### Technology

```yaml
# technology.yaml
sector: technology
sbti_pathway: aca_15c
baseline:
  scope1_agents: [MRV-001, MRV-002]
  scope2_pue_tracking: true
  scope3_priority_categories: [1, 2, 3, 11, 12]
  data_center_metering: detailed
avoided_emissions:
  scope4_enabled: true
scenarios:
  re100_alignment: true
supply_chain:
  hardware_supplier_focus: true
  cloud_provider_emissions: true
carbon_pricing:
  data_center_allocation: true
```

### Environment Variables

All environment variables use the `ENT_NET_ZERO_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENT_NET_ZERO_DB_HOST` | PostgreSQL host | `localhost` |
| `ENT_NET_ZERO_DB_PORT` | PostgreSQL port | `5432` |
| `ENT_NET_ZERO_DB_NAME` | Database name | `greenlang` |
| `ENT_NET_ZERO_REDIS_HOST` | Redis host | `localhost` |
| `ENT_NET_ZERO_REDIS_PORT` | Redis port | `6379` |
| `ENT_NET_ZERO_LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `ENT_NET_ZERO_PROVENANCE` | Enable SHA-256 provenance hashing | `true` |
| `ENT_NET_ZERO_MONTE_CARLO_WORKERS` | Parallel workers for Monte Carlo | `4` |
| `ENT_NET_ZERO_SAP_HOST` | SAP S/4HANA hostname | None |
| `ENT_NET_ZERO_SAP_CLIENT` | SAP client number | None |
| `ENT_NET_ZERO_ORACLE_HOST` | Oracle ERP Cloud hostname | None |
| `ENT_NET_ZERO_WORKDAY_TENANT` | Workday tenant identifier | None |
| `ENT_NET_ZERO_CACHE_TTL` | Cache time-to-live (seconds) | `3600` |
| `ENT_NET_ZERO_MAX_ENTITIES` | Maximum entities to process | `500` |
| `ENT_NET_ZERO_MEMORY_CEILING_MB` | Per-engine memory limit | `4096` |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `scope4_enabled` | `false` | Enable Scope 4 avoided emissions engine (opt-in) |
| `carbon_pricing_enabled` | `true` | Enable internal carbon pricing engine |
| `flag_enabled` | `false` | Enable FLAG targets (auto-set if >20% FLAG emissions) |
| `finz_enabled` | `false` | Enable FINZ portfolio targets (auto-set for financial services) |
| `cbam_enabled` | `true` | Enable EU CBAM exposure tracking |
| `sec_climate_enabled` | `true` | Enable SEC Climate Rule compliance |
| `csrd_enabled` | `true` | Enable CSRD ESRS E1 compliance |
| `sap_enabled` | `false` | Enable SAP connector |
| `oracle_enabled` | `false` | Enable Oracle connector |
| `workday_enabled` | `false` | Enable Workday connector |

---

## ERP Integration

### SAP S/4HANA Integration

PACK-027 integrates with SAP S/4HANA across 8 modules:

| SAP Module | Data Extracted | GreenLang Usage |
|------------|---------------|-----------------|
| MM (Materials Management) | Purchase orders, goods receipts, material masters | Scope 3 Cat 1, Cat 2 |
| FI (Financial Accounting) | GL postings, vendor invoices, utility bills | Scope 2, Scope 3 (spend-based fallback) |
| CO (Controlling) | Cost center allocations, internal orders | BU-level carbon allocation |
| SD (Sales & Distribution) | Sales orders, deliveries, shipping | Scope 3 Cat 9 |
| PM (Plant Maintenance) | Equipment records, refrigerant logs | Scope 1 (refrigerants) |
| HCM (Human Capital) | Headcount, travel bookings, commute surveys | Scope 3 Cat 6, Cat 7 |
| TM (Transportation) | Shipment records, carrier data, routes | Scope 3 Cat 4 |
| S/4HANA Sustainability | Sustainability Control Tower, product footprints | Cross-module integration |

```python
from integrations.sap_connector import SAPConnector

sap = SAPConnector(
    host="sap.example.com",
    client="100",
    username="gl_service_user",
    password_vault_key="sap/gl_service_user",
)

# Extract procurement data for Scope 3 Cat 1
procurement_data = sap.extract_procurement(
    company_code="1000",
    fiscal_year=2025,
    material_groups=["RAW", "COMP", "PKG"],
)

# Extract energy invoices for Scope 2
energy_data = sap.extract_energy_invoices(
    company_code="1000",
    fiscal_year=2025,
    account_groups=["electricity", "gas", "steam"],
)
```

### Oracle ERP Cloud Integration

```python
from integrations.oracle_connector import OracleConnector

oracle = OracleConnector(
    host="oracle.example.com",
    tenant="org-tenant-id",
)

procurement_data = oracle.extract_procurement(
    business_unit="Global Manufacturing",
    fiscal_year=2025,
)
```

### Workday HCM Integration

```python
from integrations.workday_connector import WorkdayConnector

workday = WorkdayConnector(
    tenant="tenant-id",
)

employee_data = workday.extract_employee_data(
    as_of_date="2025-12-31",
    include_commute_survey=True,
)

travel_data = workday.extract_travel_expense(
    fiscal_year=2025,
)
```

---

## Security Model

### Role-Based Access Control (8 Roles)

| Role | Description | Key Permissions |
|------|-------------|----------------|
| `enterprise_admin` | System administrator | Full system configuration, user management, entity management |
| `cso` | Chief Sustainability Officer | All read/write, approve targets and disclosures, view all entities |
| `sustainability_manager` | Program lead | Read/write data, run engines, generate reports, manage suppliers |
| `entity_data_owner` | BU sustainability lead | Read/write data for assigned entities only |
| `analyst` | Sustainability analyst | Read all data, run engines, generate reports (no config changes) |
| `finance_viewer` | Finance team | Read carbon-adjusted financial data only |
| `auditor` | Internal/external auditor | Read all data and workpapers (no write access) |
| `board_viewer` | Board member | Read executive dashboard and board reports only |

### Data Protection

| Control | Implementation |
|---------|---------------|
| Encryption at rest | AES-256-GCM for all emission and financial data |
| Encryption in transit | TLS 1.3 for all internal and external communication |
| Provenance hashing | SHA-256 on all calculation outputs |
| Audit trail | Immutable append-only log with cryptographic chaining |
| ERP credentials | Encrypted via HashiCorp Vault |
| GDPR compliance | Employee data anonymized at individual level |
| SOX compliance | Carbon data integrated with financial reporting controls |
| Data residency | EU data stays in EU; US data stays in US |
| Supplier access | Suppliers see only their own data in portal |
| Retention | 10 years minimum (financial record alignment) |

### Segregation of Duties

- Data entry personnel cannot modify emission factors
- Calculation methodology changes require two-person approval
- Base year recalculations require senior management sign-off
- External disclosures require board/audit committee approval
- Carbon credit transactions require finance team authorization

---

## Troubleshooting

### Common Issues

#### Health Check Fails on Engine Connectivity

**Symptom:** Health check reports engine connectivity failures.

**Cause:** MRV agents not registered or database migrations not applied.

**Resolution:**
```bash
# Verify MRV agent registration
python -c "
from integrations.health_check import HealthCheck
hc = HealthCheck()
for agent in hc.check_mrv_agents():
    if not agent.connected:
        print(f'MISSING: {agent.agent_id}')
"

# Apply missing migrations
psql -h $DB_HOST -U $DB_USER -d greenlang -f deployment/migrations/V083-PACK027-001.sql
```

#### Monte Carlo Simulation Runs Slowly

**Symptom:** Scenario modeling takes >30 minutes for 10,000 runs.

**Cause:** Insufficient Monte Carlo workers or memory.

**Resolution:**
```bash
# Increase workers (one per CPU core)
export ENT_NET_ZERO_MONTE_CARLO_WORKERS=8

# Increase memory ceiling if needed
export ENT_NET_ZERO_MEMORY_CEILING_MB=8192
```

#### SAP Connector Authentication Failure

**Symptom:** `SAPAuthenticationError` when extracting data.

**Cause:** SAP credentials expired or Vault key incorrect.

**Resolution:**
```bash
# Verify Vault key
vault kv get secret/sap/gl_service_user

# Test SAP connectivity
python -c "
from integrations.sap_connector import SAPConnector
sap = SAPConnector(host='sap.example.com', client='100')
print(sap.test_connection())
"
```

#### Multi-Entity Consolidation Mismatch

**Symptom:** Consolidated total does not match sum of entity-level totals.

**Cause:** Intercompany transactions not registered or elimination logic error.

**Resolution:**
```python
# Review intercompany elimination entries
from engines.multi_entity_consolidation_engine import MultiEntityConsolidationEngine

engine = MultiEntityConsolidationEngine(config=config)
reconciliation = engine.reconcile(result)

print(f"Sum of entities: {reconciliation.entity_sum:,.0f}")
print(f"Eliminations: {reconciliation.eliminations:,.0f}")
print(f"Consolidated: {reconciliation.consolidated:,.0f}")
print(f"Difference: {reconciliation.difference:,.0f}")

for entry in reconciliation.elimination_entries:
    print(f"  {entry.from_entity} -> {entry.to_entity}: {entry.tco2e:,.0f} tCO2e")
```

#### SBTi Criteria Validation Failures

**Symptom:** Fewer than 42 criteria passing in SBTi target validation.

**Cause:** Coverage requirements not met, base year too old, or pathway misconfigured.

**Resolution:**
```python
# Identify failing criteria
for criterion in result.criteria_results:
    if not criterion.passed:
        print(f"FAIL: {criterion.id} - {criterion.description}")
        print(f"  Current: {criterion.current_value}")
        print(f"  Required: {criterion.required_value}")
        print(f"  Remediation: {criterion.remediation}")
```

#### Data Quality Score Below Target

**Symptom:** Weighted DQ score above 2.0 (lower is better) after Year 2.

**Cause:** Too many Scope 3 categories using spend-based estimation.

**Resolution:**
```python
# Review DQ scores by category
from integrations.data_quality_guardian import DataQualityGuardian

guardian = DataQualityGuardian(config=config)
assessment = guardian.assess(baseline_result)

for category in assessment.category_scores:
    if category.dq_level > 3.0:
        print(f"LOW DQ: {category.name} (DQ: {category.dq_level})")
        print(f"  Improvement actions: {category.improvement_actions}")
```

#### Out of Memory During Large Baseline Calculation

**Symptom:** `MemoryError` when calculating baseline for 100+ entities.

**Cause:** All entities processed in memory simultaneously.

**Resolution:**
```bash
# Enable batch processing mode
export ENT_NET_ZERO_BATCH_SIZE=20
export ENT_NET_ZERO_MEMORY_CEILING_MB=4096

# Entities will be processed in batches of 20
```

#### Report Generation Timeout

**Symptom:** Template rendering exceeds timeout for large reports.

**Cause:** Full GHG inventory report with 100+ entities generates very large documents.

**Resolution:**
```python
# Generate entity-level reports separately
for entity in entities:
    report = GHGInventoryReport()
    output = report.render(entity_result, format="html", entity_filter=[entity.id])

# Or generate consolidated summary only
output = report.render(consolidated_result, format="html", detail_level="summary")
```

---

## Frequently Asked Questions

### General

**Q: How long does the initial enterprise baseline take?**

A: The computational calculation takes 2-4 hours after all data has been ingested. However, the end-to-end process including ERP integration setup, data validation, and entity mapping typically takes 6-12 weeks for a phased enterprise rollout.

**Q: Can PACK-027 handle our company with 300+ subsidiaries?**

A: Yes. PACK-027 supports up to 500 entities. For organizations with more than 500 entities, contact GreenLang Enterprise Sales for a custom configuration.

**Q: Do we need to replace our existing GHG accounting process?**

A: PACK-027 is designed to integrate with existing processes. The pack can import historical data from spreadsheets, existing tools, or other GreenLang packs. The setup wizard guides you through the migration.

**Q: What accuracy can we expect for Scope 3 emissions?**

A: PACK-027 targets +/-3% accuracy for the overall GHG inventory. Scope 3 accuracy depends on data quality level: supplier-specific data achieves +/-3%, average-data methods achieve +/-10-20%, and spend-based methods achieve +/-20-40%. The data quality guardian tracks improvement over time.

### SBTi

**Q: What is the difference between ACA and SDA pathways?**

A: The Absolute Contraction Approach (ACA) requires a minimum 4.2% per year absolute reduction for 1.5C alignment. The Sectoral Decarbonization Approach (SDA) sets intensity targets based on sector-specific pathways for 12 eligible sectors (power, cement, steel, etc.). Many enterprises use a mixed approach: ACA for general operations and SDA for sector-specific divisions.

**Q: How long does SBTi target validation typically take?**

A: PACK-027's automated 42-criteria validation runs in under 10 minutes. Once targets are submitted to SBTi, the external validation queue typically takes 4-12 weeks.

**Q: Does PACK-027 support FLAG targets?**

A: Yes. If your organization's land use emissions exceed 20% of total emissions (common in food, agriculture, and forestry sectors), PACK-027 automatically enables FLAG pathway assessment and target setting per SBTi FLAG Guidance V1.1.

### Data & Integration

**Q: Which ERP systems are supported?**

A: SAP S/4HANA, Oracle ERP Cloud, and Workday HCM are supported with dedicated connectors. Additional ERP systems can be connected via the generic REST API integration.

**Q: How often is data extracted from ERP systems?**

A: Data extraction frequency is configurable: daily (default), weekly, or monthly. The schedule can differ by data type (e.g., daily for energy consumption, weekly for procurement).

**Q: What emission factor databases are included?**

A: DEFRA 2024, EPA 2024, IEA 2024, ecoinvent 3.10, IPCC AR6 GWP values, and SBTi SDA sector benchmarks are included. Supplier-specific factors from CDP Supply Chain and WBCSD PACT are also supported.

### Assurance

**Q: Is PACK-027 ready for Big 4 audit?**

A: Yes. The external assurance workflow generates 15 workpapers in Big 4 format, performs pre-assurance control testing, and produces management assertion letter templates. The system has been designed to reduce auditor engagement time from 200-400 hours to under 80 hours.

**Q: What level of assurance is supported?**

A: Both limited assurance (ISAE 3410 negative assurance) and reasonable assurance (ISAE 3410 positive assurance) are supported. The pack also supports ISO 14064-3:2019 verification.

### Financial

**Q: How does internal carbon pricing work?**

A: PACK-027 supports both shadow pricing (hypothetical cost applied to investment decisions) and internal carbon fees (actual charges allocated to business units). The price can be set from $50-$200/tCO2e with configurable annual escalation. Carbon-adjusted NPV is calculated for all investment proposals.

**Q: Does PACK-027 handle CBAM exposure?**

A: Yes. The carbon pricing engine calculates EU CBAM certificate costs based on embedded emissions in imported goods, origin country, and projected CBAM certificate prices.

### Upgrade

**Q: We currently use PACK-026 (SME). Can we upgrade to PACK-027?**

A: Yes. PACK-027 includes an upgrade path from PACK-026 that preserves your existing baseline data and target definitions. The upgrade involves recalculating your baseline with activity-based methodology (improving accuracy from +/-20-40% to +/-3-10%), expanding Scope 3 coverage from 3 to 15 categories, and upgrading from the SBTi SME pathway to the full Corporate Standard.

**Q: Can we start with a subset of features and expand later?**

A: Absolutely. Feature flags allow you to enable capabilities incrementally. Start with the baseline engine and SBTi targets, then add carbon pricing, supply chain mapping, scenario modeling, and Scope 4 avoided emissions as your program matures.

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Detailed system architecture, data flows, and design patterns |
| [VALIDATION_REPORT.md](./VALIDATION_REPORT.md) | Test results, performance benchmarks, compliance validation |
| [SBTI_SUBMISSION_GUIDE.md](./SBTI_SUBMISSION_GUIDE.md) | SBTi Corporate Standard submission walkthrough |
| [CDP_RESPONSE_GUIDE.md](./CDP_RESPONSE_GUIDE.md) | CDP Climate Change questionnaire alignment guide |
| [TCFD_IMPLEMENTATION_GUIDE.md](./TCFD_IMPLEMENTATION_GUIDE.md) | TCFD/ISSB S2 recommendations implementation guide |
| [MULTI_ENTITY_GUIDE.md](./MULTI_ENTITY_GUIDE.md) | Multi-entity consolidation and hierarchy management |
| [ASSURANCE_GUIDE.md](./ASSURANCE_GUIDE.md) | External assurance preparation with ISO 14064-3 |
| [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) | Kubernetes deployment, database setup, production checklist |
| [CHANGELOG.md](./CHANGELOG.md) | Version history, features, known issues |

---

## License

Proprietary -- GreenLang Platform. All rights reserved.

## Support

- **Enterprise Support:** enterprise-support@greenlang.io
- **Documentation:** docs.greenlang.io/packs/net-zero/enterprise
- **SLA:** 99.9% uptime, 4-hour response for P1 issues
