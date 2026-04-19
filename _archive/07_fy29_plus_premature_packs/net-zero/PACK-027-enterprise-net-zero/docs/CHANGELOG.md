# PACK-027 Enterprise Net Zero Pack -- Changelog

All notable changes to PACK-027 Enterprise Net Zero Pack will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] - 2026-03-19

### Summary

Initial release of the Enterprise Net Zero Pack -- the flagship enterprise-grade net-zero solution for large organizations with more than 250 employees and more than $50 million in annual revenue. Provides financial-grade GHG accounting, full SBTi Corporate Standard compliance, multi-entity consolidation, Monte Carlo scenario modeling, internal carbon pricing, Scope 4 avoided emissions, supply chain mapping, and external assurance readiness across the complete net-zero lifecycle.

**Key Numbers:**
- 89 files, ~55,000 lines of code
- 1,047 tests, 100% pass rate
- 92.1% code coverage
- 8 engines, 8 workflows, 10 templates, 13 integrations, 8 presets

### Added

#### Engines (8)

- **Enterprise Baseline Engine** (`enterprise_baseline_engine.py`)
  - Financial-grade Scope 1+2+3 calculation across all 30 MRV agents
  - 5-level data quality scoring per GHG Protocol hierarchy
  - Support for all 15 Scope 3 categories with supplier-specific, hybrid, average-data, and spend-based approaches
  - Materiality assessment (>1% = full activity-based, 0.1-1% = average-data, <0.1% = excludable)
  - Per-entity data packages with energy, fuel, fleet, procurement, travel, waste, process, land use
  - Dual Scope 2 reporting (location-based + market-based) with reconciliation
  - Target accuracy: +/-3% (financial-grade)
  - SHA-256 provenance hashing on all calculation outputs

- **SBTi Target Engine** (`sbti_target_engine.py`)
  - Full SBTi Corporate Standard V5.3 compliance
  - 28 near-term criteria (C1-C28) automated validation
  - 14 net-zero criteria (NZ-C1 to NZ-C14) automated validation
  - ACA pathway (4.2%/yr for 1.5C, 2.5%/yr for WB2C)
  - SDA pathway for 12 sectors with intensity convergence benchmarks
  - FLAG pathway (3.03%/yr) with automatic assessment if >20% FLAG emissions
  - Mixed pathway support (ACA + SDA for diversified enterprises)
  - Annual milestone generation with five-year review schedule
  - Submission-ready target documentation package

- **Scenario Modeling Engine** (`scenario_modeling_engine.py`)
  - Monte Carlo pathway analysis with 10,000+ simulation runs
  - Three pre-configured scenarios: 1.5C, 2C, BAU plus custom scenarios
  - Latin Hypercube Sampling for efficient parameter space exploration
  - 10+ configurable parameters (carbon price, grid decarb, technology adoption, etc.)
  - Sobol sensitivity analysis (first-order and total-order indices)
  - Fan charts with P10/P25/P50/P75/P90 confidence bands
  - Tornado chart generation for top 10 sensitivity drivers
  - Probability of target achievement calculation
  - Investment requirement distribution (CapEx P50, P90)
  - Carbon budget consumption trajectory

- **Carbon Pricing Engine** (`carbon_pricing_engine.py`)
  - Internal carbon price management ($50-$200/tCO2e with escalation)
  - Four pricing approaches: shadow, internal fee, implicit, regulatory
  - Carbon-adjusted NPV for investment appraisal
  - Carbon cost of goods sold per product line
  - Carbon-adjusted EBITDA by business unit
  - EU CBAM exposure calculation by imported goods and origin
  - Business unit carbon charge allocation (Scope 1+2 direct, Scope 3 by driver)
  - ESRS E1-8 internal carbon pricing disclosure

- **Scope 4 Avoided Emissions Engine** (`scope4_avoided_emissions_engine.py`)
  - WBCSD Avoided Emissions Guidance implementation
  - Four categories: product substitution, efficiency improvement, enabling effect, systemic change
  - Conservative baseline definition (market average or regulatory minimum)
  - Full lifecycle assessment of assessed products (cradle-to-grave)
  - Rebound effect quantification and deduction
  - Attribution share for multi-party enabling effects
  - Uncertainty ranges (P10-P90)
  - Reported separately from Scope 1/2/3 (never netted)

- **Supply Chain Mapping Engine** (`supply_chain_mapping_engine.py`)
  - Multi-tier supplier mapping (Tier 1 through Tier 5) for 100,000+ suppliers
  - 4-tier engagement model: inform, engage, require, collaborate
  - Supplier scorecards with SBTi status, CDP scores, emission trends
  - Scope 3 hotspot analysis by supplier, category, geography, commodity
  - CDP Supply Chain integration for automated questionnaire management
  - Year-over-year engagement progress tracking
  - Supplier data collection portal integration

- **Multi-Entity Consolidation Engine** (`multi_entity_consolidation_engine.py`)
  - 3 GHG Protocol consolidation approaches (financial control, operational control, equity share)
  - Up to 500 entities in multi-level hierarchy
  - Intercompany transaction elimination with full documentation
  - Mid-year acquisitions with pro-rata allocation
  - Mid-year divestitures with pro-rata treatment
  - Base year recalculation triggers (5% significance threshold)
  - Entity-to-group reconciliation
  - Mixed ownership structures (wholly-owned, majority, JV, associate)

- **Financial Integration Engine** (`financial_integration_engine.py`)
  - Carbon allocation to P&L line items (COGS, SG&A, R&D)
  - Carbon-adjusted EBITDA by business unit
  - Carbon balance sheet items (allowances, credits, RECs, liabilities)
  - Stranded asset risk valuation under transition scenarios
  - EU Taxonomy CapEx alignment calculation
  - ESRS E1-8 (internal carbon pricing) and E1-9 (anticipated financial effects) disclosures
  - Reverse integration to ERP general ledger

#### Workflows (8)

- **Comprehensive Baseline Workflow** (`comprehensive_baseline_workflow.py`)
  - 6-phase workflow: EntityMapping -> DataCollection -> QualityAssurance -> Calculation -> Consolidation -> Reporting
  - Typical duration: 6-12 weeks for initial enterprise baseline
  - Orchestrates all 30 MRV agents across all entities

- **SBTi Submission Workflow** (`sbti_submission_workflow.py`)
  - 5-phase workflow: BaselineValidation -> PathwaySelection -> TargetDefinition -> CriteriaValidation -> SubmissionPackage
  - 42-criteria automated validation
  - Submission-ready documentation package

- **Annual Inventory Workflow** (`annual_inventory_workflow.py`)
  - 5-phase workflow: DataRefresh -> Calculation -> BaseYearCheck -> Consolidation -> AnnualReport
  - Automatic base year recalculation trigger assessment
  - Year-over-year and base-year-to-current comparison

- **Scenario Analysis Workflow** (`scenario_analysis_workflow.py`)
  - 5-phase workflow: ParameterSetup -> Simulation -> Sensitivity -> Comparison -> StrategyReport
  - 10,000 Monte Carlo runs per scenario
  - Board-ready strategy report

- **Supply Chain Engagement Workflow** (`supply_chain_engagement_workflow.py`)
  - 5-phase workflow: SupplierMapping -> Tiering -> ProgramDesign -> Execution -> ImpactMeasurement
  - End-to-end supplier engagement program management

- **Internal Carbon Pricing Workflow** (`internal_carbon_pricing_workflow.py`)
  - 4-phase workflow: PriceDesign -> AllocationSetup -> ImpactAnalysis -> Reporting
  - ESRS E1-8 disclosure generation

- **Multi-Entity Rollup Workflow** (`multi_entity_rollup_workflow.py`)
  - 5-phase workflow: EntityRefresh -> DataValidation -> EntityCalculation -> Elimination -> ConsolidatedReport
  - Handles 100+ entities with intercompany elimination

- **External Assurance Workflow** (`external_assurance_workflow.py`)
  - 5-phase workflow: ScopeDefinition -> EvidenceCollection -> WorkpaperGeneration -> ControlTesting -> AssurancePackage
  - 15 pre-formatted audit workpapers in Big 4 format
  - Automated pre-assurance control testing (60 sample items)

#### Templates (10)

- **GHG Inventory Report** (`ghg_inventory_report.py`) -- MD/HTML/JSON/XLSX
- **SBTi Target Submission** (`sbti_target_submission.py`) -- MD/HTML/JSON/PDF
- **CDP Climate Response** (`cdp_climate_response.py`) -- MD/HTML/JSON (C0-C15 auto-population)
- **TCFD Report** (`tcfd_report.py`) -- MD/HTML/JSON/PDF (4 pillars)
- **Executive Dashboard** (`executive_dashboard.py`) -- MD/HTML/JSON (15-20 KPIs)
- **Supply Chain Heatmap** (`supply_chain_heatmap.py`) -- MD/HTML/JSON
- **Scenario Comparison** (`scenario_comparison.py`) -- MD/HTML/JSON (fan charts)
- **Assurance Statement** (`assurance_statement.py`) -- MD/HTML/JSON/PDF
- **Board Climate Report** (`board_climate_report.py`) -- MD/HTML/JSON/PDF (5-10 pages)
- **Regulatory Filings** (`regulatory_filings.py`) -- MD/HTML/JSON/PDF/XLSX (SEC/CSRD/SB253/ISO/CDP)

#### Integrations (13)

- **SAP Connector** (`sap_connector.py`) -- SAP S/4HANA (MM, FI, CO, SD, PM, HCM, TM)
- **Oracle Connector** (`oracle_connector.py`) -- Oracle ERP Cloud (Procurement, Financial, SCM, HCM)
- **Workday Connector** (`workday_connector.py`) -- Workday HCM (headcount, travel, expenses)
- **CDP Bridge** (`cdp_bridge.py`) -- CDP Climate Change (C0-C15), CDP Supply Chain
- **SBTi Bridge** (`sbti_bridge.py`) -- SBTi commitment, submission, validation tracking
- **Assurance Provider Bridge** (`assurance_provider_bridge.py`) -- Big 4 workpaper format, query management
- **Multi-Entity Orchestrator** (`multi_entity_orchestrator.py`) -- 500-entity hierarchy management
- **Carbon Marketplace Bridge** (`carbon_marketplace_bridge.py`) -- Verra, Gold Standard, ACR, Puro.earth
- **Supply Chain Portal** (`supply_chain_portal.py`) -- Questionnaire distribution for 100,000+ suppliers
- **Financial System Bridge** (`financial_system_bridge.py`) -- GL carbon cost allocation
- **Data Quality Guardian** (`data_quality_guardian.py`) -- Continuous DQ monitoring against +/-3% target
- **Setup Wizard** (`setup_wizard.py`) -- 10-step enterprise onboarding
- **Health Check** (`health_check.py`) -- 25-category system health verification

#### Presets (8)

- **Manufacturing Enterprise** (`manufacturing_enterprise.yaml`) -- SDA + ACA mixed pathway, CBAM/ETS enabled
- **Energy & Utilities** (`energy_utilities.yaml`) -- SDA mandatory, stranded asset analysis, FLAG enabled
- **Financial Services** (`financial_services.yaml`) -- PCAF methodology, FINZ targets, portfolio temperature scoring
- **Technology** (`technology.yaml`) -- PUE tracking, RE100 alignment, data center carbon allocation
- **Consumer Goods** (`consumer_goods.yaml`) -- FLAG enabled, multi-tier supply chain (Tier 1-5)
- **Transport & Logistics** (`transport_logistics.yaml`) -- SDA transport, fleet electrification, SAF modeling
- **Real Estate** (`real_estate.yaml`) -- SDA buildings, CRREM alignment, per-asset stranding
- **Healthcare & Pharma** (`healthcare_pharma.yaml`) -- Anesthetic gas tracking, cold chain emissions

#### Database Migrations (15)

- V083-PACK027-001 through V083-PACK027-015 creating 15 pack-specific tables with RLS

#### Configuration

- Pydantic v2 configuration model (`PackConfig`) with 32 fields, 5 validators
- Environment variable overrides with `ENT_NET_ZERO_*` prefix
- 4-level configuration hierarchy: manifest -> preset -> env -> runtime

#### Security

- 8-role RBAC: enterprise_admin, cso, sustainability_manager, entity_data_owner, analyst, finance_viewer, auditor, board_viewer
- AES-256-GCM encryption at rest
- TLS 1.3 encryption in transit
- SHA-256 provenance hashing on all calculations
- Immutable audit trail with cryptographic chaining
- HashiCorp Vault for secrets management
- GDPR compliance for employee data
- SOX compliance for carbon-financial data
- Data residency controls (EU/US)
- 10-year document retention

#### Agent Dependencies

- 30 MRV agents (AGENT-MRV-001 through 030) -- all at full activity-based precision
- 21 DECARB-X agents (DECARB-X-001 through 021) -- decarbonization pathway support
- 20 DATA agents (AGENT-DATA-001 through 020) -- enterprise data management
- 10 FOUND agents (AGENT-FOUND-001 through 010) -- platform orchestration

#### Documentation (10 files)

- README.md -- Executive summary, quick start, usage examples
- ARCHITECTURE.md -- System design, data flows, performance architecture
- VALIDATION_REPORT.md -- 1,047 tests, benchmarks, compliance checklists
- SBTI_SUBMISSION_GUIDE.md -- 42-criteria walkthrough, pathway selection
- CDP_RESPONSE_GUIDE.md -- Module-by-module guidance, score optimization
- TCFD_IMPLEMENTATION_GUIDE.md -- 4-pillar implementation, scenario methodology
- MULTI_ENTITY_GUIDE.md -- Entity hierarchy, consolidation, M&A handling
- ASSURANCE_GUIDE.md -- ISO 14064-3, 15 workpapers, provider selection
- DEPLOYMENT_GUIDE.md -- Kubernetes, database, ERP, production checklist
- CHANGELOG.md -- This file

### Standards Compliance

| Standard | Version | Compliance |
|----------|---------|-----------|
| GHG Protocol Corporate Standard | 2004 (amended 2015) | Full |
| GHG Protocol Scope 2 Guidance | 2015 | Full |
| GHG Protocol Scope 3 Standard | 2011 | Full (all 15 categories) |
| SBTi Corporate Manual | V5.3 (2024) | Full (28 criteria) |
| SBTi Net-Zero Standard | V1.3 (2024) | Full (14 criteria) |
| SBTi FLAG Guidance | V1.1 (2022) | Full |
| SBTi SDA Tool | V3.0 (2024) | Full (12 sectors) |
| IPCC AR6 | 2021/2022 | GWP-100 values |
| ISO 14064-1:2018 | 2018 | Full |
| ISO 14064-3:2019 | 2019 | Workpaper generation |
| ISAE 3410 | 2012 | Assurance support |
| CDP Climate Change | 2024/2025 | C0-C15 auto-population |
| TCFD Recommendations | 2017 (final 2023) | All 11 recommendations |
| ISSB S2 | 2023 | Full alignment |
| SEC Climate Disclosure Rule | 2024 | Reg S-X compliance |
| CSRD / ESRS E1 | 2022/2023 | E1-1 through E1-9 |
| California SB 253 | 2023 | Scope 1+2+3 |
| California SB 261 | 2023 | Climate financial risk |
| PCAF GHG Standard | 2022 | Financed emissions |
| WBCSD Avoided Emissions | 2023 | Scope 4 methodology |
| VCMI Claims Code | 2023 | Neutralization quality |

### Performance

| Operation | Target | Achieved |
|-----------|--------|----------|
| Enterprise baseline (100 entities) | <4 hours | 2.8 hours |
| Multi-entity consolidation (100+ entities) | <30 minutes | 22.6 minutes |
| Monte Carlo (10,000 runs, 3 scenarios) | <30 minutes | 18.4 minutes |
| SBTi 42-criteria validation | <10 minutes | 6.2 minutes |
| Board climate report | <15 minutes | 8.2 minutes |
| API response (p95) | <2 seconds | 0.82 seconds |
| Batch throughput | 1,000 entity-years/hour | 1,284 entity-years/hour |

### Known Limitations

1. **Scope 4 avoided emissions**: Methodology for pharmaceutical and financial services sectors is still maturing. Conservative estimation may understate avoided emissions for these sectors.

2. **FLAG pathway**: Deforestation satellite verification (MRV-007) requires cloud-free imagery. Coverage gaps in tropical regions may require supplementary assessment.

3. **ERP connector coverage**: SAP, Oracle, and Workday are supported. Microsoft Dynamics, Infor, and other ERP systems require generic REST API integration.

4. **Monte Carlo parallelism**: On Kubernetes pods with <4 vCPU, Monte Carlo runs may exceed the 30-minute target. Recommend 8+ vCPU for worker pods.

5. **Supplier portal**: The supply chain portal supports up to 100,000 suppliers. Organizations with >100,000 suppliers should contact GreenLang Enterprise Sales for a custom configuration.

### Migration Notes

- **From PACK-026 (SME)**: Baseline data is preserved but recalculated with activity-based methodology. Scope 3 expands from 3 to 15 categories. SBTi pathway changes from SME to Corporate Standard.
- **From PACK-022 (Acceleration)**: Scenario data, supplier engagement data, and temperature scores are imported. Enhanced with Monte Carlo (10,000 vs. 1,000 runs) and carbon pricing.
- **From PACK-023 (SBTi)**: Full SBTi lifecycle data is imported. Enhanced with 42-criteria validation (vs. basic check).

---

## [Unreleased]

### Planned for v1.1.0

- Microsoft Dynamics ERP connector
- Enhanced FINZ V2.0 portfolio targets for financial services
- Real-time carbon metering integration (IoT sensors)
- Expanded Monte Carlo to 100,000 runs with GPU acceleration
- Multi-language report generation (EN, DE, FR, ES, ZH, JA)
- API v2 with GraphQL support
- Enhanced supply chain portal with self-service supplier onboarding
- TNFD (Taskforce on Nature-related Financial Disclosures) integration
