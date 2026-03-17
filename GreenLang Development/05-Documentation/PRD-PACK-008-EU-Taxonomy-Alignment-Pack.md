# PRD-PACK-008: EU Taxonomy Alignment Pack

**Status:** Approved
**Version:** 1.0.0
**Priority:** P1 - High
**Category:** Solution Pack - EU Compliance
**Author:** GreenLang Product Team
**Created:** 2026-03-15
**Regulation:** EU Taxonomy Regulation (EU) 2020/852

---

## 1. Executive Summary

PACK-008 is the EU Taxonomy Alignment Pack that packages the GL-Taxonomy-APP (APP-010) together with MRV agents, data intake agents, and foundation agents into a deployable compliance solution. It enables organizations to screen economic activities for taxonomy eligibility, assess alignment against the 6 environmental objectives, calculate mandatory KPIs (Turnover/CapEx/OpEx), compute Green Asset Ratios for financial institutions, and generate Article 8 / EBA Pillar 3 disclosures.

**Key Capabilities:**
- Activity screening against ~240 EU Taxonomy economic activities
- Substantial Contribution (SC) assessment with Technical Screening Criteria (TSC)
- Do No Significant Harm (DNSH) assessment across 6 environmental objectives
- Minimum Safeguards (MS) verification (human rights, anti-corruption, taxation, fair competition)
- KPI calculation: Turnover, CapEx, OpEx alignment ratios
- Green Asset Ratio (GAR) and Banking Book Taxonomy Alignment Ratio (BTAR) for financial institutions
- Article 8 Delegated Regulation (EU) 2021/2178 disclosure templates
- EBA Pillar 3 ESG disclosure templates (Templates 6-10)
- Cross-regulation alignment with CSRD/ESRS E1, SFDR, TCFD, CDP

---

## 2. Background

### 2.1 EU Taxonomy Regulation
The EU Taxonomy Regulation (EU) 2020/852 establishes a classification system for environmentally sustainable economic activities. Organizations subject to CSRD/NFRD must report what proportion of their economic activities qualify as taxonomy-aligned.

### 2.2 Four Alignment Conditions
An economic activity is taxonomy-aligned if it:
1. **Substantially Contributes** to at least one of six environmental objectives
2. **Does No Significant Harm** to any of the other five objectives
3. Complies with **Minimum Safeguards** (OECD Guidelines, UN Guiding Principles)
4. Meets the **Technical Screening Criteria** defined in Delegated Acts

### 2.3 Six Environmental Objectives
1. Climate Change Mitigation (CCM)
2. Climate Change Adaptation (CCA)
3. Sustainable Use and Protection of Water and Marine Resources (WTR)
4. Transition to a Circular Economy (CE)
5. Pollution Prevention and Control (PPC)
6. Protection and Restoration of Biodiversity and Ecosystems (BIO)

### 2.4 Delegated Acts
- Climate Delegated Act (EU) 2021/2139 - CCM and CCA criteria
- Environmental Delegated Act (EU) 2023/2486 - WTR, CE, PPC, BIO criteria
- Complementary Climate DA (EU) 2022/1214 - Nuclear and gas activities
- Disclosures DA (EU) 2021/2178 - Article 8 reporting templates
- Simplification DA 2025 - Omnibus simplification package

### 2.5 Existing Platform
GL-Taxonomy-APP (APP-010) provides: 10 backend engines, 16 API routers (~130 endpoints), 25 database tables (V088), ~240 economic activities, full alignment orchestration.

---

## 3. Goals & Objectives

### 3.1 Primary Goals
1. Package GL-Taxonomy-APP into a deployable solution pack
2. Connect taxonomy assessment to GHG emissions data from MRV agents
3. Enable cross-framework disclosure (CSRD E1, SFDR, TCFD, CDP)
4. Provide pre-built workflows for eligibility screening through disclosure generation
5. Support both non-financial undertakings (KPIs) and financial institutions (GAR/BTAR)

### 3.2 Success Metrics
- 100% coverage of ~240 taxonomy economic activities
- All 6 environmental objectives supported
- 4-condition alignment test automated
- Article 8 + EBA Pillar 3 disclosure generation
- 140+ tests with 0 failures

---

## 4. Architecture

### 4.1 Pack Structure
```
PACK-008-eu-taxonomy-alignment/
  pack.yaml
  __init__.py
  config/
    pack_config.py
    presets/ (5 presets)
    sectors/ (6 sector configs)
    demo/demo_config.yaml
  engines/ (10 engines)
  workflows/ (10 workflows)
  templates/ (10 templates)
  integrations/ (12 integrations)
  tests/ (20 test files)
```

### 4.2 Component Summary
| Category | Count | Description |
|----------|-------|-------------|
| GL-Taxonomy-APP | 1 | Core platform (10 engines, 16 routers) |
| MRV Agents | 30 | Scope 1/2/3 emissions for CCM/CCA TSC |
| Data Agents | 10 | PDF, Excel, ERP, quality, validation |
| Foundation Agents | 10 | Orchestrator, schema, citations, QA |
| Pack Engines | 10 | Taxonomy-specific calculation engines |
| Pack Workflows | 10 | Pre-built compliance workflows |
| Pack Templates | 10 | Disclosure and report templates |
| Pack Integrations | 12 | Bridge modules to agents and apps |
| **Total Agents** | **51** | 1 app + 30 MRV + 10 data + 10 foundation |

---

## 5. Engine Specifications (10 Engines)

### 5.1 Taxonomy Eligibility Engine
- NACE code mapping to ~240 taxonomy activities
- Sector-level eligibility screening
- Batch processing for activity portfolios
- Eligibility vs. alignment distinction
- Revenue-weighted eligibility ratios

### 5.2 Substantial Contribution Engine
- TSC evaluation per environmental objective
- Quantitative threshold checking (emissions, energy, water, waste metrics)
- Enabling activity classification (Article 16)
- Transitional activity classification (Article 10(2))
- Evidence linking to TSC compliance

### 5.3 DNSH Assessment Engine
- 6-objective DNSH matrix evaluation
- Climate risk and vulnerability assessment (DNSH for CCA)
- Water Framework Directive compliance (DNSH for WTR)
- Circular economy waste hierarchy check (DNSH for CE)
- Pollution thresholds (DNSH for PPC)
- Biodiversity impact assessment (DNSH for BIO)

### 5.4 Minimum Safeguards Engine
- Human rights due diligence (UNGP, OECD Guidelines)
- Anti-corruption procedures assessment
- Taxation compliance verification
- Fair competition assessment
- Procedural + outcome checks per topic
- 4-topic pass/fail determination

### 5.5 KPI Calculation Engine
- Turnover alignment ratio (taxonomy-aligned / total turnover)
- CapEx alignment ratio (taxonomy-aligned / total CapEx)
- OpEx alignment ratio (taxonomy-aligned / total OpEx)
- Double-counting prevention across objectives
- Activity-level financial data mapping
- CapEx plan recognition (5-year plans)
- Eligible vs. aligned KPI breakdown

### 5.6 Green Asset Ratio Engine
- GAR stock calculation (on-balance-sheet assets)
- GAR flow calculation (new originations)
- BTAR - Banking Book Taxonomy Alignment Ratio
- Exposure classification (corporate loans, debt securities, equity, mortgages, project finance)
- Counterparty taxonomy data aggregation
- EPC rating integration for real estate exposures
- De minimis threshold handling

### 5.7 Technical Screening Criteria Engine
- Criteria lookup per activity + objective
- Quantitative threshold evaluation
- Qualitative criteria assessment
- Delegated act version management
- Criteria change tracking across DA versions
- Gap identification for non-compliant criteria

### 5.8 Transition Activity Engine
- Article 10(2) transition activity identification
- Best available technology assessment
- Lock-in avoidance verification
- Transition pathway documentation
- Sunset date tracking for transitional status

### 5.9 Enabling Activity Engine
- Article 16 enabling activity classification
- Direct enablement verification
- Life-cycle considerations
- Technology lock-in avoidance
- Market distortion assessment

### 5.10 Taxonomy Reporting Engine
- Article 8 disclosure template generation
- EBA Pillar 3 Templates 6-10
- XBRL/iXBRL tagging support
- Mandatory table generation (Turnover, CapEx, OpEx)
- Nuclear/gas supplementary disclosures
- Year-over-year comparison tables

---

## 6. Workflow Specifications (10 Workflows)

### 6.1 Eligibility Screening Workflow
4 phases: Activity Inventory -> NACE Mapping -> Eligibility Assessment -> Eligibility Report

### 6.2 Alignment Assessment Workflow
5 phases: SC Evaluation -> DNSH Assessment -> MS Verification -> Alignment Determination -> Evidence Package

### 6.3 KPI Calculation Workflow
4 phases: Financial Data Collection -> Activity Mapping -> KPI Computation -> Disclosure Preparation

### 6.4 GAR Calculation Workflow
4 phases: Exposure Inventory -> Counterparty Data -> GAR/BTAR Computation -> EBA Template Generation

### 6.5 Article 8 Disclosure Workflow
4 phases: Data Validation -> Template Population -> Review & Approval -> Filing Package

### 6.6 Gap Analysis Workflow
3 phases: Current State Assessment -> Gap Identification -> Remediation Planning

### 6.7 CapEx Plan Workflow
4 phases: Plan Definition -> Alignment Projection -> Approval -> Monitoring

### 6.8 Regulatory Update Workflow
3 phases: DA Version Tracking -> Impact Assessment -> Criteria Migration

### 6.9 Cross-Framework Alignment Workflow
4 phases: Taxonomy KPI Extraction -> CSRD/ESRS Mapping -> SFDR Integration -> Consolidated Disclosure

### 6.10 Annual Taxonomy Review Workflow
5 phases: Activity Reassessment -> KPI Recalculation -> Trend Analysis -> Board Reporting -> Action Planning

---

## 7. Template Specifications (10 Templates)

### 7.1 Eligibility Matrix Report
Activity-level eligibility results per objective, NACE sector breakdown

### 7.2 Alignment Summary Report
Portfolio-level alignment results, SC/DNSH/MS pass rates, aligned vs. eligible ratios

### 7.3 Article 8 Disclosure Template
Mandatory disclosure tables (Turnover, CapEx, OpEx), nuclear/gas supplementary templates

### 7.4 EBA Pillar 3 GAR Report
Templates 6-10 for credit institutions, GAR stock/flow, BTAR

### 7.5 KPI Dashboard
Turnover/CapEx/OpEx alignment ratios, year-over-year trends, activity-level breakdown

### 7.6 Gap Analysis Report
Gap inventory, severity classification, remediation roadmap, cost estimation

### 7.7 TSC Compliance Report
Per-activity technical screening criteria results, evidence status, non-compliance details

### 7.8 DNSH Assessment Report
6-objective DNSH matrix, climate risk assessment, water/CE/pollution/biodiversity results

### 7.9 Executive Summary
Board-level overview, key metrics, regulatory compliance status, strategic recommendations

### 7.10 Detailed Assessment Report
Full audit trail, activity-level detail, evidence inventory, provenance hashes

---

## 8. Integration Specifications (12 Integrations)

### 8.1 Pack Orchestrator
10-phase pipeline: Health Check -> Configuration -> Activity Inventory -> Eligibility -> SC Assessment -> DNSH -> MS -> KPI/GAR -> Disclosure -> Audit Trail

### 8.2 Taxonomy App Bridge
Wire GL-Taxonomy-APP's 10 engines into pack workflows with unified config

### 8.3 MRV Taxonomy Bridge
Route Scope 1/2/3 emissions data to CCM/CCA TSC evaluations

### 8.4 CSRD Cross-Framework Bridge
Map taxonomy KPIs to ESRS E1 disclosures, SFDR Article 8/9, TCFD metrics

### 8.5 Financial Data Bridge
Connect ERP/finance systems for turnover, CapEx, OpEx data intake

### 8.6 Activity Registry Bridge
Manage ~240 economic activity catalog with NACE mappings and DA versions

### 8.7 Evidence Management Bridge
Link documents, certifications, audit reports to TSC/DNSH/MS assessments

### 8.8 GAR Data Bridge
Aggregate counterparty taxonomy data for financial institution GAR calculation

### 8.9 Regulatory Tracking Bridge
Monitor Delegated Act updates, manage criteria version transitions

### 8.10 Data Quality Bridge
Validate taxonomy data completeness, accuracy, and consistency

### 8.11 Health Check
20-category system verification (app, agents, config, data, APIs)

### 8.12 Setup Wizard
10-step guided configuration for taxonomy alignment assessment

---

## 9. Configuration Model

### 9.1 Pack Configuration
- `TaxonomyAlignmentConfig` - Main configuration class
- `EligibilityConfig` - Screening parameters
- `SCAssessmentConfig` - Substantial contribution settings
- `DNSHConfig` - DNSH assessment parameters
- `MinimumSafeguardsConfig` - MS verification settings
- `KPIConfig` - Financial KPI calculation parameters
- `GARConfig` - Green Asset Ratio settings
- `ReportingConfig` - Disclosure generation settings
- `RegulatoryConfig` - DA version tracking settings

### 9.2 Presets (5)
- `non_financial_undertaking.yaml` - Standard KPI focus
- `financial_institution.yaml` - GAR/BTAR focus
- `asset_manager.yaml` - Investment taxonomy focus
- `large_enterprise.yaml` - All objectives, full disclosure
- `sme_simplified.yaml` - De minimis eligible, simplified

### 9.3 Sectors (6)
- `energy.yaml` - Power generation, renewables, grid
- `manufacturing.yaml` - Industrial processes, cement, steel
- `real_estate.yaml` - Building renovation, construction, EPC
- `transport.yaml` - Low-carbon vehicles, rail, shipping
- `forestry_agriculture.yaml` - Afforestation, restoration, organic
- `financial_services.yaml` - Banking, insurance, asset management

---

## 10. Test Plan

### 10.1 Unit Tests (Target: 140+ tests)
| Test File | Focus Area | Est. Tests |
|-----------|------------|------------|
| test_manifest.py | pack.yaml validation | 19 |
| test_config.py | Configuration loading | 25 |
| test_eligibility.py | Eligibility screening | 18 |
| test_substantial_contribution.py | SC assessment | 16 |
| test_dnsh.py | DNSH 6-objective matrix | 16 |
| test_minimum_safeguards.py | MS 4-topic verification | 12 |
| test_kpi_calculation.py | Turnover/CapEx/OpEx KPIs | 16 |
| test_gar.py | GAR/BTAR calculation | 14 |
| test_tsc.py | Technical screening criteria | 12 |
| test_reporting.py | Article 8/EBA templates | 14 |
| test_workflows.py | Workflow orchestration | 10 |
| test_templates.py | Report generation | 10 |
| test_integrations.py | Bridge modules | 12 |
| test_demo.py | Demo mode | 6 |
| test_e2e.py | End-to-end flows | 8 |
| test_agent_integration.py | Live agent loading | 15 |

### 10.2 Integration Tests
- Cross-pack compatibility with PACK-001/002/003 (CSRD)
- MRV agent data flow to TSC evaluation
- GL-Taxonomy-APP bridge verification
- Agent loader registry for taxonomy agents

---

## 11. Implementation Tasks

### Phase 1: Package Infrastructure
1. Create directory structure and __init__.py files
2. Build pack.yaml manifest
3. Build pack_config.py with all config models
4. Create 5 presets and 6 sector configs

### Phase 2: Engines
5. Build 5 engines (eligibility, SC, DNSH, MS, KPI)
6. Build 5 engines (GAR, TSC, transition, enabling, reporting)

### Phase 3: Workflows & Templates
7. Build 10 workflows
8. Build 10 templates

### Phase 4: Integrations
9. Build 12 integration bridges

### Phase 5: Testing
10. Build unit tests (conftest + 16 test files)
11. Build integration tests + runner updates

---

## 12. Assets Leveraged

### Existing (No Development Needed)
- GL-Taxonomy-APP v1.0 (10 engines, 16 routers, 25 tables)
- AGENT-MRV-001 through 030 (all Scope 1/2/3 emissions)
- AGENT-DATA-001 through 019 (data intake and quality)
- AGENT-FOUND-001 through 010 (foundation agents)
- Database migration V088 (taxonomy schema)
- Auth integration (taxonomy permissions in PERMISSION_MAP)

### New Development (This Pack)
- 10 pack-specific engines
- 10 workflows
- 10 templates
- 12 integrations
- 20 test files
- Pack configuration and presets

---

## 13. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| DA version changes | Medium | High | Regulatory tracking engine with version management |
| Complex NACE mapping | Low | Medium | Pre-built ~240 activity catalog from GL-Taxonomy-APP |
| Financial data quality | Medium | High | Data quality bridge with validation rules |
| Cross-framework conflicts | Low | Medium | Cross-framework bridge with mapping tables |
| GAR data availability | Medium | Medium | Counterparty data templates with defaults |

---

## 14. Deployment Notes

### Prerequisites
- GL-Taxonomy-APP v1.0 deployed
- V088 database migration applied
- MRV agents (001-030) operational
- Data agents (001-019) operational
- Foundation agents (001-010) operational

### Configuration
- Set organization type (NFU vs. FI) in preset selection
- Map NACE codes to company's economic activities
- Configure environmental objectives for assessment
- Set disclosure period (annual, semi-annual, quarterly)

### Performance
- Activity screening: <100ms per activity
- Alignment assessment: <5s per activity (all 4 conditions)
- KPI calculation: <10s for full portfolio
- GAR calculation: <30s for banking book
- Report generation: <15s for Article 8 template
