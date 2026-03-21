# PRD-PACK-021: Net Zero Starter Pack

**Pack ID:** PACK-021-net-zero-starter
**Category:** Net Zero Packs
**Tier:** Starter
**Version:** 1.0.0
**Status:** Approved
**Author:** GreenLang Product Team
**Date:** 2026-03-18

---

## 1. Executive Summary

### 1.1 Problem Statement

Achieving net zero emissions by 2050 (or sooner) is the defining corporate sustainability challenge of our time. The Science Based Targets initiative (SBTi) Corporate Net-Zero Standard requires companies to set near-term targets (5-10 years, 1.5C-aligned, 4.2% annual linear reduction), long-term targets (by 2050, 90%+ reduction from base year), and neutralize residual emissions through permanent carbon dioxide removals. Over 9,000 companies have committed to SBTi, yet most lack the tools to build a coherent net-zero strategy that integrates:

1. **Complete GHG inventory baseline**: Net zero starts with a full Scope 1, 2, and 3 GHG inventory per GHG Protocol methodology. Most companies have Scope 1 and 2 but cover only 3-5 of 15 Scope 3 categories, understating total emissions by 40-80%.
2. **Science-aligned target setting**: SBTi Net-Zero Standard v1.2 requires absolute contraction (ACA), sector decarbonization (SDA), or FLAG pathway targets. Companies struggle with pathway selection, coverage requirements (95% Scope 1+2, 67%+ Scope 3), and the distinction between near-term vs. long-term targets.
3. **Reduction pathway planning**: Translating targets into actionable decarbonization levers (energy efficiency, electrification, fuel switching, renewable procurement, supplier engagement) with Marginal Abatement Cost Curves (MACCs), technology readiness levels, and phased implementation roadmaps.
4. **Offset/neutralization management**: SBTi distinguishes "compensation" (near-term, voluntary credits to fund climate action beyond the value chain) from "neutralization" (long-term, permanent removals for residual emissions). Companies conflate these concepts, risking greenwashing claims.
5. **Progress monitoring**: Annual tracking of absolute emissions against the reduction pathway, intensity metrics, progress toward near-term and long-term targets, and gap analysis with corrective actions.

### 1.2 Solution Overview

PACK-021 is the **Net Zero Starter Pack** -- the first pack in the new "Net Zero Packs" category. It provides a complete, guided journey from GHG baseline establishment through net-zero target setting, reduction planning, offset strategy, and progress monitoring. The pack wraps and orchestrates existing GreenLang platform components (30 MRV agents, GL-GHG-APP, GL-SBTi-APP, 21 DECARB-X agents, carbon credit agents) into a coherent net-zero workflow with 8 new engines, 6 workflows, 8 templates, 10 integrations, and 6 sector presets.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-021 Net Zero Starter Pack |
|-----------|-------------------------------|-------------------------------|
| Time to build net-zero strategy | 400-800 hours | <40 hours (10-20x faster) |
| GHG baseline accuracy | Error-prone, incomplete Scope 3 | Deterministic, all 15 Scope 3 categories |
| Target validation | Manual SBTi criteria check | Automated SBTi Net-Zero Standard v1.2 validation |
| Reduction pathway | Qualitative action lists | Quantified MACC with NPV/IRR, phased roadmap |
| Offset quality | Unverified credit purchases | Quality-scored credits with standard verification |
| Progress tracking | Annual spreadsheet update | Continuous monitoring with gap analysis and alerts |
| Audit readiness | Manual documentation | SHA-256 provenance, full calculation lineage |
| Cross-framework alignment | Manual reconciliation | Automated (GHG Protocol, SBTi, ESRS E1, CDP, TCFD) |

### 1.4 Target Users

**Primary:**
- Sustainability managers initiating net-zero programs
- Climate officers setting science-based targets
- Companies with SBTi commitments (or planning to commit)
- SMEs and mid-cap companies starting their net-zero journey

**Secondary:**
- Board members reviewing net-zero strategies
- Sustainability consultants building net-zero roadmaps for clients
- Investor relations teams communicating climate ambition
- ESG rating agencies evaluating corporate net-zero plans

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete net-zero baseline | <8 hours (vs. 100+ manual) | Time from data upload to validated baseline |
| GHG calculation accuracy | 100% match with manual verification | Tested against 500 known emission values |
| Scope 3 category coverage | 15/15 categories | Number of categories with calculations |
| SBTi target validation accuracy | 100% | Validated against SBTi criteria checker |
| Reduction pathway quantification | 100% of identified actions costed | Actions with $/tCO2e and timeline |
| Customer NPS | >50 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| SBTi Corporate Net-Zero Standard | SBTi v1.2 (2024) | Core target-setting framework; near-term + long-term + neutralization |
| GHG Protocol Corporate Standard | WRI/WBCSD (2004, 2015 update) | Scope 1 and 2 GHG inventory methodology |
| GHG Protocol Scope 3 Standard | WRI/WBCSD (2011) | Scope 3 (categories 1-15) methodology |
| IPCC AR6 | IPCC Sixth Assessment Report (2021) | GWP-100 values for all greenhouse gases |
| Paris Agreement | UNFCCC (2015) | 1.5C temperature alignment target |

### 2.2 Supporting Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| SBTi FLAG Guidance | SBTi (2022) | Land use sector targets |
| ISO 14064-1:2018 | ISO | Organization-level GHG quantification |
| ESRS E1 Climate Change | EU Delegated Reg. 2023/2772 | E1-4 targets, E1-6 emissions alignment |
| CDP Climate Change Questionnaire | CDP (2024) | C4 (Targets), C6 (Emissions), C7 (Energy) alignment |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics & Targets, Strategy alignment |
| VCMI Claims Code | VCMI (2023) | Carbon credit quality and claims guidance |
| ISO 14068-1:2023 | ISO | Carbon neutrality quantification |
| Oxford Principles for Net Zero Aligned Carbon Offsetting | Oxford (2020) | Offset quality hierarchy |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 8 | Deterministic calculation engines |
| Workflows | 6 | Multi-phase orchestration workflows |
| Templates | 8 | Report and dashboard templates |
| Integrations | 10 | Agent and application bridges |
| Presets | 6 | Sector-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose | Key Inputs | Key Outputs |
|---|--------|---------|------------|-------------|
| 1 | `net_zero_baseline_engine.py` | Unified GHG baseline assessment combining Scope 1, 2, 3 with base year selection and validation | Energy data, fuel consumption, procurement, fleet, facilities | BaselineResult: total emissions by scope, base year, data quality scores |
| 2 | `net_zero_target_engine.py` | SBTi Net-Zero Standard target setting with pathway selection (ACA/SDA/FLAG) | Baseline, sector, ambition level, pathway type | TargetResult: near-term target, long-term target, annual reduction rates, coverage |
| 3 | `net_zero_gap_engine.py` | Gap-to-net-zero analysis showing current vs. required trajectory | Current emissions, targets, projections | GapResult: gap by scope/year, required reduction rate, risk assessment |
| 4 | `reduction_pathway_engine.py` | Quantified reduction pathway with abatement options, costs, and phasing | Emissions profile, available actions, budget constraints | PathwayResult: MACC curve, phased actions, cost-effectiveness ranking |
| 5 | `residual_emissions_engine.py` | Residual emissions calculation and neutralization requirements | Long-term target, achievable reductions, sector constraints | ResidualResult: residual budget, neutralization needs, removal options |
| 6 | `offset_portfolio_engine.py` | Carbon credit portfolio management with quality scoring | Credit inventory, retirement schedule, quality criteria | PortfolioResult: portfolio summary, quality scores, SBTi compliance |
| 7 | `net_zero_scorecard_engine.py` | Net-zero readiness and maturity assessment | All pack outputs, organizational data | ScorecardResult: overall score, dimension scores, recommendations |
| 8 | `net_zero_benchmark_engine.py` | Peer benchmarking against sector averages and leaders | Company metrics, sector, region | BenchmarkResult: percentile ranking, peer comparison, gap to leader |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `net_zero_onboarding_workflow.py` | 4: DataCollection -> BaselineCalc -> DataQuality -> BaselineReport | Guided onboarding to establish GHG baseline |
| 2 | `target_setting_workflow.py` | 4: SectorAnalysis -> PathwaySelection -> TargetDef -> Validation | SBTi-aligned target setting process |
| 3 | `reduction_planning_workflow.py` | 5: EmissionsProfile -> ActionIdentify -> CostAnalysis -> Prioritize -> RoadmapGen | Build quantified reduction roadmap |
| 4 | `offset_strategy_workflow.py` | 4: ResidualCalc -> CreditScreening -> PortfolioDesign -> ComplianceCheck | Design offset/neutralization strategy |
| 5 | `progress_review_workflow.py` | 4: DataUpdate -> ProgressCalc -> GapAnalysis -> ReportGen | Annual progress review cycle |
| 6 | `full_net_zero_assessment_workflow.py` | 6: Baseline -> Targets -> Reduction -> Offsets -> Scorecard -> Strategy | End-to-end net-zero assessment |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `net_zero_strategy_report.py` | MD, HTML, JSON | Executive-level net-zero strategy document |
| 2 | `ghg_baseline_report.py` | MD, HTML, JSON | Detailed GHG inventory baseline report |
| 3 | `target_validation_report.py` | MD, HTML, JSON | SBTi target validation and compliance report |
| 4 | `reduction_roadmap_report.py` | MD, HTML, JSON | Phased reduction roadmap with MACC |
| 5 | `offset_portfolio_report.py` | MD, HTML, JSON | Carbon credit portfolio and quality report |
| 6 | `net_zero_scorecard_report.py` | MD, HTML, JSON | Net-zero maturity scorecard |
| 7 | `progress_dashboard_report.py` | MD, HTML, JSON | Progress tracking dashboard |
| 8 | `benchmark_comparison_report.py` | MD, HTML, JSON | Peer benchmarking comparison |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 8-phase DAG pipeline with retry, provenance, conditional phases |
| 2 | `mrv_bridge.py` | Routes to all 30 MRV agents for emission calculations |
| 3 | `ghg_app_bridge.py` | Bridges to GL-GHG-APP for inventory management and base year |
| 4 | `sbti_app_bridge.py` | Bridges to GL-SBTi-APP for target setting and validation |
| 5 | `decarb_bridge.py` | Bridges to 21 DECARB-X agents for reduction planning |
| 6 | `offset_bridge.py` | Bridges to carbon credit/offset agents |
| 7 | `reporting_bridge.py` | Bridges to CDP, TCFD, ESRS E1 for cross-framework reporting |
| 8 | `health_check.py` | System health verification (18 categories) |
| 9 | `setup_wizard.py` | 6-step guided configuration wizard |
| 10 | `data_bridge.py` | Bridges to 20 DATA agents for data intake |

### 3.6 Presets

| # | Preset | Sector | Key Characteristics |
|---|--------|--------|-------------------|
| 1 | `manufacturing.yaml` | Manufacturing/Industrial | High Scope 1 (process), energy-intensive, SDA pathway |
| 2 | `services.yaml` | Professional/Financial Services | Low Scope 1, high Scope 3 (Cat 1, 6, 7), ACA pathway |
| 3 | `retail.yaml` | Retail/Consumer Goods | Mixed scopes, supply chain focus, spend-based Scope 3 |
| 4 | `energy.yaml` | Energy/Utilities | Very high Scope 1, SDA pathway, FLAG if applicable |
| 5 | `technology.yaml` | Technology/Software | Low Scope 1+2, high Scope 3 (Cat 1, 2, 11), ACA pathway |
| 6 | `sme_general.yaml` | SME (any sector) | Simplified, 4 engines only, spend-based Scope 3 |

---

## 4. Agent Dependencies

### 4.1 MRV Agents (30)

All 30 AGENT-MRV agents are available as dependencies via `mrv_bridge.py`:
- **Scope 1 (8):** MRV-001 through MRV-008 (Stationary Combustion, Refrigerants, Mobile Combustion, Process Emissions, Fugitive Emissions, Land Use, Waste Treatment, Agricultural)
- **Scope 2 (5):** MRV-009 through MRV-013 (Location-Based, Market-Based, Steam/Heat, Cooling, Dual Reporting)
- **Scope 3 (15):** MRV-014 through MRV-028 (Categories 1-15)
- **Cross-cutting (2):** MRV-029 (Scope 3 Category Mapper), MRV-030 (Audit Trail)

### 4.2 Decarbonization Agents (21)

All 21 DECARB-X agents via `decarb_bridge.py`:
- DECARB-X-001: Abatement Options Library (500+ options)
- DECARB-X-002: MACC Generator
- DECARB-X-003: Target Setting Agent
- DECARB-X-004: Pathway Scenario Builder
- DECARB-X-005: Investment Prioritization
- DECARB-X-006: Technology Readiness Assessor
- DECARB-X-007: Implementation Roadmap
- DECARB-X-008: Avoided Emissions Calculator
- DECARB-X-009: Carbon Intensity Tracker
- DECARB-X-010: Renewable Energy Planner
- DECARB-X-011: Electrification Planner
- DECARB-X-012: Fuel Switching Optimizer
- DECARB-X-013: Energy Efficiency Identifier
- DECARB-X-014: Carbon Capture Assessor
- DECARB-X-015: Offset Strategy Agent
- DECARB-X-016: Supplier Engagement Planner
- DECARB-X-017: Scope 3 Reduction Planner
- DECARB-X-018: Progress Monitoring Agent
- DECARB-X-019: Scenario Comparison Agent
- DECARB-X-020: Cost-Benefit Analyzer
- DECARB-X-021: Transition Risk Assessor

### 4.3 Application Dependencies

- GL-GHG-APP: GHG inventory management, base year, aggregation
- GL-SBTi-APP: Target setting, pathway calculation, temperature scoring
- GL-CDP-APP: CDP questionnaire alignment
- GL-TCFD-APP: TCFD metrics and targets alignment

### 4.4 Data Agents (20)

All 20 AGENT-DATA agents via `data_bridge.py` for data intake and quality.

### 4.5 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema, units, audit, etc.

---

## 5. Performance Targets

| Metric | Target |
|--------|--------|
| Full baseline calculation (Scope 1+2+3) | <30 minutes |
| Target setting and validation | <5 minutes |
| Reduction pathway generation (100 actions) | <10 minutes |
| Offset portfolio assessment | <5 minutes |
| Full net-zero assessment (end-to-end) | <60 minutes |
| Memory ceiling | 4096 MB |
| Cache hit target | 70% |
| Max facilities | 500 |
| Max suppliers (Scope 3) | 10,000 |

---

## 6. Security Requirements

- JWT RS256 authentication
- RBAC with net-zero-specific roles: `net_zero_manager`, `sustainability_analyst`, `finance_reviewer`, `executive_viewer`, `external_auditor`, `admin`
- AES-256-GCM encryption at rest for all emission data
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs
- Full audit trail per SEC-005

---

## 7. Database Migrations

Inherits platform migrations V001-V128. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V083-PACK021-001 | `nz_baselines` | Net-zero baseline records |
| V083-PACK021-002 | `nz_targets` | Net-zero targets (near-term, long-term, neutralization) |
| V083-PACK021-003 | `nz_reduction_actions` | Reduction actions with costs, timelines, abatement potential |
| V083-PACK021-004 | `nz_offset_portfolio` | Carbon credit portfolio entries |
| V083-PACK021-005 | `nz_progress_records` | Annual progress tracking records |
| V083-PACK021-006 | `nz_scorecards` | Net-zero maturity scorecard snapshots |

---

## 8. File Structure

```
packs/net-zero/PACK-021-net-zero-starter/
  __init__.py
  pack.yaml
  config/
    __init__.py
    pack_config.py
    demo/
      __init__.py
      demo_config.yaml
    presets/
      __init__.py
      manufacturing.yaml
      services.yaml
      retail.yaml
      energy.yaml
      technology.yaml
      sme_general.yaml
  engines/
    __init__.py
    net_zero_baseline_engine.py
    net_zero_target_engine.py
    net_zero_gap_engine.py
    reduction_pathway_engine.py
    residual_emissions_engine.py
    offset_portfolio_engine.py
    net_zero_scorecard_engine.py
    net_zero_benchmark_engine.py
  workflows/
    __init__.py
    net_zero_onboarding_workflow.py
    target_setting_workflow.py
    reduction_planning_workflow.py
    offset_strategy_workflow.py
    progress_review_workflow.py
    full_net_zero_assessment_workflow.py
  templates/
    __init__.py
    net_zero_strategy_report.py
    ghg_baseline_report.py
    target_validation_report.py
    reduction_roadmap_report.py
    offset_portfolio_report.py
    net_zero_scorecard_report.py
    progress_dashboard_report.py
    benchmark_comparison_report.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_bridge.py
    ghg_app_bridge.py
    sbti_app_bridge.py
    decarb_bridge.py
    offset_bridge.py
    reporting_bridge.py
    data_bridge.py
    health_check.py
    setup_wizard.py
  tests/
    __init__.py
    test_manifest.py
    test_config.py
    test_baseline_engine.py
    test_target_engine.py
    test_gap_engine.py
    test_reduction_engine.py
    test_residual_engine.py
    test_offset_engine.py
    test_scorecard_engine.py
    test_benchmark_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_demo.py
    test_e2e.py
    test_orchestrator.py
```

---

## 9. Testing Requirements

| Test Type | Coverage Target | Scope |
|-----------|-----------------|-------|
| Unit Tests | >90% line coverage | All 8 engines, all config models, all presets |
| Workflow Tests | >85% | All 6 workflows with synthetic data |
| Template Tests | 100% | All 8 templates in 3 formats (MD, HTML, JSON) |
| Integration Tests | >80% | All 10 integrations with mock agents |
| E2E Tests | Core happy path | Full pipeline from data intake to strategy report |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

---

## 10. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-18 |
| Phase 2 | Engine implementation (8 engines) | 2026-03-18 |
| Phase 3 | Workflow implementation (6 workflows) | 2026-03-18 |
| Phase 4 | Template implementation (8 templates) | 2026-03-18 |
| Phase 5 | Integration implementation (10 integrations) | 2026-03-18 |
| Phase 6 | Test suite (700+ tests) | 2026-03-18 |
| Phase 7 | Documentation & Release | 2026-03-18 |

---

## 11. Future Roadmap (PACK-022+)

- **PACK-022: Net Zero Professional Pack** -- Multi-entity consolidation, detailed scenario modeling, SDA sector pathways, VCMI claims alignment, assurance-ready outputs
- **PACK-023: Net Zero Enterprise Pack** -- Multi-subsidiary orchestration, M&A emissions integration, portfolio temperature scoring, board-level reporting suite
- **PACK-024: Net Zero Financial Institutions Pack** -- PCAF-aligned financed emissions, portfolio alignment, SBTi FI targets
- **PACK-025: Net Zero Real Estate Pack** -- CRREM alignment, building performance standards, Scope 3 Cat 13 deep dive

---

*Document Version: 1.0.0 | Last Updated: 2026-03-18 | Status: Approved*
