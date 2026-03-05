# PRD: APP-008 -- GL-TCFD-APP v1.0

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-008 |
| Application | GL-TCFD-APP v1.0 (Beta) |
| Priority | P1 (High) |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-03-02 |
| Standard | TCFD Recommendations (2017/2021), ISSB IFRS S2 Climate-related Disclosures (2023) |
| Base | 30 MRV Agents (1M+ lines), GL-GHG-APP v1.0, GL-ISO14064-APP v1.0, GL-CDP-APP v1.0, Climate Hazard Connector (AGENT-DATA-020) |
| Ralphy Task ID | APP-008 |

---

## 1. Overview

### 1.1 Purpose
Build GL-TCFD-APP v1.0 as a comprehensive TCFD-aligned climate disclosure and scenario analysis platform. This application implements the Task Force on Climate-related Financial Disclosures (TCFD) four-pillar framework with full scenario analysis capabilities, physical and transition risk assessment, financial impact quantification, and ISSB/IFRS S2 cross-walk for dual-standard compliance.

**Key differentiator**: While GL-GHG-APP focuses on emissions quantification and GL-CDP-APP on CDP questionnaire responses, GL-TCFD-APP focuses on **climate-related financial risk and opportunity assessment** with forward-looking scenario analysis -- the strategic, governance, and risk management dimensions of climate disclosure that investors and regulators demand.

**Regulatory Context**: TCFD recommendations (2017, updated 2021) were formally absorbed by the IFRS Foundation in 2023. ISSB IFRS S2 (effective Jan 2024) builds directly on TCFD's four-pillar structure. Over 100 jurisdictions now mandate TCFD-aligned reporting. This platform supports both TCFD and ISSB/IFRS S2 disclosure requirements.

Core capabilities:
1. **Governance Assessment** -- Board/management climate oversight per TCFD Pillar 1
2. **Strategy Analysis** -- Climate-related risks, opportunities, and business impact per TCFD Pillar 2
3. **Scenario Analysis** -- Quantitative climate scenarios (1.5C, 2C, 3C+) with financial modeling
4. **Physical Risk Assessment** -- Acute and chronic physical climate risks per asset/location
5. **Transition Risk Assessment** -- Policy, technology, market, and reputation risks
6. **Climate Opportunity Assessment** -- Resource efficiency, energy, products, markets, resilience
7. **Financial Impact Quantification** -- Income statement, balance sheet, cash flow impacts
8. **Risk Management Integration** -- Climate risk identification, assessment, and management per TCFD Pillar 3
9. **Metrics & Targets** -- GHG emissions, climate-related metrics, targets, and progress per TCFD Pillar 4
10. **TCFD Disclosure Generator** -- Structured disclosure aligned to all 11 recommended disclosures
11. **ISSB/IFRS S2 Cross-Walk** -- Automated mapping between TCFD and IFRS S2 requirements
12. **Gap Analysis Engine** -- Identifies disclosure gaps against TCFD and ISSB requirements
13. **Dashboard & Analytics** -- Executive risk/opportunity dashboard with scenario visualizations
14. **Multi-Format Export** -- PDF, Excel, JSON, CSV, XBRL (ISSB taxonomy)

### 1.2 TCFD Recommended Disclosures Mapping

| TCFD Pillar | Recommended Disclosure | GL-TCFD-APP Feature |
|-------------|------------------------|---------------------|
| **Governance** | a) Board oversight of climate-related risks/opportunities | Governance assessment engine, board structure tracker |
| **Governance** | b) Management's role in assessing/managing climate risks | Management roles & responsibilities tracker |
| **Strategy** | a) Climate-related risks/opportunities identified | Risk/opportunity registry with categorization |
| **Strategy** | b) Impact on business, strategy, and financial planning | Financial impact quantification engine |
| **Strategy** | c) Resilience of strategy under different scenarios | Scenario analysis engine (1.5C/2C/3C+) |
| **Risk Management** | a) Processes for identifying/assessing climate risks | Risk identification workflow engine |
| **Risk Management** | b) Processes for managing climate risks | Risk management and mitigation tracker |
| **Risk Management** | c) Integration into overall risk management | ERM integration mapper |
| **Metrics & Targets** | a) Metrics used to assess climate risks/opportunities | Climate metrics registry (40+ standard metrics) |
| **Metrics & Targets** | b) Scope 1, 2, 3 GHG emissions | MRV agent integration (30 agents) |
| **Metrics & Targets** | c) Targets and performance against targets | Target-setting engine with SBTi alignment |

### 1.3 ISSB IFRS S2 Requirements Mapping

| IFRS S2 Paragraph | Requirement | GL-TCFD-APP Feature |
|--------------------|-------------|---------------------|
| para 5-9 | Governance | Governance engine (extends TCFD Gov a/b) |
| para 10-22 | Strategy | Strategy + scenario analysis + financial impact |
| para 13-15 | Climate-related risks & opportunities | Risk/opportunity registry |
| para 16-21 | Strategy & decision-making | Business model impact tracker |
| para 22 | Climate resilience & scenario analysis | Multi-scenario engine (qualitative + quantitative) |
| para 23-24 | Risk Management | Risk ID, assessment, management processes |
| para 25-28 | Cross-industry metrics | 7 cross-industry metrics engine |
| para 29 | GHG emissions (Scope 1/2/3) | MRV agent integration |
| para 30-31 | Industry-based metrics | SASB/SICS industry metric mapper |
| para 32-33 | Targets | Target engine with monitoring |
| para B1-B65 | Industry-based guidance (68 industries) | Industry profile templates |

### 1.4 Scenario Analysis Framework

| Scenario | Source | Temperature | Timeframe | Key Assumptions |
|----------|--------|-------------|-----------|-----------------|
| Net Zero 2050 (NZE) | IEA WEO 2023 | 1.5C | 2030/2040/2050 | Rapid decarbonization, aggressive policy |
| Announced Pledges (APS) | IEA WEO 2023 | 1.7C | 2030/2040/2050 | Govts deliver on pledges |
| Stated Policies (STEPS) | IEA WEO 2023 | 2.5C | 2030/2040/2050 | Current policies continue |
| Current Policies | NGFS Phase IV | 3C+ | 2030/2050/2100 | No new policies, high physical risk |
| Delayed Transition | NGFS Phase IV | 2C | 2030/2050 | Late action, disorderly transition |
| Below 2C | NGFS Phase IV | <2C | 2030/2050 | Gradual strengthening |
| Divergent Net Zero | NGFS Phase IV | 1.5C | 2030/2050 | Varied sectoral effort |
| Custom | User-defined | Configurable | Configurable | User assumptions |

### 1.5 Physical Risk Categories

| Risk Type | Category | Examples | Data Source |
|-----------|----------|----------|-------------|
| **Acute** | Extreme weather | Cyclones, floods, wildfires, heatwaves | Climate Hazard Connector (AGENT-DATA-020) |
| **Acute** | Precipitation extremes | Flash floods, droughts | CMIP6 / CORDEX downscaled |
| **Chronic** | Temperature rise | Increased mean temperature, cooling demand | IPCC AR6 WG1 projections |
| **Chronic** | Sea level rise | Coastal inundation, storm surge amplification | NASA/NOAA sea level data |
| **Chronic** | Water stress | Water scarcity, drought frequency | WRI Aqueduct |
| **Chronic** | Ecosystem degradation | Biodiversity loss, soil degradation | IUCN Red List, FAO |

### 1.6 Transition Risk Categories

| Risk Type | Sub-Category | Examples | Financial Impact |
|-----------|--------------|----------|-----------------|
| **Policy & Legal** | Carbon pricing | ETS, carbon tax escalation | Operating cost increase |
| **Policy & Legal** | Mandates & regulation | Efficiency standards, phase-outs | Capex/compliance costs |
| **Policy & Legal** | Litigation risk | Climate lawsuits, greenwashing claims | Legal liability, fines |
| **Technology** | Substitution | EVs replacing ICE, renewables replacing fossil | Asset stranding |
| **Technology** | Disruption | Energy storage, green hydrogen, CCUS | Market share shift |
| **Technology** | Investment | R&D for low-carbon technologies | Capex requirements |
| **Market** | Demand shifts | Consumer preferences, commodity prices | Revenue impact |
| **Market** | Supply chain | Resource scarcity, supplier disruption | Input cost volatility |
| **Reputation** | Stakeholder perception | Investor sentiment, brand value, talent | Market cap, cost of capital |

### 1.7 Climate Opportunity Categories

| Category | Examples | Financial Impact |
|----------|----------|-----------------|
| **Resource Efficiency** | Energy efficiency, water conservation, waste reduction | Cost savings, margin improvement |
| **Energy Source** | Renewables, distributed generation, PPAs | Lower energy costs, hedging |
| **Products & Services** | Low-carbon products, climate solutions, R&D | Revenue growth, market expansion |
| **Markets** | New geographies, public sector contracts, green bonds | Revenue diversification |
| **Resilience** | Adaptive capacity, supply chain flexibility, insurance | Risk reduction, continuity |

### 1.8 Technical Context
- **Backend**: FastAPI + Python 3.11+ + Pydantic v2
- **Frontend**: React 18 + TypeScript + Material-UI 5 + Redux Toolkit + Recharts + Vite
- **MRV Agents**: 30 production-ready agents at `greenlang/` with individual FastAPI routers
- **Support Agents**: Climate Hazard Connector (DATA-020), Scope 3 Category Mapper, Audit Trail & Lineage
- **Database**: PostgreSQL + TimescaleDB (V001-V085 already deployed)
- **Existing Apps**: GL-GHG-APP v1.0, GL-ISO14064-APP v1.0, GL-CDP-APP v1.0

---

## 2. Application Components

### 2.1 Backend: Governance Assessment Engine
- Board climate oversight structure tracking per TCFD Governance(a)
- Board committee mapping (risk, sustainability, audit, ESG)
- Board climate competency assessment (skills matrix)
- Meeting frequency and climate agenda tracking
- Management roles and responsibilities per TCFD Governance(b)
- C-suite climate accountability mapping (CEO, CFO, CSO, CRO)
- Internal reporting lines and escalation paths
- Management incentive/compensation link to climate
- Cross-functional climate committee tracking
- Governance maturity scoring (1-5 scale per dimension)
- Year-over-year governance improvement tracking
- ISSB IFRS S2 para 5-9 compliance mapping

### 2.2 Backend: Strategy Analysis Engine
- Climate-related risk identification and categorization
- Climate-related opportunity identification and sizing
- Short/medium/long-term time horizon classification
- Business model impact assessment (revenue, costs, assets, liabilities)
- Value chain impact mapping (upstream, operations, downstream)
- Strategic response tracking (mitigation, adaptation, transfer, accept)
- Business resilience assessment under each scenario
- Sector-specific strategy templates (11 TCFD sector guides)
- Strategy evolution tracking over reporting periods
- ISSB IFRS S2 para 10-22 compliance mapping

### 2.3 Backend: Scenario Analysis Engine
- Pre-built scenario pathways: IEA NZE/APS/STEPS, NGFS Phase IV (6 scenarios)
- Custom scenario builder with user-defined parameters
- Scenario parameter library:
  - Carbon price trajectories ($/tCO2e by year)
  - Energy mix projections (% renewable by year)
  - Technology adoption curves (S-curves)
  - Regulatory stringency index
  - Physical climate variables (temperature, precipitation, sea level)
- Time horizon support: 2030, 2040, 2050, 2100
- Quantitative scenario outputs:
  - Revenue impact (% change by scenario/year)
  - Cost impact (carbon cost, compliance, adaptation)
  - Asset valuation impact (stranding, impairment)
  - Capital expenditure requirements
- Qualitative scenario narratives
- Scenario comparison matrix (side-by-side)
- Sensitivity analysis (key parameter variation)
- Probability-weighted expected impact
- ISSB IFRS S2 para 22 climate resilience assessment

### 2.4 Backend: Physical Risk Assessment Engine
- Asset-level physical risk scoring (location-based)
- Geocoded asset registry with climate hazard overlay
- Acute risk assessment: cyclone, flood, wildfire, heatwave, drought
- Chronic risk assessment: temperature rise, sea level, water stress, precipitation change
- Integration with Climate Hazard Connector (AGENT-DATA-020)
- RCP/SSP scenario alignment (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5)
- Hazard exposure scoring per asset (1-5 scale)
- Vulnerability assessment (building type, elevation, infrastructure)
- Adaptive capacity scoring
- Composite physical risk index: exposure x vulnerability / adaptive capacity
- Financial damage estimation (% asset value at risk)
- Insurance cost impact modeling
- Supply chain physical risk propagation
- Portfolio-level aggregation

### 2.5 Backend: Transition Risk Assessment Engine
- Policy risk tracker:
  - Carbon pricing trajectory modeling (current + projected)
  - Regulatory compliance cost estimation
  - Phase-out timeline tracking (coal, ICE vehicles, F-gases)
  - Litigation risk scoring
- Technology risk engine:
  - Technology disruption indicators
  - Asset stranding probability by technology/sector
  - R&D investment gap analysis
  - Competitor technology benchmarking
- Market risk engine:
  - Demand elasticity modeling for product portfolio
  - Commodity price sensitivity (fossil fuels, metals, agri)
  - Supply chain cost impact propagation
- Reputation risk engine:
  - ESG rating trajectory
  - Stakeholder sentiment scoring
  - Greenwashing risk assessment
- Composite transition risk index (weighted across 4 sub-types)
- Sector-specific transition risk profiles (11 TCFD sectors)
- Transition risk maturity scoring

### 2.6 Backend: Climate Opportunity Assessment Engine
- Opportunity identification framework (5 categories)
- Revenue opportunity sizing per category
- Cost savings estimation (energy efficiency, waste reduction, water)
- New market assessment (TAM/SAM/SOM for climate solutions)
- Green product/service pipeline tracker
- Renewable energy opportunity (PPAs, on-site, community solar)
- Climate adaptation market opportunities
- Green financing opportunities (green bonds, sustainability-linked loans)
- Opportunity realization timeline and milestones
- Investment requirement and ROI modeling
- Opportunity prioritization matrix (impact vs. feasibility)

### 2.7 Backend: Financial Impact Quantification Engine
- Three financial statement impact layers:
  - **Income Statement**: Revenue impact, COGS/opex impact, carbon costs, insurance premium changes
  - **Balance Sheet**: Asset impairment, stranding, right-of-use adjustments, provisions
  - **Cash Flow**: Capex requirements, adaptation costs, carbon tax payments, green revenue
- NPV/IRR calculation for climate investments
- Marginal abatement cost curve (MACC) generation
- Carbon price sensitivity analysis ($25-$200/tCO2e)
- Sector-specific financial impact templates
- Multi-currency support (20+ currencies)
- Discounted cash flow under each scenario
- Monte Carlo uncertainty on financial estimates
- TCFD recommended financial impact categories mapping
- ISSB IFRS S2 para 16-21 financial effects

### 2.8 Backend: Risk Management Integration Engine
- Climate risk identification process tracker per TCFD RM(a)
- Risk assessment methodology (likelihood x impact matrix)
- Risk materiality threshold configuration
- Risk prioritization with heat map scoring
- Climate risk management processes per TCFD RM(b):
  - Mitigation actions and owners
  - Adaptation measures
  - Risk transfer (insurance, hedging)
  - Risk acceptance with justification
- ERM integration mapper per TCFD RM(c)
- Risk register with climate-specific fields
- Risk review cycle management (quarterly/annual)
- Risk appetite statement tracking
- Risk indicator monitoring (leading and lagging)
- Escalation and reporting workflows

### 2.9 Backend: Metrics & Targets Engine
- **Cross-industry metrics** per ISSB IFRS S2 para 25-28:
  1. GHG emissions (Scope 1, 2, 3) via MRV agents
  2. Transition risks: amount/percentage of assets/activities vulnerable
  3. Physical risks: amount/percentage of assets/activities vulnerable
  4. Climate-related opportunities: amount/percentage of revenue
  5. Capital deployment: climate-related capex/opex
  6. Internal carbon price applied
  7. Remuneration: % management incentives linked to climate
- **Industry-specific metrics** per SASB/SICS standards (68 industries)
- **Scope 1/2/3 emissions integration** with all 30 MRV agents
- Emissions intensity metrics (per revenue, per employee, per unit)
- Target types:
  - Absolute reduction targets
  - Intensity reduction targets
  - Net zero targets with interim milestones
  - SBTi-aligned targets (1.5C/WB2C)
  - Renewable energy targets (RE100)
- Target progress tracking with gap-to-target analysis
- Historical performance trend analysis
- Peer benchmarking (sector averages)
- Target ambition assessment (alignment to Paris Agreement)

### 2.10 Backend: TCFD Disclosure Generator
- Structured disclosure for all 11 recommended disclosures
- Section-by-section drafting with AI-assisted narrative
- Compliance completeness checker (% coverage per disclosure)
- Cross-reference linker (disclosure -> evidence -> data source)
- Multi-year disclosure comparison
- Version control and approval workflow
- Export formats: PDF, Word, HTML, JSON
- Regulatory adaptation layers:
  - UK mandatory TCFD (FCA/Companies Act)
  - EU CSRD (ESRS E1 climate)
  - US SEC climate rules
  - Japan FSA climate disclosure
  - Singapore SGX climate reporting
  - Hong Kong HKEX ESG
  - Australia ASRS climate standards
  - New Zealand XRB climate standards

### 2.11 Backend: ISSB/IFRS S2 Cross-Walk Engine
- Automated mapping between TCFD 11 disclosures and IFRS S2 paragraphs
- Gap identification between TCFD and IFRS S2 requirements
- IFRS S2 additional requirements tracker:
  - Industry-based metrics (SASB-derived)
  - Transition plans
  - Carbon credits/offsets
  - Financial effects (current period + anticipated)
  - Connected reporting (IFRS S1 general requirements)
- Dual-standard compliance scoring
- Migration pathway from TCFD-only to IFRS S2

### 2.12 Backend: Gap Analysis Engine
- Disclosure maturity assessment per pillar (1-5 scale)
- Requirement-by-requirement gap identification
- Peer comparison benchmarking
- Action plan generation for gap closure
- Priority scoring (regulatory urgency x effort)
- Timeline estimation for full compliance
- Resource requirement estimation
- Progress tracking over assessment cycles

### 2.13 Backend: Recommendation Engine
- AI-driven improvement recommendations based on:
  - Current disclosure gaps
  - Sector best practices
  - Peer benchmarking
  - Regulatory trajectory
- Prioritized action list with estimated impact
- Implementation guidance per recommendation
- Cost-benefit summary

### 2.14 Frontend: Executive Dashboard
- Executive KPI cards (Total risk exposure, Opportunity value, Disclosure maturity, Scenario impact range)
- Risk/Opportunity balance chart (stacked bar: physical risk + transition risk vs. opportunities)
- Scenario impact comparison chart (grouped bar: NZE/APS/STEPS/Current across financial metrics)
- Physical risk heatmap (geographic map with asset risk overlay)
- Transition risk radar chart (policy/tech/market/reputation by sector)
- Metrics & Targets progress gauge charts
- Disclosure completeness donut (11 disclosures)
- Year-over-year trend lines
- ISSB cross-industry metrics summary cards (7 metrics)

### 2.15 Frontend: Governance Page
- Board oversight structure visualization (org chart)
- Climate committee composition and meeting tracker
- Management roles and responsibilities matrix
- Governance maturity scorecard (spider/radar chart)
- Climate competency skills matrix editor
- Incentive linkage tracker
- Governance disclosure drafting panel

### 2.16 Frontend: Strategy Pages
- Risk/Opportunity registry with filters (type, time horizon, impact level)
- Risk detail drawer (description, category, financial impact, response)
- Opportunity detail drawer (description, sizing, timeline, investment)
- Business model impact visualization (Sankey diagram: revenue/cost flows)
- Value chain impact map (upstream/operations/downstream)

### 2.17 Frontend: Scenario Analysis Interactive UI
- Scenario selector with side-by-side comparison (up to 4 scenarios)
- Parameter adjustment sliders (carbon price, energy mix, temperature)
- Financial impact waterfall chart (base -> scenario adjustments -> net impact)
- Asset stranding timeline chart (% assets at risk by year)
- Revenue sensitivity tornado chart
- Custom scenario builder wizard
- Scenario narrative editor
- Export scenario analysis package

### 2.18 Frontend: Physical Risk Page
- Interactive risk map (Mapbox/Leaflet with asset pins, hazard layers)
- Asset-level risk cards (exposure, vulnerability, adaptive capacity)
- Hazard distribution chart per asset type
- Insurance cost projection chart
- Supply chain risk propagation diagram
- Physical risk trend under different RCP/SSP scenarios

### 2.19 Frontend: Transition Risk Page
- Policy risk timeline (regulation milestones, carbon price trajectory)
- Technology disruption S-curve charts
- Market demand shift projections
- Reputation risk dashboard (ESG ratings, sentiment)
- Composite risk heatmap (sectors x risk types)
- Stranded asset analysis panel

### 2.20 Frontend: Opportunity Page
- Opportunity pipeline kanban board (identified -> assessed -> pursuing -> realized)
- Revenue opportunity sizing bar chart
- Cost savings waterfall chart
- Green product portfolio tracker
- Investment/ROI analysis table
- Opportunity prioritization matrix scatter plot

### 2.21 Frontend: Financial Impact Page
- Three-panel view: Income Statement | Balance Sheet | Cash Flow
- Scenario-linked financial projections table
- MACC (Marginal Abatement Cost Curve) interactive chart
- NPV/IRR analysis for climate investments
- Carbon price sensitivity analysis chart
- Monte Carlo distribution of financial outcomes

### 2.22 Frontend: Risk Management Page
- Risk register data grid (sortable, filterable, inline edit)
- Likelihood x Impact heat matrix
- Risk response tracker (mitigate/adapt/transfer/accept)
- ERM integration status dashboard
- Risk indicator monitoring charts
- Escalation workflow timeline

### 2.23 Frontend: Metrics & Targets Page
- GHG emissions summary (Scope 1/2/3 from MRV agents)
- Emissions intensity trend chart
- Target progress gauge charts (absolute, intensity, net zero)
- SBTi alignment assessment panel
- Cross-industry metrics dashboard (7 ISSB metrics)
- Industry-specific metrics table
- Peer benchmarking comparison chart

### 2.24 Frontend: Disclosure Builder Page
- 11-disclosure checklist with completion status
- Section-by-section disclosure editor with rich text
- Evidence linking panel (attach data sources, charts, calculations)
- Compliance checker sidebar (TCFD + ISSB requirements)
- Preview panel (formatted disclosure report)
- Export dialog (PDF, Word, JSON, XBRL)
- Version history timeline

### 2.25 Frontend: Gap Analysis Page
- Maturity assessment spider chart (4 pillars)
- Requirement-level gap table (requirement, status, gap, action)
- Action plan timeline (Gantt-style)
- Peer benchmarking bar chart
- Progress tracking over time

### 2.26 Frontend: ISSB Cross-Walk Page
- Side-by-side mapping table (TCFD disclosure -> IFRS S2 paragraph)
- Dual-standard compliance scorecard
- Gap identification highlights
- Migration pathway checklist

---

## 3. File Structure

```
applications/GL-TCFD-APP/TCFD-Disclosure-Platform/
    config/
        tcfd_config.yaml                          (~350 lines)
    services/
        __init__.py                               (~100 lines)
        config.py                                 (~600 lines)
        models.py                                 (~1,400 lines)
        governance_engine.py                      (~700 lines)
        strategy_engine.py                        (~800 lines)
        scenario_analysis_engine.py               (~1,200 lines)
        physical_risk_engine.py                   (~900 lines)
        transition_risk_engine.py                 (~900 lines)
        opportunity_engine.py                     (~700 lines)
        financial_impact_engine.py                (~1,000 lines)
        risk_management_engine.py                 (~700 lines)
        metrics_targets_engine.py                 (~800 lines)
        disclosure_generator.py                   (~900 lines)
        issb_crosswalk_engine.py                  (~600 lines)
        gap_analysis_engine.py                    (~600 lines)
        recommendation_engine.py                  (~500 lines)
        data_quality_engine.py                    (~500 lines)
        setup.py                                  (~500 lines)
        api/
            __init__.py                           (~50 lines)
            governance_routes.py                  (~500 lines)
            strategy_routes.py                    (~550 lines)
            scenario_routes.py                    (~600 lines)
            physical_risk_routes.py               (~500 lines)
            transition_risk_routes.py             (~500 lines)
            opportunity_routes.py                 (~450 lines)
            financial_routes.py                   (~550 lines)
            risk_management_routes.py             (~500 lines)
            metrics_routes.py                     (~500 lines)
            disclosure_routes.py                  (~550 lines)
            dashboard_routes.py                   (~400 lines)
            gap_routes.py                         (~400 lines)
            issb_routes.py                        (~400 lines)
            settings_routes.py                    (~300 lines)
    frontend/
        package.json, tsconfig.json, vite.config.ts, index.html
        src/
            main.tsx, App.tsx
            types/index.ts                        (~700 lines)
            services/api.ts                       (~800 lines)
            store/ (index, hooks, 14 slices)
            components/
                layout/ (Sidebar, Header, Layout)
                common/ (StatCard, StatusBadge, DataTable, LoadingSpinner, RiskBadge)
                dashboard/ (RiskOpportunityChart, ScenarioComparison, PhysicalRiskMap, TransitionRadar, DisclosureDonut, MetricsSummary, TrendChart)
                governance/ (BoardStructure, CommitteeTracker, RolesMatrix, MaturityScorecard, CompetencyMatrix)
                strategy/ (RiskRegistry, OpportunityRegistry, BusinessModelImpact, ValueChainMap)
                scenario/ (ScenarioSelector, ParameterSliders, ImpactWaterfall, StrandingTimeline, SensitivityTornado, ScenarioBuilder)
                physical_risk/ (RiskMap, AssetRiskCards, HazardChart, InsuranceCost, SupplyChainRisk)
                transition_risk/ (PolicyTimeline, TechDisruption, MarketShift, ReputationDashboard, StrandedAssets)
                opportunity/ (OpportunityPipeline, RevenueSizing, CostSavings, InvestmentROI, PriorityMatrix)
                financial/ (IncomeStatement, BalanceSheet, CashFlow, MACCChart, NPVAnalysis, CarbonSensitivity, MonteCarloChart)
                risk_mgmt/ (RiskRegister, HeatMatrix, ResponseTracker, ERMIntegration, IndicatorCharts)
                metrics/ (EmissionsSummary, IntensityTrend, TargetProgress, SBTiAlignment, IndustryMetrics, PeerBenchmark)
                disclosure/ (DisclosureChecklist, SectionEditor, EvidencePanel, ComplianceChecker, PreviewPanel, ExportDialog)
                gap_analysis/ (MaturitySpider, GapTable, ActionTimeline, PeerComparison)
                crosswalk/ (MappingTable, DualScorecard, MigrationChecklist)
            pages/
                Dashboard.tsx
                Governance.tsx
                StrategyRisks.tsx
                StrategyOpportunities.tsx
                ScenarioAnalysis.tsx
                PhysicalRisk.tsx
                TransitionRisk.tsx
                Opportunities.tsx
                FinancialImpact.tsx
                RiskManagement.tsx
                MetricsTargets.tsx
                DisclosureBuilder.tsx
                GapAnalysis.tsx
                ISSBCrossWalk.tsx
                Settings.tsx
            utils/ (formatters, validators, scenarioHelpers)
    tests/
        __init__.py
        test_models.py
        test_governance_engine.py
        test_strategy_engine.py
        test_scenario_analysis_engine.py
        test_physical_risk_engine.py
        test_transition_risk_engine.py
        test_opportunity_engine.py
        test_financial_impact_engine.py
        test_risk_management_engine.py
        test_metrics_targets_engine.py
        test_disclosure_generator.py
        test_issb_crosswalk_engine.py
        test_gap_analysis_engine.py
        test_recommendation_engine.py
        test_data_quality_engine.py
        test_api_routes.py
```

---

## 4. Database Migration: V086

```sql
-- V086__tcfd_app_service.sql
-- Tables: ~22 tables + 3 hypertables + 2 continuous aggregates

-- Core tables:
-- gl_tcfd_organizations               -- Organization profiles for TCFD disclosure
-- gl_tcfd_governance_assessments      -- Board/mgmt climate oversight assessments
-- gl_tcfd_governance_roles            -- Governance roles and responsibilities
-- gl_tcfd_climate_risks               -- Climate risk registry (physical + transition)
-- gl_tcfd_climate_opportunities       -- Climate opportunity registry
-- gl_tcfd_scenarios                   -- Scenario definitions and parameters
-- gl_tcfd_scenario_results            -- Scenario analysis quantitative results
-- gl_tcfd_scenario_parameters         -- Scenario parameter sets (carbon price, energy mix, etc.)
-- gl_tcfd_physical_risk_assessments   -- Asset-level physical risk scores
-- gl_tcfd_asset_locations             -- Geocoded asset registry
-- gl_tcfd_transition_risk_assessments -- Transition risk analysis records
-- gl_tcfd_financial_impacts           -- Financial impact quantification
-- gl_tcfd_risk_management_records     -- Risk management process records
-- gl_tcfd_risk_responses              -- Risk mitigation/adaptation actions
-- gl_tcfd_metrics                     -- Climate metric definitions and values
-- gl_tcfd_targets                     -- Climate targets and progress
-- gl_tcfd_target_progress             -- Target progress tracking (time series)
-- gl_tcfd_disclosures                 -- TCFD disclosure documents
-- gl_tcfd_disclosure_sections         -- Individual disclosure sections (11 recommended)
-- gl_tcfd_gap_assessments             -- Gap analysis results
-- gl_tcfd_issb_mappings              -- ISSB/IFRS S2 crosswalk mappings
-- gl_tcfd_recommendations             -- AI-generated improvement recommendations

-- Hypertables: gl_tcfd_scenario_results, gl_tcfd_target_progress, gl_tcfd_financial_impacts
-- Continuous aggregates: quarterly risk scores, annual scenario summaries
```

---

## 5. Development Tasks (Parallel Build Plan)

### Task Group A: Backend Core Engines (Agent 1 - gl-backend-developer)
- A1: config.py (enums: RiskType, OpportunityCategory, ScenarioType, TimeHorizon, DisclosurePillar, AssetType, etc. + settings)
- A2: models.py (50+ Pydantic domain models for all entities)
- A3: governance_engine.py (board tracking, management roles, maturity scoring)
- A4: strategy_engine.py (risk/opportunity identification, business model impact, value chain)
- A5: scenario_analysis_engine.py (8 pre-built scenarios, custom builder, parameter library, quantitative outputs)
- A6: physical_risk_engine.py (asset-level scoring, hazard overlay, RCP/SSP alignment, financial damage)
- A7: transition_risk_engine.py (policy/tech/market/reputation, composite scoring, sector profiles)
- A8: opportunity_engine.py (5 categories, sizing, pipeline, prioritization)
- A9: financial_impact_engine.py (3 statement impacts, NPV/IRR, MACC, Monte Carlo)
- A10: risk_management_engine.py (identification, assessment, management, ERM integration)
- A11: metrics_targets_engine.py (7 cross-industry metrics, industry-specific, MRV integration, SBTi)
- A12: disclosure_generator.py (11 disclosures, compliance checker, multi-format export)
- A13: issb_crosswalk_engine.py (TCFD->IFRS S2 mapping, gap ID, migration pathway)
- A14: gap_analysis_engine.py (maturity assessment, peer benchmarking, action planning)
- A15: recommendation_engine.py (AI-driven improvement suggestions)
- A16: data_quality_engine.py (disclosure data quality scoring)
- A17: setup.py + __init__.py + tcfd_config.yaml

### Task Group B: Backend API Layer (Agent 2 - gl-api-developer)
- B1: api/__init__.py, governance_routes.py (8+ endpoints)
- B2: strategy_routes.py (10+ endpoints for risks and opportunities)
- B3: scenario_routes.py (10+ endpoints for scenario CRUD and analysis)
- B4: physical_risk_routes.py, transition_risk_routes.py (8+ endpoints each)
- B5: opportunity_routes.py, financial_routes.py (8+ endpoints each)
- B6: risk_management_routes.py, metrics_routes.py (8+ endpoints each)
- B7: disclosure_routes.py, dashboard_routes.py (8+ endpoints each)
- B8: gap_routes.py, issb_routes.py, settings_routes.py (5+ endpoints each)

### Task Group C: Frontend Core + Layout + Store (Agent 3 - gl-frontend-developer)
- C1: Config files (package.json, tsconfig.json, vite.config.ts, index.html, main.tsx)
- C2: types/index.ts (all TypeScript interfaces)
- C3: services/api.ts (API client with all endpoints)
- C4: Redux store (14 slices: governance, strategy, scenarios, physicalRisk, transitionRisk, opportunities, financial, riskMgmt, metrics, disclosure, gap, issb, dashboard, settings)
- C5: Layout + common components
- C6: App.tsx with routing (15 pages)

### Task Group D: Frontend Pages + Domain Components (Agent 4 - gl-frontend-developer)
- D1: Dashboard components (7) + Dashboard page
- D2: Governance components (5) + Governance page
- D3: Strategy components (4) + StrategyRisks/Opportunities pages
- D4: Scenario components (6) + ScenarioAnalysis page
- D5: Physical/Transition risk components (10) + pages
- D6: Opportunity + Financial components (10) + pages
- D7: Risk management + Metrics components (11) + pages
- D8: Disclosure + Gap + Crosswalk components (10) + pages
- D9: Settings page + utils

### Task Group E: Tests + DB Migration (Agent 5 - gl-test-engineer)
- E1: 16 backend test files (400+ tests)
- E2: V086 database migration SQL

---

## 6. Acceptance Criteria

1. Governance assessment with board/management tracking and maturity scoring (TCFD Gov a/b)
2. Strategy engine with risk/opportunity identification and business model impact (TCFD Strategy a/b)
3. Scenario analysis with 8 pre-built + custom scenarios and quantitative outputs (TCFD Strategy c)
4. Physical risk assessment with asset-level scoring and geographic overlay (5 hazard types)
5. Transition risk assessment with 4 sub-types and composite scoring (TCFD Risk categories)
6. Climate opportunity assessment with 5 categories and financial sizing
7. Financial impact quantification across 3 financial statements
8. Risk management integration with ERM mapping (TCFD RM a/b/c)
9. Metrics & Targets with MRV agent integration and SBTi alignment (TCFD M&T a/b/c)
10. TCFD disclosure generator with all 11 recommended disclosures
11. ISSB/IFRS S2 cross-walk with gap identification
12. Gap analysis with maturity scoring and action planning
13. Full React dashboard with 15 pages and interactive scenario UI
14. 65+ REST API endpoints across 14 route files
15. 400+ backend tests across 16 test files
16. V086 database migration with 22 tables + TimescaleDB hypertables
17. Auth integration with TCFD-specific permissions
