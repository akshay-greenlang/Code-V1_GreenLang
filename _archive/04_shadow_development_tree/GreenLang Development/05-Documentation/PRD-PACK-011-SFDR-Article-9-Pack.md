# PRD-PACK-011: SFDR Article 9 Pack

## Status: APPROVED
## Priority: HIGH
## Category: EU Compliance - Solution Packs
## Created: 2026-03-15
## Last Updated: 2026-03-15

---

## 1. Executive Summary

PACK-011 delivers a production-grade GreenLang Solution Pack for **SFDR Article 9 ("dark green")** financial products. Article 9 products have **sustainable investment as their objective** under Regulation (EU) 2019/2088, requiring that substantially all investments (excluding cash, hedging, and liquidity instruments) qualify as sustainable investments per Article 2(17).

This pack is the strict counterpart to PACK-010 (Article 8). While Article 8 products merely "promote" environmental or social characteristics, Article 9 products must demonstrate that every investment:
1. **Contributes** to an environmental or social objective (measurable)
2. Does **Not Significantly Harm (DNSH)** any other sustainability objective
3. Follows **good governance practices** (Article 2(17))

The pack includes 8 calculation engines, 8 compliance workflows, 8 report templates, and 10 integration bridges, covering Annex III pre-contractual disclosures, Annex V periodic reporting, mandatory PAI consideration, EU Climate Benchmark alignment, impact measurement, and carbon reduction trajectory tracking for Article 9(3) products.

### Key Differentiators from PACK-010 (Article 8)

| Aspect | Article 8 (PACK-010) | Article 9 (PACK-011) |
|--------|----------------------|----------------------|
| Investment objective | Promotes E/S characteristics | Sustainable investment as objective |
| Sustainable investment % | Optional minimum | ~100% (excl. cash/hedging) |
| PAI consideration | Optional (comply-or-explain) | Mandatory for all investments |
| DNSH assessment | On sustainable portion only | On ALL investments |
| Good governance | On sustainable portion only | On ALL investments |
| Pre-contractual template | Annex II RTS | Annex III RTS |
| Periodic reporting template | Annex IV RTS | Annex V RTS |
| EU Taxonomy disclosure | Required if environmental | Required - specific Art 5/6 reference |
| Benchmark requirement | Optional | Required for Art 9(3) carbon reduction |
| Impact measurement | Not required | Expected - measurable outcomes |
| Downgrade risk | N/A | Downgrade to Art 8 if criteria not met |

---

## 2. Regulatory Context

### 2.1 Primary Legislation
- **SFDR Level 1**: Regulation (EU) 2019/2088, Articles 2(17), 5, 9
- **SFDR RTS**: Delegated Regulation (EU) 2022/1288, Annexes III (pre-contractual) and V (periodic)
- **Taxonomy Regulation**: Regulation (EU) 2020/852, Articles 5 and 6
- **Taxonomy Disclosures DA**: Delegated Regulation (EU) 2021/2178
- **ESMA Q&A**: Supervisory guidance on Article 9 classification
- **EU Climate Benchmarks Regulation**: Regulation (EU) 2019/2089 (CTB and PAB)
- **EET Standard**: European ESG Template v1.1 (FinDatEx)

### 2.2 Article 9 Sub-Types
1. **Article 9(1)**: General sustainable investment objective
2. **Article 9(2)**: Sustainable investment objective with designated index/benchmark
3. **Article 9(3)**: Carbon emission reduction objective (must use EU Climate Benchmark - CTB or PAB)

### 2.3 Key Regulatory Requirements

**Article 2(17) Sustainable Investment Definition:**
An investment in an economic activity that:
- Contributes to an environmental objective (measured by key resource efficiency indicators: energy use, renewable energy, raw materials, water, land use, waste, GHG, biodiversity, circular economy) OR contributes to a social objective (tackling inequality, social cohesion, social integration, labour relations, human capital, economically disadvantaged communities)
- Does not significantly harm any of those objectives
- The investee companies follow good governance practices (sound management structures, employee relations, remuneration of staff, tax compliance)

**100% Sustainable Investment Requirement:**
- All investments must qualify as sustainable investments per Article 2(17)
- Exceptions: cash and cash equivalents, money market instruments for liquidity management, derivatives for hedging purposes
- ESMA guidance: minimum sustainable investment proportion should be "very high" (market practice: 80-100%)
- Pre-contractual must commit to minimum % (binding once stated)

**Mandatory PAI Consideration:**
- Article 9 products MUST consider PAI indicators (not optional)
- Must disclose how PAI indicators are considered in investment decisions
- Must report quantitative PAI values in periodic disclosures

---

## 3. Goals and Success Criteria

### 3.1 Functional Goals
1. Full Article 9 product lifecycle management (classification, disclosure, monitoring, reporting)
2. 100% sustainable investment verification with DNSH + good governance for every holding
3. Mandatory PAI indicator calculation and reporting (all 18 + selected optional)
4. EU Climate Benchmark alignment (CTB and PAB) for Article 9(3) products
5. Annex III pre-contractual and Annex V periodic disclosure generation
6. Impact measurement and KPI tracking
7. Carbon reduction trajectory calculation and monitoring
8. Downgrade risk monitoring (Article 9 to Article 8 triggers)

### 3.2 Technical Goals
1. Self-contained engines with zero-hallucination deterministic calculations
2. Full Pydantic v2 type safety across all models
3. SHA-256 provenance hashing on all engine outputs
4. Cross-pack integration with PACK-010 (Article 8) and PACK-008 (EU Taxonomy)
5. 350+ unit tests with 100% pass rate
6. Cross-pack integration test coverage

### 3.3 Acceptance Criteria
- All 8 engines produce correct, auditable results
- All 8 workflows execute successfully with proper phase management
- All 8 templates render in Markdown, HTML, and JSON formats
- All 10 integrations connect properly
- Unit test pass rate: 100%
- Cross-pack tests: PASS

---

## 4. Architecture

### 4.1 Directory Structure
```
packs/eu-compliance/PACK-011-sfdr-article-9/
  pack.yaml                          # Pack manifest
  __init__.py
  config/
    __init__.py
    pack_config.py                   # SFDRArticle9Config (Pydantic v2)
    presets/
      impact_fund.yaml               # Pure impact investment fund
      climate_fund.yaml              # Climate-focused Article 9(3)
      social_fund.yaml               # Social objective fund
      esg_leader_fund.yaml           # Best-in-class ESG fund
      transition_fund.yaml           # Transition financing fund
    demo/
      demo_config.yaml               # Quick-start demo configuration
  engines/                           # 8 calculation engines
    __init__.py
    sustainable_objective_engine.py   # Engine 1: Sustainable objective verification
    enhanced_dnsh_engine.py           # Engine 2: Enhanced DNSH (stricter than Art 8)
    full_taxonomy_alignment.py        # Engine 3: Full taxonomy alignment (Art 5/6)
    impact_measurement_engine.py      # Engine 4: Impact measurement and KPIs
    benchmark_alignment_engine.py     # Engine 5: EU Climate Benchmark (CTB/PAB)
    pai_mandatory_engine.py           # Engine 6: Mandatory PAI calculation
    carbon_trajectory_engine.py       # Engine 7: Carbon reduction trajectory
    investment_universe_engine.py     # Engine 8: Investment universe screening
  workflows/                         # 8 compliance workflows
    __init__.py
    annex_iii_disclosure.py           # Workflow 1: Annex III pre-contractual
    annex_v_reporting.py              # Workflow 2: Annex V periodic reporting
    sustainable_verification.py       # Workflow 3: Sustainable investment verification
    impact_reporting.py               # Workflow 4: Impact measurement reporting
    benchmark_monitoring.py           # Workflow 5: Benchmark alignment monitoring
    pai_mandatory_workflow.py         # Workflow 6: Mandatory PAI workflow
    downgrade_monitoring.py           # Workflow 7: Downgrade risk monitoring
    regulatory_update.py              # Workflow 8: Regulatory change management
  templates/                         # 8 report templates
    __init__.py
    annex_iii_precontractual.py       # Template 1: Annex III pre-contractual
    annex_v_periodic.py               # Template 2: Annex V periodic reporting
    impact_report.py                  # Template 3: Impact measurement report
    benchmark_methodology.py          # Template 4: Benchmark methodology
    sustainable_dashboard.py          # Template 5: Sustainable investment dashboard
    pai_mandatory_report.py           # Template 6: Mandatory PAI report
    carbon_trajectory_report.py       # Template 7: Carbon trajectory report
    audit_trail_report.py             # Template 8: Audit trail and provenance
  integrations/                      # 10 integration bridges
    __init__.py
    pack_orchestrator.py              # Integration 1: Article 9 orchestrator
    article8_pack_bridge.py           # Integration 2: PACK-010 cross-reference
    taxonomy_pack_bridge.py           # Integration 3: PACK-008 taxonomy bridge
    mrv_emissions_bridge.py           # Integration 4: MRV agents for PAI
    benchmark_data_bridge.py          # Integration 5: CTB/PAB benchmark data
    impact_data_bridge.py             # Integration 6: Impact data sources
    eet_data_bridge.py                # Integration 7: EET data (Article 9 fields)
    regulatory_bridge.py              # Integration 8: Regulatory updates
    health_check.py                   # Integration 9: System health verification
    setup_wizard.py                   # Integration 10: Guided configuration
  tests/
    __init__.py
    conftest.py                       # Shared fixtures and helpers
    test_manifest.py                  # Pack manifest validation
    test_config.py                    # Configuration tests
    test_sustainable_objective.py     # Engine 1 tests
    test_enhanced_dnsh.py             # Engine 2 tests
    test_full_taxonomy.py             # Engine 3 tests
    test_impact_measurement.py        # Engine 4 tests
    test_benchmark_alignment.py       # Engine 5 tests
    test_pai_mandatory.py             # Engine 6 tests
    test_carbon_trajectory.py         # Engine 7 tests
    test_investment_universe.py       # Engine 8 tests
    test_workflows.py                 # Workflow tests
    test_templates.py                 # Template tests
    test_integrations.py              # Integration tests
    test_demo.py                      # Demo/smoke tests
    test_e2e.py                       # End-to-end pipeline tests
    test_agent_integration.py         # Agent integration tests
```

### 4.2 Component Summary
| Component | Count | Description |
|-----------|-------|-------------|
| Engines | 8 | Sustainable objective, enhanced DNSH, full taxonomy, impact measurement, benchmark alignment, mandatory PAI, carbon trajectory, investment universe |
| Workflows | 8 | Annex III, Annex V, sustainable verification, impact reporting, benchmark monitoring, mandatory PAI, downgrade monitoring, regulatory update |
| Templates | 8 | Annex III, Annex V, impact report, benchmark methodology, sustainable dashboard, PAI report, carbon trajectory, audit trail |
| Integrations | 10 | Orchestrator, Article 8 bridge, taxonomy bridge, MRV bridge, benchmark data, impact data, EET, regulatory, health check, setup wizard |
| Presets | 5 | Impact fund, climate fund, social fund, ESG leader fund, transition fund |

---

## 5. Engine Specifications

### 5.1 Engine 1: Sustainable Objective Engine (`sustainable_objective_engine.py`)
**Purpose**: Verify that all investments qualify as sustainable investments per Article 2(17).

**Key Features:**
- Three-part test for every investment: Contribution + DNSH + Good Governance
- Environmental objective classification (6 EU Taxonomy objectives + other environmental)
- Social objective classification (inequality, cohesion, integration, labour, human capital, communities)
- Portfolio-level sustainable investment proportion calculation
- Minimum commitment tracking (binding pre-contractual %)
- Non-sustainable allocation tracking (cash, hedging, derivatives)
- Article 9 compliance scoring (must be near 100% sustainable)

**Calculation:**
```
sustainable_pct = sum(sustainable_investments_market_value) / total_nav * 100
non_sustainable_pct = sum(cash + hedging + liquidity) / total_nav * 100
compliant = sustainable_pct >= minimum_commitment AND non_sustainable_pct <= max_non_sustainable
```

### 5.2 Engine 2: Enhanced DNSH Engine (`enhanced_dnsh_engine.py`)
**Purpose**: Stricter DNSH assessment for Article 9 products (applies to ALL investments, not just sustainable portion).

**Key Features:**
- Mandatory DNSH for every holding in the portfolio
- Stricter thresholds than Article 8 DNSH (configurable per product)
- All 18 mandatory PAI indicators assessed
- Automatic exclusion on critical failures (controversial weapons, UNGC violations)
- DNSH methodology disclosure generation
- Per-environmental-objective DNSH checks (6 objectives)
- Portfolio-level DNSH compliance rate must be 100% (or near-100%)
- Remediation tracking for borderline holdings

### 5.3 Engine 3: Full Taxonomy Alignment Engine (`full_taxonomy_alignment.py`)
**Purpose**: Complete EU Taxonomy alignment calculation for Article 9 products referencing Articles 5 and 6.

**Key Features:**
- Full three-KPI alignment (turnover, CapEx, OpEx)
- Per-objective breakdown across 6 environmental objectives
- Enabling activities vs. transitional activities split
- Gas/nuclear Complementary Delegated Act disclosures
- Article 5 (environmental objective) vs Article 6 (social objective) classification
- Minimum safeguards verification (OECD Guidelines, UN Guiding Principles, ILO)
- Taxonomy-aligned sustainable vs. other sustainable breakdown
- Commitment adherence tracking

### 5.4 Engine 4: Impact Measurement Engine (`impact_measurement_engine.py`)
**Purpose**: Measure and report actual sustainability impact of Article 9 investments.

**Key Features:**
- Impact KPI definition and tracking (environmental and social)
- Theory of Change mapping (inputs -> activities -> outputs -> outcomes -> impact)
- Impact attribution methodology (investment contribution to real-world impact)
- Additionality assessment (would impact have occurred without the investment?)
- 15 environmental impact KPIs (CO2 avoided, renewable energy generated, water saved, etc.)
- 12 social impact KPIs (jobs created, people reached, training hours, etc.)
- Impact monetization (optional SROI calculation)
- Year-over-year impact comparison
- UN SDG alignment mapping (17 SDGs)
- Impact verification status tracking

### 5.5 Engine 5: Benchmark Alignment Engine (`benchmark_alignment_engine.py`)
**Purpose**: EU Climate Benchmark alignment for Article 9(3) carbon reduction products.

**Key Features:**
- EU Climate Transition Benchmark (CTB) compliance (7% annual decarbonization)
- EU Paris-Aligned Benchmark (PAB) compliance (7% annual decarbonization + exclusions)
- CTB minimum requirements: 30% lower carbon intensity than investable universe
- PAB minimum requirements: 50% lower carbon intensity than investable universe
- Year-on-year decarbonization rate calculation
- Sector exclusion compliance (PAB: fossil fuel revenue thresholds)
- Benchmark methodology disclosure
- Tracking error vs. designated benchmark
- Carbon intensity trajectory projection (to 2050)
- Deviation analysis and remediation recommendations

**CTB/PAB Calculation:**
```
# Carbon Transition Benchmark (CTB)
ctb_intensity_ratio = portfolio_carbon_intensity / universe_carbon_intensity
ctb_compliant = ctb_intensity_ratio <= 0.70  # 30% lower

# Paris-Aligned Benchmark (PAB)
pab_intensity_ratio = portfolio_carbon_intensity / universe_carbon_intensity
pab_compliant = pab_intensity_ratio <= 0.50  # 50% lower

# Annual decarbonization
decarbonization_rate = (prev_year_intensity - current_intensity) / prev_year_intensity * 100
decarbonization_compliant = decarbonization_rate >= 7.0  # 7% per year
```

### 5.6 Engine 6: PAI Mandatory Engine (`pai_mandatory_engine.py`)
**Purpose**: Mandatory PAI indicator calculation for Article 9 products (PAI consideration is required, not optional).

**Key Features:**
- All 18 mandatory PAI indicators (same calculation as PACK-010)
- At least 1 additional environmental PAI from Table 2
- At least 1 additional social PAI from Table 3
- PAI consideration in investment decisions (not just reporting)
- Investment decision integration documentation
- PAI-driven exclusion and engagement tracking
- Period-over-period comparison with improvement targets
- Action plan for adverse impacts
- Data quality scoring with minimum thresholds

### 5.7 Engine 7: Carbon Trajectory Engine (`carbon_trajectory_engine.py`)
**Purpose**: Carbon emission reduction trajectory calculation for Article 9(3) products.

**Key Features:**
- Portfolio carbon intensity trajectory (current to 2050)
- Science-Based Target alignment check (1.5C vs 2C pathways)
- Sectoral Decarbonization Approach (SDA) calculation
- Carbon budget allocation per holding
- Implied Temperature Rise (ITR) calculation
- Scope 1, 2, 3 emission trajectory modeling
- Transition plan assessment per holding
- Net Zero alignment scoring
- Carbon offset / removal credits tracking
- Paris Agreement alignment assessment

**Trajectory Calculation:**
```
# Target pathway (7% annual reduction from base year)
target_intensity_year_n = base_year_intensity * (1 - 0.07) ^ n

# Implied Temperature Rise
itr = interpolate(portfolio_trajectory, {1.5C_path, 2.0C_path, 3.0C_path, 4.0C_path})

# Carbon budget remaining
budget_remaining = 1.5C_total_budget - cumulative_financed_emissions
years_remaining = budget_remaining / annual_financed_emissions
```

### 5.8 Engine 8: Investment Universe Engine (`investment_universe_engine.py`)
**Purpose**: Screen and manage the eligible investment universe for Article 9 products.

**Key Features:**
- Sustainable investment universe definition and maintenance
- Multi-layer screening: exclusionary, norms-based, ESG best-in-class, impact
- Article 9-specific exclusion lists (controversial weapons, tobacco, thermal coal, oil sands, etc.)
- PAB-mandated exclusions (fossil fuel revenue > 1%, power gen > 100g CO2/kWh, etc.)
- CTB-mandated exclusions (controversial weapons only)
- Positive screening for sustainability contribution
- Investable universe coverage tracking
- Sector and geography diversification analysis
- New holding pre-approval screening
- Watch list management for borderline holdings

---

## 6. Workflow Specifications

### 6.1 Workflow 1: Annex III Pre-Contractual Disclosure (`annex_iii_disclosure.py`)
5-phase workflow for generating SFDR RTS Annex III pre-contractual disclosure:
1. **ProductClassification** - Verify Article 9 classification and sub-type (9(1), 9(2), 9(3))
2. **SustainableObjective** - Define and document sustainable investment objective
3. **InvestmentStrategy** - Document strategy, binding elements, good governance policy
4. **ProportionCalculation** - Calculate sustainable investment proportions and taxonomy alignment
5. **DisclosureGeneration** - Generate Annex III template with all required sections

### 6.2 Workflow 2: Annex V Periodic Reporting (`annex_v_reporting.py`)
5-phase workflow for annual periodic disclosure:
1. **DataCollection** - Gather reference period holdings, emissions, ESG data
2. **PerformanceAssessment** - Assess attainment of sustainable objective
3. **ImpactReporting** - Calculate and report sustainability impact metrics
4. **PAIReporting** - Mandatory PAI indicator values and actions
5. **ReportGeneration** - Generate Annex V template with comparatives

### 6.3 Workflow 3: Sustainable Investment Verification (`sustainable_verification.py`)
4-phase workflow for verifying 100% sustainable investment compliance:
1. **HoldingsAnalysis** - Classify all holdings against Article 2(17) definition
2. **DNSHScreening** - Enhanced DNSH assessment for all holdings
3. **GovernanceCheck** - Good governance verification for all investee companies
4. **ComplianceReport** - Generate compliance report with pass/fail status

### 6.4 Workflow 4: Impact Measurement Reporting (`impact_reporting.py`)
4-phase workflow for impact measurement:
1. **KPIDefinition** - Define impact KPIs aligned with sustainable objective
2. **DataCollection** - Collect impact data from investee companies
3. **ImpactCalculation** - Calculate portfolio-level impact metrics
4. **ReportGeneration** - Generate impact report with SDG alignment

### 6.5 Workflow 5: Benchmark Alignment Monitoring (`benchmark_monitoring.py`)
4-phase workflow for EU Climate Benchmark monitoring:
1. **BenchmarkSelection** - Select and configure CTB or PAB benchmark
2. **AlignmentAssessment** - Calculate alignment and decarbonization rate
3. **DeviationAnalysis** - Identify deviations and remediation needs
4. **TrajectoryProjection** - Project future alignment trajectory

### 6.6 Workflow 6: Mandatory PAI Workflow (`pai_mandatory_workflow.py`)
4-phase workflow for mandatory PAI consideration:
1. **DataSourcing** - Collect PAI data for all holdings
2. **PAICalculation** - Calculate all 18+ PAI indicators
3. **IntegrationAssessment** - Assess PAI integration in investment decisions
4. **ActionPlanning** - Generate action plans for adverse impacts

### 6.7 Workflow 7: Downgrade Risk Monitoring (`downgrade_monitoring.py`)
4-phase workflow for monitoring Article 9 to Article 8 downgrade risk:
1. **ComplianceCheck** - Check sustainable investment proportion
2. **ThresholdMonitoring** - Monitor DNSH, governance, and PAI thresholds
3. **RiskScoring** - Calculate downgrade risk score
4. **AlertGeneration** - Generate alerts for remediation

### 6.8 Workflow 8: Regulatory Update Management (`regulatory_update.py`)
3-phase workflow for regulatory change management:
1. **ChangeDetection** - Monitor SFDR, Taxonomy, and benchmark regulation changes
2. **ImpactAssessment** - Assess impact on Article 9 classification and disclosures
3. **MigrationPlanning** - Plan changes to maintain compliance

---

## 7. Template Specifications

### 7.1 Template 1: Annex III Pre-Contractual (`annex_iii_precontractual.py`)
SFDR RTS Annex III template for Article 9 pre-contractual disclosure:
- Product identification and sustainable investment objective statement
- "This financial product has sustainable investment as its objective"
- Sustainable investment objective description
- Investment strategy with binding elements
- Proportion of investments (near 100% sustainable, taxonomy breakdown)
- Monitoring methodology
- Data sources and limitations
- Due diligence and engagement
- Benchmark designation (if Art 9(2) or 9(3))

### 7.2 Template 2: Annex V Periodic (`annex_v_periodic.py`)
SFDR RTS Annex V template for Article 9 periodic reporting:
- Objective attainment assessment
- Top investments table
- Actual sustainable investment proportions
- Taxonomy alignment (turnover, CapEx, OpEx bar charts)
- Actions taken to attain objective
- PAI indicator values (mandatory)
- Impact metrics summary

### 7.3 Template 3: Impact Report (`impact_report.py`)
Impact measurement report:
- Theory of Change visualization
- Environmental impact KPIs
- Social impact KPIs
- UN SDG contribution mapping
- Year-over-year impact comparison
- Additionality assessment

### 7.4 Template 4: Benchmark Methodology (`benchmark_methodology.py`)
EU Climate Benchmark methodology report:
- Benchmark selection rationale (CTB vs PAB)
- Carbon intensity comparison
- Decarbonization trajectory
- Exclusion list compliance
- Deviation analysis

### 7.5 Template 5: Sustainable Investment Dashboard (`sustainable_dashboard.py`)
Executive dashboard:
- Sustainable investment proportion gauge
- DNSH compliance rate
- Good governance pass rate
- Taxonomy alignment metrics
- PAI indicator summary
- Impact highlights

### 7.6 Template 6: PAI Mandatory Report (`pai_mandatory_report.py`)
Mandatory PAI report:
- All 18 mandatory indicators with values
- Selected optional indicators
- Period-over-period comparison
- Actions taken and planned
- PAI integration in investment decisions

### 7.7 Template 7: Carbon Trajectory Report (`carbon_trajectory_report.py`)
Carbon reduction trajectory report:
- Portfolio carbon intensity trajectory chart
- Paris alignment assessment
- Science-based target alignment
- Sectoral contribution analysis
- Implied Temperature Rise

### 7.8 Template 8: Audit Trail Report (`audit_trail_report.py`)
Data lineage and provenance report:
- Calculation provenance hashes
- Data source tracking
- Methodology references
- Quality flags and estimated data %

---

## 8. Integration Specifications

### 8.1 Integration 1: Pack Orchestrator (`pack_orchestrator.py`)
10-phase Article 9 pipeline orchestrating all engines, workflows, and templates.

### 8.2 Integration 2: Article 8 Pack Bridge (`article8_pack_bridge.py`)
Cross-reference with PACK-010 for downgrade scenarios and shared PAI/taxonomy calculations.

### 8.3 Integration 3: Taxonomy Pack Bridge (`taxonomy_pack_bridge.py`)
Connection to PACK-008 EU Taxonomy Alignment for full taxonomy assessment.

### 8.4 Integration 4: MRV Emissions Bridge (`mrv_emissions_bridge.py`)
Bridge to 30 MRV agents for PAI emission indicator calculations.

### 8.5 Integration 5: Benchmark Data Bridge (`benchmark_data_bridge.py`)
CTB/PAB benchmark data intake and management.

### 8.6 Integration 6: Impact Data Bridge (`impact_data_bridge.py`)
Impact measurement data intake from investee companies and third parties.

### 8.7 Integration 7: EET Data Bridge (`eet_data_bridge.py`)
European ESG Template with Article 9-specific field mappings.

### 8.8 Integration 8: Regulatory Bridge (`regulatory_bridge.py`)
SFDR, Taxonomy, and Benchmark regulatory update tracking.

### 8.9 Integration 9: Health Check (`health_check.py`)
20-category system verification for Article 9 compliance readiness.

### 8.10 Integration 10: Setup Wizard (`setup_wizard.py`)
8-step guided configuration for Article 9 product setup.

---

## 9. Preset Specifications

### 9.1 Impact Fund Preset
- Article 9(1) classification
- Environmental + social objectives
- Impact measurement KPIs enabled
- SDG alignment tracking
- 95% minimum sustainable investment

### 9.2 Climate Fund Preset
- Article 9(3) classification
- Carbon reduction objective
- PAB benchmark alignment
- Carbon trajectory tracking
- 7% annual decarbonization target

### 9.3 Social Fund Preset
- Article 9(1) classification
- Social objective focus
- Social PAI indicators prioritized
- Gender equality, labour rights KPIs
- 90% minimum sustainable investment

### 9.4 ESG Leader Fund Preset
- Article 9(1) classification
- Best-in-class ESG approach
- Top quartile ESG scoring
- Enhanced governance requirements
- 95% minimum sustainable investment

### 9.5 Transition Fund Preset
- Article 9(1) classification
- Transition financing focus
- CTB benchmark alignment
- Enabling and transitional activities
- Science-based targets required

---

## 10. PAI Calculation Methodology

All 18 mandatory PAI indicators use the same SFDR RTS formulas as PACK-010. For Article 9, the key difference is that PAI consideration is MANDATORY (not optional) and must be integrated into investment decision-making.

### 10.1 Mandatory PAI Indicators (Table 1 Annex I RTS)
- PAI 1-6: Climate & GHG (GHG emissions, carbon footprint, GHG intensity, fossil fuel exposure, non-renewable energy, energy intensity by NACE)
- PAI 7-9: Environment (biodiversity, water emissions, hazardous waste)
- PAI 10-14: Social (UNGC/OECD violations, compliance mechanisms, gender pay gap, board diversity, controversial weapons)
- PAI 15-16: Sovereign (GHG intensity, social violations)
- PAI 17-18: Real Estate (fossil fuel exposure, energy inefficiency)

### 10.2 Additional PAI Requirements for Article 9
- Must select at least 1 additional environmental indicator from Table 2
- Must select at least 1 additional social indicator from Table 3
- Must document how PAI indicators are considered in investment decisions
- Must report actions taken and targets set for each indicator

---

## 11. EU Climate Benchmark Specifications

### 11.1 Climate Transition Benchmark (CTB)
- Minimum 30% lower carbon intensity than investable universe
- 7% year-on-year decarbonization of carbon intensity
- Controversial weapons exclusion
- No sector exclusion requirements beyond controversial weapons
- Self-decarbonization commitment by 2050

### 11.2 Paris-Aligned Benchmark (PAB)
- Minimum 50% lower carbon intensity than investable universe
- 7% year-on-year decarbonization of carbon intensity
- Exclusions required:
  - Companies deriving >= 1% revenue from fossil fuel exploration
  - Companies deriving >= 10% revenue from fossil fuel refining/processing
  - Companies deriving >= 50% revenue from fossil fuel distribution
  - Electric utilities with carbon intensity > 100g CO2/kWh
  - Controversial weapons
- Scope 3 GHG phase-in (4 years from benchmark creation)

---

## 12. Testing Strategy

### 12.1 Unit Tests (~400 tests)
- `test_manifest.py`: 20 tests (pack.yaml validation)
- `test_config.py`: 45 tests (config models, presets, validation)
- 8 engine test files: ~25 tests each (200 total)
- `test_workflows.py`: 15 tests
- `test_templates.py`: 18 tests
- `test_integrations.py`: 35 tests
- `test_demo.py`: 60 tests (smoke/importability)
- `test_e2e.py`: 12 tests (end-to-end pipelines)
- `test_agent_integration.py`: 15 tests (component wiring)

### 12.2 Cross-Pack Integration Tests
- Pack orchestrator loads and configures
- Article 8 bridge connects to PACK-010
- Taxonomy bridge connects to PACK-008
- PAI engine loads with mandatory mode
- 11-pack coexistence test
- Pack runner integration

---

## 13. Timeline

| Phase | Deliverable | Duration |
|-------|-------------|----------|
| 1 | PRD (this document) | Complete |
| 2 | Package structure + config | Parallel |
| 3 | 8 calculation engines | Parallel |
| 4 | 8 workflows + 8 templates | Parallel |
| 5 | 10 integrations | Parallel |
| 6 | Unit tests + integration tests | Parallel |
| 7 | Cross-pack tests + verification | Sequential |

---

## 14. Regulatory References

1. Regulation (EU) 2019/2088 - SFDR Level 1
2. Delegated Regulation (EU) 2022/1288 - SFDR RTS (Annexes III and V)
3. Regulation (EU) 2020/852 - Taxonomy Regulation (Articles 5, 6)
4. Delegated Regulation (EU) 2021/2178 - Taxonomy Disclosures
5. Regulation (EU) 2019/2089 - EU Climate Benchmarks
6. Commission Delegated Regulation (EU) 2020/1816 - CTB minimum standards
7. Commission Delegated Regulation (EU) 2020/1818 - PAB minimum standards
8. ESMA Q&A on SFDR - Supervisory guidance
9. ESMA Fund Names Guidelines - Naming rules for sustainability funds
10. EET v1.1 Standard - European ESG Template (FinDatEx)
