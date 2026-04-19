# PRD-PACK-024: Carbon Neutral Pack

**Pack ID:** PACK-024-carbon-neutral
**Category:** Net Zero Packs
**Tier:** Standalone
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-18
**Prerequisite:** None (standalone; enhanced with PACK-021/022/023 if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Carbon neutrality remains the most widely adopted corporate climate claim, with over 5,000 organizations globally declaring carbon neutral status or commitments. Unlike net zero -- which demands deep decarbonization with a maximum 10% residual offset via permanent removals (SBTi) -- carbon neutrality under ISO 14068-1:2023 and PAS 2060:2014 permits 100% neutralization of the declared carbon footprint through a combination of reductions and carbon credit offsets. This accessibility makes carbon neutrality the entry point for most organizations, yet the claim carries significant substantiation risks:

1. **Standards complexity**: ISO 14068-1:2023 replaced PAS 2060 as the international standard for carbon neutrality, introducing stricter quantification requirements, mandatory reduction plans, and a shift from "carbon neutral" to "carbon neutrality" terminology. Organizations must navigate both standards (many verifiers still reference PAS 2060) while preparing for ISO 14068-2 (products) and ISO 14068-3 (projects).

2. **Carbon credit quality crisis**: The voluntary carbon market (VCM) has faced credibility challenges with studies showing that up to 90% of certain credit categories (notably avoided deforestation/REDD+) may not deliver claimed reductions. The Integrity Council for the Voluntary Carbon Market (ICVCM) Core Carbon Principles (CCPs) and the Voluntary Carbon Markets Integrity Initiative (VCMI) Claims Code now set minimum quality thresholds that most corporate buyers do not systematically apply.

3. **Credit portfolio management**: Managing a carbon credit portfolio requires tracking vintage years, registry retirements, permanence risk buffers, credit-type diversification (removal vs. avoidance, nature-based vs. technological), geographic distribution, co-benefit verification, and correspondence between neutralization period and credit vintage.

4. **Claims substantiation risk**: Regulators globally are cracking down on unsubstantiated "carbon neutral" claims. The EU Green Claims Directive (proposed), UK ASA rulings, Australian ACCC enforcement, and US FTC Green Guides revision all demand transparent, verifiable, and non-misleading neutrality claims with clear scope boundaries and temporal alignment.

5. **Reduction-first credibility**: ISO 14068-1 requires a carbon management plan demonstrating genuine reduction efforts before offsetting residual emissions. Organizations that rely heavily on credits without reduction progress face reputational and regulatory risk. The Oxford Principles for Net Zero Aligned Carbon Offsetting establish a quality hierarchy that favors reduction over compensation.

6. **Annual cycle management**: Carbon neutrality requires annual declaration cycles: base year inventory, current year inventory, reduction progress assessment, credit procurement and retirement, independent verification, and public declaration. Most organizations manage this in spreadsheets with no audit trail.

7. **Registry integration complexity**: Carbon credits are issued and retired across multiple registries (Verra VCS, Gold Standard, Climate Action Reserve, American Carbon Registry, Puro.earth) with different APIs, metadata schemas, retirement procedures, and buffer pool mechanisms. Manual tracking across registries is error-prone and time-consuming.

8. **Scope boundary decisions**: Organizations must define the subject of neutrality (organization, product, service, event, building) and the scope boundary (Scope 1 only, Scope 1+2, Scope 1+2+partial Scope 3, or full Scope 1+2+3). Under-inclusive boundaries invite greenwashing accusations; over-inclusive boundaries increase cost. ISO 14068-1 requires clear boundary documentation and justification.

### 1.2 Solution Overview

PACK-024 is the **Carbon Neutral Pack** -- the fourth pack in the "Net Zero Packs" category. It provides a comprehensive carbon neutrality lifecycle management solution purpose-built for ISO 14068-1:2023 and PAS 2060:2014 compliance. Unlike the net-zero packs (PACK-021/022/023) which focus on deep decarbonization with minimal offsets, PACK-024 centers on robust carbon credit management, quality assurance, and claims substantiation while still enforcing reduction-first credibility through mandatory carbon management plans.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the full carbon neutrality lifecycle: footprint quantification, reduction planning, credit sourcing and quality assessment, portfolio optimization, registry retirement, declaration preparation, and verification support.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-024 Carbon Neutral Pack |
|-----------|-------------------------------|------------------------------|
| Time to achieve carbon neutral declaration | 600-1200 hours annually | <60 hours (10-20x faster) |
| Footprint accuracy | Incomplete, inconsistent boundaries | Deterministic, ISO 14064-1 aligned, all scopes |
| Credit quality assessment | Ad hoc broker reliance | Systematic ICVCM CCP scoring with 12 quality dimensions |
| Portfolio optimization | Manual credit selection | Cost-quality Pareto optimization with permanence weighting |
| Registry management | Manual retirement tracking | Automated multi-registry integration (5 registries) |
| Claims substantiation | Legal review of marketing copy | Automated ISO 14068-1/PAS 2060 compliance validation |
| Verification support | Manual evidence assembly | Pre-assembled verification package with full audit trail |
| Reduction credibility | Qualitative action descriptions | Quantified carbon management plan with year-over-year tracking |
| Annual cycle | Restart from scratch each year | Continuous monitoring with annual roll-forward |

### 1.4 Carbon Neutrality vs. Net Zero: Positioning

| Aspect | Carbon Neutrality (PACK-024) | Net Zero (PACK-021/022/023) |
|--------|------------------------------|----------------------------|
| **Standard** | ISO 14068-1:2023, PAS 2060:2014 | SBTi Corporate Net-Zero Standard |
| **Offset allowance** | 100% of footprint may be offset | Max 10% residual (90%+ reduction required) |
| **Credit types** | Avoidance and removal credits accepted | Only permanent carbon dioxide removals |
| **Timeline** | Annual declaration (no fixed end date) | 2050 or sooner long-term target |
| **Reduction requirement** | Must demonstrate "reduction efforts" (qualitative + plan) | 4.2%/yr absolute or SDA convergence |
| **Verification** | Third-party verification recommended, not mandatory | SBTi target validation mandatory |
| **Scope** | Flexible (org, product, service, event) | Full organizational Scope 1+2+3 |
| **Ideal for** | Organizations seeking immediate climate credibility | Organizations committed to deep decarbonization |

### 1.5 Target Users

**Primary:**
- Sustainability managers seeking carbon neutral certification for their organization
- Marketing and communications teams substantiating carbon neutral claims
- Procurement teams sourcing and managing carbon credit portfolios
- Organizations not yet ready for SBTi but wanting credible climate action

**Secondary:**
- CFOs managing carbon credit budgets and cost optimization
- Legal/compliance teams reviewing carbon neutral claim validity
- Verification bodies conducting ISO 14068-1/PAS 2060 assessments
- ESG rating agencies evaluating corporate carbon neutrality claims
- Sustainability consultants guiding clients to carbon neutral status

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to carbon neutral declaration | <60 hours annually (vs. 600+ manual) | Time from data intake to verified declaration |
| Footprint calculation accuracy | 100% match with manual verification | Tested against 500 known emission values |
| Credit quality scoring accuracy | 100% alignment with ICVCM criteria | Cross-validated against ICVCM assessment outcomes |
| ISO 14068-1 compliance rate | 100% of applicable requirements met | Automated compliance checklist coverage |
| Registry retirement accuracy | Zero discrepancies | Reconciled against registry records |
| Customer NPS | >50 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ISO 14068-1:2023 | Carbon neutrality -- Quantification, reporting and verification of GHG emissions and removals | Core standard; defines carbon neutrality requirements for organizations, replaces PAS 2060 |
| PAS 2060:2014 | Specification for the demonstration of carbon neutrality | Legacy standard still widely referenced; backward compatibility maintained |
| ISO 14064-1:2018 | GHG quantification and reporting at organization level | Underlying GHG inventory methodology for footprint quantification |
| GHG Protocol Corporate Standard | WRI/WBCSD (2004, 2015 update) | Scope 1+2 emission calculation methodology |
| GHG Protocol Scope 3 Standard | WRI/WBCSD (2011) | Scope 3 (categories 1-15) methodology |

### 2.2 Carbon Credit Quality Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| ICVCM Core Carbon Principles | ICVCM (2023) | 10 Core Carbon Principles for credit quality assessment |
| VCMI Claims Code | VCMI (2023) | Claims substantiation for use of carbon credits |
| Oxford Principles for Net Zero Aligned Offsetting | Oxford (2020) | Quality hierarchy: reduce > avoid > remove; permanence progression |
| ICROA Code of Best Practice | ICROA (2014, updated 2023) | Industry best practice for offset providers |
| Article 6 Paris Agreement | UNFCCC (2021) | Corresponding adjustments for international transfers |

### 2.3 Claims & Disclosure Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| EU Green Claims Directive | EU (proposed 2023) | Substantiation requirements for environmental claims |
| ISO 14021:2016 | Self-declared environmental claims | Type II environmental labeling |
| FTC Green Guides | US FTC (2012, revision pending) | US guidance on carbon neutral/offset marketing claims |
| UK ASA CAP Code | UK ASA (2023) | UK advertising standards for carbon neutral claims |
| ACCC Guidelines | Australia ACCC (2023) | Australian consumer protection for green claims |

### 2.4 Supporting Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| IPCC AR6 | IPCC (2021) | GWP-100 values for greenhouse gases |
| ESRS E1 Climate Change | EU (2023) | E1-7 GHG removals and carbon credits disclosure |
| CDP Climate Change | CDP (2024) | C-FI (Forest, Land and Agriculture) credit disclosure |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics & Targets for offset strategy |
| ISO 14068-2 (forthcoming) | ISO (expected 2025) | Product-level carbon neutrality (forward compatibility) |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Deterministic calculation engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report and dashboard templates |
| Integrations | 12 | Agent, app, registry, and pack bridges |
| Presets | 8 | Sector and scope-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `footprint_quantification_engine.py` | Carbon footprint calculation for the declared subject (organization, product, service, event) with configurable scope boundary (S1, S1+S2, S1+S2+S3partial, S1+S2+S3full), ISO 14064-1 aligned methodology, and data quality scoring |
| 2 | `carbon_management_plan_engine.py` | Reduction-first carbon management plan generation with quantified abatement actions, timeline, investment requirements, and year-over-year reduction progress tracking per ISO 14068-1 Section 9 |
| 3 | `credit_quality_engine.py` | 12-dimension carbon credit quality assessment aligned to ICVCM Core Carbon Principles: additionality, permanence, robust quantification, no double counting, sustainable development, governance, tracking, transparency, no net harm, contribution to net-zero transition, registry integrity, co-benefit verification |
| 4 | `portfolio_optimization_engine.py` | Carbon credit portfolio optimization using cost-quality Pareto frontier analysis with constraints on minimum quality score, permanence thresholds, removal-to-avoidance ratio, vintage windows, geographic diversification, and credit-type diversification |
| 5 | `registry_retirement_engine.py` | Multi-registry credit retirement management for Verra VCS, Gold Standard, Climate Action Reserve (CAR), American Carbon Registry (ACR), and Puro.earth with retirement record tracking, serial number validation, vintage verification, and buffer pool monitoring |
| 6 | `neutralization_balance_engine.py` | Neutralization balance calculation ensuring retired credits exactly match (or exceed) the declared carbon footprint within the neutralization period, with temporal alignment validation (credit vintage vs. reporting year) and surplus/deficit tracking |
| 7 | `claims_substantiation_engine.py` | Automated ISO 14068-1/PAS 2060 compliance validation with 35-criterion checklist covering boundary definition, footprint completeness, reduction evidence, credit quality, retirement confirmation, temporal alignment, public disclosure, and verification requirements |
| 8 | `verification_package_engine.py` | Pre-assembly of third-party verification evidence package including footprint calculation workpapers, credit retirement confirmations, carbon management plan evidence, methodology documentation, and SHA-256 provenance chains |
| 9 | `annual_cycle_engine.py` | Annual carbon neutrality cycle management with base year roll-forward, current year inventory, reduction progress assessment, credit gap calculation, procurement trigger, retirement scheduling, declaration generation, and verification timeline |
| 10 | `permanence_risk_engine.py` | Credit permanence risk assessment with reversal risk scoring (1-5 scale) across 8 risk categories (fire, pest, political, market, geological, technological, regulatory, force majeure), buffer pool adequacy evaluation, and insurance/guarantee assessment |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `carbon_neutral_onboarding_workflow.py` | 5: SubjectDefinition -> BoundaryConfig -> FootprintCalc -> DataQuality -> BaselineReport | Guided onboarding to establish carbon footprint baseline with scope boundary selection |
| 2 | `reduction_planning_workflow.py` | 4: FootprintAnalysis -> AbatementIdentify -> CostAssess -> CarbonMgmtPlan | Build ISO 14068-1 compliant carbon management plan |
| 3 | `credit_sourcing_workflow.py` | 5: GapCalc -> MarketScan -> QualityScreen -> PortfolioDesign -> ProcurementPlan | Source and quality-assess carbon credits to fill neutralization gap |
| 4 | `portfolio_management_workflow.py` | 4: InventoryUpdate -> QualityReview -> RebalanceCalc -> RetirementSchedule | Ongoing portfolio management with rebalancing and retirement scheduling |
| 5 | `declaration_workflow.py` | 5: FootprintFinalize -> RetirementConfirm -> ClaimsValidate -> DeclarationGen -> DisclosurePublish | End-to-end carbon neutral declaration preparation |
| 6 | `verification_workflow.py` | 4: EvidenceAssemble -> CompletenessCheck -> PackageGen -> VerifierHandoff | Prepare and deliver verification evidence package |
| 7 | `annual_renewal_workflow.py` | 5: YearRollforward -> InventoryUpdate -> ReductionReview -> CreditProcure -> Redeclare | Annual carbon neutrality renewal cycle |
| 8 | `full_carbon_neutral_workflow.py` | 8: Onboard -> Reduce -> Source -> Optimize -> Retire -> Substantiate -> Declare -> Verify | End-to-end carbon neutrality achievement |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `carbon_neutral_declaration.py` | MD, HTML, PDF, JSON | Formal carbon neutrality declaration document per ISO 14068-1 Annex A |
| 2 | `footprint_report.py` | MD, HTML, JSON | Detailed carbon footprint quantification report with scope breakdown |
| 3 | `carbon_management_plan_report.py` | MD, HTML, JSON | Carbon management plan with reduction actions and timeline |
| 4 | `credit_quality_report.py` | MD, HTML, JSON | ICVCM-aligned credit quality assessment with 12-dimension scoring |
| 5 | `portfolio_summary_report.py` | MD, HTML, JSON | Carbon credit portfolio summary with diversification analysis |
| 6 | `neutralization_balance_report.py` | MD, HTML, JSON | Neutralization balance reconciliation (footprint vs. retirements) |
| 7 | `claims_compliance_report.py` | MD, HTML, JSON | ISO 14068-1/PAS 2060 compliance checklist with evidence mapping |
| 8 | `verification_package_report.py` | MD, HTML, JSON | Complete verification evidence package for third-party verifiers |
| 9 | `annual_progress_report.py` | MD, HTML, JSON | Year-over-year carbon neutrality progress dashboard |
| 10 | `public_disclosure_report.py` | MD, HTML, JSON | Public-facing carbon neutrality disclosure statement |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline with retry, provenance, conditional credit/verification phases |
| 2 | `mrv_bridge.py` | Routes to all 30 MRV agents for emission calculations |
| 3 | `ghg_app_bridge.py` | Bridges to GL-GHG-APP for inventory management and base year |
| 4 | `decarb_bridge.py` | Bridges to 21 DECARB-X agents for reduction planning |
| 5 | `offset_bridge.py` | Bridges to carbon credit/offset agents for credit sourcing |
| 6 | `registry_bridge.py` | Bridges to Verra VCS, Gold Standard, CAR, ACR, Puro.earth APIs |
| 7 | `reporting_bridge.py` | Bridges to CDP, TCFD, ESRS E1 for cross-framework reporting |
| 8 | `pack021_bridge.py` | Bridge to PACK-021 Net Zero Starter (baseline, gap, offsets) |
| 9 | `pack023_bridge.py` | Bridge to PACK-023 SBTi Alignment (for dual net-zero + carbon neutral strategies) |
| 10 | `data_bridge.py` | Bridges to 20 DATA agents for data intake |
| 11 | `health_check.py` | 20-category system verification |
| 12 | `setup_wizard.py` | 7-step guided carbon neutrality configuration wizard |

### 3.6 Presets

| # | Preset | Subject/Sector | Key Characteristics |
|---|--------|---------------|---------------------|
| 1 | `organization_full.yaml` | Full Organization (S1+S2+S3) | Most comprehensive; all scopes, highest credit volume, strongest claim |
| 2 | `organization_operational.yaml` | Organization (S1+S2 only) | Excludes Scope 3; lower cost, narrower claim boundary |
| 3 | `product_service.yaml` | Product/Service | LCA-based footprint, product-level credits, ISO 14068-2 forward compatibility |
| 4 | `event.yaml` | Event/Conference | Short-duration, travel-heavy footprint, one-time credit purchase |
| 5 | `manufacturing.yaml` | Manufacturing Organization | High Scope 1 (process/combustion), energy-intensive, industrial credit focus |
| 6 | `services.yaml` | Professional/Financial Services | Low Scope 1, high Scope 3 (business travel, purchased goods), removal-weighted |
| 7 | `retail.yaml` | Retail/Consumer Goods | Supply chain focus, consumer-facing claims, co-benefit emphasis |
| 8 | `sme_simplified.yaml` | SME (any sector) | Simplified 5-engine flow, spend-based Scope 3, standard credit portfolio |

---

## 4. Engine Specifications

### 4.1 Engine 1: Footprint Quantification Engine

**Purpose:** Calculate the carbon footprint for the declared subject with configurable scope boundaries.

**Key Features:**
- Subject types: ORGANIZATION, PRODUCT, SERVICE, EVENT, BUILDING, PROJECT
- Scope boundaries: S1_ONLY, S1_S2, S1_S2_S3_PARTIAL, S1_S2_S3_FULL
- Consolidation approaches: operational control, financial control, equity share
- GHG coverage: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 (Kyoto basket)
- Data quality scoring: primary (measured), secondary (calculated), proxy (estimated), spend-based
- Exclusion justification: any exclusion >1% of total must be documented per ISO 14068-1

**Key Models:**
- `FootprintInput` - Subject definition, organizational boundary, activity data, emission factors
- `FootprintResult` - Total CO2e by scope, gas breakdown, data quality score, exclusions log
- `ScopeBoundary` - Scope inclusion/exclusion with justification
- `DataQualityAssessment` - Per-source data quality rating with uncertainty quantification
- `ExclusionRecord` - Documented exclusion with materiality justification

**Calculation Flow:**
1. Validate subject definition and organizational boundary
2. Collect activity data per scope/category
3. Apply emission factors (IPCC AR6 GWP-100)
4. Calculate total CO2e with uncertainty range
5. Score data quality and flag gaps
6. Document any exclusions with materiality assessment

### 4.2 Engine 2: Carbon Management Plan Engine

**Purpose:** Generate a reduction-first carbon management plan per ISO 14068-1 Section 9.

**ISO 14068-1 Requirements:**
- Demonstrate genuine efforts to reduce emissions before offsetting
- Quantified reduction targets with timelines
- Identified abatement actions with expected impact
- Year-over-year reduction progress tracking
- Plan review and update schedule

**Key Features:**
- Abatement action library integration (via DECARB-X-001, 500+ options)
- Cost-effectiveness ranking ($/tCO2e avoided)
- Implementation timeline with phased milestones
- Reduction progress tracking: actual vs. planned
- Plan adequacy scoring against ISO 14068-1 requirements

**Key Models:**
- `CarbonMgmtPlanInput` - Current footprint, budget, timeline, sector, constraints
- `CarbonMgmtPlanResult` - Prioritized actions, expected reductions, investment plan, adequacy score
- `AbatementAction` - Single action with cost, abatement potential, timeline, TRL
- `ReductionProgress` - Year-over-year reduction tracking with variance

### 4.3 Engine 3: Credit Quality Engine

**Purpose:** Assess carbon credit quality against ICVCM Core Carbon Principles.

**12 Quality Dimensions (ICVCM-aligned):**

| # | Dimension | Weight | Assessment Criteria |
|---|-----------|--------|-------------------|
| 1 | Additionality | 15% | Project would not occur without carbon finance; regulatory surplus test |
| 2 | Permanence | 15% | Duration of carbon storage; reversal risk; buffer pool adequacy |
| 3 | Robust Quantification | 12% | Conservative baseline; monitoring accuracy; uncertainty assessment |
| 4 | No Double Counting | 10% | Corresponding adjustments (Article 6); no double issuance/claiming/use |
| 5 | Sustainable Development | 8% | SDG co-benefits; stakeholder consultation; do-no-significant-harm |
| 6 | Governance | 8% | Registry governance; standard-setter independence; conflict of interest |
| 7 | Tracking | 7% | Serial number tracking; retirement transparency; registry integrity |
| 8 | Transparency | 7% | Public project documentation; monitoring reports; third-party validation |
| 9 | No Net Harm | 5% | Environmental and social safeguards; FPIC compliance |
| 10 | Net-Zero Transition Contribution | 5% | Credit contributes to broader transition; not locking in fossil infrastructure |
| 11 | Registry Integrity | 4% | Registry ICVCM eligibility; operational standards; audit history |
| 12 | Co-Benefit Verification | 4% | Independently verified co-benefits; SDG impact reporting |

**Credit Type Classification:**

| Type | Subtype | Permanence | Examples |
|------|---------|-----------|---------|
| Removal | Technological | High (1000+ years) | DACCS, BECCS, enhanced weathering |
| Removal | Nature-based | Medium (20-100 years) | Afforestation, soil carbon, biochar |
| Avoidance | Renewable energy | N/A (avoided emission) | Wind, solar, hydro replacing fossil |
| Avoidance | Energy efficiency | N/A (avoided emission) | Cookstove, industrial efficiency |
| Avoidance | Forestry (REDD+) | Medium (risk-adjusted) | Avoided deforestation, forest management |
| Avoidance | Methane reduction | N/A (avoided emission) | Landfill gas, coal mine methane |

**Quality Score Output:** Composite score 0-100 with per-dimension breakdown.

### 4.4 Engine 4: Portfolio Optimization Engine

**Purpose:** Optimize carbon credit portfolio across cost, quality, and diversification objectives.

**Optimization Objectives:**
- Minimize total cost subject to minimum quality floor
- Maximize quality subject to budget ceiling
- Pareto-optimal frontier: cost vs. quality trade-off curve

**Optimization Constraints:**
- Minimum composite quality score (default: 60/100)
- Maximum avoidance-to-removal ratio (default: 80:20 avoidance:removal)
- Vintage window: credits must be within reporting year +/- 3 years (ISO 14068-1)
- Maximum single-project concentration (default: 25%)
- Minimum geographic diversification (default: 2 regions)
- Permanence floor for removal credits (default: 40 years)
- Co-benefit requirements (e.g., minimum 2 verified SDGs)

**Oxford Principles Alignment:**
- Track portfolio progression along the Oxford quality ladder
- Year-over-year shift from avoidance toward removal
- Shift from nature-based toward technological removal over time
- Increasing permanence threshold year-over-year

**Key Models:**
- `PortfolioOptInput` - Available credits, budget, constraints, objectives
- `PortfolioOptResult` - Optimal portfolio allocation, cost, quality score, Pareto frontier
- `CreditAllocation` - Per-credit allocation with quantity, cost, quality, rationale
- `ParetoPoint` - Single point on cost-quality frontier

### 4.5 Engine 5: Registry Retirement Engine

**Purpose:** Manage carbon credit retirements across five registries.

**Supported Registries:**

| Registry | Abbreviation | Coverage | API Integration |
|----------|-------------|----------|-----------------|
| Verra Verified Carbon Standard | VCS | Global; largest by volume | Registry API for retirement lookup |
| Gold Standard | GS | Global; SDG co-benefits focus | Impact Registry API |
| Climate Action Reserve | CAR | North America focus | Reserve API |
| American Carbon Registry | ACR | US/international | ACR Registry API |
| Puro.earth | Puro | Engineered removals | Puro Registry API |

**Key Features:**
- Retirement record creation with serial number range validation
- Vintage year verification against neutralization period
- Retirement confirmation receipt generation
- Buffer pool status monitoring (for nature-based reversals)
- Cross-registry duplicate detection
- Retirement status polling and confirmation tracking

**Key Models:**
- `RetirementInput` - Credits to retire, beneficiary, reason, neutralization period
- `RetirementResult` - Retirement confirmations, serial numbers, registry receipts
- `RetirementRecord` - Single retirement with registry, serial range, date, status

### 4.6 Engine 6: Neutralization Balance Engine

**Purpose:** Ensure credits retired equal or exceed the declared carbon footprint.

**Balance Calculation:**
```
Neutralization Gap = Declared Footprint - (Achieved Reductions + Retired Credits)
```
Where:
- Declared Footprint = total CO2e within the declared scope boundary for the neutralization period
- Achieved Reductions = verified emission reductions from carbon management plan actions
- Retired Credits = total tCO2e of credits retired in the relevant registries

**Temporal Alignment Rules (ISO 14068-1):**
- Neutralization period must be a defined 12-month period
- Credits must be retired during or within 12 months after the neutralization period
- Credit vintage must be within the neutralization period or the preceding 3 calendar years
- Forward-crediting (using future vintage credits) is not permitted

**Key Models:**
- `BalanceInput` - Footprint, reductions achieved, credits retired, neutralization period
- `BalanceResult` - Balance status (neutral/surplus/deficit), gap quantity, temporal compliance
- `TemporalCheck` - Vintage-to-period alignment validation per credit

### 4.7 Engine 7: Claims Substantiation Engine

**Purpose:** Validate carbon neutral claims against ISO 14068-1 and PAS 2060.

**35-Criterion Compliance Checklist:**

| Category | Criteria Count | Key Requirements |
|----------|---------------|------------------|
| Subject & Boundary (SB) | 5 | SB-1 Subject defined; SB-2 Boundary documented; SB-3 Scope justified; SB-4 Exclusions <1% or documented; SB-5 Consolidation approach stated |
| Footprint Quantification (FQ) | 6 | FQ-1 ISO 14064-1 aligned; FQ-2 All material GHGs; FQ-3 GWP-100 AR6; FQ-4 Data quality scored; FQ-5 Uncertainty assessed; FQ-6 Base year defined |
| Carbon Management Plan (CM) | 5 | CM-1 Reduction targets set; CM-2 Actions quantified; CM-3 Timeline defined; CM-4 Progress tracked; CM-5 Plan reviewed annually |
| Credit Quality (CQ) | 6 | CQ-1 ICVCM-eligible standard; CQ-2 Additionality verified; CQ-3 Permanence assessed; CQ-4 No double counting; CQ-5 Co-benefits documented; CQ-6 Quality score >= threshold |
| Retirement & Balance (RB) | 5 | RB-1 Credits retired (not held); RB-2 Vintage within window; RB-3 Balance >= footprint; RB-4 Retirement confirmations obtained; RB-5 Serial numbers recorded |
| Declaration & Disclosure (DD) | 5 | DD-1 Declaration period stated; DD-2 Scope boundary disclosed; DD-3 Credit types disclosed; DD-4 Reduction progress disclosed; DD-5 Public availability |
| Verification (VR) | 3 | VR-1 Third-party verification obtained or planned; VR-2 Verifier accreditation documented; VR-3 Verification statement available |

**Result:** Overall compliance status (COMPLIANT / NON_COMPLIANT / PARTIAL), per-criterion pass/fail/warning, and remediation guidance for each gap.

### 4.8 Engine 8: Verification Package Engine

**Purpose:** Assemble evidence for third-party verification.

**Package Contents:**
1. Carbon footprint calculation workpapers with source data references
2. Emission factor sources and version documentation
3. Carbon management plan with reduction evidence
4. Credit purchase agreements and retirement confirmations
5. Registry retirement receipts with serial numbers
6. Neutralization balance reconciliation
7. Claims substantiation checklist results
8. SHA-256 provenance hashes for all calculation outputs
9. Data quality assessment and uncertainty analysis
10. Prior year comparisons (if renewal)

### 4.9 Engine 9: Annual Cycle Engine

**Purpose:** Manage the annual carbon neutrality declaration lifecycle.

**Annual Cycle Phases:**
1. **Year-Open** (Month 1): Roll forward from prior year, update organizational boundary
2. **Data Collection** (Months 1-3): Collect activity data for new reporting period
3. **Footprint Calculation** (Month 3-4): Calculate current year emissions
4. **Reduction Review** (Month 4): Assess carbon management plan progress
5. **Credit Gap** (Month 4-5): Calculate remaining emissions requiring offset
6. **Credit Procurement** (Month 5-8): Source, quality-assess, and purchase credits
7. **Retirement** (Month 8-10): Retire credits in registries
8. **Declaration** (Month 10-11): Generate and finalize carbon neutral declaration
9. **Verification** (Month 11-12): Submit to third-party verifier
10. **Publication** (Month 12): Publish verified declaration and disclosure

**Key Models:**
- `AnnualCycleInput` - Prior year records, organizational changes, current data availability
- `AnnualCycleResult` - Phase status, timeline, blockers, next actions
- `CyclePhase` - Single phase with status, start/end dates, dependencies, deliverables

### 4.10 Engine 10: Permanence Risk Engine

**Purpose:** Assess reversal risk for carbon credits, particularly nature-based solutions.

**8 Risk Categories:**

| # | Risk Category | Applies To | Assessment Method |
|---|--------------|-----------|-------------------|
| 1 | Fire Risk | Forestry, afforestation | Historical fire frequency, climate projections, fire management |
| 2 | Pest/Disease Risk | Forestry, agriculture | Regional pest history, species vulnerability, management practices |
| 3 | Political Risk | All (country-dependent) | Country governance index, political stability, rule of law |
| 4 | Market Risk | All | Carbon price volatility, project economics, abandonment risk |
| 5 | Geological Risk | CCS, enhanced weathering | Reservoir integrity, seismic risk, monitoring adequacy |
| 6 | Technological Risk | DACCS, BECCS, biochar | Technology maturity (TRL), operational track record |
| 7 | Regulatory Risk | All | Policy changes, carbon market regulation, land tenure |
| 8 | Force Majeure | All | Natural disaster frequency, insurance coverage |

**Permanence Classification:**
- **Permanent** (1000+ years): Geological CCS, mineralization, DACCS with geological storage
- **Long-term** (100-999 years): Enhanced weathering, biochar, deep soil carbon
- **Medium-term** (40-99 years): Afforestation with buffer, managed forests
- **Short-term** (20-39 years): Soil carbon, grassland restoration
- **Temporary** (<20 years): Annual crop changes, short-rotation forestry

**Output:** Per-credit reversal risk score (1-5), recommended buffer pool %, and portfolio-level permanence-weighted assessment.

---

## 5. Agent Dependencies

### 5.1 MRV Agents (30)

All 30 AGENT-MRV agents are available as dependencies via `mrv_bridge.py`:
- **Scope 1 (8):** MRV-001 through MRV-008 (Stationary Combustion, Refrigerants, Mobile Combustion, Process Emissions, Fugitive Emissions, Land Use, Waste Treatment, Agricultural)
- **Scope 2 (5):** MRV-009 through MRV-013 (Location-Based, Market-Based, Steam/Heat, Cooling, Dual Reporting)
- **Scope 3 (15):** MRV-014 through MRV-028 (Categories 1-15)
- **Cross-cutting (2):** MRV-029 (Scope 3 Category Mapper), MRV-030 (Audit Trail)

### 5.2 Decarbonization Agents (21)

All 21 DECARB-X agents via `decarb_bridge.py`:
- DECARB-X-001: Abatement Options Library (500+ options)
- DECARB-X-002: MACC Generator
- DECARB-X-005: Investment Prioritization
- DECARB-X-007: Implementation Roadmap
- DECARB-X-008: Avoided Emissions Calculator
- DECARB-X-009: Carbon Intensity Tracker
- DECARB-X-015: Offset Strategy Agent
- DECARB-X-018: Progress Monitoring Agent
- DECARB-X-020: Cost-Benefit Analyzer
- (Remaining DECARB-X agents available as needed)

### 5.3 Application Dependencies

- GL-GHG-APP: GHG inventory management, base year, aggregation
- GL-ISO14064-APP: ISO 14064-1 aligned quantification and reporting
- GL-CDP-APP: CDP questionnaire alignment (C-FI carbon credits section)
- GL-TCFD-APP: TCFD metrics and targets alignment

### 5.4 Data Agents (20)

All 20 AGENT-DATA agents via `data_bridge.py` for data intake and quality.

### 5.5 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema, units, audit, etc.

### 5.6 Optional Pack Dependencies

- PACK-021 Net Zero Starter Pack: Baseline, gap analysis, offset portfolio via `pack021_bridge.py`
- PACK-023 SBTi Alignment Pack: SBTi target alignment for dual-claim strategies via `pack023_bridge.py`

---

## 6. Performance Targets

| Metric | Target |
|--------|--------|
| Footprint calculation (S1+S2+S3) | <30 minutes |
| Carbon management plan generation | <10 minutes |
| Credit quality assessment (100 credits) | <5 minutes |
| Portfolio optimization (500 credit options) | <10 minutes |
| Registry retirement processing (single) | <2 minutes |
| Neutralization balance calculation | <1 minute |
| Claims substantiation (35 criteria) | <5 minutes |
| Verification package assembly | <15 minutes |
| Full carbon neutral workflow (end-to-end) | <4 hours |
| Annual cycle renewal | <2 hours |
| Memory ceiling | 4096 MB |
| Cache hit target | 70% |
| Max facilities | 1,000 |
| Max credit line items | 10,000 |
| Max registries concurrent | 5 |

---

## 7. Security Requirements

- JWT RS256 authentication
- RBAC with 7 roles: `carbon_neutral_manager`, `sustainability_analyst`, `credit_portfolio_manager`, `procurement_officer`, `finance_reviewer`, `external_verifier`, `admin`
- AES-256-GCM encryption at rest for all emission and credit data
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs
- Full audit trail per SEC-005
- Credit portfolio access controls (read/write separation)
- Registry API credentials encrypted via Vault (SEC-006)

---

## 8. Database Migrations

Inherits platform migrations V001-V128. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V083-PACK024-001 | `cn_footprints` | Carbon footprint records with scope boundary and subject definition |
| V083-PACK024-002 | `cn_management_plans` | Carbon management plans with abatement actions and timelines |
| V083-PACK024-003 | `cn_credit_assessments` | Credit quality assessments with 12-dimension ICVCM scoring |
| V083-PACK024-004 | `cn_credit_portfolio` | Carbon credit portfolio entries with registry, serial, vintage, quality |
| V083-PACK024-005 | `cn_retirements` | Registry retirement records with confirmation receipts |
| V083-PACK024-006 | `cn_neutralization_balance` | Neutralization balance reconciliation records |
| V083-PACK024-007 | `cn_declarations` | Carbon neutral declaration records with compliance status |
| V083-PACK024-008 | `cn_verification_packages` | Verification evidence packages with provenance chains |
| V083-PACK024-009 | `cn_annual_cycles` | Annual cycle tracking with phase status and timeline |
| V083-PACK024-010 | `cn_permanence_assessments` | Credit permanence risk assessments with reversal scoring |

---

## 9. File Structure

```
packs/net-zero/PACK-024-carbon-neutral/
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
      organization_full.yaml
      organization_operational.yaml
      product_service.yaml
      event.yaml
      manufacturing.yaml
      services.yaml
      retail.yaml
      sme_simplified.yaml
  engines/
    __init__.py
    footprint_quantification_engine.py
    carbon_management_plan_engine.py
    credit_quality_engine.py
    portfolio_optimization_engine.py
    registry_retirement_engine.py
    neutralization_balance_engine.py
    claims_substantiation_engine.py
    verification_package_engine.py
    annual_cycle_engine.py
    permanence_risk_engine.py
  workflows/
    __init__.py
    carbon_neutral_onboarding_workflow.py
    reduction_planning_workflow.py
    credit_sourcing_workflow.py
    portfolio_management_workflow.py
    declaration_workflow.py
    verification_workflow.py
    annual_renewal_workflow.py
    full_carbon_neutral_workflow.py
  templates/
    __init__.py
    carbon_neutral_declaration.py
    footprint_report.py
    carbon_management_plan_report.py
    credit_quality_report.py
    portfolio_summary_report.py
    neutralization_balance_report.py
    claims_compliance_report.py
    verification_package_report.py
    annual_progress_report.py
    public_disclosure_report.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_bridge.py
    ghg_app_bridge.py
    decarb_bridge.py
    offset_bridge.py
    registry_bridge.py
    reporting_bridge.py
    pack021_bridge.py
    pack023_bridge.py
    data_bridge.py
    health_check.py
    setup_wizard.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_footprint_engine.py
    test_carbon_mgmt_plan_engine.py
    test_credit_quality_engine.py
    test_portfolio_opt_engine.py
    test_registry_retirement_engine.py
    test_neutralization_balance_engine.py
    test_claims_substantiation_engine.py
    test_verification_package_engine.py
    test_annual_cycle_engine.py
    test_permanence_risk_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_e2e.py
    test_orchestrator.py
```

---

## 10. Testing Requirements

| Test Type | Coverage Target | Scope |
|-----------|-----------------|-------|
| Unit Tests | >90% line coverage | All 10 engines, all config models, all presets |
| Workflow Tests | >85% | All 8 workflows with synthetic data |
| Template Tests | 100% | All 10 templates in 3 formats (MD, HTML, JSON) |
| Integration Tests | >80% | All 12 integrations with mock agents and registries |
| E2E Tests | Core happy path | Full pipeline from footprint to declaration |
| Claims Validation Tests | 100% | All 35 ISO 14068-1/PAS 2060 criteria |
| Credit Quality Tests | 100% | All 12 ICVCM dimensions with edge cases |
| Registry Mock Tests | 100% | All 5 registry integrations with mock APIs |
| Permanence Risk Tests | >90% | All 8 risk categories with synthetic projects |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 550+ tests (50-60 per engine, 20-30 integration, 15-20 E2E)

---

## 11. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-18 |
| Phase 2 | Engine implementation (10 engines) | 2026-03-19 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-19 |
| Phase 4 | Template implementation (10 templates) | 2026-03-19 |
| Phase 5 | Integration implementation (12 integrations) | 2026-03-19 |
| Phase 6 | Test suite (550+ tests) | 2026-03-19 |
| Phase 7 | Documentation & Release | 2026-03-19 |

---

## 12. Future Roadmap

- **PACK-025: Product Carbon Neutrality Pack** -- ISO 14068-2 product-level carbon neutrality, LCA integration, product carbon footprint (PCF), consumer-facing labels
- **PACK-026: Carbon Neutral Assurance Pack** -- Reasonable assurance (ISAE 3410) preparation, auditor workflow, evidence chain automation, materiality assessment
- **PACK-027: VCM Intelligence Pack** -- Carbon credit market analytics, price forecasting, supply/demand modeling, registry trend analysis, credit vintage optimization

---

*Document Version: 1.0.0 | Last Updated: 2026-03-18 | Status: Draft*
