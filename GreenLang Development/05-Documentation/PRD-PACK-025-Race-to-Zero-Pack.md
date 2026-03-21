# PRD-PACK-025: Race to Zero Pack

**Pack ID:** PACK-025-race-to-zero
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

The **Race to Zero** is the UN-backed global campaign rallying non-state actors -- businesses, cities, regions, investors, and universities -- to take rigorous, immediate action to halve global emissions by 2030 and achieve net zero by 2050 at the latest. Launched by the UNFCCC Climate Champions ahead of COP26, Race to Zero has mobilized over 11,000 entities representing nearly 25% of global CO2 emissions and over 50% of GDP. Despite this unprecedented mobilization, participants face significant operational challenges across the full campaign lifecycle:

1. **Starting Line Criteria complexity**: The Race to Zero "Interpretation Guide" (updated June 2022) defines four mandatory Starting Line Criteria -- Pledge, Plan, Proceed, Publish -- each with detailed sub-requirements. Participants must demonstrate compliance within 12 months of joining, yet many lack systematic tools to self-assess readiness. Partner initiatives (SBTi, CDP, ICLEI, C40, GFANZ) have varying alignment mechanisms, creating confusion about which actions satisfy which criteria.

2. **Interim target rigor**: Race to Zero requires interim targets covering at least 2030 aligned with halving emissions (approximately 50% absolute reduction from a recent baseline). The June 2022 update tightened requirements: targets must cover all scopes, use science-based methodologies, and demonstrate a "fair share" contribution to the 1.5C pathway. Organizations struggle to translate global carbon budgets into entity-level 2030 targets, especially for Scope 3-heavy profiles.

3. **Action plan publication**: Participants must publish a climate action plan within 12 months of joining, specifying concrete actions with timelines, milestones, and resource allocation. Most organizations produce vague qualitative plans that fail the credibility test applied by the High-Level Expert Group (HLEG) on the Net-Zero Emissions Commitments of Non-State Entities.

4. **Annual reporting obligations**: Race to Zero requires annual progress reporting, typically through partner initiative disclosure channels (e.g., CDP for corporates, GFANZ for financial institutions, C40 for cities). Tracking progress across multiple reporting channels while maintaining consistency is operationally burdensome and error-prone.

5. **Credibility under HLEG scrutiny**: The UN Secretary-General's High-Level Expert Group (HLEG) report "Integrity Matters" (November 2022) established 10 recommendations with detailed criteria for credible net-zero pledges. Race to Zero has incorporated HLEG recommendations into its minimum criteria, requiring participants to demonstrate: no new fossil fuel capacity, no lobbying against climate policy, transparent annual reporting, and genuine short-term emission reductions (not just long-term commitments). Self-assessing against HLEG criteria is complex and requires deep understanding of the 10 recommendations.

6. **Sector pathway alignment**: Race to Zero encourages participants to align with sector-specific decarbonization pathways from recognized sources (IEA Net Zero by 2050, IPCC AR6 WG3 mitigation pathways, sector-specific initiatives). Different sectors face fundamentally different decarbonization challenges (heavy industry vs. services, power generation vs. transport), and mapping entity-level plans to sector pathways requires specialized analytical tools.

7. **Partnership network navigation**: Race to Zero operates through a network of 40+ partner initiatives ("accelerators") including SBTi, CDP, C40, ICLEI, We Mean Business Coalition, GFANZ, The Climate Pledge, and others. Each partner has its own requirements, reporting formats, and timelines. Participants must understand which partner(s) they join through, how partner requirements map to Race to Zero criteria, and how to avoid duplicative reporting.

8. **Non-state actor diversity**: Race to Zero encompasses corporates, financial institutions, cities, regions/states, universities, and healthcare organizations. Each actor type has distinct emission profiles, governance structures, reporting frameworks, and sector pathway relevance. A one-size-fits-all approach fails; actor-type-specific guidance and tooling is essential.

### 1.2 Solution Overview

PACK-025 is the **Race to Zero Pack** -- the fifth pack in the "Net Zero Packs" category. It provides a comprehensive Race to Zero campaign lifecycle management solution covering pledge commitment, Starting Line Criteria assessment, interim target validation, action plan generation, annual progress tracking, sector pathway alignment, partnership scoring, credibility assessment against HLEG criteria, campaign reporting, and overall race readiness scoring.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the full Race to Zero lifecycle: from initial pledge commitment through ongoing annual participation with credibility assurance.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-025 Race to Zero Pack |
|-----------|-------------------------------|----------------------------|
| Time to Starting Line compliance | 300-600 hours | <30 hours (10-20x faster) |
| Pledge eligibility assessment | Ad hoc review of campaign criteria | Automated multi-criteria eligibility validation |
| Starting Line criteria check | Manual reading of Interpretation Guide | 20-criterion automated assessment with evidence mapping |
| Interim target validation | Qualitative alignment claim | Deterministic 1.5C pathway validation with scope coverage |
| Action plan quality | Vague qualitative plans | Quantified action plan with milestones, costs, and abatement impact |
| HLEG credibility assessment | Manual review of 10 recommendations | Automated 10-recommendation assessment with 45+ sub-criteria |
| Annual reporting | Manual multi-channel submission | Automated report generation aligned to partner requirements |
| Sector pathway alignment | Qualitative sector reference | Quantified sector pathway mapping with gap-to-benchmark analysis |
| Partnership navigation | Manual research of 40+ partners | Automated partner matching with requirement crosswalk |
| Audit trail | None | SHA-256 provenance, full calculation lineage |

### 1.4 Race to Zero Campaign Structure

| Element | Description |
|---------|-------------|
| **Campaign Owner** | UNFCCC Climate Champions (appointed by COP Presidency) |
| **Launch** | June 2020 (ahead of COP26, Glasgow) |
| **Participants** | 11,000+ entities: corporates, FIs, cities, regions, universities, healthcare |
| **Coverage** | ~25% of global CO2 emissions, ~50% of GDP |
| **Partner Initiatives** | 40+ accelerators (SBTi, CDP, C40, GFANZ, ICLEI, We Mean Business, etc.) |
| **Core Commitment** | Net zero by 2050 at latest, halve emissions by 2030 |
| **Governance** | Expert Peer Review Group (EPRG), High-Level Expert Group (HLEG) |
| **Key Document** | "Interpretation Guide" (June 2022 update) |

### 1.5 Target Users

**Primary:**
- Sustainability managers preparing Race to Zero applications for corporates
- City and regional climate officers joining through C40 or ICLEI
- Financial institution sustainability teams joining through GFANZ
- Organizations already in Race to Zero tracking annual compliance

**Secondary:**
- University sustainability officers joining through Second Nature or similar
- Board members reviewing net-zero pledge commitments
- Sustainability consultants guiding clients into Race to Zero
- Partner initiative administrators validating member compliance
- Investor relations teams communicating campaign participation credibility

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to Starting Line compliance | <30 hours (vs. 300+ manual) | Time from pledge to validated Starting Line |
| Pledge eligibility accuracy | 100% | Validated against Race to Zero criteria checker |
| HLEG credibility assessment coverage | All 10 recommendations, 45+ sub-criteria | Criteria covered / total HLEG criteria |
| Interim target validation accuracy | 100% match with 1.5C pathway | Cross-validated against SBTi/IEA benchmarks |
| Action plan completeness | 100% of required elements present | Automated checklist against Interpretation Guide |
| Sector pathway coverage | 25+ sectors mapped | Number of sectors with quantified pathways |
| Customer NPS | >50 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| Race to Zero Campaign | UNFCCC Climate Champions (2020, updated 2022) | Core campaign framework; Starting Line Criteria, pledge requirements |
| Race to Zero Interpretation Guide | UNFCCC (June 2022 update) | Detailed Starting Line sub-criteria and compliance guidance |
| HLEG "Integrity Matters" Report | UN Secretary-General HLEG (November 2022) | 10 recommendations for credible net-zero pledges; incorporated into R2Z criteria |
| Paris Agreement | UNFCCC (2015) | 1.5C temperature target; global stocktake; nationally determined contributions |
| IPCC AR6 WG3 | IPCC (2022) | Mitigation pathways; sector decarbonization; carbon budget; 43% reduction by 2030 |
| SBTi Corporate Net-Zero Standard | SBTi v1.3 (2024) | Science-based target methodology; partner initiative alignment |

### 2.2 Partner Initiative Standards

| Partner Initiative | Reference | Pack Relevance |
|-------------------|-----------|----------------|
| Science Based Targets initiative (SBTi) | SBTi Corporate Manual V5.3 | Corporate near-term/long-term/net-zero target validation |
| CDP Climate Change Questionnaire | CDP (2024) | Annual disclosure channel for corporate participants |
| C40 Cities Climate Leadership Group | C40 Deadline 2020 Program | City-level climate action plans and reporting |
| ICLEI Local Governments for Sustainability | ICLEI GreenClimateCities | Regional/local government climate action framework |
| Glasgow Financial Alliance for Net Zero (GFANZ) | GFANZ Guidance (2022) | Financial institution transition plans and portfolio targets |
| We Mean Business Coalition | WMB (ongoing) | Corporate climate action coordination |
| The Climate Pledge | Amazon/Global Optimism (2019) | Corporate net-zero commitment by 2040 |
| Second Nature | Presidents' Climate Leadership Commitments | University/higher education climate commitments |

### 2.3 Sector Pathway Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| IEA Net Zero by 2050 Roadmap | IEA (2021, updated 2023) | Sector-specific milestones and benchmarks |
| IEA World Energy Outlook | IEA (annual) | Energy system transition pathways |
| TPI Global Climate Transition Centre | TPI (2024) | Sector benchmarks and carbon performance assessments |
| ACT (Assessing low-Carbon Transition) | ADEME/CDP (2023) | Sector-specific transition assessment methodology |
| Mission Possible Partnership | MPP (2022) | Hard-to-abate sector pathways (steel, cement, shipping, aviation, trucking, chemicals, aluminium) |
| CRREM | CRREM (2023) | Real estate sector decarbonization pathways |

### 2.4 Supporting Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1+2 GHG inventory methodology |
| GHG Protocol Scope 3 Standard | WRI/WBCSD (2011) | Scope 3 methodology for full value chain coverage |
| ISO 14064-1:2018 | ISO | Organization-level GHG quantification |
| ESRS E1 Climate Change | EU (2023) | E1-4 GHG reduction targets alignment |
| TCFD Recommendations | FSB/TCFD (2017) | Transition planning, metrics & targets |
| PCAF Global Standard | PCAF (2022) | Financed emissions for financial institution participants |
| ISO 14068-1:2023 | ISO | Carbon neutrality quantification (complementary) |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Race to Zero lifecycle calculation engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, and submission templates |
| Integrations | 12 | Agent, app, campaign, and partner bridges |
| Presets | 8 | Actor-type and sector-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `pledge_commitment_engine.py` | Validate pledge eligibility, commitment criteria, and actor-type-specific requirements. Assesses whether the entity qualifies for Race to Zero participation, validates the pledge statement against campaign requirements, checks partner initiative membership, and confirms commitment to net zero by 2050 at latest with interim 2030 target. |
| 2 | `starting_line_engine.py` | Assess compliance with the four Starting Line Criteria (Pledge, Plan, Proceed, Publish) with 20 sub-criteria from the Interpretation Guide. Each criterion maps to specific evidence requirements, and the engine produces a pass/fail/partial assessment with gap identification and remediation guidance for each sub-criterion. |
| 3 | `interim_target_engine.py` | Validate 2030 interim targets against 1.5C pathway requirements. Confirms approximately 50% absolute emission reduction by 2030 from a recent baseline, validates scope coverage (Scope 1+2+3), checks science-based methodology alignment, assesses "fair share" contribution, and validates target boundary and ambition level. |
| 4 | `action_plan_engine.py` | Generate and validate climate action plans per Race to Zero requirements. Produces quantified action plans with specific decarbonization levers, timelines (12-month publication deadline), milestones, resource allocation, abatement impact (tCO2e), and cost estimates. Validates plan completeness against HLEG and Interpretation Guide criteria. |
| 5 | `progress_tracking_engine.py` | Track annual progress against interim and long-term targets. Calculates year-over-year emission changes, assesses trajectory alignment with 2030 and 2050 targets, identifies on-track/off-track status, generates variance analysis, triggers corrective action recommendations, and produces annual progress disclosures. |
| 6 | `sector_pathway_engine.py` | Map entity-level decarbonization plans to sector-specific pathways from 25+ sectors. Sources pathway data from IEA NZE, IPCC AR6 WG3, TPI, MPP, and ACT. Calculates gap-to-benchmark for the entity's sector, identifies sector-specific milestones (e.g., coal phase-out dates, EV penetration targets, renewable energy share), and assesses pathway credibility. |
| 7 | `partnership_scoring_engine.py` | Assess collaboration quality and partner initiative alignment. Maps partner initiative requirements to Race to Zero criteria, scores entity participation across multiple partnerships, identifies reporting synergies and duplications, evaluates engagement quality (active vs. passive membership), and tracks partner-specific compliance deadlines. |
| 8 | `campaign_reporting_engine.py` | Generate Race to Zero annual disclosure reports aligned to campaign and partner requirements. Produces structured reports covering emissions inventory, target progress, action plan implementation, partnership engagement, and forward-looking commitments. Maps outputs to partner-specific reporting formats (CDP, GFANZ, C40, etc.). |
| 9 | `credibility_assessment_engine.py` | Evaluate pledge credibility against the HLEG "Integrity Matters" 10 recommendations with 45+ sub-criteria. Covers: net-zero pledge quality, interim target ambition, voluntary credit use, lobbying alignment, just transition considerations, financial commitment, reporting transparency, scope of pledge, internal governance, and fossil fuel phase-out commitment. |
| 10 | `race_readiness_engine.py` | Overall readiness scoring for Race to Zero campaign participation. Aggregates results from all other engines into a composite readiness score across 8 dimensions: pledge strength, Starting Line compliance, target ambition, action plan quality, progress trajectory, sector alignment, partnership engagement, and HLEG credibility. Produces a 0-100 score with RAG status and prioritized improvement actions. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `pledge_onboarding_workflow.py` | 5: EligibilityCheck -> PledgeFormulation -> PartnerSelection -> PledgeSubmission -> ConfirmationTracking | Guided onboarding for new Race to Zero participants from eligibility to confirmed pledge |
| 2 | `starting_line_assessment_workflow.py` | 4: CriteriaMapping -> EvidenceCollection -> ComplianceCheck -> GapReport | Full Starting Line Criteria assessment with evidence gathering and gap analysis |
| 3 | `action_planning_workflow.py` | 5: EmissionsProfile -> LeverIdentification -> ActionPrioritization -> PlanDrafting -> PlanValidation | End-to-end climate action plan development meeting Race to Zero publication requirements |
| 4 | `annual_reporting_workflow.py` | 5: DataCollection -> ProgressCalculation -> VarianceAnalysis -> ReportGeneration -> PartnerSubmission | Annual progress reporting cycle with multi-partner output formatting |
| 5 | `sector_pathway_workflow.py` | 4: SectorClassification -> BenchmarkMapping -> GapAnalysis -> RoadmapAlignment | Sector pathway alignment assessment and roadmap generation |
| 6 | `partnership_engagement_workflow.py` | 4: PartnerInventory -> RequirementCrosswalk -> SynergyAnalysis -> EngagementPlan | Partner initiative optimization and engagement planning |
| 7 | `credibility_review_workflow.py` | 5: HLEGMapping -> EvidenceGathering -> RecommendationAssess -> GapIdentification -> RemediationPlan | HLEG credibility assessment with remediation planning |
| 8 | `full_race_to_zero_workflow.py` | 8: Pledge -> StartingLine -> Targets -> ActionPlan -> SectorAlign -> Partners -> Credibility -> Readiness | End-to-end Race to Zero lifecycle assessment |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `pledge_commitment_letter.py` | MD, HTML, PDF, JSON | Formal Race to Zero pledge commitment letter for submission to campaign secretariat |
| 2 | `starting_line_checklist.py` | MD, HTML, JSON | Starting Line Criteria compliance checklist with 20 sub-criteria evidence mapping |
| 3 | `action_plan_document.py` | MD, HTML, PDF, JSON | Complete climate action plan document meeting Interpretation Guide publication requirements |
| 4 | `annual_progress_report.py` | MD, HTML, JSON | Annual progress disclosure report with emissions trajectory and target tracking |
| 5 | `sector_pathway_roadmap.py` | MD, HTML, JSON | Sector-specific decarbonization roadmap with benchmark alignment visualization |
| 6 | `partnership_framework.py` | MD, HTML, JSON | Partnership engagement framework with requirement crosswalk and synergy analysis |
| 7 | `credibility_assessment.py` | MD, HTML, JSON | HLEG credibility assessment report with 10-recommendation scoring matrix |
| 8 | `campaign_submission.py` | MD, HTML, PDF, JSON | Complete Race to Zero campaign submission package for partner initiative |
| 9 | `disclosure_dashboard.py` | MD, HTML, JSON | Real-time disclosure dashboard showing campaign compliance status across all dimensions |
| 10 | `race_to_zero_certificate.py` | MD, HTML, PDF, JSON | Race to Zero readiness certificate with composite score and dimension breakdown |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline with retry, provenance, conditional partner/sector phases |
| 2 | `mrv_bridge.py` | Routes to all 30 MRV agents for emission calculations (Scope 1+2+3) |
| 3 | `ghg_app_bridge.py` | Bridges to GL-GHG-APP for inventory management and base year |
| 4 | `sbti_app_bridge.py` | Bridges to GL-SBTi-APP for science-based target validation |
| 5 | `decarb_bridge.py` | Bridges to 21 DECARB-X agents for reduction planning and action identification |
| 6 | `taxonomy_bridge.py` | Bridges to EU Taxonomy alignment for climate CapEx validation |
| 7 | `data_bridge.py` | Bridges to 20 DATA agents for data intake and quality |
| 8 | `unfccc_bridge.py` | Bridges to UNFCCC/Race to Zero campaign data feeds for partner criteria, participant registry, and campaign updates |
| 9 | `cdp_bridge.py` | Bridges to GL-CDP-APP for CDP Climate Change questionnaire alignment (corporate reporting channel) |
| 10 | `gfanz_bridge.py` | Bridges to GFANZ transition plan framework for financial institution participants |
| 11 | `setup_wizard.py` | 8-step guided Race to Zero configuration wizard with actor-type selection |
| 12 | `health_check.py` | 22-category system verification covering all engines, workflows, and partner bridges |

### 3.6 Presets

| # | Preset | Actor Type | Key Characteristics |
|---|--------|-----------|---------------------|
| 1 | `corporate_commitment.yaml` | Large Corporate (>1000 employees) | Full Scope 1+2+3, SBTi partner pathway, CDP reporting, HLEG full assessment, 25+ sector pathways |
| 2 | `financial_institution.yaml` | Bank/Insurance/Asset Manager | GFANZ partner pathway, PCAF financed emissions, portfolio targets, Scope 3 Cat 15 focus |
| 3 | `city_municipality.yaml` | City/Municipality | C40/ICLEI partner pathway, community-wide inventory (GPC), transport/buildings/waste sectors |
| 4 | `region_state.yaml` | Region/State/Province | Under2 Coalition pathway, sub-national inventory, energy/land use sectors, policy levers |
| 5 | `sme_business.yaml` | SME (<250 employees) | Simplified 6-engine flow, SME Climate Hub pathway, spend-based Scope 3, streamlined reporting |
| 6 | `high_emitter.yaml` | Heavy Industry/Energy/Mining | SDA sector pathways mandatory, process emissions focus, fossil fuel phase-out assessment, CCS pathway |
| 7 | `service_sector.yaml` | Professional/Financial/Technology Services | Low Scope 1, high Scope 3 (Cat 1, 6, 7), ACA pathway, RE procurement, supply chain engagement |
| 8 | `manufacturing_sector.yaml` | General Manufacturing | Mixed scopes, energy efficiency focus, SDA or ACA pathway, supplier engagement for Scope 3 |

---

## 4. Engine Specifications

### 4.1 Engine 1: Pledge Commitment Engine

**Purpose:** Validate pledge eligibility, commitment quality, and actor-type requirements for Race to Zero participation.

**Eligibility Criteria:**

| # | Criterion | Description | Actor Types |
|---|-----------|-------------|-------------|
| 1 | Net-zero commitment by 2050 | Pledge to reach net zero no later than 2050 | All |
| 2 | Partner initiative membership | Join through a recognized Race to Zero partner initiative | All |
| 3 | Interim 2030 target | Commit to an interim target aligned with halving emissions by 2030 | All |
| 4 | Action plan commitment | Commit to publish a climate action plan within 12 months | All |
| 5 | Annual reporting commitment | Commit to report progress annually through partner channels | All |
| 6 | Scope coverage | Pledge covers all material emission scopes | Corporates, FIs |
| 7 | Governance endorsement | Board or senior leadership endorsement of pledge | All |
| 8 | Public disclosure | Commitment to publicly disclose pledge and progress | All |

**Pledge Quality Assessment:**
- **Strong:** All 8 criteria met, specific targets stated, governance approved, public commitment made
- **Adequate:** Core criteria met (1-5), minor gaps in governance or disclosure
- **Weak:** Missing interim target, vague commitment language, no governance endorsement
- **Ineligible:** No net-zero commitment, no partner initiative, fundamentally misaligned

**Key Models:**
- `PledgeInput` - Entity profile, actor type, partner initiative, commitment statement, governance records
- `PledgeResult` - Eligibility status (eligible/conditional/ineligible), pledge quality score, criterion-level assessment, gap list
- `PledgeCriterion` - Single criterion assessment with pass/fail/partial and evidence reference
- `PartnerAlignment` - Partner initiative mapping with requirement crosswalk

### 4.2 Engine 2: Starting Line Engine

**Purpose:** Assess compliance with the four Starting Line Criteria (Pledge, Plan, Proceed, Publish) per the June 2022 Interpretation Guide.

**Four Starting Line Criteria with 20 Sub-Criteria:**

| Criterion | Sub-Criteria | Requirements |
|-----------|-------------|--------------|
| **PLEDGE** | SL-P1: Net-zero target | Commit to net zero by 2050 at latest, covering all scopes |
| | SL-P2: Interim target | Set interim target for 2030 reflecting ~50% absolute reduction |
| | SL-P3: Science-based methodology | Target uses recognized science-based methodology (SBTi, IEA, IPCC) |
| | SL-P4: Fair share | Target represents a "fair share" of global effort (equity consideration) |
| | SL-P5: Scope coverage | Covers Scope 1, 2, and material Scope 3 (or community-wide for cities) |
| **PLAN** | SL-A1: Action plan published | Climate action plan published within 12 months of joining |
| | SL-A2: Quantified actions | Plan includes specific, quantified decarbonization actions |
| | SL-A3: Timeline and milestones | Actions have defined timelines and measurable milestones |
| | SL-A4: Resource allocation | Plan specifies resources (financial, human, technical) for implementation |
| | SL-A5: Sector alignment | Actions aligned with relevant sector pathway(s) |
| **PROCEED** | SL-R1: Immediate action | Demonstrable action taken (not just planned) within first year |
| | SL-R2: Emission reductions | Evidence of actual emission reductions or genuine reduction trajectory |
| | SL-R3: Investment commitment | Financial resources allocated to decarbonization actions |
| | SL-R4: Governance integration | Climate targets integrated into corporate/organizational governance |
| | SL-R5: No contradictory action | No actions contradicting climate commitment (e.g., new fossil fuel investment) |
| **PUBLISH** | SL-D1: Annual reporting | Annual progress reported through partner initiative channels |
| | SL-D2: Emissions disclosure | GHG emissions disclosed publicly (Scope 1, 2, material Scope 3) |
| | SL-D3: Target progress | Progress against targets reported with quantitative metrics |
| | SL-D4: Plan updates | Action plan updated and re-published annually |
| | SL-D5: Transparency | Methodology, assumptions, and limitations transparently documented |

**Assessment Output:**
- Per-criterion: PASS / FAIL / PARTIAL / NOT_APPLICABLE
- Overall Starting Line status: COMPLIANT / NON_COMPLIANT / PARTIALLY_COMPLIANT
- Gap list with remediation guidance and estimated effort per gap
- Evidence requirements checklist

**Key Models:**
- `StartingLineInput` - Entity data, pledge records, action plan, emissions data, reporting records
- `StartingLineResult` - Overall compliance status, per-criterion results, gap list, remediation plan
- `CriterionAssessment` - Single sub-criterion with status, evidence reference, gap description
- `RemediationItem` - Gap with recommended action, effort estimate, priority, deadline

### 4.3 Engine 3: Interim Target Engine

**Purpose:** Validate 2030 interim targets against Race to Zero requirements and 1.5C pathway alignment.

**Validation Rules:**

| Rule | Description | Threshold |
|------|-------------|-----------|
| Absolute reduction | ~50% absolute emissions reduction by 2030 from baseline | >= 42% (IPCC minimum for 1.5C) |
| Baseline year | Recent baseline year (no older than 2019) | >= 2015 (preferred >= 2019) |
| Scope coverage | Covers all material emission scopes | Scope 1+2 mandatory, Scope 3 >= 67% |
| Science-based methodology | Uses recognized science-based target methodology | SBTi ACA/SDA, IEA NZE, IPCC SR1.5 |
| Annual reduction rate | Annualized linear reduction rate consistent with 2030 target | >= 4.2%/yr (1.5C aligned) |
| No cherry-picking | Target includes high-emission sources, not just easy wins | Coverage >= 95% of Scope 1+2 |
| Fair share equity | Reflects capability and historical responsibility considerations | Qualitative + quantitative assessment |

**1.5C Pathway Alignment Check:**
- IPCC AR6 WG3: 43% reduction in CO2 by 2030 (relative to 2019)
- IEA NZE: sector-specific 2030 milestones
- SBTi: 4.2%/yr absolute contraction (1.5C), 2.5%/yr (Well-Below-2C)
- Global carbon budget: remaining 1.5C budget (~400 GtCO2 from 2023) allocation

**Key Models:**
- `InterimTargetInput` - Baseline emissions (by scope), base year, target year (2030), target value, methodology
- `InterimTargetResult` - Alignment status (aligned/partially/misaligned), pathway comparison, gap to 1.5C, annual reduction rate
- `PathwayComparison` - Entity target vs. sector/global pathway with deviation analysis
- `FairShareAssessment` - Equity-weighted assessment of target ambition

### 4.4 Engine 4: Action Plan Engine

**Purpose:** Generate and validate climate action plans meeting Race to Zero publication requirements.

**Action Plan Structure (per Interpretation Guide):**

| Section | Required Content | Validation Criteria |
|---------|-----------------|-------------------|
| 1. Emissions Profile | Current GHG inventory by scope | Complete Scope 1+2, material Scope 3 categories |
| 2. Targets | Interim (2030) and long-term (2050) targets | Validated by interim_target_engine |
| 3. Reduction Actions | Specific decarbonization actions with quantified impact | >= 10 named actions with tCO2e impact |
| 4. Timeline | Phased implementation schedule | All actions dated, milestones defined through 2030 |
| 5. Resource Plan | Financial, human, and technical resources | Budget allocated per action, FTE commitment stated |
| 6. Sector Alignment | Alignment with relevant sector pathway | Sector pathway referenced, gap-to-benchmark quantified |
| 7. Scope 3 Strategy | Value chain emission reduction approach | Supplier engagement plan, category-specific strategies |
| 8. Governance | Climate governance structure and accountability | Board oversight, executive responsibility, incentive alignment |
| 9. Just Transition | Workforce and community transition considerations | Just transition principles integrated, stakeholder engagement |
| 10. Monitoring | Progress monitoring and reporting plan | KPIs defined, reporting frequency, corrective action triggers |

**Action Library Integration:** Via DECARB-X-001 (500+ abatement options), filtered by sector, scope, cost-effectiveness, and TRL.

**Plan Completeness Score:** 0-100 across 10 sections, weighted by HLEG emphasis areas.

**Key Models:**
- `ActionPlanInput` - Emissions profile, targets, sector, budget, constraints, actor type
- `ActionPlanResult` - Complete action plan document, completeness score, per-section assessment, HLEG alignment
- `DecarbonizationAction` - Single action with category, scope impact, abatement (tCO2e), cost ($/tCO2e), timeline, TRL
- `PlanSection` - Section with content, completeness score, gap list

### 4.5 Engine 5: Progress Tracking Engine

**Purpose:** Track annual progress against Race to Zero commitments and targets.

**Key Metrics Tracked:**

| Metric | Description | Frequency |
|--------|-------------|-----------|
| Absolute emissions (tCO2e) | Total Scope 1+2+3 emissions by year | Annual |
| Reduction vs. baseline (%) | Percentage reduction from baseline year | Annual |
| Trajectory alignment | On-track/off-track vs. 2030 interim target | Annual |
| Action plan implementation | % of planned actions initiated/completed | Annual |
| Annualized reduction rate | Current annual reduction rate vs. required | Annual |
| Budget remaining | Remaining emission budget to 2030 target | Annual |
| Scope 3 coverage | % of Scope 3 covered by reduction actions | Annual |
| Partnership compliance | Compliance with partner initiative requirements | Annual |

**Progress Status Assessment:**
- **On Track (GREEN):** Cumulative reductions at or ahead of linear pathway to 2030
- **Caution (AMBER):** Reductions within 10% of required trajectory, corrective action needed
- **Off Track (RED):** Reductions >10% behind required trajectory, significant intervention needed
- **Critical (BLACK):** No measurable progress or emissions increasing, campaign standing at risk

**Corrective Action Framework:**
- Variance decomposition: activity effect vs. intensity effect vs. structural change
- Gap acceleration: additional actions identified from DECARB-X library
- Budget reallocation recommendations
- Partner notification triggers

**Key Models:**
- `ProgressInput` - Current year emissions, prior year emissions, baseline, targets, action plan status
- `ProgressResult` - Status (RAG), trajectory analysis, variance decomposition, corrective actions, partner report
- `TrajectoryPoint` - Year, actual emissions, target emissions, gap, status
- `CorrectiveAction` - Recommended action with impact estimate, priority, timeline

### 4.6 Engine 6: Sector Pathway Engine

**Purpose:** Map entity-level decarbonization plans to sector-specific pathways for 25+ sectors.

**25+ Sector Pathways:**

| # | Sector | Primary Source | Key 2030 Milestone | Key 2050 Benchmark |
|---|--------|---------------|--------------------|--------------------|
| 1 | Power Generation | IEA NZE | 60% renewable share | Near-zero emissions |
| 2 | Oil & Gas | IEA NZE | No new exploration; -30% methane | Phase-out unabated fossil |
| 3 | Coal Mining | IEA NZE | No new coal mines | Full phase-out (OECD by 2030) |
| 4 | Steel | MPP / IEA | 10% near-zero steel production | 100% near-zero steel |
| 5 | Cement | MPP / IEA | 15% CO2 capture; clinker substitution | 0.12 tCO2e/tonne |
| 6 | Aluminium | IAI / IEA | 50% recycled content | 1.31 tCO2e/tonne |
| 7 | Chemicals | IEA / MPP | Feedstock diversification; energy efficiency | Sector-specific |
| 8 | Pulp & Paper | IEA | Fuel switching; efficiency | 0.175 tCO2e/tonne |
| 9 | Aviation | ICAO / MPP | 10% SAF; efficiency | 65% SAF; net-zero CO2 |
| 10 | Maritime Shipping | IMO / MPP | 5% zero-emission fuels | 100% zero-emission fuels |
| 11 | Road Transport (Light) | IEA NZE | 60% EV sales share | 100% zero-emission sales |
| 12 | Road Transport (Heavy) | MPP | 30% zero-emission sales | 100% zero-emission sales |
| 13 | Rail | IEA | 50% electrified | 100% electrified |
| 14 | Buildings (Commercial) | CRREM / IEA | Deep retrofit rate 2.5%/yr | 3.1 kgCO2e/m2 |
| 15 | Buildings (Residential) | CRREM / IEA | Heat pump deployment | 2.3 kgCO2e/m2 |
| 16 | Agriculture | IPCC / FAO | Methane reduction; precision agriculture | 30% reduction from 2020 |
| 17 | Food & Beverage | SBTi FLAG | Deforestation-free supply chains | Sector-specific |
| 18 | Retail | IEA / TPI | RE procurement; cold chain efficiency | Near-zero operations |
| 19 | Financial Services | GFANZ / NZBA | Portfolio alignment targets | Financed emissions net-zero |
| 20 | Technology / ICT | ITU / IEA | Data center efficiency PUE < 1.3 | 100% RE; near-zero |
| 21 | Healthcare | HCWH | Energy efficiency; anesthetic gas reduction | Sector net-zero |
| 22 | Higher Education | Second Nature | Campus energy transition | Carbon neutral campus |
| 23 | Waste Management | IPCC | Methane capture; circular economy | Near-zero landfill emissions |
| 24 | Water Utilities | IWA | Energy efficiency; renewable pumping | Near-zero operations |
| 25 | Telecommunications | GSMA | Network energy efficiency; RE | Net-zero operations |

**Key Features:**
- Entity-to-sector mapping based on ISIC/NACE/GICS classification
- Gap-to-benchmark calculation per sector milestone
- Multi-sector entities: weighted average across revenue-share sectors
- Pathway credibility scoring: conservative/moderate/aggressive alignment

**Key Models:**
- `SectorPathwayInput` - Entity sector(s), current performance metrics, decarbonization plans
- `SectorPathwayResult` - Sector mapping, gap-to-benchmark per milestone, pathway alignment score, roadmap
- `SectorBenchmark` - Sector milestone with year, metric, target value, source
- `PathwayGap` - Gap between entity performance and sector benchmark with timeline to close

### 4.7 Engine 7: Partnership Scoring Engine

**Purpose:** Assess collaboration quality, partner initiative alignment, and reporting efficiency.

**Partner Initiative Assessment:**

| Dimension | Weight | Assessment Criteria |
|-----------|--------|-------------------|
| Requirement alignment | 25% | Partner requirements vs. Race to Zero Starting Line mapping completeness |
| Reporting efficiency | 20% | Degree of data reuse across partner channels (avoid duplicate effort) |
| Engagement quality | 20% | Active participation (working groups, peer learning) vs. passive membership |
| Credibility contribution | 15% | Partner initiative recognition and standing in climate governance |
| Coverage completeness | 10% | Partner coverage of Race to Zero criteria (some cover only subset) |
| Timeline alignment | 10% | Partner reporting deadlines vs. Race to Zero annual cycle |

**40+ Partner Initiatives Mapped:**

| Category | Partners | Typical Entity Type |
|----------|----------|-------------------|
| Corporate | SBTi, The Climate Pledge, WMB, SME Climate Hub, Exponential Roadmap | Corporates of all sizes |
| Financial | GFANZ, NZBA, NZAM, NZAOA, NZIA, PCAF | Banks, asset managers, insurers |
| Cities | C40, ICLEI, Global Covenant of Mayors, CDP Cities | Cities, municipalities |
| Regions | Under2 Coalition, RegionsAdapt | States, provinces, regions |
| Universities | Second Nature, University Alliance | Higher education institutions |
| Healthcare | HCWH Health Care Climate Challenge | Hospitals, health systems |

**Partnership Synergy Score:** Quantifies the efficiency of the entity's partner portfolio -- how many Race to Zero requirements are covered, how much reporting overlap exists, and where gaps remain.

**Key Models:**
- `PartnershipInput` - Entity partner memberships, engagement history, reporting records
- `PartnershipResult` - Per-partner scores, synergy analysis, coverage gaps, optimization recommendations
- `PartnerAssessment` - Single partner with alignment score, engagement quality, reporting efficiency
- `SynergyAnalysis` - Cross-partner requirement coverage with overlap and gap identification

### 4.8 Engine 8: Campaign Reporting Engine

**Purpose:** Generate Race to Zero annual disclosure reports aligned to campaign and partner-specific formats.

**Report Sections (Campaign-Aligned):**

| Section | Content | Data Source |
|---------|---------|-------------|
| 1. Entity Profile | Organization description, sector, size, geography | Configuration |
| 2. Pledge Status | Net-zero commitment details, partner initiative(s) | Pledge engine |
| 3. Starting Line Compliance | 4-criterion compliance status | Starting Line engine |
| 4. Emissions Inventory | Scope 1+2+3 emissions by category | MRV bridge |
| 5. Target Progress | Interim (2030) and long-term (2050) target tracking | Progress engine |
| 6. Action Plan Summary | Key actions, implementation status, impact | Action Plan engine |
| 7. Sector Alignment | Sector pathway alignment assessment | Sector engine |
| 8. HLEG Credibility | Credibility assessment highlights | Credibility engine |
| 9. Partnership Engagement | Partner initiative participation summary | Partnership engine |
| 10. Forward Commitments | Next-year actions and milestones | Action Plan engine |

**Partner-Specific Output Mapping:**
- CDP: Maps to C4 (Targets), C6 (Emissions Data), C12 (Engagement) modules
- GFANZ: Maps to Transition Plan framework sections
- C40: Maps to Deadline 2020 reporting format
- SBTi: Maps to progress report format
- General: Universal Race to Zero disclosure template

**Key Models:**
- `ReportingInput` - All engine results, partner requirements, reporting period
- `ReportingResult` - Complete campaign report, partner-specific formatted outputs, submission readiness
- `ReportSection` - Single section with content, data completeness, partner mapping
- `PartnerFormat` - Partner-specific report formatting with field mapping

### 4.9 Engine 9: Credibility Assessment Engine

**Purpose:** Evaluate pledge credibility against the HLEG "Integrity Matters" 10 recommendations with 45+ sub-criteria.

**HLEG 10 Recommendations Assessment:**

| # | Recommendation | Sub-Criteria | Key Assessment Areas |
|---|---------------|-------------|---------------------|
| 1 | Announce net-zero pledge | 5 | Pledge specificity, timeline, scope coverage, public availability, governance approval |
| 2 | Set interim targets | 5 | 2030 target, science-based methodology, scope coverage, annual milestones, fair share |
| 3 | Implement transition plan | 5 | Quantified actions, resource allocation, timeline, sector alignment, technology pathway |
| 4 | Phase out fossil fuels | 4 | No new fossil fuel capacity, divestment policy, phase-out timeline, stranded asset risk |
| 5 | Use voluntary credits responsibly | 5 | Credits complement (not substitute) reductions, quality criteria, ICVCM alignment, transparency, retirement |
| 6 | Align lobbying with climate goals | 4 | Trade association audit, lobbying disclosure, policy alignment, no obstruction of climate legislation |
| 7 | Plan for a just transition | 5 | Workforce planning, community engagement, stakeholder consultation, distributional impacts, human rights |
| 8 | Increase transparency | 5 | Annual public reporting, methodology disclosure, assumption transparency, third-party verification, data accessibility |
| 9 | Invest in systemic change | 3 | Climate finance contribution, R&D investment in solutions, supply chain capacity building |
| 10 | Ensure governance and accountability | 4 | Board oversight, executive incentives, climate risk integration, accountability mechanisms |

**Credibility Scoring:**
- Per-recommendation score: 0-100 with per-sub-criterion breakdown
- Overall credibility score: weighted average (higher weight on Rec 1-3, 6, 8)
- Credibility tier: **HIGH** (>=80), **MODERATE** (60-79), **LOW** (40-59), **CRITICAL** (<40)

**Key Models:**
- `CredibilityInput` - Entity policies, actions, disclosures, lobbying records, governance structure
- `CredibilityResult` - Overall credibility tier, per-recommendation scores, sub-criterion assessments, improvement priorities
- `RecommendationAssessment` - Single HLEG recommendation with score, sub-criteria results, evidence gaps
- `LobbyingAlignmentCheck` - Trade association membership audit with climate policy alignment assessment

### 4.10 Engine 10: Race Readiness Engine

**Purpose:** Overall readiness scoring for Race to Zero campaign participation.

**8 Readiness Dimensions:**

| # | Dimension | Weight | Source Engine | Scoring Basis |
|---|-----------|--------|-------------|---------------|
| 1 | Pledge Strength | 12% | pledge_commitment_engine | Pledge quality score (eligibility + quality) |
| 2 | Starting Line Compliance | 18% | starting_line_engine | 20 sub-criteria compliance rate |
| 3 | Target Ambition | 15% | interim_target_engine | 1.5C pathway alignment and scope coverage |
| 4 | Action Plan Quality | 15% | action_plan_engine | Plan completeness score across 10 sections |
| 5 | Progress Trajectory | 12% | progress_tracking_engine | On-track/off-track status and trajectory quality |
| 6 | Sector Alignment | 10% | sector_pathway_engine | Gap-to-benchmark and pathway credibility |
| 7 | Partnership Engagement | 8% | partnership_scoring_engine | Partnership synergy score and engagement quality |
| 8 | HLEG Credibility | 10% | credibility_assessment_engine | HLEG credibility tier and score |

**Composite Race Readiness Score:**
- 0-100 composite with dimension breakdown
- RAG classification: GREEN (>=75), AMBER (50-74), RED (25-49), BLACK (<25)
- Rank ordering of improvement priorities based on gap-weighted impact
- Estimated timeline to full readiness

**Readiness Levels:**

| Level | Score | Status | Description |
|-------|-------|--------|-------------|
| Race Ready | 85-100 | Campaign-ready | Meets all Starting Line criteria; strong HLEG credibility |
| Approaching | 70-84 | Near-ready | Minor gaps; 1-3 months to full readiness |
| Building | 50-69 | In development | Moderate gaps; 3-6 months of focused work needed |
| Early Stage | 25-49 | Planning phase | Significant gaps; 6-12 months of work needed |
| Pre-Pledge | 0-24 | Not yet started | Major foundational work required before pledge |

**Key Models:**
- `ReadinessInput` - All engine results aggregated
- `ReadinessResult` - Composite score, dimension scores, RAG status, readiness level, priority actions, timeline estimate
- `DimensionScore` - Single dimension with score, weight, gap-weighted priority, improvement actions
- `ImprovementPriority` - Ranked improvement action with impact, effort, timeline

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

### 5.3 Application Dependencies

- GL-GHG-APP: GHG inventory management, base year, aggregation
- GL-SBTi-APP: Science-based target setting, pathway calculation, temperature scoring
- GL-CDP-APP: CDP questionnaire alignment (corporate reporting channel)
- GL-TCFD-APP: TCFD metrics and targets alignment
- GL-Taxonomy-APP: EU Taxonomy climate CapEx alignment (for transition plan credibility)

### 5.4 Data Agents (20)

All 20 AGENT-DATA agents via `data_bridge.py` for data intake and quality.

### 5.5 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema, units, audit, etc.

### 5.6 Optional Pack Dependencies

- PACK-021 Net Zero Starter Pack: Baseline, gap analysis, reduction pathway (via integration if present)
- PACK-022 Net Zero Acceleration Pack: Scenarios, SDA pathways, supplier engagement (via integration if present)
- PACK-023 SBTi Alignment Pack: SBTi lifecycle management for corporate participants (via integration if present)
- PACK-024 Carbon Neutral Pack: Carbon neutrality claim complementary to Race to Zero pledge (via integration if present)

---

## 6. Performance Targets

| Metric | Target |
|--------|--------|
| Pledge eligibility assessment | <5 minutes |
| Starting Line assessment (20 sub-criteria) | <15 minutes |
| Interim target validation (1.5C alignment) | <5 minutes |
| Action plan generation (full 10-section plan) | <30 minutes |
| Annual progress calculation and reporting | <20 minutes |
| Sector pathway mapping (single sector) | <3 minutes |
| Sector pathway mapping (multi-sector entity) | <10 minutes |
| Partnership scoring (5 partners) | <5 minutes |
| HLEG credibility assessment (45+ sub-criteria) | <15 minutes |
| Race readiness scoring (full composite) | <5 minutes |
| Full Race to Zero lifecycle workflow | <3 hours |
| Campaign report generation (all partner formats) | <20 minutes |
| Memory ceiling | 4096 MB |
| Cache hit target | 75% |
| Max facilities | 1,000 |
| Max sectors mapped simultaneously | 25 |
| Max partner initiatives tracked | 40 |

---

## 7. Security Requirements

- JWT RS256 authentication
- RBAC with 8 roles: `race_to_zero_manager`, `sustainability_analyst`, `climate_officer`, `city_climate_lead`, `fi_transition_lead`, `partnership_coordinator`, `external_auditor`, `admin`
- AES-256-GCM encryption at rest for all emission and pledge data
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs
- Full audit trail per SEC-005
- Partner initiative API credentials encrypted via Vault (SEC-006)
- Campaign submission data access controls (read/write/submit separation)

---

## 8. Database Migrations

Inherits platform migrations V001-V128. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V083-PACK025-001 | `r2z_pledges` | Race to Zero pledge commitment records with eligibility and quality assessment |
| V083-PACK025-002 | `r2z_starting_line` | Starting Line Criteria assessment records with 20 sub-criterion results |
| V083-PACK025-003 | `r2z_interim_targets` | Interim 2030 target records with 1.5C pathway alignment validation |
| V083-PACK025-004 | `r2z_action_plans` | Climate action plan records with section-level completeness scoring |
| V083-PACK025-005 | `r2z_progress` | Annual progress tracking records with trajectory and variance analysis |
| V083-PACK025-006 | `r2z_sector_pathways` | Sector pathway alignment records with gap-to-benchmark analysis |
| V083-PACK025-007 | `r2z_partnerships` | Partnership engagement records with synergy and coverage scoring |
| V083-PACK025-008 | `r2z_campaign_reports` | Campaign report records with partner-specific formatted outputs |
| V083-PACK025-009 | `r2z_credibility` | HLEG credibility assessment records with 10-recommendation scoring |
| V083-PACK025-010 | `r2z_readiness` | Race readiness composite score records with 8-dimension breakdown |

---

## 9. File Structure

```
packs/net-zero/PACK-025-race-to-zero/
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
      corporate_commitment.yaml
      financial_institution.yaml
      city_municipality.yaml
      region_state.yaml
      sme_business.yaml
      high_emitter.yaml
      service_sector.yaml
      manufacturing_sector.yaml
  engines/
    __init__.py
    pledge_commitment_engine.py
    starting_line_engine.py
    interim_target_engine.py
    action_plan_engine.py
    progress_tracking_engine.py
    sector_pathway_engine.py
    partnership_scoring_engine.py
    campaign_reporting_engine.py
    credibility_assessment_engine.py
    race_readiness_engine.py
  workflows/
    __init__.py
    pledge_onboarding_workflow.py
    starting_line_assessment_workflow.py
    action_planning_workflow.py
    annual_reporting_workflow.py
    sector_pathway_workflow.py
    partnership_engagement_workflow.py
    credibility_review_workflow.py
    full_race_to_zero_workflow.py
  templates/
    __init__.py
    pledge_commitment_letter.py
    starting_line_checklist.py
    action_plan_document.py
    annual_progress_report.py
    sector_pathway_roadmap.py
    partnership_framework.py
    credibility_assessment.py
    campaign_submission.py
    disclosure_dashboard.py
    race_to_zero_certificate.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_bridge.py
    ghg_app_bridge.py
    sbti_app_bridge.py
    decarb_bridge.py
    taxonomy_bridge.py
    data_bridge.py
    unfccc_bridge.py
    cdp_bridge.py
    gfanz_bridge.py
    setup_wizard.py
    health_check.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_pledge_engine.py
    test_starting_line_engine.py
    test_interim_target_engine.py
    test_action_plan_engine.py
    test_progress_engine.py
    test_sector_pathway_engine.py
    test_partnership_engine.py
    test_campaign_reporting_engine.py
    test_credibility_engine.py
    test_race_readiness_engine.py
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
| Template Tests | 100% | All 10 templates in 3+ formats (MD, HTML, JSON, PDF where applicable) |
| Integration Tests | >80% | All 12 integrations with mock agents and partner APIs |
| E2E Tests | Core happy path | Full pipeline from pledge to race readiness certificate |
| Starting Line Tests | 100% | All 20 sub-criteria with edge cases (pass/fail/partial/NA) |
| HLEG Credibility Tests | 100% | All 10 recommendations with 45+ sub-criteria |
| Sector Pathway Tests | >90% | All 25+ sector pathways with benchmark validation |
| Partnership Tests | >85% | All 40+ partner initiative mappings |
| Preset Tests | 100% | All 8 actor-type presets with representative scenarios |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 600+ tests (50-60 per engine, 30-40 integration, 15-20 E2E)

---

## 11. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-18 |
| Phase 2 | Engine implementation (10 engines) | 2026-03-19 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-19 |
| Phase 4 | Template implementation (10 templates) | 2026-03-19 |
| Phase 5 | Integration implementation (12 integrations) | 2026-03-19 |
| Phase 6 | Test suite (600+ tests) | 2026-03-20 |
| Phase 7 | Documentation & Release | 2026-03-20 |

---

## 12. Appendix: Race to Zero Key Dates & Milestones

| Date | Event | Relevance |
|------|-------|-----------|
| June 2020 | Race to Zero campaign launch | Campaign inception by UNFCCC Climate Champions |
| November 2021 | COP26 Glasgow | Race to Zero mobilization peak; strengthened criteria announced |
| June 2022 | Interpretation Guide update | Tightened Starting Line criteria; added HLEG alignment |
| November 2022 | HLEG "Integrity Matters" report | 10 recommendations incorporated into Race to Zero minimum criteria |
| November 2023 | COP28 Dubai | Global Stocktake; renewed Race to Zero ambition |
| November 2024 | COP29 Baku | Enhanced transparency framework; strengthened annual reporting |
| 2025 | NDC update cycle | National targets inform entity-level "fair share" calculations |
| November 2025 | COP30 Belem | Expected further criteria tightening |
| 2030 | Interim target year | 50% absolute emission reduction milestone |
| 2050 | Net-zero target year | Full campaign objective achievement |

---

## 13. Appendix: HLEG Recommendation Detail

### Recommendation 1: Announce Net-Zero Pledge
- Specific, public, time-bound commitment to net zero by 2050 at latest
- Covers all GHG emissions (Scope 1, 2, and material Scope 3)
- Endorsed by highest governance body (board, city council, etc.)
- Registered with a recognized Race to Zero partner initiative
- Published on entity's website and through partner channels

### Recommendation 2: Set Interim Targets
- 5-year interim targets starting with 2025 and then 2030
- 2030 target aligned with halving emissions (~50% reduction from recent baseline)
- Science-based methodology (SBTi, IEA NZE, IPCC pathways)
- All scopes covered; no cherry-picking of easy categories
- Annual milestones defined for tracking

### Recommendation 3: Implement Transition Plan
- Detailed, quantified transition plan with specific actions
- Resource allocation (financial, human, technical)
- Technology deployment pathway with TRL assessment
- Sector-aligned decarbonization strategy
- Regular plan review and update cycle

### Recommendation 4: Phase Out Fossil Fuels
- No new fossil fuel exploration, development, or capacity expansion
- Phase-out timeline for existing fossil fuel assets (aligned with IEA NZE)
- Divestment policy for fossil fuel investments (for FIs and investors)
- Just transition planning for affected workers and communities

### Recommendation 5: Use Voluntary Credits Responsibly
- Credits must complement (not substitute for) deep emission reductions
- Prioritize reduction, then compensation, then neutralization
- Apply ICVCM Core Carbon Principles for credit quality
- Full transparency on credit usage, types, and quantities
- Credits do not count toward interim reduction targets

### Recommendation 6: Align Lobbying with Climate Goals
- Audit trade association memberships for climate policy alignment
- Disclose lobbying activities and expenditures related to climate policy
- No direct or indirect lobbying against climate legislation
- Publicly commit to policy advocacy consistent with 1.5C pathway
- Report on actions taken to address misaligned trade associations

### Recommendation 7: Plan for a Just Transition
- Workforce transition planning (reskilling, redeployment)
- Community impact assessment and mitigation
- Stakeholder consultation (workers, communities, Indigenous peoples)
- Distributional impact analysis of climate actions
- Human rights due diligence integrated into transition planning

### Recommendation 8: Increase Transparency
- Annual public reporting on emissions, targets, and progress
- Methodology and assumption disclosure
- Third-party verification or assurance (where feasible)
- Data accessible in machine-readable formats
- Clear communication of limitations and uncertainties

### Recommendation 9: Invest in Systemic Change
- Climate finance contributions beyond own value chain
- R&D investment in decarbonization technologies and solutions
- Supply chain capacity building for emission reductions
- Collaboration with peers on pre-competitive climate solutions
- Contribution to public goods (open-source tools, shared data)

### Recommendation 10: Ensure Governance and Accountability
- Board-level climate oversight and competency
- Executive remuneration linked to climate targets
- Climate risk integration into enterprise risk management
- External accountability mechanisms (advisory boards, audits)
- Clear escalation procedures for off-track performance

---

## 14. Appendix: Partner Initiative Crosswalk

| Race to Zero Criterion | SBTi | CDP | C40 | GFANZ | ICLEI | WMB |
|------------------------|------|-----|-----|-------|-------|-----|
| Net-zero pledge by 2050 | NZ-C1 | C4.1 | Deadline 2020 | Commitment | GreenClimate | Commit |
| Interim 2030 target | C9-C12 | C4.1b | CAP targets | Sector pathway | SECAP | Target |
| Action plan published | Not required | C3.3 | CAP required | Transition plan | SECAP | Plan |
| Annual reporting | Annual progress | Annual response | Annual report | Annual disclosure | Annual report | Annual |
| Scope 1+2 coverage | C1-C4 | C6 | Community GHG | Portfolio Scope 1+2 | Community GHG | C6 |
| Scope 3 coverage | C17-C20 | C6.5 | Community-wide | Financed emissions | Community-wide | C6.5 |
| Science-based methodology | Core requirement | C4.2 | C40 compatible | GFANZ compatible | IPCC compatible | SBTi |

---

## 15. Future Roadmap

- **PACK-026: Race to Zero Cities Pack** -- City-specific Race to Zero implementation with GPC community inventory, C40-aligned CAP, transport/buildings/waste sector pathways, citizen engagement, and city-level reporting
- **PACK-027: Race to Zero Financial Institutions Pack** -- FI-specific Race to Zero with GFANZ transition plans, PCAF financed emissions, portfolio alignment, stewardship engagement, and financial sector reporting
- **PACK-028: Race to Zero Progress Analytics Pack** -- Advanced analytics for ongoing participants: multi-year trend analysis, peer benchmarking, cohort analysis, campaign-wide performance insights, and predictive trajectory modeling

---

*Document Version: 1.0.0 | Last Updated: 2026-03-18 | Status: Draft*
