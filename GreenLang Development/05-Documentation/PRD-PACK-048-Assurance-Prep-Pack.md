# PRD-PACK-048: GHG Assurance Prep Pack

**Pack ID:** PACK-048-assurance-prep
**Category:** GHG Accounting Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-26
**Prerequisite:** PACK-041 (Scope 1-2 Complete) required; enhanced with PACK-042/043 (Scope 3), PACK-044 (Inventory Management), PACK-045 (Base Year Management), PACK-046 (Intensity Metrics), PACK-047 (Benchmark)

---

## 1. Executive Summary

### 1.1 Problem Statement

Third-party assurance of GHG emissions inventories is rapidly transitioning from voluntary best practice to regulatory mandate. The EU Corporate Sustainability Reporting Directive (CSRD) requires limited assurance from 2024 (reporting year 2025) and reasonable assurance from 2028. The US SEC Climate Disclosure Rules require attestation for large accelerated filers. California SB 253 mandates third-party verification for Scope 1, 2, and 3 emissions. The ISSB IFRS S2 expects assurance alignment. Without rigorous preparation, organisations face verification failures, qualified opinions, material misstatements, and regulatory non-compliance.

Current approaches to assurance preparation suffer from critical structural challenges:

1. **Scope-fragmented evidence**: Organisations maintain separate evidence trails for Scope 1 (combustion records, refrigerant logs), Scope 2 (utility bills, RECs, contractual instruments), and Scope 3 (supplier data, spend records, emission factor selections). Verifiers need a unified evidence package that traces the complete Scope 1+2+3 inventory with consistent documentation quality across all scopes. Without cross-scope consolidation, organisations submit piecemeal documentation that extends verification timelines by 40-60% and triggers avoidable queries.

2. **Unknown readiness gaps**: Most organisations learn about documentation gaps during verification -- when it's too late to remediate without delaying the engagement. A pre-verification readiness assessment with standard-specific checklists (ISAE 3410, ISO 14064-3, AA1000AS v3) would identify gaps 8-12 weeks before the verifier arrives. Current practice relies on ad-hoc self-assessment or expensive pre-assurance advisory services.

3. **Missing calculation provenance**: Verifiers under ISAE 3410 must understand the "subject matter information" -- how each emission figure was derived. This requires step-by-step calculation provenance showing: source data -> emission factor selection rationale -> calculation formula -> intermediate results -> final tCO2e value, all linked by SHA-256 hash chains. Without automated provenance generation, internal teams spend 100-200 hours manually reconstructing calculation trails.

4. **Inconsistent methodology documentation**: Every GHG inventory involves methodology decisions: which consolidation approach (equity share vs. operational control), which GWP values (AR5 vs. AR6), which Scope 2 method (location vs. market-based), which Scope 3 calculation tier, which emission factors and why. These decisions are often undocumented or scattered across emails and meeting notes. Verifiers expect a consolidated methodology decision register with rationale and alternative analysis.

5. **No control testing framework**: Reasonable assurance (ISAE 3410) requires the verifier to test internal controls over GHG data. Organisations that have no documented control framework, no control testing history, and no evidence of control effectiveness face significant additional verification procedures, higher fees, and potential opinion qualification. A pre-verification control self-assessment would demonstrate control maturity.

6. **Verifier collaboration friction**: During verification, auditors issue Information Requests (IRs), queries, and finding notifications. These are typically managed via email and spreadsheets, leading to lost queries, duplicate responses, missed deadlines, and version control issues. A structured query management system with evidence linking, threaded responses, and resolution tracking would reduce verification duration by 30-50%.

7. **Materiality assessment gaps**: Assurance standards require materiality determination -- what threshold of misstatement would influence stakeholder decisions. Most organisations lack a documented materiality assessment linking quantitative thresholds (e.g., 5% of total Scope 1+2 emissions) to qualitative factors. Without this, verifiers must perform their own materiality assessment, potentially using more conservative thresholds.

8. **Missing sampling methodology**: For large inventories (100+ facilities, 1000+ data points), verifiers use statistical sampling. Organisations that pre-identify the population, stratify by materiality, and propose sampling plans demonstrate maturity and can influence the verification approach. Without pre-sampling, verifiers default to conservative sample sizes that increase cost.

9. **Regulatory requirement blindspots**: Different jurisdictions impose different assurance requirements: CSRD (limited->reasonable), SEC (attestation for LAF), SB 253 (third-party verification), UK Companies Act (energy and carbon reporting), Singapore mandatory reporting. Organisations operating across jurisdictions may not know which assurance standard applies where, or whether limited or reasonable assurance is required.

10. **Cost and timeline uncertainty**: Assurance engagements typically cost EUR 15,000-150,000+ depending on scope, complexity, and assurance level. Without scoping tools, organisations cannot budget accurately, plan internal resource requirements, or negotiate effectively with assurance providers. Engagements frequently overrun both time and budget.

### 1.2 Solution Overview

PACK-048 is the **GHG Assurance Prep Pack** -- the eighth pack in the "GHG Accounting Packs" category, dedicated to preparing organisations for successful third-party GHG assurance engagements.

While PACK-043 (Scope 3 Complete) includes a Scope 3-only AssuranceEngine (Engine 10, 1,309 lines) and PACK-044 (Inventory Management) provides review/approval and documentation engines, PACK-048 provides the **unified cross-scope assurance preparation lifecycle**:

- **Cross-scope evidence consolidation** unifying Scope 1, 2, and 3 evidence packages with consistent documentation standards and provenance chains
- **Readiness assessment** with standard-specific checklists for ISAE 3410, ISO 14064-3, and AA1000AS v3 with weighted scoring and gap identification
- **Calculation provenance engine** generating step-by-step audit trails with SHA-256 hash chains across all scopes from source data through final tCO2e
- **Methodology decision register** consolidating all methodology choices with rationale, alternatives considered, and impact analysis
- **Control testing framework** for pre-verification control self-assessment aligned with COSO/COBIT frameworks
- **Verifier collaboration engine** managing Information Requests, queries, findings, and evidence sharing with threaded responses
- **Materiality assessment** with quantitative threshold calculation and qualitative factor documentation
- **Sampling plan engine** with population identification, stratification, and statistical sample size determination
- **Regulatory requirement mapper** identifying applicable assurance requirements by jurisdiction, company size, and listing status
- **Assurance reporting** generating engagement-ready evidence bundles, executive dashboards, and multi-year trend analysis

The pack includes **10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets** covering the complete assurance preparation lifecycle from readiness assessment through verifier collaboration.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-048 Assurance Prep Pack |
|-----------|------------------------------|-------------------------------|
| Evidence consolidation | Manual file gathering (2-4 weeks) | Automated cross-scope package in <1 hour |
| Readiness assessment | Ad-hoc self-check or expensive advisory | Systematic standard-specific scoring with gap analysis |
| Calculation provenance | Manual reconstruction (100-200 hours) | Automated SHA-256 hash-chained provenance in minutes |
| Methodology documentation | Scattered emails and notes | Consolidated decision register with alternatives |
| Control testing | No framework or informal checks | COSO/COBIT-aligned self-assessment with evidence |
| Verifier collaboration | Email and spreadsheets | Structured query management with threading and evidence linking |
| Materiality assessment | Undocumented or absent | Quantitative + qualitative documented methodology |
| Sampling plan | Verifier-determined (conservative) | Pre-prepared stratified sampling with population stats |
| Regulatory mapping | Manual legal research | Automated requirement mapping by jurisdiction |
| Cost estimation | Guesswork | Scoping model based on complexity and assurance level |
| Audit trail | Fragmented | SHA-256 provenance on every result, full lineage |

### 1.4 Distinction from Related Packs

| Pack | Focus | Relationship to PACK-048 |
|------|-------|--------------------------|
| PACK-043 Engine 10 | Scope 3 assurance evidence only | PACK-048 extends to all scopes, adds readiness assessment, control testing, verifier collaboration |
| PACK-044 Review/Approval | Internal review and sign-off workflow | PACK-048 adds external verifier workflow, assurance-specific checklists |
| PACK-044 Documentation | General inventory documentation | PACK-048 adds assurance-specific evidence categorisation and standard mapping |
| PACK-029/030 Assurance Evidence | Evidence storage for Net Zero packs | PACK-048 provides full lifecycle management, not just storage |
| SEC-005 Audit Logging | Platform-level audit trail infrastructure | PACK-048 leverages for GHG-specific assurance events |
| SEC-009 SOC 2 | IT controls and security assurance | PACK-048 applies similar frameworks to GHG data controls |

### 1.5 Assurance Standards Supported

| Standard | Full Name | Scope | Assurance Level |
|----------|-----------|-------|-----------------|
| ISAE 3410 | Assurance Engagements on GHG Statements | GHG inventories | Limited / Reasonable |
| ISO 14064-3 | GHG -- Verification and Validation | GHG assertions | Reasonable / Limited |
| AA1000AS v3 | AccountAbility Assurance Standard | Sustainability reports | High / Moderate |
| ISAE 3000 (Revised) | Assurance on Non-Financial Information | Broader ESG | Limited / Reasonable |
| SSAE 18 / AT-C 105 | US Attestation Standards (PCAOB/AICPA) | SEC filings | Examination / Review |

### 1.6 Target Users

**Primary:**
- Sustainability managers preparing for first-time or repeat GHG assurance engagements
- Internal audit teams conducting pre-verification readiness assessments
- CFOs and controllers responsible for assurance engagement budgeting and scoping

**Secondary:**
- External verifiers using the system to manage queries and findings
- ESG consultants helping clients achieve assurance readiness
- Board audit committees monitoring assurance maturity and readiness trends

### 1.7 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Evidence package generation | <60 minutes for full Scope 1+2+3 | Benchmark suite timer |
| Readiness score accuracy | ±5% vs. actual verification outcome | Post-verification comparison |
| Verifier query reduction | 40% fewer queries vs. unstructured approach | Engagement tracking |
| Verification timeline reduction | 30% shorter engagement duration | Engagement comparison |
| Provenance completeness | 100% of calculations with hash chain | Automated verification |
| Regulatory requirement coverage | 8+ jurisdictions mapped | Feature audit |
| Cost estimation accuracy | ±15% vs. actual engagement cost | Post-engagement comparison |

---

## 2. Technical Architecture

### 2.1 Engine Specifications (10 Engines)

#### Engine 1: Evidence Consolidation Engine (~1,200 lines)
- **Purpose**: Generate unified cross-scope evidence packages for verifier review
- **Key Features**:
  - Scope 1 evidence collection: stationary combustion records, mobile fleet logs, process emissions calculations, fugitive emissions monitoring, refrigerant tracking
  - Scope 2 evidence collection: utility bills, contractual instruments (RECs, PPAs, GoOs), residual mix factors, grid emission factors with source
  - Scope 3 evidence collection: supplier-specific data, spend-based calculations, activity-based calculations, emission factor selections with rationale
  - Cross-scope consolidation with consistent documentation quality grading
  - Evidence categorisation per ISAE 3410 requirements: source data, emission factors, calculations, assumptions, methodology decisions
  - Evidence completeness scoring by scope, category, and facility
  - Digital evidence index with SHA-256 file hashes
  - Evidence package versioning (draft, review, final)
- **Integration**: Leverages PACK-043 AssuranceEngine patterns, extends to Scope 1+2

#### Engine 2: Readiness Assessment Engine (~1,100 lines)
- **Purpose**: Assess pre-verification readiness against assurance standard requirements
- **Key Features**:
  - ISAE 3410 readiness checklist (80+ items across 10 categories)
  - ISO 14064-3 readiness checklist (60+ items across 8 categories)
  - AA1000AS v3 readiness checklist (50+ items across 6 categories)
  - Weighted scoring: each item scored 0 (absent) to 4 (exceeds requirements)
  - Category weights configurable (default: data quality 20%, methodology 15%, documentation 15%, controls 15%, completeness 10%, provenance 10%, governance 10%, boundary 5%)
  - Gap identification with prioritised remediation recommendations
  - Readiness thresholds: Ready (>=90%), Mostly Ready (>=70%), Partially Ready (>=40%), Not Ready (<40%)
  - Standard-specific pass/fail gates (e.g., no calculation without provenance can pass ISAE 3410)
  - Time-to-ready estimation based on gap severity and resource assumptions
- **Formulas**:
  - Readiness score: `R = Σ(w_cat * Σ(score_item / max_score) / n_items) * 100`
  - Time-to-ready: `T = Σ(gap_severity_i * remediation_effort_i)` in person-days

#### Engine 3: Calculation Provenance Engine (~1,100 lines)
- **Purpose**: Generate step-by-step calculation audit trails with cryptographic provenance
- **Key Features**:
  - Source data capture: raw data value, source document, extraction method, data quality grade
  - Emission factor chain: factor value, source (DEFRA, EPA, ecoinvent), version, applicability justification
  - Calculation formula documentation: formula used, GHG Protocol reference, tier level
  - Intermediate results with hash chain: each step hashed, linked to parent
  - Final tCO2e with complete lineage from source through factor through formula to result
  - Cross-scope provenance: S1+S2+S3 each traced independently then consolidated
  - Provenance completeness scoring: % of final tCO2e with full chain
  - Provenance gap detection: identify calculations missing source data, factor justification, or formula reference
  - YoY provenance comparison: detect methodology changes between periods
- **Formulas**:
  - Hash chain: `H_n = SHA256(H_{n-1} || step_data_n || timestamp_n)`
  - Provenance completeness: `PC = count(full_chain_calcs) / count(all_calcs) * 100`

#### Engine 4: Control Testing Engine (~1,000 lines)
- **Purpose**: Self-assess internal controls over GHG data aligned with COSO/COBIT
- **Key Features**:
  - Control identification for GHG data processes (25+ standard controls)
  - Control categories: data collection, data entry, calculation, review, approval, reporting, IT general
  - Control type classification: preventive, detective, corrective
  - Design effectiveness assessment: is the control properly designed?
  - Operating effectiveness assessment: is the control operating as designed?
  - Sample testing methodology: select transactions, test control application, document results
  - Control deficiency classification: deficiency, significant deficiency, material weakness
  - Remediation planning: action, owner, deadline, status
  - Control maturity model: Level 1 (ad hoc) through Level 5 (optimised)
  - Year-over-year control evolution tracking
- **Control Register** (25 standard controls):
  - DC-01 to DC-05: Data collection controls (meter calibration, data capture, supplier data validation, activity data reconciliation, completeness check)
  - CA-01 to CA-05: Calculation controls (emission factor selection, formula validation, unit conversion, aggregation review, system access)
  - RV-01 to RV-05: Review controls (peer review, management review, cross-scope reconciliation, variance analysis, sign-off)
  - RE-01 to RE-05: Reporting controls (data extraction, template accuracy, disclosure review, submission approval, archive)
  - IT-01 to IT-05: IT general controls (access management, change management, backup, audit trail, data integrity)

#### Engine 5: Verifier Collaboration Engine (~1,000 lines)
- **Purpose**: Manage verifier information requests, queries, findings, and evidence sharing
- **Key Features**:
  - Information Request (IR) management: IR creation, assignment, response, evidence linking
  - Query management: verifier questions with category, priority, deadline
  - Finding management: non-conformity, observation, opportunity, recommendation
  - Finding severity: critical (must remediate before opinion), major, minor, observation
  - Threaded response tracking: response chain with evidence attachments
  - Resolution workflow: open -> in_progress -> responded -> follow_up -> resolved -> closed
  - Evidence request fulfilment: link specific evidence items to queries/findings
  - Escalation management: auto-escalate overdue items
  - Engagement timeline tracking: planned vs. actual milestones
  - Verifier access control: read-only access to evidence, write access to queries/findings
- **SLA Tracking**:
  - Query response SLA: 5 business days (configurable)
  - Critical finding remediation SLA: 10 business days
  - Evidence request SLA: 3 business days

#### Engine 6: Materiality Assessment Engine (~900 lines)
- **Purpose**: Determine materiality thresholds for GHG assurance engagement
- **Key Features**:
  - Quantitative materiality: percentage of total emissions (configurable, default 5%)
  - Scope-specific materiality: different thresholds per scope
  - Qualitative factors: regulatory sensitivity, stakeholder visibility, reputational risk
  - Performance materiality: typically 50-75% of overall materiality for testing
  - Clearly trivial threshold: typically 5-10% of materiality for aggregation
  - Specific materiality: for individual line items (e.g., Scope 1 fugitive emissions)
  - Materiality revision: update during engagement if new information emerges
  - Documentation: full methodology, rationale for each threshold level
- **Formulas**:
  - Overall materiality: `M = total_emissions * materiality_pct`
  - Performance materiality: `PM = M * performance_pct` (default 0.65)
  - Clearly trivial: `CT = M * trivial_pct` (default 0.05)
  - Scope-specific: `M_scope = scope_emissions * scope_materiality_pct`

#### Engine 7: Sampling Plan Engine (~900 lines)
- **Purpose**: Design statistical sampling plans for verification testing
- **Key Features**:
  - Population identification: all data points, facilities, sources that comprise the inventory
  - Stratification by: scope, category, facility, materiality, risk level
  - Sample size calculation using statistical formulas (confidence level, expected error rate, tolerable misstatement)
  - Stratified sampling: allocate samples proportionally or based on risk
  - High-value items: 100% testing for items above individual materiality
  - Key items: judgmental selection of high-risk items
  - Remaining population: statistical sampling with defined confidence
  - Sample selection methods: monetary unit sampling (MUS), random, systematic
  - Projected misstatement calculation from sample results
  - Documentation: sampling methodology, population description, selection criteria
- **Formulas**:
  - MUS sample size: `n = (reliability_factor * population_value) / tolerable_misstatement`
  - Confidence level: default 95% for reasonable, 80% for limited assurance
  - Projected misstatement: `PM = (sample_errors / sample_size) * population_size`

#### Engine 8: Regulatory Requirement Engine (~950 lines)
- **Purpose**: Map applicable assurance requirements by jurisdiction and company characteristics
- **Key Features**:
  - Jurisdiction mapping: EU/EEA (CSRD), US (SEC), California (SB 253), UK (SECR/Streamlined), Singapore, Japan, Australia, South Korea
  - Company size thresholds: large (CSRD), LAF/AF/NAF (SEC), revenue thresholds (SB 253)
  - Assurance level requirements: limited vs. reasonable by jurisdiction and timeline
  - Assurance standard applicability: ISAE 3410, ISAE 3000, SSAE 18, ISO 14064-3
  - Scope requirements: which scopes require assurance per jurisdiction
  - Timeline mapping: when each requirement takes effect (2024-2028 phase-in)
  - Multi-jurisdiction consolidation: identify all applicable requirements for a global company
  - Requirement gap analysis: what the organisation has vs. what's required
  - Regulatory alert: upcoming requirements in next 12-24 months
- **Jurisdictions (12)**:
  - EU/EEA (CSRD): Limited 2025, Reasonable 2028, all scopes
  - US SEC: LAF attestation 2026, AF 2027, all filers eventually
  - California SB 253: Third-party verification 2026, S1+S2+S3
  - UK: SECR limited verification (voluntary, becoming mandatory)
  - Singapore: SGX mandatory reporting, assurance emerging
  - Japan: SSBJ alignment with ISSB, assurance expected 2027+
  - Australia: ASRS climate standards, assurance phased 2025-2030
  - South Korea: KSQF ESG disclosure, assurance phased 2025-2028
  - Hong Kong: HKEX ESG reporting, assurance voluntary->mandatory
  - Brazil: CVM sustainability reporting, assurance emerging
  - India: BRSR Core verification, mandatory for top 1000
  - Canada: CSSB alignment with ISSB, assurance expected 2027+

#### Engine 9: Cost and Timeline Engine (~850 lines)
- **Purpose**: Estimate assurance engagement cost, timeline, and internal resource requirements
- **Key Features**:
  - Engagement scoping: scope coverage, facility count, data point count, complexity factors
  - Cost estimation by assurance level: limited (base) vs. reasonable (2-3x base)
  - Cost factors: number of scopes, number of facilities, Scope 3 categories included, multi-jurisdiction, first-time vs. repeat
  - Timeline estimation: planning (2-4 weeks), fieldwork (2-6 weeks), reporting (1-2 weeks)
  - Internal resource estimation: FTE hours by role (sustainability team, finance, operations, IT)
  - Multi-year planning: engagement cost trajectory as scope expands
  - Vendor comparison: standardised RFP template for verifier selection
  - Fee benchmarking: typical ranges by company size, industry, and scope
- **Cost Model**:
  - Base cost: `C_base = f(facility_count, scope_coverage, complexity)` lookup table
  - Reasonable multiplier: `C_reasonable = C_base * 2.5` (typical)
  - Multi-jurisdiction uplift: `C_multi = C_base * (1 + 0.15 * additional_jurisdictions)`
  - First-time premium: `C_first = C_base * 1.3`
  - Scope 3 complexity: `C_s3 = C_base * (1 + 0.1 * s3_categories_count)`

#### Engine 10: Assurance Reporting Engine (~950 lines)
- **Purpose**: Generate comprehensive assurance preparation reports and dashboards
- **Key Features**:
  - Readiness dashboard: overall score, category breakdown, traffic lights
  - Evidence package index: complete list of evidence items with status
  - Control self-assessment report: control register, testing results, deficiency log
  - Verifier query register: all IRs, queries, findings with status
  - Materiality assessment report: methodology, thresholds, rationale
  - Sampling plan report: population, stratification, sample selection
  - Regulatory requirement report: applicable requirements by jurisdiction
  - Cost and timeline report: engagement scope, estimated cost, timeline, resource plan
  - Multi-year trend: readiness evolution, finding recurrence, control maturity
  - Export: Markdown, HTML, PDF, JSON, CSV, XBRL
  - Provenance: SHA-256 hash on every report

### 2.2 Workflow Specifications (8 Workflows)

1. **Readiness Assessment Workflow** (~950 lines) - StandardSelect -> ChecklistGenerate -> EvidenceCheck -> ScoreCalculation -> GapReport
2. **Evidence Collection Workflow** (~950 lines) - ScopeInventory -> SourceIdentify -> DocumentCollect -> QualityGrade -> PackageBuild
3. **Control Testing Workflow** (~850 lines) - ControlIdentify -> DesignAssess -> SampleSelect -> TestExecute -> DeficiencyReport
4. **Verifier Engagement Workflow** (~850 lines) - EngagementScope -> VerifierOnboard -> QueryManage -> FindingTrack -> CloseOut
5. **Materiality and Sampling Workflow** (~850 lines) - MaterialityCalc -> PopulationIdentify -> Stratify -> SampleSize -> SelectionPlan
6. **Regulatory Mapping Workflow** (~800 lines) - JurisdictionIdentify -> RequirementMap -> GapAnalysis -> CompliancePlan
7. **Cost and Timeline Workflow** (~800 lines) - EngagementScope -> CostEstimate -> TimelinePlan -> ResourceAlloc -> BudgetApproval
8. **Full Assurance Prep Pipeline Workflow** (~1,200 lines) - 8-phase end-to-end orchestration

### 2.3 Template Specifications (10 Templates)

1. **Assurance Readiness Dashboard** (~750 lines) - Overall score, category breakdown, traffic lights, trend
2. **Evidence Package Index** (~780 lines) - Complete evidence inventory with status, quality, source
3. **Control Self-Assessment Report** (~700 lines) - 25 controls, design+operating effectiveness, deficiencies
4. **Verifier Query Register** (~680 lines) - IR/query/finding log with SLA tracking, resolution status
5. **Materiality Assessment Report** (~650 lines) - Quantitative+qualitative materiality, performance materiality
6. **Sampling Plan Report** (~680 lines) - Population, stratification, sample size, selection criteria
7. **Regulatory Requirement Report** (~700 lines) - Jurisdiction-by-jurisdiction requirements, timeline, compliance status
8. **Cost and Timeline Report** (~650 lines) - Engagement scoping, cost estimate, timeline, resource plan
9. **ISAE 3410 Evidence Bundle** (~720 lines) - ISAE 3410-specific evidence package with section mapping, XBRL
10. **Multi-Year Assurance Trend** (~650 lines) - Readiness evolution, finding recurrence, control maturity

### 2.4 Integration Specifications (12 Integrations)

1. **Pack Orchestrator** (~820 lines) - 10-phase DAG pipeline coordinator
2. **MRV Bridge** (~400 lines) - 30 AGENT-MRV agents for calculation provenance extraction
3. **Data Bridge** (~430 lines) - AGENT-DATA agents for source data evidence
4. **PACK-041 Bridge** (~370 lines) - Scope 1+2 emissions data and evidence
5. **PACK-042/043 Bridge** (~420 lines) - Scope 3 emissions; leverages PACK-043 AssuranceEngine
6. **PACK-044 Bridge** (~380 lines) - Inventory review/approval records, documentation
7. **PACK-045 Bridge** (~350 lines) - Base year data and recalculation documentation
8. **PACK-046/047 Bridge** (~400 lines) - Intensity metrics and benchmark context for materiality
9. **Foundation Bridge** (~450 lines) - FOUND-004 assumptions, FOUND-005 citations, FOUND-008 reproducibility
10. **Health Check** (~420 lines) - 20-category system verification
11. **Setup Wizard** (~520 lines) - 8-step guided configuration
12. **Alert Bridge** (~580 lines) - Deadline/gap/query/finding/readiness alerts

### 2.5 Configuration & Presets

**pack_config.py** (~1,800 lines):
- 18 enums: AssuranceStandard, AssuranceLevel, EvidenceCategory, EvidenceQuality, ControlCategory, ControlType, ControlEffectiveness, ControlMaturity, FindingSeverity, FindingType, QueryPriority, QueryStatus, MaterialityType, SamplingMethod, Jurisdiction, CompanySize, EngagementPhase, ReportFormat
- Reference data: 25 standard controls, 12 jurisdiction requirements, ISAE 3410 checklist structure, cost model parameters
- 15+ sub-config Pydantic models with validators
- PackConfig wrapper with from_preset(), from_yaml(), merge(), validate()

**8 Presets:**
1. `corporate_general.yaml` - General corporate assurance preparation
2. `csrd_limited.yaml` - EU CSRD limited assurance (2025-2027)
3. `csrd_reasonable.yaml` - EU CSRD reasonable assurance (2028+)
4. `sec_attestation.yaml` - US SEC climate disclosure attestation
5. `california_sb253.yaml` - California SB 253 verification
6. `multi_jurisdiction.yaml` - Global companies with multiple assurance requirements
7. `financial_services.yaml` - Banks, insurers, asset managers (financed emissions)
8. `first_time_assurance.yaml` - Organisations undertaking assurance for the first time

---

## 3. Database Schema

### 3.1 Migrations (V396-V405)

All tables in `ghg_assurance` schema with `gl_ap_` prefix. UUID primary keys, NUMERIC precision, JSONB metadata, RLS with tenant isolation.

| Migration | Description | Key Tables |
|-----------|-------------|------------|
| V396 | Core schema + configurations | `gl_ap_configurations`, `gl_ap_engagements` |
| V397 | Evidence management | `gl_ap_evidence_packages`, `gl_ap_evidence_items`, `gl_ap_evidence_links` |
| V398 | Readiness assessment | `gl_ap_readiness_assessments`, `gl_ap_checklist_items`, `gl_ap_gaps` |
| V399 | Calculation provenance | `gl_ap_provenance_chains`, `gl_ap_provenance_steps` |
| V400 | Control testing | `gl_ap_controls`, `gl_ap_control_tests`, `gl_ap_deficiencies` |
| V401 | Verifier collaboration | `gl_ap_queries`, `gl_ap_findings`, `gl_ap_responses` |
| V402 | Materiality + sampling | `gl_ap_materiality_assessments`, `gl_ap_sampling_plans`, `gl_ap_sample_selections` |
| V403 | Regulatory requirements | `gl_ap_jurisdictions`, `gl_ap_requirements`, `gl_ap_compliance_status` |
| V404 | Cost + timeline | `gl_ap_cost_estimates`, `gl_ap_timeline_milestones`, `gl_ap_resource_plans` |
| V405 | Views, indexes, seed data | Materialised views, composite indexes, 25 seed controls, 12 jurisdiction seeds, ISAE 3410 checklist seeds |

---

## 4. Testing Strategy

### 4.1 Test File Plan (~18 files, ~7,000+ lines, 550+ test functions)

1. `conftest.py` (~550 lines) - Shared fixtures, mock evidence, sample controls, synthetic engagements
2. `test_evidence_consolidation_engine.py` - Cross-scope evidence, quality grading, completeness
3. `test_readiness_assessment_engine.py` - ISAE 3410/ISO 14064-3/AA1000AS checklists, scoring
4. `test_calculation_provenance_engine.py` - Hash chain generation, completeness, gap detection
5. `test_control_testing_engine.py` - 25 controls, design/operating effectiveness, deficiency classification
6. `test_verifier_collaboration_engine.py` - IR/query/finding lifecycle, SLA tracking
7. `test_materiality_engine.py` - Quantitative/qualitative materiality, performance materiality
8. `test_sampling_plan_engine.py` - MUS, stratification, sample size, projected misstatement
9. `test_regulatory_requirement_engine.py` - 12 jurisdictions, requirement mapping, gap analysis
10. `test_cost_timeline_engine.py` - Cost model, timeline estimation, resource planning
11. `test_reporting_engine.py` - Report generation, format validation, provenance
12. `test_workflows.py` - All 8 workflows, phase progression, error handling
13. `test_integrations.py` - All 12 integrations, bridge connections, orchestrator
14. `test_templates.py` - All 10 templates, render and export validation
15. `test_config.py` - Config lifecycle, preset loading, validation
16. `test_pack_yaml.py` - Manifest validation
17. `e2e/test_e2e.py` (~1,300 lines) - Full end-to-end assurance prep scenarios

---

## 5. Performance Requirements

| Metric | Target | Context |
|--------|--------|---------|
| Evidence package generation | <60 minutes for full S1+S2+S3 | Cross-scope consolidation |
| Readiness assessment | <5 minutes per standard | Checklist scoring |
| Provenance chain generation | <30 seconds per scope | SHA-256 hash chains |
| Control self-assessment | <15 minutes for 25 controls | Design + operating effectiveness |
| Regulatory mapping | <10 seconds for all jurisdictions | Requirement lookup |
| Full pipeline | <90 minutes for complete assurance prep | All engines sequential |

---

## 6. Security & Compliance

- Evidence items may contain commercially sensitive data; encryption at rest via SEC-003
- Verifier access restricted to read-only evidence, write-only queries/findings
- Tenant isolation via RLS on all tables
- SHA-256 provenance hash on every result
- Audit trail for all configuration and evidence changes
- RBAC: `assurance:read`, `assurance:write`, `assurance:admin`, `assurance:verifier` permissions

---

## 7. Glossary

| Term | Definition |
|------|-----------|
| AA1000AS | AccountAbility Assurance Standard v3 |
| COSO | Committee of Sponsoring Organisations of the Treadway Commission |
| CSRD | Corporate Sustainability Reporting Directive (EU) |
| IR | Information Request (from verifier to client) |
| ISAE 3410 | International Standard on Assurance Engagements - GHG Statements |
| ISO 14064-3 | GHG Specification for Verification and Validation |
| LAF | Large Accelerated Filer (US SEC) |
| MUS | Monetary Unit Sampling |
| SB 253 | California Senate Bill 253 (Climate Corporate Data Accountability Act) |
| SSAE 18 | Statement on Standards for Attestation Engagements No. 18 (AICPA) |

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-26 | GreenLang Product Team | Initial production release |
