# PRD: PACK-019 CSDDD Readiness Pack

**Pack Name:** PACK-019 CSDDD Readiness Pack
**Category:** EU Compliance Packs (Solution Packs)
**Tier:** Standalone
**Version:** 1.0.0
**Author:** GreenLang Platform Team
**Date:** March 2026
**Status:** Approved

---

## 1. Executive Summary

PACK-019 delivers a comprehensive readiness assessment and compliance management platform for the EU Corporate Sustainability Due Diligence Directive (CSDDD) - Directive (EU) 2024/1760. The CSDDD requires in-scope companies to integrate human rights and environmental due diligence into their governance, identify and address adverse impacts throughout their chains of activities, establish complaints mechanisms, and adopt climate transition plans aligned with the Paris Agreement.

This pack provides 8 calculation engines, 8 workflows, 8 report templates, 10 integrations, and 6 sector presets enabling companies to assess their CSDDD readiness, build due diligence processes, manage value chain risks, and prepare for supervisory authority oversight.

### Key Capabilities
- Due diligence obligation assessment per Articles 5-16
- Value chain mapping with adverse impact identification (human rights + environmental)
- Risk-based prioritization using severity and likelihood methodology
- Prevention and mitigation measure tracking with effectiveness monitoring
- Complaints mechanism / grievance procedure design and management
- Climate transition plan assessment per Article 22 (Paris Agreement alignment)
- Civil liability exposure assessment per Article 29
- Stakeholder engagement tracking per Article 11
- Cross-regulation consistency (CSRD/ESRS S1-S4, EUDR, Conflict Minerals)
- Supervisory authority readiness and administrative sanction risk assessment
- Phased compliance timeline management (2027/2028/2029 thresholds)

---

## 2. Regulatory Basis

### 2.1 Primary Regulation
**Corporate Sustainability Due Diligence Directive (CSDDD)**
- **Reference:** Directive (EU) 2024/1760
- **Adopted:** 13 June 2024
- **Published:** OJ L, 5 July 2024
- **Entry into force:** 25 July 2024
- **Transposition deadline:** 26 July 2026
- **Application dates:**
  - Phase 1 (26 July 2027): Companies with >5,000 employees AND >€1,500M worldwide net turnover
  - Phase 2 (26 July 2028): Companies with >3,000 employees AND >€900M worldwide net turnover
  - Phase 3 (26 July 2029): Companies with >1,000 employees AND >€450M worldwide net turnover
  - Non-EU companies: Same turnover thresholds for EU-generated turnover

### 2.2 Core Articles Covered

| Article | Requirement | Pack Engine |
|---------|------------|-------------|
| Art 5 | Due diligence integration into policies and risk management | Due Diligence Policy Engine |
| Art 6-7 | Identifying actual and potential adverse impacts | Adverse Impact Identification Engine |
| Art 8 | Preventing potential adverse impacts | Prevention & Mitigation Engine |
| Art 9 | Bringing actual adverse impacts to an end | Prevention & Mitigation Engine |
| Art 10 | Remediation of actual adverse impacts | Remediation Tracking Engine |
| Art 11 | Meaningful stakeholder engagement | Stakeholder Engagement Engine |
| Art 12 | Notification mechanism and complaints procedure | Grievance Mechanism Engine |
| Art 13 | Monitoring effectiveness | Monitoring & KPI Engine |
| Art 14 | Reporting and communicating | Reporting covered via CSRD bridge |
| Art 22 | Climate transition plan | Climate Transition Engine |
| Art 29 | Civil liability | Civil Liability Engine |
| Art 17-20 | Supervisory authorities and administrative sanctions | Due Diligence Policy Engine |

### 2.3 Secondary Regulations and Standards
- **CSRD (EU) 2022/2464** - ESRS S1-S4 (Own Workforce, Workers in Value Chain, Affected Communities, Consumers/End-Users) alignment
- **ESRS G1** - Governance disclosures relevant to due diligence
- **EUDR (EU) 2023/1115** - Deforestation due diligence overlap
- **EU Conflict Minerals Regulation (EU) 2017/821** - Mineral supply chain due diligence
- **OECD Due Diligence Guidance for Responsible Business Conduct (2018)** - Six-step framework
- **UN Guiding Principles on Business and Human Rights (2011)** - Protect, Respect, Remedy framework
- **ILO Core Conventions** - Forced labour, child labour, discrimination, freedom of association
- **International Bill of Human Rights** - UDHR, ICCPR, ICESCR
- **Paris Agreement** - 1.5°C alignment for transition plans
- **Minamata Convention** - Mercury pollution
- **Stockholm Convention** - Persistent Organic Pollutants
- **Basel Convention** - Hazardous waste transboundary movement
- **CITES** - Endangered species trade
- **CBD Kunming-Montreal GBF** - Biodiversity framework

### 2.4 Adverse Impacts Annex (Part I & Part II)

**Part I - Human Rights Adverse Impacts:**
1. Right to life, liberty, security
2. Freedom from torture, cruel treatment
3. Freedom from slavery, forced labour, child labour
4. Right to fair working conditions (health & safety, wages, working hours)
5. Freedom of association, collective bargaining
6. Non-discrimination
7. Right to privacy
8. Right to adequate standard of living (food, water, housing, health)
9. Rights of the child
10. Rights of indigenous peoples (FPIC)
11. Freedom of expression
12. Right to participate in cultural life

**Part II - Environmental Adverse Impacts:**
1. Pollution of air, water, soil
2. Harmful emissions (GHG, persistent pollutants)
3. Unsustainable use of natural resources (land, water, biodiversity)
4. Ecosystem degradation, deforestation, habitat loss
5. Biodiversity loss
6. Hazardous waste generation and transboundary movement
7. Mercury use and emissions
8. Persistent organic pollutants
9. Climate change contribution exceeding Paris Agreement goals

---

## 3. Architecture

### 3.1 Engines (8)

| # | Engine | Class | Purpose |
|---|--------|-------|---------|
| 1 | Due Diligence Policy Engine | `DueDiligencePolicyEngine` | Assess DD policy completeness per Art 5, integration into governance, code of conduct evaluation, scope threshold assessment (Phase 1/2/3) |
| 2 | Adverse Impact Identification Engine | `AdverseImpactEngine` | Map value chain, identify actual/potential human rights and environmental adverse impacts per Art 6-7, severity/likelihood scoring |
| 3 | Prevention & Mitigation Engine | `PreventionMitigationEngine` | Track prevention actions for potential impacts (Art 8) and corrective actions for actual impacts (Art 9), effectiveness measurement |
| 4 | Remediation Tracking Engine | `RemediationTrackingEngine` | Manage remediation of actual adverse impacts per Art 10, track remediation measures, victim engagement, financial provisions |
| 5 | Grievance Mechanism Engine | `GrievanceMechanismEngine` | Design and assess complaints procedures per Art 12, track submissions, assess accessibility, measure response effectiveness |
| 6 | Stakeholder Engagement Engine | `StakeholderEngagementEngine` | Plan and track meaningful stakeholder engagement per Art 11, affected stakeholder identification, engagement quality assessment |
| 7 | Climate Transition Engine | `ClimateTransitionEngine` | Assess climate transition plan per Art 22, Paris alignment check, target validation, emission reduction pathway, implementation milestones |
| 8 | Civil Liability Engine | `CivilLiabilityEngine` | Assess civil liability exposure per Art 29, identify liability triggers, evaluate defence positions, estimate exposure, assess insurance adequacy |

### 3.2 Workflows (8)

| # | Workflow | Class | Phases |
|---|----------|-------|--------|
| 1 | Due Diligence Assessment | `DueDiligenceAssessmentWorkflow` | 5-phase: Scope Determination -> Policy Review -> Gap Analysis -> Risk Prioritization -> Readiness Scoring |
| 2 | Value Chain Mapping | `ValueChainMappingWorkflow` | 4-phase: Supplier Mapping -> Tier Identification -> Activity Classification -> Risk Overlay |
| 3 | Impact Assessment | `ImpactAssessmentWorkflow` | 4-phase: Impact Scanning -> Severity/Likelihood Scoring -> Prioritization -> Stakeholder Validation |
| 4 | Prevention Planning | `PreventionPlanningWorkflow` | 4-phase: Measure Design -> Resource Allocation -> Implementation Timeline -> Effectiveness Metrics |
| 5 | Grievance Management | `GrievanceManagementWorkflow` | 4-phase: Mechanism Design -> Channel Setup -> Case Processing -> Resolution Tracking |
| 6 | Monitoring & Review | `MonitoringReviewWorkflow` | 4-phase: KPI Definition -> Data Collection -> Performance Analysis -> Annual Review |
| 7 | Climate Transition Planning | `ClimateTransitionWorkflow` | 4-phase: Baseline Assessment -> Target Setting -> Pathway Design -> Progress Tracking |
| 8 | Regulatory Submission | `RegulatorySubmissionWorkflow` | 4-phase: Documentation Assembly -> Supervisory Readiness Check -> Submission Package -> Compliance Tracking |

### 3.3 Templates (8)

| # | Template | Class | Purpose |
|---|----------|-------|---------|
| 1 | Due Diligence Readiness Report | `DDReadinessReportTemplate` | Overall CSDDD readiness assessment with article-by-article compliance status |
| 2 | Value Chain Risk Map | `ValueChainRiskMapTemplate` | Visual value chain with risk overlay showing adverse impact hotspots |
| 3 | Impact Assessment Report | `ImpactAssessmentReportTemplate` | Detailed adverse impact identification with severity/likelihood matrix |
| 4 | Prevention & Mitigation Report | `PreventionMitigationReportTemplate` | Prevention measures, corrective actions, and effectiveness tracking |
| 5 | Grievance Mechanism Report | `GrievanceMechanismReportTemplate` | Complaints procedure assessment, case statistics, resolution tracking |
| 6 | Stakeholder Engagement Report | `StakeholderEngagementReportTemplate` | Engagement activities, stakeholder feedback, consultation outcomes |
| 7 | Climate Transition Plan Report | `ClimateTransitionReportTemplate` | Art 22 transition plan with targets, pathway, and Paris alignment assessment |
| 8 | CSDDD Compliance Scorecard | `CSDDDScorecardTemplate` | Executive dashboard with article-by-article status and trend analysis |

### 3.4 Integrations (10)

| # | Integration | Class | Purpose |
|---|-------------|-------|---------|
| 1 | Pack Orchestrator | `CSDDDOrchestrator` | Master pipeline orchestrating all CSDDD assessment phases |
| 2 | CSRD Pack Bridge | `CSRDPackBridge` | Maps ESRS S1-S4/G1 disclosures to CSDDD requirements |
| 3 | MRV Bridge | `MRVBridge` | Routes MRV emission data for climate transition plan substantiation |
| 4 | EUDR Bridge | `EUDRBridge` | Connects EUDR due diligence data for deforestation-related adverse impacts |
| 5 | Supply Chain Bridge | `SupplyChainBridge` | Links supply chain mapping agents for value chain due diligence |
| 6 | Data Bridge | `DataBridge` | Routes data intake agents for supplier questionnaires, spend data |
| 7 | Green Claims Bridge | `GreenClaimsBridge` | Cross-validates CSDDD remediation claims with Green Claims Directive |
| 8 | Taxonomy Bridge | `TaxonomyBridge` | Validates DNSH criteria alignment with CSDDD environmental impacts |
| 9 | Health Check | `CSDDDHealthCheck` | System verification across all engines and bridges |
| 10 | Setup Wizard | `CSDDDSetupWizard` | Guided configuration for sector, scope, value chain structure |

### 3.5 Presets (6)

| # | Preset | Sector | Key Focus |
|---|--------|--------|-----------|
| 1 | Manufacturing | MANUFACTURING | Supply chain labour risks, process pollution, raw material sourcing |
| 2 | Extractives | EXTRACTIVES | Indigenous rights, environmental degradation, conflict minerals |
| 3 | Financial Services | FINANCIAL_SERVICES | Financed impacts, portfolio screening, ESG integration |
| 4 | Retail & Consumer | RETAIL | Apparel supply chains, food sourcing, consumer product safety |
| 5 | Technology | TECHNOLOGY | Mineral supply chains, data privacy, labour in electronics manufacturing |
| 6 | Agriculture & Food | AGRICULTURE | Land rights, deforestation, labour exploitation, water stress |

---

## 4. OECD Six-Step Due Diligence Framework Alignment

The CSDDD is explicitly aligned with the OECD Due Diligence Guidance. Each engine maps to OECD steps:

| OECD Step | Description | PACK-019 Engine |
|-----------|-------------|-----------------|
| Step 1 | Embed responsible business conduct into policies and management systems | Due Diligence Policy Engine |
| Step 2 | Identify and assess actual and potential adverse impacts | Adverse Impact Identification Engine |
| Step 3 | Cease, prevent, and mitigate adverse impacts | Prevention & Mitigation Engine |
| Step 4 | Track implementation and results | Monitoring & KPI Engine (via Climate Transition Engine metrics) |
| Step 5 | Communicate how impacts are addressed | CSRD Bridge (ESRS reporting) |
| Step 6 | Provide for or cooperate in remediation | Remediation Tracking Engine |

---

## 5. Data Models

### 5.1 Key Enums
- `CompanyScope` - PHASE_1, PHASE_2, PHASE_3, NOT_IN_SCOPE, VOLUNTARY
- `AdverseImpactType` - HUMAN_RIGHTS, ENVIRONMENTAL
- `ImpactSeverity` - CRITICAL, HIGH, MEDIUM, LOW
- `ImpactLikelihood` - VERY_LIKELY, LIKELY, POSSIBLE, UNLIKELY, RARE
- `ImpactStatus` - ACTUAL, POTENTIAL
- `ValueChainPosition` - OWN_OPERATIONS, UPSTREAM_DIRECT, UPSTREAM_INDIRECT, DOWNSTREAM_DIRECT, DOWNSTREAM_INDIRECT
- `MeasureType` - PREVENTION, MITIGATION, REMEDIATION, CESSATION
- `ComplianceStatus` - COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT, NOT_APPLICABLE
- `ArticleReference` - ART_5 through ART_29
- `GrievanceStatus` - RECEIVED, UNDER_REVIEW, INVESTIGATING, RESOLVED, CLOSED, ESCALATED
- `TransitionPlanStatus` - DRAFTED, APPROVED, IMPLEMENTING, ON_TRACK, BEHIND_SCHEDULE, ACHIEVED
- `StakeholderGroup` - WORKERS, TRADE_UNIONS, COMMUNITIES, INDIGENOUS_PEOPLES, NGOS, INVESTORS, CONSUMERS, REGULATORS

### 5.2 Key Models (Pydantic BaseModel)
- `CompanyProfile` - employee_count, turnover_eur, eu_turnover_eur, sector, value_chain_tiers, scope_phase
- `AdverseImpact` - impact_id, type, category, description, severity, likelihood, status, value_chain_position, affected_stakeholders, linked_rights
- `PreventionMeasure` - measure_id, type, description, target_impact_ids, responsible_person, deadline, budget_eur, effectiveness_score
- `GrievanceCase` - case_id, status, submitted_by, stakeholder_group, description, adverse_impact_ref, resolution, days_to_resolve
- `ClimateTarget` - target_id, scope, base_year, target_year, reduction_pct, aligned_with_15c, interim_milestones
- `RemediationAction` - action_id, adverse_impact_id, description, financial_provision_eur, victim_engagement, completion_status

---

## 6. Agent Dependencies

### 6.1 Total Agent Count: 64
- **MRV Agents:** 30 (Scope 1-3 emissions for climate transition plan)
- **Data Agents:** 20 (Supplier questionnaires, spend categorization, data quality)
- **Foundation Agents:** 10 (Orchestrator, schema, units, citations, access)
- **Bridged Agents:** 4 (Supply chain mapping, risk assessment, due diligence, stakeholder engagement)

### 6.2 Key EUDR Agent Reuse
The AGENT-EUDR due diligence agents (021-040) provide substantial reuse:
- AGENT-EUDR-026: Due Diligence Orchestrator (DD process management)
- AGENT-EUDR-031: Stakeholder Engagement Tool
- AGENT-EUDR-032: Grievance Mechanism Manager
- AGENT-EUDR-033: Continuous Monitoring Agent
- AGENT-EUDR-034: Annual Review Scheduler
- AGENT-EUDR-035: Improvement Plan Creator
- AGENT-EUDR-037: Due Diligence Statement Creator

---

## 7. Testing Strategy

### 7.1 Test Structure
- `conftest.py` - Dynamic module loading with `pack019_test.*` namespace
- 8 engine test files (~50 tests each)
- `test_templates.py` - All 8 templates (~45 tests)
- `test_workflows.py` - All 8 workflows (~40 tests)
- `test_integrations.py` - All 10 integrations (~35 tests)
- `test_config.py` - PackConfig and presets (~30 tests)
- `test_manifest.py` - pack.yaml validation (~25 tests)
- `test_demo.py` - Demo config (~15 tests)
- `test_agent_integration.py` - Agent bridges (~30 tests)
- `test_e2e.py` - End-to-end pipeline (~20 tests)

### 7.2 Test Targets
- Total tests: 700+
- Pass rate: 100%
- Coverage: All engines, workflows, templates, integrations, config

---

## 8. Performance Targets

| Component | Metric | Target |
|-----------|--------|--------|
| Due Diligence Assessment | Max company profiles | 1,000 |
| Value Chain Mapping | Max suppliers mapped | 50,000 |
| Impact Assessment | Max adverse impacts tracked | 10,000 |
| Grievance Processing | Max cases processed | 5,000 |
| Climate Transition | Max targets evaluated | 500 |
| Civil Liability | Max exposure assessments | 1,000 |
| Cache hit ratio | Overall | 65% |
| Memory ceiling | Peak | 8,192 MB |

---

## 9. Security & Access Control

- **Authentication:** JWT (RS256)
- **Authorization:** RBAC with impact-level and case-level access control
- **Encryption at rest:** AES-256-GCM
- **Encryption in transit:** TLS 1.3
- **Audit logging:** All due diligence activities logged
- **PII redaction:** Grievance case submitter data protected
- **Required roles:** csddd_manager, due_diligence_officer, compliance_analyst, human_rights_lead, environmental_officer, supply_chain_manager, legal_counsel, external_auditor, admin

---

## 10. Compliance Timeline

| Phase | Date | Threshold | Companies |
|-------|------|-----------|-----------|
| Transposition | 26 July 2026 | N/A | Member states adopt national laws |
| Phase 1 | 26 July 2027 | >5,000 employees AND >€1,500M turnover | ~1,300 EU companies |
| Phase 2 | 26 July 2028 | >3,000 employees AND >€900M turnover | ~3,400 EU companies |
| Phase 3 | 26 July 2029 | >1,000 employees AND >€450M turnover | ~13,000 EU companies |
| Non-EU Phase 1 | 26 July 2029 | >€450M EU-generated turnover | ~4,000 non-EU companies |

---

*End of PRD*
