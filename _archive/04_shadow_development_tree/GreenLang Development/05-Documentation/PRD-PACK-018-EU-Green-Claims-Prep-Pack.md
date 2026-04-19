# PRD-PACK-018: EU Green Claims Prep Pack

**Pack ID:** PACK-018-eu-green-claims-prep
**Category:** EU Compliance / Consumer Protection
**Tier:** Standalone (Cross-Sector)
**Version:** 1.0.0
**Status:** Approved
**Author:** GreenLang Product Team
**Date:** 2026-03-17

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Green Claims Directive (Directive proposal 2023/0085/COD) and the closely related Empowering Consumers for the Green Transition Directive (Directive (EU) 2024/825) fundamentally change how companies can make environmental claims in the European Union. Starting from 2026-2027, every explicit environmental claim must be substantiated through a life-cycle-based assessment, verified by an accredited third-party verifier, and communicated in a manner that is specific, accurate, and not misleading.

Current challenges facing companies:

1. **Unsubstantiated claims proliferation**: Studies show 53% of environmental claims in the EU are vague, misleading, or unfounded (EC screening, 2020). Companies face regulatory enforcement and reputational damage.
2. **Complex substantiation requirements**: The Directive requires life-cycle assessment (LCA/PEF) based evidence, covering all significant environmental impacts — not just carbon — across the full product or organization lifecycle.
3. **Third-party verification gap**: All explicit environmental claims must be verified by accredited conformity assessment bodies before being communicated to consumers. Most companies lack verification processes.
4. **Label proliferation**: Over 230 sustainability labels exist in the EU market. The Directive bans new national/regional public environmental labelling schemes and requires existing private schemes to meet new governance standards.
5. **Cross-regulation complexity**: Green claims compliance intersects with CSRD (sustainability reporting), EU Taxonomy (green investment), ESPR/DPP (digital product passports), and the Unfair Commercial Practices Directive.
6. **Penalty exposure**: Non-compliance can result in fines up to 4% of annual turnover, product market withdrawal, and exclusion from public procurement for up to 12 months.

### 1.2 Solution Overview

PACK-018 is a **standalone EU Green Claims Prep Pack** that provides end-to-end assessment, substantiation, verification-readiness, and compliance tracking for all environmental claims made by an undertaking. It implements 8 deterministic calculation engines, 8 compliance workflows, 8 report templates, and 10 integrations bridging to the GreenLang agent ecosystem.

The pack:
- **Inventories** all environmental claims across products, services, marketing, and corporate communications
- **Assesses** each claim against EU Green Claims Directive substantiation requirements using life-cycle analysis
- **Detects** greenwashing risks using the TerraChoice Seven Sins framework and EU-specific prohibited patterns
- **Tracks** evidence chains with SHA-256 provenance for audit readiness
- **Generates** verification-ready dossiers for third-party conformity assessment bodies
- **Monitors** compliance status and remediation progress

Zero-hallucination: Every score, assessment, and recommendation is produced by deterministic rule engines with Decimal arithmetic. No LLM is involved in any calculation or compliance determination path.

### 1.3 Key Differentiators

| Dimension | Manual Approach | PACK-018 Green Claims Prep Pack |
|-----------|-----------------|----------------------------------|
| Claim inventory time | 100-200 hours per product portfolio | <10 hours (automated extraction) |
| Substantiation assessment | Ad-hoc, inconsistent criteria | Systematic per Article 3-4, LCA-based |
| Greenwashing risk detection | Subjective legal review | Deterministic 7-Sins + 16-type scoring |
| Evidence chain management | Scattered files, no provenance | SHA-256 hashed, full audit trail |
| Verification readiness | Months of preparation | Pre-formatted dossiers per Article 10 |
| Cross-regulation alignment | Manual cross-referencing | Automated CSRD/Taxonomy/DPP bridging |
| Remediation tracking | Spreadsheet-based | Workflow-driven with progress scoring |

### 1.4 Target Users

**Primary:**
- Sustainability and ESG managers at companies making environmental claims
- Marketing and communications teams creating product/corporate claims
- Legal and compliance officers reviewing green claims
- Companies with consumer-facing products/services in the EU market

**Secondary:**
- Accredited verifiers conducting conformity assessments per Article 10
- Sustainability consultants preparing clients for Green Claims compliance
- National consumer protection authorities monitoring compliance
- Retailers evaluating supplier green claims

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Claim inventory completeness | >95% of active claims captured | Claims found vs. manual audit |
| Substantiation assessment accuracy | 100% aligned with Article 3-4 criteria | Validated against legal expert review |
| Greenwashing detection rate | >90% of problematic claims flagged | Compared to manual legal review |
| Verification dossier acceptance rate | >85% accepted by verifiers | First-submission acceptance rate |
| Remediation plan completion | >80% of issues resolved in 90 days | Issue close rate |
| Cross-regulation consistency | 100% alignment with CSRD/Taxonomy data | Cross-check with ESRS disclosures |

---

## 2. Regulatory Basis

### 2.1 Primary Regulations

| Regulation | Reference | Status | Key Dates |
|------------|-----------|--------|-----------|
| EU Green Claims Directive | COM(2023)166 / 2023/0085(COD) | Proposed March 2023, trilogue ongoing | Expected adoption 2025-2026, application 2026-2027 |
| Empowering Consumers Directive | Directive (EU) 2024/825 | Adopted Feb 2024 | Transposition by 27 March 2026, application by 27 September 2026 |
| Unfair Commercial Practices Directive | Directive 2005/29/EC (as amended) | In force | Amended by Directive (EU) 2024/825 |

### 2.2 Supporting Standards and Frameworks

| Standard / Framework | Reference | Green Claims Relevance |
|---------------------|-----------|----------------------|
| Product Environmental Footprint (PEF) | EU Commission Recommendation 2021/2279 | Article 3: LCA methodology for claim substantiation |
| Organisation Environmental Footprint (OEF) | EU Commission Recommendation 2021/2279 | Article 3: Organization-level claim substantiation |
| ISO 14040/14044 | Life Cycle Assessment | Accepted LCA methodology for substantiation |
| ISO 14021 | Self-declared environmental claims (Type II) | Baseline for claim classification |
| ISO 14024 | Environmental labelling (Type I) | Environmental label compliance |
| ISO 14025 | Environmental declarations (Type III) | EPD-based claim requirements |
| ISO 14067 | Carbon footprint of products | Carbon claim substantiation |
| EN 13432 | Packaging compostability | Compostable/biodegradable claim requirements |
| TerraChoice Seven Sins of Greenwashing | UL (2010) | Greenwashing detection framework |
| OECD Environmental Claims Guidelines | OECD (2011) | International best practice |
| CSRD / ESRS | Directive (EU) 2022/2464 | Cross-reference for corporate sustainability data |
| EU Taxonomy | Regulation (EU) 2020/852 | "Environmentally sustainable" investment claims |
| ESPR / DPP | Regulation (EU) 2024/1781 | Digital Product Passport environmental data |

### 2.3 Green Claims Directive Article Reference Map

| Article | Title | Requirement | Engine Coverage |
|---------|-------|-------------|-----------------|
| Art. 3 | Substantiation of explicit environmental claims | Life-cycle assessment, all significant impacts, primary data preference | claim_substantiation_engine |
| Art. 4 | Communication of environmental claims | Specific, accurate, not misleading, clear scope | claim_substantiation_engine |
| Art. 5 | Environmental claims relating to future performance | Binding targets, independent monitoring, publicly available progress | comparative_claims_engine |
| Art. 6 | Environmental labelling schemes | Governance, transparency, third-party verification, scientific evidence | label_compliance_engine |
| Art. 7 | Requirements for new environmental labelling schemes | EU-level added value, public consultation, periodic review | label_compliance_engine |
| Art. 8 | Verification of environmental claims by traders | Pre-market substantiation, evidence maintenance, claim update obligations | trader_obligation_engine |
| Art. 9 | Verification of environmental labelling schemes | Scheme governance, complaint mechanisms, annual reporting | label_compliance_engine |
| Art. 10 | Conformity assessment of environmental claims | Third-party verification by accredited body before communication | evidence_chain_engine |
| Art. 11 | Conformity assessment of environmental labelling schemes | Independent audit of scheme governance and methodology | label_compliance_engine |
| Art. 12 | SME support measures | Simplified assessment for micro-enterprises, financial/technical support | trader_obligation_engine |

### 2.4 Empowering Consumers Directive Key Amendments

| Amendment | UCPD Article | Requirement | Engine Coverage |
|-----------|-------------|-------------|-----------------|
| Generic environmental claims ban | Art. 6(2)(d) | Claims like "eco-friendly", "green", "sustainable" prohibited without proof | greenwashing_detection_engine |
| Sustainability label requirements | Art. 6(2)(e) | Labels must be based on certification or established by public authority | label_compliance_engine |
| Carbon offsetting transparency | Art. 6(2)(f) | Claims based on offsets must not imply reduced/zero environmental impact | claim_substantiation_engine |
| Durability feature claims | Art. 6(2)(g) | Durability/repairability claims must be substantiated | comparative_claims_engine |
| Misleading practices blacklist | Annex I, points 2a-4d | 10 new prohibited commercial practices | greenwashing_detection_engine |

---

## 3. Architecture

### 3.1 Engine Architecture (8 Engines)

| # | Engine | File | Class | Purpose |
|---|--------|------|-------|---------|
| 1 | Claim Substantiation | `claim_substantiation_engine.py` | `ClaimSubstantiationEngine` | Core substantiation assessment per Art. 3-4 with 5-dimension scoring |
| 2 | Evidence Chain | `evidence_chain_engine.py` | `EvidenceChainEngine` | Evidence collection, chain-of-custody, document management per Art. 10 |
| 3 | Lifecycle Assessment | `lifecycle_assessment_engine.py` | `LifecycleAssessmentEngine` | PEF/LCA-based lifecycle impact verification per Art. 3(1) |
| 4 | Label Compliance | `label_compliance_engine.py` | `LabelComplianceEngine` | Environmental label/scheme verification per Art. 6-9 |
| 5 | Comparative Claims | `comparative_claims_engine.py` | `ComparativeClaimsEngine` | Improvement and comparative claim validation per Art. 3(4), 5 |
| 6 | Greenwashing Detection | `greenwashing_detection_engine.py` | `GreenwashingDetectionEngine` | TerraChoice 7 Sins + EU prohibited practices detection |
| 7 | Trader Obligation | `trader_obligation_engine.py` | `TraderObligationEngine` | Trader/manufacturer obligation tracking per Art. 8, 12 |
| 8 | Green Claims Benchmark | `green_claims_benchmark_engine.py` | `GreenClaimsBenchmarkEngine` | Cross-portfolio scoring, peer comparison, maturity model |

### 3.2 Workflow Architecture (8 Workflows)

| # | Workflow | File | Class | Phases |
|---|----------|------|-------|--------|
| 1 | Claim Assessment | `claim_assessment_workflow.py` | `ClaimAssessmentWorkflow` | Intake → Classification → Substantiation → Risk → Report |
| 2 | Evidence Collection | `evidence_collection_workflow.py` | `EvidenceCollectionWorkflow` | Identify → Gather → Validate → Chain → Archive |
| 3 | Lifecycle Verification | `lifecycle_verification_workflow.py` | `LifecycleVerificationWorkflow` | Scope → Inventory → Impact → Interpretation → PEF |
| 4 | Label Audit | `label_audit_workflow.py` | `LabelAuditWorkflow` | Inventory → Classification → Governance → Compliance → Report |
| 5 | Greenwashing Screening | `greenwashing_screening_workflow.py` | `GreenwashingScreeningWorkflow` | Scan → Detect → Score → Prioritize → Remediate |
| 6 | Compliance Gap | `compliance_gap_workflow.py` | `ComplianceGapWorkflow` | Baseline → Requirements → Gap → Priority → Roadmap |
| 7 | Remediation Planning | `remediation_planning_workflow.py` | `RemediationPlanningWorkflow` | Assess → Plan → Resource → Schedule → Monitor |
| 8 | Regulatory Submission | `regulatory_submission_workflow.py` | `RegulatorySubmissionWorkflow` | Prepare → Package → Validate → Submit → Track |

### 3.3 Template Architecture (8 Templates)

| # | Template | File | Class | Outputs |
|---|----------|------|-------|---------|
| 1 | Claim Assessment Report | `claim_assessment_report.py` | `ClaimAssessmentReportTemplate` | MD, HTML, JSON |
| 2 | Evidence Dossier | `evidence_dossier_report.py` | `EvidenceDossierReportTemplate` | MD, HTML, JSON |
| 3 | Lifecycle Summary | `lifecycle_summary_report.py` | `LifecycleSummaryReportTemplate` | MD, HTML, JSON |
| 4 | Label Compliance Report | `label_compliance_report.py` | `LabelComplianceReportTemplate` | MD, HTML, JSON |
| 5 | Greenwashing Risk Report | `greenwashing_risk_report.py` | `GreenwashingRiskReportTemplate` | MD, HTML, JSON |
| 6 | Compliance Gap Report | `compliance_gap_report.py` | `ComplianceGapReportTemplate` | MD, HTML, JSON |
| 7 | Green Claims Scorecard | `green_claims_scorecard.py` | `GreenClaimsScorecardTemplate` | MD, HTML, JSON |
| 8 | Regulatory Submission | `regulatory_submission_report.py` | `RegulatorySubmissionReportTemplate` | MD, HTML, JSON |

### 3.4 Integration Architecture (10 Integrations)

| # | Integration | File | Class | Purpose |
|---|-------------|------|-------|---------|
| 1 | Pack Orchestrator | `pack_orchestrator.py` | `GreenClaimsOrchestrator` | 10-phase DAG pipeline orchestration |
| 2 | CSRD Pack Bridge | `csrd_pack_bridge.py` | `CSRDPackBridge` | Connect to PACK-001/002/003 CSRD data |
| 3 | MRV Claims Bridge | `mrv_claims_bridge.py` | `MRVClaimsBridge` | Route to 30 MRV agents for emission verification |
| 4 | Data Claims Bridge | `data_claims_bridge.py` | `DataClaimsBridge` | Route to 20 DATA agents for evidence gathering |
| 5 | Taxonomy Bridge | `taxonomy_bridge.py` | `TaxonomyBridge` | EU Taxonomy alignment verification |
| 6 | PEF Bridge | `pef_bridge.py` | `PEFBridge` | Product Environmental Footprint data exchange |
| 7 | DPP Bridge | `dpp_bridge.py` | `DPPBridge` | Digital Product Passport integration |
| 8 | ECGT Bridge | `ecgt_bridge.py` | `ECGTBridge` | Empowering Consumers Directive compliance |
| 9 | Health Check | `health_check.py` | `GreenClaimsHealthCheck` | 20-category pack verification |
| 10 | Setup Wizard | `setup_wizard.py` | `GreenClaimsSetupWizard` | 8-step configuration wizard |

### 3.5 Configuration Architecture

| Component | File | Description |
|-----------|------|-------------|
| Pack Config | `config/pack_config.py` | Main configuration with 14 enums, 12 sub-config models, 2 main config classes |
| Manufacturing Preset | `config/presets/manufacturing.yaml` | Industrial/manufacturing sector defaults |
| Retail Preset | `config/presets/retail.yaml` | Retail/consumer goods sector defaults |
| Financial Services Preset | `config/presets/financial_services.yaml` | Banking/insurance/asset management defaults |
| Energy Preset | `config/presets/energy.yaml` | Energy/utilities sector defaults |
| Technology Preset | `config/presets/technology.yaml` | Tech/SaaS sector defaults |
| SME Preset | `config/presets/sme.yaml` | Small/medium enterprise simplified defaults |
| Demo Config | `config/demo/demo_config.yaml` | EcoProducts GmbH demo scenario |

---

## 4. Claim Type Taxonomy

### 4.1 Supported Claim Types (16 Categories)

| # | Claim Type | Risk Level | Substantiation Requirements | Directive Reference |
|---|-----------|-----------|---------------------------|-------------------|
| 1 | Carbon Neutral | HIGH | Full GHG inventory (Scope 1+2+3), reduction pathway, offset registry, 3rd-party verification | Art. 3, ECGT Art. 6(2)(f) |
| 2 | Climate Positive | HIGH | Net negative calculation, full lifecycle, permanent removals, 3rd-party verification | Art. 3 |
| 3 | Net Zero | HIGH | SBTi commitment, 90%+ real reduction before offsetting, Scope 3 included | Art. 3, Art. 5 |
| 4 | Carbon Negative | HIGH | More CO2 removed than emitted, permanence proof, MRV verification | Art. 3 |
| 5 | Eco-Friendly | CRITICAL | BANNED as generic claim per ECGT — must be qualified with specific, measurable aspects | ECGT Annex I(2a) |
| 6 | Sustainable | CRITICAL | BANNED as generic claim per ECGT — must specify environmental, social, or economic dimension | ECGT Annex I(2a) |
| 7 | Green | CRITICAL | BANNED as generic claim per ECGT — requires full product/org lifecycle evidence | ECGT Annex I(2a) |
| 8 | Renewable | MEDIUM | Energy certificates (GO/REC), percentage specification, additionality evidence | Art. 3 |
| 9 | Recyclable | MEDIUM | Material composition, recycling infrastructure availability in claim geography | Art. 3(1)(e) |
| 10 | Biodegradable | HIGH | Test results per EN 13432, timeframe and conditions specification | Art. 3, Art. 4 |
| 11 | Compostable | HIGH | EN 13432 certification, industrial vs. home composting specification | Art. 3, Art. 4 |
| 12 | Plastic-Free | MEDIUM | 100% plastic-free verification across full product and packaging | Art. 3, Art. 4 |
| 13 | Zero Waste | HIGH | >90% diversion rate per ZWIA definition, methodology disclosure | Art. 3, Art. 4 |
| 14 | Low Carbon | MEDIUM | Carbon footprint calculation, benchmark comparison, methodology | Art. 3 |
| 15 | Reduced Emissions | MEDIUM | Baseline/current comparison, percentage reduction, methodology | Art. 3(4), Art. 5 |
| 16 | Environmentally Friendly | CRITICAL | BANNED as generic claim per ECGT — same as "eco-friendly" | ECGT Annex I(2a) |

### 4.2 Substantiation Scoring (5 Dimensions)

| Dimension | Weight | Description | Assessment Criteria |
|-----------|--------|-------------|-------------------|
| Scientific Validity | 30% | Quality and relevance of scientific evidence | LCA/PEF methodology, peer-reviewed data, recognized standards |
| Data Quality | 25% | Reliability and completeness of underlying data | Primary vs. secondary data, measurement methods, data age |
| Scope Completeness | 20% | Coverage of full lifecycle and significant impacts | Lifecycle stages covered, environmental impact categories, exclusions justified |
| Verification Independence | 15% | Level of third-party verification | Accredited body, ISO 17065, verification scope and frequency |
| Transparency | 10% | Clarity and accessibility of claim communication | Specificity, qualification, consumer accessibility, methodology disclosure |

---

## 5. Greenwashing Detection Framework

### 5.1 TerraChoice Seven Sins

| Sin | Description | Detection Pattern | Severity |
|-----|------------|-------------------|----------|
| Sin of Hidden Trade-Off | Claim based on narrow set of attributes while ignoring broader impacts | Single lifecycle phase only, <3 impact categories assessed | HIGH |
| Sin of No Proof | Claim not backed by accessible evidence or certification | No evidence provided, no verifier reference, no methodology | CRITICAL |
| Sin of Vagueness | Claim so broad it is likely to be misunderstood by consumer | Generic terms ("green", "eco", "natural") without qualification | HIGH |
| Sin of Worshipping False Labels | Product gives impression of third-party endorsement where none exists | Self-declared "certified" without accredited body, fake label graphics | CRITICAL |
| Sin of Irrelevance | Claim may be truthful but is irrelevant or unhelpful | Claims about legally mandated attributes (CFC-free since 1990s) | MEDIUM |
| Sin of Lesser of Two Evils | Claim may be true within category but distracts from greater impacts | Green claims on inherently high-impact product categories | HIGH |
| Sin of Fibbing | Claim is simply false | Factual inaccuracy in emission numbers, fake certification references | CRITICAL |

### 5.2 EU-Specific Prohibited Practices (per ECGT Annex I)

| # | Prohibited Practice | Detection Rule | Directive Reference |
|---|-------------------|---------------|-------------------|
| 1 | Generic environmental excellence claims without substantiation | Keyword match: "eco-friendly", "green", "sustainable", "natural", "environmentally friendly" without qualifier | ECGT Annex I, point 2a |
| 2 | Displaying sustainability labels not based on certification/public authority | Label without ISO 14024 / accredited scheme reference | ECGT Annex I, point 2b |
| 3 | Making environmental claims about entire product when only part qualifies | Scope mismatch: claim scope > evidence scope | ECGT Annex I, point 2c |
| 4 | Presenting legal requirements as distinctive features | Claiming compliance with mandatory regulation as voluntary achievement | ECGT Annex I, point 4a |
| 5 | Claims based on greenhouse gas emission offsets implying neutrality | "Carbon neutral" based >50% on offsets without disclosure | ECGT Annex I, point 4b |
| 6 | Future performance claims without clear, binding commitments | "Will be carbon neutral by 2030" without SBTi validation or binding roadmap | Art. 5 |

---

## 6. Agent Dependencies

### 6.1 GreenLang Agent Ecosystem Integration

| Agent Category | Agent Count | Integration Purpose |
|---------------|-------------|-------------------|
| AGENT-MRV (001-030) | 30 | GHG emission verification for carbon claims (Scope 1+2+3) |
| AGENT-DATA (001-020) | 20 | Evidence gathering, data quality profiling, validation |
| AGENT-FOUND (001-010) | 10 | Schema validation, reproducibility, audit trail |
| AGENT-EUDR (001-040) | 40 | Supply chain traceability for deforestation-free claims |
| GL-008 | 1 | Green Claims Verification Agent (core reuse) |
| GL-009 | 1 | Product Carbon Footprint Agent |
| GL-077 | 1 | Life Cycle Assessment Agent |
| GL-078 | 1 | Circular Economy Agent |

### 6.2 Pack Dependencies

| Pack | Bridge | Data Exchange |
|------|--------|--------------|
| PACK-001/002/003 (CSRD) | csrd_pack_bridge | ESRS disclosure data, materiality assessment |
| PACK-008 (Taxonomy) | taxonomy_bridge | Taxonomy alignment for "sustainable" claims |
| PACK-015 (Double Materiality) | via CSRD bridge | DMA results for claim relevance |
| PACK-016 (ESRS E1) | via MRV bridge | E1 climate data for carbon claims |
| PACK-017 (ESRS Full) | via CSRD bridge | Full ESRS data for environmental claims |

---

## 7. Data Model Summary

### 7.1 Core Models

- `EnvironmentalClaim` — Individual claim with text, type, scope, product/org reference
- `ClaimEvidence` — Evidence item with type, source, validity period, verification status
- `SubstantiationAssessment` — 5-dimension scoring result per claim
- `GreenwashingAlert` — Detected greenwashing pattern with severity and recommendation
- `LifecycleImpact` — PEF impact category result per lifecycle phase
- `LabelAssessment` — Environmental label compliance evaluation
- `ComplianceGap` — Identified gap between current state and regulatory requirement
- `RemediationAction` — Planned corrective action with timeline and resources
- `VerificationDossier` — Pre-formatted package for third-party verifier
- `TraderObligation` — Obligation tracking record per Article 8
- `BenchmarkResult` — Cross-portfolio and peer comparison metrics

### 7.2 Result Models

- `ClaimSubstantiationResult` — Full assessment output with provenance hash
- `EvidenceChainResult` — Evidence chain verification output
- `LifecycleAssessmentResult` — LCA/PEF summary with impact categories
- `LabelComplianceResult` — Label audit output per scheme
- `ComparativeClaimResult` — Comparative claim validation output
- `GreenwashingScreeningResult` — Portfolio-wide greenwashing risk assessment
- `TraderObligationResult` — Obligation compliance status
- `BenchmarkOutput` — Scoring and maturity level output

---

## 8. Testing Strategy

### 8.1 Test Coverage Targets

| Category | Test Count | Coverage |
|----------|-----------|----------|
| Engine unit tests (8 engines × ~50 tests) | ~400 | All calculation paths, edge cases, provenance |
| Workflow integration tests (8 workflows × ~15 tests) | ~120 | Phase execution, error handling, result format |
| Template rendering tests (8 templates × ~8 tests) | ~64 | Markdown, HTML, JSON output, sections |
| Integration bridge tests (10 bridges × ~8 tests) | ~80 | Config loading, routing, health checks |
| Config and preset tests | ~40 | All presets, validation, demo loading |
| Manifest and structural tests | ~20 | pack.yaml parsing, file existence |
| End-to-end tests | ~20 | Full pipeline execution |
| **Total** | **~744** | **85%+ line coverage** |

### 8.2 Zero-Hallucination Verification

Every engine test verifies:
- Decimal arithmetic (no floating-point) for all percentage calculations
- SHA-256 provenance hash on every result
- Deterministic output (same inputs → same outputs, including hash)
- No LLM dependency in any calculation path
- Pydantic model validation on all inputs and outputs

---

## 9. Deployment

### 9.1 Prerequisites

- Python ≥ 3.11
- pydantic ≥ 2.0
- pyyaml ≥ 6.0
- GreenLang Platform ≥ 2.0

### 9.2 Installation

```bash
gl pack install PACK-018-eu-green-claims-prep
```

### 9.3 Quick Start

```python
from packs.eu_compliance.PACK_018_eu_green_claims_prep.config import PackConfig
from packs.eu_compliance.PACK_018_eu_green_claims_prep.integrations import GreenClaimsOrchestrator

config = PackConfig.from_preset("retail")
orchestrator = GreenClaimsOrchestrator(config)
result = orchestrator.execute(claims=my_claims, evidence=my_evidence)
```

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|-----------|
| Explicit environmental claim | Any claim in text or visual form, including labels, that states or implies environmental benefit or reduced impact |
| Substantiation | The process of proving a claim is accurate through life-cycle assessment and scientific evidence |
| Conformity assessment | Third-party verification by an accredited body per Regulation (EC) 765/2008 |
| PEF | Product Environmental Footprint — EU methodology for measuring environmental performance of products |
| OEF | Organisation Environmental Footprint — EU methodology for organization-level environmental measurement |
| PEFCR | Product Environmental Footprint Category Rules — sector-specific PEF methodology |
| Greenwashing | Making misleading, unsubstantiated, or false environmental claims |
| TerraChoice Seven Sins | Framework identifying 7 common patterns of greenwashing |
| ECGT | Empowering Consumers for the Green Transition Directive (EU) 2024/825 |
| UCPD | Unfair Commercial Practices Directive 2005/29/EC |

### 10.2 Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-17 | GreenLang Product Team | Initial release |
