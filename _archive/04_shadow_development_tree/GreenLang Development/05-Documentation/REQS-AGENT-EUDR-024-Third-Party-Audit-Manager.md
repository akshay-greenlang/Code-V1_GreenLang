# REQS: AGENT-EUDR-024 -- Third-Party Audit Manager Agent
# Regulatory Requirements Document

## Document Info

| Field | Value |
|-------|-------|
| **REQS ID** | REQS-AGENT-EUDR-024 |
| **Agent ID** | GL-EUDR-TAM-024 |
| **Component** | Third-Party Audit Manager Agent |
| **Category** | EUDR Regulatory Agent -- Audit, Verification & Certification Compliance |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Author** | GL-RegulatoryIntelligence |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 (EUDR), Articles 10, 11, 14-16, 29, 30-31; ISO 19011:2018; ISO/IEC 17065:2012; ISO/IEC 17021-1:2015; FSC-STD-20-011; RSPO SCCS; PEFC ST 2003:2020; Rainforest Alliance Certification & Auditing Rules |
| **Enforcement** | December 30, 2026 (large/medium operators); June 30, 2027 (SMEs) |

---

## 1. Regulatory Basis and Legal Framework

### 1.1 Primary Regulation: EU Deforestation Regulation (EUDR)

Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023, concerning the making available on the Union market and the export from the Union of certain commodities and products associated with deforestation and forest degradation, establishes a mandatory due diligence framework for operators and traders dealing in seven regulated commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) and their derived products.

The EUDR does not mandate a specific third-party audit regime as a standalone compliance pathway. However, multiple articles create obligations, incentives, and procedural hooks that make third-party audit management a critical operational capability for compliant operators. The regulation treats third-party verification as a risk mitigation tool within the broader due diligence framework, meaning audit results feed into but do not replace the operator's own due diligence obligations.

**Enforcement Timeline (as amended by Regulation (EU) 2024/3234):**
- December 30, 2026: Large and medium operators/traders -- full obligations apply
- June 30, 2027: Micro and small operators/traders -- full obligations apply
- Competent authorities enforcement duties active from June 30, 2026

### 1.2 Article 10 -- Risk Assessment and Third-Party Verification

Article 10 of the EUDR establishes the risk assessment framework that operators must apply as part of their due diligence obligations under Article 8. Third-party audit management intersects with Article 10 in the following ways:

**Article 10(1):** Operators shall assess and identify the risk that relevant products intended to be placed on the market or exported are non-compliant. The assessment must be adequate and proportionate, taking into account the criteria in paragraph 2.

**Article 10(2)(n):** Risk assessment shall take into account "the consultation of and cooperation with relevant stakeholders, including indigenous peoples, local communities, civil society organisations, and third-party monitoring organisations." This explicitly recognizes that third-party verification findings constitute relevant input to the operator's risk assessment.

**Article 10(2)(l):** The operator must consider "any information suggesting a risk that the relevant product does not comply with Article 3, including information from relevant certification or third-party verified schemes."

**Article 10(2)(m):** Risk assessment must consider "any information or intelligence from third parties indicating risks of non-compliance," which includes audit findings, non-conformance reports, and corrective action statuses from third-party auditors.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-001: The agent SHALL ingest and structure findings from third-party audit reports as inputs to the Article 10 risk assessment process.
- REQ-024-002: The agent SHALL track audit finding severity (critical, major, minor, observation) and map each finding to Article 10(2) risk criteria.
- REQ-024-003: The agent SHALL maintain a timestamped audit evidence register that can demonstrate to competent authorities that third-party verification was considered in the risk assessment per Article 10(2)(n).

### 1.3 Article 11 -- Simplified Due Diligence and Certification Schemes

Article 11 provides for simplified due diligence obligations where the Commission has classified a country or part thereof as "low risk" under Article 29. The article is relevant to third-party audit management because:

**Article 11(1):** Simplified due diligence for products sourced from low-risk countries reduces but does not eliminate the information collection and risk assessment burden. The operator must still collect information per Article 9 and may rely on "relevant and verifiable information" including certification scheme outputs.

**Recital 52:** The EUDR explicitly states that "voluntary certification or other third-party verified schemes could be used in the risk assessment procedure" but "should not substitute the operator's responsibility" for due diligence. Certification and third-party verification are tools, not exemptions.

**Implications for Certification Scheme Audits:**
- Certification scheme audits (FSC, PEFC, RSPO, Rainforest Alliance, ISCC) can reduce residual risk but cannot replace operator-level due diligence.
- The agent must track the scope, validity, and currency of certification audits to determine their applicability to EUDR compliance.
- Where a certification scheme audit covers deforestation-free and legality requirements, the operator may use the audit findings to satisfy specific Article 10(2) criteria, provided the audit scope aligns with EUDR requirements.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-004: The agent SHALL map certification scheme audit scope to EUDR Article 10(2) risk assessment criteria and identify coverage gaps.
- REQ-024-005: The agent SHALL track certification audit validity periods and trigger re-assessment alerts when audits expire or are suspended.
- REQ-024-006: The agent SHALL distinguish between certification audit findings and EUDR-specific audit findings, maintaining separate tracking for each.

### 1.4 Articles 14-16 -- Monitoring Organizations and Substantiated Concerns

Articles 14-16 establish the framework for monitoring organizations, which are entities that can submit substantiated concerns to competent authorities regarding non-compliance.

**Article 14(1):** Natural or legal persons may submit substantiated concerns to competent authorities where they consider that one or more operators or traders are not complying with the regulation.

**Article 15:** Monitoring organizations recognized under Article 16 have a formal role in monitoring operator compliance and submitting substantiated concerns.

**Article 16:** Recognition criteria for monitoring organizations include independence from operators, adequate expertise, and commitment to transparency.

**Implications for Third-Party Audit Management:**
- Third-party audit findings that reveal non-compliance may trigger substantiated concern submissions by monitoring organizations.
- The agent must track the interface between audit findings and potential substantiated concern notifications.
- Audit reports and CAR records may be requested by monitoring organizations or submitted as evidence.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-007: The agent SHALL flag audit findings that may constitute grounds for substantiated concern submissions under Article 14.
- REQ-024-008: The agent SHALL generate audit evidence packages in a format suitable for submission to competent authorities or monitoring organizations.

### 1.5 Article 29 -- Competent Authorities and Inspection Powers

Article 29 mandates that EU Member States designate one or more competent authorities responsible for enforcing the EUDR. These competent authorities have broad audit and inspection powers that directly affect how operators manage their third-party audit programs.

**Article 29(1):** Member States shall designate one or more competent authorities responsible for the fulfillment of the obligations laid down in this Regulation.

**Article 29(3):** Competent authorities shall carry out checks to verify that operators and traders comply with the obligations set out in this Regulation. Such checks shall be risk-based and carried out at a minimum frequency.

**Article 29(5):** Competent authorities shall have the power to:
- Access relevant documentation and data, including information on due diligence systems
- Carry out inspections, including on-site inspections and audits of operators' premises
- Take samples of relevant commodities and products
- Request and obtain information from operators and traders on their supply chains

**Article 29(6):** Competent authorities shall carry out checks on at least 9% of operators and traders placing high-risk products from high-risk countries, and at least 9% of the quantity of such products.

**Article 30 -- Check Plans and Annual Reporting:** Competent authorities must establish annual check plans based on risk assessment and report results to the Commission.

**Article 31 -- Penalties:** Member States shall lay down effective, proportionate, and dissuasive penalties, including:
- Fines proportionate to environmental damage and product value, with maximum not less than 4% of EU-wide annual turnover
- Confiscation of products and revenue
- Temporary exclusion from public procurement
- Temporary prohibition from placing products on the market

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-009: The agent SHALL maintain audit-ready documentation packages that can be provided to competent authorities within the timeframes specified by national implementing legislation.
- REQ-024-010: The agent SHALL track competent authority inspection schedules and outcomes as part of the operator's audit history.
- REQ-024-011: The agent SHALL generate compliance evidence reports aligned with the Article 29(5) documentation access requirements.
- REQ-024-012: The agent SHALL calculate and display the operator's exposure to Article 31 penalties based on unresolved audit non-conformances.

### 1.6 Article 4(7) -- Record Retention Requirements

Article 4(7) requires operators and traders to retain documentation relating to due diligence, including risk assessments and risk mitigation measures, for a minimum of five years from the date the due diligence statement was placed on the market.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-013: The agent SHALL retain all audit records, CAR records, evidence packages, and audit reports for a minimum of 5 years from the associated DDS submission date.
- REQ-024-014: The agent SHALL implement immutable audit trail logging for all audit management activities, compliant with SEC-005 (Centralized Audit Logging).

---

## 2. International Audit Standards Framework

### 2.1 ISO 19011:2018 -- Guidelines for Auditing Management Systems

ISO 19011:2018 provides comprehensive guidance for auditing management systems and establishes the foundational principles, procedures, and competence requirements that the Third-Party Audit Manager Agent must implement. The standard applies to all organizations that need to plan and conduct internal or external audits of management systems or manage audit programs.

**Seven Principles of Auditing (Clause 4):**

| # | Principle | Description | Agent Implementation |
|---|-----------|-------------|----------------------|
| 1 | Integrity | Foundation of professionalism; auditors shall perform work honestly and responsibly | Auditor qualification tracking; conflict of interest detection |
| 2 | Fair Presentation | Obligation to report truthfully and accurately | Structured finding templates; evidence-linked conclusions |
| 3 | Due Professional Care | Application of diligence and judgment in auditing | Competence matrix enforcement; scope-appropriate team assignment |
| 4 | Confidentiality | Security of information | Encryption at rest (AES-256-GCM); role-based access; SEC-003 compliance |
| 5 | Independence | Basis for impartiality and objectivity | Auditor-auditee relationship tracking; rotation enforcement |
| 6 | Evidence-based Approach | Rational method for reaching reliable conclusions | Evidence chain management; sampling methodology tracking |
| 7 | Risk-based Approach | Audit approach considering risks and opportunities | Risk-based audit scheduling; resource allocation by risk score |

**Audit Program Management (Clause 5):**

ISO 19011:2018 Clause 5 establishes the requirements for managing an audit program, which the agent must implement:

- **5.2 Establishing audit program objectives:** The agent must support definition of audit objectives aligned with EUDR compliance requirements, organizational policy, and certification scheme obligations.
- **5.3 Determining and evaluating audit program risks and opportunities:** Risk-based scheduling where higher-risk suppliers, commodities, and geographies receive more frequent and intensive audits.
- **5.4 Establishing the audit program:** Annual audit calendars with resource allocation, scope definition, audit team composition, and logistical planning.
- **5.5 Implementing the audit program:** Execution tracking with milestone management (preparation, fieldwork, reporting, follow-up).
- **5.6 Monitoring the audit program:** KPI dashboards tracking audit completion rates, on-time delivery, finding closure rates, and audit effectiveness metrics.
- **5.7 Reviewing and improving the audit program:** Periodic program reviews with trend analysis and continuous improvement.

**Conducting an Audit (Clause 6):**

| Phase | ISO 19011 Clause | Agent Functionality |
|-------|-------------------|---------------------|
| Initiating the audit | 6.2 | Audit request management; scope definition; team assignment |
| Preparing audit activities | 6.3 | Audit plan generation; document review; checklist creation |
| Conducting audit activities | 6.4 | Evidence collection; interview logging; sampling tracking |
| Preparing and distributing audit report | 6.5 | Structured report generation; distribution workflow |
| Completing the audit | 6.6 | Finding classification; CAR issuance; closure confirmation |
| Conducting audit follow-up | 6.7 | CAR tracking; evidence verification; closure audit scheduling |

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-015: The agent SHALL implement the ISO 19011:2018 seven-phase audit lifecycle (initiation, preparation, execution, reporting, completion, follow-up, program review).
- REQ-024-016: The agent SHALL enforce the seven auditing principles through configurable workflow rules and validation checks.
- REQ-024-017: The agent SHALL generate audit plans per ISO 19011:2018 Clause 6.3, including objectives, scope, criteria, schedule, team roles, and methods.

### 2.2 ISO/IEC 17065:2012 -- Requirements for Certification Bodies (Products)

ISO/IEC 17065:2012 specifies requirements for bodies certifying products, processes, and services. This standard is directly relevant because certification bodies auditing EUDR-regulated supply chains (e.g., FSC, RSPO, PEFC) must be accredited to ISO/IEC 17065.

**Key Requirements:**

- **Clause 4 -- General requirements:** Certification bodies must operate in a non-discriminatory manner, must be accessible, and must be responsible for their certification decisions.
- **Clause 5 -- Structural requirements:** Organizational structure must ensure impartiality, with mechanisms to safeguard objectivity. Includes requirements for committees and governance structures.
- **Clause 6 -- Resource requirements:** Personnel involved in certification must be competent, with documented criteria for competence, training, and monitoring. Includes requirements for internal and external resources.
- **Clause 7 -- Process requirements:** Defines the certification process including application, evaluation, review, decision, and surveillance. Evaluation activities include document review, testing/inspection, and audit.
- **Clause 7.6 -- Surveillance:** Certification bodies must plan and conduct surveillance activities at sufficient frequency to maintain confidence in certified products. Surveillance can include periodic audits, product sampling, and document reviews.
- **Clause 7.7 -- Changes affecting certification:** Certification bodies must have processes to address changes in certification requirements, product specifications, or the certified organization's operations.
- **Clause 7.9 -- Suspension, withdrawal, or reduction of scope:** Procedures for adverse certification decisions including timelines, appeal rights, and public notification.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-018: The agent SHALL track certification body accreditation status per ISO/IEC 17065:2012, including accreditation body, scope, validity period, and any restrictions or suspensions.
- REQ-024-019: The agent SHALL verify that certification body auditors assigned to EUDR-relevant audits meet the competence requirements of ISO/IEC 17065 Clause 6.
- REQ-024-020: The agent SHALL track surveillance audit schedules per ISO/IEC 17065 Clause 7.6 and alert when surveillance is overdue.

### 2.3 ISO/IEC 17021-1:2015 -- Requirements for Certification Bodies (Management Systems)

ISO/IEC 17021-1:2015 contains principles and requirements for the competence, consistency, and impartiality of bodies providing audit and certification of management systems. This standard is relevant to EUDR compliance because environmental management system (EMS) certifications under ISO 14001 and forest management system certifications often form part of the evidence base for EUDR due diligence.

**Key Requirements:**

- **Clause 7 -- Resource requirements:** Certification body personnel shall be competent on the basis of education, training, technical knowledge, skills, and experience. Clause 7.1.2 specifies that the certification body shall have processes to ensure personnel demonstrate competence for the specific functions they perform.
- **Clause 7.2 -- Personnel involved in the certification activities:** Requirements for audit team leaders, auditors, and technical experts. Specifies competence criteria including management system knowledge, sector-specific knowledge, and audit skills.
- **Clause 8 -- Information requirements:** Publicly accessible information about certification activities, directory of certified clients, and procedures for handling complaints and appeals.
- **Clause 9 -- Process requirements:** Complete certification cycle including application, audit planning, initial audit (Stage 1 and Stage 2), certification decision, surveillance audits, recertification, and special audits. Stage 1 assesses documentation and readiness; Stage 2 evaluates implementation and effectiveness.
- **Clause 9.4 -- Conducting audits:** Audit time must be sufficient to cover the entire scope. Audit plans must be agreed with the client and include objectives, criteria, scope, dates, roles, and methods. On-site audit must include opening meeting, information gathering, generating findings, preparing conclusions, and closing meeting.
- **Clause 9.6 -- Surveillance activities:** At least one surveillance audit per year. First surveillance within 12 months of initial certification decision. Surveillance program covers the complete management system over the certification cycle.
- **Clause 9.7 -- Recertification:** Full system re-audit before certificate expiry. Recertification planning begins sufficiently in advance to allow timely renewal.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-021: The agent SHALL implement the ISO/IEC 17021-1 two-stage audit model (Stage 1 documentation review, Stage 2 implementation audit) for management system audits.
- REQ-024-022: The agent SHALL track surveillance audit frequencies per ISO/IEC 17021-1 Clause 9.6 (minimum annual, first within 12 months).
- REQ-024-023: The agent SHALL manage recertification timelines per ISO/IEC 17021-1 Clause 9.7, triggering planning activities sufficiently in advance of certificate expiry.

---

## 3. Certification Scheme Audit Standards

### 3.1 FSC (Forest Stewardship Council) Audit Requirements

The Forest Stewardship Council operates a global forest certification system comprising forest management (FM) certification and chain-of-custody (CoC) certification. FSC audit requirements are defined in FSC-STD-20-011 (Chain of Custody Evaluations) and FSC-STD-20-007 (Forest Management Evaluations).

**Audit Types and Frequencies:**

| Audit Type | Frequency | Duration | Scope |
|------------|-----------|----------|-------|
| Initial Assessment (Main Assessment) | Once (pre-certification) | 2-5 days (FM); 1-3 days (CoC) | Full standard evaluation |
| Annual Surveillance | Annual (minimum) | 1-3 days | Subset of standard; risk-based focus |
| Reassessment (Recertification) | Every 5 years | Full assessment duration | Complete re-evaluation |
| Transfer Audit | As needed | 1-2 days | Transition between certification bodies |
| Unannounced Audit | Ad hoc (risk-based) | 0.5-2 days | Targeted verification |

**Non-Conformance Classification (FSC):**

- **Major Non-Conformity (Major CAR):** A fundamental failure to achieve the objective of a requirement of the FSC standard. Examples: no chain-of-custody control system in place; mixing certified and uncertified material without authorization; no documented procedures for key processes. Major CARs issued during initial assessment must be closed before certificate issuance. Major CARs during surveillance must be closed within 3 months or certificate is suspended.
- **Minor Non-Conformity (Minor CAR):** A temporary, unusual, or non-systematic lapse that does not fundamentally undermine the integrity of the chain-of-custody system. Examples: incomplete records for a single transaction; minor documentation gaps; training records not up to date. Must be closed within 12 months (by the next surveillance audit).
- **Observation (OBS):** A finding that is not a non-conformity but indicates a potential risk of future non-conformity. Tracked for follow-up at next audit but does not require formal corrective action plan.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-024: The agent SHALL implement FSC non-conformance classification (Major CAR, Minor CAR, Observation) with scheme-specific closure timelines (Major: 3 months; Minor: 12 months).
- REQ-024-025: The agent SHALL track FSC certificate lifecycle including initial assessment, annual surveillance, and 5-year reassessment cycles.
- REQ-024-026: The agent SHALL monitor FSC certificate suspension and withdrawal events via FSC database integration and update compliance status accordingly.

### 3.2 PEFC (Programme for the Endorsement of Forest Certification)

PEFC certification follows PEFC ST 2003:2020 (Chain of Custody of Forest and Tree Based Products -- Requirements) and is audited by accredited certification bodies under ISO/IEC 17065.

**Audit Structure:**

| Phase | Requirement | Timeline |
|-------|-------------|----------|
| Initial Certification Audit | Full evaluation against PEFC ST 2003 | Pre-certification |
| Surveillance Audit | Annual verification; may be combined with other management system audits | Annual (within 12 months) |
| Recertification Audit | Full re-evaluation | Every 3-5 years (scheme dependent) |
| Short-Notice Audit | Triggered by complaints, market claims, or risk indicators | As needed |

**Non-Conformance Handling:**
- PEFC uses a two-tier system: Major and Minor non-conformities.
- Major non-conformities must be corrected before certification or within a defined timeline (typically 3 months) during surveillance, or the certificate is suspended.
- Minor non-conformities must be corrected by the next scheduled audit.
- PEFC requires root cause analysis for all non-conformities as part of the corrective action process.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-027: The agent SHALL implement PEFC non-conformance handling with two-tier classification (Major, Minor) and scheme-specific closure timelines.
- REQ-024-028: The agent SHALL enforce root cause analysis requirements for all PEFC non-conformities within the CAR management workflow.

### 3.3 RSPO (Roundtable on Sustainable Palm Oil)

RSPO operates the Supply Chain Certification Standard (SCCS) and Principles and Criteria (P&C) standard, audited by RSPO-approved certification bodies accredited to ISO/IEC 17065 or ISO/IEC 17021.

**Audit Types:**

| Audit Type | Application | Frequency |
|------------|-------------|-----------|
| Initial Certification Assessment | New applicants | Once (pre-certification) |
| Annual Surveillance Assessment | Certified members | Annual |
| Re-assessment (Recertification) | Certificate renewal | Every 5 years |
| Compliance Investigation Audit | Complaint-triggered | As needed |
| Partial Assessment | Scope changes | As needed |

**RSPO Non-Conformance Classification:**
- **Major Non-Conformity:** Failure to comply with a principal-level requirement or a systematic failure across multiple criteria. Triggers mandatory corrective action within 30-90 days depending on severity. If uncorrected, leads to certificate suspension.
- **Minor Non-Conformity:** Non-systematic, isolated failure that does not undermine the integrity of the management system. Must be corrected within 12 months (before next annual assessment).
- **Critical Non-Conformity (RSPO-specific):** Introduced for severe violations such as new deforestation after cut-off date, land clearing without FPIC, or use of fire for land clearing. Triggers immediate suspension pending investigation.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-029: The agent SHALL implement RSPO three-tier non-conformance classification (Critical, Major, Minor) with scheme-specific timelines (Critical: immediate; Major: 30-90 days; Minor: 12 months).
- REQ-024-030: The agent SHALL track RSPO complaint investigations and compliance investigation audits as a distinct audit type.
- REQ-024-031: The agent SHALL integrate with RSPO PalmTrace to verify certificate status and membership standing.

### 3.4 Rainforest Alliance Certification and Auditing Rules

Rainforest Alliance operates under the 2020 Certification Program (updated 2024) with dedicated Certification and Auditing Rules and Supply Chain Certification and Auditing Rules.

**Audit Framework:**

| Audit Type | Purpose | Frequency |
|------------|---------|-----------|
| Initial Certification Audit | Full standard assessment for new applicants | Year 0 |
| Annual Surveillance Audit | Compliance verification for continuous improvement | Annual |
| Renewal Audit (Recertification) | Full re-assessment for certificate renewal | Every 3 years |
| Additional Audit | Triggered by complaints, non-conformance escalation, or data irregularities | As needed |
| Remote/Desktop Audit | Document review conducted remotely (limited applicability per rules) | As authorized |

**Non-Conformance Handling (Rainforest Alliance):**
- Non-conformances are classified according to severity and relationship to core requirements.
- Not addressing major non-conformances by the specified timeline leads to immediate suspension of the certificate.
- Rainforest Alliance requires evidence-based closure verification for all corrective actions, with auditor sign-off required.
- Certification bodies must report suspension events to the Rainforest Alliance within defined timelines.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-032: The agent SHALL implement Rainforest Alliance non-conformance handling with scheme-specific escalation rules including immediate suspension triggers.
- REQ-024-033: The agent SHALL track the 3-year Rainforest Alliance certification cycle with annual surveillance milestones.

### 3.5 Certification Scheme Interoperability Requirements

Multiple certification schemes may apply to a single supply chain (e.g., FSC for timber, RSPO for palm oil derivatives in the same product). The agent must handle cross-scheme audit management.

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-034: The agent SHALL support concurrent tracking of multiple certification scheme audits for a single supplier, product, or supply chain.
- REQ-024-035: The agent SHALL map certification scheme coverage to EUDR Article 3 requirements (deforestation-free, legal production, due diligence statement) and identify gaps where certification does not cover EUDR obligations.
- REQ-024-036: The agent SHALL normalize non-conformance classifications across schemes into a unified severity taxonomy (Critical, Major, Minor, Observation) while preserving scheme-specific terminology.

---

## 4. Audit Types and Definitions

### 4.1 Audit Type Taxonomy

The Third-Party Audit Manager Agent must support the following audit types, derived from ISO 19011:2018, ISO/IEC 17021-1:2015, certification scheme standards, and EUDR competent authority requirements.

| Audit Type | Code | Description | Typical Duration | Trigger |
|------------|------|-------------|------------------|---------|
| Desktop/Document Review | DESK | Remote review of documents, records, and data | 0.5-2 days | Pre-audit, simplified assessment, COVID-era allowance |
| On-Site Initial Assessment | INIT | Full physical audit at supplier/operator premises | 2-10 days | New certification, first-time supplier |
| On-Site Surveillance | SURV | Periodic on-site verification of continued compliance | 1-5 days | Annual schedule, certification maintenance |
| Remote/Virtual Audit | RMTE | Video-enabled audit using digital tools | 1-3 days | Travel restrictions, low-risk suppliers, supplementary |
| Recertification/Renewal | RCRT | Full re-assessment before certificate expiry | 2-8 days | Certificate expiry approaching (3-5 year cycle) |
| Follow-Up/Verification | FLUP | Targeted audit to verify CAR closure | 0.5-2 days | Open CARs requiring on-site verification |
| Unannounced/Spot Check | UNAN | Surprise audit without prior notification | 0.5-3 days | Risk-based, competent authority request, complaint |
| Special/Investigation Audit | SPEC | Audit triggered by specific events or concerns | 1-5 days | Substantiated concern, media report, whistleblower |
| Scope Extension Audit | SCEX | Assessment of new activities/sites added to certification | 1-3 days | Scope change request by certified entity |
| Transfer Audit | TRAN | Transition assessment when changing certification bodies | 1-3 days | CB change request |
| Competent Authority Inspection | CAIN | Official inspection by EUDR competent authority | Variable | Article 29 check plan, risk-based selection |
| Combined/Integrated Audit | COMB | Single audit covering multiple standards/schemes | 3-10 days | Efficiency, multi-scheme certified entities |

### 4.2 Audit Classification by Party

Per ISO 19011:2018 terminology:

| Classification | Description | EUDR Relevance |
|----------------|-------------|----------------|
| First-Party Audit | Internal audit conducted by or on behalf of the organization | Operator's internal due diligence review |
| Second-Party Audit | External audit conducted by interested parties (customers, supply chain partners) | Operator auditing suppliers per Article 10 due diligence |
| Third-Party Audit | Independent audit by accredited certification bodies or regulatory authorities | Certification scheme audits; competent authority inspections |

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-037: The agent SHALL support all 12 audit types defined in the taxonomy with type-specific workflow configurations, required fields, and completion criteria.
- REQ-024-038: The agent SHALL classify audits by party (first, second, third) and apply appropriate independence and impartiality requirements per ISO 19011:2018 Clause 4.

---

## 5. Audit Scope and Criteria

### 5.1 Scope Definition Requirements

Per ISO 19011:2018 Clause 6.3.2, the audit scope defines the extent and boundaries of the audit, including locations, organizational units, activities, and processes to be audited, as well as the time period covered by the audit.

**EUDR-Specific Scope Elements:**

| Scope Element | Description | Data Source |
|---------------|-------------|-------------|
| Commodity Coverage | Which EUDR commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) | EUDR-001 supply chain mapping |
| Geographic Coverage | Countries/regions of production assessed | EUDR-002 geolocation, EUDR-016 country risk |
| Supply Chain Tiers | Which tiers of the supply chain are covered | EUDR-008 multi-tier supplier tracker |
| Temporal Coverage | Reporting/assessment period (typically 12 months) | Audit plan definition |
| Standard/Criteria | Which standards or regulatory requirements are assessed | Certification scheme, EUDR articles |
| Due Diligence Elements | Which Article 8-10 elements (information, risk assessment, mitigation) | EUDR compliance framework |
| Legislation Categories | Which Article 2(40) categories (8 categories) | EUDR-023 legal compliance verifier |
| Chain of Custody Model | Identity Preserved, Segregated, Mass Balance, Controlled Sources | EUDR-009, EUDR-010, EUDR-011 |

### 5.2 Audit Criteria

Audit criteria are the set of requirements used as a reference against which audit evidence is compared. For EUDR third-party audits, criteria include:

**Primary Criteria:**
- Regulation (EU) 2023/1115 Articles 3, 4, 8, 9, 10 (full due diligence requirements)
- Article 2(40) relevant legislation of country of production (8 categories)
- Commission Implementing Regulation on the EU Information System
- Country benchmarking risk classification (Article 29)

**Secondary Criteria (Certification Scheme Standards):**
- FSC Principles and Criteria (FSC-STD-01-001) and CoC standard (FSC-STD-40-004)
- RSPO Principles and Criteria and SCCS
- PEFC Sustainable Forest Management standard (PEFC ST 1003) and CoC (PEFC ST 2002)
- Rainforest Alliance 2020 Sustainable Agriculture Standard
- ISCC EU/PLUS certification requirements

**Tertiary Criteria (Organizational):**
- Operator's own due diligence system policies and procedures
- Supplier codes of conduct
- Group-level sustainability policies

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-039: The agent SHALL maintain a structured criteria library with version-controlled audit criteria sets for each supported standard and regulation.
- REQ-024-040: The agent SHALL allow audit scope definition with multi-dimensional coverage (commodity, geography, tier, temporal, standard, CoC model).
- REQ-024-041: The agent SHALL validate that audit scope is sufficient to cover the operator's EUDR obligations for the relevant DDS submissions.

---

## 6. Audit Planning and Risk-Based Scheduling

### 6.1 Risk-Based Audit Programming

ISO 19011:2018 Clause 5.3 requires that audit program risks and opportunities be determined and evaluated. The EUDR context adds specific risk factors that must drive audit planning.

**Risk Factors for Audit Scheduling:**

| Risk Factor | Source | Weight | Description |
|-------------|--------|--------|-------------|
| Country Risk Classification | EUDR-016 Country Risk Evaluator | High | Commission Article 29 risk classification (high/standard/low) |
| Supplier Risk Score | EUDR-017 Supplier Risk Scorer | High | Composite supplier risk score from multiple indicators |
| Commodity Risk Level | EUDR-018 Commodity Risk Analyzer | Medium | Commodity-specific deforestation and legality risk |
| Previous Audit Findings | Agent EUDR-024 (self) | High | History of non-conformances, particularly unresolved or recurring |
| Deforestation Alert Status | EUDR-020 Deforestation Alert System | Critical | Active deforestation alerts in supply region |
| Corruption Index | EUDR-019 Corruption Index Monitor | Medium | Governance and corruption risk in production country |
| Certification Status | Agent EUDR-024 (self) | Medium | Validity, scope, and health of existing certifications |
| Volume/Value at Risk | EUDR-001 Supply Chain Mapping | Medium | Financial exposure if non-compliance is found |
| Competent Authority Focus | Article 29 check plans | High | Whether competent authority has flagged the commodity/country |
| Substantiated Concerns | Article 14-16 monitoring | Critical | Active or resolved substantiated concerns |
| Time Since Last Audit | Agent EUDR-024 (self) | Medium | Elapsed time since the most recent third-party verification |

### 6.2 Audit Frequency Matrix

Based on risk assessment outputs, the agent must determine appropriate audit frequencies:

| Risk Level | Composite Score | Audit Frequency | Audit Type |
|------------|-----------------|------------------|------------|
| Critical | >= 90 | Quarterly + Unannounced | Full on-site + spot checks |
| High | 70-89 | Semi-annual | Full on-site |
| Medium | 40-69 | Annual | On-site or combined remote/on-site |
| Low | 20-39 | Annual (surveillance) | Desktop review + selective on-site |
| Minimal | < 20 | Biennial (certification cycle) | Desktop review |

### 6.3 Annual Audit Plan Requirements

Per ISO 19011:2018 Clause 5.4, the audit program shall include:

- **Objectives:** Aligned with EUDR compliance, certification maintenance, and organizational risk appetite
- **Scope:** Which suppliers, sites, commodities, and standards are covered
- **Resources:** Auditor availability, budget allocation, travel planning
- **Calendar:** Specific audit dates or windows for each planned audit
- **Methods:** On-site, remote, combined, desktop -- per risk level
- **Audit Team Composition:** Lead auditor, team auditors, technical experts, observers
- **Contingency:** Buffer capacity for unplanned audits (investigation, spot check)

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-042: The agent SHALL generate annual audit programs based on risk-weighted scheduling using inputs from EUDR-016 through EUDR-020 risk agents.
- REQ-024-043: The agent SHALL calculate audit frequency per supplier/site based on composite risk score with configurable frequency matrix.
- REQ-024-044: The agent SHALL allocate auditor resources based on competence matching (commodity, geography, language, standard) and availability.
- REQ-024-045: The agent SHALL maintain a 15-20% contingency capacity in the annual audit program for unplanned audits.
- REQ-024-046: The agent SHALL re-assess and adjust the audit program when significant risk changes occur (e.g., deforestation alert, certificate suspension, substantiated concern).

---

## 7. Auditor Competence Requirements

### 7.1 ISO 19011:2018 Competence Framework (Clause 7)

ISO 19011:2018 Clause 7 establishes a comprehensive framework for evaluating and maintaining auditor competence. The standard defines competence as "the ability to apply knowledge and skills to achieve intended results."

**Competence Elements:**

| Element | ISO 19011 Reference | Description |
|---------|---------------------|-------------|
| Personal Behavior | 7.2.2 | Ethical, open-minded, diplomatic, observant, perceptive, versatile, tenacious, decisive, self-reliant, culturally sensitive |
| Knowledge and Skills -- General | 7.2.3.1 | Audit principles, standards, applicable legal/regulatory requirements, organization context, management system fundamentals |
| Knowledge and Skills -- Discipline-Specific | 7.2.3.2 | Specific to the management system discipline (environmental, quality, forest management, supply chain) |
| Knowledge and Skills -- Sector-Specific | 7.2.3.3 | Specific to the sector (agriculture, forestry, palm oil, cocoa, rubber, cattle, soya) |
| Knowledge and Skills -- Audit Team Leader | 7.2.3.4 | Planning, team management, conflict resolution, report preparation, audit conclusion |
| Education | 7.2.4 | Formal education sufficient to support knowledge and skill requirements |
| Work Experience | 7.2.4 | Professional experience in relevant technical, managerial, or professional positions |
| Auditor Training | 7.2.4 | Training to develop knowledge and skills specific to auditing |
| Audit Experience | 7.2.4 | Participation in audits under the direction of a competent audit team leader |
| Continual Professional Development | 7.6 | Ongoing maintenance and improvement of competence through practice, training, and education |

### 7.2 EUDR-Specific Auditor Competence Requirements

Beyond ISO 19011 general competence, EUDR third-party auditors require specialized knowledge:

| Competence Area | Requirements | Verification Method |
|-----------------|-------------|---------------------|
| EUDR Regulation Knowledge | Full understanding of Regulation (EU) 2023/1115, delegated and implementing acts | Documented training, examination |
| Commodity-Specific Knowledge | Understanding of production, processing, and supply chain for each regulated commodity | Sector experience, training |
| Geospatial/GIS Competence | Ability to verify geolocation data, interpret satellite imagery, assess land-use change | Technical certification, experience |
| Country-Specific Legal Knowledge | Understanding of relevant legislation in countries of production (Article 2(40) categories) | Legal training, country experience |
| Chain of Custody Models | Understanding of Identity Preserved, Segregated, Mass Balance, Controlled Sources | Certification scheme training |
| Document Verification | Ability to authenticate permits, licenses, certificates, and legal documents | Forensic training, experience |
| Indigenous Rights / FPIC | Understanding of UNDRIP, ILO 169, and FPIC processes | Human rights training |
| Remote Sensing / Satellite Data | Ability to interpret deforestation monitoring data from satellite sources | Technical training, GIS certification |
| Language Competence | Ability to conduct audits in the language of the auditee and review documents in local languages | Language proficiency documentation |
| Due Diligence Systems | Understanding of EUDR due diligence system design, implementation, and effectiveness | EUDR-specific training |

### 7.3 Auditor Qualification Levels

| Level | Title | Minimum Requirements | Authorized Activities |
|-------|-------|----------------------|----------------------|
| L1 | Trainee Auditor | Relevant degree + 2 years sector experience + auditor training + 2 audits as observer | Participation in audit teams under supervision |
| L2 | Auditor | L1 + 5 complete audits + competence evaluation passed | Conduct audits as team member; lead limited scope audits |
| L3 | Lead Auditor | L2 + 3 complete audits as team leader under supervision + lead auditor training | Lead full scope audits; manage audit teams |
| L4 | Senior/Principal Auditor | L3 + 10 audits as lead + 5 years continuous audit practice + peer evaluation | Lead complex multi-site audits; mentor L1-L3; review audit quality |
| TE | Technical Expert | Relevant degree + 5 years sector-specific experience | Provide technical input to audit teams (not a qualified auditor) |

### 7.4 Auditor Independence and Conflict of Interest

Per ISO 19011:2018 Clause 4 (Independence principle) and ISO/IEC 17021-1 Clause 5 (Structural requirements):

- Auditors shall not audit their own work or that of their direct employer (for internal audits)
- For third-party audits, auditors shall have no financial, commercial, or personal interest in the auditee
- Auditor rotation: Lead auditors should be rotated at regular intervals (typically every 3-5 years for the same auditee)
- Cooling-off periods: Auditors who have provided consultancy to an entity must observe a minimum 2-year cooling-off period before auditing that entity

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-047: The agent SHALL maintain an auditor registry with competence profiles covering all elements from ISO 19011:2018 Clause 7.2.
- REQ-024-048: The agent SHALL enforce EUDR-specific competence requirements for auditor assignment (commodity knowledge, country knowledge, language, GIS, CoC model).
- REQ-024-049: The agent SHALL track auditor qualification levels (L1-L4, TE) and prevent assignment of auditors to activities beyond their qualification level.
- REQ-024-050: The agent SHALL enforce auditor independence rules including conflict of interest checks, rotation schedules (maximum 3-5 years for same auditee), and cooling-off periods (minimum 2 years post-consultancy).
- REQ-024-051: The agent SHALL track continuing professional development (CPD) hours and alert when auditors fall below minimum CPD thresholds.

---

## 8. Audit Evidence Collection and Sampling

### 8.1 Evidence Types

Per ISO 19011:2018 Clause 6.4, audit evidence is the basis for audit findings and conclusions. Evidence must be verifiable, relevant, and sufficient.

| Evidence Type | Description | EUDR Application |
|---------------|-------------|------------------|
| Documentary Evidence | Records, documents, permits, certificates, reports | Permits, licenses, DDS, certificates of origin, customs declarations |
| Physical/Observational Evidence | Direct observation during site visits | Plot boundary verification, land use observation, processing facility inspection |
| Testimonial Evidence | Statements from interviews with personnel | Interviews with suppliers, workers, community members, management |
| Analytical Evidence | Results of data analysis, calculations, reconciliations | Mass balance calculations, geospatial analysis, satellite data review |
| Electronic/Digital Evidence | System records, database extracts, GIS data, satellite imagery | EU Information System records, certification databases, GPS coordinates |
| Sampling-Based Evidence | Evidence obtained through statistical or judgmental sampling | Transaction sampling, batch sampling, site sampling |

### 8.2 Sampling Methodology

ISO 19011:2018 Clause A.6 provides guidance on sampling methods for audit evidence. The agent must support:

**Statistical Sampling:**
- Simple random sampling from the audit population
- Stratified sampling (by risk level, commodity, geography)
- Systematic sampling (every nth transaction/batch)
- Confidence level and margin of error calculations

**Judgmental (Non-Statistical) Sampling:**
- Risk-based selection: Focus on high-risk transactions, suppliers, or commodities
- Coverage-based selection: Ensure representation of all commodities, origins, and supply chain tiers
- Anomaly-based selection: Target transactions flagged by automated risk indicators
- Complaint-based selection: Target areas subject to substantiated concerns

**Sampling Size Guidelines (ISO 19011:2018 Annex A):**

For multi-site audits, sampling size is typically based on:
- Number of sites: y = sqrt(x) for low-complexity; y = 0.8 * sqrt(x) for medium; y = 0.6 * sqrt(x) for high-complexity (where x = number of sites)
- Risk-adjusted: Increase sample by 50-100% for high-risk populations
- Minimum sample: At least 1 site from each geographic region or commodity type

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-052: The agent SHALL support both statistical and judgmental sampling methodologies with documented selection rationale.
- REQ-024-053: The agent SHALL calculate sample sizes based on ISO 19011:2018 Annex A formulas with risk-based adjustments.
- REQ-024-054: The agent SHALL maintain evidence chain integrity with SHA-256 hashes, timestamps, and auditor attestation for all collected evidence.
- REQ-024-055: The agent SHALL track evidence sufficiency per audit finding, ensuring each finding is supported by at least one verifiable piece of evidence.

---

## 9. Non-Conformance Classification System

### 9.1 Unified Non-Conformance Taxonomy

The agent must implement a unified non-conformance classification system that harmonizes across ISO standards and certification schemes while preserving scheme-specific granularity.

**Severity Levels:**

| Level | Code | Definition | Impact | Examples |
|-------|------|-----------|--------|----------|
| Critical | NCR-C | A non-conformity that represents an immediate, severe, or systemic threat to the integrity of the compliance system, environmental protection, or human rights. The absolute risk posed makes immediate corrective action the highest priority. | Immediate suspension of certification or trading activity; potential regulatory notification | Active deforestation in certified/declared supply area; use of forced labor; complete absence of due diligence system; falsified DDS submission; major fraud in documentation |
| Major | NCR-M | A fundamental failure to achieve the objective of a requirement. Indicates either the absence of a required process or system, or a total or systematic breakdown in implementation. Significantly impacts the ability to demonstrate EUDR compliance. | Certificate suspension within 30-90 days if uncorrected; enhanced competent authority scrutiny | No risk assessment procedure; no geolocation data for declared plots; chain of custody system not implemented; no corrective action process; systematic record-keeping failures |
| Minor | NCR-m | A temporary, unusual, or non-systematic lapse that does not fundamentally undermine the compliance system. An isolated failure in implementation that can be corrected without systemic changes. | Must be corrected within 12 months; tracked at subsequent surveillance audit | Incomplete records for individual transaction; minor documentation gaps; single instance of late record entry; training records partially incomplete |
| Observation | OBS | A finding that is not a non-conformity but indicates a potential risk of future non-conformity or an area for improvement. No corrective action required, but tracked for monitoring. | Tracked at next audit; may escalate to Minor if not addressed | Emerging risk in supply region; process area where controls could be strengthened; documentation that meets minimum but could be improved |
| Opportunity for Improvement | OFI | A positive finding identifying where the organization could enhance its systems beyond minimum compliance requirements. | Advisory only; no formal tracking required | Best practice adoption; efficiency improvement; additional monitoring capability |

### 9.2 Escalation and De-Escalation Rules

| Trigger | Action |
|---------|--------|
| Minor NCR not closed within 12 months | Escalate to Major NCR |
| Major NCR not closed within 90 days | Escalate to Critical NCR; trigger certificate suspension |
| Critical NCR issued | Immediate suspension pending investigation; competent authority notification within 72 hours |
| Recurring Minor NCR (same root cause, 3+ occurrences) | Escalate to Major NCR (systematic failure) |
| Successful CAR closure with verified evidence | De-escalate or close NCR |
| Major NCR closed within timeline with root cause addressed | No de-escalation (close as Major) |

### 9.3 Non-Conformance to EUDR Article Mapping

| NCR Category | Relevant EUDR Article | Risk Impact |
|--------------|----------------------|-------------|
| Due diligence system absence/failure | Art. 4, 8 | Market access prohibition |
| Information collection gaps | Art. 9 | DDS rejection |
| Risk assessment deficiency | Art. 10 | Enhanced checks by competent authority |
| Risk mitigation inadequacy | Art. 10(3) | Product may not be placed on market |
| Geolocation data missing/inaccurate | Art. 9(1)(d) | DDS non-compliant |
| Deforestation cut-off date violation | Art. 2(6) | Market access prohibition |
| Legal production non-compliance | Art. 2(40), 3(b) | Market access prohibition |
| Record retention failure | Art. 4(7) | Enforcement penalty |
| DDS submission failure | Art. 4(2) | Market access prohibition |
| Supply chain traceability gap | Art. 9(1)(h) | Enhanced due diligence required |

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-056: The agent SHALL implement the five-level non-conformance taxonomy (Critical, Major, Minor, Observation, OFI) with configurable definitions per certification scheme.
- REQ-024-057: The agent SHALL enforce automated escalation rules when NCR closure deadlines are exceeded.
- REQ-024-058: The agent SHALL map each NCR to relevant EUDR articles and calculate the associated regulatory risk impact.
- REQ-024-059: The agent SHALL track non-conformance trends (frequency, severity, root cause category, supplier, commodity, geography) for audit program improvement.

---

## 10. Corrective Action Request (CAR) Management

### 10.1 CAR Lifecycle

The CAR lifecycle follows ISO 19011:2018 Clause 6.7 (Conducting audit follow-up) and ISO/IEC 17021-1 Clause 9.4 practices:

| Phase | Description | Responsible Party | Timeline |
|-------|-------------|-------------------|----------|
| 1. Issuance | Auditor issues CAR with finding details, classification, evidence, and required action | Audit Team Leader | During/immediately after audit |
| 2. Acknowledgement | Auditee acknowledges receipt and accepts the finding | Auditee Management | Within 5 business days |
| 3. Root Cause Analysis | Auditee identifies root cause(s) using structured methodology (5-Why, Fishbone, Fault Tree) | Auditee Quality Team | Within 15 business days (Minor); 5 business days (Major/Critical) |
| 4. Corrective Action Plan | Auditee proposes corrective actions addressing root cause and preventing recurrence | Auditee Management | Within 20 business days (Minor); 10 business days (Major); 3 business days (Critical) |
| 5. Plan Review | Auditor reviews and approves or rejects the corrective action plan | Audit Team Leader | Within 5 business days of submission |
| 6. Implementation | Auditee implements approved corrective actions | Auditee Operations | Per approved timeline |
| 7. Verification | Auditor verifies implementation through evidence review or follow-up audit | Audit Team / CB | Per CAR timeline (documentary or on-site) |
| 8. Closure | CAR closed with verified evidence; status updated | Audit Team Leader | Upon successful verification |
| 9. Effectiveness Review | Post-closure review to confirm corrective action prevented recurrence | Audit Program Manager | At next scheduled audit |

### 10.2 CAR Data Model

Each CAR record must contain the following data elements:

| Field | Type | Description |
|-------|------|-------------|
| CAR ID | String (UUID) | Unique identifier for the CAR |
| Audit ID | Foreign Key | Link to the parent audit record |
| Finding Reference | String | Cross-reference to the specific audit finding |
| NCR Classification | Enum | Critical / Major / Minor |
| Certification Scheme | String | FSC / PEFC / RSPO / RA / ISCC / EUDR / Custom |
| Standard Requirement | String | Specific clause or criterion that was not met |
| EUDR Article Reference | String | Mapped EUDR article (if applicable) |
| Finding Description | Text | Detailed description of the non-conformity with objective evidence |
| Evidence References | Array[UUID] | Links to supporting evidence (documents, photos, interview notes) |
| Root Cause | Text | Root cause analysis output |
| Root Cause Method | Enum | 5-Why / Fishbone / Fault Tree / Other |
| Corrective Action Plan | Text | Description of proposed corrective and preventive actions |
| Implementation Deadline | Date | Required completion date for corrective actions |
| Responsible Person | String | Named individual accountable for corrective action |
| Verification Method | Enum | Documentary Review / On-Site Visit / Remote Verification |
| Verification Evidence | Array[UUID] | Links to evidence demonstrating corrective action completion |
| Status | Enum | Open / Acknowledged / RCA Complete / Plan Submitted / Plan Approved / Implementing / Verification Pending / Closed / Escalated |
| Closure Date | Date | Date the CAR was verified and closed |
| Auditor Sign-Off | String | Lead auditor confirming closure |
| Effectiveness Status | Enum | Pending Review / Effective / Not Effective / Requires Further Action |

### 10.3 CAR Timeline Requirements by Scheme

| Scheme | Critical CAR | Major CAR | Minor CAR |
|--------|-------------|-----------|-----------|
| FSC | N/A (Major is highest) | Close before certification (initial) or 3 months (surveillance) | 12 months (before next surveillance) |
| PEFC | N/A (Major is highest) | Close before certification (initial) or 3 months (surveillance) | Before next scheduled audit |
| RSPO | Immediate suspension pending investigation | 30-90 days | 12 months |
| Rainforest Alliance | Immediate suspension | Per scheme rules (typically 30-60 days) | Before next annual audit |
| ISCC | Immediate suspension | 3 months | 12 months |
| EUDR (Competent Authority) | Per national implementing legislation | Per national implementing legislation | Per national implementing legislation |

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-060: The agent SHALL implement the 9-phase CAR lifecycle with status tracking, timeline enforcement, and automated escalation.
- REQ-024-061: The agent SHALL store all CAR data elements as specified in the data model with full provenance (timestamps, actor IDs, SHA-256 hashes).
- REQ-024-062: The agent SHALL enforce scheme-specific CAR timelines and trigger escalation when deadlines are missed.
- REQ-024-063: The agent SHALL require root cause analysis for all Major and Critical CARs, with structured methodology selection (5-Why, Fishbone, Fault Tree).
- REQ-024-064: The agent SHALL track CAR closure rates and average closure times as KPIs for the audit program dashboard.
- REQ-024-065: The agent SHALL implement effectiveness review scheduling to verify that closed CARs have prevented recurrence.

---

## 11. Audit Report Structure and Content

### 11.1 Report Requirements

Per ISO 19011:2018 Clause 6.5, the audit report shall provide a complete, accurate, concise, and clear record of the audit. The report is the formal deliverable of the audit process and the primary evidence artifact for EUDR compliance demonstration.

### 11.2 Mandatory Report Sections

| Section | Content | ISO Reference |
|---------|---------|---------------|
| 1. Report Header | Audit ID, report date, audit type, confidentiality marking | 6.5.1 |
| 2. Audit Objectives | Statement of audit objectives as defined in the audit plan | 6.5.1(a) |
| 3. Audit Scope | Organizational and functional units, processes, sites, and time period audited | 6.5.1(b) |
| 4. Audit Criteria | Standards, regulations, and requirements against which the audit was conducted | 6.5.1(c) |
| 5. Audit Team | Names and roles of audit team members, technical experts, and observers; qualification levels | 6.5.1 |
| 6. Audit Dates and Locations | Dates, locations, and durations of audit activities conducted | 6.5.1(d) |
| 7. Audit Methodology | Methods used including interviews, document review, sampling, observation, and analysis | 6.5.1 |
| 8. Sampling Rationale | Description of sampling approach, sample size, selection criteria, and population coverage | A.6 |
| 9. Audit Findings | Non-conformances (Critical, Major, Minor), observations, and positive findings with objective evidence | 6.5.1(e) |
| 10. Non-Conformance Details | For each NCR: classification, requirement reference, evidence, impact assessment, and CAR reference | 6.5.1(e) |
| 11. Audit Conclusions | Overall assessment of conformity; degree of fulfillment of audit criteria; systemic issues; risk assessment | 6.5.1(f) |
| 12. Corrective Action Requirements | Summary of issued CARs with deadlines; follow-up audit requirements | 6.7 |
| 13. Previous Audit Follow-Up | Status of CARs from previous audits; verification results | 6.7 |
| 14. EUDR Compliance Assessment | Specific assessment against EUDR Articles 3, 8, 9, 10 requirements; DDS readiness evaluation | EUDR-specific |
| 15. Certification Recommendation | Recommendation regarding certification (grant, maintain, suspend, withdraw, expand, reduce scope) | ISO/IEC 17021-1 Clause 9.5 |
| 16. Distribution List | Recipients of the audit report; confidentiality restrictions | 6.5.2 |
| 17. Appendices | Evidence inventory, interview list, documents reviewed, site visit itinerary, photo evidence index | 6.5.1 |

### 11.3 EUDR-Specific Report Content

Beyond the standard ISO 19011 report structure, EUDR third-party audit reports must include:

| EUDR Report Element | Content | Regulatory Basis |
|---------------------|---------|------------------|
| Article 3 Compliance Statement | Assessment of whether audited products meet deforestation-free, legal production, and DDS requirements | Art. 3(a), (b), (c) |
| Article 9 Information Adequacy | Verification that all required information elements have been collected and documented | Art. 9(1)(a)-(h) |
| Geolocation Verification | Assessment of geolocation data accuracy and completeness for all declared plots | Art. 9(1)(d) |
| Risk Assessment Quality | Evaluation of the operator's risk assessment methodology and conclusions | Art. 10(1)-(3) |
| Mitigation Measures Effectiveness | Assessment of risk mitigation measures implemented where risk was identified as non-negligible | Art. 10(3) |
| Supply Chain Traceability | Verification of supply chain transparency from production to market placement | Art. 9(1)(g)-(h) |
| Chain of Custody Integrity | Assessment of CoC system implementation (IP, SG, MB) and volume reconciliation | Art. 9, Certification scheme |
| Legal Compliance Assessment | Evaluation of compliance with relevant legislation per Article 2(40) categories | Art. 2(40), 3(b) |
| Cut-Off Date Compliance | Verification that production occurred on land not subject to deforestation after December 31, 2020 | Art. 2(6) |
| Competent Authority Readiness | Assessment of the operator's preparedness for competent authority inspection under Article 29 | Art. 29 |

### 11.4 Report Formats and Distribution

| Format | Use Case | Distribution |
|--------|----------|-------------|
| PDF (structured, signed) | Formal audit report for regulatory retention | Auditee, certification body, competent authority |
| JSON (structured data) | Machine-readable for integration with GL-EUDR-APP and DDS system | Internal systems, EU Information System interface |
| HTML (interactive) | Web-viewable report with expandable sections and evidence links | Internal stakeholders, audit portal |
| CSV/XLSX (tabular) | Findings and CAR data for analysis and tracking | Audit program management, BI systems |
| XML (regulatory) | Structured submission format for competent authority reporting systems | Competent authority submission |

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-066: The agent SHALL generate audit reports containing all 17 mandatory sections per ISO 19011:2018 Clause 6.5.
- REQ-024-067: The agent SHALL include all EUDR-specific report elements (Article 3 compliance, geolocation verification, cut-off date, legal compliance assessment).
- REQ-024-068: The agent SHALL produce reports in 5 formats (PDF, JSON, HTML, CSV/XLSX, XML) with SHA-256 provenance hashes on all generated reports.
- REQ-024-069: The agent SHALL enforce report review and approval workflow before distribution, requiring lead auditor sign-off and quality review.
- REQ-024-070: The agent SHALL maintain a complete report distribution registry tracking who received which report and when.

---

## 12. Integration with Certification Schemes

### 12.1 Certification Scheme Coverage Mapping to EUDR

The EUDR does not grant exemptions based on certification alone (Recital 52). However, certification scheme audit outputs provide substantial evidence for EUDR due diligence. The agent must map the coverage provided by each scheme to EUDR requirements.

**Coverage Matrix:**

| EUDR Requirement | FSC | PEFC | RSPO | RA | ISCC |
|------------------|-----|------|------|----|------|
| Deforestation-free (Art. 2(6)) | Partial (HCV/HCS assessment) | Partial (sustainable forest management) | Yes (no deforestation pledge, HCS) | Yes (no deforestation) | Partial (land use change) |
| Legal production (Art. 2(40)) | Partial (legality verification) | Partial (legal compliance) | Partial (legal requirements in P&C) | Partial (legal requirements) | Partial (legal compliance) |
| Geolocation (Art. 9(1)(d)) | Partial (FM unit boundaries) | Partial (forest unit boundaries) | Yes (mill GPS + supply base) | Yes (farm-level GPS) | Partial (varies by scope) |
| Supply chain traceability (Art. 9(1)(h)) | Yes (CoC standard) | Yes (CoC standard) | Yes (SCCS) | Yes (traceability system) | Yes (CoC) |
| Cut-off date (31 Dec 2020) | No (different cut-off) | No (different cut-off) | Partial (varies by P&C version) | Yes (2020 cut-off aligned) | Partial (varies) |
| Eight legislation categories (Art. 2(40)) | Partial (5-6 categories) | Partial (4-5 categories) | Partial (5-6 categories) | Partial (5-6 categories) | Partial (3-4 categories) |
| Indigenous rights / FPIC | Yes (Principle 3, 4) | Partial | Yes (Principle 2) | Yes (social requirements) | Partial |

### 12.2 Certification Scheme Audit Data Integration

The agent must ingest and normalize audit data from each certification scheme:

| Scheme | Data Source | Integration Method | Update Frequency |
|--------|------------|-------------------|------------------|
| FSC | FSC Certificate Database (info.fsc.org) | API + batch import | Daily sync |
| PEFC | PEFC Certificate Search | API + batch import | Daily sync |
| RSPO | RSPO PalmTrace | API integration | Real-time events |
| Rainforest Alliance | RA Certificate Portal | API + batch import | Daily sync |
| ISCC | ISCC Certificate Database | API + batch import | Daily sync |
| Custom/Proprietary | Operator-specific systems | File import (XML, JSON, CSV) | On-demand |

### 12.3 Certification Status Tracking

| Status | Meaning | EUDR Impact |
|--------|---------|-------------|
| Active | Certificate valid and in good standing | Audit findings can be used as EUDR risk mitigation evidence |
| Suspended | Certificate temporarily suspended due to non-conformance | Audit findings invalid for EUDR; enhanced due diligence required |
| Withdrawn | Certificate permanently revoked | No reliance on certification for EUDR; full due diligence required |
| Expired | Certificate validity period has elapsed without renewal | No reliance on certification for EUDR; renewal status must be tracked |
| Under Investigation | Formal complaint or compliance investigation in progress | Precautionary principle applies; enhanced due diligence recommended |
| Scope Reduced | Certificate scope reduced (sites, products, or activities removed) | Verify that EUDR-relevant scope remains covered |

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-071: The agent SHALL maintain the certification scheme coverage matrix and identify gaps where certification does not cover EUDR requirements.
- REQ-024-072: The agent SHALL integrate with 5 certification scheme databases (FSC, PEFC, RSPO, RA, ISCC) for real-time certificate status verification.
- REQ-024-073: The agent SHALL trigger alerts when a certification status changes to Suspended, Withdrawn, Expired, or Under Investigation.
- REQ-024-074: The agent SHALL recalculate EUDR compliance risk when certification status changes, propagating updates to EUDR-017 (Supplier Risk Scorer).

---

## 13. Competent Authority Coordination

### 13.1 Article 29 Inspection Readiness

Competent authorities may conduct inspections without prior notice. The agent must maintain continuous inspection readiness.

**Readiness Requirements:**

| Readiness Area | Requirement | Agent Capability |
|----------------|------------|------------------|
| Documentation Access | All due diligence documentation accessible within timeframes set by national law | Instant retrieval of audit records, CARs, evidence, and reports |
| Audit History | Complete audit history for 5+ years per Article 4(7) | Immutable audit trail with retention enforcement |
| CAR Status | Current status of all open and recently closed CARs | Real-time CAR dashboard with evidence links |
| Certification Status | Current certification status for all certified suppliers/sites | Live integration with certification databases |
| Risk Assessment Records | Documentation of how third-party audit findings were integrated into risk assessment | Structured risk-audit linkage per REQ-024-001 |
| DDS Linkage | Link between audit records and specific DDS submissions | Cross-reference between audits and DDS records |
| Corrective Action Evidence | Evidence packages for all closed CARs | Verified evidence with SHA-256 hashes and auditor sign-off |

### 13.2 Competent Authority Data Exchange

| Data Type | Direction | Format | Trigger |
|-----------|-----------|--------|---------|
| Audit Summary Reports | Operator to CA | PDF, XML | Upon CA request (Article 29(5)) |
| Inspection Findings | CA to Operator | Official correspondence | After CA inspection |
| Compliance Status Declarations | Operator to CA | Structured XML/JSON | DDS submission, annual reporting |
| Corrective Action Plans | Operator to CA | PDF, XML | When CA requires remediation |
| Substantiated Concern Responses | Operator to CA | Structured response | Within CA-specified timeline |
| Annual Check Plan Results | CA to Commission | Annual report | Article 30 reporting |

### 13.3 Multi-Member State Coordination

Under the EUDR, competent authorities coordinate across Member States through the EU Information System. The agent must support scenarios where:

- A single operator places products on multiple EU markets
- Competent authorities from different Member States request audit documentation
- Cross-border supply chains require coordinated audit efforts
- Information sharing between Member States triggers additional audit requirements

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-075: The agent SHALL maintain continuous inspection readiness with sub-5-second retrieval time for any audit record requested by a competent authority.
- REQ-024-076: The agent SHALL generate competent authority data exchange packages in the formats specified by national implementing legislation (PDF, XML, JSON).
- REQ-024-077: The agent SHALL track competent authority interactions (requests, inspections, findings, remediation requirements) as a distinct audit type (CAIN).
- REQ-024-078: The agent SHALL support multi-jurisdictional reporting where audit evidence must be presented to competent authorities in multiple Member States.

---

## 14. Integration with Existing EUDR Agent Ecosystem

### 14.1 Agent Dependencies and Data Flows

| Agent | Integration Direction | Data Exchanged |
|-------|----------------------|----------------|
| EUDR-001 (Supply Chain Mapping Master) | Inbound | Supply chain graph nodes, supplier identifiers, commodity flows |
| EUDR-002 (Geolocation Verification) | Inbound | Plot geolocation data, coordinate verification status |
| EUDR-008 (Multi-Tier Supplier Tracker) | Inbound | Multi-tier supplier relationships, tier-specific audit requirements |
| EUDR-009 (Chain of Custody Agent) | Bidirectional | CoC audit findings, custody model verification, batch traceability |
| EUDR-010 (Segregation Verifier) | Inbound | Segregation audit results, contamination findings |
| EUDR-011 (Mass Balance Calculator) | Inbound | Volume reconciliation data, mass balance audit findings |
| EUDR-012 (Document Authentication) | Bidirectional | Document verification results for audit evidence; audit report authentication |
| EUDR-016 (Country Risk Evaluator) | Inbound | Country risk scores for audit frequency determination |
| EUDR-017 (Supplier Risk Scorer) | Bidirectional | Supplier risk scores (inbound for scheduling); audit findings (outbound for risk adjustment) |
| EUDR-018 (Commodity Risk Analyzer) | Inbound | Commodity risk levels for audit scope prioritization |
| EUDR-019 (Corruption Index Monitor) | Inbound | Corruption risk data for audit planning and red flag detection |
| EUDR-020 (Deforestation Alert System) | Inbound | Active deforestation alerts triggering investigation audits |
| EUDR-021 (Indigenous Rights Checker) | Inbound | FPIC compliance data for audit criteria |
| EUDR-022 (Protected Area Validator) | Inbound | Protected area compliance data for audit criteria |
| EUDR-023 (Legal Compliance Verifier) | Bidirectional | Legal compliance audit findings; legislation coverage data |
| GL-EUDR-APP | Bidirectional | Audit schedules, status dashboards, CAR management UI, DDS audit evidence |

### 14.2 Event-Driven Integration

The agent publishes and subscribes to the following events via the GreenLang event bus:

**Published Events:**
- `eudr.audit.scheduled` -- New audit scheduled in the audit program
- `eudr.audit.started` -- Audit execution commenced
- `eudr.audit.completed` -- Audit execution completed with findings
- `eudr.audit.report_published` -- Audit report finalized and distributed
- `eudr.car.issued` -- New CAR issued from audit findings
- `eudr.car.escalated` -- CAR escalated due to missed deadline
- `eudr.car.closed` -- CAR verified and closed
- `eudr.certification.status_changed` -- Certificate status change detected
- `eudr.audit.inspection_ready` -- Inspection readiness package updated

**Subscribed Events:**
- `eudr.risk.country_updated` -- Country risk score changed (from EUDR-016)
- `eudr.risk.supplier_updated` -- Supplier risk score changed (from EUDR-017)
- `eudr.alert.deforestation_detected` -- Deforestation alert issued (from EUDR-020)
- `eudr.supply_chain.updated` -- Supply chain mapping changed (from EUDR-001)
- `eudr.document.authenticated` -- Document authentication result (from EUDR-012)
- `eudr.legal.compliance_updated` -- Legal compliance status changed (from EUDR-023)

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-079: The agent SHALL integrate bidirectionally with EUDR agents 001, 002, 008-012, 016-023 per the defined data flow matrix.
- REQ-024-080: The agent SHALL publish and subscribe to the defined event types via the GreenLang event bus with guaranteed message delivery (99.9% SLA).
- REQ-024-081: The agent SHALL trigger risk-based audit schedule adjustments when critical events are received (deforestation alerts, certification suspensions, country risk changes).

---

## 15. Data Retention and Immutability Requirements

### 15.1 Retention Schedule

| Record Type | Minimum Retention | Regulatory Basis | Storage |
|-------------|-------------------|------------------|---------|
| Audit Reports | 5 years from associated DDS | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| CAR Records | 5 years from closure date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Audit Evidence | 5 years from audit date | EUDR Art. 4(7); ISO 19011 | Encrypted at rest (AES-256-GCM) |
| Auditor Competence Records | Duration of auditor engagement + 3 years | ISO/IEC 17021-1 Clause 7 | Encrypted at rest (AES-256-GCM) |
| Certification Status History | 5 years from certificate expiry | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Competent Authority Correspondence | 5 years from correspondence date | EUDR Art. 29 | Encrypted at rest (AES-256-GCM) |
| Audit Program Plans | 5 years from program period end | ISO 19011 Clause 5 | Encrypted at rest (AES-256-GCM) |

### 15.2 Immutability and Provenance

All audit records must be immutable once finalized. The agent must implement:

- SHA-256 content hashes on all audit reports, CAR records, and evidence files
- Append-only audit trail for all record modifications (no deletion, no overwrite)
- Tamper detection through hash chain verification
- Digital signature on finalized audit reports (auditor and reviewer signatures)
- Integration with SEC-005 (Centralized Audit Logging) for all agent activities

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-082: The agent SHALL enforce minimum 5-year retention for all audit-related records per EUDR Article 4(7).
- REQ-024-083: The agent SHALL implement immutable record storage with SHA-256 hash chains and tamper detection.
- REQ-024-084: The agent SHALL integrate with SEC-005 (Centralized Audit Logging) for all audit management activities with 70+ event types.

---

## 16. Penalties and Risk Exposure

### 16.1 EUDR Penalty Framework (Article 31)

| Penalty Type | Description | Risk Factor |
|--------------|-------------|-------------|
| Financial Fines | Proportionate to environmental damage, product value, and losses; minimum not less than 4% of annual EU-wide turnover | Unresolved Critical/Major CARs; systematic non-conformance |
| Confiscation | Confiscation of non-compliant products and proceeds | Products linked to audit failures |
| Market Exclusion | Temporary prohibition from placing products on the EU market | Sustained non-compliance; repeated non-conformance |
| Public Procurement Exclusion | Temporary exclusion from participation in public procurement procedures | Serious compliance failures |
| Public Naming | Public disclosure of non-compliant operators | Formal enforcement action |
| Criminal Liability | Under national implementing legislation | Fraudulent DDS, deliberate non-compliance |

### 16.2 Audit-Related Risk Mitigation

Robust third-party audit management directly mitigates penalty risk by:

- Demonstrating proactive due diligence (Art. 10 risk assessment includes third-party verification)
- Providing documented evidence of compliance efforts (audit reports, CAR closure evidence)
- Identifying and remediating non-conformances before competent authority detection
- Maintaining inspection readiness to respond to Article 29 checks efficiently
- Creating a defensible compliance record that may be considered as a mitigating factor in penalty determination

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-085: The agent SHALL calculate penalty exposure scores based on unresolved audit non-conformances, using the Article 31 penalty framework.
- REQ-024-086: The agent SHALL generate compliance defense packages that demonstrate the operator's proactive audit and remediation efforts.

---

## 17. Performance and Quality Requirements

### 17.1 System Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Audit report generation | < 10 seconds | p99 latency |
| CAR creation/update | < 2 seconds | p99 latency |
| Certification status check | < 3 seconds | p99 latency per certificate |
| Risk-based schedule generation | < 30 seconds for annual program | p99 latency |
| Evidence retrieval | < 5 seconds for any audit record | p99 latency |
| Dashboard refresh | < 3 seconds | p99 latency |
| Audit program KPI calculation | < 15 seconds for full program analysis | p99 latency |
| Concurrent audit management | 500+ active audits | Capacity |
| API availability | 99.9% uptime | Monthly SLA |
| Event processing | < 500ms for event consumption and response | p99 latency |

### 17.2 Zero-Hallucination Guarantee

Consistent with the GreenLang platform standard:

- All audit findings, non-conformance classifications, and compliance assessments are deterministic
- No LLM in the critical path for compliance determinations
- SHA-256 provenance hashes on all generated outputs
- Bit-perfect reproducibility: Same inputs produce identical outputs
- All regulatory references are traceable to specific articles, clauses, and provisions

**Regulatory Requirement for Agent EUDR-024:**
- REQ-024-087: The agent SHALL meet all performance targets specified in Section 17.1.
- REQ-024-088: The agent SHALL implement zero-hallucination deterministic processing for all compliance-critical operations.

---

## 18. Requirements Traceability Matrix

| Requirement ID | Description | Regulatory Source | Priority |
|----------------|-------------|-------------------|----------|
| REQ-024-001 | Ingest third-party audit findings for Article 10 risk assessment | EUDR Art. 10(2)(n) | P0 |
| REQ-024-002 | Track finding severity and map to Article 10(2) criteria | EUDR Art. 10(2) | P0 |
| REQ-024-003 | Maintain timestamped audit evidence register | EUDR Art. 10(2)(n) | P0 |
| REQ-024-004 | Map certification audit scope to EUDR criteria | EUDR Art. 11, Recital 52 | P0 |
| REQ-024-005 | Track certification audit validity and trigger alerts | EUDR Art. 11 | P0 |
| REQ-024-006 | Distinguish certification vs. EUDR-specific findings | EUDR Recital 52 | P1 |
| REQ-024-007 | Flag findings for substantiated concern submissions | EUDR Art. 14-16 | P1 |
| REQ-024-008 | Generate evidence packages for CA/monitoring orgs | EUDR Art. 14-16, 29 | P0 |
| REQ-024-009 | Maintain audit-ready documentation for CA inspection | EUDR Art. 29(5) | P0 |
| REQ-024-010 | Track CA inspection schedules and outcomes | EUDR Art. 29 | P1 |
| REQ-024-011 | Generate compliance evidence reports per Art. 29(5) | EUDR Art. 29(5) | P0 |
| REQ-024-012 | Calculate penalty exposure from unresolved NCRs | EUDR Art. 31 | P1 |
| REQ-024-013 | 5-year minimum record retention | EUDR Art. 4(7) | P0 |
| REQ-024-014 | Immutable audit trail logging | EUDR Art. 4(7), SEC-005 | P0 |
| REQ-024-015 | ISO 19011 seven-phase audit lifecycle | ISO 19011:2018 Cl. 5-6 | P0 |
| REQ-024-016 | Enforce seven auditing principles | ISO 19011:2018 Cl. 4 | P0 |
| REQ-024-017 | Generate audit plans per ISO 19011 Cl. 6.3 | ISO 19011:2018 Cl. 6.3 | P0 |
| REQ-024-018 | Track CB accreditation per ISO/IEC 17065 | ISO/IEC 17065:2012 | P0 |
| REQ-024-019 | Verify CB auditor competence per ISO/IEC 17065 | ISO/IEC 17065:2012 Cl. 6 | P1 |
| REQ-024-020 | Track surveillance schedules per ISO/IEC 17065 | ISO/IEC 17065:2012 Cl. 7.6 | P1 |
| REQ-024-021 | Implement two-stage audit model | ISO/IEC 17021-1 Cl. 9 | P1 |
| REQ-024-022 | Track surveillance frequencies (annual minimum) | ISO/IEC 17021-1 Cl. 9.6 | P0 |
| REQ-024-023 | Manage recertification timelines | ISO/IEC 17021-1 Cl. 9.7 | P0 |
| REQ-024-024 | FSC non-conformance classification | FSC-STD-20-011 | P0 |
| REQ-024-025 | FSC certificate lifecycle tracking | FSC-STD-20-011 | P0 |
| REQ-024-026 | FSC certificate suspension/withdrawal monitoring | FSC database | P0 |
| REQ-024-027 | PEFC non-conformance handling | PEFC ST 2003:2020 | P0 |
| REQ-024-028 | PEFC root cause analysis enforcement | PEFC ST 2003:2020 | P1 |
| REQ-024-029 | RSPO three-tier NCR classification | RSPO SCCS | P0 |
| REQ-024-030 | RSPO complaint investigation tracking | RSPO SCCS | P1 |
| REQ-024-031 | RSPO PalmTrace integration | RSPO PalmTrace | P0 |
| REQ-024-032 | Rainforest Alliance NCR handling | RA Auditing Rules | P0 |
| REQ-024-033 | Rainforest Alliance 3-year cycle tracking | RA Certification Rules | P0 |
| REQ-024-034 | Multi-scheme concurrent audit tracking | Cross-scheme | P0 |
| REQ-024-035 | Certification-to-EUDR coverage mapping | EUDR Recital 52 | P0 |
| REQ-024-036 | Cross-scheme NCR taxonomy normalization | ISO 19011, all schemes | P1 |
| REQ-024-037 | Support 12 audit types | ISO 19011, ISO/IEC 17021, EUDR | P0 |
| REQ-024-038 | Classify audits by party (1st/2nd/3rd) | ISO 19011:2018 | P1 |
| REQ-024-039 | Structured criteria library with version control | ISO 19011:2018 Cl. 6.3 | P0 |
| REQ-024-040 | Multi-dimensional scope definition | ISO 19011:2018 Cl. 6.3.2 | P0 |
| REQ-024-041 | Validate scope covers EUDR obligations | EUDR Art. 8-10 | P0 |
| REQ-024-042 | Risk-weighted annual audit program generation | ISO 19011:2018 Cl. 5.3-5.4 | P0 |
| REQ-024-043 | Risk-based audit frequency calculation | ISO 19011:2018 Cl. 5.3 | P0 |
| REQ-024-044 | Competence-matched auditor allocation | ISO 19011:2018 Cl. 7 | P0 |
| REQ-024-045 | 15-20% contingency capacity in audit program | ISO 19011:2018 Cl. 5.4 | P1 |
| REQ-024-046 | Dynamic audit program adjustment on risk change | ISO 19011:2018 Cl. 5.5 | P0 |
| REQ-024-047 | Auditor registry with competence profiles | ISO 19011:2018 Cl. 7.2 | P0 |
| REQ-024-048 | EUDR-specific auditor competence enforcement | EUDR Art. 10, ISO 19011 | P0 |
| REQ-024-049 | Auditor qualification level tracking (L1-L4, TE) | ISO 19011:2018 Cl. 7.2.4 | P0 |
| REQ-024-050 | Auditor independence and conflict of interest | ISO 19011:2018 Cl. 4, ISO/IEC 17021-1 Cl. 5 | P0 |
| REQ-024-051 | CPD tracking and compliance alerts | ISO 19011:2018 Cl. 7.6 | P1 |
| REQ-024-052 | Statistical and judgmental sampling | ISO 19011:2018 Annex A | P0 |
| REQ-024-053 | ISO 19011 sample size calculation | ISO 19011:2018 Annex A | P1 |
| REQ-024-054 | Evidence chain integrity (SHA-256, timestamps) | ISO 19011, EUDR Art. 4(7) | P0 |
| REQ-024-055 | Evidence sufficiency per finding | ISO 19011:2018 Cl. 6.4 | P0 |
| REQ-024-056 | Five-level NCR taxonomy | ISO 19011, all schemes | P0 |
| REQ-024-057 | Automated NCR escalation on deadline miss | ISO/IEC 17021-1, schemes | P0 |
| REQ-024-058 | NCR to EUDR article mapping | EUDR Art. 3-10 | P0 |
| REQ-024-059 | NCR trend tracking and analysis | ISO 19011:2018 Cl. 5.7 | P1 |
| REQ-024-060 | 9-phase CAR lifecycle | ISO 19011:2018 Cl. 6.7 | P0 |
| REQ-024-061 | CAR data model with full provenance | ISO 19011, EUDR Art. 4(7) | P0 |
| REQ-024-062 | Scheme-specific CAR timelines | FSC/PEFC/RSPO/RA/ISCC | P0 |
| REQ-024-063 | Root cause analysis for Major/Critical CARs | ISO 19011, all schemes | P0 |
| REQ-024-064 | CAR closure rate KPI tracking | ISO 19011:2018 Cl. 5.6 | P1 |
| REQ-024-065 | Effectiveness review scheduling | ISO 19011:2018 Cl. 6.7 | P1 |
| REQ-024-066 | 17-section audit report generation | ISO 19011:2018 Cl. 6.5 | P0 |
| REQ-024-067 | EUDR-specific report elements | EUDR Art. 3, 8, 9, 10 | P0 |
| REQ-024-068 | 5-format report output with SHA-256 hashes | ISO 19011, EUDR | P0 |
| REQ-024-069 | Report review/approval workflow | ISO 19011:2018 Cl. 6.5 | P0 |
| REQ-024-070 | Report distribution registry | ISO 19011:2018 Cl. 6.5.2 | P1 |
| REQ-024-071 | Certification scheme EUDR coverage matrix | EUDR Recital 52 | P0 |
| REQ-024-072 | 5-scheme database integration | FSC/PEFC/RSPO/RA/ISCC | P0 |
| REQ-024-073 | Certification status change alerts | ISO/IEC 17065 Cl. 7.9 | P0 |
| REQ-024-074 | Risk recalculation on certification change | EUDR Art. 10 | P0 |
| REQ-024-075 | Sub-5-second inspection readiness | EUDR Art. 29(5) | P0 |
| REQ-024-076 | CA data exchange formats | EUDR Art. 29-30 | P0 |
| REQ-024-077 | CA interaction tracking (CAIN audit type) | EUDR Art. 29 | P0 |
| REQ-024-078 | Multi-jurisdictional reporting | EUDR Art. 29 | P1 |
| REQ-024-079 | Bidirectional EUDR agent integration | GreenLang Architecture | P0 |
| REQ-024-080 | Event bus integration (99.9% SLA) | GreenLang Architecture | P0 |
| REQ-024-081 | Event-triggered audit schedule adjustment | ISO 19011 Cl. 5.5, EUDR | P0 |
| REQ-024-082 | 5-year record retention | EUDR Art. 4(7) | P0 |
| REQ-024-083 | Immutable storage with SHA-256 hash chains | EUDR Art. 4(7), SEC-005 | P0 |
| REQ-024-084 | SEC-005 audit logging integration | SEC-005, EUDR | P0 |
| REQ-024-085 | Penalty exposure calculation | EUDR Art. 31 | P1 |
| REQ-024-086 | Compliance defense package generation | EUDR Art. 31 | P1 |
| REQ-024-087 | Performance targets per Section 17.1 | System requirement | P0 |
| REQ-024-088 | Zero-hallucination deterministic processing | GreenLang standard | P0 |

---

## 19. Glossary

| Term | Definition |
|------|-----------|
| **Audit** | A systematic, independent, and documented process for obtaining audit evidence and evaluating it objectively to determine the extent to which the audit criteria are fulfilled (ISO 19011:2018 Clause 3.1) |
| **Audit Criteria** | Set of requirements used as a reference against which objective evidence is compared (ISO 19011:2018 Clause 3.7) |
| **Audit Evidence** | Records, statements of fact, or other information which are relevant to the audit criteria and verifiable (ISO 19011:2018 Clause 3.9) |
| **Audit Finding** | Results of the evaluation of the collected audit evidence against audit criteria (ISO 19011:2018 Clause 3.10) |
| **Audit Program** | Arrangements for a set of one or more audits planned for a specific time frame and directed towards a specific purpose (ISO 19011:2018 Clause 3.13) |
| **CAR** | Corrective Action Request -- a formal document requiring the auditee to investigate the root cause of a non-conformity and implement corrective actions to prevent recurrence |
| **CB** | Certification Body -- an organization accredited to conduct certification audits and issue certificates of conformity |
| **CoC** | Chain of Custody -- a documented sequence of custody, control, and accountability for materials through all stages of the supply chain |
| **Competent Authority** | National authority designated by an EU Member State to enforce the EUDR per Article 29 |
| **Critical Non-Conformity** | A non-conformity representing an immediate, severe, or systemic threat to compliance system integrity, environmental protection, or human rights |
| **DDS** | Due Diligence Statement -- the formal statement submitted to the EU Information System per EUDR Article 4(2) |
| **EUDR** | EU Deforestation Regulation -- Regulation (EU) 2023/1115 |
| **FPIC** | Free, Prior and Informed Consent -- the right of indigenous peoples to give or withhold consent to activities affecting their lands, territories, and resources |
| **Major Non-Conformity** | A fundamental failure to achieve the objective of a requirement, indicating the absence of a required process or a systematic breakdown in implementation |
| **Minor Non-Conformity** | A temporary, unusual, or non-systematic lapse that does not fundamentally undermine the compliance system |
| **Monitoring Organization** | An entity recognized under EUDR Article 16 that monitors operator compliance and may submit substantiated concerns |
| **NCR** | Non-Conformity Report -- a formal document recording a non-conformity identified during an audit |
| **Observation** | A finding that is not a non-conformity but indicates a potential risk of future non-conformity |
| **OFI** | Opportunity for Improvement -- a positive finding identifying areas where the organization could enhance its systems beyond minimum compliance |
| **RCA** | Root Cause Analysis -- a systematic investigation to identify the underlying causes of a non-conformity |
| **Substantiated Concern** | An evidence-based notification submitted to a competent authority per EUDR Article 14 alleging non-compliance |
| **Surveillance Audit** | A periodic audit conducted to verify continued compliance between initial certification and recertification |

---

## 20. References

### 20.1 Primary Regulatory Sources

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 on deforestation-free products
2. Regulation (EU) 2024/3234 amending Regulation (EU) 2023/1115 as regards the date of application
3. European Commission EUDR Guidance Document (2nd Edition, April 2025)
4. European Commission Implementing Regulation on the EU Information System (TRACES)

### 20.2 International Standards

5. ISO 19011:2018 -- Guidelines for auditing management systems
6. ISO/IEC 17065:2012 -- Conformity assessment: Requirements for bodies certifying products, processes and services
7. ISO/IEC 17021-1:2015 -- Conformity assessment: Requirements for bodies providing audit and certification of management systems
8. ISO/IEC 17021-2:2016 -- Competence requirements for auditing and certification of environmental management systems

### 20.3 Certification Scheme Standards

9. FSC-STD-20-011 V4-0 -- Chain of Custody Evaluations
10. FSC-STD-20-007 -- Forest Management Evaluations
11. FSC-STD-40-004 V3-0 -- Chain of Custody Certification
12. PEFC ST 2003:2020 -- Chain of Custody of Forest and Tree Based Products
13. RSPO Supply Chain Certification Standard (SCCS)
14. RSPO Principles and Criteria for the Production of Sustainable Palm Oil
15. Rainforest Alliance Certification and Auditing Rules (2024 update)
16. Rainforest Alliance Supply Chain Certification and Auditing Rules
17. ISCC EU/PLUS Certification Requirements

### 20.4 GreenLang Platform References

18. PRD-AGENT-EUDR-012 -- Document Authentication Agent
19. PRD-AGENT-EUDR-016 -- Country Risk Evaluator
20. PRD-AGENT-EUDR-017 -- Supplier Risk Scorer
21. PRD-AGENT-EUDR-018 -- Commodity Risk Analyzer
22. PRD-AGENT-EUDR-019 -- Corruption Index Monitor
23. PRD-AGENT-EUDR-020 -- Deforestation Alert System
24. PRD-AGENT-EUDR-023 -- Legal Compliance Verifier Agent
25. SEC-005 -- Centralized Audit Logging

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-10 | GL-RegulatoryIntelligence | Initial regulatory requirements document |

---

*This document was prepared by GL-RegulatoryIntelligence based on analysis of Regulation (EU) 2023/1115, ISO 19011:2018, ISO/IEC 17065:2012, ISO/IEC 17021-1:2015, and certification scheme audit standards (FSC, PEFC, RSPO, Rainforest Alliance, ISCC). All regulatory references are traceable to specific articles, clauses, and provisions. This document does not constitute legal advice.*
