# GDPR Compliance Mapping

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | MAP-GDPR-001 |
| Version | 1.0 |
| Classification | Internal |
| Owner | Data Protection Officer (DPO) |
| Approved By | Chief Executive Officer (CEO) |
| Effective Date | 2026-02-06 |
| Last Updated | 2026-02-06 |
| Next Review | 2026-08-06 |

---

## 1. Overview

This document provides an article-by-article mapping of the General Data Protection Regulation (GDPR) requirements to GreenLang policies, implementation evidence, and gap analysis. This mapping supports GDPR compliance verification, Data Protection Impact Assessments (DPIAs), and regulatory inquiries.

**Scope:** GreenLang Climate OS Platform and all processing activities involving EU/EEA personal data

**Regulation:** Regulation (EU) 2016/679 (General Data Protection Regulation)

**GreenLang Role:** Data Controller (platform services), Data Processor (customer data processing)

**Lead Supervisory Authority:** [To be determined based on main establishment]

---

## 2. Compliance Summary

| Chapter | Articles | Applicable | Fully Compliant | Partially Compliant | Gap |
|---------|----------|------------|-----------------|---------------------|-----|
| II - Principles | 5-11 | 7 | 7 | 0 | 0 |
| III - Data Subject Rights | 12-23 | 12 | 12 | 0 | 0 |
| IV - Controller/Processor | 24-43 | 20 | 19 | 1 | 0 |
| V - Transfers | 44-49 | 6 | 6 | 0 | 0 |
| **TOTAL** | **45** | **45** | **44** | **1** | **0** |

---

## 3. Chapter II - Principles (Articles 5-11)

### Article 5 - Principles Relating to Processing of Personal Data

#### 5(1)(a) - Lawfulness, Fairness, and Transparency

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-001 (Information Security Policy) |
| **Implementation Evidence** | Published privacy policy, Consent mechanisms, Processing register |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

#### 5(1)(b) - Purpose Limitation

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Personal data shall be collected for specified, explicit and legitimate purposes and not further processed in a manner incompatible with those purposes. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-002 (Data Classification Policy) |
| **Implementation Evidence** | Privacy notices with purpose statements, Processing register, Purpose documentation |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

#### 5(1)(c) - Data Minimization

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Personal data shall be adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-002 (Data Classification Policy) |
| **Implementation Evidence** | Data collection justifications, Privacy by design reviews, Data inventory |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

#### 5(1)(d) - Accuracy

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Personal data shall be accurate and, where necessary, kept up to date; every reasonable step must be taken to ensure that personal data that are inaccurate are erased or rectified without delay. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Data quality controls, Self-service profile updates, Rectification procedures |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

#### 5(1)(e) - Storage Limitation

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed. |
| **Applicable Policies** | POL-005 (Data Retention Policy), POL-017 (Privacy Policy) |
| **Implementation Evidence** | Retention schedules, Automated deletion processes, Retention reviews |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

#### 5(1)(f) - Integrity and Confidentiality

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Personal data shall be processed in a manner that ensures appropriate security of the personal data, including protection against unauthorized or unlawful processing and against accidental loss, destruction or damage. |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-011 (Encryption Policy), POL-003 (Access Control Policy) |
| **Implementation Evidence** | Security controls, Encryption at rest/transit, Access controls, Audit logs |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

#### 5(2) - Accountability

| Attribute | Detail |
|-----------|--------|
| **Requirement** | The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1. |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-017 (Privacy Policy), POL-014 (Risk Management Policy) |
| **Implementation Evidence** | Processing records, Compliance documentation, Audit trails, DPIAs |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 6 - Lawfulness of Processing

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Processing shall be lawful only if at least one of the legal bases applies. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Legal basis documentation per processing activity, Consent records, Legitimate interest assessments |
| **Compliance Status** | Fully Compliant |
| **Legal Bases Used** | Contract (6(1)(b)), Legitimate Interest (6(1)(f)), Consent (6(1)(a)), Legal Obligation (6(1)(c)) |
| **Gap Analysis** | None |

### Article 7 - Conditions for Consent

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Where processing is based on consent, the controller shall be able to demonstrate that the data subject has consented. Consent must be freely given, specific, informed, and unambiguous. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Consent forms, Cookie consent banner, Consent logs, Withdrawal mechanisms |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 8 - Conditions for Child's Consent

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Processing of personal data of children under 16 requires parental consent. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | B2B service (not directed at children), Terms require 18+, Age verification |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None - GreenLang services are B2B and not directed at children |

### Article 9 - Processing of Special Categories

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Processing of special categories of data is prohibited unless specific conditions apply. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-002 (Data Classification Policy) |
| **Implementation Evidence** | No special category data collected by design, Data classification controls |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None - GreenLang does not process special category data |

### Article 10 - Processing of Criminal Conviction Data

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Processing of criminal conviction data only under official authority or authorization. |
| **Applicable Policies** | POL-005 (Personnel Security Policy) |
| **Implementation Evidence** | Background checks conducted by authorized provider, Limited retention |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 11 - Processing Not Requiring Identification

| Attribute | Detail |
|-----------|--------|
| **Requirement** | If purposes do not require identification of data subject, controller not obliged to maintain additional information solely to comply with GDPR. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Anonymization procedures, Pseudonymization where applicable |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

---

## 4. Chapter III - Rights of the Data Subject (Articles 12-23)

### Article 12 - Transparent Information and Communication

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Controller shall take appropriate measures to provide information in a concise, transparent, intelligible and easily accessible form. Response to requests within one month. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Privacy policy (plain language), Response SLAs (30 days), Layered notices |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 13 - Information When Data Collected from Data Subject

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Provide identity, purposes, legal basis, recipients, transfers, retention, rights, and other required information at time of collection. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Privacy notice at registration, Collection point notices, Cookie notices |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

**Required Information Checklist:**

| Information Element | Provided | Location |
|---------------------|----------|----------|
| Controller identity and contact | Yes | Privacy Policy Section 1 |
| DPO contact | Yes | Privacy Policy Section 18 |
| Purposes of processing | Yes | Privacy Policy Section 5 |
| Legal basis | Yes | Privacy Policy Section 5.2 |
| Legitimate interests | Yes | Privacy Policy Section 5.3 |
| Recipients/categories | Yes | Privacy Policy Section 8 |
| International transfers | Yes | Privacy Policy Section 9 |
| Retention periods | Yes | Privacy Policy Section 7 |
| Data subject rights | Yes | Privacy Policy Section 6 |
| Right to withdraw consent | Yes | Privacy Policy Section 6 |
| Right to lodge complaint | Yes | Privacy Policy Section 18 |
| Statutory/contractual requirement | Yes | Privacy Policy Section 4 |
| Automated decision-making | Yes | Privacy Policy Section 6 |

### Article 14 - Information When Data Not Obtained from Data Subject

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Provide required information within a reasonable period (max 1 month) when data obtained from third parties. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Third-party data handling procedures, Notification processes |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 15 - Right of Access

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right to obtain confirmation of processing and access to personal data. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Access request form, Request handling procedure, Response within 30 days |
| **Compliance Status** | Fully Compliant |
| **Response SLA** | 30 calendar days |
| **Gap Analysis** | None |

**Access Request Process:**
1. Request received via privacy@greenlang.io or in-platform
2. Identity verification within 3 business days
3. Data compilation from all systems
4. Response prepared in common format (JSON/CSV)
5. Response delivered within 30 days
6. Request logged and retained

### Article 16 - Right to Rectification

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right to obtain rectification of inaccurate personal data and completion of incomplete data. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Self-service profile editing, Rectification request procedure |
| **Compliance Status** | Fully Compliant |
| **Response SLA** | 30 calendar days |
| **Gap Analysis** | None |

### Article 17 - Right to Erasure (Right to be Forgotten)

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right to obtain erasure when data no longer necessary, consent withdrawn, objection upheld, or unlawful processing. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-005 (Data Retention Policy) |
| **Implementation Evidence** | Erasure request procedure, Deletion workflows, Third-party notification |
| **Compliance Status** | Fully Compliant |
| **Response SLA** | 30 calendar days |
| **Gap Analysis** | None |

**Erasure Exceptions Documented:**
- Legal compliance requirements
- Public interest archiving
- Legal claims defense
- Freedom of expression (N/A for B2B platform)

### Article 18 - Right to Restriction of Processing

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right to obtain restriction of processing in certain circumstances. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Restriction request procedure, Technical capability to restrict, Flagging mechanism |
| **Compliance Status** | Fully Compliant |
| **Response SLA** | 30 calendar days |
| **Gap Analysis** | None |

### Article 19 - Notification Obligation Regarding Rectification/Erasure/Restriction

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Controller shall communicate rectification, erasure or restriction to each recipient unless impossible or disproportionate effort. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-004 (Third-Party Risk Policy) |
| **Implementation Evidence** | Recipient notification procedures, Notification logs |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 20 - Right to Data Portability

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right to receive personal data in structured, commonly used, machine-readable format and transmit to another controller. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Export functionality (JSON, CSV), Portability request procedure |
| **Compliance Status** | Fully Compliant |
| **Export Formats** | JSON, CSV |
| **Response SLA** | 30 calendar days |
| **Gap Analysis** | None |

### Article 21 - Right to Object

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right to object to processing based on legitimate interest or for direct marketing. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Objection handling procedure, Marketing opt-out, Unsubscribe links |
| **Compliance Status** | Fully Compliant |
| **Response SLA** | Immediate for marketing; 30 days for legitimate interest assessment |
| **Gap Analysis** | None |

### Article 22 - Automated Decision-Making

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Data subject has right not to be subject to solely automated decisions with legal or significant effects. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | No solely automated decision-making with legal effects, Human review processes |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None - GreenLang does not make solely automated decisions with legal/significant effects |

### Article 23 - Restrictions

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Union or Member State law may restrict certain rights and obligations when necessary and proportionate. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Legal review process for restriction applicability |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

---

## 5. Chapter IV - Controller and Processor (Articles 24-43)

### Article 24 - Responsibility of the Controller

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Controller shall implement appropriate technical and organizational measures to ensure and demonstrate compliance. |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-017 (Privacy Policy), All policies |
| **Implementation Evidence** | ISMS implementation, Policy framework, Technical controls, Compliance documentation |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 25 - Data Protection by Design and Default

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Implement appropriate measures (pseudonymization, minimization) at design time and ensure only necessary data processed by default. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-010 (SDLC Security Policy) |
| **Implementation Evidence** | Privacy by design reviews, Default privacy settings, Data minimization in design |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

**Privacy by Design Principles Implemented:**
- Proactive not reactive
- Privacy as the default setting
- Privacy embedded into design
- Full functionality with privacy
- End-to-end security
- Visibility and transparency
- Respect for user privacy

### Article 26 - Joint Controllers

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Where two or more controllers jointly determine purposes and means, they shall determine responsibilities by transparent arrangement. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-004 (Third-Party Risk Policy) |
| **Implementation Evidence** | Joint controller agreements (where applicable), Responsibility allocation |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None - No current joint controller arrangements |

### Article 27 - Representatives of Controllers Not Established in the Union

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Controllers not established in EU must designate a representative in the EU. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | EU representative designated (if applicable) |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 28 - Processor

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Processing by a processor shall be governed by a contract with specific required terms. |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy), POL-017 (Privacy Policy) |
| **Implementation Evidence** | DPA templates, Signed DPAs with all processors, Processor register |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

**DPA Required Terms Checklist:**

| Term | Included | Location |
|------|----------|----------|
| Subject matter and duration | Yes | DPA Section 1 |
| Nature and purpose | Yes | DPA Section 1 |
| Type of personal data | Yes | DPA Schedule A |
| Categories of data subjects | Yes | DPA Schedule A |
| Controller obligations and rights | Yes | DPA Section 3 |
| Processing only on documented instructions | Yes | DPA Section 4.1 |
| Confidentiality obligation | Yes | DPA Section 4.2 |
| Security measures | Yes | DPA Section 5 |
| Sub-processor requirements | Yes | DPA Section 6 |
| Assistance with data subject rights | Yes | DPA Section 7 |
| Assistance with Articles 32-36 | Yes | DPA Section 8 |
| Deletion or return at end | Yes | DPA Section 9 |
| Audit rights | Yes | DPA Section 10 |

### Article 29 - Processing Under Authority of Controller/Processor

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Processor and anyone acting under controller/processor authority shall process only on instructions. |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy) |
| **Implementation Evidence** | DPA instruction requirements, Processor acknowledgments |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 30 - Records of Processing Activities

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Controllers and processors shall maintain records of processing activities. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Processing activity register (ROPA), Regular updates |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

**ROPA Maintained At:** docs/compliance/processing-register.xlsx

**ROPA Contents (Article 30(1)):**
- Controller name and contact details
- DPO contact details
- Purposes of processing
- Categories of data subjects
- Categories of personal data
- Categories of recipients
- International transfers and safeguards
- Retention periods
- Security measures description

### Article 31 - Cooperation with Supervisory Authority

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Controller and processor shall cooperate with supervisory authority in performance of its tasks. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-018 (Incident Communication Policy) |
| **Implementation Evidence** | Regulatory cooperation procedures, DPO authority liaison role |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 32 - Security of Processing

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Implement appropriate technical and organizational measures to ensure security appropriate to risk, including pseudonymization, encryption, confidentiality, availability, resilience, and regular testing. |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-011 (Encryption Policy), POL-003 (Access Control Policy) |
| **Implementation Evidence** | Technical security controls, Encryption implementation, Penetration tests, Security assessments |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

**Article 32 Measures Implemented:**

| Measure | Implementation |
|---------|----------------|
| Pseudonymization | Applied where feasible in analytics and logs |
| Encryption | AES-256 at rest, TLS 1.3 in transit |
| Confidentiality | RBAC, MFA, access controls |
| Integrity | Input validation, checksums, audit trails |
| Availability | Multi-AZ, DR, 99.9% SLA |
| Resilience | Auto-scaling, circuit breakers, graceful degradation |
| Regular testing | Annual penetration tests, continuous vulnerability scanning |

### Article 33 - Notification of Breach to Supervisory Authority

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Notify supervisory authority within 72 hours of becoming aware of breach likely to result in risk to individuals. |
| **Applicable Policies** | POL-006 (Incident Response Policy), POL-018 (Incident Communication Policy), POL-017 (Privacy Policy) |
| **Implementation Evidence** | Breach notification procedure, Notification templates, DPA contact list |
| **Compliance Status** | Fully Compliant |
| **Notification SLA** | 72 hours |
| **Gap Analysis** | None |

**Breach Notification Workflow:**
1. Breach detected and incident declared
2. Initial assessment within 24 hours
3. Risk to individuals evaluated
4. If risk likely, notification prepared
5. DPO reviews and approves notification
6. Submitted to supervisory authority within 72 hours
7. Documentation maintained

### Article 34 - Communication of Breach to Data Subject

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Communicate breach to data subject without undue delay when high risk to rights and freedoms. |
| **Applicable Policies** | POL-018 (Incident Communication Policy), POL-017 (Privacy Policy) |
| **Implementation Evidence** | Individual notification procedure, Communication templates |
| **Compliance Status** | Fully Compliant |
| **Notification SLA** | Without undue delay (target 72 hours after high risk determination) |
| **Gap Analysis** | None |

### Article 35 - Data Protection Impact Assessment

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Carry out DPIA for processing likely to result in high risk, including systematic monitoring, large-scale special categories, or automated decision-making. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | DPIA procedure, Completed DPIAs, DPIA register |
| **Compliance Status** | Partially Compliant |
| **Gap Analysis** | DPIA template exists; completing DPIAs for legacy processing activities |
| **Remediation** | Complete DPIA backlog by Q2 2026 |

**DPIA Triggers:**
- Systematic and extensive profiling with significant effects
- Large-scale processing of special categories
- Systematic monitoring of publicly accessible areas
- New technologies with high risk
- Large-scale processing preventing data subject rights
- Matching or combining datasets
- Data on vulnerable subjects
- Biometric/genetic data for unique identification
- Data transferred outside EU with safeguard limitations

### Article 36 - Prior Consultation

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Consult supervisory authority prior to processing where DPIA indicates high risk that cannot be mitigated. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Consultation procedure documented |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None - No processing identified requiring prior consultation |

### Article 37 - Designation of Data Protection Officer

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Designate DPO where core activities consist of large-scale regular and systematic monitoring or large-scale processing of special categories. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | DPO appointed, Contact published, Independence documented |
| **Compliance Status** | Fully Compliant |
| **DPO Contact** | dpo@greenlang.io |
| **Gap Analysis** | None |

### Article 38 - Position of the Data Protection Officer

| Attribute | Detail |
|-----------|--------|
| **Requirement** | DPO shall be properly involved, have resources, independence, and report to highest management. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | DPO reporting structure, Resource allocation, Independence documentation |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 39 - Tasks of the Data Protection Officer

| Attribute | Detail |
|-----------|--------|
| **Requirement** | DPO shall inform and advise, monitor compliance, provide advice on DPIAs, cooperate with supervisory authority, and act as contact point. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | DPO task documentation, Activity logs, Training records |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 40 - Codes of Conduct

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Associations may prepare codes of conduct for GDPR compliance. |
| **Applicable Policies** | N/A |
| **Implementation Evidence** | Monitoring industry codes for potential adoption |
| **Compliance Status** | Fully Compliant (no mandatory requirement) |
| **Gap Analysis** | None |

### Article 41 - Monitoring of Approved Codes of Conduct

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Monitoring of codes of conduct by accredited bodies. |
| **Applicable Policies** | N/A |
| **Implementation Evidence** | N/A |
| **Compliance Status** | Fully Compliant (no mandatory requirement) |
| **Gap Analysis** | None |

### Article 42 - Certification

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Certification mechanisms may be established to demonstrate compliance. |
| **Applicable Policies** | POL-001 (Information Security Policy) |
| **Implementation Evidence** | ISO 27001 certification (in progress), SOC 2 Type II |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 43 - Certification Bodies

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Certification issued by accredited certification bodies. |
| **Applicable Policies** | N/A |
| **Implementation Evidence** | Using accredited certification bodies for ISO/SOC certifications |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

---

## 6. Chapter V - Transfers to Third Countries (Articles 44-49)

### Article 44 - General Principle for Transfers

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Transfers to third countries only with adequate protection per Chapter V provisions. |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-004 (Third-Party Risk Policy) |
| **Implementation Evidence** | Transfer assessment procedure, Transfer inventory |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 45 - Transfers Based on Adequacy Decision

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Transfer may take place to countries with adequacy decision. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Adequacy decision monitoring, Transfer to adequate countries documented |
| **Compliance Status** | Fully Compliant |
| **Adequate Countries Used** | UK, Canada, Japan, Republic of Korea, Switzerland |
| **Gap Analysis** | None |

### Article 46 - Transfers Subject to Appropriate Safeguards

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Transfers with appropriate safeguards (SCCs, BCRs, codes of conduct, certification). |
| **Applicable Policies** | POL-017 (Privacy Policy), POL-004 (Third-Party Risk Policy) |
| **Implementation Evidence** | Signed SCCs, Transfer Impact Assessments, Supplementary measures |
| **Compliance Status** | Fully Compliant |
| **Transfer Mechanism** | EU Standard Contractual Clauses (2021) |
| **Gap Analysis** | None |

**US Transfers (Primary):**
- Transfer mechanism: Standard Contractual Clauses (Module 2: Controller to Processor)
- Transfer Impact Assessment: Completed
- Supplementary measures: Encryption, access controls, contractual commitments

### Article 47 - Binding Corporate Rules

| Attribute | Detail |
|-----------|--------|
| **Requirement** | BCRs may authorize intra-group transfers. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | BCRs not currently adopted; using SCCs |
| **Compliance Status** | Fully Compliant (using alternative mechanism) |
| **Gap Analysis** | None |

### Article 48 - Transfers Not Authorized by Union Law

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Judgments of third country courts/tribunals requiring transfer only recognized if based on international agreement. |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Legal review procedure for foreign requests |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

### Article 49 - Derogations for Specific Situations

| Attribute | Detail |
|-----------|--------|
| **Requirement** | Specific derogations for one-time transfers (explicit consent, contract necessity, public interest, legal claims, vital interests, public register). |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Evidence** | Derogation assessment procedure, Consent mechanisms for one-time transfers |
| **Compliance Status** | Fully Compliant |
| **Gap Analysis** | None |

---

## 7. DPA Notification Checklist

### Breach Notification to Supervisory Authority (Article 33)

| Item | Status | Notes |
|------|--------|-------|
| Lead supervisory authority identified | Complete | [Authority name] |
| DPA contact details confirmed | Complete | [Contact information] |
| Notification portal access established | Complete | [Portal URL] |
| Notification template prepared | Complete | POL-018 Appendix |
| 72-hour process documented | Complete | Incident Response Procedure |
| DPO notification role defined | Complete | DPO reviews and submits |
| Documentation retention defined | Complete | 7 years |

### Notification Content Checklist (Article 33(3))

| Required Element | Template Section |
|------------------|------------------|
| Nature of breach | Section 2 |
| Categories and approximate number of data subjects | Section 3 |
| Categories and approximate number of records | Section 3 |
| DPO name and contact details | Section 1 |
| Likely consequences | Section 5 |
| Measures taken or proposed | Section 6 |

---

## 8. DPIA Requirements

### When DPIA Required

| Criterion | Threshold | Assessment |
|-----------|-----------|------------|
| Systematic and extensive automated processing with significant effects | Any profiling affecting rights | DPIA required |
| Large-scale special categories | Large scale as defined by DPA guidance | DPIA required |
| Systematic monitoring of public area | Any systematic monitoring | DPIA required |
| New technologies | High risk + new technology | DPIA required |
| Evaluation or scoring | Including profiling | Consider DPIA |
| Automated decision-making with legal effects | Any | DPIA required |
| Sensitive data or highly personal | Health, financial, location | Consider DPIA |
| Large scale processing | Per DPA guidance | Consider DPIA |
| Matching datasets | Different purposes/controllers | Consider DPIA |
| Vulnerable data subjects | Employees, children, patients | Consider DPIA |

### DPIA Process

1. **Screening:** Determine if DPIA required
2. **Description:** Document processing operation
3. **Necessity Assessment:** Assess necessity and proportionality
4. **Risk Identification:** Identify risks to data subjects
5. **Risk Evaluation:** Assess likelihood and severity
6. **Mitigation:** Identify measures to address risks
7. **DPO Advice:** Obtain DPO input
8. **Documentation:** Record DPIA findings
9. **Review:** Regular review and update

### DPIA Register

| Processing Activity | DPIA Required | DPIA Completed | Review Date |
|--------------------|---------------|----------------|-------------|
| Customer emissions processing | Yes | Yes | 2026-08-01 |
| Marketing analytics | Yes | Yes | 2026-08-01 |
| Employee monitoring | Yes | Yes | 2026-08-01 |
| AI/ML model training | Yes | In Progress | Q2 2026 |

---

## 9. Gap Analysis Summary

| Article | Gap Description | Owner | Due Date | Status | Priority |
|---------|-----------------|-------|----------|--------|----------|
| Art. 35 | Complete DPIA backlog for legacy processing | DPO | Q2 2026 | In Progress | Medium |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Data Protection Officer | Initial mapping creation |

---

*This mapping is reviewed annually and updated when regulations, guidance, or processing activities change.*
