# POL-004: Third-Party Risk Management Policy

| Document Control | |
|------------------|---|
| **Policy ID** | POL-004 |
| **Title** | Third-Party Risk Management Policy |
| **Version** | 1.0 |
| **Classification** | Internal |
| **Owner** | Chief Information Security Officer (CISO) |
| **Approved By** | Executive Leadership Team |
| **Effective Date** | 2026-02-06 |
| **Last Review Date** | 2026-02-06 |
| **Next Review Date** | 2027-02-06 |
| **Status** | Approved |

---

## 1. Purpose

This Third-Party Risk Management Policy establishes requirements for assessing, managing, and monitoring security risks associated with vendors, suppliers, contractors, and other third parties who access GreenLang systems, data, or facilities. Effective third-party risk management protects GreenLang and our customers from:

- Data breaches originating from vendor compromise
- Supply chain attacks and vulnerabilities
- Service disruptions affecting business operations
- Regulatory non-compliance and associated penalties
- Reputational damage from vendor security failures

This policy ensures that third parties meet GreenLang's security standards throughout the relationship lifecycle.

---

## 2. Scope

### 2.1 Applicability

This policy applies to all third-party relationships where the third party:

- Accesses GreenLang systems or networks
- Processes, stores, or transmits GreenLang data
- Provides software or services integrated into GreenLang products
- Has physical access to GreenLang facilities
- Provides critical business services or infrastructure

### 2.2 Third-Party Types Covered

| Type | Examples | Typical Risk Factors |
|------|----------|---------------------|
| Cloud Service Providers | AWS, GCP, Azure, SaaS applications | Data storage, processing, availability |
| Software Vendors | Enterprise software, development tools | Supply chain, vulnerabilities, support |
| Data Processors | Analytics providers, ML/AI services | Data handling, privacy, retention |
| IT Service Providers | MSPs, MSSPs, consultants | Access privileges, insider threat |
| Professional Services | Legal, accounting, audit | Confidential information access |
| Hardware Vendors | Servers, network equipment, laptops | Supply chain, firmware security |
| Contractors/Consultants | Development, design, specialized expertise | System access, IP protection |

### 2.3 Exclusions

This policy does not cover:

- One-time purchases of commodity goods without data access
- Public SaaS tools with only Public data (still require registration)
- Government agencies acting in regulatory capacity
- Individual consumers using GreenLang products

---

## 3. Policy Statement

### 3.1 Core Requirements

All third-party relationships must:

- Undergo risk assessment prior to engagement
- Be classified according to risk level
- Meet security requirements proportionate to risk
- Execute appropriate contractual protections
- Be monitored throughout the relationship
- Follow secure offboarding upon termination

### 3.2 Risk-Based Approach

Third-party controls are calibrated to risk level. Higher-risk relationships require more rigorous assessment, stronger contractual protections, and more frequent monitoring.

---

## 4. Vendor Risk Assessment

### 4.1 Assessment Criteria

Vendors are assessed across multiple dimensions:

| Dimension | Assessment Questions | Risk Indicators |
|-----------|---------------------|-----------------|
| **Data Access** | What data types will vendor access? What classification levels? | Restricted/Confidential data access |
| **System Access** | What systems will vendor access? What privilege levels? | Production access, admin rights |
| **Service Criticality** | Is vendor essential to operations? What's the impact of failure? | Single point of failure, no alternatives |
| **Financial Exposure** | What is the contract value? What are potential losses? | High contract value, significant liability |
| **Regulatory Impact** | What compliance requirements apply? | GDPR, SOC 2, industry regulations |
| **Geographic Risk** | Where is vendor located? Where is data processed? | High-risk jurisdictions, data sovereignty |
| **Security Maturity** | What certifications? What security program maturity? | No certifications, weak controls |

### 4.2 Assessment Tools

**Standard Information Gathering (SIG) Questionnaire:**

| Vendor Risk Level | Questionnaire Version | Sections Required |
|-------------------|----------------------|-------------------|
| Low | SIG Lite (18 questions) | Core security controls |
| Medium | SIG Core (100 questions) | Security policies, access control, encryption |
| High | SIG Full (350+ questions) | Comprehensive security assessment |
| Critical | SIG Full + Custom Addendum | All sections + GreenLang-specific requirements |

**SOC 2 Report Review:**

For vendors handling Confidential or Restricted data:

- Type II report preferred (covers operating effectiveness)
- Type I acceptable for new vendors with Type II commitment
- Report must cover relevant Trust Services Criteria
- Bridge letter required if report older than 12 months
- Exceptions and management responses reviewed

**Penetration Test Results:**

For Critical vendors:

- Annual penetration test required
- Report must be within 12 months
- Critical/High findings must be remediated
- Remediation evidence required for open findings

### 4.3 Assessment Process

**Pre-Engagement Assessment:**

```
Intake -> Risk Categorization -> Assessment -> Review -> Decision -> Contract
  |            |                     |           |          |          |
  v            v                     v           v          v          v
Request    Preliminary          Questionnaire  Security   Approve/   Execute
 Form      Risk Score           + Evidence     Analysis   Deny       Agreement
```

**Timeline:**

| Phase | Duration | Owner |
|-------|----------|-------|
| Intake | 1-2 days | Business owner |
| Risk Categorization | 1 day | Security team |
| Assessment Distribution | 1 day | Security team |
| Vendor Response | 10-15 business days | Vendor |
| Security Review | 5-10 business days | Security team |
| Decision | 1-2 days | Per approval matrix |
| Contract Negotiation | Variable | Legal + Security |

---

## 5. Risk Categorization

### 5.1 Risk Levels

| Risk Level | Criteria | Example Vendors |
|------------|----------|-----------------|
| **Critical** | Access to Restricted data AND (>$1M contract OR >100 affected users OR single point of failure) | Cloud infrastructure (AWS), identity provider (Okta), primary database host |
| **High** | Access to Confidential data OR >$100K contract OR production system access | SaaS applications with customer data, development contractors, security tools |
| **Medium** | Access to Internal data OR moderate contract value OR development environment access | Productivity tools, HR software, marketing platforms |
| **Low** | No data access OR only Public data AND commodity service | Office supplies, general consultants, public website hosting |

### 5.2 Risk Categorization Matrix

| Factor | Low (1) | Medium (2) | High (3) | Critical (4) |
|--------|---------|------------|----------|--------------|
| Data Classification | Public | Internal | Confidential | Restricted |
| Contract Value | <$10K | $10K-$100K | $100K-$1M | >$1M |
| Users Affected | <10 | 10-50 | 50-100 | >100 |
| System Access | None | Non-prod | Prod read-only | Prod read/write |
| Replaceability | Easy | Moderate | Difficult | Single source |
| Regulatory Impact | None | Minimal | Moderate | Significant |

**Overall Risk Level:** Highest individual factor score determines category.

---

## 6. Due Diligence Requirements

### 6.1 Requirements by Risk Level

| Requirement | Low | Medium | High | Critical |
|-------------|-----|--------|------|----------|
| SIG Questionnaire | Lite | Core | Full | Full + Custom |
| SOC 2 Report | Not required | Preferred | Required | Required (Type II) |
| Penetration Test | Not required | Not required | Preferred | Required (annual) |
| Insurance Verification | Not required | Required | Required | Required (enhanced limits) |
| Financial Review | Not required | Not required | Preferred | Required |
| Business Continuity Plan | Not required | Required | Required | Required + tested |
| Security Contact | Not required | Required | Required | 24/7 required |
| On-site Assessment | Not required | Not required | Optional | May be required |
| Reference Check | Not required | Optional | Recommended | Required |

### 6.2 Minimum Security Requirements

All vendors accessing GreenLang systems or data must demonstrate:

**Technical Controls:**
- Encryption at rest for Confidential/Restricted data (AES-256 minimum)
- Encryption in transit (TLS 1.2+ minimum)
- Multi-factor authentication for system access
- Access logging and monitoring
- Vulnerability management program
- Endpoint protection

**Administrative Controls:**
- Information security policy
- Employee background checks
- Security awareness training
- Incident response plan
- Change management process

**Physical Controls (where applicable):**
- Facility access controls
- Visitor management
- Environmental controls

---

## 7. Contract Security Requirements

### 7.1 Required Contract Terms

**Data Protection Addendum (DPA):**

All vendors processing personal data must execute a DPA including:

| Provision | Requirement |
|-----------|-------------|
| Processing Scope | Defined data types, processing activities, purposes |
| Data Location | Approved locations for data processing and storage |
| Sub-processors | Notification and approval requirements |
| Security Measures | Technical and organizational measures |
| Audit Rights | GreenLang right to audit or request evidence |
| Breach Notification | Notification timeline and requirements |
| Data Return/Deletion | Procedures upon termination |
| Liability | Data protection liability provisions |
| GDPR Clauses | Standard Contractual Clauses where required |

**Service Level Agreement (SLA):**

| Metric | Typical Requirement | Remedies |
|--------|--------------------|---------:|
| Availability | 99.9% (Critical), 99.5% (High) | Service credits |
| Response Time | Per severity level | Escalation procedures |
| Recovery Time | RTO/RPO aligned with criticality | Business continuity |
| Support Hours | 24/7 (Critical), Business hours (Standard) | Defined coverage |

**Security-Specific Terms:**

| Term | Description |
|------|-------------|
| **Audit Rights** | Right to audit vendor security controls annually or upon incident |
| **Penetration Testing** | Permission for GreenLang to conduct security testing |
| **Breach Notification** | 24 hours for suspected, 48 hours for confirmed breaches |
| **Compliance Certification** | Annual attestation of continued compliance |
| **Insurance Requirements** | Cyber liability ($5M Critical, $2M High, $1M Medium) |
| **Indemnification** | Vendor indemnifies for data breaches caused by vendor negligence |
| **Termination Rights** | Termination for material security breach or non-compliance |

### 7.2 Breach Notification Requirements

**Vendor Notification Obligations:**

| Event Type | Notification Timeline | Notification Method |
|------------|----------------------|---------------------|
| Suspected Security Incident | Within 24 hours | Email + phone to Security |
| Confirmed Data Breach | Within 24 hours | Email + phone to Security + Legal |
| System Availability Issue | Within 1 hour | Defined escalation path |
| Compliance Finding | Within 5 business days | Written notification |
| Sub-processor Change | 30 days advance notice | Written notification |

**Notification Content Requirements:**

- Nature and scope of incident
- Data types affected
- GreenLang systems/data impacted
- Remediation actions taken
- Timeline for investigation and resolution
- Point of contact for updates

---

## 8. Ongoing Monitoring

### 8.1 Annual Reassessment

All active vendors must be reassessed annually:

| Activity | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Updated SIG questionnaire | Required | Required | Required | Optional |
| SOC 2 report review | Required | Required | Preferred | Not required |
| Insurance verification | Required | Required | Required | Not required |
| Contract review | Required | Required | Recommended | As needed |
| Performance review | Required | Required | Required | Recommended |

### 8.2 Continuous Monitoring

**Security Monitoring:**

- Threat intelligence feeds for vendor breaches
- Dark web monitoring for vendor data exposure
- Security rating services (SecurityScorecard, BitSight)
- News monitoring for security incidents

**Performance Monitoring:**

- SLA compliance tracking
- Incident frequency and resolution time
- Change management adherence
- Support responsiveness

**Triggers for Reassessment:**

- Reported security breach
- Significant security rating decline (>10 points)
- Material change in vendor services
- Acquisition or change of ownership
- Regulatory action against vendor
- Negative security news coverage

### 8.3 Vendor Performance Review

Quarterly reviews for Critical vendors, semi-annual for High:

- SLA compliance metrics
- Security incident history
- Support quality and responsiveness
- Contract compliance
- Relationship health assessment
- Action items and follow-ups

---

## 9. Subcontractor Requirements

### 9.1 Subcontractor Approval

Vendors must obtain approval before using subcontractors who:

- Access GreenLang data
- Provide material components of the service
- Have access to GreenLang systems

**Approval Process:**

1. Vendor submits subcontractor notification (30 days advance)
2. Security reviews subcontractor details
3. Risk assessment conducted on subcontractor
4. Approval granted or objection raised within 14 days
5. Vendor ensures subcontractor compliance with security requirements

### 9.2 Subcontractor Requirements

Subcontractors must meet the same security requirements as the primary vendor:

- Bound by equivalent contractual obligations
- Covered by vendor's security program
- Subject to vendor's audit rights
- Listed in vendor's SOC 2 report (if applicable)

---

## 10. Vendor Onboarding Checklist

### 10.1 Pre-Engagement

- [ ] Business justification documented
- [ ] Initial risk categorization completed
- [ ] Security assessment questionnaire distributed
- [ ] Vendor response received and reviewed
- [ ] SOC 2 report obtained and reviewed (if required)
- [ ] Penetration test results reviewed (if required)
- [ ] Insurance certificates verified
- [ ] Financial due diligence completed (if required)
- [ ] Security team approval obtained
- [ ] Risk acceptance documented (if exceptions)

### 10.2 Contract Execution

- [ ] Master Service Agreement executed
- [ ] Data Protection Addendum executed
- [ ] SLA documented and agreed
- [ ] Security exhibit/addendum executed
- [ ] NDA executed (if separate)
- [ ] Business Associate Agreement (if PHI)
- [ ] Insurance requirements met

### 10.3 Technical Onboarding

- [ ] Access requirements documented
- [ ] Accounts provisioned per access control policy
- [ ] MFA enabled for all vendor accounts
- [ ] Network access configured (VPN, firewall rules)
- [ ] Logging enabled for vendor activity
- [ ] Monitoring configured for vendor accounts
- [ ] Data transfer mechanisms established
- [ ] Encryption verified for data exchange
- [ ] Emergency contact information documented
- [ ] Escalation procedures communicated

### 10.4 Governance Setup

- [ ] Vendor added to vendor inventory
- [ ] Contract stored in contract management system
- [ ] Assessment results archived
- [ ] Review schedule established
- [ ] Relationship owner assigned
- [ ] Annual reassessment scheduled

---

## 11. Vendor Offboarding Checklist

### 11.1 Pre-Termination

- [ ] Termination notice provided per contract
- [ ] Transition plan documented (if applicable)
- [ ] Data return/deletion requirements communicated
- [ ] Outstanding obligations identified
- [ ] Final invoice reconciliation

### 11.2 Access Revocation

- [ ] All user accounts disabled
- [ ] VPN access revoked
- [ ] API keys and tokens revoked
- [ ] Service account credentials rotated
- [ ] Firewall rules removed
- [ ] SSO integrations disabled
- [ ] Physical access badges collected (if applicable)
- [ ] Access revocation confirmed by IT

### 11.3 Data Handling

- [ ] GreenLang data returned (if applicable)
- [ ] Certificate of data destruction obtained
- [ ] Backup deletion confirmed
- [ ] Sub-processor notification of termination
- [ ] Retention obligations completed

### 11.4 Administrative Closure

- [ ] Final performance review completed
- [ ] Contract closure documented
- [ ] Vendor marked as inactive in inventory
- [ ] Lessons learned documented
- [ ] Final payment processed (if applicable)

---

## 12. Incident Notification Chain

### 12.1 Vendor-to-GreenLang

**Primary Contact:**
- Email: security@greenlang.io
- Phone: [Security Hotline Number]
- Escalation: CISO direct line

**After-Hours:**
- 24/7 Security Operations Center
- On-call security engineer

### 12.2 Internal Escalation

| Severity | Timeline | Escalation |
|----------|----------|------------|
| Critical (data breach involving customer data) | Immediate | CISO -> CEO -> Legal -> Customer Success |
| High (potential data exposure) | Within 1 hour | CISO -> CTO -> affected teams |
| Medium (service disruption) | Within 4 hours | Security Lead -> Operations |
| Low (minor incident) | Within 24 hours | Security analyst |

### 12.3 Customer Notification

When vendor incident affects customer data:

1. Security team assesses scope and impact
2. Legal reviews notification requirements
3. Customer Success prepares customer communication
4. CISO approves notification timing and content
5. Affected customers notified per GDPR (72 hours) or earlier
6. Status updates provided until resolution

---

## 13. Exceptions

### 13.1 Exception Scenarios

| Scenario | Typical Resolution | Approver |
|----------|-------------------|----------|
| Vendor cannot provide SOC 2 | Alternative evidence (ISO 27001, questionnaire + attestation) | Security Team |
| Vendor breach notification >24 hours | Documented risk acceptance with enhanced monitoring | CISO |
| Missing DPA for data processor | Escalate to Legal; delay engagement until resolved | General Counsel |
| Critical vendor with security gaps | Remediation plan + interim compensating controls | Security Council |

### 13.2 Exception Process

Follow POL-001 Section 6 for exception approval.

---

## 14. Enforcement

### 14.1 Vendor Non-Compliance

| Violation | Response |
|-----------|----------|
| Failure to complete assessment | Engagement blocked |
| Material security deficiency | Remediation required before/during engagement |
| Breach notification failure | Contract penalty; potential termination |
| Sub-processor non-compliance | Remediation or sub-processor removal required |
| Repeated non-compliance | Relationship review; potential termination |

### 14.2 Internal Non-Compliance

Personnel who bypass vendor risk management processes are subject to disciplinary action per POL-001 Section 9.2.

---

## 15. Definitions

| Term | Definition |
|------|------------|
| **Data Processor** | Third party that processes personal data on behalf of GreenLang |
| **DPA** | Data Protection Addendum - contract governing data processing |
| **SIG** | Shared Information Gathering - standardized security questionnaire |
| **SOC 2** | Service Organization Control 2 - audit standard for service providers |
| **Sub-processor** | Third party engaged by a vendor to process GreenLang data |
| **Third Party** | Any external organization with access to GreenLang data, systems, or facilities |
| **Vendor** | Third party providing goods or services to GreenLang |

---

## 16. Related Documents

| Document | Location |
|----------|----------|
| POL-001: Information Security Policy | `/docs/policies/tier1-critical/` |
| POL-002: Data Classification Policy | `/docs/policies/tier1-critical/` |
| POL-003: Access Control Policy | `/docs/policies/tier1-critical/` |
| Data Protection Addendum Template | `/docs/legal/` |
| Vendor Security Questionnaire (SIG) | `/docs/compliance/` |
| Vendor Risk Assessment Template | `/docs/compliance/` |

---

## 17. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-06 | CISO | Initial policy creation |

---

## 18. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Chief Information Security Officer | | | |
| General Counsel | | | |
| Chief Financial Officer | | | |

---

**Document Classification: Internal**

**Annual Review Commitment**: This policy shall be reviewed at least annually, or more frequently when significant changes occur in the vendor landscape, regulatory requirements, or following a vendor-related security incident.
