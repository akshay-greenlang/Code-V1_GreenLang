# POL-002: Data Classification Policy

| Document Control | |
|------------------|---|
| **Policy ID** | POL-002 |
| **Title** | Data Classification Policy |
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

This Data Classification Policy establishes a framework for categorizing GreenLang information assets based on their sensitivity and criticality. Proper data classification ensures that information receives appropriate protection throughout its lifecycle, from creation to disposal. This policy enables GreenLang to:

- Protect sensitive data from unauthorized access and disclosure
- Meet regulatory and contractual obligations for data protection
- Apply security controls proportionate to data sensitivity
- Enable informed decisions about data handling and sharing
- Support compliance with GDPR, SOC 2, ISO 27001, and industry regulations

---

## 2. Scope

### 2.1 Applicability

This policy applies to:

- **All Data**: Data in any form (digital, paper, verbal) created, received, stored, processed, or transmitted by GreenLang
- **All Personnel**: Employees, contractors, consultants, and third parties who handle GreenLang data
- **All Systems**: Information systems, applications, databases, and storage media containing GreenLang data
- **All Locations**: On-premises, cloud, remote work environments, and third-party facilities

### 2.2 Data Types Covered

- Customer data (emissions data, sustainability reports, account information)
- Employee data (HR records, payroll, performance reviews)
- Business data (financial records, contracts, strategic plans)
- Technical data (source code, configurations, architecture documents)
- Operational data (logs, metrics, audit trails)

---

## 3. Policy Statement

### 3.1 Classification Requirement

All information assets must be classified according to the classification levels defined in this policy. Data owners are responsible for assigning appropriate classifications, and all personnel must handle data according to its classification level.

### 3.2 Default Classification

When classification is unknown or unassigned, data shall be treated as **Confidential** until properly classified by a data owner.

---

## 4. Classification Levels

### 4.1 Classification Definitions

| Level | Label | Definition | Examples |
|-------|-------|------------|----------|
| **Public** | PUBLIC | Information approved for public release with no impact if disclosed | Marketing materials, public documentation, press releases, published blog posts, open-source code |
| **Internal** | INTERNAL | Business information for internal use with minimal impact if disclosed | Internal communications, meeting notes, non-sensitive operational documents, org charts, internal wikis |
| **Confidential** | CONFIDENTIAL | Sensitive business information that could cause harm if disclosed | Customer data, financial records, contracts, strategic plans, non-public product roadmaps, employee lists |
| **Restricted** | RESTRICTED | Highly sensitive information that could cause significant harm if disclosed | PII, PHI, PCI data, credentials, encryption keys, security vulnerabilities, incident reports, audit findings |

### 4.2 Classification Criteria Matrix

| Criterion | Public | Internal | Confidential | Restricted |
|-----------|--------|----------|--------------|------------|
| **Disclosure Impact** | None | Minimal | Moderate | Severe |
| **Regulatory Requirements** | None | None | GDPR, contracts | GDPR, PCI-DSS, HIPAA |
| **Access Scope** | Anyone | All employees | Need-to-know | Strictly limited |
| **Financial Impact if Breached** | None | <$10K | $10K-$1M | >$1M |
| **Reputational Impact** | None | Minimal | Moderate | Severe |
| **Legal/Regulatory Penalty** | None | None | Possible | Likely |

### 4.3 Data Type Classification Reference

| Data Type | Classification | Rationale |
|-----------|---------------|-----------|
| Marketing materials | Public | Intended for public consumption |
| Public API documentation | Public | Required for customer integration |
| Internal procedures | Internal | Operational efficiency |
| Employee directory | Internal | General business operations |
| Customer emissions data | Confidential | Customer proprietary information |
| Financial statements (draft) | Confidential | Pre-public business information |
| Vendor contracts | Confidential | Business relationship details |
| Personally Identifiable Information (PII) | Restricted | Regulatory protection required |
| API keys and tokens | Restricted | Security credentials |
| Encryption keys | Restricted | Cryptographic material |
| Security incident reports | Restricted | Vulnerability information |
| Penetration test results | Restricted | Security posture details |

---

## 5. Labeling Requirements

### 5.1 Digital Assets

All digital assets must be labeled with their classification level:

**File Naming Convention:**
```
[CLASSIFICATION]_filename.extension
```

Examples:
- `PUBLIC_api-documentation.pdf`
- `INTERNAL_team-meeting-notes.docx`
- `CONFIDENTIAL_customer-contract.pdf`
- `RESTRICTED_encryption-keys.json`

**Document Headers/Footers:**
- Classification label in header or footer of all documents
- First page must display classification prominently

**Email:**
- Subject line prefix: `[CLASSIFICATION] Subject`
- Classification notice in email signature for Confidential and Restricted emails

**Code and Configuration:**
- Classification comment in file header:
```python
# Classification: RESTRICTED
# Data Owner: Security Team
# Description: Encryption key management
```

**Database Fields:**
- Metadata column or tag indicating classification
- Schema documentation must include field classifications

### 5.2 Physical Assets

- Classification label on cover or first page
- Folders and binders labeled with highest classification contained
- Removable media (USB, external drives) labeled with classification

### 5.3 Systems and Applications

- System documentation must indicate highest classification of data processed
- Application login screens display classification banner for Confidential and Restricted systems

---

## 6. Handling Requirements

### 6.1 Storage Requirements

| Requirement | Public | Internal | Confidential | Restricted |
|-------------|--------|----------|--------------|------------|
| **Encryption at Rest** | Not required | Recommended | Required (AES-256) | Required (AES-256) |
| **Access Controls** | None | Authentication required | Authentication + authorization | MFA + authorization + audit logging |
| **Storage Location** | Any | Approved platforms | Approved secure platforms | Designated secure systems only |
| **Backup Encryption** | Not required | Recommended | Required | Required |
| **Physical Storage** | Open areas | Secured areas | Locked cabinets | Safes/security rooms |

### 6.2 Transmission Requirements

| Requirement | Public | Internal | Confidential | Restricted |
|-------------|--------|----------|--------------|------------|
| **Network Encryption** | TLS recommended | TLS 1.2+ required | TLS 1.3 required | TLS 1.3 + additional encryption |
| **Email** | Standard email | Standard email | Encrypted email recommended | Encrypted email required |
| **File Transfer** | Any method | Approved platforms | Secure file transfer (SFTP, HTTPS) | Encrypted transfer + audit trail |
| **External Sharing** | Permitted | With approval | DPA required | Prohibited without CISO approval |
| **Verbal Discussion** | Anywhere | Internal settings | Private settings | Secure locations only |

### 6.3 Disposal Requirements

| Requirement | Public | Internal | Confidential | Restricted |
|-------------|--------|----------|--------------|------------|
| **Digital Deletion** | Standard delete | Secure delete | Cryptographic erasure | Cryptographic erasure + verification |
| **Paper Destruction** | Recycling | Cross-cut shredding | Cross-cut shredding | Certified destruction |
| **Media Destruction** | Recycling | Degaussing/wiping | Degaussing + physical destruction | Physical destruction + certificate |
| **Retention Verification** | Not required | Spot checks | Documented process | Documented + witnessed |

---

## 7. Special Data Categories

### 7.1 Personally Identifiable Information (PII)

**Definition:** Information that can identify an individual, directly or indirectly.

**Examples:**
- Full name combined with other identifiers
- Email addresses
- Phone numbers
- Physical addresses
- Government IDs (SSN, passport, driver's license)
- IP addresses (in some contexts)
- Biometric data

**Classification:** Restricted

**Additional Requirements:**
- GDPR Article 6 legal basis required for processing
- Data minimization principle applied
- Retention limits enforced
- Subject access requests supported within 30 days
- Breach notification within 72 hours to authorities

### 7.2 Protected Health Information (PHI)

**Definition:** Individually identifiable health information created or received by a covered entity.

**Classification:** Restricted

**Additional Requirements:**
- HIPAA compliance where applicable
- Business Associate Agreements required
- Minimum necessary standard applied
- Access logging required

### 7.3 Payment Card Information (PCI)

**Definition:** Cardholder data including primary account number (PAN), cardholder name, expiration date, and service code.

**Classification:** Restricted

**Additional Requirements:**
- PCI-DSS compliance required
- No storage of full track data, CVV, or PIN
- Encryption required for storage and transmission
- Quarterly vulnerability scans
- Annual penetration testing

### 7.4 Credentials and Cryptographic Material

**Definition:** Passwords, API keys, tokens, certificates, and encryption keys.

**Classification:** Restricted

**Additional Requirements:**
- Secrets management system required (e.g., HashiCorp Vault)
- Never stored in source code or configuration files
- Rotation according to POL-009 Password and Authentication Policy
- Access limited to automated systems and essential personnel
- Audit logging for all access

---

## 8. Roles and Responsibilities

### 8.1 Data Owners

Data owners are accountable for data assets within their domain:

| Responsibility | Description |
|----------------|-------------|
| Classification | Assign appropriate classification to data assets |
| Access Approval | Approve access requests for their data |
| Protection | Ensure appropriate controls are implemented |
| Review | Conduct periodic reviews of classification and access |
| Reclassification | Update classification when circumstances change |
| Retention | Define and enforce retention periods |

**Data Owner Assignment:**
- Customer data: VP of Customer Success
- Employee data: VP of Human Resources
- Financial data: Chief Financial Officer
- Technical data: Chief Technology Officer
- Security data: Chief Information Security Officer

### 8.2 Data Custodians

Data custodians implement technical controls for data protection:

- Implement access controls per data owner direction
- Maintain encryption and security controls
- Perform backups and recovery procedures
- Monitor for security incidents
- Report issues to data owners

### 8.3 Data Users

All personnel who access data:

- Handle data according to its classification
- Apply appropriate labeling
- Report suspected misclassification
- Complete required training
- Report security incidents

---

## 9. Reclassification Procedures

### 9.1 Triggers for Reclassification

Data should be reviewed for reclassification when:
- Business context changes
- Regulatory requirements change
- Data is aggregated or anonymized
- Retention period expires
- Public disclosure occurs (intentional or unintentional)
- Contractual obligations change

### 9.2 Reclassification Process

1. **Initiation**: Data owner or custodian identifies reclassification need
2. **Assessment**: Evaluate data against classification criteria
3. **Approval**:
   - Downgrade: Data owner approval
   - Upgrade: Data owner approval + notification to current accessors
4. **Implementation**: Update labels, access controls, and metadata
5. **Communication**: Notify affected personnel of classification change
6. **Documentation**: Record reclassification in data inventory

### 9.3 Downgrade Requirements

Before downgrading classification:
- Verify regulatory requirements permit lower classification
- Confirm contractual obligations are met
- Remove or redact sensitive elements if partial downgrade
- Update all copies and backups
- Document justification for downgrade

---

## 10. Retention Schedule

### 10.1 Retention Periods by Data Type

| Data Type | Classification | Retention Period | Legal Basis |
|-----------|---------------|------------------|-------------|
| Customer emissions data | Confidential | 7 years after contract end | Regulatory (EU CSRD) |
| Customer account information | Confidential | Duration of contract + 7 years | Legal/contractual |
| Financial records | Confidential | 7 years | Tax regulations |
| Employee records | Restricted | Duration of employment + 7 years | Employment law |
| Audit logs | Restricted | 7 years | Compliance (SOC 2, ISO 27001) |
| Security incident records | Restricted | 7 years | Compliance |
| Marketing analytics | Internal | 3 years | Business need |
| Application logs | Internal | 90 days operational, 365 days compliance | Operations/compliance |
| Backup data | Per source classification | Per source + 35 days | Disaster recovery |

### 10.2 Retention Management

- Data owners define retention periods for their data types
- Automated retention enforcement where technically feasible
- Annual review of retention schedules
- Legal hold procedures override standard retention
- Destruction documented with certificates for Restricted data

---

## 11. Exceptions

### 11.1 Exception Process

Exceptions to this policy must follow the process defined in POL-001 Information Security Policy Section 6.

### 11.2 Common Exception Scenarios

| Scenario | Typical Resolution |
|----------|-------------------|
| Legacy system cannot support encryption | Compensating controls + migration timeline |
| Third party requires unencrypted transfer | DPA with security requirements + encrypted at rest |
| Temporary elevated access for project | Time-limited exception with monitoring |

---

## 12. Enforcement

### 12.1 Compliance Monitoring

- Automated classification scanning for common patterns (PII, credentials)
- Periodic access reviews by data owners
- Random audits of data handling practices
- Incident investigation findings

### 12.2 Violations

Violations are handled according to POL-001 Information Security Policy Section 9.2.

---

## 13. Definitions

| Term | Definition |
|------|------------|
| **Data Asset** | Any collection of data that has value to the organization |
| **Data Custodian** | Individual or team responsible for technical implementation of data protection |
| **Data Owner** | Individual accountable for classification and protection decisions for a data asset |
| **Data User** | Any individual who accesses, processes, or handles data |
| **Encryption at Rest** | Encryption applied to data stored on disk or other media |
| **Encryption in Transit** | Encryption applied to data transmitted over networks |
| **PII** | Personally Identifiable Information - data that can identify an individual |
| **PHI** | Protected Health Information - health-related data subject to HIPAA |
| **PCI** | Payment Card Industry - cardholder data subject to PCI-DSS |
| **Retention Period** | Duration for which data must be preserved before disposal |
| **Sanitization** | Process of removing data from media to prevent recovery |

---

## 14. Related Documents

| Document | Location |
|----------|----------|
| POL-001: Information Security Policy | `/docs/policies/tier1-critical/` |
| POL-003: Access Control Policy | `/docs/policies/tier1-critical/` |
| POL-011: Encryption and Key Management Policy | `/docs/policies/tier3-compliance/` |
| POL-015: Media Protection Policy | `/docs/policies/tier3-compliance/` |
| Data Retention Schedule | `/docs/compliance/` |
| GDPR Mapping | `/docs/policies/compliance-mapping/` |

---

## 15. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-06 | CISO | Initial policy creation |

---

## 16. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Chief Information Security Officer | | | |
| Chief Technology Officer | | | |
| General Counsel | | | |

---

**Document Classification: Internal**

**Annual Review Commitment**: This policy shall be reviewed at least annually, or more frequently when significant changes occur in regulatory requirements, data types processed, or business operations.
