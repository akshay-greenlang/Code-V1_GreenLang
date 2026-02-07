# Evidence Collection Guide

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-EVD-001 |
| Version | 1.0 |
| Classification | Internal |
| Owner | Chief Information Security Officer (CISO) |
| Approved By | CEO |
| Effective Date | 2026-02-06 |
| Next Review | 2027-02-06 |

---

## 1. Purpose

This guide establishes the procedures for collecting, organizing, and maintaining evidence to demonstrate compliance with GreenLang policies. Proper evidence collection is essential for SOC 2 Type II, ISO 27001:2022, and regulatory audits.

---

## 2. Scope

This guide applies to:
- All policy and control owners responsible for evidence collection
- Compliance team members preparing audit packages
- IT and Security teams providing technical evidence
- External auditors reviewing GreenLang controls

---

## 3. Evidence Types by Control

### 3.1 Evidence Type Definitions

| Type | Description | Examples |
|------|-------------|----------|
| **Screenshots** | Point-in-time capture of system configuration or status | AWS IAM policy, firewall rules, user access list |
| **System Logs** | Automated records of system activity | Access logs, change logs, security events |
| **Exports** | Data extracts from systems | User list CSV, configuration JSON, report PDFs |
| **Attestations** | Written statements confirming compliance | Management attestation, vendor SOC reports |
| **Policies/Procedures** | Documented governance documents | Approved policies, runbooks, playbooks |
| **Training Records** | Evidence of security awareness | Completion certificates, attendance logs |
| **Meeting Minutes** | Records of governance activities | Risk committee meetings, change advisory board |
| **Tickets/Records** | Workflow documentation | Incident tickets, change requests, access requests |

### 3.2 Evidence Requirements by Policy

#### POL-001: Information Security Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Policy approval | Attestation | Signed approval from CEO/Board | Annual |
| Policy communication | Screenshot | Email/intranet announcement | Annual |
| Policy review | Meeting minutes | Annual review meeting documentation | Annual |

#### POL-002: Acceptable Use Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Employee acknowledgment | Export | Acknowledgment completion report | Annual |
| Monitoring notice | Screenshot | Login banner, policy portal | Annual |
| Violation handling | Tickets | Sample violation investigation records | Quarterly |

#### POL-003: Access Control Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Access provisioning | Tickets | Sample access request approvals | Quarterly |
| Access reviews | Export | Quarterly access review completion | Quarterly |
| Privileged access | Screenshot | Privileged user list with approvals | Quarterly |
| MFA enforcement | Screenshot | MFA configuration, compliance report | Quarterly |
| Termination access removal | Logs | Sample termination access removal | Quarterly |
| Password policy | Screenshot | Password policy configuration | Annual |

#### POL-004: Data Classification Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Classification schema | Policy | Approved classification definitions | Annual |
| Data inventory | Export | Data asset inventory with classifications | Annual |
| Handling procedures | Procedure | Data handling procedures by classification | Annual |
| Labeling | Screenshot | Sample labeled documents/systems | Annual |

#### POL-005: Data Retention Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Retention schedule | Policy | Approved retention schedule | Annual |
| Automated retention | Screenshot | Retention automation configuration | Annual |
| Deletion records | Logs | Data deletion/destruction logs | Quarterly |
| Legal hold | Tickets | Legal hold implementation records | As needed |

#### POL-006: Incident Response Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| IR plan | Policy | Approved incident response plan | Annual |
| IR team roster | Export | Current IR team contact list | Quarterly |
| IR tabletop | Meeting minutes | Tabletop exercise documentation | Annual |
| Incident tickets | Tickets | Sample incident records (redacted) | Quarterly |
| Post-incident review | Meeting minutes | Sample post-mortem documentation | Per incident |

#### POL-007: Business Continuity Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| BCP/DRP | Policy | Approved continuity/recovery plans | Annual |
| BIA | Export | Business impact analysis | Annual |
| DR testing | Meeting minutes | DR test results and lessons learned | Annual |
| Backup verification | Logs | Backup completion and restore test logs | Monthly |

#### POL-008: Change Management Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Change process | Policy | Approved change management procedure | Annual |
| CAB meetings | Meeting minutes | Change advisory board records | Weekly/Bi-weekly |
| Change tickets | Tickets | Sample change requests with approvals | Quarterly |
| Emergency changes | Tickets | Emergency change records with post-approval | Quarterly |

#### POL-009: Vendor Management Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Vendor inventory | Export | Critical vendor list with risk ratings | Annual |
| Vendor assessments | Export | Security assessment questionnaires | Annual per vendor |
| Vendor contracts | Attestation | Contract security clause confirmation | Per contract |
| SOC reports | Attestation | Vendor SOC 2 reports | Annual per vendor |

#### POL-010: Encryption Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Encryption at rest | Screenshot | Storage encryption configuration | Quarterly |
| Encryption in transit | Screenshot | TLS configuration, certificate status | Quarterly |
| Key management | Screenshot | KMS configuration, key rotation | Quarterly |
| Certificate inventory | Export | Certificate inventory with expiration | Monthly |

#### POL-011: Logging and Monitoring Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Log configuration | Screenshot | Logging enablement across systems | Quarterly |
| Log retention | Screenshot | Log retention configuration | Annual |
| SIEM alerts | Screenshot | Alert rules configuration | Quarterly |
| Log review | Logs | Sample log review records | Monthly |
| Alert response | Tickets | Sample alert response tickets | Monthly |

#### POL-012: Physical Security Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Data center security | Attestation | Cloud provider SOC reports | Annual |
| Office access | Export | Badge access logs (sample) | Quarterly |
| Visitor management | Logs | Visitor log samples | Quarterly |

#### POL-013: HR Security Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Background checks | Attestation | HR attestation of check completion | Annual |
| Security training | Export | Training completion report | Annual |
| Onboarding checklist | Export | Sample completed checklists | Quarterly |
| Termination checklist | Export | Sample completed checklists | Quarterly |

#### POL-014: Risk Management Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Risk assessment | Export | Risk register with ratings | Annual |
| Risk committee | Meeting minutes | Risk committee meeting records | Quarterly |
| Risk treatment | Tickets | Risk treatment plan tickets | Quarterly |
| Risk acceptance | Attestation | Signed risk acceptance forms | As needed |

#### POL-015: Privacy Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Privacy notice | Screenshot | Published privacy notice | Annual |
| DSAR process | Tickets | Sample data subject request handling | Quarterly |
| Consent management | Screenshot | Consent mechanism configuration | Annual |
| Privacy impact | Export | Privacy impact assessments | Per project |

#### POL-016: SDLC Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Secure coding standards | Policy | Approved coding standards | Annual |
| Code review | Screenshots | Sample PR reviews with security checks | Quarterly |
| Security testing | Export | SAST/DAST scan results | Per release |
| Vulnerability management | Export | Vulnerability tracking and remediation | Monthly |
| Deployment approval | Tickets | Sample deployment approvals | Quarterly |

#### POL-017: Asset Management Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Asset inventory | Export | Hardware/software asset inventory | Quarterly |
| Asset ownership | Export | Asset ownership assignments | Annual |
| Asset disposal | Logs | Asset disposal/destruction records | Quarterly |
| License management | Export | Software license compliance report | Annual |

#### POL-018: Network Security Policy

| Control | Evidence Type | Description | Frequency |
|---------|---------------|-------------|-----------|
| Network diagram | Export | Current network architecture diagram | Annual |
| Firewall rules | Screenshot | Firewall rule configuration | Quarterly |
| Segmentation | Screenshot | Network segmentation evidence | Annual |
| Vulnerability scans | Export | Network vulnerability scan results | Monthly |
| Penetration testing | Export | Annual penetration test report | Annual |

---

## 4. Collection Frequency

### 4.1 Collection Schedule

| Frequency | Description | Collection Window |
|-----------|-------------|-------------------|
| **Continuous** | Automated collection via APIs/integrations | Real-time |
| **Monthly** | Recurring operational evidence | First week of month |
| **Quarterly** | Periodic compliance evidence | First 2 weeks of quarter |
| **Annual** | Strategic and governance evidence | Q4 (audit prep) |
| **Event-driven** | Triggered by specific events | Within 5 business days |

### 4.2 Annual Evidence Calendar

| Month | Primary Collection Activities |
|-------|------------------------------|
| **January** | Q4 evidence finalization, Annual acknowledgments begin |
| **February** | Access reviews (Q1), Vendor assessment planning |
| **March** | Q1 evidence collection, Training campaign |
| **April** | SOC 2 audit prep begins, DR test planning |
| **May** | Access reviews (Q2), Penetration test |
| **June** | Q2 evidence collection, Policy review cycle |
| **July** | SOC 2 audit (typical), BIA update |
| **August** | Access reviews (Q3), Vendor assessments |
| **September** | Q3 evidence collection, Risk assessment |
| **October** | Annual policy review, Audit findings remediation |
| **November** | Access reviews (Q4), Training refresh |
| **December** | Q4 evidence collection prep, Year-end attestations |

### 4.3 Continuous Collection (Automated)

The following evidence is collected automatically:

| System | Evidence | Collection Method |
|--------|----------|-------------------|
| AWS CloudTrail | API activity logs | S3 export, SIEM ingestion |
| GitHub | Code review records | API integration |
| Okta | Authentication logs | SIEM integration |
| ServiceNow | Ticket data | API export |
| GRC Platform | Acknowledgments | Database export |
| Vulnerability scanner | Scan results | Scheduled export |

---

## 5. Storage and Retention Requirements

### 5.1 Evidence Storage Locations

| Evidence Type | Primary Storage | Backup Storage | Access Control |
|---------------|-----------------|----------------|----------------|
| Screenshots | GRC Platform | S3 (encrypted) | Compliance team + Auditors |
| System logs | SIEM (Loki) | S3 Glacier | Security team + Auditors |
| Exports | GRC Platform | S3 (encrypted) | Control owners + Compliance |
| Attestations | Document management | S3 (encrypted) | Compliance team + Legal |
| Policies | Policy repository | Git + S3 | All employees (read) |
| Training records | LMS | S3 (encrypted) | HR + Compliance |
| Meeting minutes | Document management | S3 (encrypted) | Meeting participants |
| Tickets | ITSM (ServiceNow) | Database backup | IT + Compliance |

### 5.2 Retention Periods

| Evidence Category | Retention Period | Regulatory Basis |
|-------------------|------------------|------------------|
| Audit evidence | 7 years | SOC 2, ISO 27001 |
| Access logs | 1 year (hot) + 6 years (archive) | SOC 2, GDPR |
| Security incidents | 7 years | Legal, regulatory |
| Policy versions | 7 years | SOC 2, ISO 27001 |
| Training records | 7 years | Employment law |
| Vendor assessments | Contract term + 7 years | Contract, regulatory |
| Financial records | 7 years | SOX, tax |
| HR records | Employment + 7 years | Employment law |

### 5.3 Evidence Naming Convention

```
[YEAR]-[QUARTER/MONTH]-[POLICY_ID]-[CONTROL_ID]-[EVIDENCE_TYPE]-[DESCRIPTION].[EXT]

Examples:
2026-Q1-POL003-AC01-screenshot-aws-iam-policy.png
2026-03-POL011-LM03-export-siem-alert-rules.csv
2026-annual-POL006-IR02-minutes-tabletop-exercise.pdf
```

### 5.4 Evidence Metadata

Each evidence artifact includes:

```json
{
  "evidence_id": "EVD-2026-001234",
  "filename": "2026-Q1-POL003-AC01-screenshot-aws-iam-policy.png",
  "policy_id": "POL-003",
  "control_id": "AC-01",
  "description": "AWS IAM password policy configuration",
  "evidence_type": "screenshot",
  "collection_date": "2026-03-15T10:30:00Z",
  "collected_by": "jane.smith@greenlang.io",
  "system_source": "AWS Console",
  "audit_period": "2026-Q1",
  "hash_sha256": "a1b2c3d4e5...",
  "classification": "internal",
  "retention_until": "2033-03-15"
}
```

---

## 6. Auditor Access Procedures

### 6.1 Auditor Types

| Auditor Type | Access Level | Duration | Approval |
|--------------|--------------|----------|----------|
| **SOC 2 Auditor** | Full evidence access | Engagement period | CISO |
| **ISO 27001 Auditor** | Full evidence access | Engagement period | CISO |
| **Regulatory Examiner** | Full evidence + system access | Examination period | Legal + CISO |
| **Customer Auditor** | Scoped evidence per contract | Per contract | Legal + CISO |
| **Internal Auditor** | Full evidence access | Continuous | CAO |

### 6.2 Auditor Onboarding

**Step 1: Engagement Confirmation**
- Receive engagement letter from audit firm
- Verify auditor credentials and firm reputation
- Execute NDA if not covered by engagement letter

**Step 2: Access Provisioning**
- Create auditor accounts in GRC platform (read-only)
- Provision VPN access if remote (time-limited)
- Provide secure file sharing access (auditor portal)
- Issue visitor badges for on-site visits

**Step 3: Orientation**
- Introduce audit liaison (Compliance team member)
- Provide evidence portal walkthrough
- Share evidence collection schedule
- Establish communication protocols

### 6.3 Evidence Request Handling

**Request Process:**

1. **Receipt**: Auditor submits evidence request (email or portal)
2. **Logging**: Compliance team logs request in tracking system
3. **Assignment**: Request assigned to appropriate control owner
4. **Collection**: Control owner collects/prepares evidence
5. **Review**: Compliance team reviews for completeness, sensitivity
6. **Delivery**: Evidence uploaded to auditor portal
7. **Confirmation**: Auditor confirms receipt

**SLA by Request Priority:**

| Priority | Description | SLA |
|----------|-------------|-----|
| **Critical** | Blocking audit progress | 4 hours |
| **High** | Needed this week | 24 hours |
| **Normal** | Standard request | 48 hours |
| **Low** | Nice to have | 5 business days |

### 6.4 Evidence Portal

The auditor evidence portal provides:

| Feature | Description |
|---------|-------------|
| **Secure upload** | Encrypted file transfer |
| **Folder structure** | Organized by policy/control |
| **Version control** | Track evidence updates |
| **Request tracking** | Monitor request status |
| **Download logging** | Audit trail of access |
| **Expiration** | Auto-delete after engagement |

**Portal Access:**
- URL: https://audit.greenlang.io
- Authentication: SSO with MFA
- Authorization: Role-based (read-only for auditors)

### 6.5 Auditor Off-boarding

At engagement conclusion:

- [ ] Disable auditor accounts within 24 hours
- [ ] Revoke VPN and portal access
- [ ] Collect visitor badges
- [ ] Archive engagement folder (read-only)
- [ ] Document any open items or findings
- [ ] Schedule remediation follow-up if needed

---

## 7. Evidence Package Preparation

### 7.1 SOC 2 Type II Evidence Package

**Structure:**

```
SOC2-2026/
├── 01-Security/
│   ├── POL-001-InfoSec/
│   │   ├── policy-approved.pdf
│   │   ├── acknowledgments-export.csv
│   │   └── review-minutes.pdf
│   ├── POL-003-AccessControl/
│   │   ├── access-provisioning-samples/
│   │   ├── access-review-Q1.xlsx
│   │   ├── access-review-Q2.xlsx
│   │   ├── privileged-users.csv
│   │   └── termination-samples/
│   └── [additional policies...]
├── 02-Availability/
│   ├── POL-007-BCP/
│   │   ├── bcp-plan.pdf
│   │   ├── dr-test-results.pdf
│   │   └── backup-logs/
│   └── [additional evidence...]
├── 03-ProcessingIntegrity/
│   └── [evidence...]
├── 04-Confidentiality/
│   └── [evidence...]
├── 05-Privacy/
│   └── [evidence...]
└── 00-Index/
    ├── evidence-index.xlsx
    ├── control-mapping.xlsx
    └── population-files/
```

**Control Mapping Document:**

| SOC 2 Criteria | Control Description | Evidence Reference | Collection Date |
|----------------|--------------------|--------------------|-----------------|
| CC6.1 | Access provisioning requires approval | POL-003/access-provisioning-samples/ | 2026-03-15 |
| CC6.2 | Access is removed upon termination | POL-003/termination-samples/ | 2026-03-15 |
| CC6.3 | Access is reviewed quarterly | POL-003/access-review-Q1.xlsx | 2026-04-05 |

### 7.2 ISO 27001:2022 Evidence Package

**Structure:**

```
ISO27001-2026/
├── A5-OrganizationalControls/
│   ├── A5.1-InfoSecPolicies/
│   ├── A5.2-InfoSecRoles/
│   └── [A5.3 through A5.37...]
├── A6-PeopleControls/
│   ├── A6.1-Screening/
│   ├── A6.2-Employment/
│   └── [A6.3 through A6.8...]
├── A7-PhysicalControls/
│   └── [A7.1 through A7.14...]
├── A8-TechnologicalControls/
│   └── [A8.1 through A8.34...]
├── ISMS-Documentation/
│   ├── scope-statement.pdf
│   ├── risk-assessment.xlsx
│   ├── soa-statement-of-applicability.xlsx
│   └── management-review-minutes.pdf
└── 00-Index/
    ├── evidence-index.xlsx
    └── control-mapping.xlsx
```

### 7.3 Population and Sampling

For controls requiring sample testing:

**Population File Requirements:**

| Field | Description |
|-------|-------------|
| Record ID | Unique identifier |
| Date | When the event occurred |
| Description | Brief description |
| Owner | Responsible person |
| Outcome | Result (approved, denied, completed) |

**Example Population (Access Requests):**

| Request ID | Request Date | Requestor | System | Approver | Approval Date | Status |
|------------|--------------|-----------|--------|----------|---------------|--------|
| AR-001 | 2026-01-05 | J. Smith | AWS | M. Johnson | 2026-01-06 | Approved |
| AR-002 | 2026-01-07 | A. Lee | GitHub | T. Chen | 2026-01-07 | Approved |
| AR-003 | 2026-01-08 | B. Wilson | Salesforce | K. Brown | 2026-01-09 | Denied |

**Sampling Guidelines:**

| Population Size | Sample Size | Selection Method |
|-----------------|-------------|------------------|
| 1-10 | All | N/A |
| 11-50 | 10 | Random |
| 51-250 | 25 | Random + judgmental |
| 251+ | 25-60 | Statistical + judgmental |

---

## 8. Quality Assurance

### 8.1 Evidence Quality Checklist

Before submitting evidence to auditors:

- [ ] **Complete**: All required information is present
- [ ] **Current**: Evidence is from the audit period
- [ ] **Accurate**: Information reflects actual state
- [ ] **Relevant**: Evidence addresses the control
- [ ] **Readable**: Screenshots are clear, exports are formatted
- [ ] **Redacted**: Sensitive data appropriately masked
- [ ] **Named**: Follows naming convention
- [ ] **Metadata**: Includes required metadata fields

### 8.2 Common Evidence Issues

| Issue | Impact | Prevention |
|-------|--------|------------|
| Outdated evidence | Audit finding | Use collection calendar |
| Missing approvals | Control failure | Include full workflow |
| Unclear screenshots | Auditor clarification | Annotate important areas |
| Incomplete populations | Sampling issues | Validate before submission |
| Sensitive data exposed | Privacy violation | Redaction review |
| Broken links | Delays | Verify all references |

### 8.3 Evidence Review Workflow

```
Control Owner         Compliance Team         Audit Liaison
    |                       |                       |
    |-- Collect evidence -->|                       |
    |                       |-- Quality review -->  |
    |<-- Feedback (if issues)                       |
    |-- Correct & resubmit->|                       |
    |                       |-- Approve ----------->|
    |                       |                       |-- Upload to portal
    |                       |                       |-- Notify auditor
```

---

## 9. Tools and Automation

### 9.1 Evidence Collection Tools

| Tool | Purpose | Integration |
|------|---------|-------------|
| **GRC Platform** | Central evidence repository | API, manual upload |
| **AWS Config** | AWS configuration evidence | S3 export |
| **GitHub API** | Code review evidence | REST API |
| **Okta Admin** | Access and authentication | API export |
| **ServiceNow** | Ticket and change evidence | REST API |
| **Loki** | Log evidence | Query API |
| **Snyk** | Vulnerability evidence | API export |

### 9.2 Automated Collection Scripts

**Example: AWS IAM Policy Export**

```python
#!/usr/bin/env python3
"""
Automated evidence collection for AWS IAM password policy.
Outputs: Screenshot-equivalent JSON + metadata
"""

import boto3
import json
from datetime import datetime

def collect_iam_password_policy():
    iam = boto3.client('iam')

    try:
        policy = iam.get_account_password_policy()

        evidence = {
            "evidence_id": f"EVD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "policy_id": "POL-003",
            "control_id": "AC-07",
            "description": "AWS IAM Account Password Policy",
            "collection_date": datetime.now().isoformat(),
            "system_source": "AWS IAM",
            "data": policy['PasswordPolicy']
        }

        filename = f"aws-iam-password-policy-{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(evidence, f, indent=2, default=str)

        return filename

    except Exception as e:
        print(f"Error collecting evidence: {e}")
        raise

if __name__ == "__main__":
    collect_iam_password_policy()
```

### 9.3 Collection Automation Schedule

```yaml
# Evidence collection automation (scheduled via GitHub Actions)
evidence_collection:
  daily:
    - backup_completion_check
    - certificate_expiration_scan
    - vulnerability_scan_status

  weekly:
    - access_log_export
    - change_ticket_summary
    - alert_response_metrics

  monthly:
    - user_access_report
    - terminated_user_audit
    - vendor_assessment_status

  quarterly:
    - access_review_completion
    - policy_acknowledgment_status
    - training_completion_report
    - risk_register_export
```

---

## 10. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Compliance Team** | Coordinate collection; maintain repository; prepare packages; support auditors |
| **Control Owners** | Collect evidence for assigned controls; ensure quality; meet deadlines |
| **IT/Security Teams** | Provide technical evidence; support automated collection |
| **Audit Liaison** | Primary auditor contact; manage requests; resolve issues |
| **Legal** | Review sensitive evidence; advise on regulatory requirements |
| **CISO** | Approve auditor access; oversee evidence program; escalation point |

---

## 11. Related Documents

- [Policy Index](../README.md)
- [Policy Management Guide](../POLICY_MANAGEMENT.md)
- [Acknowledgment Process](../acknowledgments/ACKNOWLEDGMENT_PROCESS.md)
- [SOC 2 TSC Mapping](../compliance-mapping/SOC2-TSC-Mapping.md)
- [ISO 27001 Controls Mapping](../compliance-mapping/ISO27001-Controls-Mapping.md)

---

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Compliance Team | Initial evidence collection guide |

---

*This document is confidential and intended for internal use only. Unauthorized distribution is prohibited.*
