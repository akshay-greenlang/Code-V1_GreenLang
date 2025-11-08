# SOC 2 Controls Mapping for GreenLang

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Classification:** Internal - Compliance Documentation
**Owner:** Security & Compliance Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [SOC 2 Trust Service Criteria Overview](#soc-2-trust-service-criteria-overview)
3. [Common Criteria (CC) Controls Mapping](#common-criteria-cc-controls-mapping)
4. [Evidence Collection Procedures](#evidence-collection-procedures)
5. [Audit Preparation Guide](#audit-preparation-guide)
6. [Control Implementation Details](#control-implementation-details)
7. [Testing Procedures](#testing-procedures)
8. [Continuous Monitoring Plan](#continuous-monitoring-plan)
9. [Appendices](#appendices)

---

## Executive Summary

### Purpose
This document maps GreenLang's security controls to SOC 2 Trust Service Criteria, providing comprehensive documentation for SOC 2 Type II audit readiness. It demonstrates how GreenLang's security architecture meets and exceeds SOC 2 requirements.

### Scope
- **Service:** GreenLang Platform (Core, CLI, API, Agents)
- **Audit Type:** SOC 2 Type II
- **Trust Service Criteria:** Security, Availability, Processing Integrity, Confidentiality, Privacy
- **Audit Period:** 12 months
- **Environment:** Production, Staging, Development

### Key Findings
- **100% Coverage:** All applicable SOC 2 controls are implemented
- **Defense in Depth:** Multiple layers of security controls
- **Automated Compliance:** Continuous monitoring and evidence collection
- **Audit Ready:** Comprehensive documentation and evidence trails

---

## SOC 2 Trust Service Criteria Overview

### Trust Service Principles

#### 1. Security (CC)
The system is protected against unauthorized access (both physical and logical).

#### 2. Availability (A)
The system is available for operation and use as committed or agreed.

#### 3. Processing Integrity (PI)
System processing is complete, valid, accurate, timely, and authorized.

#### 4. Confidentiality (C)
Information designated as confidential is protected as committed or agreed.

#### 5. Privacy (P)
Personal information is collected, used, retained, disclosed, and disposed of in conformity with commitments.

---

## Common Criteria (CC) Controls Mapping

### CC1: Control Environment

#### CC1.1: Organization Demonstrates Commitment to Integrity and Ethical Values

**Control Statement:**
The entity demonstrates a commitment to integrity and ethical values.

**GreenLang Implementation:**

1. **Code of Conduct**
   - Location: `docs/CODE_OF_CONDUCT.md`
   - Annual acknowledgment required
   - Violations tracked and reported

2. **Security Policy Framework**
   - Default-deny security model
   - Zero-trust architecture
   - Mandatory security training

3. **Ethical AI Guidelines**
   - Transparent agent decision-making
   - Bias detection and mitigation
   - Environmental impact transparency

**Evidence Artifacts:**
- Code of Conduct acknowledgment logs
- Security policy documentation
- Training completion records
- Ethics committee meeting minutes

**Testing Frequency:** Annual review, continuous monitoring

---

#### CC1.2: Board of Directors Demonstrates Independence

**Control Statement:**
The board of directors demonstrates independence from management and exercises oversight.

**GreenLang Implementation:**

1. **Governance Structure**
   - Independent security committee
   - Quarterly security reviews
   - External audit oversight

2. **Risk Management Oversight**
   - Monthly risk assessment reviews
   - Incident response oversight
   - Compliance monitoring

**Evidence Artifacts:**
- Board meeting minutes
- Security committee charters
- Risk assessment reports
- Audit committee reports

**Testing Frequency:** Quarterly

---

#### CC1.3: Management Establishes Structure, Authority, and Responsibility

**Control Statement:**
Management establishes, with board oversight, structures, reporting lines, and appropriate authorities and responsibilities.

**GreenLang Implementation:**

1. **Organizational Structure**
   ```
   CEO
   ├── CTO
   │   ├── VP Engineering
   │   │   ├── Development Teams
   │   │   └── DevOps Team
   │   └── VP Security
   │       ├── Security Engineering
   │       ├── Compliance Team
   │       └── Incident Response
   └── COO
       ├── IT Operations
       └── Business Continuity
   ```

2. **Role-Based Access Control (RBAC)**
   - Defined roles and responsibilities
   - Least privilege principle
   - Separation of duties

3. **Delegation of Authority**
   - Security decision matrix
   - Escalation procedures
   - Change approval workflows

**Evidence Artifacts:**
- Organization charts
- Role definition documents
- RACI matrices
- Approval workflow logs

**Testing Frequency:** Annual review, changes documented

---

#### CC1.4: Organization Demonstrates Commitment to Competence

**Control Statement:**
The entity demonstrates a commitment to attract, develop, and retain competent individuals.

**GreenLang Implementation:**

1. **Security Training Program**
   - New hire security orientation
   - Annual security awareness training
   - Role-specific security training
   - Phishing simulation exercises

2. **Certification Requirements**
   - Security team certifications (CISSP, CEH, OSCP)
   - Ongoing professional development
   - Conference attendance budget

3. **Knowledge Management**
   - Security documentation wiki
   - Runbooks and playbooks
   - Post-incident reviews
   - Lessons learned database

**Evidence Artifacts:**
- Training completion records
- Certification tracking
- Training materials
- Skills assessment results

**Testing Frequency:** Quarterly training completion review

---

#### CC1.5: Organization Holds Individuals Accountable

**Control Statement:**
The entity holds individuals accountable for their internal control responsibilities.

**GreenLang Implementation:**

1. **Performance Reviews**
   - Security objectives in all roles
   - Incident response participation
   - Compliance adherence metrics

2. **Accountability Mechanisms**
   - Audit log attribution
   - Code review requirements
   - Change management approval
   - Incident ownership

3. **Disciplinary Procedures**
   - Policy violation consequences
   - Progressive discipline
   - Termination procedures

**Evidence Artifacts:**
- Performance review templates
- Audit logs with user attribution
- Incident response assignments
- Disciplinary action records (anonymized)

**Testing Frequency:** Annual performance reviews, continuous monitoring

---

### CC2: Communication and Information

#### CC2.1: Organization Obtains or Generates Relevant Information

**Control Statement:**
The entity obtains or generates and uses relevant, quality information to support internal control.

**GreenLang Implementation:**

1. **Security Information Sources**
   - Threat intelligence feeds
   - Vulnerability databases (NVD, CVE)
   - Security advisories (GitHub, npm)
   - Industry best practices (OWASP, SANS)

2. **Internal Information Systems**
   - SIEM (Security Information and Event Management)
   - Log aggregation (ELK stack)
   - Metrics and monitoring (Prometheus, Grafana)
   - Audit trail database

3. **Quality Controls**
   - Data validation procedures
   - Source verification
   - Information accuracy reviews
   - Timeliness monitoring

**Evidence Artifacts:**
- Threat intelligence reports
- Vulnerability scan results
- SIEM dashboards
- Data quality metrics

**Testing Frequency:** Continuous collection, weekly review

---

#### CC2.2: Organization Communicates Information Internally

**Control Statement:**
The entity internally communicates information, including objectives and responsibilities for internal control.

**GreenLang Implementation:**

1. **Communication Channels**
   - Security Slack channel
   - Weekly security bulletins
   - Monthly all-hands security updates
   - Incident notifications

2. **Policy Distribution**
   - Centralized policy repository
   - Version control for policies
   - Change notifications
   - Acknowledgment tracking

3. **Training and Awareness**
   - Security awareness campaigns
   - Phishing simulations
   - Security tips and best practices
   - Incident lessons learned

**Evidence Artifacts:**
- Communication logs
- Policy acknowledgment records
- Training attendance sheets
- Newsletter archives

**Testing Frequency:** Ongoing, monthly review

---

#### CC2.3: Organization Communicates with External Parties

**Control Statement:**
The entity communicates with external parties regarding matters affecting internal control.

**GreenLang Implementation:**

1. **Customer Communications**
   - Security disclosure policy
   - Incident notification procedures
   - Status page (status.greenlang.io)
   - Security bulletins

2. **Regulatory Communications**
   - Compliance reporting
   - Breach notifications (GDPR, state laws)
   - Audit responses
   - Regulatory inquiries

3. **Vendor Management**
   - Third-party risk assessments
   - SLA monitoring
   - Security questionnaires
   - Incident coordination

**Evidence Artifacts:**
- Customer notification templates
- Incident communication logs
- Regulatory filing records
- Vendor assessment reports

**Testing Frequency:** Ongoing, quarterly review

---

### CC3: Risk Assessment

#### CC3.1: Organization Specifies Objectives

**Control Statement:**
The entity specifies objectives with sufficient clarity to enable identification and assessment of risks.

**GreenLang Implementation:**

1. **Security Objectives**
   - Maintain confidentiality of customer data
   - Ensure availability of platform services (99.9% uptime)
   - Protect integrity of carbon calculations
   - Achieve compliance with SOC 2, ISO 27001, GDPR, HIPAA
   - Respond to security incidents within 1 hour

2. **Compliance Objectives**
   - Pass SOC 2 Type II audit
   - Maintain ISO 27001 certification
   - Achieve GDPR compliance
   - Meet HIPAA technical safeguards
   - Zero critical vulnerabilities in production

3. **Operational Objectives**
   - Deploy security patches within 48 hours
   - Conduct quarterly penetration tests
   - Maintain 100% encryption for data at rest and in transit
   - Complete annual disaster recovery testing

**Evidence Artifacts:**
- Strategic planning documents
- OKR (Objectives and Key Results) tracking
- Security roadmap
- Compliance calendars

**Testing Frequency:** Quarterly objective review

---

#### CC3.2: Organization Identifies and Analyzes Risk

**Control Statement:**
The entity identifies risks to the achievement of its objectives and analyzes risks as a basis for determining how to manage them.

**GreenLang Implementation:**

1. **Risk Identification Process**
   - Quarterly risk assessments
   - Threat modeling for new features
   - Vulnerability scanning (daily)
   - Penetration testing (quarterly)
   - Third-party security audits

2. **Risk Analysis Framework**
   - Likelihood assessment (1-5 scale)
   - Impact assessment (1-5 scale)
   - Risk scoring matrix
   - Risk categorization (High/Medium/Low)

3. **Risk Register**
   | Risk ID | Description | Likelihood | Impact | Score | Owner | Mitigation |
   |---------|-------------|------------|--------|-------|-------|------------|
   | R-001 | Data breach via SQL injection | 2 | 5 | 10 | Security Team | Input validation, prepared statements |
   | R-002 | DDoS attack on API | 3 | 4 | 12 | DevOps | Rate limiting, WAF, CDN |
   | R-003 | Insider threat | 2 | 4 | 8 | HR/Security | Background checks, RBAC, audit logs |
   | R-004 | Third-party vendor breach | 3 | 4 | 12 | Procurement | Vendor assessments, SLAs |
   | R-005 | Ransomware attack | 2 | 5 | 10 | Security Team | EDR, backups, training |

**Evidence Artifacts:**
- Risk register
- Risk assessment reports
- Threat models
- Vulnerability scan results
- Penetration test reports

**Testing Frequency:** Quarterly comprehensive review, continuous monitoring

---

#### CC3.3: Organization Assesses Fraud Risk

**Control Statement:**
The entity considers the potential for fraud in assessing risks to the achievement of objectives.

**GreenLang Implementation:**

1. **Fraud Risk Scenarios**
   - Unauthorized access to customer data
   - Manipulation of carbon calculations
   - API key theft and abuse
   - Insider trading on carbon credits
   - Billing fraud
   - Identity theft

2. **Fraud Prevention Controls**
   - Multi-factor authentication (MFA)
   - Anomaly detection algorithms
   - Transaction monitoring
   - Separation of duties
   - Code review requirements
   - Audit logging

3. **Fraud Detection Mechanisms**
   - Automated alerting on suspicious activity
   - Regular audit log reviews
   - User behavior analytics
   - Financial reconciliation
   - Whistleblower hotline

**Evidence Artifacts:**
- Fraud risk assessment
- Anomaly detection alerts
- Investigation reports
- Control testing results

**Testing Frequency:** Annual fraud risk assessment, continuous monitoring

---

#### CC3.4: Organization Assesses Changes That Could Impact Internal Control

**Control Statement:**
The entity identifies and assesses changes that could significantly impact the system of internal control.

**GreenLang Implementation:**

1. **Change Categories Monitored**
   - Organizational changes (mergers, acquisitions, restructuring)
   - Technology changes (new platforms, migrations)
   - Regulatory changes (new compliance requirements)
   - Business model changes (new products, markets)
   - Threat landscape changes (new attack vectors)

2. **Change Assessment Process**
   ```
   Change Identified
   ↓
   Impact Analysis
   ↓
   Risk Assessment
   ↓
   Control Adjustment
   ↓
   Testing & Validation
   ↓
   Documentation & Training
   ```

3. **Recent Changes Assessed**
   - Migration to microservices architecture
   - Implementation of AI agents
   - GDPR compliance requirements
   - Remote work policies (COVID-19)
   - Cloud infrastructure expansion

**Evidence Artifacts:**
- Change impact assessments
- Risk reassessment reports
- Control update documentation
- Testing results

**Testing Frequency:** Ongoing as changes occur, quarterly review

---

### CC4: Monitoring Activities

#### CC4.1: Organization Selects, Develops, and Performs Ongoing and/or Separate Evaluations

**Control Statement:**
The entity selects, develops, and performs ongoing and/or separate evaluations to ascertain whether internal control components are present and functioning.

**GreenLang Implementation:**

1. **Ongoing Monitoring**
   - Real-time security monitoring (SIEM)
   - Automated vulnerability scanning (daily)
   - Performance monitoring (24/7)
   - Audit log review (automated + weekly manual)
   - Compliance dashboards

2. **Separate Evaluations**
   - Quarterly internal control testing
   - Annual internal audit
   - Quarterly penetration testing
   - Annual SOC 2 audit
   - Periodic ISO 27001 surveillance audits

3. **Monitoring Tools**
   - SIEM: Splunk / ELK Stack
   - Vulnerability Scanner: Nessus, OWASP ZAP
   - Application Security: Snyk, Bandit
   - Infrastructure: Prometheus, Grafana
   - Compliance: Custom dashboards

**Evidence Artifacts:**
- Monitoring dashboards
- Scan reports
- Test results
- Audit reports
- Control deficiency logs

**Testing Frequency:** Continuous monitoring, quarterly comprehensive evaluation

---

#### CC4.2: Organization Evaluates and Communicates Deficiencies

**Control Statement:**
The entity evaluates and communicates internal control deficiencies in a timely manner to responsible parties for corrective action.

**GreenLang Implementation:**

1. **Deficiency Classification**
   - **Critical:** Immediate remediation required (24 hours)
   - **High:** Remediation within 1 week
   - **Medium:** Remediation within 30 days
   - **Low:** Remediation within 90 days

2. **Communication Procedures**
   - Critical: Immediate notification to CISO, CTO, CEO
   - High: Email notification to security team + management
   - Medium: Weekly security meeting discussion
   - Low: Monthly security review

3. **Remediation Tracking**
   - Deficiency tracking system (Jira)
   - Assigned owners and due dates
   - Progress monitoring
   - Verification of remediation
   - Root cause analysis

**Evidence Artifacts:**
- Deficiency logs
- Communication records
- Remediation plans
- Verification test results
- Root cause analysis reports

**Testing Frequency:** Continuous deficiency monitoring, monthly remediation review

---

### CC5: Control Activities

#### CC5.1: Organization Selects and Develops Control Activities

**Control Statement:**
The entity selects and develops control activities that contribute to the mitigation of risks to acceptable levels.

**GreenLang Implementation:**

1. **Preventive Controls**
   - Input validation (XSS, SQL injection prevention)
   - Authentication and authorization
   - Encryption (data at rest and in transit)
   - Network segmentation
   - Firewall rules
   - Secure coding standards

2. **Detective Controls**
   - Intrusion detection systems (IDS)
   - Log monitoring and alerting
   - Vulnerability scanning
   - File integrity monitoring
   - Anomaly detection

3. **Corrective Controls**
   - Incident response procedures
   - Patch management
   - Backup and recovery
   - Disaster recovery plan
   - Business continuity plan

**GreenLang-Specific Controls:**

| Control ID | Control Name | Type | Description | Risk Mitigated |
|------------|--------------|------|-------------|----------------|
| GL-AC-001 | Multi-Factor Authentication | Preventive | MFA required for all users | Unauthorized access |
| GL-AC-002 | Role-Based Access Control | Preventive | Least privilege access | Privilege escalation |
| GL-AC-003 | Session Timeout | Preventive | 30-minute idle timeout | Session hijacking |
| GL-DC-001 | API Rate Limiting | Preventive/Detective | 1000 requests/hour | API abuse, DDoS |
| GL-DC-002 | Audit Logging | Detective | All actions logged | Unauthorized changes |
| GL-DC-003 | Intrusion Detection | Detective | Real-time threat detection | Security breaches |
| GL-ENC-001 | Data Encryption at Rest | Preventive | AES-256 encryption | Data theft |
| GL-ENC-002 | TLS 1.3 for Transit | Preventive | All network traffic encrypted | Man-in-the-middle |
| GL-IM-001 | Input Validation | Preventive | All inputs sanitized | Injection attacks |
| GL-VM-001 | Vulnerability Scanning | Detective | Daily automated scans | Exploitable vulnerabilities |
| GL-PM-001 | Patch Management | Corrective | Patches within 48 hours | Known vulnerabilities |
| GL-BC-001 | Backup & Recovery | Corrective | Daily backups, tested quarterly | Data loss |

**Evidence Artifacts:**
- Control documentation
- Configuration files
- Implementation guides
- Control testing results

**Testing Frequency:** Controls tested quarterly, configuration reviewed monthly

---

#### CC5.2: Organization Selects and Develops General Controls over Technology

**Control Statement:**
The entity selects and develops general control activities over technology to support the achievement of objectives.

**GreenLang Implementation:**

1. **Infrastructure Controls**
   - **Network Security**
     - Firewall configuration (deny-by-default)
     - Network segmentation (DMZ, internal, data tier)
     - VPN for remote access
     - DDoS protection (Cloudflare)

   - **Server Hardening**
     - Minimal services installed
     - Security patches automated
     - File integrity monitoring
     - Anti-malware software
     - Secure baseline configurations

2. **Application Security Controls**
   - **Secure Development Lifecycle (SDL)**
     - Security requirements gathering
     - Threat modeling
     - Secure code review
     - Static application security testing (SAST)
     - Dynamic application security testing (DAST)
     - Dependency vulnerability scanning

   - **Code Security**
     ```python
     # Input validation example
     from greenlang.security import validate_username, XSSValidator

     def create_user(username: str, email: str):
         # Always validate inputs
         safe_username = validate_username(username)
         safe_email = XSSValidator.sanitize_html(email)

         # Use parameterized queries
         db.execute(
             "INSERT INTO users (username, email) VALUES (?, ?)",
             (safe_username, safe_email)
         )
     ```

3. **Database Security Controls**
   - Encrypted connections (TLS)
   - Encrypted at rest (AES-256)
   - Parameterized queries (SQL injection prevention)
   - Least privilege database accounts
   - Regular backups (daily, tested quarterly)
   - Database activity monitoring

4. **API Security Controls**
   - API key authentication
   - Rate limiting (1000 req/hour per user)
   - Input validation
   - Output encoding
   - HTTPS only
   - CORS restrictions
   - API versioning

5. **Cloud Security Controls**
   - Identity and Access Management (IAM)
   - Security groups (firewall rules)
   - Encryption key management (AWS KMS)
   - VPC isolation
   - CloudTrail logging
   - GuardDuty threat detection

**Evidence Artifacts:**
- Infrastructure diagrams
- Configuration baselines
- Security scan results
- Code review records
- Penetration test reports

**Testing Frequency:** Quarterly security testing, continuous monitoring

---

#### CC5.3: Organization Deploys Control Activities Through Policies and Procedures

**Control Statement:**
The entity deploys control activities through policies that establish what is expected and procedures that put policies into action.

**GreenLang Implementation:**

1. **Security Policies**
   - Acceptable Use Policy
   - Access Control Policy
   - Encryption Policy
   - Incident Response Policy
   - Change Management Policy
   - Data Classification Policy
   - Password Policy
   - Remote Access Policy
   - BYOD Policy
   - Third-Party Risk Management Policy

2. **Operational Procedures**
   - User provisioning/deprovisioning procedure
   - Access review procedure
   - Patch management procedure
   - Backup and restore procedure
   - Incident response procedure
   - Change management procedure
   - Disaster recovery procedure

3. **Policy Enforcement Mechanisms**
   ```yaml
   # GreenLang Policy Example (OPA)
   package greenlang.access

   # Default deny
   default allow = false

   # Allow if user is authenticated AND authorized
   allow {
       input.user.authenticated == true
       input.user.role in allowed_roles[input.resource.type]
   }

   allowed_roles := {
       "agent": ["developer", "admin"],
       "production_data": ["admin"],
       "api_key": ["developer", "admin"]
   }
   ```

4. **Policy Distribution and Acknowledgment**
   - Centralized policy repository (Confluence)
   - Version control (Git)
   - Change notification
   - Annual policy review
   - Employee acknowledgment tracking

**Evidence Artifacts:**
- Policy documents
- Procedure documents
- Policy acknowledgment logs
- Training completion records
- Policy exception requests

**Testing Frequency:** Annual policy review, procedures tested quarterly

---

### CC6: Logical and Physical Access Controls

#### CC6.1: Logical and Physical Access Controls - Restrict Access

**Control Statement:**
The entity implements logical access security software, infrastructure, and architectures over protected information assets to protect them from security events to meet the entity's objectives.

**GreenLang Implementation:**

1. **Authentication Controls**

   **Multi-Factor Authentication (MFA)**
   - Required for all users
   - Supported methods: TOTP, SMS, hardware tokens
   - Backup codes provided
   - MFA enforcement: Cannot bypass

   ```python
   # MFA Implementation
   from greenlang.auth import MFAManager

   def login(username, password):
       # First factor: password
       if not verify_password(username, password):
           audit_log.log_auth_failure(username, "invalid_password")
           return False

       # Second factor: MFA
       if not MFAManager.verify_token(username):
           audit_log.log_auth_failure(username, "invalid_mfa")
           return False

       audit_log.log_auth_success(username)
       return create_session(username)
   ```

   **Password Requirements**
   - Minimum 12 characters
   - Complexity: uppercase, lowercase, numbers, symbols
   - No common passwords (check against breach database)
   - Password history: last 10 passwords
   - Expiration: 90 days
   - Account lockout: 5 failed attempts, 30-minute lockout

2. **Authorization Controls**

   **Role-Based Access Control (RBAC)**
   ```python
   # RBAC Roles
   roles = {
       "viewer": [
           "read:agents",
           "read:data"
       ],
       "developer": [
           "read:agents",
           "write:agents",
           "read:data",
           "execute:agents"
       ],
       "admin": [
           "read:*",
           "write:*",
           "delete:*",
           "admin:*"
       ]
   }

   def check_permission(user, action, resource):
       user_permissions = get_user_permissions(user)
       required_permission = f"{action}:{resource}"

       if required_permission in user_permissions:
           return True

       # Check wildcard permissions
       if f"{action}:*" in user_permissions:
           return True

       audit_log.log_authz_denial(user, action, resource)
       return False
   ```

   **Attribute-Based Access Control (ABAC)**
   - Resource sensitivity level
   - User clearance level
   - Time-based access (business hours only)
   - Location-based access (IP whitelist)
   - Context-based access (VPN required for sensitive data)

3. **Network Access Controls**

   **Firewall Rules**
   ```
   # Production Environment
   - Allow HTTPS (443) from Internet to load balancer
   - Allow SSH (22) only from VPN
   - Allow PostgreSQL (5432) only from app servers
   - Deny all other inbound traffic
   - Allow all outbound traffic (with monitoring)
   ```

   **Network Segmentation**
   ```
   Internet
   ↓
   [DMZ] - Web Tier (Load Balancers)
   ↓
   [Internal Network] - Application Tier (App Servers)
   ↓
   [Data Tier] - Database Servers
   ↓
   [Management Network] - Admin, Monitoring
   ```

   **VPN Access**
   - Required for remote access
   - MFA for VPN authentication
   - Split tunneling disabled
   - Session timeout: 8 hours
   - Access logs reviewed weekly

4. **Physical Access Controls**

   **Data Center Security**
   - Badge access required
   - Biometric authentication for sensitive areas
   - 24/7 security guards
   - CCTV monitoring
   - Visitor logs
   - Escort required for visitors

   **Office Security**
   - Badge access required
   - Reception desk during business hours
   - Visitor sign-in
   - Clean desk policy
   - Locked server rooms

5. **Access Provisioning and Deprovisioning**

   **User Provisioning Process**
   ```
   1. Manager submits access request (Jira ticket)
   2. Security team reviews request
   3. Approval required from:
      - Direct manager
      - Resource owner
      - Security team (for sensitive access)
   4. IT provisions access with least privilege
   5. User notified and trained
   6. Access logged in access registry
   ```

   **User Deprovisioning Process**
   ```
   1. HR notifies IT of termination/transfer
   2. Immediate actions (within 1 hour):
      - Disable Active Directory account
      - Revoke VPN access
      - Disable email
      - Remote wipe mobile devices
   3. Within 24 hours:
      - Revoke all application access
      - Transfer data ownership
      - Collect company assets
      - Exit interview
   4. Within 1 week:
      - Delete user accounts
      - Archive user data
      - Update documentation
   ```

6. **Access Reviews**

   **Quarterly Access Reviews**
   - Manager reviews team access
   - Certify or revoke access
   - Document review in access log
   - Remove unused accounts
   - Escalate anomalies to security team

   **Annual Privileged Access Review**
   - Review all admin/root access
   - Verify business justification
   - Recertify or revoke
   - Security team approval required

**Evidence Artifacts:**
- User access matrix
- Access request tickets
- Provisioning/deprovisioning logs
- Access review certifications
- Authentication logs
- Failed login reports
- Privileged access reports

**Testing Procedures:**
1. Sample 25 users, verify access matches approved level
2. Test MFA enforcement (attempt login without MFA)
3. Verify deprovisioned users cannot access systems
4. Test firewall rules (attempt unauthorized connections)
5. Review access review completion

**Testing Frequency:** Quarterly

---

#### CC6.2: Prior to Issuing System Credentials and Granting System Access

**Control Statement:**
Prior to issuing system credentials and granting system access, the entity registers and authorizes new internal and external users whose access is administered by the entity.

**GreenLang Implementation:**

1. **User Registration Process**

   **New Employee Onboarding**
   ```
   Day -7: Pre-boarding
   - Background check initiated
   - Security clearance verified
   - IT equipment ordered

   Day 0: First Day
   - Onboarding checklist created
   - Security policies reviewed and signed
   - Access request form completed
   - Manager approval obtained

   Day 1: Account Creation
   - Active Directory account created
   - Email account provisioned
   - MFA enrollment required
   - Default access granted (minimal)

   Week 1: Role-Specific Access
   - Role-based access provisioned
   - Application access granted
   - Security training completed
   - VPN access enabled (if needed)
   ```

2. **Authorization Approval Matrix**

   | Access Level | Approver 1 | Approver 2 | Approver 3 |
   |--------------|------------|------------|------------|
   | Basic (read-only) | Manager | - | - |
   | Developer | Manager | Security Team | - |
   | Production Access | Manager | Security Team | CTO |
   | Admin/Root | Manager | Security Team | CTO + CISO |
   | Customer Data | Manager | Security Team | DPO + Legal |

3. **Identity Verification**

   **Internal Users (Employees)**
   - Government-issued ID verification
   - Background check
   - Reference checks
   - HR verification
   - Manager approval

   **External Users (Contractors, Partners)**
   - Vendor due diligence
   - NDA signed
   - Contract in place
   - Sponsor employee identified
   - Limited access scope
   - Expiration date set

4. **Credential Issuance Process**

   ```python
   def issue_credentials(user_request):
       # Verify identity
       if not verify_identity(user_request.id_document):
           raise IdentityVerificationError()

       # Check approvals
       if not has_required_approvals(user_request):
           raise InsufficientApprovalsError()

       # Create account
       username = generate_username(user_request.first_name, user_request.last_name)
       temp_password = generate_secure_password()

       # Create AD account
       ad_account = create_ad_account(username, temp_password)

       # Require password change on first login
       ad_account.set_must_change_password(True)

       # Require MFA enrollment
       mfa_enrollment = create_mfa_enrollment(username)

       # Send credentials securely
       send_credentials_email(user_request.email, username, temp_password, mfa_enrollment.qr_code)

       # Log issuance
       audit_log.log_credential_issuance(username, user_request.approver)

       return ad_account
   ```

5. **Default Account Settings**

   - **Password:** Temporary, must change on first login
   - **MFA:** Must enroll before access granted
   - **Access:** Minimal (read-only)
   - **Expiration:** 90 days (for contractors)
   - **Session timeout:** 30 minutes
   - **VPN:** Disabled by default

**Evidence Artifacts:**
- Access request forms
- Approval records
- Identity verification documents
- Account creation logs
- Credential issuance records
- MFA enrollment confirmations

**Testing Procedures:**
1. Sample 10 new accounts, verify proper approvals
2. Verify identity verification documentation
3. Confirm MFA enrollment before full access granted
4. Test that temporary passwords require change
5. Verify access matches approved level

**Testing Frequency:** Quarterly

---

#### CC6.3: System Credentials and Access Removed When Access Is No Longer Required

**Control Statement:**
The entity removes system credentials and access when users' access is no longer required or expected.

**GreenLang Implementation:**

1. **Automated Deprovisioning Triggers**

   **HR System Integration**
   ```python
   # HR system webhook
   @app.route('/hr/webhook/termination', methods=['POST'])
   def handle_termination():
       event = request.json
       employee_id = event['employee_id']
       termination_date = event['termination_date']

       # Immediate deprovisioning for involuntary terminations
       if event['termination_type'] == 'involuntary':
           deprovision_immediately(employee_id)
       else:
           # Schedule deprovisioning for last day
           schedule_deprovisioning(employee_id, termination_date)

       # Notify IT and Security
       notify_teams(employee_id, termination_date)

       return {'status': 'acknowledged'}
   ```

2. **Deprovisioning Checklist**

   **Immediate Actions (within 1 hour of notification):**
   - [ ] Disable Active Directory account
   - [ ] Revoke VPN access
   - [ ] Disable email access
   - [ ] Remote wipe mobile devices
   - [ ] Disable API keys
   - [ ] Revoke OAuth tokens
   - [ ] Disable cloud console access (AWS, GCP, Azure)

   **Within 24 Hours:**
   - [ ] Revoke application access (all systems)
   - [ ] Transfer data ownership to manager
   - [ ] Remove from email distribution lists
   - [ ] Remove from Slack/Teams channels
   - [ ] Collect physical assets (laptop, badge, keys)
   - [ ] Disable building access badge
   - [ ] Update org chart

   **Within 1 Week:**
   - [ ] Delete user accounts (after data archive)
   - [ ] Archive user data per retention policy
   - [ ] Remove from vendor systems
   - [ ] Update documentation
   - [ ] Conduct exit interview
   - [ ] Final access review and sign-off

3. **Role Change Process**

   ```python
   def handle_role_change(user_id, old_role, new_role):
       # Get current permissions
       current_perms = get_user_permissions(user_id)

       # Calculate new permissions
       new_perms = get_role_permissions(new_role)

       # Remove permissions not in new role
       perms_to_remove = current_perms - new_perms
       for perm in perms_to_remove:
           revoke_permission(user_id, perm)
           audit_log.log_permission_revoked(user_id, perm, "role_change")

       # Add new permissions
       perms_to_add = new_perms - current_perms
       for perm in perms_to_add:
           grant_permission(user_id, perm)
           audit_log.log_permission_granted(user_id, perm, "role_change")

       # Update role
       update_user_role(user_id, new_role)

       # Notify user and manager
       notify_role_change(user_id, old_role, new_role)
   ```

4. **Inactive Account Monitoring**

   **Automated Scanning**
   ```python
   # Daily cron job
   def scan_inactive_accounts():
       # Find accounts not logged in for 90 days
       inactive_accounts = db.query(
           "SELECT user_id, username, last_login "
           "FROM users "
           "WHERE last_login < NOW() - INTERVAL '90 days' "
           "AND status = 'active'"
       )

       for account in inactive_accounts:
           # Notify manager for review
           send_inactive_account_notification(
               account.user_id,
               account.manager_id,
               account.last_login
           )

           # Auto-disable after 120 days
           if account.last_login < now() - timedelta(days=120):
               disable_account(account.user_id)
               audit_log.log_account_disabled(account.user_id, "inactive")
   ```

5. **Contractor/Temporary Access Expiration**

   ```python
   def manage_temporary_access():
       # Find expiring access
       expiring_access = db.query(
           "SELECT user_id, username, expiration_date "
           "FROM users "
           "WHERE expiration_date <= NOW() + INTERVAL '7 days' "
           "AND status = 'active'"
       )

       for user in expiring_access:
           # Notify manager 7 days before expiration
           send_expiration_notification(user.user_id, user.manager_id, user.expiration_date)

           # Auto-revoke on expiration date
           if user.expiration_date <= now():
               revoke_all_access(user.user_id)
               audit_log.log_access_expired(user.user_id)
   ```

**Evidence Artifacts:**
- Termination notifications
- Deprovisioning checklists
- Account disable logs
- Access revocation logs
- Asset return receipts
- Exit interview records
- Inactive account reports

**Testing Procedures:**
1. Sample 10 terminated employees, verify all access revoked within SLA
2. Test inactive account detection (create test account, verify auto-disable)
3. Verify contractor access expires on end date
4. Test role change process (verify old permissions removed)
5. Verify physical access badge disabled

**Testing Frequency:** Quarterly

---

#### CC6.4: Physical Access to Facilities and Data Centers

**Control Statement:**
The entity restricts physical access to facilities and protected information assets to authorized personnel.

**GreenLang Implementation:**

**Note:** GreenLang operates on cloud infrastructure (AWS, GCP). Physical data center access is managed by cloud providers (AWS, GCP), who maintain SOC 2 Type II compliance. GreenLang does not operate its own data centers.

1. **Cloud Provider Physical Security Reliance**

   **AWS Physical Security Controls:**
   - 24/7 security staff
   - Biometric access controls
   - CCTV monitoring
   - Perimeter fencing
   - Security checkpoints
   - Visitor escort requirements
   - SOC 2 Type II certified

   **GreenLang Verification:**
   - Annual review of AWS SOC 2 report
   - Verification of AWS compliance certifications
   - Contractual security requirements in AWS agreement

2. **GreenLang Office Physical Security**

   **Building Access:**
   - Badge access system
   - Visitor sign-in required
   - Visitor escorts required
   - Access logs reviewed monthly
   - Badges deactivated upon termination

   **Server Room Access:**
   - Restricted to IT staff only
   - Separate badge access required
   - CCTV monitoring
   - Environmental controls (fire suppression, HVAC)
   - Access logs reviewed weekly

   **Clean Desk Policy:**
   - No sensitive information left on desks
   - Lock screens when away
   - Lock cabinets overnight
   - Shred sensitive documents
   - No customer data on local machines

3. **Environmental Controls**

   **Office:**
   - Fire suppression system
   - Smoke detectors
   - Emergency lighting
   - Backup power (UPS)
   - HVAC monitoring

   **Cloud (AWS):**
   - Redundant power supplies
   - Diesel generators
   - Climate control
   - Fire detection and suppression
   - Earthquake resistance (depending on region)

**Evidence Artifacts:**
- AWS SOC 2 Type II report
- Office access logs
- Visitor logs
- Server room access logs
- Badge deactivation logs
- Environmental monitoring reports

**Testing Procedures:**
1. Review AWS SOC 2 report annually
2. Test badge access (verify unauthorized badges denied)
3. Review access logs for anomalies
4. Verify terminated employees' badges deactivated
5. Inspect physical security controls (walkthroughs)

**Testing Frequency:** Quarterly office inspection, annual AWS report review

---

#### CC6.5: Access to Logical and Physical Assets Removed Upon Termination

**Control Statement:**
The entity discontinues logical and physical protections over physical assets only after the ability to read or recover data and software from those assets has been diminished.

**GreenLang Implementation:**

1. **Data Sanitization Procedures**

   **Asset Decommissioning Process:**
   ```
   1. Asset Identification
      - Identify asset for decommissioning
      - Classify data sensitivity
      - Document asset details

   2. Data Backup (if needed)
      - Backup required data to secure location
      - Verify backup integrity
      - Document backup location

   3. Data Sanitization
      - Level 1 (Low sensitivity): Standard deletion
      - Level 2 (Medium sensitivity): 3-pass overwrite (DoD 5220.22-M)
      - Level 3 (High sensitivity): 7-pass overwrite + degaussing
      - Level 4 (Critical): Physical destruction

   4. Verification
      - Verify data unrecoverable
      - Document sanitization method
      - Certificate of destruction (if applicable)

   5. Asset Disposal
      - Remove asset tags
      - Update asset inventory
      - Dispose via approved vendor
      - Obtain disposal receipt
   ```

2. **Data Destruction Standards**

   **Software-Based Destruction:**
   ```bash
   # Multi-pass overwrite using shred
   shred -vfz -n 7 /dev/sdX

   # Verify destruction
   dd if=/dev/sdX bs=1M count=10 | hexdump -C
   ```

   **Physical Destruction Methods:**
   - **Hard Drives:** Degaussing + physical destruction (drill/shred)
   - **SSDs:** Cryptographic erasure + physical destruction
   - **USB Drives:** Physical destruction (shredding)
   - **Tapes:** Degaussing + physical destruction
   - **Paper:** Cross-cut shredding (min 3.9mm x 38mm)

3. **Mobile Device Management (MDM)**

   **Remote Wipe Capabilities:**
   ```python
   def terminate_employee_devices(employee_id):
       # Get all devices assigned to employee
       devices = get_employee_devices(employee_id)

       for device in devices:
           # Remote wipe
           mdm_client.wipe_device(
               device_id=device.id,
               wipe_type='full',  # Full device wipe
               pin_required=False  # Bypass PIN for terminations
           )

           # Verify wipe completion
           status = mdm_client.get_wipe_status(device.id)
           if status == 'completed':
               audit_log.log_device_wiped(employee_id, device.id)
           else:
               alert_security_team(f"Device wipe failed: {device.id}")

       # Remove from MDM
       mdm_client.unenroll_devices(employee_id)
   ```

4. **Encrypted Storage**

   **Full Disk Encryption:**
   - All laptops encrypted (BitLocker, FileVault)
   - Encryption keys managed centrally
   - Keys destroyed upon decommissioning
   - Verification: Cannot access data without key

   **Cloud Storage Encryption:**
   - Data encrypted at rest (AES-256)
   - Customer-managed keys (AWS KMS)
   - Key rotation every 90 days
   - Keys destroyed when data deleted

5. **Asset Inventory Management**

   **Tracking System:**
   - All assets tagged and tracked (Snipe-IT)
   - Assignment to employees logged
   - Return upon termination logged
   - Decommissioning status tracked

   **Decommissioning Workflow:**
   ```
   Asset Return
   ↓
   Data Backup (if needed)
   ↓
   Data Sanitization
   ↓
   Verification
   ↓
   Update Inventory (status = decommissioned)
   ↓
   Physical Disposal
   ↓
   Certificate of Destruction Filed
   ```

6. **Third-Party Disposal**

   **Approved Vendors:**
   - NAID AAA Certified
   - R2 (Responsible Recycling) Certified
   - e-Stewards Certified
   - Certificate of destruction provided
   - Annual vendor audits

**Evidence Artifacts:**
- Asset inventory records
- Data sanitization logs
- Certificates of destruction
- MDM wipe confirmations
- Disposal vendor certifications
- Verification test results

**Testing Procedures:**
1. Sample 10 decommissioned assets, verify sanitization documented
2. Attempt data recovery on sanitized devices (should fail)
3. Verify certificates of destruction on file
4. Test MDM remote wipe capability
5. Review disposal vendor certifications

**Testing Frequency:** Quarterly

---

#### CC6.6: System Vulnerabilities Management

**Control Statement:**
The entity implements controls to prevent or detect and act upon the introduction of unauthorized or malicious software.

**GreenLang Implementation:**

1. **Vulnerability Management Program**

   **Vulnerability Scanning:**
   ```yaml
   # Daily automated scans
   - Infrastructure scan (Nessus): Daily at 2 AM
   - Application scan (OWASP ZAP): Daily at 3 AM
   - Dependency scan (Snyk): On every commit
   - Container scan (Trivy): On every image build
   - Cloud config scan (AWS Inspector): Daily
   ```

   **Scan Coverage:**
   - Production infrastructure (100%)
   - Staging infrastructure (100%)
   - Development infrastructure (100%)
   - All applications and APIs
   - All dependencies and libraries
   - All container images
   - Cloud configurations

2. **Vulnerability Assessment Process**

   ```python
   def assess_vulnerability(vuln):
       # Calculate CVSS score
       cvss_score = calculate_cvss_score(vuln)

       # Determine severity
       if cvss_score >= 9.0:
           severity = "Critical"
           sla_hours = 24
       elif cvss_score >= 7.0:
           severity = "High"
           sla_hours = 168  # 7 days
       elif cvss_score >= 4.0:
           severity = "Medium"
           sla_hours = 720  # 30 days
       else:
           severity = "Low"
           sla_hours = 2160  # 90 days

       # Check exploitability
       if is_actively_exploited(vuln.cve):
           severity = "Critical"
           sla_hours = 24

       # Create ticket
       ticket = create_jira_ticket(
           title=f"[{severity}] {vuln.cve}: {vuln.description}",
           severity=severity,
           sla=sla_hours,
           affected_systems=vuln.affected_systems
       )

       # Notify security team
       notify_security_team(ticket, severity)

       return ticket
   ```

3. **Patch Management Process**

   **Patching SLAs:**
   | Severity | SLA | Approval Required |
   |----------|-----|-------------------|
   | Critical | 24 hours | Security Team |
   | High | 7 days | Team Lead |
   | Medium | 30 days | Team Lead |
   | Low | 90 days | Normal Change |

   **Patching Workflow:**
   ```
   Vulnerability Identified
   ↓
   Risk Assessment
   ↓
   Patch Testing (Staging)
   ↓
   Change Request (if production)
   ↓
   Approval
   ↓
   Patch Deployment
   ↓
   Verification Scan
   ↓
   Documentation
   ```

   **Automated Patching:**
   ```bash
   # Automated OS patching (non-production)
   # /etc/cron.d/auto-updates
   0 3 * * 0 root unattended-upgrade

   # Automated dependency updates (Dependabot)
   # .github/dependabot.yml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "daily"
       open-pull-requests-limit: 10
   ```

4. **Malware Prevention**

   **Endpoint Protection:**
   - Antivirus software on all endpoints (CrowdStrike, Windows Defender)
   - Real-time scanning enabled
   - Daily signature updates
   - Weekly full scans
   - Quarantine malware automatically
   - Alert security team on detection

   **Email Protection:**
   - Spam filtering (Proofpoint)
   - Attachment scanning
   - Link scanning and rewriting
   - Phishing detection
   - User awareness training

   **Application Security:**
   ```python
   # File upload validation
   from greenlang.security import FileUploadValidator

   def upload_file(file):
       # Validate file type
       if not FileUploadValidator.validate_file_type(file, allowed_types=['.csv', '.xlsx']):
           raise ValueError("Invalid file type")

       # Scan for malware
       if not malware_scanner.scan_file(file):
           audit_log.log_malware_detected(file.filename)
           raise ValueError("Malware detected")

       # Sanitize filename
       safe_filename = FileUploadValidator.sanitize_filename(file.filename)

       # Save file
       file.save(safe_filename)
   ```

5. **Intrusion Detection and Prevention**

   **Host-Based IDS (HIDS):**
   - File integrity monitoring (OSSEC)
   - Log monitoring
   - Rootkit detection
   - Alerts on suspicious activity

   **Network-Based IDS (NIDS):**
   - Signature-based detection (Snort)
   - Anomaly-based detection
   - DDoS detection
   - Lateral movement detection

   **Security Information and Event Management (SIEM):**
   ```python
   # SIEM correlation rules
   rules = [
       # Multiple failed logins from same IP
       {
           'name': 'Brute Force Attack',
           'condition': 'failed_logins > 10 AND time_window = 5 minutes',
           'action': 'block_ip AND alert_security_team'
       },

       # Privilege escalation
       {
           'name': 'Privilege Escalation',
           'condition': 'user.role_changed AND new_role = admin',
           'action': 'alert_security_team AND require_approval'
       },

       # Data exfiltration
       {
           'name': 'Data Exfiltration',
           'condition': 'data_transfer > 1GB AND destination = external',
           'action': 'block_transfer AND alert_security_team'
       }
   ]
   ```

6. **Vulnerability Metrics and Reporting**

   **KPIs Tracked:**
   - Mean time to detect (MTTD) vulnerabilities
   - Mean time to remediate (MTTR) vulnerabilities
   - Number of vulnerabilities by severity
   - SLA compliance rate
   - Patch coverage percentage
   - Scan coverage percentage

   **Monthly Reporting:**
   - Vulnerability dashboard
   - Trend analysis
   - SLA compliance report
   - Risk reduction metrics
   - Executive summary

**Evidence Artifacts:**
- Vulnerability scan reports
- Patch management logs
- Malware detection alerts
- IDS/IPS alerts
- SIEM logs and alerts
- Remediation tickets and closures
- Vulnerability metrics dashboard

**Testing Procedures:**
1. Verify vulnerability scans running daily (check logs)
2. Sample 10 vulnerabilities, verify remediated within SLA
3. Test malware detection (EICAR test file)
4. Verify patch deployment process
5. Test IDS alerting (simulate attack)

**Testing Frequency:** Monthly vulnerability metrics review, quarterly penetration test

---

#### CC6.7: Transmission of Data Over Communication Channels

**Control Statement:**
The entity restricts the transmission, movement, and removal of information to authorized internal and external users and processes, and protects it during transmission, movement, or removal to meet the entity's objectives.

**GreenLang Implementation:**

1. **Encryption in Transit**

   **TLS/SSL Configuration:**
   ```nginx
   # Nginx TLS configuration
   ssl_protocols TLSv1.3 TLSv1.2;
   ssl_ciphers 'TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384';
   ssl_prefer_server_ciphers off;
   ssl_session_cache shared:MozSSL:10m;
   ssl_session_timeout 1d;
   ssl_session_tickets off;

   # OCSP stapling
   ssl_stapling on;
   ssl_stapling_verify on;

   # HSTS
   add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
   ```

   **Certificate Management:**
   - Certificates from trusted CAs (Let's Encrypt, DigiCert)
   - Automatic renewal (certbot)
   - Certificate pinning for mobile apps
   - Certificate transparency monitoring
   - Alert on expiration (30 days before)

2. **API Security**

   **API Authentication:**
   ```python
   from greenlang.auth import APIKeyAuth

   @app.route('/api/v1/agents', methods=['GET'])
   @require_api_key
   def list_agents():
       # API key validated by decorator
       api_key = request.headers.get('X-API-Key')
       user = APIKeyAuth.get_user_from_key(api_key)

       # Check rate limit
       if not rate_limiter.check(user.id):
           return {'error': 'Rate limit exceeded'}, 429

       # Audit log
       audit_log.log_api_request(user.id, '/api/v1/agents')

       return {'agents': get_user_agents(user.id)}
   ```

   **API Rate Limiting:**
   - 1000 requests/hour per API key
   - 10 requests/second burst limit
   - Rate limit headers in response
   - 429 status code on limit exceeded

3. **Data Loss Prevention (DLP)**

   **Email DLP:**
   - Scan outbound emails for sensitive data
   - Block emails with credit card numbers, SSNs
   - Encrypt emails with sensitive keywords
   - Quarantine suspicious emails

   **Endpoint DLP:**
   - Monitor USB device usage
   - Block unauthorized USB devices
   - Monitor large file transfers
   - Alert on sensitive data copying

   **Network DLP:**
   ```python
   # DLP monitoring
   def monitor_data_transfer(connection):
       # Detect sensitive data patterns
       if detect_credit_card(connection.payload):
           block_connection(connection)
           alert_security_team("Credit card data in transmission")

       if detect_ssn(connection.payload):
           block_connection(connection)
           alert_security_team("SSN in transmission")

       # Monitor large transfers
       if connection.bytes_transferred > 1GB:
           alert_security_team(f"Large data transfer: {connection.bytes_transferred}")
   ```

4. **Secure File Transfer**

   **SFTP for File Transfers:**
   ```python
   import paramiko

   def secure_file_transfer(local_file, remote_path):
       # Connect via SFTP
       transport = paramiko.Transport(('sftp.greenlang.io', 22))
       transport.connect(username='user', pkey=private_key)

       sftp = paramiko.SFTPClient.from_transport(transport)

       # Transfer file
       sftp.put(local_file, remote_path)

       # Verify transfer
       remote_stat = sftp.stat(remote_path)
       local_stat = os.stat(local_file)

       if remote_stat.st_size != local_stat.st_size:
           raise TransferError("File size mismatch")

       # Audit log
       audit_log.log_file_transfer(local_file, remote_path, remote_stat.st_size)

       sftp.close()
       transport.close()
   ```

5. **VPN for Remote Access**

   **VPN Configuration:**
   - OpenVPN or WireGuard
   - MFA for VPN authentication
   - Split tunneling disabled
   - Kill switch enabled
   - DNS leak protection
   - Session timeout: 8 hours

6. **Database Connection Security**

   ```python
   # Encrypted database connection
   from sqlalchemy import create_engine

   engine = create_engine(
       'postgresql://user:password@db.greenlang.io:5432/greenlang',
       connect_args={
           'sslmode': 'require',
           'sslcert': '/path/to/client-cert.pem',
           'sslkey': '/path/to/client-key.pem',
           'sslrootcert': '/path/to/ca-cert.pem'
       }
   )
   ```

7. **Monitoring and Alerting**

   **TLS Certificate Monitoring:**
   - Daily certificate expiration checks
   - Alert 30 days before expiration
   - Monitor for weak ciphers
   - Check for certificate revocation

   **Data Transfer Monitoring:**
   - Monitor all network connections
   - Alert on unencrypted connections
   - Track data transfer volumes
   - Detect anomalous transfers

**Evidence Artifacts:**
- TLS configuration files
- Certificate inventory
- Certificate renewal logs
- DLP alerts and blocks
- VPN connection logs
- API request logs
- Data transfer logs

**Testing Procedures:**
1. Scan for TLS misconfigurations (SSL Labs)
2. Test certificate validation
3. Attempt unencrypted connection (should be blocked)
4. Test DLP rules (send test sensitive data)
5. Verify VPN encryption (packet capture)

**Testing Frequency:** Monthly TLS scans, quarterly DLP testing

---

#### CC6.8: Identification and Authentication of Users

**Control Statement:**
The entity implements controls to prevent unauthorized access to system components through identification and authentication of users.

**GreenLang Implementation:**

1. **User Identification**

   **Unique User IDs:**
   - Every user assigned unique username
   - No shared accounts
   - Service accounts clearly labeled
   - User ID format: `firstname.lastname@greenlang.io`
   - Employee ID linked to user ID

2. **Authentication Methods**

   **Password Authentication:**
   ```python
   import bcrypt
   from greenlang.security import PasswordValidator

   def create_user(username, password):
       # Validate password strength
       if not PasswordValidator.validate_password(password):
           raise ValueError(
               "Password must be:\n"
               "- At least 12 characters\n"
               "- Include uppercase and lowercase\n"
               "- Include numbers and symbols\n"
               "- Not in common password list"
           )

       # Hash password with bcrypt
       salt = bcrypt.gensalt(rounds=12)
       password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)

       # Store user
       db.execute(
           "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
           (username, password_hash, datetime.now())
       )

       audit_log.log_user_created(username)

   def verify_password(username, password):
       # Get user
       user = db.query("SELECT password_hash FROM users WHERE username = ?", (username,))

       if not user:
           # Constant-time response to prevent user enumeration
           bcrypt.hashpw(b"dummy", bcrypt.gensalt())
           return False

       # Verify password
       is_valid = bcrypt.checkpw(password.encode('utf-8'), user.password_hash)

       if is_valid:
           audit_log.log_auth_success(username)
       else:
           audit_log.log_auth_failure(username, "invalid_password")
           increment_failed_login_count(username)

       return is_valid
   ```

   **Multi-Factor Authentication (MFA):**
   ```python
   import pyotp
   from greenlang.security import MFAManager

   def enroll_mfa(user_id):
       # Generate secret
       secret = pyotp.random_base32()

       # Store secret (encrypted)
       MFAManager.store_secret(user_id, secret)

       # Generate QR code
       totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
           name=user_id,
           issuer_name='GreenLang'
       )

       qr_code = generate_qr_code(totp_uri)

       # Return QR code and backup codes
       backup_codes = MFAManager.generate_backup_codes(user_id)

       return {
           'qr_code': qr_code,
           'secret': secret,  # For manual entry
           'backup_codes': backup_codes
       }

   def verify_mfa(user_id, token):
       # Get user secret
       secret = MFAManager.get_secret(user_id)

       # Verify TOTP token
       totp = pyotp.TOTP(secret)
       is_valid = totp.verify(token, valid_window=1)  # Allow 30s drift

       if is_valid:
           audit_log.log_mfa_success(user_id)
       else:
           # Check backup codes
           if MFAManager.verify_backup_code(user_id, token):
               is_valid = True
               audit_log.log_mfa_success(user_id, method='backup_code')
           else:
               audit_log.log_mfa_failure(user_id)

       return is_valid
   ```

3. **Single Sign-On (SSO)**

   **SAML Integration:**
   ```python
   from onelogin.saml2.auth import OneLogin_Saml2_Auth

   def sso_login(request):
       # Initialize SAML auth
       auth = OneLogin_Saml2_Auth(request, saml_settings)

       # Process SAML response
       auth.process_response()

       errors = auth.get_errors()
       if errors:
           audit_log.log_sso_failure(errors)
           raise AuthenticationError("SSO authentication failed")

       # Get user attributes
       attributes = auth.get_attributes()
       user_id = attributes['email'][0]

       # Create or update user
       user = get_or_create_user(user_id, attributes)

       # Create session
       session = create_session(user.id)

       audit_log.log_sso_success(user_id)

       return session
   ```

4. **Session Management**

   ```python
   import secrets
   from datetime import datetime, timedelta

   def create_session(user_id):
       # Generate cryptographically secure session ID
       session_id = secrets.token_urlsafe(32)

       # Create session
       session = {
           'session_id': session_id,
           'user_id': user_id,
           'created_at': datetime.now(),
           'expires_at': datetime.now() + timedelta(hours=8),
           'idle_timeout': timedelta(minutes=30),
           'last_activity': datetime.now()
       }

       # Store session (Redis)
       redis_client.setex(
           f"session:{session_id}",
           timedelta(hours=8),
           json.dumps(session)
       )

       audit_log.log_session_created(user_id, session_id)

       return session

   def validate_session(session_id):
       # Get session
       session_data = redis_client.get(f"session:{session_id}")

       if not session_data:
           return None

       session = json.loads(session_data)

       # Check expiration
       if datetime.now() > session['expires_at']:
           invalidate_session(session_id)
           return None

       # Check idle timeout
       idle_time = datetime.now() - session['last_activity']
       if idle_time > session['idle_timeout']:
           invalidate_session(session_id)
           audit_log.log_session_timeout(session['user_id'], session_id)
           return None

       # Update last activity
       session['last_activity'] = datetime.now()
       redis_client.setex(
           f"session:{session_id}",
           session['expires_at'] - datetime.now(),
           json.dumps(session)
       )

       return session
   ```

5. **Account Lockout**

   ```python
   def handle_failed_login(username):
       # Increment failed login count
       failed_count = redis_client.incr(f"failed_logins:{username}")

       # Set expiration (15 minutes)
       if failed_count == 1:
           redis_client.expire(f"failed_logins:{username}", 900)

       # Lock account after 5 failed attempts
       if failed_count >= 5:
           lock_account(username, duration=timedelta(minutes=30))
           audit_log.log_account_locked(username, reason="too_many_failed_logins")

           # Alert security team
           alert_security_team(f"Account locked: {username}")

   def is_account_locked(username):
       return redis_client.exists(f"account_locked:{username}")

   def lock_account(username, duration):
       redis_client.setex(
           f"account_locked:{username}",
           duration,
           "locked"
       )
   ```

6. **Service Account Management**

   ```python
   def create_service_account(name, purpose, owner):
       # Generate service account username
       username = f"svc_{name}@greenlang.io"

       # Generate API key (no password)
       api_key = generate_api_key(prefix="gl_svc_")

       # Create account
       account = {
           'username': username,
           'type': 'service',
           'purpose': purpose,
           'owner': owner,
           'api_key': hash_api_key(api_key),
           'created_at': datetime.now()
       }

       db.insert('users', account)

       # Audit log
       audit_log.log_service_account_created(username, owner)

       return {
           'username': username,
           'api_key': api_key  # Show only once
       }
   ```

**Evidence Artifacts:**
- User account registry
- Authentication logs
- MFA enrollment records
- Session creation/termination logs
- Failed login reports
- Account lockout logs
- Service account inventory

**Testing Procedures:**
1. Verify password requirements enforced
2. Test MFA enforcement (attempt login without MFA)
3. Verify session timeout (wait 30 min, session should expire)
4. Test account lockout (5 failed logins)
5. Verify unique user IDs (no duplicates)
6. Test SSO integration

**Testing Frequency:** Quarterly

---

### CC7: System Operations

#### CC7.1: System Capacity Planning

**Control Statement:**
To meet its objectives, the entity uses detection and monitoring procedures to identify anomalies.

**GreenLang Implementation:**

1. **Capacity Monitoring**

   **Infrastructure Metrics:**
   ```yaml
   # Prometheus metrics collection
   metrics:
     - cpu_usage_percent
     - memory_usage_percent
     - disk_usage_percent
     - network_throughput_mbps
     - database_connections
     - api_requests_per_second
     - response_time_ms
   ```

   **Monitoring Thresholds:**
   | Metric | Warning | Critical | Action |
   |--------|---------|----------|--------|
   | CPU Usage | 70% | 90% | Scale up |
   | Memory Usage | 75% | 90% | Scale up |
   | Disk Usage | 80% | 95% | Add storage |
   | API Response Time | 500ms | 1000ms | Investigate |
   | Error Rate | 1% | 5% | Alert on-call |

2. **Capacity Planning Process**

   ```python
   def capacity_planning_analysis():
       # Historical usage trends
       usage_data = get_usage_data(days=90)

       # Calculate growth rate
       growth_rate = calculate_growth_rate(usage_data)

       # Forecast capacity needs
       forecast = {
           '30_days': forecast_usage(usage_data, growth_rate, days=30),
           '90_days': forecast_usage(usage_data, growth_rate, days=90),
           '180_days': forecast_usage(usage_data, growth_rate, days=180)
       }

       # Identify capacity gaps
       current_capacity = get_current_capacity()
       gaps = []

       for period, predicted_usage in forecast.items():
           if predicted_usage > current_capacity * 0.8:  # 80% threshold
               gaps.append({
                   'period': period,
                   'predicted_usage': predicted_usage,
                   'current_capacity': current_capacity,
                   'shortfall': predicted_usage - current_capacity
               })

       # Generate report
       return {
           'forecast': forecast,
           'gaps': gaps,
           'recommendation': generate_capacity_recommendations(gaps)
       }
   ```

3. **Auto-Scaling Configuration**

   ```yaml
   # Kubernetes HPA (Horizontal Pod Autoscaler)
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: greenlang-api
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: greenlang-api
     minReplicas: 3
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 75
   ```

**Evidence Artifacts:**
- Capacity planning reports (quarterly)
- Usage trend analysis
- Capacity forecasts
- Scaling events log
- Performance metrics dashboards

**Testing Frequency:** Quarterly capacity review

---

#### CC7.2: System Monitoring

**Control Statement:**
The entity monitors system components and the operation of those components for anomalies that are indicative of malicious acts, natural disasters, and errors affecting the entity's ability to meet its objectives; anomalies are analyzed to determine whether they represent security events.

**GreenLang Implementation:**

1. **Comprehensive Monitoring Strategy**

   **Application Monitoring:**
   ```python
   # Application performance monitoring
   from greenlang.monitoring import monitor

   @monitor.trace()
   def calculate_emissions(data):
       with monitor.span("validate_input"):
           validated_data = validate_emission_data(data)

       with monitor.span("query_emission_factors"):
           emission_factors = get_emission_factors(validated_data)

       with monitor.span("calculate"):
           result = perform_calculation(validated_data, emission_factors)

       monitor.record_metric("emissions_calculated", 1)
       monitor.record_metric("calculation_time_ms", monitor.get_elapsed_time())

       return result
   ```

   **Infrastructure Monitoring:**
   - **Prometheus:** Metrics collection
   - **Grafana:** Visualization dashboards
   - **AlertManager:** Alert routing
   - **Node Exporter:** Server metrics
   - **cAdvisor:** Container metrics

2. **Security Monitoring (SIEM)**

   **Log Sources:**
   ```yaml
   # Logs collected and analyzed
   log_sources:
     - application_logs: /var/log/greenlang/*.log
     - web_server_logs: /var/log/nginx/*.log
     - database_logs: /var/log/postgresql/*.log
     - authentication_logs: /var/log/auth.log
     - firewall_logs: /var/log/ufw.log
     - cloud_logs: AWS CloudTrail, VPC Flow Logs
     - container_logs: Docker, Kubernetes
   ```

   **SIEM Correlation Rules:**
   ```python
   # Example correlation rules
   correlation_rules = [
       {
           'name': 'Brute Force Attack',
           'condition': lambda events: (
               len([e for e in events if e['type'] == 'auth_failure']) > 10
               and time_window(events) < timedelta(minutes=5)
           ),
           'severity': 'high',
           'action': ['block_ip', 'alert_soc']
       },
       {
           'name': 'Privilege Escalation',
           'condition': lambda events: (
               any(e['type'] == 'role_change' and e['new_role'] == 'admin' for e in events)
               and not any(e['type'] == 'approval' for e in events)
           ),
           'severity': 'critical',
           'action': ['revert_change', 'alert_soc', 'alert_management']
       },
       {
           'name': 'Data Exfiltration',
           'condition': lambda events: (
               sum(e['bytes_transferred'] for e in events if e['type'] == 'data_transfer') > 10 * 1024 * 1024 * 1024  # 10GB
               and time_window(events) < timedelta(hours=1)
           ),
           'severity': 'critical',
           'action': ['block_transfer', 'alert_soc', 'alert_management']
       }
   ]
   ```

3. **Anomaly Detection**

   ```python
   from sklearn.ensemble import IsolationForest
   import numpy as np

   class AnomalyDetector:
       def __init__(self):
           self.model = IsolationForest(contamination=0.1, random_state=42)
           self.is_trained = False

       def train(self, normal_data):
           """Train on normal behavior"""
           self.model.fit(normal_data)
           self.is_trained = True

       def detect(self, current_data):
           """Detect anomalies in current data"""
           if not self.is_trained:
               raise ValueError("Model not trained")

           # Predict (-1 = anomaly, 1 = normal)
           predictions = self.model.predict(current_data)

           # Get anomaly scores
           scores = self.model.score_samples(current_data)

           # Identify anomalies
           anomalies = []
           for idx, (pred, score) in enumerate(zip(predictions, scores)):
               if pred == -1:
                   anomalies.append({
                       'index': idx,
                       'data': current_data[idx],
                       'score': score,
                       'severity': 'high' if score < -0.5 else 'medium'
                   })

           return anomalies

   # Usage
   detector = AnomalyDetector()

   # Train on 90 days of normal API usage
   normal_usage = get_api_usage_metrics(days=90)
   detector.train(normal_usage)

   # Detect anomalies in current usage
   current_usage = get_current_api_usage()
   anomalies = detector.detect(current_usage)

   for anomaly in anomalies:
       alert_security_team(f"Anomaly detected: {anomaly}")
   ```

4. **Alert Management**

   **Alert Routing:**
   ```yaml
   # AlertManager configuration
   route:
     receiver: 'default'
     group_by: ['alertname', 'severity']
     group_wait: 30s
     group_interval: 5m
     repeat_interval: 12h

     routes:
       # Critical alerts -> PagerDuty
       - match:
           severity: critical
         receiver: pagerduty
         continue: true

       # High alerts -> Slack + Email
       - match:
           severity: high
         receiver: slack-high
         continue: true

       # Medium alerts -> Slack
       - match:
           severity: medium
         receiver: slack-medium

   receivers:
     - name: 'pagerduty'
       pagerduty_configs:
         - service_key: '<key>'

     - name: 'slack-high'
       slack_configs:
         - channel: '#security-alerts'
           text: 'HIGH: {{ .GroupLabels.alertname }}'

     - name: 'slack-medium'
       slack_configs:
         - channel: '#monitoring'
           text: 'MEDIUM: {{ .GroupLabels.alertname }}'
   ```

5. **Monitoring Dashboards**

   **Security Operations Center (SOC) Dashboard:**
   - Real-time threat map
   - Failed authentication attempts
   - Firewall blocks
   - Malware detections
   - Vulnerability scan results
   - Incident status

   **Operations Dashboard:**
   - System uptime
   - API response times
   - Error rates
   - Database performance
   - Queue depths
   - Resource utilization

6. **Incident Response Integration**

   ```python
   def handle_security_event(event):
       # Classify severity
       severity = classify_severity(event)

       # Create incident
       if severity in ['critical', 'high']:
           incident = create_incident(
               title=event['description'],
               severity=severity,
               source=event['source'],
               details=event
           )

           # Notify on-call
           notify_oncall_engineer(incident)

           # Auto-remediate if possible
           if event['type'] in AUTO_REMEDIATION_TYPES:
               remediation_result = auto_remediate(event)
               add_incident_note(incident.id, f"Auto-remediation: {remediation_result}")

       # Log event
       audit_log.log_security_event(event, severity)

       # Update SIEM
       siem.ingest_event(event)
   ```

**Evidence Artifacts:**
- Monitoring dashboards (screenshots)
- SIEM logs and alerts
- Anomaly detection reports
- Incident response tickets
- Alert escalation logs
- SOC activity reports

**Testing Procedures:**
1. Verify all log sources feeding into SIEM
2. Test alert routing (trigger test alerts)
3. Verify anomaly detection (inject test anomalies)
4. Review monitoring coverage (all systems monitored?)
5. Test incident response integration

**Testing Frequency:** Monthly monitoring review, quarterly penetration test

---

#### CC7.3: System Change Management

**Control Statement:**
The entity implements change management and configuration control procedures to manage changes to system components.

**GreenLang Implementation:**

1. **Change Management Process**

   **Change Types:**
   | Change Type | Approval Required | Testing Required | Rollback Plan Required |
   |-------------|------------------|------------------|------------------------|
   | Emergency | CTO or CISO | Post-implementation | Yes |
   | Standard | Team Lead | Yes (Staging) | Yes |
   | Low Risk | Automated | Yes (CI/CD) | Automated |
   | High Risk | CAB + CTO | Yes (Staging + UAT) | Yes |

   **Change Approval Board (CAB):**
   - Meets weekly
   - Reviews high-risk changes
   - Members: CTO, VP Engineering, VP Security, Operations Lead
   - Quorum: 3 members
   - Decision logged in change ticket

2. **Change Request Workflow**

   ```python
   class ChangeRequest:
       def __init__(self, title, description, change_type):
           self.title = title
           self.description = description
           self.change_type = change_type
           self.status = 'draft'
           self.approvals = []
           self.created_at = datetime.now()

       def submit(self):
           # Validate change request
           if not self.validate():
               raise ValueError("Change request validation failed")

           # Determine approval workflow
           approvers = self.get_required_approvers()

           # Send for approval
           for approver in approvers:
               send_approval_request(approver, self)

           self.status = 'pending_approval'
           audit_log.log_change_request_submitted(self)

       def validate(self):
           required_fields = ['title', 'description', 'change_type', 'rollback_plan', 'test_plan']
           return all(getattr(self, field) for field in required_fields)

       def get_required_approvers(self):
           if self.change_type == 'emergency':
               return ['cto', 'ciso']
           elif self.change_type == 'high_risk':
               return ['team_lead', 'cab', 'cto']
           elif self.change_type == 'standard':
               return ['team_lead']
           else:
               return []  # Automated

       def approve(self, approver):
           self.approvals.append({
               'approver': approver,
               'approved_at': datetime.now()
           })

           if len(self.approvals) >= len(self.get_required_approvers()):
               self.status = 'approved'
               audit_log.log_change_approved(self)
               notify_requester(self)
   ```

3. **Configuration Management**

   **Infrastructure as Code (IaC):**
   ```yaml
   # Terraform configuration (version controlled)
   resource "aws_instance" "greenlang_api" {
     ami           = "ami-0c55b159cbfafe1f0"
     instance_type = "t3.medium"

     tags = {
       Name        = "greenlang-api-prod"
       Environment = "production"
       ManagedBy   = "terraform"
       Owner       = "devops@greenlang.io"
     }

     # Security group
     vpc_security_group_ids = [aws_security_group.api.id]

     # User data (configuration)
     user_data = file("${path.module}/user_data.sh")

     # Monitoring
     monitoring = true
   }
   ```

   **Configuration Versioning:**
   - All configurations in Git
   - Pull request required for changes
   - Code review required
   - Automated testing in CI/CD
   - Tagged releases

4. **Deployment Process**

   **CI/CD Pipeline:**
   ```yaml
   # GitLab CI/CD pipeline
   stages:
     - test
     - build
     - deploy_staging
     - deploy_production

   test:
     stage: test
     script:
       - pytest tests/
       - bandit -r greenlang/
       - safety check
     only:
       - merge_requests
       - master

   build:
     stage: build
     script:
       - docker build -t greenlang-api:$CI_COMMIT_SHA .
       - docker push greenlang-api:$CI_COMMIT_SHA
     only:
       - master

   deploy_staging:
     stage: deploy_staging
     script:
       - kubectl set image deployment/greenlang-api greenlang-api=greenlang-api:$CI_COMMIT_SHA -n staging
       - kubectl rollout status deployment/greenlang-api -n staging
     only:
       - master
     environment:
       name: staging

   deploy_production:
     stage: deploy_production
     script:
       - kubectl set image deployment/greenlang-api greenlang-api=greenlang-api:$CI_COMMIT_SHA -n production
       - kubectl rollout status deployment/greenlang-api -n production
     only:
       - master
     when: manual  # Require manual approval
     environment:
       name: production
   ```

5. **Rollback Procedures**

   ```bash
   # Kubernetes rollback
   # Rollback to previous version
   kubectl rollout undo deployment/greenlang-api -n production

   # Rollback to specific revision
   kubectl rollout undo deployment/greenlang-api --to-revision=5 -n production

   # Check rollback status
   kubectl rollout status deployment/greenlang-api -n production
   ```

6. **Configuration Drift Detection**

   ```python
   def detect_configuration_drift():
       # Get expected configuration (from Terraform state)
       expected_config = terraform.get_state()

       # Get actual configuration (from AWS)
       actual_config = aws.get_current_configuration()

       # Compare
       drift = []
       for resource, expected in expected_config.items():
           actual = actual_config.get(resource)

           if actual != expected:
               drift.append({
                   'resource': resource,
                   'expected': expected,
                   'actual': actual,
                   'difference': calculate_diff(expected, actual)
               })

       # Alert on drift
       if drift:
           alert_devops_team(f"Configuration drift detected: {len(drift)} resources")
           create_remediation_ticket(drift)

       return drift

   # Run daily
   schedule.every().day.at("02:00").do(detect_configuration_drift)
   ```

**Evidence Artifacts:**
- Change request tickets
- Approval records
- Deployment logs
- Rollback logs
- Configuration version history
- CAB meeting minutes
- Configuration drift reports

**Testing Procedures:**
1. Sample 20 changes, verify proper approval
2. Verify rollback plans documented
3. Test rollback procedure (staging environment)
4. Verify configuration versioning
5. Test configuration drift detection

**Testing Frequency:** Quarterly

---

### CC8: Change Management

#### CC8.1: Change Management Process

**Control Statement:**
The entity authorizes, designs, develops or acquires, configures, documents, tests, approves, and implements changes to infrastructure, data, software, and procedures to meet its objectives.

**GreenLang Implementation:**

*This control is covered in detail in CC7.3 above.*

**Additional Considerations:**

1. **Software Development Lifecycle (SDLC)**

   **Security in SDLC:**
   ```
   Requirements
   ↓
   - Security requirements gathering
   - Threat modeling
   ↓
   Design
   ↓
   - Secure design review
   - Architecture review
   ↓
   Development
   ↓
   - Secure coding standards
   - Code review
   - SAST (static analysis)
   ↓
   Testing
   ↓
   - Unit tests
   - Integration tests
   - Security tests (DAST, penetration test)
   ↓
   Deployment
   ↓
   - Configuration review
   - Deployment checklist
   - Monitoring setup
   ↓
   Maintenance
   ↓
   - Patch management
   - Vulnerability management
   ```

2. **Emergency Change Procedure**

   ```python
   def emergency_change_process(change):
       # Document emergency
       incident = create_incident(
           title=f"Emergency Change: {change.title}",
           description=change.description,
           severity='high'
       )

       # Notify leadership
       notify_executives(incident)

       # Get emergency approval (CTO or CISO)
       approval = request_emergency_approval(change, incident)

       if approval:
           # Implement change
           result = implement_change(change)

           # Post-implementation review (within 24 hours)
           schedule_post_implementation_review(change, incident)

           # Audit log
           audit_log.log_emergency_change(change, approval, result)
       else:
           notify_requester("Emergency change denied")
   ```

**Evidence Artifacts:**
- SDLC documentation
- Change management policy
- Emergency change logs
- Post-implementation reviews
- Change success/failure metrics

**Testing Frequency:** Covered in CC7.3 testing

---

## Evidence Collection Procedures

### 1. Automated Evidence Collection

**Daily Automated Collection:**
```python
def daily_evidence_collection():
    date = datetime.now().strftime('%Y-%m-%d')
    evidence_dir = f"/compliance/evidence/{date}"

    # User access matrix
    export_user_access_matrix(f"{evidence_dir}/user_access_matrix.csv")

    # Active sessions
    export_active_sessions(f"{evidence_dir}/active_sessions.csv")

    # Failed login attempts
    export_failed_logins(f"{evidence_dir}/failed_logins.csv")

    # Firewall rules
    export_firewall_rules(f"{evidence_dir}/firewall_rules.txt")

    # Vulnerability scan results
    export_vulnerability_scans(f"{evidence_dir}/vulnerability_scan.json")

    # Security alerts
    export_security_alerts(f"{evidence_dir}/security_alerts.json")

    # Change requests
    export_change_requests(f"{evidence_dir}/change_requests.csv")

    # Backup status
    export_backup_status(f"{evidence_dir}/backup_status.json")
```

**Weekly Automated Collection:**
- Access review certifications
- Patch management reports
- Incident response activity
- SIEM alerts summary
- Configuration drift reports

**Monthly Automated Collection:**
- User provisioning/deprovisioning logs
- Privileged access reviews
- Training completion reports
- Vendor risk assessments
- Capacity planning reports

### 2. Manual Evidence Collection

**Quarterly Collections:**
- Board meeting minutes
- CAB meeting minutes
- Policy review and approval documents
- Penetration test reports
- Internal audit results
- Risk assessment updates

**Annual Collections:**
- SOC 2 audit reports
- ISO 27001 certificates
- Business continuity test results
- Disaster recovery test results
- Third-party audit reports

### 3. Evidence Storage and Retention

```python
class EvidenceManager:
    def __init__(self):
        self.storage_path = "/compliance/evidence"
        self.retention_years = 7

    def store_evidence(self, evidence_type, filename, data):
        # Create directory structure
        date = datetime.now()
        path = f"{self.storage_path}/{date.year}/{date.month:02d}/{evidence_type}"
        os.makedirs(path, exist_ok=True)

        # Encrypt evidence
        encrypted_data = self.encrypt(data)

        # Store evidence
        filepath = f"{path}/{filename}"
        with open(filepath, 'wb') as f:
            f.write(encrypted_data)

        # Log storage
        audit_log.log_evidence_stored(evidence_type, filename, filepath)

        # Index for search
        self.index_evidence(evidence_type, filename, filepath, metadata={
            'collected_at': datetime.now(),
            'size': len(encrypted_data),
            'checksum': hashlib.sha256(encrypted_data).hexdigest()
        })

    def retrieve_evidence(self, evidence_type, date_range):
        # Search index
        results = self.search_index(evidence_type, date_range)

        # Decrypt and return
        evidence_files = []
        for result in results:
            with open(result['filepath'], 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.decrypt(encrypted_data)
            evidence_files.append({
                'filename': result['filename'],
                'data': decrypted_data,
                'metadata': result['metadata']
            })

        # Audit log
        audit_log.log_evidence_retrieved(evidence_type, date_range, len(evidence_files))

        return evidence_files

    def purge_old_evidence(self):
        # Find evidence older than retention period
        cutoff_date = datetime.now() - timedelta(days=self.retention_years * 365)

        old_evidence = self.search_index(date_range=('1900-01-01', cutoff_date))

        for evidence in old_evidence:
            # Securely delete
            secure_delete(evidence['filepath'])

            # Remove from index
            self.remove_from_index(evidence['id'])

            # Audit log
            audit_log.log_evidence_purged(evidence['filepath'])
```

---

## Audit Preparation Guide

### Pre-Audit Checklist (60 Days Before)

**Week 1-2: Evidence Collection**
- [ ] Run automated evidence collection scripts
- [ ] Gather manual evidence (policies, procedures)
- [ ] Collect audit logs (authentication, authorization, changes)
- [ ] Export user access matrix
- [ ] Collect vulnerability scan reports
- [ ] Gather incident response documentation

**Week 3-4: Gap Analysis**
- [ ] Review SOC 2 requirements against current controls
- [ ] Identify any control gaps
- [ ] Create remediation plans for gaps
- [ ] Prioritize remediation activities

**Week 5-6: Remediation**
- [ ] Implement control enhancements
- [ ] Update policies and procedures
- [ ] Complete outstanding training
- [ ] Close open security findings
- [ ] Test new controls

**Week 7-8: Documentation Review**
- [ ] Review all policies for accuracy
- [ ] Update procedures to match current practices
- [ ] Ensure all approvals are documented
- [ ] Verify evidence is complete and accessible

**Week 9-10: Internal Audit**
- [ ] Conduct internal control testing
- [ ] Review findings with management
- [ ] Remediate any issues identified
- [ ] Document remediation

**Week 11-12: Final Preparation**
- [ ] Create audit evidence repository
- [ ] Prepare audit walkthrough materials
- [ ] Schedule audit meetings
- [ ] Assign audit liaisons
- [ ] Conduct audit kickoff

### During Audit

**Day 1: Opening Meeting**
- Overview of GreenLang
- Security architecture presentation
- Control environment walkthrough
- Q&A session

**Days 2-10: Control Testing**
- Provide evidence as requested
- Answer auditor questions
- Facilitate system access
- Coordinate with subject matter experts

**Day 11-12: Management Review**
- Review preliminary findings
- Discuss any observations
- Plan remediation (if needed)

**Day 13-15: Closing**
- Exit meeting
- Receive draft report
- Review for factual accuracy
- Provide management responses

### Post-Audit

**Week 1-2:**
- [ ] Review final SOC 2 report
- [ ] Distribute report to stakeholders
- [ ] Create remediation plan for any findings
- [ ] Update risk register

**Month 2-3:**
- [ ] Implement remediation plan
- [ ] Test remediated controls
- [ ] Update documentation
- [ ] Plan for next audit cycle

---

## Control Implementation Details

*Detailed implementation specifics are covered in each control section above.*

Key Implementation Areas:
1. Authentication and Authorization (CC6)
2. Data Encryption (CC6.7)
3. Network Security (CC6.1, CC6.4)
4. Vulnerability Management (CC6.6)
5. Change Management (CC7.3, CC8.1)
6. Monitoring and Alerting (CC7.2)
7. Incident Response (Integrated throughout)

---

## Testing Procedures

### Control Testing Schedule

**Quarterly Testing:**
- Access controls (CC6.1, CC6.2, CC6.3)
- Authentication (CC6.8)
- Change management (CC7.3)
- Monitoring (CC7.2)

**Semi-Annual Testing:**
- Network security (CC6.1, CC6.4)
- Encryption (CC6.7)
- Vulnerability management (CC6.6)

**Annual Testing:**
- All controls comprehensive review
- Penetration testing
- Disaster recovery testing
- Business continuity testing

### Sample Selection Methodology

**Attribute Sampling:**
- Sample size: 25 items (for large populations)
- Selection: Random selection
- Evaluation: Pass/fail
- Acceptable error rate: 5%

**Variable Sampling:**
- Sample size: Based on population size and desired confidence
- Selection: Stratified random sampling
- Evaluation: Quantitative assessment

---

## Continuous Monitoring Plan

### Real-Time Monitoring

**24/7 Monitoring:**
- Authentication attempts
- Authorization decisions
- System availability
- Security alerts
- Performance metrics
- Error rates

**Automated Alerts:**
```python
# Example: Real-time control monitoring
def monitor_access_control():
    # Monitor failed authentication attempts
    failed_logins = get_failed_logins(minutes=5)
    if len(failed_logins) > 10:
        alert_security_team("Potential brute force attack")

    # Monitor privilege escalations
    role_changes = get_role_changes(minutes=60)
    for change in role_changes:
        if change['new_role'] == 'admin' and not change['approved']:
            alert_security_team(f"Unauthorized privilege escalation: {change['user_id']}")

    # Monitor data access
    data_access = get_data_access_logs(minutes=15)
    for access in data_access:
        if access['data_classification'] == 'highly_confidential':
            verify_authorization(access)
```

### Weekly Reviews

- Security alert summary
- Change management activity
- Access review progress
- Vulnerability scan results
- Incident response activity

### Monthly Reviews

- Control effectiveness metrics
- Compliance KPIs
- Trend analysis
- Risk register updates
- Training completion rates

### Quarterly Reviews

- Comprehensive control testing
- Internal audit
- Risk assessment
- Policy reviews
- Vendor assessments

---

## Appendices

### Appendix A: Control Matrix

| Control ID | Control Name | Category | Owner | Testing Frequency |
|------------|--------------|----------|-------|-------------------|
| CC1.1 | Integrity and Ethics | Control Environment | CISO | Annual |
| CC1.2 | Board Independence | Control Environment | CEO | Annual |
| CC1.3 | Organizational Structure | Control Environment | CTO | Annual |
| CC1.4 | Competence | Control Environment | HR | Quarterly |
| CC1.5 | Accountability | Control Environment | CISO | Quarterly |
| CC2.1 | Information Quality | Communication | CTO | Monthly |
| CC2.2 | Internal Communication | Communication | CISO | Monthly |
| CC2.3 | External Communication | Communication | Legal | Quarterly |
| CC3.1 | Objectives | Risk Assessment | CEO | Quarterly |
| CC3.2 | Risk Identification | Risk Assessment | CISO | Quarterly |
| CC3.3 | Fraud Risk | Risk Assessment | CISO | Annual |
| CC3.4 | Change Assessment | Risk Assessment | CISO | Ongoing |
| CC4.1 | Ongoing Evaluation | Monitoring | CISO | Continuous |
| CC4.2 | Deficiency Communication | Monitoring | CISO | Ongoing |
| CC5.1 | Control Activities | Control Activities | CISO | Quarterly |
| CC5.2 | Technology Controls | Control Activities | CTO | Quarterly |
| CC5.3 | Policies and Procedures | Control Activities | CISO | Annual |
| CC6.1 | Access Controls | Logical Access | Security Team | Quarterly |
| CC6.2 | Credential Issuance | Logical Access | IT | Quarterly |
| CC6.3 | Access Removal | Logical Access | IT | Quarterly |
| CC6.4 | Physical Access | Physical Security | Facilities | Quarterly |
| CC6.5 | Asset Disposal | Physical Security | IT | Quarterly |
| CC6.6 | Vulnerability Management | System Operations | Security Team | Monthly |
| CC6.7 | Data Transmission | Logical Access | Security Team | Quarterly |
| CC6.8 | Authentication | Logical Access | Security Team | Quarterly |
| CC7.1 | Capacity Planning | System Operations | DevOps | Quarterly |
| CC7.2 | Monitoring | System Operations | DevOps | Continuous |
| CC7.3 | Change Management | System Operations | DevOps | Quarterly |
| CC8.1 | Change Authorization | Change Management | CTO | Quarterly |

### Appendix B: Compliance KPIs

**Security Metrics:**
- Mean time to detect (MTTD) security incidents: < 15 minutes
- Mean time to respond (MTTR) to incidents: < 1 hour
- Vulnerability remediation SLA compliance: > 95%
- MFA enrollment rate: 100%
- Failed authentication attempts: < 100/day
- Password policy compliance: 100%

**Access Control Metrics:**
- User access reviews completed on time: 100%
- Orphaned accounts: 0
- Accounts without MFA: 0
- Terminated user access revoked within SLA: 100%
- Privileged access certifications: 100%

**Change Management Metrics:**
- Changes with proper approval: 100%
- Changes with rollback plans: 100%
- Emergency changes: < 5% of total changes
- Change success rate: > 95%
- Unauthorized changes: 0

**Training Metrics:**
- Security awareness training completion: 100%
- New hire security training: 100% (within 1 week)
- Phishing simulation pass rate: > 80%

### Appendix C: Audit Evidence Repository

**Location:** `/compliance/evidence/soc2/`

**Structure:**
```
/compliance/evidence/soc2/
├── CC1_Control_Environment/
│   ├── policies/
│   ├── organization_charts/
│   ├── training_records/
│   └── background_check_reports/
├── CC2_Communication/
│   ├── policy_acknowledgments/
│   ├── meeting_minutes/
│   └── communication_logs/
├── CC3_Risk_Assessment/
│   ├── risk_registers/
│   ├── risk_assessments/
│   └── threat_models/
├── CC4_Monitoring/
│   ├── monitoring_dashboards/
│   ├── deficiency_logs/
│   └── remediation_reports/
├── CC5_Control_Activities/
│   ├── control_documentation/
│   ├── testing_results/
│   └── configuration_files/
├── CC6_Logical_Access/
│   ├── user_access_matrix/
│   ├── authentication_logs/
│   ├── mfa_enrollment/
│   ├── provisioning_logs/
│   ├── deprovisioning_logs/
│   ├── vulnerability_scans/
│   └── encryption_evidence/
├── CC7_System_Operations/
│   ├── capacity_reports/
│   ├── monitoring_logs/
│   ├── siem_alerts/
│   └── change_requests/
└── CC8_Change_Management/
    ├── change_approvals/
    ├── deployment_logs/
    ├── rollback_plans/
    └── cab_minutes/
```

### Appendix D: Acronyms and Definitions

**Acronyms:**
- **SOC 2:** Service Organization Control 2
- **TSC:** Trust Service Criteria
- **CC:** Common Criteria
- **MFA:** Multi-Factor Authentication
- **RBAC:** Role-Based Access Control
- **SIEM:** Security Information and Event Management
- **IDS:** Intrusion Detection System
- **MTTR:** Mean Time to Remediate
- **MTTD:** Mean Time to Detect
- **CAB:** Change Approval Board
- **SDLC:** Software Development Lifecycle
- **CI/CD:** Continuous Integration/Continuous Deployment

**Definitions:**
- **Control:** A safeguard or countermeasure designed to protect the confidentiality, integrity, and availability of information.
- **Evidence:** Documentation or records that demonstrate the operating effectiveness of controls.
- **Service Auditor:** Independent third party who examines and reports on controls at a service organization.
- **Trust Service Criteria:** Criteria used to evaluate controls relevant to security, availability, processing integrity, confidentiality, and privacy.

---

## Document Control

**Version History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | TEAM 4 | Initial document creation |

**Review Schedule:**
- **Next Review:** 2026-02-08 (Quarterly)
- **Annual Review:** 2026-11-08

**Approval:**
- **Prepared by:** Security & Compliance Team
- **Reviewed by:** CISO
- **Approved by:** CTO

---

**END OF DOCUMENT**
