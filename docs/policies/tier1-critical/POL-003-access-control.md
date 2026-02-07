# POL-003: Access Control Policy

| Document Control | |
|------------------|---|
| **Policy ID** | POL-003 |
| **Title** | Access Control Policy |
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

This Access Control Policy establishes requirements for managing access to GreenLang information systems, applications, data, and physical facilities. The policy ensures that access is granted based on business need, appropriate authorization is obtained, and access rights are regularly reviewed and revoked when no longer required.

This policy supports:

- Protection of sensitive data and systems from unauthorized access
- Compliance with SOC 2, ISO 27001, GDPR, and industry regulations
- Implementation of the principle of least privilege
- Maintenance of audit trails for access decisions
- Timely revocation of access upon role change or termination

---

## 2. Scope

### 2.1 Applicability

This policy applies to:

- **Personnel**: All employees, contractors, consultants, temporary workers, and third parties requiring access to GreenLang systems
- **Systems**: All information systems, applications, databases, networks, cloud services, and infrastructure
- **Data**: All data classified per POL-002 Data Classification Policy
- **Facilities**: All GreenLang physical locations and data centers

### 2.2 Access Types Covered

- Logical access (user accounts, application access, database access)
- Privileged access (administrative accounts, root/sudo access)
- Remote access (VPN, SSH, remote desktop)
- API access (service accounts, API keys, tokens)
- Physical access (building entry, data center access)

---

## 3. Policy Statement

### 3.1 Access Control Principles

All access control decisions and implementations shall adhere to these principles:

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Need-to-Know** | Access granted only when required for job function | Access requests must include business justification |
| **Least Privilege** | Minimum access rights necessary to perform duties | Default deny; explicit grants required |
| **Separation of Duties** | Critical functions divided among multiple individuals | No single person can complete high-risk transactions |
| **Defense in Depth** | Multiple layers of access controls | Network + application + data layer controls |
| **Accountability** | All access actions attributable to individuals | Unique accounts; no shared credentials |

### 3.2 Access Control Model

GreenLang implements Role-Based Access Control (RBAC) as the primary access control model, supplemented by Attribute-Based Access Control (ABAC) for fine-grained authorization decisions.

**Reference Implementation:** See `greenlang/infrastructure/rbac_service/` for the technical implementation of RBAC including:
- Role hierarchy and inheritance
- Permission evaluation pipeline
- Tenant isolation
- Just-in-time access provisioning

---

## 4. User Account Management

### 4.1 Account Provisioning

**Process Overview:**

```
Request -> Manager Approval -> Security Review -> Account Creation -> Training -> Access Granted
```

**Provisioning Requirements:**

| Step | Requirement | SLA | Owner |
|------|-------------|-----|-------|
| Request Submission | Ticket with business justification | N/A | Requesting manager |
| Manager Approval | Verify business need and role appropriateness | 24 hours | Direct manager |
| Security Review | Verify role alignment and compliance | 48 hours | Security team |
| Account Creation | Create account with assigned roles | 24 hours | IT Operations |
| Training Completion | Complete security awareness training | 30 days | New user |
| Access Activation | Enable full access upon training completion | Immediate | Automated |

**Required Information:**

- Full legal name
- Job title and department
- Direct manager
- Start date
- Role(s) requested with justification
- Systems/applications required
- Data classification access level needed
- Duration (permanent or temporary)

### 4.2 Account Modification

**Triggers for Modification:**

- Role or job function change
- Department transfer
- Project assignment/completion
- Promotion or demotion
- Temporary elevated access requirement

**Modification Process:**

1. Manager submits modification request with justification
2. Security reviews for appropriate access levels
3. Current access is reviewed against new role requirements
4. Unnecessary access is removed
5. New access is provisioned
6. User notified of changes
7. Audit log updated

**SLA:** Access modifications completed within 48 hours of approved request.

### 4.3 Account Deprovisioning

**Termination Types and Response:**

| Termination Type | Access Revocation SLA | Process |
|-----------------|----------------------|---------|
| Voluntary - Standard | Within 24 hours of last day | Scheduled revocation |
| Voluntary - Immediate | Same day | Expedited revocation |
| Involuntary - Standard | Immediate upon notification | Emergency revocation |
| Involuntary - Cause | Immediate upon HR notification | Emergency revocation |
| Contractor End | Within 24 hours of contract end | Scheduled revocation |
| Third-Party Termination | Within 24 hours of notification | Coordinated revocation |

**Deprovisioning Checklist:**

- [ ] Disable Active Directory / SSO account
- [ ] Revoke VPN access
- [ ] Disable email and collaboration tools
- [ ] Revoke cloud service access (AWS, GCP, Azure)
- [ ] Revoke source code repository access
- [ ] Revoke database access
- [ ] Revoke API keys and tokens
- [ ] Remove from distribution lists and groups
- [ ] Disable physical access badges
- [ ] Recover company equipment
- [ ] Forward email (if approved by legal)
- [ ] Archive user data per retention policy
- [ ] Document deprovisioning completion

### 4.4 Account Standards

**Unique Identification:**

- Each user receives a unique user ID
- Format: `firstname.lastname` or `first.last.N` for duplicates
- No shared accounts except where technically required (with CISO approval)
- Service accounts clearly identified with naming convention: `svc-<application>-<function>`

**Account Types:**

| Type | Naming Convention | Use Case | Authentication |
|------|------------------|----------|----------------|
| Standard User | `firstname.lastname` | Day-to-day operations | SSO + MFA |
| Privileged User | `firstname.lastname-admin` | Administrative tasks | SSO + MFA + JIT |
| Service Account | `svc-<app>-<function>` | Automated processes | Certificate/API key |
| Emergency Account | `break-glass-<N>` | Emergency access only | Secure password + audit |

---

## 5. Authentication Standards

### 5.1 Multi-Factor Authentication (MFA)

**MFA Requirements:**

| Access Type | MFA Required | Approved Methods |
|-------------|--------------|------------------|
| Remote access (VPN) | Yes | TOTP, WebAuthn, Push notification |
| Cloud console access | Yes | TOTP, WebAuthn, Push notification |
| Production systems | Yes | TOTP, WebAuthn, Hardware key |
| Privileged access | Yes | WebAuthn, Hardware key preferred |
| Email (external) | Yes | TOTP, Push notification |
| Internal applications | Risk-based | TOTP, WebAuthn |

**Approved MFA Methods (in order of preference):**

1. **WebAuthn/FIDO2 Hardware Key** - Highest security, phishing-resistant
2. **WebAuthn/FIDO2 Platform Authenticator** - Built-in biometrics (Touch ID, Windows Hello)
3. **TOTP Authenticator App** - Google Authenticator, Authy, Microsoft Authenticator
4. **Push Notification** - Okta Verify, Duo Push (with number matching)
5. **SMS** - Only as fallback, with documented risk acceptance

**Prohibited:**
- SMS as primary MFA for privileged accounts
- Email-based OTP for sensitive systems
- Security questions as sole second factor

### 5.2 Single Sign-On (SSO) Integration

**Requirements:**

- All business applications must integrate with SSO (Okta)
- SAML 2.0 or OIDC protocols required
- Session timeout: 8 hours for standard, 1 hour for privileged
- Re-authentication required for sensitive operations

**Exceptions:**

Applications that cannot support SSO must:
- Receive CISO exception approval
- Implement strong local authentication
- Integrate with centralized logging
- Be reviewed annually for SSO capability

### 5.3 Password Standards

Password requirements are defined in POL-009 Password and Authentication Policy. Summary:

| Account Type | Minimum Length | Complexity | Expiration | History |
|--------------|----------------|------------|------------|---------|
| Standard | 14 characters | Mixed case, numbers, symbols | 90 days | 12 passwords |
| Privileged | 20 characters | Mixed case, numbers, symbols | 60 days | 24 passwords |
| Service | 32 characters | Randomly generated | 90 days | N/A |

---

## 6. Authorization and RBAC Standards

### 6.1 Role-Based Access Control (RBAC)

GreenLang implements RBAC with the following structure:

**Role Hierarchy:**

```
Super Admin
    |
    +-- Tenant Admin
    |       |
    |       +-- Department Manager
    |       |       |
    |       |       +-- Team Lead
    |       |               |
    |       |               +-- User
    |       |
    |       +-- Auditor (read-only)
    |
    +-- System Roles
            |
            +-- Security Admin
            +-- Platform Admin
            +-- Support Agent
```

**Standard Roles:**

| Role | Description | Typical Permissions |
|------|-------------|---------------------|
| `viewer` | Read-only access to resources | Read reports, view dashboards |
| `user` | Standard user access | Create/edit own resources, submit data |
| `analyst` | Data analysis access | Run queries, create reports, export data |
| `manager` | Team management | Approve requests, view team data, manage users |
| `admin` | Full administrative access | All permissions within tenant |
| `auditor` | Compliance audit access | Read-only access to audit logs and configurations |
| `security_admin` | Security administration | Manage security settings, review access |
| `platform_admin` | Platform-wide administration | Infrastructure management |
| `super_admin` | Unrestricted access | All system permissions |

**Role Assignment Approval:**

| Role Level | Approver | Additional Requirements |
|------------|----------|------------------------|
| Viewer, User | Direct Manager | None |
| Analyst, Manager | Director + Security | Business justification |
| Admin | VP + Security | Background check verification |
| Platform/Security Admin | CISO | Enhanced screening + training |
| Super Admin | CEO + CISO | Board notification |

### 6.2 Permission Model

Permissions follow the format: `resource:action`

**Permission Categories:**

| Resource | Actions | Example |
|----------|---------|---------|
| emissions | read, write, delete, export | `emissions:write` |
| reports | read, create, approve, submit | `reports:approve` |
| users | read, create, update, delete | `users:delete` |
| roles | read, assign, manage | `roles:manage` |
| audit_logs | read | `audit_logs:read` |
| settings | read, update | `settings:update` |

**Permission Evaluation:**

1. Check user-specific denies (deny wins)
2. Check user-specific grants
3. Check role permissions (inherited)
4. Check role hierarchy (parent roles)
5. Default deny if no explicit grant

### 6.3 Tenant Isolation

Multi-tenant data isolation requirements:

- All data access scoped to user's tenant
- Cross-tenant access prohibited except for platform admins
- Tenant context validated at API and database layers
- Audit logging includes tenant identifier

---

## 7. Privileged Access Management

### 7.1 Privileged Account Categories

| Category | Examples | Controls |
|----------|----------|----------|
| System Administrator | Root, sudo, local admin | JIT, session recording, quarterly review |
| Database Administrator | DBA accounts, schema owners | JIT, query logging, approval workflow |
| Cloud Administrator | AWS root, IAM admins | MFA, CloudTrail, break-glass only |
| Security Administrator | SIEM admins, firewall admins | Dual approval, enhanced monitoring |
| Application Administrator | App admin consoles | Role-specific, audit logging |

### 7.2 Just-In-Time (JIT) Access

**JIT Requirements:**

- Privileged access granted for limited duration (default: 4 hours, max: 8 hours)
- Access request includes: justification, target systems, duration
- Approval required from: manager + security (or automated for pre-approved tasks)
- Access automatically revoked upon expiration
- All JIT sessions logged and auditable

**JIT Workflow:**

```
Request -> Risk Assessment -> Approval -> Provisioning -> Session -> Auto-Revoke
   |            |                |            |            |           |
   v            v                v            v            v           v
 Ticket    Risk Score     Manager/Auto   Grant Access   Activity    Remove
          Calculated       Approval       Elevated       Logged     Access
```

### 7.3 Session Recording

**Requirements:**

- All privileged access sessions recorded
- Recordings stored for 90 days minimum
- Recordings encrypted at rest
- Access to recordings limited to Security and Audit teams
- Random session reviews conducted monthly

**Recorded Activities:**

- SSH sessions to production servers
- Database admin sessions
- Cloud console sessions (via screen recording)
- Local administrator sessions
- API calls with privileged tokens

### 7.4 Quarterly Privileged Access Review

**Review Process:**

1. Security generates list of all privileged accounts
2. Account owners verify continued need for each privilege
3. Unused or unnecessary privileges removed
4. Orphan accounts disabled
5. Review documented with attestation
6. Exceptions escalated to CISO

**Review Cadence:**

| Account Type | Review Frequency | Reviewer |
|--------------|-----------------|----------|
| Super Admin | Monthly | CEO + CISO |
| Platform Admin | Quarterly | CTO + CISO |
| Security Admin | Quarterly | CISO |
| Tenant Admin | Quarterly | Department VP |
| Service Accounts | Semi-annually | Application owner |

---

## 8. Remote Access and VPN

### 8.1 Remote Access Requirements

| Requirement | Standard |
|-------------|----------|
| VPN Client | Approved corporate VPN client only |
| Authentication | SSO + MFA required |
| Endpoint Security | EDR agent installed and active |
| Device Compliance | Company-managed or MDM-enrolled device |
| Split Tunneling | Disabled for access to internal resources |
| Session Timeout | 8 hours; 30-minute idle timeout |

### 8.2 VPN Access Levels

| Level | Access Scope | Requirements |
|-------|--------------|--------------|
| Standard | Internal network, business apps | VPN + MFA |
| Development | Development networks, non-prod | VPN + MFA + Dev role |
| Production | Production networks | VPN + MFA + JIT approval |
| Administrative | Infrastructure management | VPN + MFA + JIT + session recording |

### 8.3 Remote Access Security

- VPN logs retained for 90 days
- Geographic access restrictions (configurable by tenant)
- Concurrent session limits (max 2 sessions per user)
- Connection from high-risk countries requires additional verification
- Automatic disconnection for policy violations

---

## 9. Physical Access Control

### 9.1 Facility Access Zones

| Zone | Description | Access Method | Authorization |
|------|-------------|---------------|---------------|
| Public | Lobby, reception | Open during business hours | None |
| General Office | Workspaces, meeting rooms | Badge | All employees |
| Restricted | Executive offices, HR, Finance | Badge + PIN | Role-based |
| Secure | Server rooms, network closets | Badge + Biometric | IT/Facilities only |
| Data Center | Co-located equipment | Badge + Biometric + Escort | Authorized personnel only |

### 9.2 Badge Management

**Badge Issuance:**

- Issued on first day with photo verification
- Linked to employee record
- Access levels configured per role
- Temporary badges for visitors and contractors

**Badge Revocation:**

- Immediate upon termination
- Same-day collection for involuntary termination
- Badge deactivated in access control system
- Lost/stolen badges deactivated immediately

### 9.3 Visitor Management

- All visitors signed in at reception
- Visitor badge issued (no tailgating)
- Escort required for Restricted and Secure zones
- Visitor log retained for 90 days
- After-hours visits require pre-approval

---

## 10. Access Review and Recertification

### 10.1 Review Schedule

| Access Type | Review Frequency | Reviewer | Documentation |
|-------------|-----------------|----------|---------------|
| Standard user access | Semi-annually | Direct manager | Attestation form |
| Privileged access | Quarterly | Manager + Security | Detailed review |
| Third-party access | Quarterly | Vendor manager + Security | Access verification |
| Service accounts | Semi-annually | Application owner | Account justification |
| Physical access | Annually | Facilities + Security | Badge audit |

### 10.2 Review Process

1. **Initiation**: Automated notification sent to reviewers 14 days before due date
2. **Data Gathering**: Access reports generated for review scope
3. **Review**: Reviewer certifies appropriate access or requests changes
4. **Remediation**: Inappropriate access removed within 7 days
5. **Attestation**: Reviewer signs off on completed review
6. **Documentation**: Review results stored for audit purposes

### 10.3 Access Recertification Requirements

Reviewers must verify:

- User still employed/contracted
- Role still requires the access
- Access level appropriate for classification
- No segregation of duties conflicts
- Access aligned with least privilege

---

## 11. Emergency Access Procedures

### 11.1 Break-Glass Access

**Definition:** Emergency access to critical systems when normal access is unavailable.

**Authorized Scenarios:**

- Production system outage requiring immediate response
- Security incident requiring elevated access
- Disaster recovery operations
- Critical business continuity event

**Break-Glass Procedure:**

1. Attempt normal access channels first
2. Contact Security or IT on-call
3. If unavailable, use break-glass account
4. Document emergency justification
5. Notify Security within 1 hour
6. Complete incident report within 24 hours
7. Post-incident review conducted

**Break-Glass Account Controls:**

- Stored in secure password vault (separate from daily operations)
- Requires two authorized personnel to access (dual custody)
- Triggers immediate alert to Security
- Session fully recorded
- Password rotated after each use
- Quarterly testing to ensure functionality

### 11.2 Emergency Access Audit

All break-glass access is reviewed:

- Within 24 hours by Security team
- Documented in incident management system
- Unauthorized use escalated immediately
- Quarterly summary to Security Council

---

## 12. Exceptions

### 12.1 Exception Categories

| Category | Example | Approval |
|----------|---------|----------|
| Shared Account | Legacy system requiring shared credentials | CISO + compensating controls |
| Extended Access | Contractor requires 12-month access | VP + Security + quarterly review |
| Bypass MFA | Legacy application cannot support MFA | CISO + enhanced monitoring |
| Remote Access from Restricted Country | Business travel to high-risk location | Security pre-approval |

### 12.2 Exception Requirements

All exceptions must include:

- Business justification
- Risk assessment
- Compensating controls
- Defined expiration date
- Monitoring plan
- Owner and reviewer assignments

See POL-001 Section 6 for full exception process.

---

## 13. Enforcement

### 13.1 Compliance Monitoring

- Automated monitoring of access attempts and anomalies
- User behavior analytics (UEBA) for privilege misuse detection
- Failed authentication monitoring and alerting
- Periodic access reviews and audits
- Penetration testing of access controls

### 13.2 Violations

Violations are handled per POL-001 Section 9.2.

**Examples of Violations:**

- Sharing credentials or access tokens
- Accessing systems without authorization
- Circumventing access controls
- Failure to report compromised credentials
- Unauthorized privilege escalation

---

## 14. Definitions

| Term | Definition |
|------|------------|
| **ABAC** | Attribute-Based Access Control - authorization based on user, resource, and environmental attributes |
| **Authentication** | Process of verifying identity |
| **Authorization** | Process of granting access rights |
| **Break-Glass** | Emergency access procedure bypassing normal controls |
| **JIT** | Just-In-Time access provisioning for elevated privileges |
| **MFA** | Multi-Factor Authentication - requiring multiple verification methods |
| **Privileged Access** | Access rights that exceed standard user permissions |
| **RBAC** | Role-Based Access Control - authorization based on assigned roles |
| **SSO** | Single Sign-On - one authentication for multiple applications |
| **Tenant** | Isolated organizational unit in multi-tenant environment |

---

## 15. Related Documents

| Document | Location |
|----------|----------|
| POL-001: Information Security Policy | `/docs/policies/tier1-critical/` |
| POL-002: Data Classification Policy | `/docs/policies/tier1-critical/` |
| POL-005: Personnel Security Policy | `/docs/policies/tier1-critical/` |
| POL-009: Password and Authentication Policy | `/docs/policies/tier2-high/` |
| RBAC Service Implementation | `greenlang/infrastructure/rbac_service/` |
| RBAC Database Migration | `deployment/database/migrations/sql/V010__rbac_authorization.sql` |

---

## 16. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-06 | CISO | Initial policy creation |

---

## 17. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Chief Information Security Officer | | | |
| Chief Technology Officer | | | |
| VP of Human Resources | | | |

---

**Document Classification: Internal**

**Annual Review Commitment**: This policy shall be reviewed at least annually, or more frequently when significant changes occur in access control technologies, regulatory requirements, or organizational structure.
