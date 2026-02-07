# POL-009: Password and Authentication Policy

## Document Control

| Field | Value |
|-------|-------|
| Policy ID | POL-009 |
| Version | 1.0 |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |
| Owner | Chief Information Security Officer (CISO) |
| Approver | Chief Technology Officer (CTO) |
| Classification | Internal |

---

## 1. Purpose

This Password and Authentication Policy establishes standards for creating, managing, and protecting authentication credentials across GreenLang systems. Strong authentication is a critical defense against unauthorized access and data breaches.

This policy ensures that:
- Passwords and credentials meet strength requirements
- Multi-factor authentication (MFA) protects sensitive access
- Service accounts and API keys are properly managed
- Credentials are stored securely using industry-standard methods
- Account security controls protect against brute force attacks
- Password recovery processes verify identity appropriately

---

## 2. Scope

### 2.1 Applicability

This policy applies to:
- All GreenLang employees, contractors, and third parties
- All systems, applications, and services requiring authentication
- All types of authentication credentials (passwords, tokens, keys, certificates)

### 2.2 Covered Systems

This policy covers authentication to:
- Corporate identity provider (Azure AD/Okta)
- Production and non-production environments
- Cloud provider consoles (AWS, Azure, GCP)
- Development tools (GitHub, CI/CD systems)
- Internal applications and services
- Customer-facing applications
- Network devices and infrastructure
- Databases and data stores
- Third-party SaaS applications

---

## 3. Policy Statement

### 3.1 Password Complexity Requirements

#### 3.1.1 Standard User Passwords

All standard user passwords must meet the following requirements:

| Requirement | Standard |
|-------------|----------|
| **Minimum Length** | 14 characters |
| **Maximum Length** | 128 characters (no artificial limits) |
| **Character Mix** | At least 3 of 4 types: uppercase, lowercase, numbers, special characters |
| **No Dictionary Words** | Common words blocked via dictionary check |
| **No Personal Information** | Username, email, name variations prohibited |
| **No Common Patterns** | Keyboard patterns (qwerty), repetition (aaa), sequences (123) blocked |
| **No Previously Breached** | Checked against known breach databases (HIBP) |

#### 3.1.2 Privileged Account Passwords

Privileged accounts (administrators, root, service accounts with elevated access) have enhanced requirements:

| Requirement | Standard |
|-------------|----------|
| **Minimum Length** | 20 characters |
| **Character Mix** | All 4 types required |
| **Uniqueness** | Must be unique per system/service |
| **Storage** | Must be stored in approved secrets vault |
| **Access** | Just-in-time access, session recording |

#### 3.1.3 Passphrase Alternative

Passphrases are encouraged as an alternative to complex passwords:
- Minimum 20 characters
- Multiple random words separated by spaces or symbols
- No famous quotes, song lyrics, or easily guessable phrases
- Example: `correct horse battery staple`

### 3.2 Password History

#### 3.2.1 History Requirements

| Account Type | Password History |
|--------------|------------------|
| Standard users | Last 12 passwords cannot be reused |
| Privileged accounts | Last 24 passwords cannot be reused |
| Service accounts | Each rotation must be unique |

#### 3.2.2 Similarity Prevention

New passwords must not be:
- Identical to any of the last 12 passwords
- A minor variation (adding numbers, changing case only)
- A rotation pattern (Password1, Password2, Password3)

### 3.3 Password Expiration

#### 3.3.1 Expiration Schedule

| Account Type | Maximum Age | Forced Change |
|--------------|-------------|---------------|
| Standard users | 90 days | After expiration |
| Privileged accounts | 60 days | After expiration |
| Service accounts | 90 days | Automated rotation |
| API keys | 90 days | Automated rotation |
| Certificates | Per certificate policy | Before expiration |

#### 3.3.2 Expiration Notifications

- 14 days before expiration: Email notification
- 7 days before expiration: Email + portal reminder
- 3 days before expiration: Email + mandatory prompt
- Day of expiration: Access blocked until changed

#### 3.3.3 NIST 800-63B Alignment

Following NIST guidance, password expiration may be extended or removed if:
- MFA is universally enforced
- Continuous authentication monitoring is in place
- Password breach detection triggers immediate reset
- Risk-based authentication is implemented

### 3.4 Multi-Factor Authentication Requirements

#### 3.4.1 MFA Requirement Matrix

| Access Type | MFA Required |
|-------------|--------------|
| Remote access (VPN, web apps) | **Required** |
| Privileged access (admin, root) | **Required** |
| Production system access | **Required** |
| Cloud console access | **Required** |
| Code repository access | **Required** |
| Customer data access | **Required** |
| Internal office network (on-premises) | Recommended |
| Non-sensitive internal tools | Recommended |

#### 3.4.2 Approved MFA Methods

Listed in order of preference (most secure to acceptable):

| Method | Security Level | Use Case |
|--------|---------------|----------|
| **Hardware security key (WebAuthn/FIDO2)** | Highest | Privileged users, high-risk roles |
| **TOTP authenticator app** | High | Standard users (Google Authenticator, Authy, 1Password) |
| **Push notification** | High | Standard users (Okta Verify, Duo, Microsoft Authenticator) |
| **SMS OTP** | Moderate | Backup method only, not primary |
| **Email OTP** | Low | Emergency recovery only |

#### 3.4.3 Prohibited MFA Methods

- Security questions (easily researched/guessed)
- Voice calls to VoIP numbers (spoofing risk)
- SMS as primary method (SIM swap risk)
- Printed backup codes as ongoing method (single factor)

#### 3.4.4 MFA Enrollment

- New users must enroll in MFA within 24 hours of account creation
- Users must register at least two MFA methods (primary + backup)
- Enrollment verified before access to sensitive systems granted
- Annual verification of MFA contact information

### 3.5 Service Account Credentials

#### 3.5.1 No Shared Passwords

- Each service account must have a unique password
- Service accounts must not share credentials across services
- Service account credentials must not be used by humans for interactive access
- Generic accounts (admin, root, service) must have unique passwords per system

#### 3.5.2 Service Account Password Requirements

| Requirement | Standard |
|-------------|----------|
| **Minimum Length** | 32 characters (machine-generated) |
| **Character Set** | Alphanumeric + symbols |
| **Generation** | Cryptographically random (CSPRNG) |
| **Uniqueness** | Unique per service, environment, and purpose |
| **Human Memorability** | Not required (stored in vault) |

#### 3.5.3 Rotation Every 90 Days

- Service account passwords rotated every 90 days minimum
- Automated rotation through secrets management (HashiCorp Vault)
- Zero-downtime rotation with dual credential support
- Rotation logged and auditable

#### 3.5.4 Vault Storage Required

All service account credentials must be stored in HashiCorp Vault:
- No credentials in code, configuration files, or environment variables
- No credentials in CI/CD pipeline definitions
- Secrets retrieved at runtime via Vault API or agent injection
- Access controlled via Vault policies and audit logged

Reference: PRD-SEC-006: Secrets Management

### 3.6 API Key Management

#### 3.6.1 API Key Standards

| Requirement | Standard |
|-------------|----------|
| **Key Length** | Minimum 32 characters |
| **Key Format** | Cryptographically random, Base64 or hex encoded |
| **Scope** | Minimum permissions required (least privilege) |
| **Expiration** | Maximum 90 days, shorter for sensitive APIs |
| **Rotation** | Before expiration, automated where possible |

#### 3.6.2 API Key Lifecycle

1. **Request:** Developer requests API key via self-service portal
2. **Approval:** Automatic for standard, manager approval for sensitive
3. **Generation:** System generates key, never displayed again after creation
4. **Storage:** Developer stores in Vault or approved secrets manager
5. **Rotation:** Automated rotation before expiration
6. **Revocation:** Immediate revocation upon incident or termination

#### 3.6.3 API Key Restrictions

- API keys must not be embedded in client-side code
- API keys must not be committed to version control
- API keys must be transmitted only over TLS
- API keys must be scoped to specific IP ranges where possible
- API keys must have rate limits applied

Reference: PRD-SEC-006: Secrets Management, API Gateway configuration

### 3.7 Password Storage Standards

#### 3.7.1 Approved Hashing Algorithms

User passwords must be hashed using approved algorithms:

| Algorithm | Parameters | Use Case |
|-----------|------------|----------|
| **Argon2id** | m=65536, t=3, p=4 (memory 64MB, iterations 3, parallelism 4) | Preferred for all new implementations |
| **bcrypt** | Cost factor 12 minimum | Acceptable for existing systems |
| **PBKDF2-HMAC-SHA256** | 600,000 iterations minimum | Legacy systems only |

#### 3.7.2 Prohibited Methods

The following are prohibited for password storage:
- Plain text storage
- MD5, SHA-1, SHA-256 (without key stretching)
- Single-iteration hashing
- Reversible encryption
- Custom/proprietary algorithms

#### 3.7.3 Salting Requirements

- Each password hash must use a unique, random salt
- Salt length: Minimum 16 bytes (128 bits)
- Salt generation: Cryptographically secure random number generator
- Salt storage: Stored alongside the hash (standard practice for Argon2/bcrypt)

#### 3.7.4 Peppering (Optional Enhancement)

A secret pepper may be applied before hashing:
- Pepper stored separately from the password database
- Pepper at least 32 bytes, stored in HSM or Vault
- Rotation possible without re-hashing (by trying multiple peppers)

### 3.8 Account Lockout

#### 3.8.1 Lockout Thresholds

| Metric | Standard |
|--------|----------|
| **Failed Attempts** | 5 consecutive failures |
| **Lockout Duration** | 15 minutes (automatic unlock) |
| **Lockout Counter Reset** | After successful login |
| **Permanent Lockout** | After 15 failures in 24 hours (manual unlock required) |

#### 3.8.2 Lockout Notifications

- User notified of lockout via email
- Security team alerted for repeated lockouts (potential attack)
- Account owner notified of permanent lockout
- Login attempts logged for forensic analysis

#### 3.8.3 Lockout Bypass Prevention

- Rate limiting applied at network and application layers
- CAPTCHA after 3 failed attempts
- IP-based throttling for distributed attacks
- Credential stuffing detection and blocking

### 3.9 Password Recovery

#### 3.9.1 Identity Verification Requirements

Password recovery must verify identity through:
- Multi-factor verification (email + phone, or MFA device)
- Recovery codes (previously generated and stored securely by user)
- Manager verification for high-privilege accounts
- Identity verification questions (last resort, supplementary only)

#### 3.9.2 Recovery Process

1. User initiates recovery request
2. System sends verification code to registered email
3. User provides code + secondary verification (phone/backup MFA)
4. Temporary password link sent (expires in 1 hour)
5. User sets new password meeting all requirements
6. All active sessions invalidated
7. MFA re-enrollment required for privileged accounts

#### 3.9.3 Recovery Restrictions

- Recovery links valid for 1 hour maximum
- Recovery links single-use only
- Failed recovery attempts trigger lockout
- Recovery requests logged and auditable
- Help desk cannot bypass MFA without manager approval

#### 3.9.4 Help Desk Verification

For help desk-assisted recovery:
- Verify employee ID and manager name
- Callback to registered phone number
- Verification of recent activity or knowledge
- Manager approval for privileged accounts
- All interactions logged with timestamps

---

## 4. Roles and Responsibilities

### 4.1 All Users

- Create and maintain strong, unique passwords
- Enroll in and use MFA as required
- Never share passwords or credentials
- Report suspected compromised credentials immediately
- Use approved password managers
- Protect recovery codes securely

### 4.2 Managers

- Ensure team members complete authentication training
- Approve access requests requiring manager authorization
- Verify identity for help desk recovery requests
- Monitor team compliance with authentication policies

### 4.3 Information Security

- Define authentication standards and requirements
- Manage identity provider configuration
- Monitor for credential compromise indicators
- Investigate authentication-related incidents
- Review and approve MFA method exceptions

### 4.4 IT Operations

- Configure systems to enforce password policies
- Manage identity provider infrastructure
- Support password resets and MFA enrollment
- Maintain authentication audit logs
- Implement account lockout and recovery procedures

### 4.5 Development Teams

- Implement approved password hashing in applications
- Integrate with corporate identity provider (SSO)
- Never store credentials in code or configuration
- Use Vault for service account management
- Follow secure coding standards for authentication

---

## 5. Procedures

### 5.1 Password Creation

1. Access password change page in identity provider
2. Enter current password (or recovery code if resetting)
3. Enter new password meeting all requirements
4. System validates against policy rules
5. System checks against breach database
6. Confirm new password
7. Password updated and previous sessions invalidated

### 5.2 MFA Enrollment

1. Log into identity provider portal
2. Navigate to Security Settings > MFA
3. Select primary MFA method
4. Follow enrollment wizard (scan QR code, register device)
5. Complete verification to confirm enrollment
6. Register backup MFA method
7. Download and secure backup codes

### 5.3 Service Account Creation

1. Submit service account request with business justification
2. Security reviews and approves request
3. IT creates account in Active Directory / IAM
4. Service account password generated and stored in Vault
5. Access policies configured in Vault
6. Application configured to retrieve credentials from Vault
7. Rotation schedule configured

### 5.4 Credential Compromise Response

1. User reports or system detects potential compromise
2. Immediately reset password for affected account
3. Revoke all active sessions
4. Require MFA re-enrollment
5. Review recent account activity
6. Notify Security team for investigation
7. Check for lateral movement or data access

---

## 6. Exceptions

### 6.1 Exception Criteria

Exceptions may be considered for:
- Legacy systems unable to support policy requirements
- Third-party systems with fixed authentication methods
- Regulatory or contractual requirements specifying different standards
- Accessibility accommodations for users with disabilities

### 6.2 Exception Process

1. Submit exception request via GRC portal
2. Document business justification
3. Identify compensating controls
4. Security assesses risk
5. CISO approves or denies exception
6. Exceptions documented and reviewed annually
7. Remediation plan required for legacy system exceptions

### 6.3 Non-Negotiable Requirements

No exceptions granted for:
- MFA requirement for privileged access
- Password storage using approved algorithms
- Prohibition of credential sharing
- Vault requirement for service accounts
- Breach database checking

---

## 7. Enforcement

### 7.1 Technical Enforcement

Authentication policy enforced through:
- Identity provider configuration (Azure AD/Okta)
- Password policy group policy objects (GPO)
- Application-level validation
- Vault policy enforcement for service accounts
- CI/CD pipeline secret scanning

### 7.2 Non-Compliance Consequences

Violations of this policy may result in:
- Forced password reset
- Account suspension pending review
- Access revocation for repeated violations
- Disciplinary action per HR policy
- Termination for intentional credential sharing

### 7.3 Metrics and Reporting

Track and report monthly:
- MFA enrollment rate (target: 100%)
- Password policy compliance rate
- Account lockout frequency
- Average time to recover from lockout
- Service account rotation compliance
- Credential exposure incidents

---

## 8. Related Documents

| Document | Description |
|----------|-------------|
| POL-001: Information Security Policy | Master security policy |
| POL-003: Access Control Policy | Access management standards |
| POL-006: Acceptable Use Policy | Credential protection requirements |
| PRD-SEC-001: JWT Authentication Service | Token-based authentication |
| PRD-SEC-002: RBAC Authorization Layer | Role-based access control |
| PRD-SEC-006: Secrets Management | Vault deployment and usage |
| Identity Provider Admin Guide | IdP configuration procedures |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **MFA** | Multi-Factor Authentication - requiring multiple verification methods |
| **TOTP** | Time-based One-Time Password - algorithm generating time-limited codes |
| **WebAuthn/FIDO2** | Web Authentication standard for hardware security keys |
| **Argon2id** | Password hashing algorithm resistant to GPU and side-channel attacks |
| **bcrypt** | Password hashing function with built-in salting |
| **Salt** | Random data added to password before hashing to prevent rainbow table attacks |
| **Pepper** | Secret key added to password before hashing, stored separately |
| **Credential Stuffing** | Attack using breached credentials from other sites |
| **CSPRNG** | Cryptographically Secure Pseudo-Random Number Generator |
| **HIBP** | Have I Been Pwned - database of breached credentials |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | CISO | Initial policy release |

---

## Appendix A: Password Strength Examples

**Weak Passwords (Rejected):**
- `password123` - Common word + simple numbers
- `GreenLang2026` - Company name + year
- `John.Smith1` - Personal info pattern
- `qwerty!@#` - Keyboard pattern
- `P@ssw0rd` - Common substitution pattern

**Strong Passwords (Accepted):**
- `Tr0ub4dor&3#xK9m` - 16 chars, all character types
- `correct-horse-battery-staple` - 28 char passphrase
- `7hX#mK9$pL2@nQ5&` - 16 chars, random
- `My.Coffee.Is.Always.Too.Hot.2026!` - 33 char memorable passphrase

---

## Appendix B: MFA Quick Reference

| Method | Setup | Usage |
|--------|-------|-------|
| **Authenticator App** | Scan QR code with app | Enter 6-digit code at login |
| **Hardware Key** | Register key in account settings | Touch key when prompted |
| **Push Notification** | Install app, link to account | Approve on phone when prompted |
| **Backup Codes** | Generate in account settings | Enter code (single use) |

**Lost MFA Device:**
1. Use backup codes
2. Contact help desk with verification
3. Manager approval for re-enrollment
4. Register new device

---

## Appendix C: Service Account Naming Convention

Format: `svc-[environment]-[service]-[purpose]`

Examples:
- `svc-prod-cbam-api` - Production CBAM API service
- `svc-staging-ingestion-db` - Staging ingestion database access
- `svc-prod-monitoring-read` - Production monitoring read-only access

---

**Document Classification: Internal**
**Policy Owner: Chief Information Security Officer**
**Copyright 2026 GreenLang Climate OS. All Rights Reserved.**
