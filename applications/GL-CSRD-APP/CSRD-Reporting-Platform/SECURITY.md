# Security Policy

## Overview

The CSRD/ESRS Digital Reporting Platform handles sensitive corporate sustainability data and must meet stringent security requirements for regulatory compliance. This document outlines our security policy, vulnerability reporting process, and security features.

**Security Contact**: security@greenlang.io

---

## Table of Contents

1. [Supported Versions](#supported-versions)
2. [Security Features](#security-features)
3. [Vulnerability Reporting](#vulnerability-reporting)
4. [Security Audit Checklist](#security-audit-checklist)
5. [Compliance Requirements](#compliance-requirements)
6. [Data Protection](#data-protection)
7. [Access Control](#access-control)
8. [XBRL Security](#xbrl-security)
9. [Security Best Practices](#security-best-practices)
10. [Penetration Testing Guidelines](#penetration-testing-guidelines)
11. [Incident Response Plan](#incident-response-plan)
12. [Security Contacts](#security-contacts)

---

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Security Updates Until |
| ------- | ------------------ | ---------------------- |
| 1.0.x   | ✅ Yes (Current)   | 2027-10-18 (2 years)   |
| 0.x.x   | ❌ No              | N/A (Pre-release)      |

**Security Support Policy:**
- Critical security issues: Patched within 24-48 hours
- High-severity issues: Patched within 7 days
- Medium-severity issues: Patched within 30 days
- Low-severity issues: Included in next minor/major release

---

## Security Features

### 1. Data Encryption

#### Encryption at Rest
- **Algorithm**: AES-256-GCM
- **Key Management**:
  - Hardware Security Module (HSM) support
  - AWS KMS, Azure Key Vault, GCP Cloud KMS integration
  - Key rotation every 90 days (configurable)
- **Scope**: All ESG data, company profiles, calculated metrics, reports
- **Implementation**: Database-level encryption + application-level encryption for sensitive fields

```python
# Example: Encrypted field in company profile
from cryptography.fernet import Fernet

# Encryption key from environment variable or key management service
encryption_key = os.getenv("DATA_ENCRYPTION_KEY")
fernet = Fernet(encryption_key)

# Encrypt sensitive data
encrypted_lei = fernet.encrypt(company_lei.encode())
```

#### Encryption in Transit
- **Protocol**: TLS 1.3 (minimum TLS 1.2)
- **Cipher Suites**:
  - TLS_AES_256_GCM_SHA384
  - TLS_CHACHA20_POLY1305_SHA256
  - TLS_AES_128_GCM_SHA256
- **Certificate Management**: Let's Encrypt or enterprise CA
- **HSTS**: Enabled with 1-year max-age
- **Scope**: All API communications, database connections, file transfers

### 2. API Key Management

#### LLM API Keys (OpenAI, Anthropic)
- **Storage**: Environment variables or secrets management service
- **Never hardcoded** in source code
- **Rotation**: Recommended every 90 days
- **Scope limitation**: Read-only access to LLM APIs
- **Rate limiting**: Enforced to prevent abuse

```yaml
# config/csrd_config.yaml (SECURE EXAMPLE)
ai:
  openai:
    api_key: ${OPENAI_API_KEY}  # From environment variable
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}  # From environment variable
```

#### Database Credentials
- **Storage**: Environment variables or AWS Secrets Manager
- **Connection pooling**: With authentication
- **Least privilege**: Each service has minimal required permissions
- **Rotation**: Automated quarterly rotation

#### API Authentication
- **OAuth 2.0** with JWT tokens
- **Token expiration**: 1 hour (configurable)
- **Refresh tokens**: 7 days (configurable)
- **Multi-Factor Authentication (MFA)**: Supported via TOTP/WebAuthn

### 3. Input Validation

#### All User Inputs
- **JSON Schema validation** for structured data
- **Type checking** with Pydantic models
- **Size limits**:
  - File uploads: 500 MB max
  - JSON payloads: 50 MB max
  - String fields: 10,000 characters max
- **Sanitization**: HTML escaping, SQL parameterization
- **Whitelist validation**: Allowed file types, MIME types

```python
# Example: Input validation with Pydantic
from pydantic import BaseModel, Field, validator

class CompanyProfile(BaseModel):
    legal_name: str = Field(..., max_length=500, min_length=1)
    lei_code: str = Field(..., regex=r'^[A-Z0-9]{20}$')
    employee_count: int = Field(..., gt=0, lt=10_000_000)

    @validator('lei_code')
    def validate_lei_checksum(cls, v):
        # Custom LEI code checksum validation
        if not validate_lei_code(v):
            raise ValueError('Invalid LEI code checksum')
        return v
```

#### SQL Injection Protection
- **ORM-only database access** (SQLAlchemy)
- **Parameterized queries** enforced
- **No raw SQL** without security review
- **Input escaping** for all dynamic queries

#### XSS Protection
- **HTML escaping** in all report templates
- **Content Security Policy (CSP)** headers
- **Safe PDF generation** with sanitized inputs
- **XBRL validation** to prevent injection attacks

### 4. XBRL Security

#### XBRL Taxonomy Validation
- **Schema validation** against official ESRS XBRL taxonomy
- **Element validation**: All tags from official taxonomy only
- **Namespace validation**: Prevent namespace hijacking
- **File integrity**: Checksum validation of taxonomy files

#### iXBRL Generation
- **Template injection prevention**: No user-provided templates
- **HTML sanitization**: All narrative content escaped
- **XML external entity (XXE) prevention**: Disabled XML external entities
- **XBRL formula validation**: Prevent formula injection

#### ESEF Package Security
- **ZIP bomb prevention**: Size limit checks before extraction
- **Path traversal prevention**: Validate all file paths in package
- **Signature verification**: Optional digital signature support
- **Malware scanning**: Optional integration with antivirus APIs

---

## Vulnerability Reporting

### Reporting Process

#### How to Report a Security Vulnerability

**DO NOT** create a public GitHub issue for security vulnerabilities.

**Email**: security@greenlang.io
**PGP Key**: Available on request

**Include in your report:**
1. **Description**: Detailed description of the vulnerability
2. **Impact**: Potential security impact (confidentiality, integrity, availability)
3. **Steps to Reproduce**: Clear steps to reproduce the vulnerability
4. **Proof of Concept**: Code or screenshots (if applicable)
5. **Suggested Fix**: Your recommendation for fixing the issue (optional)
6. **Disclosure Timeline**: Your intended disclosure timeline

#### What to Expect

1. **Acknowledgment**: Within 24 hours of submission
2. **Initial Assessment**: Within 48 hours (severity classification)
3. **Status Updates**: Weekly updates on progress
4. **Fix Development**: Depends on severity (24 hours to 30 days)
5. **Disclosure**: Coordinated disclosure after fix is released
6. **Credit**: Public acknowledgment in release notes (if desired)

### Severity Classification

| Severity | Description | Examples | Response Time |
|----------|-------------|----------|---------------|
| **Critical** | Can lead to complete system compromise | Remote code execution, SQL injection | 24-48 hours |
| **High** | Significant impact on data security | Authentication bypass, privilege escalation | 7 days |
| **Medium** | Limited impact on security | Information disclosure, DoS | 30 days |
| **Low** | Minimal impact | Minor information leaks | Next release |

### Bug Bounty Program

**Status**: Under consideration for v2.0

**Scope (Planned):**
- Critical vulnerabilities: $500-$2,000
- High-severity vulnerabilities: $200-$500
- Medium-severity vulnerabilities: $50-$200

**Out of Scope:**
- Social engineering attacks
- Physical attacks
- Denial of service attacks
- Issues in third-party dependencies (report to upstream)

---

## Security Audit Checklist

Use this checklist before deploying the platform or before each major release.

### Pre-Deployment Security Audit

#### 1. Code Security
- [ ] **No hardcoded secrets** in source code (run `git secrets` or `trufflehog`)
- [ ] **No sensitive data in logs** (review logging configuration)
- [ ] **All dependencies up to date** (run `pip list --outdated`)
- [ ] **No known vulnerabilities** in dependencies (run `safety check`)
- [ ] **Static code analysis** complete (run `bandit -r .`)
- [ ] **Type checking** complete (run `mypy agents/`)
- [ ] **Linting** complete (run `ruff check .`)

#### 2. Authentication & Authorization
- [ ] **Strong password policy** enforced (min 12 chars, complexity requirements)
- [ ] **MFA enabled** for all admin accounts
- [ ] **OAuth 2.0 configured** correctly
- [ ] **JWT tokens expire** within 1 hour
- [ ] **RBAC roles defined** and tested
- [ ] **Principle of least privilege** applied to all service accounts

#### 3. Data Protection
- [ ] **Encryption at rest enabled** for all databases
- [ ] **TLS 1.3 enforced** for all connections
- [ ] **Encryption keys rotated** in last 90 days
- [ ] **Database backups encrypted**
- [ ] **PII data minimization** implemented
- [ ] **Data retention policy** configured (GDPR compliance)

#### 4. Network Security
- [ ] **Firewall rules configured** (whitelist approach)
- [ ] **DDoS protection enabled** (CloudFlare, AWS Shield, etc.)
- [ ] **Rate limiting configured** on all API endpoints
- [ ] **CORS policy configured** (restrictive origins)
- [ ] **Security headers configured** (CSP, HSTS, X-Frame-Options)
- [ ] **VPN required** for production database access

#### 5. Input Validation
- [ ] **All inputs validated** with JSON Schema or Pydantic
- [ ] **File upload validation** (type, size, content)
- [ ] **SQL injection protection** verified (ORM-only, no raw SQL)
- [ ] **XSS protection** verified (HTML escaping, CSP)
- [ ] **XBRL injection prevention** tested
- [ ] **Path traversal prevention** tested

#### 6. Logging & Monitoring
- [ ] **Audit logging enabled** for all sensitive operations
- [ ] **Log retention configured** (7 years for regulatory compliance)
- [ ] **Logs are immutable** (append-only storage)
- [ ] **Sensitive data redacted** from logs (passwords, API keys)
- [ ] **Security alerts configured** (failed logins, privilege escalation attempts)
- [ ] **Log analysis tools configured** (SIEM integration)

#### 7. Third-Party Services
- [ ] **LLM API keys secured** (environment variables, not hardcoded)
- [ ] **Vector database access restricted** (Pinecone, Weaviate)
- [ ] **Cloud provider security best practices** followed (AWS, Azure, GCP)
- [ ] **Third-party integrations reviewed** for security
- [ ] **Service-to-service authentication** configured (mutual TLS)

#### 8. Compliance
- [ ] **GDPR compliance verified** (data subject rights, privacy policy)
- [ ] **EU CSRD compliance verified** (audit trail, data provenance)
- [ ] **Data processing agreements** signed with sub-processors
- [ ] **Privacy impact assessment** completed
- [ ] **Data breach notification plan** in place

#### 9. Incident Response
- [ ] **Incident response plan documented** (see section below)
- [ ] **Security contacts defined** and published
- [ ] **Backup and recovery tested** (last test date: ______)
- [ ] **Forensics tools available** (disk imaging, log analysis)
- [ ] **Communication plan** for security incidents

#### 10. Penetration Testing
- [ ] **External penetration test** completed (recommended annually)
- [ ] **Internal penetration test** completed
- [ ] **Findings remediated** or risk-accepted
- [ ] **Retest of high-severity findings** completed

---

## Compliance Requirements

### GDPR Compliance (EU General Data Protection Regulation)

#### Data Subject Rights
- **Right to Access**: Exportable company data (JSON format)
- **Right to Rectification**: Data update API provided
- **Right to Erasure**: Data deletion with 30-day retention for audit
- **Right to Data Portability**: Machine-readable export (JSON, CSV)
- **Right to Object**: Opt-out mechanisms for AI processing

#### Privacy by Design
- **Data minimization**: Collect only necessary ESG data
- **Purpose limitation**: Data used only for CSRD reporting
- **Storage limitation**: Automatic deletion after retention period
- **Pseudonymization**: Employee data anonymized where possible

#### Lawful Basis
- **Legal obligation**: CSRD reporting requirement
- **Legitimate interest**: Corporate sustainability reporting
- **Consent**: For AI-powered features (materiality, narratives)

#### Data Processing Agreement
- Template DPA provided in `docs/GDPR_DPA_TEMPLATE.md`
- Sub-processors disclosed: OpenAI, Anthropic, Pinecone (if used)

### EU CSRD Compliance

#### Audit Trail Requirements
- **Complete calculation provenance**: Every metric traceable to source
- **Immutable audit logs**: 7-year retention (regulatory requirement)
- **Version control**: All reference data and formulas versioned
- **External auditor access**: Audit package generation for third-party verification

#### Data Accuracy
- **Zero-hallucination guarantee**: No LLM-based calculations
- **Deterministic calculations**: 100% reproducible results
- **Source attribution**: All data points linked to source documents

---

## Data Protection

### Data Classification

| Classification | Description | Examples | Security Controls |
|----------------|-------------|----------|-------------------|
| **Public** | Non-sensitive, publicly available | ESRS standards, taxonomy | Standard controls |
| **Internal** | Company-internal only | Aggregated ESG metrics | Encryption, access control |
| **Confidential** | Sensitive business data | Raw ESG data, financial metrics | Encryption, RBAC, audit logging |
| **Restricted** | Highly sensitive | Employee PII, board diversity details | Encryption, MFA, restricted access, audit logging |

### Data Retention

| Data Type | Retention Period | Rationale |
|-----------|------------------|-----------|
| ESG data | 7 years | EU regulatory requirement |
| Audit logs | 7 years | Compliance and forensics |
| Calculated metrics | 7 years | External audit requirement |
| CSRD reports | 10 years | Corporate record keeping |
| User activity logs | 2 years | Security monitoring |
| Temporary data | 30 days | Processing intermediate states |

### Data Deletion

**Soft Delete**: Data marked as deleted but retained for retention period
**Hard Delete**: Permanent deletion after retention period (cryptographic erasure of encryption keys)

```python
# Example: Soft delete with retention
def delete_company_data(company_id: str):
    company = db.query(Company).filter(Company.id == company_id).first()
    company.deleted_at = datetime.utcnow()
    company.hard_delete_at = datetime.utcnow() + timedelta(days=2555)  # 7 years
    db.commit()
```

---

## Access Control

### Role-Based Access Control (RBAC)

| Role | Permissions | Use Case |
|------|-------------|----------|
| **Admin** | Full system access, user management, configuration | IT administrators |
| **Sustainability Officer** | Create/edit ESG data, run reports, export | Primary users |
| **Auditor (Read-Only)** | View data, download audit packages | External auditors |
| **Data Contributor** | Submit ESG data only, no report access | Departmental contributors |
| **Executive (View)** | View reports and dashboards only | C-suite, board members |

### Permission Matrix

| Action | Admin | Sustainability Officer | Auditor | Data Contributor | Executive |
|--------|-------|------------------------|---------|------------------|-----------|
| Upload ESG data | ✅ | ✅ | ❌ | ✅ | ❌ |
| Edit company profile | ✅ | ✅ | ❌ | ❌ | ❌ |
| Run materiality assessment | ✅ | ✅ | ❌ | ❌ | ❌ |
| Generate CSRD report | ✅ | ✅ | ❌ | ❌ | ❌ |
| View audit trail | ✅ | ✅ | ✅ | ❌ | ❌ |
| Download reports | ✅ | ✅ | ✅ | ❌ | ✅ |
| Manage users | ✅ | ❌ | ❌ | ❌ | ❌ |
| Configure system | ✅ | ❌ | ❌ | ❌ | ❌ |

### Authentication Methods

1. **Username/Password** (with bcrypt hashing, salted)
2. **OAuth 2.0** (Google, Microsoft, Okta)
3. **SAML 2.0** (Enterprise SSO)
4. **Multi-Factor Authentication** (TOTP, WebAuthn/FIDO2)

---

## XBRL Security

### XBRL Taxonomy Integrity

**Problem**: Malicious modification of XBRL taxonomy could lead to incorrect reporting.

**Solution**:
1. **Checksum validation** of taxonomy files on load
2. **Digital signatures** for official ESRS taxonomy
3. **Immutable taxonomy storage** (read-only filesystem)
4. **Version pinning** to specific taxonomy release

```python
# Example: Taxonomy integrity check
import hashlib

def validate_taxonomy_integrity(taxonomy_file: Path, expected_sha256: str):
    with open(taxonomy_file, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if file_hash != expected_sha256:
        raise SecurityError(f"Taxonomy file integrity check failed: {taxonomy_file}")
```

### iXBRL Injection Prevention

**Problem**: User-provided narrative content could inject malicious XBRL elements.

**Solution**:
1. **HTML escaping** for all narrative content
2. **Whitelist of allowed XBRL elements**
3. **Validation against schema** before rendering
4. **Content Security Policy** headers

### ESEF Package Validation

**Problem**: Malicious ESEF packages could contain exploits.

**Solution**:
1. **ZIP bomb detection** (size limits, compression ratio checks)
2. **Path traversal prevention** (validate all file paths)
3. **File type validation** (only XHTML, XML, PNG allowed)
4. **Malware scanning** (optional ClamAV integration)

---

## Security Best Practices

### For Developers

1. **Never commit secrets** to version control
   - Use `.env` files (add to `.gitignore`)
   - Use environment variables for all secrets
   - Use pre-commit hooks to detect secrets (`git-secrets`, `detect-secrets`)

2. **Follow secure coding guidelines**
   - Input validation on all user-provided data
   - Output encoding for all generated content
   - Parameterized queries only (no raw SQL)
   - Least privilege principle for service accounts

3. **Dependency management**
   - Keep dependencies up to date (`pip list --outdated`)
   - Review security advisories (`safety check`)
   - Pin dependencies to specific versions (`requirements.txt`)
   - Audit new dependencies before adding

4. **Code review**
   - Security review required for authentication/authorization code
   - Peer review required for cryptographic code
   - Automated static analysis in CI/CD pipeline

### For Administrators

1. **System hardening**
   - Disable unnecessary services
   - Apply OS security patches monthly
   - Use firewall with whitelist approach
   - Enable SELinux/AppArmor

2. **Monitoring**
   - Enable audit logging for all sensitive operations
   - Configure alerts for failed authentication attempts
   - Monitor for unusual data access patterns
   - Regular log review (weekly)

3. **Backup and recovery**
   - Daily encrypted backups
   - Test recovery procedure quarterly
   - Store backups in separate security zone
   - Implement 3-2-1 backup strategy

### For End Users

1. **Strong passwords**
   - Minimum 12 characters
   - Mix of uppercase, lowercase, numbers, symbols
   - Use password manager
   - No password reuse

2. **Multi-Factor Authentication**
   - Enable MFA for all accounts
   - Use authenticator app (not SMS)
   - Backup codes stored securely

3. **Data handling**
   - Only upload necessary data
   - Review data before submission
   - Delete data when no longer needed
   - Report suspicious activity immediately

---

## Penetration Testing Guidelines

### Scope

**In Scope:**
- Web application (CSRD reporting platform)
- API endpoints
- Authentication and authorization mechanisms
- Input validation
- XBRL generation
- Database security

**Out of Scope:**
- Social engineering
- Physical attacks
- Third-party services (OpenAI, Anthropic, Pinecone)
- Denial of Service (DoS) attacks
- Attacks against other customers (if multi-tenant)

### Testing Methodology

**OWASP Top 10 Coverage:**
1. Injection (SQL, XSS, XBRL)
2. Broken Authentication
3. Sensitive Data Exposure
4. XML External Entities (XXE)
5. Broken Access Control
6. Security Misconfiguration
7. Cross-Site Scripting (XSS)
8. Insecure Deserialization
9. Using Components with Known Vulnerabilities
10. Insufficient Logging & Monitoring

**Additional Tests:**
- API security testing
- Authentication bypass attempts
- Privilege escalation attempts
- File upload vulnerabilities
- XBRL injection attempts

### Reporting

**Required Information:**
1. Vulnerability description
2. Severity rating (CVSS score)
3. Steps to reproduce
4. Proof of concept
5. Recommended remediation
6. References (CWE, CVE)

---

## Incident Response Plan

### Incident Response Team

| Role | Responsibilities | Contact |
|------|------------------|---------|
| **Incident Commander** | Overall coordination, decision-making | security@greenlang.io |
| **Technical Lead** | Technical investigation, remediation | csrd@greenlang.io |
| **Communications Lead** | Customer/public communication | communications@greenlang.io |
| **Legal Counsel** | Legal compliance, regulatory notification | legal@greenlang.io |

### Incident Response Process

#### 1. Detection & Analysis (0-2 hours)

**Detection Sources:**
- Security monitoring alerts
- User reports
- Vulnerability scan results
- External security researcher reports

**Initial Actions:**
1. Confirm incident is real (not false positive)
2. Classify severity (Critical, High, Medium, Low)
3. Activate incident response team
4. Begin incident log

#### 2. Containment (2-6 hours)

**Short-term Containment:**
- Isolate affected systems
- Block malicious IPs
- Disable compromised accounts
- Preserve evidence

**Long-term Containment:**
- Apply temporary patches
- Implement additional monitoring
- Prepare for recovery

#### 3. Eradication (6-24 hours)

- Remove malware, backdoors, unauthorized access
- Patch vulnerabilities
- Update compromised credentials
- Verify system integrity

#### 4. Recovery (24-72 hours)

- Restore systems from clean backups
- Verify functionality
- Enhanced monitoring during recovery period
- Gradual restoration of services

#### 5. Post-Incident Activities (1 week)

- Incident report (lessons learned)
- Update security controls
- Update incident response plan
- Communicate with affected parties
- Regulatory notification (if required by GDPR)

### Communication Plan

#### Internal Communication
- **Immediate**: Incident response team
- **Within 2 hours**: Executive leadership
- **Within 24 hours**: All employees

#### External Communication
- **Data Breach**: Notify affected customers within 72 hours (GDPR requirement)
- **Regulatory Notification**: Notify data protection authority within 72 hours if applicable
- **Public Disclosure**: Coordinated disclosure after incident resolved

### Data Breach Notification Template

```
Subject: Security Incident Notification

Dear [Customer Name],

We are writing to inform you of a security incident that may have affected your data.

WHAT HAPPENED:
[Description of the incident]

WHAT DATA WAS AFFECTED:
[Types of data compromised]

WHAT WE'RE DOING:
[Remediation actions taken]

WHAT YOU SHOULD DO:
[Recommended actions for affected parties]

CONTACT:
For questions, contact security@greenlang.io

Sincerely,
GreenLang CSRD Team
```

---

## Security Contacts

### Primary Security Contact
- **Email**: security@greenlang.io
- **PGP Key**: Available on request
- **Response Time**: 24 hours

### Emergency Contact
- **Phone**: +1 (XXX) XXX-XXXX (24/7 security hotline - to be established)
- **Email**: security-emergency@greenlang.io

### Vulnerability Disclosure
- **Email**: security@greenlang.io
- **HackerOne**: (To be established for v2.0)

### Compliance & Legal
- **GDPR Officer**: gdpr@greenlang.io
- **Legal Counsel**: legal@greenlang.io

### Support
- **General Support**: csrd@greenlang.io
- **Enterprise Support**: enterprise@greenlang.io

---

## Security Resources

### Internal Documentation
- `docs/SECURITY_ARCHITECTURE.md` - Security architecture details
- `docs/ENCRYPTION_GUIDE.md` - Encryption implementation guide
- `docs/GDPR_COMPLIANCE.md` - GDPR compliance documentation

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)
- [GDPR Official Text](https://gdpr-info.eu/)

---

## Acknowledgments

We thank the security research community for responsible vulnerability disclosure.

**Hall of Fame** (Responsible Disclosure):
- (None yet - submit vulnerabilities to be listed here!)

---

**Last Updated**: 2025-10-18
**Next Security Audit**: 2026-04-18 (6 months)
**Version**: 1.0.0
