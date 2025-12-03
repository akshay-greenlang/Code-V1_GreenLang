# GreenLang Security Policy

## Overview

GreenLang is committed to maintaining the highest security standards for our AI agent development platform. This document outlines our security policies, vulnerability reporting procedures, and supported versions.

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          | End of Life    |
| ------- | ------------------ | -------------- |
| 0.3.x   | :white_check_mark: | Current        |
| 0.2.x   | :white_check_mark: | March 2026     |
| 0.1.x   | :x:                | December 2024  |
| < 0.1   | :x:                | Not Supported  |

## Security Features

### Authentication

GreenLang implements enterprise-grade authentication:

- **JWT Tokens**: RS256 asymmetric signing with 2048-bit RSA keys
- **Token Expiry**: 1-hour access tokens (configurable)
- **Claims**: Includes `sub`, `tenant_id`, `roles`, `permissions`
- **JWKS Endpoint**: `/.well-known/jwks.json` for key distribution
- **Token Revocation**: JTI blacklist support

### API Keys

- **Format**: `glk_` prefix for easy identification
- **Storage**: SHA-256 hashed (plaintext never stored)
- **Rotation**: 90-day rotation recommended
- **Rate Limiting**: Per-key rate limits supported
- **Scopes**: Fine-grained permission scopes

### Multi-Tenancy

- Complete tenant isolation
- Row-Level Security (RLS) in PostgreSQL
- Tenant ID validation on all API endpoints
- Cross-tenant access prevention

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow our responsible disclosure process.

### Contact

**Email**: security@greenlang.ai

**PGP Key**: [Available upon request]

### What to Include

When reporting a vulnerability, please include:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential security impact if exploited
3. **Reproduction Steps**: Detailed steps to reproduce the issue
4. **Affected Versions**: Which versions are affected
5. **Suggested Fix**: If you have a suggested remediation (optional)

### Response Timeline

| Phase | Timeline |
|-------|----------|
| Initial Response | Within 24 hours |
| Triage & Assessment | Within 72 hours |
| Fix Development | Within 7-30 days (severity dependent) |
| Public Disclosure | 90 days after report (coordinated) |

### Severity Classification

| Severity | CVSS Score | Response SLA | Examples |
|----------|------------|--------------|----------|
| Critical | 9.0 - 10.0 | 24 hours | RCE, Auth bypass, Data breach |
| High | 7.0 - 8.9 | 7 days | Privilege escalation, XSS |
| Medium | 4.0 - 6.9 | 30 days | Information disclosure |
| Low | 0.1 - 3.9 | 90 days | Minor information leaks |

### Bug Bounty Program

We are evaluating a formal bug bounty program. In the meantime:

- Valid security reports are acknowledged in our release notes
- Significant findings may receive recognition or rewards at our discretion
- We do not pursue legal action against good-faith security researchers

## Security Best Practices

### For GreenLang Users

1. **Keep Updated**: Always use the latest supported version
2. **Rotate Credentials**: Rotate API keys every 90 days
3. **Enable MFA**: Use multi-factor authentication for admin accounts
4. **Review Permissions**: Apply principle of least privilege
5. **Monitor Logs**: Review audit logs for suspicious activity

### For Developers

1. **Never Hardcode Secrets**: Use environment variables
2. **Validate Input**: Always sanitize user input
3. **Use HTTPS**: Enforce TLS for all connections
4. **Sign Commits**: Use GPG-signed commits
5. **Review Dependencies**: Check for known vulnerabilities

### Environment Variables

Required security-related environment variables:

```bash
# JWT Configuration
GL_JWT_PRIVATE_KEY_PATH=/path/to/private.pem
GL_JWT_PUBLIC_KEY_PATH=/path/to/public.pem
GL_JWT_ISSUER=greenlang
GL_JWT_AUDIENCE=greenlang-api
GL_JWT_EXPIRY_SECONDS=3600

# Secret Key (for general encryption)
GL_SECRET_KEY=<randomly-generated-key>

# Database (with SSL)
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

## Security Scanning

Our CI/CD pipeline includes comprehensive security scanning:

### Scanners

| Scanner | Purpose | Run Frequency |
|---------|---------|---------------|
| **Trivy** | Container image scanning | Every PR, Daily |
| **Snyk** | Dependency vulnerability scanning | Every PR, Daily |
| **Bandit** | Python SAST (static analysis) | Every PR |
| **Gitleaks** | Secret detection | Every PR, Pre-commit |

### Thresholds

| Finding Type | Threshold | Action |
|--------------|-----------|--------|
| Critical CVE | 0 | Block merge |
| High CVE | 3 | Block merge |
| Secrets | 0 | Block merge |
| SAST High | 0 | Block merge |

## Compliance

GreenLang security controls align with:

- **SOC 2 Type II**: Trust Services Criteria
- **ISO 27001**: Information Security Management
- **GDPR**: Data Protection (EU)
- **NIST 800-218**: Secure Software Development Framework

### Audit Schedule

- **Internal Security Review**: Monthly
- **External Penetration Test**: Quarterly
- **SOC 2 Audit**: Annually
- **Dependency Audit**: Weekly (automated)

## Incident Response

### Contact for Active Incidents

For active security incidents requiring immediate attention:

- **Email**: incident@greenlang.ai
- **Response Time**: Within 1 hour (24/7)

### Incident Severity Levels

| Level | Description | Response |
|-------|-------------|----------|
| P1 | Critical system compromise | Immediate escalation |
| P2 | Service degradation | Within 4 hours |
| P3 | Potential vulnerability | Within 24 hours |
| P4 | Minor security concern | Within 72 hours |

## Security Updates

### Notification Channels

- **Security Advisories**: [GitHub Security Advisories](https://github.com/greenlang/greenlang/security/advisories)
- **Release Notes**: Security fixes documented in CHANGELOG.md
- **Mailing List**: Subscribe at security-announce@greenlang.ai

### Update Process

1. Security patches are backported to all supported versions
2. Patches are released as minor version updates
3. Critical patches may trigger immediate releases
4. 48-hour notice given before mandatory updates

## Third-Party Security

### Supply Chain Security

- All dependencies are scanned for vulnerabilities
- SBOM (Software Bill of Materials) generated for each release
- Signed releases using Sigstore (coming soon)
- Lock files committed for reproducible builds

### Allowed Dependencies

We maintain an allowlist of approved dependencies. New dependencies require:

1. License compatibility check
2. Security audit
3. Maintenance status review
4. Team approval

## Data Protection

### Data at Rest

- All data encrypted with AES-256
- Per-tenant encryption keys
- AWS KMS for key management

### Data in Transit

- TLS 1.3 minimum for external traffic
- mTLS for internal service communication
- Certificate pinning for critical services

### Data Handling

- PII is encrypted at field level
- Logs are sanitized (no secrets)
- Audit trails are immutable
- Data retention policies enforced

## Acknowledgments

We would like to thank the following security researchers for their responsible disclosure:

- *List will be updated as reports are received*

## Contact

For security-related questions or concerns:

- **Security Team**: security@greenlang.ai
- **General Support**: support@greenlang.ai
- **Documentation**: https://docs.greenlang.ai/security

---

**Last Updated**: December 2025
**Version**: 2.0.0
**Review Cycle**: Quarterly
