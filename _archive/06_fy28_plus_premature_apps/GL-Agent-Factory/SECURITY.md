# Security Policy

## Supported Versions

The following versions of GL-Agent-Factory are currently being supported with security updates:

| Version | Supported          | End of Support |
| ------- | ------------------ | -------------- |
| 0.9.x   | :white_check_mark: | Current        |
| 0.8.x   | :white_check_mark: | March 2025     |
| < 0.8   | :x:                | Unsupported    |

## Reporting a Vulnerability

We take the security of GL-Agent-Factory seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report vulnerabilities through one of the following channels:

1. **Email**: security@greenlang.io
2. **GitHub Security Advisories**: [Create a private security advisory](https://github.com/greenlang/GL-Agent-Factory/security/advisories/new)

### What to Include

Please include the following information in your report:

- **Type of vulnerability** (e.g., SQL injection, XSS, authentication bypass)
- **Location** of the affected source code (file path, line numbers)
- **Step-by-step instructions** to reproduce the vulnerability
- **Proof-of-concept** code or screenshots if applicable
- **Impact assessment** of the vulnerability
- **Suggested remediation** if you have one

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial Response | Within 48 hours |
| Vulnerability Assessment | Within 7 days |
| Fix Development | Within 30 days for critical issues |
| Public Disclosure | After fix is released + 30 days |

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
2. **Assessment**: Our security team will assess the vulnerability and determine its severity
3. **Updates**: We will keep you informed of our progress
4. **Credit**: If you wish, we will publicly credit you for responsible disclosure

## Security Measures

### Authentication & Authorization

GL-Agent-Factory implements multiple layers of security:

- **JWT Authentication**: Secure token-based authentication with configurable expiration
- **API Key Management**: Scoped API keys with rate limiting
- **Role-Based Access Control (RBAC)**: Granular permissions per tenant
- **OAuth 2.0 Support**: Integration with enterprise identity providers

### Data Protection

- **Encryption at Rest**: All sensitive data encrypted using AES-256
- **Encryption in Transit**: TLS 1.3 required for all connections
- **Data Isolation**: Multi-tenant architecture with strict data isolation
- **PII Handling**: Configurable data retention and anonymization

### Infrastructure Security

- **Container Security**: Minimal base images, non-root execution
- **Network Policies**: Kubernetes network policies for pod isolation
- **Secrets Management**: Kubernetes secrets with optional Vault integration
- **Audit Logging**: Comprehensive audit trail of all operations

### Code Security

- **Dependency Scanning**: Automated scanning with Dependabot and Snyk
- **SAST**: Static analysis with Bandit and Semgrep
- **DAST**: Dynamic analysis in CI/CD pipeline
- **Code Review**: All changes require security-aware review

## Security Best Practices for Users

### API Keys

```python
# DO: Use environment variables
import os
api_key = os.environ.get("GL_API_KEY")

# DON'T: Hardcode credentials
api_key = "sk-live-abc123..."  # NEVER DO THIS
```

### Environment Configuration

```bash
# Required security settings for production
SECRET_KEY=<random-32-char-minimum>
JWT_EXPIRATION_HOURS=1  # Short-lived tokens
ALLOWED_HOSTS=yourdomain.com
DEBUG=false
SECURE_SSL_REDIRECT=true
```

### Network Security

1. **Always use HTTPS** for API communication
2. **Implement IP allowlisting** for production deployments
3. **Use VPN or private networking** for internal services
4. **Enable rate limiting** to prevent abuse

### Data Handling

1. **Validate all inputs** before processing
2. **Sanitize outputs** to prevent injection attacks
3. **Implement proper error handling** without exposing internals
4. **Use parameterized queries** for database operations

## Vulnerability Disclosure Policy

### Scope

The following are in scope for security research:

- GL-Agent-Factory backend API
- Agent calculation engines
- Authentication and authorization systems
- Data storage and encryption
- CLI tools
- Official Docker images

### Out of Scope

- Third-party services and dependencies (report directly to vendors)
- Social engineering attacks
- Physical security
- Denial of service attacks (please don't test these)

### Safe Harbor

We will not pursue legal action against security researchers who:

- Make good faith efforts to comply with this policy
- Do not access or modify data belonging to others
- Do not exploit vulnerabilities beyond what is necessary to demonstrate them
- Report findings promptly
- Do not publicly disclose issues before we have addressed them

## Security Contacts

| Role | Contact |
|------|---------|
| Security Team Lead | security@greenlang.io |
| Emergency Contact | security-emergency@greenlang.io |
| Bug Bounty Program | bounty@greenlang.io |

## Security Updates

Security advisories are published through:

1. [GitHub Security Advisories](https://github.com/greenlang/GL-Agent-Factory/security/advisories)
2. Security mailing list (subscribe at security-announce@greenlang.io)
3. [CHANGELOG.md](CHANGELOG.md) security section

## Compliance

GL-Agent-Factory is designed to support compliance with:

- **SOC 2 Type II**: Security and availability controls
- **GDPR**: Data protection and privacy
- **ISO 27001**: Information security management
- **GHG Protocol**: Greenhouse gas accounting standards

## Acknowledgments

We thank the following security researchers for their responsible disclosures:

*No vulnerabilities have been reported yet. Be the first responsible reporter!*

---

*Last updated: December 2024*
*Security policy version: 1.0*
