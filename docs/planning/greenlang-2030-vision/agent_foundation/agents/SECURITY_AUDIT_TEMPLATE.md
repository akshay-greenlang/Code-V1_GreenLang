# Security Audit Report Template
## GreenLang Agent Security Assessment

### Document Control
| Item | Value |
|------|-------|
| Document Version | 1.0 |
| Audit Date | [DATE] |
| Agent ID | [GL-XXX] |
| Agent Name | [CODENAME] |
| Auditor | GreenLang Security Team |
| Classification | Internal |

---

## 1. Executive Summary

### 1.1 Overall Security Score
| Category | Score | Status |
|----------|-------|--------|
| Authentication & Authorization | [X]/10 | [PASS/FAIL] |
| Input Validation | [X]/10 | [PASS/FAIL] |
| Data Protection | [X]/10 | [PASS/FAIL] |
| Network Security | [X]/10 | [PASS/FAIL] |
| Dependency Security | [X]/10 | [PASS/FAIL] |
| Logging & Monitoring | [X]/10 | [PASS/FAIL] |
| **Total Score** | **[XX]/60** | **[STATUS]** |

### 1.2 Risk Summary
- **Critical Issues**: 0
- **High Issues**: 0
- **Medium Issues**: 0
- **Low Issues**: 0
- **Informational**: 0

---

## 2. Authentication & Authorization

### 2.1 API Authentication
- [ ] API key authentication implemented
- [ ] OAuth 2.0 with PKCE supported
- [ ] JWT token validation
- [ ] Token expiration enforced
- [ ] Refresh token rotation

### 2.2 Role-Based Access Control (RBAC)
| Role | Permissions |
|------|-------------|
| admin | Full access |
| operator | Read + Execute |
| viewer | Read only |
| service_account | API access only |

### 2.3 Findings
[List any authentication/authorization issues found]

---

## 3. Input Validation

### 3.1 Validation Checklist
- [ ] All API inputs validated with Pydantic models
- [ ] SQL injection prevention (parameterized queries)
- [ ] Command injection prevention
- [ ] Path traversal prevention
- [ ] XML/JSON injection prevention
- [ ] Type coercion handled safely

### 3.2 Input Sanitization
```python
# Example validation pattern used
from pydantic import BaseModel, validator, constr

class SafeInput(BaseModel):
    equipment_id: constr(regex=r'^[A-Z]{2}-\d{4}$')
    value: float = Field(ge=0, le=10000)

    @validator('equipment_id')
    def validate_equipment_id(cls, v):
        # Additional validation logic
        return v
```

### 3.3 Findings
[List any input validation issues found]

---

## 4. Data Protection

### 4.1 Encryption
| Data Type | At Rest | In Transit |
|-----------|---------|------------|
| Configuration | AES-256 | TLS 1.3 |
| Credentials | AES-256 + KMS | TLS 1.3 |
| Process Data | AES-256 | TLS 1.3 |
| Audit Logs | AES-256 | TLS 1.3 |

### 4.2 Secrets Management
- [ ] No hardcoded credentials in source code
- [ ] Secrets loaded from environment/vault
- [ ] Secrets rotation mechanism in place
- [ ] Sensitive data masked in logs

### 4.3 Data Classification
| Data Category | Classification | Handling |
|---------------|----------------|----------|
| Equipment IDs | Internal | Standard |
| Process Values | Confidential | Encrypted |
| User Credentials | Secret | Vault |
| API Keys | Secret | Rotated |

### 4.4 Findings
[List any data protection issues found]

---

## 5. Network Security

### 5.1 Network Controls
- [ ] TLS 1.3 enforced
- [ ] Certificate validation enabled
- [ ] Network segmentation implemented
- [ ] Firewall rules documented
- [ ] Rate limiting enabled

### 5.2 API Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000
```

### 5.3 Findings
[List any network security issues found]

---

## 6. Dependency Security

### 6.1 Vulnerability Scan Results
| Package | Version | CVE | Severity | Status |
|---------|---------|-----|----------|--------|
| [pkg] | [ver] | [CVE-XXXX] | [SEV] | [Fixed/Pending] |

### 6.2 SBOM (Software Bill of Materials)
- SBOM Format: CycloneDX 1.4
- Total Dependencies: [X]
- Direct Dependencies: [X]
- Transitive Dependencies: [X]

### 6.3 Dependency Update Policy
- Security patches: Within 24 hours
- Minor updates: Weekly
- Major updates: Monthly review

### 6.4 Findings
[List any dependency security issues found]

---

## 7. Logging & Monitoring

### 7.1 Audit Logging
| Event Type | Logged | Retention |
|------------|--------|-----------|
| Authentication attempts | Yes | 90 days |
| Authorization failures | Yes | 90 days |
| Configuration changes | Yes | 1 year |
| Data access | Yes | 90 days |
| Calculation provenance | Yes | 7 years |

### 7.2 Security Monitoring
- [ ] Failed login alerting
- [ ] Anomaly detection
- [ ] Rate limit breach alerting
- [ ] Certificate expiry monitoring

### 7.3 Findings
[List any logging/monitoring issues found]

---

## 8. Zero-Hallucination Security

### 8.1 Determinism Verification
- [ ] All calculations are deterministic
- [ ] No LLM in calculation path
- [ ] Provenance hashing implemented (SHA-256)
- [ ] Audit trail immutable

### 8.2 Provenance Security
```python
# Provenance hash verification
import hashlib

def verify_provenance(result, expected_hash):
    """Verify calculation provenance."""
    computed = hashlib.sha256(
        json.dumps(result.calculation_steps, sort_keys=True).encode()
    ).hexdigest()
    return computed == expected_hash
```

### 8.3 Findings
[List any determinism/provenance issues found]

---

## 9. Remediation Plan

### 9.1 Critical/High Priority (Immediate)
| Issue | Severity | Remediation | Owner | Due Date |
|-------|----------|-------------|-------|----------|
| [Issue] | [Sev] | [Action] | [Owner] | [Date] |

### 9.2 Medium Priority (30 days)
| Issue | Severity | Remediation | Owner | Due Date |
|-------|----------|-------------|-------|----------|
| [Issue] | [Sev] | [Action] | [Owner] | [Date] |

### 9.3 Low Priority (90 days)
| Issue | Severity | Remediation | Owner | Due Date |
|-------|----------|-------------|-------|----------|
| [Issue] | [Sev] | [Action] | [Owner] | [Date] |

---

## 10. Compliance Checklist

### 10.1 OWASP Top 10 (2021)
- [ ] A01:2021 - Broken Access Control
- [ ] A02:2021 - Cryptographic Failures
- [ ] A03:2021 - Injection
- [ ] A04:2021 - Insecure Design
- [ ] A05:2021 - Security Misconfiguration
- [ ] A06:2021 - Vulnerable Components
- [ ] A07:2021 - Authentication Failures
- [ ] A08:2021 - Software Integrity Failures
- [ ] A09:2021 - Logging Failures
- [ ] A10:2021 - SSRF

### 10.2 Industry Standards
- [ ] SOC 2 Type II controls
- [ ] ISO 27001 alignment
- [ ] NIST Cybersecurity Framework

---

## 11. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Security Lead | | | |
| Development Lead | | | |
| Operations Lead | | | |

---

*This security audit report template is part of the GreenLang Agent Security Framework v1.0*
