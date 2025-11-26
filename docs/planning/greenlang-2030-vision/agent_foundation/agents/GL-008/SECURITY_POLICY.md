# GL-008 SteamTrapInspector Security Policy

## Overview

This document defines the security requirements, credential management policies, audit logging requirements, and compliance checklist for GL-008 SteamTrapInspector (TRAPCATCHER) per IEC 62443-4-2 and OWASP guidelines.

**Agent:** GL-008 SteamTrapInspector
**Version:** 1.0.0
**Security Level:** Industrial Control System (ICS)
**Compliance:** IEC 62443-4-2, OWASP Top 10, NIST Cybersecurity Framework

---

## Table of Contents

1. [Security Requirements](#security-requirements)
2. [Credential Management](#credential-management)
3. [Audit Logging Requirements](#audit-logging-requirements)
4. [Security Validation](#security-validation)
5. [Compliance Checklist](#compliance-checklist)
6. [Incident Response](#incident-response)

---

## Security Requirements

### 1. Zero Hardcoded Credentials Policy

**Requirement:** NO credentials, API keys, or secrets may be hardcoded in:
- Source code
- Configuration files
- Container images
- Version control

**Enforcement:**
- `agents/security_validator.py` checks at startup
- Configuration validation in `config.py::_validate_security_policy()`
- Startup ABORTED if violations detected

**Prohibited Patterns:**
```python
# PROHIBITED - Never do this:
API_KEY = "sk-ant-api03-abc123..."  # Hardcoded API key
DATABASE_URL = "postgresql://user:password@host/db"  # Embedded credentials
JWT_SECRET = "change-this-secret"  # Default/placeholder secrets
```

**Required Pattern:**
```python
# REQUIRED - Use environment variables:
import os
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Or use configuration helper:
from config import TrapInspectorConfig
config = TrapInspectorConfig()
api_key = config.get_api_key("anthropic")
```

### 2. API Key Validation

**Requirements:**
- Anthropic API key: Minimum 32 characters, format `sk-ant-*`
- No test/demo/placeholder keys in production
- Keys loaded from environment variables only
- Key validation at startup

**Environment Variables:**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  # Optional
```

**Validation Checks:**
- Key length >= 32 characters
- Proper format (starts with `sk-ant-` for Anthropic)
- No weak patterns (test, demo, example, placeholder)
- Key present if LLM classification enabled

### 3. Configuration Security

**Zero Secrets Policy:**
```python
@dataclass
class TrapInspectorConfig:
    # SECURITY: All security features enabled by default
    zero_secrets: bool = True  # MUST be True
    enable_audit_logging: bool = True  # MUST be True
    enable_provenance_tracking: bool = True  # MUST be True
```

**URL Security:**
- CMMS/BMS API endpoints MUST NOT contain embedded credentials
- Use token-based authentication (Bearer tokens)
- Tokens stored in environment variables or secret manager

**Example:**
```python
# PROHIBITED:
cmms_api_endpoint = "https://user:password@cmms.example.com/api"

# REQUIRED:
cmms_api_endpoint = "https://cmms.example.com/api"
# Auth token in environment:
CMMS_API_TOKEN = "Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### 4. Deterministic Operation

**Requirements:**
- LLM temperature: MUST be 0.0 (deterministic)
- LLM seed: MUST be 42 (reproducibility)
- Timestamps: Use `DeterministicClock` for testing
- UUIDs: Use `deterministic_uuid()` for audit trails

**Rationale:**
- Reproducible calculations for regulatory compliance
- Audit trail verification
- Testing and validation

### 5. Rate Limiting

**Configuration:**
```python
max_concurrent_inspections: int = 10  # Range: 1-100
calculation_timeout_seconds: float = 30.0  # Max: 300s
monitoring_interval_seconds: int = 300  # Min: 10s (resource protection)
```

**Protection Against:**
- Resource exhaustion
- Denial of Service (DoS)
- Excessive API costs

---

## Credential Management

### 1. Environment Variables

**Required Environment Variables:**

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-api03-...  # Required for LLM classification

# Environment
GREENLANG_ENV=production  # production|development
DEBUG=false  # MUST be false in production
LOG_LEVEL=INFO  # INFO|WARNING|ERROR (not DEBUG in prod)

# Optional Integrations
CMMS_API_TOKEN=Bearer_eyJ0eXAi...  # If CMMS integration enabled
BMS_API_TOKEN=Bearer_eyJ0eXAi...  # If BMS integration enabled
```

**Environment File (.env):**
```bash
# Create .env file (NEVER commit to git)
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GREENLANG_ENV=production
DEBUG=false
LOG_LEVEL=INFO
EOF

# Protect .env file
chmod 600 .env
```

**Add to .gitignore:**
```gitignore
# Secrets and credentials
.env
.env.*
*.secret
*.key
*.pem
*_credentials.json
deployment/secret.yaml.local
deployment/*.secret.yaml
```

### 2. External Secret Management

**Recommended: External Secrets Operator (ESO)**

**AWS Secrets Manager:**
```bash
# Store secrets in AWS
aws secretsmanager create-secret \
  --name greenlang/gl-008/anthropic_api_key \
  --secret-string "sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Kubernetes ExternalSecret
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: gl-008-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
  target:
    name: gl-008-secrets
  data:
    - secretKey: anthropic_api_key
      remoteRef:
        key: greenlang/gl-008/anthropic_api_key
EOF
```

**HashiCorp Vault:**
```bash
# Store in Vault
vault kv put secret/greenlang/gl-008/anthropic_api_key \
  value="sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Kubernetes ExternalSecret
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: gl-008-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
  target:
    name: gl-008-secrets
  data:
    - secretKey: anthropic_api_key
      remoteRef:
        key: secret/greenlang/gl-008/anthropic_api_key
        property: value
EOF
```

### 3. Secret Rotation

**Policy:**
- API keys: Rotate every 90 days
- Service tokens: Rotate every 30 days
- Test credentials: Rotate after each test cycle

**Rotation Procedure:**
```bash
# 1. Generate new API key from provider
# 2. Update secret in secret manager
aws secretsmanager update-secret \
  --secret-id greenlang/gl-008/anthropic_api_key \
  --secret-string "NEW_KEY_VALUE"

# 3. Verify rotation (secret refreshed within 1 hour)
# 4. Revoke old key from provider
```

---

## Audit Logging Requirements

### 1. Required Audit Events

**MUST LOG:**
- All steam trap inspections (with provenance hash)
- Configuration changes
- Security validation results (startup)
- Authentication events (if applicable)
- Energy loss calculations (high-value)
- Failure mode detections
- API calls to external services (CMMS/BMS)

**Audit Log Format:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "inspection",
  "agent_id": "GL-008-TRAPCATCHER",
  "trap_id": "ST-001",
  "user": "operator-1",
  "action": "acoustic_analysis",
  "result": {
    "status": "failed_open",
    "confidence": 0.95,
    "energy_loss_usd_yr": 15000.0
  },
  "provenance_hash": "a3f5b9c2d1e4f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1",
  "metadata": {
    "method": "acoustic",
    "model_version": "1.0.0",
    "duration_ms": 234
  }
}
```

### 2. Provenance Tracking

**Implementation:**
```python
from greenlang.determinism import calculate_provenance_hash, deterministic_uuid

# Create deterministic UUID for inspection
inspection_id = deterministic_uuid({
    "trap_id": "ST-001",
    "timestamp": "2024-01-15T10:30:00Z",
    "method": "acoustic"
})

# Calculate provenance hash for audit trail
inspection_data = {
    "trap_id": "ST-001",
    "status": "failed_open",
    "energy_loss_usd_yr": 15000.0,
    # ... all inspection results
}
provenance_hash = calculate_provenance_hash(inspection_data)

# Store in audit log
audit_log.append({
    "inspection_id": inspection_id,
    "provenance_hash": provenance_hash,
    "timestamp": clock.now().isoformat(),
    "data": inspection_data
})
```

### 3. Audit Log Storage

**Requirements:**
- Immutable storage (append-only)
- Tamper-evident (provenance hashes)
- Retention: Minimum 7 years (regulatory compliance)
- Encrypted at rest (AES-256)
- Encrypted in transit (TLS 1.3)

**Storage Options:**
1. **AWS S3 with Object Lock:** Immutable, compliant storage
2. **PostgreSQL with append-only table:** Relational audit trail
3. **Elasticsearch:** Searchable audit logs with immutable indices

### 4. Audit Log Access Control

**Access Policy:**
- Read-only access for auditors
- Write-only access for application
- No delete permissions for anyone
- Multi-factor authentication for access

---

## Security Validation

### 1. Startup Validation

**Implementation:**
```python
from agents.security_validator import validate_startup_security
from config import TrapInspectorConfig

# Load configuration
config = TrapInspectorConfig()

# Validate security at startup
try:
    validate_startup_security(config=config, fail_fast=True)
except SecurityValidationError as e:
    logger.critical(f"Security validation failed: {e}")
    logger.critical("STARTUP ABORTED")
    sys.exit(1)

# If validation passes, continue startup
logger.info("Security validation passed - starting application")
```

**Validation Checks:**
1. Hardcoded Credentials: No credentials in environment/config
2. API Keys: Valid format, minimum length, no test keys
3. Configuration Security: zero_secrets, audit logging, provenance enabled
4. Environment: Production settings validated
5. Rate Limiting: Reasonable limits configured
6. ML Model Integrity: Paths validated, versioning configured

### 2. Runtime Validation

**Continuous Checks:**
- API key expiration monitoring
- Rate limit enforcement
- Input validation on all API endpoints
- Provenance hash verification

### 3. Dependency Scanning

**Required Tools:**
```bash
# Python dependency scanning
pip install pip-audit
pip-audit

# Code security scanning
pip install bandit
bandit -r agents/ greenlang/

# Container scanning (if using Docker)
trivy image gl-008:latest
```

**Schedule:**
- Daily: Automated dependency scans
- Weekly: Manual security review
- Monthly: Penetration testing (if applicable)

---

## Compliance Checklist

### IEC 62443-4-2 Requirements

| Requirement | Description | Status | Implementation |
|------------|-------------|--------|----------------|
| SR 1.1 | User identification and authentication | PASS | API key validation |
| SR 1.5 | Authenticator management | PASS | Environment variables, ESO |
| SR 2.1 | Authorization enforcement | PASS | Zero-secrets policy |
| SR 3.1 | Communication integrity | PASS | HTTPS only (recommended) |
| SR 3.4 | Software and information integrity | PASS | Provenance hashing |
| SR 7.1 | Denial of service protection | PASS | Rate limiting |

### OWASP Top 10 Coverage

| Risk | Mitigation | Status |
|------|-----------|--------|
| A01: Broken Access Control | API key validation, rate limiting | PASS |
| A02: Cryptographic Failures | Strong key validation, no hardcoded secrets | PASS |
| A03: Injection | Input validation on all parameters | PASS |
| A04: Insecure Design | Security-by-default configuration | PASS |
| A05: Security Misconfiguration | Startup validation, production checks | PASS |
| A07: ID & Auth Failures | API key management, no default credentials | PASS |
| A08: Software/Data Integrity | Provenance hashing, audit logging | PASS |
| A09: Security Logging Failures | Comprehensive audit logging | PASS |

### NIST Cybersecurity Framework

| Function | Category | Controls | Status |
|----------|---------|----------|--------|
| Identify | Asset Management | Configuration tracking | PASS |
| Protect | Access Control | API key validation | PASS |
| Protect | Data Security | Encryption, provenance | PASS |
| Detect | Security Monitoring | Audit logging | PASS |
| Respond | Incident Response | See Incident Response section | READY |

---

## Incident Response

### 1. Security Incident Types

**Critical Incidents:**
- Exposed API keys or credentials
- Unauthorized access attempts
- Data integrity violations (provenance hash mismatch)
- Service compromise

**Response Time:**
- Critical: 1 hour
- High: 4 hours
- Medium: 24 hours

### 2. Incident Response Procedure

**Step 1: Detection**
- Monitor audit logs for anomalies
- Alert on security validation failures
- Track API key usage patterns

**Step 2: Containment**
```bash
# Immediately rotate compromised API key
aws secretsmanager update-secret \
  --secret-id greenlang/gl-008/anthropic_api_key \
  --secret-string "NEW_EMERGENCY_KEY"

# Revoke old key at provider
# Anthropic: https://console.anthropic.com/settings/keys

# Review audit logs for unauthorized usage
grep "anthropic_api_call" audit.log | grep "COMPROMISED_KEY"
```

**Step 3: Investigation**
- Review audit logs for incident timeline
- Verify provenance hashes for data integrity
- Identify affected systems

**Step 4: Recovery**
- Deploy new secrets
- Restart services with new credentials
- Verify security validation passes

**Step 5: Lessons Learned**
- Document incident
- Update security procedures
- Conduct team training

### 3. Emergency Contacts

```yaml
Security Team:
  - Primary: security@greenlang.io
  - On-Call: +1-XXX-XXX-XXXX

Vendor Contacts:
  - Anthropic Support: https://support.anthropic.com
  - AWS Security: aws-security@amazon.com
```

---

## Security Best Practices

### 1. Development

- Never commit secrets to version control
- Use `.env` files for local development (gitignored)
- Run security validator before every deployment
- Enable pre-commit hooks for secret scanning

### 2. Testing

```python
# Use DeterministicClock for reproducible tests
from greenlang.determinism import DeterministicClock

clock = DeterministicClock(test_mode=True)
clock.set_time("2024-01-01T00:00:00Z")

# Use deterministic UUIDs for test data
from greenlang.determinism import deterministic_uuid

test_id = deterministic_uuid("test_inspection_001")
```

### 3. Production Deployment

```bash
# Pre-deployment checklist:
# 1. Run security validation
python agents/security_validator.py

# 2. Verify environment variables set
echo $ANTHROPIC_API_KEY | wc -c  # Should be > 32

# 3. Check production settings
echo $GREENLANG_ENV  # Should be 'production'
echo $DEBUG  # Should be 'false'

# 4. Run dependency scan
pip-audit

# 5. Deploy with zero-downtime
kubectl apply -f deployment/
```

### 4. Monitoring

**Security Metrics:**
- API key validation failures
- Rate limit exceedances
- Security validation failures at startup
- Provenance hash mismatches

**Alerting:**
```yaml
Alerts:
  - name: security_validation_failure
    condition: security_validation_failed == true
    severity: critical
    action: page_oncall

  - name: api_key_invalid
    condition: api_key_validation_failed == true
    severity: critical
    action: rotate_key_immediately

  - name: rate_limit_exceeded
    condition: rate_limit_exceeded > 10 per hour
    severity: warning
    action: investigate_usage
```

---

## Appendix: Security Testing

### 1. Security Validation Test

```bash
# Test security validator
cd agents/
python security_validator.py

# Expected output:
# ================================================================================
# SECURITY VALIDATION - GL-008 SteamTrapInspector
# ================================================================================
# PASS Hardcoded Credentials: No hardcoded credentials detected
# PASS API Keys: API keys are valid
# PASS Configuration Security: Configuration security is valid
# PASS Environment: Environment is 'development' (development mode)
# PASS Rate Limiting: Rate limiting configuration is valid
# PASS ML Model Integrity: ML model configuration is valid
# ================================================================================
# SECURITY VALIDATION PASSED - All checks OK
# ================================================================================
```

### 2. Credential Leak Test

```bash
# Scan for accidentally committed secrets
git secrets --scan

# Use gitleaks for comprehensive scanning
gitleaks detect --source . --verbose
```

### 3. Penetration Testing

**Annual Penetration Test:**
- API endpoint security
- Authentication bypass attempts
- Injection attacks
- Rate limiting effectiveness

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2024-01-15 | GL-SecScan | Initial security policy |

**Review Schedule:** Quarterly
**Next Review:** 2024-04-15
**Approval:** Security Team Lead

---

**END OF SECURITY POLICY**
