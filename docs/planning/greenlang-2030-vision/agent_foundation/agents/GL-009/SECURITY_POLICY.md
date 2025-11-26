# GL-009 THERMALIQ - Security Policy

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Classification:** Internal - Security Documentation
**Owner:** GreenLang Foundation Security Team

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Security Standards Compliance](#2-security-standards-compliance)
- [3. Zero Secrets Policy](#3-zero-secrets-policy)
- [4. Authentication & Authorization](#4-authentication--authorization)
- [5. Data Security](#5-data-security)
- [6. Network Security](#6-network-security)
- [7. Audit Logging](#7-audit-logging)
- [8. Vulnerability Management](#8-vulnerability-management)
- [9. Security Architecture](#9-security-architecture)
- [10. Incident Response](#10-incident-response)
- [11. Compliance & Reporting](#11-compliance--reporting)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the comprehensive security policy for GL-009 THERMALIQ (ThermalEfficiencyCalculator), a zero-hallucination thermal efficiency analysis agent for industrial processes. This policy ensures the confidentiality, integrity, and availability of thermal efficiency calculations and industrial data.

### 1.2 Scope

This policy applies to:
- All GL-009 THERMALIQ deployments (development, staging, production)
- All personnel with access to GL-009 systems
- All integrations with external industrial systems (OPC UA, SCADA, ERP, Historians)
- All data processed by GL-009 (energy measurements, efficiency calculations, reports)

### 1.3 Security Posture

**Risk Level:** HIGH
**Data Classification:** Confidential (industrial performance data)
**Compliance Requirements:** IEC 62443-4-2, SOC 2 Type II, ISO 27001
**Zero Trust:** All connections authenticated and authorized

---

## 2. Security Standards Compliance

### 2.1 Industrial Automation Security

**IEC 62443-4-2 (Industrial Automation & Control Systems Security)**

GL-009 THERMALIQ complies with IEC 62443-4-2 requirements for industrial automation software:

| Requirement | Status | Implementation |
|------------|---------|----------------|
| SR 1.1 - User Identification & Authentication | ✅ Compliant | JWT-based authentication, API key rotation |
| SR 1.2 - Software Process & Device Identification | ✅ Compliant | Unique agent ID, version tracking, provenance hashing |
| SR 1.3 - Account Management | ✅ Compliant | RBAC with 4 roles, least privilege principle |
| SR 1.4 - Identifier Management | ✅ Compliant | UUID-based execution IDs, SHA-256 provenance |
| SR 1.5 - Authenticator Management | ✅ Compliant | Kubernetes secrets, External Secrets Operator |
| SR 1.7 - Strength of Password-based Authentication | ✅ Compliant | NIST SP 800-63B compliant (12+ chars, complexity) |
| SR 2.1 - Authorization Enforcement | ✅ Compliant | RBAC enforcement on all endpoints |
| SR 2.8 - Auditable Events | ✅ Compliant | All calculations logged with SHA-256 hash |
| SR 3.1 - Communication Integrity | ✅ Compliant | TLS 1.3, certificate pinning |
| SR 3.4 - Software & Information Integrity | ✅ Compliant | SHA-256 provenance, deterministic calculations |
| SR 4.1 - Information Confidentiality | ✅ Compliant | AES-256-GCM encryption at rest, TLS 1.3 in transit |

### 2.2 Application Security

**OWASP Top 10 (2023) Mitigation:**

| OWASP Risk | Mitigation | Status |
|-----------|-----------|---------|
| A01:2023 - Broken Access Control | RBAC, JWT validation, API rate limiting | ✅ Mitigated |
| A02:2023 - Cryptographic Failures | TLS 1.3, AES-256-GCM, no hardcoded secrets | ✅ Mitigated |
| A03:2023 - Injection | Pydantic validation, parameterized queries, no dynamic SQL | ✅ Mitigated |
| A04:2023 - Insecure Design | Zero-trust architecture, least privilege | ✅ Mitigated |
| A05:2023 - Security Misconfiguration | Hardened containers, security headers, no debug in prod | ✅ Mitigated |
| A06:2023 - Vulnerable Components | Dependency scanning (weekly), pinned versions | ✅ Mitigated |
| A07:2023 - Auth & Auth Failures | MFA support, session management, JWT expiration | ✅ Mitigated |
| A08:2023 - Software & Data Integrity | SHA-256 provenance, signed images, SBOM | ✅ Mitigated |
| A09:2023 - Security Logging Failures | Audit logging (7-year retention), SIEM integration | ✅ Mitigated |
| A10:2023 - Server-Side Request Forgery | URL validation, allowlist, no credential in URLs | ✅ Mitigated |

### 2.3 Infrastructure Security

**CIS Benchmarks:**

- **CIS Kubernetes Benchmark v1.8:** Pod Security Standards (Restricted), NetworkPolicies, RBAC
- **CIS Docker Benchmark v1.6:** Non-root containers, read-only root filesystem, resource limits
- **CIS Linux Benchmark v3.0:** Minimal base images (distroless), no SSH, hardened sysctls

### 2.4 Compliance Frameworks

**SOC 2 Type II Controls:**

- CC6.1: Logical access controls (RBAC, MFA)
- CC6.6: Encryption of data at rest and in transit
- CC7.2: System monitoring and incident detection
- CC7.3: Audit logging and retention

---

## 3. Zero Secrets Policy

### 3.1 Policy Statement

**CRITICAL:** GL-009 THERMALIQ enforces a ZERO SECRETS policy. No credentials, API keys, tokens, or sensitive data may be hardcoded in source code, configuration files, container images, or logs.

### 3.2 Secret Management

**Approved Secret Storage Methods:**

| Secret Type | Storage Method | Rotation Frequency |
|------------|----------------|-------------------|
| API Keys (Anthropic, OpenAI) | Kubernetes Secrets + External Secrets Operator | 90 days |
| Database Credentials | Kubernetes Secrets, AWS Secrets Manager | 30 days |
| OPC UA Credentials | External Secrets Operator (Vault, AWS) | 30 days |
| JWT Signing Keys | Kubernetes Secrets (sealed-secrets) | 180 days |
| TLS Certificates | cert-manager (automatic renewal) | 90 days (auto) |
| SCADA/Modbus Passwords | External Secrets Operator | 30 days |

**PROHIBITED:**
- ❌ Hardcoded credentials in Python files
- ❌ Credentials in `config.py`, `requirements.txt`, or YAML manifests
- ❌ API keys in environment variables (production)
- ❌ Secrets in Git commits, Dockerfiles, or CI/CD pipelines
- ❌ Credentials in log files or error messages

### 3.3 External Secrets Operator Configuration

**Example: Kubernetes Secret with External Secrets Operator**

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: gl009-anthropic-api-key
  namespace: greenlang-agents
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: gl009-secrets
    creationPolicy: Owner
  data:
    - secretKey: ANTHROPIC_API_KEY
      remoteRef:
        key: greenlang/gl009/anthropic-api-key
        version: latest
```

### 3.4 Secret Detection & Enforcement

**Automated Scanning:**
- **Pre-commit hooks:** `detect-secrets` (Yelp)
- **CI/CD pipeline:** `trufflehog`, `gitleaks`
- **Container scanning:** `trivy` secret detection
- **Runtime monitoring:** `falco` secret access monitoring

**Enforcement Actions:**
1. Pre-commit: Block commit if secrets detected
2. CI/CD: Fail build, notify security team
3. Production: Alert SOC, rotate compromised secrets within 1 hour

---

## 4. Authentication & Authorization

### 4.1 Authentication Mechanisms

**Supported Authentication Methods:**

| Method | Use Case | Security Level | MFA Required |
|--------|----------|---------------|--------------|
| JWT Bearer Tokens | API access (machine-to-machine) | High | No (short TTL) |
| API Keys | Integration authentication | High | No |
| OAuth 2.0 (Future) | User authentication | Very High | Yes |
| mTLS (mutual TLS) | Service-to-service | Very High | N/A |

**JWT Token Configuration:**
```python
# config.py JWT settings
JWT_ALGORITHM = "HS256"  # Use RS256 for production with key rotation
JWT_EXPIRATION_MINUTES = 60  # 1 hour
JWT_REFRESH_EXPIRATION_DAYS = 7  # Refresh token: 7 days
JWT_ISSUER = "greenlang-gl009-thermaliq"
JWT_AUDIENCE = ["greenlang-api", "industrial-scada"]
```

### 4.2 Role-Based Access Control (RBAC)

**GL-009 RBAC Roles:**

| Role | Permissions | Use Case | MFA Required |
|------|------------|----------|--------------|
| **viewer** | Read-only: GET /api/v1/*, /metrics, /health | Dashboards, monitoring | No |
| **operator** | Viewer + POST /api/v1/calculate, /api/v1/analyze | Daily operations | No |
| **analyst** | Operator + POST /api/v1/benchmark, /api/v1/optimize | Engineering analysis | Yes |
| **admin** | All permissions + /admin/*, secret rotation | System administration | Yes (REQUIRED) |

**RBAC Enforcement Example:**
```python
# Endpoint protection with RBAC
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def require_role(required_role: str):
    async def role_checker(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        token = credentials.credentials
        payload = verify_jwt_token(token)  # Validates signature, expiration
        user_role = payload.get("role")

        role_hierarchy = {"viewer": 1, "operator": 2, "analyst": 3, "admin": 4}
        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 999):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_role}"
            )
        return payload
    return role_checker

# Usage in endpoints
@app.post("/api/v1/analyze", dependencies=[Depends(require_role("analyst"))])
async def analyze_efficiency(request: CalculationRequest):
    # Only analysts and admins can access
    pass
```

### 4.3 API Key Management

**API Key Lifecycle:**
1. **Generation:** Cryptographically random (32 bytes), prefix: `gl_sk_th_`
2. **Storage:** Hashed (bcrypt) in database, never stored plaintext
3. **Distribution:** Secure channel (HTTPS, encrypted email)
4. **Rotation:** Mandatory 90-day rotation, 7-day grace period
5. **Revocation:** Immediate revocation on compromise

**API Key Format:**
```
gl_sk_th_<environment>_<random>
Example: gl_sk_th_prod_a8f3k2j9x7m1p5q4
```

### 4.4 Session Management

**Session Security:**
- Session timeout: 1 hour (configurable: 15 min - 24 hours)
- Idle timeout: 15 minutes
- Concurrent sessions: Max 3 per user
- Session revocation: Global logout capability

---

## 5. Data Security

### 5.1 Data Classification

| Classification | Description | Examples | Encryption | Retention |
|---------------|-------------|----------|------------|-----------|
| **Public** | Non-sensitive, publishable | API schemas, documentation | No | Indefinite |
| **Internal** | Business data, not PII | Aggregate statistics, benchmarks | Optional | 7 years |
| **Confidential** | Sensitive business data | Energy consumption, efficiency metrics | Required | 7 years |
| **Restricted** | Highly sensitive, regulated | Process secrets, IP, PII | Required (AES-256) | 7 years |

### 5.2 Encryption at Rest

**Data Store Encryption:**

| Data Store | Encryption Method | Key Management |
|-----------|------------------|---------------|
| PostgreSQL | AES-256-GCM (pg_crypto) | AWS KMS, rotating keys |
| Redis (cache) | AES-256-CBC (stunnel) | Kubernetes Secrets |
| S3 (reports, logs) | SSE-S3 (AES-256) | AWS-managed keys |
| EBS volumes | EBS encryption (AES-256) | AWS KMS |
| Backups | GPG encryption (4096-bit RSA) | Hardware Security Module |

**Encryption Configuration Example:**
```python
# PostgreSQL encryption (application-level)
from cryptography.fernet import Fernet

def encrypt_sensitive_field(plaintext: str, key: bytes) -> str:
    """Encrypt sensitive fields before storing in database."""
    cipher = Fernet(key)
    encrypted = cipher.encrypt(plaintext.encode())
    return encrypted.decode()

def decrypt_sensitive_field(ciphertext: str, key: bytes) -> str:
    """Decrypt sensitive fields after retrieving from database."""
    cipher = Fernet(key)
    decrypted = cipher.decrypt(ciphertext.encode())
    return decrypted.decode()
```

### 5.3 Encryption in Transit

**TLS Configuration:**
- **Minimum TLS version:** TLS 1.3 (TLS 1.2 fallback for legacy systems)
- **Cipher suites:** ECDHE-RSA-AES256-GCM-SHA384, ECDHE-RSA-AES128-GCM-SHA256
- **Certificate:** Let's Encrypt (auto-renewal via cert-manager)
- **HSTS:** Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
- **Certificate pinning:** Enabled for critical integrations (OPC UA, SCADA)

**HTTP Security Headers:**
```python
# FastAPI middleware for security headers
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

# Redirect HTTP to HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

### 5.4 Data Retention & Disposal

**Retention Policy:**
| Data Type | Retention Period | Disposal Method |
|-----------|-----------------|----------------|
| Calculation results | 7 years | Secure deletion (NIST 800-88) |
| Audit logs | 7 years | Archival to S3 Glacier |
| Performance metrics | 1 year | Rolling aggregation |
| Debug logs | 30 days | Automatic purge |
| API access logs | 90 days | SIEM archival |

**Secure Deletion:**
```bash
# NIST 800-88 compliant secure deletion
# Using shred for file-based data
shred -vfz -n 3 /data/gl009/calculations/*.json

# PostgreSQL secure deletion (overwrite + VACUUM FULL)
DELETE FROM thermal_calculations WHERE created_at < NOW() - INTERVAL '7 years';
VACUUM FULL thermal_calculations;
```

---

## 6. Network Security

### 6.1 Network Segmentation

**Kubernetes NetworkPolicy:**

```yaml
# Ingress: Only allow traffic from NGINX ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gl009-ingress-policy
  namespace: greenlang-agents
spec:
  podSelector:
    matchLabels:
      app: gl009-thermaliq
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from NGINX ingress
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8009
    # Allow from Prometheus
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
  egress:
    # Allow to PostgreSQL
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432
    # Allow to Redis
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    # Allow DNS
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
    # Allow HTTPS egress for Anthropic API
    - to:
        - podSelector: {}
      ports:
        - protocol: TCP
          port: 443
```

### 6.2 Rate Limiting & DDoS Protection

**Rate Limiting Configuration:**

| Endpoint Category | Rate Limit | Burst | Enforcement |
|------------------|-----------|-------|-------------|
| /health, /ready | 100 req/min | 200 | NGINX Ingress |
| /metrics | 10 req/min | 20 | Prometheus scraper |
| /api/v1/calculate | 60 req/min per API key | 100 | FastAPI Limiter |
| /api/v1/analyze | 30 req/min per API key | 50 | FastAPI Limiter |
| /api/v1/report | 10 req/min per API key | 15 | FastAPI Limiter |

**Implementation:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/calculate")
@limiter.limit("60/minute")
async def calculate_efficiency(request: Request, calc_request: CalculationRequest):
    # Rate limited to 60 requests per minute per IP
    pass
```

### 6.3 Firewall Rules & IP Allowlisting

**Production Environment:**
```yaml
# AWS Security Group for GL-009 EKS worker nodes
Type: Ingress
Port Range: 8009 (HTTPS only)
Source:
  - ALB Security Group (NGINX Ingress)
  - Corporate VPN CIDR: 10.0.0.0/8
  - Industrial network: 192.168.100.0/24 (OPC UA, Modbus)
```

**IP Allowlist for Admin API:**
```python
# Admin endpoints restricted to corporate network
ADMIN_IP_ALLOWLIST = [
    "10.0.0.0/8",         # Corporate network
    "203.0.113.0/24",     # Engineering offices
    "198.51.100.0/24",    # SOC/NOC
]

@app.post("/admin/rotate-secrets")
async def rotate_secrets(request: Request):
    client_ip = request.client.host
    if not ip_in_allowlist(client_ip, ADMIN_IP_ALLOWLIST):
        raise HTTPException(status_code=403, detail="IP not allowed")
    # Rotate secrets
```

---

## 7. Audit Logging

### 7.1 Audit Logging Requirements

**Logged Events:**
| Event Category | Examples | Log Level | Retention |
|---------------|----------|-----------|-----------|
| Authentication | Login, logout, API key usage, JWT validation | INFO | 7 years |
| Authorization | RBAC denials, permission checks | WARN | 7 years |
| Data Access | Calculation requests, report generation | INFO | 7 years |
| Configuration Changes | Secret rotation, config updates | WARN | 7 years |
| Security Events | Failed auth, rate limit exceeded, anomalies | ERROR | 7 years |
| System Events | Startup, shutdown, health checks | INFO | 90 days |

### 7.2 Audit Log Format

**Structured Logging (JSON):**
```json
{
  "timestamp": "2025-11-26T14:32:15.123Z",
  "level": "INFO",
  "event_type": "calculation_request",
  "agent_id": "GL-009",
  "execution_id": "GL009-2025-11-26-a8f3k2j9",
  "user": {
    "api_key_id": "gl_sk_th_prod_xyz123",
    "role": "analyst",
    "ip_address": "192.168.100.45"
  },
  "action": "POST /api/v1/analyze",
  "resource": "thermal_efficiency_calculation",
  "operation_mode": "analyze",
  "result": "success",
  "provenance_hash": "sha256:7f8a3b2c1d9e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1",
  "processing_time_ms": 234.56,
  "metadata": {
    "first_law_efficiency_percent": 87.3,
    "second_law_efficiency_percent": 42.1,
    "benchmark_percentile": 75
  }
}
```

### 7.3 Provenance Tracking

**SHA-256 Provenance Hashing:**

All calculations include a SHA-256 provenance hash for audit trail and tamper detection:

```python
import hashlib
import json
from datetime import datetime, timezone

def calculate_provenance_hash(input_data: dict, result: dict) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    Ensures:
    - Input data integrity
    - Result traceability
    - Tamper detection
    """
    provenance_data = {
        'agent_id': 'GL-009',
        'version': '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'input_hash': hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest(),
        'result_summary': {
            'first_law_efficiency': result.get('first_law_efficiency_percent'),
            'second_law_efficiency': result.get('second_law_efficiency_percent')
        }
    }
    return hashlib.sha256(
        json.dumps(provenance_data, sort_keys=True).encode()
    ).hexdigest()
```

### 7.4 Log Aggregation & SIEM Integration

**Log Pipeline:**
```
GL-009 Container (JSON logs)
  → Fluentd (log aggregator)
    → Elasticsearch (indexing, 7-year retention)
      → Kibana (visualization, alerting)
      → SIEM (Splunk, QRadar, Sentinel)
```

**SIEM Alerts:**
- Failed authentication (>5 in 5 minutes)
- RBAC denials for admin endpoints
- Unusual calculation patterns (>3 std dev from baseline)
- Secret access from unexpected IPs
- High error rates (>10% of requests)

---

## 8. Vulnerability Management

### 8.1 Vulnerability Scanning Schedule

| Scan Type | Frequency | Tool | SLA for Remediation |
|-----------|-----------|------|-------------------|
| **Container image scanning** | Every build | Trivy, Grype | Critical: 24h, High: 7d |
| **Dependency scanning (Python)** | Weekly | Safety, pip-audit | Critical: 48h, High: 14d |
| **SAST (Static code analysis)** | Every commit | Bandit, Semgrep | Critical: 7d, High: 30d |
| **DAST (Dynamic testing)** | Monthly | OWASP ZAP | Critical: 14d, High: 30d |
| **Infrastructure scanning** | Weekly | Checkov, tfsec | Critical: 7d, High: 30d |
| **Penetration testing** | Quarterly | External firm | All findings: 90d |

### 8.2 Dependency Vulnerability Management

**Automated Dependency Updates:**
```yaml
# Renovate Bot configuration (.renovate.json)
{
  "extends": ["config:base"],
  "schedule": ["before 4am on monday"],
  "separateMajorMinor": true,
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"],
    "assignees": ["@security-team"]
  },
  "packageRules": [
    {
      "matchPackagePatterns": ["*"],
      "matchUpdateTypes": ["patch"],
      "automerge": true,
      "automergeType": "pr",
      "requiredStatusChecks": null
    }
  ]
}
```

**CVE Response Process:**
1. **Detection:** Automated scanning (Safety, GitHub Dependabot)
2. **Triage:** Security team reviews CVE within 4 hours (critical), 24 hours (high)
3. **Patching:** Update dependency, test, deploy
4. **Verification:** Re-scan to confirm patch
5. **Documentation:** Update security audit log

### 8.3 Known CVE Thresholds

**Acceptable CVE Levels:**
| Severity | Maximum Allowed | Action Required |
|----------|----------------|-----------------|
| Critical (CVSS 9.0-10.0) | 0 | Immediate patch (4 hours) |
| High (CVSS 7.0-8.9) | 3 | Patch within 7 days |
| Medium (CVSS 4.0-6.9) | 10 | Patch within 30 days |
| Low (CVSS 0.1-3.9) | Unlimited | Best-effort patching |

### 8.4 Security Testing

**Continuous Security Testing:**
```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday 2 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # SAST: Static code analysis
      - name: Bandit Security Scan
        run: bandit -r . -f json -o bandit-report.json

      # Dependency scanning
      - name: Safety Check
        run: safety check --json --output safety-report.json

      # Container scanning
      - name: Trivy Container Scan
        run: |
          docker build -t gl009:test .
          trivy image --severity CRITICAL,HIGH --exit-code 1 gl009:test

      # Secret scanning
      - name: TruffleHog Secret Scan
        run: trufflehog filesystem . --json > trufflehog-report.json

      # Upload results to security dashboard
      - name: Upload to DefectDojo
        run: |
          curl -X POST https://defectdojo.greenlang.io/api/v2/import-scan/ \
            -H "Authorization: Token ${{ secrets.DEFECTDOJO_TOKEN }}" \
            -F "scan_type=Bandit Scan" \
            -F "file=@bandit-report.json"
```

---

## 9. Security Architecture

### 9.1 Zero-Trust Architecture

**Principles:**
1. **Never trust, always verify:** All connections authenticated
2. **Least privilege:** Minimal permissions for all roles
3. **Assume breach:** Defense in depth, monitoring
4. **Verify explicitly:** Authenticate and authorize every request
5. **Use least privileged access:** Just-in-time, just-enough access

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Internet / Corporate Network              │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS (TLS 1.3)
                            ▼
                ┌─────────────────────────┐
                │   NGINX Ingress         │
                │   - WAF (ModSecurity)   │
                │   - Rate limiting       │
                │   - TLS termination     │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │   GL-009 THERMALIQ      │
                │   - JWT validation      │
                │   - RBAC enforcement    │
                │   - Input validation    │
                └───────────┬─────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  PostgreSQL  │ │    Redis     │ │  OPC UA      │
    │  (encrypted) │ │  (cache)     │ │  (mTLS)      │
    └──────────────┘ └──────────────┘ └──────────────┘
```

### 9.2 Secure Communication Channels

**Industrial Protocol Security:**

| Protocol | Security Enhancement | Configuration |
|----------|---------------------|---------------|
| **OPC UA** | Security Policy: Sign & Encrypt | MessageSecurityMode: SignAndEncrypt |
| **Modbus TCP** | TLS wrapper (stunnel) | Encrypted tunnel on port 8502 |
| **MQTT** | TLS + username/password | TLS 1.3, client certificates |
| **BACnet** | BACnet/SC (Secure Connect) | AES-128-GCM encryption |

### 9.3 Container Security

**Pod Security Standards (Restricted):**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gl009-thermaliq
  namespace: greenlang-agents
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    fsGroup: 10001
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: gl009
      image: greenlang/gl009-thermaliq:1.0.0
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        runAsNonRoot: true
        runAsUser: 10001
        capabilities:
          drop:
            - ALL
      resources:
        limits:
          cpu: "2000m"
          memory: "4Gi"
        requests:
          cpu: "500m"
          memory: "1Gi"
      livenessProbe:
        httpGet:
          path: /health
          port: 8009
        initialDelaySeconds: 10
        periodSeconds: 30
      readinessProbe:
        httpGet:
          path: /ready
          port: 8009
        initialDelaySeconds: 5
        periodSeconds: 10
```

---

## 10. Incident Response

### 10.1 Security Incident Classification

| Severity | Response Time | Examples | Escalation |
|----------|--------------|----------|------------|
| **P0 - Critical** | 15 minutes | Data breach, ransomware, RCE | CISO, CEO |
| **P1 - High** | 1 hour | Compromised API key, unauthorized access | Security team lead |
| **P2 - Medium** | 4 hours | Vulnerability scan findings | Security engineer |
| **P3 - Low** | 24 hours | Policy violations, false positives | Automated ticket |

### 10.2 Incident Response Playbook

**Suspected API Key Compromise:**

1. **Detection (0-5 min):**
   - Alert triggered by SIEM (unusual access pattern)
   - Manual report from user

2. **Containment (5-15 min):**
   - Immediately revoke compromised API key
   - Block source IP in WAF
   - Isolate affected systems

3. **Eradication (15-60 min):**
   - Rotate all related secrets
   - Scan for lateral movement
   - Patch vulnerability (if applicable)

4. **Recovery (1-4 hours):**
   - Issue new API key to legitimate user
   - Restore services
   - Monitor for reoccurrence

5. **Post-Incident (24-48 hours):**
   - Root cause analysis (RCA)
   - Update runbooks
   - Security awareness training

**Runbook Example:**
```bash
#!/bin/bash
# Incident Response: API Key Compromise

# 1. Revoke compromised API key
API_KEY_ID="gl_sk_th_prod_xyz123"
psql -U gl009 -d greenlang -c \
  "UPDATE api_keys SET revoked=true, revoked_at=NOW() WHERE key_id='${API_KEY_ID}';"

# 2. Block attacker IP
ATTACKER_IP="203.0.113.45"
kubectl exec -n ingress-nginx deploy/ingress-nginx-controller -- \
  nginx -s reload -c <(cat <<EOF
deny ${ATTACKER_IP};
EOF
)

# 3. Rotate related secrets
kubectl delete secret gl009-secrets -n greenlang-agents
kubectl create secret generic gl009-secrets --from-env-file=.env.prod

# 4. Notify security team
curl -X POST https://slack.com/api/chat.postMessage \
  -H "Authorization: Bearer ${SLACK_TOKEN}" \
  -d "channel=#security-incidents" \
  -d "text=INCIDENT: API key ${API_KEY_ID} revoked. Attacker IP: ${ATTACKER_IP}"
```

### 10.3 Breach Notification

**Regulatory Requirements:**
- **GDPR:** 72-hour notification to supervisory authority
- **CCPA:** "Without unreasonable delay"
- **HIPAA:** 60 days (if PHI involved)
- **SOC 2:** Notify customers within 48 hours

---

## 11. Compliance & Reporting

### 11.1 Compliance Audits

| Audit Type | Frequency | Auditor | Next Audit |
|-----------|-----------|---------|-----------|
| **SOC 2 Type II** | Annual | External (Big 4 firm) | Q2 2026 |
| **ISO 27001** | Annual | External auditor | Q3 2026 |
| **IEC 62443-4-2** | Biennial | ISA certified auditor | Q1 2027 |
| **Internal security audit** | Quarterly | Internal audit team | Q1 2026 |

### 11.2 Security Metrics & Reporting

**Monthly Security Dashboard:**
- Total vulnerabilities detected / remediated
- Mean time to patch (MTTP) for critical CVEs
- API key rotation compliance (% rotated within 90 days)
- Failed authentication attempts
- RBAC policy violations
- Audit log completeness

### 11.3 Compliance Documentation

**Required Documentation:**
- Security policy (this document)
- Incident response plan
- Business continuity plan
- Disaster recovery plan
- Acceptable use policy
- Data processing agreement (GDPR)
- Security awareness training records

---

## Appendix A: Security Contacts

| Role | Contact | Email | Phone |
|------|---------|-------|-------|
| **CISO** | Jane Smith | jane.smith@greenlang.io | +1-555-0001 |
| **Security Team Lead** | John Doe | john.doe@greenlang.io | +1-555-0002 |
| **Incident Response (24/7)** | SOC Team | soc@greenlang.io | +1-555-9999 |
| **Vulnerability Reporting** | Security Team | security@greenlang.io | - |

---

## Appendix B: Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | Security Team | Initial security policy for GL-009 THERMALIQ |

---

**Document Classification:** Internal - Security Documentation
**Review Cycle:** Quarterly
**Next Review:** 2026-02-26
**Approval:** CISO, Security Team Lead, GL-009 Product Owner
