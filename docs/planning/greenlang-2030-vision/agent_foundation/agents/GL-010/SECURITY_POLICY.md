# GL-010 EMISSIONWATCH Security Policy

**Document Classification:** REGULATORY SENSITIVE
**Agent:** GL-010 EMISSIONWATCH - EmissionsComplianceAgent
**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** GreenLang Foundation Security Team
**Review Cycle:** Quarterly

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Security Classification](#2-security-classification)
3. [Regulatory Compliance Requirements](#3-regulatory-compliance-requirements)
4. [Access Control Policy](#4-access-control-policy)
5. [Data Protection Standards](#5-data-protection-standards)
6. [Authentication and Authorization](#6-authentication-and-authorization)
7. [Secrets Management](#7-secrets-management)
8. [Network Security](#8-network-security)
9. [Audit and Logging](#9-audit-and-logging)
10. [Vulnerability Management](#10-vulnerability-management)
11. [Incident Response](#11-incident-response)
12. [Container Security](#12-container-security)
13. [Kubernetes Security](#13-kubernetes-security)
14. [API Security](#14-api-security)
15. [Cryptographic Standards](#15-cryptographic-standards)
16. [Security Testing](#16-security-testing)
17. [Compliance Verification](#17-compliance-verification)
18. [Policy Exceptions](#18-policy-exceptions)
19. [Document Control](#19-document-control)

---

## 1. Executive Summary

### 1.1 Purpose

This Security Policy document establishes the comprehensive security requirements, controls, and procedures for the GL-010 EMISSIONWATCH EmissionsComplianceAgent. The agent handles Regulatory Sensitive emissions data subject to EPA 40 CFR Parts 60 and 75, EU Industrial Emissions Directive 2010/75/EU, and China MEE GB 13223-2011 standards.

### 1.2 Scope

This policy applies to:
- All GL-010 EMISSIONWATCH components, including Python application code, configuration files, and deployment manifests
- All environments: development, staging, and production
- All personnel with access to GL-010 systems or data
- All third-party integrations and external connections
- All data processed, stored, or transmitted by the agent

### 1.3 Security Objectives

| Objective | Description | Priority |
|-----------|-------------|----------|
| Confidentiality | Protect emissions data from unauthorized disclosure | Critical |
| Integrity | Ensure accuracy and completeness of emissions calculations | Critical |
| Availability | Maintain 99.9% uptime for regulatory compliance | High |
| Auditability | Provide complete audit trails with cryptographic verification | Critical |
| Non-repudiation | Ensure provenance tracking with SHA-256 hashing | High |

### 1.4 Key Security Features

GL-010 EMISSIONWATCH implements the following core security features:

1. **Zero-Hallucination Calculations**: All emissions calculations use deterministic physics-based formulas with no LLM involvement in numeric computations
2. **SHA-256 Provenance Hashing**: Complete audit trail with cryptographic verification
3. **Zero Hardcoded Credentials**: All secrets loaded from environment variables or Kubernetes secrets
4. **Thread-Safe Operations**: Concurrent request handling with proper synchronization
5. **Input Validation**: Pydantic-based validation with field constraints

---

## 2. Security Classification

### 2.1 Data Classification Levels

| Level | Description | Examples | Handling Requirements |
|-------|-------------|----------|----------------------|
| REGULATORY SENSITIVE | Data subject to EPA/EU/China regulatory requirements | CEMS readings, emissions reports, violation records | Encryption required, 7-year retention, audit logging |
| CONFIDENTIAL | Business-sensitive operational data | Process parameters, fuel data, permit limits | Encryption required, access controls |
| INTERNAL | Non-public operational information | Configuration settings, performance metrics | Access controls, standard logging |
| PUBLIC | Information cleared for public release | API documentation, general capabilities | No restrictions |

### 2.2 GL-010 Data Classification Matrix

| Data Category | Classification | Encryption at Rest | Encryption in Transit | Retention Period |
|--------------|----------------|-------------------|----------------------|------------------|
| CEMS Measurements | REGULATORY SENSITIVE | AES-256 | TLS 1.3 | 7 years |
| Emissions Calculations | REGULATORY SENSITIVE | AES-256 | TLS 1.3 | 7 years |
| Compliance Reports | REGULATORY SENSITIVE | AES-256 | TLS 1.3 | 7 years |
| Violation Records | REGULATORY SENSITIVE | AES-256 | TLS 1.3 | 7 years |
| Audit Trails | REGULATORY SENSITIVE | AES-256 | TLS 1.3 | 7 years |
| Provenance Hashes | REGULATORY SENSITIVE | AES-256 | TLS 1.3 | 7 years |
| Fuel Analysis Data | CONFIDENTIAL | AES-256 | TLS 1.3 | 5 years |
| Process Parameters | CONFIDENTIAL | AES-256 | TLS 1.3 | 3 years |
| Performance Metrics | INTERNAL | Optional | TLS 1.3 | 1 year |
| Health Check Data | INTERNAL | No | TLS 1.3 | 30 days |

### 2.3 Regulatory Data Handling

Emissions data processed by GL-010 is subject to the following regulatory frameworks:

**EPA 40 CFR Part 75 Requirements:**
- Data must be retained for minimum 3 years after submission
- Substitute data procedures must be documented
- Quality assurance must meet RATA requirements
- Electronic data must be tamper-evident

**EU Industrial Emissions Directive 2010/75/EU:**
- Continuous monitoring data must be available for inspection
- Annual reports must be submitted to competent authorities
- Data integrity must be verifiable

**China MEE GB 13223-2011:**
- Real-time monitoring data transmission required
- Data must meet ultra-low emission verification standards
- Compliance records must be maintained

---

## 3. Regulatory Compliance Requirements

### 3.1 Compliance Framework Matrix

| Regulation | Requirement | GL-010 Implementation |
|------------|-------------|----------------------|
| EPA 40 CFR Part 60 | NSPS emissions limits | Deterministic compliance checking |
| EPA 40 CFR Part 75 | CEMS monitoring requirements | Data quality validation, substitute data |
| EU IED 2010/75/EU | BAT-AEL limits | Multi-jurisdiction compliance checking |
| China GB 13223-2011 | Ultra-low emissions | China MEE format reporting |
| SOC 2 Type II | Security controls | Access controls, audit logging |
| ISO 27001 | Information security | ISMS compliance |
| NIST 800-53 | Security controls | Control implementation |

### 3.2 SOC 2 Type II Compliance

GL-010 implements controls for all five SOC 2 Trust Service Criteria:

**Security:**
- Network segmentation and firewall rules
- Encryption at rest (AES-256) and in transit (TLS 1.3)
- Multi-factor authentication for administrative access
- Vulnerability management program

**Availability:**
- 99.9% SLA with Kubernetes high availability
- Horizontal pod autoscaling
- Pod disruption budgets
- Health checks and automatic restart

**Processing Integrity:**
- Deterministic calculations with zero hallucination
- SHA-256 provenance hashing
- Input validation with Pydantic
- Data quality verification

**Confidentiality:**
- Data classification enforcement
- Access control lists
- Encryption key management
- Secure credential handling

**Privacy:**
- Data minimization
- Purpose limitation
- Retention policies
- Access logging

### 3.3 EPA Electronic Reporting Requirements

GL-010 meets EPA ECMPS (Emissions Collection and Monitoring Plan System) requirements:

- XML schema compliance for electronic data reporting
- Digital signature support for certifier attestation
- Data validation before submission
- Error handling and correction procedures
- Audit trail for all modifications

---

## 4. Access Control Policy

### 4.1 Role-Based Access Control (RBAC) Matrix

| Role | Description | Permissions |
|------|-------------|-------------|
| emissions_admin | Full administrative access | All operations, configuration changes |
| emissions_operator | Day-to-day operations | Execute calculations, view reports |
| emissions_analyst | Report generation and analysis | View data, generate reports |
| emissions_auditor | Compliance audit access | Read-only access to all audit data |
| emissions_viewer | Basic read access | View dashboards and summaries |
| system_admin | Infrastructure administration | Kubernetes, deployment operations |
| security_admin | Security operations | Access control, security configuration |

### 4.2 Kubernetes RBAC Configuration

```yaml
# ServiceAccount for GL-010
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gl-010-emissionwatch
  namespace: greenlang-agents
  labels:
    app: gl-010-emissionwatch
    security-tier: regulatory-sensitive

---
# Role for emissions operations
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gl-010-emissions-role
  namespace: greenlang-agents
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["gl-010-secrets", "gl-010-api-keys"]
    verbs: ["get"]
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["gl-010-config"]
    verbs: ["get", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: gl-010-emissions-rolebinding
  namespace: greenlang-agents
subjects:
  - kind: ServiceAccount
    name: gl-010-emissionwatch
    namespace: greenlang-agents
roleRef:
  kind: Role
  name: gl-010-emissions-role
  apiGroup: rbac.authorization.k8s.io
```

### 4.3 Least Privilege Principle

All GL-010 components operate under the principle of least privilege:

1. **Service Account**: Minimal permissions required for operation
2. **Container Security Context**: Non-root user, read-only root filesystem
3. **Network Policies**: Ingress/egress restricted to required services
4. **File System**: Only /tmp and specific data directories writable

### 4.4 Access Review Requirements

| Review Type | Frequency | Scope | Responsible Party |
|-------------|-----------|-------|-------------------|
| User Access Review | Quarterly | All user accounts | Security Admin |
| Service Account Review | Monthly | Kubernetes service accounts | System Admin |
| API Key Audit | Monthly | All API keys and tokens | Security Admin |
| Privilege Escalation Review | Weekly | Elevated access events | Security Admin |
| Separation of Duties | Quarterly | Role assignments | Compliance Officer |

---

## 5. Data Protection Standards

### 5.1 Encryption at Rest

**Requirements:**
- All REGULATORY SENSITIVE data must be encrypted at rest using AES-256
- Encryption keys must be managed through Kubernetes Secrets or external KMS
- Key rotation must occur every 90 days

**Implementation:**

```python
# Encryption configuration in config.py
class EncryptionConfig(BaseModel):
    """Encryption configuration for data at rest."""

    algorithm: str = Field(
        default="AES-256-GCM",
        description="Encryption algorithm"
    )

    key_derivation: str = Field(
        default="PBKDF2-HMAC-SHA256",
        description="Key derivation function"
    )

    key_rotation_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Key rotation interval in days"
    )
```

### 5.2 Encryption in Transit

**Requirements:**
- All network communications must use TLS 1.3
- Mutual TLS (mTLS) required for service-to-service communication
- Certificate validity maximum 90 days

**TLS Configuration:**

```yaml
# Kubernetes Ingress TLS configuration
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gl-010-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.3"
    nginx.ingress.kubernetes.io/ssl-ciphers: "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
spec:
  tls:
    - hosts:
        - gl-010.greenlang.io
      secretName: gl-010-tls-secret
```

### 5.3 Data Retention Policy

| Data Type | Retention Period | Archive Location | Destruction Method |
|-----------|------------------|------------------|-------------------|
| CEMS Raw Data | 7 years | Cold storage | Secure deletion |
| Emissions Reports | 7 years | Archive storage | Secure deletion |
| Audit Logs | 7 years | Log archive | Secure deletion |
| Provenance Hashes | 7 years | Immutable storage | N/A (hash only) |
| API Access Logs | 1 year | Log storage | Automatic purge |
| Performance Metrics | 1 year | Metrics storage | Automatic purge |
| Cache Data | 5 minutes | Memory | Automatic expiry |

### 5.4 Data Backup and Recovery

**Backup Requirements:**
- Daily incremental backups of all emissions data
- Weekly full backups
- Geographic redundancy (minimum 2 regions)
- Backup encryption using separate key hierarchy

**Recovery Objectives:**
- Recovery Time Objective (RTO): 4 hours
- Recovery Point Objective (RPO): 1 hour
- Backup testing frequency: Monthly

---

## 6. Authentication and Authorization

### 6.1 Authentication Methods

| Method | Use Case | Requirements |
|--------|----------|--------------|
| API Key | Service-to-service | Minimum 32 characters, SHA-256 hashed storage |
| JWT Token | User authentication | RS256 signing, 1-hour expiry, refresh tokens |
| mTLS Certificate | Inter-service | X.509 v3, 2048-bit RSA or P-256 ECDSA |
| EPA CDX Certificate | EPA reporting | EPA-issued X.509, annual renewal |

### 6.2 API Key Management

**Generation Requirements:**
```python
import secrets
import hashlib

def generate_api_key() -> tuple[str, str]:
    """
    Generate a secure API key with hash for storage.

    Returns:
        Tuple of (plaintext_key, hashed_key)
    """
    # Generate 32-byte random key
    key_bytes = secrets.token_bytes(32)
    plaintext_key = secrets.token_urlsafe(32)

    # Hash for storage
    hashed_key = hashlib.sha256(plaintext_key.encode()).hexdigest()

    return plaintext_key, hashed_key
```

**API Key Security Rules:**
1. Keys must never be logged in plaintext
2. Keys must be transmitted only over TLS 1.3
3. Keys must be stored as SHA-256 hashes
4. Keys must be rotated every 90 days
5. Compromised keys must be revoked immediately

### 6.3 JWT Token Configuration

```python
# JWT configuration for GL-010
JWT_CONFIG = {
    "algorithm": "RS256",
    "access_token_expire_minutes": 60,
    "refresh_token_expire_days": 7,
    "issuer": "gl-010-emissionwatch",
    "audience": "greenlang-agents",
    "required_claims": ["sub", "iat", "exp", "roles", "jurisdiction"]
}
```

### 6.4 EPA CDX Certificate Requirements

For EPA ECMPS electronic reporting:
- X.509 v3 certificate issued by EPA CDX
- Annual certificate renewal required
- Certificate must be stored in Kubernetes Secret
- Private key must never be exported

---

## 7. Secrets Management

### 7.1 Zero Hardcoded Secrets Policy

**MANDATORY REQUIREMENT: No credentials, API keys, tokens, or sensitive configuration values may be hardcoded in source code, configuration files, or container images.**

**Enforcement:**
```python
# From config.py - validation preventing hardcoded secrets
@field_validator('zero_secrets')
@classmethod
def validate_zero_secrets(cls, v):
    """Ensure zero_secrets policy is enabled."""
    if not v:
        raise ValueError(
            "SECURITY VIOLATION: zero_secrets must be True. "
            "No credentials allowed in configuration."
        )
    return v
```

### 7.2 Secret Sources

| Secret Type | Storage Location | Access Method |
|-------------|------------------|---------------|
| API Keys | Kubernetes Secret | Environment variable |
| Database Credentials | Kubernetes Secret | Volume mount |
| TLS Certificates | Kubernetes Secret | Volume mount |
| EPA CDX Certificate | Kubernetes Secret | Volume mount |
| Encryption Keys | External KMS | KMS API |
| OAuth Client Secrets | Kubernetes Secret | Environment variable |

### 7.3 Kubernetes Secrets Configuration

```yaml
# Secret definition for GL-010
apiVersion: v1
kind: Secret
metadata:
  name: gl-010-secrets
  namespace: greenlang-agents
  labels:
    app: gl-010-emissionwatch
type: Opaque
data:
  # Base64-encoded values - actual secrets managed externally
  ANTHROPIC_API_KEY: <base64-encoded>
  EPA_ECMPS_API_KEY: <base64-encoded>
  DATABASE_URL: <base64-encoded>
  ENCRYPTION_KEY: <base64-encoded>
```

### 7.4 Secret Rotation Schedule

| Secret Type | Rotation Frequency | Rotation Method |
|-------------|-------------------|-----------------|
| API Keys | 90 days | Automated rotation |
| Database Passwords | 90 days | Kubernetes Secret update |
| TLS Certificates | 90 days | cert-manager auto-renewal |
| Encryption Keys | 90 days | KMS key rotation |
| Service Account Tokens | 24 hours | Kubernetes auto-rotation |

### 7.5 URL Credential Validation

All URLs are validated to prevent embedded credentials:

```python
# From config.py
@field_validator('webhook_url')
@classmethod
def validate_no_credentials_in_url(cls, v):
    """Validate URL does not contain embedded credentials."""
    if v is not None:
        parsed = urlparse(v)
        if parsed.username or parsed.password:
            raise ValueError(
                "SECURITY VIOLATION: URL contains embedded credentials. "
                "Use environment variables instead."
            )
    return v
```

---

## 8. Network Security

### 8.1 Network Architecture

```
                                    +-----------------+
                                    |   Load Balancer |
                                    |   (TLS 1.3)     |
                                    +--------+--------+
                                             |
                                    +--------v--------+
                                    |  Ingress        |
                                    |  Controller     |
                                    +--------+--------+
                                             |
                              +--------------+--------------+
                              |                             |
                     +--------v--------+           +--------v--------+
                     |  GL-010 Pod     |           |  GL-010 Pod     |
                     |  (Replica 1)    |           |  (Replica 2)    |
                     +--------+--------+           +--------+--------+
                              |                             |
                              +--------------+--------------+
                                             |
                              +--------------+--------------+
                              |              |              |
                     +--------v----+ +-------v------+ +----v--------+
                     | Prometheus  | | Database     | | EPA ECMPS   |
                     | (Metrics)   | | (Postgres)   | | (External)  |
                     +-------------+ +--------------+ +-------------+
```

### 8.2 Network Policies

```yaml
# Network Policy for GL-010
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gl-010-network-policy
  namespace: greenlang-agents
spec:
  podSelector:
    matchLabels:
      app: gl-010-emissionwatch
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8010
    # Allow from Prometheus
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9010
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # Allow EPA ECMPS API
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
    # Allow database
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

### 8.3 CORS Configuration

**Current Implementation (SECURITY CONCERN):**
```python
# main.py - Current overly permissive CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # SECURITY ISSUE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Required Secure Configuration:**
```python
# Recommended secure CORS configuration
ALLOWED_ORIGINS = [
    "https://dashboard.greenlang.io",
    "https://admin.greenlang.io",
    "https://epa.gov",  # EPA reporting portal
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
```

### 8.4 Firewall Rules

| Rule | Source | Destination | Port | Protocol | Action |
|------|--------|-------------|------|----------|--------|
| HTTPS Ingress | 0.0.0.0/0 | GL-010 LB | 443 | TCP | ALLOW |
| API Access | Internal CIDR | GL-010 | 8010 | TCP | ALLOW |
| Metrics | Prometheus | GL-010 | 9010 | TCP | ALLOW |
| EPA ECMPS | GL-010 | ecmps.epa.gov | 443 | TCP | ALLOW |
| Database | GL-010 | PostgreSQL | 5432 | TCP | ALLOW |
| All Other | ANY | ANY | ANY | ANY | DENY |

---

## 9. Audit and Logging

### 9.1 Audit Log Requirements

All security-relevant events must be logged with the following information:
- Timestamp (ISO 8601 format with timezone)
- Event type
- Actor (user ID, service account, or API key hash)
- Action performed
- Resource affected
- Source IP address
- Request ID for tracing
- Result (success/failure)
- Provenance hash (for data modifications)

### 9.2 Auditable Events

| Event Category | Events | Log Level |
|----------------|--------|-----------|
| Authentication | Login, logout, failed auth, token refresh | INFO/WARN |
| Authorization | Permission denied, role changes | WARN |
| Data Access | Read emissions data, generate reports | INFO |
| Data Modification | Update calculations, modify limits | INFO |
| Configuration | Change settings, update thresholds | INFO |
| Security | Violations detected, alerts triggered | WARN |
| Administrative | User management, key rotation | INFO |
| System | Startup, shutdown, errors | INFO/ERROR |

### 9.3 Log Format

```json
{
  "timestamp": "2025-11-26T10:30:00.000Z",
  "level": "INFO",
  "logger": "gl010.audit",
  "event_type": "emissions_calculation",
  "event_id": "EVT-12345678",
  "actor": {
    "type": "service_account",
    "id": "gl-010-emissionwatch",
    "ip_address": "10.0.1.50"
  },
  "action": "calculate_nox_emissions",
  "resource": {
    "type": "cems_data",
    "id": "CEMS-2025-11-26-001"
  },
  "result": {
    "status": "success",
    "nox_ppm": 45.2,
    "compliant": true
  },
  "metadata": {
    "request_id": "REQ-87654321",
    "execution_time_ms": 125.4,
    "provenance_hash": "a1b2c3d4e5f6..."
  }
}
```

### 9.4 Log Retention and Protection

| Log Type | Retention | Storage | Protection |
|----------|-----------|---------|------------|
| Audit Logs | 7 years | Immutable storage | Write-once, encryption |
| Security Logs | 7 years | Immutable storage | Write-once, encryption |
| Application Logs | 1 year | Log aggregator | Encryption |
| Access Logs | 1 year | Log aggregator | Encryption |
| Debug Logs | 30 days | Local storage | Standard protection |

### 9.5 SHA-256 Provenance Tracking

GL-010 implements SHA-256 provenance hashing for all emissions calculations:

```python
def _calculate_provenance_hash(
    self,
    input_data: Dict[str, Any],
    result: Dict[str, Any]
) -> str:
    """Calculate SHA-256 provenance hash for audit trail."""
    provenance_data = {
        'agent_id': self.config.agent_id,
        'version': self.config.version,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'input_hash': hashlib.sha256(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest(),
        'result_summary': {
            'compliance_status': result.get('compliance_status'),
            'violations_count': len(result.get('violations', []))
        }
    }
    return hashlib.sha256(
        json.dumps(provenance_data, sort_keys=True).encode()
    ).hexdigest()
```

---

## 10. Vulnerability Management

### 10.1 Vulnerability Scanning Schedule

| Scan Type | Frequency | Scope | Tool |
|-----------|-----------|-------|------|
| Static Analysis (SAST) | Every commit | Python source code | Semgrep, Bandit |
| Dependency Scan (SCA) | Daily | requirements.txt | Safety, Snyk |
| Container Scan | Every build | Docker image | Trivy, Grype |
| Dynamic Analysis (DAST) | Weekly | Running application | OWASP ZAP |
| Infrastructure Scan | Weekly | Kubernetes manifests | Checkov, Kubesec |
| Secret Scan | Every commit | All files | Gitleaks, TruffleHog |

### 10.2 Vulnerability Severity Classification

| CVSS Score | Severity | Remediation SLA | Deployment Impact |
|------------|----------|-----------------|-------------------|
| 9.0 - 10.0 | Critical | 24 hours | Block deployment |
| 7.0 - 8.9 | High | 7 days | Block deployment if >3 |
| 4.0 - 6.9 | Medium | 30 days | Warning |
| 0.1 - 3.9 | Low | 90 days | Informational |

### 10.3 Dependency Management

**Requirements.txt Security Review:**
- All dependencies must be pinned to specific versions
- Dependencies must be reviewed before upgrade
- Known vulnerable versions must be blocked

**Current Dependencies to Monitor:**
```
fastapi>=0.100.0       # Web framework - monitor for XSS/injection
pydantic>=2.0.0        # Validation - monitor for bypass vulnerabilities
uvicorn>=0.20.0        # ASGI server - monitor for DoS vulnerabilities
httpx>=0.24.0          # HTTP client - monitor for SSRF
PyYAML>=6.0            # YAML parsing - monitor for deserialization
```

### 10.4 Security Patch Process

1. **Detection**: Automated scanning identifies vulnerability
2. **Assessment**: Security team evaluates impact and exploitability
3. **Prioritization**: Assign severity and remediation timeline
4. **Remediation**: Apply patch or workaround
5. **Testing**: Verify fix and run regression tests
6. **Deployment**: Roll out to all environments
7. **Verification**: Confirm vulnerability is resolved

---

## 11. Incident Response

### 11.1 Incident Classification

| Severity | Description | Examples | Response Time |
|----------|-------------|----------|---------------|
| SEV-1 | Critical security breach | Data exfiltration, credential compromise | Immediate (15 min) |
| SEV-2 | Significant security issue | Unauthorized access attempt, malware | 1 hour |
| SEV-3 | Moderate security concern | Policy violation, suspicious activity | 4 hours |
| SEV-4 | Minor security issue | Failed authentication, configuration drift | 24 hours |

### 11.2 Incident Response Team

| Role | Responsibility | Contact Method |
|------|---------------|----------------|
| Incident Commander | Overall response coordination | PagerDuty |
| Security Lead | Technical security response | PagerDuty |
| Engineering Lead | Application and infrastructure | PagerDuty |
| Communications Lead | Stakeholder notifications | Slack |
| Compliance Officer | Regulatory reporting | Email |

### 11.3 Incident Response Procedures

**Phase 1: Detection and Triage (0-15 minutes)**
1. Alert received through monitoring
2. Initial assessment of scope and impact
3. Classify severity level
4. Notify incident response team

**Phase 2: Containment (15-60 minutes)**
1. Isolate affected systems if necessary
2. Preserve evidence for forensics
3. Implement temporary mitigations
4. Document all actions taken

**Phase 3: Eradication (1-24 hours)**
1. Identify root cause
2. Remove threat actors/malware
3. Patch vulnerabilities
4. Verify system integrity

**Phase 4: Recovery (24-72 hours)**
1. Restore systems from clean state
2. Implement permanent fixes
3. Enhance monitoring
4. Gradual return to operations

**Phase 5: Post-Incident (1-2 weeks)**
1. Conduct post-mortem analysis
2. Document lessons learned
3. Update security controls
4. Report to regulators if required

### 11.4 Regulatory Notification Requirements

| Regulation | Notification Requirement | Timeline |
|------------|-------------------------|----------|
| EPA Part 75 | Report data quality issues | Quarterly report |
| EU GDPR | Personal data breach | 72 hours |
| SOC 2 | Security incidents | Annual audit |
| State Laws | Varies by jurisdiction | 30-60 days |

---

## 12. Container Security

### 12.1 Dockerfile Security Requirements

```dockerfile
# Secure Dockerfile for GL-010
FROM python:3.11-slim-bookworm AS base

# Security: Run as non-root user
RUN groupadd -r gl010 && useradd -r -g gl010 gl010

# Security: Install only required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Security: Set working directory
WORKDIR /app

# Security: Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Security: Copy application code
COPY --chown=gl010:gl010 . .

# Security: Remove setuid/setgid binaries
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Security: Switch to non-root user
USER gl010

# Security: Read-only root filesystem support
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
```

### 12.2 Container Security Policies

| Policy | Requirement | Enforcement |
|--------|-------------|-------------|
| Base Image | Official Python slim image only | Image policy |
| Root User | Containers must run as non-root | SecurityContext |
| Capabilities | Drop ALL, add only required | SecurityContext |
| Read-Only Root | Root filesystem must be read-only | SecurityContext |
| Resource Limits | CPU and memory limits required | ResourceRequirements |
| Image Signing | Images must be signed | Admission controller |

### 12.3 Pod Security Context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  seccompProfile:
    type: RuntimeDefault
```

---

## 13. Kubernetes Security

### 13.1 Pod Security Standards

GL-010 must comply with the "restricted" Pod Security Standard:

| Control | Value | Rationale |
|---------|-------|-----------|
| Privileged | false | No privileged containers |
| hostNetwork | false | No host network access |
| hostPID | false | No host PID namespace |
| hostIPC | false | No host IPC namespace |
| runAsNonRoot | true | Prevent root execution |
| readOnlyRootFilesystem | true | Prevent file system modifications |
| allowPrivilegeEscalation | false | Prevent privilege escalation |
| capabilities | drop: ALL | Minimize capabilities |

### 13.2 Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gl-010-quota
  namespace: greenlang-agents
spec:
  hard:
    requests.cpu: "4"
    requests.memory: "8Gi"
    limits.cpu: "8"
    limits.memory: "16Gi"
    pods: "10"
    secrets: "20"
    configmaps: "20"
```

### 13.3 Admission Controllers

| Controller | Purpose | Configuration |
|------------|---------|---------------|
| PodSecurity | Enforce pod security standards | restricted |
| ImagePolicyWebhook | Verify image signatures | Required |
| ResourceQuota | Enforce resource limits | Enabled |
| LimitRanger | Default resource limits | Enabled |
| NetworkPolicy | Network segmentation | Required |

---

## 14. API Security

### 14.1 API Security Controls

| Control | Implementation | Status |
|---------|----------------|--------|
| Authentication | API Key / JWT | Implemented |
| Authorization | RBAC | Implemented |
| Rate Limiting | 100 req/min per key | Recommended |
| Input Validation | Pydantic models | Implemented |
| Output Encoding | JSON with escaping | Implemented |
| CORS | Restricted origins | NEEDS UPDATE |
| HTTPS | TLS 1.3 required | Implemented |

### 14.2 Input Validation

GL-010 uses Pydantic models with field constraints for input validation:

```python
class CEMSDataModel(BaseModel):
    """CEMS data input model with validation."""
    nox_ppm: float = Field(default=0.0, ge=0, description="NOx concentration in ppm")
    sox_ppm: float = Field(default=0.0, ge=0, description="SOx concentration in ppm")
    co2_percent: float = Field(default=0.0, ge=0, le=25, description="CO2 concentration")
    o2_percent: float = Field(default=3.0, ge=0, le=21, description="O2 concentration")
    pm_mg_m3: float = Field(default=0.0, ge=0, description="PM concentration")
    opacity_percent: float = Field(default=0.0, ge=0, le=100, description="Opacity")
    flow_rate_dscfm: float = Field(default=10000, ge=0, description="Flow rate")
    temperature_f: float = Field(default=300, description="Temperature")
    quality_code: str = Field(default="valid", description="Data quality code")
```

### 14.3 API Rate Limiting (Recommended)

```python
# Recommended rate limiting configuration
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Rate limits by endpoint
RATE_LIMITS = {
    "/api/v1/monitor": "100/minute",
    "/api/v1/report": "10/minute",
    "/api/v1/alert": "50/minute",
    "/api/v1/analyze": "20/minute",
    "/api/v1/predict": "20/minute",
    "/api/v1/audit": "10/minute",
    "/api/v1/benchmark": "20/minute",
    "/api/v1/validate": "50/minute",
}
```

---

## 15. Cryptographic Standards

### 15.1 Approved Algorithms

| Use Case | Algorithm | Key Size | Notes |
|----------|-----------|----------|-------|
| Symmetric Encryption | AES-256-GCM | 256 bits | Data at rest |
| Asymmetric Encryption | RSA-2048, ECDSA P-256 | 2048/256 bits | Key exchange |
| Hashing (Data) | SHA-256 | 256 bits | Provenance tracking |
| Hashing (Passwords) | Argon2id | N/A | If applicable |
| TLS | TLS 1.3 | N/A | All network traffic |
| Digital Signatures | RS256, ES256 | N/A | JWT, certificates |
| Key Derivation | PBKDF2-HMAC-SHA256 | N/A | Key stretching |

### 15.2 Prohibited Algorithms

The following algorithms are PROHIBITED:
- MD5 (all uses)
- SHA-1 (all uses)
- DES, 3DES
- RC4
- TLS 1.0, TLS 1.1
- SSL (all versions)
- RSA with key < 2048 bits

### 15.3 Random Number Generation

```python
import secrets

# REQUIRED: Use cryptographically secure random
api_key = secrets.token_urlsafe(32)
nonce = secrets.token_bytes(16)

# PROHIBITED: Do not use for security purposes
# import random
# random.randint(...)  # NOT cryptographically secure
```

---

## 16. Security Testing

### 16.1 Security Testing Schedule

| Test Type | Frequency | Scope | Responsible |
|-----------|-----------|-------|-------------|
| Unit Security Tests | Every commit | Security functions | Developers |
| SAST Scan | Every commit | Source code | CI/CD |
| SCA Scan | Daily | Dependencies | CI/CD |
| Secret Scan | Every commit | All files | CI/CD |
| Container Scan | Every build | Docker images | CI/CD |
| DAST Scan | Weekly | Running application | Security Team |
| Penetration Test | Annually | Full application | External vendor |

### 16.2 Security Test Requirements

**Pre-deployment Checklist:**
- [ ] All SAST findings resolved or accepted
- [ ] No critical or high SCA vulnerabilities
- [ ] No secrets detected in codebase
- [ ] Container scan passed
- [ ] Input validation tests passed
- [ ] Authentication tests passed
- [ ] Authorization tests passed

### 16.3 Security Acceptance Criteria

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| Critical vulnerabilities | 0 | Block deployment |
| High vulnerabilities | 0 | Block deployment |
| Medium vulnerabilities | <5 | Warning, review |
| Hardcoded secrets | 0 | Block deployment |
| Container CVEs (Critical) | 0 | Block deployment |
| Kubernetes misconfigurations | 0 Critical | Block deployment |

---

## 17. Compliance Verification

### 17.1 Compliance Checklist

| Requirement | Control | Verification Method | Frequency |
|-------------|---------|---------------------|-----------|
| EPA Part 75 Data Quality | Data validation | Automated checks | Continuous |
| EPA CEMS Requirements | Calibration validation | RATA audits | Annual |
| SOC 2 Security | Access controls | Audit | Annual |
| SOC 2 Availability | SLA monitoring | Automated | Continuous |
| Data Encryption | Encryption verification | Audit | Quarterly |
| Key Management | Key rotation | Automated | 90 days |
| Access Review | RBAC audit | Manual review | Quarterly |
| Vulnerability Management | Scan results | Automated | Continuous |

### 17.2 Audit Evidence Collection

GL-010 automatically collects the following audit evidence:
- SHA-256 provenance hashes for all calculations
- Complete audit logs with event correlation
- Access logs with user identification
- Configuration change history
- Security scan results
- Compliance check results

### 17.3 Third-Party Audit Support

For SOC 2 Type II audits:
- Provide read-only access to audit logs
- Generate compliance reports on demand
- Export provenance hash chains
- Demonstrate access control effectiveness

---

## 18. Policy Exceptions

### 18.1 Exception Process

1. **Request**: Submit exception request with business justification
2. **Risk Assessment**: Security team evaluates risk
3. **Approval**: CISO or delegate approves/denies
4. **Documentation**: Exception documented with expiry date
5. **Monitoring**: Enhanced monitoring during exception period
6. **Review**: Exception reviewed before expiry

### 18.2 Exception Requirements

All exceptions must include:
- Business justification
- Risk assessment
- Compensating controls
- Expiration date (maximum 90 days)
- Approval signatures
- Monitoring plan

### 18.3 Prohibited Exceptions

The following cannot be excepted:
- Hardcoded credentials in source code
- TLS version below 1.3 for production
- Missing encryption for REGULATORY SENSITIVE data
- Disabled audit logging
- Root container execution in production

---

## 19. Document Control

### 19.1 Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | Security Team | Initial release |

### 19.2 Review and Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | GL-SecScan | 2025-11-26 | Digital |
| Security Review | | | |
| Compliance Review | | | |
| CISO Approval | | | |

### 19.3 Distribution

This document is classified as INTERNAL and is distributed to:
- GreenLang Security Team
- GL-010 Development Team
- Compliance Officers
- System Administrators

### 19.4 Related Documents

- GL-010 SECURITY_SCAN_REPORT.md
- GL-010 security_validator.py
- GreenLang Master Security Policy
- GreenLang Incident Response Plan
- GreenLang Data Classification Standard

---

## Appendix A: Security Contacts

| Role | Contact | Escalation Path |
|------|---------|-----------------|
| Security Team | security@greenlang.io | PagerDuty |
| Compliance | compliance@greenlang.io | Email |
| Privacy | privacy@greenlang.io | Email |
| Emergency | security-emergency@greenlang.io | Phone |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| CEMS | Continuous Emissions Monitoring System |
| ECMPS | Emissions Collection and Monitoring Plan System |
| RATA | Relative Accuracy Test Audit |
| BAT-AEL | Best Available Techniques - Associated Emission Levels |
| IED | Industrial Emissions Directive |
| MEE | Ministry of Ecology and Environment (China) |
| CDX | EPA Central Data Exchange |
| mTLS | Mutual Transport Layer Security |
| KMS | Key Management Service |
| RBAC | Role-Based Access Control |

## Appendix C: Regulatory References

- EPA 40 CFR Part 60 - Standards of Performance for New Stationary Sources
- EPA 40 CFR Part 75 - Continuous Emissions Monitoring
- EU Directive 2010/75/EU - Industrial Emissions Directive
- China GB 13223-2011 - Emission Standard of Air Pollutants for Thermal Power Plants
- AICPA SOC 2 - Trust Services Criteria
- NIST 800-53 - Security and Privacy Controls

---

**END OF DOCUMENT**

*This policy is effective immediately upon approval and supersedes all previous versions.*
