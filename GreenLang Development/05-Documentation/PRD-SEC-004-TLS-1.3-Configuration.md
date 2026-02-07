# PRD-SEC-004: TLS 1.3 Configuration for All Services

**Status:** APPROVED
**Version:** 1.0
**Created:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-001 (EKS), INFRA-002 (PostgreSQL), INFRA-003 (Redis), INFRA-006 (Kong)

---

## 1. Overview

### 1.1 Purpose
Configure Transport Layer Security (TLS) 1.3 across all GreenLang Climate OS services to ensure encrypted communications meet current security standards. TLS 1.3 provides improved security, reduced latency (0-RTT), and simplified cipher suite negotiation.

### 1.2 Scope
- **In Scope:**
  - AWS infrastructure TLS configuration (ALB, NLB, CloudFront, API Gateway)
  - Kubernetes Ingress TLS (NGINX, Kong)
  - Service-to-service mTLS (internal cluster)
  - Database connections (PostgreSQL, Redis)
  - Python application TLS helpers
  - Certificate management automation
  - TLS monitoring and alerting
- **Out of Scope:**
  - Client-side certificate management
  - Hardware Security Modules (HSM) integration (future SEC-005)
  - Third-party API TLS validation

### 1.3 Success Criteria
- All external-facing endpoints enforce TLS 1.3 (TLS 1.2 fallback allowed)
- All internal service-to-service communication uses mTLS
- No TLS 1.0/1.1 connections permitted anywhere
- TLS certificate rotation automated with zero downtime
- Monitoring alerts for certificate expiry, TLS errors, protocol downgrades

---

## 2. Technical Requirements

### TR-001: AWS Infrastructure TLS Policies
**Priority:** P0
**Description:** Configure AWS-managed TLS termination points to enforce TLS 1.3.

**Requirements:**
1. Create Terraform module for centralized TLS policy management
2. Application Load Balancer (ALB):
   - SSL policy: `ELBSecurityPolicy-TLS13-1-2-2021-06` (TLS 1.3 + TLS 1.2 fallback)
   - Cipher suites: TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256
3. Network Load Balancer (NLB):
   - SSL policy: `ELBSecurityPolicy-TLS13-1-2-2021-06`
   - TCP passthrough for services requiring end-to-end encryption
4. CloudFront (if used):
   - Minimum protocol version: TLSv1.2_2021
   - Origin protocol policy: HTTPS only
5. API Gateway (AWS):
   - Security policy: TLS_1_2
   - Custom domain with ACM certificate

**Acceptance Criteria:**
- [ ] Terraform module `tls-policies` created
- [ ] All ALBs use TLS 1.3 policy
- [ ] All NLBs use TLS 1.3 policy
- [ ] SSL Labs score: A+ for all public endpoints

### TR-002: Kubernetes Ingress TLS Configuration
**Priority:** P0
**Description:** Configure NGINX Ingress Controller and Kong Gateway for TLS 1.3.

**Requirements:**
1. NGINX Ingress Controller:
   - `ssl-protocols: "TLSv1.3"` (strict) or `"TLSv1.2 TLSv1.3"` (compatible)
   - `ssl-ciphers`: Modern cipher suite string
   - HSTS: `max-age=31536000; includeSubDomains; preload`
   - OCSP stapling enabled
2. Kong Gateway:
   - `ssl_protocols`: TLSv1.2, TLSv1.3
   - `ssl_prefer_server_ciphers`: on
   - `ssl_session_timeout`: 1d
   - `ssl_session_cache`: shared:SSL:50m
3. ConfigMap for centralized TLS settings across all ingresses
4. NetworkPolicy allowing only HTTPS traffic on ingress

**Acceptance Criteria:**
- [ ] NGINX Ingress ConfigMap updated with TLS 1.3 settings
- [ ] Kong Gateway TLS configuration hardened
- [ ] All ingress resources inherit TLS settings
- [ ] HTTP (port 80) redirects to HTTPS (port 443)

### TR-003: Service Mesh mTLS (Internal Traffic)
**Priority:** P1
**Description:** Enable mutual TLS for all internal service-to-service communication.

**Requirements:**
1. Istio service mesh (or Linkerd alternative):
   - PeerAuthentication: STRICT mTLS
   - DestinationRule: TLS mode ISTIO_MUTUAL
   - Minimum protocol: TLS 1.3 for mesh traffic
2. Certificate rotation:
   - Automatic certificate rotation every 24 hours
   - Root CA rotation every 365 days
3. Fallback mode for legacy services during migration:
   - PERMISSIVE mode with timeline to STRICT
4. mTLS exceptions whitelist (external services, health checks)

**Acceptance Criteria:**
- [ ] PeerAuthentication resource deployed
- [ ] DestinationRule for all services
- [ ] mTLS traffic visible in service mesh dashboard
- [ ] Certificates rotate automatically

### TR-004: Database TLS Configuration
**Priority:** P0
**Description:** Enforce TLS 1.3 for all database connections.

**Requirements:**
1. PostgreSQL (Aurora):
   - `rds.force_ssl = 1` (already configured)
   - SSL mode: `verify-full` in connection strings
   - CA bundle from AWS RDS certificates
   - Connection pooler (PgBouncer) TLS passthrough
2. Redis (ElastiCache):
   - `transit_encryption_enabled = true` (already configured)
   - AUTH token + TLS required
   - stunnel proxy for TLS termination (if needed for legacy clients)
3. Python connection helpers:
   - `ssl_context` with minimum TLS 1.3
   - Certificate verification enabled
   - CA bundle path configurable

**Acceptance Criteria:**
- [ ] PostgreSQL connections use verify-full SSL mode
- [ ] Redis connections use TLS with AUTH
- [ ] Python helpers enforce TLS 1.3
- [ ] No plaintext database connections allowed

### TR-005: Application TLS Configuration Module
**Priority:** P1
**Description:** Python module for standardized TLS configuration across all GreenLang applications.

**Requirements:**
1. `greenlang/infrastructure/tls_service/` module:
   - `TLSConfig` dataclass with environment-aware defaults
   - `create_ssl_context(purpose, verify=True)` - returns configured SSLContext
   - `get_ca_bundle_path()` - returns appropriate CA bundle
   - `get_client_cert_path()` - returns mTLS client certificate
2. Integration with:
   - `httpx` / `aiohttp` for HTTP clients
   - `psycopg` for PostgreSQL
   - `redis-py` for Redis
   - `grpcio` for gRPC (if used)
3. Certificate pinning support (optional, for high-security endpoints)
4. TLS session resumption for performance

**Acceptance Criteria:**
- [ ] TLS module created with SSLContext factory
- [ ] All HTTP clients use TLS module
- [ ] Database connections use TLS module
- [ ] Unit tests verify TLS 1.3 enforcement

### TR-006: Certificate Management Automation
**Priority:** P1
**Description:** Automate certificate lifecycle with cert-manager and AWS ACM.

**Requirements:**
1. cert-manager configuration:
   - ClusterIssuer for Let's Encrypt (production + staging)
   - Certificate resources with 90-day validity
   - Renewal at 30 days before expiry
   - DNS-01 challenge for wildcard certificates
2. AWS ACM integration:
   - ACM certificates for ALB/NLB/CloudFront
   - Auto-renewal (managed by AWS)
   - DNS validation via Route53
3. Internal CA for mTLS:
   - Self-signed root CA in Kubernetes secrets
   - Intermediate CA for service certificates
   - Short-lived certificates (24-hour validity)
4. Certificate storage:
   - Kubernetes Secrets (encrypted at rest via KMS)
   - External Secrets Operator for AWS Secrets Manager sync

**Acceptance Criteria:**
- [ ] cert-manager ClusterIssuer configured
- [ ] All certificates auto-renew
- [ ] No manual certificate management required
- [ ] Certificate events logged to audit trail

### TR-007: TLS Monitoring and Alerting
**Priority:** P1
**Description:** Comprehensive monitoring for TLS health and compliance.

**Requirements:**
1. Prometheus metrics:
   - `gl_tls_connections_total{protocol, cipher, status}`
   - `gl_tls_handshake_duration_seconds{protocol}`
   - `gl_tls_certificate_expiry_seconds{domain}`
   - `gl_tls_errors_total{error_type}`
   - `gl_tls_protocol_downgrades_total`
2. Grafana dashboard:
   - Certificate expiry countdown
   - TLS protocol distribution
   - Cipher suite usage
   - Handshake latency
   - TLS error rate
3. Prometheus alerts:
   - Certificate expiring < 14 days (warning)
   - Certificate expiring < 7 days (critical)
   - TLS 1.0/1.1 connection attempt (critical)
   - High TLS error rate (warning)
   - Certificate renewal failed (critical)

**Acceptance Criteria:**
- [ ] All TLS metrics exported
- [ ] Grafana dashboard with 10+ panels
- [ ] 8+ alert rules configured
- [ ] PagerDuty integration for critical alerts

---

## 3. Architecture

### 3.1 TLS Termination Points

```
                                    Internet
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │        AWS CloudFront (CDN)           │
                    │    TLS 1.3 (ELBSecurityPolicy-TLS13)  │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │      Application Load Balancer        │
                    │    TLS 1.3 (ELBSecurityPolicy-TLS13)  │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │       NGINX Ingress Controller        │
                    │          TLS 1.3 (strict)             │
                    └───────────────────┬───────────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    ┌──────────┐  ┌──────────┐  ┌──────────┐
                    │  Kong    │  │  API     │  │  Agent   │
                    │ Gateway  │  │ Service  │  │ Service  │
                    │  mTLS    │  │  mTLS    │  │  mTLS    │
                    └────┬─────┘  └────┬─────┘  └────┬─────┘
                         │             │             │
              ┌──────────┴─────────────┴─────────────┴──────────┐
              │              Service Mesh (mTLS)                 │
              └──────────────────────┬───────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐    ┌──────────┐    ┌──────────┐
              │PostgreSQL│    │  Redis   │    │   S3     │
              │TLS 1.3   │    │TLS 1.3   │    │TLS 1.2   │
              └──────────┘    └──────────┘    └──────────┘
```

### 3.2 Certificate Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │       External Certificates          │
                    │    (Let's Encrypt / AWS ACM)         │
                    │    Validity: 90 days (LE) / 13mo    │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │        Ingress / ALB / NLB           │
                    │     (Public-facing endpoints)        │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │       Internal Root CA               │
                    │    (Self-managed, K8s Secret)        │
                    │    Validity: 10 years                │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │      Intermediate CA                 │
                    │    Validity: 1 year, auto-rotate     │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
        ┌──────────┐           ┌──────────┐           ┌──────────┐
        │Service A │           │Service B │           │Service C │
        │  Cert    │           │  Cert    │           │  Cert    │
        │ 24-hour  │           │ 24-hour  │           │ 24-hour  │
        └──────────┘           └──────────┘           └──────────┘
```

---

## 4. Implementation Phases

### Phase 1: Terraform TLS Policies Module (P0)
- Create `deployment/terraform/modules/tls-policies/`
- ALB SSL policy configuration
- NLB SSL policy configuration
- Output policy ARNs for other modules

### Phase 2: Kubernetes Ingress TLS Hardening (P0)
- Update NGINX Ingress ConfigMap
- Update Kong Gateway TLS settings
- Create TLS ConfigMap for all ingresses
- Verify HTTP to HTTPS redirect

### Phase 3: Database TLS Enforcement (P0)
- Update PostgreSQL connection strings
- Update Redis connection configuration
- Create Python TLS helpers for database connections

### Phase 4: Application TLS Module (P1)
- Create `greenlang/infrastructure/tls_service/` module
- SSLContext factory
- CA bundle management
- Integration tests

### Phase 5: Service Mesh mTLS (P1)
- Deploy PeerAuthentication resources
- Deploy DestinationRule resources
- Configure certificate rotation
- Test service-to-service encryption

### Phase 6: Certificate Management (P1)
- Enhance cert-manager configuration
- AWS ACM Terraform resources
- Internal CA setup
- Certificate lifecycle automation

### Phase 7: Monitoring & Alerting (P2)
- TLS metrics exporter
- Grafana dashboard
- Prometheus alert rules
- Runbook documentation

### Phase 8: Testing (P2)
- Unit tests for TLS module
- Integration tests for connections
- SSL Labs scan automation
- Compliance verification tests

---

## 5. Security Considerations

### 5.1 Cipher Suite Configuration
```
# TLS 1.3 Cipher Suites (in order of preference)
TLS_AES_256_GCM_SHA384
TLS_CHACHA20_POLY1305_SHA256
TLS_AES_128_GCM_SHA256

# TLS 1.2 Fallback Cipher Suites (legacy clients)
ECDHE-ECDSA-AES256-GCM-SHA384
ECDHE-RSA-AES256-GCM-SHA384
ECDHE-ECDSA-AES128-GCM-SHA256
ECDHE-RSA-AES128-GCM-SHA256
```

### 5.2 Disabled Protocols and Ciphers
- TLS 1.0, TLS 1.1: Disabled (vulnerable)
- SSLv2, SSLv3: Disabled (vulnerable)
- RC4, DES, 3DES: Disabled (weak)
- MD5, SHA1 (for signatures): Disabled (weak)
- Export ciphers: Disabled (weak)
- NULL ciphers: Disabled (no encryption)

### 5.3 Key Requirements
- RSA keys: Minimum 2048 bits (4096 recommended for long-term)
- ECDSA keys: P-256 or P-384 curves
- Diffie-Hellman: 2048+ bits (if used)

---

## 6. Compliance Mapping

| Requirement | SOC 2 | ISO 27001 | PCI DSS | GDPR |
|-------------|-------|-----------|---------|------|
| TLS 1.3 encryption | CC6.7 | A.10.1.1 | 4.1 | Art. 32 |
| Certificate management | CC6.1 | A.10.1.2 | 4.1 | Art. 32 |
| No weak ciphers | CC6.7 | A.10.1.1 | 4.1 | Art. 32 |
| Encryption monitoring | CC7.2 | A.12.4.1 | 10.6 | Art. 32 |

---

## 7. Deliverables Summary

| Component | Files | Priority |
|-----------|-------|----------|
| Terraform TLS Policies Module | 5 | P0 |
| Kubernetes Ingress Config | 4 | P0 |
| Database TLS Config | 3 | P0 |
| Application TLS Module | 6 | P1 |
| Service Mesh mTLS | 4 | P1 |
| Certificate Management | 4 | P1 |
| Monitoring & Alerting | 3 | P2 |
| Testing | 6 | P2 |
| **TOTAL** | **~35** | - |

---

## 8. Appendix

### A. SSL Labs Grade Requirements
- Overall Grade: A+
- Protocol Support: 100%
- Key Exchange: 100%
- Cipher Strength: 100%

### B. Testing Commands
```bash
# Test TLS version support
openssl s_client -connect api.greenlang.io:443 -tls1_3

# Test cipher suites
nmap --script ssl-enum-ciphers -p 443 api.greenlang.io

# SSL Labs API scan
curl "https://api.ssllabs.com/api/v3/analyze?host=api.greenlang.io"
```

### C. Environment-Specific Configuration
| Setting | Dev | Staging | Prod |
|---------|-----|---------|------|
| Min TLS Version | 1.2 | 1.2 | 1.3 |
| HSTS | Off | On | On + Preload |
| Certificate Issuer | staging | staging | prod |
| mTLS Mode | Permissive | Strict | Strict |
