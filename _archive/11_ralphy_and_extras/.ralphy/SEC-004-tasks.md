# SEC-004: TLS 1.3 Configuration - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-001, INFRA-002, INFRA-003, INFRA-006
**Result:** 37 new files + 1 modified, ~12,500 lines

---

## Phase 1: Terraform TLS Policies Module (P0)

### 1.1 Module Main Configuration
- [x] Create `deployment/terraform/modules/tls-policies/main.tf`:
  - AWS ALB SSL policy resource
  - AWS NLB SSL policy configuration
  - CloudFront TLS policy (if applicable)
  - API Gateway security policy
  - Local values for cipher suites

### 1.2 Module Variables
- [x] Create `deployment/terraform/modules/tls-policies/variables.tf`:
  - environment, project_name
  - min_tls_version (default "TLSv1.2")
  - strict_tls_13 (default false for compatibility)
  - allowed_ciphers list
  - hsts_enabled, hsts_max_age
  - tags

### 1.3 Module Outputs
- [x] Create `deployment/terraform/modules/tls-policies/outputs.tf`:
  - alb_ssl_policy_name
  - nlb_ssl_policy_name
  - cloudfront_min_protocol_version
  - cipher_suite_string
  - tls_security_policy_arn

### 1.4 Module Documentation
- [x] Create `deployment/terraform/modules/tls-policies/README.md`:
  - Usage examples
  - SSL policy comparison table
  - Cipher suite explanation
  - Integration with ALB/NLB modules

---

## Phase 2: Kubernetes Ingress TLS Hardening (P0)

### 2.1 NGINX Ingress ConfigMap
- [x] Create `deployment/kubernetes/tls/nginx-tls-configmap.yaml`:
  - ssl-protocols: "TLSv1.2 TLSv1.3"
  - ssl-ciphers: modern cipher string
  - ssl-prefer-server-ciphers: "true"
  - ssl-session-cache: shared:SSL:50m
  - ssl-session-timeout: 1d
  - hsts: "max-age=31536000; includeSubDomains; preload"
  - hsts-include-subdomains: "true"
  - hsts-preload: "true"

### 2.2 Kong Gateway TLS Update
- [x] Update `deployment/helm/kong-gateway/values.yaml`:
  - Add nginx_proxy_ssl_protocols
  - Add nginx_proxy_ssl_ciphers
  - Add ssl_prefer_server_ciphers
  - Add ssl_session_timeout
  - Add ssl_session_cache

### 2.3 TLS NetworkPolicy
- [x] Create `deployment/kubernetes/tls/tls-network-policy.yaml`:
  - Allow only HTTPS (443) ingress
  - Block HTTP (80) except for redirect
  - Allow internal mTLS traffic

### 2.4 Ingress Annotation Template
- [x] Create `deployment/kubernetes/tls/ingress-tls-annotations.yaml`:
  - Common TLS annotations for all ingresses
  - cert-manager annotations
  - HSTS annotations
  - SSL redirect annotations

---

## Phase 3: Database TLS Enforcement (P0)

### 3.1 PostgreSQL TLS Configuration
- [x] Create `greenlang/infrastructure/tls_service/database_tls.py`:
  - `get_postgres_ssl_context()` - Returns SSLContext for PostgreSQL
  - `get_postgres_connection_params()` - Returns SSL params dict
  - `verify_postgres_tls_connection()` - Verifies TLS is working
  - CA bundle path resolution for AWS RDS

### 3.2 Redis TLS Configuration
- [x] Create `greenlang/infrastructure/tls_service/redis_tls.py`:
  - `get_redis_ssl_context()` - Returns SSLContext for Redis
  - `create_redis_tls_connection()` - Creates TLS-enabled Redis client
  - `verify_redis_tls_connection()` - Verifies TLS is working
  - ElastiCache certificate handling

### 3.3 Connection String Updates
- [x] Create `deployment/kubernetes/database/secrets/tls-connection-template.yaml`:
  - Add `?sslmode=verify-full` to PostgreSQL connection strings
  - Add `ssl=true&ssl_ca_certs=/path/to/ca` to Redis URLs
  - Update ExternalSecret templates

---

## Phase 4: Application TLS Module (P1)

### 4.1 Package Init
- [x] Create `greenlang/infrastructure/tls_service/__init__.py`:
  - Public API exports
  - TLSConfig dataclass
  - TLSVersion enum
  - CipherSuite constants

### 4.2 SSL Context Factory
- [x] Create `greenlang/infrastructure/tls_service/ssl_context.py`:
  - `create_ssl_context(purpose, min_version, verify)` - Factory function
  - `create_client_ssl_context()` - For outbound connections
  - `create_server_ssl_context()` - For inbound connections
  - Purpose-specific contexts (HTTP, DB, gRPC)

### 4.3 CA Bundle Management
- [x] Create `greenlang/infrastructure/tls_service/ca_bundle.py`:
  - `get_ca_bundle_path()` - Returns appropriate CA bundle
  - `get_aws_rds_ca_bundle()` - AWS RDS CA certificates
  - `get_system_ca_bundle()` - System CA certificates
  - `refresh_ca_bundle()` - Download/update CA bundle

### 4.4 HTTP Client Integration
- [x] Create `greenlang/infrastructure/tls_service/http_tls.py`:
  - `create_httpx_client(verify_tls, min_version)` - TLS-enabled httpx client
  - `create_aiohttp_session(verify_tls)` - TLS-enabled aiohttp session
  - Certificate pinning support
  - TLS session resumption

### 4.5 TLS Metrics
- [x] Create `greenlang/infrastructure/tls_service/tls_metrics.py`:
  - `gl_tls_connections_total` Counter (protocol, cipher, status)
  - `gl_tls_handshake_duration_seconds` Histogram
  - `gl_tls_certificate_expiry_seconds` Gauge
  - `gl_tls_errors_total` Counter (error_type)
  - Lazy initialization pattern

### 4.6 TLS Utilities
- [x] Create `greenlang/infrastructure/tls_service/utils.py`:
  - `check_tls_version(host, port)` - Test TLS version
  - `get_certificate_info(host, port)` - Get cert details
  - `verify_certificate_chain(cert)` - Verify cert chain
  - `days_until_expiry(cert)` - Calculate expiry

---

## Phase 5: Service Mesh mTLS (P1)

### 5.1 PeerAuthentication
- [x] Create `deployment/kubernetes/tls/mesh/peer-authentication.yaml`:
  - Namespace-wide STRICT mTLS policy
  - Workload-specific exceptions (if needed)
  - Port-level mTLS configuration

### 5.2 DestinationRule
- [x] Create `deployment/kubernetes/tls/mesh/destination-rules.yaml`:
  - TLS mode: ISTIO_MUTUAL
  - Min TLS version: TLSv1_3
  - Subject alt name validation

### 5.3 mTLS Exceptions
- [x] Create `deployment/kubernetes/tls/mesh/mtls-exceptions.yaml`:
  - Health check endpoints
  - External service integrations
  - Legacy service compatibility

### 5.4 Certificate Rotation Config
- [x] Create `deployment/kubernetes/tls/mesh/cert-rotation.yaml`:
  - Workload certificate TTL: 24h
  - Root CA TTL: 10 years
  - Intermediate CA TTL: 1 year

---

## Phase 6: Certificate Management (P1)

### 6.1 cert-manager Enhanced Configuration
- [x] Create `deployment/kubernetes/tls/cert-manager/cluster-issuers.yaml`:
  - Let's Encrypt production issuer
  - Let's Encrypt staging issuer
  - Self-signed issuer (for internal CA)
  - DNS-01 challenge solver config

### 6.2 Certificate Templates
- [x] Create `deployment/kubernetes/tls/cert-manager/certificate-templates.yaml`:
  - External wildcard certificate
  - Internal service certificates
  - mTLS client certificates
  - Certificate renewal settings

### 6.3 AWS ACM Terraform
- [x] Create `deployment/terraform/modules/tls-policies/acm.tf`:
  - ACM certificate resource
  - DNS validation records
  - Certificate auto-renewal
  - Cross-region replication (if needed)

### 6.4 Internal CA Setup
- [x] Create `deployment/kubernetes/tls/internal-ca/internal-ca-setup.yaml`:
  - Root CA certificate
  - Intermediate CA certificate
  - CA secret encryption
  - Rotation procedures

---

## Phase 7: Monitoring & Alerting (P2)

### 7.1 TLS Metrics Exporter
- [x] Create `greenlang/infrastructure/tls_service/exporter.py`:
  - Prometheus metrics endpoint
  - Certificate expiry scanner
  - TLS connection statistics
  - Protocol version tracking

### 7.2 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/tls-security.json`:
  - Certificate expiry countdown (all domains)
  - TLS protocol version distribution
  - Cipher suite usage heatmap
  - TLS handshake latency
  - TLS error rate by type
  - mTLS connection status
  - SSL Labs grade history
  - Top TLS errors table

### 7.3 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/tls-security-alerts.yaml`:
  - CertificateExpiringWarning (< 14 days)
  - CertificateExpiringCritical (< 7 days)
  - CertificateExpired
  - TLS10or11ConnectionAttempt
  - HighTLSErrorRate
  - CertificateRenewalFailed
  - mTLSEnforcementDisabled
  - WeakCipherUsed

---

## Phase 8: Testing (P2)

### 8.1 Unit Tests
- [x] Create `tests/unit/tls_service/__init__.py`
- [x] Create `tests/unit/tls_service/test_ssl_context.py` - 25+ tests:
  - SSL context creation, TLS version enforcement, cipher validation
- [x] Create `tests/unit/tls_service/test_ca_bundle.py` - 15+ tests:
  - CA bundle loading, path resolution, refresh
- [x] Create `tests/unit/tls_service/test_database_tls.py` - 20+ tests:
  - PostgreSQL SSL, Redis TLS, connection verification
- [x] Create `tests/unit/tls_service/test_exporter.py` - 25+ tests:
  - Certificate scanner, metrics exporter, Prometheus integration

### 8.2 Integration Tests
- [x] Create `tests/integration/tls_service/__init__.py`
- [x] Create `tests/integration/tls_service/test_tls_connections.py` - 15+ tests:
  - Real TLS connections to test endpoints
- [x] Create `tests/integration/tls_service/test_mtls.py` - 10+ tests:
  - mTLS handshakes, certificate validation

### 8.3 Compliance Tests
- [x] Create `tests/compliance/test_tls_compliance.py` - 15+ tests:
  - TLS 1.3 enforcement, cipher suite compliance
  - No weak protocols, HSTS verification
  - Certificate chain validation

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Terraform TLS Policies | 4/4 | P0 | COMPLETE |
| Phase 2: Kubernetes Ingress TLS | 4/4 | P0 | COMPLETE |
| Phase 3: Database TLS | 3/3 | P0 | COMPLETE |
| Phase 4: Application TLS Module | 6/6 | P1 | COMPLETE |
| Phase 5: Service Mesh mTLS | 4/4 | P1 | COMPLETE |
| Phase 6: Certificate Management | 4/4 | P1 | COMPLETE |
| Phase 7: Monitoring & Alerting | 3/3 | P2 | COMPLETE |
| Phase 8: Testing | 9/9 | P2 | COMPLETE |
| **TOTAL** | **37/37** | - | **COMPLETE** |
