# PRD-SEC-006: Deploy Secrets Management (HashiCorp Vault)

**Status:** APPROVED
**Version:** 1.0
**Created:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-001 (EKS), SEC-001 (JWT Auth), SEC-002 (RBAC), SEC-003 (Encryption)

---

## 1. Overview

### 1.1 Purpose
Complete the deployment of HashiCorp Vault as the centralized secrets management solution for GreenLang Climate OS. This PRD covers completing the existing infrastructure, adding API endpoints, monitoring, testing, and full integration with the application stack.

### 1.2 Current State
Significant infrastructure already exists:
- **Terraform Module**: `deployment/terraform/modules/vault/` (633 lines) - Helm deployment with HA, KMS auto-unseal, K8s auth
- **Python Client**: `greenlang/execution/infrastructure/secrets/vault_client.py` (1063 lines) - Full VaultClient
- **Rotation Manager**: `greenlang/execution/infrastructure/secrets/secrets_rotation.py` (893 lines) - Automatic rotation
- **K8s Manifests**: vault-deployment.yaml, vault-config.yaml, vault-agent-injector.yaml
- **External Secrets**: ClusterSecretStore, ExternalSecrets for database credentials

### 1.3 Scope
- **In Scope:**
  - Unified secrets_service module wrapping existing vault_client
  - REST API for secrets management (/api/v1/secrets/*)
  - Prometheus metrics and Grafana dashboard
  - Alert rules for Vault health and operations
  - Complete External Secrets Operator integration
  - Vault Agent Injector configuration for all services
  - Comprehensive test suite
  - Integration with auth_service
- **Out of Scope:**
  - Vault Enterprise features (namespaces, MFA)
  - Hardware Security Module (HSM) integration
  - Multi-region Vault replication

### 1.4 Success Criteria
- All application secrets retrieved from Vault (no hardcoded secrets)
- API endpoints for secrets CRUD with RBAC protection
- Vault health monitored with <5 minute alerting
- Secrets rotation automated with zero downtime
- 100% test coverage for secrets operations
- External Secrets sync < 30 seconds

---

## 2. Technical Requirements

### TR-001: Unified Secrets Service
**Priority:** P0
**Description:** Create a centralized secrets service that wraps the existing VaultClient.

**Requirements:**
1. `greenlang/infrastructure/secrets_service/` module:
   - `SecretsServiceConfig` dataclass with environment-aware defaults
   - `SecretsService` class orchestrating VaultClient and rotation
   - Factory methods for common secret types (database, API keys, certificates)
   - Integration with auth context for tenant isolation
2. Secret types:
   - Database credentials (PostgreSQL, Redis)
   - API keys (internal, external partners)
   - TLS certificates (internal CA, mTLS)
   - Encryption keys (Transit secrets engine)
   - Service-to-service tokens
3. Multi-tenancy:
   - Tenant-scoped secret paths (secret/tenants/{tenant_id}/*)
   - Shared secrets for platform services
   - RLS-aware secret access

**Acceptance Criteria:**
- [ ] SecretsService wraps VaultClient with tenant context
- [ ] All secret types accessible via unified interface
- [ ] Multi-tenant isolation enforced

### TR-002: Secrets API Endpoints
**Priority:** P0
**Description:** REST API for secrets management with RBAC protection.

**Requirements:**
1. Secret management endpoints:
   - `GET /api/v1/secrets` - List secrets (metadata only, not values)
   - `GET /api/v1/secrets/{path}` - Get secret (requires secrets:read)
   - `POST /api/v1/secrets/{path}` - Create secret (requires secrets:write)
   - `PUT /api/v1/secrets/{path}` - Update secret (requires secrets:write)
   - `DELETE /api/v1/secrets/{path}` - Delete secret (requires secrets:admin)
   - `GET /api/v1/secrets/{path}/versions` - List versions
   - `POST /api/v1/secrets/{path}/undelete` - Undelete version
2. Rotation endpoints:
   - `POST /api/v1/secrets/rotate/{path}` - Trigger manual rotation
   - `GET /api/v1/secrets/rotation/status` - Get rotation status
   - `GET /api/v1/secrets/rotation/schedule` - Get rotation schedule
3. Health endpoints:
   - `GET /api/v1/secrets/health` - Vault health status
   - `GET /api/v1/secrets/status` - Service status

**Acceptance Criteria:**
- [ ] All endpoints protected by RBAC
- [ ] Audit logging for all operations
- [ ] Rate limiting on sensitive operations
- [ ] Secret values never logged

### TR-003: External Secrets Operator Integration
**Priority:** P0
**Description:** Complete ESO integration for Kubernetes secrets sync.

**Requirements:**
1. SecretStore/ClusterSecretStore:
   - Vault backend configuration
   - Kubernetes auth with IRSA
   - Multiple mount points (kv, database, pki)
2. ExternalSecrets for all services:
   - PostgreSQL credentials (greenlang-api, agents, jobs)
   - Redis credentials (cache, rate-limiting)
   - API keys (external integrations)
   - TLS certificates (internal mTLS)
3. PushSecret for bidirectional sync:
   - Application-generated secrets to Vault
   - Certificate renewal sync
4. Sync configuration:
   - RefreshInterval: 30 seconds
   - SecretStoreRef validation
   - Templating for connection strings

**Acceptance Criteria:**
- [ ] All services use ESO-synced secrets
- [ ] Secrets refresh within 30 seconds of Vault change
- [ ] Zero secrets in Git or ConfigMaps

### TR-004: Vault Agent Injector Configuration
**Priority:** P1
**Description:** Configure Vault Agent sidecar injection for pods.

**Requirements:**
1. Injector annotations:
   - `vault.hashicorp.com/agent-inject: "true"`
   - `vault.hashicorp.com/role: "greenlang-agents"`
   - Template annotations for secret rendering
2. Init container mode:
   - Pre-populate secrets before app starts
   - Fail-fast on secret unavailability
3. Sidecar mode:
   - Continuous secret refresh
   - Memory-only secret storage
4. Service-specific configurations:
   - greenlang-api, greenlang-agents, greenlang-jobs
   - Different policies per service

**Acceptance Criteria:**
- [ ] All pods receive secrets via Vault Agent
- [ ] Secrets rendered to /vault/secrets/
- [ ] Automatic rotation picked up without restart

### TR-005: Secrets Metrics & Dashboard
**Priority:** P1
**Description:** Prometheus metrics and Grafana dashboard for secrets operations.

**Requirements:**
1. Prometheus metrics:
   - `gl_secrets_operations_total` Counter (operation, secret_type, result)
   - `gl_secrets_operation_duration_seconds` Histogram (operation)
   - `gl_vault_auth_renewals_total` Counter (status)
   - `gl_vault_lease_ttl_seconds` Gauge (secret_type)
   - `gl_secrets_cache_hits_total` Counter
   - `gl_secrets_cache_misses_total` Counter
   - `gl_secrets_rotation_total` Counter (secret_type, status)
   - `gl_eso_sync_total` Counter (secret, status)
2. Grafana dashboard (16+ panels):
   - Vault health status
   - Secret operations rate
   - Operation latency (P50/P95/P99)
   - Cache hit ratio
   - Rotation events
   - ESO sync status
   - Lease expiry countdown
   - Authentication renewals

**Acceptance Criteria:**
- [ ] All 8 metrics exported
- [ ] Dashboard with 16+ panels
- [ ] Template variables for filtering

### TR-006: Secrets Alerts
**Priority:** P1
**Description:** Prometheus alerts for Vault and secrets operations.

**Requirements:**
1. Vault health alerts:
   - VaultSealed - Vault is sealed
   - VaultUnavailable - Cannot reach Vault
   - VaultHighLatency - Operations > 100ms P99
2. Operations alerts:
   - SecretsOperationFailures - >1% failure rate
   - SecretAccessDenied - Repeated permission denied
   - LeaseExpiringCritical - Lease < 1 hour
3. Rotation alerts:
   - RotationFailed - Rotation failure
   - RotationOverdue - Secret not rotated within policy
   - CertificateExpiring - Cert expires < 7 days
4. ESO alerts:
   - ESOSyncFailed - ExternalSecret sync failure
   - ESOStale - Secret not synced > 5 minutes

**Acceptance Criteria:**
- [ ] 10+ alert rules configured
- [ ] PagerDuty integration for critical
- [ ] Runbook URLs for all alerts

### TR-007: Testing Suite
**Priority:** P2
**Description:** Comprehensive tests for secrets management.

**Requirements:**
1. Unit tests:
   - SecretsService operations (mock Vault)
   - API routes (mock service)
   - Rotation handlers
   - Cache operations
2. Integration tests:
   - Real Vault connection (testcontainers)
   - ESO sync verification
   - End-to-end secret lifecycle
3. Load tests:
   - 1000 secret reads/sec
   - Concurrent rotation
   - Cache performance

**Acceptance Criteria:**
- [ ] 200+ tests total
- [ ] 90%+ code coverage
- [ ] Integration tests pass with real Vault

---

## 3. Architecture

### 3.1 Secrets Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GreenLang Application Layer                         │
└─────────────────────────────────────────────────────────────────────────────┘
                │                                        │
                ▼                                        ▼
    ┌───────────────────────┐                ┌───────────────────────┐
    │   Secrets Service API │                │    Vault Agent        │
    │   /api/v1/secrets/*   │                │    Injector (Sidecar) │
    └───────────┬───────────┘                └───────────┬───────────┘
                │                                        │
                ▼                                        ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                         SecretsService                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
    │  │ VaultClient │  │SecretCache  │  │RotationMgr │  │ TenantCtx   │  │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                         HashiCorp Vault                                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
    │  │  KV v2      │  │  Database   │  │  Transit    │  │    PKI      │  │
    │  │  Secrets    │  │  Engine     │  │  (Encrypt)  │  │ (Certs)     │  │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────────────────┐
    │                     Kubernetes Secrets Sync                            │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │              External Secrets Operator (ESO)                     │  │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │  │
    │  │  │ClusterSecret │  │ExternalSecret│  │   PushSecret         │   │  │
    │  │  │Store (Vault) │  │(Pull from V) │  │(Push to Vault)       │   │  │
    │  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
```

### 3.2 Secret Path Structure

```
secret/
├── data/
│   ├── greenlang/                    # Platform-wide secrets
│   │   ├── database/                 # Database credentials
│   │   │   ├── postgresql            # Main PostgreSQL
│   │   │   ├── redis                 # Redis cache
│   │   │   └── timescaledb           # TimescaleDB extension
│   │   ├── api-keys/                 # API keys
│   │   │   ├── internal              # Internal service keys
│   │   │   └── external              # Partner API keys
│   │   └── services/                 # Service-specific secrets
│   │       ├── auth-service
│   │       ├── agent-factory
│   │       └── audit-service
│   └── tenants/                      # Tenant-scoped secrets
│       ├── {tenant_id}/
│       │   ├── api-keys
│       │   └── integrations
database/                             # Dynamic database secrets
├── creds/
│   ├── greenlang-readonly
│   └── greenlang-readwrite
transit/                              # Encryption keys
├── keys/
│   ├── data-encryption-key
│   └── pii-encryption-key
pki_int/                              # Internal PKI
├── issue/
│   ├── greenlang-services
│   └── internal-mtls
```

---

## 4. Implementation Phases

### Phase 1: Secrets Service Core (P0)
- Create `greenlang/infrastructure/secrets_service/` module
- Wrap existing VaultClient with tenant context
- Add factory methods for common secret types
- Integrate with existing rotation manager

### Phase 2: Secrets API (P0)
- API endpoints for secrets CRUD
- Rotation trigger endpoints
- Health and status endpoints
- RBAC permission mapping

### Phase 3: ESO Complete Integration (P0)
- Additional ExternalSecrets for all services
- PushSecret for application-generated secrets
- SecretStore per namespace
- Sync verification

### Phase 4: Vault Agent Injector (P1)
- Service-specific injection configs
- Pod annotation templates
- Helm values for agent settings
- Init vs sidecar mode selection

### Phase 5: Monitoring (P1)
- Prometheus metrics
- Grafana dashboard
- Alert rules
- Runbook documentation

### Phase 6: Integration (P1)
- Wire into auth_service
- RBAC permissions for secrets
- Database migration for metadata

### Phase 7: Testing (P2)
- Unit tests for all components
- Integration tests with real Vault
- Load tests for performance

---

## 5. Security Considerations

### 5.1 Access Control
- RBAC permissions: secrets:read, secrets:write, secrets:admin, secrets:rotate
- Vault policies per service (least privilege)
- Tenant isolation via path-based policies
- No wildcard permissions

### 5.2 Audit Trail
- All secret access logged to audit service
- Vault audit device enabled
- No secret values in logs (only metadata)
- Correlation with request trace ID

### 5.3 Secret Hygiene
- No secrets in environment variables
- No secrets in ConfigMaps
- No secrets committed to Git
- Secrets rotated per policy

---

## 6. Compliance Mapping

| Requirement | SOC 2 | ISO 27001 | GDPR | PCI DSS |
|-------------|-------|-----------|------|---------|
| Centralized secrets | CC6.1 | A.9.4.3 | Art. 32 | 3.5 |
| Access control | CC6.3 | A.9.2.3 | Art. 32 | 3.6 |
| Rotation policy | CC6.1 | A.9.4.3 | Art. 32 | 3.6 |
| Audit logging | CC7.2 | A.12.4.1 | Art. 30 | 10.2 |
| Encryption | CC6.7 | A.10.1.1 | Art. 32 | 3.4 |

---

## 7. Deliverables Summary

| Component | Files | Priority |
|-----------|-------|----------|
| Secrets Service Core | 6 | P0 |
| Secrets API | 4 | P0 |
| ESO Integration | 8 | P0 |
| Vault Agent Config | 4 | P1 |
| Monitoring (Metrics + Dashboard) | 3 | P1 |
| Alert Rules | 1 | P1 |
| Integration | 3 | P1 |
| Testing | 10 | P2 |
| **TOTAL** | **~39** | - |

---

## 8. Appendix

### A. Vault Policies

```hcl
# greenlang-api policy
path "secret/data/greenlang/*" {
  capabilities = ["read"]
}

path "database/creds/greenlang-readwrite" {
  capabilities = ["read"]
}

path "transit/encrypt/data-encryption-key" {
  capabilities = ["update"]
}

path "transit/decrypt/data-encryption-key" {
  capabilities = ["update"]
}
```

### B. ExternalSecret Example

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: postgresql-credentials
spec:
  refreshInterval: 30s
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: postgresql-credentials
  data:
    - secretKey: POSTGRES_USER
      remoteRef:
        key: secret/data/greenlang/database/postgresql
        property: username
    - secretKey: POSTGRES_PASSWORD
      remoteRef:
        key: secret/data/greenlang/database/postgresql
        property: password
```

### C. Environment Configuration

| Setting | Dev | Staging | Prod |
|---------|-----|---------|------|
| Vault Replicas | 1 | 3 | 5 |
| ESO Refresh | 60s | 30s | 15s |
| Rotation Interval | 7d | 1d | 12h |
| Lease TTL | 24h | 12h | 6h |
| Audit Retention | 7d | 30d | 365d |
