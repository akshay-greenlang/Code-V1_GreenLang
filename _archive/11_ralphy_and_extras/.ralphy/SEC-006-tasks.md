# SEC-006: Deploy Secrets Management (HashiCorp Vault) - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-001, SEC-001, SEC-002, SEC-003
**Existing Code:** Terraform module (633 lines), VaultClient (1063 lines), SecretsRotationManager (893 lines)
**Result:** 45 new files + 2 modified, ~16,500 lines, 200+ tests

---

## Phase 1: Secrets Service Core (P0)

### 1.1 Package Init
- [x] Create `greenlang/infrastructure/secrets_service/__init__.py`:
  - Public API exports: SecretsService, SecretsServiceConfig, SecretType, etc.
  - Re-export VaultClient, SecretsRotationManager from execution/infrastructure/secrets
  - Version constant

### 1.2 Service Configuration
- [x] Create `greenlang/infrastructure/secrets_service/config.py`:
  - `SecretsServiceConfig` dataclass with environment defaults
  - Vault address, namespace, auth settings
  - Cache TTL, rotation intervals
  - ESO sync settings

### 1.3 Secret Types
- [x] Create `greenlang/infrastructure/secrets_service/secret_types.py`:
  - `SecretType` enum (database, api_key, certificate, encryption_key, service_token)
  - `SecretMetadata` dataclass (path, type, version, created, expires)
  - `SecretReference` for lazy loading

### 1.4 Secrets Service
- [x] Create `greenlang/infrastructure/secrets_service/secrets_service.py`:
  - `SecretsService` class wrapping VaultClient
  - Tenant context injection (from auth context)
  - `get_secret(path, tenant_id)` - tenant-scoped get
  - `put_secret(path, data, tenant_id)` - tenant-scoped put
  - `delete_secret(path, tenant_id)` - tenant-scoped delete
  - `list_secrets(prefix, tenant_id)` - list metadata
  - `get_secret_versions(path)` - version history
  - Factory methods: `get_database_credentials()`, `get_api_key()`, etc.
  - Integration with SecretsRotationManager

### 1.5 Tenant Isolation
- [x] Create `greenlang/infrastructure/secrets_service/tenant_context.py`:
  - `TenantSecretContext` class
  - Path prefixing for tenant isolation
  - Policy validation for cross-tenant access
  - Platform vs tenant secret differentiation

### 1.6 Secrets Cache
- [x] Create `greenlang/infrastructure/secrets_service/cache.py`:
  - `SecretsCache` with Redis L1 + memory L2
  - TTL-based invalidation
  - Version-aware caching
  - Cache metrics

---

## Phase 2: Secrets API (P0)

### 2.1 API Init
- [x] Create `greenlang/infrastructure/secrets_service/api/__init__.py`:
  - Export combined secrets_router

### 2.2 Secrets Routes
- [x] Create `greenlang/infrastructure/secrets_service/api/secrets_routes.py`:
  - `GET /api/v1/secrets` - List secrets (metadata only)
  - `GET /api/v1/secrets/{path:path}` - Get secret value
  - `POST /api/v1/secrets/{path:path}` - Create secret
  - `PUT /api/v1/secrets/{path:path}` - Update secret
  - `DELETE /api/v1/secrets/{path:path}` - Delete secret
  - `GET /api/v1/secrets/{path:path}/versions` - Version history
  - `POST /api/v1/secrets/{path:path}/undelete` - Restore version
  - Pydantic models: SecretRequest, SecretResponse, SecretListResponse

### 2.3 Rotation Routes
- [x] Create `greenlang/infrastructure/secrets_service/api/rotation_routes.py`:
  - `POST /api/v1/secrets/rotate/{path:path}` - Trigger rotation
  - `GET /api/v1/secrets/rotation/status` - Rotation status
  - `GET /api/v1/secrets/rotation/schedule` - Rotation schedule
  - `POST /api/v1/secrets/rotation/schedule` - Update schedule

### 2.4 Health Routes
- [x] Create `greenlang/infrastructure/secrets_service/api/health_routes.py`:
  - `GET /api/v1/secrets/health` - Vault health
  - `GET /api/v1/secrets/status` - Service status
  - `GET /api/v1/secrets/stats` - Operation statistics

---

## Phase 3: External Secrets Operator Integration (P0)

### 3.1 ClusterSecretStore Updates
- [x] Create `deployment/kubernetes/secrets-service/clustersecretstore-vault.yaml`:
  - Add kv-v2, database, transit mount points
  - IRSA auth configuration
  - Retry and timeout settings

### 3.2 API Service ExternalSecrets
- [x] Create `deployment/kubernetes/secrets-service/externalsecrets-api.yaml`:
  - PostgreSQL credentials for greenlang-api
  - Redis credentials
  - JWT signing keys
  - Internal API keys

### 3.3 Agent Factory ExternalSecrets
- [x] Create `deployment/kubernetes/secrets-service/externalsecrets-agents.yaml`:
  - PostgreSQL credentials for agents
  - S3 credentials for agent artifacts
  - Transit key access
  - Pack signing keys

### 3.4 Audit Service ExternalSecrets
- [x] Create `deployment/kubernetes/secrets-service/externalsecrets-audit.yaml`:
  - PostgreSQL credentials for audit tables
  - S3 credentials for log archival
  - Encryption keys for PII

### 3.5 Auth Service ExternalSecrets
- [x] Create `deployment/kubernetes/secrets-service/externalsecrets-auth.yaml`:
  - JWT signing keys (RS256)
  - Refresh token encryption key
  - Password hashing pepper

### 3.6 PushSecret for App-Generated Secrets
- [x] Create `deployment/kubernetes/secrets-service/pushsecrets.yaml`:
  - Application-generated tokens to Vault
  - Certificate renewal sync
  - Dynamic secret storage

### 3.7 Namespace SecretStores
- [x] Create `deployment/kubernetes/secrets-service/secretstores-per-ns.yaml`:
  - SecretStore for greenlang namespace
  - SecretStore for monitoring namespace
  - SecretStore for logging namespace

### 3.8 ESO Kustomization
- [x] Create `deployment/kubernetes/secrets-service/kustomization.yaml`:
  - Include all ExternalSecrets
  - Environment overlays (dev, staging, prod)

---

## Phase 4: Vault Agent Injector Configuration (P1)

### 4.1 API Service Agent Config
- [x] Create `deployment/kubernetes/vault-agent/api-service-patches.yaml`:
  - vault.hashicorp.com annotations for greenlang-api
  - Secret templates for /vault/secrets/
  - Init container mode for startup

### 4.2 Agent Factory Agent Config
- [x] Create `deployment/kubernetes/vault-agent/agent-factory-patches.yaml`:
  - vault.hashicorp.com annotations for agent-factory
  - S3 credential templates
  - Sidecar mode for rotation

### 4.3 Jobs Service Agent Config
- [x] Create `deployment/kubernetes/vault-agent/jobs-service-patches.yaml`:
  - vault.hashicorp.com annotations for greenlang-jobs
  - Database credential templates

### 4.4 Vault Agent Templates and Kustomization
- [x] Create `deployment/kubernetes/vault-agent/configmap-templates.yaml`:
  - Vault agent template ConfigMap
  - Per-service vault configuration
  - Template definitions
- [x] Create `deployment/kubernetes/vault-agent/kustomization.yaml`

---

## Phase 5: Metrics & Dashboard (P1)

### 5.1 Secrets Metrics
- [x] Create `greenlang/infrastructure/secrets_service/metrics.py`:
  - `gl_secrets_operations_total` Counter (operation, secret_type, result)
  - `gl_secrets_operation_duration_seconds` Histogram (operation)
  - `gl_vault_auth_renewals_total` Counter (status)
  - `gl_vault_lease_ttl_seconds` Gauge (secret_type)
  - `gl_secrets_cache_hits_total` Counter
  - `gl_secrets_cache_misses_total` Counter
  - `gl_secrets_rotation_total` Counter (secret_type, status)
  - `gl_eso_sync_total` Counter (secret, status)
  - Lazy initialization pattern

### 5.2 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/secrets-service.json`:
  - Vault health panel
  - Secret operations rate (by type)
  - Operation latency (P50/P95/P99)
  - Cache hit ratio
  - Rotation events timeline
  - ESO sync status
  - Lease expiry countdown (top 10)
  - Auth renewal rate
  - Error rate by operation
  - Active leases gauge
  - Vault storage usage
  - Top accessed secrets
  - Failed operations table
  - Secret version distribution
  - Rotation schedule
  - Service status summary

### 5.3 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/secrets-service-alerts.yaml`:
  - VaultSealed (critical)
  - VaultUnavailable (critical)
  - VaultHighLatency (>100ms P99)
  - SecretsOperationFailures (>1%)
  - SecretAccessDenied (repeated)
  - LeaseExpiringCritical (<1h)
  - RotationFailed (critical)
  - RotationOverdue
  - CertificateExpiring (<7d)
  - ESOSyncFailed
  - ESOStale (>5m)
  - VaultTokenRenewalFailed

---

## Phase 6: Integration (P1)

### 6.1 Auth Setup Integration
- [x] Modify `greenlang/infrastructure/auth_service/auth_setup.py`:
  - Import and include secrets_router
  - Wire SecretsService dependency

### 6.2 Route Protector Update
- [x] Update `greenlang/infrastructure/auth_service/route_protector.py`:
  - Add secrets permission mappings:
    - `GET:/api/v1/secrets` -> `secrets:list`
    - `GET:/api/v1/secrets/{path}` -> `secrets:read`
    - `POST:/api/v1/secrets/{path}` -> `secrets:write`
    - `PUT:/api/v1/secrets/{path}` -> `secrets:write`
    - `DELETE:/api/v1/secrets/{path}` -> `secrets:admin`
    - `POST:/api/v1/secrets/rotate/*` -> `secrets:rotate`
    - `GET:/api/v1/secrets/rotation/*` -> `secrets:read`
    - `GET:/api/v1/secrets/health` -> `secrets:read`

### 6.3 RBAC Permissions Migration
- [x] Create `deployment/database/migrations/sql/V014__secrets_permissions.sql`:
  - Add secrets permissions (list, read, write, admin, rotate)
  - Map permissions to roles
  - Secrets audit table

---

## Phase 7: Testing (P2)

### 7.1 Unit Tests
- [x] Create `tests/unit/secrets_service/__init__.py`
- [x] Create `tests/unit/secrets_service/conftest.py` - Shared fixtures
- [x] Create `tests/unit/secrets_service/test_secrets_service.py` - 30+ tests:
  - Service initialization
  - Tenant context injection
  - Secret CRUD operations
  - Cache behavior
- [x] Create `tests/unit/secrets_service/test_secret_types.py` - 15+ tests:
  - Type validation
  - Metadata serialization
- [x] Create `tests/unit/secrets_service/test_tenant_context.py` - 20+ tests:
  - Path prefixing
  - Cross-tenant validation
  - Platform secret access
- [x] Create `tests/unit/secrets_service/test_cache.py` - 15+ tests:
  - Cache operations
  - TTL behavior
- [x] Create `tests/unit/secrets_service/test_secrets_routes.py` - 25+ tests:
  - All API endpoints
  - Error handling
  - RBAC enforcement
- [x] Create `tests/unit/secrets_service/test_rotation_routes.py` - 15+ tests:
  - Rotation triggers
  - Schedule management
- [x] Create `tests/unit/secrets_service/test_health_routes.py` - 10+ tests:
  - Health endpoints
  - Status reporting

### 7.2 Integration Tests
- [x] Create `tests/integration/secrets_service/__init__.py`
- [x] Create `tests/integration/secrets_service/test_vault_integration.py` - 20+ tests:
  - Real Vault connection (testcontainers or local)
  - Full secret lifecycle
  - Rotation verification
- [x] Create `tests/integration/secrets_service/test_eso_sync.py` - 10+ tests:
  - ExternalSecret sync verification
  - Refresh timing
- [x] Create `tests/integration/secrets_service/test_agent_injector.py` - 10+ tests:
  - Annotation parsing
  - Template rendering

### 7.3 Load Tests
- [x] Create `tests/load/secrets_service/__init__.py`
- [x] Create `tests/load/secrets_service/test_secrets_throughput.py` - 10+ tests:
  - 1000 reads/sec
  - Concurrent rotation
  - Cache performance
  - Vault connection pooling

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Secrets Service Core | 6/6 | P0 | COMPLETE |
| Phase 2: Secrets API | 4/4 | P0 | COMPLETE |
| Phase 3: ESO Integration | 8/8 | P0 | COMPLETE |
| Phase 4: Vault Agent Injector | 5/5 | P1 | COMPLETE |
| Phase 5: Metrics & Dashboard | 3/3 | P1 | COMPLETE |
| Phase 6: Integration | 3/3 | P1 | COMPLETE |
| Phase 7: Testing | 15/15 | P2 | COMPLETE |
| **TOTAL** | **44/44** | - | **COMPLETE** |

---

## Files Created

### Python Modules (12 files)
- `greenlang/infrastructure/secrets_service/__init__.py`
- `greenlang/infrastructure/secrets_service/config.py`
- `greenlang/infrastructure/secrets_service/secret_types.py`
- `greenlang/infrastructure/secrets_service/secrets_service.py`
- `greenlang/infrastructure/secrets_service/tenant_context.py`
- `greenlang/infrastructure/secrets_service/cache.py`
- `greenlang/infrastructure/secrets_service/metrics.py`
- `greenlang/infrastructure/secrets_service/api.py`
- `greenlang/infrastructure/secrets_service/api/__init__.py`
- `greenlang/infrastructure/secrets_service/api/secrets_routes.py`
- `greenlang/infrastructure/secrets_service/api/rotation_routes.py`
- `greenlang/infrastructure/secrets_service/api/health_routes.py`

### Kubernetes Manifests (14 files)
- `deployment/kubernetes/secrets-service/namespace.yaml`
- `deployment/kubernetes/secrets-service/clustersecretstore-vault.yaml`
- `deployment/kubernetes/secrets-service/externalsecrets-api.yaml`
- `deployment/kubernetes/secrets-service/externalsecrets-agents.yaml`
- `deployment/kubernetes/secrets-service/externalsecrets-audit.yaml`
- `deployment/kubernetes/secrets-service/externalsecrets-auth.yaml`
- `deployment/kubernetes/secrets-service/pushsecrets.yaml`
- `deployment/kubernetes/secrets-service/secretstores-per-ns.yaml`
- `deployment/kubernetes/secrets-service/kustomization.yaml`
- `deployment/kubernetes/vault-agent/configmap-templates.yaml`
- `deployment/kubernetes/vault-agent/api-service-patches.yaml`
- `deployment/kubernetes/vault-agent/agent-factory-patches.yaml`
- `deployment/kubernetes/vault-agent/jobs-service-patches.yaml`
- `deployment/kubernetes/vault-agent/kustomization.yaml`

### Monitoring (3 files)
- `deployment/monitoring/dashboards/secrets-service.json`
- `deployment/monitoring/alerts/secrets-service-alerts.yaml`
- `deployment/monitoring/alerts/secrets-management-alerts.yaml`

### Database Migration (1 file)
- `deployment/database/migrations/sql/V014__secrets_permissions.sql`

### Tests (15 files)
- `tests/unit/secrets_service/__init__.py`
- `tests/unit/secrets_service/conftest.py`
- `tests/unit/secrets_service/test_secrets_service.py`
- `tests/unit/secrets_service/test_secret_types.py`
- `tests/unit/secrets_service/test_tenant_context.py`
- `tests/unit/secrets_service/test_cache.py`
- `tests/unit/secrets_service/test_secrets_routes.py`
- `tests/unit/secrets_service/test_rotation_routes.py`
- `tests/unit/secrets_service/test_health_routes.py`
- `tests/integration/secrets_service/__init__.py`
- `tests/integration/secrets_service/test_vault_integration.py`
- `tests/integration/secrets_service/test_eso_sync.py`
- `tests/integration/secrets_service/test_agent_injector.py`
- `tests/load/secrets_service/__init__.py`
- `tests/load/secrets_service/test_secrets_throughput.py`

### Modified Files (2 files)
- `greenlang/infrastructure/auth_service/auth_setup.py`
- `greenlang/infrastructure/auth_service/route_protector.py`
