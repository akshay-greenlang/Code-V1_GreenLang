# SEC-003: Encryption at Rest (AES-256) - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** SEC-001, SEC-002, INFRA-002, INFRA-003, INFRA-004
**Result:** 30 new files + 2 modified, ~15,500 lines

---

## Phase 1: Core Encryption Service (P0)

### 1.1 Package Init
- [x] Create `greenlang/infrastructure/encryption_service/__init__.py` - Public API exports:
  - EncryptionService, EnvelopeEncryptionService, FieldEncryptor, KeyManager
  - EncryptionServiceConfig dataclass
  - EncryptedData, DecryptedData result types

### 1.2 Core Encryption Service
- [x] Create `greenlang/infrastructure/encryption_service/encryption_service.py` - AES-256-GCM:
  - `EncryptionService(kms_client, config)` class
  - `encrypt(plaintext, context, aad)` -> EncryptedData (ciphertext, nonce, tag, key_version)
  - `decrypt(encrypted_data, context)` -> bytes
  - `encrypt_bytes(data, key)` - low-level AES-256-GCM with cryptography library
  - `decrypt_bytes(ciphertext, key, nonce, tag)` - low-level decryption
  - Secure nonce generation (12 bytes, CSPRNG)
  - Zero-copy where possible, zeroize keys after use

### 1.3 Envelope Encryption
- [x] Create `greenlang/infrastructure/encryption_service/envelope_encryption.py` - KMS DEK wrapping:
  - `EnvelopeEncryptionService(kms_client, config)` class
  - `generate_data_key(context)` -> { plaintext_key, encrypted_key }
  - `decrypt_data_key(encrypted_key, context)` -> plaintext_key
  - `encrypt_with_dek(plaintext, dek, aad)` -> EnvelopeEncryptedData
  - `decrypt_with_dek(envelope, context)` -> plaintext
  - Uses existing `greenlang/governance/security/kms/aws_kms.py` for KMS calls
  - Encryption context binding (tenant_id, data_class)

### 1.4 Key Management
- [x] Create `greenlang/infrastructure/encryption_service/key_management.py` - DEK caching:
  - `KeyManager(kms_client, config)` class
  - `DEKCache` - in-memory LRU cache with TTL (5 min default)
  - `get_or_generate_dek(context)` - cache-first, then KMS
  - `invalidate_dek(context)` - remove from cache
  - `rotate_dek(context)` - generate new, keep old for decryption
  - Key versioning: track active and previous versions
  - Metrics: cache hits, misses, KMS calls

---

## Phase 2: Field-Level Encryption (P0)

### 2.1 Field Encryption Utilities
- [x] Create `greenlang/infrastructure/encryption_service/field_encryption.py` - DB column encryption:
  - `FieldEncryptor(encryption_service)` class
  - `encrypt_field(value, field_name, tenant_id)` -> encrypted string (base64)
  - `decrypt_field(encrypted_value, field_name, tenant_id)` -> original value
  - `EncryptedColumn(TypeDecorator)` - SQLAlchemy custom type for transparent encrypt/decrypt
  - `create_hmac_index(value, field_name)` -> searchable hash for encrypted columns
  - Support for: str, int, datetime, JSON serialization before encryption

### 2.2 Database Migration
- [x] Create `deployment/database/migrations/sql/V011__field_level_encryption.sql`:
  - Enable pgcrypto if not already (already in V001, verify)
  - Create helper functions: `gl_encrypt(data, key)`, `gl_decrypt(data, key)`
  - Add encrypted columns to existing tables where needed (if any PII exists)
  - Create HMAC index columns for searchable encrypted fields
  - Backfill migration for existing plaintext data (with transaction safety)

---

## Phase 3: Audit, Metrics & API (P1)

### 3.1 Encryption Audit Logger
- [x] Create `greenlang/infrastructure/encryption_service/encryption_audit.py`:
  - `EncryptionAuditEventType` enum: encryption_performed, decryption_performed, key_generated, key_rotated, key_accessed, encryption_failed, decryption_failed, key_cache_hit, key_cache_miss
  - `EncryptionAuditLogger(db_pool)` class
  - `log_event(event_type, data_class, tenant_id, key_version, success, error_msg, correlation_id)`
  - Structured JSON logging for Loki (event_category="encryption")
  - Async fire-and-forget DB writes
  - NO plaintext data in logs

### 3.2 Encryption Metrics
- [x] Create `greenlang/infrastructure/encryption_service/encryption_metrics.py`:
  - `gl_encryption_operations_total` Counter (operation, status, data_class)
  - `gl_encryption_duration_seconds` Histogram (operation)
  - `gl_encryption_key_cache_hits` Counter (key_type)
  - `gl_encryption_key_cache_misses` Counter (key_type)
  - `gl_encryption_failures_total` Counter (error_type)
  - `gl_kms_calls_total` Counter (operation, status)
  - `gl_kms_latency_seconds` Histogram (operation)
  - Lazy initialization pattern (like SEC-001)

### 3.3 API Router Init
- [x] Create `greenlang/infrastructure/encryption_service/api/__init__.py`:
  - Export encryption_router

### 3.4 Encryption API Routes
- [x] Create `greenlang/infrastructure/encryption_service/api/encryption_routes.py`:
  - `POST /api/v1/encryption/encrypt` - Encrypt data (admin/service only)
  - `POST /api/v1/encryption/decrypt` - Decrypt data (admin/service only)
  - `GET /api/v1/encryption/keys` - List active key versions
  - `POST /api/v1/encryption/keys/rotate` - Trigger manual key rotation
  - `GET /api/v1/encryption/audit` - Encryption audit log (paginated)
  - `GET /api/v1/encryption/status` - Health check (encryption working, KMS reachable)
  - Pydantic request/response models
  - Rate limiting on encrypt/decrypt endpoints

---

## Phase 4: KMS Terraform Module (P1)

### 4.1 KMS Main Configuration
- [x] Create `deployment/terraform/modules/kms/main.tf`:
  - `aws_kms_key.master` - Master CMK for GreenLang
  - `aws_kms_key.database` - Database encryption CMK
  - `aws_kms_key.storage` - S3/EFS encryption CMK
  - `aws_kms_key.cache` - ElastiCache CMK
  - `aws_kms_key.secrets` - Secrets Manager CMK
  - `aws_kms_key.application` - Application DEKs CMK
  - Key aliases for each
  - Enable automatic key rotation
  - Deletion protection (30-day window)

### 4.2 KMS Variables
- [x] Create `deployment/terraform/modules/kms/variables.tf`:
  - environment, project_name, aws_region
  - enable_key_rotation (default true)
  - deletion_window_days (default 30)
  - multi_region_enabled (default false)
  - key_administrators (IAM ARNs)
  - service_roles (IAM roles allowed to use keys)
  - tags

### 4.3 KMS Outputs
- [x] Create `deployment/terraform/modules/kms/outputs.tf`:
  - All key ARNs, key IDs, aliases
  - Policy document outputs for IAM integration

### 4.4 KMS Policies
- [x] Create `deployment/terraform/modules/kms/policies.tf`:
  - Key policy documents for each CMK
  - Root account access
  - Key administrators
  - Service role access (EKS, RDS, S3, ElastiCache, Lambda)
  - Cross-account access if needed

### 4.5 KMS Documentation
- [x] Create `deployment/terraform/modules/kms/README.md`:
  - Usage examples
  - Key hierarchy diagram
  - IAM permission requirements
  - Integration guide

---

## Phase 5: Integration (P1)

### 5.1 Auth Setup Integration
- [x] Modify `greenlang/infrastructure/auth_service/auth_setup.py`:
  - Import and include encryption_router in `_include_auth_routers()`

### 5.2 Route Protector Update
- [x] Update `greenlang/infrastructure/auth_service/route_protector.py`:
  - Add encryption permission mappings to PERMISSION_MAP:
    - `POST:/api/v1/encryption/encrypt` -> `encryption:encrypt`
    - `POST:/api/v1/encryption/decrypt` -> `encryption:decrypt`
    - `GET:/api/v1/encryption/keys` -> `encryption:admin`
    - `POST:/api/v1/encryption/keys/rotate` -> `encryption:admin`
    - `GET:/api/v1/encryption/audit` -> `encryption:audit`
    - `GET:/api/v1/encryption/status` -> `encryption:read`

### 5.3 RBAC Permissions Seeding
- [x] Create `deployment/database/migrations/sql/V012__encryption_permissions.sql`:
  - Add encryption permissions to security.permissions table
  - Map permissions to appropriate roles (admin, service_account)

---

## Phase 6: Monitoring (P2)

### 6.1 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/encryption-service.json`:
  - Encryption operations over time (encrypt/decrypt)
  - DEK cache hit rate
  - KMS call latency (P50/P95/P99)
  - Encryption failures by type
  - KMS calls by operation
  - Active key versions
  - Encryption throughput

### 6.2 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/encryption-service-alerts.yaml`:
  - High encryption failure rate (> 1% over 5m)
  - KMS latency high (P99 > 100ms)
  - DEK cache hit rate low (< 90%)
  - Encryption service errors
  - KMS unreachable
  - Key rotation overdue (> 365 days)
  - Unusual decryption volume spike

---

## Phase 7: Testing (P2)

### 7.1 Unit Tests
- [x] Create `tests/unit/encryption_service/__init__.py`
- [x] Create `tests/unit/encryption_service/test_encryption_service.py` - 30+ tests:
  - encrypt/decrypt round-trip, nonce uniqueness, AAD validation, key handling
- [x] Create `tests/unit/encryption_service/test_envelope_encryption.py` - 25+ tests:
  - DEK generation, wrapping, unwrapping, context binding
- [x] Create `tests/unit/encryption_service/test_field_encryption.py` - 20+ tests:
  - Field encrypt/decrypt, EncryptedColumn type, HMAC index
- [x] Create `tests/unit/encryption_service/test_key_management.py` - 20+ tests:
  - DEK cache, rotation, versioning, invalidation

### 7.2 Integration Tests
- [x] Create `tests/integration/encryption_service/__init__.py`
- [x] Create `tests/integration/encryption_service/test_encryption_e2e.py` - 15+ tests:
  - Full encryption flow with mocked KMS
- [x] Create `tests/integration/encryption_service/test_kms_integration.py` - 10+ tests:
  - Real KMS calls (localstack or test account)

### 7.3 Compliance Tests
- [x] Create `tests/compliance/__init__.py`
- [x] Create `tests/compliance/test_encryption_compliance.py` - 10+ tests:
  - AES-256 algorithm verification, key length, nonce uniqueness
  - No plaintext in logs, audit trail completeness

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Core Encryption Service | 4/4 | P0 | COMPLETE |
| Phase 2: Field-Level Encryption | 2/2 | P0 | COMPLETE |
| Phase 3: Audit, Metrics & API | 4/4 | P1 | COMPLETE |
| Phase 4: KMS Terraform Module | 5/5 | P1 | COMPLETE |
| Phase 5: Integration | 3/3 | P1 | COMPLETE |
| Phase 6: Monitoring | 2/2 | P2 | COMPLETE |
| Phase 7: Testing | 10/10 | P2 | COMPLETE |
| **TOTAL** | **30/30** | - | **COMPLETE** |
