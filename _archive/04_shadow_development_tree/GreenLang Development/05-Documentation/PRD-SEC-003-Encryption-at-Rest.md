# PRD: SEC-003 - Encryption at Rest (AES-256)

**Document Version:** 1.0
**Date:** February 6, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Security Team
**Ralphy Task ID:** SEC-003
**Depends On:** SEC-001 (JWT Auth), INFRA-002 (PostgreSQL), INFRA-003 (Redis), INFRA-004 (S3)

---

## Executive Summary

Build a comprehensive encryption-at-rest solution using AES-256 encryption for all data stores in GreenLang. This PRD covers both infrastructure-level encryption (already partially configured) and application-level encryption (field-level encryption for PII, envelope encryption for large data). The goal is to achieve SOC 2 Type II and ISO 27001 compliance for data protection at rest.

### Current State
- **Aurora PostgreSQL**: `storage_encrypted = true` with KMS CMK ✓
- **ElastiCache Redis**: `at_rest_encryption_enabled = true` with KMS ✓
- **S3 Buckets**: SSE-KMS/AES256 encryption configured ✓
- **EKS Secrets**: Envelope encryption with KMS ✓
- **KMS Provider Library**: `greenlang/governance/security/kms/` exists (signing-focused)
- **pgcrypto Extension**: Enabled but unused
- **Application Encryption**: Only CSRD-APP has local Fernet utility (AES-128)

### Gaps Identified
1. **No centralized encryption service** in `greenlang/infrastructure/`
2. **No field-level encryption** for PII columns (email, API keys, tokens stored in DB)
3. **No envelope encryption** pattern for large data objects
4. **No centralized KMS Terraform module** for consistent key management
5. **No encryption audit/compliance verification** tooling
6. **No key rotation automation** at application level
7. **Existing KMS library is signing-focused**, not data encryption-focused
8. **AES-128 in CSRD-APP**, should be **AES-256** for compliance

### Target State
1. **Centralized Encryption Service**: `greenlang/infrastructure/encryption_service/` with AES-256-GCM
2. **Field-Level Encryption**: Transparent encryption for sensitive database columns
3. **Envelope Encryption**: AWS KMS-wrapped data encryption keys (DEKs)
4. **Centralized KMS Module**: `deployment/terraform/modules/kms/` for all services
5. **Compliance Verification**: Encryption audit tooling and monitoring
6. **Key Rotation**: Automated annual rotation with zero-downtime re-encryption
7. **85%+ test coverage** on encryption code

---

## Scope

### In Scope
1. **Encryption Service Module**: `greenlang/infrastructure/encryption_service/`
   - AES-256-GCM symmetric encryption
   - Envelope encryption with AWS KMS DEKs
   - Field-level encryption utilities
   - Key rotation support
   - Caching for performance
2. **Database Field Encryption**:
   - pgcrypto-based column encryption for PII
   - Transparent encryption/decryption in Python ORM layer
   - Migration to encrypt existing plaintext data
3. **KMS Terraform Module**: `deployment/terraform/modules/kms/`
   - Centralized CMK for all services
   - Key policies for service access
   - Multi-region support for DR
   - Automatic key rotation
4. **Encryption Audit & Monitoring**:
   - Encryption status verification scripts
   - Prometheus metrics for encryption operations
   - CloudWatch alarms for KMS access
5. **Integration with Existing Infrastructure**:
   - Verify and document existing infrastructure encryption settings
   - Ensure all Terraform modules use centralized KMS keys
6. **Comprehensive Testing**: Unit, integration, and compliance tests

### Out of Scope
- Encryption in transit (TLS) - covered by INFRA-006 (Kong) and service configs
- Tokenization of PII (SEC-006 or future)
- Hardware Security Modules (HSM) - AWS KMS uses HSM-backed keys
- Client-side encryption in browser

---

## Architecture

### Component Architecture

```
greenlang/infrastructure/encryption_service/
├── __init__.py                     # Public API exports, EncryptionServiceConfig
├── encryption_service.py           # Core AES-256-GCM encryption/decryption
├── envelope_encryption.py          # AWS KMS envelope encryption (DEK wrapping)
├── field_encryption.py             # Database field-level encryption utilities
├── key_management.py               # Key rotation, DEK caching, key lifecycle
├── encryption_audit.py             # Audit logging for encryption operations
├── encryption_metrics.py           # Prometheus metrics
└── api/
    ├── __init__.py
    └── encryption_routes.py        # /api/v1/encryption/* admin endpoints

deployment/terraform/modules/kms/
├── main.tf                         # KMS CMK definitions
├── variables.tf                    # Input variables
├── outputs.tf                      # Key ARNs, aliases
├── policies.tf                     # Key policies for service access
└── README.md                       # Documentation

deployment/database/migrations/sql/
└── V011__field_level_encryption.sql  # pgcrypto functions, encrypted columns

tests/
├── unit/encryption_service/
│   ├── test_encryption_service.py
│   ├── test_envelope_encryption.py
│   ├── test_field_encryption.py
│   └── test_key_management.py
├── integration/encryption_service/
│   ├── test_encryption_e2e.py
│   └── test_kms_integration.py
└── compliance/
    └── test_encryption_compliance.py
```

### Encryption Flow (Envelope Encryption)

```
1. Application needs to encrypt sensitive data:
   App --> EncryptionService.encrypt(plaintext, context)

2. EncryptionService checks DEK cache:
   --> Cache hit: use cached DEK
   --> Cache miss: request new DEK from KMS

3. Request DEK from AWS KMS:
   EncryptionService --> KMS.GenerateDataKey(CMK_ARN, AES_256)
   KMS --> { plaintext_dek, encrypted_dek }

4. Cache plaintext DEK (in-memory, short TTL):
   DEKCache.set(context, plaintext_dek, encrypted_dek, ttl=5min)

5. Encrypt data with DEK:
   ciphertext = AES-256-GCM.encrypt(plaintext, plaintext_dek, nonce)

6. Return encrypted envelope:
   { ciphertext, encrypted_dek, nonce, auth_tag, key_version }

7. For decryption:
   --> Use cached plaintext DEK if available
   --> Else: KMS.Decrypt(encrypted_dek) --> plaintext_dek
   --> AES-256-GCM.decrypt(ciphertext, plaintext_dek, nonce)
```

### Field-Level Encryption (Database)

```
PostgreSQL with pgcrypto:

1. Sensitive columns encrypted at write:
   INSERT INTO users (email_encrypted) VALUES (
     pgp_sym_encrypt('user@example.com', $DEK, 'cipher-algo=aes256')
   );

2. Decrypted at read:
   SELECT pgp_sym_decrypt(email_encrypted, $DEK) AS email FROM users;

3. Python ORM integration:
   class User(Base):
       email = EncryptedColumn(String(256), sensitive=True)
       # Automatically encrypts on write, decrypts on read
```

---

## Technical Requirements

### TR-001: Encryption Service Core

**AES-256-GCM symmetric encryption with secure key handling.**

**Requirements:**
1. Use AES-256-GCM (Galois/Counter Mode) for authenticated encryption
2. 256-bit keys (32 bytes) derived from AWS KMS CMK
3. 96-bit nonces (12 bytes), cryptographically random, never reused
4. Authentication tag included in ciphertext
5. Support for Associated Authenticated Data (AAD) for context binding
6. Zero plaintext keys in logs, exceptions, or error messages
7. Secure memory handling (zeroize keys after use where possible)

**Algorithms:**
| Purpose | Algorithm | Key Size | Notes |
|---------|-----------|----------|-------|
| Symmetric Encryption | AES-256-GCM | 256-bit | Authenticated encryption |
| Key Wrapping | AWS KMS | CMK | Envelope encryption |
| Random Generation | CSPRNG | N/A | os.urandom() or secrets module |
| Hashing (HKDF) | SHA-256 | 256-bit | Key derivation if needed |

### TR-002: Envelope Encryption with AWS KMS

**Data Encryption Keys (DEKs) wrapped by AWS KMS CMK.**

**Requirements:**
1. Generate DEKs via `kms:GenerateDataKey` (AES_256)
2. Store encrypted DEK alongside ciphertext
3. Cache plaintext DEKs in memory (max 5 min TTL)
4. Support multi-region KMS keys for DR scenarios
5. Encryption context binding (tenant_id, data_class)
6. DEK rotation: new DEK per encryption (or per batch)

**DEK Lifecycle:**
```
Generate --> Cache (5 min) --> Use for Encryption --> Rotate (on cache expiry or explicit request)
```

### TR-003: Field-Level Encryption (Database)

**Transparent column encryption for PII in PostgreSQL.**

**Requirements:**
1. Use pgcrypto `pgp_sym_encrypt`/`pgp_sym_decrypt` with AES-256
2. Python ORM decorator/type for automatic encrypt/decrypt
3. Support for searchable encryption via HMAC index columns
4. Migration path for existing plaintext data
5. Encrypted columns: email, phone, API keys, PII fields
6. Key rotation without downtime (dual-key decryption period)

**Sensitive Columns to Encrypt:**
| Table | Column | Sensitivity |
|-------|--------|-------------|
| `security.refresh_tokens` | `token_hash` | Already hashed, consider encryption |
| `security.password_history` | `password_hash` | Already hashed, low priority |
| `security.login_attempts` | `ip_address` | PII, encrypt |
| `users` (if exists) | `email`, `phone` | PII, encrypt |
| API keys storage | `key_value` | High sensitivity |

### TR-004: KMS Terraform Module

**Centralized AWS KMS key management for all GreenLang services.**

**Requirements:**
1. Single CMK for data encryption (multi-service)
2. Separate CMKs for: database, S3, secrets, application
3. Key policies with least-privilege access
4. Automatic annual key rotation enabled
5. Multi-region replica keys for disaster recovery
6. Key deletion protection (30-day window)
7. CloudWatch logging for key usage

**Key Hierarchy:**
```
greenlang-master-cmk (root)
├── greenlang-database-cmk  (Aurora, RDS)
├── greenlang-storage-cmk   (S3, EFS)
├── greenlang-cache-cmk     (ElastiCache, Redis)
├── greenlang-secrets-cmk   (Secrets Manager)
├── greenlang-eks-cmk       (K8s secrets)
└── greenlang-app-cmk       (Application DEKs)
```

### TR-005: Key Management & Rotation

**Automated key rotation with zero-downtime re-encryption.**

**Requirements:**
1. CMK automatic rotation (AWS KMS, annual)
2. DEK rotation (per-encryption or configurable interval)
3. Application-level re-encryption job for bulk data
4. Dual-key decryption period (old + new key)
5. Key versioning with metadata tracking
6. Rotation audit trail

### TR-006: Encryption Audit & Monitoring

**Comprehensive logging and metrics for encryption operations.**

**Audit Events:**
- `encryption_performed`, `decryption_performed`
- `key_generated`, `key_rotated`, `key_accessed`
- `encryption_failed`, `decryption_failed`
- `key_cache_hit`, `key_cache_miss`

**Prometheus Metrics:**
| Metric | Type | Labels |
|--------|------|--------|
| `gl_encryption_operations_total` | Counter | operation, status, data_class |
| `gl_encryption_duration_seconds` | Histogram | operation |
| `gl_encryption_key_cache_hits` | Counter | key_type |
| `gl_encryption_key_cache_misses` | Counter | key_type |
| `gl_encryption_failures_total` | Counter | error_type |
| `gl_kms_calls_total` | Counter | operation, status |
| `gl_kms_latency_seconds` | Histogram | operation |

### TR-007: REST API (Admin)

**Management endpoints for encryption operations.**

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| `POST` | `/api/v1/encryption/encrypt` | Encrypt data | `encryption:encrypt` |
| `POST` | `/api/v1/encryption/decrypt` | Decrypt data | `encryption:decrypt` |
| `GET` | `/api/v1/encryption/keys` | List active keys | `encryption:admin` |
| `POST` | `/api/v1/encryption/keys/rotate` | Trigger key rotation | `encryption:admin` |
| `GET` | `/api/v1/encryption/audit` | Encryption audit log | `encryption:audit` |
| `GET` | `/api/v1/encryption/status` | Encryption health check | `encryption:read` |

---

## Integration Points

### Integration with SEC-001 (JWT Authentication)
- Encrypt sensitive JWT metadata stored in database
- Use RBAC permissions for encryption API access

### Integration with SEC-002 (RBAC Authorization)
- New permissions: `encryption:encrypt`, `encryption:decrypt`, `encryption:admin`, `encryption:audit`
- Only authorized services can decrypt sensitive data

### Integration with INFRA-002 (PostgreSQL)
- pgcrypto functions for field-level encryption
- V011 migration for encrypted columns

### Integration with INFRA-003 (Redis)
- Optionally encrypt cached sensitive data in Redis
- DEK cache in Redis for distributed applications

### Integration with INFRA-004 (S3)
- Verify all buckets use SSE-KMS with centralized CMK
- Update Terraform to use centralized KMS module

### Integration with INFRA-010 (Agent Factory)
- Encrypt agent credentials and secrets
- Secure pack artifact encryption

---

## Acceptance Criteria

1. All data at rest encrypted with AES-256 (database, cache, storage)
2. Envelope encryption using AWS KMS for application-level encryption
3. Field-level encryption for all PII columns in PostgreSQL
4. Centralized KMS Terraform module used by all services
5. Automatic CMK rotation enabled (annual)
6. DEK cache improves performance (< 5ms overhead for cached operations)
7. Key rotation with zero-downtime (dual-key decryption)
8. All encryption operations logged for audit
9. Prometheus metrics for encryption monitoring
10. CloudWatch alarms for KMS access anomalies
11. 85%+ test coverage on encryption code
12. SOC 2 Type II encryption requirements met
13. ISO 27001 A.10 cryptographic controls satisfied
14. No plaintext keys in logs, error messages, or exceptions

---

## Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| SEC-001: JWT Authentication | COMPLETE | RBAC permissions for encryption |
| SEC-002: RBAC Authorization | COMPLETE | Permission enforcement |
| INFRA-002: PostgreSQL | COMPLETE | pgcrypto extension enabled |
| INFRA-003: Redis | COMPLETE | At-rest encryption configured |
| INFRA-004: S3 | COMPLETE | SSE-KMS configured |
| AWS KMS | Available | Cloud provider requirement |
| greenlang/governance/security/kms/ | EXISTS | Extend for data encryption |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| KMS latency impacts performance | Medium | Medium | DEK caching, batch encryption |
| Key rotation causes downtime | Low | High | Dual-key decryption period |
| DEK cache compromise | Low | Critical | Short TTL, secure memory, no disk persistence |
| Incorrect encryption context | Medium | Medium | Strict context validation, AAD binding |
| pgcrypto performance on large tables | Medium | Medium | Index encrypted columns with HMAC |
| Key leakage in logs | Low | Critical | Strict logging sanitization |

---

## Development Tasks (Ralphy-Compatible)

### Phase 1: Core Encryption Service (P0)
- [ ] Create `greenlang/infrastructure/encryption_service/__init__.py` - Public API, config
- [ ] Create `greenlang/infrastructure/encryption_service/encryption_service.py` - AES-256-GCM core
- [ ] Create `greenlang/infrastructure/encryption_service/envelope_encryption.py` - KMS DEK wrapping
- [ ] Create `greenlang/infrastructure/encryption_service/key_management.py` - DEK cache, rotation

### Phase 2: Field-Level Encryption (P0)
- [ ] Create `greenlang/infrastructure/encryption_service/field_encryption.py` - DB column encryption
- [ ] Create `deployment/database/migrations/sql/V011__field_level_encryption.sql` - pgcrypto setup

### Phase 3: Audit, Metrics & API (P1)
- [ ] Create `greenlang/infrastructure/encryption_service/encryption_audit.py` - Audit logging
- [ ] Create `greenlang/infrastructure/encryption_service/encryption_metrics.py` - Prometheus metrics
- [ ] Create `greenlang/infrastructure/encryption_service/api/__init__.py` - Router exports
- [ ] Create `greenlang/infrastructure/encryption_service/api/encryption_routes.py` - Admin endpoints

### Phase 4: KMS Terraform Module (P1)
- [ ] Create `deployment/terraform/modules/kms/main.tf` - CMK definitions
- [ ] Create `deployment/terraform/modules/kms/variables.tf` - Input variables
- [ ] Create `deployment/terraform/modules/kms/outputs.tf` - Key ARNs
- [ ] Create `deployment/terraform/modules/kms/policies.tf` - Key policies
- [ ] Create `deployment/terraform/modules/kms/README.md` - Documentation

### Phase 5: Integration (P1)
- [ ] Modify `greenlang/infrastructure/auth_service/auth_setup.py` - Include encryption router
- [ ] Update `greenlang/infrastructure/auth_service/route_protector.py` - Add encryption permissions

### Phase 6: Monitoring (P2)
- [ ] Create `deployment/monitoring/dashboards/encryption-service.json` - Grafana dashboard
- [ ] Create `deployment/monitoring/alerts/encryption-service-alerts.yaml` - Prometheus alerts

### Phase 7: Testing (P2)
- [ ] Create `tests/unit/encryption_service/__init__.py`
- [ ] Create `tests/unit/encryption_service/test_encryption_service.py` - 30+ tests
- [ ] Create `tests/unit/encryption_service/test_envelope_encryption.py` - 25+ tests
- [ ] Create `tests/unit/encryption_service/test_field_encryption.py` - 20+ tests
- [ ] Create `tests/unit/encryption_service/test_key_management.py` - 20+ tests
- [ ] Create `tests/integration/encryption_service/__init__.py`
- [ ] Create `tests/integration/encryption_service/test_encryption_e2e.py` - 15+ tests
- [ ] Create `tests/integration/encryption_service/test_kms_integration.py` - 10+ tests
- [ ] Create `tests/compliance/__init__.py`
- [ ] Create `tests/compliance/test_encryption_compliance.py` - 10+ compliance tests

---

## Compliance Mapping

| Requirement | Standard | Implementation |
|-------------|----------|----------------|
| Data encrypted at rest | SOC 2 CC6.7 | AES-256-GCM, KMS-wrapped DEKs |
| Key management | SOC 2 CC6.6 | AWS KMS CMK with rotation |
| Encryption audit | SOC 2 CC7.2 | Encryption audit log |
| Cryptographic controls | ISO 27001 A.10.1 | AES-256, secure key storage |
| Key lifecycle | ISO 27001 A.10.1.2 | Automated rotation, versioning |
| Data classification | GDPR Art. 32 | Field-level encryption for PII |

---

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Encryption (cached DEK) | < 1ms | AES-256-GCM is fast |
| Decryption (cached DEK) | < 1ms | Symmetric decryption |
| KMS GenerateDataKey | < 50ms | Network latency |
| KMS Decrypt (DEK) | < 30ms | Network latency |
| DEK cache hit rate | > 95% | 5-min TTL, bounded cache |
| Field encryption overhead | < 5% | Per-query impact |
