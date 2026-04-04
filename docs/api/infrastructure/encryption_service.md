# Encryption Service API Reference (SEC-003)

## Overview

The Encryption Service provides AES-256-GCM encryption at rest, key management, key rotation, and encryption audit logging. All operations are performed server-side with keys managed through the integrated secrets service.

**Router Prefix:** `/api/v1/encryption`
**Tags:** `Encryption`
**Source:** `greenlang/infrastructure/encryption_service/api/encryption_routes.py`

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/encryption/encrypt` | Encrypt data | Yes |
| POST | `/api/v1/encryption/decrypt` | Decrypt data | Yes |
| GET | `/api/v1/encryption/keys` | List encryption keys | Yes |
| POST | `/api/v1/encryption/keys/rotate` | Rotate encryption key | Yes |
| DELETE | `/api/v1/encryption/keys/cache` | Invalidate key cache | Yes |
| GET | `/api/v1/encryption/audit` | Get encryption audit log | Yes |
| GET | `/api/v1/encryption/status` | Service health status | No |

---

## Endpoints

### POST /api/v1/encryption/encrypt

Encrypt data using AES-256-GCM. Supports both text and binary (base64-encoded) payloads.

**Request Body:**

```json
{
  "data": "sensitive-emissions-data",
  "key_id": "key-prod-2026",
  "context": {"purpose": "emissions_report", "tenant_id": "t-acme-corp"}
}
```

**Response (200 OK):**

```json
{
  "ciphertext": "base64-encoded-ciphertext...",
  "key_id": "key-prod-2026",
  "algorithm": "AES-256-GCM",
  "iv": "base64-encoded-iv",
  "tag": "base64-encoded-auth-tag"
}
```

---

### POST /api/v1/encryption/decrypt

Decrypt previously encrypted data. Requires the same key ID and encryption context.

**Request Body:**

```json
{
  "ciphertext": "base64-encoded-ciphertext...",
  "key_id": "key-prod-2026",
  "iv": "base64-encoded-iv",
  "tag": "base64-encoded-auth-tag",
  "context": {"purpose": "emissions_report", "tenant_id": "t-acme-corp"}
}
```

**Response (200 OK):**

```json
{
  "data": "sensitive-emissions-data",
  "key_id": "key-prod-2026"
}
```

---

### POST /api/v1/encryption/keys/rotate

Rotate the active encryption key. Creates a new key version and marks the old key as decrypt-only.

---

### GET /api/v1/encryption/status

Service health check. Returns encryption service availability and key store connectivity.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "key_store": "connected",
  "active_key_id": "key-prod-2026",
  "active_key_version": 3,
  "total_keys": 5
}
```
