# GreenLang Infrastructure and Security API Reference

## Overview

The GreenLang platform exposes a comprehensive set of REST APIs for infrastructure management, security operations, and platform administration. All APIs use JSON request/response bodies and follow RESTful conventions.

**Base URL:** `https://api.greenlang.io`

**Authentication:** JWT Bearer token (see [Authentication Service](./auth_service.md))

**Common Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes (most endpoints) | `Bearer {access_token}` |
| `X-Tenant-Id` | Recommended | Tenant scope for multi-tenant operations |
| `X-Correlation-ID` | Optional | Distributed tracing correlation ID |
| `X-User-Id` | Set by gateway | Authenticated user ID (injected by API gateway) |
| `Content-Type` | Yes (POST/PUT) | `application/json` |

**Standard Error Response:**

```json
{
  "detail": "Human-readable error message"
}
```

**Pagination:** List endpoints support `page` (1-indexed) and `page_size` (max 100) query parameters. Responses include `total`, `has_next`, and `has_prev` fields.

---

## Service Catalog

### Security Services

| Service | Prefix | Component | Description |
|---------|--------|-----------|-------------|
| [Authentication](./auth_service.md) | `/auth` | SEC-001 | JWT login, token lifecycle, MFA, user self-service |
| [Auth Admin](./auth_admin.md) | `/auth/admin` | SEC-001 | User management, lockouts, session admin |
| [RBAC Authorization](./rbac_service.md) | `/api/v1/rbac` | SEC-002 | Roles, permissions, assignments, authorization checks |
| [Encryption](./encryption_service.md) | `/api/v1/encryption` | SEC-003 | Data encryption/decryption, key management |
| [Audit Logging](./audit_service.md) | `/api/v1/audit` | SEC-005 | Event logging, search, statistics, compliance reports, export |
| [Secrets Management](./secrets_service.md) | `/api/v1/secrets` | SEC-006 | Vault-backed secret CRUD, rotation, health |
| [Security Scanning](./security_scanning.md) | `/api/v1/security` | SEC-007 | Scan execution, vulnerability management, dashboard |
| [PII Detection](./pii_service.md) | `/api/v1/pii` | SEC-011 | PII detection, redaction, tokenization, quarantine |

### Infrastructure Services

| Service | Prefix | Component | Description |
|---------|--------|-----------|-------------|
| [Agent Factory](./agent_factory.md) | `/api/v1/factory` | INFRA-010 | Agent CRUD, execution, lifecycle, hub, queue |
| [Feature Flags](./feature_flags.md) | `/api/v1/flags` | INFRA-008 | Flag CRUD, evaluation, rollout, kill switch, overrides |

### Security Operations Services

| Service | Prefix | Component | Description |
|---------|--------|-----------|-------------|
| [SOC 2 Preparation](./soc2_preparation.md) | `/api/v1/soc2` | SEC-009 | Projects, assessments, evidence, findings, attestations |
| [Incident Response](./incident_response.md) | `/api/v1/secops/incidents` | SEC-010 | Incident lifecycle, playbooks, metrics |
| [WAF Management](./waf_management.md) | `/api/v1/secops/waf` | SEC-010 | WAF rules, attack detection, DDoS shield |
| [Threat Modeling](./threat_modeling.md) | `/api/v1/secops/threats` | SEC-010 | STRIDE analysis, DFD, risk scoring, mitigations |
| [Vulnerability Disclosure](./vulnerability_disclosure.md) | (no prefix) | SEC-010 | VDP submissions, triage, bounties, hall of fame |
| [Security Training](./security_training.md) | `/api/v1/secops` | SEC-010 | Courses, assessments, phishing campaigns, scoring |
| [Compliance Automation](./compliance_automation.md) | (no prefix) | SEC-010 | ISO 27001, GDPR, PCI-DSS, DSAR, consent |

### Foundation Agent Services

| Service | Prefix | Component | Description |
|---------|--------|-----------|-------------|
| [Orchestrator](./orchestrator.md) | `/pipelines`, `/runs`, `/approvals` | AGENT-FOUND-001 | Pipeline management, run operations, approvals |
| [Schema Validator](./schema_validator.md) | `/v1/schema` | AGENT-FOUND-002 | Schema validation, compilation, registry |

---

## Authentication Flow

All protected endpoints require a valid JWT access token. The recommended flow:

1. **Obtain token** via `POST /auth/login` (interactive) or `POST /auth/token` (service-to-service).
2. **Include token** in the `Authorization: Bearer {token}` header.
3. **Refresh token** via `POST /auth/refresh` before expiry.
4. **Validate token** via `GET /auth/validate` (introspection).

```
Client                    Auth Service              Protected API
  |                           |                          |
  |--- POST /auth/login ----->|                          |
  |<-- TokenResponse ---------|                          |
  |                           |                          |
  |--- GET /api/v1/... -------|------------------------->|
  |   Authorization: Bearer   |   (gateway validates)    |
  |<-- Response --------------|--------------------------|
```

---

## Rate Limiting

All endpoints are rate-limited:

- **Authenticated requests:** 100 requests/minute per user
- **Unauthenticated requests:** 10 requests/minute per IP
- **Admin endpoints:** 50 requests/minute per admin user

When rate limit is exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header.

---

## Versioning

All infrastructure APIs use path-based versioning (`/api/v1/...`). The authentication endpoints use the `/auth` prefix without explicit versioning for OAuth2 compatibility.
