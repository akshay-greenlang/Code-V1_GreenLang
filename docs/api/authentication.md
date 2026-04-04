# GreenLang API Authentication Guide

## Overview

GreenLang supports two authentication methods for API access:

1. **JWT Bearer Tokens** -- For interactive user sessions and OAuth2 flows.
2. **API Keys** -- For programmatic, service-to-service access.

Both methods are enforced by `AuthenticationMiddleware`, which inspects every
incoming request (except explicitly excluded paths such as `/health`, `/docs`,
and `/openapi.json`).  JWT is checked first; if no valid JWT is present the
middleware falls back to API key validation.

---

## 1. JWT Authentication (RS256)

GreenLang uses **RS256** (RSA with SHA-256) asymmetric signing.  The platform
signs tokens with a private key and validates them with the corresponding
public key.  This allows any service holding the public key to verify tokens
without access to the signing secret.

### Token Format

Every JWT issued by GreenLang contains the following claims.

**Standard Claims (RFC 7519):**

| Claim | Type   | Description |
|-------|--------|-------------|
| `sub` | string | Subject -- the `user_id` of the authenticated principal |
| `iss` | string | Issuer -- defaults to `"greenlang"` |
| `aud` | string | Audience -- defaults to `"greenlang-api"` |
| `exp` | int    | Expiration time (Unix timestamp) |
| `iat` | int    | Issued-at time (Unix timestamp) |
| `nbf` | int    | Not-before time (Unix timestamp) |
| `jti` | string | Unique JWT ID for revocation support |

**GreenLang Custom Claims:**

| Claim         | Type     | Description |
|---------------|----------|-------------|
| `tenant_id`   | string   | Tenant identifier (required for multi-tenancy) |
| `roles`       | string[] | Assigned RBAC roles (e.g., `["developer", "viewer"]`) |
| `permissions` | string[] | Explicit permission strings (e.g., `["cbam:report:export"]`) |
| `org_id`      | string   | Organization identifier (optional) |
| `email`       | string   | User email (optional) |
| `name`        | string   | Display name (optional) |
| `token_type`  | string   | `"access"` (default) |
| `scope`       | string   | OAuth2 scope (optional) |

### Token Lifetime

| Setting | Default | Environment Variable |
|---------|---------|----------------------|
| Access token expiry | 3600 seconds (1 hour) | `GL_JWT_EXPIRY_SECONDS` |
| Issuer | `greenlang` | `GL_JWT_ISSUER` |
| Audience | `greenlang-api` | `GL_JWT_AUDIENCE` |

### Obtaining a Token

Authenticate with username and password to receive a bearer token:

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "analyst@acme.com",
  "password": "********",
  "tenant_id": "tenant_abc123"
}
```

**Response (200 OK):**

```json
{
  "token_id": "tok_8f3a2b1c",
  "token_value": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNhYmYxMjM0In0...",
  "token_type": "bearer",
  "expires_at": "2026-04-04T13:00:00Z"
}
```

### Using a Token

Include the token in the `Authorization` header with the `Bearer` scheme:

```http
GET /api/v1/apps/cbam/run
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

The middleware parses the header as follows:

```
Authorization: Bearer <token>
```

1. Split on whitespace -- must yield exactly two parts.
2. First part must be `bearer` (case-insensitive).
3. Second part is the JWT token string.

### Token Validation

When a token arrives the platform performs these checks in order:

1. **Decode and verify signature** using the RS256 public key.
2. **Verify required claims** -- `sub`, `iss`, `aud`, `exp`, `iat`, `jti`, `tenant_id` must all be present.
3. **Verify issuer** -- must match configured issuer (`GL_JWT_ISSUER`).
4. **Verify audience** -- must match configured audience (`GL_JWT_AUDIENCE`).
5. **Check expiration** -- `exp` must be in the future.
6. **Check revocation** -- `jti` must not appear in the revoked-token set.

If any check fails, the request is rejected with HTTP 401.

### Token Refresh

Refresh a valid (non-expired) token to obtain a new token with a fresh expiry:

```http
POST /api/v1/auth/refresh
Authorization: Bearer <current_token>
```

**Response (200 OK):**

```json
{
  "token_value": "eyJhbGciOiJSUzI1NiIs...<new_token>...",
  "token_type": "bearer",
  "expires_at": "2026-04-04T14:00:00Z"
}
```

The refresh flow:

1. Validate the current token.
2. Revoke the current token by adding its `jti` to the blacklist.
3. Issue a new token with the same claims but a new `jti` and `exp`.

### Token Revocation

Revoke a token before its natural expiry:

```http
POST /api/v1/auth/revoke
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "jti": "550e8400-e29b-41d4-a716-446655440000"
}
```

Revoked tokens are stored in a JTI blacklist (Redis-backed in production).

### JWKS Endpoint

Public keys are distributed via the standard JWKS endpoint:

```
GET /.well-known/jwks.json
```

**Response:**

```json
{
  "keys": [
    {
      "kty": "RSA",
      "use": "sig",
      "alg": "RS256",
      "kid": "3abf1234",
      "n": "0vx7agoebGc...",
      "e": "AQAB"
    }
  ]
}
```

### Key Management

| Operation | Details |
|-----------|---------|
| Key size | 2048-bit RSA |
| Key format | PEM (PKCS8 for private, SubjectPublicKeyInfo for public) |
| Key rotation | Supported via `kid` header in JWT |
| Storage | Private key at `GL_JWT_PRIVATE_KEY_PATH`, public key at `GL_JWT_PUBLIC_KEY_PATH` |
| Permissions | Private key file must be `0600` on Unix |

---

## 2. API Key Authentication

API keys provide long-lived credentials for programmatic access (CI/CD
pipelines, ERP integrations, service accounts).

### Key Format

API keys follow the format:

```
glk_<hex_id>_<urlsafe_secret>
```

- **Prefix:** `glk_` identifies the string as a GreenLang API key.
- **ID segment:** 16 hex characters identifying the key.
- **Secret segment:** 32 bytes of URL-safe random data.

Example: `glk_a1b2c3d4e5f6g7h8_TkF3ZmVyZXdlcnRzZGZhc2Rma2xqc2RmYQ`

### Key Storage

The plaintext key is shown to the user exactly once at creation time.
GreenLang stores only the **SHA-256 hash** of the full key.  If a key is
lost, it cannot be recovered -- a new key must be generated.

### Creating an API Key

```http
POST /api/v1/auth/api-keys
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "name": "CI Pipeline Key",
  "scopes": ["read", "agent:execute"],
  "expires_in_days": 90,
  "allowed_ips": ["10.0.0.0/8"]
}
```

**Response (201 Created):**

```json
{
  "key_id": "glk_a1b2c3d4e5f6g7h8",
  "key_secret": "glk_a1b2c3d4e5f6g7h8_TkF3ZmVyZXdlcnRzZGZhc2Rma2xqc2RmYQ",
  "name": "CI Pipeline Key",
  "scopes": ["read", "agent:execute"],
  "expires_at": "2026-07-03T12:00:00Z",
  "rate_limit": 1000
}
```

**Important:** The `key_secret` value is the full API key.  Store it securely
-- it will not be returned again.

### Using an API Key

Include the key in the `X-API-Key` header:

```http
GET /api/v1/apps/cbam/run
X-API-Key: glk_a1b2c3d4e5f6g7h8_TkF3ZmVyZXdlcnRzZGZhc2Rma2xqc2RmYQ
```

### API Key Validation

Validation performs these checks:

1. **Format check** -- key must start with `glk_`.
2. **Hash lookup** -- SHA-256 hash of the provided key is matched against stored hashes.
3. **Revocation check** -- key must not be revoked.
4. **Active check** -- key must be active.
5. **Expiration check** -- key must not be expired.
6. **IP allowlist** -- client IP must be in the allowlist (if configured).
7. **Origin allowlist** -- request origin must match (if configured).
8. **Scope check** -- key must have the required scopes for the requested operation.
9. **Rate limit check** -- key must be within its per-key rate limit.

### API Key Scopes

| Scope | Description |
|-------|-------------|
| `read` | Read access to resources |
| `write` | Write access to resources |
| `admin` | Full administrative access (implies all scopes) |
| `agent:execute` | Execute agents and workflows |
| `agent:create` | Create new agents |
| `agent:delete` | Delete agents |
| `tools:read` | Read tool configurations |
| `tools:write` | Modify tool configurations |
| `registry:publish` | Publish agents to the registry |

### Key Rotation

Rotate an API key with a grace period during which both old and new keys are
accepted:

```http
POST /api/v1/auth/api-keys/{key_id}/rotate
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "grace_period_hours": 24
}
```

**Response:**

```json
{
  "key_id": "glk_a1b2c3d4e5f6g7h8",
  "new_key_secret": "glk_a1b2c3d4e5f6g7h8_<new_secret>",
  "grace_period_expires_at": "2026-04-05T12:00:00Z"
}
```

### Key Limits

| Limit | Default |
|-------|---------|
| Max keys per user | 5 |
| Default expiry | 90 days |
| Default rate limit | 1000 requests/hour |
| Rate limit window | 3600 seconds |

---

## 3. Multi-Tenancy

All authenticated requests carry a `tenant_id`.  Tenant isolation is enforced
at every layer:

- JWT tokens include `tenant_id` as a required claim.
- API keys are scoped to a tenant at creation time.
- The `TenantContextMiddleware` extracts tenant context from (in priority order):
  1. JWT `tenant_id` claim
  2. `X-Tenant-ID` header
  3. `tenant_id` query parameter

Cross-tenant access is denied unless the principal has the `super_admin` role.

---

## 4. Role-Based Access Control (RBAC)

GreenLang ships with six default roles:

| Role | Description | Key Permissions |
|------|-------------|-----------------|
| `super_admin` | Full platform access | `*:*` |
| `admin` | Tenant-scoped administration | `*:*` within tenant, user management |
| `developer` | Pipeline and agent development | pipeline/pack/dataset `*`, agent `execute` |
| `operator` | Execution and monitoring | pipeline `execute`/`read`, pack/dataset `read` |
| `viewer` | Read-only access | `*:read`, `*:list` |
| `auditor` | Compliance and audit access | `*:read`, audit `*`, compliance `*` |

### Per-Endpoint Authorization

Endpoints enforce authorization using decorators from `greenlang.auth.middleware`:

```python
from greenlang.auth.middleware import require_roles, require_permissions

@app.get("/admin/users")
@require_roles("admin", "super_admin")
async def list_users(request: Request):
    ...

@app.post("/agents")
@require_permissions("agent:create")
async def create_agent(request: Request):
    ...
```

Permission strings follow the format `resource:action` (e.g., `pipeline:execute`,
`cbam:report:export`).  Wildcard matching is supported (`*`).

---

## 5. Code Examples

### Python -- JWT Authentication

```python
import requests

# Authenticate
auth_response = requests.post(
    "https://api.greenlang.io/api/v1/auth/token",
    json={
        "username": "analyst@acme.com",
        "password": "s3cur3p@ss",
        "tenant_id": "tenant_abc123"
    }
)
token = auth_response.json()["token_value"]

# Use the token
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "https://api.greenlang.io/api/v1/apps/cbam/run",
    headers=headers,
    files={"input_file": ("imports.csv", open("imports.csv", "rb"), "text/csv")}
)
print(response.json())
```

### Python -- API Key Authentication

```python
import requests

API_KEY = "glk_a1b2c3d4e5f6g7h8_TkF3ZmVyZXdlcnRzZGZhc2Rma2xqc2RmYQ"

response = requests.get(
    "https://api.greenlang.io/api/v1/runs",
    headers={"X-API-Key": API_KEY}
)
print(response.json())
```

### cURL -- JWT Authentication

```bash
# Obtain token
TOKEN=$(curl -s -X POST https://api.greenlang.io/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"analyst@acme.com","password":"s3cur3p@ss","tenant_id":"tenant_abc123"}' \
  | jq -r '.token_value')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  https://api.greenlang.io/api/v1/runs
```

### cURL -- API Key Authentication

```bash
curl -H "X-API-Key: glk_a1b2c3d4e5f6g7h8_TkF3ZmVyZXdlcnRzZGZhc2Rma2xqc2RmYQ" \
  https://api.greenlang.io/api/v1/runs
```

---

## 6. Error Responses

| Status | Condition | Response Body |
|--------|-----------|---------------|
| 401 | Missing or invalid token/key | `{"detail": "Not authenticated"}` with `WWW-Authenticate: Bearer` header |
| 401 | Expired token | `{"error": {"code": "GL_SECURITY_TOKEN_EXPIRED_ERROR", "message": "Token has expired"}}` |
| 403 | Insufficient role | `{"detail": "Required roles: ('admin', 'super_admin')"}` |
| 403 | Insufficient permission | `{"detail": "Required permissions: ('agent:create',)"}` |
| 403 | Cross-tenant access | `{"detail": "Access denied to this tenant"}` |

---

## 7. Security Compliance

| Standard | Control | Implementation |
|----------|---------|----------------|
| SOC 2 | CC6.1 Logical Access | JWT + RBAC + API Key authentication |
| ISO 27001 | A.9.4 System and Application Access Control | Middleware-enforced auth on all endpoints |
| ISO 27001 | A.10.1 Cryptographic Controls | RS256 asymmetric signing, SHA-256 key hashing |

---

## Source Files

| File | Purpose |
|------|---------|
| `greenlang/auth/jwt_handler.py` | JWT generation, validation, refresh, revocation, JWKS |
| `greenlang/auth/api_key_manager.py` | API key generation, hashing, validation, rotation |
| `greenlang/auth/auth.py` | AuthManager, AuthToken, APIKey, ServiceAccount |
| `greenlang/auth/middleware.py` | FastAPI middleware, dependency injection, decorators |
| `greenlang/auth/rbac.py` | Role/Permission definitions, RBACManager |
| `greenlang/auth/roles.py` | Role definitions and mappings |
| `greenlang/auth/permissions.py` | Permission registry (2400+ entries for all agents) |
