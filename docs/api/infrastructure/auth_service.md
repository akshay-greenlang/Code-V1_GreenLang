# Authentication Service API Reference (SEC-001)

## Overview

The Authentication Service provides JWT-based authentication, token lifecycle management, MFA enrollment, and user self-service operations. It implements OAuth2 `client_credentials` grants for service-to-service communication and username/password authentication with optional TOTP MFA for interactive sessions.

**Router Prefix:** `/auth`
**Tags:** `authentication`, `user`
**Source:** `greenlang/infrastructure/auth_service/api/auth_routes.py`, `user_routes.py`

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/auth/login` | Authenticate with credentials + optional MFA | No |
| POST | `/auth/token` | Issue token via client_credentials grant | No |
| POST | `/auth/refresh` | Refresh access token | No (refresh token required) |
| POST | `/auth/revoke` | Revoke an access or refresh token | No |
| POST | `/auth/logout` | Logout and revoke all session tokens | Yes |
| GET | `/auth/validate` | Validate token and return claims | No (token in header) |
| GET | `/auth/me` | Get current authenticated user profile | Yes |
| GET | `/auth/jwks` | Public JWKS endpoint | No |
| POST | `/auth/password/change` | Change own password | Yes |
| POST | `/auth/password/reset` | Request password reset link | No |
| POST | `/auth/mfa/setup` | Initiate MFA TOTP enrollment | Yes |
| POST | `/auth/mfa/verify` | Verify MFA code to complete enrollment | Yes |

---

## Endpoints

### POST /auth/login

Authenticate with username/password and optional MFA code. Returns a JWT access token and an opaque refresh token on success.

**Request Body:**

```json
{
  "username": "john.doe@acme.com",
  "password": "s3cureP@ss",
  "mfa_code": "123456",
  "tenant_id": "t-acme-corp"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `username` | string | Yes | Username (1-256 chars) |
| `password` | string | Yes | Password |
| `mfa_code` | string | No | 6-digit TOTP MFA code |
| `tenant_id` | string | No | Tenant scope |

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "",
  "tenant_id": "t-acme-corp"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `access_token` | string | JWT access token |
| `refresh_token` | string | Opaque refresh token |
| `token_type` | string | Always `"Bearer"` |
| `expires_in` | integer | Token lifetime in seconds |
| `scope` | string | Granted scopes (space-separated) |
| `tenant_id` | string | Tenant context |

**Error Responses:**

| Status | Description |
|--------|-------------|
| 401 | Invalid credentials |
| 429 | Account locked due to repeated failures (>10 attempts) |
| 503 | Auth service unavailable |

---

### POST /auth/token

Issue a token using the `client_credentials` OAuth2 grant. Intended for service-to-service authentication. No refresh token is returned.

**Request Body:**

```json
{
  "grant_type": "client_credentials",
  "client_id": "svc-carbon-calc",
  "client_secret": "secret-value",
  "scope": "agents:read emissions:write",
  "tenant_id": "system"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `grant_type` | string | Yes | Must be `"client_credentials"` |
| `client_id` | string | Yes | Service account client ID |
| `client_secret` | string | Yes | Service account secret |
| `scope` | string | No | Requested scopes (space-separated) |
| `tenant_id` | string | No | Tenant scope (defaults to `"system"`) |

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "refresh_token": "",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "agents:read emissions:write",
  "tenant_id": "system"
}
```

---

### POST /auth/refresh

Exchange a valid refresh token for a new access + refresh token pair. Uses token rotation: the submitted refresh token is invalidated and a new one is returned.

**Request Body:**

```json
{
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."
}
```

**Response (200 OK):** Same as `/auth/login` response.

**Error Responses:**

| Status | Description |
|--------|-------------|
| 401 | Invalid, expired, or already-used refresh token |
| 503 | Auth service unavailable |

---

### POST /auth/revoke

Revoke an access token (by JTI) or a refresh token. Per RFC 7009, this endpoint always returns 200 even if the token is already revoked or unknown.

**Request Body:**

```json
{
  "token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type_hint": "access_token"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `token` | string | Yes | The token to revoke |
| `token_type_hint` | string | No | `"access_token"` or `"refresh_token"` |

**Response (200 OK):**

```json
{
  "status": "revoked"
}
```

---

### POST /auth/logout

Logout: revoke the current session's access and refresh tokens. Revokes ALL tokens for the user.

**Headers:** `Authorization: Bearer {access_token}` (required)

**Request Body (optional):**

```json
{
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."
}
```

**Response (200 OK):**

```json
{
  "status": "logged_out"
}
```

---

### GET /auth/validate

Validate the `Authorization: Bearer` token and return its claims. Returns `{"valid": false}` instead of raising 401, making it suitable as an introspection endpoint for API gateways.

**Headers:** `Authorization: Bearer {access_token}`

**Response (200 OK):**

```json
{
  "valid": true,
  "user_id": "u-analyst-01",
  "tenant_id": "t-acme-corp",
  "roles": ["emissions_analyst", "viewer"],
  "permissions": ["emissions:read", "reports:read"],
  "scopes": ["agents:read"],
  "expires_at": "2026-04-04T12:30:00+00:00"
}
```

---

### GET /auth/me

Return the profile of the currently authenticated user. Requires a valid Bearer token.

**Headers:** `Authorization: Bearer {access_token}` (required)

**Response (200 OK):**

```json
{
  "user_id": "u-analyst-01",
  "tenant_id": "t-acme-corp",
  "roles": ["emissions_analyst"],
  "permissions": ["emissions:read", "reports:read"],
  "scopes": [],
  "email": "analyst@acme.com",
  "name": "Jane Analyst"
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 401 | Missing or invalid bearer token |
| 503 | Auth service unavailable |

---

### GET /auth/jwks

Return the JSON Web Key Set for public-key signature verification. This endpoint is unauthenticated and cacheable. Clients and API gateways use it to verify JWT signatures without sharing private keys.

**Response (200 OK):**

```json
{
  "keys": [
    {
      "kty": "RSA",
      "kid": "greenlang-auth-2026",
      "use": "sig",
      "alg": "RS256",
      "n": "0vx7agoebGcQSuu...",
      "e": "AQAB"
    }
  ]
}
```

---

### POST /auth/password/change

Change the authenticated user's password. Requires the current password for verification. On success, all existing sessions and tokens for the user are revoked, forcing re-authentication.

**Headers:** `Authorization: Bearer {access_token}` (required)

**Request Body:**

```json
{
  "current_password": "oldP@ss123",
  "new_password": "newS3cure!Pass"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `current_password` | string | Yes | Current password |
| `new_password` | string | Yes | New password (8-128 chars) |

**Response (200 OK):**

```json
{
  "status": "password_changed",
  "message": "Password updated successfully. All sessions have been revoked."
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 401 | Current password is incorrect |
| 404 | User not found |

---

### POST /auth/password/reset

Request a password-reset link. This endpoint is unauthenticated. To prevent user enumeration, it always returns 200 regardless of whether the email address exists.

**Request Body:**

```json
{
  "email": "user@acme.com",
  "tenant_id": "t-acme-corp"
}
```

**Response (200 OK):**

```json
{
  "status": "reset_requested",
  "message": "If an account with that email exists, a password reset link has been sent."
}
```

---

### POST /auth/mfa/setup

Initiate MFA enrollment for the authenticated user. Returns a provisioning URI and QR code for TOTP authenticator apps.

**Headers:** `Authorization: Bearer {access_token}` (required)

**Request Body:**

```json
{
  "method": "totp",
  "device_name": "Work Phone"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | No | MFA method: `"totp"` or `"sms"` (default: `"totp"`) |
| `device_name` | string | No | Device label (max 64 chars) |
| `phone_number` | string | No | Phone number for SMS method |

**Response (200 OK):**

```json
{
  "device_id": "dev-abc123",
  "method": "totp",
  "provisioning_uri": "otpauth://totp/GreenLang:user@acme.com?secret=JBSWY3DPEHPK3PXP&issuer=GreenLang",
  "qr_code_base64": "iVBORw0KGgo...",
  "message": "Scan the QR code with your authenticator app, then verify with a code."
}
```

---

### POST /auth/mfa/verify

Verify an MFA code to complete enrollment or step-up authentication. On first successful verification, backup codes are generated and returned.

**Headers:** `Authorization: Bearer {access_token}` (required)

**Request Body:**

```json
{
  "device_id": "dev-abc123",
  "code": "482951"
}
```

**Response (200 OK):**

```json
{
  "verified": true,
  "backup_codes": ["ABC12345", "DEF67890", "GHI11223"],
  "message": "MFA enabled successfully. Save your backup codes securely."
}
```

---

## Audit Events

All authentication endpoints emit structured audit events to the Loki pipeline:

- `login_success` / `login_failure`
- `logout`
- `token_created` / `token_revoked`
- `password_changed` / `password_reset_requested`
- `mfa_setup_initiated` / `mfa_verified` / `mfa_verify_failed`
