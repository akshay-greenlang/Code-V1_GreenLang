# Authentication

The GreenLang Factors API accepts two credential types, in this priority order:

1. **JWT Bearer token** via `Authorization: Bearer <jwt>`.
2. **API key** via `X-API-Key: <key>` (also accepted as `Authorization: Bearer gl_pk_...`).

Each credential is mapped to a tier (`community`, `pro`, `consulting`, `enterprise`, `internal`) which gates endpoint access, rate limits, licensed-source visibility, and signed-receipt algorithm.

**Auth entry points:** `greenlang/integration/api/dependencies.py::get_current_user`, `greenlang/factors/api_auth.py`
**Tier resolution:** `greenlang/factors/tier_enforcement.py::resolve_tier`

---

## JWT Bearer tokens

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Claims we read:

| Claim | Purpose |
|---|---|
| `sub` | User id (used in rate-limit keying). |
| `tenant_id` | Tenant id for tenant-overlay and license resolution. |
| `tier` | Tier string; falls back to account record if absent. |
| `exp` | Expiry (validated). |
| `scopes` | Optional fine-grained scope list (e.g. `factors:read`, `factors:resolve`, `factors:batch`). |

### Obtaining a token

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=$GL_CLIENT_ID&client_secret=$GL_CLIENT_SECRET"
```

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "tier": "pro",
  "tenant_id": "acme-climate"
}
```

Tokens default to 1-hour TTL. Refresh ahead of `expires_in` or handle a 401 with `error: "token_expired"`.

---

## API keys

Long-lived strings prefixed with `gl_pk_`. Useful for CI jobs and machine-to-machine integrations that cannot run an OAuth client-credentials flow.

```
X-API-Key: gl_pk_7f19c84e3820a31d8a6fb22e...
```

```bash
curl -H "X-API-Key: $GL_API_KEY" "$GL_API_BASE/api/v1/factors/coverage"
```

API keys are per-tenant and carry the tenant's default tier. Rotate via dashboard or `POST /api/v1/auth/keys/rotate`. Revocation is immediate.

### Key scoping

API keys can be scoped to a subset of endpoints:

- `factors:read` — list, search, get, coverage.
- `factors:resolve` — resolve-explain, explain, alternates, quality.
- `factors:batch` — batch/*.
- `factors:export` — export + audit-bundle.
- `editions:read` — list + fetch.
- `webhooks:admin` — webhook subscription management.
- `admin:*` — everything (internal only).

Request with insufficient scope &rarr; `403` with `error: "scope_denied"`.

---

## Tier table

| Tier | Resolve / explain | Licensed sources | Preview factors | Export RPS | RPM |
|---|---|---|---|---|---|
| `community` | No (403) | Hidden (visibility clamped) | Hidden | 1 / 15min | 60 |
| `pro` | Yes | Require key (451 if none) | Visible | 5 / 15min | 600 |
| `consulting` | Yes | Require key | Visible | 5 / 15min | 600 |
| `enterprise` | Yes | Require key, `include_connector=true` allowed | Visible, connector-only visible | 20 / 15min | 6000 |
| `internal` | Yes | All visible | All visible | 200 / 15min | 60000 |

See [rate-limits](./rate-limits.md) for window mechanics and [licensing-classes](../concepts/licensing-classes.md) for connector-key flow.

---

## Error bodies (the auth 4xx family)

### 401 Unauthorized

Missing, malformed, or expired credential.

```json
{
  "error": "invalid_credentials",
  "message": "Bearer token is missing, malformed, or expired.",
  "details": {
    "reason": "token_expired",
    "resolution": "Obtain a new token via POST /api/v1/auth/token or refresh your API key."
  }
}
```

Common `details.reason` values:

- `missing_header` — no `Authorization` or `X-API-Key`.
- `malformed_token` — JWT structure invalid or signature mismatch.
- `token_expired` — JWT `exp` past.
- `api_key_revoked` — the key exists but has been revoked.
- `api_key_not_found` — the key prefix does not match any issued key.

### 402 Payment Required

The call would succeed but your billing tier cannot cover it (e.g. a seat-limited batch job).

```json
{
  "error": "billing_limit_exceeded",
  "message": "Your plan's batch-seat limit is reached for the current billing period.",
  "details": {
    "limit_type": "batch_seats",
    "limit": 3,
    "current": 3,
    "resolution": "Complete or cancel an existing batch job, or upgrade at https://dashboard.greenlang.io/billing."
  }
}
```

### 403 Forbidden

You are authenticated, but the endpoint is not available to your tier or scope.

```json
{
  "error": "tier_insufficient",
  "message": "Factor explain endpoints require Pro, Consulting, Enterprise, or Internal tier.",
  "details": {
    "current_tier": "community",
    "required_tiers": ["pro", "consulting", "enterprise", "internal"],
    "resolution": "Upgrade at https://dashboard.greenlang.io/plans."
  }
}
```

Other 403 shapes:

- `scope_denied` — API key lacks the required scope.
- `tenant_mismatch` — credential does not own the target resource (e.g. someone else's batch job).
- `license_class_forbidden` — factor's `license_class` is above your tier's ceiling.

### 451 Unavailable For Legal Reasons

A licensed connector was needed but the tenant does not have an active key for it. See [licensing-classes](../concepts/licensing-classes.md).

```json
{
  "error": "license_required",
  "message": "Resolution routed to licensed source 'ecoinvent' but no active license key is configured for tenant 'acme-climate'.",
  "details": {
    "connector_id": "ecoinvent",
    "license_class": "commercial_connector",
    "resolution": "Configure GL_FACTORS_LICENSE_ECOINVENT env var, or contact sales@greenlang.io for a connector key."
  }
}
```

Full error table: [errors](./errors.md).

---

## Tier auto-clamping

Some parameters are **clamped** to your tier rather than returning 403:

- `include_preview=true` on Community tier → silently clamped to `false`.
- `include_connector=true` on anything below Enterprise → clamped to `false`.

See `enforce_tier_on_request` in `greenlang/factors/tier_enforcement.py`. The response body's `visibility` field records the effective flags so you can detect clamping.

---

## SDK configuration

Python:

```python
from greenlang.factors.sdk.python import FactorsClient
from greenlang.factors.sdk.python.auth import APIKeyAuth, JWTAuth

# API key
client = FactorsClient(base_url="https://api.greenlang.io", api_key="gl_pk_...")

# JWT
client = FactorsClient(base_url="https://api.greenlang.io", jwt_token="eyJh...")

# Explicit AuthProvider (for token refresh flows)
client = FactorsClient(base_url="https://api.greenlang.io", auth=JWTAuth(token="..."))
```

TypeScript:

```ts
import { FactorsClient } from "@greenlang/factors";

// API key
const c = new FactorsClient({ baseUrl: "https://api.greenlang.io", apiKey: "gl_pk_..." });

// JWT
const c2 = new FactorsClient({ baseUrl: "https://api.greenlang.io", jwtToken: "eyJh..." });
```

---

## Security notes

- **Never commit API keys** to source. Use secret managers (GitHub Actions secrets, Vault, 1Password, AWS Secrets Manager).
- **Rotate keys at least quarterly** and immediately on suspected compromise.
- **Use JWT for user-facing products** where token TTL limits blast radius; use API keys for trusted automation only.
- **Bind keys to IP ranges** via dashboard where your egress is stable.
- The signed-receipt mechanism is **separate** from the auth credential — a stolen API key lets an attacker query factors but cannot forge receipts without also stealing the HMAC or Ed25519 signing key.

---

## File citations

| Piece | File |
|---|---|
| JWT validation + API key lookup | `greenlang/factors/api_auth.py` |
| Dependency injector (`get_current_user`) | `greenlang/integration/api/dependencies.py` |
| Tier resolution | `greenlang/factors/tier_enforcement.py::resolve_tier` |
| Tier enforcement + clamp | `greenlang/factors/tier_enforcement.py::enforce_tier_on_request` |
| Entitlements (scope checks) | `greenlang/factors/entitlements.py` |
| License manager (451 path) | `greenlang/factors/connectors/license_manager.py` |
