# Authentication

The SDK supports three authentication paths, all validated server-side by `greenlang.factors.api_auth`.

## Decision tree

1. **Production workloads** -> use `JWTAuth` (tokens are short-lived, tenant/tier/roles are embedded in the claim set).
2. **CI/CLI/service accounts** -> use `APIKeyAuth` (long-lived keys issued per tenant, revocable).
3. **Pro+ tiers requiring tamper-evident calls** -> wrap either of the above in `HMACAuth` for request signing.

## 1. API Key

```python
from greenlang.factors.sdk.python import FactorsClient, APIKeyAuth

client = FactorsClient(base_url="https://api.greenlang.io", api_key="gl_fac_...")
# or equivalently:
client = FactorsClient(base_url="...", auth=APIKeyAuth(api_key="gl_fac_..."))
```

Every request carries `X-API-Key: gl_fac_...`. Keys are hashed (SHA-256, constant-time compared) server-side.

### Where to store the key

- **Do not hard-code.** Use `os.environ["GREENLANG_FACTORS_API_KEY"]` or your secrets manager (Vault, AWS Secrets Manager, GCP Secret Manager).
- The SDK will not log your key; `APIKeyAuth.__repr__` only shows the class name.
- The CLI reads `GREENLANG_FACTORS_API_KEY` from the environment for the same reason.

## 2. JWT Bearer

```python
from greenlang.factors.sdk.python import FactorsClient, JWTAuth

client = FactorsClient(base_url="...", jwt_token="eyJ...")
# or:
client = FactorsClient(base_url="...", auth=JWTAuth(token="eyJ..."))
```

Every request carries `Authorization: Bearer <jwt>`. Claims extracted server-side:

- `sub`: user id
- `email`
- `tenant_id`
- `tier`: `community | pro | enterprise | internal`
- `roles`, `permissions`

## 3. HMAC Request Signing (Pro+)

For enterprise customers who need tamper-evident requests (e.g. compliance audits) the SDK can sign each request with a tenant-specific shared secret:

```python
from greenlang.factors.sdk.python import APIKeyAuth, FactorsClient, HMACAuth

auth = HMACAuth(
    api_key_id="kid-2026-q1",
    secret="<tenant-shared-secret>",
    primary=APIKeyAuth(api_key="gl_fac_..."),  # still sent alongside
)
client = FactorsClient(base_url="https://api.greenlang.io", auth=auth)
```

### Signature format

```
canonical = method + "\n" + path + "\n" + timestamp + "\n" + nonce + "\n" + sha256_hex(body)
signature = HMAC_SHA256(secret, canonical)
```

Sent as headers:

- `X-GL-Key-Id: kid-2026-q1`
- `X-GL-Timestamp: 1700000000`
- `X-GL-Nonce: <base64url-22>`
- `X-GL-Signature: sha256=<hex>`

The server uses `hmac.compare_digest` for constant-time comparison.

## Tier hierarchy (server-enforced)

The server gates endpoints via `greenlang.factors.api_auth.TIER_ORDER`:

```
community < pro < enterprise < internal
```

| Endpoint | Minimum tier |
|---|---|
| `/search`, `/search/v2`, `/search/facets`, `/coverage` | community |
| `/match`, `/export`, `/diff`, `/explain`, `/resolve-explain`, `/alternates` | pro |
| `/audit-bundle` | enterprise |

A 403 from the server is surfaced as `TierError` (or `LicenseError` when the detail mentions license/connector_only/redistribution).
