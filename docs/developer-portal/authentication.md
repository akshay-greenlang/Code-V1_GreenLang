# Authentication

The GreenLang Factors API supports two auth modes. Both resolve to the same tenant context.

| Mode | Header | When to use |
|---|---|---|
| **API key** | `Authorization: Bearer glk_...` | CLI, server-side SDKs, batch jobs |
| **OAuth2 client credentials** | `Authorization: Bearer <access_token>` | Long-running partner integrations, multi-tenant OEMs |

Every request is scoped to a single tenant. Cross-tenant queries return `403 Forbidden` except via the explicit OEM sub-tenant headers described below.

---

## 1. API keys

API keys are minted per tenant at `https://app.greenlang.io/settings/api-keys`. Each key has:

- A `key_id` (visible prefix, e.g. `glk_live_abc123`)
- A scope list (`factors:read`, `factors:resolve`, `factors:admin`)
- An optional edition pin (restricts the key to a specific `edition_id`)
- A rotation date

```bash
curl -H "Authorization: Bearer $GL_API_KEY" \
     "https://api.greenlang.io/v1/factors/resolve"
```

Rotate keys every 90 days. Revocation takes effect within 60 seconds platform-wide.

---

## 2. OAuth2 client credentials

For long-running integrations, exchange a `client_id` + `client_secret` for a short-lived access token:

```bash
curl -X POST "https://api.greenlang.io/oauth2/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=$GL_CLIENT_ID" \
  -d "client_secret=$GL_CLIENT_SECRET" \
  -d "scope=factors:read factors:resolve"
```

Response:

```json
{
  "access_token": "eyJ...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "factors:read factors:resolve"
}
```

Cache the token for `expires_in - 60` seconds.

---

## 3. Tenant context headers

| Header | Purpose |
|---|---|
| `X-GreenLang-Edition` | Pin the response to a specific `edition_id`. Server rejects if drifted. |
| `X-GreenLang-Subtenant` | OEM partners routing on behalf of a sub-tenant. Requires `oem_redistributable` entitlement. |
| `X-Request-Id` | Caller-supplied idempotency / correlation ID. Echoed on every response. |

The server always returns `X-GreenLang-Edition` with the served edition, even when the caller does not pin one. Clients MUST verify this header matches their expectation.

---

## 4. Signed requests (coming v1.1)

For high-security deployments (regulated disclosures under CBAM / CSRD third-party assurance), GreenLang Factors supports optional Ed25519-signed requests. The request body is signed with a customer-owned key; the public key is registered in the dashboard. On every mutation the server verifies the signature and records the key ID in the audit bundle.

Signed requests are **not required** for v1.0. Enable for production in v1.1 once you have a key rotation policy. See `concepts/signed_receipt.md` for the parallel response-side signing.

---

## 5. Rate limits

- 100 req/min per API key (authenticated)
- 1000 req/min per tenant (aggregate)
- Burst bucket: 20 requests

When exceeded, the server returns `429 Too Many Requests` with `Retry-After` in seconds. The SDK honours `Retry-After` automatically. See [`error-codes.md`](error-codes.md#rate-limiting).

---

## 6. Tenant isolation guarantees

- Factors tagged `licensing.redistribution_class == "customer_private"` NEVER appear in cross-tenant responses.
- Signed receipts carry the tenant ID inside the payload hash. A receipt cannot be replayed in another tenant's audit.
- All API access is logged; audit trail retrievable via `/v1/audit/bundle/{period}`.

See [`concepts/license_class.md`](concepts/license_class.md) for the data classes.
