---
title: "SDK: Python"
description: greenlang-factors -- the typed Python client for the GreenLang Factors API.
---

# Python SDK -- `greenlang-factors`

```sh
pip install greenlang-factors==1.0.0
```

Requires Python 3.10+.

## Imports

```python
from greenlang_factors import (
    FactorsClient,
    AsyncFactorsClient,
    APIKeyAuth,
    JWTAuth,
    # Errors
    FactorsAPIError,
    AuthError,
    TierError,
    LicenseError,
    LicensingGapError,
    EntitlementError,
    FactorNotFoundError,
    ValidationError,
    RateLimitError,
    EditionPinError,
    EditionMismatchError,
    # Receipt verification
    ReceiptVerificationError,
    verify_receipt,
)
```

## Constructor

```python
client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_fac_...",          # or jwt_token=...
    default_edition=None,
    pinned_edition=None,
    verify_greenlang_cert=True,    # TLS cert pinning
    timeout=30.0,
    max_retries=3,
)
```

## Method surface

| Method                                      | Endpoint                                    |
|---------------------------------------------|---------------------------------------------|
| `client.search(query, ...)`                 | `GET /factors/search`                       |
| `client.search_v2(query, ...)`              | `POST /factors/search/v2`                   |
| `client.list_factors(...)`                  | `GET /factors`                              |
| `client.get_factor(factor_id, ...)`         | `GET /factors/{id}`                         |
| `client.match(activity, ...)`               | `POST /factors/match`                       |
| `client.coverage(...)`                      | `GET /factors/coverage`                     |
| `client.resolve(request, ...)`              | `POST /factors/resolve-explain`             |
| `client.resolve_explain(factor_id, ...)`    | `GET /factors/{id}/explain`                 |
| `client.alternates(factor_id, ...)`         | `GET /factors/{id}/alternates`              |
| `client.resolve_batch([...])`               | `POST /factors/resolve/batch`               |
| `client.list_editions(...)`                 | `GET /editions`                             |
| `client.get_edition(edition_id)`            | `GET /editions/{id}/changelog`              |
| `client.diff(factor_id, left, right)`       | `GET /factors/{id}/diff`                    |
| `client.audit_bundle(factor_id, ...)`       | `GET /factors/{id}/audit-bundle`            |
| `client.list_sources(...)`                  | `GET /factors/source-registry`              |
| `client.list_method_packs()`                | `GET /method-packs`                         |
| `client.set_override(override)`             | `POST /factors/overrides`                   |
| `client.list_overrides(...)`                | `GET /factors/overrides`                    |

## Edition pinning

```python
# As a context manager (preferred)
with client.with_edition("2027.Q1-electricity") as scoped:
    resolved = scoped.resolve(request)

# As a fresh client
scoped = client.pin_edition("2027.Q1-electricity")
try:
    resolved = scoped.resolve(request)
finally:
    scoped.close()
```

## Receipt verification

```python
response = client.resolve(request)
summary = client.verify_receipt(response.model_dump(), secret=os.environ["GL_FACTORS_SIGNING_SECRET"])
print(summary["verified"], summary["key_id"])
```

For Ed25519 receipts, the SDK fetches the JWKS from `GL_FACTORS_JWKS_URL` (defaults to `https://api.greenlang.io/.well-known/jwks.json`).

## Async client

```python
import asyncio
from greenlang_factors import AsyncFactorsClient

async def main():
    async with AsyncFactorsClient(base_url="https://api.greenlang.io", api_key="gl_...") as client:
        async with client.with_edition("2027.Q1") as scoped:
            resolved = await scoped.resolve(request)

asyncio.run(main())
```

## CLI

```sh
gl-factors search "diesel"
gl-factors get-factor ef:co2:diesel:us:2026
gl-factors resolve "natural gas" --jurisdiction US
gl-factors verify-receipt response.json
```
