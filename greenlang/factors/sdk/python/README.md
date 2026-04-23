# GreenLang Factors Python SDK (v1.0.0)

Production-grade Python client for the [GreenLang Factors REST API](https://developers.greenlang.ai). Search, resolve, and audit emission factors across the global open + licensed catalog with edition pinning, signed-receipt verification, and rate-limit-aware retries.

```bash
pip install greenlang-factors
```

Requires Python 3.10+.

## What is GreenLang Factors?

GreenLang Factors is the global emission-factor catalog and resolution engine. Every API call returns a fully provenanced factor record (source, edition, license class, uncertainty band) and an optional signed receipt your auditor can verify months later.

## 60-second quickstart

```python
from greenlang_factors import FactorsClient

with FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_fac_your_key_here",
) as client:
    # 1. Search the catalog
    hits = client.search("natural gas US Scope 1", limit=5)
    for f in hits.factors:
        print(f.factor_id, f.co2e_per_unit, f.unit)

    # 2. Resolve an activity into a chosen factor + alternates + assumptions
    resolved = client.resolve({
        "activity": "natural gas combustion",
        "method_profile": "corporate_scope1",
        "jurisdiction": "US",
        "reporting_date": "2026-04-01",
        "quantity": 1000,
        "unit": "therm",
    })
    print("chosen:", resolved.chosen.factor_id)
    print("co2e:", resolved.computed_total)
```

## Edition pinning

Pin every request to a specific catalog edition so reports remain reproducible across catalog updates:

```python
with client.with_edition("2027.Q1-electricity") as scoped:
    resolved = scoped.resolve({
        "activity": "electricity consumption",
        "method_profile": "corporate_scope2_location_based",
        "jurisdiction": "US-CA",
        "quantity": 5_000,
        "unit": "kWh",
    })
    # If the server returns a different edition than the pin, an
    # EditionMismatchError is raised - we never silently accept drift.
```

The accepted edition-id formats are:

* `v1.0.0`, `v1`, `v2.1` -- semantic-version style
* `2027.Q1`, `2027.Q1-electricity` -- quarterly + scope
* `2027-04-01-freight` -- date + scope

Anything else raises `EditionPinError` before the request goes out.

## Offline signed-receipt verification

Every Pro+ response can carry a signed receipt. Verify it offline -- no network call back to GreenLang -- so audit packages remain self-contained.

```python
from greenlang_factors import verify_receipt, ReceiptVerificationError

response = client.resolve(request)  # response carries a receipt block

try:
    summary = client.verify_receipt(
        response.model_dump(),
        # secret=... or jwks_url=... depending on the algorithm
    )
    print("verified by:", summary["key_id"], "at", summary["signed_at"])
except ReceiptVerificationError as exc:
    print("AUDIT FAILURE:", exc)
```

Two algorithms supported:

| Algorithm    | Tier                         | Key material                                            |
|--------------|------------------------------|---------------------------------------------------------|
| HMAC-SHA256  | Community / Developer Pro    | Shared secret (`GL_FACTORS_SIGNING_SECRET`)             |
| Ed25519      | Consulting / Platform / Ent. | JWKS at `https://api.greenlang.io/.well-known/jwks.json`|

The verifier checks: signature, payload hash, future-timestamp drift, and (for Ed25519) the JWKS `kid`.

## Rate-limit-aware retries

Built-in: when the server returns `429 Too Many Requests`, the transport reads `Retry-After` and waits exactly that long before retrying (capped at 60s, up to `max_retries` attempts). On the final attempt a `RateLimitError` exposes the `retry_after` attribute so caller code can also back off.

```python
from greenlang_factors import RateLimitError

try:
    resolved = client.resolve(request)
except RateLimitError as exc:
    print(f"slow down -- retry after {exc.retry_after}s")
```

## Typed exceptions

| Exception              | Trigger                                                           |
|------------------------|-------------------------------------------------------------------|
| `AuthError`            | 401 -- bad/missing API key or JWT                                 |
| `TierError`            | 403 -- caller's tier insufficient                                 |
| `LicenseError`         | 403 -- factor is `connector_only` and caller lacks permission     |
| `LicensingGapError`    | 403 -- requested licensed pack not in contract                    |
| `EntitlementError`     | 403 -- plan does not include the requested feature                |
| `FactorNotFoundError`  | 404 -- factor id missing in this edition                          |
| `ValidationError`      | 400 / 422 -- bad request body                                     |
| `RateLimitError`       | 429 -- exceeded tier rate limit                                   |
| `EditionPinError`      | client-side: bad edition id, or 409/410 from server               |
| `EditionMismatchError` | server returned a different edition than the pin                  |
| `FactorsAPIError`      | catch-all base class                                              |

## CLI

```
gl-factors search "diesel US Scope 1"
gl-factors get-factor ef:co2:diesel:us:2026
gl-factors resolve "natural gas combustion" --jurisdiction US
gl-factors explain ef:co2:elec:us-ca:2027 --alternates 5
gl-factors list-editions
gl-factors verify-receipt response.json
```

Authentication is sourced from environment variables:

```
GREENLANG_FACTORS_BASE_URL    # default: http://localhost:8000
GREENLANG_FACTORS_API_KEY
GREENLANG_FACTORS_JWT
GREENLANG_FACTORS_EDITION
GL_FACTORS_SIGNING_SECRET     # for HMAC receipt verification
GL_FACTORS_JWKS_URL           # for Ed25519 receipt verification
```

## Async client

The async client mirrors the sync one method-for-method:

```python
import asyncio
from greenlang_factors import AsyncFactorsClient

async def main():
    async with AsyncFactorsClient(
        base_url="https://api.greenlang.io",
        api_key="gl_fac_your_key_here",
    ) as client:
        async with client.with_edition("2027.Q1") as scoped:
            resolved = await scoped.resolve({"activity": "diesel", ...})

asyncio.run(main())
```

## Links

* Pricing -- https://greenlang.ai/pricing
* Documentation -- https://developers.greenlang.ai
* Changelog -- https://github.com/greenlang/greenlang/blob/master/greenlang/factors/sdk/CHANGELOG.md
* Source -- https://github.com/greenlang/greenlang
