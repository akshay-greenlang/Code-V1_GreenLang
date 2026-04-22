# Python SDK Quickstart

The Python SDK wraps every `/api/v1/factors` and `/api/v1/editions` route with typed Pydantic models, ETag caching, automatic retries with exponential backoff, and signed-receipt verification helpers.

**Source:** `greenlang/factors/sdk/python/client.py`

---

## Install

```bash
pip install greenlang-factors
```

Requires Python 3.9+. The SDK depends only on `httpx` and `pydantic>=2`.

---

## Initialise a client

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_your_key_here",          # or jwt_token=...
    default_edition="2027.Q1-electricity",  # sent as X-Factors-Edition
    timeout=30.0,
    max_retries=3,
)
```

`FactorsClient` is a context manager — use `with` to guarantee connection pool cleanup:

```python
with FactorsClient(base_url="https://api.greenlang.io", api_key="gl_pk_...") as client:
    ...
```

Async variant is `AsyncFactorsClient` in the same module.

---

## Step 1: Resolve a factor

The full 7-step cascade, returning the winner plus alternates:

```python
from greenlang.factors.sdk.python import FactorsClient
from greenlang.factors.sdk.python.models import ResolutionRequest

with FactorsClient(base_url="https://api.greenlang.io",
                   api_key="gl_pk_your_key_here") as client:

    req = ResolutionRequest(
        activity="diesel combustion stationary",
        method_profile="corporate_scope1",
        jurisdiction="US",
        reporting_date="2026-06-01",
    )

    resolved = client.resolve_explain(req, edition="2027.Q1-electricity")

    print(resolved.factor_id)                    # e.g. EF:US:diesel:2024:v1
    print(resolved.co2e_per_unit, resolved.unit) # e.g. 10.21 kg/gal
    print(resolved.fallback_rank)                # which of the 7 steps won
    for alt in resolved.alternates[:3]:
        print("  alt:", alt.factor_id, alt.score)
```

Under the hood this calls `POST /api/v1/factors/resolve-explain` — see `resolve_explain` in `greenlang/factors/sdk/python/client.py`.

---

## Step 2: Explain a specific factor

To understand why a specific factor_id would win for a given context:

```python
explain = client.explain_factor(
    factor_id="EF:US:diesel:2024:v1",
    method_profile="corporate_scope1",
    limit=10,
)

print("Chosen:", explain.factor_id)
print("Tie-break reasons:")
for reason in explain.tie_break_reasons:
    print(" -", reason)

print("Gas breakdown (never CO2e-only, CTO non-negotiable):")
print("  CO2: ", explain.gas_breakdown.co2)
print("  CH4: ", explain.gas_breakdown.ch4)
print("  N2O: ", explain.gas_breakdown.n2o)
```

Under the hood: `GET /api/v1/factors/{factor_id}/explain`.

---

## Step 3: Pin an edition

Edition pinning makes your output reproducible and audit-defensible.

```python
# Option A: set default_edition at construction time (sent on every request).
client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-electricity",
)

# Option B: pin per call - overrides the default.
resolved = client.resolve_explain(req, edition="2027.Q1-electricity")

# Read back what the server used (echoed on X-GreenLang-Edition response header).
print("Pinned to:", resolved.edition_id)
```

List available editions to find the right one:

```python
editions = client.list_editions(status="stable")
for e in editions:
    print(e.edition_id, e.created_at, e.factor_count)
```

The session below lists the current stable edition catalog, shows a hotfix rollback if one occurred, and surfaces deprecations you should migrate off. See [version-pinning](../concepts/version-pinning.md) for the rollback state machine.

---

## Step 4: Verify the signed receipt

Every 2xx JSON response from the SDK carries a `_signed_receipt`. The SDK exposes a helper that re-derives and checks the HMAC / Ed25519 signature:

```python
from greenlang.factors.sdk.python.client import verify_receipt

ok = verify_receipt(
    response_body=resolved.model_dump(),
    hmac_secret="shh_hmac_key",   # from dashboard
)

assert ok, "receipt verification failed - payload may be tampered"
```

For Ed25519 (Consulting / Enterprise tiers), pass `ed25519_public_key=...` instead. See [signed-receipts](../concepts/signed-receipts.md).

---

## Complete minimal example

```python
from greenlang.factors.sdk.python import FactorsClient
from greenlang.factors.sdk.python.models import ResolutionRequest

with FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_your_key_here",
    default_edition="2027.Q1-electricity",
) as client:

    resolved = client.resolve_explain(
        ResolutionRequest(
            activity="diesel combustion stationary",
            method_profile="corporate_scope1",
            jurisdiction="US",
            reporting_date="2026-06-01",
        )
    )

    print(f"{resolved.factor_id}: {resolved.co2e_per_unit} {resolved.unit}")
    print(f"Edition: {resolved.edition_id}")
    print(f"Receipt: {resolved.signed_receipt.algorithm} @ {resolved.signed_receipt.signed_at}")
```

Expected output:

```
EF:US:diesel:2024:v1: 10.21 kg/gal
Edition: 2027.Q1-electricity
Receipt: sha256-hmac @ 2026-04-22T14:33:02Z
```

---

## Error handling

Every non-2xx response raises `FactorsAPIError`. The 429 path raises `RateLimitError` (a subclass) with `retry_after` seconds already parsed:

```python
from greenlang.factors.sdk.python.errors import FactorsAPIError, RateLimitError

try:
    resolved = client.resolve_explain(req)
except RateLimitError as e:
    print(f"Backing off {e.retry_after}s")
except FactorsAPIError as e:
    print(f"HTTP {e.status_code}: {e.error_code} - {e.message}")
```

Full error-code table: [errors](../api-reference/errors.md).

---

## Async variant

```python
import asyncio
from greenlang.factors.sdk.python import AsyncFactorsClient

async def main():
    async with AsyncFactorsClient(base_url="https://api.greenlang.io",
                                  api_key="gl_pk_...") as client:
        resolved = await client.resolve_explain(req)
        print(resolved.factor_id)

asyncio.run(main())
```

---

## Next steps

- [cURL recipes](./curl-recipes.md) for CI/CD.
- [Resolution cascade](../concepts/resolution-cascade.md) to understand what just happened.
- [Method packs](../concepts/method-packs.md) to pick the right `method_profile`.
- [Signed receipts](../concepts/signed-receipts.md) for the verification deep dive.

---

## File citations

| SDK piece | File |
|---|---|
| `FactorsClient`, `AsyncFactorsClient` | `greenlang/factors/sdk/python/client.py` |
| `APIKeyAuth`, `JWTAuth` | `greenlang/factors/sdk/python/auth.py` |
| HTTP transport (retries, ETag cache) | `greenlang/factors/sdk/python/transport.py` |
| Models (`ResolutionRequest`, `ResolvedFactor`, ...) | `greenlang/factors/sdk/python/models.py` |
| Errors | `greenlang/factors/sdk/python/errors.py` |
| CLI (`glfactors`) | `greenlang/factors/sdk/python/cli.py` |
