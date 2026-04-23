# Python SDK — `greenlang-factors`

Official Python SDK for the GreenLang Factors API. Tested against Python 3.10, 3.11, 3.12, 3.13.

**Canonical changelog:** [`greenlang/factors/sdk/CHANGELOG.md`](../../../greenlang/factors/sdk/CHANGELOG.md).

---

## Install

```bash
pip install greenlang-factors==1.2.0
```

Pin the minor. The `1.x` line maintains API stability; 2.0 will remove the back-compat receipt-key aliases documented in the SDK changelog.

---

## Auth

```python
import os
from greenlang_factors import FactorsClient

client = FactorsClient(
    api_key=os.environ["GL_API_KEY"],
    base_url="https://api.greenlang.io",  # or staging
)
```

Alternative OAuth2:

```python
client = FactorsClient.from_oauth(
    client_id=os.environ["GL_CLIENT_ID"],
    client_secret=os.environ["GL_CLIENT_SECRET"],
)
```

The SDK caches the OAuth2 token and refreshes 60s before expiry. See [`authentication.md`](../authentication.md).

---

## Resolve

```python
result = client.resolve(
    factor_family="electricity",
    quantity=12500,
    unit="kWh",
    method_profile="corporate_scope2_location_based",
    jurisdiction="IN",
    valid_at="2026-12-31",
)

print(result.chosen_factor.factor_id)           # EF:IN:grid:CEA:FY2024-25:v1
print(result.chosen_factor.factor_version)      # 1.0.0
print(result.chosen_factor.release_version)     # builtin-v1.0.0
print(result.emissions.co2e_kg)                 # 9950.0
print(result.emissions.gwp_basis)               # IPCC_AR6_100
print(result.quality.composite_fqs_0_100)       # 82.0
print(result.fallback_rank)                     # 4
print(result.licensing.redistribution_class)    # "open"
print(result.licensing.attribution_text)        # Required citation string
print(result.assumptions)                       # list[str]
print(result.audit_text)                        # Auditor-grade narrative
print(result.audit_text_draft)                  # False when template approved
```

See [`api-reference/resolve.md`](../api-reference/resolve.md).

---

## Explain

```python
explanation = client.explain(
    factor_id="EF:IN:grid:CEA:FY2024-25:v1",
    method_profile="corporate_scope2_location_based",
    quantity=12500,
    unit="kWh",
    jurisdiction="IN",
)

for tier in explanation.cascade:
    print(tier.rank, tier.label, tier.outcome)
```

---

## Verify a signed receipt (offline)

```python
from greenlang_factors.verify import verify_receipt

verified = verify_receipt(result.raw_response)
assert verified.valid is True
print(verified.alg)                      # Ed25519 or HS256
print(verified.verification_key_hint)    # key id
```

For Ed25519 the SDK fetches the JWKS from
`https://api.greenlang.io/.well-known/jwks.json` (override with `jwks_url=` or
`GL_FACTORS_JWKS_URL`). For HS256 pass `hmac_secret=...`. See [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).

---

## Batch

```python
batch = client.batch_resolve(items=[
    {"row_id": "r1", "factor_family": "electricity", "quantity": 12500,
     "unit": "kWh", "method_profile": "corporate_scope2_location_based",
     "jurisdiction": "IN", "valid_at": "2026-12-31"},
])

for r in batch.results:
    print(r.row_id, r.emissions.co2e_kg)
for e in batch.errors:
    print(e.row_id, e.error_code)
```

See [`api-reference/batch.md`](../api-reference/batch.md).

---

## Errors

The SDK raises a typed exception hierarchy. Catch by subclass as needed:

```python
from greenlang_factors.errors import (
    FactorCannotResolveSafelyError,
    LicensingGapError,
    EntitlementError,
    EditionMismatchError,
    EditionPinError,
    RateLimitError,
    UnauthorizedError,
    BadRequestError,
)

try:
    client.resolve(...)
except FactorCannotResolveSafelyError as e:
    print(e.pack_id, e.method_profile, e.evaluated_candidates_count)
except RateLimitError as e:
    print("retry after", e.retry_after, "seconds")
```

See the full mapping in [`error-codes.md`](../error-codes.md).

---

## Version pinning

Pin the client to a specific edition so every call runs against a reproducible snapshot:

```python
pinned = client.pin_edition("builtin-v1.0.0")
result = pinned.resolve(...)

# Or context-manager form
with client.with_edition("builtin-v1.0.0") as pinned:
    result = pinned.resolve(...)
```

If the server cannot serve the requested edition, `EditionMismatchError` is raised. `pin_edition` validates the `edition_id` format and raises `EditionPinError` on bad input.

---

## Rate-limit-aware retries

The transport layer honours `Retry-After` on 429 responses (default: 1 retry). For caller-side backoff, catch `RateLimitError` and wait `e.retry_after` seconds.

---

## CLI entry point

Installing the SDK also installs the `gl-factors` CLI. See [`sdks/cli.md`](cli.md).

---

## Related

- [SDK changelog](../../../greenlang/factors/sdk/CHANGELOG.md).
- [TypeScript SDK](typescript.md), [CLI](cli.md).
- [`api-reference/resolve.md`](../api-reference/resolve.md), [`api-reference/batch.md`](../api-reference/batch.md).
