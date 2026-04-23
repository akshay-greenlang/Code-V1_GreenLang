# GreenLang Factors SDK v1.0.0

**Release date (planned):** 2026-05-01
**Track:** FY27 Factors Launch -- C-3 (SDK v1.0 release in both languages)

## Overview

This is the **General Availability** release of the GreenLang Factors SDK. It ships in two flavours that share a single version number, a single REST contract, and a single docs portal:

* **Python**: `pip install greenlang-factors==1.0.0` -- requires Python 3.10+
* **TypeScript / Node**: `npm install @greenlang/factors@1.0.0` -- Node 18+

Both SDKs are typed end-to-end, have zero hard runtime dependencies on Stripe / FastAPI / OS-specific TLS plumbing, and ship with a CLI for ad-hoc work + offline receipt verification.

## Why v1.0.0

The pre-release line (0.x and 1.1.0-rc) was shipped to early-access customers between February and April 2026 and validated end-to-end against:

* The full FY27 Factors Resolution Engine (all 7 method packs)
* Edition pinning across the FY26.Q4 -> FY27.Q1 transition
* HMAC and Ed25519 signed-receipt issuance
* The new tiered SKU layout (Community / Developer Pro / Consulting&Platform / Enterprise)

The v1.0 line crystallises that contract under semantic versioning:

* The API surface is **stable** -- removals or type-narrowing changes will require a new major version.
* New optional parameters and new methods can land in 1.x without breaking callers.
* Bug fixes ship as 1.0.x patch releases.

## Highlights

### Edition pinning, the right way

Every Factors response carries an `X-GreenLang-Edition` header. The v1.0 SDK lets you **pin a client** to a specific edition so reproducibility is provably enforced on every request:

```python
from greenlang_factors import FactorsClient

with FactorsClient(base_url="https://api.greenlang.io", api_key="gl_...") as client:
    with client.with_edition("2027.Q1-electricity") as scoped:
        resolved = scoped.resolve({"activity": "electricity", ...})
```

Edition-id format is validated client-side (`v1.0.0`, `2027.Q1`, `2027.Q1-electricity`, `2027-04-01-freight` all accepted) and any server drift raises `EditionMismatchError` instead of being silently accepted.

### Signed-receipt verification, offline

Pro+ responses can carry a signed receipt. The v1.0 SDK verifies them entirely **offline** -- no network call back to GreenLang, so audit packages remain self-contained months or years after the fact:

```python
summary = client.verify_receipt(response.model_dump())
print("verified by", summary["key_id"], "at", summary["signed_at"])
```

Both algorithms are supported:

* `sha256-hmac` -- uses `GL_FACTORS_SIGNING_SECRET` (Community / Developer Pro)
* `ed25519` -- fetches and caches the JWKS document (Consulting / Platform / Enterprise)

The Python SDK uses the `cryptography` package (optional dep, install via `pip install greenlang-factors[crypto]`); the TypeScript SDK uses `jose`.

### Standalone CLI verifier

For auditors who do not want to install the full SDK in their reporting toolchain:

```sh
gl-factors verify-receipt response.json
```

Exits 0 on success, 3 on verification failure. Prints the verified summary as JSON.

### Rate-limit-aware retries

The transport already honoured `Retry-After` on 429 responses. v1.0 surfaces the `retry_after` value on the raised `RateLimitError` so caller code can back off too:

```python
from greenlang_factors import RateLimitError

try:
    client.resolve(request)
except RateLimitError as exc:
    print(f"slow down -- retry after {exc.retry_after}s")
```

### Typed exception hierarchy

Every HTTP failure surfaces as a specific exception subclass so callers can branch without parsing strings:

* `AuthError` (401)
* `TierError` (403, generic)
* `LicenseError` (403, factor licence)
* `LicensingGapError` (403, requested pack not in contract) -- **new**
* `EntitlementError` (403, plan does not include feature) -- **new**
* `FactorNotFoundError` (404)
* `ValidationError` (400/422)
* `RateLimitError` (429, exposes `retry_after`)
* `EditionPinError` (client-side validation, or 409/410) -- **new**
* `EditionMismatchError` (server returned a different edition than the pin)

## Compatibility

| Component | Range                   |
|-----------|-------------------------|
| Python    | 3.10, 3.11, 3.12, 3.13  |
| Node      | 18, 20, 22              |
| Server    | factors-api v1.0.0+     |

## Breaking changes from the pre-release line

* **`requires-python` bumped from 3.9 to 3.10.** 3.9 reaches EOL October 2026.
* **PyPI distribution renamed** from `greenlang-factors-sdk` to `greenlang-factors` (the import path stays `greenlang_factors`).

That is all. The method signatures from the 1.1.0-rc are unchanged.

## Install & verify

```sh
# Python
pip install greenlang-factors==1.0.0
python -c "import greenlang_factors; print(greenlang_factors.__version__)"
# -> 1.0.0

# Node
npm install @greenlang/factors@1.0.0
node -e "import('@greenlang/factors').then(m => console.log(m.SDK_VERSION))"
# -> 1.0.0
```

## What is NOT in v1.0

These were considered for v1.0 and deferred to 1.1:

* Streaming resolve responses for batch jobs > 10k rows.
* Browser-bundle build of the TypeScript SDK (current bundle is Node-only).
* Java / Go SDKs -- use the OpenAPI generator off the published spec for now.

## Links

* [Pricing](https://greenlang.ai/pricing)
* [Developer portal](https://developers.greenlang.ai)
* [Changelog](./CHANGELOG.md)
* [Source](https://github.com/greenlang/greenlang/tree/master/greenlang/factors/sdk)
