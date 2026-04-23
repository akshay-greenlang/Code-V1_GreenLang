# `factors-sdk` v1.2.0 — Wave 2 / 2a / 2.5 envelope release

> Python (`greenlang-factors`) and TypeScript (`@greenlang/factors`)
> **publish together** under the same semver.

## Headline

Wave 2 tightened the server-side resolver contract: signed-receipt JSON
keys were renamed, new envelope fields (`chosen_factor`, composite FQS
0-100, richer uncertainty / licensing, structured deprecation status)
were added, and Wave 2.5 introduced a narrative `audit_text` field.

This release exposes every new field as a typed attribute on
`ResolvedFactor` while still reading the deprecated receipt key names
for **one more release** so no existing customer integration breaks on
upgrade.

## What's new vs 1.1.0

### Signed receipts — canonical key names (Wave 2a)

```python
from greenlang.factors.sdk.python import FactorsClient, SignedReceipt

client = FactorsClient(base_url="https://api.greenlang.io", api_key="gl_...")
resolved = client.resolve(request)

# Wave 2a: canonical top-level key is `signed_receipt`.
receipt: SignedReceipt | None = resolved.signed_receipt
print(receipt.receipt_id, receipt.alg, receipt.payload_hash)
print(receipt.verification_key_hint)
```

```ts
import { FactorsClient } from '@greenlang/factors';

const client = new FactorsClient({ baseUrl: 'https://api.greenlang.io', apiKey });
const resolved = await client.resolve(request);
console.log(resolved.signed_receipt?.alg);           // canonical
console.log(resolved.signed_receipt?.payload_hash);  // canonical
```

Field renames (old → new):

| Position | Old (deprecated) | New (canonical) |
|---|---|---|
| Top-level response key | `_signed_receipt` | `signed_receipt` |
| Inside the receipt | `algorithm` | `alg` |
| Inside the receipt | `signed_over` | `payload_hash` |

The SDK reads both names for one release. The deprecated names will be
removed in **v2.0.0**.

### New `audit_text` / `audit_text_draft` fields (Wave 2.5)

```python
resolved = client.resolve(request)
if resolved.audit_text:
    if resolved.audit_text_draft:
        print("[DRAFT]", resolved.audit_text)
    else:
        print(resolved.audit_text)
```

`audit_text_draft=True` means the narrative came from an unapproved
template. Do **not** ship drafts into regulatory submissions without
human review.

### `FactorCannotResolveSafelyError`

```python
from greenlang.factors.sdk.python import FactorCannotResolveSafelyError

try:
    resolved = client.resolve(request)
except FactorCannotResolveSafelyError as exc:
    print(exc.pack_id, exc.method_profile, exc.evaluated_candidates_count)
```

```ts
import { FactorCannotResolveSafelyError } from '@greenlang/factors';

try {
  const resolved = await client.resolve(request);
} catch (err) {
  if (err instanceof FactorCannotResolveSafelyError) {
    console.log(err.packId, err.methodProfile, err.evaluatedCandidatesCount);
  }
}
```

The SDK emits this error automatically when the API returns HTTP 422
with `error_code: "factor_cannot_resolve_safely"` (Wave 2 resolver).

### Typed envelope models

New classes (Python) / interfaces (TS):

- `ChosenFactor` — resolver-selected factor plus `release_version`
  (distinct from `factor_version`).
- `SourceDescriptor` — nested source block on `resolved.source`.
- `QualityEnvelope` — surfaces `composite_fqs_0_100`.
- `UncertaintyEnvelope` — superset of the old `Uncertainty`.
- `LicensingEnvelope` — `license_class`, `redistribution_class`,
  `upstream_licenses[]`, `attribution`, `restrictions[]`.
- `DeprecationStatus` — structured status envelope.
- `SignedReceipt` (Python class with `.from_response_dict()`) /
  `SignedReceiptEnvelope` (TS interface) — canonical Wave 2a shape.

### CLI additions

```bash
# Wave 2.5 audit_text preview (truncated to 200 chars by default)
greenlang-factors resolve "diesel stationary IN" \
    --method-profile corporate_scope1 --pretty
# Append --show-full-audit to print the full narrative.

# Grouped envelope fields on explain
greenlang-factors explain ef:co2:diesel:us:2026 --pretty

# Read HMAC secret from file instead of the command line
greenlang-factors verify-receipt response.json --key ./secret.txt
```

## Migration guide

### If your code reads `response['_signed_receipt']` directly…

Change it to `response['signed_receipt']`. Both work right now; only the
new spelling survives v2.0.0. If you use the typed SDK surface
(`ResolvedFactor.signed_receipt` or `resolved.signed_receipt`) you do
not need to change anything — the SDK normalises the key for you and
emits a `DeprecationWarning` on the fallback.

### If your code reads `receipt['algorithm']` or `receipt['signed_over']`…

Change them to `receipt['alg']` and `receipt['payload_hash']` respectively.
The verifier accepts both forms for this release. In the SDK you can
also use the typed `SignedReceipt` (Python) /
`SignedReceiptEnvelope` (TS) which uses the canonical names.

### If your code parses `ResolvedFactor.deprecation_status` as a string…

Wave 2 may now return a structured object. Handle both:

```python
ds = resolved.deprecation_status
if isinstance(ds, str):
    status_str = ds
elif isinstance(ds, dict):
    status_str = ds.get("status")
```

### No-change upgrades

Everything else is additive. `search()` / `get_factor()` / `match()` /
receipt verification all work unchanged for clients that do not use the
new fields.

## Install

```bash
pip install -U greenlang-factors==1.2.0
# or
npm install @greenlang/factors@1.2.0
```

## Compatibility

| | Supported |
|---|---|
| **API server** | factors-api with Wave 2 + Wave 2a + Wave 2.5 patches |
| **Python** | 3.10, 3.11, 3.12, 3.13 |
| **Node** | 18, 20, 22 |
| **Auth** | JWT or API-Key |

## Tag + publish

```bash
git tag factors-sdk-v1.2.0
git push origin factors-sdk-v1.2.0
```

Tag push fires `.github/workflows/factors-sdk-publish.yml`, which:

1. Validates the version references match the tag.
2. Runs `pytest tests/factors/sdk/` and `npm test` in
   `greenlang/factors/sdk/ts/`.
3. Builds + publishes Python (`twine`) and TypeScript (`npm publish`).
4. Creates a GitHub Release with this file as the body.

---

*Cut: 2026-04-23. Backward-compatible with SDK 1.1.0 callers; one-release
deprecation window for the Wave 2a receipt-key renames.*
