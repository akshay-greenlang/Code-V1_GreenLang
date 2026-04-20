# `factors-sdk` v1.1.0 — CTO-Spec Release

> Python (`greenlang-factors-sdk`) and TypeScript (`@greenlang/factors-sdk`)
> **publish together** under the same semver.

## Headline

This SDK release matches the server-side F1 → F10 execution that brings
GreenLang Factors to 100 % of the CTO product outlook. Nothing 1.0.0 does
breaks — every addition is backward-compatible.

## What's new vs 1.0.0

### Resolution + explain

```python
from greenlang.factors.sdk import FactorsClient, FactorsConfig

client = FactorsClient(FactorsConfig(base_url="https://factors.greenlang.io"))
resolved = client.resolve(
    activity="diesel combustion stationary, India",
    method_profile="corporate_scope1",     # required — CTO non-negotiable #6
    jurisdiction="IN",
    reporting_date="2027-06-01",
)
print(resolved["chosen_factor_id"])
print(resolved["fallback_rank"], resolved["step_label"])
print(resolved["alternates"])
print(resolved["gas_breakdown"])
print(resolved["uncertainty"])
print(resolved["explainability"]["assumptions"])
```

### Mapping layer

Free-text → canonical key, client-side:

```python
from greenlang.factors.sdk.mapping import (
    map_fuel, map_transport, map_material, map_waste,
    map_electricity_market, map_classification, map_spend,
)

assert map_fuel("No. 2 distillate").canonical == "diesel"
assert map_transport("40-tonne truck, long haul").canonical["mode"] == "road"
```

### Signed receipts

```python
receipt = client.verify_receipt(response_body, attached_receipt)
assert receipt.ok  # HMAC-SHA256 or Ed25519
```

### Webhook registration

```python
sub = client.register_webhook(
    target_url="https://customer.example.com/hooks/factors",
    event_types=["factor.deprecated", "factor.updated"],
)
print(sub.subscription_id, sub.secret)
```

### Method profiles now enforced

Every resolve / match request must carry a `method_profile`. Valid values:

- `corporate_scope1`
- `corporate_scope2_location_based`
- `corporate_scope2_market_based`
- `corporate_scope3`
- `product_carbon`
- `freight_iso_14083`
- `land_removals`
- `finance_proxy`
- `eu_cbam`
- `eu_dpp`

This is CTO non-negotiable #6 — policy workflows must bind to a profile
before a factor is returned.

## Install

```bash
pip install -U greenlang-factors-sdk==1.1.0
# or
npm install @greenlang/factors-sdk@1.1.0
```

## Upgrade from 1.0.0

No action required for existing callers; `search()` / `match()` /
`calculate()` are unchanged. To start using the new resolution engine,
replace `match(...)` with `resolve(..., method_profile=...)`.

## Compatibility

| | Supported |
|---|---|
| **API server** | factors-api v1.1.0 (F1-F10 merge) |
| **Python** | 3.9, 3.10, 3.11, 3.12, 3.13 |
| **Node** | 18, 20, 22 |
| **Auth** | JWT or API-Key |

## Tag + publish

```bash
git tag factors-sdk-v1.1.0
git push origin factors-sdk-v1.1.0
```

Tag push fires `.github/workflows/factors-sdk-publish.yml`, which:

1. Validates all three version references match the tag.
2. Runs the Python SDK smoke (`from greenlang.factors.sdk import SDK_VERSION`).
3. Builds + publishes Python (`twine`) and TypeScript (`npm publish`).
4. Creates a GitHub Release with this file as the body.

## Full changelog

See `CHANGELOG.md` in the SDK directory.

---

*Cut: 2026-04-20. Signed-off under CTO F1-F10 execution.*
