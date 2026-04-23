---
title: "API: /v1/factors/resolve-explain"
description: Resolve an activity into a chosen factor with full explain payload.
---

# `POST /api/v1/factors/resolve-explain`

The headline endpoint of the Factors API. Takes an activity description (or a structured activity record), runs the 7-step cascade, and returns the chosen factor plus the alternates considered, the assumptions made, and the receipt that lets you prove it later.

## Authentication

`X-API-Key: <gl_fac_...>` or `Authorization: Bearer <jwt>`. See [pricing](https://greenlang.ai/pricing) for tier requirements -- this endpoint is **Pro+**.

## Request

```http
POST /api/v1/factors/resolve-explain
Content-Type: application/json
X-API-Key: gl_fac_...
X-GreenLang-Edition: 2027.Q1   # optional: pin the catalog edition
```

```json
{
  "activity": "natural gas combustion",
  "method_profile": "corporate_scope1",
  "jurisdiction": "US-CA",
  "reporting_date": "2026-04-01",
  "quantity": 1000,
  "unit": "therm",
  "tenant_overrides_enabled": true
}
```

| Field                       | Type    | Required | Notes                                                |
|-----------------------------|---------|----------|------------------------------------------------------|
| `activity`                  | string  | yes      | Free text or canonical activity id.                  |
| `method_profile`            | string  | yes      | One of `corporate_scope1`, `corporate_scope2_location_based`, `corporate_scope2_market_based`, `corporate_scope3`, `freight_iso_14083`, `product_carbon`, `land_removals`, `finance_proxy`, `eu_cbam`. |
| `jurisdiction`              | string  | yes      | ISO 3166 country or sub-region (`US`, `US-CA`, `EU`, `DE`). |
| `reporting_date`            | string  | no       | ISO date; defaults to today.                         |
| `quantity`                  | number  | no       | Activity amount; required if you want `computed_total`. |
| `unit`                      | string  | no       | Activity unit; required when `quantity` is supplied. |
| `tenant_overrides_enabled`  | boolean | no       | Default `true`; set `false` to bypass your custom overrides. |

## Response

```json
{
  "chosen": { /* CanonicalFactorRecord */ },
  "alternates": [ /* up to 5 next-best CanonicalFactorRecords */ ],
  "tie_break_reasons": [
    "preferred jurisdiction match",
    "newer publication year",
    "lower DQS axes spread"
  ],
  "assumptions": [
    "Natural gas density assumed at standard conditions",
    "GWP-100 from IPCC AR6"
  ],
  "computed_total": { "value": 5302.0, "unit": "kg CO2e" },
  "edition_id": "2027.Q1",
  "receipt": {
    "signature": "<base64>",
    "algorithm": "sha256-hmac",
    "signed_at": "2026-04-23T15:30:00Z",
    "key_id": "gl-factors-v1",
    "payload_hash": "<sha256 hex>"
  }
}
```

## Errors

| Status | SDK exception          | Meaning                                          |
|--------|------------------------|--------------------------------------------------|
| 400    | `ValidationError`      | Body invalid; check field types / required.     |
| 401    | `AuthError`            | Bad API key / JWT.                              |
| 403    | `TierError`            | This endpoint is Pro+; upgrade.                 |
| 403    | `LicenseError`         | Result depends on a `connector_only` factor.    |
| 403    | `LicensingGapError`    | Premium pack required; not in your contract.    |
| 409    | `EditionPinError`      | Pinned edition has been retired.                |
| 429    | `RateLimitError`       | Slow down; respect `Retry-After`.               |

## SDK examples

```python
from greenlang_factors import FactorsClient

with FactorsClient(base_url="https://api.greenlang.io", api_key="gl_...") as client:
    resolved = client.resolve({
        "activity": "natural gas combustion",
        "method_profile": "corporate_scope1",
        "jurisdiction": "US-CA",
        "quantity": 1000,
        "unit": "therm",
    })
    print(resolved.chosen.factor_id, resolved.computed_total)
```

```ts
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_FACTORS_API_KEY!,
});
const resolved = await client.resolve({
  activity: "natural gas combustion",
  method_profile: "corporate_scope1",
  jurisdiction: "US-CA",
  quantity: 1000,
  unit: "therm",
});
console.log(resolved.chosen.factor_id, resolved.computed_total);
```
