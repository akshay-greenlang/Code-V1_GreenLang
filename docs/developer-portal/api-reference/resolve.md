# API — `POST /v1/factors/resolve`

Run the 7-step resolution cascade from a request body and return the chosen factor, alternates, gas-vector breakdown, quality envelope, assumptions, and signed receipt.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: resolveFactorsPost`).

---

## Endpoint

```
POST https://api.greenlang.io/v1/factors/resolve
Authorization: Bearer <key>
Content-Type: application/json
```

| Header | Required | Notes |
|---|---|---|
| `Authorization` | yes | `Bearer glk_...` or OAuth2 `access_token`. |
| `X-GreenLang-Edition` | no | Pin the response to a specific `edition_id`. |
| `X-Request-Id` | no | Caller-supplied correlation ID (echoed on response). |

---

## Request body

```json
{
  "factor_family": "electricity",
  "quantity": 12500,
  "unit": "kWh",
  "method_profile": "corporate_scope2_location_based",
  "jurisdiction": "IN",
  "valid_at": "2026-12-31",
  "include_preview": false
}
```

| Field | Required | Notes |
|---|---|---|
| `method_profile` | yes | One of the 14 profiles (see [`concepts/method_pack.md`](../concepts/method_pack.md)). Non-negotiable #6. |
| `quantity` | yes | Positive number in the given `unit`. |
| `unit` | yes | Activity unit. Converted via ontology to the factor's denominator unit. |
| `factor_family` | yes if no `activity_text` | Discriminator for `parameters`. |
| `activity_text` | yes if no `factor_family` | Free-text activity description; resolver runs semantic match. |
| `activity_code` | no | Qualified code (`NAICS:221112`, `CN:7208`, `ISIC:D351`, `CPC:171`). |
| `jurisdiction` | no | ISO 3166 alpha-2 country (`IN`), or region (`US-CA`, `ENTSOE-DE-LU`). Falls back by hierarchy. |
| `valid_at` | no | ISO-8601 date (reporting-period end). Defaults to today. |
| `supplier_id`, `facility_id`, `utility_or_grid_region` | no | Hints for higher-priority fallback tiers. |
| `edition` | no | Body-form alternative to the `X-GreenLang-Edition` header. |
| `include_preview` | no | Default `false`. Set `true` to include preview-status factors. |

---

## curl

```bash
curl -X POST "https://api.greenlang.io/v1/factors/resolve" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_family": "electricity",
    "quantity": 12500,
    "unit": "kWh",
    "method_profile": "corporate_scope2_location_based",
    "jurisdiction": "IN",
    "valid_at": "2026-12-31"
  }'
```

## Python

```python
from greenlang_factors import FactorsClient

client = FactorsClient(api_key=os.environ["GL_API_KEY"])

result = client.resolve(
    factor_family="electricity",
    quantity=12500,
    unit="kWh",
    method_profile="corporate_scope2_location_based",
    jurisdiction="IN",
    valid_at="2026-12-31",
)

print(result.chosen_factor.factor_id)
print(result.emissions.co2e_kg)
print(result.fallback_rank)
```

## TypeScript

```ts
import { FactorsClient } from "@greenlang/factors";
const client = new FactorsClient({ apiKey: process.env.GL_API_KEY! });

const result = await client.resolve({
  factorFamily: "electricity",
  quantity: 12500,
  unit: "kWh",
  methodProfile: "corporate_scope2_location_based",
  jurisdiction: "IN",
  validAt: "2026-12-31",
});
```

---

## Response (200 OK)

```json
{
  "chosen_factor": {
    "factor_id": "EF:IN:grid:CEA:FY2024-25:v1",
    "factor_version": "1.0.0",
    "release_version": "builtin-v1.0.0",
    "factor_name": "India national grid (CEA) - FY2024-25 location-based"
  },
  "method_profile": "corporate_scope2_location_based",
  "method_pack": { "pack_id": "corporate_scope2_location_based", "version": "1.0.0" },
  "source": {
    "source_id": "india_cea_co2_baseline",
    "source_version": "v20.0",
    "authority": "Central Electricity Authority (Government of India)"
  },
  "emissions": {
    "co2_kg": 9875.0,
    "ch4_kg": 0.235,
    "n2o_kg": 0.123,
    "co2e_kg": 9950.0,
    "gwp_basis": "IPCC_AR6_100"
  },
  "quality": {
    "composite_fqs_0_100": 82.0,
    "components": { "temporal_score": 4, "geographic_score": 5,
                    "technology_score": 4, "verification_score": 4,
                    "completeness_score": 4 }
  },
  "uncertainty": { "low": -0.04, "high": 0.04, "distribution": "normal" },
  "licensing": {
    "redistribution_class": "open",
    "license_name": "Government-of-India-PD",
    "attribution_text": "CO2 Baseline Database for the Indian Power Sector, CEA (Government of India), latest edition."
  },
  "fallback_rank": 4,
  "assumptions": [
    "FY2024-25 CEA baseline used as best-available proxy for FY2026-27 reporting",
    "T&D losses NOT included (busbar basis)",
    "AR6 100-yr GWPs used for co2e derivation"
  ],
  "alternates": [ /* ... other candidates considered ... */ ],
  "audit_text": "Chosen as the Scope 2 location-based default for all loads in India. ...",
  "audit_text_draft": false,
  "signed_receipt": {
    "receipt_id": "rcpt_2026Q2_01J...",
    "signature": "...",
    "verification_key_hint": "jwk-2026Q2-primary",
    "alg": "Ed25519",
    "payload_hash": "sha256:..."
  }
}
```

See [`concepts/signed_receipt.md`](../concepts/signed_receipt.md) for receipt verification.

---

## Response headers

| Header | Meaning |
|---|---|
| `X-GreenLang-Edition` | The edition actually served (pinned or default). |
| `X-Request-Id` | Echoed caller ID. |
| `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` | Rate-limit telemetry. |
| `ETag`, `Cache-Control` | Cache validators. |

---

## Errors

| Status | Code | When |
|---|---|---|
| 400 | `bad_request` | Missing required field, malformed body. |
| 401 | `unauthorized` | Missing / invalid token. |
| 402 | `payment_required` | Method profile or source requires a paid SKU. |
| 403 | `forbidden` | SKU held but source not entitled. |
| 409 | `edition_mismatch` | Pinned edition not servable. |
| 422 | `factor_cannot_resolve_safely` | No candidate satisfies pack rules; `raise_no_safe_match` triggered. |
| 429 | `rate_limited` | 100 req/min per key exceeded. |
| 451 | `legal_blocked` | Record's `redistribution_class` blocked for caller. |

See [`error-codes.md`](../error-codes.md) for the full reference.

---

## Related

- [`/explain`](explain.md) — full derivation trace for any factor.
- [`/batch-resolve`](batch.md) — array form for bulk workloads.
- [`concepts/method_pack.md`](../concepts/method_pack.md), [`concepts/factor.md`](../concepts/factor.md).
