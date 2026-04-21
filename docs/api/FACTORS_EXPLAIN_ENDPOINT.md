# Factors Explain Endpoints (GAP-2)

> **Status:** shipping, Phase F3 (Resolution Engine) + GAP-2 close-out.
> **Tier gate:** Pro, Consulting, Enterprise, or Internal. Community → 403.
> **Source module:** `greenlang/integration/api/routes/factors.py`
> **Service layer:** `greenlang/factors/api_endpoints.py`
> **Resolution brain:** `greenlang/factors/resolution/engine.py` (7-step cascade)

## Why these endpoints exist

The Resolution Engine already computes a full explain payload every
time it resolves a factor, but until GAP-2 there was no REST surface
exposing that payload. Consultants, auditors, and Scope 3 analysts
need to justify *why* a particular factor was picked over its peers —
not just hand over a number. These endpoints let clients inspect:

- which of the 7 cascade steps produced the winner (`fallback_rank`);
- the full set of **alternates considered** (with `why_not_chosen`);
- the **quality score** and **uncertainty band**;
- the **gas breakdown** with CO2/CH4/N2O/HFCs/PFCs/SF6/NF3/biogenic_CO2
  kept as **separate components** — a CTO non-negotiable;
- the **assumptions** the engine applied;
- any **deprecation** notices and the replacement factor id.

## Endpoints

| Method | Path | Purpose |
|-------|------|---------|
| `GET` | `/api/v1/factors/{factor_id}/explain` | Explain payload for a specific factor in a default activity context. |
| `POST` | `/api/v1/factors/resolve-explain` | Run the 7-step cascade from a full `ResolutionRequest`. |
| `GET` | `/api/v1/factors/{factor_id}/alternates` | List top-N alternate factors for the same activity. |

All three routes:

- require `Authorization: Bearer <token>` with a Pro+ tier claim;
- are **rate-limited** per the standard `apply_rate_limit` middleware;
- set `X-GreenLang-Edition` and `X-GreenLang-Method-Profile` on every
  response;
- support `ETag` / `If-None-Match` conditional GETs (304 on match);
- set `Cache-Control: private, max-age=300`.

### Query parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `edition` | string | active edition | Pins to a specific catalog edition. `X-Factors-Edition` header takes priority. |
| `method_profile` | string | derived from factor | Any `MethodProfile` enum value. |
| `limit` | int | 5 | Alternates to return. **Clamped to `[1, 20]`.** |
| `include_preview` | bool | false | POST only. Allow `preview`-status factors. |
| `include_connector` | bool | false | POST only. Enterprise-only; silently clamped otherwise. |

### `GET /api/v1/factors/{factor_id}/explain`

```bash
curl -H "Authorization: Bearer $TOKEN" \
     "https://api.greenlang.io/api/v1/factors/EF:US:diesel:2024:v1/explain?limit=5"
```

Response (truncated):

```json
{
  "chosen_factor_id": "EF:US:diesel:2024:v1",
  "chosen_factor_name": "Diesel combustion (stationary)",
  "source_id": "epa_hub",
  "source_version": "2024.1",
  "factor_version": "v1",
  "vintage": 2024,
  "method_profile": "corporate_scope1",
  "fallback_rank": 5,
  "step_label": "country_or_sector_average",
  "why_chosen": "Selected at cascade step 5 (country_or_sector_average). exact-geography, verified, open-license.",
  "quality_score": 85.0,
  "uncertainty": {
    "distribution": "normal",
    "ci_95_percent": 0.05,
    "note": "exact-geography, verified, open-license"
  },
  "gas_breakdown": {
    "co2_kg": 10.154,
    "ch4_kg": 0.00041,
    "n2o_kg": 0.00008,
    "hfcs_kg": 0.0,
    "pfcs_kg": 0.0,
    "sf6_kg": 0.0,
    "nf3_kg": 0.0,
    "biogenic_co2_kg": 0.0,
    "co2e_total_kg": 10.198,
    "gwp_basis": "IPCC_AR6_100"
  },
  "assumptions": ["Resolved via step 5 (country_or_sector_average)."],
  "deprecation_status": null,
  "alternates": [],
  "resolved_at": "2026-04-20T12:34:56Z",
  "engine_version": "resolution-1.0.0"
}
```

### `POST /api/v1/factors/resolve-explain`

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
           "activity": "diesel combustion stationary",
           "method_profile": "corporate_scope1",
           "jurisdiction": "US",
           "reporting_date": "2026-06-01",
           "supplier_id": "acme-fuels"
         }' \
     "https://api.greenlang.io/api/v1/factors/resolve-explain?limit=10"
```

Request body fields (all keys of `ResolutionRequest`):

- `activity` *(required)* — free-text or canonical activity_id.
- `method_profile` *(required)* — one of `corporate_scope1`, `corporate_scope2_location_based`, `corporate_scope2_market_based`, `corporate_scope3`, `product_carbon`, `freight_iso_14083`, `land_removals`, `finance_proxy`, `eu_cbam`.
- `jurisdiction` — ISO country/region (`US`, `US-CA`, `EU`).
- `reporting_date` — ISO date; defaults to today.
- `supplier_id`, `facility_id`, `utility_or_grid_region` — for cascade
  steps 2, 3, 4.
- `tenant_id`, `activity_id` — customer overlay lookup.
- `preferred_sources` — list of `source_id` to rank first.
- `include_preview` — allow preview-status factors.
- `extras` — free-form context (e.g., `fuel_type`).

### `GET /api/v1/factors/{factor_id}/alternates`

```bash
curl -H "Authorization: Bearer $TOKEN" \
     "https://api.greenlang.io/api/v1/factors/EF:US:diesel:2024:v1/alternates?limit=10"
```

Returns the same anchor factor + up to 20 alternates, each with:

- `factor_id`;
- `tie_break_score` (lower is better);
- `why_not_chosen` (short phrase);
- `source_id`, `vintage`, `redistribution_class`.

## Python SDK usage

```python
from greenlang.sdk import FactorsClient

client = FactorsClient(api_key="gl_…")

# GET /explain
resolved = client.explain_factor("EF:US:diesel:2024:v1", limit=5)
print(resolved["chosen_factor_id"], resolved["fallback_rank"])

# POST /resolve-explain
result = client.resolve_and_explain(
    activity="diesel combustion stationary",
    method_profile="corporate_scope1",
    jurisdiction="US",
    reporting_date="2026-06-01",
    limit=10,
)
for alt in result["alternates"]:
    print(alt["factor_id"], alt["why_not_chosen"])

# GET /alternates
alts = client.list_alternates("EF:US:diesel:2024:v1", limit=10)
```

## Error codes

| HTTP | Meaning |
|------|---------|
| 200 | Explain payload returned. |
| 304 | `If-None-Match` matched the server `ETag`. |
| 400 | `ResolutionRequest` body failed Pydantic validation. |
| 403 | Caller tier is below Pro. |
| 404 | `factor_id` does not exist in the edition. |
| 422 | Cascade exhausted (no factor passes method-pack selection rule). |
| 429 | Rate limit exceeded. |

## Business rationale

The 7-step cascade is the single source of truth for factor selection
in GreenLang. Hiding it would mean:

1. Customers couldn't defend factor choice in audits (SOX, CSRD, CBAM).
2. Consultants couldn't run sensitivity analyses or justify methodology.
3. We would silently drift toward "trust us, it's correct" — the exact
   opposite of the zero-hallucination principle.

Exposing the cascade — including the alternates that *lost* — turns the
platform into a defensible, auditable system of record.

## Related

- `greenlang/factors/resolution/engine.py` — 7-step cascade implementation.
- `greenlang/factors/resolution/result.py` — `ResolvedFactor` Pydantic model.
- `greenlang/factors/resolution/request.py` — `ResolutionRequest` model.
- `greenlang/factors/resolution/tiebreak.py` — tie-break scoring inside a step.
- `greenlang/factors/method_packs/` — method-pack selection rules that gate
  each cascade step.
