# The 7-Step Resolution Cascade

When you POST to `/api/v1/factors/resolve-explain`, the engine does not "look up a factor." It runs a deterministic, ordered cascade that mirrors the way a methodology lead would pick a factor in a defensible audit. Every step is tried in order, and the first one that yields a method-pack-accepted candidate wins.

**Source of truth:** `greenlang/factors/resolution/engine.py` (`ResolutionEngine.resolve`)
**Tie-breaking:** `greenlang/factors/resolution/tiebreak.py`
**Method-pack selection rules:** `greenlang/factors/method_packs/*.py`

---

## The 7 steps, top to bottom

```
     ResolutionRequest
            |
            v
  +------------------------+
  | 1. Customer override   |  tenant overlay (your own private factors)
  +------------------------+
            | no match
            v
  +------------------------+
  | 2. Supplier-specific   |  PCF from a specific supplier
  +------------------------+
            | no match
            v
  +------------------------+
  | 3. Facility-specific   |  asset/meter-level factor
  +------------------------+
            | no match
            v
  +------------------------+
  | 4. Utility / tariff /  |  grid sub-region, eGRID sub-region,
  |    grid sub-region     |  tariff-level electricity
  +------------------------+
            | no match
            v
  +------------------------+
  | 5. Country / sector    |  national or sectoral average
  |    average             |
  +------------------------+
            | no match
            v
  +------------------------+
  | 6. Method-pack default |  the profile's published default
  +------------------------+
            | no match
            v
  +------------------------+
  | 7. Global default      |  last-resort IPCC / IEA global
  +------------------------+
            | no match
            v
        ResolutionError
```

`_STEP_ORDER` in `engine.py` is the authoritative list. The `fallback_rank` field on the `ResolvedFactor` response tells you which step produced the winner.

---

## Worked example

Request:

```json
{
  "activity": "diesel combustion stationary",
  "method_profile": "corporate_scope1",
  "jurisdiction": "US",
  "reporting_date": "2026-06-01",
  "supplier_id": null,
  "facility_id": null,
  "utility_or_grid_region": null
}
```

- **Step 1** (customer override): no tenant overlay for diesel. Miss.
- **Step 2** (supplier): `supplier_id` is null. Miss.
- **Step 3** (facility): `facility_id` is null. Miss.
- **Step 4** (utility/grid): diesel is not grid electricity; activity is not tariff-bound. Miss.
- **Step 5** (country / sector average): the EPA GHG Emission Factors Hub publishes a US stationary diesel factor with `license_class = public_us_government`. Match. `fallback_rank = 5`.

Result: `EF:US:diesel:2024:v1` wins at step 5 with CO2 + CH4 + N2O broken out per fuel gallon. Step 6 would have been the "method-pack default diesel" (not geography-aware) and step 7 the IPCC AR6 global.

---

## What each step looks for

### 1. Customer override (tenant overlay)

Your private factors loaded via `POST /api/v1/overrides` or the `tenant_overlay` writer. These always win if they match the activity + profile + jurisdiction. See `greenlang/factors/tenant_overlay.py`. Useful for supplier-specific PCFs you license directly.

### 2. Supplier / manufacturer-specific

Keyed on `request.supplier_id`. If the request carries a supplier, any factor tagged with the same supplier_id under the active method profile is considered. Scope 3 Cat 1 and Product Carbon profiles lean heavily on this step.

### 3. Facility / asset-specific

Keyed on `request.facility_id`. Factories with their own continuous-monitoring systems publish per-meter factors; this is where they land.

### 4. Utility / tariff / grid sub-region

For electricity and district heating. Routes by:

- `request.utility_or_grid_region` (explicit)
- `request.extras.region_code` (e.g. eGRID sub-region `SERC`)
- `request.jurisdiction` + method pack's `grid_lookup` helper

`corporate_scope2_market_based` gives residual-mix and contractual instruments priority at this step. See `greenlang/factors/method_packs/electricity.py`.

### 5. Country / sector average

The workhorse. IEA statistics, eGRID annual, DEFRA UK conversion, Japan METI, Australia NGA, etc. Keyed on `jurisdiction` + the method pack's `accept_sector_avg=True`.

### 6. Method-pack default

Each method pack publishes a "no-context default" — the factor you should fall back to when nothing more specific exists. For example `corporate_scope1` uses IPCC default emission factors for stationary combustion.

### 7. Global default

IPCC / IEA global averages. Last resort; every factor here has `dqs.geographical <= 2`.

---

## Selection rule (what makes a candidate "eligible")

Each method pack defines a `SelectionRule` that filters candidates before they reach the tie-breaker. The rule checks:

- `allowed_statuses` — e.g. CBAM only accepts `certified`; Preview is forbidden.
- `require_verification` — CBAM + PEF demand `verification == True`.
- `allowed_gwp_sets` — e.g. a profile may forbid AR4.
- `license_class_homogeneity` — a single resolution can't mix `open` and `restricted` factors (CTO non-negotiable #6). Enforced in `enforce_license_class_homogeneity`.

If the selection rule rejects every candidate at a step, the cascade moves on to the next step. If **all** steps reject, the engine raises `ResolutionError` and the API returns HTTP 422.

---

## Tie-breaking within a step

When multiple factors pass the selection rule at the same step, `build_tiebreak` ranks them by (in order):

1. Geographic specificity (sub-national > country > region > global).
2. Temporal specificity (reporting-year match > most recent > oldest valid).
3. Technological specificity (activity-sub-category match > generic).
4. Data-quality composite FQS (higher is better).
5. Source reputation score from `source_registry.yaml`.

Ties after all five are broken by lexicographic `factor_id` for determinism. The `tie_break_reasons` array on the response tells you which comparisons fired.

---

## Reading the response

```json
{
  "factor_id": "EF:US:diesel:2024:v1",
  "co2e_per_unit": 10.21,
  "unit": "kg/gal",
  "fallback_rank": 5,
  "method_profile": "corporate_scope1",
  "edition_id": "2027.Q1-electricity",
  "gas_breakdown": { "co2": 10.15, "ch4": 0.04, "n2o": 0.02 },
  "uncertainty_95ci": 0.05,
  "tie_break_reasons": [
    "geo_match: US exact",
    "temporal_match: 2024 vintage closest to 2026-06-01"
  ],
  "alternates": [
    { "factor_id": "EF:US:diesel:2022:v1", "score": 0.87, "step": 5 },
    { "factor_id": "EF:GLOBAL:diesel:IPCC:ar6", "score": 0.41, "step": 7 }
  ]
}
```

If `fallback_rank >= 6` you are using a generic default; that is a signal to load supplier-specific data or upgrade the tier that unlocks connector-only sources.

---

## Common questions

**Q: What if I want the raw catalog without the cascade?**
A: Use `GET /api/v1/factors/{factor_id}` for a direct fetch. But note — method-profile guardrails are bypassed and your output cannot bind to a compliance workflow that requires a `MethodProfile`.

**Q: Can I disable a step?**
A: No. The cascade order is a compliance guarantee. You can narrow inputs (don't pass `facility_id`) but you cannot re-order.

**Q: Does edition pinning affect the cascade?**
A: Edition pinning narrows the candidate pool to factors present in that edition. The cascade logic is unchanged.

---

## See also

- [Method packs](./method-packs.md) — what `SelectionRule` each profile publishes.
- [Version pinning](./version-pinning.md) — how edition pinning narrows the candidate pool.
- [Quality scores](./quality-scores.md) — how FQS ranks tie-breakers.
- [Errors](../api-reference/errors.md) — HTTP 422 `resolution_failed`.

---

## File citations

| Piece | File |
|---|---|
| 7-step cascade | `greenlang/factors/resolution/engine.py` (`ResolutionEngine.resolve`, `_STEP_ORDER`) |
| Tie-breaking | `greenlang/factors/resolution/tiebreak.py` (`build_tiebreak`) |
| `ResolutionRequest` schema | `greenlang/factors/resolution/request.py` |
| `ResolvedFactor` / `AlternateCandidate` | `greenlang/factors/resolution/result.py` |
| License-class homogeneity | `greenlang/data/canonical_v2.py::enforce_license_class_homogeneity` |
