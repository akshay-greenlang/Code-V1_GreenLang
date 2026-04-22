# Migrating from Climatiq

If you have been calling Climatiq's `/estimate` or `/search` endpoints, this is what changes and why.

**Takeaway:** GreenLang Factors is not a drop-in clone. It is method-profile-gated, edition-pinned, signed, and dual-class-licensed. Most callers move in a few hundred lines of code; the benefit is auditable output.

---

## Conceptual differences

| Thing | Climatiq | GreenLang Factors |
|---|---|---|
| Top-level call | `POST /estimate` with an `emission_factor.activity_id`. | `POST /api/v1/factors/resolve-explain` with `activity`, `method_profile`, and `jurisdiction`. |
| Methodology | Implicit; bound to the selected `activity_id`. | Explicit; **required** `method_profile` (CTO non-negotiable #6). 10 profiles registered. |
| Version pinning | `data_version="^0"` selector. | Immutable edition IDs (`2027.Q1-electricity`); header `X-Factors-Edition`. |
| Cryptographic evidence | None. | Signed receipt on every 2xx (`_signed_receipt` in body or headers). HMAC or Ed25519. |
| Licensing | Per-source paywalls, opaque. | Four explicit classes (`open`, `restricted`, `licensed`, `customer_private`) surfaced on every factor. |
| Fallback | Implicit single-candidate match. | 7-step cascade with explainable `fallback_rank`. |
| Resolution explanation | Not returned. | `tie_break_reasons` + alternates list. |
| GWP set | Configured account-level. | Per-request `gwp_set` parameter; gas breakdown kept server-side so any GWP set can be applied. |

---

## Field mapping table

### Request

| Climatiq field | GreenLang equivalent | Notes |
|---|---|---|
| `emission_factor.activity_id` | `activity` + `method_profile` + `jurisdiction` | GreenLang resolves, not selects. |
| `emission_factor.source` | `preferred_sources` (list) | GreenLang honours but the cascade can override. |
| `emission_factor.region` | `jurisdiction` | ISO 3166-1 alpha-2. |
| `emission_factor.year` | `reporting_date` | Full date; the cascade picks the closest valid vintage. |
| `emission_factor.source_lca_activity` | `extras.lca_activity` | |
| `parameters.energy` / `distance` / `money` / `weight` / `volume` | `extras.*` (activity-specific; e.g. `distance_km`, `payload_tonnes`, `revenue_eur`) | GreenLang uses typed `extras` keys per method pack. |
| `parameters.*_unit` | Baked into the activity unit. | `/resolve-explain` returns the factor's native unit; you multiply your magnitude client-side. |
| `metadata.idempotency_key` | `Idempotency-Key` header | Same semantics. |

### Response

| Climatiq field | GreenLang equivalent |
|---|---|
| `co2e` | `co2e_per_unit * your_activity_magnitude` (client-side) |
| `co2e_unit` | `unit` |
| `co2e_calculation_method` | `source.methodology` |
| `co2e_calculation_origin` | `source.organization` + `source.publication` |
| `emission_factor.activity_id` | `factor_id` |
| `emission_factor.source` | `source.organization` |
| `emission_factor.year` | `source.year` |
| `emission_factor.region` | `geography` |
| `emission_factor.category` | `sector_tags`, `activity_tags` |
| `emission_factor.lca_activity` | `extras.lca_activity` echo |
| `emission_factor.data_quality_flags` | `data_quality.*` (5 components) + `data_quality.overall_score` (0-100) |
| `constituent_gases.co2e_total` | `co2e_per_unit` (unchanged) |
| `constituent_gases.co2e_other` | `gas_breakdown.{hfcs, pfcs, sf6, nf3}` |
| `constituent_gases.co2` | `gas_breakdown.co2` |
| `constituent_gases.ch4` | `gas_breakdown.ch4` |
| `constituent_gases.n2o` | `gas_breakdown.n2o` |
| `audit_trail` | `_signed_receipt` + edition pin + full resolution payload |
| — (no equivalent) | `fallback_rank`, `tie_break_reasons`, `alternates` |
| — (no equivalent) | `license_class`, `redistribution_allowed`, `attribution_required` |
| — (no equivalent) | `edition_id` |

---

## Side-by-side example

### Climatiq

```http
POST https://beta3.api.climatiq.io/estimate
Authorization: Bearer $CLIMATIQ_API_KEY
Content-Type: application/json

{
  "emission_factor": {
    "activity_id": "fuel-type_diesel-fuel_use_stationary",
    "source": "EPA",
    "region": "US",
    "data_version": "^0"
  },
  "parameters": {
    "energy": 10000,
    "energy_unit": "kWh"
  }
}
```

### GreenLang Factors

```http
POST https://api.greenlang.io/api/v1/factors/resolve-explain
Authorization: Bearer $GL_API_KEY
X-Factors-Edition: 2027.Q1-electricity
Content-Type: application/json

{
  "activity": "diesel combustion stationary",
  "method_profile": "corporate_scope1",
  "jurisdiction": "US",
  "reporting_date": "2026-06-01",
  "preferred_sources": ["epa_hub"],
  "gwp_set": "IPCC_AR6_100"
}
```

Client-side:

```python
# Multiply by your activity magnitude.
kg_co2e = resolved.co2e_per_unit * 10000.0 * KWH_TO_UNIT_CONVERSION
```

If your activity unit does not match the factor's native unit, pass `extras.unit_override` or convert client-side via the unit graph (see `greenlang/factors/ontology/unit_graph.py`).

---

## Activity id mapping (starter kit)

A mental map from common Climatiq activity IDs to GreenLang (`activity`, `method_profile`) tuples:

| Climatiq `activity_id` | GreenLang `activity` | `method_profile` |
|---|---|---|
| `fuel-type_diesel-fuel_use_stationary` | `diesel combustion stationary` | `corporate_scope1` |
| `fuel-type_natural-gas_use_stationary` | `natural gas combustion stationary` | `corporate_scope1` |
| `electricity-supply_grid-residual_mix` | `purchased_electricity` | `corporate_scope2_market_based` |
| `electricity-supply_grid_location-based` | `purchased_electricity` | `corporate_scope2_location_based` |
| `freight_vehicle-type_truck_generic` | `freight_transport` + `extras.mode=road` | `freight_iso_14083` |
| `freight_vehicle-type_rail_electric` | `freight_transport` + `extras.mode=rail, traction=electric` | `freight_iso_14083` |
| `passenger_vehicle-type_car_generic` | `commute_generic_car` | `corporate_scope3` (Cat 7) |
| `refrigerant_hfc-134a` | `refrigerant_leakage` + `extras.gas_code=HFC-134a` | `corporate_scope1` |
| `purchased-goods_iron-and-steel` | `iron_steel_production_cradle_to_gate` | `corporate_scope3` (Cat 1) |
| `capital-goods_machinery-generic` | `capital_goods_machinery_generic` | `corporate_scope3` (Cat 2) |
| `hotel-night_global` | `business_travel_hotel` | `corporate_scope3` (Cat 6) |

---

## Migration checklist

1. **Enumerate your Climatiq calls.** Group by `activity_id`.
2. **Pick a `method_profile` per group.** This is the biggest design decision. See [method-packs](../concepts/method-packs.md).
3. **Replace activity ids with `activity` + `jurisdiction` + profile.** Most activities map 1:1; some require splitting (Scope 2 LB vs MB).
4. **Pick an edition and pin it.** Do not let your first GreenLang call run unpinned — it will return the current stable default and you will have no idea which it was.
5. **Update response parsing.** Pull `co2e_per_unit * your_magnitude` instead of `co2e`. Pull `gas_breakdown` instead of `constituent_gases`. Drop `co2e_calculation_method` in favour of `source.methodology`.
6. **Wire signed-receipt verification.** Non-negotiable for audit-defensible output. See [signed-receipts](../concepts/signed-receipts.md).
7. **Handle 451.** If any of your previous Climatiq calls used ecoinvent, IEA, or Sphera, GreenLang's licensed-connector path returns 451 when your tenant has no key. See [licensing-classes](../concepts/licensing-classes.md).
8. **Validate against a golden set.** Run 100-500 representative calls through both APIs; compare numbers; investigate deltas > 5%. Deltas are usually explained by (a) different AR / GWP set, (b) different source vintage, (c) different boundary for freight/product.

---

## What Climatiq users commonly find surprising

- **Every response is signed.** You cannot turn it off. Verify or discard.
- **Explicit method profile.** No more implicit fallback to a generic activity.
- **451 on licensed data.** Climatiq transparently proxies ecoinvent; GreenLang requires you to license it directly.
- **Gas breakdown always present.** You cannot ask for "just CO2e."
- **Edition pinning is a header, not a query parameter.** `X-Factors-Edition` on request, `X-GreenLang-Edition` on response.

---

## See also

- [5-minute quickstart](../quickstart/5-minute-quickstart.md)
- [Method packs](../concepts/method-packs.md)
- [Licensing classes](../concepts/licensing-classes.md)
- [Resolution cascade](../concepts/resolution-cascade.md)

---

## File citations

| Piece | File |
|---|---|
| Resolution engine | `greenlang/factors/resolution/engine.py` |
| Unit conversion (Climatiq unit → factor native) | `greenlang/factors/ontology/unit_graph.py` |
| Method profile enum | `greenlang/data/canonical_v2.py::MethodProfile` |
| Scope 2 MB/LB split | `greenlang/factors/method_packs/electricity.py` |
