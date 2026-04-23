# Method Pack — Product Carbon

Implements product carbon footprint methodologies. Three profile variants.

| Profile | Standard |
|---|---|
| `product_carbon` | Generic product footprint (compatible with GHG Protocol Product Standard) |
| `product_iso14067` | ISO 14067:2018 |
| `product_pact` | PACT Pathfinder Framework v3.0 |

---

## Standards alignment

### ISO 14067 variant

- **ISO 14067:2018** — *Greenhouse gases — Carbon footprint of products — Requirements and guidelines for quantification*. [Link](https://www.iso.org/standard/71206.html). Clauses 6 (goal and scope), 7 (LCI), 8 (impact assessment), 9 (reporting).
- Linked: **ISO 14040 / 14044** (general LCA methodology), **ISO 14046** (water footprint cross-reference).

### PACT variant

- **WBCSD PACT Pathfinder Framework v3.0** — Partnership for Carbon Transparency product carbon data exchange standard. [Link](https://www.carbon-transparency.com/).
- Requires `parameters.supplier_primary_data_share >= 0.5` for compliant products.

### Generic variant

- **GHG Protocol Product Standard** (WRI/WBCSD, 2011). [Link](https://ghgprotocol.org/product-standard).
- **EN 15804:2012+A2:2019** — Product Category Rules (PCR) for construction products.

---

## Covered factor family

`materials_products`.

---

## Parameters

- `boundary` — `cradle_to_gate`, `gate_to_gate`, `cradle_to_grave`. Required.
- `allocation_method` — `mass`, `economic`, `system_expansion`. Required.
- `recycled_content_assumption` — 0..1.
- `supplier_primary_data_share` — 0..1 (PACT variant requires >=0.5).
- `pcr_reference` — product category rule (e.g., `"EN 15804:2012+A2:2019"`).
- `epd_reference` — EPD document ID (e.g., `"EPD-ITB-123"`).
- `pact_compatible` — boolean.

---

## Selection rules

- `selection.allowed_families`: `["materials_products"]`.
- `selection.jurisdiction_hierarchy`: `["supplier_location", "country", "region", "global"]`.
- `selection.priority_tiers`: `["supplier_epd", "supplier_reported", "industry_epd_average", "lca_database_proxy"]`.

---

## Boundary

- `boundary.system_boundary`: configurable per functional unit. ISO 14067 requires `cradle_to_grave` unless a partial footprint is explicitly declared.
- `boundary.functional_unit`: required. E.g., `"1 kg of cold-rolled steel sheet"`, `"1 kWh of delivered battery energy"`, `"1 pair of jeans wear"`.

---

## Inclusion / exclusion

- Biogenic carbon: `separate_report` for ISO 14067; `include_biogenic` for PACT (cradle-to-gate products only).
- F-gases (refrigerants): always included in product boundary if relevant.
- Capital goods: excluded from cradle-to-gate but included in cradle-to-grave operational boundary.

---

## PACT conformance (`product_pact` only)

Additional requirements:

- `supplier_primary_data_share >= 0.5` — at least half the cradle-to-gate emissions must come from verified supplier data.
- `pcr_reference` required and must align with the PACT Sector PCR for the product category.
- Data exchange format: PACT Pathfinder JSON schema (emitted by `/resolve` when `?format=pact` is set).

---

## Sources

- **ecoinvent v3.10** (BYO at v1; contract in outreach) — LCI background data.
- **EXIOBASE v3** (BYO) — multi-regional input-output proxy for upstream.
- **EF 3.1 Secondary Datasets** (JRC) — EU Commission environmental footprint reference.
- **EC3 EPD library** (BYO via Building Transparency API) — embodied carbon in construction.
- **EPD International** — per-EPD licensing; BYO at v1.
- **PACT Pathfinder** — framework; factor values `licensed_embedded`.

See [`docs/legal/source_rights_matrix.md`](../../legal/source_rights_matrix.md) for the full matrix.

---

## Fallback

`fallback.cannot_resolve_action = raise_no_safe_match`. `fallback.global_default_tier_allowed = false` for ISO 14067 and PACT variants (regulatory filings must have supplier or verified industry data; global LCI proxies are not permitted without explicit opt-in).

---

## Related

- [`/resolve`](../api-reference/resolve.md), [`concepts/method_pack.md`](../concepts/method_pack.md).
- [EU Policy pack](eu_policy.md) for Battery / Textile DPP product footprints.
