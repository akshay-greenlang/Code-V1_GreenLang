# Canonical Factor Record Schema

Authoritative JSON-Schema: [`config/schemas/factor_record_v1.schema.json`](../../config/schemas/factor_record_v1.schema.json).
Field-by-field spec: [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md).

Every persisted factor row in the GreenLang catalog — raw ingest, normalized, or tenant overlay — MUST validate against this schema. This page shows a complete, schema-valid example so developers know exactly what comes back from `/resolve` and `/factors/{id}`.

---

## Complete example — India grid FY2024-25 location-based

```json
{
  "factor_id": "EF:IN:grid:CEA:FY2024-25:v1",
  "factor_family": "electricity",
  "factor_name": "India national grid (CEA) - FY2024-25 location-based",
  "method_profile": "corporate_scope2_location_based",
  "source_id": "india_cea_co2_baseline",
  "source_version": "v20.0",
  "factor_version": "1.0.0",
  "status": "active",
  "jurisdiction": {
    "country": "IN",
    "region": null,
    "grid_region": null
  },
  "valid_from": "2024-04-01",
  "valid_to": "2025-03-31",
  "activity_schema": {
    "category": "purchased_electricity",
    "sub_category": "grid_average",
    "classification_codes": ["NAICS:221112", "ISIC:D351"]
  },
  "numerator": {
    "co2": 0.7880,
    "ch4": 0.0000188,
    "n2o": 0.0000098,
    "f_gases": {},
    "co2e": 0.7960,
    "unit": "kg"
  },
  "denominator": {
    "value": 1.0,
    "unit": "kWh"
  },
  "gwp_set": "IPCC_AR6_100",
  "formula_type": "direct_factor",
  "parameters": {
    "scope_applicability": ["scope_2"],
    "uncertainty_low": -0.04,
    "uncertainty_high": 0.04,
    "electricity_basis": "location",
    "supplier_specific": false,
    "residual_mix_applicable": false,
    "certificate_handling": null,
    "td_loss_included": false,
    "subregion_code": null
  },
  "quality": {
    "temporal_score": 4,
    "geographic_score": 5,
    "technology_score": 4,
    "verification_score": 4,
    "completeness_score": 4,
    "composite_fqs": 82.0
  },
  "lineage": {
    "ingested_at": "2026-04-22T14:30:00Z",
    "ingested_by": "agent:ingest-india-cea@v2.0.0",
    "approved_by": "reviewer:data-ops@greenlang.io",
    "approved_at": "2026-04-22T17:10:00Z",
    "change_reason": "Initial ingest of CEA CO2 Baseline v20.0 (FY2023-24 data, published Dec 2024)",
    "previous_factor_version": null,
    "raw_record_ref": {
      "raw_record_id": "cea_v20_national_baseline",
      "raw_payload_hash": "5f3b2a9c8e1d4f7a6b9c2e5d8a1c4e7f9b3d6a8c2e5f1b4d7a9c6e3f2b5d8a1c",
      "raw_format": "xlsx",
      "storage_uri": "s3://greenlang-factors-raw/india/cea/v20.0/baseline.xlsx"
    }
  },
  "licensing": {
    "redistribution_class": "open",
    "customer_entitlement_required": false,
    "license_name": "Government-of-India-PD",
    "license_url": "https://cea.nic.in/cdm-co2-baseline-database/",
    "attribution_required": true,
    "restrictions": []
  },
  "explainability": {
    "assumptions": [
      "FY2023-24 CEA baseline used; latest available publication (Dec 2024)",
      "National average across all grid regions; no subregion granularity provided by CEA v20.0",
      "AR6 100-yr GWPs used for CO2e derivation (CH4=27.9, N2O=273)",
      "Transmission and distribution losses NOT included (busbar basis)"
    ],
    "fallback_rank": 4,
    "rationale": "Default India Scope 2 location-based factor when no supplier or subregion data is available."
  }
}
```

---

## Required top-level fields

All required by the JSON-Schema (`"required": [...]` clause):

`factor_id`, `factor_family`, `factor_name`, `method_profile`, `source_id`, `source_version`, `factor_version`, `status`, `jurisdiction`, `valid_from`, `valid_to`, `activity_schema`, `numerator`, `denominator`, `gwp_set`, `formula_type`, `parameters`, `quality`, `lineage`, `licensing`, `explainability`.

---

## Invariants enforced by writer and CI linter

1. `valid_from < valid_to` strictly.
2. `numerator.co2e` equals `co2 + ch4*GWP + n2o*GWP + sum(f_gas * GWP)` within 0.1% tolerance of the declared `gwp_set`.
3. `factor_version` matches semver.
4. `composite_fqs` equals the weighted formula within 0.5 absolute.
5. `customer_private` redistribution class implies `lineage.ingested_by ~= ^tenant:[0-9a-f-]{36}$`.
6. Keys present in `parameters` are a subset of the family discriminator's allowed set.
7. Combustion: `fossil_carbon_share + biogenic_carbon_share <= 1.0`.
8. Combustion: `HHV >= LHV` when both present.
9. A single API response MUST NOT mix more than one `licensing.redistribution_class` value.
10. `status == "deprecated"` implies `explainability.rationale` references a successor `factor_id`.
11. `status == "active"` requires `lineage.approved_by` AND `lineage.approved_at`.
12. At least one of `co2`, `ch4`, `n2o`, `co2e`, or `f_gases` populated for emission-family records.

See [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md) §3 for full invariant text.

---

## Discriminated `parameters` union

The `parameters` object's allowed keys depend on `factor_family`. Full per-family tables are in [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md) §2.9.

| `factor_family` | Key parameters |
|---|---|
| `combustion` | `fuel_code`, `LHV`, `HHV`, `density`, `oxidation_factor`, `fossil_carbon_share`, `biogenic_carbon_share`, ... |
| `electricity` | `electricity_basis`, `supplier_specific`, `residual_mix_applicable`, `certificate_handling`, `td_loss_included`, `subregion_code` |
| `transport` | `mode`, `vehicle_class`, `payload_basis`, `distance_basis`, `empty_running_assumption`, `utilization_rate`, `refrigerated`, `energy_basis` |
| `materials_products` | `boundary`, `allocation_method`, `recycled_content_assumption`, `supplier_primary_data_share`, `pcr_reference`, `epd_reference`, `pact_compatible` |
| `refrigerants` | `gas_code`, `leakage_basis`, `recharge_assumption`, `recovery_destruction_treatment`, `gwp_set_mapping` |
| `land_removals` | `land_use_category`, `sequestration_basis`, `permanence_class`, `reversal_risk_flag`, `biogenic_accounting_treatment` |
| `finance_proxies` | `asset_class`, `sector_code`, `intensity_basis`, `geography`, `proxy_confidence_class` |
| `waste` | `treatment_route`, `methane_recovery_factor`, `net_calorific_value` |

Auxiliary families (`energy_conversion`, `carbon_content`, `oxidation`, `heating_value`, `density`, `residual_mix`, `classification_mapping`) fall through to a generic parameter bucket.

---

## Related

- [`concepts/factor.md`](concepts/factor.md), [`concepts/quality_score.md`](concepts/quality_score.md).
- [`/resolve`](api-reference/resolve.md), [`/factors/{id}`](api-reference/factors.md).
- Field spec: [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md).
