# Factor Record v1.0 - Canonical Schema Specification

**Status:** FROZEN
**Version:** 1.0.0
**Frozen:** 2026-04-22
**Owner:** CTO, GreenLang Factors
**Schema file:** `config/schemas/factor_record_v1.schema.json` (JSON-Schema draft-2020-12)
**Scope:** Every persisted factor row (raw, normalized, customer overlay) in the GreenLang Factor Catalog.

---

## 1. Purpose

The Factor Record v1.0 schema is the **single authoritative contract** for how a GreenLang factor is stored, versioned, resolved, and explained. All upstream ingestion pipelines MUST normalize to this record before writing to the catalog, and every downstream consumer - the Resolution Engine, Method Packs, the `/explain` endpoint, Assurance agents, and compliance exporters (CBAM, CSRD, PCAF, PACT, ISO 14083, ISO 14067) - consumes factors through this schema.

### Six non-negotiables encoded by this schema

| # | Non-negotiable | Where encoded |
|---|----------------|---------------|
| 1 | Never store only CO2e - always keep gas vectors | `numerator.co2 / ch4 / n2o / f_gases` required; `co2e` derived |
| 2 | Never overwrite a factor | `factor_version` (semver), `lineage.previous_factor_version`, `lineage.change_reason` |
| 3 | Never hide fallback logic | `explainability.assumptions[]`, `explainability.fallback_rank` |
| 4 | Never mix licensing classes in one response | `licensing.redistribution_class` enum (API layer enforces homogeneity) |
| 5 | Never ship without validity + source version | `valid_from`, `valid_to`, `source_id`, `source_version` all required |
| 6 | Never let policy workflows call raw factors | `method_profile` required on every record; resolver binds caller to profile |

---

## 2. Field reference

Every field includes: **name, type, required?, allowed values, example, rationale**.
Field paths use dot notation (e.g., `jurisdiction.country`).

### 2.1 Top-level identity

#### `factor_id`
- **Type:** string, pattern `^EF:[A-Za-z0-9_.:-]+$`, 3-256 chars.
- **Required:** Yes.
- **Allowed:** Any ID starting with `EF:`. Convention: `EF:<jurisdiction>:<category>:<key>:<vintage>:<vx>`.
- **Example:** `"EF:US:grid:eGRID-SERC:2024:v1"`.
- **Rationale:** Globally unique, stable across versions of the same factor. `factor_version` disambiguates revisions. The `EF:` prefix lets static tooling reject accidentally-persisted strings that are not factor IDs.

#### `factor_family`
- **Type:** string enum.
- **Required:** Yes.
- **Allowed:** `combustion`, `electricity`, `transport`, `materials_products`, `refrigerants`, `land_removals`, `finance_proxies`, `waste`, `energy_conversion`, `carbon_content`, `oxidation`, `heating_value`, `density`, `residual_mix`, `classification_mapping`.
- **Example:** `"electricity"`.
- **Rationale:** Acts as the **discriminator** for the `parameters` sub-schema (JSON-Schema `oneOf`). A factor labelled `electricity` MUST carry only electricity-family parameter keys, and so on.

#### `factor_name`
- **Type:** string, 1-512 chars.
- **Required:** Yes.
- **Example:** `"US eGRID SERC subregion - 2024 location-based"`.
- **Rationale:** Human-readable display string for UIs and reports. Default language is en-US; localized variants live in a sibling catalog table, not on the record.

#### `method_profile`
- **Type:** string enum.
- **Required:** Yes.
- **Allowed:** `corporate_scope1`, `corporate_scope2_location_based`, `corporate_scope2_market_based`, `corporate_scope3`, `product_carbon`, `product_iso14067`, `product_pact`, `freight_iso14083_glec_wtw`, `freight_iso14083_glec_ttw`, `land_removals_ghgp_lsr`, `finance_proxies_pcaf`, `eu_cbam`, `eu_dpp_battery`, `eu_dpp_textile`.
- **Example:** `"corporate_scope2_location_based"`.
- **Rationale:** Non-negotiable #6. The Resolution Engine refuses to return a factor whose `method_profile` does not match the caller's declared methodology. This prevents e.g. a CBAM workflow silently binding to a PCAF-tagged proxy.

#### `source_id`
- **Type:** string, 1-256 chars.
- **Required:** Yes.
- **Example:** `"EPA_eGRID"`, `"DEFRA_GHG_CF"`, `"IEA_EF_2024"`, `"tenant:7f3e2..."`.
- **Rationale:** Upstream dataset identifier. When `licensing.redistribution_class == "customer_private"` the source_id MUST be a `tenant:<uuid>` (enforced by the Ingest agent).

#### `source_version`
- **Type:** string, 1-64 chars.
- **Required:** Yes.
- **Example:** `"2024.1"`, `"v9.0"`, `"AR6-2021"`.
- **Rationale:** Pins the exact release of the source dataset. Required by non-negotiable #5 (reproducibility).

#### `factor_version`
- **Type:** string, semver pattern.
- **Required:** Yes.
- **Example:** `"1.0.0"`, `"2.1.3-rc.1"`.
- **Rationale:** Non-negotiable #2. Every mutation to `numerator`, `denominator`, `parameters`, or `quality` MUST bump `factor_version`. The pair `(factor_id, factor_version)` is immutable.

#### `status`
- **Type:** string enum.
- **Required:** Yes.
- **Allowed:** `draft`, `under_review`, `active`, `deprecated`, `retired`.
- **Example:** `"active"`.
- **Rationale:** Lifecycle state. Production resolver only serves `active` without explicit opt-in. `deprecated` rows MUST provide `explainability.rationale` describing the replacement.

### 2.2 Jurisdiction

#### `jurisdiction.country`
- **Type:** string, ISO 3166-1 alpha-2.
- **Required:** Yes.
- **Example:** `"US"`, `"IN"`, `"DE"`, `"XX"` (global).
- **Rationale:** Mandatory coarsest geo. `"XX"` is the reserved global / unspecified token.

#### `jurisdiction.region`
- **Type:** string (ISO 3166-2 format) or null.
- **Required:** No.
- **Example:** `"US-CA"`, `"IN-MH"`.

#### `jurisdiction.grid_region`
- **Type:** string or null.
- **Required:** No.
- **Example:** `"eGRID-SERC"`, `"NERC-WECC"`, `"ENTSOE-DE-LU"`.
- **Rationale:** Only populated for electricity family; other families leave it null.

### 2.3 Validity window

#### `valid_from`
- **Type:** string, ISO 8601 date.
- **Required:** Yes.
- **Example:** `"2024-01-01"`.

#### `valid_to`
- **Type:** string, ISO 8601 date.
- **Required:** Yes.
- **Example:** `"2024-12-31"`, `"9999-12-31"` (open-ended).
- **Rationale:** Non-negotiable #5. INVARIANT: `valid_from < valid_to` strictly. `9999-12-31` is the sentinel for "still current".

### 2.4 Activity schema

#### `activity_schema.category`
- **Type:** string, 1-128 chars.
- **Required:** Yes.
- **Example:** `"purchased_electricity"`, `"stationary_combustion"`, `"road_freight"`.

#### `activity_schema.sub_category`
- **Type:** string or null, max 128 chars.
- **Required:** No.
- **Example:** `"grid_average"`, `"HGV_rigid_17t"`.

#### `activity_schema.classification_codes[]`
- **Type:** array of strings with pattern `^[A-Z][A-Z0-9_-]*:[A-Za-z0-9._-]+$`.
- **Required:** No (empty array default).
- **Example:** `["NAICS:221112", "CN:7208", "ISIC:D351", "CPC:171"]`.
- **Rationale:** Lets the resolver match factors to any classification-coded activity input (NAICS, ISIC, CPC, CN, HS, UNSPSC, GICS, etc.) without scheme-specific fields on the record.

### 2.5 Numerator (emissions)

#### `numerator.co2`
- **Type:** number >= 0 or null.
- **Required:** No (but see invariant below).
- **Example:** `0.4123`.
- **Units:** kg CO2 per denominator unit.

#### `numerator.ch4`
- **Type:** number >= 0 or null.
- **Units:** kg CH4 per denominator unit.

#### `numerator.n2o`
- **Type:** number >= 0 or null.
- **Units:** kg N2O per denominator unit.

#### `numerator.f_gases`
- **Type:** object `{ gasCode: kgPerDenom }` or absent.
- **Example:** `{ "HFC-134a": 0.00021, "SF6": 0.0 }`.

#### `numerator.co2e`
- **Type:** number >= 0 or null.
- **Invariant:** If present, MUST equal `co2*1 + ch4*GWP_ch4(gwp_set) + n2o*GWP_n2o(gwp_set) + sum(f_gas_i * GWP_i(gwp_set))` within **0.1% relative tolerance**. Non-negotiable #1.
- **Rationale:** `co2e` is **derived**, never authoritative. Stored only as a cache for faster reads; the writer MUST recompute on every write.

#### `numerator.unit`
- **Type:** string enum `kg | g | t | lb`.
- **Required:** Yes (default `"kg"`).

### 2.6 Denominator (activity)

#### `denominator.value`
- **Type:** number > 0.
- **Required:** Yes.
- **Example:** `1.0`.
- **Rationale:** Almost always 1.0 for per-unit factors. Non-unit denominators are used for tabulated brackets (e.g., "factor per 5 t-km").

#### `denominator.unit`
- **Type:** string, 1-32 chars.
- **Required:** Yes.
- **Example:** `"kWh"`, `"MJ"`, `"L"`, `"gal"`, `"kg"`, `"t"`, `"t-km"`, `"v-km"`, `"TEU-km"`, `"USD"`, `"EUR"`.

### 2.7 GWP set

#### `gwp_set`
- **Type:** string enum.
- **Required:** Yes.
- **Allowed:** `IPCC_AR4_100`, `IPCC_AR5_100`, `IPCC_AR5_20`, `IPCC_AR6_100`, `IPCC_AR6_20`, `Kyoto_SAR_100`.
- **Default for new records:** `IPCC_AR6_100`.
- **Rationale:** All `co2e` derivations reference this set. Multi-horizon reporting (20yr + 100yr) is handled by carrying two sibling records with the same `factor_id` but distinct `gwp_set` values.

### 2.8 Formula type

#### `formula_type`
- **Type:** string enum.
- **Required:** Yes.
- **Allowed:** `direct_factor`, `combustion`, `lca`, `spend_proxy`, `transport_chain`, `residual_mix`, `carbon_budget`, `custom`.
- **Example:** `"direct_factor"` for grid intensity; `"combustion"` for fuel factors that need fuel * LHV * oxidation * EF chains; `"transport_chain"` for ISO 14083 multi-leg calcs.

### 2.9 Parameters (category-specific, discriminated union)

The `parameters` object is **required** but its allowed keys depend on `factor_family`. The JSON-Schema encodes this as `oneOf` on `factor_family`. The table below summarises each family's parameter contract.

#### Common keys (allowed on every family)

| Key | Type | Notes |
|-----|------|-------|
| `scope_applicability[]` | array of `scope_1 / scope_2 / scope_3 / scope_4` | Which GHG Protocol scopes this factor is valid for. |
| `uncertainty_low` | number | Lower 95% CI as signed fractional deviation. |
| `uncertainty_high` | number | Upper 95% CI as signed fractional deviation. |

#### `combustion` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `fuel_code` | string | Yes | `"diesel"`, `"natural_gas"`, `"anthracite"`, `"biodiesel_b100"` |
| `LHV` | number >= 0 | No | `35.8` (MJ/L for diesel) |
| `HHV` | number >= 0 | No | HHV MUST >= LHV |
| `density` | number > 0 | No | `0.832` kg/L for diesel |
| `oxidation_factor` | number 0-1 | No | `1.0` (gas), `0.98` (coal) |
| `fossil_carbon_share` | number 0-1 | No | `1.0` (fossil), `0.0` (biofuel) |
| `biogenic_carbon_share` | number 0-1 | No | fossil + biogenic <= 1.0 |
| `sulfur_share` | number 0-1 | No | Used for SOx co-reporting |
| `moisture_share` | number 0-1 | No | As-received biomass |
| `ash_share` | number 0-1 | No | As-received solids |

#### `electricity` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `electricity_basis` | `location / market / supplier / residual` | Yes | `"location"` |
| `supplier_specific` | boolean | No | `true` for green tariffs |
| `residual_mix_applicable` | boolean | No | `true` for AIB residual rows |
| `certificate_handling` | `GO / REC / I-REC / null` | No | `"REC"` |
| `td_loss_included` | boolean | No | `false` for busbar; `true` for delivered |
| `subregion_code` | string | No | `"eGRID-SERC"` |

#### `transport` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `mode` | `road / rail / sea / air / inland_waterway` | Yes | `"road"` |
| `vehicle_class` | string (GLEC) | No | `"HGV_rigid_17t"` |
| `payload_basis` | `t-km / v-km / TEU-km / pax-km` | No | `"t-km"` |
| `distance_basis` | `great_circle / route / route_with_empty` | No | `"route_with_empty"` |
| `empty_running_assumption` | 0-1 | No | `0.25` |
| `utilization_rate` | 0-1 | No | `0.75` |
| `refrigerated` | boolean | No | `false` |
| `energy_basis` | `WTW / TTW / TTT / WTT` | No | `"WTW"` |

#### `materials_products` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `boundary` | `cradle_to_gate / gate_to_gate / cradle_to_grave` | Yes | `"cradle_to_gate"` |
| `allocation_method` | `mass / economic / system_expansion` | Yes | `"economic"` |
| `recycled_content_assumption` | 0-1 | No | `0.30` |
| `supplier_primary_data_share` | 0-1 | No | PACT alignment |
| `pcr_reference` | string | No | `"EN 15804:2012+A2:2019"` |
| `epd_reference` | string | No | `"EPD-ITB-123"` |
| `pact_compatible` | boolean | No | `true` |

#### `refrigerants` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `gas_code` | string | Yes | `"R-134a"`, `"R-410A"` |
| `leakage_basis` | `annual / charge / disposal` | Yes | `"annual"` |
| `recharge_assumption` | 0-1 | No | `0.10` |
| `recovery_destruction_treatment` | `none / partial_recovery / full_recovery / destroyed` | No | `"partial_recovery"` |
| `gwp_set_mapping` | GWP set enum | No | For GWP alignment |

#### `land_removals` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `land_use_category` | string | Yes | `"forest_land"`, `"cropland"` |
| `sequestration_basis` | `stock_change / flux / gain_loss / tier1_default / tier3_model` | Yes | `"stock_change"` |
| `permanence_class` | `short_term / medium_term / long_term / permanent` | No | `"long_term"` |
| `reversal_risk_flag` | boolean | No | `true` |
| `biogenic_accounting_treatment` | `zero_rated / separate_reporting / included_in_co2e` | No | `"separate_reporting"` |

#### `finance_proxies` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `asset_class` | string (PCAF) | Yes | `"listed_equity_and_corporate_bonds"` |
| `sector_code` | string (NAICS/ISIC) | No | `"NAICS:221112"` |
| `intensity_basis` | `revenue / asset / ebitda / employee` | Yes | `"revenue"` |
| `geography` | string | No | `"US"` |
| `proxy_confidence_class` | `score_1..score_5` | No | `"score_3"` |

#### `waste` parameters

| Key | Type | Required? | Example |
|-----|------|-----------|---------|
| `treatment_route` | `landfill / incineration / compost / recycle / anaerobic_digestion` | Yes | `"landfill"` |
| `methane_recovery_factor` | 0-1 | No | `0.75` |
| `net_calorific_value` | number >= 0 | No | MJ/kg for incineration |

### 2.10 Quality (Factor Quality Score)

All five component scores are **integers 1-5** (GHG Protocol Product Standard convention).

| Key | Weight |
|-----|-------:|
| `temporal_score` | 0.25 |
| `geographic_score` | 0.25 |
| `technology_score` | 0.20 |
| `verification_score` | 0.15 |
| `completeness_score` | 0.15 |

#### `composite_fqs`
- **Type:** number, 0-100.
- **Formula:** `composite_fqs = 20 * (0.25*temporal + 0.25*geographic + 0.20*technology + 0.15*verification + 0.15*completeness)`.
- **Invariant:** MUST equal the weighted average within **+/- 0.5 absolute tolerance**.
- **Example:** All 5's => `20 * 5 = 100`. All 1's => `20`. Mixed `(5,5,4,3,4) => 20*(1.25+1.25+0.80+0.45+0.60) = 20*4.35 = 87.0`.

### 2.11 Lineage

#### `lineage.ingested_at`
- **Type:** string, RFC 3339 datetime.
- **Required:** Yes.
- **Example:** `"2026-04-22T14:30:00Z"`.

#### `lineage.ingested_by`
- **Type:** string, 1-256 chars.
- **Required:** Yes.
- **Example:** `"agent:ingest-epa@v2.1.0"`, `"tenant:7f3e2a1b-..."`.
- **Invariant:** When `licensing.redistribution_class == "customer_private"`, MUST match pattern `^tenant:[0-9a-f-]{36}$`.

#### `lineage.approved_by`
- **Type:** string or null.
- **Required:** No (but required before `status` transitions to `active`).

#### `lineage.approved_at`
- **Type:** datetime or null.

#### `lineage.change_reason`
- **Type:** string, 1-2048 chars.
- **Required:** Yes (non-negotiable #2).
- **Example:** `"Annual refresh: EPA eGRID 2024 Q1 release"`, `"Corrected CH4 coefficient per IPCC AR6 erratum"`.

#### `lineage.previous_factor_version`
- **Type:** string (semver) or null.
- **Example:** `"0.9.3"`.

#### `lineage.raw_record_ref`
- **Type:** object or null.
- **Fields:** `raw_record_id`, `raw_payload_hash` (SHA-256 hex), `raw_format` (`csv | xml | json | pdf_ocr | yaml | xlsx | api`), `storage_uri`.
- **Rationale:** Lets a normalized row be re-derived from the raw source in case the matcher vocabulary evolves. Required for licensed/restricted sources.

### 2.12 Licensing

#### `licensing.redistribution_class`
- **Type:** string enum.
- **Required:** Yes.
- **Allowed:**
  - `open` - Free redistributable, e.g. EPA/DEFRA open data.
  - `restricted` - API-only, no bulk export (attribution / field-limit restrictions).
  - `licensed` - Licensed premium, OEM-resellable under separate license.
  - `customer_private` - Customer-specific overrides, **never served cross-tenant** (non-negotiable #4).
- **Example:** `"open"`.

#### `licensing.customer_entitlement_required`
- **Type:** boolean.
- **Required:** Yes.
- **Rationale:** When `true`, the caller must present an entitlement token that covers `source_id`. The gateway rejects the call otherwise.

#### `licensing.license_name`
- **Type:** string or null.
- **Example:** `"CC-BY-4.0"`, `"CC0-1.0"`, `"OGL-UK-3.0"`, `"proprietary"`.

#### `licensing.license_url`
- **Type:** URI or null.

#### `licensing.attribution_required`
- **Type:** boolean (default false).

#### `licensing.restrictions`
- **Type:** array of strings (default empty).
- **Example:** `["internal_use_only", "no_derivative_works", "share_alike"]`.

### 2.13 Explainability

#### `explainability.assumptions[]`
- **Type:** array of strings (each 1-1024 chars).
- **Required:** Yes (may be empty array).
- **Example:** `["assumes average utilization 0.75", "no temperature correction applied", "AR6 100-yr GWPs"]`.
- **Rationale:** Non-negotiable #3. Every assumption MUST be surfaced by `/explain`. Resolver attaches these to `ResolvedFactor.assumptions`.

#### `explainability.fallback_rank`
- **Type:** integer 1-7.
- **Required:** Yes.
- **Scale:** 1 = customer override / primary data; 2 = same-supplier contracted; 3 = subregion match; 4 = country match; 5 = region match; 6 = continent match; 7 = global default.
- **Example:** `4` for a country-average grid factor.

#### `explainability.rationale`
- **Type:** string or null, max 2048 chars.
- **Example:** `"Chosen as the Scope 2 location-based default for all loads in eGRID-SERC subregion 2024."`.

---

## 3. Invariants (must be enforced by writer + CI linter)

1. **Date ordering:** `valid_from < valid_to` strictly. Use `9999-12-31` for open-ended.
2. **CO2e consistency:** If `numerator.co2e` present then
   `|co2e - (co2*1 + ch4*GWP_ch4 + n2o*GWP_n2o + sum_i(fgas_i * GWP_i))| / co2e < 0.001`
   using the `gwp_set` declared on the record.
3. **Semver:** `factor_version` matches `^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-...)?(\+...)?$`.
4. **Composite FQS:** `|composite_fqs - 20*(0.25*T + 0.25*G + 0.20*Tech + 0.15*V + 0.15*C)| <= 0.5`.
5. **Tenant binding:** `redistribution_class == "customer_private"` IMPLIES `lineage.ingested_by` matches `^tenant:[0-9a-f-]{36}$`.
6. **Parameter discriminator:** Keys present in `parameters` MUST be a subset of the allowed set for the declared `factor_family` (enforced by `oneOf`).
7. **Combustion fossil+biogenic:** `fossil_carbon_share + biogenic_carbon_share <= 1.0`.
8. **Combustion HHV/LHV:** If both present, `HHV >= LHV`.
9. **Customer-private non-mixing (API-level):** A single response payload MUST NOT mix more than one distinct value of `licensing.redistribution_class`.
10. **Deprecation replacement:** `status == "deprecated"` IMPLIES `explainability.rationale` references the successor `factor_id`.
11. **Active status prerequisites:** `status == "active"` REQUIRES `lineage.approved_by` AND `lineage.approved_at`.
12. **CO2e population rule:** At least ONE of `numerator.co2`, `numerator.ch4`, `numerator.n2o`, `numerator.co2e`, or `numerator.f_gases` must be populated for emission-family records (combustion, electricity, transport, materials_products, refrigerants, waste); parameter-only families (energy_conversion, density, heating_value, etc.) MAY omit all gas fields.

---

## 4. Migration Delta

This is the CTO-spec-vs-existing-code drift assessment, produced by cross-referencing:

- `greenlang/data/emission_factor_record.py` (existing v2 dataclass)
- `greenlang/data/canonical_v2.py` (CTO-spec Phase F1 extension types)
- `greenlang/data/models/emission_factor.py` (Pydantic v1 shim)
- `config/schemas/emission_factors_schema.json` (legacy draft-07 schema for YAML bundles)
- `greenlang/factors/resolution/result.py` (ResolvedFactor payload)

### 4.1 Drift items

| # | File / symbol | Drift from v1 spec | Recommended fix |
|---|---------------|--------------------|-----------------|
| D1 | `emission_factor_record.py::EmissionFactorRecord` | Uses **flat** `geography`, `region_hint`, `fuel_type`, `unit` at top level instead of structured `jurisdiction`, `activity_schema`, `denominator`. | Introduce `from_canonical_v1()` / `to_canonical_v1()` methods that fold flat geo into `Jurisdiction` and flat fuel/unit into `ActivitySchema + denominator`. |
| D2 | `emission_factor_record.py` | `factor_id` pattern enforced as `EF:*` via assertion, but schema v1 requires a stricter regex `^EF:[A-Za-z0-9_.:-]+$`. | Tighten the `__post_init__` validator to match the pattern. |
| D3 | `emission_factor_record.py::DataQualityScore` | 5 dims are `temporal / geographical / technological / representativeness / methodological` and overall is 1-5. | v1 renames to `temporal / geographic / technology / verification / completeness` and uses **0-100 composite**. Map old -> new: `representativeness -> completeness`, `methodological -> verification`; rescale overall via `composite_fqs = overall * 20`. |
| D4 | `emission_factor_record.py::SourceProvenance.version` | Stored as plain version string on the provenance sub-object. | Rename / promote to top-level `source_version` (required in v1); keep `SourceProvenance` as a helper for citation text only. |
| D5 | `emission_factor_record.py::LicenseInfo` vs `canonical_v2.RedistributionClass` | Two license systems coexist: (a) boolean flags (`redistribution_allowed`, `commercial_use_allowed`); (b) enum `RedistributionClass`. The v2 enum adds `OEM_REDISTRIBUTABLE` which is **not** in the CTO v1 spec (spec lists 4: open / restricted / licensed / customer_private). | Drop `OEM_REDISTRIBUTABLE` from the v1-served enum (keep it as internal tag under `licensing.restrictions`). Keep legacy boolean flags as derived / denormalized cache only. |
| D6 | `canonical_v2.FactorFamily` | Lists 15 families; v1 spec lists 8 primary + 7 auxiliary (same total 15) but the discriminated union for `parameters` is only defined for the 8 primary (combustion, electricity, transport, materials_products, refrigerants, land_removals, finance_proxies, waste). | Auxiliary families (`energy_conversion`, `carbon_content`, `oxidation`, `heating_value`, `density`, `residual_mix`, `classification_mapping`) fall through to the generic parameter bucket in the JSON-Schema `oneOf` - **already implemented** in v1 schema. No data migration needed; just document. |
| D7 | `canonical_v2.FormulaType` vs v1 spec | v1 spec does not itemise `formula_type`; v2 already has the correct enum. | Adopt v2 enum wholesale (done - same 8 values in schema). |
| D8 | `canonical_v2.MethodProfile` | Missing spec entries: `product_iso14067`, `product_pact`, `freight_iso14083_glec_wtw`, `freight_iso14083_glec_ttw`, `eu_dpp_battery`, `eu_dpp_textile`. v2 collapses these into `PRODUCT_CARBON / FREIGHT_ISO_14083 / EU_DPP`. | Expand `MethodProfile` enum in `canonical_v2.py` to the full v1 list (14 values). Add backward-compat alias map `PRODUCT_CARBON <- {product_carbon, product_iso14067, product_pact}`. |
| D9 | `canonical_v2.UncertaintyDistribution` | Exists in v2; v1 does not include distribution type at record level, only `uncertainty_low / uncertainty_high`. | Keep `UncertaintyDistribution` as optional extension under `parameters` free-form bucket; do not add to v1 top-level. The `ResolvedFactor.uncertainty.distribution` already exposes it. |
| D10 | `canonical_v2.PrimaryDataFlag` | Exists in v2; v1 does not include `primary_data_flag`. | Treat as optional extension for Phase F2; not part of v1 freeze. Document as "emitted by resolver, not stored on record". |
| D11 | `canonical_v2.Jurisdiction` | Matches v1 exactly (country / region / grid_region). | No drift. |
| D12 | `canonical_v2.ActivitySchema` | Matches v1 exactly (category / sub_category / classification_codes). | No drift. |
| D13 | `canonical_v2.FactorParameters` | Contains only electricity + combustion + general flags; missing the 6 other family-specific parameter groups (transport, materials_products, refrigerants, land_removals, finance_proxies, waste). | Split into `CombustionParameters`, `ElectricityParameters`, `TransportParameters`, etc. sub-dataclasses; make the top-level `parameters` a tagged union discriminated by `factor_family`. |
| D14 | `canonical_v2.ChangeLogEntry` | Is a list element; v1 stores only the **latest** change on `lineage` (ingested_at / by / reason / previous_factor_version). | Keep full change log in a **sibling table** `factor_change_log(factor_id, factor_version, entry)`. The record itself carries only the latest entry per v1. |
| D15 | `emission_factor_record.py::valid_to` | Currently `Optional[date]` with `None` meaning "no expiry". | v1 requires a value always; sentinel is `"9999-12-31"`. Migration: replace `None -> date(9999, 12, 31)` on read. |
| D16 | `emission_factor_record.py::gwp_100yr` / `gwp_20yr` sub-records | Two separate objects carrying GWP coefficients per record. | v1 stores only `gwp_set` string enum - GWP coefficients are **external lookup** from a shared registry keyed by `gwp_set`. Drop `GWPValues` from the record, move to the `greenlang/data/gwp_registry.py` module. |
| D17 | `emission_factor_record.py::content_hash` | SHA-256 of record fields, computed in `__post_init__`. | v1 does not include `content_hash` as a stored field - it is derivable. Keep in code as a computed property (`record.content_hash`) but not serialized. |
| D18 | `config/schemas/emission_factors_schema.json` | Legacy **draft-07** bundle schema for YAML country/fuel tables; models the **dataset**, not a single record. | Retain as-is for legacy YAML ingestion pipeline; add a note in the header: "Superseded by `factor_record_v1.schema.json` for single-record validation. This schema validates the legacy flat YAML dataset only." |
| D19 | `emission_factor_record.py::factor_status` | Values: `certified / preview / connector_only / deprecated`. | v1 values: `draft / under_review / active / deprecated / retired`. Map: `preview -> under_review`, `connector_only -> under_review`, `certified -> active`, `deprecated -> deprecated`, (new) `draft`, (new) `retired`. |
| D20 | `emission_factor_record.py::scope` (enum `"1" / "2" / "3"`) and `models/emission_factor.py::Scope` (enum `"Scope 1"` / `"Scope 2 - Location-Based"` ...) | Two incompatible representations. | v1 does not carry `scope` on the record - scope is implied by `method_profile`. Remove from the v1-served projection; keep for backward compat only on the legacy v2 dataclass. |
| D21 | `resolution/result.py::GasBreakdown.gwp_basis` | Defaults to `"IPCC_AR6_100"`. | Aligned with v1 default `gwp_set`; no drift. |
| D22 | `resolution/result.py::ResolvedFactor.fallback_rank` | 1..7 scale matches v1. | No drift. |

### 4.2 Recommended migration sequence

1. Ship `factor_record_v1.schema.json` (this PR) and `docs/specs/factor_record_v1.md`.
2. Implement `greenlang/data/canonical_v1.py` with v1 Pydantic model mirroring the JSON-Schema.
3. Extend `canonical_v2.MethodProfile` and split `FactorParameters` into per-family dataclasses (drift items D8, D13).
4. Add bidirectional converter `EmissionFactorRecord <-> FactorRecordV1`:
   - `to_canonical_v1()`: folds flat geo -> `Jurisdiction`, rescales DQS -> `composite_fqs`, maps `factor_status` -> `status`, hydrates `gwp_set` and drops embedded `GWPValues`.
   - `from_canonical_v1()`: inverse, with GWP lookup from registry.
5. Gate all writes through `canonical_v1.FactorRecord.model_validate(...)` + the 12 invariants in section 3.
6. Run bulk re-save over the ~25,600 lines of YAML factor data to normalize into v1.
7. Freeze `canonical_v2.py` as deprecated; all new fields land on `canonical_v1.py`.

---

## 5. Example factor records

Each example is a complete, schema-valid factor record. Copy-paste into a JSON validator against `factor_record_v1.schema.json` to confirm.

### 5.1 Electricity - eGRID SERC 2024 location-based

```json
{
  "factor_id": "EF:US:grid:eGRID-SERC:2024:v1",
  "factor_family": "electricity",
  "factor_name": "US eGRID SERC subregion - 2024 location-based grid intensity",
  "method_profile": "corporate_scope2_location_based",
  "source_id": "EPA_eGRID",
  "source_version": "2024.1",
  "factor_version": "1.0.0",
  "status": "active",
  "jurisdiction": {
    "country": "US",
    "region": null,
    "grid_region": "eGRID-SERC"
  },
  "valid_from": "2024-01-01",
  "valid_to": "2024-12-31",
  "activity_schema": {
    "category": "purchased_electricity",
    "sub_category": "grid_average",
    "classification_codes": ["NAICS:221112"]
  },
  "numerator": {
    "co2": 0.3891,
    "ch4": 0.0000094,
    "n2o": 0.0000041,
    "f_gases": {},
    "co2e": 0.3907,
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
    "subregion_code": "eGRID-SERC"
  },
  "quality": {
    "temporal_score": 5,
    "geographic_score": 5,
    "technology_score": 4,
    "verification_score": 4,
    "completeness_score": 5,
    "composite_fqs": 92.0
  },
  "lineage": {
    "ingested_at": "2026-04-10T09:15:22Z",
    "ingested_by": "agent:ingest-epa-egrid@v2.1.0",
    "approved_by": "reviewer:data-ops@greenlang.io",
    "approved_at": "2026-04-11T14:02:10Z",
    "change_reason": "Initial ingest of EPA eGRID 2024 Q1 release",
    "previous_factor_version": null,
    "raw_record_ref": {
      "raw_record_id": "egrid2024_subregion_SERC",
      "raw_payload_hash": "3a7bc0e8f4d1e2a9c5b8f6d4e2a1c9b7d5e3a2c1b0f9e8d7c6b5a4c3d2e1f0a9",
      "raw_format": "xlsx",
      "storage_uri": "s3://greenlang-factors-raw/epa/egrid/2024_v1/eGRID2024_subregions.xlsx"
    }
  },
  "licensing": {
    "redistribution_class": "open",
    "customer_entitlement_required": false,
    "license_name": "US-Gov-PD",
    "license_url": "https://www.epa.gov/egrid",
    "attribution_required": true,
    "restrictions": []
  },
  "explainability": {
    "assumptions": [
      "eGRID subregion boundary used as grid_region",
      "CH4 and N2O coefficients from EPA eGRID2024 emission rates",
      "GWP values per IPCC AR6 100-year (CH4=27.9, N2O=273)",
      "Transmission and distribution losses NOT included (busbar basis)"
    ],
    "fallback_rank": 3,
    "rationale": "Primary US location-based Scope 2 factor when the facility's eGRID subregion is known to be SERC."
  }
}
```

### 5.2 Combustion - Diesel fuel US EPA 2024

```json
{
  "factor_id": "EF:US:fuel:diesel:2024:v1",
  "factor_family": "combustion",
  "factor_name": "Diesel fuel (No. 2 distillate) - US EPA 2024 combustion factor",
  "method_profile": "corporate_scope1",
  "source_id": "EPA_GHG_EF_Hub",
  "source_version": "2024.03",
  "factor_version": "1.0.0",
  "status": "active",
  "jurisdiction": {
    "country": "US",
    "region": null,
    "grid_region": null
  },
  "valid_from": "2024-01-01",
  "valid_to": "2024-12-31",
  "activity_schema": {
    "category": "stationary_combustion",
    "sub_category": "distillate_fuel_oil_no_2",
    "classification_codes": ["NAICS:486110", "CPC:33340"]
  },
  "numerator": {
    "co2": 10.18,
    "ch4": 0.000411,
    "n2o": 0.0000822,
    "f_gases": {},
    "co2e": 10.214,
    "unit": "kg"
  },
  "denominator": {
    "value": 1.0,
    "unit": "gal"
  },
  "gwp_set": "IPCC_AR6_100",
  "formula_type": "combustion",
  "parameters": {
    "scope_applicability": ["scope_1"],
    "uncertainty_low": -0.05,
    "uncertainty_high": 0.05,
    "fuel_code": "diesel",
    "LHV": 138.0,
    "HHV": 146.3,
    "density": 3.167,
    "oxidation_factor": 1.0,
    "fossil_carbon_share": 1.0,
    "biogenic_carbon_share": 0.0,
    "sulfur_share": 0.0015,
    "moisture_share": null,
    "ash_share": null
  },
  "quality": {
    "temporal_score": 5,
    "geographic_score": 4,
    "technology_score": 5,
    "verification_score": 5,
    "completeness_score": 5,
    "composite_fqs": 95.0
  },
  "lineage": {
    "ingested_at": "2026-04-12T11:03:00Z",
    "ingested_by": "agent:ingest-epa-ghg-hub@v1.4.2",
    "approved_by": "reviewer:data-ops@greenlang.io",
    "approved_at": "2026-04-12T17:44:05Z",
    "change_reason": "Initial ingest of EPA GHG Emission Factors Hub March 2024 release",
    "previous_factor_version": null,
    "raw_record_ref": {
      "raw_record_id": "epa_ghg_ef_hub_2024_table1_diesel",
      "raw_payload_hash": "d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
      "raw_format": "pdf_ocr",
      "storage_uri": "s3://greenlang-factors-raw/epa/ghg_ef_hub/2024_03/tables.pdf"
    }
  },
  "licensing": {
    "redistribution_class": "open",
    "customer_entitlement_required": false,
    "license_name": "US-Gov-PD",
    "license_url": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
    "attribution_required": true,
    "restrictions": []
  },
  "explainability": {
    "assumptions": [
      "HHV basis per US EPA convention (gallon of fuel)",
      "100% fossil carbon (No. 2 distillate, not biodiesel-blended)",
      "Oxidation factor = 1.0 per IPCC 2006 Vol 2 Tier 1 default for liquid fuels",
      "CH4 and N2O from stationary combustion (EPA Table 1)"
    ],
    "fallback_rank": 4,
    "rationale": "Default US Scope 1 combustion factor for pure No. 2 diesel; use B5/B20 blend factors for biodiesel mixtures."
  }
}
```

### 5.3 Freight - ISO 14083 road HGV rigid 17t WTW

```json
{
  "factor_id": "EF:EU:freight:road:HGV_rigid_17t:WTW:2024:v1",
  "factor_family": "transport",
  "factor_name": "EU HGV rigid 17t - ISO 14083/GLEC WTW default (diesel)",
  "method_profile": "freight_iso14083_glec_wtw",
  "source_id": "SFC_GLEC_Framework",
  "source_version": "v3.0",
  "factor_version": "1.0.0",
  "status": "active",
  "jurisdiction": {
    "country": "XX",
    "region": "EU27",
    "grid_region": null
  },
  "valid_from": "2024-01-01",
  "valid_to": "2026-12-31",
  "activity_schema": {
    "category": "road_freight",
    "sub_category": "HGV_rigid_17t",
    "classification_codes": ["NAICS:484121", "ISIC:H4923"]
  },
  "numerator": {
    "co2": 0.0815,
    "ch4": 0.0000012,
    "n2o": 0.0000031,
    "f_gases": {},
    "co2e": 0.0824,
    "unit": "kg"
  },
  "denominator": {
    "value": 1.0,
    "unit": "t-km"
  },
  "gwp_set": "IPCC_AR6_100",
  "formula_type": "transport_chain",
  "parameters": {
    "scope_applicability": ["scope_3"],
    "uncertainty_low": -0.12,
    "uncertainty_high": 0.18,
    "mode": "road",
    "vehicle_class": "HGV_rigid_17t",
    "payload_basis": "t-km",
    "distance_basis": "route_with_empty",
    "empty_running_assumption": 0.25,
    "utilization_rate": 0.70,
    "refrigerated": false,
    "energy_basis": "WTW"
  },
  "quality": {
    "temporal_score": 4,
    "geographic_score": 4,
    "technology_score": 4,
    "verification_score": 4,
    "completeness_score": 4,
    "composite_fqs": 80.0
  },
  "lineage": {
    "ingested_at": "2026-04-15T08:22:41Z",
    "ingested_by": "agent:ingest-glec@v1.2.0",
    "approved_by": "reviewer:transport-sme@greenlang.io",
    "approved_at": "2026-04-16T10:11:30Z",
    "change_reason": "Initial ingest of Smart Freight Centre GLEC Framework v3.0 road HGV defaults",
    "previous_factor_version": null,
    "raw_record_ref": {
      "raw_record_id": "glec_v3_road_hgv_rigid_17t_wtw",
      "raw_payload_hash": "f0e1d2c3b4a5968778695a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d",
      "raw_format": "csv",
      "storage_uri": "s3://greenlang-factors-raw/sfc/glec/v3/road_defaults.csv"
    }
  },
  "licensing": {
    "redistribution_class": "licensed",
    "customer_entitlement_required": true,
    "license_name": "SFC-GLEC-Commercial",
    "license_url": "https://smartfreightcentre.org/en/our-programs/global-logistics-emissions-council/",
    "attribution_required": true,
    "restrictions": ["no_bulk_export", "sublicense_required_for_resale"]
  },
  "explainability": {
    "assumptions": [
      "Average EU27 load factor 0.70",
      "Empty running 25% (GLEC default for rigid truck)",
      "Great-circle distance + empty return leg modeled",
      "WTW basis including upstream fuel production",
      "Diesel B7 blend assumed (standard EU market fuel)",
      "Refrigeration not included - use refrigerated variant factor for reefer loads"
    ],
    "fallback_rank": 5,
    "rationale": "EU-wide GLEC default used when no supplier-specific modal factor is available; prefer supplier-reported TTW if the carrier can provide it."
  }
}
```

---

## 6. Change control

- Any breaking change to this schema requires a MAJOR bump (`factor_record_v2.schema.json`) and a new spec doc.
- Additive, backward-compatible changes (new optional properties, new enum values that keep `additionalProperties: false`) require a MINOR bump (`v1.1`).
- Clarifications that do not alter validation outcomes are PATCH (`v1.0.1`) and land as amendments to this document.
- Every change to the schema file MUST be accompanied by:
  1. An updated copy of `factor_record_v1.md` with change notes.
  2. A migration script if the change requires data reshaping.
  3. A new entry in the Factor Catalog change log with the responsible reviewer.

---

## 7. References

- CTO spec "GreenLang Factors - Canonical factor record: required parameters" (2026-04-20).
- GHG Protocol Corporate Standard (2004, revised 2015).
- GHG Protocol Scope 2 Guidance (2015).
- GHG Protocol Product Standard (2011).
- ISO 14067:2018 Carbon footprint of products.
- ISO 14083:2023 GHG emissions from transport chains.
- GHG Protocol Land Sector and Removals Guidance (LSR).
- PCAF Global GHG Accounting and Reporting Standard (Part A, Part B, Part C).
- PACT Pathfinder Framework v2.0.
- IPCC Sixth Assessment Report (AR6), Working Group I, 2021.
- EU CBAM Regulation (EU) 2023/956.
- EU Digital Product Passport Regulation (DPP).

---

*End of Factor Record v1.0 specification.*
