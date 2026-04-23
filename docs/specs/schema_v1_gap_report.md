# Factor Record v1 Freeze — Gap Report

**Status:** FROZEN (W1-3 deliverable)
**Frozen date:** 2026-04-23
**Owner:** CTO, GreenLang Factors
**Scope:** Drift between current Python code and the v1 JSON-Schema surface.

Referenced artefacts:

- Python dataclass (read-only for Wave 1): `C:\Users\aksha\Code-V1_GreenLang\greenlang\data\emission_factor_record.py`
- CTO canonical extensions (Phase F1): `C:\Users\aksha\Code-V1_GreenLang\greenlang\data\canonical_v2.py`
- Catalog repository: `C:\Users\aksha\Code-V1_GreenLang\greenlang\factors\catalog_repository.py`
- Frozen main schema: `C:\Users\aksha\Code-V1_GreenLang\config\schemas\factor_record_v1.schema.json`
- New source schema: `C:\Users\aksha\Code-V1_GreenLang\config\schemas\source_object_v1.schema.json`
- New category schemas: `C:\Users\aksha\Code-V1_GreenLang\config\schemas\categories\*.schema.json`
- Field dictionary: `C:\Users\aksha\Code-V1_GreenLang\docs\specs\factor_record_v1.md`

---

## 0. How to read this report

Columns in every drift table:

| Col | Meaning |
|-----|---------|
| ID | Stable identifier for the drift item (referenced by Wave 2 tickets). |
| Code symbol | Python file + dataclass/field. |
| Spec field | JSON-Schema field path. |
| Drift kind | `extra_in_code` / `missing_in_code` / `type_mismatch` / `enum_mismatch` / `category_params_gap` / `freeze_risk`. |
| Migration | Concrete code change required in Wave 2. |
| LOC | Rough edit-size estimate (lines added / modified). |
| Risk | Probability of breaking existing YAML ingests or method packs. L/M/H. |

---

## 1. Fields present in code but absent from v1 spec

These are fields the current Python dataclass carries that the v1 JSON Schema does **not** surface. They are code-internal or legacy; Wave 2 keeps them as computed properties or dumps them to a sibling table.

| ID | Code symbol | Spec field | Drift kind | Migration | LOC | Risk |
|----|-------------|------------|------------|-----------|-----|------|
| X01 | `EmissionFactorRecord.content_hash` | — | extra_in_code | Keep as `@property`; drop from serialised payload (`to_dict` already does this). Already non-serialised; only update docstring. | 5 | L |
| X02 | `EmissionFactorRecord.created_at` / `updated_at` / `created_by` / `notes` / `tags` | — | extra_in_code | Move to sibling `catalog_factors.audit_meta` JSON column. Keep on dataclass for backward compat, exclude from `to_canonical_v1()`. | 30 | L |
| X03 | `EmissionFactorRecord.compliance_frameworks` | — | extra_in_code | Replace with `method_profile` (single value) + `activity_tags` (framework IDs). Derive when projecting. | 25 | M |
| X04 | `EmissionFactorRecord.biogenic_flag` | — | extra_in_code | Superseded by `combustion.parameters.biogenic_carbon_share > 0`. Keep as derived prop. | 10 | L |
| X05 | `EmissionFactorRecord.reference_temperature_c` / `pressure_bar` / `moisture_content_pct` / `ash_content_pct` | — | extra_in_code | Fold into `combustion.parameters` (moisture_share, ash_share already in v1). Temperature / pressure become free-form metadata on `parameters`. | 40 | M |
| X06 | `GHGVectors.GWP_VALUES` / `DECOMPOSITION_RATIOS` class constants | — | extra_in_code | Move to `greenlang/data/gwp_registry.py` (spec §D16). Dataclass should not carry lookup tables. | 200 | M |
| X07 | `EmissionFactorRecord.heating_value_basis` (HHV / LHV enum) | — | extra_in_code | Redundant with `combustion.parameters.lhv_mj_per_unit` / `hhv_mj_per_unit`. Remove top-level; keep on legacy dataclass only. | 15 | L |
| X08 | `EmissionFactorRecord.scope` (Scope enum `"1"/"2"/"3"`) | — | extra_in_code | v1 removes top-level `scope`; scope is implied by `method_profile` + `parameters.scope_applicability`. Migration: project existing `scope.value` into `parameters.scope_applicability = ["scope_" + scope.value]`. | 20 | M |
| X09 | `EmissionFactorRecord.boundary` (COMBUSTION / WTT / WTW / …) | — | extra_in_code | Move to `transport.parameters.wtw_ttw_label` for transport factors and `materials.parameters.boundary` for materials. Combustion factors drop it (always direct). | 30 | M |
| X10 | `GWPValues.co2e_total` computed field | — | extra_in_code | Keep as derived property on the v1 Pydantic model; never stored. | 5 | L |

---

## 2. Fields present in v1 spec but absent from code

These are the real Wave 2 work items.

| ID | Spec field | Code gap | Migration | LOC | Risk |
|----|------------|----------|-----------|-----|------|
| M01 | `factor_id` pattern `^EF:[A-Za-z0-9_.:-]+$` | `__post_init__` only checks `startswith("EF:")` | Tighten validator to the full regex; emit test fixture of rejects. | 8 | L |
| M02 | `factor_family` (15-value enum) | Optional, untyped (`Optional[str]`). | Introduce `canonical_v1.FactorFamily` Enum mirroring schema; make field required on the v1 model; fall back to `"combustion"` only during legacy YAML read. | 25 | M |
| M03 | `method_profile` (14-value enum) | `canonical_v2.MethodProfile` has **12** values; spec has 14. Missing: `product_iso14067`, `product_pact`, `freight_iso14083_glec_wtw`, `freight_iso14083_glec_ttw`, `eu_dpp_battery`, `eu_dpp_textile` (spec), plus v2 has `PRODUCT_CARBON` / `FREIGHT_ISO_14083` / `EU_DPP` consolidations not in spec. | Expand enum to the 14 spec values; add `canonical_v1.METHOD_PROFILE_ALIASES` so legacy `product_carbon` / `freight_iso_14083` / `eu_dpp` / `eu_dpp_battery` / `india_ccts` rows keep resolving. | 40 | M |
| M04 | `source_version` top-level field | `provenance.version` or `source_release` / `release_version`; 3 aliases. | Consolidate: `to_canonical_v1()` picks first non-null of `source_release -> release_version -> provenance.version`. Emit one-time warning if multiple disagree. | 20 | M |
| M05 | `jurisdiction {country, region, grid_region}` | Flat `geography`, `region_hint`, no grid_region field. | Add converter: `country = geography` (when len==2), else ISO lookup table; `region = region_hint`; `grid_region = None` unless factor_family=='electricity' and `subregion_code` present. | 40 | M |
| M06 | `valid_to` required | `Optional[date] = None`; `None` means "no expiry". | Replace `None` with `date(9999, 12, 31)` sentinel on serialise; accept either on deserialise. | 10 | L |
| M07 | `activity_schema {category, sub_category, classification_codes[]}` | Only `fuel_type` (string) + `activity_tags` (list). | Add `ActivitySchema` sub-dataclass; derive `category` from `fuel_type` via the existing taxonomy map; classification_codes from the existing `sector_tags` list. | 60 | M |
| M08 | `numerator {co2, ch4, n2o, f_gases, co2e, unit}` | `vectors` object has fields `CO2`/`CH4`/`N2O`/`HFCs`/`PFCs`/`SF6`/`NF3`/`biogenic_CO2`. Schema currently folds HFCs/PFCs/SF6/NF3 under `f_gases` dict. Spec brief also lists `biogenic_co2` at this level. | Map at serialise: `f_gases = { "HFC-mix": HFCs, "PFC-mix": PFCs, "SF6": SF6, "NF3": NF3 }` (null-safe); add `biogenic_co2` top-level. Validator recomputes `co2e` against `gwp_set` with 0.1% tolerance. | 50 | H |
| M09 | `denominator {value, unit}` | Flat `unit: str`, no `value`. | Introduce `Denominator` sub-dataclass; `value` defaults to 1.0; unit migrated verbatim. | 20 | L |
| M10 | `gwp_set` (enum) | Nested on `GWPValues.gwp_set` (AR6_100 / AR6_20 / AR5_100 / AR5_20 / AR4_100 / SAR_100). Spec enum: `IPCC_AR4_100 / IPCC_AR5_100 / IPCC_AR5_20 / IPCC_AR6_100 / IPCC_AR6_20 / Kyoto_SAR_100`. | Rename `IPCC_SAR_100 -> Kyoto_SAR_100` (value mapping); prefix all with `IPCC_` (already done in code enum); drop `gwp_20yr` from the top-level record in v1 (sibling record per §D16). | 30 | M |
| M11 | `formula_type` (enum) | Optional `str`. | Introduce `canonical_v1.FormulaType` mirroring `canonical_v2.FormulaType` (already 1:1 with spec). | 15 | L |
| M12 | `quality.{temporal_score, geographic_score, technology_score, verification_score, completeness_score, composite_fqs}` (0-100) | `DataQualityScore` with `temporal / geographical / technological / representativeness / methodological` (1-5) and `overall_score`. Dimension names differ; scale differs (1-5 → 0-100). | Rename: `geographical→geographic`, `technological→technology`, `representativeness→completeness`, `methodological→verification`. Compute `composite_fqs = 20 * (0.25*T + 0.25*G + 0.20*Tech + 0.15*V + 0.15*C)`. Keep legacy names as aliases in `from_dict`. | 50 | M |
| M13 | `lineage.{ingested_at, ingested_by, approved_by, approved_at, change_reason, previous_factor_version, raw_record_ref}` | `provenance.*` (source_org, source_publication, source_year, methodology, url, doi, version, supersedes) has overlapping but not identical shape; `change_log: List[ChangeLogEntry]` holds history. No `ingested_at` / `ingested_by` / `change_reason` / `approved_by` / `approved_at` as first-class. | Promote to `Lineage` sub-dataclass; fall back: `ingested_at = created_at`, `ingested_by = created_by`, `change_reason = change_log[-1].change_reason or "Initial ingest"`, `previous_factor_version = change_log[-1].previous_factor_version`, `approved_*` default null until QA flow populates them. `raw_record_ref` maps to `canonical_v2.RawRecordRef`. | 80 | H |
| M14 | `licensing.redistribution_class` (4 enum values in spec prompt vs 4 in frozen schema — see §5) | Dual system: boolean flags on `LicenseInfo` + 5-value `RedistributionClass` enum (has `OEM_REDISTRIBUTABLE`). | Keep 4 values for the v1-served projection: `open / restricted / licensed / customer_private` (frozen schema) OR `open / licensed_embedded / customer_private / oem_redistributable` (user brief, also used by source object schema). See §5 for the freeze decision. Boolean flags become derived. | 35 | H |
| M15 | `licensing.customer_entitlement_required` | Not in code. | Derive from `redistribution_class`: `True` when class ∈ {restricted, licensed, customer_private}`. | 10 | L |
| M16 | `licensing.attribution_text` | Not in code (brief asks for it; frozen schema also lacks it). | Add field to `LicenseInfo`. Default = source's `attribution_text` from source_object registry. | 15 | L |
| M17 | `explainability.assumptions[]` / `fallback_rank` / `audit_text` / `replacement_pointer` | `canonical_v2.Explainability` has assumptions + fallback_rank + rationale. Missing: `audit_text` (spec brief), `replacement_pointer` (spec brief). | Rename `rationale -> audit_text`; add `replacement_pointer: Optional[str]` (points to successor factor_id when status='deprecated'). Keep `rationale` as alias. | 20 | L |
| M18 | `status` enum (brief lists: active / certified / preview / connector_only / deprecated / superseded / private; frozen schema: draft / under_review / active / deprecated / retired) | Code has `certified / preview / connector_only / deprecated`. | See §5 for the freeze decision. Recommended: keep frozen-schema values as canonical; add compat mapping: `certified→active`, `preview→under_review`, `connector_only→under_review`. | 30 | H |
| M19 | `parameters` as discriminated union on `factor_family` | `canonical_v2.FactorParameters` only covers electricity + combustion + general flags. 6 other families unrepresented. | Split into 7 per-family dataclasses (`CombustionParameters`, `ElectricityParameters`, `TransportParameters`, `MaterialsProductsParameters`, `RefrigerantsParameters`, `LandRemovalsParameters`, `FinanceProxiesParameters`, `WasteParameters`) mirroring `config/schemas/categories/*.schema.json` exactly. | 450 | H |
| M20 | GWP coefficients external lookup | `GWPValues` sub-object embedded on every record (CH4_gwp, N2O_gwp, HFCs_gwp, ...) | Remove from stored v1 payload; keep `gwp_set` string enum only. Create `greenlang/data/gwp_registry.py` with `lookup(gwp_set, gas) -> float`. All 6 enum rows populated from the in-file GWP_VALUES constant. | 250 | M |

---

## 3. Type / enum mismatches on shared fields

| ID | Field | Code type | Spec type | Migration | LOC | Risk |
|----|-------|-----------|-----------|-----------|-----|------|
| T01 | `factor_status` | free string with default `"certified"` | enum: draft / under_review / active / deprecated / retired (frozen schema) | Enforce enum on write; write compat migration mapping `{certified, preview, connector_only}` to {active, under_review, under_review}. Emit deprecation warning. | 40 | H |
| T02 | `redistribution_class` / `license_class` (two parallel fields) | `RedistributionClass` enum (5 values) + free-text `license_class`. | 4-value enum (frozen schema) | Collapse to single `licensing.redistribution_class`; drop `license_class` (redundant). See §5 for 4-value decision. | 30 | H |
| T03 | `GWPSet.IPCC_SAR_100` | string `"IPCC_SAR_100"` | spec enum member `"Kyoto_SAR_100"` | Rename enum value; keep `IPCC_SAR_100` as alias for one release (v1.0 -> v1.1 sunset). | 10 | M |
| T04 | `DataQualityScore.*` (1-5 ints, equal-weighted mean) | 1-5 integers, dimensions `temporal/geographical/technological/representativeness/methodological` | 1-5 integers, weighted, different dim names | See M12. Rename + recompute composite. | (in M12) | M |
| T05 | `valid_to` | `Optional[date]` with `None` sentinel | required `date` with `9999-12-31` sentinel | See M06. | (in M06) | L |
| T06 | `scope` | enum `"1"/"2"/"3"` | not present; replaced by `parameters.scope_applicability[]` of `scope_1 / scope_2 / scope_3 / scope_4` | Projection: `["scope_" + scope.value]`; inverse: parse first matching member. | 15 | M |
| T07 | `boundary` | enum (combustion / WTT / WTW / cradle_to_gate / cradle_to_grave) | split across `transport.parameters.wtw_ttw_label` and `materials.parameters.boundary`; combustion drops it | Routing map per `factor_family`. | 20 | M |
| T08 | `methodology` (Methodology enum) | IPCC_Tier_1 / Tier_2 / Tier_3 / direct_measurement / lifecycle_assessment / hybrid / spend_based | Not present at top level. Captured implicitly via `formula_type` + `explainability.assumptions`. | Drop from v1 projection; keep on legacy dataclass for back-refs. | 10 | L |
| T09 | `SourceProvenance.citation` (computed) | f"{org} ({year}). {pub}." | `source_object.citation_text` (stored) | Move citation ownership to `source_object_v1`; factor record only references `source_id` + `source_version`. | 15 | L |
| T10 | `content_hash` | SHA-256 of a subset of fields | Not in v1 record (derived on demand) | Already non-serialised; leave code as-is. | 0 | L |

---

## 4. Category parameter schemas not represented in code

The v1 spec defines 7 category-specific `parameters` sub-schemas (combustion, electricity, transport, materials, refrigerants, land_removals, finance_proxy) + a generic bucket for 7 auxiliary families. Code coverage:

| Category | Schema fields | Code fields today | Gap |
|----------|---------------|-------------------|-----|
| combustion | 10 (fuel_code, lhv, hhv, density, oxidation_factor, fossil/biogenic/moisture/sulfur/ash shares) | Split across `fuel_type` + top-level `heating_value_basis` / `moisture_content_pct` / `ash_content_pct` / `biogenic_flag` | **category_params_gap** — no `CombustionParameters` dataclass. Wave 2: create it. ~60 LOC, M risk. |
| electricity | 7 (electricity_basis, residual_mix_applicable, supplier_specific, transmission_loss_included, certificate_handling, subregion_code, scope_applicability) | `canonical_v2.FactorParameters` has 4 of these (electricity_basis, residual_mix_applicable, supplier_specific, transmission_loss_included). | Missing `certificate_handling`, `subregion_code`. ~25 LOC, L risk. |
| transport | 9 (mode, vehicle_class, payload_basis, distance_basis, empty_running_assumption, utilization_rate, refrigerated, wtw_ttw_label, scope_applicability) | None. | **category_params_gap** — entire dataclass. ~70 LOC, M risk. |
| materials | 7 (boundary, allocation_method, recycled_content_assumption, supplier_primary_data_share, pcr_reference, epd_reference, pact_compatible) | Partial: `EndOfLifeAllocationMethod` + `UsePhaseParameters` exist; no top-level `MaterialsParameters`. | **category_params_gap** — 60 LOC, M risk. Add Cat 11 use_phase_* to the dataclass too. |
| refrigerants | 6 (gas_code, gas_blend_components, leakage_basis, recharge_assumption, recovery_treatment, destruction_treatment) | None. | **category_params_gap** — 70 LOC (blend components are nested). M risk. |
| land_removals | 6 (land_use_category, sequestration_basis, removal_basis, permanence_class, reversal_risk_flag, biogenic_accounting_treatment) | None. | **category_params_gap** — 55 LOC, M risk. |
| finance_proxy | 5 (asset_class, sector_code, intensity_basis, geography, proxy_confidence_class) | None. | **category_params_gap** — 50 LOC, L risk. |
| waste | 3 (treatment_route, methane_recovery_factor, net_calorific_value) | None (schema `oneOf` already allows it). | Add `WasteParameters` dataclass. 30 LOC, L risk. |

Total category-params LOC: ~420 (Python Pydantic/dataclass models) + ~400 (test fixtures + golden files).

---

## 5. Three explicit freeze decisions flagged to the CTO

These are places where the v1 brief handed to this work differs from what was already **on disk and marked FROZEN**. The freeze deliverable records the **canonical choice** for each.

| # | Item | Frozen schema value | Brief value | Recommendation |
|---|------|----------------------|-------------|----------------|
| F1 | `status` enum | `draft / under_review / active / deprecated / retired` (5) | `active / certified / preview / connector_only / deprecated / superseded / private` (7) | **Keep frozen schema values** (5). Map legacy `certified→active`, `preview→under_review`, `connector_only→under_review`. `superseded` collapses into `deprecated` + `explainability.replacement_pointer`. `private` is redundant with `licensing.redistribution_class == "customer_private"`. Rationale: avoids a non-backward-compatible change to an already-frozen schema. |
| F2 | `licensing.redistribution_class` enum | `open / restricted / licensed / customer_private` (4; frozen record schema) | `open / licensed_embedded / customer_private / oem_redistributable` (4; source object brief) | **Support both enumerations as sibling fields**: the record uses the frozen 4-value set; the source object schema uses the brief's 4-value set. `to_canonical_v1()` maps: `licensed_embedded→licensed`, `oem_redistributable→licensed` (plus audit tag), `restricted→restricted`. A reconciliation is punted to v1.1. Rationale: changing the record enum breaks 25 600 lines of YAML; adding a new source-level enum is additive. |
| F3 | `gwp_set` enum | 6 values: `IPCC_AR4_100 / IPCC_AR5_100 / IPCC_AR5_20 / IPCC_AR6_100 / IPCC_AR6_20 / Kyoto_SAR_100` | brief only listed AR4/AR5/AR6 variants | **Keep frozen 6 values** including `Kyoto_SAR_100`. Required for Kyoto-Protocol-anchored methodologies still in ISO 14064-1 Annex B. |

These three items are locked for v1.0 and revisit-able in v1.1.

---

## 6. Recommended migration sequence (Wave 2)

1. **Ticket FR1-A** (L risk, ~40 LOC) — Tighten `factor_id` regex (M01) + add `valid_to` sentinel (M06).
2. **Ticket FR1-B** (M risk, ~30 LOC) — Rename `GWPSet.IPCC_SAR_100 -> Kyoto_SAR_100` with alias (T03).
3. **Ticket FR2** (M risk, ~250 LOC) — Extract `GWP_VALUES` into `gwp_registry.py`; stop embedding `GWPValues` on records (X06, M20).
4. **Ticket FR3** (M risk, ~450 LOC) — Introduce 7 per-family parameter dataclasses (M19 + category_params_gap §4) as `canonical_v1.CombustionParameters` etc.
5. **Ticket FR4** (H risk, ~250 LOC) — Introduce `canonical_v1.FactorRecord` Pydantic model + `to_canonical_v1()` / `from_canonical_v1()` bidirectional converter on `EmissionFactorRecord`. Folds M02, M05, M07, M08, M09, M11, M13, M18, T01, T02, T06, T07.
6. **Ticket FR5** (M risk, ~50 LOC) — Rescale `DataQualityScore` dimensions + composite (M12, T04).
7. **Ticket FR6** (M risk, ~80 LOC) — Consolidate license fields into `canonical_v1.Licensing` (M14, M15, M16, T02).
8. **Ticket FR7** (L risk, ~40 LOC) — Finalise `canonical_v1.Explainability` (M17).
9. **Ticket FR8** (L risk, ~120 LOC) — Introduce `canonical_v1.SourceObject` Pydantic model + `greenlang/factors/data/source_registry.py` loader; wire to existing `source_registry.yaml`.
10. **Ticket FR9** (M risk, ~300 LOC) — Gate all catalog writes through `validate_non_negotiables()` + Draft-2020-12 JSON-Schema validation. Bulk re-save over the 25 600 lines of YAML factors.
11. **Ticket FR10** (L risk) — Freeze `canonical_v2.py` as deprecated; new fields land on `canonical_v1.py`.

**Total estimated Wave 2 effort:** ~1 850 LOC of code + tests, ~3 engineer-weeks. Highest risk cluster is FR4 (it touches every serialise/deserialise path).

---

## 7. Appendix — Full field matrix

The Y-column is the v1 JSON-Schema path. The F-column is the field name in the frozen schema. Code-column lists the symbol in `emission_factor_record.py` or `canonical_v2.py`. A cell of `—` means the drift item above is the authoritative resolution.

| Schema path | Frozen schema | Code location | Drift ID |
|-------------|---------------|---------------|----------|
| `factor_id` | string (EF:…) | `EmissionFactorRecord.factor_id` | M01 |
| `factor_family` | enum (15) | `EmissionFactorRecord.factor_family` (optional string) | M02 |
| `factor_name` | string | `EmissionFactorRecord.factor_name` (optional) | ok |
| `method_profile` | enum (14) | `canonical_v2.MethodProfile` (12) | M03 |
| `source_id` | string | `EmissionFactorRecord.source_id` | ok |
| `source_version` | string | `source_release` / `release_version` / `provenance.version` (3 aliases) | M04 |
| `factor_version` | semver string | `EmissionFactorRecord.factor_version` (optional) | ok |
| `status` | enum (5) | `EmissionFactorRecord.factor_status` (free string) | T01, F1 |
| `jurisdiction.*` | structured | `geography` + `region_hint` (flat) | M05 |
| `valid_from` | date | `EmissionFactorRecord.valid_from` | ok |
| `valid_to` | date (required) | `Optional[date]` | M06, T05 |
| `activity_schema.*` | structured | `fuel_type` + `activity_tags` + `sector_tags` | M07 |
| `numerator.*` | {co2, ch4, n2o, f_gases, co2e, unit} | `GHGVectors` (different field shape) | M08 |
| `denominator.*` | {value, unit} | `unit: str` (flat) | M09 |
| `gwp_set` | enum (6) | `GWPValues.gwp_set` (nested, different name) | M10, T03 |
| `formula_type` | enum (8) | `canonical_v2.FormulaType` (8, 1:1) | M11 |
| `parameters.*` | 8 family-specific shapes | `canonical_v2.FactorParameters` (2 families) | M19 + §4 |
| `quality.*` | 6 metrics (0-100 composite) | `DataQualityScore` (5 metrics, 1-5) | M12, T04 |
| `lineage.*` | 7 lineage fields + raw_record_ref | `SourceProvenance` + `change_log` + audit meta | M13 |
| `licensing.*` | 6 licensing fields | `LicenseInfo` + `RedistributionClass` + `license_class` | M14-M16, T02, F2 |
| `explainability.*` | 4 fields | `canonical_v2.Explainability` (3 fields) | M17 |

---

*End of gap report. This document is the W1-3 freeze deliverable per `GreenLang_Factors_CTO_Master_ToDo.md` task table FR1–FR8.*
