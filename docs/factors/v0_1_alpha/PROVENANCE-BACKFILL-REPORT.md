# v0.1 Alpha Provenance Backfill Report

**Status:** Wave C / TaskCreate #4 (CI gate landed) — input to TaskCreate #6 (backfill)
**Generated:** 2026-04-25 from `tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py`
**Schema:** `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json` (FROZEN)
**Gate:** `greenlang.factors.quality.alpha_provenance_gate.AlphaProvenanceGate`

---

## Headline

| Metric                                  | Value     |
| --------------------------------------- | --------- |
| Alpha sources examined                  | **6 / 6** |
| Total records produced by alpha parsers | **701**   |
| Records passing the AlphaProvenanceGate | **0**     |
| Records failing the AlphaProvenanceGate | **701**   |
| Pass rate                               | **0.00%** |

The CI gate is wired and `xfail(strict=False)` until task #6 lands.

---

## Per-source counts

| source_id                  | parser                                              | total | pass | fail |
| -------------------------- | --------------------------------------------------- | ----- | ---- | ---- |
| `cbam_default_values`      | `cbam_default_values.parse_cbam_default_values`     | 60    | 0    | 60   |
| `desnz_ghg_conversion`     | `desnz_uk.parse_desnz_uk`                           | 196   | 0    | 196  |
| `egrid`                    | `egrid.parse_egrid`                                 | 79    | 0    | 79   |
| `epa_hub`                  | `epa_ghg_hub.parse_epa_ghg_hub`                     | 84    | 0    | 84   |
| `india_cea_co2_baseline`   | `india_cea.parse_india_cea_rows`                    | 38    | 0    | 38   |
| `ipcc_2006_nggi`           | `ipcc_defaults.parse_ipcc_defaults`                 | 244   | 0    | 244  |

---

## Top failure modes (across all 701 records)

Every record fails for the same reason: the alpha-source parsers emit
records in the **v1 catalog shape** (`factor_id`, `vectors`, `provenance`,
`license_info`, `dqs`, `valid_from`, ...), but the v0.1 alpha schema
(`factor_record_v0_1.schema.json`) requires the **v0.1 shape**
(`urn`, `source_urn`, `factor_pack_urn`, `name`, `description`,
`category`, `value`, `unit_urn`, ..., `extraction`, `review`).

| Failure mode                           | Count  | Notes                                                                       |
| -------------------------------------- | ------ | --------------------------------------------------------------------------- |
| `schema[additionalProperties]`         | 701    | Every record carries v1 keys (`factor_id`, `fuel_type`, ...) that v0.1 forbids |
| `missing[urn]`                         | 701    | v1 uses `factor_id`; v0.1 requires canonical `urn:gl:factor:...`            |
| `missing[source_urn]`                  | 701    | v1 stores source inside `provenance`; v0.1 requires top-level `source_urn`  |
| `missing[factor_pack_urn]`             | 701    | New in v0.1 — derived from source + version (per `factor_record_v0_1_to_v1_map.json`) |
| `missing[extraction]`                  | 701    | Provenance block (parser_commit, raw_artifact_sha256, operator, ingested_at) |
| `missing[review]`                      | 701    | Review block (review_status, reviewer, reviewed_at, approved_by/at)         |
| `missing[gwp_basis]` (alpha layer)     | 701    | Must be exactly `"ar6"` in alpha; parsers emit `gwp_100yr` instead          |

(Plus 14 more `missing[<field>]` rows: `name`, `description`, `category`,
`value`, `unit_urn`, `gwp_horizon`, `geography_urn`, `vintage_start`,
`vintage_end`, `resolution`, `methodology_urn`, `boundary`, `licence`,
`citations`, `published_at` — each at 701 / 701.)

---

## Per-source backfill work (input to task #6)

Each of the 6 alpha parsers needs the same translation pass. The work is
identical in shape; the values differ per source.

### 1. `cbam_default_values` — 60 records

* Lift `factor_id` (`EF:CBAM:steel:CN:2024:v1`) into `urn:gl:factor:eu-cbam-defaults:<sector>:<country>-<year>:v1` via `coerce_factor_id_to_urn`.
* Set `source_urn = "urn:gl:source:cbam-default-values"`.
* Add `factor_pack_urn = "urn:gl:pack:eu-cbam-defaults:cbam-annex-iv:2024.1"`.
* Map `vectors` -> top-level `value`, `unit_urn`.
* Project `provenance` block -> `extraction` (commit, sha256, parser_id, parser_version, operator).
* Stamp `review = {review_status: "pending", reviewer: "human:methodology-lead@greenlang.io", reviewed_at: <ingest_ts>}` until methodology lead approves.
* Force `gwp_basis = "ar6"`, `gwp_horizon = 100`.
* Set `category = "cbam_default"`.

### 2. `desnz_ghg_conversion` — 196 records

* Same translation as above; `source_urn = "urn:gl:source:desnz-ghg-conversion"`.
* `factor_pack_urn = "urn:gl:pack:desnz-ghg-conversion:conversion-factors:2024.1"`.
* `category` mapping: `electricity` -> `grid_intensity`, `combustion` -> `fuel`, `refrigerant` -> `refrigerant`, fugitive -> `fugitive`.

### 3. `egrid` — 79 records

* `source_urn = "urn:gl:source:egrid"`.
* `factor_pack_urn = "urn:gl:pack:epa-egrid:subregion-rates:2022.1"`.
* `category = "scope2_location_based"` for subregion rows; `grid_intensity` for state/national rollups.
* eGRID rows are pre-converted from lb/MWh -> kg/kWh — preserve the `unit_urn = "urn:gl:unit:kgco2e/kwh"`.

### 4. `epa_hub` — 84 records

* `source_urn = "urn:gl:source:epa-hub"`.
* `factor_pack_urn = "urn:gl:pack:epa-hub:emission-factors-hub:2024.1"`.
* `category = "fuel"` for stationary/mobile combustion rows; `scope2_location_based` for the electricity section.

### 5. `india_cea_co2_baseline` — 38 records

* Parser currently returns `EmissionFactorRecord` instances; `_record_to_dict` already calls `.to_dict()`.
* `source_urn = "urn:gl:source:india-cea-co2-baseline"`.
* `factor_pack_urn = "urn:gl:pack:india-cea-baseline:national-grid:v20.0"`.
* `category = "grid_intensity"`.
* `geography_urn = "urn:gl:geo:country:in"` (or regional grid URNs for sub-grids).

### 6. `ipcc_2006_nggi` — 244 records

* `source_urn = "urn:gl:source:ipcc-2006-nggi"`.
* `factor_pack_urn = "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:2019.1"` (2019 Refinement vintage).
* `category = "fuel"` for combustion sections; `process` for IPPU; `fugitive` for fugitive sections.
* `methodology_urn = "urn:gl:methodology:ipcc-tier-1-stationary-combustion"` (or the relevant Vol/Chap-specific tier-1 methodology).

---

## Recommended path for task #6

The translation is mechanical and identical across all 6 parsers. Two
implementation options:

1. **Per-parser update** — patch each of the 6 parser modules to emit
   v0.1 shape directly (highest fidelity, biggest blast radius).
2. **Adapter layer** — keep parsers emitting v1 shape; introduce a
   single `v1_to_v0_1_translator.py` that runs after the parser and
   produces v0.1 records, driven by `factor_record_v0_1_to_v1_map.json`
   (smallest blast radius, isolates the change to one module).

Option 2 is recommended for alpha — it lets task #6 land independently
and keeps the existing v1 catalog seed working unchanged.

When #6 lands:

1. Remove `@pytest.mark.xfail` from
   `tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py`.
2. The test must then pass (0 failures) for all 6 alpha sources.
3. Re-generate this report — pass count must read 701 / 701.

---

## How to reproduce the numbers

```bash
pytest tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py \
    -v --no-header --no-cov \
    --log-cli-level=WARNING -o log_cli=true
```

The aggregate `test_alpha_catalog_summary_can_be_produced` test logs
the per-source table and the top failure modes at WARNING level. Per-source
detail is also visible in the parametrised `xfail` cases when they are
de-xfailed locally.
