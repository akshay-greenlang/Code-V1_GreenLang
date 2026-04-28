# Factor Record v0.1 Alpha — Field Reference

> **Authority**: CTO Phase 2 brief §2.1, §6.1 (URN canonical form), §19.1 (alpha provenance acceptance).
> **Schema $id**: `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json`
> **Status**: FROZEN 2026-04-25. Additive changes allowed within v0.x; field rename or removal is a breaking change requiring a new $id (v0.2+).
> **Last amended**: 2026-04-27 — added 5 OPTIONAL Phase 2 contract fields (`activity_taxonomy_urn`, `confidence`, `created_at`, `updated_at`, `superseded_by_urn`). See CHANGELOG anchor `## v0.1 - 2026-04-27 - additive`.
> **Source of truth**: `config/schemas/factor_record_v0_1.schema.json`. The Pydantic mirror at `greenlang/factors/schemas/factor_record_v0_1.py` MUST stay in lock-step (CI gate `tests/factors/v0_1_alpha/phase2/test_pydantic_mirrors_jsonschema.py`).

This document enumerates every field in the FROZEN `factor_record_v0_1` schema, grouped by the nine CTO field groups. Descriptions are pulled verbatim from the JSON Schema's `description` keywords.

Legend:

- **R** = always required (top-level `required` list).
- **C** = conditionally required (only when an enclosing field takes a specific value, enforced by `allOf`/`if-then`).
- **O** = optional.

---

## Group 1 — Identity

Pydantic submodel: `IdentityFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `urn` | string | R | `^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$` | `urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1` | Canonical public id. Format: `urn:gl:factor:<source>:<namespace>:<id>:v<version>`. Namespace AND id segments MUST be lowercase per URN spec (CTO doc §6.1.1). The canonical parser at `greenlang.factors.ontology.urn` rejects uppercase namespace segments (e.g. `:IN:` is invalid; use `:in:`). Globally unique. Never changes. |
| `factor_id_alias` | string \| null | O | `^EF:[A-Za-z0-9_.:-]+$` | `EF:DESNZ:s1_natural_gas_kwh_net_cv:UK:2024:v1` | Optional non-canonical alias retained for backward compatibility with the v1 schema's `EF:` identifier. SDK and API responses MUST treat `urn` as primary. |
| `source_urn` | string | R | `^urn:gl:source:[a-z0-9][a-z0-9-]*$` | `urn:gl:source:desnz-ghg-conversion` | URN of the upstream source (e.g., `urn:gl:source:ipcc-ar6`, `urn:gl:source:defra-2025`, `urn:gl:source:epa-ghg-hub`, `urn:gl:source:epa-egrid`, `urn:gl:source:india-cea-baseline`, `urn:gl:source:eu-cbam-defaults`). MUST resolve to a registered entry in `greenlang/factors/data/source_registry.yaml`. |
| `factor_pack_urn` | string | R | `^urn:gl:pack:[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9._-]*:v[1-9][0-9]*$` | `urn:gl:pack:desnz-ghg-conversion:conversion-factors:v1` | URN of the owning Factor Pack. Format: `urn:gl:pack:<source>:<pack-id>:v<version>`. Every factor belongs to exactly one pack. Upstream dotted versions such as `2024.1` remain in `extraction.source_version`, not in the public pack URN. |

---

## Group 2 — Value & Unit

Pydantic submodel: `ValueUnitFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `value` | number | R | `> 0` (exclusiveMinimum) | `0.23386` | The numeric emission factor value, expressed in the unit identified by `unit_urn`. MUST be > 0. |
| `unit_urn` | string | R | `^urn:gl:unit:[a-zA-Z0-9._/-]+$` | `urn:gl:unit:kgco2e/kwh` | URN of the unit. Examples: `urn:gl:unit:kgco2e/kwh`, `urn:gl:unit:kgco2e/kg`, `urn:gl:unit:kgco2e/tkm`, `urn:gl:unit:kgco2e/l`. |

---

## Group 3 — Context

Pydantic submodel: `ContextFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `name` | string | R | minLength 1, maxLength 200 | `Grid electricity, consumption mix, India` | Human-readable display name (en-US). |
| `description` | string | R | minLength 30, maxLength 2000 | `Emission factor for natural gas — UK, expressed in CO2e per kWh net CV. Boundary follows DESNZ source publication.` | 2-3 sentence description that states the boundary and exclusions. Required for auditor readability. |
| `category` | string | R | enum: `scope1`, `scope2_location_based`, `scope2_market_based`, `grid_intensity`, `fuel`, `refrigerant`, `fugitive`, `process`, `cbam_default` | `fuel` | Narrowed alpha category enum. Scope 3, freight, agricultural, waste, finance proxies are deferred to v0.5+. Note: v1 schema's broader `factor_family` enum is intentionally not adopted in alpha. |
| `geography_urn` | string | R | `^urn:gl:geo:(global\|country\|subregion\|state_or_province\|grid_zone\|bidding_zone\|balancing_authority):[a-zA-Z0-9._-]+$` | `urn:gl:geo:country:in` | URN of the geography. Examples: `urn:gl:geo:country:in`, `urn:gl:geo:country:us`, `urn:gl:geo:country:gb`, `urn:gl:geo:subregion:eu-27`, `urn:gl:geo:grid_zone:egrid-rfcw`, `urn:gl:geo:state_or_province:us-tx`, `urn:gl:geo:global:world`. |
| `methodology_urn` | string | R | `^urn:gl:methodology:[a-z0-9][a-z0-9-]*$` | `urn:gl:methodology:ipcc-tier-1-stationary-combustion` | URN of the methodology. Examples: `urn:gl:methodology:ipcc-tier-1-stationary-combustion`, `urn:gl:methodology:ghgp-corporate-scope2-location`, `urn:gl:methodology:ghgp-corporate-scope2-market`, `urn:gl:methodology:eu-cbam-default`. |
| `boundary` | string | R | minLength 10, maxLength 2000 | `Combustion only — excludes upstream extraction, refining, and transport (well-to-tank).` | Free-text statement of what is included and excluded. Auditor-facing. |
| `resolution` | string | R | enum: `annual`, `monthly`, `hourly`, `point-in-time` | `annual` | Time resolution. Most alpha factors are annual; hourly grid intensity is deferred to v2.5. |
| `activity_taxonomy_urn` | string \| null | O *(R from v0.2)* | `^urn:gl:activity:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*)?$` | `urn:gl:activity:ipcc:1-a-1` | **Phase 2 additive (2026-04-27).** URN of the activity taxonomy entry. Optional in v0.1; required from v0.2. Resolves to `factors_v0_1.activity.urn` (V502). |

---

## Group 4 — Time

Pydantic submodel: `TimeFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `vintage_start` | string (date) | R | ISO 8601 `YYYY-MM-DD` | `2024-01-01` | Earliest date covered by this factor. |
| `vintage_end` | string (date) | R | ISO 8601 `YYYY-MM-DD`; `>= vintage_start` (cross-field check) | `2099-12-31` | Latest date covered by this factor. MUST be >= `vintage_start`. |
| `published_at` | string (date-time) | R | ISO 8601 with timezone | `2026-04-25T07:42:30+00:00` | When this factor record was published to the production catalogue. Immutable after publish. |
| `deprecated_at` | string (date-time) \| null | O | ISO 8601 with timezone | `null` | If deprecated, when. Once set, never reset to null. |

---

## Group 5 — Climate Basis

Pydantic submodel: `ClimateBasisFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `gwp_basis` | string | R | enum: `ar6` | `ar6` | Alpha is AR6-only. AR4/AR5 multi-basis support is deferred to v0.5+. |
| `gwp_horizon` | integer | R | enum: `20`, `100`, `500` | `100` | GWP time horizon in years. 100 is the default for corporate inventories; 20 for short-lived climate forcers; 500 rare. Ships pinned to 100 in alpha unless explicitly required by source. |

---

## Group 6 — Quality

Pydantic submodel: `QualityFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `uncertainty` | object \| null | O (REQUIRED from v0.9+) | shape: `{distribution: 'lognormal'\|'normal'\|'uniform'\|'triangular', mean?: number, stddev?: number, p2_5?: number, p97_5?: number, pedigree?: object}`; `additionalProperties: false` | `{"distribution": "lognormal", "mean": 0.42, "stddev": 0.05}` | Optional in alpha; required from v0.9+. CTO doc §19.1. |
| `uncertainty.distribution` | string | C (when `uncertainty` set) | enum: `lognormal`, `normal`, `uniform`, `triangular` | `lognormal` | Probability distribution shape. |
| `uncertainty.mean` | number | O | — | `0.42` | Distribution mean. |
| `uncertainty.stddev` | number | O | `>= 0` | `0.05` | Standard deviation. |
| `uncertainty.p2_5` | number | O | — | `0.31` | 2.5th percentile. |
| `uncertainty.p97_5` | number | O | — | `0.55` | 97.5th percentile. |
| `uncertainty.pedigree` | object | O | — | `{"reliability": 1, "completeness": 2}` | Pedigree-matrix scoring. |
| `confidence` | number \| null | O | `0 <= confidence <= 1` | `0.85` | **Phase 2 additive (2026-04-27).** Methodology lead's subjective confidence in the factor's accuracy. Distinct from `uncertainty` (which describes the value distribution). |

---

## Group 7 — Licence

Pydantic submodel: `LicenceFields`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `licence` | string | R | minLength 1, maxLength 128 | `OGL-UK-3.0` | SPDX identifier (e.g., `CC-BY-4.0`, `OGL-UK-3.0`, `public-domain-us-gov`) or proprietary tag (e.g., `EU-COMMISSION-CBAM-DEFAULTS`, `IPCC-PUBLIC`). |
| `licence_constraints` | object \| null | O | shape: `{redistribution: 'allowed'\|'attribution'\|'restricted'\|'forbidden', attribution_required: bool, caching_seconds?: integer, notes?: string}` | `{"redistribution": "attribution", "attribution_required": true}` | Optional. |
| `licence_constraints.redistribution` | string | O | enum: `allowed`, `attribution`, `restricted`, `forbidden` | `attribution` | Redistribution policy. |
| `licence_constraints.attribution_required` | boolean | O | — | `true` | Whether downstream consumers must include attribution. |
| `licence_constraints.caching_seconds` | integer | O | — | `86400` | TTL hint for caches. |
| `licence_constraints.notes` | string | O | — | `Source requires backlink in derivative reports.` | Free-text. |

---

## Group 8 — Lineage

Pydantic submodel: `LineageFields`. Includes the nested `extraction` object (12 mandatory sub-fields), `citations` array, optional `tags`, and optional `supersedes_urn`.

### `citations` array

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `citations` | array | R | `minItems: 1`; items have `additionalProperties: false` | `[{"type": "url", "value": "https://..."}]` | DOIs, URLs, publication references, table/section refs. At least one citation required. |
| `citations[].type` | string | R | enum: `doi`, `url`, `publication`, `section`, `table` | `url` | Citation discriminator. |
| `citations[].value` | string | R | minLength 1 | `https://www.gov.uk/...` | The DOI / URL / publication / section / table reference. |
| `citations[].title` | string | O | — | `UK DESNZ GHG Conversion Factors 2025` | Optional. |
| `citations[].page` | integer \| string | O | — | `42` | Optional. |

### `extraction` object — provenance

> Every field MANDATORY in alpha (CTO doc §19.1: "provenance fields complete for alpha sources"). `additionalProperties: false`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `extraction.source_url` | string (uri) | R | URI format | `https://www.gov.uk/government/publications/...` | Canonical landing URL for the source. |
| `extraction.source_record_id` | string | R | minLength 1, maxLength 256 | `sheet=Stationary_Combustion;row=42` | Stable identifier of the row/cell within the source artefact. |
| `extraction.source_publication` | string | R | minLength 1 | `UK DESNZ GHG Conversion Factors 2025 - Full Set` | Title or DOI of the source publication. |
| `extraction.source_version` | string | R | minLength 1 | `2025.1` | Upstream version pin (e.g., `2025.1` for DEFRA, `2024.0` for eGRID, `AR6-WG3-Annex-III` for IPCC, `CBAM-Reg-2023/956-Annex-III`). Required for reproducibility. |
| `extraction.raw_artifact_uri` | string (uri) | R | URI format | `s3://greenlang-factors-raw/desnz/2025/conversion-factors-full.xlsx` | S3-compatible URI to the immutable raw artefact (PDF/CSV/XLSX/JSON). |
| `extraction.raw_artifact_sha256` | string | R | `^[a-f0-9]{64}$` | `02f6fe55e8c5779443143f86cce9edf87cd4848dd199d12d10163ba6c8b4768c` | SHA-256 (lowercase hex) over the raw artefact bytes. |
| `extraction.parser_id` | string | R | minLength 1 | `greenlang.factors.ingestion.parsers.desnz_uk` | Dotted module path of the parser. |
| `extraction.parser_version` | string | R | `^(0\|[1-9]\d*)\.(0\|[1-9]\d*)\.(0\|[1-9]\d*)(?:-[A-Za-z0-9.-]+)?$` (semver) | `0.1.0` | Semver of the parser at extraction time. |
| `extraction.parser_commit` | string | R | `^[a-f0-9]{7,40}$` | `72154a93917d98c30f8689fe6ad0814611736426` | Git commit SHA of the parser code. |
| `extraction.row_ref` | string | R | minLength 1 | `Sheet=Refrigerants; Row=12; Column=GWP_AR6` | Sheet/table/page/row reference within the raw artefact. |
| `extraction.ingested_at` | string (date-time) | R | ISO 8601 with timezone | `2026-04-25T07:42:30Z` | When the parser ingested this record. |
| `extraction.operator` | string | R | `^(bot:[a-z0-9_.-]+\|human:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})$` | `bot:parser_desnz_ghg_conversion` | Identity of who/what ran the extraction. `bot:parser_<id>` for automated ingest; `human:<email>` for manual hot-fix. |

### Optional lineage fields

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `tags` | array of string | O | each tag minLength 1, maxLength 64 | `["egrid", "us", "2024", "subregion"]` | Optional indexed tags. |
| `supersedes_urn` | string \| null | O | `^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[A-Za-z0-9._-]+){2,4}:v[1-9][0-9]*$` | `urn:gl:factor:epa-egrid:subregion-rfcw:2023-average:v1` | URN of the prior factor this record replaces. Set on revisions. |
| `superseded_by_urn` | string \| null | O | `^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$` | `urn:gl:factor:epa-egrid:subregion-rfcw:2025-average:v1` | **Phase 2 additive (2026-04-27).** Reverse pointer: when set, this factor has been superseded by the named URN. Inverse of `supersedes_urn`. Set on the prior factor when a correction is issued. |

---

## Group 9 — Lifecycle (Review)

Pydantic submodel: `LifecycleFields`. Wraps the nested `review` object.

> Conditional rules (`allOf`):
>
> - When `review.review_status == "approved"`: `review.required` extends to include `approved_by` AND `approved_at`.
> - When `review.review_status == "rejected"`: `review.required` extends to include `rejection_reason`.
>
> The Pydantic mirror enforces these via a `model_validator(mode="after")` on `ReviewMetadata`.

| Field | Type | Required | Pattern / Enum | Example | Notes |
|---|---|---|---|---|---|
| `review.review_status` | string | R | enum: `pending`, `approved`, `rejected` | `approved` | Lifecycle. Only `approved` records are visible in production. `pending` lives in staging namespace; `rejected` kept for audit. |
| `review.reviewer` | string | R | `^human:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` | `human:methodology-lead@greenlang.io` | Email of the methodology lead or delegated reviewer. |
| `review.reviewed_at` | string (date-time) | R | ISO 8601 with timezone | `2026-04-25T07:42:30Z` | When the review action was recorded. |
| `review.approved_by` | string | C (when `review_status == "approved"`) | `^human:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` | `human:methodology-lead@greenlang.io` | Email of the approver. REQUIRED when `review_status == "approved"`. |
| `review.approved_at` | string (date-time) | C (when `review_status == "approved"`) | ISO 8601 with timezone | `2026-04-25T07:42:30Z` | When approval was granted. REQUIRED when `review_status == "approved"`. |
| `review.diff_from_source_uri` | string \| null (uri) | O | URI format | `s3://greenlang-factors-raw/desnz/2025/diff-from-2024.json` | Optional URI to a diff report between the parsed record and the prior version. |
| `review.rejection_reason` | string \| null | C (when `review_status == "rejected"`) | — | `Source publication retracted` | Required when `review_status == "rejected"` (free text). |
| `created_at` | string (date-time) \| null | O | ISO 8601 with timezone | `2026-04-20T07:42:30Z` | **Phase 2 additive (2026-04-27).** Wall-clock timestamp when the record was first staged for review. Distinct from `published_at` (when it became visible in production). |
| `updated_at` | string (date-time) \| null | O | ISO 8601 with timezone | `2026-04-24T07:42:30Z` | **Phase 2 additive (2026-04-27).** Wall-clock timestamp of the most recent metadata edit pre-publish. Immutable after `published_at` is set; tracking edits during the staging window only. |

---

## Top-level invariants

These rules apply to the record as a whole and are enforced by both layers (JSON Schema + Pydantic mirror):

| Invariant | Enforcement layer | Notes |
|---|---|---|
| `additionalProperties: false` | JSON Schema; Pydantic `extra="forbid"` (inherited from `GreenLangBase`) | Rejects unknown top-level keys. |
| `vintage_end >= vintage_start` | Pydantic `model_validator(mode="after")` | Schema cannot express cross-field date ordering. |
| `value > 0` | JSON Schema `exclusiveMinimum: 0`; Pydantic `field_validator` | — |
| `gwp_basis == "ar6"` | JSON Schema `enum: ["ar6"]`; Pydantic enum + validator | Alpha is AR6-only. |
| URN pattern + canonical parse | JSON Schema `pattern`; Pydantic `field_validator` calls `greenlang.factors.ontology.urn.parse` | Catches uppercase namespaces the regex cannot easily forbid. |
| Review state requires approver | JSON Schema `allOf/if-then`; Pydantic `model_validator` on `ReviewMetadata` | Two states: `approved` -> `approved_by` + `approved_at`; `rejected` -> `rejection_reason`. |

---

## Cross-references

- **Frozen JSON Schema**: [`config/schemas/factor_record_v0_1.schema.json`](../../../config/schemas/factor_record_v0_1.schema.json)
- **Schema $id**: `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json` (locked).
- **Freeze note**: in-line in the schema's top-level `description` keyword: "FROZEN alpha contract for GreenLang Factors v0.1 Alpha (FY27 Q1)... Frozen 2026-04-25."
- **Pydantic mirror**: [`greenlang/factors/schemas/factor_record_v0_1.py`](../../../greenlang/factors/schemas/factor_record_v0_1.py) — `FactorRecordV0_1` plus the nine `*Fields` submodels.
- **Provenance gate (runtime)**: [`greenlang/factors/quality/alpha_provenance_gate.py`](../../../greenlang/factors/quality/alpha_provenance_gate.py) — the production-side hard gate that runs before every ingestion.
- **Phase 2 plan**: [`docs/factors/PHASE_2_PLAN.md`](../PHASE_2_PLAN.md) §2.1 (this work-stream).
- **CTO doc**: §6.1 (URN canonical form) and §19.1 (alpha provenance acceptance).
- **CI gates**:
  - [`tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py`](../../../tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py) — every catalog record validates against the JSON Schema AND the Pydantic mirror.
  - [`tests/factors/v0_1_alpha/phase2/test_pydantic_mirrors_jsonschema.py`](../../../tests/factors/v0_1_alpha/phase2/test_pydantic_mirrors_jsonschema.py) — the generated schema from `model_json_schema()` matches the frozen file's required-field set, property names, enum values, and patterns.
- **Audit script**: [`scripts/factors/phase2_audit_field_groups.py`](../../../scripts/factors/phase2_audit_field_groups.py) — coverage matrix per field group, used by the KPI dashboard refresh.
