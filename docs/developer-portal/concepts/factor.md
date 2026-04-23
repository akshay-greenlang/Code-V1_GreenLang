# Concept — Factor

A **factor** is a single row in the GreenLang canonical catalog, fully described by the v1 schema at [`config/schemas/factor_record_v1.schema.json`](../../../config/schemas/factor_record_v1.schema.json). Every factor encodes six non-negotiables:

1. Gas-vector storage (CO2 / CH4 / N2O / F-gases kept separately; CO2e is derived under a named `gwp_set`).
2. Immutable `(factor_id, factor_version)` — factors are never overwritten.
3. Explicit `explainability.assumptions[]` and `fallback_rank` — fallback logic is never hidden.
4. A single `licensing.redistribution_class` per record — classes never mix in an API response.
5. Required `valid_from`, `valid_to`, `source_id`, `source_version` — no factor is served without reproducibility metadata.
6. Required `method_profile` — a factor is only returned once bound to a methodology.

## Identity

A factor carries a stable `factor_id` (pattern `^EF:[A-Za-z0-9_.:-]+$`) and a semver `factor_version`. Example: `EF:IN:grid:CEA:FY2024-25:v1` with `factor_version: "1.0.0"`. Any change to `numerator`, `denominator`, `parameters`, or `quality` MUST bump `factor_version`. Supersession is tracked via `lineage.previous_factor_version`.

## Shape

Each record has four functional blocks:

- **Jurisdiction** — `country` (ISO 3166-1 alpha-2, or `"XX"` for global), optional `region` and `grid_region`.
- **Activity schema** — `category`, `sub_category`, and an open set of `classification_codes[]` (e.g., `NAICS:221112`, `CN:7208`).
- **Numerator + denominator** — the emissions-per-activity ratio. Numerator gases are kg per denominator unit; denominator is the activity unit (`kWh`, `gal`, `t-km`, `USD`, ...).
- **Parameters** — a discriminated union keyed by `factor_family` (combustion, electricity, transport, materials_products, refrigerants, land_removals, finance_proxies, waste). See `schema.md`.

## Quality

Five integer component scores (1-5) are combined into a weighted 0-100 composite called **FQS** (Factor Quality Score): `composite_fqs = 20 * (0.25*temporal + 0.25*geographic + 0.20*technology + 0.15*verification + 0.15*completeness)`. See [`concepts/quality_score.md`](quality_score.md).

## Lineage

Every factor carries `ingested_at`, `ingested_by`, `approved_by`, `approved_at`, `change_reason`, and optional `raw_record_ref` (pointing to the raw source file in immutable storage). Active status requires both `approved_by` and `approved_at` to be populated.

## Lifecycle

`status` is one of `draft`, `under_review`, `active`, `deprecated`, `retired`. Production resolver only serves `active` by default. Deprecated rows MUST reference a successor `factor_id` in `explainability.rationale`.

**See also:** [`source`](source.md), [`method_pack`](method_pack.md), [`edition`](edition.md), [API `/resolve`](../api-reference/resolve.md).
