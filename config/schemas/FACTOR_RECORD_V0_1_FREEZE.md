# Factor Record v0.1 Alpha — Freeze Note

**Status**: FROZEN
**Freeze date**: 2026-04-25
**Schema $id**: `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json`
**Authority**: GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1, §6.2, §6.3, §18, §19.1

---

## Scope

This schema is the canonical contract for every factor record produced by **GreenLang Factors v0.1 Alpha** (FY27 Q1, internal + 2 design partners). It is the FROZEN, narrower, alpha-only contract — distinct from `factor_record_v1.schema.json`, which is the broader forward schema.

The 6 alpha-source parsers MUST validate every record against this schema:

1. IPCC AR6 minimal Tier 1 defaults (`urn:gl:source:ipcc-ar6`)
2. UK DESNZ / DEFRA GHG Conversion Factors 2025 (`urn:gl:source:defra-2025`)
3. US EPA GHG Emission Factors Hub 2025 (`urn:gl:source:epa-ghg-hub`)
4. US EPA eGRID 2024 (`urn:gl:source:epa-egrid`)
5. India CEA Baseline Database, latest vintage (`urn:gl:source:india-cea-baseline`)
6. EU CBAM default values (`urn:gl:source:eu-cbam-defaults`)

CI gate: see `tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py` (task #4).

---

## Versioning policy

This schema is FROZEN. Within v0.x:

- **Additive only**. New fields may be added at v0.2 or later, never removed within v0.x.
- **No type changes**. A field's JSON Schema type is immutable within v0.x.
- **No renames**. A field name is permanent within v0.x.
- New fields require **methodology-lead approval + CHANGELOG entry**.

v0.2 candidate additions (not yet committed):
- `category` enum extension to include Scope 3 categories (`scope3_c1` through `scope3_c15`) and freight (`freight_road_wtw`, `freight_road_ttw`, `freight_sea`, `freight_air`).
- `gwp_basis` enum extension to include `ar4`, `ar5`, `custom`.
- `uncertainty` promotion from optional to required.

v1.0 GA promotion is **non-substitutable**: it gets a separate `$id` (`factor_record_v1_0.schema.json`). v0.x records and v1.0 records coexist; the resolver picks based on schema declaration.

---

## What is FROZEN at v0.1 (required fields)

```
urn                  factor_pack_urn       category         unit_urn
source_urn           name                  value            geography_urn
gwp_basis            gwp_horizon           vintage_start    vintage_end
resolution           methodology_urn       boundary         licence
citations            published_at          extraction       review
```

**Provenance gate** — `extraction` and `review` are required objects with the following fully-required sub-fields:

- `extraction`: source_url, source_record_id, source_publication, source_version, raw_artifact_uri, raw_artifact_sha256, parser_id, parser_version, parser_commit, row_ref, ingested_at, operator
- `review`: review_status, reviewer, reviewed_at; plus approved_by + approved_at when status=approved; plus rejection_reason when status=rejected

This satisfies CTO doc §19.1 acceptance criterion: *"provenance fields complete for alpha sources"*.

---

## What is INTENTIONALLY OMITTED from v0.1 alpha

Mapped to the existing `factor_record_v1.schema.json` for reference. These fields are present in v1 but **NOT** required in v0.1, because the corresponding feature is out of scope per CTO doc §19.1 "Explicitly out of scope":

| v1 field | Why omitted in v0.1 |
|---|---|
| `activity_schema` | Resolve endpoint not shipped in alpha. |
| `numerator` (per-gas split co2/ch4/n2o/co2e) | Multi-gas split + GWP-set switching deferred to v0.5. Alpha records ship single-value `value` + `gwp_basis: ar6` only. |
| `denominator` (full object) | Alpha collapses to a single `unit_urn` + `value`. |
| `parameters` (factor_family-specific subschema) | No factor-family discriminator in alpha. |
| `quality.composite_fqs` and per-axis pedigree scores | FQS not shipped in alpha. |
| `explainability.assumptions/fallback_rank/rationale` | No `/v1/factors/{urn}/explain` endpoint in alpha. |
| `licensing.redistribution_class`, `licensing.customer_entitlement_required` | No commercial tier / OEM / customer-private factors in alpha. |
| `lineage.previous_factor_version` | Signed receipts not enforced in alpha. |
| Batch / signed-receipts / edition-pinning metadata | All deferred to v0.5+. |

A complete field-by-field mapping is in `factor_record_v0_1_to_v1_map.json`.

---

## Discriminators changed from v1

- **id**: v1 uses `factor_id` with pattern `^EF:...$`. v0.1 uses `urn` with pattern `^urn:gl:factor:...:v\d+$` as the canonical primary id. The legacy `EF:...` is kept as `factor_id_alias` (optional) for backward compatibility with the existing catalog seed inputs but is **not** the primary id in any API or SDK response.
- **category**: v1 has 15-value `factor_family` enum. v0.1 has 9-value narrowed `category` enum.
- **gwp_basis**: v1 allows `ar4|ar5|ar6|custom`. v0.1 is `ar6` only.
- **methodology_urn**: v1 calls this `method_profile` with a fixed enum. v0.1 calls it `methodology_urn` and accepts any URN matching the methodology pattern (registry-driven).

---

## Compatibility with existing catalog

The 6 alpha-source parsers currently emit records with `EF:...` factor_id format. Coercing them to v0.1 alpha requires (handled by task #15 URN module + task #6 backfill):

1. Generate `urn` via `coerce_factor_id_to_urn()` and store the original `EF:...` as `factor_id_alias`.
2. URN-ize `source_id` → `source_urn`.
3. Promote `lineage.raw_record_ref` to a fully-populated `extraction` object — NEW required fields are: `source_url`, `source_record_id`, `source_publication`, `parser_id`, `parser_version`, `parser_commit`, `row_ref`, `ingested_at`, `operator`. Backfill from parser code where possible; flag records as `review.review_status: pending` until methodology lead approves.
4. Compute `raw_artifact_sha256` over the raw fixture; upload to `s3://greenlang-factors-raw/` (or local artefact store at `greenlang/factors/artifacts/`).
5. Set `gwp_basis: ar6` everywhere; reject any IPCC AR4/AR5 record.

---

## Approvals required

| Approver | Decision |
|---|---|
| Methodology Lead | Schema content acceptable for alpha publishers |
| CTO | Freeze date, versioning policy, field omissions vs v1 |
| Legal | Licence-tag enum is enforceable; SPDX vs proprietary distinction is sound |

Once approved, this file becomes the binding contract. Any change requires a new `$id` (v0.2+) or a v1.0 GA cut.
