# Phase 2 Exit Checklist

> **Authority**: CTO Phase 2 brief (2026-04-27).
> **Owner**: Factors Engineering.
> **Reviewers**: CTO, Methodology Lead, Legal, Head of Data.

Every technical box must be ticked and every accountable role must sign off before Phase 2 is declared formally complete and Phase 3 (Resolution + Pricing engines) may start.

> **Engineering verification snapshot**: 2026-04-28. Codex re-verified the local Phase 2 acceptance runner (`10/10 PASS`) and full v0.1 alpha suite (`1659 passed, 15 skipped`) on `master`. This checklist records engineering evidence; it does **not** replace the required human CTO / Methodology / Backend / Head of Data / Test / Legal signatures.

---

## Block 1 — Schema Contract (CTO §2.1)

- [x] `factor_record_v0_1.schema.json` is FROZEN and locked at `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json`.
- [x] Pydantic v2 mirror at `greenlang/factors/schemas/factor_record_v0_1.py` validates the same input set the JSON Schema does. Diff suite is empty.
- [x] Field reference doc `docs/factors/schema/FACTOR_RECORD_V0_1_FIELD_REFERENCE.md` published; every field has meaning, type, required/nullable, example, validation rule.
- [x] CI gate `tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py` is GREEN against the entire v0.1 alpha catalog.
- [x] All 7 required field groups present on every record: identity, value/unit, context, quality, licence, lineage, lifecycle.
- [ ] **Sign-off**: Methodology Lead + CTO.

## Block 2 — URN Compliance (CTO §2.2)

- [x] Public factor IDs: 100% match `urn:gl:factor:...` (audit query attached to PR).
- [x] Lowercase namespace audit: 0 uppercase segments across factor / source / pack / methodology / geography / unit / activity.
- [x] Property-based deterministic-builder tests pass: build → parse = identity, parse → render = identity, for every kind.
- [x] Legacy `EF:...` IDs only present in `factor_aliases.legacy_id`; never primary in API/SDK responses.
- [x] `factor_aliases` table populated with one alias row per legacy `EF:` id present in the catalog.
- [ ] **Sign-off**: Backend Lead + CTO.

## Block 3 — Ontology Coverage (CTO §2.3)

- [x] `geography` table seeded with at minimum: global; ISO-3166-1 country codes covering alpha sources; subregion (eu-27, asean); state_or_province (us-tx, us-ca, in-mh); grid_zone (egrid-rfcw, egrid-serc, egrid-wecc); bidding_zone (de-lu, fr, gb); balancing_authority (caiso, pjm, ercot); basin and tenant types added to the CHECK enum.
- [x] `unit` table seeded with composite-climate units (kgco2e/kwh, /kg, /tkm, /l, /m3, /passenger-km, /usd) + base units (kg, kwh, mj, l, m3, km, tkm, t, usd, eur, gbp, inr); `conversions` JSONB pre-populated for energy/mass/volume/distance.
- [x] `methodology` table seeded with: GHGP scope1, scope2-location, scope2-market, scope3-cat-1..15; IPCC tier 1/2/3 (stationary, mobile, fugitive); EU-CBAM default + actual; epa-egrid-subregion-2024; india-cea-baseline; glec-framework-v3; iso-14083-2023; pcaf-financed-emissions-v1; ecoinvent (cutoff/apos/consequential).
- [x] `activity` table seeded across 14 taxonomies: IPCC, GHGP, HS/CN, CPC, NACE, NAICS/SIC, PACT, freight, CBAM, PCF, refrigerants, agriculture, waste, land-use.
- [x] 100% of v0.1 factors have FK-resolving `geography_urn`, `unit_urn`, `methodology_urn` (and `activity_taxonomy_urn` once promoted in v0.2).
- [x] Invalid ontology references rejected at publish time (negative tests pass).
- [ ] **Sign-off**: Methodology Lead.

## Block 4 — Storage Readiness (CTO §2.4)

- [x] Alembic migrations V500–V504 (or equivalent) merged.
- [x] All 14 canonical tables exist: source, source_artifacts, factor_pack, methodology, geography, unit, activity, factor, factor_aliases, provenance_edges, changelog_events, api_keys, entitlements, release_manifests.
- [x] Rollback DOWN SQL exists for every Phase 2 migration.
- [x] Indexes present for: URN lookup, source_urn, factor_pack_urn, geography_urn, activity_urn, methodology_urn, vintage_start, lifecycle status (review_status='approved').
- [x] Source artefacts stored in versioned object storage (S3 dev: `s3://greenlang-factors-raw/`, local: `greenlang/factors/artifacts/`); each artefact has SHA-256, URI, source_version, ingestion metadata in `source_artifacts`.
- [x] `AlphaFactorRepository` exposes query by URN / source / pack / geography / activity / methodology / vintage / lifecycle.
- [x] Partition strategy ADR `docs/factors/adr/ADR-004-factor-partitioning.md` accepted. **Physical partitioning is FORMALLY DEFERRED to pre-v1.0 scale work** (trigger: row count > 100k OR P95 list-query latency > 100 ms). This is an explicit deferral — Phase 2 ships without partitioned tables.
- [ ] **Sign-off**: Head of Data + Backend Lead.

## Block 5 — Publish-Time Gates (CTO §2.5)

- [x] `greenlang/factors/quality/publish_gates.py` orchestrates 7 gates in order: schema, URN uniqueness, ontology FK, source registry, licence match, provenance completeness, lifecycle status.
- [x] All 9 negative cases REJECT at publish time:
  - [x] missing source_urn
  - [x] invalid unit_urn
  - [x] invalid methodology_urn
  - [x] uppercase URN segment
  - [x] duplicate factor URN
  - [x] missing checksum
  - [x] missing parser_version
  - [x] missing review metadata
  - [x] licence mismatch with source registry
- [x] `gl factors validate <path>` CLI dry-run available.
- [x] No code path can write to `factor` table bypassing the gate (audit via grep + repository contract test).
- [ ] **Sign-off**: GL-SpecGuardian + Backend Lead.

## Block 6 — Schema Evolution (CTO §2.6)

- [x] `docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md` published; CTO + Methodology Lead signatures remain required below.
- [x] Compatibility labels enforced: additive | breaking | deprecated | removed.
- [x] `greenlang/factors/schemas/_version_registry.py` lists every known schema $id.
- [x] CI gate `scripts/ci/check_schema_migration_notes.py` blocks PRs that change `factor_record_*.schema.json` without a CHANGELOG entry.
- [x] Migration-note template at `docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md`.
- [x] `.github/workflows/factors-schema-evolution-check.yml` is **committed to git** and visible to GitHub Actions (untracked workflows do not enforce).
- [ ] **Sign-off**: CTO.

## Block 7 — Tests & Acceptance (CTO §2.7)

- [x] All 10 test suites green:
  - [x] `test_schema_validates_alpha_catalog.py`
  - [x] `test_urn_parse_build_roundtrip.py`
  - [x] `test_urn_uniqueness_db.py`
  - [x] `test_ontology_fk_enforcement.py`
  - [x] `test_alembic_up_down.py`
  - [x] `test_seed_load.py`
  - [x] `test_publish_rejection_matrix.py`
  - [x] `test_api_query_factor_by_urn.py`
  - [x] `test_sdk_fetch_by_urn.py`
  - [x] `test_provenance_checksum.py`
- [x] Coverage ≥ 85% on new modules (`publish_gates.py`, ontology loaders, alias migration code).
- [x] No regressions in Phase 0/1 test suites.
- [ ] **Sign-off**: Test Lead.

---

## Final sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| CTO | Pending human sign-off |  |  |
| Methodology Lead | Pending human sign-off |  |  |
| Backend Lead | Pending human sign-off |  |  |
| Head of Data | Pending human sign-off |  |  |
| Test Lead | Pending human sign-off |  |  |
| Legal | Pending human sign-off |  |  |

When every box is ticked and every role has signed: **Phase 2 is COMPLETE**. Open Phase 3 (Resolution Engine + Pricing) work tracker.
