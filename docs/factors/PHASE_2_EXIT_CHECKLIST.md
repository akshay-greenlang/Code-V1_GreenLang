# Phase 2 Exit Checklist

> **Authority**: CTO Phase 2 brief (2026-04-27).
> **Owner**: Factors Engineering.
> **Reviewers**: CTO, Methodology Lead, Legal, Head of Data.

Every box must be ticked + signed off before Phase 2 is declared complete and Phase 3 (Resolution + Pricing engines) may start.

---

## Block 1 — Schema Contract (CTO §2.1)

- [ ] `factor_record_v0_1.schema.json` is FROZEN and locked at `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json`.
- [ ] Pydantic v2 mirror at `greenlang/factors/schemas/factor_record_v0_1.py` validates the same input set the JSON Schema does. Diff suite is empty.
- [ ] Field reference doc `docs/factors/schema/FACTOR_RECORD_V0_1_FIELD_REFERENCE.md` published; every field has meaning, type, required/nullable, example, validation rule.
- [ ] CI gate `tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py` is GREEN against the entire v0.1 alpha catalog.
- [ ] All 7 required field groups present on every record: identity, value/unit, context, quality, licence, lineage, lifecycle.
- [ ] **Sign-off**: Methodology Lead + CTO.

## Block 2 — URN Compliance (CTO §2.2)

- [ ] Public factor IDs: 100% match `urn:gl:factor:...` (audit query attached to PR).
- [ ] Lowercase namespace audit: 0 uppercase segments across factor / source / pack / methodology / geography / unit / activity.
- [ ] Property-based deterministic-builder tests pass: build → parse = identity, parse → render = identity, for every kind.
- [ ] Legacy `EF:...` IDs only present in `factor_aliases.legacy_id`; never primary in API/SDK responses.
- [ ] `factor_aliases` table populated with one alias row per legacy `EF:` id present in the catalog.
- [ ] **Sign-off**: Backend Lead + CTO.

## Block 3 — Ontology Coverage (CTO §2.3)

- [ ] `geography` table seeded with at minimum: global; ISO-3166-1 country codes covering alpha sources; subregion (eu-27, asean); state_or_province (us-tx, us-ca, in-mh); grid_zone (egrid-rfcw, egrid-serc, egrid-wecc); bidding_zone (de-lu, fr, gb); balancing_authority (caiso, pjm, ercot); basin and tenant types added to the CHECK enum.
- [ ] `unit` table seeded with composite-climate units (kgco2e/kwh, /kg, /tkm, /l, /m3, /passenger-km, /usd) + base units (kg, kwh, mj, l, m3, km, tkm, t, usd, eur, gbp, inr); `conversions` JSONB pre-populated for energy/mass/volume/distance.
- [ ] `methodology` table seeded with: GHGP scope1, scope2-location, scope2-market, scope3-cat-1..15; IPCC tier 1/2/3 (stationary, mobile, fugitive); EU-CBAM default + actual; epa-egrid-subregion-2024; india-cea-baseline; glec-framework-v3; iso-14083-2023; pcaf-financed-emissions-v1; ecoinvent (cutoff/apos/consequential).
- [ ] `activity` table seeded across 14 taxonomies: IPCC, GHGP, HS/CN, CPC, NACE, NAICS/SIC, PACT, freight, CBAM, PCF, refrigerants, agriculture, waste, land-use.
- [ ] 100% of v0.1 factors have FK-resolving `geography_urn`, `unit_urn`, `methodology_urn` (and `activity_taxonomy_urn` once promoted in v0.2).
- [ ] Invalid ontology references rejected at publish time (negative tests pass).
- [ ] **Sign-off**: Methodology Lead.

## Block 4 — Storage Readiness (CTO §2.4)

- [ ] Alembic migrations V500–V504 (or equivalent) merged.
- [ ] All 14 canonical tables exist: source, source_artifacts, factor_pack, methodology, geography, unit, activity, factor, factor_aliases, provenance_edges, changelog_events, api_keys, entitlements, release_manifests.
- [ ] Rollback DOWN SQL exists for every Phase 2 migration.
- [ ] Indexes present for: URN lookup, source_urn, factor_pack_urn, geography_urn, activity_urn, methodology_urn, vintage_start, lifecycle status (review_status='approved').
- [ ] Source artefacts stored in versioned object storage (S3 dev: `s3://greenlang-factors-raw/`, local: `greenlang/factors/artifacts/`); each artefact has SHA-256, URI, source_version, ingestion metadata in `source_artifacts`.
- [ ] `AlphaFactorRepository` exposes query by URN / source / pack / geography / activity / methodology / vintage / lifecycle.
- [ ] Partition strategy ADR `docs/factors/adr/ADR-004-factor-partitioning.md` accepted. **Physical partitioning is FORMALLY DEFERRED to pre-v1.0 scale work** (trigger: row count > 100k OR P95 list-query latency > 100 ms). This is an explicit, signed deferral — Phase 2 ships without partitioned tables.
- [ ] **Sign-off**: Head of Data + Backend Lead.

## Block 5 — Publish-Time Gates (CTO §2.5)

- [ ] `greenlang/factors/quality/publish_gates.py` orchestrates 7 gates in order: schema, URN uniqueness, ontology FK, source registry, licence match, provenance completeness, lifecycle status.
- [ ] All 9 negative cases REJECT at publish time:
  - [ ] missing source_urn
  - [ ] invalid unit_urn
  - [ ] invalid methodology_urn
  - [ ] uppercase URN segment
  - [ ] duplicate factor URN
  - [ ] missing checksum
  - [ ] missing parser_version
  - [ ] missing review metadata
  - [ ] licence mismatch with source registry
- [ ] `gl factors validate <path>` CLI dry-run available.
- [ ] No code path can write to `factor` table bypassing the gate (audit via grep + repository contract test).
- [ ] **Sign-off**: GL-SpecGuardian + Backend Lead.

## Block 6 — Schema Evolution (CTO §2.6)

- [ ] `docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md` published, signed by CTO + Methodology Lead.
- [ ] Compatibility labels enforced: additive | breaking | deprecated | removed.
- [ ] `greenlang/factors/schemas/_version_registry.py` lists every known schema $id.
- [ ] CI gate `scripts/ci/check_schema_migration_notes.py` blocks PRs that change `factor_record_*.schema.json` without a CHANGELOG entry.
- [ ] Migration-note template at `docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md`.
- [ ] `.github/workflows/factors-schema-evolution-check.yml` is **committed to git** and visible to GitHub Actions (untracked workflows do not enforce).
- [ ] **Sign-off**: CTO.

## Block 7 — Tests & Acceptance (CTO §2.7)

- [ ] All 10 test suites green:
  - [ ] `test_schema_validates_alpha_catalog.py`
  - [ ] `test_urn_parse_build_roundtrip.py`
  - [ ] `test_urn_uniqueness_db.py`
  - [ ] `test_ontology_fk_enforcement.py`
  - [ ] `test_alembic_up_down.py`
  - [ ] `test_seed_load.py`
  - [ ] `test_publish_rejection_matrix.py`
  - [ ] `test_api_query_factor_by_urn.py`
  - [ ] `test_sdk_fetch_by_urn.py`
  - [ ] `test_provenance_checksum.py`
- [ ] Coverage ≥ 85% on new modules (`publish_gates.py`, ontology loaders, alias migration code).
- [ ] No regressions in Phase 0/1 test suites.
- [ ] **Sign-off**: Test Lead.

---

## Final sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| CTO |  |  |  |
| Methodology Lead |  |  |  |
| Backend Lead |  |  |  |
| Head of Data |  |  |  |
| Test Lead |  |  |  |
| Legal |  |  |  |

When every box is ticked and every role has signed: **Phase 2 is COMPLETE**. Open Phase 3 (Resolution Engine + Pricing) work tracker.
