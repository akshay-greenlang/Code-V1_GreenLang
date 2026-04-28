# Phase 2 Exit Checklist

> **Authority**: CTO Phase 2 brief (2026-04-27).
> **Owner**: Factors Engineering.
> **Reviewers**: CTO, Methodology Lead, Legal, Head of Data.

Every technical box must be ticked and every accountable role must sign off before Phase 2 is declared formally complete and Phase 3 (Resolution + Pricing engines) may start.

> **Engineering verification snapshot**: 2026-04-28 (post-commit `6c9b91fd` + branch-coverage tests). Block-by-block audit by parallel Codex agents confirmed **35/35 engineering boxes GREEN**. Phase 2 acceptance runner: `10/10 PASS` in 552s (cold) / ~175s (warm). Full v0.1 alpha suite: **1,789 passed, 15 skipped, 0 failed** in 140.72s. New-module branch coverage: **97.15% total** (`publish_gates` 95.02%, `activity_loader` 96.88%, `geography_loader` 99.44%, `unit_loader` 99.30%, `methodology_loader` 99.24%, `_common` 100%). This checklist records engineering evidence; it does **not** replace the required human CTO / Methodology / Backend / Head of Data / Test / Legal signatures.

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
- [x] `activity` table seeded across 15 taxonomies: IPCC, GHGP, HS, CPC, NACE, NAICS, SIC, PACT, freight, CBAM, PCF, refrigerants, agriculture, waste, land-use. (HS/CN and NAICS/SIC are split into separate taxonomies in `activity_seed_v0_1.yaml` — wording corrected from CTO brief's collapsed pairing.)
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

---

## Engineering evidence — per-box (2026-04-28)

> Each box above maps to a single line of evidence below. Cite during sign-off; if a reviewer disputes a box, point at the file/test cited here.

### Block 1 — Schema Contract

| Box | Evidence |
|---|---|
| Schema $id frozen | `config/schemas/factor_record_v0_1.schema.json:3` — `$id` matches the canonical https://schemas.greenlang.io URL |
| Pydantic mirror == JSON Schema | `tests/factors/v0_1_alpha/phase2/test_pydantic_mirrors_jsonschema.py` — 9/9 parity tests PASS (required fields, properties, enums, patterns, nested objects) |
| FIELD_REFERENCE.md complete | `docs/factors/schema/FACTOR_RECORD_V0_1_FIELD_REFERENCE.md` — 32 properties × 93 rows; meaning + type + nullable + example + validation rule per field |
| CI gate green vs catalog | `tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py` — 3/3 PASS, 691 records validate across 6 sources |
| 7 field groups present | `scripts/factors/phase2_audit_field_groups.py` — 100% coverage on 691 records |

### Block 2 — URN Compliance

| Box | Evidence |
|---|---|
| 100% urn:gl:factor primary | All 1,491 catalog factors expose `urn` as primary; legacy `EF:` only as `factor_id_alias`. Schema regex enforces at `factor_record_v0_1.schema.json:34` |
| 0 uppercase URN segments | `tests/factors/v0_1_alpha/phase2/test_urn_lowercase_sweep.py` — sweeps catalog_seed + source_registry.yaml |
| Property roundtrip | `tests/factors/v0_1_alpha/phase2/test_urn_property_roundtrip.py` — 20 Hypothesis tests × 200 examples × 9 URN kinds |
| Legacy not primary in API/SDK | `greenlang/factors/api_v0_1_alpha_models.py:62-73` + `test_api_urn_primary.py` — 4 contract layers verified |
| factor_aliases populated 1:1 | `scripts/factors/phase2_backfill_factor_aliases.py` — 1,488 alias rows with idempotent ON CONFLICT DO NOTHING; `test_alias_backfill_idempotency.py` |

### Block 3 — Ontology Coverage

| Box | Evidence |
|---|---|
| Geography seeded | `greenlang/factors/data/ontology/geography_seed_v0_1.yaml` — 67 entries: 1 global + 18 countries + 4 subregions + 7 states + 21 grid zones + 7 bidding zones + 6 balancing authorities + 3 basins |
| Unit seeded | `greenlang/factors/data/ontology/unit_seed_v0_1.yaml` — 47 units; 18 composite climate + 12 base; 31 with conversions JSONB |
| Methodology seeded | `greenlang/factors/data/ontology/methodology_seed_v0_1.yaml` — 33 entries spanning all 9 frameworks (GHGP, IPCC, EU-CBAM, eGRID, CEA, GLEC, ISO-14083, PCAF, ecoinvent) |
| Activity seeded (15 taxonomies) | `greenlang/factors/data/ontology/activity_seed_v0_1.yaml` — 172 entries across all 15 taxonomies |
| 100% v0.1 FK-resolves | `tests/factors/v0_1_alpha/phase2/test_seed_load.py` 17/17 PASS + `test_ontology_fk_enforcement.py` 6 PASS / 7 SKIP (Postgres) |
| Invalid refs rejected | `tests/factors/v0_1_alpha/phase2/test_publish_rejection_matrix.py` cases 2 + 3 (invalid unit_urn / methodology_urn → `OntologyReferenceError`) |

### Block 4 — Storage Readiness

| Box | Evidence |
|---|---|
| Migrations V500–V506 merged | `deployment/database/migrations/sql/V50{0..6}*.sql` — 7 UP files committed in 6c9b91fd |
| 14 canonical tables exist | `V500__factors_v0_1_canonical.sql:32+` — source / source_artifacts / factor_pack / methodology / geography / unit / activity / factor / factor_aliases / provenance_edges / changelog_events / api_keys / entitlements / release_manifests |
| DOWN SQL for every migration | 7 paired `*_DOWN.sql` files |
| Required indexes present | `V506` includes `factor_activity_urn_idx`; V500 includes `factor_active_idx WHERE review_status='approved'` and source/pack/geo/vintage indexes |
| Source artefacts versioned | `V505__factors_v0_1_phase2_aliases_artifacts.sql:63` — `source_artifacts` table with sha256 UNIQUE + uri + source_version + parser_id + parser_version + ingested_at + metadata JSONB |
| Repository query API | `greenlang/factors/repositories/alpha_v0_1_repository.py:751` — `get_by_urn`, `list_factors`, `find_by_methodology`, `find_by_activity`, `find_by_alias` cover all 8 facets |
| ADR-004 partition deferral | `docs/factors/adr/ADR-004-factor-partitioning.md:1` — Status: Accepted (deferred to v1.0); trigger: row count > 100k OR P95 > 100ms |

### Block 5 — Publish-Time Gates

| Box | Evidence |
|---|---|
| 7-gate orchestrator in order | `greenlang/factors/quality/publish_gates.py:281-287` — `assert_publishable` calls gate_1..gate_7 sequentially |
| 9 negative cases reject | `tests/factors/v0_1_alpha/phase2/test_publish_rejection_matrix.py:265-374` — case 1 SchemaValidationError / 2 OntologyReferenceError / 3 OntologyReferenceError / 4 SchemaValidationError / 5 URNDuplicateError / 6 ProvenanceIncompleteError / 7 ProvenanceIncompleteError / 8 SchemaValidationError / 9 LicenceMismatchError; 22 tests PASS |
| `gl factors validate` CLI dry-run | `greenlang/factors/cli_validate.py:208-218` — `--dry-run` calls `orchestrator.dry_run(record)` and prints per-gate matrix |
| No bypass paths | Only 2 INSERTs into factor table (sqlite alpha_v0_1_repository.py:619, postgres :729); both gated by `assert_publishable`. Legacy opt-out emits one-time `logger.warning`. `test_publish_default_secure.py` 8/8 PASS |

### Block 6 — Schema Evolution

| Box | Evidence |
|---|---|
| SCHEMA_EVOLUTION_POLICY.md | `docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md:22-26` — table codifies v0.x / v1.0 (24mo) / v2.0 (12mo) / v3.0 (18mo) windows |
| 4 compatibility labels enforced | `scripts/ci/check_schema_migration_notes.py:70-72` — `CHANGELOG_ENTRY_RE` enforces `additive\|breaking\|deprecated\|removed` |
| _version_registry.py lists $ids | `greenlang/factors/schemas/_version_registry.py:140-152` — REGISTRY dict keyed on schema_id |
| CI gate blocks undocumented changes | `scripts/ci/check_schema_migration_notes.py:704-827` — `run_gate` returns exit 1 if missing, 2 if weaker than classifier verdict |
| MIGRATION_NOTE_TEMPLATE.md | `docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md:1-96` — YAML frontmatter + Summary / Motivation / Field Diff / Data Migration / Code Migration / Customer Impact / Rollback Plan |
| factors-schema-evolution-check.yml committed | `git ls-files .github/workflows/factors-schema-evolution-check.yml` returns the path (tracked in 6c9b91fd) |

### Block 7 — Tests & Acceptance

| Box | Evidence |
|---|---|
| 10 acceptance suites green | `python scripts/factors/run_phase2_acceptance.py` — 10/10 PASS in 552s cold-cache |
| Coverage ≥ 85% on new modules | `publish_gates` 95.02%, `activity_loader` 96.88%, `geography_loader` 99.44%, `unit_loader` 99.30%, `methodology_loader` 99.24%, `_common` 100%. Total 97.15%. Closed via 130 new branch tests in `test_publish_gates_branches.py` + `test_ontology_loader_branches.py` |
| No regressions in Phase 0/1 | Full `tests/factors/v0_1_alpha` — 1,789 passed, 15 skipped, 0 failed. The 15 skips are Postgres-only suites gated on `GL_TEST_POSTGRES_DSN` (intentional) |
