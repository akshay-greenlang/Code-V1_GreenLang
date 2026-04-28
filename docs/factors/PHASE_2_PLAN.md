# Phase 2 — Canonical Schema, URNs, Ontologies & Storage

> **Authority**: CTO Phase 2 brief (2026-04-27).
> **Status**: ACTIVE — Phase 1 (Source Rights) shipped 2026-04-26.
> **Goal**: Make GreenLang Factors *structurally reliable*. Every factor must have a canonical schema, canonical URN, validated ontology links, and durable Postgres-backed storage with publish-time gates.
> **Owner**: GL-Factors Engineering (Backend, Data, Calculator) + GL-SpecGuardian.
> **Target exit date**: 2026-06-15 (6-week sprint from Phase 1 close).

---

## Phase 2 vision

After Phase 2, Factors stops being “seed files plus parser outputs” and becomes a proper data platform contract. Every record traceable end-to-end: URN → factor row → source row → source artifact (sha256-checksummed in object storage) → parser commit. Every reference (geography / unit / methodology / activity / source / pack) is FK-validated. Every publish is gated. Every schema change is classified.

---

## Phase 2 KPIs

### 1. Schema Contract

| KPI | Target | Measurement |
|---|---|---|
| `factor_record_v0_1` frozen, documented, versioned | YES | Schema $id stable; freeze note signed |
| v0.1 alpha factors validating against schema | 100% | CI gate `tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py` |
| Production/publish paths bypassing schema validation | 0 | Code search + audit log review |
| Required field groups present (identity / value-unit / context / quality / licence / lineage / lifecycle) | 7/7 | Manual schema audit + automated field-group test |

### 2. URN Compliance

| KPI | Target | Measurement |
|---|---|---|
| Public factor IDs using `urn:gl:factor:...` | 100% | Repository scan |
| Uppercase namespace segments in any URN kind | 0 | Lowercase regex sweep across factor/source/pack/methodology/geography/unit/activity |
| Deterministic URN builders covered by tests | 100% | Property-based round-trip tests |
| Legacy `EF:...` IDs as primary public ID | 0 | Alias table only |

### 3. Ontology Coverage

| KPI | Target | Measurement |
|---|---|---|
| Core ontology tables exist + seeded (geography, unit, activity, methodology) | 4/4 | DDL + row-count check |
| v0.1 factors with valid `geography_urn` / `unit_urn` / `activity_taxonomy_urn` / `methodology_urn` | 100% | FK enforcement via DB + WS8 ontology gate |
| Invalid ontology references rejected at publish time | 100% | Negative test matrix |

### 4. Storage Readiness

| KPI | Target | Measurement |
|---|---|---|
| Canonical Postgres tables created via Alembic | 14/14 | source / source_artifacts / factor_pack / methodology / geography / unit / activity / factor / factor_aliases / provenance_edges / changelog_events / api_keys / entitlements / release_manifests |
| Rollback SQL exists for every Phase 2 migration | 100% | `*_DOWN.sql` files |
| Source artifacts with checksum + URI + version + ingestion meta | 100% | `extraction.raw_artifact_sha256` + `source_artifacts` row |
| Factors queryable from Postgres by URN / source / pack / geography / activity / methodology / vintage / lifecycle | YES | API + SDK + repository tests |

### 5. Schema Evolution

| KPI | Target | Measurement |
|---|---|---|
| Written policy for v0.x → v1.0 → v2.0 → v3.0 | YES | `docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md` |
| Schema changes with migration notes + compatibility classification | 100% | CI gate `scripts/ci/check_schema_migration_notes.py` |
| Undocumented schema changes blocked by CI | YES | Failing build on missing changelog entry |

---

## Detailed To-Do List (CTO 2.1 → 2.7)

### 2.1 — Freeze `factor_record_v0_1` (WS1)

> Status: Schema FROZEN 2026-04-25 at `config/schemas/factor_record_v0_1.schema.json`. Phase 2 hardens it.

- [x] Define `factor_record_v0_1.schema.json` (FROZEN 2026-04-25).
- [x] Create matching Pydantic v2 models at `greenlang/factors/schemas/factor_record_v0_1.py` with one class per field group: `IdentityFields`, `ValueUnitFields`, `ContextFields`, `TimeFields`, `ClimateBasisFields`, `QualityFields`, `LicenceFields`, `LineageFields`, `LifecycleFields`, plus a top-level `FactorRecordV0_1`. (Shipped 2026-04-27.)
- [x] Postgres table structure — `factors_v0_1.factor` exists in V500.
- [x] Document each field at `docs/factors/schema/FACTOR_RECORD_V0_1_FIELD_REFERENCE.md` (meaning, type, required/nullable, example, validation rule). (Shipped 2026-04-27.)
- [x] CI test `tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py` loads every v0.1 factor and validates against the frozen schema. (Shipped 2026-04-27 — all 691 alpha records pass.)
- [x] Field-group coverage assertion: every required group has at least one populated field on every record. (Shipped 2026-04-27 — see `scripts/factors/phase2_audit_field_groups.py` for the coverage matrix.)

**Exit**: No v0.1 record can be published unless it validates against `factor_record_v0_1`.

---

### 2.2 — Stable URN rules (WS2)

> Status: URN module robust at `greenlang/factors/ontology/urn.py`. Phase 2 finishes API/SDK/alias surfaces.

- [x] Final URN grammar defined.
- [x] Deterministic builders for: factor, source, pack, methodology, geography, unit, activity, community, partner, enterprise.
- [x] Parser/validator functions per URN type.
- [x] Lowercase namespace enforcement (parser rejects uppercase).
- [x] DB-level uniqueness (UNIQUE constraints in V500 on factor.urn, source.urn, factor_pack.urn, methodology.urn, geography.urn, unit.urn).
- [ ] Create `factor_aliases` table + Alembic migration V501 (mapping legacy `EF:...` → primary `urn`).
- [ ] Backfill alias rows for every catalog record carrying a `factor_id_alias`.
- [ ] Confirm API `/v0_1/alpha/factors` and SDK `factors.get_by_urn()` expose `urn` as primary identifier in every response shape.
- [ ] Property-based round-trip tests for every kind (parse → render = identity; build → parse = identity).
- [ ] Lowercase sweep across catalog seed (zero uppercase segments).

**Exit**: API, SDK, seed data, publish pipeline, and DB all agree on the same canonical URN. Legacy IDs work only as aliases.

---

### 2.3 — Core ontology tables (WS3 / WS4 / WS5 / WS6)

> Status: Geography, unit, methodology tables exist in V500. Activity table TBD.

#### Geography (WS3)
Required types: global, country, state/province, grid subregion, bidding zone, balancing authority, basin, tenant-defined.

- [x] V500 includes types: global, country, subregion, state_or_province, grid_zone, bidding_zone, balancing_authority.
- [ ] Migration V501 ALTER adds `basin` and `tenant` to the `geography.type` CHECK enum.
- [ ] Seed `greenlang/factors/data/ontology/geography_seed_v0_1.yaml` with: global:world; ISO-3166-1 country codes used by alpha sources (in/us/gb/de/fr/cn/jp/br/au/za/mx/ca/nl/it/es/pl); subregion:eu-27, asean, oecd, latam; state_or_province:us-tx, us-ca, us-ny, in-mh, in-ka; grid_zone:egrid-rfcw, egrid-serc, egrid-wecc, egrid-mrow, egrid-rfcw, egrid-rfce; bidding_zone:de-lu, fr, gb, es, it-no, it-cs; balancing_authority:caiso, pjm, ercot, miso, nyiso, isone.
- [ ] Loader: `greenlang/factors/data/ontology/loaders/geography_loader.py` + Alembic data migration that idempotently inserts seed rows.
- [ ] FK validation: every v0.1 factor's `geography_urn` resolves (already enforced by V500 FK).

#### Units (WS4)
Required dimensions: mass, energy, volume, distance, freight activity, currency, composite climate.

- [x] V500 `unit` table has `dimension` column.
- [ ] Pre-defined dimensions: `mass`, `energy`, `volume`, `distance`, `freight_activity`, `currency`, `composite_climate`.
- [ ] Seed `greenlang/factors/data/ontology/unit_seed_v0_1.yaml` covering composite climate (kgco2e/kwh, kgco2e/kg, kgco2e/tkm, kgco2e/l, kgco2e/m3, kgco2e/passenger-km, kgco2e/usd, kgco2e/eur, kgco2e/gbp, kgco2e/inr, kgco2e/mj, kgco2e/gj, kgco2e/btu, kgco2e/therm) plus base units (kg, t, kwh, mj, gj, btu, l, m3, km, mi, tkm, passenger-km, usd, eur, gbp, inr).
- [ ] Pre-populate `conversions` JSONB for energy (kwh↔mj↔gj↔btu↔therm), mass (kg↔t↔lb), volume (l↔m3↔gal-us), distance (km↔mi).
- [ ] Loader + Alembic data migration.
- [ ] FK validation: every v0.1 factor's `unit_urn` resolves (FK already enforced).

#### Activity taxonomy (WS5)
Required taxonomies: IPCC, GHGP, HS/CN, CPC, NACE, NAICS/SIC, PACT, freight, CBAM, PCF, refrigerants, agriculture, waste, land-use.

- [ ] Migration V502 creates `factors_v0_1.activity` (urn UNIQUE, taxonomy TEXT, code TEXT, parent_urn FK, name TEXT, description TEXT, created_at).
- [ ] URN pattern: `urn:gl:activity:{taxonomy}:{code-slug}`. (Existing `activity` builder takes a single slug — extend to optionally accept `taxonomy` + `code` and assemble `taxonomy/code-slug`. Backwards compatible.)
- [ ] Seed `greenlang/factors/data/ontology/activity_seed_v0_1.yaml` minimum coverage:
  - **IPCC**: 2006 categories 1.A.1 / 1.A.2 / 1.A.3 / 1.A.4 / 1.B / 2 / 3 / 4 / 5.
  - **GHGP**: scope1, scope2-location, scope2-market, scope3-cat-1..15.
  - **HS/CN**: chapter-level codes for CBAM scope (72, 73, 76, 25, 28, 31).
  - **CPC**: top-level divisions used by alpha factors.
  - **NACE Rev 2**: A–U sections.
  - **NAICS**: 2-digit sectors used by alpha.
  - **PACT**: top-level product categories.
  - **freight**: road-wtw, road-ttw, sea, air, rail, inland-waterway.
  - **CBAM**: cement, iron-steel, aluminium, fertiliser, hydrogen, electricity (Annex I scope).
  - **PCF**: ISO 14067 lifecycle phases.
  - **refrigerants**: hfc-134a, hfc-32, hfc-125, hfc-143a, hfc-152a, hfo-1234yf, hfo-1234ze, sf6, nf3, pfc-14, pfc-116.
  - **agriculture**: enteric-fermentation, manure-management, rice-cultivation, synthetic-fertiliser, biomass-burning.
  - **waste**: msw-landfill, msw-incineration, msw-composting, wastewater-domestic, wastewater-industrial.
  - **land-use**: deforestation, afforestation, reforestation, grassland-conversion, peatland.
- [ ] Loader + Alembic data migration.
- [ ] (Optional in v0.1, REQUIRED in v0.2) FK from `factor.activity_taxonomy_urn` once promoted.

#### Methodology (WS6)
Required: GHGP Scope 1/2/3, IPCC Tier 1/2/3, CBAM default/actual, eGRID, CEA, GLEC, ISO 14083, PCAF, ecoinvent system models.

- [x] V500 `methodology` table exists.
- [ ] Seed `greenlang/factors/data/ontology/methodology_seed_v0_1.yaml`:
  - `urn:gl:methodology:ghgp-corporate-scope1`, `…scope2-location`, `…scope2-market`, `…scope3-cat-1` … `…scope3-cat-15`.
  - `urn:gl:methodology:ipcc-tier-1-stationary-combustion`, `…tier-2-stationary`, `…tier-3-stationary`, `…tier-1-mobile`, `…tier-1-fugitive`.
  - `urn:gl:methodology:eu-cbam-default`, `…eu-cbam-actual`.
  - `urn:gl:methodology:epa-egrid-subregion-2024`.
  - `urn:gl:methodology:india-cea-baseline`.
  - `urn:gl:methodology:glec-framework-v3`.
  - `urn:gl:methodology:iso-14083-2023`.
  - `urn:gl:methodology:pcaf-financed-emissions-v1`.
  - `urn:gl:methodology:ecoinvent-cutoff`, `…apos`, `…consequential`.
- [ ] Each row populates framework, tier (where applicable), approach, boundary_template.
- [ ] FK validation: every v0.1 factor's `methodology_urn` resolves (FK already enforced).

**Exit**: Every factor references valid ontology URNs through foreign keys or strict validation. Invalid ontology references fail publish-time validation.

---

### 2.4 — Postgres canonical storage (WS7)

> Status: V500 ships 6 of 14 tables. Phase 2 adds the remaining 8 + indexes + partition strategy ADR.

Required tables (CTO list):

| # | Table | V500? | Phase 2 action |
|---|---|---|---|
| 1 | `source` | ✅ | Verify alpha_v0_1 flagging |
| 2 | `source_artifacts` | ❌ | NEW in V501 |
| 3 | `factor_packs` | ✅ (`factor_pack`) | rename or alias |
| 4 | `methodologies` | ✅ (`methodology`) | seed via WS6 |
| 5 | `geographies` | ✅ (`geography`) | seed via WS3 |
| 6 | `units` | ✅ (`unit`) | seed via WS4 |
| 7 | `activities` | ❌ | NEW in V502 |
| 8 | `factors` | ✅ (`factor`) | — |
| 9 | `factor_aliases` | ❌ | NEW in V501 |
| 10 | `provenance_edges` | ❌ | NEW in V503 |
| 11 | `changelog_events` | ❌ | NEW in V503 |
| 12 | `api_keys` | ❌ | NEW in V504 (or reuse SEC-001 table — verify) |
| 13 | `entitlements` | ❌ | NEW in V504 (DB-backed mirror of `config/entitlements/alpha_v0_1.yaml`) |
| 14 | `release_manifests` | ❌ | NEW in V504 |

- [ ] Migration V501: `source_artifacts` (sha256, uri, source_urn FK, version, ingested_at, parser_id, parser_version, parser_commit, size_bytes, content_type) + `factor_aliases` (urn FK, legacy_id, kind, retired_at).
- [ ] Migration V502: `activity` table (per WS5).
- [ ] Migration V503: `provenance_edges` (factor_urn FK, source_artifact_pk FK, row_ref, edge_type) + `changelog_events` (event_type, schema_version, urn, change_class, migration_note_uri, actor, occurred_at).
- [ ] Migration V504: `api_keys` (key_hash, tenant, scopes, created_at, revoked_at), `entitlements` (tenant, source_urn FK, granted_at, expires_at, terms_uri), `release_manifests` (release_id, factor_urns[], schema_version, signature, released_at, released_by).
- [ ] Rollback DOWN SQL for V501–V504.
- [ ] Partition stub + ADR `docs/factors/adr/ADR-003-factor-partitioning.md`: candidate keys (vintage_year first; secondary by source_urn hash). Not executed at v0.1 (≤2k rows); ready when v1.0 hits 100k rows.
- [ ] Indexes (per CTO): URN lookup (already), source, pack, geography, activity, methodology, vintage, lifecycle status.
- [ ] Repository upgrade: `AlphaFactorRepository` exposes `find_by_methodology()`, `find_by_activity()`, `find_by_source_artifact_sha()`.
- [ ] Raw-artifact storage: `greenlang/factors/artifacts/` (local dev) + S3 bucket `s3://greenlang-factors-raw/{source_id}/{version}/{filename}` (production). Versioning enabled. Checksum verified on read.
- [ ] DB-backed API tests: round-trip a record through publish → query-by-urn → query-by-filter → query-by-alias.

**Exit**: v0.1 factors load into Postgres, queryable by API/SDK, traceable back to source artifacts.

---

### 2.5 — Publish-time validation gates (WS8)

> Status: `AlphaProvenanceGate` (schema only). Phase 2 expands to 7 ordered gates.

Build `greenlang/factors/quality/publish_gates.py` orchestrating:

| Gate | Check | Failure exception |
|---|---|---|
| 1 | JSON Schema validation against frozen `factor_record_v0_1.schema.json` | `SchemaValidationError` |
| 2 | URN uniqueness — DB lookup against `factor.urn` AND `factor_aliases.legacy_id` | `URNDuplicateError` |
| 3 | Ontology FK reference — `geography_urn`, `unit_urn`, `methodology_urn`, `factor_pack_urn`, `source_urn` resolve | `OntologyReferenceError` |
| 4 | Source registry reference — `source_urn` exists with `alpha_v0_1=true` (or env override) | `SourceRegistryError` |
| 5 | Licence match — `factor.licence` consistent with source registry's `licence`; redistribution_class enforced via existing `SourceRightsService` | `LicenceMismatchError` |
| 6 | Provenance completeness — `extraction.{source_url, source_record_id, source_publication, source_version, raw_artifact_uri, raw_artifact_sha256, parser_id, parser_version, parser_commit, row_ref, ingested_at, operator}` all populated; sha256 matches stored artefact | `ProvenanceIncompleteError` |
| 7 | Lifecycle — `review.review_status='approved'` + `approved_by` + `approved_at` for production publish | `LifecycleStatusError` |

Each gate idempotent + composable. Gates run in order; first failure aborts.

Pre-publish CLI: `gl factors validate <path-to-record.json> [--dry-run]`.

Negative test matrix (Phase 2 acceptance — all must REJECT):
1. Missing `source_urn`.
2. Invalid `unit_urn` (FK miss).
3. Invalid `methodology_urn` (FK miss).
4. Uppercase URN segment.
5. Duplicate factor URN.
6. Missing `extraction.raw_artifact_sha256`.
7. Missing `extraction.parser_version`.
8. Missing `review` metadata.
9. Licence mismatch with source registry entry.

**Exit**: Invalid records cannot enter staging or production tables.

---

### 2.6 — Schema evolution policy (WS9)

> Status: NEW — write before v1.0 lock.

- [ ] Author `docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md`:
  - **v0.x**: breaking changes allowed only with migration notes + methodology-lead approval.
  - **v1.0**: public API/schema locked **24 months** from GA.
  - **v2.0**: major schema step with **12-month** v1↔v2 overlap window.
  - **v3.0**: major schema step with **18-month** v2↔v3 overlap window.
- [ ] Compatibility labels: `additive` | `breaking` | `deprecated` | `removed`.
- [ ] Schema version registry: `greenlang/factors/schemas/_version_registry.py` enumerating known $ids + classification + supersedes/superseded_by links.
- [ ] Migration note template: `docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md`.
- [ ] CI gate: `scripts/ci/check_schema_migration_notes.py` diffs any `config/schemas/factor_record_*.schema.json` against the merge base; FAILS the build if no matching entry in `docs/factors/schema/CHANGELOG.md` carries a compatibility label.
- [ ] Wire CI gate into `.github/workflows/factors-ci.yml`.

**Exit**: No schema change can merge without version classification and migration documentation.

---

### 2.7 — Tests & acceptance (WS10)

Minimum required test suites (`tests/factors/v0_1_alpha/phase2/`):

| Suite | Purpose |
|---|---|
| `test_schema_validates_alpha_catalog.py` | Every v0.1 factor validates |
| `test_urn_parse_build_roundtrip.py` | Property-based parser/builder |
| `test_urn_uniqueness_db.py` | Duplicate INSERT raises |
| `test_ontology_fk_enforcement.py` | Invalid geography/unit/methodology rejected |
| `test_alembic_up_down.py` | V500 → V504 up→down→up clean |
| `test_seed_load.py` | Full ontology seed loads with expected counts |
| `test_publish_rejection_matrix.py` | All 9 negative cases (per WS8) reject |
| `test_api_query_factor_by_urn.py` | REST `/v0_1/alpha/factors/{urn}` |
| `test_sdk_fetch_by_urn.py` | Python SDK `factors.get_by_urn()` |
| `test_provenance_checksum.py` | sha256 round-trip on raw artefacts |

**Phase 2 is complete only when all of the following are TRUE:**
1. All v0.1 factors validate against `factor_record_v0_1.schema.json`.
2. All v0.1 factors load into Postgres without error.
3. API and SDK return canonical URNs.
4. Legacy `EF:...` IDs work only as aliases.
5. Invalid schema/URN/ontology/licence/provenance records are blocked.
6. CI enforces all of the above.

---

## Workstream owners + dependencies

| WS | Owner | Depends on |
|---|---|---|
| WS1 — Schema freeze | gl-spec-guardian + gl-backend-developer | — |
| WS2 — URN compliance | gl-backend-developer | WS1 |
| WS3 — Geography seed | gl-formula-library-curator | — |
| WS4 — Unit seed | gl-formula-library-curator | — |
| WS5 — Activity taxonomy | gl-formula-library-curator + gl-backend-developer | WS2 |
| WS6 — Methodology seed | gl-formula-library-curator | — |
| WS7 — Storage tables | gl-backend-developer | WS3, WS4, WS5, WS6 |
| WS8 — Publish gates | gl-backend-developer + gl-spec-guardian | WS1, WS2, WS3, WS4, WS6, WS7 |
| WS9 — Evolution policy | gl-tech-writer + gl-spec-guardian | — |
| WS10 — Test suite | gl-test-engineer | WS1–WS9 |
| WS11 — Phase 2 docs | gl-tech-writer | WS1–WS10 |

WS3 / WS4 / WS6 / WS9 / WS11 (doc skeleton) can run in parallel from day 1.
WS7 needs ontology seeds in flight (not blocking on completion since FKs were already declared in V500).
WS8 + WS10 are the convergence point.

---

## Risk register

| Risk | Mitigation |
|---|---|
| URN grammar evolves and breaks alias backfill | Freeze grammar in `urn.py` with property tests; alias table is append-only |
| Ontology seed gaps force v0.1 record rejection | Pre-flight seed-vs-catalog diff before publish lockdown |
| Partition strategy not validated at scale | ADR-003 defines candidate keys but defers actual partitioning to v1.0 |
| Schema evolution policy disputed by methodology lead | CTO countersign required on `SCHEMA_EVOLUTION_POLICY.md` before merge |
| `EF:...` aliases collide across sources | `factor_aliases.legacy_id` UNIQUE constraint + ingestion conflict log |

---

## Out of scope for Phase 2

- v0.5+ Scope 3 / freight / ag / waste extensions (separate phase).
- Multi-gas split (`numerator.co2/ch4/n2o`) — comes with v0.5 schema.
- Hourly grid intensity (deferred to v2.5).
- AR4 / AR5 GWP basis (alpha is AR6-only).
- Signed receipts / edition pinning (Phase 3).
- LLM-driven entity resolution on activity URNs (Phase 4).

---

## How to track progress

- TaskCreate IDs #1–#11 mirror WS1–WS11.
- KPI dashboard: `docs/factors/PHASE_2_KPI_DASHBOARD.md` (refreshed weekly).
- Exit checklist: `docs/factors/PHASE_2_EXIT_CHECKLIST.md`.
- Weekly Phase 2 standup: review TaskList + KPI dashboard + risk register.
