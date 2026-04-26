# Epic: v0.1 Alpha (FY27 Q1)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | `alpha-v0.1`                                             |
| Target quarter   | FY27 Q1                                                  |
| Owner            | Akshay (interim Platform/Data owner until named lead is assigned) |
| Status           | In stabilization (Wave 5 complete; Phase 0 closed) |
| Source-of-truth  | `docs/factors/roadmap/...Source_of_Truth.docx`           |
| Release-profile  | `alpha-v0.1`                                             |

## Scope

The first public Alpha — read-only catalog over 6 authoritative
sources. Closed pilot with 1–3 design partners; SLA-light; no
billing.

* 5 public REST endpoints (always-on per release profile):
  `GET /v1/healthz`, `GET /v1/factors`, `GET /v1/factors/{urn}`,
  `GET /v1/sources`, `GET /v1/packs`.
* 6 alpha sources: IPCC AR6 (2006 NGGI), DEFRA / DESNZ 2024,
  EPA GHG Hub 2024, EPA eGRID 2022, India CEA 20.0, EU CBAM defaults
  2024.
* Frozen Factor Record v0.1 schema (`config/schemas/factor_record_v0_1.schema.json`).
* Canonical URN spec (`greenlang.factors.ontology.urn`) — namespace
  and id segments lowercase.
* Provenance gate
  (`greenlang.factors.quality.alpha_provenance_gate.AlphaProvenanceGate`)
  blocks any record missing the v0.1 audit fields.
* Python SDK v0.1 (read-only client surface).
* Postgres schema migration `V500__factors_v0_1_canonical.sql` +
  Alembic migration `0001_factors_v0_1_initial.py`.
* Grafana dashboard + Prometheus alert pack (`deployment/...`).
* Design-partner kit under `docs/factors/design-partners/`.

## Out of scope

(Lives in repo, gated off by release profile)

* `/v1/resolve`, `/v1/explain`, `/v1/batch`, `/v1/coverage`,
  `/v1/quality/fqs`, `/v1/editions` — `beta-v0.5`.
* GraphQL, ML resolve — `rc-v0.9`.
* Billing, OEM, SQL-over-HTTP, commercial packs, real-time grid —
  `ga-v1.0`.
* Marketplace, private packs, agent-native ingestion — post-GA.

## Deliverables

* [x] Catalog seed JSONs for 6 sources under
      `greenlang/factors/data/catalog_seed_v0_1/<src>/v1.json`.
* [x] Canonical URN parser + builder (`greenlang/factors/ontology/urn.py`).
* [x] Provenance gate + schema CI test.
* [x] Release-profile feature gate.
* [x] Postgres DDL + Alembic migration.
* [x] Python SDK v0.1 (`greenlang/factors/sdk/python/`).
* [x] Grafana dashboard + Prometheus alerts.
* [x] Design-partner onboarding checklist + 2 partner profiles
      (EU-MFG-01, IN-EXPORT-01).
* [x] Edition-cut runbook + alpha-publisher script.
* [ ] **Phase 0 audit cleanup (in progress 2026-04-26):**
  * [x] Lowercase namespace+id in all 691 seed URNs.
  * [x] Tighten schema regex + normalizer to enforce lowercase.
  * [x] Regression test: every seed URN parses canonically.
  * [x] CTO countersign on ADR-001.
  * [x] Decision on pack URN version-segment format (`2024.1` vs
        canonical `v<int>`) recorded in ADR-002.
  * [ ] Implement ADR-002 pack URN migration before v0.5.

## Acceptance criteria

* `python -m pytest tests/factors/v0_1_alpha -q` reports `0 failed`.
* `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1` mounts only the 5 alpha
  paths (asserted by `test_alpha_api_contract.py`).
* All 691 catalog-seed factor URNs pass
  `greenlang.factors.ontology.urn.parse` (asserted by
  `test_seed_urns_canonical_parse.py`).
* p95 `/v1/factors/{urn}` lookup ≤ 300ms (asserted by
  `test_perf_p95_lookup.py`).
* 1+ design partner accepted onboarding (`docs/factors/design-partners/MSA-NDA-status-tracker.md`).

## Source coverage

| source_id                | source                          | factor count |
| ------------------------ | ------------------------------- | -----------: |
| `ipcc_2006_nggi`         | IPCC 2006 NGGI Tier-1 defaults  |          235 |
| `desnz_ghg_conversion`   | UK DESNZ 2024 GHG conversion    |          195 |
| `epa_hub`                | US EPA GHG Hub 2024             |           84 |
| `egrid`                  | US EPA eGRID 2022 subregion     |           79 |
| `cbam_default_values`    | EU CBAM defaults 2024           |           60 |
| `india_cea_co2_baseline` | India CEA CO2 baseline 20.0     |           38 |
| **Total**                |                                 |     **691** |

## API / SDK expectations

* REST: 5 paths above. All other v1 paths must 404 in alpha (asserted
  by route filter in `release_profile.filter_app_routes`).
* SDK: read-only Python client; surface validated by
  `test_sdk_alpha_surface.py`.
* OpenAPI snapshot pinned at
  `tests/factors/v0_1_alpha/openapi_alpha_v0_1.json`.

## Security / compliance gates

* JWT secret env var (`JWT_SECRET`) set in production.
* `factors-v0.1-alpha-alerts.yaml` deployed to Prometheus.
* Edition manifest signed (or signature-placeholder logged in audit
  trail) via `alpha_edition_manifest`.

## Tickets (Phase 0 cleanup)

* [x] Lowercase URN seed cleanup (`scripts/factors_alpha_v0_1_lowercase_urns.py`).
* [x] Patch `coerce_factor_id_to_urn` to lowercase namespace + id.
* [x] Tighten `factor_record_v0_1.schema.json` URN pattern.
* [x] Add `tests/factors/v0_1_alpha/test_seed_urns_canonical_parse.py`.
* [x] CTO sign-off on ADR-001.
* [x] Decide pack URN version-segment policy (ADR-002).
* [ ] Migrate alpha seed `factor_pack_urn` values to ADR-002 canonical form.

## Dependencies

* INFRA-002 (Postgres + TimescaleDB) — production database.
* SEC-001/002 (JWT auth + RBAC) — required for design-partner keys.
* OBS-001/004 (Prometheus + alerting) — alpha alert pack consumer.

## Release risks

* Pack URN format mismatch: ADR-002 resolves the policy in favor of
  canonical public `v<int>` pack URNs. Migration remains a v0.1
  hardening task before v0.5.
* Source-vintage drift: DESNZ publishes new 2025 conversion factors
  in mid-FY27; vintage-audit doc in place but parser not yet wired
  for incremental updates.
* eGRID 2022 → 2023 release expected mid-FY27; backfill flow not
  yet exercised end-to-end.
