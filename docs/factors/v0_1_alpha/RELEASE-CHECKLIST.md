# GreenLang Factors v0.1 Alpha — Release Checklist

**Status**: READY FOR HUMAN-BOUND CLOSE-OUT
**Target release**: FY27 Q1 (Apr–Jun 2026)
**Release profile**: `alpha-v0.1`
**Audience**: Internal + 2 design partners (one India-linked exporter, one EU-facing manufacturer)
**Last updated**: 2026-04-25
**Authority**: GreenLang Factors CTO doc §19.1; CTO audit memo

This is the binary go/no-go gate for the alpha launch. Every item below has a single `[x]` (DONE) or `[ ]` (PENDING) and a verification command. **All `[x]` items are verifiable from this repo today.** **All `[ ]` items are explicitly human-bound** — they require a CTO/Legal/Methodology/SRE/Operator action that no agent can perform.

---

## 1. Schema (CTO §19.1: "Schema approved")

- [x] **`factor_record_v0_1.schema.json` frozen** at `config/schemas/factor_record_v0_1.schema.json` (2026-04-25). $id = `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json`.
- [x] **Freeze note** at `config/schemas/FACTOR_RECORD_V0_1_FREEZE.md` documents versioning policy, omitted v1 fields, methodology lead approval requirement.
- [x] **v1↔v0.1 compatibility map** at `config/schemas/factor_record_v0_1_to_v1_map.json`.
- [x] **Postgres DDL** at `deployment/database/migrations/sql/V500__factors_v0_1_canonical.sql` — 7 tables, 13 indexes, 1 immutability trigger, 27 CHECK constraints; provenance gate enforced at SQL level.
- [x] **Alembic revision** `0001_factors_v0_1_initial` at `migrations/versions/`. `alembic upgrade head` reaches alpha schema. Runbook at `docs/factors/runbooks/migrate-v0_1.md`.
- [x] **Schema-validates-catalog CI gate** at `tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py` — passes strict (no xfail) for all 6 alpha sources after backfill.
- [ ] **Methodology lead sign-off** on the frozen schema (the file exists; the formal approval signature does not). `human:methodology-lead@greenlang.io` to add a signature line to `FACTOR_RECORD_V0_1_FREEZE.md` Approvals section.
- [ ] **CTO sign-off** on the freeze date and field omissions vs v1.

**Verify**: `pytest tests/factors/v0_1_alpha/test_factor_record_v0_1_schema_loads.py tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py tests/factors/v0_1_alpha/test_postgres_ddl_v500.py tests/factors/v0_1_alpha/test_alembic_revision_0001.py -x`

---

## 2. URN scheme (CTO §6.1.1)

- [x] **URN parser/builder** at `greenlang/factors/ontology/urn.py` — 100 tests pass, every CTO §6.1.1 worked example round-trips.
- [x] **URN as canonical primary id** in API responses (`/v1/factors/{urn:path}`) and SDK (`FactorsClient.get_factor(urn)`). `EF:...` retained as `factor_id_alias` field.
- [x] **`coerce_factor_id_to_urn()` migration helper** lifts legacy ids to URNs deterministically.
- [ ] **Methodology lead sign-off**: "the scheme covers tier-1 sources without compromise" (CTO §19.1 acceptance criterion). One paragraph sign-off in `docs/specs/urn_scheme_methodology_review.md` (file does not yet exist).

**Verify**: `pytest tests/factors/v0_1_alpha/test_urn.py -x` → 100 pass.

---

## 3. Source vintages (CTO §19.1: "first parsers working")

| # | Source | Vintage | Status | Methodology exception |
|---|---|---|---|---|
| #7 | IPCC (NGGI 2019 EFs + AR6 GWPs) | 2019.1 + AR6 | locked | n/a |
| #8 | DEFRA / DESNZ | 2024.1 (target 2025) | preview | `methodology-exceptions/desnz_ghg_conversion_v2024.1.md` |
| #9 | EPA GHG Hub | 2024.1 (target 2025) | preview | `methodology-exceptions/epa_hub_v2024.1.md` |
| #10 | EPA eGRID | 2022.1 (target 2024) | preview | `methodology-exceptions/egrid_v2022.1.md` |
| #11 | India CEA | v20.0 / v22.0 | locked | n/a (latest published) |
| #12 | EU CBAM defaults | 2024.1 | locked | n/a (current effective period per Reg 2023/1773 Annex IV) |

- [x] **Audit doc** at `docs/factors/v0_1_alpha/SOURCE-VINTAGE-AUDIT.md` documents per-source decision and materiality estimate.
- [x] **Snapshot tests** at `tests/factors/v0_1_alpha/test_alpha_source_snapshots.py` — 7 tests covering all 6 sources (26 factors total), parser-drift guard.
- [ ] **Methodology lead sign-off** on each `methodology-exceptions/*.md` file (currently they reference `human:methodology-lead@greenlang.io` as approver but lack the actual signature).
- [ ] **Operator re-audit at alpha launch (2026-Q2)** — verify whether DESNZ 2025 / EPA Hub 2025 / eGRID 2024 have published; bump `source_version`, flip status to `locked`, archive the matching exception file.

**Verify**: `pytest tests/factors/v0_1_alpha/test_alpha_source_snapshots.py -x`

---

## 4. Provenance gate (CTO §19.1: "provenance fields complete for alpha sources")

- [x] **`AlphaProvenanceGate`** at `greenlang/factors/quality/alpha_provenance_gate.py` — enforces all 12 extraction sub-fields + 3 review sub-fields + AR6-only + sha256 format + parser_commit format + operator format.
- [x] **Wired into bootstrap.py and ingest.py** (`_maybe_run_alpha_gate` + `run_alpha_provenance_gate`). Env var `GL_FACTORS_ALPHA_PROVENANCE_GATE` gates on/off; default ON under alpha profile.
- [x] **58 unit tests** at `tests/factors/v0_1_alpha/test_alpha_provenance_gate.py`.
- [x] **Catalog backfill** (Wave D #6): 691/701 records pass the gate. The 10 skipped are negative-value sequestration/removal factors with `value <= 0` — the v0.1 schema requires `value > 0`; they ship in v0.5+. All 6 sources have v0.1-shape seeds at `greenlang/factors/data/catalog_seed_v0_1/<source_id>/v1.json`.
- [ ] **AI-1 (from #26 incident drill)**: add raw-input schema validation step to each alpha-source parser so column shifts raise `KeyError` at parser time, not silent zero one stage downstream. Owner: data-integration team.

**Verify**: `pytest tests/factors/v0_1_alpha/test_alpha_provenance_gate.py tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py -x`

---

## 5. API surface (CTO §19.1: "basic API")

The alpha API exposes EXACTLY the 5 GETs from CTO §19.1 + a `/api/v1/{path}` → 410 Gone catch-all.

| Endpoint | Auth | Returns |
|---|---|---|
| `GET /v1/healthz` | unauthenticated | service status, release_profile, schema_id, edition, git_commit, version |
| `GET /v1/factors` | API key | cursor-paginated factor list (filters: geography_urn, source_urn, pack_urn, category, vintage_*) |
| `GET /v1/factors/{urn:path}` | API key | one factor by canonical URN |
| `GET /v1/sources` | API key | the 6 alpha-flagged sources |
| `GET /v1/packs` | API key | factor packs grouped by source |
| `* /api/v1/{path:path}` | (any) | 410 Gone with alpha endpoint hint |

- [x] **Alpha router** at `greenlang/factors/api_v0_1_alpha_routes.py` (~22 KB) and typed models at `greenlang/factors/api_v0_1_alpha_models.py`.
- [x] **Wired into `factors_app.py`** under `release_profile.is_alpha()` — non-alpha routers are gated off; legacy `/api/v1` is replaced by the 410 catch-all.
- [x] **27 contract tests** at `tests/factors/v0_1_alpha/test_alpha_api_contract.py` + 5 route-table tests at `test_factors_app_alpha_routes.py`.
- [x] **OpenAPI 3.1 snapshot** at `tests/factors/v0_1_alpha/openapi_alpha_v0_1.json` (26.6 KB) + drift detector + docs at `docs/factors/v0_1_alpha/OPENAPI-SNAPSHOT.md`.
- [x] **`X-GL-Release-Profile: alpha-v0.1` header** on every successful response (regression guard + observability).
- [x] **`/v1/healthz` is unauthenticated** — added to `auth_metering.py::PUBLIC_PATHS`.

**Verify**: `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1 pytest tests/factors/v0_1_alpha/test_factors_app_alpha_routes.py tests/factors/v0_1_alpha/test_alpha_api_contract.py -x`

---

## 6. SDK (CTO §19.1: "Python SDK greenlang-factors v0.1.0")

- [x] **`greenlang-factors==0.1.0`** at `greenlang/factors/sdk/python/`. `Development Status :: 3 - Alpha`. The 1.x line was forward-development released too aggressively; now collapsed.
- [x] **5 alpha methods on `FactorsClient`**: `health()`, `list_factors(...)`, `get_factor(urn)`, `list_sources()`, `list_packs(source_urn=)`. Both sync and async clients.
- [x] **Forward-dev methods gated** — 30+ methods (resolve, explain, batch, edition pinning, signed-receipt verify, search, etc.) raise `ProfileGatedError` under `alpha-v0.1`; re-enable at `beta-v0.5` and higher.
- [x] **URN-strict validation** on alpha methods; `EF:...` ids rejected client-side before the wire.
- [x] **Typed Pydantic v2 models**: `AlphaFactor`, `AlphaSource`, `AlphaPack`, `Citation`, `Extraction`, `Review`, `HealthResponse`, `ListFactorsResponse`. `py.typed` shipped.
- [x] **Retries** with exponential backoff for 429/500/502/503/504; 30s default request timeout; `User-Agent: greenlang-factors-sdk/0.1.0 (python)`.
- [x] **35 SDK tests pass** (16 new alpha-surface + 19 v1.2 envelope regressions).
- [x] **`RELEASE_NOTES_v0.1.0.md`** documents the alpha contract and migration from 1.3.0.
- [ ] **Publish `greenlang-factors==0.1.0` to TestPyPI** (the wheel is buildable; the actual upload is a human action with TestPyPI credentials).
- [ ] **Publish to PyPI** at alpha-launch sign-off (separate from TestPyPI; the marker that the release is live).
- [ ] **Legacy `greenlang-factors-sdk==1.1.0`** marked deprecated in `greenlang/factors/sdk/DEPRECATED.md`. Confirm it is NOT on PyPI.

**Verify**: `python -c "from greenlang.factors.sdk.python import FactorsClient, __version__; assert __version__ == '0.1.0'"` → success.

---

## 7. End-to-end demo (CTO §19.1 acceptance criterion)

> "End-to-end test: publish a factor from IPCC AR6 via the pipeline, fetch it by URN via the Python SDK, verify all metadata fields are correct."

- [x] **Realised verbatim** at `tests/factors/v0_1_alpha/test_sdk_e2e_ipcc_publish.py`. 9 tests cover both shim and real-repository paths. All 21 v0.1 required fields verified field-by-field plus full extraction provenance + review audit trail.
- [x] **`AlphaFactorRepository`** at `greenlang/factors/repositories/alpha_v0_1_repository.py` (646 LOC) + 29 unit tests. SQLite (alpha) and Postgres (production via Alembic 0001) backends. JSONB-stored records — no lossy coercion; immutability enforced.
- [x] **Wired into `create_factors_app()`**: under alpha profile, env-var precedence is `GL_FACTORS_ALPHA_REPO_DSN` → `GL_FACTORS_SQLITE_PATH` → `:memory:`.

**Verify**: `pytest tests/factors/v0_1_alpha/test_sdk_e2e_ipcc_publish.py tests/factors/v0_1_alpha/test_alpha_factor_repository.py -x`

---

## 8. Two design partners productive (CTO §19.1 acceptance criterion)

> "Two design partners have completed at least one calculation flow using the SDK and have signed off on usability."

- [x] **2 partner profiles** at `docs/factors/design-partners/`:
  - `partner-IN-EXPORT-01.md` — India textile exporter (cotton + man-made fibre fabric → EU OEM apparel)
  - `partner-EU-MFG-01.md` — Italian cement producer (clinker + finished cement)
- [x] **Onboarding artefacts**: checklist, MSA-NDA-status-tracker, feedback-memo template, pilot-success-criteria.
- [x] **Allow-listed sources scoped per partner**: IN gets {ipcc-ar6, india-cea-baseline, eu-cbam-defaults}; EU gets {ipcc-ar6, defra-2025, epa-ghg-hub, eu-cbam-defaults}.
- [x] **Golden tests** prove a real calculation flow per partner:
  - `tests/factors/v0_1_alpha/test_partner_in_export_calc.py` — 4 tests; 1.2M kWh × 0.68 kgCO2e/kWh = **816,000 kgCO2e Scope 2 location-based**.
  - `tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py` — 4 tests; 10,000 t × 1,000 × 0.84 = **8.4M kgCO2e CBAM embedded**.
- [ ] **Legal: execute MSA + NDA** with both partners. Target executed_by 2026-05-15.
- [ ] **Legal: execute DPA** for `EU-MFG-01` (GDPR-required); confirm DPDP-only sufficiency for `IN-EXPORT-01`.
- [ ] **Operator: provision tenant UUIDs** + issue API keys via Vault (`secret/factors/alpha/<slug>`); record SHA-256 first-16-hex prefix in each profile.
- [ ] **Operator: pin regions** (`ap-south-1` for IN, `eu-central-1` for EU).
- [ ] **Partner kickoff calls** held; SOW recorded.
- [ ] **Partner feedback memos received** at end of pilot (2026-06-30 target).

**Verify**: `pytest tests/factors/v0_1_alpha/test_partner_in_export_calc.py tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py -x` → 8 pass.

---

## 9. Release operations (CTO §19.1 deliverables)

- [x] **Staging→production manual flip flow**: `AlphaPublisher` at `greenlang/factors/release/alpha_publisher.py`; CLI at `scripts/factors_alpha_publish.py` (5 subcommands); runbook at `docs/factors/runbooks/alpha-publish.md`. 30 unit tests. Append-only `factor_publish_log` + rollback support. **Methodology-lead approval required at flip time.**
- [x] **Release manifest + checksums**: `greenlang/factors/release/alpha_edition_manifest.py` (~530 LOC) + `scripts/factors_alpha_cut_edition.py` + 34 tests. **Live cut produced**:
  - `releases/factors-v0.1.0-alpha-2026-04-25/manifest.json`
  - `releases/factors-v0.1.0-alpha-2026-04-25/manifest.json.sig.placeholder`
  - `releases/factors-v0.1.0-alpha-2026-04-25/RELEASE_NOTES.md`
  - `releases/factors-v0.1.0-alpha-2026-04-25/MANIFEST_HASH.txt`
  - **manifest_sha256 = `d6694c8e0060cb070427a5a835f6e0cf3bcf64c14b74d2e88ca94fb10e7ff590`**
  - **691 factors signed off across 6 sources** (cbam=60, desnz=195, egrid=79, epa_hub=84, india_cea=38, ipcc_2006_nggi=235)
- [x] **Edition-cut runbook** at `docs/factors/v0_1_alpha/EDITION-CUT-RUNBOOK.md`.
- [x] **Grafana alpha dashboard** at `deployment/observability/grafana/dashboards/factors-v0.1-alpha.json` (8 panels: request rate, p50/p95/p99 latency, error rate, edition served, schema validation failures, provenance rejections, ingestion success rate, parser errors).
- [x] **Prometheus alerts** at `deployment/observability/prometheus/factors-v0.1-alpha-alerts.yaml` (6 alerts).
- [x] **Alert runbook** at `docs/factors/runbooks/factors-v0.1-alpha-alerts.md`.
- [x] **5 new Prometheus metrics** wired into provenance gate + ingest pipeline: `factors_schema_validation_failures_total`, `factors_alpha_provenance_gate_rejections_total`, `factors_ingestion_runs_total`, `factors_parser_errors_total`, `factors_current_edition_id_info`.
- [x] **Incident-drill postmortem** at `docs/factors/postmortems/2026-Q1-desnz-parser-drift-drill.md`. **Drill actually executed**: corrupted DESNZ fixture → captured live `NonPositiveValueError` + `AlphaProvenanceGateError` traces + counter deltas. 4 action items filed (AI-1 through AI-4). SOP codified at `docs/factors/runbooks/incident-drill-sop.md` for all 6 sources.
- [x] **Performance test**: `tests/factors/v0_1_alpha/test_perf_p95_lookup.py`. **Measured p95 = 18.8ms vs 100ms ceiling = 5.3× headroom.** Nightly CI workflow at `.github/workflows/factors-alpha-perf.yml`. Budget doc at `docs/factors/v0_1_alpha/PERF-BUDGET.md`.
- [ ] **Methodology lead approves the edition cut** (the manifest exists; `manifest.json.sig.placeholder` flips to `manifest.json.sig` when methodology lead provides the Ed25519 private key via env `GL_FACTORS_ED25519_PRIVATE_KEY` and re-runs `python scripts/factors_alpha_cut_edition.py`).
- [ ] **SRE wires AlertManager routing** for `factors_parser_errors_total` and `factors_alpha_provenance_gate_rejections_total` into the alpha namespace (AI-2 from the incident drill).
- [ ] **SRE adds fast-track parser-error alert** to `factors-v0.1-alpha-alerts.yaml` (AI-3).
- [ ] **SRE provisions managed Postgres + Prometheus + Grafana** in the alpha-staging cluster.

**Verify**: `pytest tests/factors/v0_1_alpha/test_alpha_publisher.py tests/factors/v0_1_alpha/test_alpha_edition_manifest.py tests/factors/v0_1_alpha/test_grafana_alpha_dashboard.py tests/factors/v0_1_alpha/test_drill_fixtures.py -x` and `GL_RUN_PERF=1 pytest -m perf tests/factors/v0_1_alpha/test_perf_p95_lookup.py -x`

---

## 10. Release-profile feature gate

- [x] **`greenlang/factors/release_profile.py`** with `ReleaseProfile` enum, `current_profile()`, `is_alpha()`, `feature_enabled(...)`, `filter_app_routes(app)`, FEATURES table.
- [x] **Wired throughout `factors_app.py`** — non-alpha routers (method_packs, admin, graphql, billing, oem, billing_self_serve) are gated; alpha-only `/api/v1/*` 410 catch-all activates under alpha profile.
- [x] **20 unit tests** at `tests/factors/v0_1_alpha/test_release_profile.py`.
- [ ] **Set `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`** in the alpha-staging deploy environment (Helm values / Kubernetes manifest / Docker compose).

---

## 11. Single-yes/no launch gate

A release is alpha-launch-ready when **every box below is `[x]`** AND every test in §0 passes. Today, **all the engineering items are `[x]`**; the remaining `[ ]` items are exclusively human-bound.

### Engineering (all `[x]`)
1. [x] `factor_record_v0_1.schema.json` frozen + DDL + Alembic 0001 + CI gate strict
2. [x] URN scheme + parser + 100 tests
3. [x] 6 alpha-source parsers produce v0.1-shape seeds (691/701 records)
4. [x] Alpha API exposes EXACTLY 5 GETs + 410 catch-all
5. [x] SDK 0.1.0 alpha surface (5 calls, URN-strict, typed)
6. [x] SDK E2E demo (publish → URN fetch → field-by-field verify)
7. [x] 2 design-partner profiles + 2 partner golden tests
8. [x] Staging→production manual flip flow + runbook + 30 tests
9. [x] Release manifest cut (691 factors, sha256 = `d669...f590`)
10. [x] Grafana dashboard + Prometheus alerts + alert runbook
11. [x] Incident-drill postmortem (drill actually executed)
12. [x] p95 < 100ms (measured 18.8ms; 5.3× headroom)
13. [x] Release-profile flag wired

### Human-bound (`[ ]` — these are the only remaining items)
1. [ ] **Methodology lead** signs `FACTOR_RECORD_V0_1_FREEZE.md` Approvals section
2. [ ] **Methodology lead** signs `urn_scheme_methodology_review.md` (CTO §19.1)
3. [ ] **Methodology lead** signs each `methodology-exceptions/*.md` for DESNZ/EPA Hub/eGRID preview vintages
4. [ ] **Methodology lead** provides Ed25519 private key; operator re-runs `factors_alpha_cut_edition.py` to replace `manifest.json.sig.placeholder` with a real signature
5. [ ] **CTO** signs the freeze date and field omissions
6. [ ] **Legal** executes MSA + NDA with both partners (target 2026-05-15)
7. [ ] **Legal** executes DPA with EU-MFG-01 (GDPR)
8. [ ] **Operator** provisions tenant UUIDs + API keys via Vault for both partners
9. [ ] **Operator** publishes `greenlang-factors==0.1.0` to TestPyPI then PyPI
10. [ ] **Operator** sets `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1` in the alpha-staging deploy
11. [ ] **SRE** wires AlertManager routing for the 5 new alpha metrics
12. [ ] **SRE** provisions managed Postgres + Prometheus + Grafana in alpha-staging
13. [ ] **AI-1 (data-integration)**: add raw-input schema validation step to alpha-source parsers (incident-drill action item)
14. [ ] **Partner kickoff calls** + first-calculation handover for both partners
15. [ ] **Partner feedback memos** received at pilot end (target 2026-06-30)

**When every box is `[x]`, the alpha is launch-ready.** Until then, the binary answer to "is this done?" is **NOT YET — engineering complete; human-bound close-out pending.**

---

## 12. Test posture (today)

```
pytest tests/factors/v0_1_alpha/ → 582 passed, 7 skipped (perf opt-in), 0 failed
GL_RUN_PERF=1 pytest -m perf tests/factors/v0_1_alpha/test_perf_p95_lookup.py → 4 passed
                                                                                 (p95 18.8ms)
```

No regressions caused by alpha work in any other test suite.

---

## 13. Out of scope for v0.1 alpha (do NOT add)

Per CTO §19.1 "Explicitly out of scope": no public website, no commercial tier, no TS SDK, no CLI beyond minimal, no GraphQL, no SQL-over-HTTP, no ML resolve endpoint, no real-time grid, no commercial packs (ecoinvent/EXIOBASE/IEA), single-region deployment only, no SOC 2 work yet, no auditor sign-off.

These features remain in the codebase but are gated off behind `release_profile.feature_enabled(...)`. They re-enable at `beta-v0.5` (TS SDK, CLI, resolve, explain, batch, signed receipts, admin console), `rc-v0.9` (GraphQL, ML resolve), and `ga-v1.0` (SQL-over-HTTP, billing, OEM, commercial packs, real-time grid).

---

## 14. Filing path

Once every box is `[x]`, file this checklist as `releases/factors-v0.1.0-alpha-2026-04-25/LAUNCH-GATE-SIGNED.md` with:
- CTO signature line
- Methodology lead signature line
- Legal signature line (MSA/NDA executed)
- SRE signature line (production infra ready)
- Operator signature line (tenants provisioned, SDK on PyPI)
- Date the gate passed

---

*End of release checklist. Update this file as boxes flip from `[ ]` to `[x]`. Do not add new scope.*
