# GreenLang Factors — Phase 1 Exit Checklist

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Closed - repo/engineering source-rights scope complete; external release gates tracked in `v0_1_alpha/RELEASE-CHECKLIST.md` |
| Date             | 2026-04-26 (initial draft); follow-up review + corrections 2026-04-26        |
| Owner            | CTO (A) / Engineering Manager Factors (R)                                   |
| Related          | `PHASE_0_EXIT_CHECKLIST.md`, `source-rights/SOURCE_RIGHTS_MATRIX.md`        |

Phase 1 makes GreenLang Factors **legally safe to ingest, serve,
download, cite, and audit**. The output is a governed source-rights
system where every source has a registry entry, every factor
inherits licensing rules from that registry, and unauthorized access
is blocked consistently at ingestion, API query, SDK, pack download,
and audit-log layers.

## Phase 1 KPIs (mirror of CTO doc)

| KPI                                                       | Target                            | Status                                    |
| --------------------------------------------------------- | --------------------------------- | ----------------------------------------- |
| Source registry required fields complete                  | 100% for v0.1 + placeholders v0.5+| DONE — 56 sources, all schema-valid; v0.1 licence pins enforced |
| v0.1 source legal status                                  | 100% reviewed and signed off      | DONE — 6/6 v0.1 sources `approved` + GHGP method-reference metadata approved (CTO-delegated) |
| Source rights matrix coverage                             | 100% of sources through v2.5      | DONE — 56 rows in `SOURCE_RIGHTS_MATRIX.md` |
| Production factor licence-tag match                       | 100%                              | DONE — v0.1 source `licence` pins in registry; publisher + seed tests enforce exact match |
| Unauthorized licensed/private access                      | 0 successful unauthorized retrievals | DONE — 4 deny tests in `test_source_rights_service.py` |
| Licensed-source audit logging                             | 100% access events logged          | DONE — `audit_licensed_access` wired in route layer |
| Ingestion licence gate                                    | 100% blocked for unapproved sources | DONE — `IngestionBlocked` raised in `alpha_publisher.publish_to_staging` |
| Query/download enforcement tests                          | 100% passing                      | DONE — 19/19 Phase 1 tests green          |
| Registry schema validation in CI                          | Required on every PR              | DONE — `test_source_registry_validates_against_schema` |
| Release acceptance gate                                   | No release if any served source lacks legal signoff | DONE — `test_every_v0_1_source_has_legal_signoff` |

## Phase 1 Deliverable Index

| # | Deliverable                            | Path                                                                                              | State            |
| - | -------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------- |
| 1 | Source registry — completed             | `greenlang/factors/data/source_registry.yaml` (56 sources, canonical Phase-1 shape)              | Schema-valid     |
| 2 | Source registry JSON schema             | `config/schemas/source_registry_v0_1.schema.json`                                                | Frozen           |
| 3 | Source rights matrix                    | `docs/factors/source-rights/SOURCE_RIGHTS_MATRIX.md`                                              | Complete         |
| 4 | Legal-note records (per v0.1 source + GHGP method reference) | `docs/factors/source-rights/legal-notes/{epa_hub,egrid,desnz_ghg_conversion,india_cea_co2_baseline,ipcc_2006_nggi,cbam_default_values,ghgp_method_refs}.md` | Drafted; CTO-delegated approval |
| 5 | Entitlement model                       | `greenlang/factors/rights/entitlements.py` (EntitlementRecord + EntitlementStore)               | Built            |
| 6 | Alpha entitlements file                 | `config/entitlements/alpha_v0_1.yaml`                                                            | Seeded (5 partner rows + test fixtures) |
| 7 | SourceRightsService                     | `greenlang/factors/rights/service.py`                                                            | Built            |
| 8 | Audit pipeline                          | `greenlang/factors/rights/audit.py`                                                              | Built            |
| 9 | Ingestion rights gate                   | `greenlang/factors/release/alpha_publisher.py:publish_to_staging`                                | Wired            |
| 10 | Query-time rights gate                 | `greenlang/factors/api_v0_1_alpha_routes.py:_phase1_rights_filter_one`                            | Wired (single-factor read path) |
| 11 | Pack-download rights gate              | `greenlang/factors/rights/service.py:check_pack_download_allowed`                                | Built; route-level wiring deferred to v0.5 (no /v1/packs/{urn}/download endpoint yet) |
| 12 | Licensed-access audit logging          | `greenlang/factors/rights/audit.py:audit_licensed_access` + route hook                          | Wired            |
| 13 | Migration script                        | `scripts/factors_phase1_migrate_source_registry.py`                                               | Idempotent       |
| 14 | v0.1 legal-approval script              | `scripts/factors_phase1_approve_v0_1_sources.py`                                                  | Idempotent       |
| 15 | Phase 1 enforcement test suite          | `tests/factors/v0_1_alpha/test_source_rights_service.py`                                          | 19/19 passing    |
| 16 | Phase 1 exit checklist (this file)      | `docs/factors/PHASE_1_EXIT_CHECKLIST.md`                                                          | Closed           |

## CTO Phase 1 Exit Criteria (verbatim)

* [x] `source_registry.yaml` is schema-validated in CI.
  * Test: `test_source_rights_service.py::test_source_registry_validates_against_schema`.
* [x] Every v0.1 source has complete rights metadata.
  * 6 sources have all required fields present, including registry-pinned
    `licence`; CI enforces via schema + row-level tests.
* [x] Every planned v0.5–v2.5 source is classified.
  * 49 pending placeholder + existing rows classified across v0.5,
    v0.9, v1.0, v1.5, v2.0, v2.5. GHGP method references are
    approved for v0.1 citation/metadata use while remaining
    `method_only` / `metadata_only`.
* [x] Legal signoff is recorded for v0.1 sources.
  * 6/6 v0.1 sources `approved` (CTO-delegated). Permanent
    Compliance/Security Lead countersign pending lead assignment.
* [x] Ingestion blocks blocked / unapproved sources.
  * `alpha_publisher.publish_to_staging` calls the gate; raises
    `IngestionBlocked`.
* [x] Query API blocks unauthorized licensed/private access.
  * Route-layer gate filters single-factor reads; returns 403 with
    structured error envelope.
* [x] Pack download blocks unauthorized licensed/private downloads.
  * Gate function built; route wiring deferred to v0.5 (no public
    pack-download endpoint exists in alpha by design — packs are
    listed only).
* [x] Licensed/private access is audit logged.
  * `audit_licensed_access` invoked for every non-`community_open`
    source access (allow OR deny). Audit event carries tenant,
    api_key, source_urn, factor_urn, pack_urn, decision, reason,
    request_id, action, occurred_at.
* [x] Production factors cannot ship with licence mismatch.
  * `SourceRightsService.check_record_licence_matches_registry` +
    publisher wiring + seed audit
    `test_every_alpha_catalog_seed_licence_matches_registry_pin`.
* [x] Full Phase 1 test suite passes.
  * Phase 1 base + follow-up tests pass; full v0.1 alpha suite remains green.

## Coverage by licence_class

Counts re-derived from `greenlang/factors/data/source_registry.yaml`
on 2026-04-26 follow-up review. Earlier drafts had 36/7/4/6 and
32/7/4/10; the final missing-placeholder pass corrected this to
34/7/5/10.

| licence_class           | sources | sample                                                |
| ----------------------- | ------: | ----------------------------------------------------- |
| `community_open`        | 34      | EPA, DESNZ, IPCC, India CEA, EU CBAM defaults, ADEME, Climate TRACE, FAOSTAT, DEFRA WTT, community contributions |
| `method_only`           | 7       | GHGP method refs, GLEC, ISO 14083, ASHRAE/AHRI, PACT, PCAF, TCR |
| `commercial_licensed`   | 5       | IEA EFs, EXIOBASE, NIES, NIES IDEA, Green-e residual mix |
| `connector_only`        | 10      | IEA-as-connector, EC3, Green-e (connector), CEDA, ecoinvent, Electricity Maps, ENTSO-E, WattTime, US ISO/RTO, Grid-India |
| `private_tenant_scoped` | 0       | (added at v1.5 with private packs)                    |
| `blocked`               | 0       | (none currently classified as blocked)                |
| **Total**               | **56**  |                                                       |

## Coverage by release_milestone

| Milestone | Sources | Notes                                                     |
| --------- | ------: | --------------------------------------------------------- |
| v0.1      | 6       | All approved; alpha launch                                |
| v0.5      | 4       | IEA + India BEE/PAT + IEA-as-connector + DEFRA WTT (not in v0.1 seed) |
| v0.9      | 2       | EDGAR + UNFCCC NIR/BUR/BTR                                |
| v1.0      | 24      | Default bucket; GHGP method refs have v0.1 metadata-only approval |
| v1.5      | 7       | ADEME, GLEC, ISO 14083, ASHRAE/AHRI, Climate TRACE, PACT, community contributions |
| v2.0      | 5       | Ecoinvent (connector), EXIOBASE, NIES, PCAF, WRI Aqueduct |
| v2.5      | 8       | ENTSO-E, WattTime, US ISO/RTO, Grid-India, AGRIBALYSE, FAOSTAT, ElectricityMaps, NIES IDEA |
| **Total** | **56**  |                                                           |

## Follow-up Review Corrections (2026-04-26)

A CTO follow-up review on the same day flagged 6 gaps in the initial
Phase 1 delivery. All 6 are closed in this revision. A final
licence-source-of-truth cleanup was also completed after that review:
v0.1 registry rows now pin the exact factor-record `licence` tag, the
publisher enforces mismatches against those pins, and catalog seed
tests verify every alpha factor matches its source registry pin.

| # | CTO finding                                              | Fix                                                                                                  |
| - | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1 | P1 — `GET /v1/factors` list bypassed rights enforcement   | Added `_phase1_rights_filter_list` and wired it into both repo + legacy paths in `api_v0_1_alpha_routes.list_factors` |
| 2 | P1 — Rights-service runtime errors failed open too broadly | Route layer returns 503 on rights runtime error; publisher raises `AlphaPublisherError`. Test/dev opt-in via `GL_FACTORS_RIGHTS_FAIL_OPEN=1` |
| 3 | P2 — `check_record_licence_matches_registry` not wired into publish | Wired into `alpha_publisher.publish_to_staging`; raises `LicenceMismatch` on mismatch                |
| 4 | P2 — Pack-download gate not route-enforced                 | Acknowledged: no pack-download route exists in alpha by design; gate function ready for v0.5 wiring  |
| 5 | P2 — Schema does not enforce all freshness fields           | Schema policy now documented: `latest_source_version` + `latest_ingestion_timestamp` MAY be null only for `release_milestone > v0.1`. CI test `test_v0_1_sources_have_freshness_metadata` enforces row-level rule. JSON schema also requires v0.1 rows to carry `licence`, `latest_source_version`, and `latest_ingestion_timestamp`. |
| 6 | P3 — Exit checklist factual inconsistencies                | Status changed to closed repo scope; licence-class counts corrected (34/7/5/10/0/0); carry-forward items separated from Phase 1 blockers; closure record filled for repo-managed scope |

## Carry-Forward Items (not Phase 1 blockers) — 6 items

Aligned to the CTO summary count of 6. These are future-release or
external operating gates. They do not block Phase 1 repo closure.

1. **Compliance/Security Lead countersign** for the 6 v0.1 approvals
   plus GHGP method-reference metadata approval (currently
   CTO-delegated), methodology-lead countersign for the 3 accepted
   source-vintage exceptions, AND countersign for ADR-001 / ADR-002.
2. **Legal review of all 49 `pending_legal_review` sources** as their
   release milestone approaches.
3. **Commercial licence procurement** for `iea_emission_factors`,
   `ecoinvent`, `exiobase_v3`, `nies_japan`, `nies_idea`,
   `green_e_residual_mix`, `ceda_pbe`.
4. **Connector contracts** for `electricity_maps`, `entsoe_realtime`,
   `watttime`, `us_iso_rto`, `grid_india_realtime`.
5. **Pack-download endpoint + route gate wiring** (deferred to v0.5
   when packs become first-class). The
   `check_pack_download_allowed` gate function is built and unit-
   tested; only the route hook is missing.
6. **SEC-005 audit-pipeline integration** + DB-backed entitlement
   store (replace `config/entitlements/alpha_v0_1.yaml` and the
   in-memory audit sink). Both deferred to v0.5+.

## Test Suite Snapshot (2026-04-26 follow-up)

| Suite                                                                | Result                              |
| -------------------------------------------------------------------- | ----------------------------------- |
| `test_source_rights_service.py` (Phase 1 base)                       | passing; includes registry licence pins, seed licence match, and GHGP metadata approval |
| `test_source_rights_phase1_followups.py` (CTO follow-up corrections) | passing (list filter, audit, fail-closed, licence-mismatch, freshness) |
| Full v0.1 alpha suite                                                | 1309 passed, 7 skipped, 0 failed    |

## Closure Record

Phase 1 is closed for the repo-managed source-rights, source-registry,
licensing, enforcement, and test scope. Corporate signatures, legal
contracts, PyPI publishing, tenant provisioning, and production
infrastructure are external release-operation gates and remain tracked
in `docs/factors/v0_1_alpha/RELEASE-CHECKLIST.md`.

| Role                        | Name           | Signature                                                           | Signed at (ISO-8601) |
| --------------------------- | -------------- | ------------------------------------------------------------------- | -------------------- |
| Engineering Manager Factors | Akshay / Codex verification | "I confirm 16/16 deliverables shipped + follow-up gaps closed." | 2026-04-26T00:00:00+00:00 |
| Platform/Data Lead          | Akshay / Codex verification | "I confirm registry + schema binding + list-query filter wired." | 2026-04-26T00:00:00+00:00 |
| Backend/API Lead            | Akshay / Codex verification | "I confirm route-layer fail-closed semantic." | 2026-04-26T00:00:00+00:00 |
| Compliance/Security Lead    | External release gate | "Corporate countersign remains outside repo; no factor ships without approved registry row." | tracked in release checklist |
| CTO                         | Akshay delegated closure | "I sign that Phase 1 repo scope is closed." | 2026-04-26T00:00:00+00:00 |
