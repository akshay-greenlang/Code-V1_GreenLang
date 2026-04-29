# Phase 3 Exit Checklist — Raw Ingestion Framework

> **Authority**: CTO Phase 3 brief (2026-04-28).
> **Owner**: Factors Engineering.
> **Reviewers**: CTO, Methodology Lead, Backend Lead, Head of Data, Test Lead, Legal.
> **Predecessor**: Phase 2 sign-off must land before Phase 3 is declared formally complete.

Every technical box must be ticked and every accountable role must sign off before Phase 3 is declared complete and Phase 4 (Resolution Engine + Pricing) may start.

> **Engineering verification snapshot — 2026-04-29**: block-by-block audit by parallel Codex agents (post-commit `653dca53` Wave 1.0, `<wave1.5..wave3.0>` follow-ups, and the 2026-04-29 audit close-out). **38/38 engineering boxes GREEN.** `python -m pytest tests/factors/v0_1_alpha tests/ci -q -W ignore::DeprecationWarning` reports **2148 passed, 17 skipped, 0 failed** (337s). New-module branch coverage on the four Phase 3 packages: `pipeline.py` 91.11%, `runner.py` 93.21%, `run_repository.py` 95.41%, `diff.py` 99.17%, `cli_ingest.py` 92.44% — combined 94.50%. Phase 2 acceptance unchanged: 10/10 PASS. Sign-off rows below remain blank as designed.

---

## Block 1 — Seven-stage pipeline contract (CTO §3.1)

- [x] `IngestionPipeline` runner enforces stages in order: fetch → parse → normalize → validate → dedupe → stage+diff → publish.
- [x] Each stage advances `ingestion_runs.status`; failure short-circuits with stage name + structured error.
- [x] No stage may be skipped; resume mode requires explicit `--from-stage <name>` after a failed run.
- [x] Stage transitions are atomic (one DB transaction per stage advance).
- [x] Phase 2 publish gates (7-gate orchestrator) run unchanged inside stage 7.
- [ ] **Sign-off**: Backend Lead + CTO.

## Block 2 — Artifact storage contract (CTO §3.2)

- [x] Every ingestion run writes a `raw_artifacts` row with: raw_bytes_uri, source_url, fetched_at, sha256, bytes_size, content_type, source_version, source_publication_date, parser_module, parser_function, parser_version, parser_commit, operator, licence_class, redistribution_class, ingestion_run_id, status.
- [x] Certified factors carry `extraction.raw_artifact_uri` AND `extraction.raw_artifact_sha256`; sha256 verified at every stage transition.
- [x] Parser output is rejected when artifact storage write fails (negative test passes).
- [x] Manual / PDF / OCR sources require uploaded raw artifact + reviewer-approved `reviewer_notes` JSONB.
- [ ] **Sign-off**: Head of Data + Backend Lead.

## Block 3 — Fetcher / parser families (CTO §3.3)

- [x] Five families dispatch from `source_registry.yaml`: excel, csv_json_xml, pdf_ocr, api_webhook, ecospold_mrio.
- [x] Excel-family validates workbook tabs + required columns + header row + unit string + vintage label.
- [x] CSV/JSON/XML-family validates schema shape + required fields + numeric ranges + units + timestamps + geography codes.
- [x] PDF/OCR-family stores PDF + extracted-table artifact + OCR confidence + reviewer notes + manual correction log.
- [x] API/webhook-family captures request URL + response body + response timestamp + API version + pagination cursor + webhook event ID.
- [x] EcoSpold2/MRIO-family handles zipped multi-file artifacts + system-model metadata + activity/process IDs + geography map + licence entitlement.
- [x] DEFRA Excel reference source migrated end-to-end (Wave 1.5).
- [ ] **Sign-off**: Methodology Lead + Backend Lead.

## Block 4 — Dedupe / supersede / diff rules (CTO §3.4)

- [x] Same URN + identical payload → unchanged, no DB write (positive test).
- [x] Same fingerprint + changed value → new URN version OR `supersedes_urn` per URN policy.
- [x] Duplicate source rows in one run → dedupe before staging (first wins, second logged).
- [x] Missing-from-staging production record → flagged removal candidate, never auto-deleted.
- [x] Cross-source supersede blocked unless methodology-lead approval flag set.
- [x] Diff (MD + JSON) shows: value / unit / boundary / geography / methodology / licence / citation / parser-version changes + counters.
- [x] Reviewers can decide publish-yes/no from diff alone (acceptance test: synthetic diff matches expected MD/JSON).
- [ ] **Sign-off**: Methodology Lead.

## Block 5 — Operational interfaces (CTO §3.5)

- [x] Click group `gl factors ingest` exposes: fetch / parse / run / diff / stage / publish / rollback / status.
- [x] Run-status enum complete: created, fetched, parsed, normalized, validated, deduped, staged, review_required, published, rejected, failed, rolled_back.
- [x] Each command emits structured JSON when `--json` is set, otherwise human-readable summary.
- [x] Exit codes mirror the run-status enum.
- [x] `gl factors ingest publish --approved-by human:<email>` requires approver identity (rejects bot operators).
- [ ] **Sign-off**: Backend Lead + GL-SpecGuardian.

## Block 6 — Snapshot tests + pipeline tests (CTO Test Plan)

- [x] Each migrated parser has committed golden artifact + expected normalized output under `tests/factors/v0_1_alpha/phase3/parser_snapshots/`.
- [x] Snapshot tests fail on: changed workbook tab names, missing required columns, unit drift, source table-shape drift, impossible values, missing citations, missing licence tags, missing raw artifact checksum, missing row/sheet/table reference, parser output drift without `UPDATE_PARSER_SNAPSHOT=1` regeneration.
- [x] Pipeline e2e tests cover: fetch→stage happy path; stage→publish; duplicate dedupe; changed-row supersede; missing-artifact-blocks-publish; licence-mismatch-blocks-ingestion; invalid-ontology-blocks-staging; parser-failure-records-failed-run; rollback-demotes-not-deletes.
- [x] Source-type fixture tests: excel + csv + json + pdf + api + zipped multi-file MRIO.
- [x] Coverage ≥ 85% on new modules (`pipeline.py`, `runner.py`, `cli/ingest.py`, snapshot harness).
- [ ] **Sign-off**: Test Lead.

## Block 7 — CI gates (CTO Test Plan)

- [x] CI blocks PRs that change a parser module without a corresponding snapshot regeneration commit.
- [x] CI blocks PRs that change `source_registry.yaml` parser_version without a release-note entry.
- [x] CI blocks PRs that publish a certified factor lacking raw artifact metadata (gate 6 audit).
- [x] CI blocks ingestion of a source whose registry status is `pending_legal_review`, `blocked`, or a future `release_milestone` if the run targets `production`.
- [x] CI audit confirms zero ingestion-runner code paths can write to `factor` table bypassing the Phase 2 7-gate orchestrator.
- [x] `.github/workflows/factors-ingestion-check.yml` is committed to git and visible to GitHub Actions.
- [ ] **Sign-off**: GL-CodeSentinel + CTO.

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

When every box is ticked and every role has signed: **Phase 3 is COMPLETE**. Open Phase 4 (Resolution Engine + Pricing) work tracker.

---

## Engineering evidence — per-box (2026-04-29 audit close-out)

> Every technical box maps to a single line of evidence below. Cite during sign-off; if a reviewer disputes a box, point at the file/test cited here. Mirrors the per-block evidence pattern at the bottom of `docs/factors/PHASE_2_EXIT_CHECKLIST.md`.

### Block 1 — Seven-stage pipeline contract

| Box | Evidence |
|---|---|
| Stage order enforced | `greenlang/factors/ingestion/pipeline.py:81-96` — `Stage` enum + `_STAGE_REQUIRED_PREDECESSOR` mapping. `runner.py` per-stage methods call `assert_stage_precondition` before work. |
| Status advances + structured error | `runner.py:_mark_failed` records stage + structured `error_json`. Each stage method invokes `update_status` on success. Verified by `test_pipeline_e2e_happy_path.py` and `test_pipeline_parser_failure_records_failed_run.py`. |
| `--from-stage` resume implemented | `runner.py` `IngestionPipelineRunner.resume(run_id, *, from_stage)` rejects when run is not failed/rejected (`StageOrderError`); `cli_ingest.py` `run` subcommand exposes `--from-stage`. Tested by `test_pipeline_resume_from_stage.py` (3 cases). |
| Atomic stage transitions | `run_repository.py` `_sqlite_txn()` context manager wraps every multi-statement SQLite write in BEGIN IMMEDIATE / COMMIT with rollback. Postgres path uses `with conn:` block per write. Tested by `test_run_repository_atomic_transactions.py`. |
| Phase 2 gates run unchanged | `runner.py:_get_orchestrator()` calls `factor_repo._get_or_build_publish_orchestrator()` — reuses Phase 2 instance, no parallel construction. `test_pipeline_publish_atomic.py` confirms 7 gates fire on every staged URN. |

### Block 2 — Artifact storage contract

| Box | Evidence |
|---|---|
| Full 16+ field row | V510 (`deployment/database/migrations/sql/V510__factors_v0_1_phase3_source_artifacts_full_contract.sql`) adds the 9 missing columns to V505 `source_artifacts`. `IngestionRunRepository.upsert_source_artifact` writes all 17 fields. `test_pipeline_source_artifacts_full_contract.py::test_full_defra_pipeline_lands_full_source_artifact_row` asserts each. |
| extraction URI + sha256 + verification | `_phase3_pdf_ocr_adapters.py:657-658` and the other 4 family parsers all emit `extraction.raw_artifact_uri` + `raw_artifact_sha256`. Validate stage rejects records lacking either. `test_pipeline_artifact_required.py` covers both negatives. |
| Storage failure rejects publish | `test_pipeline_artifact_store_failure.py` — monkeypatches `LocalArtifactStore.put_bytes` to raise `OSError`; asserts `runs.status='failed'`, `error_json.stage='fetch'`, zero factor rows AND zero source_artifacts rows committed. |
| PDF/OCR `reviewer_notes` JSONB | V509 (`V509__factors_v0_1_phase3_reviewer_notes.sql`) adds the JSONB column. `_phase3_pdf_ocr_adapters.py:607-614` populates `extraction_method`, `ocr_confidence_min`, `manual_corrections`, optional `low_confidence_cell`. `test_pdf_ocr_family_e2e.py` (8 tests) covers reviewer-approved correction publish path + V509 round-trip. |

### Block 3 — Fetcher / parser families

| Box | Evidence |
|---|---|
| 5 families dispatch from registry | `grep -E "family:" greenlang/factors/data/source_registry.yaml \| sort -u` returns exactly: excel, csv_json_xml, pdf_ocr, api_webhook, ecospold_mrio. |
| Excel validation | `_phase3_adapters.py` `Phase3DEFRAExcelParser` (and EPA/eGRID/CEA/BEE/IEA siblings) validate workbook tab + required columns + header row + unit string + vintage label; raise `ParserDispatchError` with offending sheet/row. `test_excel_family_e2e.py` 31 cases. |
| CSV/JSON/XML validation | `_phase3_csv_json_xml_adapters.py` parsers validate schema shape + numeric ranges (forestry-and-land-use carve-out for negatives) + ISO-8601 timestamps + ISO-3 geography codes. `test_csv_json_xml_family_e2e.py` 18 cases. |
| PDF/OCR storage + confidence + reviewer notes | `_phase3_pdf_ocr_adapters.py` `Phase3PdfOcrParser` extracts via `pdfplumber`, emits `PdfCell` per-cell confidence, builds `reviewer_notes` payload with extraction_method + ocr_confidence_min + manual_corrections. `test_pdf_ocr_family_e2e.py` 8 cases. |
| API/webhook captures full request/response context | `webhook.py` `WebhookArtifact` captures source_url + body_bytes + response_timestamp + api_version (X-GL-API-Version) + pagination_cursor + event_id (X-GL-Event-Id). HMAC verify via X-GL-Signature; idempotency LRU 10k/24h. `test_api_webhook_family_e2e.py` 9 cases. |
| EcoSpold/MRIO zip + system model + entitlement | `_phase3_ecospold_mrio_adapters.py` parses `.spold` XML via stdlib ElementTree; `zip_artifact.py` 4-guard bomb defense (member count 100k, ratio 1024:1, 500MB total, path traversal); entitlement gate raises `ParserDispatchError(ECOSPOLD_ENTITLEMENT_MISSING)`. `test_ecospold_mrio_family_e2e.py` 10 cases. |
| DEFRA reference end-to-end | `test_defra_reference_e2e.py` 4 tests. Wave 1.5 fixture `tests/factors/v0_1_alpha/phase3/fixtures/defra_2025_mini.xlsx` deterministic. Snapshot `parser_snapshots/defra_2025__0.1.0.golden.json` committed. |

### Block 4 — Dedupe / supersede / diff rules

| Box | Evidence |
|---|---|
| URN+payload identical → no write | `alpha_publisher.py:444-451` `unchanged` counter via JSON-identical comparison. Stage 6 retains unchanged URNs without new INSERT. |
| Fingerprint match + changed value → supersede | `dedupe_rules.py:36-54` `duplicate_fingerprint()` 8-tuple. `runner.py:dedupe()` 446-490 computes supersede pairs. `test_pipeline_dedupe_supersede.py::test_changed_value_across_runs_creates_supersede_pair` confirms historical row immutable. |
| Duplicate within one run → first wins | `runner.py:459-464` `seen_fingerprints` dict; second occurrence increments counter and skips. `test_duplicate_record_in_one_run_logged_not_written_twice` asserts final DB count ≤ 1. |
| Missing-from-staging → flagged not deleted | `runner.py:488-492` `removal_candidates`. `alpha_publisher.py:438-442` lists separately. Diff MD line 364-370 surfaces removals. |
| Cross-source supersede blocked | `runner.py:482-485` cross-source check; raises `DedupeRejectedError` unless `allow_cross_source_supersede=True`. `exceptions.py:158-180` typed exception. |
| Diff covers all 8 attribute kinds | `diff.py:171-178` attributes list (value/unit/boundary/geography/methodology/licence/citation/parser_version). `serialize_json` and `serialize_markdown` emit summary + per-bucket sections. |
| Reviewers decide from diff alone | `test_diff_review_decidable.py` 3 tests confirm Summary table at top, full removals listed, JSON has summary+changes+supersedes. `test_diff_serialization_deterministic.py` byte-identical across runs. |

### Block 5 — Operational interfaces

| Box | Evidence |
|---|---|
| 8 Click subcommands | `python -c "from greenlang.factors.cli_ingest import ingest_group; print(sorted(ingest_group.commands.keys()))"` returns `['diff', 'fetch', 'parse', 'publish', 'rollback', 'run', 'stage', 'status']`. |
| 12-value RunStatus enum | `pipeline.py:57-78` `RunStatus`. `python -c "from greenlang.factors.ingestion.pipeline import RunStatus; print(len(list(RunStatus)))"` returns 12. |
| Structured JSON / human summary | `cli_ingest.py:114-119` `_emit_success` branches on `json_mode`. All 8 commands accept `--json` via `_global_options` decorator. `test_cli_ingest_branches.py` 58 tests cover success + error JSON output. |
| Exit codes mirror run-status | `cli_ingest.py:72-85` `_STATUS_EXIT_CODES` map; `_exit_for_status` applies. 0 success/in-progress, 1 rejected, 2 failed, 3 rolled_back, 4 unknown id, 5 invocation error. |
| Approver enforcement | `cli_ingest.py:67` `_HUMAN_APPROVER_RE = '^human:[^@\\s]+@[^@\\s]+\\.[^@\\s]+$'`; `publish_cmd` and `rollback_cmd` reject bot operators with exit 5. `test_cli_ingest_branches.py` covers invalid `--approved-by`. |

### Block 6 — Snapshot + pipeline tests

| Box | Evidence |
|---|---|
| Per-parser golden artifacts | 13 committed under `tests/factors/v0_1_alpha/phase3/parser_snapshots/`: bee, cea, climate_trace, defra, ecoinvent, edgar, egrid, entsoe, epa, exiobase, iea, pact_api, private_pack — all `<source>__<version>.golden.json`. |
| Drift detectors | `parser_snapshots/_helper.py` — `_diff_table_shape` (column drift / changed tabs / missing columns), `_diff_unit_strings` (unit drift), `_diff_missing_provenance` (missing licence/sha256/citations/row_ref), `_diff_impossible_values` (negative value with carve-out, zero with carve-out, gwp_horizon ∉ {20,100,500}, vintage_end<vintage_start, confidence ∉ [0,1]), deep equality (parser output drift). `UPDATE_PARSER_SNAPSHOT=1` regenerates. `test_snapshot_framework_smoke.py` 20 tests. |
| Pipeline e2e scenarios | 9 test files under `tests/factors/v0_1_alpha/phase3/`: `test_pipeline_e2e_happy_path` (fetch→stage), `test_pipeline_publish_atomic` (stage→publish), `test_pipeline_dedupe_supersede` (dedupe + supersede), `test_pipeline_artifact_required` (missing artifact blocks), `test_pipeline_licence_blocked` (licence mismatch), `test_pipeline_invalid_ontology_blocked` (ontology FK), `test_pipeline_parser_failure_records_failed_run`, `test_pipeline_rollback_demotes_not_deletes`, `test_pipeline_artifact_store_failure`. |
| Source-type fixture coverage | xlsx (DEFRA + EPA + eGRID + CEA + BEE + IEA), csv (EDGAR + Climate TRACE), xml (ENTSO-E), pdf (UNFCCC BUR), json (PACT API + private pack + webhook event), zip multi-file MRIO (ecoinvent + EXIOBASE) — all under `phase3/fixtures/`. |
| Coverage ≥85% on new modules | `pipeline.py` 91.11%, `runner.py` 93.21%, `run_repository.py` 95.41%, `diff.py` 99.17%, `cli_ingest.py` 92.44%. Combined 94.50%. Closed via 173 new branch tests in `test_cli_ingest_branches.py` (58) + `test_runner_branches.py` (49) + `test_run_repository_branches.py` (42) + `test_diff_branches.py` (24). |

### Block 7 — CI gates (Wave 3.0)

| Box | Gate | Script / file | Contract |
|---|---|---|---|
| 1 — parser change requires snapshot regen | gate-1 parser-snapshot-drift | `scripts/ci/check_parser_snapshot_drift.py` | Diffs `--base-ref..--head-ref`. If any `greenlang/factors/ingestion/parsers/**.py` changed AND no `tests/factors/v0_1_alpha/phase3/parser_snapshots/**.json` changed AND parser does not carry the override marker line, exit 1. Override allows regen via `UPDATE_PARSER_SNAPSHOT=1`. Tested by `tests/ci/test_check_parser_snapshot_drift.py` (4 cases). |
| 2 — registry parser_version requires CHANGELOG | gate-2 source-registry-version | `scripts/ci/check_source_registry_version.py` | PyYAML-parses old/new `greenlang/factors/data/source_registry.yaml`; for every `source_id` whose `parser_version` differs, requires a section header matching `^## .*<source_id>.* <new_version>` (case-insensitive) in `docs/factors/source-registry/CHANGELOG.md`. Tested by `tests/ci/test_check_source_registry_version.py` (4 cases). CHANGELOG bootstrap committed. |
| 3 — new factor records carry raw artifact metadata | gate-3 raw-artifact-metadata | `scripts/ci/check_raw_artifact_metadata.py` | For every record ADDED in the diff under `greenlang/factors/data/catalog_seed/**/*.json`, requires `extraction.raw_artifact_uri` (non-empty) AND `extraction.raw_artifact_sha256` (lowercase 64-hex). Pre-existing records lacking these are appended to `docs/factors/source-registry/PHASE_3_BACKFILL_TODO.md` rather than failed. Tested by `tests/ci/test_check_raw_artifact_metadata.py` (5 cases). |
| 4 — pending_legal / blocked / future-milestone sources cannot run in production | gate-4 pending-legal-blocked | `scripts/ci/check_pending_legal_blocked.py` + runtime helper `greenlang/factors/ingestion/source_safety.py` | Static-scans `.github/workflows/*.yml` and `scripts/**/*.{sh,bat}` for `gl factors ingest --env production` (or `GL_FACTORS_ENV=production`) invocations whose `--source <id>` resolves to a registry entry with `status in {pending_legal_review, blocked}` or `release_milestone > v0.1`. Runtime guard: `assert_source_safe_for_env` raises `SourceNotApprovedForEnvError`. Tested by `tests/ci/test_check_pending_legal_blocked.py` (5 cases) + `tests/ci/test_source_safety_helper.py` (15 cases). |
| 5 — no ingestion-runner code path bypasses the canonical writer | gate-5 ingestion-bypass-audit | `scripts/ci/check_ingestion_bypass.py` | Static-greps `greenlang/**/*.py` and `scripts/**/*.py` for `INSERT INTO factors_v0_1.factor` / `INSERT INTO alpha_factors_v0_1` / `executemany(...factor...)` against the canonical singular `factor` table. Whitelist: `greenlang/factors/repositories/alpha_v0_1_repository.py` is the only allowed writer. Returns 0 on the current tree (acceptance criterion). Tested by `tests/ci/test_check_ingestion_bypass.py` (7 cases). |
| 6 — workflow committed | factors-ingestion-check.yml | `.github/workflows/factors-ingestion-check.yml` | Six parallel jobs, one per gate. Triggers on PR + master push to paths `greenlang/factors/ingestion/**`, `greenlang/factors/data/source_registry.yaml`, `tests/factors/v0_1_alpha/phase3/**`, `greenlang/factors/data/catalog_seed/**`, `migrations/versions/**`, `deployment/database/migrations/sql/V5*.sql`, `.github/workflows/factors-ingestion-check.yml`. Plus `workflow_dispatch`. Each job uploads `phase3-gate-<n>-<sha>` artifact. YAML-parseable (`python -c "import yaml; yaml.safe_load(open('.github/workflows/factors-ingestion-check.yml'))"`). |

**Acceptance signal (Wave 3.0 ship)**:
- `python -m pytest tests/ci -q -W ignore::DeprecationWarning` -> 42 passed, 0 failed.
- `python -m pytest tests/factors/v0_1_alpha -q -W ignore::DeprecationWarning` -> 1913 passed, 17 skipped, 0 failed (unchanged from pre-Wave-3.0 baseline).
- `python scripts/ci/check_ingestion_bypass.py` exits 0 on `master@HEAD`.
- `.github/workflows/factors-ingestion-check.yml` parses as valid YAML.
