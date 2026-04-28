# Phase 3 Exit Checklist — Raw Ingestion Framework

> **Authority**: CTO Phase 3 brief (2026-04-28).
> **Owner**: Factors Engineering.
> **Reviewers**: CTO, Methodology Lead, Backend Lead, Head of Data, Test Lead, Legal.
> **Predecessor**: Phase 2 sign-off must land before Phase 3 is declared formally complete.

Every technical box must be ticked and every accountable role must sign off before Phase 3 is declared complete and Phase 4 (Resolution Engine + Pricing) may start.

---

## Block 1 — Seven-stage pipeline contract (CTO §3.1)

- [ ] `IngestionPipeline` runner enforces stages in order: fetch → parse → normalize → validate → dedupe → stage+diff → publish.
- [ ] Each stage advances `ingestion_runs.status`; failure short-circuits with stage name + structured error.
- [ ] No stage may be skipped; resume mode requires explicit `--from-stage <name>` after a failed run.
- [ ] Stage transitions are atomic (one DB transaction per stage advance).
- [ ] Phase 2 publish gates (7-gate orchestrator) run unchanged inside stage 7.
- [ ] **Sign-off**: Backend Lead + CTO.

## Block 2 — Artifact storage contract (CTO §3.2)

- [ ] Every ingestion run writes a `raw_artifacts` row with: raw_bytes_uri, source_url, fetched_at, sha256, bytes_size, content_type, source_version, source_publication_date, parser_module, parser_function, parser_version, parser_commit, operator, licence_class, redistribution_class, ingestion_run_id, status.
- [ ] Certified factors carry `extraction.raw_artifact_uri` AND `extraction.raw_artifact_sha256`; sha256 verified at every stage transition.
- [ ] Parser output is rejected when artifact storage write fails (negative test passes).
- [ ] Manual / PDF / OCR sources require uploaded raw artifact + reviewer-approved `reviewer_notes` JSONB.
- [ ] **Sign-off**: Head of Data + Backend Lead.

## Block 3 — Fetcher / parser families (CTO §3.3)

- [ ] Five families dispatch from `source_registry.yaml`: excel, csv_json_xml, pdf_ocr, api_webhook, ecospold_mrio.
- [ ] Excel-family validates workbook tabs + required columns + header row + unit string + vintage label.
- [ ] CSV/JSON/XML-family validates schema shape + required fields + numeric ranges + units + timestamps + geography codes.
- [ ] PDF/OCR-family stores PDF + extracted-table artifact + OCR confidence + reviewer notes + manual correction log.
- [ ] API/webhook-family captures request URL + response body + response timestamp + API version + pagination cursor + webhook event ID.
- [ ] EcoSpold2/MRIO-family handles zipped multi-file artifacts + system-model metadata + activity/process IDs + geography map + licence entitlement.
- [ ] DEFRA Excel reference source migrated end-to-end (Wave 1.5).
- [ ] **Sign-off**: Methodology Lead + Backend Lead.

## Block 4 — Dedupe / supersede / diff rules (CTO §3.4)

- [ ] Same URN + identical payload → unchanged, no DB write (positive test).
- [ ] Same fingerprint + changed value → new URN version OR `supersedes_urn` per URN policy.
- [ ] Duplicate source rows in one run → dedupe before staging (first wins, second logged).
- [ ] Missing-from-staging production record → flagged removal candidate, never auto-deleted.
- [ ] Cross-source supersede blocked unless methodology-lead approval flag set.
- [ ] Diff (MD + JSON) shows: value / unit / boundary / geography / methodology / licence / citation / parser-version changes + counters.
- [ ] Reviewers can decide publish-yes/no from diff alone (acceptance test: synthetic diff matches expected MD/JSON).
- [ ] **Sign-off**: Methodology Lead.

## Block 5 — Operational interfaces (CTO §3.5)

- [ ] Click group `gl factors ingest` exposes: fetch / parse / run / diff / stage / publish / rollback / status.
- [ ] Run-status enum complete: created, fetched, parsed, normalized, validated, deduped, staged, review_required, published, rejected, failed, rolled_back.
- [ ] Each command emits structured JSON when `--json` is set, otherwise human-readable summary.
- [ ] Exit codes mirror the run-status enum.
- [ ] `gl factors ingest publish --approved-by human:<email>` requires approver identity (rejects bot operators).
- [ ] **Sign-off**: Backend Lead + GL-SpecGuardian.

## Block 6 — Snapshot tests + pipeline tests (CTO Test Plan)

- [ ] Each migrated parser has committed golden artifact + expected normalized output under `tests/factors/v0_1_alpha/phase3/parser_snapshots/`.
- [ ] Snapshot tests fail on: changed workbook tab names, missing required columns, unit drift, source table-shape drift, impossible values, missing citations, missing licence tags, missing raw artifact checksum, missing row/sheet/table reference, parser output drift without `UPDATE_PARSER_SNAPSHOT=1` regeneration.
- [ ] Pipeline e2e tests cover: fetch→stage happy path; stage→publish; duplicate dedupe; changed-row supersede; missing-artifact-blocks-publish; licence-mismatch-blocks-ingestion; invalid-ontology-blocks-staging; parser-failure-records-failed-run; rollback-demotes-not-deletes.
- [ ] Source-type fixture tests: excel + csv + json + pdf + api + zipped multi-file MRIO.
- [ ] Coverage ≥ 85% on new modules (`pipeline.py`, `runner.py`, `cli/ingest.py`, snapshot harness).
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

## Engineering evidence — per-box (Wave 3.0)

> Each Block 7 box maps to a single line of evidence below. Cite during sign-off; if a reviewer disputes a box, point at the file/test cited here. Mirrors the per-block evidence pattern at the bottom of `docs/factors/PHASE_2_EXIT_CHECKLIST.md`.

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
