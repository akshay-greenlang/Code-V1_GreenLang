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

- [ ] CI blocks PRs that change a parser module without a corresponding snapshot regeneration commit.
- [ ] CI blocks PRs that change `source_registry.yaml` parser_version without a release-note entry.
- [ ] CI blocks PRs that publish a certified factor lacking raw artifact metadata (gate 6 audit).
- [ ] CI blocks ingestion of a source whose registry status is `pending_legal_review`, `blocked`, or a future `release_milestone` if the run targets `production`.
- [ ] CI audit confirms zero ingestion-runner code paths can write to `factor` table bypassing the Phase 2 7-gate orchestrator.
- [ ] `.github/workflows/factors-ingestion-check.yml` is committed to git and visible to GitHub Actions.
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
