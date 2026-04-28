# Phase 3 Plan — Raw Ingestion Framework

> **Authority**: CTO Phase 3 brief (2026-04-28).
> **Owner**: Factors Engineering.
> **Reviewers**: CTO, Methodology Lead, Backend Lead, Head of Data, Test Lead, Legal.
> **Predecessor**: Phase 2 (engineering complete; awaiting human sign-off as of 2026-04-28).

---

## Goal

Make GreenLang Factors ingestion a **governed, repeatable, atomic pipeline**: raw source artifact in → canonical factor records staged, reviewed, and published atomically. After Phase 3, no certified factor can land without proof of which raw artifact, parser version, source version, checksum, row reference, licence decision, validation result, diff, reviewer, and publish batch produced it.

---

## Reality check — what already exists

The 2026-04-28 reconnaissance found ~60% of Phase 3 already implemented. Phase 3 is therefore **composition + governance**, not building primitives from scratch.

| Capability | Status | Location |
|---|---|---|
| Parser ABC + plugin registry | Built | `greenlang/factors/ingestion/parsers/__init__.py` (`BaseSourceParser`, `ParserRegistry`) |
| 30 source parsers | Built | `greenlang/factors/ingestion/parsers/*.py` |
| HTTP + file fetchers | Built | `greenlang/factors/ingestion/fetchers.py` |
| Tabular helpers (CSV/XLSX) | Built | `greenlang/factors/ingestion/tabular_fetchers.py` |
| Artifact store with sha256 | Built (local-only) | `greenlang/factors/ingestion/artifacts.py` (`LocalArtifactStore`) |
| Canonical normalizer | Built | `greenlang/factors/ingestion/normalizer.py` |
| Dedupe fingerprint | Built | `greenlang/factors/dedupe_rules.py` |
| Stage-vs-production diff | Built | `greenlang/factors/release/alpha_publisher.py` (`StagingDiff`) |
| 7-gate publish orchestrator | Built (Phase 2) | `greenlang/factors/quality/publish_gates.py` |
| Source registry YAML | Built | `greenlang/factors/data/source_registry.yaml` + `source_registry.py` |
| Bulk ingest entry | Built | `greenlang/factors/ingestion/bulk_ingest.py` |
| argparse CLI subcommands | Built | `greenlang/factors/cli.py` |
| DB tables (raw_artifacts, ingest_runs, factor_lineage) | Built but **not Alembic-versioned** | `greenlang/factors/ingestion/sqlite_metadata.py` |

**Phase 3 gap (the ~40% to build):**

- Unified 7-stage `IngestionPipeline` runner (currently fragmented across `bulk_ingest.py`, `watch.py`, `alpha_publisher.py`).
- Click subcommand group `gl factors ingest {fetch,parse,run,diff,stage,publish,rollback}` with structured-JSON + human-readable summaries and run-status enum.
- Formal Alembic V507/V508 migrations to version the existing ad-hoc tables + add `ingestion_runs.status` enum + `ingestion_run_diffs` table.
- Snapshot test framework with golden-fixture comparison + `UPDATE_PARSER_SNAPSHOT=1` regeneration env flag.
- JSON/Markdown diff export from `StagingDiff` for methodology-lead review.
- EcoSpold2/MRIO parser scaffolding (ecoinvent, EXIOBASE).
- Webhook fetcher.
- CI gates: parser-snapshot drift, source_registry parser_version drift, ingestion-runner-bypass-publish-orchestrator audit.

---

## The seven-stage pipeline contract

Every ingestion run executes these stages in order. Each stage commits a row to `ingestion_runs` advancing the `status` enum. A failure short-circuits with the stage name + structured error. No stage may be skipped except via explicit `--from-stage` resume mode after a failed run.

| # | Stage | Input | Output | DB write | Status after |
|---|---|---|---|---|---|
| 1 | **Fetch** | source_urn, source_version | raw bytes + URI + sha256 + headers | `raw_artifacts` row | `fetched` |
| 2 | **Parse** | artifact_id, parser_id+version | source-native rows + row_refs | row_count_per_sheet | `parsed` |
| 3 | **Normalize** | parsed rows + ParserContext | factor_record_v0_1 dicts (no URN dedup yet) | none | `normalized` |
| 4 | **Validate** | normalized records | reject list + accept list | per-record gate result | `validated` |
| 5 | **Dedupe** | accepted records | (final, supersede-pairs, removal-candidates) | dedupe audit | `deduped` |
| 6 | **Stage + diff** | dedupe output | staging-namespace rows + JSON+MD diff artefact | `ingestion_run_diffs` | `staged` / `review_required` |
| 7 | **Publish** | run_id, approver | atomic batch flip; `factor_publish_log` row | gates re-run, factor INSERTs | `published` |

A `rollback` action demotes a `published` run back to `staged` without deletion (history preserved via `factor_publish_log` reverse entry).

---

## Run-status enum (formal)

`created → fetched → parsed → normalized → validated → deduped → staged → review_required → published`. Terminal failure states: `rejected`, `failed`, `rolled_back`. Stored as a Postgres ENUM and a Python `Enum`; `ingestion_runs.status` carries one of these values at all times.

---

## Artifact storage contract

Every certified factor MUST be traceable to an `extraction.raw_artifact_uri` + `extraction.raw_artifact_sha256` that resolves to a row in `raw_artifacts` (Phase 2 V505 source_artifacts table is renamed/aliased). Required columns:

- `raw_bytes_uri` (file:// or s3://)
- `source_url` (where it was fetched from)
- `fetched_at` (ISO8601 UTC)
- `sha256` (lowercase hex, 64 chars)
- `bytes_size` + `content_type`
- `source_version` + `source_publication_date`
- `parser_module`, `parser_function`, `parser_version`, `parser_commit`
- `operator` (`bot:source-watch` or `human:<email>`)
- `licence_class`, `redistribution_class` (resolved from registry at fetch time)
- `ingestion_run_id` (FK → `ingestion_runs.run_id`)
- `status` enum: `fetched | parsed | normalized | staged | rejected | published | superseded`

Rules:

1. Parser output is rejected if artifact storage write failed.
2. Manual / PDF / OCR sources still need an uploaded raw artifact + reviewer-approved extraction notes (`reviewer_notes` JSONB column).
3. Checksum is verified at every stage transition (cheap re-hash of the stored bytes).

---

## Fetcher / parser families

The `source_registry.yaml` is the single dispatch authority. Each entry pins:

```yaml
- source_id: defra-2025
  source_urn: urn:gl:source:defra-2025
  family: excel             # excel | csv_json_xml | pdf_ocr | api_webhook | ecospold_mrio
  fetch_mode: http          # http | file | api | webhook | s3 | manual
  fetch_url: https://...
  cadence: annual
  parser_module: greenlang.factors.ingestion.parsers.defra_2025
  parser_function: parse
  parser_version: 1.2.0
  parser_commit_pinned: false   # if true, CI gates require parser_commit==git_HEAD on PRs touching the parser
  licence_class: ogl-uk-3.0
  redistribution_class: attribution
  legal_signoff_artifact: docs/factors/legal/2026-Q1-defra.md
  owner: methodology-lead@greenlang.io
  status: alpha_v0_1        # alpha_v0_1 | pending_legal_review | blocked | future_milestone
  release_milestone: v0.1
```

Five families ship with Wave 1 framework, with **DEFRA Excel** as the canonical reference end-to-end. Other families ship parser stubs + family-specific validation rules but production parsers migrate in Wave 2.

| Family | Required validation on every record |
|---|---|
| Excel | workbook tab name + required columns + header row + unit string + vintage label match registry |
| CSV/JSON/XML | schema shape + required fields + numeric ranges + units + timestamps + geography codes |
| PDF/OCR/manual | PDF stored + extracted-table artifact + OCR confidence + reviewer notes + manual correction log |
| API/webhook | request URL + response body + response timestamp + API version + pagination cursor + webhook event ID |
| EcoSpold2/MRIO | zipped multi-file artifact + system-model metadata + activity/process IDs + geography map + licence entitlement |

---

## Dedupe / supersede / diff rules

| Scenario | Rule | Outcome |
|---|---|---|
| Same URN, identical normalized payload | unchanged | no DB write |
| Same semantic factor (fingerprint match), changed value/source-version | new URN version OR `supersedes_urn` per URN-version policy | new row + supersede pair |
| Same source row parsed twice in one run | dedupe before staging | first one wins; second logged as duplicate |
| Production record absent from new staging run | removal candidate | flagged; **never** auto-deleted |
| Cross-source supersedes | blocked | requires methodology-lead explicit approval flag |

The diff (Markdown + JSON) shows: value changes, unit changes, boundary changes, geography changes, methodology changes, licence changes, citation changes, parser-version changes, plus counters (added / removed / changed / unchanged / superseded). **Reviewers must be able to decide publish-yes/no from the diff alone, without reading raw rows.**

---

## CLI surface (Click group)

```text
gl factors ingest fetch     --source <id> --version <v>                  # stages 1
gl factors ingest parse     --artifact <id>                              # stage 2
gl factors ingest run       --source <id> --version <v> [--auto-stage]   # stages 1-6
gl factors ingest diff      --run-id <id> [--format md|json]             # stage 6 output replay
gl factors ingest stage     --run-id <id>                                # stage 6 explicit
gl factors ingest publish   --run-id <id> --approved-by human:<email>    # stage 7
gl factors ingest rollback  --batch-id <id> --approved-by human:<email>  # demote published → staged
gl factors ingest status    --run-id <id>                                # tail current state + last error
```

Every command emits structured JSON to stdout when `--json` is set, otherwise a human-readable summary. Exit codes mirror the run-status enum (0 = terminal-success, ≥1 = terminal-failure with status code).

---

## Wave breakdown

| Wave | Scope | Acceptance | Target |
|---|---|---|---|
| **1.0 Framework** | unified `IngestionPipeline` runner (composes existing parts), Click CLI group, V507 alembic, snapshot-test infra, JSON+MD diff export, e2e tests against a mock source | acceptance runner: 10/10 PASS; full v0.1 alpha suite green; new modules ≥85% cov | This session |
| **1.5 Reference source** | DEFRA Excel migrated to new pipeline; golden snapshot committed; CI gate active | DEFRA fetch+parse+stage+publish completes via `gl factors ingest run` | Next session |
| **2.0 Parser migration** | EPA, eGRID, CEA, BEE, IEA, EDGAR, ENTSO-E, Climate TRACE migrated; family-specific validation hooks | each migrated parser has snapshot + family-validation tests | Wave 2 |
| **2.5 EcoSpold2/MRIO + webhook** | ecoinvent + EXIOBASE parsers; webhook receiver for PACT/private packs | full family coverage; webhook e2e test | Wave 3 |
| **3.0 CI gates + governance** | parser-snapshot CI gate; source-registry version-bump CI gate; ingestion-bypass audit; release-note gate | CI red on snapshot drift / unversioned parser change / direct factor INSERT | Wave 3 |

---

## KPI dashboard

| KPI | Target | Verified at |
|---|---|---|
| Factors with full extraction provenance | 100% | publish gate 6 |
| Sources with pinned parser_module + parser_version + parser_commit | 100% | source_registry CI gate |
| Snapshot coverage on production parsers | 100% (Wave 2.5+) | snapshot CI gate |
| Mean ingestion run wall-time (small Excel source) | < 60s | ingestion_runs telemetry |
| Mean diff-review-to-publish latency | < 24h | factor_publish_log |
| Rollback success on a synthetic bad batch | 100% | e2e test |
| Ingestion-bypass paths (direct factor INSERT) | 0 | grep + repository contract test |

---

## Out of scope for Phase 3

- **Physical table partitioning** — remains pre-v1.0 scale work (ADR-004 deferral).
- **Multi-region replicated artifact store** — Wave 4 / v1.0 GA.
- **Automated cross-source consensus (e.g. blending eGRID + CEA)** — Phase 5 (Resolution Engine).
- **Pricing surface** — Phase 4.

---

## Predecessors required before Phase 3 closes

- Phase 2 formal sign-off (CTO + Methodology Lead + Backend Lead + Head of Data + Test Lead + Legal).
- Phase 2 documentation lint green on master (closed 2026-04-28 via commit 7126a12f).

When Phase 3 closes, Phase 4 (Resolution Engine + Pricing) opens.
