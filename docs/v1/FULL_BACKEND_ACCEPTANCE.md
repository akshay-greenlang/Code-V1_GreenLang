# GreenLang v1 Full Backend Acceptance

This document defines acceptance criteria for "fully feature-complete production backend"
across the first app set (CBAM, CSRD, VCCI).

## Common Criteria (All Apps)

- Runtime path is executable from canonical CLI (`gl run <profile-or-ref> ...`).
- Required artifact contract is produced on successful run.
- `audit/run_manifest.json` and `audit/checksums.json` are always present.
- Signed-pack verification passes in `gl v1 check-policy`.
- Determinism parity: two identical runs produce same fileset and hash parity.
- Observability event contains required baseline fields.

## CBAM Acceptance

- Command: `gl run cbam <config.yaml> <imports.csv> <output_dir>`
- Required artifacts:
  - `cbam_report.xml`
  - `report_summary.xlsx`
  - `audit/run_manifest.json`
  - `audit/checksums.json`
- Policy/schema outcomes are surfaced and export gating is enforced.

## CSRD Acceptance

- Command: `gl run csrd <input.csv|json> <output_dir>`
- Required artifacts:
  - `esrs_report.json`
  - `audit/run_manifest.json`
  - `audit/checksums.json`
- Runtime and release gates enforce artifact presence and determinism parity.

## VCCI Acceptance

- Command: `gl run vcci <input.csv|json> <output_dir>`
- Required artifacts:
  - `scope3_inventory.json`
  - `audit/run_manifest.json`
  - `audit/checksums.json`
- Runtime and release gates enforce artifact presence and determinism parity.

## Release Gate Mapping

- `gl v1 gate`: contracts + policy + runtime conventions + full backend checks + docs contract.
- `gl v1 full-backend-checks`: executes backend lane checks directly.
