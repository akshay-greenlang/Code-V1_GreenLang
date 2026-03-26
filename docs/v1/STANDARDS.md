# GreenLang v1 Runtime Standards

## Determinism Standard

- Same input and config must produce byte-identical artifact hashes.
- Determinism verification compares two output directories:
  - file set parity
  - SHA-256 parity per matching file
- Determinism checks are required for flagship workflows in each v1 app profile.

## Auditability Standard

Required artifacts for v1 workflow outputs:

- `audit/run_manifest.json`
- `audit/checksums.json`

These artifacts must include run identifiers and integrity metadata sufficient for replay and audit.

## Observability Baseline

Every v1 workflow event should include:

- `app_id`
- `pipeline_id`
- `run_id`
- `status`
- `duration_ms`

Additional fields can be included under `extra`.

## v1 SLO Baseline

- Contract validation workflow success rate: 99%+
- v1 release gate workflow success rate: 99%+
- Determinism parity for designated test workflows: 100%
- Signed-pack policy checks for supported-tier app profiles: 100%
- Full backend lane (`gl v1 full-backend-checks`) pass rate: 100%

## Release Evidence Standard (Mandatory)

Every go/no-go packet must include immutable evidence:

- Commit SHA used for validation
- UTC timestamps for strict native runs
- `gl v1 full-backend-checks` output excerpt
- `gl v1 gate` output excerpt
- SHA-256 hashes for required app artifacts
- CI run URL(s) for blocking workflows

Narrative-only pass/fail statements are insufficient for release approval.

