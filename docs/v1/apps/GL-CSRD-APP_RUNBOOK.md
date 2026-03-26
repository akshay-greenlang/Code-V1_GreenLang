# GL-CSRD-APP v1 Runbook

- App ID: GL-CSRD-APP
- Owner: CSRD Platform Team
- Canonical runtime command: `gl run csrd <input.csv|json> <output_dir>`
- Profile runtime command: `gl v1 run-profile csrd <input.csv|json> - <output_dir> false`
- Required artifacts:
  - `esrs_report.json`
  - `audit/run_manifest.json`
  - `audit/checksums.json`
  - `audit/observability_event.json`

## Strict Native Mode

- Set `GL_V1_ALLOW_BACKEND_FALLBACK=0` for release evidence.
- Validate `audit/run_manifest.json` contains `"execution_mode": "native"`.

## Evidence Checklist

- Capture UTC timestamp and commit SHA.
- Capture command output excerpt for `gl run csrd`.
- Record SHA-256 for `esrs_report.json` and audit artifacts.

## Web Workspace Access

- Multi-app portal: `http://127.0.0.1:8001/apps`
- CSRD workspace: `http://127.0.0.1:8001/apps/csrd`
- Run center: `http://127.0.0.1:8001/runs`

