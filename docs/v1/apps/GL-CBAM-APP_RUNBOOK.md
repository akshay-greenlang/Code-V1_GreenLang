# GL-CBAM-APP v1 Runbook

- App ID: GL-CBAM-APP
- Owner: CBAM Platform Team
- Canonical runtime command: `gl run cbam <config.yaml> <imports.csv> <output_dir>`
- Profile runtime command: `gl v1 run-profile cbam <config.yaml> <imports.csv> <output_dir> false`
- Required artifacts:
  - `cbam_report.xml`
  - `report_summary.xlsx`
  - `audit/run_manifest.json`
  - `audit/checksums.json`
  - `audit/observability_event.json`

## Web Workspace (Multi-App)

- Entry URL: `http://127.0.0.1:8001/apps`
- CBAM workspace URL: `http://127.0.0.1:8001/apps/cbam`
- Legacy CBAM-only URL (backward compatible): `http://127.0.0.1:8001/`

## Strict Native Mode

- Set `GL_V1_ALLOW_BACKEND_FALLBACK=0` for release evidence.
- Validate `audit/run_manifest.json` contains `"execution_mode": "native"`.

## Evidence Checklist

- Capture UTC timestamp and commit SHA.
- Capture command output excerpt for `gl run cbam`.
- Record SHA-256 for `cbam_report.xml`, `report_summary.xlsx`, and audit artifacts.

