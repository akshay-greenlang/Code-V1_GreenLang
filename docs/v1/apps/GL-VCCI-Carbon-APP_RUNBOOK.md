# GL-VCCI-Carbon-APP v1 Runbook

- App ID: GL-VCCI-Carbon-APP
- Owner: VCCI Platform Team
- Canonical runtime command: `gl run vcci <input.csv|json> <output_dir>`
- Profile runtime command: `gl v1 run-profile vcci <input.csv|json> - <output_dir> false`
- Required artifacts:
  - `scope3_inventory.json`
  - `audit/run_manifest.json`
  - `audit/checksums.json`
  - `audit/observability_event.json`

## Strict Native Mode

- Set `GL_V1_ALLOW_BACKEND_FALLBACK=0` for release evidence.
- Validate `audit/run_manifest.json` contains `"execution_mode": "native"`.

## Evidence Checklist

- Capture UTC timestamp and commit SHA.
- Capture command output excerpt for `gl run vcci`.
- Record SHA-256 for `scope3_inventory.json` and audit artifacts.

## Web Workspace Access

- Multi-app portal: `http://127.0.0.1:8001/apps`
- VCCI workspace: `http://127.0.0.1:8001/apps/vcci`
- Run center: `http://127.0.0.1:8001/runs`

