# GreenLang v1 Multi-App Frontend UAT Checklist

## Local Startup

1. Run: `gl-cbam web --host 127.0.0.1 --port 8001`
2. Verify:
   - `http://127.0.0.1:8001/apps`
   - `http://127.0.0.1:8001/apps/cbam`
   - `http://127.0.0.1:8001/apps/csrd`
   - `http://127.0.0.1:8001/apps/vcci`
   - `http://127.0.0.1:8001/runs`

## CBAM Workspace

1. Upload valid CBAM config and imports.
2. Generate report.
3. Confirm compliance summary and artifact list.
4. Download an individual artifact.
5. Download ZIP bundle (if export is allowed).

## CSRD Workspace

1. Upload valid CSRD input file.
2. Run CSRD.
3. Confirm normalized response includes:
   - `run_id`
   - `execution_mode`
   - `artifacts`
4. Download one artifact and ZIP bundle.

## VCCI Workspace

1. Upload valid VCCI input file.
2. Run VCCI.
3. Confirm normalized response includes:
   - `run_id`
   - `execution_mode`
   - `artifacts`
4. Download one artifact and ZIP bundle.

## Security Checks

1. Invalid v1 run id to bundle endpoint returns `400`.
2. Invalid v1 run id to artifact endpoint returns `400`.
3. Export-blocked runs return `409` for v1 artifact and bundle downloads.
4. CBAM legacy security controls still pass:
   - path traversal blocked
   - oversized upload blocked
   - API key enforced when configured

## Evidence Capture

- Store screenshots for each workspace home.
- Save response payloads for one successful run per app.
- Save generated `audit/run_manifest.json` and `audit/checksums.json`.
- Record UTC timestamp and commit SHA with command log.
