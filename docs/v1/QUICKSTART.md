# GreenLang v1 Quickstart

## Goal

Bootstrap and validate v1 profiles for CBAM, CSRD, and VCCI using one shared command surface.

## Commands

```bash
# 0) Optional strict-mode lock (recommended for release evidence)
set GL_V1_ALLOW_BACKEND_FALLBACK=0

# 1) List v1 app profiles
gl v1 status

# 2) Validate v1 contracts (pack.yaml + gl.yaml)
gl v1 validate-contracts

# 3) Enforce signed-pack baseline checks
gl v1 check-policy

# 4) Run full release gate bundle
gl v1 gate

# 5) Run full backend lane checks (CSRD/VCCI runtime adapters)
gl v1 full-backend-checks

# 6) Run strict native app commands (release evidence)
gl run cbam cbam-pack-mvp/examples/sample_config.yaml cbam-pack-mvp/examples/sample_imports.csv phase1_evidence/cbam
gl run csrd applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv phase1_evidence/csrd
gl run vcci applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv phase1_evidence/vcci
```

## Multi-App Web Workspace (Local)

```bash
# Start web server (choose a free port)
gl-cbam web --host 127.0.0.1 --port 8001
```

Open:

- `http://127.0.0.1:8001/apps` (multi-app home)
- `http://127.0.0.1:8001/apps/cbam` (CBAM workspace)
- `http://127.0.0.1:8001/apps/csrd` (CSRD workspace)
- `http://127.0.0.1:8001/apps/vcci` (VCCI workspace)
- `http://127.0.0.1:8001/runs` (run center)

### Frontend UX extras (new)

- Press `Ctrl+K` / `Cmd+K` in web workspaces to open the command palette.
- Use **Try Demo Data** in each workspace for zero-setup sample processing:
  - `POST /api/v1/apps/cbam/demo-run`
  - `POST /api/v1/apps/csrd/demo-run`
  - `POST /api/v1/apps/vcci/demo-run`
- Client-side crash telemetry is captured at `POST /api/telemetry/client-error` for triage.

### Docker baseline (local-to-prod path)

```bash
docker compose -f deployment/docker-compose.v1-web.yml up --build
```

## Expected Outcome

- All three app profile contracts validate successfully.
- Signed-pack checks pass for each profile.
- Full gate command exits with code `0`.
- Full backend checks command exits with code `0`.
- Native app runs produce contract artifacts under `phase1_evidence/`.

## Immutable Evidence Requirements

Before release sign-off, capture:

- UTC timestamp of run
- Commit SHA
- `gl v1 gate` output excerpt
- SHA-256 for each required app artifact
- CI run URL(s) for blocking workflows

## Latest Local Evidence Snapshot

- UTC timestamp: `2026-03-26T16:45:17Z`
- Commit SHA: `4e5ef5a72e4b61088086fa5aeaa3834b24e4ed4e`
- Artifact hashes:
  - `phase1_evidence/cbam/cbam_report.xml` -> `d62568bc26dc3384ce7741849b133ce493eb5001ab0e9d814b545844e1a08db8`
  - `phase1_evidence/cbam/report_summary.xlsx` -> `b446802db4c788d7d1eacdd171a39616a984d8bcf187e4d1592ac0fe2e71ac2e`
  - `phase1_evidence/csrd/esrs_report.json` -> `a1e59b92c7dd6ca330ac0a1bfe8079c3ed566c92e32ac5690636fa3cd84d52ee`
  - `phase1_evidence/vcci/scope3_inventory.json` -> `bfbbc207677069f77b72cd510efba0de1eabc8b92900d9be3d82b232de9b1f6b`

## App Profile Paths

- `applications/GL-CBAM-APP/v1/`
- `applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1/`
- `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1/`

