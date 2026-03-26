# GreenLang v1 RC Soak Log

## Candidate

- Candidate version: v1.0-rc1
- Soak owner: Release Engineering
- Start time: 2026-03-26T07:28:10Z
- End time: 2026-03-26T07:40:11Z
- Commit SHA: `4e5ef5a72e4b61088086fa5aeaa3834b24e4ed4e`
- Evidence directory: `phase1_evidence/`
- CI run reference:
  - Pre-push local strict validation (same commit SHA):
    - `python -m greenlang.cli.main v1 full-backend-checks` -> pass
    - `python -m greenlang.cli.main v1 gate` -> pass
    - `pytest -q tests/v1` -> pass
  - Post-push workflow URL (must be attached before external promotion): `PENDING_POST_PUSH`

## Required Observations

- `gl v1 gate` pass/fail trend
- `tests/v1` pass/fail trend
- Determinism full-backend parity result
- Signed-pack verification pass/fail

## Soak Entries

| Timestamp | Check | Result | Notes |
|-----------|-------|--------|-------|
| 2026-03-26T07:28:10Z | `gl run cbam ... phase1_evidence/cbam` | Pass | Strict native runtime completed and generated report + audit bundle. |
| 2026-03-26T07:29:01Z | `gl run csrd ... phase1_evidence/csrd` | Pass | Strict native runtime completed and generated `esrs_report.json` + audit bundle. |
| 2026-03-26T07:29:51Z | `gl run vcci ... phase1_evidence/vcci` | Pass | Strict native runtime completed and generated `scope3_inventory.json` + audit bundle. |
| 2026-03-26T07:33:10Z | `gl v1 full-backend-checks` | Pass | CBAM/CSRD/VCCI full backend checks passed in one lane. |
| 2026-03-26T07:40:11Z | `gl v1 gate` | Pass | Contract, signed-pack, runtime conventions, full-backend checks, docs contract all green. |
| 2026-03-26T12:00:00Z | Web shell smoke (`/apps`, `/runs`) | Pass | Unified shell/workspaces render with command palette + onboarding modal + run timeline. |
| 2026-03-26T12:02:00Z | Demo mode endpoints | Pass | `cbam/csrd/vcci` demo-run endpoints return run IDs and appear in run center. |
| 2026-03-26T12:04:00Z | Frontend telemetry | Pass | `POST /api/telemetry/client-error` accepted browser error payloads. |
| 2026-03-26T16:45:17Z | Strict native app reruns + hash refresh | Pass | CBAM/CSRD/VCCI native runs regenerated `phase1_evidence` artifacts and hashes were refreshed in quickstart/UAT/go-no-go docs. |

## Command Output Excerpt

`gl v1 gate` summary:

- `[OK] applications/GL-CBAM-APP/v1 full backend`
- `[OK] applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1 full backend`
- `[OK] applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1 full backend`
- `v1 release gates passed`

