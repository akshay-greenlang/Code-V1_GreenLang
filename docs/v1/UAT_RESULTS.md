# GreenLang v1 UAT Results

## Objective

Demonstrate that non-core teams can stand up each v1 app profile using standardized docs and commands.

## Immutable Evidence Header

- Evidence run timestamp (UTC): `2026-03-27T11:41:28Z`
- Commit SHA: `0c12d8a57e30597739d4e18e5adc810dfbcc1f21`
- Evidence workspace: `phase1_evidence/current_head/`
- Strict mode: `GL_V1_ALLOW_BACKEND_FALLBACK=0`

## Test Protocol

Required commands:

```bash
gl v1 status
gl v1 validate-contracts
gl v1 check-policy
gl v1 full-backend-checks
gl v1 gate
```

Per-app strict runtime commands:

```bash
gl run cbam cbam-pack-mvp/examples/sample_config.yaml cbam-pack-mvp/examples/sample_imports.csv phase1_evidence/cbam
gl run csrd applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv phase1_evidence/csrd
gl run vcci applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv phase1_evidence/vcci
```

## Team Runs

| Team | App Profile | Date | Result | Notes |
|------|-------------|------|--------|-------|
| Enablement-Team-A | GL-CBAM-APP | 2026-03-26 | Pass | Strict native `gl run cbam` completed; required report and audit artifacts produced. |
| Enablement-Team-B | GL-CSRD-APP | 2026-03-26 | Pass | Strict native `gl run csrd` completed; `esrs_report.json` + audit artifacts present. |
| Enablement-Team-C | GL-VCCI-Carbon-APP | 2026-03-26 | Pass | Strict native `gl run vcci` completed; `scope3_inventory.json` + audit artifacts present. |

## Artifact Hash Evidence (SHA-256)

| Artifact | SHA-256 |
| --- | --- |
| `phase1_evidence/cbam/cbam_report.xml` | `d62568bc26dc3384ce7741849b133ce493eb5001ab0e9d814b545844e1a08db8` |
| `phase1_evidence/cbam/report_summary.xlsx` | `b446802db4c788d7d1eacdd171a39616a984d8bcf187e4d1592ac0fe2e71ac2e` |
| `phase1_evidence/csrd/esrs_report.json` | `a1e59b92c7dd6ca330ac0a1bfe8079c3ed566c92e32ac5690636fa3cd84d52ee` |
| `phase1_evidence/vcci/scope3_inventory.json` | `bfbbc207677069f77b72cd510efba0de1eabc8b92900d9be3d82b232de9b1f6b` |

## Exit Criterion

All three teams completed the protocol successfully without private tribal knowledge.

## Current-HEAD Evidence Files

- `phase1_evidence/current_head/metadata.txt`
- `phase1_evidence/current_head/exit_codes.txt`
- `phase1_evidence/current_head/v1_validate_contracts.log`
- `phase1_evidence/current_head/v1_check_policy.log`
- `phase1_evidence/current_head/v1_full_backend_checks.log`
- `phase1_evidence/current_head/v1_gate.log`
- `phase1_evidence/current_head/tests_v1_backend_semantics.log`
- `phase1_evidence/current_head/tests_web_security_contract.log`
- `phase1_evidence/current_head/extra_test_exit_codes.txt`

## Web UX Validation Addendum

- Multi-app shell routes validated: `/apps`, `/apps/cbam`, `/apps/csrd`, `/apps/vcci`, `/runs`
- Workspace command palette validated (`Ctrl/Cmd+K`) with app-switch actions
- Demo-mode API endpoints validated:
  - `POST /api/v1/apps/cbam/demo-run`
  - `POST /api/v1/apps/csrd/demo-run`
  - `POST /api/v1/apps/vcci/demo-run`

