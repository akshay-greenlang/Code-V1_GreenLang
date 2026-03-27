# GreenLang v1 RC Soak Log

## Candidate

- Candidate version: v1.0-rc2
- Soak owner: Release Engineering
- Start time: 2026-03-27T11:21:30Z
- End time: 2026-03-27T11:41:28Z
- Commit SHA: `0c12d8a57e30597739d4e18e5adc810dfbcc1f21`
- Evidence directory: `phase1_evidence/current_head/`
- CI run reference:
  - Pre-push local strict validation (same commit SHA):
    - `python -m greenlang.cli.main v1 full-backend-checks` -> pass
    - `python -m greenlang.cli.main v1 gate` -> pass
    - `pytest -q tests/v1/test_full_backends.py tests/v1/test_cli_v1.py` -> pass
    - `pytest -q cbam-pack-mvp/tests/test_web_security.py cbam-pack-mvp/tests/test_web_v1_csrd_vcci_endpoints.py cbam-pack-mvp/tests/test_web_v1_contract.py` -> pass
  - Post-push workflow URL (must be attached before external promotion): `PENDING_POST_PUSH`

## Required Observations

- `gl v1 gate` pass/fail trend
- `tests/v1` pass/fail trend
- Determinism full-backend parity result
- Signed-pack verification pass/fail

## Soak Entries

| Timestamp | Check | Result | Notes |
|-----------|-------|--------|-------|
| 2026-03-27T11:21:30Z | `gl v1 validate-contracts` | Pass | Current-HEAD contract checks archived to `phase1_evidence/current_head/v1_validate_contracts.log`. |
| 2026-03-27T11:22:24Z | `gl v1 check-policy` | Pass | Signed-pack policy checks archived to `phase1_evidence/current_head/v1_check_policy.log`. |
| 2026-03-27T11:28:14Z | `gl v1 full-backend-checks` | Pass | Strict native CBAM/CSRD/VCCI parity checks archived to `phase1_evidence/current_head/v1_full_backend_checks.log`. |
| 2026-03-27T11:31:06Z | `gl v1 gate` | Pass | Release gates passed and archived to `phase1_evidence/current_head/v1_gate.log`. |
| 2026-03-27T11:36:47Z | Backend semantic tests | Pass | `tests/v1/test_full_backends.py` + `tests/v1/test_cli_v1.py` archived to `phase1_evidence/current_head/tests_v1_backend_semantics.log`. |
| 2026-03-27T11:41:28Z | Web security/contract tests | Pass | API-key and sanitization parity tests archived to `phase1_evidence/current_head/tests_web_security_contract.log`. |

## Command Output Excerpt

`gl v1 gate` summary:

- `[OK] applications/GL-CBAM-APP/v1 full backend`
- `[OK] applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1 full backend`
- `[OK] applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1 full backend`
- `v1 release gates passed`

