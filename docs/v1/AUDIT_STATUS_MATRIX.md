# GreenLang v1 Audit Status Matrix

This matrix maps the 8 v1 to-dos to implementation evidence and readiness status.

Legend:
- Done: implemented with executable checks
- Partial: implemented but requires operational completion
- Not done: no implementation evidence

## 1) Create and approve v1 charter

- Status: **Done**
- Evidence:
  - `docs/v1/PHASE0_CHARTER.md`
  - `docs/v1/MILESTONE_CALENDAR.md`
  - `docs/v1/DEPENDENCY_GRAPH.md`

## 2) Standardize and version pack.yaml/gl.yaml contracts

- Status: **Done**
- Evidence:
  - Contract models/validators: `greenlang/v1/contracts.py`
  - App fixtures:
    - `applications/GL-CBAM-APP/v1/pack.yaml`, `gl.yaml`
    - `applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1/pack.yaml`, `gl.yaml`
    - `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1/pack.yaml`, `gl.yaml`
  - Tests: `tests/v1/test_contracts.py`
  - Docs: `docs/v1/CONTRACTS.md`, `docs/v1/MIGRATION_GUIDE.md`

## 3) Unify CLI command semantics and runtime behavior

- Status: **Done**
- Evidence:
  - `gl v1` command surface: `greenlang/cli/cmd_v1.py`
  - Profile command baselines: `greenlang/v1/profiles.py`
  - Runtime conformance checks: `greenlang/v1/conformance.py`
  - CLI tests: `tests/v1/test_cli_v1.py`

## 4) Shared auth/policy baseline + signed-pack verification

- Status: **Done**
- Evidence:
  - Signed-pack policy artifact: `greenlang/governance/policy/bundles/v1_signed_pack.rego`
  - Gate-level cryptographic verification: `greenlang/v1/conformance.py` using `verify_pack_signature`
  - Security docs: `docs/v1/SECURITY_POLICY_BASELINE.md`

## 5) Determinism, auditability, observability baseline

- Status: **Done**
- Evidence:
  - Standards helpers: `greenlang/v1/standards.py`
  - Backend adapters: `greenlang/v1/backends.py`
  - Determinism + artifact + observability checks in gate: `greenlang/v1/conformance.py` (`profile_full_backend_checks`)
  - Standards doc: `docs/v1/STANDARDS.md`

## 6) Multi-app smoke/integration/regression CI lanes

- Status: **Done**
- Evidence:
  - Workflow: `.github/workflows/greenlang-v1-platform-ci.yml`
  - Shared gate lane: `pytest -q tests/v1`, `gl v1 gate`
  - Profile matrix lane: `cbam`, `csrd`, `vcci` with profile-specific execution lanes and artifact assertions

## 7) Standardize docs/runbooks and bootstrap readiness

- Status: **Done**
- Evidence:
  - Docs contract: `docs/v1/DOCS_CONTRACT.md`
  - Quickstart + runbook templates:
    - `docs/v1/QUICKSTART.md`
    - `docs/v1/RUNBOOK_TEMPLATE.md`
  - App runbooks:
    - `docs/v1/apps/GL-CBAM-APP_RUNBOOK.md`
    - `docs/v1/apps/GL-CSRD-APP_RUNBOOK.md`
    - `docs/v1/apps/GL-VCCI-Carbon-APP_RUNBOOK.md`
  - UAT tracker with completed runs: `docs/v1/UAT_RESULTS.md`

## 8) Release hardening, freeze conventions, go/no-go

- Status: **Done**
- Evidence:
  - Baseline freeze file: `greenlang/v1/runtime_baseline.yaml`
  - Release checklist: `docs/v1/RELEASE_CHECKLIST.md`
  - RC process: `docs/v1/RELEASE_CANDIDATE_PROCESS.md`
  - RC soak evidence: `docs/v1/RC_SOAK_LOG.md`
  - Go/no-go decision record: `docs/v1/GO_NO_GO_RECORD.md`
  - Release notes + deferred roadmap:
    - `docs/v1/RELEASE_NOTES_v1.0.md`
    - `docs/v1/ROADMAP_v1_1.md`
  - Local audit script: `scripts/v1_release_audit.py`

## Summary

- Done: 8 / 8
- Partial: 0 / 8
- Not done: 0 / 8

