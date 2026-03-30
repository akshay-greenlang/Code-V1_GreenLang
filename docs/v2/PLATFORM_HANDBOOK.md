# GreenLang V2 Platform Handbook

## Purpose

Single operational and engineering handbook for V2 scale-and-productization delivery.

## Core Sections

1. Runtime and CLI contract
2. App profile onboarding
3. Pack tier lifecycle and signing
4. Agent lifecycle governance
5. Connector reliability operations
6. Security and policy governance
7. Determinism and audit contract
8. Release train and promotion policy
9. Enterprise UX governance
10. Go/no-go evidence model

## Golden Paths

### New App Onboarding

1. add `applications/<app>/v2/pack.yaml` and `gl.yaml`
2. register app profile in `greenlang/v2/profiles.py`
3. validate with `python -m greenlang.cli.main v2 validate-contracts`
4. pass `python -m greenlang.cli.main v2 gate` and V2 CI blockers

### Pack Promotion

1. pass contract tests
2. pass security scans
3. pass determinism checks
4. complete docs/runbook
5. apply tier promotion in governance review

## Migration Playbooks

- canonical migration procedures are in `docs/v2/MIGRATION_PLAYBOOKS.md`.
- all migrations must conclude with immutable evidence hash updates in `docs/v2/IMMUTABLE_EVIDENCE_MANIFEST.json`.

## Release Operations Runbook

### Standard gate sequence

1. `python -m greenlang.cli.main v2 validate-contracts`
2. `python -m greenlang.cli.main v2 runtime-checks`
3. `python -m greenlang.cli.main v2 docs-check`
4. `python -m greenlang.cli.main v2 gate`

### Release train promotion policy

- PR train: smoke checks and quick regressions.
- Nightly train: expanded matrix plus governance scans.
- RC train: full matrix + soak + go/no-go package.
- Stable promotion: only from a green RC with signed board decision.

## Incident and Escalation Runbook

### Severity model

- Severity 1: regulated workflow outage, policy bypass, or determinism failure.
- Severity 2: degraded release gates or connector reliability issues.
- Severity 3: non-blocking defects and documentation/process drift.

### Escalation flow

1. page on-call owner from `docs/v2/ONCALL_AND_SLOS.md`.
2. open incident record and assign incident commander.
3. apply containment (roll back release lane or freeze promotions).
4. publish status updates every 30 minutes for Severity 1.
5. close with post-incident action items and evidence link updates.

## Rollback and Recovery Criteria

- rollback required if any release blocker fails after merge (security, determinism, policy, or immutable evidence).
- rollback scope must include app profile, pack tier promotion, and workflow lane if impacted.
- after rollback, rerun full gate sequence before re-promotion.

## Evidence Ownership and Audit Chain

- release board owns `docs/v2/GO_NO_GO_RECORD.md`.
- platform governance owns `docs/v2/IMMUTABLE_EVIDENCE_MANIFEST.json`.
- phase owners own `docs/v2/PHASE*_GATE_STATUS.json` evidence updates.
- evidence updates are valid only when hashes and gate outputs match.

## Migration Acceptance Checklist

- migration playbook executed with no skipped mandatory steps.
- app/pack/agent/connector policy checks pass.
- release-train cycle evidence updated in `docs/v2/RELEASE_TRAIN_CYCLE_LOG.md`.
- UAT and soak evidence updated in:
  - `docs/v2/UAT_RESULTS.md`
  - `docs/v2/RC_SOAK_LOG.md`
- immutable manifest refreshed and verified by `python -m greenlang.cli.main v2 gate`.

## Regulated App Parity Checklist

- EUDR/GHG/ISO14064 each have:
  - native v2 runtime entrypoint (`applications/<APP>/v2/runtime_backend.py`)
  - smoke input fixture (`applications/<APP>/v2/smoke_input.json`)
  - web run endpoint (`/api/v1/apps/<app>/run`) and demo endpoint
  - deterministic + policy-aware gate behavior
  - CI blockers in platform, security, release-train, and frontend lanes
