# v1 App Runbook Template

Use this template for each app promoted into the v1 app set.

## App Identity

- App ID:
- Owner team:
- On-call channel:

## Runtime Conventions

- Canonical command:
- Success exit code:
- Blocked/policy exit code:

## Required Artifacts

- Export artifact(s):
- `audit/run_manifest.json`
- `audit/checksums.json`

## Security and Policy

- Signed-pack evidence path:
- Policy baseline checks:
- Exceptions (if any):

## Determinism Procedure

1. Run workflow with fixed input A into output dir 1.
2. Run workflow with same input A into output dir 2.
3. Compare checksums and artifact set.
4. Record result and hash parity evidence.

## SLO Targets

- Contract validation reliability:
- Gate workflow reliability:
- Determinism parity:

## Incident Procedures

- Contract check failures:
- Signed-pack policy failures:
- Determinism mismatch response:
- Release gate rollback:

