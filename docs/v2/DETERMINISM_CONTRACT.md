# GreenLang V2 Determinism Contract

## Contract

Determinism is a release-blocking platform requirement for regulated workflows.

## Required Evidence Per Workflow

1. identical input replay (run A, run B).
2. same artifact fileset.
3. zero hash diffs for contract artifacts.
4. stable audit metadata fields required by policy.

## Mandatory Artifacts

- `audit/run_manifest.json`
- `audit/checksums.json`
- domain output artifact(s) declared in `artifact_contract`

## Gate Commands

- `gl v2 validate-contracts`
- determinism replay test suite in `tests/v2`
- release lane checks in CI

## Failure Policy

- any determinism diff in regulated-critical lane blocks release train promotion.
