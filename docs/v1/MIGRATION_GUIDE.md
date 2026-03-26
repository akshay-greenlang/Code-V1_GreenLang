# GreenLang v1 Migration Guide

## Purpose

Move existing app manifests into the v1 contract shape without breaking legacy app metadata files.

This guide is normative with `docs/v1/CONTRACTS.md` for:
- contract compatibility policy,
- deprecation windows,
- version bump governance.

## Migration Strategy

1. Keep existing app metadata untouched.
2. Add `v1/pack.yaml` and `v1/gl.yaml` per app.
3. Validate contracts via `gl v1 validate-contracts`.
4. Gate CI on contract conformance before promoting app lifecycle to `supported`.
5. Attach migration evidence (validator output + fixture parity) to release gate artifacts.

## App Targets

- `applications/GL-CBAM-APP/v1/`
- `applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1/`
- `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1/`

## Checklist

- [ ] `contract_version` set to `1.0` in both files.
- [ ] `runtime` set to `greenlang-v1`.
- [ ] pipeline stages exactly match: validate -> compute -> policy -> export -> audit.
- [ ] `runtime_conventions` block contains canonical command + exit codes.
- [ ] `security.signed=true` and at least one signature file exists.
- [ ] app lifecycle status set intentionally (`draft`, `candidate`, `supported`, `deprecated`).

## Rollback

If a migration introduces regression:

1. Set lifecycle to `candidate`.
2. Revert `v1/` contract changes for that app.
3. Re-run `gl v1 validate-contracts` and multi-app smoke lanes.
4. Re-apply migration in a new branch with updated fixtures.

## Version Bump Rules

- Do not change `contract_version` from `1.0` without an approved RFC and updated compatibility matrix.
- Any new mandatory contract field must ship with:
  - migration examples for CBAM/CSRD/VCCI,
  - validator negative fixture coverage,
  - release-gate docs update.

