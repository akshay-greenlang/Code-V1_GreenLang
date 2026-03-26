# GreenLang v1 Contracts

This document defines the canonical v1 manifest contracts used by the first app set.

## Contract Files

- `pack.yaml` (v1 pack contract)
- `gl.yaml` (v1 pipeline contract)

For v1 rollout, app-specific contract instances are tracked under each app's `v1/` directory.

## `pack.yaml` Required Fields

- `contract_version`: `"1.0"`
- `name`: stable pack name
- `app_id`: app identity
- `version`: semantic version
- `kind`: `"pack"`
- `runtime`: `"greenlang-v1"`
- `entry_pipeline`: default pipeline file
- `metadata.owner_team`
- `metadata.support_channel`
- `metadata.lifecycle`: `draft | candidate | supported | deprecated`
- `security.signed`: boolean
- `security.signatures`: list of signature artifact paths

## `gl.yaml` Required Fields

- `contract_version`: `"1.0"`
- `app_id`
- `pipeline_id`
- `runtime`: `"greenlang-v1"`
- `stages`: exact ordered stage types:
  1. `validate`
  2. `compute`
  3. `policy`
  4. `export`
  5. `audit`
- `runtime_conventions.command`
- `runtime_conventions.success_exit_code`
- `runtime_conventions.blocked_exit_code`
- `runtime_conventions.artifact_contract`

## Validation

Use the v1 validator command:

```bash
gl v1 validate-contracts
```

This checks all three v1 app profile directories and ensures pack signatures referenced in `pack.yaml` exist on disk.

## Compatibility and Deprecation Policy (v1 Global Baseline)

### Contract Version Policy

- `contract_version` is immutable for a released profile.
- v1 validators currently accept only `1.0`; any future value requires:
  1. a versioned RFC approved by the release board,
  2. compatibility matrix update in this document,
  3. migration notes in `docs/v1/MIGRATION_GUIDE.md`,
  4. negative and positive validator fixture updates in `tests/v1`.

### Compatibility Matrix

| Producer Manifest | Validator Target | Expected Result |
| --- | --- | --- |
| `1.0` | `1.0` validator | pass |
| `1.0` + unknown required field removal | `1.0` validator | fail |
| non-`1.0` | `1.0` validator | fail |

### Deprecation Windows

- Lifecycle deprecation is two-stage:
  - `supported -> deprecated` (announce with migration timeline).
  - removal only after one release cycle with explicit go/no-go sign-off.
- Field deprecations require:
  - compatibility shim or explicit fail-fast validation behavior,
  - migration examples for all three app profiles,
  - CI gates proving no app-specific contract fork.

### Mandatory Governance Artifacts

- `docs/v1/CONTRACTS.md` (this file)
- `docs/v1/MIGRATION_GUIDE.md`
- `docs/v1/DOCS_CONTRACT.md`
- `tests/v1/test_contracts.py` negative + positive fixtures

