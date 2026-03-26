# GreenLang v1 Security and Policy Baseline

## Auth and Policy Conventions

- Default-deny stance for protected operations.
- Contract and policy checks are release blockers for v1 app profiles.
- Supported-tier app profiles require signed-pack evidence in `pack.yaml`.

## Signed-Pack Baseline

Each v1 profile `pack.yaml` must set:

- `security.signed: true`
- `security.signatures: [...]` with at least one existing file

## Runtime Gate Commands

- `gl v1 check-policy`
- `gl v1 gate`

## CI Enforcement

The v1 platform workflow executes contract tests and gate checks before merge.

