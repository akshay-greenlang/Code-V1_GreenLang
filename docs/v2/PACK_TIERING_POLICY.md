# GreenLang V2 Pack Tiering Policy

## Tier Definitions

- `experimental`: rapid iteration, non-blocking, limited support.
- `candidate`: pre-supported quality baseline, full validation required.
- `supported`: production-ready, signed, release-train eligible.
- `regulated-critical`: highest assurance for regulated workloads, signed and strict policy enforcement.
- `deprecated`: sunset stage with migration path and removal date.

## Promotion Criteria

1. contract validation pass (`pack.yaml` and `gl.yaml`).
2. deterministic output check pass for declared flagship workflows.
3. security scan and policy gate pass.
4. docs contract complete (runbook, quickstart, troubleshooting, migration notes).
5. ownership assigned with support channel and on-call mapping.

## Signature Rules

- `supported` and `regulated-critical` must set `security.signed=true`.
- at least one signature file is required when `security.signed=true`.
- unsigned packs in these tiers are blocked in CI and runtime conformance.

## Runtime and CI Enforcement

- runtime lifecycle evaluator: `greenlang/v2/pack_tiers.py`
- runtime install gate: `greenlang/ecosystem/packs/installer.py` via `_enforce_v2_tier_lifecycle`
- policy bundle: `greenlang/governance/policy/bundles/v2_pack_tier_policy.rego`
- CI lifecycle tests: `tests/v2/test_pack_tiers.py`, `tests/v2/test_pack_tier_lifecycle.py`
- pilot tier registry validation: `greenlang/ecosystem/packs/v2_tier_registry.yaml`

## Promotion Evidence by Tier

- `experimental`: owner + support channel required.
- `candidate`: `experimental` + docs contract evidence.
- `supported`: `candidate` + signed artifact + security scan evidence.
- `regulated-critical`: `supported` + determinism evidence.

## Demotion Rules

- recurring SLO violations or unresolved high-severity security findings trigger demotion.
- contract-breaking changes without migration notes trigger demotion.
- no active owner triggers demotion to `deprecated`.
