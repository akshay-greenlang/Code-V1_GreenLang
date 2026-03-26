# GreenLang v1 Charter

## Purpose

Define the v1 platformization release as a contract-driven program that turns MVP success into a reusable, consistent multi-app pattern.

## v1 Scope

- In scope:
  - Three strategic app patterns: CBAM, CSRD, VCCI
  - Shared `pack.yaml` and `gl.yaml` contracts (v1)
  - Unified CLI/runtime semantics for v1 app flows
  - Shared auth/policy baseline with signed-pack gate
  - Determinism, auditability, and observability baseline
  - Multi-app CI smoke and regression lanes
  - Standardized docs/runbook contract
- Out of scope:
  - Full 10-app rollout
  - Full 100+ agent hardening
  - All packs promoted to supported tier
  - Full enterprise ops stack rollout

## Release Owners

- Program owner: Platform PM
- Runtime and CLI owner: Core Platform team
- Contract owner: Schema and Pack Governance team
- Security/policy owner: Governance and Security team
- Determinism/audit owner: Provenance team
- CI/CD owner: DevEx and Release Engineering team
- Docs/DX owner: Documentation and Enablement team

## Quality Bar

- Multi-app regression suite is green for the three v1 apps.
- Signed pack verification policy is enforced in gate checks.
- Core workflow SLOs are defined and validated for release candidate.

## Exit Criteria

1. Three apps are v1 production-ready by shared gate checks.
2. Shared pack lifecycle is stable for create, validate, sign, and verify.
3. Cross-app runtime conventions are frozen as v1 baseline.

## Success Metrics

- Contract conformance: 100% for designated v1 manifests.
- Runtime conformance: 100% pass on v1 command and exit semantics tests.
- Deterministic rerun parity: 100% for required flagship workflows.
- Signed-pack enforcement: 100% for supported-tier packs in v1 lane.
- Docs-led bootstrap success: new team can run v1 smoke workflow from docs.

## Program Cadence

- Weekly architecture and release gate review.
- Weekly risk review with CBAM MVP non-regression checkpoint.
- Milestone sign-off per phase with documented pass/fail evidence.

