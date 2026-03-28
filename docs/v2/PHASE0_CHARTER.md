# GreenLang V2 Phase 0 Charter

## Mission

Scale GreenLang from V1 portfolio foundation to V2 platform productization with strict quality gates:

- 6-8 production-ready apps
- 35+ managed packs in quality tiers
- deterministic and auditable release contract
- org-level policy, security, and release governance

## Scope

- In scope: runtime harmonization, pack tiering, release trains, connector reliability, enterprise UX shell.
- Deferred: full legacy harmonization and active support for all historical/experimental code.

## V2 App Expansion Shortlist (6-8)

- GL-CBAM-APP
- GL-CSRD-APP
- GL-VCCI-Carbon-APP
- GL-EUDR-APP
- GL-GHG-APP
- GL-ISO14064-APP
- GL-CDP-APP (candidate)
- GL-TCFD-APP (candidate)

## Pack Onboarding Targets by Tier

- experimental: 10 packs
- candidate: 12 packs
- supported: 10 packs
- regulated-critical: 3 packs
- total: 35 packs

## Quality and Release Bars

- platform-wide matrix must pass runtime/contracts/pack policy checks for selected V2 app set.
- org-level security and compliance gates must be blocking on protected branches.
- determinism parity must pass for regulated workflows before RC promotion.
- migration playbooks must exist for contract/runtime/pack lifecycle changes.

## Governance Model

- **Release Board**: final authority for RC and go/no-go.
- **Architecture Board**: contract and compatibility decisions.
- **Security Council**: policy bundles and exception approvals.
- **SRE On-call Board**: SLO compliance and incident readiness.

## Release Authority Model

| Stage | Required Gates | Final Authority |
| --- | --- | --- |
| PR Smoke | contracts, docs, smoke tests | Architecture Board delegate |
| Full Gate | full matrix, policy checks, determinism checks | Release Engineering lead |
| RC Promotion | full gate + security blockers + soak checks | Release Board chair |
| Stable Release | RC success + go/no-go record + incident review | Release Board |

## Program KPIs

- 6+ apps pass production readiness checklist.
- 35+ packs onboarded with tier status and policy evidence.
- 100% runtime conformance for selected V2 apps.
- 100% determinism parity for regulated flagship workflows.
- two consecutive successful release train cycles.

## Exit Conditions for Phase 0

1. ownership matrix approved.
2. RFC process approved.
3. V2 scope and deferred scope approved.
4. release train authority model ratified.
5. app expansion shortlist and pack tier targets ratified.
