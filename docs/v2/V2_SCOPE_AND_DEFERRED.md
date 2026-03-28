# GreenLang V2 Scope and Deferred Items

## In Scope (V2)

1. expand to 6-8 production-grade apps under one runtime contract.
2. onboard 35+ packs into managed lifecycle tiers.
3. enforce agent lifecycle/version/deprecation governance.
4. implement connector reliability standards with SLO-backed operations.
5. enforce org-level security and policy gates in release workflows.
6. formalize release trains with blocking quality tiers.
7. deliver enterprise multi-app UX shell governance and quality gates.

## In-Scope Portfolio Table

| Area | Target |
| --- | --- |
| App portfolio | 6-8 active V2 apps |
| Pack ecosystem | 35+ packs across managed tiers |
| Governance domains | runtime, pack lifecycle, agent lifecycle, security, determinism, release trains, UX |
| Operations | on-call model, runbooks, SLO accountability |

## Deferred (Post-V2)

1. complete harmonization of all legacy or experimental repository code.
2. universal support for all historical app branches.
3. wholesale migration of every historical pack to supported tier.
4. non-prioritized low-adoption connectors outside SLO program.

## Deferred Legacy Support Policy

- legacy and experimental code remains discoverable but is not part of guaranteed production support.
- fixes for deferred legacy areas require explicit RFC approval and release-board prioritization.
- deferred legacy components are excluded from mandatory V2 release train pass criteria.

## Non-Goals

- "everything in repo" as active production surface.
- bypassing release train policy for ad-hoc production deployment.
- allowing unsigned or ungoverned packs into supported tiers.
