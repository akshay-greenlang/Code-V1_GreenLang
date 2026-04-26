# Epic: v3.0 (FY31 Q1)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | post-GA                                                  |
| Target quarter   | FY31 Q1                                                  |
| Owner            | CTO / Factors GM                                         |
| Status           | Planned                                                  |

## Scope

Climate-OS native primitive. Per document:

* Factors becomes the single emissions-factor primitive across
  every GreenLang application (Scope 1/2/3 calculators, CSRD app,
  EUDR app, CBAM app, VCCI, building-BPS, taxonomy, etc.).
* Cross-product factor-pack reuse via shared registry.
* Federated catalog: regional partners run their own factor servers
  that federate into the GreenLang registry.
* Hourly grid factors at country granularity for every G20 grid.

## Out of scope

* Anything beyond G20 grid hourly coverage — v3.5+.

## Deliverables

* Cross-product factor SDK (Python + TS + Go).
* Federation protocol + registry-of-registries.
* Hourly grid coverage matrix complete for G20.

## Acceptance criteria

* Every GreenLang app uses the same factors primitive (no
  app-private factor stores).
* Federation: ≥ 5 regional partner registries live.
* Hourly grid coverage: 20/20 G20 economies with < 60s publish
  latency.

## Tickets

* [ ] Cross-product SDK consolidation.
* [ ] Federation protocol design + reference implementation.
* [ ] Regional partner onboarding (5).
* [ ] G20 hourly-grid coverage gap fill.

## Dependencies

* v2.5 shipped; agent-ingestion stable for the long tail of
  regional grid feeds.
* Cross-product PMs aligned on the migration path off app-private
  factor stores.

## Release risks

* Migration risk on apps that already shipped private factor
  stores.
* Federation protocol — once published, hard to change.
