# Epic: v1.0 GA (FY27 Q4)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | `ga-v1.0`                                                |
| Target quarter   | FY27 Q4                                                  |
| Owner            | Engineering Manager, Factors                             |
| Status           | Planned                                                  |

## Scope

General Availability. Commercial surfaces open: billing, OEM,
real-time grid, SQL-over-HTTP, commercial packs.

* Stripe-backed metered billing.
* OEM endpoints (white-label per-tenant routing).
* Real-time grid factors (ENTSO-E, CAISO, ERCOT live feed).
* SQL-over-HTTP read endpoint (read-only DuckDB-style query layer).
* Commercial pack marketplace (curated packs only — private
  marketplace defers to v2.0).

## Out of scope

* Private packs uploaded by tenants — v2.0.
* Agent-native ingestion — v2.5.
* Marketplace open to 3rd-party publishers — v2.0.

## Deliverables

* Billing service (`greenlang/factors/billing/`) — metering on
  resolve + batch + ML calls; Stripe Connect for revenue share to
  data partners.
* OEM router — per-tenant API key, custom domain, branded receipts.
* Real-time grid feed — ENTSO-E + CAISO + ERCOT pollers with
  sub-minute publish latency.
* SQL endpoint (`/v1/sql`) — read-only, parameterised; allowlisted
  table set; per-tenant row-level security.
* Commercial pack catalog — at minimum 5 curated packs published.

## Acceptance criteria

* SLA: 99.9% monthly uptime on `/v1/factors`, `/v1/resolve`, `/v1/batch`.
* Billing reconciles end-of-month within $0.01 over 10k tenants.
* Real-time grid latency p95 < 60s from upstream publish to API.
* SQL endpoint passes SQL-injection + RLS test suite.
* SOC 2 Type II report issued (SEC-009 milestone).

## Source coverage

* v0.9 set + ENTSO-E live, CAISO live, ERCOT live, AIB GO certificates,
  PJM live, Australia AEMO live.

## API / SDK expectations

| Surface                  | v0.9 | v1.0 |
| ------------------------ | ---- | ---- |
| Billing endpoints        | OFF  | ON   |
| OEM routing              | OFF  | ON   |
| `/v1/sql`                | OFF  | ON   |
| Real-time grid factors   | OFF  | ON   |
| Commercial packs         | OFF  | ON   |

## Security / compliance gates

* SOC 2 Type II (SEC-009).
* PCI-DSS scope: Stripe handles all card data; we never see PANs.
* Real-time grid feed signed-by-source where available; provenance
  receipt pins source signature in audit trail.

## Tickets

* [ ] Billing service + Stripe integration.
* [ ] OEM router + custom-domain provisioning.
* [ ] Real-time grid pollers (ENTSO-E, CAISO, ERCOT, PJM, AEMO).
* [ ] SQL-over-HTTP gateway with RLS.
* [ ] Commercial pack catalog + first 5 curated packs.
* [ ] SOC 2 Type II audit completion.

## Dependencies

* v0.9 RC shipped + open beta seasoning data ≥ 30 days.
* SEC-009 (SOC 2 Type II) closed.
* INFRA-006 (API Gateway) at production capacity.

## Release risks

* SOC 2 audit timeline — long lead, may slip GA by a quarter.
* Real-time grid feed reliability — upstream outages bound our SLA.
* Stripe Connect onboarding for revenue-share data partners — KYC
  lead time can be > 4 weeks.
