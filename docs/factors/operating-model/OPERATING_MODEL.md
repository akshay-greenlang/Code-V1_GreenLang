# GreenLang Factors — Operating Model

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Accepted - Phase 0 closed; interim named-human assignments recorded.        |
| Date             | 2026-04-26                                                                  |
| Owner            | CTO                                                                         |
| Related          | `RACI.md`, `engineering/ENGINEERING_CHARTER.md`, `epics/`                   |

The operating model makes ownership explicit. Phase 0 assigns a
named interim person per area. Permanent replacements require a CTO
update to this file and a RACI update if decision rights change.

## 1. Functional Areas + Owner Roles

| Area                       | Owner role                       | Named human                              | Primary responsibility                                                                                       |
| -------------------------- | -------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Platform / Data            | Platform/Data Lead               | Akshay (interim CTO owner)               | Schema, storage, release infrastructure, edition cuts, release manifests                                     |
| Backend / API              | Backend/API Lead                 | Akshay (interim CTO owner)               | REST API, GraphQL (v0.9+), auth, entitlements, route filter, OpenAPI contract                                |
| Data Engineering / Parsers | Data Engineering Lead            | Akshay (interim CTO owner)               | Source fetchers, parsers, normalizer, ingestion pipeline, parser snapshot tests                              |
| Climate Methodology        | Climate Methodology Lead         | Akshay (interim CTO owner)               | Methodology correctness, source registry curation, factor-record review, exception methodology docs          |
| SRE / Infra                | SRE Lead                         | Akshay (interim CTO owner)               | Deployment, observability, alerting, performance budgets, incident response                                  |
| DevRel / Docs              | DevRel Lead                      | Akshay (interim shared-FTE owner)        | Public docs, SDK examples, onboarding guides, partner-facing tutorials, release-notes drafting               |
| Compliance / Security      | Compliance/Security Lead         | Akshay (interim shared-FTE owner)        | SOC 2 path, legal, source licensing enforcement, audit controls, pen-test program                            |
| Design Partner Success     | Partner Success Lead             | Akshay (interim through v0.5)            | Partner onboarding, MSA/NDA tracking, feedback memo collection, gap-to-ticket conversion                     |

## 2. Engineering Org Layout (Reference)

* **Engineering Manager, Factors** — single-threaded delivery owner
  across the 8 areas. Reports to CTO.
* **Platform/Data**, **Backend/API**, **Data Engineering** — three
  delivery squads, each with a Tech Lead in the role above.
* **Climate Methodology** — embedded specialist (often dual-hatted
  with another product); reviews every parser approval and
  methodology-pack change.
* **SRE** — shared platform team; named SRE assigned to Factors.
* **DevRel** — shared; one named developer-advocate for Factors.
* **Compliance / Security** — shared; named compliance contact for
  Factors.
* **Partner Success** — shared (or interim CTO-led until v0.5).

## 3. Decision Cadence

| Cadence       | Forum                              | Required attendees                                          | Output                                  |
| ------------- | ---------------------------------- | ----------------------------------------------------------- | --------------------------------------- |
| Daily         | Async Slack stand-up               | All squad TLs                                               | Blocker list, in-progress tickets       |
| Weekly        | Factors product council            | CTO, Eng Mgr Factors, Platform/Data Lead, Backend/API Lead, Data Eng Lead, Climate Methodology Lead | Release-readiness, source coverage decisions, RACI escalations |
| Bi-weekly     | Partner-success review             | Partner Success Lead, CTO, Eng Mgr Factors                  | Partner status table, gap-to-ticket triage |
| Per release   | Release council                    | Platform/Data Lead (chair), CTO, SRE Lead, Backend/API Lead, Climate Methodology Lead, Partner Success Lead | Release manifest sign-off |
| Per incident  | Incident channel + post-mortem     | SRE Lead (incident commander), affected component owner     | Post-mortem in `docs/factors/postmortems/` |
| Quarterly     | Roadmap review                     | CTO, all area owners, downstream-product PMs                | Updated `epics/` + ADR set              |

## 4. Hand-off Rules

* No code change to a component without the area owner's review.
* No source added or removed without Climate Methodology Lead +
  Compliance/Security Lead joint approval (licence + methodology).
* No production deployment without Platform/Data Lead and SRE Lead
  joint sign-off on the release manifest.
* No customer-impacting communication without Partner Success Lead
  + CTO (or designate) sign-off.

## 5. On-Call

* Factors API (read path) — SRE on-call rotation; severity-1
  paging tied to `factors-v0.1-alpha-alerts.yaml`.
* Source ingestion (write path) — Data Engineering on-call;
  severity-2 — best-effort within 4 business hours.
* Release pipeline failures — Platform/Data Lead is single point of
  contact during a release window; SRE assists.

## 6. Hiring / Capacity Plan (Reference)

| Area                       | Phase 0 / v0.1 | v0.5      | v0.9      | v1.0      |
| -------------------------- | -------------- | --------- | --------- | --------- |
| Platform / Data            | 1 TL + 1 SDE   | +1 SDE    | same      | +1 SDE    |
| Backend / API              | 1 TL + 1 SDE   | +1 SDE    | +1 SDE    | same      |
| Data Engineering / Parsers | 1 TL + 1 SDE   | +1 SDE    | +1 SDE    | +1 SDE    |
| Climate Methodology        | 1 specialist   | +1        | same      | +1        |
| SRE / Infra                | 0.5 FTE        | 1 FTE     | 1 FTE     | 1.5 FTE   |
| DevRel / Docs              | 0.5 FTE        | 1 FTE     | 1 FTE     | 1 FTE     |
| Compliance / Security      | 0.25 FTE       | 0.5 FTE   | 0.5 FTE   | 1 FTE     |
| Design Partner Success     | interim        | 1 FTE     | 1 FTE     | 1 FTE     |

This is the planning baseline; exact headcount is tracked outside
this repo. Phase 0 confirms the shared-FTE baseline for DevRel / Docs
and Compliance / Security through v0.5. If headcount is not hired by
the start of v0.5, the interim CTO owner remains accountable for
release-blocking decisions and may delegate execution, but not
approval.

## 7. Phase 0 Closure Decisions

* Named-human assignments for all 8 areas are closed with Akshay as
  interim CTO owner until named leads are recorded in this file.
* DevRel / Docs allocation is confirmed at 0.5 FTE for v0.1 and 1 FTE
  planned for v0.5.
* Compliance / Security allocation is confirmed at 0.25 FTE for v0.1
  and 0.5 FTE planned for v0.5.
* Design Partner Success remains interim CTO-owned through v0.5 or
  until a named Partner Success Lead is appointed, whichever happens
  first.
