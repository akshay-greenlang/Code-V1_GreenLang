# v0.5 Closed Beta — Partner Pipeline

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Phase 0 baseline (candidate list seeded; promotion happens during v0.5 ramp) |
| Owner            | Partner Success Lead (R) / CTO (A)                                          |
| Linked release   | `epics/epic-v0.5-closed-beta.md`                                            |
| Target           | 5–8 candidate partners; ≥ 5 admitted to v0.5 closed beta                    |

## Why a pipeline, not a final list

The CTO doc requires us to *identify* 5–8 candidates in Phase 0, not
to lock final admissions. The pipeline below is the working list.
Partners are promoted to active beta only when:

* v0.1 alpha exits with both v0.1 partners green;
* the partner has executed an MSA + NDA (+ DPA if applicable);
* the resolve / explain stack (`/v1/resolve`, `/v1/explain`) is
  serving the partner's expected method profile in staging;
* the partner has a named technical sponsor and a sponsor calendar
  for the beta window.

## Candidate Profile Slots

The list below seeds the slots. Real partner names replace the
slot ids as Partner Success closes intro conversations. Each slot
calls out the segment we want to validate against; the goal is
**diversity of use-case**, not just count.

| Slot id           | Segment                                       | Region               | Why this slot                                                                                                  | Owner               |
| ----------------- | --------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------- |
| `EU-OEM-01`       | EU automotive OEM (tier-0 supplier)           | EU                   | Validates resolve over GLEC freight + DEFRA + Ecoinvent lookup; confirms `/v1/explain` audit-trail clarity     | Partner Success Lead |
| `US-CONSULT-01`   | US-based GHG consultant                       | US                   | Validates explain + edition-pin workflow for CSRD-prep filings                                                  | Partner Success Lead |
| `IN-MFG-01`       | India steel / metals manufacturer              | IN                   | Validates CBAM exporter view for steel + iron defaults; tests resolve + explain in CBAM family                 | Partner Success Lead |
| `EU-RETAIL-01`    | EU retailer / consumer goods                  | EU                   | Validates Scope 3 cat-1 (purchased goods) resolve over multi-source spend categoriser                          | Partner Success Lead |
| `US-DATA-01`      | US tech / data platform                        | US                   | Validates real-time grid lookup APIs (location-based + market-based) at scale; pre-cursor to v1.0 OEM         | Partner Success Lead |
| `EU-FIN-01`       | EU bank / asset manager                       | EU                   | Validates portfolio-level Scope 3 cat-15 (investments) factor coverage; precursor to GL-Taxonomy-APP integration | Partner Success Lead |
| `IN-CONSULT-01`   | India-based sustainability consultancy         | IN                   | Validates BRSR + India CBAM exporter workflow; complements `IN-EXPORT-01` from v0.1                             | Partner Success Lead |
| `GLOBAL-PLATFORM-01` | Global ESG platform (re-distributor)        | Multi                | Validates SDK contract from a downstream-platform integrator perspective; contributes to v1.0 OEM design         | Partner Success Lead |

## Slot Owner / Status Tracker

This compact tracker lives below; full status migrates to
`DESIGN_PARTNER_TRACKER.md` once a slot is filled with a real partner.

| Slot id              | Conversation status   | Named partner (when filled) | Notes |
| -------------------- | --------------------- | --------------------------- | ----- |
| `EU-OEM-01`          | not started           |                             |       |
| `US-CONSULT-01`      | not started           |                             |       |
| `IN-MFG-01`          | not started           |                             |       |
| `EU-RETAIL-01`       | not started           |                             |       |
| `US-DATA-01`         | not started           |                             |       |
| `EU-FIN-01`          | not started           |                             |       |
| `IN-CONSULT-01`      | not started           |                             |       |
| `GLOBAL-PLATFORM-01` | not started           |                             |       |

## Diversity Goals

The closed-beta cohort SHOULD cover, before promotion:

* ≥ 2 regions (EU + IN + US)
* ≥ 2 verticals (manufacturer / consultant / retailer / financial)
* ≥ 1 platform integrator (drives SDK contract feedback)
* ≥ 1 use-case in each of CBAM, Scope 2, Scope 3 cat-1, freight

## Operating Cadence

* Partner Success Lead pings each slot weekly during the v0.5
  ramp.
* CTO reviews the slot tracker bi-weekly.
* Slot conversion to live beta partner triggers a row in
  `DESIGN_PARTNER_TRACKER.md` and a feedback-memo schedule.

## Cross-References

* Tracker: `DESIGN_PARTNER_TRACKER.md`
* v0.5 epic: `../epics/epic-v0.5-closed-beta.md`
* v0.1 plan: `V0_1_PARTNER_PLAN.md`
