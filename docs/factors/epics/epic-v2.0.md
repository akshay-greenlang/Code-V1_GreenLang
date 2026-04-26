# Epic: v2.0 (FY29 Q2)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | post-GA                                                  |
| Target quarter   | FY29 Q2                                                  |
| Owner            | Enterprise Platform Lead                                 |
| Status           | Planned                                                  |

## Scope

Marketplace + community expansion. Per document:

* Open marketplace for 3rd-party factor packs (publishers under MSA).
* Community packs (free / open) listed alongside commercial packs.
* Pack quality / trust score (PQS) shown in catalog UI.
* Spend-data categoriser productised end-to-end.

## Out of scope

* Agent-native ingestion — v2.5.
* OEM-of-OEMs (white-label of white-label) — v3.0.

## Deliverables

* Marketplace publisher portal + onboarding KYC.
* PQS scoring service.
* Spend-data → factor-mapping pipeline (productised).

## Acceptance criteria

* ≥ 25 publishers onboarded with ≥ 5 packs each.
* PQS score correlates ≥ 0.8 with audit-team manual review on a
  100-pack sample.
* Spend-data pipeline: P@1 ≥ 75% on EBA-spend gold set.

## Tickets

* [ ] Marketplace portal + KYC.
* [ ] PQS scoring + dashboard.
* [ ] Spend categoriser productised + UI.

## Dependencies

* v1.5 shipped + private-pack ingestion stable for ≥ 6 months.

## Release risks

* Marketplace policy / liability — need legal sign-off on pack
  quality SLAs.
* PQS gaming — adversarial publishers may juice metrics; need
  red-team review.
