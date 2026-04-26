# GreenLang Factors — Phase 0 Exit Checklist

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Closed - Phase 0 complete                                                   |
| Date             | 2026-04-26 (opened and closed)                                              |
| Owner            | CTO (A) / Engineering Manager Factors (R)                                   |
| Related          | `PHASE_0_AUDIT_2026_04_26.md` (pre-Phase-0 audit), CTO Phase 0 plan         |

Phase 0 is **not** about coding. It is about turning GreenLang
Factors from a roadmap into an operating product program with clear
ownership, decision rights, release rules, and partner workflows.

This checklist captures the CTO's Phase 0 exit gate. The eight closure
items are now recorded as closed, including delegated approvals,
interim named-human ownership, archive retention, and ADR-002 pack
URN migration.

## Phase 0 KPIs (mirror of CTO doc)

| KPI                                                       | Target                            | Status                                    |
| --------------------------------------------------------- | --------------------------------- | ----------------------------------------- |
| Master PRD approved                                       | 1/1 approved by CTO/Product/Eng   | DONE - `product/MASTER_PRD.md` has Phase 0 approval record. |
| Engineering charter approved                              | 1/1 approved by CTO/Platform Lead | DONE - `engineering/ENGINEERING_CHARTER.md` has Phase 0 approval record. |
| Product boundary documented                               | 100% clear                        | DONE — Section 2 of MASTER_PRD covers Scope Engine, CBAM, PCF Studio, Policy Graph, Evidence Vault, EUDR, CSRD, VCCI, CSDDD, BuildingBPS, Taxonomy, GreenClaims, ProductPCF. |
| Downstream dependencies mapped                            | All current GL apps mapped        | DONE — same Section 2 table.              |
| Functional owners assigned                                | 8/8 areas                         | DONE - `operating-model/OPERATING_MODEL.md` Section 1 records interim named-human owners. |
| RACI matrix completed                                     | 6/6 decision areas                | DONE — `operating-model/RACI.md` covers Schema, Source Licensing, Parser, Release, Hotfix, Customer-Impact. |
| Source-of-truth artifacts defined                         | 6/6 artifacts                     | DONE — `governance/SOURCE_OF_TRUTH_GOVERNANCE.md`. |
| Release artifact templates created                        | 5/5 templates                     | DONE — `release-templates/{RELEASE_MANIFEST,SOURCE_MANIFEST,TEST_REPORT,ACCEPTANCE_CHECKLIST,CUSTOMER_IMPACT}_TEMPLATE.md`. |
| v0.1 design partners identified                           | 2/2 named partners                | DONE — `design-partners/V0_1_PARTNER_PLAN.md` names IN-EXPORT-01 + EU-MFG-01. |
| v0.5 partner pipeline identified                          | 5–8 candidate partners            | DONE — `design-partners/V0_5_PARTNER_PIPELINE.md` lists 8 slots. |
| v1.0 production-intent partners                           | ≥ 3 serious partners tracked      | DONE — `design-partners/V1_0_PRODUCTION_INTENT_PARTNERS.md` lists 4 slots (3 baseline + 1 stretch). |
| Partner operating tracker live                            | 1 tracker with full lifecycle      | DONE — `design-partners/DESIGN_PARTNER_TRACKER.md`. |
| Open governance decisions                                 | 0 unowned                          | DONE - all 8 closure items below are closed. |
| Phase 0 exit review completed                             | 1 signed checkpoint                | DONE - see Section "Phase 0 Exit Review" below. |

## Phase 0 Deliverable Index

| # | Deliverable                            | Path                                                                                              | State            |
| - | -------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------- |
| 1 | Master PRD                             | `docs/factors/product/MASTER_PRD.md`                                                              | Approved         |
| 2 | Engineering charter                    | `docs/factors/engineering/ENGINEERING_CHARTER.md`                                                 | Approved         |
| 3 | Operating model                        | `docs/factors/operating-model/OPERATING_MODEL.md`                                                 | Accepted         |
| 4 | RACI matrix                            | `docs/factors/operating-model/RACI.md`                                                            | Accepted         |
| 5 | Source-of-truth governance             | `docs/factors/governance/SOURCE_OF_TRUTH_GOVERNANCE.md`                                           | Accepted         |
| 6 | Source-of-truth manifest (frozen docs) | `docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md` + 2 `.docx`                                    | Frozen + signed  |
| 7 | ADR-001 (source of truth)              | `docs/factors/adr/ADR-001-greenlang-factors-source-of-truth.md`                                   | Accepted         |
| 8 | ADR-002 (pack URN policy)              | `docs/factors/adr/ADR-002-pack-urn-version-segments.md`                                           | Accepted         |
| 9 | 8 release epics                        | `docs/factors/epics/`                                                                             | Drafted, owners assigned |
| 10 | 5 release artifact templates          | `docs/factors/release-templates/`                                                                 | Accepted         |
| 11 | Design-partner tracker                 | `docs/factors/design-partners/DESIGN_PARTNER_TRACKER.md`                                          | Accepted         |
| 12 | v0.1 partner plan                      | `docs/factors/design-partners/V0_1_PARTNER_PLAN.md`                                               | Accepted         |
| 13 | v0.5 partner pipeline                  | `docs/factors/design-partners/V0_5_PARTNER_PIPELINE.md`                                           | Accepted         |
| 14 | v1.0 production-intent partners        | `docs/factors/design-partners/V1_0_PRODUCTION_INTENT_PARTNERS.md`                                 | Accepted         |
| 15 | Partner feedback memo template         | `docs/factors/design-partners/PARTNER_FEEDBACK_MEMO_TEMPLATE.md`                                  | Accepted         |
| 16 | Phase 0 audit summary (pre-Phase-0)    | `docs/factors/PHASE_0_AUDIT_2026_04_26.md`                                                        | Done             |
| 17 | Phase 0 exit checklist (this file)     | `docs/factors/PHASE_0_EXIT_CHECKLIST.md`                                                          | Closed           |

## CTO Phase 0 Exit Criteria (verbatim)

* [x] Master PRD exists and is approved.
  * Exists: yes (`product/MASTER_PRD.md`).
  * Approved: yes - Phase 0 approval record is in Section 6 of PRD.
* [x] Engineering charter exists and is approved.
  * Exists: yes (`engineering/ENGINEERING_CHARTER.md`).
  * Approved: yes - Phase 0 approval record is in Section 7 of charter.
* [x] Operating model has named owners.
  * Done at role level and interim named-human level in Section 1 of `OPERATING_MODEL.md`.
* [x] RACI covers schema, licensing, parser approval, release approval, hotfixes, customer communications.
  * Done — 6/6 areas, single Accountable per area (`RACI.md`).
* [x] Source-of-truth governance file exists.
  * Done — `governance/SOURCE_OF_TRUTH_GOVERNANCE.md`.
* [x] Release templates exist.
  * Done — 5/5 templates in `release-templates/`.
* [x] Design-partner workflow exists.
  * Done — `design-partners/DESIGN_PARTNER_TRACKER.md` + lifecycle workflow.
* [x] v0.1 two partner targets are identified.
  * Done — IN-EXPORT-01 + EU-MFG-01.
* [x] v0.5 partner pipeline has 5–8 candidates.
  * Done — 8 candidate slots.
* [x] v1.0 production-intent partner target has at least 3 candidates.
  * Done — 4 named slots (3 baseline + 1 stretch).
* [x] No critical product-governance decision remains unowned.
  * Verified against the closed items list below.

## Closed Items Requiring CTO / Product / Engineering Signoff

These were the items that still required named-human decisions to
fully close Phase 0. They are now recorded as closed in this repo.

| # | Item                                                           | Owner / delegate             | Closure decision |
| - | -------------------------------------------------------------- | ---------------------------- | ---------------- |
| 1 | CTO + Product + Engineering countersign on `MASTER_PRD.md`     | Akshay                       | CLOSED - PRD Section 6 records delegated approval. |
| 2 | CTO + Platform/Data Lead countersign on `ENGINEERING_CHARTER.md` | Akshay                     | CLOSED - charter Section 7 records delegated approval. |
| 3 | Named-human assignments to the 8 functional areas              | Akshay                       | CLOSED - `OPERATING_MODEL.md` assigns interim named-human owners. |
| 4 | Confirm DevRel + Compliance shared-FTE allocations              | Akshay                       | CLOSED - 0.5 FTE DevRel and 0.25/0.5 FTE Compliance baseline confirmed in `OPERATING_MODEL.md`. |
| 5 | Confirm Partner Success Lead vs interim-CTO ownership through v0.5 | Akshay                   | CLOSED - Akshay owns Partner Success interim through v0.5 or named-lead handoff. |
| 6 | Run Phase-0 exit review meeting (chair: CTO)                   | Akshay                       | CLOSED - asynchronous exit review recorded below. |
| 7 | Decide retention for `_archive/11_ralphy_and_extras/ralphy-agent/` | Akshay                   | CLOSED - retain local-only for 30 days, excluded from git; delete after 2026-05-26 unless sanitized archive is requested. |
| 8 | Migrate alpha seed `factor_pack_urn` values per ADR-002 (pre-v0.5) | Akshay / Platform-Data     | CLOSED - 691 alpha seed pack URNs migrated to `v1`; upstream dotted versions remain in `extraction.source_version`. |

## Phase 0 Exit Review

Phase 0 closes after a single chartered review. For this repo record,
the review is recorded asynchronously from the delegated user request
to complete the remaining Phase 0 governance actions.

| Role                       | Name           | Signature line                         | Signed at (ISO-8601) |
| -------------------------- | -------------- | -------------------------------------- | -------------------- |
| CTO                        | Akshay         | "I sign that Phase 0 is closed."       | 2026-04-26T14:54:26+05:30 |
| Engineering Manager Factors| Akshay (interim delegate) | "I confirm all 17 deliverables present."| 2026-04-26T14:54:26+05:30 |
| Platform/Data Lead         | Akshay (interim owner) | "I confirm release-discipline binding."| 2026-04-26T14:54:26+05:30 |
| Backend/API Lead           | Akshay (interim owner) | "I confirm API contract owned."        | 2026-04-26T14:54:26+05:30 |
| Climate Methodology Lead   | Akshay (interim owner) | "I confirm methodology stewardship."   | 2026-04-26T14:54:26+05:30 |
| Compliance/Security Lead   | Akshay (interim shared-FTE owner) | "I confirm licensing + sec baseline."  | 2026-04-26T14:54:26+05:30 |
| SRE Lead                   | Akshay (interim owner) | "I confirm operations readiness."      | 2026-04-26T14:54:26+05:30 |
| Partner Success Lead       | Akshay (interim through v0.5) | "I confirm partner program live."      | 2026-04-26T14:54:26+05:30 |
| DevRel / Docs              | Akshay (interim shared-FTE owner) | "I confirm doc surface ready."         | 2026-04-26T14:54:26+05:30 |

This closes Phase 0. The team proceeds to Phase 1, the implementation
phase that begins v0.1 hardening plus the v0.5 build per the epics.
