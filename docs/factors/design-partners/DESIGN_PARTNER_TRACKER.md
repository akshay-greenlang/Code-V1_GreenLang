# GreenLang Factors — Design Partner Tracker

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Phase 0 baseline (live tracker)                                             |
| Owner            | Partner Success Lead (R) / CTO (A)                                          |
| Update cadence   | Weekly during alpha; bi-weekly during beta; monthly thereafter              |
| Confidentiality  | This file holds tracker *status*, never partner identifying data            |
| Related          | `MSA-NDA-status-tracker.md` (legal status), `V0_1_PARTNER_PLAN.md`, `V0_5_PARTNER_PIPELINE.md`, `V1_0_PRODUCTION_INTENT_PARTNERS.md` |

This is the cross-release operating tracker. The two existing
v0.1-specific documents (`MSA-NDA-status-tracker.md` for contract
state and per-partner profile files) feed this tracker; this is the
single roll-up the Partner Success Lead and CTO read at the
weekly partner review.

## Status Legend

| Field group                      | Allowed values                                              |
| -------------------------------- | ----------------------------------------------------------- |
| `MSA status`                     | `not_started`, `in_review`, `in_redline`, `signed`, `expired` |
| `NDA status`                     | `not_started`, `in_review`, `in_redline`, `signed`, `not_required`, `expired` |
| `DPA status`                     | `not_started`, `in_review`, `in_redline`, `signed`, `not_required`, `expired` |
| `API key status`                 | `not_issued`, `issued`, `revoked`, `expired`                |
| `SDK calculation status`         | `not_started`, `in_progress`, `successful`, `blocked`       |
| `Feedback memo status`           | `not_due`, `requested`, `received`, `late`                  |
| `Partner intent`                 | `evaluating`, `pilot`, `production-intent`, `paid-customer`  |

## Tracker Table — v0.1 Alpha (FY27 Q1)

Source-of-record: `partner-IN-EXPORT-01.md`, `partner-EU-MFG-01.md`,
`MSA-NDA-status-tracker.md`.

| Partner ID      | Segment                  | Region | Use case                              | MSA   | NDA   | DPA              | API key      | SDK calc       | Feedback memo | Product gaps         | Owner               |
| --------------- | ------------------------ | ------ | ------------------------------------- | ----- | ----- | ---------------- | ------------ | -------------- | ------------- | -------------------- | ------------------- |
| `IN-EXPORT-01`  | India textile exporter   | IN     | India electricity Scope 2 + CBAM view | `pending_legal_review` | `pending_legal_review` | `not_required` | `not_issued` | `not_started` | `not_due`     | tracked in `partner-IN-EXPORT-01.md` | Partner Success Lead |
| `EU-MFG-01`     | EU cement manufacturer   | EU-IT  | CBAM cement clinker default lookup    | `pending_legal_review` | `pending_legal_review` | `pending_legal_review` | `not_issued` | `not_started` | `not_due`     | tracked in `partner-EU-MFG-01.md`    | Partner Success Lead |

## Tracker Table — v0.5 Closed Beta (FY27 Q2 — pipeline)

See `V0_5_PARTNER_PIPELINE.md` for the candidate list (5–8 partners).
Partners promoted from pipeline to active beta are appended to this
table once MSA work begins.

| Partner ID      | Segment                  | Region | Use case                              | MSA   | NDA   | DPA   | API key | SDK calc | Feedback memo | Product gaps | Owner |
| --------------- | ------------------------ | ------ | ------------------------------------- | ----- | ----- | ----- | ------- | -------- | ------------- | ------------ | ----- |
| (to be filled as partners are promoted from `V0_5_PARTNER_PIPELINE.md`) |

## Tracker Table — v1.0 Production-Intent (FY27 Q4 — pipeline)

See `V1_0_PRODUCTION_INTENT_PARTNERS.md` for the named target list
(at least 3 serious partners).

| Partner ID      | Segment                  | Region | Use case                              | Intent              | Notes |
| --------------- | ------------------------ | ------ | ------------------------------------- | ------------------- | ----- |
| (to be filled from `V1_0_PRODUCTION_INTENT_PARTNERS.md`)             |

## Lifecycle Workflow

```
candidate (V0_5_PARTNER_PIPELINE / V1_0_PRODUCTION_INTENT_PARTNERS)
    │
    │  Partner Success Lead opens conversation
    ▼
intro / qualification
    │
    │  CTO + PM approve onboarding
    ▼
contract execution (MSA → NDA → DPA if applicable)
    │
    │  Compliance/Security Lead validates licence + region
    ▼
tenant provisioning (API key + entitlement)
    │
    │  SRE Lead provisions; key never written to repo
    ▼
SDK onboarding (1+ successful client.calculate(...) call)
    │
    │  DevRel Lead supports
    ▼
30-day pilot window (per release)
    │
    │  Partner files feedback memo (template:
    │  PARTNER_FEEDBACK_MEMO_TEMPLATE.md)
    ▼
gap-to-ticket triage
    │
    │  Partner Success Lead opens tickets under matching epic
    ▼
release-window review
    │
    │  promote / hold / churn decision
    ▼
production-intent or graduation to next release wave
```

## Operating Rules

1. A row in this tracker MUST link to either a partner profile
   file (`partner-<slug>.md`) or, for unprovisioned candidates,
   an entry in `V0_5_PARTNER_PIPELINE.md` /
   `V1_0_PRODUCTION_INTENT_PARTNERS.md`.
2. `API key issued` is gated by MSA + NDA + (DPA if applicable)
   all in `signed` state. Operator has the deny-list authority.
3. Any row stuck `not_started` for `SDK calc` for > 14 days
   triggers a Partner Success → CTO escalation.
4. Any feedback memo status going `late` triggers a Partner
   Success → Partner re-engagement.
5. Tracker row deletions are forbidden — partners that churn move
   to `intent: lost`, with a reason note. History is append-only.

## Linked Files

* `MSA-NDA-status-tracker.md` — legal contract status (operator-facing).
* `V0_1_PARTNER_PLAN.md` — v0.1 alpha named partners.
* `V0_5_PARTNER_PIPELINE.md` — v0.5 closed-beta candidates.
* `V1_0_PRODUCTION_INTENT_PARTNERS.md` — v1.0 GA production-intent targets.
* `PARTNER_FEEDBACK_MEMO_TEMPLATE.md` — per-partner feedback memo.
* `onboarding-checklist.md` — canonical onboarding flow.
* `pilot-success-criteria.md` — pass/fail bar.
* `partner-IN-EXPORT-01.md`, `partner-EU-MFG-01.md` — v0.1 partner profiles.
