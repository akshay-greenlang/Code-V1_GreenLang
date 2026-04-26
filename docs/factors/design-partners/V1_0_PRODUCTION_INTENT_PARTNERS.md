# v1.0 GA — Production-Intent Partners

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Phase 0 baseline (target list)                                              |
| Owner            | Partner Success Lead (R) / CTO (A)                                          |
| Linked release   | `epics/epic-v1.0-ga.md`                                                     |
| Target           | At least 3 serious production-intent partners tracked through v1.0 GA       |

## What "production-intent" means

A production-intent partner is one whose declared intent is to put
GreenLang Factors on the critical path of a production workflow at
GA, not a pilot. Concretely:

* Has named a technical sponsor and an executive sponsor.
* Has a documented production use-case (CSRD filing, CBAM filing,
  CDP submission, internal Scope 1/2/3 calculator, etc.).
* Has a budget envelope (paid contract or paid-pilot) for the
  v1.0 timeframe.
* Is willing to sign the v1.0 commercial MSA (post-pilot).

The Phase 0 ask is to identify ≥ 3 such partners before v0.9 starts;
they are promoted out of the v0.5 cohort or recruited fresh.

## Target Slots (≥ 3)

| Slot id                  | Segment                              | Region | Production use-case                                                                | Status         | Owner               |
| ------------------------ | ------------------------------------ | ------ | ---------------------------------------------------------------------------------- | -------------- | ------------------- |
| `PROD-EU-CSRD-01`        | EU large-cap (CSRD wave-1 reporter)  | EU     | ESRS E1 disclosure-grade factor coverage; signed receipts in audit binder         | not started    | Partner Success Lead |
| `PROD-IN-CBAM-01`        | India exporter (CBAM-covered import) | IN/EU  | Quarterly CBAM declarations using GreenLang factor packs as the reference         | not started    | Partner Success Lead |
| `PROD-US-SEC-CLIM-01`    | US large-cap (SEC climate rule)      | US     | Annual climate-related disclosure with auditable factor lineage                   | not started    | Partner Success Lead |
| `PROD-PLATFORM-OEM-01`   | Sustainability platform (OEM)        | Multi  | White-labelled GreenLang Factors OEM endpoints in their commercial product         | not started    | Partner Success Lead |

The first three are the **minimum 3 serious partners** required by
the CTO doc. The fourth (`PROD-PLATFORM-OEM-01`) is a stretch slot
that drives the v1.0 OEM surface design.

## Promotion Criteria

A v0.5 candidate slot becomes a v1.0 production-intent slot when:

* Partner has filed at least one feedback memo against v0.5.
* Partner has executed a paid (or paid-pilot) MSA addendum.
* Partner has named both a technical and an executive sponsor.
* Partner has signed off on the v1.0 commercial pricing envelope
  (per Pricing & Packaging — held outside this repo).

## Operating Rules

1. The 3-partner minimum is gate criteria for the v1.0 epic exit.
2. A partner that misses 2 consecutive feedback windows is
   demoted from production-intent to evaluating; replacement
   target is added.
3. Production-intent rows roll up into the master tracker
   (`DESIGN_PARTNER_TRACKER.md` `Tracker Table — v1.0 Production-Intent`).
4. CTO reviews the production-intent list quarterly; CTO has
   override authority to add or remove slots.

## Cross-References

* Tracker: `DESIGN_PARTNER_TRACKER.md`
* v1.0 epic: `../epics/epic-v1.0-ga.md`
* v0.5 pipeline: `V0_5_PARTNER_PIPELINE.md`
