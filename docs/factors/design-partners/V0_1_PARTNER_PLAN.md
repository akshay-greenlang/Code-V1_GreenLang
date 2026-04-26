# v0.1 Alpha Design-Partner Plan (FY27 Q1)

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Phase 0 baseline                                                            |
| Owner            | Partner Success Lead (R) / CTO (A)                                          |
| Linked release   | `epics/epic-v0.1-alpha.md`                                                  |
| Acceptance gate  | Two named partners + one successful SDK calculation each before alpha exit  |

## Target

The CTO doc §19.1 binds v0.1 alpha to **two design-partner tenants**.
This document names them, fixes their use-case, sets the onboarding
date, and binds the success criteria.

## Named Partners

### 1. `IN-EXPORT-01` — India textile exporter

| Field                | Value                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Partner profile file | [`partner-IN-EXPORT-01.md`](./partner-IN-EXPORT-01.md)                                                 |
| Region               | India (Tamil Nadu / Maharashtra)                                                                       |
| Segment              | Mid-cap textile exporter; supplies EU OEMs                                                             |
| Use case             | India electricity Scope 2 location-based; CBAM exporter view (downstream of cement / steel embedded)   |
| Target SDK call      | `client.calculate(activity_kwh=1_200_000, factor_urn="urn:gl:factor:india-cea-co2-baseline:in:all_india:2025-26:cea-v22.0:v1")` |
| Allow-listed sources | `india_cea_co2_baseline`, `cbam_default_values`, `desnz_ghg_conversion` (read-only)                    |
| Methodology profile  | `corporate_scope2_location`                                                                            |
| Onboarding date      | Target FY27 Q1 W1; gated by Legal MSA execution                                                        |
| API key issuance     | Operator-issued via Vault; key sha256 prefix recorded in partner profile                               |
| Success criteria     | At least 1 successful SDK calculation run by partner in their environment within 14 days of API key    |

### 2. `EU-MFG-01` — EU cement manufacturer

| Field                | Value                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Partner profile file | [`partner-EU-MFG-01.md`](./partner-EU-MFG-01.md)                                                       |
| Region               | EU (Italy)                                                                                             |
| Segment              | Mid-cap cement producer; CBAM-covered importer for cement clinker imports                              |
| Use case             | CBAM default-value lookup for clinker; cross-check vs verified actual values                           |
| Target SDK call      | `client.get_factor("urn:gl:factor:cbam-default-values:cbam:cement:eu:2024:v1")` plus partner-side calc |
| Allow-listed sources | `cbam_default_values`, `desnz_ghg_conversion`                                                          |
| Methodology profile  | `cbam_transitional_default`                                                                            |
| Onboarding date      | Target FY27 Q1 W3; gated by Legal MSA + DPA execution                                                  |
| API key issuance     | Operator-issued via Vault; data residency pinned `eu-central-1`                                        |
| Success criteria     | At least 1 successful default-value lookup followed by 1 partner-side calculation within 14 days       |

## Joint Onboarding Calendar

| Phase                       | Target date              | Gate                                                      |
| --------------------------- | ------------------------ | --------------------------------------------------------- |
| MSA / NDA / DPA execution   | FY27 Q1 W1 (IN), W2 (EU) | Legal — see `MSA-NDA-status-tracker.md`                   |
| Tenant provisioning         | T+1 business day         | SRE Lead — Vault key; tenant id + entitlement record      |
| Onboarding kickoff call     | T+2 business days        | Partner Success Lead chairs                               |
| First SDK call by partner   | T+5 business days        | Partner self-serve; DevRel on standby                     |
| First successful calc       | T+14 business days       | Partner-confirmed, captured in partner profile checklist  |
| Mid-pilot review            | T+30 days                | Partner Success Lead + Partner contact                    |
| Feedback memo due           | end of FY27 Q1 (2026-06-30) | Template: `PARTNER_FEEDBACK_MEMO_TEMPLATE.md`         |

## Success Bar

The v0.1 alpha exits the partner-program gate when ALL of the
following are true (per `pilot-success-criteria.md`):

* Both partners have signed MSA + NDA (+ DPA where applicable).
* Both partners have at least one successful SDK calculation in
  their own environment (not in our staging).
* Both partners have filed a written feedback memo (one per
  partner).
* Both partners have logged at least one user-reported defect or
  product gap (the team treats zero defects as "not actually
  using the SDK").
* CTO has reviewed the two memos and signed off on the gap-to-ticket
  triage.

## Risks + Mitigations

| Risk                                           | Mitigation                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| Legal MSA delay slips W1/W2 onboarding         | Compliance/Security Lead drives weekly Legal stand-up               |
| Partner cannot deploy SDK in 14 days            | DevRel pairs 1:1; fall back to TestClient demo if SDK env blocked   |
| CBAM resolver gap (CBAM family stuck at 0%)    | Pre-flag to EU partner; v0.1 surface is direct factor lookup, not resolve; CBAM resolver is v0.5 work |
| API key leak                                   | Vault-only issuance; rotation playbook in `runbooks/`               |

## Cross-References

* Tracker: `DESIGN_PARTNER_TRACKER.md`
* Legal: `MSA-NDA-status-tracker.md`
* Onboarding: `onboarding-checklist.md`
* Success criteria: `pilot-success-criteria.md`
* Feedback template: `PARTNER_FEEDBACK_MEMO_TEMPLATE.md`
