# GreenLang Factors v0.1 Alpha — Design-Partner Pilot Success Criteria

**Linked CTO acceptance:** §19.1 — "Two design-partner accounts live with API keys; each has signed MSA + NDA; each has at least one successful SDK-based calculation in their own environment." Plus: "Two design partners have completed at least one calculation flow using the SDK and have signed off on usability (brief written feedback memo; not a vendor NPS)."

**Pilot window:** 30 days from the second of the two partners' tenant activation (the clock starts when both tenants are live, not when the first one is live).

**Decision body:** GreenLang Factors PM (chair), Engineering Lead, Legal Lead. Unanimous PASS required to advance from alpha to beta.

---

## Pass / Fail Bar

| # | Criterion | Pass Threshold | Source of Truth |
|---|-----------|----------------|-----------------|
| 1 | Each partner completes **at least 1 successful SDK-based calculation in their own environment** | Both partners: `>= 1` end-to-end calculation in their own infra (not in GL-hosted demo) | Partner-side log + GL-side audit trail with matching `request_id` |
| 2 | Each partner files a **feedback memo** before the alpha→beta transition gate | Both partners: memo committed under `docs/factors/design-partners/feedback/` and counter-signed by PM | Repo file presence + sign-off table in memo |
| 3 | Average partner SDK uptime **>= 99%** over the 30-day pilot window | Per-tenant uptime measured against `client.health()` and `client.get_factor()` synthetic probes | Prometheus tenant-availability metric (alpha SLO dashboard) |
| 4 | At least **1 user-reported defect logged per partner** | Both partners: `>= 1` issue filed in the alpha defect tracker. Zero defects = zero usage = automatic FAIL on this criterion. | Issue tracker (Linear/GitHub Issues alpha label) |
| 5 | Both partners agree to participate in a **30-min quarterly review for v0.5** | Both partners: scheduled calendar invite OR written confirmation | Calendar invite + email confirmation in CRM |

---

## Failure Modes & Remediation

- **Criterion 1 fails:** the partner did not put the SDK in their own environment. Cause is almost always Legal blocking or Operator provisioning lag, not product. Extend pilot by 30 days; do not transition to beta.
- **Criterion 2 fails:** memo missing. Block alpha→beta transition until memo received; do not silently advance.
- **Criterion 3 fails:** uptime below 99%. Triage on-call incidents; if root cause is GL-side, write postmortem and address before beta. If root cause is partner network, document and exempt.
- **Criterion 4 fails (zero defects):** treat as a red flag, not a green one. Schedule 1:1 call with partner technical lead to confirm actual usage; if usage is in fact zero, restart the pilot for that partner with hands-on enablement.
- **Criterion 5 fails:** reduce v0.5 scope; we lose roadmap input from a paying-attention pilot user.

---

## Out-of-Scope for the Alpha Pilot

The following are **not** alpha pilot pass/fail criteria; they apply at GA:

- Production SLAs (best-effort support only during alpha)
- Multi-region failover
- Partner-billable usage (alpha is free)
- Full CBAM-family resolver coverage (acknowledged structural gap; cement default-lookup path is in scope, embedded-emissions resolution is not)
- Vendor NPS scores (explicitly excluded; we want a written memo, not a number)
