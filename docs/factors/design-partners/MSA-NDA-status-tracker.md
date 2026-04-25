# MSA / NDA / DPA Status Tracker (Operator-facing)

**Owner:** `human:legal@greenlang.io`
**Update cadence:** Legal updates this file on each contract execution event; Factors PM countersigns on review.
**Confidentiality:** This file contains contract *status*, not contract *content*. Real signed PDFs live in the Legal vault, not in the repo.
**Last updated:** 2026-04-25

---

## Status Legend

- `pending_legal_review` — draft circulated, not yet executed
- `in_redline` — partner has returned redlines; under negotiation
- `executed` — fully countersigned, both parties bound
- `not_required` — explicitly waived; rationale in `notes` column
- `expired` — past `expiry_at`; tenant must be deactivated until renewed

---

## Tracker Table

| Partner Slug | Contract | Status | executed_at | executed_by (GL) | executed_by (Partner) | expiry_at | Notes |
|--------------|----------|--------|-------------|------------------|------------------------|-----------|-------|
| `IN-EXPORT-01` | MSA | `pending_legal_review` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `executed_at + 24 months` | India-domiciled textile exporter; standard alpha MSA |
| `IN-EXPORT-01` | NDA | `pending_legal_review` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `executed_at + 36 months` | Mutual NDA; covers EU OEM customer data partner uploads |
| `IN-EXPORT-01` | DPA | `not_required` | n/a | n/a | n/a | n/a | India domicile; DPDP applies, not GDPR. Confirmed by Legal 2026-04-25 (placeholder confirmation, replace at activation) |
| `EU-MFG-01` | MSA | `pending_legal_review` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `executed_at + 24 months` | Italian cement producer; standard alpha MSA |
| `EU-MFG-01` | NDA | `pending_legal_review` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `executed_at + 36 months` | Mutual NDA; covers facility-level activity data |
| `EU-MFG-01` | DPA | `pending_legal_review` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | `# placeholder, replace at activation` | aligned with MSA term | REQUIRED — EU partner; GDPR applies. EU SCCs not required because residency pinned to `eu-central-1` (no third-country transfer under alpha) |

---

## Required Actions Before Tenant Activation

A partner cannot move from `placeholder` to `live` until:

1. MSA status is `executed` for that partner.
2. NDA status is `executed` for that partner.
3. DPA status is `executed` OR `not_required` for that partner.
4. Operator has acknowledged the status row on this file.

If any of (1)–(3) regress to `expired`, Operator must deactivate the tenant within 1 business day per security runbook.
