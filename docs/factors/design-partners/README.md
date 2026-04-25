# GreenLang Factors v0.1 Alpha — Design Partner Program

**Program owner:** GreenLang Factors PM (`product-factors@greenlang.io`)
**Legal owner:** `human:legal@greenlang.io`
**Operator owner:** `human:operator@greenlang.io`
**Status:** Wave C / WS8-T1 — paper trail in repo, partner activation pending Legal sign-off.
**Last updated:** 2026-04-25
**Linked CTO acceptance criterion:** §19.1 — "Two design-partner accounts live with API keys; each has signed MSA + NDA; each has at least one successful SDK-based calculation in their own environment."

---

## 1. Program Overview

Per CTO doc §19.1, GreenLang Factors v0.1 Alpha ships with **2 (two) design-partner tenants**:

| Partner Slug | Profile | Primary Use Case | File |
|--------------|---------|------------------|------|
| `IN-EXPORT-01` | India-linked exporter (textile sector) | India electricity Scope 2 location-based + CBAM exporter view | [`partner-IN-EXPORT-01.md`](./partner-IN-EXPORT-01.md) |
| `EU-MFG-01` | EU-facing manufacturer (Italian cement producer) | CBAM covered-good default lookup for cement clinker | [`partner-EU-MFG-01.md`](./partner-EU-MFG-01.md) |

Both tenant artifacts in this directory are **placeholder records**. They become live tenants only after Legal executes MSA/NDA (and DPA for EU partner), and Operator provisions the actual tenant-id and issues the API key against the production allow-list.

## 2. What Partners Get Under Alpha v0.1

- 1 tenant ID + 1 API key, scoped to the allow-listed sources and methodology profiles in their partner profile.
- Read access to the v0.1 factor catalog (1,491 factors, frozen for the alpha window).
- Read access to provenance records and SHA-256 hashes for every factor lookup.
- SDK access (`pip install greenlang-factors==0.1.0`) with the documented `client.health()`, `client.get_factor(...)`, and `client.calculate(...)` surface.
- One golden e2e test in `tests/factors/v0_1_alpha/` demonstrating their canonical calculation path (written in WS8-T2).
- Direct Slack/email channel to the Factors PM for the 30-day pilot window.
- One quarterly review slot for v0.5 input.

## 3. What Partners Do NOT Get Under Alpha v0.1

- No write access to the catalog (proposals only via the gold-set submission flow).
- No access to other partners' tenant data.
- No SLAs (best-effort support, severity defined in `docs/factors/support_boundaries_and_severity.md`).
- No methodology profiles outside their allow-list.
- No CBAM-family resolver guarantees beyond the 2 covered goods scoped per partner (open structural gap; tracked in MEMORY.md note "CBAM family stuck at 0%").

## 4. Partner Commitments

- Sign MSA + NDA (and DPA if EU-domiciled) before tenant activation.
- Run at least one SDK-based calculation in their own environment within the 30-day pilot window.
- File a written feedback memo using [`feedback-memo-template.md`](./feedback-memo-template.md) by the alpha→beta transition gate.
- Log at least 1 user-reported defect (otherwise we conclude the partner is not actually using the SDK).
- Attend a 30-min quarterly review call for v0.5 input.

## 5. Pilot Success Criteria

See [`pilot-success-criteria.md`](./pilot-success-criteria.md) for the explicit pass/fail bar.

## 6. Onboarding Checklist (Canonical)

See [`onboarding-checklist.md`](./onboarding-checklist.md). Each partner's profile file embeds a per-partner copy of this checklist with `[ ] PENDING` items.

## 7. Legal & Contract Tracking

See [`MSA-NDA-status-tracker.md`](./MSA-NDA-status-tracker.md). Operator-facing; updated by Legal at execution.

## 8. Feedback Memo

Template path: [`feedback-memo-template.md`](./feedback-memo-template.md)
Target submission window: end of FY27 Q1 (2026-06-30).

## 9. Secret Handling

- This directory contains **NO real keys, NO real PII, NO real partner names**.
- The `keys/` subdirectory exists only as a placeholder structure and is `.gitkeep`-only.
- Real API keys are issued via Operator's secrets pipeline (Vault) and never committed.
- API key fields in partner profiles record only the `sha256` first-16-hex prefix at issuance time.

---

**File index in this directory:**

- `README.md` (this file) — index + program overview
- `partner-IN-EXPORT-01.md` — India-linked textile exporter profile
- `partner-EU-MFG-01.md` — EU-facing Italian cement producer profile
- `onboarding-checklist.md` — canonical per-partner checklist template
- `feedback-memo-template.md` — partner-fillable end-of-pilot memo template
- `MSA-NDA-status-tracker.md` — operator-facing legal tracker
- `pilot-success-criteria.md` — pass/fail bar for the alpha pilot
- `keys/.gitkeep` — placeholder for key issuance log structure (no real keys ever)
