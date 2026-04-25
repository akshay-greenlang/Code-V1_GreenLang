# Design Partner Onboarding Checklist (Canonical Template)

**Use:** Copy this list into each partner profile and track per-partner status.
**Owner:** GreenLang Factors PM, with Legal + Operator co-signers as noted.
**Last updated:** 2026-04-25

---

## Pre-activation (Legal + Sales)

- [ ] Partner kickoff call held; record SOW
  - Owner: Factors PM
  - Artifact: SOW PDF stored in CRM
- [ ] MSA executed (Legal)
  - Owner: `human:legal@greenlang.io`
  - Artifact: signed MSA in legal vault
- [ ] NDA executed (Legal)
  - Owner: `human:legal@greenlang.io`
  - Artifact: signed NDA in legal vault
- [ ] DPA executed if EU partner (Legal)
  - Owner: `human:legal@greenlang.io`
  - Required when partner is EU-domiciled OR transfers EU personal data
  - Artifact: signed DPA + EU SCCs (if applicable) in legal vault

## Tenant Provisioning (Operator)

- [ ] Tenant ID provisioned
  - Owner: `human:operator@greenlang.io`
  - Artifact: tenant row in `gl_factors_tenants` table; UUIDv4 recorded in partner profile
- [ ] API key issued; hash recorded
  - Owner: `human:operator@greenlang.io`
  - Artifact: SHA-256 first-16-hex prefix recorded in partner profile; full key in Vault `secret/factors/alpha/<partner-slug>` only
- [ ] Allow-listed sources scoped
  - Owner: `human:operator@greenlang.io`
  - Artifact: tenant policy row in `gl_factors_tenant_source_allowlist`
- [ ] Allow-listed methodology profiles scoped
  - Owner: `human:operator@greenlang.io`
  - Artifact: tenant policy row in `gl_factors_tenant_profile_allowlist`

## Partner Activation (Partner-side)

- [ ] Partner SDK environment installed (`pip install greenlang-factors==0.1.0`)
  - Owner: Partner technical contact
  - Verification: partner reports successful pip install in their dev/QA env
- [ ] Partner first health check (`client.health()`)
  - Owner: Partner technical contact
  - Verification: HTTP 200 + tenant_id echo in response
- [ ] Partner first factor lookup (`client.get_factor(...)`)
  - Owner: Partner technical contact
  - Verification: lookup against an allow-listed source returns a factor with `sha256` provenance
- [ ] Partner first end-to-end calculation flow validated
  - Owner: Partner technical contact + Factors PM (joint sign-off)
  - Verification: calc result matches the partner's expected order-of-magnitude; provenance bundle present

## Pilot Wrap-up

- [ ] Feedback memo received
  - Owner: Partner business contact
  - Artifact: filed memo in `docs/factors/design-partners/feedback/<partner-slug>-memo-<quarter>.md`
- [ ] Quarterly review scheduled
  - Owner: Factors PM
  - Artifact: calendar invite for v0.5 input review (30 min)
