# Design Partner Profile — `EU-MFG-01`

**Status:** `placeholder, pending_legal_review` — replace at activation
**Linked CTO acceptance:** §19.1 (one EU-facing manufacturer)
**Last updated:** 2026-04-25

---

## 1. Partner Identity

| Field | Value |
|-------|-------|
| Partner slug | `EU-MFG-01` |
| Partner type | EU-facing manufacturer |
| Sector chosen | **Italian cement producer** (clinker + finished cement, sold within EU + small EU-export volumes) |
| Sector rationale | Cement is a CBAM Annex I covered good with explicit `eu-cbam-defaults` published values for clinker, which gives us a clean alpha demo of the CBAM default-lookup path without depending on the still-broken CBAM-family resolver behavior (we are exercising the *covered-good default lookup*, not full embedded-emissions resolution). Picking Italian cement (rather than German chemicals) lets us hit the CBAM cement clinker default factor explicitly listed in `eu-cbam-defaults`. |
| Domicile | Italy / EU (placeholder) |
| Public-facing partner name | `# placeholder, replace at activation` |
| Partner technical contact | `# placeholder, replace at activation` |
| Partner business contact | `# placeholder, replace at activation` |

## 2. Tenant Provisioning

| Field | Value |
|-------|-------|
| Account ID (UUIDv4) | `7c2e9d18-4a3f-4b6e-8c1a-5d2f9e8b7a3d`  `# placeholder, replace at activation` |
| Tenant slug | `tenant-eu-mfg-01`  `# placeholder, replace at activation` |
| Region pinning | `eu-central-1` (Frankfurt) — GDPR data-residency requirement |
| Provisioned at | `# placeholder, replace at activation` (ISO-8601 UTC) |
| Provisioned by | `human:operator@greenlang.io` |

## 3. API Key Issuance Record

> **No real keys ever live in this repo.** Only the SHA-256 first-16-hex prefix is recorded.

| Field | Value |
|-------|-------|
| Key label | `gl_alpha_eu_mfg_01__{sha256_first_16}__placeholder` |
| Key SHA-256 prefix (16 hex) | `0000000000000000`  `# placeholder, replace at activation` |
| Issued at | `# placeholder, replace at activation` |
| Issued by | `human:operator@greenlang.io` |
| Expiry | issued_at + 90 days (alpha window) |
| Rotation policy | one rotation at the alpha→beta transition gate |
| Storage | Vault path `secret/factors/alpha/eu-mfg-01` (production); never committed to repo |

## 4. Allow-listed Sources

The partner's tenant policy allows lookups against **only** the following 4 sources during the alpha:

- `ipcc-ar6` (gases, GWP factors)
- `defra-2025` (UK/EU defaults — DEFRA's 2025 emission factor set, broadly used in EU corporate disclosure)
- `epa-ghg-hub` (US EPA — Scope 1 stationary combustion factors used by EU manufacturer's US sister plant)
- `eu-cbam-defaults` (EU CBAM Annex I covered-good default values; this is the headline source for the partner)

All other sources return `403 source_not_allowlisted`.

## 5. Allow-listed Methodology Profiles

- `ghgp-corporate-scope1`
- `ghgp-corporate-scope2-market`
- `eu-cbam-default`

All other profiles return `403 profile_not_allowlisted`.

## 6. Legal Status

| Field | Value |
|-------|-------|
| MSA status | `pending_legal_review` |
| NDA status | `pending_legal_review` |
| DPA status | `pending_legal_review` (REQUIRED — partner is EU-domiciled, GDPR applies) |
| Legal lead | `human:legal@greenlang.io` |
| Target executed by | `2026-05-15` |
| Notes | EU/Italy-domiciled. DPA is mandatory because the partner uploads facility-level activity data which may contain operator personal data (named site managers). EU SCCs not required since data residency is `eu-central-1` and no cross-border transfer occurs under the alpha. |

See [`MSA-NDA-status-tracker.md`](./MSA-NDA-status-tracker.md) for the canonical status row.

## 7. Expected SDK Calculation

**Canonical calc:** EU CBAM covered-good default lookup for **cement clinker** (CN code 2523 10 00), used by the partner to validate the platform's published default value against their internal calculated value.

**Pseudocode:**

```python
from greenlang_factors import Client

client = Client(api_key="gl_alpha_eu_mfg_01__...")  # placeholder
client.health()

factor = client.get_factor(
    activity="cbam_covered_good_default",
    cn_code="25231000",
    good="cement_clinker",
    source="eu-cbam-defaults",
    methodology_profile="eu-cbam-default",
    vintage_year=2026,
)

result = client.calculate(
    activity_amount=10_000,            # tonnes clinker
    activity_unit="t",
    factor=factor,
)
# expected output: embedded emissions in tCO2e, plus provenance bundle with SHA-256
```

**Expected golden-test path:** `tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py` (delivered Wave E / TaskCreate #22 / WS8-T2, 2026-04-25).

**Expected provenance bundle:** must include `factor_id`, `source=eu-cbam-defaults`, `methodology_profile=eu-cbam-default`, `cn_code=25231000`, `vintage_year=2026`, `sha256` of the resolved factor record. NOTE: this exercises the CBAM **default-lookup** path, not the broken CBAM-family resolver path; that's intentional.

### Reference golden test

| Field | Value |
|-------|-------|
| Test file | [`tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py`](../../../tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py) |
| Shared helpers | [`tests/factors/v0_1_alpha/_partner_helpers.py`](../../../tests/factors/v0_1_alpha/_partner_helpers.py) |
| Resolved factor URN | `urn:gl:factor:cbam-default-values:CBAM:cement:CN:2024:v1` (CBAM Annex IV cement-sector rollup, China origin row) |
| Factor value | `0.84` `kgCO2e/kg-product` (vintage 2024-01-01 ..) |
| Activity input | `10,000 tonnes` (= `10,000,000 kg`) of CBAM cement-class import |
| Computed embedded emissions | `8,400,000 kgCO2e` (= 10,000,000 × 0.84) |
| Test count | 4 tests (`pytest -v` green; 2026-04-25) |
| Pytest marker | `@pytest.mark.alpha_v0_1_acceptance` (CTO §19.1 ship-readiness gate) |
| Catalog v0.1 caveat | The v0.1 alpha catalog seed carries CBAM Annex IV defaults at the *sector* rollup granularity (one row per sector × country-of-origin) rather than the per-CN-code row level. The "cement" rollup is the closest match to the partner's "cement clinker (CN 25231000) default" ask. The exact CN-code-level breakdown is scheduled for v0.2 catalog expansion. |

## 8. Onboarding Checklist (per-partner instance)

> Canonical template: [`onboarding-checklist.md`](./onboarding-checklist.md)

- [ ] PENDING — Partner kickoff call held; record SOW
- [ ] PENDING — MSA executed (Legal)
- [ ] PENDING — NDA executed (Legal)
- [ ] PENDING — DPA executed (Legal — REQUIRED for EU partner)
- [ ] PENDING — Tenant ID provisioned
- [ ] PENDING — API key issued; hash recorded
- [ ] PENDING — Allow-listed sources scoped
- [ ] PENDING — Allow-listed methodology profiles scoped
- [ ] PENDING — Partner SDK environment installed (`pip install greenlang-factors==0.1.0`)
- [ ] PENDING — Partner first health check (`client.health()`)
- [ ] PENDING — Partner first factor lookup (`client.get_factor(...)`)
- [x] DONE (2026-04-25) — Partner first end-to-end calculation flow validated; the SDK round-trip for the CBAM cement-sector default factor (URN `urn:gl:factor:cbam-default-values:CBAM:cement:CN:2024:v1`) is locked by the golden test at [`tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py`](../../../tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py).
- [ ] PENDING — Feedback memo received
- [ ] PENDING — Quarterly review scheduled

## 9. Feedback Memo

| Field | Value |
|-------|-------|
| Template | [`feedback-memo-template.md`](./feedback-memo-template.md) |
| Target submission | `2026-06-30` (end of FY27 Q1) |
| Memo received at | `# placeholder, replace at activation` |
| Memo file path (post-receipt) | `docs/factors/design-partners/feedback/EU-MFG-01-memo-2026Q1.md` (to be created when memo arrives) |
