# Design Partner Profile — `IN-EXPORT-01`

**Status:** `placeholder, pending_legal_review` — replace at activation
**Linked CTO acceptance:** §19.1 (one India-linked exporter)
**Last updated:** 2026-04-25

---

## 1. Partner Identity

| Field | Value |
|-------|-------|
| Partner slug | `IN-EXPORT-01` |
| Partner type | India-linked exporter |
| Sector chosen | **Textile exporter** (cotton + man-made fibre fabric) selling into EU OEM apparel supply chains |
| Sector rationale | Textile exporters are the highest-volume India→EU trade flow with embedded Scope 2 grid emissions; CBAM does NOT yet cover textiles, but EU OEM customers are demanding the upstream Scope 2 attribution today, which exercises the `india-cea-baseline` source + `ghgp-corporate-scope2-location` profile end-to-end. Picking textile (rather than steel or chemicals) avoids the CBAM-family resolver gap noted in MEMORY.md and gives us a clean alpha success demo. |
| Domicile | India (placeholder) |
| Public-facing partner name | `# placeholder, replace at activation` |
| Partner technical contact | `# placeholder, replace at activation` |
| Partner business contact | `# placeholder, replace at activation` |

## 2. Tenant Provisioning

| Field | Value |
|-------|-------|
| Account ID (UUIDv4) | `b3f4a5c2-1d8e-4a9c-9f2b-7e6d3a4b5c1f`  `# placeholder, replace at activation` |
| Tenant slug | `tenant-in-export-01`  `# placeholder, replace at activation` |
| Region pinning | `ap-south-1` (Mumbai) — partner data residency preference |
| Provisioned at | `# placeholder, replace at activation` (ISO-8601 UTC) |
| Provisioned by | `human:operator@greenlang.io` |

## 3. API Key Issuance Record

> **No real keys ever live in this repo.** Only the SHA-256 first-16-hex prefix is recorded.

| Field | Value |
|-------|-------|
| Key label | `gl_alpha_in_export_01__{sha256_first_16}__placeholder` |
| Key SHA-256 prefix (16 hex) | `0000000000000000`  `# placeholder, replace at activation` |
| Issued at | `# placeholder, replace at activation` |
| Issued by | `human:operator@greenlang.io` |
| Expiry | issued_at + 90 days (alpha window) |
| Rotation policy | one rotation at the alpha→beta transition gate |
| Storage | Vault path `secret/factors/alpha/in-export-01` (production); never committed to repo |

## 4. Allow-listed Sources

The partner's tenant policy allows lookups against **only** the following 3 sources during the alpha:

- `ipcc-ar6` (gases, GWP factors)
- `india-cea-baseline` (India CEA grid emissions baseline; needed for Scope 2 location-based)
- `eu-cbam-defaults` (read-only EU CBAM exporter default values; for the partner's EU OEM customer ask)

All other sources return `403 source_not_allowlisted`.

## 5. Allow-listed Methodology Profiles

- `ghgp-corporate-scope1`
- `ghgp-corporate-scope2-location`
- `eu-cbam-default`

All other profiles return `403 profile_not_allowlisted`.

## 6. Legal Status

| Field | Value |
|-------|-------|
| MSA status | `pending_legal_review` |
| NDA status | `pending_legal_review` |
| DPA status | `not_required` (partner is India-domiciled; Indian DPDP applies, not GDPR) |
| Legal lead | `human:legal@greenlang.io` |
| Target executed by | `2026-05-15` |
| Notes | India-domiciled exporter; DPDP review by Legal required. EU OEM data shared by partner is treated as partner-confidential, not partner-PII. |

See [`MSA-NDA-status-tracker.md`](./MSA-NDA-status-tracker.md) for the canonical status row.

## 7. Expected SDK Calculation

**Canonical calc:** India electricity Scope 2 location-based emissions for FY2026, used by the partner to attribute embedded Scope 2 of fabric production to one EU OEM customer's purchased-goods footprint.

**Pseudocode:**

```python
from greenlang_factors import Client

client = Client(api_key="gl_alpha_in_export_01__...")  # placeholder
client.health()

factor = client.get_factor(
    activity="electricity_consumption",
    region="IN",
    source="india-cea-baseline",
    methodology_profile="ghgp-corporate-scope2-location",
    vintage_year=2026,
)

result = client.calculate(
    activity_amount=1_250_000,         # kWh
    activity_unit="kWh",
    factor=factor,
)
# expected output: emissions in kgCO2e, plus provenance bundle with SHA-256
```

**Expected golden-test path:** `tests/factors/v0_1_alpha/test_partner_in_export_calc.py` (delivered Wave E / TaskCreate #22 / WS8-T2, 2026-04-25).

**Expected provenance bundle:** must include `factor_id`, `source=india-cea-baseline`, `methodology_profile=ghgp-corporate-scope2-location`, `vintage_year=2026`, `sha256` of the resolved factor record, and `resolver_path` ending in a non-CBAM-family resolver.

### Reference golden test

| Field | Value |
|-------|-------|
| Test file | [`tests/factors/v0_1_alpha/test_partner_in_export_calc.py`](../../../tests/factors/v0_1_alpha/test_partner_in_export_calc.py) |
| Shared helpers | [`tests/factors/v0_1_alpha/_partner_helpers.py`](../../../tests/factors/v0_1_alpha/_partner_helpers.py) |
| Resolved factor URN | `urn:gl:factor:india-cea-co2-baseline:IN:all_india:2025-26:cea-v22.0:v1` (all-India composite grid, FY2025-26) |
| Factor value | `0.68` `kgCO2e/kWh` (CEA v22.0, vintage 2025-04-01 .. 2026-03-31) |
| Activity input | `1,200,000 kWh` (FY2026 textile factory consumption) |
| Computed Scope 2 location-based emissions | `816,000 kgCO2e` (= 1,200,000 × 0.68) |
| Test count | 4 tests (`pytest -v` green; 2026-04-25) |
| Pytest marker | `@pytest.mark.alpha_v0_1_acceptance` (CTO §19.1 ship-readiness gate) |

## 8. Onboarding Checklist (per-partner instance)

> Canonical template: [`onboarding-checklist.md`](./onboarding-checklist.md)

- [ ] PENDING — Partner kickoff call held; record SOW
- [ ] PENDING — MSA executed (Legal)
- [ ] PENDING — NDA executed (Legal)
- [ ] PENDING — DPA executed if EU partner (N/A for `IN-EXPORT-01`)
- [ ] PENDING — Tenant ID provisioned
- [ ] PENDING — API key issued; hash recorded
- [ ] PENDING — Allow-listed sources scoped
- [ ] PENDING — Allow-listed methodology profiles scoped
- [ ] PENDING — Partner SDK environment installed (`pip install greenlang-factors==0.1.0`)
- [ ] PENDING — Partner first health check (`client.health()`)
- [ ] PENDING — Partner first factor lookup (`client.get_factor(...)`)
- [x] DONE (2026-04-25) — Partner first end-to-end calculation flow validated; the SDK round-trip for the FY2026 all-India CEA Scope 2 location-based factor (URN `urn:gl:factor:india-cea-co2-baseline:IN:all_india:2025-26:cea-v22.0:v1`) is locked by the golden test at [`tests/factors/v0_1_alpha/test_partner_in_export_calc.py`](../../../tests/factors/v0_1_alpha/test_partner_in_export_calc.py).
- [ ] PENDING — Feedback memo received
- [ ] PENDING — Quarterly review scheduled

## 9. Feedback Memo

| Field | Value |
|-------|-------|
| Template | [`feedback-memo-template.md`](./feedback-memo-template.md) |
| Target submission | `2026-06-30` (end of FY27 Q1) |
| Memo received at | `# placeholder, replace at activation` |
| Memo file path (post-receipt) | `docs/factors/design-partners/feedback/IN-EXPORT-01-memo-2026Q1.md` (to be created when memo arrives) |
