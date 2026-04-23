# Method Pack — Finance (PCAF)

Implements the **Partnership for Carbon Accounting Financials (PCAF) Global GHG Accounting and Reporting Standard for the Financial Industry**. Profile: `finance_proxies_pcaf`.

---

## Standards alignment

- **PCAF Global GHG Accounting and Reporting Standard Part A (v2.0, 2022)** — financed emissions (6 asset classes).
- **PCAF Part B (2024)** — facilitated emissions (capital markets).
- **PCAF Part C** — insurance-associated emissions.
- **GHG Protocol Scope 3 Category 15 (Investments)** — underlying standard for corporate reporters.

Additional alignments: **TCFD**, **ISSB IFRS S2**, **ESRS E1** (Section AR 32-38 on financed emissions).

---

## Covered asset classes (Part A)

| PCAF `asset_class` | Notes |
|---|---|
| `listed_equity_and_corporate_bonds` | Enterprise value including cash (EVIC) attribution. |
| `business_loans_and_unlisted_equity` | EVIC or balance-sheet attribution. |
| `project_finance` | Project-level attribution by outstanding balance. |
| `commercial_real_estate` | Property-level building operational emissions. |
| `mortgages` | Residential property operational emissions. |
| `motor_vehicle_loans` | Vehicle tailpipe + upstream fuel. |

Part B adds `facilitated_emissions` (underwriting, advisory).

---

## Covered factor family

`finance_proxies`.

---

## Parameters

Every PCAF factor carries:

- `asset_class` — one of the six PCAF Part A asset classes (or Part B for facilitated).
- `sector_code` — NAICS / ISIC / GICS classification of the investee.
- `intensity_basis` — `revenue`, `asset`, `ebitda`, `employee`.
- `geography` — ISO-3166 country of the investee (or `XX` global for MNC averages).
- `proxy_confidence_class` — PCAF's own Data Quality Score: `score_1` (reported verified) through `score_5` (proxy with low confidence).

---

## Selection rules

- `selection.allowed_families`: `["finance_proxies"]`.
- `selection.jurisdiction_hierarchy`: `["country", "region", "global"]`.
- `selection.priority_tiers`:
  `["investee_reported_verified", "investee_reported_unverified", "industry_average_geo", "industry_average_global", "spend_proxy"]`.
  Maps to PCAF Score 1 through Score 5.

---

## Attribution formula

Financed emissions per PCAF Part A §4:

```
financed_emissions_i = sum_over_portfolio( attribution_factor_c * scope1_c + attribution_factor_c * scope2_c + attribution_factor_c * scope3_c )
```

Where `attribution_factor_c = outstanding_amount / EVIC` for listed equity and corporate bonds.

The pack does NOT compute the attribution itself — the caller passes `outstanding_amount`, `evic` or `total_equity_plus_debt`, and `sector_code`; the resolver returns the emission-per-unit-financial intensity factor. The caller multiplies.

---

## Boundary

- `boundary.system_boundary`: mirrors the investee's scope. For listed equity / business loans: Scope 1 + Scope 2 + Scope 3 of the investee.
- `boundary.allowed_scopes`: depends on PCAF asset class; Scope 3 for certain sectors (energy, automotive, steel, cement) is mandatory per PCAF §5.4.

---

## Data quality score alignment

PCAF's Data Quality Score (1-5 scale, 1 best) maps to GreenLang's FQS `verification_score` and `technology_score`:

| PCAF score | `verification_score` | `technology_score` |
|---|:--:|:--:|
| Score 1 (reported verified) | 5 | 5 |
| Score 2 (reported unverified) | 3 | 4 |
| Score 3 (physical activity proxy) | 3 | 3 |
| Score 4 (economic activity proxy, sector-geo) | 2 | 2 |
| Score 5 (economic activity proxy, sector only) | 1 | 1 |

Composite FQS follows from the weighted formula — see [`concepts/quality_score.md`](../concepts/quality_score.md).

---

## Sources

- **PCAF Global Standard v2 + Part B** — methodology + default sector factors. License class `licensed_embedded` (PCAF Attribution terms).
- **US EPA SUSEEIO v2** — US sector factors (`open`).
- **EXIOBASE v3** — multi-regional EEIO (BYO at v1).
- **CEDA / PBE** — EEIO alternative (BYO at v1).
- **Investee Scope 1/2/3 disclosures** — customer-uploaded primary data (`customer_private`).

See [`docs/legal/source_rights_matrix.md`](../../legal/source_rights_matrix.md).

---

## Fallback

`fallback.cannot_resolve_action = raise_no_safe_match`. `fallback.global_default_tier_allowed = true` for exploratory portfolio screening; `false` for Certified disclosures (which require Score <= 3 at minimum for regulated markets).

---

## Related

- [`/resolve`](../api-reference/resolve.md), [`concepts/method_pack.md`](../concepts/method_pack.md).
- [`concepts/quality_score.md`](../concepts/quality_score.md).
