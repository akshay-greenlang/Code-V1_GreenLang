# Method Pack — Corporate

Implements **GHG Protocol Corporate Accounting and Reporting Standard** (2004, revised 2015), **Scope 2 Guidance** (2015), and **Scope 3 Calculation Guidance** (2011 + 2013 tools). Covers four of the fourteen method profiles.

| Profile | Scope | Uses |
|---|---|---|
| `corporate_scope1` | 1 | Stationary + mobile combustion, refrigerants, process emissions, fugitive |
| `corporate_scope2_location_based` | 2 | Grid-average location-based electricity, steam, cooling |
| `corporate_scope2_market_based` | 2 | REC / GO / supplier-specific / residual-mix (cascade per Scope 2 QC 1-7) |
| `corporate_scope3` | 3 | All 15 Scope 3 categories (Cat 1 Purchased Goods → Cat 15 Investments) |

---

## Standards alignment

- **GHG Protocol Corporate Standard** (WRI/WBCSD, 2004; revised 2015). [Link](https://ghgprotocol.org/corporate-standard).
- **GHG Protocol Scope 2 Guidance** (2015) — governs `corporate_scope2_*`. Specifically §6.1 (location-based) and §6.2 (market-based); Quality Criteria §7.1-7.7 (QC 1-7) gate certificate acceptance.
- **GHG Protocol Scope 3 Standard** (2011) + **Scope 3 Calculation Tool** (2013) — governs `corporate_scope3`.
- **IPCC 2006 Guidelines for National Greenhouse Gas Inventories** (2019 Refinement) — underlies combustion + fugitive factors.
- **IPCC AR6** (2021) — default `gwp_set`.

Additional regulatory mappings: `CSRD_E1` (EU), `CA_SB253`, `UK_SECR`, `TCFD`, `IFRS_S2`.

---

## Selection rules

- `selection.allowed_families`:
  - Scope 1 → `combustion`, `refrigerants`, `land_removals` (if in-boundary), `waste` (if in-boundary)
  - Scope 2 (LB) → `electricity`
  - Scope 2 (MB) → `electricity` + `residual_mix`
  - Scope 3 → any family (Cat 1-15 dependent)
- `selection.require_verification`: `false` by default; Certified regulatory variants flip to `true`.
- `selection.priority_tiers`:
  Scope 1: `["facility", "country", "region", "global"]`
  Scope 2 LB: `["utility", "grid_subregion", "country", "global"]`
  Scope 2 MB: `["supplier_contracted", "certificate", "utility_mix", "residual_mix"]`
  Scope 3: `["supplier_specific", "activity_physical", "spend_proxy", "industry_average"]`

---

## Boundary

- `boundary.allowed_scopes`: `["1"]`, `["2"]`, or `["3"]` (one per profile); for Scope 3, the 15 categories are enumerated in `boundary.scope3_categories`.
- `boundary.system_boundary`: `cradle_to_gate` (default for Scope 3), `gate_to_gate` (operational Scope 1/2).
- `boundary.include_transmission_losses`: `false` for location-based Scope 2 (busbar basis); `null` otherwise.

---

## Gas-to-CO2e

- `gas_to_co2e.default_gwp_set`: `IPCC_AR6_100`.
- `gas_to_co2e.allowed_override_sets`: `["IPCC_AR5_100", "IPCC_AR4_100", "IPCC_AR6_20"]`.
- `gas_to_co2e.metric`: `GWP`. `GWP*` not enabled for Corporate (methane-heavy packs only).

---

## Biogenic treatment

`separate_report`. Biogenic CO2 is calculated and surfaced in a separate block on the resolved factor; the headline `co2e_kg` excludes biogenic CO2 per GHG Protocol Scope 2 and Scope 3 conventions.

---

## Market-instrument treatment (Scope 2 MB only)

`market_instruments.treatment`: `require_certificate` for REC/GO/I-REC; `allowed` for PPAs; `allowed` for residual mix fallback when no certificate is held.

Quality Criteria 1-7 applied:

1. Contract information conveys the energy generation attribute
2. Attributes are claimed and retired on behalf of the reporter
3. Attributes are tracked and redeemed / retired / cancelled
4. Issued and redeemed as close in time as possible to the energy generation (15 months)
5. Generator is operational within the same market as the reporter
6. Issued in a market with residual-mix calculation (to avoid double counting)
7. Acquired by a single party claim

Residual-mix sources: see [`method-packs/electricity.md`](electricity.md).

---

## Fallback

`fallback.cannot_resolve_action = raise_no_safe_match`. `fallback.global_default_tier_allowed = true` for exploratory Scope 3 only; `false` for Scope 1/2 regulated disclosures.

---

## Audit text

Templates at `greenlang/factors/method_packs/audit_texts/corporate_*.j2`. Every template's frontmatter carries `standard_citation` (e.g., `"GHG Protocol Corporate Standard §4.1"`). Unapproved templates render with a `[Draft — Methodology Review Required]` banner per [`docs/specs/audit_text_template_policy.md`](../../specs/audit_text_template_policy.md).

---

## Related

- [API `/resolve`](../api-reference/resolve.md), [`/method-packs`](../api-reference/method-packs.md).
- [`concepts/method_pack.md`](../concepts/method_pack.md), [electricity pack](electricity.md).
- Method pack spec: [`docs/specs/method_pack_template.md`](../../specs/method_pack_template.md).
