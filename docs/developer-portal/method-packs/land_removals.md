# Method Pack — Land & Removals

Implements the **GHG Protocol Land Sector and Removals Guidance** (LSR). Profile: `land_removals_ghgp_lsr`.

---

## Standards alignment

- **GHG Protocol Land Sector and Removals Guidance** (WRI/WBCSD, 2024). Covers biogenic emissions, land-use change, soil carbon, forest carbon, and technological removals. [Link](https://ghgprotocol.org/land-sector-and-removals-guidance).
- **IPCC 2006 Guidelines Volume 4 (AFOLU)** with 2019 Refinement — underlying Tier 1 defaults.
- **Science Based Targets Initiative (SBTi) FLAG** — forest, land, and agriculture guidance (for sector-specific target alignment).

---

## Covered factor families

- `land_removals` — primary family.
- Subset of `combustion` (biomass combustion) and `materials_products` (forest products) when used in LSR context.

---

## Parameters

Every land-removals factor carries:

- `land_use_category` — `forest_land`, `cropland`, `grassland`, `wetlands`, `settlements`, `other`.
- `sequestration_basis` — `stock_change`, `flux`, `gain_loss`, `tier1_default`, `tier3_model`.
- `permanence_class` — `short_term` (<5y), `medium_term` (5-40y), `long_term` (40-100y), `permanent` (>100y).
- `reversal_risk_flag` — boolean; triggers explicit uncertainty and monitoring requirements.
- `biogenic_accounting_treatment` — `zero_rated`, `separate_reporting`, `included_in_co2e`.

---

## Selection rules

- `selection.allowed_families`: `["land_removals", "combustion"]`.
- `selection.jurisdiction_hierarchy`: `["region", "country", "global"]`.
- `selection.priority_tiers`: `["tier3_project_specific", "tier2_country_specific", "tier1_ipcc_default"]`.

Most LSR factors are `tier1_ipcc_default`; Certified packs require `tier3_project_specific` for project-level removals claims.

---

## Boundary

- `boundary.system_boundary`: `cradle_to_grave` (for harvested wood products) or custom land boundary defined by project.
- `boundary.allowed_scopes`: Scope 1 (direct land-use emissions) and Scope 3 Cat 4 / Cat 12 (upstream transport and end-of-life of harvested wood products).

---

## Biogenic treatment

`separate_report`. Biogenic CO2 emissions and removals are reported **separately** from the gross inventory per GHG Protocol LSR §3.5. The pack never rolls biogenic CO2 into the headline `co2e_kg`.

Two additional disclosure blocks are emitted:

- `biogenic_emissions` — CO2 from biomass combustion, land conversion, soil carbon loss.
- `biogenic_removals` — CO2 sequestered in living biomass, soils, harvested wood products.

The **net** LSR position is `biogenic_removals - biogenic_emissions` — reported separately from Scope 1.

---

## GWP treatment for methane (LSR-specific)

- `gas_to_co2e.metric`: `GWP` (default) or `GWP*` (opt-in for agriculture / wetland methane where the pulse-vs-stock distinction is material per IPCC AR6 Ch.7 SM §7.SM.2.4).
- `gas_to_co2e.gwp_star_allowed`: `true` (methane-heavy pack opt-in).

---

## Reversal risk and permanence

Factors with `reversal_risk_flag = true` (e.g., forest carbon projects with wildfire exposure) require:

- Buffer pool allocation per project methodology (VCS, Gold Standard, etc.).
- Monitoring cadence per `permanence_class` (short: annual; long/permanent: 5-year MRV cycle).

The pack emits a `buffer_pool_required` label and surfaces the buffer percentage in `assumptions[]`.

---

## Licensing

LSR factors inherit WRI/WBCSD terms (`licensed_embedded`). At v1, Tier 1 IPCC defaults ship via `open`-class IPCC sources (pending IPCC copyright confirmation; see [`docs/legal/source_rights_matrix.md`](../../legal/source_rights_matrix.md)). Project-specific Tier 3 factors are always `customer_private` tenant uploads.

---

## Fallback

`fallback.cannot_resolve_action = raise_no_safe_match`. `fallback.global_default_tier_allowed = true` for exploratory use, `false` for Certified regulated packs.

---

## Related

- [`/resolve`](../api-reference/resolve.md), [`concepts/method_pack.md`](../concepts/method_pack.md).
- [Corporate pack](corporate.md) — Scope 1 interactions with direct land-use emissions.
