# Method Pack — EU Policy (CBAM + DPP)

Implements EU regulatory methodology profiles for the **Carbon Border Adjustment Mechanism** (CBAM) and the **Digital Product Passport** (DPP) regulations. Three profiles.

| Profile | Scope | Regulation |
|---|---|---|
| `eu_cbam` | Embedded emissions in imported goods | Regulation (EU) 2023/956 |
| `eu_dpp_battery` | Battery carbon footprint | Regulation (EU) 2023/1542 |
| `eu_dpp_textile` | Textile carbon footprint | EU DPP Textile Delegated Act |

---

## Standards alignment

### CBAM

- **Regulation (EU) 2023/956** — Carbon Border Adjustment Mechanism. Annex I defines in-scope goods (iron & steel, aluminium, cement, fertilizers, electricity, hydrogen). [Link](https://eur-lex.europa.eu/eli/reg/2023/956/oj).
- **Implementing Regulation (EU) 2023/1773** — CBAM reporting obligations during the transitional period (Oct 2023 - Dec 2025) and from 1 Jan 2026.
- **EU Commission default values (DG TAXUD)** — installation-specific default emission factors for iron/steel, aluminium, cement, fertilizer, electricity, hydrogen.

### Battery DPP

- **Regulation (EU) 2023/1542** — Batteries Regulation. Article 7 (carbon footprint), Annex II (calculation methodology). [Link](https://eur-lex.europa.eu/eli/reg/2023/1542/oj).
- **PEFCR Rechargeable Batteries** — Product Environmental Footprint Category Rules for rechargeable batteries (JRC).

### Textile DPP

- **ESPR — Ecodesign for Sustainable Products Regulation** (EU) 2024/1781.
- **PEFCR Apparel and Footwear** (JRC, draft v1.1) — applicable PEF category rules.

---

## CBAM — `eu_cbam`

### Covered goods (Annex I)

| CN chapter | Goods |
|---|---|
| 72 | Iron and steel |
| 76 | Aluminium |
| 25 | Cement |
| 31 | Fertilizers |
| 27 | Electricity |
| 28 | Hydrogen |

Selection is driven by `activity_schema.classification_codes[]` containing a `CN:<code>` matching an Annex I entry. Non-Annex-I goods return `FactorCannotResolveSafelyError` under `eu_cbam`.

### Selection

- `selection.allowed_families`: `["combustion", "electricity", "materials_products"]`.
- `selection.allowed_statuses`: `["active"]` only (no preview allowed for regulatory filing).
- `selection.require_verification`: `true` from 2026-01-01 onwards (definitive period).
- `selection.priority_tiers`:
  `["installation_verified", "installation_reported", "country_sector_default", "eu_default"]`.

### Boundary

- `boundary.system_boundary`: `cradle_to_gate`.
- `boundary.include_indirect`: `true` for iron/steel, aluminium, hydrogen; `false` for cement, fertilizer (per Annex III).

### Default values

When installation-specific data is unavailable, the resolver falls through to DG TAXUD default values. The Commission publishes annual updates; each default is tagged `fallback_rank = 6` (method_pack_default).

### Carve-out

DG TAXUD published tables are stored as `open`-class (pending Legal review of EU Commission copyright terms under Decision 2011/833/EU). See [`docs/legal/source_rights_matrix.md`](../../legal/source_rights_matrix.md) row `eu_cbam`.

---

## Battery DPP — `eu_dpp_battery`

Supported battery categories (Annex I of Reg. 2023/1542):

- LMT batteries (light means of transport)
- EV batteries
- Industrial batteries >2 kWh
- SLI starter / lighting / ignition batteries

### Selection

- `selection.allowed_families`: `["materials_products", "combustion", "electricity"]`.
- `boundary.system_boundary`: `cradle_to_grave` (production + use + end-of-life, per Article 7 Annex II).
- `boundary.functional_unit`: `"1 kWh of total energy provided over service life"`.

Primary data required for manufacturing phase (Tier 1 per PEFCR); allocation by mass or economic per Annex II §5.

---

## Textile DPP — `eu_dpp_textile`

Selection follows **PEFCR Apparel and Footwear** category rules. `boundary.functional_unit`: `"1 garment wear (use-phase normalized)"`.

---

## Fallback

All three profiles: `fallback.cannot_resolve_action = raise_no_safe_match`; `fallback.global_default_tier_allowed = false`. Regulatory profiles MUST NOT return weak global defaults.

---

## Related

- [`/resolve`](../api-reference/resolve.md), [`/method-packs`](../api-reference/method-packs.md).
- [`concepts/method_pack.md`](../concepts/method_pack.md), [Product Carbon pack](product_carbon.md).
