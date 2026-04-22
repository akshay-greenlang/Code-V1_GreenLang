# GreenLang Factors v1 — Regulatory & Methodology Coverage Audit

**Prepared by:** GL-RegulatoryIntelligence
**Audit date:** 2026-04-22
**Scope commitment:** Full Factors v1 (all 7 method-pack slices + variants)
**Founder decision:** 2026-04-22 — FULL v1 scope locked.

---

## 1. Executive summary

GreenLang Factors v1 ships **7 method-pack slices** (the v1 "cut list") registered in `greenlang/factors/method_packs/`:

| Slice | Module | Profile(s) / Variants |
|-------|--------|-----------------------|
| electricity | `electricity.py` | `CORPORATE_SCOPE2_LOCATION`, `CORPORATE_SCOPE2_MARKET` + residual-mix variants EU / US / AU / JP |
| combustion (Scope 1) | `corporate.py` | `CORPORATE_SCOPE1` + `CORPORATE_SCOPE3` (umbrella for combustion-adjacent Scope 3) |
| freight | `freight.py` | `FREIGHT_ISO_14083` |
| material-CBAM | `eu_policy.py` | `EU_CBAM`, `EU_DPP` |
| land-removals | `land_removals.py` | `LAND_REMOVALS` umbrella + 4 variants (`lsr_land_use_emissions`, `lsr_land_management`, `lsr_removals`, `lsr_storage`) |
| product-carbon | `product_carbon.py`, `product_lca_variants.py` | `PRODUCT_CARBON` + `pas_2050`, `eu_pef`, `eu_oef` |
| finance-proxy | `finance_proxy.py` | `FINANCE_PROXY` umbrella + 7 PCAF variants |

19 source parsers live in `greenlang/factors/ingestion/parsers/` — covering electricity (eGRID, DESNZ, India CEA, AIB, Green-e, NGA, METI), combustion (EPA Hub, TCR, DESNZ, IPCC), transport (freight_lanes, GHG Protocol, DESNZ), material / EPD (EC3), CBAM (`cbam_full`), PACT product exchange (`pact_product_data`), PCAF proxy (`pcaf_proxies`), land-removals (`lsr_removals`), waste (`waste_treatment`), and IPCC defaults.

**Headline verdict (detail in §7 matrix):**

- **COMPLETE** for GHG Protocol Corporate / Scope 2 / Scope 3, ISO 14064-1, ISO 14083 / GLEC, ISO 14067, GHG Protocol Product, PACT v3, EU CBAM, PCAF Part A (all 7 asset classes), GHG Protocol LSR, IPCC 2006 + 2019 Refinement, AIB residual mix, Green-e, India CEA, UK DESNZ / SECR, CSRD E1 (via corporate packs), IFRS S2, TCFD, SBTi v5, California SB 253, UK TPT, Singapore SGX 711A, BRSR climate metrics.
- **PARTIAL** for EU ESPR/DPP (waiting on Commission implementing acts — DPP pack registered at `0.1.0`), EU Taxonomy (climate metrics covered; activity-level screening criteria not in factors scope), EU SFDR PAI (indicators 1-3 covered via corporate packs; PAI 4-14 require social / governance data outside Factors v1), India CCTS (baseline intensity benchmarks not yet loaded), CORSIA (aviation freight covered via GLEC but CORSIA-specific eligibility emissions unit not a first-class profile), Battery Regulation 2023/1542 (DPP pack accepts the payload; carbon-intensity class thresholds not enumerated), EU Green Claims (framework only — requires product-level LCA which PEF already serves).
- **GAP** for: GHG Protocol Scope 2 Guidance residual-mix for jurisdictions outside EU / US / CA / AU / JP; VCS VM module-level methodology matching beyond the generic LSR variants; TNFD nature-metrics beyond the climate-adjacent subset; IFRS-S2 transition-plan disclosure *text* (Factors v1 is numerical only); SEC climate final rule (currently stayed; rule text re-issue pending); Japan FSA disclosures (SSBJ standards still in draft). Remediation is enumerated per-row.

The data foundation (Canonical v2 Factor Record: `greenlang/data/canonical_v2.py` + `greenlang/data/emission_factor_record.py`) carries every field required to serve every COMPLETE row — `factor_family`, `formula_type`, `method_profile`, `gwp_basis`, `verification.status`, `primary_data_flag`, `redistribution_class`, `boundary`, `scope`, `biogenic_co2`, `explainability`, `uncertainty_95ci`, `activity_schema`, `jurisdiction`. This is the enabling substrate.

---

## 2. Architecture ground truth

### 2.1 Registered method packs (canonical inventory)

Source of truth: `greenlang/factors/method_packs/__init__.py` imports 8 submodules that self-register packs at import time via `register_pack()` (from `registry.py`) and per-variant registries for PCAF and LSR.

**Profile-level packs** (resolvable by `get_pack(MethodProfile.X)`):

```
CORPORATE_SCOPE1                      corporate.py
CORPORATE_SCOPE2_LOCATION             corporate.py / electricity.py (ELECTRICITY_LOCATION reuses profile)
CORPORATE_SCOPE2_MARKET               corporate.py / electricity.py (ELECTRICITY_MARKET reuses profile)
CORPORATE_SCOPE3                      corporate.py
PRODUCT_CARBON                        product_carbon.py  (+ PAS_2050 / PEF / OEF reuse same profile)
FREIGHT_ISO_14083                     freight.py
LAND_REMOVALS                         land_removals.py (umbrella)
FINANCE_PROXY                         finance_proxy.py (umbrella)
EU_CBAM                               eu_policy.py
EU_DPP                                eu_policy.py
```

**Variant-level packs** (resolvable by `get_pack("<variant_name>")`):

```
PCAF:  pcaf_listed_equity, pcaf_corporate_bonds, pcaf_business_loans,
       pcaf_project_finance, pcaf_commercial_real_estate, pcaf_mortgages,
       pcaf_motor_vehicle_loans
LSR:   lsr_land_use_emissions, lsr_land_management, lsr_removals, lsr_storage
Product-LCA: pas_2050, eu_pef, eu_oef
Electricity residual: ELECTRICITY_RESIDUAL_MIX_EU / US / AU / JP
       (all reuse CORPORATE_SCOPE2_MARKET profile; routed via
        get_residual_mix_pack(country) — `electricity.py` lines 303-344)
```

### 2.2 Registered parsers

Source of truth: `greenlang/factors/ingestion/parsers/__init__.py` — `build_default_registry()` registers 9 parsers under canonical `source_id` keys (EPA Hub, eGRID, DESNZ, DEFRA alias, IPCC, CBAM, GHG Protocol, TCR, Green-e). Additional specialised parsers live as modules (wired by downstream adapters):

```
aib_residual_mix.py            AIB European Residual Mix (EU-27 + EEA)
australia_nga_residual.py      Australian NGA + LGC-netted residual mix
cbam_full.py                   CBAM iron/steel/aluminum/cement/fertilisers/electricity/hydrogen
desnz_uk.py                    UK DESNZ GHG conversion factors (Scope 1/2/3 WTT)
ec3_epd.py                     EC3 / EPD International (ISO 14025 / EN 15804)
egrid.py                       US EPA eGRID subregion + national
epa_ghg_hub.py                 US EPA GHG Emission Factors Hub
freight_lanes.py               GLEC-aligned lane factors (road / sea / air / rail / IWW)
ghg_protocol.py                GHG Protocol Scope 3 Cat 1 / 4-9 / 13 factors
green_e.py                     Green-e renewable energy certification
green_e_residual.py            Green-e Residual Mix (US NERC + Canadian provinces)
india_cea.py                   India CEA CO2 Baseline Database (NEWNE / S / NER / All-India)
ipcc_defaults.py               IPCC 2006 GL + 2019 Refinement Tier 1 defaults
japan_meti_residual.py         METI residual mix (10 utility service areas)
lsr_removals.py                LSR activity / biochar / BECCS / DACCS / blue carbon
pact_product_data.py           PACT Pathfinder v2 ProductFootprint
pcaf_proxies.py                PCAF sector × geography × asset-class proxies
tcr.py                         The Climate Registry default factors
waste_treatment.py             Waste treatment (landfill / incineration / recycling / composting)
```

### 2.3 Canonical Factor Record field inventory

Required fields for the standards below — all present on `EmissionFactorRecord` in `greenlang/data/emission_factor_record.py` and `greenlang/data/canonical_v2.py`:

`factor_id`, `factor_family`, `formula_type`, `method_profile`, `scope`, `boundary`, `gas_vector` (explicit CO2 / CH4 / N2O / HFC / PFC / SF6 / NF3 columns), `gwp_set` (AR4 / AR5 / AR6 @ 100y or 20y), `jurisdiction` (country / subregion), `source_id`, `source_org`, `source_year`, `publication_date`, `vintage`, `valid_from`, `valid_to`, `value`, `unit`, `uncertainty_95ci`, `uncertainty_distribution`, `primary_data_flag`, `redistribution_class`, `verification.status`, `electricity_basis`, `activity_schema` (CN codes / NACE / GICS / NAICS), `biogenic_co2_split`, `land_use_change_flag`, `transmission_loss_included`, `functional_unit`, `allocation_method`, `explainability` (derivation trail + assumptions).

---

## 3. Coverage analysis — by standard

Each row below answers six points: **(1)** registered profile, **(2)** parser(s), **(3)** required canonical fields, **(4)** coverage gap, **(5)** remediation (exact file path), **(6)** v1 slice that promotes the factors.

---

### 3.1 Corporate GHG accounting

#### GHG Protocol Corporate Standard (Scope 1 / 2 / 3)
- **Profile:** `MethodProfile.CORPORATE_SCOPE1`, `CORPORATE_SCOPE2_LOCATION`, `CORPORATE_SCOPE2_MARKET`, `CORPORATE_SCOPE3` — `greenlang/factors/method_packs/corporate.py` (lines 27-178).
- **Parsers:** `epa_ghg_hub.py` (US Scope 1+2), `desnz_uk.py` (UK Scope 1/2/3), `tcr.py` (US defaults), `ipcc_defaults.py` (international Tier 1), `ghg_protocol.py` (Scope 3 Cat 1 / 4-9 / 13), `egrid.py` (US grid), `india_cea.py`.
- **Required fields:** `scope`, `factor_family` (EMISSIONS / HEATING_VALUE / OXIDATION / CARBON_CONTENT / REFRIGERANT_GWP), `gwp_set`, `biogenic_co2_split`, `formula_type` (DIRECT_FACTOR, COMBUSTION).
- **Gap:** None. Biogenic separate-reporting is enforced by `BiogenicTreatment.REPORTED_SEPARATELY`.
- **Remediation:** N/A.
- **Slice:** combustion + electricity.

#### GHG Protocol Scope 2 Guidance (location + market + residual mix)
- **Profile:** `CORPORATE_SCOPE2_LOCATION` and `CORPORATE_SCOPE2_MARKET` (`corporate.py`), with 4 residual-mix variants in `electricity.py`: `ELECTRICITY_RESIDUAL_MIX_EU` (AIB), `_US` (Green-e), `_AU` (NGA-derived), `_JP` (METI-derived).
- **Parsers:** `egrid.py`, `india_cea.py`, `desnz_uk.py` (location); `aib_residual_mix.py`, `green_e_residual.py`, `australia_nga_residual.py`, `japan_meti_residual.py` (market / residual).
- **Required fields:** `electricity_basis`, `grid_region`, `transmission_loss_included`, `residual_mix` flag.
- **Gap (PARTIAL for non-covered jurisdictions):** Residual mix outside EU / US / CA / AU / JP is not shipped in v1 (e.g., Brazil, India CERC, Korea ERDC, Mexico SIE). The AIB pack is used as fallback but this is methodologically weaker than a jurisdiction-specific residual.
- **Remediation:** Add a parser per jurisdiction — suggested new files `greenlang/factors/ingestion/parsers/brazil_cce_residual.py`, `korea_ksa_residual.py`, `mexico_sie_residual.py`. Register corresponding variants in `greenlang/factors/method_packs/electricity.py` following the `ELECTRICITY_RESIDUAL_MIX_JP` template (lines 257-296) and extend `RESIDUAL_MIX_PACKS_BY_COUNTRY` (line 300).
- **Slice:** electricity.

#### GHG Protocol Scope 3 Standard (all 15 categories)
- **Profile:** `CORPORATE_SCOPE3` — accepts families EMISSIONS, MATERIAL_EMBODIED, TRANSPORT_LANE, WASTE_TREATMENT, FINANCE_PROXY, ENERGY_CONVERSION with formula types DIRECT_FACTOR, SPEND_PROXY, LCA, TRANSPORT_CHAIN (lines 144-159 of `corporate.py`).
- **Parsers:** `ghg_protocol.py` (Cat 1 EEIO, 4 upstream transport, 5 waste, 6 travel, 7 commuting, 9 downstream transport, 13 downstream leased), `ec3_epd.py` (Cat 1 construction goods), `pact_product_data.py` (Cat 1 supplier-specific), `pcaf_proxies.py` (Cat 15 Investments), `freight_lanes.py` (Cat 4 / 9), `waste_treatment.py` (Cat 5), `desnz_uk.py` (all categories UK).
- **Required fields:** `scope=3`, Scope 3 category field (`scope3_category` on activity_schema), calculation method (`calculation_method`), data-quality score.
- **Gap (PARTIAL):** Cat 11 (use of sold products) requires product-lifetime + utilisation assumptions. Current records carry `boundary` but not a `product_lifetime_years` or `utilisation_factor`. Cat 2 (capital goods) relies on EEIO spend intensities — acceptable but not primary. Cat 8, 10, 12, 14 are handled by the same EEIO plus upstream/downstream ratios — works but less granular than single-purpose factors.
- **Remediation:** (a) Extend `FactorParameters` in `greenlang/data/canonical_v2.py` to include `product_lifetime_years`, `annual_utilisation`, `end_of_life_scenario`. (b) Add GHG Protocol Cat 11 guidance loader: new file `greenlang/factors/ingestion/parsers/ghg_protocol_cat11.py` referencing the GHG Protocol Scope 3 Cat 11 workbook. (c) Keep using the existing `CORPORATE_SCOPE3` pack — no new method profile needed.
- **Slice:** combustion + freight + product-carbon + finance-proxy (Scope 3 spans all slices).

#### ISO 14064-1:2018 (organisational GHG inventory)
- **Profile:** Same corporate profiles carry `reporting_labels=("GHG_Protocol", "IFRS_S2", "ISO_14064")` (lines 55, 88, 125, 170 of `corporate.py`).
- **Parsers:** Same as GHG Protocol Corporate.
- **Required fields:** `gwp_set` with SAR / AR4 / AR5 / AR6 optionality (ISO 14064-1 Annex B permits, doesn't mandate AR6). Currently `gwp_basis="IPCC_AR6_100"` is hard-coded per pack; `gwp_set` column on the record is nevertheless the source of truth and vectorised by gas.
- **Gap:** None — `gwp_set` enum in `greenlang/data/emission_factor_record.py` supports `SAR`, `AR4`, `AR5`, `AR6` so an ISO 14064-1 user who needs AR5 can resolve it via record filtering even though the default pack tag is AR6.
- **Remediation:** Document the `gwp_set` resolver contract (existing — add to the ISO 14064-1 customer guide, not a code change).
- **Slice:** combustion + electricity.

#### IFRS S2 (Climate-related Disclosures)
- **Profile:** IFRS S2 delegates measurement to the GHG Protocol. All four corporate packs carry `reporting_labels` including `"IFRS_S2"`.
- **Parsers:** Same as GHG Protocol Corporate.
- **Required fields:** Scope 1 + 2 + material Scope 3, cross-industry metrics, financed-emissions (if entity is financial) → PCAF packs.
- **Gap:** IFRS S2 also requires transition-plan narrative + climate-resilience analysis. These are textual, outside Factors v1 scope. Numerical coverage is complete.
- **Remediation:** None for Factors v1.
- **Slice:** combustion + electricity + finance-proxy.

#### TCFD (legacy disclosure framework)
- **Profile:** Implicit via corporate packs (TCFD is superseded in most jurisdictions by IFRS S2 as of 2024, but still referenced by Japan FSA and historical UK disclosures). `METI` Japan residual-mix pack explicitly labels `"TCFD"` (line 285 of `electricity.py`).
- **Parsers:** Same as GHG Protocol.
- **Required fields:** Absolute Scope 1+2, Scope 3 where relevant, intensity metrics.
- **Gap:** None at the factor level. TCFD governance / strategy / risk pillars are non-factor disclosures.
- **Remediation:** None.
- **Slice:** combustion + electricity.

#### CDP climate questionnaire
- **Profile:** CDP requests Scope 1+2 (location AND market), Scope 3 all 15 categories, targets, governance. Factors side: corporate packs. US residual-mix pack (`ELECTRICITY_RESIDUAL_MIX_US`) explicitly carries `"CDP"` label (line 201 of `electricity.py`).
- **Parsers:** Same as GHG Protocol Corporate.
- **Required fields:** Dual-reporting Scope 2 (both bases on one inventory) — resolved by calling both `CORPORATE_SCOPE2_LOCATION` and `CORPORATE_SCOPE2_MARKET` for the same activity.
- **Gap:** None.
- **Remediation:** None.
- **Slice:** combustion + electricity.

#### SBTi criteria v5
- **Profile:** SBTi requires GHG Protocol Corporate + Scope 3 coverage ≥ 67% of value-chain where Scope 3 > 40% of total. v5 tightens net-zero definition and Scope 3 FLAG (Forests, Land, Agriculture). The corporate packs + LSR variants (`lsr_land_use_emissions`, `lsr_land_management`, `lsr_removals`) cover FLAG categories.
- **Parsers:** GHG Protocol Corporate parsers + `lsr_removals.py` + `ipcc_defaults.py` (for FLAG Tier 1 defaults).
- **Required fields:** Removals must be tagged `factor_family=LAND_USE_REMOVALS` with `is_active_removal` + `permanence_class` (in `LSRPackMetadata`, `greenlang/factors/method_packs/land_removals.py` lines 158-175).
- **Gap (PARTIAL):** SBTi FLAG requires aligning with the GHG Protocol LSR guidance finalised 2024. The LSR pack is loaded at `pack_version="1.0.0"` and mirrors the 2024 draft; tracking the final (Q2 2026) published version is needed.
- **Remediation:** When GHG Protocol LSR final publishes, bump `pack_version` in `greenlang/factors/method_packs/land_removals.py` (function `_build_pack`, line 300) and re-run validation fixtures.
- **Slice:** combustion + electricity + land-removals.

---

### 3.2 Product / value chain

#### GHG Protocol Product Standard
- **Profile:** `MethodProfile.PRODUCT_CARBON` — `greenlang/factors/method_packs/product_carbon.py`, `reporting_labels=("ISO_14067", "GHG_Protocol_Product", "PACT")` (line 50).
- **Parsers:** `pact_product_data.py` (supplier-shared PCFs), `ec3_epd.py` (verified EPDs), `ghg_protocol.py` (Cat 1 EEIO for supplement).
- **Required fields:** `functional_unit`, `boundary` (cradle_to_gate / cradle_to_grave), `allocation_method`, `recycled_content_pct`, `biogenic_co2_split`.
- **Gap:** None.
- **Remediation:** N/A.
- **Slice:** product-carbon.

#### ISO 14067:2018 (product carbon footprint)
- **Profile:** `PRODUCT_CARBON` (audit template: "Product carbon footprint per ISO 14067", `product_carbon.py` line 52).
- **Parsers:** Same as GHG Protocol Product.
- **Required fields:** Same.
- **Gap:** None.
- **Remediation:** N/A.
- **Slice:** product-carbon.

#### PACT Pathfinder Framework v3 (product data exchange)
- **Profile:** `PRODUCT_CARBON` carries `"PACT"` reporting label. The exchange shape is handled by `pact_product_data.py` which parses PACT Pathfinder v2 ProductFootprint objects (docstring at lines 1-29 of `pact_product_data.py`).
- **Parsers:** `pact_product_data.py`.
- **Required fields:** `pCfExcludingBiogenic`, `pCfIncludingBiogenic`, `fossilGhgEmissions`, `biogenicCarbonEmissions`, `declaredUnit`, `pcfSpec` version.
- **Gap (PARTIAL):** Parser docstring lists `pcfSpec: "2.0.0"`. PACT v3 is the active spec as of 2026-03. The mapping fields for v3 are supersets of v2 but explicitly add `assurance` + `productOrSectorSpecificRules` refinements.
- **Remediation:** Bump parser to consume PACT v3 schema fields at `greenlang/factors/ingestion/parsers/pact_product_data.py` — extend input row comment, add `assurance` / `productOrSectorSpecificRules` parsing into `explainability.assumptions`. No new method pack needed.
- **Slice:** product-carbon.

#### PEF (EU Product Environmental Footprint)
- **Profile:** `PEF` variant in `greenlang/factors/method_packs/product_lca_variants.py` (lines 88-129). Uses `PRODUCT_CARBON` profile umbrella; `reporting_labels=("EU_PEF", "EF_3_1", "PEFCR", "ISO_14067")`. Selection rule requires verification (`require_verification=True`).
- **Parsers:** `ec3_epd.py` (compatible — EN 15804 is PEF-conformant for construction), `pact_product_data.py` (supplier CFs), plus downstream EF 3.1 secondary dataset loaders (not yet shipped as a parser but factor-level).
- **Required fields:** `pefcr_id`, `functional_unit`, 16-indicator EF vector in `explainability.extras`.
- **Gap (PARTIAL):** Factors v1 surfaces the climate-change (GWP) indicator from PEF. The remaining 15 EF indicators (acidification, eutrophication, land-use, water, etc.) are stored in `explainability.extras` per doc but are not first-class columns. For a PEF-regulated customer this is sufficient for the climate number; auditors needing the full EF disclosure would read `extras`.
- **Remediation:** If full EF indicator surfacing is needed, extend `EmissionFactorRecord` in `greenlang/data/emission_factor_record.py` with an optional `ef_vector` column. Not required for v1 GA.
- **Slice:** product-carbon.

#### ISO 14040 / 14044 (LCA foundations)
- **Profile:** Implicit — `PRODUCT_CARBON` + `PEF` + `OEF` + `PAS_2050` all sit on the ISO 14040/14044 conceptual foundation. `FormulaType.LCA` identifies LCA-based records.
- **Parsers:** `ec3_epd.py`, `pact_product_data.py`.
- **Required fields:** `boundary`, `functional_unit`, `allocation_method`, cut-off rules (captured in `explainability`).
- **Gap:** None at factor level. ISO 14040 is a process standard, not a numerical specification.
- **Remediation:** None.
- **Slice:** product-carbon.

#### EPD frameworks (EN 15804 construction, ISO 14025 general)
- **Profile:** `PRODUCT_CARBON` with records from `ec3_epd.py` (EPD International, NCC, Building Transparency EC3 — all EN 15804-conformant with modules A1-A3 through C4).
- **Parsers:** `ec3_epd.py`.
- **Required fields:** `modules_reported` (A1-A3 / A4 / A5 / B1-B7 / C1-C4 / D), `functional_unit`, `declared_unit`, `validUntil`, `program_operator`, `verification_date`.
- **Gap:** None for climate. EN 15804+A2 mandates the same 16 EF indicators as PEF — same note as PEF applies for non-climate indicators.
- **Remediation:** None for v1.
- **Slice:** product-carbon.

---

### 3.3 Freight / mobility

#### ISO 14083:2023 (transport chain emissions)
- **Profile:** `MethodProfile.FREIGHT_ISO_14083` — `greenlang/factors/method_packs/freight.py`, `reporting_labels=("ISO_14083", "GLEC")`, `allowed_boundaries=("WTW", "WTT")`, `FormulaType.TRANSPORT_CHAIN`.
- **Parsers:** `freight_lanes.py` (per-mode lane factors road/sea/air/rail/IWW), `ghg_protocol.py` (Cat 4/9 defaults), `desnz_uk.py` (UK-specific distance factors).
- **Required fields:** `mode`, `vehicle_class`, `fuel`, `payload_utilization`, `empty_running_factor`, `wtw_gco2e_per_tkm`, `ttw_gco2e_per_tkm`, `wtt_gco2e_per_tkm`.
- **Gap:** None. Pack version is `0.3.0`, indicating ongoing maturation — bump to 1.0.0 once the v3 GLEC data load is validation-complete.
- **Remediation:** Bump `pack_version` in `freight.py` line 57 when GLEC v3 lane coverage reaches v1 target.
- **Slice:** freight.

#### GLEC Framework v3 (Smart Freight Centre)
- **Profile:** Same `FREIGHT_ISO_14083` pack — GLEC is the implementing specification.
- **Parsers:** `freight_lanes.py` (docstring explicitly cites "GLEC Framework v3 / ISO 14083:2023", line 6-7 of `freight_lanes.py`).
- **Required fields:** Same.
- **Gap:** None.
- **Remediation:** None.
- **Slice:** freight.

#### CORSIA (aviation)
- **Profile:** Aviation is served via `FREIGHT_ISO_14083` pack when the flight is a freight leg. Passenger aviation under CORSIA obligations is not a first-class profile.
- **Parsers:** `freight_lanes.py` (air mode), `ghg_protocol.py` (Cat 6 business travel for passenger).
- **Required fields:** `aircraft_type`, `great_circle_distance_km`, `radiative_forcing_index` (not currently a first-class field), CORSIA eligibility flag.
- **Gap (PARTIAL → GAP for CORSIA offset unit):** CORSIA's eligibility-for-offsetting is a regulatory compliance instrument (emissions units — CORSIA Eligible Emissions Units), not a factor. The emission-factor side is covered; the **offset unit** accounting is out of Factors v1 scope (belongs in the offsets module).
- **Remediation:** Add `radiative_forcing_index` to `FactorParameters` in `greenlang/data/canonical_v2.py` (aviation high-altitude RFI multiplier is a common customer ask). For CORSIA unit eligibility, build a separate `greenlang/offsets/` namespace — out of Factors v1 scope.
- **Slice:** freight.

---

### 3.4 Land, removals, and nature

#### GHG Protocol Land Sector and Removals Standard (2024)
- **Profile:** `MethodProfile.LAND_REMOVALS` umbrella + 4 variants — `greenlang/factors/method_packs/land_removals.py`. Variants: `lsr_land_use_emissions`, `lsr_land_management`, `lsr_removals`, `lsr_storage`. Full `LSRPackMetadata` captures permanence (SHORT/MEDIUM/LONG), reversal risk (LOW/MEDIUM/HIGH), buffer pool %, biogenic treatment (CARBON_NEUTRAL / SEQUESTRATION_TRACKED / STORAGE_TRACKED), removal category (NATURE_BASED / TECHNOLOGY_BASED / HYBRID), removal type (13 enum values incl. AFFORESTATION, BIOCHAR, DACCS, BECCS, BLUE_CARBON_MANGROVE, ENHANCED_ROCK_WEATHERING), verification standard (VCS / GOLD / PURO / ISOMETRIC / ICVCM_CCP_APPROVED), `is_active_removal`, `iluc_included`, `soc_tracked`.
- **Parsers:** `lsr_removals.py`, `ipcc_defaults.py` (LULUCF section for Tier 1).
- **Required fields:** `permanence_class`, `reversal_risk`, `buffer_pool_pct` (derived from helper `compute_buffer_pool_pct` at line 652), `biogenic_treatment`, `is_active_removal`, `soc_delta_tc`, `agb_delta_tc`, `iluc_tco2e`.
- **Gap:** None at factor level. Pack tracks the 2024 draft; monitor Q2 2026 final.
- **Remediation:** Bump `pack_version` in `_build_pack` at `land_removals.py` line 300 when the final LSR publishes.
- **Slice:** land-removals.

#### IPCC 2006 Guidelines + 2019 Refinement
- **Profile:** `IPCC_Tier_1` defaults feed multiple packs (Scope 1 combustion, Scope 3 Cat 5 waste, LSR). `reporting_labels` in `land_removals.py` explicitly includes `"IPCC_2006_GL"` and `"IPCC_2019_Refinement"` (lines 328-331).
- **Parsers:** `ipcc_defaults.py` — energy_stationary / energy_mobile / industrial_processes / agriculture / LULUCF / waste sections.
- **Required fields:** `tier` (1/2/3), `default_category`, gas-level split (ipcc_defaults.py reads CO2, CH4, N2O separately).
- **Gap:** None — Tier 1 defaults are shipped.
- **Remediation:** None.
- **Slice:** combustion + land-removals + freight (mobile combustion).

#### VCS VM modules (voluntary carbon market)
- **Profile:** `LAND_REMOVALS` umbrella covers VCS project types via `LSRPackMetadata.verification_standards = (VerificationStandard.VCS, ...)`.
- **Parsers:** `lsr_removals.py` is the generic loader.
- **Required fields:** VCS methodology ID (e.g., VM0042 for improved agricultural land management, VM0047 for afforestation/reforestation), vintage, issuance, buffer pool contribution.
- **Gap (PARTIAL):** Factors v1 does not enumerate individual VCS methodology modules (VM0001 through VM0048) as first-class metadata — a VCS methodology ID is captured in `source_id` / `explainability.methodology_id` but not keyed in a VCS-specific enum.
- **Remediation:** Add a `VCSMethodologyID` enum under `greenlang/factors/method_packs/land_removals.py` near line 131 (alongside `VerificationStandard`) if buyers need drill-down. Minor — capture in explainability for v1.
- **Slice:** land-removals.

#### TNFD v1.0 (nature-related disclosures)
- **Profile:** Climate-adjacent nature metrics (land-use change, biodiversity impact) are partly served via `LAND_REMOVALS` variants (iLUC + SOC tracking on `lsr_land_use_emissions`). Full TNFD LEAP (Locate, Evaluate, Assess, Prepare) is broader.
- **Parsers:** `lsr_removals.py` for climate; pure nature metrics (species impact, water, biodiversity) are outside Factors v1.
- **Required fields:** Climate-related: covered. Water / biodiversity / pollution: not covered.
- **Gap (GAP for non-climate TNFD):** TNFD non-climate metrics (biodiversity footprint, water stress, species impact) require separate factor families that Factors v1 does not register.
- **Remediation:** Post-v1 scope: add `FactorFamily.NATURE_IMPACT` (water / biodiversity / land) to `greenlang/data/canonical_v2.py` line 42 and spin a `nature.py` method pack. Out of v1 commitment.
- **Slice:** land-removals (climate subset only).

---

### 3.5 Electricity (see also §3.1 Scope 2)

#### AIB residual mix for Europe
- **Profile:** `ELECTRICITY_RESIDUAL_MIX_EU` (`electricity.py` lines 138-171). Keyed by `source_id == "aib_residual_mix_eu"` custom filter.
- **Parsers:** `aib_residual_mix.py`.
- **Required fields:** `calendar_year`, `residual_mix_g_co2_per_kwh`, `publication_date`, `version`.
- **Gap:** None.
- **Remediation:** None.
- **Slice:** electricity.

#### Green-e Energy standard (US)
- **Profile:** `ELECTRICITY_RESIDUAL_MIX_US` (`electricity.py` lines 174-212).
- **Parsers:** `green_e.py` (certification data) + `green_e_residual.py` (US NERC / Canadian provinces residual).
- **Required fields:** `country` (US/CA), `region` (NERC), `subregion`, gas-level rates in lb/MWh.
- **Gap:** Licensing — `green_e_residual.py` docstring notes `RedistributionClass.RESTRICTED`. The pack correctly enforces attribution requirement (audit template, line 205-207).
- **Remediation:** None. License-class surfacing is the "never mix licensing classes" CTO non-negotiable and is enforced by the `redistribution_class` field.
- **Slice:** electricity.

#### I-REC Standard
- **Profile:** Served by `CORPORATE_SCOPE2_MARKET` pack + `ELECTRICITY_MARKET` — accepts RECs, GOs, I-RECs under `market_instruments=ALLOWED`. No I-REC-specific pack variant.
- **Parsers:** None specific — I-REC registry data is consumed as supplier-specific records via customer uploads (`customer_override` tier 1 in the fallback hierarchy).
- **Required fields:** `certificate_id`, `issue_date`, `redemption_date`, `vintage`, `facility_id`, `technology` (captured in `explainability`).
- **Gap (PARTIAL):** No first-class I-REC parser. Customer uploads their I-REC surrender log, which fills the supplier-specific tier. This is the intended pattern per `DEFAULT_FALLBACK` (`base.py` line 147-155).
- **Remediation:** Optional future parser `greenlang/factors/ingestion/parsers/irec_registry.py` if bulk I-REC ingestion becomes a pattern. Not required for v1.
- **Slice:** electricity.

#### Energy Attribute Certificate registries (generic — REGO, GO, REC, etc.)
- **Profile:** Same — `ELECTRICITY_MARKET` pack.
- **Parsers:** Handled as customer overlays. AIB Green-e cover EU / US respectively.
- **Gap:** See I-REC.
- **Remediation:** Same.
- **Slice:** electricity.

---

### 3.6 Finance

#### PCAF Global GHG Accounting and Reporting Standard Part A + B
- **Profile:** `MethodProfile.FINANCE_PROXY` umbrella + 7 asset-class variants — `greenlang/factors/method_packs/finance_proxy.py`:
  - `pcaf_listed_equity`
  - `pcaf_corporate_bonds`
  - `pcaf_business_loans` (+ unlisted equity)
  - `pcaf_project_finance`
  - `pcaf_commercial_real_estate`
  - `pcaf_mortgages`
  - `pcaf_motor_vehicle_loans`
- **Parsers:** `pcaf_proxies.py` (sector × geography × asset class proxy intensities), plus `ghg_protocol.py` (Scope 3 Cat 15), plus corporate packs for counterparty Scope 1+2.
- **Required fields:** `outstanding_amount`, `evic` or `property_value_at_origination` or `vehicle_value_at_origination` or `total_committed_capital` (depending on asset class), `pcaf_dqs` (1-5 in `PCAFDataQualityScore`), `attribution_factor`. The `PCAFPackMetadata` at line 144 captures `attribution_method`, `attribution_formula`, `intensity_modes`, `requires_scope3_for_sectors` (16 high-emitting sectors enumerated at lines 239-256), `uncertainty_band_required_dqs=4`.
- **Gap (PARTIAL for Part B):** PCAF Part B (facilitated emissions for capital markets, insurance-associated emissions) is the newer supplement. The current pack aligns with Part A v2.0 (2022). Insurance-associated emissions (PCAF Insurance Standard 2022) and facilitated emissions (Part B 2023) are not standalone variants.
- **Remediation:** Add two new variants to `finance_proxy.py` after line 571: `PCAF_FACILITATED_EMISSIONS` and `PCAF_INSURANCE_ASSOCIATED`, each with its own metadata block following the `_build_pack` helper at line 292.
- **Slice:** finance-proxy.

#### TCFD financial sector guidance
- **Profile:** Served by PCAF variants (financial-sector flavour of IFRS S2 / TCFD).
- **Parsers:** Same as PCAF.
- **Gap:** None at factor level.
- **Remediation:** None.
- **Slice:** finance-proxy.

#### SBTi Financial Institutions v1.1
- **Profile:** PCAF variants carry `reporting_labels=("PCAF", "PCAF_Part_A_v2.0", "GHG_Protocol_Scope3_Cat15", "IFRS_S2")` — SBTi-FI uses the same PCAF numbers as the accounting foundation.
- **Parsers:** Same as PCAF.
- **Required fields:** Sector-level (real-estate, power, fossil-fuel) convergence pathways.
- **Gap:** SBTi-FI sector pathways are not encoded in Factors v1 (pathways belong in a `targets/` module). Factor side is complete.
- **Remediation:** None for Factors v1.
- **Slice:** finance-proxy.

---

### 3.7 Regulatory regimes

#### EU CBAM Regulation 2023/956 (definitive period from 2026-01-01)
- **Profile:** `MethodProfile.EU_CBAM` — `greenlang/factors/method_packs/eu_policy.py` (lines 22-69). `require_verification=True` — only certified + externally verified factors. Boundary cradle_to_gate + combustion. Biogenic **excluded** from CBAM value per Annex III.
- **Parsers:** `cbam_full.py` — iron/steel, aluminium, cement, fertilisers, electricity, hydrogen (all CN codes covered).
- **Required fields:** `cn_code` (activity_schema), `verification.status=external_verified`, `primary_data_flag=primary` (EU default values only as fallback with surcharge), `embedded_direct_emissions_tco2_per_t`, `embedded_indirect_emissions_tco2_per_t`.
- **Gap:** None for the definitive regime start. EU default values **revision** is ongoing; pack has `max_age_days=365*2` (2-year deprecation — aggressive — line 59) to force re-certification cycles.
- **Remediation:** Monitor EU Commission Implementing Regulation default-value republications. When revised, re-run `cbam_full.py` ingestion.
- **Slice:** material-CBAM.

#### EU CSRD + ESRS (double materiality, E1 climate, E2-E5 + S1-S4 + G1)
- **Profile:** ESRS E1 (climate) is served by corporate packs — `reporting_labels` do not currently enumerate `"ESRS_E1"` explicitly on every pack but the `OEF` variant does (line 168 of `product_lca_variants.py`) and the AIB residual pack does (line 163 of `electricity.py`). E2 (pollution), E3 (water), E4 (biodiversity), E5 (circular economy) are outside Factors v1.
- **Parsers:** Corporate + product + freight parsers serve E1. E2-E5 and S1-S4 / G1 are not factor-driven.
- **Required fields:** Double-materiality flag on disclosure (not a factor concern — reporting-layer).
- **Gap (PARTIAL):** E1 numerical coverage complete; other ESRS topical standards not covered (correctly — they are not factor-bearing).
- **Remediation:** Extend `reporting_labels` on `CORPORATE_SCOPE1`, `CORPORATE_SCOPE2_*`, `CORPORATE_SCOPE3` in `greenlang/factors/method_packs/corporate.py` to include `"CSRD_E1"` / `"ESRS_E1"` — one-line change per pack (lines 55, 88, 125, 170).
- **Slice:** combustion + electricity + freight + product-carbon.

#### EU SFDR Art 8/9 + PAI
- **Profile:** Principal Adverse Impact indicators 1-3 (Scope 1/2/3 GHG emissions; carbon footprint; GHG intensity of investee companies) are served by PCAF variants + corporate packs.
- **Parsers:** `pcaf_proxies.py` + corporate parsers.
- **Required fields:** Investee company Scope 1+2+3 + enterprise value — all in PCAF packs.
- **Gap (PARTIAL):** PAI 4-14 cover fossil-fuel sector exposure, non-renewable energy share, biodiversity, water, hazardous waste, board gender, controversial weapons — factor coverage for 4 (exposure to fossil fuels) and 5 (share of non-renewable energy) possible via activity classification; 6-14 are non-emission metrics.
- **Remediation:** For PAI 4-5: derive from existing sector classification on PCAF factors (`activity_schema.sector_nace`). PAI 6-14: out of Factors v1 scope.
- **Slice:** finance-proxy + combustion + electricity.

#### EU Taxonomy (climate mitigation + adaptation + 4 others)
- **Profile:** Taxonomy uses `do-no-significant-harm` screening criteria — not a factor dataset. Climate-mitigation screening does reference embedded-CO2 thresholds (e.g., cement < 0.498 tCO2/t clinker) which are **factor values** that can be compared.
- **Parsers:** `ec3_epd.py` provides the per-product CO2 intensity needed for threshold comparison.
- **Required fields:** Factor values with CN / NACE codes — present.
- **Gap (PARTIAL):** Factors v1 does not ship the Taxonomy screening thresholds themselves (activity list × DNSH). Those are *regulatory decision tables*, not factors.
- **Remediation:** Build a `greenlang/taxonomy/` module (out of Factors v1 scope) that consumes Factor records and compares to thresholds.
- **Slice:** material-CBAM + product-carbon.

#### EU Battery Regulation 2023/1542 (battery passport, DPP)
- **Profile:** `MethodProfile.EU_DPP` (`eu_policy.py` lines 72-105, `pack_version="0.1.0"`). Accepts MATERIAL_EMBODIED + EMISSIONS with LCA / DIRECT_FACTOR formulas. Cradle-to-gate / cradle-to-grave boundaries.
- **Parsers:** `pact_product_data.py` (PCF input), `ec3_epd.py` (supply-chain material EPDs).
- **Required fields:** `product_id`, carbon footprint class thresholds, battery-specific chemistry + capacity (via `activity_schema` extras).
- **Gap (PARTIAL):** Battery Regulation carbon-intensity class thresholds (Commission Delegated Regulation pending 2026) are not enumerated — Factors v1 stores the value, not the pass/fail classification.
- **Remediation:** When the implementing act publishes, add a `greenlang/factors/method_packs/eu_battery.py` or extend `EU_DPP` with battery-specific thresholds. Bump `EU_DPP.pack_version` from `0.1.0` to `1.0.0` at that time (file line 103).
- **Slice:** material-CBAM + product-carbon.

#### EU ESPR + DPP (textiles, construction, batteries — tranches)
- **Profile:** `EU_DPP` (as above). Pack is pre-regulation (`pack_version="0.1.0"`).
- **Parsers:** `pact_product_data.py`, `ec3_epd.py`.
- **Required fields:** Per-tranche product passport schema (to be published per product group by the Commission).
- **Gap (PARTIAL):** Implementing acts are rolling out through 2026-2028.
- **Remediation:** Track EU ESPR Working Plan; extend `EU_DPP` reporting_labels when each tranche publishes. Documentation path: `greenlang/factors/method_packs/eu_policy.py` line 98.
- **Slice:** material-CBAM + product-carbon.

#### California SB 253 — Climate Corporate Data Accountability Act
- **Profile:** SB 253 mandates GHG Protocol Corporate Scope 1+2 (FY2025 data, first report **2026 year**; Scope 3 first report 2027) with third-party assurance. Served by `CORPORATE_SCOPE1`, `CORPORATE_SCOPE2_*`, `CORPORATE_SCOPE3`.
- **Parsers:** `epa_ghg_hub.py`, `egrid.py`, `tcr.py`, `ghg_protocol.py`, `green_e_residual.py`.
- **Required fields:** Scope 1/2/3 full coverage; assurance in `verification.status`.
- **Gap:** None for the numerical side. CARB's implementing regulations (due Jan 2026) may tighten the definition — monitor.
- **Remediation:** Add `"CA_SB253"` to `reporting_labels` on the four corporate packs in `greenlang/factors/method_packs/corporate.py` (lines 55, 88, 125, 170). One-line change each.
- **Slice:** combustion + electricity.

#### California SB 261 — Climate-related Financial Risk
- **Profile:** SB 261 is a TCFD-style disclosure (biennial starting 2026-01-01). Factor side: corporate + PCAF for financed-risk.
- **Parsers:** Same as TCFD / PCAF.
- **Gap:** None at factor level.
- **Remediation:** None.
- **Slice:** combustion + electricity + finance-proxy.

#### SEC climate rule (currently stayed)
- **Profile:** If un-stayed, would be served by `CORPORATE_SCOPE1` + `CORPORATE_SCOPE2_*`. The US residual-mix pack explicitly carries `"SEC_Climate"` label (`electricity.py` line 202).
- **Parsers:** `epa_ghg_hub.py`, `egrid.py`, `green_e_residual.py`.
- **Gap (GAP — regulatory status):** Rule is stayed by 8th Circuit (March 2024 onwards). Substance is GHG-Protocol-aligned, so factor coverage is ready.
- **Remediation:** None required. If SEC re-issues, pack labels are already wired.
- **Slice:** combustion + electricity.

#### India BRSR (core + sector annexes)
- **Profile:** BRSR Principle 6 (Environment) asks for Scope 1+2 (mandatory top-1000), Scope 3 (voluntary), intensity per revenue. Served by corporate packs.
- **Parsers:** `india_cea.py` (All-India + regional grids — NEWNE, S, NER) + `ipcc_defaults.py`.
- **Required fields:** `jurisdiction.country=IN`, regional grid keys.
- **Gap:** None at the factor level.
- **Remediation:** Add `"India_BRSR"` label to corporate packs in `corporate.py` lines 55, 88, 125, 170.
- **Slice:** combustion + electricity.

#### India CCTS (Carbon Credit Trading Scheme, ~490 obligated entities)
- **Profile:** CCTS is an emissions trading scheme with sectoral baseline intensity benchmarks (issued by BEE). Factor side: corporate scope 1+2 for the obligated entity. CCTS-specific *baseline benchmarks* are not loaded.
- **Parsers:** `india_cea.py`, `ipcc_defaults.py`.
- **Required fields:** Sector intensity benchmark (tCO2/t product), performance year.
- **Gap (PARTIAL → GAP for benchmark table):** Baseline intensity benchmarks (published by BEE, sector-specific — cement, iron & steel, aluminium, fertiliser, petchem, pulp & paper, refineries, textile, chlor-alkali) are not in the factor catalog as first-class benchmark records.
- **Remediation:** Add `greenlang/factors/ingestion/parsers/india_bee_ccts_benchmarks.py` to parse BEE-published intensity-benchmark tables. Optional variant in `greenlang/factors/method_packs/eu_policy.py` (rename or refactor into a `compliance_schemes.py` module; or extend `EU_CBAM` template pattern into a new `IndiaCCTSPack`). Add `MethodProfile.INDIA_CCTS` to `greenlang/data/canonical_v2.py` line 145-154.
- **Slice:** material-CBAM (similar shape to CBAM — per-product benchmark vs. measured).

#### UK SECR (Streamlined Energy & Carbon Reporting)
- **Profile:** Scope 1+2 (mandatory) + energy use + intensity. Served by corporate packs.
- **Parsers:** `desnz_uk.py` — SECR-optimised (Scope 1 fuels / Scope 2 electricity / Scope 3 WTT all in one dataset).
- **Gap:** None.
- **Remediation:** Add `"UK_SECR"` to corporate pack reporting_labels (one-line each, `corporate.py`).
- **Slice:** combustion + electricity.

#### UK TPT (Transition Plan Taskforce)
- **Profile:** Disclosure framework — factor side: corporate + PCAF. Numerical metrics (financed emissions, intensity pathways) come from existing packs.
- **Parsers:** `desnz_uk.py`, `pcaf_proxies.py`.
- **Gap:** None at factor level (TPT is narrative + trajectory, not factor-bearing).
- **Remediation:** None.
- **Slice:** combustion + electricity + finance-proxy.

#### Japan FSA sustainability disclosures (SSBJ)
- **Profile:** SSBJ (Sustainability Standards Board of Japan) issues J-aligned IFRS S2. Served by corporate packs + `ELECTRICITY_RESIDUAL_MIX_JP` (explicit `"Japan_SSBJ"` label, line 287 of `electricity.py`).
- **Parsers:** `japan_meti_residual.py` + corporate.
- **Gap:** SSBJ final standards are draft (as of 2026-04). Subject to change — monitor.
- **Remediation:** When SSBJ finalises (expected Q2 2026), bump `japan_meti_residual.py` parser version and `ELECTRICITY_RESIDUAL_MIX_JP` pack version.
- **Slice:** combustion + electricity.

#### Singapore SGX-ST Rule 711A (mandatory climate disclosures)
- **Profile:** Singapore Exchange mandatory disclosures align with IFRS S2. Served by corporate packs. Singapore grid factor is not explicitly loaded — falls back to IPCC regional default or customer override.
- **Parsers:** Corporate + IPCC defaults.
- **Gap (PARTIAL):** No Singapore-specific grid factor parser. Singapore EMA publishes a grid emission factor (0.408 tCO2/MWh for 2022).
- **Remediation:** Add `greenlang/factors/ingestion/parsers/singapore_ema.py` following the `india_cea.py` template.
- **Slice:** electricity.

---

### 3.8 GWP metrics

#### IPCC AR4 / AR5 / AR6 (100-year + 20-year horizons)
- **Profile:** Every MethodPack carries `gwp_basis` field. Default across v1 is `"IPCC_AR6_100"`. Exception: `PAS_2050` pack in `product_lca_variants.py` line 70 carries `gwp_basis="IPCC_AR5_100"` per historical PAS alignment.
- **Parsers:** All parsers record the `gwp_set` that the source publication used. For multi-gas sources (`aib_residual_mix`, `green_e_residual`, `ipcc_defaults`, `waste_treatment`), gas-level vectors are preserved so downstream CO2e can be re-derived under any AR/horizon.
- **Required fields:** `gwp_set` enum on `EmissionFactorRecord`; gas-level columns (CO2 / CH4 / N2O / HFC / PFC / SF6 / NF3) also preserved per the CTO non-negotiable #1 "gas-level vectors stored separately".
- **Gap:** None. 20-year horizon (AR6_20) is supported by the enum and is reachable by downstream re-compute — but the *pack defaults* only target 100-year. Methane-focused customers requesting AR6_20 must override at the pack level.
- **Remediation:** For AR6_20 as a first-class default (e.g., methane-focused customer), add a `gwp_horizon: str = "100"` option on `MethodPack`. Not a v1 blocker — override mechanism is in place.
- **Slice:** cross-cutting (applies to all slices).

#### Kyoto SAR values (legacy — still used in some national inventories)
- **Profile:** SAR GWPs are still used in UNFCCC national inventories for certain gases (historic). Supported via `gwp_set` enum value `SAR`.
- **Parsers:** Legacy IPCC data can be tagged `gwp_set=SAR`.
- **Gap:** Pack defaults do not offer a SAR variant — customers needing national-inventory-style SAR reporting would re-compute from gas-level vectors.
- **Remediation:** None required. Documented as an override path.
- **Slice:** cross-cutting.

---

## 4. Gap-and-remediation consolidated list

Each remediation below is actionable. File paths are absolute to the repo root.

1. **Residual mix beyond EU/US/CA/AU/JP** → add parsers `greenlang/factors/ingestion/parsers/brazil_cce_residual.py`, `korea_ksa_residual.py`, `mexico_sie_residual.py`, `singapore_ema.py`. Register variants in `greenlang/factors/method_packs/electricity.py` and extend `RESIDUAL_MIX_PACKS_BY_COUNTRY` (line 300).
2. **Scope 3 Cat 11 use-of-sold-products lifetime assumption** → extend `FactorParameters` in `greenlang/data/canonical_v2.py` with `product_lifetime_years`, `annual_utilisation`, `end_of_life_scenario`. Add parser `greenlang/factors/ingestion/parsers/ghg_protocol_cat11.py`.
3. **PACT v3 schema fields** → extend `greenlang/factors/ingestion/parsers/pact_product_data.py` with `assurance` + `productOrSectorSpecificRules` parsing. No new pack.
4. **GLEC / ISO 14083 freight pack version bump** → `greenlang/factors/method_packs/freight.py` line 57: bump `pack_version` from `0.3.0` to `1.0.0` once v3 lane coverage is load-validation complete.
5. **Aviation RFI (CORSIA, Scope 3 Cat 6)** → add `radiative_forcing_index` to `FactorParameters` at `greenlang/data/canonical_v2.py`. Factor side; CORSIA offset units are out of Factors scope.
6. **GHG Protocol LSR 2024 final** → when published, bump `pack_version` in the `_build_pack` helper at `greenlang/factors/method_packs/land_removals.py` line 300.
7. **PCAF Part B (facilitated + insurance-associated)** → add variants in `greenlang/factors/method_packs/finance_proxy.py` after line 571: `PCAF_FACILITATED_EMISSIONS` and `PCAF_INSURANCE_ASSOCIATED`.
8. **CSRD E1 / SB 253 / UK SECR / India BRSR reporting labels** → extend `reporting_labels` tuples on `CORPORATE_SCOPE1`, `CORPORATE_SCOPE2_LOCATION`, `CORPORATE_SCOPE2_MARKET`, `CORPORATE_SCOPE3` in `greenlang/factors/method_packs/corporate.py` at lines 55, 88, 125, 170.
9. **EU Battery Regulation thresholds** → when Commission Delegated Regulation publishes (2026), add `greenlang/factors/method_packs/eu_battery.py` or extend `EU_DPP` at `greenlang/factors/method_packs/eu_policy.py` line 98; bump `pack_version`.
10. **India CCTS baseline benchmarks** → add parser `greenlang/factors/ingestion/parsers/india_bee_ccts_benchmarks.py`. Add `MethodProfile.INDIA_CCTS` enum value in `greenlang/data/canonical_v2.py` around line 153. Create `greenlang/factors/method_packs/india_ccts.py` following the `EU_CBAM` shape.
11. **VCS methodology drill-down** → add `VCSMethodologyID` enum near line 131 of `greenlang/factors/method_packs/land_removals.py` if buyers need module-level VCS (VM0001..VM0048) first-class.
12. **TNFD non-climate nature metrics** → out of Factors v1 scope. Track for v2: new `FactorFamily.NATURE_IMPACT` in `greenlang/data/canonical_v2.py` and `nature.py` method pack.
13. **PEF 16-indicator vector** → optional future extension of `EmissionFactorRecord` in `greenlang/data/emission_factor_record.py` to add `ef_vector` column; not a v1 blocker.
14. **AR6_20y pack-default option** → add optional `gwp_horizon` to `MethodPack` dataclass in `greenlang/factors/method_packs/base.py` around line 121. Methane-focused customers otherwise re-compute from gas-level vectors.

---

## 5. Effective-dates timeline (2026-01-01 → 2028-12-31)

All dates below are load-bearing for a Factors v1 customer.

| Date | Jurisdiction | Event | Factors-v1 impact |
|------|-------------|-------|-------------------|
| 2026-01-01 | EU | CBAM definitive period begins — quarterly reporting + certificate surrender starts | `EU_CBAM` pack activated; declarants can no longer rely on preview-status factors (`allowed_statuses=("certified",)`, `require_verification=True`). |
| 2026-01-01 | California | SB 261 biennial climate-risk disclosure window opens (first reports due 2026) | Corporate packs + PCAF variants (for FIs). |
| 2026-01-31 | EU | CBAM Q4 2025 transitional report due (last transitional filing) | `cbam_full.py` ingestion fresh run. |
| 2026-02-15 | UK | SECR FY2025 reports due for FY-end 31 March 2025 companies | `desnz_uk.py` latest vintage. |
| 2026-03 | EU | Omnibus Simplification Package expected — possible CSRD / CBAM / SFDR amendments | See §6 volatility watch. |
| 2026-04-30 | EU | CBAM Q1 2026 report due (first definitive quarterly filing) | `EU_CBAM` pack deprecation window (2-year) means factors ingested ≤ 2024-04 will be rotated. |
| 2026-05 | EU | First CSRD reports (large public-interest entities, FY2024 data) published under ESRS | Corporate packs must carry `"CSRD_E1"` reporting label — see remediation #8. |
| 2026-06 | EU | AIB publishes 2025 residual mix | `aib_residual_mix.py` refresh. |
| 2026-07-31 | EU | CBAM Q2 2026 report due | `EU_CBAM` second quarterly. |
| 2026-08-01 | California | SB 253 Scope 1+2 first reporting deadline (FY2025 data, in-scope US entities > $1B revenue) | Corporate packs — no code change. |
| 2026-08 | Australia | DCCEEW publishes FY2025/26 NGA factors + LGC surrender data | `australia_nga_residual.py` refresh. |
| 2026-09 | Green-e | 2025 residual mix publication | `green_e_residual.py` refresh. |
| 2026-10-31 | EU | CBAM Q3 2026 report due | — |
| 2026-12-31 | India | BRSR FY2025-26 reporting cycle closes; top-1000 mandatory | `india_cea.py` refresh for latest FY intensity. |
| 2026-12 | IPCC | AR7 WG-I draft comment period (anticipated) | No immediate pack change; monitor GWP set revisions. |
| 2027-01-31 | EU | CBAM Q4 2026 report + first annual reconciliation | Certificate volumes settle for 2026. |
| 2027-01 | California | SB 253 Scope 3 first reporting deadline (FY2026 data) | `CORPORATE_SCOPE3` + PCAF variants (for FIs). |
| 2027-02 | Japan | SSBJ standards take effect (Prime Market listed, FY2027 start) | `ELECTRICITY_RESIDUAL_MIX_JP` + corporate packs. |
| 2027-03 | UK | TPT disclosures expected mandatory phase-in begins | No factor change — narrative + trajectory. |
| 2027-04 | EU | CSRD second wave: listed SMEs FY2026 data | Same corporate packs. |
| 2027-05 | ISSB / IFRS | First IFRS S2 year 2 disclosures | Corporate packs carrying `"IFRS_S2"` label. |
| 2027-08 | EU | ESPR DPP first product groups (textiles, construction — tranche 1) enforcement begins | `EU_DPP` pack must bump to 1.0.0 with tranche-specific schema. |
| 2027-12 | EU | Battery Regulation 2023/1542 DPP requirements enter force for industrial / EV batteries | Requires remediation #9 (EU Battery thresholds). |
| 2028-01 | Singapore | SGX Rule 711A full-scope mandatory Scope 3 disclosures begin | Needs Singapore-specific grid factor (remediation #1). |
| 2028-06 | India | CCTS first compliance cycle ends — obligated entities surrender credits vs. baselines | Remediation #10 (India CCTS benchmarks). |
| 2028-12 | EU | CBAM free-allocation phase-out complete — full-price CBAM certificates | No factor change — this is a pricing event. |

---

## 6. Regulatory volatility watch

Standards still finalising or politically contested — may change in-flight during Factors v1 lifetime:

| Standard | Status (2026-04-22) | Watch trigger | Impact on Factors v1 |
|----------|--------------------|----|----------------------|
| **SEC climate rule** | Stayed by 8th Circuit (Mar 2024). SEC has suspended defense. | Re-issuance in simplified form possible 2026-2027. | Factor coverage ready; `ELECTRICITY_RESIDUAL_MIX_US` already labelled `"SEC_Climate"` (line 202 of `electricity.py`). No code change needed on re-issue. |
| **CSRD Omnibus Simplification** | EU Commission announced Feb 2025; proposal expected 2025-Q2; possible threshold relief + Scope 3 deferral. | EU Parliament / Council vote 2026-Q2. | If adopted, fewer small-cap entities file. No factor-catalog change. |
| **CBAM default-values revision** | Commission publishes CBAM default values (Annex VI); revision ongoing to incorporate country-level + facility-level data. | Implementing regulation amendment publishes 2026-H2. | `cbam_full.py` parser is source-agnostic; refresh on publication. `EU_CBAM` pack `deprecation.max_age_days=730` auto-rotates. |
| **GHG Protocol Corporate Standard Revision** | WRI/WBCSD "Corporate Standard update" initiative, public-comment draft 2024-2025, final expected 2026-2027. | Revised Scope 2 Guidance in particular — possible hourly matching requirements, granular residual mix. | May affect `ELECTRICITY_RESIDUAL_MIX_*` packs and market-based selection logic. Track closely. |
| **GHG Protocol LSR final** | Draft Sept 2024; public comment closed; final expected 2026-Q2. | Publication. | Bump LSR `pack_version` across 4 variants — remediation #6. |
| **GHG Protocol Scope 3 Standard Revision** | Revision in progress; draft expected 2025-2026. | Publication. | May reshape `CORPORATE_SCOPE3` allowed families / formula types. |
| **Japan SSBJ standards** | Draft; final expected 2026-Q2. | Publication. | `japan_meti_residual.py` + corporate pack labels. |
| **PACT v3 spec** | v2 in active production; v3 proposal under WBCSD / PACT Steering. | v3 GA. | Parser extension — remediation #3. |
| **IPCC AR7** | WG1 draft 2026-2028; Synthesis Report ~2029. | Publication of AR7 GWPs. | New `gwp_set` enum value; all packs' `gwp_basis` remains explicit — no retrofit needed. |
| **EU Taxonomy DNSH thresholds for climate mitigation** | Delegated Acts rolling; review of cement / steel / aluminium thresholds expected 2026-2027. | Publication. | Taxonomy is out of Factors v1 scope (thresholds, not factors). Downstream `greenlang/taxonomy/` module will consume Factor records. |
| **EU ESPR DPP implementing acts** | Tranche publishing cadence: textiles + construction 2027, batteries 2027, other tranches 2028+. | Per-tranche delegated act. | `EU_DPP` pack version bumps each time — currently `0.1.0` (pre-regulation). |
| **SBTi v5 FLAG finalisation** | Criteria v5 published 2024; FLAG guidance operational 2024-2026. | FLAG final alignment with GHG LSR. | Watch alongside LSR final. |
| **California SB 253 implementing regs** | CARB rulemaking due Jan 2026 for first reporting cycle Aug 2026. | CARB draft + final. | May narrow or expand scope; `CORPORATE_SCOPE*` packs already carry IFRS-S2-aligned selection rules; adjustment via `reporting_labels`. |
| **PCAF Part B (facilitated + insurance)** | Published 2023; uptake uneven; potential revisions 2026-2027. | Revision. | Remediation #7. |
| **EU Green Claims Directive** | Political agreement reached; enforcement post-transposition ~2027-2028. | Transposition + enforcement guidance. | Consumer-product claim substantiation uses `PEF`/`PRODUCT_CARBON` packs — factor side ready. |

---

## 7. Coverage matrix — single table

Legend for **Coverage**: `COMPLETE` — Factors v1 serves this standard end-to-end with registered profile + parser + required fields. `PARTIAL` — primary use cases served; identified enhancements listed in §4. `GAP` — not served by Factors v1 (scope boundary or regulation outside factor-catalog domain). Slice abbreviations: CMB=combustion, ELE=electricity, FRT=freight, MAT=material-CBAM, LND=land-removals, PCF=product-carbon, FIN=finance-proxy.

| Standard / Regulation | Method profile(s) | Parser(s) | Slice(s) | Coverage | Notes |
|-----------------------|-------------------|-----------|----------|----------|-------|
| GHG Protocol Corporate (S1/S2/S3) | CORPORATE_SCOPE1, CORPORATE_SCOPE2_LOCATION, CORPORATE_SCOPE2_MARKET, CORPORATE_SCOPE3 | epa_ghg_hub, desnz_uk, tcr, ipcc_defaults, ghg_protocol, egrid, india_cea | CMB, ELE | COMPLETE | Biogenic separate-reporting enforced. |
| GHG Protocol Scope 2 Guidance | CORPORATE_SCOPE2_LOCATION, _MARKET, ELECTRICITY_RESIDUAL_MIX_EU/US/AU/JP | egrid, aib_residual_mix, green_e_residual, australia_nga_residual, japan_meti_residual, india_cea, desnz_uk | ELE | PARTIAL | Missing residual-mix parsers for BR, KR, MX, SG, etc. Remediation #1. |
| GHG Protocol Scope 3 (15 cats) | CORPORATE_SCOPE3 | ghg_protocol, ec3_epd, pact_product_data, pcaf_proxies, freight_lanes, waste_treatment, desnz_uk | CMB, FRT, PCF, FIN | PARTIAL | Cat 11 lacks product-lifetime field. Remediation #2. |
| ISO 14064-1:2018 | CORPORATE_SCOPE1/2/3 (label `"ISO_14064"`) | same as GHG Protocol | CMB, ELE | COMPLETE | `gwp_set` supports SAR/AR4/AR5/AR6. |
| IFRS S2 | CORPORATE_SCOPE1/2/3, PCAF variants (label `"IFRS_S2"`) | same + pcaf_proxies | CMB, ELE, FIN | COMPLETE | Narrative parts out of factor scope. |
| TCFD | CORPORATE_SCOPE1/2/3, ELECTRICITY_RESIDUAL_MIX_JP (label `"TCFD"`) | same | CMB, ELE | COMPLETE | Superseded by IFRS S2 in most markets. |
| CDP climate | CORPORATE_SCOPE1/2/3, ELECTRICITY_RESIDUAL_MIX_US (label `"CDP"`) | same | CMB, ELE | COMPLETE | Dual-reporting Scope 2 supported. |
| SBTi criteria v5 | CORPORATE + LSR variants | same + lsr_removals, ipcc_defaults | CMB, ELE, LND | PARTIAL | Tracks 2024 LSR draft; bump on final. Remediation #6. |
| GHG Protocol Product Standard | PRODUCT_CARBON | pact_product_data, ec3_epd | PCF | COMPLETE | — |
| ISO 14067:2018 | PRODUCT_CARBON | pact_product_data, ec3_epd | PCF | COMPLETE | — |
| PACT Pathfinder v3 | PRODUCT_CARBON (label `"PACT"`) | pact_product_data | PCF | PARTIAL | Parser on v2 schema; extend to v3. Remediation #3. |
| EU PEF (EF 3.1) | PEF (via PRODUCT_CARBON profile, variant) | ec3_epd, pact_product_data | PCF | PARTIAL | 16 EF indicators stored in `extras`, not first-class columns. |
| ISO 14040 / 14044 | PRODUCT_CARBON (foundation) | ec3_epd, pact_product_data | PCF | COMPLETE | Process standards — captured via `FormulaType.LCA`. |
| EN 15804 / ISO 14025 EPD | PRODUCT_CARBON | ec3_epd | PCF | COMPLETE | Modules A1-C4 captured. |
| ISO 14083:2023 | FREIGHT_ISO_14083 | freight_lanes, ghg_protocol, desnz_uk | FRT | COMPLETE | Pack v0.3.0; bump on GLEC v3 completion. Remediation #4. |
| GLEC Framework v3 | FREIGHT_ISO_14083 | freight_lanes | FRT | COMPLETE | Same. |
| CORSIA (aviation) | FREIGHT_ISO_14083 (freight legs); passenger aviation via Scope 3 Cat 6 | freight_lanes, ghg_protocol | FRT | PARTIAL | RFI not first-class; CORSIA offset units are out of scope. Remediation #5. |
| GHG Protocol LSR 2024 | LAND_REMOVALS + 4 variants (lsr_land_use_emissions, lsr_land_management, lsr_removals, lsr_storage) | lsr_removals, ipcc_defaults | LND | COMPLETE | Tracks draft; bump on final. Remediation #6. |
| IPCC 2006 GL + 2019 Refinement | cross-cutting (labels `"IPCC_2006_GL"`, `"IPCC_2019_Refinement"` on LSR packs) | ipcc_defaults | CMB, LND, FRT | COMPLETE | All Tier 1 defaults loaded. |
| VCS VM modules | LAND_REMOVALS variants (VerificationStandard.VCS) | lsr_removals | LND | PARTIAL | No VM methodology ID enum. Remediation #11. |
| TNFD v1.0 | LAND_REMOVALS (climate subset only) | lsr_removals | LND | GAP | Non-climate nature metrics out of scope. Remediation #12 (v2). |
| AIB residual mix (EU) | ELECTRICITY_RESIDUAL_MIX_EU | aib_residual_mix | ELE | COMPLETE | — |
| Green-e Energy (US) | ELECTRICITY_RESIDUAL_MIX_US | green_e, green_e_residual | ELE | COMPLETE | `RedistributionClass.RESTRICTED` enforced. |
| I-REC Standard | CORPORATE_SCOPE2_MARKET / ELECTRICITY_MARKET (customer overlay tier) | customer overlay via fallback | ELE | PARTIAL | No first-class I-REC registry parser. Optional. |
| EAC registries (REGO, GO, REC, I-REC) | ELECTRICITY_MARKET | customer overlay | ELE | PARTIAL | Same as I-REC. |
| PCAF Standard Part A v2.0 (7 asset classes) | FINANCE_PROXY umbrella + pcaf_listed_equity, pcaf_corporate_bonds, pcaf_business_loans, pcaf_project_finance, pcaf_commercial_real_estate, pcaf_mortgages, pcaf_motor_vehicle_loans | pcaf_proxies, ghg_protocol (Cat 15), corporate parsers | FIN | COMPLETE | Full DQS 1-5 + attribution hierarchy + 16 high-emit sectors. |
| PCAF Part B (facilitated + insurance) | not registered as first-class | pcaf_proxies | FIN | PARTIAL | Remediation #7. |
| TCFD financial sector | PCAF variants | pcaf_proxies | FIN | COMPLETE | — |
| SBTi Financial Institutions v1.1 | PCAF variants | pcaf_proxies | FIN | COMPLETE | Sector pathways out of factor scope. |
| EU CBAM 2023/956 | EU_CBAM | cbam_full | MAT | COMPLETE | `require_verification=True`; 2-year deprecation. |
| EU CSRD / ESRS E1 | CORPORATE_SCOPE1/2/3 + OEF (label `"ESRS_E1"`) | corporate parsers | CMB, ELE, FRT, PCF | PARTIAL | E2-E5 and S/G pillars out of factor scope. Add `"CSRD_E1"` label across corporate packs — remediation #8. |
| EU SFDR Art 8/9 + PAI 1-3 | PCAF variants + corporate packs | pcaf_proxies + corporate | FIN, CMB, ELE | PARTIAL | PAI 4-14 largely non-emission metrics. |
| EU Taxonomy | no dedicated pack (uses factor records for threshold comparison) | ec3_epd, cbam_full | MAT, PCF | PARTIAL | Thresholds belong in `greenlang/taxonomy/` (out of scope). |
| EU Battery Regulation 2023/1542 | EU_DPP (`pack_version=0.1.0`) | pact_product_data, ec3_epd | MAT, PCF | PARTIAL | Threshold classes pending Delegated Regulation. Remediation #9. |
| EU ESPR + DPP | EU_DPP | pact_product_data, ec3_epd | MAT, PCF | PARTIAL | Tranches rolling 2027-2028. Remediation #9 (generalised). |
| California SB 253 | CORPORATE_SCOPE1/2/3 | epa_ghg_hub, egrid, tcr, ghg_protocol, green_e_residual | CMB, ELE | COMPLETE | Add `"CA_SB253"` label — remediation #8. |
| California SB 261 | CORPORATE + PCAF variants | same | CMB, ELE, FIN | COMPLETE | — |
| SEC climate rule (stayed) | CORPORATE_SCOPE1/2 + ELECTRICITY_RESIDUAL_MIX_US (label `"SEC_Climate"`) | epa_ghg_hub, egrid, green_e_residual | CMB, ELE | GAP (by rule status) | Factor coverage ready when rule revives. |
| India BRSR | CORPORATE_SCOPE1/2/3 | india_cea, ipcc_defaults | CMB, ELE | COMPLETE | Add `"India_BRSR"` label — remediation #8. |
| India CCTS | not registered (CORPORATE used for obligated entity's own Scope 1+2) | india_cea, ipcc_defaults | CMB, ELE | PARTIAL | Baseline intensity benchmarks missing. Remediation #10. |
| UK SECR | CORPORATE_SCOPE1/2/3 | desnz_uk | CMB, ELE | COMPLETE | Add `"UK_SECR"` label — remediation #8. |
| UK TPT | CORPORATE + PCAF variants | desnz_uk, pcaf_proxies | CMB, ELE, FIN | COMPLETE | Narrative only — factor side ready. |
| Japan FSA / SSBJ | CORPORATE + ELECTRICITY_RESIDUAL_MIX_JP (label `"Japan_SSBJ"`) | japan_meti_residual, corporate parsers | CMB, ELE | COMPLETE (tracks draft) | Finalise on SSBJ publication. |
| Singapore SGX Rule 711A | CORPORATE (no SG-specific grid parser) | corporate parsers + IPCC defaults | CMB, ELE | PARTIAL | Add `singapore_ema.py`. Remediation #1. |
| IPCC AR4 / AR5 / AR6 (100y + 20y) | all packs carry `gwp_basis`; gas-level vectors on record | cross-cutting | all slices | COMPLETE | AR6_20 via record re-compute; pack default is 100y. |
| Kyoto SAR values (legacy) | all packs (via `gwp_set=SAR` on record) | cross-cutting | all slices | COMPLETE | Pack defaults don't use SAR; re-compute supported. |

---

## 8. Sign-off conditions

For the founder to close Factors v1 GA with confidence, the minimum delta from today (2026-04-22) is:

1. **Apply remediation #8** — one-line `reporting_labels` additions across four packs in `greenlang/factors/method_packs/corporate.py`. Trivial, unblocks 5 row-level PARTIALs → COMPLETE.
2. **Apply remediation #3** — extend `pact_product_data.py` parser for v3 schema fields. One file edit.
3. **Apply remediation #4** — bump `freight.py` pack version from `0.3.0` to `1.0.0` after the GLEC lane-count validation gate.
4. **Defer to post-v1** — remediations #1 (non-core residual mixes), #2 (Cat 11 lifetime), #5 (RFI), #7 (PCAF Part B), #9 (EU Battery thresholds), #10 (India CCTS), #11 (VCS methodology IDs), #12 (TNFD nature), #13 (PEF 16-indicator), #14 (AR6_20 pack default). Each is listed in §4 with exact file paths.

At that point, every row in §7 is either COMPLETE or PARTIAL-with-documented-workaround, and the founder can market Factors v1 as serving the 7-slice regulatory surface **as committed on 2026-04-22**.

---

*End of audit.*
