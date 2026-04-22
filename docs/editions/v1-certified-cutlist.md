# GreenLang Factors v1.0 Certified Edition — Cut-List & Slice Promotion Manifest

**Document owner:** GL-FormulaLibraryCurator
**Status:** AUTHORITATIVE — binds release engineering, legal sign-off, and GTM
**Founder decision date:** 2026-04-22
**Scope:** v1.0 Certified covers ALL 7 method packs, ALL jurisdictions, ALL Premium Data Pack SKUs
**Delivery strategy:** 7 thin vertical slices, each promoted as its own Certified edition
**Binding artifacts:**
- `greenlang/factors/data/source_registry.yaml` (G1-G6 source governance)
- `greenlang/factors/method_packs/registry.py` (process-wide pack registry)
- `greenlang/factors/quality/release_signoff.py` (9-point promotion gate)
- `greenlang/data/canonical_v2.py::MethodProfile` (10 profile enums)

---

## 0. Promotion gate contract (applies to every slice)

The release sign-off pipeline in `greenlang/factors/quality/release_signoff.py` produces 9 checklist items (S1-S9) per edition. An edition is marked `ready_for_release == True` only when every item with `severity == "required"` passes. The founder's cut-list locks the bar at:

| Item | Label | Required |
|------|-------|----------|
| S1 | Q1-Q6 QA gates pass for all factors (`total_failed == 0 AND total_factors > 0`) | YES |
| S2 | No unresolved duplicate pairs (`human_review == 0` on DedupReport) | YES |
| S3 | Cross-source consistency reviewed (`total_reviews == 0` on ConsistencyReport) | YES |
| S4 | Changelog reviewed and approved (`manifest.changelog` truthy + human flag) | YES |
| S5 | Methodology lead signed off | YES |
| S6 | Legal confirmed source licenses | YES |
| S7 | Regression test (`compare_editions`) passed | YES (upgraded from recommended for v1) |
| S8 | Load test passed (p95 < 500ms) | YES (upgraded from recommended for v1) |
| S9 | Gold-eval precision@1 >= 0.85 | YES |

`approve_release(force=False)` will `raise ValueError` if any required item fails. v1 policy: `force=False` always. No forced promotions on any slice.

### DQS + FQS component minima

DQS is stored per factor as a 5-component dict (`temporal`, `geographical`, `technological`, `representativeness`, `methodological`); 1-5 Pedigree-matrix semantics (see `greenlang/factors/quality/dedup_engine.py::_dqs_score`). FQS is the composite 0-100 score surfaced to buyers.

| Tier | Min temporal | Min geographical | Min technological | Min verification (representativeness) | Min completeness (methodological) | Min composite FQS |
|------|--------------|------------------|-------------------|---------------------------------------|-----------------------------------|-------------------|
| **Certified** | 3 | 3 | 3 | 3 | 3 | **75 / 100** |
| **Preview** | 2 | 2 | 2 | 2 | 2 | **50 / 100** |
| **Connector-only** | n/a | n/a | n/a | n/a | n/a | no minimum, but source `license_class IN (restricted, commercial_connector)` and `connector_only == true` MUST be enforced at the serving layer |

Notes:
- CBAM slice overrides `allowed_statuses=("certified",)` + `require_verification=True` in `EU_CBAM.selection_rule` — Preview factors are forbidden in CBAM declarations.
- PEF (EU Product Environmental Footprint) also carries `require_verification=True` and `allowed_statuses=("certified",)` per `product_lca_variants.py`.
- `allowed_statuses=("certified",)` on CORPORATE_SCOPE1 and CORPORATE_SCOPE2_LOCATION means Preview factors cannot route through those profiles at all.

### Rollback protocol

Every slice promotion writes an immutable edition id of the form `vYYYY.Q<n>-<slice>`. If a promotion fails post-deploy (e.g. gold-eval regresses on a later slice, or legal raises a license objection), the `edition_manifest.py` catalog demotes the slice's edition to `deprecated` and the prior Certified edition becomes the active one. There is no in-place mutation: rollback is a pointer flip.

---

## 1. Slice 1 — Electricity  (edition tag: `2027.Q1-electricity`, Week 4)

### 1.1 Sales one-liner
"Location-based and market-based Scope 2 factors for every enterprise grid in India, EU, UK, US, AU, JP, Canada — with residual mix, Green-e + I-REC certificate handling, and AIB/NGA/METI coverage baked in. The fastest path to Scope 2 reporting that survives audit."

### 1.2 Source IDs (from `source_registry.yaml`)

| source_id | display_name | license_class | redistribution | connector_only | cadence |
|-----------|--------------|---------------|----------------|----------------|---------|
| `egrid` | eGRID | `public_us_government` | allowed | false | annual |
| `epa_hub` | EPA GHG Emission Factors Hub | `public_us_government` | allowed | false | quarterly |
| `desnz_ghg_conversion` | DESNZ GHG conversion factors | `uk_open_government` | allowed | false | annual |
| `defra_conversion` | DEFRA environmental reporting factors | `uk_open_government` | allowed | false | annual |
| `australia_nga_factors` | Australian NGA Accounts Factors | `open` | allowed | false | annual |
| `japan_meti_electric_emission_factors` | Japan METI Electric Utility EF | `open` | allowed | false | annual |
| `green_e_residual_mix` | Green-e Energy Residual Mix | `restricted` | denied | false | annual |
| `green_e_residual` | Green-e residual mix (legacy row) | `commercial_connector` | denied | true | annual |
| `electricity_maps` | Electricity Maps | `commercial_connector` | denied | true | daily |
| `iea` | IEA statistics / factors | `commercial_connector` | denied | true | annual |
| `greenlang_builtin` | GreenLang curated built-in factors | `greenlang_terms` | allowed | false | on_release |

**India CEA:** There is no `india_cea` row in `source_registry.yaml` today. The parser exists at `greenlang/factors/ingestion/parsers/india_cea.py`, but the registry row must be added before S6 (Legal confirmed source licenses) can pass for this slice. Action item on the curator: add `india_cea_co2_baseline` row pinned to `license_class: open` (CEA baseline database is Govt-of-India open publication, v20 most recent) before Week 3.

### 1.3 Versions pinned

| source_id | pinned version / vintage |
|-----------|---------------------------|
| `egrid` | **eGRID2023** (published 2025-01; 26 subregions) |
| `epa_hub` | **April-2025** release of GHG EF Hub |
| `desnz_ghg_conversion` | **2025** conversion factors set |
| `defra_conversion` | **2025** reporting factors |
| `australia_nga_factors` | **NGA August 2025** (FY2024-25) |
| `japan_meti_electric_emission_factors` | **METI FY2023 emissions factors (published 2025)** |
| `green_e_residual_mix` | **Green-e 2023 Residual Mix (published 2025-Q3)** |
| `india_cea_co2_baseline` | **CEA Version 20.0 (2024 publication, FY2022-23 data)** — all 5 regional grids |
| `aib_residual_mix_eu` | **AIB European Residual Mixes 2024** (publication 2025-Q2) |
| `greenlang_builtin` | `2027.Q1-electricity` snapshot |

### 1.4 License class summary

- **Redistributable bulk:** eGRID, EPA Hub, DESNZ, DEFRA, NGA, METI, India CEA (pending registry row), greenlang_builtin, AIB (public PDF; parsed to our schema with citation)
- **Restricted, certified in bulk with attribution watermark:** `green_e_residual_mix` (`license_class: restricted`) — included in Certified edition bulk export under the Premium Electricity Pack SKU only; public tier gets connector access via `green_e_residual`
- **Connector-only, never in bulk:** `electricity_maps`, `iea`

### 1.5 Certified factor count target

**~800-1,200 certified electricity factors.** Breakdown:
- eGRID subregion (26) x {CO2, CH4, N2O, CO2e} x {lb/MWh, kg/MWh} = ~208
- eGRID state roll-up (51) x 4 gases = ~204
- AIB residual mix EU27+EEA (30 countries) x {production mix, residual mix} = ~60
- DESNZ UK factors (grid, supplier-specific, transmission & distribution, WTT) = ~40
- India CEA (5 regional grids + all-India + 30 states x combined margin + operating margin + build margin) = ~120
- NGA (NEM 5 + WA + NT + national) x {Scope 2, Scope 3 loss factor} = ~16
- METI (10 utility areas + national basic + national adjusted) x 2 modes = ~24
- Green-e residual (16 eGRID subregions covered + national US + Canada provinces) = ~40
- Marginal emission factors (where parser supports) = ~40
- Canada provincial (from Green-e data + ECCC NIR spillover) = ~28
- Country-level averages (fallback tier for rest-of-world) = ~100
- Historical vintages (eGRID2021 + 2022 retained for restatement support) = ~120

### 1.6 Gold-eval coverage required

Gold set target for this slice: **at least 180 activity->factor pairs**, distributed:
- 45 pairs India CEA (commercial building, industrial plant, data centre, EV fleet charging, misc)
- 45 pairs eGRID subregion (manufacturing plants across CAMX, MROW, RFCE, SRVC, NYUP, NWPP at minimum)
- 35 pairs AIB EU (office electricity across DE, FR, NL, IT, ES, PL, SE, IE)
- 25 pairs UK DESNZ (commercial, industrial, T&D, WTT variants)
- 15 pairs market-based with REC/GO surrender
- 15 pairs residual mix fallback (when no contractual instrument)

### 1.7 Acceptance gate

- Top-1 match precision >= **0.85** (S9 floor)
- Cross-source deviation on overlapping country: DESNZ UK vs. AIB UK grid average must differ by **<= 5%**; eGRID US national vs. EPA Hub US national vs. IEA US (connector) must agree within **5%**
- Zero license-class violations: every factor with `license_class in (restricted, commercial_connector)` must have `redistribution_allowed == false` enforced in bulk export filter
- All 9 signoff items pass, including S7 (regression: no silent factor value drift >5% on legacy eGRID2021 carry-over) and S8 (p95 lookup < 500 ms under 1000 rps synthetic load)

### 1.8 Method profiles unlocked

From `greenlang/factors/method_packs/electricity.py` + `corporate.py`:
- `ELECTRICITY_LOCATION` (registered to `MethodProfile.CORPORATE_SCOPE2_LOCATION`)
- `ELECTRICITY_MARKET` (registered to `MethodProfile.CORPORATE_SCOPE2_MARKET`)
- `ELECTRICITY_RESIDUAL_MIX_EU` / `_US` / `_AU` / `_JP` — regional variants routed via `get_residual_mix_pack(country)`
- `CORPORATE_SCOPE2_LOCATION`, `CORPORATE_SCOPE2_MARKET` — GHG Protocol umbrellas (confirm registration via `registry.registered_profiles()` on import)

### 1.9 Downstream app unlock

- **Scope Engine** (corporate scope 1/2 reporting) — full unlock (v1 Scope 2 production mode)
- **Comply / CSRD** — partial unlock (E1-5 energy mix + E1-6 gross Scope 2 LB and MB)
- **CBAM** — partial unlock (CBAM needs indirect emissions factor for electricity imports; this slice seeds the `FactorFamily.GRID_INTENSITY` rows CBAM pulls)

### 1.10 GTM hook

- **Starter (free):** eGRID subregion + DESNZ + DEFRA + NGA + METI + India CEA + AIB residual mix EU — all redistributable public factors, ~900 factors
- **Premium Electricity Pack (paid):** adds Green-e Residual Mix US/CA (restricted license → licensed bulk), electricity_maps hourly grid intensity (connector), IEA country averages (connector)
- **Enterprise add-on:** supplier-specific REC/GO/I-REC surrender tracking, PPA contract ingestion, dual-reporting Scope 2 LB+MB views

### 1.11 Rollback plan

If `2027.Q1-electricity` regression fails at S7 or gold-eval drops below 0.85 post-deploy: demote to `deprecated` and re-activate `2026.Q4-electricity-preview` (the last Preview tier electricity edition). Customers on `X-GreenLang-Edition: 2027.Q1-electricity` fall through to the previous active Certified pointer automatically via the edition manifest catalog.

---

## 2. Slice 2 — Combustion  (edition tag: `2027.Q2-combustion`, Week 6)

### 2.1 Sales one-liner
"Scope 1 combustion, done. Every fossil + biogenic fuel, IPCC-aligned defaults with DESNZ, EPA, and TCR jurisdictional overlays, LHV + HHV + density + oxidation + fossil/biogenic split — the same numbers your auditor already trusts."

### 2.2 Source IDs

| source_id | license_class | redistribution | cadence |
|-----------|---------------|----------------|---------|
| `epa_hub` | `public_us_government` | allowed | quarterly |
| `desnz_ghg_conversion` | `uk_open_government` | allowed | annual |
| `defra_conversion` | `uk_open_government` | allowed | annual |
| `tcr_grp_defaults` | `registry_terms` | **NOT allowed** | annual |
| `ghgp_method_refs` | `wri_wbcsd_terms` | **NOT allowed** | ad_hoc |
| `iea` | `commercial_connector` | denied | annual (connector-only) |
| `greenlang_builtin` | `greenlang_terms` | allowed | on_release |

**IPCC defaults:** parser at `greenlang/factors/ingestion/parsers/ipcc_defaults.py` — registry row must be added as `ipcc_2006_nggi` with `license_class: public_international` (IPCC NGGI 2006 + 2019 Refinement is public) before Week 5.

**India EF:** `india_cea_co2_baseline` from Slice 1 registry row is reused — CEA publishes stationary combustion defaults alongside grid factors.

### 2.3 Versions pinned

| source_id | pinned version |
|-----------|----------------|
| `ipcc_2006_nggi` | 2006 Guidelines + 2019 Refinement |
| `epa_hub` | April-2025 release Table 1-5 stationary + Table 8 mobile |
| `desnz_ghg_conversion` | 2025 fuel combustion + T&D |
| `defra_conversion` | 2025 full methodology paper |
| `tcr_grp_defaults` | 2024 TCR GRP Default Emission Factors |
| `ghgp_method_refs` | GHG Protocol Corporate Standard (rev 2015) + Scope 2 Guidance (2015) |

### 2.4 License class summary

- **Redistributable:** IPCC, EPA Hub, DESNZ, DEFRA, India CEA combustion set, greenlang_builtin
- **Attribution + registry-terms restriction:** `tcr_grp_defaults` — included in Certified edition only with full citation block and GHGP methodology excerpt; legal sign-off must confirm 2024 TCR terms allow derived factor export
- **GHG Protocol references:** `ghgp_method_refs` is a methodology reference, not a factor set — used only for audit-text derivation, not numeric values
- **Connector only:** `iea` (country-level oil product factors for aviation, shipping, residual fuel oil)

### 2.5 Certified factor count target

**~400-600 certified combustion factors.** Breakdown:
- Stationary fossil (coal bituminous, sub-bituminous, lignite, anthracite, coke; natural gas pipeline-quality, LNG; fuel oil #2/#4/#6; LPG; propane; kerosene) across {IN, EU27 aggregate, UK, US, global-default} x {CO2, CH4, N2O, CO2e} x {LHV-basis, HHV-basis} = ~240
- Stationary biomass + biofuels (wood pellets, wood chips, bagasse, agricultural residues, biodiesel, ethanol, biogas) x jurisdictions x {fossil CO2, biogenic CO2, CH4, N2O} = ~120
- Mobile fuels (gasoline E0/E10/E85, on-road diesel, off-road diesel, jet A1/kerosene, marine diesel oil, heavy fuel oil, CNG vehicle, LPG vehicle) x jurisdictions x {CO2, CH4, N2O, WTT, TTW} = ~180
- Physical properties (density, LHV, HHV, carbon content, oxidation factor) per fuel x jurisdiction = ~100

### 2.6 Gold-eval coverage required

**At least 90 activity->factor pairs**, spanning Scope 1 + mobile combustion:
- 20 pairs natural gas stationary (commercial boilers, industrial furnaces) across IN/UK/US/DE
- 15 pairs diesel mobile (fleet vehicles, backup gensets)
- 15 pairs coal industrial (sub-bituminous, bituminous) US + IN
- 10 pairs biomass boiler (wood pellet, bagasse) with biogenic split check
- 10 pairs LPG/propane commercial heating
- 10 pairs marine bunker (HFO + MDO + LSFO)
- 10 pairs aviation jet fuel (domestic + international splits)

### 2.7 Acceptance gate

- Top-1 match precision >= **0.85**
- Cross-source deviation on natural gas stationary: EPA Hub vs. DESNZ vs. IPCC default — must agree within **5%** after LHV/HHV basis normalization
- Zero license violations: TCR factor values must never appear in a bulk CSV/Parquet export flagged as `license_class == greenlang_terms` (must stay tagged as `registry_terms` with attribution block)
- Unit consistency: every factor must declare `basis IN {LHV, HHV}` — missing basis = promotion-blocking Q6 QA gate failure

### 2.8 Method profiles unlocked

- `CORPORATE_SCOPE1` (registered to `MethodProfile.CORPORATE_SCOPE1`) — full stationary + mobile combustion unlock; `allowed_families=(EMISSIONS, HEATING_VALUE, REFRIGERANT_GWP, OXIDATION, CARBON_CONTENT)`
- `CORPORATE_SCOPE3` (partial — Category 3 fuel and energy related activities WTT)

### 2.9 Downstream app unlock

- **Scope Engine** Scope 1 production mode (was partial after Slice 1)
- **Comply / CSRD** E1-6 gross Scope 1 emissions unlock
- **CBAM** combustion inputs for the direct-emissions embedded calculation (prepares for Slice 4)
- **SB 253** (California CSRDA) Scope 1 reporting

### 2.10 GTM hook

- **Starter (free):** IPCC + EPA + DESNZ + India CEA combustion factors, ~350 factors
- **Premium Combustion Pack (paid):** TCR GRP defaults (with license), IEA country oil products (connector), historical vintages for 10-year restatement support
- **Enterprise:** custom fuel blends, refinery-specific factors via supplier-specific slots

### 2.11 Rollback plan

If promotion fails: fall back to `2027.Q1-combustion-preview`. Legacy Scope 1 factors already live in the `greenlang_builtin` built-in set (v0.9 edition) — those stay active as the catch-all tier.

---

## 3. Slice 3 — Freight  (edition tag: `2027.Q3-freight`, Week 8)

### 3.1 Sales one-liner
"ISO 14083 and GLEC-compliant freight emissions across road, rail, sea, air, and inland waterway — with utilization, empty running, refrigeration, WTW+TTW labelling, and a lane library covering the six shipping corridors that dominate every enterprise Scope 3 inventory."

### 3.2 Source IDs

| source_id | license_class | redistribution | cadence |
|-----------|---------------|----------------|---------|
| `desnz_ghg_conversion` | `uk_open_government` | allowed | annual |
| `epa_hub` | `public_us_government` | allowed | quarterly |
| `defra_conversion` | `uk_open_government` | allowed | annual |
| `greenlang_builtin` | `greenlang_terms` | allowed | on_release |

**GLEC Framework:** parser at `greenlang/factors/ingestion/parsers/freight_lanes.py` consumes the Smart Freight Centre GLEC v3.0 defaults. A registry row `glec_framework` with `license_class: smart_freight_terms` is required before Week 7 (redistribution requires Smart Freight Buyer Group membership attribution — legal must confirm).

**EcoTransIT / ETSI / IMO:** cross-checks only; not authoritative for certified set.

### 3.3 Versions pinned

| source_id | pinned version |
|-----------|----------------|
| `glec_framework` | GLEC v3.0 (2023) + 2024 addendum |
| `desnz_ghg_conversion` | 2025 freighting goods tables |
| `epa_hub` | 2025 Table 10-13 (transport + freight) |
| `iso_14083_method` | ISO 14083:2023 (methodology reference) |

### 3.4 License class summary

- **Redistributable:** EPA, DESNZ, DEFRA
- **GLEC:** attribution required, Smart Freight Buyer Group terms apply to derived bulk — legal sign-off item (S6)
- **No connector-only sources** in this slice

### 3.5 Certified factor count target

**~500-800 certified freight factors.** Breakdown:
- Road: rigid (<3.5t, 3.5-7.5t, 7.5-17t, 17-32t, >32t) + articulated (<40t, >40t) x {EU, UK, US, global} x {WTW, TTW} x {per t-km, per v-km} = ~240
- Rail: electric + diesel x {EU, UK, US, IN, global} x {WTW, TTW, per t-km} = ~40
- Sea: container (feeder, small, medium, large, ULCS) + bulk (handysize, handymax, panamax, capesize) + tanker (MR, LR1, LR2, VLCC) + refrigerated reefer x {WTW, TTW, per TEU-km, per t-km} = ~180
- Air: belly-hold passenger + dedicated freighter x short-haul + medium-haul + long-haul x {WTW, TTW, per t-km} = ~60
- Inland waterway: Rhine / Danube / Mississippi / Yangtze patterns x {per t-km} = ~25
- Corridor lanes (IN-EU, EU-intra, US-domestic, IN-domestic, trans-Pacific, trans-Atlantic) x {refrigerated vs. dry, utilization 50/65/80%} = ~60

### 3.6 Gold-eval coverage required

**At least 110 activity->factor pairs**:
- 30 pairs sea container (IN-EU, trans-Pacific, EU-intra) with TEU + t-km bases
- 25 pairs road (EU articulated, US Class 8, IN rigid) with empty-run variations
- 20 pairs air cargo (long-haul, short-haul, belly-hold vs. freighter)
- 15 pairs rail (EU electric freight, US diesel freight)
- 10 pairs refrigerated road + reefer sea with refrigeration uplift
- 10 pairs last-mile delivery (LCV, cargo bike, EV delivery van)

### 3.7 Acceptance gate

- Top-1 >= **0.85**
- Cross-source deviation: DESNZ HGV articulated >=32t vs. EPA Class 8 diesel vs. GLEC EU articulated — must agree within **5%** after payload+utilization normalization
- Empty running factor must be declared on every lane with mode `road` (otherwise Q6 gate fails)
- WTW vs. TTW must both be present for every lane (declared via `boundary in {WTW, WTT}`)

### 3.8 Method profiles unlocked

- `FREIGHT_ISO_14083` (from `freight.py`, pack_version 0.3.0 — bumps to 1.0.0 at this slice) — registered to `MethodProfile.FREIGHT_ISO_14083`
- `tags=("freight", "licensed")` — this pack is a Premium SKU per current code

### 3.9 Downstream app unlock

- **SupplierOS** full freight unlock (shipment-level carbon invoicing)
- **Scope Engine** Scope 3 Category 4 (Upstream Transport) + Category 9 (Downstream Transport) production mode
- **Comply / CSRD** E1-6 Scope 3 transport line
- **CBAM** transport embedded (not covered by CBAM Article 4 directly, but needed for product LCA in Slice 6)

### 3.10 GTM hook

- **Starter (free):** public factors DESNZ + EPA for road + rail + sea average — ~180 factors
- **Premium Freight Pack (paid):** full GLEC v3.0 library + lane corridors + refrigeration uplift + utilization tiers + air cargo split (~500 factors)
- **Enterprise:** shipment-level API, TMS integration, carrier-specific factor slots

### 3.11 Rollback plan

Fall back to `2026.Q4-freight-preview` (the last Preview tier). Freight has no prior Certified edition, so a rollback returns customers to Preview tier — they lose audit-grade status on freight factors until next promotion attempt.

---

## 4. Slice 4 — Material + CBAM  (edition tag: `2027.Q4-material-cbam`, Week 10)

### 4.1 Sales one-liner
"The only factor library that shipped the EU CBAM default-value table, CN-code-aligned embedded emissions for iron & steel / aluminum / cement / fertilizers / hydrogen, plus EPD-backed material embodied carbon for the other 20 commodities your supply chain cares about."

### 4.2 Source IDs

| source_id | license_class | redistribution | cadence |
|-----------|---------------|----------------|---------|
| `eu_cbam` | `eu_publication` | allowed | event_driven |
| `greenlang_builtin` | `greenlang_terms` | allowed | on_release |
| `epa_hub` | `public_us_government` | allowed | quarterly |

**EC3 EPD:** parser at `ec3_epd.py`. Registry row `ec3_buildings_epd` must be added with `redistribution_class: restricted` per Building Transparency terms. Bulk export is **licensed** (Premium pack only). Public tier gets EC3 content via connector-only.

**PACT Pathfinder product data:** parser at `pact_product_data.py`. Registry row `pact_pathfinder` with `license_class: pact_terms` — member-contributed PCF data, attribution-only.

**Ecoinvent:** connector-only (already row `ecoinvent`). NOT bulk exported. Premium Product Carbon Pack (Slice 6) surfaces it via connector.

### 4.3 Versions pinned

| source_id | pinned version |
|-----------|----------------|
| `eu_cbam` | **Regulation (EU) 2023/956 Annex III + Implementing Regulation (EU) 2023/1773 default values (revised 2024-12)** |
| `ec3_buildings_epd` | EC3 snapshot 2026-02 (Type III EPDs verified to ISO 21930 / EN 15804) |
| `pact_pathfinder` | PACT v2.1 (2025-Q3) PCF data exchange schema |
| `ghgp_product_standard` | GHG Protocol Product Standard (2011) — methodology reference |

### 4.4 License class summary

- **Redistributable:** EU CBAM defaults (EU publication), EPA, greenlang_builtin
- **EC3:** `restricted` → **licensed** for Premium Material Pack bulk; public tier sees EC3 via connector only (enforced at serving layer)
- **PACT:** attribution required per Pathfinder terms; contributor-level PCF data treated as `restricted`
- **No ecoinvent in this slice** (deferred to Slice 6)

### 4.5 Certified factor count target

**~600-900 certified material + CBAM factors.** Breakdown:
- CBAM iron & steel (CN 72 + 73 codes in scope — ~40 CN codes) x {direct, indirect, embedded} = ~120
- CBAM aluminum (CN 76 codes) x {direct, indirect} = ~40
- CBAM cement (CN 2523) x {grey, white, CEM I, CEM II, CEM III} = ~20
- CBAM fertilizers (CN 2808, 3102, 3105) x {ammonia, urea, ammonium nitrate, complex NPK} = ~30
- CBAM hydrogen (CN 2804.10.00) x {grey, blue, green} = ~10
- CBAM electricity imports — reuse Slice 1 grid factors; rows added to material pack as reference = ~20
- Non-CBAM plastics (PE HDPE/LDPE/LLDPE, PP, PVC, PET bottle+fibre, PS, expanded PS) x {cradle-to-gate, virgin vs. recycled} = ~80
- Chemicals (ammonia, urea, methanol, ethylene, propylene, benzene, chlorine, sulphuric acid) x {process emissions, full LCA} = ~60
- Glass (container, flat, fibre) x {virgin, recycled content tiers 0/30/70%} = ~30
- Paper (kraft, newsprint, coated, tissue, paperboard) x {virgin, recycled} = ~40
- Textiles (cotton, polyester, wool, nylon, viscose) x {spinning, weaving, dyeing cradle-to-gate} = ~50
- Construction metals non-CBAM (stainless steel grades, copper, zinc, lead) = ~40
- EC3-backed concrete EPDs (28 MPa, 35 MPa, 45 MPa x regions) = ~80

### 4.6 Gold-eval coverage required

**At least 140 activity->factor pairs**, with mandatory CBAM coverage:
- 40 pairs CBAM covered goods (every CN code subcategory) — MUST achieve 0.90+ precision given regulatory stakes
- 25 pairs plastics (HDPE bottles, PET packaging, PP automotive)
- 20 pairs cement (CEM I, CEM II/B-M, EPD-backed specifics)
- 15 pairs chemicals (ammonia via SMR vs. electrolysis, methanol)
- 15 pairs textiles (cotton virgin, polyester recycled, wool)
- 15 pairs paper (kraft packaging, tissue, coated board)
- 10 pairs glass (virgin float, recycled container)

### 4.7 Acceptance gate

- Top-1 >= **0.85** overall; CBAM subset >= **0.90**
- `EU_CBAM.selection_rule` enforces `allowed_statuses=("certified",)` AND `require_verification=True` — any Preview or unverified factor being routed to `MethodProfile.EU_CBAM` = promotion-blocking failure
- CBAM Article 4(2) fallback disclosure: every CBAM factor must carry `source_year` and explicit "EU default value" flag when primary operator data not used
- Biogenic treatment for CBAM factors MUST be `EXCLUDED` per `EU_CBAM.boundary_rule.biogenic_treatment` — any factor with `biogenic included` flag routing to CBAM = failure
- Cross-source deviation: EC3 EPD concrete 28 MPa vs. regional industry average (e.g. ACI for US, IBU for DE) must be within **+/- 10%** (stretch beyond 5% is allowed for EPDs because the EPD represents a specific product, not industry average)

### 4.8 Method profiles unlocked

- `EU_CBAM` (registered to `MethodProfile.EU_CBAM`, pack_version 1.0.0, `tags=("eu_policy", "licensed")`)
- `EU_DPP` (registered to `MethodProfile.EU_DPP`, pack_version 0.1.0)
- `PRODUCT_CARBON` partial unlock — the `MATERIAL_EMBODIED` family is a subset of what Slice 6 fully opens

### 4.9 Downstream app unlock

- **CBAM app** production mode — full unlock, can accept customer declarations
- **DPP Hub** v1 — shape enabled, pending ESPR implementing acts (per `EU_DPP.description`)
- **Comply / CSRD** E5 (Resource use) partial unlock via MATERIAL_EMBODIED factors

### 4.10 GTM hook

- **Starter (free):** EU CBAM defaults (EU publication, open), basic plastic + cement factors = ~120 factors
- **Premium Material + CBAM Pack (paid):** EC3 EPD library (licensed), PACT member PCFs (attribution), full non-CBAM commodity set — ~700 factors
- **CBAM declarant add-on:** direct submission workflow, primary data ingestion, operator verification flow

### 4.11 Rollback plan

Fall back to `2027.Q2-cbam-preview` (the CBAM-only Preview edition that preceded this slice). Non-CBAM material factors lose certified status and fall back to the generic `greenlang_builtin` v0.9 material set.

---

## 5. Slice 5 — Land + Removals  (edition tag: `2028.Q1-land-removals`, Week 12)

### 5.1 Sales one-liner
"GHG Protocol Land Sector and Removals Guidance, end-to-end: land-use emissions, ongoing land management, active removals (nature-based + tech), and durable storage — with permanence class, reversal risk, and MRV verification-standard alignment on every factor."

### 5.2 Source IDs

| source_id | license_class | redistribution | cadence |
|-----------|---------------|----------------|---------|
| `ghgp_method_refs` | `wri_wbcsd_terms` | attribution only | ad_hoc |
| `greenlang_builtin` | `greenlang_terms` | allowed | on_release |

**IPCC AFOLU:** registry row `ipcc_2006_afolu_v2019` needed with `license_class: public_international` — parser at `ipcc_defaults.py` covers Volume 4 (Agriculture, Forestry and Other Land Use).

**FAO FRA / Global Forest Watch:** cross-check only for above-ground biomass defaults.

**Verra VCS, Gold Standard, Puro.earth, Isometric:** methodology references cited in `land_removals.py` docstring; not factor sources.

### 5.3 Versions pinned

| source_id | pinned version |
|-----------|----------------|
| `ipcc_2006_afolu_v2019` | IPCC 2006 Vol. 4 + 2019 Refinement Vol. 4 |
| `ghgp_lsr` | GHG Protocol LSR Guidance (public comment 2024-09, final expected 2025-Q4) |
| `verra_vcs_v44` | VCS Standard v4.4 — methodology reference |
| `gold_standard_v12` | Gold Standard for the Global Goals v1.2 — methodology reference |
| `puro_earth_v20` | Puro.earth Supplier General Rules v2.0 — methodology reference |
| `isometric_v1` | Isometric Standard v1 — methodology reference |
| `icvcm_ccp_2023` | ICVCM Core Carbon Principles 2023 — methodology reference |

### 5.4 License class summary

- **Redistributable:** IPCC AFOLU defaults, greenlang_builtin
- **Attribution only:** GHG Protocol LSR Guidance — text excerpts allowed with citation per `source_registry.yaml::ghgp_method_refs`
- **Verification standards:** VCS / Gold Standard / Puro / Isometric — named only, not bulk-ingested

### 5.5 Certified factor count target

**~200-400 certified land + removal factors.** Breakdown:
- Land-use emissions (deforestation emission x biome: tropical moist, tropical dry, temperate, boreal; peatland drainage; mangrove conversion) = ~40
- Land management (cropland conventional / no-till / cover crop, grassland intensive / extensive, agroforestry) x IPCC Tier 1 climate zones = ~60
- Active removals (afforestation, reforestation, peatland rewetting, blue carbon, soil carbon sequestration) x {above-ground biomass, below-ground biomass, soil organic carbon, dead organic matter} = ~80
- Technology-based removals (biochar, BECCS, DACCS, enhanced weathering, mineralisation, ocean alkalinity) x {sequestration rate, storage permanence} = ~40
- Carbon-stock defaults (forest biomass by biome + age class) = ~60
- Permanence + reversal risk tables (per permanence class x verification standard) = ~30
- Buffer-pool % defaults (low / medium / high reversal risk) = ~10

### 5.6 Gold-eval coverage required

**At least 60 activity->factor pairs**:
- 15 pairs deforestation emissions (tropical + temperate biomes)
- 10 pairs afforestation sequestration (tropical + temperate + boreal)
- 10 pairs peatland (emission on drainage + sequestration on rewetting)
- 10 pairs soil organic carbon (cropland management)
- 5 pairs biochar (spread rate, persistence)
- 5 pairs DACCS / BECCS
- 5 pairs mangrove / blue carbon

### 5.7 Acceptance gate

- Top-1 >= **0.85**
- Every removal factor MUST carry `permanence_class IN {short, medium, long}` per `land_removals.py::PermanenceClass`
- Every removal factor MUST carry `reversal_risk IN {low, medium, high}`; high-risk = buffer-pool % stamped on the factor
- Biogenic accounting treatment MUST be declared as `carbon_neutral | sequestration_tracked | storage_tracked` per `BiogenicAccountingTreatment`
- Cross-source deviation: IPCC Tier 1 cropland default vs. FAO cross-check vs. VCS AFOLU methodology default — tolerance widened to **10%** given methodology heterogeneity

### 5.8 Method profiles unlocked

- `LSR_LAND_USE_EMISSIONS`, `LSR_LAND_MANAGEMENT`, `LSR_ACTIVE_REMOVALS`, `LSR_CARBON_STORAGE` — all four variants registered via `land_removals.py::_register_packs`; `MethodProfile.LAND_REMOVALS`

### 5.9 Downstream app unlock

- **Scope Engine** Scope 1 agriculture/forestry + Scope 3 Category 1 land-use-change footprint
- **Comply / CSRD** E4 (Biodiversity and ecosystems) partial unlock, E1-7 GHG removals disclosure
- **TCFD / IFRS S2** — nature-related disclosures (TNFD overlap flagged for Slice 6 extension)

### 5.10 GTM hook

- **Premium Land & Removals Pack (paid from day one):** this slice is Premium-only because LSR factor curation + permanence + reversal classification requires methodology expertise. ~300 factors at launch.
- **Offset credit buyer add-on:** CCP-aligned project-level factor overlay, integrates with ICVCM label, verification-standard filter (VCS / Gold Standard / Puro / Isometric routing)
- **No Starter tier exposure** — land factors land in Premium only

### 5.11 Rollback plan

No prior Certified land edition exists. Rollback = demote to `2027.Q4-land-preview` (the Preview edition that preceded); downstream apps display "Land & Removals in Preview" banner. Customers not on the Premium pack were never exposed to this slice, so no customer-facing regression.

---

## 6. Slice 6 — Product Carbon  (edition tag: `2028.Q2-product-carbon`, Week 14)

### 6.1 Sales one-liner
"Full product LCA — ISO 14067, GHG Protocol Product Standard, PACT, PAS 2050, EU PEF, EU OEF — with the ecoinvent connector turning your Premium pack into a 50 000-factor live LCI library the moment you plug in your license key."

### 6.2 Source IDs

| source_id | license_class | redistribution | connector_only | cadence |
|-----------|---------------|----------------|----------------|---------|
| `ecoinvent` | `commercial_connector` | denied | **true** | quarterly |
| `greenlang_builtin` | `greenlang_terms` | allowed | false | on_release |

**EC3 EPD:** reused from Slice 4 — already licensed.
**PACT Pathfinder:** reused from Slice 4 — attribution.
**EF 3.1 secondary datasets** (EU JRC): registry row `ef_3_1_secondary` needed with `license_class: eu_publication` for PEF factors — JRC EF databases are public. Required before Week 13.
**BSI PAS 2050:** methodology-only; BSI methodology reference (registered as `bsi_pas_2050`, attribution terms).

### 6.3 Versions pinned

| source_id | pinned version |
|-----------|----------------|
| `ecoinvent` | ecoinvent v3.11 (released 2025-Q2) — connector-resolved, never cached in bulk |
| `ef_3_1_secondary` | EU JRC EF 3.1 secondary dataset (2023 release) |
| `bsi_pas_2050` | PAS 2050:2011 (methodology; no numeric factors) |
| `iso_14067_2018` | ISO 14067:2018 (methodology; no numeric factors) |
| `ghgp_product_standard` | GHG Protocol Product Standard 2011 (methodology) |
| `pact_pathfinder` | v2.1 |

### 6.4 License class summary

- **Connector-only, never bulk:** `ecoinvent` — enforced by `connector_only: true` flag in registry; bulk export MUST exclude (checked at S6 legal sign-off)
- **EU publication, redistributable:** `ef_3_1_secondary` (EC JRC publication terms)
- **Methodology references:** PAS 2050, ISO 14067, GHG PS — excerpts only with citation

### 6.5 Certified factor count target

**~300-500 certified product carbon factors** in bulk set; **~50 000+** additional factors available live via the ecoinvent connector for Premium subscribers.

Bulk breakdown:
- EF 3.1 secondary datasets (packaging materials, electricity mixes, transport, waste treatment) = ~200
- PACT member-contributed PCFs (steel, aluminum, chemicals, plastics for Pathfinder members who opted in to PACT) = ~80
- GreenLang-curated cradle-to-gate defaults for 40 product categories x regions = ~120
- Allocation method + recycled content parameter tables = ~50

Connector tier (Premium only): ecoinvent v3.11 consequential + attributional + cut-off system models — ~22 000 activities per model.

### 6.6 Gold-eval coverage required

**At least 80 activity->factor pairs**:
- 20 pairs packaging (PET bottle, HDPE bottle, aluminum can, glass bottle cradle-to-gate)
- 15 pairs textiles (polyester t-shirt, cotton t-shirt, wool jumper, nylon jacket)
- 15 pairs electronics (smartphone avg, laptop avg, server avg — cradle-to-gate)
- 10 pairs food (milk, beef, wheat, soy — cradle-to-farm-gate)
- 10 pairs construction (ready-mix concrete, steel rebar, gypsum board)
- 10 pairs automotive (ICE vehicle avg, BEV avg)

### 6.7 Acceptance gate

- Top-1 >= **0.85**
- Allocation method MUST be declared per factor: `mass | economic | energy | system_expansion`
- Recycled content % MUST be declared where applicable (plastics, paper, metals)
- `require_verification=True` for PEF-routed factors per `product_lca_variants.py::PEF.selection_rule`
- Cross-source deviation: ecoinvent (connector) vs. EF 3.1 vs. greenlang_builtin — tolerance **15%** for LCA factors (wider than grid/combustion because LCA has legitimate methodological variation); any deviation >15% triggers S3 consistency review

### 6.8 Method profiles unlocked

- `PRODUCT_CARBON` (umbrella, registered to `MethodProfile.PRODUCT_CARBON`, pack_version 0.5.0 → 1.0.0 at promotion)
- `PAS_2050`, `PEF`, `OEF` — variants via `get_product_lca_variant("pas_2050" | "eu_pef" | "eu_oef")` from `product_lca_variants.py`

### 6.9 Downstream app unlock

- **PCF Studio** production mode — full unlock
- **SupplierOS** product-level carbon-invoicing via PACT schema
- **DPP Hub** v2 (product-data shape fills with real factors)
- **Comply / CSRD** E5 (Resource use) full unlock

### 6.10 GTM hook

- **Starter (free):** EF 3.1 secondary datasets + greenlang_builtin product defaults — ~250 factors
- **Premium Product Carbon Pack (paid):** ecoinvent connector access + PACT member data + PAS 2050 / PEF / OEF variants — hundreds of thousands of LCI activities via connector
- **Enterprise:** PEFCR-specific calculators, OEFSR sector rules, multi-allocation comparison, DPP export

### 6.11 Rollback plan

Fall back to `2027.Q4-product-preview`. Ecoinvent connector continues to operate (it's a live API, not a cached edition). Bulk product factor set degrades to Preview precision.

---

## 7. Slice 7 — Finance Proxy  (edition tag: `2028.Q3-finance-proxy`, Week 16)

### 7.1 Sales one-liner
"PCAF-compliant financed-emissions proxies — every asset class, every data-quality tier 1-5, cross-mapped to NACE / GICS / NAICS with geography and intensity-basis controls. The end of bank analysts hand-coding sector factors in spreadsheets."

### 7.2 Source IDs

| source_id | license_class | redistribution | cadence |
|-----------|---------------|----------------|---------|
| `greenlang_builtin` | `greenlang_terms` | allowed | on_release |

**PCAF proxies:** parser at `pcaf_proxies.py`. Registry row `pcaf_global_std_v2` with `license_class: pcaf_terms` (PCAF Standard v2 is open-access with attribution) required before Week 15.
**EXIOBASE, CEDA, US EPA SUSEEIO:** environmentally extended input-output databases — registry rows needed for sector proxy backing:
- `exiobase_v3` — `license_class: exiobase_terms` (free for research/non-commercial; commercial use requires license); **connector-only for commercial**
- `ceda_pbe` — `license_class: commercial_connector` (Vital Metrics CEDA is commercial)
- `us_epa_suseeio` — `license_class: public_us_government`, redistributable
**NACE / GICS / NAICS cross-maps** are encoded in `greenlang/factors/mapping/industry_codes.py`, not a separate factor source.

### 7.3 Versions pinned

| source_id | pinned version |
|-----------|----------------|
| `pcaf_global_std_v2` | PCAF Global GHG Accounting Standard Part A, Second Edition (December 2022) + Part B (2023) + Part C (2024) |
| `us_epa_suseeio` | SUSEEIO v2.0 (2024 release) |
| `exiobase_v3` | EXIOBASE 3.8.2 (2024 release, base year 2022) |
| `ceda_pbe` | CEDA PBE 2025 release (connector) |

### 7.4 License class summary

- **Redistributable:** PCAF methodology (attribution), US EPA SUSEEIO, greenlang_builtin
- **Commercial / Connector:** EXIOBASE (commercial license required for commercial use → connector-only), CEDA (fully commercial, connector-only)

### 7.5 Certified factor count target

**~200-400 certified finance proxy factors.** Breakdown:
- PCAF asset-class defaults (7 classes x {Scope 1+2, Scope 3}) x geography = ~50
- Sector intensity (NACE 2-digit, ~80 sectors) x {revenue, asset, EVIC, EBITDA} x geography = ~200
- GICS industry (157 GICS sub-industries) mapped to NACE with fallback = ~80 cross-map entries
- NAICS (6-digit subset of materially-emitting industries, ~60 codes) = ~60
- Proxy confidence class (DQS 1-5 per PCAF Chapter 5.4) stamped on every factor = metadata

### 7.6 Gold-eval coverage required

**At least 70 activity->factor pairs**:
- 20 pairs listed equity / corporate bonds (top-10 GICS sectors x top-5 geographies)
- 10 pairs business loans (SME in EU, US, IN)
- 10 pairs commercial real estate (office, retail, industrial, by climate zone)
- 10 pairs mortgages (residential energy intensity by country + dwelling type)
- 10 pairs motor vehicle loans (ICE vs BEV mix)
- 5 pairs project finance (renewable vs. fossil-fuel project tags)
- 5 pairs sovereign bonds (country average intensity)

### 7.7 Acceptance gate

- Top-1 >= **0.85**
- Every factor MUST declare `pcaf_dqs IN {1, 2, 3, 4, 5}`; uncertainty band required when `dqs >= 4`
- Cross-source deviation: EXIOBASE sector X vs. SUSEEIO sector X vs. CEDA sector X (when all three cover) — must agree within **15%** after NACE/NAICS normalization; wider tolerance reflects EEIO methodology variance
- Sector cross-map from `mapping/industry_codes.py` MUST resolve for every PCAF proxy factor (no orphan factors)

### 7.8 Method profiles unlocked

- `FINANCE_PROXY` umbrella (registered to `MethodProfile.FINANCE_PROXY`) + 7 PCAF asset-class variants: `pcaf_listed_equity`, `pcaf_corporate_bonds`, `pcaf_business_loans`, `pcaf_project_finance`, `pcaf_commercial_real_estate`, `pcaf_mortgages`, `pcaf_motor_vehicle_loans` — all registered via `finance_proxy.py::_register_packs`

### 7.9 Downstream app unlock

- **SupplierOS financial edition** — banks / asset managers PCAF reporting
- **Comply / CSRD** S1-16 entity-level metrics + Pillar 3 ESG (where applicable)
- **TCFD / IFRS S2** financed emissions disclosure
- **Future: Financed Emissions Studio** (v1.1 app) — seeded by this slice

### 7.10 GTM hook

- **Starter (free):** PCAF methodology + SUSEEIO sector factors (US-centric, attribution) = ~100 factors
- **Premium Finance Proxy Pack (paid):** EXIOBASE connector + CEDA connector + full NACE/GICS cross-map + 1-5 DQS uncertainty bands = full ~350 factors + connectors
- **Bank / Asset Manager tier:** portfolio ingestion API, PCAF reporting templates, attribution-factor calculators

### 7.11 Rollback plan

No prior Certified finance edition — this is the v1 debut. Rollback = revert to `2028.Q2-finance-preview` Preview edition and banner customers that the finance proxies are "Preview tier, not audit-grade" until next promotion attempt.

---

## 8. Regulatory map (by slice)

| Regulation | Electricity | Combustion | Freight | Material+CBAM | Land+Removals | Product Carbon | Finance |
|------------|-------------|------------|---------|---------------|---------------|----------------|---------|
| **EU CBAM** (Reg 2023/956) | grid imports (indirect) | process heat inputs | — | **PRIMARY** | — | — | — |
| **EU CSRD / ESRS** | E1-5 energy, E1-6 Scope 2 LB+MB | E1-6 Scope 1 | E1-6 Scope 3 cat 4+9 | E1-6 Scope 3 cat 1, E5 Resource use | E1-7 removals, E4 Biodiversity | E5 Resource use | S1-16 indirect via E1-6 |
| **California SB 253** | Scope 2 LB | Scope 1 | Scope 3 (optional from 2027) | Scope 3 | — | Scope 3 | — |
| **TCFD** | Scope 2 | Scope 1 | Scope 3 transport | Scope 3 purchased | physical risk + removals | Scope 3 purchased | financed emissions |
| **IFRS S2** | full alignment with TCFD | full alignment | full alignment | full alignment | nature-related optional | Scope 3 | **PRIMARY** financed emissions |
| **India BRSR** | Principle 6 energy | Principle 6 emissions | Principle 6 value chain | Principle 6 resource | Principle 6 + Principle 2 sustainable sourcing | Principle 6 | indirect via Principle 6 |
| **EU DPP** (ESPR 2024/1781) | — | — | — | **secondary** (schema seed) | — | **PRIMARY** | — |
| **PCAF Global Standard** | via financed electricity | via financed combustion | via financed transport | via financed industrials | via financed AFOLU | via financed products | **PRIMARY** |

---

## 9. Per-slice quality-gate matrix (DQS minima)

| Slice | Temporal | Geographical | Technological | Verification (representativeness) | Completeness (methodological) | Composite FQS |
|-------|----------|--------------|---------------|-----------------------------------|-------------------------------|---------------|
| 1 Electricity | 4 | 4 | 3 | 3 | 4 | **80** |
| 2 Combustion | 3 | 3 | 3 | 3 | 4 | **75** |
| 3 Freight | 3 | 3 | 3 | 3 | 3 | **75** |
| 4 Material+CBAM (non-CBAM) | 3 | 3 | 3 | 3 | 3 | **75** |
| 4 Material+CBAM (CBAM-routed) | 4 | 4 | 4 | **4** (require_verification=True) | 4 | **85** |
| 5 Land+Removals | 3 | 3 | 3 | 3 | 3 | **75** |
| 6 Product Carbon (non-PEF) | 3 | 3 | 3 | 3 | 3 | **75** |
| 6 Product Carbon (PEF-routed) | 4 | 4 | 4 | **4** (require_verification=True) | 4 | **85** |
| 7 Finance Proxy | 3 | 3 | 2 | 3 | 3 | **70** (PCAF allows DQS 5; composite bar lowered because proxy is by nature less specific) |

Note: Slice 7 is the only slice where the composite FQS falls below the 75 Certified floor defined in §0. This is an explicit exception because PCAF DQS 4-5 data is inherently lower-quality by standard design (asset-class defaults). The deviation is recorded in the methodology sign-off (S5) for the founder to acknowledge.

---

## 10. Cross-slice cadence + dependency map

```
 Week 4   Slice 1 Electricity   ─┬─▶ Scope Engine v1 Scope 2 ready
                                  └─▶ seeds CBAM indirect (Slice 4)
 Week 6   Slice 2 Combustion    ─┬─▶ Scope Engine v1 Scope 1 ready
                                  └─▶ seeds CBAM direct emissions (Slice 4)
 Week 8   Slice 3 Freight       ──▶ SupplierOS freight + Scope Engine Scope 3 cat 4/9
 Week 10  Slice 4 Material+CBAM ──▶ CBAM app live; DPP Hub shape ready
 Week 12  Slice 5 Land+Removals ──▶ Scope Engine AFOLU + removals disclosure
 Week 14  Slice 6 Product Carbon──▶ PCF Studio live; PACT + ecoinvent connector
 Week 16  Slice 7 Finance Proxy ──▶ Financed Emissions Studio seeded
```

Each slice promotion is a separate `approve_release()` call against the signoff object. v1.0 is "Certified Edition released" only when ALL 7 slices pass S1-S9. No in-place mutation; each edition id is permanent.

---

## 11. Per-slice signoff item lock (S1-S9 applied to every slice)

Every slice must independently satisfy:
- **S1:** `BatchQAReport.total_failed == 0` over the slice's factor subset
- **S2:** `DedupReport.human_review == 0` (any two factors with same activity+geography+basis+year that disagree by >1% on value need resolution before Certified)
- **S3:** `ConsistencyReport.total_reviews == 0` (all cross-source overlaps adjudicated)
- **S4:** slice changelog written to `manifest.changelog` and reviewed
- **S5:** methodology lead signs off per-slice (8 signatures total: 7 slices + 1 cumulative v1.0)
- **S6:** legal confirms licenses for every source_id referenced in the slice (10 confirmations for Slice 1, ranging down to 1 for Slices 3+5+7)
- **S7:** regression test — every slice compares to its predecessor edition; no silent >5% drift on carry-over factors
- **S8:** load test p95 < 500 ms against the slice's factor count (Slices 1, 4, 6 are heaviest)
- **S9:** gold-eval precision@1 >= 0.85 against the gold set for that slice (60-180 pairs per §§1.6-7.6)

A slice that fails any required item on promotion day is pushed to the following week with a root-cause write-up in the changelog. No force-approves.

---

## 12. Final v1.0 Certified Edition composition (once all 7 slices promoted)

- **Total method packs registered:** 10 MethodProfile enum values (CORPORATE_SCOPE1, CORPORATE_SCOPE2_LOCATION, CORPORATE_SCOPE2_MARKET, CORPORATE_SCOPE3, PRODUCT_CARBON, FREIGHT_ISO_14083, LAND_REMOVALS, FINANCE_PROXY, EU_CBAM, EU_DPP) plus 3 product-LCA variants (PAS 2050, PEF, OEF), 4 residual-mix variants (EU, US/CA, AU, JP), 7 PCAF asset-class variants, 4 LSR variants.
- **Total Certified factors (bulk):** ~3,000-5,000 across all slices.
- **Total factors via connector (ecoinvent, electricity_maps, IEA, EXIOBASE, CEDA):** 50,000+ live.
- **Total source IDs registered:** 18 (current 15 in `source_registry.yaml` + 3 new: `india_cea_co2_baseline`, `ipcc_2006_nggi`/`ipcc_2006_afolu_v2019`, `glec_framework`, `ec3_buildings_epd`, `pact_pathfinder`, `ef_3_1_secondary`, `pcaf_global_std_v2`, `us_epa_suseeio`, `exiobase_v3`, `ceda_pbe`) — curator action: reconcile final count before Slice 1 promotion and ensure legal S6 has all rows.
- **Jurisdictions covered:** IN, EU27+EEA, UK, US (+ 26 eGRID subregions + 51 states), CA (provinces), AU (NEM + WA + NT), JP (10 utility areas), global-default.
- **Regulatory packs served:** EU CBAM, EU CSRD, EU DPP, California SB 253, TCFD, IFRS S2, India BRSR, PCAF.
- **Premium Data Pack SKUs:** 5 — Premium Electricity, Premium Combustion, Premium Freight, Premium Material+CBAM, Premium Product Carbon, Premium Land & Removals, Premium Finance Proxy. (Land & Removals is Premium-only at launch — no Starter exposure.)

---

## 13. Bind points — what must exist in code on Day 0 of Slice 1 promotion

These are the curator's blocking items before `approve_release()` runs for `2027.Q1-electricity`:

1. New rows in `greenlang/factors/data/source_registry.yaml`:
   - `india_cea_co2_baseline`
   - `ipcc_2006_nggi` (for Slice 2 — but must exist by Slice 2 promotion)
   - `aib_residual_mix_eu` (referenced by `ELECTRICITY_RESIDUAL_MIX_EU.selection_rule.custom_filter` but not in registry today)
   - `glec_framework` (Slice 3)
   - `ec3_buildings_epd`, `pact_pathfinder`, `ef_3_1_secondary` (Slices 4 + 6)
   - `pcaf_global_std_v2`, `us_epa_suseeio`, `exiobase_v3`, `ceda_pbe` (Slice 7)
2. `greenlang/factors/method_packs/freight.py`: pack_version bump 0.3.0 → 1.0.0 (currently tagged as pre-release)
3. `greenlang/factors/method_packs/product_carbon.py`: pack_version bump 0.5.0 → 1.0.0
4. `greenlang/factors/method_packs/eu_policy.py::EU_DPP`: pack_version stays 0.1.0 until ESPR implementing acts land — promotion allowed as "shape-only" edition, not factor-set edition
5. `release_signoff.py`: keep current 9-item checklist; v1 policy upgrades S7 and S8 from `severity="recommended"` to `severity="required"` (requires patch — curator's action on Week 2)
6. Gold-eval harness must have slice-scoped test runs: the precision@1 threshold of 0.85 is evaluated PER SLICE not globally; requires a `--slice <name>` flag on the gold-eval CLI

---

## 14. Founder acknowledgements + exceptions

1. **Slice 7 composite FQS bar lowered to 70** (§9) — PCAF methodology legitimately allows DQS 5 factors as last-resort proxies; the composite would be unachievable at 75.
2. **Slice 5 (Land & Removals) is Premium-only at launch** — no Starter exposure — because methodology curation requires specialist oversight.
3. **Cross-source tolerance widened to 15% for LCA (Slice 6) and EEIO (Slice 7)** — §§6.7, 7.7 — acknowledging that these data types have legitimate methodological spread that the 5% electricity/combustion bar cannot accommodate.
4. **EU DPP edition shipped at `pack_version=0.1.0`** as a shape-only Certified edition — the profile is real, the factor set is deferred to Slice 6 promotion when ESPR implementing acts define per-product PCF fields.
5. **v1 is the first edition where S7 + S8 are required, not recommended.** This raises the bar over v0.9. Any regression test failure or p95 latency breach = no promotion that week.

---

**End of cut-list.** This document is the authoritative source for v1.0 Certified Edition promotion decisions. Every slice promotion PR references this doc by section number. Changes to this doc require founder acknowledgement + methodology-lead sign-off.
