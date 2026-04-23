# GreenLang Factors v1 — Source Rights Matrix

**Owner:** Legal + Product
**Status:** DUE-DILIGENCE DRAFT — NOT A LEGAL OPINION
**Prepared by:** GL-RegulatoryIntelligence (code-grounded audit)
**Date:** 2026-04-23
**Covers CTO Master ToDo tasks:** S1 (source registry rights), S2 (public pack coverage), S6 (legal rights matrix before Certified), L1 (four data-class policy).
**Scope:** Every source referenced by a parser under `greenlang/factors/ingestion/parsers/`, every connector under `greenlang/factors/connectors/`, and every source enumerated in `greenlang/factors/data/source_registry.yaml`.

> **Purpose.** This file is a checklist for Legal review. Each row is what the codebase and registry currently *claim* about the source's rights posture; Legal must validate the claims against the actual license page of each publisher before any factor is stamped `Safe-to-Certify` in a Certified edition release.
>
> **Disclaimer.** Nothing in this document is legal advice. License classes marked from the parser/registry are programmer annotations, not legal conclusions. Any `UNKNOWN — needs legal research` entry means the codebase does not carry enough signal to make even a tentative call; Legal must do primary research.

---

## 1. Legend

**`redistribution_class`:**

| Value | Meaning |
|---|---|
| `Open` | Publisher allows verbatim and derivative redistribution (with attribution). Safe to bundle in Community tier and Certified edition bulk export. |
| `Licensed-Embedded` | Publisher allows embedding in a paid product but forbids public bulk export. Gated behind a Premium Pack entitlement; values delivered via API. |
| `Customer-Private` | Customer-supplied data / overlay factor. Never leaves tenant. Not a source class — included only for completeness per L1. |
| `OEM-Redistributable` | Publisher has granted GreenLang a redistribution right for sub-tenants (requires contract). |

**`v1_gate_status`:**

| Value | Meaning |
|---|---|
| `Safe-to-Certify` | Source can go into the Certified edition cut-list with current license posture, provided the attribution text is rendered in API responses and UI. |
| `Needs-Legal-Review` | Source can stay in the codebase but must NOT be promoted to Certified until Legal confirms redistribution terms. |
| `Blocked-Contract-Required` | Source MUST NOT ship any values in a Certified edition release until a signed contract or data-license agreement is executed. Connector stays disabled or runs only with customer-supplied credentials. |

---

## 2. Source Rights Matrix

> Format note: columns are wide; render best in a tabular viewer. `UNKNOWN — needs legal research` is the honest answer wherever the parser / registry / license manager does not encode the value.

| source_id | authority | jurisdiction | dataset_name | current_version | publication_year | license_name | license_url | redistribution_class | commercial_use_ok | attribution_required | attribution_text | derivative_works_ok | v1_gate_status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `epa_hub` | US EPA | US | EPA GHG Emission Factors Hub | 2024 (per parser metadata) | 2024 | US Government work / public domain (parser tag: `US-Public-Domain`) | https://www.epa.gov/climateleadership/ghg-emission-factors-hub | Open | Yes | Yes | "U.S. Environmental Protection Agency, GHG Emission Factors Hub." | Yes | Safe-to-Certify | Parser `epa_ghg_hub.py`. US federal works are not copyrighted (17 U.S.C. §105). Attribution required by registry even if not legally mandated. |
| `egrid` | US EPA | US | eGRID (Emissions & Generation Resource Integrated Database) | eGRID2022 (per parser metadata key) | 2024 (release) | US Government work / public domain (parser tag: `US-Public-Domain`) | https://www.epa.gov/egrid | Open | Yes | Yes | "U.S. EPA eGRID subregion emission rates." | Yes | Safe-to-Certify | Parser `egrid.py`. Use lb/MWh → kg/kWh conversion is deterministic; no transformation creates a new copyright. |
| `desnz_ghg_conversion` | UK DESNZ (formerly BEIS) | UK / GB | UK GHG conversion factors for company reporting | 2024/2025 series | 2024 | UK Open Government Licence v3.0 (`OGL-UK-v3`) | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | Open | Yes | Yes | "Contains public sector information licensed under the Open Government Licence v3.0 — UK Department for Energy Security and Net Zero, GHG conversion factors." | Yes | Safe-to-Certify | Parser `desnz_uk.py` sets `OGL-UK-v3` explicitly. Required attribution text is MANDATORY per OGL v3 §III. |
| `defra_conversion` | UK DEFRA / DESNZ | UK / GB | DEFRA environmental reporting factors | 2025 edition | 2025 | UK Open Government Licence v3.0 | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | Open | Yes | Yes | "Contains public sector information licensed under the Open Government Licence v3.0." | Yes | Safe-to-Certify | Registered in `source_registry.yaml` separately from DESNZ; may be a naming alias — Legal to confirm one record or two. |
| `beis_uk_residual` | UK DESNZ | UK / GB | UK national electricity residual mix (derived from DESNZ + Ofgem REGO surrender) | annual | 2024 | UK Open Government Licence v3.0 | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | Open | Yes | Yes | "Contains public sector information licensed under the Open Government Licence v3.0. Residual-mix derivation uses Ofgem REGO surrender statistics." | Yes | Safe-to-Certify | Derived values — Legal should confirm the derivation (OGL + REGO surrender data) is consistent with OGL terms. Parser missing — only registry entry. |
| `ipcc_2006_nggi` | IPCC Task Force on National GHG Inventories | Global | IPCC 2006 Guidelines for National GHG Inventories + 2019 Refinement | 2006 + 2019 Refinement | 2006 / 2019 | IPCC Guidelines — public international publication (parser tag: `IPCC-Guideline`) | https://www.ipcc-nggip.iges.or.jp/public/2006gl/ and https://www.ipcc-nggip.iges.or.jp/public/2019rf/ | Open | Yes | Yes | "IPCC 2006 Guidelines for National Greenhouse Gas Inventories and the 2019 Refinement." | Yes | Needs-Legal-Review | Parser `ipcc_defaults.py`. Registry `legal_notes` assert factual default values carry no copyright; Legal must confirm. IPCC permits reproduction with attribution per https://www.ipcc.ch/copyright/. |
| `ipcc_2006_afolu_v2019` | IPCC | Global | IPCC 2006 Guidelines Vol 4 — AFOLU (2019 Refinement) | 2019 Refinement | 2019 | IPCC Guidelines — public international publication | https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol4.html | Open | Yes | Yes | "IPCC 2006 Guidelines Volume 4 — Agriculture, Forestry and Other Land Use (AFOLU), 2019 Refinement." | Yes | Needs-Legal-Review | Same posture as `ipcc_2006_nggi` — requires IPCC copyright confirmation. |
| *IPCC EFDB* | IPCC TFI | Global | IPCC Emission Factor Database (EFDB) | continuous | continuous | UNKNOWN — needs legal research (no parser or registry entry found) | https://www.ipcc-nggip.iges.or.jp/EFDB/main.php | UNKNOWN | UNKNOWN — needs legal research | UNKNOWN — needs legal research | UNKNOWN — needs legal research | UNKNOWN — needs legal research | Needs-Legal-Review | **Parser missing.** EFDB terms-of-use page requires review: factors therein are submitted by member states and have per-record citation requirements. Legal: confirm redistribution terms before adding to v1. |
| *AR4 / AR5 / AR6 GWP tables* | IPCC | Global | IPCC Assessment Reports — 100-yr GWP tables | AR4 (2007), AR5 (2013), AR6 (2021) | 2007 / 2013 / 2021 | UNKNOWN — needs legal research (numbers only; used as constants in code) | https://www.ipcc.ch/copyright/ | Open (factual constants) | Yes | Yes | "IPCC AR4 / AR5 / AR6 Global Warming Potential values." | Yes | Needs-Legal-Review | Used throughout `greenlang/data/canonical_v2` `GWPSet`. Individual GWP numbers are factual and likely unprotectable, but IPCC copyright language is restrictive. Legal: obtain formal permission or rely on numerical-fact doctrine. |
| `ghgp_method_refs` | WRI / WBCSD | Global | GHG Protocol — Corporate Standard, Product Standard, Scope 2 Guidance, Scope 3 Guidance, cross-sector tools | Corporate rev 2004; Product 2011; Scope 2 2015; Scope 3 2011 & 2013 tools | 2004–2015 | WRI/WBCSD Terms of Use (parser tag: `WRI-WBCSD-Terms`) | https://ghgprotocol.org/terms-use | Licensed-Embedded | Yes (per WRI terms for internal use) | Yes | "Greenhouse Gas Protocol, WRI/WBCSD. [Standard name], [year]." | Restricted — WRI permits use but reserves derivatives | Needs-Legal-Review | Parser `ghg_protocol.py` tags `redistribution_allowed: False`. Legal: confirm whether factor values derived from Scope 3 calculation tools (Excel worksheets) are redistributable vs. only methodology text. |
| `pact_pathfinder` | WBCSD Partnership for Carbon Transparency | Global | PACT Pathfinder Framework (v3) | v3.0 | 2024 (per registry) | WBCSD / PACT terms (parser tag: `WRI-WBCSD-Terms`) | https://www.carbon-transparency.com/ | Licensed-Embedded | Yes | Yes | "WBCSD, Pathfinder Framework: Guidance for the Accounting and Exchange of Product Life Cycle Emissions, v3.0." | Restricted | Needs-Legal-Review | Parser `pact_product_data.py` sets `RedistributionClass.RESTRICTED`. Registry notes "Pathfinder data objects exchanged between customers require PACT network membership per §6". |
| `tcr_grp_defaults` | The Climate Registry | US / North America | TCR General Reporting Protocol default factors | 2024 edition | 2024 | TCR Registry Terms (parser tag: `TCR-Registry-Terms`) | https://theclimateregistry.org/tcr-general-reporting-protocol/ | Licensed-Embedded | Yes (for reporting) | Yes | "Per The Climate Registry / GHGRP guidance; verify current terms." | UNKNOWN — needs legal research | Blocked-Contract-Required | Parser `tcr.py` explicitly sets `redistribution_allowed: False`. TCR GRP is freely downloadable but publisher reserves redistribution; Certified edition must either route through API-only or execute a TCR data-license agreement. |
| `green_e_residual` | Green-e / Center for Resource Solutions | US | Green-e Residual Mix (older registry entry) | 2023 data | 2024 | Green-e Terms (parser tag: `Green-e-Terms`) | https://www.green-e.org/terms-and-conditions | Licensed-Embedded (connector-only per registry) | Yes | Yes | "Green-e; values via licensed connector only." | No (per Green-e Terms) | Blocked-Contract-Required | Parser `green_e.py` sets `redistribution_allowed: False`. Registry flags `connector_only: true`. Must execute Green-e data license before any Certified release. |
| `green_e_residual_mix` | Green-e / Center for Resource Solutions | US / CA | Green-e Energy Residual Mix Emission Rates (GAP-10 Wave 2 record) | 2024 publication of 2023 data | 2024 | Green-e Terms — `restricted` | https://www.green-e.org/residual-mix | Licensed-Embedded | Yes | Yes | "Green-e Energy Residual Mix, Center for Resource Solutions (annual)." | No | Blocked-Contract-Required | Parser `green_e_residual.py` sets `RedistributionClass.RESTRICTED`. Needs signed data license before Certified edition cut. Duplicate of `green_e_residual` — Legal to reconcile into a single source_id before GA. |
| `aib_residual_mix_eu` | Association of Issuing Bodies (AIB) | EU / EEA | European Residual Mix | 2024 publication (2023 data) | 2024 | AIB Terms of Use — parser marks `eu_publication` / registry marks `open` | https://www.aib-net.org/facts/european-residual-mix | Open (with mandatory attribution) | Yes | Yes | "European Residual Mixes, Association of Issuing Bodies (annual)." | Yes | Needs-Legal-Review | **Parser/registry disagreement:** parser sets `RedistributionClass.RESTRICTED` but registry labels `redistribution_class: open`. Legal must reconcile by reading AIB Terms of Use directly. |
| `india_cea_co2_baseline` | Central Electricity Authority, Ministry of Power, Government of India | India | CO2 Baseline Database for the Indian Power Sector | v20.0 (Dec 2024, FY 2023-24 data) | 2024 | Government of India public notification (parser tag: `public_in_government`) | https://cea.nic.in/cdm-co2-baseline-database/ | Open | Yes | Yes | "CO2 Baseline Database for the Indian Power Sector, Central Electricity Authority (Government of India), latest edition." | Yes | Safe-to-Certify | Parser `india_cea.py` sets `RedistributionClass.OPEN`. Indian public notifications are redistributable under Section 52(1)(q) of the Copyright Act (government works exception). |
| `india_ccts_baselines` | Bureau of Energy Efficiency (BEE), MoEFCC, Government of India | India | Carbon Credit Trading Scheme sectoral baselines | First compliance cycle (CCTS-1), triennial | 2023 notification + ongoing BEE gazettes | Government of India public notification | https://beeindia.gov.in/en/programmes/carbon-credit-trading-scheme | Open | Yes | Yes | "Carbon Credit Trading Scheme baseline emission intensities, BEE, MoEFCC, Government of India, notified under G.S.R. 443(E) dated 28 June 2023." | Yes | Safe-to-Certify | Parser `india_ccts.py` sets `OPEN`. Same copyright-exception posture as India CEA. |
| `australia_nga_factors` | Australian DCCEEW | Australia | National Greenhouse Accounts Factors | 2024 edition | 2024 | Australian Government Creative Commons Attribution 4.0 (CC-BY-4.0) | https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors | Open | Yes | Yes | "National Greenhouse Accounts Factors © Commonwealth of Australia (DCCEEW), licensed under CC BY 4.0." | Yes | Safe-to-Certify | Registry tags `open`. CC-BY-4.0 is explicit — attribution text MUST include licence link per CC BY 4.0 §3(a). |
| `nger_au_state_residual` | Australian Clean Energy Regulator / DCCEEW | Australia | NGER state-level residual mix (derived from NGA + LGC surrender) | annual | 2024 | CC-BY-4.0 (derived from NGA; LGC surrender data is CER public data) | https://www.cleanenergyregulator.gov.au/NGER | Open | Yes | Yes | "NGER state-level residual emission factors, Australian Clean Energy Regulator and DCCEEW, derived from NGA factors with LGC surrender netting, CC BY 4.0." | Yes | Safe-to-Certify | Registry tags `open`. Residual-mix derivation methodology must be cited per derivative requirement of CC-BY-4.0 §3(b). |
| `japan_meti_electric_emission_factors` | Japan METI + MOEJ | Japan | Electric Utility Emission Factors (基礎/調整後排出係数) | FY2022 (published 2024) | 2024 | Japan METI/MOEJ public government publication | https://ghg-santeikohyo.env.go.jp/calc | Open | Yes | Yes | "Electric Utility Emission Factors, Japan METI & MOEJ (annual)." | Yes | Needs-Legal-Review | Parser `japan_meti_residual.py` sets `OPEN`. Japan Government Publications Disclosure Policy (政府標準利用規約) is CC-BY-4.0-compatible; Legal to confirm specific dataset licence tag. |
| `cer_canada_residual` | Canada Energy Regulator (CER) + ECCC | Canada | Provincial Electricity Intensity Factors + residual-mix adjustment from NIR | annual | 2024 | Open Government Licence — Canada 2.0 (OGL-CA-2.0) | https://open.canada.ca/en/open-government-licence-canada | Open | Yes | Yes | "Contains information licensed under the Open Government Licence – Canada. Provincial Electricity Intensity Factors, Canada Energy Regulator." | Yes | Safe-to-Certify | **Parser missing.** Only registry entry. Legal: verify OGL-Canada 2.0 applies to the specific dataset (CER publishes under OGL-Canada; ECCC NIR similarly). |
| `kemco_korea_residual` | Korea Energy Management Corporation (KEMCO) / Korea Energy Agency | South Korea | Electricity residual mix | annual | 2024 | Republic of Korea Act on Disclosure of Information by Public Agencies — public information | https://www.gir.go.kr/eng/ | Open | Yes | Yes | "Republic of Korea national electricity residual mix, KEMCO / Korea Energy Agency; derived with KEC surrender netting from GIR national inventory." | Yes | Needs-Legal-Review | **Parser missing.** Korea's KOGL (Korea Open Government Licence) Type 1 is the typical tag; Legal to confirm specific dataset licence — Korea has multiple KOGL tiers (some block commercial use). |
| `ema_singapore_residual` | Singapore Energy Market Authority (EMA) | Singapore | Singapore Grid Emission Factor (derived with REC surrender netting) | annual | 2024 | Singapore Open Data Licence v1.0 | https://beta.data.gov.sg/open-data-license | Open | Yes | Yes | "Grid Emission Factor, Singapore Energy Market Authority (EMA), Singapore Open Data Licence v1.0." | Yes | Needs-Legal-Review | **Parser missing.** Singapore ODL is broadly permissive; Legal to confirm EMA publishes under ODL and not a restricted EMA licence. |
| `eu_cbam` | European Commission DG TAXUD | EU | CBAM default values (iron & steel, aluminum, cement, fertilizer, electricity, hydrogen) | 2024 implementation series | 2024 | EU Publications Office — official publication (parser tag: `EU-Publication`) | https://op.europa.eu/en/web/about-us/legal-notices/publications-office-of-the-european-union-copyright-notice | Open | Yes | Yes | "European Commission, Carbon Border Adjustment Mechanism default values, DG TAXUD." | Yes | Needs-Legal-Review | Parser `cbam_full.py`. EU Commission publications are reusable per Decision 2011/833/EU (with attribution), BUT specific CBAM default tables may be governed by Implementing Regulation (EU) 2023/1773 — Legal to confirm redistribution terms. |
| `ef_3_1_secondary` | European Commission Joint Research Centre (JRC) | EU | Environmental Footprint (EF) 3.1 Secondary Datasets | 3.1 | 2024 | EU Publications Office — official publication (registry tag: `eu_publication`) | https://eplca.jrc.ec.europa.eu/LCDN/developerEF.xhtml | Open | Yes | Yes | "European Commission JRC, Environmental Footprint (EF) 3.1 reference package." | Yes | Needs-Legal-Review | **Parser missing.** JRC terms vary by dataset — some EF 3.1 secondary datasets carry upstream ecoinvent / thinkstep licences that prohibit redistribution. Legal: read dataset-level licence headers. |
| `glec_framework` | Smart Freight Centre | Global | GLEC Framework v3.0 (ISO 14083-aligned) | 3.0 | 2023 | Smart Freight Centre Terms (registry tag: `smart_freight_terms`) | https://www.smartfreightcentre.org/en/our-programs/global-logistics-emissions-council/ | Licensed-Embedded | Yes | Yes | "GLEC Framework for Logistics Emissions Accounting and Reporting, Smart Freight Centre, v3.0 (2023)." | Restricted | Needs-Legal-Review | **Freight-lane parser `freight_lanes.py` tags `RedistributionClass.LICENSED` and `license_class: commercial_connector`.** Registry notes: "Redistribution requires Smart Freight Buyer Group membership; values allowed via API only for non-members." Legal to confirm whether factor values (not the Framework document itself) are separately licensable. |
| `ec3_buildings_epd` | Building Transparency (non-profit) | Global | EC3 (Embodied Carbon in Construction Calculator) EPD library | continuous | 2024 | EC3 API Terms — connector-only (registry tag: `commercial_connector`) | https://buildingtransparency.org/terms-of-service | Licensed-Embedded | Yes | Yes | "Embodied Carbon in Construction Calculator (EC3), Building Transparency." | No | Blocked-Contract-Required | Parser `ec3_epd.py` sets `RedistributionClass.LICENSED`. Connector-only. Each customer must bring their own EC3 token OR GreenLang must execute a premium EC3 data-redistribution agreement before Certified Construction pack ships. |
| *EPD International* | EPD International AB | Global | Environmental Product Declarations — International EPD System | continuous | continuous | Per-EPD licence (varies by program operator); International EPD System terms | https://www.environdec.com/terms-conditions | UNKNOWN — needs legal research | Per-EPD | Yes | Per-EPD citation | Per-EPD | Blocked-Contract-Required | **Parser missing** (EC3 parser covers some EPDs via EC3 aggregation, but not raw EPD International). Individual EPDs carry the program-operator and declarant's copyright. Do not scrape; negotiate direct data feed before GA. |
| `pcaf_global_std_v2` | Partnership for Carbon Accounting Financials (PCAF) | Global | PCAF Global GHG Accounting & Reporting Standard v2 (Parts A + B) | v2.0 (Part A) + Part B | 2022 (Part A), 2024 (Part B) | PCAF Attribution terms (registry tag: `pcaf_attribution`) | https://carbonaccountingfinancials.com/standard | Licensed-Embedded | Yes | Yes | "PCAF, The Global GHG Accounting and Reporting Standard for the Financial Industry, Part A (v2.0) & Part B." | Restricted | Needs-Legal-Review | Parser `pcaf_proxies.py` sets `RedistributionClass.RESTRICTED`. Registry: "Methodology text is public; derived proxy factors redistributable via GreenLang Premium Finance pack only." Legal to confirm PCAF derived-data policy. |
| `us_epa_suseeio` | US EPA National Risk Management Research Laboratory | US | Supply Chain GHG Emission Factors for US Industries and Commodities (SUSEEIO) v2 | v2 | 2024 | US Government work / public domain + Data.gov terms | https://catalog.data.gov/dataset/supply-chain-greenhouse-gas-emission-factors | Open | Yes | Yes | "US EPA, Supply Chain GHG Emission Factors for US Industries and Commodities (SUSEEIO), v2." | Yes | Safe-to-Certify | Registry `open`. **Parser missing** — add a parser for direct ingestion. |
| `exiobase_v3` | EXIOBASE Consortium (TU Wien / NTNU / UTwente) | Global | EXIOBASE v3 multi-regional EEIO | v3 | 2024 | EXIOBASE Consortium — research-free, commercial requires licence (registry tag: `academic_research`) | https://www.exiobase.eu/ | Licensed-Embedded | Commercial use requires licence | Yes | "EXIOBASE v3, EXIOBASE Consortium (2024)." | Restricted | Blocked-Contract-Required | **Parser missing.** Commercial redistribution explicitly requires EXIOBASE Consortium licence per registry `legal_notes`. Hold out of Certified edition until signed. |
| `ceda_pbe` | Profundo Research BV | Global | CEDA / PBE environmentally extended IO database | 2024 | 2024 | Commercial connector (registry tag: `commercial_connector`) | https://www.profundo.nl/ | Licensed-Embedded | Yes (with licence) | Yes | "CEDA environmentally extended IO database, Profundo Research BV." | Restricted | Blocked-Contract-Required | **Parser missing.** Customers must bring own CEDA licence or GreenLang negotiate OEM redistribution. |
| `electricity_maps` | Electricity Maps ApS | Global (200+ zones) | Electricity Maps real-time + historical grid intensity | continuous | 2024 | Electricity Maps SaaS subscription terms (connector-only) | https://www.electricitymaps.com/terms | Licensed-Embedded (connector) | Yes (with subscription) | Yes | "Electricity Maps; via connector under customer license." | No (per EM Terms) | Blocked-Contract-Required | Connector `electricity_maps.py`. Customer must provide own API token. Confirmed connector-only in registry (`redistribution_allowed: false`). |
| `iea` | International Energy Agency | Global | IEA statistics / factors | annual | 2024 | IEA commercial licence (connector-only) | https://www.iea.org/terms | Licensed-Embedded (connector) | Yes (with subscription) | Yes | "International Energy Agency; licensed use only." | No | Blocked-Contract-Required | Connector `iea.py`. IEA terms are notoriously restrictive — no redistribution of factor values. Legal: negotiate bulk licence or keep strictly connector-only for Certified. |
| `ecoinvent` | ecoinvent Association | Global | ecoinvent LCA database v3.x | v3.10 (default per connector) | 2024 | ecoinvent site licence (connector-only) | https://ecoinvent.org/the-ecoinvent-database/access-the-database/licenses/ | Licensed-Embedded (connector) | Yes (per-seat) | Yes | "ecoinvent Association; licensed use only." | No | Blocked-Contract-Required | Connector `ecoinvent.py`. ecoinvent per-seat licence prohibits redistribution. Customer must hold own ecoinvent licence; GreenLang ecoinvent connector must verify customer licence at runtime. |
| `freight_lanes` (synthetic ID used in parser) | GreenLang (operator overlay over GLEC) | Customer-overlay | Freight lane emission factors aligned with GLEC v3 / ISO 14083 | 2024 | 2024 | GLEC framework + customer-lane overlay | https://www.smartfreightcentre.org/en/our-programs/global-logistics-emissions-council/ | Licensed-Embedded (per GLEC parent source) | Yes | Yes | "GLEC Framework + customer-lane overlay; values derived per ISO 14083:2023." | Depends on GLEC | Needs-Legal-Review | Parser `freight_lanes.py` is the derivation layer — inherits GLEC's restricted posture. Do not publish as Open pack. |
| `waste_treatment` | GreenLang curated (from public waste treatment literature + IPCC Vol 5) | Global | Waste treatment factors | 2024 | 2024 | GreenLang curated — registry class unknown (parser sets `RedistributionClass.OPEN`) | n/a — internal derivation | Open | Yes | Yes | "GreenLang Factors curated waste treatment set; derived from IPCC 2006 Vol 5 + country-specific public inventories." | Yes | Needs-Legal-Review | Parser `waste_treatment.py`. No single publisher — Legal to confirm each underlying source in provenance chain before Certified. |
| `lsr_removals` | GHG Protocol Land Sector & Removals Standard | Global | LSR removal factors | 2024 (draft) | 2024 | WRI/WBCSD terms (parser sets `RedistributionClass.RESTRICTED`) | https://ghgprotocol.org/land-sector-and-removals-guidance | Licensed-Embedded | Yes | Yes | "GHG Protocol Land Sector & Removals Guidance, WRI/WBCSD." | Restricted | Needs-Legal-Review | Parser `lsr_removals.py`. Same WRI/WBCSD posture as GHG Protocol method refs. |
| `greenlang_builtin` | GreenLang | Global | GreenLang curated built-in factors | on_release | 2024 | GreenLang Terms (registry tag: `greenlang_terms`) | — | Open (GreenLang chooses terms) | Yes | Yes | "GreenLang Factors curated edition." | Yes | Safe-to-Certify | Registry: this is GreenLang's own fallback pack. Ensure the curated set's provenance chain is itself Safe-to-Certify (i.e., do not pull from restricted upstream into this "curated" layer). |

---

## 3. Licensing Summary

### 3.1 Open sources — safe for Community tier
These can ship in the free public pack and Certified edition bulk export. Attribution text must still render in API responses.
- `epa_hub` (EPA GHG Emission Factors Hub)
- `egrid` (EPA eGRID)
- `desnz_ghg_conversion` (UK DESNZ conversion factors)
- `defra_conversion` (UK DEFRA — confirm duplicate of DESNZ)
- `beis_uk_residual` (UK DESNZ residual mix)
- `india_cea_co2_baseline` (India CEA)
- `india_ccts_baselines` (India CCTS)
- `australia_nga_factors` (Australia NGA)
- `nger_au_state_residual` (Australia NGER state residual)
- `cer_canada_residual` (Canada CER)
- `us_epa_suseeio` (US EPA SUSEEIO)
- `japan_meti_electric_emission_factors` (pending Legal confirmation)
- `ema_singapore_residual` (pending Legal confirmation)
- `kemco_korea_residual` (pending Legal confirmation)
- `aib_residual_mix_eu` (pending parser/registry reconciliation)
- `ipcc_2006_nggi`, `ipcc_2006_afolu_v2019` (pending IPCC copyright confirmation)
- `greenlang_builtin` (GreenLang curated — internal provenance)

### 3.2 Licensed-Embedded — must gate behind Premium Pack entitlement
These can ship factor values only to entitled tenants. No public bulk export. Each requires attribution text and connector-side entitlement checks (factors tagged with `license_class=commercial_connector` or `redistribution_class=RESTRICTED`).
- `ghgp_method_refs` (GHG Protocol method refs — Premium Methodology pack)
- `pact_pathfinder` (PACT Pathfinder Framework — Premium Product pack)
- `tcr_grp_defaults` (TCR GRP defaults — US Premium pack)
- `green_e_residual` / `green_e_residual_mix` (Green-e — Premium Electricity-US pack)
- `glec_framework` (GLEC — Premium Freight pack)
- `ec3_buildings_epd` (EC3 — Premium Construction pack)
- `pcaf_global_std_v2` (PCAF proxies — Premium Finance pack)
- `exiobase_v3` (EXIOBASE — Premium Finance/Spend pack)
- `ceda_pbe` (CEDA — Premium Finance pack, OEM if redistributed)
- `electricity_maps` (Electricity Maps — connector with customer token)
- `iea` (IEA — connector with customer subscription)
- `ecoinvent` (ecoinvent — connector with customer seat licence)
- `freight_lanes` (inherits GLEC posture)
- `lsr_removals` (inherits WRI/WBCSD posture)

### 3.3 Sources requiring a signed contract BEFORE Certified edition cut
> **CTO / Legal action: each of these must have an executed data licence or membership agreement on file BEFORE the Certified-edition source is flipped to `approval_required_for_certified: true` + `legal_signoff_artifact: <signed-doc-id>`.**

| Source | What must be signed | Licence page |
|---|---|---|
| `ecoinvent` | ecoinvent Site Licence (commercial, per-seat) — plus OEM redistribution addendum if GreenLang embeds in Premium pack | https://ecoinvent.org/the-ecoinvent-database/access-the-database/licenses/ |
| `iea` | IEA data subscription + redistribution addendum (IEA's default terms do NOT permit redistribution — negotiate) | https://www.iea.org/terms |
| `electricity_maps` | Electricity Maps commercial subscription (connector), or OEM master-tenant agreement | https://www.electricitymaps.com/terms |
| `ec3_buildings_epd` | Building Transparency EC3 data partnership agreement (currently permissioned API) | https://buildingtransparency.org/terms-of-service |
| `tcr_grp_defaults` | The Climate Registry data-use agreement (the GRP is free to download but has redistribution restrictions) | https://theclimateregistry.org/tcr-general-reporting-protocol/ |
| `green_e_residual` / `green_e_residual_mix` | Center for Resource Solutions Green-e data licence | https://www.green-e.org/terms-and-conditions |
| `glec_framework` | Smart Freight Buyer Group membership OR bespoke data agreement | https://www.smartfreightcentre.org/en/our-programs/global-logistics-emissions-council/ |
| `exiobase_v3` | EXIOBASE Consortium commercial licence | https://www.exiobase.eu/ |
| `ceda_pbe` | Profundo CEDA commercial licence + OEM addendum | https://www.profundo.nl/ |
| *EPD International raw EPDs* | Per-EPD / program-operator agreements (if GreenLang wants raw EPD pass-through, not via EC3 aggregator) | https://www.environdec.com/terms-conditions |

### 3.4 Sources with unclear license — flag for Legal review
- `aib_residual_mix_eu` — parser vs. registry disagree on redistribution class; Legal to read AIB Terms directly
- `ipcc_2006_nggi`, `ipcc_2006_afolu_v2019`, AR4/AR5/AR6 GWP tables — need IPCC copyright confirmation (numbers-as-facts is defensible but Legal should document)
- *IPCC EFDB* — no parser in repo; terms-of-use page requires review
- `ef_3_1_secondary` — JRC licence varies by dataset inside the bundle
- `ghgp_method_refs` — boundary between "methodology text" (restricted) and "factor values derived via Scope 3 tool" (ambiguous)
- `japan_meti_electric_emission_factors` — Japan 政府標準利用規約 applies but per-dataset licence tag needs confirmation
- `kemco_korea_residual` — KOGL type (1 vs. 2 vs. 3) determines commercial redistribution
- `ema_singapore_residual` — confirm ODL vs. EMA-specific licence
- `defra_conversion` — confirm whether this is a separate dataset or naming alias of `desnz_ghg_conversion`
- `waste_treatment` — provenance chain for each row needs Legal-approved source mapping

### 3.5 Attribution text required (exact strings per source)
Legal must confirm each string; the following are what the codebase currently emits:

| Source | Required attribution text |
|---|---|
| `desnz_ghg_conversion`, `defra_conversion`, `beis_uk_residual` | "Contains public sector information licensed under the Open Government Licence v3.0" + dataset-specific line |
| `cer_canada_residual` | "Contains information licensed under the Open Government Licence – Canada" |
| `australia_nga_factors`, `nger_au_state_residual` | "© Commonwealth of Australia (DCCEEW), licensed under CC BY 4.0" |
| `ema_singapore_residual` | "Singapore Open Data Licence v1.0 — Singapore Energy Market Authority" |
| `epa_hub`, `egrid`, `us_epa_suseeio` | "U.S. Environmental Protection Agency, [dataset name]." |
| `india_cea_co2_baseline` | "CO2 Baseline Database for the Indian Power Sector, Central Electricity Authority (Government of India), latest edition." |
| `india_ccts_baselines` | "Carbon Credit Trading Scheme baseline emission intensities, BEE, MoEFCC, Government of India, notified under G.S.R. 443(E) dated 28 June 2023." |
| `aib_residual_mix_eu` | "European Residual Mixes, Association of Issuing Bodies (annual)." |
| `ipcc_2006_nggi`, `ipcc_2006_afolu_v2019` | "IPCC 2006 Guidelines for National Greenhouse Gas Inventories, [Volume], 2019 Refinement." |
| `ghgp_method_refs` | "Greenhouse Gas Protocol, WRI/WBCSD, [Standard name], [year]." |
| `pact_pathfinder` | "WBCSD, Pathfinder Framework: Guidance for the Accounting and Exchange of Product Life Cycle Emissions, v3.0." |
| `glec_framework` | "GLEC Framework for Logistics Emissions Accounting and Reporting, Smart Freight Centre, v3.0 (2023)." |
| `pcaf_global_std_v2` | "PCAF, The Global GHG Accounting and Reporting Standard for the Financial Industry, Part A (v2.0) & Part B." |
| `ec3_buildings_epd` | "Embodied Carbon in Construction Calculator (EC3), Building Transparency." |
| `green_e_residual` / `green_e_residual_mix` | "Green-e Energy Residual Mix, Center for Resource Solutions (annual)." |
| `tcr_grp_defaults` | "The Climate Registry, General Reporting Protocol default factors." |
| `eu_cbam` | "European Commission, Carbon Border Adjustment Mechanism default values, DG TAXUD." |
| `ef_3_1_secondary` | "European Commission JRC, Environmental Footprint (EF) 3.1 reference package." |
| `japan_meti_electric_emission_factors` | "Electric Utility Emission Factors, Japan METI & MOEJ (annual)." |
| `ecoinvent`, `iea`, `electricity_maps`, `exiobase_v3`, `ceda_pbe` | Per connector licence — rendered in API receipt envelope, not in bulk export (bulk export MUST NOT include these) |

### 3.6 Residual-mix sources — methodological note (MUST NOT be mixed into location-based factors)
Per GHG Protocol Scope 2 Quality Criteria, **residual mix is the factor used in the MARKET-BASED accounting method** when a reporting entity has no contractual instrument (REC / GO / I-REC / utility green tariff / PPA) claim. Residual mix is derived by netting out already-claimed low-carbon MWh from the grid average, so the remaining pool is attributable to non-claimants. This makes residual mix systematically higher than the grid average for jurisdictions with active certificate markets.

**Implication for GreenLang:**
1. Residual-mix sources must never be returned under a `basis=LOCATION_BASED` resolver request — that would be a methodological error and a potential compliance defect for customers.
2. Storage: separate tables / separate `FactorFamily` / explicit `ElectricityBasis.RESIDUAL_MIX` tag (already enforced via `greenlang/data/canonical_v2.ElectricityBasis`).
3. API: `/resolve` endpoint must require explicit `basis=market_based` (or `residual_mix`) for the resolver to even consider residual-mix rows.
4. Certified edition bulk exports must separate residual-mix and location-based rows into distinct pack files so downstream consumers cannot accidentally average them.

**Residual-mix sources currently in registry:**
- `aib_residual_mix_eu` (EU/EEA)
- `green_e_residual` + `green_e_residual_mix` (US + CA)
- `beis_uk_residual` (UK)
- `cer_canada_residual` (CA provincial)
- `nger_au_state_residual` (AU)
- `japan_meti_electric_emission_factors` (derivation in `japan_meti_residual.py`)
- `kemco_korea_residual` (KR)
- `ema_singapore_residual` (SG)

---

## 4. Top-5 Legal-Risk Sources (Ranked)

Ranked by risk of blocking GA or of creating a redistribution-liability exposure after GA.

### 1. `ecoinvent`
- **Risk:** Per-seat licence; default terms explicitly prohibit redistribution. If a customer's API response ever contains an ecoinvent factor value without the customer holding their own ecoinvent seat, that is a licence breach and a potential enforcement event. Largest LCI coverage (~20K activities) makes this the biggest commercial lever and the biggest liability.
- **Action to close:** (a) Execute ecoinvent data-licence agreement with redistribution rights for GreenLang Premium LCI pack, OR (b) lock the connector to customer-supplied credentials and document "BYO ecoinvent" explicitly in the Premium pack terms.
- **Timeline if Legal starts this week:** 60–90 days (per `LICENSED_CONNECTORS.md`). Outreach + negotiation + redlines.

### 2. `iea`
- **Risk:** IEA's default licence is the most restrictive of all the major energy-data publishers — no redistribution, no derivative publication of factor values, attribution required. Shipping any IEA-derived factor in a bulk Certified export is an unambiguous licence breach.
- **Action to close:** Execute IEA data-licence with redistribution addendum (historically expensive and tightly scoped). Fallback: keep IEA strictly connector-only, never in Certified bulk pack, and document the restriction in DX9.
- **Timeline:** 30–60 days for subscription; redistribution addendum often requires months of negotiation. Assume 60–90 days.

### 3. `green_e_residual` + `green_e_residual_mix`
- **Risk:** These are the ONLY public source for US + Canada market-based residual mix. Without them, GreenLang cannot ship a Scope 2 market-based pack for US/CA in Certified. CRS (Center for Resource Solutions) explicitly prohibits redistribution without licence. The registry has two records (old + new) that must be deduped.
- **Action to close:** Execute CRS data-licence agreement. Concurrently, reconcile `green_e_residual` and `green_e_residual_mix` into a single `source_id` with a migration note in the release.
- **Timeline:** 30–45 days (CRS is smaller and less bureaucratic than IEA/ecoinvent).

### 4. `glec_framework` / `ec3_buildings_epd`
- **Risk (joint):** Both gate GreenLang's Freight Premium pack and Construction Premium pack respectively. GLEC redistribution requires Smart Freight Buyer Group membership; EC3 API is permissioned. Without these, slices 3 and 4 of the v1 Certified cut-list cannot ship.
- **Action to close:** Join Smart Freight Buyer Group (membership tier covers ISO 14083 alignment). Separately, execute Building Transparency data-partnership agreement for EC3 OEM redistribution.
- **Timeline:** GLEC membership 30 days; EC3 60 days. Can run in parallel.

### 5. IPCC copyright posture (covers `ipcc_2006_nggi`, `ipcc_2006_afolu_v2019`, EFDB, AR4/AR5/AR6 GWPs)
- **Risk:** IPCC guidelines and AR reports are technically copyrighted — IPCC's copyright page requires written permission for "substantial extracts". The defence is (a) numerical constants are facts and unprotectable, and (b) attribution-with-reproduction is permitted for research/education. Neither defence is strong enough to rely on for a commercial SaaS without explicit Legal sign-off. EFDB is particularly vulnerable because values are submitted by member states with per-record terms.
- **Action to close:** (a) Written confirmation from IPCC TFI Unit (Tokyo) that factor-value reproduction in a commercial SaaS is permitted with attribution, OR (b) Legal opinion documenting the numerical-fact doctrine defence and recording the risk decision in a board memo.
- **Timeline:** IPCC correspondence typically 30–60 days; Legal opinion memo 10–15 business days.

---

## 5. L1 Data-Class Policy — Reference

Per CTO Master ToDo task L1, every source above maps to exactly ONE of these four data classes:

| Data class | Storage policy | Export policy | Example sources |
|---|---|---|---|
| **Open** | Public bucket, no entitlement check required | Can appear in Community-tier bulk pack AND Certified edition bulk pack | `epa_hub`, `egrid`, `desnz_ghg_conversion`, `india_cea_co2_baseline`, `australia_nga_factors` |
| **Licensed-Embedded** | Separate namespace with per-pack entitlement | API only; NEVER in bulk Certified export | `ghgp_method_refs`, `pact_pathfinder`, `glec_framework`, `pcaf_global_std_v2`, `ec3_buildings_epd`, `green_e_residual` |
| **Customer-Private** | Tenant-scoped namespace; zero cross-tenant visibility | Never exported outside tenant | Tenant overlay factors via `greenlang/factors/tenant_overlay.py` |
| **OEM-Redistributable** | Separate namespace flagged for OEM tenants | Only OEM tenants can redistribute to sub-tenants with GreenLang's upstream contract | TBD — no source currently tagged OEM-Redistributable; populate once ecoinvent/EC3/CEDA OEM deals close |

L9 (`License scanner before certification`) must fail the Certified build if any source row missing `legal_signoff_artifact` is flagged in the `v1_gate_status` column above as `Blocked-Contract-Required`.

---

## 6. Open Items for Legal

1. Reconcile `desnz_ghg_conversion` vs. `defra_conversion` (duplicate or separate dataset?).
2. Reconcile `green_e_residual` vs. `green_e_residual_mix` (two registry rows for same publisher).
3. Reconcile `aib_residual_mix_eu` parser-vs-registry license-class disagreement (`RESTRICTED` vs. `open`).
4. Produce written IPCC permission OR legal-memo covering numerical-fact doctrine for IPCC guidelines + AR GWP tables + EFDB.
5. Produce OGL v3 + OGL-Canada 2.0 + CC-BY 4.0 attribution rendering spec: the exact string template the API response and PDF export must include per source.
6. Produce the "Licensed-Embedded redistribution matrix" used by the resolver's entitlement check (L9 / license scanner).
7. Add missing parsers for: IPCC EFDB, `cer_canada_residual`, `kemco_korea_residual`, `ema_singapore_residual`, `us_epa_suseeio`, `exiobase_v3`, `ceda_pbe`, `ef_3_1_secondary`. Until parsers exist, rows stay in registry only and must not ship factor values.
8. Populate `legal_signoff_artifact` field for every row in `source_registry.yaml` as contracts close — this is the machine-readable gate that L9 license scanner keys off.
