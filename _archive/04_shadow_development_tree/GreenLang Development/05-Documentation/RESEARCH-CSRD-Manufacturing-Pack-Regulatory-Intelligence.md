# CSRD Manufacturing Pack -- Regulatory Intelligence Research

**Version**: 1.0
**Date**: 2026-03-16
**Author**: GL-RegulatoryIntelligence Agent
**Purpose**: Comprehensive regulatory research to inform the CSRD Manufacturing Pack PRD
**Status**: COMPLETE

---

## Table of Contents

1. [ESRS Sector Standards for Manufacturing](#1-esrs-sector-standards-for-manufacturing)
2. [Manufacturing Emissions Profile](#2-manufacturing-emissions-profile)
3. [EU ETS and CBAM for Manufacturers](#3-eu-ets-and-cbam-for-manufacturers)
4. [Industrial Decarbonization Pathways](#4-industrial-decarbonization-pathways)
5. [Product Carbon Footprint (PCF)](#5-product-carbon-footprint-pcf)
6. [Circular Economy and Waste](#6-circular-economy-and-waste)
7. [Water and Pollution for Manufacturing](#7-water-and-pollution-for-manufacturing)
8. [Biodiversity and Land Use for Manufacturing](#8-biodiversity-and-land-use-for-manufacturing)
9. [Social Standards in Manufacturing](#9-social-standards-in-manufacturing)
10. [Manufacturing-Specific KPIs](#10-manufacturing-specific-kpis)
11. [BAT and Technology Reference](#11-bat-and-technology-reference)
12. [Supply Chain Emissions for Manufacturers](#12-supply-chain-emissions-for-manufacturers)

---

## 1. ESRS Sector Standards for Manufacturing

### 1.1 Current Status After Omnibus I (February 2026)

**Key Finding**: Sector-specific ESRS standards have been eliminated by the Omnibus I Directive.

On 26 February 2026, the EU published Directive (EU) 2026/470 (the Omnibus I Directive), which effectuates far-reaching changes to both the CSRD and the CSDDD. The original plan included sector-specific standards:

- **ESRS SEC1**: Sector Classification Standard (was to define NACE-based sector groupings)
- **ESRS SEC2**: Originally planned for Oil and Gas; mining/quarrying also in development
- **Manufacturing-adjacent sectors**: Automotive, chemicals, textiles, construction were in EFRAG's sector development pipeline

**All sector-specific ESRS have been permanently removed from the legislative mandate.** Companies must instead rely on:
- Cross-sector ESRS standards (ESRS E1-E5, S1-S4, G1) applied via materiality assessment
- Entity-specific disclosures where material sustainability issues are not adequately covered by existing ESRS

### 1.2 Revised ESRS Framework Post-Omnibus

| Change | Detail |
|--------|--------|
| Data point reduction | 61% reduction in mandatory data points (EFRAG November 2025 submission) |
| Materiality first | ALL topical disclosures (including Climate E1) are now 100% subject to materiality |
| Value chain cap | Companies with <=1,000 employees exempt from providing data beyond "voluntary standards" scope |
| Scope threshold | Only EU entities with >1,000 employees AND >EUR 450M net turnover are in scope |
| Reporting timeline | Revised ESRS target application for FY beginning on/after 1 January 2027; voluntary early use for FY2026 |
| Wave 2/3 postponement | Companies due to report in 2026 and 2027 receive postponement |

### 1.3 Implications for the Manufacturing Pack

Since sector-specific ESRS will not exist, the Manufacturing Pack must:

1. **Build sector intelligence internally** -- provide manufacturing-specific guidance, benchmarks, and KPIs that the regulation itself no longer mandates at sector level
2. **Leverage cross-sector ESRS comprehensively** -- apply ESRS E1-E5, S1-S4, G1 with manufacturing materiality logic
3. **Use entity-specific disclosures** -- generate entity-specific disclosures for manufacturing topics not covered by standard ESRS (process emissions, product carbon footprints, BAT compliance, etc.)
4. **Maintain competitive advantage** -- sector-specific intelligence becomes a product differentiator since the regulation does not require it

### 1.4 Manufacturing NACE Classification

Manufacturing spans NACE Rev. 2.1 Division C (Codes C10-C33):

| NACE Code | Sub-sector | Emission Profile |
|-----------|------------|-----------------|
| C10-C12 | Food, Beverages, Tobacco | Energy-intensive processing, agricultural supply chain |
| C13-C15 | Textiles, Leather | Water-intensive, chemical use, labor supply chain |
| C16-C18 | Wood, Paper, Printing | Biomass, process emissions, forestry supply chain |
| C19 | Coke and Petroleum | EU ETS covered, high Scope 1 |
| C20-C21 | Chemicals, Pharmaceuticals | Process emissions, SVHC/REACH, water discharge |
| C22-C23 | Rubber/Plastics, Non-metallic minerals | Cement/glass/ceramics process emissions |
| C24-C25 | Basic metals, Fabricated metals | EU ETS, CBAM, high Scope 1 |
| C26-C27 | Electronics, Electrical equipment | Supply chain Scope 3 dominant, conflict minerals |
| C28-C30 | Machinery, Vehicles, Transport equipment | Scope 3 Cat 11 (use of sold products) dominant |
| C31-C33 | Furniture, Other manufacturing, Repair | Mixed profiles, circular economy focus |

### 1.5 Regulatory References

- Directive (EU) 2026/470 (Omnibus I Directive), published 26 February 2026
- Delegated Regulation (EU) 2023/2772 (original ESRS standards), 31 July 2023
- EFRAG Amended ESRS submission, November 2025
- EFRAG SEC1 Working Paper on Sector Classification (September 2024)

---

## 2. Manufacturing Emissions Profile

### 2.1 Scope 1 Emissions -- Direct Emissions from Owned/Controlled Sources

| Source Category | GHG Protocol Source | Applicable Sub-sectors | GreenLang Agent |
|----------------|--------------------|-----------------------|-----------------|
| Stationary combustion | Boilers, furnaces, kilns, turbines, heaters | All manufacturing | AGENT-MRV-001 |
| Process emissions | Chemical/physical transformation of materials | Cement, steel, glass, chemicals, aluminum | AGENT-MRV-004 |
| Fugitive emissions | Equipment leaks, venting, flaring | Chemicals, petroleum, gas-using industries | AGENT-MRV-005 |
| Refrigerant leakage | HFCs, PFCs from industrial cooling | Food/beverage, pharmaceuticals, electronics | AGENT-MRV-002 |
| Mobile combustion | Forklifts, on-site vehicles, mobile equipment | All manufacturing | AGENT-MRV-003 |
| Land use change | On-site land management | Extractive-adjacent manufacturing | AGENT-MRV-006 |
| Waste treatment | On-site waste incineration, composting | All manufacturing with on-site treatment | AGENT-MRV-007 |

#### Key Calculation Methodologies

**Stationary Combustion** (AGENT-MRV-001):
- Fuel-based: Emissions = Fuel consumption (GJ) x Emission factor (tCO2e/GJ)
- Direct measurement: CEMS (Continuous Emissions Monitoring Systems) for EU ETS installations
- Gases: CO2, CH4, N2O; GWP per IPCC AR6 (CO2=1, CH4=27.9, N2O=273)

**Process Emissions** (AGENT-MRV-004):
- Cement clinker: CO2 from CaCO3 calcination = 0.525 tCO2/t clinker (stoichiometric)
- Iron/steel: CO2 from reduction = varies by route (BF-BOF: ~1.85 tCO2/t; EAF: ~0.4 tCO2/t)
- Aluminum: CO2 from anode consumption + PFC from anode effects
- Glass: CO2 from raw material decomposition (soda ash, limestone)
- Chemicals: Process-specific (e.g., ammonia production: ~1.6 tCO2/t NH3)

### 2.2 Scope 2 Emissions -- Indirect from Purchased Energy

| Energy Type | Methodology | GreenLang Agent |
|-------------|------------|-----------------|
| Electricity | Location-based (grid average EF) + Market-based (contractual EF) | AGENT-MRV-009 + AGENT-MRV-010 |
| Steam/heat | Purchased steam EF from supplier or default | AGENT-MRV-011 |
| Cooling | Purchased chilled water/cooling EF | AGENT-MRV-012 |
| Dual reporting | Reconciliation between location and market | AGENT-MRV-013 |

**Dual Reporting** is mandatory under GHG Protocol Scope 2 Guidance (2015) and ESRS E1.
- Location-based: grid emission factors from IEA, national inventories, or eGRID (US)
- Market-based: RECs, GOs, PPAs, green tariffs, residual mix factor

### 2.3 Scope 3 Emissions -- Value Chain

#### Dominant Scope 3 Categories by Manufacturing Sub-sector

| Sub-sector | Dominant Categories | % of Total Scope 3 | Key Driver |
|-----------|--------------------|--------------------|------------|
| **Automotive** | Cat 1 (purchased goods), Cat 11 (use of sold products) | Cat 1: 25-35%, Cat 11: 40-55% | Steel/aluminum/battery materials; vehicle fuel/electricity consumption |
| **Chemicals** | Cat 1 (raw materials), Cat 10 (processing), Cat 11 (use) | Cat 1: 30-50%, Cat 11: 20-40% | Feedstock extraction; downstream chemical use |
| **Food & Beverage** | Cat 1 (agricultural inputs), Cat 4 (transport), Cat 12 (end-of-life packaging) | Cat 1: 50-75% | Agriculture (meat/dairy = ~50% of Cat 1) |
| **Textiles** | Cat 1 (fiber production), Cat 11 (care/washing), Cat 12 (disposal) | Cat 1: 40-60%, Cat 11: 15-25% | Nylon/polyester from fossil fuels; dyeing energy intensity |
| **Electronics** | Cat 1 (components), Cat 11 (electricity in use phase) | Cat 1: 30-50%, Cat 11: 30-50% | Semiconductor manufacturing; product energy consumption |
| **Steel/metals** | Cat 1 (ore/scrap), Cat 3 (fuel/energy), Cat 4 (transport) | Cat 1: 20-40%, Cat 3: 15-25% | Raw material extraction; coking coal upstream |
| **Cement** | Cat 1 (raw materials), Cat 3 (fuel/energy), Cat 9 (downstream transport) | Cat 3: 20-30%, Cat 9: 10-20% | Fossil fuel upstream; bulk product transport |
| **Machinery/equipment** | Cat 1 (components), Cat 11 (energy in use) | Cat 1: 30-45%, Cat 11: 25-40% | Materials/components; operational energy |
| **Pharmaceuticals** | Cat 1 (chemicals/APIs), Cat 4 (transport), Cat 5 (waste) | Cat 1: 40-60% | Chemical feedstocks; cold chain logistics |

#### Full Scope 3 Category Coverage for Manufacturing

| Category | Name | Manufacturing Relevance | GreenLang Agent |
|----------|------|------------------------|-----------------|
| Cat 1 | Purchased goods & services | **Critical** -- raw materials, components | AGENT-MRV-014 |
| Cat 2 | Capital goods | Moderate -- machinery, equipment | AGENT-MRV-015 |
| Cat 3 | Fuel & energy related (not in Scope 1/2) | **High** -- upstream fuel extraction, T&D losses | AGENT-MRV-016 |
| Cat 4 | Upstream transportation & distribution | **High** -- inbound logistics | AGENT-MRV-017 |
| Cat 5 | Waste generated in operations | **High** -- industrial waste streams | AGENT-MRV-018 |
| Cat 6 | Business travel | Low-moderate | AGENT-MRV-019 |
| Cat 7 | Employee commuting | Low-moderate | AGENT-MRV-020 |
| Cat 8 | Upstream leased assets | Low (unless significant leased facilities) | AGENT-MRV-021 |
| Cat 9 | Downstream transportation | Moderate-high (heavy/bulk products) | AGENT-MRV-022 |
| Cat 10 | Processing of sold products | **High** (intermediate goods manufacturers) | AGENT-MRV-023 |
| Cat 11 | Use of sold products | **Critical** (energy-using products) | AGENT-MRV-024 |
| Cat 12 | End-of-life treatment | **High** (EPR-covered products) | AGENT-MRV-025 |
| Cat 13 | Downstream leased assets | Low (unless lessor of equipment) | AGENT-MRV-026 |
| Cat 14 | Franchises | Rarely applicable | AGENT-MRV-027 |
| Cat 15 | Investments | Low (unless holding company) | AGENT-MRV-028 |

### 2.4 Regulatory References

- GHG Protocol Corporate Standard (Revised, 2004/2015)
- GHG Protocol Scope 2 Guidance (2015)
- GHG Protocol Scope 3 Standard (2011) + Technical Guidance for Calculating Scope 3 Emissions
- GHG Protocol Product Life Cycle Accounting and Reporting Standard (2011)
- IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 3 (Industrial Processes)
- IPCC AR6 GWP values (2021/2022)
- CDP Technical Note: Relevance of Scope 3 Categories by Sector

---

## 3. EU ETS and CBAM for Manufacturers

### 3.1 EU ETS Phase IV (2021-2030)

EU ETS (Directive 2003/87/EC as amended by Directive (EU) 2023/959) directly affects manufacturing installations above capacity thresholds.

#### Covered Manufacturing Sectors and Thresholds

| Sector | Activity | Capacity Threshold |
|--------|----------|-------------------|
| Cement/clinker | Production of cement clinker | All capacities (kilns) |
| Iron and steel | Production and processing | >2.5 t/h crude steel |
| Aluminum | Primary aluminum smelting | All capacities |
| Glass | Manufacture of glass | >20 t/day |
| Ceramics | Manufacture of ceramic products (firing) | >75 t/day |
| Pulp/paper | Production of pulp or paper/cardboard | >20 t/day |
| Chemicals | Bulk organic/inorganic chemicals | Various thresholds |
| Refineries | Refining of mineral oil | All capacities |
| Lime | Production of lime | >50 t/day |
| **NEW -- Batteries** | Large battery manufacturing | >15,000 t/year (IED 2.0) |

#### Phase IV Key Parameters

| Parameter | Value |
|-----------|-------|
| Linear reduction factor | 4.3% per year (from 2024, up from 2.2%) |
| Cap reduction target | 62% below 2005 levels by 2030 |
| Free allocation basis | Product benchmarks updated per Commission Delegated Regulation |
| Carbon Leakage List | Updated 2021-2025, then 2026-2030 |
| Auction revenue | Innovation Fund (EUR 40B+) and Modernisation Fund |
| Maritime inclusion | Full from 2026 |
| Aviation | Full auctioning for intra-EEA from 2026 |

#### EU ETS Compliance Calendar for Manufacturers

| Deadline | Obligation | Article Reference |
|----------|-----------|------------------|
| January 31 | Submit monitoring methodology plan updates | Art. 14(1) |
| March 31 | Submit verified annual emissions report (AER) | Art. 14(3) |
| April 30 | Verified tonnes determination by accredited verifier | Art. 15 |
| September 30 | Surrender allowances equal to verified emissions | Art. 12(3) |
| Ongoing | Record keeping for 10 years minimum | Art. 14(2) |

### 3.2 CBAM -- Carbon Border Adjustment Mechanism

**Regulation (EU) 2023/956** (CBAM Regulation) + **Implementing Regulation (EU) 2023/1773**.

#### CBAM Scope -- Covered Goods (Annex I)

| Sector | CN Codes (examples) | Covered Products |
|--------|---------------------|-----------------|
| Cement | 2523 10 00 - 2523 90 00 | Portland, aluminous, hydraulic cements |
| Iron & Steel | 72xx, 73xx | Pig iron, ferro-alloys, flat/long products, tubes, wire |
| Aluminum | 7601 - 7616 | Unwrought aluminum, bars, wire, plates, foil, structures |
| Fertilizers | 2808, 2814, 3102-3105 | Nitric acid, ammonia, nitrates, phosphates, mixed |
| Electricity | 2716 | Electrical energy |
| Hydrogen | 2804 10 00 | Hydrogen gas |

#### CBAM Phase-In / Free Allocation Phase-Out Schedule

| Year | CBAM Factor (% of full carbon cost on imports) | Remaining Free Allocation (% for EU producers) |
|------|-----------------------------------------------|------------------------------------------------|
| 2026 | 2.5% | 97.5% |
| 2027 | 5% | 95% |
| 2028 | 10% | 90% |
| 2029 | 22.5% | 77.5% |
| 2030 | 48.5% | 51.5% |
| 2031 | 61% | 39% |
| 2032 | 73.5% | 26.5% |
| 2033 | 86% | 14% |
| 2034 | 100% | 0% |

#### CBAM Embedded Emissions Calculation

For simple goods:
```
Embedded emissions (tCO2/t product) = Direct specific emissions + Indirect specific emissions

Direct specific emissions = Attribution factor x (Direct emissions from installation / Activity level)

Indirect specific emissions = Electricity consumption per tonne x Grid emission factor
```

For complex goods (containing precursors):
```
Embedded emissions = Direct emissions from final production + SUM(Precursor embedded emissions x Mass of precursor per tonne of product)
```

#### CBAM Key Obligations

| Obligation | Timeline | Penalty |
|-----------|----------|---------|
| Quarterly CBAM Reports | Oct 2023 - Dec 2025 (transitional) | Administrative sanctions per Member State |
| Authorized Declarant registration | Before Jan 2026 | Cannot import CBAM goods without authorization |
| Annual CBAM Declaration | By May 31 each year (from 2027 for 2026) | EUR 10-50 per tCO2e undeclared |
| CBAM Certificate purchase | Weekly from national authority | Price = weekly average EU ETS auction price |
| Certificate surrender | By May 31 (min 80% by end of each quarter) | EUR 100/tonne excess emissions not covered |
| Verification of embedded emissions | From 2026 definitive period | Must use accredited verifier |

### 3.3 Impact on Different Manufacturer Types

| Manufacturer Type | EU ETS Impact | CBAM Impact (as importer) | CBAM Impact (as exporter) |
|-------------------|---------------|--------------------------|--------------------------|
| EU steel producer | Direct compliance, free allocation declining | Importing scrap/ore may trigger | Export rebate discussion ongoing (not yet adopted) |
| EU cement producer | Direct compliance, no carbon leakage protection post-2034 | Importing clinker triggers CBAM | No export adjustment yet |
| EU aluminum smelter | Direct compliance (electricity-intensive) | Importing aluminum triggers | Under discussion |
| EU downstream manufacturer (cars, machinery) | Usually below EU ETS threshold | Importing steel/aluminum components triggers CBAM | Indirect exposure through input costs |
| Non-EU manufacturer selling to EU | Not applicable | Their EU importers must declare embedded emissions | Must provide emissions data to EU importers |

### 3.4 Regulatory References

- Directive 2003/87/EC (EU ETS) as amended by Directive (EU) 2023/959
- Regulation (EU) 2023/956 (CBAM Regulation)
- Implementing Regulation (EU) 2023/1773 (CBAM transitional rules)
- Commission Implementing Regulation (EU) 2024/XXX (CBAM definitive period simplifications, adopted late 2025)
- Regulation (EU) 2023/857 (Effort Sharing Regulation -- for non-ETS manufacturing)
- Free allocation benchmarks: Commission Delegated Regulation (EU) 2019/331

---

## 4. Industrial Decarbonization Pathways

### 4.1 EU Fit for 55 Package -- Manufacturing-Relevant Components

The Fit for 55 legislative package (adopted 2021-2024) includes multiple instruments affecting manufacturers:

| Instrument | Key Manufacturing Obligation | Status |
|-----------|------------------------------|--------|
| EU ETS reform (Directive 2023/959) | Tighter cap, 62% reduction by 2030, expanded scope | In force |
| CBAM (Regulation 2023/956) | Carbon pricing on imports of industrial goods | Definitive from Jan 2026 |
| Energy Efficiency Directive (Directive 2023/1791) | Mandatory energy audits/management systems | Transposition by Oct 2025 |
| Renewable Energy Directive III (Directive 2023/2413) | 42.5% renewables by 2030, industrial targets | In force |
| Effort Sharing Regulation (2023/857) | National targets for non-ETS sectors | In force |
| CO2 emission standards vehicles (2023/851) | 55% reduction by 2030, 100% by 2035 for new cars | In force |
| Alternative Fuels Infrastructure (2023/1804) | Charging/refueling infrastructure for transport | In force |
| Hydrogen & Gas Package (2024) | Hydrogen market rules, CCS framework | Member State transposition by Aug 2026 |

### 4.2 Energy Efficiency Directive (Directive (EU) 2023/1791) -- Article 11 Obligations

| Annual Energy Consumption | Obligation | Deadline |
|--------------------------|-----------|----------|
| >85 TJ/year | Implement certified energy management system (ISO 50001 or equivalent) | October 2027 |
| 10-85 TJ/year | Conduct independent energy audit every 4 years | First audit by 11 October 2026 |
| <10 TJ/year | No mandatory EED obligation (but may face national requirements) | N/A |

**Action Plan Requirement**: Enterprises must draw up concrete, feasible action plans based on energy audit recommendations. Plans must identify measures to implement each recommendation where technically and economically feasible.

**Energy consumption thresholds in context**:
- 10 TJ/year is approximately 2.78 GWh or the energy use of a medium-sized factory
- 85 TJ/year is approximately 23.6 GWh or the energy use of a large industrial site

### 4.3 Industrial Emissions Directive (IED) Recast -- Directive (EU) 2024/XXX

The IED 2.0 entered into force on 4 August 2024 with Member State transposition deadline of 1 July 2026.

#### Key Changes for Manufacturers

| Change | Detail |
|--------|--------|
| Stricter emission limits | Competent Authorities must set the **strictest achievable** BAT-AELs, not merely comply with BAT-AEL ranges |
| Resource efficiency | Permits must contain **binding quantitative requirements** for materials, water, and energy efficiency |
| Battery manufacturing | Large battery manufacturing (>15,000 t/year capacity) now covered as a new Annex I activity |
| Transformation plans | Installations must prepare **transformation plans** aligned with EU 2050 climate neutrality |
| Public access | Enhanced public access to environmental performance data |
| Penalties | Effective, proportionate, and dissuasive penalties; minimum EUR 3M or 3% of annual turnover |

#### BREF (BAT Reference Documents) Coverage for Manufacturing

| BREF Document | Manufacturing Sectors Covered | Status |
|---------------|------------------------------|--------|
| Iron and Steel Production | Blast furnaces, BOF, EAF, rolling, coating | Published 2012, under review |
| Non-ferrous Metals Industries | Aluminum, copper, zinc, lead, precious metals | Published 2017 |
| Cement, Lime and Magnesium Oxide | Cement clinker, lime, MgO production | Published 2013 |
| Glass Manufacturing | Container, flat, special glass, fibers | Published 2013 |
| Ceramics Manufacturing | Bricks, tiles, refractory, tableware | Published 2007 |
| Pulp and Paper | Chemical/mechanical pulp, paper/board | Published 2015 |
| Large Volume Organic Chemicals | Olefins, aromatics, chlor-alkali | Published 2017 |
| Large Volume Inorganic Chemicals | Ammonia, acids, solid inorganics | Published 2007 |
| Food, Drink, Milk | Dairy, brewing, meat, fish, starch | Published 2019 |
| Textiles | Finishing, dyeing, coating | Published 2003, review planned |
| Refining | Petroleum refining | Published 2015 |
| Surface Treatment Using Solvents | Coating, painting, printing, degreasing | Published 2007 |
| Surface Treatment of Metals/Plastics | Electroplating, anodising, etching | Published 2006 |
| Smitheries and Foundries | Ferrous/non-ferrous casting | Published 2005, review planned |
| Common Waste Water/Gas Treatment | Cross-cutting for chemical sector | Published 2016 |
| Energy Efficiency | Cross-cutting for all installations | Published 2009 |
| Industrial Cooling Systems | Cross-cutting for all | Published 2001, review planned |
| Emissions from Storage | Cross-cutting for bulk materials/liquids | Published 2006 |

### 4.4 Technology Readiness for Decarbonization

| Technology | TRL | Applicable Sectors | Timeline | Cost Impact |
|-----------|-----|-------------------|----------|-------------|
| Electrification of heat (heat pumps, resistance) | 7-9 | Food, textiles, chemicals (<200C) | Ready now | +10-30% operating cost |
| Electrification of high-temp heat (>1000C) | 4-6 | Steel, glass, cement | 2030+ | Requires R&D |
| Green hydrogen (electrolysis) | 6-8 | Steel (DRI-H2), chemicals, refining | 2028-2035 | EUR 3-6/kg H2 target <EUR 2/kg |
| CCS (Carbon Capture and Storage) | 7-8 | Cement, chemicals, steel | 2030+ | EUR 60-120/tCO2 captured |
| CCUS (Carbon Capture, Utilization, Storage) | 5-7 | Cement, chemicals | 2030+ | Varies by utilization pathway |
| Biomass co-firing | 9 | Cement, pulp/paper, power | Ready now | Feedstock availability limited |
| Circular economy/recycling | 8-9 | Steel (scrap EAF), aluminum, plastics | Ready now | Reduces Scope 1 by 50-80% (steel) |
| Process intensification | 6-8 | Chemicals, food processing | 2025-2030 | 20-40% energy reduction |
| Digital twins for energy optimization | 7-9 | All manufacturing | Ready now | 5-15% energy reduction |

### 4.5 EU Hydrogen Policy for Industry

Under the Hydrogen and Decarbonised Gas Market Package (adopted August 2024, transposition by August 2026):

- **50% of hydrogen used in industry** must come from renewable sources by 2030 (RED III target)
- Renewable hydrogen must achieve **70% GHG reduction** vs fossil fuel comparator (max 28.2 gCO2e/MJ)
- Industrial renewable share must increase by at least **1.1 percentage points annually**
- Investment requirements: EUR 24-42B for electrolysers, EUR 65B for hydrogen distribution networks by 2030

### 4.6 Regulatory References

- Directive (EU) 2023/1791 (Energy Efficiency Directive recast), Article 11
- Directive (EU) 2023/959 (EU ETS revision)
- Directive (EU) 2023/2413 (Renewable Energy Directive III), Article 22a
- IED 2.0 (Directive published August 2024), transposition by 1 July 2026
- BREF documents: available at https://eippcb.jrc.ec.europa.eu/reference
- Hydrogen and Decarbonised Gas Market Package (2024), transposition by August 2026
- EU Industrial Strategy (updated March 2020, Clean Industry Deal 2025)

---

## 5. Product Carbon Footprint (PCF)

### 5.1 ISO 14067:2018 -- Carbon Footprint of Products

ISO 14067 provides principles, requirements, and guidelines for quantification and reporting of the carbon footprint of a product (CFP). It builds on ISO 14040/14044 (LCA standards).

#### Key Methodological Requirements

| Element | ISO 14067 Requirement |
|---------|----------------------|
| Scope | Single impact category: climate change only |
| System boundary | Cradle-to-grave (full CFP) or cradle-to-gate (partial CFP) |
| Functional unit | Quantified performance of a product system |
| Data sources | Primary data preferred; secondary data with quality assessment |
| Allocation | Avoid allocation where possible; otherwise physical then economic |
| Biogenic carbon | Tracked and reported separately from fossil carbon |
| GWP | 100-year time horizon per IPCC; CO2e basis |
| Cut-off criteria | Maximum 5% of total mass/energy/environmental significance |
| Uncertainty | Quantitative uncertainty analysis required |
| Critical review | Required for public assertions/comparative assertions |

#### PCF Life Cycle Stages

```
Raw Material  ->  Manufacturing  ->  Distribution  ->  Use Phase  ->  End-of-Life
Extraction        & Processing       & Transport       & Maintenance    Treatment
(Upstream)        (Core)             (Downstream)      (Downstream)    (Downstream)
```

### 5.2 EU Ecodesign for Sustainable Products Regulation (ESPR) -- Regulation (EU) 2024/1781

The ESPR entered into force on 18 July 2024 and establishes a framework for setting ecodesign requirements for nearly all physical goods placed on the EU market.

#### Key Requirements for Manufacturers

| Requirement | Detail | Timeline |
|------------|--------|----------|
| Product performance requirements | Durability, reparability, recyclability, energy/resource efficiency | Delegated acts per product group (rolling 2025-2030) |
| Information requirements | Material composition, carbon footprint, recycled content | Via DPP |
| Digital Product Passport (DPP) | Machine-readable product data record | First products: Feb 2027 |
| Substances of concern | SVHC presence and concentration declaration | Per delegated act |
| Unsold goods destruction ban | Prohibition on destroying unsold consumer products | Textiles/footwear: specific dates |

#### Product Groups with Priority Delegated Acts

| Product Group | Expected Delegated Act | DPP Date |
|---------------|----------------------|----------|
| Batteries (EV, industrial) | Battery Regulation 2023/1542 already in force | Feb 2027 |
| Textiles/apparel | 2025-2026 | Mid-2027 |
| Iron and steel | 2026-2027 | 2028 |
| Aluminum | 2026-2027 | 2028 |
| Electronics/ICT | 2026-2027 | 2028-2029 |
| Furniture | 2027 | 2029 |
| Detergents | 2025-2026 | 2027-2028 |
| Paints and coatings | 2027 | 2029 |
| Plastics | 2027-2028 | 2029-2030 |

### 5.3 Digital Product Passport (DPP) Requirements

DPPs under ESPR require manufacturers to provide standardized, machine-readable data for each product model via QR code, NFC chip, or RFID tag, linked to a cloud-hosted registry.

#### Mandatory DPP Data Categories (100+ attributes)

| Data Category | Required Information |
|--------------|---------------------|
| Identification | Manufacturer ID, product model, batch/serial number, GTIN |
| Manufacturer/importer | Name, address, EORI number, facility location |
| Material composition | Bill of materials, geographic origin (esp. conflict minerals) |
| Environmental footprint | Carbon footprint (lifecycle stages), Product Environmental Footprint (PEF) |
| Recycled content | Percentage of recycled material by type (pre/post-consumer) |
| Durability | Expected lifetime, warranty terms, repairability score |
| Recyclability | Disassembly instructions, material separation guidance |
| Compliance | CE marking, REACH/RoHS/CLP compliance status |
| Substances of concern | SVHC presence >0.1% w/w (per REACH Art. 33) |
| Performance | Energy efficiency class, state-of-health metrics |

#### DPP Technical Requirements

- Unique identifier per product: ISO/IEC 15459 for global traceability
- Data carrier: QR code (ISO/IEC 18004), or NFC (ISO/IEC 14443), or RFID
- Data format: Standardized, machine-readable (JSON-LD with schema.org or EPCIS)
- Data hosting: Cloud-based, accessible for product lifetime + recycling period
- Access control: Tiered access (public, authorized economic operators, market surveillance, customs)
- Interoperability: Must connect to EU Digital Product Passport Registry

### 5.4 EU Battery Regulation (Regulation (EU) 2023/1542) -- Specific PCF Requirements

| Obligation | Battery Types | Deadline |
|-----------|--------------|----------|
| Carbon footprint declaration | EV batteries | 18 February 2025 |
| Carbon footprint declaration | Industrial batteries >2 kWh | 18 August 2026 |
| Carbon footprint declaration | LMT batteries | 18 February 2028 |
| Carbon footprint performance class | EV, industrial >2 kWh | 18 August 2026 |
| Maximum carbon footprint threshold | EV, industrial >2 kWh | 18 February 2028 |
| Battery passport (DPP) | EV + industrial >2 kWh + LMT | 18 February 2027 |
| QR code with passport link | All covered batteries | 18 February 2027 |
| Recycled content declaration | EV + industrial + SLI | 18 August 2028 |
| Minimum recycled content targets | Cobalt 16%, Lead 85%, Lithium 6%, Nickel 6% | 18 August 2031 |
| Higher recycled content targets | Cobalt 26%, Lead 85%, Lithium 12%, Nickel 15% | 18 August 2036 |

#### Battery Carbon Footprint Calculation Methodology

Per Commission Delegated Regulation (EU) 2023/1791 (implementing Art. 7(2)):
- Scope: Cradle-to-gate (raw material extraction through battery manufacturing, including mining, refining, cell manufacturing, module/pack assembly)
- Functional unit: 1 kWh of battery capacity at battery system level
- System boundary: Raw material acquisition, preprocessing, active material production, cell manufacturing, battery assembly
- Data requirements: Primary data for core processes; secondary data from recognized databases (ecoinvent, GaBi)
- Reporting unit: kgCO2e/kWh

### 5.5 GHG Protocol Product Life Cycle Standard

The GHG Protocol Product Life Cycle Accounting and Reporting Standard (2011) provides complementary guidance:

- Covers all six Kyoto gases + NF3
- Requires cradle-to-grave or cradle-to-gate boundary definition
- Attributional LCA approach (not consequential)
- Data quality indicators on 5-point scale for: technological, temporal, geographical, completeness, reliability
- Requires uncertainty assessment and sensitivity analysis

### 5.6 Regulatory References

- ISO 14067:2018 (Carbon footprint of products)
- ISO 14040:2006 / ISO 14044:2006 (LCA framework and requirements)
- Regulation (EU) 2024/1781 (ESPR), entered into force 18 July 2024
- Regulation (EU) 2023/1542 (Battery Regulation)
- Commission Delegated Regulation (EU) 2023/1791 (battery carbon footprint calculation rules)
- GHG Protocol Product Life Cycle Accounting and Reporting Standard (2011)
- PEF/OEF methods: Commission Recommendation 2013/179/EU (Product and Organisation Environmental Footprint)

---

## 6. Circular Economy and Waste

### 6.1 EU Circular Economy Action Plan (CEAP)

The CEAP (adopted March 2020, updated) drives multiple legislative instruments affecting manufacturers:

| Instrument | Manufacturing Impact | Status |
|-----------|---------------------|--------|
| ESPR | Ecodesign requirements for product durability, recyclability | In force (July 2024) |
| Waste Framework Directive revision | New EPR obligations, food waste targets | In force (October 2025) |
| Packaging and Packaging Waste Regulation | Recycled content, reuse targets, recyclability | Adopted 2024 |
| Right to Repair Directive | Repair obligations beyond warranty | Adopted 2024 |
| Green Claims Directive | Substantiation of environmental claims | Withdrawn proposal (June 2025) -- but anti-greenwashing rules remain via Empowering Consumers Directive 2024/825 |
| Circular Economy Act | Comprehensive circular economy legislation | Expected end of 2026 |
| Critical Raw Materials Regulation | Recycling targets, strategic stockpiling | In force (May 2024) |

### 6.2 Waste Framework Directive (Directive 2008/98/EC as amended)

Revised Waste Framework Directive entered into force 16 October 2025.

#### Manufacturing Obligations

| Obligation | Detail | Deadline |
|-----------|--------|----------|
| Waste hierarchy compliance | Prevention > Reuse > Recycling > Recovery > Disposal | Ongoing |
| Waste classification | Classify waste per European Waste Catalogue (EWC) codes | Ongoing |
| Record keeping | Document waste generation, treatment, disposal | Ongoing |
| Hazardous waste management | Stricter controls on hazardous waste streams | Ongoing |
| EPR for textiles | Producers pay per-product fees covering collection, sorting, reuse, recycling | Transposition: 20 months, EPR establishment: 30 months from Oct 2025 |
| Food waste reduction | 10% reduction in processing/manufacturing by 2030 | 2030 |
| Separate collection | Mandatory separate collection of hazardous household waste, textile waste | Per Member State schedule |

#### EPR Fee Structure (Textiles)

EPR fees will be **eco-modulated** based on sustainability criteria from ESPR:
- Higher fees for products with lower durability
- Higher fees for products with lower recyclability
- Higher fees for products containing substances of concern
- Lower fees for products with higher recycled content

### 6.3 ESRS E5 -- Resource Use and Circular Economy

Key disclosure requirements under ESRS E5:

| Disclosure | Data Points |
|-----------|------------|
| E5-1 | Policies related to resource use and circular economy |
| E5-2 | Actions and resources related to resource use and circular economy |
| E5-3 | Targets related to resource use and circular economy |
| E5-4 | Resource inflows: total weight of products/materials used, % from biological/non-biological sources, % recycled/reused content |
| E5-5 | Resource outflows: total waste by type and treatment, % hazardous, recycling rate |
| E5-6 | Potential financial effects from resource use and circular economy (transition risks, physical risks) |

#### Manufacturing-Relevant E5 Metrics

| Metric | Unit | Calculation |
|--------|------|-------------|
| Material circularity rate | % | (Recycled input + Reused input) / Total material input |
| Waste intensity | kg/unit or kg/EUR revenue | Total waste generated / Production volume or Revenue |
| Recycling rate | % | Waste sent for recycling / Total waste generated |
| Landfill diversion rate | % | (Total waste - Waste to landfill) / Total waste |
| Hazardous waste ratio | % | Hazardous waste / Total waste |
| Product recyclability score | 0-100 | Weighted assessment of material separability and recyclability |

### 6.4 Critical Raw Materials Regulation (Regulation (EU) 2024/1252)

Affects manufacturers using strategic/critical raw materials:

| Obligation | Detail |
|-----------|--------|
| Recycling targets | Minimum recycling capacity: 25% of annual consumption for strategic raw materials by 2030 |
| Supply chain mapping | Audit of supply chain dependencies for strategic raw materials |
| Substitution assessment | Assess alternatives to critical raw materials in products |
| Circular design | Design products for recoverability of critical raw materials |
| Reporting | Report recycled content of critical raw materials in products |

### 6.5 Regulatory References

- Directive 2008/98/EC (Waste Framework Directive) as amended (revision in force Oct 2025)
- Regulation (EU) 2024/1781 (ESPR)
- Regulation (EU) 2024/1252 (Critical Raw Materials Regulation), in force May 2024
- Regulation (EU) 2024/XXX (Packaging and Packaging Waste Regulation)
- Directive (EU) 2024/XXX (Right to Repair Directive)
- Directive (EU) 2024/825 (Empowering Consumers for the Green Transition)
- ESRS E5 (Resource Use and Circular Economy), per Delegated Regulation (EU) 2023/2772

---

## 7. Water and Pollution for Manufacturing

### 7.1 ESRS E2 -- Pollution

ESRS E2 requires disclosure on pollution of air, water, and soil, substances of concern, and substances of very high concern.

#### Key Disclosure Requirements

| Disclosure | Content | Data Required |
|-----------|---------|---------------|
| E2-1 | Policies related to pollution | Description of pollution prevention/control policies |
| E2-2 | Actions and resources | Pollution reduction actions, expenditures, timelines |
| E2-3 | Targets related to pollution | Quantitative targets for pollutant reduction |
| E2-4 | Pollution of air, water, soil | **Quantitative**: tonnes of each pollutant emitted to air, water, soil |
| E2-5 | Substances of concern (SOC) and SVHC | Names, quantities, hazard classes per CLP Regulation |
| E2-6 | Potential financial effects | Fines, remediation costs, transition costs |

#### E2-4 Air Pollutant Categories (Manufacturing-Relevant)

| Pollutant Category | Specific Substances | Typical Manufacturing Sources |
|-------------------|--------------------|-----------------------------|
| NOx | NO, NO2 | Combustion, acid production |
| SOx | SO2, SO3 | Combustion of sulfur-containing fuels, metal smelting |
| Particulate matter | PM2.5, PM10, TSP | Grinding, crushing, combustion, material handling |
| VOCs | Various organic compounds | Painting, coating, printing, solvent use |
| Heavy metals | Pb, Hg, Cd, Cr, As, Ni | Metal processing, surface treatment |
| CO | Carbon monoxide | Incomplete combustion |
| Dioxins/furans | PCDD/PCDF | Combustion, metal smelting, chemical processes |
| Microplastics | Primary (manufactured) and secondary (degradation) | Plastic manufacturing, textile production |

#### E2-5 Substances of Concern -- REACH/CLP Integration

| Requirement | Detail |
|------------|--------|
| SVHC identification | Disclose names of all SVHC present >0.1% w/w in articles (per REACH Art. 33) |
| Grouping | Group SVHC data by CLP hazard class, avoid double counting |
| Categories | Distinguish: manufactured, used as input, imported, sold in products |
| Quantification | Report total tonnes of SOC/SVHC by category |
| SCIP database | Separate obligation: notify ECHA SCIP database for articles containing SVHC >0.1% w/w |

### 7.2 ESRS E3 -- Water and Marine Resources

| Disclosure | Content |
|-----------|---------|
| E3-1 | Policies related to water and marine resources |
| E3-2 | Actions and resources (water efficiency, treatment improvements) |
| E3-3 | Targets related to water and marine resources |
| E3-4 | Water consumption: total withdrawal by source, consumption, discharge by destination, in areas of water stress |
| E3-5 | Potential financial effects from water risks |

#### E3-4 Water Data Points for Manufacturers

| Data Point | Unit | Detail |
|-----------|------|--------|
| Total water withdrawal | m3 | By source: surface water, groundwater, municipal, third-party, rainwater |
| Total water discharge | m3 | By destination: surface water, groundwater, ocean, third-party/municipal |
| Total water consumption | m3 | Withdrawal - Discharge |
| Water withdrawal in water-stressed areas | m3 | Locations identified via WRI Aqueduct or equivalent |
| Water discharge quality | mg/L | COD, BOD, TSS, heavy metals, temperature |
| Water recycling/reuse rate | % | Recycled water / Total water use |

### 7.3 Water Stress Assessment Methodologies

#### WRI Aqueduct 4.0

| Indicator | Description | Stress Threshold |
|-----------|------------|-----------------|
| Baseline water stress | Ratio of total water withdrawals to available renewable supply | High: 40-80%, Extremely high: >80% |
| Baseline water depletion | Ratio of total water consumption to available renewable supply | High: 50-75%, Extremely high: >75% |
| Interannual variability | Variation in water supply between years | Higher = more risk |
| Seasonal variability | Variation in water supply within year | Higher = more risk |
| Groundwater table decline | Rate of groundwater level decrease | Site-specific |
| Riverine flood risk | Probability of riverine flooding | Site-specific |
| Coastal flood risk | Probability of coastal flooding | Site-specific |
| Drought risk | Probability of drought occurrence | Site-specific |
| Regulatory and reputational risk | Composite of water-related regulatory and social risks | Site-specific |

#### CDP Water Security Questionnaire (W-series)

Key sections for manufacturers:
- W1: Current state (water accounting, water-related risks/opportunities)
- W2: Business impacts of water risks
- W3: Procedures for identifying/assessing water-related risks
- W4: Risks and opportunities (specific risk events, financial impact)
- W5: Facility-level water accounting (water-stressed facilities)
- W7: Business strategy alignment
- W8: Targets (water consumption, efficiency, quality)

### 7.4 REACH Regulation (Regulation (EC) No 1907/2006)

| Obligation | Threshold | Manufacturer Duty |
|-----------|----------|-------------------|
| Registration | >1 tonne/year per substance | Register with ECHA; chemical safety assessment |
| Evaluation | Triggered by ECHA | Respond to information requests |
| Authorization | SVHC on Annex XIV | Apply for authorization to use/place on market |
| Restriction | Annex XVII | Comply with conditions for restricted substances |
| SVHC notification | >0.1% w/w in articles, >1 tonne/year | Notify ECHA within 6 months of inclusion |
| SVHC communication | >0.1% w/w in articles | Inform downstream users with safety information |
| SCIP notification | >0.1% w/w in articles | Submit to ECHA SCIP database |

### 7.5 CLP Regulation (Regulation (EC) No 1272/2008)

Classifies hazard categories relevant to E2 reporting:
- Physical hazards (flammable, explosive, oxidising)
- Health hazards (acute toxicity, carcinogenicity, mutagenicity, reproductive toxicity, STOT)
- Environmental hazards (aquatic acute/chronic toxicity, hazardous to ozone layer)

### 7.6 Regulatory References

- ESRS E2 (Pollution), per Delegated Regulation (EU) 2023/2772
- ESRS E3 (Water and Marine Resources), per Delegated Regulation (EU) 2023/2772
- Regulation (EC) No 1907/2006 (REACH), as amended
- Regulation (EC) No 1272/2008 (CLP), as amended
- Directive 2010/75/EU (Industrial Emissions Directive) / IED 2.0
- WRI Aqueduct 4.0 methodology (2023)
- CDP Water Security questionnaire guidance (2025)
- Industrial Emission Portal (IEP) Regulation -- new under IED 2.0

---

## 8. Biodiversity and Land Use for Manufacturing

### 8.1 ESRS E4 -- Biodiversity and Ecosystems

| Disclosure | Content |
|-----------|---------|
| E4-1 | Transition plan and consideration of biodiversity in strategy |
| E4-2 | Policies related to biodiversity and ecosystems |
| E4-3 | Actions and resources (biodiversity conservation, restoration) |
| E4-4 | Targets related to biodiversity |
| E4-5 | Impact metrics on biodiversity |
| E4-6 | Potential financial effects |

#### E4-5 Key Metrics for Manufacturers

| Metric | Description | Manufacturing Relevance |
|--------|------------|------------------------|
| Land use change | Area of land converted or affected | Factory siting, facility expansion |
| Sites in/near biodiversity-sensitive areas | Number of operational sites near protected areas, KBAs | Site selection, environmental permits |
| Impact on species | Threatened species potentially affected | Supply chain raw materials |
| Ecosystem condition | State of ecosystems in areas of operation | Water quality, soil contamination |
| Mitigation hierarchy | Avoid, minimize, restore, offset | All operations |

### 8.2 TNFD Framework -- LEAP Approach for Manufacturing

The Taskforce on Nature-related Financial Disclosures (TNFD) LEAP approach provides a structured process:

| Phase | Manufacturing Application |
|-------|--------------------------|
| **L** -- Locate | Map all manufacturing sites, identify interface with nature (water sources, land, biodiversity-sensitive areas). Use tools: IBAT, ENCORE, WRI Aqueduct |
| **E** -- Evaluate | Assess dependencies (water supply, raw materials, ecosystem services) and impacts (pollution, land use, water extraction) at each site |
| **A** -- Assess | Quantify nature-related risks (physical: resource scarcity; transition: regulation; systemic: ecosystem collapse) and opportunities (nature-based solutions, circular models) |
| **P** -- Prepare | Develop response strategies, set targets, prepare TNFD-aligned disclosures |

#### Manufacturing Site-Level Assessment

Manufacturers should:
1. Generate **site-level heatmaps** ranking all facilities by impact/dependency scores
2. Prioritize sites with highest composite nature risk
3. Conduct detailed assessments for high-risk sites
4. Set site-specific biodiversity targets where material

### 8.3 EUDR Overlap -- Supply Chain Deforestation Risk

For manufacturers using commodities covered by EUDR (Regulation (EU) 2023/1115):

| Commodity | Manufacturing Sub-sectors Affected |
|-----------|-----------------------------------|
| Palm oil | Food/beverage (C10-C11), cosmetics, chemicals |
| Soy | Food/beverage (C10-C11), animal feed |
| Cocoa | Food/beverage (C10) |
| Coffee | Food/beverage (C10) |
| Rubber | Automotive (C29), rubber products (C22) |
| Cattle (leather) | Leather/footwear (C15), automotive interiors |
| Wood | Furniture (C31), paper/packaging (C17) |

Manufacturers must ensure these commodities in their supply chain are **deforestation-free** (no deforestation after 31 December 2020) and **legally produced**.

### 8.4 Regulatory References

- ESRS E4 (Biodiversity and Ecosystems), per Delegated Regulation (EU) 2023/2772
- TNFD Recommendations v1.0 (September 2023)
- TNFD LEAP Approach Guidance (2023)
- Regulation (EU) 2023/1115 (EUDR)
- Kunming-Montreal Global Biodiversity Framework (GBF), December 2022
- EU Biodiversity Strategy for 2030 (COM(2020) 380)
- Nature Restoration Regulation (Regulation (EU) 2024/1991)

---

## 9. Social Standards in Manufacturing

### 9.1 ESRS S1 -- Own Workforce

Manufacturing is a sector with significant occupational health and safety (OHS) risks, making ESRS S1 a priority material topic.

#### Key Disclosure Requirements

| Disclosure | Content | Manufacturing Focus |
|-----------|---------|---------------------|
| S1-1 | Policies related to own workforce | OHS policy, collective bargaining, living wage, D&I |
| S1-2 | Processes for engaging with workers | Works councils, safety committees, union recognition |
| S1-3 | Remediation processes | Grievance mechanisms, incident response |
| S1-4 | Actions on material impacts | Specific programs for worker safety, training, well-being |
| S1-5 | Targets | Quantitative OHS, diversity, training targets |
| S1-6 | Characteristics of employees | Headcount by gender, contract type, region |
| S1-7 | Characteristics of non-employees in workforce | Agency workers, contractors by type |
| S1-8 | Collective bargaining coverage | % of employees covered by CBAs; in EEA, social dialogue coverage |
| S1-9 | Diversity metrics | Gender pay gap, age distribution, disability |
| S1-10 | Adequate wages | Whether all employees receive at least applicable adequate wage (living wage benchmark) |
| S1-11 | Social protection | Coverage of social protection for workers |
| S1-12 | Persons with disabilities | Percentage, accommodations |
| S1-13 | Training and skills | Average training hours, training expenditure |
| S1-14 | Health and safety | **Work-related accidents, injuries, fatalities, ill health. LTIR, TRIR, fatality rate** |
| S1-15 | Work-life balance | Family leave, flexible working |
| S1-16 | Remuneration metrics | Pay ratio, variable pay |
| S1-17 | Incidents, complaints, impacts | Number and type of incidents, discrimination, harassment |

#### S1-14 Health and Safety Metrics (Critical for Manufacturing)

| Metric | Calculation | Unit |
|--------|------------|------|
| Fatal accident rate | (Number of fatalities / Hours worked) x 1,000,000 | Per million hours |
| LTIR (Lost Time Injury Rate) | (Number of lost-time injuries / Hours worked) x 1,000,000 | Per million hours |
| TRIR (Total Recordable Incident Rate) | (Recordable injuries / Hours worked) x 200,000 | Per 200,000 hours |
| LTIFR (Lost Time Injury Frequency Rate) | (Number of lost-time injuries / Hours worked) x 1,000,000 | Per million hours |
| Lost days rate | (Number of lost workdays / Hours scheduled) x 200,000 | Per 200,000 hours |
| Occupational disease rate | (New cases of occupational disease / Workers) x 10,000 | Per 10,000 workers |
| Near miss frequency rate | (Near misses reported / Hours worked) x 1,000,000 | Per million hours |
| Process safety event rate | (Process safety events / Hours worked) x 200,000 | Per 200,000 hours |

### 9.2 ESRS S2 -- Workers in the Value Chain

| Disclosure | Content | Manufacturing Application |
|-----------|---------|--------------------------|
| S2-1 | Policies on value chain workers | Supplier codes of conduct, audit policies |
| S2-2 | Engagement with value chain workers | Supplier engagement programs, worker voice |
| S2-3 | Remediation for value chain workers | Operational grievance mechanisms for supply chain |
| S2-4 | Actions on material impacts | Supplier capacity building, living wage programs |
| S2-5 | Targets | Supplier audit coverage, certification targets |

**Manufacturing-specific value chain risks**:
- Forced labor in raw material extraction (minerals, cotton, palm oil)
- Child labor in agricultural supply chains (cocoa, cobalt)
- Unsafe working conditions in tier 2-3 suppliers (garments, electronics)
- Below-living-wage in developing country manufacturing suppliers
- Freedom of association restrictions in certain jurisdictions
- Excessive working hours in peak production periods

### 9.3 CSDDD -- Corporate Sustainability Due Diligence Directive

**Directive (EU) 2024/1760**, entered into force 25 July 2024.

| Element | Detail |
|---------|--------|
| Scope | EU companies: >5,000 employees AND >EUR 1.5B global turnover; Non-EU: >EUR 1.5B EU turnover |
| Application date | Single deadline: 26 July 2029 (Omnibus postponed from phased approach) |
| Transposition | Member States by 26 July 2026 |
| Due diligence scope | Own operations + subsidiaries + "chain of activities" (upstream suppliers, some downstream) |
| Human rights | ILO core conventions, ICCPR, ICESCR, UDHR, children's rights, indigenous peoples' rights |
| Environmental | Paris Agreement alignment, biodiversity, pollution, waste |
| Transition plan | Climate transition plan aligned with 1.5C pathway |
| Penalties | Up to 5% of worldwide net annual turnover |
| Civil liability | Victims can claim compensation for damage from failure to conduct due diligence |
| Supervisory authority | Each Member State designates one |

#### ILO Core Conventions (Relevant to Manufacturing Supply Chains)

| Convention | Topic | Manufacturing Relevance |
|-----------|-------|------------------------|
| C029 & C105 | Forced Labour | Mining, agriculture, garment supply chains |
| C087 & C098 | Freedom of Association & Collective Bargaining | All manufacturing, esp. in countries with restrictions |
| C100 & C111 | Equal Remuneration & Non-Discrimination | Gender pay equity in manufacturing workforce |
| C138 & C182 | Minimum Age & Worst Forms of Child Labour | Agricultural inputs, artisanal mining (cobalt) |

### 9.4 Regulatory References

- ESRS S1 (Own Workforce), per Delegated Regulation (EU) 2023/2772
- ESRS S2 (Workers in the Value Chain), per Delegated Regulation (EU) 2023/2772
- Directive (EU) 2024/1760 (CSDDD), entered into force 25 July 2024
- Directive (EU) 2026/470 (Omnibus I) -- postponed CSDDD phased approach to single date
- ILO Declaration on Fundamental Principles and Rights at Work (1998, amended 2022)
- UN Guiding Principles on Business and Human Rights (2011)
- OECD Due Diligence Guidance for Responsible Business Conduct (2018)
- EU Pay Transparency Directive (2023/970), transposition by 7 June 2026

---

## 10. Manufacturing-Specific KPIs

### 10.1 Energy Intensity Metrics

| KPI | Formula | Unit | Sector Benchmark |
|-----|---------|------|------------------|
| Energy intensity (physical) | Total energy consumption / Production volume | MJ/tonne or MJ/unit | Varies by product |
| Energy intensity (economic) | Total energy consumption / Revenue | MJ/EUR or GJ/EUR million | Manufacturing avg: 3-8 GJ/EUR M |
| Specific energy consumption (SEC) | Energy for process / Process output | kWh/tonne or MJ/tonne | BAT-AEL defined per BREF |
| Thermal energy intensity | Thermal energy / Production | MJ(th)/tonne | Critical for cement, glass, ceramics |
| Electrical energy intensity | Electrical energy / Production | kWh/tonne | Critical for aluminum, electronics |
| Renewable energy share | Renewable energy / Total energy | % | EU target: 42.5% by 2030 |

#### Sector-Specific Energy Benchmarks

| Sub-sector | BAT-AEL Energy Intensity | Source |
|-----------|--------------------------|--------|
| Cement clinker | 3,000-3,400 MJ/t clinker (thermal) | Cement BREF |
| Float glass | 5.0-7.0 GJ/t melted glass | Glass BREF |
| Crude steel (BF-BOF) | 17-23 GJ/t crude steel | Iron & Steel BREF |
| Crude steel (EAF) | 1.5-3.5 GJ/t crude steel | Iron & Steel BREF |
| Primary aluminum | 13,500-14,500 kWh/t Al (electrolysis) | Non-ferrous BREF |
| Newsprint paper | 3.5-7.0 GJ/t paper | Pulp & Paper BREF |
| Ammonia | 28-33 GJ/t NH3 | LVIC BREF |
| Ethylene (naphtha cracking) | 14-22 GJ/t ethylene | LVOC BREF |
| Beer | 100-200 MJ/hL beer | Food BREF |
| Milk processing | 0.3-0.8 GJ/t milk processed | Food BREF |

### 10.2 Emission Intensity Metrics

| KPI | Formula | Unit |
|-----|---------|------|
| GHG intensity (physical) | Total GHG emissions / Production volume | tCO2e/tonne or kgCO2e/unit |
| GHG intensity (economic) | Total GHG emissions / Revenue | tCO2e/EUR million |
| Scope 1 intensity | Scope 1 emissions / Production | tCO2e/tonne |
| Scope 1+2 intensity | (Scope 1 + Scope 2) / Production | tCO2e/tonne |
| Full value chain intensity | (Scope 1+2+3) / Production | tCO2e/tonne |
| Process emission ratio | Process emissions / Total Scope 1 | % |
| Carbon productivity | Revenue / Total GHG emissions | EUR/tCO2e |

#### EU ETS Product Benchmarks (Free Allocation Reference Values)

| Product | Benchmark (tCO2/t product) | Carbon Leakage Status |
|---------|---------------------------|----------------------|
| Cement clinker | 0.766 | Yes |
| Hot metal (pig iron) | 1.328 | Yes |
| EAF carbon steel | 0.283 | Yes |
| EAF high alloy steel | 0.352 | Yes |
| Sintered ore | 0.171 | Yes |
| Float glass | 0.453 | Yes |
| Container glass | 0.382 | Yes |
| Primary aluminum (electrolysis) | 1.514 | Yes |
| Alumina refining | 0.252 | Yes |
| Ammonia | 1.619 | Yes |
| Nitric acid | 0.302 | Yes |
| Hydrogen | 8.85 | Yes |
| Lime | 0.954 | Yes |
| Newsprint | 0.298 | No |

### 10.3 Water Intensity Metrics

| KPI | Formula | Unit |
|-----|---------|------|
| Water intensity (physical) | Total water consumption / Production volume | m3/tonne or L/unit |
| Water intensity (economic) | Total water consumption / Revenue | m3/EUR million |
| Wastewater intensity | Total wastewater discharge / Production | m3/tonne |
| Water recycling rate | Recycled water volume / Total water use | % |
| Water stress exposure | Revenue from water-stressed operations / Total revenue | % |

#### Sector Water Intensity Benchmarks

| Sub-sector | Water Intensity | Source |
|-----------|----------------|--------|
| Steel (BF-BOF) | 10-30 m3/t crude steel | World Steel Association |
| Steel (EAF) | 0.6-5.3 m3/t crude steel | Literature range |
| Aluminum | 5-10 m3/t aluminum | IAI |
| Paper & board | 10-50 m3/t paper | Pulp & Paper BREF |
| Textiles (finishing) | 100-300 L/kg fabric | Textiles BREF |
| Food & beverage | 1-10 m3/t product | Variable by product |
| Beer | 3-6 hL water/hL beer | Food BREF |
| Dairy | 1-3 m3/t milk processed | Food BREF |
| Semiconductors | 20-40 m3/wafer start (300mm) | Industry reports |
| Automotive | 3-7 m3/vehicle | Manufacturer reports |

### 10.4 Waste Intensity Metrics

| KPI | Formula | Unit |
|-----|---------|------|
| Waste intensity (physical) | Total waste / Production volume | kg/tonne or kg/unit |
| Waste intensity (economic) | Total waste / Revenue | kg/EUR million |
| Hazardous waste ratio | Hazardous waste / Total waste | % |
| Waste diversion rate | (Total waste - Landfill) / Total waste | % |
| Material yield | Finished product mass / Raw material input mass | % |
| Scrap rate | Scrap generated / Total production | % |

### 10.5 OEE with Sustainability Overlay

#### Traditional OEE

```
OEE = Availability x Performance x Quality

Availability = (Planned Production Time - Stop Time) / Planned Production Time
Performance = (Ideal Cycle Time x Total Count) / Run Time
Quality = Good Count / Total Count
```

**World-class benchmark**: OEE >= 85% (Availability 90%, Performance 95%, Quality 99.9%)

#### Overall Environmental Equipment Effectiveness (OEEE)

Extends OEE with environmental dimensions:

```
OEEE = OEE x Environmental Efficiency

Environmental Efficiency = f(Energy efficiency, Material efficiency, Emission efficiency, Waste efficiency)

Where:
- Energy efficiency = Theoretical minimum energy / Actual energy consumption
- Material efficiency = Product output mass / Material input mass
- Emission efficiency = BAT emission level / Actual emission level (capped at 1.0)
- Waste efficiency = 1 - (Waste generated / Theoretical zero-waste benchmark)
```

**Sustainable OEE target**: OEEE >= 75% (per academic research on apparel/manufacturing)

### 10.6 Sector-Specific KPI Dashboards

#### Automotive (NACE C29)

| KPI | Target Range | Unit |
|-----|-------------|------|
| Energy intensity | 4-8 MWh/vehicle | MWh/vehicle |
| GHG intensity (Scope 1+2) | 0.5-1.5 tCO2e/vehicle | tCO2e/vehicle |
| Water intensity | 3-7 m3/vehicle | m3/vehicle |
| Waste-to-landfill | <5% | % of total waste |
| VOC emissions | 20-45 g/m2 body surface | g/m2 |
| Renewable electricity | >50% | % |

#### Chemicals (NACE C20)

| KPI | Target Range | Unit |
|-----|-------------|------|
| Energy intensity | 5-30 GJ/t product | GJ/t (varies by chemical) |
| GHG intensity | 0.5-3.0 tCO2e/t product | tCO2e/t |
| Water intensity | 5-50 m3/t product | m3/t |
| Hazardous waste ratio | <10% | % of total waste |
| Process safety event rate | <0.1 | Per 200,000 hours |
| SVHC substitution progress | Annual reduction targets | % reduction |

#### Food & Beverage (NACE C10-C11)

| KPI | Target Range | Unit |
|-----|-------------|------|
| Energy intensity | 0.5-3.0 GJ/t product | GJ/t |
| GHG intensity (Scope 1+2) | 0.05-0.5 tCO2e/t product | tCO2e/t |
| Water intensity | 1-10 m3/t product | m3/t |
| Food waste in manufacturing | <2% of input | % |
| Packaging recyclability | >80% | % |
| Sustainable sourcing | >50% certified | % |

#### Textiles (NACE C13-C14)

| KPI | Target Range | Unit |
|-----|-------------|------|
| Energy intensity | 15-40 MJ/kg fabric | MJ/kg |
| Water intensity | 100-300 L/kg fabric | L/kg |
| Chemical usage intensity | 0.5-2.0 kg chemicals/kg fabric | kg/kg |
| Wastewater quality (COD) | <200 mg/L discharge | mg/L |
| Recycled fiber content | >20% | % |
| Worker safety (TRIR) | <3.0 | Per 200,000 hours |

#### Electronics (NACE C26-C27)

| KPI | Target Range | Unit |
|-----|-------------|------|
| Energy intensity | 50-200 kWh/unit | kWh/unit (varies by product) |
| GHG intensity (Scope 1+2) | 5-50 kgCO2e/unit | kgCO2e/unit |
| Water intensity (semiconductors) | 20-40 m3/wafer start | m3/300mm wafer |
| F-gas emissions | Annual reduction targets | tCO2e |
| Conflict mineral due diligence | 100% coverage | % of suppliers audited |
| Product energy efficiency (use phase) | Top quartile of class | Energy label class |

---

## 11. BAT and Technology Reference

### 11.1 Best Available Techniques for Key Manufacturing Sectors

#### Steel -- Decarbonization Pathway

| Technology | Current Status | TRL | Emission Reduction | Cost Impact |
|-----------|---------------|-----|-------------------|-------------|
| BF-BOF (conventional) | Dominant (70% global) | 9 | Baseline (~1.85 tCO2/t) | Baseline |
| BF-BOF with CCS | Pilot/demo | 6-7 | 50-60% reduction | +EUR 40-80/t steel |
| EAF with scrap (recycling) | Established (30% global) | 9 | 75% vs BF-BOF (~0.4 tCO2/t) | Similar to BF-BOF |
| DRI-Natural Gas + EAF | Commercial | 9 | 40-50% vs BF-BOF | +EUR 10-30/t |
| DRI-Hydrogen + EAF | Pilot (HYBRIT, H2GreenSteel) | 6-7 | 90-95% vs BF-BOF | +EUR 100-200/t (at current H2 price) |
| Electrolysis of iron ore | Lab/pilot | 3-5 | Near-zero (with green electricity) | Unknown at scale |
| Biomass/biochar in BF | Demo | 5-7 | 20-40% vs conventional BF | +EUR 20-50/t |

#### Cement -- Decarbonization Pathway

| Technology | TRL | Emission Reduction | Notes |
|-----------|-----|-------------------|-------|
| Alternative fuels (waste, biomass) | 9 | 10-30% of thermal emissions | Widely deployed in EU |
| Energy efficiency (waste heat recovery) | 9 | 5-10% | Already standard in modern kilns |
| Clinker substitution (SCMs) | 9 | Up to 30-40% (depends on substitution rate) | GGBS, fly ash, natural pozzolans |
| Oxy-fuel combustion + CCS | 5-6 | 90%+ of CO2 capture | LEILAC, Heidelberg Brevik projects |
| Post-combustion CCS (amine) | 6-7 | 90%+ of CO2 capture | Heidelberg Brevik: first full-scale by 2030 |
| Novel cements (LC3, Celitement) | 5-7 | 30-40% vs OPC | Limited availability of SCMs at scale |
| Carbon curing (CO2 mineralization) | 5-6 | Net CO2 uptake in product | CarbonCure, Solidia |
| Electrification of kiln | 3-4 | Process emissions remain (~60%) | Only addresses thermal emissions |

#### Aluminum -- Decarbonization Pathway

| Technology | TRL | Emission Reduction | Notes |
|-----------|-----|-------------------|-------|
| Renewable electricity for smelting | 9 | Up to 75% of Scope 2 | Already standard in Norway, Iceland |
| Inert anode technology | 5-7 | Eliminates anode CO2 (process emissions) | ELYSIS joint venture (Rio Tinto/Alcoa) |
| Secondary aluminum (recycling) | 9 | 95% vs primary | Limited by scrap availability and contamination |
| Carbon capture on pot gas | 4-5 | 50-80% of process emissions | Complex gas composition |
| Mechanical vapor recompression (alumina) | 7-8 | 30-50% of refining energy | Targeting Bayer process |

### 11.2 Marginal Abatement Cost Curves (MACCs) -- Indicative

| Abatement Measure | Cost (EUR/tCO2e) | Sector |
|-------------------|------------------|--------|
| Energy efficiency improvements | -50 to +20 | Cross-sector |
| Fuel switching (gas to biomass) | 20-80 | Heat-intensive |
| Electrification of low-temp heat (<200C) | 30-100 | Food, chemicals, textiles |
| Renewable electricity procurement | 0-50 | All (depends on PPA) |
| Scrap-based EAF (vs BF-BOF) | -20 to +30 | Steel |
| Clinker substitution | -10 to +30 | Cement |
| CCS (industrial) | 60-120 | Cement, steel, chemicals |
| Green hydrogen for DRI | 100-250 | Steel |
| Inert anodes (aluminum) | 50-150 | Aluminum |
| CCUS with utilization | 80-200 | Cross-sector |

### 11.3 Regulatory References

- BREF documents per sector (JRC EIPPCB)
- EU Innovation Fund projects database
- IEA Energy Technology Perspectives 2023
- EU Clean Industry Deal (2025)
- Regulation (EU) 2024/XXX (Net-Zero Industry Act)
- European Hydrogen Bank (first auctions 2023-2024)

---

## 12. Supply Chain Emissions for Manufacturers

### 12.1 Scope 3 Upstream (Categories 1-8)

#### Category 1 -- Purchased Goods and Services (Dominant for Most Manufacturers)

**Calculation Methods** (per GHG Protocol Scope 3 Technical Guidance):

| Method | Data Quality | Accuracy | Description |
|--------|------------|----------|-------------|
| Supplier-specific | Score 1 (highest) | Best | Supplier provides product-level carbon footprint (PCF) |
| Hybrid | Score 2 | Good | Combine supplier activity data with emission factors |
| Average-data | Score 3 | Moderate | Mass/volume x cradle-to-gate EF from LCA databases |
| Spend-based | Score 4-5 (lowest) | Rough | Spend (EUR) x EEIO emission factor (kgCO2e/EUR) |

**Data Quality Scoring** (PCAF-inspired for supply chain data):

| Score | Data Type | Description |
|-------|----------|-------------|
| 1 | Verified supplier-specific PCF | Third-party verified, product-level carbon footprint from supplier |
| 2 | Unverified supplier-specific | Supplier-provided PCF, not independently verified |
| 3 | Activity-based with average EFs | Physical data (kg, kWh) with industry average emission factors |
| 4 | Spend-based with sector EFs | Financial data with sector-specific environmentally extended IO factors |
| 5 | Spend-based with economy-wide EFs | Financial data with economy-wide average factors |

#### Category 4 -- Upstream Transportation and Distribution

| Transport Mode | Emission Factor Range | Unit |
|---------------|----------------------|------|
| Road (heavy truck, diesel) | 62-150 gCO2e/tkm | gCO2e per tonne-km |
| Rail (electric) | 5-30 gCO2e/tkm | gCO2e per tonne-km |
| Rail (diesel) | 20-40 gCO2e/tkm | gCO2e per tonne-km |
| Maritime (container, avg) | 8-20 gCO2e/tkm | gCO2e per tonne-km |
| Maritime (bulk, avg) | 3-10 gCO2e/tkm | gCO2e per tonne-km |
| Air freight | 500-1,000 gCO2e/tkm | gCO2e per tonne-km |
| Inland waterway | 30-50 gCO2e/tkm | gCO2e per tonne-km |

#### Category 5 -- Waste Generated in Operations

| Waste Treatment Method | Emission Factor Range | Unit |
|-----------------------|----------------------|------|
| Landfill (mixed waste) | 0.4-1.2 tCO2e/t waste | tCO2e per tonne |
| Incineration (without energy recovery) | 0.8-1.5 tCO2e/t waste | tCO2e per tonne |
| Incineration (with energy recovery) | 0.2-0.8 tCO2e/t waste | tCO2e per tonne |
| Recycling (metals) | -1.0 to -4.0 tCO2e/t (avoided) | tCO2e per tonne |
| Recycling (plastics) | -0.5 to -2.0 tCO2e/t (avoided) | tCO2e per tonne |
| Composting (organic) | 0.03-0.1 tCO2e/t | tCO2e per tonne |
| Anaerobic digestion | -0.1 to +0.3 tCO2e/t | tCO2e per tonne |

### 12.2 Scope 3 Downstream (Categories 9-15)

#### Category 11 -- Use of Sold Products (Often Dominant for Manufacturers of Energy-Using Products)

**Calculation for energy-using products**:
```
Cat 11 emissions = SUM over products sold:
  Units sold x Expected product lifetime (years) x Annual energy consumption (kWh) x Grid emission factor (kgCO2e/kWh)
```

**Calculation for fuels/feedstocks**:
```
Cat 11 emissions = SUM over fuels sold:
  Volume sold (GJ) x Emission factor of combustion (tCO2e/GJ)
```

#### Category 12 -- End-of-Life Treatment of Sold Products

**Calculation**:
```
Cat 12 emissions = SUM over products sold:
  Units sold x Product mass (kg) x
  SUM over waste fractions:
    (Fraction to treatment method x Emission factor of treatment method)
```

### 12.3 Supplier Engagement Strategies

| Strategy | Scope | Implementation |
|----------|-------|----------------|
| Supplier questionnaire (CDP Supply Chain) | Tier 1 suppliers | Annual disclosure request via CDP platform |
| Direct data collection (PCF requests) | Top 50-100 suppliers | Bilateral engagement, data sharing agreements |
| Supplier capability building | Strategic suppliers | Training on carbon accounting, joint target setting |
| Procurement criteria (carbon weighting) | All new contracts | Include carbon performance in RFP/RFQ scoring |
| Contractual obligations | Key suppliers | SBTi commitment, annual emission reporting in contracts |
| Industry collaboration | Sector-wide | WBCSD PACT, Catena-X (automotive), TfS (chemicals) |
| Audit and verification | High-risk suppliers | Third-party verification of supplier emission data |

#### Industry Data Exchange Platforms

| Platform | Sector | Scope | Data Standard |
|----------|--------|-------|---------------|
| Catena-X | Automotive | Full value chain PCF exchange | WBCSD Pathfinder Framework |
| Together for Sustainability (TfS) | Chemicals | PCF at product level | TfS PCF Guideline |
| WBCSD PACT | Cross-sector | Interoperable PCF exchange | Pathfinder Framework v2.0 |
| Ecoinvent | Cross-sector | Background LCA data | ecoinvent methodology |
| GaBi/Sphera | Cross-sector | LCA database | GaBi methodology |
| CDP Supply Chain | Cross-sector | Supplier climate disclosure | CDP questionnaire |

### 12.4 PCAF-Style Data Quality Scoring for Manufacturing Supply Chains

Adapting the PCAF framework (originally for financial institutions) to manufacturing supply chain data quality:

| Score | Level | Data Source | Recommended For |
|-------|-------|------------|-----------------|
| 1 | Reported, verified | Supplier-specific, third-party verified PCF or Scope 1+2 data | Strategic suppliers (top 10-20 by emissions) |
| 2 | Reported, unverified | Supplier-provided data, not externally verified | Major suppliers (top 50) |
| 3 | Estimated, activity-based | Physical data (kWh, kg, km) with industry EFs | All direct suppliers with activity data |
| 4 | Estimated, spend-based (sector) | Spend data with sector-level EEIO factors | Suppliers where physical data unavailable |
| 5 | Estimated, spend-based (generic) | Spend data with economy-wide average factors | Tail-end suppliers, initial screening |

**Target trajectory**: Improve average data quality score by 0.5 per year, targeting average score <=2.5 within 3 years.

### 12.5 Regulatory References

- GHG Protocol Scope 3 Standard (2011)
- GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
- GHG Protocol Product Life Cycle Standard (2011)
- PCAF Global GHG Accounting and Reporting Standard (Part A, 3rd edition)
- WBCSD Pathfinder Framework v2.0 (2023)
- CDP Supply Chain Program guidance
- TfS Product Carbon Footprint Guideline v3.0
- ISO 14064-1:2018 (Organization-level GHG quantification)
- ESRS E1 (Climate Change) -- value chain emission disclosure requirements

---

## Appendix A: Consolidated Regulatory Timeline for Manufacturing Pack

| Date | Regulation/Event | Impact |
|------|-----------------|--------|
| **Already in force** | | |
| July 2024 | ESPR entered into force | Product ecodesign requirements framework |
| July 2024 | CSDDD entered into force | Supply chain due diligence |
| August 2024 | IED 2.0 entered into force | Stricter industrial emission limits |
| October 2024 | CSRD Wave 1 reporting begins (large PIEs FY2024) | First ESRS reports due April 2026 |
| **2025** | | |
| February 2025 | Battery carbon footprint declaration (EV batteries) | Mandatory for EV battery manufacturers |
| October 2025 | EED transposition deadline | Energy audit/management obligations |
| October 2025 | Revised Waste Framework Directive enters into force | EPR textile obligations triggered |
| **2026** | | |
| January 2026 | CBAM definitive period begins | Importers must buy/surrender certificates |
| February 2026 | Omnibus I Directive published | CSRD scope narrowed, sector standards eliminated |
| June 2026 | EU Pay Transparency Directive transposition | Gender pay gap reporting |
| July 2026 | IED 2.0 Member State transposition | New permits with stricter BAT-AELs |
| July 2026 | CSDDD Member State transposition | National due diligence laws |
| August 2026 | Hydrogen/Gas Package Member State transposition | Hydrogen market rules |
| August 2026 | Battery carbon footprint declaration (industrial >2kWh) | Mandatory for industrial battery manufacturers |
| October 2026 | First EED energy audits due (10-85 TJ/year) | Manufacturing facilities audit obligation |
| **2027** | | |
| January 2027 | Revised ESRS application (FY2027 reporting) | CSRD reports using amended standards |
| February 2027 | Battery Passport (DPP) for EV + industrial batteries | QR code + digital passport required |
| Mid-2027 | Textile DPP expected | Product passport for textiles |
| October 2027 | EED energy management systems (>85 TJ/year) | ISO 50001 mandatory for large consumers |
| **2028** | | |
| 2028 | Iron/steel, aluminum DPP delegated acts expected | Product passport for metals |
| August 2028 | Battery recycled content declaration | Cobalt, lead, lithium, nickel |
| **2029** | | |
| July 2029 | CSDDD single application date | All in-scope companies must comply |
| **2030** | | |
| 2030 | EU ETS: 62% below 2005 levels | Free allocation at 51.5% for CBAM sectors |
| 2030 | 42.5% renewable energy target | Industrial renewable share must increase |
| 2030 | Food waste reduction target | 10% reduction in manufacturing |
| 2030 | CBAM factor reaches 48.5% | Material carbon cost impact |
| **2034** | | |
| 2034 | Free allocation fully phased out | CBAM at 100%, no more free EU ETS allowances |

## Appendix B: GreenLang Agent Mapping for Manufacturing Pack

| Regulatory Area | ESRS Standard | Key Agents | Pack Engine Needed |
|----------------|---------------|------------|-------------------|
| GHG emissions (Scope 1) | E1 | AGENT-MRV-001 to 008 | Scope 1 Manufacturing Engine |
| GHG emissions (Scope 2) | E1 | AGENT-MRV-009 to 013 | Scope 2 Dual Reporting Engine |
| GHG emissions (Scope 3) | E1 | AGENT-MRV-014 to 030 | Scope 3 Manufacturing Engine |
| Pollution | E2 | Data Quality agents + custom | Pollution Tracking Engine |
| Water | E3 | AGENT-DATA-020 (Climate Hazard) + custom | Water Intensity Engine |
| Biodiversity | E4 | AGENT-EUDR (supply chain) + custom | Biodiversity Assessment Engine |
| Circular economy | E5 | AGENT-DATA-008 to 010 + custom | Circular Economy Engine |
| Own workforce | S1 | Custom (HR data integration) | OHS Metrics Engine |
| Value chain workers | S2 | AGENT-DATA-008 (Supplier Questionnaire) | Supply Chain Social Engine |
| EU ETS compliance | (Regulatory) | AGENT-MRV + CBAM agents | EU ETS Compliance Engine |
| CBAM | (Regulatory) | GL-CBAM-APP | CBAM Integration Bridge |
| Product carbon footprint | (Regulatory) | AGENT-MRV + LCA agents | PCF Calculation Engine |
| Digital Product Passport | (Regulatory) | Custom | DPP Data Assembly Engine |
| Energy management (EED) | (Regulatory) | AGENT-MRV + AGENT-DATA | Energy Management Engine |
| BAT compliance | (Regulatory) | Custom | BAT Performance Engine |

## Appendix C: Risk and Penalty Summary

| Regulation | Non-Compliance Penalty | Risk Level |
|-----------|----------------------|------------|
| CSRD/ESRS | Member State defined; market exclusion risk; auditor liability | High |
| EU ETS | EUR 100/tCO2e excess emissions + name-and-shame | Very High |
| CBAM | EUR 10-50/tCO2e undeclared; EUR 100/tonne unsurrendered certificates | Very High |
| IED | Min EUR 3M or 3% annual turnover; permit revocation | Very High |
| EED | Member State defined; mandatory corrective measures | Medium-High |
| CSDDD | Up to 5% worldwide net annual turnover + civil liability | Very High |
| REACH | Up to EUR 50,000 per substance per Member State; market withdrawal | High |
| Battery Regulation | Withdrawal of non-compliant batteries from EU market | High |
| ESPR/DPP | Withdrawal from market; customs seizure | High |
| Waste Framework Directive | Member State defined; administrative sanctions | Medium |
| EUDR | Fines proportionate to environmental damage + product confiscation | Very High |

---

## Sources

- [EFRAG Sector-specific ESRS Workstream](https://www.efrag.org/en/sustainability-reporting/esrs-workstreams/sectorspecific-esrs)
- [Omnibus I Directive -- Crowell & Moring Analysis](https://www.crowell.com/en/insights/client-alerts/eu-sustainability-reporting-revamp-key-updates-to-the-csrd-and-the-cs3d-from-the-omnibus-i-directive)
- [Omnibus I Directive Published -- PwC Viewpoint](https://viewpoint.pwc.com/gx/en/pwc/in-briefs/ib_int202527.html)
- [CSRD Omnibus Update 2026 -- IntegrityNext](https://www.integritynext.com/resources/blog/article/csrd-omnibus-update-2026-esrs-simplification-what-companies-must-do-now)
- [Amended ESRS Explained -- Coolset](https://www.coolset.com/academy/the-amended-esrs-what-has-changed-and-what-it-means-for-2026-csrd-reporting)
- [CBAM Timeline and Deadlines -- Coolset](https://www.coolset.com/academy/cbam-timeline-deadlines-phases-what-to-expect-2026)
- [CBAM Official Page -- European Commission](https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en)
- [EU ETS Reforms -- ICAP](https://icapcarbonaction.com/en/ets/eu-emissions-trading-system-eu-ets)
- [CBAM Phase-In Schedule -- Belgian Climate Portal](https://climat.be/cbam-en/cbam-certificates/gradual-cbam-phase-in)
- [Energy Efficiency Directive -- European Commission](https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive_en)
- [EED Company Obligations 2026-2030 -- Wirtek](https://www.wirtek.com/blog/energy-efficiency-directive-eu-company-obligations-from-2026-to-2030)
- [ISO 50001 and EED -- Accevo](https://accevo.com/blog/iso-50001-and-eu-directive-2023-1791-energy-management-in-manufacturing-facilities/)
- [IED 2.0 -- European Commission](https://environment.ec.europa.eu/topics/industrial-emissions-and-safety/industrial-and-livestock-rearing-emissions-directive-ied-20_en)
- [IED Revised Provisions -- Desotec](https://www.desotec.com/en-us/knowledge-hub/article/the-revised-industrial-emission-directive)
- [ISO 14067:2018 -- ISO](https://www.iso.org/standard/71206.html)
- [Digital Product Passport Guide -- Climatiq](https://www.climatiq.io/blog/digital-product-passports-what-you-need-to-know-to-be-ready-for-regulatory-compliance-in-2025)
- [ESPR DPP Requirements -- TracexTech](https://tracextech.com/espr-dpp-regulation/)
- [DPP 2025-2030 Timeline -- Fluxy](https://fluxy.one/post/digital-product-passport-dpp-eu-guide-2025-2030)
- [EU Battery Regulation -- Flash Battery](https://www.flashbattery.tech/en/blog/eu-battery-regulation-obligations-updates/)
- [Battery Passport Requirements -- Circularise](https://www.circularise.com/blogs/eu-battery-passport-regulation-requirements)
- [Waste Framework Directive Revision -- European Commission](https://environment.ec.europa.eu/topics/waste-and-recycling/waste-framework-directive_en)
- [Revised WFD Enters Into Force -- European Commission](https://environment.ec.europa.eu/news/revised-waste-framework-directive-enters-force-2025-10-16_en)
- [ESRS E2 Pollution After Omnibus -- Coolset](https://www.coolset.com/academy/esrs-e2-pollution)
- [REACH Regulation -- European Commission](https://environment.ec.europa.eu/topics/chemicals/reach-regulation_en)
- [REACH SVHC Requirements -- Enviropass](https://getenviropass.com/reach-svhc/)
- [TNFD LEAP Approach -- TNFD](https://tnfd.global/publication/additional-guidance-on-assessment-of-nature-related-issues-the-leap-approach/)
- [WRI Aqueduct 4.0 -- WRI](https://www.wri.org/research/aqueduct-40-updated-decision-relevant-global-water-risk-indicators)
- [CDP Water Security Guidance -- CDP](https://guidance.cdp.net/en/guidance?cid=6980&ctype=record&idtype=RecordID&incchild=1&microsite=0&otype=Guidance)
- [ESRS S1 After Omnibus -- Coolset](https://www.coolset.com/academy/esrs-s1-requirements-own-workforce)
- [ESRS S2 After Omnibus -- Coolset](https://www.coolset.com/academy/esrs-s2-workers-in-the-value-chain)
- [CSDDD Overview -- csddd.com](https://www.corporate-sustainability-due-diligence-directive.com/)
- [CSDDD Explained -- Normative](https://normative.io/insight/csddd/)
- [CSDDD Obligations -- White & Case](https://www.whitecase.com/insight-alert/time-get-know-your-supply-chain-eu-adopts-corporate-sustainability-due-diligence)
- [Scope 3 Industrial Emissions -- Cambridge CIIP](https://www.ciip.group.cam.ac.uk/reports-and-articles/understanding-industrial-scope-3-emissions/)
- [CDP Scope 3 Relevance by Sector](https://cdn.cdp.net/cdp-production/cms/guidance_docs/pdfs/000/003/504/original/CDP-technical-note-scope-3-relevance-by-sector.pdf)
- [GHG Protocol Scope 3 Technical Guidance](https://ghgprotocol.org/sites/default/files/standards/Scope3_Calculation_Guidance_0.pdf)
- [OEE -- oee.com](https://www.oee.com/)
- [OEEE Sustainability Metric -- MDPI](https://www.mdpi.com/2071-1050/7/7/9031)
- [Fit for 55 and Hydrogen -- Hydrogen Council](https://hydrogencouncil.com/en/fitfor55-fit-for-purpose/)
- [EU Hydrogen Strategy -- European Hydrogen Observatory](https://observatory.clean-hydrogen.europa.eu/eu-policy/eu-hydrogen-strategy-under-eu-green-deal)
- [Supplier Engagement for Scope 3 -- CarbonCloud](https://carboncloud.com/blog/engaging-suppliers/)
- [PCAF Data Quality -- StepChange](https://www.stepchange.earth/blog/how-data-quality-works-under-pcaf-the-foundation-of-credible-financed-emissions-reporting)
