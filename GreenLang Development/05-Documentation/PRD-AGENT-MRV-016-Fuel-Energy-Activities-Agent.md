# PRD: AGENT-MRV-016 -- Scope 3 Category 3 Fuel & Energy Activities Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-003 |
| **Internal Label** | AGENT-MRV-016 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/fuel_energy_activities/` |
| **DB Migration** | V067 |
| **Metrics Prefix** | `gl_fea_` |
| **Table Prefix** | `gl_fea_` |
| **API** | `/api/v1/fuel-energy-activities` |
| **Env Prefix** | `GL_FEA_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The Fuel & Energy Activities Agent implements **GHG Protocol Scope 3
Category 3: Fuel- and Energy-Related Activities (not included in Scope 1
or Scope 2)**. This category captures emissions from the upstream
lifecycle of fuels and energy that are consumed by the reporting company
but fall outside the boundaries of direct combustion (Scope 1) and
purchased electricity generation (Scope 2). It is the essential
"missing piece" that completes the full lifecycle accounting for every
unit of fuel burned and every kilowatt-hour of electricity consumed.

Category 3 encompasses four distinct sub-activities:

- **Activity 3a** -- Upstream emissions of purchased fuels (well-to-tank):
  extraction, refining, processing, and transportation of fuels that are
  ultimately combusted in Scope 1 operations.
- **Activity 3b** -- Upstream emissions of purchased electricity, steam,
  heat, and cooling: the upstream fuel lifecycle emissions associated
  with the generation of energy purchased under Scope 2.
- **Activity 3c** -- Transmission and distribution (T&D) losses:
  generation-phase emissions attributable to electricity lost during
  transmission and distribution between the power plant and the
  reporting company's meter.
- **Activity 3d** -- Generation of purchased electricity that is sold to
  end users: applicable only to utilities and energy retailers that
  purchase electricity for resale.

These four sub-activities are universally relevant -- every company that
reports Scope 1 fuel combustion or Scope 2 electricity consumption has
corresponding Category 3 emissions. As a result, Category 3 is almost
always material (typically 5-20% of total Scope 3) and is one of the
most commonly reported Scope 3 categories. The agent automates the
historically manual process of matching fuel consumption records to
well-to-tank (WTT) emission factors, applying country-specific grid
upstream factors, calculating T&D losses with regional loss percentages,
and ensuring zero double-counting against Scope 1 and Scope 2.

### Justification for Dedicated Agent

1. **Universal relevance** -- Unlike most Scope 3 categories that are
   industry-specific, Category 3 applies to every organization that
   consumes fuel or electricity, making it one of the highest-coverage
   categories in GHG inventories
2. **Complex WTT factor landscape** -- Well-to-tank emission factors vary
   by fuel type (20+ fuels), supply chain stage (extraction, refining,
   transport), and source database (DEFRA, EPA, ecoinvent, IEA), requiring
   a dedicated factor management engine
3. **T&D loss factor variability** -- Transmission and distribution loss
   percentages vary dramatically by country (3.4% in South Korea to 19%
   in India), grid region, and voltage level, requiring a specialized
   geographic factor engine
4. **Double-counting prevention against Scope 1 AND Scope 2** -- Category 3
   sits precisely at the boundary between Scopes 1, 2, and 3, requiring
   rigorous boundary enforcement: WTT factors must exclude combustion
   (Scope 1) and upstream factors must exclude generation-point emissions
   (Scope 2)
5. **Dual accounting approach** -- Activity 3b requires both location-based
   and market-based upstream calculations, mirroring the Scope 2 dual
   reporting methodology
6. **Regulatory urgency** -- CSRD (FY2025+), SB 253 (FY2027), SBTi, CDP
   all require or strongly encourage Category 3 reporting; its universal
   applicability makes it a baseline expectation for credible disclosure
7. **Integration dependency** -- Category 3 calculations depend directly on
   Scope 1 fuel consumption data (Activity 3a) and Scope 2 electricity
   consumption data (Activities 3b/3c), requiring tight integration with
   MRV agents 001-013

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011) -- Chapter 5
- GHG Protocol Scope 3 Technical Guidance (2013) -- Chapter 3: Category 3
- GHG Protocol Scope 3 Calculation Guidance (online)
- GHG Protocol Scope 2 Guidance (2015) -- Location vs Market-based methods
- GHG Protocol Quantitative Uncertainty Guidance
- DEFRA/DESNZ Greenhouse Gas Reporting Conversion Factors (annual) -- WTT tables
- EPA Emission Factors Hub -- US-specific upstream factors
- IEA World Energy Outlook -- Country-specific energy data
- IEA Electricity Information -- Grid emission factors and T&D loss rates
- ecoinvent v3.11 -- LCA database with upstream fuel lifecycle data
- eGRID (EPA) -- US subregional grid emission factors
- World Bank Indicators -- T&D loss data by country
- CSRD/ESRS E1 -- Scope 3 disclosure (E1-6 para 44)
- California SB 253 -- Mandatory Scope 3 by FY2027 for entities >$1B revenue
- CDP Climate Change Questionnaire -- C6.5 Scope 3 Category 3
- SBTi Corporate Manual v5.3 -- Scope 3 target required if >40% of total
- GRI 305 -- Emissions (Scope 3 disclosure)
- ISO 14064-1:2018 -- Category 4 indirect GHG emissions

### Terminology

| Term | Definition | Scope Mapping |
|------|-----------|---------------|
| **WTT** (Well-to-Tank) | Upstream lifecycle emissions from extraction through delivery to point of combustion | Scope 3 Category 3 (Activity 3a) |
| **TTW** (Tank-to-Wheel) | Direct combustion emissions at point of use | Scope 1 |
| **WTW** (Well-to-Wheel) | Complete lifecycle: WTT + TTW combined | Scope 1 + Scope 3 Cat 3 |
| **Upstream EF** | Emission factor for energy generation fuel lifecycle (not the generation itself) | Scope 3 Category 3 (Activity 3b) |
| **T&D Loss** | Electricity lost during transmission and distribution, measured as % of generation | Scope 3 Category 3 (Activity 3c) |
| **Generation EF** | Emission factor at the point of electricity generation | Scope 2 |
| **Grid Mix** | Composition of fuel sources used in electricity generation for a region | Applies to Activities 3b/3c |

---

## 2. Methodology

### 2.1 The Four Sub-Activities of Category 3

Category 3 is unique among Scope 3 categories in that it explicitly
decomposes into four sub-activities, each with its own calculation
methodology, data requirements, and factor databases. Every reporting
company must assess which sub-activities are applicable:

| Sub-Activity | Description | Applicability | Typical Contribution |
|-------------|-------------|--------------|---------------------|
| **3a** | Upstream emissions of purchased fuels (WTT) | All companies with Scope 1 fuel combustion | 40-60% of Cat 3 total |
| **3b** | Upstream emissions of purchased electricity/steam/heat/cooling | All companies with Scope 2 energy purchases | 25-40% of Cat 3 total |
| **3c** | T&D losses | All companies purchasing grid electricity | 10-25% of Cat 3 total |
| **3d** | Generation of purchased electricity sold to end users | Utilities and energy retailers ONLY | 0% (most companies) or 100% (utilities) |

**Decision tree for sub-activity applicability:**

```
Does the company combust fuels in owned/controlled operations (Scope 1)?
  YES --> Activity 3a is applicable
  NO  --> Activity 3a is NOT applicable

Does the company purchase electricity, steam, heat, or cooling (Scope 2)?
  YES --> Activity 3b is applicable
  NO  --> Activity 3b is NOT applicable

Does the company purchase electricity from a grid (not 100% on-site generation)?
  YES --> Activity 3c is applicable
  NO  --> Activity 3c is NOT applicable

Does the company purchase electricity and sell it to end users (utility/retailer)?
  YES --> Activity 3d is applicable
  NO  --> Activity 3d is NOT applicable
```

### 2.2 Activity 3a -- Upstream Emissions of Purchased Fuels (WTT)

#### 2.2.1 Concept

Well-to-tank (WTT) emissions encompass all greenhouse gas emissions
generated during the extraction, processing, refining, and transportation
of fuels BEFORE they are combusted by the reporting company. For example,
the WTT emissions of natural gas include methane leakage during
extraction, energy consumed in processing plants, compression for
pipeline transport, and any CO2 vented during purification.

**WTT emission stages:**

```
Stage 1: EXTRACTION
  - Mining (coal), drilling (oil/gas), harvesting (biomass)
  - Fugitive methane from wells, mines, seams
  - Energy consumed by extraction equipment

Stage 2: PROCESSING / REFINING
  - Crude oil refining to diesel, gasoline, jet fuel, fuel oil
  - Natural gas processing (sweetening, dehydrating, NGL removal)
  - Coal washing and preparation
  - Biofuel production (fermentation, transesterification)

Stage 3: TRANSPORTATION / DISTRIBUTION
  - Pipeline transport (natural gas, crude oil, refined products)
  - Tanker/ship transport (LNG, crude oil)
  - Rail and truck transport (coal, refined fuels)
  - Storage emissions (tank breathing, evaporative losses)

Stage 4: DELIVERY TO POINT OF USE
  - Local distribution network losses
  - Final-mile truck delivery
  - On-site storage and handling
```

#### 2.2.2 Calculation Method

**Core formula:**

```
Emissions_3a = SUM over all fuel types i:
    Fuel_consumed_i * WTT_EF_i
```

**Per-gas breakdown:**

```
Emissions_3a_CO2  = SUM(Fuel_consumed_i * WTT_EF_CO2_i)
Emissions_3a_CH4  = SUM(Fuel_consumed_i * WTT_EF_CH4_i * GWP_CH4)
Emissions_3a_N2O  = SUM(Fuel_consumed_i * WTT_EF_N2O_i * GWP_N2O)
Emissions_3a_total = Emissions_3a_CO2 + Emissions_3a_CH4 + Emissions_3a_N2O
```

**With unit conversion:**

```
Emissions_3a = SUM over all fuel types i:
    Fuel_consumed_i * Unit_conversion_factor_i * WTT_EF_i

Where Unit_conversion_factor converts from input unit to EF reference unit
(e.g., litres -> kWh using fuel density and heating value)
```

#### 2.2.3 WTT Emission Factors by Fuel Type

**Fossil Fuels:**

| Fuel Type | WTT EF (kgCO2e/kWh) | WTT as % of Combustion | Primary Source |
|-----------|---------------------|----------------------|---------------|
| Natural Gas | 0.025 - 0.039 | 12 - 19% | DEFRA 2025, EPA |
| Diesel / Gas Oil | 0.055 - 0.063 | 18 - 24% | DEFRA 2025 |
| Petrol / Gasoline | 0.054 - 0.059 | 18 - 24% | DEFRA 2025 |
| Coal (bituminous) | 0.035 - 0.045 | 10 - 13% | DEFRA 2025 |
| Coal (sub-bituminous) | 0.030 - 0.040 | 9 - 12% | DEFRA 2025 |
| Coal (anthracite) | 0.038 - 0.048 | 11 - 14% | DEFRA 2025 |
| Fuel Oil (residual) | 0.048 - 0.055 | 16 - 20% | DEFRA 2025 |
| Fuel Oil (distillate) | 0.052 - 0.060 | 17 - 22% | DEFRA 2025 |
| LPG | 0.030 - 0.040 | 12 - 16% | DEFRA 2025 |
| Kerosene / Jet Fuel | 0.050 - 0.058 | 17 - 22% | DEFRA 2025 |
| Naphtha | 0.048 - 0.055 | 16 - 20% | ecoinvent 3.11 |
| Propane | 0.028 - 0.036 | 11 - 15% | DEFRA 2025 |
| Butane | 0.030 - 0.038 | 12 - 16% | DEFRA 2025 |
| CNG (Compressed Natural Gas) | 0.030 - 0.045 | 14 - 22% | DEFRA 2025 |
| LNG (Liquefied Natural Gas) | 0.045 - 0.065 | 20 - 30% | DEFRA 2025 |
| Petroleum Coke | 0.025 - 0.035 | 7 - 10% | ecoinvent 3.11 |

**Biofuels:**

| Fuel Type | WTT EF (kgCO2e/kWh) | WTT as % of Combustion | Primary Source |
|-----------|---------------------|----------------------|---------------|
| Bioethanol (corn) | 0.080 - 0.120 | N/A (biogenic CO2) | EPA RFS |
| Bioethanol (sugarcane) | 0.030 - 0.060 | N/A (biogenic CO2) | ecoinvent 3.11 |
| Biodiesel (FAME, rapeseed) | 0.070 - 0.110 | N/A (biogenic CO2) | DEFRA 2025 |
| Biodiesel (FAME, soy) | 0.060 - 0.100 | N/A (biogenic CO2) | EPA RFS |
| HVO / Renewable Diesel | 0.025 - 0.055 | N/A (biogenic CO2) | DEFRA 2025 |
| Biogas (anaerobic digestion) | 0.015 - 0.035 | N/A (biogenic CO2) | DEFRA 2025 |
| Biomethane (upgraded biogas) | 0.010 - 0.030 | N/A (biogenic CO2) | DEFRA 2025 |
| Wood Pellets | 0.020 - 0.045 | N/A (biogenic CO2) | IEA Bioenergy |
| Wood Chips | 0.010 - 0.025 | N/A (biogenic CO2) | IEA Bioenergy |

**Key notes on biofuels:** For biofuels, the combustion (TTW) CO2 is
classified as biogenic and reported separately (Scope 1 memo item), but
the WTT emissions include fossil-based upstream emissions from
cultivation, processing, and transport -- these are always reported in
Category 3.

#### 2.2.4 WTT Factor Sources

| Source | Coverage | Update Frequency | Granularity |
|--------|----------|-----------------|-------------|
| DEFRA/DESNZ Conversion Factors | 30+ fuel types, UK-specific with global applicability | Annual (June) | Per-gas (CO2, CH4, N2O) and CO2e |
| EPA Emission Factors Hub | 20+ fuel types, US-specific | Annual | CO2e aggregate |
| ecoinvent v3.11 | 50+ fuel types, multi-regional | Biennial | Full LCA per-process |
| IEA World Energy Outlook | 15+ fuel types, 50+ countries | Annual | Country-specific WTT |
| IPCC Emission Factor Database (EFDB) | Tier 1 default factors, global | AR cycle (~7 years) | Global defaults |

### 2.3 Activity 3b -- Upstream Emissions of Purchased Electricity

#### 2.3.1 Concept

Activity 3b captures the upstream emissions associated with producing
the fuels that are combusted in power plants to generate the electricity,
steam, heat, or cooling that the reporting company purchases under
Scope 2. This is NOT the generation-point emission (that is Scope 2);
it is the WTT emissions of the fuels used by the power generators.

**Example:** When a company purchases 1 MWh of electricity from a grid
that is 40% natural gas, 30% coal, 20% nuclear, and 10% wind:

```
Upstream EF = (0.40 * WTT_gas_per_kWh_generated)
            + (0.30 * WTT_coal_per_kWh_generated)
            + (0.20 * WTT_nuclear_per_kWh_generated)
            + (0.10 * WTT_wind_per_kWh_generated)  -- effectively zero
```

#### 2.3.2 Calculation Methods

**Location-based upstream calculation:**

```
Emissions_3b_location = SUM over all energy types e:
    Energy_consumed_e * Upstream_EF_location_e
```

Where `Upstream_EF_location_e` is derived from the grid-mix-weighted
average of WTT factors for all fuel sources in the generation mix of
the local grid region.

**Market-based upstream calculation:**

```
Emissions_3b_market = SUM over all energy types e:
    Energy_consumed_e * Upstream_EF_market_e
```

Where `Upstream_EF_market_e` reflects the upstream emissions of the
specific contractual instrument (e.g., a supplier-specific factor from
a named generator, or a residual mix upstream factor).

**For steam, heat, and cooling:**

```
Emissions_3b_thermal = Energy_consumed_thermal * Upstream_EF_thermal

Where Upstream_EF_thermal depends on:
  - Fuel type used by the thermal plant (gas, coal, biomass)
  - Boiler/chiller efficiency
  - CHP allocation method (if applicable)
```

#### 2.3.3 Upstream Electricity Factors by Country

| Country | Grid Carbon Intensity (gCO2e/kWh, generation) | Upstream EF (gCO2e/kWh) | Upstream as % of Generation |
|---------|----------------------------------------------|------------------------|---------------------------|
| United States | 370 - 410 | 45 - 65 | 12 - 16% |
| United Kingdom | 180 - 220 | 25 - 35 | 13 - 16% |
| Germany | 340 - 400 | 40 - 55 | 12 - 14% |
| France | 55 - 75 | 8 - 12 | 14 - 16% |
| Japan | 450 - 500 | 55 - 70 | 12 - 14% |
| Australia | 550 - 650 | 50 - 65 | 8 - 10% |
| China | 530 - 600 | 55 - 75 | 10 - 13% |
| India | 650 - 750 | 55 - 70 | 8 - 9% |
| Brazil | 60 - 90 | 8 - 15 | 13 - 17% |
| Canada | 120 - 160 | 15 - 25 | 12 - 16% |
| South Korea | 400 - 460 | 50 - 60 | 12 - 13% |
| Italy | 250 - 310 | 30 - 42 | 12 - 14% |
| Spain | 150 - 200 | 20 - 30 | 13 - 15% |
| Netherlands | 320 - 380 | 38 - 50 | 12 - 13% |
| Poland | 650 - 750 | 50 - 60 | 7 - 8% |
| Sweden | 10 - 25 | 2 - 5 | 15 - 20% |
| Norway | 8 - 20 | 1 - 4 | 13 - 20% |
| South Africa | 850 - 950 | 55 - 70 | 6 - 7% |
| Mexico | 400 - 460 | 50 - 60 | 12 - 13% |
| Turkey | 380 - 440 | 42 - 55 | 11 - 13% |
| Indonesia | 700 - 780 | 55 - 65 | 8 - 8% |
| Thailand | 420 - 480 | 50 - 60 | 12 - 13% |
| Vietnam | 500 - 580 | 45 - 55 | 9 - 10% |
| Philippines | 550 - 650 | 50 - 60 | 9 - 9% |
| Argentina | 300 - 370 | 38 - 48 | 13 - 13% |
| Chile | 320 - 400 | 35 - 48 | 11 - 12% |
| Colombia | 120 - 180 | 15 - 25 | 13 - 14% |
| Singapore | 360 - 420 | 42 - 52 | 12 - 12% |
| UAE | 420 - 500 | 50 - 62 | 12 - 12% |
| Saudi Arabia | 550 - 650 | 55 - 70 | 10 - 11% |

### 2.4 Activity 3c -- Transmission & Distribution Losses

#### 2.4.1 Concept

When electricity is transmitted over the grid from the power plant to
the consumer, a percentage is lost as heat due to resistance in
transmission lines and transformers. These "T&D losses" represent
electricity that was generated (with associated emissions) but never
delivered to any end user. Category 3 Activity 3c captures the
generation-phase emissions attributable to this lost electricity.

**The full T&D loss emission includes both:**
1. The generation-point emissions of the lost electricity (Scope 2-equivalent
   emissions that "belong to no one")
2. The upstream (WTT) emissions of the fuels used to generate that lost
   electricity

#### 2.4.2 Calculation Method

**Core formula:**

```
Emissions_3c = Electricity_consumed * T&D_loss_% / (1 - T&D_loss_%)
             * (Generation_EF + Upstream_EF)
```

**Alternative simpler formula (when T&D loss EF is pre-calculated):**

```
Emissions_3c = Electricity_consumed * T&D_loss_EF
```

**Note:** The division by `(1 - T&D_loss_%)` converts from delivered
electricity (what the company meters) to generated electricity (what the
power plant produced). Some factor databases (e.g., DEFRA) provide
pre-calculated T&D loss emission factors that already include this
conversion.

**Location-based T&D loss:**

```
Emissions_3c_location = Elec_consumed * T&D_loss_%_country
                      * (Grid_EF_location + Upstream_EF_location)
                      / (1 - T&D_loss_%_country)
```

**Market-based T&D loss:**

```
Emissions_3c_market = Elec_consumed * T&D_loss_%_region
                    * (Grid_EF_market + Upstream_EF_market)
                    / (1 - T&D_loss_%_region)
```

#### 2.4.3 T&D Loss Factors by Country

| Country | T&D Loss (%) | Source | Year | Notes |
|---------|-------------|--------|------|-------|
| United States | 5.0 - 5.9 | EIA, eGRID | 2024 | Varies by state (2.5-8%) |
| United Kingdom | 7.8 - 8.4 | DEFRA/DESNZ | 2025 | Includes losses at all voltage levels |
| Germany | 3.9 | IEA | 2023 | One of the lowest globally |
| France | 6.2 | IEA | 2023 | |
| Japan | 4.3 | IEA | 2023 | |
| Australia | 4.8 | IEA | 2023 | Varies by state (3.5-7%) |
| China | 4.5 | IEA | 2023 | National average; rural higher |
| India | 18.0 - 19.0 | World Bank, CEA | 2023 | Among highest globally; state variance 8-30% |
| Brazil | 15.8 | IEA, ANEEL | 2023 | Varies significantly by region |
| Canada | 5.8 | IEA | 2023 | |
| South Korea | 3.4 | IEA | 2023 | Among lowest globally |
| Italy | 6.0 | IEA | 2023 | |
| Spain | 8.5 | IEA | 2023 | |
| Netherlands | 4.2 | IEA | 2023 | |
| Poland | 6.5 | IEA | 2023 | |
| Sweden | 6.5 | IEA | 2023 | |
| Norway | 6.0 | IEA | 2023 | |
| South Africa | 8.5 | Eskom, IEA | 2023 | Load shedding impacts accuracy |
| Mexico | 13.5 | IEA, CFE | 2023 | |
| Turkey | 12.0 | IEA | 2023 | |
| Indonesia | 9.5 | IEA, PLN | 2023 | Island variability 6-15% |
| Thailand | 6.0 | IEA | 2023 | |
| Vietnam | 7.5 | IEA | 2023 | |
| Philippines | 10.5 | IEA | 2023 | Island grid losses higher |
| Argentina | 14.0 | IEA | 2023 | |
| Chile | 6.5 | IEA | 2023 | |
| Colombia | 12.0 | IEA | 2023 | |
| Singapore | 2.5 | IEA | 2023 | Small island grid; very low losses |
| UAE | 7.0 | IEA | 2023 | |
| Saudi Arabia | 7.5 | IEA | 2023 | |
| New Zealand | 6.5 | IEA | 2023 | |
| Switzerland | 5.5 | IEA | 2023 | |
| Belgium | 4.5 | IEA | 2023 | |
| Austria | 5.0 | IEA | 2023 | |
| Denmark | 5.5 | IEA | 2023 | |
| Finland | 3.5 | IEA | 2023 | |
| Ireland | 7.5 | IEA | 2023 | |
| Portugal | 8.0 | IEA | 2023 | |
| Greece | 6.5 | IEA | 2023 | |
| Czech Republic | 5.0 | IEA | 2023 | |
| Romania | 11.0 | IEA | 2023 | |
| Hungary | 10.5 | IEA | 2023 | |
| Pakistan | 17.0 | World Bank | 2023 | |
| Bangladesh | 12.0 | World Bank | 2023 | |
| Nigeria | 16.0 | World Bank | 2023 | |
| Egypt | 11.0 | IEA | 2023 | |
| Kenya | 18.5 | World Bank | 2023 | |
| Malaysia | 5.5 | IEA | 2023 | |
| Taiwan | 4.0 | IEA | 2023 | |
| Israel | 3.5 | IEA | 2023 | |

**US eGRID Subregional T&D Loss Factors:**

| eGRID Subregion | Code | T&D Loss (%) | States Covered |
|----------------|------|-------------|----------------|
| ASCC Alaska Grid | AKGD | 5.5 | AK |
| ASCC Miscellaneous | AKMS | 6.0 | AK (rural) |
| WECC California | CAMX | 4.8 | CA, NV (partial) |
| WECC Northwest | NWPP | 5.2 | WA, OR, ID, MT, WY (partial) |
| WECC Rockies | RMPA | 5.5 | CO, WY (partial) |
| WECC Southwest | AZNM | 5.8 | AZ, NM |
| SPP | SPSO | 5.5 | OK, KS, NE (partial) |
| ERCOT | ERCT | 4.5 | TX (most) |
| MRO East | MROE | 5.0 | WI, MN (partial) |
| MRO West | MROW | 5.2 | IA, MN, ND, SD, NE (partial) |
| RFC East | RFCE | 5.5 | PA, NJ, DE, MD, DC |
| RFC Michigan | RFCM | 5.8 | MI |
| RFC West | RFCW | 5.5 | OH, IN, WV |
| SERC Midwest | SRMW | 5.5 | MO, IL (partial) |
| SERC South | SRSO | 5.2 | AL, MS, GA (partial) |
| SERC Southeast | SRTV | 5.0 | TN, VA, NC (partial) |
| SERC Virginia/Carolina | SRVC | 5.5 | VA, NC, SC (partial) |
| FRCC | FRCC | 5.0 | FL |
| NPCC New England | NEWE | 5.5 | CT, MA, ME, NH, RI, VT |
| NPCC NYC/Westchester | NYCW | 7.5 | NY (NYC metro) |
| NPCC Long Island | NYLI | 6.0 | NY (Long Island) |
| NPCC Upstate NY | NYUP | 5.0 | NY (upstate) |
| HICC Oahu | HIMS | 5.0 | HI (Oahu) |
| HICC Miscellaneous | HIOA | 6.0 | HI (other islands) |
| PR | PRMS | 8.5 | Puerto Rico |

### 2.5 Activity 3d -- Generation of Purchased Electricity Sold to End Users

#### 2.5.1 Concept

Activity 3d applies exclusively to companies that purchase electricity
and sell it to end users -- typically electric utilities, energy
retailers, and community choice aggregators (CCAs). For these
organizations, the purchased electricity they sell is not consumed in
their own operations but is a product they distribute.

**Applicability filter:**

```
Is the company an electricity utility, energy retailer, or CCA?
  YES --> Activity 3d may be applicable for purchased (not self-generated) electricity sold
  NO  --> Activity 3d is NOT applicable (skip entirely)
```

#### 2.5.2 Calculation Method

```
Emissions_3d = Electricity_sold_to_end_users * Lifecycle_EF

Where:
  Lifecycle_EF = Generation_EF + Upstream_EF + T&D_EF
```

**For utilities that both generate and purchase:**

```
Emissions_3d = Electricity_purchased_for_resale * Lifecycle_EF_purchased

Note: Self-generated electricity sold to end users is reported
in Scope 1 (combustion) and does NOT appear in Category 3.
```

### 2.6 Supplier-Specific Approach

When direct supplier data is available for upstream emissions, it
takes precedence over default WTT factors. The supplier-specific
approach integrates:

**For fuels (Activity 3a):**
- Fuel supplier environmental product declarations (EPDs)
- Supplier-reported lifecycle emissions per unit of fuel delivered
- Product Carbon Footprints (PCFs) from refineries
- Certified upstream emissions from natural gas producers (MiQ, OGMP 2.0)

**For electricity (Activities 3b/3c):**
- Utility-specific upstream factors from sustainability reports
- Generator-specific fuel mix and upstream data
- Green tariff or PPA contracts with specified upstream data
- Residual mix factors from regional bodies (AIB, Green-e)

**Formula with supplier-specific data:**

```
Emissions_3a_ss = SUM over suppliers s:
    Fuel_purchased_from_s * WTT_EF_supplier_s

Emissions_3b_ss = SUM over suppliers s:
    Electricity_purchased_from_s * Upstream_EF_supplier_s
```

### 2.7 Data Quality Indicator (DQI)

Per GHG Protocol Scope 3 Standard Chapter 7, five data quality indicators
are assessed on a 1-5 scale:

| Indicator | Score 1 (Very Good) | Score 3 (Fair) | Score 5 (Very Poor) |
|-----------|--------------------|-----------------|--------------------|
| Temporal | Data from reporting year | Data within 6 years | Data older than 10 years |
| Geographical | Same country/region | Same continent | Global average |
| Technological | Same fuel type and supply chain | Related fuel category | Different energy type entirely |
| Completeness | All fuel types and facilities included | 50-80% covered | Less than 20% covered |
| Reliability | Third-party verified supplier data | Established database (DEFRA/EPA) | Estimate or assumption |

**Composite DQI:**

```
DQI_composite = (DQI_temporal + DQI_geographical + DQI_technological
                + DQI_completeness + DQI_reliability) / 5
```

**Quality classification:**

| DQI Range | Classification | Recommended Action |
|-----------|---------------|-------------------|
| 1.0 - 1.5 | Very Good | Maintain current data quality |
| 1.6 - 2.5 | Good | Monitor for improvements |
| 2.6 - 3.5 | Fair | Prioritize improvement plan |
| 3.6 - 4.5 | Poor | Active improvement required |
| 4.6 - 5.0 | Very Poor | Urgent data quality intervention |

### 2.8 Uncertainty Ranges

| Data Source | Typical DQI Range | Uncertainty Range | Confidence Level |
|-------------|------------------|------------------|-----------------|
| Supplier-specific WTT (verified EPD/MiQ) | 1.0 - 1.5 | +/- 10-20% | Very High |
| Supplier-specific WTT (unverified) | 1.5 - 2.5 | +/- 20-35% | High |
| DEFRA/DESNZ WTT factors | 2.0 - 2.5 | +/- 20-40% | High |
| EPA WTT factors | 2.0 - 3.0 | +/- 25-45% | Medium-High |
| ecoinvent LCA upstream factors | 2.0 - 3.0 | +/- 20-40% | Medium-High |
| IEA country-level factors | 2.5 - 3.5 | +/- 30-50% | Medium |
| T&D loss factors (country-specific) | 2.0 - 3.0 | +/- 15-30% | High |
| T&D loss factors (global default) | 3.5 - 4.5 | +/- 40-60% | Low |
| Grid upstream (location-based, eGRID/IEA) | 2.0 - 3.0 | +/- 20-40% | Medium-High |
| Grid upstream (market-based, supplier) | 1.5 - 2.5 | +/- 15-30% | High |
| Global average fallback | 4.0 - 5.0 | +/- 50-100% | Very Low |

**Pedigree matrix uncertainty factors:**

| DQI Score | Uncertainty Factor (sigma) |
|-----------|--------------------------|
| 1 | 1.00 (no additional uncertainty) |
| 2 | 1.05 (+/- 5% additional) |
| 3 | 1.10 (+/- 10% additional) |
| 4 | 1.20 (+/- 20% additional) |
| 5 | 1.50 (+/- 50% additional) |

**Combined uncertainty:**

```
Sigma_combined = sqrt(sigma_base^2 + sigma_temporal^2 + sigma_geo^2
                    + sigma_tech^2 + sigma_completeness^2 + sigma_reliability^2)
```

### 2.9 Category Boundaries & Double-Counting Prevention

Double-counting prevention is especially critical for Category 3 because
it sits at the precise boundary between Scope 1, Scope 2, and Scope 3.
The agent enforces the following rules:

**Included in Category 3:**

| Sub-Activity | What Is Included | Boundary |
|-------------|-----------------|----------|
| 3a | WTT emissions of fuels combusted in Scope 1 | Extraction through delivery; EXCLUDES combustion |
| 3b | Upstream emissions of fuels used to GENERATE purchased electricity/steam/heat/cooling | Fuel lifecycle of generation mix; EXCLUDES generation-point emissions |
| 3c | Emissions from electricity lost in T&D | Generation + upstream of lost electricity; EXCLUDES delivered electricity |
| 3d | Full lifecycle of purchased electricity sold to end users | Generation + upstream + T&D; utilities ONLY |

**Double-Counting Prevention Rules:**

| Rule | Scope Boundary | Enforcement |
|------|---------------|-------------|
| WTT factors EXCLUDE combustion emissions | vs Scope 1 | WTT EFs verified to exclude TTW component; validation check that WTT + TTW = WTW |
| Upstream factors EXCLUDE generation-point emissions | vs Scope 2 | Upstream EFs verified to exclude grid generation EF; sum check vs lifecycle total |
| Fuels consumed by reporting company -> Category 3 | vs Category 1 | Fuels for own combustion route to Cat 3 (not Cat 1 purchased goods) |
| Energy infrastructure (power plant, grid hardware) -> Category 2 | vs Category 2 | Energy infrastructure CapEx routes to Cat 2 (not Cat 3) |
| Fuel/energy purchased for resale -> Category 3d | vs Category 1 | Energy products for resale route to Cat 3d (utilities only) |
| T&D losses do not double with Scope 2 | vs Scope 2 | Scope 2 is metered consumption; T&D losses are ADDITIONAL electricity beyond meter |
| On-site generation fuel -> Scope 1 only | vs Category 3 | Self-generated electricity fuel is Scope 1 combustion; no Cat 3 upstream for own generation |

**Cross-category validation checks:**

```
Check 1: WTT + TTW = WTW (lifecycle completeness)
  For each fuel type, verify that:
  WTT_EF (Cat 3) + Combustion_EF (Scope 1) = WTW_EF (lifecycle total)
  Tolerance: +/- 5%

Check 2: Scope 2 + Cat 3 upstream < lifecycle
  For each grid region, verify that:
  Generation_EF (Scope 2) + Upstream_EF (Cat 3b) < Lifecycle_EF
  And: Upstream_EF / Lifecycle_EF is in expected range (8-20%)

Check 3: Fuel consumption match
  Total fuel consumed in Activity 3a = Total fuel in Scope 1
  Any mismatch flags a coverage gap or double-count risk

Check 4: Electricity consumption match
  Total electricity in Activity 3b + 3c = Total electricity in Scope 2
  Any mismatch flags a coverage gap or double-count risk
```

### 2.10 Coverage & Materiality

**Category 3 as percentage of total Scope 3 by sector:**

| Industry Sector | Cat 3 as % of Total S3 | Cat 3 as % of S1+S2+S3 | Primary Driver |
|----------------|----------------------|----------------------|----------------|
| Mining & metals | 15 - 25% | 8 - 15% | Heavy fuel use (Scope 1 upstream) |
| Oil & gas | 10 - 20% | 5 - 12% | Significant own fuel consumption |
| Utilities (power) | 5 - 15% | 3 - 10% | Fuel upstream for generation |
| Manufacturing (heavy) | 8 - 15% | 5 - 10% | Industrial fuel + grid electricity |
| Transportation & logistics | 10 - 20% | 6 - 12% | Fleet fuel upstream |
| Construction | 5 - 12% | 3 - 8% | Equipment fuel + site electricity |
| Retail | 5 - 12% | 3 - 8% | Store electricity T&D losses |
| Technology/Software | 8 - 15% | 4 - 8% | Data center electricity upstream |
| Healthcare | 5 - 10% | 3 - 6% | Facility heating + electricity |
| Financial services | 8 - 15% | 3 - 7% | Office electricity upstream |
| Food & beverage | 5 - 12% | 3 - 7% | Processing fuel + cold chain |
| Pharmaceuticals | 5 - 10% | 3 - 6% | Clean room energy |

**Coverage thresholds:**

| Level | Target | Description |
|-------|--------|-------------|
| Minimum viable | >= 90% of Scope 1 fuel + Scope 2 electricity covered | Required for credible reporting |
| Good practice | >= 95% of Scope 1 fuel + Scope 2 electricity covered | Recommended by CDP/SBTi |
| Best practice | 100% coverage with supplier-specific data for top fuels | Leading practice |

### 2.11 Emission Factor Selection Hierarchy

The agent implements a 6-level EF priority hierarchy for Category 3:

| Priority | Source | DQI Score | Applicability |
|----------|--------|-----------|--------------|
| 1 | Supplier-specific WTT/upstream data (verified EPD, MiQ, OGMP) | 1.0-1.5 | Activity 3a (fuel suppliers) |
| 2 | Supplier-specific utility upstream data (utility report, PPA) | 1.5-2.5 | Activity 3b (electricity suppliers) |
| 3 | DEFRA/DESNZ WTT factors (current year) | 2.0-2.5 | Activity 3a (UK-applicable, widely used globally) |
| 4 | EPA / ecoinvent upstream factors (current year) | 2.0-3.0 | Activities 3a/3b (US / global) |
| 5 | IEA country-level upstream factors | 2.5-3.5 | Activities 3b/3c (50+ countries) |
| 6 | IPCC Tier 1 default / global average | 4.0-5.0 | Fallback for all activities |

### 2.12 Key Formulas Summary

**Activity 3a (fuel upstream):**

```
Emissions_3a = SUM_i(Fuel_consumed_i * WTT_EF_i)
```

**Activity 3b (electricity upstream, location-based):**

```
Emissions_3b_loc = SUM_e(Energy_consumed_e * Upstream_EF_loc_e)
```

**Activity 3b (electricity upstream, market-based):**

```
Emissions_3b_mkt = SUM_e(Energy_consumed_e * Upstream_EF_mkt_e)
```

**Activity 3c (T&D losses):**

```
Emissions_3c = Elec_consumed * TD_loss_pct / (1 - TD_loss_pct) * (Gen_EF + Upstream_EF)
```

**Activity 3d (utility resale):**

```
Emissions_3d = Elec_sold * Lifecycle_EF
```

**Total Category 3:**

```
Emissions_cat3 = Emissions_3a + Emissions_3b + Emissions_3c + Emissions_3d
```

**Emissions intensity:**

```
Intensity_revenue   = Emissions_cat3 / Revenue_total       (tCO2e per $M)
Intensity_employee  = Emissions_cat3 / FTE_count            (tCO2e per FTE)
Intensity_energy    = Emissions_cat3 / Total_energy_consumed (tCO2e per MWh)
Intensity_fuel      = Emissions_3a / Total_fuel_consumed     (tCO2e per MWh fuel)
Intensity_elec      = (Emissions_3b + Emissions_3c) / Total_electricity (tCO2e per MWh elec)
```

**Year-over-year change decomposition:**

```
Delta_emissions = Delta_activity + Delta_EF + Delta_method + Delta_scope

Where:
  Delta_activity = (Energy_current - Energy_prior) * EF_prior
  Delta_EF       = Energy_current * (EF_current - EF_prior)
  Delta_method   = Emissions_current_new_method - Emissions_current_old_method
  Delta_scope    = Change from boundary or scope changes
```

**Weighted DQI for total inventory:**

```
DQI_weighted = SUM_i( (Emissions_i / Emissions_total) * DQI_i )
```

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
+-----------------------------------------------------------+
|                    AGENT-MRV-016                           |
|         Fuel & Energy Activities Agent                     |
|                                                            |
|  +------------------------------------------------------+ |
|  | Engine 1: WTTFuelDatabaseEngine                      | |
|  |   - WTT emission factors by fuel type (20+ fuels)   | |
|  |   - DEFRA/EPA/ecoinvent/IEA source databases         | |
|  |   - Per-gas breakdown (CO2, CH4, N2O)                | |
|  |   - Biofuel WTT factors (9+ biofuels)               | |
|  |   - Fuel heating values (HHV/LHV)                   | |
|  |   - Fuel density factors                             | |
|  |   - Unit conversion (litres, kg, m3, kWh, GJ, MMBTU)| |
|  |   - WTT supply chain stage decomposition             | |
|  |   - Fuel category classification mapping              | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 2: UpstreamFuelCalculatorEngine                | |
|  |   - Activity 3a: Fuel_consumed * WTT_EF              | |
|  |   - Per-gas breakdown (CO2, CH4, N2O -> CO2e)        | |
|  |   - GWP application (AR4/AR5/AR6)                    | |
|  |   - Biofuel upstream calculation                      | |
|  |   - Unit normalization and conversion                 | |
|  |   - Fuel blend handling (e.g., E10, B20)             | |
|  |   - Batch fuel consumption processing                 | |
|  |   - Integration with Scope 1 fuel records             | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 3: UpstreamElectricityCalculatorEngine         | |
|  |   - Activity 3b: Elec_consumed * Upstream_EF          | |
|  |   - Grid-mix-weighted upstream factor calculation     | |
|  |   - Location-based upstream (country/region grid)     | |
|  |   - Market-based upstream (supplier-specific)         | |
|  |   - Steam/heat/cooling upstream calculation           | |
|  |   - Dual reporting (location + market)                | |
|  |   - Integration with Scope 2 electricity records      | |
|  |   - Residual mix upstream factors                     | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 4: TDLossCalculatorEngine                      | |
|  |   - Activity 3c: Elec * TD_loss% * (Gen_EF + Up_EF) | |
|  |   - Country-specific T&D loss percentages (50+)      | |
|  |   - US eGRID subregional loss factors (26 subregions)| |
|  |   - Location-based T&D loss calculation               | |
|  |   - Market-based T&D loss calculation                 | |
|  |   - Voltage-level loss decomposition (optional)       | |
|  |   - Self-generation offset (no T&D for on-site gen)  | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 5: SupplierSpecificCalculatorEngine            | |
|  |   - Fuel supplier EPD/PCF/MiQ data integration       | |
|  |   - Utility supplier upstream factor integration      | |
|  |   - Product-level WTT emission factors                | |
|  |   - Supplier data quality validation                  | |
|  |   - OGMP 2.0 methane intensity data                  | |
|  |   - PPA/green tariff upstream factor resolution       | |
|  |   - Supplier engagement tracking                      | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 6: ComplianceCheckerEngine                     | |
|  |   - 7 frameworks: GHG Protocol Scope 3, CSRD/ESRS,  | |
|  |     CDP, SBTi, SB 253, GRI 305, ISO 14064           | |
|  |   - Sub-activity coverage validation (3a/3b/3c/3d)   | |
|  |   - DQI scoring validation                           | |
|  |   - Double-counting prevention verification          | |
|  |   - Scope 1/2 consistency checks                     | |
|  |   - Methodology documentation checks                 | |
|  |   - Dual reporting validation (3b location + market)  | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 7: FuelEnergyPipelineEngine                    | |
|  |   - 10-stage pipeline orchestration                   | |
|  |   - Batch multi-period processing                     | |
|  |   - Multi-facility aggregation                        | |
|  |   - Sub-activity breakdown (3a/3b/3c/3d)             | |
|  |   - Double-counting prevention enforcement            | |
|  |   - Export (JSON/CSV/Excel/PDF)                       | |
|  |   - Compliance-ready outputs (CDP, CSRD, SBTi)       | |
|  |   - Provenance chain assembly                         | |
|  +------------------------------------------------------+ |
+-----------------------------------------------------------+
```

### 3.2 Ten-Stage Pipeline

```
Stage 1: VALIDATE
  - Schema validation of fuel and electricity consumption records
  - Required field checks (fuel type, quantity, unit, reporting period)
  - Data type enforcement (Decimal for quantities and factors)
  - Duplicate detection (same fuel/facility/period)
  - Unit consistency validation (kWh, litres, kg, m3, GJ)

Stage 2: CLASSIFY
  - Fuel type classification (20+ fossil + 9+ biofuels)
  - Energy type classification (electricity, steam, heat, cooling)
  - Sub-activity assignment (3a, 3b, 3c, 3d)
  - Grid region resolution (country, eGRID subregion, custom)
  - Utility identification for Activity 3d applicability

Stage 3: NORMALIZE
  - Unit conversion to EF reference unit
  - Fuel heating value application (HHV vs LHV)
  - Fuel density conversion (litres to kg/kWh)
  - Blend decomposition (E10 -> 90% petrol + 10% ethanol)
  - Period normalization (monthly/quarterly/annual)

Stage 4: RESOLVE_EFS
  - 6-level EF hierarchy resolution for each fuel/energy type
  - WTT factor selection (Activity 3a)
  - Upstream electricity factor selection (Activity 3b)
  - T&D loss factor selection (Activity 3c)
  - Supplier-specific factor prioritization
  - EF source and version tracking
  - Missing EF flagging with fallback

Stage 5: CALCULATE_3A
  - Fuel_consumed * WTT_EF for all fuel types
  - Per-gas breakdown (CO2, CH4, N2O)
  - GWP application (AR4/AR5/AR6)
  - Biofuel upstream calculation
  - Blend-adjusted calculation

Stage 6: CALCULATE_3B
  - Electricity_consumed * Upstream_EF (location-based)
  - Electricity_consumed * Upstream_EF (market-based)
  - Steam/heat/cooling upstream calculation
  - Grid-mix-weighted factor application
  - Dual reporting output (location + market)

Stage 7: CALCULATE_3C
  - T&D loss calculation using country/region-specific loss %
  - Generation + upstream EF application to lost electricity
  - Location-based and market-based T&D outputs
  - Self-generation offset (no T&D for on-site gen)

Stage 8: COMPLIANCE
  - 7-framework compliance check
  - Sub-activity coverage validation
  - Double-counting prevention verification
  - Scope 1/2 consistency checks
  - Dual reporting completeness (3b)
  - Gap identification and recommendations

Stage 9: AGGREGATE
  - Total Category 3 emissions
  - By sub-activity (3a, 3b, 3c, 3d)
  - By fuel type (Activity 3a)
  - By energy type (Activity 3b)
  - By country/region (Activities 3b/3c)
  - By facility/entity
  - Intensity metrics (revenue, FTE, energy)
  - Year-over-year change decomposition

Stage 10: SEAL
  - SHA-256 provenance hash
  - Audit trail assembly
  - Export generation (JSON/CSV/Excel/PDF)
  - Result persistence
  - Provenance chain linking to Scope 1/2 source data
```

### 3.3 File Structure

```
greenlang/fuel_energy_activities/
+-- __init__.py                              # Lazy imports, module exports
+-- models.py                                # Pydantic v2 models, enums, constants
+-- config.py                                # GL_FEA_ prefixed configuration
+-- metrics.py                               # Prometheus metrics (gl_fea_*)
+-- provenance.py                            # SHA-256 provenance chain
+-- wtt_fuel_database.py                     # Engine 1: WTT factor database
+-- upstream_fuel_calculator.py              # Engine 2: Activity 3a calculation
+-- upstream_electricity_calculator.py       # Engine 3: Activity 3b calculation
+-- td_loss_calculator.py                    # Engine 4: Activity 3c calculation
+-- supplier_specific_calculator.py          # Engine 5: Supplier-specific data
+-- compliance_checker.py                    # Engine 6: Compliance checking
+-- fuel_energy_pipeline.py                  # Engine 7: Pipeline orchestration
+-- setup.py                                 # Service facade
+-- api/
    +-- __init__.py                          # API package
    +-- router.py                            # FastAPI REST endpoints

tests/unit/mrv/test_fuel_energy_activities/
+-- __init__.py
+-- conftest.py
+-- test_models.py
+-- test_config.py
+-- test_metrics.py
+-- test_provenance.py
+-- test_wtt_fuel_database.py
+-- test_upstream_fuel_calculator.py
+-- test_upstream_electricity_calculator.py
+-- test_td_loss_calculator.py
+-- test_supplier_specific_calculator.py
+-- test_compliance_checker.py
+-- test_fuel_energy_pipeline.py
+-- test_setup.py
+-- test_api.py

deployment/database/migrations/sql/
+-- V067__fuel_energy_activities_service.sql
```

### 3.4 Database Schema (V067)

16 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `gl_fea_fuel_types` | Fuel type taxonomy (20+ fossil, 9+ biofuel) with category, heating values, density | Seed (30+ rows) |
| `gl_fea_wtt_emission_factors` | WTT emission factors by fuel type, source database, per-gas breakdown | Seed (100+ rows: 30 fuels x 4 sources) |
| `gl_fea_upstream_electricity_factors` | Upstream electricity EFs by country/region, location and market-based | Seed (60+ rows: 30 countries x 2 methods) |
| `gl_fea_td_loss_factors` | T&D loss percentages by country and eGRID subregion | Seed (75+ rows: 50 countries + 25 eGRID) |
| `gl_fea_grid_regions` | Grid region definitions (country, state, eGRID subregion, custom) | Seed (100+ rows) |
| `gl_fea_fuel_consumption` | Fuel consumption records linked to Scope 1 data | Hypertable (partitioned by period_start) |
| `gl_fea_electricity_consumption` | Electricity/steam/heat/cooling consumption records linked to Scope 2 | Hypertable (partitioned by period_start) |
| `gl_fea_calculations` | Calculation results with sub-activity breakdown | Hypertable (partitioned by calculated_at) |
| `gl_fea_calculation_details` | Line-item detail per calculation (per fuel type, per energy type) | Regular |
| `gl_fea_activity_breakdown` | Emissions breakdown by sub-activity (3a, 3b, 3c, 3d) | Regular |
| `gl_fea_supplier_data` | Supplier-specific WTT and upstream factor records | Regular |
| `gl_fea_compliance_records` | Compliance check results (7 frameworks) | Regular |
| `gl_fea_dqi_scores` | Data quality scores per line item | Regular |
| `gl_fea_aggregations` | Aggregated results by dimension (fuel type, region, period) | Regular |
| `gl_fea_batch_jobs` | Batch processing jobs | Regular |
| `gl_fea_audit_entries` | Audit trail entries for all operations | Regular |
| `gl_fea_hourly_stats` | Hourly calculation statistics | Continuous Aggregate |
| `gl_fea_daily_stats` | Daily calculation statistics | Continuous Aggregate |

**Key seed data:**

- `gl_fea_fuel_types`: 30+ rows covering all fossil fuels (natural gas, diesel,
  petrol, coal types, fuel oils, LPG, kerosene, naphtha, propane, butane, CNG,
  LNG, petroleum coke) and biofuels (bioethanol corn/sugarcane, biodiesel
  rapeseed/soy, HVO, biogas, biomethane, wood pellets, wood chips)
- `gl_fea_wtt_emission_factors`: 100+ rows -- each fuel type with WTT EFs from
  multiple sources (DEFRA, EPA, ecoinvent, IEA), including per-gas CO2/CH4/N2O
  breakdown and aggregate CO2e
- `gl_fea_upstream_electricity_factors`: 60+ rows -- 30 countries with both
  location-based and market-based upstream factors, grid-mix-weighted
- `gl_fea_td_loss_factors`: 75+ rows -- 50 countries with national T&D loss
  percentages plus 25 US eGRID subregional factors
- `gl_fea_grid_regions`: 100+ rows -- grid region definitions with country code,
  region code (eGRID subregion, state, province), and generation mix composition

**Schema design principles:**

- Row-Level Security (RLS) on all tenant-facing tables via `tenant_id`
- TimescaleDB hypertables on `gl_fea_fuel_consumption`, `gl_fea_electricity_consumption`,
  and `gl_fea_calculations` (partitioned by temporal columns)
- Continuous aggregates for hourly and daily calculation statistics
- Foreign key relationships from `gl_fea_calculation_details` to `gl_fea_calculations`
- Foreign key relationships from `gl_fea_fuel_consumption` to `gl_fea_fuel_types`
- GIN indexes on JSONB columns for metadata queries
- B-tree indexes on fuel_type_id, country_code, grid_region_code, tenant_id
- Partial indexes for active/current records
- Composite indexes on (tenant_id, reporting_period, fuel_type_id) for fast lookups

### 3.5 API Endpoints (20)

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/calculate` | Run single calculation (specify sub-activities) |
| 2 | POST | `/calculate/batch` | Batch multi-period calculation |
| 3 | GET | `/calculations` | List calculations with filtering |
| 4 | GET | `/calculations/{calculation_id}` | Get calculation result with sub-activity breakdown |
| 5 | DELETE | `/calculations/{calculation_id}` | Delete a calculation |
| 6 | POST | `/fuel-consumption` | Create or update fuel consumption record |
| 7 | GET | `/fuel-consumption` | List fuel consumption records with filtering |
| 8 | PUT | `/fuel-consumption/{record_id}` | Update a fuel consumption record |
| 9 | POST | `/electricity-consumption` | Create or update electricity consumption record |
| 10 | GET | `/electricity-consumption` | List electricity consumption records with filtering |
| 11 | PUT | `/electricity-consumption/{record_id}` | Update an electricity consumption record |
| 12 | GET | `/emission-factors` | List WTT and upstream EFs with filtering |
| 13 | POST | `/emission-factors/custom` | Register custom WTT or upstream emission factor |
| 14 | GET | `/td-loss-factors` | List T&D loss factors by country/region |
| 15 | POST | `/td-loss-factors/custom` | Register custom T&D loss factor |
| 16 | POST | `/compliance/check` | Run compliance check (all 7 frameworks) |
| 17 | GET | `/compliance/{calculation_id}` | Get compliance results for a calculation |
| 18 | POST | `/uncertainty` | Run uncertainty analysis on a calculation |
| 19 | GET | `/aggregations` | Get aggregated results (by sub-activity, fuel, region, period) |
| 20 | GET | `/health` | Health check and service statistics |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All emission calculations use Python `Decimal` (8 decimal places)
- No LLM calls in any calculation path -- deterministic lookups only
- Every calculation step recorded in a provenance trace
- SHA-256 provenance hash for every emission result
- Bit-perfect reproducibility: same input always produces same output
- WTT factor lookup is exact-match by fuel type, source database, and version year
- Upstream electricity factor lookup is exact-match by country/region and method (location/market)
- T&D loss factor lookup is exact-match by country and eGRID subregion code
- GWP values are deterministic lookup by assessment report version (AR4/AR5/AR6)
- Unit conversion uses fixed conversion factors (no floating-point arithmetic in emission path)
- Fuel blend decomposition uses exact percentage splits
- Double-counting checks use deterministic boundary rules, not heuristic inference

### 4.2 Enumerations (22)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | WTT_DEFAULT, SUPPLIER_SPECIFIC, HYBRID | 3 calculation approach methods |
| `FuelType` | NATURAL_GAS, DIESEL, PETROL, COAL_BITUMINOUS, COAL_SUB_BITUMINOUS, COAL_ANTHRACITE, FUEL_OIL_RESIDUAL, FUEL_OIL_DISTILLATE, LPG, KEROSENE, NAPHTHA, PROPANE, BUTANE, CNG, LNG, PETROLEUM_COKE, BIOETHANOL_CORN, BIOETHANOL_SUGARCANE, BIODIESEL_RAPESEED, BIODIESEL_SOY, HVO, BIOGAS, BIOMETHANE, WOOD_PELLETS, WOOD_CHIPS | 25 fuel types |
| `FuelCategory` | GASEOUS_FOSSIL, LIQUID_FOSSIL, SOLID_FOSSIL, BIOFUEL_LIQUID, BIOFUEL_GASEOUS, BIOFUEL_SOLID | 6 fuel categories |
| `EnergyType` | ELECTRICITY, STEAM, HEAT, COOLING | 4 purchased energy types |
| `ActivityType` | ACTIVITY_3A, ACTIVITY_3B, ACTIVITY_3C, ACTIVITY_3D | 4 sub-activities |
| `WTTFactorSource` | DEFRA_2025, DEFRA_2024, EPA_2024, ECOINVENT_V311, IEA_2023, IPCC_EFDB, CUSTOM | 7 WTT factor source databases |
| `GridRegionType` | COUNTRY, STATE, EGRID_SUBREGION, CUSTOM, RESIDUAL_MIX | 5 grid region types |
| `TDLossSource` | IEA, WORLD_BANK, EIA, EGRID, DEFRA, CUSTOM | 6 T&D loss data sources |
| `SupplierDataSource` | EPD, PCF, MIQ_CERTIFICATE, OGMP_20, CDP_SUPPLY_CHAIN, UTILITY_REPORT, PPA_CONTRACT, GREEN_TARIFF, DIRECT_MEASUREMENT, ESTIMATED | 10 supplier data sources |
| `AllocationMethod` | ENERGY, ECONOMIC, PHYSICAL, MASS, HYBRID | 5 allocation methods |
| `CurrencyCode` | USD, EUR, GBP, JPY, CNY, CHF, CAD, AUD, KRW, INR, BRL, MXN, SEK, NOK, DKK, SGD, HKD, NZD, ZAR, AED | 20 currencies |
| `DQIDimension` | TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL, COMPLETENESS, RELIABILITY | 5 DQI dimensions |
| `DQIScore` | VERY_GOOD (1), GOOD (2), FAIR (3), POOR (4), VERY_POOR (5) | 5 quality scores |
| `UncertaintyMethod` | MONTE_CARLO, ANALYTICAL, TIER_DEFAULT | 3 uncertainty methods |
| `ComplianceFramework` | GHG_PROTOCOL_SCOPE3, CSRD_ESRS_E1, CDP, SBTI, SB_253, GRI_305, ISO_14064 | 7 regulatory frameworks |
| `ComplianceStatus` | COMPLIANT, PARTIAL, NON_COMPLIANT | 3 compliance statuses |
| `PipelineStage` | VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE_3A, CALCULATE_3B, CALCULATE_3C, COMPLIANCE, AGGREGATE, SEAL | 10 pipeline stages |
| `ExportFormat` | JSON, CSV, XLSX, PDF | 4 export formats |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED | 4 batch statuses |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | 4 GWP assessment report versions |
| `EmissionGas` | CO2, CH4, N2O, CO2E | 4 emission gases |
| `AccountingMethod` | LOCATION_BASED, MARKET_BASED | 2 Scope 2 dual-reporting methods |

### 4.3 Models (25)

| Model | Description | Key Fields |
|-------|-------------|------------|
| `FuelTypeRecord` | Fuel type definition with properties | fuel_type_id, name, category, heating_value_hhv, heating_value_lhv, density, unit_reference |
| `FuelConsumptionRecord` | Fuel consumption input for Activity 3a | record_id, facility_id, fuel_type, quantity, unit, period_start, period_end, scope1_reference |
| `ElectricityConsumptionRecord` | Electricity/steam/heat/cooling input for Activities 3b/3c | record_id, facility_id, energy_type, quantity_kwh, grid_region, accounting_method, scope2_reference |
| `WTTEmissionFactor` | WTT emission factor record | ef_id, fuel_type, ef_co2e_per_kwh, ef_co2_per_kwh, ef_ch4_per_kwh, ef_n2o_per_kwh, source, source_year |
| `UpstreamElectricityFactor` | Upstream electricity EF by country/region | ef_id, country_code, grid_region, ef_upstream_per_kwh, accounting_method, source, source_year, grid_mix |
| `TDLossFactor` | T&D loss percentage by country/region | factor_id, country_code, grid_region, loss_percentage, source, source_year |
| `SupplierWTTData` | Supplier-specific WTT data for fuels | record_id, supplier_id, supplier_name, fuel_type, wtt_ef, data_source, verification_status, valid_from, valid_to |
| `SupplierUpstreamData` | Supplier-specific upstream data for electricity | record_id, supplier_id, supplier_name, upstream_ef, data_source, contract_type, valid_from, valid_to |
| `GridMixComposition` | Grid generation mix by fuel source | mix_id, country_code, grid_region, gas_pct, coal_pct, nuclear_pct, hydro_pct, wind_pct, solar_pct, biomass_pct, oil_pct, other_pct, source_year |
| `Activity3aResult` | Result from upstream fuel (WTT) calculation | result_id, fuel_type, fuel_consumed, wtt_ef, emissions_co2, emissions_ch4, emissions_n2o, emissions_co2e, ef_source |
| `Activity3bResult` | Result from upstream electricity calculation | result_id, energy_type, quantity_kwh, upstream_ef, emissions_co2e, accounting_method, grid_region, ef_source |
| `Activity3cResult` | Result from T&D loss calculation | result_id, electricity_kwh, td_loss_pct, generation_ef, upstream_ef, lost_kwh, emissions_co2e, grid_region |
| `Activity3dResult` | Result from utility resale calculation | result_id, electricity_sold_kwh, lifecycle_ef, emissions_co2e |
| `CategoryResult` | Combined Category 3 result | result_id, emissions_3a, emissions_3b_location, emissions_3b_market, emissions_3c_location, emissions_3c_market, emissions_3d, total_location, total_market |
| `DQIAssessment` | Data quality assessment across 5 dimensions | assessment_id, temporal, geographical, technological, completeness, reliability, composite_score, classification |
| `UncertaintyResult` | Uncertainty analysis result | result_id, central_estimate, lower_bound, upper_bound, confidence_level, method, sigma_combined |
| `ComplianceRequirement` | Framework-specific compliance requirement | requirement_id, framework, requirement_code, description, is_mandatory, data_points_required |
| `ComplianceCheckResult` | Compliance check result per framework | result_id, framework, status, score, findings, gaps, recommendations |
| `AggregationResult` | Aggregated emissions by dimension | aggregation_id, dimension, breakdowns, total_emissions_location, total_emissions_market, intensity_metrics |
| `FuelBlendDefinition` | Fuel blend composition | blend_id, blend_name, components (list of fuel_type + percentage) |
| `CalculationRequest` | API request to run a calculation | sub_activities, fuel_records, electricity_records, reporting_year, organization_id, gwp_source, options |
| `BatchRequest` | Batch calculation request | batch_id, requests, reporting_years, options |
| `CalculationResult` | Complete calculation output | calculation_id, emissions_by_activity, total_co2e, dqi_score, provenance_hash, processing_time_ms, created_at |
| `CoverageReport` | Coverage analysis across sub-activities | report_id, coverage_3a, coverage_3b, coverage_3c, coverage_3d, overall_coverage, gaps |
| `ProvenanceRecord` | Provenance chain entry linking to Scope 1/2 sources | provenance_id, calculation_id, scope1_source_ids, scope2_source_ids, ef_versions, hash_sha256 |

### 4.4 Constant Tables (13)

#### 4.4.1 GWP_VALUES

| Gas | AR4 (100yr) | AR5 (100yr) | AR6 (100yr) | AR6 (20yr) |
|-----|-------------|-------------|-------------|------------|
| CO2 | 1 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 27.9 | 82.5 |
| N2O | 298 | 265 | 273 | 273 |

#### 4.4.2 WTT_FUEL_EMISSION_FACTORS

Complete WTT emission factor table with per-gas breakdown (see Section
2.2.3 for full factor ranges). Each entry contains:

| Column | Type | Description |
|--------|------|-------------|
| fuel_type | FuelType | Fuel type enum value |
| ef_co2e_per_kwh | Decimal | Total CO2e WTT factor (kgCO2e/kWh) |
| ef_co2_per_kwh | Decimal | CO2 component (kgCO2/kWh) |
| ef_ch4_per_kwh | Decimal | CH4 component (kgCH4/kWh) |
| ef_n2o_per_kwh | Decimal | N2O component (kgN2O/kWh) |
| ef_co2e_per_litre | Decimal | Alternative unit (kgCO2e/litre, for liquid fuels) |
| ef_co2e_per_kg | Decimal | Alternative unit (kgCO2e/kg, for solid fuels) |
| ef_co2e_per_m3 | Decimal | Alternative unit (kgCO2e/m3, for gaseous fuels) |
| source | WTTFactorSource | Source database |
| source_year | int | Factor year |

**Representative values (DEFRA 2025 source):**

| Fuel Type | CO2e (kgCO2e/kWh) | CO2 (kgCO2/kWh) | CH4 (kgCH4/kWh) | N2O (kgN2O/kWh) |
|-----------|-------------------|-----------------|-----------------|-----------------|
| NATURAL_GAS | 0.03108 | 0.02519 | 0.00019 | 0.00000020 |
| DIESEL | 0.05826 | 0.05261 | 0.00017 | 0.00000120 |
| PETROL | 0.05558 | 0.05024 | 0.00016 | 0.00000110 |
| COAL_BITUMINOUS | 0.03951 | 0.03654 | 0.00088 | 0.00000050 |
| LPG | 0.03422 | 0.03098 | 0.00010 | 0.00000008 |
| KEROSENE | 0.05402 | 0.04881 | 0.00016 | 0.00000120 |
| FUEL_OIL_RESIDUAL | 0.05149 | 0.04650 | 0.00015 | 0.00000100 |
| LNG | 0.05500 | 0.03850 | 0.00054 | 0.00000030 |

#### 4.4.3 UPSTREAM_ELECTRICITY_FACTORS

Country-level upstream electricity emission factors (see Section 2.3.3 for
full table). Key structure:

| Column | Type | Description |
|--------|------|-------------|
| country_code | str | ISO 3166-1 alpha-2 |
| upstream_ef_location | Decimal | Location-based upstream EF (kgCO2e/kWh) |
| upstream_ef_market | Decimal | Market-based upstream EF (kgCO2e/kWh, if available) |
| generation_ef_location | Decimal | Generation-point EF for cross-check (kgCO2e/kWh) |
| source | str | IEA, eGRID, DEFRA, or utility |
| source_year | int | Factor year |

#### 4.4.4 TD_LOSS_FACTORS

T&D loss percentages by country and subregion (see Section 2.4.3 for full
table). Key structure:

| Column | Type | Description |
|--------|------|-------------|
| country_code | str | ISO 3166-1 alpha-2 |
| region_code | str | eGRID subregion or state code (optional) |
| loss_percentage | Decimal | T&D loss as decimal (e.g., 0.058 for 5.8%) |
| source | TDLossSource | IEA, World Bank, EIA, eGRID, DEFRA |
| source_year | int | Factor year |
| voltage_level | str | HIGH, MEDIUM, LOW, ALL (optional decomposition) |

#### 4.4.5 GRID_MIX_COMPOSITIONS

Grid generation mix by fuel source for 30+ countries. Used to derive
grid-mix-weighted upstream factors.

| Column | Type | Description |
|--------|------|-------------|
| country_code | str | ISO 3166-1 alpha-2 |
| region_code | str | Subregion code (optional) |
| gas_pct | Decimal | Natural gas share (%) |
| coal_pct | Decimal | Coal share (%) |
| nuclear_pct | Decimal | Nuclear share (%) |
| hydro_pct | Decimal | Hydroelectric share (%) |
| wind_pct | Decimal | Wind share (%) |
| solar_pct | Decimal | Solar share (%) |
| biomass_pct | Decimal | Biomass share (%) |
| oil_pct | Decimal | Oil/petroleum share (%) |
| other_pct | Decimal | Other sources share (%) |
| source_year | int | Mix year |

#### 4.4.6 FUEL_UNIT_CONVERSIONS

| From Unit | To Unit | Fuel Type | Conversion Factor | Notes |
|-----------|---------|-----------|------------------|-------|
| litres | kWh | Diesel | 10.68 | Based on density 0.832 kg/L, NCV 43.33 MJ/kg |
| litres | kWh | Petrol | 9.48 | Based on density 0.749 kg/L, NCV 44.75 MJ/kg |
| litres | kWh | Kerosene | 10.37 | Based on density 0.800 kg/L, NCV 43.80 MJ/kg |
| litres | kWh | LPG (liquid) | 7.11 | Based on density 0.510 kg/L, NCV 46.00 MJ/kg |
| litres | kWh | Fuel Oil | 11.19 | Based on density 0.940 kg/L, NCV 40.40 MJ/kg |
| kg | kWh | Natural Gas | 14.89 | Based on NCV 48.00 MJ/kg |
| kg | kWh | Coal (bituminous) | 7.33 | Based on NCV 26.40 MJ/kg |
| kg | kWh | Coal (sub-bituminous) | 5.28 | Based on NCV 19.00 MJ/kg |
| kg | kWh | Coal (anthracite) | 7.56 | Based on NCV 27.20 MJ/kg |
| kg | kWh | Wood pellets | 4.81 | Based on NCV 17.30 MJ/kg |
| m3 | kWh | Natural Gas | 10.55 | Based on density 0.714 kg/m3, NCV 48.00 MJ/kg |
| m3 | kWh | Biogas | 6.50 | ~60% methane content |
| GJ | kWh | All | 277.78 | 1 GJ = 277.78 kWh |
| MMBTU | kWh | All | 293.07 | 1 MMBTU = 293.07 kWh |
| therms | kWh | All | 29.31 | 1 therm = 29.31 kWh |
| MJ | kWh | All | 0.2778 | 1 MJ = 0.2778 kWh |

#### 4.4.7 DQI_SCORE_VALUES

| Score | Label | Numeric | Description |
|-------|-------|---------|-------------|
| 1 | Very Good | 1.0 | Best available; third-party verified supplier data |
| 2 | Good | 2.0 | Established databases (DEFRA/EPA); same region |
| 3 | Fair | 3.0 | Related fuel category; within 6 years |
| 4 | Poor | 4.0 | Proxy data; different region or fuel category |
| 5 | Very Poor | 5.0 | Estimate or assumption; >10 years old |

#### 4.4.8 UNCERTAINTY_RANGES

| Data Source | Lower Bound (%) | Upper Bound (%) | Default Sigma |
|-------------|-----------------|-----------------|---------------|
| Supplier-specific WTT (verified EPD/MiQ) | 10 | 20 | 0.15 |
| Supplier-specific WTT (unverified) | 20 | 35 | 0.27 |
| DEFRA/DESNZ WTT factors | 20 | 40 | 0.30 |
| EPA upstream factors | 25 | 45 | 0.35 |
| ecoinvent LCA upstream | 20 | 40 | 0.30 |
| IEA country-level factors | 30 | 50 | 0.40 |
| T&D loss (country-specific IEA) | 15 | 30 | 0.22 |
| T&D loss (global default) | 40 | 60 | 0.50 |
| Global average fallback | 50 | 100 | 0.75 |

#### 4.4.9 COVERAGE_THRESHOLDS

| Level | Minimum (%) | Label | Description |
|-------|-------------|-------|-------------|
| MINIMUM | 90 | Minimum Viable | >= 90% of Scope 1 fuel + Scope 2 electricity |
| GOOD | 95 | Good Practice | Recommended by CDP/SBTi |
| BEST | 100 | Best Practice | Full coverage with supplier-specific for top fuels |

#### 4.4.10 EF_HIERARCHY_PRIORITY

| Priority | Source Type | DQI Range | Description |
|----------|-----------|-----------|-------------|
| 1 | SUPPLIER_VERIFIED | 1.0-1.5 | Verified supplier EPD, MiQ, OGMP 2.0 |
| 2 | SUPPLIER_UNVERIFIED | 1.5-2.5 | Unverified supplier data, utility reports |
| 3 | DEFRA_DESNZ | 2.0-2.5 | DEFRA/DESNZ annual conversion factors |
| 4 | EPA_ECOINVENT | 2.0-3.0 | EPA Emission Factors Hub, ecoinvent LCA |
| 5 | IEA_COUNTRY | 2.5-3.5 | IEA country-level upstream factors |
| 6 | GLOBAL_DEFAULT | 4.0-5.0 | IPCC Tier 1 / global average fallback |

#### 4.4.11 FUEL_HEATING_VALUES

| Fuel Type | NCV / LHV (MJ/kg) | GCV / HHV (MJ/kg) | Density (kg/L or kg/m3) |
|-----------|-------------------|-------------------|------------------------|
| Natural Gas | 48.00 | 53.10 | 0.714 kg/m3 |
| Diesel | 43.33 | 45.77 | 0.832 kg/L |
| Petrol | 44.75 | 47.31 | 0.749 kg/L |
| Kerosene | 43.80 | 46.20 | 0.800 kg/L |
| LPG | 46.00 | 49.32 | 0.510 kg/L |
| Fuel Oil (residual) | 40.40 | 42.50 | 0.940 kg/L |
| Fuel Oil (distillate) | 42.79 | 45.17 | 0.876 kg/L |
| Coal (bituminous) | 26.40 | 27.20 | N/A (solid) |
| Coal (sub-bituminous) | 19.00 | 19.80 | N/A (solid) |
| Coal (anthracite) | 27.20 | 28.00 | N/A (solid) |
| Propane | 46.35 | 50.33 | 0.493 kg/L |
| Butane | 45.72 | 49.51 | 0.573 kg/L |
| Naphtha | 44.50 | 47.00 | 0.720 kg/L |
| Petroleum Coke | 32.50 | 33.60 | N/A (solid) |
| Wood Pellets | 17.30 | 18.50 | N/A (solid) |
| Wood Chips | 9.40 | 10.20 | N/A (solid) |

#### 4.4.12 FUEL_DENSITY_FACTORS

| Fuel Type | Density | Unit | Temperature Basis |
|-----------|---------|------|------------------|
| Diesel | 0.832 | kg/L | 15 C |
| Petrol | 0.749 | kg/L | 15 C |
| Kerosene / Jet Fuel | 0.800 | kg/L | 15 C |
| LPG (liquid) | 0.510 | kg/L | 15 C |
| Fuel Oil (residual) | 0.940 | kg/L | 15 C |
| Fuel Oil (distillate) | 0.876 | kg/L | 15 C |
| Propane (liquid) | 0.493 | kg/L | 15 C |
| Butane (liquid) | 0.573 | kg/L | 15 C |
| Naphtha | 0.720 | kg/L | 15 C |
| Natural Gas | 0.714 | kg/m3 | 15 C, 1 atm |
| Biogas | 1.150 | kg/m3 | 15 C, 1 atm (60% CH4) |
| Biomethane | 0.668 | kg/m3 | 15 C, 1 atm (97% CH4) |
| Bioethanol | 0.789 | kg/L | 15 C |
| Biodiesel (FAME) | 0.880 | kg/L | 15 C |
| HVO | 0.780 | kg/L | 15 C |

#### 4.4.13 FRAMEWORK_REQUIRED_DISCLOSURES

| Framework | Required Disclosures for Category 3 |
|-----------|-------------------------------------|
| GHG Protocol Scope 3 | Total Cat 3 emissions (tCO2e), breakdown by sub-activity (3a/3b/3c/3d), methodology per sub-activity, WTT factor sources, T&D loss sources, dual reporting for 3b (location + market), data quality assessment, double-counting prevention documentation |
| CSRD/ESRS E1 | E1-6 para 44: Cat 3 gross emissions with sub-activity breakdown, methodology, % supplier-specific, upstream factor sources; para 46: intensity metrics; para 48: value chain engagement for fuel/energy suppliers |
| CDP | C6.5: Cat 3 relevance, emissions figure by sub-activity, methodology, % calculated using primary (supplier-specific) data, emissions intensity, explanation of YoY changes, dual reporting for upstream electricity |
| SBTi | Scope 3 screening results, Cat 3 significance assessment, data quality improvement plan, supplier engagement targets for fuel/energy suppliers, coverage validation |
| SB 253 | Total Cat 3 in tCO2e with sub-activity breakdown, methodology description, WTT and upstream data sources, safe harbor attestation (2027-2030), third-party assurance timeline |
| GRI 305 | Scope 3 Cat 3 emissions if significant, methodology and EF sources, base year and recalculation policy, sub-activity disclosure |
| ISO 14064 | Category 4 indirect emissions including upstream fuel/energy, methodology selection and justification, uncertainty quantification per sub-activity, verification statement, comparison with prior periods |

### 4.5 Regulatory Frameworks (7)

1. **GHG Protocol Scope 3 Standard** -- Chapter 5 Category 3 definition,
   explicit requirement to report sub-activities (3a/3b/3c) separately
   where feasible, dual reporting for Activity 3b mirroring Scope 2 dual
   method, Chapter 7 DQI, Chapter 9 reporting requirements.
2. **CSRD/ESRS E1** -- E1-6 para 44a/44b/44c Scope 3 by category,
   methodology, data sources; para 46 intensity metrics; para 48 value
   chain engagement. Category 3 must be reported with sub-activity
   breakdown.
3. **California SB 253** -- Scope 3 mandatory by FY2027 for >$1B revenue
   entities; GHG Protocol methodology; safe harbor 2027-2030; up to
   $500K penalty. Category 3 must be reported as it is almost universally
   material.
4. **CDP Climate Change** -- C6.5 Category 3 relevance assessment and
   calculation; sub-activity methodology; supplier-specific data
   percentage; year-over-year explanation for fluctuations.
5. **SBTi v5.3** -- Scope 3 target required if >40% of total; 67%
   coverage; supplier engagement targets specifically for energy
   suppliers; near-term 5-10 years.
6. **GRI 305** -- Scope 3 disclosure if significant; methodology and EF
   sources; base year and recalculation policy.
7. **ISO 14064-1:2018** -- Category 4 indirect GHG emissions; methodology-
   neutral; uncertainty quantification per sub-activity; third-party
   verification.

### 4.6 Performance Targets

| Metric | Target |
|--------|--------|
| Activity 3a calculation (100 fuel records) | < 100ms |
| Activity 3a calculation (1,000 fuel records) | < 500ms |
| Activity 3b calculation (100 electricity records, dual) | < 150ms |
| Activity 3b calculation (1,000 electricity records, dual) | < 800ms |
| Activity 3c T&D loss calculation (100 records) | < 100ms |
| Activity 3c T&D loss calculation (1,000 records) | < 500ms |
| Supplier-specific WTT lookup (single fuel) | < 5ms |
| WTT factor resolution (6-level hierarchy, single fuel) | < 10ms |
| T&D loss factor lookup (single country/region) | < 5ms |
| Upstream electricity factor lookup (single country, dual) | < 10ms |
| DQI scoring (full inventory) | < 200ms |
| Compliance check (all 7 frameworks) | < 300ms |
| Unit conversion (single fuel record) | < 1ms |
| Full pipeline (1,000 fuel + 500 electricity, all activities) | < 5s |
| Full pipeline (5,000 fuel + 2,000 electricity, all activities) | < 15s |
| Double-counting cross-check vs Scope 1/2 | < 200ms |

---

## 5. Acceptance Criteria

### 5.1 Core Calculation -- Activity 3a (Upstream Fuel Emissions)

- [ ] WTT calculation for all 16 fossil fuel types using DEFRA factors
- [ ] WTT calculation for all 9 biofuel types using DEFRA/EPA factors
- [ ] Per-gas breakdown (CO2, CH4, N2O) with GWP application
- [ ] GWP selection (AR4/AR5/AR6/AR6-20yr) with deterministic lookup
- [ ] Unit conversion from litres, kg, m3, GJ, MMBTU, therms to kWh
- [ ] Fuel density and heating value application (HHV/LHV selectable)
- [ ] Fuel blend decomposition (e.g., E10 = 90% petrol + 10% bioethanol)
- [ ] Batch processing of 1,000+ fuel consumption records
- [ ] Integration with Scope 1 fuel consumption data (cross-reference IDs)
- [ ] Supplier-specific WTT factor override (EPD, MiQ, OGMP 2.0)

### 5.2 Core Calculation -- Activity 3b (Upstream Electricity Emissions)

- [ ] Location-based upstream calculation using country/region grid-mix-weighted factors
- [ ] Market-based upstream calculation using supplier-specific or residual mix factors
- [ ] Dual reporting output (both location and market) for every electricity record
- [ ] Steam, heat, and cooling upstream calculation
- [ ] Upstream factors for 30+ countries with IEA/DEFRA/eGRID sources
- [ ] Grid mix composition tracking for upstream factor derivation
- [ ] Supplier-specific utility upstream factor integration
- [ ] PPA/green tariff upstream factor resolution

### 5.3 Core Calculation -- Activity 3c (T&D Losses)

- [ ] T&D loss calculation using country-specific loss percentages (50+ countries)
- [ ] T&D loss calculation using US eGRID subregional factors (26 subregions)
- [ ] Generation + upstream EF application to lost electricity
- [ ] Correct formula: `Elec * TD% / (1 - TD%) * (Gen_EF + Upstream_EF)`
- [ ] Location-based and market-based T&D loss outputs
- [ ] Self-generation offset (exclude on-site generated electricity from T&D)
- [ ] Custom T&D loss factor support for organization-specific data

### 5.4 Core Calculation -- Activity 3d (Utility Resale)

- [ ] Applicability filter (utility/energy retailer only flag)
- [ ] Lifecycle EF application to electricity sold to end users
- [ ] Exclusion of self-generated electricity from Activity 3d
- [ ] Skip/zero output for non-utility organizations

### 5.5 WTT Factor Database

- [ ] 25+ fuel types with WTT emission factors from 4+ source databases
- [ ] Per-gas (CO2, CH4, N2O) and aggregate CO2e factors per fuel
- [ ] Multi-unit WTT factors (kgCO2e/kWh, kgCO2e/litre, kgCO2e/kg, kgCO2e/m3)
- [ ] Fuel category classification (6 categories)
- [ ] Fuel heating values (NCV/LHV and GCV/HHV) for 16+ fuels
- [ ] Fuel density factors for 15+ fuels
- [ ] Unit conversion table (15+ conversion factors)
- [ ] Factor versioning (source database, source year, update tracking)
- [ ] Custom WTT factor registration via API
- [ ] 6-level EF selection hierarchy with automatic fallback

### 5.6 Double-Counting Prevention

- [ ] WTT factors verified to EXCLUDE combustion emissions (vs Scope 1)
- [ ] Upstream factors verified to EXCLUDE generation-point emissions (vs Scope 2)
- [ ] WTT + TTW = WTW lifecycle completeness cross-check (+/- 5% tolerance)
- [ ] Scope 2 + Cat 3 upstream < lifecycle total cross-check
- [ ] Fuel consumption match: Activity 3a total = Scope 1 fuel total
- [ ] Electricity consumption match: Activity 3b+3c total = Scope 2 electricity total
- [ ] Fuel for own combustion routed to Cat 3 (not Cat 1)
- [ ] Energy infrastructure CapEx routed to Cat 2 (not Cat 3)
- [ ] On-site generation fuel excluded from Activity 3a (Scope 1 only)

### 5.7 Data Quality

- [ ] 5-dimension DQI scoring (temporal, geographical, technological, completeness, reliability)
- [ ] Composite DQI score (1.0-5.0 scale, arithmetic mean)
- [ ] Quality classification (Very Good through Very Poor)
- [ ] Pedigree matrix uncertainty quantification
- [ ] Weighted DQI for total inventory (emission-weighted)
- [ ] DQI improvement recommendations per fuel type and grid region
- [ ] Uncertainty analysis (Monte Carlo, analytical, tier default)

### 5.8 Compliance

- [ ] 7 regulatory framework compliance checks
- [ ] GHG Protocol sub-activity breakdown requirement (3a/3b/3c/3d)
- [ ] GHG Protocol dual reporting for Activity 3b (location + market)
- [ ] CSRD/ESRS E1 data point coverage (para 44a/44b/44c)
- [ ] SB 253 methodology and coverage validation
- [ ] CDP C6.5 scoring criteria alignment
- [ ] SBTi coverage threshold validation (67% of Scope 3)
- [ ] GRI 305 and ISO 14064 compliance flags
- [ ] Double-counting prevention documentation generation

### 5.9 Infrastructure

- [ ] 20 REST API endpoints at `/api/v1/fuel-energy-activities`
- [ ] V067 database migration (16 tables, 3 hypertables, 2 continuous aggregates)
- [ ] SHA-256 provenance on every calculation result
- [ ] Prometheus metrics with `gl_fea_` prefix
- [ ] Auth integration (route_protector.py + auth_setup.py with fea_router)
- [ ] 20 PERMISSION_MAP entries for fuel-energy-activities
- [ ] 1,000+ unit tests
- [ ] All calculations use Python `Decimal` (no floating point in emission path)
- [ ] Export in JSON, CSV, Excel, and PDF formats
- [ ] Row-Level Security (RLS) on all tenant-facing tables
- [ ] Provenance chain linking to Scope 1 and Scope 2 source data

---

## 6. Key Differentiators from Adjacent Categories

This section documents the boundaries between Category 3 and related
Scope/Category boundaries to prevent confusion during implementation.

### 6.1 Category 3 vs Scope 1

| Aspect | Scope 1 (Direct) | Category 3 Activity 3a (Upstream) |
|--------|-----------------|----------------------------------|
| Emission source | Combustion of fuel at reporting company | Extraction, refining, transport of fuel BEFORE combustion |
| Factor type | Combustion EF (TTW) | WTT EF (upstream only) |
| Lifecycle stage | Tank-to-Wheel | Well-to-Tank |
| Gases | CO2 (dominant), CH4, N2O | CO2, CH4 (often significant for gas), N2O |
| Agent | AGENT-MRV-001 through 005 | AGENT-MRV-016 (this agent) |
| Relationship | Same fuel, same quantity | Same fuel, same quantity, DIFFERENT EF |

### 6.2 Category 3 vs Scope 2

| Aspect | Scope 2 (Purchased Energy) | Category 3 Activity 3b (Upstream) |
|--------|--------------------------|----------------------------------|
| Emission source | Generation of purchased electricity at power plant | Upstream fuel lifecycle for power generation |
| Factor type | Grid EF (generation-point) | Upstream EF (WTT of generation fuels) |
| Typical magnitude | 100% of generation emissions | 8-20% additional (upstream) |
| Dual reporting | Location + market methods | Location + market methods (mirrors Scope 2) |
| Agent | AGENT-MRV-009 through 013 | AGENT-MRV-016 (this agent) |

### 6.3 Category 3 vs Category 1

| Aspect | Category 1 (Purchased Goods) | Category 3 (Fuel & Energy) |
|--------|-----------------------------|-----------------------------|
| What is covered | Cradle-to-gate of non-energy purchased goods | WTT of fuels and upstream of energy |
| Fuels for own use | NOT Category 1 | Category 3 |
| Fuel purchased for resale | Category 1 (non-fuel product) | NOT Category 3 (unless energy product) |
| Energy infrastructure CapEx | NOT Category 1 (Category 2) | NOT Category 3 (Category 2) |

### 6.4 Category 3 vs Category 2

| Aspect | Category 2 (Capital Goods) | Category 3 (Fuel & Energy) |
|--------|--------------------------|----------------------------|
| What is covered | Embodied emissions of capital assets (PP&E) | Upstream lifecycle of consumed fuel and energy |
| Power plant equipment | Category 2 (embodied in infrastructure) | NOT Category 3 |
| Grid infrastructure CapEx | Category 2 (embodied in T&D hardware) | NOT Category 3 |
| Fuel consumed in operations | NOT Category 2 | Category 3 Activity 3a |

---

## 7. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models, validation |
| FastAPI | REST API framework |
| prometheus_client | Prometheus metrics |
| psycopg[binary] | PostgreSQL driver |
| TimescaleDB | Hypertables and continuous aggregates |
| AGENT-MRV-001 | Stationary Combustion Agent (Scope 1 fuel consumption data for Activity 3a) |
| AGENT-MRV-003 | Mobile Combustion Agent (Scope 1 mobile fuel data for Activity 3a) |
| AGENT-MRV-005 | Fugitive Emissions Agent (fugitive fuel-related data) |
| AGENT-MRV-009 | Scope 2 Location-Based Agent (electricity consumption data for Activities 3b/3c) |
| AGENT-MRV-010 | Scope 2 Market-Based Agent (market-based electricity data for Activities 3b/3c) |
| AGENT-MRV-011 | Steam/Heat Purchase Agent (steam/heat consumption data for Activity 3b) |
| AGENT-MRV-012 | Cooling Purchase Agent (cooling consumption data for Activity 3b) |
| AGENT-MRV-014 | Purchased Goods & Services Agent (Category 1 boundary check: fuel for own use vs goods) |
| AGENT-MRV-015 | Capital Goods Agent (Category 2 boundary check: energy infrastructure vs fuel upstream) |
| AGENT-DATA-002 | Excel/CSV Normalizer (fuel and electricity consumption spreadsheets) |
| AGENT-DATA-003 | ERP/Finance Connector (utility billing data, fuel purchase records) |
| AGENT-DATA-008 | Supplier Questionnaire Processor (fuel supplier EPD, utility upstream data) |
| AGENT-DATA-010 | Data Quality Profiler (input data quality scoring) |
| AGENT-FOUND-003 | Unit & Reference Normalizer (kWh/GJ/MMBTU/litre/kg/m3 conversion) |
| AGENT-FOUND-005 | Citations & Evidence Agent (WTT and upstream EF source citations) |
| AGENT-FOUND-001 | Orchestrator (DAG pipeline execution) |
| AGENT-FOUND-008 | Reproducibility Agent (artifact hashing, drift detection) |
| AGENT-FOUND-009 | QA Test Harness (golden file testing) |
| AGENT-FOUND-010 | Observability Agent (metrics, traces, SLO tracking) |

---

## 8. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-25 | Initial PRD |
