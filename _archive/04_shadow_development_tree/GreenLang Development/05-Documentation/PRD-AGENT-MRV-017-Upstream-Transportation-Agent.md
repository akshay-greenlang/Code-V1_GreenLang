# PRD: AGENT-MRV-017 -- Scope 3 Category 4 Upstream Transportation & Distribution Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-004 |
| **Internal Label** | AGENT-MRV-017 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/upstream_transportation/` |
| **DB Migration** | V068 |
| **Metrics Prefix** | `gl_uto_` |
| **Table Prefix** | `gl_uto_` |
| **API** | `/api/v1/upstream-transportation` |
| **Env Prefix** | `GL_UTO_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The Upstream Transportation & Distribution Agent implements **GHG Protocol
Scope 3 Category 4: Upstream Transportation and Distribution**, which
covers transportation and distribution of products purchased by the
reporting company in vehicles and facilities that are NOT owned or
controlled by the reporting company, paid for by the reporting company.
Category 4 also includes third-party warehousing and distribution center
emissions associated with inbound logistics.

Category 4 encompasses four distinct sub-activities:

- **Inbound logistics** -- Transportation of purchased goods from Tier 1
  suppliers to the reporting company's own facilities (factories,
  warehouses, offices), where the reporting company pays for or arranges
  the transport.
- **Company-paid outbound logistics** -- Transportation of sold products
  from the reporting company's facilities to customers or distribution
  points, where the reporting company pays for the freight (as determined
  by Incoterms such as DDP, CIF, CIP, DAP, DPU).
- **Third-party inter-facility transport** -- Transportation between the
  reporting company's own facilities (e.g., factory to distribution center)
  performed by third-party carriers.
- **Third-party warehousing** -- Energy consumption and emissions from
  warehouses, distribution centers, and storage facilities operated by
  third parties on behalf of the reporting company.

Category 4 is material for virtually all manufacturing, retail, and
distribution companies. It typically represents 3-15% of total Scope 3
emissions, with higher proportions in industries with long, complex supply
chains (food & beverage, retail, automotive, chemicals). The agent
automates the historically manual process of collecting shipment data
from carriers, matching transport modes and routes to emission factors,
handling multi-leg intermodal journeys, applying allocation methods for
shared transport, and ensuring zero double-counting against Category 1
(cradle-to-gate), Category 3 (fuel WTT), and Category 9 (downstream
transportation).

### Justification for Dedicated Agent

1. **Multi-modal complexity** -- Six transport modes (road, rail, maritime,
   air, pipeline, intermodal) each with distinct vehicle/vessel types,
   emission factor structures, and distance measurement conventions
2. **Multi-leg journey handling** -- Real-world logistics chains involve
   multiple legs (e.g., truck to port, ship across ocean, rail to inland
   hub, truck to final destination) with transshipment hub emissions at
   each interchange point
3. **Four calculation methods** -- Distance-based, fuel-based, spend-based,
   and supplier-specific methods with different data requirements, accuracy
   levels, and applicability by transport mode
4. **Allocation complexity** -- Shared transport (LTL trucking, container
   shipping, rail wagons) requires allocation by mass, volume, TEU, pallet
   positions, revenue, chargeable weight, or floor area
5. **Category 4 vs Category 9 boundary** -- The payment boundary (who pays
   for transport) determines whether emissions belong to Category 4 or
   Category 9, and Incoterms define this boundary precisely
6. **Refrigerated transport** -- Temperature-controlled (reefer) transport
   adds 10-40% additional energy consumption depending on mode and
   temperature regime, requiring dedicated uplift factors
7. **Warehousing emissions** -- Third-party storage facilities contribute
   significant emissions from heating, cooling, lighting, and material
   handling equipment
8. **Regulatory urgency** -- CSRD (FY2025+), ISO 14083 (transport-specific),
   GLEC Framework, SBTi FLAG, CDP all require or strongly encourage
   Category 4 reporting with mode-level granularity

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011) -- Chapter 5
- GHG Protocol Scope 3 Technical Guidance (2013) -- Chapter 4: Category 4
- GHG Protocol Scope 3 Calculation Guidance (online)
- GHG Protocol Quantitative Uncertainty Guidance
- ISO 14083:2023 -- Quantification and reporting of GHG emissions arising
  from transport chain operations
- GLEC Framework v3.0 (2023) -- Global Logistics Emissions Council methodology
  for calculating logistics emissions
- Smart Freight Centre -- Clean Cargo Working Group methodology
- DEFRA/DESNZ Greenhouse Gas Reporting Conversion Factors (annual) -- Freight
  transport tables (road, rail, maritime, air)
- EPA SmartWay -- Carrier performance data and benchmarks
- IMO Fourth GHG Study 2020 -- Maritime emission factors and vessel categories
- ICAO Carbon Emissions Calculator -- Aviation freight methodology
- CSRD/ESRS E1 -- Scope 3 disclosure (E1-6 para 44)
- California SB 253 -- Mandatory Scope 3 by FY2027 for entities >$1B revenue
- CDP Climate Change Questionnaire -- C6.5 Scope 3 Category 4
- SBTi Corporate Manual v5.3 -- Scope 3 target required if >40% of total
- GRI 305 -- Emissions (Scope 3 disclosure)
- ISO 14064-1:2018 -- Category 4 indirect GHG emissions
- ICC Incoterms 2020 -- International commercial terms defining transport
  payment responsibility

### Terminology

| Term | Definition | Scope Mapping |
|------|-----------|---------------|
| **TTW** (Tank-to-Wheel) | Direct combustion emissions from vehicle/vessel fuel use | Transport operation emissions |
| **WTT** (Well-to-Tank) | Upstream lifecycle emissions from fuel extraction through delivery | Fuel lifecycle emissions |
| **WTW** (Well-to-Wheel) | Complete lifecycle: WTT + TTW combined | Total transport fuel lifecycle |
| **tonne-km** | One tonne of cargo transported one kilometer | Standard freight activity unit |
| **TEU** | Twenty-foot Equivalent Unit (standard container measure) | Maritime/intermodal allocation unit |
| **LTL** | Less-Than-Truckload (shared truck capacity) | Road freight allocation scenario |
| **FTL** | Full Truckload (dedicated truck capacity) | Road freight (no allocation needed) |
| **GCD** | Great Circle Distance (shortest air path between two points) | Aviation distance convention |
| **SFD** | Shortest Feasible Distance (actual shipping route distance) | Maritime distance convention |
| **Reefer** | Refrigerated transport container or vehicle | Temperature-controlled logistics |
| **Laden** | Loaded state of vehicle/vessel (0%, 50%, 100%, average) | Load factor for EF selection |
| **Dead-head** | Empty return journey of a vehicle/vessel | Empty running rate component |
| **Incoterm** | International Commercial Term defining transport cost responsibility | Category 4 vs 9 boundary |
| **GLEC** | Global Logistics Emissions Council | Transport emission methodology body |
| **Transshipment** | Transfer of cargo between transport modes or vehicles at a hub | Hub emission source |

---

## 2. Methodology

### 2.1 Category 4 Boundary Definition

Category 4 covers upstream transportation and distribution services that
the reporting company PURCHASES. The critical boundary question is: **who
pays for the transportation?**

**GHG Protocol boundary rule:**

```
IF the reporting company pays for or arranges the transportation:
  --> Category 4 (Upstream Transportation & Distribution)

IF the customer pays for or arranges the transportation:
  --> Category 9 (Downstream Transportation & Distribution)

IF the reporting company owns/controls the vehicles:
  --> Scope 1 (Direct Emissions) + Scope 2 (Electricity for EVs)
```

### 2.2 Incoterms and Category Assignment

The ICC Incoterms 2020 define precisely where transport cost
responsibility transfers from seller to buyer. This directly determines
Category 4 vs Category 9 assignment:

| Incoterm | Full Name | Transport Paid By | Category Assignment |
|----------|-----------|-------------------|-------------------|
| **EXW** | Ex Works | Buyer (reporting co) pays ALL transport | Cat 4 for buyer; Cat 9 for seller |
| **FCA** | Free Carrier | Buyer pays from named place onward | Cat 4 for buyer (from named place); Cat 9 for seller (to named place) |
| **FAS** | Free Alongside Ship | Buyer pays from port of shipment | Cat 4 for buyer (from port); Cat 9 for seller (to port) |
| **FOB** | Free On Board | Buyer pays from port of loading | Cat 4 for buyer (from loading); Cat 9 for seller (to loading) |
| **CFR** | Cost and Freight | Seller pays main carriage; buyer pays from destination port | Cat 9 for seller (main carriage); Cat 4 for buyer (from destination) |
| **CIF** | Cost, Insurance & Freight | Seller pays main carriage + insurance | Cat 9 for seller (main carriage); Cat 4 for buyer (from destination) |
| **CPT** | Carriage Paid To | Seller pays to named destination | Cat 9 for seller (to destination); Cat 4 for buyer (beyond) |
| **CIP** | Carriage & Insurance Paid To | Seller pays carriage + insurance to destination | Cat 9 for seller; Cat 4 for buyer (beyond destination) |
| **DAP** | Delivered At Place | Seller pays to named destination (unloaded) | Cat 9 for seller (all transport); minimal Cat 4 for buyer |
| **DPU** | Delivered at Place Unloaded | Seller pays to destination including unloading | Cat 9 for seller; minimal Cat 4 for buyer |
| **DDP** | Delivered Duty Paid | Seller pays ALL transport + duties | Cat 9 for seller (all transport); NO Cat 4 for buyer |

**Decision logic for the reporting company as BUYER:**

```
Reporting company is the BUYER of goods:
  Incoterm is EXW, FCA, FAS, FOB:
    --> Most/all inbound transport = Category 4 for reporting company
  Incoterm is CFR, CIF, CPT, CIP:
    --> Only last-mile/destination transport = Category 4
  Incoterm is DAP, DPU, DDP:
    --> Minimal or no Category 4 (seller bears transport)
```

**Decision logic for the reporting company as SELLER:**

```
Reporting company is the SELLER of goods:
  Incoterm is DDP, DAP, DPU:
    --> Most/all outbound transport paid by seller = Category 4 (if 3rd party)
  Incoterm is CFR, CIF, CPT, CIP:
    --> Main carriage paid by seller = Category 4 (if 3rd party carrier)
  Incoterm is EXW, FCA, FAS, FOB:
    --> Minimal or no seller-paid transport = no Category 4
```

### 2.3 Four Calculation Methods

The GHG Protocol Technical Guidance and ISO 14083 define four methods for
Category 4, listed from most to least accurate:

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | Supplier-Specific | Carrier-reported emissions per shipment | Highest (+/-5-20%) | Lowest (key carriers) |
| 2 | Fuel-Based | Fuel consumed per shipment or carrier | High (+/-10-30%) | Low-Medium |
| 3 | Distance-Based | Mass, distance, mode, vehicle type | Medium (+/-20-50%) | High |
| 4 | Spend-Based | Freight spend amounts + EEIO factors | Lowest (+/-50-100%) | Highest (all spend) |

**Decision tree for method selection:**

```
Is carrier-reported emission data available (GLEC-accredited, SmartWay)?
  YES --> Use Supplier-Specific Method
  NO  --> Is fuel consumption data available per shipment?
            YES --> Use Fuel-Based Method
            NO  --> Is mass + distance + mode data available?
                      YES --> Use Distance-Based Method
                      NO  --> Is freight spend data available?
                                YES --> Use Spend-Based Method
                                NO  --> Estimate using industry benchmarks
```

### 2.4 Distance-Based Method

The distance-based method is the most commonly used approach for Category 4.
It multiplies the mass of goods transported by the distance traveled and a
mode-specific emission factor per tonne-km.

**Core formula:**

```
Emissions_leg = Mass_tonnes * Distance_km * EF_tonne_km
```

**With load factor adjustment:**

```
Emissions_leg = Mass_tonnes * Distance_km * EF_tonne_km_full
              / Load_factor_actual
```

Where `Load_factor_actual = Actual_payload / Max_payload` and
`EF_tonne_km_full` is the factor at 100% laden.

**With empty running adjustment:**

```
Emissions_leg = Mass_tonnes * Distance_km * EF_tonne_km
              * (1 + Empty_running_rate)
```

Where `Empty_running_rate` is the proportion of return journeys that are
empty (typically 15-30% for road, 30-45% for maritime).

**Per-gas breakdown:**

```
Emissions_CO2  = Mass * Distance * EF_CO2_per_tonne_km
Emissions_CH4  = Mass * Distance * EF_CH4_per_tonne_km * GWP_CH4
Emissions_N2O  = Mass * Distance * EF_N2O_per_tonne_km * GWP_N2O
Emissions_total = Emissions_CO2 + Emissions_CH4 + Emissions_N2O
```

**Distance conventions by mode:**

| Mode | Distance Convention | Adjustment | Source |
|------|-------------------|------------|--------|
| Road | Actual road distance or routing software estimate | None (or +5% for rural detour) | Google Maps, HERE, OpenStreetMap |
| Rail | Published rail distances (station-to-station) | None | Railway operator data |
| Maritime | Shortest Feasible Distance (SFD) via sea routes | Avoids land, canals accounted | SeaRoutes, MarineTraffic |
| Air | Great Circle Distance (GCD) between airport pairs | x 1.09 DEFRA uplift (routing/stacking) | ICAO, IATA |
| Pipeline | Pipeline length (published or estimated) | None | Operator data |
| Inland waterway | Published river/canal distances | None | Waterway authority data |

**Air freight GCD uplift:**

```
Distance_air_adjusted = GCD_km * 1.09

Where 1.09 = DEFRA/DESNZ uplift factor accounting for:
  - Non-great-circle routing (airway structure)
  - Holding patterns and stacking
  - Diversion to alternate airports
  - Wind correction (headwind penalties)
```

### 2.5 Fuel-Based Method

The fuel-based method directly uses fuel consumption data from carriers or
fuel purchase records to calculate emissions.

**Core formula (TTW only):**

```
Emissions_TTW = Fuel_consumed * Fuel_EF_TTW
```

**Well-to-Wheel (WTW) total:**

```
Emissions_WTW = Fuel_consumed * (Fuel_EF_TTW + Fuel_EF_WTT)
             = Fuel_consumed * Fuel_EF_WTW
```

**ISO 14083 requirement:** ISO 14083 mandates WTW scope for transport
emissions reporting. The agent supports both TTW-only and WTW calculations,
with WTW as the default and recommended scope.

**Fuel unit conversion:**

```
Fuel_kg = Fuel_litres * Density_kg_per_litre
Fuel_kg = Fuel_gallons * 3.78541 * Density_kg_per_litre
Fuel_m3 = Fuel_litres / 1000
```

**Multi-fuel support:**

| Fuel Type | TTW EF (kgCO2e/litre) | WTT EF (kgCO2e/litre) | WTW Total | Primary Mode |
|-----------|----------------------|----------------------|-----------|-------------|
| Diesel (road) | 2.5121 | 0.6244 | 3.1365 | Road, Rail |
| Petrol/Gasoline | 2.1634 | 0.5929 | 2.7563 | Road (LCV) |
| Jet Kerosene (A-1) | 2.5393 | 0.5888 | 3.1281 | Air |
| HFO (Heavy Fuel Oil) | 3.1144 | 0.5219 | 3.6363 | Maritime |
| VLSFO (Very Low Sulphur FO) | 3.1510 | 0.5300 | 3.6810 | Maritime |
| MGO (Marine Gas Oil) | 3.2063 | 0.5600 | 3.7663 | Maritime |
| LNG (transport) | 2.5520 | 0.4980 | 3.0500 | Maritime, Road |
| Methanol | 1.3746 | 0.3200 | 1.6946 | Maritime (emerging) |
| CNG | 2.5400 | 0.5100 | 3.0500 | Road |
| Electricity (grid avg) | 0.0000 (TTW) | Varies by grid | Grid EF | Road (EV), Rail (electric) |
| Hydrogen (green) | 0.0000 (TTW) | 0.8000 - 3.5000 | Varies | Road (emerging) |
| Biodiesel B20 | 2.0097 | 0.4995 | 2.5092 | Road |
| Biodiesel B100 | 0.0000 (biogenic) | 1.2400 | 1.2400 | Road |
| HVO (Hydrotreated Vegetable Oil) | 0.0000 (biogenic) | 0.8100 | 0.8100 | Road |

### 2.6 Spend-Based Method

The spend-based method estimates emissions by multiplying freight spend
by EEIO emission factors for transport sectors.

**Core formula:**

```
Emissions_spend = Freight_spend * EEIO_factor_transport_sector
```

**With currency conversion and inflation adjustment:**

```
Emissions_spend = (Freight_spend_local / FX_rate_to_base)
                * (CPI_base_year / CPI_spend_year)
                * EEIO_factor
```

**Margin removal (purchaser-to-producer price):**

```
Spend_producer = Spend_purchaser * (1 - margin_rate_transport)
```

Where `margin_rate_transport` is typically 10-25% for freight services.

**EEIO transport sector factors:**

| NAICS Code | Sector Description | EF (kgCO2e/USD) | Source |
|------------|-------------------|-----------------|--------|
| 484110 | General freight trucking, local | 0.720 | EPA USEEIO v1.2 |
| 484121 | General freight trucking, long-distance TL | 0.650 | EPA USEEIO v1.2 |
| 484122 | General freight trucking, long-distance LTL | 0.580 | EPA USEEIO v1.2 |
| 482110 | Rail transportation | 0.410 | EPA USEEIO v1.2 |
| 483111 | Deep sea freight transportation | 0.320 | EPA USEEIO v1.2 |
| 483113 | Coastal and Great Lakes freight | 0.350 | EPA USEEIO v1.2 |
| 483211 | Inland water freight transportation | 0.280 | EPA USEEIO v1.2 |
| 481112 | Scheduled freight air transportation | 1.850 | EPA USEEIO v1.2 |
| 481212 | Nonscheduled freight air transportation | 2.100 | EPA USEEIO v1.2 |
| 486110 | Pipeline transportation of crude oil | 0.150 | EPA USEEIO v1.2 |
| 486210 | Pipeline transportation of natural gas | 0.180 | EPA USEEIO v1.2 |
| 486910 | Pipeline transportation, other | 0.160 | EPA USEEIO v1.2 |
| 493110 | General warehousing and storage | 0.220 | EPA USEEIO v1.2 |
| 493120 | Refrigerated warehousing and storage | 0.380 | EPA USEEIO v1.2 |
| 493130 | Farm product warehousing and storage | 0.200 | EPA USEEIO v1.2 |
| 488510 | Freight transportation arrangement | 0.450 | EPA USEEIO v1.2 |

**EXIOBASE transport sector factors:**

| Sector | Region | EF (kgCO2e/EUR) | Source |
|--------|--------|-----------------|--------|
| Transport via railways | EU average | 0.380 | EXIOBASE 3.8 |
| Other land transport | EU average | 0.620 | EXIOBASE 3.8 |
| Transport via pipelines | EU average | 0.140 | EXIOBASE 3.8 |
| Sea and coastal water transport | EU average | 0.290 | EXIOBASE 3.8 |
| Inland water transport | EU average | 0.260 | EXIOBASE 3.8 |
| Air transport | EU average | 1.650 | EXIOBASE 3.8 |
| Warehousing and support for transport | EU average | 0.200 | EXIOBASE 3.8 |

### 2.7 Supplier-Specific Method

Uses primary emission data reported by carriers and logistics service
providers. This is the gold standard and the method preferred by ISO 14083
and the GLEC Framework.

**Core formula:**

```
Emissions_carrier = Carrier_reported_emissions_per_shipment
```

**With allocation (when carrier reports total, not per-shipment):**

```
Emissions_shipment = Carrier_total_emissions * Allocation_factor

Where Allocation_factor depends on method:
  Mass:    = Shipment_mass / Total_mass_carried
  Volume:  = Shipment_volume / Total_volume_carried
  Revenue: = Shipment_revenue / Total_revenue
  TEU:     = Shipment_TEUs / Total_TEUs
```

**Carrier data sources:**

| Source | Coverage | Accreditation | Typical DQI |
|--------|----------|--------------|-------------|
| GLEC-accredited carrier | Per-route, per-mode data with methodology documentation | GLEC Framework v3.0 | 1.0-1.5 |
| SmartWay carrier | US/Canada carrier performance data | EPA SmartWay | 1.5-2.5 |
| Clean Cargo Working Group | Container shipping carrier data | Smart Freight Centre | 1.5-2.0 |
| Carrier sustainability report | Annual total emissions with allocation | Varies | 2.0-3.0 |
| EcoTransIT World | Modeled per-shipment emissions using carrier parameters | Institut fur Energie- und Umweltforschung | 2.0-2.5 |
| Direct carrier measurement | Telematics, fuel card data, on-board diagnostics | None (primary data) | 1.0-2.0 |

### 2.8 Multi-Leg Transport Chain Calculation

Real-world logistics involves multi-leg journeys with mode changes and
hub transshipment operations. The agent decomposes transport chains into
legs and hubs, calculates emissions for each component, and aggregates
the total.

**Transport chain structure:**

```
Origin --> [Leg 1] --> Hub A --> [Leg 2] --> Hub B --> [Leg 3] --> Destination

Example: Shanghai factory to Munich warehouse
  Leg 1: Truck (Shanghai factory --> Shanghai port)          [road, 45 km]
  Hub A: Shanghai Yangshan Container Terminal                [container handling]
  Leg 2: Container ship (Shanghai --> Hamburg)               [maritime, 19,800 km SFD]
  Hub B: Hamburg HHLA Container Terminal                     [container handling]
  Leg 3: Rail (Hamburg --> Munich intermodal terminal)       [rail, 780 km]
  Hub C: Munich Riem intermodal terminal                     [rail-to-truck transfer]
  Leg 4: Truck (Munich terminal --> Munich warehouse)        [road, 25 km]
```

**Total chain emissions:**

```
Emissions_chain = SUM(Emissions_leg_i) + SUM(Emissions_hub_j)

Where:
  Emissions_leg_i = Mass * Distance_i * EF_mode_i * Adjustments_i
  Emissions_hub_j = Throughput_j * Hub_EF_j
```

**Hub/transshipment emission factors:**

| Hub Type | EF (kgCO2e/tonne) | EF (kgCO2e/TEU) | Activities Included |
|----------|-------------------|-----------------|-------------------|
| Container terminal (large) | 4.50 | 9.00 | Crane operations, yard tractors, reefer power, lighting |
| Container terminal (small) | 3.20 | 6.40 | Crane operations, yard handling |
| Airport cargo terminal | 8.50 | N/A | Ground handling, ULD build-up/break-down, cold chain |
| Rail intermodal terminal | 2.80 | 5.60 | Crane/reach stacker, shunting, lighting |
| Logistics hub / cross-dock | 2.50 | N/A | Forklift operations, lighting, HVAC |
| Cold storage hub | 12.00 | N/A | Refrigeration, lighting, material handling |
| Inland port / river terminal | 3.00 | 6.00 | Crane operations, barge loading/unloading |

### 2.9 Allocation Methods for Shared Transport

When cargo shares transport capacity with other shippers (LTL, container
shipping, rail wagons, air cargo), emissions must be allocated to the
reporting company's shipment.

**Seven allocation methods:**

| Method | Formula | Best For | ISO 14083 Preference |
|--------|---------|----------|---------------------|
| Mass | `Alloc = Shipment_mass / Total_mass` | Bulk, heavy goods | Preferred for mass-limited |
| Volume | `Alloc = Shipment_volume / Total_volume` | Volumetric goods, light/bulky | Preferred for volume-limited |
| Pallet positions | `Alloc = Shipment_pallets / Total_pallets` | Palletized road/warehouse | Common in EU road freight |
| TEU | `Alloc = Shipment_TEU / Total_TEU` | Container shipping, intermodal | Standard for maritime |
| Revenue | `Alloc = Shipment_revenue / Total_revenue` | Mixed cargo, 3PL | Fallback when physical unavailable |
| Chargeable weight | `Alloc = max(Actual_wt, Vol_wt) / Total_chargeable` | Air freight | IATA standard (6000 cm3/kg) |
| Floor area | `Alloc = Shipment_floor_m2 / Total_floor_m2` | Non-stackable, odd-shaped | Specialized road freight |

**Chargeable weight for air freight:**

```
Volumetric_weight_kg = (Length_cm * Width_cm * Height_cm) / 6000
Chargeable_weight = max(Actual_weight_kg, Volumetric_weight_kg)
```

**ISO 14083 allocation hierarchy:**

```
1. Use carrier-reported allocated emissions (if available)
2. Use mass allocation (for mass-limited transport)
3. Use volume allocation (for volume-limited transport)
4. Use TEU/pallet allocation (for containerized/palletized cargo)
5. Use revenue allocation (as last resort)
```

### 2.10 Refrigerated Transport (Reefer) Emissions

Temperature-controlled transport consumes additional energy for
refrigeration units, adding 10-40% to baseline transport emissions
depending on mode, temperature regime, and ambient conditions.

**Reefer uplift calculation:**

```
Emissions_reefer = Emissions_baseline * Reefer_uplift_factor

OR

Emissions_reefer = Reefer_fuel_consumption * Fuel_EF
```

**Reefer uplift factors by mode:**

| Mode | Temperature Regime | Uplift Factor | Source |
|------|-------------------|--------------|--------|
| Road (rigid) | Chilled (0-5C) | 1.15 | DEFRA 2025 |
| Road (rigid) | Frozen (-18C to -25C) | 1.30 | DEFRA 2025 |
| Road (articulated) | Chilled (0-5C) | 1.12 | DEFRA 2025 |
| Road (articulated) | Frozen (-18C to -25C) | 1.25 | DEFRA 2025 |
| Rail | Chilled | 1.10 | GLEC Framework |
| Rail | Frozen | 1.20 | GLEC Framework |
| Maritime (reefer container) | Chilled | 1.20 | IMO GHG Study |
| Maritime (reefer container) | Frozen | 1.40 | IMO GHG Study |
| Air | Chilled (active ULD) | 1.05 | IATA |
| Air | Frozen (active ULD) | 1.10 | IATA |

### 2.11 Warehousing & Distribution Center Emissions

Third-party warehousing emissions are included in Category 4. The agent
calculates warehousing emissions based on energy intensity, floor area,
storage duration, and temperature control requirements.

**Core formula:**

```
Emissions_warehouse = Energy_intensity * Floor_area * Duration_fraction
                    * Grid_EF

Where:
  Energy_intensity = kWh per m2 per year (varies by type and region)
  Floor_area = m2 of space used by reporting company
  Duration_fraction = Storage_days / 365
  Grid_EF = Local grid emission factor (kgCO2e/kWh)
```

**Warehouse energy intensities:**

| Warehouse Type | Energy Intensity (kWh/m2/yr) | Region | Source |
|---------------|----------------------------|--------|--------|
| Standard (ambient) | 85 - 120 | EU / temperate | BREEAM, CIBSE |
| Standard (ambient) | 100 - 150 | US average | CBECS |
| Standard (ambient) | 60 - 90 | Tropical | Estimated |
| Cold storage (chilled 0-5C) | 250 - 400 | EU / temperate | IIR, GCCA |
| Cold storage (chilled 0-5C) | 300 - 450 | US average | GCCA |
| Frozen (-18C to -25C) | 450 - 700 | EU / temperate | IIR, GCCA |
| Frozen (-18C to -25C) | 500 - 800 | US average | GCCA |
| Deep frozen (-30C to -40C) | 700 - 1100 | All | GCCA |
| Controlled atmosphere | 200 - 350 | All | GCCA |
| Hazardous materials | 150 - 250 | All | HSE standards |

### 2.12 WTW Scope Selection

The agent supports both TTW-only and WTW scope. ISO 14083 mandates WTW.

**Scope options:**

| Scope | Includes | Use Case |
|-------|----------|----------|
| TTW (Tank-to-Wheel) | Direct fuel combustion only | GHG Protocol default, traditional |
| WTT (Well-to-Tank) | Upstream fuel lifecycle only | Informational, Category 3 cross-check |
| WTW (Well-to-Wheel) | TTW + WTT combined | ISO 14083 requirement, best practice |

**Default behavior:**

```
IF framework == ISO_14083 or GLEC_FRAMEWORK:
  scope = WTW (mandatory)
ELSE:
  scope = TTW (with WTT reported separately as memo item)
  user may override to WTW
```

### 2.13 Data Quality Indicator (DQI)

Per GHG Protocol Scope 3 Standard Chapter 7, five data quality indicators
are assessed on a 1-5 scale:

| Indicator | Score 1 (Very Good) | Score 3 (Fair) | Score 5 (Very Poor) |
|-----------|--------------------|-----------------|--------------------|
| Temporal | Data from reporting year | Data within 6 years | Data older than 10 years |
| Geographical | Same country/region and route | Same continent | Global average |
| Technological | Same vehicle/vessel type and fuel | Related mode category | Different mode entirely |
| Completeness | All shipments and legs included | 50-80% of shipments covered | Less than 20% covered |
| Reliability | Third-party verified carrier data (GLEC) | Established database (DEFRA/EPA) | Estimate or assumption |

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

### 2.14 Uncertainty Ranges

| Method | Typical DQI Range | Uncertainty Range | Confidence Level |
|--------|------------------|------------------|-----------------|
| Supplier-specific (GLEC-accredited) | 1.0 - 1.5 | +/- 5-15% | Very High |
| Supplier-specific (SmartWay) | 1.5 - 2.5 | +/- 10-25% | High |
| Fuel-based (direct carrier fuel data) | 1.5 - 2.0 | +/- 10-20% | High |
| Fuel-based (allocated fuel data) | 2.0 - 3.0 | +/- 20-40% | Medium-High |
| Distance-based (DEFRA/GLEC per-tonne-km) | 2.0 - 3.0 | +/- 20-50% | Medium |
| Distance-based (generic mode average) | 3.0 - 4.0 | +/- 40-70% | Low-Medium |
| Spend-based (EEIO transport sector) | 3.5 - 4.5 | +/- 50-100% | Low |
| Hybrid | 2.0 - 3.0 | +/- 15-40% | Medium-High |

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

### 2.15 Category Boundaries & Double-Counting Prevention

**Included in Category 4:**

| Sub-Activity | What Is Included | Boundary |
|-------------|-----------------|----------|
| Inbound logistics | 3rd-party transport of purchased goods to reporting company | Supplier dock to reporting company dock |
| Company-paid outbound | 3rd-party transport of sold products where reporting company pays | Reporting company dock to customer dock (company-paid legs) |
| Inter-facility transport | 3rd-party transport between reporting company's own facilities | Facility A dock to Facility B dock |
| 3rd-party warehousing | Energy emissions from storage in 3rd-party warehouses | Warehouse energy for reporting company's goods |

**Double-Counting Prevention Rules:**

| Rule | Scope/Category Boundary | Enforcement |
|------|------------------------|-------------|
| DOUBLE_COUNT_CAT1 | vs Category 1 (Purchased Goods) | If supplier cradle-to-gate EF INCLUDES transport, do NOT add separate Cat 4 for that leg |
| DOUBLE_COUNT_CAT3 | vs Category 3 (Fuel & Energy) | WTT scope in Cat 4 must not overlap with Cat 3 WTT; if Cat 4 uses WTW, Cat 3 does NOT add WTT for transport fuel |
| DOUBLE_COUNT_CAT9 | vs Category 9 (Downstream Transport) | Payment boundary: company-paid = Cat 4, customer-paid = Cat 9; same shipment NEVER in both |
| DOUBLE_COUNT_SCOPE1 | vs Scope 1 (Direct Emissions) | Company-owned vehicles = Scope 1, not Cat 4; fleet boundary check |
| DOUBLE_COUNT_SCOPE2 | vs Scope 2 (Purchased Energy) | Company-operated electric vehicles = Scope 2 electricity, not Cat 4 |

**Cross-category validation checks:**

```
Check 1: Incoterm boundary
  For each shipment, verify that the Incoterm assignment correctly routes
  transport legs to Cat 4 vs Cat 9 (no leg in both).

Check 2: Cradle-to-gate overlap
  For each supplier, if the Cat 1 emission factor is "cradle-to-gate
  INCLUDING transport", flag any Cat 4 entries for the same inbound route
  as potential double-counting.

Check 3: Fleet ownership
  For each carrier, verify that the carrier is NOT a subsidiary or
  controlled entity of the reporting company (which would be Scope 1).

Check 4: WTT scope overlap
  If Cat 4 calculation uses WTW scope, verify that Cat 3 Activity 3a
  does NOT include WTT for the same transport fuel volume.

Check 5: Inbound vs outbound completeness
  Total Cat 4 + Cat 9 should account for all 3rd-party transport.
  Verify no gaps in coverage.
```

### 2.16 Coverage & Materiality

**Category 4 as percentage of total Scope 3 by sector:**

| Industry Sector | Cat 4 as % of Total S3 | Cat 4 as % of S1+S2+S3 | Primary Driver |
|----------------|----------------------|----------------------|----------------|
| Food & beverage manufacturing | 8 - 15% | 5 - 10% | Cold chain, global sourcing |
| Retail (grocery) | 8 - 15% | 5 - 10% | Distribution network, reefer |
| Retail (apparel) | 5 - 12% | 3 - 8% | Global sourcing, air freight |
| Automotive manufacturing | 5 - 10% | 3 - 7% | Heavy components, global supply chain |
| Chemicals | 5 - 10% | 3 - 7% | Bulk liquid/gas transport |
| Construction | 4 - 8% | 2 - 5% | Heavy materials, local delivery |
| Electronics manufacturing | 3 - 8% | 2 - 5% | Air freight for components |
| Pharmaceuticals | 5 - 12% | 3 - 7% | Cold chain, air freight |
| Mining & metals | 3 - 8% | 2 - 5% | Bulk transport, maritime |
| Technology/Software | 1 - 3% | 0.5 - 2% | Hardware logistics only |
| Financial services | 0.5 - 2% | 0.3 - 1% | Office supplies logistics |

**Coverage thresholds:**

| Level | Target | Description |
|-------|--------|-------------|
| Minimum viable | >= 80% of freight spend or shipment mass | Required for credible reporting |
| Good practice | >= 90% of freight spend or shipment mass | Recommended by CDP/SBTi |
| Best practice | >= 95% with distance-based or better for top carriers | Leading practice |

### 2.17 Emission Factor Selection Hierarchy

The agent implements a 7-level EF priority hierarchy for Category 4:

| Priority | Source | DQI Score | Applicability |
|----------|--------|-----------|--------------|
| 1 | Carrier-specific GLEC-accredited emission data | 1.0-1.5 | All modes (carrier reports) |
| 2 | Carrier-specific SmartWay performance data | 1.5-2.5 | Road/rail in US/Canada |
| 3 | Fuel-based with actual fuel consumption | 1.5-2.0 | All modes (fuel data available) |
| 4 | DEFRA/DESNZ freight transport factors (current year) | 2.0-3.0 | All modes (distance-based) |
| 5 | GLEC Framework default factors | 2.0-3.0 | All modes (distance-based) |
| 6 | IMO/ICAO mode-specific factors | 2.5-3.5 | Maritime/air |
| 7 | EEIO transport sector factors (spend-based) | 3.5-5.0 | Spend-based fallback |

### 2.18 Key Formulas Summary

**Distance-based (per leg):**

```
Emissions_leg = Mass_t * Distance_km * EF_tonne_km * (1 + Empty_rate)
              * Reefer_uplift / Load_factor
```

**Fuel-based (per shipment):**

```
Emissions_fuel = Fuel_consumed * Fuel_EF * Allocation_factor
```

**Spend-based:**

```
Emissions_spend = Freight_spend * (1 / FX_rate) * (CPI_base / CPI_yr)
                * (1 - margin_rate) * EEIO_factor
```

**Hub/transshipment:**

```
Emissions_hub = Throughput_tonnes * Hub_EF_per_tonne
```

**Warehousing:**

```
Emissions_wh = Energy_intensity * Floor_area * (Days / 365) * Grid_EF
```

**Total transport chain:**

```
Emissions_chain = SUM(Emissions_legs) + SUM(Emissions_hubs) + Emissions_warehouse
```

**Total Category 4:**

```
Emissions_cat4 = SUM(Emissions_chain_k) for all transport chains k
```

**Emissions intensity:**

```
Intensity_revenue   = Emissions_cat4 / Revenue_total       (tCO2e per $M)
Intensity_mass      = Emissions_cat4 / Total_mass_shipped   (kgCO2e per tonne)
Intensity_tonne_km  = Emissions_cat4 / Total_tonne_km       (gCO2e per tonne-km)
Intensity_spend     = Emissions_cat4 / Total_freight_spend   (tCO2e per $M spend)
```

**Year-over-year change decomposition:**

```
Delta_emissions = Delta_activity + Delta_EF + Delta_method + Delta_scope

Where:
  Delta_activity = (Tonne_km_current - Tonne_km_prior) * EF_prior
  Delta_EF       = Tonne_km_current * (EF_current - EF_prior)
  Delta_method   = Emissions_current_new_method - Emissions_current_old_method
  Delta_scope    = Change from boundary or scope changes
```

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
+-----------------------------------------------------------+
|                    AGENT-MRV-017                           |
|       Upstream Transportation & Distribution Agent         |
|                                                            |
|  +------------------------------------------------------+ |
|  | Engine 1: TransportDatabaseEngine                    | |
|  |   - Vehicle/vessel/aircraft classification tables    | |
|  |   - Road EFs (30+ entries by vehicle type, fuel,     | |
|  |     laden state)                                     | |
|  |   - Rail EFs (diesel, electric, by region)           | |
|  |   - Maritime EFs (container, bulk, tanker, general)  | |
|  |   - Air EFs (narrowbody, widebody, belly, express)   | |
|  |   - Pipeline EFs (crude, refined, gas, chemicals)    | |
|  |   - Fuel EFs (TTW + WTT for 14+ fuel types)         | |
|  |   - EEIO factors (16 transport NAICS sectors)        | |
|  |   - Hub/transshipment EFs (7 hub types)              | |
|  |   - Reefer uplift factors by mode and temp regime    | |
|  |   - Warehouse energy intensities by type and region  | |
|  |   - Load factor defaults and empty running rates     | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 2: DistanceBasedCalculatorEngine               | |
|  |   - Mass * Distance * EF_tonne_km calculation        | |
|  |   - Mode-specific EF selection (vehicle, fuel, laden)| |
|  |   - GCD uplift for air (x 1.09 DEFRA)               | |
|  |   - SFD for maritime routes                          | |
|  |   - Load factor and empty running adjustment         | |
|  |   - Reefer uplift application                        | |
|  |   - Per-gas breakdown (CO2, CH4, N2O -> CO2e)        | |
|  |   - Unit conversion (tonnes/kg, km/miles/nmi)        | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 3: FuelBasedCalculatorEngine                   | |
|  |   - Fuel_consumed * Fuel_EF calculation              | |
|  |   - TTW, WTT, and WTW scope support                 | |
|  |   - Fuel unit conversion (L/kg/gal/m3)              | |
|  |   - Multi-fuel support (14+ fuel types)             | |
|  |   - Biofuel blending (B20, B100, HVO, FAME)         | |
|  |   - Per-shipment and per-carrier aggregation        | |
|  |   - Allocation for shared fuel data                 | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 4: SpendBasedCalculatorEngine                  | |
|  |   - Spend * EEIO_factor calculation                  | |
|  |   - USEEIO (16 transport NAICS sectors)              | |
|  |   - EXIOBASE (7 transport sectors x regions)         | |
|  |   - DEFRA spend-based transport factors              | |
|  |   - Currency conversion (20 currencies)              | |
|  |   - CPI deflation to EEIO base year                 | |
|  |   - Margin removal (purchaser-to-producer)           | |
|  |   - DQI scoring (lowest tier)                        | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 5: MultiLegCalculatorEngine                    | |
|  |   - Transport chain decomposition (legs + hubs)      | |
|  |   - Per-leg emission calculation (delegates to       | |
|  |     distance/fuel/supplier engines)                  | |
|  |   - Hub/transshipment emission calculation           | |
|  |   - Intermodal routing (truck-rail, truck-ship, etc.)| |
|  |   - Carrier-specific data integration                | |
|  |   - SmartWay / GLEC carrier performance data         | |
|  |   - 7 allocation methods (mass/volume/TEU/pallet/    | |
|  |     revenue/chargeable wt/floor area)                | |
|  |   - Reefer surcharge per leg                         | |
|  |   - Warehousing emissions per hub                    | |
|  |   - Total chain emissions aggregation                | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 6: ComplianceCheckerEngine                     | |
|  |   - 7 frameworks: GHG Protocol Scope 3, ISO 14083,  | |
|  |     GLEC Framework, CSRD/ESRS E1, CDP, SBTi, GRI 305| |
|  |   - Category 4-specific compliance rules:            | |
|  |     PAYMENT_BOUNDARY, INCOTERMS_CLASSIFICATION,      | |
|  |     MODE_COVERAGE, WTW_MANDATORY (ISO 14083),        | |
|  |     TRANSPORT_CHAIN_COMPLETENESS,                    | |
|  |     ALLOCATION_METHOD_CONSISTENCY,                    | |
|  |     DOUBLE_COUNTING_CAT1/CAT3/CAT9,                 | |
|  |     REEFER_INCLUSION, WAREHOUSING_INCLUSION,         | |
|  |     DATA_QUALITY_MINIMUM                             | |
|  |   - DQI scoring validation                           | |
|  |   - Gap identification and recommendations           | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 7: TransportPipelineEngine                     | |
|  |   - 10-stage pipeline orchestration                  | |
|  |   - Batch multi-period processing                    | |
|  |   - Multi-facility aggregation                       | |
|  |   - Payment boundary enforcement                     | |
|  |   - Export (JSON/CSV/Excel/PDF)                      | |
|  |   - Compliance-ready outputs (CDP, CSRD, ISO 14083)  | |
|  |   - Provenance chain assembly                        | |
|  +------------------------------------------------------+ |
+-----------------------------------------------------------+
```

### 3.2 Ten-Stage Pipeline

```
Stage 1: VALIDATE
  - Schema validation of shipment, fuel, and spend records
  - Required field checks (mode, mass/spend, origin, destination)
  - Data type enforcement (Decimal for quantities and factors)
  - Duplicate detection (same shipment/carrier/period)
  - Unit consistency validation (tonnes/kg, km/miles, litres/gallons)
  - Incoterm validity check

Stage 2: CLASSIFY
  - Transport mode classification (road, rail, maritime, air, pipeline, intermodal)
  - Vehicle/vessel type classification within each mode
  - Calculation method selection (distance/fuel/spend/supplier)
  - Temperature control classification (ambient, chilled, frozen, heated)
  - Incoterm-based Category 4 vs Category 9 assignment
  - Hub type classification

Stage 3: NORMALIZE
  - Unit conversion to EF reference units (tonnes, km, litres)
  - Currency conversion for spend-based records
  - CPI deflation to EEIO base year
  - Distance normalization (miles to km, nautical miles to km)
  - Mass normalization (kg to tonnes, lbs to tonnes)
  - Fuel volume normalization (gallons to litres)
  - Period normalization (monthly/quarterly/annual)

Stage 4: RESOLVE_EFS
  - 7-level EF hierarchy resolution for each leg
  - Mode-specific EF selection (vehicle type, fuel, laden state)
  - Carrier-specific override if supplier data available
  - Fuel-specific EF for fuel-based records
  - EEIO sector EF for spend-based records
  - Hub EF selection by hub type
  - Warehouse energy intensity by type and region
  - Missing EF flagging with fallback

Stage 5: CALCULATE_LEGS
  - Per-leg emission calculation using appropriate method:
    - Distance-based: Mass * Distance * EF * adjustments
    - Fuel-based: Fuel * EF * allocation
    - Spend-based: Spend * EEIO * adjustments
    - Supplier-specific: Carrier-reported emissions
  - Load factor adjustment
  - Empty running rate adjustment
  - Reefer uplift application
  - GCD uplift for air legs (x 1.09)
  - Per-gas breakdown (CO2, CH4, N2O)
  - GWP application (AR4/AR5/AR6)
  - WTW scope if ISO 14083 or user-selected

Stage 6: CALCULATE_HUBS
  - Hub/transshipment emission calculation per hub
  - Warehouse emission calculation (energy intensity * area * duration)
  - Cold storage additional energy for reefer warehousing
  - Hub throughput-based allocation if shared facility

Stage 7: ALLOCATE
  - Apply allocation method for shared transport legs
  - Mass, volume, TEU, pallet, revenue, chargeable weight, floor area
  - Consistency check: same method within transport chain
  - Allocation factor validation (sum to 1.0 or less)

Stage 8: COMPLIANCE
  - 7-framework compliance check
  - Payment boundary validation (Cat 4 vs Cat 9)
  - Incoterms classification audit
  - Mode coverage completeness
  - WTW mandatory check for ISO 14083
  - Double-counting prevention verification (Cat 1, Cat 3, Cat 9)
  - Data quality minimum thresholds
  - Gap identification and recommendations

Stage 9: AGGREGATE
  - Total Category 4 emissions
  - By transport mode (road, rail, maritime, air, pipeline)
  - By carrier / logistics provider
  - By route / trade lane
  - By product category / commodity
  - By origin-destination pair
  - By reporting period
  - Intensity metrics (per tonne, per tonne-km, per $M revenue, per $M spend)
  - Year-over-year change decomposition
  - Hot-spot analysis (top routes, top carriers, top modes)

Stage 10: SEAL
  - SHA-256 provenance hash
  - Audit trail assembly
  - Export generation (JSON/CSV/Excel/PDF)
  - Result persistence
  - Provenance chain linking to shipment source data
```

### 3.3 File Structure

```
greenlang/upstream_transportation/
+-- __init__.py                              # Lazy imports, module exports
+-- models.py                                # Pydantic v2 models, enums, constants
+-- config.py                                # GL_UTO_ prefixed configuration
+-- metrics.py                               # Prometheus metrics (gl_uto_*)
+-- provenance.py                            # SHA-256 provenance chain
+-- transport_database.py                    # Engine 1: Transport EF database
+-- distance_based_calculator.py             # Engine 2: Distance-based calculation
+-- fuel_based_calculator.py                 # Engine 3: Fuel-based calculation
+-- spend_based_calculator.py                # Engine 4: Spend-based calculation
+-- multi_leg_calculator.py                  # Engine 5: Multi-leg chain calculation
+-- compliance_checker.py                    # Engine 6: Compliance checking
+-- transport_pipeline.py                    # Engine 7: Pipeline orchestration
+-- setup.py                                 # Service facade
+-- api/
    +-- __init__.py                          # API package
    +-- router.py                            # FastAPI REST endpoints

tests/unit/mrv/test_upstream_transportation/
+-- __init__.py
+-- conftest.py
+-- test_models.py
+-- test_config.py
+-- test_metrics.py
+-- test_provenance.py
+-- test_transport_database.py
+-- test_distance_based_calculator.py
+-- test_fuel_based_calculator.py
+-- test_spend_based_calculator.py
+-- test_multi_leg_calculator.py
+-- test_compliance_checker.py
+-- test_transport_pipeline.py
+-- test_setup.py
+-- test_api.py

deployment/database/migrations/sql/
+-- V068__upstream_transportation_service.sql
```

### 3.4 Database Schema (V068)

16 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `gl_uto_transport_modes` | Transport mode taxonomy (6 modes) with sub-classifications | Seed (6+ rows) |
| `gl_uto_vehicle_types` | Vehicle/vessel/aircraft type definitions with capacity | Seed (40+ rows) |
| `gl_uto_distance_emission_factors` | Distance-based EFs by mode, vehicle type, fuel, laden state | Seed (100+ rows) |
| `gl_uto_fuel_emission_factors` | Fuel-based EFs (TTW, WTT, WTW) by fuel type | Seed (14+ rows) |
| `gl_uto_eeio_transport_factors` | EEIO factors for transport sectors (USEEIO + EXIOBASE) | Seed (25+ rows) |
| `gl_uto_hub_emission_factors` | Hub/transshipment EFs by hub type | Seed (7+ rows) |
| `gl_uto_reefer_uplift_factors` | Reefer uplift factors by mode and temperature regime | Seed (10+ rows) |
| `gl_uto_warehouse_energy` | Warehouse energy intensities by type, region | Seed (20+ rows) |
| `gl_uto_load_factor_defaults` | Default load factors and empty running rates by mode | Seed (12+ rows) |
| `gl_uto_shipments` | Shipment input records (mass, origin, destination, mode, carrier) | Hypertable (partitioned by shipped_at) |
| `gl_uto_transport_chains` | Multi-leg transport chain definitions (legs + hubs) | Regular |
| `gl_uto_calculations` | Calculation results with per-leg and per-hub breakdown | Hypertable (partitioned by calculated_at) |
| `gl_uto_calculation_details` | Line-item detail per calculation (per leg, per hub) | Regular |
| `gl_uto_carrier_data` | Carrier-specific emission data (GLEC, SmartWay) | Regular |
| `gl_uto_compliance_records` | Compliance check results (7 frameworks) | Regular |
| `gl_uto_dqi_scores` | Data quality scores per shipment/leg | Regular |
| `gl_uto_aggregations` | Aggregated results by dimension (mode, carrier, route, period) | Regular |
| `gl_uto_batch_jobs` | Batch processing jobs | Regular |
| `gl_uto_audit_entries` | Audit trail entries for all operations | Hypertable (partitioned by created_at) |
| `gl_uto_hourly_stats` | Hourly calculation statistics | Continuous Aggregate |
| `gl_uto_daily_stats` | Daily calculation statistics | Continuous Aggregate |

**Key seed data:**

- `gl_uto_transport_modes`: 6 rows covering ROAD, RAIL, MARITIME, AIR,
  PIPELINE, INTERMODAL with sub-classifications
- `gl_uto_vehicle_types`: 40+ rows covering all vehicle/vessel/aircraft
  types: road (LCV, rigid 3.5-7.5t, 7.5-17t, 17-26t, articulated 33t,
  40-44t, road train, EV, CNG, LNG, H2), rail (diesel, electric), maritime
  (container feeder through ULCV, bulk handysize through capesize, tanker
  aframax through VLCC, general cargo, RoRo, inland barge), air (narrowbody
  freighter, widebody freighter, belly cargo, express integrator),
  pipeline (crude, refined, gas, chemicals)
- `gl_uto_distance_emission_factors`: 100+ rows from DEFRA 2025 and GLEC
  Framework v3.0, covering each vehicle type x fuel type x laden state
- `gl_uto_fuel_emission_factors`: 14+ rows with TTW, WTT, and WTW factors
  for all transport fuels (diesel, petrol, jet kerosene, HFO, VLSFO, MGO,
  LNG, methanol, CNG, electricity, hydrogen, B20, B100, HVO)
- `gl_uto_eeio_transport_factors`: 25+ rows from EPA USEEIO v1.2 (16 NAICS
  transport sectors) and EXIOBASE 3.8 (7 EU transport sectors)
- `gl_uto_hub_emission_factors`: 7+ rows covering all hub types
- `gl_uto_reefer_uplift_factors`: 10+ rows by mode x temperature regime
- `gl_uto_warehouse_energy`: 20+ rows by warehouse type x region
- `gl_uto_load_factor_defaults`: 12+ rows default load factors and empty
  running rates per mode and vehicle type

**Schema design principles:**

- Row-Level Security (RLS) on all tenant-facing tables via `tenant_id`
- TimescaleDB hypertables on `gl_uto_shipments`, `gl_uto_calculations`,
  and `gl_uto_audit_entries` (partitioned by temporal columns)
- Continuous aggregates for hourly and daily calculation statistics
- Foreign key relationships from `gl_uto_calculation_details` to
  `gl_uto_calculations`
- Foreign key relationships from `gl_uto_transport_chains` to
  `gl_uto_shipments`
- GIN indexes on JSONB columns for metadata queries
- B-tree indexes on transport_mode, vehicle_type_id, carrier_id, tenant_id
- Partial indexes for active/current records
- Composite indexes on (tenant_id, reporting_period, transport_mode) for
  fast lookups

### 3.5 API Endpoints (20)

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/calculate` | Run single shipment or transport chain calculation |
| 2 | POST | `/calculate/batch` | Batch calculate emissions for multiple shipments |
| 3 | GET | `/calculations` | List calculations with filtering (mode, carrier, period) |
| 4 | GET | `/calculations/{calculation_id}` | Get calculation result with per-leg breakdown |
| 5 | DELETE | `/calculations/{calculation_id}` | Delete a calculation |
| 6 | POST | `/transport-chains` | Create a multi-leg transport chain definition |
| 7 | GET | `/transport-chains` | List transport chains with filtering |
| 8 | GET | `/transport-chains/{chain_id}` | Get transport chain with legs and hubs |
| 9 | GET | `/emission-factors` | List transport EFs with filtering (mode, vehicle, fuel) |
| 10 | GET | `/emission-factors/{ef_id}` | Get specific emission factor details |
| 11 | POST | `/emission-factors/custom` | Register custom transport emission factor |
| 12 | POST | `/classify` | Classify shipment (mode, vehicle type, method selection) |
| 13 | POST | `/compliance/check` | Run compliance check (all 7 frameworks) |
| 14 | GET | `/compliance/{calculation_id}` | Get compliance results for a calculation |
| 15 | POST | `/uncertainty` | Run uncertainty analysis on a calculation |
| 16 | GET | `/aggregations` | Get aggregated results (by mode, carrier, route, period) |
| 17 | GET | `/hot-spots` | Get top emission contributors (routes, carriers, modes) |
| 18 | POST | `/export` | Export results (JSON/CSV/Excel/PDF) |
| 19 | GET | `/health` | Health check and service statistics |
| 20 | GET | `/stats` | Detailed service statistics and throughput metrics |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All emission calculations use Python `Decimal` (8 decimal places)
- No LLM calls in any calculation path -- deterministic lookups only
- Every calculation step recorded in a provenance trace
- SHA-256 provenance hash for every emission result
- Bit-perfect reproducibility: same input always produces same output
- Distance-based EF lookup is exact-match by mode, vehicle type, fuel type, and laden state
- Fuel-based EF lookup is exact-match by fuel type and scope (TTW/WTT/WTW)
- EEIO factor lookup is exact-match by NAICS/NACE sector code and database version
- Hub EF lookup is exact-match by hub type
- Reefer uplift lookup is exact-match by mode and temperature regime
- GWP values are deterministic lookup by assessment report version (AR4/AR5/AR6)
- Allocation calculations use exact fraction arithmetic (no floating-point rounding)
- Incoterm classification uses deterministic lookup table, not heuristic inference

### 4.2 Enumerations (22)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | DISTANCE_BASED, FUEL_BASED, SPEND_BASED, SUPPLIER_SPECIFIC, HYBRID | 5 calculation methods |
| `TransportMode` | ROAD, RAIL, MARITIME, AIR, PIPELINE, INTERMODAL | 6 transport modes |
| `RoadVehicleType` | LCV_PETROL, LCV_DIESEL, RIGID_3_5_7_5T, RIGID_7_5_17T, RIGID_17_26T, ARTICULATED_33T, ARTICULATED_40_44T, ROAD_TRAIN, ELECTRIC_VAN, ELECTRIC_TRUCK, CNG_TRUCK, LNG_TRUCK, HYDROGEN_TRUCK | 13 road vehicle types |
| `RailType` | DIESEL, ELECTRIC, AVERAGE | 3 rail traction types |
| `MaritimeVesselType` | CONTAINER_FEEDER, CONTAINER_SUB_PANAMAX, CONTAINER_PANAMAX, CONTAINER_POST_PANAMAX, CONTAINER_ULCV, BULK_HANDYSIZE, BULK_HANDYMAX, BULK_PANAMAX, BULK_CAPESIZE, TANKER_AFRAMAX, TANKER_SUEZMAX, TANKER_VLCC, GENERAL_CARGO, RORO, INLAND_BARGE | 15 maritime vessel types |
| `AircraftType` | NARROWBODY_FREIGHTER, WIDEBODY_FREIGHTER, BELLY_CARGO, EXPRESS_INTEGRATOR | 4 aircraft types |
| `PipelineType` | CRUDE_OIL, REFINED_PRODUCTS, NATURAL_GAS, CHEMICALS | 4 pipeline types |
| `FuelType` | DIESEL, PETROL, JET_KEROSENE, HFO, VLSFO, MGO, LNG, METHANOL, CNG, ELECTRICITY, HYDROGEN, BIODIESEL_B20, BIODIESEL_B100, HVO, AMMONIA | 15 fuel types |
| `LadenState` | EMPTY, HALF, FULL, AVERAGE | 4 laden states |
| `AllocationMethod` | MASS, VOLUME, PALLET_POSITIONS, TEU, REVENUE, CHARGEABLE_WEIGHT, FLOOR_AREA | 7 allocation methods |
| `Incoterm` | EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP | 11 Incoterms |
| `HubType` | LOGISTICS_HUB, CONTAINER_TERMINAL, AIRPORT_CARGO, RAIL_TERMINAL, WAREHOUSE, COLD_STORAGE_HUB, INLAND_PORT | 7 hub types |
| `TemperatureControl` | AMBIENT, CHILLED, FROZEN, HEATED | 4 temperature regimes |
| `EFScope` | TTW, WTT, WTW | 3 emission factor scopes |
| `DistanceMethod` | ACTUAL, SFD, GCD, ESTIMATED | 4 distance determination methods |
| `CurrencyCode` | USD, EUR, GBP, JPY, CNY, CHF, CAD, AUD, KRW, INR, BRL, MXN, SEK, NOK, DKK, SGD, HKD, NZD, ZAR, AED | 20 currencies |
| `DQIDimension` | TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL, COMPLETENESS, RELIABILITY | 5 DQI dimensions |
| `DQIScore` | VERY_GOOD (1), GOOD (2), FAIR (3), POOR (4), VERY_POOR (5) | 5 quality scores |
| `UncertaintyMethod` | MONTE_CARLO, ANALYTICAL, TIER_DEFAULT | 3 uncertainty methods |
| `ComplianceFramework` | GHG_PROTOCOL_SCOPE3, ISO_14083, GLEC_FRAMEWORK, CSRD_ESRS_E1, CDP, SBTI, GRI_305 | 7 regulatory frameworks |
| `ComplianceStatus` | COMPLIANT, PARTIAL, NON_COMPLIANT | 3 compliance statuses |
| `PipelineStage` | VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE_LEGS, CALCULATE_HUBS, ALLOCATE, COMPLIANCE, AGGREGATE, SEAL | 10 pipeline stages |
| `ExportFormat` | JSON, CSV, XLSX, PDF | 4 export formats |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED | 4 batch statuses |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | 4 GWP assessment report versions |
| `EmissionGas` | CO2, CH4, N2O, CO2E | 4 emission gases |

### 4.3 Models (25)

| Model | Description | Key Fields |
|-------|-------------|------------|
| `TransportEmissionFactor` | Distance-based EF record | ef_id, mode, vehicle_type, fuel_type, laden_state, ef_co2e_per_tonne_km, ef_co2_per_tonne_km, ef_ch4_per_tonne_km, ef_n2o_per_tonne_km, source, source_year |
| `FuelEmissionFactor` | Fuel-based EF record | ef_id, fuel_type, ef_ttw_per_litre, ef_wtt_per_litre, ef_wtw_per_litre, density_kg_per_litre, source |
| `EEIOTransportFactor` | Spend-based EEIO factor | ef_id, naics_code, nace_code, sector_description, ef_kgco2e_per_usd, ef_kgco2e_per_eur, database, base_year |
| `HubEmissionFactor` | Hub/transshipment EF | ef_id, hub_type, ef_per_tonne, ef_per_teu, activities_included, source |
| `ReeferUpliftFactor` | Reefer uplift factor | factor_id, mode, temperature_control, uplift_factor, source |
| `WarehouseEnergyIntensity` | Warehouse energy intensity | record_id, warehouse_type, region, energy_kwh_per_m2_yr, source |
| `TransportLeg` | Single transport leg definition | leg_id, chain_id, sequence, mode, vehicle_type, fuel_type, origin, destination, distance_km, distance_method, laden_state, mass_tonnes |
| `TransportHub` | Hub/transshipment point | hub_id, chain_id, sequence, hub_type, name, location, throughput_tonnes, dwell_time_hours |
| `TransportChain` | Multi-leg transport chain | chain_id, shipment_id, legs (list of TransportLeg), hubs (list of TransportHub), total_distance_km, total_mass_tonnes |
| `ShipmentInput` | Shipment input record | shipment_id, origin, destination, mass_tonnes, volume_m3, commodity, incoterm, carrier_id, shipped_at, method |
| `FuelConsumptionInput` | Fuel consumption input | record_id, shipment_id, fuel_type, quantity, unit, carrier_id |
| `SpendInput` | Freight spend input | record_id, carrier_id, amount, currency, naics_code, description, period |
| `SupplierEmissionInput` | Carrier-reported emission | record_id, carrier_id, carrier_name, emissions_kgco2e, data_source, accreditation, shipment_id |
| `AllocationConfig` | Allocation configuration | method, shipment_mass, total_mass, shipment_volume, total_volume, shipment_teu, total_teu, shipment_pallets, total_pallets |
| `ReeferConfig` | Reefer configuration | temperature_control, target_temp_c, reefer_fuel_litres (optional), use_uplift_factor |
| `WarehouseConfig` | Warehouse configuration | warehouse_type, floor_area_m2, storage_days, region, grid_ef_kgco2e_per_kwh |
| `LegResult` | Per-leg emission result | leg_id, mode, distance_km, emissions_co2, emissions_ch4, emissions_n2o, emissions_co2e, ef_source, method, ef_scope |
| `HubResult` | Per-hub emission result | hub_id, hub_type, emissions_co2e, throughput, energy_kwh |
| `TransportChainResult` | Total chain result | chain_id, leg_results, hub_results, total_emissions_co2e, total_distance_km, allocation_method, reefer_uplift_applied |
| `CalculationRequest` | API calculation request | shipments, transport_chains, fuel_records, spend_records, supplier_records, reporting_year, organization_id, gwp_source, ef_scope, options |
| `CalculationResult` | Complete calculation output | calculation_id, chain_results, total_co2e, by_mode, by_carrier, dqi_score, provenance_hash, processing_time_ms |
| `BatchResult` | Batch calculation output | batch_id, results, total_co2e, success_count, failure_count, processing_time_ms |
| `ComplianceRequirement` | Framework compliance requirement | requirement_id, framework, rule_code, description, is_mandatory, check_type |
| `ComplianceCheckResult` | Compliance check result | result_id, framework, status, score, findings, gaps, recommendations |
| `AggregationResult` | Aggregated emissions by dimension | aggregation_id, dimension, breakdowns, total_co2e, intensity_metrics, hot_spots |

### 4.4 Constant Tables (13)

#### 4.4.1 GWP_VALUES

| Gas | AR4 (100yr) | AR5 (100yr) | AR6 (100yr) | AR6 (20yr) |
|-----|-------------|-------------|-------------|------------|
| CO2 | 1 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 27.9 | 82.5 |
| N2O | 298 | 265 | 273 | 273 |

#### 4.4.2 ROAD_EMISSION_FACTORS

Distance-based EFs for road freight (per tonne-km):

| Vehicle Type | Fuel | Laden State | EF (kgCO2e/t-km) | Source |
|-------------|------|-------------|------------------|--------|
| LCV (petrol) | Petrol | Average | 0.58620 | DEFRA 2025 |
| LCV (diesel) | Diesel | Average | 0.46944 | DEFRA 2025 |
| Rigid 3.5-7.5t | Diesel | 0% laden | 0.56378 | DEFRA 2025 |
| Rigid 3.5-7.5t | Diesel | 50% laden | 0.30505 | DEFRA 2025 |
| Rigid 3.5-7.5t | Diesel | 100% laden | 0.21098 | DEFRA 2025 |
| Rigid 3.5-7.5t | Diesel | Average | 0.28916 | DEFRA 2025 |
| Rigid 7.5-17t | Diesel | 0% laden | 0.48461 | DEFRA 2025 |
| Rigid 7.5-17t | Diesel | 50% laden | 0.19170 | DEFRA 2025 |
| Rigid 7.5-17t | Diesel | 100% laden | 0.13137 | DEFRA 2025 |
| Rigid 7.5-17t | Diesel | Average | 0.18507 | DEFRA 2025 |
| Rigid 17-26t | Diesel | 0% laden | 0.60086 | DEFRA 2025 |
| Rigid 17-26t | Diesel | 50% laden | 0.14226 | DEFRA 2025 |
| Rigid 17-26t | Diesel | 100% laden | 0.09340 | DEFRA 2025 |
| Rigid 17-26t | Diesel | Average | 0.13753 | DEFRA 2025 |
| Articulated 33t | Diesel | 0% laden | 0.53459 | DEFRA 2025 |
| Articulated 33t | Diesel | 50% laden | 0.09073 | DEFRA 2025 |
| Articulated 33t | Diesel | 100% laden | 0.05771 | DEFRA 2025 |
| Articulated 33t | Diesel | Average | 0.08661 | DEFRA 2025 |
| Articulated 40-44t | Diesel | 0% laden | 0.58862 | DEFRA 2025 |
| Articulated 40-44t | Diesel | 50% laden | 0.07527 | DEFRA 2025 |
| Articulated 40-44t | Diesel | 100% laden | 0.04631 | DEFRA 2025 |
| Articulated 40-44t | Diesel | Average | 0.07218 | DEFRA 2025 |
| Road train | Diesel | Average | 0.04100 | GLEC Framework |
| Electric van (EV) | Electricity | Average | 0.06200 | GLEC Framework |
| Electric truck | Electricity | Average | 0.02400 | GLEC Framework |
| CNG truck | CNG | Average | 0.09800 | DEFRA 2025 |
| LNG truck | LNG | Average | 0.08900 | DEFRA 2025 |
| Hydrogen truck | Hydrogen | Average | 0.01500 | GLEC Framework |

#### 4.4.3 RAIL_EMISSION_FACTORS

| Rail Type | Region | EF (kgCO2e/t-km) | Source |
|-----------|--------|------------------|--------|
| Diesel freight | Global average | 0.02738 | DEFRA 2025 |
| Diesel freight | US | 0.02300 | EPA SmartWay |
| Diesel freight | EU | 0.02800 | GLEC Framework |
| Electric freight | Global average | 0.01678 | DEFRA 2025 |
| Electric freight | EU | 0.01200 | GLEC Framework |
| Electric freight | US | 0.01800 | eGRID + GLEC |
| Electric freight | China | 0.02500 | IEA |
| Average (mix) | UK | 0.02548 | DEFRA 2025 |
| Average (mix) | Global | 0.02200 | GLEC Framework |

#### 4.4.4 MARITIME_EMISSION_FACTORS

| Vessel Type | Size Class | EF (kgCO2e/t-km) | Source |
|------------|-----------|------------------|--------|
| Container (feeder) | <1,000 TEU | 0.03112 | DEFRA 2025 |
| Container (sub-panamax) | 1,000-5,100 TEU | 0.01642 | DEFRA 2025 |
| Container (panamax) | 5,100-10,000 TEU | 0.01200 | DEFRA 2025 |
| Container (post-panamax) | 10,000-14,500 TEU | 0.00932 | DEFRA 2025 |
| Container (ULCV) | >14,500 TEU | 0.00758 | DEFRA 2025 |
| Container (average) | All sizes | 0.01604 | DEFRA 2025 |
| Bulk carrier (handysize) | 10,000-40,000 DWT | 0.00768 | DEFRA 2025 |
| Bulk carrier (handymax) | 40,000-60,000 DWT | 0.00604 | DEFRA 2025 |
| Bulk carrier (panamax) | 60,000-100,000 DWT | 0.00520 | DEFRA 2025 |
| Bulk carrier (capesize) | >100,000 DWT | 0.00395 | DEFRA 2025 |
| Bulk carrier (average) | All sizes | 0.00526 | DEFRA 2025 |
| Tanker (aframax) | 80,000-120,000 DWT | 0.00571 | DEFRA 2025 |
| Tanker (suezmax) | 120,000-200,000 DWT | 0.00474 | DEFRA 2025 |
| Tanker (VLCC) | >200,000 DWT | 0.00357 | DEFRA 2025 |
| General cargo | Various | 0.01240 | DEFRA 2025 |
| RoRo | Vehicle carrier | 0.03250 | DEFRA 2025 |
| Inland waterway (barge) | Various | 0.03100 | GLEC Framework |

#### 4.4.5 AIR_EMISSION_FACTORS

| Aircraft Type | Distance Band | EF (kgCO2e/t-km) | Source |
|--------------|-------------|------------------|--------|
| Narrowbody freighter | <3,700 km | 1.12600 | DEFRA 2025 |
| Widebody freighter | >3,700 km | 0.59400 | DEFRA 2025 |
| Belly cargo (short-haul) | <3,700 km | 0.99200 | DEFRA 2025 |
| Belly cargo (long-haul) | >3,700 km | 0.49400 | DEFRA 2025 |
| Express integrator (short) | <3,700 km | 1.28500 | DEFRA 2025 |
| Express integrator (long) | >3,700 km | 0.68100 | DEFRA 2025 |
| Average (all) | All | 0.60200 | DEFRA 2025 |
| Average (international) | >3,700 km | 0.52300 | DEFRA 2025 |
| Average (domestic) | <3,700 km | 1.05400 | DEFRA 2025 |

**Note:** Air freight EFs include DEFRA 1.09x GCD uplift by default.
If user provides actual routing distance, set `apply_gcd_uplift = False`.

#### 4.4.6 PIPELINE_EMISSION_FACTORS

| Pipeline Type | EF (kgCO2e/t-km) | Notes | Source |
|--------------|------------------|-------|--------|
| Crude oil pipeline | 0.00420 | Based on pump energy consumption | DEFRA 2025 |
| Refined products pipeline | 0.00380 | Lower viscosity, less pump energy | DEFRA 2025 |
| Natural gas pipeline | 0.01850 | Per tonne equivalent; includes compressor stations | EPA |
| Chemicals pipeline | 0.00400 | Similar to refined products | Estimated from EPA |

#### 4.4.7 FUEL_EMISSION_FACTORS

Complete TTW + WTT factors for all transport fuels (per litre unless noted):

| Fuel Type | TTW (kgCO2e/L) | WTT (kgCO2e/L) | WTW (kgCO2e/L) | Density (kg/L) | Source |
|-----------|---------------|----------------|----------------|---------------|--------|
| Diesel | 2.5121 | 0.6244 | 3.1365 | 0.832 | DEFRA 2025 |
| Petrol | 2.1634 | 0.5929 | 2.7563 | 0.749 | DEFRA 2025 |
| Jet Kerosene (A-1) | 2.5393 | 0.5888 | 3.1281 | 0.800 | DEFRA 2025 |
| HFO (Heavy Fuel Oil) | 3.1144 | 0.5219 | 3.6363 | 0.940 | DEFRA 2025 |
| VLSFO | 3.1510 | 0.5300 | 3.6810 | 0.920 | IMO |
| MGO (Marine Gas Oil) | 3.2063 | 0.5600 | 3.7663 | 0.876 | DEFRA 2025 |
| LNG (per kg) | 2.7500 | 0.5400 | 3.2900 | 0.450 | IMO |
| Methanol | 1.3746 | 0.3200 | 1.6946 | 0.792 | GLEC |
| CNG (per kg) | 2.5400 | 0.5100 | 3.0500 | N/A | DEFRA 2025 |
| Electricity (per kWh) | 0.0000 | Varies | Grid EF | N/A | Grid-dependent |
| Hydrogen green (per kg) | 0.0000 | 0.8000 | 0.8000 | N/A | GLEC |
| Hydrogen grey (per kg) | 0.0000 | 9.0000 | 9.0000 | N/A | IEA |
| Biodiesel B20 | 2.0097 | 0.4995 | 2.5092 | 0.840 | DEFRA 2025 |
| Biodiesel B100 | 0.0000* | 1.2400 | 1.2400 | 0.880 | DEFRA 2025 |
| HVO | 0.0000* | 0.8100 | 0.8100 | 0.780 | DEFRA 2025 |

*Biogenic CO2 from B100 and HVO combustion reported separately (memo item).

#### 4.4.8 EEIO_TRANSPORT_FACTORS

See Section 2.6 for complete USEEIO and EXIOBASE transport sector factor
tables (16 NAICS sectors + 7 EXIOBASE sectors).

#### 4.4.9 HUB_EMISSION_FACTORS

See Section 2.8 for complete hub/transshipment emission factor table
(7 hub types with per-tonne and per-TEU factors).

#### 4.4.10 REEFER_UPLIFT_FACTORS

See Section 2.10 for complete reefer uplift factor table (10 entries
by mode x temperature regime).

#### 4.4.11 WAREHOUSE_ENERGY_INTENSITIES

See Section 2.11 for complete warehouse energy intensity table (10 entries
by warehouse type x region).

#### 4.4.12 LOAD_FACTOR_DEFAULTS

| Mode | Vehicle Type | Default Load Factor | Empty Running Rate | Source |
|------|-------------|--------------------|--------------------|--------|
| Road | LCV | 0.50 | 0.25 | DEFRA 2025 |
| Road | Rigid (all) | 0.55 | 0.20 | DEFRA 2025 |
| Road | Articulated (all) | 0.60 | 0.18 | DEFRA 2025 |
| Road | Road train | 0.65 | 0.15 | GLEC |
| Rail | Diesel | 0.60 | 0.30 | DEFRA 2025 |
| Rail | Electric | 0.60 | 0.30 | DEFRA 2025 |
| Maritime | Container (all) | 0.70 | 0.40 | IMO |
| Maritime | Bulk (all) | 0.80 | 0.45 | IMO |
| Maritime | Tanker (all) | 0.85 | 0.50 | IMO |
| Air | Freighter | 0.65 | 0.10 | IATA |
| Air | Belly cargo | 0.45 | 0.00 | IATA |
| Pipeline | All | 1.00 | 0.00 | N/A |

#### 4.4.13 EMPTY_RUNNING_RATES

| Mode | Region | Empty Running Rate | Source |
|------|--------|-------------------|--------|
| Road | EU | 0.21 | Eurostat |
| Road | US | 0.18 | BTS |
| Road | Global average | 0.20 | GLEC |
| Rail | EU | 0.30 | GLEC |
| Rail | US | 0.28 | AAR |
| Maritime (container) | Global | 0.40 | IMO |
| Maritime (bulk) | Global | 0.45 | IMO |
| Maritime (tanker) | Global | 0.50 | IMO |
| Air | Global | 0.10 | IATA |

### 4.5 Regulatory Frameworks (7)

1. **GHG Protocol Scope 3 Standard** -- Chapter 5 Category 4 definition;
   transport modes, allocation, method hierarchy; Chapter 7 DQI;
   Chapter 9 reporting requirements; payment boundary rule for Cat 4 vs Cat 9.

2. **ISO 14083:2023** -- Transport-specific quantification standard; mandates
   WTW scope (TTW + WTT); defines transport chain decomposition methodology;
   requires per-leg calculation with hub emissions; prescribes allocation
   hierarchy (mass > volume > TEU > revenue); requires data quality
   classification per ISO 14083 Annex B.

3. **GLEC Framework v3.0** -- Aligns with ISO 14083; provides default
   emission factors by mode/vehicle/fuel; carrier accreditation program;
   defines GLEC-compliant reporting format; includes hub and warehousing
   emissions in transport chain boundary.

4. **CSRD/ESRS E1** -- E1-6 para 44a/44b/44c Scope 3 by category,
   methodology, data sources; para 46 intensity metrics; para 48 value
   chain engagement for logistics providers.

5. **CDP Climate Change** -- C6.5 Category 4 relevance assessment and
   calculation; transport mode breakdown; methodology per mode; carrier
   engagement; year-over-year explanation for fluctuations.

6. **SBTi v5.3** -- Scope 3 target required if >40% of total; 67%
   coverage; logistics provider engagement targets; near-term 5-10 years;
   FLAG sector guidance for food/agriculture supply chains.

7. **GRI 305** -- Scope 3 disclosure if significant; methodology and EF
   sources; base year and recalculation policy.

**Category 4-specific compliance rules:**

| Rule Code | Description | Framework |
|-----------|-------------|-----------|
| `PAYMENT_BOUNDARY` | Only company-paid transport included in Cat 4 | GHG Protocol |
| `INCOTERMS_CLASSIFICATION` | Incoterms correctly route legs to Cat 4 vs Cat 9 | GHG Protocol |
| `MODE_COVERAGE` | All material transport modes included | All |
| `WTW_MANDATORY` | WTW scope required (TTW + WTT) | ISO 14083, GLEC |
| `TRANSPORT_CHAIN_COMPLETENESS` | All legs and hubs in chain accounted | ISO 14083, GLEC |
| `ALLOCATION_METHOD_CONSISTENCY` | Same allocation method within transport chain | ISO 14083 |
| `DOUBLE_COUNTING_CAT1` | No overlap with cradle-to-gate supplier factors | GHG Protocol |
| `DOUBLE_COUNTING_CAT3` | WTT scope consistency with Category 3 | GHG Protocol |
| `DOUBLE_COUNTING_CAT9` | Payment boundary enforcement, no overlap | GHG Protocol |
| `REEFER_INCLUSION` | Temperature-controlled transport accounted | GLEC, ISO 14083 |
| `WAREHOUSING_INCLUSION` | 3rd-party storage included in Cat 4 | GHG Protocol, GLEC |
| `DATA_QUALITY_MINIMUM` | Method hierarchy compliance; DQI documented | All |

**Framework required disclosures:**

| Framework | Required Disclosures for Category 4 |
|-----------|-------------------------------------|
| GHG Protocol Scope 3 | Total Cat 4 emissions (tCO2e), breakdown by mode (road/rail/maritime/air/pipeline), methodology per mode, allocation methods used, payment boundary documentation, carrier engagement level, warehousing emissions |
| ISO 14083 | WTW emissions per transport chain, per-leg breakdown, hub emissions, allocation methodology, data quality classification per Annex B, carrier data source and accreditation |
| GLEC Framework | Per-mode emissions with GLEC-compliant factors, transport chain definition, allocation method, carrier accreditation status, data quality tier |
| CSRD/ESRS E1 | E1-6 para 44: Cat 4 gross emissions with mode breakdown, methodology, % supplier-specific, carrier data sources; para 46: intensity per tonne-km and per revenue; para 48: logistics provider engagement |
| CDP | C6.5: Cat 4 relevance, emissions by mode, methodology, % carrier-specific data, transport intensity metrics, YoY explanation |
| SBTi | Scope 3 screening, Cat 4 significance, logistics engagement targets, coverage validation |
| GRI 305 | Scope 3 Cat 4 if significant, methodology and EF sources, base year |

### 4.6 Performance Targets

| Metric | Target |
|--------|--------|
| Single shipment calculation (1 leg, distance-based) | < 20ms |
| Multi-leg chain calculation (5 legs, 4 hubs) | < 100ms |
| Batch calculation (100 shipments) | < 2s |
| Batch calculation (1,000 shipments) | < 10s |
| Fuel-based calculation (single shipment) | < 15ms |
| Spend-based calculation (100 line items) | < 200ms |
| Supplier-specific lookup (single carrier) | < 5ms |
| EF resolution (7-level hierarchy, single leg) | < 10ms |
| Hub emission calculation (single hub) | < 5ms |
| Warehouse emission calculation (single facility) | < 5ms |
| Allocation calculation (single shipment, any method) | < 5ms |
| DQI scoring (full inventory) | < 200ms |
| Compliance check (all 7 frameworks) | < 300ms |
| Full pipeline (500 shipments, mixed methods) | < 15s |
| Hot-spot analysis (full inventory) | < 500ms |

---

## 5. Acceptance Criteria

### 5.1 Core Calculation -- Distance-Based Method

- [ ] Mass * Distance * EF calculation for all 6 transport modes
- [ ] Road EFs for 13 vehicle types x 4 laden states
- [ ] Rail EFs for diesel, electric, average by region
- [ ] Maritime EFs for 15 vessel types
- [ ] Air EFs for 4 aircraft types with GCD uplift (x1.09)
- [ ] Pipeline EFs for 4 pipeline types
- [ ] Load factor adjustment (actual vs max payload)
- [ ] Empty running rate adjustment by mode and region
- [ ] Reefer uplift application by mode and temperature regime
- [ ] Per-gas breakdown (CO2, CH4, N2O) with GWP application (AR4/AR5/AR6)
- [ ] Unit conversion (tonnes/kg/lbs, km/miles/nautical miles)
- [ ] Distance method classification (actual, SFD, GCD, estimated)

### 5.2 Core Calculation -- Fuel-Based Method

- [ ] Fuel consumed * Fuel EF for 15 fuel types
- [ ] TTW, WTT, and WTW scope support with configurable default
- [ ] Fuel unit conversion (litres, kg, gallons, m3)
- [ ] Biofuel blending (B20, B100, HVO with biogenic CO2 memo)
- [ ] Multi-fuel per-shipment support
- [ ] Per-carrier fuel aggregation
- [ ] ISO 14083 WTW mandatory enforcement

### 5.3 Core Calculation -- Spend-Based Method

- [ ] Freight spend * EEIO factor for 16 NAICS transport sectors
- [ ] EXIOBASE factors for 7 EU transport sectors
- [ ] Currency conversion for 20 currencies
- [ ] CPI deflation to EEIO base year
- [ ] Margin removal (purchaser-to-producer price)
- [ ] Transport sector NAICS/NACE code mapping
- [ ] DQI scoring (lowest tier for spend-based)

### 5.4 Core Calculation -- Supplier-Specific Method

- [ ] Carrier-reported emission data integration
- [ ] GLEC-accredited carrier data processing
- [ ] SmartWay carrier performance data
- [ ] Clean Cargo Working Group data
- [ ] Allocation from carrier total to shipment (mass/revenue)
- [ ] Carrier data quality validation
- [ ] Accreditation status tracking

### 5.5 Multi-Leg Transport Chain

- [ ] Transport chain decomposition into legs + hubs
- [ ] Per-leg emission calculation with method selection per leg
- [ ] Hub/transshipment emission calculation for 7 hub types
- [ ] Intermodal routing (truck-rail, truck-ship, truck-air, etc.)
- [ ] Total chain emission aggregation
- [ ] Reefer surcharge applied per leg where applicable
- [ ] Warehousing emissions (energy intensity * area * duration * grid EF)

### 5.6 Allocation

- [ ] 7 allocation methods (mass, volume, pallet, TEU, revenue, chargeable wt, floor area)
- [ ] Chargeable weight for air freight (IATA 6000 cm3/kg volumetric)
- [ ] Allocation consistency within transport chain (ISO 14083)
- [ ] Allocation factor validation (<=1.0, non-negative)

### 5.7 Category Boundary & Double-Counting Prevention

- [ ] Incoterm-based Category 4 vs Category 9 assignment
- [ ] Payment boundary enforcement (company-paid = Cat 4)
- [ ] Cradle-to-gate overlap check vs Category 1
- [ ] WTT scope overlap check vs Category 3
- [ ] Fleet ownership check (company-owned = Scope 1, not Cat 4)
- [ ] Inbound vs outbound completeness verification
- [ ] No single shipment in both Cat 4 and Cat 9

### 5.8 Data Quality & Uncertainty

- [ ] 5-dimension DQI scoring (temporal, geographical, technological, completeness, reliability)
- [ ] Composite DQI score (1.0-5.0 scale)
- [ ] Quality classification (Very Good through Very Poor)
- [ ] Pedigree matrix uncertainty quantification
- [ ] Weighted DQI for total inventory (emission-weighted)
- [ ] Uncertainty analysis (Monte Carlo, analytical, tier default)

### 5.9 Compliance

- [ ] 7 regulatory framework compliance checks
- [ ] 12 Category 4-specific compliance rules
- [ ] ISO 14083 WTW mandatory check
- [ ] GLEC Framework accreditation validation
- [ ] CSRD/ESRS E1 data point coverage
- [ ] CDP C6.5 scoring criteria alignment
- [ ] SBTi coverage threshold validation (67% of Scope 3)
- [ ] Double-counting prevention documentation generation
- [ ] Gap identification with actionable recommendations

### 5.10 Infrastructure

- [ ] 20 REST API endpoints at `/api/v1/upstream-transportation`
- [ ] V068 database migration (16 tables, 3 hypertables, 2 continuous aggregates)
- [ ] SHA-256 provenance on every calculation result
- [ ] Prometheus metrics with `gl_uto_` prefix (12 metrics)
- [ ] Auth integration (route_protector.py + auth_setup.py with uto_router)
- [ ] 20 PERMISSION_MAP entries for upstream-transportation
- [ ] 575+ unit tests
- [ ] All calculations use Python `Decimal` (no floating point in emission path)
- [ ] Export in JSON, CSV, Excel, and PDF formats
- [ ] Row-Level Security (RLS) on all tenant-facing tables
- [ ] Provenance chain linking to shipment source data

---

## 6. Prometheus Metrics (12)

All prefixed `gl_uto_`:

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | calculations_total | Counter | transport_mode, method, status |
| 2 | emissions_kg_co2e_total | Counter | transport_mode, fuel_type, gas |
| 3 | transport_chain_calculations_total | Counter | leg_count, status |
| 4 | factor_selections_total | Counter | method, source, mode |
| 5 | allocation_operations_total | Counter | allocation_method, mode |
| 6 | uncertainty_runs_total | Counter | method |
| 7 | compliance_checks_total | Counter | framework, status |
| 8 | batch_jobs_total | Counter | status |
| 9 | calculation_duration_seconds | Histogram | operation |
| 10 | batch_size | Histogram | method |
| 11 | active_calculations | Gauge | - |
| 12 | shipments_processed | Gauge | transport_mode |

---

## 7. Auth Integration

### 7.1 PERMISSION_MAP Entries (20)

The following 20 permission entries are added to `PERMISSION_MAP` in
`auth_setup.py` for the upstream-transportation resource:

| # | Method | Path Pattern | Permission |
|---|--------|-------------|------------|
| 1 | POST | /api/v1/upstream-transportation/calculate | upstream-transportation:execute |
| 2 | POST | /api/v1/upstream-transportation/calculate/batch | upstream-transportation:execute |
| 3 | GET | /api/v1/upstream-transportation/calculations | upstream-transportation:read |
| 4 | GET | /api/v1/upstream-transportation/calculations/{id} | upstream-transportation:read |
| 5 | DELETE | /api/v1/upstream-transportation/calculations/{id} | upstream-transportation:write |
| 6 | POST | /api/v1/upstream-transportation/transport-chains | upstream-transportation:write |
| 7 | GET | /api/v1/upstream-transportation/transport-chains | upstream-transportation:read |
| 8 | GET | /api/v1/upstream-transportation/transport-chains/{id} | upstream-transportation:read |
| 9 | GET | /api/v1/upstream-transportation/emission-factors | upstream-transportation:read |
| 10 | GET | /api/v1/upstream-transportation/emission-factors/{id} | upstream-transportation:read |
| 11 | POST | /api/v1/upstream-transportation/emission-factors/custom | upstream-transportation:write |
| 12 | POST | /api/v1/upstream-transportation/classify | upstream-transportation:execute |
| 13 | POST | /api/v1/upstream-transportation/compliance/check | upstream-transportation:execute |
| 14 | GET | /api/v1/upstream-transportation/compliance/{id} | upstream-transportation:read |
| 15 | POST | /api/v1/upstream-transportation/uncertainty | upstream-transportation:execute |
| 16 | GET | /api/v1/upstream-transportation/aggregations | upstream-transportation:read |
| 17 | GET | /api/v1/upstream-transportation/hot-spots | upstream-transportation:read |
| 18 | POST | /api/v1/upstream-transportation/export | upstream-transportation:read |
| 19 | GET | /api/v1/upstream-transportation/health | upstream-transportation:read |
| 20 | GET | /api/v1/upstream-transportation/stats | upstream-transportation:read |

### 7.2 Router Registration

In `auth_setup.py`, add:

```python
from greenlang.upstream_transportation.api.router import router as uto_router
```

Register in `configure_auth(app)`:

```python
app.include_router(uto_router, prefix="/api/v1/upstream-transportation", tags=["upstream-transportation"])
```

Apply route protection:

```python
protect_routes(app, uto_router, permission_map=PERMISSION_MAP)
```

---

## 8. Test Suite (575+ tests)

### 8.1 Test Files (15)

| # | File | Tests | Coverage Area |
|---|------|-------|--------------|
| 1 | `test_models.py` | 55 | All 22 enums, 25 models, 13 constant tables, frozen immutability |
| 2 | `test_config.py` | 15 | GL_UTO_ env vars, defaults, validation |
| 3 | `test_metrics.py` | 15 | 12 Prometheus metrics registration, labels, types |
| 4 | `test_provenance.py` | 20 | SHA-256 hashing, chain linking, determinism |
| 5 | `test_transport_database.py` | 60 | EF lookup by mode/vehicle/fuel/laden, hierarchy resolution, seed data integrity |
| 6 | `test_distance_based_calculator.py` | 80 | All modes, laden states, GCD uplift, empty running, reefer, per-gas, units |
| 7 | `test_fuel_based_calculator.py` | 50 | All fuels, TTW/WTT/WTW, unit conversion, biofuel blending |
| 8 | `test_spend_based_calculator.py` | 40 | EEIO factors, currency, CPI, margin removal, NAICS mapping |
| 9 | `test_multi_leg_calculator.py` | 70 | Chain decomposition, per-leg calc, hubs, allocation (7 methods), reefer, warehousing |
| 10 | `test_compliance_checker.py` | 55 | 7 frameworks, 12 rules, Incoterm classification, double-counting checks |
| 11 | `test_transport_pipeline.py` | 50 | 10 stages, batch processing, error handling, provenance |
| 12 | `test_setup.py` | 15 | Service facade initialization, engine wiring |
| 13 | `test_api.py` | 30 | 20 endpoints, request/response validation, error responses |
| 14 | `conftest.py` | 10 | Shared fixtures, mock data, test transport chains |
| 15 | `__init__.py` | 0 | Package marker |
| | **TOTAL** | **575+** | |

### 8.2 Key Test Scenarios

**Distance-based calculation tests:**

- Road: Articulated 40-44t diesel, 100% laden, 500 km, 20 tonnes = expected kgCO2e
- Road: LCV petrol, average laden, 50 km, 0.5 tonnes = expected kgCO2e
- Rail: Electric freight EU, 1000 km, 50 tonnes = expected kgCO2e
- Maritime: Container panamax, SFD 19,800 km, 5 TEU (12.5 tonnes each) = expected kgCO2e
- Air: Widebody freighter, GCD 8,500 km * 1.09 uplift, 2 tonnes = expected kgCO2e
- Pipeline: Natural gas, 2,000 km, 500 tonnes = expected kgCO2e
- Reefer: Articulated frozen, 1.25 uplift, 300 km, 18 tonnes = expected kgCO2e
- Empty running: Rigid 17-26t, 20% empty rate, 200 km, 10 tonnes = expected kgCO2e

**Multi-leg chain tests:**

- Shanghai-to-Munich 4-leg chain (truck + ship + rail + truck) with 3 hubs
- US coast-to-coast intermodal (truck + rail + truck) with 2 hubs
- Air freight with cold chain (truck + air + truck) with reefer on road legs
- Simple FTL road shipment (1 leg, no hubs, no allocation)

**Allocation tests:**

- Mass allocation: 5 tonnes of 20 tonne shipment = 0.25 factor
- TEU allocation: 2 TEU of 200 TEU vessel = 0.01 factor
- Chargeable weight: actual 50kg vs volumetric 80kg = 80kg chargeable
- Revenue allocation: $5,000 of $100,000 total = 0.05 factor

**Compliance tests:**

- ISO 14083 rejects TTW-only scope (WTW mandatory)
- GHG Protocol flags Incoterm DDP shipment in Cat 4 (should be Cat 9 for buyer)
- Double-counting alert when Cat 1 supplier EF includes transport
- GLEC Framework requires transport chain completeness

**Boundary tests:**

- Incoterm EXW: all transport assigned to Cat 4 for buyer
- Incoterm DDP: no Cat 4 for buyer, all Cat 9 for seller
- Incoterm FOB: ocean leg assigned to Cat 4 for buyer, pre-ocean to Cat 9
- Company-owned truck shipment rejected (Scope 1, not Cat 4)

---

## 9. Key Differentiators from Adjacent Categories

### 9.1 Category 4 vs Category 9

| Aspect | Category 4 (Upstream Transport) | Category 9 (Downstream Transport) |
|--------|-------------------------------|----------------------------------|
| Payment boundary | Reporting company PAYS for transport | Customer PAYS for transport |
| Direction | Inbound to reporting company + company-paid outbound | Customer-paid outbound from reporting company |
| Incoterms (buyer) | EXW, FCA, FAS, FOB = Cat 4 dominant | DDP, DAP, DPU = Cat 9 (for seller) |
| Incoterms (seller) | DDP, DAP, CIF, CFR = Cat 4 (if 3rd party) | EXW, FOB = no seller-paid outbound |
| Warehousing | 3rd-party warehousing of PURCHASED goods | 3rd-party warehousing of SOLD goods |
| Agent | AGENT-MRV-017 (this agent) | Future AGENT-MRV-022 (Downstream Transport) |

### 9.2 Category 4 vs Category 1

| Aspect | Category 1 (Purchased Goods) | Category 4 (Upstream Transport) |
|--------|-----------------------------|---------------------------------|
| What is covered | Cradle-to-gate emissions of purchased goods | Transport of purchased goods to reporting company |
| Transport inclusion | Some Cat 1 EFs include transport (cradle-to-gate-delivered) | Separate transport-only emissions |
| Double-counting risk | If Cat 1 EF includes transport, Cat 4 should exclude that leg | Flag cradle-to-gate EFs that include transport component |
| Agent | AGENT-MRV-014 | AGENT-MRV-017 (this agent) |

### 9.3 Category 4 vs Category 3

| Aspect | Category 3 (Fuel & Energy) | Category 4 (Upstream Transport) |
|--------|--------------------------|--------------------------------|
| What is covered | WTT of fuels consumed by reporting company | Transport emissions from 3rd-party carriers |
| WTT overlap risk | Cat 3 includes WTT of ALL fuels consumed by company | Cat 4 WTW includes WTT of carrier fuel; not double-counted with Cat 3 |
| Fuel scope | Fuels consumed in company operations | Fuels consumed by 3rd-party carriers |
| Agent | AGENT-MRV-016 | AGENT-MRV-017 (this agent) |

### 9.4 Category 4 vs Scope 1

| Aspect | Scope 1 (Direct) | Category 4 (Upstream Transport) |
|--------|------------------|---------------------------------|
| Vehicle ownership | Company-owned or controlled vehicles | 3rd-party carrier vehicles |
| Fleet boundary | Company fleet (owned, leased, operated) | Contracted carriers |
| Fuel purchase | Company buys fuel for own fleet | Carrier buys fuel (included in EF) |
| Agent | AGENT-MRV-003 (Mobile Combustion) | AGENT-MRV-017 (this agent) |

---

## 10. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models, validation |
| FastAPI | REST API framework |
| prometheus_client | Prometheus metrics |
| psycopg[binary] | PostgreSQL driver |
| TimescaleDB | Hypertables and continuous aggregates |
| AGENT-MRV-014 | Purchased Goods Agent (Cat 1 double-counting check: cradle-to-gate EF transport overlap) |
| AGENT-MRV-016 | Fuel & Energy Agent (Cat 3 WTT scope overlap check) |
| AGENT-DATA-002 | Excel/CSV Normalizer (shipment data spreadsheets) |
| AGENT-DATA-003 | ERP/Finance Connector (freight invoice data, PO logistics data) |
| AGENT-DATA-008 | Supplier Questionnaire Processor (carrier emission data, SmartWay, GLEC) |
| AGENT-DATA-009 | Spend Data Categorizer (freight spend NAICS classification) |
| AGENT-DATA-010 | Data Quality Profiler (input data quality scoring) |
| AGENT-FOUND-001 | Orchestrator (DAG pipeline execution) |
| AGENT-FOUND-003 | Unit & Reference Normalizer (tonnes/kg, km/miles, litres/gallons conversion) |
| AGENT-FOUND-005 | Citations & Evidence Agent (EF source citations, DEFRA/GLEC/IMO) |
| AGENT-FOUND-008 | Reproducibility Agent (artifact hashing, drift detection) |
| AGENT-FOUND-009 | QA Test Harness (golden file testing) |
| AGENT-FOUND-010 | Observability Agent (metrics, traces, SLO tracking) |

---

## 11. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-25 | Initial PRD |
