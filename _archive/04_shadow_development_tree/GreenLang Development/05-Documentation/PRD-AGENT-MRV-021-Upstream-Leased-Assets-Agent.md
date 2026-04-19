# PRD: AGENT-MRV-021 -- Scope 3 Category 8 Upstream Leased Assets Agent

---

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-008 |
| **Internal Label** | AGENT-MRV-021 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/upstream_leased_assets/` |
| **DB Migration** | V072 |
| **Metrics Prefix** | `gl_ula_` |
| **Table Prefix** | `gl_ula_` |
| **API** | `/api/v1/upstream-leased-assets` |
| **Env Prefix** | `GL_ULA_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The **GL-MRV-S3-008 Upstream Leased Assets Agent** implements GHG Protocol Scope 3 Category 8 emissions accounting for the operation of assets leased by the reporting company (lessee) in the reporting year that are not already included in the company's Scope 1 and Scope 2 inventories. This agent automates calculation of greenhouse gas emissions from leased buildings, vehicles, equipment, and IT infrastructure where the reporting organization is the lessee under operating leases.

Category 8 covers the following sub-activities as defined in the GHG Protocol Scope 3 Standard (Chapter 8):

- **Leased Buildings (8a)** -- Emissions from operating leased office space, retail stores, warehouses, data centers, and other commercial buildings. This includes electricity, natural gas, heating oil, district heating/cooling, and on-site renewable energy. The largest sub-category for most service-sector companies.
- **Leased Vehicles (8b)** -- Emissions from operating leased vehicle fleets including passenger cars, light-duty trucks, heavy-duty trucks, vans, and specialty vehicles. Covers both fuel combustion (Scope 1 equivalent) and upstream fuel (WTT).
- **Leased Equipment (8c)** -- Emissions from operating leased industrial equipment, manufacturing machinery, construction equipment, agricultural equipment, and generators. Includes both fuel-powered and electrically-powered equipment.
- **Leased IT Assets (8d)** -- Emissions from operating leased IT infrastructure including servers, networking equipment, storage systems, printers/copiers, and co-located data center space. Covers both electricity consumption and embodied carbon amortization.

Upstream leased assets typically represent **1-15% of total Scope 3 emissions** depending on the organization's real estate footprint, fleet size, and accounting approach (operational vs financial control). For companies with large leased real estate portfolios (retail, hospitality, professional services), this can be one of the most material Scope 3 categories. The distinction between operating and finance leases under IFRS 16/ASC 842 is critical -- finance leases are typically included in Scope 1 and 2, while operating leases fall under Category 8.

The agent automates extraction and calculation of emissions from lease management systems, energy bills, utility data, fleet telemetry, equipment logs, and building management systems (BMS), producing audit-ready outputs with full provenance chains, data quality scoring, and multi-framework regulatory compliance.

### Justification for Dedicated Agent

1. **Lease classification complexity** -- Operating vs finance lease distinction (IFRS 16 / ASC 842) determines whether emissions belong in Scope 1/2 or Category 8, requiring automated accounting treatment classification
2. **Multi-asset diversity** -- Buildings, vehicles, equipment, and IT each have fundamentally different emission calculation methodologies, emission factors, and data sources
3. **Allocation methods** -- Shared/multi-tenant leased spaces require allocation by floor area, headcount, or revenue, with partial-year and partial-space adjustments
4. **Energy disaggregation** -- Building emissions require disaggregation of electricity, natural gas, heating oil, district heating/cooling, and on-site generation
5. **Double-counting prevention** -- Complex boundaries with Scope 1 (owned assets), Scope 2 (purchased electricity for leased buildings), Category 1 (purchased goods), Category 2 (capital goods), and Category 13 (downstream leased assets)
6. **Lessor data integration** -- Organizations may receive emissions data directly from lessors (lessor-specific method) requiring validation and normalization
7. **Portfolio-level aggregation** -- Large organizations may have hundreds of leased assets across multiple countries with different grid factors, building codes, and energy mixes
8. **Regulatory urgency** -- CSRD ESRS E1, CDP, SBTi, and SB 253 all require Category 8 disclosure with specific data quality requirements

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard, Chapter 8 (Category 8)
- GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 8: Upstream Leased Assets
- GHG Protocol Scope 2 Guidance (for electricity emissions methodology alignment)
- IFRS 16 Leases -- Lease Classification Criteria
- ASC 842 Leases -- US GAAP Lease Classification
- DEFRA/DESNZ 2024 Government GHG Conversion Factors -- Energy & Fuels (Tables 1-4)
- IEA CO2 Emissions from Fuel Combustion (2024 Edition) -- Grid Emission Factors
- EPA eGRID 2022 -- US Sub-Regional Grid Emission Factors
- IPCC Sixth Assessment Report (AR6) -- Global Warming Potentials (Table 7.15)
- CSRD / ESRS E1 -- Climate Change (paragraphs 48-56, AR 46-48)
- California SB 253 Climate Corporate Data Accountability Act -- Scope 3 Disclosure
- CDP Climate Change Questionnaire 2024 -- C6.3 Leased Assets
- SBTi Corporate Net-Zero Standard v1.2 -- Scope 3 Boundary Requirements
- GRI 305: Emissions 2016 -- Disclosure 305-3
- ISO 14064-1:2018 -- Quantification and reporting of GHG emissions and removals
- ENERGY STAR Portfolio Manager -- Building Energy Benchmarking (US)
- EU Energy Performance of Buildings Directive (EPBD) -- Building Energy Certificates
- ASHRAE 90.1 -- Energy Standard for Commercial Buildings
- CRREM Carbon Risk Real Estate Monitor -- Building Decarbonization Pathways

### Terminology

| Term | Definition | Scope Mapping |
|------|-----------|---------------|
| Operating Lease | Lease where lessee does NOT obtain substantially all economic benefits and risks of the underlying asset; asset stays on lessor's balance sheet | Category 8 (Scope 3) |
| Finance Lease | Lease where lessee obtains substantially all economic benefits and risks; asset on lessee's balance sheet | Scope 1 and 2 |
| Lessee | Organization that leases (rents) an asset from a lessor | Reporting company for Cat 8 |
| Lessor | Organization that owns the asset and leases it to the lessee | Data provider |
| Floor Area (NLA) | Net Lettable Area -- the usable floor space leased by the organization (sqm or sqft) | Allocation basis |
| Energy Use Intensity (EUI) | Annual energy consumption per unit floor area (kWh/sqm/year or kBtu/sqft/year) | Building benchmarking |
| Grid Emission Factor | CO2e per kWh of electricity from the grid, varying by location and time | Location-based EF |
| Residual Mix Factor | Grid factor adjusted for contractual instruments (RECs, PPAs, GOs) | Market-based EF |
| District Heating/Cooling | Centralized heating or cooling supplied via network to multiple buildings | Non-electric energy |
| Building Management System (BMS) | Automated control system for building HVAC, lighting, and energy | Primary data source |
| Fleet Telematics | GPS and vehicle data systems tracking fuel consumption and distance | Vehicle data source |
| Power Usage Effectiveness (PUE) | Ratio of total data center energy to IT equipment energy (typically 1.2-2.0) | Data center metric |
| Allocation Factor | Percentage of shared building/asset emissions attributed to the lessee | Partial occupancy |
| WTT (Well-to-Tank) | Upstream emissions from fuel extraction, refining, and distribution | Fuel supply chain |
| TTW (Tank-to-Wheel) | Direct emissions from fuel combustion at the asset | Operational emissions |
| Scope 1-equivalent | Emissions that would be Scope 1 if the asset were owned (e.g., gas boiler in leased building) | Included in Cat 8 |
| Scope 2-equivalent | Emissions that would be Scope 2 if the asset were owned (e.g., purchased electricity) | Included in Cat 8 |
| ENERGY STAR Score | EPA benchmarking score (1-100) for US commercial buildings | Building performance |
| EPC Rating | Energy Performance Certificate rating (A-G) for EU buildings | EU building label |

---

## 2. Methodology

### 2.1 Category Boundary Definition

```
Category 8: Upstream Leased Assets
+-- INCLUDED
|   +-- Operating leased buildings (office, retail, warehouse, industrial, data center)
|   +-- Operating leased vehicles (cars, trucks, vans, specialty)
|   +-- Operating leased equipment (industrial, construction, generators)
|   +-- Operating leased IT assets (servers, network, storage, printers)
|   +-- Electricity consumed in leased buildings (Scope 2-equivalent)
|   +-- Natural gas / heating fuel in leased buildings (Scope 1-equivalent)
|   +-- District heating / cooling in leased buildings
|   +-- Fuel combustion in leased vehicles (Scope 1-equivalent)
|   +-- WTT (well-to-tank) emissions for all fuel types
|   +-- Refrigerant leakage from leased HVAC equipment (if data available)
|   +-- Multi-tenant allocation for shared spaces
|   +-- Partial-year leases (prorated by occupancy period)
|
+-- EXCLUDED (reported elsewhere)
|   +-- Finance-leased assets --> Scope 1 and Scope 2
|   +-- Owned assets --> Scope 1 and Scope 2
|   +-- Assets leased to others (reporting org is lessor) --> Category 13
|   +-- Purchased goods and services used in leased buildings --> Category 1
|   +-- Capital goods installed in leased buildings --> Category 2
|   +-- Fuel and energy activities (upstream) --> Category 3
|   +-- Waste generated in leased buildings --> Category 5
|   +-- Employee commuting to leased buildings --> Category 7
|   +-- End-of-life treatment of leased assets --> Category 12
|
+-- BOUNDARY TESTS
    +-- IF operating lease (IFRS 16/ASC 842) --> Include in Cat 8
    +-- IF finance lease --> Exclude (report in Scope 1/2)
    +-- IF owned asset --> Exclude (report in Scope 1/2)
    +-- IF company is lessor --> Exclude (report in Cat 13)
    +-- IF partial occupancy (multi-tenant) --> Allocate by area/headcount/revenue
    +-- IF partial-year lease --> Prorate by months of occupancy
    +-- IF sub-lease (company is intermediate lessee) --> Include head lease portion
    +-- IF energy included in lease --> Include (may need estimation from rent)
    +-- IF energy billed separately --> Use actual energy data
```

### 2.2 Calculation Methods

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | **Asset-specific** | Actual energy consumption data per leased asset (utility bills, BMS, telematics) | Highest | Low-Medium |
| 2 | **Lessor-specific** | Emissions data provided directly by the lessor (landlord reports, green lease data) | High | Medium |
| 3 | **Average-data** | Asset type + floor area/count x average energy intensity benchmarks | Medium | High |
| 4 | **Spend-based** | Lease expenditure x EEIO emission factors | Low | Highest |

```
Method Selection Decision Tree:
-----------------------------------
START
+-- Actual energy data available per asset?
|   +-- YES --> Method 1: Asset-specific
|   |   +-- Utility bills, BMS data, fleet telematics
|   |   +-- Apply location/market grid factors
|   +-- NO (next)
+-- Lessor provides emissions data?
|   +-- YES --> Method 2: Lessor-specific
|   |   +-- Validate lessor methodology
|   |   +-- Check boundary completeness
|   +-- NO (next)
+-- Floor area / asset count + type known?
|   +-- YES --> Method 3: Average-data
|   |   +-- Use building EUI benchmarks
|   |   +-- Use vehicle average km/year
|   +-- NO (next)
+-- Only lease spend data available?
    +-- YES --> Method 4: Spend-based
        +-- Lease payments x EEIO factor
```

### 2.3 Asset-Specific Method (Building)

```
Per-Building Annual Emissions:
  Building_CO2e = Electricity_CO2e + Gas_CO2e + Heating_CO2e + Cooling_CO2e + Refrigerant_CO2e

  Electricity_CO2e = Annual_kWh x Grid_EF (kgCO2e/kWh)
  Gas_CO2e = Annual_therms x Gas_EF (kgCO2e/therm)
  Heating_CO2e = Annual_kWh_heat x District_Heating_EF (kgCO2e/kWh)
  Cooling_CO2e = Annual_kWh_cool x District_Cooling_EF (kgCO2e/kWh)
  Refrigerant_CO2e = Leakage_kg x GWP

  WTT:
    WTT_Electricity = Annual_kWh x WTT_Grid_EF
    WTT_Gas = Annual_therms x WTT_Gas_EF
    Total_WTT = WTT_Electricity + WTT_Gas

  Multi-Tenant Allocation:
    Allocated_CO2e = Total_Building_CO2e x Allocation_Factor
    Where:
      Allocation_Factor = Leased_Area / Total_Building_Area (area-based)
      OR Allocation_Factor = Lessee_Headcount / Total_Headcount (headcount-based)
      OR Allocation_Factor = Lessee_Revenue / Total_Revenue (revenue-based)

  Partial-Year Adjustment:
    Prorated_CO2e = Annual_CO2e x (Lease_Months / 12)

  Total Per Building:
    Total = Allocated_CO2e + WTT
```

### 2.4 Asset-Specific Method (Vehicle Fleet)

```
Per-Vehicle Annual Emissions:
  Method A: Distance-Based
    CO2e = Annual_km x EF_per_km (kgCO2e/km)
    WTT = Annual_km x WTT_per_km (kgCO2e/km)

  Method B: Fuel-Based
    CO2e = Annual_Fuel_Litres x EF_per_litre (kgCO2e/litre)
    WTT = Annual_Fuel_Litres x WTT_per_litre (kgCO2e/litre)

  Method C: Electric Vehicle
    CO2e = Annual_kWh x Grid_EF (kgCO2e/kWh)
    Where: Annual_kWh = Annual_km x Consumption_kWh_per_km

Fleet Total:
  Fleet_CO2e = SUM(Vehicle_i_CO2e + Vehicle_i_WTT) for i in fleet

Vehicle EF Selection:
  Uses same DEFRA 2024 vehicle factors as Category 3/6/7:
  - 12 vehicle size/fuel combinations
  - WTT factors per vehicle type
  - Age degradation factors (new/mid/old)
```

### 2.5 Asset-Specific Method (Equipment)

```
Per-Equipment Annual Emissions:
  Method A: Energy-Based (for electric equipment)
    CO2e = Power_kW x Operating_Hours x Load_Factor x Grid_EF
    Where:
      Power_kW = rated power of equipment
      Operating_Hours = annual hours of operation
      Load_Factor = average load as fraction of rated (typically 0.4-0.8)

  Method B: Fuel-Based (for diesel/gas equipment)
    CO2e = Annual_Fuel_Litres x EF_per_litre
    WTT = Annual_Fuel_Litres x WTT_per_litre

  Method C: Output-Based (for generators)
    CO2e = Annual_kWh_generated x Generator_EF (kgCO2e/kWh)
    Where:
      Generator_EF depends on fuel type and efficiency
```

### 2.6 IT Assets Method

```
Per-IT-Asset Annual Emissions:
  Server:
    CO2e = Rated_Power_kW x PUE x 8760_hours x Utilization x Grid_EF
    Where:
      PUE = Power Usage Effectiveness (1.2-2.0, industry avg 1.58)
      Utilization = average CPU/power utilization (0.1-0.9)
      8760 = hours per year (or actual operating hours)

  Networking Equipment:
    CO2e = Rated_Power_kW x 8760 x Grid_EF (always-on assumption)

  Storage:
    CO2e = Rated_Power_kW x 8760 x Grid_EF

  Desktop/Laptop:
    CO2e = Rated_Power_kW x Operating_Hours x Grid_EF
    Where: Operating_Hours typically 2000-2500 per year (business use)

  Printer/Copier:
    CO2e = Active_Power_kW x Active_Hours x Grid_EF
         + Standby_Power_kW x Standby_Hours x Grid_EF

Co-Located Data Center Space:
  CO2e = Allocated_kW x PUE x 8760 x Grid_EF
  Where: Allocated_kW = contracted power capacity
```

### 2.7 Lessor-Specific Method

```
Lessor-Provided Emissions:
  CO2e = Lessor_Reported_CO2e x Allocation_Factor

  Validation Checks:
    - Verify lessor methodology (GHG Protocol aligned?)
    - Check boundary completeness (all Scope 1+2 emissions?)
    - Validate emission factors used (current year? reputable source?)
    - Apply allocation if shared asset
    - Document data quality tier (DQI)

  If lessor provides energy data (not emissions):
    Apply lessee's own emission factors to lessor-provided energy data
```

### 2.8 Average-Data Method

```
Building Average-Data:
  CO2e = Floor_Area_sqm x EUI_kWh_per_sqm x Grid_EF
  Where:
    EUI = Energy Use Intensity benchmark by building type and climate zone

  Building EUI Benchmarks (kWh/sqm/year):
    | Building Type | Cool Climate | Temperate | Warm Climate |
    |--------------|-------------|-----------|-------------|
    | Office | 180 | 150 | 200 |
    | Retail | 220 | 190 | 260 |
    | Warehouse | 80 | 65 | 95 |
    | Industrial | 200 | 170 | 230 |
    | Data Center | 800 | 800 | 900 |
    | Hotel | 250 | 220 | 300 |
    | Healthcare | 350 | 300 | 400 |
    | Education | 140 | 120 | 160 |

Vehicle Average-Data:
  CO2e = Number_of_Vehicles x Average_Annual_km x EF_per_km
  Where:
    Average_Annual_km by vehicle type:
      - Passenger car: 15,000 km/year
      - Light truck/van: 20,000 km/year
      - Heavy truck: 50,000 km/year
      - Specialty/construction: 5,000 km/year

Equipment Average-Data:
  CO2e = Number_of_Units x Rated_Power_kW x Default_Operating_Hours x Load_Factor x Grid_EF
  Where:
    Default_Operating_Hours by equipment type:
      - Manufacturing equipment: 4,000 hours/year (2 shifts)
      - Construction equipment: 1,500 hours/year
      - Generator: 500 hours/year (backup)
      - Office equipment: 2,500 hours/year
```

### 2.9 Spend-Based Method

```
Spend-Based Calculation:
  CO2e = Lease_Spend_USD x EEIO_Factor (kgCO2e/USD)

  With CPI deflation:
    Deflated_Spend = Lease_Spend x (CPI_EEIO_Base_Year / CPI_Reporting_Year)
    CO2e = Deflated_Spend x EEIO_Factor

  Currency conversion:
    USD_Spend = Foreign_Spend x Exchange_Rate_to_USD

  EEIO Factors by NAICS Code:
    | NAICS | Description | kgCO2e/USD |
    |-------|-----------|------------|
    | 531110 | Lessors of residential buildings | 0.19 |
    | 531120 | Lessors of nonresidential buildings | 0.22 |
    | 531130 | Lessors of miniwarehouses/self-storage | 0.18 |
    | 531190 | Lessors of other real estate property | 0.20 |
    | 532100 | Automotive equipment rental/leasing | 0.24 |
    | 532400 | Commercial/industrial machinery rental | 0.28 |
    | 532200 | Consumer goods rental | 0.21 |
    | 518210 | Data processing/hosting services | 0.35 |
    | 541500 | Computer systems design (IT leasing) | 0.16 |
    | 238000 | Specialty trade contractors (equipment) | 0.32 |
```

### 2.10 Building Energy Emission Factors

**Electricity Grid Emission Factors (IEA 2024, kgCO2e/kWh):**

| Country | Grid EF | WTT EF | Source |
|---------|---------|--------|--------|
| US (national) | 0.37170 | 0.04878 | EPA eGRID |
| GB | 0.20707 | 0.02477 | DEFRA 2024 |
| DE | 0.33800 | 0.04200 | IEA 2024 |
| FR | 0.05100 | 0.00820 | IEA 2024 |
| JP | 0.43400 | 0.05900 | IEA 2024 |
| CN | 0.55600 | 0.08900 | IEA 2024 |
| IN | 0.70800 | 0.11300 | IEA 2024 |
| AU | 0.65600 | 0.10500 | IEA 2024 |
| CA | 0.12000 | 0.01800 | IEA 2024 |
| BR | 0.07400 | 0.01200 | IEA 2024 |
| GLOBAL | 0.43600 | 0.06500 | IEA 2024 |

**Natural Gas Emission Factors (DEFRA 2024):**

| Fuel | kgCO2e/kWh | WTT (kgCO2e/kWh) | kgCO2e/therm |
|------|-----------|-------------------|-------------|
| Natural gas | 0.18316 | 0.02391 | 5.37100 |
| Heating oil | 0.24674 | 0.05757 | -- |
| LPG | 0.21449 | 0.03252 | -- |
| Coal | 0.32390 | 0.03923 | -- |
| Wood pellets | 0.01553 | 0.01264 | -- |
| District heating (EU avg) | 0.16200 | 0.02600 | -- |
| District cooling (EU avg) | 0.07100 | 0.01100 | -- |

**Vehicle Emission Factors (DEFRA 2024, kgCO2e/km):**

| Vehicle Type | Petrol | Diesel | Hybrid | BEV | WTT (petrol) |
|-------------|--------|--------|--------|-----|-------------|
| Small car | 0.14930 | 0.13105 | 0.09814 | 0.04350 | 0.03775 |
| Medium car | 0.18210 | 0.16192 | 0.12050 | 0.05020 | 0.04607 |
| Large car | 0.22180 | 0.20867 | 0.16270 | 0.06730 | 0.05613 |
| SUV | 0.20980 | 0.18790 | 0.15030 | 0.06200 | 0.05309 |
| Light van | 0.24200 | 0.22430 | -- | 0.07100 | 0.06126 |
| Heavy van | 0.33500 | 0.31200 | -- | 0.09800 | 0.08478 |
| Light truck (<3.5t) | 0.33500 | 0.31200 | -- | -- | 0.08478 |
| Heavy truck (>3.5t) | -- | 0.58600 | -- | -- | 0.08314 |

### 2.11 Double-Counting Prevention Rules

| Rule | Code | Description |
|------|------|-------------|
| DC-ULA-001 | Exclude finance leases | Finance leases (IFRS 16/ASC 842) belong in Scope 1/2, not Cat 8 |
| DC-ULA-002 | Exclude owned assets | Assets owned by the reporting company are Scope 1/2 |
| DC-ULA-003 | No overlap with Scope 2 | If electricity for a leased building is already in Scope 2, exclude from Cat 8 |
| DC-ULA-004 | No overlap with Scope 1 | If gas/fuel for a leased asset is already in Scope 1, exclude from Cat 8 |
| DC-ULA-005 | No overlap with Cat 1 | Purchased goods used in leased buildings are Cat 1, not Cat 8 |
| DC-ULA-006 | No overlap with Cat 2 | Capital goods installed in leased buildings are Cat 2, not Cat 8 |
| DC-ULA-007 | No overlap with Cat 3 | WTT for fuels: if WTT is in Cat 3, do not double-count in Cat 8 WTT |
| DC-ULA-008 | No overlap with Cat 5 | Waste from leased buildings is Cat 5, not Cat 8 |
| DC-ULA-009 | No overlap with Cat 13 | Assets leased TO others (lessor) belong in Cat 13, not Cat 8 |
| DC-ULA-010 | Sub-lease allocation | For sub-leased portions, exclude the sub-leased area (reported by sub-lessee) |

---

## 3. Architecture

### 3.1 Seven-Engine Design

```
+-------------------------------------------------------------------+
|                AGENT-MRV-021: Upstream Leased Assets               |
|                        (7-Engine Pipeline)                         |
+-------------------------------------------------------------------+
|                                                                    |
|  Engine 1: UpstreamLeasedDatabaseEngine                           |
|    - Building EUI benchmarks, grid EFs, fuel EFs, EEIO factors    |
|    - Vehicle EFs, equipment EFs, IT power ratings                  |
|    - Climate zone mapping, allocation defaults, CPI deflators      |
|                                                                    |
|  Engine 2: BuildingCalculatorEngine                                |
|    - Asset-specific (utility bills, BMS data)                      |
|    - Average-data (EUI benchmarks x floor area)                    |
|    - Multi-tenant allocation (area/headcount/revenue)              |
|    - Partial-year proration                                        |
|    - District heating/cooling, refrigerants                        |
|                                                                    |
|  Engine 3: VehicleFleetCalculatorEngine                            |
|    - Distance-based (km x EF)                                      |
|    - Fuel-based (litres x EF)                                      |
|    - Electric vehicle (kWh x grid EF)                               |
|    - Fleet aggregation, vehicle age factors                         |
|                                                                    |
|  Engine 4: EquipmentCalculatorEngine                               |
|    - Energy-based (power x hours x load factor)                    |
|    - Fuel-based (diesel/gas consumption)                           |
|    - Output-based (generators)                                     |
|    - Equipment type benchmarks                                     |
|                                                                    |
|  Engine 5: ITAssetsCalculatorEngine                                |
|    - Server emissions (PUE, utilization)                            |
|    - Network/storage equipment                                      |
|    - Desktop/laptop fleet                                           |
|    - Co-located data center allocation                              |
|                                                                    |
|  Engine 6: ComplianceCheckerEngine                                  |
|    - 7 regulatory frameworks                                        |
|    - 10 double-counting rules (DC-ULA-001 through DC-ULA-010)      |
|    - Lease classification validation (operating vs finance)         |
|    - Disclosure completeness checks                                 |
|                                                                    |
|  Engine 7: UpstreamLeasedPipelineEngine                            |
|    - 10-stage pipeline orchestration                                |
|    - Validate -> Classify -> Normalize -> Resolve EFs ->            |
|      Calculate -> Allocate -> Aggregate -> Compliance ->            |
|      Provenance -> Seal                                             |
+-------------------------------------------------------------------+
```

### 3.2 Pipeline Stages

| Stage | Name | Description |
|-------|------|-------------|
| 1 | VALIDATE | Validate asset inventory, lease types, energy data, floor areas |
| 2 | CLASSIFY | Classify lease type (operating/finance), asset type, energy sources |
| 3 | NORMALIZE | Unit conversions (sqft->sqm, therms->kWh, gallons->litres), currency conversion |
| 4 | RESOLVE_EFS | Resolve emission factors: grid EFs, fuel EFs, EUI benchmarks, EEIO factors |
| 5 | CALCULATE | Calculate emissions per asset using appropriate method |
| 6 | ALLOCATE | Apply multi-tenant allocation and partial-year proration |
| 7 | AGGREGATE | Aggregate by asset type, building type, geography, energy source |
| 8 | COMPLIANCE | Check 7 regulatory frameworks, 10 DC rules, disclosure completeness |
| 9 | PROVENANCE | SHA-256 chain hashing, Merkle root, stage-level audit trail |
| 10 | SEAL | Seal calculation with final hash, timestamp, immutable flag |

---

## 4. Data Models

### 4.1 Enumerations (22 total)

| Enum | Values | Description |
|------|--------|-------------|
| CalculationMethod | asset_specific, lessor_specific, average_data, spend_based | GHG Protocol methods |
| LeaseType | operating, finance | IFRS 16/ASC 842 classification |
| AssetCategory | building, vehicle, equipment, it_asset | Top-level asset type |
| BuildingType | office, retail, warehouse, industrial, data_center, hotel, healthcare, education | 8 building types |
| VehicleType | small_car, medium_car, large_car, suv, light_van, heavy_van, light_truck, heavy_truck | 8 vehicle types |
| FuelType | petrol, diesel, hybrid, bev, lpg, cng, hydrogen | 7 fuel types |
| EquipmentType | manufacturing, construction, generator, agricultural, mining, hvac | 6 equipment types |
| ITAssetType | server, network_switch, storage, desktop, laptop, printer, copier | 7 IT asset types |
| EnergySource | electricity, natural_gas, heating_oil, lpg, coal, district_heating, district_cooling, wood_pellets, on_site_solar | 9 energy sources |
| AllocationMethod | floor_area, headcount, revenue, equal_share, custom | 5 allocation methods |
| ClimateZone | tropical, arid, temperate, continental, polar | 5 ASHRAE climate zones |
| EFSource | defra_2024, iea_2024, epa_egrid, ipcc_ar6, energy_star, custom | 6 EF sources |
| ComplianceFramework | ghg_protocol, iso_14064, csrd_esrs, cdp, sbti, sb_253, gri | 7 frameworks |
| DataQualityTier | measured, calculated, estimated | 3 tiers |
| ProvenanceStage | validate, classify, normalize, resolve_efs, calculate, allocate, aggregate, compliance, provenance, seal | 10 stages |
| UncertaintyMethod | monte_carlo, analytical, ipcc_tier2 | 3 methods |
| DQIDimension | reliability, completeness, temporal, geographical, technological | 5 DQI dimensions |
| DQIScore | very_good, good, fair, poor, very_poor | 5 quality scores |
| ComplianceStatus | pass, fail, warning | 3 statuses |
| GWPVersion | ar4, ar5, ar6, ar6_20yr | 4 IPCC versions |
| EmissionGas | co2, ch4, n2o | 3 GHGs |
| CurrencyCode | USD, EUR, GBP, CAD, AUD, JPY, CNY, INR, CHF, BRL, ZAR, KRW | 12 currencies |

### 4.2 Pydantic Input Models (12 total)

| Model | Key Fields | Frozen |
|-------|-----------|--------|
| BuildingAssetInput | building_type, floor_area_sqm, electricity_kwh, gas_kwh, district_heating_kwh, country_code, allocation_factor, lease_months | Yes |
| VehicleAssetInput | vehicle_type, fuel_type, annual_km, fuel_litres, count, country_code | Yes |
| EquipmentAssetInput | equipment_type, power_kw, operating_hours, load_factor, fuel_type, fuel_litres | Yes |
| ITAssetInput | it_type, power_kw, count, pue, utilization, operating_hours, country_code | Yes |
| LessorDataInput | asset_id, lessor_name, reported_co2e_kg, methodology, boundary, allocation_factor | Yes |
| SpendInput | naics_code, amount, currency, reporting_year, lease_type | Yes |
| AssetInventoryInput | assets (list of mixed types), reporting_year, organization_id, consolidation_approach | Yes |
| AllocationInput | total_area_sqm, leased_area_sqm, method, headcount_total, headcount_lessee | Yes |
| BatchAssetInput | items (list), max_items, parallel | Yes |
| ComplianceCheckInput | frameworks, total_co2e, method_used, reporting_period | Yes |
| UncertaintyInput | method, iterations, confidence_level | Yes |
| PortfolioInput | buildings, vehicles, equipment, it_assets, reporting_year | Yes |

### 4.3 Constant Tables (14 total)

| Table | Entries | Key Fields | Source |
|-------|---------|-----------|--------|
| BUILDING_EUI_BENCHMARKS | 40 (8 types x 5 climate zones) | building_type, climate_zone, eui_kwh_sqm, gas_fraction | ENERGY STAR / ASHRAE |
| GRID_EMISSION_FACTORS | 11 countries + 26 eGRID | country_code, ef_per_kwh, wtt_per_kwh | IEA 2024 / eGRID |
| FUEL_EMISSION_FACTORS | 8 fuels | fuel_type, ef_per_kwh, ef_per_litre, wtt | DEFRA 2024 |
| VEHICLE_EMISSION_FACTORS | 32 (8 types x 4 fuels) | vehicle_type, fuel_type, ef_per_km, wtt | DEFRA 2024 |
| EQUIPMENT_BENCHMARKS | 6 types | equipment_type, default_hours, load_factor | Industry standards |
| IT_POWER_RATINGS | 7 types | it_type, typical_power_kw, utilization | EPA / Industry |
| EEIO_FACTORS | 10 NAICS codes | naics_code, description, ef_per_usd | EPA USEEIO v2.0 |
| CURRENCY_RATES | 12 currencies | currency_code, usd_rate | XE / World Bank |
| CPI_DEFLATORS | 11 years (2015-2025) | year, deflator | BLS / World Bank |
| ALLOCATION_DEFAULTS | 5 methods | method, description, precedence | GHG Protocol |
| CLIMATE_ZONE_MAP | 50+ countries | country_code, default_zone | ASHRAE 169 |
| REFRIGERANT_GWPS | 15 refrigerants | refrigerant, gwp_ar6, gwp_ar5 | IPCC AR6 |
| DISTRICT_HEATING_EFS | 8 regions | region, ef_kwh, wtt | DEFRA / IEA |
| DQI_SCORING | 5 dimensions | dimension, weight, scale_1_5 | GHG Protocol |

---

## 5. API Endpoints (22 total)

### Calculations (10 POST endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/calculate` | Full asset-inventory pipeline calculation |
| POST | `/calculate/building` | Single building calculation |
| POST | `/calculate/vehicle` | Single vehicle/fleet calculation |
| POST | `/calculate/equipment` | Single equipment calculation |
| POST | `/calculate/it-asset` | Single IT asset calculation |
| POST | `/calculate/lessor` | Lessor-specific method calculation |
| POST | `/calculate/spend` | Spend-based EEIO calculation |
| POST | `/calculate/batch` | Batch multi-asset calculation |
| POST | `/calculate/portfolio` | Full portfolio calculation |
| POST | `/compliance/check` | Compliance check against frameworks |

### Data Access (10 GET endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/calculations/{calc_id}` | Get calculation by ID |
| GET | `/calculations` | List calculations with filters |
| DELETE | `/calculations/{calc_id}` | Delete calculation |
| GET | `/emission-factors/{ef_type}` | Get emission factors by type |
| GET | `/building-benchmarks` | Get EUI benchmarks |
| GET | `/grid-factors/{country}` | Get grid emission factors |
| GET | `/lease-classification` | Get lease type guidance |
| GET | `/aggregations` | Get aggregated results |
| GET | `/provenance/{calc_id}` | Get provenance chain |
| GET | `/health` | Health check |

### Analysis (2 POST endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/uncertainty/analyze` | Uncertainty analysis |
| POST | `/portfolio/analyze` | Portfolio-level analysis |

---

## 6. Database Schema (V072)

### Tables (16 total)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| gl_ula_calculations | Master calculation records | id, org_id, method, total_co2e, reporting_year |
| gl_ula_building_assets | Leased building inventory | id, calc_id, building_type, floor_area_sqm, country_code |
| gl_ula_building_emissions | Per-building emissions | id, asset_id, elec_co2e, gas_co2e, heating_co2e, wtt |
| gl_ula_vehicle_assets | Leased vehicle inventory | id, calc_id, vehicle_type, fuel_type, annual_km, count |
| gl_ula_vehicle_emissions | Per-vehicle emissions | id, asset_id, ttw_co2e, wtt_co2e, method |
| gl_ula_equipment_assets | Leased equipment inventory | id, calc_id, equipment_type, power_kw, hours |
| gl_ula_equipment_emissions | Per-equipment emissions | id, asset_id, co2e, method |
| gl_ula_it_assets | Leased IT asset inventory | id, calc_id, it_type, power_kw, pue, count |
| gl_ula_it_emissions | Per-IT-asset emissions | id, asset_id, co2e, annual_kwh |
| gl_ula_allocations | Multi-tenant allocations | id, calc_id, asset_id, method, factor, allocated_co2e |
| gl_ula_compliance_results | Compliance check results | id, calc_id, framework, status, findings |
| gl_ula_emission_factors | EF audit log | id, calc_id, ef_type, source, value, year |
| gl_ula_provenance | Provenance chain entries | id, calc_id, stage, input_hash, output_hash |
| gl_ula_aggregations | Aggregated results | id, calc_id, dimension, group_key, co2e |
| gl_ula_spend_calculations | Spend-based calculation details | id, calc_id, naics_code, amount, co2e |
| gl_ula_lessor_data | Lessor-provided emissions data | id, calc_id, lessor_name, reported_co2e, validated |

### Hypertables (3)
- `gl_ula_calculations` (partitioned by `created_at`)
- `gl_ula_building_emissions` (partitioned by `created_at`)
- `gl_ula_vehicle_emissions` (partitioned by `created_at`)

### Continuous Aggregates (2)
- `gl_ula_daily_emissions` -- Daily emissions rollup by asset type
- `gl_ula_monthly_portfolio` -- Monthly portfolio-level aggregation

---

## 7. Compliance Frameworks

### 7.1 Framework Requirements

| Framework | Key Requirements for Category 8 |
|-----------|-------------------------------|
| GHG Protocol | Lease classification (operating vs finance), all Scope 1+2 equivalent, allocation documented |
| ISO 14064 | Quantification methodology documented, uncertainty assessment, base year |
| CSRD ESRS E1 | Energy consumption in leased assets, total Cat 8 CO2e, decarbonization targets |
| CDP | C6.3 upstream leased assets disclosure, methodology, data quality assessment |
| SBTi | Include if >1% of Scope 3, calculate with approved methods |
| SB 253 | Full Category 8 disclosure if material, third-party assurance |
| GRI 305 | Disclosure 305-3 with methodology, source of EFs, GWP version |

### 7.2 Compliance Rules per Framework

**GHG Protocol (9 rules: GHG-ULA-001 to GHG-ULA-009):**
- Lease classification documented
- Calculation methodology documented
- EF sources identified
- Allocation method justified
- WTT included
- Scope 1/2 boundary confirmed
- Data quality assessed
- Base year recalculation triggered on structural changes
- Organizational boundary aligned with consolidation approach

**ISO 14064 (7 rules):** Quantification, uncertainty, base year, completeness, consistency, transparency, GWP version

**CSRD ESRS E1 (8 rules):** Energy consumption, total CO2e, methodology, targets, data quality, double materiality, assurance readiness, DNSH assessment

**CDP (6 rules):** C6.3 total, methodology, data quality, boundary, reduction targets, trend analysis

**SBTi (6 rules):** Materiality threshold (>1%), methodology alignment, pathway, reduction target, coverage, recalculation

**SB 253 (6 rules):** Third-party assurance, methodology documentation, Category completeness, data quality, reporting timeline, correction/restatement

**GRI 305 (7 rules):** 305-3 disclosure, biogenic separate, methodology documented, EF sources, GWP version, base year, exclusions justified

---

## 8. File Manifest

### Source Files (15 files)

| # | File | Lines (est.) | Description |
|---|------|-------------|-------------|
| 1 | `__init__.py` | ~120 | Package init with graceful imports for 7 engines |
| 2 | `models.py` | ~2,800 | 22 enums, 14 constant tables, 12 Pydantic models |
| 3 | `config.py` | ~2,900 | GL_ULA_ env prefix, 15 config sections, singleton |
| 4 | `metrics.py` | ~2,100 | Prometheus metrics with gl_ula_ prefix |
| 5 | `provenance.py` | ~3,000 | SHA-256 chain-hashed audit trails, 10-stage pipeline |
| 6 | `upstream_leased_database.py` | ~2,500 | Engine 1: EF lookups for buildings, vehicles, equipment, IT |
| 7 | `building_calculator.py` | ~2,800 | Engine 2: Building emissions (asset-specific + average-data) |
| 8 | `vehicle_fleet_calculator.py` | ~2,500 | Engine 3: Vehicle fleet (distance + fuel + EV) |
| 9 | `equipment_calculator.py` | ~2,200 | Engine 4: Equipment (energy + fuel + output) |
| 10 | `it_assets_calculator.py` | ~2,200 | Engine 5: IT assets (servers, network, storage, desktop) |
| 11 | `compliance_checker.py` | ~3,200 | Engine 6: 7 frameworks, 10 DC rules |
| 12 | `upstream_leased_pipeline.py` | ~2,700 | Engine 7: 10-stage orchestration |
| 13 | `api/__init__.py` | ~2 | API package init |
| 14 | `api/router.py` | ~2,900 | 22 REST endpoints, request/response models |
| 15 | `setup.py` | ~2,700 | Service facade wiring 7 engines, get_service() |

### Test Files (14 files)

| # | File | Tests (est.) | Description |
|---|------|-------------|-------------|
| 1 | `conftest.py` | -- | Fixtures for all asset types, compliance, config |
| 2 | `test_models.py` | ~220 | 22 enums, 14 constant tables, 12 Pydantic models |
| 3 | `test_config.py` | ~140 | 15 config sections, singleton, env vars |
| 4 | `test_upstream_leased_database.py` | ~90 | EF lookups for all asset types |
| 5 | `test_building_calculator.py` | ~50 | Building calculations, allocation |
| 6 | `test_vehicle_fleet_calculator.py` | ~40 | Vehicle fleet calculations |
| 7 | `test_equipment_calculator.py` | ~35 | Equipment calculations |
| 8 | `test_it_assets_calculator.py` | ~35 | IT asset calculations |
| 9 | `test_provenance.py` | ~95 | SHA-256, chain integrity, stages |
| 10 | `test_compliance_checker.py` | ~55 | 7 frameworks, DC rules |
| 11 | `test_upstream_leased_pipeline.py` | ~55 | 10 stages, batch |
| 12 | `test_api.py` | ~40 | 22 endpoints |
| 13 | `test_setup.py` | ~30 | Service facade |
| 14 | `__init__.py` | -- | Test package init |

**Total: 15 source files (~32K lines) + 14 test files (~700+ tests)**

### Database Migration
- `V072__upstream_leased_assets_service.sql` (~950 lines)

---

## 9. Auth Integration

### Permission Map Entries
```
POST:/api/v1/upstream-leased-assets/calculate             -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/building     -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/vehicle      -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/equipment    -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/it-asset     -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/lessor       -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/spend        -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/batch        -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/calculate/portfolio    -> upstream-leased-assets:calculate
POST:/api/v1/upstream-leased-assets/compliance/check       -> upstream-leased-assets:compliance
GET:/api/v1/upstream-leased-assets/calculations/{calc_id}  -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/calculations            -> upstream-leased-assets:read
DELETE:/api/v1/upstream-leased-assets/calculations/{calc_id} -> upstream-leased-assets:delete
GET:/api/v1/upstream-leased-assets/emission-factors/{type} -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/building-benchmarks     -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/grid-factors/{country}  -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/lease-classification    -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/aggregations            -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/provenance/{calc_id}    -> upstream-leased-assets:read
GET:/api/v1/upstream-leased-assets/health                  -> upstream-leased-assets:read
POST:/api/v1/upstream-leased-assets/uncertainty/analyze    -> upstream-leased-assets:analyze
POST:/api/v1/upstream-leased-assets/portfolio/analyze      -> upstream-leased-assets:analyze
```

---

## 10. Testing Strategy

- Unit tests per engine (Engines 1-7)
- Integration tests for full pipeline
- Known-value verification against hand-calculated examples
- Double-counting rule validation
- Boundary test cases (operating vs finance lease)
- Multi-tenant allocation accuracy
- Partial-year proration correctness
- Currency conversion and CPI deflation
- Edge cases: zero floor area, zero vehicles, negative inputs
- Provenance hash determinism
- Thread-safety of singleton engines
- Compliance framework completeness

---

## 11. Dependencies

- **Internal**: greenlang.infrastructure (auth, RBAC, Redis, PostgreSQL)
- **Python**: pydantic, decimal, hashlib, logging, asyncio
- **Database**: PostgreSQL + TimescaleDB (V072 migration)
- **Monitoring**: Prometheus client (gl_ula_ metrics)
- **API**: FastAPI, uvicorn
- **No external API calls** -- all emission factors are embedded in models.py
