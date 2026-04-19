# PRD: AGENT-MRV-024 — Use of Sold Products Agent (GL-MRV-S3-011)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-011 |
| **Component** | AGENT-MRV-024 |
| **Category** | Scope 3, Category 11: Use of Sold Products |
| **Version** | 1.0.0 |
| **Table Prefix** | `gl_usp_` |
| **Module** | `greenlang/use_of_sold_products/` |
| **Tests** | `tests/unit/mrv/test_use_of_sold_products/` |
| **Migration** | `V075__use_of_sold_products_service.sql` |
| **Standard** | GHG Protocol Scope 3 Standard, Chapter 6 |

## 2. Scope & Applicability

Category 11 covers **total expected lifetime emissions** from the USE of goods and services sold by the reporting company in the reporting period. This is often the LARGEST Scope 3 category for manufacturers of energy-consuming products (vehicles, appliances, electronics, HVAC).

**Two emission types:**
- **Direct use-phase emissions**: GHG released directly during product use (e.g., fuel combustion in vehicles, refrigerant leakage from HVAC)
- **Indirect use-phase emissions**: GHG from generation of energy consumed during product use (e.g., electricity for appliances, heating fuel for furnaces)

**Applicable when:**
- Company sells products that consume energy during use (electricity, fuel, gas)
- Company sells products that directly emit GHGs during use (vehicles, chemical products, refrigeration)
- Company sells fuels or feedstocks that release GHGs when combusted/used

**NOT applicable when:**
- Product does not consume energy or emit GHGs during use (passive products like furniture, clothing)
- Emissions occur during manufacturing (→ Cat 1 or Cat 10)
- Emissions occur at end-of-life (→ Cat 12)

### 2.1 Product Use Categories (10)
| Code | Category | Direct | Indirect | Examples |
|------|----------|--------|----------|----------|
| `VEHICLES` | Vehicles & Transportation | Yes | No | Cars, trucks, motorcycles, aircraft engines |
| `APPLIANCES` | Appliances & Electronics | No | Yes | Refrigerators, washing machines, dishwashers |
| `HVAC` | HVAC & Refrigeration | Yes | Yes | Air conditioners, heat pumps, chillers |
| `LIGHTING` | Lighting Products | No | Yes | LED bulbs, fluorescent tubes, smart lighting |
| `IT_EQUIPMENT` | IT & Office Equipment | No | Yes | Computers, servers, printers, monitors |
| `INDUSTRIAL_EQUIPMENT` | Industrial Machinery | Yes | Yes | Generators, compressors, boilers, turbines |
| `FUELS_FEEDSTOCKS` | Fuels & Feedstocks | Yes | No | Gasoline, diesel, natural gas, coal, LPG |
| `BUILDING_PRODUCTS` | Building & Insulation | No | Yes | Windows, insulation, HVAC ducts |
| `CONSUMER_PRODUCTS` | Consumer Chemicals | Yes | No | Aerosols, solvents, fertilizers, cleaning agents |
| `MEDICAL_DEVICES` | Medical Devices | No | Yes | Imaging equipment, ventilators, lab equipment |

### 2.2 Use-Phase Emission Types (6)
| Code | Type | Description |
|------|------|-------------|
| `DIRECT_FUEL_COMBUSTION` | Direct — Fuel Combustion | GHG from burning fuel in product (vehicles, generators) |
| `DIRECT_REFRIGERANT_LEAKAGE` | Direct — Refrigerant Leakage | GHG from refrigerant/F-gas leaks (HVAC, fridges) |
| `DIRECT_CHEMICAL_RELEASE` | Direct — Chemical Release | GHG from chemical products during use (aerosols, fertilizers) |
| `INDIRECT_ELECTRICITY` | Indirect — Electricity | GHG from electricity consumed during use (appliances, IT) |
| `INDIRECT_FUEL_HEATING` | Indirect — Fuel for Heating | GHG from fuel for heating during use (furnaces, boilers) |
| `INDIRECT_STEAM_COOLING` | Indirect — Steam/Cooling | GHG from district heating/cooling consumed during use |

## 3. Seven-Engine Architecture

### Engine 1: ProductUseDatabaseEngine
- **Purpose**: Emission factor storage and retrieval for use-phase calculations
- **Key data**: Product energy consumption profiles, fuel EFs, grid EFs (130+ countries), refrigerant GWPs (AR5/AR6), product lifetime estimates, annual usage profiles
- **Lookups**: By product category, energy type, country, fuel type, refrigerant type

### Engine 2: DirectEmissionsCalculatorEngine
- **Purpose**: Calculate direct use-phase emissions (fuel combustion, refrigerant leakage, chemical release)
- **Methods**:
  - Fuel combustion: `E = Σ(units_sold × lifetime × annual_fuel × fuel_EF)`
  - Refrigerant leakage: `E = Σ(units_sold × charge × annual_leak_rate × GWP × lifetime)`
  - Chemical release: `E = Σ(units_sold × content × release_fraction × GWP)`
- **DQI**: High for product-specific data (70-90)

### Engine 3: IndirectEmissionsCalculatorEngine
- **Purpose**: Calculate indirect use-phase emissions (electricity, heating fuel, steam/cooling)
- **Methods**:
  - Electricity: `E = Σ(units_sold × lifetime × annual_energy_kWh × grid_EF)`
  - Heating fuel: `E = Σ(units_sold × lifetime × annual_fuel × fuel_EF)`
  - Steam/cooling: `E = Σ(units_sold × lifetime × annual_steam_MJ × steam_EF)`
- **DQI**: Medium-high (60-85)

### Engine 4: FuelsAndFeedstocksCalculatorEngine
- **Purpose**: Calculate emissions from sold fuels/feedstocks when combusted by end users
- **Methods**:
  - Fuel sales: `E = Σ(volume_sold × combustion_EF)`
  - Feedstock: `E = Σ(mass_sold × process_EF × oxidation_factor)`
- **Data**: 15 fuel types with NCV and combustion EFs
- **DQI**: High (75-95) — well-characterized EFs

### Engine 5: LifetimeModelingEngine
- **Purpose**: Product lifetime estimation, usage patterns, and degradation modeling
- **Features**:
  - Default lifetime tables by product category (years)
  - Annual usage profiles (hours/year, km/year, cycles/year)
  - Energy degradation over lifetime (efficiency loss curves)
  - Replacement/repair impact on lifetime extension
  - Fleet survival curves (Weibull distribution)
  - Discounting for future emissions (optional)

### Engine 6: ComplianceCheckerEngine
- **Purpose**: Validate against 7 regulatory frameworks
- **Frameworks**: GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305
- **Rules**: ~50 compliance rules, 8 double-counting prevention rules
- **Features**: Lifetime assumption validation, boundary checks, completeness scoring

### Engine 7: UseOfSoldProductsPipelineEngine
- **Purpose**: 10-stage orchestration pipeline
- **Stages**: validate → classify → normalize → resolve_efs → calculate → lifetime → aggregate → compliance → provenance → seal
- **Features**: Dual-path (direct + indirect), portfolio analysis, product-level breakdown

## 4. Calculation Methods

### 4.1 Direct Use-Phase Emissions

**Formula A — Fuel Combustion (Vehicles, Generators):**
```
E_direct_fuel = Σᵢ (Q_sold_i × L_i × AU_fuel_i × EF_fuel)
where:
  Q_sold_i   = units of product i sold in reporting period
  L_i        = expected lifetime of product i (years)
  AU_fuel_i  = annual fuel consumption of product i (liters/year or m³/year)
  EF_fuel    = emission factor for fuel type (kgCO2e/liter or kgCO2e/m³)
```

**Formula B — Refrigerant Leakage (HVAC, Refrigeration):**
```
E_direct_ref = Σᵢ (Q_sold_i × C_i × ALR_i × GWP_ref × L_i)
where:
  C_i        = refrigerant charge per unit (kg)
  ALR_i      = annual leakage rate (fraction, typically 2-10%)
  GWP_ref    = global warming potential of refrigerant (100-year)
  L_i        = lifetime (years)
```

**Formula C — Chemical Release:**
```
E_direct_chem = Σᵢ (Q_sold_i × M_ghg_i × RF_i × GWP_ghg)
where:
  M_ghg_i    = mass of GHG-containing substance per unit (kg)
  RF_i       = release fraction during use (0-1)
  GWP_ghg    = GWP of the released gas
```

### 4.2 Indirect Use-Phase Emissions

**Formula D — Electricity Consumption:**
```
E_indirect_elec = Σᵢ (Q_sold_i × L_i × AE_i × EF_grid)
where:
  AE_i       = annual electricity consumption (kWh/year)
  EF_grid    = grid emission factor for use-region (kgCO2e/kWh)
```

**Formula E — Heating Fuel:**
```
E_indirect_heat = Σᵢ (Q_sold_i × L_i × AF_heat_i × EF_fuel)
where:
  AF_heat_i  = annual heating fuel consumption (liters/year or m³/year)
```

**Formula F — Steam/Cooling:**
```
E_indirect_steam = Σᵢ (Q_sold_i × L_i × AS_i × EF_steam)
where:
  AS_i       = annual steam/cooling consumption (MJ/year)
  EF_steam   = emission factor for steam/cooling (kgCO2e/MJ)
```

### 4.3 Fuels & Feedstocks Sold

**Formula G — Fuel Combustion by End Users:**
```
E_fuels = Σⱼ (V_sold_j × EF_combustion_j)
where:
  V_sold_j      = volume of fuel j sold (liters, m³, or tonnes)
  EF_combustion_j = combustion emission factor for fuel j (kgCO2e/unit)
```

**Formula H — Feedstock Oxidation:**
```
E_feedstock = Σⱼ (M_sold_j × C_content_j × OF_j × 44/12)
where:
  C_content_j = carbon content fraction
  OF_j        = oxidation factor (fraction combusted, typically 0.95-1.0)
  44/12       = molecular weight ratio CO2/C
```

## 5. Emission Factor Tables

### 5.1 Product Energy Profiles (Default Lifetime & Annual Consumption)
| Product Category | Default Lifetime (years) | Annual Use (kWh or liters) | Unit |
|-----------------|------------------------|--------------------------|------|
| `VEHICLES` — Passenger Car (gasoline) | 15 | 1,200 L/yr | liters/year |
| `VEHICLES` — Passenger Car (diesel) | 15 | 1,000 L/yr | liters/year |
| `VEHICLES` — Passenger Car (EV) | 15 | 3,500 kWh/yr | kWh/year |
| `VEHICLES` — Light Truck | 15 | 1,800 L/yr | liters/year |
| `VEHICLES` — Heavy Truck | 10 | 30,000 L/yr | liters/year |
| `VEHICLES` — Motorcycle | 12 | 500 L/yr | liters/year |
| `APPLIANCES` — Refrigerator | 15 | 400 kWh/yr | kWh/year |
| `APPLIANCES` — Washing Machine | 12 | 200 kWh/yr | kWh/year |
| `APPLIANCES` — Dishwasher | 12 | 290 kWh/yr | kWh/year |
| `APPLIANCES` — Dryer | 12 | 550 kWh/yr | kWh/year |
| `APPLIANCES` — Oven/Range | 15 | 320 kWh/yr | kWh/year |
| `HVAC` — Room AC | 12 | 1,200 kWh/yr | kWh/year |
| `HVAC` — Central AC | 15 | 3,500 kWh/yr | kWh/year |
| `HVAC` — Heat Pump | 15 | 4,000 kWh/yr | kWh/year |
| `HVAC` — Gas Furnace | 20 | 1,500 m³/yr | m³/year |
| `LIGHTING` — LED Bulb | 15 | 10 kWh/yr | kWh/year |
| `LIGHTING` — CFL Bulb | 8 | 14 kWh/yr | kWh/year |
| `IT_EQUIPMENT` — Laptop | 5 | 50 kWh/yr | kWh/year |
| `IT_EQUIPMENT` — Desktop | 6 | 200 kWh/yr | kWh/year |
| `IT_EQUIPMENT` — Server | 5 | 4,500 kWh/yr | kWh/year |
| `IT_EQUIPMENT` — Monitor | 7 | 80 kWh/yr | kWh/year |
| `INDUSTRIAL_EQUIPMENT` — Diesel Generator | 15 | 20,000 L/yr | liters/year |
| `INDUSTRIAL_EQUIPMENT` — Gas Boiler | 20 | 25,000 m³/yr | m³/year |
| `INDUSTRIAL_EQUIPMENT` — Compressor | 15 | 15,000 kWh/yr | kWh/year |

### 5.2 Fuel Combustion Emission Factors (kgCO2e/unit)
| Fuel | Code | EF (kgCO2e/L or m³ or kg) | NCV (MJ/unit) |
|------|------|--------------------------|---------------|
| Gasoline | `GASOLINE` | 2.315 | 34.2 |
| Diesel | `DIESEL` | 2.706 | 38.6 |
| Natural Gas (m³) | `NATURAL_GAS` | 2.024 | 38.3 |
| LPG | `LPG` | 1.557 | 26.1 |
| Kerosene | `KEROSENE` | 2.541 | 37.0 |
| Heavy Fuel Oil | `HFO` | 3.114 | 40.4 |
| Aviation Fuel | `JET_FUEL` | 2.548 | 37.4 |
| Ethanol | `ETHANOL` | 0.020 | 26.7 |
| Biodiesel | `BIODIESEL` | 0.015 | 37.0 |
| Coal (kg) | `COAL` | 2.883 | 25.8 |
| Wood pellets (kg) | `WOOD_PELLETS` | 0.015 | 17.0 |
| Propane | `PROPANE` | 1.530 | 25.3 |
| Hydrogen (kg) | `HYDROGEN` | 0.000 | 120.0 |
| CNG (m³) | `CNG` | 2.024 | 38.3 |
| LNG (kg) | `LNG` | 2.750 | 49.5 |

### 5.3 Refrigerant GWPs (100-year, AR5 & AR6)
| Refrigerant | Code | GWP-AR5 | GWP-AR6 | Typical Charge (kg) | Annual Leak Rate |
|-------------|------|---------|---------|--------------------|-----------------|
| R-134a | `R134A` | 1,430 | 1,530 | 0.15-3.0 | 3-8% |
| R-410A | `R410A` | 2,088 | 2,088 | 1.5-5.0 | 3-6% |
| R-32 | `R32` | 675 | 771 | 0.8-3.0 | 2-5% |
| R-290 (Propane) | `R290` | 3 | 0.02 | 0.1-0.5 | 1-3% |
| R-404A | `R404A` | 3,922 | 3,922 | 2.0-8.0 | 5-15% |
| R-407C | `R407C` | 1,774 | 1,774 | 1.5-5.0 | 3-8% |
| R-507A | `R507A` | 3,985 | 3,985 | 2.0-8.0 | 5-15% |
| R-1234yf | `R1234YF` | 4 | 0.501 | 0.3-1.0 | 2-5% |
| R-1234ze | `R1234ZE` | 7 | 1.37 | 0.5-2.0 | 2-4% |
| R-744 (CO2) | `R744` | 1 | 1 | 0.5-5.0 | 2-10% |

### 5.4 Grid Emission Factors (kgCO2e/kWh) — Same as MRV-023
16 regions: US=0.417, GB=0.233, DE=0.348, FR=0.052, CN=0.555, IN=0.708, JP=0.462, KR=0.424, BR=0.075, CA=0.120, AU=0.656, MX=0.431, IT=0.256, ES=0.175, PL=0.635, GLOBAL=0.475

### 5.5 Product Lifetime Adjustment Factors
| Factor | Code | Multiplier | Description |
|--------|------|------------|-------------|
| Standard use | `STANDARD` | 1.00 | Default assumption |
| Heavy use | `HEAVY` | 0.80 | Reduced lifetime (commercial/fleet) |
| Light use | `LIGHT` | 1.20 | Extended lifetime (light residential) |
| Industrial 24/7 | `INDUSTRIAL` | 0.60 | Continuous industrial use |
| Seasonal | `SEASONAL` | 0.50 | Only used part of year (AC in temperate) |

### 5.6 Energy Degradation Curves
| Product Category | Annual Degradation (%) | Applies To |
|-----------------|----------------------|------------|
| `VEHICLES` | 1.5% | Fuel efficiency loss over time |
| `APPLIANCES` | 0.5% | Energy efficiency decline |
| `HVAC` | 1.0% | Refrigerant charge depletion, compressor wear |
| `LIGHTING` | 2.0% | Lumen depreciation |
| `IT_EQUIPMENT` | 0.0% | Typically constant until failure |
| `INDUSTRIAL_EQUIPMENT` | 1.0% | Mechanical wear, efficiency loss |

### 5.7 Data Quality Indicator Dimensions
Same 5-dimension DQI as MRV-023 (reliability, completeness, temporal, geographical, technological).

### 5.8 Uncertainty Ranges by Method
| Method | Min | Default | Max |
|--------|-----|---------|-----|
| Direct (product-specific fuel data) | ±10% | ±15% | ±25% |
| Direct (refrigerant, product-specific) | ±10% | ±20% | ±35% |
| Direct (chemical release) | ±15% | ±25% | ±40% |
| Indirect (product-specific energy) | ±10% | ±20% | ±30% |
| Indirect (average energy profile) | ±20% | ±30% | ±50% |
| Fuels & Feedstocks | ±5% | ±10% | ±15% |

## 6. Double-Counting Prevention Rules (8)

| Rule ID | Rule | Description |
|---------|------|-------------|
| DC-USP-001 | vs Scope 1 | Exclude direct emissions from own use of products |
| DC-USP-002 | vs Scope 2 | Exclude electricity from own product use |
| DC-USP-003 | vs Cat 1 | No overlap with upstream production |
| DC-USP-004 | vs Cat 3 | No overlap with fuel & energy activities |
| DC-USP-005 | vs Cat 10 | No overlap with processing (pre-use) |
| DC-USP-006 | vs Cat 12 | No overlap with end-of-life (post-use) |
| DC-USP-007 | vs Cat 13 | No overlap with downstream leased assets |
| DC-USP-008 | Fuel double-count | Don't count fuel sold if already counted in vehicle use-phase |

## 7. Compliance Frameworks (7)

| Framework | Key Requirements |
|-----------|-----------------|
| **GHG Protocol** | Scope 3 Ch. 6 Cat 11, lifetime assumptions, direct vs indirect split |
| **ISO 14064-1** | Clause 5.2.4, methodology documentation, verification |
| **CSRD ESRS E1** | E1-6 Scope 3, value chain emissions, DNSH |
| **CDP** | C6.5 Cat 11, methodology, data quality |
| **SBTi** | Target coverage ≥67%, base year recalculation |
| **SB 253** | California Climate Disclosure Act |
| **GRI 305** | 305-3 other indirect GHG |

## 8. API Endpoints (22)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/api/v1/use-of-sold-products/calculate` | Full pipeline calculation |
| 2 | POST | `/api/v1/use-of-sold-products/calculate/direct/fuel` | Direct fuel combustion |
| 3 | POST | `/api/v1/use-of-sold-products/calculate/direct/refrigerant` | Direct refrigerant leakage |
| 4 | POST | `/api/v1/use-of-sold-products/calculate/direct/chemical` | Direct chemical release |
| 5 | POST | `/api/v1/use-of-sold-products/calculate/indirect/electricity` | Indirect electricity |
| 6 | POST | `/api/v1/use-of-sold-products/calculate/indirect/heating` | Indirect heating fuel |
| 7 | POST | `/api/v1/use-of-sold-products/calculate/indirect/steam` | Indirect steam/cooling |
| 8 | POST | `/api/v1/use-of-sold-products/calculate/fuels` | Fuels & feedstocks sold |
| 9 | POST | `/api/v1/use-of-sold-products/calculate/batch` | Batch calculation |
| 10 | POST | `/api/v1/use-of-sold-products/calculate/portfolio` | Portfolio analysis |
| 11 | POST | `/api/v1/use-of-sold-products/compliance/check` | Compliance validation |
| 12 | GET | `/api/v1/use-of-sold-products/calculations/{id}` | Get calculation by ID |
| 13 | GET | `/api/v1/use-of-sold-products/calculations` | List calculations |
| 14 | DELETE | `/api/v1/use-of-sold-products/calculations/{id}` | Delete calculation |
| 15 | GET | `/api/v1/use-of-sold-products/emission-factors/{category}` | Get EFs by product category |
| 16 | GET | `/api/v1/use-of-sold-products/energy-profiles` | List product energy profiles |
| 17 | GET | `/api/v1/use-of-sold-products/refrigerant-gwps` | Get refrigerant GWP table |
| 18 | GET | `/api/v1/use-of-sold-products/fuel-factors` | Get fuel combustion EFs |
| 19 | GET | `/api/v1/use-of-sold-products/lifetime-estimates` | Get lifetime tables |
| 20 | GET | `/api/v1/use-of-sold-products/aggregations` | Get aggregated results |
| 21 | GET | `/api/v1/use-of-sold-products/provenance/{id}` | Get provenance chain |
| 22 | GET | `/api/v1/use-of-sold-products/health` | Health check |

## 9. Database Schema (21 Tables)

### 9.1 Reference Tables
1. `gl_usp_product_energy_profiles` — Product energy consumption profiles (24 products)
2. `gl_usp_fuel_emission_factors` — 15 fuel combustion EFs with NCV
3. `gl_usp_grid_emission_factors` — Grid EFs by country (16 regions)
4. `gl_usp_refrigerant_gwps` — 10 refrigerants with AR5/AR6 GWPs, charge, leak rates
5. `gl_usp_product_lifetimes` — Default lifetimes by category with adjustment factors
6. `gl_usp_energy_degradation` — Degradation curves by product category
7. `gl_usp_usage_adjustment_factors` — Standard/heavy/light/industrial/seasonal
8. `gl_usp_chemical_products` — Chemical product GHG content and release fractions
9. `gl_usp_steam_cooling_factors` — Steam/cooling emission factors by source

### 9.2 Result Tables
10. `gl_usp_calculations` — Main results (TimescaleDB hypertable)
11. `gl_usp_calculation_details` — Per-product breakdown
12. `gl_usp_direct_emissions` — Direct emissions detail (fuel/refrigerant/chemical)
13. `gl_usp_indirect_emissions` — Indirect emissions detail (electricity/heating/steam)
14. `gl_usp_fuel_sales_emissions` — Fuel/feedstock sales emissions
15. `gl_usp_aggregations` — Aggregated results (TimescaleDB hypertable)
16. `gl_usp_provenance_records` — Provenance chain hashes

### 9.3 Operational Tables
17. `gl_usp_compliance_results` — Compliance checks (TimescaleDB hypertable)
18. `gl_usp_data_quality_scores` — DQI scoring
19. `gl_usp_uncertainty_results` — Uncertainty analysis
20. `gl_usp_audit_trail` — Audit log
21. `gl_usp_batch_jobs` — Batch job tracking

## 10. Enumerations (22)

1. `ProductUseCategory` — 10 categories (VEHICLES through MEDICAL_DEVICES)
2. `UsePhaseEmissionType` — 6 types (direct fuel/refrigerant/chemical, indirect electricity/heating/steam)
3. `VehicleType` — 6 types (passenger_car_gasoline/diesel/ev, light_truck, heavy_truck, motorcycle)
4. `ApplianceType` — 5 types (refrigerator, washing_machine, dishwasher, dryer, oven)
5. `HVACType` — 4 types (room_ac, central_ac, heat_pump, gas_furnace)
6. `ITEquipmentType` — 4 types (laptop, desktop, server, monitor)
7. `IndustrialType` — 3 types (diesel_generator, gas_boiler, compressor)
8. `FuelType` — 15 fuels
9. `RefrigerantType` — 10 refrigerants
10. `GWPStandard` — 2 standards (AR5, AR6)
11. `GridRegion` — 16 regions
12. `LifetimeAdjustment` — 5 factors (standard, heavy, light, industrial, seasonal)
13. `CalculationMethod` — 8 methods (direct_fuel, direct_refrigerant, direct_chemical, indirect_electricity, indirect_heating, indirect_steam, fuels_sold, feedstocks_sold)
14. `DataQualityTier` — 3 tiers
15. `DQIDimension` — 5 dimensions
16. `ComplianceFramework` — 7 frameworks
17. `ComplianceStatus` — 4 statuses
18. `PipelineStage` — 10 stages
19. `ProvenanceStage` — 10 stages
20. `UncertaintyMethod` — 3 methods
21. `BatchStatus` — 4 statuses
22. `AuditAction` — 6 actions

## 11. Pydantic Models (14)

1. `ProductInput` — product_id, category, product_type, units_sold, lifetime_years, annual_energy, fuel_type, energy_type, use_region, refrigerant_type, charge_kg, leak_rate, chemical_content_kg, release_fraction
2. `FuelSalesInput` — fuel_type, volume_sold, unit, region
3. `DirectEmissionsInput` — org_id, reporting_year, products list (direct emissions)
4. `IndirectEmissionsInput` — org_id, reporting_year, products list (indirect emissions)
5. `FuelsAndFeedstocksInput` — org_id, reporting_year, fuel_sales list
6. `CalculationResult` — calc_id, org_id, reporting_year, direct_emissions_kg, indirect_emissions_kg, fuel_sales_emissions_kg, total_emissions_kg, total_tco2e, product_breakdowns, dqi, uncertainty, provenance_hash
7. `ProductBreakdown` — product_id, category, units_sold, lifetime, direct_emissions_kg, indirect_emissions_kg, total_emissions_kg, method, dqi
8. `DirectEmissionDetail` — product_id, emission_type, fuel_type/refrigerant/chemical, emissions_kg, formula_used
9. `IndirectEmissionDetail` — product_id, emission_type, energy_kwh/fuel_l, grid_ef, emissions_kg
10. `AggregationResult` — period, total_tco2e, direct_tco2e, indirect_tco2e, fuels_tco2e, by_category, by_emission_type
11. `ComplianceResult` — framework, status, rules_checked/passed/failed, findings
12. `ProvenanceRecord` — stage, input_hash, output_hash, timestamp
13. `DataQualityScore` — 5 dimensions + overall
14. `UncertaintyResult` — method, mean, std_dev, ci_lower, ci_upper

## 12. File Manifest

### Source Files (15)
| # | File | Engine | Est. Lines |
|---|------|--------|-----------|
| 1 | `__init__.py` | Package init | ~130 |
| 2 | `models.py` | Data models, enums, constants | ~2,400 |
| 3 | `config.py` | Configuration (GL_USP_ env prefix) | ~2,400 |
| 4 | `metrics.py` | Prometheus metrics (gl_usp_ prefix) | ~1,300 |
| 5 | `provenance.py` | SHA-256 chain hashing | ~2,100 |
| 6 | `product_use_database.py` | Engine 1: EF storage & retrieval | ~2,400 |
| 7 | `direct_emissions_calculator.py` | Engine 2: Direct use-phase | ~2,500 |
| 8 | `indirect_emissions_calculator.py` | Engine 3: Indirect use-phase | ~2,000 |
| 9 | `fuels_feedstocks_calculator.py` | Engine 4: Fuels & feedstocks sold | ~1,500 |
| 10 | `lifetime_modeling.py` | Engine 5: Lifetime & degradation | ~1,800 |
| 11 | `compliance_checker.py` | Engine 6: 7 frameworks, 8 DC rules | ~3,200 |
| 12 | `use_of_sold_products_pipeline.py` | Engine 7: 10-stage pipeline | ~1,800 |
| 13 | `setup.py` | Service facade wiring 7 engines | ~1,800 |
| 14 | `api/__init__.py` | API subpackage | ~1 |
| 15 | `api/router.py` | 22 REST endpoints | ~2,400 |

### Test Files (14)
| # | File | Tests |
|---|------|-------|
| 1 | `__init__.py` | Package marker |
| 2 | `conftest.py` | Fixtures, factories, singletons |
| 3 | `test_models.py` | Enums, constants, Pydantic models |
| 4 | `test_config.py` | Config loading, env overrides |
| 5 | `test_product_use_database.py` | EF lookups, GWPs, profiles |
| 6 | `test_direct_emissions_calculator.py` | Fuel/refrigerant/chemical |
| 7 | `test_indirect_emissions_calculator.py` | Electricity/heating/steam |
| 8 | `test_fuels_feedstocks_calculator.py` | Fuel sales, feedstock oxidation |
| 9 | `test_lifetime_modeling.py` | Lifetime, degradation, survival |
| 10 | `test_compliance_checker.py` | 7 frameworks, 8 DC rules |
| 11 | `test_provenance.py` | Hash chains, Merkle trees |
| 12 | `test_use_of_sold_products_pipeline.py` | 10-stage pipeline |
| 13 | `test_api.py` | 22 endpoints |
| 14 | `test_setup.py` | Service wiring |

### Migration (1)
| # | File | Description |
|---|------|-------------|
| 1 | `V075__use_of_sold_products_service.sql` | 21 tables, 3 hypertables, 2 cont. aggs |

## 13. Key Architectural Decisions

1. **Dual-path calculation**: Separate direct (fuel/refrigerant/chemical) and indirect (electricity/heating/steam) emission paths merged in pipeline
2. **Lifetime modeling**: Dedicated engine for product lifetime estimation with degradation curves — critical for Cat 11 accuracy
3. **Fuels & feedstocks**: Separate calculator for companies selling fuels (often dominates Cat 11 for oil/gas companies)
4. **Refrigerant GWP versioning**: Support both AR5 and AR6 GWP values with framework-specific requirements
5. **Usage adjustment**: Standard/heavy/light/industrial/seasonal multipliers for lifetime assumptions
6. **Energy degradation**: Model efficiency loss over product lifetime (vehicles lose ~1.5%/year fuel efficiency)
7. **Direct + indirect split**: Track and report both separately as required by GHG Protocol
8. **Product-level granularity**: Per-product breakdown supporting portfolio-level analysis
9. **Thread-safe singletons**: All engines use threading.RLock
10. **Decimal precision**: ROUND_HALF_UP for all regulatory calculations
