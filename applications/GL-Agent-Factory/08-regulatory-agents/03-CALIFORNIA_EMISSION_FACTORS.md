# California Emission Factors for SB 253 Compliance

**Version:** 1.0.0
**Date:** 2025-12-04
**Source Authority:** EPA eGRID 2023, CARB, EPA GHG EF Hub
**GWP Basis:** IPCC AR6 (GWP-100)

---

## 1. Overview

This document provides the authoritative emission factors required for California SB 253 compliance. All factors are sourced from EPA, CARB, and IPCC and are aligned with the GHG Protocol Corporate Standard.

**Key Sources:**
- EPA eGRID 2023 (electricity grid factors)
- EPA GHG Emission Factors Hub 2024 (fuel combustion)
- California Air Resources Board (CARB) factors
- IPCC AR6 GWP values (2021)

---

## 2. California Electricity Grid Factor (Scope 2)

### 2.1 CAMX Subregion (California)

```yaml
grid_factor:
  subregion_code: CAMX
  subregion_name: "WECC California"
  states: ["California"]

  emission_factor:
    value: 0.254
    unit: "kg CO2e / kWh"
    co2_factor: 0.252
    ch4_factor: 0.00001
    n2o_factor: 0.000001

  source: "EPA eGRID 2023"
  source_uri: "https://www.epa.gov/egrid/download-data"
  data_year: 2022
  last_updated: "2024-11-01"

  generation_mix_2022:
    natural_gas: 42.1%
    solar: 17.8%
    large_hydroelectric: 14.6%
    wind: 11.9%
    nuclear: 8.5%
    biomass: 2.3%
    geothermal: 2.2%
    small_hydroelectric: 0.5%
    coal: 0.1%

  notes: |
    California has one of the cleanest electricity grids in the United States
    due to aggressive renewable energy policies (SB 100 targets 100% clean
    energy by 2045). The CAMX factor is significantly lower than the US
    national average (0.417 kg CO2e/kWh).
```

### 2.2 California Utility-Specific Factors

For market-based Scope 2 reporting, utility-specific factors may be used:

```yaml
california_utilities:
  pacific_gas_electric:
    utility_name: "Pacific Gas & Electric (PG&E)"
    service_territory: "Northern and Central California"
    emission_factor_2023:
      value: 0.210
      unit: "kg CO2e / kWh"
    renewable_percentage: 52%
    source: "PG&E Power Content Label 2023"

  southern_california_edison:
    utility_name: "Southern California Edison (SCE)"
    service_territory: "Southern California (excluding LA)"
    emission_factor_2023:
      value: 0.195
      unit: "kg CO2e / kWh"
    renewable_percentage: 48%
    source: "SCE Power Content Label 2023"

  san_diego_gas_electric:
    utility_name: "San Diego Gas & Electric (SDG&E)"
    service_territory: "San Diego County"
    emission_factor_2023:
      value: 0.235
      unit: "kg CO2e / kWh"
    renewable_percentage: 45%
    source: "SDG&E Power Content Label 2023"

  los_angeles_dwp:
    utility_name: "Los Angeles Department of Water and Power (LADWP)"
    service_territory: "City of Los Angeles"
    emission_factor_2023:
      value: 0.340
      unit: "kg CO2e / kWh"
    renewable_percentage: 38%
    source: "LADWP Power Content Label 2023"
```

### 2.3 Adjacent Grid Factors (Multi-State Operations)

Companies with facilities outside California need adjacent grid factors:

```yaml
adjacent_grids:
  aznm_southwest:
    subregion_code: AZNM
    subregion_name: "WECC Southwest"
    states: ["Arizona", "New Mexico"]
    emission_factor:
      value: 0.458
      unit: "kg CO2e / kWh"
    source: "EPA eGRID 2023"

  nwpp_northwest:
    subregion_code: NWPP
    subregion_name: "WECC Northwest"
    states: ["Washington", "Oregon", "Idaho", "Montana", "Wyoming", "Nevada", "Utah", "Colorado"]
    emission_factor:
      value: 0.354
      unit: "kg CO2e / kWh"
    source: "EPA eGRID 2023"

  rmpa_rocky_mountain:
    subregion_code: RMPA
    subregion_name: "WECC Rockies"
    states: ["Colorado", "Nebraska", "Wyoming"]
    emission_factor:
      value: 0.684
      unit: "kg CO2e / kWh"
    source: "EPA eGRID 2023"
```

---

## 3. Scope 1 Emission Factors (Fuel Combustion)

### 3.1 Stationary Combustion

```yaml
stationary_combustion:
  source: "EPA GHG Emission Factors Hub 2024"
  source_uri: "https://www.epa.gov/climateleadership/ghg-emission-factors-hub"
  gwp_basis: "IPCC AR6"

  natural_gas:
    description: "Pipeline quality natural gas"
    factors:
      per_therm:
        value: 5.30
        unit: "kg CO2e / therm"
        co2: 5.27
        ch4: 0.005
        n2o: 0.0001
      per_kwh:
        value: 0.181
        unit: "kg CO2e / kWh"
      per_mcf:
        value: 53.1
        unit: "kg CO2e / MCF"
      per_mmbtu:
        value: 53.1
        unit: "kg CO2e / MMBtu"

  diesel_fuel:
    description: "Diesel #2 / Distillate Fuel Oil"
    factors:
      per_gallon:
        value: 10.21
        unit: "kg CO2e / gallon"
        co2: 10.15
        ch4: 0.0004
        n2o: 0.0001
      per_liter:
        value: 2.70
        unit: "kg CO2e / liter"

  propane_lpg:
    description: "Propane / LPG"
    factors:
      per_gallon:
        value: 5.72
        unit: "kg CO2e / gallon"
        co2: 5.68
        ch4: 0.0003
        n2o: 0.0001

  fuel_oil_2:
    description: "Fuel Oil #2 / Heating Oil"
    factors:
      per_gallon:
        value: 10.21
        unit: "kg CO2e / gallon"

  fuel_oil_4:
    description: "Fuel Oil #4"
    factors:
      per_gallon:
        value: 11.27
        unit: "kg CO2e / gallon"

  fuel_oil_6:
    description: "Fuel Oil #6 / Residual"
    factors:
      per_gallon:
        value: 11.27
        unit: "kg CO2e / gallon"

  kerosene:
    description: "Kerosene"
    factors:
      per_gallon:
        value: 10.15
        unit: "kg CO2e / gallon"
```

### 3.2 Mobile Combustion

```yaml
mobile_combustion:
  source: "EPA GHG Emission Factors Hub 2024"
  gwp_basis: "IPCC AR6"

  gasoline:
    description: "Motor gasoline"
    factors:
      per_gallon:
        value: 8.78
        unit: "kg CO2e / gallon"
        co2: 8.73
        ch4: 0.0003
        n2o: 0.0002
      per_liter:
        value: 2.32
        unit: "kg CO2e / liter"

  diesel:
    description: "Diesel fuel for vehicles"
    factors:
      per_gallon:
        value: 10.21
        unit: "kg CO2e / gallon"
      per_liter:
        value: 2.70
        unit: "kg CO2e / liter"

  e10_gasoline:
    description: "Gasoline with 10% ethanol"
    factors:
      per_gallon:
        value: 8.53
        unit: "kg CO2e / gallon"

  e85_gasoline:
    description: "Gasoline with 85% ethanol"
    factors:
      per_gallon:
        value: 6.10
        unit: "kg CO2e / gallon"

  biodiesel_b20:
    description: "Diesel with 20% biodiesel"
    factors:
      per_gallon:
        value: 9.20
        unit: "kg CO2e / gallon"

  california_carb_diesel:
    description: "CARB low-carbon diesel"
    factors:
      per_gallon:
        value: 9.98
        unit: "kg CO2e / gallon"
    source: "CARB LCFS"
```

### 3.3 Vehicle-Specific Emission Factors

```yaml
vehicle_emission_factors:
  source: "EPA Automotive Trends 2024"

  passenger_vehicles:
    average_car:
      emission_factor: 0.191
      unit: "kg CO2e / km"
    hybrid_car:
      emission_factor: 0.107
      unit: "kg CO2e / km"
    electric_vehicle_california:
      emission_factor: 0.053
      unit: "kg CO2e / km"
      grid_factor: "CAMX 0.254 kg/kWh"
      efficiency: "0.21 kWh/km"

  light_trucks:
    average_suv:
      emission_factor: 0.254
      unit: "kg CO2e / km"
    pickup_truck:
      emission_factor: 0.290
      unit: "kg CO2e / km"

  heavy_duty_vehicles:
    delivery_truck_medium:
      emission_factor: 0.495
      unit: "kg CO2e / km"
    semi_truck:
      emission_factor: 1.014
      unit: "kg CO2e / km"
```

---

## 4. Fugitive Emissions (Refrigerants)

### 4.1 IPCC AR6 GWP Values

```yaml
refrigerant_gwp:
  source: "IPCC AR6 (2021)"
  gwp_timeframe: "100 years"

  hfcs:
    r134a:
      chemical_name: "1,1,1,2-Tetrafluoroethane"
      gwp_100: 1530
      common_use: ["Automotive AC", "Refrigeration"]

    r410a:
      chemical_name: "50% R-32, 50% R-125"
      gwp_100: 2088
      common_use: ["Commercial HVAC", "Residential AC"]

    r407c:
      chemical_name: "23% R-32, 25% R-125, 52% R-134a"
      gwp_100: 1774
      common_use: ["Commercial HVAC"]

    r404a:
      chemical_name: "44% R-125, 52% R-143a, 4% R-134a"
      gwp_100: 3922
      common_use: ["Commercial refrigeration"]

    r507a:
      chemical_name: "50% R-125, 50% R-143a"
      gwp_100: 3985
      common_use: ["Commercial refrigeration"]

    r32:
      chemical_name: "Difluoromethane"
      gwp_100: 675
      common_use: ["Modern HVAC systems"]

  hcfcs_phaseout:
    r22:
      chemical_name: "Chlorodifluoromethane"
      gwp_100: 1810
      common_use: ["Legacy HVAC systems"]
      phaseout_status: "Production ended 2020"

  natural_refrigerants:
    r744_co2:
      gwp_100: 1
      common_use: ["Commercial refrigeration"]

    r717_ammonia:
      gwp_100: 0
      common_use: ["Industrial refrigeration"]

    r290_propane:
      gwp_100: 3
      common_use: ["Small refrigeration units"]
```

### 4.2 Leak Rate Assumptions

```yaml
refrigerant_leak_rates:
  source: "EPA GHG Inventory"

  equipment_type:
    commercial_refrigeration:
      annual_leak_rate: 25%
      range: "15-35%"

    industrial_refrigeration:
      annual_leak_rate: 20%
      range: "10-30%"

    commercial_ac_hvac:
      annual_leak_rate: 10%
      range: "5-15%"

    residential_ac:
      annual_leak_rate: 4%
      range: "2-8%"

    vehicle_ac:
      annual_leak_rate: 18%
      range: "10-25%"

    chiller_systems:
      annual_leak_rate: 8%
      range: "3-12%"
```

---

## 5. Scope 3 Emission Factors

### 5.1 Category 1: Purchased Goods and Services (EEIO)

```yaml
category_1_eeio_factors:
  source: "EPA Environmentally-Extended Input-Output (EEIO)"
  source_uri: "https://www.epa.gov/land-research/us-environmentally-extended-input-output-useeio-technical-content"
  methodology: "Economic input-output analysis"

  manufacturing_sectors:
    naics_31_33:
      sector: "Manufacturing (average)"
      factor: 0.40
      unit: "kg CO2e / USD"

    naics_3241:
      sector: "Petroleum and Coal Products"
      factor: 1.85
      unit: "kg CO2e / USD"

    naics_3251:
      sector: "Basic Chemicals"
      factor: 0.72
      unit: "kg CO2e / USD"

    naics_3311:
      sector: "Iron and Steel Mills"
      factor: 0.95
      unit: "kg CO2e / USD"

    naics_3361:
      sector: "Motor Vehicles"
      factor: 0.35
      unit: "kg CO2e / USD"

    naics_3341:
      sector: "Computer and Peripheral Equipment"
      factor: 0.22
      unit: "kg CO2e / USD"

  services_sectors:
    naics_541:
      sector: "Professional, Scientific, Technical Services"
      factor: 0.15
      unit: "kg CO2e / USD"

    naics_5415:
      sector: "Computer Systems Design"
      factor: 0.12
      unit: "kg CO2e / USD"

    naics_5511:
      sector: "Management of Companies"
      factor: 0.14
      unit: "kg CO2e / USD"

    naics_561:
      sector: "Administrative Support Services"
      factor: 0.18
      unit: "kg CO2e / USD"

  retail_sectors:
    naics_44_45:
      sector: "Retail Trade (average)"
      factor: 0.25
      unit: "kg CO2e / USD"

    naics_445:
      sector: "Food and Beverage Stores"
      factor: 0.32
      unit: "kg CO2e / USD"
```

### 5.2 Category 3: Fuel and Energy-Related Activities

```yaml
category_3_factors:
  source: "DEFRA 2024 / EPA"

  well_to_tank_factors:
    description: "Upstream emissions from fuel extraction, processing, and transport"

    natural_gas:
      wtt_factor: 0.62
      unit: "kg CO2e / therm"
      percentage_of_combustion: 12%

    diesel:
      wtt_factor: 0.61
      unit: "kg CO2e / gallon"
      percentage_of_combustion: 6%

    gasoline:
      wtt_factor: 0.52
      unit: "kg CO2e / gallon"
      percentage_of_combustion: 6%

    electricity_california:
      wtt_factor: 0.025
      unit: "kg CO2e / kWh"
      description: "Upstream of generation"

  transmission_distribution_losses:
    description: "Emissions from T&D losses"

    california:
      t_d_loss_rate: 5.5%
      t_d_emission_factor: 0.014
      unit: "kg CO2e / kWh consumed"

    us_average:
      t_d_loss_rate: 5.0%
      t_d_emission_factor: 0.021
      unit: "kg CO2e / kWh consumed"
```

### 5.3 Category 4 & 9: Transportation and Distribution

```yaml
transportation_factors:
  source: "GLEC Framework 2.0 / EPA SmartWay"

  freight_transport:
    road_truck:
      average:
        factor: 0.062
        unit: "kg CO2e / tonne-km"
      ltl_truck:
        factor: 0.154
        unit: "kg CO2e / tonne-km"
      ftl_truck:
        factor: 0.047
        unit: "kg CO2e / tonne-km"

    rail:
      factor: 0.024
      unit: "kg CO2e / tonne-km"

    air_freight:
      belly_cargo:
        factor: 0.602
        unit: "kg CO2e / tonne-km"
      dedicated_freighter:
        factor: 0.537
        unit: "kg CO2e / tonne-km"

    sea_freight:
      container_ship:
        factor: 0.011
        unit: "kg CO2e / tonne-km"
      bulk_carrier:
        factor: 0.008
        unit: "kg CO2e / tonne-km"
      tanker:
        factor: 0.005
        unit: "kg CO2e / tonne-km"

    intermodal:
      truck_rail:
        factor: 0.035
        unit: "kg CO2e / tonne-km"
      truck_ship:
        factor: 0.018
        unit: "kg CO2e / tonne-km"
```

### 5.4 Category 5: Waste Generated in Operations

```yaml
waste_factors:
  source: "EPA WARM Model 2024"

  disposal_methods:
    landfill:
      mixed_msw:
        factor: 520
        unit: "kg CO2e / tonne"
      paper:
        factor: 1,560
        unit: "kg CO2e / tonne"
      food_waste:
        factor: 580
        unit: "kg CO2e / tonne"
      plastics:
        factor: 20
        unit: "kg CO2e / tonne"

    recycling:
      mixed_paper:
        factor: -1,020
        unit: "kg CO2e / tonne"
        notes: "Negative = avoided emissions"
      corrugated_cardboard:
        factor: -1,500
        unit: "kg CO2e / tonne"
      aluminum:
        factor: -9,110
        unit: "kg CO2e / tonne"
      steel:
        factor: -1,810
        unit: "kg CO2e / tonne"
      plastics_mixed:
        factor: -690
        unit: "kg CO2e / tonne"

    composting:
      food_waste:
        factor: -210
        unit: "kg CO2e / tonne"
      yard_waste:
        factor: -230
        unit: "kg CO2e / tonne"

    incineration:
      mixed_msw:
        factor: 50
        unit: "kg CO2e / tonne"
        notes: "With energy recovery"
```

### 5.5 Category 6: Business Travel

```yaml
business_travel_factors:
  source: "DEFRA 2024"

  air_travel:
    description: "Includes radiative forcing multiplier of 1.891"

    domestic:
      economy:
        factor: 0.255
        unit: "kg CO2e / passenger-km"
      business:
        factor: 0.382
        unit: "kg CO2e / passenger-km"

    short_haul:
      description: "<3,700 km"
      economy:
        factor: 0.156
        unit: "kg CO2e / passenger-km"
      business:
        factor: 0.234
        unit: "kg CO2e / passenger-km"

    long_haul:
      description: ">3,700 km"
      economy:
        factor: 0.147
        unit: "kg CO2e / passenger-km"
      premium_economy:
        factor: 0.235
        unit: "kg CO2e / passenger-km"
      business:
        factor: 0.441
        unit: "kg CO2e / passenger-km"
      first:
        factor: 0.588
        unit: "kg CO2e / passenger-km"

  rail:
    national_rail:
      factor: 0.041
      unit: "kg CO2e / passenger-km"
    high_speed_rail:
      factor: 0.006
      unit: "kg CO2e / passenger-km"

  road:
    rental_car_average:
      factor: 0.192
      unit: "kg CO2e / passenger-km"
    taxi:
      factor: 0.210
      unit: "kg CO2e / passenger-km"

  hotel:
    factor: 31.1
    unit: "kg CO2e / room-night"
```

### 5.6 Category 7: Employee Commuting

```yaml
commuting_factors:
  source: "DEFRA 2024 / EPA"

  commute_modes:
    car_gasoline:
      factor: 0.192
      unit: "kg CO2e / passenger-km"

    car_diesel:
      factor: 0.171
      unit: "kg CO2e / passenger-km"

    car_hybrid:
      factor: 0.107
      unit: "kg CO2e / passenger-km"

    car_electric_california:
      factor: 0.053
      unit: "kg CO2e / passenger-km"
      grid_factor: "CAMX"

    carpool_2_person:
      factor: 0.096
      unit: "kg CO2e / passenger-km"

    carpool_4_person:
      factor: 0.048
      unit: "kg CO2e / passenger-km"

    bus:
      factor: 0.103
      unit: "kg CO2e / passenger-km"

    rail_light:
      factor: 0.035
      unit: "kg CO2e / passenger-km"

    rail_metro:
      factor: 0.030
      unit: "kg CO2e / passenger-km"

    bicycle:
      factor: 0.000
      unit: "kg CO2e / passenger-km"

    walking:
      factor: 0.000
      unit: "kg CO2e / passenger-km"

    remote_work:
      factor: 0.000
      unit: "kg CO2e / day"
      notes: "Home office emissions typically de minimis"

  california_defaults:
    average_commute_distance: 24
    unit: "km one-way"
    average_work_days: 230
    unit: "days/year"
    remote_work_percentage: 30%
    notes: "Post-COVID California average"
```

### 5.7 Category 15: Investments

```yaml
investment_factors:
  source: "PCAF (Partnership for Carbon Accounting Financials)"

  methodology: "Financed emissions"

  attribution_factors:
    listed_equity:
      attribution: "Ownership share"
      calculation: "(Investee Scope 1+2) x (Outstanding shares held / Total shares)"

    corporate_bonds:
      attribution: "Financing share"
      calculation: "(Investee Scope 1+2) x (Bond value / Investee EVIC)"

    project_finance:
      attribution: "Financing share"
      calculation: "Project emissions x (Financing amount / Total project cost)"

  sector_average_intensities:
    utilities:
      factor: 350
      unit: "kg CO2e / $1000 revenue"

    oil_gas:
      factor: 420
      unit: "kg CO2e / $1000 revenue"

    manufacturing:
      factor: 180
      unit: "kg CO2e / $1000 revenue"

    technology:
      factor: 25
      unit: "kg CO2e / $1000 revenue"

    financial_services:
      factor: 15
      unit: "kg CO2e / $1000 revenue"
```

---

## 6. Data Quality and Updates

### 6.1 Factor Update Schedule

```yaml
update_schedule:
  epa_egrid:
    frequency: "Annual"
    typical_release: "Q4"
    lag_period: "2 years (e.g., 2023 release contains 2022 data)"

  epa_ghg_ef_hub:
    frequency: "Annual"
    typical_release: "Q1"

  defra_factors:
    frequency: "Annual"
    typical_release: "June"

  ipcc_gwp:
    frequency: "Per assessment report (7-8 years)"
    current: "AR6 (2021)"
    next_expected: "AR7 (~2029)"

  carb_factors:
    frequency: "As updated"
    monitoring: "LCFS program updates"
```

### 6.2 Data Quality Indicators

```yaml
data_quality:
  tier_1:
    description: "National/regional averages from government sources"
    uncertainty: "+/- 5%"
    examples: ["EPA eGRID", "EPA GHG EF Hub"]

  tier_2:
    description: "Technology-specific factors"
    uncertainty: "+/- 10%"
    examples: ["Vehicle-specific factors", "Industry-specific EEIO"]

  tier_3:
    description: "Site-specific or supplier-specific data"
    uncertainty: "+/- 15-20%"
    examples: ["Supplier emission reports", "Facility measurements"]
```

---

## 7. Implementation Notes

### 7.1 California-Specific Considerations

```yaml
california_specific:
  clean_grid:
    description: |
      California's grid is significantly cleaner than the US average.
      Using the correct CAMX factor (0.254 vs US average 0.417) is
      critical for accurate Scope 2 reporting.

  carb_lcfs:
    description: |
      California's Low Carbon Fuel Standard affects transportation
      fuel carbon intensity. Use CARB-specific factors when available.

  electric_vehicles:
    description: |
      EV adoption is higher in California. EV emission factors should
      use California grid (CAMX) factor, not national average.

  renewable_energy:
    description: |
      Many California utilities offer high renewable content.
      Market-based Scope 2 using utility-specific factors may
      show lower emissions than location-based.
```

### 7.2 Factor Selection Hierarchy

```yaml
factor_hierarchy:
  1_supplier_specific:
    priority: "Highest"
    use_when: "Supplier provides verified emission data"
    data_quality: "Tier 3"

  2_technology_specific:
    priority: "High"
    use_when: "Specific technology/process known"
    data_quality: "Tier 2"

  3_regional_average:
    priority: "Medium"
    use_when: "Regional data available (e.g., eGRID subregion)"
    data_quality: "Tier 1"

  4_national_average:
    priority: "Low"
    use_when: "Only national data available"
    data_quality: "Tier 1"

  5_global_average:
    priority: "Lowest"
    use_when: "No other data available"
    data_quality: "Tier 1"
```

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-SB253-PM | Initial emission factors specification |

**Sources Cited:**
- EPA eGRID 2023: https://www.epa.gov/egrid
- EPA GHG EF Hub: https://www.epa.gov/climateleadership/ghg-emission-factors-hub
- DEFRA 2024: https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting
- IPCC AR6: https://www.ipcc.ch/assessment-report/ar6/
- GLEC Framework: https://www.smartfreightcentre.org/en/how-to-implement-items/what-is-glec-framework/
- EPA WARM: https://www.epa.gov/warm
- PCAF: https://carbonaccountingfinancials.com/

---

**END OF DOCUMENT**
