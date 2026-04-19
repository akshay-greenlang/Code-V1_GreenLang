# PRD: AGENT-MRV-009 — Scope 2 Location-Based Emissions Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE2-009 |
| **Internal Label** | AGENT-MRV-009 |
| **Category** | Layer 3 — MRV / Accounting Agents (Scope 2) |
| **Package** | `greenlang/scope2_location/` |
| **DB Migration** | V060 |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

Calculates **Scope 2 location-based GHG emissions** from purchased electricity,
steam, heating, and cooling using **grid-average emission factors** per the
GHG Protocol Scope 2 Guidance (2015). The location-based method reflects the
average emissions intensity of the grid on which energy consumption occurs.

### Standards & References

- GHG Protocol Scope 2 Guidance (2015) — Primary methodology
- GHG Protocol Corporate Standard (Revised 2015) — Scope 2 boundary
- IEA CO2 Emission Factors (2024) — Country/regional grid factors
- EPA eGRID (2022) — US sub-regional emission factors (26 subregions)
- EU EEA — European grid emission factors
- DEFRA/DESNZ (2024) — UK emission factors for electricity/steam/heat
- IPCC 2006/2019 — Default T&D loss factors
- ISO 14064-1:2018 — Scope 2 reporting requirements
- CSRD/ESRS E1 — EU double materiality Scope 2 disclosure

---

## 2. Scope 2 Location-Based Methodology

### 2.1 Core Formula

```
Emissions (tCO2e) = Activity Data × Grid Emission Factor × (1 + T&D Loss Factor)
```

Where:
- **Activity Data**: Energy consumed (MWh, GJ, etc.)
- **Grid Emission Factor**: Grid-average EF (tCO2e/MWh or tCO2e/GJ)
- **T&D Loss Factor**: Transmission & distribution losses (%)

### 2.2 Energy Types

| Energy Type | Unit | Typical EF Range | Source |
|------------|------|------------------|--------|
| Electricity | MWh | 0.0–1.2 tCO2e/MWh | eGRID/IEA/National |
| Steam | GJ | 0.05–0.12 tCO2e/GJ | DEFRA/National |
| Heating | GJ | 0.04–0.10 tCO2e/GJ | District heating EFs |
| Cooling | GJ | 0.02–0.08 tCO2e/GJ | Absorption/electric chiller |

### 2.3 Grid Emission Factor Sources

1. **EPA eGRID** (US): 26 subregions with CO2, CH4, N2O per MWh
   - CAMX (0.225), ERCT (0.380), FRCC (0.392), MROE (0.482), etc.
2. **IEA** (Global): 130+ countries with tCO2/MWh
   - France (0.056), Germany (0.338), China (0.555), India (0.708), etc.
3. **EU EEA**: 27 member states + country-specific
4. **DEFRA/DESNZ** (UK): Annual UK factors with generation + T&D
5. **National Inventories**: Country-specific where available
6. **Custom Factors**: User-supplied factors with quality tracking

### 2.4 Transmission & Distribution Losses

| Region/Country | T&D Loss % | Source |
|---------------|-----------|--------|
| US Average | 5.0% | EIA |
| EU Average | 6.5% | Eurostat |
| UK | 7.7% | DEFRA |
| India | 19.4% | CEA |
| China | 5.8% | NBS |
| Sub-Saharan Africa | 15.0–25.0% | IEA |
| World Average | 8.3% | IEA |

### 2.5 Per-Gas Emission Factors

Grid emission factors are decomposed into three GHG gases:
- **CO2**: Dominant (typically 95–99% of grid EF)
- **CH4**: From natural gas combustion in power plants
- **N2O**: From fuel combustion and biomass co-firing

Each gas is converted to CO2e using the selected GWP source (AR4/AR5/AR6).

### 2.6 Time-of-Use Factors

Advanced feature for hourly/monthly emission factor variation:
- **Annual average**: Default, single EF for entire year
- **Monthly average**: 12 monthly EFs reflecting seasonal generation mix
- **Hourly marginal**: 8760 hourly EFs for real-time carbon accounting

---

## 3. Architecture — 7 Engines

### Engine 1: GridEmissionFactorDatabaseEngine (`grid_factor_database.py`)
- Stores grid emission factors from 6+ authoritative sources
- 130+ country/region factors (IEA), 26 US eGRID subregions
- EU 27 member state factors, UK DEFRA factors
- Steam/heat/cooling default factors by region
- T&D loss factors by country/region
- Historical factor series (2015–2025) for trend analysis
- Custom factor CRUD with quality scoring
- Factor lookup by: country code, eGRID subregion, grid region ID
- 25+ public methods

### Engine 2: ElectricityEmissionsEngine (`electricity_emissions.py`)
- Core Scope 2 electricity emission calculations
- Formula: MWh × EF × (1 + T&D_loss)
- Support for 3 granularity levels: annual, monthly, hourly
- Per-gas breakdown: CO2, CH4, N2O with individual GWP conversion
- Multi-facility aggregation
- Consumption data: total MWh, by facility, by meter
- GWP source selection (AR4/AR5/AR6)
- IPCC Tier 1 (country EF) and Tier 2 (subregional EF)
- Biogenic vs fossil CO2 tracking for biomass co-firing grids

### Engine 3: SteamHeatCoolingEngine (`steam_heat_cooling.py`)
- Emissions from purchased steam, heating, and cooling
- Steam: GJ × steam_EF (typically from natural gas boilers)
- Heating: GJ × heating_EF (district heating, direct fuel)
- Cooling: GJ × cooling_EF (absorption/electric chiller)
- Efficiency-adjusted factors
- Combined heat and power (CHP) allocation methods
- District energy system accounting
- Default DEFRA/IPCC factors with regional overrides

### Engine 4: TransmissionLossEngine (`transmission_loss.py`)
- Calculates T&D losses to adjust scope 2 emissions
- Country-specific T&D loss factors (IEA/EIA data)
- Optional upstream emissions from T&D losses
- Grid loss allocation: proportional, marginal, or fixed
- Support for on-site generation deduction
- Net vs gross consumption accounting

### Engine 5: UncertaintyQuantifierEngine (`uncertainty_quantifier.py`)
- Monte Carlo simulation for Scope 2 uncertainty
- Analytical error propagation
- Grid EF uncertainty ranges (IPCC default ±5–50%)
- Activity data uncertainty (meter accuracy, estimation methods)
- DQI scoring for emission factors
- Parameter distributions specific to electricity/grid factors

### Engine 6: ComplianceCheckerEngine (`compliance_checker.py`)
- Multi-framework regulatory compliance checking
- 7 frameworks: GHG Protocol Scope 2, IPCC 2006, ISO 14064, CSRD/ESRS, EPA GHGRP, DEFRA, CDP
- Dual reporting readiness (location + market-based)
- EF source validation
- Temporal consistency checks
- Boundary completeness assessment

### Engine 7: Scope2LocationPipelineEngine (`scope2_location_pipeline.py`)
- 8-stage orchestrated calculation pipeline
- Stage 1: Validate input (facility, energy type, consumption, grid region)
- Stage 2: Resolve grid region and emission factors
- Stage 3: Apply T&D losses
- Stage 4: Calculate electricity emissions (per-gas)
- Stage 5: Calculate steam/heat/cooling emissions
- Stage 6: Apply GWP conversion
- Stage 7: Run compliance checks (optional)
- Stage 8: Assemble results with provenance

---

## 4. Data Models (18 Enumerations, 18 Data Models)

### Enumerations
1. `EnergyType` — 4 members: electricity, steam, heating, cooling
2. `EnergyUnit` — 5 members: kwh, mwh, gj, mmbtu, therms
3. `GridRegionSource` — 6 members: eGRID, IEA, EU_EEA, DEFRA, national, custom
4. `CalculationMethod` — 6 members: ipcc_tier_1, ipcc_tier_2, ipcc_tier_3, mass_balance, direct_measurement, spend_based
5. `EmissionGas` — 3 members: co2, ch4, n2o
6. `GWPSource` — 4 members: AR4, AR5, AR6, AR6_20YR
7. `EmissionFactorSource` — 7 members: eGRID, IEA, DEFRA, EU_EEA, national, custom, IPCC
8. `DataQualityTier` — 3 members: tier_1, tier_2, tier_3
9. `FacilityType` — 8 members: office, warehouse, manufacturing, retail, data_center, hospital, school, other
10. `GridRegionType` — 4 members: country, subregion, state, custom
11. `TDLossMethod` — 3 members: country_average, regional, custom
12. `TimeGranularity` — 3 members: annual, monthly, hourly
13. `ComplianceStatus` — 4 members: compliant, non_compliant, partial, not_assessed
14. `ReportingPeriod` — 4 members: annual, quarterly, monthly, custom
15. `ConsumptionDataSource` — 4 members: meter, invoice, estimate, benchmark
16. `SteamType` — 3 members: natural_gas, coal, biomass
17. `CoolingType` — 3 members: electric_chiller, absorption, district
18. `HeatingType` — 3 members: district, gas_boiler, electric

### Constants (all Decimal)
- `GWP_VALUES` — AR4/AR5/AR6/AR6_20YR for CO2, CH4, N2O
- `EGRID_FACTORS` — 26 US subregions with CO2/CH4/N2O (kg/MWh)
- `IEA_COUNTRY_FACTORS` — 130+ countries (tCO2/MWh)
- `EU_COUNTRY_FACTORS` — EU 27 member states
- `DEFRA_FACTORS` — UK electricity/steam/heat/cooling
- `TD_LOSS_FACTORS` — 50+ countries T&D loss percentages
- `STEAM_DEFAULT_EF` — By fuel source (GJ basis)
- `HEAT_DEFAULT_EF` — By heating type
- `COOLING_DEFAULT_EF` — By cooling type
- `UNIT_CONVERSIONS` — MWh↔GJ↔MMBTU↔therms

### Data Models
1. `FacilityInfo` — Facility registration with grid region
2. `GridRegion` — Grid region with source, country, subregion
3. `GridEmissionFactor` — EF entry with CO2/CH4/N2O, year, source
4. `EnergyConsumption` — Consumption record (energy type, quantity, unit)
5. `ElectricityConsumptionRequest` — Electricity calculation request
6. `SteamHeatCoolingRequest` — Steam/heat/cooling calculation request
7. `TransmissionLossInput` — T&D loss parameters
8. `CalculationRequest` — Unified multi-energy calculation request
9. `GasEmissionDetail` — Per-gas emission result
10. `CalculationResult` — Complete result with per-gas breakdown
11. `BatchCalculationRequest` — Batch of calculation requests
12. `BatchCalculationResult` — Batch results aggregate
13. `ComplianceCheckResult` — Regulatory compliance result
14. `UncertaintyRequest` — Monte Carlo uncertainty request
15. `UncertaintyResult` — Uncertainty analysis result
16. `AggregationResult` — Aggregated emissions by facility/period
17. `GridFactorLookupResult` — Factor lookup result with metadata
18. `TDLossResult` — Transmission & distribution loss result

---

## 5. Database Schema (V060)

### Tables (12)
1. `s2l_facilities` — Facility registration with grid region
2. `s2l_grid_regions` — Grid region definitions
3. `s2l_grid_emission_factors` — Emission factor database
4. `s2l_energy_consumption` — Energy consumption records
5. `s2l_td_loss_factors` — T&D loss factors by region
6. `s2l_calculations` — Calculation results
7. `s2l_calculation_details` — Per-gas/per-energy details
8. `s2l_meter_readings` — Meter reading data
9. `s2l_custom_factors` — User-supplied custom factors
10. `s2l_compliance_records` — Compliance check results
11. `s2l_audit_entries` — Provenance/audit trail
12. `s2l_factor_updates` — Factor version tracking

### Hypertables (3)
- `s2l_calculation_events` — Time-series calculation events
- `s2l_consumption_events_ts` — Consumption time series
- `s2l_compliance_events` — Compliance check events

### Continuous Aggregates (2)
- `s2l_hourly_calculation_stats` — Hourly stats
- `s2l_daily_emission_totals` — Daily emission totals

---

## 6. REST API (20 Endpoints)

Prefix: `/api/v1/scope2-location`

| Method | Path | Permission | Description |
|--------|------|-----------|-------------|
| POST | /calculations | scope2-location:calculate | Single calculation |
| POST | /calculations/batch | scope2-location:calculate | Batch calculation |
| GET | /calculations | scope2-location:read | List calculations |
| GET | /calculations/{id} | scope2-location:read | Get single calculation |
| DELETE | /calculations/{id} | scope2-location:delete | Delete calculation |
| POST | /facilities | scope2-location:facilities:write | Register facility |
| GET | /facilities | scope2-location:facilities:read | List facilities |
| PUT | /facilities/{id} | scope2-location:facilities:write | Update facility |
| POST | /consumption | scope2-location:consumption:write | Record consumption |
| GET | /consumption | scope2-location:consumption:read | List consumption |
| GET | /grid-factors | scope2-location:factors:read | List grid factors |
| GET | /grid-factors/{region} | scope2-location:factors:read | Get factor for region |
| POST | /grid-factors/custom | scope2-location:factors:write | Add custom factor |
| GET | /td-losses | scope2-location:factors:read | List T&D loss factors |
| POST | /compliance/check | scope2-location:compliance:check | Run compliance check |
| GET | /compliance/{id} | scope2-location:compliance:read | Get compliance result |
| POST | /uncertainty | scope2-location:uncertainty:run | Run uncertainty analysis |
| GET | /aggregations | scope2-location:read | Get aggregated emissions |
| GET | /health | — | Service health check |
| GET | /stats | — | Service statistics |

---

## 7. Prometheus Metrics (12)

All prefixed `gl_s2l_`:

1. `gl_s2l_calculations_total` — Counter (energy_type, calculation_method)
2. `gl_s2l_calculation_duration_seconds` — Histogram (energy_type)
3. `gl_s2l_emissions_co2e_tonnes` — Counter (energy_type, gas)
4. `gl_s2l_consumption_mwh_total` — Counter (energy_type, facility_type)
5. `gl_s2l_electricity_calculations_total` — Counter
6. `gl_s2l_steam_heat_cooling_calculations_total` — Counter (energy_type)
7. `gl_s2l_compliance_checks_total` — Counter (framework, status)
8. `gl_s2l_uncertainty_runs_total` — Counter (method)
9. `gl_s2l_errors_total` — Counter (error_type)
10. `gl_s2l_active_facilities` — Gauge
11. `gl_s2l_grid_factor_lookups_total` — Counter (source)
12. `gl_s2l_td_loss_adjustments_total` — Counter

---

## 8. Key Implementation Notes

### Zero-Hallucination Guarantees
- All Decimal arithmetic with ROUND_HALF_UP
- SHA-256 provenance hash at every calculation stage
- No LLM in any calculation path
- Deterministic results: same input → same output → same hash

### GHG Protocol Compliance
- Location-based method uses grid-average emission factors
- Must reflect actual grid mix where consumption occurs
- NOT contractual instruments (that's market-based = MRV-010)
- Dual reporting capability: location-based results can be combined with MRV-010

### Emission Factor Hierarchy
1. Custom factors (user-supplied, quality-scored)
2. National inventory factors (country-specific)
3. eGRID subregion factors (US only)
4. IEA country factors (global)
5. IPCC default factors (fallback)

### Unit Conversions
- 1 MWh = 3.6 GJ
- 1 GJ = 0.2778 MWh
- 1 MMBTU = 1.055 GJ
- 1 therm = 0.1055 GJ
- All conversions done in Decimal to prevent floating-point errors

---

## 9. Acceptance Criteria

1. Calculates Scope 2 location-based emissions for electricity, steam, heat, cooling
2. Supports 130+ countries (IEA) and 26 US eGRID subregions
3. Applies T&D loss adjustments with 50+ country factors
4. Per-gas breakdown (CO2, CH4, N2O) with GWP conversion (AR4/AR5/AR6)
5. 7-engine architecture following established MRV pattern
6. V060 database migration with 12 tables, 3 hypertables
7. 20 REST API endpoints with full RBAC
8. 12 Prometheus metrics with gl_s2l_ prefix
9. SHA-256 provenance chain for complete audit trail
10. 600+ unit tests with 85%+ coverage target
11. 7 regulatory framework compliance checks
12. Monte Carlo + analytical uncertainty quantification
