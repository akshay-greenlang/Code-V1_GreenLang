# PRD: AGENT-MRV-011 — Scope 2 Steam/Heat Purchase Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-X-022 |
| **Internal Label** | AGENT-MRV-011 |
| **Category** | Layer 3 — MRV / Accounting Agents (Scope 2) |
| **Package** | `greenlang/steam_heat_purchase/` |
| **DB Migration** | V062 |
| **Metrics Prefix** | `gl_shp_` |
| **Table Prefix** | `shp_` |
| **API** | `/api/v1/steam-heat-purchase` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

Calculates **Scope 2 GHG emissions from purchased steam, district heating,
and district cooling** per the GHG Protocol Scope 2 Guidance (2015).
While MRV-009 and MRV-010 handle electricity-based Scope 2 emissions
(location-based and market-based respectively), this agent handles the
**non-electricity energy** portion of Scope 2 — steam delivered via
pipelines, district heating networks, and district cooling systems.

These are significant emission sources for:
- Industrial facilities purchasing process steam
- Commercial buildings on district heating/cooling networks
- Campuses and hospitals with central steam plants
- Food/beverage manufacturers purchasing process heat
- Data centers using district cooling

### Standards & References

- GHG Protocol Scope 2 Guidance (2015) — Chapters 3, 6, 7 (non-electricity energy)
- GHG Protocol Corporate Standard (Revised 2015) — Scope 2 boundary definition
- IPCC Guidelines (2006/2019 Refinement) Vol 2 — Stationary combustion EFs
- ISO 14064-1:2018 — Quantification of GHG emissions (Scope 2)
- CSRD/ESRS E1 — EU double materiality Scope 2 disclosure
- CDP Climate Change Questionnaire (2024) — C8.2 Scope 2 reporting
- SBTi Corporate Manual — Scope 2 target-setting
- EPA eGRID (2024) — US CHP allocation and grid factors
- DEFRA Conversion Factors (2024) — UK steam/heat emission factors
- Ecoinvent 3.10 — District heat/cooling system-level EFs
- IEA District Heating Report (2024) — Global DH emission factors
- EU Energy Efficiency Directive (2023/1791) — CHP allocation methods
- ASHRAE Standard 90.1 — Building cooling efficiency (COP/EER)

---

## 2. Steam, Heat & Cooling Methodology

### 2.1 Core Concept

Scope 2 includes emissions from **all purchased energy**, not just
electricity. When an organization purchases steam from a supplier,
the combustion of fuel to generate that steam occurs at the supplier's
facility — making the resulting GHG emissions the purchaser's Scope 2.
The same applies to district heating (hot water) and district cooling
(chilled water or absorption cooling).

### 2.2 Emission Source Categories

| Category | Description | Common Sources | Typical EF Range |
|----------|-------------|----------------|-----------------|
| **Purchased Steam** | Steam delivered via pipelines | Industrial boilers, CHP plants | 60-100 kgCO2e/GJ |
| **District Heating** | Hot water from DH networks | Gas boilers, waste incineration, geothermal | 30-120 kgCO2e/GJ |
| **District Cooling** | Chilled water from DC networks | Electric chillers, absorption chillers | 20-80 kgCO2e/GJ |
| **CHP Steam** | Steam from cogeneration plants | Combined heat & power plants | Allocation-dependent |

### 2.3 Core Formulas

**Direct steam/heat emissions (supplier EF known):**
```
Emissions (kgCO2e) = Consumption (GJ) x Supplier_EF (kgCO2e/GJ)
```

**Default steam emissions (fuel-based):**
```
Emissions (kgCO2e) = Consumption (GJ) / Boiler_Efficiency x Fuel_EF (kgCO2e/GJ_fuel)
```

**District heating emissions:**
```
Emissions (kgCO2e) = Consumption (GJ) x DH_Network_EF (kgCO2e/GJ)
```

**Electric-driven cooling (COP-based):**
```
Electrical_Input (kWh) = Cooling_Output (kWh_thermal) / COP
Emissions (kgCO2e) = Electrical_Input (kWh) x Grid_EF (kgCO2e/kWh)
```

**Absorption cooling (heat-driven):**
```
Heat_Input (GJ) = Cooling_Output (GJ_thermal) / COP_absorption
Emissions (kgCO2e) = Heat_Input (GJ) x Heat_Source_EF (kgCO2e/GJ)
```

**CHP allocation (efficiency method):**
```
Heat_Share = (Heat_Output / η_heat) / ((Heat_Output / η_heat) + (Power_Output / η_power))
Emissions_heat = Total_Fuel_Emissions x Heat_Share
```

**CHP allocation (energy method):**
```
Heat_Share = Heat_Output_GJ / (Heat_Output_GJ + Power_Output_GJ)
Emissions_heat = Total_Fuel_Emissions x Heat_Share
```

**CHP allocation (exergy method):**
```
Exergy_heat = Heat_Output x (1 - T_ambient / T_steam)
Heat_Share = Exergy_heat / (Exergy_heat + Power_Output)
Emissions_heat = Total_Fuel_Emissions x Heat_Share
```

### 2.4 Boiler Fuel Types & Default Emission Factors

| Fuel Type | EF (kgCO2/GJ) | EF CH4 (kgCH4/GJ) | EF N2O (kgN2O/GJ) | Default η |
|-----------|---------------|-------------------|-------------------|-----------|
| Natural Gas | 56.100 | 0.001 | 0.0001 | 0.85 |
| Fuel Oil #2 (Distillate) | 74.100 | 0.003 | 0.0006 | 0.82 |
| Fuel Oil #6 (Residual) | 77.400 | 0.003 | 0.0006 | 0.80 |
| Coal (Bituminous) | 94.600 | 0.001 | 0.0015 | 0.78 |
| Coal (Sub-bituminous) | 96.100 | 0.001 | 0.0015 | 0.75 |
| Coal (Lignite) | 101.000 | 0.001 | 0.0015 | 0.72 |
| LPG / Propane | 63.100 | 0.001 | 0.0001 | 0.85 |
| Biomass (Wood) | 112.000* | 0.030 | 0.004 | 0.70 |
| Biomass (Biogas) | 54.600* | 0.001 | 0.0001 | 0.80 |
| Municipal Waste | 91.700 | 0.030 | 0.004 | 0.65 |
| Waste Heat | 0.000 | 0.000 | 0.000 | 1.00 |
| Geothermal | 0.000 | 0.000 | 0.000 | 1.00 |
| Solar Thermal | 0.000 | 0.000 | 0.000 | 1.00 |
| Electric Boiler | Grid-dependent | 0.000 | 0.000 | 0.98 |

*Biomass CO2 is reported separately as biogenic; CH4 and N2O are non-biogenic.

### 2.5 District Heating Network Emission Factors

| Region / Network Type | EF (kgCO2e/GJ) | Notes |
|----------------------|-----------------|-------|
| Denmark (average) | 36.0 | High renewable share (biomass, waste) |
| Sweden (average) | 18.0 | Biomass-dominant |
| Finland (average) | 55.0 | CHP mix |
| Germany (average) | 72.0 | Gas-dominated |
| Poland (average) | 105.0 | Coal-dominated |
| Netherlands (average) | 58.0 | Gas-dominated |
| France (average) | 42.0 | Mix with waste heat |
| UK (average) | 65.0 | Gas-dominated |
| US (average) | 75.0 | Gas-dominated |
| China (average) | 110.0 | Coal-dominated |
| Japan (average) | 68.0 | Gas mix |
| South Korea (average) | 72.0 | Gas/waste mix |
| Global Default | 70.0 | IPCC default |

### 2.6 District Cooling System Types

| Cooling Technology | COP Range | Default COP | Energy Source |
|-------------------|-----------|-------------|--------------|
| Electric Chiller (Centrifugal) | 5.0 - 7.0 | 6.0 | Electricity |
| Electric Chiller (Screw) | 4.0 - 5.5 | 4.5 | Electricity |
| Electric Chiller (Reciprocating) | 3.5 - 5.0 | 4.0 | Electricity |
| Absorption Chiller (Single-Effect) | 0.6 - 0.8 | 0.7 | Steam/Heat |
| Absorption Chiller (Double-Effect) | 1.0 - 1.4 | 1.2 | Steam/Heat |
| Absorption Chiller (Triple-Effect) | 1.5 - 1.8 | 1.6 | Steam/Heat |
| Free Cooling (Sea/Lake Water) | 15.0 - 30.0 | 20.0 | Electricity (pumps) |
| Ice Storage | 3.0 - 4.5 | 3.5 | Electricity |
| Thermal Energy Storage | 4.0 - 6.0 | 5.0 | Electricity |

### 2.7 CHP Allocation Methods

Per GHG Protocol and EU Energy Efficiency Directive, three allocation
methods are supported:

| Method | Formula | Best For | Regulatory Use |
|--------|---------|----------|----------------|
| **Efficiency** | Based on individual heat/power efficiencies | Most common, GHG Protocol recommended | GHG Protocol, EPA |
| **Energy** | Based on energy output ratio | Simple allocation | DEFRA, some EU MS |
| **Exergy** | Based on thermodynamic quality (Carnot) | Thermodynamically rigorous | EU EED, academic |

### 2.8 Tier Levels

| Tier | Data Quality | Emission Factor Source | Uncertainty |
|------|-------------|----------------------|-------------|
| **Tier 1** | Default EFs | IPCC/EPA defaults | ±30-50% |
| **Tier 2** | Country/regional EFs | National inventories, DEFRA | ±15-30% |
| **Tier 3** | Supplier-specific EFs | Direct measurement, CHP meters | ±5-15% |

### 2.9 Market-Based vs Location-Based for Steam/Heat

Both methods apply to non-electricity energy:
- **Location-based**: Uses published default EFs for the region/network
- **Market-based**: Uses supplier-specific EFs if available, otherwise
  residual/default factors

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 AGENT-MRV-011                            │
│         Steam/Heat Purchase Agent                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 1: SteamHeatDatabaseEngine                │    │
│  │   - Boiler fuel EFs (14 fuel types)              │    │
│  │   - DH network EFs (13 regions)                  │    │
│  │   - Cooling system COPs (9 technologies)         │    │
│  │   - CHP default efficiencies                     │    │
│  │   - GWP values (AR4/AR5/AR6)                     │    │
│  │   - Unit conversions (GJ, MWh, kWh, MMBtu)       │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 2: SteamEmissionsCalculatorEngine          │    │
│  │   - Direct steam EF calculation                   │    │
│  │   - Fuel-based steam calculation (with η)         │    │
│  │   - Multi-fuel blended steam                      │    │
│  │   - Steam condensate return adjustment            │    │
│  │   - Biogenic CO2 separation                       │    │
│  │   - Per-gas breakdown (CO2, CH4, N2O)             │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 3: HeatCoolingCalculatorEngine             │    │
│  │   - District heating calculation                   │    │
│  │   - Electric cooling (COP-based)                   │    │
│  │   - Absorption cooling (heat-driven)               │    │
│  │   - Free cooling adjustment                        │    │
│  │   - Thermal storage losses                         │    │
│  │   - Network distribution losses                    │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 4: CHPAllocationEngine                     │    │
│  │   - Efficiency method allocation                   │    │
│  │   - Energy method allocation                       │    │
│  │   - Exergy method (Carnot)                         │    │
│  │   - Multi-product CHP (heat+power+cooling)         │    │
│  │   - Back-pressure vs extraction turbine             │    │
│  │   - Primary energy savings calculation              │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 5: UncertaintyQuantifierEngine             │    │
│  │   - Monte Carlo simulation (10,000 iterations)     │    │
│  │   - Analytical error propagation                   │    │
│  │   - Activity data uncertainty                      │    │
│  │   - Emission factor uncertainty                    │    │
│  │   - Efficiency/COP uncertainty                     │    │
│  │   - CHP allocation uncertainty                     │    │
│  │   - Combined uncertainty (quadrature)              │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 6: ComplianceCheckerEngine                 │    │
│  │   - GHG Protocol Scope 2 Guidance                 │    │
│  │   - ISO 14064-1:2018                               │    │
│  │   - CSRD/ESRS E1                                   │    │
│  │   - CDP Climate Change                             │    │
│  │   - SBTi Corporate Manual                          │    │
│  │   - EU EED (CHP allocation)                        │    │
│  │   - EPA Mandatory Reporting Rule                   │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 7: SteamHeatPipelineEngine                 │    │
│  │   - Full pipeline orchestration                    │    │
│  │   - DB 1→Calc 2/3→CHP 4→Uncertainty 5→Compliance 6│    │
│  │   - Batch processing                               │    │
│  │   - Aggregation (facility/fuel/period/source)      │    │
│  │   - Provenance chain assembly                      │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
greenlang/steam_heat_purchase/
├── __init__.py                        # Lazy imports, module exports
├── models.py                          # Pydantic v2 models, enums, constants
├── config.py                          # GL_SHP_ prefixed configuration
├── metrics.py                         # Prometheus metrics (gl_shp_*)
├── provenance.py                      # SHA-256 provenance chain
├── steam_heat_database.py             # Engine 1: EF database
├── steam_emissions_calculator.py      # Engine 2: Steam calculations
├── heat_cooling_calculator.py         # Engine 3: Heat & cooling
├── chp_allocation.py                  # Engine 4: CHP allocation
├── uncertainty_quantifier.py          # Engine 5: Uncertainty
├── compliance_checker.py              # Engine 6: Compliance
├── steam_heat_pipeline.py             # Engine 7: Pipeline
├── setup.py                           # Service facade (20+ methods)
└── api/
    ├── __init__.py                    # API package
    └── router.py                      # FastAPI REST endpoints

tests/unit/mrv/test_steam_heat_purchase/
├── __init__.py
├── conftest.py                        # Shared fixtures, autouse resets
├── test_models.py                     # Model validation tests
├── test_config.py                     # Configuration tests
├── test_metrics.py                    # Metrics registration tests
├── test_provenance.py                 # Provenance chain tests
├── test_steam_heat_database.py        # Engine 1 tests
├── test_steam_emissions_calculator.py # Engine 2 tests
├── test_heat_cooling_calculator.py    # Engine 3 tests
├── test_chp_allocation.py            # Engine 4 tests
├── test_uncertainty_quantifier.py     # Engine 5 tests
├── test_compliance_checker.py         # Engine 6 tests
├── test_steam_heat_pipeline.py        # Engine 7 tests
├── test_setup.py                      # Setup facade tests
└── test_api.py                        # API endpoint tests

deployment/database/migrations/sql/
└── V062__steam_heat_purchase_service.sql  # Database migration
```

### 3.3 Database Schema (V062)

14 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `shp_fuel_emission_factors` | Boiler fuel EFs (14 types) | Dimension |
| `shp_district_heating_factors` | DH network EFs by region | Dimension |
| `shp_cooling_system_factors` | Cooling technology COPs | Dimension |
| `shp_chp_defaults` | CHP default parameters | Dimension |
| `shp_facilities` | Facility registry with location | Dimension |
| `shp_steam_suppliers` | Steam/heat supplier info | Dimension |
| `shp_supplier_emission_factors` | Supplier-specific EFs | Dimension |
| `shp_calculations` | Calculation results | Hypertable |
| `shp_calculation_details` | Per-gas emission breakdown | Regular |
| `shp_chp_allocations` | CHP allocation results | Regular |
| `shp_uncertainty_results` | Uncertainty analysis | Hypertable |
| `shp_compliance_checks` | Compliance check results | Regular |
| `shp_batch_jobs` | Batch processing jobs | Regular |
| `shp_aggregations` | Aggregated results | Hypertable |
| `shp_hourly_stats` | Hourly calculation stats | Continuous Aggregate |
| `shp_daily_stats` | Daily calculation stats | Continuous Aggregate |

### 3.4 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/calculate/steam` | Calculate steam emissions |
| POST | `/calculate/heating` | Calculate district heating emissions |
| POST | `/calculate/cooling` | Calculate district cooling emissions |
| POST | `/calculate/chp` | Calculate CHP-allocated emissions |
| POST | `/calculate/batch` | Batch calculation |
| GET | `/factors/fuels` | List fuel emission factors |
| GET | `/factors/fuels/{fuel_type}` | Get specific fuel EF |
| GET | `/factors/heating/{region}` | Get DH network EF |
| GET | `/factors/cooling/{technology}` | Get cooling system COP |
| GET | `/factors/chp-defaults` | Get CHP default parameters |
| POST | `/facilities` | Register facility |
| GET | `/facilities/{facility_id}` | Get facility |
| POST | `/suppliers` | Register steam/heat supplier |
| GET | `/suppliers/{supplier_id}` | Get supplier with EF |
| POST | `/uncertainty` | Run uncertainty analysis |
| POST | `/compliance/check` | Run compliance check |
| GET | `/compliance/frameworks` | List available frameworks |
| POST | `/aggregate` | Aggregate results |
| GET | `/calculations/{calc_id}` | Get calculation result |
| GET | `/health` | Health check |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All calculations use Python `Decimal` with 8 decimal places
- No LLM calls in the calculation path
- Every step recorded in calculation trace
- SHA-256 provenance hash for every result
- Identical inputs always produce identical outputs
- All emission factors from published authoritative sources

### 4.2 Thread Safety

- Stateless per-calculation
- Mutable counters protected by `threading.RLock`
- Singleton pattern with autouse test resets

### 4.3 Integration Points

| Integration | Direction | Description |
|-------------|-----------|-------------|
| MRV-009 (Scope 2 Location) | Input | Grid EFs for electric cooling |
| MRV-010 (Scope 2 Market) | Input | Market EFs for electric cooling |
| MRV-007 (CHP/Cogeneration) | Input | CHP parameters for allocation |
| AGENT-FOUND-001 (Orchestrator) | Output | DAG pipeline integration |
| AGENT-FOUND-010 (Observability) | Output | Metrics and tracing |
| AGENT-DATA-001..020 | Input | Data quality and lineage |

### 4.4 Performance Targets

| Metric | Target |
|--------|--------|
| Single calculation latency | < 50ms |
| Batch (1,000 calculations) | < 5s |
| CHP allocation | < 100ms |
| Monte Carlo (10,000 iterations) | < 2s |
| Compliance check (all frameworks) | < 200ms |

### 4.5 Regulatory Frameworks (7)

1. **GHG Protocol Scope 2 Guidance** — Chapters 3, 6-7 non-electricity
2. **ISO 14064-1:2018** — Scope 2 steam/heat reporting
3. **CSRD/ESRS E1** — EU double materiality disclosure
4. **CDP Climate Change** — C8.2 Scope 2 reporting
5. **SBTi Corporate Manual** — Scope 2 target methodology
6. **EU Energy Efficiency Directive** — CHP allocation requirements
7. **EPA Mandatory Reporting Rule** — US CHP and steam reporting

---

## 5. Enumerations & Models

### 5.1 Enumerations (18)

| Enum | Values | Description |
|------|--------|-------------|
| `EnergyType` | STEAM, DISTRICT_HEATING, DISTRICT_COOLING, CHP_STEAM, CHP_HEATING | Energy source type |
| `FuelType` | NATURAL_GAS, FUEL_OIL_2, FUEL_OIL_6, COAL_BITUMINOUS, COAL_SUBBITUMINOUS, COAL_LIGNITE, LPG, BIOMASS_WOOD, BIOMASS_BIOGAS, MUNICIPAL_WASTE, WASTE_HEAT, GEOTHERMAL, SOLAR_THERMAL, ELECTRIC | Boiler fuel types |
| `CoolingTechnology` | CENTRIFUGAL_CHILLER, SCREW_CHILLER, RECIPROCATING_CHILLER, ABSORPTION_SINGLE, ABSORPTION_DOUBLE, ABSORPTION_TRIPLE, FREE_COOLING, ICE_STORAGE, THERMAL_STORAGE | Cooling technologies |
| `CHPAllocMethod` | EFFICIENCY, ENERGY, EXERGY | CHP allocation methods |
| `CalculationMethod` | DIRECT_EF, FUEL_BASED, COP_BASED, CHP_ALLOCATED | Calc methods |
| `EmissionGas` | CO2, CH4, N2O, CO2E, BIOGENIC_CO2 | Emission gases |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | IPCC assessment report |
| `ComplianceStatus` | COMPLIANT, NON_COMPLIANT, PARTIAL, NOT_APPLICABLE | Compliance status |
| `DataQualityTier` | TIER_1, TIER_2, TIER_3 | Data quality tier |
| `EnergyUnit` | GJ, MWH, KWH, MMBTU, THERM, MJ | Energy units |
| `TemperatureUnit` | CELSIUS, FAHRENHEIT, KELVIN | Temperature units |
| `SteamPressure` | LOW, MEDIUM, HIGH, VERY_HIGH | Steam pressure class |
| `SteamQuality` | SATURATED, SUPERHEATED, WET | Steam quality |
| `NetworkType` | MUNICIPAL, INDUSTRIAL, CAMPUS, MIXED | DH network type |
| `FacilityType` | INDUSTRIAL, COMMERCIAL, INSTITUTIONAL, RESIDENTIAL, DATA_CENTER, CAMPUS | Facility types |
| `ReportingPeriod` | MONTHLY, QUARTERLY, ANNUAL | Reporting periods |
| `AggregationType` | BY_FACILITY, BY_FUEL, BY_ENERGY_TYPE, BY_SUPPLIER, BY_PERIOD | Aggregation types |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED, PARTIAL | Batch job status |

### 5.2 Data Models (20)

| Model | Fields | Description |
|-------|--------|-------------|
| `FuelEmissionFactor` | fuel_type, co2_ef, ch4_ef, n2o_ef, default_efficiency, is_biogenic | Fuel EF record |
| `DistrictHeatingFactor` | region, network_type, ef_kgco2e_per_gj, distribution_loss_pct, source | DH network EF |
| `CoolingSystemFactor` | technology, cop_min, cop_max, cop_default, energy_source | Cooling system parameters |
| `CHPParameters` | electrical_efficiency, thermal_efficiency, fuel_type, power_output_mw, heat_output_mw | CHP plant specs |
| `FacilityInfo` | facility_id, name, facility_type, country, region, steam_suppliers, heating_network | Facility |
| `SteamSupplier` | supplier_id, name, fuel_mix, boiler_efficiency, supplier_ef_kgco2e_per_gj | Supplier |
| `SteamCalculationRequest` | facility_id, consumption_gj, supplier_id, fuel_type, boiler_efficiency, steam_pressure | Steam calc |
| `HeatingCalculationRequest` | facility_id, consumption_gj, region, network_type, supplier_ef | Heating calc |
| `CoolingCalculationRequest` | facility_id, cooling_output_gj, technology, cop, grid_ef_kwh | Cooling calc |
| `CHPAllocationRequest` | facility_id, total_fuel_gj, fuel_type, heat_output_gj, power_output_gj, method | CHP alloc |
| `GasEmissionDetail` | gas, emission_kg, gwp_value, gwp_source, co2e_kg | Per-gas detail |
| `CalculationResult` | calc_id, status, energy_type, total_co2e_kg, gas_details, trace, provenance_hash | Result |
| `CHPAllocationResult` | heat_share, power_share, heat_emissions_kg, power_emissions_kg, method | CHP result |
| `BatchCalculationRequest` | requests, tenant_id | Batch request |
| `BatchCalculationResult` | batch_id, results, total_co2e_kg, status | Batch result |
| `UncertaintyRequest` | calc_result, method, iterations, confidence_level | Uncertainty request |
| `UncertaintyResult` | mean, std_dev, ci_lower, ci_upper, confidence_level, method | Uncertainty |
| `ComplianceCheckResult` | framework, status, requirements, findings, score | Compliance result |
| `AggregationRequest` | calc_ids, aggregation_type | Aggregation request |
| `AggregationResult` | total_co2e_kg, breakdown, count, aggregation_type | Aggregation result |

---

## 6. Acceptance Criteria

- [ ] 14 boiler fuel emission factors from IPCC/EPA sources
- [ ] 13+ district heating network EFs by region
- [ ] 9 cooling technology COPs with min/max/default
- [ ] 3 CHP allocation methods (efficiency, energy, exergy)
- [ ] Tier 1/2/3 calculation support
- [ ] Biogenic CO2 separated from fossil CO2
- [ ] Steam condensate return adjustment
- [ ] Distribution network loss adjustment
- [ ] COP-based electric cooling with grid EF integration
- [ ] Absorption cooling with heat source EF
- [ ] Multi-product CHP allocation (heat+power+cooling)
- [ ] Monte Carlo and analytical uncertainty (10,000 iterations)
- [ ] 7 regulatory framework compliance checks
- [ ] 84+ compliance requirements
- [ ] 20 REST API endpoints
- [ ] V062 database migration (14 tables, 3 hypertables, 2 CAs)
- [ ] 1,000+ unit tests with 85%+ coverage
- [ ] SHA-256 provenance on every result
- [ ] Decimal arithmetic throughout
- [ ] Thread-safe singleton pattern
- [ ] Auth integration (route_protector.py + auth_setup.py)

---

## 7. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models |
| FastAPI | REST API |
| prometheus_client | Metrics |
| psycopg[binary] | PostgreSQL |
| TimescaleDB | Hypertables |
| numpy | Monte Carlo |

---

## 8. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-20 | Initial PRD |
