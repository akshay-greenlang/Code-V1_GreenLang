# PRD: AGENT-MRV-012 — Scope 2 Cooling Purchase Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-X-023 |
| **Internal Label** | AGENT-MRV-012 |
| **Category** | Layer 3 — MRV / Accounting Agents (Scope 2) |
| **Package** | `greenlang/cooling_purchase/` |
| **DB Migration** | V063 |
| **Metrics Prefix** | `gl_cp_` |
| **Table Prefix** | `cp_` |
| **API** | `/api/v1/cooling-purchase` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

Calculates **Scope 2 GHG emissions from purchased district cooling and
cooling services** per the GHG Protocol Scope 2 Guidance (2015). While
MRV-011 handles steam and district heating, this agent handles the
**cooling-specific** portion of Scope 2 emissions with deep modeling of:

- Electric chiller technologies (centrifugal, screw, reciprocating, scroll)
- Absorption chiller systems (single/double/triple-effect)
- Free cooling from natural sources (seawater, lake, river, ambient air)
- Thermal energy storage (TES) with temporal emission shifting
- Part-load performance curves (IPLV/NPLV per AHRI 550/590)
- District cooling network distribution losses and pump energy
- Refrigerant leakage tracking (cross-reference with MRV-002)

### Justification for Dedicated Agent

Cooling differs fundamentally from steam/heat (MRV-011):
1. **COP/EER/IPLV efficiency metrics** unique to refrigeration cycles (0.6-30.0 range)
2. **18 distinct technologies** each with different COP curves and energy inputs
3. **Absorption cooling bridges thermal+electrical domains** (heat in, cooling out)
4. **Thermal storage shifts emissions temporally** (charge off-peak, discharge on-peak)
5. **Free cooling provides near-zero emissions** (seawater: 66% CO2 reduction)
6. **Part-load behavior** differs from design conditions (87% of hours at 50-75% load)
7. **Refrigerant leakage** creates cross-cutting Scope 1 emissions

### Standards & References

- GHG Protocol Scope 2 Guidance (2015) — Non-electricity energy, Appendix A
- AHRI Standard 550/590 — Water-chilling packages, IPLV/NPLV
- ASHRAE Standard 90.1 — Minimum chiller efficiency requirements
- ISO 14064-1:2018 — Scope 2 reporting for cooling
- CSRD/ESRS E1 — EU double materiality Scope 2 disclosure
- CDP Climate Change Questionnaire (2024) — C8.2 Scope 2
- SBTi Corporate Manual — Scope 2 target-setting
- EU Energy Efficiency Directive (2023/1791) — District cooling
- EU F-Gas Regulation (2024/573) — Refrigerant phase-down
- EPA eGRID (2024) — US grid factors for electric cooling
- DEFRA Conversion Factors (2024) — UK cooling emission factors
- IEA District Cooling Report (2024) — Global district cooling EFs

---

## 2. Cooling Purchase Methodology

### 2.1 Core Concept

Scope 2 includes emissions from **all purchased cooling** where the
generation occurs at the supplier's facility. This covers district
cooling networks, campus chiller plants operated by third parties,
and cooling-as-a-service contracts. The emissions depend on the
cooling generation technology, its efficiency (COP), and the energy
source powering it.

### 2.2 Cooling Technologies (18)

| Category | Technology | COP Range | Default COP | IPLV | Energy Source |
|----------|-----------|-----------|-------------|------|--------------|
| **Electric - Centrifugal** | Water-cooled centrifugal | 5.0-7.5 | 6.1 | 9.0 | Electricity |
| **Electric - Centrifugal** | Air-cooled centrifugal | 2.8-3.8 | 3.2 | 4.5 | Electricity |
| **Electric - Screw** | Water-cooled screw | 4.0-5.5 | 4.7 | 6.5 | Electricity |
| **Electric - Screw** | Air-cooled screw | 2.5-3.5 | 3.0 | 4.0 | Electricity |
| **Electric - Reciprocating** | Water-cooled recip | 3.5-5.0 | 4.2 | 5.5 | Electricity |
| **Electric - Scroll** | Air-cooled scroll | 2.5-3.5 | 2.8 | 3.8 | Electricity |
| **Absorption - Single** | Single-effect LiBr | 0.6-0.8 | 0.70 | 0.72 | Steam/Hot water |
| **Absorption - Double** | Double-effect LiBr | 1.0-1.4 | 1.20 | 1.30 | Steam/Direct fire |
| **Absorption - Triple** | Triple-effect LiBr | 1.4-1.8 | 1.60 | 1.70 | Direct fire |
| **Absorption - NH3** | Ammonia absorption | 0.5-0.7 | 0.55 | 0.58 | Steam/Waste heat |
| **Free Cooling** | Seawater | 15.0-30.0 | 20.0 | -- | Electricity (pumps) |
| **Free Cooling** | Lake water | 12.0-25.0 | 18.0 | -- | Electricity (pumps) |
| **Free Cooling** | River water | 10.0-20.0 | 15.0 | -- | Electricity (pumps) |
| **Free Cooling** | Ambient air (dry cooler) | 8.0-15.0 | 10.0 | -- | Electricity (fans) |
| **TES - Ice** | Ice storage | 2.8-4.0 | 3.2 | -- | Electricity |
| **TES - Chilled Water** | Chilled water storage | 4.5-6.5 | 5.5 | -- | Electricity |
| **TES - PCM** | Phase-change material | 3.5-5.5 | 4.5 | -- | Electricity |
| **District** | District cooling network | Varies | 4.0 | -- | Mixed |

### 2.3 Core Formulas

**Electric chiller emissions (full load):**
```
Electrical_Input (kWh) = Cooling_Output (kWh_th) / COP
Emissions (kgCO2e) = Electrical_Input (kWh) x Grid_EF (kgCO2e/kWh)
```

**Electric chiller (IPLV part-load weighted):**
```
IPLV = 0.01 x COP_100% + 0.42 x COP_75% + 0.45 x COP_50% + 0.12 x COP_25%
Electrical_Input (kWh) = Cooling_Output (kWh_th) / IPLV
Emissions (kgCO2e) = Electrical_Input (kWh) x Grid_EF
```

**Absorption chiller emissions:**
```
Heat_Input (GJ) = Cooling_Output (GJ) / COP_absorption
Parasitic_Electricity (kWh) = Cooling_Output (kWh_th) x Parasitic_Ratio
Emissions_thermal (kgCO2e) = Heat_Input (GJ) x Heat_Source_EF (kgCO2e/GJ)
Emissions_electric (kgCO2e) = Parasitic_Electricity (kWh) x Grid_EF
Total_Emissions = Emissions_thermal + Emissions_electric
```

**Free cooling emissions:**
```
Pump_Energy (kWh) = Cooling_Output (kWh_th) / COP_free
Emissions (kgCO2e) = Pump_Energy (kWh) x Grid_EF (kgCO2e/kWh)
```

**Thermal energy storage (TES):**
```
Charge_Energy (kWh) = (TES_Capacity (kWh_th) / COP_charge) / Round_Trip_Efficiency
Emissions (kgCO2e) = Charge_Energy (kWh) x Grid_EF_at_charge_time
Emission_Savings = Grid_EF_peak x (Cooling / COP_peak) - Grid_EF_offpeak x Charge_Energy
```

**District cooling network:**
```
Adjusted_Cooling = Cooling_Output / (1 - Distribution_Loss_Pct)
Pump_Emissions = Pump_Energy_kWh x Grid_EF
Generation_Emissions = Adjusted_Cooling / COP_plant x Generation_EF
Total = Generation_Emissions + Pump_Emissions
```

**Refrigerant leakage (informational, cross-ref MRV-002):**
```
Leakage_Emissions (kgCO2e) = Charge_kg x Annual_Leak_Rate x GWP_refrigerant
```

### 2.4 Efficiency Metric Conversions

| From | To | Formula |
|------|----|---------|
| COP | EER | EER = COP x 3.412 |
| EER | COP | COP = EER / 3.412 |
| COP | kW/ton | kW/ton = 3.517 / COP |
| kW/ton | COP | COP = 3.517 / kW_per_ton |
| SEER | COP | COP = SEER / 3.412 |

### 2.5 Part-Load Performance (AHRI 550/590)

| Load % | Weighting | Typical COP Multiplier |
|--------|-----------|----------------------|
| 100% | 1% | 1.00 |
| 75% | 42% | 1.15 |
| 50% | 45% | 1.30 |
| 25% | 12% | 1.10 |

IPLV is typically 30-50% higher than full-load COP.

### 2.6 District Cooling Network Emission Factors

| Region / Network | Default EF (kgCO2e/GJ_cooling) | Technology Mix | Notes |
|-----------------|-------------------------------|---------------|-------|
| Dubai/UAE | 45.0 | Electric + absorption | High grid carbon |
| Singapore | 35.0 | Electric centrifugal | Efficient district |
| Hong Kong | 38.0 | Electric centrifugal | Sea-cooled |
| US Sun Belt | 42.0 | Electric mixed | High cooling demand |
| EU Nordic | 12.0 | Free cooling + electric | Seawater/lake |
| EU Central | 25.0 | Electric + absorption | Mixed |
| Japan | 32.0 | Electric + absorption | High efficiency |
| South Korea | 35.0 | Electric centrifugal | LNG-heavy grid |
| India | 55.0 | Electric mixed | High grid carbon |
| Australia | 48.0 | Electric centrifugal | Coal-heavy grid |
| China | 52.0 | Electric + absorption | Coal grid |
| Global Default | 40.0 | Mixed | IEA estimate |

### 2.7 Heat Source Emission Factors (for Absorption Chillers)

| Heat Source | EF (kgCO2e/GJ) | Notes |
|------------|-----------------|-------|
| Natural gas steam | 70.1 | 56.1/0.80 boiler efficiency |
| District heating | 70.0 | Global default |
| Waste heat (industrial) | 0.0 | Zero-cost byproduct |
| CHP exhaust heat | CHP-allocated | Cross-ref MRV-011 |
| Solar thermal | 0.0 | Zero operational emissions |
| Geothermal | 0.0 | Zero operational emissions |
| Biogas steam | 0.0* | Biogenic (CH4/N2O counted) |
| Fuel oil steam | 96.8 | 77.4/0.80 |
| Coal steam | 126.1 | 94.6/0.75 |
| Electric boiler | Grid-dependent | Grid EF / 0.98 |
| Heat pump output | Grid-dependent | Grid EF / COP_HP |

### 2.8 Refrigerant Data (Common Chillers)

| Refrigerant | GWP (AR5) | GWP (AR6) | Common Use | Phase-Down |
|-------------|-----------|-----------|------------|------------|
| R-134a | 1,430 | 1,530 | Centrifugal chillers | Kigali 2029 |
| R-410A | 2,088 | 2,088 | Scroll/screw | Kigali 2024+ |
| R-407C | 1,774 | 1,774 | Recip/scroll | Kigali 2024+ |
| R-32 | 675 | 771 | New split systems | Transitional |
| R-1234ze(E) | 7 | 7 | Low-GWP centrifugal | Next-gen |
| R-1234yf | 4 | <1 | Automotive, small | Next-gen |
| R-513A | 631 | 631 | Drop-in for R-134a | Transitional |
| R-514A | 2 | 2 | Low-GWP centrifugal | Next-gen |
| R-290 (Propane) | 3 | 0.02 | Small commercial | Natural ref |
| R-717 (NH3) | 0 | 0 | Industrial | Natural ref |
| R-718 (Water/LiBr) | 0 | 0 | Absorption chillers | N/A |

### 2.9 Tier Levels

| Tier | Data Quality | COP Source | Uncertainty |
|------|-------------|------------|-------------|
| **Tier 1** | Default COP + IPCC EFs | Technology category defaults | ±30-50% |
| **Tier 2** | Nameplate COP + regional EFs | Manufacturer specs, IPLV | ±15-25% |
| **Tier 3** | Measured COP + supplier EFs | Metered performance data | ±5-15% |

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 AGENT-MRV-012                            │
│          Cooling Purchase Agent                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 1: CoolingDatabaseEngine                  │    │
│  │   - 18 cooling technology COPs (full + IPLV)     │    │
│  │   - 12 district cooling regional EFs             │    │
│  │   - 11 heat source EFs (for absorption)          │    │
│  │   - 11 refrigerant GWP values                    │    │
│  │   - 6 efficiency metric conversions              │    │
│  │   - 9 cooling unit conversions                   │    │
│  │   - Part-load curves (AHRI 550/590)              │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 2: ElectricChillerCalculatorEngine         │    │
│  │   - Full-load COP calculation                     │    │
│  │   - IPLV/NPLV part-load weighted calculation      │    │
│  │   - Condenser type adjustment (water/air)         │    │
│  │   - Auxiliary energy (pumps, fans, cooling towers) │    │
│  │   - Multi-chiller plant optimization              │    │
│  │   - Per-gas breakdown (CO2, CH4, N2O via grid)    │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 3: AbsorptionCoolingCalculatorEngine       │    │
│  │   - Single/double/triple-effect calculations      │    │
│  │   - 11 heat source emission factors               │    │
│  │   - Waste heat zero-path optimization             │    │
│  │   - Parasitic electricity calculation             │    │
│  │   - CHP integration (cross-ref MRV-011)           │    │
│  │   - Hybrid absorption+electric plants             │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 4: DistrictCoolingCalculatorEngine         │    │
│  │   - Free cooling (seawater/lake/river/air)        │    │
│  │   - Thermal energy storage (ice/CW/PCM)           │    │
│  │   - Temporal emission shifting (TOU grid EFs)     │    │
│  │   - Distribution network losses                   │    │
│  │   - District cooling network EFs by region        │    │
│  │   - Multi-source plant aggregation                │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 5: UncertaintyQuantifierEngine             │    │
│  │   - Monte Carlo simulation (10,000 iterations)     │    │
│  │   - Analytical error propagation                   │    │
│  │   - COP uncertainty (design vs operating)          │    │
│  │   - Part-load weighting uncertainty                │    │
│  │   - Grid EF temporal uncertainty                   │    │
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
│  │   - ASHRAE 90.1 (efficiency standards)             │    │
│  │   - EU F-Gas Regulation (refrigerant tracking)     │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 7: CoolingPurchasePipelineEngine           │    │
│  │   - Full pipeline orchestration                    │    │
│  │   - DB 1→Calc 2/3/4→Uncertainty 5→Compliance 6    │    │
│  │   - Batch processing                               │    │
│  │   - Aggregation (facility/tech/period/region)      │    │
│  │   - Provenance chain assembly                      │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
greenlang/cooling_purchase/
├── __init__.py                          # Lazy imports, module exports
├── models.py                            # Pydantic v2 models, enums, constants
├── config.py                            # GL_CP_ prefixed configuration
├── metrics.py                           # Prometheus metrics (gl_cp_*)
├── provenance.py                        # SHA-256 provenance chain
├── cooling_database.py                  # Engine 1: Technology database
├── electric_chiller_calculator.py       # Engine 2: Electric chiller calcs
├── absorption_cooling_calculator.py     # Engine 3: Absorption chiller calcs
├── district_cooling_calculator.py       # Engine 4: District/free/TES calcs
├── uncertainty_quantifier.py            # Engine 5: Uncertainty
├── compliance_checker.py                # Engine 6: Compliance
├── cooling_purchase_pipeline.py         # Engine 7: Pipeline
├── setup.py                             # Service facade
└── api/
    ├── __init__.py                      # API package
    └── router.py                        # FastAPI REST endpoints

tests/unit/mrv/test_cooling_purchase/
├── __init__.py
├── conftest.py
├── test_models.py
├── test_config.py
├── test_metrics.py
├── test_provenance.py
├── test_cooling_database.py
├── test_electric_chiller_calculator.py
├── test_absorption_cooling_calculator.py
├── test_district_cooling_calculator.py
├── test_uncertainty_quantifier.py
├── test_compliance_checker.py
├── test_cooling_purchase_pipeline.py
├── test_setup.py
└── test_api.py

deployment/database/migrations/sql/
└── V063__cooling_purchase_service.sql
```

### 3.3 Database Schema (V063)

14 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `cp_cooling_technologies` | 18 technology specs with COP profiles | Dimension |
| `cp_district_cooling_factors` | Regional district cooling EFs | Dimension |
| `cp_heat_source_factors` | Heat source EFs for absorption | Dimension |
| `cp_refrigerant_data` | Refrigerant GWP values | Dimension |
| `cp_part_load_curves` | AHRI 550/590 part-load data | Dimension |
| `cp_facilities` | Facility registry | Dimension |
| `cp_cooling_suppliers` | Cooling service suppliers | Dimension |
| `cp_calculations` | Calculation results | Hypertable |
| `cp_calculation_details` | Per-component breakdown | Regular |
| `cp_tes_calculations` | TES temporal shifting results | Regular |
| `cp_uncertainty_results` | Uncertainty analysis | Hypertable |
| `cp_compliance_checks` | Compliance check results | Regular |
| `cp_batch_jobs` | Batch processing jobs | Regular |
| `cp_aggregations` | Aggregated results | Hypertable |
| `cp_hourly_stats` | Hourly calculation stats | Continuous Aggregate |
| `cp_daily_stats` | Daily calculation stats | Continuous Aggregate |

### 3.4 API Endpoints (20)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/calculate/electric` | Electric chiller emissions |
| POST | `/calculate/absorption` | Absorption chiller emissions |
| POST | `/calculate/district` | District cooling emissions |
| POST | `/calculate/free-cooling` | Free cooling emissions |
| POST | `/calculate/tes` | TES temporal shifting |
| POST | `/calculate/batch` | Batch calculation |
| GET | `/technologies` | List 18 cooling technologies |
| GET | `/technologies/{tech_id}` | Get specific technology COP |
| GET | `/factors/district/{region}` | Get district cooling EF |
| GET | `/factors/heat-source/{source}` | Get heat source EF |
| GET | `/factors/refrigerants` | List refrigerant GWPs |
| POST | `/facilities` | Register facility |
| GET | `/facilities/{facility_id}` | Get facility |
| POST | `/suppliers` | Register cooling supplier |
| GET | `/suppliers/{supplier_id}` | Get supplier |
| POST | `/uncertainty` | Run uncertainty analysis |
| POST | `/compliance/check` | Run compliance check |
| GET | `/compliance/frameworks` | List frameworks |
| POST | `/aggregate` | Aggregate results |
| GET | `/health` | Health check |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees
- All COP/EER/IPLV calculations use Python `Decimal` (8 decimal places)
- No LLM calls in calculation path
- Every step recorded in calculation trace
- SHA-256 provenance hash for every result

### 4.2 Enumerations (18)

| Enum | Values | Description |
|------|--------|-------------|
| `CoolingTechnology` | 18 values | All cooling generation technologies |
| `CompressorType` | CENTRIFUGAL, SCREW, RECIPROCATING, SCROLL | Electric chiller compressor types |
| `CondenserType` | WATER_COOLED, AIR_COOLED | Condenser cooling method |
| `AbsorptionType` | SINGLE_EFFECT, DOUBLE_EFFECT, TRIPLE_EFFECT, AMMONIA | Absorption chiller types |
| `FreeCoolingSource` | SEAWATER, LAKE, RIVER, AMBIENT_AIR | Free cooling sources |
| `TESType` | ICE, CHILLED_WATER, PCM | Thermal energy storage types |
| `HeatSource` | 11 values | Absorption chiller heat sources |
| `EfficiencyMetric` | COP, EER, KW_PER_TON, IPLV, NPLV, SEER | Efficiency metrics |
| `CoolingUnit` | TON_HOUR, KWH_TH, GJ, BTU, MMBTU, MJ, TR | Cooling energy units |
| `EmissionGas` | CO2, CH4, N2O, CO2E | Emission gases |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | IPCC GWP sources |
| `ComplianceStatus` | COMPLIANT, NON_COMPLIANT, PARTIAL, NOT_APPLICABLE | Compliance states |
| `DataQualityTier` | TIER_1, TIER_2, TIER_3 | Data quality tiers |
| `FacilityType` | COMMERCIAL, DATA_CENTER, HOSPITAL, CAMPUS, INDUSTRIAL, DISTRICT | Facility types |
| `ReportingPeriod` | MONTHLY, QUARTERLY, ANNUAL | Reporting periods |
| `AggregationType` | BY_FACILITY, BY_TECHNOLOGY, BY_REGION, BY_SUPPLIER, BY_PERIOD | Aggregation |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED, PARTIAL | Batch status |
| `Refrigerant` | 11 values | Common refrigerant types |

### 4.3 Regulatory Frameworks (7)

1. **GHG Protocol Scope 2 Guidance** — Non-electricity cooling
2. **ISO 14064-1:2018** — Scope 2 cooling reporting
3. **CSRD/ESRS E1** — EU double materiality disclosure
4. **CDP Climate Change** — C8.2 Scope 2 reporting
5. **SBTi Corporate Manual** — Scope 2 target methodology
6. **ASHRAE 90.1** — Minimum chiller efficiency compliance
7. **EU F-Gas Regulation** — Refrigerant tracking and phase-down

### 4.4 Performance Targets

| Metric | Target |
|--------|--------|
| Single calculation latency | < 50ms |
| Batch (1,000 calculations) | < 5s |
| IPLV calculation | < 10ms |
| Monte Carlo (10,000 iterations) | < 2s |
| Compliance check (all frameworks) | < 200ms |

---

## 5. Acceptance Criteria

- [ ] 18 cooling technology COP profiles (full load + IPLV)
- [ ] 12 district cooling regional emission factors
- [ ] 11 heat source EFs for absorption chillers
- [ ] 11 refrigerant GWP values (AR5 + AR6)
- [ ] IPLV/NPLV part-load weighted calculation per AHRI 550/590
- [ ] 6 efficiency metric conversions (COP/EER/kW_ton/IPLV/NPLV/SEER)
- [ ] Electric chiller calculation with condenser type adjustment
- [ ] Absorption chiller with parasitic electricity
- [ ] Free cooling from 4 natural sources
- [ ] TES temporal emission shifting with time-of-use grid EFs
- [ ] District cooling network losses and pump energy
- [ ] Refrigerant leakage tracking (informational)
- [ ] Monte Carlo and analytical uncertainty
- [ ] 7 regulatory framework compliance checks (84 requirements)
- [ ] 20 REST API endpoints
- [ ] V063 database migration (14 tables, 3 hypertables, 2 CAs)
- [ ] 1,000+ unit tests
- [ ] SHA-256 provenance on every result
- [ ] Auth integration (route_protector.py + auth_setup.py)

---

## 6. Dependencies

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

## 7. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-22 | Initial PRD |
