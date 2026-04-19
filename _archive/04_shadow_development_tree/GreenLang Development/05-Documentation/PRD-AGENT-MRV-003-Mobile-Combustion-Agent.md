# PRD: AGENT-MRV-003 Mobile Combustion Agent (GL-MRV-SCOPE1-003)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE1-003 |
| **Internal Label** | AGENT-MRV-003 |
| **Category** | Layer 3 - MRV / Accounting Agents (Scope 1) |
| **Package** | `greenlang/mobile_combustion/` |
| **DB Migration** | V053 |
| **Priority** | P1 - Critical |
| **Regulatory Drivers** | GHG Protocol Corporate Standard Ch.5, EPA 40 CFR Part 98 Subpart C, CSRD/ESRS E1, ISO 14064-1, UK SECR, EU ETS MRR |

## 2. Problem Statement

Mobile combustion sources (company-owned/controlled vehicles, off-road equipment, marine vessels, aviation, and rail) constitute a significant portion of Scope 1 GHG emissions for many organizations. Calculating these emissions accurately requires:

- Vehicle type-specific emission factors for CO2, CH4, and N2O
- Multiple calculation methodologies (fuel-based, distance-based, spend-based)
- Vehicle age and emission control technology adjustments for CH4/N2O
- Off-road equipment categories (construction, agriculture, industrial, mining)
- Marine vessel classifications and load-factor adjustments
- Aviation fuel burn rates and flight segment types
- Biofuel blending percentages and biogenic CO2 tracking
- Fleet-level aggregation with vehicle lifecycle tracking
- Regulatory compliance across multiple jurisdictions

## 3. Existing Layer 1 Capabilities

File: `greenlang/agents/mrv/scope1_combustion.py` (973 lines total, ~200 mobile-specific)

**Current capabilities:**
- Basic mobile combustion emission factors for 5 fuel types (gasoline, diesel, LPG, natural gas, jet fuel)
- Simple quantity-based calculation (fuel consumption x EF)
- Single GWP source (AR6 only)
- No vehicle type differentiation for CH4/N2O
- No distance-based calculation method
- No off-road equipment support
- No fleet management or vehicle lifecycle tracking

## 4. Gaps Requiring Layer 3 Production Implementation

| # | Gap | Impact |
|---|-----|--------|
| 1 | No vehicle type-specific CH4/N2O factors | Inaccurate non-CO2 emissions (up to 10x error) |
| 2 | No distance-based calculation method | Cannot use distance activity data |
| 3 | No emission control technology adjustment | CH4/N2O vary significantly by vehicle technology |
| 4 | No off-road equipment categories | Missing construction/mining/agriculture sources |
| 5 | No marine vessel emissions | Missing shipping/boat fleet emissions |
| 6 | No aviation emission factors | Missing corporate aviation emissions |
| 7 | No biofuel blending support | Cannot track renewable fuel percentages |
| 8 | No fleet management | No vehicle registry or lifecycle tracking |
| 9 | No multi-GWP support | Only AR6, missing AR4/AR5 for legacy reporting |
| 10 | No uncertainty quantification | No Monte Carlo or analytical uncertainty |
| 11 | No regulatory compliance checking | No GHG Protocol/EPA/CSRD validation |
| 12 | No multi-tier calculation | No Tier 1/2/3 methodology selection |
| 13 | No trip/journey tracking | Cannot link emissions to specific trips |
| 14 | No fleet analytics | No insights on emission intensity, efficiency trends |

## 5. Architecture (7 Engines)

| Engine | Class | File | Purpose |
|--------|-------|------|---------|
| 1 | VehicleDatabaseEngine | vehicle_database.py | Vehicle types, fuel types, emission factors database |
| 2 | EmissionCalculatorEngine | emission_calculator.py | Fuel-based, distance-based, spend-based calculations |
| 3 | FleetManagerEngine | fleet_manager.py | Vehicle registry, trip tracking, fleet analytics |
| 4 | DistanceEstimatorEngine | distance_estimator.py | Distance-based EFs, fuel economy, mode factors |
| 5 | UncertaintyQuantifierEngine | uncertainty_quantifier.py | Monte Carlo simulation, DQI scoring |
| 6 | ComplianceCheckerEngine | compliance_checker.py | GHG Protocol, EPA, CSRD, ISO 14064 validation |
| 7 | MobileCombustionPipelineEngine | mobile_combustion_pipeline.py | 8-stage pipeline orchestration |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | SDK facade with graceful imports |
| `config.py` | GL_MOBILE_COMBUSTION_ env prefix, thread-safe singleton |
| `models.py` | Enums, Pydantic v2 models |
| `metrics.py` | 12 Prometheus metrics with gl_mc_ prefix |
| `provenance.py` | SHA-256 chain-hashed audit trails |
| `setup.py` | MobileCombustionService facade |
| `api/router.py` | 20 REST endpoints at /api/v1/mobile-combustion |

## 6. Key Features

### 6.1 Vehicle Categories (18 types)
**On-Road:**
- Passenger car (gasoline, diesel, hybrid, plug-in hybrid, electric, CNG/LNG)
- Light-duty truck (gasoline, diesel)
- Medium-duty truck (gasoline, diesel)
- Heavy-duty truck (diesel)
- Bus/coach (diesel, CNG, electric)
- Motorcycle
- Van/light commercial vehicle

**Off-Road:**
- Construction equipment (excavators, loaders, bulldozers, cranes)
- Agricultural equipment (tractors, harvesters, combines)
- Industrial equipment (forklifts, generators, compressors)
- Mining equipment (haul trucks, drills, loaders)

**Marine:**
- Inland waterway vessels
- Coastal/short-sea vessels
- Ocean-going vessels

**Aviation:**
- Corporate/business jets
- Helicopters
- Turboprop aircraft

**Rail:**
- Diesel locomotives

### 6.2 Fuel Types (15 types)
Gasoline, diesel, biodiesel (B5/B20/B100), ethanol (E10/E85), CNG, LNG, LPG, propane, jet fuel (Jet-A/JP-8), aviation gasoline (Avgas), marine diesel oil (MDO), heavy fuel oil (HFO), sustainable aviation fuel (SAF)

### 6.3 Calculation Methods (3 methods)
1. **Fuel-based**: Fuel consumption x emission factor (Tier 1/2/3)
2. **Distance-based**: Distance x vehicle-specific emission factor (g CO2e/km or g CO2e/mile)
3. **Spend-based**: Fuel expenditure / fuel price x emission factor (screening method)

### 6.4 Emission Factor Sources (5 sources)
- EPA GHG Emission Factors Hub (2024)
- IPCC 2006 Guidelines / 2019 Refinement
- DEFRA/BEIS UK Conversion Factors
- EU ETS Monitoring and Reporting Regulation
- Custom/facility-specific factors

### 6.5 GWP Sources (4 sources)
- IPCC AR4 (SAR for legacy)
- IPCC AR5
- IPCC AR6 (100-year, default)
- IPCC AR6 (20-year)

### 6.6 Vehicle Technology & Control
- Emission control technology categories (uncontrolled, catalyst, advanced catalyst, Euro 1-6)
- Model year-based CH4/N2O adjustment factors
- Vehicle age degradation curves
- Fuel economy deterioration factors

### 6.7 Biofuel Blending
- Biodiesel blends (B5, B20, B100) - biogenic CO2 fraction tracking
- Ethanol blends (E10, E85) - biogenic CO2 fraction tracking
- Sustainable Aviation Fuel (SAF) - fossil/bio ratio
- Renewable natural gas (RNG) percentage
- Separate biogenic and fossil CO2 reporting per GHG Protocol

### 6.8 Fleet Management
- Vehicle registry with VIN, make, model, year, fuel type
- Trip/journey logging with start/end, distance, fuel consumed
- Fleet segmentation by vehicle type, department, location
- Vehicle lifecycle tracking (acquisition, operation, disposal)
- Fleet emission intensity metrics (gCO2e/km, gCO2e/tonne-km)

### 6.9 Regulatory Frameworks (6)
1. **GHG Protocol Corporate Standard** (Chapter 5)
2. **ISO 14064-1:2018** (Category 1)
3. **CSRD/ESRS E1** (E1-6 GHG emissions)
4. **EPA 40 CFR Part 98** (Subpart C)
5. **UK SECR** (Streamlined Energy and Carbon Reporting)
6. **EU ETS MRR** (Monitoring and Reporting Regulation)

## 7. Database Schema (V053)

### Schema: mobile_combustion_service

### Tables (10)
| Table | Purpose |
|-------|---------|
| mc_vehicle_types | Vehicle category definitions and default parameters |
| mc_fuel_types | Fuel properties, heating values, densities |
| mc_emission_factors | EFs by vehicle type, fuel type, source (CO2, CH4, N2O) |
| mc_vehicle_registry | Company vehicle fleet registration |
| mc_trips | Trip/journey records with distance and fuel data |
| mc_calculations | Individual emission calculations |
| mc_calculation_details | Per-gas emission breakdown |
| mc_fleet_aggregations | Fleet-level emission aggregations |
| mc_compliance_records | Regulatory compliance check results |
| mc_audit_entries | Provenance and audit trail entries |

### Hypertables (3) - 7-day chunks
| Hypertable | Purpose |
|------------|---------|
| mc_calculation_events | Time-series calculation tracking |
| mc_trip_events | Time-series trip/journey tracking |
| mc_compliance_events | Time-series compliance check tracking |

### Continuous Aggregates (2)
| Aggregate | Purpose |
|-----------|---------|
| mc_hourly_calculation_stats | Hourly calculation volume and emissions |
| mc_daily_emission_totals | Daily emission totals by vehicle type and fuel |

### Row-Level Security
- All tables with tenant_id policy

### RBAC Permissions (18)
- mobile-combustion:read, mobile-combustion:write, mobile-combustion:execute for each resource type

## 8. Prometheus Metrics (12)

All prefixed `gl_mc_`:

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | calculations_total | Counter | vehicle_type, method, status |
| 2 | emissions_kg_co2e_total | Counter | vehicle_type, fuel_type, gas |
| 3 | vehicle_lookups_total | Counter | source |
| 4 | factor_selections_total | Counter | method, source |
| 5 | fleet_operations_total | Counter | operation_type, vehicle_type |
| 6 | uncertainty_runs_total | Counter | method |
| 7 | compliance_checks_total | Counter | framework, status |
| 8 | batch_jobs_total | Counter | status |
| 9 | calculation_duration_seconds | Histogram | operation |
| 10 | batch_size | Histogram | method |
| 11 | active_calculations | Gauge | - |
| 12 | vehicles_registered | Gauge | vehicle_type |

## 9. REST API (20 Endpoints)

Prefix: `/api/v1/mobile-combustion`

| Method | Path | Description |
|--------|------|-------------|
| POST | /calculate | Calculate mobile combustion emissions |
| POST | /calculate/batch | Batch calculate emissions |
| GET | /calculations | List calculations |
| GET | /calculations/{calc_id} | Get calculation details |
| POST | /vehicles | Register a vehicle |
| GET | /vehicles | List registered vehicles |
| GET | /vehicles/{vehicle_id} | Get vehicle details |
| POST | /trips | Log a trip/journey |
| GET | /trips | List trips |
| GET | /trips/{trip_id} | Get trip details |
| POST | /fuels | Register custom fuel type |
| GET | /fuels | List fuel types |
| POST | /factors | Register custom emission factor |
| GET | /factors | List emission factors |
| POST | /aggregate | Aggregate fleet emissions |
| GET | /aggregations | List aggregations |
| POST | /uncertainty | Run uncertainty analysis |
| POST | /compliance/check | Run compliance check |
| GET | /health | Health check |
| GET | /stats | Service statistics |

## 10. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Single calculation latency | < 50ms |
| Batch calculation (100 records) | < 2s |
| Monte Carlo (5000 iterations) | < 5s |
| Decimal precision | 8+ decimal places |
| Zero-hallucination | No LLM in calculation path |
| Test coverage | 85%+ |
| Test count | 1000+ |

## 11. Dependencies

| Agent | Usage |
|-------|-------|
| AGENT-FOUND-001 (Orchestrator) | DAG execution |
| AGENT-FOUND-003 (Unit Normalizer) | Unit conversions |
| AGENT-FOUND-005 (Citations) | EF source citations |
| AGENT-FOUND-008 (Reproducibility) | Determinism verification |
| AGENT-FOUND-010 (Observability) | Metrics and tracing |
| AGENT-DATA-010 (Data Quality) | Input quality scoring |
| AGENT-MRV-001 (Stationary Combustion) | Shared Scope 1 infrastructure |
