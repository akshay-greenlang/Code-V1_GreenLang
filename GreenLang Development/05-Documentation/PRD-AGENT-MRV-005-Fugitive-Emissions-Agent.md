# PRD: AGENT-MRV-005 Fugitive Emissions Agent (GL-MRV-SCOPE1-005)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE1-005 |
| **Internal Label** | AGENT-MRV-005 |
| **Category** | Layer 3 - MRV / Accounting Agents (Scope 1) |
| **Package** | `greenlang/fugitive_emissions/` |
| **DB Migration** | V055 |
| **Priority** | P1 - Critical |
| **Regulatory Drivers** | GHG Protocol Corporate Standard Ch.5, EPA 40 CFR Part 98 Subparts W/HH/II, EPA LDAR (40 CFR 60/61/63), CSRD/ESRS E1, ISO 14064-1, EU ETS MRR, EU Methane Regulation 2024/1787 |

## 2. Problem Statement

Fugitive emissions are unintentional GHG releases from equipment leaks, system losses, and uncontrolled venting in industrial operations. Distinct from refrigerant losses (covered by MRV-002), these include:

- Oil and natural gas system leaks (CH4) from production, processing, transmission, distribution
- Coal mine methane (CH4) from underground/surface mining and post-mining activities
- Equipment component leaks (valves, flanges, pumps, compressors, connectors)
- Wastewater treatment emissions (CH4 from anaerobic conditions, N2O from nitrification/denitrification)
- Industrial gas system losses (CO2, N2, compressed air systems)
- Tank storage losses (VOC/CH4 from breathing and working losses)
- Pneumatic device venting (high-bleed, low-bleed, intermittent)

## 3. Existing Layer 1 Capabilities

No dedicated fugitive emissions file exists. References to fugitive emissions are scattered across:
- `greenlang/agents/mrv/energy/fuel_supply_chain_mrv.py` - basic fugitive factors for fuel supply chains
- `greenlang/agents/mrv/waste/composting_mrv.py` - fugitive CH4 from composting

**Current gaps:** No LDAR tracking, no component-level emission factors, no EPA Subpart W methodology, no coal mine methane, no wastewater emissions, no pneumatic device calculations.

## 4. Gaps Requiring Layer 3 Production Implementation

| # | Gap | Impact |
|---|-----|--------|
| 1 | No equipment component-level leak factors | Cannot calculate emissions from valves, flanges, pumps |
| 2 | No LDAR program tracking | Cannot track leak detection and repair compliance |
| 3 | No oil/gas system methodology | Missing EPA Subpart W calculations |
| 4 | No coal mine methane calculation | Missing underground/surface mining CH4 |
| 5 | No wastewater treatment emissions | Missing CH4 and N2O from treatment processes |
| 6 | No pneumatic device calculations | Missing high/low-bleed venting emissions |
| 7 | No tank storage loss calculations | Missing breathing and working losses |
| 8 | No component screening data | Cannot use EPA emission factor approach |
| 9 | No correlation equations | Missing EPA Method 21 correlation |
| 10 | No multi-GWP support | Only single GWP source |
| 11 | No uncertainty quantification | No Monte Carlo for leak rate variability |
| 12 | No regulatory compliance | No LDAR/Subpart W/EU Methane Reg validation |

## 5. Architecture (7 Engines)

| Engine | Class | File | Purpose |
|--------|-------|------|---------|
| 1 | FugitiveSourceDatabaseEngine | fugitive_source_database.py | Source types, component emission factors, gas compositions |
| 2 | EmissionCalculatorEngine | emission_calculator.py | EPA average EF, screening ranges, engineering estimates, mass balance |
| 3 | LeakDetectionEngine | leak_detection.py | LDAR program tracking, OGI surveys, Method 21 screening |
| 4 | EquipmentComponentEngine | equipment_component.py | Component registry, leak rates, repair tracking, tank losses |
| 5 | UncertaintyQuantifierEngine | uncertainty_quantifier.py | Monte Carlo simulation, DQI scoring |
| 6 | ComplianceCheckerEngine | compliance_checker.py | EPA LDAR, Subpart W/HH/II, EU Methane Reg, GHG Protocol validation |
| 7 | FugitiveEmissionsPipelineEngine | fugitive_emissions_pipeline.py | 8-stage pipeline orchestration |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | SDK facade with graceful imports |
| `config.py` | GL_FUGITIVE_EMISSIONS_ env prefix, thread-safe singleton |
| `models.py` | Enums, Pydantic v2 models |
| `metrics.py` | 12 Prometheus metrics with gl_fe_ prefix |
| `provenance.py` | SHA-256 chain-hashed audit trails |
| `setup.py` | FugitiveEmissionsService facade |
| `api/router.py` | 20 REST endpoints at /api/v1/fugitive-emissions |

## 6. Key Features

### 6.1 Fugitive Source Categories (20 types)

**Oil & Natural Gas Systems:**
- Natural gas production (wellheads, separators, dehydrators, pneumatic controllers)
- Natural gas processing (compressors, acid gas removal, glycol dehydrators)
- Natural gas transmission (pipeline leaks, compressor stations, metering)
- Natural gas distribution (mains, services, meters, regulators)
- Crude oil production (tank batteries, wellheads, separators)
- Crude oil refining (process units, storage, loading)
- LNG facilities (liquefaction, storage, regasification)

**Coal Mining:**
- Underground mining (ventilation air methane, degasification)
- Surface mining (exposed coal seams, overburden)
- Post-mining (abandoned mines, coal handling, storage)

**Wastewater:**
- Industrial wastewater treatment (anaerobic lagoons, digesters)
- Municipal wastewater treatment (primary/secondary/tertiary)

**Equipment Components:**
- Valves (gas, light liquid, heavy liquid)
- Pumps (light liquid, heavy liquid)
- Compressor seals
- Pressure relief devices
- Connectors/flanges
- Open-ended lines
- Sampling connections

**Storage:**
- Fixed roof tanks (breathing/working losses)
- Floating roof tanks (rim seals, fittings, deck)
- Pressurized tanks

### 6.2 Emission Gases (4 primary)
- CH4 (primary fugitive gas from natural gas, coal, wastewater)
- CO2 (from acid gas removal, some natural gas venting)
- N2O (from wastewater nitrification/denitrification)
- VOCs (volatile organic compounds from storage, converted to CO2e)

### 6.3 Calculation Methods (5 methods)
1. **Average Emission Factor**: EPA default factors × component count
2. **Screening Ranges**: EPA Method 21 screening value × leak/no-leak factors
3. **Correlation Equations**: Log-linear regression from screening values
4. **Engineering Estimates**: Process-specific calculations (tank losses, pneumatics)
5. **Direct Measurement**: Hi-flow sampler, bagging, CEMS data

### 6.4 EPA Emission Factor Sources
- EPA Protocol for Equipment Leak Emission Estimates (EPA-453/R-95-017)
- EPA AP-42 Chapter 5 (Petroleum Industry) and Chapter 7 (Liquid Storage)
- EPA Subpart W Emission Factors
- EPA GHG Emission Factors Hub (2024)
- IPCC 2006 Guidelines Vol 2 Ch 4 (Fugitive Emissions)
- Custom/facility-specific factors

### 6.5 LDAR Program Features
- Survey scheduling (OGI quarterly, Method 21 annual, AVO daily)
- Component tagging and tracking
- Leak definition thresholds (10,000 ppm EPA, 500 ppm LDAR)
- Repair tracking with first/final attempt dates
- Delay of repair (DOR) justification
- LDAR compliance reporting per 40 CFR 60/61/63
- Survey coverage tracking
- Inspector certification tracking

### 6.6 Coal Mine Methane Features
- Ventilation air methane (VAM) flow × CH4 concentration
- Degasification well production rates
- Surface mine specific emission factors (m3 CH4/tonne coal)
- Post-mining emission decay curves
- Coal rank-based emission factors (anthracite through lignite)
- Mine depth and gas content adjustments

### 6.7 Wastewater Emission Features
- BOD/COD-based CH4 emission factors
- Maximum CH4 producing capacity (Bo)
- Methane correction factor (MCF) by treatment type
- Protein-based N2O emission factors (IPCC methodology)
- Effluent discharge N2O factors
- Sludge management emissions
- Biogas capture and utilization credit

### 6.8 Regulatory Frameworks (7)
1. **GHG Protocol Corporate Standard** (Chapter 5 - Fugitive)
2. **ISO 14064-1:2018** (Category 1)
3. **CSRD/ESRS E1** (E1-6 GHG emissions)
4. **EPA 40 CFR Part 98 Subpart W** (Petroleum and Natural Gas Systems)
5. **EPA 40 CFR Part 98 Subpart HH** (Municipal Solid Waste Landfills)
6. **EPA LDAR** (40 CFR 60/61/63 - Leak Detection and Repair)
7. **EU Methane Regulation 2024/1787** (Methane emissions in energy sector)

## 7. Database Schema (V055)

### Schema: fugitive_emissions_service

### Tables (10)
| Table | Purpose |
|-------|---------|
| fe_source_types | Fugitive source category definitions |
| fe_component_types | Equipment component type definitions and default EFs |
| fe_emission_factors | EFs by source type, component, method |
| fe_equipment_registry | Facility equipment and component registry |
| fe_ldar_surveys | LDAR survey records and results |
| fe_calculations | Individual emission calculations |
| fe_calculation_details | Per-gas emission breakdown |
| fe_leak_repairs | Leak repair tracking records |
| fe_compliance_records | Regulatory compliance check results |
| fe_audit_entries | Provenance and audit trail entries |

### Hypertables (3) - 7-day chunks
| Hypertable | Purpose |
|------------|---------|
| fe_calculation_events | Time-series calculation tracking |
| fe_survey_events | Time-series LDAR survey tracking |
| fe_compliance_events | Time-series compliance check tracking |

### Continuous Aggregates (2)
| Aggregate | Purpose |
|-----------|---------|
| fe_hourly_calculation_stats | Hourly calculation volume and emissions |
| fe_daily_emission_totals | Daily emission totals by source type |

### Row-Level Security
- All tables with tenant_id policy

### RBAC Permissions (18)
- fugitive-emissions:read, fugitive-emissions:write, fugitive-emissions:execute for each resource type

## 8. Prometheus Metrics (12)

All prefixed `gl_fe_`:

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | calculations_total | Counter | source_type, method, status |
| 2 | emissions_kg_co2e_total | Counter | source_type, gas |
| 3 | source_lookups_total | Counter | source |
| 4 | factor_selections_total | Counter | method, source |
| 5 | ldar_surveys_total | Counter | survey_type, status |
| 6 | uncertainty_runs_total | Counter | method |
| 7 | compliance_checks_total | Counter | framework, status |
| 8 | batch_jobs_total | Counter | status |
| 9 | calculation_duration_seconds | Histogram | operation |
| 10 | batch_size | Histogram | method |
| 11 | active_calculations | Gauge | - |
| 12 | components_registered | Gauge | component_type |

## 9. REST API (20 Endpoints)

Prefix: `/api/v1/fugitive-emissions`

| Method | Path | Description |
|--------|------|-------------|
| POST | /calculate | Calculate fugitive emissions |
| POST | /calculate/batch | Batch calculate emissions |
| GET | /calculations | List calculations |
| GET | /calculations/{calc_id} | Get calculation details |
| POST | /sources | Register fugitive source type |
| GET | /sources | List source types |
| GET | /sources/{source_id} | Get source details |
| POST | /components | Register equipment component |
| GET | /components | List components |
| GET | /components/{component_id} | Get component details |
| POST | /surveys | Log LDAR survey |
| GET | /surveys | List LDAR surveys |
| POST | /factors | Register custom emission factor |
| GET | /factors | List emission factors |
| POST | /repairs | Log leak repair |
| GET | /repairs | List repair records |
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
