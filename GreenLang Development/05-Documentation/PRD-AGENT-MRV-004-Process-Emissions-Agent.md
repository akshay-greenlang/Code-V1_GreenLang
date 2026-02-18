# PRD: AGENT-MRV-004 Process Emissions Agent (GL-MRV-SCOPE1-004)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE1-004 |
| **Internal Label** | AGENT-MRV-004 |
| **Category** | Layer 3 - MRV / Accounting Agents (Scope 1) |
| **Package** | `greenlang/process_emissions/` |
| **DB Migration** | V054 |
| **Priority** | P1 - Critical |
| **Regulatory Drivers** | GHG Protocol Corporate Standard Ch.5, EPA 40 CFR Part 98 Subparts F/H/N/O/Q/S/X/Y/Z/AA/BB/CC/DD/EE/FF/GG, CSRD/ESRS E1, ISO 14064-1, EU ETS MRR, UK SECR |

## 2. Problem Statement

Industrial process emissions are non-combustion GHG emissions released during chemical or physical transformations of raw materials. These constitute a major portion of Scope 1 emissions for manufacturing, mining, and chemical companies. Calculating these emissions accurately requires:

- Process-type-specific emission factors for CO2, CH4, N2O, PFCs, SF6, NF3, and HFCs
- Multiple calculation methodologies (mass balance, stoichiometric, emission factor, direct measurement)
- Material composition and carbonate content tracking
- Abatement technology effectiveness and destruction efficiencies
- Process-specific parameters (clinker ratio, slag ratio, EAF vs BOF split)
- By-product and co-product emission credit tracking
- Raw material composition variability
- Tier 1/2/3 methodology selection per IPCC guidelines
- Regulatory compliance across multiple EPA subparts and EU ETS MRR
- Uncertainty quantification with process-specific ranges

## 3. Existing Layer 1 Capabilities

File: `greenlang/agents/mrv/process_emissions.py` (335 lines)

**Current capabilities:**
- 11 process types with simple emission factor lookups
- Single CO2/N2O/PFC emission factor per process
- Basic abatement efficiency multiplier
- No material balance methodology
- No process-specific parameters
- No multi-GWP support
- No Tier 1/2/3 distinction
- No raw material tracking
- No by-product credits
- No regulatory compliance checking

## 4. Gaps Requiring Layer 3 Production Implementation

| # | Gap | Impact |
|---|-----|--------|
| 1 | Only 11 process types | Missing 14+ major industrial processes |
| 2 | No mass balance methodology | Cannot use raw material composition data |
| 3 | No process-specific parameters | Inaccurate clinker/slag/reduction ratios |
| 4 | No abatement technology database | Cannot track abatement effectiveness |
| 5 | No by-product credits | Overstates emissions from co-production |
| 6 | No raw material composition | Cannot account for carbonate variability |
| 7 | No multi-GWP support | Only AR6, missing AR4/AR5 for legacy |
| 8 | No Tier 1/2/3 methodology | Cannot select IPCC-appropriate methodology |
| 9 | No uncertainty quantification | No Monte Carlo or DQI scoring |
| 10 | No regulatory compliance | No EPA subpart / EU ETS / CSRD validation |
| 11 | No PFC modeling for aluminum | Missing anode effect frequency/duration |
| 12 | No facility/process unit tracking | Cannot aggregate by production line |
| 13 | No batch vs continuous distinction | Cannot apply correct temporal factors |
| 14 | No sector analytics | No benchmarking or efficiency metrics |

## 5. Architecture (7 Engines)

| Engine | Class | File | Purpose |
|--------|-------|------|---------|
| 1 | ProcessDatabaseEngine | process_database.py | Process types, raw materials, emission factors, stoichiometric data |
| 2 | EmissionCalculatorEngine | emission_calculator.py | Mass balance, stoichiometric, EF-based, direct measurement calculations |
| 3 | MaterialBalanceEngine | material_balance.py | Raw material tracking, carbonate content, by-product credits |
| 4 | AbatementTrackerEngine | abatement_tracker.py | Abatement technologies, destruction efficiencies, recovery rates |
| 5 | UncertaintyQuantifierEngine | uncertainty_quantifier.py | Monte Carlo simulation, DQI scoring, process-specific ranges |
| 6 | ComplianceCheckerEngine | compliance_checker.py | EPA subparts, EU ETS MRR, CSRD, GHG Protocol, ISO 14064 validation |
| 7 | ProcessEmissionsPipelineEngine | process_emissions_pipeline.py | 8-stage pipeline orchestration |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | SDK facade with graceful imports |
| `config.py` | GL_PROCESS_EMISSIONS_ env prefix, thread-safe singleton |
| `models.py` | Enums, Pydantic v2 models |
| `metrics.py` | 12 Prometheus metrics with gl_pe_ prefix |
| `provenance.py` | SHA-256 chain-hashed audit trails |
| `setup.py` | ProcessEmissionsService facade |
| `api/router.py` | 20 REST endpoints at /api/v1/process-emissions |

## 6. Key Features

### 6.1 Industrial Process Categories (25 types)

**Mineral Industry:**
- Cement production (clinker calcination)
- Lime production (quicklime, hydrated lime, dolomitic lime)
- Glass production (flat, container, fiber, specialty)
- Ceramics production (bricks, tiles, refractories)
- Soda ash production (natural, Solvay process)

**Chemical Industry:**
- Ammonia production (steam methane reforming, partial oxidation)
- Nitric acid production (N2O from ammonia oxidation)
- Adipic acid production (N2O from cyclohexanone oxidation)
- Carbide production (calcium carbide, silicon carbide)
- Petrochemical production (ethylene, methanol, carbon black, ethylene dichloride)
- Hydrogen production (SMR, ATR, partial oxidation)
- Phosphoric acid production
- Titanium dioxide production (chloride, sulfate process)

**Metal Industry:**
- Iron and steel (blast furnace-BOF, EAF, direct reduction)
- Aluminum smelting (prebake, Soderberg anode, PFC emissions)
- Ferroalloy production (ferrosilicon, ferrochromium, ferromanganese)
- Lead production (primary, secondary)
- Zinc production (electrolytic, Imperial Smelting)
- Magnesium production (electrolytic, Pidgeon, SF6 cover gas)
- Copper smelting

**Electronics:**
- Semiconductor manufacturing (PFC/HFC/NF3/SF6 etching and CVD)

**Pulp & Paper:**
- Pulp and paper (lime kiln, chemical recovery, tall oil)

**Other:**
- Mineral wool production (stone wool, glass wool)
- Carbon anode consumption
- Food and drink (sugar, brewing CO2)

### 6.2 Emission Gases (7 types)
- CO2 (process, biogenic tracking)
- CH4 (from incomplete reactions, pyrolysis)
- N2O (nitric acid, adipic acid processes)
- PFCs: CF4 (tetrafluoromethane), C2F6 (hexafluoroethane) - aluminum smelting
- SF6 (magnesium production cover gas, semiconductor)
- NF3 (semiconductor CVD chamber cleaning)
- HFCs (semiconductor etching)

### 6.3 Calculation Methods (4 methods)
1. **Emission Factor Method**: Activity data x process-specific emission factor (Tier 1)
2. **Mass Balance Method**: Carbon/material input - output - stock change (Tier 2/3)
3. **Stoichiometric Method**: Based on chemical reaction stoichiometry (Tier 2)
4. **Direct Measurement**: CEMS or periodic stack testing data (Tier 3)

### 6.4 Calculation Tiers (IPCC)
- **Tier 1**: Default emission factors from IPCC Guidelines
- **Tier 2**: Country-specific or technology-specific emission factors
- **Tier 3**: Facility-specific data, mass balance, or direct measurement

### 6.5 Emission Factor Sources (5 sources)
- EPA GHG Emission Factors Hub (2024)
- IPCC 2006 Guidelines / 2019 Refinement
- DEFRA/BEIS UK Conversion Factors
- EU ETS Monitoring and Reporting Regulation
- Custom/facility-specific factors

### 6.6 GWP Sources (4 sources)
- IPCC AR4 (SAR for legacy)
- IPCC AR5
- IPCC AR6 (100-year, default)
- IPCC AR6 (20-year)

### 6.7 Material Balance Features
- Raw material composition tracking (CaCO3, MgCO3, FeCO3, etc.)
- Carbonate content and calcination factor
- Carbon content of inputs (coke, coal, limestone, scrap)
- Product and by-product carbon content
- Stock change accounting
- Waste/dust loss factors
- Clinker-to-cement ratio
- Slag/scrap ratio for steel

### 6.8 Abatement Technologies
- N2O abatement for nitric/adipic acid (catalytic, thermal)
- Destruction efficiency tracking (0-99.9%)
- PFC anode effect control (CWPB, SWPB, VSS, HSS)
- SF6 capture and recycling
- Carbon capture readiness tracking
- Abatement cost-effectiveness analysis

### 6.9 Process Unit Management
- Process unit registry (kiln, furnace, smelter, reactor)
- Production capacity and utilization tracking
- Batch vs. continuous process distinction
- Multi-product facility allocation
- Process unit lifecycle (commissioning, operation, decommission)
- Production intensity metrics (tCO2e/tonne product)

### 6.10 Regulatory Frameworks (6)
1. **GHG Protocol Corporate Standard** (Chapter 5 - Industrial Processes)
2. **ISO 14064-1:2018** (Category 1)
3. **CSRD/ESRS E1** (E1-6 GHG emissions)
4. **EPA 40 CFR Part 98** (Subparts F, H, N, O, Q, S, X, Y, Z, AA, BB, CC, DD, EE, FF, GG)
5. **UK SECR** (Streamlined Energy and Carbon Reporting)
6. **EU ETS MRR** (Monitoring and Reporting Regulation - process emissions)

## 7. Database Schema (V054)

### Schema: process_emissions_service

### Tables (10)
| Table | Purpose |
|-------|---------|
| pe_process_types | Industrial process category definitions |
| pe_raw_materials | Raw material properties and compositions |
| pe_emission_factors | EFs by process type, source (CO2, CH4, N2O, PFC, SF6, NF3) |
| pe_process_units | Facility process unit registry |
| pe_material_inputs | Material input tracking per calculation |
| pe_calculations | Individual emission calculations |
| pe_calculation_details | Per-gas emission breakdown |
| pe_abatement_records | Abatement technology tracking |
| pe_compliance_records | Regulatory compliance check results |
| pe_audit_entries | Provenance and audit trail entries |

### Hypertables (3) - 7-day chunks
| Hypertable | Purpose |
|------------|---------|
| pe_calculation_events | Time-series calculation tracking |
| pe_material_events | Time-series material input tracking |
| pe_compliance_events | Time-series compliance check tracking |

### Continuous Aggregates (2)
| Aggregate | Purpose |
|-----------|---------|
| pe_hourly_calculation_stats | Hourly calculation volume and emissions |
| pe_daily_emission_totals | Daily emission totals by process type and gas |

### Row-Level Security
- All tables with tenant_id policy

### RBAC Permissions (18)
- process-emissions:read, process-emissions:write, process-emissions:execute for each resource type

## 8. Prometheus Metrics (12)

All prefixed `gl_pe_`:

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | calculations_total | Counter | process_type, method, status |
| 2 | emissions_kg_co2e_total | Counter | process_type, gas |
| 3 | process_lookups_total | Counter | source |
| 4 | factor_selections_total | Counter | tier, source |
| 5 | material_operations_total | Counter | operation_type, material_type |
| 6 | uncertainty_runs_total | Counter | method |
| 7 | compliance_checks_total | Counter | framework, status |
| 8 | batch_jobs_total | Counter | status |
| 9 | calculation_duration_seconds | Histogram | operation |
| 10 | batch_size | Histogram | method |
| 11 | active_calculations | Gauge | - |
| 12 | process_units_registered | Gauge | process_type |

## 9. REST API (20 Endpoints)

Prefix: `/api/v1/process-emissions`

| Method | Path | Description |
|--------|------|-------------|
| POST | /calculate | Calculate process emissions |
| POST | /calculate/batch | Batch calculate emissions |
| GET | /calculations | List calculations |
| GET | /calculations/{calc_id} | Get calculation details |
| POST | /processes | Register a process type |
| GET | /processes | List process types |
| GET | /processes/{process_id} | Get process details |
| POST | /materials | Register raw material |
| GET | /materials | List raw materials |
| GET | /materials/{material_id} | Get material details |
| POST | /units | Register process unit |
| GET | /units | List process units |
| POST | /factors | Register custom emission factor |
| GET | /factors | List emission factors |
| POST | /abatement | Register abatement technology |
| GET | /abatement | List abatement records |
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
