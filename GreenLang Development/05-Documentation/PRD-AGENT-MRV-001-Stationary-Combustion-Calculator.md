# PRD: AGENT-MRV-001 — GL-MRV-X-001 Stationary Combustion Calculator

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-MRV-X-001 |
| Internal Label | AGENT-MRV-001 |
| Category | Layer 3 – MRV / Accounting Agents |
| Purpose | Calculate Scope 1 GHG emissions from stationary combustion sources |
| Estimated Variants | 21,600 |
| Status | PLANNED |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| DB Migration | V048 |

## 2. Problem Statement
Scope 1 stationary combustion is the largest source of direct GHG emissions for most industrial organizations. Calculating these emissions requires matching fuel consumption data to authoritative emission factors (EPA, IPCC AR6, DEFRA), applying heating value conversions, oxidation factors, and GWP multipliers for CO2, CH4, and N2O. Existing Layer 1 code in `greenlang.agents.mrv.scope1_combustion` provides basic calculation but lacks the new platform architecture (7-engine pattern, 20 REST API endpoints, Prometheus metrics, provenance chains, K8s manifests, auth integration). A dedicated Layer 3 agent is needed for production-grade, auditable Scope 1 stationary combustion calculations.

## 3. Existing Layer 1 Capabilities
- `greenlang.agents.mrv.scope1_combustion` — v1 combustion calculator (15 fuel types, stationary + mobile, basic provenance)
- `greenlang.agents.calculation.emissions.core_calculator` — Core calculation engine with FactorResolution, fallback levels
- `greenlang.agents.calculation.emissions.scope1_calculator` — Scope 1 calculator wrapper with gas breakdown
- `greenlang.agents.calculation.emissions.gas_decomposition` — CO2/CH4/N2O decomposition with AR6 GWPs
- `greenlang.agents.calculation.emissions.unit_converter` — Deterministic unit conversions
- `greenlang.agents.calculation.emissions.uncertainty` — Monte Carlo uncertainty quantification
- `greenlang.agents.calculation.emissions.audit_trail` — Basic provenance tracking

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2/3 Needed |
|---|-----|---------|------------------|
| 1 | Fuel database management | Hardcoded dicts | Versioned, extensible emission factor database with source tracking |
| 2 | Multi-source emission factors | EPA only | EPA + IPCC AR6 + DEFRA + EU ETS + custom factors |
| 3 | Equipment profiling | None | Equipment type (boiler/furnace/heater/turbine/generator), efficiency curves, load factors |
| 4 | Tier-based EF selection | Basic fallback | 3-tier (Tier 1 default, Tier 2 country-specific, Tier 3 facility-specific) with documented methodology |
| 5 | Uncertainty quantification | Basic Monte Carlo | Full Monte Carlo + analytical propagation + data quality tier scoring |
| 6 | Biogenic carbon tracking | CO2=0 flag only | Separate biogenic CO2 reporting per GHG Protocol Scope 1 guidance |
| 7 | Equipment efficiency adjustment | None | Load-factor-adjusted emissions, part-load efficiency curves |
| 8 | Multi-facility aggregation | Simple sum | Facility roll-up with equity/operational/financial control approaches |
| 9 | Regulatory compliance mapping | None | Map calculations to GHG Protocol, ISO 14064-1, CSRD/ESRS E1, EPA 40 CFR Part 98 |
| 10 | Temporal granularity | Annual only | Monthly, quarterly, annual with temporal alignment |
| 11 | Batch calculation pipeline | Sequential only | Parallel batch processing with checkpointing and provenance chains |
| 12 | REST API + observability | None | 20 endpoints, 12 Prometheus metrics, auth integration |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | FuelDatabaseEngine | Manage 327+ emission factors across 5 sources (EPA, IPCC, DEFRA, EU ETS, custom); fuel properties (heating values HHV/LHV, density, carbon content); oxidation factors; versioning and source tracking |
| 2 | CombustionCalculatorEngine | Core calculation: Activity × HV × EF × OF × GWP; Decimal-precision arithmetic; CO2/CH4/N2O decomposition; biogenic carbon separation; GWP application (AR5/AR6, 20yr/100yr) |
| 3 | EquipmentProfilerEngine | Equipment types (boiler, furnace, heater, turbine, generator, kiln, oven, dryer); rated capacity; load factors; efficiency curves (polynomial); maintenance status; age degradation |
| 4 | EmissionFactorSelectorEngine | 3-tier selection: Tier 1 (IPCC default), Tier 2 (country-specific), Tier 3 (facility-measured); fallback chain with documentation; factor resolution tracking; geographic scope matching |
| 5 | UncertaintyQuantifierEngine | Monte Carlo simulation (configurable iterations); analytical error propagation; data quality scoring (1-5 scale); confidence intervals (90%, 95%, 99%); uncertainty contribution analysis |
| 6 | AuditTrailEngine | Complete calculation lineage; SHA-256 provenance chains; regulatory compliance mapping (GHG Protocol, ISO 14064-1, CSRD E1, EPA 40 CFR 98); export to audit-ready formats (JSON, CSV, PDF-ready) |
| 7 | StationaryCombustionPipelineEngine | End-to-end orchestration: validate inputs → select EF → convert units → calculate → quantify uncertainty → generate audit trail; batch processing with checkpointing; facility aggregation |

### 5.2 Database Schema (V048)
- `sc_fuel_types` — fuel type registry (15+ fuels, properties, classifications)
- `sc_emission_factors` — emission factor database (factor_id, gas, value, unit, source, tier, geography, effective_date)
- `sc_heating_values` — heating values (HHV/LHV per fuel per unit, source)
- `sc_oxidation_factors` — oxidation factors per fuel type (default + custom)
- `sc_equipment_profiles` — equipment registry (type, rated_capacity, efficiency_curve, load_factor)
- `sc_calculations` — calculation results (inputs, outputs, provenance_hash, tier_used, uncertainty)
- `sc_calculation_details` — per-gas breakdown (CO2, CH4, N2O, biogenic CO2, CO2e)
- `sc_facility_aggregations` — facility-level roll-ups (control approach, reporting period)
- `sc_audit_entries` — audit trail events (calculation_id, step, input, output, factor_used)
- `sc_custom_factors` — user-defined emission factors (tenant-scoped, approval workflow)
- 3 hypertables: `sc_calculation_events`, `sc_factor_updates`, `sc_audit_events` (7-day chunks)
- 2 continuous aggregates: `sc_hourly_calculation_stats`, `sc_daily_emission_totals`

### 5.3 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_sc_calculations_total | Counter | Calculations by fuel_type, tier, status |
| gl_sc_emissions_kg_co2e_total | Counter | Total emissions calculated (kg CO2e) |
| gl_sc_fuel_lookups_total | Counter | Fuel database lookups by source |
| gl_sc_factor_selections_total | Counter | EF selections by tier |
| gl_sc_equipment_profiles_total | Counter | Equipment profiles created/updated |
| gl_sc_uncertainty_runs_total | Counter | Monte Carlo simulation runs |
| gl_sc_audit_entries_total | Counter | Audit trail entries generated |
| gl_sc_batch_jobs_total | Counter | Batch calculation jobs by status |
| gl_sc_calculation_duration_seconds | Histogram | Calculation duration by operation |
| gl_sc_batch_size | Histogram | Records per batch |
| gl_sc_active_calculations | Gauge | Currently running calculations |
| gl_sc_emission_factors_loaded | Gauge | Number of EFs in database |

### 5.4 REST API Endpoints (20)
| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | /api/v1/stationary-combustion/calculate | Calculate emissions for a single fuel consumption record |
| 2 | POST | /api/v1/stationary-combustion/calculate/batch | Batch calculate for multiple records |
| 3 | GET | /api/v1/stationary-combustion/calculations | List calculation results with pagination |
| 4 | GET | /api/v1/stationary-combustion/calculations/{calc_id} | Get calculation details + audit trail |
| 5 | POST | /api/v1/stationary-combustion/fuels | Register a custom fuel type |
| 6 | GET | /api/v1/stationary-combustion/fuels | List all fuel types |
| 7 | GET | /api/v1/stationary-combustion/fuels/{fuel_id} | Get fuel properties |
| 8 | POST | /api/v1/stationary-combustion/factors | Register a custom emission factor |
| 9 | GET | /api/v1/stationary-combustion/factors | List emission factors with filtering |
| 10 | GET | /api/v1/stationary-combustion/factors/{factor_id} | Get emission factor details |
| 11 | POST | /api/v1/stationary-combustion/equipment | Register equipment profile |
| 12 | GET | /api/v1/stationary-combustion/equipment | List equipment profiles |
| 13 | GET | /api/v1/stationary-combustion/equipment/{equip_id} | Get equipment profile details |
| 14 | POST | /api/v1/stationary-combustion/aggregate | Aggregate calculations for facility |
| 15 | GET | /api/v1/stationary-combustion/aggregations | List facility aggregations |
| 16 | POST | /api/v1/stationary-combustion/uncertainty | Run uncertainty analysis |
| 17 | GET | /api/v1/stationary-combustion/audit/{calc_id} | Get audit trail for calculation |
| 18 | POST | /api/v1/stationary-combustion/validate | Validate input data without calculating |
| 19 | GET | /api/v1/stationary-combustion/health | Health check |
| 20 | GET | /api/v1/stationary-combustion/stats | Service statistics |

### 5.5 Emission Factor Sources
| Source | Authority | Coverage | Factors |
|--------|-----------|----------|---------|
| EPA | US EPA GHG EF Hub | US fuels, stationary combustion | 50+ factors |
| IPCC | IPCC AR6 2021 | Global defaults (Tier 1) | 100+ factors |
| DEFRA | UK DEFRA | UK-specific factors | 40+ factors |
| EU ETS | European Commission | EU Monitoring & Reporting | 30+ factors |
| Custom | Tenant-provided | Facility-specific (Tier 3) | Unlimited |

### 5.6 Fuel Types Supported (15+)
| Category | Fuels |
|----------|-------|
| Gaseous | Natural gas, LPG, propane, biogas, landfill gas |
| Liquid | Diesel, gasoline, kerosene, jet fuel, fuel oil #2, fuel oil #6 |
| Solid | Coal (bituminous, anthracite, sub-bituminous, lignite), wood, biomass, peat |

### 5.7 Equipment Types
| Equipment | Typical Efficiency | Load Factor Range |
|-----------|-------------------|-------------------|
| Boiler (fire-tube) | 80-85% | 0.3-1.0 |
| Boiler (water-tube) | 82-90% | 0.4-1.0 |
| Furnace (industrial) | 70-85% | 0.5-1.0 |
| Heater (process) | 75-90% | 0.3-0.9 |
| Turbine (gas) | 30-42% | 0.5-1.0 |
| Generator (diesel) | 35-45% | 0.3-0.8 |
| Kiln (rotary) | 55-75% | 0.6-1.0 |
| Oven (industrial) | 65-80% | 0.4-0.9 |
| Dryer (rotary) | 60-75% | 0.5-1.0 |

## 6. Calculation Methodology

### 6.1 Core Formula
```
Emissions (kg CO2e) = Activity × HV × EF × OF × GWP

Where:
  Activity = Fuel consumption (in native units)
  HV = Heating value (GJ per native unit) — converts fuel to energy
  EF = Emission factor (kg gas per GJ) — gas-specific (CO2, CH4, N2O)
  OF = Oxidation factor (fraction, e.g., 0.99)
  GWP = Global Warming Potential (AR6 100-year: CO2=1, CH4=29.8, N2O=273)
```

### 6.2 Gas Decomposition
```
Total CO2e = (Activity × HV × EF_CO2 × OF × 1)
           + (Activity × HV × EF_CH4 × OF × 29.8)
           + (Activity × HV × EF_N2O × OF × 273)
```

### 6.3 Biogenic Carbon
- Wood, biomass, biogas, landfill gas: fossil CO2 = 0
- Biogenic CO2 reported separately (GHG Protocol Guidance)
- CH4 and N2O from biogenic fuels are still counted in Scope 1

### 6.4 Tier Selection Logic
1. **Tier 3 (Facility-specific)**: Direct measurement or facility-calibrated factors → highest accuracy
2. **Tier 2 (Country-specific)**: National statistics, technology-adjusted defaults → medium accuracy
3. **Tier 1 (IPCC default)**: Global default factors → baseline accuracy
4. Fallback: If Tier 3 unavailable → Tier 2 → Tier 1

## 7. Non-Functional Requirements
| Requirement | Target |
|-------------|--------|
| Calculation latency | <10ms per single calculation |
| Batch throughput | 10,000 records in <15 seconds |
| Decimal precision | 8 decimal places (Decimal arithmetic) |
| Provenance chain | SHA-256 hash for every calculation |
| Test coverage | 85%+ |
| Test count | 500+ |
| Zero hallucination | No LLM in any calculation path |
| Determinism | Same input → bit-identical output |

## 8. Regulatory Compliance Mapping
| Standard | Requirement | How Met |
|----------|-------------|---------|
| GHG Protocol Corporate Standard | Scope 1 stationary combustion | Direct implementation of methodology |
| ISO 14064-1:2018 | Category 1 direct GHG emissions | Calculation with full uncertainty |
| CSRD / ESRS E1 | GHG emissions (E1-6) | Activity-based, auditable |
| EPA 40 CFR Part 98 | Subpart C – General stationary fuel combustion | Tier-based approach |
| UK Streamlined Energy & Carbon Reporting | Scope 1 emissions reporting | DEFRA factors |

## 9. Dependencies
| Component | Purpose |
|-----------|---------|
| AGENT-FOUND-001 (Orchestrator) | DAG execution for pipeline |
| AGENT-FOUND-003 (Unit Normalizer) | Unit conversions |
| AGENT-FOUND-005 (Citations) | Emission factor source citations |
| AGENT-FOUND-008 (Reproducibility) | Determinism verification |
| AGENT-FOUND-010 (Observability) | Metrics and tracing |
| AGENT-DATA-010 (Data Quality Profiler) | Input data quality scoring |
