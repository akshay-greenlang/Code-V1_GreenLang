# PRD: AGENT-MRV-002 — GL-MRV-SCOPE1-002 Refrigerants & F-Gas Agent

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-MRV-SCOPE1-002 |
| Internal Label | AGENT-MRV-002 |
| Category | Layer 3 – MRV / Accounting Agents (Scope 1) |
| Purpose | Calculate Scope 1 GHG emissions from refrigerant leakage and fluorinated gas sources |
| Estimated Variants | 18,400 |
| Status | PLANNED |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| DB Migration | V052 |

## 2. Problem Statement
Fluorinated gases (HFCs, PFCs, SF6, NF3) are potent greenhouse gases with GWP values ranging from 1 (HFOs) to 25,200 (SF6). Scope 1 fugitive emissions from refrigerant leakage in HVAC&R, switchgear, fire suppression, and semiconductor manufacturing represent a significant and growing share of organizational carbon footprints. The EU F-Gas Regulation 2024/573 mandates phase-down of HFCs by 85% by 2036 (from 2015 baseline), the Kigali Amendment to the Montreal Protocol sets global HFC phase-down schedules, and EPA 40 CFR Part 98 Subparts DD/OO/L require facility-level reporting. Existing Layer 1 code in `greenlang.agents.mrv.refrigerants_fgas` provides basic calculation with 21 refrigerant types and 3 methods but lacks the production platform architecture (7-engine pattern, 20 REST API endpoints, Prometheus metrics, provenance chains, regulatory compliance tracking, blend decomposition, lifecycle tracking, K8s manifests, auth integration). A dedicated Layer 3 agent is needed for production-grade, auditable Scope 1 F-gas emissions calculations.

## 3. Existing Layer 1 Capabilities
- `greenlang.agents.mrv.refrigerants_fgas` — v1 F-gas calculator (21 refrigerant types, 10 equipment types, 3 methods)
- Equipment-based method with default leakage rates
- Mass balance method (purchases - disposals)
- Screening method (simplified estimation)
- GWP values from IPCC AR6 for 21 gases
- Basic SHA-256 provenance hashing
- Pydantic models for input/output

## 4. Identified Gaps (14)
| # | Gap | Layer 1 | Layer 2/3 Needed |
|---|-----|---------|------------------|
| 1 | Refrigerant database management | Hardcoded dicts (21 types) | Versioned database with 50+ pure gases, 30+ blends, multi-source GWP (AR4/AR5/AR6) |
| 2 | HFC blend decomposition | None | Decompose blend GWP into component gases (R-410A = 50% R-32 + 50% R-125) |
| 3 | Equipment lifecycle tracking | None | Installation, servicing, decommissioning event tracking with charge history |
| 4 | Leak rate estimation models | Fixed defaults | Equipment-specific, age-adjusted, climate-zone-adjusted leak rate models |
| 5 | Multi-method calculation engine | Basic 3 methods | Equipment-based, mass balance, screening, direct measurement, top-down |
| 6 | Uncertainty quantification | None | Monte Carlo + analytical propagation for each method |
| 7 | Regulatory compliance tracking | None | EU F-Gas 2024/573 phase-down, Kigali Amendment, EPA 40 CFR 98 DD/OO/L |
| 8 | F-Gas quota management | None | Track HFC quotas, phase-down compliance, CO2e budget consumption |
| 9 | Leak detection & repair (LDAR) | None | LDAR program integration, leak event logging, repair tracking |
| 10 | Refrigerant recovery tracking | None | Recovery, recycling, reclamation, destruction tracking |
| 11 | Multi-facility aggregation | Simple sum | Roll-up with operational/financial/equity share control approaches |
| 12 | Temporal granularity | Annual only | Monthly, quarterly, annual with seasonal leak rate adjustment |
| 13 | Batch calculation pipeline | Sequential only | Parallel batch processing with checkpointing and provenance chains |
| 14 | REST API + observability | None | 20 endpoints, 12 Prometheus metrics, auth integration |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | RefrigerantDatabaseEngine | Manage 50+ pure refrigerant gases and 30+ blends across 4 GWP sources (AR4, AR5, AR6, AR6-20yr); physical properties (boiling point, molecular weight, ODP, atmospheric lifetime); blend composition with weight fractions; regulatory classification (Group I/II HFCs, PFCs, SF6, NF3, HFOs, HCFCs); versioning and source tracking |
| 2 | EmissionCalculatorEngine | Core calculation methods: equipment-based (Charge × LeakRate × GWP), mass balance (BI + Purchases - Sales - EI - CapChange), screening (Total × DefaultRate × GWP), direct measurement, top-down estimation; Decimal-precision arithmetic; blend decomposition into component gas emissions; biogenic vs synthetic classification |
| 3 | EquipmentRegistryEngine | Equipment types (commercial/industrial refrigeration, residential/commercial AC, chillers, heat pumps, transport refrigeration, switchgear, semiconductor, fire suppression, foam blowing, aerosols, solvents); charge sizes; installation dates; service history; decommissioning tracking |
| 4 | LeakRateEstimatorEngine | Default leak rates by equipment type and application; age-adjusted rates (new, 5yr, 10yr, 15yr+); installation/operating/end-of-life lifecycle leak rates per IPCC 2006 Vol 3 Ch 7; climate zone adjustments; LDAR program effectiveness factors; custom rate overrides |
| 5 | UncertaintyQuantifierEngine | Monte Carlo simulation (configurable iterations); analytical error propagation for each method; data quality scoring (1-5 scale); confidence intervals (90%, 95%, 99%); method-specific uncertainty ranges per IPCC guidelines (±10% mass balance, ±30% equipment-based, ±50% screening) |
| 6 | ComplianceTrackerEngine | EU F-Gas Regulation 2024/573 HFC phase-down schedule (2015 baseline, 2030=31%, 2036=15%); Kigali Amendment A5/non-A5 schedules; EPA 40 CFR Part 98 Subpart DD/OO/L reporting thresholds; GHG Protocol Corporate Standard Scope 1; ISO 14064-1:2018; CSRD/ESRS E1; UK F-Gas; quota tracking with CO2e budget |
| 7 | RefrigerantPipelineEngine | End-to-end orchestration: validate inputs → lookup refrigerant → estimate leak rate → calculate emissions → decompose blends → quantify uncertainty → check compliance → generate audit trail; batch processing with checkpointing; facility aggregation |

### 5.2 Database Schema (V052)
- `rf_refrigerant_types` — refrigerant registry (50+ gases, properties, classifications, ODP)
- `rf_refrigerant_blends` — blend compositions (component gases, weight fractions)
- `rf_gwp_values` — GWP database (gas_id, gwp_source, timeframe, value, effective_date)
- `rf_equipment_registry` — equipment profiles (type, refrigerant, charge_kg, install_date, status)
- `rf_leak_rates` — leak rate database (equipment_type, lifecycle_stage, rate, source)
- `rf_service_events` — service history (equipment_id, event_type, refrigerant_added_kg, recovered_kg)
- `rf_calculations` — calculation results (method, inputs, outputs, provenance_hash, uncertainty)
- `rf_calculation_details` — per-gas breakdown (gas, loss_kg, gwp, emissions_co2e)
- `rf_compliance_records` — regulatory compliance tracking (framework, quota, usage, status)
- `rf_audit_entries` — audit trail events (calculation_id, step, input, output, factor_used)
- 3 hypertables: `rf_calculation_events`, `rf_service_events_ts`, `rf_compliance_events` (7-day chunks)
- 2 continuous aggregates: `rf_hourly_calculation_stats`, `rf_daily_emission_totals`

### 5.3 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_rf_calculations_total | Counter | Calculations by method, refrigerant_type, status |
| gl_rf_emissions_kg_co2e_total | Counter | Total emissions calculated (kg CO2e) |
| gl_rf_refrigerant_lookups_total | Counter | Refrigerant database lookups by source |
| gl_rf_leak_rate_selections_total | Counter | Leak rate selections by equipment_type, lifecycle_stage |
| gl_rf_equipment_events_total | Counter | Equipment events (install, service, decommission) |
| gl_rf_uncertainty_runs_total | Counter | Monte Carlo simulation runs by method |
| gl_rf_compliance_checks_total | Counter | Compliance checks by framework, status |
| gl_rf_batch_jobs_total | Counter | Batch calculation jobs by status |
| gl_rf_calculation_duration_seconds | Histogram | Calculation duration by operation |
| gl_rf_batch_size | Histogram | Records per batch |
| gl_rf_active_calculations | Gauge | Currently running calculations |
| gl_rf_refrigerants_loaded | Gauge | Number of refrigerants in database |

### 5.4 REST API Endpoints (20)
| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | /api/v1/refrigerants-fgas/calculate | Calculate F-gas emissions for a single record |
| 2 | POST | /api/v1/refrigerants-fgas/calculate/batch | Batch calculate for multiple records |
| 3 | GET | /api/v1/refrigerants-fgas/calculations | List calculation results with pagination |
| 4 | GET | /api/v1/refrigerants-fgas/calculations/{calc_id} | Get calculation details + audit trail |
| 5 | POST | /api/v1/refrigerants-fgas/refrigerants | Register a custom refrigerant |
| 6 | GET | /api/v1/refrigerants-fgas/refrigerants | List all refrigerants |
| 7 | GET | /api/v1/refrigerants-fgas/refrigerants/{ref_id} | Get refrigerant properties |
| 8 | POST | /api/v1/refrigerants-fgas/equipment | Register equipment profile |
| 9 | GET | /api/v1/refrigerants-fgas/equipment | List equipment profiles |
| 10 | GET | /api/v1/refrigerants-fgas/equipment/{equip_id} | Get equipment details |
| 11 | POST | /api/v1/refrigerants-fgas/service-events | Log a service event |
| 12 | GET | /api/v1/refrigerants-fgas/service-events | List service events |
| 13 | POST | /api/v1/refrigerants-fgas/leak-rates | Register custom leak rate |
| 14 | GET | /api/v1/refrigerants-fgas/leak-rates | List leak rates |
| 15 | POST | /api/v1/refrigerants-fgas/compliance/check | Check regulatory compliance |
| 16 | GET | /api/v1/refrigerants-fgas/compliance | List compliance records |
| 17 | POST | /api/v1/refrigerants-fgas/uncertainty | Run uncertainty analysis |
| 18 | GET | /api/v1/refrigerants-fgas/audit/{calc_id} | Get audit trail for calculation |
| 19 | GET | /api/v1/refrigerants-fgas/health | Health check |
| 20 | GET | /api/v1/refrigerants-fgas/stats | Service statistics |

### 5.5 GWP Sources
| Source | Authority | Timeframe | Coverage |
|--------|-----------|-----------|----------|
| AR4 | IPCC 4th Assessment Report (2007) | 100-year | Legacy compatibility |
| AR5 | IPCC 5th Assessment Report (2014) | 100-year | EU ETS, many regulations |
| AR6 | IPCC 6th Assessment Report (2021) | 100-year | Latest science, GHG Protocol |
| AR6-20yr | IPCC 6th Assessment Report (2021) | 20-year | Short-lived climate forcers |

### 5.6 Refrigerant Types Supported (50+)
| Category | Gases |
|----------|-------|
| HFCs (high GWP) | R-32, R-125, R-134a, R-143a, R-152a, R-227ea, R-236fa, R-245fa, R-365mfc, R-23, R-41 |
| HFC Blends | R-404A, R-407A/C/F, R-410A, R-413A, R-417A, R-422D, R-427A, R-438A, R-448A, R-449A, R-452A, R-454B, R-507A, R-508B |
| HFOs (low GWP) | R-1234yf, R-1234ze(E), R-1233zd(E), R-1336mzz(Z) |
| PFCs | CF4, C2F6, C3F8, c-C4F8, C4F10, C5F12, C6F14 |
| Other F-gases | SF6, NF3, SO2F2 |
| HCFCs (legacy) | R-22, R-123, R-141b, R-142b |
| CFCs (legacy) | R-11, R-12, R-113, R-114, R-115, R-502 |
| Natural (reference) | R-717 (NH3), R-744 (CO2), R-290 (propane), R-600a (isobutane) |

### 5.7 Equipment Types
| Equipment | Typical Charge (kg) | Default Annual Leak Rate | Lifetime (years) |
|-----------|---------------------|--------------------------|-------------------|
| Commercial refrigeration (centralized) | 50-500 | 15-25% | 15-20 |
| Commercial refrigeration (standalone) | 0.2-5 | 2-5% | 10-15 |
| Industrial refrigeration | 100-10,000 | 8-15% | 20-30 |
| Residential AC (split) | 0.5-3 | 2-5% | 15-20 |
| Commercial AC (packaged) | 5-50 | 4-8% | 15-20 |
| Chillers (centrifugal) | 50-500 | 2-5% | 20-30 |
| Chillers (screw/scroll) | 10-100 | 3-6% | 15-25 |
| Heat pumps | 1-20 | 3-5% | 15-20 |
| Transport refrigeration | 2-10 | 15-30% | 10-15 |
| Switchgear (SF6) | 1-50 | 0.5-2% | 30-40 |
| Semiconductor manufacturing | 0.1-5 | 5-10% | 10-15 |
| Fire suppression | 10-500 | 1-3% | 20-30 |
| Foam blowing | 0.01-1 | 1-5% (banked) | 30-50 |
| Aerosols/MDI | 0.01-0.1 | 100% (emissive) | 1 |
| Solvents | 0.1-10 | 50-90% | 1-5 |

## 6. Calculation Methodology

### 6.1 Equipment-Based Method (Tier 1)
```
Emissions (kg CO2e) = Charge × Count × LeakRate × GWP

Where:
  Charge = Refrigerant charge per unit (kg)
  Count = Number of equipment units
  LeakRate = Annual leakage rate (fraction, e.g. 0.06 for 6%)
  GWP = Global Warming Potential (100-year)
```

### 6.2 Mass Balance Method (Tier 2)
```
Emissions (kg) = BI + Purchases - Sales + Acquisitions
                 - Divestitures - EI - CapacityChange

Emissions (kg CO2e) = Emissions (kg) × GWP

Where:
  BI = Beginning inventory (kg)
  Purchases = Refrigerant purchased/received (kg)
  Sales = Refrigerant sold/transferred (kg)
  Acquisitions = Refrigerant from acquired equipment (kg)
  Divestitures = Refrigerant in divested equipment (kg)
  EI = Ending inventory (kg)
  CapacityChange = Net change in total system capacity (kg)
```

### 6.3 Screening Method (Tier 0)
```
Emissions (kg CO2e) = TotalInstalledCharge × DefaultLeakRate × GWP
```

### 6.4 Blend Decomposition
```
For blend R-410A (50% R-32, 50% R-125):
  GWP_blend = Σ (weight_fraction_i × GWP_i)
  GWP_R410A = 0.50 × 771 + 0.50 × 3740 = 2255.5

Component emissions:
  CO2e_R32 = Loss_kg × 0.50 × GWP_R32
  CO2e_R125 = Loss_kg × 0.50 × GWP_R125
```

### 6.5 Lifecycle Leak Rate Model
```
Total lifecycle emissions = Install + Operating + EOL

Where:
  Install = Charge × InstallLeakRate (typically 0.5-2%)
  Operating = Charge × AnnualLeakRate × Years
  EOL = Charge × (1 - RecoveryEfficiency) × (1 - already leaked)
```

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
| GHG Protocol Corporate Standard | Scope 1 fugitive emissions (refrigerants) | Equipment-based and mass balance methods |
| ISO 14064-1:2018 | Category 1 direct GHG emissions (fugitive) | Full lifecycle tracking with uncertainty |
| CSRD / ESRS E1 | GHG emissions (E1-6), F-gas specific | Auditable, blend-decomposed reporting |
| EPA 40 CFR Part 98 Subpart DD | Electrical equipment (SF6) | SF6 mass balance tracking |
| EPA 40 CFR Part 98 Subpart OO | Suppliers of industrial gases | Supply chain tracking |
| EPA 40 CFR Part 98 Subpart L | Fluorinated gas production | Production emission tracking |
| EU F-Gas Regulation 2024/573 | HFC phase-down, quotas, bans | Quota tracking, phase-down compliance |
| Kigali Amendment | HFC phase-down (A5/non-A5 parties) | Schedule tracking, CO2e budgets |
| UK F-Gas Regulations | GB-specific F-Gas requirements | DEFRA emission factors |

## 9. Dependencies
| Component | Purpose |
|-----------|---------|
| AGENT-FOUND-001 (Orchestrator) | DAG execution for pipeline |
| AGENT-FOUND-003 (Unit Normalizer) | Unit conversions (kg, lb, oz) |
| AGENT-FOUND-005 (Citations) | GWP source citations |
| AGENT-FOUND-008 (Reproducibility) | Determinism verification |
| AGENT-FOUND-010 (Observability) | Metrics and tracing |
| AGENT-DATA-010 (Data Quality Profiler) | Input data quality scoring |
| AGENT-MRV-001 (Stationary Combustion) | Shared Scope 1 reporting infrastructure |
