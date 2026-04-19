# PRD: AGENT-MRV-010 — Scope 2 Market-Based Emissions Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE2-010 |
| **Internal Label** | AGENT-MRV-010 |
| **Category** | Layer 3 — MRV / Accounting Agents (Scope 2) |
| **Package** | `greenlang/scope2_market/` |
| **DB Migration** | V061 |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

Calculates **Scope 2 market-based GHG emissions** from purchased electricity,
steam, heating, and cooling using **contractual instruments** per the
GHG Protocol Scope 2 Guidance (2015). The market-based method reflects
emissions from electricity that companies have purposefully chosen through
contractual arrangements such as RECs, PPAs, GOs, supplier-specific factors,
and residual mix factors for uncovered consumption.

### Standards & References

- GHG Protocol Scope 2 Guidance (2015) — Primary methodology (Chapter 6-7)
- GHG Protocol Corporate Standard (Revised 2015) — Scope 2 boundary
- GHG Protocol Scope 2 Quality Criteria — Contractual instrument requirements
- RE100 Technical Criteria (2023) — Renewable electricity reporting
- Green-e Standard (2024) — US voluntary renewable energy certification
- AIB European Residual Mix (2024) — EU residual mix factors
- Ofgem Fuel Mix Disclosure (2024) — UK supplier-specific factors
- IEA CO2 Emission Factors (2024) — Fallback country grid factors
- ISO 14064-1:2018 — Scope 2 reporting requirements
- CSRD/ESRS E1 — EU double materiality Scope 2 disclosure
- CDP Climate Change Questionnaire (2024) — C8.2d market-based reporting
- SBTi Corporate Manual — RE100 and market-based accounting

---

## 2. Scope 2 Market-Based Methodology

### 2.1 Core Concept

The market-based method allows companies to claim specific emission factors
based on contractual arrangements rather than grid-average factors. This
creates a financial signal for renewable energy development and allows
companies to reduce their reported Scope 2 emissions through procurement
decisions.

### 2.2 Core Formulas

**For instrument-covered consumption:**
```
Emissions_covered (tCO2e) = Σ (Instrument_MWh × Instrument_EF)
```

**For uncovered consumption (residual):**
```
Emissions_uncovered (tCO2e) = Uncovered_MWh × Residual_Mix_EF
```

**Total market-based emissions:**
```
Total_Emissions = Emissions_covered + Emissions_uncovered
```

**Coverage percentage:**
```
Coverage_% = Instrument_MWh / Total_MWh × 100
```

### 2.3 GHG Protocol Instrument Hierarchy

Per GHG Protocol Scope 2 Guidance, Table 6.3, contractual instruments
are applied in this priority order:

| Priority | Instrument Type | Description | Typical EF |
|----------|----------------|-------------|------------|
| 1 | Energy attribute certificates (bundled) | RECs/GOs bundled with energy delivery | 0.000 (renewable) |
| 2 | Direct contracts (PPAs) | Physical or virtual PPA with specific generation | Supplier-specific |
| 3 | Supplier-specific EF | Utility supplier emission factor (fuel mix disclosure) | Varies by supplier |
| 4 | Unbundled certificates | RECs, GOs, I-RECs purchased separately | 0.000 (renewable) |
| 5 | Green tariffs | Green-e or equivalent green energy tariff | 0.000 (renewable) |
| 6 | Residual mix | Grid average minus renewable claims | Higher than grid avg |

### 2.4 Contractual Instrument Types

| Instrument | Region | Standard | Validity | Description |
|-----------|--------|----------|----------|-------------|
| **REC** | US/Canada | Green-e | 1 year vintage | Renewable Energy Certificate |
| **GO** | EU | AIB/EECS | 1 year vintage | Guarantee of Origin (EU Directive 2018/2001) |
| **REGO** | UK | Ofgem | 1 year vintage | Renewable Energy Guarantee of Origin |
| **I-REC** | International | I-REC Standard | 1 year vintage | International REC (non-EU/US markets) |
| **T-REC** | Taiwan | Bureau of Energy | 1 year vintage | Taiwan Renewable Energy Certificate |
| **J-Credit** | Japan | METI | 1 year vintage | Japan renewable certificate |
| **LGC** | Australia | CER | 1 year vintage | Large-scale Generation Certificate |
| **PPA** | Global | Contract-specific | Contract term | Power Purchase Agreement |
| **Green Tariff** | US/EU | Utility-specific | Contract term | Green energy tariff program |
| **Supplier-specific** | Global | Fuel mix disclosure | Annual | Utility supplier emission factor |

### 2.5 GHG Protocol Quality Criteria

Per GHG Protocol Scope 2 Guidance, Section 7.3, instruments must meet:

1. **Convey unique claim**: The instrument uniquely conveys the attribute claim
2. **Associated with delivered energy**: Generated from the same market/grid
3. **Issued/claimed close to energy consumption**: Vintage within reporting period
4. **From generators on same market/grid**: Same interconnected grid
5. **Not double-counted**: Retired/cancelled after use, not resold
6. **From a recognized tracking system**: Verified registry (Green-e, AIB, Ofgem, I-REC)
7. **Represent energy generated**: 1 MWh = 1 certificate

### 2.6 Residual Mix Factors

Residual mix factors represent the emission intensity of the grid after
removing all contractual claims (renewable certificates). They are typically
HIGHER than location-based (grid average) factors.

| Region | Residual Mix (kgCO2e/kWh) | Location-Based (kgCO2e/kWh) | Ratio |
|--------|--------------------------|------------------------------|-------|
| EU Average | 0.380 | 0.296 | 1.28x |
| Germany | 0.520 | 0.350 | 1.49x |
| France | 0.085 | 0.052 | 1.63x |
| UK | 0.285 | 0.207 | 1.38x |
| US Average | 0.425 | 0.386 | 1.10x |
| US-CAMX | 0.285 | 0.225 | 1.27x |
| US-ERCT | 0.420 | 0.380 | 1.11x |
| Australia | 0.750 | 0.656 | 1.14x |
| Japan | 0.520 | 0.465 | 1.12x |
| Singapore | 0.425 | 0.408 | 1.04x |
| Global | 0.500 | 0.436 | 1.15x |

### 2.7 Dual Reporting Requirement

GHG Protocol Scope 2 Guidance requires companies to report BOTH:
- **Location-based** result (from MRV-009)
- **Market-based** result (from MRV-010)

This agent provides dual reporting comparison capabilities integrated
with MRV-009 (Scope 2 Location-Based Agent).

### 2.8 Energy Source Emission Factors

| Energy Source | EF (kgCO2e/kWh) | Notes |
|-------------|-----------------|-------|
| Solar PV | 0.000 | Zero operational emissions |
| Wind | 0.000 | Zero operational emissions |
| Hydropower | 0.000 | Zero operational emissions |
| Nuclear | 0.000 | Zero operational emissions (Scope 1) |
| Geothermal | 0.000 | Zero operational (some count CO2) |
| Biomass | 0.000 | Biogenic CO2 (reported separately) |
| Natural Gas CCGT | 0.340 | Combined cycle gas turbine |
| Natural Gas OCGT | 0.490 | Open cycle gas turbine |
| Coal | 0.910 | Pulverized coal |
| Oil | 0.650 | Heavy fuel oil |
| Mixed/Grid Average | 0.436 | IEA world average |

---

## 3. Architecture — 7 Engines

### Engine 1: ContractualInstrumentDatabaseEngine (`contractual_instrument_database.py`)
- Master database of 10+ instrument types with quality criteria
- Residual mix factors: 30+ regions (US subregions, EU countries, APAC, Americas)
- Supplier-specific emission factor registry with annual fuel mix disclosure
- Energy source emission factors (11 sources with kgCO2e/kWh)
- Instrument validity rules: vintage period, geographic matching, tracking system
- Quality scoring per GHG Protocol 7 quality criteria
- CRUD operations for custom instruments and supplier factors
- Factor lookup by: instrument type, region, supplier, energy source
- Historical residual mix series (2019-2025)
- 25+ public methods

### Engine 2: InstrumentAllocationEngine (`instrument_allocation.py`)
- GHG Protocol instrument hierarchy enforcement (6 priority levels)
- Instrument-to-consumption matching with time/geography alignment
- Vintage year validation (must match or precede reporting period)
- Geographic matching (same market/interconnected grid)
- Double-counting prevention (certificate retirement tracking)
- Coverage tracking: covered vs uncovered MWh per facility
- Instrument stacking rules (multiple instruments per purchase)
- Partial allocation for instruments covering less than total consumption
- Over-allocation detection and warning
- Certificate retirement status validation
- 20+ public methods

### Engine 3: MarketEmissionsCalculatorEngine (`market_emissions_calculator.py`)
- Core emission calculations for instrument-covered portions
- Per-instrument emission factor application
- Renewable instrument zero-emission calculation (REC, GO, PPA with renewables)
- Supplier-specific emission factor calculation
- Residual mix emission factor for uncovered consumption
- Per-gas breakdown: CO2, CH4, N2O with individual GWP conversion
- GWP source selection (AR4/AR5/AR6/AR6_20YR)
- Biogenic CO2 tracking for biomass instruments
- Unit conversions (kWh, MWh, GJ, MMBtu, therms)
- Multi-facility aggregation and roll-up
- 20+ public methods

### Engine 4: DualReportingEngine (`dual_reporting.py`)
- Integrates MRV-009 (location-based) with MRV-010 (market-based) results
- Side-by-side comparison: location-based vs market-based emissions
- Identifies emission reduction from contractual instruments
- Coverage gap analysis: what percentage is covered by instruments
- Renewable procurement impact quantification
- SBTi RE100 progress tracking
- GHG Protocol dual reporting format generation
- Year-over-year comparison of both methods
- Additionality assessment for renewable instruments
- 15+ public methods

### Engine 5: UncertaintyQuantifierEngine (`uncertainty_quantifier.py`)
- Monte Carlo simulation for market-based uncertainty
- Instrument quality uncertainty (verified vs unverified)
- Residual mix factor uncertainty ranges
- Supplier-specific EF uncertainty
- Activity data uncertainty (meter, invoice, estimate)
- DQI scoring for instrument data quality
- Combined uncertainty propagation
- Parameter sensitivity analysis
- 20+ public methods

### Engine 6: ComplianceCheckerEngine (`compliance_checker.py`)
- Multi-framework regulatory compliance checking
- 7 frameworks: GHG Protocol Scope 2, ISO 14064, CSRD/ESRS, CDP, RE100, SBTi, Green-e
- GHG Protocol Quality Criteria validation (7 criteria per instrument)
- Instrument validity checks (vintage, geography, tracking system, retirement)
- Dual reporting completeness (both methods reported)
- Double-counting detection across facilities
- Certificate retirement verification
- Supplier disclosure quality assessment
- 25+ public methods

### Engine 7: Scope2MarketPipelineEngine (`scope2_market_pipeline.py`)
- 8-stage orchestrated calculation pipeline
- Stage 1: Validate input (facility, instruments, consumption data)
- Stage 2: Resolve instrument data (quality criteria, validity, EFs)
- Stage 3: Allocate instruments (hierarchy, matching, coverage)
- Stage 4: Calculate covered emissions (per-instrument EF × MWh)
- Stage 5: Calculate uncovered emissions (residual mix × uncovered MWh)
- Stage 6: Apply GWP conversion (AR4/AR5/AR6)
- Stage 7: Run compliance checks (optional)
- Stage 8: Assemble results with provenance and dual reporting link

---

## 4. Data Models (20 Enumerations, 20 Data Models)

### Enumerations
1. `InstrumentType` — 10 members: ppa, rec, go, rego, i_rec, t_rec, j_credit, lgc, green_tariff, supplier_specific
2. `InstrumentStatus` — 5 members: active, retired, expired, cancelled, pending
3. `EnergySource` — 11 members: solar, wind, hydro, nuclear, biomass, geothermal, natural_gas_ccgt, natural_gas_ocgt, coal, oil, mixed
4. `EnergyType` — 4 members: electricity, steam, heating, cooling
5. `EnergyUnit` — 5 members: kwh, mwh, gj, mmbtu, therms
6. `CalculationMethod` — 4 members: instrument_based, supplier_specific, residual_mix, hybrid
7. `EmissionGas` — 3 members: co2, ch4, n2o
8. `GWPSource` — 4 members: AR4, AR5, AR6, AR6_20YR
9. `QualityCriterion` — 7 members: unique_claim, associated_delivery, temporal_match, geographic_match, no_double_count, recognized_registry, represents_generation
10. `TrackingSystem` — 8 members: green_e, aib_eecs, ofgem, i_rec_standard, m_rets, nar, wregis, custom
11. `ResidualMixSource` — 5 members: aib, green_e, national, estimated, custom
12. `FacilityType` — 8 members: office, warehouse, manufacturing, retail, data_center, hospital, school, other
13. `ComplianceStatus` — 4 members: compliant, non_compliant, partial, not_assessed
14. `CoverageStatus` — 4 members: fully_covered, partially_covered, uncovered, over_covered
15. `ReportingPeriod` — 4 members: annual, quarterly, monthly, custom
16. `ContractType` — 4 members: physical_ppa, virtual_ppa, sleeved_ppa, direct_purchase
17. `DataQualityTier` — 3 members: tier_1, tier_2, tier_3
18. `DualReportingStatus` — 3 members: complete, location_only, market_only
19. `AllocationMethod` — 3 members: priority_based, proportional, custom
20. `ConsumptionDataSource` — 4 members: meter, invoice, estimate, benchmark

### Constants (all Decimal)
- `GWP_VALUES` — AR4/AR5/AR6/AR6_20YR for CO2, CH4, N2O
- `RESIDUAL_MIX_FACTORS` — 30+ regions (US subregions, EU countries, APAC, Americas)
- `ENERGY_SOURCE_EF` — 11 energy source emission factors (kgCO2e/kWh)
- `SUPPLIER_DEFAULT_EF` — Default supplier EFs by country
- `INSTRUMENT_QUALITY_WEIGHTS` — Weight per quality criterion
- `VINTAGE_VALIDITY_YEARS` — Maximum vintage age by instrument type
- `UNIT_CONVERSIONS` — MWh↔GJ↔MMBtu↔therms

### Data Models
1. `ContractualInstrument` — Instrument with type, quantity, source, EF, vintage, tracking system
2. `InstrumentQualityAssessment` — Quality scoring per 7 GHG Protocol criteria
3. `SupplierEmissionFactor` — Supplier-specific EF with fuel mix disclosure
4. `ResidualMixFactor` — Region-specific residual mix factor with year/source
5. `EnergyPurchase` — Energy purchase record with instruments attached
6. `FacilityInfo` — Facility with grid region, country, instrument portfolio
7. `AllocationResult` — Instrument allocation with covered/uncovered breakdown
8. `CoveredEmissionResult` — Emissions from instrument-covered consumption
9. `UncoveredEmissionResult` — Emissions from residual mix (uncovered)
10. `GasEmissionDetail` — Per-gas emission result (CO2, CH4, N2O)
11. `MarketBasedResult` — Complete market-based calculation result
12. `DualReportingResult` — Combined location + market-based results
13. `CalculationRequest` — Unified calculation request
14. `BatchCalculationRequest` — Batch of calculation requests
15. `BatchCalculationResult` — Batch results aggregate
16. `ComplianceCheckResult` — Regulatory compliance result
17. `InstrumentValidationResult` — Instrument quality validation
18. `UncertaintyRequest` — Monte Carlo uncertainty request
19. `UncertaintyResult` — Uncertainty analysis result
20. `AggregationResult` — Aggregated emissions by facility/instrument/period

---

## 5. Database Schema (V061)

### Tables (14)
1. `s2m_facilities` — Facility registration with grid region and instrument portfolio
2. `s2m_instruments` — Contractual instrument registry (REC/GO/PPA/etc.)
3. `s2m_instrument_allocations` — Instrument-to-consumption allocation records
4. `s2m_residual_mix_factors` — Residual mix emission factors by region/year
5. `s2m_supplier_factors` — Supplier-specific emission factors with fuel mix
6. `s2m_energy_source_factors` — Energy source emission factors
7. `s2m_energy_purchases` — Energy purchase records
8. `s2m_calculations` — Calculation results
9. `s2m_calculation_details` — Per-instrument/per-gas calculation details
10. `s2m_dual_reporting` — Dual reporting (location + market) combined results
11. `s2m_compliance_records` — Compliance check results
12. `s2m_certificate_retirements` — Certificate retirement tracking (double-counting prevention)
13. `s2m_quality_assessments` — Instrument quality assessment records
14. `s2m_audit_entries` — Provenance/audit trail

### Hypertables (3)
- `s2m_calculation_events` — Time-series calculation events
- `s2m_instrument_events` — Instrument lifecycle events (issue, transfer, retire)
- `s2m_compliance_events` — Compliance check events

### Continuous Aggregates (2)
- `s2m_hourly_calculation_stats` — Hourly stats
- `s2m_daily_emission_totals` — Daily emission totals

---

## 6. REST API (20 Endpoints)

Prefix: `/api/v1/scope2-market`

| Method | Path | Permission | Description |
|--------|------|-----------|-------------|
| POST | /calculations | scope2-market:calculate | Single market-based calculation |
| POST | /calculations/batch | scope2-market:calculate | Batch calculation |
| GET | /calculations | scope2-market:read | List calculations |
| GET | /calculations/{id} | scope2-market:read | Get single calculation |
| DELETE | /calculations/{id} | scope2-market:delete | Delete calculation |
| POST | /instruments | scope2-market:instruments:write | Register instrument (REC/GO/PPA) |
| GET | /instruments | scope2-market:instruments:read | List instruments |
| PUT | /instruments/{id} | scope2-market:instruments:write | Update instrument |
| POST | /instruments/{id}/retire | scope2-market:instruments:write | Retire certificate |
| GET | /residual-mix | scope2-market:factors:read | List residual mix factors |
| GET | /residual-mix/{region} | scope2-market:factors:read | Get residual mix for region |
| POST | /residual-mix/custom | scope2-market:factors:write | Add custom residual mix |
| POST | /suppliers | scope2-market:suppliers:write | Register supplier EF |
| GET | /suppliers | scope2-market:suppliers:read | List supplier EFs |
| POST | /dual-reporting | scope2-market:dual:run | Generate dual report |
| GET | /dual-reporting/{id} | scope2-market:dual:read | Get dual report result |
| POST | /compliance/check | scope2-market:compliance:check | Run compliance check |
| GET | /compliance/{id} | scope2-market:compliance:read | Get compliance result |
| POST | /uncertainty | scope2-market:uncertainty:run | Run uncertainty analysis |
| GET | /aggregations | scope2-market:read | Get aggregated emissions |

---

## 7. Prometheus Metrics (12)

All prefixed `gl_s2m_`:

1. `gl_s2m_calculations_total` — Counter (instrument_type, calculation_method)
2. `gl_s2m_calculation_duration_seconds` — Histogram (calculation_method)
3. `gl_s2m_emissions_co2e_tonnes` — Counter (instrument_type, gas)
4. `gl_s2m_instruments_registered_total` — Counter (instrument_type, status)
5. `gl_s2m_instruments_retired_total` — Counter (instrument_type)
6. `gl_s2m_coverage_percentage` — Gauge (facility_id)
7. `gl_s2m_compliance_checks_total` — Counter (framework, status)
8. `gl_s2m_uncertainty_runs_total` — Counter (method)
9. `gl_s2m_errors_total` — Counter (error_type)
10. `gl_s2m_dual_reports_total` — Counter (status)
11. `gl_s2m_residual_mix_lookups_total` — Counter (source)
12. `gl_s2m_active_instruments` — Gauge (instrument_type)

---

## 8. Key Implementation Notes

### Zero-Hallucination Guarantees
- All Decimal arithmetic with ROUND_HALF_UP
- SHA-256 provenance hash at every calculation stage
- No LLM in any calculation path
- Deterministic results: same input → same output → same hash

### GHG Protocol Compliance
- Market-based method uses contractual instruments per Chapter 6-7
- Instrument hierarchy enforced per Table 6.3
- Quality criteria validated per Section 7.3 (7 mandatory criteria)
- Residual mix used for ALL uncovered consumption (not location-based fallback)
- Dual reporting mandatory: must pair with MRV-009 location-based result

### Instrument Hierarchy Rules
1. Instruments applied in priority order (bundled certs → PPAs → supplier → unbundled → green tariff → residual)
2. Each MWh of consumption matched to at most one instrument (no double-counting)
3. Certificates must be retired after use
4. Vintage must match or precede reporting period (typically ±1 year)
5. Geographic boundary: same market/interconnected grid

### Residual Mix vs Location-Based
- Residual mix ≠ grid average (location-based)
- Residual = grid average MINUS renewable claims from instruments
- Residual mix factors are typically HIGHER than location-based factors
- Ensures that renewable claims reduce market-based but don't create "free" emission reductions

### Unit Conversions
- 1 MWh = 3.6 GJ = 3.412 MMBtu
- 1 GJ = 0.2778 MWh
- 1 MMBtu = 1.055 GJ
- 1 therm = 0.1055 GJ
- All conversions done in Decimal to prevent floating-point errors

---

## 9. Acceptance Criteria

1. Calculates Scope 2 market-based emissions with 10+ contractual instrument types
2. Enforces GHG Protocol instrument hierarchy (6 priority levels)
3. Validates 7 GHG Protocol quality criteria per instrument
4. Supports 30+ regions for residual mix factors
5. Per-gas breakdown (CO2, CH4, N2O) with GWP conversion (AR4/AR5/AR6)
6. Dual reporting integration with MRV-009 (location-based)
7. Certificate retirement tracking for double-counting prevention
8. 7-engine architecture following established MRV pattern
9. V061 database migration with 14 tables, 3 hypertables
10. 20 REST API endpoints with full RBAC
11. 12 Prometheus metrics with gl_s2m_ prefix
12. SHA-256 provenance chain for complete audit trail
13. 600+ unit tests with 85%+ coverage target
14. 7 regulatory framework compliance checks (GHG Protocol, ISO 14064, CSRD, CDP, RE100, SBTi, Green-e)
15. Monte Carlo + analytical uncertainty quantification
16. Supplier-specific emission factor management
