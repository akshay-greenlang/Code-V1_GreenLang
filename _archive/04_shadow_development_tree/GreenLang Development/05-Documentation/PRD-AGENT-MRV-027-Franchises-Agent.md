# PRD: AGENT-MRV-027 -- Franchises (Scope 3 Category 14) Agent

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-AGENT-MRV-027 |
| Agent ID | GL-MRV-S3-014 |
| Component | AGENT-MRV-027 |
| Category | GHG Protocol Scope 3, Category 14 |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-26 |

---

## 1. Overview

### 1.1 Purpose
Build a production-grade Franchises (Category 14) agent that calculates GHG emissions from the operation of franchises not included in the franchisor's Scope 1 and Scope 2. This category is relevant for **franchisors** (companies that grant franchises/licenses to other entities).

### 1.2 Scope Boundary
- **Category 14**: Emissions from the operation of franchises -- reported by the **franchisor**
- **Reporter role**: FRANCHISOR (brand owner who grants franchise rights)
- **Applicable when**: Franchisor uses **financial control** or **equity share** consolidation approach, and franchise operations are NOT included in Scope 1/Scope 2
- **NOT in scope**: If franchisor uses operational control and does NOT have operational control of franchisees, Cat 14 applies. If franchisor DOES have operational control, report in Scope 1/2 directly

### 1.3 Key Distinction: Cat 14 vs Cat 8/13
| Aspect | Cat 8 (Upstream Leased) | Cat 13 (Downstream Leased) | Cat 14 (Franchises) |
|--------|------------------------|---------------------------|---------------------|
| Reporter | Lessee | Lessor | Franchisor |
| Relationship | Lease agreement | Lease agreement | Franchise agreement |
| Assets | Physical assets leased | Physical assets owned & leased | Franchise operations (brand, system, know-how) |
| Typical reporters | Most companies | REITs, fleet cos | Restaurant chains, hotels, retail brands |
| MRV Agent | MRV-021 | MRV-026 | MRV-027 |

### 1.4 Typical Reporters
- Quick-service restaurant (QSR) chains (McDonald's, Subway, KFC, Domino's)
- Hotel chains (Marriott, Hilton, IHG, Wyndham)
- Retail franchise brands (7-Eleven, Circle K)
- Automotive dealership networks
- Fitness/gym franchises (Planet Fitness, Anytime Fitness)
- Convenience store networks
- Gas station/fuel retail brands
- Real estate brokerage franchises

---

## 2. Regulatory Requirements

### 2.1 GHG Protocol Scope 3 Standard (Chapter 14)
- Franchisors using financial control or equity share MUST report Cat 14
- 3 calculation methods: franchise-specific (metered data), average-data (benchmarks), spend-based
- Franchise emissions include: Scope 1 (stationary/mobile combustion, refrigerants, process) + Scope 2 (purchased electricity, heating, cooling) of each franchisee
- Franchisor should collect data from franchisees or estimate using benchmarks

### 2.2 Franchise Industry Context
- **Franchise Disclosure Document (FDD)**: US FTC Rule 436 requires franchisors to disclose material info
- **International Franchise Association (IFA)**: Industry standards for franchise operations
- **Franchise Agreement**: Contractual basis for data collection requirements
- **Master Franchise**: Sub-franchisor model adds complexity (multi-tier)
- **Area Development**: Multiple units under single franchisee

### 2.3 Compliance Frameworks
| Framework | Requirement |
|-----------|-------------|
| GHG Protocol Scope 3 | Category 14 mandatory if material; franchise-specific preferred |
| ISO 14064-1:2018 | Clause 5.2.4 indirect GHG emissions |
| CSRD / ESRS E1 | E1-6 GHG Scope 3 downstream; franchise value chain |
| CDP Climate Change | C6.5 Scope 3 Category 14 disclosure |
| SBTi | Include if >=40% of total Scope 3; franchise engagement target |
| SB 253 (California) | All Scope 3 categories >1% materiality |
| GRI 305 | 305-3 Other indirect GHG emissions |

---

## 3. Architecture

### 3.1 Seven-Engine Design

| # | Engine | Class | Responsibility |
|---|--------|-------|---------------|
| 1 | Franchise Database | `FranchiseDatabaseEngine` | Franchise type EFs, EUI benchmarks, industry benchmarks, grid factors |
| 2 | Franchise-Specific Calculator | `FranchiseSpecificCalculatorEngine` | Metered energy/fuel data per franchise unit (Tier 1) |
| 3 | Average-Data Calculator | `AverageDataCalculatorEngine` | Industry benchmarks by franchise type, size, region (Tier 2) |
| 4 | Spend-Based Calculator | `SpendBasedCalculatorEngine` | Franchise revenue/royalty x EEIO factor (Tier 3) |
| 5 | Hybrid Aggregator | `HybridAggregatorEngine` | Method waterfall, network aggregation, tiered data quality |
| 6 | Compliance Checker | `ComplianceCheckerEngine` | 7 frameworks, DC rules, boundary validation |
| 7 | Pipeline | `FranchisesPipelineEngine` | 10-stage orchestration |

### 3.2 Franchise Types (10 categories)

| Franchise Type | Description | Typical Energy Profile |
|---------------|-------------|----------------------|
| qsr_restaurant | Quick-service restaurants | High cooking energy, refrigeration, HVAC |
| full_service_restaurant | Sit-down restaurants | Higher HVAC, lighting |
| hotel | Hotels, motels, resorts | High HVAC, water heating, laundry |
| convenience_store | Convenience/gas stations | Refrigeration, lighting 24/7 |
| retail_store | General retail | HVAC, lighting |
| fitness_center | Gyms, fitness studios | HVAC, equipment, water heating |
| automotive_service | Auto repair, oil change, car wash | Equipment, water, chemicals |
| healthcare_clinic | Urgent care, dental, optometry | Medical equipment, HVAC, sterilization |
| education_center | Tutoring, test prep, daycare | HVAC, lighting, IT |
| other_service | Cleaning, printing, business services | Varies |

### 3.3 Emission Sources per Franchise Unit

**Scope 1 of franchisee (reported by franchisor as Cat 14):**
- Stationary combustion (cooking, heating, backup generators)
- Mobile combustion (delivery vehicles, company cars)
- Refrigerant leakage (F-gases in HVAC and commercial refrigeration)
- Process emissions (if applicable)

**Scope 2 of franchisee (reported by franchisor as Cat 14):**
- Purchased electricity
- Purchased heating/steam
- Purchased cooling

### 3.4 Calculation Methods

#### Method A: Franchise-Specific (Tier 1, +/-10%)
```
E_unit = SUM_sources [ activity_data x EF ]
E_total = SUM_franchises [ E_unit ]
```
Franchisor collects metered energy data from each franchise unit.

#### Method B: Average-Data (Tier 2, +/-30%)
```
E_unit = floor_area x EUI_benchmark(type, climate, region) x grid_EF
E_total = SUM_franchises [ E_unit ]
```
OR
```
E_unit = revenue_per_unit x revenue_intensity_EF(type)
```
Uses industry benchmarks (ENERGY STAR Portfolio Manager, CBECS, IEA).

#### Method C: Spend-Based (Tier 3, +/-50%)
```
E_total = total_franchise_revenue x EEIO_factor(NAICS)
```
OR
```
E_total = total_royalty_income x royalty_to_emission_factor
```

#### Method D: Hybrid (best available per unit)
Waterfall: franchise_specific -> average_data -> spend_based
Tiered approach: top franchisees provide metered data, remainder use benchmarks.

### 3.5 Franchise Network Features
- **Multi-tier franchise**: Master franchisee -> sub-franchisees
- **Area development agreements**: Single franchisee, multiple units
- **Company-owned vs franchised**: Only franchised units in Cat 14 (company-owned in Scope 1/2)
- **New openings / closures**: Pro-rata for partial-year operations
- **Franchise data collection**: Survey-based, utility data sharing, portal-based
- **Data coverage tiers**: Tier 1 (100% metered), Tier 2 (top 80% metered + 20% estimated), Tier 3 (all estimated)

### 3.6 Double-Counting Prevention Rules

| Rule ID | Description |
|---------|-------------|
| DC-FRN-001 | Company-owned units MUST be in Scope 1/2, NOT Cat 14 |
| DC-FRN-002 | Do not double-count with Cat 13 (downstream leased assets) if same property |
| DC-FRN-003 | Franchisee's own Scope 1/2 becomes franchisor's Cat 14 -- boundary clarity |
| DC-FRN-004 | Multi-brand: if franchisee operates multiple brands, allocate by brand |
| DC-FRN-005 | Master franchise: do not count sub-franchisee twice (once at master, once at sub) |
| DC-FRN-006 | Transition units: company-owned converting to franchise (pro-rata) |
| DC-FRN-007 | Scope 2: grid electricity at franchise -- boundary with franchisor's Scope 2 |
| DC-FRN-008 | Supply chain: Cat 1 (purchased goods) vs Cat 14 (franchise operations) boundary |

---

## 4. Data Models

### 4.1 Enumerations (20)
1. `FranchiseType` (10): qsr_restaurant, full_service_restaurant, hotel, convenience_store, retail_store, fitness_center, automotive_service, healthcare_clinic, education_center, other_service
2. `OwnershipType` (3): franchised, company_owned, joint_venture
3. `FranchiseAgreementType` (4): single_unit, multi_unit, area_development, master_franchise
4. `CalculationMethod` (4): franchise_specific, average_data, spend_based, hybrid
5. `EmissionSource` (7): stationary_combustion, mobile_combustion, refrigerant_leakage, process_emissions, purchased_electricity, purchased_heating, purchased_cooling
6. `FuelType` (8): natural_gas, propane, diesel, gasoline, fuel_oil, lpg, biomass, electricity
7. `ClimateZone` (5): tropical, arid, temperate, continental, polar
8. `EFSource` (6): DEFRA_2024, EPA_2024, IEA_2024, EGRID_2024, IPCC_AR6, CUSTOM
9. `DataQualityTier` (3): tier_1, tier_2, tier_3
10. `DQIDimension` (5): temporal, geographical, technological, completeness, reliability
11. `ComplianceFramework` (7): ghg_protocol, iso_14064, csrd_esrs, cdp, sbti, sb_253, gri
12. `ComplianceStatus` (4): compliant, non_compliant, partial, not_applicable
13. `PipelineStage` (10): validate, classify, normalize, resolve_efs, calculate, allocate, aggregate, compliance, provenance, seal
14. `UncertaintyMethod` (3): monte_carlo, analytical, ipcc_tier2
15. `BatchStatus` (4): pending, processing, completed, failed
16. `GWPSource` (2): AR5, AR6
17. `DataCollectionMethod` (4): metered, survey, estimated, default
18. `UnitStatus` (4): active, temporarily_closed, permanently_closed, under_construction
19. `ConsolidationApproach` (3): financial_control, equity_share, operational_control
20. `RefrigerantType` (10): R_410A, R_32, R_134a, R_404A, R_507A, R_22, R_407C, R_290, R_744, R_1234yf

### 4.2 Constant Tables (15)
1. `FRANCHISE_EUI_BENCHMARKS` -- 10 franchise types x 5 climate zones (kWh/m2/yr)
2. `FRANCHISE_REVENUE_INTENSITY` -- 10 franchise types (kgCO2e/$ revenue)
3. `COOKING_FUEL_CONSUMPTION` -- QSR/FSR fuel profiles (natural_gas, propane, electricity)
4. `REFRIGERATION_LEAKAGE_RATES` -- By equipment type (walk-in, reach-in, display case)
5. `GRID_EMISSION_FACTORS` -- 12 countries + 26 eGRID subregions
6. `FUEL_EMISSION_FACTORS` -- 8 fuel types (kgCO2e/unit)
7. `REFRIGERANT_GWPS` -- 10 common refrigerants (IPCC AR6)
8. `EEIO_SPEND_FACTORS` -- 10 NAICS codes for franchise industries
9. `HOTEL_ENERGY_BENCHMARKS` -- By hotel class (economy, midscale, upscale, luxury) and climate
10. `VEHICLE_EMISSION_FACTORS` -- Delivery fleet EFs
11. `DC_RULES` -- 8 double-counting rules
12. `COMPLIANCE_FRAMEWORK_RULES` -- 7 frameworks with requirements
13. `DQI_SCORING` -- 5 dimensions x 3 tiers
14. `UNCERTAINTY_RANGES` -- 4 methods x 3 tiers
15. `COUNTRY_CLIMATE_ZONES` -- 30+ country-to-climate-zone mappings

### 4.3 Pydantic Models (14 frozen)
1. `FranchiseUnitInput` -- Unit-level input with type, area, energy data, location
2. `FranchiseNetworkInput` -- Network-level input with unit count, coverage
3. `CookingEnergyInput` -- Restaurant-specific cooking energy data
4. `RefrigerationInput` -- Refrigerant charge and leakage data
5. `DeliveryFleetInput` -- Franchise delivery vehicle data
6. `HotelOperationsInput` -- Hotel-specific: rooms, occupancy, laundry
7. `FranchiseCalculationResult` -- Per-unit calculation result
8. `NetworkAggregationResult` -- Network-wide aggregation
9. `ComplianceResult` -- Per-framework compliance result
10. `ProvenanceRecord` -- SHA-256 hash chain record
11. `DataQualityScore` -- 5-dimension DQI
12. `UncertaintyResult` -- Monte Carlo / analytical result
13. `DataCoverageReport` -- Metered vs estimated vs default breakdown
14. `AggregationResult` -- Time-series aggregation

---

## 5. Database Schema (V078)

### 5.1 Tables (21)

**Reference Tables (10):**
1. `gl_frn_franchise_benchmarks` -- EUI by franchise type and climate zone
2. `gl_frn_revenue_intensity_factors` -- Revenue-based EFs by franchise type
3. `gl_frn_cooking_fuel_profiles` -- Cooking energy by restaurant type
4. `gl_frn_refrigeration_factors` -- Leakage rates by equipment type
5. `gl_frn_grid_emission_factors` -- Country/region grid EFs
6. `gl_frn_fuel_emission_factors` -- Fuel-type EFs
7. `gl_frn_eeio_spend_factors` -- NAICS-based spend EFs
8. `gl_frn_refrigerant_gwps` -- Refrigerant GWP values
9. `gl_frn_hotel_benchmarks` -- Hotel energy by class and climate
10. `gl_frn_vehicle_emission_factors` -- Delivery fleet EFs

**Operational Tables (8):**
11. `gl_frn_calculations` -- **HYPERTABLE** (7-day): Main calculation records
12. `gl_frn_unit_results` -- Per-unit calculation details
13. `gl_frn_network_aggregations` -- Network-level aggregations
14. `gl_frn_compliance_checks` -- **HYPERTABLE** (30-day): Compliance results
15. `gl_frn_aggregations` -- **HYPERTABLE** (30-day): Period aggregations
16. `gl_frn_provenance_records` -- SHA-256 hash chains
17. `gl_frn_audit_trail` -- Operation audit log
18. `gl_frn_batch_jobs` -- Batch processing status

**Supporting Tables (3):**
19. `gl_frn_data_quality_scores` -- 5-dimension DQI
20. `gl_frn_uncertainty_results` -- Uncertainty analysis results
21. `gl_frn_data_coverage` -- Per-unit data collection tracking

---

## 6. API Design

### 6.1 Endpoints (22 at `/api/v1/franchises`)

**Calculation Endpoints (10 POST):**
1. `POST /calculate` -- Full pipeline calculation
2. `POST /calculate/franchise-specific` -- Unit-level metered data
3. `POST /calculate/franchise-specific/restaurant` -- Restaurant-specific
4. `POST /calculate/franchise-specific/hotel` -- Hotel-specific
5. `POST /calculate/franchise-specific/retail` -- Retail-specific
6. `POST /calculate/average-data` -- Benchmark-based
7. `POST /calculate/spend-based` -- Revenue/royalty EEIO
8. `POST /calculate/hybrid` -- Hybrid method waterfall
9. `POST /calculate/batch` -- Batch (up to 10,000 units)
10. `POST /calculate/network` -- Full network analysis

**Compliance & Analysis (2 POST):**
11. `POST /compliance/check` -- Multi-framework compliance check
12. `POST /calculate/portfolio` -- Portfolio/brand analysis

**Data Retrieval (10 GET + 1 DELETE):**
13. `GET /calculations/{id}` -- Get calculation detail
14. `GET /calculations` -- List calculations
15. `DELETE /calculations/{id}` -- Soft-delete
16. `GET /emission-factors/{franchise_type}` -- EFs by type
17. `GET /benchmarks` -- EUI benchmarks
18. `GET /grid-factors` -- Grid EFs
19. `GET /data-coverage` -- Data collection coverage report
20. `GET /aggregations` -- Time-series aggregations
21. `GET /provenance/{id}` -- Provenance chain
22. `GET /health` -- Health check

---

## 7. File Structure

### 7.1 Source Files (15)
```
greenlang/franchises/
    __init__.py                    (~130 lines)
    models.py                      (~2,200 lines)
    config.py                      (~2,300 lines)
    metrics.py                     (~1,200 lines)
    provenance.py                  (~1,600 lines)
    franchise_database.py          (~2,400 lines)
    franchise_specific_calculator.py (~2,200 lines)
    average_data_calculator.py     (~2,100 lines)
    spend_based_calculator.py      (~2,000 lines)
    hybrid_aggregator.py           (~2,300 lines)
    compliance_checker.py          (~2,500 lines)
    franchises_pipeline.py         (~1,700 lines)
    setup.py                       (~1,400 lines)
    api/
        __init__.py                (~1 line)
        router.py                  (~3,000 lines)
```

### 7.2 Test Files (14)
```
tests/unit/mrv/test_franchises/
    __init__.py, conftest.py, test_models.py, test_config.py,
    test_franchise_database.py, test_franchise_specific_calculator.py,
    test_average_data_calculator.py, test_spend_based_calculator.py,
    test_hybrid_aggregator.py, test_compliance_checker.py,
    test_franchises_pipeline.py, test_provenance.py, test_setup.py, test_api.py
```

### 7.3 Migration
```
deployment/database/migrations/sql/V078__franchises_service.sql (~1,100 lines)
```

---

## 8. Auth Integration

### 8.1 Permission Map (22 entries)
```
POST:/api/v1/franchises/calculate                            -> franchises:calculate
POST:/api/v1/franchises/calculate/franchise-specific          -> franchises:calculate
POST:/api/v1/franchises/calculate/franchise-specific/restaurant -> franchises:calculate
POST:/api/v1/franchises/calculate/franchise-specific/hotel    -> franchises:calculate
POST:/api/v1/franchises/calculate/franchise-specific/retail   -> franchises:calculate
POST:/api/v1/franchises/calculate/average-data               -> franchises:calculate
POST:/api/v1/franchises/calculate/spend-based                -> franchises:calculate
POST:/api/v1/franchises/calculate/hybrid                     -> franchises:calculate
POST:/api/v1/franchises/calculate/batch                      -> franchises:calculate
POST:/api/v1/franchises/calculate/network                    -> franchises:calculate
POST:/api/v1/franchises/calculate/portfolio                  -> franchises:calculate
POST:/api/v1/franchises/compliance/check                     -> franchises:compliance
GET:/api/v1/franchises/calculations/{id}                     -> franchises:read
GET:/api/v1/franchises/calculations                          -> franchises:read
DELETE:/api/v1/franchises/calculations/{id}                  -> franchises:delete
GET:/api/v1/franchises/emission-factors/{franchise_type}     -> franchises:read
GET:/api/v1/franchises/benchmarks                            -> franchises:read
GET:/api/v1/franchises/grid-factors                          -> franchises:read
GET:/api/v1/franchises/data-coverage                         -> franchises:read
GET:/api/v1/franchises/aggregations                          -> franchises:read
GET:/api/v1/franchises/provenance/{id}                       -> franchises:read
GET:/api/v1/franchises/health                                -> franchises:read
```

---

## 9. Acceptance Criteria

1. All 15 source files with production-quality code
2. All 14 test files with 600+ tests
3. V078 migration with 21 tables, 3 hypertables, 2 continuous aggregates
4. Auth integration: 22 permission entries + router registration
5. Zero-hallucination: all calculations use deterministic Decimal arithmetic
6. 10 franchise types with industry-specific benchmarks
7. 4 calculation methods with proper tier classification
8. 8 double-counting prevention rules enforced
9. 7 compliance frameworks validated
10. SHA-256 provenance chains with Merkle trees
11. Thread-safe singletons on all engines
12. Memory files updated with MRV-027 entry
