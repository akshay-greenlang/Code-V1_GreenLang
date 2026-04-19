# PRD: AGENT-MRV-026 -- Downstream Leased Assets (Scope 3 Category 13) Agent

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-AGENT-MRV-026 |
| Agent ID | GL-MRV-S3-013 |
| Component | AGENT-MRV-026 |
| Category | GHG Protocol Scope 3, Category 13 |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-26 |

---

## 1. Overview

### 1.1 Purpose
Build a production-grade Downstream Leased Assets (Category 13) agent that calculates GHG emissions from the operation of assets owned by the reporting company (as lessor) and leased to other entities. This is the **mirror image** of Category 8 (Upstream Leased Assets) -- where Cat 8 covers assets the reporter LEASES FROM others, Cat 13 covers assets the reporter OWNS and LEASES TO others.

### 1.2 Scope Boundary
- **Category 13**: Emissions from assets OWNED by the reporting company and LEASED TO downstream tenants/lessees
- **Reporter role**: LESSOR (asset owner who leases out)
- **Accounting approach**: Applicable when the reporter uses the **financial control** or **equity share** consolidation approach, and leased-out assets are NOT already included in the reporter's Scope 1/Scope 2 inventory
- **NOT in scope**: Operating lease assets that ARE included in Scope 1/Scope 2 under operational control (those go into Scope 1/2 directly)

### 1.3 Key Distinction from Category 8
| Aspect | Category 8 (Upstream) | Category 13 (Downstream) |
|--------|----------------------|--------------------------|
| Reporter role | Lessee (tenant) | Lessor (owner) |
| Assets | Leased BY reporter | Leased TO others |
| MRV Agent | AGENT-MRV-021 | AGENT-MRV-026 |
| Agent ID | GL-MRV-S3-008 | GL-MRV-S3-013 |
| Relevance | Common (most companies lease) | Less common (asset-heavy companies: REITs, fleet companies, equipment rental, cloud providers) |
| Data source | Metered data from lessor | Tenant energy data or benchmarks |

### 1.4 Typical Reporters
- Real estate investment trusts (REITs) and property companies
- Fleet leasing companies (vehicles, aircraft)
- Equipment rental companies (construction, manufacturing)
- IT infrastructure/cloud providers (servers, data centers)
- Industrial conglomerates with leased manufacturing facilities

---

## 2. Regulatory Requirements

### 2.1 GHG Protocol Scope 3 Standard (Chapter 13)
- Lessors using financial control or equity share approach MUST report Cat 13
- If lessor uses operational control AND retains operational control of leased assets, report in Scope 1/2 (Cat 13 = 0)
- 4 calculation methods: asset-specific, lessor-specific (tenant data), average-data, spend-based
- Allocation required when lessor owns part of a building or shared asset

### 2.2 IFRS 16 / ASC 842 Lease Classification
- **Finance lease** (IFRS 16) / **Sales-type or Direct financing** (ASC 842): Lessee recognizes ROU asset; emissions may shift to lessee Scope 1/2
- **Operating lease**: Lessor retains ownership; emissions in Cat 13 unless operational control retained
- Lease classification affects boundary determination

### 2.3 Compliance Frameworks
| Framework | Requirement |
|-----------|-------------|
| GHG Protocol Scope 3 | Category 13 mandatory if material; asset-specific preferred |
| ISO 14064-1:2018 | Clause 5.2.4 indirect GHG emissions |
| CSRD / ESRS E1 | E1-6 GHG Scope 3 downstream; real estate ESRS E1-5 energy performance |
| CDP Climate Change | C6.5 Scope 3 Category 13 disclosure |
| SBTi | Include if >=40% of total Scope 3; FLAG for land-intensive |
| SB 253 (California) | All Scope 3 categories >1% materiality |
| GRI 305 | 305-3 Other indirect GHG emissions |

---

## 3. Architecture

### 3.1 Seven-Engine Design

| # | Engine | Class | Responsibility |
|---|--------|-------|---------------|
| 1 | Asset Database | `DownstreamAssetDatabaseEngine` | Asset type EFs, EUI benchmarks, grid factors, EEIO factors, building/vehicle/equipment/IT asset data |
| 2 | Asset-Specific Calculator | `AssetSpecificCalculatorEngine` | Metered energy data per leased-out asset (Tier 1) |
| 3 | Average-Data Calculator | `AverageDataCalculatorEngine` | Benchmark EUI/intensity by asset type, climate zone (Tier 2) |
| 4 | Spend-Based Calculator | `SpendBasedCalculatorEngine` | Lease revenue x EEIO factor (Tier 3) |
| 5 | Hybrid Aggregator | `HybridAggregatorEngine` | Method waterfall, portfolio aggregation, allocation |
| 6 | Compliance Checker | `ComplianceCheckerEngine` | 7 frameworks, DC rules, boundary validation |
| 7 | Pipeline | `DownstreamLeasedAssetsPipelineEngine` | 10-stage orchestration |

### 3.2 Asset Categories (4 types, mirroring Cat 8)

#### 3.2.1 Buildings (8 types x 5 climate zones)
| Building Type | Description |
|--------------|-------------|
| office | Office buildings |
| retail | Retail stores, shopping centers |
| warehouse | Warehouses, distribution centers |
| industrial | Manufacturing facilities, factories |
| data_center | Data centers, server farms |
| hotel | Hotels, hospitality properties |
| healthcare | Hospitals, clinics, labs |
| residential_multifamily | Multifamily residential (apartments) |

Climate zones: `tropical`, `arid`, `temperate`, `continental`, `polar`

EUI benchmarks (kWh/m2/yr) by type and climate zone for average-data method.

#### 3.2.2 Vehicles (8 types x 7 fuel types)
| Vehicle Type | Typical Lease |
|-------------|---------------|
| small_car | Employee pool cars |
| medium_car | Fleet sedans |
| large_car | Executive vehicles |
| suv | Utility vehicles |
| light_van | Delivery vans |
| heavy_van | Large panel vans |
| light_truck | Pickup trucks |
| heavy_truck | HGV/articulated |

Fuel types: `gasoline`, `diesel`, `lpg`, `cng`, `hybrid`, `phev`, `bev`

#### 3.2.3 Equipment (6 types)
| Equipment Type | Description |
|---------------|-------------|
| manufacturing | CNC, presses, assembly lines |
| construction | Excavators, cranes, loaders |
| generator | Diesel/gas generators |
| agricultural | Tractors, harvesters |
| mining | Drilling, hauling, crushers |
| hvac | Chillers, boilers, AHUs |

#### 3.2.4 IT Assets (7 types)
| IT Asset Type | Description |
|--------------|-------------|
| server | Rack/blade servers |
| network_switch | Network infrastructure |
| storage | SAN/NAS arrays |
| desktop | Desktop workstations |
| laptop | Laptops |
| printer | Printers/MFPs |
| copier | High-volume copiers |

### 3.3 Calculation Methods

#### Method A: Asset-Specific (Tier 1, +/-10%)
```
E = SUM_assets [ energy_consumed x grid_EF x lease_share ]
```
For buildings: energy from meter data (electricity, gas, steam, cooling).
For vehicles: fuel consumed x mobile combustion EF.
For equipment: fuel consumed x stationary/mobile EF.
For IT: power_rating x hours x PUE x grid_EF.

#### Method B: Average-Data (Tier 2, +/-30%)
```
E = SUM_assets [ floor_area x EUI_benchmark x grid_EF x lease_share ]  (buildings)
E = SUM_assets [ distance x EF_per_km x lease_share ]                   (vehicles)
E = SUM_assets [ rated_power x load_factor x hours x EF x lease_share ] (equipment)
E = SUM_assets [ power_rating x PUE x hours x grid_EF x lease_share ]  (IT)
```

#### Method C: Spend-Based (Tier 3, +/-50%)
```
E = SUM_assets [ lease_revenue x EEIO_factor x (1 + margin_adjustment) ]
```

#### Method D: Hybrid (best available per asset)
Waterfall: asset_specific -> average_data -> spend_based
DQI-weighted aggregation across portfolio.

### 3.4 Lessor-Specific Considerations
- **Tenant energy data collection**: Green lease clauses, utility data sharing agreements
- **Allocation methods**: Floor area, headcount, revenue, FTE, equal share, custom
- **Common area energy**: Separately metered or proportionally allocated
- **Vacancy adjustment**: Vacant periods may have base load emissions
- **Sub-metering**: Individual tenant metering vs. whole-building with allocation
- **NABERS / Energy Star ratings**: Can be used as proxy for EUI

### 3.5 Double-Counting Prevention Rules

| Rule ID | Description |
|---------|-------------|
| DC-DLA-001 | If lessor uses operational control AND retains operational control of asset, report in Scope 1/2 NOT Cat 13 |
| DC-DLA-002 | Do not double-count with Cat 8 (upstream leased assets) -- same asset cannot be in both |
| DC-DLA-003 | Finance lease: if lessee has operational control, lessee reports in their Scope 1/2 |
| DC-DLA-004 | Sub-leasing: intermediate lessor does NOT report in Cat 13 if not the asset owner |
| DC-DLA-005 | Common area energy: allocate proportionally, do not double-count with tenant allocations |
| DC-DLA-006 | Scope 2: grid electricity in leased buildings -- boundary with reporter's Scope 2 |
| DC-DLA-007 | REITs: distinguish between properties in Scope 1/2 (operational control) vs Cat 13 |
| DC-DLA-008 | Fleet: do not double-count with Cat 11 (use of sold products) if sold, not leased |

---

## 4. Data Models

### 4.1 Enumerations (22)
1. `AssetCategory` (4): building, vehicle, equipment, it_asset
2. `BuildingType` (8): office, retail, warehouse, industrial, data_center, hotel, healthcare, residential_multifamily
3. `VehicleType` (8): small_car, medium_car, large_car, suv, light_van, heavy_van, light_truck, heavy_truck
4. `FuelType` (7): gasoline, diesel, lpg, cng, hybrid, phev, bev
5. `EquipmentType` (6): manufacturing, construction, generator, agricultural, mining, hvac
6. `ITAssetType` (7): server, network_switch, storage, desktop, laptop, printer, copier
7. `ClimateZone` (5): tropical, arid, temperate, continental, polar
8. `CalculationMethod` (4): asset_specific, average_data, spend_based, hybrid
9. `AllocationMethod` (6): floor_area, headcount, revenue, fte, equal_share, custom
10. `LeaseType` (3): operating, finance, sale_leaseback
11. `ConsolidationApproach` (3): financial_control, equity_share, operational_control
12. `EFSource` (6): DEFRA_2024, EPA_2024, IEA_2024, EGRID_2024, IPCC_AR6, CUSTOM
13. `DataQualityTier` (3): tier_1, tier_2, tier_3
14. `DQIDimension` (5): temporal, geographical, technological, completeness, reliability
15. `ComplianceFramework` (7): ghg_protocol, iso_14064, csrd_esrs, cdp, sbti, sb_253, gri
16. `ComplianceStatus` (4): compliant, non_compliant, partial, not_applicable
17. `PipelineStage` (10): validate, classify, normalize, resolve_efs, calculate, allocate, aggregate, compliance, provenance, seal
18. `UncertaintyMethod` (3): monte_carlo, analytical, ipcc_tier2
19. `BatchStatus` (4): pending, processing, completed, failed
20. `GWPSource` (2): AR5, AR6
21. `EnergyType` (4): electricity, natural_gas, steam, chilled_water
22. `OccupancyStatus` (3): occupied, vacant, partially_occupied

### 4.2 Constant Tables
1. `BUILDING_EUI_BENCHMARKS` -- 8 building types x 5 climate zones (kWh/m2/yr)
2. `VEHICLE_EMISSION_FACTORS` -- 8 vehicle types x 7 fuel types (kgCO2e/km)
3. `EQUIPMENT_FUEL_CONSUMPTION` -- 6 equipment types (L/hr or kWh/hr at rated load)
4. `IT_ASSET_POWER_RATINGS` -- 7 IT types (kW typical, PUE default)
5. `GRID_EMISSION_FACTORS` -- 12 countries + 26 eGRID subregions (kgCO2e/kWh)
6. `FUEL_EMISSION_FACTORS` -- 8 fuel types (kgCO2e/L or kgCO2e/kWh)
7. `EEIO_SPEND_FACTORS` -- 10 leasing/rental NAICS codes (kgCO2e/$)
8. `ALLOCATION_DEFAULTS` -- Default allocation percentages by building area use
9. `VACANCY_BASE_LOAD` -- Base load fractions by building type when vacant
10. `REFRIGERANT_GWPS` -- 15 common refrigerants (IPCC AR6)
11. `COUNTRY_CLIMATE_ZONES` -- 30+ country-to-climate-zone mappings
12. `DC_RULES` -- 8 double-counting prevention rules
13. `COMPLIANCE_FRAMEWORK_RULES` -- 7 frameworks with requirement lists
14. `DQI_SCORING` -- 5 dimensions x 3 tiers
15. `UNCERTAINTY_RANGES` -- 4 methods x 3 tiers

### 4.3 Pydantic Models (14 frozen)
1. `BuildingAssetInput` -- Building-specific input with floor_area, EUI, metered data
2. `VehicleAssetInput` -- Vehicle-specific input with distance, fuel consumption
3. `EquipmentAssetInput` -- Equipment input with hours, load factor, fuel type
4. `ITAssetInput` -- IT asset input with power rating, PUE, hours
5. `LeaseInfo` -- Lease details (type, term, share, start/end dates)
6. `AllocationInfo` -- Allocation method and parameters
7. `AssetCalculationResult` -- Per-asset calculation result
8. `PortfolioResult` -- Portfolio aggregation result
9. `AvoidedEmissions` -- Avoided emissions from energy efficiency measures
10. `ComplianceResult` -- Compliance check result per framework
11. `ProvenanceRecord` -- SHA-256 hash chain record
12. `DataQualityScore` -- 5-dimension DQI
13. `UncertaintyResult` -- Monte Carlo / analytical result
14. `AggregationResult` -- Time-series aggregation

---

## 5. Database Schema (V077)

### 5.1 Tables (21)

**Reference Tables (10):**
1. `gl_dla_building_benchmarks` -- EUI by building type and climate zone
2. `gl_dla_vehicle_emission_factors` -- Per-km EFs by vehicle and fuel type
3. `gl_dla_equipment_factors` -- Fuel consumption and EFs by equipment type
4. `gl_dla_it_asset_factors` -- Power ratings, PUE defaults by IT asset type
5. `gl_dla_grid_emission_factors` -- Country/region grid EFs
6. `gl_dla_fuel_emission_factors` -- Fuel-type EFs with WTT
7. `gl_dla_eeio_spend_factors` -- NAICS-based spend EFs
8. `gl_dla_refrigerant_gwps` -- Refrigerant GWP values
9. `gl_dla_vacancy_factors` -- Base load by building type during vacancy
10. `gl_dla_allocation_defaults` -- Default allocation percentages

**Operational Tables (8):**
11. `gl_dla_calculations` -- **HYPERTABLE** (7-day chunks): Main calculation records
12. `gl_dla_asset_results` -- Per-asset calculation details
13. `gl_dla_allocation_records` -- Allocation audit trail
14. `gl_dla_compliance_checks` -- **HYPERTABLE** (30-day chunks): Compliance results
15. `gl_dla_aggregations` -- **HYPERTABLE** (30-day chunks): Period aggregations
16. `gl_dla_provenance_records` -- SHA-256 hash chains
17. `gl_dla_audit_trail` -- Operation audit log
18. `gl_dla_batch_jobs` -- Batch processing status

**Supporting Tables (3):**
19. `gl_dla_data_quality_scores` -- 5-dimension DQI
20. `gl_dla_uncertainty_results` -- Uncertainty analysis results
21. `gl_dla_tenant_energy_data` -- Collected tenant energy consumption data

### 5.2 Hypertables & Aggregates
- 3 hypertables: calculations (7-day), compliance_checks (30-day), aggregations (30-day)
- 2 continuous aggregates: `gl_dla_daily_by_asset_type`, `gl_dla_monthly_by_category`

### 5.3 RLS Policies
- All operational tables have tenant_id-based RLS
- `app.current_tenant_id` session variable

---

## 6. API Design

### 6.1 Endpoints (22 at `/api/v1/downstream-leased-assets`)

**Calculation Endpoints (10 POST):**
1. `POST /calculate` -- Full pipeline calculation
2. `POST /calculate/asset-specific` -- Asset-specific (metered data)
3. `POST /calculate/asset-specific/building` -- Building-specific calculation
4. `POST /calculate/asset-specific/vehicle` -- Vehicle fleet calculation
5. `POST /calculate/asset-specific/equipment` -- Equipment calculation
6. `POST /calculate/asset-specific/it-asset` -- IT infrastructure calculation
7. `POST /calculate/average-data` -- Benchmark-based calculation
8. `POST /calculate/spend-based` -- Revenue/spend-based EEIO
9. `POST /calculate/hybrid` -- Hybrid method waterfall
10. `POST /calculate/batch` -- Batch processing (up to 10,000 assets)

**Compliance & Analysis (2 POST):**
11. `POST /compliance/check` -- Multi-framework compliance check
12. `POST /calculate/portfolio` -- Portfolio analysis with allocation

**Data Retrieval (10 GET + 1 DELETE):**
13. `GET /calculations/{id}` -- Get calculation detail
14. `GET /calculations` -- List calculations with pagination
15. `DELETE /calculations/{id}` -- Soft-delete calculation
16. `GET /emission-factors/{asset_type}` -- Lookup EFs by asset type
17. `GET /building-benchmarks` -- EUI benchmarks by building type
18. `GET /grid-factors` -- Grid emission factors by region
19. `GET /allocation-methods` -- Available allocation methods
20. `GET /aggregations` -- Time-series aggregations
21. `GET /provenance/{id}` -- Provenance chain for calculation
22. `GET /health` -- Health check

---

## 7. File Structure

### 7.1 Source Files (15)
```
greenlang/downstream_leased_assets/
    __init__.py                           (~130 lines)
    models.py                             (~2,200 lines)
    config.py                             (~2,300 lines)
    metrics.py                            (~1,200 lines)
    provenance.py                         (~1,600 lines)
    downstream_asset_database.py          (~2,400 lines)
    asset_specific_calculator.py          (~2,200 lines)
    average_data_calculator.py            (~2,100 lines)
    spend_based_calculator.py             (~2,000 lines)
    hybrid_aggregator.py                  (~2,300 lines)
    compliance_checker.py                 (~2,500 lines)
    downstream_leased_assets_pipeline.py  (~1,700 lines)
    setup.py                              (~1,400 lines)
    api/
        __init__.py                       (~1 line)
        router.py                         (~3,000 lines)
```

### 7.2 Test Files (14)
```
tests/unit/mrv/test_downstream_leased_assets/
    __init__.py
    conftest.py                           (~1,500 lines)
    test_models.py                        (~1,300 lines)
    test_config.py                        (~450 lines)
    test_downstream_asset_database.py     (~400 lines)
    test_asset_specific_calculator.py     (~400 lines)
    test_average_data_calculator.py       (~350 lines)
    test_spend_based_calculator.py        (~350 lines)
    test_hybrid_aggregator.py             (~400 lines)
    test_compliance_checker.py            (~350 lines)
    test_downstream_leased_assets_pipeline.py (~330 lines)
    test_provenance.py                    (~600 lines)
    test_setup.py                         (~320 lines)
    test_api.py                           (~430 lines)
```

### 7.3 Migration
```
deployment/database/migrations/sql/V077__downstream_leased_assets_service.sql (~1,100 lines)
```

---

## 8. Auth Integration

### 8.1 Permission Map (22 entries)
```
POST:/api/v1/downstream-leased-assets/calculate                        -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/asset-specific          -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/asset-specific/building -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/asset-specific/vehicle  -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/asset-specific/equipment -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/asset-specific/it-asset -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/average-data           -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/spend-based            -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/hybrid                 -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/batch                  -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/calculate/portfolio              -> downstream-leased-assets:calculate
POST:/api/v1/downstream-leased-assets/compliance/check                 -> downstream-leased-assets:compliance
GET:/api/v1/downstream-leased-assets/calculations/{id}                 -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/calculations                      -> downstream-leased-assets:read
DELETE:/api/v1/downstream-leased-assets/calculations/{id}              -> downstream-leased-assets:delete
GET:/api/v1/downstream-leased-assets/emission-factors/{asset_type}     -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/building-benchmarks               -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/grid-factors                      -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/allocation-methods                -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/aggregations                      -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/provenance/{id}                   -> downstream-leased-assets:read
GET:/api/v1/downstream-leased-assets/health                            -> downstream-leased-assets:read
```

### 8.2 Router Registration
```python
from greenlang.downstream_leased_assets.setup import get_router as get_dla_router
```

---

## 9. Testing Strategy

### 9.1 Test Targets
- 600+ tests (440+ base test functions, 40+ parametrize decorators)
- All 7 engines covered with unit tests
- Thread safety tested with 10-12 concurrent threads
- Decimal precision validated throughout
- Provenance hash determinism verified
- Avoided emissions always separate from gross
- DC rules tested individually
- Allocation methods tested with edge cases

### 9.2 Key Test Scenarios
- Building portfolio: Mixed office/retail/warehouse with different climate zones
- Vehicle fleet: Mixed fuel types, BEV zero-tailpipe validation
- Equipment rental: Load factor variations, seasonal usage
- IT assets: PUE adjustment, data center vs. individual devices
- Vacancy: Base load during vacant periods
- Allocation: Floor area vs. headcount vs. revenue
- Boundary: Operational control exclusion (Scope 1/2 boundary)
- DC rules: Cat 8 vs Cat 13 boundary, finance lease classification

---

## 10. Dependencies & Integration

### 10.1 Shared Infrastructure
- `greenlang.infrastructure.auth_service` (JWT + RBAC)
- `greenlang.infrastructure.observability` (Prometheus, OTel)
- Shared grid emission factors with MRV-009/010 (Scope 2 agents)
- Shared vehicle EFs with MRV-003 (Mobile Combustion)
- Shared building EFs with MRV-021 (Upstream Leased Assets)

### 10.2 Mirror Relationship with Cat 8
- Same 4 asset categories (buildings, vehicles, equipment, IT)
- Same EUI benchmarks and EFs (different perspective: lessor vs lessee)
- Same allocation methods
- Key difference: data availability (lessor may have less visibility into tenant operations)
- Key difference: vacancy handling (lessor perspective)
- Key difference: tenant data collection mechanisms

---

## 11. Acceptance Criteria

1. All 15 source files created with production-quality code
2. All 14 test files with 600+ tests
3. V077 migration with 21 tables, 3 hypertables, 2 continuous aggregates
4. Auth integration: 22 permission entries + router registration
5. Zero-hallucination: all calculations use deterministic Decimal arithmetic
6. IPCC AR5/AR6 GWP support
7. 4 asset categories fully implemented with all sub-types
8. 4 calculation methods with proper tier classification
9. 8 double-counting prevention rules enforced
10. 7 compliance frameworks validated
11. SHA-256 provenance chains with Merkle trees
12. Thread-safe singletons on all engines
13. Memory files updated with MRV-026 entry
