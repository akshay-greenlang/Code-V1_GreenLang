# PRD: AGENT-MRV-006 Land Use Emissions Agent (GL-MRV-SCOPE1-006)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE1-006 |
| **Internal Label** | AGENT-MRV-006 |
| **Category** | Layer 3 - MRV / Accounting Agents (Scope 1) |
| **Package** | `greenlang/land_use_emissions/` |
| **DB Migration** | V057 |
| **Priority** | P1 - Critical |
| **Regulatory Drivers** | IPCC 2006 Guidelines Vol 4 (AFOLU), IPCC 2019 Refinement, GHG Protocol Land Sector & Removals Guidance (2022), GHG Protocol Corporate Standard, ISO 14064-1:2018, CSRD/ESRS E1, EU LULUCF Regulation 2018/841 (amended 2023/839), SBTi FLAG Guidance, UNFCCC National Inventory Guidelines |

## 2. Problem Statement

Land use, land-use change, and forestry (LULUCF) represent one of the largest and most complex sources of GHG emissions globally—accounting for approximately 23% of total anthropogenic greenhouse gas emissions according to the IPCC. Organizations with significant land holdings, agricultural operations, forestry activities, or real estate portfolios must accurately quantify:

- **Carbon stock changes** in biomass (above-ground and below-ground), dead organic matter (dead wood and litter), and soil organic carbon across all managed land categories
- **Direct emissions** from land-use change (deforestation, peatland drainage, wetland conversion, grassland conversion)
- **Non-CO2 emissions** from managed land (N2O from fertilizer application and soil disturbance, CH4 from wetland/peatland management)
- **Carbon removals** from afforestation, reforestation, improved forest management, and soil carbon sequestration
- **Land-use transition matrices** tracking conversions between forest land, cropland, grassland, wetlands, settlements, and other land over 20-year default transition periods

This is critical for:
- GHG Protocol Scope 1 reporting (direct land-related emissions from owned/controlled land)
- SBTi FLAG (Forest, Land and Agriculture) target setting
- EU LULUCF Regulation compliance for member state inventories
- CSRD/ESRS E1 disclosure requirements for land-related emissions and removals
- TCFD physical risk assessment for land-dependent businesses
- Nature-based solution (NBS) quantification for carbon credit programs

## 3. Existing Layer 1 Capabilities

Two basic land use files exist in the Layer 1 agent codebase:

- `greenlang/agents/mrv/nbs/land_use_change.py` (GL-MRV-NBS-006): Basic LULUCF stock-change and gain-loss methods, simple deforestation/afforestation calculations
- `greenlang/agents/mrv/agriculture/land_use_change.py` (GL-MRV-AGR-004): Basic CO2 from land management with lookup tables by climate zone

**Current gaps:** No comprehensive carbon pool tracking across 5 pools, no land-use transition matrix, no IPCC Tier 2/3 methods, no soil organic carbon modeling with management/input factors, no dead organic matter (DOM) calculations, no 20-year transition period tracking, no fire/harvest disturbance modeling, no peatland-specific calculations, no regulatory compliance checking, no uncertainty quantification, no provenance tracking.

## 4. Gaps Requiring Layer 3 Production Implementation

| # | Gap | Impact |
|---|-----|--------|
| 1 | No 5-pool carbon stock framework | Cannot track AGB, BGB, dead wood, litter, SOC separately |
| 2 | No land-use transition matrix | Cannot track 6×6 land category conversions over 20 years |
| 3 | No IPCC Tier 2/3 biomass methods | Limited to global default factors only |
| 4 | No SOC reference stock modeling | Cannot apply IPCC climate/soil/land use/management/input factors |
| 5 | No dead organic matter calculations | Missing dead wood and litter pool dynamics |
| 6 | No fire/harvest disturbance modeling | Cannot calculate emissions from biomass burning or logging |
| 7 | No peatland-specific calculations | Missing CO2/CH4/N2O from drained/rewetted peatlands |
| 8 | No N2O from soil management | Missing fertilizer, residue, and mineralization N2O |
| 9 | No gain-loss method implementation | Only stock-difference, not annual increment/loss tracking |
| 10 | No 20-year transition period tracking | Cannot model gradual SOC/biomass changes post-conversion |
| 11 | No uncertainty quantification | No Monte Carlo for carbon stock variability |
| 12 | No regulatory compliance checking | No GHG Protocol/IPCC/SBTi FLAG/EU LULUCF validation |
| 13 | No carbon removal accounting | Cannot separate gross emissions from net emissions |
| 14 | No stratification by climate/soil/vegetation | Missing IPCC climate zone and soil type classification |

## 5. Architecture (7 Engines)

| Engine | Class | File | Purpose |
|--------|-------|------|---------|
| 1 | LandUseDatabaseEngine | land_use_database.py | Land categories, carbon stock factors, IPCC default values, climate/soil zones |
| 2 | CarbonStockCalculatorEngine | carbon_stock_calculator.py | Stock-difference and gain-loss methods for 5 carbon pools |
| 3 | LandUseChangeTrackerEngine | land_use_change_tracker.py | 6×6 transition matrix, conversion tracking, 20-year transitions |
| 4 | SoilOrganicCarbonEngine | soil_organic_carbon.py | IPCC SOC reference stocks, land use/management/input factors |
| 5 | UncertaintyQuantifierEngine | uncertainty_quantifier.py | Monte Carlo simulation, error propagation, DQI scoring |
| 6 | ComplianceCheckerEngine | compliance_checker.py | GHG Protocol, IPCC, CSRD, SBTi FLAG, EU LULUCF validation |
| 7 | LandUsePipelineEngine | land_use_pipeline.py | 8-stage pipeline orchestration |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | SDK facade with graceful imports |
| `config.py` | GL_LAND_USE_ env prefix, thread-safe singleton |
| `models.py` | Enums, Pydantic v2 models, IPCC constant tables |
| `metrics.py` | 12 Prometheus metrics with gl_lu_ prefix |
| `provenance.py` | SHA-256 chain-hashed audit trails |
| `setup.py` | LandUseEmissionsService facade |
| `api/router.py` | 20 REST endpoints at /api/v1/land-use-emissions |

## 6. Key Features

### 6.1 Land-Use Categories (6 IPCC Categories)

Per IPCC 2006 Guidelines Volume 4, Chapter 3:

1. **Forest Land** (FL)
   - Managed forests (timber plantations, community forests, protected forests)
   - Natural/semi-natural forests (primary, secondary, degraded)
   - Sub-categories: tropical, subtropical, temperate, boreal
   - Carbon density ranges: 20-400 tC/ha (above-ground biomass)

2. **Cropland** (CL)
   - Annual crops (cereals, oilseeds, pulses, vegetables)
   - Perennial crops (orchards, vineyards, oil palm, rubber)
   - Agroforestry systems
   - Fallow land (rotational, abandoned)

3. **Grassland** (GL)
   - Managed pastures and rangelands
   - Natural grasslands and savannas
   - Shrublands and tundra
   - Improved vs. degraded grassland

4. **Wetlands** (WL)
   - Peatlands (tropical, temperate, boreal)
   - Mangroves and tidal marshes
   - Freshwater swamps and floodplains
   - Constructed wetlands
   - Managed: drained, rewetted, extraction sites

5. **Settlements** (SL)
   - Urban areas (parks, gardens, street trees)
   - Rural settlements
   - Infrastructure corridors (roads, railways, pipelines)

6. **Other Land** (OL)
   - Bare rock, ice, sand dunes
   - Unmanaged land
   - Land not classified elsewhere

### 6.2 Carbon Pools (5 IPCC Pools)

Per IPCC 2006 Guidelines Volume 4, Chapter 2:

1. **Above-Ground Biomass (AGB)**
   - Living trees, shrubs, herbs, crops
   - Measured: forest inventories, allometric equations, remote sensing
   - Default: IPCC Table 4.7/4.8 (tC/ha by biome)

2. **Below-Ground Biomass (BGB)**
   - Root systems, tubers, rhizomes
   - Estimated via root-to-shoot ratios (R:S)
   - Default R:S: 0.20-0.56 depending on biome/AGB class

3. **Dead Wood (DW)**
   - Standing dead trees, logs, stumps, coarse woody debris (>10cm)
   - Default: fraction of AGB (0.01-0.25 by biome/disturbance)
   - Turnover rates by climate zone

4. **Litter (LT)**
   - Non-living organic matter on/near soil surface (<10cm diameter)
   - Leaf litter, fine woody debris, bark, seeds
   - Default: 1.5-40 tC/ha by biome

5. **Soil Organic Carbon (SOC)**
   - Organic carbon in mineral soils to 30cm depth (Tier 1)
   - Extendable to 100cm for Tier 2/3
   - IPCC reference stocks by climate zone and soil type
   - Modified by land use, management, and input factors

### 6.3 Calculation Methods (3 Tiers)

**Tier 1 (Default Factors)**
- IPCC default carbon stock values by biome and climate zone
- Global default emission factors for land-use conversions
- Default SOC reference stocks and modification factors
- Suitable for screening-level assessments

**Tier 2 (Country-Specific)**
- National/regional carbon stock data
- Country-specific emission factors and growth rates
- Stratified by ecological zone, soil type, management practice
- National forest inventory data integration

**Tier 3 (Model-Based)**
- Process-based models (e.g., CENTURY, RothC, CBM-CFS3 approaches)
- Spatially explicit modeling with GIS integration
- Annual carbon flux tracking
- Remote sensing data integration (NDVI, biomass maps)
- Validation against field measurements

### 6.4 Stock-Difference Method

```
ΔC = (C_t2 - C_t1) / (t2 - t1)
```
Where:
- ΔC = annual carbon stock change (tC/yr)
- C_t1 = carbon stock at time t1 (tC)
- C_t2 = carbon stock at time t2 (tC)
- Applied to each of the 5 carbon pools independently

### 6.5 Gain-Loss Method (for AGB/BGB)

```
ΔC = ΔC_gain - ΔC_loss
ΔC_gain = A × G_w × CF
ΔC_loss = (L_wood_removals + L_fuelwood + L_disturbance) × CF
```
Where:
- A = area of land category (ha)
- G_w = average annual above-ground biomass growth (t d.m./ha/yr)
- CF = carbon fraction of dry matter (default 0.47)
- L = biomass losses from harvesting, fuelwood, and disturbance

### 6.6 Soil Organic Carbon Method (IPCC Tier 1)

```
SOC = SOC_ref × F_LU × F_MG × F_I
ΔSOC = (SOC_0 - SOC_prev) / D
```
Where:
- SOC_ref = reference SOC stock for climate zone and soil type (tC/ha)
- F_LU = land-use factor (e.g., 1.0 for forest, 0.69 for long-term cultivation)
- F_MG = management factor (e.g., 1.0 for full tillage, 1.08 for no-till)
- F_I = input factor (e.g., 1.0 for medium, 1.11 for high with manure)
- D = time period for transition (default 20 years)

### 6.7 Land-Use Conversion Emissions

For conversions between categories (e.g., forest → cropland):

```
ΔC_conversion = ΔC_biomass + ΔC_DOM + ΔC_SOC
E_fire = M_fire × C_f × G_ef    (if burning involved)
E_total = -ΔC_conversion × (44/12) + E_fire + E_N2O
```

Where:
- ΔC_biomass = difference in equilibrium biomass stocks
- ΔC_DOM = difference in dead organic matter stocks
- ΔC_SOC = soil carbon change over 20-year transition
- M_fire = mass of fuel burned (dry matter)
- C_f = combustion factor
- G_ef = emission factor per gas (CO2, CH4, N2O, CO, NOx)

### 6.8 Peatland-Specific Emissions

```
E_CO2 = A_drained × EF_CO2_drain   (on-site CO2 from oxidation)
E_CH4 = A_drained × EF_CH4_drain   (ditches and drains)
E_DOC = A_drained × EF_DOC         (dissolved organic carbon export)
E_fire_peat = A_burned × D_burn × BD × EF_combustion
```

IPCC Wetlands Supplement (2013) emission factors:
- Tropical drained peatland: 5.3-73 tCO2/ha/yr (depending on use)
- Temperate drained peatland: 1.7-10.3 tCO2/ha/yr
- Boreal drained peatland: 0.25-6.1 tCO2/ha/yr
- Rewetted peatland: -0.5 to 2.4 tCO2/ha/yr (net may be removal)

### 6.9 Non-CO2 Emissions from Managed Land

**N2O from Soil Management:**
```
N2O_direct = (F_SN + F_ON + F_CR + F_SOM) × EF_1 × (44/28)
N2O_indirect = N2O_ATD + N2O_leaching
```
Where:
- F_SN = synthetic fertilizer nitrogen input
- F_ON = organic fertilizer nitrogen input
- F_CR = crop residue nitrogen returned to soil
- F_SOM = nitrogen from soil organic matter mineralization
- EF_1 = IPCC default 0.01 kg N2O-N / kg N input (Tier 1)

**CH4 from Wetlands/Peatlands:**
```
CH4 = A_managed × EF_CH4_wetland
```

### 6.10 Emission Gases (4 primary)
- **CO2**: From carbon stock changes in all 5 pools, liming, urea
- **CH4**: From wetlands, peatlands, biomass burning, rice paddies
- **N2O**: From soil management, fertilization, biomass burning, peat fires
- **CO**: From biomass burning (trace gas, converted to CO2 equivalent)

### 6.11 IPCC Climate Zones (12)
1. Tropical Wet (>2000mm, no dry season)
2. Tropical Moist (1000-2000mm, short dry)
3. Tropical Dry (<1000mm, extended dry)
4. Tropical Montane (>1000m elevation)
5. Warm Temperate Moist (MAT 10-20°C, >1000mm)
6. Warm Temperate Dry (MAT 10-20°C, <1000mm)
7. Cool Temperate Moist (MAT 0-10°C, >1000mm)
8. Cool Temperate Dry (MAT 0-10°C, <1000mm)
9. Boreal Moist (MAT <0°C, >400mm)
10. Boreal Dry (MAT <0°C, <400mm)
11. Polar Moist
12. Polar Dry

### 6.12 IPCC Soil Types (7)
1. High-Activity Clay (HAC) - Vertisols, Mollisols, high CEC
2. Low-Activity Clay (LAC) - Oxisols, Ultisols, Ferralsols
3. Sandy Soils - Arenosols, Psamments (>70% sand)
4. Spodic Soils - Podzols, Spodosols (acidic, leached)
5. Volcanic Soils - Andisols, Andosols (high SOC)
6. Wetland Soils - Histosols, organic soils (>20% organic C)
7. Other Soils - Not classified elsewhere

### 6.13 Regulatory Frameworks (7)

1. **IPCC 2006 Guidelines Volume 4** (AFOLU) - Primary methodology
2. **IPCC 2019 Refinement** - Updated emission factors and methods
3. **GHG Protocol Land Sector & Removals Guidance (2022)** - Corporate reporting
4. **ISO 14064-1:2018** - Category 1 direct emissions
5. **CSRD/ESRS E1** - E1-6 GHG emissions, E1-7 GHG removals
6. **EU LULUCF Regulation 2018/841** (amended 2023/839) - EU member state reporting
7. **SBTi FLAG Guidance** - Science-based targets for land-intensive sectors

### 6.14 Emission Factor Sources (6)
1. IPCC 2006 Guidelines Volume 4 default tables
2. IPCC 2019 Refinement updated tables
3. IPCC Wetlands Supplement (2013) for peatlands
4. National GHG Inventory data (country-specific Tier 2)
5. Published literature / peer-reviewed studies
6. Custom / facility-specific measurements

## 7. Database Schema (V057)

### Schema: land_use_emissions_service

### Tables (10)

| Table | Purpose |
|-------|---------|
| lu_land_parcels | Land parcel registry with area, location, climate zone, soil type |
| lu_carbon_stock_factors | IPCC default carbon stock values by biome, pool, and category |
| lu_emission_factors | Emission factors by conversion type, climate zone, method |
| lu_land_use_transitions | Land-use change records (from_category → to_category with dates) |
| lu_carbon_stock_snapshots | Point-in-time carbon stock measurements per parcel per pool |
| lu_calculations | Individual emission/removal calculation results |
| lu_calculation_details | Per-gas, per-pool emission breakdown |
| lu_soc_assessments | Soil organic carbon assessment records with factors |
| lu_compliance_records | Regulatory compliance check results |
| lu_audit_entries | Provenance and audit trail entries |

### Hypertables (3) - 7-day chunks

| Hypertable | Purpose |
|------------|---------|
| lu_calculation_events | Time-series calculation tracking |
| lu_transition_events | Time-series land-use change tracking |
| lu_compliance_events | Time-series compliance check tracking |

### Continuous Aggregates (2)

| Aggregate | Purpose |
|-----------|---------|
| lu_hourly_calculation_stats | Hourly calculation volume and emissions |
| lu_daily_emission_totals | Daily emission totals by land category and pool |

### Row-Level Security
- All tables: tenant_id-based RLS policies
- Read: `gl_readonly`, `gl_service`, `gl_admin`
- Write: `gl_service`, `gl_admin`
- Delete: `gl_admin` only
- Agent registry: `gl_agent_service`

## 8. API Endpoints (20)

### Base Path: `/api/v1/land-use-emissions`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/calculations` | Execute single land use emission calculation |
| POST | `/calculations/batch` | Execute batch calculations (max 10,000) |
| GET | `/calculations/{id}` | Get calculation by ID |
| GET | `/calculations` | List calculations with filters |
| DELETE | `/calculations/{id}` | Delete calculation |
| POST | `/carbon-stocks` | Record carbon stock snapshot |
| GET | `/carbon-stocks/{parcel_id}` | Get carbon stock history for parcel |
| GET | `/carbon-stocks/{parcel_id}/summary` | Summarize stocks across all pools |
| POST | `/land-parcels` | Register land parcel |
| GET | `/land-parcels` | List land parcels |
| PUT | `/land-parcels/{id}` | Update land parcel metadata |
| POST | `/transitions` | Record land-use transition |
| GET | `/transitions` | List transitions with filters |
| GET | `/transitions/matrix` | Get 6x6 transition matrix summary |
| POST | `/soc-assessments` | Record SOC assessment |
| GET | `/soc-assessments/{parcel_id}` | Get SOC assessment history |
| POST | `/compliance/check` | Run compliance check |
| GET | `/compliance/{id}` | Get compliance result |
| POST | `/uncertainty` | Run Monte Carlo uncertainty analysis |
| GET | `/aggregations` | Get aggregated emissions by category/pool/period |

## 9. Auth Integration

### Permission Map Entries

| Route Pattern | Permission |
|--------------|------------|
| `POST /calculations` | `land-use:calculate` |
| `POST /calculations/batch` | `land-use:calculate` |
| `GET /calculations/*` | `land-use:read` |
| `DELETE /calculations/*` | `land-use:delete` |
| `POST /carbon-stocks` | `land-use:carbon-stocks:write` |
| `GET /carbon-stocks/*` | `land-use:carbon-stocks:read` |
| `POST /land-parcels` | `land-use:parcels:write` |
| `GET /land-parcels` | `land-use:parcels:read` |
| `PUT /land-parcels/*` | `land-use:parcels:write` |
| `POST /transitions` | `land-use:transitions:write` |
| `GET /transitions/*` | `land-use:transitions:read` |
| `POST /soc-assessments` | `land-use:soc:write` |
| `GET /soc-assessments/*` | `land-use:soc:read` |
| `POST /compliance/check` | `land-use:compliance:check` |
| `GET /compliance/*` | `land-use:compliance:read` |
| `POST /uncertainty` | `land-use:uncertainty:run` |
| `GET /aggregations` | `land-use:read` |

## 10. Testing Requirements

### Target: 1,000+ tests across 13 test files

| Test File | Covers | Target Tests |
|-----------|--------|-------------|
| test_models.py | Enums, Pydantic models, constants | 120 |
| test_config.py | Configuration, env vars, validation | 40 |
| test_metrics.py | Prometheus metrics registration | 25 |
| test_provenance.py | SHA-256 chain hashing, audit trails | 45 |
| test_land_use_database.py | Engine 1: Land categories, carbon factors | 110 |
| test_carbon_stock_calculator.py | Engine 2: Stock-difference, gain-loss | 130 |
| test_land_use_change_tracker.py | Engine 3: Transition matrix, conversions | 120 |
| test_soil_organic_carbon.py | Engine 4: SOC stocks, factors | 110 |
| test_uncertainty_quantifier.py | Engine 5: Monte Carlo, DQI | 85 |
| test_compliance_checker.py | Engine 6: 7 frameworks, requirements | 100 |
| test_land_use_pipeline.py | Engine 7: 8-stage orchestration | 90 |
| test_setup.py | Service facade, lifecycle | 30 |
| test_api.py | 20 REST endpoints | 95 |

### Test Categories
- **Unit tests**: Each engine method independently
- **Integration tests**: Cross-engine data flow
- **Regulatory tests**: Known IPCC examples with expected results
- **Edge cases**: Zero areas, negative stocks, extreme climate zones
- **Determinism**: Decimal arithmetic, reproducible results

## 11. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | All 6 IPCC land categories with subcategories | Database engine returns correct factors |
| 2 | All 5 carbon pools tracked independently | Calculator produces per-pool results |
| 3 | Stock-difference method matches IPCC examples | Test against Table 4.7/4.8 examples |
| 4 | Gain-loss method matches IPCC examples | Test annual increment/loss calculations |
| 5 | SOC calculation with 3 modification factors | Matches IPCC Table 5.5/5.10 examples |
| 6 | 6×6 transition matrix tracks all conversions | Tracker correctly classifies transitions |
| 7 | 20-year transition period for SOC changes | Linear interpolation over default period |
| 8 | Peatland emissions per Wetlands Supplement | CO2/CH4/DOC matches supplement factors |
| 9 | N2O from soil management per IPCC Ch 11 | Direct and indirect N2O calculations |
| 10 | Fire emissions per IPCC Ch 2 Table 2.4/2.5 | Combustion factors and gas EFs correct |
| 11 | Monte Carlo with configurable iterations | 95% CI within expected ranges |
| 12 | All 7 regulatory frameworks validated | Compliance checker covers all requirements |
| 13 | 1,000+ tests passing | pytest --tb=short returns 0 |
| 14 | Deterministic Decimal arithmetic | Hash-identical results across runs |
| 15 | Auth integration complete | All 17 permission entries in PERMISSION_MAP |

## 12. Dependencies

### Internal Dependencies
- `greenlang/infrastructure/auth_service/` - JWT/RBAC integration
- `greenlang/observability/prometheus/` - Metrics export
- `greenlang/observability/tracing/` - OpenTelemetry spans
- `greenlang/agents/foundational/provenance/` - Audit trail patterns
- `greenlang/agents/foundational/uncertainty/` - Monte Carlo patterns

### External Dependencies
- `pydantic>=2.0` - Data models
- `fastapi>=0.100` - REST API
- `prometheus_client>=0.17` - Metrics
- `numpy>=1.24` - Monte Carlo sampling (optional, fallback to random)

## 13. Glossary

| Term | Definition |
|------|-----------|
| AFOLU | Agriculture, Forestry and Other Land Use |
| AGB | Above-Ground Biomass |
| BGB | Below-Ground Biomass |
| CF | Carbon Fraction (of dry matter, default 0.47) |
| DOM | Dead Organic Matter (dead wood + litter) |
| DW | Dead Wood |
| FLAG | Forest, Land and Agriculture (SBTi guidance) |
| LT | Litter |
| LULUCF | Land Use, Land-Use Change and Forestry |
| R:S | Root-to-Shoot ratio |
| SOC | Soil Organic Carbon |
| SOC_ref | Reference SOC stock for native vegetation |
| F_LU | Land-Use factor for SOC modification |
| F_MG | Management factor for SOC modification |
| F_I | Input factor for SOC modification |
