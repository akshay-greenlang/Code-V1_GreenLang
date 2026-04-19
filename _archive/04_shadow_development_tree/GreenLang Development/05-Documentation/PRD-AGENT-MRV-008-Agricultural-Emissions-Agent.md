# PRD: AGENT-MRV-008 Agricultural Emissions Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-X-008 |
| **Internal Label** | AGENT-MRV-008 |
| **Category** | Layer 3 - MRV / Accounting Agents (Scope 1) |
| **Package** | `greenlang.agricultural_emissions` |
| **DB Migration** | V059 |
| **Priority** | Scope 1 - Direct Emissions |
| **Regulatory Drivers** | IPCC 2006 Vol 4 (AFOLU), IPCC 2019 Refinement, GHG Protocol Agricultural Guidance, ISO 14064-1, CSRD/ESRS E1, EPA 40 CFR 98 Subpart JJ, DEFRA Guidelines |
| **Dimensions** | Crop x Geography (6,000 estimated variants) |
| **Gases** | CO2, CH4, N2O |
| **IPCC Chapters** | Vol 4 Ch 10 (Livestock & Manure), Vol 4 Ch 11 (Soils & Fertilizer), Vol 4 Ch 5.5 (Rice Cultivation) |

## 2. Problem Statement

Agricultural activities are responsible for approximately 10-14% of global anthropogenic GHG emissions, with livestock (enteric fermentation and manure management) contributing ~5.8 Gt CO2e/yr and cropland management (fertilizers, rice cultivation, liming) contributing ~3.5 Gt CO2e/yr. Organizations with agricultural operations must accurately quantify these emissions for regulatory compliance under CSRD, GHG Protocol, and national reporting frameworks.

The Agricultural Emissions Agent must accurately quantify:

1. **Enteric fermentation CH4** from livestock digestive processes (cattle, buffalo, sheep, goats, swine, horses, etc.)
2. **Manure management CH4** from anaerobic decomposition in storage systems (lagoons, pits, solid storage, etc.)
3. **Manure management N2O** from nitrification/denitrification during storage and treatment
4. **Direct soil N2O** from synthetic fertilizers, organic amendments, crop residues, and organic soils
5. **Indirect soil N2O** from atmospheric deposition of volatilized NH3/NOx and leaching/runoff of NO3-
6. **CO2 from liming** (limestone and dolomite application to agricultural soils)
7. **CO2 from urea** application and other carbon-containing fertilizers
8. **Rice cultivation CH4** from anaerobic decomposition in flooded paddies
9. **Field burning CH4 and N2O** from burning of crop residues

## 3. Existing Layer 1 Capabilities

Existing preliminary implementations in `greenlang/agents/mrv/agriculture/`:

- `base.py` (504 lines) - Base agriculture MRV agent class
- `livestock.py` (694 lines) - Basic livestock calculations
- `crop_production.py` (431 lines) - Crop production stubs
- `fertilizer.py` (200 lines) - Basic fertilizer N2O
- `rice_cultivation.py` (172 lines) - Rice CH4 stubs
- `land_use_change.py` (179 lines) - Land use overlap
- `agricultural_machinery.py` (162 lines) - Farm equipment (Scope 1 mobile, covered by MRV-003)
- `irrigation.py` (209 lines) - Irrigation energy (Scope 2, not Scope 1)
- `food_processing.py` (223 lines) - Food processing (Scope 1 stationary, covered by MRV-001)

**Gaps**: No 7-engine architecture, no database migration, no API endpoints, no compliance checking, no uncertainty quantification, no provenance tracking, no Prometheus metrics, no test suite.

## 4. Gaps Requiring Layer 3 Production Implementation

| # | Gap | Impact |
|---|-----|--------|
| 1 | No Tier 2 enteric fermentation (GE-based calculation) | Cannot use country-specific or farm-level feed data |
| 2 | No AWMS-specific manure CH4 calculation | Cannot differentiate 15+ manure management systems |
| 3 | No manure N2O by AWMS type | Missing N2O pathway from manure management |
| 4 | No indirect N2O (volatilization + leaching) | Underestimates soil emissions by 30-40% |
| 5 | No liming/urea CO2 calculations | Missing significant CO2 source in agricultural soils |
| 6 | No rice cultivation CH4 with IPCC scaling factors | Cannot model water regime, organic amendments, soil type effects |
| 7 | No field burning calculations | Missing CH4/N2O from crop residue burning |
| 8 | No multi-framework compliance checking | Cannot verify IPCC/GHG Protocol/CSRD compliance |
| 9 | No Monte Carlo uncertainty quantification | Cannot provide confidence intervals on estimates |
| 10 | No provenance/audit trail | Cannot demonstrate calculation reproducibility |
| 11 | No database schema for agricultural data | No persistent storage for farms, herds, crops, calculations |
| 12 | No REST API endpoints | No programmatic access for frontend/integrations |

## 5. Architecture (7 Engines)

### Engine Architecture

| Engine | Class | File | Purpose |
|--------|-------|------|---------|
| 1 | `AgriculturalDatabaseEngine` | `agricultural_database.py` | IPCC emission factor database, animal parameters, soil/crop reference data |
| 2 | `EntericFermentationEngine` | `enteric_fermentation.py` | Livestock CH4 from enteric fermentation (Tier 1/2/3) |
| 3 | `ManureManagementEngine` | `manure_management.py` | Manure CH4 + N2O from 15+ AWMS types |
| 4 | `CroplandEmissionsEngine` | `cropland_emissions.py` | Soil N2O, rice CH4, liming/urea CO2, field burning |
| 5 | `UncertaintyQuantifierEngine` | `uncertainty_quantifier.py` | Monte Carlo uncertainty with ag-specific parameter distributions |
| 6 | `ComplianceCheckerEngine` | `compliance_checker.py` | 7 regulatory framework compliance checking |
| 7 | `AgriculturalPipelineEngine` | `agricultural_pipeline.py` | 8-stage pipeline orchestrating all engines |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | SDK facade with graceful engine imports |
| `models.py` | Pydantic v2 models, enums, IPCC constant tables |
| `config.py` | `GL_AGRICULTURAL_` env prefix, 80+ config parameters |
| `metrics.py` | 12 Prometheus metrics with `gl_ag_` prefix |
| `provenance.py` | SHA-256 chain-hashed audit trail |
| `setup.py` | `AgriculturalEmissionsService` facade |
| `api/router.py` | 20 FastAPI REST endpoints |
| `api/__init__.py` | API package marker |

## 6. Key Features

### 6.1 Animal Types (20 livestock categories)

| # | Animal Type | Typical Enteric EF (kg CH4/head/yr) | Key Parameters |
|---|-------------|--------------------------------------|----------------|
| 1 | Dairy cattle | 128 (developed), 68 (developing) | Milk yield, fat %, feed digestibility |
| 2 | Non-dairy cattle | 53 (developed), 47 (developing) | Body weight, weight gain, activity |
| 3 | Buffalo | 55 | Milk yield, feed quality |
| 4 | Sheep | 8 | Fleece weight, feed quality |
| 5 | Goats | 5 | Milk yield, body weight |
| 6 | Camels | 46 | Feed quality, body weight |
| 7 | Horses | 18 | Body weight, workload |
| 8 | Mules/Asses | 10 | Body weight |
| 9 | Swine (market) | 1.5 | Feed intake, body weight |
| 10 | Swine (breeding) | 1.5 | Parity, body weight |
| 11 | Poultry - layers | Negligible | N excretion for manure only |
| 12 | Poultry - broilers | Negligible | N excretion for manure only |
| 13 | Turkeys | Negligible | N excretion for manure only |
| 14 | Ducks | Negligible | N excretion for manure only |
| 15 | Deer | 20 | Body weight |
| 16 | Elk | 25 | Body weight |
| 17 | Alpacas/Llamas | 8 | Body weight |
| 18 | Rabbits | 0.3 | Body weight |
| 19 | Fur-bearing animals | 0.3 | Body weight |
| 20 | Other livestock | Variable | User-defined EF |

### 6.2 Enteric Fermentation Methodology

**Tier 1 (IPCC 2006 Vol 4 Eq 10.19):**
```
CH4_enteric = SUM_over_T[ EF_T * N_T * 10^-6 ]  (Gg CH4/yr)

Where:
  EF_T = emission factor for animal type T (kg CH4/head/yr) [IPCC Table 10.11]
  N_T  = population of animal type T (heads)
```

**Tier 2 (IPCC 2006 Vol 4 Eq 10.21):**
```
EF = (GE * Ym/100 * 365) / 55.65  (kg CH4/head/yr)

Where:
  GE  = gross energy intake (MJ/head/day)
  Ym  = methane conversion factor (% of GE) [3-7.5% typical]
  55.65 = energy content of CH4 (MJ/kg)

GE = [(NE_m + NE_a + NE_l + NE_work + NE_p) / REM + NE_g / REG] / (DE/100)

Where:
  NE_m    = net energy for maintenance = Cfi * BW^0.75 (MJ/day)
  NE_a    = net energy for activity = Ca * NE_m
  NE_l    = net energy for lactation = Milk * (1.47 + 0.40 * Fat) (MJ/day)
  NE_work = net energy for work (draught animals)
  NE_p    = net energy for pregnancy = 0.10 * NE_m
  NE_g    = net energy for growth = 22.02 * (BW/(C*BW_mature))^0.75 * WG^1.097
  REM     = ratio of NE for maintenance to DE
  REG     = ratio of NE for growth to DE
  DE      = digestible energy (% of GE) [55-85%]
  Cfi     = maintenance coefficient (0.322-0.386 for cattle)
  BW      = body weight (kg)
  Ca      = activity coefficient (0.0-0.36)
```

### 6.3 Manure Management Systems (15 AWMS types)

| # | AWMS Type | CH4 MCF Range | N2O EF (kg N2O-N/kg N) |
|---|-----------|--------------|------------------------|
| 1 | Pasture/Range/Paddock | 0.001-0.02 | 0.02 |
| 2 | Daily spread | 0.001-0.005 | 0.0 |
| 3 | Solid storage | 0.02-0.05 | 0.005 |
| 4 | Dry lot | 0.01-0.05 | 0.02 |
| 5 | Liquid/Slurry (no crust) | 0.10-0.80 | 0.0 |
| 6 | Liquid/Slurry (with crust) | 0.10-0.40 | 0.005 |
| 7 | Uncovered anaerobic lagoon | 0.66-0.80 | 0.0 |
| 8 | Covered anaerobic lagoon | 0.0-0.10 | 0.0 |
| 9 | Pit storage (<1 month) | 0.03-0.30 | 0.002 |
| 10 | Pit storage (>1 month) | 0.10-0.80 | 0.002 |
| 11 | Deep bedding (no mixing) | 0.17-0.50 | 0.01 |
| 12 | Deep bedding (active mixing) | 0.17-0.50 | 0.07 |
| 13 | Composting - static pile | 0.005-0.01 | 0.006 |
| 14 | Composting - intensive | 0.005-0.01 | 0.10 |
| 15 | Anaerobic digester | 0.0-0.10 | 0.0 |

**Manure CH4 (IPCC 2006 Vol 4 Eq 10.22-10.23):**
```
CH4_manure = SUM_T[ (EF_T * N_T) ] * 10^-6  (Gg CH4/yr)

EF_T = VS_T * 365 * Bo_T * 0.67 * SUM_S[ MCF_S * MS_T,S ]

Where:
  VS_T    = daily volatile solids excreted (kg VS/head/day) [IPCC Table 10A-4]
  Bo_T    = max CH4 producing capacity (m3 CH4/kg VS) [IPCC Table 10A-7]
  0.67    = conversion factor (kg CH4/m3 CH4 at STP)
  MCF_S   = methane conversion factor for AWMS S [IPCC Table 10.17]
  MS_T,S  = fraction of manure handled by AWMS S for animal T
```

**Manure N2O (IPCC 2006 Vol 4 Eq 10.25):**
```
N2O_manure = SUM_S[ SUM_T[ (N_T * Nex_T * MS_T,S) ] * EF3_S ] * 44/28

Where:
  Nex_T  = annual N excretion rate (kg N/head/yr) [IPCC Table 10.19]
  MS_T,S = fraction of manure in AWMS S for animal T
  EF3_S  = N2O emission factor for AWMS S (kg N2O-N/kg N) [IPCC Table 10.21]
  44/28  = molecular weight conversion N2O-N to N2O
```

### 6.4 Agricultural Soils - N2O Emissions

**Direct N2O (IPCC 2006 Vol 4 Eq 11.1):**
```
N2O_direct = [(F_SN + F_ON + F_CR + F_SOM) * EF1 + F_OS_CG * EF2_CG + F_OS_F * EF2_F + F_PRP * EF3_PRP] * 44/28

Where:
  F_SN     = synthetic fertilizer N applied (kg N/yr) - EF1 = 0.01 (1%)
  F_ON     = organic N applied (manure, compost, sewage) - EF1 = 0.01
  F_CR     = crop residue N returned to soil - EF1 = 0.01
  F_SOM    = N from soil organic matter mineralization - EF1 = 0.01
  F_OS_CG  = area of drained/managed organic soils (cropland/grassland)
  EF2_CG   = 8 kg N2O-N/ha/yr (cropland), 2.5 (grassland)
  F_PRP    = N from pasture/range/paddock deposits
  EF3_PRP  = 0.02 (cattle/poultry), 0.01 (other)
```

**Indirect N2O from Volatilization (IPCC 2006 Vol 4 Eq 11.9):**
```
N2O_ATD = [(F_SN * Frac_GASF) + (F_ON + F_PRP * Frac_GASM)] * EF4 * 44/28

Where:
  Frac_GASF = 0.10 (fraction of synthetic N volatilized as NH3/NOx)
  Frac_GASM = 0.20 (fraction of organic/PRP N volatilized)
  EF4       = 0.01 (kg N2O-N per kg NH3-N + NOx-N volatilized)
```

**Indirect N2O from Leaching (IPCC 2006 Vol 4 Eq 11.10):**
```
N2O_L = (F_SN + F_ON + F_PRP + F_CR + F_SOM) * Frac_LEACH * EF5 * 44/28

Where:
  Frac_LEACH = 0.30 (fraction of N leached/runoff)
  EF5        = 0.0075 (kg N2O-N per kg N leached)
```

### 6.5 Liming and Urea CO2

**Liming (IPCC 2006 Vol 4 Eq 11.12):**
```
CO2_liming = (M_limestone * EF_limestone + M_dolomite * EF_dolomite) * 44/12

Where:
  M_limestone   = mass of limestone (CaCO3) applied (tonnes)
  EF_limestone  = 0.12 (tonne C/tonne limestone)
  M_dolomite    = mass of dolomite (CaMg(CO3)2) applied (tonnes)
  EF_dolomite   = 0.13 (tonne C/tonne dolomite)
  44/12         = molecular weight conversion C to CO2
```

**Urea (IPCC 2006 Vol 4 Eq 11.13):**
```
CO2_urea = M_urea * EF_urea * 44/12

Where:
  M_urea  = mass of urea CO(NH2)2 applied (tonnes)
  EF_urea = 0.20 (tonne C/tonne urea)
```

### 6.6 Rice Cultivation CH4

**IPCC 2006 Vol 4 Eq 5.1:**
```
CH4_rice = SUM_i,j,k[ EF_ijk * t_ijk * A_ijk * 10^-6 ]  (Gg CH4/yr)

Where:
  EF_ijk = daily emission factor (kg CH4/ha/day) for conditions i,j,k
  t_ijk  = cultivation period (days)
  A_ijk  = harvested area (ha)

EF_ijk = EF_c * SF_w * SF_p * SF_o * SF_s * SF_r

Where:
  EF_c  = baseline emission factor = 1.30 kg CH4/ha/day (IPCC default)
  SF_w  = scaling factor for water regime during cultivation
  SF_p  = scaling factor for pre-season water regime
  SF_o  = scaling factor for organic amendments = (1 + SUM[ROA_i * CFOA_i])^0.59
  SF_s  = scaling factor for soil type (if available)
  SF_r  = scaling factor for rice cultivar (if available)
```

**Water Regime Scaling Factors (SF_w):**

| Water Regime | SF_w |
|-------------|------|
| Continuously flooded | 1.0 |
| Intermittent - single aeration | 0.60 |
| Intermittent - multiple aeration | 0.52 |
| Rainfed - regular | 0.28 |
| Rainfed - drought prone | 0.25 |
| Deep water (>100 cm) | 0.31 |
| Upland (never flooded) | 0.0 |

**Pre-Season Flooding Scaling Factors (SF_p):**

| Pre-Season Condition | SF_p (short) | SF_p (long) |
|---------------------|-------------|------------|
| Not flooded (<180 days) | 1.00 | 1.00 |
| Flooded (>180 days) | 1.90 | 2.41 |
| Not known | 1.22 | 1.22 |

**Organic Amendment Correction Factors (CFOA):**

| Amendment Type | CFOA |
|---------------|------|
| Straw (incorporated shortly before) | 1.0 |
| Straw (incorporated long before) | 0.29 |
| Compost | 0.05 |
| Farm yard manure | 0.14 |
| Green manure | 0.50 |

### 6.7 Field Burning of Agricultural Residues

**IPCC 2006 Vol 4 Eq 2.27:**
```
L_fire = A * M_B * C_f * G_ef * 10^-3  (tonnes gas/yr)

Where:
  A    = area burned (ha)
  M_B  = mass of fuel (dry matter) per unit area (tonnes dm/ha)
  C_f  = combustion factor (fraction of biomass actually burned)
  G_ef = emission factor for gas (g gas/kg dm burned)
```

**Crop Residue Emission Factors (g/kg dry matter burned):**

| Crop Type | CH4 EF | N2O EF | Residue:Product Ratio | Dry Matter (fraction) |
|-----------|--------|--------|----------------------|----------------------|
| Wheat | 2.7 | 0.07 | 1.3 | 0.88 |
| Corn/Maize | 2.7 | 0.07 | 1.0 | 0.87 |
| Rice | 2.7 | 0.07 | 1.4 | 0.86 |
| Sugarcane | 2.7 | 0.07 | 0.3 | 0.25 |
| Cotton | 2.7 | 0.07 | 2.1 | 0.90 |
| Soybean | 2.7 | 0.07 | 1.5 | 0.87 |
| Other cereals | 2.7 | 0.07 | 1.2 | 0.85 |

### 6.8 GWP Values

| Gas | AR4 | AR5 | AR6 | AR6 (20-yr) |
|-----|-----|-----|-----|-------------|
| CO2 | 1 | 1 | 1 | 1 |
| CH4 (fossil) | 25 | 28 | 29.8 | 82.5 |
| CH4 (biogenic) | 25 | 28 | 27.0 | 79.7 |
| N2O | 298 | 265 | 273 | 273 |

### 6.9 Calculation Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | IPCC_TIER_1 | Default emission factors by animal/crop type |
| 2 | IPCC_TIER_2 | Country-specific data (GE-based enteric, VS-based manure) |
| 3 | IPCC_TIER_3 | Model-based (COWPOLL, DairyGEM, DAYCENT) |
| 4 | MASS_BALANCE | Nitrogen mass balance across farm system |
| 5 | DIRECT_MEASUREMENT | On-farm CH4/N2O measurement chambers |
| 6 | SPEND_BASED | Economic input-based estimation |

### 6.10 Regulatory Frameworks (7)

| # | Framework | Scope | Key Requirements |
|---|-----------|-------|-----------------|
| 1 | IPCC_2006_VOL4 | AFOLU Ch 10-11 | Tier 1/2/3, livestock + soils + rice |
| 2 | IPCC_2019 | Refinement to 2006 | Updated EFs, improved uncertainty methods |
| 3 | GHG_PROTOCOL | Agricultural Guidance | Scope 1 livestock/crops, Scope 3 supply chain |
| 4 | ISO_14064 | ISO 14064-1:2018 | Category 1, quantification approach |
| 5 | CSRD_ESRS | ESRS E1 + E4 | GHG emissions + biodiversity & ecosystems |
| 6 | EPA_40CFR98 | Subpart JJ | Manure management, enteric fermentation |
| 7 | DEFRA | UK Guidelines | DEFRA/BEIS conversion factors for agriculture |

## 7. Database Schema

**Schema:** `agricultural_emissions_service`

### Tables (12)

| # | Table | Purpose |
|---|-------|---------|
| 1 | `ag_farms` | Farm/facility registry with location, climate zone, type |
| 2 | `ag_livestock_populations` | Animal herds by type, count, body weight, productivity |
| 3 | `ag_manure_systems` | AWMS allocation per animal type per farm |
| 4 | `ag_emission_factors` | IPCC/EPA/DEFRA emission factor records |
| 5 | `ag_cropland_inputs` | Fertilizer, lime, urea, crop residue inputs |
| 6 | `ag_rice_fields` | Rice paddies with water regime, organic amendments |
| 7 | `ag_calculations` | Calculation results with provenance hash |
| 8 | `ag_calculation_details` | Per-gas per-source breakdown |
| 9 | `ag_field_burning_events` | Crop residue burning events |
| 10 | `ag_compliance_records` | Regulatory compliance check results |
| 11 | `ag_audit_entries` | SHA-256 chain-hashed provenance trail |
| 12 | `ag_feed_characteristics` | Feed composition for Tier 2 enteric (GE, DE%, Ym) |

### Hypertables (3)

| # | Hypertable | Chunk | Purpose |
|---|-----------|-------|---------|
| 1 | `ag_calculation_events` | 7 days | Calculation telemetry time-series |
| 2 | `ag_livestock_events_ts` | 7 days | Livestock population change events |
| 3 | `ag_compliance_events` | 7 days | Compliance check telemetry |

### Continuous Aggregates (2)

| # | Aggregate | Granularity |
|---|-----------|-------------|
| 1 | `ag_hourly_calculation_stats` | Hourly by emission source, calculation method |
| 2 | `ag_daily_emission_totals` | Daily by emission source, animal type/crop type |

### Row-Level Security

- All tables with `tenant_id` get RLS policies for `greenlang_app`, `greenlang_readonly`, `greenlang_admin`
- Reference tables (`ag_emission_factors`, `ag_feed_characteristics`) readable by all, writable by admin

### RBAC Permissions (20)

```
agricultural:calculate, agricultural:read, agricultural:delete
agricultural:farms:write, agricultural:farms:read
agricultural:livestock:write, agricultural:livestock:read
agricultural:manure:write, agricultural:manure:read
agricultural:cropland:write, agricultural:cropland:read
agricultural:rice:write, agricultural:rice:read
agricultural:burning:write, agricultural:burning:read
agricultural:compliance:check, agricultural:compliance:read
agricultural:uncertainty:run
agricultural:feed:write, agricultural:feed:read
```

## 8. API Endpoints (20)

**Base path:** `/api/v1/agricultural-emissions`

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/calculations` | Single agricultural emission calculation |
| 2 | POST | `/calculations/batch` | Batch calculation (up to 10,000) |
| 3 | GET | `/calculations` | List calculations (paginated) |
| 4 | GET | `/calculations/{calc_id}` | Get calculation by ID |
| 5 | DELETE | `/calculations/{calc_id}` | Delete calculation |
| 6 | POST | `/farms` | Register farm/facility |
| 7 | GET | `/farms` | List farms (paginated) |
| 8 | PUT | `/farms/{farm_id}` | Update farm |
| 9 | POST | `/livestock` | Register livestock population |
| 10 | GET | `/livestock` | List livestock by farm |
| 11 | PUT | `/livestock/{herd_id}` | Update livestock population |
| 12 | POST | `/cropland-inputs` | Register cropland inputs (fertilizer, lime, urea) |
| 13 | GET | `/cropland-inputs` | List cropland inputs |
| 14 | POST | `/rice-fields` | Register rice field |
| 15 | GET | `/rice-fields` | List rice fields |
| 16 | POST | `/compliance/check` | Run compliance check |
| 17 | GET | `/compliance/{check_id}` | Get compliance result |
| 18 | POST | `/uncertainty` | Run uncertainty analysis |
| 19 | GET | `/aggregations` | Get aggregated emissions |
| 20 | GET | `/health` | Service health check |

## 9. Auth Integration

| Method:Path | Permission |
|-------------|------------|
| `POST:/api/v1/agricultural-emissions/calculations` | `agricultural:calculate` |
| `POST:/api/v1/agricultural-emissions/calculations/batch` | `agricultural:calculate` |
| `GET:/api/v1/agricultural-emissions/calculations` | `agricultural:read` |
| `GET:/api/v1/agricultural-emissions/calculations/{calc_id}` | `agricultural:read` |
| `DELETE:/api/v1/agricultural-emissions/calculations/{calc_id}` | `agricultural:delete` |
| `POST:/api/v1/agricultural-emissions/farms` | `agricultural:farms:write` |
| `GET:/api/v1/agricultural-emissions/farms` | `agricultural:farms:read` |
| `PUT:/api/v1/agricultural-emissions/farms/{farm_id}` | `agricultural:farms:write` |
| `POST:/api/v1/agricultural-emissions/livestock` | `agricultural:livestock:write` |
| `GET:/api/v1/agricultural-emissions/livestock` | `agricultural:livestock:read` |
| `PUT:/api/v1/agricultural-emissions/livestock/{herd_id}` | `agricultural:livestock:write` |
| `POST:/api/v1/agricultural-emissions/cropland-inputs` | `agricultural:cropland:write` |
| `GET:/api/v1/agricultural-emissions/cropland-inputs` | `agricultural:cropland:read` |
| `POST:/api/v1/agricultural-emissions/rice-fields` | `agricultural:rice:write` |
| `GET:/api/v1/agricultural-emissions/rice-fields` | `agricultural:rice:read` |
| `POST:/api/v1/agricultural-emissions/compliance/check` | `agricultural:compliance:check` |
| `GET:/api/v1/agricultural-emissions/compliance/{check_id}` | `agricultural:compliance:read` |
| `POST:/api/v1/agricultural-emissions/uncertainty` | `agricultural:uncertainty:run` |
| `GET:/api/v1/agricultural-emissions/aggregations` | `agricultural:read` |
| `GET:/api/v1/agricultural-emissions/health` | (public) |

## 10. Testing Requirements

**Target:** 1,000+ tests across 15 test files

| # | Test File | Covers | Target Tests |
|---|-----------|--------|-------------|
| 1 | `conftest.py` | Shared fixtures | - |
| 2 | `test_models.py` | Enums, constants, Pydantic models | 150+ |
| 3 | `test_config.py` | Configuration, env vars, validation | 80+ |
| 4 | `test_metrics.py` | Prometheus metrics, collectors | 50+ |
| 5 | `test_agricultural_database.py` | Emission factors, animal params, reference data | 80+ |
| 6 | `test_enteric_fermentation.py` | Tier 1/2 enteric CH4 calculations | 90+ |
| 7 | `test_manure_management.py` | Manure CH4 + N2O, 15 AWMS types | 90+ |
| 8 | `test_cropland_emissions.py` | Soil N2O, liming CO2, urea CO2, rice CH4, burning | 100+ |
| 9 | `test_uncertainty_quantifier.py` | Monte Carlo, analytical, DQI | 60+ |
| 10 | `test_compliance_checker.py` | 7 framework compliance checking | 60+ |
| 11 | `test_agricultural_pipeline.py` | 8-stage pipeline orchestration | 60+ |
| 12 | `test_setup.py` | Service facade, engine access, CRUD | 50+ |
| 13 | `test_provenance.py` | Audit trail, chain hashing | 50+ |
| 14 | `test_api.py` | REST endpoints, request/response validation | 80+ |
| 15 | `__init__.py` | Package marker | - |

## 11. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | Enteric CH4 within 2% of IPCC Tier 1 example calculations | Compare against IPCC Vol 4 Table 10.11 worked examples |
| 2 | Manure CH4 correct for all 15 AWMS types | Verify VS x Bo x MCF x 0.67 for each system |
| 3 | Manure N2O correct with nitrogen balance | Verify Nex x MS x EF3 x 44/28 for each AWMS |
| 4 | Direct soil N2O correct with all 5 N input sources | Verify EF1 application to F_SN, F_ON, F_CR, F_SOM |
| 5 | Indirect N2O (vol + leaching) within 5% of hand calc | Verify Frac_GASF/GASM/LEACH and EF4/EF5 |
| 6 | Liming CO2 correct for limestone and dolomite | Verify EF x 44/12 conversion |
| 7 | Urea CO2 correct | Verify 0.20 tC/t urea x 44/12 |
| 8 | Rice CH4 correct with all scaling factors | Verify EF_c x SF_w x SF_p x SF_o x t x A |
| 9 | Field burning CH4/N2O correct per crop type | Verify A x MB x Cf x Gef formula |
| 10 | GWP conversion correct for AR4/AR5/AR6 | Verify gas x GWP x molecular weight factors |
| 11 | All calculations use Decimal arithmetic | No float in emission calculation paths |
| 12 | SHA-256 provenance on every calculation | 64-char hex hash, chain-hashed |
| 13 | 7 compliance frameworks checked | Each returns pass/fail with findings |
| 14 | Monte Carlo with reproducible seeds | Same seed produces identical CI bounds |
| 15 | 1000+ unit tests with 85%+ coverage | pytest with coverage report |

## 12. Dependencies

### Internal Dependencies
- `greenlang.infrastructure.auth_service` - Authentication and RBAC
- `AGENT-FOUND-001` (Orchestrator) - DAG execution
- `AGENT-FOUND-002` (Schema Compiler) - Input validation
- `AGENT-FOUND-003` (Unit Normalizer) - Unit conversion
- `AGENT-FOUND-004` (Assumptions Registry) - IPCC parameter assumptions
- `AGENT-FOUND-005` (Citations) - IPCC reference citations
- `AGENT-FOUND-008` (Reproducibility) - Calculation reproducibility
- `AGENT-DATA-010` (Data Quality) - Input data quality profiling

### External Dependencies
- `pydantic>=2.0` - Data validation
- `fastapi>=0.100` - REST API framework
- `prometheus_client>=0.17` - Metrics collection
- `psycopg>=3.0` - PostgreSQL driver (optional)

## 13. Glossary

| Term | Definition |
|------|-----------|
| AFOLU | Agriculture, Forestry and Other Land Use |
| AWMS | Animal Waste Management System |
| Bo | Maximum methane producing capacity (m3 CH4/kg VS) |
| CH4 | Methane |
| CO2 | Carbon dioxide |
| CO2e | Carbon dioxide equivalent |
| Cfi | Maintenance coefficient for net energy calculations |
| DE | Digestible energy (% of gross energy) |
| EF | Emission factor |
| Frac_GASF | Fraction of synthetic N fertilizer that volatilizes |
| Frac_GASM | Fraction of organic N that volatilizes |
| Frac_LEACH | Fraction of N that is leached/runoff |
| GE | Gross energy intake (MJ/head/day) |
| GWP | Global warming potential |
| MCF | Methane correction factor |
| MS | Fraction of manure managed by specific AWMS |
| N2O | Nitrous oxide |
| NE | Net energy (maintenance, activity, lactation, growth, pregnancy, work) |
| Nex | Annual nitrogen excretion rate (kg N/head/yr) |
| REM | Ratio of net energy for maintenance to digestible energy |
| REG | Ratio of net energy for growth to digestible energy |
| RPR | Residue-to-product ratio |
| SF | Scaling factor (rice cultivation) |
| VS | Volatile solids excreted (kg/head/day) |
| Ym | Methane conversion factor (% of GE, enteric) |
