# PRD: AGENT-MRV-007 Waste Treatment Emissions Agent (GL-MRV-SCOPE1-007)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-SCOPE1-007 |
| **Internal Label** | AGENT-MRV-007 |
| **Category** | Layer 3 - MRV / Accounting Agents (Scope 1) |
| **Package** | `greenlang/waste_treatment_emissions/` |
| **DB Migration** | V058 |
| **Priority** | P1 - Critical |
| **Regulatory Drivers** | IPCC 2006 Guidelines Vol 5 (Waste), IPCC 2019 Refinement Ch 5, GHG Protocol Corporate Standard (2015), GHG Protocol Scope 3 Cat 5 (Waste Generated in Operations), ISO 14064-1:2018, CSRD/ESRS E1 & E5, EU Waste Framework Directive 2008/98/EC (amended 2018/851), EU Industrial Emissions Directive 2010/75/EU, EPA 40 CFR Part 98 Subpart HH/TT, DEFRA Environmental Reporting Guidelines |

## 2. Problem Statement

On-site waste treatment represents a significant and complex source of Scope 1 GHG emissions across manufacturing, food processing, chemical production, pharmaceutical, and waste management sectors. Organizations operating on-site waste treatment facilities must accurately quantify:

- **Biological treatment emissions** (CH4 and N2O from composting, anaerobic digestion, mechanical-biological treatment) accounting for process conditions, feedstock composition, and gas recovery
- **Thermal treatment emissions** (CO2, CH4, N2O from incineration, pyrolysis, gasification, and open burning) including fossil vs biogenic carbon separation
- **Wastewater treatment emissions** (CH4 from anaerobic conditions, N2O from nitrification/denitrification) from on-site industrial and process wastewater treatment
- **Chemical treatment emissions** (CO2 from neutralization, oxidation, reduction processes)
- **Methane recovery and utilization** (captured CH4 flared, used for energy, or vented)
- **Avoided emissions** from energy recovery and material recycling (offset credits)

This is critical for:
- GHG Protocol Scope 1 reporting (direct emissions from owned/controlled on-site treatment)
- GHG Protocol Scope 3 Category 5 (waste generated in operations sent to third-party treatment)
- CSRD/ESRS E1 disclosure (GHG emissions) and E5 (resource use and circular economy)
- EU Industrial Emissions Directive (IED) compliance for permitted installations
- EPA Mandatory Greenhouse Gas Reporting (40 CFR Part 98) for waste treatment facilities
- Carbon pricing and EU ETS for waste-to-energy installations
- Circular economy metrics and waste hierarchy compliance

## 3. Existing Layer 1 Capabilities

Several basic waste-related files exist in the Layer 1 agent codebase:

- `greenlang/agents/mrv/waste/base.py` (GL-MRV-WST-BASE): Base class with WasteType, TreatmentMethod, LandfillType enums, IPCC DOC/MCF/L0 tables, FOD calculation helper, provenance hashing
- `greenlang/agents/mrv/waste/landfill_mrv.py` (GL-MRV-WST-001): Basic landfill methane with FOD
- `greenlang/agents/mrv/waste/incineration_mrv.py` (GL-MRV-WST-002): Basic incineration CO2/N2O
- `greenlang/agents/mrv/waste/composting_mrv.py` (GL-MRV-WST-003): Basic composting CH4/N2O
- `greenlang/agents/mrv/waste/recycling_mrv.py` (GL-MRV-WST-004): Basic recycling avoided emissions
- `greenlang/agents/mrv/waste/hazardous_waste_mrv.py` (GL-MRV-WST-005): Basic hazardous waste
- `greenlang/agents/mrv/waste/plastic_waste_mrv.py` (GL-MRV-WST-006): Basic plastic waste
- `greenlang/agents/mrv/water/wastewater.py` (GL-MRV-WAT-002): Basic wastewater CH4/N2O

**Current gaps:** No unified on-site waste treatment agent covering all treatment methods, no IPCC 2019 updated factors, no multi-stream treatment tracking, no biological/thermal/wastewater engine separation, no treatment efficiency tracking, no methane recovery/utilization accounting, no regulatory compliance checking across 7+ frameworks, no uncertainty quantification, no provenance chain tracking, no waste composition analysis engine, no treatment cost-emissions optimization, no batch processing, no REST API.

## 4. Gaps Requiring Layer 3 Production Implementation

| # | Gap | Impact |
|---|-----|--------|
| 1 | No unified multi-treatment-method agent | Cannot handle facilities with multiple treatment streams in one calculation |
| 2 | No IPCC 2019 updated emission factors | Using outdated 2006 factors only; missing 2019 refinements for biological treatment |
| 3 | No biological treatment engine (composting/AD/MBT) | Limited composting calculations, no anaerobic digestion biogas modeling, no MBT tracking |
| 4 | No thermal treatment engine (incineration/pyrolysis/gasification) | Basic incineration only; missing continuous/batch mode, fossil/biogenic carbon split, APC residue |
| 5 | No wastewater treatment engine for on-site industrial wastewater | Separate wastewater agent exists but not integrated for on-site Scope 1 |
| 6 | No methane recovery and utilization tracking | Cannot account for CH4 captured, flared, or used for energy generation |
| 7 | No waste composition analysis | Cannot determine DOC, fossil carbon fraction, moisture content from waste streams |
| 8 | No treatment efficiency metrics | Cannot track destruction removal efficiency (DRE), gas collection efficiency |
| 9 | No multi-framework regulatory compliance | No automated checking against IPCC/GHG Protocol/CSRD/EPA/EU IED/DEFRA |
| 10 | No uncertainty quantification | No Monte Carlo for waste composition variability and EF uncertainty |
| 11 | No fossil vs biogenic CO2 separation | Critical for ETS and carbon pricing - biogenic CO2 often excluded |
| 12 | No waste-to-energy offset calculations | Cannot account for electricity/heat generated displacing grid emissions |
| 13 | No batch processing for multiple facilities | Cannot process multiple treatment facilities/streams simultaneously |
| 14 | No provenance chain tracking | No SHA-256 audit trail for regulatory verification |

## 5. Architecture (7 Engines)

| Engine | Class | File | Purpose |
|--------|-------|------|---------|
| 1 | WasteTreatmentDatabaseEngine | waste_treatment_database.py | Waste types, treatment methods, IPCC/EPA/DEFRA emission factors, DOC/MCF tables |
| 2 | BiologicalTreatmentEngine | biological_treatment.py | Composting CH4/N2O, anaerobic digestion with biogas, MBT, vermicomposting |
| 3 | ThermalTreatmentEngine | thermal_treatment.py | Incineration, pyrolysis, gasification, open burning, fossil/biogenic CO2 split |
| 4 | WastewaterTreatmentEngine | wastewater_treatment.py | On-site industrial wastewater CH4/N2O, BOD/COD-based calculations |
| 5 | UncertaintyQuantifierEngine | uncertainty_quantifier.py | Monte Carlo simulation, error propagation, DQI scoring |
| 6 | ComplianceCheckerEngine | compliance_checker.py | GHG Protocol, IPCC, CSRD, EPA, EU IED, DEFRA, ISO 14064 validation |
| 7 | WasteTreatmentPipelineEngine | waste_treatment_pipeline.py | 8-stage pipeline orchestration |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | SDK facade with graceful imports |
| `config.py` | GL_WASTE_TREATMENT_ env prefix, thread-safe singleton |
| `models.py` | Enums, Pydantic v2 models, IPCC constant tables |
| `metrics.py` | 12 Prometheus metrics with gl_wt_ prefix |
| `provenance.py` | SHA-256 chain-hashed audit trails |
| `setup.py` | WasteTreatmentEmissionsService facade |
| `api/router.py` | 20 REST endpoints at /api/v1/waste-treatment-emissions |

## 6. Key Features

### 6.1 Waste Types (19 Categories)

Per IPCC 2006 Guidelines Volume 5 and 2019 Refinement:

1. **Municipal Solid Waste (MSW)** - Mixed residential and commercial waste
2. **Industrial Waste** - Manufacturing and production waste
3. **Construction & Demolition (C&D)** - Building materials, concrete, wood
4. **Organic Waste** - Biodegradable organic fraction
5. **Food Waste** - Pre/post-consumer food waste
6. **Yard/Garden Waste** - Green waste, leaves, branches
7. **Paper** - Office paper, newspaper, magazines
8. **Cardboard** - Corrugated and flat cardboard
9. **Plastic** - All polymer types (PE, PP, PET, PS, PVC)
10. **Metal** - Ferrous and non-ferrous metals
11. **Glass** - Container and flat glass
12. **Textiles** - Natural and synthetic fibers
13. **Wood** - Treated and untreated wood waste
14. **Rubber** - Tires and rubber products
15. **E-Waste** - Electronic and electrical waste
16. **Hazardous Waste** - Chemical, biological, radioactive
17. **Medical Waste** - Clinical and pharmaceutical waste
18. **Sludge** - Sewage sludge, industrial sludge
19. **Mixed Waste** - Unsorted or commingled waste

### 6.2 Treatment Methods (15 Methods)

1. **Landfill** - Managed anaerobic disposal (reference only for on-site)
2. **Landfill with Gas Capture** - Managed with LFG collection system
3. **Incineration** - Mass burn, modular, rotary kiln
4. **Incineration with Energy Recovery** - Waste-to-energy (WtE) plants
5. **Recycling** - Material recovery and reprocessing
6. **Composting** - Aerobic decomposition (windrow, in-vessel, aerated static pile)
7. **Anaerobic Digestion (AD)** - Biogas production (wet/dry, mesophilic/thermophilic)
8. **Mechanical-Biological Treatment (MBT)** - Combined mechanical sorting + biological treatment
9. **Pyrolysis** - Thermal decomposition without oxygen
10. **Gasification** - Partial oxidation to syngas
11. **Chemical Treatment** - Neutralization, oxidation, precipitation
12. **Thermal Treatment** - Other thermal processes (autoclaving, microwave)
13. **Biological Treatment** - Bioaugmentation, bioremediation
14. **Open Burning** - Uncontrolled combustion (developing regions)
15. **Open Dumping** - Unmanaged disposal (reference)

### 6.3 Calculation Methods (7 Methods)

**IPCC First Order Decay (FOD)**
For landfill methane generation:
```
CH4_generated(T) = DDOCm_decomp(T) * F * 16/12
DDOCm_decomp(T) = DDOCma(T-1) * (1 - e^(-k))
DDOCma(T) = DDOCmd(T) + DDOCma(T-1) * e^(-k)
DDOCmd = W * DOC * DOCf * MCF
```
Where:
- W = mass of waste deposited (Gg)
- DOC = degradable organic carbon fraction
- DOCf = fraction of DOC that decomposes (default 0.5)
- MCF = methane correction factor by site type
- F = fraction of CH4 in landfill gas (default 0.5)
- k = decay rate constant (1/yr) = ln(2)/half-life
- T = year of inventory

**IPCC Tier 1 (Default Factors)**
For biological treatment:
```
CH4_composting = M * EF_CH4     (kg CH4 = Gg waste * kg CH4/Gg waste)
N2O_composting = M * EF_N2O     (kg N2O = Gg waste * kg N2O/Gg waste)
```
IPCC 2019 defaults:
- Composting CH4: 4 g/kg waste (well-managed) to 10 g/kg (poorly managed)
- Composting N2O: 0.24 g/kg waste (well-managed) to 0.6 g/kg (poorly managed)
- AD CH4: 0.8 g/kg waste (with gas recovery) to 2 g/kg (venting)

**IPCC Tier 2 (Country-Specific)**
Incineration CO2:
```
CO2 = Sigma_j (SW_j * dm_j * CF_j * FCF_j * OF_j) * 44/12
```
Where:
- SW_j = mass of waste type j incinerated (Gg)
- dm_j = dry matter content fraction
- CF_j = carbon fraction of dry matter
- FCF_j = fossil carbon fraction
- OF_j = oxidation factor (default 1.0 for modern incinerators)
- 44/12 = molecular weight ratio CO2/C

**IPCC Tier 3 (Facility-Specific)**
Using continuous emissions monitoring (CEMS) or detailed process data

**Mass Balance Method**
```
Emissions = (C_in - C_out - C_stored) * 44/12
```
Where:
- C_in = total carbon entering treatment process
- C_out = carbon in outputs (residues, recyclates)
- C_stored = carbon sequestered in products

**Direct Measurement**
CEMS data from stack monitoring or biogas analysis

**Spend-Based Method**
Using spend data with waste-sector emission factors (DEFRA/EPA/Ecoinvent)

### 6.4 Biological Treatment Calculations

#### Composting Emissions
```
CH4_emitted = M_organic * EF_CH4 * (1 - R_CH4)
N2O_emitted = M_organic * EF_N2O
```
Where:
- M_organic = mass of organic waste treated (tonnes)
- EF_CH4 = CH4 emission factor (kg/tonne)
- R_CH4 = CH4 recovery fraction (biofilter efficiency)
- EF_N2O = N2O emission factor (kg/tonne)

IPCC 2019 Emission Factors (Table 5.1):
| Treatment Type | CH4 (g/kg waste) | N2O (g/kg waste) |
|---------------|------------------|------------------|
| Composting (well-managed) | 4.0 | 0.24 |
| Composting (poorly managed) | 10.0 | 0.6 |
| Anaerobic digestion (vented) | 2.0 | 0.0 |
| Anaerobic digestion (flared) | 0.8 | 0.0 |
| MBT (aerobic) | 4.0 | 0.3 |
| MBT (anaerobic pre-treatment) | 2.0 | 0.1 |

#### Anaerobic Digestion Biogas
```
V_biogas = M_vs * BMP * eta_digestion
V_CH4 = V_biogas * X_CH4
CH4_generated = V_CH4 * rho_CH4
CH4_emitted = CH4_generated * (1 - eta_capture) * (1 - eta_flare) * (1 - eta_utilize)
```
Where:
- M_vs = volatile solids mass (tonnes)
- BMP = biochemical methane potential (m3 CH4/tonne VS)
- eta_digestion = digestion efficiency (0.5-0.9)
- X_CH4 = methane fraction in biogas (0.50-0.70)
- rho_CH4 = density of CH4 (0.0007168 tonnes/m3 at STP)
- eta_capture/flare/utilize = capture/flare/utilization efficiencies

### 6.5 Thermal Treatment Calculations

#### Incineration CO2 (Fossil Component)
```
CO2_fossil = Sum_j [ IW_j * CCW_j * FCF_j * EF_j * OF_j ] * 44/12
CO2_biogenic = Sum_j [ IW_j * CCW_j * (1 - FCF_j) * EF_j * OF_j ] * 44/12
```
Where:
- IW_j = amount of waste type j incinerated (tonnes, wet basis)
- CCW_j = carbon content of waste type j (fraction of wet weight)
- FCF_j = fossil carbon fraction of total carbon
- EF_j = burn-out efficiency (default 0.95-1.0)
- OF_j = oxidation factor (1.0 for modern incinerators)

IPCC Fossil Carbon Fractions (Table 5.2):
| Waste Component | Carbon Content (% wet) | Fossil Carbon Fraction |
|----------------|----------------------|----------------------|
| Food waste | 15% | 0% |
| Paper/cardboard | 35-40% | 1% |
| Plastics | 60-75% | 100% |
| Textiles (synthetic) | 40-50% | 80% |
| Textiles (natural) | 40-50% | 0% |
| Rubber/leather | 40-60% | 20% |
| Wood | 43-50% | 0% |
| Garden waste | 17-20% | 0% |
| Nappies/diapers | 24-30% | 10% |
| Sludge | 5-15% | 0% |

#### Incineration N2O and CH4
```
N2O = Sum_j (IW_j * EF_N2O_j)
CH4 = Sum_j (IW_j * EF_CH4_j)
```
IPCC Default EFs (Table 5.3):
| Technology | N2O (kg/Gg waste) | CH4 (kg/Gg waste) |
|-----------|-------------------|-------------------|
| Stoker/grate | 50 | 0.2 |
| Fluidized bed | 56 | 0.68 |
| Rotary kiln | 50 | 0.2 |
| Semi-continuous | 60 | 6.0 |
| Batch type | 60 | 60 |

#### Open Burning
```
CO2 = Sum_j (OW_j * DM_j * CF_j * FCF_j * OF_open) * 44/12
CH4 = Sum_j (OW_j * DM_j * CF_j * EF_CH4_open) * 16/12
N2O = Sum_j (OW_j * DM_j * CF_j * EF_N2O_open) * 44/28
```
Where OF_open = 0.58 (incomplete combustion)

#### Energy Recovery Credits
```
E_recovered = M_waste * NCV * eta_recovery
E_displaced = E_recovered * EF_grid
```
Where:
- NCV = net calorific value of waste (GJ/tonne)
- eta_recovery = energy recovery efficiency (0.15-0.35)
- EF_grid = grid emission factor (tCO2e/GJ)

### 6.6 Wastewater Treatment Calculations

#### CH4 from Wastewater (IPCC Ch 6 Method)
```
CH4_wastewater = (TOW - S) * Bo * MCF_ww * 0.001 - R
```
Where:
- TOW = total organic waste in wastewater (kg BOD or COD/yr)
- S = organic component removed as sludge (kg BOD or COD/yr)
- Bo = maximum CH4 producing capacity (kg CH4/kg BOD = 0.6 or kg CH4/kg COD = 0.25)
- MCF_ww = methane correction factor by treatment system
- R = CH4 recovered (tonnes/yr)

IPCC Wastewater MCF Values:
| Treatment System | MCF |
|-----------------|-----|
| Untreated (sea/river discharge) | 0.1 |
| Aerobic treatment (well-managed) | 0.0 |
| Aerobic treatment (overloaded) | 0.3 |
| Anaerobic reactor (no recovery) | 0.8 |
| Anaerobic reactor (with recovery) | 0.8 (but R offsets) |
| Anaerobic shallow lagoon (<2m) | 0.2 |
| Anaerobic deep lagoon (>2m) | 0.8 |
| Septic system | 0.5 |
| Latrine (dry, groundwater below) | 0.1 |
| Latrine (wet, groundwater intersects) | 0.7 |

#### N2O from Wastewater Treatment
```
N2O_plant = P * T_protein * F_NPR * F_NON_CON * F_IND_COM * EF_plant * 44/28
N2O_effluent = N_effluent * EF_effluent * 44/28
```
Where:
- P = population or production-equivalent (persons)
- T_protein = annual per-capita protein consumption (kg/person/yr)
- F_NPR = fraction of nitrogen in protein (default 0.16)
- F_NON_CON = factor for non-consumed protein
- F_IND_COM = factor for industrial/commercial co-discharge
- EF_plant = N2O emission factor for plant (default 0.016 kg N2O-N/kg N)
- EF_effluent = N2O emission factor for effluent (default 0.005 kg N2O-N/kg N)

### 6.7 Methane Recovery and Utilization

```
CH4_net = CH4_generated - CH4_captured
CH4_captured = CH4_generated * eta_collection
CH4_flared = CH4_captured * f_flare * (1 - eta_destruction)
CH4_utilized = CH4_captured * f_utilize * (1 - eta_conversion)
CH4_vented = CH4_captured * f_vent
CH4_emitted = CH4_net + CH4_flared + CH4_vented
```
Where:
- eta_collection = collection efficiency (0.50-0.95)
- f_flare/utilize/vent = fraction routed to flare/engine/vent
- eta_destruction = flare destruction efficiency (0.96-0.99)
- eta_conversion = utilization conversion efficiency (0.90-0.98)

### 6.8 Emission Gases (4 Primary)

- **CO2**: From fossil carbon in incineration, chemical treatment, open burning
- **CH4**: From anaerobic decomposition, incomplete combustion, biogas leaks
- **N2O**: From nitrification/denitrification, thermal NOx in combustion
- **CO**: From incomplete combustion (trace gas, converted to CO2e)

### 6.9 GWP Values (IPCC AR6 100-year)

| Gas | GWP-100 (AR6) | GWP-100 (AR5) | GWP-100 (AR4) | GWP-20 (AR6) |
|-----|---------------|---------------|---------------|--------------|
| CO2 | 1 | 1 | 1 | 1 |
| CH4 (fossil) | 29.8 | 28 | 25 | 82.5 |
| CH4 (biogenic) | 27.0 | 28 | 25 | 80.8 |
| N2O | 273 | 265 | 298 | 273 |
| CO | 4.06 | 1.9 | 1.9 | 4.06 |

### 6.10 Regulatory Frameworks (7)

1. **IPCC 2006 Guidelines Volume 5** (Waste) - Primary methodology for all treatment types
2. **IPCC 2019 Refinement** - Updated biological treatment factors and methods
3. **GHG Protocol Corporate Standard (2015)** - Scope 1 and Scope 3 Category 5
4. **ISO 14064-1:2018** - Category 1 direct emissions, Category 3 indirect
5. **CSRD/ESRS E1 & E5** - E1-6 GHG emissions, E5 resource use and circular economy
6. **EPA 40 CFR Part 98 Subpart HH/TT** - US mandatory reporting for waste treatment
7. **DEFRA Environmental Reporting Guidelines** - UK emissions conversion factors

### 6.11 Emission Factor Sources (7)

1. IPCC 2006 Guidelines Volume 5 default tables
2. IPCC 2019 Refinement updated tables
3. EPA AP-42 compilation of emission factors
4. DEFRA/BEIS Greenhouse Gas Conversion Factors (annual)
5. Ecoinvent database (life cycle emission factors)
6. National inventory data (country-specific Tier 2)
7. Custom / facility-specific measurements (Tier 3)

## 7. Database Schema (V058)

### Schema: waste_treatment_emissions_service

### Tables (10)

| Table | Purpose |
|-------|---------|
| wt_treatment_facilities | Treatment facility registry with capacity, methods, location |
| wt_waste_streams | Waste stream definitions with composition, volume, source |
| wt_emission_factors | IPCC/EPA/DEFRA emission factors by treatment method and waste type |
| wt_treatment_events | Individual waste treatment event records |
| wt_calculations | Emission calculation results with gas breakdown |
| wt_calculation_details | Per-gas, per-stream emission breakdown |
| wt_methane_recovery | Methane recovery, flaring, and utilization tracking |
| wt_energy_recovery | Energy recovery and grid displacement credits |
| wt_compliance_records | Regulatory compliance check results |
| wt_audit_entries | Provenance and audit trail entries |

### Hypertables (3) - 7-day chunks

| Hypertable | Purpose |
|------------|---------|
| wt_calculation_events | Time-series calculation tracking |
| wt_treatment_events_ts | Time-series treatment event tracking |
| wt_compliance_events | Time-series compliance check tracking |

### Continuous Aggregates (2)

| Aggregate | Purpose |
|-----------|---------|
| wt_hourly_calculation_stats | Hourly calculation volume and emissions by treatment method |
| wt_daily_emission_totals | Daily emission totals by treatment method and waste type |

### Row-Level Security
- All tables: tenant_id-based RLS policies
- Read: `gl_readonly`, `gl_service`, `gl_admin`
- Write: `gl_service`, `gl_admin`
- Delete: `gl_admin` only
- Agent registry: `gl_agent_service`

## 8. API Endpoints (20)

### Base Path: `/api/v1/waste-treatment-emissions`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/calculations` | Execute single waste treatment emission calculation |
| POST | `/calculations/batch` | Execute batch calculations (max 10,000) |
| GET | `/calculations/{id}` | Get calculation by ID |
| GET | `/calculations` | List calculations with filters |
| DELETE | `/calculations/{id}` | Delete calculation |
| POST | `/facilities` | Register treatment facility |
| GET | `/facilities` | List treatment facilities |
| PUT | `/facilities/{id}` | Update facility metadata |
| POST | `/waste-streams` | Register waste stream |
| GET | `/waste-streams` | List waste streams |
| PUT | `/waste-streams/{id}` | Update waste stream composition |
| POST | `/treatment-events` | Record treatment event |
| GET | `/treatment-events` | List treatment events with filters |
| POST | `/methane-recovery` | Record methane recovery event |
| GET | `/methane-recovery/{facility_id}` | Get methane recovery history |
| POST | `/compliance/check` | Run compliance check |
| GET | `/compliance/{id}` | Get compliance result |
| POST | `/uncertainty` | Run Monte Carlo uncertainty analysis |
| GET | `/aggregations` | Get aggregated emissions by method/waste-type/period |
| GET | `/health` | Health check endpoint |

## 9. Auth Integration

### Permission Map Entries

| Route Pattern | Permission |
|--------------|------------|
| `POST /calculations` | `waste-treatment:calculate` |
| `POST /calculations/batch` | `waste-treatment:calculate` |
| `GET /calculations/*` | `waste-treatment:read` |
| `DELETE /calculations/*` | `waste-treatment:delete` |
| `POST /facilities` | `waste-treatment:facilities:write` |
| `GET /facilities` | `waste-treatment:facilities:read` |
| `PUT /facilities/*` | `waste-treatment:facilities:write` |
| `POST /waste-streams` | `waste-treatment:streams:write` |
| `GET /waste-streams` | `waste-treatment:streams:read` |
| `PUT /waste-streams/*` | `waste-treatment:streams:write` |
| `POST /treatment-events` | `waste-treatment:events:write` |
| `GET /treatment-events` | `waste-treatment:events:read` |
| `POST /methane-recovery` | `waste-treatment:recovery:write` |
| `GET /methane-recovery/*` | `waste-treatment:recovery:read` |
| `POST /compliance/check` | `waste-treatment:compliance:check` |
| `GET /compliance/*` | `waste-treatment:compliance:read` |
| `POST /uncertainty` | `waste-treatment:uncertainty:run` |
| `GET /aggregations` | `waste-treatment:read` |

## 10. Testing Requirements

### Target: 1,000+ tests across 15 test files

| Test File | Covers | Target Tests |
|-----------|--------|-------------|
| test_models.py | Enums, Pydantic models, constants | 120 |
| test_config.py | Configuration, env vars, validation | 40 |
| test_metrics.py | Prometheus metrics registration | 25 |
| test_provenance.py | SHA-256 chain hashing, audit trails | 45 |
| test_waste_treatment_database.py | Engine 1: Waste types, EFs, DOC/MCF tables | 110 |
| test_biological_treatment.py | Engine 2: Composting, AD, MBT | 130 |
| test_thermal_treatment.py | Engine 3: Incineration, pyrolysis, gasification | 130 |
| test_wastewater_treatment.py | Engine 4: BOD/COD, CH4/N2O | 100 |
| test_uncertainty_quantifier.py | Engine 5: Monte Carlo, DQI | 85 |
| test_compliance_checker.py | Engine 6: 7 frameworks, requirements | 100 |
| test_waste_treatment_pipeline.py | Engine 7: 8-stage orchestration | 90 |
| test_setup.py | Service facade, lifecycle | 30 |
| test_api.py | 20 REST endpoints | 95 |
| conftest.py | Shared fixtures | N/A |

### Test Categories
- **Unit tests**: Each engine method independently
- **Integration tests**: Cross-engine data flow
- **Regulatory tests**: Known IPCC examples with expected results
- **Edge cases**: Zero waste, negative recovery, extreme compositions
- **Determinism**: Decimal arithmetic, reproducible results

## 11. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | All 19 waste types with DOC/carbon content data | Database engine returns correct factors |
| 2 | All 15 treatment methods supported | Calculator produces correct per-method results |
| 3 | Biological treatment (composting/AD/MBT) CH4/N2O | Matches IPCC 2019 Table 5.1 examples |
| 4 | Thermal treatment fossil/biogenic CO2 split | Matches IPCC Vol 5 Ch 5 examples |
| 5 | Wastewater treatment CH4/N2O calculations | Matches IPCC Vol 5 Ch 6 BOD/COD methods |
| 6 | Methane recovery and utilization tracking | Accounts for capture, flare, utilize, vent |
| 7 | Energy recovery offset calculations | Correctly displaces grid emissions |
| 8 | First Order Decay model for landfill gas | Multi-year decay with half-life by climate/waste |
| 9 | Monte Carlo with configurable iterations | 95% CI within expected ranges |
| 10 | All 7 regulatory frameworks validated | Compliance checker covers all requirements |
| 11 | 1,000+ tests passing | pytest --tb=short returns 0 |
| 12 | Deterministic Decimal arithmetic | Hash-identical results across runs |
| 13 | Auth integration complete | All 18 permission entries in PERMISSION_MAP |
| 14 | Fossil vs biogenic CO2 separated | Correctly identifies fossil carbon in waste streams |
| 15 | Batch processing for multiple facilities | Handles 10,000 concurrent calculations |

## 12. Dependencies

### Internal Dependencies
- `greenlang/infrastructure/auth_service/` - JWT/RBAC integration
- `greenlang/observability/prometheus/` - Metrics export
- `greenlang/observability/tracing/` - OpenTelemetry spans
- `greenlang/agents/foundational/provenance/` - Audit trail patterns
- `greenlang/agents/foundational/uncertainty/` - Monte Carlo patterns
- `greenlang/agents/mrv/waste/base.py` - Base waste MRV classes and constants

### External Dependencies
- `pydantic>=2.0` - Data models
- `fastapi>=0.100` - REST API
- `prometheus_client>=0.17` - Metrics
- `numpy>=1.24` - Monte Carlo sampling (optional, fallback to random)

## 13. Glossary

| Term | Definition |
|------|-----------|
| AD | Anaerobic Digestion |
| APC | Air Pollution Control (residue from incineration) |
| BMP | Biochemical Methane Potential |
| BOD | Biochemical Oxygen Demand |
| CCW | Carbon Content of Waste |
| CEMS | Continuous Emissions Monitoring System |
| COD | Chemical Oxygen Demand |
| DOC | Degradable Organic Carbon |
| DOCf | Fraction of DOC that decomposes |
| DRE | Destruction and Removal Efficiency |
| FCF | Fossil Carbon Fraction |
| FOD | First Order Decay (IPCC method) |
| IED | Industrial Emissions Directive (EU) |
| LFG | Landfill Gas |
| MBT | Mechanical-Biological Treatment |
| MCF | Methane Correction Factor |
| NCV | Net Calorific Value |
| OF | Oxidation Factor |
| STP | Standard Temperature and Pressure |
| TOW | Total Organic Waste (in wastewater) |
| VS | Volatile Solids |
| WtE | Waste-to-Energy |
