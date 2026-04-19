# California SB 253 Climate Disclosure Agent - Implementation To-Do List

**Team:** GL-SB253-PM (US State Climate Disclosure Platform)
**Version:** 1.0.0
**Date:** 2025-12-04
**Total Duration:** 36 weeks (leverage GL-VCCI-APP 55% complete)
**Deadline:** June 30, 2026 (Scope 1 & 2), June 30, 2027 (Scope 3)
**Priority:** P1-HIGH
**Total Tasks:** 50 detailed implementation tasks

---

## Executive Summary

California SB 253 (Climate Corporate Data Accountability Act) requires companies with $1B+ revenue doing business in California to disclose Scope 1, 2, and 3 greenhouse gas emissions. This implementation plan defines 50 tasks covering all GHG Protocol scopes and third-party assurance requirements.

**Key Regulatory Requirements:**
- **Applicability:** Companies with $1B+ annual revenue doing business in California
- **Scope 1 & 2 Deadline:** June 30, 2026 (first reporting year: 2025)
- **Scope 3 Deadline:** June 30, 2027 (first reporting year: 2026)
- **Third-Party Assurance:** Limited assurance (2026), Reasonable assurance (2030)
- **Penalty:** Up to $500,000 per year for non-compliance
- **Estimated Scope:** 5,400+ companies

**Leverage Existing GL-VCCI-APP (55% Complete):**
- Reuse Scope 1, 2, 3 calculation engines (all 15 categories)
- Reuse ERP connectors (SAP, Oracle, Workday)
- Reuse provenance tracking (SHA-256 audit trails)
- Add California-specific compliance engine
- Add CARB portal integration
- Add third-party assurance package generation

---

## Task Distribution Summary

| Category | Tasks | Priority | Weeks |
|----------|-------|----------|-------|
| Scope 1 - Direct Emissions | 8 | P0 | 1-4 |
| Scope 2 - Energy Indirect | 6 | P0 | 5-8 |
| Scope 3 - Value Chain (15 categories) | 30 | P1 | 9-24 |
| Verification & Assurance | 6 | P0 | 25-32 |
| **Total** | **50** | - | **36 weeks** |

---

## SCOPE 1: Direct Emissions (8 Tasks)

**Definition:** Direct GHG emissions from sources owned or controlled by the company
**GHG Protocol Reference:** Corporate Standard Chapter 4
**California Grid Factor:** CAMX (California) = 0.254 kg CO2e/kWh (EPA eGRID 2023)

### Task S1-001: Stationary Combustion Calculator

**Priority:** P0-CRITICAL
**Duration:** 1 week
**Dependencies:** Emission factor database (EPA GHG EF Hub)

**Description:**
Build calculator for stationary combustion emissions from boilers, furnaces, generators, and other fixed equipment.

**Calculation Formula:**
```
Scope1_Stationary = SUM(Fuel_Consumed[i] × Emission_Factor[i])
```

**Implementation Details:**
- [ ] Implement fuel consumption data ingestion (natural gas, diesel, propane, fuel oil)
- [ ] Integrate EPA GHG Emission Factors Hub (stationary combustion)
- [ ] Support multiple fuel units (therms, gallons, MMBtu, kWh)
- [ ] Implement unit conversion utility
- [ ] Add CO2, CH4, N2O breakdown per IPCC AR6 GWP values
- [ ] Generate audit trail with SHA-256 provenance hash
- [ ] Create golden tests (10+ test cases)

**Emission Factors (EPA GHG EF Hub 2024):**
| Fuel | Factor | Unit | Source |
|------|--------|------|--------|
| Natural Gas | 5.30 | kg CO2e/therm | EPA |
| Diesel | 10.21 | kg CO2e/gallon | EPA |
| Propane | 5.72 | kg CO2e/gallon | EPA |
| Fuel Oil #2 | 10.21 | kg CO2e/gallon | EPA |

**Acceptance Criteria:**
- Calculation accuracy: +/- 1%
- Audit trail complete with factor provenance
- Unit tests: 90%+ coverage

---

### Task S1-002: Mobile Combustion Calculator (Fleet Vehicles)

**Priority:** P0-CRITICAL
**Duration:** 1 week
**Dependencies:** Task S1-001

**Description:**
Build calculator for mobile combustion emissions from company-owned vehicles.

**Calculation Formula:**
```
Scope1_Mobile = SUM(Fuel_Consumed[i] × Emission_Factor[i])
OR
Scope1_Mobile = SUM(Miles_Traveled[i] × (1/MPG[i]) × Emission_Factor[i])
```

**Implementation Details:**
- [ ] Implement fuel-based calculation method
- [ ] Implement distance-based calculation method with fuel economy
- [ ] Support vehicle categories (passenger cars, light trucks, heavy trucks)
- [ ] Integrate EPA emission factors for mobile sources
- [ ] Add vehicle type selection (gasoline, diesel, hybrid, BEV)
- [ ] Support fleet aggregation
- [ ] Generate audit trail

**Emission Factors (EPA GHG EF Hub 2024):**
| Fuel Type | Factor | Unit | Source |
|-----------|--------|------|--------|
| Gasoline | 8.78 | kg CO2e/gallon | EPA |
| Diesel | 10.21 | kg CO2e/gallon | EPA |
| E10 (10% ethanol) | 8.53 | kg CO2e/gallon | EPA |

**Acceptance Criteria:**
- Support both fuel-based and distance-based methods
- Calculation accuracy: +/- 2%
- Unit tests: 90%+ coverage

---

### Task S1-003: Fugitive Emissions Calculator (Refrigerants)

**Priority:** P0-CRITICAL
**Duration:** 1 week
**Dependencies:** IPCC AR6 GWP values

**Description:**
Build calculator for fugitive emissions from refrigerant leakage (HVAC, chillers, refrigeration).

**Calculation Formula:**
```
Scope1_Fugitive = Refrigerant_Leaked[kg] × GWP[refrigerant]
```

**Implementation Details:**
- [ ] Implement refrigerant inventory tracking
- [ ] Support common refrigerants (R-134a, R-410A, R-407C, R-22)
- [ ] Integrate IPCC AR6 GWP-100 values
- [ ] Support leak rate estimation (2-10% annual for commercial)
- [ ] Add equipment type classification
- [ ] Generate audit trail

**GWP Values (IPCC AR6):**
| Refrigerant | GWP-100 | Common Use |
|-------------|---------|------------|
| R-134a | 1,530 | Auto AC |
| R-410A | 2,088 | Commercial HVAC |
| R-407C | 1,774 | Commercial HVAC |
| R-22 (HCFC) | 1,810 | Legacy systems |

**Acceptance Criteria:**
- GWP values align with IPCC AR6
- Support equipment-based leak rate estimation
- Unit tests: 85%+ coverage

---

### Task S1-004: Process Emissions Calculator

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** Industrial sector classification

**Description:**
Build calculator for process emissions from industrial chemical/physical processes (cement, steel, aluminum, chemicals).

**Calculation Formula:**
```
Scope1_Process = Production_Output[tonnes] × Process_Emission_Factor[tonnes CO2e/tonne product]
```

**Implementation Details:**
- [ ] Implement industry sector classification (NAICS codes)
- [ ] Support cement production (clinker calcination)
- [ ] Support steel production (BF-BOF, EAF)
- [ ] Support chemical manufacturing (ammonia, nitric acid)
- [ ] Integrate IPCC Tier 1 default emission factors
- [ ] Add process-specific calculation methodologies
- [ ] Generate audit trail

**Process Emission Factors (IPCC 2019):**
| Process | Factor | Unit | Source |
|---------|--------|------|--------|
| Cement clinker | 0.52 | tonnes CO2/tonne | IPCC |
| Steel (BF-BOF) | 2.1 | tonnes CO2/tonne | IEA |
| Steel (EAF) | 0.5 | tonnes CO2/tonne | IEA |
| Ammonia | 2.3 | tonnes CO2/tonne | IPCC |

**Acceptance Criteria:**
- Support major industrial sectors
- Calculation accuracy: +/- 3%
- Unit tests: 85%+ coverage

---

### Task S1-005: Scope 1 Aggregator

**Priority:** P0-CRITICAL
**Duration:** 0.5 week
**Dependencies:** Tasks S1-001 through S1-004

**Description:**
Build aggregator to combine all Scope 1 emission sources into total Scope 1 inventory.

**Calculation Formula:**
```
Scope1_Total = Scope1_Stationary + Scope1_Mobile + Scope1_Fugitive + Scope1_Process
```

**Implementation Details:**
- [ ] Implement Scope 1 category aggregation
- [ ] Support facility-level breakdown
- [ ] Support organizational boundary (equity share vs. operational control)
- [ ] Add year-over-year comparison capability
- [ ] Generate consolidated audit trail
- [ ] Create Scope 1 summary report

**Acceptance Criteria:**
- Sum validation (total equals sum of components)
- Support multiple organizational boundaries
- Unit tests: 95%+ coverage

---

### Task S1-006: ERP Integration - Fuel Consumption Data

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** SAP/Oracle connectors (from GL-VCCI-APP)

**Description:**
Integrate with ERP systems to automatically extract fuel consumption data for Scope 1 calculations.

**Implementation Details:**
- [ ] Leverage GL-VCCI-APP SAP S/4HANA connector
- [ ] Leverage GL-VCCI-APP Oracle Fusion connector
- [ ] Map ERP cost centers to emission sources
- [ ] Extract fuel purchase records
- [ ] Extract fleet fuel card data
- [ ] Implement data validation rules
- [ ] Add manual data entry fallback

**Acceptance Criteria:**
- Automated data extraction from SAP/Oracle
- Data validation with error reporting
- Manual override capability

---

### Task S1-007: California Facility Registry

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** CARB facility database

**Description:**
Build California facility registry to track facilities doing business in California for SB 253 applicability.

**Implementation Details:**
- [ ] Create facility data model (name, address, NAICS, California presence)
- [ ] Implement California revenue tracking
- [ ] Add $1B revenue threshold validation
- [ ] Integrate CARB facility registry (if available)
- [ ] Support multi-state facility tracking

**Acceptance Criteria:**
- Track California-based facilities
- Validate revenue threshold applicability
- Support organizational hierarchy

---

### Task S1-008: Scope 1 Golden Tests

**Priority:** P0-CRITICAL
**Duration:** 0.5 week
**Dependencies:** Tasks S1-001 through S1-005

**Description:**
Create golden test suite for all Scope 1 calculations.

**Implementation Details:**
- [ ] Create 20 golden tests for stationary combustion
- [ ] Create 15 golden tests for mobile combustion
- [ ] Create 10 golden tests for fugitive emissions
- [ ] Create 10 golden tests for process emissions
- [ ] Create 5 golden tests for aggregation
- [ ] Validate against known-good calculations
- [ ] Document test methodology

**Acceptance Criteria:**
- 60+ golden tests for Scope 1
- All tests pass with 100% accuracy
- Tests reviewed by Climate Science Team

---

## SCOPE 2: Energy Indirect Emissions (6 Tasks)

**Definition:** Indirect GHG emissions from purchased electricity, steam, heating, and cooling
**GHG Protocol Reference:** Scope 2 Guidance (2015)
**Key Methods:** Location-based (required) and Market-based (optional)

### Task S2-001: Location-Based Electricity Calculator

**Priority:** P0-CRITICAL
**Duration:** 1 week
**Dependencies:** EPA eGRID database

**Description:**
Build calculator for location-based Scope 2 emissions using grid average emission factors.

**Calculation Formula:**
```
Scope2_Location = Electricity_Consumed[kWh] × Grid_Emission_Factor[kg CO2e/kWh]
```

**Implementation Details:**
- [ ] Integrate EPA eGRID 2023 subregional emission factors
- [ ] Support all 26 US eGRID subregions
- [ ] Implement California-specific CAMX factor (0.254 kg CO2e/kWh)
- [ ] Support facility-to-grid mapping
- [ ] Add historical grid factor support (multi-year reporting)
- [ ] Generate audit trail with grid factor provenance

**California-Specific Emission Factors (EPA eGRID 2023):**
| Grid Region | Factor (kg CO2e/kWh) | States |
|-------------|---------------------|--------|
| CAMX | 0.254 | California |
| AZNM | 0.458 | Arizona, New Mexico |
| NWPP | 0.354 | Pacific Northwest |

**Acceptance Criteria:**
- All 26 eGRID subregions supported
- California CAMX factor accurately applied
- Unit tests: 95%+ coverage

---

### Task S2-002: Market-Based Electricity Calculator

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** REC/PPA tracking

**Description:**
Build calculator for market-based Scope 2 emissions using contractual instruments.

**Calculation Formula:**
```
Scope2_Market = (Total_Electricity - REC_Covered - PPA_Covered) × Residual_Mix_Factor
                + REC_Covered × REC_Factor (typically 0)
                + PPA_Covered × PPA_Factor
```

**Implementation Details:**
- [ ] Implement REC (Renewable Energy Certificate) tracking
- [ ] Implement PPA (Power Purchase Agreement) tracking
- [ ] Support Green-e certified RECs
- [ ] Implement residual mix factor calculation
- [ ] Support supplier-specific emission factors
- [ ] Add contractual instrument validation
- [ ] Generate market-based audit trail

**Market Instruments:**
| Instrument | Emission Factor | Requirement |
|------------|----------------|-------------|
| Green RECs | 0.0 kg CO2e/kWh | Green-e certified |
| Bundled PPA | Supplier-specific | Contract terms |
| Grid default | Residual mix | Location-specific |

**Acceptance Criteria:**
- Support both bundled and unbundled RECs
- Validate contractual instrument documentation
- Unit tests: 90%+ coverage

---

### Task S2-003: Purchased Steam/Heat Calculator

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** District energy emission factors

**Description:**
Build calculator for Scope 2 emissions from purchased steam, heat, and cooling.

**Calculation Formula:**
```
Scope2_Steam = Steam_Consumed[MMBtu] × Steam_Emission_Factor[kg CO2e/MMBtu]
```

**Implementation Details:**
- [ ] Support district heating systems
- [ ] Support industrial steam purchases
- [ ] Implement supplier-specific emission factors
- [ ] Add default factors where supplier data unavailable
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support purchased steam, heat, and cooling
- Default factors for missing supplier data
- Unit tests: 85%+ coverage

---

### Task S2-004: Scope 2 Dual Reporting Module

**Priority:** P0-CRITICAL
**Duration:** 0.5 week
**Dependencies:** Tasks S2-001 and S2-002

**Description:**
Implement dual reporting for both location-based and market-based Scope 2 as required by GHG Protocol.

**Implementation Details:**
- [ ] Generate side-by-side location vs. market comparison
- [ ] Implement GHG Protocol Scope 2 Quality Criteria
- [ ] Validate market-based claims against contractual documentation
- [ ] Add disclosure guidance for each method
- [ ] Create Scope 2 summary report with both methods

**Acceptance Criteria:**
- Both methods reported as per GHG Protocol
- Market-based claims validated
- Clear disclosure of methods used

---

### Task S2-005: Utility Bill Data Integration

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** Utility API connectors

**Description:**
Integrate with utility billing systems to automatically extract electricity consumption data.

**Implementation Details:**
- [ ] Build utility bill parser (PDF, CSV, XML)
- [ ] Integrate California utility APIs (PG&E, SCE, SDG&E)
- [ ] Support Green Button data standard
- [ ] Map utility accounts to facilities
- [ ] Implement data validation rules
- [ ] Add historical data import

**California Major Utilities:**
| Utility | API Support | Data Format |
|---------|-------------|-------------|
| PG&E | Green Button | XML |
| SCE | Green Button | XML |
| SDG&E | Green Button | XML |

**Acceptance Criteria:**
- Automated utility data extraction
- Support Green Button standard
- Data validation with error reporting

---

### Task S2-006: Scope 2 Golden Tests

**Priority:** P0-CRITICAL
**Duration:** 0.5 week
**Dependencies:** Tasks S2-001 through S2-004

**Description:**
Create golden test suite for all Scope 2 calculations.

**Implementation Details:**
- [ ] Create 25 golden tests for location-based method
- [ ] Create 25 golden tests for market-based method
- [ ] Create 10 golden tests for steam/heat
- [ ] Create 10 golden tests for dual reporting
- [ ] Validate California-specific scenarios
- [ ] Document test methodology

**Acceptance Criteria:**
- 70+ golden tests for Scope 2
- All tests pass with 100% accuracy
- Tests reviewed by Climate Science Team

---

## SCOPE 3: Value Chain Emissions (30 Tasks - All 15 Categories)

**Definition:** All other indirect emissions in the company's value chain
**GHG Protocol Reference:** Corporate Value Chain (Scope 3) Standard
**Note:** Scope 3 typically represents 70-90% of total emissions

### Category 1: Purchased Goods and Services (2 Tasks)

#### Task S3-01-001: Spend-Based Calculator (Category 1)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** EPA EEIO emission factors

**Description:**
Build spend-based calculator for Category 1 emissions using economic input-output analysis.

**Calculation Formula:**
```
Cat1_Emissions = SUM(Spend[$] × EEIO_Factor[kg CO2e/$])
```

**Implementation Details:**
- [ ] Integrate EPA Environmentally-Extended Input-Output (EEIO) factors
- [ ] Map procurement categories to NAICS codes
- [ ] Support supplier-specific factors where available
- [ ] Implement hybrid calculation method (spend + supplier data)
- [ ] Add data quality scoring (GHG Protocol DQI)
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support 100+ NAICS categories
- Hybrid method for improved accuracy
- Unit tests: 85%+ coverage

---

#### Task S3-01-002: Supplier-Specific Data Collection (Category 1)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** Supplier engagement platform

**Description:**
Build supplier data collection workflow for primary Category 1 emissions data.

**Implementation Details:**
- [ ] Create supplier questionnaire template
- [ ] Implement CDP-aligned data collection
- [ ] Support supplier-specific emission factors
- [ ] Build supplier data validation rules
- [ ] Track supplier engagement rate
- [ ] Calculate data quality improvement over time

**Acceptance Criteria:**
- Supplier questionnaire aligned with CDP
- Data validation for supplier responses
- Track supplier engagement metrics

---

### Category 2: Capital Goods (2 Tasks)

#### Task S3-02-001: Capital Goods Calculator (Category 2)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** EPA EEIO factors

**Description:**
Build calculator for Category 2 emissions from capital goods purchases.

**Calculation Formula:**
```
Cat2_Emissions = SUM(CapEx_Spend[$] × EEIO_Factor[kg CO2e/$])
```

**Implementation Details:**
- [ ] Implement spend-based calculation for capital goods
- [ ] Map CapEx categories to NAICS codes
- [ ] Support amortization approach (spread over useful life)
- [ ] Add asset register integration
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support spend-based and amortization methods
- Integration with asset register
- Unit tests: 85%+ coverage

---

#### Task S3-02-002: Capital Goods Asset Register Integration (Category 2)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** ERP fixed asset module

**Description:**
Integrate with ERP fixed asset module to extract capital goods data.

**Implementation Details:**
- [ ] Connect to SAP/Oracle fixed asset module
- [ ] Extract capital goods purchases by category
- [ ] Map asset categories to emission factors
- [ ] Support depreciation schedule integration
- [ ] Add manual data entry fallback

**Acceptance Criteria:**
- Automated CapEx extraction from ERP
- Asset category mapping
- Data validation

---

### Category 3: Fuel and Energy-Related Activities (2 Tasks)

#### Task S3-03-001: Upstream Energy Calculator (Category 3)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** Well-to-tank emission factors

**Description:**
Build calculator for Category 3 emissions from upstream fuel and energy activities.

**Calculation Formula:**
```
Cat3_Emissions = (Fuel_Consumed × WTT_Factor) + (Electricity_Consumed × T&D_Loss_Factor × Grid_Factor)
```

**Implementation Details:**
- [ ] Implement well-to-tank (WTT) factors for all fuels
- [ ] Implement transmission & distribution (T&D) loss factors
- [ ] Support California-specific T&D losses (5-7%)
- [ ] Add generation emissions from electricity sold to end users
- [ ] Generate audit trail

**Well-to-Tank Factors (DEFRA 2024):**
| Fuel | WTT Factor | Unit |
|------|-----------|------|
| Natural Gas | 0.62 | kg CO2e/therm |
| Diesel | 0.61 | kg CO2e/gallon |
| Gasoline | 0.52 | kg CO2e/gallon |

**Acceptance Criteria:**
- WTT factors for all Scope 1 fuels
- T&D loss factors by region
- Unit tests: 90%+ coverage

---

#### Task S3-03-002: T&D Loss Calculator (Category 3)

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** Grid T&D loss data

**Description:**
Calculate emissions from transmission and distribution losses for purchased electricity.

**Implementation Details:**
- [ ] Integrate California-specific T&D loss rates
- [ ] Calculate T&D loss emissions by grid region
- [ ] Support multi-grid facilities
- [ ] Generate audit trail

**Acceptance Criteria:**
- California T&D loss rates accurate
- Grid region mapping
- Unit tests: 85%+ coverage

---

### Category 4: Upstream Transportation and Distribution (2 Tasks)

#### Task S3-04-001: Inbound Logistics Calculator (Category 4)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** GLEC Framework emission factors

**Description:**
Build calculator for Category 4 emissions from upstream transportation and distribution.

**Calculation Formula:**
```
Cat4_Emissions = SUM(Shipment_Weight[tonnes] × Distance[km] × Mode_Factor[kg CO2e/tonne-km])
```

**Implementation Details:**
- [ ] Integrate GLEC Framework emission factors
- [ ] Support transportation modes (road, rail, air, sea)
- [ ] Implement distance-based calculation
- [ ] Support spend-based fallback
- [ ] Add third-party logistics provider integration
- [ ] Generate audit trail

**GLEC Framework Factors:**
| Mode | Factor (kg CO2e/tonne-km) | Source |
|------|---------------------------|--------|
| Road (truck) | 0.062 | GLEC |
| Rail | 0.024 | GLEC |
| Air freight | 0.602 | GLEC |
| Sea (container) | 0.011 | GLEC |

**Acceptance Criteria:**
- Support all transportation modes
- Distance-based and spend-based methods
- Unit tests: 85%+ coverage

---

#### Task S3-04-002: Freight Data Integration (Category 4)

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** TMS/WMS connectors

**Description:**
Integrate with transportation management systems to extract shipment data.

**Implementation Details:**
- [ ] Build TMS connector (SAP TM, Oracle TMS)
- [ ] Extract shipment records (origin, destination, weight, mode)
- [ ] Calculate distances using routing APIs
- [ ] Support carrier emission reports
- [ ] Add manual data entry fallback

**Acceptance Criteria:**
- Automated shipment data extraction
- Distance calculation accuracy
- Data validation

---

### Category 5: Waste Generated in Operations (2 Tasks)

#### Task S3-05-001: Operational Waste Calculator (Category 5)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** EPA WARM model factors

**Description:**
Build calculator for Category 5 emissions from waste generated in operations.

**Calculation Formula:**
```
Cat5_Emissions = SUM(Waste_Quantity[tonnes] × Disposal_Factor[kg CO2e/tonne])
```

**Implementation Details:**
- [ ] Integrate EPA WARM model emission factors
- [ ] Support waste types (landfill, recycling, composting, incineration)
- [ ] Support waste streams (general, hazardous, e-waste)
- [ ] Implement waste composition breakdown
- [ ] Generate audit trail

**EPA WARM Factors (2024):**
| Disposal Method | Factor (kg CO2e/tonne) |
|-----------------|------------------------|
| Landfill (mixed MSW) | 520 |
| Recycling (paper) | -1,020 |
| Composting | -210 |
| Incineration | 50 |

**Acceptance Criteria:**
- Support all disposal methods
- Waste composition tracking
- Unit tests: 85%+ coverage

---

#### Task S3-05-002: Waste Management Data Integration (Category 5)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Waste hauler reports

**Description:**
Integrate with waste management providers to extract disposal data.

**Implementation Details:**
- [ ] Build waste hauler data parser
- [ ] Extract waste manifests by type and disposal
- [ ] Support waste audit integration
- [ ] Add manual data entry fallback
- [ ] Track waste diversion rates

**Acceptance Criteria:**
- Automated waste data extraction
- Waste diversion tracking
- Data validation

---

### Category 6: Business Travel (2 Tasks)

#### Task S3-06-001: Business Travel Calculator (Category 6)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** DEFRA travel emission factors

**Description:**
Build calculator for Category 6 emissions from business travel.

**Calculation Formula:**
```
Cat6_Emissions = SUM(Distance[km] × Passenger × Mode_Factor[kg CO2e/passenger-km])
```

**Implementation Details:**
- [ ] Integrate DEFRA 2024 travel emission factors
- [ ] Support air travel (domestic, short-haul, long-haul by class)
- [ ] Support rail travel
- [ ] Support rental cars and taxis
- [ ] Include radiative forcing for air travel (1.891 multiplier)
- [ ] Support hotel stays
- [ ] Generate audit trail

**DEFRA Air Travel Factors (2024, includes RF):**
| Flight Type | Class | Factor (kg CO2e/passenger-km) |
|-------------|-------|-------------------------------|
| Domestic | Economy | 0.255 |
| Short-haul | Economy | 0.156 |
| Long-haul | Economy | 0.147 |
| Long-haul | Business | 0.441 |

**Acceptance Criteria:**
- Support all travel modes
- Radiative forcing included for air
- Unit tests: 90%+ coverage

---

#### Task S3-06-002: Travel Booking System Integration (Category 6)

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** TMC/Concur integration

**Description:**
Integrate with travel management systems to extract booking data.

**Implementation Details:**
- [ ] Build Concur/SAP Travel connector
- [ ] Extract flight bookings with route and class
- [ ] Extract hotel stays
- [ ] Extract rental car bookings
- [ ] Support expense report integration
- [ ] Add manual data entry fallback

**Acceptance Criteria:**
- Automated travel data extraction
- Route and class capture for flights
- Data validation

---

### Category 7: Employee Commuting (2 Tasks)

#### Task S3-07-001: Employee Commuting Calculator (Category 7)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** Commuting survey data

**Description:**
Build calculator for Category 7 emissions from employee commuting.

**Calculation Formula:**
```
Cat7_Emissions = Employees × Average_Commute_Distance × Days_Worked × Mode_Factor
```

**Implementation Details:**
- [ ] Implement survey-based calculation method
- [ ] Implement average-data calculation method
- [ ] Support commute modes (car, public transit, cycling, remote work)
- [ ] Adjust for remote work policies (post-COVID)
- [ ] Support California-specific commute patterns
- [ ] Generate audit trail

**Commuting Emission Factors:**
| Mode | Factor (kg CO2e/passenger-km) |
|------|-------------------------------|
| Car (gasoline) | 0.192 |
| Car (hybrid) | 0.107 |
| Car (EV) | 0.053 |
| Bus | 0.103 |
| Rail | 0.041 |

**Acceptance Criteria:**
- Survey and average-data methods
- Remote work adjustment
- Unit tests: 85%+ coverage

---

#### Task S3-07-002: Employee Commuting Survey Tool (Category 7)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** HR system integration

**Description:**
Build employee commuting survey tool for primary data collection.

**Implementation Details:**
- [ ] Create commuting survey questionnaire
- [ ] Support commute mode, distance, frequency
- [ ] Calculate response rate and data quality
- [ ] Generate anonymized commuting profile
- [ ] Support annual survey cadence

**Acceptance Criteria:**
- Survey captures mode, distance, frequency
- Data quality scoring
- Privacy-compliant aggregation

---

### Category 8: Upstream Leased Assets (2 Tasks)

#### Task S3-08-001: Upstream Leased Assets Calculator (Category 8)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Lease asset register

**Description:**
Build calculator for Category 8 emissions from upstream leased assets not in Scope 1/2.

**Calculation Formula:**
```
Cat8_Emissions = Leased_Space[sqft] × Energy_Intensity[kWh/sqft] × Grid_Factor
```

**Implementation Details:**
- [ ] Implement asset-based calculation method
- [ ] Support leased buildings, vehicles, equipment
- [ ] Integrate with lease management system
- [ ] Apply appropriate organizational boundary
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support leased buildings and equipment
- Organizational boundary application
- Unit tests: 85%+ coverage

---

#### Task S3-08-002: Lease Management Integration (Category 8)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Real estate management system

**Description:**
Integrate with lease management systems to extract leased asset data.

**Implementation Details:**
- [ ] Connect to real estate management system
- [ ] Extract leased space by location
- [ ] Map leased assets to energy profiles
- [ ] Support IFRS 16 lease accounting data
- [ ] Add manual data entry fallback

**Acceptance Criteria:**
- Automated lease data extraction
- Asset energy profiling
- Data validation

---

### Category 9: Downstream Transportation and Distribution (2 Tasks)

#### Task S3-09-001: Outbound Logistics Calculator (Category 9)

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** Task S3-04-001 (reuse methodology)

**Description:**
Build calculator for Category 9 emissions from downstream transportation of sold products.

**Calculation Formula:**
```
Cat9_Emissions = SUM(Product_Weight[tonnes] × Distance[km] × Mode_Factor[kg CO2e/tonne-km])
```

**Implementation Details:**
- [ ] Reuse Category 4 transportation calculator
- [ ] Support customer delivery tracking
- [ ] Implement allocation methods for shared logistics
- [ ] Generate audit trail

**Acceptance Criteria:**
- Reuse Category 4 methodology
- Customer delivery tracking
- Unit tests: 85%+ coverage

---

#### Task S3-09-002: Customer Delivery Data Integration (Category 9)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Order management system

**Description:**
Integrate with order management to extract customer delivery data.

**Implementation Details:**
- [ ] Connect to order management system
- [ ] Extract delivery addresses and weights
- [ ] Calculate delivery distances
- [ ] Support carrier tracking data
- [ ] Add estimation for unknown routes

**Acceptance Criteria:**
- Automated delivery data extraction
- Distance estimation for unknown routes
- Data validation

---

### Category 10: Processing of Sold Products (2 Tasks)

#### Task S3-10-001: Downstream Processing Calculator (Category 10)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Product processing data

**Description:**
Build calculator for Category 10 emissions from processing of intermediate products sold.

**Calculation Formula:**
```
Cat10_Emissions = Product_Sold[units] × Processing_Factor[kg CO2e/unit]
```

**Implementation Details:**
- [ ] Implement product-based calculation method
- [ ] Support industry-specific processing factors
- [ ] Identify products requiring downstream processing
- [ ] Apply appropriate allocation methods
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support intermediate products
- Industry-specific factors
- Unit tests: 80%+ coverage

---

#### Task S3-10-002: Product Processing Mapping (Category 10)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Product master data

**Description:**
Map sold products to downstream processing requirements.

**Implementation Details:**
- [ ] Create product processing taxonomy
- [ ] Map SKUs to processing categories
- [ ] Estimate processing energy requirements
- [ ] Support customer-provided processing data
- [ ] Document assumptions

**Acceptance Criteria:**
- Product taxonomy coverage
- Processing energy estimates
- Documented assumptions

---

### Category 11: Use of Sold Products (2 Tasks)

#### Task S3-11-001: Product Use Phase Calculator (Category 11)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** Product energy consumption data

**Description:**
Build calculator for Category 11 emissions from use of sold products.

**Calculation Formula:**
```
Cat11_Emissions = Units_Sold × Energy_Per_Use × Uses_Per_Year × Product_Life × Grid_Factor
```

**Implementation Details:**
- [ ] Implement direct use-phase calculation (energy-using products)
- [ ] Implement indirect use-phase calculation (fuels, feedstocks)
- [ ] Support product-specific lifetime assumptions
- [ ] Apply geography-specific grid factors for electricity use
- [ ] Generate audit trail

**Example (Electronics):**
```
10,000 laptops × 50 kWh/year × 5 years × 0.417 kg CO2e/kWh = 1,042.5 tonnes CO2e
```

**Acceptance Criteria:**
- Support energy-using products
- Product lifetime assumptions documented
- Unit tests: 85%+ coverage

---

#### Task S3-11-002: Product Energy Profile Database (Category 11)

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** Product specifications

**Description:**
Build database of product energy consumption profiles.

**Implementation Details:**
- [ ] Create product energy profile schema
- [ ] Populate profiles for major product categories
- [ ] Support ENERGY STAR ratings integration
- [ ] Allow product-specific customization
- [ ] Document data sources and assumptions

**Acceptance Criteria:**
- Product energy profiles populated
- ENERGY STAR integration
- Documented assumptions

---

### Category 12: End-of-Life Treatment of Sold Products (2 Tasks)

#### Task S3-12-001: End-of-Life Calculator (Category 12)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** EPA WARM model factors

**Description:**
Build calculator for Category 12 emissions from end-of-life treatment of sold products.

**Calculation Formula:**
```
Cat12_Emissions = Units_Sold × Product_Weight × EOL_Factor
```

**Implementation Details:**
- [ ] Reuse Category 5 waste factors
- [ ] Support product material composition
- [ ] Apply regional recycling rates
- [ ] Support take-back program data
- [ ] Generate audit trail

**Acceptance Criteria:**
- Product material composition
- Regional recycling rates
- Unit tests: 80%+ coverage

---

#### Task S3-12-002: Product Material Composition Database (Category 12)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Bill of materials data

**Description:**
Build database of product material compositions for EOL calculations.

**Implementation Details:**
- [ ] Create material composition schema
- [ ] Map products to material percentages
- [ ] Integrate BOM data where available
- [ ] Support industry-average compositions
- [ ] Document data sources

**Acceptance Criteria:**
- Material compositions populated
- BOM integration
- Documented assumptions

---

### Category 13: Downstream Leased Assets (1 Task)

#### Task S3-13-001: Downstream Leased Assets Calculator (Category 13)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Asset register

**Description:**
Build calculator for Category 13 emissions from assets owned and leased to others.

**Implementation Details:**
- [ ] Identify company-owned assets leased to third parties
- [ ] Apply asset-based calculation method
- [ ] Support building, vehicle, and equipment leases
- [ ] Apply lessor organizational boundary
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support leased-out assets
- Lessor boundary application
- Unit tests: 80%+ coverage

---

### Category 14: Franchises (1 Task)

#### Task S3-14-001: Franchise Emissions Calculator (Category 14)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Franchise data

**Description:**
Build calculator for Category 14 emissions from franchises.

**Implementation Details:**
- [ ] Identify franchise operations (not in Scope 1/2)
- [ ] Apply franchise-specific or average emission factors
- [ ] Support franchise data collection workflow
- [ ] Implement franchisor vs. franchisee boundary
- [ ] Generate audit trail

**Acceptance Criteria:**
- Support franchise operations
- Franchisor boundary application
- Unit tests: 80%+ coverage

---

### Category 15: Investments (2 Tasks)

#### Task S3-15-001: Financed Emissions Calculator (Category 15)

**Priority:** P1-HIGH
**Duration:** 1 week
**Dependencies:** PCAF methodology

**Description:**
Build calculator for Category 15 emissions from investments (equity, debt, project finance).

**Calculation Formula:**
```
Cat15_Emissions = Investment_Amount × Attribution_Factor × Investee_Emissions
```

**Implementation Details:**
- [ ] Integrate PCAF (Partnership for Carbon Accounting Financials) methodology
- [ ] Support equity investments
- [ ] Support debt investments
- [ ] Support project finance
- [ ] Calculate attribution factors based on ownership/financing share
- [ ] Generate audit trail

**PCAF Asset Classes:**
| Asset Class | Attribution Factor |
|-------------|-------------------|
| Listed equity | Ownership % |
| Corporate bonds | Financing % |
| Project finance | Financing % |

**Acceptance Criteria:**
- PCAF methodology implemented
- Support major asset classes
- Unit tests: 85%+ coverage

---

#### Task S3-15-002: Investment Portfolio Integration (Category 15)

**Priority:** P2-MEDIUM
**Duration:** 0.5 week
**Dependencies:** Investment management system

**Description:**
Integrate with investment management systems to extract portfolio data.

**Implementation Details:**
- [ ] Connect to investment management system
- [ ] Extract portfolio holdings and values
- [ ] Link investees to emission profiles (CDP, SBTI)
- [ ] Support manual investee data entry
- [ ] Track data coverage and quality

**Acceptance Criteria:**
- Automated portfolio data extraction
- Investee emission profiles
- Data quality tracking

---

## VERIFICATION & ASSURANCE (6 Tasks)

### Task V-001: Audit Trail Generation

**Priority:** P0-CRITICAL
**Duration:** 1 week
**Dependencies:** All calculation tasks complete

**Description:**
Generate comprehensive audit trails for all emission calculations to support third-party assurance.

**Implementation Details:**
- [ ] Implement SHA-256 provenance hashing for all calculations
- [ ] Track input data sources with timestamps
- [ ] Track emission factor sources with versions
- [ ] Track calculation methodology applied
- [ ] Generate audit trail JSON for each calculation
- [ ] Support audit trail export (JSON, Excel)

**Audit Trail Schema:**
```json
{
  "calculation_id": "string",
  "timestamp": "ISO8601",
  "scope": "1|2|3",
  "category": "string",
  "input_data": {
    "source": "string",
    "hash": "SHA256",
    "timestamp": "ISO8601"
  },
  "emission_factor": {
    "factor_id": "string",
    "value": "number",
    "unit": "string",
    "source": "EPA|DEFRA|IPCC",
    "source_uri": "URL",
    "version": "string"
  },
  "calculation": {
    "methodology": "string",
    "formula": "string",
    "result_kg_co2e": "number"
  },
  "provenance_hash": "SHA256"
}
```

**Acceptance Criteria:**
- SHA-256 hashing for all calculations
- Complete provenance chain
- Third-party verifiable

---

### Task V-002: Third-Party Assurance Package Generator

**Priority:** P0-CRITICAL
**Duration:** 1 week
**Dependencies:** Task V-001

**Description:**
Generate assurance packages for third-party verification (Big 4 audit firms).

**Implementation Details:**
- [ ] Create assurance package template (ISAE 3410 aligned)
- [ ] Include all audit trails for reporting period
- [ ] Include emission factor documentation
- [ ] Include calculation methodology documentation
- [ ] Include data quality assessment
- [ ] Generate assurance-ready PDF report
- [ ] Support Big 4 audit requirements (Deloitte, EY, PwC, KPMG)

**Assurance Package Contents:**
1. Executive Summary
2. GHG Inventory (Scope 1, 2, 3)
3. Methodology Description
4. Emission Factor Sources
5. Data Quality Assessment
6. Audit Trails (full detail)
7. Organizational Boundary Definition
8. Completeness Assessment

**Acceptance Criteria:**
- ISAE 3410 alignment
- Big 4 audit requirements met
- Complete documentation package

---

### Task V-003: Data Quality Indicator (DQI) Scoring

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** GHG Protocol DQI framework

**Description:**
Implement GHG Protocol Data Quality Indicator scoring for all Scope 3 categories.

**Implementation Details:**
- [ ] Implement 5-dimension DQI scoring (temporal, geographical, technological, completeness, reliability)
- [ ] Calculate DQI score for each Scope 3 category
- [ ] Generate DQI improvement recommendations
- [ ] Track DQI improvement over time
- [ ] Support DQI disclosure in reports

**DQI Scoring Dimensions:**
| Dimension | Score 1 (Best) | Score 5 (Worst) |
|-----------|----------------|-----------------|
| Temporal | Same year | >10 years old |
| Geographical | Same region | Different continent |
| Technological | Same process | Generic average |
| Completeness | All data points | <50% data |
| Reliability | Verified data | Unverified estimate |

**Acceptance Criteria:**
- 5-dimension DQI scoring
- Category-level DQI tracking
- Improvement recommendations

---

### Task V-004: Limited vs. Reasonable Assurance Tracker

**Priority:** P1-HIGH
**Duration:** 0.5 week
**Dependencies:** SB 253 assurance requirements

**Description:**
Track assurance level requirements and readiness for SB 253 compliance.

**Implementation Details:**
- [ ] Track limited assurance requirements (2026-2029)
- [ ] Track reasonable assurance requirements (2030+)
- [ ] Assess readiness for each assurance level
- [ ] Generate assurance gap analysis
- [ ] Create assurance roadmap

**SB 253 Assurance Timeline:**
| Year | Assurance Level | Scope Coverage |
|------|-----------------|----------------|
| 2026 | Limited | Scope 1, 2 |
| 2027 | Limited | Scope 1, 2, 3 |
| 2030 | Reasonable | Scope 1, 2, 3 |

**Acceptance Criteria:**
- Assurance level tracking
- Readiness assessment
- Gap analysis

---

### Task V-005: CARB Portal Integration

**Priority:** P0-CRITICAL
**Duration:** 2 weeks
**Dependencies:** CARB API specification (when available)

**Description:**
Integrate with California Air Resources Board (CARB) portal for SB 253 filing.

**Implementation Details:**
- [ ] Research CARB SB 253 reporting portal requirements
- [ ] Implement CARB data schema mapping
- [ ] Build filing XML/JSON generator
- [ ] Support CARB portal API integration (when available)
- [ ] Implement filing status tracking
- [ ] Generate filing confirmation receipts

**Note:** CARB portal specifications pending. Monitor CARB announcements for updates.

**Acceptance Criteria:**
- CARB schema compliance
- Filing automation
- Status tracking

---

### Task V-006: Verification Golden Tests

**Priority:** P0-CRITICAL
**Duration:** 0.5 week
**Dependencies:** Tasks V-001 through V-005

**Description:**
Create golden test suite for verification and assurance features.

**Implementation Details:**
- [ ] Create 20 golden tests for audit trail generation
- [ ] Create 15 golden tests for assurance package generation
- [ ] Create 10 golden tests for DQI scoring
- [ ] Create 5 golden tests for CARB filing
- [ ] Validate assurance package completeness
- [ ] Document test methodology

**Acceptance Criteria:**
- 50+ golden tests for verification
- All tests pass with 100% accuracy
- Tests reviewed by Climate Science Team

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-8)
- **Weeks 1-4:** Scope 1 tasks (S1-001 through S1-008)
- **Weeks 5-8:** Scope 2 tasks (S2-001 through S2-006)

### Phase 2: Value Chain (Weeks 9-24)
- **Weeks 9-12:** Scope 3 Categories 1-5
- **Weeks 13-16:** Scope 3 Categories 6-10
- **Weeks 17-20:** Scope 3 Categories 11-15
- **Weeks 21-24:** Scope 3 integration and testing

### Phase 3: Verification (Weeks 25-32)
- **Weeks 25-28:** Verification tasks (V-001 through V-003)
- **Weeks 29-32:** Assurance and CARB integration (V-004 through V-006)

### Phase 4: Launch (Weeks 33-36)
- **Weeks 33-34:** Beta testing with pilot companies
- **Weeks 35-36:** Production deployment and monitoring

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Scope 1 calculation accuracy | +/- 1% | Benchmark comparison |
| Scope 2 calculation accuracy | +/- 2% | Benchmark comparison |
| Scope 3 calculation accuracy | +/- 5% | Benchmark comparison |
| Audit trail completeness | 100% | Provenance coverage |
| Golden test pass rate | 100% | Automated testing |
| Third-party assurance readiness | Yes | Big 4 audit review |
| CARB filing compliance | 100% | Schema validation |
| Total golden tests | 300+ | Test count |

---

## Dependencies

### GL-VCCI-APP Leverage (55% Complete)
- Scope 1 calculation engines: **Reuse**
- Scope 2 calculation engines: **Reuse**
- Scope 3 calculation engines (all 15 categories): **Reuse**
- ERP connectors (SAP, Oracle, Workday): **Reuse**
- Provenance tracking (SHA-256 audit trails): **Reuse**
- California-specific compliance engine: **New**
- CARB portal integration: **New**
- Third-party assurance package generation: **New**

### External Dependencies
| Dependency | Status | Impact |
|------------|--------|--------|
| EPA eGRID 2023 | Available | Scope 2 grid factors |
| EPA GHG EF Hub | Available | Scope 1 fuel factors |
| DEFRA 2024 | Available | Travel and Scope 3 factors |
| GLEC Framework | Available | Transportation factors |
| PCAF Methodology | Available | Financed emissions |
| CARB SB 253 Portal | Pending | Filing integration |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CARB portal delayed | High | Medium | Manual filing fallback |
| Data quality issues | Medium | High | DQI scoring and improvement |
| Supplier data gaps | High | Medium | Spend-based fallback |
| Assurance requirements unclear | Medium | Medium | Big 4 early engagement |
| California grid factor changes | Low | Low | Version-controlled factors |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-SB253-PM | Initial 50-task implementation plan |

**Approvals:**

- Climate Science Team Lead: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______
- Program Director: ___________________ Date: _______

---

**END OF DOCUMENT - Total Tasks: 50**

### Task Summary by Scope

| Scope | Category | Tasks |
|-------|----------|-------|
| **Scope 1** | Direct Emissions | 8 |
| **Scope 2** | Energy Indirect | 6 |
| **Scope 3** | Purchased Goods (Cat 1) | 2 |
| **Scope 3** | Capital Goods (Cat 2) | 2 |
| **Scope 3** | Fuel/Energy Activities (Cat 3) | 2 |
| **Scope 3** | Upstream Transport (Cat 4) | 2 |
| **Scope 3** | Waste Generated (Cat 5) | 2 |
| **Scope 3** | Business Travel (Cat 6) | 2 |
| **Scope 3** | Employee Commuting (Cat 7) | 2 |
| **Scope 3** | Upstream Leased (Cat 8) | 2 |
| **Scope 3** | Downstream Transport (Cat 9) | 2 |
| **Scope 3** | Processing of Sold (Cat 10) | 2 |
| **Scope 3** | Use of Sold (Cat 11) | 2 |
| **Scope 3** | End-of-Life (Cat 12) | 2 |
| **Scope 3** | Downstream Leased (Cat 13) | 1 |
| **Scope 3** | Franchises (Cat 14) | 1 |
| **Scope 3** | Investments (Cat 15) | 2 |
| **Verification** | Assurance & Compliance | 6 |
| **TOTAL** | | **50** |
