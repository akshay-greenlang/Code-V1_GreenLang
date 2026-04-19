# PRD: AGENT-MRV-019 -- Scope 3 Category 6 Business Travel Agent

---

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-006 |
| **Internal Label** | AGENT-MRV-019 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/business_travel/` |
| **DB Migration** | V070 |
| **Metrics Prefix** | `gl_bt_` |
| **Table Prefix** | `gl_bt_` |
| **API** | `/api/v1/business-travel` |
| **Env Prefix** | `GL_BT_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The **GL-MRV-S3-006 Business Travel Agent** implements GHG Protocol Scope 3 Category 6 emissions accounting for all business-related travel activities undertaken by employees for work purposes in vehicles not owned or operated by the reporting company. This agent automates the calculation of greenhouse gas emissions from air travel, rail travel, ground transportation (rental cars, taxis, ride-shares, buses), and hotel/accommodation stays during business trips.

Category 6 covers the following sub-activities as defined in the GHG Protocol Scope 3 Standard (Chapter 7):

- **Air Travel (6a)** -- Emissions from commercial flights including short-haul (<500 km), medium-haul (500-3,700 km), and long-haul (>3,700 km) with cabin class differentiation (economy, premium economy, business, first) and optional radiative forcing index (RFI) for aviation's non-CO2 climate effects
- **Rail Travel (6b)** -- Emissions from intercity rail, commuter rail, light rail, subway/metro, and high-speed rail services
- **Ground Transport (6c)** -- Emissions from rental cars, taxis, ride-shares, company car mileage reimbursement, bus/coach, and ferry travel
- **Hotel Stays (6d)** -- Emissions from accommodation during business trips, differentiated by hotel class, country, and number of nights

Business travel typically represents **2-15% of total Scope 3 emissions** depending on industry sector, with professional services, consulting, and financial services at the higher end. For companies with large, globally distributed workforces, business travel can be one of the most significant and most actionable Scope 3 categories.

The agent automates extraction and calculation of emissions from travel booking systems, expense reports, corporate travel management companies (TMCs), and T&E platforms (Concur, Navan, TripActions), producing audit-ready outputs with full provenance chains, data quality scoring, and multi-framework regulatory compliance.

### Justification for Dedicated Agent

1. **Multi-modal complexity** -- Business travel spans 4+ transport modes (air, rail, road, accommodation), each with distinct emission factor databases, calculation methodologies, and data source patterns
2. **Aviation-specific methodology** -- Air travel requires specialized distance calculations (great-circle distance with uplift), cabin class allocation, radiative forcing multipliers, and ICAO/DEFRA-specific emission factor hierarchies that no other Category agent needs
3. **Cabin class allocation** -- Unique multiplier system (economy=1.0x, premium economy=1.6x, business=2.9x, first=4.0x) based on seat area/weight allocation that requires dedicated logic
4. **Radiative forcing decision** -- Aviation's non-CO2 effects (contrails, NOx, water vapor) require configurable RFI multipliers (1.0-2.7) with framework-specific requirements (SBTi recommends inclusion, CDP requests separate disclosure)
5. **Double-counting prevention** -- Complex boundaries with Category 4 (upstream transport of purchased goods), Category 7 (employee commuting), Category 9 (downstream transport), and Scope 1/2 (company-owned/leased vehicles)
6. **Hotel emissions methodology** -- Accommodation emissions use country-specific and class-specific emission factors per room-night, distinct from any transport methodology
7. **Spend-based fallback** -- When detailed trip data is unavailable, EEIO spend-based calculation using T&E expense categories requires dedicated mapping logic
8. **Regulatory urgency** -- CSRD ESRS E1, CDP, SBTi, and California SB 253 all require Category 6 disclosure with increasing granularity requirements

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard, Chapter 7 (Category 6)
- GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 6: Business Travel
- GHG Protocol Scope 3 Calculation Guidance -- Appendix: Emission Factors for Business Travel
- DEFRA/DESNZ 2024 Government GHG Conversion Factors for Company Reporting -- Business Travel (Tables 8-10)
- EPA Center for Corporate Climate Leadership -- Emission Factors Hub: Business Travel
- ICAO Carbon Emissions Calculator Methodology (Version 12, 2024)
- IPCC Sixth Assessment Report (AR6) -- Global Warming Potentials (Table 7.15)
- CSRD / ESRS E1 -- Climate Change (paragraphs 48-56, AR 46-48)
- California SB 253 Climate Corporate Data Accountability Act -- Scope 3 Disclosure
- CDP Climate Change Questionnaire 2024 -- C6.5 Business Travel
- SBTi Corporate Net-Zero Standard v1.2 -- Scope 3 Boundary Requirements
- GRI 305: Emissions 2016 -- Disclosure 305-3
- ISO 14064-1:2018 -- Quantification and reporting of GHG emissions and removals
- IEA CO2 Emissions from Fuel Combustion (2024 Edition) -- Transport Sector
- UIC/CER Rail Transport and Environment: Facts and Figures (2024)
- GLEC Framework v3.0 -- Global Logistics Emissions Council
- IATA Aviation Carbon Exchange -- Passenger Allocation Methodology

### Terminology

| Term | Definition | Scope Mapping |
|------|-----------|---------------|
| Great-Circle Distance (GCD) | Shortest distance between two points on Earth's surface along a sphere, used as base flight distance | Air travel distance calculation |
| Uplift Factor | Percentage added to GCD to account for non-direct routing, holding patterns, and detours (typically 8-9%) | Air travel distance correction |
| Radiative Forcing Index (RFI) | Multiplier (1.0-2.7) applied to aviation CO2 to account for non-CO2 climate effects (contrails, NOx, water vapor) | Aviation non-CO2 effects |
| Cabin Class Multiplier | Factor reflecting per-passenger floor area/weight allocation by cabin class relative to economy | Air travel allocation |
| Short-Haul Flight | Flight distance < 500 km (domestic/regional) | DEFRA distance band |
| Medium-Haul Flight | Flight distance 500-3,700 km | DEFRA distance band |
| Long-Haul Flight | Flight distance > 3,700 km (international) | DEFRA distance band |
| With RF | Emission factor including radiative forcing multiplier for aviation | CDP/SBTi preferred |
| Without RF | CO2-only emission factor for aviation (excludes non-CO2 effects) | GHG Protocol base |
| Well-to-Tank (WTT) | Upstream emissions from fuel extraction, refining, and distribution | Supply chain emissions |
| Tank-to-Wheel (TTW) | Direct emissions from fuel combustion in the vehicle | Operational emissions |
| Well-to-Wheel (WTW) | Total lifecycle fuel emissions (WTT + TTW) | ISO 14083 preferred |
| TMC | Travel Management Company (e.g., Amex GBT, BCD, CWT) | Data source |
| T&E | Travel & Expense management system (e.g., Concur, Navan) | Data source |
| Room-Night | Unit of hotel accommodation: one room for one night | Hotel emissions unit |
| EEIO | Environmentally Extended Input-Output model for spend-based estimation | Spend-based method |
| Passenger-km (pkm) | One passenger transported one kilometer | Distance-based unit |
| ICAO | International Civil Aviation Organization | Aviation standards body |
| Load Factor | Average occupancy rate of a vehicle/aircraft (passengers / capacity) | Already embedded in per-pkm EFs |
| CRS/GDS | Computer Reservation System / Global Distribution System (Amadeus, Sabre, Travelport) | Booking data source |
| PNR | Passenger Name Record -- booking reference from airline/GDS | Trip identification |
| Haversine Formula | Trigonometric formula for GCD on a sphere | Distance calculation |
| Vincenty Formula | More accurate ellipsoidal distance formula (WGS-84) | Precision distance calculation |

---

## 2. Methodology

### 2.1 Category Boundary Definition

```
Category 6: Business Travel
├── INCLUDED
│   ├── Air travel on commercial airlines (not company-owned aircraft)
│   ├── Rail travel (intercity, commuter, high-speed, metro)
│   ├── Road transport: rental cars, taxis, ride-shares, buses, coaches
│   ├── Road transport: personal vehicle mileage reimbursed by company
│   ├── Ferry/water transport for business purposes
│   ├── Hotel/accommodation stays during business trips
│   ├── WTT (well-to-tank) emissions for all fuel types
│   └── Radiative forcing from aviation (optional but recommended)
│
├── EXCLUDED (reported elsewhere)
│   ├── Company-owned/leased vehicles → Scope 1
│   ├── Company-operated aircraft → Scope 1
│   ├── Employee commuting (home ↔ office) → Category 7
│   ├── Transport of goods/materials → Category 4 or Category 9
│   ├── Downstream customer travel → Category 9
│   └── Teleworking/remote work energy → Category 7 (if reported)
│
└── BOUNDARY TESTS
    ├── IF vehicle owned/leased by company → Scope 1 (exclude from Cat 6)
    ├── IF trip purpose = commuting → Category 7 (exclude from Cat 6)
    ├── IF trip purpose = goods transport → Category 4/9 (exclude from Cat 6)
    ├── IF personal car + mileage reimbursed → Include in Cat 6
    ├── IF personal car + no reimbursement → Exclude (not business travel)
    └── IF chartered/private jet not owned → Include in Cat 6
```

### 2.2 Calculation Methods

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | **Supplier-specific** | Airline/hotel/rail-specific emission data (EPDs, verified reports) | Highest | Low |
| 2 | **Distance-based** | Origin/destination, mode, class, distance | High | Medium |
| 3 | **Average-data** | Trip count by mode, average distances | Medium | High |
| 4 | **Spend-based** | Travel expense amounts by category | Low | Highest |

```
Method Selection Decision Tree:
─────────────────────────────────────
START
├── Supplier provides verified CO2 data (e.g., airline CO2 report)?
│   ├── YES → Method 1: Supplier-specific
│   └── NO ↓
├── Origin/destination or distance available?
│   ├── YES → Method 2: Distance-based
│   │   ├── Air → DEFRA/ICAO per-pkm by haul & class
│   │   ├── Rail → DEFRA/IEA per-pkm by rail type
│   │   ├── Road → DEFRA per-vkm by vehicle type
│   │   └── Hotel → DEFRA/Cornell per room-night by country
│   └── NO ↓
├── Trip counts and average distances available?
│   ├── YES → Method 3: Average-data
│   │   └── Average trip distance × EF per mode
│   └── NO ↓
└── Only spend data available?
    └── YES → Method 4: Spend-based (EEIO)
        └── Spend × EEIO factor per travel category
```

### 2.3 Air Travel Emissions -- Distance-Based Method

The primary calculation method for air travel follows DEFRA 2024 and ICAO methodology:

```
Step 1: Calculate Great-Circle Distance
  GCD = 2 × R × arcsin(√(sin²((φ₂-φ₁)/2) + cos(φ₁) × cos(φ₂) × sin²((λ₂-λ₁)/2)))
  Where:
    R = Earth's mean radius (6,371 km)
    φ₁, φ₂ = latitude of origin and destination (radians)
    λ₁, λ₂ = longitude of origin and destination (radians)

Step 2: Apply Uplift Factor
  Flight_Distance = GCD × (1 + Uplift_Factor)
  Where:
    Uplift_Factor = 0.08 (8% for DEFRA, per ICAO recommendation)

Step 3: Determine Distance Band
  IF Flight_Distance < 500 km → Short-haul (domestic)
  IF 500 km ≤ Flight_Distance ≤ 3,700 km → Medium-haul
  IF Flight_Distance > 3,700 km → Long-haul (international)

Step 4: Apply Cabin Class Multiplier
  Effective_Passengers = Passengers × Class_Multiplier
  Where:
    Economy = 1.0
    Premium_Economy = 1.6
    Business = 2.9
    First = 4.0

Step 5: Calculate Emissions
  CO2e_without_RF = Flight_Distance × Passengers × Class_Multiplier × EF_per_pkm
  CO2e_with_RF = CO2e_without_RF × RFI
  Where:
    EF_per_pkm = DEFRA 2024 emission factor for distance band (kgCO2e/pkm)
    RFI = Radiative Forcing Index (default: 1.891 per DEFRA with RF factor)

Step 6: Add WTT Emissions
  WTT_CO2e = Flight_Distance × Passengers × Class_Multiplier × WTT_EF_per_pkm
  Total_CO2e = CO2e_with_RF + WTT_CO2e
```

### 2.4 Air Travel Emission Factors (DEFRA 2024)

| Distance Band | Without RF (kgCO2e/pkm) | With RF (kgCO2e/pkm) | WTT (kgCO2e/pkm) |
|--------------|------------------------|---------------------|-------------------|
| Domestic (<500 km) | 0.24587 | 0.27916 | 0.05765 |
| Short-haul (<3,700 km) | 0.15353 | 0.17435 | 0.03600 |
| Long-haul (>3,700 km) | 0.19309 | 0.21932 | 0.04528 |
| International (average) | 0.18362 | 0.20856 | 0.04306 |

**Cabin Class Multipliers (DEFRA 2024):**

| Cabin Class | Multiplier | Basis |
|------------|------------|-------|
| Economy | 1.0 | Base seat area allocation |
| Premium Economy | 1.6 | ~1.6x seat area vs economy |
| Business | 2.9 | ~2.9x seat area vs economy |
| First | 4.0 | ~4.0x seat area vs economy |

### 2.5 Radiative Forcing

Aviation's total climate impact exceeds CO2-only effects due to non-CO2 effects at altitude:

```
Non-CO2 Aviation Effects:
├── Contrails and contrail-induced cirrus clouds
├── NOx emissions → ozone formation (warming) + methane destruction (cooling)
├── Water vapor emissions at cruise altitude
├── Sulfate aerosol emissions (cooling)
└── Soot/black carbon particles (warming)

RFI Application:
  Total_Climate_Impact = CO2_Emissions × RFI
  Where:
    RFI = 1.0 (CO2-only, no RF adjustment)
    RFI = 1.7 (IPCC central estimate, commonly used)
    RFI = 1.891 (DEFRA 2024 embedded RF multiplier ratio)
    RFI = 2.0 (GHG Protocol suggested)
    RFI = 2.7 (IPCC upper bound)

Framework Requirements:
  GHG Protocol: Optional, report separately if included
  DEFRA 2024: Provides both with-RF and without-RF factors
  CDP: Request both with-RF and without-RF, separate disclosure
  SBTi: Recommends including RF in target boundary
  CSRD ESRS E1: Report total Scope 3, RF inclusion at entity discretion
```

### 2.6 Rail Travel Emissions

```
Rail_CO2e = Distance_km × Passengers × EF_per_pkm
WTT_CO2e = Distance_km × Passengers × WTT_EF_per_pkm
Total_CO2e = Rail_CO2e + WTT_CO2e

Where:
  EF_per_pkm = emission factor by rail type (kgCO2e/pkm)
  WTT_EF_per_pkm = well-to-tank emission factor (kgCO2e/pkm)
```

**Rail Emission Factors (DEFRA 2024):**

| Rail Type | TTW (kgCO2e/pkm) | WTT (kgCO2e/pkm) | Total (kgCO2e/pkm) |
|-----------|-------------------|-------------------|---------------------|
| National Rail (average) | 0.03549 | 0.00434 | 0.03983 |
| International Rail | 0.00446 | 0.00086 | 0.00532 |
| Light Rail / Tram | 0.02904 | 0.00612 | 0.03516 |
| London Underground | 0.02781 | 0.00586 | 0.03367 |
| Eurostar | 0.00446 | 0.00086 | 0.00532 |
| High-Speed Rail (TGV) | 0.00324 | 0.00068 | 0.00392 |
| US Intercity Rail (Amtrak) | 0.08900 | 0.01100 | 0.10000 |
| US Commuter Rail | 0.10500 | 0.01300 | 0.11800 |

### 2.7 Road Transport Emissions

```
Distance-Based:
  Road_CO2e = Distance_km × EF_per_vkm
  Where:
    EF_per_vkm = emission factor per vehicle-km by vehicle type

Fuel-Based (rental cars):
  Road_CO2e = Fuel_Litres × EF_per_litre
  Where:
    EF_per_litre = fuel-specific emission factor (kgCO2e/litre)
```

**Road Vehicle Emission Factors (DEFRA 2024):**

| Vehicle Type | kgCO2e/vkm | kgCO2e/pkm | Assumed Occupancy |
|-------------|------------|------------|-------------------|
| Average Car (unknown fuel) | 0.27145 | 0.17082 | 1.59 |
| Small Car (petrol) | 0.20755 | 0.13053 | 1.59 |
| Medium Car (petrol) | 0.25594 | 0.16106 | 1.59 |
| Large Car (petrol) | 0.35388 | 0.22258 | 1.59 |
| Small Car (diesel) | 0.19290 | 0.12132 | 1.59 |
| Medium Car (diesel) | 0.23280 | 0.14642 | 1.59 |
| Large Car (diesel) | 0.29610 | 0.18629 | 1.59 |
| Hybrid Car | 0.17830 | 0.11214 | 1.59 |
| Plug-in Hybrid | 0.10250 | 0.06447 | 1.59 |
| Battery EV | 0.07005 | 0.04406 | 1.59 |
| Taxi (regular) | 0.20920 | 0.14880 | 1.41 |
| Taxi (black cab) | 0.31477 | 0.22378 | 1.41 |
| Bus (local) | -- | 0.10312 | -- |
| Bus (coach) | -- | 0.02732 | -- |
| Motorcycle | 0.11337 | 0.11337 | 1.0 |
| Ferry (foot passenger) | -- | 0.01877 | -- |
| Ferry (car passenger) | -- | 0.12952 | -- |

**Fuel Emission Factors (DEFRA 2024):**

| Fuel Type | kgCO2e/litre | WTT (kgCO2e/litre) |
|-----------|-------------|---------------------|
| Petrol | 2.31480 | 0.58549 |
| Diesel | 2.70370 | 0.60927 |
| LPG | 1.55370 | 0.32149 |
| CNG | 2.53970 (per kg) | 0.50870 (per kg) |
| E85 (bioethanol blend) | 0.34728 | 0.07890 |

### 2.8 Hotel Stay Emissions

```
Hotel_CO2e = Room_Nights × EF_per_room_night
Where:
  EF_per_room_night = country-specific and class-specific factor (kgCO2e/room-night)
```

**Hotel Emission Factors (DEFRA 2024 / Cornell CHSU):**

| Country/Region | kgCO2e/room-night |
|---------------|-------------------|
| United Kingdom | 12.32 |
| United States | 21.12 |
| Canada | 14.40 |
| France | 7.26 |
| Germany | 13.50 |
| Spain | 10.60 |
| Italy | 11.10 |
| Netherlands | 10.00 |
| Japan | 28.85 |
| China | 34.56 |
| India | 22.08 |
| Australia | 25.90 |
| Brazil | 8.28 |
| Singapore | 27.00 |
| UAE | 37.50 |
| Global Average | 20.90 |

### 2.9 Spend-Based Method (EEIO)

```
Spend_CO2e = Spend_Amount × Currency_Conversion × CPI_Deflator × EEIO_Factor
Where:
  Spend_Amount = total spend in reporting currency
  Currency_Conversion = exchange rate to USD
  CPI_Deflator = inflation adjustment to base year (2021)
  EEIO_Factor = EEIO emission factor per USD spend (kgCO2e/$)
```

**EEIO Emission Factors for Business Travel (EPA / USEEIO v2.0):**

| NAICS Code | Category | kgCO2e/$ (2021 USD) |
|-----------|----------|---------------------|
| 481000 | Air transportation | 0.4770 |
| 482000 | Rail transportation | 0.3100 |
| 485000 | Ground passenger transport | 0.2600 |
| 485310 | Taxi/ride-hailing | 0.2800 |
| 532100 | Automotive rental/leasing | 0.1950 |
| 721100 | Hotels and motels | 0.1490 |
| 721200 | RV parks and camps | 0.1200 |
| 722500 | Restaurants (travel meals) | 0.2050 |
| 483000 | Water transportation | 0.5200 |
| 487000 | Scenic/sightseeing transport | 0.3400 |

### 2.10 Data Quality Indicator (DQI)

**5-Dimension DQI Scoring:**

| Dimension | Score 5 (Very High) | Score 4 (High) | Score 3 (Medium) | Score 2 (Low) | Score 1 (Very Low) |
|-----------|--------------------|----|----|----|-----|
| Representativeness | Specific trip data from booking system | Trip-level data from expense reports | Aggregate data by mode and region | Department-level estimates | Company-wide average spend |
| Completeness | 100% of trips captured | >90% captured | >70% captured | >50% captured | <50% captured |
| Temporal | Reporting year data | Within 2 years | Within 3 years | Within 5 years | >5 years old |
| Geographical | Route-specific EFs | Country-specific EFs | Region-specific EFs | Continent-level EFs | Global average EFs |
| Technological | Mode/class/vehicle-specific EFs | Mode-specific EFs | Transport sector EFs | Economy-wide average | EEIO average |

```
Composite DQI = Σ (Dimension_Score × Weight) / Σ Weights
Default weights: Representativeness=0.3, Completeness=0.25, Temporal=0.15, Geographical=0.15, Technological=0.15
```

| Classification | Score Range | Description |
|---------------|------------|-------------|
| Excellent | 4.5 - 5.0 | High confidence, audit-ready |
| Good | 3.5 - 4.4 | Acceptable for reporting |
| Fair | 2.5 - 3.4 | Improvement recommended |
| Poor | 1.5 - 2.4 | Significant gaps |
| Very Poor | 1.0 - 1.4 | Estimate only, not reliable |

### 2.11 Uncertainty Ranges

| Method | DQI Range | Uncertainty (±%) | Confidence Level |
|--------|-----------|-----------------|-----------------|
| Supplier-specific | 4.0-5.0 | ±5-10% | 95% |
| Distance-based (air) | 3.5-4.5 | ±10-20% | 95% |
| Distance-based (rail) | 3.0-4.0 | ±15-25% | 95% |
| Distance-based (road) | 3.0-4.0 | ±15-30% | 95% |
| Hotel (country-specific) | 3.0-4.0 | ±20-35% | 95% |
| Average-data | 2.5-3.5 | ±25-40% | 95% |
| Spend-based (EEIO) | 1.5-2.5 | ±40-60% | 95% |

```
Pedigree Matrix Uncertainty:
  U_total = √(U_activity² + U_ef² + U_method²)

  Where:
    U_activity = activity data uncertainty (trip counts, distances)
    U_ef = emission factor uncertainty (varies by source)
    U_method = calculation method uncertainty (model assumptions)

Combined Uncertainty (Monte Carlo):
  Run N=10,000 simulations
  Sample activity data from lognormal distribution
  Sample EFs from triangular (min, mode, max) distribution
  Report: mean, std_dev, 2.5th/97.5th percentile CI
```

### 2.12 Category Boundaries & Double-Counting Prevention

**Included in Category 6:**

| Activity | Data Source | EF Source |
|----------|-----------|----------|
| Commercial flights (economy/business/first) | Booking data, expense reports | DEFRA, ICAO |
| Rail travel (all types) | Booking data, travel passes | DEFRA, UIC |
| Rental cars | Rental receipts, fuel records | DEFRA, EPA |
| Taxis and ride-shares | Receipts, expense reports | DEFRA |
| Bus/coach travel | Tickets, expense reports | DEFRA |
| Personal car (reimbursed mileage) | Mileage claims | DEFRA |
| Hotel accommodation | Hotel invoices, booking data | DEFRA, Cornell |
| Ferry (business travel) | Tickets | DEFRA |
| Private/chartered flights (not owned) | Charter invoices | DEFRA, ICAO |

**Excluded from Category 6 (reported elsewhere):**

| Activity | Where Reported | Agent |
|----------|---------------|-------|
| Company-owned vehicles | Scope 1 (mobile combustion) | AGENT-MRV-003 |
| Company-owned/leased aircraft | Scope 1 | AGENT-MRV-003 |
| Employee commuting (home ↔ work) | Category 7 | AGENT-MRV-020 |
| Transport of purchased goods | Category 4 | AGENT-MRV-017 |
| Downstream distribution/transport | Category 9 | Planned |
| Telework/WFH energy use | Category 7 (optional) | AGENT-MRV-020 |
| Travel meals (food emissions) | Category 1 | AGENT-MRV-014 |

**Double-Counting Prevention Rules:**

| Rule | Description | Implementation |
|------|-----------|----------------|
| DC-BT-001 | Exclude company-owned/leased vehicles | Check vehicle_ownership != COMPANY_OWNED |
| DC-BT-002 | Exclude commuting trips | Check trip_purpose != COMMUTING |
| DC-BT-003 | Exclude goods transport trips | Check trip_purpose != FREIGHT |
| DC-BT-004 | No overlap with Cat 4 upstream transport | Check against AGENT-MRV-017 shipment IDs |
| DC-BT-005 | No overlap with Scope 1 mobile combustion | Cross-reference fleet vehicle registry |
| DC-BT-006 | Hotel food/beverage separate from room emissions | Use room-only EFs, not full-service |
| DC-BT-007 | WTT emissions not double-counted with Cat 3 | WTT included in Cat 6 EFs, exclude from Cat 3 |

```
Cross-category validation checks:
  IF vehicle in company fleet registry → REJECT (Scope 1)
  IF trip marked as commuting → REDIRECT to Cat 7
  IF trip includes freight → SPLIT passenger / freight portions
  IF airline reports Scope 1 emissions to Cat 4 → Flag overlap
  IF hotel food charges included → STRIP from room-night calculation
```

### 2.13 Coverage & Materiality

| Industry Sector | % of Scope 3 | Primary Drivers |
|----------------|-------------|-----------------|
| Professional Services / Consulting | 10-15% | Client site visits, global teams |
| Financial Services / Banking | 5-12% | Client meetings, conferences |
| Technology / Software | 3-8% | Sales, conferences, support |
| Pharmaceuticals | 3-7% | Clinical trials, regulatory visits |
| Manufacturing | 1-3% | Supplier visits, trade shows |
| Retail | 1-2% | Buying trips, store visits |
| Mining / Oil & Gas | 2-5% | Remote site travel |
| Government / Public Sector | 5-10% | Inter-office, policy meetings |
| Education / Research | 3-8% | Conferences, field research |
| Healthcare | 2-5% | Conferences, specialist travel |

**Coverage Threshold Levels:**

| Level | Coverage | Description |
|-------|---------|-------------|
| Comprehensive | >90% of trips | All travel captured from TMC/T&E system |
| Substantial | 70-90% | Major travel captured, some gaps |
| Moderate | 50-70% | Primary modes captured |
| Minimal | <50% | Spend-based estimates only |

### 2.14 Emission Factor Selection Hierarchy

| Priority | Source | Description | DQI Score |
|---------|--------|-------------|-----------|
| 1 | Supplier-specific | Airline/hotel/rail verified emissions data | 5.0 |
| 2 | DEFRA 2024 | UK Government conversion factors | 4.5 |
| 3 | ICAO Calculator | ICAO certified aviation methodology | 4.5 |
| 4 | EPA/US | EPA emission factors for US travel | 4.0 |
| 5 | IEA | International Energy Agency country factors | 3.5 |
| 6 | EXIOBASE/EEIO | Spend-based MRIO/EEIO factors | 2.5 |
| 7 | Custom/User | User-provided emission factors | Variable |

### 2.15 Key Formulas Summary

```
AIR TRAVEL (Distance-Based):
  GCD = 2R × arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))
  Distance = GCD × (1 + 0.08)
  CO2e_noRF = Distance × Pax × ClassMultiplier × EF_pkm
  CO2e_withRF = CO2e_noRF × RFI
  WTT = Distance × Pax × ClassMultiplier × WTT_EF_pkm
  Total = CO2e_withRF + WTT

RAIL TRAVEL:
  CO2e = Distance × Pax × EF_rail_pkm
  WTT = Distance × Pax × WTT_rail_pkm
  Total = CO2e + WTT

ROAD TRANSPORT (Distance):
  CO2e = Distance × EF_vkm (per vehicle-km)
  WTT = Distance × WTT_vkm
  Total = CO2e + WTT

ROAD TRANSPORT (Fuel):
  CO2e = Fuel_Litres × EF_litre
  WTT = Fuel_Litres × WTT_litre
  Total = CO2e + WTT

HOTEL STAYS:
  CO2e = Room_Nights × EF_room_night
  Total = CO2e (WTT embedded in room-night EF)

SPEND-BASED (EEIO):
  CO2e = Spend_USD × CPI_Deflator × EEIO_Factor
```

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Engine 1: BusinessTravelDatabaseEngine                        │
│  • Emission factor lookups (DEFRA, ICAO, EPA, EEIO)           │
│  • Airport/station geocoding for distance calculation          │
│  • Hotel country-class EF database                             │
│  • Classification: transport mode, vehicle type, cabin class   │
│  • EF source fallback chain: Supplier > DEFRA > ICAO > EPA    │
├─────────────────────────────────────────────────────────────────┤
│  Engine 2: AirTravelCalculatorEngine                           │
│  • Great-circle distance (Haversine/Vincenty)                 │
│  • Uplift factor application (8% default)                      │
│  • Distance band classification (domestic/short/long-haul)     │
│  • Cabin class multiplier allocation                           │
│  • Radiative forcing index (configurable 1.0-2.7)              │
│  • WTT emissions calculation                                    │
│  • Per-flight and batch calculation                             │
├─────────────────────────────────────────────────────────────────┤
│  Engine 3: GroundTransportCalculatorEngine                     │
│  • Rail emissions (8 rail types, per-pkm)                      │
│  • Road emissions: distance-based (per-vkm, 15 vehicle types) │
│  • Road emissions: fuel-based (5 fuel types)                   │
│  • Taxi/ride-share emissions                                    │
│  • Bus/coach emissions (per-pkm)                               │
│  • Ferry emissions (foot / car passenger)                       │
│  • Motorcycle emissions                                         │
│  • WTT emissions for all ground modes                           │
├─────────────────────────────────────────────────────────────────┤
│  Engine 4: HotelStayCalculatorEngine                           │
│  • Country-specific room-night emission factors                │
│  • Hotel class adjustment (budget/mid/luxury multiplier)       │
│  • Extended stay vs standard stay handling                      │
│  • WTT embedded in room-night factors                           │
│  • Multi-night aggregation                                      │
├─────────────────────────────────────────────────────────────────┤
│  Engine 5: SpendBasedCalculatorEngine                          │
│  • EEIO factor lookup by NAICS code                            │
│  • Currency conversion (20 currencies → USD)                   │
│  • CPI deflation to base year                                   │
│  • Margin removal for spend categories                          │
│  • Travel category classification from expense data             │
├─────────────────────────────────────────────────────────────────┤
│  Engine 6: ComplianceCheckerEngine                             │
│  • 7 regulatory frameworks (GHG Protocol, ISO 14064, CSRD,    │
│    CDP, SBTi, SB 253, GRI)                                     │
│  • RF disclosure requirements per framework                     │
│  • Double-counting prevention rules (DC-BT-001 to DC-BT-007) │
│  • Category boundary validation                                │
│  • Required disclosure fields per framework                     │
├─────────────────────────────────────────────────────────────────┤
│  Engine 7: BusinessTravelPipelineEngine                        │
│  • 10-stage pipeline orchestration                              │
│  • Batch processing with error isolation                        │
│  • Multi-trip aggregation (by mode, period, department)         │
│  • Export: JSON, CSV, Excel, PDF                                │
│  • Provenance chain: SHA-256 across all 10 stages               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Ten-Stage Pipeline

1. **VALIDATE** -- Validate input data: trip records, expense entries, booking data. Check required fields by calculation method. Reject records with missing critical data. Validate coordinates, IATA codes, dates.

2. **CLASSIFY** -- Classify transport mode (air/rail/road/hotel). Determine vehicle type, cabin class, rail type. Map expense categories to travel modes. Identify boundary (Cat 6 vs Cat 7 vs Scope 1).

3. **NORMALIZE** -- Convert units (miles→km, gallons→litres, local currency→USD). Parse dates to reporting period. Standardize airport codes (IATA→coordinates). Apply CPI deflation for spend-based.

4. **RESOLVE_EFS** -- Select emission factors by hierarchy (Supplier>DEFRA>ICAO>EPA>EEIO). Resolve by transport mode, distance band, vehicle type, cabin class, country. Log EF source and version for provenance.

5. **CALCULATE_FLIGHTS** -- Calculate air travel emissions: GCD, uplift, distance band, class multiplier, EF application, RF option, WTT. Per-flight granularity with rollup.

6. **CALCULATE_GROUND** -- Calculate ground transport and hotel emissions: rail, road (distance/fuel), taxi, bus, ferry, hotel room-nights. Per-trip granularity.

7. **ALLOCATE** -- Allocate emissions to organizational units (department, cost center, project). Split multi-purpose trips. Handle shared travel (group bookings).

8. **COMPLIANCE** -- Run compliance checks against 7 frameworks. Validate RF disclosure, boundary completeness, DQI thresholds. Flag double-counting violations.

9. **AGGREGATE** -- Aggregate results by mode, period, department, geography, cabin class. Calculate DQI scores. Generate summary statistics. Build hot-spot analysis.

10. **SEAL** -- Seal provenance chain with Merkle root. Generate SHA-256 audit hash. Timestamp with ISO 8601. Mark calculation as immutable.

### 3.3 File Structure

```
greenlang/business_travel/
├── __init__.py
├── models.py                          (~2200 lines)
├── config.py                          (~1800 lines)
├── metrics.py                         (~350 lines)
├── provenance.py                      (~350 lines)
├── business_travel_database.py        (~700 lines)
├── air_travel_calculator.py           (~900 lines)
├── ground_transport_calculator.py     (~800 lines)
├── hotel_stay_calculator.py           (~500 lines)
├── spend_based_calculator.py          (~500 lines)
├── compliance_checker.py              (~600 lines)
├── business_travel_pipeline.py        (~600 lines)
├── setup.py                           (~500 lines)
├── api/
│   ├── __init__.py
│   └── router.py                      (~900 lines)
tests/unit/mrv/test_business_travel/
├── __init__.py
├── conftest.py
├── test_models.py
├── test_config.py
├── test_metrics.py
├── test_provenance.py
├── test_business_travel_database.py
├── test_air_travel_calculator.py
├── test_ground_transport_calculator.py
├── test_hotel_stay_calculator.py
├── test_spend_based_calculator.py
├── test_compliance_checker.py
├── test_business_travel_pipeline.py
├── test_setup.py
└── test_api.py
deployment/database/migrations/sql/
└── V070__business_travel_service.sql
```

### 3.4 Database Schema (V070)

**16 tables, 3 hypertables, 2 continuous aggregates**

| Table | Description | Type |
|-------|-----------|------|
| `gl_bt_trips` | Trip master records (origin, destination, mode, dates) | Reference |
| `gl_bt_travelers` | Traveler profile (department, cost center) | Reference |
| `gl_bt_airports` | Airport/station database (IATA code, lat, lon) | Reference |
| `gl_bt_air_emission_factors` | Aviation EFs by distance band, class, RF option | Reference |
| `gl_bt_ground_emission_factors` | Rail/road/ferry EFs by vehicle type | Reference |
| `gl_bt_hotel_emission_factors` | Hotel EFs by country/region and class | Reference |
| `gl_bt_eeio_factors` | EEIO spend-based factors by NAICS code | Reference |
| `gl_bt_calculations` | Main calculation results (time-series) | **Hypertable** |
| `gl_bt_flight_results` | Per-flight calculation details | **Hypertable** |
| `gl_bt_ground_results` | Per-ground-trip calculation details | Operational |
| `gl_bt_hotel_results` | Per-hotel-stay calculation details | Operational |
| `gl_bt_spend_results` | Spend-based calculation details | Operational |
| `gl_bt_compliance_checks` | Compliance results with JSONB findings | Operational |
| `gl_bt_uncertainty_analyses` | Monte Carlo/analytical uncertainty results | Operational |
| `gl_bt_aggregations` | Period aggregations with breakdowns | **Hypertable** |
| `gl_bt_provenance` | SHA-256 chain with stage tracking | Operational |

**3 Hypertables:** `gl_bt_calculations` (7-day chunks), `gl_bt_flight_results` (7-day chunks), `gl_bt_aggregations` (30-day chunks)

**2 Continuous Aggregates:** `gl_bt_hourly_emissions` (1-hour refresh), `gl_bt_daily_emissions` (6-hour refresh)

**Key Seed Data:** 500+ airport coordinates (IATA→lat/lon), DEFRA 2024 air/rail/road/hotel EFs, EPA EEIO factors, GWP values (AR4/AR5/AR6)

**Schema Design Principles:** Tenant isolation via RLS, TimescaleDB hypertables for time-series, JSONB for flexible metadata, Decimal(20,8) for all emissions values, SHA-256 provenance hashing

### 3.5 API Endpoints (20)

| # | Method | Endpoint | Description |
|---|--------|---------|-------------|
| 1 | POST | `/calculate` | Full pipeline calculation (auto-detect mode) |
| 2 | POST | `/calculate/batch` | Batch calculation for multiple trips |
| 3 | POST | `/calculate/flight` | Air travel calculation |
| 4 | POST | `/calculate/rail` | Rail travel calculation |
| 5 | POST | `/calculate/road` | Road transport calculation |
| 6 | POST | `/calculate/hotel` | Hotel stay calculation |
| 7 | POST | `/calculate/spend` | Spend-based calculation |
| 8 | GET | `/calculations/{id}` | Get calculation detail |
| 9 | GET | `/calculations` | List calculations (paginated) |
| 10 | DELETE | `/calculations/{id}` | Soft delete calculation |
| 11 | GET | `/emission-factors` | List emission factors (filterable) |
| 12 | GET | `/emission-factors/{mode}` | Get EFs by transport mode |
| 13 | GET | `/airports` | List/search airports (IATA) |
| 14 | GET | `/transport-modes` | List available transport modes |
| 15 | GET | `/cabin-classes` | List cabin classes with multipliers |
| 16 | POST | `/compliance/check` | Multi-framework compliance check |
| 17 | POST | `/uncertainty/analyze` | Uncertainty analysis |
| 18 | GET | `/aggregations/{period}` | Get aggregated results |
| 19 | POST | `/hot-spots/analyze` | Hot-spot analysis by mode/route |
| 20 | GET | `/provenance/{id}` | Get provenance chain |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All calculations use Python `Decimal` type with `ROUND_HALF_UP` and 8 decimal places
- No LLM calls in any calculation path; purely deterministic formula evaluation
- Every emission factor retrieved from audited constant tables, never generated
- Great-circle distance calculated via Haversine formula with exact trigonometric functions
- All airport coordinates from IATA-verified database (no geocoding API calls)
- Cabin class multipliers from DEFRA 2024 published values only
- RFI values constrained to published range (1.0-2.7)
- Hotel EFs from DEFRA/Cornell peer-reviewed sources only
- EEIO factors from EPA USEEIO v2.0 model only
- CPI deflators from Bureau of Labor Statistics published series
- Currency conversions from fixed reference rates (no live API calls)
- All intermediate values recorded with SHA-256 provenance hash
- Bit-perfect reproducibility: same inputs always produce identical outputs
- Every GWP value from IPCC AR4/AR5/AR6 published tables (Table 7.15)
- Distance band thresholds exactly as published in DEFRA methodology

### 4.2 Enumerations (26)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | SUPPLIER_SPECIFIC, DISTANCE_BASED, AVERAGE_DATA, SPEND_BASED | GHG Protocol Cat 6 methods |
| `TransportMode` | AIR, RAIL, ROAD, BUS, TAXI, FERRY, MOTORCYCLE, HOTEL | 8 transport modes |
| `FlightDistanceBand` | DOMESTIC, SHORT_HAUL, LONG_HAUL, INTERNATIONAL_AVG | DEFRA distance bands |
| `CabinClass` | ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST | Cabin classes |
| `RailType` | NATIONAL, INTERNATIONAL, LIGHT_RAIL, UNDERGROUND, EUROSTAR, HIGH_SPEED, US_INTERCITY, US_COMMUTER | Rail types |
| `RoadVehicleType` | CAR_AVERAGE, CAR_SMALL_PETROL, CAR_MEDIUM_PETROL, CAR_LARGE_PETROL, CAR_SMALL_DIESEL, CAR_MEDIUM_DIESEL, CAR_LARGE_DIESEL, HYBRID, PLUGIN_HYBRID, BEV, TAXI_REGULAR, TAXI_BLACK_CAB, MOTORCYCLE | Vehicle types |
| `FuelType` | PETROL, DIESEL, LPG, CNG, E85 | 5 fuel types |
| `BusType` | LOCAL, COACH | Bus/coach types |
| `FerryType` | FOOT_PASSENGER, CAR_PASSENGER | Ferry types |
| `HotelClass` | BUDGET, STANDARD, UPSCALE, LUXURY | Hotel class tiers |
| `TripPurpose` | BUSINESS, CONFERENCE, CLIENT_VISIT, TRAINING, OTHER | Trip purposes |
| `EFSource` | SUPPLIER, DEFRA, ICAO, EPA, IEA, EEIO, CUSTOM | 7 EF sources |
| `ComplianceFramework` | GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, SB_253, GRI | 7 frameworks |
| `DataQualityTier` | TIER_1, TIER_2, TIER_3 | IPCC tiers |
| `RFOption` | WITH_RF, WITHOUT_RF, BOTH | Radiative forcing option |
| `ProvenanceStage` | VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE_FLIGHTS, CALCULATE_GROUND, ALLOCATE, COMPLIANCE, AGGREGATE, SEAL | 10 pipeline stages |
| `UncertaintyMethod` | MONTE_CARLO, ANALYTICAL, IPCC_TIER_2 | 3 methods |
| `DQIDimension` | REPRESENTATIVENESS, COMPLETENESS, TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL | 5 dimensions |
| `DQIScore` | VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW | 5-point scale |
| `ComplianceStatus` | PASS, FAIL, WARNING | 3 statuses |
| `GWPVersion` | AR4, AR5, AR6, AR6_20YR | IPCC assessment reports |
| `EmissionGas` | CO2, CH4, N2O, CO2_BIOGENIC | 4 emission gases |
| `CurrencyCode` | USD, EUR, GBP, CAD, AUD, JPY, CNY, INR, CHF, SGD, BRL, ZAR | 12 currencies |
| `ExportFormat` | JSON, CSV, EXCEL, PDF | 4 formats |
| `BatchStatus` | PENDING, PROCESSING, COMPLETED, FAILED, PARTIAL | 5 batch states |
| `AllocationMethod` | EQUAL, HEADCOUNT, COST_CENTER, DEPARTMENT, PROJECT, CUSTOM | 6 allocation methods |

### 4.3 Models (28)

| Model | Description | Key Fields |
|-------|-----------|------------|
| `FlightInput` | Single flight input | origin_iata, destination_iata, cabin_class, passengers, round_trip, rf_option |
| `RailInput` | Rail trip input | rail_type, distance_km, passengers |
| `RoadDistanceInput` | Road trip (distance) | vehicle_type, distance_km |
| `RoadFuelInput` | Road trip (fuel) | fuel_type, litres |
| `TaxiInput` | Taxi/ride-share | distance_km, taxi_type |
| `BusInput` | Bus/coach trip | bus_type, distance_km, passengers |
| `FerryInput` | Ferry trip | ferry_type, distance_km, passengers |
| `HotelInput` | Hotel stay | country_code, room_nights, hotel_class |
| `SpendInput` | Spend-based | naics_code, amount, currency, year |
| `TripInput` | Generic trip wrapper | mode, trip_data, trip_purpose, tenant_id |
| `BatchTripInput` | Batch of trips | trips (list), reporting_period |
| `AirEmissionFactor` | Aviation EF record | distance_band, cabin_class, ef_without_rf, ef_with_rf, wtt_ef, source |
| `GroundEmissionFactor` | Ground EF record | mode, vehicle_type, ef_per_vkm, ef_per_pkm, wtt_ef, source |
| `HotelEmissionFactor` | Hotel EF record | country_code, hotel_class, ef_per_room_night, source |
| `FlightResult` | Flight calculation | distance_km, distance_band, co2e_without_rf, co2e_with_rf, wtt_co2e, total_co2e |
| `RailResult` | Rail calculation | distance_km, co2e, wtt_co2e, total_co2e |
| `RoadResult` | Road calculation | distance_km, co2e, wtt_co2e, total_co2e, method (distance/fuel) |
| `HotelResult` | Hotel calculation | room_nights, co2e, country_code |
| `SpendResult` | Spend calculation | spend_usd, eeio_factor, co2e |
| `TripCalculationResult` | Per-trip result | mode, method, total_co2e, dqi_score, provenance_hash |
| `BatchResult` | Batch result | results (list), total_co2e, count, errors |
| `AggregationResult` | Aggregated result | total_co2e, by_mode, by_period, by_department |
| `ComplianceCheckResult` | Compliance result | framework, status, score, findings, recommendations |
| `UncertaintyResult` | Uncertainty result | mean, std_dev, ci_lower, ci_upper, method, iterations |
| `DataQualityResult` | DQI result | overall_score, dimensions, classification, tier |
| `ProvenanceRecord` | Provenance entry | stage, input_hash, output_hash, chain_hash, timestamp |
| `HotSpotResult` | Hot-spot analysis | top_routes, top_modes, reduction_opportunities |
| `MetricsSummary` | Agent statistics | total_calculations, total_co2e, avg_dqi |

### 4.4 Constant Tables (15)

**4.4.1** `GWP_VALUES` -- GWP multipliers for CH4, N2O by AR4/AR5/AR6/AR6_20yr

**4.4.2** `AIR_EMISSION_FACTORS` -- DEFRA 2024 air travel EFs by distance band (See Section 2.4)

**4.4.3** `CABIN_CLASS_MULTIPLIERS` -- Cabin class allocation multipliers (See Section 2.4)

**4.4.4** `RAIL_EMISSION_FACTORS` -- Rail EFs by rail type, TTW + WTT (See Section 2.6)

**4.4.5** `ROAD_VEHICLE_EMISSION_FACTORS` -- Road vehicle EFs per vkm/pkm (See Section 2.7)

**4.4.6** `FUEL_EMISSION_FACTORS` -- Fuel combustion EFs per litre/kg (See Section 2.7)

**4.4.7** `BUS_EMISSION_FACTORS` -- Bus/coach EFs per pkm

**4.4.8** `FERRY_EMISSION_FACTORS` -- Ferry EFs per pkm

**4.4.9** `HOTEL_EMISSION_FACTORS` -- Country-specific hotel EFs per room-night (See Section 2.8)

**4.4.10** `HOTEL_CLASS_MULTIPLIERS` -- Hotel class adjustment: Budget=0.75, Standard=1.0, Upscale=1.35, Luxury=1.80

**4.4.11** `EEIO_FACTORS` -- EPA USEEIO v2.0 spend-based factors (See Section 2.9)

**4.4.12** `AIRPORT_DATABASE` -- 500+ airports with IATA code, name, latitude, longitude, country

**4.4.13** `CURRENCY_RATES` -- Exchange rates to USD (2024 reference)

**4.4.14** `CPI_DEFLATORS` -- CPI index by year (2015-2025, base year 2021)

**4.4.15** `DQI_SCORING` -- DQI dimension weights and score mappings (See Section 2.10)

### 4.5 Regulatory Frameworks (7)

**4.5.1 GHG Protocol (Scope 3 Standard, Chapter 7)**
Category 6 requires reporting all business travel emissions from vehicles not owned/operated by the reporting company. RF inclusion is optional but should be reported separately if included. Distance-based method preferred over spend-based where data available.

**4.5.2 ISO 14064-1:2018**
Requires disclosure of indirect GHG emissions from transportation. Business travel classified as "Indirect GHG emissions from transportation" (Category 3 in ISO mapping). Requires uncertainty analysis.

**4.5.3 CSRD / ESRS E1**
ESRS E1 paragraphs 48-56 require disclosure of Scope 3 emissions by category. Business travel must be separately identified. AR 46 requires disclosure of significant Scope 3 categories with methodology description.

**4.5.4 CDP Climate Change**
CDP C6.5 specifically asks for business travel emissions. Requires disclosure of both with-RF and without-RF figures for aviation. Asks for breakdown by transport mode.

**4.5.5 SBTi Corporate Net-Zero Standard**
Business travel must be included in Scope 3 boundary if >1% of total Scope 3 or if emissions from the category are significant. SBTi recommends including RF for aviation in target boundary. Requires near-term and long-term targets.

**4.5.6 California SB 253**
Requires disclosure of all material Scope 3 categories starting 2027 (for fiscal year 2026). Category 6 must be included if material (>1% threshold per GHG Protocol). Third-party assurance required.

**4.5.7 GRI 305**
GRI 305-3 requires disclosure of other indirect (Scope 3) GHG emissions. Business travel is specifically listed as a Scope 3 source. Requires reporting in metric tonnes CO2e with gases included.

**Category-specific compliance rules:**

| Rule Code | Description | Framework |
|-----------|-----------|-----------|
| CR-BT-001 | Report air travel with and without RF separately | CDP |
| CR-BT-002 | Include RF in target boundary | SBTi |
| CR-BT-003 | Breakdown by transport mode required | CDP, CSRD |
| CR-BT-004 | Materiality threshold: >1% of Scope 3 | GHG Protocol, SB 253 |
| CR-BT-005 | Methodology description required | CSRD, GRI |
| CR-BT-006 | Uncertainty analysis required | ISO 14064, CSRD |
| CR-BT-007 | DQI scoring required | GHG Protocol |
| CR-BT-008 | Base year recalculation policy required | SBTi, GHG Protocol |
| CR-BT-009 | Third-party assurance required | SB 253 |
| CR-BT-010 | Report in metric tonnes CO2e | All |

**Framework required disclosures:**

| Framework | Required Disclosures |
|-----------|---------------------|
| GHG Protocol | Total CO2e, method used, EF sources, exclusions, DQI |
| ISO 14064 | Total CO2e, uncertainty, base year, methodology |
| CSRD ESRS E1 | Total CO2e by category, methodology, targets, actions |
| CDP | Total CO2e, with/without RF, mode breakdown, verification |
| SBTi | Total CO2e, target coverage, RF inclusion, progress |
| SB 253 | Total CO2e, methodology, assurance opinion |
| GRI | Total CO2e, gases included, base year, standards used |

### 4.6 Performance Targets

| Metric | Target |
|--------|--------|
| Single trip calculation latency | < 50 ms |
| Flight GCD calculation | < 5 ms |
| Batch 100 trips | < 2 seconds |
| Batch 1,000 trips | < 15 seconds |
| EF lookup latency | < 3 ms |
| Airport search latency | < 5 ms |
| Compliance check (7 frameworks) | < 200 ms |
| Uncertainty analysis (10,000 Monte Carlo) | < 5 seconds |
| API response (single calculation) | < 100 ms |
| Memory per calculation | < 2 MB |
| Database write (single result) | < 20 ms |
| Provenance chain seal | < 10 ms |

---

## 5. Acceptance Criteria

### 5.1 Air Travel Calculations

- [ ] Haversine/Vincenty great-circle distance accurate to ±1 km for any airport pair
- [ ] Uplift factor (8%) correctly applied to all GCD calculations
- [ ] Distance band classification matches DEFRA thresholds exactly
- [ ] All 4 cabin class multipliers applied correctly (1.0, 1.6, 2.9, 4.0)
- [ ] CO2e calculated with and without RF, both stored
- [ ] WTT emissions calculated and added to total
- [ ] Round-trip flights correctly doubled
- [ ] Per-passenger allocation for group bookings
- [ ] 500+ airport IATA codes with lat/lon coordinates

### 5.2 Ground Transport Calculations

- [ ] All 8 rail types with correct DEFRA EFs
- [ ] All 13 road vehicle types with correct DEFRA EFs
- [ ] Fuel-based calculation for 5 fuel types
- [ ] Taxi (regular + black cab) with correct occupancy-based EFs
- [ ] Bus (local + coach) per-pkm calculations
- [ ] Ferry (foot + car passenger) calculations
- [ ] Motorcycle calculations
- [ ] WTT emissions for all ground modes
- [ ] Unit conversion (miles→km, gallons→litres) handling

### 5.3 Hotel Stay Calculations

- [ ] 16+ country-specific room-night EFs
- [ ] 4 hotel class multipliers (budget/standard/upscale/luxury)
- [ ] Multi-night aggregation
- [ ] Global average fallback for unlisted countries
- [ ] Extended stay discount factor

### 5.4 Spend-Based Calculations

- [ ] 10 NAICS travel categories with EEIO factors
- [ ] 12 currency conversions to USD
- [ ] CPI deflation to base year (2021)
- [ ] Margin removal logic
- [ ] Category auto-classification from expense descriptions

### 5.5 Classification & Reference Data

- [ ] Transport mode auto-detection from trip data
- [ ] IATA airport code validation
- [ ] EF source fallback chain (Supplier > DEFRA > ICAO > EPA > EEIO)
- [ ] Distance band auto-classification
- [ ] Cabin class validation

### 5.6 Category Boundary & Double-Counting Prevention

- [ ] 7 double-counting prevention rules (DC-BT-001 to DC-BT-007)
- [ ] Scope 1 vehicle exclusion (company fleet cross-reference)
- [ ] Commuting trip exclusion (purpose check)
- [ ] Freight trip exclusion (purpose check)
- [ ] Cat 4 overlap detection
- [ ] Hotel food/beverage stripping

### 5.7 Data Quality & Uncertainty

- [ ] 5-dimension DQI scoring with configurable weights
- [ ] DQI classification (Excellent/Good/Fair/Poor/Very Poor)
- [ ] Monte Carlo uncertainty analysis (10,000 iterations)
- [ ] IPCC Tier 2 uncertainty method
- [ ] Analytical uncertainty propagation
- [ ] Confidence interval reporting (2.5th/97.5th percentile)

### 5.8 Compliance

- [ ] 7 framework compliance checks
- [ ] RF disclosure validation (CDP, SBTi)
- [ ] Mode breakdown validation (CDP, CSRD)
- [ ] Materiality threshold check (>1%)
- [ ] Required disclosure field validation per framework
- [ ] 10 compliance rules (CR-BT-001 to CR-BT-010)

### 5.9 Infrastructure

- [ ] 20 REST API endpoints operational
- [ ] V070 database migration with 16 tables, 3 hypertables, 2 continuous aggregates
- [ ] SHA-256 provenance chain across 10 pipeline stages
- [ ] 12 Prometheus metrics with `gl_bt_` prefix
- [ ] JWT auth integration with 20 PERMISSION_MAP entries
- [ ] 575+ unit tests with >90% coverage
- [ ] All calculations use `Decimal` type (no floating-point)
- [ ] JSON/CSV/Excel/PDF export
- [ ] RLS tenant isolation on operational tables
- [ ] Thread-safe singleton engines

---

## 6. Prometheus Metrics (12)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_bt_calculations_total` | Counter | method, mode, status |
| 2 | `gl_bt_emissions_kg_co2e_total` | Counter | mode, rf_option |
| 3 | `gl_bt_flights_total` | Counter | distance_band, cabin_class |
| 4 | `gl_bt_ground_trips_total` | Counter | mode, vehicle_type |
| 5 | `gl_bt_hotel_nights_total` | Counter | country |
| 6 | `gl_bt_factor_selections_total` | Counter | source, mode |
| 7 | `gl_bt_compliance_checks_total` | Counter | framework, status |
| 8 | `gl_bt_batch_jobs_total` | Counter | status |
| 9 | `gl_bt_calculation_duration_seconds` | Histogram | method, mode |
| 10 | `gl_bt_batch_size` | Histogram | -- |
| 11 | `gl_bt_active_calculations` | Gauge | -- |
| 12 | `gl_bt_distance_km_total` | Counter | mode |

---

## 7. Auth Integration

### 7.1 PERMISSION_MAP Entries (20)

| # | Method | Path Pattern | Permission |
|---|--------|-------------|------------|
| 1 | POST | `/api/v1/business-travel/calculate` | business-travel:calculate |
| 2 | POST | `/api/v1/business-travel/calculate/batch` | business-travel:calculate |
| 3 | POST | `/api/v1/business-travel/calculate/flight` | business-travel:calculate |
| 4 | POST | `/api/v1/business-travel/calculate/rail` | business-travel:calculate |
| 5 | POST | `/api/v1/business-travel/calculate/road` | business-travel:calculate |
| 6 | POST | `/api/v1/business-travel/calculate/hotel` | business-travel:calculate |
| 7 | POST | `/api/v1/business-travel/calculate/spend` | business-travel:calculate |
| 8 | GET | `/api/v1/business-travel/calculations/{id}` | business-travel:read |
| 9 | GET | `/api/v1/business-travel/calculations` | business-travel:read |
| 10 | DELETE | `/api/v1/business-travel/calculations/{id}` | business-travel:delete |
| 11 | GET | `/api/v1/business-travel/emission-factors` | business-travel:read |
| 12 | GET | `/api/v1/business-travel/emission-factors/{mode}` | business-travel:read |
| 13 | GET | `/api/v1/business-travel/airports` | business-travel:read |
| 14 | GET | `/api/v1/business-travel/transport-modes` | business-travel:read |
| 15 | GET | `/api/v1/business-travel/cabin-classes` | business-travel:read |
| 16 | POST | `/api/v1/business-travel/compliance/check` | business-travel:compliance |
| 17 | POST | `/api/v1/business-travel/uncertainty/analyze` | business-travel:analyze |
| 18 | GET | `/api/v1/business-travel/aggregations/{period}` | business-travel:read |
| 19 | POST | `/api/v1/business-travel/hot-spots/analyze` | business-travel:analyze |
| 20 | GET | `/api/v1/business-travel/provenance/{id}` | business-travel:read |

### 7.2 Router Registration

```python
# Import
from greenlang.business_travel.setup import get_router as get_bt_router

# Include
app.include_router(_bt_router)

# Protect
protect_routes(app, PERMISSION_MAP)
```

---

## 8. Test Suite (575+ tests)

### 8.1 Test Files (15)

| # | File | Tests | Coverage Area |
|---|------|-------|---------------|
| 1 | `test_models.py` | 55 | Enums, constants, input/result model validation |
| 2 | `test_config.py` | 40 | Configuration dataclasses, env loading, singleton |
| 3 | `test_metrics.py` | 25 | Prometheus metrics, graceful fallback |
| 4 | `test_provenance.py` | 30 | SHA-256 chain, validation, Merkle tree |
| 5 | `test_business_travel_database.py` | 45 | EF lookups, airport search, fallback chains |
| 6 | `test_air_travel_calculator.py` | 80 | GCD, uplift, distance bands, cabin class, RF |
| 7 | `test_ground_transport_calculator.py` | 70 | Rail, road, taxi, bus, ferry, motorcycle |
| 8 | `test_hotel_stay_calculator.py` | 40 | Country EFs, class multipliers, multi-night |
| 9 | `test_spend_based_calculator.py` | 35 | EEIO factors, currency, CPI, categories |
| 10 | `test_compliance_checker.py` | 50 | 7 frameworks, RF disclosure, double-counting |
| 11 | `test_business_travel_pipeline.py` | 45 | 10-stage pipeline, batch, aggregation |
| 12 | `test_setup.py` | 20 | Service facade, singleton, router |
| 13 | `test_api.py` | 35 | FastAPI router, all 20 endpoints |
| 14 | `conftest.py` | 5 | Shared fixtures (trips, EFs, config) |
| 15 | `__init__.py` | -- | Package init |
| **Total** | | **575+** | |

### 8.2 Key Test Scenarios

**Air Travel:**
- LHR→JFK (5,555 km, long-haul, economy) → verify GCD, uplift, EF, CO2e
- LHR→CDG (344 km, domestic, business class) → 2.9x multiplier, short-haul EF
- SYD→LAX (12,054 km, long-haul, first class) → 4.0x multiplier, with/without RF
- Round-trip calculation → exactly 2x one-way
- Invalid IATA code → validation error
- Group booking (5 pax) → 5x emissions

**Rail Travel:**
- London→Edinburgh (National Rail, 640 km) → DEFRA national rail EF
- Paris→London (Eurostar, 450 km) → international rail EF
- NYC→Washington (Amtrak, 365 km) → US intercity EF
- Tokyo Shinkansen (high-speed, 500 km) → high-speed EF

**Road Transport:**
- Rental car (medium petrol, 300 km) → distance-based EF
- Rental car (40L diesel) → fuel-based EF
- Taxi (regular, 25 km) → taxi EF with occupancy
- Coach (London→Manchester, 320 km) → coach per-pkm

**Hotel:**
- London (3 nights, standard) → UK EF × 3 × 1.0
- New York (2 nights, luxury) → US EF × 2 × 1.8
- Unknown country → global average fallback

**Compliance:**
- CDP check: with-RF and without-RF both present → PASS
- SBTi check: RF included in target → PASS
- Commuting trip submitted → REJECT (Cat 7)
- Company vehicle submitted → REJECT (Scope 1)

---

## 9. Key Differentiators from Adjacent Categories

### 9.1 Category 6 (Business Travel) vs Category 7 (Employee Commuting)

| Aspect | Category 6 (Business Travel) | Category 7 (Employee Commuting) |
|--------|-------|------|
| Trip purpose | Work-related travel away from regular workplace | Regular home-to-office commute |
| Vehicle ownership | Not owned by reporting company | Not owned by reporting company |
| Frequency | Irregular, event-driven | Regular, daily/weekly |
| Data source | TMC/T&E/booking systems | Employee surveys, HR records |
| Hotel included | Yes (accommodation during travel) | No |
| Agent | AGENT-MRV-019 | AGENT-MRV-020 (planned) |

### 9.2 Category 6 (Business Travel) vs Category 4 (Upstream Transportation)

| Aspect | Category 6 (Business Travel) | Category 4 (Upstream Transport) |
|--------|-------|------|
| What moves | People (employees) | Goods/materials |
| Purpose | Employee travel for work | Transport of purchased goods |
| Mode focus | Air, rail, road, hotel | Road, rail, maritime, air freight |
| EF basis | Per passenger-km | Per tonne-km |
| Agent | AGENT-MRV-019 | AGENT-MRV-017 |

### 9.3 Category 6 (Business Travel) vs Scope 1 (Mobile Combustion)

| Aspect | Category 6 (Business Travel) | Scope 1 (Mobile Combustion) |
|--------|-------|------|
| Vehicle ownership | NOT owned/leased by company | Owned/leased by company |
| Scope | Scope 3 | Scope 1 |
| Control | No operational control | Operational control |
| Reporting | Indirect emissions | Direct emissions |
| Agent | AGENT-MRV-019 | AGENT-MRV-003 |

---

## 10. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic 2.x | Data models and validation |
| FastAPI 0.109+ | REST API framework |
| prometheus_client 0.20+ | Metrics collection |
| psycopg 3.x + psycopg_pool | Async PostgreSQL |
| TimescaleDB 2.x | Time-series hypertables |
| AGENT-MRV-003 (Mobile Combustion) | Cross-reference: Scope 1 fleet exclusion |
| AGENT-MRV-017 (Upstream Transport) | Cross-reference: Cat 4 overlap check |
| AGENT-MRV-020 (Employee Commuting) | Cross-reference: Cat 7 boundary check |
| AGENT-DATA-002 (Excel/CSV Normalizer) | T&E data ingestion |
| AGENT-DATA-003 (ERP Connector) | Travel expense data |
| AGENT-FOUND-001 (Orchestrator) | DAG execution |
| AGENT-FOUND-005 (Citations) | EF source citations |
| AGENT-FOUND-008 (Reproducibility) | Determinism verification |

---

## 11. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-25 | Initial PRD for AGENT-MRV-019 Business Travel Agent |
