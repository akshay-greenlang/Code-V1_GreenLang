# PRD: AGENT-MRV-020 -- Scope 3 Category 7 Employee Commuting Agent

---

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-007 |
| **Internal Label** | AGENT-MRV-020 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/employee_commuting/` |
| **DB Migration** | V071 |
| **Metrics Prefix** | `gl_ec_` |
| **Table Prefix** | `gl_ec_` |
| **API** | `/api/v1/employee-commuting` |
| **Env Prefix** | `GL_EC_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The **GL-MRV-S3-007 Employee Commuting Agent** implements GHG Protocol Scope 3 Category 7 emissions accounting for transportation of employees between their homes and worksites in vehicles not owned or operated by the reporting company. This agent automates calculation of greenhouse gas emissions from daily commuting patterns, remote/hybrid work arrangements, and alternative transportation programs.

Category 7 covers the following sub-activities as defined in the GHG Protocol Scope 3 Standard (Chapter 7):

- **Single-Occupancy Vehicle Commuting (7a)** -- Emissions from employees driving alone in personal vehicles (petrol, diesel, hybrid, EV), the largest commuting source for most organizations
- **Carpooling/Vanpooling (7b)** -- Emissions from shared vehicle commuting with occupancy-adjusted allocation per passenger
- **Public Transit (7c)** -- Emissions from bus, metro/subway, light rail, commuter rail, and ferry services used for commuting
- **Active Transportation (7d)** -- Zero-emission modes: cycling, walking, e-bike, e-scooter (tracked for mode share reporting)
- **Motorcycle/Moped (7e)** -- Emissions from two-wheeled motorized commuting
- **Telework/Remote Work (7f)** -- Emissions from home office energy consumption as partial or full substitute for commuting (optional per GHG Protocol, increasingly required by CSRD)

Employee commuting typically represents **2-10% of total Scope 3 emissions** depending on workforce size, geography, and remote work policies. For service-sector companies with large office-based workforces, commuting can be a highly material and actionable category. Post-pandemic hybrid work patterns have fundamentally changed this category, requiring sophisticated modeling of variable commuting frequencies, telework energy offsets, and mode shift interventions.

The agent automates extraction and calculation of emissions from employee surveys, HR/badge data, parking systems, transit pass programs, and workplace management platforms, producing audit-ready outputs with full provenance chains, data quality scoring, and multi-framework regulatory compliance.

### Justification for Dedicated Agent

1. **Survey-based data complexity** -- Employee commuting relies heavily on periodic surveys (annual/biannual) with statistical extrapolation, response rate adjustments, and sample weighting that no other Category agent requires
2. **Telework/hybrid modeling** -- Post-pandemic work patterns require dedicated modeling of variable commuting frequencies (1-5 days/week), home office energy consumption, and net emissions impact of remote work
3. **Working days normalization** -- Unique requirement to annualize daily commute data using working days calendars, PTO policies, holiday schedules, and regional work patterns
4. **Mode share analysis** -- Complex multi-modal commuting patterns (e.g., drive to station + rail + walk) require trip segmentation and per-segment EF application
5. **Double-counting prevention** -- Complex boundaries with Category 6 (business travel), Scope 1 (company vehicles), Scope 2 (office electricity), and telework energy overlap
6. **Organizational scaling** -- Must extrapolate from survey samples to full workforce using statistical methods (stratified sampling, weighting, confidence intervals)
7. **Intervention modeling** -- Unique requirement to model emissions reduction from programs: transit subsidies, EV incentives, cycle-to-work schemes, compressed work weeks
8. **Regulatory urgency** -- CSRD ESRS E1, CDP, SBTi, and SB 253 all require Category 7 disclosure; CSRD specifically asks for telework emissions

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard, Chapter 7 (Category 7)
- GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 7: Employee Commuting
- GHG Protocol Scope 3 Calculation Guidance -- Appendix: Emission Factors for Employee Commuting
- DEFRA/DESNZ 2024 Government GHG Conversion Factors -- Business Travel & Commuting (Tables 8-10)
- EPA Center for Corporate Climate Leadership -- Emission Factors Hub: Transportation
- EPA MOVES Model -- Vehicle Emission Rates for Commuting
- IPCC Sixth Assessment Report (AR6) -- Global Warming Potentials (Table 7.15)
- CSRD / ESRS E1 -- Climate Change (paragraphs 48-56, AR 46-48)
- California SB 253 Climate Corporate Data Accountability Act -- Scope 3 Disclosure
- CDP Climate Change Questionnaire 2024 -- C6.5 Employee Commuting
- SBTi Corporate Net-Zero Standard v1.2 -- Scope 3 Boundary Requirements
- GRI 305: Emissions 2016 -- Disclosure 305-3
- ISO 14064-1:2018 -- Quantification and reporting of GHG emissions and removals
- IEA CO2 Emissions from Fuel Combustion (2024 Edition)
- US Census Bureau American Community Survey -- Commuting Data
- UK Department for Transport National Travel Survey (2024)
- APTA Public Transportation Fact Book (2024)

### Terminology

| Term | Definition | Scope Mapping |
|------|-----------|---------------|
| Working Days | Number of days per year an employee commutes to the worksite, adjusted for PTO, holidays, sick days, and remote days | Annualization factor |
| Commute Distance | One-way distance from home to primary worksite (km or miles) | Activity data |
| Round-Trip | Two-way commute distance (home to work to home) | 2x one-way distance |
| Mode Share | Percentage distribution of employees across commute modes | Survey output |
| Mode Split | Division of a single commute across multiple transport modes | Multi-modal trips |
| Single-Occupancy Vehicle (SOV) | Employee driving alone in a personal car | Highest per-capita emissions |
| Carpool | Shared vehicle commuting with 2+ occupants, emissions divided by occupancy | Reduced per-capita emissions |
| Vanpool | Organized shared van with 7-15 passengers, often employer-facilitated | Low per-capita emissions |
| Vehicle-km (vkm) | Distance traveled by one vehicle regardless of occupancy | Vehicle-level metric |
| Passenger-km (pkm) | Distance traveled by one passenger (vkm / occupancy) | Passenger-level metric |
| Telework | Working from home or a non-office location, avoiding commute trip | Zero commute, adds home energy |
| Hybrid Work | Split schedule between office and remote (e.g., 3 days office / 2 days home) | Reduced commuting frequency |
| Commuting Frequency | Number of days per week an employee commutes to office | 1-7 scale |
| WFH Energy | Home office electricity and heating energy consumption | Optional Scope 3 or Cat 7 |
| Transit Pass | Employer-provided public transit benefits | Data source for transit trips |
| Response Rate | Percentage of employees completing commute survey | Sample quality metric |
| Extrapolation Factor | Multiplier to scale survey sample to full workforce | = total_employees / respondents |
| Compressed Work Week | 4x10 or 9/80 schedules reducing weekly commute trips | Fewer trips, same hours |
| Active Transport | Non-motorized commuting: walking, cycling, e-bike, e-scooter | Zero direct emissions |
| Fuel Economy | Vehicle fuel consumption rate (L/100km or MPG) | Used for fuel-based calculation |
| Grid Emission Factor | CO2e per kWh for home electricity (for telework calculations) | Location-specific |
| AADT | Annual Average Daily Traffic -- used in area-based estimation | Regional data |
| Park-and-Ride | Multi-modal commute: drive to transit hub then take public transit | Trip segmentation required |
| Shuttle Service | Employer-operated bus/van service between transit hub and worksite | Scope 1 or Cat 7 depending on ownership |
| Modal Shift | Change in commute mode distribution, e.g., SOV to transit | Intervention outcome |
| Well-to-Tank (WTT) | Upstream emissions from fuel extraction, refining, and distribution | Supply chain emissions |
| Tank-to-Wheel (TTW) | Direct emissions from fuel combustion in the vehicle | Operational emissions |

---

## 2. Methodology

### 2.1 Category Boundary Definition

```
Category 7: Employee Commuting
+-- INCLUDED
|   +-- Driving alone (SOV) in personal vehicles
|   +-- Carpooling/vanpooling (allocated by occupancy)
|   +-- Public transit: bus, metro, light rail, commuter rail
|   +-- Ferry/water transport for commuting
|   +-- Motorcycle/moped commuting
|   +-- E-bike/e-scooter commuting (electricity emissions)
|   +-- WTT (well-to-tank) emissions for all fuel types
|   +-- Telework/WFH energy consumption (optional but recommended)
|   +-- Cycling/walking (tracked for mode share, zero emissions)
|   +-- Park-and-ride multi-modal commutes (segmented)
|   +-- Shared mobility (bikeshare, scooter share)
|
+-- EXCLUDED (reported elsewhere)
|   +-- Company-owned/leased vehicles --> Scope 1
|   +-- Company shuttle buses --> Scope 1 or Scope 3 Cat 4
|   +-- Business travel (non-commute trips) --> Category 6
|   +-- Goods/freight transport --> Category 4 or Category 9
|   +-- Office electricity/heating --> Scope 2
|   +-- Company fleet vehicles used for commuting --> Scope 1
|
+-- BOUNDARY TESTS
    +-- IF vehicle owned/leased by company --> Scope 1 (exclude from Cat 7)
    +-- IF trip purpose = business travel --> Category 6 (exclude from Cat 7)
    +-- IF company shuttle/bus --> Scope 1 or procured service
    +-- IF personal car for regular commute --> Include in Cat 7
    +-- IF telework (home office energy) --> Include in Cat 7 (optional)
    +-- IF cycling/walking --> Include (zero emission, track mode share)
    +-- IF shared mobility (bikeshare, scooter share) --> Include in Cat 7
    +-- IF park-and-ride --> Include all legs in Cat 7 (segmented)
```

### 2.2 Calculation Methods

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | **Employee-specific** | Individual commute surveys: distance, mode, frequency, vehicle type | Highest | Low-Medium |
| 2 | **Average-data** | Workforce size, average commute distance by region, mode share | Medium | High |
| 3 | **Spend-based** | Transit subsidies, parking costs, mileage reimbursement spend | Low | Highest |

```
Method Selection Decision Tree:
-----------------------------------
START
+-- Employee commute survey data available?
|   +-- YES --> Method 1: Employee-specific
|   |   +-- Individual distance + mode + frequency
|   |   +-- Extrapolate to full workforce
|   |   +-- Weight by response rate
|   +-- NO (next)
+-- Workforce headcount + regional data available?
|   +-- YES --> Method 2: Average-data
|   |   +-- Use census/national average commute distance
|   |   +-- Apply regional mode share
|   |   +-- Scale by employee count
|   +-- NO (next)
+-- Only spend data available?
    +-- YES --> Method 3: Spend-based
        +-- Transit subsidy spend x EEIO factor
```

### 2.3 Employee-Specific Method (Distance-Based)

```
Per-Employee Annual Emissions:
  Annual_CO2e = Distance_one_way x 2 x Working_Days x Commuting_Frequency x EF_per_km

  Where:
    Distance_one_way = home-to-office distance (km)
    2 = round-trip multiplier
    Working_Days = annual working days (typically 230-250)
    Commuting_Frequency = fraction of work days commuting (e.g., 0.6 for 3 days/week)
    EF_per_km = emission factor for commute mode (kgCO2e/pkm or kgCO2e/vkm)

For SOV (driving alone):
  CO2e = Distance x 2 x Working_Days x Frequency x EF_per_vkm

For Carpool:
  CO2e = Distance x 2 x Working_Days x Frequency x EF_per_vkm / Occupancy

For Public Transit:
  CO2e = Distance x 2 x Working_Days x Frequency x EF_per_pkm

WTT Emissions:
  WTT_CO2e = Distance x 2 x Working_Days x Frequency x WTT_EF_per_km

Total Per Employee:
  Total = CO2e + WTT_CO2e

Organization Total:
  Org_CO2e = SUM (Employee_i CO2e) x Extrapolation_Factor
  Where:
    Extrapolation_Factor = Total_Employees / Survey_Respondents
```

### 2.4 Commute Mode Emission Factors (DEFRA 2024)

| Commute Mode | kgCO2e/vkm | kgCO2e/pkm | WTT (kgCO2e/pkm) | Default Occupancy |
|-------------|------------|------------|-------------------|-------------------|
| Car (average, unknown fuel) | 0.27145 | 0.17082 | 0.03965 | 1.59 |
| Car (small petrol) | 0.20755 | 0.13053 | 0.03301 | 1.59 |
| Car (medium petrol) | 0.25594 | 0.16106 | 0.04074 | 1.59 |
| Car (large petrol) | 0.35388 | 0.22258 | 0.05631 | 1.59 |
| Car (small diesel) | 0.19290 | 0.12132 | 0.02734 | 1.59 |
| Car (medium diesel) | 0.23280 | 0.14642 | 0.03299 | 1.59 |
| Car (large diesel) | 0.29610 | 0.18629 | 0.04198 | 1.59 |
| Hybrid car | 0.17830 | 0.11214 | 0.02838 | 1.59 |
| Plug-in hybrid | 0.10250 | 0.06447 | 0.01363 | 1.59 |
| Battery EV | 0.07005 | 0.04406 | 0.01479 | 1.59 |
| Motorcycle | 0.11337 | 0.11337 | 0.02867 | 1.0 |
| Bus (local average) | -- | 0.10312 | 0.01847 | -- |
| Bus (coach) | -- | 0.02732 | 0.00489 | -- |
| Rail (national average) | -- | 0.03549 | 0.00434 | -- |
| Rail (light rail/tram) | -- | 0.02904 | 0.00612 | -- |
| Rail (underground/metro) | -- | 0.02781 | 0.00586 | -- |
| Rail (commuter rail) | -- | 0.10500 | 0.01300 | -- |
| Ferry (foot passenger) | -- | 0.01877 | 0.00572 | -- |
| E-bike | -- | 0.00500 | 0.00100 | -- |
| E-scooter | -- | 0.00350 | 0.00070 | -- |
| Cycling | -- | 0.00000 | 0.00000 | -- |
| Walking | -- | 0.00000 | 0.00000 | -- |

### 2.5 SOV Fuel-Based Calculation

```
For employees reporting fuel consumption:
  CO2e = Annual_Fuel_Litres x EF_per_litre
  WTT = Annual_Fuel_Litres x WTT_per_litre
  Total = CO2e + WTT

For employees reporting fuel economy:
  Annual_Fuel = (Distance x 2 x Working_Days x Frequency) / Fuel_Economy_km_per_litre
  CO2e = Annual_Fuel x EF_per_litre
```

**Fuel Emission Factors (DEFRA 2024):**

| Fuel Type | kgCO2e/litre | WTT (kgCO2e/litre) |
|-----------|-------------|---------------------|
| Petrol | 2.31480 | 0.58549 |
| Diesel | 2.70370 | 0.60927 |
| LPG | 1.55370 | 0.32149 |
| E10 (10% ethanol) | 2.09780 | 0.52100 |
| B7 (7% biodiesel) | 2.53090 | 0.57200 |
| CNG | 2.53970 (per kg) | 0.50870 (per kg) |

### 2.6 Carpool/Vanpool Allocation

```
Carpool:
  CO2e_per_person = Vehicle_CO2e / Occupancy
  Where:
    Vehicle_CO2e = total vehicle emissions (distance x EF_per_vkm)
    Occupancy = number of passengers (2-5 for carpool)

Vanpool:
  CO2e_per_person = Van_CO2e / Occupancy
  Where:
    Van_CO2e uses van/minibus EF
    Occupancy = typical 7-15 passengers
    Van EF = 0.27439 kgCO2e/vkm (average van DEFRA 2024)

Driver Rotation:
  When driver rotates, each driver bears equal share of vehicle emissions.
  CO2e_per_person = Van_CO2e / Occupancy (same regardless of driver)

Pool Size Tracking:
  System tracks average occupancy per pool over reporting period.
  If actual occupancy varies, use time-weighted average.
  Minimum occupancy = 2 (below that, classify as SOV)
```

### 2.7 Multi-Modal Trip Segmentation

```
Park-and-Ride Example:
  Leg 1: Drive 8 km from home to train station (SOV, car EF)
  Leg 2: Train 25 km from station to downtown (rail EF)
  Leg 3: Walk 0.5 km from station to office (zero emissions)

  Total = Leg_1_CO2e + Leg_2_CO2e + Leg_3_CO2e

Segmentation Rules:
  - Each leg uses its own mode-specific EF
  - Transfer time does not generate emissions
  - If mode is unknown for a leg, use WALK default
  - Maximum legs per trip: 5
  - Each leg must have distance > 0.0 km
```

### 2.8 Telework/Remote Work Emissions

```
Home Office Energy Consumption:
  Annual_kWh = Daily_kWh x Telework_Days_per_Year

  Daily_kWh Estimates (IEA/DEFRA):
    Laptop + monitor: 0.3 kWh/day
    Desktop + monitor: 0.5 kWh/day
    Heating supplement: 2.0-5.0 kWh/day (seasonal, climate-dependent)
    Cooling supplement: 1.0-3.0 kWh/day (seasonal, climate-dependent)
    Lighting: 0.2 kWh/day
    Router/peripherals: 0.1 kWh/day
    Total typical range: 2.5-8.6 kWh/day

  Telework_CO2e = Annual_kWh x Grid_EF_per_kWh
  Where:
    Grid_EF = location-specific grid emission factor (kgCO2e/kWh)

Seasonal Adjustment:
  Q1 (Jan-Mar): Heating factor x 1.3 (northern hemisphere)
  Q2 (Apr-Jun): Base factor x 1.0
  Q3 (Jul-Sep): Cooling factor x 1.2 (hot climates)
  Q4 (Oct-Dec): Heating factor x 1.2
  Note: Southern hemisphere inverts Q1/Q3 adjustments

Allocation Method:
  Proportion of home used for work:
    Dedicated home office = floor_area_office / floor_area_home
    Shared space = 0.10 (default, 10% of home energy)
  Working hours proportion:
    work_hours / total_waking_hours = 8/16 = 0.50

  Allocated_kWh = Total_Home_kWh x space_fraction x time_fraction

Equipment Lifecycle Emissions (Amortized):
  Laptop: 300 kgCO2e over 4 years = 75 kgCO2e/year
  Monitor: 400 kgCO2e over 6 years = 66.7 kgCO2e/year
  Desk/Chair: 100 kgCO2e over 10 years = 10 kgCO2e/year
  Note: Equipment emissions are OPTIONAL and reported as memo items

Net Impact:
  Net_CO2e = Telework_CO2e - Avoided_Commute_CO2e
  Where:
    Avoided_Commute = emissions from trips NOT taken due to telework
    Note: Avoided commute is NOT subtracted from total (memo item only)
    Telework_CO2e IS added to Cat 7 total (if organization reports telework)
```

**Grid Emission Factors (IEA 2024, selected):**

| Country/Region | kgCO2e/kWh | Source Year |
|---------------|-----------|------------|
| United Kingdom | 0.20707 | 2024 |
| United States (average) | 0.37170 | 2024 |
| Germany | 0.33800 | 2024 |
| France | 0.05100 | 2024 |
| Japan | 0.43400 | 2024 |
| China | 0.53700 | 2024 |
| India | 0.70800 | 2024 |
| Canada | 0.12000 | 2024 |
| Australia | 0.65600 | 2024 |
| Brazil | 0.07400 | 2024 |
| South Korea | 0.41500 | 2024 |
| Italy | 0.25800 | 2024 |
| Spain | 0.18400 | 2024 |
| Netherlands | 0.31200 | 2024 |
| Sweden | 0.00800 | 2024 |
| Norway | 0.00700 | 2024 |
| Poland | 0.66200 | 2024 |
| Singapore | 0.40800 | 2024 |
| Global Average | 0.43600 | 2024 |

**US eGRID Subregional Factors (selected):**

| Subregion | kgCO2e/kWh | States Covered |
|-----------|-----------|---------------|
| CAMX (California) | 0.22500 | CA |
| ERCT (Texas) | 0.38800 | TX |
| RFCW (Midwest) | 0.49200 | OH, IN, WV |
| SRMW (Central) | 0.62100 | MO, KS, NE |
| NYUP (New York) | 0.14300 | NY (upstate) |
| NEWE (New England) | 0.19800 | MA, CT, ME, NH, VT, RI |
| SRSO (Southeast) | 0.38900 | AL, GA, MS |
| NWPP (Northwest) | 0.28100 | WA, OR, ID, MT |

### 2.9 Average-Data Method

```
Org_CO2e = Employees x Avg_Commute_Distance x 2 x Working_Days x Mode_Share x EF

Where:
  Employees = total FTE headcount
  Avg_Commute_Distance = national/regional average one-way commute (km)
  Working_Days = 230 (default, adjusted by region)
  Mode_Share = percentage using each mode (from census/survey)
  EF = emission factor per mode (kgCO2e/pkm)
```

**Average Commute Distances (Census/National Statistics):**

| Country | Avg One-Way (km) | Source |
|---------|------------------|--------|
| United States | 21.7 | US Census ACS 2023 |
| United Kingdom | 14.4 | DfT NTS 2024 |
| Germany | 17.0 | Federal Statistical Office |
| France | 13.3 | INSEE 2023 |
| Japan | 19.5 | Statistics Bureau |
| Canada | 15.1 | StatCan 2023 |
| Australia | 16.8 | ABS Census |
| India | 10.2 | NSSO Survey |
| China | 9.8 | NBS Annual |
| South Korea | 16.2 | KOSIS 2023 |
| Global Average | 15.0 | IEA estimate |

**Default Mode Shares (Census Data):**

| Mode | US Share (%) | UK Share (%) | EU Average (%) | Global Default (%) |
|------|-------------|-------------|----------------|-------------------|
| Drive alone | 72.8 | 61.0 | 50.0 | 55.0 |
| Carpool | 8.9 | 5.0 | 8.0 | 8.0 |
| Public transit | 4.9 | 17.0 | 22.0 | 18.0 |
| Walk | 2.6 | 10.0 | 12.0 | 10.0 |
| Cycle | 0.5 | 3.0 | 5.0 | 4.0 |
| Telework | 8.2 | 2.0 | 2.0 | 3.0 |
| Other | 2.1 | 2.0 | 1.0 | 2.0 |

### 2.10 Spend-Based Method (EEIO)

```
Spend_CO2e = Spend_Amount x Currency_Conversion x CPI_Deflator x EEIO_Factor
```

**EEIO Factors for Commuting:**

| NAICS Code | Category | kgCO2e/$ (2021 USD) |
|-----------|----------|---------------------|
| 485000 | Ground passenger transport | 0.2600 |
| 485110 | Urban transit systems | 0.2200 |
| 485210 | Interurban bus | 0.2400 |
| 487110 | Scenic railroad | 0.3100 |
| 488490 | Other support for road transport | 0.1900 |
| 532100 | Automotive rental/leasing | 0.1950 |
| 811100 | Automotive repair (commuting) | 0.1500 |
| 447000 | Gasoline stations (commute fuel) | 0.3200 |

### 2.11 Working Days Calculation

```
Working_Days = Calendar_Days - Weekends - Public_Holidays - PTO_Days - Sick_Days

Effective_Commute_Days = Working_Days x Commuting_Frequency
Where:
  Commuting_Frequency = office_days_per_week / 5
  Example: 3 days/week --> 0.6 frequency

Default Working Days by Region:
  US: 250 - 10 holidays - 10 PTO - 5 sick = 225
  UK: 250 - 8 holidays - 25 PTO - 5 sick = 212
  Germany: 250 - 10 holidays - 30 PTO - 10 sick = 200
  Japan: 250 - 16 holidays - 10 PTO - 5 sick = 219
  France: 250 - 11 holidays - 25 PTO - 5 sick = 209
  Canada: 250 - 11 holidays - 15 PTO - 5 sick = 219
  Australia: 250 - 8 holidays - 20 PTO - 5 sick = 217
  India: 250 - 15 holidays - 15 PTO - 5 sick = 215
  China: 250 - 11 holidays - 5 PTO - 5 sick = 229
  Brazil: 250 - 12 holidays - 30 PTO - 5 sick = 203
  Global Default: 230

Compressed Work Week Adjustments:
  4x10 schedule: Working_Days x (4/5) = 80% of normal trips (longer days)
  9/80 schedule: Working_Days x (9/10) = 90% of normal trips
  Standard 5x8: Working_Days x 1.0 = 100%
```

### 2.12 Part-Time Employee Handling

```
Part-Time Adjustment:
  Effective_Working_Days = Working_Days x FTE_Fraction
  Where:
    FTE_Fraction:
      FULL_TIME = 1.0
      PART_TIME_80 = 0.8 (4 days/week)
      PART_TIME_60 = 0.6 (3 days/week)
      PART_TIME_50 = 0.5 (2.5 days/week)

  CO2e = Distance x 2 x Effective_Working_Days x EF
  Note: Part-time employees commute fewer days, reducing total emissions.
```

### 2.13 Survey Methodology & Statistical Extrapolation

```
Survey Processing Pipeline:
  1. Parse survey responses
  2. Validate fields (mode, distance, frequency, site)
  3. Calculate per-respondent emissions
  4. Aggregate by stratum (site, department, region)
  5. Weight by response rate per stratum
  6. Extrapolate to full workforce
  7. Calculate confidence intervals

Extrapolation Methods:

  Simple Extrapolation:
    Org_CO2e = (SUM(Respondent_CO2e) / Respondents) x Total_Employees
    Extrapolation_Factor = Total_Employees / Respondents

  Stratified Extrapolation:
    Org_CO2e = SUM over strata s:
      (SUM(Respondent_CO2e_s) / Respondents_s) x Total_Employees_s
    Where each stratum s = site/department/region combination

  Weighted Extrapolation:
    Weight_s = (N_s / n_s) / SUM(N_j / n_j)
    Where:
      N_s = total employees in stratum s
      n_s = respondents in stratum s

Confidence Interval Calculation:
  Standard Error: SE = std_dev / sqrt(n)
  95% CI: mean +/- 1.96 x SE
  99% CI: mean +/- 2.576 x SE

  Minimum Sample Size for 95% CI with 5% margin:
    n = (Z^2 x p x (1-p)) / E^2
    n = (1.96^2 x 0.5 x 0.5) / 0.05^2 = 385

Response Rate Quality:
  >80% --> DQI Score 5 (Excellent)
  50-80% --> DQI Score 4 (Good)
  30-50% --> DQI Score 3 (Fair)
  10-30% --> DQI Score 2 (Poor)
  <10% --> DQI Score 1 (Very Poor, consider average-data method)
```

### 2.14 Data Quality Indicator (DQI)

**5-Dimension DQI Scoring:**

| Dimension | Score 5 (Very High) | Score 4 (High) | Score 3 (Medium) | Score 2 (Low) | Score 1 (Very Low) |
|-----------|-------|------|------|------|------|
| Representativeness | Individual employee survey, >80% response rate | Employee survey, 50-80% response | Sample survey <50%, extrapolated | Department-level estimates | National average only |
| Completeness | All employees, all modes, all sites | >90% of workforce | >70% of workforce | >50% of workforce | <50% of workforce |
| Temporal | Current year survey | Within 1 year | Within 2 years | Within 3 years | >3 years old |
| Geographical | Site-specific commute data | City-level averages | State/region averages | Country averages | Global averages |
| Technological | Vehicle-specific EFs with fuel type | Mode-specific EFs | Transport-sector averages | Economy-wide EFs | EEIO averages |

```
Composite DQI = SUM (Dimension_Score x Weight) / SUM Weights
Default weights: Representativeness=0.30, Completeness=0.25, Temporal=0.15, Geographical=0.15, Technological=0.15
```

| Classification | Score Range | Description |
|---------------|------------|-------------|
| Excellent | 4.5 - 5.0 | High confidence, audit-ready |
| Good | 3.5 - 4.4 | Acceptable for reporting |
| Fair | 2.5 - 3.4 | Improvement recommended |
| Poor | 1.5 - 2.4 | Significant gaps |
| Very Poor | 1.0 - 1.4 | Estimate only, not reliable |

### 2.15 Uncertainty Ranges

| Method | DQI Range | Uncertainty (+/-%) | Confidence Level |
|--------|-----------|-----------------|-----------------|
| Employee-specific (>80% response) | 4.0-5.0 | +/-10-15% | 95% |
| Employee-specific (<50% response) | 3.0-4.0 | +/-15-25% | 95% |
| Average-data (census-based) | 2.5-3.5 | +/-25-40% | 95% |
| Spend-based (EEIO) | 1.5-2.5 | +/-40-60% | 95% |
| Telework (estimated kWh) | 2.0-3.0 | +/-30-50% | 95% |

```
Pedigree Matrix Uncertainty:
  U_total = sqrt(U_activity^2 + U_ef^2 + U_method^2)

  Where:
    U_activity = activity data uncertainty (survey sampling, distances)
    U_ef = emission factor uncertainty (DEFRA/EPA source quality)
    U_method = calculation method uncertainty (model assumptions)

Combined Uncertainty (Monte Carlo):
  Run N=10,000 simulations
  Sample distances from normal distribution (mean, std_dev from survey)
  Sample working days from uniform distribution (min, max by region)
  Sample EFs from triangular (min, mode, max) distribution
  Sample response rate from beta distribution
  Report: mean, std_dev, 2.5th/97.5th percentile CI
```

### 2.16 Category Boundaries & Double-Counting Prevention

**Included in Category 7:**

| Activity | Data Source | EF Source |
|----------|-----------|----------|
| SOV commuting (all fuel types) | Employee survey, parking data | DEFRA, EPA |
| Carpool/vanpool | Employee survey | DEFRA (vkm / occupancy) |
| Public transit (all modes) | Employee survey, transit passes | DEFRA, APTA |
| Motorcycle/moped | Employee survey | DEFRA |
| E-bike/e-scooter | Employee survey | Grid EF |
| Cycling/walking | Employee survey | Zero (mode share tracking) |
| Telework energy (optional) | Estimated kWh | Grid EF |
| WTT for all motorized modes | Included with primary EF | DEFRA |
| Park-and-ride (all legs) | Employee survey | Per-leg EF |

**Excluded from Category 7:**

| Activity | Where Reported | Agent |
|----------|---------------|-------|
| Company-owned vehicles | Scope 1 | AGENT-MRV-003 |
| Company shuttle services | Scope 1 or Cat 4 | AGENT-MRV-003/017 |
| Business travel (non-commute) | Category 6 | AGENT-MRV-019 |
| Office electricity | Scope 2 | AGENT-MRV-009/010 |
| Company fleet for commuting | Scope 1 | AGENT-MRV-003 |
| Upstream fuel production | Category 3 | AGENT-MRV-016 |

**Double-Counting Prevention Rules:**

| Rule | Description | Implementation |
|------|-----------|----------------|
| DC-EC-001 | Exclude company-owned/leased vehicles | Check vehicle_ownership != COMPANY |
| DC-EC-002 | Exclude business travel trips | Check trip_purpose != BUSINESS_TRAVEL |
| DC-EC-003 | Exclude company shuttle services | Check mode != COMPANY_SHUTTLE |
| DC-EC-004 | Telework energy not double-counted with Scope 2 | Use home-only grid EF, exclude office energy |
| DC-EC-005 | No overlap with Cat 6 business travel | Cross-reference AGENT-MRV-019 |
| DC-EC-006 | Avoid commute counted as Cat 4 freight | Check trip_type = COMMUTE |
| DC-EC-007 | Active transport zero emissions (not negative) | Floor at 0.0 |
| DC-EC-008 | WTT not double-counted with Cat 3 | WTT included in Cat 7 EFs, exclude from Cat 3 |
| DC-EC-009 | Shuttle to transit hub: allocate correctly | If company-owned shuttle --> Scope 1, else Cat 7 |
| DC-EC-010 | EV charging at office not in Cat 7 | Office EV charging --> Scope 2, home charging --> Cat 7 |

```
Cross-category validation checks:
  IF vehicle in company fleet registry --> REJECT (Scope 1)
  IF trip marked as business travel --> REDIRECT to Cat 6
  IF shuttle is company-owned --> REDIRECT to Scope 1
  IF telework electricity at office --> REDIRECT to Scope 2
  IF EV charging at company facility --> REDIRECT to Scope 2
  IF employee claims zero distance + zero telework --> FLAG for review
```

### 2.17 Coverage & Materiality

| Industry Sector | % of Scope 3 | Primary Drivers |
|----------------|-------------|-----------------|
| Professional Services / Consulting | 5-10% | Large office workforce |
| Financial Services / Banking | 3-8% | Concentrated urban offices |
| Technology / Software | 2-5% | Hybrid work reduces |
| Government / Public Sector | 5-12% | Large distributed workforce |
| Healthcare | 3-8% | Shift workers, multiple sites |
| Education | 4-10% | Staff + campus commuting |
| Manufacturing | 1-3% | Factory shift workers |
| Retail | 2-5% | Store workers, shift-based |
| Hospitality | 2-4% | Shift-based, low-wage commuters |
| Nonprofits | 3-6% | Office-based, moderate travel |

### 2.18 Emission Factor Selection Hierarchy

| Priority | Source | Description | DQI Score |
|---------|--------|-------------|-----------|
| 1 | Employee-reported | Actual fuel consumption or transit receipts | 5.0 |
| 2 | DEFRA 2024 | UK Government conversion factors | 4.5 |
| 3 | EPA/US | EPA vehicle emission factors | 4.0 |
| 4 | IEA | International Energy Agency country factors | 3.5 |
| 5 | Census/National | Census-based average commute data | 3.0 |
| 6 | EEIO | Spend-based input-output factors | 2.5 |
| 7 | Custom/User | User-provided emission factors | Variable |

### 2.19 Key Formulas Summary

```
SOV COMMUTING (Distance-Based):
  CO2e = Dist_one_way x 2 x Working_Days x Frequency x EF_vkm
  WTT = Dist_one_way x 2 x Working_Days x Frequency x WTT_vkm
  Total = CO2e + WTT

SOV COMMUTING (Fuel-Based):
  Annual_Fuel = (Dist x 2 x Working_Days x Frequency) / Fuel_Economy
  CO2e = Annual_Fuel x EF_per_litre
  WTT = Annual_Fuel x WTT_per_litre

CARPOOL:
  CO2e_per_person = (Dist x 2 x Working_Days x Freq x EF_vkm) / Occupancy

VANPOOL:
  CO2e_per_person = (Dist x 2 x Working_Days x Freq x Van_EF_vkm) / Occupancy

PUBLIC TRANSIT:
  CO2e = Dist x 2 x Working_Days x Frequency x EF_pkm

MULTI-MODAL:
  CO2e = SUM over legs: (Leg_Dist x EF_leg_mode)
  Annual = Single_Trip_CO2e x 2 x Working_Days x Frequency

TELEWORK:
  CO2e = Telework_Days x Daily_kWh x Grid_EF

AVERAGE-DATA METHOD:
  CO2e = Employees x Avg_Dist x 2 x Working_Days x SUM(Mode_Share_i x EF_i)

SPEND-BASED METHOD:
  CO2e = Spend_USD x CPI_Deflator x EEIO_Factor

ORGANIZATIONAL TOTAL:
  Org_CO2e = SUM(Employee_CO2e_i) x (Total_Employees / Respondents)

INTERVENTION MODELING:
  Reduction = Baseline_CO2e - Scenario_CO2e
  Where Scenario adjusts mode share, frequency, or distances
```

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
+---------------------------------------------------------------+
|  Engine 1: EmployeeCommutingDatabaseEngine                     |
|  * Commute mode emission factor lookups (DEFRA, EPA)           |
|  * Vehicle type EF database with fuel variants                 |
|  * Public transit EFs by mode                                  |
|  * Grid emission factors by country for telework               |
|  * eGRID subregional factors for US telework                   |
|  * Working days calendars by region                            |
|  * Average commute distances by country                        |
|  * Mode share defaults by region                               |
|  * EF source fallback chain: Employee > DEFRA > EPA > EEIO    |
+---------------------------------------------------------------+
|  Engine 2: CommuteModeCalculatorEngine                         |
|  * SOV distance-based calculation (12 vehicle types)           |
|  * SOV fuel-based calculation (6 fuel types)                   |
|  * Carpool/vanpool with occupancy allocation                   |
|  * Public transit (bus, metro, rail, ferry)                    |
|  * Motorcycle/moped emissions                                  |
|  * E-bike/e-scooter emissions (grid-based)                    |
|  * Active transport tracking (zero emissions)                  |
|  * WTT emissions for all motorized modes                       |
|  * Multi-modal trip segmentation (up to 5 legs)               |
|  * Park-and-ride composite calculation                         |
+---------------------------------------------------------------+
|  Engine 3: TeleworkCalculatorEngine                            |
|  * Home office energy consumption modeling                     |
|  * Country-specific grid emission factors (19 countries)       |
|  * eGRID subregional factors (26 US subregions)               |
|  * Seasonal heating/cooling adjustment (4 quarters)            |
|  * Avoided commute emissions (memo item)                       |
|  * Net telework impact analysis                                |
|  * Hybrid work frequency modeling (1-5 days/week)             |
|  * Equipment lifecycle emissions (amortized, optional)         |
|  * Space allocation method (dedicated vs shared)               |
+---------------------------------------------------------------+
|  Engine 4: SurveyProcessorEngine                               |
|  * Survey response parsing and validation                      |
|  * Statistical extrapolation (sample to workforce)             |
|  * Response rate weighting                                     |
|  * Stratified sampling adjustment                              |
|  * Confidence interval calculation (95%, 99%)                  |
|  * Working days normalization by region                        |
|  * Annualization of commute patterns                           |
|  * Part-time employee handling                                 |
|  * Compressed work week adjustment                             |
|  * Minimum sample size validation                              |
+---------------------------------------------------------------+
|  Engine 5: SpendBasedCalculatorEngine                          |
|  * EEIO factor lookup by NAICS code (8 categories)            |
|  * Transit subsidy spend processing                            |
|  * Parking cost-based estimation                               |
|  * Mileage reimbursement spend processing                      |
|  * Currency conversion (12 currencies) and CPI deflation       |
+---------------------------------------------------------------+
|  Engine 6: ComplianceCheckerEngine                             |
|  * 7 regulatory frameworks                                     |
|  * Telework disclosure requirements (CSRD)                     |
|  * Mode share reporting validation (CDP)                       |
|  * Double-counting prevention rules (DC-EC-001 to DC-EC-010)  |
|  * Survey methodology documentation check                      |
|  * Materiality threshold validation (>1% of Scope 3)          |
|  * Base year recalculation policy check                        |
|  * Third-party assurance readiness (SB 253)                    |
+---------------------------------------------------------------+
|  Engine 7: EmployeeCommutingPipelineEngine                     |
|  * 10-stage pipeline orchestration                             |
|  * Batch processing per employee                               |
|  * Workforce-level aggregation                                 |
|  * Mode share analysis and hot-spot identification             |
|  * Intervention scenario modeling                              |
|  * Export: JSON, CSV, Excel, PDF                               |
|  * Provenance chain: SHA-256 across all 10 stages              |
+---------------------------------------------------------------+
```

### 3.2 Ten-Stage Pipeline

1. **VALIDATE** -- Validate input data: survey responses, employee records, distance data. Check required fields by method. Validate distances, modes, frequencies. Reject zero-distance non-telework records. Validate employee IDs and site codes.

2. **CLASSIFY** -- Classify commute mode (SOV/carpool/transit/active/telework). Determine vehicle type, fuel type. Parse multi-modal trips into segments. Classify telework frequency (full remote/hybrid/office-based). Apply boundary tests.

3. **NORMALIZE** -- Convert units (miles to km, gallons to litres, mpg to L/100km). Normalize commute frequency to annual working days. Adjust for part-time employees. Apply compressed work week factors. Standardize currency for spend-based inputs.

4. **RESOLVE_EFS** -- Select emission factors by hierarchy. Resolve by commute mode, vehicle type, fuel type, transit type. Look up grid EFs for telework (country or eGRID subregion). Log EF source and version for provenance.

5. **CALCULATE_COMMUTE** -- Calculate commute emissions per employee: distance x frequency x working_days x EF. Apply carpool occupancy division. Calculate WTT. Segment multi-modal trips and sum per-leg emissions.

6. **CALCULATE_TELEWORK** -- Calculate telework emissions: days x kWh x grid_EF. Apply seasonal adjustment. Calculate avoided commute (memo). Determine net impact. Apply space allocation method.

7. **EXTRAPOLATE** -- Scale survey sample to full workforce. Apply response rate weighting. Calculate confidence intervals. Handle stratified sampling. Validate minimum sample size. Flag low-response strata.

8. **COMPLIANCE** -- Run compliance checks against 7 frameworks. Validate telework disclosure. Check mode share reporting. Flag double-counting. Verify materiality threshold. Check methodology documentation.

9. **AGGREGATE** -- Aggregate by mode, site, department, commute distance band. Calculate organization-wide mode share. Generate hot-spot analysis. Build intervention scenarios. Calculate reduction potential.

10. **SEAL** -- Seal provenance chain with Merkle root. Generate SHA-256 audit hash. Timestamp with ISO 8601. Mark immutable. Store chain in provenance table.

### 3.3 File Structure

```
greenlang/employee_commuting/
+-- __init__.py
+-- models.py                          (~2400 lines)
+-- config.py                          (~1800 lines)
+-- metrics.py                         (~350 lines)
+-- provenance.py                      (~350 lines)
+-- employee_commuting_database.py     (~750 lines)
+-- commute_mode_calculator.py         (~950 lines)
+-- telework_calculator.py             (~650 lines)
+-- survey_processor.py                (~750 lines)
+-- spend_based_calculator.py          (~500 lines)
+-- compliance_checker.py              (~650 lines)
+-- employee_commuting_pipeline.py     (~650 lines)
+-- setup.py                           (~500 lines)
+-- api/
|   +-- __init__.py
|   +-- router.py                      (~950 lines)
tests/unit/mrv/test_employee_commuting/
+-- __init__.py
+-- conftest.py
+-- test_models.py
+-- test_config.py
+-- test_metrics.py
+-- test_provenance.py
+-- test_employee_commuting_database.py
+-- test_commute_mode_calculator.py
+-- test_telework_calculator.py
+-- test_survey_processor.py
+-- test_spend_based_calculator.py
+-- test_compliance_checker.py
+-- test_employee_commuting_pipeline.py
+-- test_setup.py
+-- test_api.py
deployment/database/migrations/sql/
+-- V071__employee_commuting_service.sql
```

### 3.4 Database Schema (V071)

**16 tables, 3 hypertables, 2 continuous aggregates**

| Table | Description | Type |
|-------|-----------|------|
| `gl_ec_employees` | Employee profile (site, department, work schedule, FTE fraction) | Reference |
| `gl_ec_commute_profiles` | Individual commute patterns (mode, distance, frequency, vehicle type) | Reference |
| `gl_ec_vehicle_emission_factors` | Vehicle EFs by type and fuel (DEFRA 2024) | Reference |
| `gl_ec_transit_emission_factors` | Public transit EFs by mode | Reference |
| `gl_ec_grid_emission_factors` | Grid EFs by country/eGRID for telework | Reference |
| `gl_ec_working_days` | Working days calendar by region | Reference |
| `gl_ec_commute_averages` | Average commute distances by country | Reference |
| `gl_ec_eeio_factors` | EEIO spend-based factors | Reference |
| `gl_ec_calculations` | Main calculation results | **Hypertable** |
| `gl_ec_commute_results` | Per-employee commute results | **Hypertable** |
| `gl_ec_telework_results` | Telework emissions results | Operational |
| `gl_ec_survey_results` | Survey processing/extrapolation results | Operational |
| `gl_ec_spend_results` | Spend-based calculation results | Operational |
| `gl_ec_compliance_checks` | Compliance results with JSONB findings | Operational |
| `gl_ec_aggregations` | Period aggregations with breakdowns | **Hypertable** |
| `gl_ec_provenance` | SHA-256 chain with stage tracking | Operational |

**3 Hypertables:** `gl_ec_calculations` (7-day chunks), `gl_ec_commute_results` (7-day chunks), `gl_ec_aggregations` (30-day chunks)

**2 Continuous Aggregates:** `gl_ec_hourly_emissions` (1-hour refresh), `gl_ec_daily_emissions` (6-hour refresh)

**Key Seed Data:** DEFRA 2024 vehicle/transit/fuel EFs, IEA 2024 grid EFs (19 countries), eGRID 2024 subregional EFs (26 subregions), working days calendars (11 regions), average commute distances (11 countries), default mode shares (4 regions), EEIO factors (8 NAICS codes), GWP values (AR4/AR5/AR6)

**Schema Design Principles:** Tenant isolation via RLS, TimescaleDB hypertables for time-series, JSONB for flexible metadata (survey responses, multi-modal legs), Decimal(20,8) for all emissions values, SHA-256 provenance hashing, soft delete with `deleted_at` timestamps

**Row-Level Security:**
```
All operational tables enforce RLS:
  CREATE POLICY tenant_isolation ON gl_ec_calculations
    USING (tenant_id = current_setting('app.current_tenant')::uuid);
```

**Key Indexes:**
```
gl_ec_calculations: (tenant_id, calculation_date)
gl_ec_commute_results: (tenant_id, employee_id, calculation_id)
gl_ec_vehicle_emission_factors: (vehicle_type, fuel_type, source)
gl_ec_grid_emission_factors: (country_code, source, year)
gl_ec_employees: (tenant_id, site_id, department)
```

### 3.5 API Endpoints (22)

| # | Method | Endpoint | Description |
|---|--------|---------|-------------|
| 1 | POST | `/calculate` | Full pipeline calculation |
| 2 | POST | `/calculate/batch` | Batch calculation for multiple employees |
| 3 | POST | `/calculate/commute` | Single commute mode calculation |
| 4 | POST | `/calculate/telework` | Telework emissions calculation |
| 5 | POST | `/calculate/survey` | Process employee survey data |
| 6 | POST | `/calculate/average-data` | Average-data method calculation |
| 7 | POST | `/calculate/spend` | Spend-based calculation |
| 8 | POST | `/calculate/multi-modal` | Multi-modal trip calculation |
| 9 | GET | `/calculations/{id}` | Get calculation detail |
| 10 | GET | `/calculations` | List calculations (paginated) |
| 11 | DELETE | `/calculations/{id}` | Soft delete calculation |
| 12 | GET | `/emission-factors` | List emission factors |
| 13 | GET | `/emission-factors/{mode}` | Get EFs by commute mode |
| 14 | GET | `/commute-modes` | List commute modes |
| 15 | GET | `/working-days/{region}` | Get working days by region |
| 16 | GET | `/commute-averages` | Get average commute distances |
| 17 | GET | `/grid-factors/{country}` | Get grid EFs for telework |
| 18 | POST | `/compliance/check` | Multi-framework compliance check |
| 19 | POST | `/uncertainty/analyze` | Uncertainty analysis |
| 20 | GET | `/aggregations/{period}` | Get aggregated results |
| 21 | POST | `/mode-share/analyze` | Mode share analysis |
| 22 | GET | `/provenance/{id}` | Get provenance chain |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All calculations use Python `Decimal` type with `ROUND_HALF_UP` and 8 decimal places
- No LLM calls in any calculation path; purely deterministic formula evaluation
- Every emission factor from audited DEFRA/EPA/IEA constant tables
- Working days from published regional calendars, never estimated
- Average commute distances from census/national statistics only
- Mode shares from official surveys (Census ACS, DfT NTS)
- Grid emission factors from IEA published data only
- eGRID subregional factors from EPA published data only
- Carpool occupancy from survey data or published defaults
- Currency conversions from fixed reference rates (no live API calls)
- All intermediate values recorded with SHA-256 provenance hash
- Bit-perfect reproducibility: same inputs always produce identical outputs
- Statistical extrapolation uses transparent formula with documented confidence intervals
- Every GWP value from IPCC AR4/AR5/AR6 published tables
- Telework daily kWh from IEA/DEFRA peer-reviewed estimates
- Seasonal adjustment factors from published climate data
- No stochastic elements in main calculation path (Monte Carlo only in uncertainty engine)

### 4.2 Enumerations (27)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | EMPLOYEE_SPECIFIC, AVERAGE_DATA, SPEND_BASED | 3 GHG Protocol methods |
| `CommuteMode` | SOV, CARPOOL, VANPOOL, BUS, METRO, LIGHT_RAIL, COMMUTER_RAIL, FERRY, MOTORCYCLE, E_BIKE, E_SCOOTER, CYCLING, WALKING, TELEWORK | 14 commute modes |
| `VehicleType` | CAR_AVERAGE, CAR_SMALL_PETROL, CAR_MEDIUM_PETROL, CAR_LARGE_PETROL, CAR_SMALL_DIESEL, CAR_MEDIUM_DIESEL, CAR_LARGE_DIESEL, HYBRID, PLUGIN_HYBRID, BEV, VAN_AVERAGE, MOTORCYCLE | 12 vehicle types |
| `FuelType` | PETROL, DIESEL, LPG, E10, B7, CNG | 6 fuel types |
| `TransitType` | BUS_LOCAL, BUS_COACH, METRO, LIGHT_RAIL, COMMUTER_RAIL, FERRY | 6 transit types |
| `TeleworkFrequency` | FULL_REMOTE, HYBRID_4, HYBRID_3, HYBRID_2, HYBRID_1, OFFICE_FULL | 6 telework patterns |
| `WorkSchedule` | FULL_TIME, PART_TIME_80, PART_TIME_60, PART_TIME_50 | 4 schedules |
| `CompressedSchedule` | STANDARD_5X8, COMPRESSED_4X10, COMPRESSED_9_80 | 3 schedules |
| `EFSource` | EMPLOYEE, DEFRA, EPA, IEA, EGRID, CENSUS, EEIO, CUSTOM | 8 EF sources |
| `ComplianceFramework` | GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, SB_253, GRI | 7 frameworks |
| `DataQualityTier` | TIER_1, TIER_2, TIER_3 | 3 tiers |
| `ProvenanceStage` | VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE_COMMUTE, CALCULATE_TELEWORK, EXTRAPOLATE, COMPLIANCE, AGGREGATE, SEAL | 10 stages |
| `UncertaintyMethod` | MONTE_CARLO, ANALYTICAL, IPCC_TIER_2 | 3 methods |
| `DQIDimension` | REPRESENTATIVENESS, COMPLETENESS, TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL | 5 dimensions |
| `DQIScore` | VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW | 5-point scale |
| `ComplianceStatus` | PASS, FAIL, WARNING | 3 statuses |
| `GWPVersion` | AR4, AR5, AR6, AR6_20YR | 4 GWP versions |
| `EmissionGas` | CO2, CH4, N2O | 3 gases |
| `CurrencyCode` | USD, EUR, GBP, CAD, AUD, JPY, CNY, INR, CHF, SGD, BRL, ZAR | 12 currencies |
| `ExportFormat` | JSON, CSV, EXCEL, PDF | 4 export formats |
| `BatchStatus` | PENDING, PROCESSING, COMPLETED, FAILED, PARTIAL | 5 batch states |
| `RegionCode` | US, GB, DE, FR, JP, CA, AU, IN, CN, BR, KR, GLOBAL | 12 regions |
| `DistanceBand` | SHORT_0_5, MEDIUM_5_15, LONG_15_30, VERY_LONG_30_PLUS | 4 commute distance bands |
| `SurveyMethod` | FULL_CENSUS, STRATIFIED_SAMPLE, RANDOM_SAMPLE, CONVENIENCE | 4 survey methods |
| `AllocationMethod` | EQUAL, HEADCOUNT, SITE, DEPARTMENT, COST_CENTER | 5 allocation methods |
| `SeasonalAdjustment` | NONE, HEATING_ONLY, COOLING_ONLY, FULL_SEASONAL | 4 seasonal adjustments |
| `VehicleOwnership` | PERSONAL, COMPANY, LEASED_PERSONAL, LEASED_COMPANY | 4 ownership types |

### 4.3 Models (30)

| Model | Description | Key Fields |
|-------|-----------|------------|
| `CommuteInput` | Single employee commute | mode, distance_km, frequency_days_per_week, vehicle_type, fuel_type, occupancy, working_days |
| `FuelBasedCommuteInput` | Fuel-based commute | fuel_type, annual_litres OR fuel_economy_km_per_l, distance_km |
| `CarpoolInput` | Carpool commute | vehicle_type, distance_km, occupancy, frequency |
| `VanpoolInput` | Vanpool commute | distance_km, occupancy, frequency |
| `TransitInput` | Transit commute | transit_type, distance_km, frequency |
| `MultiModalInput` | Multi-modal trip | legs (list of CommuteInput), frequency |
| `TeleworkInput` | Telework data | days_per_week, country_code, daily_kwh, seasonal_adjustment, egrid_subregion |
| `SurveyInput` | Survey batch | responses (list), total_employees, response_rate, survey_method |
| `SurveyResponseInput` | Individual response | employee_id, commute_segments (list), telework_days, site, department |
| `AverageDataInput` | Average method | total_employees, country_code, region, custom_mode_share |
| `SpendInput` | Spend-based | naics_code, amount, currency, year |
| `EmployeeInput` | Employee wrapper | employee_id, commute_data, telework_data, department, site, work_schedule, compressed_schedule, tenant_id |
| `BatchEmployeeInput` | Batch of employees | employees (list), reporting_period |
| `CommuteResult` | Commute calculation | mode, distance_km, annual_co2e, wtt_co2e, total_co2e, working_days_used, ef_source |
| `MultiModalResult` | Multi-modal result | legs (list of CommuteResult), total_co2e |
| `TeleworkResult` | Telework calculation | days_per_year, annual_kwh, telework_co2e, avoided_commute_co2e, net_impact, grid_ef_used |
| `EmployeeResult` | Per-employee total | commute_co2e, telework_co2e, total_co2e, modes_used, dqi_score |
| `SurveyResult` | Survey processing | total_org_co2e, respondent_count, extrapolation_factor, confidence_interval, mode_share |
| `BatchResult` | Batch result | results (list), total_co2e, count, errors |
| `AggregationResult` | Aggregated result | total_co2e, by_mode, by_site, by_department, by_distance_band, mode_share |
| `ModeShareResult` | Mode share analysis | mode_shares, weighted_ef, reduction_opportunities |
| `ComplianceCheckResult` | Compliance result | framework, status, score, findings, recommendations |
| `UncertaintyResult` | Uncertainty | mean, std_dev, ci_lower, ci_upper, method, iterations |
| `DataQualityResult` | DQI result | overall_score, dimensions, classification, tier |
| `ProvenanceRecord` | Provenance entry | stage, input_hash, output_hash, chain_hash, timestamp |
| `ProvenanceChainResult` | Chain result | records, is_valid, chain_hash |
| `SpendResult` | Spend calculation | naics_code, spend_usd, eeio_factor, co2e |
| `WorkingDaysResult` | Working days info | region, total_days, holidays, pto, sick, net_working_days |
| `GridEFResult` | Grid EF info | country_code, ef_per_kwh, source, year, subregion |
| `HotSpotResult` | Hot-spot analysis | top_sites, top_modes, high_distance_commuters, reduction_potential |
| `MetricsSummary` | Agent statistics | total_calculations, total_co2e, total_employees_processed, avg_dqi |

### 4.4 Constant Tables (18)

**4.4.1** `GWP_VALUES` -- AR4/AR5/AR6/AR6_20yr for CH4, N2O

**4.4.2** `VEHICLE_EMISSION_FACTORS` -- 12 vehicle types with vkm/pkm/WTT EFs (See 2.4)

**4.4.3** `FUEL_EMISSION_FACTORS` -- 6 fuel types with per-litre EFs (See 2.5)

**4.4.4** `TRANSIT_EMISSION_FACTORS` -- 6 transit types with per-pkm EFs (See 2.4)

**4.4.5** `MICRO_MOBILITY_EFS` -- E-bike, e-scooter EFs

**4.4.6** `GRID_EMISSION_FACTORS` -- 19 countries + global average (See 2.8)

**4.4.7** `EGRID_FACTORS` -- 26 US subregional grid EFs (See 2.8)

**4.4.8** `WORKING_DAYS_DEFAULTS` -- 12 regions with holidays/PTO/sick defaults (See 2.11)

**4.4.9** `AVERAGE_COMMUTE_DISTANCES` -- 11 countries + global (See 2.9)

**4.4.10** `DEFAULT_MODE_SHARES` -- US/UK/EU/Global mode share distributions (See 2.9)

**4.4.11** `TELEWORK_ENERGY_DEFAULTS` -- Daily kWh by equipment type + seasonal heating/cooling factors

**4.4.12** `VAN_EMISSION_FACTORS` -- Van/minibus EFs for vanpool

**4.4.13** `EEIO_FACTORS` -- 8 NAICS commuting categories (See 2.10)

**4.4.14** `CURRENCY_RATES` -- 12 currencies to USD (2024 reference)

**4.4.15** `CPI_DEFLATORS` -- 2015-2025 deflators (base 2021)

**4.4.16** `DQI_SCORING` -- 5 dimensions x 5 scores with weights

**4.4.17** `UNCERTAINTY_RANGES` -- 5 methods x 3 tiers

**4.4.18** `COMPRESSED_SCHEDULE_FACTORS` -- 3 schedule types with trip reduction factors

### 4.5 Regulatory Frameworks (7)

**4.5.1 GHG Protocol (Scope 3 Standard, Chapter 7)**
Category 7 requires reporting employee commuting emissions. Telework optional but recommended. Survey-based or average-data methods acceptable. Companies should describe the methodology used, data sources, and quality of estimates. GHG Protocol recommends reporting telework emissions separately from commute emissions.

**4.5.2 ISO 14064-1:2018**
Indirect GHG emissions from employee transportation. Requires uncertainty analysis. Must document organizational boundary and quantification methodology. Requires base year emissions and recalculation policy.

**4.5.3 CSRD / ESRS E1**
Specifically requires disclosure of employee commuting emissions. ESRS E1 AR 46 asks about telework emissions and mode share. Paragraph 48 requires disclosure of Scope 3 by category. Must include actions taken to reduce commuting emissions and targets set. CSRD is the first framework to explicitly require telework emissions disclosure.

**4.5.4 CDP Climate Change**
CDP C6.5 asks for employee commuting breakdown. Mode share disclosure requested. Asks whether the company provides incentives for sustainable commuting. Requires methodological description and data quality assessment.

**4.5.5 SBTi Corporate Net-Zero Standard**
Include if >1% of total Scope 3 or if significant. Telework recommended. Near-term (5-10 year) and long-term (by 2050) targets required if included in boundary. Requires disclosure of base year and progress.

**4.5.6 California SB 253**
Requires disclosure of all material Scope 3 categories starting 2027 (for FY 2026). Category 7 must be included if material (>1% threshold). Third-party assurance required for reported figures.

**4.5.7 GRI 305**
Disclosure 305-3 requires other indirect Scope 3 emissions including commuting. Must report in metric tonnes CO2e. Requires disclosure of gases included, base year, and standards/methodologies used.

**Category-specific compliance rules:**

| Rule Code | Description | Framework |
|-----------|-----------|-----------|
| CR-EC-001 | Report telework emissions separately | CSRD, CDP |
| CR-EC-002 | Mode share breakdown required | CDP, CSRD |
| CR-EC-003 | Materiality threshold: >1% of Scope 3 | GHG Protocol, SB 253 |
| CR-EC-004 | Methodology description required | CSRD, GRI, ISO 14064 |
| CR-EC-005 | Uncertainty analysis required | ISO 14064, CSRD |
| CR-EC-006 | DQI scoring required | GHG Protocol |
| CR-EC-007 | Base year recalculation policy required | SBTi, GHG Protocol |
| CR-EC-008 | Third-party assurance required | SB 253 |
| CR-EC-009 | Report in metric tonnes CO2e | All |
| CR-EC-010 | Survey methodology documentation | GHG Protocol, ISO 14064 |
| CR-EC-011 | Commuting reduction targets disclosed | SBTi, CSRD |
| CR-EC-012 | Sustainable commuting incentives reported | CDP |

**Framework required disclosures:**

| Framework | Required Disclosures |
|-----------|---------------------|
| GHG Protocol | Total CO2e, method used, EF sources, exclusions, DQI, survey methodology |
| ISO 14064 | Total CO2e, uncertainty, base year, methodology, boundary |
| CSRD ESRS E1 | Total CO2e by category, telework emissions, mode share, targets, actions |
| CDP | Total CO2e, mode breakdown, commuting incentives, verification status |
| SBTi | Total CO2e, target coverage, progress vs base year, reduction actions |
| SB 253 | Total CO2e, methodology, assurance opinion, third-party verification |
| GRI | Total CO2e, gases included, base year, standards used, methodology |

### 4.6 Performance Targets

| Metric | Target |
|--------|--------|
| Single employee calculation | < 50 ms |
| Batch 100 employees | < 2 seconds |
| Batch 1,000 employees | < 15 seconds |
| Batch 10,000 employees | < 120 seconds |
| Survey processing (1,000 responses) | < 5 seconds |
| EF lookup latency | < 3 ms |
| Grid EF lookup (with eGRID) | < 5 ms |
| Compliance check (7 frameworks) | < 200 ms |
| Uncertainty analysis (10,000 Monte Carlo) | < 5 seconds |
| API response (single calculation) | < 100 ms |
| Memory per calculation | < 2 MB |
| Database write (single result) | < 20 ms |
| Provenance chain seal | < 10 ms |
| Survey extrapolation (10,000 employees) | < 10 seconds |

---

## 5. Acceptance Criteria

### 5.1 Commute Mode Calculations
- [ ] All 12 vehicle types with correct DEFRA EFs
- [ ] SOV distance-based: distance x 2 x working_days x frequency x EF_vkm
- [ ] SOV fuel-based: litres x EF_per_litre (6 fuel types)
- [ ] Carpool: vehicle emissions / occupancy (2-5 persons)
- [ ] Vanpool: van emissions / occupancy (7-15 persons)
- [ ] 6 transit types with per-pkm EFs
- [ ] Motorcycle/moped calculations
- [ ] E-bike/e-scooter with grid-based emissions
- [ ] Active transport (cycling/walking) tracked as zero emissions
- [ ] WTT for all motorized modes
- [ ] Multi-modal trip segmentation (up to 5 legs)
- [ ] Park-and-ride composite calculation
- [ ] Distance band classification for aggregation

### 5.2 Telework Calculations
- [ ] Home office kWh estimation (laptop + heating + lighting + cooling)
- [ ] 19 country grid emission factors (IEA 2024)
- [ ] 26 US eGRID subregional factors
- [ ] Seasonal heating/cooling adjustment (4 quarters)
- [ ] Avoided commute calculation (memo item only)
- [ ] Net telework impact analysis
- [ ] Hybrid work frequency modeling (1-5 days/week)
- [ ] Equipment lifecycle emissions (amortized, optional memo)
- [ ] Space allocation: dedicated vs shared home office

### 5.3 Survey Processing
- [ ] Survey response parsing and validation
- [ ] Statistical extrapolation to full workforce
- [ ] Response rate weighting
- [ ] Confidence interval calculation (95%, 99%)
- [ ] Stratified sampling support
- [ ] Working days normalization by 12 regions
- [ ] Part-time employee handling (4 FTE levels)
- [ ] Compressed work week adjustment (3 schedule types)
- [ ] Minimum sample size validation

### 5.4 Average-Data Method
- [ ] 11 country average commute distances
- [ ] Regional mode share defaults (US/UK/EU/Global)
- [ ] Working days by 12 regions
- [ ] FTE headcount scaling

### 5.5 Spend-Based Method
- [ ] 8 NAICS commuting categories with EEIO factors
- [ ] 12 currency conversions to USD
- [ ] CPI deflation to base year (2021)
- [ ] Transit subsidy and parking cost processing

### 5.6 Category Boundary & Double-Counting
- [ ] 10 double-counting prevention rules (DC-EC-001 to DC-EC-010)
- [ ] Company vehicle exclusion
- [ ] Business travel exclusion
- [ ] Telework/Scope 2 separation
- [ ] EV office charging vs home charging separation
- [ ] Shuttle ownership boundary test

### 5.7 Data Quality & Uncertainty
- [ ] 5-dimension DQI scoring with configurable weights
- [ ] DQI classification (Excellent/Good/Fair/Poor/Very Poor)
- [ ] Monte Carlo uncertainty (10,000 iterations)
- [ ] Analytical uncertainty propagation
- [ ] Survey sampling error propagation
- [ ] Confidence intervals for survey extrapolation

### 5.8 Compliance
- [ ] 7 framework compliance checks
- [ ] 12 compliance rules (CR-EC-001 to CR-EC-012)
- [ ] Telework disclosure validation (CSRD)
- [ ] Mode share reporting validation (CDP)
- [ ] Survey methodology documentation check
- [ ] Materiality threshold validation

### 5.9 Infrastructure
- [ ] 22 REST API endpoints operational
- [ ] V071 migration with 16 tables, 3 hypertables, 2 continuous aggregates
- [ ] SHA-256 provenance chain across 10 pipeline stages
- [ ] 14 Prometheus metrics with `gl_ec_` prefix
- [ ] JWT auth with 22 PERMISSION_MAP entries
- [ ] 700+ unit tests with >90% coverage
- [ ] All Decimal calculations (no floating-point)
- [ ] JSON/CSV/Excel/PDF export
- [ ] RLS tenant isolation on operational tables
- [ ] Thread-safe singleton engines

---

## 6. Prometheus Metrics (14)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_ec_calculations_total` | Counter | method, mode, status |
| 2 | `gl_ec_emissions_kg_co2e_total` | Counter | mode, category |
| 3 | `gl_ec_commutes_total` | Counter | mode, vehicle_type |
| 4 | `gl_ec_telework_days_total` | Counter | country |
| 5 | `gl_ec_employees_processed_total` | Counter | method |
| 6 | `gl_ec_factor_selections_total` | Counter | source, mode |
| 7 | `gl_ec_compliance_checks_total` | Counter | framework, status |
| 8 | `gl_ec_batch_jobs_total` | Counter | status |
| 9 | `gl_ec_calculation_duration_seconds` | Histogram | method, mode |
| 10 | `gl_ec_batch_size` | Histogram | -- |
| 11 | `gl_ec_active_calculations` | Gauge | -- |
| 12 | `gl_ec_survey_response_rate` | Gauge | -- |
| 13 | `gl_ec_extrapolation_factor` | Gauge | -- |
| 14 | `gl_ec_mode_share_pct` | Gauge | mode |

---

## 7. Auth Integration

### 7.1 PERMISSION_MAP Entries (22)

| # | Method | Path Pattern | Permission |
|---|--------|-------------|------------|
| 1 | POST | `/api/v1/employee-commuting/calculate` | employee-commuting:calculate |
| 2 | POST | `/api/v1/employee-commuting/calculate/batch` | employee-commuting:calculate |
| 3 | POST | `/api/v1/employee-commuting/calculate/commute` | employee-commuting:calculate |
| 4 | POST | `/api/v1/employee-commuting/calculate/telework` | employee-commuting:calculate |
| 5 | POST | `/api/v1/employee-commuting/calculate/survey` | employee-commuting:calculate |
| 6 | POST | `/api/v1/employee-commuting/calculate/average-data` | employee-commuting:calculate |
| 7 | POST | `/api/v1/employee-commuting/calculate/spend` | employee-commuting:calculate |
| 8 | POST | `/api/v1/employee-commuting/calculate/multi-modal` | employee-commuting:calculate |
| 9 | GET | `/api/v1/employee-commuting/calculations/{id}` | employee-commuting:read |
| 10 | GET | `/api/v1/employee-commuting/calculations` | employee-commuting:read |
| 11 | DELETE | `/api/v1/employee-commuting/calculations/{id}` | employee-commuting:delete |
| 12 | GET | `/api/v1/employee-commuting/emission-factors` | employee-commuting:read |
| 13 | GET | `/api/v1/employee-commuting/emission-factors/{mode}` | employee-commuting:read |
| 14 | GET | `/api/v1/employee-commuting/commute-modes` | employee-commuting:read |
| 15 | GET | `/api/v1/employee-commuting/working-days/{region}` | employee-commuting:read |
| 16 | GET | `/api/v1/employee-commuting/commute-averages` | employee-commuting:read |
| 17 | GET | `/api/v1/employee-commuting/grid-factors/{country}` | employee-commuting:read |
| 18 | POST | `/api/v1/employee-commuting/compliance/check` | employee-commuting:compliance |
| 19 | POST | `/api/v1/employee-commuting/uncertainty/analyze` | employee-commuting:analyze |
| 20 | GET | `/api/v1/employee-commuting/aggregations/{period}` | employee-commuting:read |
| 21 | POST | `/api/v1/employee-commuting/mode-share/analyze` | employee-commuting:analyze |
| 22 | GET | `/api/v1/employee-commuting/provenance/{id}` | employee-commuting:read |

### 7.2 Router Registration

```python
# Import
from greenlang.employee_commuting.setup import get_router as get_ec_router

# Include
ec_router = get_ec_router()
app.include_router(ec_router)

# Protect
protect_routes(app, PERMISSION_MAP)
```

---

## 8. Configuration

### 8.1 Environment Variables (GL_EC_ prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_EC_ENABLED` | `true` | Enable/disable agent |
| `GL_EC_LOG_LEVEL` | `INFO` | Logging level |
| `GL_EC_DB_URL` | -- | PostgreSQL connection string |
| `GL_EC_DB_POOL_MIN` | `2` | Minimum DB pool connections |
| `GL_EC_DB_POOL_MAX` | `10` | Maximum DB pool connections |
| `GL_EC_REDIS_URL` | -- | Redis cache connection |
| `GL_EC_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `GL_EC_DEFAULT_WORKING_DAYS` | `230` | Default annual working days |
| `GL_EC_DEFAULT_COMMUTE_FREQ` | `5` | Default commute days per week |
| `GL_EC_DEFAULT_REGION` | `GLOBAL` | Default region for EF lookup |
| `GL_EC_DEFAULT_GWP` | `AR6` | Default GWP version |
| `GL_EC_INCLUDE_WTT` | `true` | Include WTT in calculations |
| `GL_EC_INCLUDE_TELEWORK` | `true` | Include telework emissions |
| `GL_EC_TELEWORK_DAILY_KWH` | `3.5` | Default daily kWh for telework |
| `GL_EC_SEASONAL_ADJUSTMENT` | `FULL_SEASONAL` | Default seasonal method |
| `GL_EC_SURVEY_MIN_RESPONSE_RATE` | `0.10` | Minimum response rate (10%) |
| `GL_EC_MONTE_CARLO_ITERATIONS` | `10000` | Monte Carlo iterations |
| `GL_EC_CONFIDENCE_LEVEL` | `0.95` | Confidence level for CI |
| `GL_EC_BATCH_MAX_SIZE` | `10000` | Maximum batch size |
| `GL_EC_EXPORT_FORMATS` | `JSON,CSV,EXCEL,PDF` | Enabled export formats |
| `GL_EC_PROVENANCE_ENABLED` | `true` | Enable provenance chain |
| `GL_EC_DQI_WEIGHTS` | `0.30,0.25,0.15,0.15,0.15` | DQI dimension weights |
| `GL_EC_MAX_TRIP_LEGS` | `5` | Maximum multi-modal legs |
| `GL_EC_EF_FALLBACK_ENABLED` | `true` | Enable EF source fallback |

### 8.2 Configuration Sections (15)

1. **DatabaseConfig** -- Connection pool, timeouts, retry policy
2. **CacheConfig** -- Redis URL, TTL, key prefix
3. **CalculationConfig** -- Decimal precision, rounding mode, GWP version
4. **WorkingDaysConfig** -- Default region, holidays, PTO, sick days
5. **TeleworkConfig** -- Daily kWh, seasonal factors, grid EF source, equipment lifecycle
6. **SurveyConfig** -- Min response rate, confidence level, extrapolation method, stratification
7. **SpendConfig** -- Base year, currency rates, CPI deflators
8. **ComplianceConfig** -- Enabled frameworks, materiality threshold, required disclosures
9. **UncertaintyConfig** -- Method, iterations, distribution parameters
10. **DQIConfig** -- Dimension weights, classification thresholds
11. **BatchConfig** -- Max size, concurrency, error handling
12. **ExportConfig** -- Enabled formats, template paths
13. **ProvenanceConfig** -- Hash algorithm, chain validation interval
14. **MetricsConfig** -- Prometheus endpoint, metric prefix, enabled metrics
15. **SecurityConfig** -- RLS enforcement, tenant isolation, auth integration

---

## 9. Non-Functional Requirements

### 9.1 Performance

| Requirement | Specification |
|-------------|--------------|
| Throughput | 100 calculations/second sustained |
| Latency p50 | < 50 ms per employee |
| Latency p99 | < 500 ms per employee |
| Batch p99 | < 15 seconds for 1,000 employees |
| Survey processing | < 5 seconds for 1,000 responses |
| Memory | < 512 MB for 10,000-employee batch |
| CPU | < 2 cores sustained for batch processing |
| Database connections | Max 10 per instance |

### 9.2 Availability

| Requirement | Specification |
|-------------|--------------|
| SLA | 99.9% uptime |
| Planned downtime | < 4 hours/month (during maintenance windows) |
| Recovery Time Objective | < 15 minutes |
| Recovery Point Objective | < 1 minute (WAL replication) |
| Health check | `/health` endpoint, 5-second interval |
| Circuit breaker | Opens after 5 failures, 30-second half-open |

### 9.3 Security

| Requirement | Specification |
|-------------|--------------|
| Authentication | JWT RS256 via SEC-001 |
| Authorization | RBAC with 5 permissions (calculate, read, delete, compliance, analyze) |
| Tenant isolation | RLS on all operational tables |
| Encryption at rest | AES-256-GCM via SEC-003 |
| Encryption in transit | TLS 1.3 via SEC-004 |
| PII protection | Employee IDs, home locations pseudonymized via SEC-011 |
| Audit logging | All mutations logged via SEC-005 |
| Input validation | Pydantic strict mode, max field sizes |

### 9.4 Observability

| Requirement | Specification |
|-------------|--------------|
| Metrics | 14 Prometheus metrics with `gl_ec_` prefix |
| Logging | Structured JSON via OBS-009 (Loki) |
| Tracing | OpenTelemetry spans via OBS-003 |
| Dashboards | Grafana dashboard via OBS-002 |
| Alerts | SLO-based via OBS-004/005 |
| Provenance | SHA-256 chain with 10-stage pipeline |

---

## 10. Test Suite (700+ tests)

### 10.1 Test Files (15)

| # | File | Tests | Coverage Area |
|---|------|-------|---------------|
| 1 | `test_models.py` | 65 | Enums, constants, input/result models, validation |
| 2 | `test_config.py` | 45 | Configuration, env loading, singleton, defaults |
| 3 | `test_metrics.py` | 25 | Prometheus metrics, fallback, labels |
| 4 | `test_provenance.py` | 35 | SHA-256 chain, validation, Merkle root |
| 5 | `test_employee_commuting_database.py` | 55 | EF lookups, working days, mode shares, eGRID |
| 6 | `test_commute_mode_calculator.py` | 95 | SOV, carpool, vanpool, transit, active, WTT, multi-modal |
| 7 | `test_telework_calculator.py` | 55 | Home energy, grid EF, avoided commute, seasonal |
| 8 | `test_survey_processor.py` | 65 | Extrapolation, weighting, confidence, stratification |
| 9 | `test_spend_based_calculator.py` | 35 | EEIO, currency, CPI, NAICS categories |
| 10 | `test_compliance_checker.py` | 60 | 7 frameworks, telework, double-counting, 12 rules |
| 11 | `test_employee_commuting_pipeline.py` | 55 | Pipeline, batch, aggregation, mode share |
| 12 | `test_setup.py` | 25 | Service facade, singleton, router registration |
| 13 | `test_api.py` | 45 | FastAPI endpoints, auth, validation |
| 14 | `conftest.py` | 5 | Shared fixtures |
| 15 | `__init__.py` | -- | Package init |
| **Total** | | **700+** | |

### 10.2 Key Test Scenarios

**Commute Calculations:**
- SOV 20km one-way, medium petrol, 5 days/week, 230 working days --> verify annual CO2e = 20 x 2 x 230 x 1.0 x 0.25594 = 2,354.65 kgCO2e
- SOV 20km one-way, BEV, 5 days/week, 230 working days --> verify annual CO2e = 20 x 2 x 230 x 1.0 x 0.07005 = 644.46 kgCO2e (73% reduction)
- Carpool 30km, 3 occupants, medium petrol --> CO2e = (30 x 2 x 230 x 0.25594) / 3 = 1,176.32 kgCO2e
- Metro 15km, 5 days/week --> CO2e = 15 x 2 x 230 x 0.02781 = 192.01 kgCO2e
- Cycling 8km --> zero emissions, tracked in mode share
- Hybrid work: 3 days office SOV 20km, 2 days telework --> frequency = 0.6
- Multi-modal: drive 8km + rail 25km + walk 0.5km --> segment per leg
- E-bike 10km --> CO2e = 10 x 2 x 230 x 0.005 = 23.0 kgCO2e
- Vanpool 25km, 10 occupants --> verify van EF / 10

**Fuel-Based Calculations:**
- Employee reports 1,500L petrol/year --> CO2e = 1500 x 2.31480 = 3,472.20 kgCO2e
- Employee reports 8.5 L/100km fuel economy, 20km one-way --> Annual fuel = (20 x 2 x 230) / (100/8.5) = 782.35L
- Diesel vehicle, 45 mpg --> convert to L/100km then calculate

**Telework:**
- Full remote, UK grid (0.207 kgCO2e/kWh), 3.5 kWh/day, 230 days --> CO2e = 230 x 3.5 x 0.20707 = 166.69 kgCO2e
- Full remote, India grid (0.708) --> CO2e = 230 x 3.5 x 0.708 = 569.94 kgCO2e
- Hybrid 3 days office/2 days WFH, US (California eGRID) --> 2 days x 46 weeks = 92 telework days
- Avoided commute memo: Full remote employee who would commute 25km SOV --> avoided = 25 x 2 x 230 x 0.27145 = 3,123.68 kgCO2e
- Net impact: telework CO2e (166.69) - avoided commute (3,123.68) = -2,956.99 kgCO2e net reduction
- Seasonal adjustment: Q1 heating 1.3x, Q3 cooling 1.2x

**Survey Processing:**
- 500 responses from 2,000 employees --> extrapolation factor = 4.0
- Stratified: Site A (200/800), Site B (300/1200) --> per-site extrapolation
- Response rate weighting: Site A 25%, Site B 25% --> weighted average
- 95% confidence interval with 500 responses: margin = +/-4.4%
- Minimum sample size for 5% margin: 385 needed
- Low response rate (<10%) --> flag warning, recommend average-data fallback

**Average-Data Method:**
- 5,000 US employees, no survey --> 5000 x 21.7km x 2 x 225 x SUM(mode_share x EF)
- Verify mode share weighted EF for US: (0.728 x 0.27145 + 0.089 x 0.27145/2.5 + 0.049 x 0.10312 + ...)

**Compliance:**
- CSRD check: telework emissions disclosed --> PASS
- CSRD check: telework emissions not disclosed --> WARNING (increasingly required)
- CDP: mode share breakdown present --> PASS
- CDP: mode share missing --> FAIL
- Company vehicle submitted --> REJECT (DC-EC-001, Scope 1)
- Business travel trip submitted --> REJECT (DC-EC-002, Cat 6)
- SB 253: third-party assurance flag present --> PASS
- Materiality check: Cat 7 = 0.5% of Scope 3 --> WARNING (below 1% threshold)

**Edge Cases:**
- Employee with 0 km commute distance and not telework --> reject with validation error
- Employee with > 200 km one-way commute --> flag for review (outlier)
- Occupancy = 1 for carpool --> reclassify as SOV
- Negative distance --> reject
- Future date working days --> reject
- Survey with 0 respondents --> reject
- CNG fuel for personal vehicle --> use CNG EF per kg
- Part-time 50% employee, 3 days/week --> 0.5 x 3/5 x 230 = 69 effective days

---

## 11. Key Differentiators from Adjacent Categories

### 11.1 Category 7 (Employee Commuting) vs Category 6 (Business Travel)

| Aspect | Category 7 (Commuting) | Category 6 (Business Travel) |
|--------|-------|------|
| Trip purpose | Regular home-to-office commute | Work travel away from regular site |
| Frequency | Daily/regular | Irregular, event-driven |
| Data source | Employee surveys, HR data | TMC/T&E booking systems |
| Telework included | Yes (optional) | No |
| Hotel included | No | Yes |
| Air travel included | No (extremely rare) | Yes (primary mode) |
| Survey extrapolation | Yes (core methodology) | No (trip-level data) |
| Mode share analysis | Yes (key output) | Secondary |
| Agent | AGENT-MRV-020 | AGENT-MRV-019 |

### 11.2 Category 7 vs Scope 1 (Company Vehicles)

| Aspect | Category 7 | Scope 1 |
|--------|-------|------|
| Vehicle ownership | NOT owned by company | Owned/leased by company |
| Control | No operational control | Operational control |
| Reporting | Indirect (Scope 3) | Direct (Scope 1) |
| Agent | AGENT-MRV-020 | AGENT-MRV-003 |

### 11.3 Category 7 vs Scope 2 (Telework Electricity)

| Aspect | Category 7 Telework | Scope 2 Office Electricity |
|--------|-------|------|
| Location | Employee's home | Company's office/facility |
| Control | No operational control | Operational or financial control |
| Grid EF | Home location-specific | Office location-specific |
| Agent | AGENT-MRV-020 | AGENT-MRV-009/010 |

### 11.4 Category 7 vs Category 3 (Fuel & Energy Upstream)

| Aspect | Category 7 WTT | Category 3 WTT |
|--------|-------|------|
| Scope | WTT for commute fuels | WTT for company's own fuel/energy |
| Inclusion | Embedded in Cat 7 EFs | Separate Cat 3 calculation |
| Double-counting | DC-EC-008 prevents overlap | Cat 3 excludes Cat 7 WTT |
| Agent | AGENT-MRV-020 | AGENT-MRV-016 |

---

## 12. Intervention Modeling

### 12.1 Scenario Analysis

The Employee Commuting Agent supports modeling of emissions reduction interventions by adjusting mode share, commute frequency, or distances:

```
Intervention Types:
  1. Modal Shift: Move X% of SOV commuters to transit/cycling/EV
  2. Frequency Reduction: Increase telework days per week
  3. Distance Reduction: Encourage relocation or satellite offices
  4. Fleet Electrification: Increase BEV share of SOV commuters
  5. Transit Subsidies: Model uptake of employer-provided transit passes
  6. Cycle-to-Work: Model uptake of cycling schemes
  7. Compressed Work Weeks: 4x10 or 9/80 schedules

Scenario Calculation:
  Baseline_CO2e = current mode share x EFs x employees x working days
  Scenario_CO2e = modified mode share x EFs x employees x working days
  Reduction = Baseline_CO2e - Scenario_CO2e
  Reduction_% = Reduction / Baseline_CO2e x 100
```

### 12.2 Common Intervention Impacts

| Intervention | Typical Reduction | Implementation Effort |
|-------------|------------------|----------------------|
| 10% SOV to transit | 5-8% total Cat 7 | Medium (subsidies, infrastructure) |
| 1 extra telework day/week | 15-20% total Cat 7 | Low (policy change) |
| 10% SOV to BEV | 7-10% total Cat 7 | Medium (EV incentives, charging) |
| Cycle-to-work scheme | 2-4% total Cat 7 | Low (scheme setup) |
| Compressed 4x10 schedule | 15-20% total Cat 7 | Medium (policy, culture) |
| Carpool matching program | 3-5% total Cat 7 | Low (platform, incentives) |
| Shuttle service | 5-10% total Cat 7 | High (vehicle, routes, drivers) |

---

## 13. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic 2.x | Data models and validation |
| FastAPI 0.109+ | REST API framework |
| prometheus_client 0.20+ | Metrics collection |
| psycopg 3.x + psycopg_pool | Async PostgreSQL |
| TimescaleDB 2.x | Time-series hypertables |
| numpy 1.26+ | Statistical calculations (CI, Monte Carlo) |
| AGENT-MRV-003 (Mobile Combustion) | Cross-reference: Scope 1 fleet exclusion |
| AGENT-MRV-019 (Business Travel) | Cross-reference: Cat 6 boundary check |
| AGENT-MRV-009/010 (Scope 2) | Cross-reference: Telework grid EF, Scope 2 separation |
| AGENT-MRV-016 (Fuel & Energy) | Cross-reference: Cat 3 WTT double-counting |
| AGENT-DATA-002 (Excel/CSV Normalizer) | Survey data ingestion |
| AGENT-DATA-003 (ERP Connector) | HR headcount data |
| AGENT-FOUND-001 (Orchestrator) | DAG execution |
| AGENT-FOUND-005 (Citations & Evidence) | EF source citations |
| AGENT-FOUND-008 (Reproducibility) | Determinism verification |
| AGENT-FOUND-009 (QA Test Harness) | Golden-file testing |

---

## 14. Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Low survey response rates | High | Medium | Support average-data fallback; gamification; management communication |
| Inaccurate self-reported distances | Medium | Medium | Geocoding validation; distance cap alerts (>200km flagged) |
| Telework energy estimates high variance | High | Low | Use peer-reviewed IEA/DEFRA defaults; allow company-specific override |
| Hybrid work patterns changing rapidly | Medium | Medium | Configurable frequency; re-survey recommended annually |
| eGRID/IEA factor updates mid-year | Low | Low | Versioned EF database; lock EFs per reporting period |
| CSRD telework requirement changes | Medium | Medium | Modular telework engine; configurable inclusion/exclusion |
| Double-counting with Cat 6 | Medium | High | Cross-agent validation via AGENT-MRV-019 boundary checks |
| Part-time/contractor classification | Medium | Low | Configurable FTE thresholds; contractor exclusion option |
| Multi-modal trip complexity | Low | Medium | 5-leg maximum; default WALK for unknown segments |
| Survey bias (self-selection) | High | Medium | Stratified sampling; non-response weighting |

---

## 15. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-25 | Initial PRD for AGENT-MRV-020 Employee Commuting Agent |
| 1.1.0 | 2026-02-26 | Expanded to comprehensive specification: added multi-modal trips, eGRID factors, intervention modeling, expanded configuration, 700+ test target, 22 API endpoints, 10 double-counting rules, 12 compliance rules, risk analysis |
