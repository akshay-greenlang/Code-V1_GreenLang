# Building Performance Standards - US/EU Regulatory Requirements
**Multiple Jurisdictions - Rolling Deadlines 2025-2027**

## Executive Summary
Building Performance Standards (BPS) are outcome-based policies requiring existing buildings to meet energy or emissions performance targets, with various jurisdictions implementing different approaches and timelines.

## Key Requirements by Jurisdiction

### European Union - Energy Performance of Buildings Directive (EPBD)
**Directive (EU) 2024/1275 - Recast**

#### Requirements
- **Zero-emission buildings**: New buildings by 2028 (public) / 2030 (all)
- **Minimum Energy Performance Standards (MEPS)**:
  - Worst performing 15% renovated by 2030
  - Worst performing 25% by 2033
- **Energy Performance Certificates**: Required for all buildings
- **Solar installations**: Mandatory for new buildings >250m²
- **Building Renovation Passports**: By 2025

#### Deadlines
- **2025**: National Building Renovation Plans due
- **2027**: All new public buildings zero-emission
- **2028**: Solar on all new non-residential >250m²
- **2030**: All new buildings zero-emission
- **2033**: Worst performing residential improved

### United States - Federal and Local Standards

#### Federal - Building Performance Standard (GSA/DOE)
- **Target**: 30% energy reduction by 2030 (from 2003 baseline)
- **Covered buildings**: Federal facilities >25,000 sq ft
- **Reporting**: Annual via ENERGY STAR Portfolio Manager

#### New York City - Local Law 97
- **2024-2029 Limits**:
  - Offices: 9.96 kgCO2e/sq ft
  - Multifamily: 6.75 kgCO2e/sq ft
  - Retail: 11.81 kgCO2e/sq ft
- **2030-2034 Limits**: 40% more stringent
- **Covered buildings**: >25,000 sq ft
- **Penalties**: $268 per metric ton over limit

#### Washington DC - Building Energy Performance Standards (BEPS)
- **Cycle 1 (2021-2026)**: 20% below median ENERGY STAR score
- **Cycle 2 (2027-2031)**: Site EUI standards by property type
- **Covered buildings**: >25,000 sq ft (>10,000 sq ft from 2027)

#### Boston - Building Emissions Reduction and Disclosure Ordinance (BERDO)
- **2025-2029**: 50% reduction from baseline
- **2030-2034**: 65% reduction
- **2035+**: Net zero
- **Covered buildings**: >35,000 sq ft (>20,000 from 2025)

## Reporting Templates and Formats

### EU Energy Performance Certificate Data
```json
{
  "building_id": "EU-EPC-2025-XXXXX",
  "general_information": {
    "address": "Building Address",
    "construction_year": 1995,
    "building_type": "office",
    "gross_floor_area": 5000,
    "conditioned_area": 4500
  },
  "energy_performance": {
    "energy_class": "C",
    "primary_energy_use": 150,
    "primary_energy_use_unit": "kWh/m²/year",
    "co2_emissions": 35,
    "co2_emissions_unit": "kgCO2/m²/year",
    "renewable_energy_ratio": 0.20
  },
  "technical_systems": {
    "heating": {
      "type": "gas_boiler",
      "efficiency": 0.85,
      "age": 10
    },
    "cooling": {
      "type": "air_conditioning",
      "cop": 3.2
    },
    "ventilation": {
      "type": "mechanical_heat_recovery",
      "efficiency": 0.75
    },
    "hot_water": {
      "type": "gas_boiler",
      "solar_thermal": false
    },
    "lighting": {
      "type": "led",
      "coverage": 0.80
    },
    "renewable_systems": {
      "solar_pv": {
        "capacity_kwp": 50,
        "annual_generation": 45000
      }
    }
  },
  "recommendations": [
    {
      "measure": "Insulation upgrade",
      "energy_savings": 25,
      "cost_estimate": 50000,
      "payback_years": 7
    }
  ]
}
```

### US Building Performance Report (ENERGY STAR)
```json
{
  "property_information": {
    "pm_property_id": 12345678,
    "property_name": "Building Name",
    "address": "US Address",
    "property_type": "Office",
    "gross_floor_area": 100000,
    "year_built": 1990
  },
  "energy_performance": {
    "reporting_year": 2025,
    "energy_star_score": 75,
    "site_eui": 65.5,
    "source_eui": 180.2,
    "weather_normalized_eui": 68.0,
    "total_ghg_emissions": 520,
    "total_ghg_intensity": 5.2
  },
  "energy_consumption": {
    "electricity": {
      "usage": 3500000,
      "unit": "kWh",
      "cost": 350000
    },
    "natural_gas": {
      "usage": 50000,
      "unit": "therms",
      "cost": 45000
    },
    "district_steam": {
      "usage": 0,
      "unit": "klbs"
    }
  },
  "benchmarking_metrics": {
    "national_median_eui": 69.0,
    "performance_target": 55.0,
    "percent_better_than_median": 5
  }
}
```

## Data Requirements and Sources

### Building Characteristics
- Gross floor area (GFA)
- Conditioned floor area
- Building type and use
- Occupancy patterns
- Operating hours
- Construction year
- Envelope characteristics

### Energy Consumption Data
- Monthly utility bills (electricity, gas, oil, district energy)
- Sub-metered data by end use
- On-site renewable generation
- Peak demand data
- Power factor and load profiles

### Operational Data
- HVAC system types and age
- Control system setpoints
- Maintenance records
- Equipment efficiency ratings
- Occupancy sensors data
- Indoor environmental quality metrics

### Data Sources
- Utility data platforms
- Building Management Systems (BMS)
- Energy management systems
- Smart meter data
- Weather stations
- Occupancy tracking systems

## Calculation Methodologies

### Energy Use Intensity (EUI)
```
Site EUI = Total Site Energy (kBtu) / Gross Floor Area (sq ft)

Source EUI = Σ(Energy Type × Source Factor) / GFA

Where Source Factors (US):
- Electricity: 2.80
- Natural Gas: 1.05
- District Steam: 1.20
- District Chilled Water: 1.04
```

### EU Primary Energy Calculation
```
Primary Energy = Σ(Final Energy × Primary Energy Factor)

Where PEF (EU defaults):
- Grid Electricity: 2.1
- Natural Gas: 1.1
- District Heating: 1.3
- Renewable Energy: 0.0
```

### GHG Emissions Calculations
```
Total Emissions = Σ(Energy Consumption × Emission Factor)

Emission Factors vary by:
- Grid region
- Fuel type
- Time of use (hourly factors)
- Renewable energy certificates
```

### Building Performance Score (NYC LL97)
```
Carbon Intensity = Annual GHG Emissions (tCO2e) / Gross Floor Area (sq ft)

Compliance = Carbon Intensity ≤ Building Type Limit
```

## Penalties for Non-Compliance

### EU EPBD Penalties
- **Member State Specific**: Each country sets penalties
- **Typical ranges**: €500-€50,000 per building
- **Recurring fines**: Monthly until compliance
- **Market restrictions**: Sale/rent prohibitions for worst performers

### US Jurisdiction Penalties

#### New York City (LL97)
- **Standard penalty**: $268 per metric ton CO2e over limit
- **False reporting**: Up to $500,000
- **Failure to report**: $50,000 per building

#### Washington DC (BEPS)
- **Alternative Compliance Payment**: $10 per square foot
- **Maximum penalty**: $7.5 million per building
- **Failure to report**: Up to $100 per day

#### Boston (BERDO)
- **Standard rate**: $234 per metric ton CO2e
- **Maximum**: $1,000 per metric ton
- **Hardship compliance available**

## Scope - Which Buildings Must Comply

### EU Coverage
- **All buildings**: With limited exemptions
- **Priority**: Public buildings and worst performers
- **Exemptions**:
  - Historic buildings (if protected)
  - Temporary buildings (<2 years)
  - Religious buildings
  - Industrial/agricultural buildings

### US Coverage Thresholds

#### Typical Coverage
- **Large buildings**: Generally >25,000 sq ft
- **Groups of buildings**: On same tax lot
- **Mixed use**: Primary use determines category

#### Property Types (Varies by jurisdiction)
- Commercial offices
- Multifamily residential
- Hotels
- Retail
- Industrial
- Healthcare
- Education

#### Common Exemptions
- Houses of worship (some jurisdictions)
- Industrial process loads
- Buildings with low energy use
- Affordable housing (with alternatives)

## Implementation Requirements

### Technical Systems Needed

#### Energy Monitoring Platform
- Real-time energy monitoring
- Automated utility data collection
- Weather normalization
- Anomaly detection
- Predictive analytics

#### Building Analytics System
- Energy model calibration
- Measure identification
- Cost-benefit analysis
- Performance tracking
- Fault detection and diagnostics

#### Compliance Management
- Deadline tracking
- Document management
- Audit preparation
- Penalty calculations
- Alternative compliance paths

### Key Implementation Steps

1. **Baseline Assessment**
   - Energy audit
   - Building characteristics survey
   - System inventory
   - Performance benchmarking

2. **Target Setting**
   - Regulatory requirement analysis
   - Performance gap assessment
   - Intermediate milestone setting

3. **Action Planning**
   - Energy Conservation Measures (ECMs)
   - Capital planning
   - Financing options
   - Implementation schedule

4. **Implementation**
   - Retrofit execution
   - Commissioning
   - Measurement & Verification
   - Occupant engagement

5. **Compliance Reporting**
   - Data quality assurance
   - Report preparation
   - Third-party verification
   - Submission to authorities

## Technology Solutions

### Immediate Measures
- LED lighting upgrades
- HVAC controls optimization
- Building envelope sealing
- Occupancy-based controls
- Power factor correction

### Medium-term Investments
- HVAC system upgrades
- Building automation systems
- Heat recovery ventilation
- Variable frequency drives
- Insulation improvements

### Deep Retrofits
- Facade replacement
- Electrification of heating
- Ground source heat pumps
- On-site renewable energy
- Thermal energy storage