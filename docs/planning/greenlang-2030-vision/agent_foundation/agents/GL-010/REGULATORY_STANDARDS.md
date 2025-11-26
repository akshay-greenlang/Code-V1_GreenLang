# GL-010 EMISSIONWATCH Regulatory Standards Compliance

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Compliance Scope:** US EPA, EU IED, ISO Standards, China MEE

---

## Overview

This document details how GL-010 EMISSIONWATCH implements compliance with major environmental regulatory standards worldwide. Each standard is documented with its requirements, GL-010 implementation approach, calculation methods used, and supported reporting formats.

---

## Table of Contents

1. [EPA Clean Air Act](#1-epa-clean-air-act)
2. [40 CFR Part 60 - NSPS](#2-40-cfr-part-60---nsps)
3. [40 CFR Part 75 - Acid Rain Program](#3-40-cfr-part-75---acid-rain-program)
4. [40 CFR Part 98 - GHG Reporting](#4-40-cfr-part-98---ghg-reporting)
5. [EU Industrial Emissions Directive](#5-eu-industrial-emissions-directive)
6. [EU ETS MRV Regulation](#6-eu-ets-mrv-regulation)
7. [ISO 14064](#7-iso-14064)
8. [ISO 14001](#8-iso-14001)
9. [China MEE Standards](#9-china-mee-standards)

---

## 1. EPA Clean Air Act

### 1.1 Regulatory Overview

The Clean Air Act (CAA) is the primary federal law governing air quality in the United States. It establishes the framework for:

- National Ambient Air Quality Standards (NAAQS)
- New Source Performance Standards (NSPS)
- National Emission Standards for Hazardous Air Pollutants (NESHAPs)
- Title V Operating Permits
- Prevention of Significant Deterioration (PSD)

### 1.2 Requirements Summary

| Program | Applicability | Key Requirements |
|---------|---------------|------------------|
| NAAQS | All areas | Attainment of 6 criteria pollutants |
| NSPS | New/modified sources | Technology-based emission limits |
| MACT | Major HAP sources | Maximum achievable control |
| Title V | Major sources | Comprehensive permits |
| PSD | Major new sources | Best available control |

### 1.3 GL-010 Implementation

**Supported Programs:**
- NSPS compliance checking (40 CFR Part 60)
- MACT/NESHAP applicability
- Title V permit condition tracking
- PSD modeling inputs

**Calculation Methods:**
- EPA Method 19 (F-factors)
- EPA AP-42 (Emission factors)
- EPA Reference Methods (Stack testing)

**Reporting Formats:**
- EIS XML (National Emissions Inventory)
- Title V compliance certifications
- Deviation reports

### 1.4 Compliance Matrix

| CAA Requirement | GL-010 Tool | Implementation |
|-----------------|-------------|----------------|
| Emission limits | check_compliance_status | Real-time limit comparison |
| Monitoring | calculate_*_emissions | CEMS data processing |
| Reporting | generate_regulatory_report | Formatted submissions |
| Record keeping | generate_audit_trail | 5-year retention |

---

## 2. 40 CFR Part 60 - NSPS

### 2.1 Regulatory Overview

New Source Performance Standards (NSPS) establish technology-based emission limits for new, modified, or reconstructed sources in specific industrial categories.

### 2.2 Applicable Subparts

| Subpart | Category | GL-010 Support |
|---------|----------|----------------|
| D | Fossil Fuel Steam Generators | Full |
| Da | Electric Utility Steam Generators | Full |
| Db | Industrial-Commercial Steam Generators | Full |
| Dc | Small ICI Steam Generators | Full |
| GG | Stationary Gas Turbines | Full |
| KKKK | Stationary Combustion Turbines | Full |
| JJJJ | SI Reciprocating Engines | Full |
| IIII | CI Reciprocating Engines | Full |

### 2.3 Emission Limits Database

**Subpart D (Large Boilers >250 MMBtu/hr):**

| Pollutant | Fuel | Limit | Unit | Citation |
|-----------|------|-------|------|----------|
| NOx | Natural Gas | 0.20 | lb/MMBtu | 60.44(a) |
| NOx | Distillate Oil | 0.30 | lb/MMBtu | 60.44(a) |
| NOx | Coal | 0.70 | lb/MMBtu | 60.44(a) |
| PM | All | 0.030 | lb/MMBtu | 60.42(a) |
| Opacity | All | 20% | 6-min avg | 60.42(b) |

**Subpart Db (Industrial Boilers >100 MMBtu/hr):**

| Pollutant | Fuel | Limit | Unit | Citation |
|-----------|------|-------|------|----------|
| NOx | Natural Gas | 0.10 | lb/MMBtu | 60.44b(a)(1) |
| NOx | Distillate Oil | 0.20 | lb/MMBtu | 60.44b(a)(2) |
| SO2 | Coal | 0.50 | lb/MMBtu | 60.42b(a) |
| PM | All | 0.030 | lb/MMBtu | 60.43b |

**Subpart KKKK (Gas Turbines >850 MMBtu/hr):**

| Pollutant | Fuel | Limit | Unit | Citation |
|-----------|------|-------|------|----------|
| NOx | Natural Gas | 25 | ppm @ 15% O2 | 60.4320 |
| NOx | Distillate Oil | 42 | ppm @ 15% O2 | 60.4320 |
| SO2 | All | 110 | ng/J | 60.4330 |

### 2.4 GL-010 Calculation Methods

**Method 19 F-Factor Approach:**
```
E (lb/dscf) = C (ppm) * MW * (P/RT) * (10^-6)

E (lb/MMBtu) = E (lb/dscf) * Fd * (20.9 / (20.9 - %O2))

Where:
- Fd = Dry F-factor (dscf/MMBtu)
- MW = Molecular weight
- P = Standard pressure (29.92 inHg)
- T = Standard temperature (528 R)
```

**F-Factors (40 CFR 60 Appendix A):**

| Fuel | Fd (dscf/MMBtu) | Fw (wscf/MMBtu) | Fc (scf CO2/MMBtu) |
|------|-----------------|-----------------|-------------------|
| Natural Gas | 8,710 | 10,610 | 1,040 |
| Fuel Oil #2 | 9,190 | 10,320 | 1,420 |
| Fuel Oil #6 | 9,220 | 10,260 | 1,420 |
| Bituminous Coal | 9,780 | 10,640 | 1,800 |
| Subbituminous Coal | 9,820 | 10,580 | 1,840 |

### 2.5 Monitoring Requirements

| Parameter | Method | Frequency | GL-010 Support |
|-----------|--------|-----------|----------------|
| NOx | Method 7E or CEMS | Continuous | CEMS data processing |
| SO2 | Method 6C or CEMS | Continuous | CEMS data processing |
| PM | Method 5 | Annual | Stack test input |
| O2/CO2 | Method 3A or CEMS | Continuous | O2 correction |
| Opacity | Method 9 or COM | Continuous | Opacity tracking |

### 2.6 Reporting Formats

**Quarterly CEMS Reports:**
- Electronic Data Reporting (EDR) format
- XML schema per EPA specifications
- Submission via ECMPS

**Annual Compliance Certifications:**
- Excess emissions summary
- Monitoring system performance
- Calibration records

---

## 3. 40 CFR Part 75 - Acid Rain Program

### 3.1 Regulatory Overview

The Acid Rain Program established under Title IV of the Clean Air Act requires continuous emissions monitoring and reporting for affected electric utility units.

### 3.2 Applicability

| Source Type | Threshold | Affected |
|-------------|-----------|----------|
| Electric utility units | >25 MW | Yes |
| Cogeneration units | >25 MW, >1/3 sales | Yes |
| Industrial units | Opt-in available | Optional |

### 3.3 Monitoring Requirements

**Required CEMS Parameters:**

| Parameter | Method | Data Availability |
|-----------|--------|-------------------|
| SO2 concentration | CEMS | 95% minimum |
| NOx concentration | CEMS | 95% minimum |
| CO2 concentration | CEMS or calculated | 95% minimum |
| Stack flow rate | CEMS | 95% minimum |
| Diluent (O2 or CO2) | CEMS | 95% minimum |

**Quality Assurance Requirements (Appendix B):**

| Test | Frequency | Criteria |
|------|-----------|----------|
| Daily calibration error | Daily | <2.5% span |
| Linearity check | Quarterly | <5% |
| RATA | Annual | <10% bias |
| CGA | Quarterly | <5% |

### 3.4 GL-010 Implementation

**CEMS Data Processing:**
```python
# Part 75 compliant hourly calculation
def calculate_hourly_mass(concentration_ppm, flow_dscfh, mw):
    """
    Calculate hourly mass emissions per 40 CFR 75.

    Formula: lb/hr = ppm * flow (dscf/hr) * MW * K
    Where K = 1.66 x 10^-7 (conversion factor)
    """
    K = 1.66e-7
    return concentration_ppm * flow_dscfh * mw * K
```

**Missing Data Substitution:**
- <2 hours: Linear interpolation
- 2-24 hours: Prior 2160 quality-assured hours
- >24 hours: 90th percentile of prior 2160 hours

### 3.5 Reporting Requirements

**Quarterly Reports:**

| Element | Content | Format |
|---------|---------|--------|
| Unit ID information | Source identification | EDR XML |
| Operating data | Hours, load, heat input | EDR XML |
| Emission data | Hourly mass emissions | EDR XML |
| QA data | Calibrations, RATAs | EDR XML |
| Certification data | Accuracy tests | EDR XML |

**GL-010 Report Generation:**
```yaml
report_type: epa_part75_quarterly
format: edr_xml
schema_version: "3.0"
elements:
  - unit_identification
  - operating_hours
  - hourly_emissions
  - quarterly_totals
  - qa_test_results
  - certification_status
```

### 3.6 Compliance Calculations

**Heat Input (Method 19):**
```
HI (MMBtu/hr) = Flow (scfh) * HHV (Btu/scf) / 10^6 (for gas)
HI (MMBtu/hr) = Flow (lb/hr) * HHV (Btu/lb) / 10^6 (for solid/liquid)
```

**Mass Emissions:**
```
SO2 (lb/hr) = SO2 (ppm) * Flow (dscfh) * 64.06 * 1.66e-7
NOx (lb/hr) = NOx (ppm) * Flow (dscfh) * 46.01 * 1.66e-7
CO2 (short tons/hr) = CO2 (%) * Flow (scfh) * 44.01 * 5.18e-8
```

---

## 4. 40 CFR Part 98 - GHG Reporting

### 4.1 Regulatory Overview

The Greenhouse Gas Reporting Program (GHGRP) requires reporting of GHG emissions from large sources emitting 25,000 metric tons CO2e or more per year.

### 4.2 Applicable Subparts

| Subpart | Category | GL-010 Support |
|---------|----------|----------------|
| C | General Stationary Fuel Combustion | Full |
| D | Electricity Generation | Full |
| P | Hydrogen Production | Partial |
| W | Petroleum Refining | Partial |
| Y | Petroleum Refineries | Partial |

### 4.3 Calculation Methods (Subpart C)

**Tier 1 - Default Emission Factors:**
```
CO2 (metric tons) = Fuel * EF * 0.001

Where:
- Fuel = Quantity (MMBtu, scf, gallons, tons)
- EF = Default emission factor (kg CO2/unit)
```

**Default Emission Factors (Table C-1):**

| Fuel | EF (kg CO2/MMBtu) | EF (kg CH4/MMBtu) | EF (kg N2O/MMBtu) |
|------|-------------------|-------------------|-------------------|
| Natural Gas | 53.06 | 0.001 | 0.0001 |
| Distillate Oil | 73.96 | 0.003 | 0.0006 |
| Residual Oil | 75.10 | 0.003 | 0.0006 |
| Bituminous Coal | 93.28 | 0.011 | 0.0016 |
| Subbituminous | 97.17 | 0.011 | 0.0016 |

**Tier 2 - Facility-Specific HHV:**
```
CO2 = Fuel * HHV_facility * EF_C * (44/12) * 0.001
```

**Tier 3 - Carbon Content Analysis:**
```
CO2 = Fuel * CC_measured * (44/12) * Oxidation_Factor
```

**Tier 4 - CEMS:**
```
CO2 = Sum of hourly CO2 mass (from certified CEMS)
```

### 4.4 GL-010 Implementation

**Multi-Tier Calculator:**
```python
class GHGCalculator:
    def calculate_co2(self, fuel_data, tier):
        if tier == 1:
            return self._tier1_default_ef(fuel_data)
        elif tier == 2:
            return self._tier2_facility_hhv(fuel_data)
        elif tier == 3:
            return self._tier3_carbon_content(fuel_data)
        elif tier == 4:
            return self._tier4_cems(fuel_data)
```

### 4.5 Reporting Requirements

**Annual Report Elements:**

| Element | Content | e-GGRT Field |
|---------|---------|--------------|
| Facility ID | EPA facility identifier | GHGRP ID |
| Reporter info | Company/contact | Reporter details |
| Fuel consumption | By fuel type | Fuel quantity |
| GHG emissions | CO2, CH4, N2O, CO2e | Emissions data |
| Methodology | Tier used | Calculation method |
| Verification | QA/QC | Verification status |

**GL-010 Report Output:**
```json
{
  "report_type": "epa_part98_annual",
  "reporting_year": 2025,
  "facility_id": "GHGRP-12345",
  "emissions": {
    "co2_metric_tons": 125000,
    "ch4_metric_tons": 2.5,
    "n2o_metric_tons": 0.5,
    "co2e_metric_tons": 125200
  },
  "methodology": {
    "tier": 2,
    "subpart": "C"
  }
}
```

### 4.6 Deadline and Submission

| Milestone | Date | GL-010 Support |
|-----------|------|----------------|
| Data collection | Ongoing | CEMS integration |
| Report preparation | Feb 1-Mar 31 | Report generation |
| Submission | March 31 | e-GGRT XML |
| Corrections | Apr 1-Jun 30 | Amendment support |

---

## 5. EU Industrial Emissions Directive

### 5.1 Regulatory Overview

Directive 2010/75/EU (IED) is the main EU instrument regulating pollutant emissions from industrial installations through integrated permitting and BAT requirements.

### 5.2 Scope

| Sector | Activity | Threshold |
|--------|----------|-----------|
| Energy | Combustion | >50 MW thermal |
| Metals | Ferrous/non-ferrous | Various |
| Minerals | Cement, glass | Various |
| Chemicals | Organic/inorganic | Various |
| Waste | Incineration | >3 t/hr |
| Other | Pulp/paper, food | Various |

### 5.3 BAT-AEL Limits (Large Combustion Plants BREF)

**Natural Gas Boilers (>300 MWth, new plants):**

| Pollutant | BAT-AEL Range | Unit | Reference |
|-----------|---------------|------|-----------|
| NOx | 30-50 | mg/Nm3 @ 3% O2 | BAT 28 |
| CO | 30-100 | mg/Nm3 @ 3% O2 | BAT 29 |
| NH3 (SCR) | <5 | mg/Nm3 | BAT 6 |

**Hard Coal Boilers (>300 MWth, new plants):**

| Pollutant | BAT-AEL Range | Unit | Reference |
|-----------|---------------|------|-----------|
| NOx | 65-85 | mg/Nm3 @ 6% O2 | BAT 21 |
| SO2 | 10-130 | mg/Nm3 @ 6% O2 | BAT 22 |
| PM | 2-8 | mg/Nm3 @ 6% O2 | BAT 23 |
| Hg | 1-3 | ug/Nm3 | BAT 25 |

**Gas Turbines (>50 MWth, new plants):**

| Pollutant | BAT-AEL Range | Unit | Reference |
|-----------|---------------|------|-----------|
| NOx (NG) | 10-30 | mg/Nm3 @ 15% O2 | BAT 34 |
| NOx (Oil) | 30-50 | mg/Nm3 @ 15% O2 | BAT 34 |
| CO | 5-30 | mg/Nm3 @ 15% O2 | BAT 35 |

### 5.4 GL-010 Implementation

**Unit Conversion (EPA to EU):**
```python
def lb_mmbtu_to_mg_nm3(value_lb_mmbtu, pollutant, fuel_type, o2_ref):
    """
    Convert lb/MMBtu to mg/Nm3 at reference O2.

    Formula:
    mg/Nm3 = lb/MMBtu * Fd * 1e6 * lb_to_kg / Nm3_per_dscf

    Where:
    - Fd = F-factor (dscf/MMBtu)
    - Nm3/dscf = 0.02832
    """
    Fd = get_f_factor(fuel_type)
    conversion = value_lb_mmbtu * Fd * 453.592 / 28.317
    return round(conversion, 1)
```

**O2 Correction (EU Standard):**
```
C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)
```

### 5.5 Monitoring Requirements

| Pollutant | Method | Frequency |
|-----------|--------|-----------|
| NOx | EN 14792 (CEMS) | Continuous |
| SO2 | EN 14791 (CEMS) | Continuous |
| PM | EN 13284-2 (CEMS) | Continuous |
| CO | EN 15058 (CEMS) | Continuous |
| HCl | EN 1911 | Periodic |
| Hg | EN 13211 | Periodic |

### 5.6 Reporting

**E-PRTR Reporting:**
- Threshold-based pollutant releases
- Annual submission
- XML format

**GL-010 E-PRTR Support:**
```yaml
report_type: eu_eprtr
reporting_year: 2025
facility:
  prtr_id: "EU-12345"
  nace_code: "35.11"
releases:
  air:
    - pollutant: "NOx (as NO2)"
      amount_kg: 125000
      method: "M"  # Measurement
    - pollutant: "SOx (as SO2)"
      amount_kg: 50000
      method: "M"
```

---

## 6. EU ETS MRV Regulation

### 6.1 Regulatory Overview

Regulation (EU) 2018/2066 establishes monitoring and reporting rules for the EU Emissions Trading System (EU ETS).

### 6.2 Monitoring Approaches

| Approach | Description | Uncertainty |
|----------|-------------|-------------|
| Calculation | Fuel/material balance | Tier-based |
| Measurement | CEMS | <2.5% |
| Fall-back | Estimated data | Last resort |

### 6.3 Calculation Tiers

**CO2 from Combustion:**
```
CO2 = Activity Data * Emission Factor * Oxidation Factor
```

**Tier Requirements (Annex II):**

| Tier | AD Uncertainty | EF Uncertainty | OF |
|------|----------------|----------------|-----|
| 1 | +/- 7.5% | IPCC default | 1.0 |
| 2a | +/- 5% | Country-specific | 1.0 |
| 2b | +/- 2.5% | Type-specific | 0.99 |
| 3 | +/- 1.5% | Fuel-specific | Analyzed |
| 4 | CEMS | CEMS | N/A |

### 6.4 GL-010 Implementation

**MRV Calculation Engine:**
```python
class EUETSCalculator:
    def calculate_co2(self, fuel_data, tier_level):
        """Calculate CO2 per MRV Regulation Annex II."""
        ad = fuel_data.quantity * (1 + self.uncertainty_ad[tier_level])
        ef = self.get_emission_factor(fuel_data.fuel_type, tier_level)
        of = self.get_oxidation_factor(fuel_data.fuel_type, tier_level)

        return ad * ef * of
```

### 6.5 Annual Emissions Report (AER)

**Required Elements:**

| Section | Content |
|---------|---------|
| Operator identification | Name, permit, EUTL ID |
| Installation data | Capacity, category |
| Monitoring methodology | Tiers, methods |
| Activity data | Fuel/material quantities |
| Emission factors | Values, sources |
| Calculated emissions | CO2 by source stream |
| Uncertainty assessment | Combined uncertainty |

**GL-010 AER Output:**
```xml
<AnnualEmissionsReport>
  <Operator>
    <OperatorID>EU-EUTL-12345</OperatorID>
    <PermitID>GHG-2020-0001</PermitID>
  </Operator>
  <Emissions>
    <SourceStream name="Natural Gas Combustion">
      <ActivityData unit="TJ">5000</ActivityData>
      <EmissionFactor unit="tCO2/TJ">56.1</EmissionFactor>
      <OxidationFactor>1.0</OxidationFactor>
      <CO2Emissions unit="tCO2">280500</CO2Emissions>
    </SourceStream>
  </Emissions>
</AnnualEmissionsReport>
```

### 6.6 Verification Requirements

| Category | Installation Emissions | Verification |
|----------|------------------------|--------------|
| A | <50,000 tCO2e/yr | Simplified |
| B | 50,000-500,000 | Standard |
| C | >500,000 | Enhanced |

---

## 7. ISO 14064

### 7.1 Standard Overview

ISO 14064 provides specifications for GHG quantification, reporting, and verification at organizational and project levels.

| Part | Title | Scope |
|------|-------|-------|
| 14064-1 | Organization level | GHG inventories |
| 14064-2 | Project level | GHG reductions |
| 14064-3 | Verification | Validation/verification |

### 7.2 ISO 14064-1 Requirements

**Organizational Boundaries:**
- Equity share approach
- Financial control approach
- Operational control approach

**Operational Boundaries:**
- Scope 1: Direct emissions
- Scope 2: Indirect (electricity)
- Scope 3: Other indirect

### 7.3 GL-010 Implementation

**Scope Classification:**
```python
class ISO14064Inventory:
    def classify_emissions(self, source):
        """Classify emissions by ISO 14064 scope."""
        if source.is_owned_operated:
            return "Scope 1 - Direct"
        elif source.is_purchased_energy:
            return "Scope 2 - Indirect (Energy)"
        else:
            return "Scope 3 - Other Indirect"
```

**GHG Inventory Categories (Annex B):**

| Category | Examples | GL-010 Support |
|----------|----------|----------------|
| Stationary combustion | Boilers, heaters | Full |
| Mobile combustion | Fleet vehicles | Full |
| Process emissions | Chemical reactions | Partial |
| Fugitive emissions | Leaks, venting | Partial |
| Purchased electricity | Grid power | Full |
| Purchased steam/heat | District heating | Full |

### 7.4 Quantification Methods

**Direct Monitoring:**
```
GHG = Activity_data * Emission_factor * GWP
```

**Mass Balance:**
```
GHG = (Input_carbon - Output_carbon) * (44/12) * GWP
```

**Engineering Calculations:**
```
GHG = f(equipment_specs, operating_conditions, emission_factors)
```

### 7.5 Reporting Requirements

**GHG Inventory Report:**

| Element | Requirement |
|---------|-------------|
| Organizational description | Boundaries, activities |
| Reporting period | Usually annual |
| Direct emissions | By category and gas |
| Indirect emissions | Scope 2 and 3 |
| Quantification methodology | Methods, factors |
| Uncertainty | Analysis and statement |

**GL-010 ISO 14064 Report:**
```yaml
report_type: iso14064_inventory
organization: "Example Corporation"
reporting_period: "2025"
boundaries:
  approach: "operational_control"
emissions:
  scope_1:
    stationary_combustion:
      co2_metric_tons: 50000
      ch4_metric_tons: 2
      n2o_metric_tons: 0.5
    mobile_sources:
      co2_metric_tons: 5000
  scope_2:
    purchased_electricity:
      co2_metric_tons: 25000
      method: "location_based"
  scope_3:
    business_travel:
      co2_metric_tons: 1000
```

---

## 8. ISO 14001

### 8.1 Standard Overview

ISO 14001:2015 specifies requirements for environmental management systems (EMS) that organizations can use to enhance environmental performance.

### 8.2 Key Requirements

| Clause | Requirement | GL-010 Support |
|--------|-------------|----------------|
| 4 | Context of organization | Facility profiles |
| 5 | Leadership | Policy tracking |
| 6 | Planning | Aspects/impacts |
| 7 | Support | Documentation |
| 8 | Operation | Operational control |
| 9 | Performance evaluation | Monitoring |
| 10 | Improvement | Corrective actions |

### 8.3 GL-010 EMS Support

**Environmental Aspects (Clause 6.1.2):**
```yaml
environmental_aspects:
  - aspect: "Air emissions from combustion"
    impact: "Air quality degradation"
    significance: "High"
    controls:
      - "CEMS monitoring"
      - "Emission limits compliance"
      - "Permit conditions"
    monitoring_tool: "GL-010 EMISSIONWATCH"
```

**Monitoring and Measurement (Clause 9.1.1):**
- Continuous emissions monitoring
- Compliance tracking
- Trend analysis
- Performance indicators

**Documented Information (Clause 7.5):**
- Emission records
- Compliance reports
- Audit trails
- Calibration records

### 8.4 Performance Indicators

| KPI | Metric | GL-010 Tool |
|-----|--------|-------------|
| Compliance rate | % within limits | check_compliance_status |
| Emission intensity | kg CO2/MWh | calculate_co2_emissions |
| Violation frequency | Events/year | detect_violations |
| Data availability | % uptime | Data quality metrics |

---

## 9. China MEE Standards

### 9.1 Regulatory Framework

China's Ministry of Ecology and Environment (MEE) establishes emission standards through:
- National emission standards (GB)
- Regional standards (DB)
- Facility permits

### 9.2 Key Standards

| Standard | Title | Pollutants |
|----------|-------|------------|
| GB 13271-2014 | Boiler Air Pollutants | PM, SO2, NOx |
| GB 13223-2011 | Thermal Power Plants | PM, SO2, NOx, Hg |
| GB 31571-2015 | Petroleum Refining | VOCs, PM |
| GB 16297-1996 | Air Pollutants (General) | Multiple |

### 9.3 Emission Limits (GB 13223-2011)

**Coal-Fired Power Plants:**

| Pollutant | Existing | New | Ultra-low | Unit |
|-----------|----------|-----|-----------|------|
| PM | 30 | 20 | 5 | mg/Nm3 |
| SO2 | 100 | 50 | 35 | mg/Nm3 |
| NOx | 100 | 50 | 50 | mg/Nm3 |
| Hg | 0.03 | 0.03 | 0.03 | mg/Nm3 |

**Gas-Fired Units:**

| Pollutant | Limit | Unit |
|-----------|-------|------|
| NOx | 30 | mg/Nm3 @ 15% O2 |
| PM | 5 | mg/Nm3 |
| SO2 | 35 | mg/Nm3 |

### 9.4 GL-010 Implementation

**China MEE Limits Database:**
```python
CHINA_MEE_LIMITS = {
    "coal_power_plant": {
        "new": {
            "pm_mg_nm3": 20,
            "so2_mg_nm3": 50,
            "nox_mg_nm3": 50,
            "hg_mg_nm3": 0.03,
            "o2_reference": 6.0
        },
        "ultra_low": {
            "pm_mg_nm3": 5,
            "so2_mg_nm3": 35,
            "nox_mg_nm3": 50,
            "hg_mg_nm3": 0.03,
            "o2_reference": 6.0
        }
    },
    "gas_turbine": {
        "nox_mg_nm3": 30,
        "o2_reference": 15.0
    }
}
```

### 9.5 Monitoring Requirements

| Parameter | Method | Frequency |
|-----------|--------|-----------|
| PM | GB/T 16157 or CEMS | Continuous |
| SO2 | HJ 57 or CEMS | Continuous |
| NOx | HJ 692 or CEMS | Continuous |
| Opacity | HJ/T 76 | Continuous |

### 9.6 Reporting

**National Pollutant Release Reporting:**
- Monthly emissions data
- Quarterly compliance reports
- Annual emissions inventory

**GL-010 China MEE Report:**
```yaml
report_type: china_mee_monthly
facility_id: "CHN-12345"
reporting_period: "2025-01"
emissions:
  nox_tonnes: 125
  so2_tonnes: 50
  pm_tonnes: 10
compliance_status: "compliant"
cems_availability: 98.5
```

---

## Appendix A: Regulatory Cross-Reference

| Parameter | EPA (40 CFR 60) | EU IED | China GB |
|-----------|-----------------|--------|----------|
| O2 Ref (Boiler) | 3% | 3% / 6% | 6% |
| O2 Ref (Turbine) | 15% | 15% | 15% |
| NOx Unit | lb/MMBtu | mg/Nm3 | mg/Nm3 |
| SO2 Unit | lb/MMBtu | mg/Nm3 | mg/Nm3 |
| PM Unit | lb/MMBtu | mg/Nm3 | mg/Nm3 |
| Averaging | 30-day | Annual | Hourly |

---

## Appendix B: Unit Conversion Reference

| Conversion | Formula |
|------------|---------|
| lb/MMBtu to mg/Nm3 (gas @ 3% O2) | value * 410 |
| lb/MMBtu to mg/Nm3 (coal @ 6% O2) | value * 462 |
| ppm to mg/Nm3 (NOx) | ppm * 2.05 |
| ppm to mg/Nm3 (SO2) | ppm * 2.86 |
| short ton to metric ton | value * 0.9072 |
| MMBtu to GJ | value * 1.055 |

---

## Appendix C: Compliance Calendar

| Regulation | Report | Deadline | GL-010 Support |
|------------|--------|----------|----------------|
| EPA Part 75 | Quarterly CEMS | 30 days after quarter | Full |
| EPA Part 98 | Annual GHG | March 31 | Full |
| EU ETS | Annual Emissions | March 31 | Full |
| EU E-PRTR | Annual Release | Per Member State | Full |
| China MEE | Monthly | 15th of following month | Full |

---

*Document generated by GL-SpecGuardian v1.0*
*Regulatory Standards Reference: November 2025*
