# GL-010 EMISSIONWATCH Tool Specifications

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Spec Compliance:** GreenLang Pack Spec v1.0

---

## Overview

This document provides comprehensive specifications for all 12 tools implemented in GL-010 EMISSIONWATCH. Each tool is designed for deterministic, zero-hallucination emissions calculations with full audit trail support.

**Design Principles:**
- All calculations are deterministic (same inputs always produce same outputs)
- Physics-based calculations with documented methodologies
- Full regulatory provenance tracking
- SHA-256 hashed audit trails
- Temperature 0.0, Seed 42 for all operations

---

## Table of Contents

1. [calculate_nox_emissions](#1-calculate_nox_emissions)
2. [calculate_sox_emissions](#2-calculate_sox_emissions)
3. [calculate_co2_emissions](#3-calculate_co2_emissions)
4. [calculate_particulate_matter](#4-calculate_particulate_matter)
5. [check_compliance_status](#5-check_compliance_status)
6. [generate_regulatory_report](#6-generate_regulatory_report)
7. [detect_violations](#7-detect_violations)
8. [predict_exceedances](#8-predict_exceedances)
9. [calculate_emission_factors](#9-calculate_emission_factors)
10. [analyze_fuel_composition](#10-analyze_fuel_composition)
11. [calculate_dispersion](#11-calculate_dispersion)
12. [generate_audit_trail](#12-generate_audit_trail)

---

## 1. calculate_nox_emissions

### 1.1 Description

Calculates nitrogen oxide (NOx) emissions from combustion sources using physics-based thermal and fuel NOx formation mechanisms. Implements the Zeldovich mechanism for thermal NOx and empirical correlations for fuel-bound nitrogen conversion.

### 1.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NOxCalculationInput",
  "type": "object",
  "required": ["fuel_type", "heat_input_mmbtu_hr"],
  "properties": {
    "fuel_type": {
      "type": "string",
      "enum": [
        "natural_gas",
        "fuel_oil_no2",
        "fuel_oil_no6",
        "coal_bituminous",
        "coal_subbituminous",
        "coal_lignite",
        "coal_anthracite",
        "diesel",
        "propane",
        "butane",
        "wood",
        "biomass"
      ],
      "description": "Type of fuel being combusted"
    },
    "heat_input_mmbtu_hr": {
      "type": "number",
      "minimum": 0,
      "maximum": 50000,
      "description": "Heat input rate in MMBtu/hr"
    },
    "excess_air_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 500,
      "default": 15,
      "description": "Excess air percentage"
    },
    "combustion_temperature_k": {
      "type": "number",
      "minimum": 300,
      "maximum": 3000,
      "description": "Peak flame temperature in Kelvin"
    },
    "fuel_nitrogen_weight_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 5,
      "default": 0,
      "description": "Fuel-bound nitrogen content (weight %)"
    },
    "residence_time_seconds": {
      "type": "number",
      "minimum": 0,
      "maximum": 10,
      "default": 1.0,
      "description": "Combustion zone residence time"
    },
    "o2_measured_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 21,
      "description": "Measured stack O2 percentage"
    },
    "control_device": {
      "type": "string",
      "enum": ["none", "low_nox_burner", "flue_gas_recirculation", "scr", "sncr"],
      "default": "none",
      "description": "NOx control technology"
    }
  }
}
```

### 1.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NOxCalculationOutput",
  "type": "object",
  "required": [
    "thermal_nox_lb_mmbtu",
    "fuel_nox_lb_mmbtu",
    "total_nox_lb_mmbtu",
    "calculation_method",
    "provenance_hash"
  ],
  "properties": {
    "thermal_nox_lb_mmbtu": {
      "type": "number",
      "description": "Thermal NOx emission factor (lb/MMBtu)"
    },
    "fuel_nox_lb_mmbtu": {
      "type": "number",
      "description": "Fuel NOx emission factor (lb/MMBtu)"
    },
    "total_nox_lb_mmbtu": {
      "type": "number",
      "description": "Total NOx emission factor (lb/MMBtu)"
    },
    "nox_mass_lb_hr": {
      "type": "number",
      "description": "NOx mass emission rate (lb/hr)"
    },
    "nox_ppm_at_ref_o2": {
      "type": "number",
      "description": "NOx concentration at reference O2"
    },
    "nox_kg_gj": {
      "type": "number",
      "description": "NOx emission factor (kg/GJ)"
    },
    "nox_mg_nm3": {
      "type": "number",
      "description": "NOx concentration (mg/Nm3)"
    },
    "calculation_method": {
      "type": "string",
      "description": "Calculation methodology used"
    },
    "uncertainty_percent": {
      "type": "number",
      "description": "Calculation uncertainty (+/- %)"
    },
    "calculation_steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "step_number": {"type": "integer"},
          "description": {"type": "string"},
          "formula": {"type": "string"},
          "inputs": {"type": "object"},
          "output_value": {"type": "number"},
          "output_unit": {"type": "string"}
        }
      }
    },
    "regulatory_reference": {
      "type": "string",
      "description": "Applicable regulatory citation"
    },
    "provenance_hash": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$",
      "description": "SHA-256 hash of calculation inputs"
    }
  }
}
```

### 1.4 Physics Basis

**Thermal NOx (Zeldovich Mechanism):**
```
N2 + O  <-> NO + N     (k1 = 1.8e14 * exp(-38370/T))
N + O2  <-> NO + O     (k2 = 9.0e9 * exp(-3160/T))
N + OH  <-> NO + H     (k3 = 2.8e13 * exp(-22080/T))

Rate-limiting step: N2 + O -> NO + N
d[NO]/dt = k1[N2][O]exp(-Ea/RT)
```

**Fuel NOx (Conversion Model):**
```
Fuel-N conversion efficiency:
eta = f(O2, temperature, stoichiometry)

NOx_fuel = N_fuel * eta * (30/14) * heat_input

Where:
- N_fuel = fuel nitrogen content (lb/MMBtu)
- eta = conversion efficiency (0.15-0.40 typical)
- 30/14 = MW ratio (NO/N)
```

### 1.5 Regulatory Reference

| Regulation | Requirement | How Implemented |
|------------|-------------|-----------------|
| EPA 40 CFR 60.44 | NSPS boiler NOx limits | Limit comparison |
| EPA 40 CFR 60.4320 | NSPS turbine NOx limits | Limit comparison |
| EPA Method 7E | NOx measurement | Reference method |
| EPA Method 19 | F-factor calculation | Emission factor |
| EU IED Annex V | BAT-AEL limits | EU compliance |

### 1.6 Accuracy Requirements

| Parameter | Accuracy | Basis |
|-----------|----------|-------|
| Emission Factor | +/- 10% | EPA AP-42 |
| Mass Rate | +/- 5% | Method 19 |
| Concentration | +/- 2.5% | Method 7E |

### 1.7 Determinism Guarantee

- Temperature: 0.0
- Seed: 42
- Rounding: ROUND_HALF_UP
- Precision: 4 decimal places
- Hash Algorithm: SHA-256

### 1.8 Example Input/Output

**Input:**
```json
{
  "fuel_type": "natural_gas",
  "heat_input_mmbtu_hr": 100,
  "excess_air_percent": 15,
  "combustion_temperature_k": 1800,
  "control_device": "low_nox_burner"
}
```

**Output:**
```json
{
  "thermal_nox_lb_mmbtu": 0.0523,
  "fuel_nox_lb_mmbtu": 0.0000,
  "total_nox_lb_mmbtu": 0.0523,
  "nox_mass_lb_hr": 5.23,
  "nox_ppm_at_ref_o2": 45.2,
  "calculation_method": "Zeldovich + EPA AP-42",
  "uncertainty_percent": 10.0,
  "regulatory_reference": "EPA AP-42 Section 1.4",
  "provenance_hash": "a1b2c3d4e5f6..."
}
```

---

## 2. calculate_sox_emissions

### 2.1 Description

Calculates sulfur dioxide (SO2) and sulfur trioxide (SO3) emissions from fuel sulfur content using stoichiometric mass balance. All fuel sulfur is assumed to oxidize to SO2, with a small fraction further oxidizing to SO3.

### 2.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SOxCalculationInput",
  "type": "object",
  "required": ["fuel_type", "heat_input_mmbtu_hr"],
  "properties": {
    "fuel_type": {
      "type": "string",
      "enum": [
        "natural_gas",
        "fuel_oil_no2",
        "fuel_oil_no6",
        "coal_bituminous",
        "coal_subbituminous",
        "diesel",
        "petroleum_coke"
      ]
    },
    "heat_input_mmbtu_hr": {
      "type": "number",
      "minimum": 0,
      "maximum": 50000
    },
    "sulfur_weight_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 10,
      "description": "Fuel sulfur content (weight %)"
    },
    "higher_heating_value_btu_lb": {
      "type": "number",
      "minimum": 5000,
      "maximum": 25000,
      "description": "Fuel higher heating value"
    },
    "control_device": {
      "type": "string",
      "enum": ["none", "dry_fgd", "wet_fgd", "dsi", "seawater_scrubber"],
      "default": "none"
    },
    "control_efficiency_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 99.9,
      "default": 0
    }
  }
}
```

### 2.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SOxCalculationOutput",
  "type": "object",
  "required": [
    "so2_lb_mmbtu",
    "so3_lb_mmbtu",
    "total_sox_lb_mmbtu",
    "calculation_method",
    "provenance_hash"
  ],
  "properties": {
    "so2_lb_mmbtu": {
      "type": "number",
      "description": "SO2 emission factor (lb/MMBtu)"
    },
    "so3_lb_mmbtu": {
      "type": "number",
      "description": "SO3 emission factor (lb/MMBtu)"
    },
    "total_sox_lb_mmbtu": {
      "type": "number",
      "description": "Total SOx as SO2 (lb/MMBtu)"
    },
    "sox_mass_lb_hr": {
      "type": "number",
      "description": "SOx mass emission rate (lb/hr)"
    },
    "sox_kg_gj": {
      "type": "number",
      "description": "SOx emission factor (kg/GJ)"
    },
    "sox_mg_nm3": {
      "type": "number",
      "description": "SOx concentration (mg/Nm3)"
    },
    "uncontrolled_sox_lb_mmbtu": {
      "type": "number",
      "description": "Pre-control SOx emission factor"
    },
    "control_efficiency_actual": {
      "type": "number",
      "description": "Applied control efficiency (%)"
    },
    "calculation_steps": {
      "type": "array",
      "items": {"type": "object"}
    },
    "provenance_hash": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$"
    }
  }
}
```

### 2.4 Physics Basis

**Stoichiometric Sulfur Oxidation:**
```
S + O2 -> SO2

Mass Balance:
SO2 (lb) = S (lb) * (64.066 / 32.06)
SO2 (lb) = S (lb) * 1.998

Emission Factor:
EF_SO2 = (S% / 100) * (1/HHV) * 2.0 * 10^6 lb SO2/MMBtu

Where:
- S% = sulfur weight percent
- HHV = higher heating value (Btu/lb)
- 2.0 = MW ratio (SO2/S)
```

**SO3 Formation:**
```
2 SO2 + O2 -> 2 SO3

Typical SO3 fraction: 1-5% of total SOx
Depends on:
- Catalyst presence (V2O5 in SCR)
- Temperature profile
- Excess oxygen
```

### 2.5 Regulatory Reference

| Regulation | Requirement |
|------------|-------------|
| EPA 40 CFR 60.42b | NSPS SO2 limits |
| EPA 40 CFR 60.4330 | Turbine SO2 limits |
| EPA Method 6C | SO2 measurement |
| EU IED Annex V | BAT-AEL SO2 limits |

### 2.6 Accuracy Requirements

| Parameter | Accuracy | Basis |
|-----------|----------|-------|
| Emission Factor | +/- 5% | Stoichiometry |
| Sulfur Analysis | +/- 0.01% | ASTM D4294 |
| Control Efficiency | +/- 2% | Stack test |

### 2.7 Example Input/Output

**Input:**
```json
{
  "fuel_type": "coal_bituminous",
  "heat_input_mmbtu_hr": 500,
  "sulfur_weight_percent": 2.5,
  "higher_heating_value_btu_lb": 12500,
  "control_device": "wet_fgd",
  "control_efficiency_percent": 95
}
```

**Output:**
```json
{
  "so2_lb_mmbtu": 0.20,
  "so3_lb_mmbtu": 0.004,
  "total_sox_lb_mmbtu": 0.204,
  "sox_mass_lb_hr": 102.0,
  "uncontrolled_sox_lb_mmbtu": 4.0,
  "control_efficiency_actual": 95.0,
  "calculation_method": "Stoichiometric mass balance",
  "provenance_hash": "b2c3d4e5f6a7..."
}
```

---

## 3. calculate_co2_emissions

### 3.1 Description

Calculates carbon dioxide (CO2) emissions from fuel combustion using carbon balance methodology. Compliant with GHG Protocol, EPA 40 CFR Part 98, and ISO 14064 standards.

### 3.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CO2CalculationInput",
  "type": "object",
  "required": ["fuel_type", "fuel_quantity"],
  "properties": {
    "fuel_type": {
      "type": "string",
      "enum": [
        "natural_gas",
        "fuel_oil_no2",
        "fuel_oil_no6",
        "coal_bituminous",
        "coal_subbituminous",
        "diesel",
        "gasoline",
        "propane",
        "jet_fuel",
        "biodiesel",
        "ethanol"
      ]
    },
    "fuel_quantity": {
      "type": "number",
      "minimum": 0,
      "description": "Fuel quantity"
    },
    "fuel_quantity_unit": {
      "type": "string",
      "enum": ["mmbtu", "mscf", "gallons", "short_tons", "metric_tons", "kg", "liters"],
      "default": "mmbtu"
    },
    "carbon_content_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 100,
      "description": "Carbon content (weight %)"
    },
    "oxidation_factor": {
      "type": "number",
      "minimum": 0.9,
      "maximum": 1.0,
      "default": 0.99,
      "description": "Carbon oxidation factor"
    },
    "reporting_standard": {
      "type": "string",
      "enum": ["epa_part98", "ghg_protocol", "iso14064", "eu_ets"],
      "default": "epa_part98"
    }
  }
}
```

### 3.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CO2CalculationOutput",
  "type": "object",
  "required": [
    "co2_metric_tons",
    "co2_kg_gj",
    "calculation_method",
    "provenance_hash"
  ],
  "properties": {
    "co2_metric_tons": {
      "type": "number",
      "description": "CO2 emissions (metric tons)"
    },
    "co2_short_tons": {
      "type": "number",
      "description": "CO2 emissions (short tons)"
    },
    "co2_kg_gj": {
      "type": "number",
      "description": "CO2 emission factor (kg/GJ)"
    },
    "co2_lb_mmbtu": {
      "type": "number",
      "description": "CO2 emission factor (lb/MMBtu)"
    },
    "co2e_metric_tons": {
      "type": "number",
      "description": "CO2-equivalent including CH4, N2O"
    },
    "ch4_emissions_kg": {
      "type": "number",
      "description": "Methane emissions (kg)"
    },
    "n2o_emissions_kg": {
      "type": "number",
      "description": "N2O emissions (kg)"
    },
    "biogenic_co2_metric_tons": {
      "type": "number",
      "description": "Biogenic CO2 (if applicable)"
    },
    "scope": {
      "type": "string",
      "enum": ["scope_1", "scope_2", "scope_3"],
      "description": "GHG Protocol scope"
    },
    "reporting_standard": {
      "type": "string"
    },
    "emission_factor_source": {
      "type": "string"
    },
    "calculation_steps": {
      "type": "array"
    },
    "provenance_hash": {
      "type": "string"
    }
  }
}
```

### 3.4 Physics Basis

**Carbon Balance Method:**
```
C + O2 -> CO2

CO2 = Fuel * C% * 44.009/12.011 * Oxidation_Factor

Where:
- 44.009 = Molecular weight of CO2
- 12.011 = Atomic weight of C
- Ratio = 3.664
```

**EPA 40 CFR Part 98 Method:**
```
CO2 (metric tons) = Fuel (MMBtu) * EF (kg CO2/MMBtu) * 0.001

Default Emission Factors (kg CO2/MMBtu):
- Natural Gas: 53.06
- Coal (bit): 93.28
- Fuel Oil #2: 73.96
- Fuel Oil #6: 75.10
```

**GHG Protocol CO2e:**
```
CO2e = CO2 + (CH4 * GWP_CH4) + (N2O * GWP_N2O)

Where (IPCC AR6):
- GWP_CH4 = 28
- GWP_N2O = 273
```

### 3.5 Regulatory Reference

| Regulation | Coverage |
|------------|----------|
| EPA 40 CFR Part 98 | US GHG Reporting |
| EU MRV 2018/2066 | EU ETS Monitoring |
| ISO 14064-1:2018 | GHG Quantification |
| GHG Protocol | Corporate Standard |

### 3.6 Accuracy Requirements

| Parameter | Accuracy | Basis |
|-----------|----------|-------|
| Emission Factor | +/- 5% | Part 98 Table C-1 |
| Carbon Content | +/- 2% | ASTM analysis |
| Total CO2 | +/- 3% | Combined |

### 3.7 Example Input/Output

**Input:**
```json
{
  "fuel_type": "natural_gas",
  "fuel_quantity": 10000,
  "fuel_quantity_unit": "mmbtu",
  "reporting_standard": "ghg_protocol"
}
```

**Output:**
```json
{
  "co2_metric_tons": 530.6,
  "co2_kg_gj": 56.1,
  "co2_lb_mmbtu": 117.0,
  "co2e_metric_tons": 531.8,
  "ch4_emissions_kg": 10.0,
  "n2o_emissions_kg": 1.0,
  "scope": "scope_1",
  "reporting_standard": "ghg_protocol",
  "emission_factor_source": "EPA 40 CFR Part 98 Table C-1",
  "provenance_hash": "c3d4e5f6a7b8..."
}
```

---

## 4. calculate_particulate_matter

### 4.1 Description

Calculates particulate matter emissions (Total PM, PM10, PM2.5) from combustion sources using EPA AP-42 emission factors and control device efficiency curves.

### 4.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PMCalculationInput",
  "type": "object",
  "required": ["fuel_type", "heat_input_mmbtu_hr"],
  "properties": {
    "fuel_type": {
      "type": "string"
    },
    "heat_input_mmbtu_hr": {
      "type": "number",
      "minimum": 0
    },
    "ash_weight_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 50
    },
    "source_type": {
      "type": "string",
      "enum": ["boiler", "furnace", "kiln", "incinerator", "engine", "turbine"]
    },
    "control_device": {
      "type": "string",
      "enum": [
        "none",
        "cyclone",
        "multicyclone",
        "baghouse",
        "electrostatic_precipitator",
        "wet_scrubber",
        "venturi_scrubber"
      ],
      "default": "none"
    },
    "control_efficiency_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 99.99
    }
  }
}
```

### 4.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PMCalculationOutput",
  "type": "object",
  "required": [
    "total_pm_lb_mmbtu",
    "pm10_lb_mmbtu",
    "pm2_5_lb_mmbtu",
    "provenance_hash"
  ],
  "properties": {
    "total_pm_lb_mmbtu": {
      "type": "number"
    },
    "pm10_lb_mmbtu": {
      "type": "number"
    },
    "pm2_5_lb_mmbtu": {
      "type": "number"
    },
    "filterable_pm_lb_mmbtu": {
      "type": "number"
    },
    "condensable_pm_lb_mmbtu": {
      "type": "number"
    },
    "pm_mass_lb_hr": {
      "type": "number"
    },
    "uncontrolled_pm_lb_mmbtu": {
      "type": "number"
    },
    "control_efficiency_applied": {
      "type": "object",
      "properties": {
        "total_pm": {"type": "number"},
        "pm10": {"type": "number"},
        "pm2_5": {"type": "number"}
      }
    },
    "calculation_steps": {"type": "array"},
    "provenance_hash": {"type": "string"}
  }
}
```

### 4.4 Physics Basis

**Emission Factor Method (EPA AP-42):**
```
PM = EF * Heat_Input

For coal:
EF = EF_base * (Ash% / 10%)

PM10 = Filterable_PM * PM10_fraction
PM2.5 = Filterable_PM * PM2.5_fraction
Total_PM = Filterable_PM + Condensable_PM
```

**Control Device Efficiency:**
```
PM_controlled = PM_uncontrolled * (1 - efficiency)

Typical Efficiencies:
- Baghouse: 99% (total), 99% (PM10), 99% (PM2.5)
- ESP: 99.5% (total), 99% (PM10), 97% (PM2.5)
- Cyclone: 70% (total), 30% (PM10), 10% (PM2.5)
```

### 4.5 Regulatory Reference

| Regulation | Requirement |
|------------|-------------|
| EPA 40 CFR 60.42 | NSPS PM limits |
| EPA Method 5 | Filterable PM |
| EPA Method 202 | Condensable PM |
| EPA Method 201A | PM10/PM2.5 |

### 4.6 Example Input/Output

**Input:**
```json
{
  "fuel_type": "coal_bituminous",
  "heat_input_mmbtu_hr": 500,
  "ash_weight_percent": 10,
  "control_device": "baghouse"
}
```

**Output:**
```json
{
  "total_pm_lb_mmbtu": 0.006,
  "pm10_lb_mmbtu": 0.00075,
  "pm2_5_lb_mmbtu": 0.00025,
  "filterable_pm_lb_mmbtu": 0.005,
  "condensable_pm_lb_mmbtu": 0.001,
  "pm_mass_lb_hr": 3.0,
  "uncontrolled_pm_lb_mmbtu": 0.60,
  "control_efficiency_applied": {
    "total_pm": 99.0,
    "pm10": 99.0,
    "pm2_5": 99.0
  },
  "provenance_hash": "d4e5f6a7b8c9..."
}
```

---

## 5. check_compliance_status

### 5.1 Description

Performs multi-jurisdiction regulatory compliance checking against EPA NSPS, EU IED BAT-AELs, and state/permit-specific limits. Returns detailed compliance status with margin calculations.

### 5.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ComplianceCheckInput",
  "type": "object",
  "required": ["pollutant", "measured_value", "source_category", "jurisdiction"],
  "properties": {
    "pollutant": {
      "type": "string",
      "enum": ["NOx", "SOx", "SO2", "CO2", "CO", "PM", "PM10", "PM2.5", "Opacity"]
    },
    "measured_value": {
      "type": "number"
    },
    "unit": {
      "type": "string",
      "enum": ["lb/MMBtu", "ppm", "mg/Nm3", "kg/MWh", "%"]
    },
    "source_category": {
      "type": "string",
      "enum": ["boiler", "gas_turbine", "reciprocating_engine", "incinerator"]
    },
    "jurisdiction": {
      "type": "string",
      "enum": ["epa_federal", "eu", "california", "texas", "permit_specific"]
    },
    "averaging_period": {
      "type": "string",
      "enum": ["1_hour", "3_hour", "24_hour", "30_day", "annual"]
    },
    "o2_percent": {
      "type": "number",
      "description": "O2 at measurement for correction"
    },
    "fuel_type": {
      "type": "string"
    },
    "capacity_mmbtu_hr": {
      "type": "number"
    }
  }
}
```

### 5.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ComplianceCheckOutput",
  "type": "object",
  "required": [
    "status",
    "measured_value_corrected",
    "applicable_limit",
    "margin_percent"
  ],
  "properties": {
    "status": {
      "type": "string",
      "enum": ["compliant", "non_compliant", "approaching_limit", "insufficient_data"]
    },
    "measured_value_corrected": {
      "type": "number"
    },
    "applicable_limit": {
      "type": "object",
      "properties": {
        "value": {"type": "number"},
        "unit": {"type": "string"},
        "averaging_period": {"type": "string"},
        "o2_reference": {"type": "number"},
        "citation": {"type": "string"}
      }
    },
    "margin_percent": {
      "type": "number",
      "description": "Negative = exceedance"
    },
    "margin_absolute": {
      "type": "number"
    },
    "o2_correction_applied": {
      "type": "boolean"
    },
    "regulatory_program": {
      "type": "string",
      "enum": ["nsps", "mact", "ied", "bat_ael", "title_v", "permit"]
    },
    "compliance_notes": {
      "type": "string"
    },
    "provenance_hash": {
      "type": "string"
    }
  }
}
```

### 5.4 Compliance Logic

**O2 Correction Formula:**
```
C_corrected = C_measured * (20.9 - O2_ref) / (20.9 - O2_measured)

Where:
- O2_ref = 3% for boilers, 15% for turbines
- 20.9% = O2 in ambient air
```

**Compliance Determination:**
```
if measured > limit:
    status = "non_compliant"
elif measured > 0.9 * limit:
    status = "approaching_limit"
else:
    status = "compliant"

margin_percent = ((limit - measured) / limit) * 100
```

### 5.5 Supported Limits Database

| Jurisdiction | Program | Pollutants | Source Types |
|--------------|---------|------------|--------------|
| EPA Federal | NSPS | NOx, SO2, PM | Boilers, Turbines |
| EPA Federal | MACT | HAPs | Various |
| EU | IED BAT-AEL | NOx, SO2, PM, CO | LCP |
| California | SCAQMD | NOx, CO | Boilers |
| Texas | TCEQ | NOx | Houston area |

### 5.6 Example Input/Output

**Input:**
```json
{
  "pollutant": "NOx",
  "measured_value": 0.08,
  "unit": "lb/MMBtu",
  "source_category": "boiler",
  "jurisdiction": "epa_federal",
  "averaging_period": "30_day",
  "o2_percent": 4.5,
  "fuel_type": "natural_gas"
}
```

**Output:**
```json
{
  "status": "compliant",
  "measured_value_corrected": 0.0873,
  "applicable_limit": {
    "value": 0.10,
    "unit": "lb/MMBtu",
    "averaging_period": "30_day",
    "o2_reference": 3.0,
    "citation": "40 CFR 60.44b(a)(1)"
  },
  "margin_percent": 12.7,
  "margin_absolute": 0.0127,
  "o2_correction_applied": true,
  "regulatory_program": "nsps",
  "provenance_hash": "e5f6a7b8c9d0..."
}
```

---

## 6. generate_regulatory_report

### 6.1 Description

Generates regulatory reports in EPA, EU, and other jurisdiction-specific formats. Supports CEMS quarterly reports, annual emissions inventories, and E-PRTR submissions.

### 6.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ReportGenerationInput",
  "type": "object",
  "required": ["report_type", "facility_id", "reporting_period"],
  "properties": {
    "report_type": {
      "type": "string",
      "enum": [
        "epa_cems_quarterly",
        "epa_annual_inventory",
        "epa_part98_ghg",
        "eu_eprtr",
        "eu_ets_verification",
        "state_annual",
        "custom"
      ]
    },
    "facility_id": {
      "type": "string"
    },
    "reporting_period": {
      "type": "object",
      "properties": {
        "start_date": {"type": "string", "format": "date"},
        "end_date": {"type": "string", "format": "date"}
      }
    },
    "emission_data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "pollutant": {"type": "string"},
          "value": {"type": "number"},
          "unit": {"type": "string"},
          "source_id": {"type": "string"}
        }
      }
    },
    "output_format": {
      "type": "string",
      "enum": ["xml", "json", "csv", "pdf"],
      "default": "xml"
    }
  }
}
```

### 6.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ReportGenerationOutput",
  "type": "object",
  "required": ["report_content", "report_format", "validation_status"],
  "properties": {
    "report_content": {
      "type": "string",
      "description": "Base64 encoded report content"
    },
    "report_format": {
      "type": "string"
    },
    "validation_status": {
      "type": "string",
      "enum": ["valid", "warnings", "errors"]
    },
    "validation_messages": {
      "type": "array",
      "items": {"type": "string"}
    },
    "report_summary": {
      "type": "object",
      "properties": {
        "total_emissions_by_pollutant": {"type": "object"},
        "compliance_status": {"type": "string"},
        "data_completeness_percent": {"type": "number"}
      }
    },
    "submission_ready": {
      "type": "boolean"
    },
    "provenance_hash": {
      "type": "string"
    }
  }
}
```

### 6.4 Supported Report Formats

| Report Type | Format | Schema | Deadline |
|-------------|--------|--------|----------|
| EPA CEMS Quarterly | EDR XML | EPA Part 75 | 30 days |
| EPA Annual Inventory | EIS XML | NEI | March 31 |
| EPA Part 98 GHG | e-GGRT XML | 40 CFR 98 | March 31 |
| EU E-PRTR | XML | Regulation 166/2006 | Per MS |
| EU ETS | AER XML | MRV Regulation | March 31 |

---

## 7. detect_violations

### 7.1 Description

Real-time violation detection engine that monitors emissions against regulatory limits and generates alerts based on configurable thresholds.

### 7.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ViolationDetectionInput",
  "type": "object",
  "required": ["emission_data", "applicable_limits"],
  "properties": {
    "emission_data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "string", "format": "date-time"},
          "pollutant": {"type": "string"},
          "value": {"type": "number"},
          "unit": {"type": "string"},
          "source_id": {"type": "string"}
        }
      }
    },
    "applicable_limits": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "pollutant": {"type": "string"},
          "limit_value": {"type": "number"},
          "unit": {"type": "string"},
          "averaging_period": {"type": "string"}
        }
      }
    },
    "alert_thresholds": {
      "type": "object",
      "properties": {
        "warning_percent": {"type": "number", "default": 80},
        "critical_percent": {"type": "number", "default": 95}
      }
    }
  }
}
```

### 7.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ViolationDetectionOutput",
  "type": "object",
  "required": ["violations", "warnings", "overall_status"],
  "properties": {
    "violations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "violation_id": {"type": "string"},
          "timestamp": {"type": "string"},
          "pollutant": {"type": "string"},
          "measured_value": {"type": "number"},
          "limit_value": {"type": "number"},
          "exceedance_percent": {"type": "number"},
          "severity": {"type": "string", "enum": ["warning", "critical", "violation"]},
          "regulatory_citation": {"type": "string"},
          "recommended_action": {"type": "string"}
        }
      }
    },
    "warnings": {
      "type": "array"
    },
    "overall_status": {
      "type": "string",
      "enum": ["normal", "elevated", "critical", "violation"]
    },
    "statistics": {
      "type": "object",
      "properties": {
        "total_violations": {"type": "integer"},
        "total_warnings": {"type": "integer"},
        "affected_pollutants": {"type": "array"},
        "affected_sources": {"type": "array"}
      }
    },
    "provenance_hash": {"type": "string"}
  }
}
```

---

## 8. predict_exceedances

### 8.1 Description

Predictive analytics engine that forecasts potential limit exceedances using time-series analysis and trend detection. Enables proactive compliance management.

### 8.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExceedancePredictionInput",
  "type": "object",
  "required": ["historical_data", "prediction_horizon"],
  "properties": {
    "historical_data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "string"},
          "pollutant": {"type": "string"},
          "value": {"type": "number"}
        }
      }
    },
    "prediction_horizon": {
      "type": "string",
      "enum": ["1_hour", "24_hours", "7_days", "30_days"]
    },
    "applicable_limit": {
      "type": "number"
    },
    "confidence_level": {
      "type": "number",
      "minimum": 0.5,
      "maximum": 0.99,
      "default": 0.95
    },
    "model_type": {
      "type": "string",
      "enum": ["linear_regression", "arima", "exponential_smoothing"],
      "default": "exponential_smoothing"
    }
  }
}
```

### 8.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExceedancePredictionOutput",
  "type": "object",
  "required": ["prediction", "exceedance_probability", "confidence_interval"],
  "properties": {
    "prediction": {
      "type": "object",
      "properties": {
        "predicted_value": {"type": "number"},
        "prediction_timestamp": {"type": "string"},
        "trend_direction": {"type": "string", "enum": ["increasing", "stable", "decreasing"]}
      }
    },
    "exceedance_probability": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "confidence_interval": {
      "type": "object",
      "properties": {
        "lower_bound": {"type": "number"},
        "upper_bound": {"type": "number"},
        "confidence_level": {"type": "number"}
      }
    },
    "time_to_exceedance": {
      "type": "string",
      "description": "Estimated time until limit exceedance (if applicable)"
    },
    "recommended_actions": {
      "type": "array",
      "items": {"type": "string"}
    },
    "model_metrics": {
      "type": "object",
      "properties": {
        "r_squared": {"type": "number"},
        "mae": {"type": "number"},
        "mape": {"type": "number"}
      }
    },
    "provenance_hash": {"type": "string"}
  }
}
```

---

## 9. calculate_emission_factors

### 9.1 Description

Retrieves and calculates emission factors from authoritative sources including EPA AP-42, WebFIRE database, and EU EMEP/EEA Guidebook.

### 9.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EmissionFactorInput",
  "type": "object",
  "required": ["pollutant", "source_category"],
  "properties": {
    "pollutant": {
      "type": "string",
      "enum": ["NOx", "SO2", "CO2", "CO", "PM", "PM10", "PM2.5", "CH4", "N2O", "VOC"]
    },
    "source_category": {
      "type": "string",
      "description": "SCC or SNAP code"
    },
    "fuel_type": {
      "type": "string"
    },
    "source_type": {
      "type": "string",
      "enum": ["boiler", "turbine", "engine", "heater", "kiln"]
    },
    "control_technology": {
      "type": "string"
    },
    "factor_database": {
      "type": "string",
      "enum": ["ap42", "webfire", "emep_eea", "ipcc"],
      "default": "ap42"
    },
    "output_unit": {
      "type": "string",
      "enum": ["lb/MMBtu", "kg/GJ", "g/hp-hr", "lb/1000gal"],
      "default": "lb/MMBtu"
    }
  }
}
```

### 9.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EmissionFactorOutput",
  "type": "object",
  "required": ["emission_factor", "unit", "source_reference"],
  "properties": {
    "emission_factor": {
      "type": "number"
    },
    "unit": {
      "type": "string"
    },
    "source_reference": {
      "type": "object",
      "properties": {
        "database": {"type": "string"},
        "section": {"type": "string"},
        "table": {"type": "string"},
        "date_published": {"type": "string"},
        "scc_code": {"type": "string"}
      }
    },
    "quality_rating": {
      "type": "string",
      "enum": ["A", "B", "C", "D", "E"]
    },
    "uncertainty_range": {
      "type": "object",
      "properties": {
        "lower": {"type": "number"},
        "upper": {"type": "number"},
        "distribution": {"type": "string"}
      }
    },
    "alternative_factors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "factor": {"type": "number"},
          "source": {"type": "string"},
          "applicability": {"type": "string"}
        }
      }
    },
    "provenance_hash": {"type": "string"}
  }
}
```

---

## 10. analyze_fuel_composition

### 10.1 Description

Analyzes fuel composition using ultimate and proximate analysis data. Calculates heating values, stoichiometric air requirements, and emission potential.

### 10.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "FuelAnalysisInput",
  "type": "object",
  "required": ["analysis_type"],
  "properties": {
    "analysis_type": {
      "type": "string",
      "enum": ["ultimate", "proximate", "both"]
    },
    "ultimate_analysis": {
      "type": "object",
      "properties": {
        "carbon_percent": {"type": "number"},
        "hydrogen_percent": {"type": "number"},
        "oxygen_percent": {"type": "number"},
        "nitrogen_percent": {"type": "number"},
        "sulfur_percent": {"type": "number"},
        "ash_percent": {"type": "number"},
        "moisture_percent": {"type": "number"}
      }
    },
    "proximate_analysis": {
      "type": "object",
      "properties": {
        "volatile_matter_percent": {"type": "number"},
        "fixed_carbon_percent": {"type": "number"},
        "ash_percent": {"type": "number"},
        "moisture_percent": {"type": "number"}
      }
    },
    "analysis_basis": {
      "type": "string",
      "enum": ["as_received", "dry", "dry_ash_free"],
      "default": "as_received"
    }
  }
}
```

### 10.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "FuelAnalysisOutput",
  "type": "object",
  "required": ["higher_heating_value", "stoichiometric_air"],
  "properties": {
    "higher_heating_value_btu_lb": {
      "type": "number"
    },
    "lower_heating_value_btu_lb": {
      "type": "number"
    },
    "stoichiometric_air_lb_lb_fuel": {
      "type": "number"
    },
    "theoretical_co2_lb_mmbtu": {
      "type": "number"
    },
    "theoretical_so2_lb_mmbtu": {
      "type": "number"
    },
    "flue_gas_volume_scf_mmbtu": {
      "type": "number"
    },
    "composition_converted": {
      "type": "object",
      "description": "Composition on different bases"
    },
    "fuel_classification": {
      "type": "string"
    },
    "provenance_hash": {"type": "string"}
  }
}
```

---

## 11. calculate_dispersion

### 11.1 Description

Calculates atmospheric dispersion using Gaussian plume model with Briggs plume rise and Pasquill-Gifford dispersion coefficients. AERMOD-compatible outputs.

### 11.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DispersionInput",
  "type": "object",
  "required": ["stack_parameters", "meteorological_conditions", "receptor_location"],
  "properties": {
    "stack_parameters": {
      "type": "object",
      "properties": {
        "height_m": {"type": "number"},
        "diameter_m": {"type": "number"},
        "exit_velocity_m_s": {"type": "number"},
        "exit_temperature_k": {"type": "number"},
        "emission_rate_g_s": {"type": "number"}
      }
    },
    "meteorological_conditions": {
      "type": "object",
      "properties": {
        "wind_speed_m_s": {"type": "number"},
        "wind_direction_deg": {"type": "number"},
        "ambient_temperature_k": {"type": "number"},
        "stability_class": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]},
        "mixing_height_m": {"type": "number"}
      }
    },
    "receptor_location": {
      "type": "object",
      "properties": {
        "x_m": {"type": "number", "description": "Downwind distance"},
        "y_m": {"type": "number", "description": "Crosswind distance"},
        "z_m": {"type": "number", "description": "Height above ground"}
      }
    }
  }
}
```

### 11.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DispersionOutput",
  "type": "object",
  "required": ["ground_level_concentration", "effective_stack_height"],
  "properties": {
    "ground_level_concentration_ug_m3": {
      "type": "number"
    },
    "effective_stack_height_m": {
      "type": "number"
    },
    "plume_rise_m": {
      "type": "number"
    },
    "sigma_y_m": {
      "type": "number",
      "description": "Lateral dispersion coefficient"
    },
    "sigma_z_m": {
      "type": "number",
      "description": "Vertical dispersion coefficient"
    },
    "max_concentration_ug_m3": {
      "type": "number"
    },
    "distance_to_max_m": {
      "type": "number"
    },
    "calculation_steps": {"type": "array"},
    "provenance_hash": {"type": "string"}
  }
}
```

### 11.4 Physics Basis

**Gaussian Plume Equation:**
```
C(x,y,z) = (Q / (2*pi*u*sigma_y*sigma_z)) *
           exp(-y^2 / (2*sigma_y^2)) *
           [exp(-(z-H)^2 / (2*sigma_z^2)) + exp(-(z+H)^2 / (2*sigma_z^2))]
```

**Briggs Plume Rise:**
```
Buoyancy flux: F = g * v * d^2 * (Ts - Ta) / (4 * Ts)
Plume rise: delta_h = 1.6 * F^(1/3) * x_f^(2/3) / u
```

---

## 12. generate_audit_trail

### 12.1 Description

Generates comprehensive audit trails with SHA-256 provenance hashing for all calculations. Supports regulatory audit requirements and data integrity verification.

### 12.2 Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AuditTrailInput",
  "type": "object",
  "required": ["calculation_type", "inputs", "outputs"],
  "properties": {
    "calculation_type": {
      "type": "string"
    },
    "inputs": {
      "type": "object",
      "description": "All calculation inputs"
    },
    "outputs": {
      "type": "object",
      "description": "All calculation outputs"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "user_id": {
      "type": "string"
    },
    "facility_id": {
      "type": "string"
    },
    "regulatory_context": {
      "type": "string"
    }
  }
}
```

### 12.3 Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AuditTrailOutput",
  "type": "object",
  "required": ["audit_record", "provenance_hash"],
  "properties": {
    "audit_record": {
      "type": "object",
      "properties": {
        "record_id": {"type": "string", "format": "uuid"},
        "timestamp": {"type": "string"},
        "calculation_type": {"type": "string"},
        "inputs_hash": {"type": "string"},
        "outputs_hash": {"type": "string"},
        "combined_hash": {"type": "string"},
        "calculation_steps": {"type": "array"},
        "regulatory_citations": {"type": "array"},
        "data_sources": {"type": "array"}
      }
    },
    "provenance_chain": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "step": {"type": "integer"},
          "hash": {"type": "string"},
          "parent_hash": {"type": "string"}
        }
      }
    },
    "verification_status": {
      "type": "string",
      "enum": ["verified", "unverified", "tampered"]
    },
    "provenance_hash": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$"
    }
  }
}
```

### 12.4 Hash Algorithm

```
Input Hash: SHA-256(JSON.stringify(sorted(inputs)))
Output Hash: SHA-256(JSON.stringify(sorted(outputs)))
Combined Hash: SHA-256(input_hash + output_hash + timestamp)
```

---

## Appendix A: Common Unit Conversions

| From | To | Factor |
|------|----|--------|
| lb/MMBtu | kg/GJ | 0.4299 |
| ppm (NOx) | mg/Nm3 | 2.05 (at 0C, 1 atm) |
| short ton | metric ton | 0.9072 |
| scf | Nm3 | 0.02679 |
| Btu | kJ | 1.055 |

---

## Appendix B: Determinism Guarantee

All tools in GL-010 EMISSIONWATCH are designed to produce identical outputs for identical inputs:

1. **Temperature:** 0.0 (no randomness)
2. **Seed:** 42 (fixed for reproducibility)
3. **Rounding:** ROUND_HALF_UP (IEEE 754)
4. **Precision:** 4 decimal places
5. **Hash Algorithm:** SHA-256

---

## Appendix C: Regulatory References Quick Lookup

| Tool | Primary Regulation | Secondary |
|------|-------------------|-----------|
| calculate_nox_emissions | EPA AP-42 | EU LCP BREF |
| calculate_sox_emissions | EPA AP-42 | EU LCP BREF |
| calculate_co2_emissions | 40 CFR Part 98 | ISO 14064 |
| calculate_particulate_matter | EPA Method 5 | EU BAT |
| check_compliance_status | 40 CFR Part 60 | EU IED |
| generate_regulatory_report | 40 CFR Part 75 | MRV Reg |
| detect_violations | Title V | EU ETS |
| predict_exceedances | N/A | N/A |
| calculate_emission_factors | EPA AP-42 | EMEP/EEA |
| analyze_fuel_composition | ASTM | ISO |
| calculate_dispersion | AERMOD | ISC3 |
| generate_audit_trail | 40 CFR Part 75 | ISO 27001 |

---

*Document generated by GL-SpecGuardian v1.0*
*Tool Specification Standard: GreenLang Pack Spec v1.0*
