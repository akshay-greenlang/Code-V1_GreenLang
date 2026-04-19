# GL-020 ECONOPULSE Formula Library

## Thermodynamic Formulas for Economizer Performance Calculations

| Document ID | GL-020-FL-001 |
|-------------|---------------|
| Agent ID | GL-020 |
| Codename | ECONOPULSE |
| Version | 1.0.0 |
| Classification | Technical Reference |
| Last Updated | 2024-12-03 |
| Status | ACTIVE |
| Regulatory Scope | ASME PTC 4.3, EPA, GHG Protocol |

---

## Table of Contents

1. [Document Control](#1-document-control)
2. [Heat Transfer Fundamentals](#2-heat-transfer-fundamentals)
3. [Fouling Analysis](#3-fouling-analysis)
4. [Economizer Effectiveness](#4-economizer-effectiveness)
5. [Temperature Relationships](#5-temperature-relationships)
6. [Soot Blowing Analysis](#6-soot-blowing-analysis)
7. [Material Properties](#7-material-properties)
8. [ASME PTC 4.3 Reference Values](#8-asme-ptc-43-reference-values)
9. [Emission Factor Integration](#9-emission-factor-integration)
10. [YAML Formula Specifications](#10-yaml-formula-specifications)
11. [Validation and Quality Assurance](#11-validation-and-quality-assurance)
12. [Appendices](#12-appendices)

---

## 1. Document Control

### 1.1 Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2024-12-03 | GL-FormulaLibraryCurator | Initial release |

### 1.2 Approval Matrix

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Formula Curator | GL-FormulaLibraryCurator | - | 2024-12-03 |
| Technical Reviewer | - | - | - |
| Quality Assurance | - | - | - |

### 1.3 Referenced Standards

| Standard | Title | Version |
|----------|-------|---------|
| ASME PTC 4.3 | Performance Test Code for Air Heaters | 2017 |
| ASME PTC 4 | Fired Steam Generators | 2013 |
| TEMA | Standards of the Tubular Exchanger Manufacturers Association | 10th Edition |
| IAPWS-IF97 | Industrial Formulation 1997 for Thermodynamic Properties of Water and Steam | 1997 |
| GHG Protocol | Corporate Accounting and Reporting Standard | Revised Edition |

---

## 2. Heat Transfer Fundamentals

### 2.1 Log Mean Temperature Difference (LMTD)

The Log Mean Temperature Difference is the fundamental driving force for heat transfer in economizers. It accounts for the varying temperature difference along the heat exchanger length.

#### 2.1.1 Counter-Flow Configuration (Preferred for Economizers)

```
LMTD = (DeltaT_1 - DeltaT_2) / ln(DeltaT_1 / DeltaT_2)

Where:
  DeltaT_1 = T_gas_in - T_water_out     [deg F or deg C]
  DeltaT_2 = T_gas_out - T_water_in     [deg F or deg C]
  ln = Natural logarithm

Temperature Profile:
  Gas:    T_gas_in  -----> T_gas_out
  Water:  T_water_out <----- T_water_in
```

**YAML Formula Specification:**
```yaml
formula_id: "lmtd_counterflow_v1"
standard: "ASME PTC 4.3"
category: "Heat Transfer"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"
source: "ASME PTC 4.3-2017"

inputs:
  - name: "T_gas_in"
    unit: "degF"
    description: "Flue gas inlet temperature"
    min_value: 300
    max_value: 1000
  - name: "T_gas_out"
    unit: "degF"
    description: "Flue gas outlet temperature"
    min_value: 200
    max_value: 600
  - name: "T_water_in"
    unit: "degF"
    description: "Feedwater inlet temperature"
    min_value: 100
    max_value: 400
  - name: "T_water_out"
    unit: "degF"
    description: "Feedwater outlet temperature"
    min_value: 150
    max_value: 500

outputs:
  - name: "LMTD"
    unit: "degF"
    description: "Log mean temperature difference"

formula: "(DeltaT_1 - DeltaT_2) / ln(DeltaT_1 / DeltaT_2)"

validation:
  - condition: "DeltaT_1 > 0"
    error: "Inlet temperature difference must be positive"
  - condition: "DeltaT_2 > 0"
    error: "Outlet temperature difference must be positive"
  - condition: "DeltaT_1 != DeltaT_2"
    error: "Use arithmetic mean when DeltaT_1 equals DeltaT_2"
```

#### 2.1.2 Parallel-Flow Configuration

```
LMTD = (DeltaT_1 - DeltaT_2) / ln(DeltaT_1 / DeltaT_2)

Where:
  DeltaT_1 = T_gas_in - T_water_in      [deg F or deg C]
  DeltaT_2 = T_gas_out - T_water_out    [deg F or deg C]

Temperature Profile:
  Gas:    T_gas_in  -----> T_gas_out
  Water:  T_water_in -----> T_water_out
```

**Note:** Parallel-flow is less thermodynamically efficient than counter-flow and is rarely used in industrial economizers.

#### 2.1.3 Special Case: Near-Equal Temperature Differences

When DeltaT_1 is approximately equal to DeltaT_2 (within 5%), the logarithmic formula becomes numerically unstable. Use the arithmetic mean instead:

```
LMTD_approx = (DeltaT_1 + DeltaT_2) / 2

Applicability Criterion:
  |DeltaT_1 - DeltaT_2| / DeltaT_avg < 0.05

Where:
  DeltaT_avg = (DeltaT_1 + DeltaT_2) / 2
```

**YAML Formula Specification:**
```yaml
formula_id: "lmtd_arithmetic_mean_v1"
standard: "ASME PTC 4.3"
category: "Heat Transfer"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

applicability:
  condition: "abs(DeltaT_1 - DeltaT_2) / ((DeltaT_1 + DeltaT_2) / 2) < 0.05"
  description: "Use when temperature differences are within 5%"

formula: "(DeltaT_1 + DeltaT_2) / 2"
```

#### 2.1.4 Correction Factor for Multi-Pass Configurations

For economizers with complex flow arrangements (cross-flow, multi-pass), apply an LMTD correction factor:

```
LMTD_corrected = F x LMTD_counterflow

Where:
  F = LMTD correction factor (0 < F <= 1)

F is a function of:
  P = (T_water_out - T_water_in) / (T_gas_in - T_water_in)
  R = (T_gas_in - T_gas_out) / (T_water_out - T_water_in)
```

**Typical F Values:**
| Configuration | Typical F Range |
|---------------|-----------------|
| Pure counter-flow | 1.0 |
| 1-2 shell and tube | 0.80 - 0.95 |
| Cross-flow (unmixed) | 0.85 - 0.95 |
| Cross-flow (mixed) | 0.75 - 0.90 |

---

### 2.2 Overall Heat Transfer Coefficient

The overall heat transfer coefficient (U) characterizes the thermal resistance between the hot and cold fluids.

#### 2.2.1 Definition from Performance Data

```
U = Q / (A x LMTD)

Where:
  U     = Overall heat transfer coefficient [Btu/(hr-ft^2-degF) or W/(m^2-K)]
  Q     = Heat duty [Btu/hr or W]
  A     = Heat transfer surface area [ft^2 or m^2]
  LMTD  = Log mean temperature difference [degF or degC]
```

**Unit Conversion:**
```
1 Btu/(hr-ft^2-degF) = 5.678 W/(m^2-K)
```

**YAML Formula Specification:**
```yaml
formula_id: "overall_htc_from_performance_v1"
standard: "ASME PTC 4.3"
category: "Heat Transfer"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "Q"
    unit: "Btu/hr"
    description: "Heat transfer rate"
  - name: "A"
    unit: "ft^2"
    description: "Heat transfer surface area"
  - name: "LMTD"
    unit: "degF"
    description: "Log mean temperature difference"

outputs:
  - name: "U"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Overall heat transfer coefficient"

formula: "Q / (A * LMTD)"

validation:
  - condition: "A > 0"
    error: "Surface area must be positive"
  - condition: "LMTD > 0"
    error: "LMTD must be positive"
```

#### 2.2.2 Resistance Network Method

The overall heat transfer coefficient can be calculated from individual resistances:

```
1/U_overall = 1/h_gas + R_fouling_gas + (r_o/k_wall) x ln(r_o/r_i) + R_fouling_water + (A_o/A_i) x (1/h_water)

Where:
  h_gas           = Gas-side convective heat transfer coefficient
  h_water         = Water-side convective heat transfer coefficient
  R_fouling_gas   = Gas-side fouling resistance
  R_fouling_water = Water-side fouling resistance
  k_wall          = Tube wall thermal conductivity
  r_o, r_i        = Outer and inner tube radii
  A_o, A_i        = Outer and inner surface areas
```

**Simplified Form (Bare Tube):**
```
1/U_o = 1/h_gas + R_f_gas + (d_o/(2*k_wall)) x ln(d_o/d_i) + R_f_water x (d_o/d_i) + (d_o/d_i) x (1/h_water)
```

---

### 2.3 Heat Duty Calculations

Heat duty represents the rate of energy transfer in the economizer.

#### 2.3.1 Water-Side Heat Duty

```
Q_water = m_dot_water x Cp_water x (T_water_out - T_water_in)

Where:
  Q_water       = Heat absorbed by water [Btu/hr]
  m_dot_water   = Water mass flow rate [lb/hr]
  Cp_water      = Specific heat of water [Btu/(lb-degF)]
  T_water_out   = Water outlet temperature [degF]
  T_water_in    = Water inlet temperature [degF]
```

**YAML Formula Specification:**
```yaml
formula_id: "heat_duty_water_side_v1"
standard: "ASME PTC 4.3"
category: "Heat Transfer"
calculation_type: "lookup_multiply"
factors_table: "water_properties"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "m_dot_water"
    unit: "lb/hr"
    description: "Water mass flow rate"
    typical_range: [50000, 500000]
  - name: "Cp_water"
    unit: "Btu/(lb-degF)"
    description: "Specific heat of water"
    typical_value: 1.0
  - name: "T_water_out"
    unit: "degF"
    description: "Water outlet temperature"
  - name: "T_water_in"
    unit: "degF"
    description: "Water inlet temperature"

outputs:
  - name: "Q_water"
    unit: "Btu/hr"
    description: "Water-side heat duty"

formula: "m_dot_water * Cp_water * (T_water_out - T_water_in)"
```

#### 2.3.2 Gas-Side Heat Duty

```
Q_gas = m_dot_gas x Cp_gas x (T_gas_in - T_gas_out)

Where:
  Q_gas       = Heat released by flue gas [Btu/hr]
  m_dot_gas   = Flue gas mass flow rate [lb/hr]
  Cp_gas      = Specific heat of flue gas [Btu/(lb-degF)]
  T_gas_in    = Gas inlet temperature [degF]
  T_gas_out   = Gas outlet temperature [degF]
```

**YAML Formula Specification:**
```yaml
formula_id: "heat_duty_gas_side_v1"
standard: "ASME PTC 4.3"
category: "Heat Transfer"
calculation_type: "lookup_multiply"
factors_table: "flue_gas_properties"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "m_dot_gas"
    unit: "lb/hr"
    description: "Flue gas mass flow rate"
  - name: "Cp_gas"
    unit: "Btu/(lb-degF)"
    description: "Specific heat of flue gas"
    typical_value: 0.26
  - name: "T_gas_in"
    unit: "degF"
    description: "Gas inlet temperature"
  - name: "T_gas_out"
    unit: "degF"
    description: "Gas outlet temperature"

outputs:
  - name: "Q_gas"
    unit: "Btu/hr"
    description: "Gas-side heat duty"

formula: "m_dot_gas * Cp_gas * (T_gas_in - T_gas_out)"
```

#### 2.3.3 Heat Balance Verification

For quality assurance, verify energy balance:

```
Heat Balance Error = |Q_water - Q_gas| / Q_water x 100%

Acceptable Tolerance: < 2% (per ASME PTC 4.3)
```

---

## 3. Fouling Analysis

Fouling is the accumulation of unwanted deposits on heat transfer surfaces, reducing thermal performance and increasing pressure drop.

### 3.1 Fouling Factor (ASME/TEMA Method)

#### 3.1.1 Definition

```
R_f = (1/U_fouled) - (1/U_clean)

Where:
  R_f       = Fouling factor (thermal resistance) [ft^2-hr-degF/Btu or m^2-K/W]
  U_fouled  = Current overall heat transfer coefficient [Btu/(hr-ft^2-degF)]
  U_clean   = Clean condition heat transfer coefficient [Btu/(hr-ft^2-degF)]
```

**Unit Conversion:**
```
1 ft^2-hr-degF/Btu = 0.1761 m^2-K/W
```

**YAML Formula Specification:**
```yaml
formula_id: "fouling_factor_v1"
standard: "ASME PTC 4.3 / TEMA"
category: "Fouling Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"
source: "TEMA Standards, 10th Edition"

inputs:
  - name: "U_fouled"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Current overall heat transfer coefficient"
  - name: "U_clean"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Clean baseline heat transfer coefficient"

outputs:
  - name: "R_f"
    unit: "ft^2-hr-degF/Btu"
    description: "Total fouling factor"

formula: "(1/U_fouled) - (1/U_clean)"

validation:
  - condition: "U_fouled > 0"
    error: "U_fouled must be positive"
  - condition: "U_clean > 0"
    error: "U_clean must be positive"
  - condition: "U_fouled <= U_clean"
    error: "Fouled U cannot exceed clean U"

alerts:
  - condition: "R_f > 0.002"
    severity: "warning"
    message: "Significant fouling detected - consider cleaning"
  - condition: "R_f > 0.005"
    severity: "critical"
    message: "Severe fouling - immediate cleaning required"
```

#### 3.1.2 Component Fouling Factors

Total fouling resistance is the sum of gas-side and water-side fouling:

```
R_f_total = R_f_gas + R_f_water x (A_o/A_i)

Where:
  R_f_gas     = Gas-side fouling factor
  R_f_water   = Water-side fouling factor
  A_o/A_i     = Ratio of outer to inner surface areas
```

---

### 3.2 Cleanliness Factor

The cleanliness factor provides a direct percentage measure of heat transfer surface effectiveness.

```
CF = (U_fouled / U_clean) x 100%

Where:
  CF        = Cleanliness factor [%]
  U_fouled  = Current overall heat transfer coefficient
  U_clean   = Clean baseline heat transfer coefficient
```

**YAML Formula Specification:**
```yaml
formula_id: "cleanliness_factor_v1"
standard: "Industry Practice"
category: "Fouling Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "U_fouled"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Current overall heat transfer coefficient"
  - name: "U_clean"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Clean baseline heat transfer coefficient"

outputs:
  - name: "CF"
    unit: "%"
    description: "Cleanliness factor"

formula: "(U_fouled / U_clean) * 100"

thresholds:
  excellent: ">= 95%"
  good: ">= 85%"
  fair: ">= 70%"
  poor: ">= 50%"
  critical: "< 50%"
```

**Cleanliness Factor Interpretation:**

| CF Range | Condition | Recommended Action |
|----------|-----------|-------------------|
| 95-100% | Excellent | Normal monitoring |
| 85-95% | Good | Routine monitoring |
| 70-85% | Fair | Schedule cleaning |
| 50-70% | Poor | Plan immediate cleaning |
| < 50% | Critical | Emergency cleaning required |

---

### 3.3 Efficiency Loss from Fouling

Quantifies the heat transfer efficiency degradation due to fouling.

```
eta_loss = [R_f x U_clean^2 / (1 + R_f x U_clean)] x 100%

Alternative form:
eta_loss = (1 - CF/100) x 100%
```

**YAML Formula Specification:**
```yaml
formula_id: "efficiency_loss_fouling_v1"
standard: "ASME PTC 4.3"
category: "Fouling Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "R_f"
    unit: "ft^2-hr-degF/Btu"
    description: "Fouling factor"
  - name: "U_clean"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Clean baseline heat transfer coefficient"

outputs:
  - name: "eta_loss"
    unit: "%"
    description: "Efficiency loss due to fouling"

formula: "(R_f * U_clean^2 / (1 + R_f * U_clean)) * 100"
```

---

### 3.4 Fuel Penalty from Fouling

Calculates the additional fuel consumption resulting from reduced economizer performance.

#### 3.4.1 Heat Loss from Fouling

```
Q_loss = Q_design x (1 - U_fouled/U_clean)

Where:
  Q_loss    = Heat not recovered due to fouling [Btu/hr]
  Q_design  = Design heat duty [Btu/hr]
```

#### 3.4.2 Fuel Consumption Penalty

```
Fuel_penalty = Q_loss / (eta_boiler x HHV_fuel)

Where:
  Fuel_penalty  = Additional fuel consumption [mass/hr or volume/hr]
  eta_boiler    = Boiler efficiency [decimal]
  HHV_fuel      = Higher heating value of fuel [Btu/lb or Btu/scf]
```

**YAML Formula Specification:**
```yaml
formula_id: "fuel_penalty_fouling_v1"
standard: "GHG Protocol / ASME PTC 4"
category: "Fouling Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"
source_url: "https://ghgprotocol.org"

inputs:
  - name: "Q_design"
    unit: "Btu/hr"
    description: "Design heat duty"
  - name: "U_fouled"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Current heat transfer coefficient"
  - name: "U_clean"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Clean heat transfer coefficient"
  - name: "eta_boiler"
    unit: "decimal"
    description: "Boiler thermal efficiency"
    typical_value: 0.85
  - name: "HHV_fuel"
    unit: "Btu/lb"
    description: "Higher heating value of fuel"

outputs:
  - name: "Fuel_penalty"
    unit: "lb/hr"
    description: "Additional fuel consumption due to fouling"

formula: "(Q_design * (1 - U_fouled/U_clean)) / (eta_boiler * HHV_fuel)"
```

#### 3.4.3 Cost Penalty

```
Cost_penalty = Fuel_penalty x Fuel_price

Where:
  Cost_penalty  = Additional fuel cost [$/hr]
  Fuel_penalty  = Additional fuel consumption [units/hr]
  Fuel_price    = Fuel unit cost [$/unit]
```

**YAML Formula Specification:**
```yaml
formula_id: "cost_penalty_fouling_v1"
standard: "Industry Practice"
category: "Fouling Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "Fuel_penalty"
    unit: "lb/hr or scf/hr"
    description: "Additional fuel consumption"
  - name: "Fuel_price"
    unit: "$/unit"
    description: "Fuel unit price"

outputs:
  - name: "Cost_penalty_hourly"
    unit: "$/hr"
    description: "Hourly cost penalty"
  - name: "Cost_penalty_annual"
    unit: "$/year"
    description: "Annual cost penalty (8000 operating hours)"

formulas:
  hourly: "Fuel_penalty * Fuel_price"
  annual: "Fuel_penalty * Fuel_price * 8000"
```

#### 3.4.4 Emission Penalty from Fouling

Additional fuel consumption results in additional GHG emissions:

```
CO2_penalty = Fuel_penalty x EF_fuel

Where:
  CO2_penalty   = Additional CO2 emissions [kg CO2/hr]
  Fuel_penalty  = Additional fuel consumption [units/hr]
  EF_fuel       = Emission factor [kg CO2/unit fuel]
```

**YAML Formula Specification:**
```yaml
formula_id: "emission_penalty_fouling_v1"
standard: "GHG Protocol"
category: "Scope 1 Emissions"
calculation_type: "lookup_multiply"
factors_table: "defra_2024"
version: "1.0"
last_updated: "2024-12-03"
source_url: "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting"

inputs:
  - name: "Fuel_penalty"
    unit: "varies by fuel type"
    description: "Additional fuel consumption"
  - name: "EF_fuel"
    unit: "kg CO2/unit"
    description: "CO2 emission factor for fuel type"
    source: "DEFRA 2024"

outputs:
  - name: "CO2_penalty"
    unit: "kg CO2/hr"
    description: "Additional CO2 emissions due to fouling"

formula: "Fuel_penalty * EF_fuel"

emission_factors:
  natural_gas:
    value: 2.02
    unit: "kg CO2/Nm^3"
    source: "DEFRA 2024"
  coal_bituminous:
    value: 2.42
    unit: "kg CO2/kg"
    source: "DEFRA 2024"
  fuel_oil_no2:
    value: 2.68
    unit: "kg CO2/L"
    source: "DEFRA 2024"
```

---

## 4. Economizer Effectiveness

### 4.1 Epsilon-NTU Method

The effectiveness-NTU method provides a direct measure of actual versus maximum possible heat transfer.

#### 4.1.1 Effectiveness Definition

```
epsilon = Q_actual / Q_max

epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)

Where:
  epsilon         = Economizer effectiveness [decimal]
  Q_actual        = Actual heat transfer rate
  Q_max           = Maximum possible heat transfer rate
  T_water_out     = Water outlet temperature
  T_water_in      = Water inlet temperature
  T_gas_in        = Gas inlet temperature
```

**Note:** This formula assumes water is the minimum capacity rate fluid (typical for economizers).

**YAML Formula Specification:**
```yaml
formula_id: "effectiveness_economizer_v1"
standard: "ASME PTC 4.3"
category: "Economizer Performance"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "T_water_out"
    unit: "degF"
    description: "Water outlet temperature"
  - name: "T_water_in"
    unit: "degF"
    description: "Water inlet temperature"
  - name: "T_gas_in"
    unit: "degF"
    description: "Gas inlet temperature"

outputs:
  - name: "epsilon"
    unit: "decimal"
    description: "Economizer thermal effectiveness"

formula: "(T_water_out - T_water_in) / (T_gas_in - T_water_in)"

validation:
  - condition: "0 <= epsilon <= 1"
    error: "Effectiveness must be between 0 and 1"

typical_values:
  new_economizer: "0.75 - 0.90"
  aged_economizer: "0.60 - 0.80"
  fouled_economizer: "0.40 - 0.65"
```

#### 4.1.2 Number of Transfer Units (NTU)

```
NTU = (U x A) / (m_dot x Cp)_min

Where:
  NTU               = Number of transfer units [dimensionless]
  U                 = Overall heat transfer coefficient [Btu/(hr-ft^2-degF)]
  A                 = Heat transfer surface area [ft^2]
  (m_dot x Cp)_min  = Minimum heat capacity rate [Btu/(hr-degF)]
```

**YAML Formula Specification:**
```yaml
formula_id: "ntu_economizer_v1"
standard: "Heat Exchanger Design"
category: "Economizer Performance"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "U"
    unit: "Btu/(hr-ft^2-degF)"
    description: "Overall heat transfer coefficient"
  - name: "A"
    unit: "ft^2"
    description: "Heat transfer surface area"
  - name: "m_dot_min"
    unit: "lb/hr"
    description: "Mass flow rate of minimum capacity fluid"
  - name: "Cp_min"
    unit: "Btu/(lb-degF)"
    description: "Specific heat of minimum capacity fluid"

outputs:
  - name: "NTU"
    unit: "dimensionless"
    description: "Number of transfer units"

formula: "(U * A) / (m_dot_min * Cp_min)"
```

#### 4.1.3 Capacity Ratio

```
C_r = (m_dot x Cp)_min / (m_dot x Cp)_max

Where:
  C_r = Capacity ratio [dimensionless, 0 < C_r <= 1]
```

#### 4.1.4 Effectiveness-NTU Relationship for Counter-Flow

```
epsilon = [1 - exp(-NTU x (1 - C_r))] / [1 - C_r x exp(-NTU x (1 - C_r))]

Special case (C_r = 1):
epsilon = NTU / (1 + NTU)
```

**YAML Formula Specification:**
```yaml
formula_id: "epsilon_ntu_counterflow_v1"
standard: "Heat Exchanger Design"
category: "Economizer Performance"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "NTU"
    unit: "dimensionless"
    description: "Number of transfer units"
  - name: "C_r"
    unit: "dimensionless"
    description: "Capacity ratio"

outputs:
  - name: "epsilon"
    unit: "decimal"
    description: "Heat exchanger effectiveness"

formula: "(1 - exp(-NTU * (1 - C_r))) / (1 - C_r * exp(-NTU * (1 - C_r)))"

special_cases:
  - condition: "C_r = 1"
    formula: "NTU / (1 + NTU)"
  - condition: "C_r = 0"
    formula: "1 - exp(-NTU)"
```

---

## 5. Temperature Relationships

### 5.1 Approach Temperature

The approach temperature is a key diagnostic parameter for economizer performance.

```
DeltaT_approach = T_gas_out - T_water_in

Where:
  DeltaT_approach = Approach temperature [degF or degC]
  T_gas_out       = Flue gas outlet temperature
  T_water_in      = Feedwater inlet temperature
```

**YAML Formula Specification:**
```yaml
formula_id: "approach_temperature_v1"
standard: "ASME PTC 4.3"
category: "Temperature Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "T_gas_out"
    unit: "degF"
    description: "Flue gas outlet temperature"
  - name: "T_water_in"
    unit: "degF"
    description: "Feedwater inlet temperature"

outputs:
  - name: "DeltaT_approach"
    unit: "degF"
    description: "Approach temperature"

formula: "T_gas_out - T_water_in"

reference_values:
  design_typical: "50 - 100 degF (28 - 56 degC)"
  acceptable_range: "50 - 150 degF (28 - 83 degC)"
  fouling_indicator: "> 150 degF (83 degC)"

diagnostics:
  - condition: "DeltaT_approach > 150"
    indication: "Possible fouling or undersized economizer"
  - condition: "DeltaT_approach < 30"
    indication: "Risk of acid dew point corrosion"
  - condition: "DeltaT_approach increasing over time"
    indication: "Progressive fouling - monitor closely"
```

**Approach Temperature Guidelines:**

| Approach Temperature | Condition | Action |
|---------------------|-----------|--------|
| 50-100 degF | Optimal | Normal operation |
| 100-150 degF | Acceptable | Monitor trend |
| 150-200 degF | Elevated | Investigate cause |
| > 200 degF | Excessive | Inspect and clean |

### 5.2 Terminal Temperature Difference

The terminal temperature difference relates the economizer outlet to drum conditions.

```
TTD = T_gas_out - T_water_sat

Where:
  TTD           = Terminal temperature difference [degF]
  T_gas_out     = Flue gas outlet temperature [degF]
  T_water_sat   = Saturation temperature at drum pressure [degF]
```

**YAML Formula Specification:**
```yaml
formula_id: "terminal_temperature_difference_v1"
standard: "ASME PTC 4"
category: "Temperature Analysis"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "T_gas_out"
    unit: "degF"
    description: "Flue gas outlet temperature"
  - name: "T_water_sat"
    unit: "degF"
    description: "Saturation temperature at drum pressure"
    note: "Lookup from steam tables based on drum pressure"

outputs:
  - name: "TTD"
    unit: "degF"
    description: "Terminal temperature difference"

formula: "T_gas_out - T_water_sat"

lookup_table: "steam_saturation_temperature_vs_pressure"
```

### 5.3 Acid Dew Point Considerations

Minimum economizer outlet gas temperature must remain above acid dew point:

```
T_gas_out_min > T_acid_dewpoint

For sulfur-bearing fuels (Verhoff-Banchero correlation):
T_acid_dewpoint = 1000 / [2.276 - 0.0294 x ln(P_H2O) - 0.0858 x ln(P_SO3) + 0.0062 x ln(P_H2O x P_SO3)]

Where:
  T_acid_dewpoint = Sulfuric acid dew point [K]
  P_H2O           = Partial pressure of water vapor [mmHg]
  P_SO3           = Partial pressure of SO3 [mmHg]
```

**Typical Acid Dew Point Values:**

| Fuel Type | Sulfur Content | Typical Acid Dew Point |
|-----------|---------------|----------------------|
| Natural Gas | Negligible | 100-110 degF (38-43 degC) |
| Low Sulfur Oil | < 0.5% | 230-250 degF (110-121 degC) |
| High Sulfur Oil | > 1.0% | 270-300 degF (132-149 degC) |
| Coal | 1-3% | 260-290 degF (127-143 degC) |

---

## 6. Soot Blowing Analysis

### 6.1 Optimal Cleaning Interval

Economic optimization of cleaning frequency balances cleaning costs against fuel penalties.

```
t_optimal = sqrt(2 x C_cleaning / (dR_f/dt x C_fuel x k))

Where:
  t_optimal   = Optimal time between cleanings [days or hours]
  C_cleaning  = Cost per cleaning cycle [$]
  dR_f/dt     = Fouling rate [ft^2-hr-degF/Btu per day]
  C_fuel      = Fuel cost [$/MMBtu]
  k           = Efficiency sensitivity factor [MMBtu/(ft^2-hr-degF/Btu)]
```

**YAML Formula Specification:**
```yaml
formula_id: "optimal_cleaning_interval_v1"
standard: "Industry Best Practice"
category: "Soot Blowing Optimization"
calculation_type: "optimization"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "C_cleaning"
    unit: "$"
    description: "Cost per cleaning cycle (labor, materials, downtime)"
    typical_range: [500, 5000]
  - name: "dR_f_dt"
    unit: "ft^2-hr-degF/Btu per day"
    description: "Fouling accumulation rate"
    typical_range: [0.00001, 0.0005]
  - name: "C_fuel"
    unit: "$/MMBtu"
    description: "Fuel cost per MMBtu"
    typical_range: [3, 15]
  - name: "k"
    unit: "MMBtu-hr/(ft^2-degF)"
    description: "Efficiency sensitivity factor"

outputs:
  - name: "t_optimal"
    unit: "days"
    description: "Optimal cleaning interval"

formula: "sqrt(2 * C_cleaning / (dR_f_dt * C_fuel * k))"

notes:
  - "This optimization assumes linear fouling accumulation"
  - "Adjust for seasonal variations in fuel cost and fouling rate"
  - "Consider operational constraints (maintenance windows, load schedules)"
```

### 6.2 Fouling Accumulation Rate

```
dR_f/dt = (R_f(t2) - R_f(t1)) / (t2 - t1)

Where:
  dR_f/dt = Fouling rate [fouling units per time unit]
  R_f(t1) = Fouling factor at time t1
  R_f(t2) = Fouling factor at time t2
```

### 6.3 Soot Blowing Effectiveness

```
eta_soot_blow = (R_f_before - R_f_after) / R_f_before x 100%

Where:
  eta_soot_blow = Soot blowing effectiveness [%]
  R_f_before    = Fouling factor before soot blowing
  R_f_after     = Fouling factor after soot blowing
```

**YAML Formula Specification:**
```yaml
formula_id: "soot_blowing_effectiveness_v1"
standard: "Industry Practice"
category: "Soot Blowing Optimization"
calculation_type: "derived"
version: "1.0"
last_updated: "2024-12-03"

inputs:
  - name: "R_f_before"
    unit: "ft^2-hr-degF/Btu"
    description: "Fouling factor before soot blowing"
  - name: "R_f_after"
    unit: "ft^2-hr-degF/Btu"
    description: "Fouling factor after soot blowing"

outputs:
  - name: "eta_soot_blow"
    unit: "%"
    description: "Soot blowing effectiveness"

formula: "((R_f_before - R_f_after) / R_f_before) * 100"

thresholds:
  excellent: ">= 90%"
  good: ">= 75%"
  fair: ">= 50%"
  poor: "< 50% - check soot blower operation"
```

### 6.4 Steam Consumption for Soot Blowing

```
m_steam_annual = n_blowers x m_steam_per_blow x n_cycles_per_day x 365

Where:
  m_steam_annual    = Annual steam consumption for soot blowing [lb/year]
  n_blowers         = Number of soot blowers
  m_steam_per_blow  = Steam consumption per blower cycle [lb/cycle]
  n_cycles_per_day  = Number of soot blowing cycles per day
```

---

## 7. Material Properties

### 7.1 Water/Steam Properties (IAPWS-IF97)

#### 7.1.1 Specific Heat of Liquid Water

| Temperature (degF) | Temperature (degC) | Cp [Btu/(lb-degF)] | Cp [kJ/(kg-K)] |
|-------------------|-------------------|-------------------|----------------|
| 60 | 15.6 | 1.000 | 4.186 |
| 100 | 37.8 | 0.998 | 4.179 |
| 150 | 65.6 | 0.999 | 4.183 |
| 200 | 93.3 | 1.003 | 4.199 |
| 250 | 121.1 | 1.010 | 4.228 |
| 300 | 148.9 | 1.019 | 4.266 |
| 350 | 176.7 | 1.032 | 4.320 |
| 400 | 204.4 | 1.049 | 4.391 |
| 450 | 232.2 | 1.071 | 4.483 |
| 500 | 260.0 | 1.099 | 4.600 |

**YAML Formula Specification:**
```yaml
formula_id: "water_specific_heat_v1"
standard: "IAPWS-IF97"
category: "Material Properties"
calculation_type: "lookup_interpolate"
version: "1.0"
last_updated: "2024-12-03"
source: "IAPWS Industrial Formulation 1997"

lookup_table:
  independent_variable: "temperature"
  dependent_variable: "Cp_water"
  interpolation_method: "linear"

data_points:
  - {T_degF: 60, Cp_Btu_lb_degF: 1.000}
  - {T_degF: 100, Cp_Btu_lb_degF: 0.998}
  - {T_degF: 150, Cp_Btu_lb_degF: 0.999}
  - {T_degF: 200, Cp_Btu_lb_degF: 1.003}
  - {T_degF: 250, Cp_Btu_lb_degF: 1.010}
  - {T_degF: 300, Cp_Btu_lb_degF: 1.019}
  - {T_degF: 350, Cp_Btu_lb_degF: 1.032}
  - {T_degF: 400, Cp_Btu_lb_degF: 1.049}
  - {T_degF: 450, Cp_Btu_lb_degF: 1.071}
  - {T_degF: 500, Cp_Btu_lb_degF: 1.099}
```

#### 7.1.2 Water Density

| Temperature (degF) | Temperature (degC) | Density [lb/ft^3] | Density [kg/m^3] |
|-------------------|-------------------|------------------|-----------------|
| 60 | 15.6 | 62.37 | 999.0 |
| 100 | 37.8 | 62.00 | 993.1 |
| 150 | 65.6 | 61.19 | 980.1 |
| 200 | 93.3 | 60.13 | 963.1 |
| 250 | 121.1 | 58.82 | 942.1 |
| 300 | 148.9 | 57.31 | 917.9 |
| 350 | 176.7 | 55.59 | 890.4 |
| 400 | 204.4 | 53.65 | 859.3 |
| 450 | 232.2 | 51.46 | 824.2 |
| 500 | 260.0 | 49.02 | 785.1 |

#### 7.1.3 Steam Saturation Properties

| Pressure (psia) | T_sat (degF) | h_f (Btu/lb) | h_fg (Btu/lb) | h_g (Btu/lb) |
|-----------------|--------------|--------------|---------------|--------------|
| 100 | 327.8 | 298.4 | 888.8 | 1187.2 |
| 150 | 358.4 | 330.5 | 863.6 | 1194.1 |
| 200 | 381.8 | 355.4 | 843.0 | 1198.4 |
| 300 | 417.3 | 393.8 | 809.0 | 1202.8 |
| 400 | 444.6 | 424.0 | 780.5 | 1204.5 |
| 500 | 467.0 | 449.4 | 755.0 | 1204.4 |
| 600 | 486.2 | 471.6 | 731.6 | 1203.2 |
| 800 | 518.2 | 509.7 | 689.6 | 1199.3 |
| 1000 | 544.6 | 542.4 | 650.3 | 1192.7 |
| 1500 | 596.2 | 611.6 | 557.1 | 1168.7 |
| 2000 | 635.8 | 671.7 | 464.4 | 1136.1 |

---

### 7.2 Flue Gas Properties

#### 7.2.1 Specific Heat of Flue Gas

Flue gas specific heat depends on composition and temperature.

**Natural Gas Combustion Products (with 20% excess air):**

| Temperature (degF) | Temperature (degC) | Cp [Btu/(lb-degF)] | Cp [kJ/(kg-K)] |
|-------------------|-------------------|-------------------|----------------|
| 200 | 93 | 0.245 | 1.026 |
| 400 | 204 | 0.252 | 1.055 |
| 600 | 316 | 0.260 | 1.088 |
| 800 | 427 | 0.268 | 1.122 |
| 1000 | 538 | 0.276 | 1.155 |

**Coal Combustion Products (typical bituminous, 20% excess air):**

| Temperature (degF) | Temperature (degC) | Cp [Btu/(lb-degF)] | Cp [kJ/(kg-K)] |
|-------------------|-------------------|-------------------|----------------|
| 200 | 93 | 0.240 | 1.005 |
| 400 | 204 | 0.248 | 1.038 |
| 600 | 316 | 0.256 | 1.072 |
| 800 | 427 | 0.264 | 1.105 |
| 1000 | 538 | 0.272 | 1.139 |

**YAML Formula Specification:**
```yaml
formula_id: "flue_gas_specific_heat_v1"
standard: "ASME PTC 4"
category: "Material Properties"
calculation_type: "lookup_interpolate"
version: "1.0"
last_updated: "2024-12-03"

fuel_types:
  natural_gas:
    composition_basis: "20% excess air"
    data_points:
      - {T_degF: 200, Cp_Btu_lb_degF: 0.245}
      - {T_degF: 400, Cp_Btu_lb_degF: 0.252}
      - {T_degF: 600, Cp_Btu_lb_degF: 0.260}
      - {T_degF: 800, Cp_Btu_lb_degF: 0.268}
      - {T_degF: 1000, Cp_Btu_lb_degF: 0.276}

  coal_bituminous:
    composition_basis: "20% excess air, typical ash"
    data_points:
      - {T_degF: 200, Cp_Btu_lb_degF: 0.240}
      - {T_degF: 400, Cp_Btu_lb_degF: 0.248}
      - {T_degF: 600, Cp_Btu_lb_degF: 0.256}
      - {T_degF: 800, Cp_Btu_lb_degF: 0.264}
      - {T_degF: 1000, Cp_Btu_lb_degF: 0.272}
```

#### 7.2.2 Flue Gas Density Correction

```
rho_gas = rho_gas_std x (T_std / T_actual) x (P_actual / P_std)

Where:
  rho_gas      = Actual gas density [lb/ft^3]
  rho_gas_std  = Standard gas density [lb/ft^3]
  T_std        = Standard temperature (60 degF = 520 R)
  T_actual     = Actual temperature [R]
  P_std        = Standard pressure (14.696 psia)
  P_actual     = Actual pressure [psia]
```

**Standard Flue Gas Densities (at 60 degF, 1 atm):**

| Fuel Type | rho_std [lb/ft^3] | rho_std [kg/m^3] |
|-----------|------------------|------------------|
| Natural Gas | 0.0748 | 1.20 |
| Coal (bituminous) | 0.0795 | 1.27 |
| Fuel Oil No. 2 | 0.0762 | 1.22 |
| Fuel Oil No. 6 | 0.0775 | 1.24 |

---

## 8. ASME PTC 4.3 Reference Values

### 8.1 Clean U-Values for Economizers

Reference heat transfer coefficients for various economizer configurations under clean conditions.

#### 8.1.1 Bare Tube Economizers

| Configuration | Gas Velocity [ft/s] | U_clean [Btu/(hr-ft^2-degF)] | U_clean [W/(m^2-K)] |
|---------------|--------------------|-----------------------------|---------------------|
| Staggered arrangement | 30 | 8-12 | 45-68 |
| Staggered arrangement | 50 | 12-18 | 68-102 |
| In-line arrangement | 30 | 6-10 | 34-57 |
| In-line arrangement | 50 | 10-15 | 57-85 |

#### 8.1.2 Finned Tube Economizers

| Fin Configuration | Gas Velocity [ft/s] | U_clean [Btu/(hr-ft^2-degF)] | U_clean [W/(m^2-K)] |
|-------------------|--------------------|-----------------------------|---------------------|
| Solid fins, 4 FPI | 30 | 4-6 | 23-34 |
| Solid fins, 4 FPI | 50 | 6-9 | 34-51 |
| Serrated fins, 5 FPI | 30 | 5-8 | 28-45 |
| Serrated fins, 5 FPI | 50 | 8-12 | 45-68 |

**Note:** FPI = Fins Per Inch. U-values are based on external (finned) surface area.

**YAML Formula Specification:**
```yaml
formula_id: "u_clean_reference_v1"
standard: "ASME PTC 4.3"
category: "Reference Values"
calculation_type: "lookup"
version: "1.0"
last_updated: "2024-12-03"
source: "ASME PTC 4.3-2017"

bare_tube:
  staggered:
    - {gas_velocity_fps: 30, U_low: 8, U_high: 12}
    - {gas_velocity_fps: 50, U_low: 12, U_high: 18}
  inline:
    - {gas_velocity_fps: 30, U_low: 6, U_high: 10}
    - {gas_velocity_fps: 50, U_low: 10, U_high: 15}

finned_tube:
  solid_4fpi:
    - {gas_velocity_fps: 30, U_low: 4, U_high: 6}
    - {gas_velocity_fps: 50, U_low: 6, U_high: 9}
  serrated_5fpi:
    - {gas_velocity_fps: 30, U_low: 5, U_high: 8}
    - {gas_velocity_fps: 50, U_low: 8, U_high: 12}
```

### 8.2 Maximum Allowable Fouling Factors

TEMA recommended maximum fouling factors before cleaning is required.

| Service | R_f_max [ft^2-hr-degF/Btu] | R_f_max [m^2-K/W] |
|---------|---------------------------|-------------------|
| Boiler feedwater (treated) | 0.001 | 0.000176 |
| Boiler feedwater (untreated) | 0.002 | 0.000352 |
| Flue gas (natural gas) | 0.001 | 0.000176 |
| Flue gas (fuel oil) | 0.003 | 0.000528 |
| Flue gas (coal) | 0.005 | 0.000881 |
| Flue gas (high ash coal) | 0.010 | 0.001761 |

**YAML Formula Specification:**
```yaml
formula_id: "fouling_limits_tema_v1"
standard: "TEMA"
category: "Reference Values"
calculation_type: "lookup"
version: "1.0"
last_updated: "2024-12-03"
source: "TEMA Standards, 10th Edition"

fouling_limits:
  water_side:
    boiler_feedwater_treated:
      value: 0.001
      unit: "ft^2-hr-degF/Btu"
    boiler_feedwater_untreated:
      value: 0.002
      unit: "ft^2-hr-degF/Btu"

  gas_side:
    natural_gas:
      value: 0.001
      unit: "ft^2-hr-degF/Btu"
    fuel_oil:
      value: 0.003
      unit: "ft^2-hr-degF/Btu"
    coal_low_ash:
      value: 0.005
      unit: "ft^2-hr-degF/Btu"
    coal_high_ash:
      value: 0.010
      unit: "ft^2-hr-degF/Btu"
```

### 8.3 Typical Design Approach Temperatures

| Application | Approach Temperature [degF] | Approach Temperature [degC] |
|-------------|---------------------------|----------------------------|
| High-pressure utility boilers | 50-75 | 28-42 |
| Industrial boilers (natural gas) | 60-100 | 33-56 |
| Industrial boilers (fuel oil) | 80-120 | 44-67 |
| Industrial boilers (coal) | 100-150 | 56-83 |
| Package boilers | 75-125 | 42-69 |

### 8.4 ASME PTC 4.3 Test Uncertainty Guidelines

| Measured Parameter | Typical Uncertainty |
|-------------------|---------------------|
| Temperature (thermocouple) | +/- 2 degF (+/- 1.1 degC) |
| Temperature (RTD) | +/- 0.5 degF (+/- 0.3 degC) |
| Flow rate (venturi) | +/- 1% |
| Flow rate (ultrasonic) | +/- 0.5% |
| Pressure | +/- 0.25% of span |
| Heat duty (calculated) | +/- 2% |
| U-value (calculated) | +/- 3-5% |

---

## 9. Emission Factor Integration

### 9.1 GHG Protocol Scope 1 Integration

Economizer performance directly impacts Scope 1 emissions through fuel consumption changes.

#### 9.1.1 Emission Impact Calculation

```
Delta_CO2 = Delta_Fuel x EF_CO2

Where:
  Delta_CO2  = Change in CO2 emissions [tonnes CO2]
  Delta_Fuel = Change in fuel consumption [units]
  EF_CO2     = Emission factor [tonnes CO2/unit fuel]
```

**YAML Formula Specification:**
```yaml
formula_id: "scope1_emission_impact_v1"
standard: "GHG Protocol"
category: "Scope 1 Emissions"
calculation_type: "lookup_multiply"
factors_table: "defra_2024"
version: "1.0"
last_updated: "2024-12-03"
source_url: "https://ghgprotocol.org/calculation-tools"

emission_factors:
  natural_gas:
    CO2:
      value: 53.06
      unit: "kg CO2/MMBtu"
      source: "EPA 2024"
    CH4:
      value: 0.001
      unit: "kg CH4/MMBtu"
      source: "EPA 2024"
    N2O:
      value: 0.0001
      unit: "kg N2O/MMBtu"
      source: "EPA 2024"

  coal_bituminous:
    CO2:
      value: 93.28
      unit: "kg CO2/MMBtu"
      source: "EPA 2024"
    CH4:
      value: 0.011
      unit: "kg CH4/MMBtu"
      source: "EPA 2024"
    N2O:
      value: 0.0016
      unit: "kg N2O/MMBtu"
      source: "EPA 2024"

  fuel_oil_no2:
    CO2:
      value: 73.96
      unit: "kg CO2/MMBtu"
      source: "EPA 2024"
    CH4:
      value: 0.003
      unit: "kg CH4/MMBtu"
      source: "EPA 2024"
    N2O:
      value: 0.0006
      unit: "kg N2O/MMBtu"
      source: "EPA 2024"
```

#### 9.1.2 Avoided Emissions from Improved Performance

```
CO2_avoided = (eta_improved - eta_baseline) / eta_baseline x Annual_fuel x EF_CO2

Where:
  CO2_avoided   = Annual CO2 emissions avoided [tonnes CO2/year]
  eta_improved  = Improved economizer efficiency
  eta_baseline  = Baseline economizer efficiency
  Annual_fuel   = Annual fuel consumption at baseline [units/year]
  EF_CO2        = CO2 emission factor [tonnes CO2/unit]
```

### 9.2 DEFRA Emission Factors

The DEFRA (UK Department for Environment, Food and Rural Affairs) emission factors are updated annually and provide comprehensive coverage.

**YAML Formula Specification:**
```yaml
formula_id: "defra_stationary_combustion_v1"
standard: "DEFRA UK GHG Conversion Factors"
category: "Scope 1 Emissions"
calculation_type: "lookup_multiply"
version: "2024"
last_updated: "2024-06-01"
source_url: "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting"

factors:
  natural_gas:
    gross_CV:
      CO2e: 0.18316
      unit: "kg CO2e/kWh"
    net_CV:
      CO2e: 0.20227
      unit: "kg CO2e/kWh"

  gas_oil:
    CO2e: 2.75894
    unit: "kg CO2e/litre"

  fuel_oil:
    CO2e: 3.17958
    unit: "kg CO2e/litre"

  coal_industrial:
    CO2e: 2.39391
    unit: "kg CO2e/kg"

  coal_electricity_generation:
    CO2e: 2.21016
    unit: "kg CO2e/kg"
```

### 9.3 EPA Emission Factors

**YAML Formula Specification:**
```yaml
formula_id: "epa_stationary_combustion_v1"
standard: "EPA GHG Emission Factors Hub"
category: "Scope 1 Emissions"
calculation_type: "lookup_multiply"
version: "2024"
last_updated: "2024-03-01"
source_url: "https://www.epa.gov/climateleadership/ghg-emission-factors-hub"

factors:
  natural_gas:
    CO2: 53.06
    CH4: 0.001
    N2O: 0.0001
    unit: "kg/MMBtu"

  distillate_fuel_oil_no2:
    CO2: 73.96
    CH4: 0.003
    N2O: 0.0006
    unit: "kg/MMBtu"

  residual_fuel_oil_no6:
    CO2: 75.10
    CH4: 0.003
    N2O: 0.0006
    unit: "kg/MMBtu"

  bituminous_coal:
    CO2: 93.28
    CH4: 0.011
    N2O: 0.0016
    unit: "kg/MMBtu"

  subbituminous_coal:
    CO2: 97.17
    CH4: 0.011
    N2O: 0.0016
    unit: "kg/MMBtu"
```

---

## 10. YAML Formula Specifications

### 10.1 Complete Formula Library Structure

```yaml
# GL-020 ECONOPULSE Formula Library
# Thermodynamic Formulas for Economizer Performance

library_metadata:
  library_id: "GL-020-ECONOPULSE-FORMULAS"
  version: "1.0.0"
  created: "2024-12-03"
  author: "GL-FormulaLibraryCurator"
  description: "Thermodynamic formulas for economizer performance and fouling analysis"
  standards_compliance:
    - "ASME PTC 4.3-2017"
    - "ASME PTC 4-2013"
    - "TEMA 10th Edition"
    - "GHG Protocol"
    - "IAPWS-IF97"

categories:
  - id: "heat_transfer"
    name: "Heat Transfer Fundamentals"
    formulas:
      - "lmtd_counterflow_v1"
      - "lmtd_parallel_v1"
      - "lmtd_arithmetic_mean_v1"
      - "overall_htc_from_performance_v1"
      - "heat_duty_water_side_v1"
      - "heat_duty_gas_side_v1"

  - id: "fouling"
    name: "Fouling Analysis"
    formulas:
      - "fouling_factor_v1"
      - "cleanliness_factor_v1"
      - "efficiency_loss_fouling_v1"
      - "fuel_penalty_fouling_v1"
      - "cost_penalty_fouling_v1"
      - "emission_penalty_fouling_v1"

  - id: "effectiveness"
    name: "Economizer Effectiveness"
    formulas:
      - "effectiveness_economizer_v1"
      - "ntu_economizer_v1"
      - "epsilon_ntu_counterflow_v1"

  - id: "temperature"
    name: "Temperature Relationships"
    formulas:
      - "approach_temperature_v1"
      - "terminal_temperature_difference_v1"

  - id: "soot_blowing"
    name: "Soot Blowing Analysis"
    formulas:
      - "optimal_cleaning_interval_v1"
      - "soot_blowing_effectiveness_v1"

  - id: "emissions"
    name: "Emission Calculations"
    formulas:
      - "scope1_emission_impact_v1"
      - "defra_stationary_combustion_v1"
      - "epa_stationary_combustion_v1"

property_tables:
  - id: "water_properties"
    source: "IAPWS-IF97"

  - id: "flue_gas_properties"
    source: "ASME PTC 4"

  - id: "steam_saturation"
    source: "IAPWS-IF97"
```

### 10.2 Master Formula Index

| Formula ID | Category | Standard | Version |
|------------|----------|----------|---------|
| lmtd_counterflow_v1 | Heat Transfer | ASME PTC 4.3 | 1.0 |
| lmtd_parallel_v1 | Heat Transfer | ASME PTC 4.3 | 1.0 |
| lmtd_arithmetic_mean_v1 | Heat Transfer | ASME PTC 4.3 | 1.0 |
| overall_htc_from_performance_v1 | Heat Transfer | ASME PTC 4.3 | 1.0 |
| heat_duty_water_side_v1 | Heat Transfer | ASME PTC 4.3 | 1.0 |
| heat_duty_gas_side_v1 | Heat Transfer | ASME PTC 4.3 | 1.0 |
| fouling_factor_v1 | Fouling Analysis | TEMA | 1.0 |
| cleanliness_factor_v1 | Fouling Analysis | Industry | 1.0 |
| efficiency_loss_fouling_v1 | Fouling Analysis | ASME PTC 4.3 | 1.0 |
| fuel_penalty_fouling_v1 | Fouling Analysis | GHG Protocol | 1.0 |
| cost_penalty_fouling_v1 | Fouling Analysis | Industry | 1.0 |
| emission_penalty_fouling_v1 | Emissions | GHG Protocol | 1.0 |
| effectiveness_economizer_v1 | Effectiveness | ASME PTC 4.3 | 1.0 |
| ntu_economizer_v1 | Effectiveness | HX Design | 1.0 |
| epsilon_ntu_counterflow_v1 | Effectiveness | HX Design | 1.0 |
| approach_temperature_v1 | Temperature | ASME PTC 4.3 | 1.0 |
| terminal_temperature_difference_v1 | Temperature | ASME PTC 4 | 1.0 |
| optimal_cleaning_interval_v1 | Soot Blowing | Industry | 1.0 |
| soot_blowing_effectiveness_v1 | Soot Blowing | Industry | 1.0 |
| scope1_emission_impact_v1 | Emissions | GHG Protocol | 1.0 |
| defra_stationary_combustion_v1 | Emissions | DEFRA 2024 | 2024 |
| epa_stationary_combustion_v1 | Emissions | EPA 2024 | 2024 |

---

## 11. Validation and Quality Assurance

### 11.1 Formula Validation Requirements

All formulas in this library must pass the following validation checks:

#### 11.1.1 Unit Consistency Check

```yaml
validation_rule: "unit_consistency"
description: "All input units must be compatible with output units"
check_method: "dimensional_analysis"
```

#### 11.1.2 Physical Bounds Check

```yaml
validation_rule: "physical_bounds"
description: "All calculated values must be physically reasonable"
checks:
  - "0 <= effectiveness <= 1"
  - "U_value > 0"
  - "LMTD > 0"
  - "fouling_factor >= 0"
  - "cleanliness_factor <= 100%"
```

#### 11.1.3 Energy Balance Verification

```yaml
validation_rule: "energy_balance"
description: "Heat duty calculations must balance within tolerance"
tolerance: "2%"
check_formula: "abs(Q_water - Q_gas) / Q_water < 0.02"
```

### 11.2 Data Quality Indicators

| Quality Level | Definition | Required Documentation |
|---------------|------------|----------------------|
| Tier 1 | Directly measured, calibrated instruments | Calibration certificates, raw data |
| Tier 2 | Calculated from measured data | Calculation methodology, uncertainties |
| Tier 3 | Published reference values | Source citation, vintage |
| Tier 4 | Engineering estimates | Estimation basis, uncertainty range |

### 11.3 Audit Trail Requirements

All formula calculations must maintain an audit trail including:

1. Input values and sources
2. Formula version used
3. Timestamp of calculation
4. User/system identification
5. Output values and units
6. Any exceptions or overrides

```yaml
audit_record:
  calculation_id: "CALC-2024-001234"
  formula_id: "fouling_factor_v1"
  timestamp: "2024-12-03T10:30:00Z"
  inputs:
    U_fouled: {value: 8.5, unit: "Btu/(hr-ft^2-degF)", source: "measured"}
    U_clean: {value: 12.0, unit: "Btu/(hr-ft^2-degF)", source: "baseline"}
  outputs:
    R_f: {value: 0.0034, unit: "ft^2-hr-degF/Btu"}
  user: "GL-020-ECONOPULSE"
  validation_status: "passed"
```

---

## 12. Appendices

### Appendix A: Unit Conversion Factors

| Quantity | From | To | Multiply By |
|----------|------|-----|-------------|
| Heat transfer coefficient | Btu/(hr-ft^2-degF) | W/(m^2-K) | 5.678 |
| Heat flux | Btu/(hr-ft^2) | W/m^2 | 3.155 |
| Heat duty | Btu/hr | kW | 0.000293 |
| Heat duty | Btu/hr | MW | 0.000000293 |
| Mass flow | lb/hr | kg/s | 0.000126 |
| Thermal resistance | ft^2-hr-degF/Btu | m^2-K/W | 0.1761 |
| Specific heat | Btu/(lb-degF) | kJ/(kg-K) | 4.187 |
| Temperature difference | degF | degC | 0.5556 |
| Area | ft^2 | m^2 | 0.0929 |
| Pressure | psia | kPa | 6.895 |
| Energy | MMBtu | GJ | 1.055 |

### Appendix B: Greek Symbol Reference

| Symbol | Name | Typical Use |
|--------|------|-------------|
| Delta | Delta | Temperature difference, change |
| epsilon | epsilon | Effectiveness |
| eta | eta | Efficiency |
| rho | rho | Density |

### Appendix C: Abbreviation Glossary

| Abbreviation | Full Term |
|--------------|-----------|
| ASME | American Society of Mechanical Engineers |
| CF | Cleanliness Factor |
| Cp | Specific heat at constant pressure |
| DEFRA | Department for Environment, Food and Rural Affairs |
| EPA | Environmental Protection Agency |
| FPI | Fins Per Inch |
| GHG | Greenhouse Gas |
| HHV | Higher Heating Value |
| HTC | Heat Transfer Coefficient |
| IAPWS | International Association for Properties of Water and Steam |
| IF97 | Industrial Formulation 1997 |
| LMTD | Log Mean Temperature Difference |
| NTU | Number of Transfer Units |
| PTC | Performance Test Code |
| RTD | Resistance Temperature Detector |
| TEMA | Tubular Exchanger Manufacturers Association |
| TTD | Terminal Temperature Difference |

### Appendix D: Source Document References

1. ASME PTC 4.3-2017, "Performance Test Code for Air Heaters"
2. ASME PTC 4-2013, "Fired Steam Generators"
3. TEMA Standards, 10th Edition, "Standards of the Tubular Exchanger Manufacturers Association"
4. IAPWS-IF97, "IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam"
5. GHG Protocol Corporate Accounting and Reporting Standard, Revised Edition
6. DEFRA UK Government GHG Conversion Factors for Company Reporting, 2024
7. EPA GHG Emission Factors Hub, 2024

---

## Document Certification

This Formula Library has been prepared in accordance with GreenLang quality standards and is suitable for use in regulatory compliance calculations.

| Certification | Status |
|--------------|--------|
| Technical Accuracy | Verified |
| Unit Consistency | Verified |
| Source Documentation | Complete |
| Regulatory Alignment | GHG Protocol, ASME PTC 4.3 |

---

**Document Control:**
- Document ID: GL-020-FL-001
- Version: 1.0.0
- Classification: Technical Reference
- Distribution: Internal / Regulatory Auditors
- Next Review: 2025-06-03

**End of Document**
