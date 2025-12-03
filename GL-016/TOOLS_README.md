# GL-016 WATERGUARD - Deterministic Tools Documentation

## Overview

The `tools.py` module provides comprehensive deterministic calculations for boiler water treatment optimization. All calculations are based on industry-standard formulas (ASME, ABMA) and physical/chemical principles. **No LLM involvement** - purely mathematical and deterministic.

**File:** `C:/Users/aksha/Code-V1_GreenLang/GL-016/tools.py`
**Lines of Code:** 1996
**Language:** Python 3.8+

---

## Table of Contents

1. [Water Chemistry Analysis](#1-water-chemistry-analysis)
2. [Blowdown Optimization](#2-blowdown-optimization)
3. [Chemical Dosing](#3-chemical-dosing)
4. [Scale and Corrosion Prediction](#4-scale-and-corrosion-prediction)
5. [Energy and Cost Analysis](#5-energy-and-cost-analysis)
6. [Compliance Checking](#6-compliance-checking)
7. [Data Classes](#data-classes)
8. [Usage Examples](#usage-examples)

---

## 1. Water Chemistry Analysis

### 1.1 `calculate_langelier_saturation_index()`

Calculates LSI to predict scale-forming or corrosive tendency of water.

**Formula:**
```
LSI = pH - pHs
pHs = (9.3 + A + B) - (C + D)
A = (Log10[TDS] - 1) / 10
B = -13.12 × Log10(°C + 273) + 34.55
C = Log10[Ca2+ as CaCO3] - 0.4
D = Log10[Alkalinity as CaCO3]
```

**Parameters:**
- `pH` (float): Actual pH of water (0-14)
- `temperature` (float): Water temperature in Celsius
- `calcium_hardness` (float): Calcium hardness in mg/L as CaCO3
- `alkalinity` (float): Total alkalinity in mg/L as CaCO3
- `tds` (float): Total dissolved solids in mg/L

**Returns:** `float` - LSI value (typically -3 to +3)

**Interpretation:**
- LSI > 0: Scale forming (supersaturated)
- LSI = 0: Neutral (balanced)
- LSI < 0: Corrosive (undersaturated)

**Thread-Safe:** Yes

---

### 1.2 `calculate_ryznar_stability_index()`

Calculates RSI for corrosion/scale prediction.

**Formula:**
```
RSI = 2 × pHs - pH
```

**Parameters:**
- `pH` (float): Actual pH of water
- `pHs` (float): Saturation pH (from LSI calculation)

**Returns:** `float` - RSI value (typically 4-10)

**Interpretation:**
- RSI < 6.0: Heavy scale formation
- RSI 6.0-6.5: Light scale formation
- RSI 6.5-7.0: Little scale or corrosion
- RSI 7.0-7.5: Corrosion likely
- RSI > 7.5: Heavy corrosion

---

### 1.3 `calculate_puckorius_scaling_index()`

Calculates PSI for practical scaling prediction considering buffering capacity.

**Parameters:**
- `pH` (float): Actual pH
- `alkalinity` (float): Total alkalinity in mg/L as CaCO3
- `calcium_hardness` (float, optional): Calcium hardness in mg/L
- `temperature` (float, optional): Temperature in Celsius (default: 25°C)

**Returns:** `float` - PSI value

**Interpretation:**
- PSI < 6.0: Scale forming
- PSI 6.0-7.0: Minimal scale or corrosion
- PSI > 7.0: Corrosive

---

### 1.4 `calculate_larson_skold_index()`

Calculates corrosivity index based on chloride/sulfate to alkalinity ratio.

**Formula:**
```
LSK = (Cl⁻ + SO₄²⁻) / (HCO₃⁻ + CO₃²⁻)
All in equivalents (meq/L)
```

**Parameters:**
- `chloride` (float): Chloride concentration in mg/L
- `sulfate` (float): Sulfate concentration in mg/L
- `alkalinity` (float): Total alkalinity in mg/L as CaCO3

**Returns:** `float` - LSK index value

**Interpretation:**
- LSK < 0.2: Low corrosion risk
- LSK 0.2-0.5: Moderate corrosion risk
- LSK 0.5-1.0: High corrosion risk
- LSK > 1.0: Very high corrosion risk

---

### 1.5 `analyze_water_quality()`

Comprehensive water quality analysis with all indices and compliance checking.

**Parameters:**
- `chemistry_data` (Dict): Dictionary containing:
  - `pH`: pH value
  - `temperature`: Temperature in Celsius
  - `calcium_hardness`: mg/L as CaCO3
  - `alkalinity`: mg/L as CaCO3
  - `tds`: Total dissolved solids in mg/L
  - `chloride`: mg/L
  - `sulfate`: mg/L
  - `pressure`: Boiler pressure in bar (optional)

**Returns:** `WaterQualityAnalysis` dataclass with:
- `lsi_value`: Langelier Saturation Index
- `rsi_value`: Ryznar Stability Index
- `psi_value`: Puckorius Scaling Index
- `larson_skold_index`: Larson-Skold Index
- `scale_tendency`: "scaling", "neutral", or "corrosive"
- `corrosion_risk`: "low", "moderate", "high", or "severe"
- `compliance_status`: "PASS", "WARNING", or "FAIL"
- `violations`: List of compliance violations
- `recommendations`: List of recommended actions
- `timestamp`: UTC ISO timestamp
- `provenance_hash`: SHA-256 hash for auditability

---

## 2. Blowdown Optimization

### 2.1 `calculate_cycles_of_concentration()`

Calculates cycles of concentration for boiler water.

**Formula:**
```
Cycles = Blowdown Conductivity / Makeup Conductivity
```

**Parameters:**
- `makeup_conductivity` (float): Conductivity of makeup water in µS/cm
- `blowdown_conductivity` (float): Conductivity of blowdown water in µS/cm

**Returns:** `float` - Cycles of concentration (typically 3-10)

**Note:** Higher cycles = better water efficiency but increased scale/corrosion risk.

---

### 2.2 `calculate_blowdown_rate()`

Calculates optimal blowdown rate based on cycles and steam demand.

**Formula:**
```
Blowdown Rate = Makeup Rate / (Cycles - 1)
or
Blowdown Rate = Steam Rate / (Cycles - 1)
```

**Parameters:**
- `steam_rate` (float): Steam generation rate in kg/hr
- `cycles` (float): Cycles of concentration
- `makeup_rate` (float, optional): Makeup water rate in kg/hr

**Returns:** `float` - Blowdown rate in kg/hr

---

### 2.3 `calculate_blowdown_heat_loss()`

Calculates heat energy loss due to blowdown.

**Formula:**
```
Heat Loss = Blowdown Rate × Cp × ΔT
Cp = 4.186 kJ/(kg·°C)
```

**Parameters:**
- `blowdown_rate` (float): Blowdown rate in kg/hr
- `temperature` (float): Blowdown temperature in Celsius
- `ambient_temp` (float, optional): Ambient temperature (default: 25°C)

**Returns:** `float` - Heat loss in kW

---

### 2.4 `optimize_blowdown_schedule()`

Comprehensive blowdown optimization considering water quality limits and economics.

**Parameters:**
- `water_data` (Dict): Dictionary containing:
  - `makeup_conductivity`: µS/cm
  - `blowdown_conductivity`: µS/cm
  - `tds`: mg/L
  - `alkalinity`: mg/L
  - `temperature`: °C
  - `pressure`: bar
  - `water_cost`: $/m3
  - `energy_cost`: $/kWh
- `steam_demand` (float): Steam generation rate in kg/hr

**Returns:** `BlowdownOptimization` dataclass with:
- `optimal_cycles`: Recommended cycles of concentration
- `recommended_blowdown_rate`: Blowdown rate in kg/hr
- `continuous_blowdown_percent`: Percentage for continuous blowdown
- `bottom_blowdown_frequency`: "hourly", "every_4h", "every_8h", "daily"
- `heat_recovery_potential`: Recoverable heat in kW
- `water_savings`: Water saved vs. baseline in m3/day
- `cost_savings`: Daily cost savings in $/day
- `energy_loss`: Heat loss in kW
- `tds_control`: TDS control parameters
- `timestamp`: UTC ISO timestamp
- `provenance_hash`: SHA-256 hash

---

## 3. Chemical Dosing

### 3.1 `calculate_phosphate_dosing()`

Calculates phosphate dosing for scale and corrosion control.

**Parameters:**
- `residual_target` (float): Target phosphate residual in ppm as PO4 (typically 30-60 ppm)
- `volume` (float): Boiler water volume in m3
- `current_level` (float, optional): Current phosphate level in ppm (default: 0)
- `steam_rate` (float, optional): Steam rate for continuous dosing in kg/hr

**Returns:** `float` - Dosing rate in kg/day (or kg for shock dosing)

**Note:** Phosphate maintains alkaline pH and precipitates residual hardness.

---

### 3.2 `calculate_oxygen_scavenger_dosing()`

Calculates oxygen scavenger dosing to remove dissolved oxygen.

**Stoichiometric Ratios (kg scavenger per kg O2):**
- Sodium sulfite: 7.88
- Sodium bisulfite: 6.67
- Hydrazine: 1.0
- DEHA: 1.4
- Carbohydrazide: 1.1

**Parameters:**
- `dissolved_oxygen` (float): DO in makeup water in ppb
- `steam_rate` (float): Steam generation rate in kg/hr
- `scavenger_type` (ScavengerType): Type of scavenger (default: SODIUM_SULFITE)

**Returns:** `float` - Dosing rate in kg/day (includes 15% excess)

---

### 3.3 `calculate_amine_dosing()`

Calculates amine dosing for condensate pH control and corrosion protection.

**Typical Dosing:**
- Neutralizing amines: 0.5-2.0 ppm (maintain pH 8.0-9.0)
- Filming amines: 0.1-0.5 ppm (protective film formation)

**Parameters:**
- `condensate_pH_target` (float): Target condensate pH (typically 8.5-9.0)
- `steam_rate` (float): Steam generation rate in kg/hr
- `amine_type` (AmineType): Type of amine (default: NEUTRALIZING_AMINE)
- `condensate_return_percent` (float, optional): Condensate return % (default: 80%)

**Returns:** `float` - Dosing rate in kg/day

---

### 3.4 `calculate_polymer_dosing()`

Calculates polymer dosing for sludge conditioning and dispersion.

**Typical Dosing:** 5-20 ppm based on hardness and fouling tendency

**Parameters:**
- `sludge_conditioner_need` (float): Sludge conditioning need factor (0-100)
- `water_hardness` (float): Total hardness in mg/L as CaCO3
- `steam_rate` (float, optional): Steam rate in kg/hr (default: 1000)

**Returns:** `float` - Dosing rate in kg/day

---

### 3.5 `optimize_chemical_consumption()`

Comprehensive chemical consumption optimization for cost and effectiveness.

**Parameters:**
- `current_usage` (Dict): Current chemical usage rates in kg/day
  - `phosphate`, `oxygen_scavenger`, `amine`, `polymer`
- `water_quality` (Dict): Current water quality parameters
- `targets` (Dict): Target parameters for optimization

**Returns:** `ChemicalOptimization` dataclass with:
- `phosphate_dosing`: Optimized phosphate dosing in kg/day
- `oxygen_scavenger_dosing`: Optimized scavenger dosing in kg/day
- `amine_dosing`: Optimized amine dosing in kg/day
- `polymer_dosing`: Optimized polymer dosing in kg/day
- `total_chemical_cost`: Total daily cost in $/day
- `cost_reduction_potential`: Monthly savings in $/month
- `feed_schedule`: Detailed feed schedule for each chemical
- `residual_targets`: Target residuals for monitoring
- `optimization_score`: Score 0-100
- `timestamp`: UTC ISO timestamp
- `provenance_hash`: SHA-256 hash

---

## 4. Scale and Corrosion Prediction

### 4.1 `predict_calcium_carbonate_scale()`

Predicts calcium carbonate scale formation rate.

**Parameters:**
- `lsi` (float): Langelier Saturation Index
- `temperature` (float): Water temperature in Celsius
- `velocity` (float, optional): Water velocity in m/s (default: 1.0)

**Returns:** `float` - Scale formation rate in mm/year

**Note:** Scale rate increases with positive LSI, temperature, and low velocity.

---

### 4.2 `predict_silica_scale()`

Predicts silica scale risk level based on concentration, temperature, and pH.

**Parameters:**
- `silica_concentration` (float): Silica concentration in mg/L as SiO2
- `temperature` (float): Water temperature in Celsius
- `pH` (float): pH of water

**Returns:** `str` - Risk level: "low", "moderate", "high", or "severe"

**Note:** High risk at >150°C and pH >9 (hard, glassy deposits).

---

### 4.3 `predict_oxygen_corrosion()`

Predicts oxygen corrosion rate in carbon steel.

**Parameters:**
- `dissolved_oxygen` (float): DO in ppb
- `temperature` (float): Temperature in Celsius
- `pH` (float): pH of water

**Returns:** `float` - Corrosion rate in mils per year (mpy)

**Note:** Peak corrosion typically at 60-80°C.

---

### 4.4 `predict_acid_corrosion()`

Predicts acid corrosion rate in carbon steel.

**Parameters:**
- `pH` (float): pH of water
- `temperature` (float): Temperature in Celsius

**Returns:** `float` - Corrosion rate in mpy

**Note:** Significant corrosion below pH 7, accelerates below pH 5.

---

### 4.5 `calculate_corrosion_allowance()`

Calculates required corrosion allowance for equipment design.

**Parameters:**
- `material` (str): Material type ("carbon_steel", "stainless_steel", "copper_alloy")
- `environment` (str): Environment ("boiler_water", "condensate", "feedwater", "raw_water")
- `service_life` (float): Design service life in years

**Returns:** `float` - Required corrosion allowance in mm (includes 2× safety factor)

---

## 5. Energy and Cost Analysis

### 5.1 `calculate_blowdown_energy_savings()`

Calculates annual energy savings from improved cycles of concentration.

**Parameters:**
- `before_cycles` (float): Cycles before optimization
- `after_cycles` (float): Cycles after optimization
- `steam_cost` (float): Cost of steam in $/ton
- `steam_rate` (float, optional): Steam rate in kg/hr (default: 1000)

**Returns:** `float` - Annual savings in $/year

---

### 5.2 `calculate_chemical_cost()`

Calculates daily chemical cost.

**Parameters:**
- `dosing_rates` (Dict[str, float]): Chemical dosing rates in kg/day
- `chemical_prices` (Dict[str, float]): Chemical unit prices in $/kg

**Returns:** `float` - Total daily cost in $/day

---

### 5.3 `calculate_water_treatment_roi()`

Calculates ROI for water treatment optimization projects.

**Formula:**
```
ROI = (Annual Savings - Annual Costs) / Implementation Cost × 100%
```

**Parameters:**
- `costs` (Dict[str, float]): Annual operating costs
- `savings` (Dict[str, float]): Annual savings
- `implementation_cost` (float): One-time implementation cost

**Returns:** `float` - ROI percentage

**Note:** Also calculates payback period internally.

---

### 5.4 `calculate_makeup_water_cost()`

Calculates daily makeup water cost including treatment.

**Parameters:**
- `usage` (float): Makeup water usage in m3/day
- `water_price` (float): Raw water price in $/m3
- `treatment_cost` (float, optional): Water treatment cost in $/m3 (default: 0.5)

**Returns:** `float` - Total daily cost in $/day

---

## 6. Compliance Checking

### 6.1 `check_asme_compliance()`

Checks ASME boiler water quality compliance.

**ASME Pressure Ranges:**
- 0-20 bar: Less stringent limits
- 20-60 bar: Moderate limits
- >60 bar: Most stringent limits

**Parameters:**
- `water_chemistry` (Dict): Water chemistry parameters
  - `pH`, `tds`, `alkalinity`, `chloride`, `silica`, `hardness`
- `pressure` (float): Boiler operating pressure in bar

**Returns:** `ComplianceResult` dataclass with:
- `standard`: "ASME"
- `compliance_status`: "PASS", "WARNING", or "FAIL"
- `parameters_checked`: Number of parameters checked
- `violations`: List of violations
- `warnings`: List of warnings
- `margin_percent`: Compliance margin percentage
- `recommended_actions`: List of recommended actions
- `timestamp`: UTC ISO timestamp
- `provenance_hash`: SHA-256 hash

---

### 6.2 `check_abma_guidelines()`

Checks ABMA (American Boiler Manufacturers Association) guidelines compliance.

**Parameters:**
- `water_chemistry` (Dict): Water chemistry parameters
  - `pH`, `tds`, `phosphate_residual`, `sulfite_residual`
- `boiler_type` (BoilerType): Type of boiler (FIRE_TUBE, WATER_TUBE, etc.)

**Returns:** `ComplianceResult` dataclass (same structure as ASME)

**ABMA Focus:** Internal treatment residuals (phosphate, sulfite) and pH control.

---

### 6.3 `validate_treatment_program()`

Validates water treatment program effectiveness and compatibility.

**Supported Programs:**
- **Phosphate:** Traditional coordinated phosphate/pH program
- **All-Volatile Treatment (AVT):** For high-pressure boilers
- **Chelant:** Chelant-based treatment
- **Polymer:** Polymer dispersant programs

**Parameters:**
- `program_type` (str): Type of treatment program
- `chemistry` (Dict): Current water chemistry

**Returns:** `ValidationResult` dataclass with:
- `program_type`: Program type validated
- `is_valid`: Boolean validation result
- `effectiveness_score`: Score 0-100
- `chemistry_compatibility`: Boolean compatibility
- `issues`: List of issues found
- `recommendations`: List of recommendations
- `timestamp`: UTC ISO timestamp
- `provenance_hash`: SHA-256 hash

---

## Data Classes

All functions return dataclasses with comprehensive results:

### Enumerations
- `ScavengerType`: SODIUM_SULFITE, SODIUM_BISULFITE, HYDRAZINE, DEHA, CARBOHYDRAZIDE
- `AmineType`: CYCLOHEXYLAMINE, MORPHOLINE, NEUTRALIZING_AMINE, FILMING_AMINE
- `BoilerType`: FIRE_TUBE, WATER_TUBE, PACKAGE, INDUSTRIAL, UTILITY

### Result Classes
- `WaterQualityAnalysis`
- `BlowdownOptimization`
- `ChemicalOptimization`
- `ComplianceResult`
- `ValidationResult`

All result classes include:
- Relevant calculation results
- `timestamp`: UTC ISO format
- `provenance_hash`: SHA-256 hash for auditability

---

## Usage Examples

### Example 1: Water Quality Analysis

```python
from tools import WaterTreatmentTools

chemistry_data = {
    'pH': 11.2,
    'temperature': 180.0,
    'calcium_hardness': 1.5,
    'alkalinity': 400.0,
    'tds': 2500.0,
    'chloride': 80.0,
    'sulfate': 50.0,
    'pressure': 15.0
}

analysis = WaterTreatmentTools.analyze_water_quality(chemistry_data)

print(f"LSI: {analysis.lsi_value}")
print(f"Scale Tendency: {analysis.scale_tendency}")
print(f"Compliance: {analysis.compliance_status}")
for rec in analysis.recommendations:
    print(f"  - {rec}")
```

### Example 2: Blowdown Optimization

```python
water_data = {
    'makeup_conductivity': 200.0,
    'blowdown_conductivity': 1500.0,
    'tds': 2000.0,
    'alkalinity': 400.0,
    'temperature': 180.0,
    'pressure': 12.0,
    'water_cost': 0.5,
    'energy_cost': 0.08
}

optimization = WaterTreatmentTools.optimize_blowdown_schedule(
    water_data=water_data,
    steam_demand=5000.0
)

print(f"Optimal Cycles: {optimization.optimal_cycles}")
print(f"Cost Savings: ${optimization.cost_savings}/day")
print(f"Water Savings: {optimization.water_savings} m3/day")
```

### Example 3: Chemical Dosing

```python
from tools import ScavengerType, AmineType

# Oxygen scavenger
scavenger_dose = WaterTreatmentTools.calculate_oxygen_scavenger_dosing(
    dissolved_oxygen=200.0,  # ppb
    steam_rate=5000.0,       # kg/hr
    scavenger_type=ScavengerType.SODIUM_SULFITE
)

# Amine for condensate protection
amine_dose = WaterTreatmentTools.calculate_amine_dosing(
    condensate_pH_target=8.8,
    steam_rate=5000.0,
    amine_type=AmineType.NEUTRALIZING_AMINE,
    condensate_return_percent=80.0
)

print(f"Scavenger: {scavenger_dose} kg/day")
print(f"Amine: {amine_dose} kg/day")
```

### Example 4: Compliance Checking

```python
from tools import BoilerType

chemistry = {
    'pH': 11.0,
    'tds': 2000.0,
    'alkalinity': 450.0,
    'chloride': 80.0,
    'silica': 40.0,
    'hardness': 1.5,
    'phosphate_residual': 45.0,
    'sulfite_residual': 25.0
}

# ASME compliance
asme = WaterTreatmentTools.check_asme_compliance(chemistry, pressure=15.0)

# ABMA compliance
abma = WaterTreatmentTools.check_abma_guidelines(
    chemistry,
    BoilerType.WATER_TUBE
)

print(f"ASME: {asme.compliance_status}")
print(f"ABMA: {abma.compliance_status}")
```

### Example 5: Provenance Tracking

```python
# All results include provenance hash for auditability
analysis = WaterTreatmentTools.analyze_water_quality(chemistry_data)

print(f"Timestamp: {analysis.timestamp}")
print(f"Provenance Hash: {analysis.provenance_hash}")

# Store in database for audit trail
# db.store_analysis(analysis.provenance_hash, analysis)
```

---

## Thread Safety

All static methods in `WaterTreatmentTools` are **thread-safe** using `_calculation_lock` for critical calculations. Safe for use in multi-threaded environments.

---

## Error Handling

All functions include comprehensive error handling:
- Input validation with descriptive error messages
- Graceful handling of edge cases (e.g., zero division, out-of-range values)
- Logging for debugging and audit trails
- Exceptions propagated with context

---

## Dependencies

```python
import hashlib       # SHA-256 provenance hashing
import logging       # Error and debug logging
import math          # Mathematical functions
import threading     # Thread-safe locks
import numpy as np   # Numerical calculations
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
```

---

## References

**Standards and Guidelines:**
- ASME (American Society of Mechanical Engineers) Boiler & Pressure Vessel Code
- ABMA (American Boiler Manufacturers Association) Guidelines
- NACE (National Association of Corrosion Engineers) Standards
- Betz Handbook of Industrial Water Conditioning
- NALCO Water Handbook

**Formulas:**
- Langelier Saturation Index (LSI)
- Ryznar Stability Index (RSI)
- Puckorius Scaling Index (PSI)
- Larson-Skold Index (LSK)
- Standard thermodynamic calculations

---

## Certification

This module is part of **GL-016 WATERGUARD** certified agent system.

**Certification Status:** Ready for certification audit
**Determinism Level:** 100% (zero LLM involvement)
**Test Coverage:** Verification script provided
**Audit Trail:** All functions include provenance hashing

---

## Support

For questions or issues:
- Review inline documentation and docstrings
- Run verification script: `python test_tools_verification.py`
- Check logs for detailed error messages
- Refer to ASME/ABMA standards for formula validation

---

**Last Updated:** 2025-12-02
**Version:** 1.0.0
**Author:** DETERMINISTIC TOOLS ENGINEERING TEAM
