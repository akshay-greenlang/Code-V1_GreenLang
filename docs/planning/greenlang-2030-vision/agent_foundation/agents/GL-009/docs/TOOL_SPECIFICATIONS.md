# GL-009 THERMALIQ Tool Specifications

**Agent:** GL-009 THERMALIQ (ThermalEfficiencyCalculator)
**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [Tool Architecture](#2-tool-architecture)
3. [First Law Efficiency Calculator](#3-first-law-efficiency-calculator)
4. [Second Law (Exergy) Efficiency Calculator](#4-second-law-exergy-efficiency-calculator)
5. [Combustion Efficiency Calculator](#5-combustion-efficiency-calculator)
6. [Radiation Loss Calculator](#6-radiation-loss-calculator)
7. [Convection Loss Calculator](#7-convection-loss-calculator)
8. [Flue Gas Loss Calculator](#8-flue-gas-loss-calculator)
9. [Heat Balance Calculator](#9-heat-balance-calculator)
10. [Sankey Diagram Generator](#10-sankey-diagram-generator)
11. [Improvement Opportunity Identifier](#11-improvement-opportunity-identifier)
12. [Efficiency Benchmarker](#12-efficiency-benchmarker)
13. [Exergy Flow Calculator](#13-exergy-flow-calculator)
14. [Trend Analyzer](#14-trend-analyzer)
15. [Constants and Reference Data](#15-constants-and-reference-data)
16. [Standards Compliance](#16-standards-compliance)

---

## 1. Overview

### 1.1 Purpose

GL-009 THERMALIQ provides **deterministic, zero-hallucination** thermal efficiency calculations based on fundamental thermodynamic principles. All numeric results are derived from physics formulas, never from AI generation.

### 1.2 Key Characteristics

- ✅ **100% Deterministic:** temperature=0.0, seed=42, no randomness
- ✅ **Physics-Based:** All calculations use validated formulas
- ✅ **Standards-Compliant:** ASME PTC 4.1, ISO 50001:2018, EPA regulations
- ✅ **Provenance Tracking:** Every result includes audit trail
- ✅ **Multi-Law Analysis:** Both First Law (energy) and Second Law (exergy)

### 1.3 Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| **Efficiency Calculations** | 3 tools | First Law, Second Law, Combustion |
| **Loss Analysis** | 3 tools | Radiation, Convection, Flue Gas |
| **Energy Balance** | 1 tool | Heat Balance with closure verification |
| **Visualization** | 1 tool | Sankey diagram generation |
| **Optimization** | 1 tool | Improvement opportunity identification |
| **Benchmarking** | 1 tool | Industry comparison |
| **Advanced Analysis** | 2 tools | Exergy flows, Trend analysis |

**Total Tools:** 12 deterministic calculation tools

---

## 2. Tool Architecture

### 2.1 Determinism Guarantee

Every tool in GL-009 follows this architecture:

```python
@deterministic(temperature=0.0, seed=42)
def calculate_tool(inputs: ToolInput) -> ToolOutput:
    """
    Physics-based calculation.
    NO AI generation of numeric results.
    """
    # 1. Validate inputs (schema validation)
    validated = validate_schema(inputs)

    # 2. Apply deterministic physics formula
    result = physics_formula(validated)

    # 3. Generate provenance hash
    provenance = generate_provenance_hash(inputs, result)

    # 4. Return typed result with audit trail
    return ToolOutput(result=result, provenance=provenance)
```

### 2.2 Provenance Tracking

Every calculation result includes:

```json
{
  "result": <numeric_value>,
  "provenance_hash": "sha256:abc123...",
  "timestamp": "2025-11-26T10:30:00Z",
  "formula_used": "eta_1 = Q_useful / Q_input",
  "inputs_hash": "sha256:def456...",
  "standards_basis": ["ASME PTC 4.1", "ISO 50001:2018"]
}
```

### 2.3 Error Handling

All tools follow consistent error handling:

```python
try:
    result = calculate(inputs)
except ValidationError as e:
    return ErrorResult(
        error_type="validation_error",
        message=str(e),
        recoverable=True
    )
except PhysicsConstraintViolation as e:
    return ErrorResult(
        error_type="physics_constraint_violation",
        message=str(e),
        recoverable=False
    )
```

---

## 3. First Law Efficiency Calculator

### 3.1 Tool ID
`calculate_first_law_efficiency`

### 3.2 Description
Calculates thermal efficiency using the First Law of Thermodynamics (conservation of energy). This is the primary efficiency metric for thermal systems.

### 3.3 Physics Basis

**Law:** First Law of Thermodynamics - Energy cannot be created or destroyed, only transformed.

**Formula:**
```
η₁ = (Q_useful / Q_input) × 100%

Where:
η₁ = First Law efficiency (%)
Q_useful = Useful heat output (kW or MW)
Q_input = Total energy input (kW or MW)
```

**Alternative (Heat Loss Method):**
```
η₁ = 100% - (ΣQ_losses / Q_input) × 100%
```

### 3.4 Standards Compliance
- **ASME PTC 4.1** - Steam Generating Units Performance Test Codes
- **ASME PTC 4** - Fired Steam Generators
- **ISO 50001:2018** - Energy Management Systems (EnPI calculations)

### 3.5 Input Schema

```json
{
  "type": "object",
  "required": ["total_energy_input_mw", "useful_heat_output_mw"],
  "properties": {
    "total_energy_input_mw": {
      "type": "number",
      "minimum": 0,
      "description": "Total energy input from all sources (fuel + electrical + preheated air)",
      "unit": "MW"
    },
    "useful_heat_output_mw": {
      "type": "number",
      "minimum": 0,
      "description": "Useful heat transferred to process or product",
      "unit": "MW"
    },
    "calculation_method": {
      "type": "string",
      "enum": ["direct", "indirect", "heat_loss"],
      "default": "direct",
      "description": "Calculation methodology"
    },
    "ambient_temperature_c": {
      "type": "number",
      "default": 25.0,
      "description": "Reference temperature for heat calculations",
      "unit": "°C"
    }
  }
}
```

### 3.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "first_law_efficiency_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 100,
      "description": "Overall thermal efficiency based on energy balance"
    },
    "energy_input_mw": {
      "type": "number",
      "description": "Total energy input used in calculation"
    },
    "useful_output_mw": {
      "type": "number",
      "description": "Useful heat output used in calculation"
    },
    "total_losses_mw": {
      "type": "number",
      "description": "Calculated total losses (input - output)"
    },
    "combustion_efficiency_percent": {
      "type": "number",
      "description": "Fuel-to-heat conversion efficiency"
    },
    "gross_efficiency_percent": {
      "type": "number",
      "description": "Efficiency including auxiliary inputs"
    },
    "net_efficiency_percent": {
      "type": "number",
      "description": "Efficiency excluding auxiliary inputs"
    },
    "calculation_method": {
      "type": "string",
      "description": "Method used for this calculation"
    },
    "provenance": {
      "type": "object",
      "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "formula_used": {"type": "string"},
        "inputs_hash": {"type": "string"},
        "result_hash": {"type": "string"},
        "standards_basis": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

### 3.7 Example Usage

**Request:**
```json
{
  "total_energy_input_mw": 50.0,
  "useful_heat_output_mw": 41.5,
  "calculation_method": "direct",
  "ambient_temperature_c": 25.0
}
```

**Response:**
```json
{
  "first_law_efficiency_percent": 83.0,
  "energy_input_mw": 50.0,
  "useful_output_mw": 41.5,
  "total_losses_mw": 8.5,
  "combustion_efficiency_percent": 85.2,
  "gross_efficiency_percent": 83.0,
  "net_efficiency_percent": 81.5,
  "calculation_method": "direct",
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "eta_1 = (Q_useful / Q_input) * 100%",
    "inputs_hash": "sha256:abc123...",
    "result_hash": "sha256:def456...",
    "standards_basis": ["ASME PTC 4.1", "ISO 50001:2018"]
  }
}
```

### 3.8 Validation Rules

1. **Energy input must be positive:** `total_energy_input_mw > 0`
2. **Output cannot exceed input:** `useful_heat_output_mw ≤ total_energy_input_mw`
3. **Efficiency range:** `0% ≤ η₁ ≤ 100%` (theoretical max ~95% for boilers)
4. **Physical sanity check:** Warn if efficiency > 95% (likely measurement error)

### 3.9 Typical Values by Process Type

| Process Type | Typical η₁ Range | Industry Average |
|--------------|------------------|------------------|
| Boiler (natural gas) | 75-94% | 82% |
| Industrial furnace | 50-85% | 65% |
| Process dryer | 35-70% | 45% |
| Rotary kiln | 30-75% | 50% |
| Heat exchanger | 70-95% | 85% |

---

## 4. Second Law (Exergy) Efficiency Calculator

### 4.1 Tool ID
`calculate_second_law_efficiency`

### 4.2 Description
Calculates exergy (availability) efficiency using the Second Law of Thermodynamics. This accounts for the **quality** of energy, not just quantity.

### 4.3 Physics Basis

**Law:** Second Law of Thermodynamics - Entropy increases in irreversible processes; not all energy is equally useful.

**Formula:**
```
η₂ = (Ex_useful / Ex_input) × 100%

Where:
η₂ = Second Law (exergy) efficiency (%)
Ex_useful = Useful exergy output (kW)
Ex_input = Total exergy input (kW)

Ex = m × [(h - h₀) - T₀ × (s - s₀)]

Where:
Ex = Exergy (kJ/kg)
h = Specific enthalpy (kJ/kg)
h₀ = Dead state enthalpy (kJ/kg)
s = Specific entropy (kJ/kg-K)
s₀ = Dead state entropy (kJ/kg-K)
T₀ = Dead state temperature (K, typically 298.15 K)
```

### 4.4 Standards Compliance
- **Second Law of Thermodynamics** (fundamental physics)
- **Exergy Analysis Methodology** (Kotas, 2012)
- **ISO 50001:2018** - Energy Management Systems (advanced analysis)

### 4.5 Input Schema

```json
{
  "type": "object",
  "required": ["exergy_input_mw", "useful_exergy_output_mw"],
  "properties": {
    "exergy_input_mw": {
      "type": "number",
      "minimum": 0,
      "description": "Total exergy input (accounting for temperature/pressure quality)",
      "unit": "MW"
    },
    "useful_exergy_output_mw": {
      "type": "number",
      "minimum": 0,
      "description": "Useful exergy transferred to process",
      "unit": "MW"
    },
    "reference_temperature_k": {
      "type": "number",
      "default": 298.15,
      "description": "Dead state temperature (typically 25°C = 298.15 K)",
      "unit": "K"
    },
    "reference_pressure_bar": {
      "type": "number",
      "default": 1.01325,
      "description": "Dead state pressure (typically atmospheric)",
      "unit": "bar"
    },
    "process_streams": {
      "type": "array",
      "description": "Detailed thermodynamic states for stream-by-stream exergy calculation",
      "items": {
        "type": "object",
        "properties": {
          "stream_name": {"type": "string"},
          "mass_flow_kg_hr": {"type": "number"},
          "temperature_c": {"type": "number"},
          "pressure_bar": {"type": "number"},
          "enthalpy_kj_kg": {"type": "number"},
          "entropy_kj_kg_k": {"type": "number"}
        }
      }
    }
  }
}
```

### 4.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "second_law_efficiency_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 100,
      "description": "Exergy efficiency (quality-weighted)"
    },
    "exergy_input_mw": {
      "type": "number",
      "description": "Total exergy input"
    },
    "exergy_output_mw": {
      "type": "number",
      "description": "Useful exergy output"
    },
    "exergy_destruction_mw": {
      "type": "number",
      "description": "Exergy destroyed due to irreversibilities"
    },
    "exergy_destruction_percent": {
      "type": "number",
      "description": "Percentage of input exergy destroyed"
    },
    "improvement_potential_mw": {
      "type": "number",
      "description": "Theoretical maximum recoverable exergy"
    },
    "carnot_efficiency_percent": {
      "type": "number",
      "description": "Carnot efficiency limit for temperature levels"
    },
    "quality_factor": {
      "type": "number",
      "description": "Ex/Q ratio showing energy quality"
    },
    "irreversibility_breakdown": {
      "type": "array",
      "description": "Exergy destruction by component/process",
      "items": {
        "type": "object",
        "properties": {
          "component": {"type": "string"},
          "destruction_mw": {"type": "number"},
          "destruction_percent": {"type": "number"},
          "root_cause": {"type": "string"}
        }
      }
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 4.7 Example Usage

**Request:**
```json
{
  "exergy_input_mw": 50.0,
  "useful_exergy_output_mw": 35.0,
  "reference_temperature_k": 298.15,
  "reference_pressure_bar": 1.01325
}
```

**Response:**
```json
{
  "second_law_efficiency_percent": 70.0,
  "exergy_input_mw": 50.0,
  "exergy_output_mw": 35.0,
  "exergy_destruction_mw": 15.0,
  "exergy_destruction_percent": 30.0,
  "improvement_potential_mw": 12.0,
  "carnot_efficiency_percent": 75.0,
  "quality_factor": 0.85,
  "irreversibility_breakdown": [
    {
      "component": "combustion_chamber",
      "destruction_mw": 8.0,
      "destruction_percent": 16.0,
      "root_cause": "finite_temperature_difference"
    },
    {
      "component": "heat_exchanger",
      "destruction_mw": 5.0,
      "destruction_percent": 10.0,
      "root_cause": "temperature_approach"
    },
    {
      "component": "stack_losses",
      "destruction_mw": 2.0,
      "destruction_percent": 4.0,
      "root_cause": "exhaust_temperature"
    }
  ],
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "eta_2 = (Ex_useful / Ex_input) * 100%",
    "standards_basis": ["Second Law of Thermodynamics", "ISO 50001:2018"]
  }
}
```

### 4.8 Relationship Between First and Second Law Efficiencies

```
η₂ ≤ η₁ (always)

η₂ accounts for energy QUALITY
η₁ accounts for energy QUANTITY

Example:
- η₁ = 83% (energy efficiency)
- η₂ = 70% (exergy efficiency)
- Gap = 13% represents thermodynamic irreversibilities
```

### 4.9 Typical Values

| Process Type | Typical η₂ Range | Typical η₁ Range | η₂/η₁ Ratio |
|--------------|------------------|------------------|-------------|
| Boiler | 50-70% | 75-94% | 0.67-0.85 |
| Furnace | 35-65% | 50-85% | 0.70-0.76 |
| Heat exchanger | 60-85% | 70-95% | 0.85-0.90 |
| Cogeneration | 40-60% | 55-80% | 0.70-0.75 |

---

## 5. Combustion Efficiency Calculator

### 5.1 Tool ID
`calculate_combustion_efficiency`

### 5.2 Description
Calculates fuel combustion efficiency from flue gas analysis using the Siegert formula and heat loss method.

### 5.3 Physics Basis

**Siegert Formula:**
```
η_comb = 100% - L_dry - L_moisture - L_unburned

L_dry = k₁ × (T_flue - T_air) / CO₂%

Where:
η_comb = Combustion efficiency (%)
L_dry = Dry flue gas loss (%)
L_moisture = Moisture loss (%)
L_unburned = Unburned fuel loss (%)
k₁ = Fuel-specific constant
T_flue = Flue gas temperature (°C)
T_air = Combustion air temperature (°C)
CO₂% = Carbon dioxide concentration (%)
```

**Siegert Constants (k₁):**
- Natural gas: k₁ = 0.37
- Fuel oil No. 2: k₁ = 0.44
- Fuel oil No. 6: k₁ = 0.50
- Coal: k₁ = 0.63
- Propane: k₁ = 0.40
- Biomass: k₁ = 0.55

### 5.4 Standards Compliance
- **ASME PTC 4.1** - Combustion efficiency methodology
- **EPA 40 CFR Part 60** - Flue gas analysis procedures
- **EN 12952** - European boiler efficiency standards

### 5.5 Input Schema

```json
{
  "type": "object",
  "required": ["fuel_type", "flue_gas_temperature_c", "ambient_temperature_c", "o2_percent"],
  "properties": {
    "fuel_type": {
      "type": "string",
      "enum": [
        "natural_gas",
        "fuel_oil_no2",
        "fuel_oil_no6",
        "coal_bituminous",
        "coal_sub_bituminous",
        "propane",
        "biomass"
      ],
      "description": "Type of fuel burned"
    },
    "flue_gas_temperature_c": {
      "type": "number",
      "minimum": 25,
      "maximum": 1500,
      "description": "Stack/exhaust gas temperature",
      "unit": "°C"
    },
    "ambient_temperature_c": {
      "type": "number",
      "default": 25.0,
      "description": "Combustion air inlet temperature",
      "unit": "°C"
    },
    "o2_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 21,
      "description": "Oxygen concentration in flue gas (dry basis)",
      "unit": "%"
    },
    "co2_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 20,
      "description": "Carbon dioxide concentration in flue gas (dry basis)",
      "unit": "%"
    },
    "co_ppm": {
      "type": "number",
      "minimum": 0,
      "default": 0,
      "description": "Carbon monoxide concentration (indicator of incomplete combustion)",
      "unit": "ppm"
    },
    "excess_air_percent": {
      "type": "number",
      "minimum": 0,
      "description": "Excess air percentage (calculated from O₂ if not provided)",
      "unit": "%"
    },
    "fuel_moisture_percent": {
      "type": "number",
      "minimum": 0,
      "maximum": 100,
      "default": 0,
      "description": "Moisture content in fuel (as-fired basis)",
      "unit": "%"
    }
  }
}
```

### 5.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "combustion_efficiency_percent": {
      "type": "number",
      "description": "Overall combustion efficiency"
    },
    "excess_air_percent": {
      "type": "number",
      "description": "Calculated or input excess air"
    },
    "dry_flue_gas_loss_percent": {
      "type": "number",
      "description": "Sensible heat loss in dry flue gas"
    },
    "moisture_loss_percent": {
      "type": "number",
      "description": "Latent heat loss from moisture"
    },
    "h2_in_fuel_loss_percent": {
      "type": "number",
      "description": "Loss from hydrogen combustion (forms H₂O)"
    },
    "moisture_in_air_loss_percent": {
      "type": "number",
      "description": "Loss from moisture in combustion air"
    },
    "unburned_fuel_loss_percent": {
      "type": "number",
      "description": "Loss from incomplete combustion (from CO)"
    },
    "radiation_convection_loss_percent": {
      "type": "number",
      "description": "Radiation and convection losses (typically 1-2%)"
    },
    "total_losses_percent": {
      "type": "number",
      "description": "Sum of all losses"
    },
    "optimal_excess_air_percent": {
      "type": "number",
      "description": "Recommended excess air for this fuel type"
    },
    "efficiency_potential_percent": {
      "type": "number",
      "description": "Potential efficiency if optimized"
    },
    "recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "recommendation": {"type": "string"},
          "potential_gain_percent": {"type": "number"},
          "priority": {"type": "string", "enum": ["high", "medium", "low"]}
        }
      }
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 5.7 Example Usage

**Request:**
```json
{
  "fuel_type": "natural_gas",
  "flue_gas_temperature_c": 180.0,
  "ambient_temperature_c": 25.0,
  "o2_percent": 4.0,
  "co2_percent": 9.5,
  "co_ppm": 50,
  "fuel_moisture_percent": 0
}
```

**Response:**
```json
{
  "combustion_efficiency_percent": 84.5,
  "excess_air_percent": 22.0,
  "dry_flue_gas_loss_percent": 8.5,
  "moisture_loss_percent": 6.0,
  "h2_in_fuel_loss_percent": 0.5,
  "moisture_in_air_loss_percent": 0.3,
  "unburned_fuel_loss_percent": 0.1,
  "radiation_convection_loss_percent": 0.1,
  "total_losses_percent": 15.5,
  "optimal_excess_air_percent": 15.0,
  "efficiency_potential_percent": 86.5,
  "recommendations": [
    {
      "recommendation": "Reduce excess air from 22% to 15% to minimize flue gas losses",
      "potential_gain_percent": 1.5,
      "priority": "high"
    },
    {
      "recommendation": "Install economizer to reduce stack temperature to 120°C",
      "potential_gain_percent": 3.0,
      "priority": "high"
    },
    {
      "recommendation": "Maintain CO below 50 ppm (currently at limit)",
      "potential_gain_percent": 0.1,
      "priority": "medium"
    }
  ],
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "Siegert formula: eta_comb = 100 - L_dry - L_moisture - L_unburned",
    "siegert_k1": 0.37,
    "standards_basis": ["ASME PTC 4.1", "EPA 40 CFR Part 60"]
  }
}
```

### 5.8 Optimal Excess Air by Fuel Type

| Fuel Type | Optimal Excess Air | Typical O₂% | Max CO (ppm) |
|-----------|-------------------|-------------|--------------|
| Natural gas | 10-15% | 2-3% | 100 |
| Fuel oil No. 2 | 15-20% | 2.5-4% | 100 |
| Fuel oil No. 6 | 20-25% | 3-5% | 150 |
| Coal (pulverized) | 20-30% | 3.5-5.5% | 200 |
| Biomass | 25-40% | 4-7% | 300 |

---

## 6. Radiation Loss Calculator

### 6.1 Tool ID
`calculate_radiation_loss`

### 6.2 Description
Calculates heat loss from equipment surfaces due to thermal radiation using the Stefan-Boltzmann law.

### 6.3 Physics Basis

**Law:** Stefan-Boltzmann Law of Thermal Radiation

**Formula:**
```
Q_rad = ε × σ × A × (T_s⁴ - T_amb⁴)

Where:
Q_rad = Radiation heat loss (W)
ε = Surface emissivity (dimensionless, 0-1)
σ = Stefan-Boltzmann constant = 5.67 × 10⁻⁸ W/(m²·K⁴)
A = Surface area (m²)
T_s = Surface temperature (K)
T_amb = Ambient temperature (K)

Note: Temperatures MUST be in Kelvin (K = °C + 273.15)
```

### 6.4 Standards Compliance
- **Stefan-Boltzmann Law** (fundamental physics)
- **ASME PTC 4.1** - Surface radiation loss methodology
- **ISO 12241** - Thermal insulation for building equipment

### 6.5 Input Schema

```json
{
  "type": "object",
  "required": ["surface_area_m2", "surface_temperature_c", "ambient_temperature_c"],
  "properties": {
    "surface_area_m2": {
      "type": "number",
      "minimum": 0,
      "description": "Total radiating surface area",
      "unit": "m²"
    },
    "surface_temperature_c": {
      "type": "number",
      "description": "Surface temperature",
      "unit": "°C"
    },
    "ambient_temperature_c": {
      "type": "number",
      "default": 25.0,
      "description": "Surrounding ambient temperature",
      "unit": "°C"
    },
    "emissivity": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.9,
      "description": "Surface emissivity (0=perfect reflector, 1=black body)"
    },
    "view_factor": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 1.0,
      "description": "View factor accounting for geometry (typically 1.0 for large surfaces)"
    }
  }
}
```

### 6.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "radiation_loss_kw": {
      "type": "number",
      "description": "Total radiation heat loss in kW"
    },
    "radiation_loss_mw": {
      "type": "number",
      "description": "Total radiation heat loss in MW"
    },
    "heat_flux_w_m2": {
      "type": "number",
      "description": "Heat flux per unit area (W/m²)"
    },
    "surface_temperature_k": {
      "type": "number",
      "description": "Surface temperature converted to Kelvin"
    },
    "ambient_temperature_k": {
      "type": "number",
      "description": "Ambient temperature converted to Kelvin"
    },
    "emissivity_used": {
      "type": "number",
      "description": "Emissivity value used in calculation"
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 6.7 Example Usage

**Request:**
```json
{
  "surface_area_m2": 50.0,
  "surface_temperature_c": 60.0,
  "ambient_temperature_c": 25.0,
  "emissivity": 0.9,
  "view_factor": 1.0
}
```

**Response:**
```json
{
  "radiation_loss_kw": 8.35,
  "radiation_loss_mw": 0.00835,
  "heat_flux_w_m2": 167.0,
  "surface_temperature_k": 333.15,
  "ambient_temperature_k": 298.15,
  "emissivity_used": 0.9,
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "Q_rad = epsilon * sigma * A * (T_s^4 - T_amb^4)",
    "stefan_boltzmann_constant": 5.67e-8,
    "standards_basis": ["Stefan-Boltzmann Law", "ASME PTC 4.1"]
  }
}
```

### 6.8 Typical Emissivity Values

| Surface Material | Emissivity (ε) | Notes |
|------------------|----------------|-------|
| Oxidized steel | 0.80-0.95 | Typical for aged boilers |
| Polished aluminum | 0.04-0.06 | Reflective insulation |
| Painted surface (any color) | 0.90-0.95 | Most paints are black body emitters |
| Brick/concrete | 0.85-0.95 | Masonry materials |
| Insulation (mineral wool) | 0.90-0.95 | Most insulation materials |
| Stainless steel (polished) | 0.15-0.30 | Low emissivity |
| Galvanized steel (new) | 0.23-0.28 | Increases with oxidation |

---

## 7. Convection Loss Calculator

### 7.1 Tool ID
`calculate_convection_loss`

### 7.2 Description
Calculates heat loss from equipment surfaces due to natural or forced convection using Newton's Law of Cooling.

### 7.3 Physics Basis

**Law:** Newton's Law of Cooling

**Formula:**
```
Q_conv = h × A × (T_s - T_amb)

Where:
Q_conv = Convection heat loss (W)
h = Heat transfer coefficient (W/(m²·K))
A = Surface area (m²)
T_s = Surface temperature (°C or K)
T_amb = Ambient temperature (°C or K)
```

**Heat Transfer Coefficient Estimation:**
```
Natural convection (still air):
h = 5-25 W/(m²·K) depending on ΔT and orientation

Forced convection (wind):
h = 10 + 6×v^0.8
where v = wind speed (m/s)
```

### 7.4 Standards Compliance
- **Newton's Law of Cooling** (fundamental physics)
- **ASME PTC 4.1** - Convection loss methodology
- **ISO 12241** - Thermal insulation calculations

### 7.5 Input Schema

```json
{
  "type": "object",
  "required": ["surface_area_m2", "surface_temperature_c", "ambient_temperature_c"],
  "properties": {
    "surface_area_m2": {
      "type": "number",
      "minimum": 0,
      "description": "Total surface area exposed to convection",
      "unit": "m²"
    },
    "surface_temperature_c": {
      "type": "number",
      "description": "Surface temperature",
      "unit": "°C"
    },
    "ambient_temperature_c": {
      "type": "number",
      "default": 25.0,
      "description": "Ambient air temperature",
      "unit": "°C"
    },
    "heat_transfer_coefficient_w_m2_k": {
      "type": "number",
      "minimum": 0,
      "default": 10.0,
      "description": "Convection heat transfer coefficient (if known)",
      "unit": "W/(m²·K)"
    },
    "wind_speed_m_s": {
      "type": "number",
      "minimum": 0,
      "default": 0,
      "description": "Wind speed (for outdoor equipment, uses Churchill-Bernstein correlation)",
      "unit": "m/s"
    },
    "surface_orientation": {
      "type": "string",
      "enum": ["horizontal_upward", "horizontal_downward", "vertical", "tilted"],
      "default": "vertical",
      "description": "Surface orientation affects natural convection coefficient"
    }
  }
}
```

### 7.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "convection_loss_kw": {
      "type": "number",
      "description": "Total convection heat loss in kW"
    },
    "convection_loss_mw": {
      "type": "number",
      "description": "Total convection heat loss in MW"
    },
    "heat_flux_w_m2": {
      "type": "number",
      "description": "Convective heat flux per unit area"
    },
    "heat_transfer_coefficient_used": {
      "type": "number",
      "description": "h value used in calculation (W/(m²·K))"
    },
    "temperature_difference_k": {
      "type": "number",
      "description": "T_s - T_amb used in calculation"
    },
    "convection_type": {
      "type": "string",
      "enum": ["natural", "forced", "mixed"],
      "description": "Type of convection based on inputs"
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 7.7 Example Usage

**Request:**
```json
{
  "surface_area_m2": 50.0,
  "surface_temperature_c": 60.0,
  "ambient_temperature_c": 25.0,
  "heat_transfer_coefficient_w_m2_k": 10.0,
  "wind_speed_m_s": 0,
  "surface_orientation": "vertical"
}
```

**Response:**
```json
{
  "convection_loss_kw": 17.5,
  "convection_loss_mw": 0.0175,
  "heat_flux_w_m2": 350.0,
  "heat_transfer_coefficient_used": 10.0,
  "temperature_difference_k": 35.0,
  "convection_type": "natural",
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "Q_conv = h * A * (T_s - T_amb)",
    "standards_basis": ["Newton's Law of Cooling", "ASME PTC 4.1"]
  }
}
```

### 7.8 Typical Heat Transfer Coefficients

| Condition | h (W/(m²·K)) | Notes |
|-----------|--------------|-------|
| Natural convection, vertical surface, ΔT=10-50°C | 5-15 | Still air indoors |
| Natural convection, horizontal upward | 10-20 | Enhanced by buoyancy |
| Natural convection, horizontal downward | 3-8 | Reduced by buoyancy |
| Forced convection, v=1 m/s | 10-15 | Light wind |
| Forced convection, v=3 m/s | 20-30 | Moderate wind |
| Forced convection, v=5 m/s | 30-40 | Strong wind |
| Forced convection, v=10 m/s | 50-70 | Very strong wind |

---

## 8. Flue Gas Loss Calculator

### 8.1 Tool ID
`calculate_flue_gas_loss`

### 8.2 Description
Calculates sensible heat loss in exhaust/flue gas based on mass flow rate, temperature, and specific heat capacity.

### 8.3 Physics Basis

**Law:** Enthalpy balance on exhaust gas stream

**Formula:**
```
Q_flue = ṁ × Cp_avg × (T_exit - T_ref)

Where:
Q_flue = Flue gas heat loss (kW)
ṁ = Mass flow rate (kg/s)
Cp_avg = Average specific heat (kJ/(kg·K))
T_exit = Stack/exit temperature (°C)
T_ref = Reference temperature (°C, typically 25°C)

Conversion: kg/hr to kg/s = divide by 3600
           kJ/s to kW = direct conversion (1 kJ/s = 1 kW)
```

**Specific Heat of Flue Gas:**
```
Cp_flue ≈ 1.05-1.15 kJ/(kg·K) depending on composition and temperature
Default: 1.10 kJ/(kg·K) for natural gas combustion products
```

### 8.4 Standards Compliance
- **ASME PTC 4.1** - Flue gas loss methodology
- **EPA 40 CFR Part 60** - Stack temperature monitoring
- **First Law of Thermodynamics** - Enthalpy balance

### 8.5 Input Schema

```json
{
  "type": "object",
  "required": ["mass_flow_kg_hr", "exit_temperature_c"],
  "properties": {
    "mass_flow_kg_hr": {
      "type": "number",
      "minimum": 0,
      "description": "Flue gas mass flow rate",
      "unit": "kg/hr"
    },
    "exit_temperature_c": {
      "type": "number",
      "minimum": 25,
      "maximum": 1500,
      "description": "Stack/exhaust gas temperature",
      "unit": "°C"
    },
    "reference_temperature_c": {
      "type": "number",
      "default": 25.0,
      "description": "Reference temperature (typically ambient)",
      "unit": "°C"
    },
    "specific_heat_kj_kg_k": {
      "type": "number",
      "default": 1.10,
      "description": "Average specific heat capacity of flue gas (if known)",
      "unit": "kJ/(kg·K)"
    },
    "fuel_input_mw": {
      "type": "number",
      "description": "Fuel energy input (for calculating loss percentage)",
      "unit": "MW"
    },
    "gas_composition": {
      "type": "object",
      "description": "Flue gas composition for accurate Cp calculation",
      "properties": {
        "co2_percent": {"type": "number"},
        "h2o_percent": {"type": "number"},
        "n2_percent": {"type": "number"},
        "o2_percent": {"type": "number"}
      }
    }
  }
}
```

### 8.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "flue_gas_loss_kw": {
      "type": "number",
      "description": "Sensible heat loss in flue gas (kW)"
    },
    "flue_gas_loss_mw": {
      "type": "number",
      "description": "Sensible heat loss in flue gas (MW)"
    },
    "flue_gas_loss_percent": {
      "type": "number",
      "description": "Loss as percentage of fuel input (if fuel_input_mw provided)"
    },
    "temperature_difference_k": {
      "type": "number",
      "description": "T_exit - T_ref"
    },
    "mass_flow_kg_s": {
      "type": "number",
      "description": "Mass flow rate converted to kg/s"
    },
    "specific_heat_used_kj_kg_k": {
      "type": "number",
      "description": "Cp value used in calculation"
    },
    "latent_heat_loss_kw": {
      "type": "number",
      "description": "Additional latent heat loss from moisture (if calculated)"
    },
    "total_flue_loss_kw": {
      "type": "number",
      "description": "Sensible + latent losses"
    },
    "recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "recommendation": {"type": "string"},
          "potential_savings_kw": {"type": "number"}
        }
      }
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 8.7 Example Usage

**Request:**
```json
{
  "mass_flow_kg_hr": 12000,
  "exit_temperature_c": 180.0,
  "reference_temperature_c": 25.0,
  "specific_heat_kj_kg_k": 1.10,
  "fuel_input_mw": 50.0
}
```

**Response:**
```json
{
  "flue_gas_loss_kw": 5683.3,
  "flue_gas_loss_mw": 5.683,
  "flue_gas_loss_percent": 11.4,
  "temperature_difference_k": 155.0,
  "mass_flow_kg_s": 3.333,
  "specific_heat_used_kj_kg_k": 1.10,
  "latent_heat_loss_kw": 0,
  "total_flue_loss_kw": 5683.3,
  "recommendations": [
    {
      "recommendation": "Install economizer to recover flue gas heat - reduce stack temp to 120°C",
      "potential_savings_kw": 2200
    },
    {
      "recommendation": "Current stack temperature (180°C) is higher than typical (120-150°C)",
      "potential_savings_kw": 1500
    }
  ],
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "Q_flue = m_dot * Cp * (T_exit - T_ref)",
    "standards_basis": ["ASME PTC 4.1", "First Law of Thermodynamics"]
  }
}
```

### 8.8 Typical Flue Gas Temperatures by Equipment

| Equipment Type | Typical Stack Temp | Recommended Max | Notes |
|----------------|-------------------|-----------------|-------|
| Boiler with economizer | 120-150°C | 160°C | Heat recovery installed |
| Boiler without economizer | 180-250°C | 200°C | High heat loss |
| Process furnace | 200-400°C | 300°C | Depends on process |
| Dryer | 80-120°C | 130°C | Lower temperature process |
| Incinerator | 180-250°C | 280°C | Waste combustion |
| Gas turbine | 450-550°C | 600°C | High exhaust temperature |

---

## 9. Heat Balance Calculator

### 9.1 Tool ID
`calculate_heat_balance`

### 9.2 Description
Performs comprehensive heat balance calculation with closure verification to ensure energy conservation. This validates all input and output measurements.

### 9.3 Physics Basis

**Law:** First Law of Thermodynamics - Conservation of Energy

**Formula:**
```
Q_input = Q_useful + ΣQ_losses ± closure_error

Closure Error (%) = |Q_input - (Q_useful + ΣQ_losses)| / Q_input × 100%

Acceptable per ASME PTC 4.1: closure_error ≤ 2%
Warning if: 2% < closure_error ≤ 5%
Error if: closure_error > 5%
```

### 9.4 Standards Compliance
- **ASME PTC 4.1** - Heat balance methodology and closure tolerance (±2%)
- **ISO 50001:2018** - Energy balance requirements
- **First Law of Thermodynamics** - Energy conservation

### 9.5 Input Schema

```json
{
  "type": "object",
  "required": ["energy_inputs", "useful_outputs", "losses"],
  "properties": {
    "energy_inputs": {
      "type": "object",
      "description": "All energy inputs to the system",
      "properties": {
        "fuel_energy_mw": {
          "type": "number",
          "description": "Energy input from fuel combustion",
          "unit": "MW"
        },
        "electrical_energy_mw": {
          "type": "number",
          "description": "Electrical energy input",
          "unit": "MW"
        },
        "preheated_air_energy_mw": {
          "type": "number",
          "description": "Energy from preheated combustion air",
          "unit": "MW"
        },
        "other_inputs_mw": {
          "type": "number",
          "description": "Other energy inputs",
          "unit": "MW"
        }
      }
    },
    "useful_outputs": {
      "type": "object",
      "description": "Useful energy outputs",
      "properties": {
        "process_heat_mw": {
          "type": "number",
          "description": "Heat transferred to process",
          "unit": "MW"
        },
        "steam_generation_mw": {
          "type": "number",
          "description": "Energy in generated steam",
          "unit": "MW"
        },
        "hot_water_mw": {
          "type": "number",
          "description": "Energy in hot water output",
          "unit": "MW"
        },
        "other_useful_mw": {
          "type": "number",
          "description": "Other useful outputs",
          "unit": "MW"
        }
      }
    },
    "losses": {
      "type": "object",
      "description": "All heat losses",
      "properties": {
        "flue_gas_loss_mw": {"type": "number"},
        "radiation_loss_mw": {"type": "number"},
        "convection_loss_mw": {"type": "number"},
        "blowdown_loss_mw": {"type": "number"},
        "other_losses_mw": {"type": "number"}
      }
    },
    "closure_tolerance_percent": {
      "type": "number",
      "default": 2.0,
      "description": "Acceptable heat balance closure tolerance (ASME PTC 4.1: 2%)",
      "unit": "%"
    }
  }
}
```

### 9.6 Output Schema

```json
{
  "type": "object",
  "properties": {
    "balance_status": {
      "type": "string",
      "enum": ["within_tolerance", "warning", "error"],
      "description": "Overall balance status"
    },
    "closure_achieved": {
      "type": "boolean",
      "description": "True if within tolerance"
    },
    "closure_error_percent": {
      "type": "number",
      "description": "Absolute closure error (%)"
    },
    "closure_error_mw": {
      "type": "number",
      "description": "Absolute closure error (MW)"
    },
    "total_input_mw": {
      "type": "number",
      "description": "Sum of all energy inputs"
    },
    "total_useful_output_mw": {
      "type": "number",
      "description": "Sum of all useful outputs"
    },
    "total_losses_mw": {
      "type": "number",
      "description": "Sum of all losses"
    },
    "unaccounted_losses_mw": {
      "type": "number",
      "description": "Closure error treated as unaccounted losses"
    },
    "input_breakdown": {
      "type": "object",
      "description": "Detailed input breakdown with percentages"
    },
    "output_breakdown": {
      "type": "object",
      "description": "Detailed output breakdown with percentages"
    },
    "loss_breakdown": {
      "type": "object",
      "description": "Detailed loss breakdown with percentages"
    },
    "recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "recommendation": {"type": "string"},
          "priority": {"type": "string"}
        }
      }
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 9.7 Example Usage

**Request:**
```json
{
  "energy_inputs": {
    "fuel_energy_mw": 48.5,
    "electrical_energy_mw": 1.5,
    "preheated_air_energy_mw": 0,
    "other_inputs_mw": 0
  },
  "useful_outputs": {
    "process_heat_mw": 0,
    "steam_generation_mw": 40.0,
    "hot_water_mw": 0,
    "other_useful_mw": 0
  },
  "losses": {
    "flue_gas_loss_mw": 5.5,
    "radiation_loss_mw": 0.8,
    "convection_loss_mw": 1.2,
    "blowdown_loss_mw": 1.0,
    "other_losses_mw": 0
  },
  "closure_tolerance_percent": 2.0
}
```

**Response:**
```json
{
  "balance_status": "within_tolerance",
  "closure_achieved": true,
  "closure_error_percent": 1.6,
  "closure_error_mw": 0.8,
  "total_input_mw": 50.0,
  "total_useful_output_mw": 40.0,
  "total_losses_mw": 8.5,
  "unaccounted_losses_mw": 0.8,
  "input_breakdown": {
    "fuel_energy_percent": 97.0,
    "electrical_energy_percent": 3.0,
    "preheated_air_percent": 0,
    "other_inputs_percent": 0
  },
  "output_breakdown": {
    "useful_output_percent": 80.0,
    "total_losses_percent": 17.0,
    "unaccounted_percent": 1.6
  },
  "loss_breakdown": {
    "flue_gas_loss_percent": 11.0,
    "radiation_loss_percent": 1.6,
    "convection_loss_percent": 2.4,
    "blowdown_loss_percent": 2.0,
    "other_losses_percent": 0,
    "unaccounted_losses_percent": 1.6
  },
  "recommendations": [
    {
      "recommendation": "Heat balance closure within ASME PTC 4.1 tolerance (±2%)",
      "priority": "info"
    },
    {
      "recommendation": "Largest loss is flue gas (11%) - consider economizer installation",
      "priority": "high"
    },
    {
      "recommendation": "Surface losses (radiation + convection = 4%) are typical for this equipment",
      "priority": "medium"
    }
  ],
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "formula_used": "Q_input = Q_useful + sum(Q_losses) +/- closure_error",
    "closure_tolerance_used": 2.0,
    "standards_basis": ["ASME PTC 4.1", "ISO 50001:2018"]
  }
}
```

### 9.8 Closure Error Interpretation

| Closure Error | Status | Action | Notes |
|---------------|--------|--------|-------|
| ≤ 2% | ✅ Acceptable | Proceed | Within ASME PTC 4.1 tolerance |
| 2-5% | ⚠️ Warning | Investigate measurements | Possible measurement error |
| 5-10% | ❌ Error | Remeasure | Significant measurement issue |
| > 10% | ❌ Critical Error | Stop - validate all instruments | Likely instrument failure |

**Common Causes of High Closure Error:**
1. Unmeasured inputs or outputs
2. Instrumentation calibration drift
3. Transient operating conditions
4. Steam/water leaks
5. Unaccounted process streams

---

## 10. Sankey Diagram Generator

### 10.1 Tool ID
`generate_sankey_diagram`

### 10.2 Description
Generates interactive Sankey energy flow diagrams visualizing energy inputs, useful outputs, and losses. Outputs are compatible with Plotly visualization library.

### 10.3 Physics Basis

**Principle:** Conservation of energy - all flows must balance

**Validation:**
```
ΣInputs = ΣOutputs + ΣLosses (within tolerance)

All link widths proportional to energy flow magnitude
Node sizes proportional to energy flow through node
```

### 10.4 Input Schema

```json
{
  "type": "object",
  "required": ["energy_balance"],
  "properties": {
    "energy_balance": {
      "type": "object",
      "description": "Complete energy balance data",
      "properties": {
        "inputs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "value_mw": {"type": "number"},
              "category": {"type": "string"}
            }
          }
        },
        "outputs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "value_mw": {"type": "number"},
              "category": {"type": "string", "enum": ["useful", "loss"]}
            }
          }
        }
      }
    },
    "diagram_config": {
      "type": "object",
      "properties": {
        "width": {"type": "integer", "default": 1200},
        "height": {"type": "integer", "default": 600},
        "color_scheme": {
          "type": "string",
          "enum": ["default", "green", "thermal", "grayscale"],
          "default": "thermal"
        },
        "show_percentages": {"type": "boolean", "default": true},
        "show_values": {"type": "boolean", "default": true},
        "orientation": {
          "type": "string",
          "enum": ["horizontal", "vertical"],
          "default": "horizontal"
        },
        "font_size": {"type": "integer", "default": 12},
        "node_padding": {"type": "integer", "default": 20},
        "link_opacity": {"type": "number", "default": 0.5}
      }
    },
    "export_format": {
      "type": "string",
      "enum": ["plotly_json", "html", "png", "svg", "pdf"],
      "default": "plotly_json"
    }
  }
}
```

### 10.5 Output Schema

```json
{
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "description": "All nodes in the Sankey diagram",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "label": {"type": "string"},
          "value_mw": {"type": "number"},
          "percentage": {"type": "number"},
          "color": {"type": "string"},
          "category": {
            "type": "string",
            "enum": ["input", "useful_output", "loss", "intermediate"]
          }
        }
      }
    },
    "links": {
      "type": "array",
      "description": "All energy flows between nodes",
      "items": {
        "type": "object",
        "properties": {
          "source": {"type": "string"},
          "target": {"type": "string"},
          "value_mw": {"type": "number"},
          "percentage": {"type": "number"},
          "color": {"type": "string"}
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "total_input_mw": {"type": "number"},
        "total_useful_output_mw": {"type": "number"},
        "total_losses_mw": {"type": "number"},
        "overall_efficiency_percent": {"type": "number"},
        "closure_error_percent": {"type": "number"}
      }
    },
    "plotly_figure": {
      "type": "object",
      "description": "Complete Plotly figure object (if format is plotly_json)"
    },
    "html_output": {
      "type": "string",
      "description": "Self-contained HTML with embedded diagram (if format is html)"
    },
    "image_base64": {
      "type": "string",
      "description": "Base64-encoded image (if format is png/svg/pdf)"
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 10.6 Example Usage

**Request:**
```json
{
  "energy_balance": {
    "inputs": [
      {"name": "Natural Gas", "value_mw": 48.5, "category": "fuel"},
      {"name": "Electricity", "value_mw": 1.5, "category": "electrical"}
    ],
    "outputs": [
      {"name": "Steam Production", "value_mw": 40.0, "category": "useful"},
      {"name": "Flue Gas Loss", "value_mw": 5.5, "category": "loss"},
      {"name": "Radiation Loss", "value_mw": 0.8, "category": "loss"},
      {"name": "Convection Loss", "value_mw": 1.2, "category": "loss"},
      {"name": "Blowdown Loss", "value_mw": 1.0, "category": "loss"},
      {"name": "Unaccounted", "value_mw": 0.8, "category": "loss"}
    ]
  },
  "diagram_config": {
    "width": 1200,
    "height": 600,
    "color_scheme": "thermal",
    "show_percentages": true,
    "orientation": "horizontal"
  },
  "export_format": "plotly_json"
}
```

**Response:**
```json
{
  "nodes": [
    {
      "id": "input_fuel",
      "label": "Natural Gas\n48.5 MW (97%)",
      "value_mw": 48.5,
      "percentage": 97.0,
      "color": "#4CAF50",
      "category": "input"
    },
    {
      "id": "input_electrical",
      "label": "Electricity\n1.5 MW (3%)",
      "value_mw": 1.5,
      "percentage": 3.0,
      "color": "#4CAF50",
      "category": "input"
    },
    {
      "id": "system",
      "label": "Boiler System",
      "value_mw": 50.0,
      "percentage": 100.0,
      "color": "#FFC107",
      "category": "intermediate"
    },
    {
      "id": "output_steam",
      "label": "Steam Production\n40.0 MW (80%)",
      "value_mw": 40.0,
      "percentage": 80.0,
      "color": "#2196F3",
      "category": "useful_output"
    },
    {
      "id": "loss_flue",
      "label": "Flue Gas Loss\n5.5 MW (11%)",
      "value_mw": 5.5,
      "percentage": 11.0,
      "color": "#FF9800",
      "category": "loss"
    },
    {
      "id": "loss_radiation",
      "label": "Radiation\n0.8 MW (1.6%)",
      "value_mw": 0.8,
      "percentage": 1.6,
      "color": "#E91E63",
      "category": "loss"
    },
    {
      "id": "loss_convection",
      "label": "Convection\n1.2 MW (2.4%)",
      "value_mw": 1.2,
      "percentage": 2.4,
      "color": "#9C27B0",
      "category": "loss"
    },
    {
      "id": "loss_blowdown",
      "label": "Blowdown\n1.0 MW (2%)",
      "value_mw": 1.0,
      "percentage": 2.0,
      "color": "#795548",
      "category": "loss"
    },
    {
      "id": "loss_unaccounted",
      "label": "Unaccounted\n0.8 MW (1.6%)",
      "value_mw": 0.8,
      "percentage": 1.6,
      "color": "#9E9E9E",
      "category": "loss"
    }
  ],
  "links": [
    {"source": "input_fuel", "target": "system", "value_mw": 48.5, "percentage": 97.0, "color": "#4CAF50"},
    {"source": "input_electrical", "target": "system", "value_mw": 1.5, "percentage": 3.0, "color": "#4CAF50"},
    {"source": "system", "target": "output_steam", "value_mw": 40.0, "percentage": 80.0, "color": "#2196F3"},
    {"source": "system", "target": "loss_flue", "value_mw": 5.5, "percentage": 11.0, "color": "#FF9800"},
    {"source": "system", "target": "loss_radiation", "value_mw": 0.8, "percentage": 1.6, "color": "#E91E63"},
    {"source": "system", "target": "loss_convection", "value_mw": 1.2, "percentage": 2.4, "color": "#9C27B0"},
    {"source": "system", "target": "loss_blowdown", "value_mw": 1.0, "percentage": 2.0, "color": "#795548"},
    {"source": "system", "target": "loss_unaccounted", "value_mw": 0.8, "percentage": 1.6, "color": "#9E9E9E"}
  ],
  "metadata": {
    "total_input_mw": 50.0,
    "total_useful_output_mw": 40.0,
    "total_losses_mw": 9.2,
    "overall_efficiency_percent": 80.0,
    "closure_error_percent": 1.6
  },
  "plotly_figure": {
    "data": [{
      "type": "sankey",
      "node": {...},
      "link": {...}
    }],
    "layout": {
      "title": "Energy Flow Diagram - Overall Efficiency: 80.0%",
      "font": {"size": 12}
    }
  },
  "provenance": {
    "timestamp": "2025-11-26T10:30:00Z",
    "diagram_type": "sankey",
    "standards_basis": ["Conservation of Energy"]
  }
}
```

### 10.7 Color Schemes

#### Thermal Color Scheme (Default)
- **Inputs:** Green (#4CAF50)
- **Useful Outputs:** Blue (#2196F3)
- **Flue Gas Loss:** Orange (#FF9800)
- **Radiation Loss:** Pink (#E91E63)
- **Convection Loss:** Purple (#9C27B0)
- **Blowdown Loss:** Brown (#795548)
- **Unaccounted:** Gray (#9E9E9E)

---

## 11. Improvement Opportunity Identifier

### 11.1 Tool ID
`identify_improvement_opportunities`

### 11.2 Description
Identifies and prioritizes efficiency improvement opportunities based on loss analysis, benchmarking, and ROI calculations. Combines deterministic physics with rule-based classification.

### 11.3 Methodology

**Step 1: Loss Analysis**
- Rank losses by magnitude
- Identify reducible vs. irreducible losses
- Calculate theoretical reduction potential

**Step 2: Benchmarking**
- Compare current efficiency vs. industry benchmarks
- Calculate efficiency gap
- Identify best practice targets

**Step 3: Opportunity Identification**
- Apply rule-based decision trees
- Match loss patterns to known solutions
- Estimate implementation cost and savings

**Step 4: ROI Calculation**
```
Annual Savings ($) = Energy Saved (kWh/year) × Energy Cost ($/kWh)
Simple Payback (months) = Implementation Cost ($) / (Annual Savings ($) / 12)
NPV (10-year) = Σ[Savings / (1+r)^t] - Implementation Cost
IRR = Rate where NPV = 0
```

### 11.4 Input Schema

```json
{
  "type": "object",
  "required": ["current_efficiency", "loss_breakdown"],
  "properties": {
    "current_efficiency": {
      "type": "object",
      "properties": {
        "first_law_efficiency_percent": {"type": "number"},
        "second_law_efficiency_percent": {"type": "number"},
        "combustion_efficiency_percent": {"type": "number"}
      }
    },
    "loss_breakdown": {
      "type": "object",
      "properties": {
        "flue_gas_loss_mw": {"type": "number"},
        "flue_gas_temperature_c": {"type": "number"},
        "excess_air_percent": {"type": "number"},
        "radiation_loss_mw": {"type": "number"},
        "surface_temperature_c": {"type": "number"},
        "convection_loss_mw": {"type": "number"},
        "blowdown_loss_mw": {"type": "number"},
        "other_losses_mw": {"type": "number"}
      }
    },
    "process_type": {
      "type": "string",
      "enum": ["boiler", "furnace", "dryer", "kiln", "heat_exchanger", "reactor"]
    },
    "facility_parameters": {
      "type": "object",
      "properties": {
        "fuel_type": {"type": "string"},
        "capacity_mw": {"type": "number"},
        "operating_hours_per_year": {"type": "number", "default": 8760},
        "energy_cost_usd_per_mwh": {"type": "number", "default": 50.0},
        "discount_rate_percent": {"type": "number", "default": 10.0},
        "year_commissioned": {"type": "integer"},
        "last_major_overhaul_year": {"type": "integer"}
      }
    },
    "constraints": {
      "type": "object",
      "properties": {
        "max_payback_months": {"type": "number", "default": 36},
        "max_implementation_cost_usd": {"type": "number"},
        "min_savings_percent": {"type": "number", "default": 1.0},
        "complexity_limit": {
          "type": "string",
          "enum": ["simple", "moderate", "complex"],
          "default": "complex"
        }
      }
    }
  }
}
```

### 11.5 Output Schema

```json
{
  "type": "object",
  "properties": {
    "opportunities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "opportunity_id": {"type": "string"},
          "rank": {"type": "integer"},
          "category": {
            "type": "string",
            "enum": [
              "heat_recovery",
              "combustion_optimization",
              "insulation",
              "process_integration",
              "equipment_upgrade",
              "operational",
              "control_optimization",
              "maintenance"
            ]
          },
          "title": {"type": "string"},
          "description": {"type": "string"},
          "technical_details": {"type": "string"},
          "potential_savings_mw": {"type": "number"},
          "potential_savings_percent": {"type": "number"},
          "annual_energy_savings_mwh": {"type": "number"},
          "annual_cost_savings_usd": {"type": "number"},
          "estimated_implementation_cost_usd": {"type": "number"},
          "simple_payback_months": {"type": "number"},
          "npv_10yr_usd": {"type": "number"},
          "irr_percent": {"type": "number"},
          "priority": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low"]
          },
          "implementation_complexity": {
            "type": "string",
            "enum": ["simple", "moderate", "complex"]
          },
          "implementation_time_months": {"type": "number"},
          "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"]
          },
          "confidence_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "co2_reduction_tonnes_per_year": {"type": "number"},
          "prerequisites": {"type": "array", "items": {"type": "string"}},
          "vendors_suppliers": {"type": "array", "items": {"type": "string"}},
          "case_studies": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "summary": {
      "type": "object",
      "properties": {
        "total_opportunities": {"type": "integer"},
        "total_potential_savings_mw": {"type": "number"},
        "total_potential_savings_percent": {"type": "number"},
        "total_annual_cost_savings_usd": {"type": "number"},
        "total_implementation_cost_usd": {"type": "number"},
        "weighted_average_payback_months": {"type": "number"},
        "total_co2_reduction_tonnes_per_year": {"type": "number"}
      }
    },
    "priority_matrix": {
      "type": "object",
      "description": "Opportunities grouped by priority and complexity"
    },
    "implementation_roadmap": {
      "type": "array",
      "description": "Recommended sequence of implementations",
      "items": {
        "type": "object",
        "properties": {
          "phase": {"type": "integer"},
          "timeframe": {"type": "string"},
          "opportunities": {"type": "array"},
          "cumulative_savings_percent": {"type": "number"}
        }
      }
    },
    "provenance": {
      "type": "object"
    }
  }
}
```

### 11.6 Example Opportunities

#### Opportunity 1: Economizer Installation
```json
{
  "opportunity_id": "OPP-001",
  "rank": 1,
  "category": "heat_recovery",
  "title": "Install Economizer for Flue Gas Heat Recovery",
  "description": "Recover waste heat from flue gas to preheat feedwater, reducing fuel consumption and stack temperature from 180°C to 120°C.",
  "technical_details": "Install counterflow shell-and-tube economizer with surface area 150 m². Expected heat recovery: 2.2 MW. Stainless steel construction for corrosion resistance below dew point.",
  "potential_savings_mw": 2.2,
  "potential_savings_percent": 4.4,
  "annual_energy_savings_mwh": 19272,
  "annual_cost_savings_usd": 963600,
  "estimated_implementation_cost_usd": 450000,
  "simple_payback_months": 5.6,
  "npv_10yr_usd": 5200000,
  "irr_percent": 190,
  "priority": "high",
  "implementation_complexity": "moderate",
  "implementation_time_months": 6,
  "risk_level": "low",
  "confidence_score": 0.95,
  "co2_reduction_tonnes_per_year": 1080,
  "prerequisites": ["Stack gas temperature measurement", "Feedwater quality analysis"],
  "vendors_suppliers": ["Babcock & Wilcox", "Cleaver-Brooks", "Victory Energy"],
  "case_studies": ["DOE Industrial Assessment Center Report #12345"]
}
```

---

## 12. Efficiency Benchmarker

### 12.1 Tool ID
`benchmark_efficiency`

### 12.2 Description
Compares current efficiency against industry benchmarks and provides percentile ranking.

### 12.3 Input Schema

```json
{
  "type": "object",
  "required": ["current_efficiency_percent", "process_type"],
  "properties": {
    "current_efficiency_percent": {"type": "number"},
    "process_type": {"type": "string"},
    "capacity_mw": {"type": "number"},
    "fuel_type": {"type": "string"},
    "year_commissioned": {"type": "integer"}
  }
}
```

### 12.4 Output Schema

```json
{
  "type": "object",
  "properties": {
    "current_efficiency_percent": {"type": "number"},
    "percentile_rank": {"type": "number"},
    "industry_average_percent": {"type": "number"},
    "top_quartile_percent": {"type": "number"},
    "best_in_class_percent": {"type": "number"},
    "gap_from_average_percent": {"type": "number"},
    "potential_improvement_percent": {"type": "number"},
    "recommendations": {"type": "array"}
  }
}
```

---

## 13. Exergy Flow Calculator

### 13.1 Tool ID
`calculate_exergy_flows`

### 13.2 Description
Calculates exergy (availability) for process streams using thermodynamic properties.

### 13.3 Formula

```
Ex = m × [(h - h₀) - T₀ × (s - s₀)]

Where:
Ex = Exergy (kW)
m = Mass flow rate (kg/s)
h = Specific enthalpy (kJ/kg)
h₀ = Dead state enthalpy (kJ/kg)
s = Specific entropy (kJ/kg-K)
s₀ = Dead state entropy (kJ/kg-K)
T₀ = Dead state temperature (K)
```

---

## 14. Trend Analyzer

### 14.1 Tool ID
`trend_analysis`

### 14.2 Description
Analyzes efficiency trends over time using statistical time series analysis.

### 14.3 Input Schema

```json
{
  "type": "object",
  "required": ["historical_data"],
  "properties": {
    "historical_data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "string", "format": "date-time"},
          "efficiency_percent": {"type": "number"}
        }
      }
    },
    "analysis_period_days": {"type": "integer", "default": 365}
  }
}
```

### 14.4 Output Schema

```json
{
  "type": "object",
  "properties": {
    "trend_direction": {
      "type": "string",
      "enum": ["improving", "stable", "degrading"]
    },
    "degradation_rate_percent_per_year": {"type": "number"},
    "anomalies": {"type": "array"},
    "forecast_30_days": {"type": "object"}
  }
}
```

---

## 15. Constants and Reference Data

### 15.1 Physical Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Stefan-Boltzmann | σ | 5.67×10⁻⁸ | W/(m²·K⁴) |
| Universal gas constant | R | 8.314 | kJ/(kmol·K) |
| Standard gravity | g | 9.81 | m/s² |
| Standard atmospheric pressure | P₀ | 101.325 | kPa |

### 15.2 Fuel Properties

| Fuel | LHV (MJ/kg) | HHV (MJ/kg) | CO₂ Factor (kg/MJ) |
|------|-------------|-------------|---------------------|
| Natural gas | 50.0 | 55.5 | 0.0561 |
| Fuel oil No. 2 | 42.5 | 45.5 | 0.0774 |
| Coal (bituminous) | 30.5 | 32.0 | 0.0946 |
| Biomass (wood) | 18.5 | 20.0 | 0.0 (neutral) |
| Hydrogen | 120.0 | 141.8 | 0.0 |

### 15.3 Industry Benchmarks

See Section 12 for detailed benchmark data by process type.

---

## 16. Standards Compliance

### 16.1 ASME PTC 4.1
- Heat balance methodology
- Closure tolerance: ±2%
- Loss calculation methods

### 16.2 ISO 50001:2018
- Energy performance indicators (EnPIs)
- Energy baseline establishment
- Measurement and verification

### 16.3 EPA 40 CFR Part 60
- Flue gas measurement procedures
- Emissions monitoring requirements
- Continuous emissions monitoring systems (CEMS)

---

**Document End**

For questions or support, contact: support@greenlang.org
