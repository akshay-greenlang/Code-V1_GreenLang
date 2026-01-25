# GL-017 CONDENSYNC Tools Documentation

## Deterministic Tool Functions

**Version:** 1.0.0
**Last Updated:** December 2025

---

## Overview

GL-017 CONDENSYNC implements 10 deterministic calculation and optimization tools for steam condenser operations. All tools follow zero-hallucination principles with no LLM involvement in numeric calculations.

All calculations are based on:
- HEI (Heat Exchange Institute) Standards - HEI-2629 (2022)
- ASME Performance Test Code - PTC 12.2 (2022)
- TEMA Standards (2019)
- EPRI Condenser Optimization Guidelines

---

## Table of Contents

1. [analyze_condenser_performance](#1-analyze_condenser_performance)
2. [calculate_heat_transfer_coefficient](#2-calculate_heat_transfer_coefficient)
3. [optimize_vacuum_pressure](#3-optimize_vacuum_pressure)
4. [detect_air_inleakage](#4-detect_air_inleakage)
5. [calculate_fouling_factor](#5-calculate_fouling_factor)
6. [predict_tube_cleaning_schedule](#6-predict_tube_cleaning_schedule)
7. [optimize_cooling_water_flow](#7-optimize_cooling_water_flow)
8. [generate_performance_report](#8-generate_performance_report)
9. [calculate_condenser_duty](#9-calculate_condenser_duty)
10. [assess_condenser_health](#10-assess_condenser_health)

---

## Tool Executor

All tools are executed through the `CondenserToolExecutor` class:

```python
from greenlang.GL_017.tools import CondenserToolExecutor

executor = CondenserToolExecutor()
result = await executor.execute_tool(tool_name, parameters)
```

---

## 1. analyze_condenser_performance

### Description

Analyze overall condenser performance including heat transfer efficiency, vacuum levels, and operating trends. Provides a comprehensive assessment of condenser health and identifies areas for improvement.

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `condenser_id` | string | Yes | Unique identifier for the condenser unit |
| `time_range` | object | No | Analysis time period |
| `time_range.start` | string (ISO datetime) | Yes* | Start of analysis period |
| `time_range.end` | string (ISO datetime) | Yes* | End of analysis period |
| `include_trends` | boolean | No | Include historical trend data (default: true) |
| `operating_data` | object | Yes | Current operating measurements |
| `operating_data.cw_inlet_temp` | number | Yes | Cooling water inlet temperature (Celsius) |
| `operating_data.cw_outlet_temp` | number | Yes | Cooling water outlet temperature (Celsius) |
| `operating_data.cw_flow_rate` | number | No | Cooling water flow rate (m3/hr) |
| `operating_data.vacuum_pressure` | number | Yes | Condenser vacuum pressure (mbar abs) |
| `operating_data.steam_flow` | number | No | Exhaust steam flow rate (kg/hr) |
| `operating_data.turbine_load` | number | No | Turbine generator load (MW) |

### Output

```json
{
  "condenser_id": "COND-001",
  "heat_duty": 280000.0,
  "u_value": 2850.5,
  "cleanliness_factor": 0.82,
  "lmtd": 14.2,
  "ttd": 5.8,
  "dca": 2.1,
  "vacuum_pressure": 45.0,
  "saturation_temp": 31.2,
  "efficiency": 92.5,
  "trends": {
    "u_value": [2900, 2880, 2860, 2850],
    "cleanliness": [0.85, 0.84, 0.83, 0.82]
  },
  "status": "normal",
  "recommendations": [
    "Cleanliness factor declining - monitor fouling rate",
    "Schedule tube inspection within 60 days"
  ],
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:abc123..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "analyze_condenser_performance",
    {
        "condenser_id": "COND-001",
        "time_range": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-15T00:00:00Z"
        },
        "include_trends": True,
        "operating_data": {
            "cw_inlet_temp": 22.0,
            "cw_outlet_temp": 32.0,
            "cw_flow_rate": 45000.0,
            "vacuum_pressure": 45.0,
            "turbine_load": 450.0
        }
    }
)
```

---

## 2. calculate_heat_transfer_coefficient

### Description

Calculate the overall heat transfer coefficient (U-value) for the condenser using formula-based deterministic calculation. Computes cleanliness factor, LMTD, TTD, and fouling resistance based on HEI standards.

### Calculation Method

```
LMTD = (deltaT2 - deltaT1) / ln(deltaT2 / deltaT1)

where:
  deltaT1 = T_steam - T_cw_out (TTD)
  deltaT2 = T_steam - T_cw_in

U = Q / (A * LMTD)

Cleanliness Factor = U_actual / U_design
```

### Input Parameters

| Parameter | Type | Required | Description | Range |
|-----------|------|----------|-------------|-------|
| `cw_inlet_temp` | number | Yes | CW inlet temperature (Celsius) | 0-50 |
| `cw_outlet_temp` | number | Yes | CW outlet temperature (Celsius) | 0-60 |
| `cw_flow_rate` | number | Yes | CW flow rate (m3/hr) | > 0 |
| `steam_temp` | number | Yes | Steam saturation temp (Celsius) | 20-60 |
| `heat_duty` | number | Yes | Heat duty (kW) | > 0 |
| `tube_surface_area` | number | No | Tube surface area (m2) | > 0 |
| `design_u_value` | number | No | Design U-value W/(m2.K) | default: 3000 |
| `tube_od` | number | No | Tube outer diameter (mm) | default: 25.4 |
| `tube_thickness` | number | No | Tube wall thickness (mm) | default: 1.24 |
| `tube_material` | string | No | Tube material type | See enum |

**Tube Material Enum:**
- `admiralty_brass`
- `copper_nickel_90_10`
- `copper_nickel_70_30`
- `titanium`
- `stainless_steel_304`
- `stainless_steel_316`
- `carbon_steel`

### Output

```json
{
  "u_value": 2847.5,
  "u_design": 3000.0,
  "cleanliness_factor": 0.949,
  "lmtd": 12.8,
  "ttd": 4.5,
  "heat_duty": 250000.0,
  "heat_flux": 15625.0,
  "fouling_resistance": 0.000018,
  "tube_side_coefficient": 8500.0,
  "shell_side_coefficient": 12000.0,
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:def456..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "calculate_heat_transfer_coefficient",
    {
        "cw_inlet_temp": 20.0,
        "cw_outlet_temp": 30.0,
        "cw_flow_rate": 50000.0,
        "steam_temp": 35.0,
        "heat_duty": 300000.0,
        "tube_surface_area": 16000.0,
        "design_u_value": 3000.0,
        "tube_material": "titanium"
    }
)

print(f"U-value: {result['u_value']} W/(m2.K)")
print(f"Cleanliness: {result['cleanliness_factor'] * 100:.1f}%")
```

---

## 3. optimize_vacuum_pressure

### Description

Optimize condenser vacuum pressure setpoint based on turbine load, ambient conditions, and equipment constraints. Uses engineering optimization algorithms to determine optimal vacuum for maximum efficiency.

### Optimization Logic

```
Achievable_Vacuum = f(CW_inlet_temp, TTD_design, Cleanliness_factor)

Optimal_Vacuum = min(
    Achievable_Vacuum,
    Turbine_Backpressure_Limit,
    Equipment_Constraints
)

Expected_Efficiency_Gain = (Current_Vacuum - Optimal_Vacuum) * Heat_Rate_Sensitivity
```

### Input Parameters

| Parameter | Type | Required | Description | Range |
|-----------|------|----------|-------------|-------|
| `current_vacuum` | number | Yes | Current vacuum (mbar abs) | 20-150 |
| `turbine_load` | number | Yes | Turbine load (MW) | > 0 |
| `ambient_conditions` | object | Yes | Weather conditions | - |
| `ambient_conditions.dry_bulb_temp` | number | Yes | Dry bulb temp (Celsius) | - |
| `ambient_conditions.wet_bulb_temp` | number | Yes | Wet bulb temp (Celsius) | - |
| `ambient_conditions.relative_humidity` | number | No | Relative humidity (%) | 0-100 |
| `cw_inlet_temp` | number | No | CW inlet temperature (Celsius) | - |
| `cooling_tower_approach` | number | No | CT approach temp (Celsius) | default: 5.0 |
| `condenser_design_ttd` | number | No | Design TTD (K) | default: 3.0 |
| `turbine_backpressure_limit` | number | No | Max backpressure (mbar abs) | default: 100 |

### Output

```json
{
  "current_vacuum": 52.0,
  "optimal_vacuum": 45.0,
  "expected_efficiency_gain": 0.35,
  "expected_power_gain": 1750.0,
  "limiting_factor": "cooling_water_temperature",
  "achievable_vacuum": 43.0,
  "action_items": [
    "Reduce vacuum setpoint to 45 mbar abs",
    "Monitor cooling tower performance",
    "Check air ejector capacity"
  ],
  "cost_benefit": {
    "annual_savings_usd": 450000.0,
    "efficiency_improvement_pct": 0.35,
    "heat_rate_improvement_btu_kwh": 35.0
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:ghi789..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "optimize_vacuum_pressure",
    {
        "current_vacuum": 55.0,
        "turbine_load": 450.0,
        "ambient_conditions": {
            "dry_bulb_temp": 30.0,
            "wet_bulb_temp": 24.0,
            "relative_humidity": 65.0
        },
        "cw_inlet_temp": 28.0,
        "cooling_tower_approach": 5.0,
        "turbine_backpressure_limit": 90.0
    }
)
```

---

## 4. detect_air_inleakage

### Description

Detect and assess air inleakage into the condenser based on vacuum trends and air ejector operating data. Uses pattern recognition algorithms to identify probable leak locations and severity.

### Detection Algorithm

```
Leakage_Rate = Air_Ejector_Load / Design_Capacity * Baseline_Leakage

Severity = classify(Leakage_Rate / Reference_Rate_per_MW)
  - < 0.5: NONE
  - 0.5-1.0: MINOR
  - 1.0-2.0: MODERATE
  - 2.0-4.0: SEVERE
  - > 4.0: CRITICAL

Vacuum_Impact = Leakage_Rate * Vacuum_Sensitivity_Factor
```

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `condenser_id` | string | Yes | Unique condenser identifier |
| `vacuum_trend` | object | Yes | Historical vacuum data |
| `vacuum_trend.timestamps` | array[string] | Yes | ISO datetime timestamps |
| `vacuum_trend.values` | array[number] | Yes | Vacuum readings (mbar abs) |
| `air_ejector_data` | object | Yes | Air removal system data |
| `air_ejector_data.ejector_type` | string | No | Ejector type (see enum) |
| `air_ejector_data.suction_pressure` | number | Yes | Suction pressure (mbar abs) |
| `air_ejector_data.discharge_temp` | number | No | Discharge temperature (Celsius) |
| `air_ejector_data.motive_steam_flow` | number | No | Motive steam flow (kg/hr) |
| `air_ejector_data.design_capacity` | number | Yes | Design air removal (kg/hr) |
| `air_ejector_data.current_load` | number | Yes | Current air removal (kg/hr) |
| `turbine_load` | number | No | Turbine load for normalization (MW) |

**Ejector Type Enum:**
- `steam_jet_ejector`
- `liquid_ring_pump`
- `rotary_vane_pump`
- `hybrid_system`

### Output

```json
{
  "condenser_id": "COND-001",
  "estimated_leakage_rate": 12.5,
  "severity": "moderate",
  "probable_locations": [
    {
      "location": "LP turbine gland seals",
      "probability": 0.45,
      "typical_leak_rate": 5.0
    },
    {
      "location": "Expansion joint bellows",
      "probability": 0.30,
      "typical_leak_rate": 3.0
    },
    {
      "location": "Valve stem packing",
      "probability": 0.15,
      "typical_leak_rate": 2.0
    }
  ],
  "vacuum_degradation": 3.5,
  "ejector_load_percent": 65.0,
  "detection_confidence": 0.85,
  "recommended_actions": [
    "Conduct ultrasonic leak survey",
    "Inspect LP turbine gland steam system",
    "Check expansion joint condition",
    "Review recent maintenance activities"
  ],
  "estimated_repair_priority": 2,
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:jkl012..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "detect_air_inleakage",
    {
        "condenser_id": "COND-001",
        "vacuum_trend": {
            "timestamps": [
                "2025-01-15T09:00:00Z",
                "2025-01-15T09:15:00Z",
                "2025-01-15T09:30:00Z",
                "2025-01-15T09:45:00Z"
            ],
            "values": [48.0, 49.5, 51.0, 52.0]
        },
        "air_ejector_data": {
            "ejector_type": "steam_jet_ejector",
            "suction_pressure": 45.0,
            "design_capacity": 50.0,
            "current_load": 32.5
        },
        "turbine_load": 450.0
    }
)
```

---

## 5. calculate_fouling_factor

### Description

Calculate fouling factor and resistance using HEI standard methodology. Compares design vs actual U-values to determine cleanliness and estimate deposit thickness.

### Calculation Method (HEI 2629)

```
Fouling_Resistance = (1 / U_actual) - (1 / U_design)

Fouling_Factor = U_actual / U_design

Deposit_Thickness = Fouling_Resistance * k_deposit / 1000  (mm)

where k_deposit ~ 1.0 W/(m.K) for typical fouling deposits
```

### Input Parameters

| Parameter | Type | Required | Description | Range |
|-----------|------|----------|-------------|-------|
| `design_U` | number | Yes | Design U-value W/(m2.K) | 1000-6000 |
| `actual_U` | number | Yes | Actual U-value W/(m2.K) | 500-6000 |
| `tube_material` | string | Yes | Tube material type | See enum |
| `operating_hours` | number | Yes | Hours since last cleaning | >= 0 |
| `water_source` | string | No | Cooling water source | See enum |
| `historical_fouling_data` | array | No | Historical data for trending | - |

**Water Source Enum:**
- `clean_fresh_water` (Rf = 0.000044 m2.K/W)
- `treated_cooling_tower` (Rf = 0.000088 m2.K/W)
- `untreated_cooling_tower` (Rf = 0.000176 m2.K/W)
- `brackish_water` (Rf = 0.000352 m2.K/W)
- `seawater` (Rf = 0.000088 m2.K/W)
- `river_water` (Rf = 0.000352 m2.K/W)
- `well_water` (Rf = 0.000176 m2.K/W)

### Output

```json
{
  "fouling_resistance": 0.000125,
  "fouling_factor": 0.78,
  "degradation_rate": 0.000015,
  "estimated_deposit_thickness": 0.125,
  "cleaning_recommended": true,
  "cleaning_urgency": "scheduled",
  "expected_improvement": 22.0,
  "time_to_critical": 2500.0,
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:mno345..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "calculate_fouling_factor",
    {
        "design_U": 3000.0,
        "actual_U": 2340.0,
        "tube_material": "titanium",
        "operating_hours": 6500.0,
        "water_source": "treated_cooling_tower"
    }
)

print(f"Fouling resistance: {result['fouling_resistance']:.6f} m2.K/W")
print(f"Cleanliness factor: {result['fouling_factor'] * 100:.1f}%")
print(f"Cleaning urgency: {result['cleaning_urgency']}")
```

---

## 6. predict_tube_cleaning_schedule

### Description

Predict optimal tube cleaning schedule based on fouling trends and production constraints. Uses predictive analytics to balance cleaning cost against efficiency losses.

### Prediction Model

```
Time_to_Threshold = (Current_CF - Threshold_CF) / Degradation_Rate

Optimal_Date = max(
    Time_to_Threshold,
    Next_Planned_Outage
) excluding High_Demand_Periods

Net_Benefit = (Efficiency_Gain * Electricity_Price * Runtime) - Cleaning_Cost - Lost_Production
```

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `condenser_id` | string | Yes | Unique condenser identifier |
| `fouling_trend` | object | Yes | Historical fouling data |
| `fouling_trend.timestamps` | array[string] | Yes | ISO datetime timestamps |
| `fouling_trend.cleanliness_factors` | array[number] | Yes | Cleanliness factor values |
| `fouling_trend.fouling_resistances` | array[number] | No | Fouling resistance values |
| `production_schedule` | object | No | Production constraints |
| `production_schedule.planned_outages` | array | No | Planned outage periods |
| `production_schedule.high_demand_periods` | array | No | Peak demand periods |
| `production_schedule.electricity_price_forecast` | number | No | Price (USD/MWh) |
| `current_cleanliness` | number | Yes | Current cleanliness factor (0-1) |
| `cleaning_threshold` | number | No | Trigger threshold | default: 0.75 |
| `cleaning_cost` | number | No | Cleaning cost (USD) | default: 50000 |
| `unit_capacity` | number | No | Unit capacity (MW) |

### Output

```json
{
  "condenser_id": "COND-001",
  "recommended_cleaning_date": "2025-03-15",
  "days_until_cleaning": 59,
  "cleaning_method": "mechanical_brushing",
  "expected_u_value_recovery": 18.5,
  "expected_efficiency_gain": 0.28,
  "estimated_duration": 16.0,
  "estimated_cost": 45000.0,
  "production_impact": 7200.0,
  "net_benefit": 125000.0,
  "confidence_level": 0.88,
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:pqr678..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "predict_tube_cleaning_schedule",
    {
        "condenser_id": "COND-001",
        "fouling_trend": {
            "timestamps": [
                "2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01"
            ],
            "cleanliness_factors": [0.92, 0.88, 0.84, 0.80]
        },
        "production_schedule": {
            "planned_outages": [
                {"start_date": "2025-03-10", "end_date": "2025-03-20", "type": "planned"}
            ],
            "high_demand_periods": [
                {"start_date": "2025-06-01", "end_date": "2025-08-31"}
            ],
            "electricity_price_forecast": 55.0
        },
        "current_cleanliness": 0.80,
        "cleaning_threshold": 0.75,
        "unit_capacity": 500.0
    }
)
```

---

## 7. optimize_cooling_water_flow

### Description

Optimize cooling water flow rate for given heat duty and target vacuum. Balances pump energy consumption against condenser performance using hydraulic optimization.

### Optimization Algorithm

```
Required_Flow = Heat_Duty / (Cp * deltaT_target * rho)

Pump_Power = (Flow * Head * rho * g) / (Efficiency * 1000)  [kW]

Power varies as Flow^3 (affinity laws)

Optimal_Flow = minimize(
    Pump_Power + Vacuum_Penalty,
    subject_to: Min_Flow <= Flow <= Max_Flow
)
```

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `heat_duty` | number | Yes | Heat duty (kW) |
| `cw_inlet_temp` | number | Yes | CW inlet temperature (Celsius) |
| `target_vacuum` | number | Yes | Target vacuum (mbar abs) |
| `current_flow_rate` | number | No | Current flow rate (m3/hr) |
| `pump_characteristics` | object | No | Pump curve data |
| `pump_characteristics.design_flow` | number | No | Design flow (m3/hr) |
| `pump_characteristics.design_head` | number | No | Design head (m) |
| `pump_characteristics.design_power` | number | No | Design power (kW) |
| `pump_characteristics.efficiency` | number | No | Pump efficiency (0-1) |
| `pump_characteristics.min_flow` | number | No | Minimum flow (m3/hr) |
| `pump_characteristics.max_flow` | number | No | Maximum flow (m3/hr) |
| `pump_characteristics.vfd_equipped` | boolean | No | VFD installed |
| `condenser_surface_area` | number | No | Surface area (m2) |
| `design_u_value` | number | No | Design U-value W/(m2.K) |
| `electricity_cost` | number | No | Electricity cost (USD/kWh) | default: 0.08 |

### Output

```json
{
  "current_flow_rate": 50000.0,
  "optimal_flow_rate": 45000.0,
  "flow_change": -10.0,
  "pump_power_current": 850.0,
  "pump_power_optimal": 620.0,
  "pump_energy_savings": 230.0,
  "vacuum_impact": 1.5,
  "efficiency_impact": -0.05,
  "annual_savings": 145000.0,
  "constraints": [
    "Minimum tube velocity maintained",
    "VFD speed within operating range"
  ],
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:stu901..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "optimize_cooling_water_flow",
    {
        "heat_duty": 280000.0,
        "cw_inlet_temp": 25.0,
        "target_vacuum": 45.0,
        "current_flow_rate": 50000.0,
        "pump_characteristics": {
            "design_flow": 55000.0,
            "design_head": 25.0,
            "design_power": 450.0,
            "efficiency": 0.82,
            "min_flow": 35000.0,
            "max_flow": 60000.0,
            "vfd_equipped": True
        },
        "condenser_surface_area": 16000.0,
        "electricity_cost": 0.075
    }
)
```

---

## 8. generate_performance_report

### Description

Generate comprehensive condenser performance report for specified period. Aggregates metrics, identifies trends, and provides actionable recommendations.

**Note:** This tool may use LLM assistance for narrative generation, but all numeric calculations are deterministic.

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `condenser_id` | string | Yes | Unique condenser identifier |
| `report_period` | object | Yes | Reporting time range |
| `report_period.start` | string (date) | Yes | Start date |
| `report_period.end` | string (date) | Yes | End date |
| `metrics_to_include` | array[string] | No | Metrics to include |
| `operating_data_summary` | object | No | Summary operating data |
| `include_recommendations` | boolean | No | Include recommendations | default: true |
| `include_cost_analysis` | boolean | No | Include cost analysis | default: true |

**Available Metrics:**
- `heat_duty`
- `u_value`
- `cleanliness_factor`
- `ttd`
- `vacuum_pressure`
- `efficiency`
- `air_inleakage`
- `cw_flow_rate`
- `cw_temperatures`
- `energy_savings`

### Output

```json
{
  "condenser_id": "COND-001",
  "report_period": "2025-01-01 to 2025-01-31",
  "summary_metrics": {
    "avg_heat_duty_mw": 285.0,
    "avg_vacuum_mbar": 47.5,
    "avg_u_value": 2780.0,
    "avg_cleanliness_factor": 0.82,
    "avg_ttd_k": 5.2,
    "total_operating_hours": 720.0
  },
  "trend_analysis": {
    "u_value_trend": "declining",
    "u_value_slope": -2.5,
    "cleanliness_trend": "declining",
    "vacuum_trend": "stable"
  },
  "efficiency_statistics": {
    "min_efficiency": 90.5,
    "max_efficiency": 94.2,
    "avg_efficiency": 92.3,
    "std_dev": 0.8
  },
  "fouling_status": {
    "current_cleanliness": 0.80,
    "days_since_cleaning": 180,
    "days_until_cleaning": 45,
    "fouling_rate": 0.001
  },
  "maintenance_events": [
    {
      "date": "2025-01-10",
      "type": "inspection",
      "findings": "Minor debris in waterboxes"
    }
  ],
  "recommendations": [
    "Schedule tube cleaning during March outage",
    "Investigate declining cleanliness factor",
    "Consider increasing chlorination frequency"
  ],
  "kpis": {
    "availability_pct": 99.2,
    "vacuum_within_target_pct": 94.5,
    "energy_savings_usd": 12500.0
  },
  "timestamp": "2025-02-01T08:00:00Z",
  "provenance_hash": "sha256:vwx234..."
}
```

---

## 9. calculate_condenser_duty

### Description

Calculate condenser heat duty from turbine exhaust conditions or cooling water temperature rise. Deterministic thermodynamic calculation based on energy balance equations.

### Calculation Methods

**Method 1: Cooling Water Side**
```
Q = m_cw * Cp * (T_out - T_in)

where:
  m_cw = rho * V_flow / 3600  [kg/s]
  Cp = 4.186 kJ/(kg.K)
  rho = 1000 kg/m3
```

**Method 2: Steam Side**
```
Q = m_steam * (h_fg * x + Cp_liq * subcooling)

where:
  h_fg = latent heat at condenser pressure
  x = steam quality
```

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `calculation_method` | string | Yes | `cooling_water` or `steam_side` |
| `cw_inlet_temp` | number | Conditional | CW inlet temperature (Celsius) |
| `cw_outlet_temp` | number | Conditional | CW outlet temperature (Celsius) |
| `cw_flow_rate` | number | Conditional | CW flow rate (m3/hr) |
| `steam_flow` | number | Conditional | Steam flow rate (kg/hr) |
| `steam_quality` | number | No | Steam dryness fraction | default: 0.9 |
| `condenser_pressure` | number | No | Condenser pressure (mbar abs) |

### Output

```json
{
  "heat_duty": 295000.0,
  "cw_temp_rise": 10.5,
  "specific_duty": 5.9,
  "calculation_method": "cooling_water",
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:yza567..."
}
```

### Example Usage

```python
# Cooling water method
result = await executor.execute_tool(
    "calculate_condenser_duty",
    {
        "calculation_method": "cooling_water",
        "cw_inlet_temp": 22.0,
        "cw_outlet_temp": 32.5,
        "cw_flow_rate": 48000.0
    }
)

# Steam side method
result = await executor.execute_tool(
    "calculate_condenser_duty",
    {
        "calculation_method": "steam_side",
        "steam_flow": 450000.0,
        "steam_quality": 0.92,
        "condenser_pressure": 45.0
    }
)
```

---

## 10. assess_condenser_health

### Description

Comprehensive health assessment of condenser system including tubes, air removal system, and instrumentation. Combines multiple diagnostic parameters into overall health score using deterministic scoring algorithm.

### Scoring Algorithm

```
Health_Score = 100 - sum(Deductions)

Deductions:
  - Cleanliness factor < 0.85: (0.85 - CF) * 50
  - Air inleakage > limit: excess * 2
  - Vacuum deviation > 3 mbar: deviation * 3
  - TTD deviation > 2 K: deviation * 5
  - Tube pluggage > 2%: (pluggage - 2) * 8

Status Classification:
  - >= 90: "excellent"
  - 75-89: "good"
  - 60-74: "fair"
  - < 60: "poor"
```

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `condenser_id` | string | Yes | Unique condenser identifier |
| `cleanliness_factor` | number | Yes | Current cleanliness factor (0-1) |
| `air_inleakage_rate` | number | No | Air inleakage rate (kg/hr) |
| `vacuum_deviation` | number | No | Vacuum deviation from expected (mbar) |
| `ttd_deviation` | number | No | TTD deviation from design (K) |
| `tube_pluggage_percent` | number | No | Percentage of tubes plugged |
| `operating_hours_since_overhaul` | number | No | Hours since major overhaul |
| `unit_capacity` | number | No | Unit capacity (MW) |

### Output

```json
{
  "condenser_id": "COND-001",
  "overall_health_score": 78.5,
  "component_scores": {
    "tube_cleanliness": 75.0,
    "air_removal_system": 85.0,
    "vacuum_performance": 80.0,
    "structural_integrity": 90.0
  },
  "health_status": "good",
  "risk_factors": [
    {
      "factor": "declining_cleanliness",
      "severity": "medium",
      "trend": "degrading"
    },
    {
      "factor": "elevated_air_inleakage",
      "severity": "low",
      "trend": "stable"
    }
  ],
  "recommended_actions": [
    "Schedule tube cleaning within 60 days",
    "Conduct ultrasonic leak survey",
    "Review water treatment program"
  ],
  "next_inspection_date": "2025-04-15",
  "timestamp": "2025-01-15T10:30:00Z",
  "provenance_hash": "sha256:bcd890..."
}
```

### Example Usage

```python
result = await executor.execute_tool(
    "assess_condenser_health",
    {
        "condenser_id": "COND-001",
        "cleanliness_factor": 0.80,
        "air_inleakage_rate": 8.5,
        "vacuum_deviation": 4.5,
        "ttd_deviation": 1.8,
        "tube_pluggage_percent": 1.5,
        "operating_hours_since_overhaul": 35000,
        "unit_capacity": 500.0
    }
)

print(f"Health Score: {result['overall_health_score']}")
print(f"Status: {result['health_status']}")
```

---

## Provenance Tracking

All tools generate a provenance hash for audit trail purposes:

```python
def _generate_provenance_hash(self, inputs: Dict, outputs: Dict) -> str:
    """Generate SHA-256 hash for calculation provenance."""
    data = {
        "inputs": inputs,
        "outputs": outputs,
        "timestamp": datetime.utcnow().isoformat(),
        "tool_version": "1.0.0"
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
```

---

## Error Handling

All tools follow consistent error handling:

| Error Type | HTTP Code | Description |
|------------|-----------|-------------|
| `ValueError` | 400 | Invalid input parameters |
| `RuntimeError` | 500 | Tool execution failure |
| `TimeoutError` | 504 | Calculation timeout |

Example error response:

```json
{
  "error": "ValueError",
  "message": "Missing required parameters: ['cw_inlet_temp']",
  "tool": "calculate_heat_transfer_coefficient",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## Performance Metrics

Tool execution metrics are exposed via Prometheus:

```
gl017_calculation_duration_seconds{calculation_type="heat_transfer"}
gl017_calculations_total{tool="calculate_fouling_factor", status="success"}
gl017_calculation_errors_total{tool="optimize_vacuum_pressure", error_type="validation"}
```

---

## Related Documentation

- [README.md](README.md) - Main documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [gl.yaml](gl.yaml) - Agent specification

---

*GL-017 CONDENSYNC Tools Documentation - Version 1.0.0*
