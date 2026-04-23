# GL-005 Calculator Quick Start Guide

## Installation & Import

```python
# Import all calculators
from calculators import (
    CombustionStabilityCalculator,
    FuelAirOptimizer,
    HeatOutputCalculator,
    PIDController,
    SafetyValidator,
    EmissionsCalculator
)

# Import input/output models
from calculators import (
    StabilityInput, StabilityResult,
    OptimizerInput, OptimizerResult,
    HeatOutputInput, HeatOutputResult,
    PIDInput, PIDOutput,
    SafetyValidatorInput, SafetyValidatorOutput, SafetyLimits,
    EmissionsInput, EmissionsResult
)

# Import enums
from calculators import (
    FuelType, ControlMode, SafetyLevel, ComplianceStatus
)
```

---

## 1. Combustion Stability Calculator

**Purpose:** Check if combustion is stable

```python
calc = CombustionStabilityCalculator()

input_data = StabilityInput(
    temperature_readings=[950, 955, 948, 952, 951],  # Last 5 readings
    temperature_setpoint=950.0,
    pressure_readings=[1000, 1005, 998, 1002, 1001],
    pressure_setpoint=1000.0,
    sampling_rate_hz=10.0,
    fuel_flow_rate=400.0,
    air_flow_rate=6000.0
)

result = calc.calculate_stability_index(input_data)

# Check results
if result.stability_index > 0.85:
    print("✅ STABLE")
elif result.requires_intervention:
    print("⚠️  INTERVENTION NEEDED")
    print(result.recommended_actions)
```

---

## 2. Fuel-Air Optimizer

**Purpose:** Find optimal fuel-air ratio

```python
optimizer = FuelAirOptimizer()

input_data = OptimizerInput(
    target_heat_output_kw=5000.0,
    fuel_type=FuelType.NATURAL_GAS,
    fuel_heating_value_mj_per_kg=50.0,
    fuel_composition={'C': 75, 'H': 25, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
    current_fuel_flow_kg_per_hr=400.0,
    current_air_flow_kg_per_hr=6000.0,
    max_fuel_flow_kg_per_hr=500.0
)

result = optimizer.calculate_optimal_ratio(input_data)

# Apply adjustments
fuel_adjustment = result.fuel_flow_adjustment_percent
air_adjustment = result.air_flow_adjustment_percent

print(f"Adjust fuel: {fuel_adjustment:+.1f}%")
print(f"Adjust air: {air_adjustment:+.1f}%")
```

---

## 3. Heat Output Calculator

**Purpose:** Calculate actual heat output and efficiency

```python
calc = HeatOutputCalculator()

input_data = HeatOutputInput(
    fuel_flow_rate_kg_per_hr=400.0,
    fuel_lower_heating_value_mj_per_kg=50.0,
    air_flow_rate_kg_per_hr=6000.0,
    flue_gas_temperature_c=250.0,
    flue_gas_o2_percent=4.0,
    flue_gas_co_ppm=50.0,
    ambient_temperature_c=25.0,
    target_heat_output_kw=5000.0
)

result = calc.calculate_heat_output(input_data)

print(f"Heat Output: {result.net_heat_output_kw:.2f} kW")
print(f"Efficiency: {result.net_thermal_efficiency_percent:.2f}%")
print(f"Stack Loss: {result.stack_loss_percent:.2f}%")
```

---

## 4. PID Controller

**Purpose:** Control temperature/pressure

```python
controller = PIDController(kp=1.0, ki=0.1, kd=0.01)

input_data = PIDInput(
    setpoint=950.0,
    process_variable=945.0,  # Current temperature
    timestamp=1.0,
    kp=1.0,
    ki=0.1,
    kd=0.01,
    output_min=0.0,
    output_max=100.0,
    control_mode=ControlMode.AUTO
)

result = controller.calculate_control_output(input_data)

# Apply control output
actuator_position = result.control_output
print(f"Set actuator to: {actuator_position:.2f}%")
```

---

## 5. Safety Validator

**Purpose:** Check all safety interlocks

```python
validator = SafetyValidator()

limits = SafetyLimits(
    max_combustion_temperature=1500.0,
    max_fuel_flow_rate=500.0,
    max_air_flow_rate=7500.0,
    max_co_ppm=400.0
)

input_data = SafetyValidatorInput(
    combustion_temperature_c=1200.0,
    flue_gas_temperature_c=250.0,
    combustion_pressure_pa=1000.0,
    fuel_supply_pressure_pa=500000.0,
    fuel_flow_rate_kg_per_hr=400.0,
    air_flow_rate_kg_per_hr=6000.0,
    o2_percent=4.0,
    co_ppm=50.0,
    safety_limits=limits,
    burner_firing=True,
    flame_detected=True
)

result = validator.validate_all_safety_interlocks(input_data)

if not result.is_safe_to_operate:
    print("🚨 UNSAFE - SHUTDOWN REQUIRED")
    print(result.required_actions)
elif result.safety_level == SafetyLevel.ALARM:
    print("⚠️  ALARMS ACTIVE")
else:
    print("✅ SAFE TO OPERATE")
```

---

## 6. Emissions Calculator

**Purpose:** Calculate emissions and check compliance

```python
calc = EmissionsCalculator()

input_data = EmissionsInput(
    fuel_type='natural_gas',
    fuel_flow_rate_kg_per_hr=400.0,
    fuel_properties={'C': 75, 'H': 25, 'S': 0, 'N': 0, 'ash': 0, 'moisture': 0},
    fuel_heating_value_mj_per_kg=50.0,
    air_flow_rate_kg_per_hr=6000.0,
    combustion_temperature_c=1200.0,
    excess_air_percent=15.0,
    flue_gas_o2_percent=4.0,
    flue_gas_co_ppm=50.0,
    flue_gas_temperature_c=250.0,
    emission_limits={'nox_mg_per_nm3': 200, 'co_mg_per_nm3': 100}
)

result = calc.calculate_all_emissions(input_data)

print(f"NOx: {result.nox_mg_per_nm3:.2f} mg/Nm³")
print(f"CO: {result.co_mg_per_nm3:.2f} mg/Nm³")
print(f"CO2: {result.co2_tonnes_per_year:.2f} tonnes/year")
print(f"Compliance: {result.overall_compliance}")
```

---

## Complete Control Cycle Example

```python
class CombustionController:
    def __init__(self):
        self.stability = CombustionStabilityCalculator()
        self.optimizer = FuelAirOptimizer()
        self.heat = HeatOutputCalculator()
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
        self.safety = SafetyValidator()
        self.emissions = EmissionsCalculator()

    def control_cycle(self, sensor_data):
        # 1. SAFETY FIRST (always check first)
        safety_result = self.safety.validate_all_safety_interlocks(
            sensor_data.to_safety_input()
        )

        if not safety_result.is_safe_to_operate:
            return self.emergency_shutdown(safety_result)

        # 2. Check stability
        stability_result = self.stability.calculate_stability_index(
            sensor_data.to_stability_input()
        )

        # 3. Optimize if unstable
        if stability_result.requires_intervention:
            optimizer_result = self.optimizer.calculate_optimal_ratio(
                sensor_data.to_optimizer_input()
            )
            self.apply_adjustments(optimizer_result)

        # 4. PID control
        pid_output = self.pid.calculate_control_output(
            sensor_data.to_pid_input()
        )

        # 5. Monitor performance
        heat_result = self.heat.calculate_heat_output(
            sensor_data.to_heat_input()
        )

        # 6. Track emissions
        emissions_result = self.emissions.calculate_all_emissions(
            sensor_data.to_emissions_input()
        )

        return {
            'safety': safety_result,
            'stability': stability_result,
            'pid': pid_output,
            'heat': heat_result,
            'emissions': emissions_result
        }
```

---

## Common Patterns

### Pattern 1: Safety Check Before Everything
```python
# ALWAYS check safety first
safety_result = validator.validate_all_safety_interlocks(input_data)

if not safety_result.is_safe_to_operate:
    # STOP - execute shutdown
    execute_shutdown(safety_result.shutdown_sequence)
    return
```

### Pattern 2: Stability-Based Control
```python
stability_result = calc.calculate_stability_index(input_data)

if stability_result.stability_index < 0.7:
    # Unstable - optimize fuel-air ratio
    optimizer_result = optimizer.calculate_optimal_ratio(opt_input)
    apply_adjustments(optimizer_result)
```

### Pattern 3: PID with Anti-Windup
```python
pid_input = PIDInput(
    setpoint=950.0,
    process_variable=current_temp,
    timestamp=current_time,
    kp=1.0, ki=0.1, kd=0.01,
    enable_anti_windup=True,  # Important!
    control_mode=ControlMode.AUTO
)

result = controller.calculate_control_output(pid_input)
```

### Pattern 4: Emission Monitoring
```python
# Calculate and check compliance
emissions_result = calc.calculate_all_emissions(input_data)

if emissions_result.exceeds_any_limit:
    # Alert operator
    send_alarm(emissions_result.violations)

    # Reduce load or adjust combustion
    adjust_for_emissions(emissions_result)
```

---

## Quick Troubleshooting

### Problem: Unstable Combustion
```python
# Check stability
stability_result = calc.calculate_stability_index(input_data)

if stability_result.blowout_risk_score > 0.7:
    # Increase fuel or decrease air
    pass
elif stability_result.flashback_risk_score > 0.7:
    # Decrease fuel or increase air
    pass
```

### Problem: Low Efficiency
```python
# Calculate heat output
heat_result = calc.calculate_heat_output(input_data)

if heat_result.stack_loss_percent > 15:
    # Reduce flue gas temperature
    # Check for excess air
    pass
```

### Problem: High Emissions
```python
# Check emissions
emissions_result = calc.calculate_all_emissions(input_data)

if emissions_result.nox_mg_per_nm3 > 200:
    # Reduce combustion temperature
    # Reduce excess air
    pass
```

---

## Performance Tips

1. **Reuse calculator instances** (don't create new ones each cycle)
2. **Run safety check first** (fastest, most critical)
3. **Skip optimization if stable** (saves time)
4. **Use measurement data when available** (more accurate than models)
5. **Cache emission factors** (avoid repeated lookups)

---

## Common Mistakes to Avoid

❌ **Don't skip safety checks**
```python
# BAD
result = optimizer.calculate_optimal_ratio(input_data)
```

✅ **Always check safety first**
```python
# GOOD
safety_result = validator.validate_all_safety_interlocks(safety_input)
if safety_result.is_safe_to_operate:
    result = optimizer.calculate_optimal_ratio(input_data)
```

❌ **Don't ignore warnings**
```python
# BAD
result = calc.calculate_stability_index(input_data)
# Ignore result.requires_intervention
```

✅ **Act on recommendations**
```python
# GOOD
result = calc.calculate_stability_index(input_data)
if result.requires_intervention:
    for action in result.recommended_actions:
        execute_action(action)
```

---

## Getting Help

- **Documentation:** `README.md` (comprehensive API docs)
- **Examples:** See sections above
- **Standards:** Referenced in each module docstring
- **Issues:** Report via project tracker

---

**Quick Start Version:** 1.0.0
**Last Updated:** 2025-11-18
**Status:** ✅ Production Ready
