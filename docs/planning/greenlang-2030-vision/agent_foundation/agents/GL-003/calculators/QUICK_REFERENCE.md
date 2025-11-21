# GL-003 Calculators - Quick Reference

**Zero-Hallucination Steam System Calculators**

---

## Import All Calculators

```python
from calculators import (
    # Core utilities
    ProvenanceTracker,

    # Calculators
    SteamPropertiesCalculator,
    DistributionEfficiencyCalculator,
    LeakDetectionCalculator,
    HeatLossCalculator,
    CondensateOptimizer,
    SteamTrapAnalyzer,
    PressureAnalysisCalculator,
    EmissionsCalculator,
    KPICalculator,

    # Data classes
    PipeSegment,
    FlowMeasurement,
    CondensateData,
    SteamTrapData,
    PipeFlowData,
    FuelConsumptionData,
    SystemMetrics
)
```

---

## 1. Steam Properties

```python
calc = SteamPropertiesCalculator()

# Method 1: Get all properties
props = calc.properties_from_pressure_temperature(10.0, 200.0)
# Returns: SteamProperties(enthalpy, entropy, volume, quality, region, ...)

# Method 2: Saturation temperature from pressure
T_sat = calc.saturation_temperature_from_pressure(10.0)  # °C

# Method 3: Saturation pressure from temperature
P_sat = calc.saturation_pressure_from_temperature(180.0)  # bar

# Method 4: Steam quality (dryness fraction)
quality = calc.quality_from_enthalpy_pressure(2500.0, 10.0)  # 0-1
```

---

## 2. Distribution Efficiency

```python
calc = DistributionEfficiencyCalculator()

segments = [
    PipeSegment(
        length_m=100,
        diameter_mm=150,
        insulation_thickness_mm=50,
        insulation_type='mineral_wool',  # or calcium_silicate, cellular_glass
        ambient_temperature_c=20,
        steam_temperature_c=180,
        steam_pressure_bar=10
    )
]

result = calc.calculate_distribution_efficiency(
    pipe_segments=segments,
    steam_flow_rate_kg_hr=5000,
    steam_enthalpy_inlet_kj_kg=2800
)

print(f"Efficiency: {result.distribution_efficiency_percent:.1f}%")
print(f"Heat Loss: {result.total_heat_loss_kw:.1f} kW")
print(f"Annual Cost: ${result.annual_cost_loss:,.0f}")
```

---

## 3. Leak Detection

```python
calc = LeakDetectionCalculator()

inlet = [FlowMeasurement("2024-01-01T10:00", 5000, 10.0, 180, "Inlet")]
outlet = [FlowMeasurement("2024-01-01T10:00", 4700, 9.5, 175, "Outlet")]

result = calc.detect_leaks(
    inlet_measurements=inlet,
    outlet_measurements=outlet,
    expected_pressure_drop_bar=0.5
)

if result.leak_detected:
    print(f"LEAK! Confidence: {result.confidence_percent:.1f}%")
    print(f"Rate: {result.estimated_leak_rate_kg_hr:.1f} kg/hr")
```

---

## 4. Heat Loss

```python
calc = HeatLossCalculator()

result = calc.calculate_pipe_heat_loss(
    length_m=50,
    outer_diameter_m=0.2,
    surface_temperature_c=100,
    ambient_temperature_c=20,
    emissivity=0.85
)

print(f"Total: {result.total_heat_loss_w:.1f} W")
print(f"Convection: {result.convection_loss_w:.1f} W")
print(f"Radiation: {result.radiation_loss_w:.1f} W")

# Calculate insulation thickness
thickness_mm = calc.calculate_insulation_thickness(
    inner_diameter_mm=150,
    steam_temperature_c=180,
    ambient_temperature_c=20,
    target_surface_temp_c=50,
    k_insulation_w_mk=0.045
)
```

---

## 5. Condensate Optimizer

```python
optimizer = CondensateOptimizer()

data = CondensateData(
    condensate_flow_rate_kg_hr=4000,
    condensate_temperature_c=140,
    condensate_pressure_bar=5.0,
    flash_vessel_pressure_bar=2.0,
    feedwater_temperature_c=80,
    steam_generation_pressure_bar=10,
    return_rate_percent=75.0
)

result = optimizer.optimize_condensate_system(data)

print(f"Flash Steam: {result.flash_steam_available_kg_hr:.1f} kg/hr")
print(f"Savings: ${result.annual_savings_potential:,.0f}")
print(f"Optimal Return Rate: {result.optimal_return_rate_percent:.1f}%")
```

---

## 6. Steam Trap Analyzer

```python
analyzer = SteamTrapAnalyzer()

# Single trap
trap = SteamTrapData(
    trap_type='mechanical',  # or thermostatic, thermodynamic, inverted_bucket
    steam_pressure_bar=8.0,
    orifice_size_mm=12.0,
    operating_temperature_c=175,
    expected_condensate_load_kg_hr=200,
    trap_condition='blowing_steam'  # or operational, plugged
)

result = analyzer.analyze_trap(trap)

print(f"Efficiency: {result.trap_efficiency_percent:.1f}%")
print(f"Loss: {result.steam_loss_rate_kg_hr:.1f} kg/hr")
print(f"Annual Cost: ${result.annual_cost_loss:,.0f}")
print(f"Action: {result.recommended_action}")

# Fleet analysis
traps = [trap1, trap2, trap3, ...]
fleet = analyzer.analyze_trap_population(traps)
print(f"Failure Rate: {fleet['failure_rate_percent']:.1f}%")
```

---

## 7. Pressure Analysis

```python
calc = PressureAnalysisCalculator()

data = PipeFlowData(
    flow_rate_kg_hr=5000,
    pipe_diameter_mm=150,
    pipe_length_m=200,
    pipe_roughness_mm=0.045,  # Commercial steel
    steam_pressure_bar=10,
    steam_temperature_c=200,
    fittings={'elbow_90': 4, 'valve_gate_open': 2}
)

result = calc.analyze_pressure_drop(data)

print(f"ΔP: {result.pressure_drop_bar:.3f} bar ({result.pressure_drop_percent:.1f}%)")
print(f"Velocity: {result.velocity_m_s:.1f} m/s")
print(f"Re: {result.reynolds_number:.0f}")
print(f"Acceptable: {result.is_pressure_drop_acceptable}")
```

**Fitting Types**: elbow_90, elbow_45, tee_line_flow, tee_branch_flow, valve_gate_open, valve_globe_open, valve_ball_open, valve_butterfly_open

---

## 8. Emissions Calculator

```python
calc = EmissionsCalculator()

data = FuelConsumptionData(
    fuel_type='natural_gas',  # or fuel_oil, coal_bituminous, biomass_wood
    fuel_consumption_kg=10000,
    fuel_heating_value_kj_kg=50000,
    boiler_efficiency_percent=85
)

result = calc.calculate_emissions(data)

print(f"CO2: {result.co2_emissions_tonnes:.2f} tonnes")
print(f"NOx: {result.nox_emissions_kg:.2f} kg")
print(f"SOx: {result.sox_emissions_kg:.2f} kg")
print(f"Intensity: {result.emission_intensity_kg_co2_per_gj:.1f} kg CO2/GJ")
```

---

## 9. KPI Calculator

```python
calc = KPICalculator()

metrics = SystemMetrics(
    fuel_input_gj=1000,
    steam_output_tonnes=120,
    steam_enthalpy_kj_kg=2800,
    steam_generated_tonnes=120,
    steam_delivered_tonnes=114,
    condensate_returned_tonnes=100,
    distribution_loss_gj=25,
    trap_losses_kg_hr=50,
    operating_hours=8760,
    number_of_steam_traps=100,
    failed_traps=8
)

dashboard = calc.calculate_kpis(metrics)

print(f"Overall Efficiency: {dashboard.overall_system_efficiency_percent:.1f}%")
print(f"Distribution: {dashboard.distribution_efficiency_percent:.1f}%")
print(f"Condensate Return: {dashboard.condensate_return_rate_percent:.1f}%")
print(f"Trap Failure: {dashboard.steam_trap_failure_rate_percent:.1f}%")
print(f"Performance: {dashboard.performance_rating.upper()}")
print(f"Savings: ${dashboard.estimated_annual_savings:,.0f}/year")
```

---

## 10. Provenance Tracking

```python
tracker = ProvenanceTracker(
    calculation_id="unique_id",
    calculation_type="description",
    version="1.0.0"
)

# Record inputs
tracker.record_inputs({'pressure': 10, 'temperature': 200})

# Record calculation step
tracker.record_step(
    operation="multiply",
    description="Calculate energy",
    inputs={'mass': 100, 'enthalpy': 2800},
    output_value=280000,
    output_name="energy",
    formula="E = m * h",
    units="kJ"
)

# Get provenance record
provenance = tracker.get_provenance_record(final_result)
print(f"Hash: {provenance.provenance_hash}")

# Validate
from calculators import ProvenanceValidator
is_valid = ProvenanceValidator.validate_hash(provenance)
```

---

## Complete Example

```python
# 1. Calculate steam properties
steam_calc = SteamPropertiesCalculator()
props = steam_calc.properties_from_pressure_temperature(10.0, 200.0)

# 2. Analyze distribution
dist_calc = DistributionEfficiencyCalculator()
segments = [PipeSegment(100, 150, 50, 'mineral_wool', 20, 180, 10)]
dist_result = dist_calc.calculate_distribution_efficiency(
    segments, 5000, props.enthalpy_kj_kg
)

# 3. Check for leaks
leak_calc = LeakDetectionCalculator()
inlet = [FlowMeasurement("2024-01-01T10:00", 5000, 10, 180, "Inlet")]
outlet = [FlowMeasurement("2024-01-01T10:00", 4800, 9.5, 175, "Outlet")]
leak_result = leak_calc.detect_leaks(inlet, outlet)

# 4. Optimize condensate
cond_opt = CondensateOptimizer()
cond_data = CondensateData(4000, 140, 5, 2, 80, 10, 75)
cond_result = cond_opt.optimize_condensate_system(cond_data)

# 5. Calculate KPIs
kpi_calc = KPICalculator()
metrics = SystemMetrics(1000, 120, props.enthalpy_kj_kg, 120, 114, 100, 25, 50, 8760, 100, 8)
dashboard = kpi_calc.calculate_kpis(metrics)

# Print summary
print("=== STEAM SYSTEM ANALYSIS ===")
print(f"Steam Properties: {props.enthalpy_kj_kg:.1f} kJ/kg @ {props.region}")
print(f"Distribution Eff: {dist_result.distribution_efficiency_percent:.1f}%")
print(f"Leak Status: {'DETECTED' if leak_result.leak_detected else 'OK'}")
print(f"Condensate Return: {cond_result.optimal_return_rate_percent:.1f}%")
print(f"Overall Performance: {dashboard.performance_rating.upper()}")
print(f"Savings Opportunity: ${dashboard.estimated_annual_savings:,.0f}/year")
```

---

## Common Conversions

```python
# Pressure
bar_to_psi = lambda bar: bar * 14.5038
psi_to_bar = lambda psi: psi / 14.5038

# Temperature
c_to_f = lambda c: c * 9/5 + 32
f_to_c = lambda f: (f - 32) * 5/9

# Mass flow
kg_hr_to_lb_hr = lambda kg_hr: kg_hr * 2.20462
lb_hr_to_kg_hr = lambda lb_hr: lb_hr / 2.20462

# Energy
kj_to_btu = lambda kj: kj * 0.947817
btu_to_kj = lambda btu: btu / 0.947817
```

---

## Typical Values

| Parameter | Typical Range | Unit |
|-----------|---------------|------|
| Boiler efficiency | 75-90 | % |
| Distribution efficiency | 90-98 | % |
| Condensate return rate | 70-95 | % |
| Steam trap failure rate | 3-25 | % |
| Steam velocity | 20-50 | m/s |
| Insulation thickness | 25-100 | mm |
| Pressure drop (acceptable) | <5 | % |

---

**Version**: 1.0.0
**Author**: GL-CalculatorEngineer
**Standards**: IAPWS-IF97, ASME, ASHRAE, EPA AP-42
