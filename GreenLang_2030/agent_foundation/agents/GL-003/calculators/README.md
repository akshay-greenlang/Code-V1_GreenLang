# GL-003 Steam System Calculators

**Zero-Hallucination, Production-Quality Calculation Suite for Steam System Analysis**

## Overview

This calculator suite provides deterministic, bit-perfect calculations for comprehensive steam system analysis. Every calculation is backed by authoritative engineering standards and includes complete provenance tracking for audit trails.

## Key Guarantees

- **Zero Hallucination**: No LLM involvement in calculations - pure physics and mathematics
- **Bit-Perfect Reproducibility**: Same inputs always produce identical outputs
- **Complete Provenance**: SHA-256 hash for every calculation with full audit trail
- **Standards Compliant**: IAPWS-IF97, ASME, ASHRAE, EPA AP-42, GHG Protocol
- **Production Ready**: Comprehensive input validation, error handling, and documentation

## Calculator Modules

### 1. Steam Properties (`steam_properties.py`)

**IAPWS-IF97 Standard Implementation**

Calculates thermodynamic properties of water and steam:
- Saturation properties (temperature ↔ pressure)
- Superheated steam properties
- Enthalpy, entropy, specific volume
- Steam quality (dryness fraction)
- Density and internal energy

```python
from calculators import SteamPropertiesCalculator

calc = SteamPropertiesCalculator()

# Get properties at 10 bar, 200°C
props = calc.properties_from_pressure_temperature(
    pressure_bar=10.0,
    temperature_c=200.0
)

print(f"Enthalpy: {props.enthalpy_kj_kg:.2f} kJ/kg")
print(f"Entropy: {props.entropy_kj_kg_k:.3f} kJ/(kg·K)")
print(f"Region: {props.region}")
```

**Standards**: IAPWS-IF97, ASME Steam Tables

---

### 2. Distribution Efficiency (`distribution_efficiency.py`)

**Heat Transfer and Network Efficiency**

Calculates heat losses in steam distribution networks:
- Pipeline heat loss (conduction, convection, radiation)
- Insulation effectiveness analysis
- Distribution efficiency metrics
- Economic insulation thickness
- Surface temperature calculations

```python
from calculators import DistributionEfficiencyCalculator, PipeSegment

calc = DistributionEfficiencyCalculator()

segments = [
    PipeSegment(
        length_m=100,
        diameter_mm=150,
        insulation_thickness_mm=50,
        insulation_type='mineral_wool',
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

print(f"Distribution Efficiency: {result.distribution_efficiency_percent:.1f}%")
print(f"Heat Loss: {result.total_heat_loss_kw:.1f} kW")
```

**Standards**: ASHRAE Handbook, ISO 12241

---

### 3. Leak Detection (`leak_detection.py`)

**Multi-Method Leak Detection**

Detects steam leaks using:
- Mass balance analysis
- Pressure drop anomaly detection
- Flow rate deviation (statistical process control)
- Leak localization algorithms

```python
from calculators import LeakDetectionCalculator, FlowMeasurement

calc = LeakDetectionCalculator()

inlet = [
    FlowMeasurement("2024-01-01T10:00", 5000, 10.0, 180, "Inlet")
]

outlet = [
    FlowMeasurement("2024-01-01T10:00", 4500, 9.5, 175, "Outlet")
]

result = calc.detect_leaks(
    inlet_measurements=inlet,
    outlet_measurements=outlet
)

if result.leak_detected:
    print(f"LEAK DETECTED - Confidence: {result.confidence_percent:.1f}%")
    print(f"Estimated leak rate: {result.estimated_leak_rate_kg_hr:.1f} kg/hr")
```

**Standards**: ASME PTC 12.4, ISO 20823

---

### 4. Heat Loss Calculator (`heat_loss_calculator.py`)

**Comprehensive Heat Transfer**

Calculates heat losses via:
- Natural and forced convection
- Radiation (Stefan-Boltzmann law)
- Conduction through insulation
- Optimal insulation thickness

```python
from calculators import HeatLossCalculator

calc = HeatLossCalculator()

result = calc.calculate_pipe_heat_loss(
    length_m=50,
    outer_diameter_m=0.2,
    surface_temperature_c=100,
    ambient_temperature_c=20,
    emissivity=0.85
)

print(f"Total Heat Loss: {result.total_heat_loss_w:.1f} W")
print(f"Convection: {result.convection_loss_w:.1f} W")
print(f"Radiation: {result.radiation_loss_w:.1f} W")
```

**Standards**: ASHRAE, Heat Transfer by Holman

---

### 5. Condensate Optimizer (`condensate_optimizer.py`)

**Condensate Recovery Analysis**

Optimizes condensate systems:
- Flash steam generation calculations
- Heat recovery potential
- Optimal return rate determination
- Flash vessel sizing
- Economic analysis

```python
from calculators import CondensateOptimizer, CondensateData

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

print(f"Flash Steam Available: {result.flash_steam_available_kg_hr:.1f} kg/hr")
print(f"Annual Savings Potential: ${result.annual_savings_potential:,.0f}")
```

**Standards**: Spirax Sarco, ASHRAE

---

### 6. Steam Trap Analyzer (`steam_trap_analyzer.py`)

**Trap Performance and Failure Detection**

Analyzes steam trap performance:
- Efficiency calculation by trap type
- Failed trap detection
- Energy loss from failures
- Fleet-wide analysis
- Maintenance prioritization

```python
from calculators import SteamTrapAnalyzer, SteamTrapData

analyzer = SteamTrapAnalyzer()

trap = SteamTrapData(
    trap_type='mechanical',
    steam_pressure_bar=8.0,
    orifice_size_mm=12.0,
    operating_temperature_c=175,
    expected_condensate_load_kg_hr=200,
    trap_condition='blowing_steam'
)

result = analyzer.analyze_trap(trap)

print(f"Trap Efficiency: {result.trap_efficiency_percent:.1f}%")
print(f"Steam Loss: {result.steam_loss_rate_kg_hr:.1f} kg/hr")
print(f"Annual Cost Loss: ${result.annual_cost_loss:,.0f}")
print(f"Priority: {result.replacement_priority}")
```

**Standards**: ASME PTC 12.4, Spirax Sarco

---

### 7. Pressure Analysis (`pressure_analysis.py`)

**Darcy-Weisbach Pressure Drop**

Calculates flow characteristics:
- Pressure drop (Darcy-Weisbach equation)
- Friction factor (Moody diagram correlations)
- Reynolds number
- Velocity analysis
- Pipe sizing optimization
- Erosion risk assessment

```python
from calculators import PressureAnalysisCalculator, PipeFlowData

calc = PressureAnalysisCalculator()

data = PipeFlowData(
    flow_rate_kg_hr=5000,
    pipe_diameter_mm=150,
    pipe_length_m=200,
    pipe_roughness_mm=0.045,
    steam_pressure_bar=10,
    steam_temperature_c=200,
    fittings={'elbow_90': 4, 'valve_gate_open': 2}
)

result = calc.analyze_pressure_drop(data)

print(f"Pressure Drop: {result.pressure_drop_bar:.3f} bar ({result.pressure_drop_percent:.1f}%)")
print(f"Velocity: {result.velocity_m_s:.1f} m/s")
print(f"Velocity OK: {result.is_velocity_acceptable}")
```

**Standards**: ASME B31.1, Crane TP-410, ISO 5167

---

### 8. Emissions Calculator (`emissions_calculator.py`)

**EPA AP-42 Emission Factors**

Calculates emissions from fuel consumption:
- CO2 emissions (IPCC factors)
- NOx emissions
- SOx emissions
- Emission intensity metrics
- Carbon footprint

```python
from calculators import EmissionsCalculator, FuelConsumptionData

calc = EmissionsCalculator()

data = FuelConsumptionData(
    fuel_type='natural_gas',
    fuel_consumption_kg=10000,
    fuel_heating_value_kj_kg=50000,
    boiler_efficiency_percent=85
)

result = calc.calculate_emissions(data)

print(f"CO2 Emissions: {result.co2_emissions_tonnes:.2f} tonnes")
print(f"NOx Emissions: {result.nox_emissions_kg:.2f} kg")
print(f"Intensity: {result.emission_intensity_kg_co2_per_gj:.1f} kg CO2/GJ")
print(f"Source: {result.emission_factor_source}")
```

**Standards**: EPA AP-42, GHG Protocol, IPCC 2006

---

### 9. KPI Calculator (`kpi_calculator.py`)

**Comprehensive Performance Dashboard**

Calculates system-wide KPIs:
- Overall system efficiency
- Distribution efficiency
- Condensate return rate
- Steam trap performance index
- Energy savings opportunities
- Performance benchmarking

```python
from calculators import KPICalculator, SystemMetrics

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
print(f"Distribution Efficiency: {dashboard.distribution_efficiency_percent:.1f}%")
print(f"Condensate Return: {dashboard.condensate_return_rate_percent:.1f}%")
print(f"Performance Rating: {dashboard.performance_rating}")
print(f"Savings Opportunity: {dashboard.total_savings_opportunity_gj:.1f} GJ/year")
```

**Standards**: Industry best practices

---

### 10. Provenance Tracker (`provenance.py`)

**SHA-256 Audit Trail**

Provides complete provenance for all calculations:
- SHA-256 hash generation
- Step-by-step calculation recording
- Tamper detection
- Reproducibility validation

```python
from calculators import ProvenanceTracker

tracker = ProvenanceTracker(
    calculation_id="example_001",
    calculation_type="efficiency_calc",
    version="1.0.0"
)

tracker.record_inputs({'pressure': 10, 'temperature': 200})

tracker.record_step(
    operation="multiply",
    description="Calculate energy",
    inputs={'mass': 100, 'enthalpy': 2800},
    output_value=280000,
    output_name="total_energy",
    formula="E = m * h",
    units="kJ"
)

provenance = tracker.get_provenance_record(280000)
print(f"Provenance Hash: {provenance.provenance_hash}")
```

---

## Complete Example: Steam System Analysis

```python
from calculators import (
    SteamPropertiesCalculator,
    DistributionEfficiencyCalculator,
    LeakDetectionCalculator,
    CondensateOptimizer,
    KPICalculator,
    PipeSegment,
    CondensateData,
    SystemMetrics
)

# 1. Calculate steam properties
steam_calc = SteamPropertiesCalculator()
props = steam_calc.properties_from_pressure_temperature(10.0, 200.0)
print(f"Steam enthalpy: {props.enthalpy_kj_kg:.1f} kJ/kg")

# 2. Analyze distribution efficiency
dist_calc = DistributionEfficiencyCalculator()
segments = [
    PipeSegment(100, 150, 50, 'mineral_wool', 20, 180, 10)
]
dist_result = dist_calc.calculate_distribution_efficiency(
    segments, 5000, props.enthalpy_kj_kg
)
print(f"Distribution efficiency: {dist_result.distribution_efficiency_percent:.1f}%")

# 3. Optimize condensate recovery
cond_data = CondensateData(4000, 140, 5.0, 2.0, 80, 10, 75.0)
cond_optimizer = CondensateOptimizer()
cond_result = cond_optimizer.optimize_condensate_system(cond_data)
print(f"Condensate savings: ${cond_result.annual_savings_potential:,.0f}")

# 4. Calculate overall KPIs
kpi_calc = KPICalculator()
metrics = SystemMetrics(
    fuel_input_gj=1000,
    steam_output_tonnes=120,
    steam_enthalpy_kj_kg=props.enthalpy_kj_kg,
    steam_generated_tonnes=120,
    steam_delivered_tonnes=114,
    condensate_returned_tonnes=100,
    distribution_loss_gj=25,
    trap_losses_kg_hr=50,
    operating_hours=8760,
    number_of_steam_traps=100,
    failed_traps=8
)
dashboard = kpi_calc.calculate_kpis(metrics)
print(f"\n=== SYSTEM DASHBOARD ===")
print(f"Overall Efficiency: {dashboard.overall_system_efficiency_percent:.1f}%")
print(f"Performance Rating: {dashboard.performance_rating.upper()}")
print(f"Savings Opportunity: ${dashboard.estimated_annual_savings:,.0f}/year")
```

## Testing

All calculators include comprehensive test coverage. Run tests with:

```bash
pytest calculators/ -v --cov=calculators --cov-report=html
```

## Quality Standards

- **Precision**: Match regulatory requirements (2-3 decimal places)
- **Reproducibility**: 100% bit-perfect (same input → same output)
- **Provenance**: SHA-256 hash for every calculation
- **Performance**: <5ms per calculation target
- **Test Coverage**: >95% for all modules
- **Zero Hallucination**: NO LLM in calculation path
- **Audit Trail**: Complete documentation of all calculation steps

## Standards Compliance

| Calculator | Standards |
|-----------|-----------|
| Steam Properties | IAPWS-IF97, ASME Steam Tables |
| Distribution | ASHRAE Handbook, ISO 12241 |
| Leak Detection | ASME PTC 12.4, ISO 20823 |
| Heat Loss | ASHRAE, Holman Heat Transfer |
| Condensate | Spirax Sarco, ASHRAE |
| Steam Traps | ASME PTC 12.4, Spirax Sarco |
| Pressure | ASME B31.1, Crane TP-410, ISO 5167 |
| Emissions | EPA AP-42, GHG Protocol, IPCC |

## Authors

- GL-CalculatorEngineer (GreenLang Zero-Hallucination Calculation Engine)

## Version

1.0.0 - Production Release

## License

Proprietary - GreenLang Corporation
