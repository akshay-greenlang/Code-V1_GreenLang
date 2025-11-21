# GL-002: Boiler Efficiency Optimizer - Calculator Modules

## Overview

The GL-002 calculator modules provide comprehensive computational capabilities for boiler efficiency optimization, emissions reduction, and performance analysis. All calculations follow international standards including ASME PTC 4, EPA methods, and ISO 50001.

## Module Structure

```
calculators/
├── __init__.py
├── efficiency_calculator.py    # Core efficiency calculations
├── combustion_optimizer.py     # Combustion optimization algorithms
├── heat_loss_analyzer.py       # Heat loss analysis
├── nox_predictor.py            # NOx emission prediction
├── feedwater_optimizer.py      # Feedwater system optimization
├── sootblowing_optimizer.py    # Soot blowing scheduling
├── load_optimizer.py           # Multi-boiler load distribution
└── utils/
    ├── steam_tables.py         # Steam property calculations
    ├── fuel_properties.py      # Fuel analysis utilities
    └── conversions.py          # Unit conversions
```

## Core Calculators

### 1. Efficiency Calculator

**Purpose:** Calculate boiler efficiency using multiple methods for accuracy validation.

```python
from gl002.calculators import EfficiencyCalculator

# Initialize calculator
calc = EfficiencyCalculator()

# Calculate using direct method
direct_result = calc.calculate_direct(
    steam_flow=50000,      # lb/hr
    steam_pressure=600,     # psig
    steam_temperature=485,  # F
    feedwater_temp=230,     # F
    fuel_flow=3500,        # lb/hr
    fuel_heating_value=18500  # BTU/lb
)

print(f"Direct Method Efficiency: {direct_result['efficiency']:.2f}%")
print(f"Heat Rate: {direct_result['heat_rate']:.0f} BTU/lb")

# Calculate using indirect method
indirect_result = calc.calculate_indirect(
    flue_gas_temp=350,     # F
    ambient_temp=70,       # F
    oxygen_percent=3.5,    # %
    co_ppm=50,            # ppm
    fuel_type='natural_gas'
)

print(f"Indirect Method Efficiency: {indirect_result['efficiency']:.2f}%")
print(f"Stack Loss: {indirect_result['losses']['stack_loss']:.2f}%")
```

**Key Methods:**

- `calculate_direct()`: Input-output method per ASME PTC 4
- `calculate_indirect()`: Heat loss method per BS EN 12953
- `calculate_fuel_to_steam()`: F/S ratio calculation
- `calculate_heat_rate()`: Heat rate in BTU/kWh

### 2. Combustion Optimizer

**Purpose:** Optimize combustion parameters for efficiency and emissions.

```python
from gl002.calculators import CombustionOptimizer

# Initialize optimizer
optimizer = CombustionOptimizer()

# Optimize combustion
result = optimizer.optimize(
    current_o2=4.5,        # %
    current_co=100,        # ppm
    current_nox=45,        # ppm
    load_percent=75,       # %
    fuel_type='natural_gas',
    constraints={
        'max_nox': 30,     # ppm
        'max_co': 50,      # ppm
        'min_o2': 2.0,     # %
    }
)

print(f"Optimal O2: {result['optimal_o2']:.1f}%")
print(f"Optimal air/fuel ratio: {result['air_fuel_ratio']:.2f}")
print(f"Expected efficiency gain: {result['efficiency_gain']:.1f}%")
print(f"NOx reduction: {result['nox_reduction']:.1f}%")

# Get implementation steps
for step in result['implementation_steps']:
    print(f"Step {step['order']}: {step['action']}")
    print(f"  Parameter: {step['parameter']}")
    print(f"  Change: {step['current']} -> {step['target']}")
```

**Optimization Features:**

- Multi-objective optimization (efficiency vs emissions)
- Constraint satisfaction (safety and regulatory)
- Gradual implementation planning
- Real-time adaptation

### 3. Heat Loss Analyzer

**Purpose:** Identify and quantify heat losses for recovery opportunities.

```python
from gl002.calculators import HeatLossAnalyzer

# Initialize analyzer
analyzer = HeatLossAnalyzer()

# Analyze stack losses
stack_loss = analyzer.calculate_stack_loss(
    flue_gas_temp=380,    # F
    ambient_temp=70,      # F
    o2_percent=4.0,       # %
    fuel_heating_value=21500  # BTU/lb
)

print(f"Stack loss: {stack_loss['percentage']:.2f}%")
print(f"Annual cost: ${stack_loss['annual_cost']:,.0f}")
print(f"Recovery potential: {stack_loss['recovery_potential']:.1f}%")

# Analyze radiation losses
radiation_loss = analyzer.calculate_radiation_loss(
    boiler_capacity=100,   # MMBtu/hr
    surface_temp=180,      # F
    ambient_temp=70,       # F
    load_factor=0.75       # fraction
)

print(f"Radiation loss: {radiation_loss['percentage']:.2f}%")
print(f"Insulation savings: ${radiation_loss['insulation_savings']:,.0f}/year")

# Complete heat balance
heat_balance = analyzer.complete_heat_balance(
    operating_data={...},  # Full operating data
    include_minor_losses=True
)

print(f"Total losses: {heat_balance['total_losses']:.2f}%")
print(f"Efficiency: {heat_balance['efficiency']:.2f}%")
```

**Loss Categories:**

- Stack/dry gas losses
- Moisture losses (fuel and air)
- Unburned carbon losses
- Radiation and convection losses
- Blowdown losses
- Unaccounted losses

### 4. NOx Predictor

**Purpose:** Predict and optimize NOx emissions using ML models.

```python
from gl002.calculators import NOxPredictor

# Initialize predictor with trained model
predictor = NOxPredictor(model_path='models/nox_model.pkl')

# Predict NOx emissions
prediction = predictor.predict(
    load=75,              # % MCR
    excess_o2=3.5,        # %
    flame_temp=2800,      # F
    fuel_nitrogen=0.15,   # %
    burner_type='low_nox',
    steam_injection=0.05  # ratio
)

print(f"Predicted NOx: {prediction['nox_ppm']:.1f} ppm")
print(f"Confidence interval: [{prediction['ci_lower']:.1f}, {prediction['ci_upper']:.1f}]")
print(f"Primary factors: {prediction['contributing_factors']}")

# Optimize for NOx reduction
optimization = predictor.optimize_for_target(
    target_nox=25,        # ppm
    current_conditions={...},
    allowed_changes=['excess_o2', 'steam_injection', 'burner_tilt']
)

print(f"Recommended changes:")
for param, change in optimization['changes'].items():
    print(f"  {param}: {change['from']} -> {change['to']}")
print(f"Expected NOx: {optimization['expected_nox']:.1f} ppm")
print(f"Efficiency impact: {optimization['efficiency_impact']:.2f}%")
```

**ML Features:**

- Neural network prediction model
- Real-time adaptation
- Uncertainty quantification
- Sensitivity analysis

### 5. Feedwater Optimizer

**Purpose:** Optimize feedwater temperature and treatment for efficiency.

```python
from gl002.calculators import FeedwaterOptimizer

# Initialize optimizer
fw_optimizer = FeedwaterOptimizer()

# Optimize deaerator
da_result = fw_optimizer.optimize_deaerator(
    feedwater_flow=45000,     # lb/hr
    makeup_percent=10,         # %
    condensate_temp=180,       # F
    steam_pressure=600,        # psig
    current_da_pressure=5      # psig
)

print(f"Optimal DA pressure: {da_result['optimal_pressure']:.1f} psig")
print(f"Optimal DA temperature: {da_result['optimal_temp']:.1f} F")
print(f"Steam consumption: {da_result['steam_required']:.0f} lb/hr")
print(f"O2 removal: {da_result['o2_removal']:.1f} ppb")

# Optimize economizer
econ_result = fw_optimizer.optimize_economizer(
    flue_gas_inlet_temp=450,  # F
    flue_gas_flow=55000,       # lb/hr
    feedwater_flow=45000,      # lb/hr
    feedwater_inlet_temp=230,  # F
    sulfur_in_fuel=0.5         # %
)

print(f"Optimal FW outlet temp: {econ_result['optimal_fw_temp']:.1f} F")
print(f"Heat recovery: {econ_result['heat_recovery']:.1f} MMBtu/hr")
print(f"Minimum metal temp: {econ_result['min_metal_temp']:.1f} F")
print(f"Acid dewpoint: {econ_result['acid_dewpoint']:.1f} F")
```

### 6. Soot Blowing Optimizer

**Purpose:** Optimize soot blowing frequency and sequence.

```python
from gl002.calculators import SootBlowingOptimizer

# Initialize optimizer
sb_optimizer = SootBlowingOptimizer()

# Analyze fouling trend
fouling_analysis = sb_optimizer.analyze_fouling(
    heat_transfer_data=historical_data,
    time_period_days=30
)

print(f"Fouling rate: {fouling_analysis['fouling_rate']:.3f}/day")
print(f"Current fouling factor: {fouling_analysis['current_fouling']:.3f}")
print(f"Efficiency impact: {fouling_analysis['efficiency_loss']:.2f}%")

# Optimize schedule
schedule = sb_optimizer.optimize_schedule(
    fouling_rate=fouling_analysis['fouling_rate'],
    steam_cost=0.015,          # $/lb
    efficiency_value=100000,    # $/% annually
    sootblowing_steam=500       # lb per cycle
)

print(f"Optimal frequency: every {schedule['frequency_hours']:.0f} hours")
print(f"Annual steam cost: ${schedule['steam_cost']:,.0f}")
print(f"Efficiency benefit: ${schedule['efficiency_benefit']:,.0f}")
print(f"Net benefit: ${schedule['net_benefit']:,.0f}")

# Intelligent sequencing
sequence = sb_optimizer.intelligent_sequence(
    fouling_map=current_fouling_distribution,
    available_sootblowers=24
)

for step in sequence['sequence']:
    print(f"Activate sootblower {step['id']} at {step['time']}")
    print(f"  Zone: {step['zone']}, Duration: {step['duration']}s")
```

### 7. Load Optimizer

**Purpose:** Optimize load distribution across multiple boilers.

```python
from gl002.calculators import LoadOptimizer

# Initialize optimizer
load_opt = LoadOptimizer()

# Define boiler fleet
boilers = [
    {'id': 'B1', 'capacity': 100000, 'min_load': 30000, 'efficiency_curve': {...}},
    {'id': 'B2', 'capacity': 75000, 'min_load': 22500, 'efficiency_curve': {...}},
    {'id': 'B3', 'capacity': 50000, 'min_load': 15000, 'efficiency_curve': {...}}
]

# Optimize load distribution
distribution = load_opt.optimize_distribution(
    total_demand=150000,       # lb/hr
    boilers=boilers,
    fuel_cost=5.50,            # $/MMBtu
    startup_cost=500,          # $ per start
    constraints={
        'max_starts_per_day': 2,
        'min_runtime_hours': 4
    }
)

print("Optimal load distribution:")
for boiler_id, load in distribution['loads'].items():
    print(f"  {boiler_id}: {load:,.0f} lb/hr ({load/150000*100:.1f}%)")

print(f"System efficiency: {distribution['system_efficiency']:.2f}%")
print(f"Fuel cost: ${distribution['fuel_cost']:.2f}/hr")
print(f"Compared to equal distribution: ${distribution['savings']:.2f}/hr saved")
```

## Utility Functions

### Steam Tables

```python
from gl002.calculators.utils import SteamTables

# Get steam properties
steam = SteamTables()

# Saturation properties
sat_temp = steam.saturation_temperature(pressure=600)  # psig
sat_pressure = steam.saturation_pressure(temperature=485)  # F

# Enthalpy calculations
h_steam = steam.enthalpy(pressure=600, temperature=485)  # BTU/lb
h_water = steam.enthalpy_liquid(temperature=230)  # BTU/lb

# Specific volume
v_steam = steam.specific_volume(pressure=600, temperature=485)  # ft³/lb
```

### Fuel Properties

```python
from gl002.calculators.utils import FuelProperties

# Get fuel properties
fuel = FuelProperties('natural_gas')

# Heating values
hhv = fuel.higher_heating_value  # BTU/scf
lhv = fuel.lower_heating_value   # BTU/scf

# Combustion calculations
air_required = fuel.theoretical_air(excess_percent=10)  # lb/lb_fuel
products = fuel.combustion_products(excess_air=10)  # composition

# Emissions factors
co2_factor = fuel.co2_emission_factor  # lb CO2/MMBtu
```

### Unit Conversions

```python
from gl002.calculators.utils import convert

# Temperature conversions
celsius = convert(485, 'F', 'C')
kelvin = convert(485, 'F', 'K')

# Pressure conversions
bar = convert(600, 'psig', 'bar')
kpa = convert(600, 'psig', 'kPa')

# Flow conversions
kg_hr = convert(50000, 'lb/hr', 'kg/hr')
```

## Standards Compliance

All calculators comply with:

| Standard | Description | Implementation |
|----------|-------------|----------------|
| ASME PTC 4 | Fired Steam Generators | Efficiency calculations |
| ASME PTC 4.1 | Test Uncertainty | Error propagation |
| EPA Method 3A | O2 and CO2 Determination | Emission calculations |
| EPA Method 7E | NOx Determination | NOx predictions |
| ISO 50001 | Energy Management | Optimization algorithms |
| EN 12952-15 | Boiler Efficiency | European methods |

## Performance Specifications

| Calculator | Operation | Response Time | Accuracy |
|------------|-----------|---------------|----------|
| Efficiency | Single calculation | <50ms | ±0.5% |
| Combustion | Optimization | <2s | ±2% |
| Heat Loss | Full analysis | <100ms | ±1% |
| NOx Predictor | Prediction | <30ms | ±5 ppm |
| Load Optimizer | 5 boilers | <5s | Optimal |

## Testing

```bash
# Run calculator tests
pytest tests/calculators/

# Test specific calculator
pytest tests/calculators/test_efficiency_calculator.py -v

# Performance benchmarks
pytest tests/calculators/benchmarks/ --benchmark-only
```

## API Usage

All calculators are accessible via REST API:

```http
POST /api/v1/gl002/calculate/efficiency
Content-Type: application/json

{
    "method": "direct",
    "parameters": {
        "steam_flow": 50000,
        "fuel_flow": 3500,
        ...
    }
}
```

## Support

For calculator-specific questions:
- Documentation: https://docs.greenlang.io/gl002/calculators
- API Reference: https://api.greenlang.io/gl002/calculators
- Support: gl002-calculators@greenlang.io