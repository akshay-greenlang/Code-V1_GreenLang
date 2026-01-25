# GL-004 BURNMASTER

Burner Optimization Agent for industrial combustion control, air-fuel ratio optimization, and emissions reduction.

## Overview

GL-004 BURNMASTER is a specialized AI agent for optimizing burner operations in industrial furnaces, boilers, and heaters. It provides real-time advisory and closed-loop control support for combustion efficiency maximization while minimizing emissions.

**Note:** This agent has been consolidated into GL-018 UNIFIEDCOMBUSTION as part of the GreenLang agent consolidation initiative.

### Key Features

- **Air-Fuel Ratio Optimization**: Real-time optimization of excess O2 and lambda values for optimal combustion
- **Flame Stability Monitoring**: Advanced flame scanner signal analysis with stability index tracking
- **Emissions Control**: NOx and CO prediction with reduction strategies
- **Turndown Optimization**: Efficient operation across the full burner turndown range
- **Explainability**: SHAP/LIME feature attribution for all recommendations
- **Zero-Hallucination**: Deterministic calculations using physics-based models

## Architecture

```
GL-004_Burnmaster/
+-- combustion/           # Combustion calculations (stoichiometry, efficiency)
+-- calculators/          # Air-fuel ratio, emissions, flame stability calculators
+-- optimization/         # Optimization engines (air-fuel, NOx, turndown)
+-- control/              # Process control logic (O2 trim, damper control)
+-- safety/               # Safety envelope enforcement
+-- explainability/       # SHAP/LIME feature attribution
+-- causal/               # Root cause analysis for combustion issues
+-- climate/              # Emissions reporting, M&V
+-- ml_models/            # ML model management and inference
+-- schemas/              # Kafka/Avro schema definitions
+-- api/                  # REST, GraphQL, gRPC interfaces
+-- monitoring/           # Metrics, alerts, observability
+-- audit/                # Compliance audit trail
+-- integration/          # OPC-UA, DCS, CEMS connectors
+-- streaming/            # Kafka producers/consumers
+-- deployment/           # Docker, Kubernetes, Helm
+-- tests/                # Unit, integration, and validation tests
+-- docs/                 # Documentation
```

## Installation

```bash
# Standard installation
pip install gl004-burnmaster

# Development installation
pip install -e .[dev]

# All optional dependencies
pip install -e .[all]
```

## Quick Start

### Air-Fuel Ratio Calculation

```python
from decimal import Decimal
from GL_Agents.GL004_Burnmaster.combustion.stoichiometry import (
    compute_stoichiometric_ratio,
    compute_excess_air,
)

# Calculate stoichiometric air-fuel ratio for natural gas
stoich_ratio = compute_stoichiometric_ratio(
    fuel_type="natural_gas",
    fuel_composition={
        "CH4": Decimal("0.95"),
        "C2H6": Decimal("0.03"),
        "C3H8": Decimal("0.01"),
        "N2": Decimal("0.01"),
    }
)

print(f"Stoichiometric A/F Ratio: {stoich_ratio}")

# Calculate excess air from O2 measurement
excess_air = compute_excess_air(
    o2_percent=Decimal("3.0"),
    fuel_type="natural_gas",
)

print(f"Excess Air: {excess_air}%")
```

### Emissions Prediction

```python
from GL_Agents.GL004_Burnmaster.calculators.emissions_calculator import EmissionsCalculator

calculator = EmissionsCalculator()

result = calculator.predict_nox(
    flame_temperature_c=Decimal("1800"),
    excess_air_percent=Decimal("15"),
    fuel_nitrogen_ppm=Decimal("50"),
    residence_time_ms=Decimal("100"),
)

print(f"Predicted NOx: {result.nox_ppm} ppm")
print(f"Confidence: {result.confidence_percent}%")
```

### Optimization Recommendations

```python
from GL_Agents.GL004_Burnmaster.optimization.air_fuel_optimizer import AirFuelOptimizer

optimizer = AirFuelOptimizer()

recommendation = optimizer.optimize(
    current_o2_percent=Decimal("4.5"),
    current_co_ppm=Decimal("50"),
    current_nox_ppm=Decimal("80"),
    burner_load_percent=Decimal("75"),
)

print(f"Recommended O2: {recommendation.target_o2_percent}%")
print(f"Expected NOx Reduction: {recommendation.expected_nox_reduction_percent}%")
print(f"Expected Efficiency Gain: {recommendation.expected_efficiency_gain_percent}%")
```

## API Services

### gRPC Services

- `CombustionService`: Air-fuel ratio and combustion calculations
- `OptimizationService`: Setpoint recommendations
- `EmissionsService`: NOx/CO predictions and reduction strategies
- `DiagnosticsService`: Flame stability and burner diagnostics
- `StreamingService`: Real-time data streaming

### REST API

```bash
# Start the API server
gl004-server --host 0.0.0.0 --port 8000

# Calculate excess air
curl -X POST http://localhost:8000/api/v1/combustion/excess-air \
  -H "Content-Type: application/json" \
  -d '{"o2_percent": 3.0, "fuel_type": "natural_gas"}'

# Get optimization recommendation
curl -X POST http://localhost:8000/api/v1/optimization/air-fuel \
  -H "Content-Type: application/json" \
  -d '{"current_o2_percent": 4.5, "current_nox_ppm": 80, "burner_load_percent": 75}'
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m performance    # Performance benchmarks
pytest -m validation     # Combustion equation validation

# With coverage
pytest --cov=GL_Agents --cov-report=html
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Stoichiometric calculation | < 1 ms |
| Excess air calculation | < 0.5 ms |
| NOx prediction | < 5 ms |
| Optimization recommendation | < 10 ms |
| Schema serialization | < 0.1 ms |
| Concurrent throughput | > 500/sec |

## Combustion Theory

### Stoichiometric Combustion

For natural gas (primarily methane):
```
CH4 + 2O2 -> CO2 + 2H2O
```

Stoichiometric air-fuel ratio (by mass) for methane: 17.2:1

### Excess Air

```
Excess Air (%) = (O2_measured / (21 - O2_measured)) * 100
```

### Lambda (Air-Fuel Equivalence Ratio)

```
Lambda = Actual A/F Ratio / Stoichiometric A/F Ratio
```

- Lambda = 1.0: Stoichiometric combustion
- Lambda > 1.0: Lean combustion (excess air)
- Lambda < 1.0: Rich combustion (fuel-rich)

### Combustion Efficiency

```
Efficiency (%) = 100 - Stack Losses - Radiation Losses
```

Where stack losses include:
- Dry flue gas losses
- Moisture losses (fuel H2 + combustion air humidity)
- Unburned fuel losses (CO)

## Safety Features

- **Flame Failure Detection**: Real-time flame scanner monitoring
- **CO High Alarm**: Automatic alert on high CO levels
- **Excess Air Limits**: Prevents unsafe fuel-rich operation
- **Rate Limiting**: Prevents rapid setpoint changes
- **Interlock Integration**: DCS safety interlock coordination

## Climate Intelligence

Supports emissions reporting for:
- GHG Protocol (Scope 1)
- ISO 14064
- EPA 40 CFR Part 98
- EU ETS
- TCFD

## Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run linter
ruff check .

# Run type checker
mypy GL_Agents

# Format code
black .

# Run security scan
bandit -r GL_Agents

# Generate gRPC stubs
python api/protos/generate_stubs.py
```

## Contributing

1. Follow GreenLang coding standards
2. Ensure 85%+ test coverage
3. Add docstrings for all public methods
4. Include type hints for all functions
5. Run pre-commit hooks before committing

## License

Proprietary - GreenLang

## Contact

- Team: gl004@greenlang.io
- Documentation: https://docs.greenlang.io/gl004
