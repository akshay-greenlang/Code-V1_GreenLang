# GL-003 UNIFIEDSTEAM

Unified Steam System Optimization Agent for industrial steam network optimization, predictive maintenance, and emissions reduction.

## Overview

GL-003 UNIFIEDSTEAM consolidates multiple steam system optimization capabilities:
- **GL-008**: Steam trap diagnostics and predictive maintenance
- **GL-012**: Desuperheater optimization
- **GL-017**: Condensate recovery optimization

### Key Features

- **IAPWS-IF97 Thermodynamics**: High-precision steam property calculations compliant with international standards
- **Enthalpy Balance**: Real-time mass and energy balance computations for steam networks
- **Causal Root Cause Analysis**: AI-powered fault diagnosis with counterfactual reasoning
- **Explainability**: SHAP/LIME feature attribution for all ML predictions
- **Climate Intelligence**: GHG Protocol compliant emissions tracking and M&V reporting
- **MLOps Governance**: Model cards, versioning, drift detection, and deployment governance

## Architecture

```
GL-003_UnifiedSteam/
├── thermodynamics/      # IAPWS-IF97 steam property calculations
├── calculators/         # Enthalpy balance, KPI computation
├── causal/              # Root cause analysis, counterfactuals
├── control/             # Process control logic
├── safety/              # Safety envelope enforcement
├── explainability/      # SHAP/LIME feature attribution
├── optimization/        # Setpoint optimization
├── climate/             # Emissions, M&V, sustainability
├── mlops/               # Model governance, monitoring
├── schemas/             # Kafka/Avro schema definitions
├── api/                 # REST, GraphQL, gRPC interfaces
├── monitoring/          # Metrics, alerts, observability
├── audit/               # Compliance audit trail
├── integration/         # OPC-UA, historian connectors
├── streaming/           # Kafka producers/consumers
└── deployment/          # Docker, Kubernetes, Helm
```

## Installation

```bash
# Standard installation
pip install gl003-unifiedsteam

# Development installation
pip install -e .[dev]

# All optional dependencies
pip install -e .[all]
```

## Quick Start

### Steam Property Calculation

```python
from decimal import Decimal
from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
    compute_properties_pt,
)

# Calculate steam properties at 4 MPa, 400°C
result = compute_properties_pt(
    pressure_kpa=Decimal("4000"),
    temperature_c=Decimal("400"),
)

print(f"Enthalpy: {result.enthalpy_kj_kg} kJ/kg")
print(f"Entropy: {result.entropy_kj_kg_k} kJ/kg-K")
print(f"Superheat: {result.superheat_c} °C")
```

### Emissions Calculation

```python
from decimal import Decimal
from GL_Agents.GL003_UnifiedSteam.climate.co2e_calculator import CO2eCalculator
from GL_Agents.GL003_UnifiedSteam.climate.emission_factors import FuelType

calculator = CO2eCalculator()

result = calculator.calculate_fuel_emissions(
    fuel_type=FuelType.NATURAL_GAS,
    fuel_consumption_gj=Decimal("1000"),
)

print(f"CO2e: {result.total_co2e_kg} kg")
```

## API Services

### gRPC Services

- `SteamPropertiesService`: Thermodynamic calculations
- `OptimizationService`: Setpoint recommendations
- `DiagnosticsService`: Steam trap diagnostics
- `StreamingService`: Real-time data streaming
- `RCAService`: Root cause analysis

### REST API

```bash
# Start the API server
gl003-server --host 0.0.0.0 --port 8000

# Calculate steam properties
curl -X POST http://localhost:8000/api/v1/properties \
  -H "Content-Type: application/json" \
  -d '{"pressure_kpa": 4000, "temperature_c": 400}'
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m performance    # Performance benchmarks
pytest -m validation     # IF97 reference validation
pytest -m property       # Property-based tests

# With coverage
pytest --cov=GL_Agents --cov-report=html
```

## Performance Targets

| Metric | Target |
|--------|--------|
| IF97 property calculation | < 1 ms |
| Saturation lookup | < 0.5 ms |
| Enthalpy balance | < 2 ms |
| Recommendation generation | < 5 ms |
| Schema serialization | < 0.1 ms |
| Concurrent throughput | > 500/sec |

## Climate Intelligence

Supports reporting standards:
- GHG Protocol (Scope 1, 2, 3)
- ISO 14064
- IPMVP for M&V
- CDP/TCFD
- EU ETS

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

# Generate gRPC stubs
python api/protos/generate_stubs.py
```

## License

Proprietary - GreenLang

## Contact

- Team: gl003@greenlang.io
- Documentation: https://docs.greenlang.io/gl003
