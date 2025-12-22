# GL-002 Flameguard

**Boiler Efficiency & Combustion Optimization Agent**

[![GreenLang Framework](https://img.shields.io/badge/GreenLang-v1.0-green.svg)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![ASME PTC 4.1](https://img.shields.io/badge/ASME-PTC%204.1-orange.svg)]()

## Overview

GL-002_Flameguard is an AI-powered agent for boiler efficiency optimization and combustion control. It implements ASME PTC 4.1 efficiency calculations, EPA emission factors, and O2 trim control with safety interlocks per NFPA 85 and IEC 61511.

## Key Features

- **ASME PTC 4.1 Efficiency**: Direct and indirect method efficiency calculations
- **Emissions Calculations**: EPA 40 CFR Part 98 compliant CO2, NOx, SO2 tracking
- **O2 Trim Control**: Adaptive excess air optimization with CO cross-limiting
- **Fuel Blending**: Multi-fuel optimization for cost and emissions reduction
- **Heat Balance**: Complete boiler heat balance with loss breakdown
- **Safety Interlocks**: SIL-2 compliant flame safety and burner management
- **SHA-256 Provenance**: Full audit trail for regulatory compliance

## Installation

```bash
# Clone repository
git clone https://github.com/greenlang/GL-002_Flameguard.git
cd GL-002_Flameguard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Python API

```python
from flameguard import BoilerEfficiencyOrchestrator
from flameguard.calculators import EfficiencyCalculator, EmissionsCalculator

# Initialize calculators
efficiency_calc = EfficiencyCalculator()
emissions_calc = EmissionsCalculator()

# Calculate efficiency (ASME PTC 4.1 indirect method)
result = efficiency_calc.calculate_indirect(
    fuel_type="natural_gas",
    fuel_flow_kg_h=1000,
    stack_temp_c=180,
    ambient_temp_c=25,
    o2_percent=3.0,
    steam_flow_kg_h=8500
)

print(f"Efficiency: {result.efficiency_percent:.2f}%")
print(f"Provenance Hash: {result.computation_hash}")
```

### REST API

```bash
# Start API server
uvicorn api.rest_api:app --host 0.0.0.0 --port 8000

# Calculate efficiency
curl -X POST http://localhost:8000/api/v1/efficiency/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "boiler_id": "boiler-01",
    "fuel_type": "natural_gas",
    "fuel_flow_kg_h": 1000,
    "stack_temp_c": 180,
    "o2_percent": 3.0
  }'
```

## Architecture

```
GL-002_Flameguard/
├── api/                    # REST API endpoints
├── audit/                  # Provenance tracking
├── calculators/            # Core calculation modules
│   ├── efficiency_calculator.py   # ASME PTC 4.1
│   ├── emissions_calculator.py    # EPA emission factors
│   ├── heat_balance_calculator.py # Heat balance analysis
│   └── fuel_blending_calculator.py # Multi-fuel optimization
├── core/                   # Schemas and orchestrator
├── deployment/             # Docker & Kubernetes
├── docs/                   # Documentation
├── explainability/         # Decision explanations
├── optimization/           # O2 trim controller
├── safety/                 # Safety interlocks
├── streaming/              # Kafka integration
└── tests/                  # Test suites
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_002_LOG_LEVEL` | `INFO` | Logging level |
| `GL_002_O2_SETPOINT_MIN` | `1.5` | Minimum O2 % setpoint |
| `GL_002_O2_SETPOINT_MAX` | `8.0` | Maximum O2 % setpoint |
| `GL_002_CO_LIMIT_PPM` | `100` | CO cross-limit threshold |
| `GL_002_SAFETY_ENABLED` | `true` | Enable safety interlocks |

## Efficiency Calculations

### ASME PTC 4.1 Loss Categories

1. **Dry Flue Gas Loss** - Heat lost to stack gases
2. **Moisture in Fuel** - Evaporation of fuel moisture
3. **H2 Combustion Loss** - Water from hydrogen combustion
4. **Moisture in Air** - Humidity in combustion air
5. **CO Loss** - Incomplete combustion
6. **Unburned Carbon** - Carbon in ash/refuse
7. **Radiation Loss** - Surface heat loss
8. **Blowdown Loss** - Boiler blowdown heat
9. **Other Losses** - Manufacturer margins

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html --cov-fail-under=80

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/test_integration/ -v
```

## Compliance Standards

- **ASME PTC 4.1**: Boiler performance testing
- **EPA 40 CFR Part 98**: GHG emission factors
- **NFPA 85**: Boiler combustion safety
- **IEC 61511**: Safety instrumented systems
- **ISO 50001**: Energy management

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/efficiency/calculate` | Calculate efficiency |
| POST | `/api/v1/emissions/calculate` | Calculate emissions |
| POST | `/api/v1/optimize` | Optimize O2 setpoint |
| GET | `/api/v1/boiler/{id}/status` | Boiler status |
| GET | `/api/v1/audit/{hash}` | Audit record |

## Kafka Topics

| Topic | Description |
|-------|-------------|
| `flameguard.telemetry` | Real-time boiler data |
| `flameguard.efficiency` | Efficiency calculations |
| `flameguard.emissions` | Emission records |
| `flameguard.safety` | Safety events |
| `flameguard.recommendations` | O2 setpoint recommendations |

## Deployment

### Docker

```bash
docker build -t gl-002-flameguard:latest -f deployment/Dockerfile .
docker run -p 8000:8000 gl-002-flameguard:latest
```

### Kubernetes

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml
```

## License

Proprietary - GreenLang Platform. All rights reserved.

## Support

- Documentation: https://docs.greenlang.io/agents/gl-002
- Issues: https://github.com/greenlang/GL-002_Flameguard/issues
