# GL-001 Thermalcommand

**Multi-Equipment Thermal Asset Optimization Agent**

[![GreenLang Framework](https://img.shields.io/badge/GreenLang-v1.0-green.svg)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()

## Overview

GL-001_Thermalcommand is an industrial-grade AI agent for optimizing thermal assets including boilers, furnaces, and heat recovery systems. It provides real-time optimization recommendations while ensuring safety through IEC 61511 SIL-2 compliant safety interlocks.

## Key Features

- **MILP Optimization**: Mixed-Integer Linear Programming for multi-equipment thermal scheduling
- **Cascade PID Control**: ISA-standard cascade controllers with anti-windup and bumpless transfer
- **Safety Integration**: IEC 61511 SIL-2 compliant 2oo3 voting logic for safety-critical decisions
- **SHA-256 Provenance**: Full audit trail with Merkle tree verification for all calculations
- **Explainability**: SHAP and LIME explanations for optimization recommendations
- **Real-time Streaming**: Kafka integration for telemetry and recommendations

## Installation

```bash
# Clone the repository
git clone https://github.com/greenlang/GL-001_Thermalcommand.git
cd GL-001_Thermalcommand

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Python API

```python
from thermalcommand import ThermalCommandOrchestrator
from thermalcommand.core.schemas import OptimizationRequest

# Initialize orchestrator
orchestrator = ThermalCommandOrchestrator(
    track_provenance=True,
    safety_enabled=True
)

# Create optimization request
request = OptimizationRequest(
    equipment_ids=["boiler-01", "boiler-02", "furnace-01"],
    optimization_horizon_hours=24,
    objective="minimize_fuel_cost",
    constraints={
        "min_efficiency": 0.85,
        "max_emissions_kg_h": 500
    }
)

# Run optimization
result = orchestrator.optimize(request)
print(f"Optimal schedule: {result.schedule}")
print(f"Provenance hash: {result.computation_hash}")
```

### REST API

```bash
# Start the API server
uvicorn api.rest_api:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Run optimization
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"equipment_ids": ["boiler-01"], "horizon_hours": 8}'
```

### GraphQL API

```graphql
query GetOptimization {
  optimizeThermalAssets(
    equipmentIds: ["boiler-01", "boiler-02"]
    horizonHours: 24
  ) {
    schedule {
      equipmentId
      setpoints
      efficiency
    }
    provenanceHash
    executionTimeMs
  }
}
```

## Architecture

```
GL-001_Thermalcommand/
├── api/                    # REST, GraphQL, gRPC endpoints
├── audit/                  # Provenance tracking and evidence packs
├── control/                # Cascade PID controllers
├── core/                   # Orchestrator and schemas
├── data_contracts/         # Tag dictionaries and domain schemas
├── deployment/             # Kubernetes manifests
├── explainability/         # SHAP/LIME explainers
├── integrations/           # OPC-UA, webhooks, CMMS
├── monitoring/             # Metrics and health checks
├── optimization/           # MILP optimizer and scenarios
├── safety/                 # SIS integration and boundary engine
├── streaming/              # Kafka producers/consumers
└── tests/                  # Unit, integration, chaos tests
```

## Configuration

Configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_001_LOG_LEVEL` | `INFO` | Logging level |
| `GL_001_TRACK_PROVENANCE` | `true` | Enable SHA-256 tracking |
| `GL_001_SAFETY_ENABLED` | `true` | Enable SIS interlocks |
| `GL_001_OPTIMIZATION_TIMEOUT_S` | `300` | Max optimization time |
| `GL_001_KAFKA_BOOTSTRAP` | `localhost:9092` | Kafka brokers |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html --cov-fail-under=80

# Run specific test categories
pytest tests/test_unit/ -v          # Unit tests
pytest tests/test_integration/ -v   # Integration tests
pytest tests/chaos/ -v              # Chaos engineering tests
```

## Compliance

- **IEC 61511**: Safety Instrumented Systems for process industries
- **ASME PTC 4.1**: Steam generating units performance
- **ISO 50001**: Energy management systems
- **GHG Protocol**: Scope 1 & 2 emissions calculation

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/optimize` | Run optimization |
| GET | `/api/v1/equipment/{id}` | Get equipment status |
| POST | `/api/v1/setpoint` | Update setpoint |
| GET | `/api/v1/audit/{hash}` | Retrieve audit record |

### Kafka Topics

| Topic | Description |
|-------|-------------|
| `thermalcommand.telemetry` | Real-time sensor data |
| `thermalcommand.recommendations` | Optimization outputs |
| `thermalcommand.safety` | Safety events |
| `thermalcommand.audit` | Provenance records |

## Deployment

### Docker

```bash
docker build -t gl-001-thermalcommand:latest .
docker run -p 8000:8000 gl-001-thermalcommand:latest
```

### Kubernetes

```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/configmap.yaml
```

## License

Proprietary - GreenLang Platform. All rights reserved.

## Support

- Documentation: https://docs.greenlang.io/agents/gl-001
- Issues: https://github.com/greenlang/GL-001_Thermalcommand/issues
