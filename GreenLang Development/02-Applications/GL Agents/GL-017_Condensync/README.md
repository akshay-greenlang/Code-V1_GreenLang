# GL-017 CONDENSYNC

**Condenser Optimization Agent**

GL-017 CONDENSYNC is a production-grade condenser performance monitoring and optimization agent that implements zero-hallucination principles for industrial steam surface condensers. The agent calculates condenser efficiency, identifies fouling conditions, and provides recommendations for maintenance and operation optimization based on Heat Exchange Institute (HEI) standards.

## Key Features

- **HEI Standards Compliance**: All calculations based on HEI 3098 (11th Edition) and ASME PTC 12.2
- **Zero-Hallucination Guarantee**: Deterministic physics-based calculations with no AI inference
- **Performance Monitoring**: Real-time cleanliness factor, TTD, DCA, and U-value tracking
- **Economic Analysis**: Heat rate penalty quantification with fuel cost and CO2 impact
- **Optimization Engine**: Cleaning schedule and cooling water flow optimization
- **Explainability**: Factor attribution and evidence chains for all recommendations
- **Provenance Tracking**: SHA-256 hashes for complete audit trail

## Quick Start

### Docker

```bash
# Run with default settings
docker run -d \
  --name condensync \
  -p 8017:8017 \
  -p 9017:9017 \
  greenlang/condensync:latest

# Verify health
curl http://localhost:8017/api/v1/health
```

### Python

```bash
# Clone the repository
git clone https://github.com/greenlang/gl-017-condensync.git
cd gl-017-condensync

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m condensync.main
```

### Kubernetes

```bash
# Add Helm repository
helm repo add greenlang https://charts.greenlang.io

# Install
helm install condensync greenlang/condensync \
  --namespace condensync \
  --create-namespace
```

## API Usage

### Analyze Condenser

```bash
curl -X POST "http://localhost:8017/api/v1/analyze" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "condenser_id": "COND-001",
    "unit_id": "Unit1",
    "process_data": {
      "condenser_pressure_kpa": 5.0,
      "hotwell_temperature_c": 33.0,
      "heat_duty_mw": 500.0
    },
    "cooling_water": {
      "inlet_temperature_c": 20.0,
      "outlet_temperature_c": 30.0,
      "flow_rate_m3_s": 12.0,
      "velocity_m_s": 2.1
    },
    "design_data": {
      "surface_area_m2": 25000,
      "tube_od_mm": 25.4,
      "tube_material": "titanium",
      "design_u_w_m2k": 3200
    }
  }'
```

### Response

```json
{
  "condenser_id": "COND-001",
  "analysis_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "timestamp": "2025-12-30T10:30:00Z",
  "performance": {
    "cleanliness_factor_pct": 82.5,
    "actual_u_w_m2k": 2640,
    "ttd_k": 5.2,
    "dca_k": 3.0,
    "fouling_resistance_m2k_w": 0.000065
  },
  "condition": {
    "state": "light_fouling",
    "severity": "moderate",
    "confidence": 0.92,
    "days_to_threshold": 45
  },
  "impact": {
    "heat_rate_penalty_btu_kwh": 25.5,
    "annual_fuel_cost_usd": 125000,
    "annual_co2_tonnes": 850
  }
}
```

## Architecture

```
+-----------------------------------------------------------------------------+
|                           GL-017 CONDENSYNC                                   |
+-----------------------------------------------------------------------------+
|                                                                               |
|  +-----------------+    +-----------------+    +-----------------+           |
|  |   Process Data  |    |  Cooling Water  |    |   Design Data   |           |
|  +-----------------+    +-----------------+    +-----------------+           |
|            |                    |                      |                      |
|            v                    v                      v                      |
|  +-------------------------------------------------------------------+       |
|  |                     Integration Layer                              |       |
|  |          (OPC-UA, Modbus, MQTT, Historian, REST)                  |       |
|  +-------------------------------------------------------------------+       |
|                                   |                                           |
|                                   v                                           |
|  +-------------------------------------------------------------------+       |
|  |                   Core Processing Engine                           |       |
|  |                                                                    |       |
|  |  +------------+  +------------+  +------------+  +------------+   |       |
|  |  |    HEI     |  |    LMTD    |  |  Fouling   |  | Cleanliness|   |       |
|  |  | Calculator |  | Calculator |  | Calculator |  |   Factor   |   |       |
|  |  +------------+  +------------+  +------------+  +------------+   |       |
|  +-------------------------------------------------------------------+       |
|                                   |                                           |
|                                   v                                           |
|  +-------------------------------------------------------------------+       |
|  |                    Optimization Engine                             |       |
|  |                                                                    |       |
|  |  +------------+  +------------+  +------------+  +------------+   |       |
|  |  |  Cleaning  |  |     CW     |  | Backpres.  |  |  Economic  |   |       |
|  |  |  Schedule  |  |    Flow    |  |   Impact   |  |  Analysis  |   |       |
|  |  +------------+  +------------+  +------------+  +------------+   |       |
|  +-------------------------------------------------------------------+       |
|                                   |                                           |
|                                   v                                           |
|  +-------------------------------------------------------------------+       |
|  |                         Output Layer                               |       |
|  |       (REST API, Prometheus Metrics, DCS Feedback, Reports)       |       |
|  +-------------------------------------------------------------------+       |
|                                                                               |
+-------------------------------------------------------------------------------+
```

## Deployment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Edge** | Standalone at plant site | Real-time monitoring, offline operation |
| **Edge+Central** | Edge processing with cloud sync | Fleet analytics, cross-plant insights |
| **Offline** | Batch analysis of historical data | Assessments, commissioning, audits |

## Calculation Methodology

CONDENSYNC uses deterministic calculations based on HEI 3098 standards:

1. **LMTD Calculation**: Log mean temperature difference for condensing steam
2. **U-Value Calculation**: Overall heat transfer coefficient from heat duty and LMTD
3. **Cleanliness Factor**: Ratio of actual to design U-value (corrected for conditions)
4. **Heat Rate Impact**: Backpressure sensitivity for turbine efficiency degradation
5. **Economic Analysis**: Fuel cost and CO2 impact from heat rate penalty

All formulas are documented with references to specific HEI sections. See [CALCULATIONS.md](docs/CALCULATIONS.md) for complete derivations.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/analyze` | POST | Single condenser analysis |
| `/analyze/batch` | POST | Multi-condenser analysis |
| `/optimize/cleaning` | POST | Cleaning schedule optimization |
| `/optimize/cooling-water` | POST | CW flow optimization |
| `/status` | GET | Agent status |
| `/config` | GET/PUT | Configuration management |

## Prometheus Metrics

```
condensync_cleanliness_factor{unit, condenser}
condensync_ttd_kelvin{unit, condenser}
condensync_dca_kelvin{unit, condenser}
condensync_heat_rate_penalty_btu_kwh{unit}
condensync_diagnoses_total{condition, severity}
condensync_diagnosis_duration_seconds{}
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CONDENSYNC_MODE` | Operation mode | production |
| `CONDENSYNC_LOG_LEVEL` | Log level | INFO |
| `CONDENSYNC_API_PORT` | API port | 8017 |
| `CONDENSYNC_METRICS_PORT` | Metrics port | 9017 |
| `CONDENSYNC_DB_URL` | Database URL | - |

Configuration file: See [config.yaml](config/config.yaml)

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and component breakdown |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API documentation |
| [CALCULATIONS.md](docs/CALCULATIONS.md) | HEI calculation methodology |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, Kubernetes, Helm deployment |
| [openapi.yaml](docs/openapi.yaml) | OpenAPI 3.0 specification |

## Standards Compliance

- **HEI 3098**: Standards for Steam Surface Condensers (11th Edition)
- **ASME PTC 12.2**: Steam Surface Condenser Performance Test Code
- **IAPWS-IF97**: Steam and Water Properties
- **EPRI TR-107397**: Condenser Performance Monitoring

## Project Structure

```
gl-017-condensync/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── condensync/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── condenser_analyzer.py
│   │   ├── bounds_validator.py
│   │   └── guardrails.py
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── hei_calculator.py
│   │   ├── lmtd_calculator.py
│   │   ├── fouling_calculator.py
│   │   ├── cleanliness_factor_calculator.py
│   │   └── economic_calculator.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── cleaning_optimizer.py
│   │   └── cw_optimizer.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── opc_ua_connector.py
│   │   ├── historian_connector.py
│   │   └── cmms_connector.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── diagnostic_explainer.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── models/
│       ├── __init__.py
│       └── dataclasses.py
├── config/
│   └── config.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── property/
├── deploy/
│   ├── kubernetes/
│   └── helm/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── CALCULATIONS.md
│   ├── DEPLOYMENT.md
│   └── openapi.yaml
└── .github/
    └── workflows/
        └── ci.yml
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=condensync --cov-report=html

# Run linting
ruff check condensync/
mypy condensync/

# Run formatting
black condensync/
```

## License

Proprietary - GreenLang Inc.

## Support

- **Documentation**: https://docs.greenlang.io/condensync
- **Issues**: https://github.com/greenlang/gl-017-condensync/issues
- **Support**: support@greenlang.io
