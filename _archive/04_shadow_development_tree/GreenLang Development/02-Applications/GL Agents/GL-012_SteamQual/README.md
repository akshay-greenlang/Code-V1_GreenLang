# GL-012_SteamQual

## Steam Quality Controller

Real-time steam quality monitoring, dryness fraction estimation, and carryover risk assessment for industrial steam systems.

**Agent ID:** GL-012
**Version:** 1.0.0
**Category:** Steam
**Type:** Hybrid (Calculator + Optimizer + Analyzer)

## Overview

GL-012 STEAMQUAL provides comprehensive steam quality control capabilities:

- **Real-time Quality Estimation**: Dryness fraction (x) estimation using physics-based soft sensors
- **Carryover Risk Assessment**: Moisture carryover detection and early warning
- **Separator Monitoring**: Separator/scrubber efficiency tracking and health analytics
- **Quality Events**: Detection, root-cause analysis, and operator guidance
- **Control Recommendations**: Supervisory control recommendations (advisory mode)
- **GL-003 Integration**: Seamless integration with UNIFIEDSTEAM optimizer

## Key Features

### Quality Estimation
- Enthalpy method: x = (h - hf) / hfg
- Entropy method: x = (s - sf) / sfg
- Specific volume method: x = (v - vf) / (vg - vf)
- Throttling calorimeter method
- Kalman filter state estimation
- Uncertainty propagation with confidence intervals

### Carryover Risk Detection
- Drum level impact analysis
- Load swing correlation
- Foaming tendency assessment
- Droplet entrainment risk scoring
- Early warning with time-to-event prediction

### Separator/Drain System
- Mass balance modeling: m_removed = η_sep × moisture_in
- Online efficiency estimation
- Capacity constraint monitoring
- Trap health analytics

## Standards Compliance

- ASME PTC 19.11 Steam Quality
- IAPWS-IF97 Steam Tables
- GreenLang Framework v1.0

## Installation

```bash
pip install -e ".[dev]"
```

For full installation with all optional dependencies:

```bash
pip install -e ".[all]"
```

## Quick Start

```python
from gl_012_steamqual import SteamQualAgent, SteamMeasurement

# Create agent
agent = SteamQualAgent()

# Estimate quality from measurements
measurement = SteamMeasurement(
    pressure_kpa=1000.0,
    temperature_c=180.0,
    flow_kg_s=5.0,
)
result = agent.estimate_quality(measurement)

print(f"Dryness fraction: {result.x_est:.4f} ± {result.uncertainty:.4f}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Provenance: {result.provenance_hash[:16]}...")
```

## API Server

Start the API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation: `http://localhost:8000/docs`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/estimate-quality` | POST | Estimate steam quality from measurements |
| `/assess-carryover-risk` | POST | Assess moisture carryover risk |
| `/quality-state/{header_id}` | GET | Get current quality state |
| `/events` | GET | Get quality events |
| `/recommendations` | POST | Get control recommendations |
| `/health` | GET | Health check |

## Configuration

Configuration via environment variables (prefix: `GL_012_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_012_X_MIN_DEFAULT` | 0.95 | Minimum dryness fraction |
| `GL_012_SUPERHEAT_MARGIN_MIN` | 3.0 | Minimum superheat margin (°C) |
| `GL_012_CARRYOVER_RISK_THRESHOLD` | 0.7 | Carryover risk alert threshold |
| `GL_012_LOG_LEVEL` | INFO | Logging level |
| `GL_012_TRACK_PROVENANCE` | true | Enable provenance tracking |

## Event Types

| Event | Severity | Description |
|-------|----------|-------------|
| LOW_DRYNESS | S2-S3 | Quality below threshold |
| HIGH_MOISTURE | S2-S3 | Excessive moisture detected |
| CARRYOVER_RISK | S1-S2 | Elevated carryover risk |
| SEPARATOR_FLOODING | S2-S3 | Separator drain issues |
| WATER_HAMMER_RISK | S2-S3 | Condensate accumulation risk |
| DATA_QUALITY_DEGRADED | S1 | Sensor/data quality issues |

## Testing

Run tests:

```bash
pytest tests/ -v --cov
```

Run golden master tests:

```bash
pytest tests/golden/ -v -m golden
```

## Provenance

All calculations include SHA-256 provenance tracking:

- `computation_hash`: Combined hash of inputs + outputs + parameters
- `inputs_hash`: Hash of input data
- `timestamp`: UTC timestamp of calculation
- `agent_version`: Agent version for reproducibility

## Architecture

```
GL-012_SteamQual/
├── core/           # Configuration, orchestration
├── calculators/    # Deterministic calculations
├── thermodynamics/ # Steam property functions
├── estimators/     # Soft sensors, ML models
├── safety/         # Circuit breakers, validators
├── monitoring/     # Metrics, alerting
├── explainability/ # SHAP, root cause analysis
├── audit/          # Provenance, compliance
├── integration/    # Sensor connectors, GL-003 interface
├── control/        # Control recommendations
├── models/         # Pydantic data models
├── api/            # FastAPI REST API
├── streaming/      # Real-time data streaming
├── compliance/     # Standards compliance
├── tests/          # Unit, integration, golden tests
└── deploy/         # Docker, Kubernetes
```

## GL-003 Integration

GL-012 integrates with GL-003 UNIFIEDSTEAM as a domain module:

```python
from gl_012_steamqual import SteamQualAgent

agent = SteamQualAgent()

# Get quality constraints for optimizer
constraints = agent.get_quality_constraints("HP_HEADER_1")
# Returns: QualityConstraints(x_min=0.95, delta_t_min=3.0, ...)

# Get current quality state
state = agent.get_quality_state("HP_HEADER_1")
# Returns: QualityState(x_est=0.97, r_carry=0.15, ...)

# Get quality events
events = agent.get_events(severity_min="S1")
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Sensor-to-metric latency | < 5 seconds |
| Event emission latency | < 10 seconds |
| API response time | < 100ms (p99) |
| Service availability | 99.9% |
| Test coverage | ≥ 85% |

## License

Proprietary - GreenLang Platform

## Contributing

See CONTRIBUTING.md for development guidelines.

## Support

- Documentation: https://docs.greenlang.ai/agents/gl-012
- Issues: https://github.com/greenlang/gl-agents/issues
