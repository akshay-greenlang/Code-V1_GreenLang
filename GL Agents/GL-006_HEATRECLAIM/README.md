# GL-006 HEATRECLAIM - Heat Recovery Maximizer

**Agent ID:** GL-006
**Codename:** HEATRECLAIM
**Name:** HeatRecoveryMaximizer
**Class:** Optimizer
**Domain:** Heat Recovery
**Version:** 1.0.0

## Overview

GL-006 HEATRECLAIM is an AI agent specialized in heat exchanger network (HEN) synthesis and optimization. It uses pinch analysis, MILP optimization, and multi-objective Pareto methods to design optimal heat recovery systems for industrial processes.

## Key Features

- **Pinch Analysis**: Problem table algorithm, composite curves, grand composite curves
- **HEN Synthesis**: Pinch Design Method for heat exchanger network generation
- **MILP Optimization**: Mixed-integer linear programming for optimal network design
- **Multi-objective Optimization**: Pareto frontier generation (cost vs. recovery vs. complexity)
- **Uncertainty Quantification**: Monte Carlo and Latin Hypercube sampling
- **Explainability**: SHAP, LIME, and causal inference for decision transparency
- **Deterministic Calculations**: SHA-256 provenance for all computations

## Installation

```bash
# Clone repository
git clone https://github.com/greenlang/gl-agents.git
cd gl-agents/GL-006_HEATRECLAIM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Quick Start

### Python API

```python
from core.orchestrator import HeatReclaimOrchestrator
from core.schemas import HeatStream
from core.config import StreamType, Phase

# Create streams
hot_streams = [
    HeatStream(
        stream_id="H1",
        stream_name="Process Effluent",
        stream_type=StreamType.HOT,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=150.0,
        T_target_C=60.0,
        m_dot_kg_s=2.0,
        Cp_kJ_kgK=4.18,
    ),
]

cold_streams = [
    HeatStream(
        stream_id="C1",
        stream_name="Feed Preheater",
        stream_type=StreamType.COLD,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=20.0,
        T_target_C=135.0,
        m_dot_kg_s=1.5,
        Cp_kJ_kgK=4.18,
    ),
]

# Optimize
orchestrator = HeatReclaimOrchestrator()
result = orchestrator.optimize(
    hot_streams=hot_streams,
    cold_streams=cold_streams,
    delta_t_min=10.0,
)

print(f"Pinch Temperature: {result.pinch_analysis.pinch_temperature_C}°C")
print(f"Heat Recovered: {result.recommended_design.total_heat_recovered_kW} kW")
print(f"Exchangers: {result.recommended_design.exchanger_count}")
```

### REST API

```bash
# Start server
uvicorn api.rest_api:app --port 8006

# Run optimization
curl -X POST http://localhost:8006/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d @examples/simple_problem.json
```

### GraphQL

```graphql
mutation {
  optimize(
    hotStreams: [
      { streamId: "H1", TSupplyC: 150, TTargetC: 60, mDotKgS: 2.0, CpKJKgK: 4.18 }
    ]
    coldStreams: [
      { streamId: "C1", TSupplyC: 20, TTargetC: 135, mDotKgS: 1.5, CpKJKgK: 4.18 }
    ]
    options: { deltaTMinC: 10.0, objective: "minimize_cost" }
  ) {
    pinchAnalysis { pinchTemperatureC maximumHeatRecoveryKW }
    recommendedDesign { exchangerCount totalHeatRecoveredKW }
  }
}
```

## Architecture

```
GL-006_HEATRECLAIM/
├── core/                    # Core modules
│   ├── config.py           # Configuration and enums
│   ├── schemas.py          # Pydantic data models
│   ├── orchestrator.py     # Main agent orchestrator
│   └── handlers.py         # Request handlers
├── calculators/            # Calculation engines
│   ├── pinch_analysis.py   # Pinch analysis calculator
│   ├── hen_synthesis.py    # HEN design method
│   ├── lmtd_calculator.py  # LMTD/NTU calculations
│   ├── exergy_calculator.py    # Second-law analysis
│   └── economic_calculator.py  # Economic analysis
├── optimization/           # Optimization engines
│   ├── milp_optimizer.py   # MILP for HEN synthesis
│   ├── pareto_generator.py # Multi-objective optimization
│   └── uncertainty_quantifier.py  # UQ engine
├── explainability/         # Explainability modules
│   ├── shap_explainer.py   # SHAP feature attribution
│   ├── lime_explainer.py   # LIME local explanations
│   ├── causal_analyzer.py  # Causal inference
│   └── engineering_rationale.py  # Rule-based explanations
├── api/                    # API layer
│   ├── rest_api.py         # FastAPI endpoints
│   ├── graphql_schema.py   # Strawberry GraphQL
│   └── middleware.py       # Rate limiting, auth
├── tests/                  # Test suite
├── deploy/                 # Deployment configs
└── pack.yaml              # GreenLang manifest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/optimize` | POST | Run full optimization |
| `/api/v1/pinch-analysis` | POST | Pinch analysis only |
| `/api/v1/validate-streams` | POST | Validate stream data |
| `/api/v1/status/{job_id}` | GET | Check async job status |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_AGENT_ID` | GL-006 | Agent identifier |
| `GL_ENV` | development | Environment |
| `GL_LOG_LEVEL` | INFO | Logging level |
| `GL_DELTA_T_MIN` | 10.0 | Default ΔTmin (°C) |
| `REDIS_URL` | - | Redis connection URL |
| `KAFKA_BOOTSTRAP_SERVERS` | - | Kafka servers |

## Deployment

### Docker

```bash
# Build image
docker build -t gl-006-heatreclaim:latest .

# Run container
docker run -p 8006:8006 gl-006-heatreclaim:latest

# Docker Compose
docker-compose up -d
```

### Kubernetes

```bash
kubectl apply -f deploy/kubernetes/
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_pinch_analysis.py -v
```

## Standards Compliance

- **Zero-Hallucination**: All calculations are deterministic with SHA-256 provenance
- **Audit Trail**: Complete traceability of optimization decisions
- **GreenLang v1.0**: Compliant with GreenLang specification
- **ISO 50001**: Aligned with energy management standards

## References

- Linnhoff, B., & Hindmarsh, E. (1983). The pinch design method for heat exchanger networks. *Chemical Engineering Science*, 38(5), 745-763.
- Yee, T. F., & Grossmann, I. E. (1990). Simultaneous optimization models for heat integration. *Computers & Chemical Engineering*, 14(10), 1165-1184.
- Kemp, I. C. (2011). *Pinch Analysis and Process Integration*. Butterworth-Heinemann.

## License

Proprietary - GreenLang Technologies

## Support

For support, contact: support@greenlang.io
