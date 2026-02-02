# GL-016 Waterguard

**Boiler Water Treatment Agent - Supervisory Control and Optimization**

Version: 1.0.0 | Status: Development | Priority: P0

---

## Overview

GL-016 Waterguard is a supervisory control and optimization agent for industrial boiler water treatment systems. It provides real-time cycles of concentration (CoC) optimization, intelligent blowdown control, chemical dosing optimization, and comprehensive water/energy savings tracking with full explainability and audit trail capabilities.

### Key Capabilities

- **Cycles of Concentration Optimization**: Maximize water efficiency while maintaining chemistry limits
- **Blowdown Control**: Continuous and intermittent blowdown optimization
- **Chemical Dosing Control**: Phosphate, polymer, oxygen scavenger, and amine dosing
- **Chemistry Constraint Monitoring**: Real-time monitoring of pH, conductivity, silica, and dissolved oxygen
- **Water/Energy Savings Tracking**: Quantify savings and emissions avoided
- **Explainability**: SHAP and LIME explanations for all recommendations
- **Audit Trail**: 7-year retention with SHA-256 provenance tracking

### Standards Compliance

| Standard | Description |
|----------|-------------|
| ASME | American Society of Mechanical Engineers - Boiler Codes |
| ABMA | American Boiler Manufacturers Association - Guidelines |
| IEC 62443 | Industrial Automation Cybersecurity |
| IEC 61511 | Functional Safety - SIL-3 Compliant |
| ISO 50001 | Energy Management Systems |
| ISO 14001 | Environmental Management Systems |
| NIST 800-82 | Industrial Control Systems Security |

---

## Architecture

```
+------------------------------------------------------------------+
|                     GL-016 WATERGUARD AGENT                       |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |   API Layer      |  |  Control Layer   |  |  Safety Layer    |  |
|  |------------------|  |------------------|  |------------------|  |
|  | REST API (8080)  |  | Blowdown Control |  | SIL-3 Interlocks |  |
|  | GraphQL          |  | Chemical Dosing  |  | Rate Limiting    |  |
|  | gRPC (50051)     |  | CoC Optimization |  | Fail-Safe Mode   |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |  Analytics       |  |  Explainability  |  |  Audit Layer     |  |
|  |------------------|  |------------------|  |------------------|  |
|  | Savings Tracking |  | SHAP Analysis    |  | Event Logging    |  |
|  | Emissions Calc   |  | LIME Explanations|  | 7-Year Retention |  |
|  | Anomaly Detect   |  | Decision Trace   |  | SHA-256 Hash     |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |  OT Integration  |  |  Streaming       |  |  Database        |  |
|  |------------------|  |------------------|  |------------------|  |
|  | OPC UA (4840)    |  | Kafka Topics     |  | PostgreSQL       |  |
|  | Modbus TCP (502) |  | Event Streaming  |  | TimescaleDB      |  |
|  | MQTT             |  | State Management |  | Redis Cache      |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    BOILER WATER TREATMENT SYSTEM                  |
+------------------------------------------------------------------+
| Sensors: Conductivity, pH, Silica, DO, Temperature, Level        |
| Actuators: Blowdown Valves, Dosing Pumps, Makeup Valves          |
+------------------------------------------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.10 or higher (< 3.13)
- PostgreSQL 14+
- Redis 7+
- Apache Kafka 3.x (optional, for streaming)
- Docker and Docker Compose (recommended)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/greenlang/gl-016-waterguard.git
cd gl-016-waterguard

# Start all services
docker-compose up -d

# Verify the agent is running
curl http://localhost:8080/health
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Set environment variables
export WATERGUARD_DB_URL="postgresql://user:password@localhost:5432/waterguard"
export WATERGUARD_REDIS_URL="redis://localhost:6379/0"
export WATERGUARD_KAFKA_BOOTSTRAP="localhost:9092"

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WATERGUARD_DB_URL` | PostgreSQL connection URL | `postgresql://localhost/waterguard` |
| `WATERGUARD_REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `WATERGUARD_KAFKA_BOOTSTRAP` | Kafka bootstrap servers | `localhost:9092` |
| `WATERGUARD_OPC_UA_URL` | OPC UA server URL | `opc.tcp://localhost:4840` |
| `WATERGUARD_MODBUS_HOST` | Modbus TCP host | `localhost` |
| `WATERGUARD_MODBUS_PORT` | Modbus TCP port | `502` |
| `WATERGUARD_OPERATING_MODE` | Operating mode | `advisory` |
| `WATERGUARD_SIL_LEVEL` | Safety Integrity Level | `3` |
| `WATERGUARD_LOG_LEVEL` | Logging level | `INFO` |

### Configuration File

Create a `config.yaml` file:

```yaml
# GL-016 Waterguard Configuration

agent:
  id: GL-016
  mode: advisory  # advisory, supervisory, autonomous

boiler:
  pressure_psig: 150
  steam_flow_max_klb_hr: 100
  blowdown_type: continuous  # continuous, intermittent, hybrid

chemistry_limits:
  ph:
    min: 10.5
    max: 11.5
    alarm_low: 10.3
    alarm_high: 11.7
    trip_low: 10.0
    trip_high: 12.0

  conductivity_umho:
    max: 5000
    alarm_high: 4500
    trip_high: 5500

  silica_ppm:
    max: 150
    alarm_high: 120
    trip_high: 180

  dissolved_oxygen_ppb:
    max: 7
    alarm_high: 5
    trip_high: 10

optimization:
  coc_target: 6.0
  coc_min: 3.0
  coc_max: 10.0
  optimization_interval_minutes: 15

safety:
  sil_level: 3
  watchdog_timeout_ms: 1000
  heartbeat_interval_ms: 500
  fail_safe_mode: maintain_current_setpoints
```

---

## API Reference

### REST API

Base URL: `http://localhost:8080/api/v1/waterguard`

#### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "agent_id": "GL-016",
  "version": "1.0.0",
  "mode": "advisory",
  "uptime_seconds": 3600
}
```

#### Get Current Chemistry State

```http
GET /chemistry/state
```

Response:
```json
{
  "timestamp": "2025-12-27T10:00:00Z",
  "ph": 11.0,
  "conductivity_umho": 3500,
  "silica_ppm": 100,
  "dissolved_oxygen_ppb": 5,
  "cycles_of_concentration": 6.2,
  "compliance_status": "compliant",
  "recommendations": [
    {
      "type": "blowdown_adjustment",
      "action": "reduce",
      "magnitude": 5,
      "unit": "percent",
      "rationale": "CoC below target, safe to reduce blowdown"
    }
  ]
}
```

#### Submit Chemistry Reading

```http
POST /chemistry/readings
Content-Type: application/json

{
  "timestamp": "2025-12-27T10:00:00Z",
  "ph": 11.0,
  "conductivity_umho": 3500,
  "silica_ppm": 100,
  "dissolved_oxygen_ppb": 5,
  "feedwater_conductivity_umho": 500
}
```

#### Get Optimization Recommendation

```http
GET /optimization/recommend
```

Response:
```json
{
  "timestamp": "2025-12-27T10:00:00Z",
  "optimal_coc": 6.5,
  "recommended_blowdown_percent": 2.5,
  "estimated_water_savings_m3_day": 15.2,
  "estimated_energy_savings_kwh_day": 450,
  "estimated_emissions_avoided_kg_co2e": 180,
  "explanation": {
    "shap_values": {
      "conductivity": 0.35,
      "silica": 0.25,
      "steam_demand": 0.20,
      "water_cost": 0.15,
      "treatment_cost": 0.05
    },
    "decision_rationale": "Increasing CoC from 6.0 to 6.5 is safe given current silica levels (100 ppm) well below limit (150 ppm). This will reduce blowdown by 8% and save approximately 15 m3/day of makeup water."
  }
}
```

#### Set Operating Mode

```http
POST /mode
Content-Type: application/json

{
  "mode": "supervisory",
  "operator_id": "OP-12345",
  "authorization_code": "AUTH-ABC123"
}
```

### GraphQL API

Endpoint: `http://localhost:8080/graphql/waterguard`

```graphql
query GetChemistryState {
  chemistryState {
    timestamp
    ph
    conductivity
    silica
    dissolvedOxygen
    cyclesOfConcentration
    complianceStatus
  }
}

mutation UpdateSetpoint($input: SetpointInput!) {
  updateSetpoint(input: $input) {
    success
    newSetpoint
    auditId
  }
}
```

### gRPC API

Port: `50016`

See `api/proto/waterguard.proto` for service definitions.

---

## Operating Modes

### Advisory Mode (Default)

- Displays recommendations only
- No automatic control actions
- Operator must manually implement changes
- Safety Level: SIL-0

### Supervisory Mode

- Agent proposes control actions
- Operator must confirm each action
- Automatic execution after approval
- Safety Level: SIL-1

### Autonomous Mode

- Fully automatic control within safety gates
- Operator monitoring only
- Automatic fallback on safety violations
- Safety Level: SIL-3

### Fallback Mode

- Conservative fixed-schedule operation
- Activated on sensor failure, communication loss, or safety gate trip
- Maintains safe operating conditions
- Safety Level: SIL-3

---

## Safety Considerations

### Safety Integrity Level (SIL-3)

GL-016 Waterguard is designed to meet IEC 61511 SIL-3 requirements for process safety.

### Safety Interlocks

| Interlock | Condition | Action |
|-----------|-----------|--------|
| Low Water Level | Level < 10% | Close blowdown, alarm |
| High Pressure | Pressure > 110% setpoint | Close blowdown, alarm |
| High Conductivity | Conductivity > trip limit | Increase blowdown |
| Chemical Leak | Leak detected | Stop dosing pumps |
| Communication Loss | No heartbeat > 5s | Enter fallback mode |

### Rate Limits

- Blowdown valve: Max 10%/second
- Dosing pumps: Max 5%/second
- CoC change: Max 0.5 units/hour

### Fail-Safe Behavior

On any safety-critical failure, the agent:
1. Maintains current setpoints
2. Enters fallback mode
3. Alerts operators
4. Logs the event with full audit trail

---

## Deployment

### Docker Deployment

```bash
# Build the image
docker build -t ghcr.io/greenlang/gl-016-waterguard:1.0.0 .

# Run with environment variables
docker run -d \
  --name waterguard \
  -p 8080:8080 \
  -p 50016:50016 \
  -e WATERGUARD_DB_URL="postgresql://user:password@db:5432/waterguard" \
  -e WATERGUARD_REDIS_URL="redis://redis:6379/0" \
  ghcr.io/greenlang/gl-016-waterguard:1.0.0
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-016-waterguard
  namespace: greenlang-ot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: waterguard
  template:
    metadata:
      labels:
        app: waterguard
    spec:
      containers:
      - name: waterguard
        image: ghcr.io/greenlang/gl-016-waterguard:1.0.0
        ports:
        - containerPort: 8080
        - containerPort: 50016
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
```

### Helm Chart

```bash
# Add GreenLang Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Install Waterguard
helm install waterguard greenlang/gl-016-waterguard \
  --namespace greenlang-ot \
  --set config.dbUrl="postgresql://user:password@db:5432/waterguard" \
  --set config.redisUrl="redis://redis:6379/0"
```

---

## Monitoring

### Prometheus Metrics

The agent exposes metrics at `/metrics`:

```
# Water efficiency metrics
gl016_waterguard_cycles_of_concentration{boiler_id="B001"} 6.2
gl016_waterguard_blowdown_rate_percent{boiler_id="B001"} 2.5
gl016_waterguard_water_savings_m3_total{boiler_id="B001"} 15432

# Chemistry metrics
gl016_waterguard_ph{boiler_id="B001"} 11.0
gl016_waterguard_conductivity_umho{boiler_id="B001"} 3500
gl016_waterguard_silica_ppm{boiler_id="B001"} 100
gl016_waterguard_dissolved_oxygen_ppb{boiler_id="B001"} 5

# Safety metrics
gl016_waterguard_constraint_compliance{boiler_id="B001"} 1.0
gl016_waterguard_safety_interlock_active{boiler_id="B001", interlock="none"} 0

# System metrics
gl016_waterguard_recommendation_acceptance_rate 0.95
gl016_waterguard_api_requests_total{endpoint="/chemistry/state"} 10000
gl016_waterguard_api_latency_seconds{endpoint="/chemistry/state", quantile="0.99"} 0.05
```

### Grafana Dashboard

Import the provided dashboard from `deployment/grafana/waterguard-dashboard.json`.

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/safety/
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
ruff check .
pylint core/ api/

# Type checking
mypy .

# Security scan
bandit -r core/ api/
safety check
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Support

- **Documentation**: https://docs.greenlang.io/agents/waterguard
- **Issues**: https://github.com/greenlang/gl-016-waterguard/issues
- **Email**: support@greenlang.io
- **Slack**: #gl-016-waterguard

---

## License

Copyright (c) 2025 GreenLang. All rights reserved.

See [LICENSE](LICENSE) for details.
