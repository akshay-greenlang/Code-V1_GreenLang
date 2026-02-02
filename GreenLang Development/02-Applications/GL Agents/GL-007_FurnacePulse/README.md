# GL-007 FURNACEPULSE - Furnace Performance Monitor

**Agent ID:** GL-007
**Codename:** FURNACEPULSE
**Name:** FurnacePerformanceMonitor
**Class:** Monitor
**Domain:** Industrial Furnaces
**Version:** 1.0.0

---

## Overview

GL-007 FURNACEPULSE is an AI agent specialized in industrial furnace performance monitoring, predictive maintenance, and NFPA 86 compliance. It provides real-time KPI tracking, Tube Metal Temperature (TMT) monitoring, hotspot detection via IR thermography, Remaining Useful Life (RUL) predictions, and comprehensive safety evidence packaging.

### Key Objectives

- **Performance Optimization**: Track thermal efficiency, specific fuel consumption (SFC), excess air, and stack losses in real-time
- **Predictive Maintenance**: Forecast equipment failures before they occur using survival analysis and ML models
- **Safety Compliance**: Automate NFPA 86 evidence collection with HAZOP/LOPA integration
- **Explainability**: Provide transparent, auditable predictions via SHAP/LIME feature attribution

### Important Safety Disclaimer

> **ADVISORY ONLY**: GL-007 FURNACEPULSE is an advisory system designed to support operational decision-making. It does NOT replace certified safety systems including:
> - Burner Management Systems (BMS)
> - Safety Instrumented Systems (SIS)
> - Safety Instrumented Functions (SIFs)
> - Emergency Shutdown Systems (ESD)
>
> All safety-critical decisions must be validated by qualified personnel and implemented through certified safety systems. This agent provides recommendations only and should never be used as a primary safety control.

---

## Key Features

### Real-Time KPI Tracking

| KPI | Description | Unit |
|-----|-------------|------|
| Thermal Efficiency | Heat absorbed vs. fuel energy input | % |
| Specific Fuel Consumption (SFC) | Fuel used per unit production | MJ/kg |
| Excess Air | Combustion air above stoichiometric | % |
| Stack Loss | Energy lost through exhaust gases | % |
| Availability | Operational uptime ratio | % |

### Tube Metal Temperature (TMT) Monitoring

- Real-time TMT tracking across radiant, convection, shield, and crossover zones
- Rate-of-rise detection (max 10 deg C/min threshold)
- Design limit enforcement (default 950 deg C max)
- Spatial clustering for hotspot localization

### IR Thermography Integration

- Direct integration with industrial IR cameras
- Automated thermal frame processing
- Hotspot detection using spatial clustering algorithms
- Severity classification (LOW, MEDIUM, HIGH, CRITICAL)
- Evidence snapshot capture for audit trails

### Predictive Maintenance

- Weibull and Cox Proportional Hazards survival analysis
- Component-level RUL predictions with confidence intervals
- Maintenance scheduling optimization
- CMMS work order automation

### NFPA 86 Compliance

- Automated compliance checklist verification
- Evidence package generation for audits
- Continuous monitoring of:
  - Flame supervision systems
  - Combustion air interlocks
  - Fuel shutoff interlocks
  - Purge verification
  - Emergency shutdown systems

### HAZOP/LOPA Integration

- Deviation tracking linked to HAZOP nodes
- Protection layer verification per LOPA methodology
- Layers supported:
  - BPCS (Basic Process Control System)
  - Operator Response
  - Safety Instrumented Function (SIF)
  - Mechanical Relief
  - Emergency Shutdown

### Explainability (SHAP/LIME)

- SHAP feature attribution for global and local explanations
- LIME local interpretable explanations
- Engineering rationale linking predictions to physical phenomena
- Model cards with performance metrics and drift sensitivity
- Stakeholder-specific report generation

### CMMS Integration

- Automated work order creation
- Asset hierarchy synchronization
- Maintenance schedule coordination
- Spare parts correlation

---

## Architecture

### High-Level Architecture Diagram

```
+------------------+     +------------------+     +------------------+
|   Data Sources   |     |   FurnacePulse   |     |    Consumers     |
+------------------+     |      Agent       |     +------------------+
                         +------------------+
+--------+               +------------------+     +------------------+
| OPC-UA |-------------->|                  |---->| REST API         |
| Server |    Tags       |  Orchestrator    |     | /api/v1/*        |
+--------+               |                  |     +------------------+
                         |  +------------+  |
+--------+               |  | Calculators|  |     +------------------+
|  IR    |-------------->|  +------------+  |---->| GraphQL          |
| Camera |   Frames      |  | Efficiency |  |     | /graphql         |
+--------+               |  | Hotspot    |  |     +------------------+
                         |  | RUL        |  |
+--------+               |  | Draft      |  |     +------------------+
| Kafka  |<------------->|  +------------+  |---->| Kafka Topics     |
| Broker |   Events      |                  |     | furnacepulse.*   |
+--------+               |  +------------+  |     +------------------+
                         |  | Explainers |  |
+--------+               |  +------------+  |     +------------------+
| CMMS   |<------------->|  | SHAP       |  |---->| CMMS             |
| System |  Work Orders  |  | LIME       |  |     | Work Orders      |
+--------+               |  | Rationale  |  |     +------------------+
                         |  +------------+  |
+--------+               |                  |     +------------------+
| Histor-|-------------->|  +------------+  |---->| Prometheus       |
| ian DB |   Time Series |  | Safety     |  |     | /metrics         |
+--------+               |  +------------+  |     +------------------+
                         |  | NFPA86     |  |
                         |  | LOPA/HAZOP |  |     +------------------+
                         |  | Evidence   |  |---->| Dashboards       |
                         |  +------------+  |     | Grafana/Custom   |
                         +------------------+     +------------------+
```

### Component Architecture

```
GL-007_FurnacePulse/
|-- core/                       # Core orchestration
|   |-- orchestrator.py         # Main agent orchestrator
|   |-- config.py               # Configuration management
|   |-- schemas.py              # Pydantic data models
|   +-- handlers.py             # Request handlers
|
|-- calculators/                # Deterministic calculators
|   |-- efficiency_calculator.py    # Thermal efficiency, SFC
|   |-- hotspot_detector.py         # TMT monitoring, clustering
|   |-- rul_predictor.py            # Remaining Useful Life
|   +-- draft_analyzer.py           # Draft pressure analysis
|
|-- explainability/             # ML explainability
|   |-- shap_explainer.py       # SHAP feature attribution
|   |-- lime_explainer.py       # LIME local explanations
|   |-- engineering_rationale.py    # Physical phenomenon mapping
|   |-- model_cards.py          # Model documentation
|   +-- report_generator.py     # Stakeholder reports
|
|-- safety/                     # Safety & compliance
|   |-- nfpa86_compliance.py    # NFPA 86 checklist management
|   |-- lopa_hazop_integration.py   # LOPA/HAZOP linking
|   |-- safety_alerts.py        # Safety alert taxonomy
|   +-- evidence_packager.py    # Audit evidence packaging
|
|-- alerts/                     # Alert management
|   |-- alert_orchestrator.py   # Alert taxonomy, scoring
|   |-- notification_service.py # Multi-channel notifications
|   +-- response_playbooks.py   # Response procedures
|
|-- integration/                # External integrations
|   |-- opcua_client.py         # OPC-UA connectivity
|   |-- kafka_producer.py       # Kafka event publishing
|   |-- kafka_consumer.py       # Kafka event consumption
|   |-- cmms_integration.py     # CMMS work orders
|   |-- ir_camera_client.py     # IR camera integration
|   +-- historian_client.py     # Time-series database
|
|-- api/                        # API layer
|   |-- rest_api.py             # FastAPI REST endpoints
|   |-- graphql_schema.py       # Strawberry GraphQL
|   +-- middleware.py           # Auth, rate limiting
|
|-- monitoring/                 # Observability
|   |-- metrics.py              # Prometheus metrics
|   +-- health.py               # Health checks
|
|-- deploy/                     # Deployment configs
|   |-- docker/                 # Docker files
|   |-- kubernetes/             # K8s manifests
|   +-- helm/                   # Helm charts
|
|-- tests/                      # Test suite
|   |-- unit/                   # Unit tests
|   |-- integration/            # Integration tests
|   +-- e2e/                    # End-to-end tests
|
+-- pack.yaml                   # GreenLang manifest
```

### Data Flow

1. **Ingestion**: OPC-UA client subscribes to furnace tags (TMT, temperatures, draft, fuel flow)
2. **Canonical Transform**: Raw telemetry normalized to canonical schema with quality scoring
3. **Streaming**: Canonical data published to Kafka (`furnacepulse.telemetry.canonical`)
4. **Calculation**: Deterministic calculators process data with SHA-256 provenance
5. **Prediction**: ML models generate RUL forecasts and hotspot probabilities
6. **Explainability**: SHAP/LIME generate feature attributions for predictions
7. **Alerting**: Alert orchestrator evaluates thresholds and generates alerts
8. **Integration**: Alerts trigger CMMS work orders and notifications
9. **Compliance**: Safety module packages evidence for NFPA 86 audits
10. **Serving**: REST/GraphQL APIs serve dashboards and external systems

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for containerized deployment)
- Access to OPC-UA server (for live furnace data)
- Kafka cluster (for event streaming)
- Redis (optional, for caching)

### Installation

#### Option 1: pip install

```bash
# Clone repository
git clone https://github.com/greenlang/gl-agents.git
cd gl-agents/GL-007_FurnacePulse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

#### Option 2: Docker

```bash
# Build image
docker build -t gl-007-furnacepulse:latest .

# Run container
docker run -p 8007:8007 \
  -e FURNACEPULSE_OPCUA_HOST=opc.tcp://your-server:4840 \
  -e FURNACEPULSE_KAFKA_BROKERS=kafka:9092 \
  gl-007-furnacepulse:latest

# Or use Docker Compose
docker-compose up -d
```

### Configuration

Create a `.env` file or set environment variables:

```bash
# Core
FURNACEPULSE_LOG_LEVEL=INFO
FURNACEPULSE_ENV=development

# OPC-UA
FURNACEPULSE_OPCUA_HOST=opc.tcp://localhost:4840
FURNACEPULSE_OPCUA_NAMESPACE=2
FURNACEPULSE_OPCUA_CERT_PATH=/certs/client.pem

# Kafka
FURNACEPULSE_KAFKA_BROKERS=localhost:9092
FURNACEPULSE_KAFKA_SECURITY_PROTOCOL=SASL_SSL

# Database
FURNACEPULSE_DB_CONNECTION=postgresql://user:pass@localhost:5432/furnacepulse

# CMMS
FURNACEPULSE_CMMS_URL=https://cmms.example.com/api
FURNACEPULSE_CMMS_API_KEY=your-api-key

# Alerting
ALERT_WEBHOOK_URL=https://alerts.example.com/webhook
CMMS_WEBHOOK_URL=https://cmms.example.com/webhook
```

### Running the Agent

```bash
# Start the API server
uvicorn api.rest_api:app --host 0.0.0.0 --port 8007

# Or run with the orchestrator directly
python -m core.orchestrator
```

### Python API Example

```python
from core.orchestrator import FurnacePulseOrchestrator
from calculators import EfficiencyCalculator, EfficiencyInputs

# Initialize orchestrator
orchestrator = FurnacePulseOrchestrator()

# Calculate thermal efficiency
calc = EfficiencyCalculator(agent_id="GL-007")
result = calc.calculate(EfficiencyInputs(
    fuel_mass_flow_kg_s=0.5,
    fuel_lhv_kj_kg=42000,
    useful_heat_output_kw=8000
))

print(f"Thermal Efficiency: {result.result.thermal_efficiency_pct:.2f}%")
print(f"Provenance Hash: {result.provenance.sha256_hash}")
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FURNACEPULSE_LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `FURNACEPULSE_ENV` | development | Environment (development, staging, production) |
| `FURNACEPULSE_OPCUA_HOST` | - | OPC-UA server endpoint URL |
| `FURNACEPULSE_OPCUA_NAMESPACE` | 2 | OPC-UA namespace index |
| `FURNACEPULSE_KAFKA_BROKERS` | - | Kafka bootstrap servers |
| `FURNACEPULSE_DB_CONNECTION` | - | Database connection string |
| `FURNACEPULSE_CMMS_URL` | - | CMMS API base URL |
| `ALERT_WEBHOOK_URL` | - | Webhook URL for alerts |

### pack.yaml Structure

The `pack.yaml` file defines the agent manifest:

```yaml
name: gl-007-furnacepulse
version: 1.0.0
description: Furnace Performance Monitoring Agent

metadata:
  agent_id: GL-007
  agent_name: FURNACEPULSE
  standards:
    - NFPA 86
    - API 560
    - API 530
    - ISO 13705
    - ASME PTC 4
  greenlang:
    zero_hallucination: true
    provenance_tracking: true
    deterministic_calculations: true

runtime:
  language: python
  python_version: ">=3.10"
  entrypoint: core.orchestrator:FurnacePulseOrchestrator

inputs:
  furnace_temperature: {...}
  fuel_consumption: {...}
  tube_temps_tmt: {...}
  ir_camera_feeds: {...}

outputs:
  performance_kpis: {...}
  maintenance_alerts: {...}
  rul_predictions: {...}
  compliance_status: {...}

safety:
  constraints:
    - tag: TMT_MAX
      max_value: 950.0
      unit: C
    - tag: TMT_RATE_OF_RISE
      max_value: 10.0
      unit: C/min
```

### Tag Registry Setup

Configure OPC-UA tag mappings in `config/tag_registry.yaml`:

```yaml
tag_registry:
  version: "1.0"
  furnaces:
    - furnace_id: FRN-001
      name: Reformer Furnace 1
      tags:
        tmt:
          - tag: "ns=2;s=FRN001.TMT.RAD.T001"
            tube_id: "RAD-T001"
            zone: RADIANT
            design_limit_C: 950
          - tag: "ns=2;s=FRN001.TMT.RAD.T002"
            tube_id: "RAD-T002"
            zone: RADIANT
            design_limit_C: 950
        temperatures:
          - tag: "ns=2;s=FRN001.ZONE1.TEMP"
            zone: RADIANT
          - tag: "ns=2;s=FRN001.STACK.TEMP"
            zone: STACK
        fuel:
          - tag: "ns=2;s=FRN001.FUEL.FLOW"
            type: natural_gas
            unit: kg/h
```

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check with component status |
| `/api/v1/furnaces` | GET | List monitored furnaces |
| `/api/v1/furnaces/{id}` | GET | Get furnace details |
| `/api/v1/tmt` | GET | Get current TMT readings |
| `/api/v1/tmt/{furnace_id}` | GET | Get TMT for specific furnace |
| `/api/v1/kpis` | GET | Get performance KPIs |
| `/api/v1/kpis/{furnace_id}` | GET | Get KPIs for specific furnace |
| `/api/v1/alerts` | GET | List active alerts |
| `/api/v1/alerts/{id}` | GET | Get alert details |
| `/api/v1/alerts/{id}/acknowledge` | POST | Acknowledge alert |
| `/api/v1/hotspots` | GET | List detected hotspots |
| `/api/v1/hotspots/{id}` | GET | Get hotspot details with IR snapshot |
| `/api/v1/rul` | GET | Get RUL predictions |
| `/api/v1/rul/{component_id}` | GET | Get RUL for specific component |
| `/api/v1/compliance` | GET | Get compliance status |
| `/api/v1/compliance/evidence` | POST | Generate evidence package |
| `/api/v1/explain/{prediction_id}` | GET | Get explanation for prediction |

### REST API Example

```bash
# Get health status
curl http://localhost:8007/api/v1/health

# Get current TMT readings
curl http://localhost:8007/api/v1/tmt/FRN-001 \
  -H "Authorization: Bearer $TOKEN"

# Get RUL predictions
curl http://localhost:8007/api/v1/rul \
  -H "Authorization: Bearer $TOKEN"

# Generate compliance evidence
curl -X POST http://localhost:8007/api/v1/compliance/evidence \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"furnace_id": "FRN-001", "period": "2025-Q1"}'
```

### GraphQL Schema Overview

```graphql
type Query {
  furnace(id: ID!): Furnace
  furnaces: [Furnace!]!
  tmtReadings(furnaceId: ID!, limit: Int): [TMTReading!]!
  kpis(furnaceId: ID!): EfficiencyKPI!
  alerts(status: AlertStatus, severity: AlertSeverity): [Alert!]!
  hotspots(furnaceId: ID!, minSeverity: HotspotSeverity): [Hotspot!]!
  rulPredictions(furnaceId: ID!): [RULPrediction!]!
  complianceStatus(furnaceId: ID!): ComplianceReport!
}

type Mutation {
  acknowledgeAlert(alertId: ID!, userId: ID!): Alert!
  createWorkOrder(input: WorkOrderInput!): WorkOrder!
  generateEvidencePackage(furnaceId: ID!, period: String!): EvidencePackage!
}

type Subscription {
  tmtUpdates(furnaceId: ID!): TMTReading!
  alertCreated: Alert!
  hotspotDetected: Hotspot!
}

type Furnace {
  id: ID!
  name: String!
  location: String!
  status: FurnaceStatus!
  currentKpis: EfficiencyKPI!
  tmtReadings: [TMTReading!]!
  activeAlerts: [Alert!]!
}

type TMTReading {
  tubeId: String!
  temperatureC: Float!
  rateOfRiseCMin: Float!
  zone: TubeZone!
  designLimitC: Float!
  timestamp: DateTime!
}

type EfficiencyKPI {
  thermalEfficiencyPercent: Float!
  sfcMjKg: Float!
  excessAirPercent: Float!
  stackLossPercent: Float!
  calculationHash: String!
  timestamp: DateTime!
}
```

### Authentication & RBAC

Authentication uses OAuth2 with JWT bearer tokens:

```python
# Request token
response = requests.post(
    "https://auth.greenlang.io/oauth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "scope": "furnacepulse:read furnacepulse:write"
    }
)
token = response.json()["access_token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
```

**Role-Based Access Control (RBAC):**

| Role | Permissions |
|------|-------------|
| `viewer` | Read KPIs, alerts, TMT readings |
| `operator` | All viewer + acknowledge alerts |
| `engineer` | All operator + configure thresholds |
| `safety` | All engineer + compliance evidence |
| `admin` | Full access including user management |

---

## Alert Taxonomy

GL-007 implements a comprehensive alert taxonomy with 30 alert types organized by category:

### TMT/Hotspot Alerts (A-001 to A-009)

| Code | Name | Severity | Trigger |
|------|------|----------|---------|
| A-001 | TMT Advisory | ADVISORY | TMT at 85% of design limit |
| A-002 | TMT Warning | WARNING | TMT at 95% of design limit |
| A-003 | TMT Urgent | URGENT | TMT exceeds design limit |
| A-004 | Rate of Rise Advisory | ADVISORY | Rate of rise > 5 deg C/min |
| A-005 | Rate of Rise Warning | WARNING | Rate of rise > 8 deg C/min |
| A-006 | Rate of Rise Urgent | URGENT | Rate of rise > 10 deg C/min |
| A-007 | Hotspot Detected | WARNING | IR camera detects localized hotspot |
| A-008 | Hotspot Spreading | WARNING | Hotspot area expanding |
| A-009 | Hotspot Critical | URGENT | Multiple critical hotspots detected |

### Efficiency Alerts (A-010 to A-019)

| Code | Name | Severity | Trigger |
|------|------|----------|---------|
| A-010 | Efficiency Degradation | ADVISORY | Efficiency drops 5% from baseline |
| A-011 | Efficiency Warning | WARNING | Efficiency drops 10% from baseline |
| A-012 | SFC Elevated | ADVISORY | SFC exceeds target by 10% |
| A-013 | Excess Air High | ADVISORY | Excess air > 25% |
| A-014 | Excess Air Low | WARNING | Excess air < 10% (incomplete combustion risk) |
| A-015 | Stack Loss High | ADVISORY | Stack loss > 15% |
| A-016 | Stack Temp High | WARNING | Stack temperature > 400 deg C |
| A-017 | Combustion Anomaly | WARNING | O2/CO2 ratio abnormal |
| A-018 | Heat Balance Deviation | ADVISORY | Heat balance deviation > 5% |
| A-019 | Fouling Detected | ADVISORY | Convection section fouling suspected |

### Draft Alerts (A-020 to A-024)

| Code | Name | Severity | Trigger |
|------|------|----------|---------|
| A-020 | Draft Low | WARNING | Draft pressure < -200 Pa |
| A-021 | Draft High | WARNING | Draft pressure > 20 Pa (puffing risk) |
| A-022 | Draft Instability | ADVISORY | Draft pressure oscillating |
| A-023 | Damper Anomaly | ADVISORY | Damper position vs. draft mismatch |
| A-024 | Air Register Imbalance | WARNING | Uneven air distribution |

### Equipment Alerts (A-025 to A-029)

| Code | Name | Severity | Trigger |
|------|------|----------|---------|
| A-025 | RUL Low | ADVISORY | Component RUL < 30 days |
| A-026 | RUL Critical | WARNING | Component RUL < 7 days |
| A-027 | Refractory Degradation | ADVISORY | Refractory condition score < 70 |
| A-028 | Burner Performance | ADVISORY | Burner efficiency deviation > 5% |
| A-029 | Flame Instability | WARNING | UV/IR flame signal unstable |

### Sensor Alerts (A-030)

| Code | Name | Severity | Trigger |
|------|------|----------|---------|
| A-030 | Sensor Anomaly | ADVISORY | Sensor drift, stuck, or missing signal |

### Response Playbooks

Each alert type has an associated response playbook. Example for A-003 (TMT Urgent):

```
PLAYBOOK: A-003 TMT Urgent Response

OWNER: Operations Engineer
ESCALATION: Shift Supervisor -> Plant Manager -> Safety Lead

STEPS:
1. IMMEDIATE (0-5 min)
   - Acknowledge alert in CMMS
   - Verify TMT reading with independent sensor
   - Check IR camera for hotspot confirmation
   - Notify shift supervisor

2. ASSESSMENT (5-15 min)
   - Review recent operating parameter trends
   - Check for flame impingement indicators
   - Assess process feed rate and composition
   - Evaluate convection section performance

3. RESPONSE (15-30 min)
   - Consider reducing firing rate by 10-20%
   - Adjust excess air if combustion anomaly
   - Redistribute process load if possible
   - Initiate burner inspection if flame issue

4. DOCUMENTATION
   - Log all actions in CMMS
   - Update evidence package for NFPA 86
   - Complete incident report if threshold exceeded

REFERENCES:
- NFPA 86 Section 8.7.3 - Temperature Monitoring
- API 530 - Heater Tube Thickness Calculation
- Site Operating Procedure SOP-FRN-003
```

---

## Compliance & Safety

### NFPA 86 Evidence Collection

GL-007 automates evidence collection for NFPA 86 compliance:

**Monitored Requirements:**

| Section | Requirement | Evidence Type |
|---------|-------------|---------------|
| 8.7.1 | Flame Supervision | Flame detector status logs |
| 8.7.2 | Combustion Air Interlock | Airflow/damper position records |
| 8.7.3 | Temperature Monitoring | TMT trend data with timestamps |
| 8.7.4 | Fuel Shutoff Interlock | Valve position confirmation |
| 8.9.1 | Purge Verification | Purge cycle completion records |
| 8.10 | Emergency Shutdown | ESD test records, trip logs |

**Evidence Package Generation:**

```python
from safety import NFPA86ComplianceManager, EvidencePackager

compliance_mgr = NFPA86ComplianceManager(config)
packager = EvidencePackager(config)

# Check current compliance status
status = compliance_mgr.get_compliance_status(furnace_id="FRN-001")
print(f"Status: {status.overall_status}")
print(f"Passed: {status.items_passed}/{status.total_items}")

# Generate evidence package for audit
package = packager.create_package(
    furnace_id="FRN-001",
    period_start="2025-01-01",
    period_end="2025-03-31",
    include_sections=["8.7", "8.9", "8.10"]
)
print(f"Package ID: {package.id}")
print(f"SHA-256: {package.integrity_hash}")
```

### LOPA/HAZOP Integration

GL-007 links telemetry deviations to HAZOP/LOPA documentation:

```python
from safety import LOPAHAZOPIntegrator

integrator = LOPAHAZOPIntegrator(config)

# Link deviation to protection layers
deviation = {
    "type": "HIGH_TMT",
    "value": 960,
    "threshold": 950,
    "furnace_id": "FRN-001",
    "tube_id": "RAD-T005"
}

# Get relevant HAZOP node and LOPA layers
lopa_response = integrator.evaluate_deviation(deviation)
print(f"HAZOP Node: {lopa_response.hazop_node}")
print(f"IPLs Engaged: {lopa_response.ipls_engaged}")
print(f"RRF Required: {lopa_response.rrf_required}")
```

**LOPA Layers Tracked:**

1. **BPCS**: Basic process control actions logged
2. **Operator Response**: Operator acknowledgment tracked
3. **SIF**: Safety instrumented function activations
4. **Mechanical Relief**: Relief valve activations
5. **Emergency Shutdown**: ESD activations

### Audit Trail

All operations are logged with immutable audit records:

```json
{
  "event_id": "evt_abc123",
  "timestamp": "2025-01-15T10:30:00Z",
  "event_type": "kpi_calculated",
  "agent_id": "GL-007",
  "furnace_id": "FRN-001",
  "details": {
    "thermal_efficiency_pct": 87.5,
    "calculation_inputs_hash": "sha256:abc123...",
    "calculation_outputs_hash": "sha256:def456..."
  },
  "user_id": null,
  "provenance": {
    "calculation_version": "1.0.0",
    "model_version": null,
    "deterministic": true
  }
}
```

---

## Development

### Project Structure

```
GL-007_FurnacePulse/
|-- core/                   # Core orchestration
|-- calculators/            # Calculation engines
|-- explainability/         # SHAP/LIME modules
|-- safety/                 # NFPA 86, LOPA/HAZOP
|-- alerts/                 # Alert management
|-- integration/            # External integrations
|-- api/                    # REST/GraphQL APIs
|-- monitoring/             # Metrics and health
|-- deploy/                 # Deployment configs
|-- tests/                  # Test suite
|   |-- unit/               # Unit tests
|   |-- integration/        # Integration tests
|   +-- e2e/                # End-to-end tests
|-- docs/                   # Documentation
|-- examples/               # Example scripts
|-- pack.yaml               # GreenLang manifest
|-- requirements.txt        # Python dependencies
|-- Dockerfile              # Container build
+-- docker-compose.yaml     # Local development stack
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests (requires Docker)
pytest tests/integration/ -v --docker

# Run specific test file
pytest tests/unit/test_efficiency_calculator.py -v

# Run tests matching pattern
pytest tests/ -k "hotspot" -v
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy . --strict

# Formatting
black .

# Security scan
bandit -r . -ll
```

### Contributing Guidelines

1. **Branch Naming**: Use `feature/`, `fix/`, `docs/` prefixes
2. **Commit Messages**: Follow Conventional Commits specification
3. **Pull Requests**: Include description, testing notes, and screenshots
4. **Tests**: All new features require unit tests with >80% coverage
5. **Documentation**: Update README and docstrings for API changes

**Code Review Checklist:**

- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Type hints added for new functions
- [ ] Docstrings follow Google style
- [ ] No security vulnerabilities (bandit clean)
- [ ] CHANGELOG updated

---

## Deployment

### Docker

```bash
# Build image
docker build -t gl-007-furnacepulse:latest .

# Run with environment file
docker run -p 8007:8007 --env-file .env gl-007-furnacepulse:latest

# Docker Compose for full stack
docker-compose up -d
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Or use Helm
helm install furnacepulse deploy/helm/furnacepulse \
  --namespace greenlang \
  --values deploy/helm/values-production.yaml
```

### Resource Requirements

| Environment | CPU | Memory | Replicas |
|-------------|-----|--------|----------|
| Development | 1 core | 2 GB | 1 |
| Staging | 2 cores | 4 GB | 2 |
| Production | 4 cores | 16 GB | 2+ |

---

## Standards Compliance

- **GreenLang v1.0**: Full compliance with GreenLang agent specification
- **Zero-Hallucination**: All calculations are deterministic with SHA-256 provenance
- **NFPA 86**: Standard for Ovens and Furnaces
- **API 560**: Fired Heaters for General Refinery Service
- **API 530**: Calculation of Heater-Tube Thickness
- **ISO 13705**: Petroleum and Petrochemical Industries Heaters
- **ASME PTC 4**: Fired Steam Generators

---

## References

- NFPA 86: Standard for Ovens and Furnaces, 2023 Edition
- API Standard 560: Fired Heaters for General Refinery Service
- API Recommended Practice 530: Calculation of Heater-Tube Thickness in Petroleum Refineries
- ISO 13705: Petroleum, Petrochemical and Natural Gas Industries - Fired Heaters for General Refinery Service
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.

---

## License

Proprietary - GreenLang Technologies

All rights reserved. Unauthorized copying, distribution, or modification of this software is strictly prohibited.

---

## Support

- **Documentation**: https://docs.greenlang.io/furnacepulse
- **API Status**: https://status.greenlang.io
- **Support**: support@greenlang.io
- **Security Issues**: security@greenlang.io

---

## Disclaimer

> **IMPORTANT**: This software is provided "as is" without warranty of any kind, express or implied. GL-007 FURNACEPULSE is an advisory system only and does not replace:
>
> - Certified Burner Management Systems (BMS)
> - Safety Instrumented Systems (SIS)
> - Safety Instrumented Functions (SIFs)
> - Emergency Shutdown Systems (ESD)
> - Human operator judgment and training
>
> Users are solely responsible for ensuring compliance with all applicable safety regulations and standards. All safety-critical decisions must be validated by qualified personnel and implemented through certified safety systems.
>
> GreenLang Technologies assumes no liability for any damages, injuries, or losses arising from the use of this software.
