# GL-017 CONDENSYNC

## Condenser Optimization Agent

**Version:** 1.0.0
**Status:** Production Ready
**Domain:** Steam Systems - Condenser Optimization
**TAM:** $4B
**Target:** Q2 2026

---

## Executive Summary

GL-017 CONDENSYNC is GreenLang's advanced condenser optimization agent designed for steam turbine power generation facilities. The agent provides real-time performance monitoring, vacuum optimization, fouling prediction, and cooling water flow optimization to maximize thermal efficiency and reduce operational costs.

### Key Value Propositions

- **Real-time condenser performance optimization** - Continuous monitoring and adjustment recommendations
- **Predictive fouling detection and cleaning scheduling** - Reduce unplanned outages by 30%
- **Air in-leakage monitoring and leak localization** - Protect vacuum system integrity
- **Heat rate improvement through vacuum optimization** - Target 0.5% efficiency improvement
- **Cooling water pump energy optimization** - Target 10% pump energy savings
- **Zero-hallucination calculations** - All numeric computations use deterministic, auditable formulas

### Business Impact

| KPI | Target | Measurement |
|-----|--------|-------------|
| Vacuum Improvement | 1.0 kPa | Monthly |
| Heat Rate Improvement | 0.5% | Monthly |
| CW Pump Energy Savings | 10% | Monthly |
| Cleaning Optimization | 20% | Quarterly |
| Unplanned Outage Reduction | 30% | Annual |

---

## Table of Contents

1. [Key Features](#key-features)
2. [Technical Architecture Overview](#technical-architecture-overview)
3. [Quick Start Guide](#quick-start-guide)
4. [Configuration Reference](#configuration-reference)
5. [API Reference](#api-reference)
6. [Integration Guide](#integration-guide)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Compliance](#compliance)
10. [Contributing Guidelines](#contributing-guidelines)

---

## Key Features

### Condenser Optimization

The CONDENSYNC agent continuously monitors condenser performance and provides optimization recommendations:

- **Vacuum Pressure Optimization** - Calculates achievable vacuum based on cooling water conditions and equipment constraints
- **Heat Transfer Coefficient Tracking** - Real-time U-value calculation with HEI cleanliness factor
- **Terminal Temperature Difference (TTD)** - Monitors approach to design conditions
- **Log Mean Temperature Difference (LMTD)** - Heat exchanger performance metric

### Vacuum Control

Comprehensive vacuum system monitoring and optimization:

- **Design vs. actual vacuum comparison** - Track vacuum deviation from expected values
- **Saturation temperature correlation** - Steam property calculations per IAPWS-IF97
- **Backpressure impact on turbine efficiency** - Heat rate correction factors
- **Vacuum improvement recommendations** - Actionable setpoint adjustments

### Heat Recovery

Maximize heat transfer efficiency:

- **Cleanliness factor trending** - Track degradation over time
- **Fouling factor calculation** - Per HEI 2629 standards
- **Heat duty optimization** - Balance condenser load with cooling capacity
- **Condensate subcooling monitoring** - Minimize thermal losses

### Fouling Detection

Predictive maintenance for tube cleaning:

- **Fouling rate prediction** - Model degradation trajectory
- **Optimal cleaning schedule** - Balance cleaning cost vs. efficiency loss
- **Cleaning method recommendation** - Mechanical, chemical, or hybrid approaches
- **Economic impact analysis** - ROI for cleaning interventions

### Air In-leakage Detection

Protect vacuum system integrity:

- **Leakage rate estimation** - Based on air ejector loading
- **Severity classification** - None/Minor/Moderate/Severe/Critical
- **Probable location identification** - LP turbine seals, expansion joints, valves
- **Repair prioritization** - Risk-based maintenance planning

---

## Technical Architecture Overview

```
+------------------------------------------------------------------+
|                     GL-017 CONDENSYNC Agent                       |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+   +-------------------+   +---------------+ |
|  |  Data Acquisition |   |   Calculations    |   |  Optimization | |
|  |-------------------|   |-------------------|   |---------------| |
|  | - OPC-UA Client   |   | - Heat Transfer   |   | - Vacuum SP   | |
|  | - Modbus TCP      |   | - Fouling Factor  |   | - CW Flow     | |
|  | - REST APIs       |   | - Air Inleakage   |   | - Cleaning    | |
|  | - Historian       |   | - Performance     |   | - Scheduling  | |
|  +-------------------+   +-------------------+   +---------------+ |
|           |                      |                      |          |
|           v                      v                      v          |
|  +------------------------------------------------------------------+
|  |                    Message Bus (Kafka)                          |
|  +------------------------------------------------------------------+
|           |                      |                      |          |
|           v                      v                      v          |
|  +-------------------+   +-------------------+   +---------------+ |
|  |   Data Storage    |   |    Alerting       |   |   Reporting   | |
|  |-------------------|   |-------------------|   |---------------| |
|  | - InfluxDB (TS)   |   | - PagerDuty       |   | - PDF Export  | |
|  | - PostgreSQL      |   | - Slack           |   | - Excel       | |
|  | - Redis (Cache)   |   | - Email           |   | - S3 Storage  | |
|  +-------------------+   +-------------------+   +---------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

### Core Components

| Component | Description | Technology |
|-----------|-------------|------------|
| Orchestrator | Main agent coordination | Python/FastAPI |
| Tool Executor | Deterministic calculations | Python |
| Data Collector | Real-time data acquisition | OPC-UA, Modbus |
| Message Bus | Event streaming | Kafka |
| Cache Layer | Performance optimization | Redis |
| Time Series DB | Metric storage | InfluxDB |
| Relational DB | Configuration/Audit | PostgreSQL |

---

## Quick Start Guide

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Access to SCADA/OPC-UA server
- PostgreSQL 15+
- Redis 7+
- InfluxDB 2.x

### Installation

#### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/greenlang/gl-017-condensync.git
cd gl-017-condensync

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Start services
docker-compose up -d

# Verify deployment
curl http://localhost:8017/health/ready
```

#### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/condensync"
export REDIS_URL="redis://localhost:6379/0"

# Run the agent
python -m greenlang.GL_017
```

### First Run Configuration

1. **Configure Condenser Parameters**

```python
from greenlang.GL_017.config import AgentConfiguration

config = AgentConfiguration.create_default_configuration(
    condenser_id="COND-001",
    turbine_capacity_mw=500.0
)
```

2. **Connect to SCADA**

```yaml
# config/scada.yaml
scada_integration:
  scada_system_name: "Plant SCADA"
  protocol: "OPC-UA"
  server_address: "opc.tcp://scada.plant.local:4840"
  polling_interval_seconds: 5
```

3. **Start Monitoring**

```python
from greenlang.GL_017 import CondenserOptimizationAgent

agent = CondenserOptimizationAgent(config)
result = await agent.execute()

print(f"Performance Score: {result.performance_score}")
print(f"Vacuum: {result.vacuum_data.pressure_in_hg_abs} in Hg abs")
```

---

## Configuration Reference

### Agent Configuration

```yaml
# gl.yaml - Main agent configuration
apiVersion: greenlang.io/v1
kind: Agent

metadata:
  id: GL-017
  codename: CONDENSYNC
  name: CondenserOptimizationAgent
  version: "1.0.0"

spec:
  runtime:
    language: python
    version: "3.11"
    framework: fastapi
    deterministic:
      enabled: true
      temperature: 0.0
      seed: 42
      provenance_tracking: true
```

### Condenser Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `condenser_id` | string | Unique identifier | Required |
| `condenser_type` | enum | surface, direct_contact, air_cooled, hybrid | surface |
| `design_heat_duty_mmbtu_hr` | float | Design heat duty (MMBtu/hr) | Required |
| `design_vacuum_in_hg_abs` | float | Design vacuum (in Hg abs) | Required |
| `design_ttd_f` | float | Design TTD (F) | 7.0 |
| `surface_area_sqft` | float | Heat transfer area (sq ft) | Required |

### Cooling Water Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `design_flow_gpm` | float | Design flow rate (GPM) | Required |
| `design_inlet_temp_f` | float | Design inlet temperature (F) | Required |
| `design_outlet_temp_f` | float | Design outlet temperature (F) | Required |
| `number_of_pumps` | int | Number of CW pumps | 3 |
| `vfd_enabled` | bool | Variable frequency drive | false |

### Alert Thresholds

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `vacuum_warning_deviation_in_hg` | float | Warning threshold | 0.5 |
| `vacuum_critical_deviation_in_hg` | float | Critical threshold | 1.0 |
| `fouling_warning_pct` | float | Cleanliness warning | 75.0 |
| `fouling_critical_pct` | float | Cleanliness critical | 60.0 |
| `air_inleakage_warning_scfm` | float | Air leak warning | 10.0 |
| `air_inleakage_critical_scfm` | float | Air leak critical | 20.0 |

---

## API Reference

### REST Endpoints

#### Health Checks

```http
GET /health/live
GET /health/ready
GET /health/startup
```

#### Performance Analysis

```http
POST /api/v1/condenser/{condenser_id}/analyze
Content-Type: application/json

{
  "time_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-02T00:00:00Z"
  },
  "include_trends": true
}
```

**Response:**

```json
{
  "condenser_id": "COND-001",
  "performance_score": 87.5,
  "performance_status": "NORMAL",
  "heat_transfer": {
    "u_value_actual_btu_hr_sqft_f": 485.0,
    "cleanliness_factor_pct": 82.5,
    "ttd_f": 8.2,
    "lmtd_f": 15.4
  },
  "vacuum_data": {
    "pressure_in_hg_abs": 1.82,
    "deviation_in_hg": 0.32,
    "air_inleakage_scfm": 7.5
  },
  "recommendations": [
    "Schedule tube cleaning within 30 days",
    "Investigate elevated air inleakage"
  ],
  "provenance_hash": "sha256:abc123..."
}
```

#### Vacuum Optimization

```http
POST /api/v1/condenser/{condenser_id}/optimize/vacuum
Content-Type: application/json

{
  "current_vacuum": 50,
  "turbine_load": 450,
  "ambient_conditions": {
    "dry_bulb_temp": 30,
    "wet_bulb_temp": 24
  }
}
```

#### Fouling Analysis

```http
GET /api/v1/condenser/{condenser_id}/fouling
```

#### Generate Report

```http
POST /api/v1/condenser/{condenser_id}/report
Content-Type: application/json

{
  "report_period": {
    "start": "2025-01-01",
    "end": "2025-01-31"
  },
  "format": "pdf"
}
```

### Python SDK

```python
from greenlang.GL_017 import CondenserOptimizationAgent
from greenlang.GL_017.tools import CondenserToolExecutor

# Initialize executor
executor = CondenserToolExecutor()

# Calculate heat transfer coefficient
result = await executor.execute_tool(
    "calculate_heat_transfer_coefficient",
    {
        "cw_inlet_temp": 20.0,
        "cw_outlet_temp": 30.0,
        "cw_flow_rate": 50000.0,
        "steam_temp": 35.0,
        "heat_duty": 300000.0
    }
)

print(f"U-value: {result['u_value']} W/(m2.K)")
print(f"Cleanliness: {result['cleanliness_factor'] * 100}%")
```

---

## Integration Guide

### SCADA Integration (OPC-UA)

```yaml
data_sources:
  - id: scada_primary
    type: opc_ua
    connection:
      endpoint: "${SCADA_PRIMARY_ENDPOINT}"
      security_mode: SignAndEncrypt
      security_policy: Basic256Sha256
    polling:
      interval_seconds: 5
      batch_size: 150
    tags:
      - condenser_vacuum_pressure
      - cooling_water_inlet_temp
      - cooling_water_outlet_temp
      - cooling_water_flow_rate
```

### Cooling Tower Integration

The agent coordinates with cooling tower PLCs for inlet temperature optimization:

```yaml
cooling_tower_integration:
  tower_id: "CT-001"
  enabled: true
  number_of_cells: 4
  design_wet_bulb_f: 78.0
  design_approach_f: 10.0
  vfd_enabled: true
  scada_integration: true
```

**Coordination Logic:**

1. Monitor cooling water inlet temperature
2. Calculate optimal CT cold water temperature for target vacuum
3. Adjust CT fan speed setpoints (if auto-optimization enabled)
4. Track approach temperature trends

### DCS Integration

```yaml
data_sources:
  - id: dcs_secondary
    type: opc_ua
    connection:
      endpoint: "${DCS_SECONDARY_ENDPOINT}"
    tags:
      - condenser_shell_pressure
      - tube_bundle_dp
      - waterbox_inlet_pressure
      - expansion_joint_temp
```

### Historian Integration

```yaml
data_sources:
  - id: historian
    type: opc_hda
    connection:
      endpoint: "${HISTORIAN_ENDPOINT}"
    query:
      max_rows: 50000
      default_interval_minutes: 15
    data_categories:
      - condenser_performance
      - vacuum_trends
      - fouling_progression
```

### CMMS Integration

Work order generation for maintenance activities:

```yaml
data_sinks:
  - id: cmms
    type: rest_api
    connection:
      base_url: "${CMMS_API_URL}"
      auth_type: oauth2
    events:
      - tube_cleaning_required
      - air_inleakage_investigation
      - vacuum_pump_maintenance
    work_order_templates:
      - id: condenser_tube_cleaning
        priority: high
        craft: mechanical
        estimated_hours: 24
```

---

## Monitoring and Alerting

### Prometheus Metrics

The agent exposes metrics on port 9017:

```
# Vacuum pressure
gl017_condenser_vacuum_kpa_abs{condenser_id="COND-001"}

# Heat transfer coefficient
gl017_heat_transfer_coefficient_w_m2k{condenser_id="COND-001"}

# Cleanliness factor
gl017_cleanliness_factor_percent{condenser_id="COND-001"}

# Air inleakage
gl017_air_inleakage_kg_hr{condenser_id="COND-001"}

# Calculation duration
gl017_calculation_duration_seconds{calculation_type="heat_transfer"}
```

### Alert Rules

| Alert | Condition | Severity | Channels |
|-------|-----------|----------|----------|
| High Vacuum Deviation | > 3.0 kPa | Critical | PagerDuty, SMS |
| Low HTC | < 60% design | Critical | PagerDuty, Slack |
| Air Inleakage Alarm | > 2x limit | Critical | PagerDuty, SMS |
| Fouling Critical | CF < 60% | High | Slack, Email |
| CW Flow Low | < 70% design | Critical | PagerDuty |

### Grafana Dashboards

Pre-configured dashboards available:

1. **Condenser Overview** - Key performance indicators
2. **Heat Transfer Analysis** - U-value and fouling trends
3. **Vacuum Monitoring** - Pressure and air inleakage
4. **Cooling Water System** - Flow and temperature
5. **Optimization Performance** - Savings tracking

---

## Troubleshooting Guide

### Common Issues

#### 1. Vacuum Lower Than Expected

**Symptoms:**
- Vacuum pressure higher than design (poorer vacuum)
- TTD above normal range
- Air ejector working harder

**Diagnostic Steps:**

```python
# Run air inleakage assessment
result = await executor.execute_tool(
    "detect_air_inleakage",
    {
        "condenser_id": "COND-001",
        "vacuum_trend": vacuum_data,
        "air_ejector_data": ejector_data
    }
)
```

**Common Causes:**
- Air in-leakage at LP turbine glands
- Expansion joint bellows failure
- Valve packing leaks
- Hotwell level control issues

#### 2. Poor Heat Transfer

**Symptoms:**
- Cleanliness factor declining
- TTD increasing
- Higher vacuum pressure

**Diagnostic Steps:**

```python
# Calculate fouling factor
result = await executor.execute_tool(
    "calculate_fouling_factor",
    {
        "design_U": 3000,
        "actual_U": 2400,
        "tube_material": "titanium",
        "operating_hours": 5000
    }
)
```

**Common Causes:**
- Tube fouling (biological, mineral scale)
- Plugged tubes
- Air blanketing
- Inadequate CW flow

#### 3. High Condensate Subcooling

**Symptoms:**
- Condensate temperature below saturation
- Subcooling > 5 deg C

**Common Causes:**
- Low hotwell level
- Air blanketing at hotwell
- Excessive CW flow
- Tube bank arrangement issues

### Log Analysis

```bash
# View recent logs
kubectl logs -l app=gl-017-condensync -n greenlang --tail=100

# Search for errors
kubectl logs -l app=gl-017-condensync -n greenlang | grep ERROR

# Export logs for analysis
kubectl logs -l app=gl-017-condensync -n greenlang > condensync_logs.txt
```

### Health Check Endpoints

```bash
# Liveness check
curl http://localhost:8017/health/live

# Readiness check (includes dependency checks)
curl http://localhost:8017/health/ready

# Startup check
curl http://localhost:8017/health/startup
```

---

## Compliance

### Industry Standards

GL-017 CONDENSYNC implements calculations and methodologies from:

| Standard | Code | Description |
|----------|------|-------------|
| HEI | HEI-2629 (2022) | Steam Surface Condenser Standards |
| ASME PTC | ASME PTC 12.2 (2022) | Performance Test Code for Condensers |
| TEMA | TEMA (2019) | Tubular Exchanger Standards |
| EPRI | - | Condenser Optimization Guidelines |
| ASME BPV | Section VIII | Pressure Vessel Code |
| ISO | ISO 5167 | Flow Measurement Standards |

### Zero-Hallucination Guarantees

All numeric calculations are performed using deterministic formulas with no LLM involvement:

**LLM Prohibited For:**
- Heat transfer calculations
- Vacuum pressure calculations
- Fouling factor calculations
- Air inleakage calculations
- Performance corrections
- Mass/energy balance calculations

**LLM Allowed For:**
- Natural language report generation
- Recommendation explanations
- Anomaly descriptions
- Maintenance guidance text

### Audit Trail

All calculations include provenance tracking:

```json
{
  "result": 2847.5,
  "provenance_hash": "sha256:a3f2...",
  "timestamp": "2025-01-15T10:30:00Z",
  "calculation_path": [
    "calculate_heat_duty",
    "calculate_lmtd",
    "calculate_u_value"
  ],
  "data_sources": ["SCADA", "Historian"],
  "model_version": "1.0.0"
}
```

### Data Retention

| Data Type | Retention | Archive |
|-----------|-----------|---------|
| Real-time measurements | 90 days | S3 (Parquet) |
| Hourly aggregates | 365 days | S3 |
| Daily summaries | 7 years | S3 |
| Audit logs | 7 years | PostgreSQL |
| Reports | 7 years | S3 |

---

## Contributing Guidelines

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/gl-017-condensync.git
cd gl-017-condensync

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check .
mypy .
```

### Code Standards

- Python 3.11+ with type hints
- PEP 8 style (enforced by ruff)
- 100% type coverage (mypy strict mode)
- Minimum 80% test coverage
- All calculations must include provenance hash

### Pull Request Process

1. Create feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit PR with description of changes
6. Code review required from 2 reviewers
7. Merge after approval

### Calculation Guidelines

When adding new calculations:

1. Reference industry standard (HEI, ASME, etc.)
2. Implement formula exactly as documented
3. Include unit tests with known results
4. Add provenance hash generation
5. Document in TOOLS_README.md

---

## Support

- **Documentation:** https://docs.greenlang.io/agents/GL-017
- **API Reference:** https://docs.greenlang.io/agents/GL-017/api
- **Support Email:** support@greenlang.io
- **Slack Channel:** #gl-017-condensync
- **JIRA Project:** GL-017

---

## License

Copyright 2025 GreenLang. All rights reserved.

---

*GL-017 CONDENSYNC - Optimizing Steam Condenser Performance with Zero-Hallucination Calculations*
