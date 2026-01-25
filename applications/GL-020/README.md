# GL-020 ECONOPULSE

## Economizer Performance Monitoring Agent

| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-020 |
| **Codename** | ECONOPULSE |
| **Name** | EconomizerPerformanceAgent |
| **Category** | Heat Recovery |
| **Type** | Monitor |
| **Version** | 1.0.0 |
| **Status** | Production Ready |

---

## Overview

GL-020 ECONOPULSE is a comprehensive economizer performance monitoring agent designed for industrial boiler systems. It provides real-time monitoring of economizer fouling, heat transfer performance, and efficiency degradation, enabling predictive maintenance through intelligent cleaning optimization.

The agent continuously monitors feedwater temperature gain, flue gas temperature drop, and soot buildup indicators to calculate fouling resistance, generate cleaning alerts, track performance trends, and quantify efficiency losses in both energy (MMBtu) and cost ($) terms.

### Key Features

- **Real-Time Heat Transfer Monitoring**: Continuous calculation of U-value, LMTD, and effectiveness
- **Fouling Detection and Trending**: Automatic fouling resistance (Rf) calculation with severity classification
- **Predictive Cleaning Alerts**: Threshold-based and predictive alerts for optimal cleaning timing
- **Efficiency Loss Quantification**: MMBtu/hr and $/hr impact from fouling degradation
- **Soot Blower Integration**: Closed-loop optimization of cleaning cycles
- **Zero-Hallucination Calculations**: 100% deterministic formulas with full provenance tracking
- **ASME PTC 4.3 Compliance**: Industry-standard heat transfer calculations

### Business Value

| Benefit | Impact |
|---------|--------|
| Soot Blowing Reduction | 10% fewer cleaning cycles through optimized scheduling |
| Efficiency Improvement | 0.5% boiler efficiency gain from reduced fouling |
| Annual Savings | $100K+ for typical industrial boiler installation |
| ROI Payback | 6-12 months typical payback period |
| Unplanned Downtime | Significant reduction through predictive maintenance |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/greenlang/gl-020-econopulse.git
cd gl-020-econopulse

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
import asyncio
from greenlang.GL_020 import (
    EconomizerPerformanceAgent,
    AgentConfiguration,
    EconomizerConfiguration,
    EconomizerType,
)

# Define economizer physical configuration
economizer_config = EconomizerConfiguration(
    economizer_id="ECON-001",
    economizer_type=EconomizerType.FINNED_TUBE,
    economizer_name="Main Boiler Economizer",
    tube_count=200,
    tube_rows=10,
    total_heat_transfer_area_sqft=2500.0,
    design_water_flow_gpm=500.0,
    design_water_inlet_temp_f=200.0,
    design_water_outlet_temp_f=280.0,
    design_gas_inlet_temp_f=550.0,
    design_gas_outlet_temp_f=350.0,
    design_heat_duty_mmbtu_hr=8.0,
)

# Create agent configuration
config = AgentConfiguration(
    economizers=[economizer_config],
    fuel_cost_per_mmbtu=4.0,
    operating_hours_per_year=8000,
    enable_fouling_prediction=True,
    enable_adaptive_cleaning=True,
)

# Initialize and execute agent
agent = EconomizerPerformanceAgent(config)

async def main():
    result = await agent.execute()

    print(f"Performance Status: {result.performance_status.value}")
    print(f"Fouling Rf: {result.performance_metrics.fouling_resistance_rf:.5f} hr-ft2-F/BTU")
    print(f"Effectiveness: {result.performance_metrics.effectiveness_pct:.1f}%")
    print(f"Efficiency Loss: {result.efficiency_loss.cost_loss_usd_hr:.2f} $/hr")

    for alert in result.cleaning_alerts:
        print(f"Alert: {alert.message}")

asyncio.run(main())
```

### Docker Deployment

```bash
# Build container
docker build -t gl-020-econopulse:latest .

# Run with environment variables
docker run -d \
  --name econopulse \
  -e SCADA_HOST=192.168.1.100 \
  -e SCADA_PORT=4840 \
  -p 8000:8000 \
  gl-020-econopulse:latest
```

---

## Configuration

### Agent Configuration

```yaml
# config/agent_config.yaml
agent_name: "GL-020 ECONOPULSE"
version: "1.0.0"
environment: "production"

economizers:
  - economizer_id: "ECON-001"
    economizer_type: "finned_tube"
    tube_count: 200
    total_heat_transfer_area_sqft: 2500.0
    design_water_flow_gpm: 500.0
    design_heat_duty_mmbtu_hr: 8.0

baseline_configuration:
  clean_u_value_btu_hr_ft2_f: 15.0
  expected_effectiveness_pct: 75.0
  expected_approach_temp_f: 50.0
  max_acceptable_fouling_resistance: 0.005
  typical_fouling_rate_per_day: 0.0001

alert_configuration:
  fouling_resistance_threshold:
    warning_high: 0.002
    critical_high: 0.004
  effectiveness_threshold:
    warning_low: 60.0
    critical_low: 50.0
  escalation_enabled: true
  escalation_time_minutes: 30

soot_blower_configuration:
  system_id: "SB-SYS-001"
  media_type: "steam"
  blowing_interval_hours: 8.0
  adaptive_scheduling_enabled: true
  fouling_trigger_rf: 0.003

fuel_cost_per_mmbtu: 4.0
steam_cost_per_klb: 10.0
operating_hours_per_year: 8000
```

### SCADA Integration

```yaml
# config/scada_config.yaml
scada_integration:
  enabled: true
  scada_system: "Wonderware"
  polling_interval_seconds: 10

  # Temperature tags
  water_inlet_temp_tag: "ECON.WATER_IN_TEMP"
  water_outlet_temp_tag: "ECON.WATER_OUT_TEMP"
  gas_inlet_temp_tag: "ECON.GAS_IN_TEMP"
  gas_outlet_temp_tag: "ECON.GAS_OUT_TEMP"

  # Flow tags
  water_flow_tag: "ECON.WATER_FLOW"
  gas_flow_tag: "ECON.GAS_FLOW"

  # Soot blower tags
  soot_blower_active_tag: "ECON.SB_ACTIVE"
  last_soot_blow_time_tag: "ECON.SB_LAST_TIME"
```

---

## API Reference

### Base URL

```
https://api.greenlang.io/gl-020/api/v1
```

### Authentication

All endpoints require JWT Bearer token authentication:

```bash
curl -X GET "https://api.greenlang.io/gl-020/api/v1/economizers" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Core Endpoints

#### Get Performance Metrics

```http
GET /api/v1/economizers/{economizer_id}/performance
```

**Response:**

```json
{
  "economizer_id": "ECON-001",
  "timestamp": "2025-12-03T10:30:00Z",
  "u_value_btu_hr_ft2_f": 12.5,
  "u_value_ratio": 0.833,
  "fouling_resistance_rf": 0.00167,
  "fouling_severity": "LIGHT",
  "lmtd_f": 145.2,
  "heat_duty_mmbtu_hr": 6.8,
  "effectiveness_pct": 72.5,
  "approach_temp_f": 55.0,
  "performance_status": "GOOD"
}
```

#### Get Fouling Status

```http
GET /api/v1/economizers/{economizer_id}/fouling
```

**Response:**

```json
{
  "economizer_id": "ECON-001",
  "timestamp": "2025-12-03T10:30:00Z",
  "current_rf": 0.00167,
  "fouling_rate_per_day": 0.00008,
  "fouling_rate_trend": "STABLE",
  "predicted_cleaning_date": "2025-12-17T10:30:00Z",
  "days_until_cleaning": 14.0,
  "confidence_level": 0.75,
  "cleaning_recommended": false,
  "cleaning_urgency": "ROUTINE"
}
```

#### Trigger Soot Blowing

```http
POST /api/v1/economizers/{economizer_id}/soot-blowers/trigger
```

**Request:**

```json
{
  "sequence": "standard",
  "reason": "High fouling score detected",
  "delay_seconds": 0
}
```

**Response:**

```json
{
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "INITIATED",
  "estimated_duration_minutes": 12,
  "message": "Soot blow sequence initiated"
}
```

#### Get Efficiency Loss Report

```http
GET /api/v1/economizers/{economizer_id}/efficiency/loss
```

**Response:**

```json
{
  "economizer_id": "ECON-001",
  "timestamp": "2025-12-03T10:30:00Z",
  "heat_loss_mmbtu_hr": 0.85,
  "cost_loss_usd_hr": 3.40,
  "projected_annual_loss_usd": 27200.00,
  "fuel_penalty_pct": 1.2,
  "efficiency_degradation_pct": 8.5,
  "recoverable_heat_mmbtu_hr": 0.85
}
```

For complete API documentation, see [API_README.md](./API_README.md).

---

## Architecture Overview

### System Architecture

```
                                    +-------------------+
                                    |   SCADA System    |
                                    |  (OPC-UA/Modbus)  |
                                    +--------+----------+
                                             |
                                             v
+-------------------+           +------------------------+
|   Soot Blower     |<--------->|  Sensor Data Intake    |
|   Integration     |           |  Agent                 |
+-------------------+           +------------------------+
                                             |
                                             v
                                +------------------------+
                                |  Heat Transfer         |
                                |  Calculator Agent      |
                                |  (ASME PTC 4.3)        |
                                +------------------------+
                                             |
                                             v
                                +------------------------+
                                |  Fouling Analysis      |
                                |  Agent                 |
                                +------------------------+
                                             |
                                             v
                                +------------------------+
                                |  Alert Generation      |
                                |  Agent                 |
                                +------------------------+
                                             |
                                             v
                                +------------------------+
                                |  Performance Trending  |
                                |  Agent                 |
                                +------------------------+
                                             |
                                             v
                                +------------------------+
                                |  Reporting &           |
                                |  Visualization         |
                                +------------------------+
                                             |
                      +----------------------+----------------------+
                      |                      |                      |
                      v                      v                      v
              +---------------+      +---------------+      +---------------+
              |   REST API    |      |   WebSocket   |      |   Grafana     |
              |   (FastAPI)   |      |   (Real-time) |      |   Dashboards  |
              +---------------+      +---------------+      +---------------+
```

### Data Flow

1. **Sensor Data Ingestion**: Temperature, flow, and pressure readings from SCADA
2. **Heat Transfer Calculation**: LMTD, U-value, heat duty, effectiveness (ASME PTC 4.3)
3. **Fouling Analysis**: Rf calculation, severity classification, rate trending
4. **Alert Generation**: Threshold and predictive alerts with recommended actions
5. **Performance Trending**: Load-corrected trends, efficiency loss quantification
6. **Reporting**: Real-time dashboards, scheduled reports, cleaning effectiveness tracking

---

## Calculation Methodology

### Zero-Hallucination Guarantee

All calculations in GL-020 ECONOPULSE use deterministic formulas from recognized industry standards. No AI/ML models are used for core calculations - only for narrative generation in reports.

### Heat Transfer Calculations (ASME PTC 4.3)

#### Log Mean Temperature Difference (LMTD)

For counter-flow configuration:

```
LMTD = (dT1 - dT2) / ln(dT1/dT2)

where:
  dT1 = T_gas_in - T_water_out   (hot end approach)
  dT2 = T_gas_out - T_water_in   (cold end approach)
```

#### Heat Duty

```
Q = m_dot * Cp * (T_out - T_in)

where:
  m_dot = mass flow rate (lb/hr)
  Cp = specific heat (BTU/lb-F)
```

#### Overall Heat Transfer Coefficient (U-value)

```
U = Q / (A * LMTD)

where:
  Q = heat duty (BTU/hr)
  A = heat transfer surface area (ft2)
  LMTD = log mean temperature difference (F)
```

### Fouling Calculations (TEMA)

#### Fouling Resistance

```
Rf = (1/U_fouled) - (1/U_clean)

where:
  Rf = fouling factor (hr-ft2-F/BTU)
  U_fouled = current U-value
  U_clean = clean baseline U-value
```

#### Fouling Severity Classification

| Rf (hr-ft2-F/BTU) | Severity | Action |
|-------------------|----------|--------|
| < 0.001 | CLEAN | Monitor |
| 0.001 - 0.002 | LIGHT | Routine monitoring |
| 0.002 - 0.003 | MODERATE | Schedule cleaning |
| 0.003 - 0.004 | HEAVY | Plan immediate cleaning |
| > 0.004 | SEVERE | Emergency cleaning required |

### Effectiveness (Epsilon-NTU Method)

```
epsilon = Q_actual / Q_max
        = (T_water_out - T_water_in) / (T_gas_in - T_water_in)
```

### Efficiency Loss Quantification

```
Heat_Loss (MMBtu/hr) = Q_design * (1 - U_fouled/U_clean)
Cost_Loss ($/hr) = Heat_Loss * Fuel_Cost
Annual_Loss ($) = Cost_Loss * Operating_Hours
```

For complete formula documentation, see [FORMULA_LIBRARY.md](./FORMULA_LIBRARY.md).

---

## Inputs and Outputs

### Required Inputs

| Input | Source | Units | Range |
|-------|--------|-------|-------|
| Feedwater Inlet Temperature | RTD/Thermocouple | F | 100-500 |
| Feedwater Outlet Temperature | RTD/Thermocouple | F | 150-600 |
| Flue Gas Inlet Temperature | RTD/Thermocouple | F | 400-1000 |
| Flue Gas Outlet Temperature | RTD/Thermocouple | F | 200-600 |
| Feedwater Flow Rate | Orifice/Ultrasonic | lb/hr | 0-500,000 |
| Flue Gas Flow Rate | Orifice/Pitot | SCFH | 0-5,000,000 |
| Differential Pressure | DP Transmitter | inH2O | 0-5 |
| Soot Blower Status | Limit Switches | Boolean | On/Off |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Performance Metrics | Real-time | U-value, LMTD, effectiveness, NTU |
| Fouling Analysis | Real-time | Rf, severity, rate, cleaning prediction |
| Cleaning Alerts | Event-driven | Threshold, predictive, rate-of-change |
| Efficiency Loss Report | Periodic | Heat loss, cost, fuel penalty |
| Performance Trends | Periodic | Hourly, daily, weekly averages |
| Cleaning Recommendations | On-demand | Method, urgency, ROI |

---

## Integration Points

### SCADA Systems

- OPC-UA client for modern SCADA systems
- Modbus TCP/RTU for legacy systems
- Profinet and EtherNet/IP support
- Data quality validation and timestamp synchronization

### Soot Blower Systems

- Automatic trigger on fouling thresholds
- Interlock checking (steam availability, cooldown period)
- Effectiveness tracking (pre/post Rf comparison)
- Sequence optimization based on fouling patterns

### Historian Integration

- OSIsoft PI historian support
- AspenTech InfoPlus.21 integration
- InfluxDB time-series storage
- Automatic data backfill capability

### CMMS Integration

- Work order generation for cleaning
- Maintenance scheduling coordination
- Equipment history tracking
- Cost tracking and reporting

---

## Alert Types

| Alert Type | Trigger | Priority | Action |
|------------|---------|----------|--------|
| FOULING_THRESHOLD | Rf > threshold | CRITICAL/WARNING | Initiate cleaning |
| FOULING_RATE | dRf/dt > limit | HIGH | Investigate cause |
| PREDICTIVE | Days to clean < 3 | MEDIUM | Schedule cleaning |
| EFFECTIVENESS_LOW | Eff < 60% | CRITICAL | Immediate attention |
| APPROACH_HIGH | Approach > 80F | WARNING | Monitor closely |
| SENSOR_FAULT | Quality = BAD | HIGH | Check instrumentation |

---

## Security

### Authentication

- OAuth2 + JWT token authentication
- Role-based access control (RBAC)
- Service accounts for SCADA integration

### Data Protection

- TLS 1.3 for all communications
- AES-256 encryption at rest
- SHA-256 provenance hashing for audit trail

### Industrial Security

- IEC 62443 target compliance
- OPC-UA Sign+Encrypt mode
- Network segmentation support

---

## Testing

### Run Tests

```bash
# Full test suite
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m asme          # ASME PTC 4.3 validation
pytest -m iapws         # IAPWS-IF97 validation
```

### Test Coverage

| Component | Target Coverage |
|-----------|----------------|
| Heat Transfer Calculator | 95% |
| Fouling Calculator | 95% |
| Alert Manager | 95% |
| Integration Tests | 85% |
| **Overall** | **90%+** |

---

## Support

- **Documentation**: https://docs.greenlang.io/gl-020
- **API Status**: https://status.greenlang.io
- **Support Email**: support@greenlang.io
- **Community**: https://community.greenlang.io

---

## License

Copyright 2025 GreenLang. All rights reserved.

---

## References

1. ASME PTC 4.3-2017, "Performance Test Code for Air Heaters"
2. ASME PTC 4-2013, "Fired Steam Generators"
3. TEMA Standards, 10th Edition, "Standards of the Tubular Exchanger Manufacturers Association"
4. IAPWS-IF97, "Industrial Formulation 1997 for Thermodynamic Properties of Water and Steam"
5. EPRI Fouling Management Guidelines
