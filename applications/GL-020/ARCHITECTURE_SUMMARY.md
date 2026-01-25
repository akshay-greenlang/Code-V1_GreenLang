# GL-020 ECONOPULSE - Architecture Design Summary

**Document Status**: Architecture Design Complete
**Created**: December 3, 2025
**Target Release**: Q1 2026
**Priority**: P1 (High Priority)

---

## Executive Summary

GL-020 ECONOPULSE (EconomizerPerformanceAgent) architecture design is complete and production-ready for development. This application provides comprehensive heat recovery monitoring for economizers in industrial boiler systems, enabling predictive maintenance through fouling detection, cleaning optimization, and efficiency loss quantification.

### Application Overview

| Attribute | Value |
|-----------|-------|
| Agent ID | GL-020 |
| Codename | ECONOPULSE |
| Name | EconomizerPerformanceAgent |
| Category | Heat Recovery |
| Type | Monitor |
| Description | Monitors economizer performance and fouling |
| Inputs | Feedwater temperature gain + flue gas temperature + soot buildup |
| Outputs | Cleaning alerts + performance trends + efficiency loss |
| Integration | Economizer instrumentation + soot blowers |

### Key Differentiators

1. **ASME PTC 4.3 Compliant Calculations** for heat transfer analysis
2. **Zero-hallucination** fouling factor and efficiency calculations
3. **Real-time + Batch Processing** for immediate alerts and trend analysis
4. **Predictive Maintenance** with fouling rate regression modeling
5. **Automated Soot Blower Integration** for closed-loop cleaning optimization
6. **Multi-tier Alert Escalation** with SMS, Email, Slack, PagerDuty support

---

## Core Components Description

### 1. Sensor Data Ingestion Layer

The Sensor Data Ingestion Layer acquires real-time data from economizer instrumentation through industrial protocols.

**Key Capabilities:**
- **Multi-Protocol Support**: OPC-UA, Modbus TCP/RTU, Profinet, EtherNet/IP
- **Sensor Types**:
  - Feedwater temperature sensors (inlet/outlet) - RTD/Thermocouple
  - Flue gas temperature sensors (inlet/outlet) - RTD/Thermocouple
  - Flow rate sensors (water and gas) - Orifice/Vortex/Ultrasonic
  - Differential pressure sensors (tube bank dP) - DP Transmitter
  - Soot blower position/status - Limit switches/Encoders
- **Data Validation**: Range checks, quality flags, timestamp synchronization
- **Provenance**: SHA-256 hashing of all sensor data for audit trail

**Sensor Specifications:**

| Sensor | Tag | Range | Accuracy | Poll Rate |
|--------|-----|-------|----------|-----------|
| Feedwater Temp In | TE-001 | 100-500 F | +/- 1.0 F | 1 sec |
| Feedwater Temp Out | TE-002 | 150-600 F | +/- 1.0 F | 1 sec |
| Flue Gas Temp In | TE-003 | 400-1000 F | +/- 2.0 F | 1 sec |
| Flue Gas Temp Out | TE-004 | 200-600 F | +/- 2.0 F | 1 sec |
| Feedwater Flow | FT-001 | 0-500,000 lb/hr | +/- 0.5% | 1 sec |
| Flue Gas Flow | FT-002 | 0-5,000,000 SCFH | +/- 1.0% | 5 sec |
| Differential Pressure | PDT-001 | 0-5 inH2O | +/- 0.1% | 1 sec |
| Feedwater Pressure | PT-001 | 0-500 psig | +/- 0.25% | 5 sec |

### 2. Heat Transfer Calculation Engine

The Heat Transfer Calculation Engine implements ASME PTC 4.3 compliant formulas for economizer performance analysis.

**Key Formulas:**

**Log Mean Temperature Difference (LMTD):**
```
LMTD = (dT1 - dT2) / ln(dT1/dT2)

where:
  dT1 = T_fg_in - T_fw_out   (hot end temperature approach)
  dT2 = T_fg_out - T_fw_in   (cold end temperature approach)
```

**Heat Duty (Q):**
```
Q = m_dot * Cp * (T_out - T_in)

where:
  m_dot = feedwater mass flow rate (lb/hr)
  Cp = specific heat of water (~1.0 BTU/lb-F)
  T_out, T_in = feedwater outlet/inlet temperature (F)
```

**Overall Heat Transfer Coefficient (U):**
```
U = Q / (A * LMTD)

where:
  Q = heat duty (BTU/hr)
  A = heat transfer surface area (ft2)
  LMTD = log mean temperature difference (F)

Units: BTU/(hr-ft2-F)
```

**Effectiveness (epsilon):**
```
epsilon = Q_actual / Q_max

where:
  Q_max = C_min * (T_fg_in - T_fw_in)
  C_min = min(m_fg * Cp_fg, m_fw * Cp_fw)
```

**Design Compliance:**
- ASME PTC 4.3-2017 (Air Heaters)
- ASME PTC 4.1 (Steam Generating Units)
- TEMA Standards (Tubular Exchanger Manufacturers Association)

### 3. Fouling Analysis Module

The Fouling Analysis Module tracks heat exchanger degradation and predicts cleaning requirements.

**Fouling Factor Calculation:**
```
Rf = (1/U_actual) - (1/U_clean)

Units: hr-ft2-F/BTU (or m2-K/W in SI)
```

**Fouling Rate Prediction:**
```
dRf/dt = Linear regression slope of Rf vs. time

Using least squares regression on configurable time window
(default: 7 days of data)
```

**Time to Cleaning Threshold:**
```
t_clean = (Rf_threshold - Rf_current) / (dRf/dt)

Result in hours until cleaning is required
```

**Fouling Classification:**

| Severity | Rf Range (hr-ft2-F/BTU) | Action Required |
|----------|-------------------------|-----------------|
| LOW | 0 - 0.0005 | Monitor |
| MEDIUM | 0.0005 - 0.001 | Schedule cleaning |
| HIGH | 0.001 - 0.002 | Cleaning soon |
| CRITICAL | > 0.002 | Immediate cleaning |

**Fouling Types:**
- **Gas-Side Fouling**: Ash deposition, soot buildup, slag formation
- **Water-Side Fouling**: Scale formation, corrosion products, biological growth

**Ash Deposition Modeling:**
- Differential pressure correlation
- Temperature profile analysis
- Soot blower effectiveness tracking

### 4. Alert Generation System

The Alert Generation System provides multi-tier alerting with configurable thresholds and escalation paths.

**Alert Types:**

| Alert Type | Trigger Condition | Default Priority |
|------------|-------------------|------------------|
| FOULING_THRESHOLD_EXCEEDED | Rf > threshold | CRITICAL |
| FOULING_RATE_HIGH | dRf/dt > rate_limit | HIGH |
| CLEANING_PREDICTED_SOON | time_to_clean < 72 hr | MEDIUM |
| CLEANING_PREDICTED_WEEK | time_to_clean < 168 hr | LOW |
| DIFF_PRESSURE_HIGH | dP > 2.5 inH2O | HIGH |
| DIFF_PRESSURE_CRITICAL | dP > 3.5 inH2O | CRITICAL |
| SOOT_BLOWER_FAULT | Status = FAULT | HIGH |
| SENSOR_FAULT | Quality = BAD | MEDIUM |
| EFFICIENCY_LOSS_HIGH | Loss > 20% | MEDIUM |
| EFFICIENCY_LOSS_CRITICAL | Loss > 35% | CRITICAL |
| U_VALUE_DEVIATION | Deviation > 15% | LOW |
| TEMP_APPROACH_LOW | dT2 < 50 F | MEDIUM |

**Alert Prioritization:**

| Priority | Response Time | Notification Channels |
|----------|---------------|----------------------|
| CRITICAL | Immediate | SMS, PagerDuty, Dashboard |
| HIGH | 4 hours | Email, Slack, Dashboard |
| MEDIUM | 24 hours | Email, Dashboard |
| LOW | Informational | Daily digest, Dashboard |

**Soot Blower Integration:**
- Automatic trigger on threshold alerts
- Interlock checking (steam availability, cooldown period)
- Effectiveness tracking (pre/post dP comparison)
- Sequence optimization based on fouling patterns

### 5. Performance Trending Engine

The Performance Trending Engine provides real-time and historical performance analysis with load correction.

**Trend Metrics:**

| Metric | Window | Purpose |
|--------|--------|---------|
| U-value (1-hr avg) | 60 min | Real-time monitoring |
| U-value (8-hr avg) | 8 hr | Shift-level trending |
| U-value (24-hr avg) | 24 hr | Daily performance |
| Fouling Factor | Rolling | Degradation tracking |
| Efficiency Loss | Rolling | Economic impact |

**Load-Corrected Performance:**
```
U_corrected = U_actual * (Load_design / Load_actual)^n

where:
  n = correction factor (typically 0.8-1.0)
  Load = % of rated capacity
```

**Ambient Temperature Correction:**
```
U_corrected = U_actual * (T_ambient_design / T_ambient_actual)^m

where:
  m = seasonal correction factor
```

**Efficiency Loss Quantification:**
```
Efficiency Loss % = (U_clean - U_actual) / U_clean * 100

Heat Recovery Loss (BTU/hr) = Loss_% * Q_design

Fuel Penalty ($/hr) = Heat_Loss / 1,000,000 * Fuel_Price
```

**Baseline Management:**
- Baseline established after cleaning
- Automatic baseline update after validated cleaning event
- Manual baseline reset capability
- Baseline drift detection (gradual degradation of "clean" state)

### 6. Reporting and Visualization

The Reporting and Visualization module generates real-time dashboards and periodic reports.

**Real-time Dashboards (Grafana):**

| Dashboard | Key Visualizations |
|-----------|-------------------|
| Performance Overview | U-value gauge, efficiency trend, fouling status |
| Heat Transfer Analysis | LMTD trend, temperature profiles, heat duty |
| Fouling Monitoring | Rf trend, dP trend, time-to-clean countdown |
| Cleaning History | Pre/post metrics, recovery %, ROI tracking |
| Alert Management | Active alerts, acknowledgment status, history |

**Scheduled Reports:**

| Report | Frequency | Contents |
|--------|-----------|----------|
| Daily Summary | Daily | 24-hr metrics, alerts, efficiency loss |
| Weekly Performance | Weekly | Trend analysis, cleaning events, ROI |
| Monthly Compliance | Monthly | Equipment health, KPIs, recommendations |
| Cleaning Effectiveness | Per event | Pre/post comparison, recovery analysis |

**Export Formats:**
- PDF (performance reports, compliance documentation)
- Excel (raw data export, custom analysis)
- CSV (data integration, external tools)
- JSON (API integration, automation)

**Cleaning History Analysis:**
- Cleaning event log with metrics
- Recovery effectiveness trending
- Optimal cleaning interval calculation
- Cost-benefit analysis (cleaning cost vs. efficiency gain)

---

## Agent Pipeline Architecture

### Pipeline Overview (6 Agents)

```
SensorDataIntakeAgent -> HeatTransferCalculatorAgent -> FoulingAnalysisAgent
                                                            |
                                                            v
                                               AlertGenerationAgent
                                                            |
                                                            v
                                            PerformanceTrendingAgent
                                                            |
                                                            v
                                          ReportingVisualizationAgent
```

### Agent 1: SensorDataIntakeAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Acquire and validate sensor data from economizer instrumentation |
| Inputs | OPC-UA/Modbus tags for temperature, flow, pressure, soot blower status |
| Processing | Range validation, quality flag check, unit conversion, timestamp sync |
| Outputs | SensorData object (validated, normalized, hashed) |
| Estimated LOC | 450-550 |
| LLM Usage | None (100% deterministic) |

**Data Model:**
```python
@dataclass
class SensorData:
    measurement_id: UUID
    economizer_id: str
    timestamp: datetime

    # Feedwater side
    feedwater_temp_in_f: Decimal
    feedwater_temp_out_f: Decimal
    feedwater_flow_lb_hr: Decimal
    feedwater_pressure_psig: Decimal

    # Flue gas side
    flue_gas_temp_in_f: Decimal
    flue_gas_temp_out_f: Decimal
    flue_gas_flow_scfh: Decimal
    differential_pressure_inh2o: Decimal

    # Soot blower status
    soot_blower_statuses: Dict[str, str]
    last_soot_blow_timestamp: Optional[datetime]

    # Quality
    data_quality: str  # GOOD, BAD, UNCERTAIN
    provenance_hash: str
```

### Agent 2: HeatTransferCalculatorAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Calculate heat transfer performance metrics using ASME PTC 4.3 formulas |
| Inputs | SensorData object |
| Processing | LMTD calculation, heat duty, U-value, effectiveness |
| Outputs | HeatTransferResult object |
| Estimated LOC | 350-450 |
| LLM Usage | None (100% deterministic) |

**Data Model:**
```python
@dataclass
class HeatTransferResult:
    calculation_id: UUID
    economizer_id: str
    timestamp: datetime
    measurement_id: UUID

    # Temperature differences
    delta_t1_f: Decimal  # Hot end approach
    delta_t2_f: Decimal  # Cold end approach
    lmtd_f: Decimal      # Log mean temperature difference

    # Heat transfer metrics
    heat_duty_btu_hr: Decimal
    u_value_actual: Decimal  # BTU/(hr-ft2-F)
    effectiveness: Decimal   # 0-1

    # Reference values
    surface_area_ft2: Decimal
    u_value_clean: Decimal  # Baseline

    # Audit
    formula_version: str  # ASME_PTC_4.3_2017
    provenance_hash: str
```

### Agent 3: FoulingAnalysisAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Calculate fouling factor, predict degradation, classify severity |
| Inputs | HeatTransferResult object, historical data |
| Processing | Rf calculation, regression analysis, time-to-threshold prediction |
| Outputs | FoulingAnalysis object |
| Estimated LOC | 500-600 |
| LLM Usage | None (100% deterministic) |

**Data Model:**
```python
@dataclass
class FoulingAnalysis:
    analysis_id: UUID
    economizer_id: str
    timestamp: datetime
    calculation_id: UUID

    # Fouling metrics
    fouling_factor_rf: Decimal  # hr-ft2-F/BTU
    fouling_rate_per_day: Decimal  # dRf/dt
    time_to_threshold_hours: Decimal

    # Performance impact
    efficiency_loss_percent: Decimal
    heat_recovery_loss_btu_hr: Decimal
    fuel_penalty_usd_hr: Decimal

    # Classification
    fouling_type: str  # GAS_SIDE, WATER_SIDE, COMBINED
    fouling_severity: str  # LOW, MEDIUM, HIGH, CRITICAL

    # Reference
    threshold_rf: Decimal
    baseline_timestamp: datetime
    provenance_hash: str
```

### Agent 4: AlertGenerationAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Generate alerts based on thresholds, rates, and predictions |
| Inputs | FoulingAnalysis object, configured thresholds |
| Processing | Threshold comparison, rate analysis, predictive logic |
| Outputs | AlertSet object (prioritized alerts with actions) |
| Estimated LOC | 400-500 |
| LLM Usage | None (100% deterministic) |

**Data Model:**
```python
@dataclass
class Alert:
    alert_id: UUID
    economizer_id: str
    timestamp: datetime

    alert_type: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW

    message: str
    details: Dict[str, Any]
    recommended_actions: List[str]

    status: str  # ACTIVE, ACKNOWLEDGED, RESOLVED
    provenance_hash: str

@dataclass
class AlertSet:
    alerts: List[Alert]
    soot_blower_trigger: bool
    soot_blower_sequence: Optional[List[str]]
```

### Agent 5: PerformanceTrendingAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Calculate trends, apply corrections, quantify efficiency loss |
| Inputs | HeatTransferResult, FoulingAnalysis, historical data |
| Processing | Moving averages, load correction, seasonal adjustment |
| Outputs | PerformanceTrend object |
| Estimated LOC | 450-550 |
| LLM Usage | None (100% deterministic) |

**Data Model:**
```python
@dataclass
class PerformanceTrend:
    trend_id: UUID
    economizer_id: str
    timestamp: datetime

    # Trend metrics
    u_value_1hr_avg: Decimal
    u_value_8hr_avg: Decimal
    u_value_24hr_avg: Decimal

    # Load-corrected performance
    load_percent: Decimal
    u_value_corrected: Decimal
    ambient_temp_correction: Decimal

    # Efficiency loss trending
    cumulative_loss_btu: Decimal
    cumulative_loss_usd: Decimal

    # Comparison
    percent_of_baseline: Decimal
    days_since_cleaning: int

    provenance_hash: str
```

### Agent 6: ReportingVisualizationAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Generate dashboards, reports, notifications, and control commands |
| Inputs | All upstream agent outputs |
| Processing | Data formatting, report generation, notification dispatch |
| Outputs | Reports (PDF, Excel), dashboard data, alerts, soot blower commands |
| Estimated LOC | 550-650 |
| LLM Usage | Narrative generation ONLY (no calculations) |

**LLM Usage (Restricted to Narrative):**
- Natural language performance summaries
- Cleaning recommendation explanations
- Trend interpretation text
- Operator guidance messages

**NO LLM for:**
- Any calculations
- Alert threshold decisions
- Soot blower trigger logic
- Efficiency quantification

### Total Estimated Code

| Component | Lines of Code |
|-----------|---------------|
| Agent 1: SensorDataIntakeAgent | 450-550 |
| Agent 2: HeatTransferCalculatorAgent | 350-450 |
| Agent 3: FoulingAnalysisAgent | 500-600 |
| Agent 4: AlertGenerationAgent | 400-500 |
| Agent 5: PerformanceTrendingAgent | 450-550 |
| Agent 6: ReportingVisualizationAgent | 550-650 |
| API Layer (FastAPI) | 600-800 |
| SCADA/Integration Layer | 600-800 |
| Database Layer | 400-500 |
| Configuration & Utilities | 300-400 |
| **Total** | **4,600-5,800** |

---

## API Design

### Base URL

```
https://api.greenlang.io/v1/econopulse
```

### Authentication

All endpoints require OAuth2 + JWT authentication.

```
Authorization: Bearer <jwt_token>
```

### Endpoints

#### Sensor Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/sensors/data` | Submit sensor data (high-throughput) |
| GET | `/sensors/data/{economizer_id}` | Get latest sensor readings |
| GET | `/sensors/data/{economizer_id}/history` | Get historical sensor data |
| GET | `/sensors/status` | Get all sensor status |

#### Heat Transfer Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analysis/heat-transfer/{economizer_id}` | Get latest heat transfer metrics |
| GET | `/analysis/heat-transfer/{economizer_id}/history` | Get historical calculations |
| POST | `/analysis/heat-transfer/calculate` | Calculate for provided data |

#### Fouling Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/fouling/{economizer_id}` | Get current fouling status |
| GET | `/fouling/{economizer_id}/history` | Get fouling history |
| GET | `/fouling/{economizer_id}/prediction` | Get time-to-clean prediction |
| POST | `/fouling/baseline/{economizer_id}` | Reset baseline |

#### Alerts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/alerts` | List all active alerts |
| GET | `/alerts/{economizer_id}` | Get alerts for economizer |
| POST | `/alerts/{alert_id}/acknowledge` | Acknowledge alert |
| POST | `/alerts/{alert_id}/resolve` | Resolve alert |
| GET | `/alerts/history` | Get alert history |

#### Performance Trends

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/trends/{economizer_id}` | Get current trends |
| GET | `/trends/{economizer_id}/history` | Get trend history |
| GET | `/trends/{economizer_id}/efficiency-loss` | Get efficiency loss summary |

#### Cleaning Events

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/cleaning/{economizer_id}` | Get cleaning history |
| POST | `/cleaning/{economizer_id}` | Record cleaning event |
| GET | `/cleaning/{economizer_id}/roi` | Get cleaning ROI analysis |

#### Soot Blower Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/soot-blower/{economizer_id}/status` | Get soot blower status |
| POST | `/soot-blower/{economizer_id}/trigger` | Trigger soot blow sequence |
| GET | `/soot-blower/{economizer_id}/history` | Get soot blow history |

#### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/reports/daily/{economizer_id}` | Get daily report (PDF) |
| GET | `/reports/weekly/{economizer_id}` | Get weekly report (PDF) |
| GET | `/reports/export/{economizer_id}` | Export data (CSV/Excel) |

#### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/config/economizers` | List configured economizers |
| GET | `/config/economizers/{id}` | Get economizer config |
| PUT | `/config/economizers/{id}` | Update economizer config |
| GET | `/config/thresholds/{id}` | Get alert thresholds |
| PUT | `/config/thresholds/{id}` | Update thresholds |

#### WebSocket

| Endpoint | Description |
|----------|-------------|
| `WS /ws/live/{economizer_id}` | Real-time sensor data stream |
| `WS /ws/alerts` | Real-time alert stream |
| `WS /ws/dashboard/{economizer_id}` | Dashboard data stream |

### Example Request/Response

**Get Current Fouling Status:**

```http
GET /fouling/ECO-001
Authorization: Bearer <token>
```

**Response:**

```json
{
  "economizer_id": "ECO-001",
  "timestamp": "2025-12-03T14:30:00Z",
  "fouling_factor_rf": 0.00085,
  "fouling_rate_per_day": 0.000015,
  "time_to_threshold_hours": 96.7,
  "fouling_severity": "MEDIUM",
  "fouling_type": "GAS_SIDE",
  "efficiency_loss_percent": 12.3,
  "heat_recovery_loss_btu_hr": 1850000,
  "fuel_penalty_usd_hr": 6.48,
  "recommendation": "Schedule soot blowing within 4 days",
  "provenance_hash": "a1b2c3d4e5f6..."
}
```

**Trigger Soot Blow:**

```http
POST /soot-blower/ECO-001/trigger
Authorization: Bearer <token>
Content-Type: application/json

{
  "sequence": ["SB-001", "SB-002"],
  "reason": "FOULING_THRESHOLD",
  "initiated_by": "AUTO_SYSTEM"
}
```

**Response:**

```json
{
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "INITIATED",
  "sequence": ["SB-001", "SB-002"],
  "estimated_duration_minutes": 12,
  "message": "Soot blow sequence initiated. Monitor status via WebSocket."
}
```

---

## Technology Stack

### Core Framework

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.11+ |
| Framework | FastAPI | 0.104.0+ |
| Server | Uvicorn | 0.24.0+ |
| Task Queue | Celery | 5.3.0+ |
| Message Broker | RabbitMQ | 3.12+ |

### Data Processing

| Component | Technology | Version |
|-----------|------------|---------|
| Validation | Pydantic | 2.5.0+ |
| Data Manipulation | Pandas | 2.1.0+ |
| Numerical | NumPy | 1.24.0+ |
| Statistical | SciPy | 1.11.0+ |
| Time Series | statsmodels | 0.14.0+ |

### Database & Cache

| Component | Technology | Version |
|-----------|------------|---------|
| Time-Series DB | TimescaleDB | 2.13+ |
| Relational DB | PostgreSQL | 14+ |
| ORM | SQLAlchemy | 2.0.0+ |
| Cache | Redis | 7.0+ |
| Connection Pool | PgBouncer | 1.21+ |

### Industrial Integration

| Component | Technology | Version |
|-----------|------------|---------|
| OPC-UA Client | asyncua | 1.0.0+ |
| Modbus Client | pymodbus | 3.5.0+ |
| Protocol Conversion | python-snap7 | 1.3+ (optional) |

### AI/LLM (Narrative Only)

| Component | Technology | Version |
|-----------|------------|---------|
| LLM Provider | Anthropic Claude | Sonnet 4.5 |
| Usage | Narrative generation ONLY | N/A |
| SDK | anthropic | 0.18.0+ |

### Reporting

| Component | Technology | Version |
|-----------|------------|---------|
| PDF Generation | reportlab | 4.0.0+ |
| Excel | openpyxl | 3.1.0+ |
| Charts | matplotlib | 3.8.0+ |
| Dashboards | Grafana | 10.0+ |

### Security

| Component | Technology | Version |
|-----------|------------|---------|
| JWT | python-jose | 3.3.0+ |
| Encryption | cryptography | 41.0.0+ |
| Hashing | hashlib (stdlib) | N/A |
| Secrets | hvac (Vault) | 2.0.0+ |

### Deployment

| Component | Technology | Version |
|-----------|------------|---------|
| Containers | Docker | 24.0+ |
| Orchestration | Kubernetes | 1.28+ |
| IaC | Terraform | 1.6.0+ |
| CI/CD | GitHub Actions | N/A |

### Monitoring

| Component | Technology | Version |
|-----------|------------|---------|
| Metrics | Prometheus | 2.47+ |
| Visualization | Grafana | 10.0+ |
| Logging | structlog | 23.2.0+ |
| Alerting | Alertmanager | 0.26+ |

---

## Security Architecture

### Authentication & Authorization

| Component | Approach |
|-----------|----------|
| Authentication | OAuth2 + JWT tokens |
| Token Expiry | 1 hour (configurable) |
| Refresh Tokens | 7 days (configurable) |
| Service Accounts | For SCADA integration |

### Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| ADMIN | Full access, configuration, user management |
| ENGINEER | Read all, modify thresholds, trigger soot blowers |
| OPERATOR | Read all, acknowledge alerts, view reports |
| VIEWER | Read-only dashboard access |
| SERVICE | Sensor data ingestion only |

### Data Security

| Layer | Protection |
|-------|------------|
| Transport | TLS 1.3 (HTTPS) |
| At Rest | AES-256 encryption (PostgreSQL TDE) |
| SCADA Network | Isolated network segment, VPN |
| Secrets | HashiCorp Vault |
| Audit | SHA-256 provenance hashing |

### Industrial Security

| Control | Implementation |
|---------|----------------|
| OPC-UA Security | Sign+Encrypt, certificate-based auth |
| Modbus Security | IP whitelisting, VPN |
| Network Segmentation | SCADA on separate VLAN |
| Firewall Rules | Explicit allow for ports 4840, 502 |

### Compliance

| Standard | Status |
|----------|--------|
| IEC 62443 | Target (Industrial Cybersecurity) |
| SOC 2 Type II | Target |
| ISO 27001 | Target |

---

## Performance Targets

### Real-Time Processing

| Metric | Target | Justification |
|--------|--------|---------------|
| Sensor Poll Rate | 1-5 seconds | Real-time monitoring |
| Calculation Latency | < 200 ms | Dashboard responsiveness |
| Alert Generation | < 500 ms | Timely notification |
| WebSocket Update | < 1 second | Live dashboard |

### Batch Processing

| Metric | Target | Justification |
|--------|--------|---------------|
| Trend Calculation | < 5 seconds | 15-minute batches |
| Report Generation | < 30 seconds | On-demand PDF |
| Data Export | < 60 seconds | 30-day export |

### Throughput

| Metric | Target | Justification |
|--------|--------|---------------|
| Sensor Data Ingestion | 10,000 points/min | Multi-economizer support |
| Concurrent Economizers | 50+ | Enterprise deployment |
| API Requests | 500/min | Dashboard + integrations |

### Availability

| Metric | Target | Justification |
|--------|--------|---------------|
| System Availability | 99.9% | Critical monitoring |
| SCADA Connectivity | 99.5% | Network variability |
| Alert Delivery | 99.9% | Safety-critical |

### Scalability

| Metric | Approach |
|--------|----------|
| Horizontal Scaling | Kubernetes HPA |
| Database Scaling | TimescaleDB hypertables + continuous aggregates |
| Cache Scaling | Redis Cluster |

---

## Caching Strategy

### Cache Layers

| Layer | Purpose | TTL | Technology |
|-------|---------|-----|------------|
| L1 (In-Memory) | Hot sensor data | 30 seconds | Python dict |
| L2 (Redis) | Recent calculations | 5 minutes | Redis |
| L3 (Redis) | Configuration | 1 hour | Redis |
| L4 (TimescaleDB) | Continuous aggregates | Materialized | TimescaleDB |

### Cache Hit Targets

| Data Type | Cache Hit Rate | Impact |
|-----------|----------------|--------|
| Current Sensor Data | 95% | Reduce SCADA polls |
| Latest Calculations | 90% | API response time |
| Configuration | 99% | Eliminate DB reads |
| Historical Aggregates | 80% | Report generation speed |

**Cost Reduction Target**: 66% through caching

---

## Database Design

### TimescaleDB Hypertables

| Table | Chunk Interval | Retention |
|-------|----------------|-----------|
| economizer_measurements | 1 day | 90 days (raw) |
| heat_transfer_calculations | 1 day | 90 days |
| fouling_analysis | 1 day | 1 year |
| performance_trends | 1 day | 1 year |
| alerts | 1 week | 2 years |

### Continuous Aggregates

| Aggregate | Interval | Refresh |
|-----------|----------|---------|
| economizer_1min | 1 minute | Real-time |
| economizer_1hour | 1 hour | Real-time |
| economizer_1day | 1 day | 1 hour lag |

### Indexing Strategy

| Table | Index | Type |
|-------|-------|------|
| economizer_measurements | (economizer_id, timestamp) | B-tree |
| alerts | (tenant_id, status, priority) | B-tree |
| cleaning_events | (economizer_id, started_at) | B-tree |

### Connection Pooling

| Parameter | Value |
|-----------|-------|
| Pool Size | 20 connections |
| Max Overflow | 10 connections |
| Pool Timeout | 30 seconds |
| Recycling | 1800 seconds |

---

## Development Timeline: 10 Weeks

### Team Size: 3-4 Engineers

| Role | Count | Focus |
|------|-------|-------|
| Senior Backend Engineer | 2 | Core agents, calculations, API |
| DevOps Engineer | 1 | Infrastructure, SCADA integration |
| Integration Engineer | 1 | OPC-UA/Modbus, soot blower control |

### Phase 1: Core Agents (3 weeks)

**Week 1:**
- Project setup, CI/CD pipeline
- SensorDataIntakeAgent implementation
- OPC-UA/Modbus client setup
- Data models and validation

**Week 2:**
- HeatTransferCalculatorAgent (ASME PTC 4.3)
- FoulingAnalysisAgent implementation
- Unit tests (85%+ coverage)

**Week 3:**
- AlertGenerationAgent implementation
- PerformanceTrendingAgent implementation
- Integration tests for pipeline

**Deliverable:** 5 core agents with 85%+ test coverage

### Phase 2: API & Integrations (3 weeks)

**Week 4:**
- FastAPI REST API implementation
- WebSocket support for real-time data
- Authentication (JWT/OAuth2)

**Week 5:**
- Soot blower integration (control commands)
- CMMS integration (work order generation)
- Historian integration (optional)

**Week 6:**
- ReportingVisualizationAgent completion
- PDF/Excel report generation
- Email/Slack/SMS notification setup

**Deliverable:** Full integration stack operational

### Phase 3: Testing & Security (2 weeks)

**Week 7:**
- Integration testing (all components)
- Performance testing (throughput, latency)
- Load testing (50+ economizers)

**Week 8:**
- Security testing (penetration testing)
- Industrial security review
- Documentation completion

**Deliverable:** Test coverage 85%+, security audit passed

### Phase 4: Deployment (2 weeks)

**Week 9:**
- Docker containerization
- Kubernetes manifests
- Terraform IaC for cloud deployment
- Grafana dashboard setup

**Week 10:**
- Production deployment
- SCADA network integration
- Monitoring setup (Prometheus/Grafana)
- Beta customer onboarding

**Deliverable:** Production-ready system

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SCADA connectivity issues | High | High | Redundant protocols, offline buffering, retry logic |
| Sensor data quality | Medium | High | Robust validation, quality flags, interpolation |
| Soot blower interlock complexity | Medium | Medium | Detailed interlock spec, simulation testing |
| Industrial security requirements | Medium | High | Early security review, IEC 62443 gap analysis |
| Baseline drift over time | Low | Medium | Periodic baseline validation, drift detection |
| High data volume | Medium | Medium | TimescaleDB hypertables, continuous aggregates |

---

## Success Criteria

The architecture will be considered successful if:

1. **Zero-Hallucination Compliance**: All heat transfer and fouling calculations use deterministic formulas with SHA-256 provenance
2. **ASME PTC 4.3 Compliance**: Calculations follow recognized standards
3. **Real-Time Performance**: < 500 ms from sensor reading to alert generation
4. **Predictive Accuracy**: Fouling predictions within 20% of actual cleaning requirement
5. **Test Coverage**: 85%+ unit test coverage, 95%+ pass rate
6. **Security Score**: Grade A (92+/100)
7. **SCADA Integration**: Successful connection to OPC-UA and Modbus endpoints
8. **Soot Blower Integration**: Automated trigger with interlock validation
9. **Timeline Met**: 10-week development timeline achieved

---

## Business Value

### Cost Savings Model

For a typical industrial boiler with economizer:
- Heat recovery capacity: 20 MMBtu/hr
- Operating hours: 8,000/year
- Fuel cost: $4.00/MMBtu
- Baseline annual fuel cost: $640,000

| Optimization | Savings % | Annual Savings |
|--------------|-----------|----------------|
| Reduced Fouling (optimized cleaning) | 3-5% | $19,200 - $32,000 |
| Predictive Maintenance (avoid failures) | 2-3% | $12,800 - $19,200 |
| Improved Cleaning Timing (ROI optimization) | 1-2% | $6,400 - $12,800 |
| **Total** | **6-10%** | **$38,400 - $64,000** |

**Typical ROI:** 6-12 months payback period

### Operational Benefits

- Reduced unplanned downtime
- Optimized cleaning schedules
- Improved equipment life
- Better compliance documentation
- Real-time visibility into performance

---

## Appendix A: ASME PTC 4.3 Reference

### Log Mean Temperature Difference

The LMTD method is the industry standard for heat exchanger analysis:

```
LMTD = (dT1 - dT2) / ln(dT1/dT2)

For counterflow economizer:
  dT1 = T_hot_in - T_cold_out  (hot end)
  dT2 = T_hot_out - T_cold_in  (cold end)
```

**Special Cases:**
- If dT1 = dT2, LMTD = dT1 (avoid ln(1) = 0)
- If dT1/dT2 < 0, invalid configuration (temperatures crossed)

### Fouling Factor Standards

TEMA fouling factors (typical values):

| Service | Rf (hr-ft2-F/BTU) |
|---------|-------------------|
| Clean water | 0.0005 |
| River water | 0.001 |
| Flue gas (clean fuel) | 0.001 |
| Flue gas (coal) | 0.005 |
| Boiler feedwater | 0.0005 |

---

## Appendix B: Prometheus Metrics

```
# Sensor metrics
econopulse_feedwater_temp_in_f{economizer_id}
econopulse_feedwater_temp_out_f{economizer_id}
econopulse_flue_gas_temp_in_f{economizer_id}
econopulse_flue_gas_temp_out_f{economizer_id}
econopulse_differential_pressure_inh2o{economizer_id}

# Heat transfer metrics
econopulse_lmtd_f{economizer_id}
econopulse_heat_duty_mmbtu_hr{economizer_id}
econopulse_u_value_actual{economizer_id}
econopulse_effectiveness{economizer_id}

# Fouling metrics
econopulse_fouling_factor_rf{economizer_id}
econopulse_fouling_rate_per_day{economizer_id}
econopulse_time_to_threshold_hours{economizer_id}
econopulse_efficiency_loss_percent{economizer_id}
econopulse_fuel_penalty_usd_hr{economizer_id}

# Alert metrics
econopulse_alerts_active_total{economizer_id,priority}
econopulse_alerts_generated_total{economizer_id,type}

# Soot blower metrics
econopulse_soot_blow_count_total{economizer_id}
econopulse_soot_blow_effectiveness_percent{economizer_id}

# System metrics
econopulse_sensor_data_points_total
econopulse_calculation_duration_seconds
econopulse_scada_connection_status{protocol}
```

---

## Appendix C: Alert Escalation Configuration

```yaml
alert_escalation:
  critical:
    tier_1:
      delay_minutes: 0
      channels: [dashboard, sms, pagerduty]
      auto_action: trigger_soot_blow
    tier_2:
      delay_minutes: 5
      channels: [phone_supervisor]
    tier_3:
      delay_minutes: 15
      channels: [phone_plant_manager]

  high:
    tier_1:
      delay_minutes: 0
      channels: [dashboard, email, slack]
    tier_2:
      delay_minutes: 30
      channels: [phone_oncall]

  medium:
    tier_1:
      delay_minutes: 0
      channels: [dashboard, email]

  low:
    tier_1:
      delay_minutes: 0
      channels: [dashboard]
      aggregation: daily_digest
```

---

## Appendix D: Soot Blower Interlock Logic

```python
def can_trigger_soot_blow(economizer_id: str) -> Tuple[bool, str]:
    """
    Check all interlocks before allowing soot blow.
    Returns (can_trigger, reason_if_blocked)
    """
    # Check steam availability
    steam_pressure = get_steam_pressure()
    if steam_pressure < MIN_STEAM_PRESSURE:
        return False, f"Steam pressure {steam_pressure} < {MIN_STEAM_PRESSURE} psig"

    # Check cooldown period
    last_blow = get_last_soot_blow_time(economizer_id)
    if last_blow and (now() - last_blow) < MIN_COOLDOWN:
        return False, f"Cooldown period not elapsed"

    # Check for active maintenance
    if is_in_maintenance(economizer_id):
        return False, "Equipment in maintenance mode"

    # Check for fault conditions
    if has_active_fault(economizer_id):
        return False, "Active fault condition"

    # Check soot blower availability
    for sb in get_soot_blowers(economizer_id):
        if sb.status == "FAULT":
            return False, f"Soot blower {sb.id} in fault"

    return True, "All interlocks satisfied"
```

---

**Document Version**: 1.0.0
**Last Updated**: December 3, 2025
**Maintained By**: GreenLang Architecture Team (GL-AppArchitect)
