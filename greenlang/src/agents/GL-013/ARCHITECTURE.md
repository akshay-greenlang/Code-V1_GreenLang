# GL-013 PREDICTMAINT - Architecture Documentation

```
================================================================================
              ARCHITECTURE DOCUMENTATION - GL-013 PREDICTMAINT
                    Predictive Maintenance Agent Architecture
================================================================================
```

**Document Version:** 1.0.0
**Last Updated:** 2024-12-01
**Author:** GreenLang Architecture Team
**Status:** Production

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Context](#system-context)
3. [Component Architecture](#component-architecture)
4. [Calculator Layer](#calculator-layer)
5. [Integration Layer](#integration-layer)
6. [Data Flow Architecture](#data-flow-architecture)
7. [Calculation Methodologies](#calculation-methodologies)
8. [Security Architecture](#security-architecture)
9. [Deployment Architecture](#deployment-architecture)
10. [Monitoring Architecture](#monitoring-architecture)
11. [Data Architecture](#data-architecture)
12. [Performance Architecture](#performance-architecture)
13. [Disaster Recovery](#disaster-recovery)
14. [Architecture Decision Records](#architecture-decision-records)

---

## Architecture Overview

### Design Principles

GL-013 PREDICTMAINT is designed around the following architectural principles:

| Principle | Description |
|-----------|-------------|
| **Zero-Hallucination** | All numeric outputs from deterministic calculators, never AI-generated |
| **Deterministic Processing** | Same inputs always produce identical outputs |
| **Separation of Concerns** | Clear boundaries between AI classification and numeric calculation |
| **High Availability** | 99.95% uptime with redundant components |
| **Horizontal Scalability** | Scale-out architecture for processing capacity |
| **Standards Compliance** | ISO 10816, ISO 13373, ISO 17359, ISO 55000, IEC 61511 |
| **Auditability** | Complete provenance tracking with SHA-256 hashing |
| **Loose Coupling** | Components interact via well-defined interfaces |

### Architecture Style

GL-013 employs a **layered microservices architecture** with the following tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY LAYER                              │
│                    (REST, WebSocket, GraphQL, gRPC)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                           ORCHESTRATION LAYER                               │
│              (Request Routing, Pipeline Management, Caching)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                            CALCULATOR LAYER                                 │
│     (RUL, Failure Probability, Vibration, Thermal, Scheduling, etc.)       │
├─────────────────────────────────────────────────────────────────────────────┤
│                           INTEGRATION LAYER                                 │
│              (CMMS, CMS, IoT, Agent Coordination, Data Transform)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                              DATA LAYER                                     │
│              (PostgreSQL, TimescaleDB, Redis, Object Storage)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           MONITORING LAYER                                  │
│              (Prometheus, Grafana, Alert Manager, Logging)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Context

### Context Diagram

```
                                    ┌─────────────────┐
                                    │   Operators     │
                                    │   Engineers     │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────┐              ┌─────────────────────────┐              ┌─────────────────┐
│                 │              │                         │              │                 │
│  CMMS Systems   │◄────────────►│    GL-013 PREDICTMAINT  │◄────────────►│  CMS Systems    │
│  (SAP, Maximo)  │              │                         │              │  (SKF, Emerson) │
│                 │              │  Predictive Maintenance │              │                 │
└─────────────────┘              │         Agent           │              └─────────────────┘
                                 │                         │
┌─────────────────┐              │   ┌─────────────────┐   │              ┌─────────────────┐
│                 │              │   │   Calculators   │   │              │                 │
│   IoT Sensors   │─────────────►│   │   Integrators   │   │◄────────────►│  Other Agents   │
│   (MQTT, OPC)   │              │   │   Orchestrator  │   │              │  (GL-001, etc)  │
│                 │              │   └─────────────────┘   │              │                 │
└─────────────────┘              │                         │              └─────────────────┘
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │      Monitoring         │
                                 │   (Prometheus/Grafana)  │
                                 └─────────────────────────┘
```

### External Systems

| System | Protocol | Direction | Purpose |
|--------|----------|-----------|---------|
| SAP PM | REST API | Bidirectional | Work order management, asset data |
| IBM Maximo | REST API | Bidirectional | Work order management, inventory |
| Oracle EAM | REST API | Bidirectional | Asset management, maintenance |
| SKF Enlight | REST API | Inbound | Vibration monitoring data |
| Emerson AMS | OPC-UA | Inbound | Process data, alerts |
| GE Bently | OPC-UA | Inbound | Machinery protection data |
| Honeywell | OPC-UA | Inbound | Condition monitoring data |
| Process Historian | OPC-UA | Inbound | Historical time-series data |
| SCADA | OPC-UA | Inbound | Real-time process data |
| GL-001 THERMOSYNC | Message Bus | Bidirectional | Thermal data coordination |
| GL-014 | Message Bus | Bidirectional | Energy optimization data |

---

## Component Architecture

### High-Level Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GL-013 PREDICTMAINT                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         API GATEWAY                                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │  REST   │  │ WebSocket│  │ GraphQL │  │  gRPC   │  │ Health  │    │   │
│  │  │ /api/v1 │  │  /ws    │  │ /graphql│  │ :50051  │  │ /health │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         ORCHESTRATOR                                  │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │   │
│  │  │ Request Router │  │ Pipeline Mgr   │  │ Cache Manager  │         │   │
│  │  │                │  │                │  │                │         │   │
│  │  │ - Validation   │  │ - Async Proc   │  │ - Redis Cache  │         │   │
│  │  │ - Routing      │  │ - Checkpoints  │  │ - TTL: 300s    │         │   │
│  │  │ - Rate Limit   │  │ - Retries      │  │ - Invalidation │         │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CALCULATOR LAYER                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │   RUL    │ │ Failure  │ │Vibration │ │ Thermal  │ │ Anomaly  │   │   │
│  │  │Calculator│ │Probability│ │ Analyzer │ │Degradation│ │ Detector │   │   │
│  │  │          │ │Calculator│ │          │ │Calculator│ │          │   │   │
│  │  │ Weibull  │ │ Survival │ │ ISO10816 │ │Arrhenius │ │ Z-Score  │   │   │
│  │  │ Exponent.│ │ Analysis │ │ FFT      │ │ 10-Degree│ │ IQR      │   │   │
│  │  │ LogNorm  │ │ Hazard   │ │ Bearing  │ │ Rule     │ │Mahalanob.│   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │Maintenanc│ │  Spare   │ │  Health  │ │Provenance│               │   │
│  │  │Scheduler │ │  Parts   │ │  Index   │ │ Tracker  │               │   │
│  │  │          │ │Calculator│ │Calculator│ │          │               │   │
│  │  │ LP Optim │ │ EOQ      │ │ Weighted │ │ SHA-256  │               │   │
│  │  │ Cost-Ben │ │ Safety   │ │ Multi-Par│ │ Audit    │               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        INTEGRATION LAYER                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │   CMMS   │ │   CMS    │ │   IoT    │ │  Agent   │ │   Data   │   │   │
│  │  │Connector │ │Connector │ │Connector │ │Coordinat.│ │Transform │   │   │
│  │  │          │ │          │ │          │ │          │ │          │   │   │
│  │  │ SAP PM   │ │ SKF      │ │ MQTT     │ │ GL-001   │ │ Units    │   │   │
│  │  │ Maximo   │ │ Emerson  │ │ OPC-UA   │ │ GL-014   │ │ Schema   │   │   │
│  │  │ Oracle   │ │ Bently   │ │ REST     │ │          │ │ Valid.   │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           DATA LAYER                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │PostgreSQL│ │TimescaleDB│ │  Redis   │ │  S3/Blob │               │   │
│  │  │          │ │          │ │          │ │  Storage │               │   │
│  │  │ Metadata │ │ Time     │ │ Cache    │ │ Archive  │               │   │
│  │  │ Config   │ │ Series   │ │ Sessions │ │ Reports  │               │   │
│  │  │ Audit    │ │ Sensor   │ │ Results  │ │ Models   │               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

#### API Gateway

| Component | Purpose | Technology |
|-----------|---------|------------|
| REST API | Primary API for synchronous requests | FastAPI |
| WebSocket | Real-time streaming updates | WebSocket |
| GraphQL | Flexible query interface | Strawberry GraphQL |
| gRPC | High-performance inter-service communication | grpcio |
| Health | Kubernetes health/readiness probes | FastAPI |

#### Orchestrator

| Component | Purpose | Technology |
|-----------|---------|------------|
| Request Router | Route requests to appropriate calculators | Custom Router |
| Pipeline Manager | Manage async processing pipelines | Celery/asyncio |
| Cache Manager | Manage Redis caching layer | Redis |

#### Calculator Layer

| Calculator | Algorithm | Standard |
|------------|-----------|----------|
| RUL Calculator | Weibull, Exponential, Log-Normal | IEC 60300-3-1 |
| Failure Probability | Survival Analysis, Hazard Rate | FMEA |
| Vibration Analyzer | ISO 10816, FFT, Bearing Frequencies | ISO 10816 |
| Thermal Degradation | Arrhenius, 10-Degree Rule | ISO 13373 |
| Anomaly Detector | Z-Score, IQR, Mahalanobis | ISO 17359 |
| Maintenance Scheduler | Linear Programming, Cost-Benefit | ISO 55000 |
| Spare Parts | EOQ, Safety Stock | Inventory Theory |
| Health Index | Weighted Multi-Parameter | Custom |
| Provenance Tracker | SHA-256 Hashing | Audit |

---

## Calculator Layer

### Calculator Architecture

Each calculator follows a standardized architecture pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CALCULATOR                               │
├─────────────────────────────────────────────────────────────────┤
│  Input Validation     │  Validate inputs against schema          │
│  Unit Conversion      │  Convert to standard units               │
│  Calculation Engine   │  Execute deterministic algorithm         │
│  Result Formatting    │  Format output with confidence           │
│  Provenance Recording │  Record audit trail with SHA-256         │
└─────────────────────────────────────────────────────────────────┘
```

### RUL Calculator Architecture

```python
class RULCalculator:
    """
    Remaining Useful Life Calculator

    Supports Weibull, Exponential, and Log-Normal reliability models.
    All calculations are deterministic with full provenance tracking.
    """

    def __init__(self, precision: int = 6, store_provenance_records: bool = True):
        self.precision = precision
        self.store_provenance = store_provenance_records
        self.provenance_tracker = ProvenanceTracker()

    def calculate_weibull_rul(
        self,
        equipment_type: str,
        operating_hours: Decimal,
        current_health_score: Optional[Decimal] = None,
        confidence_level: str = "95%"
    ) -> RULResult:
        """
        Calculate RUL using Weibull reliability model.

        Formula:
            R(t) = exp(-(t/eta)^beta)
            RUL = eta * (-ln(R_target))^(1/beta) - t_current

        Parameters:
            equipment_type: Equipment type for Weibull parameters
            operating_hours: Current operating hours
            current_health_score: Optional health score adjustment
            confidence_level: Confidence level for interval

        Returns:
            RULResult with hours, days, confidence interval, provenance
        """
        # Implementation details...
```

### Weibull Parameters by Equipment Type

| Equipment Type | Beta (Shape) | Eta (Scale) | Reference |
|----------------|--------------|-------------|-----------|
| motor_ac_induction_large | 2.5 | 131,400 | IEEE 493 |
| motor_ac_induction_small | 2.2 | 87,600 | IEEE 493 |
| pump_centrifugal | 2.0 | 75,000 | OREDA |
| compressor_reciprocating | 1.8 | 55,000 | OREDA |
| fan_centrifugal | 2.3 | 95,000 | Manufacturer |
| gearbox_industrial | 2.5 | 100,000 | Manufacturer |
| bearing_rolling_element | 3.0 | 50,000 | ISO 281 |
| turbine_steam | 2.8 | 175,000 | OREDA |
| transformer_power | 3.5 | 300,000 | IEEE |

### Vibration Analyzer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIBRATION ANALYZER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   Time Domain    │    │  Frequency Domain │                   │
│  │    Analysis      │    │     Analysis      │                   │
│  │                  │    │                   │                   │
│  │ - RMS Velocity   │    │ - FFT Spectrum    │                   │
│  │ - Peak Value     │    │ - Power Spectral  │                   │
│  │ - Crest Factor   │    │ - Envelope        │                   │
│  │ - Kurtosis       │    │ - Waterfall       │                   │
│  └────────┬─────────┘    └─────────┬─────────┘                   │
│           │                        │                              │
│           └───────────┬────────────┘                              │
│                       ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              ISO 10816 SEVERITY ASSESSMENT                │    │
│  │                                                           │    │
│  │  Class I   │  Class II  │  Class III │  Class IV          │    │
│  │  <15 kW    │  15-75 kW  │  Rigid     │  Flexible          │    │
│  │            │            │  Foundation│  Foundation         │    │
│  │  Zone A    │  Zone A    │  Zone A    │  Zone A             │    │
│  │  0-0.71    │  0-1.12    │  0-1.8     │  0-2.8   mm/s       │    │
│  │  Zone B    │  Zone B    │  Zone B    │  Zone B             │    │
│  │  0.71-1.8  │  1.12-2.8  │  1.8-4.5   │  2.8-7.1 mm/s       │    │
│  │  Zone C    │  Zone C    │  Zone C    │  Zone C             │    │
│  │  1.8-4.5   │  2.8-7.1   │  4.5-11.2  │  7.1-18  mm/s       │    │
│  │  Zone D    │  Zone D    │  Zone D    │  Zone D             │    │
│  │  >4.5      │  >7.1      │  >11.2     │  >18.0   mm/s       │    │
│  └──────────────────────────────────────────────────────────┘    │
│                       │                                           │
│                       ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              BEARING FAULT DETECTION                      │    │
│  │                                                           │    │
│  │  BPFO = (n/2) * (1 - Bd/Pd * cos(phi)) * RPM/60          │    │
│  │  BPFI = (n/2) * (1 + Bd/Pd * cos(phi)) * RPM/60          │    │
│  │  BSF  = (Pd/2Bd) * (1 - (Bd/Pd * cos(phi))^2) * RPM/60   │    │
│  │  FTF  = (1/2) * (1 - Bd/Pd * cos(phi)) * RPM/60          │    │
│  │                                                           │    │
│  │  Where: n = number of rolling elements                    │    │
│  │         Bd = ball diameter                                │    │
│  │         Pd = pitch diameter                               │    │
│  │         phi = contact angle                               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Thermal Degradation Calculator

The thermal degradation calculator uses the Arrhenius equation to predict equipment life based on operating temperature:

```
┌─────────────────────────────────────────────────────────────────┐
│               THERMAL DEGRADATION CALCULATOR                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ARRHENIUS EQUATION:                                            │
│                                                                  │
│    L = L0 * exp[Ea/k * (1/T - 1/T0)]                            │
│                                                                  │
│    Where:                                                        │
│      L  = Life at temperature T                                  │
│      L0 = Baseline life at reference temperature T0              │
│      Ea = Activation energy (0.8-1.1 eV typical)                │
│      k  = Boltzmann constant (8.617e-5 eV/K)                    │
│      T  = Operating temperature (Kelvin)                         │
│      T0 = Reference temperature (Kelvin)                         │
│                                                                  │
│  10-DEGREE RULE (Simplified):                                   │
│                                                                  │
│    Life_reduction = 2^((T - T_ref) / 10)                        │
│                                                                  │
│    For every 10C increase, life is halved                       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INSULATION CLASS LIMITS (IEC 60085):                           │
│                                                                  │
│    Class A: 105C maximum                                         │
│    Class E: 120C maximum                                         │
│    Class B: 130C maximum                                         │
│    Class F: 155C maximum                                         │
│    Class H: 180C maximum                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Anomaly Detection Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANOMALY DETECTOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                STATISTICAL METHODS                       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  Z-SCORE METHOD:                                         │    │
│  │    z = (x - mu) / sigma                                  │    │
│  │    Anomaly if |z| > 3                                    │    │
│  │                                                          │    │
│  │  IQR METHOD:                                             │    │
│  │    IQR = Q3 - Q1                                         │    │
│  │    Lower = Q1 - 1.5 * IQR                                │    │
│  │    Upper = Q3 + 1.5 * IQR                                │    │
│  │    Anomaly if x < Lower or x > Upper                     │    │
│  │                                                          │    │
│  │  MAHALANOBIS DISTANCE:                                   │    │
│  │    D = sqrt((x - mu)' * S^-1 * (x - mu))                 │    │
│  │    Anomaly if D > threshold (chi-square)                 │    │
│  │                                                          │    │
│  │  CONTROL CHARTS (SHEWHART):                              │    │
│  │    UCL = mu + 3 * sigma                                  │    │
│  │    LCL = mu - 3 * sigma                                  │    │
│  │    CL  = mu                                              │    │
│  │    Anomaly if x > UCL or x < LCL                         │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                SEVERITY CLASSIFICATION                   │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  Severity    │  Z-Score   │  Response                    │    │
│  │  ───────────────────────────────────────────────────────│    │
│  │  INFO        │  2.0-2.5   │  Log only                    │    │
│  │  WARNING     │  2.5-3.0   │  Notify operator             │    │
│  │  ALERT       │  3.0-4.0   │  Increase monitoring         │    │
│  │  ALARM       │  4.0-5.0   │  Maintenance required        │    │
│  │  DANGER      │  > 5.0     │  Immediate shutdown          │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Layer

### CMMS Connector Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CMMS CONNECTOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CONNECTION MANAGER                          │    │
│  │                                                          │    │
│  │  - Connection pooling (max 10 connections)               │    │
│  │  - Automatic retry with exponential backoff              │    │
│  │  - Circuit breaker (5 failures = open)                   │    │
│  │  - Health monitoring                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│  ┌───────────┬───────────┴───────────┬───────────┐              │
│  │           │                       │           │              │
│  ▼           ▼                       ▼           ▼              │
│  ┌───────┐   ┌───────┐               ┌───────┐   ┌───────┐      │
│  │ SAP   │   │ IBM   │               │Oracle │   │Custom │      │
│  │ PM    │   │Maximo │               │ EAM   │   │ CMMS  │      │
│  │Adapter│   │Adapter│               │Adapter│   │Adapter│      │
│  └───┬───┘   └───┬───┘               └───┬───┘   └───┬───┘      │
│      │           │                       │           │          │
│      └───────────┴───────────┬───────────┴───────────┘          │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AUTHENTICATION HANDLER                      │    │
│  │                                                          │    │
│  │  OAuth 2.0   │  API Key   │  Basic Auth  │  Certificate │    │
│  │  ─────────────────────────────────────────────────────  │    │
│  │  Token cache │  Header    │  Base64      │  mTLS        │    │
│  │  Refresh     │  injection │  encoding    │  validation  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                DATA OPERATIONS                           │    │
│  │                                                          │    │
│  │  Work Orders     │  Equipment      │  Inventory          │    │
│  │  ────────────────────────────────────────────────────── │    │
│  │  Create          │  Sync           │  Query stock        │    │
│  │  Update          │  Get details    │  Reserve parts      │    │
│  │  Complete        │  Update status  │  Update quantities  │    │
│  │  List            │  Get hierarchy  │  Create requisition │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Supported CMMS Systems

| System | Protocol | Authentication | Features |
|--------|----------|----------------|----------|
| SAP PM | REST API | OAuth 2.0 | Work orders, notifications, equipment |
| IBM Maximo | REST API | API Key | Work orders, assets, inventory |
| Oracle EAM | REST API | OAuth 2.0 | Work requests, assets, resources |
| Custom | REST/SOAP | Configurable | Flexible mapping |

### Agent Coordination Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT COORDINATOR                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 MESSAGE BUS                              │    │
│  │                 (RabbitMQ/Kafka)                         │    │
│  │                                                          │    │
│  │  Exchange: greenlang.agents                              │    │
│  │  Routing:  agent.{agent_id}.{operation}                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              COORDINATION PATTERNS                       │    │
│  │                                                          │    │
│  │  Request-Response                                        │    │
│  │  ──────────────────                                      │    │
│  │  GL-013 ──request──> GL-001                              │    │
│  │  GL-013 <──response── GL-001                             │    │
│  │                                                          │    │
│  │  Publish-Subscribe                                       │    │
│  │  ────────────────────                                    │    │
│  │  GL-013 ──publish──> predictions.failure                 │    │
│  │                      ├──> GL-001 (subscriber)            │    │
│  │                      ├──> GL-014 (subscriber)            │    │
│  │                      └──> Alert Manager (subscriber)     │    │
│  │                                                          │    │
│  │  Scatter-Gather                                          │    │
│  │  ────────────────                                        │    │
│  │  GL-013 ──scatter──> [GL-001, GL-002, GL-014]            │    │
│  │  GL-013 <──gather─── [responses]                         │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              COORDINATED AGENTS                          │    │
│  │                                                          │    │
│  │  GL-001 THERMOSYNC  │  Thermal data, energy management   │    │
│  │  GL-002             │  Boiler control, steam systems     │    │
│  │  GL-003             │  Heat recovery optimization        │    │
│  │  GL-014             │  Energy optimization               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Request Processing Flow

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Client  │   │   API    │   │  Orchest │   │Calculator│   │   Data   │
│          │   │ Gateway  │   │  rator   │   │  Layer   │   │  Layer   │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │              │
     │  POST /api/v1/analyze       │              │              │
     │──────────────>│              │              │              │
     │              │              │              │              │
     │              │  Authenticate & Validate    │              │
     │              │──────────────>│              │              │
     │              │              │              │              │
     │              │              │  Check Cache │              │
     │              │              │──────────────────────────────>│
     │              │              │              │              │
     │              │              │  Cache Miss  │              │
     │              │              │<──────────────────────────────│
     │              │              │              │              │
     │              │              │  Route to RUL Calculator    │
     │              │              │──────────────>│              │
     │              │              │              │              │
     │              │              │              │  Get Weibull │
     │              │              │              │  Parameters  │
     │              │              │              │──────────────>│
     │              │              │              │              │
     │              │              │              │<──────────────│
     │              │              │              │              │
     │              │              │              │  Calculate   │
     │              │              │              │  RUL         │
     │              │              │              │──────┐       │
     │              │              │              │      │       │
     │              │              │              │<─────┘       │
     │              │              │              │              │
     │              │              │              │  Record      │
     │              │              │              │  Provenance  │
     │              │              │              │──────────────>│
     │              │              │              │              │
     │              │              │  Return Result              │
     │              │              │<──────────────│              │
     │              │              │              │              │
     │              │              │  Cache Result │              │
     │              │              │──────────────────────────────>│
     │              │              │              │              │
     │              │  Return Response            │              │
     │              │<──────────────│              │              │
     │              │              │              │              │
     │  200 OK + Result            │              │              │
     │<──────────────│              │              │              │
     │              │              │              │              │
```

### Real-Time Streaming Flow

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   IoT    │   │   MQTT   │   │  Stream  │   │Calculator│   │WebSocket │
│ Sensors  │   │  Broker  │   │ Processor│   │  Layer   │   │  Server  │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │              │
     │  Publish sensor data        │              │              │
     │──────────────>│              │              │              │
     │              │              │              │              │
     │              │  Subscribe: sensors/#       │              │
     │              │<──────────────│              │              │
     │              │              │              │              │
     │              │  Forward message            │              │
     │              │──────────────>│              │              │
     │              │              │              │              │
     │              │              │  Validate & │              │
     │              │              │  Transform  │              │
     │              │              │──────┐      │              │
     │              │              │      │      │              │
     │              │              │<─────┘      │              │
     │              │              │              │              │
     │              │              │  Calculate   │              │
     │              │              │  Anomaly     │              │
     │              │              │──────────────>│              │
     │              │              │              │              │
     │              │              │  Anomaly Detected!          │
     │              │              │<──────────────│              │
     │              │              │              │              │
     │              │              │  Broadcast Alert            │
     │              │              │──────────────────────────────>│
     │              │              │              │              │
     │              │              │              │              │ Push to
     │              │              │              │              │ Connected
     │              │              │              │              │ Clients
```

---

## Calculation Methodologies

### RUL Calculation (Weibull Model)

The Weibull distribution is the primary model for equipment RUL prediction:

**Reliability Function:**
```
R(t) = exp(-(t/eta)^beta)
```

**RUL Calculation:**
```
RUL = eta * (-ln(R_target))^(1/beta) - t_current
```

**Confidence Interval:**
```
CI_lower = RUL * exp(-z * sqrt(Var(RUL)) / RUL)
CI_upper = RUL * exp(+z * sqrt(Var(RUL)) / RUL)

Where z = 1.96 for 95% confidence
```

### Failure Probability (Survival Analysis)

**Cumulative Distribution Function:**
```
F(t) = 1 - R(t) = 1 - exp(-(t/eta)^beta)
```

**Hazard Rate (Instantaneous Failure Rate):**
```
h(t) = (beta/eta) * (t/eta)^(beta-1)
```

**Mean Time Between Failures:**
```
MTBF = eta * Gamma(1 + 1/beta)
```

### Vibration Analysis (ISO 10816)

**RMS Velocity Calculation:**
```
v_rms = sqrt((1/T) * integral(v(t)^2 dt))
```

**Zone Classification (Class II Machines):**
```
Zone A: v_rms <= 1.12 mm/s  (Newly commissioned)
Zone B: 1.12 < v_rms <= 2.8 mm/s  (Acceptable)
Zone C: 2.8 < v_rms <= 7.1 mm/s  (Limited operation)
Zone D: v_rms > 7.1 mm/s  (Damage imminent)
```

### Thermal Degradation (Arrhenius)

**Life Calculation:**
```
L = L0 * exp[(Ea/k) * (1/T - 1/T0)]
```

**Acceleration Factor:**
```
AF = exp[(Ea/k) * (1/T_ref - 1/T_op)]
```

**10-Degree Rule (Simplified):**
```
Life_factor = 2^((T_ref - T_op) / 10)
```

### Anomaly Detection Thresholds

| Method | Formula | Default Threshold |
|--------|---------|-------------------|
| Z-Score | z = (x - mu) / sigma | abs(z) > 3 |
| IQR | Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR | Outside bounds |
| Mahalanobis | D = sqrt((x-mu)' * S^-1 * (x-mu)) | D > chi2_0.95 |
| Control Charts | UCL/LCL = mu +/- 3*sigma | Outside limits |

---

## Security Architecture

### Authentication Flow

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Client  │   │   API    │   │ Keycloak │   │   GL-013 │
│          │   │ Gateway  │   │   IdP    │   │   Agent  │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │
     │  POST /auth/token           │              │
     │  (client_credentials)       │              │
     │──────────────────────────────>│              │
     │              │              │              │
     │              │              │  Validate    │
     │              │              │  credentials │
     │              │              │──────┐       │
     │              │              │      │       │
     │              │              │<─────┘       │
     │              │              │              │
     │  200 OK + JWT token         │              │
     │<──────────────────────────────│              │
     │              │              │              │
     │  POST /api/v1/analyze       │              │
     │  Authorization: Bearer JWT  │              │
     │──────────────>│              │              │
     │              │              │              │
     │              │  Verify JWT  │              │
     │              │──────────────>│              │
     │              │              │              │
     │              │  Token valid │              │
     │              │<──────────────│              │
     │              │              │              │
     │              │  Forward request            │
     │              │──────────────────────────────>│
     │              │              │              │
```

### Authorization (RBAC)

| Role | Permissions | Scope |
|------|-------------|-------|
| admin | read, write, delete, configure | All resources |
| maintenance_engineer | read, write, acknowledge_alerts | Equipment, Work Orders |
| operator | read, acknowledge_alerts | Equipment status |
| viewer | read | Reports only |

### Encryption Standards

| Layer | Algorithm | Key Size | Rotation |
|-------|-----------|----------|----------|
| At Rest | AES-256-GCM | 256 bits | 90 days |
| In Transit | TLS 1.3 | 256 bits | 365 days |
| Token Signing | RS256 | 2048 bits | 90 days |
| Data Hashing | SHA-256 | 256 bits | N/A |

---

## Deployment Architecture

### Kubernetes Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KUBERNETES CLUSTER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        NAMESPACE: greenlang                         │     │
│  │                                                                     │     │
│  │  ┌──────────────────────────────────────────────────────────────┐  │     │
│  │  │              DEPLOYMENT: gl-013-predictmaint                  │  │     │
│  │  │              Replicas: 3 | Strategy: RollingUpdate            │  │     │
│  │  │                                                               │  │     │
│  │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                   │  │     │
│  │  │  │  Pod 1  │    │  Pod 2  │    │  Pod 3  │                   │  │     │
│  │  │  │ Node A  │    │ Node B  │    │ Node C  │                   │  │     │
│  │  │  │         │    │         │    │         │                   │  │     │
│  │  │  │ :8000   │    │ :8000   │    │ :8000   │                   │  │     │
│  │  │  │ :9090   │    │ :9090   │    │ :9090   │                   │  │     │
│  │  │  └─────────┘    └─────────┘    └─────────┘                   │  │     │
│  │  └──────────────────────────────────────────────────────────────┘  │     │
│  │                              │                                      │     │
│  │  ┌──────────────────────────────────────────────────────────────┐  │     │
│  │  │              SERVICE: gl-013-predictmaint-svc                 │  │     │
│  │  │              Type: ClusterIP | Port: 8000, 9090               │  │     │
│  │  └──────────────────────────────────────────────────────────────┘  │     │
│  │                              │                                      │     │
│  │  ┌──────────────────────────────────────────────────────────────┐  │     │
│  │  │              INGRESS: gl-013-predictmaint-ingress             │  │     │
│  │  │              Host: api.greenlang.io                           │  │     │
│  │  │              TLS: greenlang-tls-cert                          │  │     │
│  │  └──────────────────────────────────────────────────────────────┘  │     │
│  │                                                                     │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │     │
│  │  │  ConfigMap   │  │   Secret     │  │     PVC      │             │     │
│  │  │ gl-013-cfg   │  │ gl-013-sec   │  │ gl-013-pvc   │             │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │     │
│  │                                                                     │     │
│  │  ┌──────────────────────────────────────────────────────────────┐  │     │
│  │  │                    HPA: gl-013-hpa                            │  │     │
│  │  │         Min: 2 | Max: 10 | Target CPU: 70%                    │  │     │
│  │  └──────────────────────────────────────────────────────────────┘  │     │
│  │                                                                     │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Resource Configuration

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 1000m |
| Memory | 512Mi | 1Gi |
| Ephemeral Storage | 256Mi | 512Mi |

### Health Probes

| Probe | Path | Initial Delay | Period | Timeout |
|-------|------|---------------|--------|---------|
| Liveness | /health | 60s | 15s | 10s |
| Readiness | /health/ready | 30s | 10s | 5s |
| Startup | /health | 10s | 10s | 5s |

---

## Monitoring Architecture

### Prometheus Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │    GL-013        │    │   Prometheus     │    │    Grafana       │       │
│  │    Agent         │────│                  │────│                  │       │
│  │    :9090/metrics │    │   :9090          │    │   :3000          │       │
│  └──────────────────┘    └────────┬─────────┘    └──────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌──────────────────┐                               │
│                          │  Alert Manager   │                               │
│                          │                  │                               │
│                          │  - PagerDuty     │                               │
│                          │  - Slack         │                               │
│                          │  - Email         │                               │
│                          └──────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `predictmaint_predictions_total` | Counter | type, equipment | Total predictions |
| `predictmaint_prediction_latency_seconds` | Histogram | type | Latency distribution |
| `predictmaint_equipment_health_index` | Gauge | equipment_id | Health score 0-100 |
| `predictmaint_equipment_rul_hours` | Gauge | equipment_id | RUL in hours |
| `predictmaint_vibration_velocity_mm_s` | Gauge | equipment_id, point | Vibration RMS |
| `predictmaint_temperature_celsius` | Gauge | equipment_id, location | Temperature |
| `predictmaint_anomaly_score` | Gauge | equipment_id | Anomaly score 0-1 |
| `predictmaint_alerts_total` | Counter | severity | Alert count |
| `predictmaint_cache_hit_ratio` | Gauge | - | Cache hit percentage |
| `predictmaint_integration_status` | Gauge | system | Connector status |

### Alert Rules

| Alert | Condition | Severity | Actions |
|-------|-----------|----------|---------|
| HighFailureProbability | failure_prob > 0.8 | critical | PagerDuty, Email |
| LowHealthScore | health_score < 40 | critical | PagerDuty, Slack |
| RULThreshold | rul_days < 30 | warning | Email, Slack |
| AnomalyDetected | anomaly_severity == ALARM | high | Slack |
| PredictionLatency | p99 > 2000ms | warning | Slack |
| IntegrationDown | status == 0 | warning | Slack |

---

## Data Architecture

### Database Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE SCHEMA                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POSTGRESQL (Metadata)                                                       │
│  ─────────────────────                                                       │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   equipment     │    │  work_orders    │    │  predictions    │          │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤          │
│  │ id (PK)         │    │ id (PK)         │    │ id (PK)         │          │
│  │ equipment_id    │    │ equipment_id(FK)│    │ equipment_id(FK)│          │
│  │ equipment_type  │    │ work_order_type │    │ prediction_type │          │
│  │ manufacturer    │    │ description     │    │ value           │          │
│  │ model           │    │ status          │    │ confidence      │          │
│  │ installation_dt │    │ priority        │    │ timestamp       │          │
│  │ criticality     │    │ scheduled_date  │    │ provenance_hash │          │
│  │ created_at      │    │ created_at      │    │ created_at      │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                              │
│  TIMESCALEDB (Time Series)                                                   │
│  ─────────────────────────                                                   │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  sensor_data    │    │  vibration_data │    │ temperature_data│          │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤          │
│  │ time (PK)       │    │ time (PK)       │    │ time (PK)       │          │
│  │ equipment_id    │    │ equipment_id    │    │ equipment_id    │          │
│  │ sensor_type     │    │ velocity_rms    │    │ bearing_temp    │          │
│  │ value           │    │ acceleration    │    │ winding_temp    │          │
│  │ quality         │    │ frequency       │    │ ambient_temp    │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                              │
│  REDIS (Cache)                                                               │
│  ─────────────                                                               │
│                                                                              │
│  Key Pattern              │  TTL   │  Purpose                                │
│  ─────────────────────────────────────────────────────────────────          │
│  rul:{equipment_id}       │  300s  │  Cached RUL predictions                 │
│  health:{equipment_id}    │  300s  │  Cached health scores                   │
│  config:{key}             │  3600s │  Configuration cache                    │
│  session:{token}          │  1800s │  User session data                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Retention

| Data Type | Retention | Compression | Archive |
|-----------|-----------|-------------|---------|
| Raw Sensor Data | 90 days | Yes | S3 Glacier |
| Processed Data | 365 days | Yes | S3 Standard |
| Predictions | 7 years | Yes | S3 Standard |
| Audit Logs | 7 years | No | S3 Standard |
| Reports | 7 years | Yes | S3 Standard |

---

## Performance Architecture

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Latency (p50) | < 200ms | Prometheus histogram |
| API Latency (p99) | < 2000ms | Prometheus histogram |
| Throughput | 1000 req/s | Load testing |
| Cache Hit Ratio | > 80% | Redis stats |
| Availability | 99.95% | Uptime monitoring |

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      CACHING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  L1: In-Memory Cache (LRU)                                      │
│  ─────────────────────────                                      │
│  - Weibull parameters                                            │
│  - ISO 10816 thresholds                                          │
│  - Bearing frequency data                                        │
│  - TTL: Process lifetime                                         │
│                                                                  │
│  L2: Redis Cache                                                 │
│  ────────────────                                                │
│  - RUL predictions (TTL: 300s)                                   │
│  - Health scores (TTL: 300s)                                     │
│  - Equipment metadata (TTL: 3600s)                               │
│  - Session data (TTL: 1800s)                                     │
│                                                                  │
│  L3: PostgreSQL                                                  │
│  ────────────────                                                │
│  - Historical predictions                                        │
│  - Equipment registry                                            │
│  - Work order history                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Disaster Recovery

### Recovery Objectives

| Objective | Target |
|-----------|--------|
| RTO (Recovery Time Objective) | 4 hours |
| RPO (Recovery Point Objective) | 1 hour |
| Backup Frequency | Hourly |
| Backup Retention | 30 days |

### Backup Strategy

| Component | Method | Frequency | Retention |
|-----------|--------|-----------|-----------|
| PostgreSQL | pg_dump + WAL | Continuous | 30 days |
| TimescaleDB | pg_dump | Daily | 30 days |
| Redis | RDB + AOF | Hourly | 7 days |
| Configuration | Git | On change | Unlimited |
| Secrets | Vault Backup | Daily | 30 days |

### Failover Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISASTER RECOVERY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PRIMARY REGION (us-east-1)        DR REGION (us-west-2)        │
│  ─────────────────────────         ─────────────────────        │
│                                                                  │
│  ┌─────────────────┐               ┌─────────────────┐          │
│  │  GL-013 Pods    │               │  GL-013 Pods    │          │
│  │  (Active)       │               │  (Standby)      │          │
│  └────────┬────────┘               └────────┬────────┘          │
│           │                                 │                    │
│  ┌────────┴────────┐               ┌────────┴────────┐          │
│  │   PostgreSQL    │──────────────►│   PostgreSQL    │          │
│  │   (Primary)     │  Streaming    │   (Replica)     │          │
│  └────────┬────────┘  Replication  └────────┬────────┘          │
│           │                                 │                    │
│  ┌────────┴────────┐               ┌────────┴────────┐          │
│  │     Redis       │──────────────►│     Redis       │          │
│  │   (Primary)     │  Replication  │   (Replica)     │          │
│  └─────────────────┘               └─────────────────┘          │
│                                                                  │
│  Failover Trigger: Manual or automated (health check failures)  │
│  DNS Update: Route53 health-based routing                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Decision Records

### ADR-001: Deterministic Calculations

**Status:** Accepted
**Context:** Need to ensure numeric outputs are reproducible and auditable
**Decision:** All numeric calculations performed by deterministic Python modules using Decimal precision
**Consequences:**
- Guaranteed reproducibility
- Full audit trail with provenance
- Increased code complexity
- Slightly higher computational cost

### ADR-002: Zero-Hallucination Policy

**Status:** Accepted
**Context:** AI models can generate plausible but incorrect numeric values
**Decision:** AI is prohibited from generating any numeric outputs; restricted to classification and NLG tasks
**Consequences:**
- Regulatory compliance for numeric reporting
- Clear separation of concerns
- AI model constrained to non-numeric tasks
- Additional validation required at boundaries

### ADR-003: Weibull as Primary RUL Model

**Status:** Accepted
**Context:** Multiple reliability models exist for RUL prediction
**Decision:** Use Weibull distribution as primary model with fallback to Exponential
**Consequences:**
- Handles both early failures and wear-out
- Well-established in reliability engineering
- Requires equipment-specific parameters
- May not fit all failure patterns

### ADR-004: ISO 10816 for Vibration Analysis

**Status:** Accepted
**Context:** Need standardized vibration severity classification
**Decision:** Implement ISO 10816 as primary vibration standard
**Consequences:**
- Industry-standard compliance
- Consistent severity zones
- Machine class dependency
- Limited to velocity RMS measurements

### ADR-005: Redis for Caching Layer

**Status:** Accepted
**Context:** Need low-latency caching for prediction results
**Decision:** Use Redis with 300-second TTL for prediction cache
**Consequences:**
- Sub-millisecond cache access
- Reduced database load
- Additional infrastructure component
- Cache invalidation complexity

### ADR-006: Kubernetes for Container Orchestration

**Status:** Accepted
**Context:** Need scalable, resilient deployment platform
**Decision:** Deploy on Kubernetes with HPA and PodDisruptionBudget
**Consequences:**
- Horizontal scaling
- Self-healing
- Complex configuration
- Kubernetes expertise required

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| RUL | Remaining Useful Life |
| MTBF | Mean Time Between Failures |
| MTTR | Mean Time To Repair |
| FFT | Fast Fourier Transform |
| EOQ | Economic Order Quantity |
| CMMS | Computerized Maintenance Management System |
| CMS | Condition Monitoring System |
| BPFO | Ball Pass Frequency Outer |
| BPFI | Ball Pass Frequency Inner |
| BSF | Ball Spin Frequency |
| FTF | Fundamental Train Frequency |

### References

1. ISO 10816-1:1995 - Mechanical vibration evaluation
2. ISO 13373-1:2002 - Condition monitoring and diagnostics
3. ISO 17359:2011 - General guidelines for condition monitoring
4. ISO 55000:2014 - Asset management
5. IEC 60300-3-1:2003 - Dependability management
6. IEEE 493-2007 - Recommended practice for industrial reliability
7. OREDA Handbook - Offshore reliability data

---

```
================================================================================
                    GL-013 PREDICTMAINT - ARCHITECTURE v1.0.0
                         GreenLang Inc. - Team Iota
================================================================================
```
