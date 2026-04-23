# GL-017 CONDENSYNC Architecture

## Overview

GL-017 CONDENSYNC is a production-grade condenser optimization agent that implements zero-hallucination principles for industrial condenser performance monitoring, diagnostics, and optimization. The agent calculates condenser efficiency, identifies fouling conditions, and provides recommendations for maintenance and operation optimization based on Heat Exchange Institute (HEI) standards.

## System Architecture

```
+-----------------------------------------------------------------------------+
|                           GL-017 CONDENSYNC                                   |
+-----------------------------------------------------------------------------+
|                                                                               |
|  +-------------+    +-------------+    +-------------+    +-------------+    |
|  |   Process   |    |   Cooling   |    |   Ambient   |    |   External  |    |
|  |    Data     |    |    Water    |    | Conditions  |    |     DCS     |    |
|  +------+------+    +------+------+    +------+------+    +------+------+    |
|         |                  |                  |                  |           |
|         v                  v                  v                  v           |
|  +-------------------------------------------------------------------+      |
|  |                     Integration Layer                              |      |
|  |   +------------+ +------------+ +------------+ +------------+     |      |
|  |   |  OPC-UA    | |   Modbus   | |    MQTT    | |  REST API  |     |      |
|  |   | Connector  | | Connector  | | Connector  | | Connector  |     |      |
|  |   +------------+ +------------+ +------------+ +------------+     |      |
|  +-------------------------------------------------------------------+      |
|                                   |                                          |
|                                   v                                          |
|  +-------------------------------------------------------------------+      |
|  |                      Core Processing Engine                        |      |
|  |                                                                    |      |
|  |   +------------------------------------------------------------+  |      |
|  |   |                   Bounds Validator                          |  |      |
|  |   |   (Pressure: 0.5-15 psia, Temp: 273-373 K, TTD: 0-30 K)   |  |      |
|  |   +------------------------------------------------------------+  |      |
|  |                              |                                     |      |
|  |   +------------+  +------------+  +------------+  +------------+  |      |
|  |   |    HEI     |  |   LMTD     |  |  Fouling   |  |  Cleanliness|  |      |
|  |   | Calculator |  | Calculator |  | Calculator |  |   Factor    |  |      |
|  |   |            |  |            |  |            |  | Calculator  |  |      |
|  |   +-----+------+  +-----+------+  +-----+------+  +-----+------+  |      |
|  |         |               |               |               |         |      |
|  |         +---------------+---------------+---------------+         |      |
|  |                              |                                     |      |
|  |                              v                                     |      |
|  |   +------------------------------------------------------------+  |      |
|  |   |              Condenser Performance Analyzer                 |  |      |
|  |   |         (U-Value, TTD, DCA, Cleanliness Factor)            |  |      |
|  |   |                                                            |  |      |
|  |   |   +------------+  +------------+  +------------+           |  |      |
|  |   |   |  Design    |  |  Current   |  |  Deviation |           |  |      |
|  |   |   |   Data     |  |   Data     |  |  Analysis  |           |  |      |
|  |   |   +------------+  +------------+  +------------+           |  |      |
|  |   +------------------------------------------------------------+  |      |
|  |                              |                                     |      |
|  +------------------------------+-------------------------------------+      |
|                                 |                                            |
|                                 v                                            |
|  +-------------------------------------------------------------------+      |
|  |                    Optimization Engine                             |      |
|  |                                                                    |      |
|  |   +------------+  +------------+  +------------+  +------------+  |      |
|  |   | Cleaning   |  |  Cooling   |  |  Backpres. |  |  Economic  |  |      |
|  |   | Schedule   |  |   Water    |  |    Impact  |  |  Analysis  |  |      |
|  |   | Optimizer  |  | Optimizer  |  | Calculator |  |            |  |      |
|  |   +------------+  +------------+  +------------+  +------------+  |      |
|  +-------------------------------------------------------------------+      |
|                                 |                                            |
|                                 v                                            |
|  +-------------------------------------------------------------------+      |
|  |                    Explainability Engine                           |      |
|  |                                                                    |      |
|  |   +------------+  +------------+  +------------+                  |      |
|  |   |   Factor   |  |  Evidence  |  | Recomm.    |                  |      |
|  |   | Attribution|  |   Chain    |  | Justif.    |                  |      |
|  |   +------------+  +------------+  +------------+                  |      |
|  +-------------------------------------------------------------------+      |
|                                 |                                            |
|                                 v                                            |
|  +-------------------------------------------------------------------+      |
|  |                         Output Layer                               |      |
|  |                                                                    |      |
|  |   +------------+  +------------+  +------------+  +------------+  |      |
|  |   |  REST API  |  | Prometheus |  |  Climate   |  |    DCS     |  |      |
|  |   | /diagnose  |  |  Metrics   |  |  Reporter  |  |  Feedback  |  |      |
|  |   +------------+  +------------+  +------------+  +------------+  |      |
|  +-------------------------------------------------------------------+      |
|                                                                               |
+-------------------------------------------------------------------------------+
```

## Component Details

### 1. Integration Layer

Handles data acquisition from multiple industrial protocols and data sources:

| Connector | Protocol | Purpose |
|-----------|----------|---------|
| `opc_ua_connector.py` | OPC-UA | DCS process data (pressures, temps, flows) |
| `modbus_connector.py` | Modbus TCP | Cooling water instrumentation |
| `mqtt_connector.py` | MQTT | IoT sensors, wireless instruments |
| `historian_connector.py` | REST/OPC-UA | OSIsoft PI, Honeywell PHD |
| `dcs_connector.py` | Vendor-specific | Emerson, Honeywell, ABB integration |

### 2. Calculators

Deterministic calculation engines based on HEI Standards:

| Calculator | Function | Standard |
|------------|----------|----------|
| `hei_calculator.py` | Heat transfer coefficient calculations | HEI 3098 |
| `lmtd_calculator.py` | Log Mean Temperature Difference | HEI 3098 |
| `fouling_calculator.py` | Fouling resistance calculations | HEI 3098 |
| `cleanliness_factor_calculator.py` | CF based on design vs actual U | HEI 3098 |
| `backpressure_calculator.py` | Turbine backpressure impact | HEI 3098 |
| `economic_calculator.py` | Heat rate penalty and cost impact | - |

### 3. Core Analyzer

**Condenser Performance Analyzer** (`core/condenser_analyzer.py`)

Primary analysis engine that computes:

- **Overall Heat Transfer Coefficient (U)**: Actual vs design comparison
- **Terminal Temperature Difference (TTD)**: Saturation temp minus CW outlet
- **Drain Cooler Approach (DCA)**: Hotwell temp minus CW inlet
- **Cleanliness Factor (CF)**: Ratio of actual to design U-value
- **Fouling Resistance (Rf)**: Calculated from HEI methodology
- **Air In-Leakage Detection**: Based on pressure and subcooling data

Performance states:

- `CLEAN`: CF >= 85%, normal operation
- `LIGHT_FOULING`: 75% <= CF < 85%
- `MODERATE_FOULING`: 60% <= CF < 75%
- `SEVERE_FOULING`: CF < 60%, cleaning recommended
- `AIR_BINDING`: Elevated subcooling, potential air leak
- `WATERBOX_FOULING`: Asymmetric performance between passes

### 4. Optimization Engine

**Optimization Components** (`optimization/`)

- **Cleaning Schedule Optimizer**: Determines optimal cleaning timing based on fouling rate, heat rate penalty, and cleaning cost
- **Cooling Water Optimizer**: Optimizes CW flow rate vs pump power consumption
- **Backpressure Analyzer**: Quantifies turbine efficiency impact from elevated condenser pressure

### 5. Explainability

**Diagnostic Explainer** (`explainability/diagnostic_explainer.py`)

- Factor attribution for performance degradation
- Evidence chain with supporting observations
- Recommendation justification with economic impact
- Three explanation styles: Technical, Operator, Executive

### 6. Monitoring

**Prometheus Metrics** (`monitoring/metrics.py`)

```
condensync_cleanliness_factor{unit, condenser}
condensync_ttd_kelvin{unit, condenser}
condensync_dca_kelvin{unit, condenser}
condensync_fouling_resistance{unit, condenser}
condensync_backpressure_kpa{unit}
condensync_heat_rate_penalty_percent{unit}
condensync_diagnoses_total{condition, severity}
condensync_diagnosis_duration_seconds{}
condensync_cw_flow_m3s{unit, condenser}
condensync_cw_inlet_temp_kelvin{unit}
```

### 7. Bounds Validation

**Physical Bounds** (`core/bounds_validator.py`)

| Parameter | Min | Max | Unit | Standard |
|-----------|-----|-----|------|----------|
| Condenser Pressure | 0.5 | 15 | psia | HEI 3098 |
| CW Inlet Temp | 273 | 318 | K | HEI 3098 |
| CW Outlet Temp | 275 | 325 | K | HEI 3098 |
| TTD | 0 | 30 | K | HEI 3098 |
| DCA | 0 | 20 | K | HEI 3098 |
| Cleanliness Factor | 0 | 100 | % | HEI 3098 |
| CW Velocity | 0.3 | 3.0 | m/s | HEI 3098 |
| Heat Load | 0 | 2000 | MW | - |
| Tube Count | 1 | 100000 | - | - |

## Data Flow

```
1. Data Ingestion
   +-- Process Data: Condenser pressure, hotwell temp, extraction flows
   +-- Cooling Water: Inlet/outlet temps, flow rate, pump power
   +-- Design Data: Tube geometry, surface area, design U-value
   +-- Ambient: Wet bulb temp, atmospheric pressure

2. Validation
   +-- Bounds checking (physical limits per HEI)
   +-- Data quality flags (sensor health, range checks)
   +-- Steady-state detection (avoid transient analysis)

3. Performance Calculation
   +-- LMTD from CW inlet/outlet and saturation temperature
   +-- Actual U-value from heat duty, area, and LMTD
   +-- Design U-value from HEI curves (CW velocity, temp, tube material)
   +-- Cleanliness Factor = U_actual / U_design

4. Diagnostic Classification
   +-- CF-based condition assessment
   +-- TTD deviation analysis
   +-- Air in-leakage detection
   +-- Waterbox asymmetry check

5. Optimization Analysis
   +-- Fouling trend projection
   +-- Optimal cleaning timing
   +-- CW flow optimization
   +-- Economic impact quantification

6. Explainability
   +-- Factor contribution breakdown
   +-- Evidence chain construction
   +-- Recommendation generation

7. Output
   +-- Performance metrics with confidence
   +-- Heat rate penalty in BTU/kWh and $/year
   +-- CO2 impact from efficiency loss
   +-- Maintenance recommendations with ROI
```

## Deployment Modes

### Edge Mode

Standalone deployment at plant site for real-time monitoring.

```
+------------------+     +------------------+
|   Plant DCS      |---->|  Edge Gateway    |
|   (OPC-UA)       |     |  (Condensync)    |
+------------------+     +--------+---------+
                                  |
                                  v
                         +------------------+
                         |  Local Dashboard |
                         |  (Grafana)       |
                         +------------------+
```

**Characteristics:**
- Full functionality without cloud connectivity
- Local data storage (SQLite/TimescaleDB)
- Prometheus metrics for local Grafana
- Offline operation capable
- Low latency (<100ms response)

### Edge + Central Mode

Edge processing with central aggregation and fleet analytics.

```
+------------------+     +------------------+     +------------------+
|   Plant DCS      |---->|  Edge Gateway    |---->|  Central Cloud   |
|   (OPC-UA)       |     |  (Condensync)    |     |  (GreenLang)     |
+------------------+     +--------+---------+     +--------+---------+
                                  |                        |
                                  v                        v
                         +------------------+     +------------------+
                         |  Local Dashboard |     | Fleet Analytics  |
                         +------------------+     +------------------+
```

**Characteristics:**
- Real-time local processing
- Periodic sync to central (configurable: 1min - 1hr)
- Fleet-wide benchmarking
- Cross-plant optimization insights
- Survives network outages

### Offline Mode

Batch analysis for historical data or assessment projects.

```
+------------------+     +------------------+     +------------------+
|   Data Export    |---->|  Batch Processor |---->|  Report Output   |
|   (CSV/Excel)    |     |  (Condensync)    |     |  (PDF/Excel)     |
+------------------+     +------------------+     +------------------+
```

**Characteristics:**
- No connectivity required
- Process historical datasets
- Generate assessment reports
- Commissioning validation
- Performance guarantee testing

## Integration Points

### Input Integrations

| System | Protocol | Data |
|--------|----------|------|
| DCS | OPC-UA | Real-time process variables |
| Historian | PI-SDK/REST | Historical trends |
| CMMS | REST | Maintenance records, cleaning history |
| Weather | REST | Ambient conditions |
| ERP | SAP RFC | Cost data, fuel prices |

### Output Integrations

| System | Protocol | Data |
|--------|----------|------|
| DCS | OPC-UA Write | Setpoint recommendations |
| CMMS | REST | Work order generation |
| Dashboard | Prometheus/Grafana | Real-time visualization |
| Reporting | REST/Email | Scheduled reports |
| Central | gRPC/REST | Fleet aggregation |

## Zero-Hallucination Guarantees

1. **Deterministic Calculations**: All formulas from HEI 3098 Standards
2. **No AI Inference**: Classification uses explicit threshold-based rules
3. **Reproducibility**: Same inputs produce identical outputs (SHA-256 verified)
4. **Frozen Dataclasses**: All data structures are immutable
5. **Provenance Tracking**: Every calculation traceable with hashes
6. **Standard References**: All equations cite specific HEI sections

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

## Resource Requirements

### Edge Deployment

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 250m | 1000m |
| Memory | 256Mi | 1Gi |
| Storage | 1Gi | 10Gi |

### Central Deployment

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 2000m |
| Memory | 512Mi | 2Gi |
| Storage | 10Gi | 100Gi |

## Standards Compliance

- **HEI 3098**: Standards for Steam Surface Condensers (11th Edition)
- **ASME PTC 12.2**: Steam Surface Condenser Performance Test Code
- **EPRI Guidelines**: Condenser Performance Monitoring
- **OpenMetrics**: Prometheus metrics format
- **OpenTelemetry**: Distributed tracing

## Security

- **Authentication**: JWT/OAuth2 for API access
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: TLS 1.3 for all communications
- **Audit Logging**: All operations logged with user context
- **Network Isolation**: Kubernetes network policies
