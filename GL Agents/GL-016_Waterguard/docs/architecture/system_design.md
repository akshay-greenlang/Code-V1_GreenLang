# GL-016 Waterguard - System Architecture Design

## Document Information

| Field | Value |
|-------|-------|
| Document ID | GL-016-ARCH-001 |
| Version | 1.0.0 |
| Status | Approved |
| Author | GreenLang Engineering |
| Last Updated | 2025-12-27 |
| Classification | Internal |

---

## 1. Executive Summary

GL-016 Waterguard is a supervisory control and optimization agent designed for industrial boiler water treatment systems. This document describes the system architecture, including component design, data flows, integration patterns, and safety considerations.

### 1.1 Purpose

This architecture document serves to:
- Define the technical architecture of the Waterguard agent
- Establish design patterns and integration approaches
- Document safety-critical design decisions
- Provide guidance for implementation and deployment

### 1.2 Scope

This document covers:
- High-level system architecture
- Component design and responsibilities
- Data models and flows
- Integration patterns (OT, IT, Cloud)
- Safety and security architecture
- Deployment architecture

---

## 2. System Overview

### 2.1 Context Diagram

```
                                    +------------------+
                                    |   Cloud Platform |
                                    |   (GreenLang)    |
                                    +--------+---------+
                                             |
                                             | HTTPS/gRPC
                                             |
+---------------------------+       +--------v---------+       +---------------------------+
|    Enterprise Systems     |       |                  |       |    Historian/SCADA        |
|---------------------------|       |    GL-016        |       |---------------------------|
| - ERP (SAP, Oracle)       |<----->|   WATERGUARD     |<----->| - OSIsoft PI              |
| - CMMS (Maximo, SAP PM)   |       |                  |       | - Honeywell PHD           |
| - Environmental Reporting |       |   AGENT          |       | - Wonderware              |
+---------------------------+       |                  |       +---------------------------+
            ^                       +--------+---------+                    ^
            |                                |                              |
            | REST API                       | OPC UA / Modbus              | Sensor Data
            |                                |                              |
            v                                v                              v
+---------------------------+       +--------+---------+       +---------------------------+
|    Reporting & Analytics  |       |  Boiler Water    |       |    Field Instruments      |
|---------------------------|       |  Treatment       |       |---------------------------|
| - Grafana Dashboards      |       |  Equipment       |       | - Conductivity Analyzers  |
| - Custom Reports          |       +------------------+       | - pH Transmitters         |
| - Regulatory Submissions  |       | - Blowdown Valves|       | - Silica Analyzers        |
+---------------------------+       | - Dosing Pumps   |       | - DO Sensors              |
                                    | - Makeup Valves  |       | - Flow Meters             |
                                    +------------------+       +---------------------------+
```

### 2.2 System Boundaries

**In Scope:**
- Boiler water chemistry optimization
- Blowdown control (continuous and intermittent)
- Chemical dosing control
- Water and energy savings tracking
- Regulatory reporting support

**Out of Scope:**
- Combustion control (see GL-002 Flameguard)
- Steam header pressure control
- Feedwater heater control
- Boiler startup/shutdown sequences

---

## 3. Architecture Principles

### 3.1 Design Principles

| Principle | Description |
|-----------|-------------|
| **Safety First** | All designs prioritize safety; fail-safe defaults |
| **Deterministic** | Calculations produce reproducible results |
| **Explainable** | All recommendations include rationale |
| **Auditable** | Complete audit trail with SHA-256 provenance |
| **Resilient** | Graceful degradation on failures |
| **Scalable** | Horizontal scaling for high availability |

### 3.2 Quality Attributes

| Attribute | Target | Measurement |
|-----------|--------|-------------|
| Availability | 99.95% | Uptime percentage |
| Latency | < 100ms | P99 API response time |
| Throughput | 1000 req/s | Concurrent API requests |
| Safety Response | < 100ms | Interlock response time |
| Data Retention | 7 years | Audit trail availability |

---

## 4. Component Architecture

### 4.1 High-Level Component Diagram

```
+------------------------------------------------------------------+
|                       GL-016 WATERGUARD                           |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------------+  +------------------------+            |
|  |     API Gateway        |  |    Event Processor     |            |
|  |------------------------|  |------------------------|            |
|  | - REST (FastAPI)       |  | - Kafka Consumer       |            |
|  | - GraphQL (Strawberry) |  | - Event Router         |            |
|  | - gRPC                 |  | - State Machine        |            |
|  | - WebSocket            |  | - Alarm Manager        |            |
|  +------------------------+  +------------------------+            |
|            |                           |                           |
|            v                           v                           |
|  +------------------------+  +------------------------+            |
|  |   Chemistry Engine     |  |   Control Engine       |            |
|  |------------------------|  |------------------------|            |
|  | - CoC Calculator       |  | - Blowdown Controller  |            |
|  | - Mass Balance         |  | - Dosing Controller    |            |
|  | - Constraint Monitor   |  | - Setpoint Manager     |            |
|  | - Trend Analysis       |  | - Rate Limiter         |            |
|  +------------------------+  +------------------------+            |
|            |                           |                           |
|            v                           v                           |
|  +------------------------+  +------------------------+            |
|  |  Optimization Engine   |  |    Safety Engine       |            |
|  |------------------------|  |------------------------|            |
|  | - CVXPY Optimizer      |  | - SIL-3 Interlocks     |            |
|  | - Cost Function        |  | - Safety Gates         |            |
|  | - Constraint Solver    |  | - Watchdog             |            |
|  | - Scenario Analysis    |  | - Fail-Safe Logic      |            |
|  +------------------------+  +------------------------+            |
|            |                           |                           |
|            v                           v                           |
|  +------------------------+  +------------------------+            |
|  |  Explainability Layer  |  |    Audit Layer         |            |
|  |------------------------|  |------------------------|            |
|  | - SHAP Analyzer        |  | - Event Logger         |            |
|  | - LIME Explainer       |  | - Provenance Tracker   |            |
|  | - Decision Trace       |  | - Hash Generator       |            |
|  | - Rationale Generator  |  | - Archive Manager      |            |
|  +------------------------+  +------------------------+            |
|                                                                    |
+------------------------------------------------------------------+
|                        DATA LAYER                                  |
+------------------------------------------------------------------+
|  +----------------+  +----------------+  +----------------+        |
|  | PostgreSQL     |  | TimescaleDB    |  | Redis Cache    |        |
|  | (Config/Audit) |  | (Time Series)  |  | (State/Session)|        |
|  +----------------+  +----------------+  +----------------+        |
+------------------------------------------------------------------+
```

### 4.2 Component Descriptions

#### 4.2.1 API Gateway

**Responsibilities:**
- HTTP request routing and validation
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning

**Technology Stack:**
- FastAPI for REST endpoints
- Strawberry GraphQL for GraphQL
- gRPC with reflection
- JWT-based authentication

#### 4.2.2 Event Processor

**Responsibilities:**
- Kafka topic consumption
- Event routing and dispatch
- State machine management
- Alarm detection and notification

**Key Patterns:**
- Event sourcing for state management
- CQRS for read/write separation
- Saga pattern for complex workflows

#### 4.2.3 Chemistry Engine

**Responsibilities:**
- Cycles of concentration calculation
- Mass balance modeling
- Chemistry constraint monitoring
- Trend analysis and forecasting

**Algorithms:**
- CoC calculation: `CoC = Blowdown_TDS / Makeup_TDS`
- Mass balance: Conservation equations for TDS, silica, phosphate
- Constraint checking: Min/max limits with alarm thresholds

#### 4.2.4 Control Engine

**Responsibilities:**
- Blowdown valve control (0-100%)
- Chemical dosing pump control
- Setpoint management
- Rate limiting for safe transitions

**Control Modes:**
- Continuous modulating control
- Intermittent timed control
- Cascade control (conductivity -> blowdown)

#### 4.2.5 Optimization Engine

**Responsibilities:**
- Economic optimization of CoC
- Multi-objective optimization (water, energy, chemicals)
- Constraint satisfaction
- What-if scenario analysis

**Optimization Formulation:**

```
Minimize: Total_Cost = Water_Cost + Energy_Cost + Chemical_Cost

Subject to:
  - CoC_min <= CoC <= CoC_max
  - Conductivity <= Conductivity_max
  - Silica <= Silica_max
  - pH_min <= pH <= pH_max
  - DO <= DO_max
```

#### 4.2.6 Safety Engine

**Responsibilities:**
- SIL-3 interlock implementation
- Safety gate enforcement
- Watchdog monitoring
- Fail-safe state transitions

**Safety Logic:**

```python
class SafetyGate:
    def check(self, state: ChemistryState) -> bool:
        if state.conductivity > TRIP_HIGH:
            return self.trip("high_conductivity")
        if state.level < LEVEL_LOW:
            return self.trip("low_level")
        if state.pressure > PRESSURE_HIGH:
            return self.trip("high_pressure")
        return True
```

#### 4.2.7 Explainability Layer

**Responsibilities:**
- SHAP value computation
- LIME local explanations
- Decision trace generation
- Human-readable rationale

**Explanation Format:**

```json
{
  "recommendation": "Increase CoC from 6.0 to 6.5",
  "confidence": 0.92,
  "shap_values": {
    "conductivity": 0.35,
    "silica_margin": 0.25,
    "steam_demand": 0.20,
    "water_cost": 0.15,
    "treatment_cost": 0.05
  },
  "rationale": "Current silica (100 ppm) is 33% below limit (150 ppm), allowing safe CoC increase. This will reduce blowdown by 8% and save 15 m3/day of makeup water.",
  "risks": [
    {
      "risk": "silica_approach",
      "probability": 0.15,
      "mitigation": "Monitor silica trend; reduce CoC if approaching 120 ppm"
    }
  ]
}
```

#### 4.2.8 Audit Layer

**Responsibilities:**
- Event logging with timestamps
- SHA-256 hash chain for integrity
- 7-year data retention
- Regulatory report generation

**Audit Record Format:**

```json
{
  "event_id": "EVT-20251227-001234",
  "timestamp": "2025-12-27T10:00:00.000Z",
  "event_type": "SETPOINT_CHANGE",
  "actor": "AGENT",
  "previous_value": 6.0,
  "new_value": 6.5,
  "parameter": "coc_target",
  "rationale": "Optimization recommendation accepted",
  "approval": {
    "type": "AUTOMATIC",
    "mode": "AUTONOMOUS"
  },
  "hash": "sha256:a1b2c3d4...",
  "previous_hash": "sha256:9z8y7x6w..."
}
```

---

## 5. Data Architecture

### 5.1 Data Model

#### 5.1.1 Core Entities

```
+------------------+       +------------------+       +------------------+
|     Boiler       |       | ChemistryReading |       | ControlAction    |
+------------------+       +------------------+       +------------------+
| id: UUID         |       | id: UUID         |       | id: UUID         |
| name: String     |       | boiler_id: FK    |       | boiler_id: FK    |
| pressure_psig    |       | timestamp: TS    |       | timestamp: TS    |
| capacity_klb_hr  |       | ph: Float        |       | action_type      |
| type: Enum       |       | conductivity     |       | parameter        |
+------------------+       | silica_ppm       |       | previous_value   |
        |                  | dissolved_o2_ppb |       | new_value        |
        |                  | feedwater_cond   |       | status: Enum     |
        v                  +------------------+       +------------------+
+------------------+                |                         |
| ChemistryLimit   |                |                         |
+------------------+                v                         v
| id: UUID         |       +------------------+       +------------------+
| boiler_id: FK    |       | Recommendation   |       | AuditEvent       |
| parameter        |       +------------------+       +------------------+
| min_value        |       | id: UUID         |       | id: UUID         |
| max_value        |       | boiler_id: FK    |       | event_type       |
| alarm_low        |       | timestamp: TS    |       | timestamp: TS    |
| alarm_high       |       | recommendation   |       | actor            |
| trip_low         |       | confidence       |       | payload: JSONB   |
| trip_high        |       | explanation      |       | hash: String     |
+------------------+       | status: Enum     |       | prev_hash: String|
                           +------------------+       +------------------+
```

### 5.2 Data Flows

#### 5.2.1 Real-Time Data Flow

```
Field Sensors                   Waterguard Agent                      Control Outputs
+-------------+                +----------------------+               +-------------+
| Conductivity|----+           | 1. Ingest            |               | Blowdown    |
| Analyzer    |    |           | 2. Validate          |               | Valve       |
+-------------+    |           | 3. Store (TimescaleDB)|----+         +-------------+
                   |           | 4. Calculate CoC     |    |                ^
+-------------+    +---------->| 5. Check Constraints |    |                |
| pH Probe    |--------------->| 6. Optimize          |----+-------->OPC UA/Modbus
+-------------+    +---------->| 7. Generate Command  |    |                |
                   |           | 8. Apply Rate Limit  |    |                v
+-------------+    |           | 9. Execute           |    |         +-------------+
| Silica      |----+           | 10. Audit Log        |    |         | Dosing      |
| Analyzer    |                +----------------------+    |         | Pumps       |
+-------------+                         |                  |         +-------------+
                                        v                  |
                               +----------------------+    |
                               | Kafka Topics         |    |
                               | - sensor.readings    |    |
                               | - setpoints.commands |    |
                               | - alarms.events      |    |
                               | - audit.trail        |<---+
                               +----------------------+
```

#### 5.2.2 Optimization Cycle

```
Every 15 minutes:

1. Collect Current State
   +----------------------------+
   | Chemistry Readings (avg)   |
   | Operating Conditions       |
   | Current Setpoints          |
   +----------------------------+
              |
              v
2. Run Optimization
   +----------------------------+
   | Cost Function Evaluation   |
   | Constraint Checking        |
   | CVXPY Solver               |
   +----------------------------+
              |
              v
3. Generate Recommendation
   +----------------------------+
   | Optimal CoC                |
   | Blowdown Adjustment        |
   | Chemical Dosing Changes    |
   +----------------------------+
              |
              v
4. Explain Decision
   +----------------------------+
   | SHAP Feature Importance    |
   | Risk Assessment            |
   | Human-Readable Rationale   |
   +----------------------------+
              |
              v
5. Execute (Based on Mode)
   +----------------------------+
   | Advisory: Display Only     |
   | Supervisory: Await Approval|
   | Autonomous: Auto-Execute   |
   +----------------------------+
```

---

## 6. Integration Architecture

### 6.1 OT Integration

#### 6.1.1 OPC UA Integration

```yaml
opc_ua:
  server_url: "opc.tcp://plc-001:4840"
  security_mode: SignAndEncrypt
  security_policy: Basic256Sha256
  authentication:
    type: certificate
    cert_path: /certs/waterguard.pem
    key_path: /certs/waterguard.key

  subscriptions:
    - node_id: "ns=2;s=Boiler1.Conductivity"
      sampling_interval_ms: 1000
      publish_interval_ms: 1000

    - node_id: "ns=2;s=Boiler1.pH"
      sampling_interval_ms: 1000
      publish_interval_ms: 1000

  write_nodes:
    - node_id: "ns=2;s=Boiler1.BlowdownSetpoint"
      data_type: Float
      access_level: ReadWrite
```

#### 6.1.2 Modbus Integration

```yaml
modbus:
  host: "192.168.1.100"
  port: 502
  unit_id: 1

  registers:
    # Read registers (Input Registers)
    - address: 30001
      name: conductivity
      scale: 0.1
      unit: umho

    - address: 30002
      name: ph
      scale: 0.01
      unit: dimensionless

    # Write registers (Holding Registers)
    - address: 40001
      name: blowdown_setpoint
      scale: 0.1
      unit: percent

    - address: 40002
      name: dosing_pump_speed
      scale: 0.1
      unit: percent
```

### 6.2 IT Integration

#### 6.2.1 Enterprise API Integration

```yaml
integrations:
  sap:
    type: REST
    base_url: "https://sap.company.com/api/v1"
    auth:
      type: oauth2
      token_url: "https://sap.company.com/oauth/token"
    endpoints:
      work_orders: /pm/workorders
      materials: /mm/materials

  cmms:
    type: REST
    base_url: "https://maximo.company.com/api"
    auth:
      type: api_key
    endpoints:
      assets: /mxasset
      meters: /mxmeter
```

### 6.3 Cloud Integration

#### 6.3.1 GreenLang Platform Integration

```yaml
greenlang_platform:
  api_url: "https://api.greenlang.io/v1"
  agent_id: "GL-016"
  region: "us-east-1"

  telemetry:
    enabled: true
    interval_seconds: 60
    metrics:
      - coc
      - water_savings
      - energy_savings
      - emissions_avoided

  reporting:
    enabled: true
    schedule: "0 0 * * *"  # Daily at midnight
    formats:
      - json
      - pdf
```

---

## 7. Safety Architecture

### 7.1 Safety Layers

```
+------------------------------------------------------------------+
|                     LAYER 4: PHYSICAL SAFEGUARDS                  |
|   Relief valves, rupture disks, manual overrides                  |
+------------------------------------------------------------------+
                              ^
                              |
+------------------------------------------------------------------+
|                     LAYER 3: SAFETY INSTRUMENTED SYSTEM (SIS)     |
|   Independent hardwired interlocks, low water cutoff              |
+------------------------------------------------------------------+
                              ^
                              |
+------------------------------------------------------------------+
|                     LAYER 2: GL-016 WATERGUARD SAFETY ENGINE      |
|   Software interlocks, rate limiting, fail-safe logic             |
+------------------------------------------------------------------+
                              ^
                              |
+------------------------------------------------------------------+
|                     LAYER 1: BASIC PROCESS CONTROL (BCS)          |
|   PID loops, cascade control, alarm management                    |
+------------------------------------------------------------------+
                              ^
                              |
+------------------------------------------------------------------+
|                     LAYER 0: PROCESS DESIGN                       |
|   Inherently safer design, materials selection                    |
+------------------------------------------------------------------+
```

### 7.2 Safety Interlock Logic

```python
class SafetyInterlock:
    """SIL-3 rated safety interlock implementation."""

    def __init__(self):
        self.interlocks = {
            "low_water_level": LowWaterLevelInterlock(),
            "high_pressure": HighPressureInterlock(),
            "high_conductivity": HighConductivityInterlock(),
            "chemical_leak": ChemicalLeakInterlock(),
            "communication_loss": CommunicationLossInterlock(),
        }

    def check_all(self, state: SystemState) -> InterlockResult:
        results = []
        for name, interlock in self.interlocks.items():
            result = interlock.check(state)
            if result.tripped:
                self.execute_safe_action(name, result)
            results.append(result)
        return InterlockResult.aggregate(results)

    def execute_safe_action(self, interlock_name: str, result: InterlockResult):
        """Execute fail-safe action for tripped interlock."""
        actions = {
            "low_water_level": [
                ("blowdown_valve", "CLOSE"),
                ("dosing_pumps", "STOP"),
            ],
            "high_pressure": [
                ("blowdown_valve", "OPEN_FULL"),
            ],
            "high_conductivity": [
                ("blowdown_valve", "OPEN_FULL"),
            ],
            "chemical_leak": [
                ("dosing_pumps", "STOP"),
                ("chemical_valves", "CLOSE"),
            ],
            "communication_loss": [
                ("mode", "FALLBACK"),
            ],
        }
        for actuator, action in actions.get(interlock_name, []):
            self.execute(actuator, action)
```

### 7.3 Watchdog Architecture

```
+------------------+       +------------------+       +------------------+
|   Main Process   |       |   Watchdog       |       |   Safety PLC     |
|------------------|       |   Process        |       |   (Hardware)     |
| - Control Logic  |       |------------------|       |------------------|
| - Optimization   |<----->| - Heartbeat Check|<----->| - Independent    |
| - API Server     |       | - Timeout Monitor|       |   Monitoring     |
|                  |       | - Failsafe Trigger|      | - Hardwired      |
|                  |       |                  |       |   Interlocks     |
+------------------+       +------------------+       +------------------+
        |                          |                          |
        | Heartbeat (500ms)        | Trip Signal              | Direct Wire
        v                          v                          v
+------------------------------------------------------------------+
|                        SAFE STATE                                 |
| - Maintain current setpoints                                      |
| - Enter fallback mode                                             |
| - Alert operators                                                 |
| - Log event with full audit trail                                 |
+------------------------------------------------------------------+
```

---

## 8. Deployment Architecture

### 8.1 Production Deployment

```
+------------------------------------------------------------------+
|                     KUBERNETES CLUSTER                            |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |   Waterguard     |  |   Waterguard     |  |   Waterguard     |  |
|  |   Pod 1          |  |   Pod 2          |  |   Pod 3          |  |
|  |------------------|  |------------------|  |------------------|  |
|  | - API Container  |  | - API Container  |  | - API Container  |  |
|  | - Worker         |  | - Worker         |  | - Worker         |  |
|  | - Sidecar        |  | - Sidecar        |  | - Sidecar        |  |
|  +------------------+  +------------------+  +------------------+  |
|            |                   |                   |               |
|            +-------------------+-------------------+               |
|                               |                                    |
|                    +----------v-----------+                        |
|                    |   Service (LoadBalancer)                      |
|                    |   - REST: 8080                                |
|                    |   - gRPC: 50016                               |
|                    +----------+-----------+                        |
|                               |                                    |
+------------------------------------------------------------------+
                               |
                    +----------v-----------+
                    |   Ingress Controller  |
                    |   - TLS Termination   |
                    |   - Path Routing      |
                    +-----------------------+
```

### 8.2 High Availability Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-016-waterguard
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: waterguard
              topologyKey: kubernetes.io/hostname
      containers:
        - name: waterguard
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
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
```

---

## 9. Security Architecture

### 9.1 Security Layers

```
+------------------------------------------------------------------+
|                     NETWORK SECURITY                              |
|   - Firewall rules, network segmentation, VPN                     |
+------------------------------------------------------------------+
                              |
+------------------------------------------------------------------+
|                     TRANSPORT SECURITY                            |
|   - TLS 1.3, mTLS, certificate management                        |
+------------------------------------------------------------------+
                              |
+------------------------------------------------------------------+
|                     APPLICATION SECURITY                          |
|   - JWT authentication, RBAC authorization, input validation     |
+------------------------------------------------------------------+
                              |
+------------------------------------------------------------------+
|                     DATA SECURITY                                 |
|   - Encryption at rest (AES-256), key management                 |
+------------------------------------------------------------------+
                              |
+------------------------------------------------------------------+
|                     AUDIT & COMPLIANCE                            |
|   - Audit logging, access tracking, regulatory reporting         |
+------------------------------------------------------------------+
```

### 9.2 Authentication & Authorization

```yaml
security:
  authentication:
    type: jwt
    issuer: "https://auth.greenlang.io"
    audience: "gl-016-waterguard"
    algorithms:
      - RS256

  authorization:
    type: rbac
    roles:
      operator:
        permissions:
          - read:chemistry
          - read:recommendations
          - execute:approved_actions

      engineer:
        permissions:
          - read:*
          - write:setpoints
          - write:configuration
          - approve:recommendations

      administrator:
        permissions:
          - "*"
```

---

## 10. Appendices

### 10.1 Glossary

| Term | Definition |
|------|------------|
| CoC | Cycles of Concentration - ratio of TDS in blowdown to TDS in makeup |
| TDS | Total Dissolved Solids |
| SIL | Safety Integrity Level per IEC 61511 |
| DO | Dissolved Oxygen |

### 10.2 References

1. IEC 61511 - Functional Safety
2. IEC 62443 - Industrial Cybersecurity
3. ASME Boiler and Pressure Vessel Code
4. ABMA Guidelines for Industrial Boilers

### 10.3 Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-27 | GreenLang Engineering | Initial release |
