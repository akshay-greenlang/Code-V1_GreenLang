# GL-017 CONDENSYNC Architecture

## System Architecture Documentation

**Version:** 1.0.0
**Last Updated:** December 2025
**Author:** GreenLang Team

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Component Descriptions](#component-descriptions)
3. [Data Flow](#data-flow)
4. [Integration Points](#integration-points)
5. [Deployment Topology](#deployment-topology)
6. [Scalability Design](#scalability-design)
7. [High Availability](#high-availability)

---

## System Architecture Overview

### High-Level Architecture Diagram

```
+==============================================================================+
|                        GL-017 CONDENSYNC SYSTEM ARCHITECTURE                  |
+==============================================================================+

                              EXTERNAL SYSTEMS
    +--------+  +--------+  +--------+  +--------+  +--------+  +--------+
    | SCADA  |  |  DCS   |  |Historian|  |Cooling |  |  Air   |  | Weather|
    |OPC-UA  |  |OPC-UA  |  | OPC-HDA |  | Tower  |  |Removal |  |  API   |
    +---+----+  +---+----+  +---+----+  +---+----+  +---+----+  +---+----+
        |           |           |           |           |           |
        +-----------+-----------+-----------+-----------+-----------+
                                    |
                        +-----------v-----------+
                        |   DATA ACQUISITION    |
                        |      LAYER            |
                        +-----------+-----------+
                                    |
+==============================================================================+
|                           GREENLANG PLATFORM                                  |
|==============================================================================|
|                                                                               |
|  +-------------------------------------------------------------------------+ |
|  |                        MESSAGE BUS (KAFKA)                              | |
|  |   +-------------+  +-------------+  +-------------+  +-------------+   | |
|  |   |  Events     |  | Metrics     |  |   Alerts    |  |Optimization |   | |
|  |   |   Topic     |  |   Topic     |  |   Topic     |  |   Topic     |   | |
|  |   +-------------+  +-------------+  +-------------+  +-------------+   | |
|  +-------------------------------------------------------------------------+ |
|                                    |                                         |
|        +---------------------------+---------------------------+             |
|        |                           |                           |             |
|        v                           v                           v             |
|  +-----------+              +-----------+              +-----------+         |
|  |  AGENT    |              |  AGENT    |              |  AGENT    |         |
|  | REPLICA 1 |              | REPLICA 2 |              | REPLICA 3 |         |
|  +-----------+              +-----------+              +-----------+         |
|        |                           |                           |             |
|        +---------------------------+---------------------------+             |
|                                    |                                         |
|                    +---------------v---------------+                         |
|                    |    CONDENSYNC AGENT CORE      |                         |
|                    |-------------------------------|                         |
|                    |  +-------------------------+  |                         |
|                    |  |    ORCHESTRATOR         |  |                         |
|                    |  |  CondenserOptimization  |  |                         |
|                    |  |        Agent            |  |                         |
|                    |  +------------+------------+  |                         |
|                    |               |               |                         |
|                    |  +------------v------------+  |                         |
|                    |  |    TOOL EXECUTOR        |  |                         |
|                    |  |  CondenserToolExecutor  |  |                         |
|                    |  +------------+------------+  |                         |
|                    |               |               |                         |
|                    |  +------------v------------+  |                         |
|                    |  |     CALCULATORS         |  |                         |
|                    |  | - Heat Transfer         |  |                         |
|                    |  | - Vacuum                |  |                         |
|                    |  | - Fouling               |  |                         |
|                    |  | - Efficiency            |  |                         |
|                    |  +-------------------------+  |                         |
|                    +---------------+---------------+                         |
|                                    |                                         |
|        +---------------------------+---------------------------+             |
|        |                           |                           |             |
|        v                           v                           v             |
|  +-----------+              +-----------+              +-----------+         |
|  |  Redis    |              | InfluxDB  |              |PostgreSQL |         |
|  |  (Cache)  |              |   (TSDB)  |              |  (Config) |         |
|  +-----------+              +-----------+              +-----------+         |
|                                                                               |
+==============================================================================+
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
  +-----------+              +-----------+              +-----------+
  |  CMMS     |              |  Reports  |              | Alerting  |
  | Maximo/   |              |   S3      |              | PagerDuty |
  |  SAP PM   |              |  Storage  |              |   Slack   |
  +-----------+              +-----------+              +-----------+
```

### Layer Description

| Layer | Purpose | Components |
|-------|---------|------------|
| External Systems | Plant data sources | SCADA, DCS, Historian, PLCs |
| Data Acquisition | Collect and validate | OPC-UA clients, Modbus drivers |
| Message Bus | Event streaming | Apache Kafka |
| Agent Core | Processing logic | Orchestrator, Tools, Calculators |
| Storage | Persistence | Redis, InfluxDB, PostgreSQL |
| Output Systems | Deliver results | CMMS, Reports, Alerts |

---

## Component Descriptions

### 1. Orchestrator (CondenserOptimizationAgent)

The main orchestration component that coordinates all condenser optimization activities.

**Responsibilities:**
- Execute main workflow cycle
- Coordinate data gathering
- Invoke calculation tools
- Generate optimization results
- Manage historical data storage
- Calculate performance scores
- Generate provenance hashes

**Location:** `condenser_optimization_agent.py`

```
+------------------------------------------------------+
|           CondenserOptimizationAgent                  |
|------------------------------------------------------|
| Attributes:                                          |
|   - config: AgentConfiguration                       |
|   - message_bus: MessageBus                          |
|   - task_scheduler: TaskScheduler                    |
|   - safety_monitor: SafetyMonitor                    |
|   - coordination_layer: CoordinationLayer            |
|------------------------------------------------------|
| Methods:                                             |
|   + orchestrate(input_data) -> OrchestrationResult   |
|   + execute() -> CondenserPerformanceResult          |
|   + analyze_condenser_performance()                  |
|   + optimize_vacuum_pressure()                       |
|   + optimize_cooling_water_flow()                    |
|   + calculate_heat_transfer_efficiency()             |
|   + detect_air_inleakage()                          |
|   + predict_fouling()                               |
+------------------------------------------------------+
```

### 2. Tool Executor (CondenserToolExecutor)

Executes deterministic calculation tools with zero-hallucination guarantees.

**Responsibilities:**
- Validate tool parameters
- Execute calculation logic
- Generate provenance hashes
- Track execution metrics
- Handle errors gracefully

**Location:** `tools.py`

```
+------------------------------------------------------+
|           CondenserToolExecutor                       |
|------------------------------------------------------|
| Attributes:                                          |
|   - tools: List[Dict]                                |
|   - _provenance_enabled: bool                        |
|   - _tool_map: Dict[str, Dict]                       |
|------------------------------------------------------|
| Methods:                                             |
|   + execute_tool(name, params) -> Dict               |
|   + _validate_parameters(name, params)               |
|   + _calculate_heat_transfer_coefficient()           |
|   + _optimize_vacuum_pressure()                      |
|   + _detect_air_inleakage()                         |
|   + _calculate_fouling_factor()                      |
|   + _optimize_cooling_water_flow()                   |
|   + _generate_provenance_hash()                      |
+------------------------------------------------------+
```

### 3. Configuration Manager (AgentConfiguration)

Manages all agent configuration with Pydantic validation.

**Location:** `config.py`

```
+------------------------------------------------------+
|           Configuration Hierarchy                     |
|------------------------------------------------------|
|                                                      |
|  AgentConfiguration                                  |
|   |                                                  |
|   +-- CondenserConfiguration[]                       |
|   |    +-- tube_material: TubeMaterial               |
|   |    +-- tube_pattern: TubePattern                 |
|   |    +-- design parameters                         |
|   |                                                  |
|   +-- CoolingWaterConfig                             |
|   |    +-- system_type: CoolingSystemType            |
|   |    +-- pump configuration                        |
|   |                                                  |
|   +-- VacuumSystemConfig                             |
|   |    +-- primary_pump_type: VacuumPumpType         |
|   |    +-- air removal capacity                      |
|   |                                                  |
|   +-- WaterQualityLimits                             |
|   +-- PerformanceTargets                             |
|   +-- AlertThresholds                                |
|   +-- SCADAIntegration                               |
|   +-- CoolingTowerIntegration                        |
|   +-- TurbineCoordination                            |
|                                                      |
+------------------------------------------------------+
```

### 4. Calculators

Specialized calculation modules for domain-specific computations.

**Location:** `calculators/`

| Calculator | File | Purpose |
|------------|------|---------|
| Heat Transfer | `heat_transfer_calculator.py` | U-value, LMTD, TTD calculations |
| Vacuum | `vacuum_calculator.py` | Pressure, saturation temp |
| Fouling | `fouling_calculator.py` | Fouling factor, cleaning prediction |
| Efficiency | `efficiency_calculator.py` | Performance metrics |
| Provenance | `provenance.py` | Hash generation, audit trail |

### 5. Integrations

External system connectors.

**Location:** `integrations/`

| Integration | File | Purpose |
|-------------|------|---------|
| SCADA | `scada_integration.py` | OPC-UA data acquisition |
| Cooling Tower | `cooling_tower_integration.py` | CT coordination |
| DCS | `dcs_integration.py` | Control system interface |
| Historian | `historian_integration.py` | Historical data queries |
| CMMS | `cmms_integration.py` | Work order generation |
| Message Bus | `message_bus_integration.py` | Kafka event streaming |

---

## Data Flow

### Primary Data Flow Diagram

```
+-----------------------------------------------------------------------------+
|                        CONDENSYNC DATA FLOW                                  |
+-----------------------------------------------------------------------------+

  COOLING WATER              VACUUM SYSTEM              CONDENSATE
       |                          |                          |
       v                          v                          v
+-------------+            +-------------+            +-------------+
| CW Inlet T  |            | Vacuum P    |            | Flow Rate   |
| CW Outlet T |            | Air Ejector |            | Temperature |
| Flow Rate   |            | Hotwell     |            | Quality     |
| Pressure    |            | Level       |            |             |
+------+------+            +------+------+            +------+------+
       |                          |                          |
       +----------+---------------+--------------+-----------+
                  |                              |
                  v                              v
         +----------------+             +----------------+
         | Data Validation|             | Data Validation|
         +-------+--------+             +-------+--------+
                 |                              |
                 v                              v
         +----------------+             +----------------+
         | CoolingWater   |             | VacuumData     |
         |    Data        |             |                |
         +-------+--------+             +-------+--------+
                 |                              |
                 +---------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |   PERFORMANCE ANALYSIS      |
                  |-----------------------------|
                  | - Calculate Heat Duty       |
                  | - Calculate U-Value         |
                  | - Calculate LMTD/TTD        |
                  | - Check Compliance          |
                  +-------------+---------------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
   +-----------+         +-----------+         +-----------+
   | Heat      |         | Vacuum    |         | Fouling   |
   | Transfer  |         | Optimize  |         | Predict   |
   | Analysis  |         |           |         |           |
   +-----------+         +-----------+         +-----------+
          |                     |                     |
          +---------------------+---------------------+
                                |
                                v
                  +-----------------------------+
                  |   OPTIMIZATION RESULT       |
                  |-----------------------------|
                  | - Optimal Vacuum SP         |
                  | - Optimal CW Flow           |
                  | - Cleaning Schedule         |
                  | - Expected Savings          |
                  +-------------+---------------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
   +-----------+         +-----------+         +-----------+
   | SCADA     |         | CMMS      |         | Reports   |
   | Setpoints |         | Work      |         | Alerts    |
   |           |         | Orders    |         |           |
   +-----------+         +-----------+         +-----------+
```

### Data Model Relationships

```
+----------------------------------------------------------------------+
|                      DATA MODEL RELATIONSHIPS                         |
+----------------------------------------------------------------------+

  CondenserPerformanceResult
         |
         +-- CoolingWaterData
         |       +-- timestamp: datetime
         |       +-- inlet_temp_f: float
         |       +-- outlet_temp_f: float
         |       +-- flow_rate_gpm: float
         |       +-- pressure_drop_psi: float
         |       +-- water_quality: Optional
         |
         +-- VacuumData
         |       +-- timestamp: datetime
         |       +-- pressure_in_hg_abs: float
         |       +-- saturation_temp_f: float
         |       +-- air_inleakage_scfm: float
         |       +-- hotwell_level_pct: float
         |
         +-- CondensateData
         |       +-- timestamp: datetime
         |       +-- flow_rate_lb_hr: float
         |       +-- temperature_f: float
         |       +-- subcooling_f: float
         |       +-- dissolved_oxygen_ppb: float
         |
         +-- HeatTransferData
         |       +-- u_value_actual: float
         |       +-- u_value_design: float
         |       +-- lmtd_f: float
         |       +-- ttd_f: float
         |       +-- cleanliness_factor_pct: float
         |       +-- fouling_factor: float
         |
         +-- FoulingAssessment
         |       +-- fouling_factor: float
         |       +-- cleanliness_trend: str
         |       +-- cleaning_due: bool
         |       +-- days_until_cleaning: float
         |       +-- recommended_method: CleaningMethod
         |
         +-- OptimizationResult
                 +-- optimal_vacuum: float
                 +-- optimal_cw_flow: float
                 +-- expected_savings_usd: float
                 +-- recommendations: List[str]

+----------------------------------------------------------------------+
```

### Calculation Pipeline

```
+----------------------------------------------------------------------+
|                     CALCULATION PIPELINE                              |
+----------------------------------------------------------------------+

STEP 1: Data Collection
+------------------+     +------------------+     +------------------+
|   SCADA Read     | --> |   Validation     | --> |   Data Model     |
+------------------+     +------------------+     +------------------+

STEP 2: Heat Transfer Analysis
+------------------+     +------------------+     +------------------+
|   Heat Duty      | --> |      LMTD        | --> |    U-Value       |
| Q = mCp*deltaT   |     | (T2-T1)/ln(T2/T1)|     | Q / (A * LMTD)   |
+------------------+     +------------------+     +------------------+

STEP 3: Performance Metrics
+------------------+     +------------------+     +------------------+
|   Cleanliness    | --> |   Fouling        | --> |   Efficiency     |
|   U_act/U_des    |     | 1/U_act - 1/U_cl |     |  Perf Score      |
+------------------+     +------------------+     +------------------+

STEP 4: Optimization
+------------------+     +------------------+     +------------------+
|   Vacuum Opt     | --> |   CW Flow Opt    | --> |   Cleaning Sched |
| f(CW,TTD,Limit)  |     | f(Q,pump,cost)   |     | f(CF,trend,cost) |
+------------------+     +------------------+     +------------------+

STEP 5: Output Generation
+------------------+     +------------------+     +------------------+
|   Setpoints      | --> |   Reports        | --> |   Alerts         |
+------------------+     +------------------+     +------------------+

+----------------------------------------------------------------------+
```

---

## Integration Points

### OPC-UA Integration Architecture

```
+----------------------------------------------------------------------+
|                     OPC-UA INTEGRATION                                |
+----------------------------------------------------------------------+

                    +---------------------------+
                    |      SCADA SERVER         |
                    |       (OPC-UA)            |
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    |     Security Layer        |
                    | - Sign and Encrypt        |
                    | - Basic256Sha256          |
                    | - X.509 Certificates      |
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    |    OPC-UA Client          |
                    | (asyncua library)         |
                    +-------------+-------------+
                                  |
          +-----------------------+------------------------+
          |                       |                        |
          v                       v                        v
   +------------+          +------------+          +------------+
   | Subscribed |          | On-Demand  |          | Historical |
   |   Tags     |          |   Read     |          |   Query    |
   | (5 sec)    |          |            |          |  (OPC-HDA) |
   +------------+          +------------+          +------------+
          |                       |                        |
          v                       v                        v
   +--------------------------------------------------------------+
   |                    TAG PROCESSOR                              |
   |--------------------------------------------------------------|
   | - Condenser vacuum pressure     - CW inlet temperature       |
   | - CW outlet temperature         - CW flow rate               |
   | - Steam flow                    - Condensate flow            |
   | - Hotwell level                 - Air ejector data           |
   +--------------------------------------------------------------+

+----------------------------------------------------------------------+
```

### Modbus TCP Integration

```
+----------------------------------------------------------------------+
|                     MODBUS TCP INTEGRATION                            |
+----------------------------------------------------------------------+

   +------------------+        +------------------+
   | Cooling Tower    |        | CW Pump          |
   |      PLC         |        |    PLC           |
   +--------+---------+        +--------+---------+
            |                           |
            v                           v
   +------------------+        +------------------+
   | Host: CT_PLC_IP  |        | Host: CW_PLC_IP  |
   | Port: 502        |        | Port: 502        |
   | Unit ID: 1       |        | Unit ID: 2       |
   +--------+---------+        +--------+---------+
            |                           |
            +-------------+-------------+
                          |
            +-------------v-------------+
            |    Modbus TCP Client      |
            |   (pymodbus library)      |
            +-------------+-------------+
                          |
   +--------------------------------------------------------------+
   |                  REGISTER MAPPING                             |
   |--------------------------------------------------------------|
   | Address | Name                    | Type    | Scale          |
   |---------|-------------------------|---------|----------------|
   | 40001   | CT Outlet Temp          | Float32 | 0.1            |
   | 40003   | CT Inlet Temp           | Float32 | 0.1            |
   | 40005   | CT Fan Speed            | UInt16  | 1              |
   | 40101   | CW Pump Speed           | UInt16  | 1              |
   | 40103   | CW Pump Power           | Float32 | 0.1            |
   | 40105   | CW Discharge Pressure   | Float32 | 0.01           |
   +--------------------------------------------------------------+

+----------------------------------------------------------------------+
```

### External API Integrations

```
+----------------------------------------------------------------------+
|                    EXTERNAL API INTEGRATIONS                          |
+----------------------------------------------------------------------+

   +------------------+   +------------------+   +------------------+
   |   Weather API    |   |    LIMS API      |   |   Vibration API  |
   +--------+---------+   +--------+---------+   +--------+---------+
            |                      |                      |
            v                      v                      v
   +------------------+   +------------------+   +------------------+
   | GET /weather     |   | OAuth2 Auth      |   | API Key Auth     |
   | X-API-Key        |   | POST /token      |   | X-API-Key        |
   | Polling: 15 min  |   | Polling: 60 min  |   | Polling: 60 sec  |
   +------------------+   +------------------+   +------------------+
            |                      |                      |
            +----------------------+----------------------+
                                   |
                    +--------------v--------------+
                    |      REST Client Pool       |
                    | - Connection pooling        |
                    | - Retry with backoff        |
                    | - Circuit breaker           |
                    +-----------------------------+

+----------------------------------------------------------------------+
```

---

## Deployment Topology

### Kubernetes Deployment Architecture

```
+==============================================================================+
|                    KUBERNETES DEPLOYMENT TOPOLOGY                             |
+==============================================================================+

                           KUBERNETES CLUSTER
+------------------------------------------------------------------------------+
|                                                                               |
|   NAMESPACE: greenlang                                                        |
|   +------------------------------------------------------------------------+ |
|   |                                                                        | |
|   |   DEPLOYMENT: gl-017-condensync (replicas: 3)                          | |
|   |   +------------------+  +------------------+  +------------------+     | |
|   |   |   POD: rep-1     |  |   POD: rep-2     |  |   POD: rep-3     |     | |
|   |   |------------------|  |------------------|  |------------------|     | |
|   |   | condensync:8017  |  | condensync:8017  |  | condensync:8017  |     | |
|   |   | metrics:9017     |  | metrics:9017     |  | metrics:9017     |     | |
|   |   +--------+---------+  +--------+---------+  +--------+---------+     | |
|   |            |                     |                     |               | |
|   |            +---------------------+---------------------+               | |
|   |                                  |                                     | |
|   |                    +-------------v-------------+                       | |
|   |                    |    SERVICE: condensync    |                       | |
|   |                    |    ClusterIP              |                       | |
|   |                    |    Port: 8017, 9017       |                       | |
|   |                    +-------------+-------------+                       | |
|   |                                  |                                     | |
|   +----------------------------------+-------------------------------------+ |
|                                      |                                       |
|   +----------------------------------v-------------------------------------+ |
|   |                         INGRESS                                        | |
|   |   Host: gl-017.greenlang.io                                            | |
|   |   TLS: enabled                                                         | |
|   +------------------------------------------------------------------------+ |
|                                                                               |
|   SUPPORTING SERVICES                                                         |
|   +------------------+  +------------------+  +------------------+            |
|   |  PostgreSQL      |  |    Redis         |  |   InfluxDB       |            |
|   |  StatefulSet     |  |   StatefulSet    |  |   StatefulSet    |            |
|   |  PVC: 100Gi      |  |   PVC: 10Gi      |  |   PVC: 500Gi     |            |
|   +------------------+  +------------------+  +------------------+            |
|                                                                               |
+------------------------------------------------------------------------------+

                              EXTERNAL SERVICES
+------------------------------------------------------------------------------+
|  +--------------+  +--------------+  +--------------+  +--------------+      |
|  |    Kafka     |  |   Vault      |  |    S3        |  |  PagerDuty   |      |
|  |   Cluster    |  |   Secrets    |  |   Reports    |  |   Alerts     |      |
|  +--------------+  +--------------+  +--------------+  +--------------+      |
+------------------------------------------------------------------------------+
```

### Pod Structure

```
+----------------------------------------------------------------------+
|                          POD STRUCTURE                                |
+----------------------------------------------------------------------+

  POD: gl-017-condensync-xxxxx
  +-----------------------------------------------------------------+
  |                                                                  |
  |  INIT CONTAINERS                                                 |
  |  +----------------------------+                                  |
  |  | init-permissions           |                                  |
  |  | - Set directory ownership  |                                  |
  |  | - Configure permissions    |                                  |
  |  +----------------------------+                                  |
  |                                                                  |
  |  MAIN CONTAINER                                                  |
  |  +----------------------------+                                  |
  |  | condensync                 |                                  |
  |  | - Image: greenlang/gl-017  |                                  |
  |  | - Ports: 8017, 9017        |                                  |
  |  | - Resources:               |                                  |
  |  |   - CPU: 250m-1000m        |                                  |
  |  |   - Memory: 512Mi-2Gi      |                                  |
  |  | - Probes:                  |                                  |
  |  |   - Startup: /health/start |                                  |
  |  |   - Liveness: /health/live |                                  |
  |  |   - Readiness: /health/ok  |                                  |
  |  +----------------------------+                                  |
  |                                                                  |
  |  VOLUMES                                                         |
  |  +-------------+  +-------------+  +-------------+               |
  |  | config      |  | data-pvc    |  | tls-certs   |               |
  |  | (ConfigMap) |  | (PVC)       |  | (Secret)    |               |
  |  +-------------+  +-------------+  +-------------+               |
  |                                                                  |
  +-----------------------------------------------------------------+

+----------------------------------------------------------------------+
```

---

## Scalability Design

### Horizontal Scaling

```
+----------------------------------------------------------------------+
|                    HORIZONTAL POD AUTOSCALER                          |
+----------------------------------------------------------------------+

                    HorizontalPodAutoscaler
                    +---------------------------+
                    | Min Replicas: 2           |
                    | Max Replicas: 10          |
                    +-------------+-------------+
                                  |
          +-----------------------+------------------------+
          |                       |                        |
          v                       v                        v
   +------------+          +------------+          +------------+
   | CPU Target |          | Memory     |          | Custom     |
   |    70%     |          | Target 80% |          | Metric     |
   +------------+          +------------+          +------------+
                                                        |
                                                        v
                                            +-----------------------+
                                            | calculations_per_sec  |
                                            |      Target: 100      |
                                            +-----------------------+

  SCALE UP BEHAVIOR                    SCALE DOWN BEHAVIOR
  +------------------------+           +------------------------+
  | Stabilization: 60s     |           | Stabilization: 300s    |
  | Max: 50% increase      |           | Max: 1 pod decrease    |
  | Period: 60s            |           | Period: 120s           |
  +------------------------+           +------------------------+

+----------------------------------------------------------------------+
```

### Data Partitioning Strategy

```
+----------------------------------------------------------------------+
|                    DATA PARTITIONING                                  |
+----------------------------------------------------------------------+

  INFLUXDB TIME SERIES
  +------------------------------------------------------------------+
  | Measurement: condenser_metrics                                    |
  | Tags: condenser_id, unit_id                                       |
  | Retention Policies:                                               |
  |   - Raw data: 90 days                                            |
  |   - 1-hour aggregates: 365 days                                  |
  |   - 1-day aggregates: 7 years                                    |
  +------------------------------------------------------------------+

  POSTGRESQL TABLES
  +------------------------------------------------------------------+
  | Table: condenser_events                                           |
  | Partitioning: RANGE by timestamp (monthly)                        |
  | Indexes: condenser_id, event_type, timestamp                      |
  +------------------------------------------------------------------+
  | Table: optimization_results                                       |
  | Partitioning: RANGE by timestamp (monthly)                        |
  | Indexes: condenser_id, timestamp                                  |
  +------------------------------------------------------------------+

  KAFKA TOPICS
  +------------------------------------------------------------------+
  | Topic: gl017_condenser_events                                     |
  | Partitions: 12                                                    |
  | Replication Factor: 3                                            |
  | Partition Key: condenser_id                                       |
  +------------------------------------------------------------------+

+----------------------------------------------------------------------+
```

---

## High Availability

### HA Architecture

```
+==============================================================================+
|                    HIGH AVAILABILITY ARCHITECTURE                             |
+==============================================================================+

                           AVAILABILITY ZONE A
  +------------------------------------------------------------------------+
  |                                                                        |
  |  +------------------+     +------------------+     +------------------+ |
  |  |   POD: rep-1     |     | PostgreSQL       |     |   Redis          | |
  |  |   (Primary)      |     | Primary          |     |   Primary        | |
  |  +------------------+     +------------------+     +------------------+ |
  |                                                                        |
  +------------------------------------------------------------------------+
                                    |
                                    | Synchronous Replication
                                    |
  +------------------------------------------------------------------------+
  |                         AVAILABILITY ZONE B                            |
  |                                                                        |
  |  +------------------+     +------------------+     +------------------+ |
  |  |   POD: rep-2     |     | PostgreSQL       |     |   Redis          | |
  |  |   (Secondary)    |     | Standby          |     |   Replica        | |
  |  +------------------+     +------------------+     +------------------+ |
  |                                                                        |
  +------------------------------------------------------------------------+
                                    |
                                    | Async Replication
                                    |
  +------------------------------------------------------------------------+
  |                         AVAILABILITY ZONE C                            |
  |                                                                        |
  |  +------------------+     +------------------+     +------------------+ |
  |  |   POD: rep-3     |     | PostgreSQL       |     |   Redis          | |
  |  |   (Secondary)    |     | Read Replica     |     |   Replica        | |
  |  +------------------+     +------------------+     +------------------+ |
  |                                                                        |
  +------------------------------------------------------------------------+

  LOAD BALANCER
  +------------------------------------------------------------------------+
  | Health Checks: /health/ready                                           |
  | Algorithm: Round-robin with sticky sessions                            |
  | Failover: Automatic (30s timeout)                                      |
  +------------------------------------------------------------------------+

+==============================================================================+
```

### Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: gl-017-condensync-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: gl-017-condensync
```

### Failure Recovery Matrix

| Failure Scenario | Detection | Recovery | RTO | RPO |
|-----------------|-----------|----------|-----|-----|
| Single Pod Failure | 30s (liveness) | Auto-restart | 60s | 0s |
| Node Failure | 60s (node watch) | Pod rescheduling | 120s | 0s |
| Database Failure | 10s (readiness) | Failover to standby | 30s | 0s |
| Cache Failure | 5s | Reconnect/rebuild | 10s | 0s |
| SCADA Disconnect | 30s | Retry/alert | 60s | 5min |
| Full AZ Failure | 60s | Cross-AZ failover | 180s | 0s |

### Circuit Breaker Configuration

```
+----------------------------------------------------------------------+
|                    CIRCUIT BREAKER STATES                             |
+----------------------------------------------------------------------+

         CLOSED                  OPEN                   HALF-OPEN
    +-------------+         +-------------+         +-------------+
    |  Normal     |  Fail   |   Block     |  Timer  |   Test      |
    |  Operation  | ------> |   Requests  | ------> |   Traffic   |
    +------+------+   >5    +------+------+   60s   +------+------+
           |                       |                       |
           |                       |                       |
           | Success               | Timer                 | Success
           |                       | Reset                 | (3 calls)
           |                       |                       |
           +<----------------------+<----------------------+
                        Failure Threshold Reset

  CIRCUIT CONFIGURATION
  +------------------------------------------------------------------+
  | Service           | Failure Threshold | Timeout | Recovery       |
  |-------------------|-------------------|---------|----------------|
  | SCADA Connection  |         5         |   30s   |      60s       |
  | DCS Connection    |         5         |   30s   |      60s       |
  | Historian         |         3         |   60s   |     120s       |
  | Database          |         3         |   10s   |      30s       |
  | Cooling Tower PLC |         5         |   10s   |      30s       |
  +------------------------------------------------------------------+

+----------------------------------------------------------------------+
```

---

## Security Architecture

### Defense in Depth

```
+==============================================================================+
|                        SECURITY ARCHITECTURE                                  |
+==============================================================================+

  LAYER 1: NETWORK
  +------------------------------------------------------------------------+
  | - Network Policies (Kubernetes)                                        |
  | - Ingress/Egress filtering                                            |
  | - TLS 1.3 minimum                                                     |
  +------------------------------------------------------------------------+
                                    |
  LAYER 2: AUTHENTICATION
  +------------------------------------------------------------------------+
  | - OAuth2 / OIDC (Keycloak)                                            |
  | - Service account tokens                                              |
  | - mTLS for service-to-service                                         |
  +------------------------------------------------------------------------+
                                    |
  LAYER 3: AUTHORIZATION
  +------------------------------------------------------------------------+
  | - RBAC roles (admin, engineer, operator, viewer)                      |
  | - Permission enforcement                                              |
  | - Setpoint write controls                                             |
  +------------------------------------------------------------------------+
                                    |
  LAYER 4: DATA PROTECTION
  +------------------------------------------------------------------------+
  | - AES-256-GCM encryption at rest                                      |
  | - TLS 1.3 in transit                                                  |
  | - Secrets in HashiCorp Vault                                          |
  +------------------------------------------------------------------------+
                                    |
  LAYER 5: AUDIT
  +------------------------------------------------------------------------+
  | - Full audit logging (7 year retention)                               |
  | - Immutable audit trail                                               |
  | - Provenance hashing                                                  |
  +------------------------------------------------------------------------+

+==============================================================================+
```

---

## Related Documentation

- [README.md](README.md) - Main documentation
- [TOOLS_README.md](TOOLS_README.md) - Tool documentation
- [deployment/README.md](deployment/README.md) - Deployment guide
- [runbooks/](runbooks/) - Operational runbooks

---

*GL-017 CONDENSYNC Architecture Document - Version 1.0.0*
