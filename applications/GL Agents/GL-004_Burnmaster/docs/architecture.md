# GL-004 BURNMASTER Architecture

## System Overview

GL-004 BURNMASTER is designed as a modular, scalable system for industrial combustion optimization. This document describes the system architecture, component interactions, and design decisions.

## High-Level Architecture

```
+------------------------------------------------------------------+
|                      GL-004 BURNMASTER                            |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+  +-------------------+  +----------------+ |
|  |   API Layer       |  |  Processing Core  |  |  Integrations  | |
|  +-------------------+  +-------------------+  +----------------+ |
|  | - REST API        |  | - Combustion Calc |  | - OPC-UA       | |
|  | - GraphQL API     |  | - Optimization    |  | - Kafka        | |
|  | - gRPC Services   |  | - ML Inference    |  | - DCS/PLC      | |
|  | - WebSocket       |  | - Safety Logic    |  | - CEMS         | |
|  +-------------------+  +-------------------+  +----------------+ |
|           |                     |                     |          |
|           v                     v                     v          |
|  +-----------------------------------------------------------+  |
|  |                    Message Bus (Kafka)                     |  |
|  +-----------------------------------------------------------+  |
|           |                     |                     |          |
|           v                     v                     v          |
|  +-------------------+  +-------------------+  +----------------+ |
|  |   Data Storage    |  |   Monitoring      |  |   Audit        | |
|  +-------------------+  +-------------------+  +----------------+ |
|  | - PostgreSQL      |  | - Prometheus      |  | - Audit Log    | |
|  | - InfluxDB        |  | - Grafana         |  | - Provenance   | |
|  | - Redis Cache     |  | - OpenTelemetry   |  | - Evidence     | |
|  +-------------------+  +-------------------+  +----------------+ |
|                                                                   |
+------------------------------------------------------------------+
```

## Component Architecture

### 1. Core Processing Layer

#### Combustion Module (`combustion/`)
Physics-based combustion calculations:

```
combustion/
+-- __init__.py
+-- stoichiometry.py       # Stoichiometric ratio calculations
+-- excess_air.py          # Excess air and lambda calculations
+-- efficiency.py          # Combustion efficiency calculations
+-- flame_temperature.py   # Adiabatic flame temperature
+-- flue_gas_losses.py     # Stack loss calculations
+-- fuel_properties.py     # Fuel composition and heating values
```

**Design Principles:**
- All calculations use Decimal for precision
- Physics equations validated against reference sources
- No LLM involvement in numeric calculations

#### Calculators Module (`calculators/`)
High-level calculation engines:

```
calculators/
+-- __init__.py
+-- air_fuel_ratio_calculator.py    # A/F ratio computations
+-- emissions_calculator.py          # NOx/CO predictions
+-- flame_stability_calculator.py    # Flame health metrics
+-- turndown_calculator.py           # Turndown optimization
+-- combustion_kpi_calculator.py     # KPI aggregations
```

#### Optimization Module (`optimization/`)
Optimization engines for setpoint recommendations:

```
optimization/
+-- __init__.py
+-- air_fuel_optimizer.py       # O2 trim optimization
+-- nox_reduction_optimizer.py  # NOx minimization
+-- turndown_optimizer.py       # Load optimization
+-- burner_tuning_optimizer.py  # Burner performance tuning
+-- recommendation_engine.py    # Unified recommendation generation
```

**Optimization Strategy:**
1. Multi-objective optimization (efficiency vs emissions)
2. Constraint satisfaction (safety limits)
3. Pareto-optimal solutions with trade-off analysis

### 2. Control Layer

#### Control Module (`control/`)
Process control logic for closed-loop operation:

```
control/
+-- __init__.py
+-- air_fuel_controller.py     # A/F ratio control
+-- o2_trim_controller.py      # O2 trim control
+-- flame_stability_controller.py
+-- damper_position_controller.py
+-- mode_manager.py            # Advisory vs closed-loop modes
```

**Control Modes:**
- **Advisory Mode:** Provides recommendations for operator action
- **Supervisory Mode:** Adjusts setpoints within DCS limits
- **Closed-Loop Mode:** Direct control with safety interlocks

### 3. Safety Layer

#### Safety Module (`safety/`)
Safety envelope enforcement:

```
safety/
+-- __init__.py
+-- combustion_safety_envelope.py  # Operating limits
+-- flameout_protection.py         # Flame failure handling
+-- emissions_limit_monitor.py     # Regulatory compliance
+-- interlock_manager.py           # DCS interlock coordination
```

**Safety Principles:**
- Defense in depth with multiple layers
- Fail-safe defaults
- Human oversight for critical changes
- Full auditability

### 4. Explainability Layer

#### Explainability Module (`explainability/`)
Feature attribution and explanation generation:

```
explainability/
+-- __init__.py
+-- combustion_physics_explainer.py  # Physics-based explanations
+-- shap_explainer.py                # SHAP feature attribution
+-- lime_explainer.py                # LIME local explanations
+-- explanation_generator.py         # Natural language generation
```

### 5. Integration Layer

#### Integration Module (`integration/`)
Connectivity to external systems:

```
integration/
+-- __init__.py
+-- opcua_connector.py      # OPC-UA client
+-- dcs_connector.py        # DCS/PLC integration
+-- cems_connector.py       # CEMS data acquisition
+-- historian_connector.py  # Process historian
+-- tag_mapper.py           # Tag name mapping
```

### 6. API Layer

#### API Module (`api/`)
External interfaces:

```
api/
+-- __init__.py
+-- rest_api.py        # FastAPI REST endpoints
+-- graphql_api.py     # Strawberry GraphQL schema
+-- grpc_services.py   # gRPC service definitions
+-- websocket.py       # Real-time streaming
+-- auth.py            # Authentication/authorization
+-- rate_limiter.py    # Request throttling
```

## Data Flow

### Real-Time Processing Flow

```
+----------+    +-----------+    +------------+    +----------+
|  OPC-UA  |--->| Transform |--->| Calculate  |--->| Optimize |
|  Data    |    |  & Valid  |    | Combustion |    | Setpoint |
+----------+    +-----------+    +------------+    +----------+
     |               |                |                 |
     v               v                v                 v
+----------+    +-----------+    +------------+    +----------+
|  Kafka   |    |  InfluxDB |    | PostgreSQL |    |  Kafka   |
|  Raw     |    |  Time     |    | Results    |    |  Reco    |
+----------+    |  Series   |    +------------+    +----------+
                +-----------+
```

### Request-Response Flow

```
Client Request
     |
     v
+-------------------+
|   API Gateway     |
+-------------------+
     |
     v
+-------------------+
|   Authentication  |
+-------------------+
     |
     v
+-------------------+
|   Rate Limiting   |
+-------------------+
     |
     v
+-------------------+
|   Validation      |
+-------------------+
     |
     v
+-------------------+
|   Processing      |
+-------------------+
     |
     v
+-------------------+
|   Response        |
+-------------------+
```

## Data Models

### Core Data Models (Pydantic)

```python
class BurnerProcessData(BaseModel):
    """Real-time burner process data."""
    timestamp: datetime
    burner_id: str
    fuel_flow_rate_kg_s: Decimal
    air_flow_rate_kg_s: Decimal
    o2_percent: Decimal
    co_ppm: Decimal
    nox_ppm: Decimal
    flame_intensity: Decimal
    flame_stability_index: Decimal
    burner_load_percent: Decimal

class CombustionProperties(BaseModel):
    """Calculated combustion properties."""
    stoichiometric_ratio: Decimal
    actual_air_fuel_ratio: Decimal
    excess_air_percent: Decimal
    lambda_value: Decimal
    combustion_efficiency_percent: Decimal
    adiabatic_flame_temp_c: Decimal

class OptimizationRecommendation(BaseModel):
    """Optimization recommendation output."""
    target_o2_percent: Decimal
    expected_efficiency_gain_percent: Decimal
    expected_nox_reduction_percent: Decimal
    confidence_level: Decimal
    provenance_hash: str
    explanation: str
```

## Deployment Architecture

### Kubernetes Deployment

```yaml
# High-level deployment structure
namespace: gl004-burnmaster
deployments:
  - api-server (3 replicas)
  - optimization-engine (2 replicas)
  - data-processor (3 replicas)
  - streaming-consumer (2 replicas)

services:
  - api-service (LoadBalancer)
  - internal-grpc (ClusterIP)

configmaps:
  - app-config
  - fuel-properties

secrets:
  - api-keys
  - certificates
```

### Scaling Strategy

| Component | Scaling Metric | Min | Max |
|-----------|---------------|-----|-----|
| API Server | CPU/Requests | 2 | 10 |
| Optimizer | Queue Length | 1 | 5 |
| Processor | Kafka Lag | 2 | 20 |

## Performance Considerations

### Latency Targets

| Operation | Target | 99th Percentile |
|-----------|--------|-----------------|
| Stoichiometric calc | < 1 ms | < 2 ms |
| Excess air calc | < 0.5 ms | < 1 ms |
| NOx prediction | < 5 ms | < 10 ms |
| Full optimization | < 10 ms | < 25 ms |

### Caching Strategy

- **Redis:** Hot fuel properties, recent calculations
- **In-Memory:** Constant lookup tables, validation rules
- **InfluxDB:** Time-series aggregations

## Security Architecture

### Authentication & Authorization

```
+------------------+
|   API Gateway    |
+------------------+
        |
        v
+------------------+
|   JWT Validation |
+------------------+
        |
        v
+------------------+
|   RBAC Check     |
+------------------+
        |
        v
+------------------+
|   Audit Log      |
+------------------+
```

### Security Controls

- TLS 1.3 for all communications
- JWT tokens with short expiry
- Role-based access control (RBAC)
- Secrets in Kubernetes secrets/Vault
- Network policies for pod isolation

## Monitoring & Observability

### Metrics (Prometheus)

```python
# Key metrics exposed
gl004_combustion_calculations_total
gl004_optimization_recommendations_total
gl004_api_request_duration_seconds
gl004_safety_violations_total
gl004_active_burners
```

### Tracing (OpenTelemetry)

- Distributed tracing for all requests
- Span correlation across services
- Performance bottleneck identification

### Logging (Structured)

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "service": "gl004-optimizer",
  "trace_id": "abc123",
  "message": "Optimization completed",
  "burner_id": "BNR-001",
  "duration_ms": 8.5
}
```

## Disaster Recovery

### Backup Strategy

- PostgreSQL: Daily snapshots, WAL archiving
- InfluxDB: Continuous backup to S3
- Configuration: GitOps with version control

### Recovery Time Objectives

| Component | RTO | RPO |
|-----------|-----|-----|
| API Service | 5 min | 0 |
| Database | 30 min | 5 min |
| ML Models | 1 hour | 24 hours |

## Future Considerations

### Planned Enhancements

1. **Multi-Burner Coordination:** Optimize across burner arrays
2. **Predictive Maintenance:** Burner component degradation prediction
3. **Flame Image Analysis:** CNN-based flame diagnostics
4. **Digital Twin Integration:** High-fidelity simulation coupling

### Consolidation into GL-018

GL-004 BURNMASTER will be consolidated into GL-018 UNIFIEDCOMBUSTION:
- All APIs maintained via compatibility layer
- Enhanced multi-system optimization
- Unified combustion domain model
