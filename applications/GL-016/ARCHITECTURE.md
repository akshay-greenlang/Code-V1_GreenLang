# GL-016 WATERGUARD - Technical Architecture

**Intelligent Boiler Water Treatment Agent - System Design Documentation**

Version: 1.0.0
Last Updated: December 2025
Status: Production Ready

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Design Principles](#system-design-principles)
- [Component Architecture](#component-architecture)
- [Data Flow Architecture](#data-flow-architecture)
- [Integration Architecture](#integration-architecture)
- [Message Bus Patterns](#message-bus-patterns)
- [State Management](#state-management)
- [Determinism Guarantees](#determinism-guarantees)
- [Security Architecture](#security-architecture)
- [Scalability Considerations](#scalability-considerations)
- [Failure Handling](#failure-handling)
- [Performance Optimization](#performance-optimization)

---

## Architecture Overview

GL-016 WATERGUARD employs a **modular, event-driven, microservices-inspired architecture** designed for industrial deployment. The system prioritizes:

- **Deterministic Operation**: All calculations are reproducible and auditable
- **Real-time Processing**: Sub-second response times for critical alerts
- **High Availability**: 99.9% uptime target with graceful degradation
- **Scalability**: Horizontal scaling to support multiple boilers/plants
- **Security**: Defense-in-depth with encryption, authentication, authorization

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                 │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │  Grafana     │  │   REST API   │  │   Web UI     │                │
│  │  Dashboard   │  │   Clients    │  │              │                │
│  └──────────────┘  └──────────────┘  └──────────────┘                │
└───────────────────────────────────────────────────────────────────────┘
                               │
                               │ HTTPS/WSS
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY LAYER                             │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (Port 8000)                               │  │
│  │  - Authentication/Authorization                                │  │
│  │  - Rate Limiting                                               │  │
│  │  - Request Validation                                          │  │
│  │  - WebSocket Subscriptions                                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                               │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  BoilerWaterTreatmentAgent (Main Orchestrator)                 │  │
│  │  - Workflow Coordination                                       │  │
│  │  - Event Handling                                              │  │
│  │  - State Management                                            │  │
│  │  - Error Recovery                                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  CALCULATION     │ │  INTEGRATION     │ │  MONITORING      │
│  ENGINE          │ │  LAYER           │ │  LAYER           │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Water Chem   │     │ SCADA Client │     │ Prometheus   │
│ Calculator   │     │ (OPC-UA)     │     │ Exporter     │
├──────────────┤     ├──────────────┤     ├──────────────┤
│ Blowdown     │     │ Modbus Client│     │ Grafana      │
│ Optimizer    │     ├──────────────┤     │ Dashboard    │
├──────────────┤     │ ERP          │     ├──────────────┤
│ Chemical     │     │ Connector    │     │ Alert        │
│ Dosing       │     ├──────────────┤     │ Manager      │
├──────────────┤     │ Agent        │     └──────────────┘
│ Risk         │     │ Coordinator  │
│ Assessment   │     └──────────────┘
└──────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│           DATA LAYER                      │
│                                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Config  │  │ Time    │  │  Cache  │  │
│  │ Store   │  │ Series  │  │ (Redis) │  │
│  │ (YAML)  │  │ DB      │  │         │  │
│  └─────────┘  └─────────┘  └─────────┘  │
└──────────────────────────────────────────┘
```

---

## System Design Principles

### 1. Deterministic Computing

**Principle**: All water chemistry calculations must be 100% reproducible and auditable.

**Implementation**:
- Fixed-point arithmetic for financial calculations
- Deterministic random seeds for any stochastic processes
- SHA-256 provenance hashing for audit trail
- Version pinning of all dependencies
- Explicit floating-point rounding strategies

**Zero Hallucination Policy**:
```python
# LLMs PROHIBITED for these operations:
- Numeric calculations
- Water chemistry formulas
- Blowdown rate calculations
- Chemical dosing calculations
- Risk score calculations

# LLMs ALLOWED for these operations:
- Narrative report generation
- Recommendation explanations
- Incident descriptions
- Best practice guidance
```

### 2. Event-Driven Architecture

**Benefits**:
- Loose coupling between components
- Asynchronous processing
- Easy scaling
- Resilience to component failures

**Event Types**:
```python
class EventType(Enum):
    WATER_CHEMISTRY_UPDATED = "water_chemistry_updated"
    ALARM_TRIGGERED = "alarm_triggered"
    ALARM_CLEARED = "alarm_cleared"
    BLOWDOWN_OPTIMIZED = "blowdown_optimized"
    CHEMICAL_DOSING_ADJUSTED = "chemical_dosing_adjusted"
    RISK_ASSESSMENT_COMPLETED = "risk_assessment_completed"
    SCADA_CONNECTION_LOST = "scada_connection_lost"
    SCADA_CONNECTION_RESTORED = "scada_connection_restored"
```

### 3. Defense in Depth Security

**Layers**:
1. Network Security (Firewall, VPN)
2. Transport Security (TLS 1.3)
3. Application Security (OAuth2, JWT)
4. Data Security (AES-256 encryption at rest)
5. Audit Logging (Immutable logs)

### 4. Fault Tolerance

**Strategies**:
- Circuit breakers for external integrations
- Exponential backoff for retries
- Graceful degradation when sensors fail
- Last-known-good value caching
- Health monitoring and auto-recovery

---

## Component Architecture

### 1. API Gateway (FastAPI)

**Responsibilities**:
- HTTP/HTTPS request handling
- WebSocket connections for real-time data
- Authentication and authorization
- Rate limiting
- Request/response validation
- CORS handling

**Technology Stack**:
- FastAPI 0.104+
- Uvicorn ASGI server
- Pydantic for validation
- JWT for authentication

**Endpoints**:
```
POST   /api/v1/water-chemistry          Submit water chemistry data
POST   /api/v1/optimize-blowdown        Get blowdown optimization
POST   /api/v1/optimize-chemicals       Get chemical dosing recommendations
POST   /api/v1/assess-scale-risk        Assess scale formation risk
POST   /api/v1/assess-corrosion-risk    Assess corrosion risk
GET    /api/v1/performance-report/:id   Get performance report
GET    /api/v1/alarms/active            Get active alarms
POST   /api/v1/alarms/:id/acknowledge   Acknowledge alarm
GET    /api/v1/scada/status             Get SCADA connection status
WS     /ws/realtime                     WebSocket for real-time updates
GET    /health                          Health check
GET    /metrics                         Prometheus metrics
```

### 2. Main Orchestrator

**Class**: `BoilerWaterTreatmentAgent`

**Responsibilities**:
- Coordinate all water treatment operations
- Manage agent lifecycle
- Route requests to appropriate calculators
- Aggregate results from multiple tools
- Emit events for monitoring
- Handle errors and retries

**State Machine**:
```
┌─────────────┐
│ INITIALIZING│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   READY     │◄──────┐
└──────┬──────┘       │
       │              │
       ▼              │
┌─────────────┐       │
│ PROCESSING  │───────┘
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   PAUSED    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   STOPPED   │
└─────────────┘
```

### 3. Calculation Engine

**Modules**:

#### Water Chemistry Calculator
```python
class WaterChemistryCalculator:
    """
    Deterministic water chemistry calculations.
    """

    def calculate_cycles_of_concentration(
        self,
        boiler_conductivity: float,
        makeup_conductivity: float
    ) -> float:
        """
        Calculate COC from conductivity measurements.
        Formula: COC = Boiler_Conductivity / Makeup_Conductivity
        """
        if makeup_conductivity <= 0:
            raise ValueError("Makeup conductivity must be > 0")

        coc = boiler_conductivity / makeup_conductivity
        return round(coc, 2)

    def calculate_langelier_saturation_index(
        self,
        ph: float,
        calcium_ppm: float,
        alkalinity_ppm: float,
        tds_ppm: float,
        temperature_c: float
    ) -> float:
        """
        Calculate LSI (Langelier Saturation Index).
        LSI = pH - pHs

        Where pHs (pH of saturation) = (9.3 + A + B) - (C + D)
        A = (log10[TDS] - 1) / 10
        B = -13.12 * log10(T + 273) + 34.55
        C = log10[Ca²⁺ as CaCO3] - 0.4
        D = log10[Alkalinity as CaCO3]

        LSI > 0: Scale-forming (supersaturated)
        LSI = 0: Balanced
        LSI < 0: Corrosive (undersaturated)
        """
        import math

        # Calculate pHs components
        A = (math.log10(tds_ppm) - 1) / 10
        B = -13.12 * math.log10(temperature_c + 273) + 34.55
        C = math.log10(calcium_ppm) - 0.4
        D = math.log10(alkalinity_ppm)

        pHs = (9.3 + A + B) - (C + D)
        lsi = ph - pHs

        return round(lsi, 2)
```

#### Blowdown Optimizer
```python
class BlowdownOptimizer:
    """
    Optimize blowdown rate to minimize water and energy waste
    while maintaining water quality.
    """

    def optimize_blowdown_rate(
        self,
        current_coc: float,
        max_allowable_tds: float,
        makeup_tds: float,
        steam_flow_kg_hr: float,
        feedwater_enthalpy_kj_kg: float,
        blowdown_enthalpy_kj_kg: float,
        fuel_cost_per_mj: float,
        water_cost_per_m3: float
    ) -> BlowdownOptimizationResult:
        """
        Multi-objective optimization:
        - Maximize COC (minimize water consumption)
        - Respect TDS limits
        - Minimize total cost (water + energy)
        """
        # Maximum COC based on TDS limit
        max_coc = max_allowable_tds / makeup_tds

        # Optimal COC (balance water and energy costs)
        # Higher COC = less water, but more concentrated blowdown heat loss
        optimal_coc = self._find_optimal_coc(
            max_coc=max_coc,
            steam_flow=steam_flow_kg_hr,
            fuel_cost=fuel_cost_per_mj,
            water_cost=water_cost_per_m3
        )

        # Calculate blowdown rate
        blowdown_percent = 100 / (optimal_coc - 1)
        blowdown_rate_kg_hr = steam_flow_kg_hr * blowdown_percent / 100

        # Energy loss from blowdown
        energy_loss_kw = (
            blowdown_rate_kg_hr *
            (blowdown_enthalpy_kj_kg - feedwater_enthalpy_kj_kg) / 3600
        )

        return BlowdownOptimizationResult(
            optimal_coc=optimal_coc,
            optimal_blowdown_rate_kg_hr=blowdown_rate_kg_hr,
            optimal_blowdown_percent=blowdown_percent,
            energy_loss_kw=energy_loss_kw
        )
```

#### Chemical Dosing Calculator
```python
class ChemicalDosingCalculator:
    """
    Calculate optimal chemical dosing rates.
    """

    def calculate_oxygen_scavenger_dosage(
        self,
        dissolved_oxygen_ppb: float,
        feedwater_flow_kg_hr: float,
        scavenger_type: str,
        safety_factor: float = 1.5
    ) -> float:
        """
        Calculate oxygen scavenger dosage.

        For sodium sulfite (Na₂SO₃):
        Na₂SO₃ + ½O₂ → Na₂SO₄
        Theoretical: 7.88 kg sulfite per kg O₂

        For hydrazine (N₂H₄):
        N₂H₄ + O₂ → N₂ + 2H₂O
        Theoretical: 1 kg hydrazine per kg O₂
        """
        # Convert ppb to kg/hr
        oxygen_kg_hr = (
            dissolved_oxygen_ppb * feedwater_flow_kg_hr / 1e9
        )

        # Stoichiometric ratio
        if scavenger_type == "SULFITE":
            theoretical_ratio = 7.88
        elif scavenger_type == "HYDRAZINE":
            theoretical_ratio = 1.0
        else:
            theoretical_ratio = 8.0  # Conservative default

        # Apply safety factor for residual
        dosage_kg_hr = (
            oxygen_kg_hr * theoretical_ratio * safety_factor
        )

        return round(dosage_kg_hr, 3)
```

### 4. Integration Layer

#### SCADA Client

**Architecture**:
```
┌─────────────────────────────────────────┐
│         SCADA Client                    │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Connection Manager               │  │
│  │  - Heartbeat monitoring           │  │
│  │  - Auto-reconnect                 │  │
│  │  - Connection pooling             │  │
│  └───────────────────────────────────┘  │
│                 │                       │
│  ┌──────────────┼──────────────┐        │
│  │              │              │        │
│  ▼              ▼              ▼        │
│ ┌────────┐  ┌────────┐  ┌────────┐     │
│ │OPC-UA  │  │Modbus  │  │Profinet│     │
│ │Protocol│  │Protocol│  │Protocol│     │
│ └────────┘  └────────┘  └────────┘     │
│      │            │            │        │
│      └────────────┼────────────┘        │
│                   │                     │
│  ┌────────────────▼──────────────────┐  │
│  │  Tag Registry & Subscription Mgr  │  │
│  │  - Tag metadata                   │  │
│  │  - Value caching                  │  │
│  │  - Change notifications           │  │
│  └───────────────────────────────────┘  │
│                   │                     │
│  ┌────────────────▼──────────────────┐  │
│  │  Alarm Manager                    │  │
│  │  - Limit checking                 │  │
│  │  - Alarm generation               │  │
│  │  - Alarm history                  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Key Features**:
- Async I/O for non-blocking operations
- Automatic reconnection with exponential backoff
- Tag value caching with TTL
- Batch read/write operations
- Historical data buffer
- Quality checking (GOOD/BAD/UNCERTAIN)

### 5. Monitoring Layer

**Prometheus Metrics**:
```python
# Gauge metrics (current values)
waterguard_water_quality_score = Gauge(
    'waterguard_water_quality_score',
    'Current water quality score (0-100)',
    ['boiler_id']
)

waterguard_cycles_of_concentration = Gauge(
    'waterguard_cycles_of_concentration',
    'Current cycles of concentration',
    ['boiler_id']
)

# Counter metrics (cumulative)
waterguard_optimizations_total = Counter(
    'waterguard_optimizations_total',
    'Total number of optimizations performed',
    ['boiler_id', 'optimization_type']
)

waterguard_alerts_total = Counter(
    'waterguard_alerts_total',
    'Total alerts generated',
    ['boiler_id', 'severity', 'alert_type']
)

# Histogram metrics (distributions)
waterguard_calculation_duration_seconds = Histogram(
    'waterguard_calculation_duration_seconds',
    'Time spent on calculations',
    ['calculation_type']
)
```

---

## Data Flow Architecture

### Real-time Data Flow

```
┌──────────────┐
│ SCADA System │
│ (OPC-UA)     │
└──────┬───────┘
       │ Poll every 5s
       ▼
┌──────────────────────┐
│ SCADA Client         │
│ - Read tag values    │
│ - Scale to eng units │
│ - Check quality      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Tag Value Cache      │
│ (TTL: 1 second)      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Event Bus            │
│ - Publish event      │
│ - Route to handlers  │
└──────┬───────────────┘
       │
       ├───────────────────┐
       │                   │
       ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│ Limit Checker   │  │ Data Aggregator │
│ - Compare limits│  │ - Collect values│
│ - Generate alarm│  │ - Build dataset │
└─────────────────┘  └────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Calculation      │
                    │ Engine           │
                    │ - Water chem     │
                    │ - Blowdown opt   │
                    │ - Risk assess    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Result Store     │
                    │ - Time series DB │
                    │ - Cache          │
                    └────────┬─────────┘
                             │
                  ┌──────────┼──────────┐
                  │          │          │
                  ▼          ▼          ▼
          ┌──────────┐ ┌────────┐ ┌────────┐
          │ Grafana  │ │ API    │ │WebSocket│
          │ Dashboard│ │Response│ │Clients  │
          └──────────┘ └────────┘ └────────┘
```

### Batch Processing Flow

For historical analysis and reporting:

```
┌────────────────────┐
│ Historical Data    │
│ Request            │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Time Series DB     │
│ Query              │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Data Processor     │
│ - Resample         │
│ - Aggregate        │
│ - Filter           │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Analytics Engine   │
│ - Trend analysis   │
│ - Anomaly detect   │
│ - Statistics       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Report Generator   │
│ - Format results   │
│ - Generate charts  │
│ - Create PDF       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Response           │
└────────────────────┘
```

---

## Integration Architecture

### SCADA/DCS Integration Patterns

#### Pattern 1: Poll-Based (Default)

```
Agent              SCADA
  │                  │
  │───Read Tags──────>│
  │<──Tag Values──────│
  │                  │
  ├─(Cache values)   │
  │                  │
  │ (5 sec delay)    │
  │                  │
  │───Read Tags──────>│
  │<──Tag Values──────│
  │                  │
```

**Pros**: Simple, reliable, predictable bandwidth
**Cons**: Higher latency, may miss rapid changes

#### Pattern 2: Subscription-Based

```
Agent              SCADA
  │                  │
  │──Subscribe────────>│
  │<──Ack─────────────│
  │                  │
  │                  │ (value changes)
  │<──Notification────│
  │                  │
  │<──Notification────│ (value changes)
  │                  │
```

**Pros**: Lower latency, event-driven, less bandwidth
**Cons**: More complex, requires reliable connection

### Message Bus Pattern

For agent-to-agent communication:

```
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ GL-016      │        │   Message   │        │  GL-002     │
│ WATERGUARD  │───────>│    Bus      │───────>│  Boiler     │
│             │        │  (Redis)    │        │  Efficiency │
└─────────────┘        └─────────────┘        └─────────────┘
      │                       │                       │
      │                       │                       │
      v                       v                       v
  Publish event         Route message          Subscribe to event

Event: "blowdown_optimized"
Payload: {
  "boiler_id": "BOILER-001",
  "optimal_blowdown_rate": 450,
  "water_savings_m3_yr": 1500,
  "timestamp": "2025-12-02T10:30:00Z"
}
```

**Supported Message Brokers**:
- Redis Pub/Sub
- RabbitMQ
- Apache Kafka (for high-throughput)

---

## State Management

### Agent State

```python
class AgentState(BaseModel):
    """Agent operational state."""

    status: AgentStatus  # INITIALIZING, READY, PROCESSING, PAUSED, ERROR
    started_at: datetime
    last_calculation_at: Optional[datetime]

    # Connection states
    scada_connected: bool
    erp_connected: bool
    message_bus_connected: bool

    # Boiler states
    boilers: Dict[str, BoilerState]

    # Statistics
    total_calculations: int
    total_optimizations: int
    total_alarms: int

    # Health metrics
    health_score: float  # 0-100
    error_count_last_hour: int
```

### Boiler State

```python
class BoilerState(BaseModel):
    """Per-boiler operational state."""

    boiler_id: str

    # Latest measurements
    last_water_chemistry: Optional[WaterChemistryData]
    last_blowdown_data: Optional[BlowdownData]
    last_chemical_dosing: Optional[ChemicalDosingData]

    # Latest calculations
    current_coc: Optional[float]
    scale_risk_score: Optional[float]
    corrosion_risk_score: Optional[float]

    # Operational status
    water_quality_compliant: bool
    active_alarms: List[str]

    # Control state
    auto_dosing_enabled: bool
    last_setpoint_change: Optional[datetime]
```

### State Persistence

```python
# State stored in Redis for fast access
redis_client.hset(
    f"waterguard:state:boiler:{boiler_id}",
    mapping={
        "coc": str(state.current_coc),
        "scale_risk": str(state.scale_risk_score),
        "last_update": state.last_update.isoformat()
    }
)

# State also persisted to TimescaleDB for historical analysis
```

---

## Determinism Guarantees

### Sources of Non-Determinism (Eliminated)

1. **Floating-Point Arithmetic**
   - Solution: Fixed precision rounding (`round(value, N)`)
   - Use Decimal type for financial calculations

2. **Random Number Generation**
   - Solution: Fixed seed (`random.seed(42)`)
   - Seeded RNG for any stochastic processes

3. **Timestamp Variations**
   - Solution: Use provided timestamps, not system time
   - Freeze time in tests

4. **Concurrent Execution**
   - Solution: Sequential processing of calculations
   - Locks on shared state

5. **LLM Non-Determinism**
   - Solution: Zero-hallucination policy
   - LLMs only for narrative, never calculations
   - Temperature = 0.0 for reproducibility

### Provenance Tracking

Every calculation includes provenance hash:

```python
def calculate_with_provenance(
    self,
    inputs: Dict[str, Any],
    formula_name: str
) -> Tuple[float, str]:
    """
    Calculate result with provenance tracking.

    Returns:
        (result, provenance_hash)
    """
    # Serialize inputs deterministically
    input_json = json.dumps(inputs, sort_keys=True)

    # Perform calculation
    result = self._calculate(inputs, formula_name)

    # Generate provenance hash
    provenance_data = {
        "formula": formula_name,
        "inputs": input_json,
        "result": result,
        "timestamp": inputs.get("timestamp"),
        "agent_version": __version__
    }
    provenance_json = json.dumps(provenance_data, sort_keys=True)
    provenance_hash = hashlib.sha256(
        provenance_json.encode()
    ).hexdigest()

    return result, provenance_hash
```

---

## Security Architecture

### Defense-in-Depth Layers

#### Layer 1: Network Security

```
┌─────────────────────────────────────┐
│         FIREWALL                    │
│  - Block all except:                │
│    - 8000 (API)                     │
│    - 9090 (Metrics)                 │
│    - 4840 (OPC-UA)                  │
│  - Rate limiting                    │
│  - DDoS protection                  │
└─────────────────────────────────────┘
```

#### Layer 2: Transport Security

- TLS 1.3 for all HTTP connections
- Certificate-based authentication for OPC-UA
- Mutual TLS (mTLS) for agent-to-agent communication

#### Layer 3: Application Security

**Authentication**:
```python
# OAuth2 with JWT tokens
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm):
    user = authenticate_user(form_data.username, form_data.password)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(hours=1)
    )
    return {"access_token": access_token, "token_type": "bearer"}
```

**Authorization** (RBAC):
```python
class Role(Enum):
    ADMIN = "admin"
    ENGINEER = "water_treatment_engineer"
    OPERATOR = "operator"
    VIEWER = "viewer"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    ADJUST_SETPOINTS = "adjust_setpoints"
    DELETE = "delete"

ROLE_PERMISSIONS = {
    Role.ADMIN: [Permission.READ, Permission.WRITE,
                 Permission.ADJUST_SETPOINTS, Permission.DELETE],
    Role.ENGINEER: [Permission.READ, Permission.WRITE,
                    Permission.ADJUST_SETPOINTS],
    Role.OPERATOR: [Permission.READ, Permission.ADJUST_SETPOINTS],
    Role.VIEWER: [Permission.READ]
}

def require_permission(permission: Permission):
    def decorator(func):
        @wraps(func)
        async def wrapper(current_user: User = Depends(get_current_user)):
            if permission not in ROLE_PERMISSIONS[current_user.role]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(current_user)
        return wrapper
    return decorator
```

#### Layer 4: Data Security

**Encryption at Rest**:
```python
from cryptography.fernet import Fernet

# Encrypt sensitive configuration
cipher = Fernet(encryption_key)
encrypted_password = cipher.encrypt(password.encode())

# Store encrypted
config["scada"]["password"] = encrypted_password
```

**Encryption in Transit**:
- All SCADA communication over secure channels
- OPC-UA security mode: SignAndEncrypt

#### Layer 5: Audit Logging

```python
@app.middleware("http")
async def audit_log_middleware(request: Request, call_next):
    """Log all API requests for audit trail."""

    user = get_current_user_from_token(request)

    audit_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user.username if user else "anonymous",
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }

    # Log to immutable audit store
    audit_logger.info(json.dumps(audit_log))

    response = await call_next(request)
    audit_log["status_code"] = response.status_code

    return response
```

---

## Scalability Considerations

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
│                   (NGINX/HAProxy)                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ WATERGUARD   │ │ WATERGUARD   │ │ WATERGUARD   │
│ Instance 1   │ │ Instance 2   │ │ Instance 3   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  Shared State │
                │  (Redis)      │
                └───────────────┘
```

**Scaling Strategy**:
- Stateless API layer (scales easily)
- Shared state in Redis cluster
- Each instance monitors subset of boilers
- Consistent hashing for boiler assignment

### Performance Optimization

**Caching Strategy**:
```python
# Three-level cache

# L1: In-memory cache (fastest, smallest)
@lru_cache(maxsize=1000)
def get_water_quality_limits(boiler_id: str, pressure: float):
    pass

# L2: Redis cache (fast, shared across instances)
def get_boiler_config(boiler_id: str):
    cached = redis_client.get(f"config:boiler:{boiler_id}")
    if cached:
        return json.loads(cached)

    config = load_from_db(boiler_id)
    redis_client.setex(
        f"config:boiler:{boiler_id}",
        3600,  # 1 hour TTL
        json.dumps(config)
    )
    return config

# L3: Time-series DB (persistent)
def get_historical_data(boiler_id: str, start: datetime, end: datetime):
    return timescaledb.query(...)
```

**Async Processing**:
```python
# Process multiple boilers concurrently
async def monitor_all_boilers():
    boilers = get_all_boilers()

    tasks = [
        monitor_boiler(boiler_id)
        for boiler_id in boilers
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for boiler_id, result in zip(boilers, results):
        if isinstance(result, Exception):
            logger.error(f"Error monitoring {boiler_id}: {result}")
```

---

## Failure Handling

### Circuit Breaker Pattern

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def read_from_scada(tag_name: str):
    """
    Read from SCADA with circuit breaker.

    If 5 consecutive failures, circuit opens for 60 seconds.
    """
    return await scada_client.read_tag(tag_name)
```

### Graceful Degradation

```python
async def get_water_chemistry_data(boiler_id: str):
    """
    Get water chemistry with graceful degradation.
    """
    try:
        # Try real-time SCADA data
        data = await scada_client.read_multiple_tags(water_tags)
        return data
    except SCADAConnectionError:
        logger.warning("SCADA unavailable, using cached data")

        # Fall back to last known good values
        cached_data = redis_client.get(f"last_good_data:{boiler_id}")
        if cached_data:
            return json.loads(cached_data)

        # Last resort: manual entry mode
        raise DataUnavailableError(
            "SCADA unavailable and no cached data. Manual entry required."
        )
```

### Error Recovery

```python
class RetryStrategy:
    """Exponential backoff with jitter."""

    @staticmethod
    async def retry_with_backoff(
        func,
        max_retries=3,
        initial_delay=1.0,
        max_delay=60.0
    ):
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                # Exponential backoff with jitter
                jitter = random.uniform(0, delay * 0.1)
                sleep_time = min(delay + jitter, max_delay)

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {sleep_time:.1f}s"
                )

                await asyncio.sleep(sleep_time)
                delay *= 2
```

---

## Performance Optimization

### Benchmarks

Target performance metrics:

| Operation | Target Latency (p95) | Throughput |
|-----------|---------------------|------------|
| Water chemistry analysis | < 100 ms | 1000 req/s |
| Blowdown optimization | < 200 ms | 500 req/s |
| Scale risk assessment | < 150 ms | 750 req/s |
| SCADA tag read | < 50 ms | 2000 req/s |
| SCADA tag write | < 100 ms | 1000 req/s |

### Optimization Techniques

1. **Connection Pooling**
```python
# Reuse SCADA connections
scada_pool = ConnectionPool(
    max_connections=10,
    timeout=30
)
```

2. **Batch Processing**
```python
# Read multiple tags in single request
tags = ["FW_PH_01", "FW_COND_01", "BW_PH_01", "BW_TDS_01"]
values = await scada_client.read_tags(tags)  # Single network call
```

3. **Lazy Loading**
```python
# Only load data when needed
class BoilerConfig:
    _historical_data = None

    @property
    def historical_data(self):
        if self._historical_data is None:
            self._historical_data = load_historical_data()
        return self._historical_data
```

4. **Database Indexing**
```sql
-- Optimize time-series queries
CREATE INDEX idx_water_chemistry_boiler_time
ON water_chemistry (boiler_id, timestamp DESC);

-- Optimize alarm queries
CREATE INDEX idx_alarms_boiler_severity
ON alarms (boiler_id, severity, activated_at DESC);
```

---

## Deployment Architecture

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-016-waterguard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-016-waterguard
  template:
    metadata:
      labels:
        app: gl-016-waterguard
    spec:
      containers:
      - name: waterguard
        image: greenlang/gl-016-waterguard:1.0.0
        ports:
        - containerPort: 8000
          name: api
        - containerPort: 9090
          name: metrics
        env:
        - name: LOG_LEVEL
          value: INFO
        - name: DETERMINISTIC_MODE
          value: "true"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## Conclusion

GL-016 WATERGUARD's architecture prioritizes:

- **Reliability**: Fault-tolerant, self-healing, graceful degradation
- **Performance**: Sub-second response times, horizontal scaling
- **Security**: Multi-layer defense, encryption, audit trails
- **Determinism**: Reproducible calculations, provenance tracking
- **Maintainability**: Modular design, clear separation of concerns

This architecture enables WATERGUARD to meet the demanding requirements of industrial water treatment control while maintaining the flexibility to adapt to diverse deployment scenarios.

---

**Document Version**: 1.0.0
**Last Updated**: December 2025
**Maintained By**: GreenLang Architecture Team
**Contact**: architecture@greenlang.io
