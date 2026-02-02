# GL-019 HEATSCHEDULER - Architecture Design Summary

**Document Status**: Architecture Design Complete
**Created**: December 3, 2025
**Target Release**: Q1 2026
**Priority**: P1 (High Priority)

---

## Executive Summary

GL-019 HEATSCHEDULER (ProcessHeatingScheduler) architecture design is complete and production-ready for development. The application follows proven GreenLang patterns from GL-018 FLUEFLOW with a 6-agent pipeline optimized for process heating schedule optimization to minimize energy costs.

### Application Overview

| Attribute | Value |
|-----------|-------|
| Agent ID | GL-019 |
| Codename | HEATSCHEDULER |
| Name | ProcessHeatingScheduler |
| Category | Planning |
| Type | Coordinator |
| Description | Schedules process heating operations to minimize energy costs |
| Inputs | Production schedule + energy tariffs + equipment availability |
| Outputs | Optimized heating schedule + cost savings forecast |

### Key Differentiators

1. **Mixed-Integer Linear Programming (MILP)** optimization for globally optimal schedules
2. **Zero-hallucination** energy cost calculations with complete audit trail
3. **Multi-source integration** (SAP, Oracle, Workday, SCADA, utility APIs)
4. **Real-time tariff awareness** for Time-of-Use (ToU) and demand charge optimization
5. **Equipment-aware scheduling** with maintenance windows, capacity, and efficiency curves

---

## Core Components Description

### 1. Schedule Ingestion Layer

The Schedule Ingestion Layer acquires production schedules from enterprise systems and prepares them for optimization.

**Key Capabilities:**
- **ERP Integration**: SAP PP/PM (RFC/BAPI), Oracle SCM (REST), Workday (REST)
- **File Import**: CSV, JSON, XML, Excel (xlsx/xls)
- **Data Validation**: Schema validation (Pydantic), range checks, duplicate detection
- **Provenance**: SHA-256 hashing of all input data for audit trail

**Data Extracted:**
- Production jobs (product, quantity, deadline)
- Heating requirements (kWh, temperature setpoint, duration)
- Priority levels (1-5 scale)
- Constraints (earliest start, latest end, must-run windows)

### 2. Tariff Management Module

The Tariff Management Module maintains current energy pricing and calculates costs for any time period.

**Key Capabilities:**
- **Time-of-Use (ToU) Rates**: Off-peak, mid-peak, on-peak, critical peak
- **Demand Charges**: Monthly peak demand charges ($/kW)
- **Real-Time Pricing**: Optional integration with energy markets
- **Holiday/Weekend Handling**: Different rate schedules by day type

**Tariff Structure:**

| Period | Hours | Energy Rate | Demand Rate | Category |
|--------|-------|-------------|-------------|----------|
| Off-Peak | 00:00-06:00 | $0.045/kWh | $8.00/kW | Lowest cost |
| Mid-Peak | 06:00-10:00 | $0.085/kWh | $12.00/kW | Moderate |
| On-Peak | 10:00-14:00 | $0.125/kWh | $18.00/kW | High cost |
| Critical | 14:00-18:00 | $0.145/kWh | $22.00/kW | Avoid if possible |
| Mid-Peak | 18:00-22:00 | $0.085/kWh | $12.00/kW | Moderate |
| Off-Peak | 22:00-00:00 | $0.045/kWh | $8.00/kW | Lowest cost |

### 3. Equipment Registry

The Equipment Registry tracks all heating equipment, their capabilities, and real-time availability.

**Equipment Data Model:**
- Equipment ID, name, type (gas boiler, electric heater, heat pump)
- Capacity (kW), minimum load (kW)
- Efficiency (% or COP for heat pumps)
- Fuel type (electricity, natural gas, propane)
- Operating constraints (min run time, max starts per day, startup energy)

**Real-Time Status (from SCADA/BMS):**
- Current availability (online, offline, maintenance)
- Current load (kW)
- Efficiency degradation (if applicable)
- Maintenance windows (scheduled and unscheduled)

### 4. Optimization Engine (MILP Solver)

The Optimization Engine is the core of HEATSCHEDULER, using Mixed-Integer Linear Programming to find globally optimal schedules.

**Solver Options:**
- **PuLP** (open source, default)
- **OR-Tools** (Google, high performance)
- **Gurobi** (commercial, fastest, optional)

**Objective Function:**
```
minimize: SUM(energy_cost_i + demand_charge) for all time intervals i
```

**Constraints:**
1. **Production Requirements**: All jobs must be completed by deadline
2. **Equipment Capacity**: Load <= equipment capacity at all times
3. **Equipment Availability**: No scheduling during maintenance windows
4. **Peak Demand Limits**: Respect contractual demand limits
5. **Minimum Run Times**: Equipment must run for minimum duration once started
6. **Maximum Starts**: Limit equipment cycling to reduce wear
7. **Workforce Availability**: Optional shift pattern constraints

**Output:**
- Equipment assignments (which heater for which job)
- Start/end times for each heating operation
- Load profiles (kW per 15-minute interval)
- Peak demand prediction
- Constraint violation warnings (if any soft constraints violated)

### 5. Cost Calculator (Zero-Hallucination)

The Cost Calculator computes energy costs using deterministic formulas only. No LLM is used in any cost calculation.

**Deterministic Formulas:**

```
Energy Cost = SUM(kWh_interval * rate_interval) for all intervals

Demand Charge = MAX(kW_peak) * demand_rate_for_billing_period

Total Cost = Energy Cost + Demand Charge + Fixed Charges

Savings = Baseline Cost - Optimized Cost

Savings % = (Savings / Baseline Cost) * 100

Equipment Energy = Power_kW * Duration_hours * (1 / Efficiency)
```

**Baseline Calculation:**
- Baseline uses naive FIFO scheduling (first-in, first-out)
- No tariff optimization in baseline
- Provides realistic "what-if" comparison

**Output:**
- Total cost (energy + demand + fixed)
- Cost breakdown by tariff period
- Cost breakdown by equipment
- Savings vs. baseline (absolute and percentage)
- Provenance hash for audit trail

### 6. Forecast Engine

The Forecast Engine generates predictive cost savings analysis for planning periods.

**Forecast Horizons:**
- Daily (next 24 hours)
- Weekly (next 7 days)
- Monthly (next 30 days)

**Forecast Components:**
- Expected energy costs based on scheduled production
- Expected savings vs. baseline
- Peak demand prediction
- Equipment utilization forecast

### 7. Integration Hub

The Integration Hub manages all external system connections.

**ERP Connectors:**
| System | Protocol | Authentication | Data |
|--------|----------|----------------|------|
| SAP ERP | RFC/BAPI | SSO/Basic | Production schedules, maintenance |
| Oracle ERP | REST API | OAuth2 | Supply chain, manufacturing |
| Workday | REST API | OAuth2 | Workforce schedules |

**Energy Management:**
| System | Protocol | Authentication | Data |
|--------|----------|----------------|------|
| Utility APIs | REST/SOAP | API Key | Tariff rates, demand charges |
| Energy Markets | WebSocket | API Key | Real-time prices (optional) |

**Equipment Systems:**
| System | Protocol | Authentication | Data |
|--------|----------|----------------|------|
| SCADA | OPC-UA | Certificates | Equipment status |
| BMS | BACnet | Certificates | Building systems |
| Modbus | Modbus TCP | IP-based | Legacy equipment |

---

## Agent Pipeline Architecture

### Pipeline Overview (6 Agents)

```
ScheduleIntakeAgent -> TariffManagementAgent -> EquipmentRegistryAgent
                                    |
                                    v
                      ScheduleOptimizationAgent
                                    |
                                    v
                         CostCalculatorAgent
                                    |
                                    v
                       ForecastReportingAgent
```

### Agent 1: ScheduleIntakeAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Acquire and validate production schedules from ERP/MES systems |
| Inputs | SAP/Oracle/Workday API, CSV, JSON, Excel files |
| Processing | Parse schedules, extract heating requirements, validate data, generate SHA-256 hash |
| Outputs | ProductionScheduleData object (validated, timestamped, hashed) |
| Estimated LOC | 400-500 |
| LLM Usage | None (100% deterministic) |

### Agent 2: TariffManagementAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Manage energy tariffs and calculate rates for any time period |
| Inputs | Utility APIs, tariff files, market data feeds |
| Processing | Parse ToU schedules, calculate demand charges, handle day types |
| Outputs | TariffData object (rates by interval, demand charges) |
| Estimated LOC | 350-450 |
| LLM Usage | None (100% deterministic) |

### Agent 3: EquipmentRegistryAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Track heating equipment availability, capacity, and efficiency |
| Inputs | SCADA/BMS, equipment database, maintenance schedules |
| Processing | Poll equipment status, calculate available capacity, track efficiency |
| Outputs | EquipmentStatus object (availability, capacity, constraints) |
| Estimated LOC | 400-500 |
| LLM Usage | None (100% deterministic) |

### Agent 4: ScheduleOptimizationAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Generate optimal heating schedule using MILP solver |
| Inputs | ProductionScheduleData, TariffData, EquipmentStatus |
| Processing | Build MILP model, solve optimization, handle constraints |
| Outputs | OptimizedSchedule object (assignments, timing, load profiles) |
| Estimated LOC | 800-1000 |
| LLM Usage | None (100% deterministic - MILP solver) |

### Agent 5: CostCalculatorAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Calculate energy costs with zero-hallucination guarantee |
| Inputs | OptimizedSchedule, TariffData |
| Processing | Calculate energy costs, demand charges, savings vs. baseline |
| Outputs | CostAnalysis object (total cost, breakdown, savings) |
| Estimated LOC | 300-400 |
| LLM Usage | None (100% deterministic) |

### Agent 6: ForecastReportingAgent

| Attribute | Value |
|-----------|-------|
| Purpose | Generate reports, forecasts, and notifications |
| Inputs | OptimizedSchedule, CostAnalysis |
| Processing | Generate PDF/Excel, update dashboards, send notifications, sync to ERP |
| Outputs | Reports (PDF, Excel), dashboard data, email/Slack alerts |
| Estimated LOC | 500-600 |
| LLM Usage | Narrative generation ONLY (no calculations) |

### Total Estimated Code

| Component | Lines of Code |
|-----------|---------------|
| Agent 1: ScheduleIntakeAgent | 400-500 |
| Agent 2: TariffManagementAgent | 350-450 |
| Agent 3: EquipmentRegistryAgent | 400-500 |
| Agent 4: ScheduleOptimizationAgent | 800-1000 |
| Agent 5: CostCalculatorAgent | 300-400 |
| Agent 6: ForecastReportingAgent | 500-600 |
| API Layer (FastAPI) | 600-800 |
| Integration Layer | 500-700 |
| Configuration & Utilities | 300-400 |
| **Total** | **4,150-5,350** |

---

## Data Models

### ProductionScheduleData

```python
@dataclass
class ProductionJob:
    job_id: str
    product_id: str
    product_name: str
    quantity: Decimal
    unit: str  # kg, pieces, liters

    heating_requirement_kwh: Decimal
    temperature_setpoint_c: Decimal
    duration_minutes: int

    earliest_start: datetime
    latest_end: datetime  # deadline
    priority: int  # 1-5 (1=highest)

    constraints: Optional[JobConstraints]

@dataclass
class ProductionScheduleData:
    schedule_id: UUID
    tenant_id: UUID
    schedule_date: date

    jobs: List[ProductionJob]

    source_system: str  # SAP, Oracle, CSV, API
    source_reference: str
    provenance_hash: str  # SHA-256
    created_at: datetime
```

### TariffData

```python
@dataclass
class TariffPeriod:
    day_type: str  # weekday, weekend, holiday
    start_time: time
    end_time: time

    energy_rate_kwh: Decimal  # $/kWh
    demand_rate_kw: Decimal  # $/kW
    period_category: str  # off-peak, mid-peak, on-peak, critical

@dataclass
class TariffData:
    tariff_id: UUID
    tenant_id: UUID
    tariff_name: str

    utility_provider: str
    rate_schedule_code: str

    effective_date: date
    expiry_date: Optional[date]

    periods: List[TariffPeriod]

    provenance_hash: str  # SHA-256
```

### EquipmentStatus

```python
@dataclass
class HeatingEquipment:
    equipment_id: str
    equipment_name: str
    equipment_type: str  # gas_boiler, electric, heat_pump

    capacity_kw: Decimal
    min_load_kw: Decimal
    efficiency_percent: Decimal  # or COP for heat pumps

    fuel_type: str  # electricity, natural_gas, propane
    fuel_rate_per_kwh: Decimal

    min_run_time_minutes: int
    max_starts_per_day: int
    startup_energy_kwh: Decimal

@dataclass
class EquipmentAvailability:
    equipment_id: str
    unavailable_start: datetime
    unavailable_end: datetime
    reason: str  # maintenance, repair, offline

@dataclass
class EquipmentStatus:
    equipment: List[HeatingEquipment]
    availability: List[EquipmentAvailability]
    real_time_status: Dict[str, str]  # equipment_id -> status

    provenance_hash: str  # SHA-256
```

### OptimizedSchedule

```python
@dataclass
class ScheduleAssignment:
    assignment_id: UUID
    job_id: str
    equipment_id: str

    scheduled_start: datetime
    scheduled_end: datetime
    power_kw: Decimal

    tariff_period: str  # off-peak, mid-peak, on-peak, critical
    energy_cost_usd: Decimal
    demand_contribution_kw: Decimal

@dataclass
class OptimizedSchedule:
    schedule_id: UUID
    optimization_run_id: UUID

    assignments: List[ScheduleAssignment]

    peak_demand_kw: Decimal
    total_energy_kwh: Decimal

    solver_type: str  # PuLP, OR-Tools, Gurobi
    solver_time_seconds: Decimal
    objective_value: Decimal

    constraints_violated: int
    warnings: List[str]

    provenance_hash: str  # SHA-256
```

### CostAnalysis

```python
@dataclass
class CostBreakdown:
    period_category: str  # off-peak, mid-peak, on-peak, critical
    energy_kwh: Decimal
    energy_cost_usd: Decimal
    percentage_of_total: Decimal

@dataclass
class CostAnalysis:
    analysis_id: UUID
    schedule_id: UUID

    # Optimized costs
    energy_cost_usd: Decimal
    demand_charge_usd: Decimal
    fixed_charges_usd: Decimal
    total_cost_usd: Decimal

    # Baseline comparison
    baseline_cost_usd: Decimal
    savings_usd: Decimal
    savings_percent: Decimal

    # Breakdowns
    cost_by_period: List[CostBreakdown]
    cost_by_equipment: Dict[str, Decimal]

    # Audit
    provenance_hash: str  # SHA-256
    calculation_timestamp: datetime
```

---

## API Design

### Base URL

```
https://api.greenlang.io/v1/heatscheduler
```

### Authentication

All endpoints require OAuth2 + JWT authentication.

```
Authorization: Bearer <jwt_token>
```

### Endpoints

#### Schedule Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/schedules` | Create new schedule from ERP data or file |
| GET | `/schedules` | List schedules for tenant |
| GET | `/schedules/{id}` | Get schedule details |
| DELETE | `/schedules/{id}` | Delete schedule (if not published) |
| POST | `/schedules/{id}/optimize` | Run optimization on schedule |
| POST | `/schedules/{id}/publish` | Publish optimized schedule |

#### Tariff Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tariffs` | Create/update tariff |
| GET | `/tariffs` | List tariffs for tenant |
| GET | `/tariffs/{id}` | Get tariff details |
| GET | `/tariffs/current` | Get current active tariff |
| POST | `/tariffs/sync` | Sync tariffs from utility API |

#### Equipment Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/equipment` | Register new equipment |
| GET | `/equipment` | List equipment for tenant |
| GET | `/equipment/{id}` | Get equipment details |
| PATCH | `/equipment/{id}` | Update equipment |
| POST | `/equipment/{id}/maintenance` | Schedule maintenance window |
| GET | `/equipment/status` | Get real-time equipment status |

#### Optimization

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/optimize` | Run optimization (async) |
| GET | `/optimize/{run_id}` | Get optimization status |
| GET | `/optimize/{run_id}/result` | Get optimization result |
| POST | `/optimize/simulate` | Simulate optimization without saving |

#### Cost Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/costs/{schedule_id}` | Get cost analysis for schedule |
| POST | `/costs/calculate` | Calculate costs for custom schedule |
| GET | `/costs/forecast` | Get cost forecast (daily/weekly/monthly) |

#### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/reports/schedule/{id}/pdf` | Download schedule PDF |
| GET | `/reports/schedule/{id}/excel` | Download schedule Excel |
| GET | `/reports/savings` | Get savings summary report |
| GET | `/reports/equipment-utilization` | Get equipment utilization report |

#### WebSocket

| Endpoint | Description |
|----------|-------------|
| `WS /ws/schedule/{id}` | Real-time schedule updates |
| `WS /ws/optimization/{run_id}` | Optimization progress |

### Example Request/Response

**Create Schedule from CSV:**

```http
POST /schedules
Content-Type: multipart/form-data

file: production_schedule.csv
source_system: CSV
schedule_date: 2025-12-05
```

**Response:**

```json
{
  "schedule_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "tenant-123",
  "schedule_date": "2025-12-05",
  "status": "DRAFT",
  "jobs_count": 15,
  "total_heating_kwh": 4500.00,
  "provenance_hash": "a1b2c3d4...",
  "created_at": "2025-12-03T10:30:00Z"
}
```

**Run Optimization:**

```http
POST /schedules/550e8400-e29b-41d4-a716-446655440000/optimize
Content-Type: application/json

{
  "solver": "OR-Tools",
  "time_limit_seconds": 300,
  "gap_tolerance": 0.01
}
```

**Response:**

```json
{
  "optimization_run_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "RUNNING",
  "message": "Optimization started. Poll for status or subscribe to WebSocket."
}
```

**Get Optimization Result:**

```http
GET /optimize/660e8400-e29b-41d4-a716-446655440001/result
```

**Response:**

```json
{
  "optimization_run_id": "660e8400-e29b-41d4-a716-446655440001",
  "schedule_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "SUCCESS",
  "solver_time_seconds": 12.34,
  "baseline_cost_usd": 1850.00,
  "optimized_cost_usd": 1425.00,
  "savings_usd": 425.00,
  "savings_percent": 22.97,
  "peak_demand_kw": 450.00,
  "assignments_count": 15,
  "constraints_violated": 0,
  "provenance_hash": "e5f6g7h8..."
}
```

---

## Security Architecture

### Authentication & Authorization

| Component | Approach |
|-----------|----------|
| Authentication | OAuth2 + JWT tokens |
| Token Expiry | 1 hour (configurable) |
| Refresh Tokens | 7 days (configurable) |
| MFA | Optional (enterprise tier) |

### Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| ADMIN | Full access, configuration, user management |
| PLANNER | Create/optimize/publish schedules, view reports |
| OPERATOR | View schedules, acknowledge alerts |
| VIEWER | Read-only dashboard access |

### Data Security

| Layer | Protection |
|-------|------------|
| Transport | TLS 1.3 (HTTPS) |
| At Rest | AES-256 encryption (PostgreSQL TDE) |
| Secrets | HashiCorp Vault / AWS Secrets Manager |
| Audit | SHA-256 provenance hashing |

### API Security

| Control | Configuration |
|---------|---------------|
| Rate Limiting | 100 requests/minute per client |
| Input Validation | Pydantic schemas |
| CORS | Configurable allowed origins |
| Headers | Security headers (HSTS, CSP, X-Frame-Options) |

### Compliance

| Standard | Status |
|----------|--------|
| SOC 2 Type II | Target |
| ISO 27001 | Target |
| GDPR | Compliant (data residency options) |

---

## Scalability Considerations

### Horizontal Scaling

| Component | Scaling Approach |
|-----------|------------------|
| API Pods | Horizontal (3-10 replicas based on load) |
| Optimization Workers | Horizontal (separate pod pool for CPU-intensive solvers) |
| Database | PostgreSQL with read replicas |
| Cache | Redis Cluster (3+ nodes) |
| Queue | RabbitMQ (3-node cluster) |

### Performance Targets

| Metric | Target | Justification |
|--------|--------|---------------|
| API Latency (p95) | < 200 ms | User experience |
| Optimization Time | < 5 min for 100 jobs | Practical planning cycles |
| Throughput | 100 schedules/hour | Enterprise workload |
| Availability | 99.9% | Business-critical application |

### Caching Strategy

| Cache Layer | Purpose | TTL |
|-------------|---------|-----|
| L1 (In-Memory) | Hot data, current schedule | 5 minutes |
| L2 (Redis) | Tariff data, equipment status | 15 minutes |
| L3 (Database) | All persistent data | N/A |

**Cache Hit Target**: 66% cost reduction through caching

### Message Queue (RabbitMQ)

| Queue | Purpose |
|-------|---------|
| `schedule.intake` | New schedule processing |
| `optimization.request` | Optimization job dispatch |
| `optimization.result` | Optimization completion |
| `report.generate` | Report generation requests |
| `notification.send` | Email/Slack notifications |

### Database Optimization

| Strategy | Implementation |
|----------|----------------|
| Indexing | Composite indexes on (tenant_id, schedule_date) |
| Partitioning | Monthly partitions for optimization_runs |
| Connection Pooling | PgBouncer (100 connections) |
| Query Optimization | Prepared statements, EXPLAIN ANALYZE |

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
| Optimization | PuLP | 2.7.0+ |
| Optimization (Alt) | OR-Tools | 9.7+ |

### Database & Cache

| Component | Technology | Version |
|-----------|------------|---------|
| Primary Database | PostgreSQL | 14+ |
| ORM | SQLAlchemy | 2.0.0+ |
| Cache | Redis | 7.0+ |
| Connection Pool | PgBouncer | 1.21+ |

### AI/LLM (Narrative Only)

| Component | Technology | Version |
|-----------|------------|---------|
| LLM Provider | Anthropic Claude | Sonnet 4.5 |
| Usage | Narrative generation ONLY | N/A |
| SDK | anthropic | 0.18.0+ |

### Integrations

| Component | Technology | Version |
|-----------|------------|---------|
| SAP Connector | pyrfc | 3.3+ |
| HTTP Client | httpx | 0.25.0+ |
| OPC-UA | asyncua | 1.0.0+ |
| Excel | openpyxl | 3.1.0+ |
| PDF | reportlab | 4.0.0+ |

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

## Development Timeline: 10 Weeks

### Team Size: 3-4 Engineers

| Role | Count | Focus |
|------|-------|-------|
| Senior Backend Engineer | 2 | Core agents, optimization, API |
| DevOps Engineer | 1 | Infrastructure, deployment, monitoring |
| Integration Engineer | 1 | ERP connectors, SCADA integration |

### Phase 1: Core Agents (3 weeks)

**Week 1:**
- Project setup, CI/CD pipeline
- ScheduleIntakeAgent implementation
- Data models and validation

**Week 2:**
- TariffManagementAgent implementation
- EquipmentRegistryAgent implementation
- Unit tests for all three agents (85%+ coverage)

**Week 3:**
- ScheduleOptimizationAgent (MILP solver)
- CostCalculatorAgent implementation
- Integration tests for pipeline

**Deliverable:** 5 working agents with 85%+ test coverage

### Phase 2: API & Integrations (3 weeks)

**Week 4:**
- FastAPI REST API implementation
- WebSocket support
- Authentication (JWT/OAuth2)

**Week 5:**
- SAP connector implementation
- Oracle connector implementation
- File import (CSV, JSON, Excel)

**Week 6:**
- SCADA/BMS integration (OPC-UA, Modbus)
- Utility tariff API integration
- ForecastReportingAgent completion

**Deliverable:** Full integration stack operational

### Phase 3: Testing & Security (2 weeks)

**Week 7:**
- Integration testing (all components)
- Performance testing (solver benchmarks)
- Load testing (100 schedules/hour target)

**Week 8:**
- Security testing (penetration testing)
- Documentation completion
- Security score target: Grade A (92+)

**Deliverable:** Test coverage 85%+, security audit passed

### Phase 4: Deployment (2 weeks)

**Week 9:**
- Docker containerization
- Kubernetes manifests
- Terraform IaC for cloud deployment

**Week 10:**
- Production deployment
- Monitoring setup (Prometheus/Grafana)
- Beta customer onboarding
- Go-live support

**Deliverable:** Production-ready system

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MILP solver performance | Medium | High | Use OR-Tools/Gurobi for large problems, time limits, presolve |
| ERP connectivity issues | High | High | Mock ERP for testing, CSV fallback, early firewall coordination |
| Tariff data accuracy | Medium | Medium | Validate against utility bills, manual override option |
| Equipment integration | Medium | Medium | Support multiple protocols, manual entry fallback |
| Multi-tenant data isolation | Low | Critical | Row-level security, tenant_id on all queries, penetration testing |

---

## Success Criteria

The architecture will be considered successful if:

1. **Zero-Hallucination Compliance**: All cost calculations use deterministic formulas with SHA-256 provenance
2. **Optimization Quality**: Consistent savings of 10-25% vs. baseline scheduling
3. **Performance Targets Met**: <5 min optimization for 100 jobs, <200ms API latency
4. **Test Coverage**: 85%+ unit test coverage, 95%+ pass rate
5. **Security Score**: Grade A (92+/100)
6. **ERP Integration**: Successful connection to SAP and Oracle
7. **Deployment Ready**: Docker + Kubernetes deployment functional
8. **Timeline Met**: 10-week development timeline achieved

---

## Business Value

### Cost Savings Model

For a manufacturing facility with:
- 10 process heaters (total 2,000 kW capacity)
- 8,000 operating hours/year
- Average load factor: 60%
- Energy rate: $0.08/kWh average

**Annual Energy Cost (Baseline):** $768,000

| Optimization | Savings % | Annual Savings |
|--------------|-----------|----------------|
| ToU Shifting | 8-15% | $61,000 - $115,000 |
| Peak Shaving | 5-10% | $38,000 - $77,000 |
| Equipment Optimization | 2-5% | $15,000 - $38,000 |
| **Total** | **15-30%** | **$115,000 - $230,000** |

**Typical ROI:** 3-6 months payback period

---

## Appendix A: Optimization Formulas

### Objective Function (MILP)

```
minimize:
  SUM_i(SUM_j(SUM_t(
    x[i,j,t] * power[j] * duration * rate[t]
  ))) +
  peak_demand * demand_rate

where:
  i = job index
  j = equipment index
  t = time interval index
  x[i,j,t] = 1 if job i assigned to equipment j at time t, 0 otherwise
  power[j] = power consumption of equipment j (kW)
  rate[t] = energy rate at time t ($/kWh)
  peak_demand = MAX(SUM_j(x[*,j,t] * power[j])) over all t
  demand_rate = demand charge ($/kW)
```

### Constraints

```
1. Each job assigned exactly once:
   SUM_j(SUM_t(x[i,j,t])) = 1  for all i

2. Equipment capacity:
   SUM_i(x[i,j,t] * power[j]) <= capacity[j]  for all j, t

3. Equipment availability:
   x[i,j,t] = 0  if equipment j unavailable at time t

4. Job deadlines:
   end_time[i,j,t] <= deadline[i]  for all assignments

5. Minimum run time:
   If x[i,j,t] = 1, then x[i,j,t+1] = 1 for min_run_time intervals
```

---

## Appendix B: API Rate Limits

| Endpoint Category | Rate Limit | Burst |
|-------------------|------------|-------|
| Schedule CRUD | 100/min | 150 |
| Optimization | 10/min | 15 |
| Reports | 30/min | 50 |
| Equipment Status | 200/min | 300 |
| WebSocket | 5 connections | N/A |

---

## Appendix C: Prometheus Metrics

```
# Schedule metrics
heatscheduler_schedules_total{status="draft|optimized|published"}
heatscheduler_jobs_total{status="pending|assigned|completed",priority="1-5"}

# Optimization metrics
heatscheduler_optimization_duration_seconds{solver="PuLP|OR-Tools|Gurobi"}
heatscheduler_optimization_gap_percent
heatscheduler_constraints_violated_total

# Cost metrics
heatscheduler_cost_savings_usd_total
heatscheduler_savings_percent
heatscheduler_energy_cost_usd{period="off-peak|mid-peak|on-peak|critical"}
heatscheduler_demand_charge_usd

# Equipment metrics
heatscheduler_equipment_utilization{equipment_id}
heatscheduler_equipment_available{equipment_id}

# Integration metrics
heatscheduler_erp_sync_duration_seconds{system="SAP|Oracle|Workday"}
heatscheduler_tariff_updates_total
```

---

**Document Version**: 1.0.0
**Last Updated**: December 3, 2025
**Maintained By**: GreenLang Architecture Team (GL-AppArchitect)
