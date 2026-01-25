# GL-019 HEATSCHEDULER

**Intelligent Process Heating Schedule Optimization Agent**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/greenlang/gl-019-heatscheduler)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Integration Guides](#integration-guides)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

GL-019 HEATSCHEDULER is an AI-powered coordinator agent that optimizes process heating operation schedules to minimize energy costs while meeting all production deadlines. It integrates production planning systems, real-time energy tariffs, and equipment availability to deliver optimized heating schedules with quantified cost savings forecasts.

### Purpose

Industrial process heating operations (furnaces, kilns, dryers, ovens, heat treatment systems) consume 15-25% of total manufacturing energy costs. HEATSCHEDULER addresses key challenges:

- **Peak Energy Costs**: Shift heating operations from peak tariff periods ($0.15-$0.35/kWh) to off-peak periods ($0.05-$0.10/kWh)
- **Demand Charge Reduction**: Coordinate equipment startups to avoid demand spikes that trigger $10-$25/kW monthly charges
- **Production Compliance**: Ensure all production deadlines are met while optimizing energy usage
- **Demand Response Integration**: Automatically participate in utility DR programs for additional revenue
- **Thermal Storage Optimization**: Leverage thermal mass and storage for intelligent load shifting

### Agent Specifications

| Field | Value |
|-------|-------|
| **Agent ID** | GL-019 |
| **Codename** | HEATSCHEDULER |
| **Name** | ProcessHeatingScheduler |
| **Category** | Planning |
| **Type** | Coordinator |
| **Inputs** | Production schedule + Energy tariffs + Equipment availability |
| **Outputs** | Optimized heating schedule + Cost savings forecast |
| **Integration Points** | Production planning systems + Energy management |

### Target Market

- **Total Addressable Market (TAM)**: $7 Billion
- **Industries**: Industrial Manufacturing, Chemical Processing, Oil & Gas Refineries, Food & Beverage, Pulp & Paper
- **Equipment Types**: Furnaces, kilns, dryers, ovens, heat treatment systems, boilers
- **Priority**: P1 (High Priority)
- **Target Release**: Q1 2026

### Business Value

- **Energy Cost Reduction**: 15% average reduction through optimized scheduling
- **Demand Charge Savings**: 25% reduction in peak demand charges
- **Peak Load Reduction**: 30% reduction during peak tariff hours
- **Demand Response Revenue**: $50K-$500K annually per facility
- **Production Compliance**: 99%+ on-time delivery maintained
- **ROI**: Less than 12 months payback period

---

## Key Features

### Production Schedule Integration

- **Real-time Sync**: Automatic import from SAP PP/DS, Oracle SCM, Rockwell FactoryTalk MES
- **Batch Tracking**: Batch ID, product type, start/end times, equipment requirements
- **Dependency Handling**: Respects batch sequencing and equipment dependencies
- **Maintenance Awareness**: Integrates planned maintenance windows from CMMS
- **Multi-Shift Support**: Handles 8/12/24-hour shift patterns

### Energy Tariff Management

- **Time-of-Use (ToU)**: Support for up to 8 periods per day with seasonal variations
- **Demand Charges**: Peak 15/30-minute demand tracking with ratchet clause handling
- **Real-Time Pricing**: Integration with ISO markets (PJM, ERCOT, CAISO, NYISO)
- **Day-Ahead Pricing**: LMP forecasting for next-day optimization
- **Rate Import**: Automatic import from OpenEI or manual configuration

### Schedule Optimization Engine

- **MILP Optimization**: Mixed-Integer Linear Programming for optimal schedules
- **Multi-Objective**: Minimize cost, emissions, and peak demand simultaneously
- **Constraint Satisfaction**: Respects deadlines, equipment capacity, and ramp rates
- **Thermal Storage**: Optimizes charge/discharge of molten salt, hot water, steam accumulators
- **Fast Computation**: Weekly schedules in less than 5 minutes

### Load Shifting Recommendations

- **Flexibility Analysis**: Identifies operations with schedule slack
- **Savings Quantification**: Calculates potential savings for each shift
- **One-Click Accept**: Easy implementation of recommendations
- **Learning System**: Improves suggestions based on acceptance patterns

### Demand Response Integration

- **OpenADR 2.0b**: Certified DR protocol support
- **Automatic Bidding**: Generates curtailment bids protecting production deadlines
- **Performance Tracking**: Real-time monitoring of curtailment vs. commitment
- **Revenue Reporting**: Tracks DR payments by event and cumulative

### Cost Savings Forecasting

- **Baseline Calculation**: Historical scheduling approach baseline
- **Savings Breakdown**: Energy charges, demand charges, DR revenue
- **ROI Tracking**: Payback period and return on investment calculation
- **Variance Analysis**: Actual vs. forecast comparison

---

## Architecture Overview

HEATSCHEDULER employs a modular, event-driven architecture optimized for industrial environments:

```
+------------------------------------------------------------------+
|                      HEATSCHEDULER GL-019                        |
|              Process Heating Schedule Optimization                |
+------------------------------------------------------------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+---------------+      +---------------+      +---------------+
|  Data Layer   |      | Logic Layer   |      | Integration   |
|               |      |               |      |    Layer      |
+---------------+      +---------------+      +---------------+
        |                      |                      |
   +----+----+          +------+------+         +----+----+
   |         |          |             |         |         |
   v         v          v             v         v         v
Config   Models   Calculators     Tools     ERP/MES   SCADA
```

### Core Components

1. **Configuration Layer** (`config.py`)
   - Pydantic models for type-safe configuration
   - Equipment specifications and operating parameters
   - Energy tariff structures (ToU, demand, RTP)
   - Integration settings and credentials

2. **Calculators** (`calculators/`)
   - Energy Cost Calculator: ToU, demand charges, real-time pricing
   - Schedule Optimizer: MILP-based schedule optimization
   - Load Forecaster: Production-based load prediction
   - Savings Calculator: ROI, payback, NPV analysis

3. **Integration Layer** (`integrations/`)
   - ERP Connector: SAP, Oracle, MES integration
   - Tariff Provider: Utility and ISO market integration
   - SCADA Integration: OPC UA equipment connectivity
   - Energy Management: OpenADR demand response

4. **API Layer** (`api/`)
   - FastAPI REST endpoints
   - WebSocket real-time updates
   - Authentication and authorization

### Data Flow

```
Production Schedule (ERP/MES) --> Data Validation -->
  --> Schedule Optimization (MILP) --> Feasibility Check -->
    --> Cost Calculation --> Savings Forecast -->
      --> Schedule Publication (MES) --> Execution Monitoring
```

**Key Principles**:
- **Deterministic Calculations**: All optimization uses MILP/LP solvers (no LLM hallucinations)
- **Zero Hallucination**: AI only used for explanations, never for numeric decisions
- **Provenance Tracking**: SHA-256 hash of all calculations for audit trail
- **Real-time Processing**: Sub-second response times for schedule queries

See [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for detailed technical documentation.

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux, Windows, macOS
- **Memory**: 4 GB RAM minimum (8 GB recommended for large schedules)
- **CPU**: 4 cores minimum
- **Network**: Access to ERP/MES and SCADA networks

### Option 1: Docker Installation (Recommended)

```bash
# Pull the Docker image
docker pull greenlang/gl-019-heatscheduler:latest

# Run the container
docker run -d \
  --name heatscheduler \
  -p 8000:8000 \
  -p 9090:9090 \
  -e ERP_ENDPOINT=https://sap-server:443 \
  -e SCADA_ENDPOINT=opc.tcp://scada-server:4840 \
  -e LOG_LEVEL=INFO \
  -v /path/to/config:/app/config \
  greenlang/gl-019-heatscheduler:latest
```

### Option 2: Python Package Installation

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clone repository
git clone https://github.com/greenlang/gl-019-heatscheduler.git
cd gl-019-heatscheduler

# Install dependencies
pip install -r requirements.txt

# Install agent
pip install -e .
```

### Option 3: From Source

```bash
# Clone repository
git clone https://github.com/greenlang/gl-019-heatscheduler.git
cd gl-019-heatscheduler

# Install dependencies
pip install -r requirements.txt

# Run agent
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check version
python -c "from greenlang.GL_019 import __version__; print(__version__)"

# Health check
curl http://localhost:8000/health
```

---

## Configuration

### Configuration File Structure

HEATSCHEDULER uses a YAML-based configuration system. Create a `config/heatscheduler.yaml` file:

```yaml
agent:
  agent_id: GL-019
  agent_name: HEATSCHEDULER
  version: 1.0.0
  optimization_interval_minutes: 60

# Equipment definitions
heating_equipment:
  - equipment_id: FURNACE-001
    equipment_type: electric_furnace
    capacity_kw: 500
    min_load_kw: 100
    ramp_rate_kw_per_hour: 200
    warm_up_time_min: 45
    cool_down_time_min: 30
    efficiency: 0.92

  - equipment_id: KILN-001
    equipment_type: gas_kiln
    capacity_kw: 1200
    min_load_kw: 300
    ramp_rate_kw_per_hour: 150
    warm_up_time_min: 90
    cool_down_time_min: 60
    efficiency: 0.88

# Thermal storage (optional)
thermal_storage:
  - storage_id: MOLTEN-SALT-001
    capacity_kwh: 5000
    max_charge_rate_kw: 500
    max_discharge_rate_kw: 500
    thermal_loss_pct_hr: 0.5
    charge_efficiency: 0.95
    discharge_efficiency: 0.95
    min_state_pct: 10
    max_state_pct: 95

# Energy tariff configuration
energy_tariffs:
  tariff_type: time_of_use
  utility_name: "Example Electric"
  base_charge_per_month: 500.00

  time_periods:
    on_peak:
      hours: [14, 15, 16, 17, 18, 19]  # 2 PM - 8 PM
      rate_per_kwh: 0.25
      days: [0, 1, 2, 3, 4]  # Monday-Friday

    mid_peak:
      hours: [6, 7, 8, 9, 10, 11, 12, 13, 20, 21]
      rate_per_kwh: 0.15
      days: [0, 1, 2, 3, 4]

    off_peak:
      hours: [0, 1, 2, 3, 4, 5, 22, 23]
      rate_per_kwh: 0.08
      days: [0, 1, 2, 3, 4, 5, 6]  # All days

  demand_charges:
    on_peak_rate_per_kw: 15.00
    off_peak_rate_per_kw: 5.00
    ratchet_pct: 80  # 80% of max demand for 12 months

# Production planning integration
erp_integration:
  system: SAP_PP_DS
  endpoint: https://sap-server.company.com:443
  client: "100"
  auth_type: oauth2
  polling_interval_seconds: 300

# SCADA integration
scada_integration:
  protocol: OPC_UA
  server_address: opc.tcp://scada-server:4840
  polling_interval_seconds: 10
  authentication_required: true
  username: heatscheduler
  enable_ssl: true

# Demand response configuration
demand_response:
  enabled: true
  protocol: OpenADR_2_0b
  vtn_url: https://dr-provider.com/openadr/
  ven_id: FACILITY_001
  program_types:
    - capacity
    - energy
  min_curtailment_kw: 100
  max_curtailment_kw: 2000
  production_protection: true

# Optimization settings
optimization:
  solver: highs  # Options: highs, gurobi, cplex
  max_solve_time_seconds: 300
  optimality_gap: 0.01  # 1% gap tolerance
  horizon_hours: 168  # 1 week
  time_slot_minutes: 60
  objectives:
    primary: minimize_cost
    secondary: minimize_peak_demand
```

### Environment Variables

```bash
# ERP Connection
export ERP_ENDPOINT="https://sap-server:443"
export ERP_CLIENT="100"
export ERP_USERNAME="heatscheduler"
export ERP_PASSWORD="secure_password"

# SCADA Connection
export SCADA_ENDPOINT="opc.tcp://192.168.1.100:4840"
export SCADA_USERNAME="scada_user"
export SCADA_PASSWORD="scada_password"

# ISO Market (for real-time pricing)
export ISO_API_KEY="your_iso_api_key"
export ISO_REGION="PJM"  # PJM, ERCOT, CAISO, NYISO

# Logging
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"

# Determinism
export DETERMINISTIC_MODE="true"
export ZERO_HALLUCINATION="true"
export RANDOM_SEED="42"
```

---

## Usage

### Quick Start

```python
from greenlang.GL_019 import ProcessHeatingScheduler
from greenlang.GL_019.config import SchedulerConfig

# Initialize scheduler
scheduler = ProcessHeatingScheduler(config_path="config/heatscheduler.yaml")

# Generate optimized schedule for next week
schedule = await scheduler.optimize_schedule(
    horizon_days=7,
    production_source="erp",  # Auto-fetch from ERP
    optimize_for="cost"
)

print(f"Total Energy Cost: ${schedule.total_cost:,.2f}")
print(f"Baseline Cost: ${schedule.baseline_cost:,.2f}")
print(f"Savings: ${schedule.savings:,.2f} ({schedule.savings_pct:.1f}%)")
print(f"Peak Demand: {schedule.peak_demand_kw:.0f} kW")
```

### Basic Operations

#### Generate Optimized Schedule

```python
from greenlang.GL_019 import ProcessHeatingScheduler
from greenlang.GL_019.calculators import ScheduleOptimizer

# Initialize
scheduler = ProcessHeatingScheduler(config_path="config/heatscheduler.yaml")

# Define production jobs
from greenlang.GL_019.calculators.schedule_optimizer import HeatingJob

jobs = [
    HeatingJob(
        job_id="BATCH-001",
        energy_required_kwh=500,
        min_power_kw=50,
        max_power_kw=100,
        deadline_hour=12,
        earliest_start_hour=0,
        priority=1.0
    ),
    HeatingJob(
        job_id="BATCH-002",
        energy_required_kwh=800,
        min_power_kw=80,
        max_power_kw=150,
        deadline_hour=18,
        earliest_start_hour=6,
        priority=2.0
    )
]

# Optimize schedule
result = await scheduler.optimize(jobs)

print(f"Optimized Schedule:")
for op in result.schedule:
    print(f"  {op.job_id}: Hour {op.hour}, Power {op.power_kw} kW, Cost ${op.cost:.2f}")

print(f"\nTotal Cost: ${result.total_cost:.2f}")
print(f"Savings vs Flat: ${result.cost_savings_vs_flat:.2f}")
```

#### Calculate Energy Costs

```python
from greenlang.GL_019.calculators.energy_cost_calculator import (
    EnergyCostCalculator,
    TariffStructure,
    TariffType,
    TariffRate,
    TimePeriod,
    HourlyLoad,
    EnergyCostInput
)

# Define tariff
tariff = TariffStructure(
    tariff_type=TariffType.TIME_OF_USE,
    rates=[
        TariffRate(rate_per_kwh=0.25, period=TimePeriod.PEAK),
        TariffRate(rate_per_kwh=0.15, period=TimePeriod.SHOULDER),
        TariffRate(rate_per_kwh=0.08, period=TimePeriod.OFF_PEAK),
        TariffRate(rate_per_kwh=0.0, demand_rate_per_kw=15.00)  # Demand charge
    ],
    peak_hours=list(range(14, 20)),
    shoulder_hours=list(range(6, 14)) + list(range(20, 22)),
    off_peak_hours=list(range(0, 6)) + list(range(22, 24))
)

# Define hourly loads
loads = [
    HourlyLoad(hour=h, energy_kwh=100.0, peak_demand_kw=120.0, date="2025-01-15")
    for h in range(24)
]

# Calculate costs
calculator = EnergyCostCalculator()
inputs = EnergyCostInput(tariff=tariff, hourly_loads=loads)
result, provenance = calculator.calculate(inputs)

print(f"Total Cost: ${result.total_cost:.2f}")
print(f"  Energy Cost: ${result.energy_cost:.2f}")
print(f"  Demand Cost: ${result.demand_cost:.2f}")
print(f"  Base Cost: ${result.base_cost:.2f}")
print(f"Average Rate: ${result.average_rate_per_kwh:.4f}/kWh")
print(f"\nCost by Period:")
for period, cost in result.cost_by_period.items():
    print(f"  {period}: ${cost:.2f}")
```

#### Forecast Savings

```python
from greenlang.GL_019.calculators.savings_calculator import (
    SavingsCalculator,
    ScheduleComparison,
    HourlyScheduleData,
    ProjectCosts,
    SavingsCalculatorInput
)

# Create baseline and optimized schedules
baseline = [
    HourlyScheduleData(
        hour=h,
        energy_kwh=100,
        demand_kw=120,
        energy_rate=0.25 if 14 <= h < 20 else 0.08,
        is_peak_hour=(14 <= h < 20)
    )
    for h in range(24)
]

optimized = [
    HourlyScheduleData(
        hour=h,
        energy_kwh=100 if h < 6 or h >= 20 else 50,  # Shifted to off-peak
        demand_kw=100,
        energy_rate=0.08 if h < 6 or h >= 22 else 0.15,
        is_peak_hour=(14 <= h < 20)
    )
    for h in range(24)
]

# Define project costs
project_costs = ProjectCosts(
    capital_cost=50000,
    implementation_cost=10000,
    annual_maintenance_cost=5000,
    software_license_annual=12000,
    training_cost=3000
)

# Calculate savings
calculator = SavingsCalculator()
inputs = SavingsCalculatorInput(
    schedule_comparison=ScheduleComparison(baseline, optimized, analysis_period_days=30),
    project_costs=project_costs,
    projection_years=5,
    discount_rate=0.08
)

result, provenance = calculator.calculate(inputs)

print(f"Period Savings: ${result.period_savings:,.2f}")
print(f"Annual Projection: ${result.annual_savings_projection:,.2f}")
print(f"Savings Percentage: {result.savings_percentage:.1f}%")
print(f"\nROI Metrics:")
print(f"  Simple Payback: {result.roi_metrics.simple_payback_years:.1f} years")
print(f"  NPV: ${result.roi_metrics.npv:,.2f}")
print(f"  IRR: {result.roi_metrics.irr * 100:.1f}%")
print(f"  ROI: {result.roi_metrics.roi_percentage:.1f}%")
```

### REST API Usage

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "agent_id": "GL-019",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "erp_connected": true,
  "scada_connected": true,
  "last_optimization": "2025-12-03T10:30:00Z"
}
```

#### Submit Optimization Request

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "horizon_days": 7,
    "optimize_for": "cost",
    "include_dr_opportunities": true
  }'
```

Response:
```json
{
  "job_id": "opt_abc123",
  "status": "completed",
  "schedule": {
    "total_cost": 45230.50,
    "baseline_cost": 53210.00,
    "savings": 7979.50,
    "savings_pct": 15.0,
    "peak_demand_kw": 2850,
    "peak_demand_reduction_kw": 650,
    "operations": [...]
  },
  "provenance_hash": "sha256:abc123..."
}
```

#### Get Load Shift Recommendations

```bash
curl http://localhost:8000/api/v1/recommendations?top=10
```

Response:
```json
{
  "recommendations": [
    {
      "operation_id": "BATCH-001",
      "current_window": ["2025-12-03T14:00:00Z", "2025-12-03T18:00:00Z"],
      "recommended_window": ["2025-12-03T02:00:00Z", "2025-12-03T06:00:00Z"],
      "savings_usd": 450.00,
      "confidence": 0.95,
      "explanation": "Shift to off-peak hours (2-6 AM) saves $450 from lower energy rate ($0.08 vs $0.25/kWh)"
    }
  ]
}
```

---

## API Reference

### Core Classes

#### ProcessHeatingScheduler

Main orchestrator for heating schedule optimization.

```python
class ProcessHeatingScheduler:
    def __init__(self, config_path: str):
        """Initialize scheduler with configuration."""

    async def optimize_schedule(
        self,
        horizon_days: int = 7,
        production_source: str = "erp",
        optimize_for: str = "cost"
    ) -> ScheduleResult:
        """Generate optimized heating schedule."""

    async def get_recommendations(
        self,
        max_recommendations: int = 10
    ) -> List[LoadShiftRecommendation]:
        """Get load shifting recommendations."""

    async def respond_to_dr_event(
        self,
        event: DemandResponseEvent
    ) -> CurtailmentPlan:
        """Generate curtailment plan for DR event."""

    async def get_savings_forecast(
        self,
        period_days: int = 30
    ) -> SavingsForecast:
        """Forecast savings for upcoming period."""
```

See [API_README.md](API_README.md) for complete API documentation.

---

## Integration Guides

### ERP/MES Integration

HEATSCHEDULER supports multiple ERP and MES systems:

#### SAP PP/DS Integration

```python
from greenlang.GL_019.integrations import create_erp_connector

erp = create_erp_connector(
    system="SAP_PP_DS",
    endpoint="https://sap-server:443",
    client="100",
    username="heatscheduler",
    password="password"
)

# Fetch production orders
orders = await erp.get_production_orders(
    plant="1000",
    work_center="FURNACE*",
    date_from="2025-01-01",
    date_to="2025-01-07"
)

print(f"Found {len(orders)} production orders")
```

#### Oracle SCM Integration

```python
erp = create_erp_connector(
    system="Oracle_SCM",
    endpoint="https://oracle-server:443",
    auth_type="oauth2",
    client_id="your_client_id",
    client_secret="your_secret"
)

orders = await erp.get_production_orders(
    organization_id="102",
    resource_group="HEAT_TREAT"
)
```

### SCADA/DCS Integration

```python
from greenlang.GL_019.integrations import create_scada_client

scada = create_scada_client(
    protocol="OPC_UA",
    endpoint="opc.tcp://192.168.1.100:4840",
    username="scada_user",
    password="password"
)

await scada.connect()

# Read equipment status
furnace_status = await scada.read_tag("FURNACE_001.Status")
furnace_power = await scada.read_tag("FURNACE_001.Power_kW")

print(f"Furnace Status: {furnace_status}")
print(f"Current Power: {furnace_power} kW")
```

### Demand Response Integration

```python
from greenlang.GL_019.integrations import DemandResponseManager

dr_manager = DemandResponseManager(
    protocol="OpenADR_2_0b",
    vtn_url="https://dr-provider.com/openadr/",
    ven_id="FACILITY_001"
)

# Register for DR events
await dr_manager.register()

# Handle incoming DR event
@dr_manager.on_event
async def handle_dr_event(event):
    curtailment = await scheduler.respond_to_dr_event(event)
    await dr_manager.submit_opt_in(event.event_id, curtailment.offered_kw)
```

---

## Monitoring and Alerting

### Prometheus Metrics

HEATSCHEDULER exports metrics on port 9090:

```bash
curl http://localhost:9090/metrics
```

**Key Metrics**:
```
# Schedule optimization metrics
heatscheduler_optimizations_total{status="success"} 2847
heatscheduler_optimization_duration_seconds{quantile="0.95"} 45.2

# Cost metrics
heatscheduler_total_cost_usd{period="daily"} 45230.50
heatscheduler_savings_usd{period="daily"} 7979.50
heatscheduler_savings_percentage 15.0

# Demand metrics
heatscheduler_peak_demand_kw 2850
heatscheduler_demand_reduction_kw 650

# DR metrics
heatscheduler_dr_events_total{status="completed"} 12
heatscheduler_dr_revenue_usd 45000

# Equipment metrics
heatscheduler_equipment_utilization{equipment_id="FURNACE-001"} 0.85
```

### Alert Configuration

Configure alerts in `config/alerts.yaml`:

```yaml
alerts:
  - name: high_energy_cost
    condition: "daily_cost > baseline_cost * 1.1"
    severity: warning
    notification_channels:
      - email: energy@company.com
      - slack: #energy-alerts

  - name: optimization_failure
    condition: "optimization_status == 'failed'"
    severity: critical
    notification_channels:
      - pagerduty: optimization-oncall

  - name: deadline_risk
    condition: "deadline_margin_hours < 2"
    severity: high
    notification_channels:
      - email: operations@company.com
```

---

## Troubleshooting

### Common Issues

#### Schedule Optimization Takes Too Long

**Symptoms**: Optimization exceeds 5-minute target

**Solutions**:
1. Reduce optimization horizon (try 3 days instead of 7)
2. Increase time slot duration (2 hours instead of 1)
3. Use faster solver (Gurobi/CPLEX instead of HiGHS)
4. Reduce number of jobs by aggregating similar batches

#### ERP Connection Failures

**Symptoms**: Cannot fetch production schedule

**Solutions**:
1. Verify ERP endpoint and credentials
2. Check network connectivity and firewall rules
3. Validate OAuth2 token refresh
4. Review ERP API rate limits

#### Infeasible Schedule

**Symptoms**: Optimizer returns "infeasible" status

**Solutions**:
1. Check production deadlines vs. available capacity
2. Verify equipment availability windows
3. Review maintenance schedule conflicts
4. Consider relaxing constraints (extend deadlines if possible)

### Diagnostic Commands

```bash
# Check agent status
curl http://localhost:8000/health

# Get ERP connection status
curl http://localhost:8000/api/v1/integrations/erp/status

# Get SCADA connection status
curl http://localhost:8000/api/v1/integrations/scada/status

# Review optimization logs
curl http://localhost:8000/api/v1/logs?level=INFO&hours=24
```

---

## Contributing

We welcome contributions from the community!

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/gl-019-heatscheduler.git
cd gl-019-heatscheduler

# Create development environment
python3.10 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=. --cov-report=html
```

### Code Standards

- **Python Version**: 3.10+
- **Style Guide**: PEP 8, enforced by Black and Flake8
- **Type Hints**: Required for all functions
- **Docstrings**: Google style
- **Test Coverage**: Minimum 85% overall, 95% for calculators

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run linting: `black . && flake8 . && isort .`
5. Run tests: `pytest tests/ -v`
6. Commit: `git commit -m "feat: add my feature"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

---

## License

```
Copyright 2025 GreenLang Technologies

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Support

- **Documentation**: [https://greenlang.io/agents/GL-019](https://greenlang.io/agents/GL-019)
- **API Reference**: [https://api.greenlang.io/GL-019](https://api.greenlang.io/GL-019)
- **Technical Support**: support@greenlang.io
- **Community Slack**: [greenlang-community.slack.com](https://greenlang-community.slack.com)
- **GitHub Issues**: [github.com/greenlang/gl-019-heatscheduler/issues](https://github.com/greenlang/gl-019-heatscheduler/issues)

---

**GL-019 HEATSCHEDULER** - Intelligent Process Heating Schedule Optimization

*Minimize energy costs, meet every deadline, maximize savings.*

For questions or support, contact: support@greenlang.io
