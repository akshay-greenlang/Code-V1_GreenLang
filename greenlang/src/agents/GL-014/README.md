# GL-014 EXCHANGER-PRO: Heat Exchanger Optimization Agent

## Industrial Heat Exchanger Performance Monitoring, Fouling Prediction, and Intelligent Cleaning Schedule Generation

**Agent ID:** GL-014
**Codename:** EXCHANGER-PRO
**Version:** 1.0.0
**Category:** Heat Exchangers
**Type:** Optimizer
**Business Priority:** P1
**Target Market TAM:** $6B
**License:** Apache-2.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Key Capabilities](#key-capabilities)
3. [Architecture Overview](#architecture-overview)
4. [Quick Start Guide](#quick-start-guide)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [API Reference Summary](#api-reference-summary)
8. [Integration Guide](#integration-guide)
9. [Calculation Methodology](#calculation-methodology)
10. [Deployment Instructions](#deployment-instructions)
11. [Monitoring and Observability](#monitoring-and-observability)
12. [Security](#security)
13. [Troubleshooting Guide](#troubleshooting-guide)
14. [FAQ](#faq)
15. [Compliance and Standards](#compliance-and-standards)
16. [Contributing](#contributing)
17. [Changelog](#changelog)
18. [Support](#support)

---

## Executive Summary

EXCHANGER-PRO is a comprehensive industrial heat exchanger optimization agent that delivers real-time performance monitoring, predictive fouling analysis, and cost-optimized cleaning schedules. Built on GreenLang's zero-hallucination architecture, all calculations are deterministic, fully traceable, and compliant with industry standards including TEMA, HTRI, API, and ASME.

### Value Proposition

Heat exchangers are critical assets in refining, chemical, power generation, and HVAC industries. Fouling-induced efficiency losses typically cost industrial facilities **$15,000 to $100,000 per exchanger annually** in wasted energy and unplanned maintenance. EXCHANGER-PRO addresses these challenges by:

- **Reducing energy costs by 15-30%** through optimized cleaning timing
- **Preventing unplanned shutdowns** with predictive fouling analysis
- **Extending equipment life** by identifying fouling mechanisms early
- **Automating maintenance workflows** through CMMS integration
- **Ensuring regulatory compliance** with complete audit trails

### Target Industries

| Industry | Use Cases | Typical Savings |
|----------|-----------|-----------------|
| Oil & Gas Refining | Crude preheat trains, overhead condensers | $50K-$500K/year |
| Chemical Processing | Reactor feed/effluent, distillation | $30K-$300K/year |
| Power Generation | Condenser optimization, feedwater heaters | $100K-$1M/year |
| Petrochemicals | Cracker heat recovery, polymerization | $40K-$400K/year |
| Pulp & Paper | Black liquor evaporators, recovery | $25K-$200K/year |
| Food & Beverage | Pasteurizers, evaporators | $10K-$100K/year |

### Zero-Hallucination Guarantee

EXCHANGER-PRO operates with a **zero-hallucination design**:

- All thermal calculations use deterministic engineering formulas
- No LLM involvement in any calculation path
- Complete provenance tracking with SHA-256 hashes
- Bit-perfect reproducibility (same input always produces same output)
- Immutable results using frozen dataclasses
- Full compliance with TEMA Standards, 10th Edition

---

## Key Capabilities

### 1. Thermal Performance Monitoring

Real-time monitoring and analysis of heat exchanger thermal performance:

```
+------------------------------------------+
|     THERMAL PERFORMANCE MONITORING       |
+------------------------------------------+
|                                          |
|  Heat Duty Calculation                   |
|  - Q = m_dot * Cp * delta_T              |
|  - Energy balance verification           |
|  - Hot/cold side reconciliation          |
|                                          |
|  LMTD Analysis                           |
|  - Counter-flow, parallel, crossflow     |
|  - Correction factors (F-factors)        |
|  - Multi-pass configurations             |
|                                          |
|  Effectiveness-NTU Method                |
|  - All flow arrangements supported       |
|  - Rating and design calculations        |
|  - Capacity ratio analysis               |
|                                          |
|  Overall Heat Transfer Coefficient       |
|  - U-value calculation and trending      |
|  - Clean vs. fouled comparison           |
|  - Thermal resistance breakdown          |
|                                          |
+------------------------------------------+
```

**Supported Heat Exchanger Types:**

| Type | TEMA Designation | Applications |
|------|------------------|--------------|
| Shell-and-Tube | E, F, G, H, J, K, X shells | Refinery, chemical, power |
| Plate Heat Exchanger | - | HVAC, food, pharmaceutical |
| Air-Cooled (Fin-Fan) | - | Refineries, gas processing |
| Spiral Heat Exchanger | - | Viscous fluids, slurries |
| Double-Pipe | - | Small duties, high pressure |
| Plate-Fin | - | Cryogenic, aerospace |
| Printed Circuit | PCHE | LNG, offshore |

**Temperature Monitoring:**

```python
# Example temperature data structure
temperature_data = {
    "hot_inlet_temp_c": 150.0,      # Hot stream inlet
    "hot_outlet_temp_c": 80.0,      # Hot stream outlet
    "cold_inlet_temp_c": 25.0,      # Cold stream inlet
    "cold_outlet_temp_c": 65.0,     # Cold stream outlet
    "ambient_temp_c": 25.0,         # Ambient reference
    "measurement_timestamp": "2025-12-01T10:30:00Z"
}
```

### 2. Fouling Analysis and Prediction

Comprehensive fouling analysis using industry-standard models:

```
+------------------------------------------+
|         FOULING ANALYSIS ENGINE          |
+------------------------------------------+
|                                          |
|  Current State Assessment                |
|  - Fouling factor calculation            |
|  - Shell-side vs. tube-side breakdown    |
|  - Cleanliness factor (CF%)              |
|  - Normalized fouling factor (R_f*)      |
|                                          |
|  Fouling Mechanism Identification        |
|  - Particulate (sedimentation)           |
|  - Crystallization (scaling)             |
|  - Biological (biofilm)                  |
|  - Corrosion (oxide layers)              |
|  - Chemical reaction (coking)            |
|                                          |
|  Predictive Models                       |
|  - Kern-Seaton asymptotic model          |
|  - Ebert-Panchal threshold model         |
|  - Linear fouling progression            |
|  - Falling-rate fouling model            |
|                                          |
|  Severity Classification                 |
|  - Clean / Light / Moderate              |
|  - Heavy / Severe / Critical             |
|  - Time-to-critical estimation           |
|                                          |
+------------------------------------------+
```

**Fouling Severity Levels:**

| Level | R_f* Range | Cleanliness Factor | Action Required |
|-------|------------|-------------------|-----------------|
| Clean | < 0.1 | > 95% | Normal operation |
| Light | 0.1 - 0.3 | 85% - 95% | Monitor closely |
| Moderate | 0.3 - 0.6 | 70% - 85% | Plan cleaning |
| Heavy | 0.6 - 0.9 | 55% - 70% | Schedule cleaning |
| Severe | 0.9 - 1.2 | 40% - 55% | Urgent cleaning |
| Critical | >= 1.2 | <= 40% | Immediate action |

**Fouling Factor Calculation:**

The fundamental fouling resistance calculation:

```
R_f = (1/U_fouled) - (1/U_clean)

Where:
  R_f       = Total fouling resistance (m^2*K/W)
  U_fouled  = Current overall heat transfer coefficient (W/m^2*K)
  U_clean   = Clean overall heat transfer coefficient (W/m^2*K)
```

**Kern-Seaton Asymptotic Model:**

```
R_f(t) = R_f_max * (1 - exp(-t/tau))

Where:
  R_f(t)    = Fouling resistance at time t
  R_f_max   = Asymptotic (maximum) fouling resistance
  tau       = Fouling time constant
  t         = Operating time since last cleaning
```

### 3. Intelligent Cleaning Schedule Optimization

Cost-benefit optimized cleaning schedule generation:

```
+------------------------------------------+
|     CLEANING SCHEDULE OPTIMIZER          |
+------------------------------------------+
|                                          |
|  Economic Analysis                       |
|  - Energy loss quantification            |
|  - Cleaning cost estimation              |
|  - Downtime cost calculation             |
|  - NPV of cleaning decisions             |
|                                          |
|  Optimal Timing Calculation              |
|  - Cost-benefit ratio analysis           |
|  - Payback period calculation            |
|  - Marginal economics evaluation         |
|                                          |
|  Cleaning Method Selection               |
|  - Chemical cleaning (CIP)               |
|  - Mechanical cleaning (hydroblast)      |
|  - Offline vs. online options            |
|  - Method-specific effectiveness         |
|                                          |
|  CMMS Integration                        |
|  - Work order generation                 |
|  - Preventive maintenance scheduling     |
|  - Resource planning                     |
|                                          |
+------------------------------------------+
```

**Cleaning Method Comparison:**

| Method | Typical Cost | Downtime | Effectiveness | Best For |
|--------|-------------|----------|---------------|----------|
| Chemical (Online) | $8,000 | 0 hrs | 70-85% | Light fouling |
| Chemical (Offline) | $15,000-20,000 | 24-36 hrs | 80-90% | Moderate fouling |
| Mechanical | $25,000-35,000 | 36-48 hrs | 90-98% | Heavy/scaling |
| Hydroblasting | $30,000-45,000 | 24-36 hrs | 85-95% | Hard deposits |
| Ultrasonic | $20,000-30,000 | 12-24 hrs | 75-85% | Precision cleaning |

**Cleaning ROI Calculation:**

```
ROI = (Annual_Energy_Savings - Cleaning_Costs) / Cleaning_Costs * 100%

Simple_Payback = Cleaning_Investment / Monthly_Savings
```

### 4. Economic Impact Analysis

Comprehensive financial analysis of fouling impact:

```
+------------------------------------------+
|       ECONOMIC IMPACT CALCULATOR         |
+------------------------------------------+
|                                          |
|  Energy Loss Quantification              |
|  - kW loss from reduced efficiency       |
|  - Hourly/daily/monthly/annual costs     |
|  - Utility rate sensitivity              |
|                                          |
|  Production Impact                       |
|  - Throughput reduction estimation       |
|  - Quality impact assessment             |
|  - Revenue loss calculation              |
|                                          |
|  Total Cost of Ownership (TCO)           |
|  - Capital costs (replacement)           |
|  - Operating costs (energy)              |
|  - Maintenance costs (cleaning)          |
|  - Downtime costs                        |
|                                          |
|  ROI Analysis                            |
|  - Net Present Value (NPV)               |
|  - Internal Rate of Return (IRR)         |
|  - Payback period                        |
|  - Sensitivity analysis                  |
|                                          |
|  Carbon Impact                           |
|  - CO2 emissions from energy waste       |
|  - Carbon cost calculation               |
|  - Environmental reporting               |
|                                          |
+------------------------------------------+
```

**Economic Analysis Output Example:**

```json
{
  "energy_loss": {
    "power_loss_kw": 250.5,
    "daily_energy_loss_kwh": 6012.0,
    "monthly_energy_loss_mwh": 180.36,
    "annual_energy_loss_mwh": 2164.32,
    "annual_energy_cost_usd": 216432.00
  },
  "cleaning_economics": {
    "recommended_cleaning_cost_usd": 25000.00,
    "downtime_cost_usd": 12000.00,
    "total_investment_usd": 37000.00,
    "monthly_savings_usd": 18036.00,
    "simple_payback_days": 62,
    "annual_roi_percent": 485.0
  },
  "carbon_impact": {
    "annual_co2_emissions_tonnes": 865.73,
    "carbon_cost_at_50_per_tonne_usd": 43286.50
  }
}
```

### 5. Process Historian Integration

Seamless integration with major process historians:

```
+------------------------------------------+
|    PROCESS HISTORIAN INTEGRATION         |
+------------------------------------------+
|                                          |
|  OSIsoft PI                              |
|  - PI Web API integration                |
|  - Real-time and historical data         |
|  - Tag search and discovery              |
|  - Bulk data retrieval                   |
|                                          |
|  Honeywell PHD                           |
|  - PHD API connectivity                  |
|  - Time-series data access               |
|  - Point configuration                   |
|                                          |
|  AspenTech IP.21                         |
|  - IP.21 SQLplus queries                 |
|  - Historical data retrieval             |
|  - Tag metadata access                   |
|                                          |
|  OPC-UA (Generic)                        |
|  - OPC-UA client implementation          |
|  - Subscribe/poll modes                  |
|  - Certificate-based security            |
|                                          |
+------------------------------------------+
```

**Supported Tag Types:**

| Tag Category | Tags | Unit |
|--------------|------|------|
| Temperature | hot_inlet, hot_outlet, cold_inlet, cold_outlet | C or F |
| Pressure | shell_inlet, shell_outlet, tube_inlet, tube_outlet | kPa or psi |
| Flow | hot_flow, cold_flow | kg/s or GPM |
| Calculated | heat_duty, u_value, effectiveness | kW, W/m2K, % |

### 6. CMMS Integration

Automated maintenance workflow integration:

```
+------------------------------------------+
|         CMMS INTEGRATION                 |
+------------------------------------------+
|                                          |
|  SAP Plant Maintenance (PM)              |
|  - Functional location sync              |
|  - Equipment master data                 |
|  - Notification creation                 |
|  - Work order generation                 |
|  - Maintenance plan integration          |
|                                          |
|  IBM Maximo                              |
|  - Asset management sync                 |
|  - Work order creation                   |
|  - PM schedule integration               |
|  - Service request automation            |
|                                          |
|  Oracle EAM                              |
|  - Asset registry sync                   |
|  - Maintenance work orders               |
|  - Resource planning                     |
|  - Cost tracking                         |
|                                          |
|  Common Features                         |
|  - Bi-directional data sync              |
|  - Cleaning history tracking             |
|  - Cost accumulation                     |
|  - Performance trending                  |
|                                          |
+------------------------------------------+
```

---

## Architecture Overview

### High-Level Architecture

```
+------------------------------------------------------------------+
|                    GL-014 EXCHANGER-PRO                          |
|                  Heat Exchanger Optimizer                        |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------------+    +---------------------------+     |
|  |    API Layer           |    |   Integration Layer       |     |
|  |  +-----------------+   |    |  +---------------------+  |     |
|  |  | REST API        |   |    |  | Process Historian   |  |     |
|  |  | (FastAPI)       |   |    |  | - OSIsoft PI        |  |     |
|  |  +-----------------+   |    |  | - Honeywell PHD     |  |     |
|  |  | GraphQL API     |   |    |  | - AspenTech IP.21   |  |     |
|  |  | (Optional)      |   |    |  | - OPC-UA            |  |     |
|  |  +-----------------+   |    |  +---------------------+  |     |
|  |  | WebSocket       |   |    |  | CMMS Integration    |  |     |
|  |  | (Real-time)     |   |    |  | - SAP PM            |  |     |
|  |  +-----------------+   |    |  | - IBM Maximo        |  |     |
|  +------------------------+    |  | - Oracle EAM        |  |     |
|                                |  +---------------------+  |     |
|  +------------------------+    |  | Agent Coordination  |  |     |
|  |   Calculator Layer     |    |  | - GL-001 THERMOSYNC |  |     |
|  |  +-----------------+   |    |  | - GL-006 HEATRECLAIM|  |     |
|  |  | Heat Transfer   |   |    |  | - GL-013 PREDICTMNT |  |     |
|  |  | Calculator      |   |    |  +---------------------+  |     |
|  |  +-----------------+   |    +---------------------------+     |
|  |  | Fouling         |   |                                      |
|  |  | Calculator      |   |    +---------------------------+     |
|  |  +-----------------+   |    |   Persistence Layer       |     |
|  |  | Pressure Drop   |   |    |  +---------------------+  |     |
|  |  | Calculator      |   |    |  | PostgreSQL          |  |     |
|  |  +-----------------+   |    |  | (Primary DB)        |  |     |
|  |  | Cleaning        |   |    |  +---------------------+  |     |
|  |  | Optimizer       |   |    |  | TimescaleDB         |  |     |
|  |  +-----------------+   |    |  | (Time-series)       |  |     |
|  |  | Economic        |   |    |  +---------------------+  |     |
|  |  | Calculator      |   |    |  | Redis               |  |     |
|  |  +-----------------+   |    |  | (Cache)             |  |     |
|  +------------------------+    |  +---------------------+  |     |
|                                +---------------------------+     |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |                    Observability Layer                      |  |
|  |  +----------------+  +----------------+  +----------------+ |  |
|  |  | Prometheus     |  | OpenTelemetry  |  | Structured     | |  |
|  |  | Metrics        |  | Tracing        |  | Logging        | |  |
|  |  +----------------+  +----------------+  +----------------+ |  |
|  +------------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

### Data Flow Diagram

```
+----------------+     +------------------+     +------------------+
|  Process       |     |  GL-014          |     |  CMMS            |
|  Historian     |---->|  EXCHANGER-PRO   |---->|  System          |
|  (PI/PHD/IP21) |     |                  |     |  (SAP/Maximo)    |
+----------------+     +------------------+     +------------------+
        |                      |                        |
        |                      |                        |
        v                      v                        v
+----------------+     +------------------+     +------------------+
| Temperature,   |     | Fouling Analysis |     | Work Orders      |
| Pressure,      |     | Performance Calc |     | Cleaning         |
| Flow Data      |     | Economic Impact  |     | Schedules        |
+----------------+     +------------------+     +------------------+
        |                      |                        |
        +----------+-----------+------------------------+
                   |
                   v
        +------------------+
        | TimescaleDB      |
        | Historical Store |
        +------------------+
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| API Layer | HTTP/WebSocket endpoints | FastAPI, Pydantic |
| Heat Transfer Calculator | LMTD, NTU, U-value calculations | Python, Decimal |
| Fouling Calculator | Fouling factor, prediction models | Python, Decimal |
| Pressure Drop Calculator | Shell/tube pressure drop | Python, Decimal |
| Cleaning Optimizer | Schedule optimization, ROI | Python, Decimal |
| Economic Calculator | NPV, IRR, TCO analysis | Python, Decimal |
| Process Historian Connector | PI, PHD, IP.21, OPC-UA | asyncio, aiohttp |
| CMMS Connector | SAP, Maximo, Oracle EAM | asyncio, aiohttp |
| Persistence | Data storage, caching | PostgreSQL, Redis |
| Observability | Metrics, traces, logs | Prometheus, OTLP |

---

## Quick Start Guide

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Process historian access (optional)
- CMMS access (optional)

### 5-Minute Quick Start

**Step 1: Clone and Setup**

```bash
# Clone the repository
git clone https://github.com/greenlang/agents.git
cd agents/gl-014

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Configure Environment**

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

Minimum required configuration:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/gl014

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

**Step 3: Start the Agent**

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or run directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Step 4: Verify Installation**

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", "agent": "GL-014"}
```

**Step 5: Make Your First API Call**

```bash
# Analyze heat exchanger performance
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "exchanger_id": "HX-001",
    "temperature_data": {
      "hot_inlet_temp_c": 150.0,
      "hot_outlet_temp_c": 80.0,
      "cold_inlet_temp_c": 25.0,
      "cold_outlet_temp_c": 65.0
    },
    "flow_data": {
      "hot_mass_flow_kg_s": 10.0,
      "cold_mass_flow_kg_s": 15.0
    },
    "exchanger_parameters": {
      "design_heat_duty_kw": 3000.0,
      "design_u_w_m2k": 500.0,
      "heat_transfer_area_m2": 100.0
    }
  }'
```

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| Memory | 2 GB | 4+ GB |
| Storage | 10 GB | 50+ GB |
| Python | 3.11 | 3.11+ |
| PostgreSQL | 14 | 15+ |
| Redis | 6 | 7+ |

### Installation Methods

#### Method 1: Docker (Recommended)

```bash
# Pull the official image
docker pull greenlang/gl-014:latest

# Run with Docker
docker run -d \
  --name gl-014 \
  -p 8000:8000 \
  -p 8001:8001 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/gl014 \
  -e REDIS_URL=redis://redis:6379/0 \
  greenlang/gl-014:latest
```

#### Method 2: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  gl-014:
    image: greenlang/gl-014:latest
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/gl014
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=gl014
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

```bash
docker-compose up -d
```

#### Method 3: Kubernetes (Helm)

```bash
# Add GreenLang Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Install GL-014
helm install gl-014 greenlang/gl-014 \
  --namespace greenlang \
  --create-namespace \
  --set database.url=postgresql://user:pass@postgres:5432/gl014 \
  --set redis.url=redis://redis:6379/0
```

#### Method 4: Pip Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install from PyPI
pip install greenlang-gl014

# Or install from source
pip install git+https://github.com/greenlang/agents.git#subdirectory=gl-014
```

### Post-Installation Verification

```bash
# Run health check
curl http://localhost:8000/health

# Run self-test
curl http://localhost:8000/api/v1/self-test

# Check metrics endpoint
curl http://localhost:8001/metrics
```

---

## Configuration

### Environment Variables

#### Core Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `API_HOST` | API bind address | `0.0.0.0` | No |
| `API_PORT` | API port | `8000` | No |
| `METRICS_PORT` | Prometheus metrics port | `8001` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `LOG_FORMAT` | Log format (json/text) | `json` | No |

#### Database Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_POOL_SIZE` | Connection pool size | `10` |
| `DB_MAX_OVERFLOW` | Max overflow connections | `20` |
| `DB_POOL_TIMEOUT` | Pool timeout (seconds) | `30` |
| `DB_POOL_RECYCLE` | Connection recycle time | `1800` |

#### Cache Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CACHE_TTL` | Cache TTL (seconds) | `300` |
| `CACHE_MAX_ENTRIES` | Max cache entries | `10000` |
| `REDIS_POOL_SIZE` | Redis pool size | `10` |

#### Process Historian Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PI_WEB_API_URL` | OSIsoft PI Web API URL | - |
| `PI_WEB_API_USER` | PI Web API username | - |
| `PI_WEB_API_PASSWORD` | PI Web API password | - |
| `PHD_API_URL` | Honeywell PHD API URL | - |
| `IP21_HOST` | AspenTech IP.21 host | - |
| `OPC_UA_ENDPOINT` | OPC-UA server endpoint | - |

#### CMMS Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SAP_RFC_HOST` | SAP RFC gateway host | - |
| `SAP_SYSTEM_NUMBER` | SAP system number | - |
| `SAP_CLIENT` | SAP client number | - |
| `MAXIMO_API_URL` | IBM Maximo API URL | - |
| `ORACLE_EAM_URL` | Oracle EAM API URL | - |

#### Calculation Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ELECTRICITY_COST_PER_KWH` | Electricity cost ($/kWh) | `0.10` |
| `STEAM_COST_PER_KG` | Steam cost ($/kg) | `0.03` |
| `COOLING_WATER_COST_PER_M3` | Cooling water cost ($/m3) | `0.50` |
| `DOWNTIME_COST_PER_HOUR` | Downtime cost ($/hour) | `5000.0` |
| `CLEANING_LABOR_COST_PER_HOUR` | Labor cost ($/hour) | `150.0` |

### Configuration File (gl.yaml)

```yaml
# GL-014 Configuration
agent:
  id: GL-014
  codename: EXCHANGER-PRO
  version: "1.0.0"

configuration:
  runtime:
    deterministic: true
    temperature: 0.0
    seed: 42
    zero_hallucination: true
    max_retries: 3
    timeout_seconds: 300

  cache:
    enabled: true
    ttl_seconds: 300
    max_entries: 10000
    strategy: lru

  batch:
    enabled: true
    max_batch_size: 500
    parallel_workers: 4
    chunk_size: 100

  thresholds:
    # Fouling factor thresholds (m2-K/W)
    fouling_light: 0.0001
    fouling_moderate: 0.0003
    fouling_heavy: 0.0005
    fouling_severe: 0.001
    fouling_critical: 0.002

    # Performance thresholds
    efficiency_optimal: 95.0
    efficiency_good: 85.0
    efficiency_degraded: 75.0
    efficiency_poor: 65.0
    efficiency_critical: 50.0

    # U-value ratio thresholds
    u_ratio_optimal: 0.95
    u_ratio_good: 0.85
    u_ratio_degraded: 0.70
    u_ratio_poor: 0.55
    u_ratio_critical: 0.40

  economics:
    electricity_cost_per_kwh: 0.10
    steam_cost_per_kg: 0.03
    cooling_water_cost_per_m3: 0.50
    downtime_cost_per_hour: 5000.0
    cleaning_labor_cost_per_hour: 150.0

  cleaning:
    chemical_cleaning_cost: 15000.0
    mechanical_cleaning_cost: 25000.0
    hydroblast_cleaning_cost: 35000.0
    chemical_downtime_hours: 24.0
    mechanical_downtime_hours: 48.0
    hydroblast_downtime_hours: 36.0
    chemical_effectiveness: 85.0
    mechanical_effectiveness: 95.0
    hydroblast_effectiveness: 90.0

  fouling_model:
    default_time_constant_days: 180.0
    max_fouling_factor: 0.002
    fouling_rate_acceleration_threshold: 0.00001

  safety:
    max_pressure_drop_ratio: 2.0
    max_temperature_approach_k: 50.0
    min_flow_ratio: 0.5
    max_tube_velocity_m_s: 3.0
    min_tube_velocity_m_s: 0.5
```

### Heat Exchanger Registration

Register heat exchangers in the system:

```json
{
  "exchanger_id": "HX-001",
  "exchanger_name": "Crude Preheat #1",
  "exchanger_type": "shell_and_tube",
  "flow_arrangement": "counterflow",
  "design_heat_duty_kw": 5000.0,
  "design_u_w_m2k": 450.0,
  "clean_u_w_m2k": 500.0,
  "heat_transfer_area_m2": 250.0,
  "shell_type": "E",
  "tube_layout": "triangular_30",
  "number_of_tubes": 500,
  "tube_length_m": 6.0,
  "tube_od_mm": 19.05,
  "tube_id_mm": 15.75,
  "tube_pitch_mm": 25.4,
  "tube_material": "stainless_316",
  "number_of_passes": 2,
  "baffle_spacing_mm": 300.0,
  "baffle_cut_percent": 25.0,
  "shell_diameter_mm": 600.0,
  "design_fouling_shell_m2kw": 0.0003,
  "design_fouling_tube_m2kw": 0.0002,
  "fluid_properties": {
    "hot_fluid": {
      "name": "Crude Oil",
      "fluid_type": "crude_oil",
      "phase": "liquid",
      "cp_j_kgk": 2100.0,
      "density_kg_m3": 850.0,
      "viscosity_pa_s": 0.005,
      "thermal_conductivity_w_mk": 0.13
    },
    "cold_fluid": {
      "name": "Desalter Effluent",
      "fluid_type": "crude_oil",
      "phase": "liquid",
      "cp_j_kgk": 2000.0,
      "density_kg_m3": 870.0,
      "viscosity_pa_s": 0.008,
      "thermal_conductivity_w_mk": 0.12
    }
  }
}
```

---

## API Reference Summary

### Base URL

```
https://api.greenlang.io/v1/gl-014
```

### Authentication

All API requests require JWT Bearer token authentication:

```bash
curl -X GET https://api.greenlang.io/v1/gl-014/exchangers \
  -H "Authorization: Bearer <your_token>"
```

### Core Endpoints

#### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agent": "GL-014",
  "timestamp": "2025-12-01T10:30:00Z"
}
```

#### Analyze Heat Exchanger

```http
POST /api/v1/analyze
Content-Type: application/json
Authorization: Bearer {token}
```

Request Body:
```json
{
  "exchanger_id": "HX-001",
  "temperature_data": {
    "hot_inlet_temp_c": 150.0,
    "hot_outlet_temp_c": 80.0,
    "cold_inlet_temp_c": 25.0,
    "cold_outlet_temp_c": 65.0
  },
  "pressure_data": {
    "shell_inlet_pressure_kpa": 500.0,
    "shell_outlet_pressure_kpa": 480.0,
    "tube_inlet_pressure_kpa": 600.0,
    "tube_outlet_pressure_kpa": 575.0
  },
  "flow_data": {
    "hot_mass_flow_kg_s": 10.0,
    "cold_mass_flow_kg_s": 15.0
  }
}
```

Response:
```json
{
  "exchanger_id": "HX-001",
  "analysis_timestamp": "2025-12-01T10:30:00Z",
  "performance_metrics": {
    "current_heat_duty_kw": 2940.0,
    "design_heat_duty_kw": 3000.0,
    "heat_duty_ratio": 0.98,
    "current_u_w_m2k": 420.0,
    "design_u_w_m2k": 500.0,
    "u_ratio": 0.84,
    "lmtd_k": 35.0,
    "effectiveness": 0.73,
    "performance_status": "good"
  },
  "fouling_analysis": {
    "total_fouling_factor": 0.00038,
    "shell_side_fouling_factor": 0.00020,
    "tube_side_fouling_factor": 0.00018,
    "fouling_state": "moderate",
    "predicted_days_to_threshold": 45
  },
  "cleaning_schedule": {
    "recommended_cleaning_date": "2026-01-15",
    "recommended_cleaning_method": "chemical",
    "urgency_level": "planned",
    "cost_benefit_ratio": 3.5,
    "estimated_cleaning_cost": 15000.0,
    "payback_period_days": 45
  },
  "economic_impact": {
    "daily_energy_cost_loss": 250.0,
    "monthly_energy_cost_loss": 7500.0,
    "annual_energy_cost_loss": 91250.0,
    "cleaning_roi_percent": 485.0
  },
  "provenance_hash": "sha256:a1b2c3d4e5f6..."
}
```

#### Get Cleaning Schedule

```http
GET /api/v1/exchangers/{exchanger_id}/cleaning-schedule
Authorization: Bearer {token}
```

#### Calculate Fouling

```http
POST /api/v1/fouling/calculate
Content-Type: application/json
Authorization: Bearer {token}
```

#### Get Economic Impact

```http
GET /api/v1/exchangers/{exchanger_id}/economic-impact
Authorization: Bearer {token}
```

#### Batch Analysis

```http
POST /api/v1/analyze/batch
Content-Type: application/json
Authorization: Bearer {token}
```

### Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input parameters |
| 401 | Unauthorized - Invalid or missing token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Exchanger not found |
| 422 | Validation Error - Input validation failed |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

Error Response Format:
```json
{
  "error": "validation_error",
  "message": "Hot inlet temperature must be greater than hot outlet temperature",
  "details": {
    "field": "temperature_data.hot_inlet_temp_c",
    "value": 80.0,
    "constraint": "hot_inlet_temp_c > hot_outlet_temp_c"
  },
  "request_id": "req_abc123"
}
```

---

## Integration Guide

### Process Historian Integration

#### OSIsoft PI Web API

```python
from gl014.integrations import ProcessHistorianConnector, HistorianType

# Configure PI Web API connection
connector = ProcessHistorianConnector(
    historian_type=HistorianType.OSISOFT_PI,
    config={
        "base_url": "https://pi-server.example.com/piwebapi",
        "username": "pi_user",
        "password": "pi_password",
        "verify_ssl": True
    }
)

# Connect to PI server
await connector.connect()

# Define tag mapping for heat exchanger
tag_mapping = {
    "hot_inlet_temp": "HX001.TI.101.PV",
    "hot_outlet_temp": "HX001.TI.102.PV",
    "cold_inlet_temp": "HX001.TI.103.PV",
    "cold_outlet_temp": "HX001.TI.104.PV",
    "hot_flow": "HX001.FI.101.PV",
    "cold_flow": "HX001.FI.102.PV"
}

# Retrieve current snapshot
snapshot = await connector.get_heat_exchanger_snapshot(
    exchanger_id="HX-001",
    tag_mapping=tag_mapping
)

# Retrieve historical data
history = await connector.get_time_series_data(
    tags=list(tag_mapping.values()),
    start_time="2025-11-01T00:00:00Z",
    end_time="2025-12-01T00:00:00Z",
    interval="1h"
)
```

#### Honeywell PHD

```python
connector = ProcessHistorianConnector(
    historian_type=HistorianType.HONEYWELL_PHD,
    config={
        "server": "phd-server.example.com",
        "port": 3000,
        "username": "phd_user",
        "password": "phd_password"
    }
)

await connector.connect()
```

#### AspenTech IP.21

```python
connector = ProcessHistorianConnector(
    historian_type=HistorianType.ASPENTECH_IP21,
    config={
        "host": "ip21-server.example.com",
        "port": 10014,
        "datasource": "IP21_MAIN"
    }
)

await connector.connect()
```

#### OPC-UA

```python
connector = ProcessHistorianConnector(
    historian_type=HistorianType.OPC_UA,
    config={
        "endpoint": "opc.tcp://opc-server.example.com:4840",
        "security_mode": "SignAndEncrypt",
        "certificate_path": "/path/to/cert.pem",
        "private_key_path": "/path/to/key.pem"
    }
)

await connector.connect()
```

### CMMS Integration

#### SAP Plant Maintenance

```python
from gl014.integrations import CMSSConnector, CMSSType

# Configure SAP PM connection
cmms = CMSSConnector(
    cmms_type=CMSSType.SAP_PM,
    config={
        "gateway_host": "sap-gw.example.com",
        "system_number": "00",
        "client": "100",
        "username": "sap_user",
        "password": "sap_password"
    }
)

await cmms.connect()

# Sync equipment from SAP
equipment = await cmms.sync_equipment(
    functional_location="PLANT-A/AREA-1/HX*"
)

# Create maintenance notification
notification = await cmms.create_notification(
    equipment_id="HX-001",
    notification_type="M2",  # Maintenance request
    description="Scheduled cleaning based on fouling analysis",
    priority="2",
    recommended_date="2026-01-15"
)

# Create work order
work_order = await cmms.create_work_order(
    equipment_id="HX-001",
    work_order_type="PM01",  # Preventive maintenance
    description="Heat exchanger cleaning - Chemical",
    planned_start="2026-01-15",
    planned_end="2026-01-16",
    operations=[
        {
            "operation": "0010",
            "description": "Isolate and depressurize",
            "work_center": "MECH01",
            "duration_hours": 2.0
        },
        {
            "operation": "0020",
            "description": "Chemical cleaning circulation",
            "work_center": "MECH01",
            "duration_hours": 16.0
        },
        {
            "operation": "0030",
            "description": "Flush and return to service",
            "work_center": "MECH01",
            "duration_hours": 4.0
        }
    ]
)
```

#### IBM Maximo

```python
cmms = CMSSConnector(
    cmms_type=CMSSType.IBM_MAXIMO,
    config={
        "base_url": "https://maximo.example.com/maximo/oslc",
        "api_key": "maximo_api_key",
        "site_id": "PLANT_A"
    }
)

await cmms.connect()

# Create work order
work_order = await cmms.create_work_order(
    asset_num="HX-001",
    work_type="PM",
    description="Heat exchanger cleaning - Chemical",
    target_start="2026-01-15T08:00:00",
    target_finish="2026-01-16T08:00:00",
    status="WAPPR"
)
```

#### Oracle EAM

```python
cmms = CMSSConnector(
    cmms_type=CMSSType.ORACLE_EAM,
    config={
        "base_url": "https://oracle-eam.example.com/api/v1",
        "username": "eam_user",
        "password": "eam_password",
        "organization_id": "1234"
    }
)

await cmms.connect()
```

### Agent Coordination

EXCHANGER-PRO can coordinate with other GreenLang agents:

```python
from gl014.coordination import AgentCoordinator

coordinator = AgentCoordinator()

# Coordinate with GL-001 THERMOSYNC (Steam Trap Optimization)
steam_data = await coordinator.request_data(
    agent_id="GL-001",
    data_type="steam_system_performance",
    parameters={"area": "PLANT-A"}
)

# Coordinate with GL-006 HEATRECLAIM (Heat Recovery)
heat_network = await coordinator.request_data(
    agent_id="GL-006",
    data_type="heat_network_status",
    parameters={"network_id": "HRN-001"}
)

# Coordinate with GL-013 PREDICTMAINT (Predictive Maintenance)
failure_prediction = await coordinator.request_data(
    agent_id="GL-013",
    data_type="failure_probability",
    parameters={"equipment_id": "HX-001"}
)
```

---

## Calculation Methodology

### Heat Transfer Calculations

#### Log Mean Temperature Difference (LMTD)

```
LMTD = (dT1 - dT2) / ln(dT1 / dT2)

For counter-flow:
  dT1 = T_hot_in - T_cold_out
  dT2 = T_hot_out - T_cold_in

For parallel flow:
  dT1 = T_hot_in - T_cold_in
  dT2 = T_hot_out - T_cold_out

For multi-pass exchangers:
  LMTD_corrected = F * LMTD

  Where F is the correction factor from TEMA charts
```

#### Effectiveness-NTU Method

```
Effectiveness (e) = Q_actual / Q_max

Q_max = C_min * (T_hot_in - T_cold_in)

C_min = min(m_hot * Cp_hot, m_cold * Cp_cold)
C_max = max(m_hot * Cp_hot, m_cold * Cp_cold)

C_r = C_min / C_max

NTU = U * A / C_min

For counter-flow (C_r < 1):
  e = (1 - exp(-NTU * (1 - C_r))) / (1 - C_r * exp(-NTU * (1 - C_r)))

For counter-flow (C_r = 1):
  e = NTU / (1 + NTU)
```

#### Overall Heat Transfer Coefficient

```
1/U = 1/h_o + R_fo + (r_o * ln(r_o/r_i))/(k_w) + R_fi * (r_o/r_i) + (1/h_i) * (r_o/r_i)

Where:
  U     = Overall heat transfer coefficient (W/m2-K)
  h_o   = Outside (shell) film coefficient (W/m2-K)
  h_i   = Inside (tube) film coefficient (W/m2-K)
  R_fo  = Outside fouling resistance (m2-K/W)
  R_fi  = Inside fouling resistance (m2-K/W)
  k_w   = Tube wall thermal conductivity (W/m-K)
  r_o   = Tube outer radius (m)
  r_i   = Tube inner radius (m)
```

### Fouling Models

#### Fouling Resistance Calculation

```
R_f = (1/U_fouled) - (1/U_clean)

Cleanliness Factor (CF) = U_actual / U_clean * 100%

Normalized Fouling Factor = R_f / R_f_design
```

#### Kern-Seaton Asymptotic Model

```
R_f(t) = R_f_max * (1 - exp(-t / tau))

Where:
  R_f(t)   = Fouling resistance at time t (m2-K/W)
  R_f_max  = Asymptotic fouling resistance (m2-K/W)
  tau      = Time constant (hours)
  t        = Operating time (hours)

Time to reach 63.2% of R_f_max = tau
Time to reach 95% of R_f_max = 3 * tau
Time to reach 99% of R_f_max = 5 * tau
```

#### Ebert-Panchal Threshold Fouling Model

```
dR_f/dt = alpha * Re^beta * Pr^gamma * exp(-E_a / (R * T_f)) - c_removal * tau_w * R_f

Where:
  alpha     = Deposition coefficient
  beta      = Reynolds number exponent (typically -0.66)
  gamma     = Prandtl number exponent (typically 0.33)
  E_a       = Activation energy (J/mol)
  R         = Gas constant (8.314 J/mol-K)
  T_f       = Film temperature (K)
  c_removal = Removal coefficient
  tau_w     = Wall shear stress (Pa)
  R_f       = Fouling resistance (m2-K/W)
```

### Economic Calculations

#### Energy Loss Calculation

```
Energy_Loss_kW = Q_design - Q_actual

Where:
  Q_design = U_design * A * LMTD
  Q_actual = U_actual * A * LMTD

Annual_Energy_Loss_MWh = Energy_Loss_kW * Operating_Hours / 1000

Annual_Energy_Cost_Loss = Annual_Energy_Loss_MWh * Electricity_Cost
```

#### Cleaning ROI

```
Cleaning_Investment = Cleaning_Cost + (Downtime_Hours * Downtime_Cost_per_Hour)

Monthly_Savings = (Energy_Loss_per_Month * Energy_Cost) * Expected_Recovery_Percent

Simple_Payback_Days = Cleaning_Investment / (Monthly_Savings / 30)

Annual_ROI = ((Annual_Savings - Cleaning_Investment) / Cleaning_Investment) * 100%

NPV = Sum over n years of (Cash_Flow_n / (1 + Discount_Rate)^n) - Initial_Investment
```

### Reference Standards

| Standard | Description | Application |
|----------|-------------|-------------|
| TEMA Standards, 10th Ed. | Tubular Exchanger Manufacturers Association | Shell-and-tube design |
| API 660, 9th Ed. | Shell-and-Tube Heat Exchangers for Refinery | Refinery exchangers |
| API 661, 7th Ed. | Air-Cooled Heat Exchangers | Air coolers |
| ASME VIII, Div. 1 | Pressure Vessel Code | Mechanical design |
| HTRI Design Manual | Heat Transfer Research Inc. | Thermal design |
| ISO 50001:2018 | Energy Management Systems | Energy optimization |
| Kern-Seaton (1959) | Asymptotic Fouling Model | Fouling prediction |
| Ebert-Panchal (1995) | Threshold Fouling Model | Chemical fouling |

---

## Deployment Instructions

### Docker Deployment

#### Single Container

```bash
# Pull image
docker pull greenlang/gl-014:latest

# Create network
docker network create gl-network

# Start PostgreSQL
docker run -d \
  --name gl014-db \
  --network gl-network \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=gl014 \
  -v gl014-pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg15

# Start Redis
docker run -d \
  --name gl014-redis \
  --network gl-network \
  -v gl014-redisdata:/data \
  redis:7-alpine

# Start GL-014
docker run -d \
  --name gl-014 \
  --network gl-network \
  -p 8000:8000 \
  -p 8001:8001 \
  -e DATABASE_URL=postgresql://postgres:password@gl014-db:5432/gl014 \
  -e REDIS_URL=redis://gl014-redis:6379/0 \
  greenlang/gl-014:latest
```

#### Docker Compose (Production)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  gl-014:
    image: greenlang/gl-014:latest
    restart: always
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/gl014
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  db:
    image: timescale/timescaledb:latest-pg15
    restart: always
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=gl014
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  redisdata:
```

### Kubernetes Deployment

#### Deployment Manifest

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-014
  namespace: greenlang
  labels:
    app: gl-014
    agent: exchanger-pro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-014
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: gl-014
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: gl-014
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: gl-014
          image: greenlang/gl-014:latest
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8000
            - name: metrics
              containerPort: 8001
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: gl-014-secrets
                  key: database-url
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: gl-014-secrets
                  key: redis-url
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /liveness
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /readiness
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          securityContext:
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
```

#### Service Manifest

```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gl-014
  namespace: greenlang
  labels:
    app: gl-014
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: metrics
      port: 8001
      targetPort: 8001
  selector:
    app: gl-014
```

#### Horizontal Pod Autoscaler

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-014
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-014
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Helm Chart Installation

```bash
# Add GreenLang Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Install with custom values
helm install gl-014 greenlang/gl-014 \
  --namespace greenlang \
  --create-namespace \
  --values values.yaml

# values.yaml
replicaCount: 3

image:
  repository: greenlang/gl-014
  tag: latest
  pullPolicy: Always

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

database:
  url: postgresql://user:pass@postgres:5432/gl014

redis:
  url: redis://redis:6379/0

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: gl-014.example.com
      paths:
        - path: /
          pathType: Prefix
```

---

## Monitoring and Observability

### Prometheus Metrics

EXCHANGER-PRO exposes Prometheus metrics on port 8001:

```
# Endpoint
http://localhost:8001/metrics
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `gl014_heat_exchanger_efficiency` | Gauge | Current heat exchanger efficiency (%) |
| `gl014_fouling_factor` | Gauge | Current fouling factor (m2-K/W) |
| `gl014_u_value` | Gauge | Current U-value (W/m2-K) |
| `gl014_days_to_cleaning` | Gauge | Days until recommended cleaning |
| `gl014_cleaning_recommendations_total` | Counter | Total cleaning recommendations |
| `gl014_analysis_duration_seconds` | Histogram | Analysis execution time |
| `gl014_api_requests_total` | Counter | Total API requests |
| `gl014_api_request_duration_seconds` | Histogram | API request duration |

**Prometheus Configuration:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gl-014'
    static_configs:
      - targets: ['gl-014:8001']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Grafana Dashboards

Pre-built Grafana dashboards are available:

1. **Agent Performance Dashboard** - API metrics, response times, error rates
2. **Heat Exchanger Efficiency Dashboard** - Real-time efficiency monitoring
3. **Fouling Trends Dashboard** - Fouling factor trends over time
4. **Cleaning Schedule Dashboard** - Upcoming cleaning schedules
5. **Economic Impact Dashboard** - Cost analysis and savings

**Import Dashboard:**

```bash
# Download dashboard JSON
curl -o gl014-dashboard.json \
  https://raw.githubusercontent.com/greenlang/agents/main/gl-014/grafana/dashboards/main.json

# Import via Grafana API
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
  -d @gl014-dashboard.json
```

### OpenTelemetry Tracing

EXCHANGER-PRO supports distributed tracing via OpenTelemetry:

```yaml
# Configuration
observability:
  tracing:
    enabled: true
    exporter: otlp
    endpoint: "http://jaeger:4317"
    sampling_rate: 0.1
```

**Trace Attributes:**

| Attribute | Description |
|-----------|-------------|
| `exchanger.id` | Heat exchanger identifier |
| `calculation.type` | Type of calculation performed |
| `fouling.factor` | Calculated fouling factor |
| `u.value` | Calculated U-value |
| `provenance.hash` | SHA-256 provenance hash |

### Alerting

**AlertManager Rules:**

```yaml
# alertmanager-rules.yml
groups:
  - name: gl-014-alerts
    rules:
      - alert: CriticalFouling
        expr: gl014_fouling_factor > 0.002
        for: 5m
        labels:
          severity: critical
          agent: gl-014
        annotations:
          summary: "Critical fouling detected on {{ $labels.exchanger_id }}"
          description: "Fouling factor {{ $value }} exceeds critical threshold"

      - alert: LowEfficiency
        expr: gl014_heat_exchanger_efficiency < 70
        for: 15m
        labels:
          severity: warning
          agent: gl-014
        annotations:
          summary: "Low efficiency on {{ $labels.exchanger_id }}"
          description: "Efficiency {{ $value }}% below 70% threshold"

      - alert: OverdueCleaning
        expr: gl014_days_to_cleaning < 0
        for: 1h
        labels:
          severity: warning
          agent: gl-014
        annotations:
          summary: "Overdue cleaning on {{ $labels.exchanger_id }}"
          description: "Cleaning overdue by {{ $value | abs }} days"

      - alert: HighPressureDrop
        expr: gl014_pressure_drop_ratio > 1.5
        for: 10m
        labels:
          severity: critical
          agent: gl-014
        annotations:
          summary: "High pressure drop on {{ $labels.exchanger_id }}"
          description: "Pressure drop {{ $value }}x design value"
```

### Structured Logging

EXCHANGER-PRO uses structured JSON logging:

```json
{
  "timestamp": "2025-12-01T10:30:00.123Z",
  "level": "INFO",
  "logger": "gl014.analysis",
  "message": "Analysis completed",
  "exchanger_id": "HX-001",
  "fouling_factor": 0.00038,
  "u_value": 420.0,
  "performance_status": "good",
  "calculation_time_ms": 45.2,
  "provenance_hash": "sha256:a1b2c3...",
  "correlation_id": "req-abc123",
  "trace_id": "trace-xyz789"
}
```

**Log Levels:**

| Level | Description |
|-------|-------------|
| DEBUG | Detailed debugging information |
| INFO | General operational information |
| WARNING | Warning conditions |
| ERROR | Error conditions |
| CRITICAL | Critical conditions requiring immediate attention |

---

## Security

### Authentication

EXCHANGER-PRO supports multiple authentication methods:

#### JWT Bearer Token

```bash
# Obtain token
curl -X POST https://api.greenlang.io/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=your_id&client_secret=your_secret"

# Use token
curl -X GET https://api.greenlang.io/v1/gl-014/exchangers \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

#### API Key

```bash
curl -X GET https://api.greenlang.io/v1/gl-014/exchangers \
  -H "X-API-Key: your_api_key"
```

### Authorization (RBAC)

Role-based access control with the following roles:

| Role | Permissions |
|------|-------------|
| `viewer` | Read analysis results, view dashboards |
| `operator` | Viewer + trigger manual analysis |
| `engineer` | Operator + modify exchanger configuration |
| `admin` | Full access including user management |

### Encryption

- **In Transit:** TLS 1.3 for all API communications
- **At Rest:** AES-256 encryption for sensitive data
- **Secrets:** Kubernetes secrets or HashiCorp Vault integration

### Audit Logging

All operations are logged with complete audit trails:

```json
{
  "timestamp": "2025-12-01T10:30:00Z",
  "event_type": "analysis.completed",
  "user_id": "user123",
  "exchanger_id": "HX-001",
  "action": "fouling_calculation",
  "result": "success",
  "provenance_hash": "sha256:a1b2c3...",
  "ip_address": "10.0.0.1",
  "user_agent": "GreenLang-SDK/1.0"
}
```

### Security Best Practices

1. **Network Security**
   - Deploy in private subnet
   - Use network policies to restrict traffic
   - Enable firewall rules

2. **Container Security**
   - Run as non-root user
   - Read-only root filesystem
   - Drop all capabilities
   - Use security contexts

3. **Secret Management**
   - Never store secrets in code
   - Use Kubernetes secrets or Vault
   - Rotate secrets regularly

4. **API Security**
   - Enable rate limiting
   - Validate all inputs
   - Use HTTPS only
   - Implement CORS policies

---

## Troubleshooting Guide

### Common Issues

#### Issue: Connection to Process Historian Failed

**Symptoms:**
- "Connection refused" errors
- Timeout when retrieving tags

**Solutions:**

1. Verify network connectivity:
```bash
# Test connectivity to PI server
curl -v https://pi-server.example.com/piwebapi

# Test DNS resolution
nslookup pi-server.example.com
```

2. Check credentials:
```bash
# Verify environment variables
echo $PI_WEB_API_URL
echo $PI_WEB_API_USER
```

3. Review firewall rules:
```bash
# Check if port is open
nc -zv pi-server.example.com 443
```

#### Issue: High Memory Usage

**Symptoms:**
- Container OOMKilled
- Slow response times

**Solutions:**

1. Increase memory limits:
```yaml
resources:
  limits:
    memory: 4Gi
```

2. Optimize batch size:
```yaml
batch:
  max_batch_size: 100  # Reduce from 500
```

3. Clear cache:
```bash
curl -X POST http://localhost:8000/api/v1/admin/cache/clear
```

#### Issue: Incorrect Fouling Calculations

**Symptoms:**
- Negative fouling factors
- Unrealistic U-values

**Solutions:**

1. Verify input data:
   - Check that U_clean > U_fouled
   - Verify temperature data is reasonable
   - Confirm flow rates are positive

2. Check design parameters:
   - Ensure design U-value is set correctly
   - Verify heat transfer area

3. Review provenance:
```bash
# Get calculation provenance
curl http://localhost:8000/api/v1/exchangers/HX-001/analysis?include_provenance=true
```

#### Issue: CMMS Work Orders Not Created

**Symptoms:**
- Cleaning recommendations but no work orders
- CMMS sync errors

**Solutions:**

1. Verify CMMS credentials:
```bash
# Test CMMS connection
curl http://localhost:8000/api/v1/integrations/cmms/test
```

2. Check functional location mapping:
```bash
# Verify equipment exists in CMMS
curl http://localhost:8000/api/v1/integrations/cmms/equipment/HX-001
```

3. Review CMMS logs:
```bash
docker logs gl-014 | grep "cmms"
```

### Diagnostic Commands

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/readiness

# Liveness check
curl http://localhost:8000/liveness

# Self-test
curl http://localhost:8000/api/v1/self-test

# Debug info
curl http://localhost:8000/api/v1/debug/info

# Connection status
curl http://localhost:8000/api/v1/debug/connections
```

### Log Analysis

```bash
# View recent logs
docker logs --tail 100 gl-014

# Filter error logs
docker logs gl-014 2>&1 | grep "ERROR"

# Follow logs in real-time
docker logs -f gl-014

# Export logs for analysis
docker logs gl-014 > gl014-logs.txt 2>&1
```

---

## FAQ

### General Questions

**Q: What heat exchanger types are supported?**

A: EXCHANGER-PRO supports:
- Shell-and-tube (all TEMA types: E, F, G, H, J, K, X)
- Plate heat exchangers
- Air-cooled (fin-fan)
- Spiral heat exchangers
- Double-pipe
- Plate-fin
- Printed circuit (PCHE)

**Q: How accurate are the fouling predictions?**

A: Fouling predictions using the Kern-Seaton model typically achieve 85-95% accuracy when calibrated with historical data from the specific exchanger. The Ebert-Panchal model provides additional insight for chemical fouling mechanisms.

**Q: Can EXCHANGER-PRO work without a process historian?**

A: Yes. While process historian integration enables real-time monitoring, EXCHANGER-PRO can also accept manual data entry through the API or batch file uploads.

**Q: How often should I run analyses?**

A: Recommended frequencies:
- Real-time monitoring: Every 5 minutes
- Full performance analysis: Every 1-4 hours
- Economic impact analysis: Daily
- Cleaning schedule optimization: Weekly

### Technical Questions

**Q: How are calculations made deterministic?**

A: All calculations use Python's `Decimal` class for arbitrary-precision arithmetic, ensuring bit-perfect reproducibility. The random seed is fixed (42), and no LLM is involved in the calculation path.

**Q: What is a provenance hash?**

A: Every calculation result includes a SHA-256 hash of all inputs, calculation steps, and outputs. This enables complete audit trails and verification that results have not been tampered with.

**Q: How does caching work?**

A: EXCHANGER-PRO uses Redis for caching with configurable TTL (default: 5 minutes). Cache keys are based on input parameter hashes. Cache can be bypassed using the `no_cache` query parameter.

**Q: What database is required?**

A: PostgreSQL 14+ is required. TimescaleDB extension is recommended for optimal time-series data storage. Redis 6+ is required for caching.

### Integration Questions

**Q: Which process historians are supported?**

A: Native integrations are available for:
- OSIsoft PI (PI Web API)
- Honeywell PHD
- AspenTech IP.21
- Generic OPC-UA

**Q: Which CMMS systems are supported?**

A: Native integrations are available for:
- SAP Plant Maintenance (PM)
- IBM Maximo
- Oracle Enterprise Asset Management (EAM)

**Q: Can I integrate with other systems?**

A: Yes. EXCHANGER-PRO provides:
- REST API for custom integrations
- Webhook support for event notifications
- MQTT for IoT integration
- File-based batch import/export

### Pricing and Licensing

**Q: What license is EXCHANGER-PRO released under?**

A: EXCHANGER-PRO is released under the Apache-2.0 license for the open-source version. Enterprise features require a commercial license.

**Q: Is there a free tier?**

A: Yes. The open-source version includes all core calculation capabilities. The enterprise version adds advanced features like multi-tenant support, SSO, and premium integrations.

---

## Compliance and Standards

### Industry Standards

EXCHANGER-PRO calculations comply with:

| Standard | Description | Compliance Level |
|----------|-------------|------------------|
| TEMA Standards, 10th Ed. | Tubular Exchanger Manufacturers Association | Full |
| API 660, 9th Ed. | Shell-and-Tube Heat Exchangers | Full |
| API 661, 7th Ed. | Air-Cooled Heat Exchangers | Full |
| ASME Section VIII | Pressure Vessel Code | Reference |
| HTRI Design Manual | Heat Transfer Research Inc. | Reference |
| ISO 50001:2018 | Energy Management Systems | Aligned |

### Audit Trail Compliance

EXCHANGER-PRO maintains complete audit trails:

- **Provenance Tracking:** SHA-256 hashes for all calculations
- **Data Retention:** Configurable retention (default: 7 years)
- **Immutable Records:** Frozen dataclasses prevent modification
- **Timestamping:** UTC timestamps for all records
- **User Attribution:** All actions attributed to authenticated users

### Data Privacy

- **Data Residency:** Configurable deployment location
- **Data Encryption:** AES-256 at rest, TLS 1.3 in transit
- **Access Control:** RBAC with audit logging
- **Data Deletion:** Support for data retention policies

---

## Contributing

We welcome contributions to EXCHANGER-PRO! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/agents.git
cd agents/gl-014

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check .
mypy .
```

### Code Style

- Python code follows PEP 8 and is enforced by `ruff`
- Type hints are required for all functions
- Docstrings follow Google style
- All calculations must include provenance tracking

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

### Recent Changes

**v1.0.0 (2025-12-01)**
- Initial release
- Heat transfer calculations (LMTD, NTU, U-value)
- Fouling analysis (Kern-Seaton, Ebert-Panchal)
- Cleaning schedule optimization
- Economic impact analysis
- Process historian integration (PI, PHD, IP.21, OPC-UA)
- CMMS integration (SAP, Maximo, Oracle EAM)
- Prometheus metrics and OpenTelemetry tracing
- Zero-hallucination provenance tracking

---

## Support

### Documentation

- **User Guide:** https://docs.greenlang.io/agents/gl-014
- **API Reference:** https://api.greenlang.io/docs/gl-014
- **Tutorials:** https://learn.greenlang.io/gl-014

### Community

- **GitHub Issues:** https://github.com/greenlang/agents/issues
- **Discussions:** https://github.com/greenlang/agents/discussions
- **Discord:** https://discord.gg/greenlang

### Enterprise Support

- **Email:** support@greenlang.io
- **Portal:** https://support.greenlang.io
- **SLA:** Response within 4 hours (business hours)

---

## Authors

- **GreenLang AI Agent Factory** - agents@greenlang.io

## Repository

- **GitHub:** https://github.com/greenlang/agents/gl-014
- **Documentation:** https://docs.greenlang.io/agents/gl-014

---

## License

```
Copyright 2025 GreenLang

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

**GL-014 EXCHANGER-PRO** - Industrial Heat Exchanger Optimization with Zero-Hallucination Guarantee

*Built by GreenLang for Industrial Sustainability*
