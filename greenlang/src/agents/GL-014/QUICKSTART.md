# GL-014 EXCHANGER-PRO Quick Start Guide

## Get Started in 5 Minutes

This guide will help you get GL-014 EXCHANGER-PRO running and analyzing your first heat exchanger in under 5 minutes.

---

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.11+ installed
- [ ] Docker and Docker Compose (recommended) OR local PostgreSQL/Redis
- [ ] API credentials (if using GreenLang Cloud)

**Check your Python version:**

```bash
python --version
# Should output: Python 3.11.x or higher
```

---

## Installation Options

Choose one of the following installation methods:

### Option A: Docker Compose (Recommended)

The fastest way to get started with all dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/greenlang/agents.git
cd agents/gl-014

# 2. Start all services
docker-compose up -d

# 3. Verify services are running
docker-compose ps
```

**Expected output:**

```
NAME                SERVICE     STATUS
gl-014-api          gl-014      running (healthy)
gl-014-db           db          running (healthy)
gl-014-redis        redis       running (healthy)
```

### Option B: Local Installation

Install directly with pip for development.

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install GL-014
pip install greenlang-gl014

# 3. Set environment variables
export DATABASE_URL=postgresql://user:password@localhost:5432/gl014
export REDIS_URL=redis://localhost:6379/0

# 4. Run the agent
uvicorn gl014.main:app --host 0.0.0.0 --port 8000
```

### Option C: GreenLang Cloud (SaaS)

No installation required - use our hosted API.

```bash
# Get your API key from https://console.greenlang.io

export GREENLANG_API_KEY=your_api_key
```

---

## Step 1: Verify Installation

Check that GL-014 is running correctly:

```bash
curl http://localhost:8000/health
```

**Expected response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agent": "GL-014",
  "codename": "EXCHANGER-PRO",
  "timestamp": "2025-12-01T10:30:00Z"
}
```

---

## Step 2: Your First API Call

Analyze a heat exchanger's performance with a simple API call.

### Using cURL

```bash
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

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
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
    }
)

result = response.json()
print(f"Current U-value: {result['performance_metrics']['current_u_w_m2k']} W/m2K")
print(f"Fouling State: {result['fouling_analysis']['fouling_state']}")
print(f"Recommended Cleaning: {result['cleaning_schedule']['recommended_cleaning_date']}")
```

---

## Step 3: Understanding the Response

Here's what the analysis returns:

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

### Key Metrics Explained

| Metric | Description | Good Range |
|--------|-------------|------------|
| `u_ratio` | Current U / Design U | > 0.85 |
| `fouling_state` | Fouling severity level | clean, light, moderate |
| `cleaning_roi_percent` | Return on cleaning investment | > 100% |
| `provenance_hash` | Audit trail verification hash | Always present |

---

## Step 4: Configure Your Exchanger

Register a heat exchanger with full specifications:

```bash
curl -X POST http://localhost:8000/api/v1/exchangers \
  -H "Content-Type: application/json" \
  -d '{
    "exchanger_id": "HX-001",
    "exchanger_name": "Crude Preheat #1",
    "exchanger_type": "shell_and_tube",
    "flow_arrangement": "counterflow",
    "design_heat_duty_kw": 5000.0,
    "design_u_w_m2k": 450.0,
    "clean_u_w_m2k": 500.0,
    "heat_transfer_area_m2": 250.0,
    "shell_type": "E",
    "number_of_passes": 2,
    "hot_fluid": {
      "name": "Crude Oil",
      "fluid_type": "crude_oil",
      "cp_j_kgk": 2100.0
    },
    "cold_fluid": {
      "name": "Desalter Effluent",
      "fluid_type": "crude_oil",
      "cp_j_kgk": 2000.0
    }
  }'
```

---

## Step 5: Set Up Monitoring (Optional)

### Enable Prometheus Metrics

Metrics are automatically exposed on port 8001:

```bash
curl http://localhost:8001/metrics
```

### Add to Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gl-014'
    static_configs:
      - targets: ['localhost:8001']
```

### Import Grafana Dashboard

```bash
# Download and import the pre-built dashboard
curl -o gl014-dashboard.json \
  https://raw.githubusercontent.com/greenlang/agents/main/gl-014/grafana/dashboards/main.json
```

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@localhost:5432/gl014` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_PORT` | API server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ELECTRICITY_COST_PER_KWH` | Energy cost for economics | `0.10` |
| `DOWNTIME_COST_PER_HOUR` | Downtime cost | `5000.0` |

### Example .env File

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/gl014

# Cache
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Economics
ELECTRICITY_COST_PER_KWH=0.10
STEAM_COST_PER_KG=0.03
DOWNTIME_COST_PER_HOUR=5000.0
```

---

## Common API Calls

### Get Fouling Analysis

```bash
curl http://localhost:8000/api/v1/exchangers/HX-001/fouling
```

### Get Cleaning Schedule

```bash
curl http://localhost:8000/api/v1/exchangers/HX-001/cleaning-schedule
```

### Get Economic Impact

```bash
curl http://localhost:8000/api/v1/exchangers/HX-001/economic-impact
```

### Batch Analysis (Multiple Exchangers)

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "exchangers": [
      {"exchanger_id": "HX-001", ...},
      {"exchanger_id": "HX-002", ...},
      {"exchanger_id": "HX-003", ...}
    ]
  }'
```

---

## Troubleshooting

### Issue: Connection Refused

```
curl: (7) Failed to connect to localhost port 8000
```

**Solution:** Ensure the service is running:

```bash
# Docker
docker-compose ps
docker-compose logs gl-014

# Local
ps aux | grep uvicorn
```

### Issue: Database Connection Failed

```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:** Check database URL and connectivity:

```bash
# Verify PostgreSQL is running
docker-compose ps db

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

### Issue: Invalid Temperature Data

```json
{
  "error": "validation_error",
  "message": "Hot inlet temperature must be greater than hot outlet temperature"
}
```

**Solution:** Verify your temperature values:

- Hot inlet > Hot outlet (heat is released)
- Cold outlet > Cold inlet (heat is absorbed)
- Temperature difference is reasonable (5-150 K typical)

### Issue: Negative Fouling Factor

```json
{
  "fouling_analysis": {
    "total_fouling_factor": -0.00012
  }
}
```

**Solution:** This indicates U_actual > U_clean, which is physically impossible. Check:

- Design U-value is set correctly
- Temperature measurements are accurate
- Flow measurements are accurate

---

## Next Steps

Now that you have GL-014 running:

1. **Connect to Process Historian**
   - See [Integration Guide](README.md#integration-guide) for PI, PHD, IP.21

2. **Set Up CMMS Integration**
   - Automate work order creation in SAP, Maximo, or Oracle EAM

3. **Configure Alerts**
   - Set up AlertManager rules for fouling thresholds

4. **Deploy to Production**
   - See [Deployment Instructions](README.md#deployment-instructions)

5. **Explore the Full API**
   - Visit http://localhost:8000/docs for interactive API documentation

---

## Getting Help

- **Documentation:** [https://docs.greenlang.io/agents/gl-014](https://docs.greenlang.io/agents/gl-014)
- **GitHub Issues:** [https://github.com/greenlang/agents/issues](https://github.com/greenlang/agents/issues)
- **Community Discord:** [https://discord.gg/greenlang](https://discord.gg/greenlang)
- **Enterprise Support:** support@greenlang.io

---

## Quick Reference Card

```
+------------------------------------------------------------------+
|              GL-014 EXCHANGER-PRO Quick Reference                 |
+------------------------------------------------------------------+
|                                                                  |
|  START:      docker-compose up -d                                |
|  STOP:       docker-compose down                                 |
|  LOGS:       docker-compose logs -f gl-014                       |
|  STATUS:     curl localhost:8000/health                          |
|                                                                  |
|  ENDPOINTS:                                                      |
|    POST /api/v1/analyze              - Analyze exchanger         |
|    POST /api/v1/analyze/batch        - Batch analysis            |
|    GET  /api/v1/exchangers           - List exchangers           |
|    POST /api/v1/exchangers           - Register exchanger        |
|    GET  /api/v1/exchangers/{id}/fouling     - Fouling data       |
|    GET  /api/v1/exchangers/{id}/cleaning    - Cleaning schedule  |
|    GET  /api/v1/exchangers/{id}/economics   - Economic impact    |
|                                                                  |
|  DOCS:       http://localhost:8000/docs                          |
|  METRICS:    http://localhost:8001/metrics                       |
|                                                                  |
+------------------------------------------------------------------+
```

---

*GL-014 EXCHANGER-PRO - Industrial Heat Exchanger Optimization*
*Copyright 2025 GreenLang*
