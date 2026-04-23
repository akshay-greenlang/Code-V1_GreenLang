# GL-017 CONDENSYNC API Reference

## Base URL

```
http://localhost:8017/api/v1
```

## Authentication

API key authentication via header:

```
Authorization: Bearer <api_key>
```

## Endpoints

### Health and Status

#### GET /health

Liveness probe for Kubernetes.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T10:30:00Z"
}
```

#### GET /health/ready

Readiness probe for Kubernetes.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T10:30:00Z",
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 2.5
    },
    {
      "name": "hei_calculator",
      "status": "healthy"
    },
    {
      "name": "opc_ua_connector",
      "status": "healthy",
      "latency_ms": 15.2
    }
  ]
}
```

#### GET /metrics

Prometheus metrics endpoint.

**Response:** OpenMetrics format

```
# HELP condensync_cleanliness_factor Condenser cleanliness factor percentage
# TYPE condensync_cleanliness_factor gauge
condensync_cleanliness_factor{unit="Unit1",condenser="A"} 82.5
condensync_cleanliness_factor{unit="Unit1",condenser="B"} 79.3

# HELP condensync_ttd_kelvin Terminal Temperature Difference in Kelvin
# TYPE condensync_ttd_kelvin gauge
condensync_ttd_kelvin{unit="Unit1",condenser="A"} 5.2

# HELP condensync_diagnoses_total Total diagnoses performed
# TYPE condensync_diagnoses_total counter
condensync_diagnoses_total{condition="light_fouling",severity="moderate"} 1523
```

#### GET /status

Agent operational status.

**Response:**

```json
{
  "agent_id": "GL-017",
  "agent_name": "CONDENSYNC",
  "version": "1.0.0",
  "status": "running",
  "mode": "monitoring",
  "uptime_seconds": 86400,
  "statistics": {
    "total_analyses": 15230,
    "fouling_alerts": 142,
    "avg_cleanliness_factor": 81.5,
    "total_heat_rate_penalty_btu_kwh": 45.2
  },
  "connections": {
    "opc_ua": "connected",
    "historian": "connected",
    "cmms": "connected"
  }
}
```

---

### Analysis Endpoints

#### POST /analyze

Analyze a single condenser performance.

**Request:**

```json
{
  "condenser_id": "COND-001",
  "unit_id": "Unit1",
  "timestamp": "2025-12-30T10:30:00Z",
  "process_data": {
    "condenser_pressure_kpa": 5.0,
    "hotwell_temperature_c": 33.0,
    "steam_flow_kg_s": 250.0,
    "heat_duty_mw": 500.0
  },
  "cooling_water": {
    "inlet_temperature_c": 20.0,
    "outlet_temperature_c": 30.0,
    "flow_rate_m3_s": 12.0,
    "velocity_m_s": 2.1
  },
  "design_data": {
    "surface_area_m2": 25000,
    "tube_od_mm": 25.4,
    "tube_material": "titanium",
    "tube_count": 18000,
    "design_u_w_m2k": 3200,
    "design_pressure_kpa": 4.5,
    "design_ttd_k": 3.0
  },
  "include_optimization": true,
  "include_explanation": true
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `condenser_id` | string | Yes | Unique condenser identifier |
| `unit_id` | string | Yes | Power unit identifier |
| `timestamp` | string | No | ISO 8601 timestamp (defaults to current) |
| `process_data` | object | Yes | Process measurements |
| `cooling_water` | object | Yes | Cooling water measurements |
| `design_data` | object | Yes | Design specifications |
| `include_optimization` | boolean | No | Include optimization recommendations (default: false) |
| `include_explanation` | boolean | No | Include explainability data (default: true) |

**Process Data Fields:**

| Field | Type | Unit | Valid Range | Description |
|-------|------|------|-------------|-------------|
| `condenser_pressure_kpa` | float | kPa | 0.5-15 | Absolute pressure at condenser |
| `hotwell_temperature_c` | float | C | 20-60 | Hotwell/condensate temperature |
| `steam_flow_kg_s` | float | kg/s | 0-1000 | Exhaust steam flow rate |
| `heat_duty_mw` | float | MW | 0-2000 | Total heat rejection |

**Cooling Water Fields:**

| Field | Type | Unit | Valid Range | Description |
|-------|------|------|-------------|-------------|
| `inlet_temperature_c` | float | C | 0-45 | CW inlet temperature |
| `outlet_temperature_c` | float | C | 5-55 | CW outlet temperature |
| `flow_rate_m3_s` | float | m3/s | 0-100 | Volumetric flow rate |
| `velocity_m_s` | float | m/s | 0.3-3.0 | Tube-side velocity |

**Response (200 OK):**

```json
{
  "condenser_id": "COND-001",
  "unit_id": "Unit1",
  "analysis_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "timestamp": "2025-12-30T10:30:00Z",
  "performance": {
    "cleanliness_factor_pct": 82.5,
    "actual_u_w_m2k": 2640,
    "design_u_w_m2k": 3200,
    "ttd_k": 5.2,
    "dca_k": 3.0,
    "lmtd_k": 8.5,
    "heat_duty_mw": 500.0,
    "fouling_resistance_m2k_w": 0.000065
  },
  "condition": {
    "state": "light_fouling",
    "severity": "moderate",
    "confidence": 0.92,
    "trend": "degrading",
    "days_to_threshold": 45
  },
  "impact": {
    "backpressure_deviation_kpa": 0.5,
    "heat_rate_penalty_btu_kwh": 25.5,
    "heat_rate_penalty_pct": 0.28,
    "annual_fuel_cost_usd": 125000,
    "annual_co2_tonnes": 850
  },
  "optimization": {
    "recommended_action": "schedule_cleaning",
    "optimal_cleaning_date": "2025-02-15",
    "cleaning_roi_months": 2.5,
    "cw_flow_recommendation": {
      "current_m3_s": 12.0,
      "optimal_m3_s": 13.5,
      "pump_power_impact_kw": 45,
      "net_benefit_kw": 120
    }
  },
  "explainability": {
    "top_factors": [
      {
        "factor": "cleanliness_factor",
        "value": 82.5,
        "contribution": 0.45,
        "description": "CF below 85% indicates fouling buildup"
      },
      {
        "factor": "ttd_elevation",
        "value": 2.2,
        "contribution": 0.30,
        "description": "TTD 2.2K above design indicates reduced heat transfer"
      },
      {
        "factor": "trend_rate",
        "value": -0.5,
        "contribution": 0.25,
        "description": "CF declining at 0.5%/week suggests active fouling"
      }
    ],
    "evidence_chain": [
      {
        "step": 1,
        "type": "observation",
        "data": "Cleanliness factor at 82.5%",
        "inference": "Below 85% threshold for clean condition"
      },
      {
        "step": 2,
        "type": "observation",
        "data": "TTD at 5.2K vs design 3.0K",
        "inference": "Elevated TTD indicates fouling resistance"
      },
      {
        "step": 3,
        "type": "calculation",
        "data": "Rf = 0.000065 m2K/W",
        "inference": "Fouling resistance consistent with light biological fouling"
      },
      {
        "step": 4,
        "type": "recommendation",
        "data": "45 days to CF threshold of 75%",
        "inference": "Schedule cleaning before severe fouling develops"
      }
    ],
    "counterfactual": "If cleanliness factor were above 85%, condition would be CLEAN"
  },
  "provenance": {
    "input_hash": "sha256:a1b2c3d4e5f6...",
    "output_hash": "sha256:f6e5d4c3b2a1...",
    "calculation_version": "HEI_3098_v11",
    "calculator_hashes": {
      "hei_calculator": "sha256:abc123...",
      "lmtd_calculator": "sha256:def456..."
    }
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `performance` | object | Calculated performance metrics |
| `condition` | object | Diagnostic classification |
| `impact` | object | Economic and environmental impact |
| `optimization` | object | Optimization recommendations (if requested) |
| `explainability` | object | Factor attribution and evidence chain |
| `provenance` | object | Audit trail with calculation hashes |

---

#### POST /analyze/batch

Analyze multiple condensers in a single request.

**Request:**

```json
{
  "analyses": [
    {
      "condenser_id": "COND-001",
      "unit_id": "Unit1",
      "process_data": { ... },
      "cooling_water": { ... },
      "design_data": { ... }
    },
    {
      "condenser_id": "COND-002",
      "unit_id": "Unit1",
      "process_data": { ... },
      "cooling_water": { ... },
      "design_data": { ... }
    }
  ],
  "include_fleet_summary": true
}
```

**Response:**

```json
{
  "batch_id": "batch-a1b2c3d4...",
  "timestamp": "2025-12-30T10:30:00Z",
  "analyses": [
    { ... },
    { ... }
  ],
  "fleet_summary": {
    "total_condensers": 2,
    "clean_count": 0,
    "light_fouling_count": 1,
    "moderate_fouling_count": 1,
    "severe_fouling_count": 0,
    "avg_cleanliness_factor_pct": 78.5,
    "total_heat_rate_penalty_btu_kwh": 52.3,
    "total_annual_cost_usd": 285000,
    "total_annual_co2_tonnes": 1950
  }
}
```

---

### Optimization Endpoints

#### POST /optimize/cleaning

Calculate optimal cleaning schedule.

**Request:**

```json
{
  "condenser_id": "COND-001",
  "unit_id": "Unit1",
  "current_cf_pct": 82.5,
  "fouling_rate_pct_day": 0.07,
  "cleaning_cost_usd": 25000,
  "cleaning_duration_hours": 48,
  "fuel_cost_usd_mmbtu": 3.50,
  "co2_price_usd_tonne": 50,
  "heat_rate_design_btu_kwh": 9000,
  "capacity_mw": 500,
  "capacity_factor_pct": 85
}
```

**Response:**

```json
{
  "condenser_id": "COND-001",
  "optimal_cleaning_date": "2025-02-15",
  "days_until_optimal": 47,
  "cf_at_optimal_date_pct": 79.2,
  "economic_analysis": {
    "current_heat_rate_penalty_usd_yr": 125000,
    "penalty_at_cleaning_usd": 185000,
    "cleaning_cost_usd": 25000,
    "lost_generation_usd": 35000,
    "total_cleaning_cost_usd": 60000,
    "net_annual_benefit_usd": 310000,
    "simple_payback_months": 2.3
  },
  "sensitivity": {
    "early_cleaning_30d": {
      "net_benefit_usd": 280000,
      "payback_months": 2.6
    },
    "delayed_cleaning_30d": {
      "net_benefit_usd": 295000,
      "payback_months": 2.4
    }
  },
  "provenance_hash": "sha256:abc123..."
}
```

---

#### POST /optimize/cooling-water

Optimize cooling water flow rate.

**Request:**

```json
{
  "condenser_id": "COND-001",
  "unit_id": "Unit1",
  "current_flow_m3_s": 12.0,
  "pump_curve": {
    "flow_m3_s": [8, 10, 12, 14, 16],
    "head_m": [45, 42, 38, 33, 27],
    "efficiency_pct": [75, 80, 82, 80, 75]
  },
  "motor_efficiency_pct": 95,
  "electricity_price_usd_mwh": 50,
  "condenser_data": {
    "heat_duty_mw": 500,
    "cw_inlet_temp_c": 20,
    "design_u_w_m2k": 3200,
    "surface_area_m2": 25000,
    "tube_count": 18000,
    "tube_id_mm": 22.9
  }
}
```

**Response:**

```json
{
  "condenser_id": "COND-001",
  "current_state": {
    "flow_m3_s": 12.0,
    "pump_power_kw": 1850,
    "backpressure_kpa": 5.0,
    "ttd_k": 5.2
  },
  "optimal_state": {
    "flow_m3_s": 13.5,
    "pump_power_kw": 2150,
    "backpressure_kpa": 4.7,
    "ttd_k": 4.5
  },
  "impact": {
    "pump_power_increase_kw": 300,
    "backpressure_reduction_kpa": 0.3,
    "heat_rate_improvement_btu_kwh": 8.5,
    "generation_increase_kw": 425,
    "net_benefit_kw": 125,
    "annual_benefit_usd": 54750
  },
  "recommendation": "Increase CW flow to 13.5 m3/s for net benefit of 125 kW",
  "provenance_hash": "sha256:def456..."
}
```

---

### Configuration

#### GET /config

Get current configuration.

**Response:**

```json
{
  "agent_id": "GL-017",
  "mode": "monitoring",
  "analysis_interval_seconds": 300,
  "thresholds": {
    "cf_clean_pct": 85,
    "cf_light_fouling_pct": 75,
    "cf_moderate_fouling_pct": 60,
    "ttd_warning_k": 5,
    "ttd_critical_k": 10
  },
  "hei_standard": {
    "version": "HEI_3098_11th_Edition",
    "correction_factors_enabled": true
  },
  "integrations": {
    "opc_ua": {
      "enabled": true,
      "endpoint": "opc.tcp://localhost:4840"
    },
    "historian": {
      "enabled": true,
      "type": "osisoft_pi"
    }
  }
}
```

#### PUT /config

Update configuration (requires admin role).

**Request:**

```json
{
  "mode": "survey",
  "thresholds": {
    "cf_clean_pct": 90
  }
}
```

---

## Error Responses

All errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://greenlang.io/errors/validation",
  "title": "Validation Error",
  "status": 400,
  "detail": "condenser_pressure_kpa=20.0 exceeds maximum 15.0 kPa",
  "instance": "/api/v1/analyze",
  "violations": [
    {
      "parameter": "condenser_pressure_kpa",
      "value": 20.0,
      "min_bound": 0.5,
      "max_bound": 15.0,
      "unit": "kPa",
      "standard": "HEI 3098"
    }
  ]
}
```

### Error Codes

| Status | Type | Description |
|--------|------|-------------|
| 400 | validation | Input validation failed |
| 401 | unauthorized | Missing or invalid API key |
| 403 | forbidden | Insufficient permissions |
| 404 | not_found | Resource not found |
| 422 | bounds_violation | Physical bounds exceeded |
| 429 | rate_limited | Too many requests |
| 500 | internal | Internal server error |
| 503 | unavailable | Service temporarily unavailable |

---

## Rate Limits

| Tier | Requests/min | Burst |
|------|--------------|-------|
| Standard | 100 | 20 |
| Premium | 1000 | 100 |

Rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1735556200
```

---

## Webhooks

Configure webhooks for real-time alerts:

```json
POST /webhooks
{
  "url": "https://your-server.com/condensync-alerts",
  "events": ["fouling_alert", "critical_backpressure", "cleaning_due"],
  "secret": "your-webhook-secret"
}
```

Webhook payload:

```json
{
  "event": "fouling_alert",
  "timestamp": "2025-12-30T10:30:00Z",
  "condenser_id": "COND-001",
  "unit_id": "Unit1",
  "condition": "moderate_fouling",
  "severity": "high",
  "cleanliness_factor_pct": 72.5,
  "heat_rate_penalty_btu_kwh": 45.2,
  "recommended_action": "Schedule cleaning within 2 weeks",
  "signature": "sha256=..."
}
```

---

## SDKs

- Python: `pip install greenlang-condensync`
- JavaScript: `npm install @greenlang/condensync`
- Go: `go get github.com/greenlang/condensync-go`

### Python SDK Example

```python
from greenlang import CondensyncClient

client = CondensyncClient(
    api_key="your_api_key",
    base_url="https://condensync.greenlang.io/api/v1"
)

# Analyze condenser
result = client.analyze(
    condenser_id="COND-001",
    unit_id="Unit1",
    process_data={
        "condenser_pressure_kpa": 5.0,
        "hotwell_temperature_c": 33.0,
        "steam_flow_kg_s": 250.0,
        "heat_duty_mw": 500.0
    },
    cooling_water={
        "inlet_temperature_c": 20.0,
        "outlet_temperature_c": 30.0,
        "flow_rate_m3_s": 12.0,
        "velocity_m_s": 2.1
    },
    design_data={
        "surface_area_m2": 25000,
        "tube_od_mm": 25.4,
        "tube_material": "titanium",
        "tube_count": 18000,
        "design_u_w_m2k": 3200
    }
)

print(f"Cleanliness Factor: {result.performance.cleanliness_factor_pct}%")
print(f"Condition: {result.condition.state}")
print(f"Heat Rate Penalty: {result.impact.heat_rate_penalty_btu_kwh} BTU/kWh")
```

### cURL Example

```bash
curl -X POST "https://condensync.greenlang.io/api/v1/analyze" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "condenser_id": "COND-001",
    "unit_id": "Unit1",
    "process_data": {
      "condenser_pressure_kpa": 5.0,
      "hotwell_temperature_c": 33.0,
      "steam_flow_kg_s": 250.0,
      "heat_duty_mw": 500.0
    },
    "cooling_water": {
      "inlet_temperature_c": 20.0,
      "outlet_temperature_c": 30.0,
      "flow_rate_m3_s": 12.0,
      "velocity_m_s": 2.1
    },
    "design_data": {
      "surface_area_m2": 25000,
      "tube_od_mm": 25.4,
      "tube_material": "titanium",
      "tube_count": 18000,
      "design_u_w_m2k": 3200
    }
  }'
```
