# GL-008 TRAPCATCHER - API Reference Documentation

**Agent:** GL-008 SteamTrapInspector
**Version:** 1.0.0
**API Version:** v1
**Status:** Production Ready
**Last Updated:** 2025-01-22

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL & Endpoints](#base-url--endpoints)
4. [Common Request/Response Format](#common-requestresponse-format)
5. [Tool APIs](#tool-apis)
   - [analyze_acoustic_signature](#analyze_acoustic_signature)
   - [analyze_thermal_pattern](#analyze_thermal_pattern)
   - [diagnose_trap_failure](#diagnose_trap_failure)
   - [calculate_energy_loss](#calculate_energy_loss)
   - [prioritize_maintenance](#prioritize_maintenance)
   - [predict_remaining_useful_life](#predict_remaining_useful_life)
   - [calculate_cost_benefit](#calculate_cost_benefit)
6. [Operation Modes](#operation-modes)
7. [Error Codes](#error-codes)
8. [Rate Limiting](#rate-limiting)
9. [Webhooks](#webhooks)
10. [Code Examples](#code-examples)
11. [SDK Reference](#sdk-reference)
12. [Changelog](#changelog)

---

## Overview

The TRAPCATCHER API provides programmatic access to all steam trap inspection, diagnosis, and optimization capabilities. All endpoints return deterministic results for the same inputs (temperature=0.0, seed=42).

**Key Features**:
- RESTful JSON API
- Deterministic calculations (zero hallucination)
- Industry-standard physics equations
- Comprehensive error handling
- Full provenance tracking

**API Design Principles**:
- **Idempotent**: Same request returns same result
- **Stateless**: No server-side session management
- **Versioned**: `/v1/` prefix for backward compatibility
- **Self-Documenting**: OpenAPI 3.0 spec available at `/openapi.json`

---

## Authentication

### API Key Authentication

**Method**: HTTP Header

**Header**: `X-API-Key: <your_api_key>`

**Example**:
```bash
curl -H "X-API-Key: gl_sk_abc123..." \
     https://api.greenlang.org/agents/gl-008/v1/analyze_acoustic
```

**Obtaining API Key**:
1. Log in to GreenLang console: https://console.greenlang.org
2. Navigate to Settings > API Keys
3. Click "Create API Key"
4. Copy key (shown only once)

### OAuth 2.0 (Enterprise)

**Grant Type**: Client Credentials

**Token Endpoint**: `https://auth.greenlang.org/oauth/token`

**Request**:
```bash
curl -X POST https://auth.greenlang.org/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=your_client_id" \
  -d "client_secret=your_client_secret" \
  -d "scope=agents:read agents:write"
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "agents:read agents:write"
}
```

**Usage**:
```bash
curl -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \
     https://api.greenlang.org/agents/gl-008/v1/analyze_acoustic
```

---

## Base URL & Endpoints

**Production**: `https://api.greenlang.org/agents/gl-008/v1`

**Staging**: `https://staging-api.greenlang.org/agents/gl-008/v1`

**Local Development**: `http://localhost:8080/v1`

### Health Check Endpoints

#### GET /health

**Description**: Basic health check (liveness probe)

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-22T10:30:00Z"
}
```

#### GET /ready

**Description**: Readiness check (dependencies available)

**Response**:
```json
{
  "status": "ready",
  "dependencies": {
    "database": "connected",
    "cache": "connected",
    "ml_models": "loaded"
  },
  "timestamp": "2025-01-22T10:30:00Z"
}
```

#### GET /metrics

**Description**: Prometheus metrics endpoint

**Response**: Prometheus text format

---

## Common Request/Response Format

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `X-API-Key` | Yes* | API key (*or Authorization) |
| `Authorization` | Yes* | Bearer token (*or X-API-Key) |
| `X-Request-ID` | No | Idempotency key (recommended) |
| `X-Provenance-Track` | No | `true` to enable provenance recording |

### Standard Response Structure

**Success (HTTP 200)**:
```json
{
  "status": "success",
  "data": {
    // Tool-specific output
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2025-01-22T10:30:00Z",
    "execution_time_ms": 1234,
    "agent_version": "1.0.0"
  },
  "provenance": {
    // Optional, if X-Provenance-Track: true
    "input_hash": "sha256:...",
    "output_hash": "sha256:...",
    "computation_dag": "..."
  }
}
```

**Error (HTTP 4xx/5xx)**:
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid steam_pressure_psig",
    "details": {
      "field": "steam_pressure_psig",
      "value": -10,
      "constraint": "Must be >= 0 and <= 600"
    }
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2025-01-22T10:30:00Z"
  }
}
```

---

## Tool APIs

### analyze_acoustic_signature

**Endpoint**: `POST /v1/analyze_acoustic`

**Description**: Analyze ultrasonic acoustic signature for steam trap failure detection using FFT and spectral analysis.

**Physics Basis**: ASTM E1316 Ultrasonic Testing, ISO 18436-8 Condition Monitoring

**Deterministic**: Yes

#### Request Body

```json
{
  "trap_id": "TRAP-001",
  "signal": [0.012, 0.015, 0.011, ...],  // Array of 10,000+ samples
  "sampling_rate_hz": 250000,
  "trap_type": "thermodynamic"
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `trap_id` | string | Yes | Max 50 chars | Unique trap identifier |
| `signal` | array[number] | Yes | Length 1000-1000000 | Time-domain acoustic signal |
| `sampling_rate_hz` | integer | Yes | 100000-500000 | Signal sampling rate |
| `trap_type` | string | No | Enum (see below) | Trap type for specialized analysis |

**trap_type Enum**:
- `"thermostatic"`
- `"mechanical"`
- `"thermodynamic"`
- `"inverted_bucket"`
- `"float_thermostatic"`
- `"disc"`
- `"bimetallic"`
- `"balanced_pressure"`

#### Response

```json
{
  "status": "success",
  "data": {
    "trap_id": "TRAP-001",
    "failure_probability": 0.87,
    "failure_mode": "failed_open",
    "confidence_score": 0.92,
    "spectral_features": {
      "peak_frequency_hz": 68500,
      "energy_density": 0.042,
      "harmonic_ratio": 0.18,
      "broadband_noise": 0.65
    },
    "analysis_method": "fft_anomaly_detection",
    "model_version": "acoustic_v1.2.0"
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2025-01-22T10:30:00Z",
    "execution_time_ms": 1150
  }
}
```

#### Response Schema

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `failure_probability` | number | 0.0-1.0 | Probability trap has failed (0=normal, 1=failed) |
| `failure_mode` | string | Enum | Detected failure mode |
| `confidence_score` | number | 0.0-1.0 | Model confidence in prediction |
| `spectral_features` | object | - | Extracted FFT features |

**failure_mode Enum**:
- `"normal"` - Operating correctly
- `"failed_open"` - Continuously discharging steam
- `"failed_closed"` - Not discharging condensate
- `"leaking"` - Partial leak (wear, corrosion)
- `"plugged"` - Blocked by debris
- `"cavitation"` - Cavitation damage
- `"worn_seat"` - Valve seat erosion

#### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_SIGNAL_LENGTH` | 400 | Signal array too short/long |
| `INVALID_SAMPLING_RATE` | 400 | Sampling rate outside valid range |
| `INVALID_TRAP_TYPE` | 400 | Unknown trap type |
| `SIGNAL_PROCESSING_ERROR` | 500 | FFT computation failed |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/analyze_acoustic \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_id": "TRAP-001",
    "signal": [0.012, 0.015, 0.011, ...],
    "sampling_rate_hz": 250000,
    "trap_type": "thermodynamic"
  }'
```

---

### analyze_thermal_pattern

**Endpoint**: `POST /v1/analyze_thermal`

**Description**: Analyze IR thermal imaging pattern for trap health assessment and condensate pooling detection.

**Physics Basis**: Stefan-Boltzmann law, ASME PTC 19.3 Temperature Measurement

**Deterministic**: Yes

#### Request Body

```json
{
  "trap_id": "TRAP-001",
  "temperature_upstream_c": 150.0,
  "temperature_downstream_c": 92.0,
  "thermal_image": [[120.5, 121.2, ...], [119.8, ...], ...]  // Optional
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `trap_id` | string | Yes | Max 50 chars | Unique trap identifier |
| `temperature_upstream_c` | number | Yes | 0-300 | Upstream temperature (°C) |
| `temperature_downstream_c` | number | Yes | 0-300 | Downstream temperature (°C) |
| `thermal_image` | array[array[number]] | No | 240x320 to 1024x768 | IR image array (optional) |

#### Response

```json
{
  "status": "success",
  "data": {
    "trap_id": "TRAP-001",
    "trap_health_score": 75,
    "temperature_differential_c": 58.0,
    "anomalies_detected": [
      {
        "type": "hot_spot",
        "location": {"x": 120, "y": 85},
        "temperature_c": 165.0,
        "severity": "medium"
      }
    ],
    "condensate_pooling_detected": false,
    "recommended_action": "monitor",
    "analysis_method": "temperature_differential"
  },
  "metadata": {
    "request_id": "req_abc456",
    "timestamp": "2025-01-22T10:31:00Z",
    "execution_time_ms": 850
  }
}
```

#### Response Schema

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `trap_health_score` | number | 0-100 | Overall health (100=perfect, 0=failed) |
| `temperature_differential_c` | number | - | ΔT = T_upstream - T_downstream |
| `anomalies_detected` | array | - | List of thermal anomalies |
| `condensate_pooling_detected` | boolean | - | True if condensate backup detected |

**Health Score Interpretation**:
- **90-100**: Excellent
- **70-89**: Good
- **50-69**: Fair - inspection recommended
- **30-49**: Poor - likely failing
- **0-29**: Critical - immediate action required

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/analyze_thermal \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_id": "TRAP-001",
    "temperature_upstream_c": 150.0,
    "temperature_downstream_c": 92.0
  }'
```

---

### diagnose_trap_failure

**Endpoint**: `POST /v1/diagnose`

**Description**: Multi-modal failure diagnosis integrating acoustic, thermal, and operational data with root cause analysis.

**Physics Basis**: Multi-sensor fusion, weighted voting classification

**Deterministic**: Yes

#### Request Body

```json
{
  "trap_id": "TRAP-001",
  "sensor_data": {
    "pressure_upstream_psig": 100.0,
    "pressure_downstream_psig": 5.0,
    "flow_rate_kg_hr": 450.0,
    "operating_hours": 12500,
    "cycle_count": 45000
  },
  "acoustic_result": {
    "failure_probability": 0.87,
    "failure_mode": "failed_open",
    "confidence_score": 0.92
  },
  "thermal_result": {
    "trap_health_score": 35,
    "temperature_differential_c": 3.0
  }
}
```

#### Request Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trap_id` | string | Yes | Unique trap identifier |
| `sensor_data` | object | Yes | Operational sensor readings |
| `acoustic_result` | object | No | Output from analyze_acoustic_signature |
| `thermal_result` | object | No | Output from analyze_thermal_pattern |

**sensor_data Schema**:

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `pressure_upstream_psig` | number | Yes | 0-600 |
| `pressure_downstream_psig` | number | Yes | 0-600 |
| `flow_rate_kg_hr` | number | No | >= 0 |
| `operating_hours` | integer | No | >= 0 |
| `cycle_count` | integer | No | >= 0 |

#### Response

```json
{
  "status": "success",
  "data": {
    "trap_id": "TRAP-001",
    "failure_mode": "failed_open",
    "root_cause": "valve_seat_erosion",
    "failure_severity": "high",
    "recommended_action": "replace_trap_urgently",
    "urgency_hours": 24,
    "contributing_factors": [
      "high_operating_pressure",
      "excessive_cycling",
      "age_over_5_years"
    ],
    "confidence_score": 0.89,
    "fusion_method": "weighted_voting",
    "data_sources_used": ["acoustic", "thermal", "operational"]
  },
  "metadata": {
    "request_id": "req_abc789",
    "timestamp": "2025-01-22T10:32:00Z",
    "execution_time_ms": 450
  }
}
```

#### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `failure_mode` | string | Primary failure mode |
| `root_cause` | string | Diagnosed root cause |
| `failure_severity` | string | Enum: normal, low, medium, high, critical |
| `recommended_action` | string | Specific action to take |
| `urgency_hours` | integer | Hours until action required |

**failure_severity Thresholds**:
- **normal**: Failure probability < 0.1
- **low**: 0.1-0.3
- **medium**: 0.3-0.6
- **high**: 0.6-0.8
- **critical**: > 0.8

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/diagnose \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_id": "TRAP-001",
    "sensor_data": {
      "pressure_upstream_psig": 100.0,
      "pressure_downstream_psig": 5.0
    },
    "acoustic_result": {
      "failure_probability": 0.87,
      "failure_mode": "failed_open"
    },
    "thermal_result": {
      "trap_health_score": 35
    }
  }'
```

---

### calculate_energy_loss

**Endpoint**: `POST /v1/calculate_energy_loss`

**Description**: Calculate steam loss, energy waste, and cost impact using Napier's equation for orifice flow.

**Physics Basis**: Napier's equation, Spirax Sarco Steam Engineering, DOE Steam Tip Sheet #1

**Formula**: `W = 24.24 × P × D² × C` (lb/hr steam loss)

**Deterministic**: Yes

#### Request Body

```json
{
  "trap_id": "TRAP-001",
  "orifice_diameter_in": 0.125,
  "steam_pressure_psig": 100.0,
  "failure_severity": 0.87,
  "operating_hours_per_year": 8760,
  "steam_cost_usd_per_1000lb": 8.50
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `trap_id` | string | Yes | Max 50 chars | Unique trap identifier |
| `orifice_diameter_in` | number | Yes | 0.0625-1.0 | Trap orifice diameter (inches) |
| `steam_pressure_psig` | number | Yes | 0-600 | Steam pressure (psig) |
| `failure_severity` | number | Yes | 0.0-1.0 | Failure severity multiplier |
| `operating_hours_per_year` | integer | No | Default: 8760 | Annual operating hours |
| `steam_cost_usd_per_1000lb` | number | No | Default: 8.50 | Steam cost ($/1000 lb) |

#### Response

```json
{
  "status": "success",
  "data": {
    "trap_id": "TRAP-001",
    "steam_loss_kg_hr": 12.4,
    "steam_loss_lb_hr": 27.3,
    "energy_loss_gj_yr": 245.2,
    "energy_loss_mmbtu_yr": 232.4,
    "cost_loss_usd_yr": 2031.48,
    "co2_emissions_kg_yr": 13760.0,
    "co2_emissions_tons_yr": 13.76,
    "calculation_method": "napier_equation",
    "assumptions": {
      "discharge_coefficient": 0.97,
      "latent_heat_kj_kg": 2257,
      "co2_factor_kg_per_gj": 56.1
    }
  },
  "metadata": {
    "request_id": "req_def123",
    "timestamp": "2025-01-22T10:33:00Z",
    "execution_time_ms": 120
  }
}
```

#### Response Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `steam_loss_kg_hr` | number | kg/hr | Mass flow rate of steam loss |
| `steam_loss_lb_hr` | number | lb/hr | Mass flow rate (imperial) |
| `energy_loss_gj_yr` | number | GJ/yr | Annual energy loss |
| `cost_loss_usd_yr` | number | USD/yr | Annual cost impact |
| `co2_emissions_kg_yr` | number | kg CO2/yr | Annual CO2 emissions |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/calculate_energy_loss \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_id": "TRAP-001",
    "orifice_diameter_in": 0.125,
    "steam_pressure_psig": 100.0,
    "failure_severity": 0.87
  }'
```

---

### prioritize_maintenance

**Endpoint**: `POST /v1/prioritize_fleet`

**Description**: Fleet-wide maintenance prioritization using multi-factor scoring and ROI optimization.

**Physics Basis**: Linear weighted scoring, NPV/payback analysis

**Deterministic**: Yes

#### Request Body

```json
{
  "trap_fleet": [
    {
      "trap_id": "TRAP-001",
      "failure_mode": "failed_open",
      "energy_loss_usd_yr": 2031.48,
      "process_criticality": 8,
      "rul_days": 15,
      "maintenance_cost_usd": 150,
      "replacement_cost_usd": 500
    },
    {
      "trap_id": "TRAP-002",
      "failure_mode": "normal",
      "energy_loss_usd_yr": 0,
      "process_criticality": 5,
      "rul_days": 450,
      "maintenance_cost_usd": 150,
      "replacement_cost_usd": 500
    }
  ],
  "crew_capacity_per_day": 4,
  "working_days_per_month": 20
}
```

#### Request Schema

**trap_fleet Item Schema**:

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `trap_id` | string | Yes | Max 50 chars | Unique trap identifier |
| `failure_mode` | string | Yes | Enum | Current failure mode |
| `energy_loss_usd_yr` | number | Yes | >= 0 | Annual energy loss ($) |
| `process_criticality` | integer | Yes | 1-10 | Criticality score (10=highest) |
| `rul_days` | number | No | >= 0 | Remaining useful life (days) |
| `maintenance_cost_usd` | number | No | Default: 150 | Cost to repair ($) |
| `replacement_cost_usd` | number | No | Default: 500 | Cost to replace ($) |

#### Response

```json
{
  "status": "success",
  "data": {
    "priority_list": [
      {
        "rank": 1,
        "trap_id": "TRAP-001",
        "priority_score": 8542.5,
        "energy_loss_usd_yr": 2031.48,
        "rul_days": 15,
        "recommended_action": "replace",
        "urgency": "critical"
      },
      {
        "rank": 2,
        "trap_id": "TRAP-002",
        "priority_score": 125.0,
        "energy_loss_usd_yr": 0,
        "rul_days": 450,
        "recommended_action": "monitor",
        "urgency": "low"
      }
    ],
    "recommended_schedule": [
      {
        "month": 1,
        "week": 1,
        "trap_ids": ["TRAP-001"],
        "estimated_crew_days": 0.25
      }
    ],
    "fleet_metrics": {
      "total_traps": 2,
      "failed_traps": 1,
      "total_energy_loss_usd_yr": 2031.48,
      "total_maintenance_cost_usd": 500,
      "expected_roi_percent": 306.3,
      "payback_months": 3.0
    }
  },
  "metadata": {
    "request_id": "req_ghi456",
    "timestamp": "2025-01-22T10:34:00Z",
    "execution_time_ms": 680
  }
}
```

#### Priority Score Calculation

```
score = (
  energy_loss_usd_yr × 0.40 +
  process_criticality × 1000 × 0.30 +
  (1 / max(rul_days, 1)) × 0.20 +
  failure_probability × 5000 × 0.10
)
```

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/prioritize_fleet \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_fleet": [
      {
        "trap_id": "TRAP-001",
        "failure_mode": "failed_open",
        "energy_loss_usd_yr": 2031.48,
        "process_criticality": 8,
        "rul_days": 15
      }
    ]
  }'
```

---

### predict_remaining_useful_life

**Endpoint**: `POST /v1/predict_rul`

**Description**: Weibull distribution-based RUL prediction with confidence intervals.

**Physics Basis**: Weibull reliability analysis, MTBF statistical modeling

**Formula**: `R(t) = exp(-(t/η)^β)` where β=2.5, η from MTBF

**Deterministic**: Yes (with fixed seed)

#### Request Body

```json
{
  "trap_id": "TRAP-001",
  "current_age_days": 1200,
  "current_health_score": 65,
  "historical_failures": [1825, 1650, 1920, 1750],
  "trap_type": "thermodynamic",
  "operating_conditions": "normal"
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `trap_id` | string | Yes | Max 50 chars | Unique trap identifier |
| `current_age_days` | integer | Yes | >= 0 | Days since installation |
| `current_health_score` | number | Yes | 0-100 | Current health (from thermal analysis) |
| `historical_failures` | array[integer] | No | - | Ages at failure (days) for similar traps |
| `trap_type` | string | No | Enum | Trap type |
| `operating_conditions` | string | No | Enum | normal, severe, mild |

#### Response

```json
{
  "status": "success",
  "data": {
    "trap_id": "TRAP-001",
    "rul_days": 285,
    "rul_confidence_lower": 195,
    "rul_confidence_upper": 410,
    "failure_probability_curve": [
      {"days_from_now": 0, "probability": 0.15},
      {"days_from_now": 30, "probability": 0.22},
      {"days_from_now": 90, "probability": 0.45},
      {"days_from_now": 180, "probability": 0.68},
      {"days_from_now": 285, "probability": 0.90}
    ],
    "current_reliability": 0.52,
    "recommended_inspection_date": "2025-11-01",
    "weibull_parameters": {
      "beta": 2.5,
      "eta": 1825,
      "mtbf_days": 1625
    }
  },
  "metadata": {
    "request_id": "req_jkl789",
    "timestamp": "2025-01-22T10:35:00Z",
    "execution_time_ms": 920
  }
}
```

#### Response Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `rul_days` | number | days | Expected remaining useful life |
| `rul_confidence_lower` | number | days | 5th percentile (pessimistic) |
| `rul_confidence_upper` | number | days | 95th percentile (optimistic) |
| `failure_probability_curve` | array | - | Probability vs. time |
| `current_reliability` | number | 0-1 | Current probability of survival |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/predict_rul \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_id": "TRAP-001",
    "current_age_days": 1200,
    "current_health_score": 65,
    "historical_failures": [1825, 1650, 1920, 1750]
  }'
```

---

### calculate_cost_benefit

**Endpoint**: `POST /v1/calculate_cost_benefit`

**Description**: Financial analysis for repair vs. replace decisions with NPV, IRR, payback calculation.

**Physics Basis**: Discounted cash flow analysis, NPV/IRR calculations

**Formula**: `NPV = Σ(Savings_t / (1+r)^t) - Initial_Cost`

**Deterministic**: Yes

#### Request Body

```json
{
  "trap_id": "TRAP-001",
  "action": "replace",
  "annual_energy_loss_usd": 2031.48,
  "action_cost_usd": 500,
  "expected_service_life_years": 5,
  "discount_rate": 0.08,
  "inflation_rate": 0.03
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `trap_id` | string | Yes | Max 50 chars | Unique trap identifier |
| `action` | string | Yes | Enum: repair, replace | Type of action |
| `annual_energy_loss_usd` | number | Yes | >= 0 | Current annual loss ($) |
| `action_cost_usd` | number | Yes | >= 0 | Cost of repair/replacement ($) |
| `expected_service_life_years` | integer | Yes | 1-15 | Expected life after action (years) |
| `discount_rate` | number | No | Default: 0.08 | Annual discount rate (fraction) |
| `inflation_rate` | number | No | Default: 0.03 | Annual inflation rate (fraction) |

#### Response

```json
{
  "status": "success",
  "data": {
    "trap_id": "TRAP-001",
    "action": "replace",
    "npv_usd": 7215.60,
    "irr_percent": 402.1,
    "payback_months": 3.0,
    "decision_recommendation": "PROCEED",
    "justification": "High ROI, short payback period",
    "cash_flow_projection": [
      {"year": 0, "cash_flow_usd": -500},
      {"year": 1, "cash_flow_usd": 2031},
      {"year": 2, "cash_flow_usd": 2092},
      {"year": 3, "cash_flow_usd": 2155},
      {"year": 4, "cash_flow_usd": 2220},
      {"year": 5, "cash_flow_usd": 2286}
    ],
    "cumulative_savings_usd_5yr": 10784,
    "total_cost_usd": 500,
    "net_benefit_usd": 10284
  },
  "metadata": {
    "request_id": "req_mno012",
    "timestamp": "2025-01-22T10:36:00Z",
    "execution_time_ms": 250
  }
}
```

#### Response Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `npv_usd` | number | USD | Net present value |
| `irr_percent` | number | % | Internal rate of return |
| `payback_months` | number | months | Simple payback period |
| `decision_recommendation` | string | Enum | PROCEED, DEFER, REJECT |

**decision_recommendation Logic**:
- **PROCEED**: NPV > 0 AND payback < 24 months
- **DEFER**: NPV > 0 AND payback 24-48 months
- **REJECT**: NPV <= 0 OR payback > 48 months

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/calculate_cost_benefit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d '{
    "trap_id": "TRAP-001",
    "action": "replace",
    "annual_energy_loss_usd": 2031.48,
    "action_cost_usd": 500,
    "expected_service_life_years": 5
  }'
```

---

## Operation Modes

### Monitor Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "monitor",
  "trap_data": {
    "trap_id": "TRAP-001",
    "trap_type": "thermodynamic",
    "acoustic_data": {
      "signal": [...],
      "sampling_rate_hz": 250000
    },
    "thermal_data": {
      "temperature_upstream_c": 150.0,
      "temperature_downstream_c": 90.0
    },
    "orifice_diameter_in": 0.125,
    "steam_pressure_psig": 100.0
  }
}
```

**Description**: Continuous monitoring with multi-modal analysis. Returns health status, failure probability, and energy loss.

---

### Diagnose Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "diagnose",
  "trap_data": {
    "trap_id": "TRAP-001",
    "sensor_data": {...},
    "acoustic_result": {...},
    "thermal_result": {...}
  }
}
```

**Description**: Comprehensive failure diagnosis with root cause analysis.

---

### Predict Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "predict",
  "trap_data": {
    "trap_id": "TRAP-001",
    "current_age_days": 1200,
    "current_health_score": 65,
    "historical_failures": [1825, 1650]
  }
}
```

**Description**: Predictive maintenance with RUL calculation.

---

### Fleet Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "fleet",
  "fleet_data": [
    {"trap_id": "TRAP-001", ...},
    {"trap_id": "TRAP-002", ...}
  ]
}
```

**Description**: Multi-trap coordination and optimization.

---

## Error Codes

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Success |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Missing/invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Endpoint doesn't exist |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Agent error |
| 503 | Service Unavailable | Agent offline |

### Application Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `VALIDATION_ERROR` | Input validation failed | Check constraints |
| `INVALID_TRAP_TYPE` | Unknown trap type | Use valid enum value |
| `INVALID_SIGNAL_LENGTH` | Signal array invalid | Check min/max length |
| `SENSOR_DATA_MISSING` | Required sensor data absent | Provide all required fields |
| `CALCULATION_ERROR` | Physics calculation failed | Contact support |
| `ML_MODEL_ERROR` | ML model inference failed | Retry or contact support |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |

---

## Rate Limiting

### Standard Limits

| Tier | Requests/Minute | Requests/Hour | Requests/Day |
|------|-----------------|---------------|--------------|
| Free | 10 | 100 | 1,000 |
| Pro | 60 | 1,000 | 20,000 |
| Enterprise | Custom | Custom | Custom |

### Rate Limit Headers

**Response Headers**:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1706789400
```

### Rate Limit Exceeded Response

**HTTP 429**:
```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 2025-01-22T10:40:00Z",
    "retry_after": 120
  }
}
```

---

## Webhooks

### Webhook Configuration

**Endpoint**: `POST /v1/webhooks`

**Request**:
```json
{
  "url": "https://your-app.com/webhooks/trapcatcher",
  "events": ["trap.failure_detected", "trap.inspection_completed"],
  "secret": "whsec_your_secret"
}
```

### Webhook Events

#### trap.failure_detected

**Payload**:
```json
{
  "event": "trap.failure_detected",
  "timestamp": "2025-01-22T10:37:00Z",
  "data": {
    "trap_id": "TRAP-001",
    "failure_mode": "failed_open",
    "failure_severity": "critical",
    "energy_loss_usd_yr": 2031.48,
    "recommended_action": "replace_urgently"
  }
}
```

#### trap.inspection_completed

**Payload**:
```json
{
  "event": "trap.inspection_completed",
  "timestamp": "2025-01-22T10:38:00Z",
  "data": {
    "trap_id": "TRAP-001",
    "health_score": 75,
    "failure_probability": 0.25,
    "next_inspection_date": "2025-04-22"
  }
}
```

### Webhook Signature Verification

**Algorithm**: HMAC-SHA256

**Header**: `X-Webhook-Signature`

**Verification**:
```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)
```

---

## Code Examples

### Python SDK

```python
from greenlang import SteamTrapInspector

# Initialize agent
agent = SteamTrapInspector(api_key="gl_sk_abc123...")

# Acoustic analysis
acoustic_result = agent.analyze_acoustic_signature(
    trap_id="TRAP-001",
    signal=[0.012, 0.015, 0.011, ...],
    sampling_rate_hz=250000,
    trap_type="thermodynamic"
)

print(f"Failure probability: {acoustic_result.failure_probability}")
print(f"Failure mode: {acoustic_result.failure_mode}")

# Energy loss calculation
energy_loss = agent.calculate_energy_loss(
    trap_id="TRAP-001",
    orifice_diameter_in=0.125,
    steam_pressure_psig=100.0,
    failure_severity=acoustic_result.failure_probability
)

print(f"Annual cost loss: ${energy_loss.cost_loss_usd_yr:,.2f}")
print(f"CO2 emissions: {energy_loss.co2_emissions_kg_yr:,.0f} kg/yr")
```

### JavaScript/TypeScript SDK

```typescript
import { SteamTrapInspector } from '@greenlang/agents';

const agent = new SteamTrapInspector({ apiKey: 'gl_sk_abc123...' });

// Thermal analysis
const thermalResult = await agent.analyzeThermalPattern({
  trapId: 'TRAP-001',
  temperatureUpstreamC: 150.0,
  temperatureDownstreamC: 92.0,
});

console.log(`Health score: ${thermalResult.trapHealthScore}`);

// Fleet prioritization
const prioritized = await agent.prioritizeMaintenance({
  trapFleet: [
    { trapId: 'TRAP-001', energyLossUsdYr: 2031.48, processCriticality: 8 },
    { trapId: 'TRAP-002', energyLossUsdYr: 450.0, processCriticality: 5 },
  ],
});

console.log(`Top priority: ${prioritized.priorityList[0].trapId}`);
```

### cURL Examples

**Diagnose Trap**:
```bash
curl -X POST https://api.greenlang.org/agents/gl-008/v1/diagnose \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_abc123..." \
  -d @- <<EOF
{
  "trap_id": "TRAP-001",
  "sensor_data": {
    "pressure_upstream_psig": 100.0,
    "pressure_downstream_psig": 5.0
  },
  "acoustic_result": {
    "failure_probability": 0.87,
    "failure_mode": "failed_open"
  }
}
EOF
```

---

## SDK Reference

### Python SDK

**Installation**:
```bash
pip install greenlang-agents
```

**Documentation**: https://docs.greenlang.org/sdks/python

### JavaScript/TypeScript SDK

**Installation**:
```bash
npm install @greenlang/agents
```

**Documentation**: https://docs.greenlang.org/sdks/javascript

### Go SDK

**Installation**:
```bash
go get github.com/greenlang/agents-go
```

**Documentation**: https://docs.greenlang.org/sdks/go

---

## Changelog

### v1.0.0 (2025-01-22)

**Initial Release**

- 7 deterministic tools
- Multi-modal analysis (acoustic + thermal + operational)
- Physics-based energy loss calculation (Napier equation)
- Weibull RUL prediction
- Fleet optimization with ROI analysis
- Full provenance tracking
- Industry standards compliance (ASME, ISO, ASTM)

---

**API Version**: v1
**Documentation Version**: 1.0.0
**Last Updated**: 2025-01-22
**Support**: api-support@greenlang.org
**License**: Apache-2.0

---

For questions, bug reports, or feature requests, please contact api-support@greenlang.org or open an issue at https://github.com/greenlang/agents/issues
