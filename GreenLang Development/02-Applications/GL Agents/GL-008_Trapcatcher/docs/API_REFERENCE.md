# GL-008 TRAPCATCHER API Reference

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

API key authentication via header:
```
Authorization: Bearer <api_key>
```

## Endpoints

### Health & Status

#### GET /health
Liveness probe for Kubernetes.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-24T10:30:00Z"
}
```

#### GET /health/ready
Readiness probe for Kubernetes.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-24T10:30:00Z",
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 2.5
    }
  ]
}
```

#### GET /metrics
Prometheus metrics endpoint.

**Response:** OpenMetrics format
```
# HELP trapcatcher_diagnoses_total Total diagnoses performed
# TYPE trapcatcher_diagnoses_total counter
trapcatcher_diagnoses_total{trap_type="thermodynamic",condition="normal"} 1523
```

#### GET /status
Agent operational status.

**Response:**
```json
{
  "agent_id": "GL-008",
  "agent_name": "TRAPCATCHER",
  "version": "1.0.0",
  "status": "running",
  "mode": "monitoring",
  "uptime_seconds": 86400,
  "statistics": {
    "total_diagnoses": 15230,
    "failed_traps_detected": 142,
    "total_energy_loss_kw": 450.5,
    "total_co2_kg": 12500
  }
}
```

### Diagnostics

#### POST /diagnose
Diagnose a single steam trap.

**Request:**
```json
{
  "trap_id": "ST-001",
  "trap_type": "thermodynamic",
  "pressure_bar": 5.5,
  "inlet_temperature_c": 158.0,
  "outlet_temperature_c": 145.0,
  "acoustic": {
    "level_db": 75.0,
    "frequency_khz": 38.0,
    "spectral_entropy": 0.65
  },
  "operating_hours_year": 8000,
  "installation_date": "2020-01-15"
}
```

**Response:**
```json
{
  "trap_id": "ST-001",
  "diagnosis_id": "d8f7e6a5-4c3b-2a1d-0e9f-8g7h6i5j4k3l",
  "timestamp": "2024-12-24T10:30:00Z",
  "classification": {
    "condition": "leaking",
    "severity": "moderate",
    "confidence": 0.87
  },
  "energy_loss": {
    "steam_loss_kg_hr": 12.5,
    "energy_loss_kw": 8.2,
    "annual_cost_usd": 2450.00,
    "co2_emissions_kg_yr": 4500
  },
  "explainability": {
    "top_features": [
      {
        "feature": "acoustic_level_db",
        "value": 75.0,
        "contribution": 0.35,
        "direction": "toward_failure"
      },
      {
        "feature": "delta_temperature_c",
        "value": 13.0,
        "contribution": 0.28,
        "direction": "toward_failure"
      }
    ],
    "evidence_chain": [
      {
        "step": 1,
        "observation": "Elevated acoustic level (75 dB)",
        "inference": "Steam passing through orifice"
      }
    ],
    "counterfactual": "If acoustic level were below 50 dB, diagnosis would be NORMAL"
  },
  "recommendation": {
    "action": "schedule_repair",
    "priority": "high",
    "estimated_payback_months": 3.2
  },
  "provenance": {
    "input_hash": "a1b2c3d4e5f6...",
    "output_hash": "f6e5d4c3b2a1...",
    "formula_version": "NAPIER_v1.0"
  }
}
```

#### POST /analyze/fleet
Analyze entire fleet of steam traps.

**Request:**
```json
{
  "facility_id": "PLANT-001",
  "traps": [
    {
      "trap_id": "ST-001",
      "trap_type": "thermodynamic",
      "pressure_bar": 5.5,
      "inlet_temperature_c": 158.0,
      "outlet_temperature_c": 145.0
    },
    {
      "trap_id": "ST-002",
      "trap_type": "thermostatic",
      "pressure_bar": 3.0,
      "inlet_temperature_c": 143.0,
      "outlet_temperature_c": 140.0
    }
  ],
  "options": {
    "include_maintenance_priority": true,
    "include_survey_route": true
  }
}
```

**Response:**
```json
{
  "analysis_id": "a1b2c3d4...",
  "facility_id": "PLANT-001",
  "timestamp": "2024-12-24T10:30:00Z",
  "summary": {
    "total_traps": 150,
    "normal": 120,
    "leaking": 18,
    "blocked": 8,
    "blowthrough": 4,
    "fleet_health_score": 82.5
  },
  "energy_impact": {
    "total_energy_loss_kw": 125.5,
    "annual_cost_usd": 45000,
    "co2_emissions_tonnes_yr": 85.2
  },
  "maintenance_priority": [
    {
      "rank": 1,
      "trap_id": "ST-045",
      "condition": "blowthrough",
      "energy_loss_kw": 25.0,
      "roi_months": 1.2
    }
  ],
  "survey_route": {
    "optimized_sequence": ["ST-045", "ST-023", "ST-089"],
    "estimated_duration_hours": 4.5
  }
}
```

### Configuration

#### GET /config
Get current configuration.

**Response:**
```json
{
  "agent_id": "GL-008",
  "mode": "monitoring",
  "thresholds": {
    "acoustic_warning_db": 60,
    "acoustic_critical_db": 80,
    "delta_t_warning_c": 10,
    "delta_t_critical_c": 5
  },
  "weights": {
    "acoustic": 0.40,
    "thermal": 0.40,
    "contextual": 0.20
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
    "acoustic_warning_db": 55
  }
}
```

## Error Responses

All errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://greenlang.io/errors/validation",
  "title": "Validation Error",
  "status": 400,
  "detail": "pressure_bar=30.0 exceeds maximum 25.0 bar",
  "instance": "/api/v1/diagnose",
  "violations": [
    {
      "parameter": "pressure_bar",
      "value": 30.0,
      "constraint": "max=25.0",
      "standard": "ASME PTC 39"
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
| 500 | internal | Internal server error |
| 503 | unavailable | Service temporarily unavailable |

## Rate Limits

| Tier | Requests/min | Burst |
|------|-------------|-------|
| Standard | 100 | 20 |
| Premium | 1000 | 100 |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703416200
```

## Webhooks

Configure webhooks for real-time alerts:

```json
POST /webhooks
{
  "url": "https://your-server.com/trapcatcher-alerts",
  "events": ["failure_detected", "critical_loss"],
  "secret": "your-webhook-secret"
}
```

Webhook payload:
```json
{
  "event": "failure_detected",
  "timestamp": "2024-12-24T10:30:00Z",
  "trap_id": "ST-045",
  "condition": "blowthrough",
  "severity": "critical",
  "energy_loss_kw": 25.0,
  "signature": "sha256=..."
}
```

## SDKs

- Python: `pip install greenlang-trapcatcher`
- JavaScript: `npm install @greenlang/trapcatcher`
- Go: `go get github.com/greenlang/trapcatcher-go`
