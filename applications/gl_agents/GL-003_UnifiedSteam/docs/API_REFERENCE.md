# GL-003 UnifiedSteam API Reference

**Version:** 1.0.0
**Base URL:** `https://api.greenlang.io/gl-003/v1`
**OpenAPI Spec:** [openapi.yaml](./openapi.yaml)

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Steam Properties API](#steam-properties-api)
- [Balance Calculations API](#balance-calculations-api)
- [Optimization API](#optimization-api)
- [Trap Diagnostics API](#trap-diagnostics-api)
- [Recommendations API](#recommendations-api)
- [Root Cause Analysis API](#root-cause-analysis-api)
- [KPI and Analytics API](#kpi-and-analytics-api)
- [Audit and Compliance API](#audit-and-compliance-api)
- [System API](#system-api)
- [WebSocket API](#websocket-api)
- [Appendix](#appendix)

---

## Overview

The GL-003 UnifiedSteam API provides comprehensive steam system optimization capabilities including:

- **Steam Properties**: IAPWS-IF97 compliant thermodynamic calculations
- **Optimization**: Desuperheater, condensate recovery, and network optimization
- **Diagnostics**: Steam trap condition monitoring and predictive maintenance
- **Analytics**: KPIs, climate impact, and energy efficiency metrics
- **Audit**: Full audit trail with SHA-256 provenance tracking

### API Design Principles

| Principle | Description |
|-----------|-------------|
| **Zero-Hallucination** | All numeric calculations use deterministic IAPWS-IF97 formulas |
| **Full Provenance** | SHA-256 hashing of all inputs/outputs for audit trails |
| **Explainability** | SHAP/LIME explanations for all recommendations |
| **Regulatory Compliance** | GHG Protocol, IPMVP, ISO 50001 aligned |

### Content Types

All endpoints accept and return `application/json` unless otherwise specified.

```
Content-Type: application/json
Accept: application/json
```

---

## Authentication

### Bearer Token Authentication

All API requests require a valid JWT bearer token in the Authorization header.

```bash
curl -X POST https://api.greenlang.io/gl-003/v1/steam/properties \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"pressure_kpa": 1000, "temperature_c": 200}'
```

### Required Permissions

| Permission | Description |
|------------|-------------|
| `steam:properties:compute` | Compute steam properties |
| `steam:balance:compute` | Compute energy/mass balances |
| `optimization:request` | Request optimization analysis |
| `optimization:execute` | Execute optimization recommendations |
| `trap:read` | Read trap diagnostic data |
| `trap:diagnose` | Perform trap diagnostics |
| `recommendation:read` | Read recommendations |
| `recommendation:approve` | Approve/reject recommendations |
| `rca:analyze` | Perform root cause analysis |
| `kpi:read` | Read KPI metrics |
| `audit:read` | Read audit logs |
| `audit:write` | Write audit entries |

### Example: Obtaining a Token

```bash
curl -X POST https://auth.greenlang.io/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "grant_type": "client_credentials",
    "scope": "steam:properties:compute optimization:request"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "steam:properties:compute optimization:request"
}
```

---

## Rate Limiting

| Tier | Requests/min | Burst | Concurrent |
|------|-------------|-------|------------|
| Free | 60 | 10 | 5 |
| Standard | 600 | 50 | 20 |
| Enterprise | 6000 | 200 | 100 |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 599
X-RateLimit-Reset: 1703404800
```

---

## Error Handling

### Error Response Format

```json
{
  "error_code": "VALIDATION_ERROR",
  "error_message": "Pressure must be positive",
  "details": {
    "field": "pressure_kpa",
    "constraint": "ge=0"
  },
  "timestamp": "2024-12-24T10:00:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `AUTHENTICATION_ERROR` | 401 | Invalid or missing token |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `COMPUTATION_ERROR` | 500 | Internal computation failed |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Steam Properties API

### POST /steam/properties

Compute thermodynamic properties of steam/water using IAPWS-IF97 formulation.

**Permission Required:** `steam:properties:compute`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "pressure_kpa": 1000.0,
  "temperature_c": 200.0,
  "include_transport_properties": true,
  "include_derivatives": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | UUID | No | Client-provided request ID |
| `pressure_kpa` | float | No* | Pressure in kPa |
| `temperature_c` | float | No* | Temperature in Celsius |
| `specific_enthalpy_kj_kg` | float | No* | Specific enthalpy in kJ/kg |
| `specific_entropy_kj_kg_k` | float | No* | Specific entropy in kJ/(kg*K) |
| `quality` | float | No* | Steam quality (0-1) for two-phase |
| `include_transport_properties` | bool | No | Include viscosity, conductivity |
| `include_derivatives` | bool | No | Include partial derivatives |

*At least two independent properties must be provided.

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "steam_state": {
    "pressure_kpa": 1000.0,
    "temperature_c": 200.0,
    "specific_enthalpy_kj_kg": 2827.9,
    "specific_entropy_kj_kg_k": 6.694,
    "specific_volume_m3_kg": 0.206,
    "density_kg_m3": 4.854,
    "phase": "superheated_vapor",
    "region": "region_2",
    "cp_kj_kg_k": 2.315,
    "cv_kj_kg_k": 1.774,
    "speed_of_sound_m_s": 518.5,
    "viscosity_pa_s": 1.58e-5,
    "thermal_conductivity_w_m_k": 0.0345
  },
  "computation_time_ms": 2.5,
  "created_at": "2024-12-24T10:00:00Z"
}
```

#### cURL Example

```bash
curl -X POST https://api.greenlang.io/gl-003/v1/steam/properties \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "pressure_kpa": 1000,
    "temperature_c": 200,
    "include_transport_properties": true
  }'
```

---

### POST /steam/balance

Compute enthalpy balance for steam system equipment.

**Permission Required:** `steam:balance:compute`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "equipment_id": "DESUPER-001",
  "equipment_name": "Main Desuperheater",
  "streams": [
    {
      "stream_id": "inlet-steam",
      "stream_name": "Inlet Superheated Steam",
      "mass_flow_kg_s": 10.0,
      "pressure_kpa": 4000,
      "temperature_c": 450,
      "is_inlet": true
    },
    {
      "stream_id": "spray-water",
      "stream_name": "Spray Water",
      "mass_flow_kg_s": 1.5,
      "pressure_kpa": 5000,
      "temperature_c": 80,
      "is_inlet": true
    },
    {
      "stream_id": "outlet-steam",
      "stream_name": "Outlet Desuperheated Steam",
      "mass_flow_kg_s": 11.5,
      "pressure_kpa": 3800,
      "temperature_c": 350,
      "is_inlet": false
    }
  ],
  "heat_input_kw": 0,
  "heat_loss_kw": 50
}
```

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "equipment_id": "DESUPER-001",
  "success": true,
  "total_inlet_enthalpy_kw": 35250.5,
  "total_outlet_enthalpy_kw": 35180.2,
  "enthalpy_imbalance_kw": 70.3,
  "enthalpy_imbalance_percent": 0.2,
  "stream_enthalpies": {
    "inlet-steam": 33200.0,
    "spray-water": 2050.5,
    "outlet-steam": 35180.2
  },
  "balance_closed": true,
  "tolerance_percent": 2.0,
  "data_quality_score": 95.0,
  "created_at": "2024-12-24T10:00:00Z"
}
```

#### cURL Example

```bash
curl -X POST https://api.greenlang.io/gl-003/v1/steam/balance \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "DESUPER-001",
    "equipment_name": "Main Desuperheater",
    "streams": [
      {"stream_id": "inlet", "stream_name": "Inlet", "mass_flow_kg_s": 10, "pressure_kpa": 4000, "temperature_c": 450, "is_inlet": true},
      {"stream_id": "outlet", "stream_name": "Outlet", "mass_flow_kg_s": 10, "pressure_kpa": 3800, "temperature_c": 350, "is_inlet": false}
    ]
  }'
```

---

## Optimization API

### POST /optimization/desuperheater

Optimize spray water flow for desuperheater temperature control.

**Permission Required:** `optimization:request`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "desuperheater_id": "DESUPER-001",
  "inlet_steam_pressure_kpa": 4000,
  "inlet_steam_temperature_c": 450,
  "inlet_steam_flow_kg_s": 10.0,
  "target_outlet_temperature_c": 350,
  "target_temperature_tolerance_c": 2.0,
  "spray_water_pressure_kpa": 5000,
  "spray_water_temperature_c": 80,
  "min_superheat_c": 10,
  "max_spray_water_flow_kg_s": 3.0,
  "objective": "minimize_energy",
  "optimization_horizon_hours": 24
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `desuperheater_id` | string | Yes | Desuperheater identifier |
| `inlet_steam_pressure_kpa` | float | Yes | Inlet steam pressure |
| `inlet_steam_temperature_c` | float | Yes | Inlet steam temperature |
| `inlet_steam_flow_kg_s` | float | Yes | Inlet steam mass flow |
| `target_outlet_temperature_c` | float | Yes | Target outlet temperature |
| `spray_water_pressure_kpa` | float | Yes | Spray water pressure |
| `spray_water_temperature_c` | float | Yes | Spray water temperature |
| `min_superheat_c` | float | No | Minimum superheat (default: 10) |
| `max_spray_water_flow_kg_s` | float | No | Maximum spray water flow |
| `objective` | enum | No | Optimization objective |

**Objective Options:** `minimize_cost`, `minimize_emissions`, `maximize_efficiency`, `minimize_energy`, `balance_cost_emissions`

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "desuperheater_id": "DESUPER-001",
  "success": true,
  "optimal_spray_water_flow_kg_s": 1.47,
  "optimal_outlet_temperature_c": 350.0,
  "spray_water_energy_kw": 294.0,
  "desuperheating_efficiency": 0.95,
  "recommendations": [
    {
      "recommendation_id": "rec-001",
      "recommendation_type": "desuperheater",
      "priority": "medium",
      "title": "Optimize spray water control",
      "description": "Current control can be improved for better efficiency",
      "estimated_energy_savings_kw": 25.0,
      "confidence_score": 0.85
    }
  ],
  "computation_time_ms": 45.2,
  "created_at": "2024-12-24T10:00:00Z"
}
```

#### cURL Example

```bash
curl -X POST https://api.greenlang.io/gl-003/v1/optimization/desuperheater \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "desuperheater_id": "DESUPER-001",
    "inlet_steam_pressure_kpa": 4000,
    "inlet_steam_temperature_c": 450,
    "inlet_steam_flow_kg_s": 10.0,
    "target_outlet_temperature_c": 350,
    "spray_water_pressure_kpa": 5000,
    "spray_water_temperature_c": 80
  }'
```

---

### POST /optimization/condensate

Optimize condensate recovery system for maximum energy and water savings.

**Permission Required:** `optimization:request`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "system_id": "COND-SYS-001",
  "condensate_sources": [
    {
      "source_id": "process-1",
      "flow_rate_kg_h": 500,
      "temperature_c": 95,
      "pressure_kpa": 200
    },
    {
      "source_id": "process-2",
      "flow_rate_kg_h": 300,
      "temperature_c": 85,
      "pressure_kpa": 150
    }
  ],
  "current_recovery_rate_percent": 65.0,
  "flash_tank_pressure_kpa": 120,
  "condensate_return_temperature_c": 80,
  "makeup_water_temperature_c": 15,
  "makeup_water_cost_usd_m3": 2.5,
  "condensate_treatment_cost_usd_m3": 0.5,
  "makeup_treatment_cost_usd_m3": 3.0,
  "objective": "minimize_cost"
}
```

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "system_id": "COND-SYS-001",
  "success": true,
  "optimal_recovery_rate_percent": 85.0,
  "current_vs_optimal_delta_percent": 20.0,
  "optimal_flash_tank_pressure_kpa": 110,
  "flash_steam_recovery_kg_s": 0.025,
  "annual_water_savings_m3": 15000,
  "annual_energy_savings_mwh": 850,
  "annual_cost_savings_usd": 125000,
  "implementation_cost_usd": 75000,
  "simple_payback_years": 0.6,
  "annual_co2_reduction_tonnes": 170,
  "recommendations": [
    {
      "recommendation_id": "rec-002",
      "recommendation_type": "condensate_recovery",
      "priority": "high",
      "title": "Increase condensate recovery to 85%",
      "description": "Install additional collection points and upgrade return pumps",
      "estimated_cost_savings_usd_year": 125000,
      "estimated_emissions_reduction_kg_co2_year": 170000,
      "confidence_score": 0.92
    }
  ],
  "computation_time_ms": 125.5,
  "created_at": "2024-12-24T10:00:00Z"
}
```

---

### POST /optimization/network

Optimize steam distribution network for cost and emissions balance.

**Permission Required:** `optimization:execute`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "network_id": "STEAM-NET-001",
  "headers": [
    {
      "header_id": "HP-HDR",
      "name": "High Pressure Header",
      "design_pressure_kpa": 4000,
      "current_pressure_kpa": 3950
    },
    {
      "header_id": "MP-HDR",
      "name": "Medium Pressure Header",
      "design_pressure_kpa": 1000,
      "current_pressure_kpa": 980
    },
    {
      "header_id": "LP-HDR",
      "name": "Low Pressure Header",
      "design_pressure_kpa": 400,
      "current_pressure_kpa": 395
    }
  ],
  "generators": [
    {
      "generator_id": "BOILER-1",
      "name": "Main Boiler",
      "max_output_kg_s": 20,
      "min_output_kg_s": 5,
      "efficiency": 0.88,
      "fuel_cost_usd_gj": 8.5,
      "co2_factor_kg_gj": 56
    },
    {
      "generator_id": "WHB-1",
      "name": "Waste Heat Boiler",
      "max_output_kg_s": 8,
      "min_output_kg_s": 0,
      "efficiency": 0.95,
      "fuel_cost_usd_gj": 0,
      "co2_factor_kg_gj": 0
    }
  ],
  "consumers": [
    {
      "consumer_id": "PROC-1",
      "name": "Process Heat Exchanger",
      "demand_kg_s": 5,
      "header_id": "HP-HDR"
    }
  ],
  "total_demand_kg_s": 15.0,
  "objective": "balance_cost_emissions",
  "cost_weight": 0.5,
  "emissions_weight": 0.5,
  "optimization_horizon_hours": 24
}
```

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "network_id": "STEAM-NET-001",
  "success": true,
  "optimal_header_pressures_kpa": {
    "HP-HDR": 3980,
    "MP-HDR": 990,
    "LP-HDR": 398
  },
  "optimal_generator_outputs_kg_s": {
    "BOILER-1": 7.5,
    "WHB-1": 7.5
  },
  "optimal_letdown_flows_kg_s": {
    "HP-to-MP": 2.5,
    "MP-to-LP": 1.0
  },
  "total_generation_kg_s": 15.0,
  "total_consumption_kg_s": 15.0,
  "network_efficiency_percent": 92.5,
  "total_operating_cost_usd_h": 125.5,
  "marginal_cost_by_header_usd_kg": {
    "HP-HDR": 0.025,
    "MP-HDR": 0.020,
    "LP-HDR": 0.015
  },
  "total_emissions_kg_co2_h": 450,
  "emissions_by_source": {
    "BOILER-1": 450,
    "WHB-1": 0
  },
  "all_constraints_satisfied": true,
  "solver_status": "optimal",
  "computation_time_ms": 350.2,
  "created_at": "2024-12-24T10:00:00Z"
}
```

---

## Trap Diagnostics API

### GET /traps/{trap_id}/diagnostics

Retrieve diagnostics and status for a specific steam trap.

**Permission Required:** `trap:read`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `trap_id` | string | Steam trap identifier |

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_prediction` | bool | true | Include failure prediction |

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "trap_id": "TRAP-001",
  "success": true,
  "status": {
    "trap_id": "TRAP-001",
    "trap_name": "Process Area Steam Trap 1",
    "trap_type": "thermodynamic",
    "condition": "good",
    "condition_confidence": 0.92,
    "location": "Building A, Level 2, Bay 5",
    "header_id": "LP-HDR",
    "inlet_pressure_kpa": 400,
    "outlet_pressure_kpa": 101.325,
    "inlet_temperature_c": 145,
    "outlet_temperature_c": 95,
    "differential_temperature_c": 50,
    "cycle_rate_per_min": 4.5,
    "estimated_steam_loss_kg_h": 0,
    "estimated_energy_loss_kw": 0,
    "estimated_annual_cost_loss_usd": 0
  },
  "failure_prediction": {
    "trap_id": "TRAP-001",
    "failure_probability_30d": 0.05,
    "failure_probability_90d": 0.12,
    "predicted_failure_mode": "leaking",
    "predicted_remaining_life_days": 365,
    "risk_factors": [
      "Age > 3 years",
      "High cycling frequency"
    ],
    "risk_score": 15.0,
    "recommended_action": "Continue monitoring",
    "priority": "low",
    "model_confidence": 0.88
  },
  "diagnostic_method": "multi-sensor",
  "diagnostic_confidence": 0.92,
  "anomalies_detected": [],
  "computation_time_ms": 15.5
}
```

#### cURL Example

```bash
curl -X GET "https://api.greenlang.io/gl-003/v1/traps/TRAP-001/diagnostics?include_prediction=true" \
  -H "Authorization: Bearer $TOKEN"
```

---

### POST /traps/batch-diagnostics

Perform diagnostics on multiple steam traps.

**Permission Required:** `trap:diagnose`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "traps": [
    {
      "trap_id": "TRAP-001",
      "inlet_pressure_kpa": 400,
      "outlet_pressure_kpa": 101.325,
      "inlet_temperature_c": 145,
      "outlet_temperature_c": 95
    },
    {
      "trap_id": "TRAP-002",
      "inlet_pressure_kpa": 400,
      "outlet_pressure_kpa": 101.325,
      "inlet_temperature_c": 145,
      "outlet_temperature_c": 140
    }
  ],
  "include_summary": true,
  "include_prioritization": true
}
```

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "results": [
    {
      "trap_id": "TRAP-001",
      "success": true,
      "status": {
        "condition": "good",
        "condition_confidence": 0.92
      }
    },
    {
      "trap_id": "TRAP-002",
      "success": true,
      "status": {
        "condition": "leaking",
        "condition_confidence": 0.88,
        "estimated_steam_loss_kg_h": 25,
        "estimated_energy_loss_kw": 62.5,
        "estimated_annual_cost_loss_usd": 18250
      }
    }
  ],
  "total_traps": 2,
  "traps_good": 1,
  "traps_leaking": 1,
  "traps_blocked": 0,
  "traps_failed": 0,
  "total_estimated_steam_loss_kg_h": 25,
  "total_estimated_energy_loss_kw": 62.5,
  "total_estimated_annual_cost_loss_usd": 18250,
  "prioritized_actions": [
    {
      "trap_id": "TRAP-002",
      "condition": "leaking",
      "priority": "high",
      "action": "Repair",
      "estimated_savings_usd_year": 18250
    }
  ],
  "computation_time_ms": 45.0
}
```

---

## Recommendations API

### GET /recommendations

Retrieve list of optimization recommendations.

**Permission Required:** `recommendation:read`

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `status_filter` | enum | Filter by status (pending, implemented, rejected) |
| `priority_filter` | enum | Filter by priority (critical, high, medium, low) |
| `type_filter` | enum | Filter by type (desuperheater, condensate_recovery, network) |
| `page` | int | Page number (default: 1) |
| `page_size` | int | Items per page (default: 20, max: 100) |

#### Response

```json
{
  "recommendations": [
    {
      "recommendation_id": "550e8400-e29b-41d4-a716-446655440000",
      "recommendation_type": "condensate_recovery",
      "priority": "high",
      "status": "pending",
      "title": "Increase condensate recovery rate",
      "description": "Current recovery rate of 65% can be increased to 85%",
      "rationale": "Higher recovery reduces makeup water and energy costs",
      "estimated_energy_savings_kw": 150.0,
      "estimated_cost_savings_usd_year": 75000,
      "estimated_emissions_reduction_kg_co2_year": 50000,
      "confidence_score": 0.92,
      "created_at": "2024-12-24T10:00:00Z"
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

#### cURL Example

```bash
curl -X GET "https://api.greenlang.io/gl-003/v1/recommendations?priority_filter=high&status_filter=pending" \
  -H "Authorization: Bearer $TOKEN"
```

---

### GET /recommendations/{rec_id}/explanation

Get SHAP/LIME explainability for a recommendation.

**Permission Required:** `recommendation:read`

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `explanation_type` | enum | shap | Type: shap, lime, or both |

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "recommendation_id": "rec-001",
  "success": true,
  "shap_feature_contributions": [
    {
      "feature_name": "condensate_recovery_rate",
      "feature_value": 0.65,
      "contribution_score": 0.35,
      "contribution_direction": "positive",
      "explanation": "Current low recovery rate is main driver for recommendation"
    },
    {
      "feature_name": "makeup_water_cost",
      "feature_value": 2.5,
      "contribution_score": 0.25,
      "contribution_direction": "positive",
      "explanation": "Higher makeup water cost increases savings potential"
    }
  ],
  "shap_base_value": 0.5,
  "shap_output_value": 0.92,
  "plain_english_explanation": "This recommendation is primarily driven by the current low condensate recovery rate of 65%. Given the high cost of makeup water and the high temperature of available condensate, increasing recovery would significantly reduce both water and energy costs.",
  "technical_explanation": "SHAP analysis indicates condensate_recovery_rate contributes 35% to the recommendation score. The model predicts 92% confidence in potential savings based on similar facilities.",
  "key_drivers": [
    "Low condensate recovery rate (65%)",
    "High makeup water cost ($2.50/m3)",
    "High condensate temperature (85C)"
  ],
  "computation_time_ms": 45.0
}
```

---

## Root Cause Analysis API

### POST /rca/analyze

Perform causal root cause analysis for an event or anomaly.

**Permission Required:** `rca:analyze`

#### Request Body

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "target_event": "Header pressure drop",
  "event_timestamp": "2024-12-24T08:30:00Z",
  "event_severity": "high",
  "affected_equipment": ["HP-HDR", "BOILER-1"],
  "affected_variables": ["header_pressure", "steam_flow", "temperature"],
  "lookback_hours": 24,
  "lookahead_hours": 2,
  "include_counterfactuals": true,
  "max_causal_factors": 10,
  "min_confidence_threshold": 0.5
}
```

#### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "target_event": "Header pressure drop",
  "event_timestamp": "2024-12-24T08:30:00Z",
  "root_causes": [
    {
      "factor_id": "fc-001",
      "factor_name": "Steam trap failure",
      "factor_description": "Upstream steam trap failed in open position",
      "causal_strength": 0.85,
      "confidence": 0.82,
      "is_root_cause": true,
      "is_contributing_factor": false,
      "supporting_evidence": [
        "Temperature spike detected 15 min before event",
        "Trap acoustic signature changed 2 hours prior"
      ],
      "related_variables": ["trap_temperature", "trap_acoustic"]
    }
  ],
  "contributing_factors": [
    {
      "factor_id": "fc-002",
      "factor_name": "High system load",
      "factor_description": "System operating at 95% capacity",
      "causal_strength": 0.45,
      "confidence": 0.75,
      "is_root_cause": false,
      "is_contributing_factor": true,
      "supporting_evidence": ["Steam demand 15% above normal"]
    }
  ],
  "causal_chain": ["trap_failure", "steam_loss", "pressure_drop", "temperature_spike"],
  "counterfactual_scenarios": [
    {
      "scenario_name": "Trap maintained at schedule",
      "intervention_variable": "trap_maintenance_days",
      "intervention_value": 90,
      "baseline_value": 180,
      "predicted_outcome": 0.15,
      "baseline_outcome": 0.85,
      "outcome_change_percent": -82.4,
      "prediction_confidence": 0.78
    }
  ],
  "executive_summary": "The event was primarily caused by a steam trap failure that resulted in steam loss and subsequent pressure drop. High system load contributed to the severity of the impact.",
  "recommended_actions": [
    "Replace failed steam trap immediately",
    "Inspect adjacent traps for similar degradation",
    "Review trap maintenance schedule"
  ],
  "analysis_method": "causal_discovery",
  "model_confidence": 0.82,
  "computation_time_ms": 850.5
}
```

---

## KPI and Analytics API

### GET /kpis

Retrieve KPI metrics for steam system performance.

**Permission Required:** `kpi:read`

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period_hours` | int | 24 | Analysis period (1-720) |

#### Response

```json
{
  "success": true,
  "period_start": "2024-12-23T10:00:00Z",
  "period_end": "2024-12-24T10:00:00Z",
  "aggregation_period": "hourly",
  "energy_kpis": [
    {
      "kpi_name": "Total Steam Consumption",
      "category": "energy",
      "current_value": 15000,
      "target_value": 14000,
      "unit": "kg/h",
      "trend": "stable",
      "is_on_target": false
    },
    {
      "kpi_name": "Boiler Efficiency",
      "category": "efficiency",
      "current_value": 88.5,
      "target_value": 90.0,
      "unit": "%",
      "trend": "up",
      "trend_percent": 0.5,
      "is_on_target": false
    }
  ],
  "efficiency_kpis": [
    {
      "kpi_name": "Condensate Recovery Rate",
      "category": "efficiency",
      "current_value": 72.0,
      "target_value": 85.0,
      "unit": "%",
      "trend": "up",
      "is_on_target": false
    }
  ],
  "emissions_kpis": [
    {
      "kpi_name": "CO2 Emissions",
      "category": "emissions",
      "current_value": 2500,
      "target_value": 2200,
      "unit": "kg/h",
      "trend": "down",
      "is_on_target": false
    }
  ],
  "overall_performance_score": 78.5,
  "kpis_on_target": 1,
  "kpis_off_target": 4,
  "kpis_improving": 3,
  "kpis_declining": 1
}
```

---

### GET /climate-impact

Retrieve climate and energy impact metrics.

**Permission Required:** `kpi:read`

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period_days` | int | 30 | Analysis period (1-365) |

#### Response

```json
{
  "success": true,
  "period_start": "2024-11-24T00:00:00Z",
  "period_end": "2024-12-24T00:00:00Z",
  "energy_metrics": {
    "total_steam_consumption_kg_h": 15000,
    "total_steam_generation_kg_h": 15500,
    "total_energy_consumption_mw": 12.5,
    "boiler_efficiency_percent": 88.5,
    "system_efficiency_percent": 82.0,
    "condensate_recovery_percent": 72.0,
    "flash_steam_recovery_percent": 45.0,
    "energy_intensity_mj_per_unit": 3.2
  },
  "emissions_metrics": {
    "total_co2_emissions_kg_h": 2500,
    "co2_emissions_by_source": {
      "boiler_1": 1500,
      "boiler_2": 1000
    },
    "total_nox_emissions_kg_h": 2.5,
    "total_sox_emissions_kg_h": 0.5,
    "carbon_intensity_kg_co2_per_mwh": 200,
    "avoided_emissions_kg_co2_h": 150
  },
  "annual_emissions_target_tonnes_co2": 20000,
  "ytd_emissions_tonnes_co2": 18000,
  "on_track_for_target": true,
  "reporting_standard": "GHG Protocol",
  "verification_status": "unverified"
}
```

---

## Audit and Compliance API

### GET /audit/entries

Query audit log entries.

**Permission Required:** `audit:read`

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_time` | datetime | Start of time window (ISO 8601) |
| `end_time` | datetime | End of time window (ISO 8601) |
| `event_type` | enum | Filter by event type |
| `user_id` | string | Filter by user ID |
| `asset_id` | string | Filter by asset ID |
| `correlation_id` | string | Filter by correlation ID |
| `page` | int | Page number |
| `page_size` | int | Items per page |

#### Response

```json
{
  "entries": [
    {
      "entry_id": "550e8400-e29b-41d4-a716-446655440000",
      "event_type": "CALCULATION",
      "timestamp": "2024-12-24T10:00:00Z",
      "user_id": "operator-1",
      "agent_id": "GL-003",
      "correlation_id": "req-12345",
      "asset_id": "DESUPER-001",
      "event_data": {
        "calc_type": "steam_balance",
        "inputs_hash": "abc123...",
        "outputs_hash": "def456...",
        "formula_id": "STEAM_ENTHALPY_V1",
        "duration_ms": 15.5
      },
      "previous_hash": "000000...",
      "sequence_number": 1,
      "entry_hash": "789xyz..."
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 20
}
```

---

### POST /audit/verify-chain

Verify integrity of the audit log hash chain.

**Permission Required:** `audit:read`

#### Request Body

```json
{
  "start_sequence": 0,
  "end_sequence": 1000
}
```

#### Response

```json
{
  "is_valid": true,
  "verified_entries": 1000,
  "failed_entries": 0,
  "first_failure_sequence": null,
  "verification_timestamp": "2024-12-24T10:00:00Z",
  "errors": [],
  "warnings": [],
  "verification_duration_ms": 250.5,
  "entries_per_second": 3992.0
}
```

---

## System API

### GET /health

Health check endpoint for load balancers.

**No authentication required.**

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-12-24T10:00:00Z",
  "version": "1.0.0"
}
```

---

### GET /ready

Readiness check for Kubernetes deployments.

**No authentication required.**

#### Response

```json
{
  "status": "ready",
  "timestamp": "2024-12-24T10:00:00Z",
  "services": {
    "steam_properties": "ready",
    "optimization": "ready",
    "diagnostics": "ready",
    "database": "ready"
  }
}
```

---

### GET /metrics

Prometheus metrics endpoint.

**No authentication required.**

#### Response (text/plain)

```
# HELP gl003_requests_total Total HTTP requests
# TYPE gl003_requests_total counter
gl003_requests_total{method="POST",endpoint="/steam/properties",status="200"} 1234

# HELP gl003_request_duration_seconds Request duration histogram
# TYPE gl003_request_duration_seconds histogram
gl003_request_duration_seconds_bucket{endpoint="/steam/properties",le="0.01"} 800
gl003_request_duration_seconds_bucket{endpoint="/steam/properties",le="0.05"} 1100
gl003_request_duration_seconds_bucket{endpoint="/steam/properties",le="0.1"} 1200
```

---

## WebSocket API

### WS /ws/live-data

Real-time steam system data streaming.

**Authentication:** Token in query parameter or first message.

#### Connection

```javascript
const ws = new WebSocket('wss://api.greenlang.io/gl-003/v1/ws/live-data?token=YOUR_TOKEN');

ws.onopen = () => {
  // Subscribe to data streams
  ws.send(JSON.stringify({
    action: 'subscribe',
    streams: ['steam_properties', 'trap_diagnostics', 'kpis']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

#### Message Types

**Subscribe Request:**
```json
{
  "action": "subscribe",
  "streams": ["steam_properties", "trap_diagnostics", "kpis"],
  "assets": ["DESUPER-001", "TRAP-001"]
}
```

**Data Update:**
```json
{
  "type": "data_update",
  "stream": "steam_properties",
  "asset_id": "DESUPER-001",
  "timestamp": "2024-12-24T10:00:00Z",
  "data": {
    "inlet_temperature_c": 450.5,
    "outlet_temperature_c": 350.2,
    "spray_flow_kg_s": 1.45
  }
}
```

**Alert:**
```json
{
  "type": "alert",
  "severity": "high",
  "asset_id": "TRAP-002",
  "message": "Steam trap condition changed to LEAKING",
  "timestamp": "2024-12-24T10:00:00Z"
}
```

---

## Appendix

### A. IAPWS-IF97 Regions

| Region | Description | P Range (MPa) | T Range (K) |
|--------|-------------|---------------|-------------|
| 1 | Compressed liquid | 0 - 100 | 273.15 - 623.15 |
| 2 | Superheated vapor | 0 - 100 | 273.15 - 1073.15 |
| 3 | Near-critical | 0 - 100 | 623.15 - 863.15 |
| 4 | Two-phase | 0 - 22.064 | Saturation line |
| 5 | High-temperature | 0 - 50 | 1073.15 - 2273.15 |

### B. Steam Trap Conditions

| Condition | Description | Action |
|-----------|-------------|--------|
| `good` | Operating normally | Continue monitoring |
| `leaking` | Passing steam continuously | Repair within 2 weeks |
| `blocked` | Not passing condensate | Repair within 1 week |
| `blow_through` | Fully open, passing live steam | Immediate repair |
| `failed_open` | Stuck in open position | Immediate repair |
| `failed_closed` | Stuck in closed position | Urgent repair |
| `degraded` | Performance declining | Schedule maintenance |

### C. Optimization Objectives

| Objective | Description |
|-----------|-------------|
| `minimize_cost` | Minimize total operating cost |
| `minimize_emissions` | Minimize GHG emissions |
| `maximize_efficiency` | Maximize system efficiency |
| `minimize_energy` | Minimize energy consumption |
| `balance_cost_emissions` | Weighted balance of cost and emissions |

### D. GHG Protocol Scopes

| Scope | Description | Examples |
|-------|-------------|----------|
| Scope 1 | Direct emissions | Boiler combustion |
| Scope 2 | Indirect from purchased energy | Grid electricity |
| Scope 3 | Other indirect | Supply chain |

### E. SDK Examples

#### Python SDK

```python
from greenlang import GL003Client

client = GL003Client(
    base_url="https://api.greenlang.io/gl-003/v1",
    api_key="your-api-key"
)

# Compute steam properties
result = client.steam.compute_properties(
    pressure_kpa=1000,
    temperature_c=200
)
print(f"Enthalpy: {result.steam_state.specific_enthalpy_kj_kg} kJ/kg")

# Optimize desuperheater
opt_result = client.optimization.desuperheater(
    desuperheater_id="DESUPER-001",
    inlet_steam_pressure_kpa=4000,
    inlet_steam_temperature_c=450,
    inlet_steam_flow_kg_s=10.0,
    target_outlet_temperature_c=350,
    spray_water_pressure_kpa=5000,
    spray_water_temperature_c=80
)
print(f"Optimal spray flow: {opt_result.optimal_spray_water_flow_kg_s} kg/s")
```

#### JavaScript SDK

```javascript
import { GL003Client } from '@greenlang/gl-003-sdk';

const client = new GL003Client({
  baseUrl: 'https://api.greenlang.io/gl-003/v1',
  apiKey: 'your-api-key'
});

// Compute steam properties
const result = await client.steam.computeProperties({
  pressureKpa: 1000,
  temperatureC: 200
});
console.log(`Enthalpy: ${result.steamState.specificEnthalpyKjKg} kJ/kg`);
```

---

## Changelog

### Version 1.0.0 (2024-12-24)

- Initial API release
- Steam properties (IAPWS-IF97)
- Optimization endpoints (desuperheater, condensate, network)
- Trap diagnostics with ML-based prediction
- Root cause analysis with causal inference
- KPI and climate impact analytics
- Full audit trail with hash chain verification

---

**Support:** [support@greenlang.io](mailto:support@greenlang.io)
**Documentation:** [https://docs.greenlang.io/gl-003](https://docs.greenlang.io/gl-003)
**Status Page:** [https://status.greenlang.io](https://status.greenlang.io)
