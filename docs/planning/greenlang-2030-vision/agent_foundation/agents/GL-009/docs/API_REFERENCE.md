# GL-009 THERMALIQ - API Reference Documentation

**Agent:** GL-009 ThermalEfficiencyCalculator
**Version:** 1.0.0
**API Version:** v1
**Status:** Production Ready
**Last Updated:** 2025-11-26

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL & Endpoints](#base-url--endpoints)
4. [Common Request/Response Format](#common-requestresponse-format)
5. [Tool APIs](#tool-apis)
   - [calculate_first_law_efficiency](#calculate_first_law_efficiency)
   - [calculate_second_law_efficiency](#calculate_second_law_efficiency)
   - [calculate_heat_losses](#calculate_heat_losses)
   - [generate_sankey_diagram](#generate_sankey_diagram)
   - [benchmark_efficiency](#benchmark_efficiency)
   - [analyze_improvement_opportunities](#analyze_improvement_opportunities)
   - [quantify_uncertainty](#quantify_uncertainty)
   - [calculate_fuel_energy](#calculate_fuel_energy)
   - [calculate_steam_energy](#calculate_steam_energy)
   - [calculate_electrical_efficiency](#calculate_electrical_efficiency)
6. [Operation Modes](#operation-modes)
7. [Error Codes](#error-codes)
8. [Rate Limiting](#rate-limiting)
9. [Webhooks](#webhooks)
10. [Code Examples](#code-examples)
11. [SDK Reference](#sdk-reference)
12. [Changelog](#changelog)

---

## Overview

The THERMALIQ API provides programmatic access to all thermal efficiency calculation, analysis, and visualization capabilities. All endpoints return deterministic results for the same inputs (zero-hallucination guarantee).

**Key Features**:
- RESTful JSON API
- Deterministic physics-based calculations
- Industry-standard formulas (First Law, Second Law, Napier)
- Sankey diagram generation
- Comprehensive error handling
- Full provenance tracking with SHA-256

**API Design Principles**:
- **Idempotent**: Same request returns same result (guaranteed)
- **Stateless**: No server-side session management
- **Versioned**: `/v1/` prefix for backward compatibility
- **Self-Documenting**: OpenAPI 3.0 spec available at `/openapi.json`

**Zero-Hallucination Guarantee**:
```python
result1 = api.calculate_first_law_efficiency(data, seed=42)
result2 = api.calculate_first_law_efficiency(data, seed=42)
assert result1 == result2  # Always true - byte-exact match
```

---

## Authentication

### API Key Authentication

**Method**: HTTP Header

**Header**: `X-API-Key: <your_api_key>`

**Example**:
```bash
curl -H "X-API-Key: gl_sk_thermaliq_abc123..." \
     https://api.greenlang.org/agents/gl-009/v1/calculate_efficiency
```

**Obtaining API Key**:
1. Log in to GreenLang console: https://console.greenlang.org
2. Navigate to Settings > API Keys
3. Click "Create API Key" for GL-009 THERMALIQ
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
  -d "scope=agents:gl-009:read agents:gl-009:write"
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "agents:gl-009:read agents:gl-009:write"
}
```

**Usage**:
```bash
curl -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \
     https://api.greenlang.org/agents/gl-009/v1/calculate_efficiency
```

---

## Base URL & Endpoints

**Production**: `https://api.greenlang.org/agents/gl-009/v1`

**Staging**: `https://staging-api.greenlang.org/agents/gl-009/v1`

**Local Development**: `http://localhost:8080/v1`

### Health Check Endpoints

#### GET /health

**Description**: Basic health check (liveness probe)

**Response**:
```json
{
  "status": "healthy",
  "agent_id": "GL-009",
  "version": "1.0.0",
  "timestamp": "2025-11-26T10:30:00Z"
}
```

#### GET /ready

**Description**: Readiness check (dependencies available)

**Response**:
```json
{
  "status": "ready",
  "dependencies": {
    "cache": "connected",
    "calculators": "loaded",
    "steam_tables": "loaded",
    "fuel_database": "loaded"
  },
  "timestamp": "2025-11-26T10:30:00Z"
}
```

#### GET /metrics

**Description**: Prometheus metrics endpoint

**Response**: Prometheus text format

```
# HELP thermaliq_calculations_total Total efficiency calculations
# TYPE thermaliq_calculations_total counter
thermaliq_calculations_total{calculation_type="first_law",status="success"} 1234
thermaliq_calculations_total{calculation_type="second_law",status="success"} 567

# HELP thermaliq_calculation_duration_seconds Calculation processing time
# TYPE thermaliq_calculation_duration_seconds histogram
thermaliq_calculation_duration_seconds_bucket{le="0.1"} 450
thermaliq_calculation_duration_seconds_bucket{le="0.5"} 1150
thermaliq_calculation_duration_seconds_bucket{le="+Inf"} 1234
```

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
    "request_id": "req_thermaliq_abc123",
    "timestamp": "2025-11-26T10:30:00Z",
    "execution_time_ms": 245,
    "agent_id": "GL-009",
    "agent_version": "1.0.0",
    "deterministic": true
  },
  "provenance": {
    "input_hash": "sha256:a1b2c3d4e5f6...",
    "output_hash": "sha256:f6e5d4c3b2a1...",
    "calculation_method": "first_law_direct",
    "standards": ["ASME_PTC_4", "ISO_50001"]
  }
}
```

**Error (HTTP 4xx/5xx)**:
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Energy input must be positive",
    "details": {
      "field": "energy_input_kw",
      "value": -100,
      "constraint": "minimum: 0.1"
    }
  },
  "metadata": {
    "request_id": "req_thermaliq_abc123",
    "timestamp": "2025-11-26T10:30:00Z"
  }
}
```

---

## Tool APIs

### calculate_first_law_efficiency

**Endpoint**: `POST /v1/calculate_first_law_efficiency`

**Description**: Calculate First Law thermal efficiency based on energy conservation (Q_useful / Q_input).

**Physics Basis**: First Law of Thermodynamics (Energy Conservation)

**Formula**: `eta = (Q_useful / Q_input) x 100%`

**Standards**: ASME PTC 4, ISO 50001, DOE AMO

**Deterministic**: Yes

#### Request Body

```json
{
  "energy_input_kw": 1000.0,
  "useful_output_kw": 850.0,
  "losses_breakdown": {
    "radiation_kw": 30.0,
    "convection_kw": 20.0,
    "flue_gas_kw": 80.0,
    "unburned_fuel_kw": 20.0
  }
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `energy_input_kw` | number | Yes | 0.1 - 1,000,000 | Total energy input (kW) |
| `useful_output_kw` | number | Yes | 0 - energy_input | Useful energy output (kW) |
| `losses_breakdown` | object | No | - | Optional breakdown of losses |
| `losses_breakdown.radiation_kw` | number | No | >= 0 | Radiation losses (kW) |
| `losses_breakdown.convection_kw` | number | No | >= 0 | Convection losses (kW) |
| `losses_breakdown.conduction_kw` | number | No | >= 0 | Conduction losses (kW) |
| `losses_breakdown.flue_gas_kw` | number | No | >= 0 | Flue gas losses (kW) |
| `losses_breakdown.unburned_fuel_kw` | number | No | >= 0 | Unburned fuel losses (kW) |

#### Response

```json
{
  "status": "success",
  "data": {
    "efficiency_percent": 85.0,
    "energy_input_kw": 1000.0,
    "useful_output_kw": 850.0,
    "total_losses_kw": 150.0,
    "losses_breakdown_kw": {
      "radiation_kw": 30.0,
      "convection_kw": 20.0,
      "conduction_kw": 0.0,
      "flue_gas_kw": 80.0,
      "unburned_fuel_kw": 20.0
    },
    "losses_breakdown_percent": {
      "radiation_percent": 20.0,
      "convection_percent": 13.3,
      "conduction_percent": 0.0,
      "flue_gas_percent": 53.3,
      "unburned_fuel_percent": 13.3
    },
    "balance_error_percent": 0.0,
    "balance_closure_valid": true,
    "calculation_method": "first_law_direct",
    "standards": ["ASME_PTC_4", "ISO_50001"]
  },
  "metadata": {
    "request_id": "req_thermaliq_001",
    "timestamp": "2025-11-26T10:30:00Z",
    "execution_time_ms": 45
  },
  "provenance": {
    "input_hash": "sha256:a1b2c3d4e5f6...",
    "output_hash": "sha256:f6e5d4c3b2a1..."
  }
}
```

#### Response Schema

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `efficiency_percent` | number | 0-100 | First Law efficiency (%) |
| `energy_input_kw` | number | - | Confirmed input (kW) |
| `useful_output_kw` | number | - | Confirmed output (kW) |
| `total_losses_kw` | number | - | Sum of all losses (kW) |
| `balance_error_percent` | number | 0-100 | Energy balance closure error (%) |
| `balance_closure_valid` | boolean | - | True if error < 2% |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/calculate_first_law_efficiency \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d '{
    "energy_input_kw": 1000.0,
    "useful_output_kw": 850.0,
    "losses_breakdown": {
      "radiation_kw": 30.0,
      "convection_kw": 20.0,
      "flue_gas_kw": 80.0,
      "unburned_fuel_kw": 20.0
    }
  }'
```

---

### calculate_second_law_efficiency

**Endpoint**: `POST /v1/calculate_second_law_efficiency`

**Description**: Calculate Second Law (Exergy) efficiency based on available work potential.

**Physics Basis**: Second Law of Thermodynamics (Exergy Analysis)

**Formula**: `eta = (Exergy_out / Exergy_in) x 100%` where `Exergy = H - T0 x S`

**Standards**: ASME PTC 4.1, Kotas Exergy Method

**Deterministic**: Yes

#### Request Body

```json
{
  "enthalpy_in_kj_kg": 3200.0,
  "entropy_in_kj_kg_k": 7.5,
  "enthalpy_out_kj_kg": 2800.0,
  "entropy_out_kj_kg_k": 7.2,
  "mass_flow_kg_s": 10.0,
  "ambient_temperature_k": 298.15
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `enthalpy_in_kj_kg` | number | Yes | - | Inlet enthalpy (kJ/kg) |
| `entropy_in_kj_kg_k` | number | Yes | >= 0 | Inlet entropy (kJ/kg-K) |
| `enthalpy_out_kj_kg` | number | Yes | - | Outlet enthalpy (kJ/kg) |
| `entropy_out_kj_kg_k` | number | Yes | >= 0 | Outlet entropy (kJ/kg-K) |
| `mass_flow_kg_s` | number | Yes | > 0 | Mass flow rate (kg/s) |
| `ambient_temperature_k` | number | No | 250-350 | Reference temperature (K), default 298.15 |

#### Response

```json
{
  "status": "success",
  "data": {
    "exergy_efficiency_percent": 72.5,
    "exergy_input_kw": 9635.0,
    "exergy_output_kw": 6985.4,
    "exergy_destruction_kw": 2649.6,
    "irreversibility_percent": 27.5,
    "specific_exergy_in_kj_kg": 963.5,
    "specific_exergy_out_kj_kg": 698.54,
    "ambient_temperature_k": 298.15,
    "calculation_method": "exergy_analysis",
    "standards": ["ASME_PTC_4.1", "Kotas_Method"]
  },
  "metadata": {
    "request_id": "req_thermaliq_002",
    "timestamp": "2025-11-26T10:31:00Z",
    "execution_time_ms": 52
  }
}
```

#### Response Schema

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `exergy_efficiency_percent` | number | 0-100 | Second Law efficiency (%) |
| `exergy_input_kw` | number | - | Exergy input rate (kW) |
| `exergy_output_kw` | number | - | Exergy output rate (kW) |
| `exergy_destruction_kw` | number | - | Exergy destroyed (irreversibility) (kW) |
| `irreversibility_percent` | number | 0-100 | Percent of exergy destroyed (%) |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/calculate_second_law_efficiency \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d '{
    "enthalpy_in_kj_kg": 3200.0,
    "entropy_in_kj_kg_k": 7.5,
    "enthalpy_out_kj_kg": 2800.0,
    "entropy_out_kj_kg_k": 7.2,
    "mass_flow_kg_s": 10.0
  }'
```

---

### calculate_heat_losses

**Endpoint**: `POST /v1/calculate_heat_losses`

**Description**: Calculate comprehensive heat loss breakdown (radiation, convection, conduction, flue gas, unburned fuel).

**Physics Basis**: Stefan-Boltzmann Law, Newton's Law of Cooling, Fourier's Law

**Standards**: ASME PTC 4, Incropera & DeWitt Heat Transfer, DOE AMO

**Deterministic**: Yes

#### Request Body

```json
{
  "surface_area_m2": 50.0,
  "surface_temp_c": 200.0,
  "ambient_temp_c": 25.0,
  "emissivity": 0.85,
  "heat_transfer_coeff_w_m2k": 10.0,
  "flue_gas": {
    "flow_rate_kg_hr": 5000.0,
    "temperature_c": 180.0,
    "specific_heat_kj_kgk": 1.1
  }
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `surface_area_m2` | number | Yes | 0.1 - 10000 | Equipment surface area (m2) |
| `surface_temp_c` | number | Yes | -50 to 1500 | Surface temperature (C) |
| `ambient_temp_c` | number | Yes | -50 to 60 | Ambient temperature (C) |
| `emissivity` | number | No | 0.1 - 1.0 | Surface emissivity, default 0.85 |
| `heat_transfer_coeff_w_m2k` | number | No | 1 - 1000 | Convection coefficient (W/m2-K), default 10 |
| `flue_gas.flow_rate_kg_hr` | number | No | >= 0 | Flue gas flow rate (kg/hr) |
| `flue_gas.temperature_c` | number | No | - | Flue gas temperature (C) |
| `flue_gas.specific_heat_kj_kgk` | number | No | 0.5 - 2.0 | Flue gas specific heat, default 1.1 |

#### Response

```json
{
  "status": "success",
  "data": {
    "total_loss_kw": 317.2,
    "radiation_loss_kw": 142.5,
    "convection_loss_kw": 87.5,
    "conduction_loss_kw": 0.0,
    "flue_gas_loss_kw": 236.8,
    "unburned_fuel_loss_kw": 0.0,
    "loss_breakdown_percent": {
      "radiation_percent": 30.5,
      "convection_percent": 18.7,
      "conduction_percent": 0.0,
      "flue_gas_percent": 50.7,
      "unburned_fuel_percent": 0.0
    },
    "calculation_details": {
      "radiation": {
        "formula": "stefan_boltzmann",
        "emissivity": 0.85,
        "t_surface_k": 473.15,
        "t_ambient_k": 298.15
      },
      "convection": {
        "formula": "newton_cooling",
        "heat_transfer_coeff": 10.0,
        "delta_t_c": 175.0
      },
      "flue_gas": {
        "formula": "sensible_heat",
        "mass_flow_kg_s": 1.389,
        "delta_t_c": 155.0
      }
    },
    "standards": ["ASME_PTC_4", "Incropera_DeWitt"]
  },
  "metadata": {
    "request_id": "req_thermaliq_003",
    "timestamp": "2025-11-26T10:32:00Z",
    "execution_time_ms": 38
  }
}
```

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/calculate_heat_losses \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d '{
    "surface_area_m2": 50.0,
    "surface_temp_c": 200.0,
    "ambient_temp_c": 25.0,
    "emissivity": 0.85,
    "flue_gas": {
      "flow_rate_kg_hr": 5000.0,
      "temperature_c": 180.0
    }
  }'
```

---

### generate_sankey_diagram

**Endpoint**: `POST /v1/generate_sankey_diagram`

**Description**: Generate interactive Sankey energy flow diagram with energy balance validation.

**Technology**: Plotly.js

**Output Formats**: JSON (Plotly spec), HTML, PNG, SVG

**Deterministic**: Yes

#### Request Body

```json
{
  "energy_inputs": {
    "natural_gas": 900.0,
    "electrical": 100.0
  },
  "useful_outputs": {
    "steam_production": 720.0,
    "hot_water": 80.0
  },
  "losses": {
    "flue_gas": 120.0,
    "radiation": 40.0,
    "convection": 25.0,
    "blowdown": 15.0
  },
  "title": "Boiler Energy Balance",
  "output_format": "json"
}
```

#### Request Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `energy_inputs` | object | Yes | Input energy flows (name: value_kw) |
| `useful_outputs` | object | Yes | Useful output flows (name: value_kw) |
| `losses` | object | Yes | Loss flows (name: value_kw) |
| `title` | string | No | Diagram title, default "Energy Balance" |
| `output_format` | string | No | json, html, png, svg - default "json" |

#### Response

```json
{
  "status": "success",
  "data": {
    "diagram_type": "sankey",
    "title": "Boiler Energy Balance",
    "nodes": {
      "labels": ["Natural Gas", "Electrical", "Process", "Steam Production",
                 "Hot Water", "Flue Gas Loss", "Radiation Loss",
                 "Convection Loss", "Blowdown Loss"],
      "colors": ["#2ecc71", "#2ecc71", "#3498db", "#f39c12", "#f39c12",
                 "#e74c3c", "#e74c3c", "#e74c3c", "#e74c3c"]
    },
    "links": {
      "sources": [0, 1, 2, 2, 2, 2, 2, 2],
      "targets": [2, 2, 3, 4, 5, 6, 7, 8],
      "values": [900, 100, 720, 80, 120, 40, 25, 15],
      "colors": ["rgba(46, 204, 113, 0.5)", "rgba(46, 204, 113, 0.5)",
                 "rgba(243, 156, 18, 0.5)", "rgba(243, 156, 18, 0.5)",
                 "rgba(231, 76, 60, 0.5)", "rgba(231, 76, 60, 0.5)",
                 "rgba(231, 76, 60, 0.5)", "rgba(231, 76, 60, 0.5)"]
    },
    "energy_balance": {
      "total_input_kw": 1000.0,
      "total_output_kw": 800.0,
      "total_losses_kw": 200.0,
      "efficiency_percent": 80.0,
      "balance_error_percent": 0.0,
      "balance_valid": true
    },
    "plotly_figure": {
      "data": [...],
      "layout": {...}
    }
  },
  "metadata": {
    "request_id": "req_thermaliq_004",
    "timestamp": "2025-11-26T10:33:00Z",
    "execution_time_ms": 320
  }
}
```

#### Energy Balance Validation

The Sankey generator enforces energy conservation:

```
SUM(energy_inputs) = SUM(useful_outputs) + SUM(losses)
```

**Validation Rule**:
- Balance error must be < 2% or request is rejected
- Error message includes the imbalance amount

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/generate_sankey_diagram \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d '{
    "energy_inputs": {"natural_gas": 900.0, "electrical": 100.0},
    "useful_outputs": {"steam_production": 720.0, "hot_water": 80.0},
    "losses": {"flue_gas": 120.0, "radiation": 40.0, "convection": 25.0, "blowdown": 15.0},
    "title": "Boiler Energy Balance"
  }'
```

---

### benchmark_efficiency

**Endpoint**: `POST /v1/benchmark_efficiency`

**Description**: Compare efficiency against industry benchmarks and best practices.

**Data Sources**: DOE AMO, EPA ENERGY STAR, IEA Industrial Efficiency

**Deterministic**: Yes

#### Request Body

```json
{
  "efficiency_percent": 82.5,
  "equipment_type": "boiler_steam",
  "custom_benchmark": null
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `efficiency_percent` | number | Yes | 0 - 100 | Current efficiency (%) |
| `equipment_type` | string | Yes | Enum | Equipment type for benchmark lookup |
| `custom_benchmark` | object | No | - | Optional custom benchmark values |

**equipment_type Enum**:
- `boiler_steam` - Steam boilers
- `furnace_process` - Process heating furnaces
- `heat_exchanger` - Heat exchangers
- `cogeneration_chp` - Combined heat and power

#### Response

```json
{
  "status": "success",
  "data": {
    "current_efficiency_percent": 82.5,
    "equipment_type": "boiler_steam",
    "percentile_ranking": 68.5,
    "ranking_category": "Above Average",
    "benchmark_values": {
      "bottom_quartile": 70.0,
      "median": 80.0,
      "top_quartile": 85.0,
      "best_in_class": 92.0,
      "theoretical_max": 95.0
    },
    "improvement_potential": {
      "to_median_percent": -2.5,
      "to_top_quartile_percent": 2.5,
      "to_best_in_class_percent": 9.5,
      "to_theoretical_max_percent": 12.5
    },
    "recommendations": [
      "Current efficiency is above median - good performance",
      "Achievable improvement of 2.5% to reach top quartile",
      "Best-in-class performance requires 9.5% improvement",
      "Consider economizer, air preheater, or condensing upgrades"
    ],
    "data_source": "DOE_AMO_Steam_Best_Practices",
    "standards": ["DOE_AMO", "EPA_ENERGY_STAR", "ISO_50001"]
  },
  "metadata": {
    "request_id": "req_thermaliq_005",
    "timestamp": "2025-11-26T10:34:00Z",
    "execution_time_ms": 28
  }
}
```

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/benchmark_efficiency \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d '{
    "efficiency_percent": 82.5,
    "equipment_type": "boiler_steam"
  }'
```

---

### analyze_improvement_opportunities

**Endpoint**: `POST /v1/analyze_improvement_opportunities`

**Description**: Identify and prioritize efficiency improvement opportunities with ROI analysis.

**Deterministic**: Yes

#### Request Body

```json
{
  "current_efficiency_percent": 80.0,
  "energy_input_kw": 1000.0,
  "losses_breakdown_kw": {
    "flue_gas_kw": 100.0,
    "radiation_kw": 40.0,
    "convection_kw": 30.0,
    "blowdown_kw": 30.0
  },
  "operating_hours_per_year": 8000,
  "energy_cost_usd_per_kwh": 0.08,
  "equipment_age_years": 15
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "current_state": {
      "efficiency_percent": 80.0,
      "annual_energy_cost_usd": 640000.0,
      "annual_losses_kw": 200.0,
      "annual_loss_cost_usd": 128000.0
    },
    "improvement_opportunities": [
      {
        "rank": 1,
        "opportunity": "Install economizer",
        "target_loss": "flue_gas",
        "efficiency_gain_percent": 5.0,
        "annual_savings_usd": 32000.0,
        "implementation_cost_usd": 75000.0,
        "simple_payback_years": 2.3,
        "npv_10yr_usd": 185000.0,
        "technical_feasibility": "high"
      },
      {
        "rank": 2,
        "opportunity": "Improve insulation",
        "target_loss": "radiation_convection",
        "efficiency_gain_percent": 2.5,
        "annual_savings_usd": 16000.0,
        "implementation_cost_usd": 25000.0,
        "simple_payback_years": 1.6,
        "npv_10yr_usd": 95000.0,
        "technical_feasibility": "high"
      },
      {
        "rank": 3,
        "opportunity": "Optimize blowdown",
        "target_loss": "blowdown",
        "efficiency_gain_percent": 1.5,
        "annual_savings_usd": 9600.0,
        "implementation_cost_usd": 15000.0,
        "simple_payback_years": 1.6,
        "npv_10yr_usd": 57000.0,
        "technical_feasibility": "high"
      }
    ],
    "combined_potential": {
      "max_efficiency_percent": 89.0,
      "total_annual_savings_usd": 57600.0,
      "total_implementation_cost_usd": 115000.0,
      "combined_payback_years": 2.0
    },
    "recommendations": [
      "Prioritize economizer installation for highest savings impact",
      "Insulation improvements offer quick payback with low risk",
      "Consider comprehensive upgrade package for optimal ROI"
    ]
  },
  "metadata": {
    "request_id": "req_thermaliq_006",
    "timestamp": "2025-11-26T10:35:00Z",
    "execution_time_ms": 85
  }
}
```

---

### quantify_uncertainty

**Endpoint**: `POST /v1/quantify_uncertainty`

**Description**: Quantify measurement uncertainty in efficiency calculations using Monte Carlo simulation.

**Method**: Monte Carlo simulation with deterministic seed (GUM compliant)

**Standards**: GUM (Guide to Uncertainty in Measurement), ISO/IEC Guide 98-3

**Deterministic**: Yes (fixed seed)

#### Request Body

```json
{
  "efficiency_percent": 85.0,
  "input_uncertainties": {
    "energy_input_percent": 2.0,
    "useful_output_percent": 1.5,
    "temperature_percent": 0.5
  },
  "n_simulations": 10000,
  "seed": 42
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "efficiency_percent": 85.0,
    "combined_uncertainty_percent": 2.55,
    "expanded_uncertainty_k2": 5.1,
    "confidence_intervals": {
      "68_percent": {
        "lower": 82.8,
        "upper": 87.2
      },
      "90_percent": {
        "lower": 80.8,
        "upper": 89.2
      },
      "95_percent": {
        "lower": 79.8,
        "upper": 90.1
      }
    },
    "input_uncertainties": {
      "energy_input_percent": 2.0,
      "useful_output_percent": 1.5,
      "temperature_percent": 0.5
    },
    "sensitivity_analysis": {
      "energy_input": 0.78,
      "useful_output": 0.59,
      "temperature": 0.20
    },
    "n_simulations": 10000,
    "seed": 42,
    "deterministic": true,
    "standards": ["GUM", "ISO_IEC_Guide_98-3"]
  },
  "metadata": {
    "request_id": "req_thermaliq_007",
    "timestamp": "2025-11-26T10:36:00Z",
    "execution_time_ms": 450
  }
}
```

---

### calculate_fuel_energy

**Endpoint**: `POST /v1/calculate_fuel_energy`

**Description**: Calculate fuel energy input using Higher/Lower Heating Values (HHV/LHV).

**Data Sources**: API, ASTM D240, DOE Biomass

**Deterministic**: Yes

#### Request Body

```json
{
  "fuel_type": "natural_gas",
  "mass_flow_kg_hr": 100.0,
  "use_hhv": true,
  "custom_heating_value_mj_kg": null
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `fuel_type` | string | Yes | Enum | Fuel type |
| `mass_flow_kg_hr` | number | Yes | > 0 | Fuel mass flow rate (kg/hr) |
| `use_hhv` | boolean | No | - | Use HHV (true) or LHV (false), default true |
| `custom_heating_value_mj_kg` | number | No | > 0 | Override heating value (MJ/kg) |

**fuel_type Enum**:
- `natural_gas`, `propane`, `diesel`, `fuel_oil_2`, `fuel_oil_6`
- `coal_bituminous`, `coal_anthracite`
- `wood_chips`, `biomass_pellets`, `hydrogen`

#### Response

```json
{
  "status": "success",
  "data": {
    "fuel_type": "natural_gas",
    "mass_flow_kg_hr": 100.0,
    "heating_value_mj_kg": 55.5,
    "heating_value_type": "HHV",
    "energy_input_kw": 1541.67,
    "energy_input_mj_hr": 5550.0,
    "energy_input_mmbtu_hr": 5.26,
    "annual_energy_gj": 48618.0,
    "co2_emission_factor_kg_per_gj": 56.1,
    "annual_co2_kg": 2727430.0,
    "data_source": "API_ASTM_D240",
    "standards": ["API", "ASTM_D240"]
  },
  "metadata": {
    "request_id": "req_thermaliq_008",
    "timestamp": "2025-11-26T10:37:00Z",
    "execution_time_ms": 18
  }
}
```

---

### calculate_steam_energy

**Endpoint**: `POST /v1/calculate_steam_energy`

**Description**: Calculate steam enthalpy and energy using IAPWS-IF97 steam tables.

**Physics Basis**: IAPWS-IF97 Industrial Formulation for Steam Properties

**Deterministic**: Yes

#### Request Body

```json
{
  "pressure_bar": 10.0,
  "temperature_c": 250.0,
  "mass_flow_kg_hr": 5000.0,
  "reference_state": {
    "pressure_bar": 1.0,
    "temperature_c": 25.0
  }
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "steam_properties": {
      "pressure_bar": 10.0,
      "temperature_c": 250.0,
      "enthalpy_kj_kg": 2943.0,
      "entropy_kj_kg_k": 6.925,
      "specific_volume_m3_kg": 0.2328,
      "phase": "superheated_steam"
    },
    "reference_state": {
      "enthalpy_kj_kg": 104.9,
      "entropy_kj_kg_k": 0.367
    },
    "energy_flow": {
      "mass_flow_kg_hr": 5000.0,
      "mass_flow_kg_s": 1.389,
      "enthalpy_flow_kw": 4087.5,
      "net_enthalpy_kw": 3942.0,
      "exergy_flow_kw": 3450.0
    },
    "annual_energy_gj": 124356.0,
    "steam_tables_source": "IAPWS_IF97",
    "standards": ["IAPWS_IF97", "ASME_Steam_Tables"]
  },
  "metadata": {
    "request_id": "req_thermaliq_009",
    "timestamp": "2025-11-26T10:38:00Z",
    "execution_time_ms": 65
  }
}
```

---

### calculate_electrical_efficiency

**Endpoint**: `POST /v1/calculate_electrical_efficiency`

**Description**: Calculate motor and pump electrical efficiency and energy consumption.

**Standards**: NEMA MG 1, IEC 60034-30-1

**Deterministic**: Yes

#### Request Body

```json
{
  "motor_power_kw": 75.0,
  "motor_efficiency_percent": 95.5,
  "load_factor_percent": 80.0,
  "pump_efficiency_percent": 82.0,
  "vfd_efficiency_percent": 97.0,
  "operating_hours_per_year": 8000
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "motor": {
      "rated_power_kw": 75.0,
      "efficiency_percent": 95.5,
      "load_factor_percent": 80.0,
      "shaft_power_kw": 60.0,
      "electrical_input_kw": 62.83,
      "losses_kw": 2.83
    },
    "pump": {
      "efficiency_percent": 82.0,
      "hydraulic_power_kw": 49.2
    },
    "vfd": {
      "efficiency_percent": 97.0,
      "electrical_input_with_vfd_kw": 64.77,
      "vfd_losses_kw": 1.94
    },
    "overall_system": {
      "wire_to_water_efficiency_percent": 75.94,
      "electrical_input_kw": 64.77,
      "useful_hydraulic_kw": 49.2,
      "total_losses_kw": 15.57
    },
    "annual_energy": {
      "consumption_kwh": 518160,
      "cost_at_0_10_usd": 51816.0
    },
    "efficiency_class": "IE3",
    "standards": ["NEMA_MG1", "IEC_60034-30-1"]
  },
  "metadata": {
    "request_id": "req_thermaliq_010",
    "timestamp": "2025-11-26T10:39:00Z",
    "execution_time_ms": 32
  }
}
```

---

## Operation Modes

### Calculate Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "calculate",
  "calculation_type": "first_law",
  "data": {
    "energy_input_kw": 1000.0,
    "useful_output_kw": 850.0
  }
}
```

**Description**: Core efficiency calculation. Returns First Law or Second Law efficiency.

---

### Analyze Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "analyze",
  "data": {
    "efficiency_percent": 80.0,
    "losses_breakdown_kw": {...}
  }
}
```

**Description**: Detailed loss breakdown analysis with improvement recommendations.

---

### Benchmark Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "benchmark",
  "data": {
    "efficiency_percent": 82.5,
    "equipment_type": "boiler_steam"
  }
}
```

**Description**: Industry benchmark comparison with percentile ranking.

---

### Visualize Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "visualize",
  "visualization_type": "sankey",
  "data": {
    "energy_inputs": {...},
    "useful_outputs": {...},
    "losses": {...}
  }
}
```

**Description**: Generate Sankey diagram, waterfall chart, or trend analysis.

---

### Report Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "report",
  "report_format": "pdf",
  "include_sections": ["efficiency", "losses", "benchmark", "sankey", "recommendations"],
  "data": {...}
}
```

**Description**: Generate comprehensive PDF/HTML report with all analyses.

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
| 422 | Unprocessable Entity | Validation failed (e.g., energy balance error) |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Agent error |
| 503 | Service Unavailable | Agent offline |

### Application Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `VALIDATION_ERROR` | Input validation failed | Check field constraints |
| `ENERGY_BALANCE_ERROR` | Energy balance > 2% closure error | Verify input/output/loss values |
| `INVALID_FUEL_TYPE` | Unknown fuel type | Use valid fuel type enum |
| `INVALID_EQUIPMENT_TYPE` | Unknown equipment type | Use valid equipment type enum |
| `CALCULATION_ERROR` | Physics calculation failed | Check input ranges |
| `STEAM_TABLE_ERROR` | Steam properties lookup failed | Verify pressure/temperature range |
| `BENCHMARK_NOT_FOUND` | No benchmark data for equipment | Use custom_benchmark |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |

### Example Error Response

```json
{
  "status": "error",
  "error": {
    "code": "ENERGY_BALANCE_ERROR",
    "message": "Energy balance closure error exceeds 2% tolerance",
    "details": {
      "total_input_kw": 1000.0,
      "total_output_kw": 850.0,
      "total_losses_kw": 100.0,
      "expected_losses_kw": 150.0,
      "balance_error_percent": 5.0,
      "tolerance_percent": 2.0
    }
  },
  "metadata": {
    "request_id": "req_thermaliq_err001",
    "timestamp": "2025-11-26T10:40:00Z"
  }
}
```

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
X-RateLimit-Reset: 1732616400
```

### Rate Limit Exceeded Response

**HTTP 429**:
```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 2025-11-26T10:45:00Z",
    "retry_after_seconds": 120
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
  "url": "https://your-app.com/webhooks/thermaliq",
  "events": ["efficiency.calculated", "efficiency.threshold_crossed"],
  "secret": "whsec_your_secret"
}
```

### Webhook Events

#### efficiency.calculated

**Payload**:
```json
{
  "event": "efficiency.calculated",
  "timestamp": "2025-11-26T10:41:00Z",
  "data": {
    "equipment_id": "boiler-001",
    "efficiency_percent": 82.5,
    "calculation_type": "first_law",
    "provenance_hash": "sha256:abc123..."
  }
}
```

#### efficiency.threshold_crossed

**Payload**:
```json
{
  "event": "efficiency.threshold_crossed",
  "timestamp": "2025-11-26T10:42:00Z",
  "data": {
    "equipment_id": "boiler-001",
    "efficiency_percent": 74.5,
    "threshold_percent": 75.0,
    "direction": "below",
    "alert_level": "warning"
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
from greenlang import ThermalEfficiencyCalculator

# Initialize agent
agent = ThermalEfficiencyCalculator(api_key="gl_sk_thermaliq_abc123...")

# First Law efficiency calculation
result = agent.calculate_first_law_efficiency(
    energy_input_kw=1000.0,
    useful_output_kw=850.0,
    losses_breakdown={
        "radiation_kw": 30.0,
        "convection_kw": 20.0,
        "flue_gas_kw": 80.0,
        "unburned_fuel_kw": 20.0
    }
)

print(f"Efficiency: {result.efficiency_percent}%")
print(f"Balance valid: {result.balance_closure_valid}")

# Generate Sankey diagram
sankey = agent.generate_sankey_diagram(
    energy_inputs={"natural_gas": 900.0, "electrical": 100.0},
    useful_outputs={"steam": 720.0, "hot_water": 80.0},
    losses={"flue_gas": 120.0, "radiation": 40.0, "convection": 40.0, "blowdown": 20.0},
    title="Boiler Energy Balance"
)

# Save Sankey as HTML
with open("sankey.html", "w") as f:
    f.write(sankey.to_html())

# Benchmark comparison
benchmark = agent.benchmark_efficiency(
    efficiency_percent=82.5,
    equipment_type="boiler_steam"
)

print(f"Percentile: {benchmark.percentile_ranking}")
print(f"Improvement potential: {benchmark.improvement_potential.to_best_in_class_percent}%")
```

### JavaScript/TypeScript SDK

```typescript
import { ThermalEfficiencyCalculator } from '@greenlang/agents';

const agent = new ThermalEfficiencyCalculator({ apiKey: 'gl_sk_thermaliq_abc123...' });

// First Law efficiency
const efficiency = await agent.calculateFirstLawEfficiency({
  energyInputKw: 1000.0,
  usefulOutputKw: 850.0,
});

console.log(`Efficiency: ${efficiency.efficiencyPercent}%`);

// Uncertainty quantification
const uncertainty = await agent.quantifyUncertainty({
  efficiencyPercent: 85.0,
  inputUncertainties: {
    energyInputPercent: 2.0,
    usefulOutputPercent: 1.5,
  },
});

console.log(`95% CI: [${uncertainty.confidenceIntervals['95_percent'].lower}, ${uncertainty.confidenceIntervals['95_percent'].upper}]`);
```

### cURL Examples

**Calculate First Law Efficiency**:
```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/calculate_first_law_efficiency \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d '{
    "energy_input_kw": 1000.0,
    "useful_output_kw": 850.0
  }'
```

**Generate Sankey Diagram**:
```bash
curl -X POST https://api.greenlang.org/agents/gl-009/v1/generate_sankey_diagram \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_thermaliq_abc123..." \
  -d @- <<EOF
{
  "energy_inputs": {"natural_gas": 900, "electrical": 100},
  "useful_outputs": {"steam": 720, "hot_water": 80},
  "losses": {"flue_gas": 120, "radiation": 40, "convection": 25, "blowdown": 15},
  "title": "Boiler Energy Balance"
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

**Documentation**: https://docs.greenlang.org/sdks/python/gl-009

### JavaScript/TypeScript SDK

**Installation**:
```bash
npm install @greenlang/agents
```

**Documentation**: https://docs.greenlang.org/sdks/javascript/gl-009

### Go SDK

**Installation**:
```bash
go get github.com/greenlang/agents-go/gl009
```

**Documentation**: https://docs.greenlang.org/sdks/go/gl-009

---

## Changelog

### v1.0.0 (2025-11-26)

**Initial Release**

- 10 deterministic calculation tools
- First Law and Second Law efficiency calculations
- Comprehensive heat loss analysis (radiation, convection, conduction, flue gas)
- Interactive Sankey diagram generation (Plotly)
- Industry benchmark comparison (DOE AMO, EPA ENERGY STAR)
- Improvement opportunity analysis with ROI
- Monte Carlo uncertainty quantification
- Fuel energy calculator (HHV/LHV)
- Steam energy calculator (IAPWS-IF97)
- Electrical motor/pump efficiency
- Full provenance tracking (SHA-256)
- Industry standards compliance (ASME PTC 4, ISO 50001)

---

**API Version**: v1
**Documentation Version**: 1.0.0
**Last Updated**: 2025-11-26
**Support**: api-support@greenlang.org
**License**: Apache-2.0

---

For questions, bug reports, or feature requests, please contact api-support@greenlang.org or open an issue at https://github.com/greenlang/agents/issues
