# GL-010 EMISSIONWATCH - API Reference Documentation

**Agent:** GL-010 EmissionsComplianceAgent
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
   - [calculate_nox_emissions](#calculate_nox_emissions)
   - [calculate_sox_emissions](#calculate_sox_emissions)
   - [calculate_co2_emissions](#calculate_co2_emissions)
   - [calculate_pm_emissions](#calculate_pm_emissions)
   - [calculate_co_emissions](#calculate_co_emissions)
   - [check_compliance](#check_compliance)
   - [detect_violations](#detect_violations)
   - [calculate_dispersion](#calculate_dispersion)
   - [generate_report](#generate_report)
   - [configure_alerts](#configure_alerts)
   - [get_emission_factors](#get_emission_factors)
   - [analyze_fuel](#analyze_fuel)
6. [Operation Modes](#operation-modes)
7. [Error Codes](#error-codes)
8. [Rate Limiting](#rate-limiting)
9. [Webhooks](#webhooks)
10. [Real-Time Streaming](#real-time-streaming)
11. [Code Examples](#code-examples)
12. [SDK Reference](#sdk-reference)
13. [Changelog](#changelog)

---

## Overview

The EMISSIONWATCH API provides programmatic access to all emissions calculation, compliance monitoring, violation detection, and regulatory reporting capabilities. All endpoints return deterministic results for the same inputs (zero-hallucination guarantee).

**Key Features**:
- RESTful JSON API
- Deterministic physics-based calculations (EPA Method 19, combustion stoichiometry)
- Multi-jurisdiction compliance (US EPA, EU IED, China MEE)
- Real-time CEMS data processing
- Violation detection with multi-channel alerting
- Regulatory report generation (EPA CEDRI, EU ETS, XBRL)
- Full provenance tracking with SHA-256

**API Design Principles**:
- **Idempotent**: Same request returns same result (guaranteed)
- **Stateless**: No server-side session management
- **Versioned**: `/v1/` prefix for backward compatibility
- **Self-Documenting**: OpenAPI 3.0 spec available at `/openapi.json`

**Zero-Hallucination Guarantee**:
```python
result1 = api.calculate_nox_emissions(data, seed=42)
result2 = api.calculate_nox_emissions(data, seed=42)
assert result1 == result2  # Always true - byte-exact match
```

---

## Authentication

### API Key Authentication

**Method**: HTTP Header

**Header**: `X-API-Key: <your_api_key>`

**Example**:
```bash
curl -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
     https://api.greenlang.org/agents/gl-010/v1/calculate_nox
```

**Obtaining API Key**:
1. Log in to GreenLang console: https://console.greenlang.org
2. Navigate to Settings > API Keys
3. Click "Create API Key" for GL-010 EMISSIONWATCH
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
  -d "scope=agents:gl-010:read agents:gl-010:write"
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "agents:gl-010:read agents:gl-010:write"
}
```

**Usage**:
```bash
curl -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \
     https://api.greenlang.org/agents/gl-010/v1/calculate_nox
```

---

## Base URL & Endpoints

**Production**: `https://api.greenlang.org/agents/gl-010/v1`

**Staging**: `https://staging-api.greenlang.org/agents/gl-010/v1`

**Local Development**: `http://localhost:8080/v1`

### Health Check Endpoints

#### GET /health

**Description**: Basic health check (liveness probe)

**Response**:
```json
{
  "status": "healthy",
  "agent_id": "GL-010",
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
    "database": "connected",
    "cache": "connected",
    "cems_connector": "connected",
    "emission_factors": "loaded",
    "compliance_rules": "loaded"
  },
  "timestamp": "2025-11-26T10:30:00Z"
}
```

#### GET /metrics

**Description**: Prometheus metrics endpoint

**Response**: Prometheus text format

```
# HELP emissionwatch_calculations_total Total emissions calculations
# TYPE emissionwatch_calculations_total counter
emissionwatch_calculations_total{pollutant="NOx",method="epa_method_19",status="success"} 12345
emissionwatch_calculations_total{pollutant="SO2",method="sulfur_balance",status="success"} 8765

# HELP emissionwatch_violations_total Total violations detected
# TYPE emissionwatch_violations_total counter
emissionwatch_violations_total{pollutant="NOx",severity="major",jurisdiction="epa"} 23

# HELP emissionwatch_calculation_duration_seconds Calculation processing time
# TYPE emissionwatch_calculation_duration_seconds histogram
emissionwatch_calculation_duration_seconds_bucket{le="0.05"} 9500
emissionwatch_calculation_duration_seconds_bucket{le="0.1"} 11800
emissionwatch_calculation_duration_seconds_bucket{le="+Inf"} 12345
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
| `X-Facility-ID` | No | Facility identifier for multi-site deployments |

### Standard Response Structure

**Success (HTTP 200)**:
```json
{
  "status": "success",
  "data": {
    // Tool-specific output
  },
  "metadata": {
    "request_id": "req_emissionwatch_abc123",
    "timestamp": "2025-11-26T10:30:00Z",
    "execution_time_ms": 45,
    "agent_id": "GL-010",
    "agent_version": "1.0.0",
    "deterministic": true
  },
  "provenance": {
    "input_hash": "sha256:a1b2c3d4e5f6...",
    "output_hash": "sha256:f6e5d4c3b2a1...",
    "calculation_method": "epa_method_19",
    "standards": ["40_CFR_Part_60_App_A", "Method_19"]
  }
}
```

**Error (HTTP 4xx/5xx)**:
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "NOx concentration exceeds valid range",
    "details": {
      "field": "nox_ppm",
      "value": 6000,
      "constraint": "Must be 0-5000 ppm"
    }
  },
  "metadata": {
    "request_id": "req_emissionwatch_abc123",
    "timestamp": "2025-11-26T10:30:00Z"
  }
}
```

---

## Tool APIs

### calculate_nox_emissions

**Endpoint**: `POST /v1/calculate_nox_emissions`

**Description**: Calculate NOx emissions using EPA Method 19 F-factor approach.

**Physics Basis**: EPA Method 19 F-factor method

**Formula**: `E_NOx = C_NOx * F_d * (20.9 / (20.9 - %O2)) * Q_fuel`

**Standards**: 40 CFR Part 60, Appendix A, Method 19

**Deterministic**: Yes

#### Request Body

```json
{
  "nox_ppm": 150.0,
  "o2_percent": 3.5,
  "fuel_type": "natural_gas",
  "fuel_rate_mmbtu_hr": 100.0,
  "reference_o2": 3.0,
  "measurement_basis": "dry"
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `nox_ppm` | number | Yes | 0 - 5000 | Measured NOx concentration (ppmvd) |
| `o2_percent` | number | Yes | 0 - 20.9 | Stack O2 percentage (dry basis) |
| `fuel_type` | string | Yes | Enum | Type of fuel being burned |
| `fuel_rate_mmbtu_hr` | number | Yes | > 0 | Fuel firing rate (MMBtu/hr) |
| `reference_o2` | number | No | 0 - 15, default: 3.0 | Reference O2 for correction (%) |
| `measurement_basis` | string | No | Enum: dry, wet | Measurement basis, default: dry |

**fuel_type Enum**:
- `natural_gas`, `propane`, `butane`
- `no1_oil`, `no2_oil`, `no4_oil`, `no5_oil`, `no6_oil`
- `bituminous_coal`, `subbituminous_coal`, `lignite`, `anthracite`
- `wood_biomass`

#### Response

```json
{
  "status": "success",
  "data": {
    "nox_ppm_measured": 150.0,
    "nox_ppm_corrected": 152.87,
    "nox_lb_per_mmbtu": 0.182,
    "nox_lb_per_hr": 18.2,
    "nox_kg_per_hr": 8.26,
    "nox_tons_per_year": 79.72,
    "o2_correction_factor": 1.019,
    "f_factor_used": 8710,
    "fuel_type": "natural_gas",
    "calculation_method": "epa_method_19",
    "standards": ["40_CFR_Part_60_App_A", "Method_19"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_001",
    "timestamp": "2025-11-26T10:30:00Z",
    "execution_time_ms": 35
  },
  "provenance": {
    "input_hash": "sha256:a1b2c3d4e5f6...",
    "output_hash": "sha256:f6e5d4c3b2a1..."
  }
}
```

#### Response Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `nox_ppm_measured` | number | ppm | Original measured NOx concentration |
| `nox_ppm_corrected` | number | ppm | O2-corrected NOx concentration |
| `nox_lb_per_mmbtu` | number | lb/MMBtu | NOx emission rate |
| `nox_lb_per_hr` | number | lb/hr | NOx mass emission rate |
| `nox_kg_per_hr` | number | kg/hr | NOx mass emission rate (metric) |
| `nox_tons_per_year` | number | tons/yr | Annual NOx emissions (8760 hrs) |
| `o2_correction_factor` | number | - | Diluent correction factor |
| `f_factor_used` | number | dscf/MMBtu | F-factor for fuel type |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/calculate_nox_emissions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "nox_ppm": 150.0,
    "o2_percent": 3.5,
    "fuel_type": "natural_gas",
    "fuel_rate_mmbtu_hr": 100.0
  }'
```

---

### calculate_sox_emissions

**Endpoint**: `POST /v1/calculate_sox_emissions`

**Description**: Calculate SOx emissions from fuel sulfur content using sulfur balance method.

**Physics Basis**: Complete sulfur conversion to SO2 (stoichiometry)

**Formula**: `E_SOx = S_fuel * (64/32) * m_fuel / HHV_fuel`

**Standards**: 40 CFR Part 75, Appendix D

**Deterministic**: Yes

#### Request Body

```json
{
  "sulfur_percent": 1.2,
  "fuel_type": "bituminous_coal",
  "fuel_rate_lb_hr": 50000.0,
  "hhv_btu_per_lb": 12500,
  "so2_removal_efficiency": 0.95
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `sulfur_percent` | number | Yes | 0 - 10 | Fuel sulfur content (weight %) |
| `fuel_type` | string | Yes | Enum | Type of fuel |
| `fuel_rate_lb_hr` | number | Yes | > 0 | Fuel consumption rate (lb/hr) |
| `hhv_btu_per_lb` | number | No | > 0 | Higher heating value (Btu/lb) |
| `so2_removal_efficiency` | number | No | 0 - 1, default: 0 | FGD removal efficiency |

#### Response

```json
{
  "status": "success",
  "data": {
    "sulfur_percent": 1.2,
    "so2_generated_lb_hr": 1200.0,
    "so2_removed_lb_hr": 1140.0,
    "so2_emitted_lb_hr": 60.0,
    "so2_emitted_kg_hr": 27.22,
    "sox_lb_per_mmbtu": 0.096,
    "so2_tons_per_year": 262.8,
    "removal_efficiency": 0.95,
    "hhv_used_btu_per_lb": 12500,
    "calculation_method": "sulfur_balance",
    "standards": ["40_CFR_Part_75_App_D"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_002",
    "timestamp": "2025-11-26T10:31:00Z",
    "execution_time_ms": 28
  }
}
```

#### Response Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `so2_generated_lb_hr` | number | lb/hr | SO2 generated before control |
| `so2_removed_lb_hr` | number | lb/hr | SO2 removed by FGD |
| `so2_emitted_lb_hr` | number | lb/hr | SO2 emitted to atmosphere |
| `sox_lb_per_mmbtu` | number | lb/MMBtu | SOx emission rate |
| `so2_tons_per_year` | number | tons/yr | Annual SO2 emissions |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/calculate_sox_emissions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "sulfur_percent": 1.2,
    "fuel_type": "bituminous_coal",
    "fuel_rate_lb_hr": 50000.0,
    "so2_removal_efficiency": 0.95
  }'
```

---

### calculate_co2_emissions

**Endpoint**: `POST /v1/calculate_co2_emissions`

**Description**: Calculate CO2 emissions using either combustion stoichiometry or EPA emission factors.

**Physics Basis**: Complete combustion carbon balance or EPA Tier 2 factors

**Formula**: `E_CO2 = m_fuel * C_fuel * (44.01/12.01)` or `E_CO2 = Q * EF`

**Standards**: 40 CFR Part 98, Subpart C

**Deterministic**: Yes

#### Request Body (Stoichiometry Method)

```json
{
  "calculation_method": "stoichiometry",
  "fuel_mass_kg_hr": 5000.0,
  "carbon_content": 0.75,
  "fuel_type": "bituminous_coal"
}
```

#### Request Body (Emission Factor Method)

```json
{
  "calculation_method": "emission_factor",
  "heat_input_mmbtu_hr": 500.0,
  "fuel_type": "natural_gas",
  "custom_ef_kg_per_mmbtu": null
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `calculation_method` | string | Yes | Enum: stoichiometry, emission_factor | Calculation approach |
| `fuel_mass_kg_hr` | number | Conditional | > 0 | Fuel mass flow (stoichiometry) |
| `carbon_content` | number | Conditional | 0 - 1 | Carbon mass fraction (stoichiometry) |
| `heat_input_mmbtu_hr` | number | Conditional | > 0 | Heat input rate (emission_factor) |
| `fuel_type` | string | Yes | Enum | Fuel type |
| `custom_ef_kg_per_mmbtu` | number | No | > 0 | Custom emission factor override |

#### Response

```json
{
  "status": "success",
  "data": {
    "co2_kg_hr": 26530.0,
    "co2_lb_hr": 58498.0,
    "co2_tonnes_per_year": 232402.8,
    "co2_short_tons_per_year": 256262.3,
    "emission_factor_kg_per_mmbtu": 53.06,
    "heat_input_mmbtu_hr": 500.0,
    "fuel_type": "natural_gas",
    "calculation_method": "epa_tier2_emission_factor",
    "standards": ["40_CFR_Part_98_Table_C-1"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_003",
    "timestamp": "2025-11-26T10:32:00Z",
    "execution_time_ms": 22
  }
}
```

#### Response Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `co2_kg_hr` | number | kg/hr | CO2 mass emission rate |
| `co2_lb_hr` | number | lb/hr | CO2 mass emission rate (imperial) |
| `co2_tonnes_per_year` | number | tonnes/yr | Annual CO2 emissions (metric) |
| `co2_short_tons_per_year` | number | short tons/yr | Annual CO2 emissions (US) |
| `emission_factor_kg_per_mmbtu` | number | kg/MMBtu | Emission factor used |

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/calculate_co2_emissions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "calculation_method": "emission_factor",
    "heat_input_mmbtu_hr": 500.0,
    "fuel_type": "natural_gas"
  }'
```

---

### calculate_pm_emissions

**Endpoint**: `POST /v1/calculate_pm_emissions`

**Description**: Calculate particulate matter emissions using EPA AP-42 emission factors.

**Physics Basis**: Empirical emission factors from source testing

**Standards**: EPA AP-42, 5th Edition, Volume I

**Deterministic**: Yes

#### Request Body

```json
{
  "source_type": "coal_pulverized",
  "heat_input_mmbtu_hr": 500.0,
  "ash_content_percent": 10.0,
  "control_efficiency": 0.99
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `source_type` | string | Yes | Enum | Emission source type |
| `heat_input_mmbtu_hr` | number | Yes | > 0 | Heat input rate (MMBtu/hr) |
| `ash_content_percent` | number | Conditional | 0 - 30 | Ash content for coal (%) |
| `control_efficiency` | number | No | 0 - 1, default: 0 | Control device efficiency |

**source_type Enum**:
- `natural_gas_boiler`, `no2_oil_boiler`, `no6_oil_boiler`
- `coal_pulverized`, `coal_stoker`, `coal_cfb`
- `wood_boiler`, `biomass_boiler`

#### Response

```json
{
  "status": "success",
  "data": {
    "PM_uncontrolled_lb_hr": 350.0,
    "PM_emitted_lb_hr": 3.5,
    "PM_kg_hr": 1.59,
    "PM_tons_per_year": 15.33,
    "PM10_uncontrolled_lb_hr": 250.0,
    "PM10_emitted_lb_hr": 2.5,
    "PM10_kg_hr": 1.13,
    "PM10_tons_per_year": 10.95,
    "PM2.5_uncontrolled_lb_hr": 115.0,
    "PM2.5_emitted_lb_hr": 1.15,
    "PM2.5_kg_hr": 0.52,
    "PM2.5_tons_per_year": 5.04,
    "source_type": "coal_pulverized",
    "heat_input_mmbtu_hr": 500.0,
    "ash_content_percent": 10.0,
    "control_efficiency": 0.99,
    "calculation_method": "ap42_emission_factors",
    "standards": ["EPA_AP-42_Vol_I"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_004",
    "timestamp": "2025-11-26T10:33:00Z",
    "execution_time_ms": 18
  }
}
```

---

### calculate_co_emissions

**Endpoint**: `POST /v1/calculate_co_emissions`

**Description**: Calculate carbon monoxide emissions from incomplete combustion.

**Physics Basis**: Incomplete combustion stoichiometry and AP-42 factors

**Standards**: EPA AP-42, combustion efficiency relationships

**Deterministic**: Yes

#### Request Body

```json
{
  "fuel_type": "natural_gas",
  "heat_input_mmbtu_hr": 100.0,
  "combustion_efficiency_percent": 99.5,
  "co_measured_ppm": null
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `fuel_type` | string | Yes | Enum | Type of fuel |
| `heat_input_mmbtu_hr` | number | Yes | > 0 | Heat input rate (MMBtu/hr) |
| `combustion_efficiency_percent` | number | No | 90 - 100, default: 99 | Combustion efficiency (%) |
| `co_measured_ppm` | number | No | >= 0 | Measured CO concentration (overrides calculation) |

#### Response

```json
{
  "status": "success",
  "data": {
    "co_lb_per_hr": 8.4,
    "co_kg_per_hr": 3.81,
    "co_lb_per_mmbtu": 0.084,
    "co_tons_per_year": 36.79,
    "co_ppm_equivalent": 75.0,
    "combustion_efficiency_percent": 99.5,
    "fuel_type": "natural_gas",
    "calculation_method": "combustion_efficiency",
    "standards": ["EPA_AP-42", "combustion_principles"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_005",
    "timestamp": "2025-11-26T10:34:00Z",
    "execution_time_ms": 15
  }
}
```

---

### check_compliance

**Endpoint**: `POST /v1/check_compliance`

**Description**: Check emissions against multi-jurisdiction regulatory limits.

**Jurisdictions**: US EPA, EU IED, China MEE

**Deterministic**: Yes

#### Request Body

```json
{
  "emissions": {
    "nox_lb_per_mmbtu": 0.35,
    "so2_lb_per_mmbtu": 0.12,
    "pm_lb_per_mmbtu": 0.012,
    "co_ppm": 350
  },
  "jurisdiction": "epa",
  "source_category": "utility_boiler_subpart_Da",
  "averaging_period": "hourly"
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `emissions` | object | Yes | - | Emission values by pollutant |
| `jurisdiction` | string | Yes | Enum: epa, eu, china | Regulatory jurisdiction |
| `source_category` | string | Yes | - | Source category for limit lookup |
| `averaging_period` | string | No | Enum | Averaging period, default: hourly |

**jurisdiction-specific source_category values**:

*EPA*:
- `utility_boiler_subpart_Da`, `utility_boiler_subpart_Db`
- `industrial_boiler_subpart_Dc`

*EU*:
- `large_combustion_plant_coal`, `large_combustion_plant_gas`
- `large_combustion_plant_oil`

*China*:
- `coal_fired_key_region`, `coal_fired_general`, `gas_fired`

#### Response

```json
{
  "status": "success",
  "data": {
    "overall_compliant": true,
    "violation_count": 0,
    "violations": [],
    "compliance_status": {
      "nox_lb_per_mmbtu": {
        "measured": 0.35,
        "limit": 0.40,
        "limit_key": "NOx_lb_per_mmbtu",
        "compliant": true,
        "exceedance_percent": 0,
        "margin_percent": 12.5
      },
      "so2_lb_per_mmbtu": {
        "measured": 0.12,
        "limit": 0.15,
        "limit_key": "SO2_lb_per_mmbtu",
        "compliant": true,
        "exceedance_percent": 0,
        "margin_percent": 20.0
      },
      "pm_lb_per_mmbtu": {
        "measured": 0.012,
        "limit": 0.015,
        "limit_key": "PM_lb_per_mmbtu",
        "compliant": true,
        "exceedance_percent": 0,
        "margin_percent": 20.0
      },
      "co_ppm": {
        "measured": 350,
        "limit": 400,
        "limit_key": "CO_ppm",
        "compliant": true,
        "exceedance_percent": 0,
        "margin_percent": 12.5
      }
    },
    "jurisdiction": "epa",
    "source_category": "utility_boiler_subpart_Da",
    "averaging_period": "hourly",
    "timestamp": "2025-11-26T10:35:00Z"
  },
  "metadata": {
    "request_id": "req_emissionwatch_006",
    "timestamp": "2025-11-26T10:35:00Z",
    "execution_time_ms": 12
  }
}
```

#### Response with Violations

```json
{
  "status": "success",
  "data": {
    "overall_compliant": false,
    "violation_count": 1,
    "violations": [
      {
        "pollutant": "nox_lb_per_mmbtu",
        "measured": 0.52,
        "limit": 0.40,
        "exceedance_percent": 30.0,
        "severity": "moderate"
      }
    ],
    "compliance_status": {
      "nox_lb_per_mmbtu": {
        "measured": 0.52,
        "limit": 0.40,
        "compliant": false,
        "exceedance_percent": 30.0
      }
    },
    "jurisdiction": "epa",
    "source_category": "utility_boiler_subpart_Da",
    "timestamp": "2025-11-26T10:35:00Z"
  }
}
```

#### Example Request

```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/check_compliance \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "emissions": {
      "nox_lb_per_mmbtu": 0.35,
      "so2_lb_per_mmbtu": 0.12
    },
    "jurisdiction": "epa",
    "source_category": "utility_boiler_subpart_Da"
  }'
```

---

### detect_violations

**Endpoint**: `POST /v1/detect_violations`

**Description**: Real-time violation detection with rolling average calculations.

**Features**:
- Instantaneous threshold detection
- Rolling averages (1-hr, 24-hr, 30-day)
- Exceedance counting
- Trend analysis

**Deterministic**: Yes

#### Request Body

```json
{
  "pollutant": "NOx",
  "current_value": 0.45,
  "timestamp": "2025-11-26T10:36:00Z",
  "averaging_periods": ["instantaneous", "1_hour", "24_hour"],
  "limits": {
    "NOx_instantaneous": 0.50,
    "NOx_1hr": 0.40,
    "NOx_24hr": 0.35
  }
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `pollutant` | string | Yes | - | Pollutant identifier |
| `current_value` | number | Yes | >= 0 | Current measured value |
| `timestamp` | string | Yes | ISO8601 | Measurement timestamp |
| `averaging_periods` | array | No | - | Periods to check |
| `limits` | object | Yes | - | Limit values by period |

#### Response

```json
{
  "status": "success",
  "data": {
    "pollutant": "NOx",
    "current_value": 0.45,
    "timestamp": "2025-11-26T10:36:00Z",
    "violation_detected": true,
    "warning_detected": true,
    "violations": [
      {
        "period": "1_hour",
        "average_value": 0.42,
        "limit": 0.40,
        "exceedance_percent": 5.0,
        "exceedance_count": 3
      }
    ],
    "warnings": [
      {
        "period": "instantaneous",
        "average_value": 0.45,
        "limit": 0.50,
        "percent_of_limit": 90.0
      }
    ],
    "rolling_averages": {
      "1_hour": 0.42,
      "24_hour": 0.38
    }
  },
  "metadata": {
    "request_id": "req_emissionwatch_007",
    "timestamp": "2025-11-26T10:36:00Z",
    "execution_time_ms": 8
  }
}
```

---

### calculate_dispersion

**Endpoint**: `POST /v1/calculate_dispersion`

**Description**: Gaussian plume dispersion modeling for ambient concentration estimation.

**Physics Basis**: Gaussian distribution, Briggs plume rise

**Standards**: EPA AERMOD principles (simplified)

**Deterministic**: Yes

#### Request Body

```json
{
  "emission_rate_g_s": 100.0,
  "stack_height_m": 50.0,
  "stack_diameter_m": 3.0,
  "stack_exit_velocity_m_s": 15.0,
  "stack_exit_temp_k": 450.0,
  "ambient_temp_k": 298.15,
  "wind_speed_m_s": 5.0,
  "stability_class": "D",
  "receptors": [
    {"distance_m": 500, "crosswind_m": 0, "height_m": 1.5},
    {"distance_m": 1000, "crosswind_m": 0, "height_m": 1.5},
    {"distance_m": 2000, "crosswind_m": 0, "height_m": 1.5}
  ]
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `emission_rate_g_s` | number | Yes | > 0 | Pollutant emission rate (g/s) |
| `stack_height_m` | number | Yes | > 0 | Physical stack height (m) |
| `stack_diameter_m` | number | Yes | > 0 | Stack diameter (m) |
| `stack_exit_velocity_m_s` | number | Yes | > 0 | Exit velocity (m/s) |
| `stack_exit_temp_k` | number | Yes | > 273 | Exit temperature (K) |
| `ambient_temp_k` | number | No | 250 - 320 | Ambient temperature (K) |
| `wind_speed_m_s` | number | Yes | 0.5 - 30 | Wind speed at stack height (m/s) |
| `stability_class` | string | No | Enum: A-F | Pasquill-Gifford stability class |
| `receptors` | array | Yes | - | Receptor locations |

#### Response

```json
{
  "status": "success",
  "data": {
    "plume_rise_m": 25.4,
    "effective_stack_height_m": 75.4,
    "receptor_concentrations": [
      {
        "distance_m": 500,
        "crosswind_m": 0,
        "height_m": 1.5,
        "concentration_ug_m3": 45.2,
        "sigma_y_m": 35.2,
        "sigma_z_m": 18.5
      },
      {
        "distance_m": 1000,
        "crosswind_m": 0,
        "height_m": 1.5,
        "concentration_ug_m3": 28.7,
        "sigma_y_m": 62.1,
        "sigma_z_m": 32.4
      },
      {
        "distance_m": 2000,
        "crosswind_m": 0,
        "height_m": 1.5,
        "concentration_ug_m3": 12.3,
        "sigma_y_m": 110.5,
        "sigma_z_m": 55.8
      }
    ],
    "max_ground_level_concentration_ug_m3": 52.1,
    "max_concentration_distance_m": 350,
    "stability_class": "D",
    "calculation_method": "gaussian_plume",
    "standards": ["EPA_AERMOD_simplified"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_008",
    "timestamp": "2025-11-26T10:37:00Z",
    "execution_time_ms": 85
  }
}
```

---

### generate_report

**Endpoint**: `POST /v1/generate_report`

**Description**: Generate regulatory compliance reports in various formats.

**Output Formats**: EPA CEDRI, EU ETS, XBRL, PDF, JSON

**Deterministic**: Yes

#### Request Body

```json
{
  "report_type": "cedri",
  "facility_id": "ORIS-12345",
  "reporting_period": {
    "start_date": "2025-01-01",
    "end_date": "2025-03-31"
  },
  "include_sections": ["emissions", "compliance", "violations", "trends"],
  "output_format": "json"
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `report_type` | string | Yes | Enum: cedri, ets, xbrl, summary | Report type |
| `facility_id` | string | Yes | - | Facility identifier (ORIS, EU registry) |
| `reporting_period` | object | Yes | - | Start and end dates |
| `include_sections` | array | No | - | Sections to include |
| `output_format` | string | No | Enum: json, xml, pdf | Output format |

#### Response

```json
{
  "status": "success",
  "data": {
    "report_type": "cedri",
    "facility_id": "ORIS-12345",
    "reporting_period": {
      "start_date": "2025-01-01",
      "end_date": "2025-03-31",
      "quarter": "Q1 2025"
    },
    "emissions_summary": {
      "nox_tons": 245.8,
      "so2_tons": 89.2,
      "co2_short_tons": 125420.5,
      "pm_tons": 12.4
    },
    "compliance_summary": {
      "total_operating_hours": 2160,
      "compliant_hours": 2148,
      "compliance_percentage": 99.44,
      "violations_count": 3
    },
    "violations": [
      {
        "date": "2025-02-15",
        "pollutant": "NOx",
        "duration_hours": 4,
        "max_exceedance_percent": 15.2,
        "corrective_action": "Burner tuning"
      }
    ],
    "data_availability": {
      "nox_percent": 99.8,
      "so2_percent": 99.9,
      "co2_percent": 100.0,
      "flow_percent": 99.7
    },
    "certification_statement": {
      "certifier_name": "To be completed",
      "title": "Environmental Manager",
      "certification_date": null
    },
    "report_generated_at": "2025-11-26T10:38:00Z",
    "provenance_hash": "sha256:abc123..."
  },
  "metadata": {
    "request_id": "req_emissionwatch_009",
    "timestamp": "2025-11-26T10:38:00Z",
    "execution_time_ms": 2850
  }
}
```

---

### configure_alerts

**Endpoint**: `POST /v1/configure_alerts`

**Description**: Configure alert thresholds and notification channels.

**Channels**: Email, SMS, Webhook, Slack, Microsoft Teams

#### Request Body

```json
{
  "facility_id": "ORIS-12345",
  "alert_rules": [
    {
      "rule_id": "nox_warning",
      "pollutant": "NOx",
      "threshold_type": "percent_of_limit",
      "threshold_value": 80,
      "averaging_period": "1_hour",
      "severity": "warning",
      "enabled": true
    },
    {
      "rule_id": "nox_critical",
      "pollutant": "NOx",
      "threshold_type": "exceedance",
      "threshold_value": 0,
      "averaging_period": "1_hour",
      "severity": "critical",
      "enabled": true
    }
  ],
  "notification_channels": [
    {
      "channel_type": "email",
      "recipients": ["env-team@company.com"],
      "severity_filter": ["warning", "critical"]
    },
    {
      "channel_type": "webhook",
      "url": "https://company.com/api/emissions-alerts",
      "severity_filter": ["critical"]
    }
  ],
  "escalation_policy": {
    "initial_delay_minutes": 0,
    "escalation_delay_minutes": 30,
    "max_escalations": 3
  }
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "facility_id": "ORIS-12345",
    "alert_rules_configured": 2,
    "notification_channels_configured": 2,
    "configuration_hash": "sha256:def456...",
    "effective_timestamp": "2025-11-26T10:39:00Z"
  },
  "metadata": {
    "request_id": "req_emissionwatch_010",
    "timestamp": "2025-11-26T10:39:00Z",
    "execution_time_ms": 120
  }
}
```

---

### get_emission_factors

**Endpoint**: `GET /v1/emission_factors`

**Description**: Retrieve EPA AP-42 emission factors for various source categories.

**Database**: EPA AP-42, 5th Edition, Volume I (with amendments)

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_type` | string | No | Filter by source type |
| `fuel_type` | string | No | Filter by fuel type |
| `pollutant` | string | No | Filter by pollutant |

#### Response

```json
{
  "status": "success",
  "data": {
    "emission_factors": [
      {
        "source_type": "natural_gas_boiler",
        "fuel_type": "natural_gas",
        "pollutant": "NOx",
        "factor_value": 0.098,
        "factor_unit": "lb/MMBtu",
        "ap42_section": "1.4",
        "rating": "A",
        "notes": "Uncontrolled, small boilers (<100 MMBtu/hr)"
      },
      {
        "source_type": "natural_gas_boiler",
        "fuel_type": "natural_gas",
        "pollutant": "CO",
        "factor_value": 0.084,
        "factor_unit": "lb/MMBtu",
        "ap42_section": "1.4",
        "rating": "A"
      }
    ],
    "database_version": "AP-42_5th_Edition_2023",
    "last_updated": "2023-09-15"
  },
  "metadata": {
    "request_id": "req_emissionwatch_011",
    "timestamp": "2025-11-26T10:40:00Z",
    "execution_time_ms": 25
  }
}
```

---

### analyze_fuel

**Endpoint**: `POST /v1/analyze_fuel`

**Description**: Analyze fuel composition and calculate emission-relevant properties.

**Standards**: ASTM D3176 (coal), ASTM D4057 (petroleum)

#### Request Body

```json
{
  "fuel_type": "bituminous_coal",
  "ultimate_analysis": {
    "carbon_percent": 72.5,
    "hydrogen_percent": 5.1,
    "oxygen_percent": 8.2,
    "nitrogen_percent": 1.4,
    "sulfur_percent": 2.3,
    "ash_percent": 10.5
  },
  "hhv_btu_per_lb": 12800,
  "moisture_percent": 8.0
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "fuel_type": "bituminous_coal",
    "ultimate_analysis": {
      "carbon_percent": 72.5,
      "hydrogen_percent": 5.1,
      "oxygen_percent": 8.2,
      "nitrogen_percent": 1.4,
      "sulfur_percent": 2.3,
      "ash_percent": 10.5
    },
    "calculated_properties": {
      "hhv_btu_per_lb": 12800,
      "lhv_btu_per_lb": 12280,
      "stoichiometric_air_lb_per_lb_fuel": 9.82,
      "f_factor_dscf_per_mmbtu": 9780,
      "co2_emission_factor_lb_per_mmbtu": 205.2,
      "so2_potential_lb_per_mmbtu": 4.06
    },
    "quality_classification": {
      "rank": "High-Volatile Bituminous A",
      "sulfur_class": "Medium Sulfur (1-3%)",
      "ash_class": "Medium Ash (8-12%)"
    },
    "analysis_method": "astm_d3176_ultimate",
    "standards": ["ASTM_D3176", "ASTM_D5865"]
  },
  "metadata": {
    "request_id": "req_emissionwatch_012",
    "timestamp": "2025-11-26T10:41:00Z",
    "execution_time_ms": 35
  }
}
```

---

## Operation Modes

### Monitor Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "monitor",
  "cems_data": {
    "source_id": "STACK-001",
    "timestamp": "2025-11-26T10:42:00Z",
    "nox_ppm": 145.0,
    "so2_ppm": 85.0,
    "co2_percent": 8.5,
    "o2_percent": 3.2,
    "flow_scfm": 125000,
    "stack_temp_f": 350
  },
  "fuel_data": {
    "fuel_type": "natural_gas",
    "fuel_rate_mmbtu_hr": 150.0
  }
}
```

**Description**: Real-time CEMS data monitoring with automatic emission calculations and compliance checking.

---

### Calculate Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "calculate",
  "calculation_type": "all_pollutants",
  "fuel_data": {
    "fuel_type": "bituminous_coal",
    "fuel_rate_lb_hr": 50000,
    "sulfur_percent": 1.2,
    "ash_percent": 10.0
  },
  "operating_data": {
    "heat_input_mmbtu_hr": 500.0,
    "operating_hours": 720
  }
}
```

**Description**: Calculate emissions from fuel/process data without real-time CEMS.

---

### Validate Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "validate",
  "emissions": {
    "nox_lb_per_mmbtu": 0.35,
    "so2_lb_per_mmbtu": 0.12,
    "pm_lb_per_mmbtu": 0.012
  },
  "jurisdictions": ["epa", "eu"],
  "source_categories": {
    "epa": "utility_boiler_subpart_Da",
    "eu": "large_combustion_plant_coal"
  }
}
```

**Description**: Multi-jurisdiction compliance validation.

---

### Report Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "report",
  "report_config": {
    "report_type": "cedri",
    "facility_id": "ORIS-12345",
    "reporting_period": {
      "start_date": "2025-01-01",
      "end_date": "2025-03-31"
    }
  }
}
```

**Description**: Generate regulatory compliance reports.

---

### Alert Mode

**Endpoint**: `POST /v1/run`

**Request Body**:
```json
{
  "operation_mode": "alert",
  "alert_action": "check_and_notify",
  "facility_id": "ORIS-12345"
}
```

**Description**: Check current conditions against alert rules and send notifications.

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
| 422 | Unprocessable Entity | Validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Agent error |
| 503 | Service Unavailable | Agent offline |

### Application Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `VALIDATION_ERROR` | Input validation failed | Check field constraints |
| `INVALID_FUEL_TYPE` | Unknown fuel type | Use valid fuel type enum |
| `INVALID_JURISDICTION` | Unknown jurisdiction | Use epa, eu, or china |
| `INVALID_SOURCE_CATEGORY` | Unknown source category | Check jurisdiction-specific categories |
| `EMISSION_FACTOR_NOT_FOUND` | No factor for source/fuel | Use custom emission factor |
| `CEMS_DATA_MISSING` | Required CEMS fields missing | Provide all required fields |
| `CEMS_DATA_QUALITY_ERROR` | CEMS data failed QA checks | Review data quality |
| `COMPLIANCE_LIMIT_NOT_FOUND` | No limit for pollutant/category | Check source category |
| `CALCULATION_ERROR` | Physics calculation failed | Check input ranges |
| `REPORT_GENERATION_ERROR` | Report generation failed | Contact support |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |

### Example Error Response

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "NOx concentration exceeds valid range",
    "details": {
      "field": "nox_ppm",
      "value": 6000,
      "constraint": "Must be 0-5000 ppm",
      "suggestion": "Verify CEMS calibration if reading is correct"
    }
  },
  "metadata": {
    "request_id": "req_emissionwatch_err001",
    "timestamp": "2025-11-26T10:43:00Z"
  }
}
```

---

## Rate Limiting

### Standard Limits

| Tier | Requests/Minute | Requests/Hour | Requests/Day |
|------|-----------------|---------------|--------------|
| Free | 10 | 100 | 1,000 |
| Pro | 100 | 2,000 | 50,000 |
| Enterprise | Custom | Custom | Custom |

### Rate Limit Headers

**Response Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
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
    "retry_after_seconds": 60
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
  "url": "https://your-app.com/webhooks/emissionwatch",
  "events": ["violation.detected", "compliance.report_ready", "alert.triggered"],
  "secret": "whsec_your_secret"
}
```

### Webhook Events

#### violation.detected

**Payload**:
```json
{
  "event": "violation.detected",
  "timestamp": "2025-11-26T10:44:00Z",
  "data": {
    "facility_id": "ORIS-12345",
    "source_id": "STACK-001",
    "pollutant": "NOx",
    "measured_value": 0.52,
    "limit_value": 0.40,
    "exceedance_percent": 30.0,
    "severity": "moderate",
    "averaging_period": "1_hour"
  }
}
```

#### compliance.report_ready

**Payload**:
```json
{
  "event": "compliance.report_ready",
  "timestamp": "2025-11-26T10:45:00Z",
  "data": {
    "facility_id": "ORIS-12345",
    "report_type": "cedri",
    "reporting_period": "Q1 2025",
    "report_id": "rpt_abc123",
    "download_url": "https://api.greenlang.org/agents/gl-010/v1/reports/rpt_abc123/download",
    "expires_at": "2025-11-27T10:45:00Z"
  }
}
```

#### alert.triggered

**Payload**:
```json
{
  "event": "alert.triggered",
  "timestamp": "2025-11-26T10:46:00Z",
  "data": {
    "facility_id": "ORIS-12345",
    "alert_rule_id": "nox_warning",
    "severity": "warning",
    "message": "NOx at 85% of limit (0.34 lb/MMBtu vs 0.40 limit)",
    "current_value": 0.34,
    "threshold_value": 0.32,
    "threshold_type": "percent_of_limit"
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

## Real-Time Streaming

### WebSocket Connection

**Endpoint**: `wss://api.greenlang.org/agents/gl-010/v1/stream`

**Authentication**: Include API key as query parameter or header

**Connection**:
```javascript
const ws = new WebSocket('wss://api.greenlang.org/agents/gl-010/v1/stream?api_key=gl_sk_...');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Emission update:', data);
};
```

### Subscribe to Facility

**Subscribe Message**:
```json
{
  "action": "subscribe",
  "facility_id": "ORIS-12345",
  "sources": ["STACK-001", "STACK-002"],
  "pollutants": ["NOx", "SO2", "CO2"]
}
```

### Stream Data Format

**Emission Update**:
```json
{
  "type": "emission_update",
  "facility_id": "ORIS-12345",
  "source_id": "STACK-001",
  "timestamp": "2025-11-26T10:47:00Z",
  "data": {
    "nox_ppm": 142.5,
    "nox_lb_per_mmbtu": 0.178,
    "so2_ppm": 82.3,
    "so2_lb_per_mmbtu": 0.102,
    "co2_percent": 8.4,
    "compliance_status": "compliant"
  }
}
```

**Violation Alert**:
```json
{
  "type": "violation_alert",
  "facility_id": "ORIS-12345",
  "source_id": "STACK-001",
  "timestamp": "2025-11-26T10:48:00Z",
  "data": {
    "pollutant": "NOx",
    "severity": "warning",
    "message": "NOx approaching limit (90%)"
  }
}
```

---

## Code Examples

### Python SDK

```python
from greenlang import EmissionsComplianceAgent

# Initialize agent
agent = EmissionsComplianceAgent(api_key="gl_sk_emissionwatch_abc123...")

# Calculate NOx emissions
nox_result = agent.calculate_nox_emissions(
    nox_ppm=150.0,
    o2_percent=3.5,
    fuel_type="natural_gas",
    fuel_rate_mmbtu_hr=100.0
)

print(f"NOx emission rate: {nox_result.nox_lb_per_mmbtu} lb/MMBtu")
print(f"NOx mass rate: {nox_result.nox_lb_per_hr} lb/hr")

# Check compliance
compliance_result = agent.check_compliance(
    emissions={
        "nox_lb_per_mmbtu": nox_result.nox_lb_per_mmbtu,
        "so2_lb_per_mmbtu": 0.12
    },
    jurisdiction="epa",
    source_category="utility_boiler_subpart_Da"
)

print(f"Compliant: {compliance_result.overall_compliant}")
if not compliance_result.overall_compliant:
    for violation in compliance_result.violations:
        print(f"Violation: {violation.pollutant} - {violation.exceedance_percent}% over limit")

# Real-time monitoring
async for update in agent.stream_emissions(facility_id="ORIS-12345"):
    print(f"[{update.timestamp}] NOx: {update.nox_ppm} ppm")
    if update.violation_detected:
        print(f"  VIOLATION: {update.violation_message}")
```

### JavaScript/TypeScript SDK

```typescript
import { EmissionsComplianceAgent } from '@greenlang/agents';

const agent = new EmissionsComplianceAgent({ apiKey: 'gl_sk_emissionwatch_abc123...' });

// Calculate CO2 emissions
const co2Result = await agent.calculateCO2Emissions({
  calculationMethod: 'emission_factor',
  heatInputMmbtuHr: 500.0,
  fuelType: 'natural_gas',
});

console.log(`CO2 emissions: ${co2Result.co2TonnesPerYear} tonnes/year`);

// Check multi-jurisdiction compliance
const complianceResult = await agent.checkCompliance({
  emissions: {
    noxLbPerMmbtu: 0.35,
    so2LbPerMmbtu: 0.12,
  },
  jurisdiction: 'epa',
  sourceCategory: 'utility_boiler_subpart_Da',
});

console.log(`EPA Compliant: ${complianceResult.overallCompliant}`);

// Generate CEDRI report
const report = await agent.generateReport({
  reportType: 'cedri',
  facilityId: 'ORIS-12345',
  reportingPeriod: {
    startDate: '2025-01-01',
    endDate: '2025-03-31',
  },
});

console.log(`Report generated: ${report.reportId}`);
```

### cURL Examples

**Calculate NOx Emissions**:
```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/calculate_nox_emissions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "nox_ppm": 150.0,
    "o2_percent": 3.5,
    "fuel_type": "natural_gas",
    "fuel_rate_mmbtu_hr": 100.0
  }'
```

**Check Compliance**:
```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/check_compliance \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "emissions": {
      "nox_lb_per_mmbtu": 0.35,
      "so2_lb_per_mmbtu": 0.12,
      "pm_lb_per_mmbtu": 0.012
    },
    "jurisdiction": "epa",
    "source_category": "utility_boiler_subpart_Da"
  }'
```

**Generate Report**:
```bash
curl -X POST https://api.greenlang.org/agents/gl-010/v1/generate_report \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gl_sk_emissionwatch_abc123..." \
  -d '{
    "report_type": "cedri",
    "facility_id": "ORIS-12345",
    "reporting_period": {
      "start_date": "2025-01-01",
      "end_date": "2025-03-31"
    }
  }'
```

---

## SDK Reference

### Python SDK

**Installation**:
```bash
pip install greenlang-agents
```

**Documentation**: https://docs.greenlang.org/sdks/python/gl-010

### JavaScript/TypeScript SDK

**Installation**:
```bash
npm install @greenlang/agents
```

**Documentation**: https://docs.greenlang.org/sdks/javascript/gl-010

### Go SDK

**Installation**:
```bash
go get github.com/greenlang/agents-go/gl010
```

**Documentation**: https://docs.greenlang.org/sdks/go/gl-010

### Java SDK

**Installation** (Maven):
```xml
<dependency>
  <groupId>org.greenlang</groupId>
  <artifactId>agents-gl010</artifactId>
  <version>1.0.0</version>
</dependency>
```

**Documentation**: https://docs.greenlang.org/sdks/java/gl-010

---

## Changelog

### v1.0.0 (2025-11-26)

**Initial Release**

- 12 deterministic calculation tools
- NOx calculation (EPA Method 19 F-factor)
- SOx calculation (sulfur balance)
- CO2 calculation (stoichiometry + EPA Tier 2 factors)
- PM calculation (AP-42 emission factors)
- CO calculation (combustion efficiency)
- Multi-jurisdiction compliance checking (EPA, EU IED, China MEE)
- Real-time violation detection with rolling averages
- Gaussian plume dispersion modeling
- Regulatory report generation (EPA CEDRI, EU ETS, XBRL)
- Alert configuration and multi-channel notifications
- EPA AP-42 emission factors database
- Fuel composition analysis
- Full provenance tracking (SHA-256)
- WebSocket streaming for real-time data
- Industry standards compliance (40 CFR Part 60/75/98, EU IED)

---

**API Version**: v1
**Documentation Version**: 1.0.0
**Last Updated**: 2025-11-26
**Support**: api-support@greenlang.org
**License**: Apache-2.0

---

For questions, bug reports, or feature requests, please contact api-support@greenlang.org or open an issue at https://github.com/greenlang/agents/issues
