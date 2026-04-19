# GL-018 FLUEFLOW - FastAPI REST API Documentation

Complete REST API implementation for GL-018 FLUEFLOW - Flue Gas Combustion Optimization.

## Overview

This FastAPI implementation provides production-grade REST endpoints for:
- Flue gas composition analysis
- Combustion efficiency calculation
- Air-fuel ratio optimization
- Emissions compliance monitoring
- Performance reporting and trends

## Quick Start

### Installation

```bash
cd GL-018
pip install -r requirements.txt
```

### Run the API

```bash
# Development mode (with auto-reload)
python tools.py

# Production mode
uvicorn tools:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI Spec**: http://localhost:8000/api/openapi.json

## Authentication

### Get Access Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo@flueflow.io&password=demo_password_123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Default Credentials:**
- Username: `demo@flueflow.io`
- Password: `demo_password_123`

### Use Token

```bash
export TOKEN="your_access_token_here"

curl -X POST "http://localhost:8000/api/v1/analyze-flue-gas" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

## API Endpoints

### 1. POST /api/v1/analyze-flue-gas

Analyze flue gas composition and assess combustion quality.

**Rate Limit:** 100 requests/minute

**Request:**
```json
{
  "burner_id": "burner_001",
  "composition": {
    "o2_percent": 3.5,
    "co2_percent": 12.5,
    "co_ppm": 50,
    "nox_ppm": 150,
    "sox_ppm": 20,
    "temperature_celsius": 180
  },
  "timestamp": "2025-12-02T10:30:00Z"
}
```

**Response:**
```json
{
  "burner_id": "burner_001",
  "combustion_quality": "good",
  "excess_air_percent": 20.0,
  "air_fuel_ratio": 20.64,
  "stoichiometric_ratio": 17.2,
  "combustion_completeness": 99.5,
  "unburned_losses_percent": 0.625,
  "analysis_timestamp": "2025-12-02T10:30:00Z",
  "recommendations": [
    "Excess air is within optimal range",
    "CO levels are acceptable"
  ],
  "provenance_hash": "a1b2c3..."
}
```

### 2. POST /api/v1/calculate-efficiency

Calculate combustion and thermal efficiency with loss breakdown.

**Rate Limit:** 100 requests/minute

**Request:**
```json
{
  "burner_id": "burner_001",
  "flue_gas": {
    "o2_percent": 3.5,
    "co2_percent": 12.5,
    "co_ppm": 50,
    "nox_ppm": 150,
    "temperature_celsius": 180
  },
  "operating_data": {
    "fuel_type": "natural_gas",
    "fuel_flow_rate": 1000,
    "air_flow_rate": 12000,
    "steam_output": 10000,
    "feedwater_temp": 105,
    "steam_pressure": 10,
    "ambient_temp": 25
  }
}
```

**Response:**
```json
{
  "burner_id": "burner_001",
  "combustion_efficiency": 87.5,
  "thermal_efficiency": 88.2,
  "stack_loss": 9.5,
  "radiation_loss": 1.5,
  "unaccounted_loss": 1.5,
  "efficiency_rating": "good",
  "improvement_potential": 4.5,
  "annual_savings_potential": 125000.0,
  "recommendations": [
    "Operating at good efficiency",
    "Consider economizer for further improvement"
  ],
  "timestamp": "2025-12-02T10:30:00Z",
  "provenance_hash": "x7y8z9..."
}
```

### 3. POST /api/v1/optimize-air-fuel-ratio

Get optimization recommendations for air-fuel ratio.

**Rate Limit:** 100 requests/minute

**Request:**
```json
{
  "burner_id": "burner_001",
  "current_flue_gas": {
    "o2_percent": 5.0,
    "co2_percent": 11.0,
    "co_ppm": 30,
    "nox_ppm": 180,
    "temperature_celsius": 185
  },
  "current_operating_data": {
    "fuel_type": "natural_gas",
    "fuel_flow_rate": 1000,
    "air_flow_rate": 13000,
    "steam_output": 10000,
    "feedwater_temp": 105,
    "steam_pressure": 10
  },
  "optimization_priority": "balanced"
}
```

**Response:**
```json
{
  "burner_id": "burner_001",
  "current_air_fuel_ratio": 15.6,
  "recommended_air_fuel_ratio": 17.72,
  "current_excess_air_percent": 31.25,
  "recommended_excess_air_percent": 15.0,
  "expected_efficiency_gain": 1.6,
  "expected_nox_reduction": 10.0,
  "expected_co_change": 30.0,
  "air_flow_adjustment": -13.2,
  "implementation_steps": [
    "1. Gradually reduce air damper opening by 13.2%",
    "2. Monitor O2 levels - target: 3.0%",
    "3. Monitor CO levels - ensure CO remains below 100 ppm",
    "..."
  ],
  "warnings": [
    "Large air flow reduction required. Make changes gradually."
  ],
  "timestamp": "2025-12-02T10:30:00Z",
  "provenance_hash": "m3n4o5..."
}
```

### 4. GET /api/v1/emissions-compliance/{burner_id}

Get emissions compliance report.

**Rate Limit:** 100 requests/minute

**Query Parameters:**
- `standard` (optional): Regulatory standard (EPA, EU_ETS, CUSTOM). Default: EPA

**Response:**
```json
{
  "burner_id": "burner_001",
  "compliance_status": "COMPLIANT",
  "regulatory_standard": "EPA",
  "nox_compliance": {
    "current_ppm": 180,
    "limit_ppm": 200,
    "margin_percent": 10.0,
    "status": "COMPLIANT"
  },
  "sox_compliance": {
    "current_ppm": 25,
    "limit_ppm": 50,
    "margin_percent": 50.0,
    "status": "COMPLIANT"
  },
  "co_compliance": {
    "current_ppm": 75,
    "limit_ppm": 100,
    "margin_percent": 25.0,
    "status": "COMPLIANT"
  },
  "overall_margin_percent": 10.0,
  "violations": [],
  "warnings": [],
  "corrective_actions": [
    "No corrective actions required"
  ],
  "next_report_due": "2025-01-01T20:30:00Z",
  "timestamp": "2025-12-02T20:30:00Z",
  "provenance_hash": "s9t0u1..."
}
```

### 5. GET /api/v1/performance-report/{burner_id}

Get performance metrics and trends.

**Rate Limit:** 100 requests/minute

**Query Parameters:**
- `period` (optional): Time period (1h, 24h, 7d, 30d). Default: 24h

**Response:**
```json
{
  "burner_id": "burner_001",
  "period": "24h",
  "avg_efficiency": 87.5,
  "min_efficiency": 85.5,
  "max_efficiency": 89.5,
  "efficiency_trend": "stable",
  "avg_nox_ppm": 175.0,
  "avg_co_ppm": 65.0,
  "avg_o2_percent": 3.5,
  "uptime_percent": 98.5,
  "total_fuel_consumed": 23640.0,
  "total_steam_generated": 236400.0,
  "performance_score": 92.5,
  "trends": {
    "efficiency_trend": {
      "direction": "stable",
      "change_percent": 0.2
    }
  },
  "alerts": [
    "All parameters within normal range"
  ],
  "timestamp": "2025-12-02T20:30:00Z"
}
```

### 6. GET /health

Health check endpoint.

**Rate Limit:** 1000 requests/minute

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-02T20:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.0,
  "checks": {
    "api": true,
    "authentication": true,
    "calculations": true
  }
}
```

### 7. GET /metrics

Prometheus metrics endpoint.

**Rate Limit:** 1000 requests/minute

**Response:** Prometheus text format metrics

```
# HELP flueflow_api_requests_total Total API requests
# TYPE flueflow_api_requests_total counter
flueflow_api_requests_total{endpoint="/api/v1/analyze-flue-gas",method="POST",status="200"} 1250.0

# HELP flueflow_efficiency_percent Combustion efficiency calculations
# TYPE flueflow_efficiency_percent histogram
...
```

## Error Handling

All errors return standardized JSON responses:

```json
{
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2025-12-02T20:30:00Z"
}
```

### Common Error Codes

- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Missing or invalid authentication token
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

## Rate Limits

| Endpoint Category | Rate Limit |
|------------------|------------|
| Analysis endpoints | 100 requests/minute |
| Status endpoints | 100 requests/minute |
| Health & Metrics | 1000 requests/minute |

Rate limits are per IP address.

## Data Models

### FlueGasComposition

```python
{
  "o2_percent": float,        # 0-21%, Oxygen concentration
  "co2_percent": float,       # 0-20%, CO2 concentration
  "co_ppm": float,            # 0-10000, CO concentration
  "nox_ppm": float,           # 0-1000, NOx concentration
  "sox_ppm": float,           # 0-5000, SOx concentration (optional)
  "temperature_celsius": float # 0-1500, Flue gas temperature
}
```

### BurnerOperatingData

```python
{
  "fuel_type": string,        # natural_gas, diesel, heavy_oil, coal
  "fuel_flow_rate": float,    # kg/hr or m3/hr
  "air_flow_rate": float,     # m3/hr
  "steam_output": float,      # kg/hr
  "feedwater_temp": float,    # °C
  "steam_pressure": float,    # bar
  "ambient_temp": float       # °C (optional, default: 25)
}
```

## Deterministic Calculations

All calculations use **zero-hallucination deterministic algorithms**:

### Excess Air Calculation

```
Excess Air (%) = (O2 / (21 - O2)) × 100
```

### Combustion Efficiency (ASME PTC 4)

```
Efficiency (%) = 100 - (Stack Loss + Radiation Loss + Unburned Loss)

Stack Loss = K × (T_flue - T_ambient) / (21 - O2)
where K ≈ 0.68 for natural gas
```

### Air-Fuel Ratio

```
Actual AFR = (Air Flow × Air Density) / Fuel Flow
Stoichiometric AFR = 17.2 for natural gas
```

## Monitoring

### Prometheus Metrics

The API exports the following metrics at `/metrics`:

- `flueflow_api_requests_total`: Total API requests (by method, endpoint, status)
- `flueflow_api_request_duration_seconds`: Request duration histogram
- `flueflow_analysis_total`: Total analyses performed (by type)
- `flueflow_efficiency_percent`: Efficiency calculation distribution

### Logging

All requests are logged with:
- Timestamp
- HTTP method and path
- Status code
- Response time
- User ID (if authenticated)

## Security

### Authentication

- JWT bearer token authentication
- OAuth2 password flow
- 30-minute token expiration
- Secure password hashing with bcrypt

### Middleware

- **CORS**: Configured for specific domains
- **Trusted Host**: Prevents host header attacks
- **Rate Limiting**: Per-IP rate limits
- **Request Logging**: All requests logged for audit

## Deployment

### Docker

```bash
docker build -t flueflow-api:latest .
docker run -d -p 8000:8000 \
  -e JWT_SECRET_KEY=your-secret-key \
  --name flueflow-api \
  flueflow-api:latest
```

### Production Configuration

```bash
# Environment variables
export JWT_SECRET_KEY="your-production-secret-key"
export ALLOWED_ORIGINS="https://flueflow.io,https://app.flueflow.io"
export LOG_LEVEL="INFO"

# Run with multiple workers
uvicorn tools:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing

### Unit Tests

```bash
pytest tests/ -v --cov=. --cov-report=html
```

### Integration Tests

```bash
# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=demo@flueflow.io&password=demo_password_123"

# Test analysis endpoint
curl -X POST http://localhost:8000/api/v1/analyze-flue-gas \
  -H "Authorization: Bearer $TOKEN" \
  -d @test_data/flue_gas_sample.json
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- **FastAPI** 0.104.1 - Web framework
- **Uvicorn** 0.24.0 - ASGI server
- **Pydantic** 2.5.0 - Data validation
- **python-jose** 3.3.0 - JWT handling
- **passlib** 1.7.4 - Password hashing
- **slowapi** 0.1.9 - Rate limiting
- **prometheus-client** 0.19.0 - Metrics

## Support

For issues, questions, or contributions:
- **Email**: support@greenlang.io
- **Documentation**: https://docs.greenlang.io/flueflow
- **API Status**: https://status.greenlang.io

## License

Copyright © 2025 GreenLang. All rights reserved.

---

**GL-018 FLUEFLOW API** - Production-grade combustion optimization API for climate compliance.
