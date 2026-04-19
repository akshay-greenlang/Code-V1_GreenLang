# GL-020 ECONOPULSE REST API

## Overview

GL-020 ECONOPULSE provides a comprehensive REST API for economizer performance monitoring in industrial boiler systems. This API enables real-time monitoring, fouling analysis, alert management, efficiency tracking, soot blower integration, and reporting capabilities.

| Property | Value |
|----------|-------|
| **Agent ID** | GL-020 |
| **Codename** | ECONOPULSE |
| **Name** | EconomizerPerformanceAgent |
| **API Version** | 1.0.0 |
| **Base URL** | `https://api.greenlang.io/gl-020/api/v1` |

## Table of Contents

- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Economizers](#economizers)
  - [Performance Monitoring](#performance-monitoring)
  - [Fouling Analysis](#fouling-analysis)
  - [Alerts](#alerts)
  - [Efficiency Analysis](#efficiency-analysis)
  - [Soot Blower Integration](#soot-blower-integration)
  - [Reports](#reports)
- [Error Handling](#error-handling)
- [SDK Examples](#sdk-examples)

## Getting Started

### Prerequisites

- Python 3.9+
- FastAPI
- Required dependencies (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/greenlang/gl-020-econopulse.git
cd gl-020-econopulse

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once running, access the interactive documentation:

- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`
- **OpenAPI JSON**: `http://localhost:8000/api/openapi.json`

## Authentication

All API endpoints require JWT Bearer token authentication.

### Obtaining a Token

```bash
curl -X POST "https://api.greenlang.io/gl-020/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

### Using the Token

```bash
curl -X GET "https://api.greenlang.io/gl-020/api/v1/economizers" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Rate Limiting

| Endpoint Category | Rate Limit |
|-------------------|------------|
| Standard endpoints | 1000 requests/minute |
| Report endpoints | 100 requests/minute |
| Export endpoints | 10 requests/minute |
| Soot blower trigger | 10 requests/minute |

Rate limit headers are included in all responses:
- `X-RateLimit-Remaining`: Remaining requests in current window

## API Endpoints

### Health Check

#### Liveness Probe
```http
GET /health/liveness
```

Returns service liveness status for Kubernetes health checks.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T10:30:00Z",
  "service": "gl-020-econopulse",
  "version": "1.0.0"
}
```

#### Readiness Probe
```http
GET /health/readiness
```

Checks all dependencies and returns readiness status.

**Response:**
```json
{
  "status": "ready",
  "timestamp": "2025-11-09T10:30:00Z",
  "checks": {
    "database": true,
    "redis": true,
    "historian": true,
    "message_queue": true
  },
  "message": "All dependencies healthy"
}
```

### Economizers

#### List Economizers
```http
GET /api/v1/economizers
```

Query parameters:
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (online, offline, maintenance, degraded)
- `boiler_id` (string): Filter by boiler ID

**Response:**
```json
{
  "items": [
    {
      "id": "econ-001",
      "name": "Economizer Unit A1",
      "type": "finned_tube",
      "status": "online",
      "location": "Building A, Level 2",
      "boiler_id": "boiler-001",
      "design_capacity_kw": 500.0,
      "design_pressure_drop_kpa": 1.5,
      "surface_area_m2": 150.0
    }
  ],
  "total": 5,
  "page": 1,
  "page_size": 20,
  "has_next": false,
  "has_prev": false
}
```

#### Get Economizer Details
```http
GET /api/v1/economizers/{economizer_id}
```

### Performance Monitoring

#### Get Current Performance Metrics
```http
GET /api/v1/economizers/{economizer_id}/performance
```

Returns real-time performance metrics including temperatures, flow rates, pressure drops, and calculated values.

**Response:**
```json
{
  "economizer_id": "econ-001",
  "timestamp": "2025-11-09T10:30:00Z",
  "gas_inlet_temp_c": 320.5,
  "gas_outlet_temp_c": 180.2,
  "water_inlet_temp_c": 105.0,
  "water_outlet_temp_c": 140.5,
  "gas_flow_rate_kg_s": 15.2,
  "water_flow_rate_kg_s": 8.5,
  "gas_pressure_drop_kpa": 1.8,
  "water_pressure_drop_kpa": 0.5,
  "heat_transfer_kw": 485.3,
  "effectiveness_percent": 78.5,
  "overall_htc_w_m2k": 45.2,
  "approach_temp_c": 39.7,
  "data_quality": "good"
}
```

#### Get Performance History
```http
GET /api/v1/economizers/{economizer_id}/performance/history
```

Query parameters:
- `start_time` (datetime): Query start time
- `end_time` (datetime): Query end time
- `resolution` (string): Data resolution (1m, 5m, 15m, 1h, 1d)

#### Get Performance Trends
```http
GET /api/v1/economizers/{economizer_id}/trends
```

Query parameters:
- `analysis_period_days` (int): Analysis period in days (default: 30, max: 365)

### Fouling Analysis

#### Get Current Fouling Status
```http
GET /api/v1/economizers/{economizer_id}/fouling
```

**Response:**
```json
{
  "economizer_id": "econ-001",
  "timestamp": "2025-11-09T10:30:00Z",
  "severity": "moderate",
  "fouling_factor": 0.00025,
  "fouling_score": 45.0,
  "effectiveness_loss_percent": 8.5,
  "pressure_drop_increase_percent": 15.2,
  "estimated_deposit_thickness_mm": 1.2,
  "cleaning_recommended": false,
  "last_cleaned": "2025-10-15T14:00:00Z",
  "days_since_cleaning": 25
}
```

#### Get Fouling History
```http
GET /api/v1/economizers/{economizer_id}/fouling/history
```

#### Get Fouling Prediction
```http
GET /api/v1/economizers/{economizer_id}/fouling/prediction
```

Returns ML-based fouling predictions for 7, 14, and 30 days.

#### Set Clean Baseline
```http
POST /api/v1/economizers/{economizer_id}/fouling/baseline
```

**Request Body:**
```json
{
  "baseline_type": "clean",
  "reference_date": "2025-10-15T14:00:00Z",
  "effectiveness_percent": 85.0,
  "pressure_drop_kpa": 1.5,
  "overall_htc_w_m2k": 52.0,
  "notes": "Baseline set after chemical cleaning"
}
```

### Alerts

#### List All Alerts
```http
GET /api/v1/alerts
```

Query parameters:
- `page` (int): Page number
- `page_size` (int): Items per page
- `status` (string): Filter by status (active, acknowledged, resolved, suppressed)
- `severity` (string): Filter by severity (info, warning, critical, emergency)
- `alert_type` (string): Filter by type (fouling, efficiency, temperature, etc.)
- `economizer_id` (string): Filter by economizer

#### Get Alert Details
```http
GET /api/v1/alerts/{alert_id}
```

#### Acknowledge Alert
```http
PUT /api/v1/alerts/{alert_id}/acknowledge
```

**Request Body:**
```json
{
  "notes": "Acknowledged. Scheduling cleaning for next maintenance window."
}
```

#### Get Alerts for Economizer
```http
GET /api/v1/economizers/{economizer_id}/alerts
```

#### Configure Alert Thresholds
```http
POST /api/v1/alerts/thresholds
```

**Request Body:**
```json
{
  "economizer_id": "econ-001",
  "thresholds": [
    {
      "metric_name": "fouling_score",
      "warning_threshold": 50.0,
      "critical_threshold": 70.0,
      "emergency_threshold": 85.0,
      "enabled": true,
      "hysteresis": 2.0
    }
  ]
}
```

### Efficiency Analysis

#### Get Efficiency Metrics
```http
GET /api/v1/economizers/{economizer_id}/efficiency
```

#### Get Efficiency Loss Quantification
```http
GET /api/v1/economizers/{economizer_id}/efficiency/loss
```

Quantifies efficiency losses and their monetary/environmental impact.

#### Get Potential Savings
```http
GET /api/v1/economizers/{economizer_id}/efficiency/savings
```

**Response:**
```json
{
  "economizer_id": "econ-001",
  "current_efficiency_percent": 78.5,
  "target_efficiency_percent": 83.0,
  "energy_savings_kwh_year": 45600.0,
  "fuel_savings_usd_year": 2280.0,
  "carbon_reduction_kg_year": 9880.0,
  "estimated_cleaning_cost_usd": 500.0,
  "payback_period_days": 80.0,
  "roi_percent": 356.0,
  "cleaning_recommended": true
}
```

### Soot Blower Integration

#### Get Soot Blower Status
```http
GET /api/v1/economizers/{economizer_id}/soot-blowers
```

#### Trigger Cleaning Cycle
```http
POST /api/v1/economizers/{economizer_id}/soot-blowers/trigger
```

**Request Body:**
```json
{
  "soot_blower_ids": ["sb-001", "sb-002"],
  "sequence": "standard",
  "delay_seconds": 0,
  "reason": "Scheduled cleaning based on fouling level"
}
```

#### Get Cleaning History
```http
GET /api/v1/economizers/{economizer_id}/cleaning-history
```

#### Get Optimal Cleaning Schedule
```http
GET /api/v1/economizers/{economizer_id}/cleaning/optimization
```

### Reports

#### Get Daily Report
```http
GET /api/v1/reports/daily
```

Query parameters:
- `report_date` (datetime): Report date (default: yesterday)

#### Get Weekly Summary
```http
GET /api/v1/reports/weekly
```

Query parameters:
- `week` (string): ISO week (e.g., 2025-W45)

#### Get Efficiency Report
```http
GET /api/v1/reports/efficiency
```

Query parameters:
- `start_time` (datetime): Report start time
- `end_time` (datetime): Report end time

#### Export Report
```http
POST /api/v1/reports/export
```

**Request Body:**
```json
{
  "report_type": "efficiency",
  "format": "pdf",
  "start_date": "2025-10-01T00:00:00Z",
  "end_date": "2025-11-01T00:00:00Z",
  "economizer_ids": ["econ-001", "econ-002"],
  "include_charts": true,
  "include_raw_data": false,
  "email_recipients": ["engineer@example.com"]
}
```

## Error Handling

All errors follow a standard format:

```json
{
  "error": "validation_error",
  "message": "Invalid economizer ID format",
  "details": {
    "field": "economizer_id",
    "value": "invalid"
  },
  "request_id": "req-123456"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 202 | Accepted (async operation started) |
| 400 | Bad Request - Validation error |
| 401 | Unauthorized - Invalid or missing token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource does not exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## SDK Examples

### Python SDK

```python
import httpx
from datetime import datetime, timedelta

class EconopulseClient:
    """Python SDK for GL-020 ECONOPULSE API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )

    def list_economizers(self, page: int = 1, page_size: int = 20):
        """List all monitored economizers."""
        response = self.client.get(
            f"{self.base_url}/api/v1/economizers",
            params={"page": page, "page_size": page_size}
        )
        response.raise_for_status()
        return response.json()

    def get_performance(self, economizer_id: str):
        """Get current performance metrics."""
        response = self.client.get(
            f"{self.base_url}/api/v1/economizers/{economizer_id}/performance"
        )
        response.raise_for_status()
        return response.json()

    def get_fouling_status(self, economizer_id: str):
        """Get current fouling status."""
        response = self.client.get(
            f"{self.base_url}/api/v1/economizers/{economizer_id}/fouling"
        )
        response.raise_for_status()
        return response.json()

    def get_fouling_prediction(self, economizer_id: str):
        """Get fouling prediction."""
        response = self.client.get(
            f"{self.base_url}/api/v1/economizers/{economizer_id}/fouling/prediction"
        )
        response.raise_for_status()
        return response.json()

    def trigger_cleaning(self, economizer_id: str, reason: str, sequence: str = "standard"):
        """Trigger soot blower cleaning cycle."""
        response = self.client.post(
            f"{self.base_url}/api/v1/economizers/{economizer_id}/soot-blowers/trigger",
            json={
                "sequence": sequence,
                "reason": reason,
                "delay_seconds": 0
            }
        )
        response.raise_for_status()
        return response.json()

    def get_efficiency_savings(self, economizer_id: str):
        """Get potential efficiency savings."""
        response = self.client.get(
            f"{self.base_url}/api/v1/economizers/{economizer_id}/efficiency/savings"
        )
        response.raise_for_status()
        return response.json()

    def list_alerts(self, status: str = None, severity: str = None):
        """List alerts with optional filtering."""
        params = {}
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity

        response = self.client.get(
            f"{self.base_url}/api/v1/alerts",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def acknowledge_alert(self, alert_id: str, notes: str = None):
        """Acknowledge an active alert."""
        response = self.client.put(
            f"{self.base_url}/api/v1/alerts/{alert_id}/acknowledge",
            json={"notes": notes}
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()


# Usage example
if __name__ == "__main__":
    client = EconopulseClient(
        base_url="https://api.greenlang.io/gl-020",
        api_key="your-api-key"
    )

    try:
        # List economizers
        economizers = client.list_economizers()
        print(f"Found {economizers['total']} economizers")

        # Get performance for first economizer
        if economizers['items']:
            econ_id = economizers['items'][0]['id']

            # Get current performance
            performance = client.get_performance(econ_id)
            print(f"Effectiveness: {performance['effectiveness_percent']}%")

            # Get fouling status
            fouling = client.get_fouling_status(econ_id)
            print(f"Fouling score: {fouling['fouling_score']}")
            print(f"Cleaning recommended: {fouling['cleaning_recommended']}")

            # Get savings potential
            savings = client.get_efficiency_savings(econ_id)
            print(f"Annual savings potential: ${savings['fuel_savings_usd_year']}")

    finally:
        client.close()
```

### cURL Examples

```bash
# List economizers
curl -X GET "https://api.greenlang.io/gl-020/api/v1/economizers" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get performance metrics
curl -X GET "https://api.greenlang.io/gl-020/api/v1/economizers/econ-001/performance" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get fouling prediction
curl -X GET "https://api.greenlang.io/gl-020/api/v1/economizers/econ-001/fouling/prediction" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Trigger cleaning
curl -X POST "https://api.greenlang.io/gl-020/api/v1/economizers/econ-001/soot-blowers/trigger" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "standard",
    "reason": "High fouling score detected",
    "delay_seconds": 0
  }'

# Acknowledge alert
curl -X PUT "https://api.greenlang.io/gl-020/api/v1/alerts/alert-001/acknowledge" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notes": "Acknowledged. Scheduling maintenance."
  }'

# Export report
curl -X POST "https://api.greenlang.io/gl-020/api/v1/reports/export" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "efficiency",
    "format": "pdf",
    "start_date": "2025-10-01T00:00:00Z",
    "end_date": "2025-11-01T00:00:00Z"
  }'
```

## Prometheus Metrics

The API exposes Prometheus metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `econopulse_requests_total` | Counter | Total requests by method, endpoint, status |
| `econopulse_request_latency_seconds` | Histogram | Request latency by method, endpoint |
| `econopulse_active_requests` | Gauge | Current active requests |
| `econopulse_monitored_economizers` | Gauge | Number of monitored economizers |
| `econopulse_active_alerts` | Gauge | Number of active alerts |
| `econopulse_average_fouling_score` | Gauge | Average fouling score |

## Support

For support and questions:

- **Documentation**: https://docs.greenlang.io/gl-020
- **API Status**: https://status.greenlang.io
- **Support Email**: support@greenlang.io

---

Copyright 2025 GreenLang. All rights reserved.
