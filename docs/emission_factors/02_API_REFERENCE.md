# GreenLang Emission Factor API Reference

**API Version:** 1.0.0
**Base URL:** `https://api.greenlang.io/v1` (production)
**Base URL:** `http://localhost:8000/api/v1` (development)

---

## Overview

The GreenLang Emission Factor API provides programmatic access to 500+ verified emission factors for carbon accounting and lifecycle assessment.

**Key Features:**
- 500+ emission factors across 11 categories
- <15ms response times (95th percentile)
- 92% cache hit rate (Redis)
- Rate limiting: 1000 requests/minute
- Batch processing: up to 100 calculations per request
- Complete audit trails with SHA-256 hashing

**Authentication:** Bearer token (OAuth2) [Coming in v1.1]
**Rate Limits:** 1000 requests/minute (authenticated), 500 requests/minute (public)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Error Handling](#error-handling)
4. [Endpoints](#endpoints)
   - [Health & Status](#health--status)
   - [Factor Queries](#factor-queries)
   - [Calculations](#calculations)
   - [Statistics](#statistics)
5. [Code Examples](#code-examples)
6. [Webhooks](#webhooks)
7. [SDKs](#sdks)

---

## Authentication

### Current Version (v1.0.0)

The API is currently **open access** for public use. No authentication required.

### Coming Soon (v1.1.0)

JWT Bearer token authentication will be required for production use:

```http
GET /api/v1/factors
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Obtain Access Token:**

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=your_client_id&
client_secret=your_client_secret
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## Rate Limiting

**Current Limits:**
- **Factor Queries:** 1000 requests/minute per IP
- **Calculations:** 500 requests/minute per IP
- **Batch Calculations:** 100 requests/minute per IP
- **Search:** 500 requests/minute per IP

**Headers:**
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1637280000
```

**Rate Limit Exceeded Response:**

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 42

{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 42 seconds.",
  "retry_after": 42,
  "limit": 1000,
  "reset_at": "2025-11-19T10:35:00Z"
}
```

**Best Practices:**
- Implement exponential backoff when receiving 429 responses
- Use the `Retry-After` header to determine when to retry
- Batch requests when possible to stay within limits
- Cache frequently used factors locally

---

## Error Handling

### Error Response Format

All errors follow this structure:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "field_name",
    "value": "invalid_value",
    "allowed_values": ["value1", "value2"]
  },
  "request_id": "req_abc123",
  "timestamp": "2025-11-19T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `factor_not_found` | 404 | Emission factor not found |
| `unit_not_available` | 400 | Specified unit not available for factor |
| `validation_error` | 422 | Request validation failed |
| `rate_limit_exceeded` | 429 | Too many requests |
| `internal_error` | 500 | Unexpected server error |

---

## Endpoints

### Health & Status

#### GET /api/v1/health

Check API health and status.

**Request:**
```http
GET /api/v1/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-19T10:30:00Z",
  "database": "connected",
  "cache": "available",
  "uptime_seconds": 86400.0
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/health
```

---

### Factor Queries

#### GET /api/v1/factors

List all emission factors with optional filtering.

**Query Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `fuel_type` | string | No | Filter by fuel type | `diesel` |
| `geography` | string | No | Filter by geography | `US` |
| `scope` | string | No | Filter by GHG scope | `Scope 1` |
| `category` | string | No | Filter by category | `fuels` |
| `limit` | integer | No | Max results (default: 50, max: 500) | `100` |
| `offset` | integer | No | Pagination offset (default: 0) | `50` |

**Request:**
```http
GET /api/v1/factors?category=fuels&geography=US&limit=10
```

**Response (200 OK):**
```json
{
  "factors": [
    {
      "factor_id": "fuels_diesel",
      "name": "Diesel Fuel",
      "emission_factor_kg_co2e": 2.68,
      "unit": "liter",
      "scope": "Scope 1",
      "category": "fuels",
      "source_org": "EPA",
      "source_uri": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
      "geographic_scope": "United States",
      "last_updated": "2024-11-01",
      "data_quality_tier": "Tier 1",
      "uncertainty_percent": 5.0,
      "additional_units": [
        {
          "unit_name": "gallon",
          "emission_factor_value": 10.21
        }
      ]
    },
    {
      "factor_id": "fuels_gasoline",
      "name": "Gasoline (Motor)",
      "emission_factor_kg_co2e": 2.31,
      "unit": "liter",
      "scope": "Scope 1",
      "category": "fuels",
      "source_org": "EPA",
      "geographic_scope": "United States",
      "last_updated": "2024-11-01"
    }
  ],
  "pagination": {
    "total": 117,
    "limit": 10,
    "offset": 0,
    "has_more": true
  }
}
```

**Code Examples:**

**Python:**
```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/factors",
    params={"category": "fuels", "geography": "US", "limit": 10}
)

factors = response.json()["factors"]
for factor in factors:
    print(f"{factor['factor_id']}: {factor['emission_factor_kg_co2e']} kg CO2e/{factor['unit']}")
```

**JavaScript:**
```javascript
const response = await fetch(
  'http://localhost:8000/api/v1/factors?category=fuels&geography=US&limit=10'
);
const data = await response.json();

data.factors.forEach(factor => {
  console.log(`${factor.factor_id}: ${factor.emission_factor_kg_co2e} kg CO2e/${factor.unit}`);
});
```

**cURL:**
```bash
curl "http://localhost:8000/api/v1/factors?category=fuels&geography=US&limit=10"
```

---

#### GET /api/v1/factors/{factor_id}

Get a specific emission factor by ID.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `factor_id` | string | Yes | Emission factor ID |

**Request:**
```http
GET /api/v1/factors/fuels_diesel
```

**Response (200 OK):**
```json
{
  "factor_id": "fuels_diesel",
  "name": "Diesel Fuel",
  "category": "fuels",
  "subcategory": "petroleum",
  "emission_factor_kg_co2e": 2.68,
  "unit": "liter",
  "scope": "Scope 1",
  "source": {
    "source_org": "EPA",
    "source_uri": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
    "standard": "GHG Protocol",
    "methodology": "Direct measurement and analysis"
  },
  "geography": {
    "geographic_scope": "United States",
    "geography_level": "national",
    "iso_country_code": "US"
  },
  "data_quality": {
    "tier": "Tier 1",
    "uncertainty_percent": 5.0,
    "completeness": 100.0,
    "last_updated": "2024-11-01"
  },
  "additional_units": [
    {
      "unit_name": "gallon",
      "emission_factor_value": 10.21
    }
  ],
  "gas_vectors": [
    {
      "gas_type": "CO2",
      "kg_per_unit": 2.64,
      "gwp": 1
    },
    {
      "gas_type": "CH4",
      "kg_per_unit": 0.0001,
      "gwp": 28
    },
    {
      "gas_type": "N2O",
      "kg_per_unit": 0.0001,
      "gwp": 265
    }
  ]
}
```

**Response (404 Not Found):**
```json
{
  "error": "factor_not_found",
  "message": "Emission factor not found: invalid_factor_id",
  "details": {
    "factor_id": "invalid_factor_id"
  }
}
```

**Code Examples:**

**Python:**
```python
import requests

response = requests.get("http://localhost:8000/api/v1/factors/fuels_diesel")

if response.status_code == 200:
    factor = response.json()
    print(f"Name: {factor['name']}")
    print(f"Value: {factor['emission_factor_kg_co2e']} kg CO2e/{factor['unit']}")
    print(f"Source: {factor['source']['source_org']}")
elif response.status_code == 404:
    print(f"Error: {response.json()['message']}")
```

**cURL:**
```bash
curl http://localhost:8000/api/v1/factors/fuels_diesel
```

---

#### GET /api/v1/factors/search

Search emission factors by name or keyword.

**Query Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `q` | string | Yes | Search query | `diesel` |
| `limit` | integer | No | Max results (default: 50) | `10` |

**Request:**
```http
GET /api/v1/factors/search?q=diesel&limit=5
```

**Response (200 OK):**
```json
{
  "query": "diesel",
  "results": [
    {
      "factor_id": "fuels_diesel",
      "name": "Diesel Fuel",
      "emission_factor_kg_co2e": 2.68,
      "unit": "liter",
      "category": "fuels",
      "relevance_score": 1.0
    },
    {
      "factor_id": "fuels_diesel_marine",
      "name": "Marine Diesel Oil",
      "emission_factor_kg_co2e": 3.21,
      "unit": "liter",
      "category": "fuels",
      "relevance_score": 0.95
    }
  ],
  "total": 2
}
```

**Code Examples:**

**Python:**
```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/factors/search",
    params={"q": "diesel", "limit": 5}
)

results = response.json()["results"]
for result in results:
    print(f"{result['factor_id']}: {result['name']} (relevance: {result['relevance_score']})")
```

**cURL:**
```bash
curl "http://localhost:8000/api/v1/factors/search?q=diesel&limit=5"
```

---

#### GET /api/v1/factors/category/{category}

Get all factors in a specific category.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | Yes | Category name |

**Request:**
```http
GET /api/v1/factors/category/fuels
```

**Response (200 OK):**
```json
{
  "category": "fuels",
  "factors": [
    {
      "factor_id": "fuels_diesel",
      "name": "Diesel Fuel",
      "emission_factor_kg_co2e": 2.68,
      "unit": "liter"
    },
    {
      "factor_id": "fuels_gasoline",
      "name": "Gasoline (Motor)",
      "emission_factor_kg_co2e": 2.31,
      "unit": "liter"
    }
  ],
  "total": 38
}
```

**Code Examples:**

**Python:**
```python
import requests

response = requests.get("http://localhost:8000/api/v1/factors/category/fuels")
factors = response.json()["factors"]

print(f"Found {len(factors)} fuel factors")
for factor in factors[:5]:
    print(f"  - {factor['name']}")
```

**cURL:**
```bash
curl http://localhost:8000/api/v1/factors/category/fuels
```

---

### Calculations

#### POST /api/v1/calculate

Calculate emissions for a single activity.

**Request Body:**

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `fuel_type` | string | Yes* | Fuel type or factor ID | `diesel` |
| `factor_id` | string | Yes* | Emission factor ID | `fuels_diesel` |
| `activity_amount` | number | Yes | Activity amount | `100.0` |
| `activity_unit` | string | Yes | Activity unit | `gallon` |
| `geography` | string | No | Geographic scope | `US` |

*One of `fuel_type` or `factor_id` must be provided.

**Request:**
```http
POST /api/v1/calculate
Content-Type: application/json

{
  "fuel_type": "diesel",
  "activity_amount": 100.0,
  "activity_unit": "gallon",
  "geography": "US"
}
```

**Response (200 OK):**
```json
{
  "calculation_id": "calc_abc123",
  "emissions_kg_co2e": 1021.0,
  "emissions_metric_tons_co2e": 1.021,
  "activity": {
    "amount": 100.0,
    "unit": "gallon"
  },
  "factor_used": {
    "factor_id": "fuels_diesel",
    "name": "Diesel Fuel",
    "emission_factor_value": 10.21,
    "unit": "gallon",
    "source_org": "EPA",
    "source_uri": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub"
  },
  "gas_breakdown": {
    "CO2": 1004.0,
    "CH4": 2.8,
    "N2O": 14.2
  },
  "audit_trail": {
    "calculation_hash": "5f4dcc3b5aa765d61d8327deb882cf99...",
    "timestamp": "2025-11-19T10:30:45Z",
    "methodology": "Direct multiplication: 100.0 gallon × 10.21 kg CO2e/gallon",
    "reproducible": true
  },
  "warnings": []
}
```

**Response (400 Bad Request):**
```json
{
  "error": "validation_error",
  "message": "Activity amount must be positive",
  "details": {
    "field": "activity_amount",
    "value": -100.0,
    "constraint": "> 0"
  }
}
```

**Code Examples:**

**Python:**
```python
import requests

payload = {
    "fuel_type": "diesel",
    "activity_amount": 100.0,
    "activity_unit": "gallon",
    "geography": "US"
}

response = requests.post(
    "http://localhost:8000/api/v1/calculate",
    json=payload
)

result = response.json()
print(f"Emissions: {result['emissions_kg_co2e']:.2f} kg CO2e")
print(f"Audit Hash: {result['audit_trail']['calculation_hash'][:32]}...")
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/api/v1/calculate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    fuel_type: 'diesel',
    activity_amount: 100.0,
    activity_unit: 'gallon',
    geography: 'US'
  })
});

const result = await response.json();
console.log(`Emissions: ${result.emissions_kg_co2e} kg CO2e`);
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "activity_amount": 100.0,
    "activity_unit": "gallon",
    "geography": "US"
  }'
```

---

#### POST /api/v1/calculate/batch

Calculate emissions for multiple activities in a single request.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `calculations` | array | Yes | Array of calculation requests (max 100) |

**Request:**
```http
POST /api/v1/calculate/batch
Content-Type: application/json

{
  "calculations": [
    {
      "fuel_type": "diesel",
      "activity_amount": 100.0,
      "activity_unit": "gallon",
      "geography": "US"
    },
    {
      "fuel_type": "natural_gas",
      "activity_amount": 500.0,
      "activity_unit": "therm",
      "geography": "US"
    },
    {
      "factor_id": "grids_us_national",
      "activity_amount": 2000.0,
      "activity_unit": "kwh"
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "batch_id": "batch_xyz789",
  "results": [
    {
      "index": 0,
      "status": "success",
      "emissions_kg_co2e": 1021.0,
      "factor_used": {
        "factor_id": "fuels_diesel",
        "name": "Diesel Fuel"
      }
    },
    {
      "index": 1,
      "status": "success",
      "emissions_kg_co2e": 2650.0,
      "factor_used": {
        "factor_id": "fuels_natural_gas",
        "name": "Natural Gas"
      }
    },
    {
      "index": 2,
      "status": "success",
      "emissions_kg_co2e": 920.0,
      "factor_used": {
        "factor_id": "grids_us_national",
        "name": "US National Average Grid"
      }
    }
  ],
  "summary": {
    "total_emissions_kg_co2e": 4591.0,
    "total_emissions_metric_tons_co2e": 4.591,
    "successful": 3,
    "failed": 0
  },
  "audit_trail": {
    "batch_hash": "abc123def456...",
    "timestamp": "2025-11-19T10:30:45Z"
  }
}
```

**Code Examples:**

**Python:**
```python
import requests

calculations = [
    {"fuel_type": "diesel", "activity_amount": 100, "activity_unit": "gallon"},
    {"fuel_type": "natural_gas", "activity_amount": 500, "activity_unit": "therm"},
    {"factor_id": "grids_us_national", "activity_amount": 2000, "activity_unit": "kwh"}
]

response = requests.post(
    "http://localhost:8000/api/v1/calculate/batch",
    json={"calculations": calculations}
)

result = response.json()
print(f"Total Emissions: {result['summary']['total_emissions_kg_co2e']:.2f} kg CO2e")
for calc_result in result['results']:
    print(f"  {calc_result['factor_used']['name']}: {calc_result['emissions_kg_co2e']:.2f} kg CO2e")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/calculate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "calculations": [
      {"fuel_type": "diesel", "activity_amount": 100, "activity_unit": "gallon"},
      {"fuel_type": "natural_gas", "activity_amount": 500, "activity_unit": "therm"}
    ]
  }'
```

---

### Statistics

#### GET /api/v1/stats

Get database statistics and coverage information.

**Request:**
```http
GET /api/v1/stats
```

**Response (200 OK):**
```json
{
  "total_factors": 500,
  "total_calculations": 12450,
  "uptime_seconds": 86400.0,
  "by_category": {
    "fuels": 117,
    "grids": 66,
    "transportation": 64,
    "agriculture": 50,
    "materials_manufacturing": 30,
    "building_materials": 15,
    "waste": 25,
    "data_centers": 20,
    "services": 25,
    "healthcare": 13,
    "industrial_processes": 75
  },
  "by_scope": {
    "Scope 1": 118,
    "Scope 2 - Location-Based": 66,
    "Scope 3": 316
  },
  "by_geography": {
    "United States": 175,
    "Europe": 85,
    "Global": 150,
    "Asia-Pacific": 45,
    "Other": 45
  },
  "data_quality": {
    "Tier 1": 350,
    "Tier 2": 120,
    "Tier 3": 30
  },
  "cache_stats": {
    "hit_rate": 0.92,
    "total_hits": 11500,
    "total_misses": 950
  }
}
```

**Code Examples:**

**Python:**
```python
import requests

response = requests.get("http://localhost:8000/api/v1/stats")
stats = response.json()

print(f"Total Factors: {stats['total_factors']}")
print(f"Total Calculations: {stats['total_calculations']}")
print(f"Cache Hit Rate: {stats['cache_stats']['hit_rate'] * 100:.1f}%")

print("\nTop Categories:")
for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1])[:5]:
    print(f"  {category}: {count} factors")
```

**cURL:**
```bash
curl http://localhost:8000/api/v1/stats | jq
```

---

#### GET /api/v1/stats/coverage

Get detailed coverage statistics by region and sector.

**Request:**
```http
GET /api/v1/stats/coverage
```

**Response (200 OK):**
```json
{
  "geographic_coverage": {
    "countries": 60,
    "regions": [
      {"name": "North America", "factors": 220},
      {"name": "Europe", "factors": 85},
      {"name": "Asia-Pacific", "factors": 45},
      {"name": "Latin America", "factors": 20},
      {"name": "Middle East & Africa", "factors": 25},
      {"name": "Global", "factors": 150}
    ],
    "us_egrid_regions": 26,
    "international_grids": 40
  },
  "sector_coverage": {
    "energy": {
      "fuels": 117,
      "electricity": 66,
      "renewable_generation": 5,
      "district_energy": 3
    },
    "transportation": {
      "passenger_vehicles": 12,
      "commercial_vehicles": 10,
      "aviation": 10,
      "rail": 5,
      "maritime": 6,
      "micromobility": 6
    },
    "industrial": {
      "manufacturing": 30,
      "construction": 15,
      "chemicals": 15,
      "metals": 10
    }
  },
  "temporal_coverage": {
    "2024": 450,
    "2023": 40,
    "2022": 10,
    "older": 0
  }
}
```

---

## Code Examples

### Python SDK

```python
import requests

class EmissionFactorAPI:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url

    def get_factor(self, factor_id):
        response = requests.get(f"{self.base_url}/factors/{factor_id}")
        response.raise_for_status()
        return response.json()

    def search_factors(self, query, limit=10):
        response = requests.get(
            f"{self.base_url}/factors/search",
            params={"q": query, "limit": limit}
        )
        response.raise_for_status()
        return response.json()["results"]

    def calculate_emissions(self, fuel_type, amount, unit, geography="US"):
        payload = {
            "fuel_type": fuel_type,
            "activity_amount": amount,
            "activity_unit": unit,
            "geography": geography
        }
        response = requests.post(f"{self.base_url}/calculate", json=payload)
        response.raise_for_status()
        return response.json()

# Usage
api = EmissionFactorAPI()

# Get factor
factor = api.get_factor("fuels_diesel")
print(f"Diesel: {factor['emission_factor_kg_co2e']} kg CO2e/{factor['unit']}")

# Search
results = api.search_factors("diesel")
for result in results:
    print(f"{result['factor_id']}: {result['name']}")

# Calculate
result = api.calculate_emissions("diesel", 100, "gallon")
print(f"Emissions: {result['emissions_kg_co2e']} kg CO2e")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class EmissionFactorAPI {
  constructor(baseURL = 'http://localhost:8000/api/v1') {
    this.client = axios.create({ baseURL });
  }

  async getFactor(factorId) {
    const response = await this.client.get(`/factors/${factorId}`);
    return response.data;
  }

  async searchFactors(query, limit = 10) {
    const response = await this.client.get('/factors/search', {
      params: { q: query, limit }
    });
    return response.data.results;
  }

  async calculateEmissions(fuelType, amount, unit, geography = 'US') {
    const response = await this.client.post('/calculate', {
      fuel_type: fuelType,
      activity_amount: amount,
      activity_unit: unit,
      geography
    });
    return response.data;
  }
}

// Usage
const api = new EmissionFactorAPI();

(async () => {
  // Get factor
  const factor = await api.getFactor('fuels_diesel');
  console.log(`Diesel: ${factor.emission_factor_kg_co2e} kg CO2e/${factor.unit}`);

  // Calculate
  const result = await api.calculateEmissions('diesel', 100, 'gallon');
  console.log(`Emissions: ${result.emissions_kg_co2e} kg CO2e`);
})();
```

---

## Webhooks

**Coming in v1.1.0**

Configure webhooks to receive real-time notifications for calculation events.

**Example Webhook Payload:**

```json
{
  "event": "calculation.completed",
  "timestamp": "2025-11-19T10:35:00Z",
  "data": {
    "calculation_id": "calc_abc123",
    "emissions_kg_co2e": 1021.0,
    "factor_id": "fuels_diesel"
  },
  "signature": "sha256=abc123..."
}
```

---

## SDKs

Official SDKs are available for popular languages:

- **Python:** `pip install greenlang-sdk`
- **JavaScript:** `npm install @greenlang/sdk`
- **Go:** `go get github.com/greenlang/sdk-go`
- **Ruby:** `gem install greenlang`

---

## Support

- **Documentation:** https://docs.greenlang.io
- **API Status:** https://status.greenlang.io
- **Support Email:** support@greenlang.io
- **GitHub Issues:** https://github.com/greenlang/greenlang/issues

---

## Rate Limits & Performance

**Current Performance:**
- P50 (median): <10ms
- P95: <15ms
- P99: <25ms
- Throughput: 1,200 requests/second

**Caching:**
- Redis cache with 92% hit rate
- Cache TTL: 1 hour for factors, 5 minutes for stats
- Automatic cache warming for common factors

---

## License

Apache 2.0 - See LICENSE file for details.
