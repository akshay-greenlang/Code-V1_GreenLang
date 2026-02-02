# GreenLang Emission Factor API

Production-grade REST API for querying emission factors and calculating GHG emissions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn greenlang.api.main:app --reload --port 8000

# Access documentation
open http://localhost:8000/api/docs
```

## Features

- **327+ Emission Factors**: US, EU, UK with multi-gas breakdown
- **Multi-Scope Calculations**: Scope 1, 2, and 3 emissions
- **Batch Processing**: Up to 100 calculations per request
- **High Performance**: <50ms response time, 1000 req/sec
- **Production Ready**: Docker, Redis caching, rate limiting, horizontal scaling

## API Endpoints

### Factor Queries

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/factors` | List all factors (paginated) |
| GET | `/api/v1/factors/{id}` | Get specific factor by ID |
| GET | `/api/v1/factors/search?q=diesel` | Search factors |
| GET | `/api/v1/factors/category/{fuel_type}` | Get by fuel type |
| GET | `/api/v1/factors/scope/{scope}` | Get by scope |

### Calculations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/calculate` | Calculate emissions |
| POST | `/api/v1/calculate/batch` | Batch calculate (max 100) |
| POST | `/api/v1/calculate/scope1` | Scope 1 calculation |
| POST | `/api/v1/calculate/scope2` | Scope 2 calculation |
| POST | `/api/v1/calculate/scope3` | Scope 3 calculation |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/stats` | API statistics |
| GET | `/api/v1/stats/coverage` | Coverage statistics |

## Examples

### Calculate Diesel Emissions

```bash
curl -X POST "http://localhost:8000/api/v1/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "activity_amount": 100,
    "activity_unit": "gallons",
    "geography": "US"
  }'
```

**Response:**
```json
{
  "calculation_id": "calc_abc123xyz",
  "emissions_kg_co2e": 1021.0,
  "emissions_tonnes_co2e": 1.021,
  "emissions_by_gas": {
    "CO2": 1018.0,
    "CH4": 2.3,
    "N2O": 0.7
  },
  "factor_used": {
    "factor_id": "EF:US:diesel:2024:v1",
    "fuel_type": "diesel",
    "co2e_per_unit": 10.21,
    "source": "EPA",
    "data_quality_score": 4.6
  },
  "timestamp": "2025-11-19T10:30:00Z"
}
```

### Batch Calculate

```bash
curl -X POST "http://localhost:8000/api/v1/calculate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "calculations": [
      {
        "fuel_type": "diesel",
        "activity_amount": 100,
        "activity_unit": "gallons",
        "geography": "US"
      },
      {
        "fuel_type": "natural_gas",
        "activity_amount": 500,
        "activity_unit": "therms",
        "geography": "US"
      }
    ]
  }'
```

## Docker Deployment

```bash
# Build image
docker build -t greenlang-api:1.0.0 .

# Run with Docker Compose (API + Redis)
docker-compose up -d

# Check health
curl http://localhost:8000/api/v1/health
```

## Testing

```bash
# Run all tests
pytest greenlang/api/tests/ -v

# Run with coverage
pytest greenlang/api/tests/ --cov=greenlang.api --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Performance

- **Response Time**: <50ms (95th percentile)
- **Throughput**: 1000 requests/second
- **Caching**: Redis with 1-hour TTL
- **Rate Limiting**: 500-1000 requests/minute per IP
- **Scaling**: Horizontal (add more instances)

## Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI Spec**: http://localhost:8000/api/openapi.json
- **Deployment Guide**: [docs/API_DEPLOYMENT.md](../../docs/API_DEPLOYMENT.md)

## Architecture

```
FastAPI Application
├── main.py           # API endpoints and middleware
├── models.py         # Pydantic request/response models
├── cache_config.py   # Redis caching configuration
├── Dockerfile        # Production container
├── docker-compose.yml # Local dev environment
├── requirements.txt  # Python dependencies
└── tests/
    ├── test_api.py   # Integration tests
    └── conftest.py   # Test fixtures
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `CACHE_ENABLED` | Enable caching | `true` |
| `LOG_LEVEL` | Logging level | `info` |
| `WORKERS` | Uvicorn workers | `4` |

## Support

- **Documentation**: https://docs.greenlang.io
- **Issues**: https://github.com/greenlang/issues
- **Email**: support@greenlang.io

## License

Apache 2.0
