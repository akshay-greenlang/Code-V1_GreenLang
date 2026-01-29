# GL Normalizer Service

FastAPI-based REST API service for the GreenLang Normalizer (GL-FOUND-X-003).

## Overview

This service provides a production-ready REST API for climate data normalization, including unit conversion, entity normalization, and vocabulary management. It is designed for high availability, observability, and compliance with regulatory standards.

## Features

- **Single Value Normalization**: Normalize individual values with confidence scoring
- **Batch Processing**: Process up to 10,000 items synchronously
- **Async Jobs**: Handle 100K+ items with background processing
- **Vocabulary Management**: Access normalization vocabularies and mappings
- **JWT/API Key Authentication**: Secure access control
- **Rate Limiting**: Configurable rate limits per client
- **Audit Logging**: Complete audit trails for compliance
- **Observability**: Prometheus metrics, OpenTelemetry tracing
- **Health Checks**: Kubernetes-ready health endpoints

## Installation

```bash
pip install gl-normalizer-service
```

## Quick Start

```bash
# Start the service (development)
uvicorn gl_normalizer_service.main:app --reload

# Production
uvicorn gl_normalizer_service.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with CLI
gl-normalizer-service --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, access the interactive documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Configuration

Environment variables (prefix: `GL_NORMALIZER_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (development/staging/production) | development |
| `DEBUG` | Enable debug mode | false |
| `SECRET_KEY` | JWT signing secret (32+ chars) | (required in production) |
| `API_KEY_HEADER` | API key header name | X-API-Key |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379/0 |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | true |
| `RATE_LIMIT_REQUESTS` | Requests per window | 100 |
| `RATE_LIMIT_WINDOW` | Window duration (seconds) | 60 |
| `BATCH_MAX_ITEMS` | Max batch size | 10000 |

## API Endpoints

### Normalization

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/normalize` | Normalize a single value |
| POST | `/v1/normalize/batch` | Batch normalize (up to 10K items) |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/jobs` | Create async job (100K+ items) |
| GET | `/v1/jobs/{job_id}` | Get job status |

### Vocabularies

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/vocabularies` | List available vocabularies |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check |
| GET | `/v1/ready` | Readiness probe |
| GET | `/v1/live` | Liveness probe |

## Usage Examples

### Single Value Normalization

```bash
curl -X POST http://localhost:8000/v1/normalize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "value": "1500",
    "unit": "kg CO2",
    "target_unit": "metric_ton_co2e"
  }'
```

Response:
```json
{
  "api_revision": "2026-01-30",
  "canonical_value": 1.5,
  "canonical_unit": "metric_ton_co2e",
  "confidence": 0.95,
  "needs_review": false,
  "review_reasons": [],
  "audit_id": "aud_a1b2c3d4e5f6",
  "source_value": "1500",
  "source_unit": "kg CO2",
  "conversion_factor": 0.001
}
```

### Batch Normalization

```bash
curl -X POST http://localhost:8000/v1/normalize/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "items": [
      {"id": "item_001", "value": "1500", "unit": "kg CO2"},
      {"id": "item_002", "value": "250", "unit": "MWh"},
      {"id": "item_003", "value": "1000", "unit": "gallons"}
    ],
    "batch_mode": "PARTIAL"
  }'
```

### Async Job Processing

```bash
# Create job
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "items": [...],
    "batch_mode": "PARTIAL",
    "priority": 5
  }'

# Check status
curl -X GET http://localhost:8000/v1/jobs/{job_id} \
  -H "X-API-Key: your-api-key"
```

### Health Check

```bash
curl http://localhost:8000/v1/health
```

## Error Codes

| Code | Description |
|------|-------------|
| GLNORM-001 | Invalid input value |
| GLNORM-002 | Unknown unit |
| GLNORM-003 | Incompatible unit conversion |
| GLNORM-004 | Batch size exceeded |
| GLNORM-005 | Job not found |
| GLNORM-006 | Vocabulary not found |
| GLNORM-007 | Authentication failed |
| GLNORM-008 | Rate limit exceeded |
| GLNORM-009 | Internal processing error |
| GLNORM-010 | Validation failed |

## Docker

```bash
# Build image
docker build -t gl-normalizer-service .

# Run container
docker run -p 8000:8000 \
  -e GL_NORMALIZER_SECRET_KEY=your-secret-key-32-chars-min \
  gl-normalizer-service
```

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-normalizer-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-normalizer-service
  template:
    metadata:
      labels:
        app: gl-normalizer-service
    spec:
      containers:
      - name: service
        image: gl-normalizer-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: GL_NORMALIZER_ENV
          value: production
        - name: GL_NORMALIZER_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: normalizer-secrets
              key: jwt-secret
        livenessProbe:
          httpGet:
            path: /v1/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /v1/ready
            port: 8000
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/
```

## License

Copyright (c) 2024-2026 GreenLang. All rights reserved.
