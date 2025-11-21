# GreenLang Health Check API

Production-ready health check endpoints for Kubernetes orchestration of the GreenLang Agent Foundation API.

## Overview

This module implements three critical health check endpoints required for Kubernetes deployment:

1. **Liveness Probe** (`/healthz`) - Fast process alive check
2. **Readiness Probe** (`/ready`) - Comprehensive dependency health check
3. **Startup Probe** (`/startup`) - One-time initialization verification

## Quick Start

### Running the API

```bash
# Development mode
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation
python -m api.main

# Production mode with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With auto-reload (development)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing Health Endpoints

```bash
# Liveness probe
curl http://localhost:8000/healthz

# Readiness probe
curl http://localhost:8000/ready

# Startup probe
curl http://localhost:8000/startup

# API info
curl http://localhost:8000/api/v1/info
```

## Health Check Endpoints

### 1. Liveness Probe - `/healthz`

**Purpose**: Kubernetes uses this to determine if the process is alive and should be restarted.

**Characteristics**:
- **Response Time**: <10ms (fast check)
- **Dependencies**: None (no external calls)
- **Check**: Process is responding
- **Use Case**: Detect deadlocks, infinite loops, unrecoverable errors

**Response Example**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-14T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.0,
  "components": [
    {
      "name": "process",
      "status": "healthy",
      "message": "Process is alive",
      "response_time_ms": 0.1,
      "last_checked": "2025-11-14T10:30:00Z"
    }
  ]
}
```

**Kubernetes Configuration**:
```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 1
  failureThreshold: 3  # Restart after 15s of failures
```

### 2. Readiness Probe - `/ready`

**Purpose**: Kubernetes uses this to determine if the pod should receive traffic.

**Characteristics**:
- **Response Time**: <1 second (comprehensive check)
- **Dependencies**: All critical services (DB, Redis, LLM, Vector DB)
- **Caching**: 5-second TTL (avoid hammering dependencies)
- **Use Case**: Prevent routing traffic to unhealthy pods

**Components Checked**:
- PostgreSQL database connection
- Redis cache connection
- LLM provider availability
- Vector database accessibility

**Response Example (Healthy)**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-14T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.0,
  "components": [
    {
      "name": "postgresql",
      "status": "healthy",
      "message": "Connected to primary database",
      "response_time_ms": 12.5,
      "last_checked": "2025-11-14T10:30:00Z",
      "metadata": {
        "pool_available": true,
        "latency_acceptable": true
      }
    },
    {
      "name": "redis",
      "status": "healthy",
      "message": "Redis responding to PING",
      "response_time_ms": 5.2,
      "last_checked": "2025-11-14T10:30:00Z"
    },
    {
      "name": "llm_providers",
      "status": "healthy",
      "message": "LLM providers available",
      "response_time_ms": 8.1,
      "last_checked": "2025-11-14T10:30:00Z"
    },
    {
      "name": "vector_db",
      "status": "healthy",
      "message": "Vector store accessible",
      "response_time_ms": 15.3,
      "last_checked": "2025-11-14T10:30:00Z"
    }
  ]
}
```

**Response Example (Unhealthy)**:
```json
{
  "status": "unhealthy",
  "timestamp": "2025-11-14T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.0,
  "components": [
    {
      "name": "postgresql",
      "status": "unhealthy",
      "message": "Connection refused",
      "response_time_ms": 1000.0,
      "last_checked": "2025-11-14T10:30:00Z"
    }
  ]
}
```

**HTTP Status Codes**:
- `200 OK` - All components healthy, ready to serve traffic
- `503 Service Unavailable` - One or more components unhealthy

**Kubernetes Configuration**:
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3  # Remove from service after 30s
```

### 3. Startup Probe - `/startup`

**Purpose**: Kubernetes uses this for slow-starting containers before enabling liveness/readiness checks.

**Characteristics**:
- **Response Time**: Variable (can take 30-60s for full initialization)
- **Dependencies**: All components (fresh checks, no caching)
- **One-time**: Only used during container startup
- **Use Case**: Prevent premature restarts during initialization

**Response Example**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-14T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 45.0,
  "components": [
    {
      "name": "startup",
      "status": "healthy",
      "message": "Startup complete",
      "response_time_ms": 0.0,
      "last_checked": "2025-11-14T10:30:00Z",
      "metadata": {
        "uptime_seconds": 45.0
      }
    },
    {
      "name": "postgresql",
      "status": "healthy",
      "message": "Connected to primary database",
      "response_time_ms": 12.5,
      "last_checked": "2025-11-14T10:30:00Z"
    }
  ]
}
```

**HTTP Status Codes**:
- `200 OK` - Startup complete, all components initialized
- `503 Service Unavailable` - Startup in progress or failed

**Kubernetes Configuration**:
```yaml
startupProbe:
  httpGet:
    path: /startup
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 30  # Allow up to 150s for startup
```

## Architecture

### Health Check Flow

```
Kubernetes → Health Endpoint → HealthCheckManager → Component Checks
                                                           ↓
                                    Database, Redis, LLM, Vector DB
```

### Component Health States

- **HEALTHY** - Component fully operational
- **DEGRADED** - Component working but with issues (e.g., high latency)
- **UNHEALTHY** - Component not working
- **UNKNOWN** - Component not initialized or check failed

### Caching Strategy

**Readiness Checks**:
- Cache TTL: 5 seconds
- Avoids hammering dependencies with frequent checks
- Balances freshness with performance

**Startup Checks**:
- No caching (fresh checks every time)
- Ensures accurate initialization verification

**Liveness Checks**:
- No caching (instant response)
- No external dependencies

## Integration with Dependencies

### Registering Dependencies

```python
from api.health import health_manager
from database.postgres_manager import PostgresManager
from cache.redis_manager import RedisManager
from llm.llm_router import LLMRouter
from rag.vector_store import VectorStore

# Initialize dependencies
db_manager = PostgresManager(config)
await db_manager.initialize()

redis_manager = RedisManager(config)
await redis_manager.initialize()

llm_router = LLMRouter(strategy="least_cost")
vector_store = VectorStore()

# Register with health manager
health_manager.set_dependencies(
    db_manager=db_manager,
    redis_manager=redis_manager,
    llm_router=llm_router,
    vector_store=vector_store
)

# Mark startup complete
health_manager.mark_startup_complete()
```

## Testing

### Running Tests

```bash
# Run all health check tests
pytest api/test_health_api.py -v

# Run specific test class
pytest api/test_health_api.py::TestLivenessProbe -v

# Run with coverage
pytest api/test_health_api.py --cov=api.health --cov-report=html
```

### Test Coverage

The test suite covers:
- Response format validation
- HTTP status codes
- Component health reporting
- Caching behavior
- Performance requirements (<10ms liveness, <1s readiness)
- Security headers
- Request ID tracking
- Error handling

## Kubernetes Deployment

### Complete Example

See `kubernetes_health_checks.yaml` for a complete Kubernetes deployment configuration with:

- **Deployment** with 3 replicas for high availability
- **Service** for load balancing
- **HorizontalPodAutoscaler** for auto-scaling (3-10 pods)
- **PodDisruptionBudget** to maintain availability during updates
- **ServiceMonitor** for Prometheus metrics

### Deployment Commands

```bash
# Apply configuration
kubectl apply -f api/kubernetes_health_checks.yaml

# Check deployment status
kubectl get deployment greenlang-api -n greenlang-production

# Check pod health
kubectl get pods -n greenlang-production -l app=greenlang-api

# View pod logs
kubectl logs -n greenlang-production -l app=greenlang-api

# Check health endpoints
kubectl port-forward -n greenlang-production svc/greenlang-api-service 8000:80
curl http://localhost:8000/healthz
```

## Monitoring and Observability

### Metrics

Health check endpoints expose metrics for monitoring:

- **Health check success/failure counts**
- **Component response times**
- **Cache hit/miss ratios**
- **Request/response timing**

### Logging

Structured logs include:

- **Request ID** for distributed tracing
- **Response time** for performance monitoring
- **Component status** for debugging
- **Error details** for troubleshooting

### Alerting

Recommended alerts:

```yaml
# Liveness failures
- alert: HighLivenessFailureRate
  expr: rate(health_liveness_failures[5m]) > 0.1
  annotations:
    summary: "High liveness check failure rate"

# Readiness failures
- alert: HighReadinessFailureRate
  expr: rate(health_readiness_failures[5m]) > 0.2
  annotations:
    summary: "High readiness check failure rate"

# Component unhealthy
- alert: ComponentUnhealthy
  expr: health_component_status{status="unhealthy"} == 1
  for: 5m
  annotations:
    summary: "Component {{ $labels.component }} is unhealthy"
```

## Performance Characteristics

### Liveness Probe

- **Target**: <10ms
- **Actual**: ~0.1ms (instant response)
- **No network calls**

### Readiness Probe

- **Target**: <1000ms
- **Actual**: ~50ms (with caching)
- **Components checked in parallel**

### Startup Probe

- **Target**: <60s
- **Actual**: Variable (depends on dependency initialization)
- **Fresh checks (no caching)**

## Security Features

### Security Headers

All responses include:

- `Strict-Transport-Security` (HSTS)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy: default-src 'self'`

### CORS Configuration

Restricted to specific domains:

- `https://*.greenlang.io`
- `http://localhost:3000` (development only)

### Request ID Tracking

Every request gets a unique ID for distributed tracing:

- Generated automatically or from `X-Request-ID` header
- Included in all logs
- Returned in response headers

## Troubleshooting

### Liveness Probe Failing

**Symptom**: Pods restarting frequently

**Possible Causes**:
- Process deadlock
- Out of memory
- Infinite loop

**Solution**:
1. Check pod logs: `kubectl logs <pod-name>`
2. Increase memory limits if OOM
3. Review application code for deadlocks

### Readiness Probe Failing

**Symptom**: Pods not receiving traffic

**Possible Causes**:
- Database connection issues
- Redis connection issues
- LLM provider unavailable
- Vector DB not accessible

**Solution**:
1. Check component status: `curl http://localhost:8000/ready`
2. Verify database connectivity
3. Check Redis connectivity
4. Verify LLM API keys
5. Check network policies

### Startup Probe Failing

**Symptom**: Pods never become ready

**Possible Causes**:
- Slow dependency initialization
- Configuration errors
- Missing environment variables

**Solution**:
1. Increase `failureThreshold` in startup probe
2. Check initialization logs
3. Verify all environment variables set
4. Check dependency availability

## API Documentation

### OpenAPI/Swagger

Interactive API documentation available at:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## Files

- **`health.py`** - Health check implementation with component checks
- **`main.py`** - FastAPI application with routes and middleware
- **`test_health_api.py`** - Comprehensive test suite
- **`kubernetes_health_checks.yaml`** - Kubernetes deployment configuration
- **`README.md`** - This documentation

## Dependencies

```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
asyncpg>=0.29.0
redis>=5.0.0
anthropic>=0.7.0
pytest>=7.4.0
httpx>=0.25.0
```

## License

Proprietary - GreenLang Inc. All rights reserved.

## Support

For support and questions:
- Email: support@greenlang.io
- Documentation: https://docs.greenlang.io
- Status Page: https://status.greenlang.io
