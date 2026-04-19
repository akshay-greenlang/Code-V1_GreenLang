# GreenLang ML Platform

Production-grade LLM model API, evaluation harness, and intelligent routing for zero-hallucination applications.

## Components

### 1. Model API (`model_api.py`)

FastAPI-based REST API for model invocation and evaluation.

**Features:**
- JWT authentication with bearer tokens
- Rate limiting (100 requests/60s per user)
- Provenance tracking (SHA-256 hashes)
- Performance metrics (latency, tokens, cost)
- Model evaluation endpoints
- Health check and monitoring

**Endpoints:**

```
POST /v1/auth/token              - Get JWT access token
POST /v1/models/invoke           - Invoke a model
POST /v1/models/evaluate         - Evaluate model response
GET  /v1/models/{id}/metrics     - Get model metrics
GET  /v1/models                  - List available models
GET  /health                     - Health check
```

**Example Usage:**

```python
from greenlang.ml_platform.model_api import app, ModelInvokeRequest
import uvicorn

# Start API server
uvicorn.run(app, host="0.0.0.0", port=8000)

# In another script, make requests:
import requests

# Get token
response = requests.post(
    "http://localhost:8000/v1/auth/token",
    headers={"api-key": "gl_your_api_key"}
)
token = response.json()["access_token"]

# Invoke model
response = requests.post(
    "http://localhost:8000/v1/models/invoke",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "model_id": "claude-sonnet-4",
        "prompt": "What is carbon footprint?",
        "temperature": 0.0
    }
)

result = response.json()
print(f"Response: {result['response_text']}")
print(f"Latency: {result['latency_ms']}ms")
print(f"Cost: ${result['cost_usd']}")
```

### 2. Evaluation Harness (`evaluation.py`)

Golden test execution and determinism validation.

**Features:**
- Golden test execution (expected vs actual)
- Determinism validation (bit-perfect reproducibility)
- Performance metrics collection (latency, tokens, cost)
- Comprehensive evaluation reports
- P50/P95/P99 latency tracking

**Classes:**

- `GoldenTestExecutor` - Runs golden tests against models
- `DeterminismValidator` - Validates bit-perfect reproducibility
- `MetricsCollector` - Collects and aggregates metrics
- `EvaluationReport` - Complete test results with metrics

**Example Usage:**

```python
from greenlang.ml_platform.evaluation import (
    GoldenTest,
    GoldenTestExecutor,
    DeterminismValidator
)

# Create golden test suite
tests = [
    GoldenTest(
        id="test1",
        name="Basic calculation",
        prompt="Calculate 2+2",
        expected_response="4",
        temperature=0.0
    ),
    GoldenTest(
        id="test2",
        name="Capital query",
        prompt="What is the capital of France?",
        expected_response="Paris",
        temperature=0.0
    )
]

# Run tests
executor = GoldenTestExecutor()
report = await executor.run_golden_tests(
    model_id="claude-sonnet-4",
    tests=tests,
    test_suite_name="basic_tests",
    check_determinism=True
)

print(f"Pass rate: {report.pass_rate*100:.1f}%")
print(f"Avg latency: {report.performance_metrics.avg_latency_ms:.2f}ms")
print(f"Total cost: ${report.performance_metrics.total_cost_usd:.6f}")

# Check determinism
validator = DeterminismValidator()
result = await validator.validate_determinism(
    model_id="claude-sonnet-4",
    prompt="What is 2+2?",
    runs=5,
    temperature=0.0
)

print(f"All identical: {result.all_identical}")
print(f"Unique responses: {result.unique_responses}")
```

### 3. Model Router (`router.py`)

Intelligent routing with cost optimization and fallback logic.

**Features:**
- Smart model selection based on criteria
- Multiple routing strategies (lowest cost, lowest latency, balanced)
- Automatic fallback on failure
- Cost optimization heuristics
- Load balancing across models
- Routing statistics and monitoring

**Classes:**

- `ModelRouter` - Main routing engine
- `RoutingCriteria` - Model selection criteria
- `CostOptimizer` - Cost optimization heuristics
- `LoadBalancer` - Load balancing across models

**Example Usage:**

```python
from greenlang.ml_platform.router import (
    ModelRouter,
    RoutingCriteria,
    RoutingStrategy
)
from greenlang.registry.model_registry import ModelCapability

# Create router
router = ModelRouter()

# Define routing criteria
criteria = RoutingCriteria(
    capability=ModelCapability.TEXT_GENERATION,
    max_cost_per_1k_tokens=0.005,
    certified_only=True,
    strategy=RoutingStrategy.LOWEST_COST,
    enable_fallback=True,
    max_fallback_attempts=3
)

# Select best model
model = router.select_model(criteria)
print(f"Selected: {model.name} (${model.avg_cost_per_1k_tokens}/1k tokens)")

# Invoke with automatic fallback
result = await router.invoke_with_fallback(
    criteria=criteria,
    prompt="What is carbon footprint?",
    temperature=0.0
)

print(f"Response: {result.response}")
print(f"Model used: {result.model_id}")
print(f"Fallback used: {result.fallback_used}")
print(f"Attempts: {result.attempts}")
```

## Routing Strategies

### LOWEST_COST
Selects cheapest model that meets requirements.

```python
criteria = RoutingCriteria(
    capability=ModelCapability.TEXT_GENERATION,
    strategy=RoutingStrategy.LOWEST_COST
)
```

### LOWEST_LATENCY
Selects fastest model that meets requirements.

```python
criteria = RoutingCriteria(
    capability=ModelCapability.CODE_GENERATION,
    strategy=RoutingStrategy.LOWEST_LATENCY
)
```

### HIGHEST_QUALITY
Prefers certified models, then lowest cost.

```python
criteria = RoutingCriteria(
    capability=ModelCapability.FUNCTION_CALLING,
    strategy=RoutingStrategy.HIGHEST_QUALITY
)
```

### BALANCED
Balances cost and quality (50/50 weighting).

```python
criteria = RoutingCriteria(
    capability=ModelCapability.JSON_MODE,
    strategy=RoutingStrategy.BALANCED
)
```

## Zero-Hallucination Guarantees

### Temperature = 0.0
Always use `temperature=0.0` for deterministic outputs:

```python
request = ModelInvokeRequest(
    model_id="claude-sonnet-4",
    prompt="Calculate emissions",
    temperature=0.0  # REQUIRED for zero-hallucination
)
```

### Certified Models Only
Filter for zero-hallucination certified models:

```python
criteria = RoutingCriteria(
    capability=ModelCapability.TEXT_GENERATION,
    certified_only=True  # Only certified models
)
```

### Determinism Validation
Validate bit-perfect reproducibility:

```python
validator = DeterminismValidator()
result = await validator.validate_determinism(
    model_id="claude-sonnet-4",
    prompt="Calculate 2+2",
    runs=5,
    temperature=0.0
)

assert result.all_identical, "Non-deterministic output detected!"
```

## Cost Optimization

### Automatic Cost Optimization

```python
from greenlang.ml_platform.router import CostOptimizer

optimizer = CostOptimizer()

# Optimize criteria based on prompt complexity
optimized_criteria = optimizer.optimize_routing_criteria(
    base_criteria=criteria,
    prompt="Simple query"
)

# Will use cheaper model for simple prompts
```

### Cost Constraints

```python
criteria = RoutingCriteria(
    capability=ModelCapability.TEXT_GENERATION,
    max_cost_per_1k_tokens=0.003,  # Max $0.003 per 1k tokens
    strategy=RoutingStrategy.LOWEST_COST
)
```

## Performance Metrics

### Latency Tracking

```python
# Metrics automatically tracked in reports
print(f"Avg latency: {report.performance_metrics.avg_latency_ms:.2f}ms")
print(f"P50 latency: {report.performance_metrics.p50_latency_ms:.2f}ms")
print(f"P95 latency: {report.performance_metrics.p95_latency_ms:.2f}ms")
print(f"P99 latency: {report.performance_metrics.p99_latency_ms:.2f}ms")
```

### Cost Tracking

```python
print(f"Total tokens: {report.performance_metrics.total_tokens}")
print(f"Total cost: ${report.performance_metrics.total_cost_usd:.6f}")
```

### Success Rate

```python
collector = MetricsCollector()
metrics = collector.get_aggregated_metrics("claude-sonnet-4")
print(f"Success rate: {metrics['success_rate']*100:.1f}%")
```

## Running the API

### Start API Server

```bash
# Development mode
cd core/greenlang/ml_platform
python -m uvicorn model_api:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn model_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Get token
curl -X POST http://localhost:8000/v1/auth/token \
  -H "api-key: gl_your_key"

# Invoke model
curl -X POST http://localhost:8000/v1/models/invoke \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "claude-sonnet-4",
    "prompt": "What is carbon footprint?",
    "temperature": 0.0
  }'
```

## Running Tests

```bash
# Run all tests
cd core/greenlang/ml_platform
python test_ml_platform.py

# Run with pytest
pytest test_ml_platform.py -v

# Run specific test
pytest test_ml_platform.py::test_golden_test_execution -v
```

## Architecture

```
ml_platform/
├── __init__.py           # Module exports
├── model_api.py          # FastAPI application (authentication, endpoints)
├── evaluation.py         # Golden tests, determinism validation, metrics
├── router.py             # Model routing, cost optimization, load balancing
├── test_ml_platform.py   # Comprehensive test suite
└── README.md             # This file
```

## Integration with Model Registry

All components integrate with the Model Registry:

```python
from greenlang.registry.model_registry import model_registry

# List certified models
models = model_registry.list_models(certified_only=True)

# Get model metrics
model = model_registry.get_model("claude-sonnet-4")
print(f"Total requests: {model.total_requests}")
print(f"Total cost: ${model.total_cost_usd:.2f}")

# Update metrics
model_registry.update_metrics(
    model_id="claude-sonnet-4",
    requests=1,
    tokens=500,
    cost_usd=0.0015,
    latency_ms=250
)
```

## Production Checklist

- [ ] Replace JWT secret with secure value
- [ ] Implement Redis-based rate limiting
- [ ] Replace mock model invocations with actual SDK calls
- [ ] Configure CORS origins appropriately
- [ ] Set up PostgreSQL for model registry
- [ ] Implement API key validation against database
- [ ] Add comprehensive logging
- [ ] Set up metrics dashboards (Prometheus/Grafana)
- [ ] Configure SSL/TLS certificates
- [ ] Implement request tracing (OpenTelemetry)

## Next Steps

1. **Replace Mock Invocations**: Integrate actual Anthropic/OpenAI SDKs
2. **Database Integration**: Move from in-memory to PostgreSQL
3. **Caching**: Add Redis for rate limiting and response caching
4. **Monitoring**: Set up Prometheus metrics and Grafana dashboards
5. **CI/CD**: Add automated testing and deployment pipelines
6. **Documentation**: Generate OpenAPI docs and API client libraries
7. **Security**: Implement API key rotation and audit logging

## License

Copyright 2025 GreenLang. All rights reserved.
