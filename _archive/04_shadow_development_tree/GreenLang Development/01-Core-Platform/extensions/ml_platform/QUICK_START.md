# GreenLang ML Platform - Quick Start Guide

Get up and running with the ML Platform in 5 minutes.

---

## 1. Start the API Server (30 seconds)

```bash
cd C:\Users\aksha\Code-V1_GreenLang\core

# Start API server
python -m uvicorn greenlang.ml_platform.model_api:app --reload --port 8000
```

Visit: http://localhost:8000/docs (Interactive API documentation)

---

## 2. Make Your First API Call (1 minute)

```python
import requests

# Get authentication token
response = requests.post(
    "http://localhost:8000/v1/auth/token",
    headers={"api-key": "gl_test_key"}
)
token = response.json()["access_token"]

# Invoke a model
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
print(f"Cost: ${result['cost_usd']:.6f}")
print(f"Latency: {result['latency_ms']:.2f}ms")
```

---

## 3. Run Golden Tests (1 minute)

```python
import asyncio
from greenlang.ml_platform.evaluation import GoldenTest, GoldenTestExecutor

# Create test
test = GoldenTest(
    id="test1",
    name="Basic test",
    prompt="What is 2+2?",
    expected_response="4",
    temperature=0.0
)

# Run test
async def run_test():
    executor = GoldenTestExecutor()
    report = await executor.run_golden_tests(
        model_id="claude-sonnet-4",
        tests=[test],
        test_suite_name="quick_test"
    )
    print(f"Pass rate: {report.pass_rate*100:.1f}%")
    return report

# Execute
report = asyncio.run(run_test())
```

---

## 4. Smart Model Routing (1 minute)

```python
from greenlang.ml_platform.router import ModelRouter, RoutingCriteria, RoutingStrategy
from greenlang.registry.model_registry import ModelCapability

# Create router
router = ModelRouter()

# Define criteria
criteria = RoutingCriteria(
    capability=ModelCapability.TEXT_GENERATION,
    max_cost_per_1k_tokens=0.005,
    certified_only=True,
    strategy=RoutingStrategy.LOWEST_COST
)

# Select best model
model = router.select_model(criteria)
print(f"Selected: {model.name}")
print(f"Cost: ${model.avg_cost_per_1k_tokens:.6f}/1k tokens")
```

---

## 5. Validate Determinism (1 minute)

```python
import asyncio
from greenlang.ml_platform.evaluation import DeterminismValidator

async def check_determinism():
    validator = DeterminismValidator()
    result = await validator.validate_determinism(
        model_id="claude-sonnet-4",
        prompt="Calculate: 100 * 0.5",
        runs=5,
        temperature=0.0
    )
    print(f"All identical: {result.all_identical}")
    print(f"Unique responses: {result.unique_responses}")
    return result

# Execute
result = asyncio.run(check_determinism())
```

---

## 6. Run Complete Example Suite (30 seconds)

```bash
cd C:\Users\aksha\Code-V1_GreenLang\core

# Run all examples
python -m greenlang.ml_platform.examples
```

Output shows 9 complete examples with results.

---

## 7. Run Test Suite (30 seconds)

```bash
cd C:\Users\aksha\Code-V1_GreenLang\core

# Run all tests
python -m greenlang.ml_platform.test_ml_platform

# Or with pytest
pytest greenlang/ml_platform/test_ml_platform.py -v
```

---

## Common Tasks

### List Available Models

```python
from greenlang.registry.model_registry import model_registry

# List all models
models = model_registry.list_models()
for model in models:
    print(f"{model.name}: ${model.avg_cost_per_1k_tokens:.6f}/1k tokens")

# List certified models only
certified = model_registry.list_models(certified_only=True)
print(f"Certified models: {len(certified)}")
```

### Get Model Metrics

```python
# Via registry
model = model_registry.get_model("claude-sonnet-4")
print(f"Total requests: {model.total_requests}")
print(f"Total cost: ${model.total_cost_usd:.2f}")

# Via API
response = requests.get(
    "http://localhost:8000/v1/models/claude-sonnet-4/metrics",
    headers={"Authorization": f"Bearer {token}"}
)
metrics = response.json()
print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
```

### Invoke with Fallback

```python
import asyncio
from greenlang.ml_platform.router import ModelRouter, RoutingCriteria

async def invoke_with_fallback():
    router = ModelRouter()
    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        enable_fallback=True,
        max_fallback_attempts=3
    )

    result = await router.invoke_with_fallback(
        criteria=criteria,
        prompt="Your prompt here",
        temperature=0.0
    )

    print(f"Model: {result.model_id}")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Response: {result.response}")

asyncio.run(invoke_with_fallback())
```

### Collect Performance Metrics

```python
from greenlang.ml_platform.evaluation import MetricsCollector

collector = MetricsCollector()

# Record invocations (e.g., in a loop)
collector.record_invocation(
    model_id="claude-sonnet-4",
    latency_ms=150.5,
    input_tokens=50,
    output_tokens=100,
    cost_usd=0.00045,
    success=True
)

# Get aggregated metrics
metrics = collector.get_aggregated_metrics("claude-sonnet-4")
print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Success rate: {metrics['success_rate']*100:.1f}%")
```

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/auth/token` | POST | Get JWT token |
| `/v1/models/invoke` | POST | Invoke model |
| `/v1/models/evaluate` | POST | Evaluate response |
| `/v1/models/{id}/metrics` | GET | Get model metrics |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |

---

## File Locations

| Component | Path |
|-----------|------|
| Model API | `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\model_api.py` |
| Evaluation | `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\evaluation.py` |
| Router | `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\router.py` |
| Tests | `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\test_ml_platform.py` |
| Examples | `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\examples.py` |
| README | `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\README.md` |

---

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the core directory
cd C:\Users\aksha\Code-V1_GreenLang\core

# Run with module syntax
python -m greenlang.ml_platform.examples
```

### API Not Starting

```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use different port
uvicorn greenlang.ml_platform.model_api:app --port 8001
```

### Authentication Errors

- Make sure to get token first: `POST /v1/auth/token`
- Include token in header: `Authorization: Bearer YOUR_TOKEN`
- Tokens expire after 24 hours

---

## Next Steps

1. Read full README: `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\README.md`
2. Review examples: `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\ml_platform\examples.py`
3. Run tests: `python -m greenlang.ml_platform.test_ml_platform`
4. Explore API docs: http://localhost:8000/docs

---

**You're ready to go!** Start building with the GreenLang ML Platform.
