# Resilience Patterns - Quick Reference Card

**Team 2 - Resilience Patterns | GL-VCCI Scope 3 Platform**

---

## 🚀 Quick Start

```python
from greenlang.resilience import retry, timeout, fallback, CircuitBreaker
from services.resilience import degradation_handler, DegradationTier

# Full resilience stack
@retry(max_retries=3, base_delay=1.0)
@timeout(timeout_seconds=10.0)
@fallback(strategy=FallbackStrategy.CACHED)
@degradation_handler(min_tier=DegradationTier.TIER_2_CORE)
def resilient_api_call():
    return external_api.fetch_data()
```

---

## 📦 Import Cheat Sheet

```python
# Retry
from greenlang.resilience import retry, async_retry, RetryStrategy

# Timeout
from greenlang.resilience import timeout, async_timeout, OperationType

# Fallback
from greenlang.resilience import fallback, FallbackStrategy, get_cached_fallback

# Rate Limiting
from greenlang.resilience import RateLimiter, RateLimitConfig, get_rate_limiter

# Circuit Breaker
from greenlang.resilience import CircuitBreaker, CircuitBreakerConfig

# Graceful Degradation
from services.resilience import (
    get_degradation_manager,
    degradation_handler,
    DegradationTier
)
```

---

## ⚡ Common Patterns

### Pattern 1: External API Call
```python
@retry(max_retries=3, base_delay=1.0)
@timeout(operation_type=OperationType.EXTERNAL_API)
@fallback(strategy=FallbackStrategy.CACHED)
def fetch_data():
    return api.get("/data")
```

### Pattern 2: Factor Lookup
```python
@retry(max_retries=3)
@timeout(operation_type=OperationType.FACTOR_LOOKUP)
@fallback(strategy=FallbackStrategy.CACHED)
def get_factor(category: str):
    return factor_api.get(category)
```

### Pattern 3: LLM Inference
```python
@retry(max_retries=2, base_delay=2.0)
@timeout(operation_type=OperationType.LLM_INFERENCE)
@fallback(strategy=FallbackStrategy.CACHED)
@degradation_handler(min_tier=DegradationTier.TIER_1_FULL)
def generate_text(prompt: str):
    return llm_api.generate(prompt)
```

### Pattern 4: Database Query
```python
@retry(max_retries=3, base_delay=1.0)
@timeout(operation_type=OperationType.DATABASE_QUERY)
def query_db(sql: str):
    return db.execute(sql)
```

### Pattern 5: ERP Integration
```python
@retry(max_retries=5, base_delay=2.0)
@timeout(operation_type=OperationType.ERP_API_CALL)
@fallback(strategy=FallbackStrategy.CACHED)
def fetch_erp_data(supplier_id: str):
    limiter.check_limit("erp_api")
    return erp_api.fetch(supplier_id)
```

---

## ⏱️ Timeout Values

| Operation | Timeout | Config |
|-----------|---------|--------|
| Cache | 2s | `OperationType.CACHE_OPERATION` |
| Factor Lookup | 5s | `OperationType.FACTOR_LOOKUP` |
| Database | 10s | `OperationType.DATABASE_QUERY` |
| ERP API | 10s | `OperationType.ERP_API_CALL` |
| External API | 15s | `OperationType.EXTERNAL_API` |
| Computation | 20s | `OperationType.COMPUTATION` |
| File Upload | 30s | `OperationType.FILE_UPLOAD` |
| LLM Inference | 30s | `OperationType.LLM_INFERENCE` |
| Report Gen | 60s | `OperationType.REPORT_GENERATION` |

---

## 🔄 Retry Strategies

```python
from greenlang.resilience import RetryStrategy

# Exponential: 1s, 2s, 4s, 8s... (RECOMMENDED)
@retry(strategy=RetryStrategy.EXPONENTIAL)

# Linear: 1s, 2s, 3s, 4s...
@retry(strategy=RetryStrategy.LINEAR)

# Constant: 1s, 1s, 1s, 1s...
@retry(strategy=RetryStrategy.CONSTANT)

# Fibonacci: 1s, 1s, 2s, 3s, 5s...
@retry(strategy=RetryStrategy.FIBONACCI)
```

---

## 🛡️ Fallback Strategies

```python
from greenlang.resilience import FallbackStrategy

# Return cached value
@fallback(strategy=FallbackStrategy.CACHED)

# Return default value
@fallback(strategy=FallbackStrategy.DEFAULT, default_value=[])

# Call fallback function
@fallback(
    strategy=FallbackStrategy.FUNCTION,
    fallback_function=get_backup_data
)

# Return None
@fallback(strategy=FallbackStrategy.NONE)

# Re-raise exception
@fallback(strategy=FallbackStrategy.RAISE)
```

---

## 🚦 Rate Limiting

```python
from greenlang.resilience import get_rate_limiter, RateLimitConfig

# Configure rate limit
limiter = get_rate_limiter()
limiter.configure(
    "api_name",
    RateLimitConfig(
        requests_per_second=10.0,
        burst_size=20
    )
)

# Check before calling
limiter.check_limit("api_name")
result = api.call()

# Or with wait
limiter.check_limit("api_name", wait=True)
```

---

## ⚡ Circuit Breaker

```python
from greenlang.resilience import CircuitBreaker, CircuitBreakerConfig

# Create circuit breaker
cb = CircuitBreaker(
    CircuitBreakerConfig(
        name="api_name",
        fail_max=5,
        timeout_duration=60
    )
)

# Use as context manager
with cb:
    result = risky_operation()

# Or as decorator
@cb.protect
def risky_operation():
    return external_api.call()

# Check state
if cb.state == CircuitState.OPEN:
    use_fallback()
```

---

## 📊 Degradation Tiers

| Tier | Name | LLM | APIs | Writes | Complex Calc |
|------|------|-----|------|--------|--------------|
| 1 | Full | ✅ | ✅ | ✅ | ✅ |
| 2 | Core | ❌ | ✅ | ✅ | ✅ |
| 3 | Read-Only | ❌ | ❌ | ❌ | ❌ |
| 4 | Maintenance | ❌ | ❌ | ❌ | ❌ |

```python
from services.resilience import get_degradation_manager, DegradationTier

# Update service health
manager = get_degradation_manager()
manager.update_health("factor_api", healthy=True, response_time_ms=150)
manager.update_health("llm_api", healthy=False, error="Timeout")

# Check current tier
tier = manager.get_current_tier()

# Protect operations
@degradation_handler(min_tier=DegradationTier.TIER_2_CORE)
def requires_core():
    return operation()
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_resilience_patterns.py -v

# Run specific category
pytest tests/test_resilience_patterns.py::TestRetry -v

# With coverage
pytest tests/test_resilience_patterns.py --cov
```

---

## 📈 Monitoring

### Metrics
```python
# Circuit breaker stats
from greenlang.resilience import get_circuit_breaker_stats
stats = get_circuit_breaker_stats("api_name")

# Rate limiter stats
stats = limiter.get_stats("api_name")

# Degradation stats
stats = manager.get_stats()
```

### Prometheus Metrics
- `greenlang_circuit_breaker_state`
- `greenlang_circuit_breaker_calls_total`
- `greenlang_circuit_breaker_failure_rate`

---

## ⚙️ Configuration

### Development
```python
retry: max_retries=2, base_delay=0.5s
timeout: 2x production
rate_limit: 100 req/s
circuit_breaker: fail_max=10
```

### Production
```python
retry: max_retries=3, base_delay=1.0s, jitter=True
timeout: strict (see table above)
rate_limit: 10 req/s
circuit_breaker: fail_max=5, timeout=60s
```

---

## 🆘 Troubleshooting

### MaxRetriesExceeded
```python
# Increase retries or fix root cause
@retry(max_retries=5)  # More retries
@retry(retryable_exceptions=(ConnectionError,))  # Specific exceptions
```

### TimeoutError
```python
# Increase timeout or optimize operation
@timeout(timeout_seconds=30.0)  # Longer timeout
```

### RateLimitExceeded
```python
# Increase rate limit or add wait
config = RateLimitConfig(
    requests_per_second=20.0,  # Higher limit
    wait_on_limit=True  # Wait instead of fail
)
```

### CircuitBreakerOpen
```python
# Check service health and wait for recovery
if cb.state == CircuitState.OPEN:
    use_fallback()
```

---

## 📚 Resources

- **Full Documentation**: `services/resilience/README.md`
- **Integration Examples**: `services/resilience/integration_examples.py`
- **Configuration Guide**: `services/resilience/config_recommendations.py`
- **Test Suite**: `tests/test_resilience_patterns.py`
- **Delivery Report**: `RESILIENCE_PATTERNS_DELIVERY_REPORT.md`

---

**Version**: 1.0.0 | **Status**: Production Ready | **Team**: Team 2 - Resilience Patterns
