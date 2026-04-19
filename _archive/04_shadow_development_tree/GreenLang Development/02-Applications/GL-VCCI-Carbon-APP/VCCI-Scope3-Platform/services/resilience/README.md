# GL-VCCI Resilience Patterns

**Team 2: Resilience Patterns Team**
**Mission**: Implement retry logic, timeout configurations, graceful degradation, and fallback mechanisms for GL-VCCI Scope 3 Platform.

---

## Overview

This package provides production-grade resilience patterns for the GL-VCCI Scope 3 Platform, ensuring the system can handle transient failures, network issues, and API rate limits without crashing.

**Features**:
- **Retry Logic**: Exponential backoff with jitter
- **Timeout Management**: Operation-specific timeout configurations
- **Fallback Mechanisms**: Graceful degradation strategies
- **Rate Limiting**: Token bucket and leaky bucket algorithms
- **Circuit Breakers**: Prevent cascade failures
- **Graceful Degradation**: Tier-based functionality reduction

---

## Architecture

### Resilience Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│            (GL-VCCI Agents: Calculator, Intake, etc.)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Resilience Pattern Layer                    │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Retry   │  │ Timeout │  │ Fallback │  │  Circuit │     │
│  │ Logic   │  │ Handler │  │ Strategy │  │  Breaker │     │
│  └─────────┘  └─────────┘  └──────────┘  └──────────┘     │
│  ┌──────────────┐  ┌──────────────────────────────┐        │
│  │ Rate Limiter │  │ Graceful Degradation Manager │        │
│  └──────────────┘  └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│    (Factor APIs, LLM, ERP, Database, Cache)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Retry Logic (`greenlang.resilience.retry`)

**Features**:
- Exponential backoff: 1s, 2s, 4s, 8s...
- Configurable max retries (default: 3)
- Jitter to prevent thundering herd
- Specific exceptions to retry vs fail fast
- Support for sync and async functions

**Usage**:
```python
from greenlang.resilience import retry, RetryStrategy

@retry(max_retries=3, base_delay=1.0, strategy=RetryStrategy.EXPONENTIAL)
def fetch_emission_factor(category: str):
    return factor_api.get(category)
```

**Strategies**:
- `EXPONENTIAL`: 1s, 2s, 4s, 8s... (recommended)
- `LINEAR`: 1s, 2s, 3s, 4s...
- `CONSTANT`: 1s, 1s, 1s, 1s...
- `FIBONACCI`: 1s, 1s, 2s, 3s, 5s...

---

### 2. Timeout Management (`greenlang.resilience.timeout`)

**Features**:
- Operation-specific timeout values
- Support for sync and async functions
- Graceful timeout handling

**Timeout Values**:
- Factor lookup: **5 seconds**
- LLM inference: **30 seconds**
- ERP API calls: **10 seconds**
- Database queries: **10 seconds**
- Report generation: **60 seconds**

**Usage**:
```python
from greenlang.resilience import timeout, OperationType

@timeout(operation_type=OperationType.FACTOR_LOOKUP)
def lookup_factor(category: str):
    return factor_db.query(category)

@timeout(timeout_seconds=30.0)
def custom_timeout_operation():
    return long_running_task()
```

---

### 3. Fallback Mechanisms (`greenlang.resilience.fallback`)

**Features**:
- Fallback to cached data when API unavailable
- Fallback to lower-tier calculation when primary fails
- Default responses for non-critical failures
- Chained fallback strategies

**Strategies**:
- `CACHED`: Return cached value
- `DEFAULT`: Return default value
- `FUNCTION`: Call fallback function
- `NONE`: Return None
- `RAISE`: Re-raise exception

**Usage**:
```python
from greenlang.resilience import fallback, FallbackStrategy

@fallback(strategy=FallbackStrategy.CACHED)
def get_emission_factors(category: str):
    return external_api.fetch(category)

@fallback(strategy=FallbackStrategy.DEFAULT, default_value=[])
def get_suppliers():
    return supplier_api.fetch_all()
```

---

### 4. Rate Limiting (`greenlang.resilience.rate_limit_handler`)

**Features**:
- Token bucket algorithm for burst handling
- Leaky bucket algorithm for smooth rate limiting
- Handle 429 Too Many Requests
- Respect Retry-After headers
- Per-tenant and per-endpoint limiting

**Usage**:
```python
from greenlang.resilience import RateLimiter, RateLimitConfig

limiter = RateLimiter()
limiter.configure(
    "factor_api",
    RateLimitConfig(requests_per_second=10.0, burst_size=20)
)

# Check rate limit
limiter.check_limit("factor_api")
```

**Algorithms**:
- `TOKEN_BUCKET`: Allows bursts, refills over time
- `LEAKY_BUCKET`: Smooth output rate
- `FIXED_WINDOW`: Reset counter at intervals
- `SLIDING_WINDOW`: Rolling time window

---

### 5. Circuit Breaker (`greenlang.resilience.circuit_breaker`)

**Features**:
- Three states: CLOSED, OPEN, HALF_OPEN
- Automatic recovery attempts
- Configurable failure thresholds
- Prometheus metrics integration

**Usage**:
```python
from greenlang.resilience import CircuitBreaker, CircuitBreakerConfig

cb = CircuitBreaker(
    CircuitBreakerConfig(
        name="external_api",
        fail_max=5,
        timeout_duration=60
    )
)

with cb:
    result = external_api.call()
```

---

### 6. Graceful Degradation (`services.resilience.graceful_degradation`)

**Degradation Tiers**:

| Tier | Name | Features |
|------|------|----------|
| 1 | Full Functionality | All features enabled |
| 2 | Core Functionality | External APIs work, no LLM |
| 3 | Read-Only Mode | No writes, no external APIs |
| 4 | Maintenance Mode | Database issues, minimal functionality |

**Usage**:
```python
from services.resilience import (
    get_degradation_manager,
    degradation_handler,
    DegradationTier
)

# Update service health
manager = get_degradation_manager()
manager.update_health("factor_api", healthy=True, response_time_ms=150)
manager.update_health("llm_api", healthy=False, error="Timeout")

# Check current tier
tier = manager.get_current_tier()

# Protect operations
@degradation_handler(
    min_tier=DegradationTier.TIER_2_CORE,
    fallback_value=None
)
def requires_core_functionality():
    return perform_operation()
```

---

## Integration Examples

### Example 1: Calculator Agent with Full Resilience

```python
from greenlang.resilience import retry, timeout, fallback, CircuitBreaker
from services.resilience import degradation_handler, DegradationTier

class ResilientCalculatorAgent:
    def __init__(self):
        self.circuit = CircuitBreaker(
            CircuitBreakerConfig(name="factor_api", fail_max=5)
        )

    @retry(max_retries=3, base_delay=1.0)
    @timeout(operation_type=OperationType.EXTERNAL_API)
    @fallback(strategy=FallbackStrategy.CACHED)
    @degradation_handler(min_tier=DegradationTier.TIER_2_CORE)
    def get_emission_factor(self, category: str):
        return self.circuit.call(
            self._fetch_from_api,
            category
        )
```

### Example 2: LLM Provider with Timeout + Fallback

```python
@retry(max_retries=2, base_delay=2.0)
@timeout(operation_type=OperationType.LLM_INFERENCE)
@fallback(strategy=FallbackStrategy.CACHED)
@degradation_handler(
    min_tier=DegradationTier.TIER_1_FULL,
    fallback_value="[LLM unavailable]"
)
def generate_llm_response(prompt: str):
    return llm_api.generate(prompt)
```

### Example 3: ERP Connector with Rate Limiting

```python
from greenlang.resilience import get_rate_limiter

limiter = get_rate_limiter()
limiter.configure(
    "erp_api",
    RateLimitConfig(requests_per_second=5.0, burst_size=10)
)

@retry(max_retries=5, base_delay=2.0)
@timeout(operation_type=OperationType.ERP_API_CALL)
@fallback(strategy=FallbackStrategy.CACHED)
def fetch_procurement_data(supplier_id: str):
    limiter.check_limit("erp_api")
    return erp_api.fetch(supplier_id)
```

---

## Configuration Recommendations

### Development Environment
- Retry: max_retries=2, base_delay=0.5s
- Timeout: Relaxed (2x production)
- Rate limits: 100 req/s
- Circuit breaker: fail_max=10

### Production Environment
- Retry: max_retries=3, base_delay=1.0s, jitter=True
- Timeout: Strict (as specified)
- Rate limits: 10 req/s (factor API)
- Circuit breaker: fail_max=5, timeout=60s

### Per-Operation Recommendations

| Operation | Retry | Timeout | Fallback | Rate Limit |
|-----------|-------|---------|----------|------------|
| Factor Lookup | 3 retries | 5s | Cached | 10 req/s |
| LLM Inference | 2 retries | 30s | Cached | 5 req/s |
| ERP API | 5 retries | 10s | Cached | 5 req/s |
| Database | 3 retries | 10s | - | - |
| Report Gen | 2 retries | 60s | Default | - |

---

## Testing

Run the test suite:

```bash
# Run all resilience tests
pytest tests/test_resilience_patterns.py -v

# Run specific test categories
pytest tests/test_resilience_patterns.py::TestRetry -v
pytest tests/test_resilience_patterns.py::TestCircuitBreaker -v
pytest tests/test_resilience_patterns.py::TestDegradationManager -v
```

**Test Coverage**:
- Retry logic (exponential backoff, max retries, jitter)
- Timeout handling (sync and async)
- Fallback strategies (cached, default, function)
- Rate limiting (token bucket, leaky bucket)
- Circuit breaker (state transitions, recovery)
- Graceful degradation (tier calculations, callbacks)

---

## Monitoring

### Metrics Exposed

**Circuit Breakers**:
- `greenlang_circuit_breaker_state` - Current state (0=closed, 1=open, 2=half_open)
- `greenlang_circuit_breaker_calls_total` - Total calls by status
- `greenlang_circuit_breaker_state_transitions_total` - State transitions
- `greenlang_circuit_breaker_call_duration_seconds` - Call duration histogram
- `greenlang_circuit_breaker_failure_rate` - Current failure rate

**Rate Limiters**:
- Current tokens/level
- Request counts
- Rate limit violations

**Degradation Manager**:
- Current tier
- Service health status
- Tier change events

### Logging

All resilience patterns include structured logging:
- Retry attempts with backoff timing
- Timeout occurrences
- Fallback activations
- Rate limit violations
- Circuit breaker state changes
- Degradation tier changes

---

## Best Practices

1. **Always combine patterns**: Use retry + timeout + fallback together
2. **Set appropriate timeouts**: Match timeout to operation complexity
3. **Use circuit breakers for external services**: Prevent cascade failures
4. **Enable rate limiting**: Respect API quotas and prevent overload
5. **Monitor degradation tier**: Alert on tier changes
6. **Test failure scenarios**: Verify resilience under load
7. **Configure per environment**: Use stricter limits in production
8. **Cache aggressively**: Reduce external dependencies
9. **Implement health checks**: Keep degradation manager updated
10. **Document fallback behavior**: Make degraded mode transparent

---

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `greenlang/resilience/__init__.py` | 100 | Main module exports |
| `greenlang/resilience/retry.py` | 475 | Retry decorator with backoff |
| `greenlang/resilience/timeout.py` | 368 | Timeout decorator |
| `greenlang/resilience/fallback.py` | 553 | Fallback mechanisms |
| `greenlang/resilience/rate_limit_handler.py` | 620 | Rate limiting algorithms |
| `greenlang/resilience/circuit_breaker.py` | 558 | Circuit breaker pattern |
| `services/resilience/graceful_degradation.py` | 530 | Tier-based degradation |
| `services/resilience/integration_examples.py` | 588 | Integration examples |
| `services/resilience/config_recommendations.py` | 529 | Configuration guide |
| `tests/test_resilience_patterns.py` | 400+ | Test suite |

**Total**: ~4,800 lines of production-ready resilience infrastructure

---

## Support

For questions or issues:
- Review integration examples in `integration_examples.py`
- Check configuration recommendations in `config_recommendations.py`
- Run tests to verify behavior
- Consult GreenLang documentation

---

**Version**: 1.0.0
**Status**: Production Ready
**Author**: Team 2 - Resilience Patterns Team
**Date**: November 2025
