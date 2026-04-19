# Circuit Breaker Developer Guide

**GL-VCCI Scope 3 Platform**
**Version**: 2.0
**Last Updated**: 2025-11-09
**Team**: Platform Engineering

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Adding Circuit Breakers](#adding-circuit-breakers)
4. [Configuration Guide](#configuration-guide)
5. [Testing Circuit Breakers](#testing-circuit-breakers)
6. [Monitoring & Observability](#monitoring--observability)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Introduction

### What is a Circuit Breaker?

A circuit breaker is a resilience pattern that prevents cascading failures by:
- **Detecting failures** in external dependencies
- **Opening the circuit** after a threshold of failures
- **Failing fast** while the circuit is open (instead of wasting resources)
- **Automatically recovering** after a timeout period

### When to Use Circuit Breakers

✅ **USE circuit breakers for:**
- External API calls (Factor Broker, LLM services, ERP connectors)
- Database queries that may timeout
- Any operation that can fail independently
- Services with known reliability issues

❌ **DON'T use circuit breakers for:**
- Internal function calls
- Operations that must succeed (e.g., data writes)
- Already fast-failing operations
- Deterministic computations

### Circuit Breaker States

```
┌─────────┐
│ CLOSED  │ ◄───┐ Normal operation, requests pass through
└─────────┘     │
     │          │
     │ threshold│ success_threshold
     │ failures │ successes
     ▼          │
┌─────────┐    │
│  OPEN   │────┘ Fast-fail, don't call service
└─────────┘
     │
     │ recovery_timeout
     ▼
┌─────────┐
│HALF_OPEN│ Test if service recovered
└─────────┘
```

---

## Quick Start

### Basic Example

```python
from greenlang.intelligence.providers.resilience import ResilientHTTPClient

# Create circuit breaker for Factor Broker
factor_broker_client = ResilientHTTPClient(
    failure_threshold=5,      # Open after 5 consecutive failures
    recovery_timeout=60.0,    # Try recovery after 60 seconds
    max_retries=3,            # Retry up to 3 times per request
)

# Use the circuit breaker
async def get_emission_factor(category: str):
    try:
        result = await factor_broker_client.call(
            fetch_from_api,   # Your API call function
            category=category
        )
        return result
    except CircuitBreakerError:
        # Circuit is open, fall back
        return get_default_factor(category)
```

### With Fallback Manager (Multi-Model)

```python
from greenlang.intelligence.fallback import FallbackManager, ModelConfig

# Configure fallback chain
chain = [
    ModelConfig(model="gpt-4o", provider="openai", timeout=30.0),
    ModelConfig(model="gpt-3.5-turbo", provider="openai", timeout=20.0),
    ModelConfig(model="claude-3-sonnet", provider="anthropic", timeout=30.0),
]

manager = FallbackManager(
    fallback_chain=chain,
    enable_circuit_breaker=True
)

# Execute with automatic fallback
async def categorize_supplier(description: str):
    async def execute(config):
        return await call_llm(config.model, description)

    result = await manager.execute_with_fallback(execute)

    if result.success:
        return result.response
    else:
        raise Exception("All models failed")
```

---

## Adding Circuit Breakers

### Step 1: Identify the Service

Determine which external service needs protection:
- Factor Broker API
- LLM categorization service
- ERP connector
- Custom external API

### Step 2: Create the Circuit Breaker

```python
# File: app/services/factor_broker/client.py

from greenlang.intelligence.providers.resilience import ResilientHTTPClient
import httpx

class FactorBrokerClient:
    def __init__(self):
        # Create circuit breaker
        self.circuit_breaker = ResilientHTTPClient(
            failure_threshold=5,
            recovery_timeout=60.0,
            max_retries=3,
            base_delay=1.0,
        )

        # HTTP client
        self.http_client = httpx.AsyncClient(
            base_url="https://api.factor-broker.com",
            timeout=30.0
        )

    async def get_emission_factor(self, category: str) -> dict:
        """
        Get emission factor with circuit breaker protection

        Args:
            category: Emission category (e.g., "electricity_grid")

        Returns:
            Emission factor data

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async def api_call():
            response = await self.http_client.get(
                f"/v1/factors/{category}"
            )
            response.raise_for_status()
            return response.json()

        try:
            return await self.circuit_breaker.call(api_call)
        except CircuitBreakerError:
            # Circuit open - use fallback
            logger.warning(f"Circuit breaker open for Factor Broker")
            raise
```

### Step 3: Add Fallback Logic

```python
class FactorBrokerService:
    def __init__(self):
        self.client = FactorBrokerClient()
        self.cache = CacheManager()
        self.defaults = DEFAULT_EMISSION_FACTORS

    async def get_factor_with_fallback(self, category: str) -> dict:
        """Get factor with multi-tier fallback"""

        # Tier 1: Try primary API with circuit breaker
        try:
            return await self.client.get_emission_factor(category)
        except CircuitBreakerError:
            logger.warning("Circuit breaker open, trying fallbacks")

        # Tier 2: Check cache
        cached = await self.cache.get(f"factor:{category}")
        if cached:
            logger.info("Using cached factor")
            return {**cached, "source": "cache"}

        # Tier 3: Use defaults
        logger.warning("Using default factor")
        return {
            "category": category,
            "factor": self.defaults.get(category, 1.0),
            "source": "default",
            "degraded": True,
        }
```

### Step 4: Add Monitoring

```python
from greenlang.telemetry import MetricsCollector

metrics = MetricsCollector(namespace="factor_broker")

async def get_factor_with_metrics(category: str):
    try:
        result = await client.get_emission_factor(category)
        metrics.increment("api_calls_success")
        return result
    except CircuitBreakerError:
        metrics.increment("circuit_breaker_open")
        metrics.gauge("circuit_breaker_state", 1)  # 1 = open
        raise
    except Exception as e:
        metrics.increment("api_calls_failed")
        raise
```

---

## Configuration Guide

### Basic Configuration

```yaml
# config/resilience.yaml

resilience:
  # Circuit breaker settings
  circuit_breaker:
    failure_threshold: 5        # Open after 5 failures
    recovery_timeout: 60.0      # Wait 60s before recovery
    success_threshold: 3        # Require 3 successes to close

  # Retry settings
  retry:
    max_retries: 3
    base_delay: 1.0            # Start with 1s delay
    max_delay: 10.0            # Cap at 10s
    exponential_base: 2        # 2x multiplier

  # Timeout settings
  timeout:
    default: 30.0
    llm_api: 45.0
    factor_broker: 30.0
    erp_connector: 60.0
```

### Per-Service Configuration

```python
# config/services.py

CIRCUIT_BREAKER_CONFIG = {
    "factor_broker": {
        "failure_threshold": 5,
        "recovery_timeout": 60.0,
        "success_threshold": 3,
    },
    "llm_service": {
        "failure_threshold": 3,      # More sensitive
        "recovery_timeout": 30.0,    # Faster recovery
        "success_threshold": 2,
    },
    "erp_connector": {
        "failure_threshold": 10,     # Less sensitive
        "recovery_timeout": 120.0,   # Slower recovery
        "success_threshold": 5,
    },
}

def get_circuit_breaker(service: str) -> ResilientHTTPClient:
    """Factory for circuit breakers"""
    config = CIRCUIT_BREAKER_CONFIG.get(service, {})
    return ResilientHTTPClient(**config)
```

### Dynamic Configuration

```python
# Allow runtime configuration updates
from greenlang.config import ConfigManager

config_manager = ConfigManager()

async def update_circuit_breaker_config(service: str, **kwargs):
    """Update circuit breaker configuration at runtime"""
    await config_manager.update(
        f"resilience.circuit_breaker.{service}",
        kwargs
    )

    # Reload circuit breaker
    circuit_breakers[service] = ResilientHTTPClient(**kwargs)
```

---

## Testing Circuit Breakers

### Unit Tests

```python
# tests/test_circuit_breaker.py

import pytest
import asyncio
from greenlang.intelligence.providers.resilience import ResilientHTTPClient

@pytest.mark.asyncio
async def test_circuit_opens_after_failures():
    """Test circuit opens after threshold failures"""
    client = ResilientHTTPClient(failure_threshold=3)

    # Cause failures
    async def failing_call():
        raise Exception("Service down")

    for _ in range(3):
        try:
            await client.call(failing_call)
        except:
            pass

    # Circuit should be open
    stats = client.get_stats()
    assert stats.state.value == "open"


@pytest.mark.asyncio
async def test_circuit_recovers():
    """Test circuit recovers after timeout"""
    client = ResilientHTTPClient(
        failure_threshold=3,
        recovery_timeout=1.0
    )

    # Open circuit
    async def failing():
        raise Exception("Fail")

    for _ in range(3):
        try:
            await client.call(failing)
        except:
            pass

    # Wait for recovery
    await asyncio.sleep(1.5)

    # Should allow request
    async def succeeding():
        return {"status": "ok"}

    result = await client.call(succeeding)
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_fallback_on_circuit_open():
    """Test fallback activates when circuit open"""
    client = ResilientHTTPClient(failure_threshold=2)

    # Open circuit
    for _ in range(2):
        try:
            await client.call(lambda: asyncio.coroutine(lambda: 1/0)())
        except:
            pass

    # Use fallback
    async def get_with_fallback():
        try:
            return await client.call(lambda: None)
        except:
            return {"source": "fallback"}

    result = await get_with_fallback()
    assert result["source"] == "fallback"
```

### Integration Tests

```python
# tests/integration/test_factor_broker_resilience.py

@pytest.mark.asyncio
async def test_factor_broker_with_circuit_breaker(mock_factor_api):
    """Test Factor Broker client with circuit breaker"""

    # Mock failing API
    mock_factor_api.side_effect = [
        Exception("Fail 1"),
        Exception("Fail 2"),
        Exception("Fail 3"),
        {"factor": 0.185},  # Eventually succeeds
    ]

    service = FactorBrokerService()

    # Should fall back to cache/defaults
    result = await service.get_factor_with_fallback("electricity")

    assert result["source"] in ["cache", "default"]
    assert "factor" in result
```

### Local Testing

```python
# scripts/test_circuit_breaker_local.py

import asyncio
from services.factor_broker.client import FactorBrokerClient

async def test_manually():
    """Manual testing script"""
    client = FactorBrokerClient()

    print("=== Testing Circuit Breaker ===")

    # Test 1: Normal operation
    print("\n1. Normal operation:")
    result = await client.get_emission_factor("electricity_grid")
    print(f"   Result: {result}")

    # Test 2: Simulate failures
    print("\n2. Simulating failures...")
    for i in range(6):
        try:
            # Force failure by using invalid category
            await client.get_emission_factor("invalid_category")
        except Exception as e:
            print(f"   Attempt {i+1}: {e}")

    # Test 3: Circuit should be open
    print("\n3. Circuit should be open:")
    stats = client.circuit_breaker.get_stats()
    print(f"   State: {stats.state}")
    print(f"   Failures: {stats.failure_count}")

    # Test 4: Wait for recovery
    print("\n4. Waiting for recovery (60s)...")
    await asyncio.sleep(60)

    result = await client.get_emission_factor("electricity_grid")
    print(f"   Recovered! Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_manually())
```

---

## Monitoring & Observability

### Metrics to Track

```python
# Key metrics for circuit breakers

metrics_to_track = {
    # State metrics
    "circuit_breaker_state": "Gauge (0=closed, 1=open, 2=half_open)",
    "circuit_breaker_transitions": "Counter (state changes)",

    # Failure metrics
    "circuit_breaker_failures": "Counter (total failures)",
    "circuit_breaker_failure_rate": "Gauge (failures / total)",

    # Recovery metrics
    "circuit_breaker_recoveries": "Counter (successful recoveries)",
    "circuit_breaker_recovery_time": "Histogram (time to recover)",

    # Request metrics
    "circuit_breaker_requests_blocked": "Counter (rejected by open circuit)",
    "circuit_breaker_fallback_usage": "Counter (fallback activations)",
}
```

### Grafana Dashboard

```yaml
# dashboards/circuit_breaker.json

{
  "dashboard": {
    "title": "Circuit Breaker Health",
    "panels": [
      {
        "title": "Circuit Breaker States",
        "targets": [{
          "expr": "circuit_breaker_state{service=~\".*\"}"
        }],
        "type": "graph"
      },
      {
        "title": "Failure Rate",
        "targets": [{
          "expr": "rate(circuit_breaker_failures[5m])"
        }]
      },
      {
        "title": "Blocked Requests",
        "targets": [{
          "expr": "sum(circuit_breaker_requests_blocked)"
        }]
      }
    ]
  }
}
```

### Alerts

```yaml
# alerts/circuit_breaker.yaml

groups:
  - name: circuit_breaker
    rules:
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 5m
        severity: critical
        annotations:
          summary: "Circuit breaker {{ $labels.service }} is open"

      - alert: HighCircuitBreakerFailureRate
        expr: rate(circuit_breaker_failures[5m]) > 0.1
        for: 2m
        severity: warning
        annotations:
          summary: "High failure rate for {{ $labels.service }}"

      - alert: FrequentCircuitBreakerTransitions
        expr: rate(circuit_breaker_transitions[10m]) > 5
        severity: warning
        annotations:
          summary: "Circuit breaker {{ $labels.service }} unstable"
```

---

## Best Practices

### 1. Choose Appropriate Thresholds

```python
# Guidelines for threshold selection

THRESHOLD_GUIDELINES = {
    "critical_service": {
        "failure_threshold": 3,      # Fail fast
        "recovery_timeout": 30.0,    # Quick recovery
        "success_threshold": 5,      # Cautious recovery
    },
    "standard_service": {
        "failure_threshold": 5,
        "recovery_timeout": 60.0,
        "success_threshold": 3,
    },
    "batch_service": {
        "failure_threshold": 10,     # More tolerant
        "recovery_timeout": 120.0,   # Slower recovery
        "success_threshold": 3,
    },
}
```

### 2. Always Provide Fallbacks

```python
async def robust_api_call():
    """Always have a fallback plan"""
    try:
        # Primary: API with circuit breaker
        return await circuit_breaker.call(api_function)
    except CircuitBreakerError:
        # Secondary: Cache
        if cached := await cache.get("key"):
            return cached
    except Exception:
        # Tertiary: Default value
        return DEFAULT_VALUE
```

### 3. Log Circuit Breaker Events

```python
import logging

logger = logging.getLogger(__name__)

# Log state transitions
def on_circuit_open(service: str, stats):
    logger.error(
        f"Circuit breaker OPEN for {service}",
        extra={
            "service": service,
            "failure_count": stats.failure_count,
            "last_failure": stats.last_failure_time,
        }
    )

def on_circuit_recover(service: str):
    logger.info(f"Circuit breaker RECOVERED for {service}")
```

### 4. Test Regularly

```bash
# Add to CI/CD pipeline
pytest tests/resilience/ -v

# Monthly chaos tests
python -m tests.chaos.test_resilience_chaos

# Load testing with circuit breakers
locust -f tests/load/test_with_circuit_breakers.py
```

### 5. Document Configuration

```python
class FactorBrokerClient:
    """
    Factor Broker API client with circuit breaker protection

    Circuit Breaker Configuration:
    - failure_threshold: 5 (chosen based on API SLA of 99%)
    - recovery_timeout: 60s (vendor's typical recovery time)
    - success_threshold: 3 (ensure stability before closing)

    Fallback Strategy:
    1. Circuit breaker call to API
    2. Redis cache (24h TTL)
    3. Default emission factors

    Monitoring:
    - Grafana dashboard: /d/factor-broker
    - Alerts: CircuitBreakerOpen, HighFailureRate
    """
    pass
```

---

## Common Pitfalls

### ❌ Pitfall 1: Too Sensitive Threshold

```python
# BAD: Opens too easily
client = ResilientHTTPClient(failure_threshold=1)

# GOOD: Tolerates transient failures
client = ResilientHTTPClient(failure_threshold=5)
```

### ❌ Pitfall 2: No Fallback

```python
# BAD: No fallback when circuit opens
async def get_data():
    return await circuit_breaker.call(api_call)  # Raises exception

# GOOD: Always have fallback
async def get_data():
    try:
        return await circuit_breaker.call(api_call)
    except CircuitBreakerError:
        return await get_from_cache()
```

### ❌ Pitfall 3: Sharing Circuit Breakers

```python
# BAD: Shared circuit breaker for different services
shared_cb = ResilientHTTPClient()
await shared_cb.call(service_a_call)
await shared_cb.call(service_b_call)  # Service B affected by Service A failures!

# GOOD: Separate circuit breakers
service_a_cb = ResilientHTTPClient()
service_b_cb = ResilientHTTPClient()
```

### ❌ Pitfall 4: Ignoring Recovery Timeout

```python
# BAD: Too short recovery timeout
client = ResilientHTTPClient(recovery_timeout=5.0)  # Service needs 60s to recover

# GOOD: Match vendor recovery SLA
client = ResilientHTTPClient(recovery_timeout=60.0)
```

### ❌ Pitfall 5: Not Monitoring

```python
# BAD: No monitoring
await circuit_breaker.call(api_call)

# GOOD: Track metrics
try:
    result = await circuit_breaker.call(api_call)
    metrics.increment("success")
except CircuitBreakerError:
    metrics.increment("circuit_open")
    metrics.gauge("circuit_state", 1)
```

---

## Troubleshooting

### Circuit Opens Unexpectedly

**Symptoms**: Circuit opens during normal operation

**Diagnosis**:
```python
# Check failure patterns
stats = circuit_breaker.get_stats()
print(f"Failures: {stats.failed_calls} / {stats.total_calls}")
print(f"Failure rate: {stats.failed_calls / stats.total_calls}")

# Review logs
grep "record_failure" /var/log/app.log | tail -50
```

**Solutions**:
- Increase `failure_threshold` if transient failures are common
- Improve retry logic to handle transient errors
- Check if upstream service has issues

### Circuit Won't Recover

**Symptoms**: Circuit stays open indefinitely

**Diagnosis**:
```python
# Check recovery attempts
grep "HALF_OPEN" /var/log/app.log

# Check if service is actually recovered
curl https://api.external-service.com/health
```

**Solutions**:
- Verify `recovery_timeout` is appropriate
- Check if service is still down
- Manually reset circuit breaker if needed:
  ```python
  circuit_breaker.reset()
  ```

### High Fallback Usage

**Symptoms**: Most requests using fallback data

**Diagnosis**:
```python
# Check circuit breaker states
curl http://localhost:8000/health/circuit-breakers

# Check dependency health
curl http://localhost:8000/health/dependencies
```

**Solutions**:
- Investigate underlying service issues
- Verify network connectivity
- Check rate limits / quotas

---

## API Reference

### ResilientHTTPClient

```python
class ResilientHTTPClient:
    """
    HTTP client with circuit breaker and retry logic

    Args:
        failure_threshold (int): Failures before opening circuit
        recovery_timeout (float): Seconds before recovery attempt
        max_retries (int): Maximum retry attempts per request
        base_delay (float): Base delay for exponential backoff

    Methods:
        call(func, *args, **kwargs): Execute function with circuit breaker
        get_stats(): Get circuit breaker statistics
        reset(): Manually reset circuit breaker
    """
```

### FallbackManager

```python
class FallbackManager:
    """
    Manage fallback chain with circuit breakers

    Args:
        fallback_chain (List[ModelConfig]): Models in fallback order
        enable_circuit_breaker (bool): Enable circuit breaker per model

    Methods:
        execute_with_fallback(fn, quality_check_fn, min_quality): Execute with fallback
        get_metrics(): Get fallback statistics
    """
```

---

## Additional Resources

- **Runbooks**: `GL-VCCI-Carbon-APP/runbooks/`
- **Test Suite**: `tests/resilience/`
- **Examples**: `examples/circuit_breaker_examples.py`
- **Monitoring**: https://grafana.greenlang.com/d/resilience

## Support

- **Documentation**: https://docs.greenlang.com/resilience
- **Slack**: #platform-engineering
- **On-Call**: PagerDuty

---

**Last Updated**: 2025-11-09
**Maintained By**: Platform Engineering Team
