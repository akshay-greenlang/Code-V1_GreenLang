# GL-VCCI Circuit Breaker Implementation

Production-ready circuit breakers for all external dependencies in the GL-VCCI Scope 3 Platform.

## Overview

Circuit breakers protect the platform from cascading failures when external services are unavailable. When a service fails repeatedly, the circuit "opens" and requests are immediately rejected or routed to fallbacks, preventing resource exhaustion and improving system stability.

## Features

- **Three States**: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- **Automatic Recovery**: Attempts to reconnect after configurable timeout
- **Fallback Support**: Cache-based fallbacks when services are unavailable
- **Prometheus Metrics**: Full observability of circuit breaker states
- **Thread-Safe**: Safe for concurrent use
- **Configurable**: YAML-based configuration for all settings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  Circuit Breaker Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Factor  │  │   LLM    │  │   ERP    │  │  Email   │   │
│  │  Broker  │  │ Provider │  │Connector │  │ Service  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼──────────┐
│                  External Services                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ecoinvent │  │  Claude  │  │   SAP    │  │SendGrid  │   │
│  │  DESNZ   │  │  OpenAI  │  │  Oracle  │  │          │   │
│  │   EPA    │  │          │  │ Workday  │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Core Circuit Breaker (`greenlang/resilience/circuit_breaker.py`)

Base circuit breaker implementation using `pybreaker` with GreenLang telemetry integration.

**Line count**: 587 lines

### 2. Service-Specific Wrappers

#### Factor Broker Circuit Breaker (`factor_broker_cb.py`)

Protects emission factor API calls (ecoinvent, DESNZ, EPA).

**Line count**: 245 lines

**Features**:
- Separate circuits for each factor source
- Cache-based fallback
- Conservative default values when all sources fail

#### LLM Provider Circuit Breaker (`llm_provider_cb.py`)

Protects LLM API calls (Claude, OpenAI).

**Line count**: 336 lines

**Features**:
- Automatic failover between providers
- Response caching for repeated prompts
- Token usage tracking

#### ERP Connector Circuit Breaker (`erp_connector_cb.py`)

Protects ERP system integrations (SAP, Oracle, Workday).

**Line count**: 337 lines

**Features**:
- Per-system circuit breakers
- Connection pool protection
- Cached data fallback

#### Email Service Circuit Breaker (`email_service_cb.py`)

Protects email service (SendGrid).

**Line count**: 370 lines

**Features**:
- Queue-based fallback
- Automatic retry when service recovers
- Priority-based queuing

## Usage Examples

### Example 1: Factor Broker Integration

```python
from services.circuit_breakers import get_factor_broker_cb

# Get singleton instance
factor_cb = get_factor_broker_cb()

# Fetch emission factor with circuit breaker protection
try:
    factor = factor_cb.get_emission_factor(
        source="ecoinvent",
        activity="electricity_production",
        region="US"
    )
    print(f"Emission factor: {factor['value']} {factor['unit']}")
except CircuitOpenError:
    print("Factor service temporarily unavailable - using fallback")
```

### Example 2: LLM Provider Integration

```python
from services.circuit_breakers import get_llm_provider_cb

# Get singleton instance
llm_cb = get_llm_provider_cb()

# Generate text with automatic failover
response = llm_cb.generate(
    prompt="Classify this emission source: electricity consumption",
    model="claude-3-sonnet",
    max_tokens=500,
    cache_key="classify_electricity"  # Enable caching
)

print(f"Response: {response['text']}")
print(f"Provider: {response['provider']}")
```

### Example 3: ERP Connector Integration

```python
from services.circuit_breakers import get_erp_connector_cb

# Get singleton instance
erp_cb = get_erp_connector_cb()

# Fetch suppliers with circuit breaker protection
suppliers = erp_cb.fetch_suppliers(
    system="sap",
    filters={"status": "active", "country": "US"},
    limit=1000
)

print(f"Retrieved {len(suppliers)} suppliers")

# Fetch purchase orders
purchases = erp_cb.fetch_purchases(
    system="sap",
    start_date="2024-01-01",
    end_date="2024-12-31",
    filters={"min_amount": 10000}
)

print(f"Retrieved {len(purchases)} purchase orders")
```

### Example 4: Email Service Integration

```python
from services.circuit_breakers import get_email_service_cb

# Get singleton instance
email_cb = get_email_service_cb()

# Send email with circuit breaker protection
result = email_cb.send_email(
    to="user@example.com",
    subject="Carbon Report Ready",
    body="Your Scope 3 carbon report is ready for review.",
    priority="high"
)

if result["status"] == "sent":
    print("Email sent successfully")
elif result["status"] == "queued":
    print("Email queued for retry - service temporarily down")

# Process queued emails when service recovers
queue_result = email_cb.process_queue()
print(f"Processed {queue_result['processed']} queued emails")
```

### Example 5: Decorator Pattern

```python
from greenlang.resilience import with_circuit_breaker

# Protect any function with a decorator
@with_circuit_breaker(
    name="my_external_api",
    fail_max=5,
    timeout_duration=60
)
def call_external_api(data):
    import requests
    response = requests.post("https://api.example.com/data", json=data)
    return response.json()

# Circuit breaker is automatically applied
result = call_external_api({"value": 123})
```

## Integration with Existing Code

### Before (No Circuit Breaker)

```python
# services/factor_broker/broker.py
def fetch_factor(self, source, activity, region):
    # Direct API call - no protection
    response = requests.get(f"https://{source}.api/factor", params={...})
    return response.json()
```

### After (With Circuit Breaker)

```python
# services/factor_broker/broker.py
from services.circuit_breakers import get_factor_broker_cb

def fetch_factor(self, source, activity, region):
    # Protected by circuit breaker
    factor_cb = get_factor_broker_cb()
    return factor_cb.get_emission_factor(
        source=source,
        activity=activity,
        region=region
    )
```

## Monitoring

### Prometheus Metrics

All circuit breakers expose Prometheus metrics:

```
# Circuit breaker state (0=closed, 1=open, 2=half_open)
greenlang_circuit_breaker_state{circuit_name="factor_broker_ecoinvent"} 0

# Total calls by status
greenlang_circuit_breaker_calls_total{circuit_name="factor_broker_ecoinvent",status="success"} 1250
greenlang_circuit_breaker_calls_total{circuit_name="factor_broker_ecoinvent",status="failure"} 5
greenlang_circuit_breaker_calls_total{circuit_name="factor_broker_ecoinvent",status="rejected"} 0

# State transitions
greenlang_circuit_breaker_state_transitions_total{circuit_name="factor_broker_ecoinvent",from_state="closed",to_state="open"} 2

# Call duration histogram
greenlang_circuit_breaker_call_duration_seconds{circuit_name="factor_broker_ecoinvent",status="success"} 0.145

# Failure rate
greenlang_circuit_breaker_failure_rate{circuit_name="factor_broker_ecoinvent"} 0.004
```

### Getting Statistics

```python
from services.circuit_breakers import get_factor_broker_cb

factor_cb = get_factor_broker_cb()

# Get stats for all factor sources
stats = factor_cb.get_stats()
print(stats)
# {
#   "ecoinvent": {
#     "name": "factor_broker_ecoinvent",
#     "state": "closed",
#     "fail_counter": 0,
#     "fail_max": 5,
#     "timeout_duration": 90,
#     "total_calls": 1250,
#     "total_failures": 5,
#     "failure_rate": 0.004
#   },
#   ...
# }
```

## Configuration

All circuit breakers are configured via `config/circuit_breaker_config.yaml`:

```yaml
factor_broker:
  ecoinvent:
    enabled: true
    fail_max: 5
    timeout_duration: 90
    reset_timeout: 60
    cache_ttl: 86400
```

### Environment-Specific Settings

```yaml
environments:
  production:
    global:
      defaults:
        fail_max: 10              # More tolerant in production
        timeout_duration: 120
```

## Testing

### Manual Testing

```python
from services.circuit_breakers import get_factor_broker_cb

factor_cb = get_factor_broker_cb()

# Manually open circuit for testing
factor_cb.ecoinvent_cb.open()

# Try to fetch - will use fallback
factor = factor_cb.get_emission_factor(
    source="ecoinvent",
    activity="test",
    region="US"
)
assert factor["quality"] == "fallback"

# Reset circuit
factor_cb.ecoinvent_cb.reset()
```

### Unit Tests

```python
import pytest
from greenlang.resilience import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError

def test_circuit_breaker_opens_after_failures():
    config = CircuitBreakerConfig(name="test", fail_max=3, timeout_duration=60)
    cb = CircuitBreaker(config)

    def failing_function():
        raise Exception("Service unavailable")

    # Should fail 3 times then open
    for _ in range(3):
        with pytest.raises(Exception):
            cb.call(failing_function)

    # Circuit should now be open
    assert cb.state.value == "open"

    # Next call should be immediately rejected
    with pytest.raises(CircuitOpenError):
        cb.call(failing_function)
```

## Best Practices

1. **Always use singleton instances**: Use `get_*_cb()` functions to get singleton instances
2. **Handle CircuitOpenError**: Always catch and handle circuit open errors gracefully
3. **Enable caching**: Use cache_key parameter for repeated operations
4. **Monitor metrics**: Set up Prometheus alerts for circuit state changes
5. **Configure per environment**: Use different thresholds for dev/staging/prod
6. **Process email queue**: Schedule periodic queue processing for email service
7. **Test circuit breakers**: Manually test circuit breaker behavior in staging

## Troubleshooting

### Circuit Won't Close

**Problem**: Circuit stays open even though service is back up.

**Solution**: Check the `timeout_duration` setting. Wait for the timeout period to elapse, then the circuit will move to HALF_OPEN and test recovery.

### High Failure Rate

**Problem**: Circuit opens frequently due to high failure rate.

**Solution**:
- Increase `fail_max` threshold
- Increase `timeout_duration` for slower services
- Check if external service has rate limits

### Queued Emails Not Processing

**Problem**: Email queue keeps growing.

**Solution**:
- Check if SendGrid circuit is still open
- Manually call `email_cb.process_queue()`
- Schedule periodic queue processing:

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(
    lambda: email_cb.process_queue(),
    'interval',
    minutes=5
)
scheduler.start()
```

## Files Created

1. **Core Implementation**: `greenlang/resilience/circuit_breaker.py` (587 lines)
2. **Factor Broker CB**: `services/circuit_breakers/factor_broker_cb.py` (245 lines)
3. **LLM Provider CB**: `services/circuit_breakers/llm_provider_cb.py` (336 lines)
4. **ERP Connector CB**: `services/circuit_breakers/erp_connector_cb.py` (337 lines)
5. **Email Service CB**: `services/circuit_breakers/email_service_cb.py` (370 lines)
6. **Configuration**: `config/circuit_breaker_config.yaml` (256 lines)
7. **Documentation**: `services/circuit_breakers/README.md` (this file)

**Total**: 2,131 lines of production-ready code + comprehensive documentation

## Next Steps

1. Update `requirements.txt` with `pybreaker` dependency
2. Integrate circuit breakers into existing services
3. Set up Prometheus alerts for circuit state changes
4. Schedule email queue processing
5. Load test circuit breaker thresholds
6. Document circuit breaker behavior in runbooks

## Support

For questions or issues, contact the GreenLang Platform Team.
