# Circuit Breaker Quick Start Guide

**For**: GL-VCCI Development Team
**Purpose**: Get started with circuit breakers in 5 minutes

---

## What Are Circuit Breakers?

Circuit breakers protect your application from cascading failures when external services fail. Think of them like electrical circuit breakers in your home - when something goes wrong, they "trip" to prevent damage.

### Three States

```
CLOSED (Healthy) -> Normal operation - calls go through
         |
         | Too many failures
         v
OPEN (Unhealthy) -> Circuit tripped - calls blocked, uses fallback
         |
         | Timeout expired
         v
HALF_OPEN (Testing) -> Testing recovery - limited calls
         |
         | Success
         v
    Back to CLOSED
```

---

## Quick Start: 3 Easy Steps

### Step 1: Import the Circuit Breaker

```python
from services.circuit_breakers import get_factor_broker_cb

factor_cb = get_factor_broker_cb()
```

### Step 2: Use It to Protect External Calls

```python
# Old way (NO PROTECTION)
factor = api.get_factor(activity, region)  # Could fail and crash!

# New way (PROTECTED)
factor = factor_cb.get_emission_factor(
    source="ecoinvent",
    activity=activity,
    region=region
)
# Automatically uses cache if API fails
```

### Step 3: Handle Errors (Optional)

```python
from greenlang.resilience import CircuitOpenError

try:
    factor = factor_cb.get_emission_factor(...)
except CircuitOpenError:
    # All sources down - use conservative default
    factor = {"value": 1.0, "quality": "fallback"}
```

---

## Complete File List

| File | Purpose | Lines |
|------|---------|-------|
| greenlang/resilience/circuit_breaker.py | Core implementation | 558 |
| services/circuit_breakers/factor_broker_cb.py | Factor API protection | 290 |
| services/circuit_breakers/llm_provider_cb.py | LLM API protection | 384 |
| services/circuit_breakers/erp_connector_cb.py | ERP protection | 398 |
| services/circuit_breakers/email_service_cb.py | Email protection | 414 |
| config/circuit_breaker_config.yaml | Configuration | 257 |
| services/circuit_breakers/README.md | Documentation | 437 |
| examples/circuit_breaker_integration.py | Integration examples | 517 |

**Total**: 2,714 lines of production-ready code

See full documentation in services/circuit_breakers/README.md
