# GL-007 FurnacePerformanceMonitor - Best Practices Guide

**Version:** 1.0
**Date:** 2025-11-19
**Purpose:** Coding standards and best practices for GL-007 development

---

## Overview

This guide documents the best practices demonstrated in GL-007 FurnacePerformanceMonitor and provides standards for future development. Following these practices ensures consistent, maintainable, and production-quality code.

---

## 1. Type Annotations

### Standard: 100% Type Coverage

**Rule:** All functions must have complete type annotations for parameters and return values.

**Good:**
```python
async def check_health(self) -> HealthResponse:
    """Perform comprehensive health check."""
    # Implementation

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> None:
    """Configure logging."""
    # Implementation
```

**Bad:**
```python
async def check_health(self):  # Missing return type
    # Implementation

def setup_logging(log_level, log_file=None):  # Missing all types
    # Implementation
```

**Advanced Patterns:**

```python
from typing import Dict, List, Optional, Union, Literal, Protocol

# Use Literal for string constants
def set_environment(env: Literal["development", "staging", "production"]) -> None:
    pass

# Use Protocol for duck typing
class HealthCheckProtocol(Protocol):
    async def check(self) -> ComponentHealth:
        ...

# Use generics for containers
def process_metrics(metrics: Dict[str, float]) -> List[str]:
    pass

# Use Union sparingly (prefer Optional)
def get_config(name: str) -> Optional[str]:  # Preferred
    pass

def get_config(name: str) -> Union[str, None]:  # Avoid
    pass
```

---

## 2. Documentation

### Standard: Google-Style Docstrings

**Rule:** All public functions, classes, and modules must have comprehensive docstrings.

**Good:**
```python
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> None:
    """
    Configure structured logging for GL-007.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for no file logging)
        enable_console: Enable console output

    Raises:
        ValueError: If log_level is invalid
        IOError: If log_file directory cannot be created

    Example:
        >>> setup_logging(log_level="DEBUG", enable_console=True)
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    # Implementation
```

**Module Docstrings:**
```python
"""
Structured logging configuration for GL-007 FurnacePerformanceMonitor.

Provides:
- JSON-formatted structured logging
- Correlation ID tracking
- Context propagation
- Log levels configuration
- Log rotation
- ELK/Loki integration

Example:
    >>> from monitoring import setup_logging, get_logger
    >>> setup_logging(log_level="INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Started")
"""
```

**Class Docstrings:**
```python
class HealthChecker:
    """
    Comprehensive health check system for GL-007.

    Performs concurrent health checks on:
    - Application startup
    - Database connectivity
    - Cache connectivity
    - SCADA connectivity
    - ML model availability
    - System resources

    Attributes:
        config: Application configuration
        start_time: Application start timestamp
        last_check_time: Last health check timestamp

    Example:
        >>> checker = HealthChecker(config)
        >>> health = await checker.check_health()
        >>> assert health.status == HealthStatus.HEALTHY
    """
```

---

## 3. Error Handling

### Standard: Comprehensive with Graceful Degradation

**Rule:** All I/O operations must have try-except blocks with appropriate error handling.

**Good:**
```python
async def _check_database(self) -> ComponentHealth:
    """Check database connectivity."""
    start = time.time()
    try:
        await db.execute("SELECT 1")
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            details={"connected": True}
        )
    except DatabaseError as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            details={},
            error=str(e)
        )
```

**Graceful Degradation Pattern:**
```python
async def _check_cache(self) -> ComponentHealth:
    """Check cache with graceful degradation."""
    try:
        await cache.ping()
        return ComponentHealth(name="cache", status=HealthStatus.HEALTHY, ...)
    except CacheError as e:
        logger.warning(f"Cache unavailable: {e}")
        # Cache is optional - degrade gracefully
        return ComponentHealth(
            name="cache",
            status=HealthStatus.DEGRADED,  # Not UNHEALTHY
            error=str(e)
        )
```

**Exception Specificity:**
```python
# Good - Specific exceptions
try:
    await db.execute(query)
except DatabaseConnectionError:
    # Handle connection issues
except DatabaseTimeoutError:
    # Handle timeouts
except DatabaseQueryError:
    # Handle query errors

# Bad - Generic exceptions
try:
    await db.execute(query)
except Exception:  # Too broad
    pass
```

---

## 4. Async/Await Best Practices

### Standard: Proper Async Patterns

**Concurrent Execution:**
```python
# Good - Concurrent execution
async def check_health(self):
    checks = [
        self._check_database(),
        self._check_cache(),
        self._check_scada(),
    ]
    results = await asyncio.gather(*checks, return_exceptions=True)

# Bad - Sequential execution
async def check_health(self):
    db_health = await self._check_database()
    cache_health = await self._check_cache()
    scada_health = await self._check_scada()
```

**Async/Sync Decorator Wrapper:**
```python
def traced(span_name: str):
    """Decorator supporting both async and sync functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Async implementation
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Sync implementation
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
```

**Avoid Blocking I/O:**
```python
# Good - Non-blocking
async def read_config():
    async with aiofiles.open('config.yaml', 'r') as f:
        content = await f.read()
    return yaml.safe_load(content)

# Bad - Blocking in async function
async def read_config():
    with open('config.yaml', 'r') as f:  # Blocks event loop!
        content = f.read()
    return yaml.safe_load(content)
```

---

## 5. Logging Best Practices

### Standard: Structured Logging with Context

**Use Correlation IDs:**
```python
from monitoring import set_correlation_id, get_logger

logger = get_logger(__name__)

async def process_request(request_id: str):
    set_correlation_id(request_id)
    logger.info("Processing request")  # Includes correlation_id automatically
```

**Log Levels:**
```python
# DEBUG: Detailed diagnostic information
logger.debug("Cache lookup", extra={'key': cache_key, 'ttl': 300})

# INFO: Normal operation milestones
logger.info("Health check completed", extra={'status': 'healthy', 'duration_ms': 45})

# WARNING: Unexpected but recoverable issues
logger.warning("Cache miss rate high", extra={'miss_rate': 0.75})

# ERROR: Errors requiring attention
logger.error("Database connection failed", exc_info=True)

# CRITICAL: System-wide failures
logger.critical("All health checks failed - system down")
```

**Structured Logging:**
```python
# Good - Structured with extra fields
logger.info(
    "Thermal efficiency calculated",
    extra={
        'furnace_id': 'F-001',
        'efficiency_percent': 85.2,
        'calculation_time_ms': 42,
        'data_points': 1000,
    }
)

# Bad - Unstructured string
logger.info(f"Efficiency for F-001: 85.2% (42ms, 1000 points)")
```

**Context Managers:**
```python
from monitoring import LogContext

# Temporary logging context
with LogContext(furnace_id='F-001', user_id='USR-123'):
    logger.info("Processing furnace data")  # Includes furnace_id, user_id
    # ... processing ...
# Context automatically cleared
```

---

## 6. Metrics Best Practices

### Standard: Comprehensive Instrumentation

**Metric Naming:**
```python
# Pattern: {agent}_{component}_{metric}_{unit}
furnace_thermal_efficiency_percent = Gauge(
    'gl_007_furnace_thermal_efficiency_percent',
    'Furnace thermal efficiency (%)',
    ['furnace_id', 'zone']
)
```

**Metric Types:**
```python
# Counter - Monotonically increasing (counts, totals)
http_requests_total = Counter(
    'gl_007_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Gauge - Point-in-time values (temperatures, percentages)
furnace_temperature_celsius = Gauge(
    'gl_007_furnace_temperature_celsius',
    'Furnace temperature (°C)',
    ['furnace_id', 'zone']
)

# Histogram - Distribution of values (latencies, sizes)
http_request_duration_seconds = Histogram(
    'gl_007_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Summary - Similar to Histogram with percentiles
response_size_bytes = Summary(
    'gl_007_response_size_bytes',
    'Response body size in bytes'
)
```

**Decorator Pattern:**
```python
@track_calculation_metrics('thermal_efficiency')
async def calculate_thermal_efficiency(furnace_id: str) -> float:
    # Automatically tracked:
    # - Duration
    # - Success/failure
    # - Request count
    return efficiency
```

---

## 7. Distributed Tracing

### Standard: OpenTelemetry Integration

**Decorator Pattern:**
```python
from monitoring import traced, add_span_attributes, add_span_event

@traced("calculate_efficiency", kind=SpanKind.INTERNAL)
async def calculate_thermal_efficiency(furnace_id: str) -> float:
    add_span_attributes(furnace_id=furnace_id)
    add_span_event("calculation_started")

    # ... calculation logic ...

    add_span_event("calculation_completed", {"result": efficiency})
    return efficiency
```

**Context Manager Pattern:**
```python
from monitoring import TracingContext, FurnaceTracing

# Generic tracing context
with TracingContext("process_data", furnace_id="F-001"):
    # ... processing ...
    pass

# Domain-specific tracing
with FurnaceTracing.trace_calculation("thermal_efficiency", "F-001"):
    efficiency = calculate_efficiency()
```

**Span Attributes:**
```python
# Good attributes
add_span_attributes(
    furnace_id="F-001",
    calculation_type="thermal_efficiency",
    data_points=1000,
    result=85.2
)

# Avoid high-cardinality attributes
add_span_attributes(
    timestamp=datetime.now().isoformat(),  # Don't use timestamps
    raw_data=sensor_data,  # Don't use large objects
)
```

---

## 8. Code Organization

### Standard: Clear Module Structure

**File Organization:**
```
GL-007/
├── monitoring/
│   ├── __init__.py           # Package exports
│   ├── health_checks.py      # Health check implementation
│   ├── logging_config.py     # Logging configuration
│   ├── metrics.py            # Metrics definitions
│   ├── tracing_config.py     # Tracing configuration
│   └── types.py              # Shared type definitions
├── tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── conftest.py           # Shared fixtures
├── docs/                     # Documentation
├── validate_spec.py          # Spec validation
└── README.md                 # Project overview
```

**Import Organization:**
```python
# 1. Standard library imports
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# 2. Third-party imports
from prometheus_client import Counter, Gauge
from opentelemetry import trace

# 3. Local application imports
from monitoring.types import HealthStatus
from monitoring.exceptions import HealthCheckError
```

---

## 9. Configuration Management

### Standard: Environment-Based Configuration

**Configuration Pattern:**
```python
# Use environment variables with defaults
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
LOG_DIR = Path(os.getenv('LOG_DIR', '/var/log/greenlang'))

# Environment-specific presets
CONFIGS = {
    'development': {
        'log_level': 'DEBUG',
        'enable_json': False,
        'enable_console': True,
    },
    'production': {
        'log_level': 'INFO',
        'enable_json': True,
        'log_file': str(LOG_DIR / 'gl-007' / 'app.log'),
    },
}

# Load based on environment
config = CONFIGS.get(ENVIRONMENT, CONFIGS['production'])
```

**Portable Paths:**
```python
from pathlib import Path
import os

# Good - Portable
log_dir = Path(os.getenv('LOG_DIR', '/var/log'))
log_file = log_dir / 'greenlang' / 'gl-007' / 'app.log'

# Bad - Hardcoded Unix path
log_file = '/var/log/greenlang/gl-007/app.log'

# Bad - Hardcoded Windows path
log_file = 'C:\\Logs\\greenlang\\gl-007\\app.log'
```

---

## 10. Code Complexity

### Standard: Keep It Simple

**Cyclomatic Complexity < 10:**
```python
# Good - Low complexity (CC: 4)
async def check_readiness(self) -> ReadinessResponse:
    checks = {
        "database": await self._is_database_ready(),
        "cache": await self._is_cache_ready(),
        "scada": await self._is_scada_ready(),
    }
    overall_ready = all(checks.values())
    return ReadinessResponse(ready=overall_ready, checks=checks)

# Bad - High complexity (CC: 12)
def process_data(data):
    if data:
        if data.valid:
            if data.type == 'A':
                if data.priority > 5:
                    # ... deeply nested logic ...
```

**Function Length < 50 Lines:**
```python
# Good - Single responsibility
async def _check_database(self) -> ComponentHealth:
    """Check database connectivity."""
    # 20 lines of focused logic

# Bad - Multiple responsibilities (150 lines)
async def process_furnace_data_and_generate_report(...):
    # Fetch data
    # Validate data
    # Calculate metrics
    # Generate report
    # Send notifications
    # Update database
    # ... (too much in one function)
```

**Extract Complex Logic:**
```python
# Good - Extracted helper
def _calculate_thermal_efficiency(
    heat_input: float,
    heat_output: float,
    heat_losses: Dict[str, float]
) -> float:
    """Calculate thermal efficiency."""
    total_losses = sum(heat_losses.values())
    useful_heat = heat_output - total_losses
    return (useful_heat / heat_input) * 100

# Use in main function
async def analyze_furnace(furnace_id: str):
    data = await fetch_data(furnace_id)
    efficiency = _calculate_thermal_efficiency(
        data.heat_input,
        data.heat_output,
        data.heat_losses
    )
```

---

## 11. Testing Practices

### Standard: Comprehensive Test Coverage

**Test Structure:**
```python
# tests/unit/test_health_checks.py
import pytest
from monitoring import HealthChecker, HealthStatus

@pytest.fixture
def health_checker():
    """Create health checker instance."""
    config = {"app_version": "1.0.0", "environment": "test"}
    return HealthChecker(config)

@pytest.mark.asyncio
async def test_health_check_all_healthy(health_checker):
    """Test health check when all components healthy."""
    # Arrange (setup already in fixture)

    # Act
    response = await health_checker.check_health()

    # Assert
    assert response.status == HealthStatus.HEALTHY
    assert "database" in response.components
    assert response.components["database"].status == HealthStatus.HEALTHY

@pytest.mark.asyncio
async def test_health_check_with_database_failure(health_checker, mocker):
    """Test health check handles database failure gracefully."""
    # Arrange
    mocker.patch.object(
        health_checker,
        '_check_database',
        side_effect=Exception("Connection refused")
    )

    # Act
    response = await health_checker.check_health()

    # Assert
    assert response.status == HealthStatus.UNHEALTHY
    assert response.components["database"].error is not None
```

**Test Categories:**
- Unit tests (90%+ coverage target)
- Integration tests (key workflows)
- Performance tests (latency benchmarks)
- Chaos tests (failure scenarios)

---

## 12. Security Practices

### Standard: Defense in Depth

**Input Validation:**
```python
def set_log_level(level: str) -> None:
    """Set logging level with validation."""
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {level}")
    logging.getLogger().setLevel(getattr(logging, level.upper()))
```

**No Secrets in Code:**
```python
# Good - Environment variables
database_url = os.getenv('DATABASE_URL')
api_key = os.getenv('API_KEY')

# Bad - Hardcoded secrets
database_url = "postgresql://user:password@localhost/db"
api_key = "sk-1234567890abcdef"
```

**Logging Security:**
```python
# Good - Sanitized logging
logger.info(
    "User authenticated",
    extra={'user_id': user.id}  # Don't log passwords!
)

# Bad - Sensitive data in logs
logger.info(f"User {user.email} logged in with password {password}")
```

---

## 13. Performance Practices

### Standard: Optimize for Production

**Use Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_config(environment: str) -> Dict[str, Any]:
    """Get configuration (cached)."""
    return load_config_from_file(environment)
```

**Connection Pooling:**
```python
# Good - Reuse connections
async with db_pool.acquire() as conn:
    result = await conn.execute(query)

# Bad - New connection every time
conn = await create_connection()
result = await conn.execute(query)
await conn.close()
```

**Batch Operations:**
```python
# Good - Batch database inserts
async with db.transaction():
    await db.executemany(insert_query, records)

# Bad - Individual inserts
for record in records:
    await db.execute(insert_query, record)
```

---

## 14. Code Review Checklist

Use this checklist for all code reviews:

### Type Safety
- [ ] All functions have type annotations
- [ ] Return types are specified
- [ ] Optional types used correctly
- [ ] No `Any` types unless necessary

### Documentation
- [ ] Module docstring present
- [ ] Class docstrings complete
- [ ] Function docstrings with Args/Returns
- [ ] Complex logic has comments

### Error Handling
- [ ] All I/O has try-except blocks
- [ ] Specific exceptions caught
- [ ] Errors logged with context
- [ ] Graceful degradation where appropriate

### Testing
- [ ] Unit tests written
- [ ] Edge cases covered
- [ ] Mocking used appropriately
- [ ] Test names are descriptive

### Code Quality
- [ ] Cyclomatic complexity < 10
- [ ] Function length < 50 lines
- [ ] No code duplication
- [ ] Clear variable names

### Performance
- [ ] Async/await used correctly
- [ ] Concurrent operations where possible
- [ ] No blocking I/O in async functions
- [ ] Appropriate caching

### Security
- [ ] Input validation present
- [ ] No hardcoded secrets
- [ ] Sensitive data not logged
- [ ] SQL injection prevented

---

## 15. Common Patterns

### Health Check Pattern
```python
async def _check_component(self) -> ComponentHealth:
    """Standard health check pattern."""
    start = time.time()
    try:
        # Perform check
        result = await self._do_check()
        latency = (time.time() - start) * 1000

        return ComponentHealth(
            name="component",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            details=result
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Component check failed: {e}")
        return ComponentHealth(
            name="component",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            details={},
            error=str(e)
        )
```

### Context Manager Pattern
```python
class ResourceManager:
    """Manage resource lifecycle."""

    async def __aenter__(self):
        """Acquire resource."""
        self.resource = await self._acquire()
        return self.resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release resource."""
        await self._release(self.resource)
        if exc_type:
            logger.error(f"Error in context: {exc_val}")
        return False  # Don't suppress exceptions

# Usage
async with ResourceManager() as resource:
    await resource.use()
```

---

## Conclusion

Following these best practices ensures:
- **High code quality:** Maintainable, readable code
- **Production readiness:** Robust error handling and monitoring
- **Developer efficiency:** Clear patterns and standards
- **System reliability:** Performance and security built-in

**Remember:** These are guidelines based on GL-007's success. Adapt as needed for specific use cases, but maintain the high quality bar.
