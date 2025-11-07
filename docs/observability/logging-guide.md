# GreenLang Logging Guide

## Overview

GreenLang uses structured JSON logging for easy parsing and analysis by SIEM systems and log aggregation tools.

## Quick Start

### Basic Configuration

```python
from greenlang.observability import configure_logging

# Configure JSON logging
configure_logging(
    level="INFO",
    format_json=True,
    log_file="/var/log/greenlang/app.log"
)
```

### Using Structured Logger

```python
from greenlang.observability import get_logger, LogContext

# Create logger with context
context = LogContext(
    tenant_id="customer1",
    component="api",
    environment="production"
)
logger = get_logger("my_module", context)

# Log messages
logger.info("Processing request", request_id="req123", count=42)
logger.warning("High latency detected", latency_ms=1500)
logger.error("Database connection failed", exception=e, retry_count=3)
```

## Log Format

### JSON Structure

```json
{
  "timestamp": "2025-01-15T10:30:45.123456Z",
  "level": "INFO",
  "message": "Request processed successfully",
  "tenant_id": "customer1",
  "component": "api",
  "operation": "process_request",
  "data": {
    "request_id": "req123",
    "duration_ms": 145,
    "status_code": 200
  }
}
```

### Error Logs

```json
{
  "timestamp": "2025-01-15T10:31:12.456789Z",
  "level": "ERROR",
  "message": "Failed to process request",
  "tenant_id": "customer1",
  "component": "worker",
  "exception": {
    "type": "ValueError",
    "message": "Invalid input data",
    "traceback": "Traceback (most recent call last):\\n..."
  }
}
```

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical messages for very serious errors

## Contextual Logging

### Temporary Context

```python
logger = get_logger("api")

with logger.with_context(request_id="req456", user_id="user789"):
    logger.info("Starting request processing")
    # Context is automatically included
    logger.info("Request validated")
    logger.info("Request completed")
# Context is restored after block
```

### Correlation IDs

```python
import uuid

context = LogContext(
    request_id=str(uuid.uuid4()),
    trace_id=str(uuid.uuid4())
)
logger = get_logger("service", context)
```

## Log Aggregation

### Using LogAggregator

```python
from greenlang.observability import get_log_aggregator, LogLevel

aggregator = get_log_aggregator()

# Get recent errors
errors = aggregator.get_logs(level=LogLevel.ERROR, limit=50)

# Get logs for specific component
api_logs = aggregator.get_logs(component="api", limit=100)

# Get statistics
stats = aggregator.get_statistics()
print(f"Total logs: {stats['total_logs']}")
print(f"Error count: {stats['log_counts']['ERROR']}")

# Get error summary
error_summary = aggregator.get_error_summary()
```

## Loki Integration

### Promtail Configuration

Logs are automatically shipped to Loki when using the Docker Compose setup. Configuration in `promtail-config.yml`:

```yaml
scrape_configs:
  - job_name: greenlang
    static_configs:
      - targets:
          - localhost
        labels:
          job: greenlang
          __path__: /var/log/greenlang/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            component: component
            tenant_id: tenant_id
      - labels:
          level:
          component:
          tenant_id:
```

### Querying in Grafana

```logql
# All error logs
{job="greenlang"} | json | level="ERROR"

# Logs for specific component
{job="greenlang"} | json | component="api"

# Logs with pattern
{job="greenlang"} | json | message =~ ".*timeout.*"
```

## Best Practices

### 1. Use Structured Data

```python
# Good: Structured data
logger.info("User login", user_id="123", ip="192.168.1.1", success=True)

# Bad: Unstructured string
logger.info(f"User 123 logged in from 192.168.1.1 successfully")
```

### 2. Include Context

```python
# Good: Rich context
logger.error(
    "Payment processing failed",
    transaction_id="tx789",
    amount=99.99,
    currency="USD",
    error_code="INSUFFICIENT_FUNDS"
)

# Bad: Minimal context
logger.error("Payment failed")
```

### 3. Log at Appropriate Levels

```python
# DEBUG: Detailed trace information
logger.debug("Entering function", args=args, kwargs=kwargs)

# INFO: General flow
logger.info("Processing started", job_id=job_id)

# WARNING: Recoverable issues
logger.warning("Retry attempt", attempt=3, max_attempts=5)

# ERROR: Errors that need attention
logger.error("Failed to connect", exception=e)

# CRITICAL: System-threatening errors
logger.critical("Database unavailable", all_retries_exhausted=True)
```

### 4. Don't Log Sensitive Data

```python
# Bad: Logging passwords
logger.info("User login", password=password)  # Never do this!

# Good: Sanitize sensitive data
logger.info("User login", username=username, password_length=len(password))
```

## Performance Considerations

### Async Logging

For high-throughput applications, use async logging:

```python
from greenlang.observability import LogShipper

shipper = get_log_shipper()
shipper.start_shipping()

# Logs are batched and shipped asynchronously
# Minimal impact on application performance
```

### Log Sampling

For very high-volume logs:

```python
import random

if random.random() < 0.1:  # Sample 10% of debug logs
    logger.debug("Detailed trace", data=large_data)
```

### Log Rotation

Configure rotation to prevent disk space issues:

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    '/var/log/greenlang/app.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=10  # Keep 10 backup files
)
```

## Troubleshooting

### Finding Errors

```python
# Get recent errors
aggregator = get_log_aggregator()
recent_errors = aggregator.get_logs(level=LogLevel.ERROR, limit=20)

for error in recent_errors:
    print(f"{error.timestamp}: {error.message}")
    if error.exception:
        print(f"  Type: {error.exception['type']}")
        print(f"  Message: {error.exception['message']}")
```

### Error Patterns

```python
# Analyze error patterns
aggregator = get_log_aggregator()
summary = aggregator.get_error_summary()

for error_type, info in summary['error_types'].items():
    print(f"{error_type}: {info['count']} occurrences")
    print(f"  Latest: {info['latest']}")
    print(f"  Components: {info['components']}")
```

## Integration Examples

### Flask Application

```python
from flask import Flask, request, g
from greenlang.observability import get_logger, LogContext
import uuid

app = Flask(__name__)

@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.logger = get_logger(
        "flask_app",
        LogContext(request_id=g.request_id)
    )
    g.logger.info("Request started", method=request.method, path=request.path)

@app.after_request
def after_request(response):
    g.logger.info(
        "Request completed",
        status_code=response.status_code,
        content_length=response.content_length
    )
    return response
```

### Async Application

```python
import asyncio
from greenlang.observability import get_logger, LogContext

async def process_task(task_id):
    logger = get_logger("async_worker", LogContext(task_id=task_id))

    logger.info("Task started")
    try:
        await asyncio.sleep(1)  # Simulate work
        logger.info("Task completed", duration_s=1.0)
    except Exception as e:
        logger.error("Task failed", exception=e)
        raise
```
