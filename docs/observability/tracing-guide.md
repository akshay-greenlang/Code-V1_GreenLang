# GreenLang Distributed Tracing Guide

## Overview

GreenLang uses OpenTelemetry for distributed tracing, providing end-to-end visibility into request flows across services and components.

## Quick Start

### Initialize Tracing

```python
from greenlang.observability import TracingManager, TraceConfig

# Configure tracing
config = TraceConfig(
    service_name="greenlang-api",
    environment="production",
    jaeger_endpoint="localhost:6831",
    sampling_rate=1.0  # 100% sampling
)

manager = TracingManager(config)
```

### Using Trace Decorator

```python
from greenlang.observability import trace_operation

@trace_operation("process_request")
def process_request(data):
    # Automatically traced
    return process_data(data)

@trace_operation("process_data")
def process_data(data):
    # Nested span
    return transformed_data
```

## Manual Span Creation

### Basic Spans

```python
from greenlang.observability import get_tracing_manager, SpanKind

manager = get_tracing_manager()

with manager.create_span("database_query", kind=SpanKind.CLIENT) as span:
    result = db.execute_query()
```

### Adding Attributes

```python
from greenlang.observability import add_span_attributes

@trace_operation("api_handler")
def handle_request(request):
    add_span_attributes(
        http_method=request.method,
        http_url=request.url,
        user_id=request.user.id
    )
    return process(request)
```

### Adding Events

```python
from greenlang.observability import add_span_event

@trace_operation("long_operation")
def long_operation():
    add_span_event("validation_started")
    validate()

    add_span_event("processing_started")
    process()

    add_span_event("completed")
```

## Trace Context Propagation

### HTTP Headers

```python
from greenlang.observability import get_tracing_manager

manager = get_tracing_manager()

# Inject context into HTTP headers
headers = {}
manager.inject_context(headers)

# Make HTTP request with trace context
response = requests.get(url, headers=headers)

# Extract context from incoming request
incoming_context = manager.extract_context(request.headers)
```

### Async Operations

```python
from greenlang.observability import get_trace_context_manager
import asyncio

context_manager = get_trace_context_manager()

# Save context before async operation
context = context_manager.save_context("operation_123")

async def async_task():
    # Restore context in async task
    context = context_manager.restore_context("operation_123")
    # Continue tracing with same trace ID
```

## Sampling Strategies

### Rate-Based Sampling

```python
from greenlang.observability import TraceConfig

# Sample 10% of traces
config = TraceConfig(sampling_rate=0.1)
```

### Error Sampling

```python
from greenlang.observability import ErrorSampler, CompositeSampler, RateSampler

# Sample all errors + 10% of successful requests
sampler = CompositeSampler([
    ErrorSampler(),
    RateSampler(0.1)
])
```

## Viewing Traces

### Jaeger UI

Access Jaeger at http://localhost:16686

**Features:**
- Search traces by service, operation, tags
- View trace timeline
- Analyze critical path
- Compare traces
- Find errors and anomalies

**Search Examples:**
```
service=greenlang-api operation=process_request
tags: error=true
tags: http.status_code=500
duration>1s
```

## Integration Examples

### FastAPI Application

```python
from fastapi import FastAPI, Request
from greenlang.observability import trace_operation, add_span_attributes

app = FastAPI()

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    with create_span(f"{request.method} {request.url.path}") as span:
        add_span_attributes(
            http_method=request.method,
            http_url=str(request.url),
            http_user_agent=request.headers.get("user-agent")
        )

        response = await call_next(request)

        add_span_attributes(http_status_code=response.status_code)
        return response

@app.get("/api/data")
@trace_operation("get_data")
async def get_data():
    return {"data": "value"}
```

### Database Queries

```python
@trace_operation("db_query", kind=SpanKind.CLIENT)
def execute_query(query):
    add_span_attributes(
        db_system="postgresql",
        db_statement=query,
        db_name="greenlang"
    )

    result = db.execute(query)
    add_span_attributes(db_rows_affected=result.rowcount)
    return result
```

### External API Calls

```python
import requests
from greenlang.observability import trace_operation, add_span_attributes

@trace_operation("external_api_call", kind=SpanKind.CLIENT)
def call_external_api(url):
    add_span_attributes(
        http_url=url,
        http_method="GET"
    )

    response = requests.get(url)

    add_span_attributes(
        http_status_code=response.status_code,
        http_response_size=len(response.content)
    )

    return response.json()
```

## Best Practices

### 1. Meaningful Span Names

```python
# Good: Descriptive names
@trace_operation("fetch_user_profile")
@trace_operation("calculate_emissions")
@trace_operation("send_notification_email")

# Bad: Generic names
@trace_operation("function1")
@trace_operation("process")
```

### 2. Add Relevant Attributes

```python
# Good: Rich context
add_span_attributes(
    user_id=user.id,
    tenant_id=tenant.id,
    operation_type="create",
    resource_count=len(resources)
)

# Bad: Minimal context
add_span_attributes(action="done")
```

### 3. Use Appropriate Span Kinds

```python
# Internal operations
@trace_operation("calculate", kind=SpanKind.INTERNAL)

# HTTP client calls
@trace_operation("http_call", kind=SpanKind.CLIENT)

# HTTP server handlers
@trace_operation("api_handler", kind=SpanKind.SERVER)

# Message producers
@trace_operation("publish_event", kind=SpanKind.PRODUCER)

# Message consumers
@trace_operation("consume_event", kind=SpanKind.CONSUMER)
```

### 4. Don't Over-Trace

```python
# Good: Trace meaningful operations
@trace_operation("process_payment")
def process_payment(amount):
    validate(amount)
    charge_card(amount)
    send_receipt()

# Bad: Don't trace every tiny function
@trace_operation("add_numbers")  # Too granular
def add(a, b):
    return a + b
```

## Performance Optimization

### Adjust Sampling

```python
# Production: Lower sampling rate
config = TraceConfig(sampling_rate=0.1)  # 10%

# Staging: Higher sampling
config = TraceConfig(sampling_rate=0.5)  # 50%

# Development: Full sampling
config = TraceConfig(sampling_rate=1.0)  # 100%
```

### Conditional Tracing

```python
import os

ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"

if ENABLE_TRACING:
    @trace_operation("expensive_operation")
    def expensive_operation():
        pass
else:
    def expensive_operation():
        pass
```

## Troubleshooting

### Finding Slow Requests

1. Open Jaeger UI
2. Select service: "greenlang-api"
3. Search with: `duration>1s`
4. Analyze critical path in timeline

### Finding Errors

1. Search with: `tags: error=true`
2. Filter by service and time range
3. Examine error traces
4. Check span tags for error details

### Trace Not Appearing

**Check:**
- Jaeger endpoint connectivity
- Sampling configuration
- Trace export is enabled
- Network between services

**Debug:**
```python
config = TraceConfig(console_export=True)  # Print to console
```

## Advanced Topics

### Custom Exporters

```python
from greenlang.observability import TraceConfig

config = TraceConfig(
    otlp_endpoint="http://otel-collector:4317",  # OpenTelemetry Collector
    jaeger_endpoint=None  # Disable direct Jaeger export
)
```

### Span Links

```python
# Link related traces
with create_span("batch_process") as batch_span:
    for item in items:
        with create_span("process_item") as item_span:
            # Processing
            pass
```

### Baggage

```python
from greenlang.observability import SpanContext

context = SpanContext(
    baggage={"customer_tier": "premium", "region": "us-west"}
)
# Baggage propagates through entire trace
```
