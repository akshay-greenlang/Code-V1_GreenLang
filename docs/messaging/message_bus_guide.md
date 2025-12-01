# MessageBus User Guide

## Overview

The GreenLang MessageBus provides standardized event-driven communication between agents. It implements publish/subscribe patterns, request-reply messaging, and comprehensive monitoring for production-grade agent orchestration.

**Key Features:**
- Event-type-based routing with wildcard support
- Priority queuing for critical events
- Request-reply pattern with timeouts
- Dead letter queue for failed deliveries
- Comprehensive metrics and monitoring
- Zero-hallucination compliance (deterministic routing)

**Based on:**
- GL-001 (ThermoSync) MessageBus implementation
- GL-003 (SteamSync) orchestration patterns

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MessageBus                         │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Priority  │───>│   Event      │───>│ Handlers  │ │
│  │    Queue    │    │   Router     │    │           │ │
│  └─────────────┘    └──────────────┘    └───────────┘ │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐                  │
│  │ Dead Letter │    │   Metrics    │                  │
│  │    Queue    │    │  Collector   │                  │
│  └─────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from greenlang.core.messaging import (
    InMemoryMessageBus,
    create_event,
    StandardEvents,
    EventPriority
)

# Create and start message bus
bus = InMemoryMessageBus()
await bus.start()

# Define event handler
async def handle_calculation(event):
    print(f"Result: {event.payload['result']}")

# Subscribe to events
await bus.subscribe(
    StandardEvents.CALCULATION_COMPLETED,
    handle_calculation,
    "my-agent"
)

# Publish event
event = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="GL-001",
    payload={"result": 42.5, "units": "kW"}
)
await bus.publish(event)

# Cleanup
await bus.close()
```

### With Configuration

```python
from greenlang.core.messaging import InMemoryMessageBus, MessageBusConfig

config = MessageBusConfig(
    max_queue_size=10000,
    enable_dead_letter=True,
    max_retries=3,
    retry_delay_seconds=1.0,
    request_timeout_seconds=30.0
)

bus = InMemoryMessageBus(config)
await bus.start()
```

## Event Types

### StandardEvents Catalog

The `StandardEvents` class provides a standardized catalog of event types:

#### Lifecycle Events
```python
StandardEvents.AGENT_STARTED           # Agent started and ready
StandardEvents.AGENT_STOPPED           # Agent stopped gracefully
StandardEvents.AGENT_ERROR             # Agent error occurred
StandardEvents.AGENT_HEARTBEAT         # Health monitoring heartbeat
StandardEvents.AGENT_CONFIGURATION_CHANGED  # Config updated
```

#### Calculation Events
```python
StandardEvents.CALCULATION_STARTED     # Calculation begun
StandardEvents.CALCULATION_COMPLETED   # Calculation finished successfully
StandardEvents.CALCULATION_FAILED      # Calculation failed
StandardEvents.CALCULATION_VALIDATED   # Results validated
StandardEvents.CALCULATION_INVALIDATED # Validation failed
```

#### Orchestration Events
```python
StandardEvents.TASK_ASSIGNED           # Task assigned to agent
StandardEvents.TASK_COMPLETED          # Task completed
StandardEvents.TASK_FAILED             # Task execution failed
StandardEvents.COORDINATION_REQUESTED  # Multi-agent coordination needed
StandardEvents.WORKFLOW_STARTED        # Multi-step workflow begun
StandardEvents.WORKFLOW_COMPLETED      # Workflow finished
```

#### Integration Events
```python
StandardEvents.INTEGRATION_CALL_STARTED     # External call started
StandardEvents.INTEGRATION_CALL_COMPLETED   # External call succeeded
StandardEvents.INTEGRATION_CALL_FAILED      # External call failed
StandardEvents.INTEGRATION_DATA_RECEIVED    # Data received from external system
StandardEvents.INTEGRATION_CONNECTION_LOST  # Connection lost
```

#### Compliance Events
```python
StandardEvents.COMPLIANCE_CHECK_STARTED        # Compliance check begun
StandardEvents.COMPLIANCE_CHECK_PASSED         # All checks passed
StandardEvents.COMPLIANCE_VIOLATION_DETECTED   # Violation found
StandardEvents.COMPLIANCE_THRESHOLD_EXCEEDED   # Threshold exceeded
StandardEvents.COMPLIANCE_REPORT_GENERATED     # Report created
```

#### Safety Events
```python
StandardEvents.SAFETY_ALERT                 # Safety alert triggered
StandardEvents.SAFETY_INTERLOCK_TRIGGERED   # Safety interlock activated
StandardEvents.SAFETY_LIMIT_EXCEEDED        # Safety limit exceeded
StandardEvents.SAFETY_EMERGENCY_SHUTDOWN    # Emergency shutdown initiated
```

#### Data Events
```python
StandardEvents.DATA_RECEIVED           # Data received for processing
StandardEvents.DATA_VALIDATED          # Data passed validation
StandardEvents.DATA_VALIDATION_FAILED  # Validation failed
StandardEvents.DATA_TRANSFORMED        # Data transformed
StandardEvents.DATA_QUALITY_ISSUE      # Data quality problem detected
```

### Creating Events

```python
from greenlang.core.messaging import create_event, EventPriority

# Simple event
event = create_event(
    event_type=StandardEvents.AGENT_STARTED,
    source_agent="GL-001",
    payload={"status": "ready"}
)

# High-priority event
critical_event = create_event(
    event_type=StandardEvents.SAFETY_ALERT,
    source_agent="GL-003",
    payload={"severity": "critical", "message": "Temperature exceeded limit"},
    priority=EventPriority.CRITICAL
)

# Event with correlation (for request-reply)
request_event = create_event(
    event_type=StandardEvents.CALCULATION_STARTED,
    source_agent="orchestrator",
    payload={"calculation_id": "calc-123"},
    correlation_id="req-456",
    target_agent="GL-001"
)
```

## Subscription Patterns

### Exact Match
```python
# Subscribe to specific event type
await bus.subscribe(
    StandardEvents.AGENT_STARTED,
    handler,
    "my-agent"
)
```

### Single-Level Wildcard (*)
```python
# Subscribe to all agent lifecycle events
await bus.subscribe(
    "agent.*",
    handler,
    "lifecycle-monitor"
)
# Matches: agent.started, agent.stopped, agent.error, etc.
```

### Multi-Level Wildcard (**)
```python
# Subscribe to all orchestration events
await bus.subscribe(
    "orchestration.**",
    handler,
    "orchestration-monitor"
)
# Matches: orchestration.task_assigned, orchestration.workflow.started, etc.
```

### All Events
```python
# Subscribe to everything (use with caution!)
await bus.subscribe(
    "*",
    handler,
    "global-monitor"
)
```

### With Filter Function
```python
# Subscribe with additional filtering
def high_priority_filter(event):
    return event.priority in (EventPriority.HIGH, EventPriority.CRITICAL)

await bus.subscribe(
    StandardEvents.SAFETY_ALERT,
    handler,
    "safety-monitor",
    filter_fn=high_priority_filter
)
```

## Priority Handling

Events are processed in priority order:

```python
EventPriority.CRITICAL  # Highest priority (safety, critical errors)
EventPriority.HIGH      # High priority (compliance, important alerts)
EventPriority.MEDIUM    # Normal priority (default)
EventPriority.LOW       # Low priority (informational, logging)
```

**Example:**
```python
# Critical safety event processed first
safety_event = create_event(
    event_type=StandardEvents.SAFETY_EMERGENCY_SHUTDOWN,
    source_agent="GL-003",
    payload={"reason": "pressure exceeded"},
    priority=EventPriority.CRITICAL
)

# Normal calculation event processed later
calc_event = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="GL-001",
    payload={"result": 42.5},
    priority=EventPriority.MEDIUM
)

await bus.publish(calc_event)
await bus.publish(safety_event)  # Published second, but processed first!
```

## Request-Reply Pattern

For synchronous request-response communication:

```python
# Requester side
request = create_event(
    event_type="calculation.request",
    source_agent="orchestrator",
    payload={"input": 21}
)

# Send request and wait for reply
reply = await bus.request_reply(request, timeout_seconds=5.0)

if reply:
    print(f"Result: {reply.payload['result']}")
else:
    print("Request timed out")

# Responder side
async def handle_request(event):
    # Process request
    result = event.payload["input"] * 2

    # Send reply
    reply = create_event(
        event_type="calculation.reply",
        source_agent="calculator",
        payload={"result": result},
        correlation_id=event.event_id  # Link to request
    )
    await bus.publish(reply)

await bus.subscribe("calculation.request", handle_request, "calculator")
```

## Error Handling

### Automatic Retries

The MessageBus automatically retries failed event deliveries:

```python
config = MessageBusConfig(
    max_retries=3,           # Retry up to 3 times
    retry_delay_seconds=1.0  # Wait 1 second between retries
)

bus = InMemoryMessageBus(config)
```

### Dead Letter Queue

Failed events are sent to the dead letter queue:

```python
# Get dead-lettered events
dead_letters = bus.get_dead_letter_queue()

for event in dead_letters:
    print(f"Failed event: {event.event_id}")

# Replay a dead-lettered event
success = await bus.replay_dead_letter(event.event_id)
```

### Error Event Publishing

Publish error events for monitoring:

```python
try:
    result = perform_calculation()
except Exception as e:
    error_event = create_event(
        event_type=StandardEvents.AGENT_ERROR,
        source_agent="GL-001",
        payload={
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        },
        priority=EventPriority.HIGH
    )
    await bus.publish(error_event)
```

## Monitoring

### Using MessageBusMonitor

```python
from greenlang.core.messaging.monitoring import MessageBusMonitor

# Create monitor
monitor = MessageBusMonitor(
    bus,
    check_interval=10.0,           # Check every 10 seconds
    max_queue_utilization=0.8,     # Alert if queue >80% full
    max_error_rate=0.05,           # Alert if error rate >5%
    max_delivery_time_ms=1000.0    # Alert if avg delivery >1s
)

await monitor.start()

# Check health
health = monitor.check_health()
print(f"Status: {health.status}")  # healthy, degraded, or unhealthy

if health.issues:
    print(f"Issues: {health.issues}")

# Get performance metrics
perf = monitor.get_performance_metrics()
print(f"Events/sec: {perf.events_per_second}")
print(f"Avg delivery time: {perf.avg_delivery_time_ms}ms")
print(f"Error rate: {perf.error_rate}")

# Get comprehensive summary
summary = monitor.get_metrics_summary()
```

### Prometheus Metrics

Export metrics for Prometheus:

```python
# Export Prometheus format
prom_metrics = monitor.export_prometheus_metrics()
print(prom_metrics)

# Serve via HTTP endpoint (example)
from aiohttp import web

async def metrics_handler(request):
    metrics = monitor.export_prometheus_metrics()
    return web.Response(text=metrics, content_type="text/plain")

app = web.Application()
app.router.add_get("/metrics", metrics_handler)
```

### Basic Metrics

Get raw metrics from the bus:

```python
metrics = bus.get_metrics()

print(f"Events published: {metrics.events_published}")
print(f"Events delivered: {metrics.events_delivered}")
print(f"Events failed: {metrics.events_failed}")
print(f"Queue size: {metrics.queue_size}")
print(f"Active subscriptions: {metrics.active_subscriptions}")
print(f"Avg delivery time: {metrics.avg_delivery_time_ms}ms")
```

## Agent Integration Examples

### GL-001 ThermoSync Example

```python
from greenlang.core.messaging import InMemoryMessageBus, create_event, StandardEvents

class ProcessHeatOrchestrator:
    def __init__(self, config):
        self.message_bus = InMemoryMessageBus()
        self._setup_event_handlers()

    async def start(self):
        """Start orchestrator."""
        await self.message_bus.start()

        # Emit started event
        await self.message_bus.publish(create_event(
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={"version": "1.0.0", "status": "ready"}
        ))

    def _setup_event_handlers(self):
        """Setup event subscriptions."""
        # Subscribe to calculation completion
        await self.message_bus.subscribe(
            StandardEvents.CALCULATION_COMPLETED,
            self._handle_calculation_completed,
            "GL-001"
        )

        # Subscribe to safety alerts
        await self.message_bus.subscribe(
            "safety.**",
            self._handle_safety_event,
            "GL-001"
        )

    async def _handle_calculation_completed(self, event):
        """Handle calculation completed event."""
        result = event.payload.get("result")
        print(f"Calculation completed: {result}")

        # Emit workflow event
        await self.message_bus.publish(create_event(
            event_type=StandardEvents.WORKFLOW_COMPLETED,
            source_agent="GL-001",
            payload={"calculation_result": result},
            correlation_id=event.correlation_id
        ))

    async def _handle_safety_event(self, event):
        """Handle safety-related events."""
        if event.priority == EventPriority.CRITICAL:
            # Emergency response
            print(f"CRITICAL SAFETY EVENT: {event.payload}")
            await self.emergency_shutdown()
```

### GL-003 SteamSync Example

```python
class SteamSystemOrchestrator:
    def __init__(self):
        self.message_bus = InMemoryMessageBus()

    async def monitor_steam_system(self):
        """Monitor steam system via events."""

        # Subscribe to sensor data events
        await self.message_bus.subscribe(
            StandardEvents.INTEGRATION_DATA_RECEIVED,
            self._handle_sensor_data,
            "GL-003"
        )

        # Subscribe to compliance violations
        await self.message_bus.subscribe(
            StandardEvents.COMPLIANCE_VIOLATION_DETECTED,
            self._handle_compliance_violation,
            "GL-003"
        )

    async def _handle_sensor_data(self, event):
        """Process sensor data."""
        sensor_type = event.payload.get("sensor_type")
        value = event.payload.get("value")

        # Validate data
        if self._is_valid_reading(sensor_type, value):
            await self.message_bus.publish(create_event(
                event_type=StandardEvents.DATA_VALIDATED,
                source_agent="GL-003",
                payload={"sensor_type": sensor_type, "value": value}
            ))
        else:
            await self.message_bus.publish(create_event(
                event_type=StandardEvents.DATA_VALIDATION_FAILED,
                source_agent="GL-003",
                payload={
                    "sensor_type": sensor_type,
                    "value": value,
                    "reason": "out_of_range"
                },
                priority=EventPriority.HIGH
            ))

    async def _handle_compliance_violation(self, event):
        """Handle compliance violations."""
        violation_type = event.payload.get("violation_type")

        # Log for audit trail
        await self.message_bus.publish(create_event(
            event_type=StandardEvents.AUDIT_LOG_CREATED,
            source_agent="GL-003",
            payload={
                "event": "compliance_violation",
                "violation_type": violation_type,
                "timestamp": event.timestamp
            }
        ))
```

## Best Practices

### 1. Use Standard Event Types

Always use `StandardEvents` constants instead of strings:

```python
# ✅ Good
await bus.publish(create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="GL-001",
    payload={"result": 42}
))

# ❌ Bad
await bus.publish(create_event(
    event_type="calc.done",  # Non-standard
    source_agent="GL-001",
    payload={"result": 42}
))
```

### 2. Set Appropriate Priorities

Use priority levels correctly:

```python
# CRITICAL: Safety-critical events
create_event(
    event_type=StandardEvents.SAFETY_EMERGENCY_SHUTDOWN,
    priority=EventPriority.CRITICAL
)

# HIGH: Compliance violations, important alerts
create_event(
    event_type=StandardEvents.COMPLIANCE_VIOLATION_DETECTED,
    priority=EventPriority.HIGH
)

# MEDIUM: Normal operations (default)
create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    priority=EventPriority.MEDIUM
)

# LOW: Informational, logging
create_event(
    event_type=StandardEvents.AGENT_HEARTBEAT,
    priority=EventPriority.LOW
)
```

### 3. Include Provenance Information

Include provenance data in event payloads:

```python
event = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="GL-001",
    payload={
        "result": 42.5,
        "calculation_id": "calc-123",
        "input_hash": "sha256:abc123...",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
)
```

### 4. Handle Errors Gracefully

Always handle errors in event handlers:

```python
async def safe_handler(event):
    try:
        result = process_event(event)

        # Emit success event
        await bus.publish(create_event(
            event_type=StandardEvents.TASK_COMPLETED,
            source_agent="my-agent",
            payload={"result": result}
        ))

    except Exception as e:
        # Emit error event
        await bus.publish(create_event(
            event_type=StandardEvents.AGENT_ERROR,
            source_agent="my-agent",
            payload={
                "error": str(e),
                "error_type": type(e).__name__
            },
            priority=EventPriority.HIGH
        ))
```

### 5. Use Correlation IDs

Link related events with correlation IDs:

```python
# Initial request
correlation_id = str(uuid.uuid4())

request = create_event(
    event_type=StandardEvents.TASK_ASSIGNED,
    source_agent="orchestrator",
    payload={"task": "calculate"},
    correlation_id=correlation_id
)

# Later, reply with same correlation ID
reply = create_event(
    event_type=StandardEvents.TASK_COMPLETED,
    source_agent="worker",
    payload={"result": 42},
    correlation_id=correlation_id  # Links to original request
)
```

### 6. Monitor Bus Health

Always monitor the message bus in production:

```python
from greenlang.core.messaging.monitoring import MessageBusMonitor

monitor = MessageBusMonitor(bus)
await monitor.start()

# Periodically check health
async def health_check_loop():
    while True:
        health = monitor.check_health()
        if health.status != "healthy":
            logger.warning(f"Bus health degraded: {health.issues}")
        await asyncio.sleep(10)
```

### 7. Cleanup Resources

Always close the message bus when done:

```python
try:
    bus = InMemoryMessageBus()
    await bus.start()

    # Use the bus...

finally:
    await bus.close()
```

## Performance Considerations

### Queue Sizing

```python
# For high-throughput applications
config = MessageBusConfig(max_queue_size=50000)

# For memory-constrained environments
config = MessageBusConfig(max_queue_size=1000)
```

### Handler Performance

Keep handlers fast and non-blocking:

```python
# ✅ Good: Fast, async handler
async def fast_handler(event):
    await quick_database_update(event.payload)

# ❌ Bad: Slow, blocking handler
async def slow_handler(event):
    time.sleep(5)  # Blocks the event loop!
    heavy_calculation()
```

### Subscription Limits

Limit subscriptions per topic:

```python
config = MessageBusConfig(max_handlers_per_topic=10)
```

## Troubleshooting

### Events Not Being Delivered

1. Check that the bus is started: `await bus.start()`
2. Verify subscription pattern matches event type
3. Check for errors in handler that cause retries to fail
4. Inspect dead letter queue: `bus.get_dead_letter_queue()`

### High Latency

1. Check queue size: `metrics.queue_size`
2. Review handler performance
3. Consider increasing max retries delay
4. Check for slow handlers blocking the queue

### Memory Issues

1. Reduce queue size: `max_queue_size`
2. Clear dead letter queue periodically
3. Limit subscription count: `max_handlers_per_topic`

## See Also

- [Event-Driven Architecture Guide](event_driven_architecture.md)
- [GL-001 ThermoSync Documentation](../applications/GL-001/README.md)
- [GL-003 SteamSync Documentation](../applications/GL-003/README.md)
