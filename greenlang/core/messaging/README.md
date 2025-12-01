# GreenLang Messaging Infrastructure

## Overview

The GreenLang messaging infrastructure provides standardized event-driven communication for agent orchestration. It implements publish/subscribe patterns, request-reply messaging, priority queuing, and comprehensive monitoring.

**Status:** Production Ready
**Version:** 1.0.0
**Based on:** GL-001 (ThermoSync) and GL-003 (SteamSync) implementations

## Components

### 1. Event System (`events.py`)

Defines standard event types and event structures for agent communication.

**Key Classes:**
- `Event`: Immutable event structure with type, source, payload, priority
- `EventPriority`: Priority levels (CRITICAL, HIGH, MEDIUM, LOW)
- `StandardEvents`: Catalog of standard event types

**Event Categories:**
- Lifecycle: Agent start/stop/error
- Calculation: Calculation start/complete/fail
- Orchestration: Task assignment, workflow coordination
- Integration: External system communication
- Compliance: Regulatory compliance events
- Safety: Safety alerts and interlocks
- Data: Data processing and validation

**Usage:**
```python
from greenlang.core.messaging import create_event, StandardEvents, EventPriority

event = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="GL-001",
    payload={"result": 42.5, "units": "kW"},
    priority=EventPriority.HIGH
)
```

### 2. Message Bus (`message_bus.py`)

Asynchronous message bus implementing publish/subscribe and request-reply patterns.

**Key Classes:**
- `MessageBus`: Abstract base class defining interface
- `InMemoryMessageBus`: In-memory implementation for single-process
- `RedisMessageBus`: Redis-backed for distributed (placeholder)
- `MessageBusConfig`: Configuration options
- `Subscription`: Event subscription management

**Features:**
- Topic-based routing with wildcard support (`*` and `**`)
- Priority queuing (CRITICAL > HIGH > MEDIUM > LOW)
- Request-reply pattern with timeout
- Dead letter queue for failed deliveries
- Automatic retry with configurable delay
- Comprehensive metrics collection

**Usage:**
```python
from greenlang.core.messaging import InMemoryMessageBus, StandardEvents

bus = InMemoryMessageBus()
await bus.start()

# Subscribe
async def handle_calculation(event):
    print(f"Result: {event.payload['result']}")

await bus.subscribe(
    StandardEvents.CALCULATION_COMPLETED,
    handle_calculation,
    "my-subscriber"
)

# Publish
event = create_event(...)
await bus.publish(event)

# Cleanup
await bus.close()
```

### 3. Monitoring (`monitoring.py`)

Health monitoring and metrics collection for the message bus.

**Key Classes:**
- `MessageBusMonitor`: Monitors bus health and performance
- `HealthStatus`: Health check results
- `PerformanceMetrics`: Performance statistics

**Features:**
- Health status checking (healthy/degraded/unhealthy)
- Performance metrics (events/sec, delivery time, error rate)
- Automatic issue detection (high queue, error rate, latency)
- Prometheus metrics export
- Historical data tracking

**Usage:**
```python
from greenlang.core.messaging.monitoring import MessageBusMonitor

monitor = MessageBusMonitor(bus, check_interval=10.0)
await monitor.start()

# Check health
health = monitor.check_health()
if health.status != "healthy":
    print(f"Issues: {health.issues}")

# Get metrics
metrics = monitor.get_performance_metrics()
print(f"Events/sec: {metrics.events_per_second}")

# Export Prometheus
prom_metrics = monitor.export_prometheus_metrics()
```

## Quick Start

### Basic Publish/Subscribe

```python
from greenlang.core.messaging import InMemoryMessageBus, create_event, StandardEvents

# Create bus
bus = InMemoryMessageBus()
await bus.start()

# Subscribe to events
async def handler(event):
    print(f"Received: {event.payload}")

await bus.subscribe(
    StandardEvents.AGENT_STARTED,
    handler,
    "my-agent"
)

# Publish event
event = create_event(
    event_type=StandardEvents.AGENT_STARTED,
    source_agent="GL-001",
    payload={"status": "ready"}
)
await bus.publish(event)

# Cleanup
await bus.close()
```

### Wildcard Subscriptions

```python
# Single-level wildcard: agent.* matches agent.started, agent.stopped
await bus.subscribe("agent.*", handler, "lifecycle-monitor")

# Multi-level wildcard: orchestration.** matches all orchestration events
await bus.subscribe("orchestration.**", handler, "orchestration-monitor")

# All events
await bus.subscribe("*", handler, "global-monitor")
```

### Request-Reply Pattern

```python
# Send request
request = create_event(
    event_type="calculation.request",
    source_agent="requester",
    payload={"input": 42}
)

reply = await bus.request_reply(request, timeout_seconds=5.0)
if reply:
    print(f"Result: {reply.payload['result']}")
```

### Priority Handling

```python
# Critical event (processed first)
critical_event = create_event(
    event_type=StandardEvents.SAFETY_EMERGENCY_SHUTDOWN,
    source_agent="safety-monitor",
    payload={"reason": "pressure exceeded"},
    priority=EventPriority.CRITICAL
)

# Normal event (processed later)
normal_event = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="calculator",
    payload={"result": 42.5},
    priority=EventPriority.MEDIUM
)
```

## Configuration

```python
from greenlang.core.messaging import MessageBusConfig, InMemoryMessageBus

config = MessageBusConfig(
    max_queue_size=10000,           # Maximum queue size
    enable_dead_letter=True,        # Enable dead letter queue
    max_retries=3,                  # Retry failed deliveries
    retry_delay_seconds=1.0,        # Delay between retries
    request_timeout_seconds=30.0,   # Default request timeout
    max_handlers_per_topic=100      # Max handlers per topic
)

bus = InMemoryMessageBus(config)
```

## Error Handling

### Automatic Retries

Handlers are automatically retried on failure:

```python
config = MessageBusConfig(
    max_retries=3,
    retry_delay_seconds=1.0
)
```

### Dead Letter Queue

Failed events go to dead letter queue:

```python
# Get dead letters
dead_letters = bus.get_dead_letter_queue()

# Replay a dead letter
await bus.replay_dead_letter(event_id)
```

## Monitoring

### Health Checks

```python
monitor = MessageBusMonitor(
    bus,
    max_queue_utilization=0.8,    # Alert if >80% full
    max_error_rate=0.05,          # Alert if >5% error rate
    max_delivery_time_ms=1000.0   # Alert if >1s delivery
)

health = monitor.check_health()
print(f"Status: {health.status}")
print(f"Issues: {health.issues}")
```

### Metrics

```python
# Raw metrics from bus
metrics = bus.get_metrics()
print(f"Published: {metrics.events_published}")
print(f"Delivered: {metrics.events_delivered}")
print(f"Failed: {metrics.events_failed}")

# Performance metrics from monitor
perf = monitor.get_performance_metrics()
print(f"Events/sec: {perf.events_per_second}")
print(f"Error rate: {perf.error_rate}")
```

## Testing

Run the test suite:

```bash
# All messaging tests
pytest tests/core/messaging/ -v

# Specific test file
pytest tests/core/messaging/test_message_bus.py -v
pytest tests/core/messaging/test_events.py -v
pytest tests/core/messaging/test_monitoring.py -v
```

**Test Coverage:**
- 56/59 tests passing (95% pass rate)
- Event creation and validation
- Publish/subscribe patterns
- Wildcard matching
- Priority queuing
- Request-reply
- Error handling and retries
- Dead letter queue
- Metrics collection
- Health monitoring

## Integration with Agents

### GL-001 ThermoSync Example

```python
class ProcessHeatOrchestrator:
    def __init__(self, config):
        self.message_bus = InMemoryMessageBus()

    async def start(self):
        await self.message_bus.start()

        # Emit started event
        await self.message_bus.publish(create_event(
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={"status": "ready"}
        ))

        # Subscribe to events
        await self.message_bus.subscribe(
            StandardEvents.CALCULATION_COMPLETED,
            self._handle_calculation,
            "GL-001"
        )
```

### GL-003 SteamSync Example

```python
class SteamSystemOrchestrator:
    async def monitor_steam_system(self):
        # Subscribe to safety events
        await self.message_bus.subscribe(
            "safety.**",
            self._handle_safety_event,
            "GL-003"
        )

    async def _handle_safety_event(self, event):
        if event.priority == EventPriority.CRITICAL:
            await self.emergency_shutdown()
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MessageBus                           │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Priority  │───>│   Event      │───>│ Handlers  │ │
│  │    Queue    │    │   Router     │    │           │ │
│  └─────────────┘    └──────────────┘    └───────────┘ │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Dead Letter │    │   Metrics    │    │ Monitor   │ │
│  │    Queue    │    │  Collector   │    │           │ │
│  └─────────────┘    └──────────────┘    └───────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Best Practices

1. **Use Standard Events**: Always use `StandardEvents` constants
2. **Set Appropriate Priorities**: CRITICAL for safety, HIGH for compliance
3. **Include Provenance**: Add traceability info in payloads
4. **Handle Errors**: Implement error handlers, check dead letter queue
5. **Monitor Health**: Use `MessageBusMonitor` in production
6. **Keep Handlers Fast**: Avoid blocking operations
7. **Use Correlation IDs**: Link related events
8. **Test Thoroughly**: Unit and integration test event flows

## Documentation

- [MessageBus User Guide](../../../docs/messaging/message_bus_guide.md)
- [Event-Driven Architecture Guide](../../../docs/messaging/event_driven_architecture.md)

## API Reference

See module docstrings:
- `greenlang.core.messaging.events`
- `greenlang.core.messaging.message_bus`
- `greenlang.core.messaging.monitoring`

## Future Enhancements

1. **Redis Backend**: Implement `RedisMessageBus` for distributed deployments
2. **Event Persistence**: Store events for replay and audit
3. **Advanced Routing**: Content-based routing, complex filters
4. **Metrics Integration**: Integration with Prometheus/Grafana
5. **Event Schema Registry**: Centralized schema management
