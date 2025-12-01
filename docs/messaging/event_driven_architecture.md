# Event-Driven Architecture Guide

## Introduction

GreenLang's event-driven architecture enables loosely coupled, scalable agent communication. This guide explains the architectural patterns, design principles, and implementation strategies for building event-driven GreenLang applications.

## Core Concepts

### Event-Driven vs. Direct Communication

**Traditional Direct Communication:**
```python
# ❌ Tight coupling
class AgentA:
    def process(self):
        result = agent_b.calculate(data)  # Direct dependency
        agent_c.store(result)             # Direct dependency
        agent_d.notify(result)            # Direct dependency
```

**Event-Driven Communication:**
```python
# ✅ Loose coupling
class AgentA:
    async def process(self):
        result = self.calculate(data)

        # Emit event - no knowledge of consumers
        await self.bus.publish(create_event(
            event_type=StandardEvents.CALCULATION_COMPLETED,
            source_agent="agent-a",
            payload={"result": result}
        ))

# Other agents subscribe independently
class AgentB:
    async def setup(self):
        await self.bus.subscribe(
            StandardEvents.CALCULATION_COMPLETED,
            self.handle_result,
            "agent-b"
        )
```

**Benefits:**
- Agents don't need to know about each other
- Easy to add/remove consumers
- Better testability and modularity
- Natural scalability

## Architectural Patterns

### 1. Publish-Subscribe Pattern

**Use Case:** Broadcasting events to multiple interested parties

**Example: Sensor Data Distribution**
```python
class SensorDataPublisher:
    """Publishes sensor readings to all interested agents."""

    async def publish_reading(self, sensor_id: str, value: float):
        await self.bus.publish(create_event(
            event_type=StandardEvents.INTEGRATION_DATA_RECEIVED,
            source_agent="sensor-gateway",
            payload={
                "sensor_id": sensor_id,
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "unit": "celsius"
            }
        ))

class TemperatureMonitor:
    """Subscribes to temperature data for monitoring."""

    async def setup(self):
        await self.bus.subscribe(
            StandardEvents.INTEGRATION_DATA_RECEIVED,
            self.check_temperature,
            "temp-monitor",
            filter_fn=lambda e: e.payload.get("sensor_id") == "temp-001"
        )

    async def check_temperature(self, event):
        temp = event.payload["value"]
        if temp > self.threshold:
            await self.bus.publish(create_event(
                event_type=StandardEvents.SAFETY_ALERT,
                source_agent="temp-monitor",
                payload={"temperature": temp, "threshold": self.threshold},
                priority=EventPriority.HIGH
            ))

class DataLogger:
    """Subscribes to all sensor data for logging."""

    async def setup(self):
        await self.bus.subscribe(
            StandardEvents.INTEGRATION_DATA_RECEIVED,
            self.log_data,
            "data-logger"
        )

    async def log_data(self, event):
        await self.db.insert(event.to_dict())
```

### 2. Request-Reply Pattern

**Use Case:** Synchronous request-response communication

**Example: Calculation Service**
```python
class CalculationRequester:
    """Requests calculations and waits for results."""

    async def calculate_efficiency(self, input_data: dict) -> float:
        # Create request event
        request = create_event(
            event_type="calculation.efficiency.request",
            source_agent="orchestrator",
            payload={"input": input_data}
        )

        # Send request and wait for reply
        reply = await self.bus.request_reply(request, timeout_seconds=5.0)

        if reply:
            return reply.payload["efficiency"]
        else:
            raise TimeoutError("Calculation service did not respond")

class CalculationService:
    """Responds to calculation requests."""

    async def setup(self):
        await self.bus.subscribe(
            "calculation.efficiency.request",
            self.handle_request,
            "calc-service"
        )

    async def handle_request(self, event):
        # Perform calculation
        input_data = event.payload["input"]
        efficiency = self._calculate_efficiency(input_data)

        # Send reply
        reply = create_event(
            event_type="calculation.efficiency.reply",
            source_agent="calc-service",
            payload={
                "efficiency": efficiency,
                "calculation_id": str(uuid.uuid4())
            },
            correlation_id=event.event_id  # Link to request
        )
        await self.bus.publish(reply)
```

### 3. Event Sourcing Pattern

**Use Case:** Maintaining complete audit trail and state reconstruction

**Example: Process State Management**
```python
class ProcessStateManager:
    """Maintains process state via event sourcing."""

    def __init__(self):
        self.events: List[Event] = []
        self.current_state = {}

    async def setup(self):
        # Subscribe to all process events
        await self.bus.subscribe(
            "process.**",
            self.record_event,
            "state-manager"
        )

    async def record_event(self, event):
        # Store event
        self.events.append(event)
        await self.db.store_event(event.to_dict())

        # Update state
        self._apply_event(event)

    def _apply_event(self, event):
        """Apply event to current state."""
        if event.event_type == "process.started":
            self.current_state["status"] = "running"
            self.current_state["start_time"] = event.timestamp

        elif event.event_type == "process.parameter_changed":
            param = event.payload["parameter"]
            value = event.payload["value"]
            self.current_state[param] = value

        elif event.event_type == "process.stopped":
            self.current_state["status"] = "stopped"
            self.current_state["end_time"] = event.timestamp

    async def reconstruct_state(self, up_to_timestamp: str):
        """Reconstruct state at a specific point in time."""
        state = {}
        for event in self.events:
            if event.timestamp <= up_to_timestamp:
                self._apply_event_to_state(event, state)
            else:
                break
        return state
```

### 4. Saga Pattern (Distributed Transactions)

**Use Case:** Coordinating multi-step workflows across agents

**Example: Equipment Commissioning Workflow**
```python
class CommissioningSaga:
    """Coordinates multi-step commissioning workflow."""

    def __init__(self, equipment_id: str):
        self.equipment_id = equipment_id
        self.saga_id = str(uuid.uuid4())
        self.state = "initialized"

    async def execute(self):
        """Execute commissioning saga."""
        try:
            # Step 1: Validate equipment
            await self._validate_equipment()

            # Step 2: Configure parameters
            await self._configure_parameters()

            # Step 3: Run tests
            await self._run_tests()

            # Step 4: Activate equipment
            await self._activate()

            # Success event
            await self.bus.publish(create_event(
                event_type=StandardEvents.WORKFLOW_COMPLETED,
                source_agent="commissioning-saga",
                payload={
                    "saga_id": self.saga_id,
                    "equipment_id": self.equipment_id
                },
                correlation_id=self.saga_id
            ))

        except Exception as e:
            # Compensate on failure
            await self._compensate()
            raise

    async def _validate_equipment(self):
        """Step 1: Validate equipment."""
        request = create_event(
            event_type="equipment.validate.request",
            source_agent="commissioning-saga",
            payload={"equipment_id": self.equipment_id},
            correlation_id=self.saga_id
        )

        reply = await self.bus.request_reply(request, timeout_seconds=10.0)
        if not reply or not reply.payload.get("valid"):
            raise ValueError("Equipment validation failed")

        self.state = "validated"

    async def _compensate(self):
        """Rollback changes on failure."""
        if self.state in ("configured", "tested"):
            # Revert configuration
            await self.bus.publish(create_event(
                event_type="equipment.revert.request",
                source_agent="commissioning-saga",
                payload={"equipment_id": self.equipment_id}
            ))
```

### 5. CQRS Pattern (Command Query Responsibility Segregation)

**Use Case:** Separating read and write operations

**Example: Data Management**
```python
class CommandHandler:
    """Handles write operations via commands."""

    async def setup(self):
        await self.bus.subscribe(
            "command.**",
            self.handle_command,
            "command-handler"
        )

    async def handle_command(self, event):
        command_type = event.event_type

        if command_type == "command.update_setpoint":
            # Execute command
            equipment_id = event.payload["equipment_id"]
            new_setpoint = event.payload["setpoint"]

            await self.db.update_setpoint(equipment_id, new_setpoint)

            # Emit event for query side
            await self.bus.publish(create_event(
                event_type=StandardEvents.DATA_TRANSFORMED,
                source_agent="command-handler",
                payload={
                    "equipment_id": equipment_id,
                    "setpoint": new_setpoint
                }
            ))

class QueryHandler:
    """Handles read operations via queries."""

    def __init__(self):
        self.read_model = {}  # Optimized for queries

    async def setup(self):
        # Subscribe to events to update read model
        await self.bus.subscribe(
            StandardEvents.DATA_TRANSFORMED,
            self.update_read_model,
            "query-handler"
        )

    async def update_read_model(self, event):
        """Update optimized read model."""
        equipment_id = event.payload["equipment_id"]
        setpoint = event.payload["setpoint"]

        self.read_model[equipment_id] = {
            "setpoint": setpoint,
            "last_updated": event.timestamp
        }

    async def get_equipment_status(self, equipment_id: str) -> dict:
        """Query optimized read model."""
        return self.read_model.get(equipment_id, {})
```

## Event Design Guidelines

### Event Naming Conventions

Use hierarchical, descriptive names:

```python
# ✅ Good: Clear hierarchy
"agent.started"
"calculation.efficiency.started"
"integration.erp.connection_lost"
"safety.temperature.limit_exceeded"

# ❌ Bad: Flat, unclear
"start"
"calc"
"lost"
"hot"
```

### Event Payload Design

Include sufficient context, avoid redundancy:

```python
# ✅ Good: Complete, structured payload
{
    "equipment_id": "HX-001",
    "parameter": "temperature",
    "value": 450.5,
    "unit": "celsius",
    "timestamp": "2025-12-01T10:30:00Z",
    "sensor_id": "TC-001",
    "quality": "good"
}

# ❌ Bad: Incomplete, unstructured
{
    "val": 450.5,
    "eq": "HX-001"
}
```

### Event Versioning

Plan for schema evolution:

```python
# Version 1
event_v1 = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="calc-engine",
    payload={
        "result": 42.5,
        "schema_version": "1.0"
    }
)

# Version 2: Add new fields, keep backward compatibility
event_v2 = create_event(
    event_type=StandardEvents.CALCULATION_COMPLETED,
    source_agent="calc-engine",
    payload={
        "result": 42.5,
        "confidence": 0.95,      # New field
        "methodology": "ASME",    # New field
        "schema_version": "2.0"
    }
)

# Handlers check version
async def handle_calculation(event):
    version = event.payload.get("schema_version", "1.0")

    if version == "1.0":
        result = event.payload["result"]
        confidence = 1.0  # Default for v1
    elif version == "2.0":
        result = event.payload["result"]
        confidence = event.payload["confidence"]
```

## Orchestration Patterns

### Centralized Orchestration

**Use Case:** Complex workflows with strict ordering

```python
class WorkflowOrchestrator:
    """Centralized orchestrator for multi-step workflow."""

    async def execute_workflow(self, workflow_id: str):
        # Step 1
        await self.bus.publish(create_event(
            event_type=StandardEvents.TASK_ASSIGNED,
            source_agent="orchestrator",
            payload={"task": "validate", "workflow_id": workflow_id},
            target_agent="validator"
        ))

        # Wait for completion
        await self._wait_for_task_completion(workflow_id, "validate")

        # Step 2
        await self.bus.publish(create_event(
            event_type=StandardEvents.TASK_ASSIGNED,
            source_agent="orchestrator",
            payload={"task": "calculate", "workflow_id": workflow_id},
            target_agent="calculator"
        ))

        await self._wait_for_task_completion(workflow_id, "calculate")

        # Step 3
        await self.bus.publish(create_event(
            event_type=StandardEvents.TASK_ASSIGNED,
            source_agent="orchestrator",
            payload={"task": "report", "workflow_id": workflow_id},
            target_agent="reporter"
        ))
```

### Choreographed Orchestration

**Use Case:** Loosely coupled, reactive workflows

```python
class Validator:
    """Validates data and triggers next step."""

    async def setup(self):
        await self.bus.subscribe(
            StandardEvents.DATA_RECEIVED,
            self.validate,
            "validator"
        )

    async def validate(self, event):
        is_valid = self._validate_data(event.payload)

        if is_valid:
            # Emit validation success - calculator will react
            await self.bus.publish(create_event(
                event_type=StandardEvents.DATA_VALIDATED,
                source_agent="validator",
                payload=event.payload,
                correlation_id=event.correlation_id
            ))

class Calculator:
    """Calculates when validation succeeds."""

    async def setup(self):
        # React to validation success
        await self.bus.subscribe(
            StandardEvents.DATA_VALIDATED,
            self.calculate,
            "calculator"
        )

    async def calculate(self, event):
        result = self._perform_calculation(event.payload)

        # Emit calculation result - reporter will react
        await self.bus.publish(create_event(
            event_type=StandardEvents.CALCULATION_COMPLETED,
            source_agent="calculator",
            payload={"result": result},
            correlation_id=event.correlation_id
        ))
```

## Error Handling Strategies

### Dead Letter Queue Pattern

```python
class DeadLetterProcessor:
    """Processes failed events from dead letter queue."""

    async def process_dead_letters(self):
        dlq = self.bus.get_dead_letter_queue()

        for event in dlq:
            # Analyze failure
            failure_reason = self._diagnose_failure(event)

            # Log for investigation
            await self.bus.publish(create_event(
                event_type=StandardEvents.AUDIT_LOG_CREATED,
                source_agent="dlq-processor",
                payload={
                    "event_id": event.event_id,
                    "failure_reason": failure_reason,
                    "original_event": event.to_dict()
                }
            ))

            # Attempt recovery if possible
            if self._can_recover(event, failure_reason):
                await self.bus.replay_dead_letter(event.event_id)
```

### Circuit Breaker Pattern

```python
class CircuitBreakerHandler:
    """Implements circuit breaker for event handlers."""

    def __init__(self, failure_threshold: int = 5):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.state = "closed"  # closed, open, half-open

    async def handle_with_circuit_breaker(self, event):
        if self.state == "open":
            # Circuit open - reject immediately
            await self.bus.publish(create_event(
                event_type=StandardEvents.AGENT_ERROR,
                source_agent="circuit-breaker",
                payload={
                    "error": "Circuit breaker open",
                    "failed_event_id": event.event_id
                }
            ))
            return

        try:
            # Attempt processing
            await self._process_event(event)

            # Success - reset failure count
            if self.state == "half-open":
                self.state = "closed"
            self.failure_count = 0

        except Exception as e:
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                await self._trip_circuit_breaker()
            else:
                raise
```

## Performance Optimization

### Batching Events

```python
class EventBatcher:
    """Batches events for efficient processing."""

    def __init__(self, batch_size: int = 100, batch_timeout: float = 5.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.batch = []
        self.last_flush = time.time()

    async def add_event(self, event):
        self.batch.append(event)

        # Flush if batch full or timeout reached
        if len(self.batch) >= self.batch_size:
            await self._flush_batch()
        elif time.time() - self.last_flush >= self.batch_timeout:
            await self._flush_batch()

    async def _flush_batch(self):
        if not self.batch:
            return

        # Process batch
        await self.bus.publish(create_event(
            event_type="batch.processed",
            source_agent="batcher",
            payload={
                "batch_size": len(self.batch),
                "events": [e.to_dict() for e in self.batch]
            }
        ))

        self.batch = []
        self.last_flush = time.time()
```

### Event Filtering

```python
# Filter at subscription level
def high_value_filter(event):
    """Only process high-value events."""
    return event.payload.get("value", 0) > 1000

await bus.subscribe(
    StandardEvents.DATA_RECEIVED,
    handler,
    "high-value-processor",
    filter_fn=high_value_filter
)
```

## Testing Strategies

### Unit Testing Events

```python
import pytest

@pytest.mark.asyncio
async def test_event_handler():
    """Test event handler in isolation."""
    # Arrange
    handler = MyEventHandler()
    test_event = create_event(
        event_type=StandardEvents.CALCULATION_COMPLETED,
        source_agent="test",
        payload={"result": 42}
    )

    # Act
    await handler.handle_event(test_event)

    # Assert
    assert handler.last_result == 42
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_event_flow():
    """Test complete event flow."""
    bus = InMemoryMessageBus()
    await bus.start()

    received_events = []

    async def collector(event):
        received_events.append(event)

    # Subscribe
    await bus.subscribe(
        StandardEvents.CALCULATION_COMPLETED,
        collector,
        "test-collector"
    )

    # Publish
    event = create_event(
        event_type=StandardEvents.CALCULATION_COMPLETED,
        source_agent="calculator",
        payload={"result": 42}
    )
    await bus.publish(event)

    # Wait for processing
    await asyncio.sleep(0.1)

    # Verify
    assert len(received_events) == 1
    assert received_events[0].payload["result"] == 42

    await bus.close()
```

## Best Practices Summary

1. **Use Standard Events**: Always use `StandardEvents` constants
2. **Set Priorities**: Use appropriate `EventPriority` levels
3. **Include Correlation IDs**: Link related events
4. **Design for Idempotency**: Handlers should be idempotent
5. **Monitor Bus Health**: Use `MessageBusMonitor` in production
6. **Handle Errors**: Implement retry and dead letter strategies
7. **Keep Handlers Fast**: Avoid blocking operations
8. **Version Payloads**: Include schema versions for evolution
9. **Test Thoroughly**: Unit and integration test event flows
10. **Document Events**: Maintain event catalog documentation

## See Also

- [MessageBus User Guide](message_bus_guide.md)
- [GreenLang Architecture Overview](../architecture/README.md)
- [Agent Development Guide](../guides/agent_development.md)
