# GreenLang Orchestration Patterns

This document describes the orchestration patterns available in the GreenLang framework for building multi-agent systems. These patterns are implemented in the `greenlang.core` module and provide standardized approaches for agent coordination, task distribution, messaging, and safety monitoring.

## Overview

The GreenLang orchestration infrastructure consists of four core components:

1. **BaseOrchestrator** - Abstract base class for custom orchestrators
2. **MessageBus** - Event-driven async messaging
3. **TaskScheduler** - Task scheduling with load balancing
4. **CoordinationLayer** - Multi-agent coordination patterns
5. **SafetyMonitor** - Safety oversight and constraint validation

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           BaseOrchestrator              │
                    │  - Lifecycle management                 │
                    │  - Execution tracking                   │
                    │  - Provenance hashing                   │
                    └────────────────┬────────────────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
       ▼                             ▼                             ▼
┌──────────────┐           ┌──────────────────┐           ┌──────────────┐
│  MessageBus  │           │  TaskScheduler   │           │ Coordination │
│  - Pub/Sub   │           │  - Load balance  │           │    Layer     │
│  - Priority  │           │  - Retry logic   │           │  - Locking   │
│  - Dead LQ   │           │  - Timeout       │           │  - Sagas     │
└──────────────┘           └──────────────────┘           └──────────────┘
                                     │
                           ┌─────────┴─────────┐
                           │   SafetyMonitor   │
                           │  - Constraints    │
                           │  - Circuit break  │
                           │  - Rate limiting  │
                           └───────────────────┘
```

## BaseOrchestrator

The `BaseOrchestrator` is an abstract base class that provides the foundation for building custom orchestrators. It integrates all orchestration components and provides lifecycle management.

### Usage

```python
from greenlang.core import (
    BaseOrchestrator,
    OrchestratorConfig,
    MessageBus,
    MessageBusConfig,
    TaskScheduler,
    TaskSchedulerConfig,
    CoordinationLayer,
    CoordinationConfig,
    SafetyMonitor,
    SafetyConfig,
)

class MyOrchestrator(BaseOrchestrator[Dict, Dict]):
    """Custom orchestrator implementation."""

    def _create_message_bus(self) -> MessageBus:
        return MessageBus(MessageBusConfig())

    def _create_task_scheduler(self) -> TaskScheduler:
        return TaskScheduler(TaskSchedulerConfig())

    def _create_coordinator(self) -> CoordinationLayer:
        return CoordinationLayer(CoordinationConfig())

    def _create_safety_monitor(self) -> SafetyMonitor:
        return SafetyMonitor(SafetyConfig())

    async def orchestrate(self, input_data: Dict) -> Dict:
        # Your orchestration logic here
        result = await self._process_data(input_data)
        return result

# Usage
config = OrchestratorConfig(
    orchestrator_id="my-orchestrator",
    name="MyOrchestrator",
    enable_safety_monitoring=True,
)
orchestrator = MyOrchestrator(config)
await orchestrator.start()

result = await orchestrator.execute({"key": "value"})
print(result.success)  # True
print(result.provenance_hash)  # SHA-256 hash

await orchestrator.shutdown()
```

### Key Features

- **Generic typing**: `BaseOrchestrator[InT, OutT]` supports typed input/output
- **Lifecycle management**: `start()`, `shutdown()` methods
- **Safety validation**: Automatic validation before execution
- **Provenance tracking**: SHA-256 hash for audit trails
- **Metrics collection**: Execution time, success/failure rates

## MessageBus

The `MessageBus` provides event-driven async messaging with topic-based routing and priority queuing.

### Topic Patterns

The message bus supports wildcard patterns for topic subscriptions:

- `*` - Matches a single level (e.g., `agent.*` matches `agent.thermal`)
- `#` - Matches multiple levels (e.g., `agent.#` matches `agent.thermal.status`)

### Usage

```python
from greenlang.core import (
    MessageBus,
    MessageBusConfig,
    Message,
    MessageType,
    MessagePriority,
)

# Create message bus
config = MessageBusConfig(
    max_queue_size=10000,
    enable_dead_letter=True,
    max_retries=3,
)
bus = MessageBus(config)
await bus.start()

# Subscribe to topic
async def handle_thermal(msg: Message):
    print(f"Received: {msg.payload}")

await bus.subscribe("agent.thermal.*", handle_thermal, "my-subscriber")

# Publish message
message = Message(
    sender_id="orchestrator",
    recipient_id="thermal-agent",
    message_type=MessageType.COMMAND,
    topic="agent.thermal.optimize",
    payload={"target_temp": 450},
    priority=MessagePriority.HIGH,
)
await bus.publish(message)

# Request-response pattern
response = await bus.request_response(message, timeout_seconds=30.0)

await bus.close()
```

### Message Types

- `COMMAND` - Direct commands to agents
- `EVENT` - Broadcast events
- `QUERY` - Information requests
- `RESPONSE` - Query responses
- `HEARTBEAT` - Health checks
- `ERROR` - Error notifications

### Priority Levels

- `CRITICAL` - Highest priority
- `HIGH` - High priority
- `NORMAL` - Default priority
- `LOW` - Low priority

## TaskScheduler

The `TaskScheduler` provides task distribution with multiple load balancing strategies.

### Load Balancing Strategies

1. **ROUND_ROBIN** - Distribute tasks in rotation
2. **LEAST_LOADED** - Assign to agent with lowest load
3. **PRIORITY_WEIGHTED** - Consider performance scores
4. **CAPABILITY_MATCH** - Match task type to agent capabilities
5. **RANDOM** - Random distribution

### Usage

```python
from greenlang.core import (
    TaskScheduler,
    TaskSchedulerConfig,
    Task,
    TaskPriority,
    AgentCapacity,
    LoadBalanceStrategy,
)

# Create scheduler
config = TaskSchedulerConfig(
    load_balance_strategy=LoadBalanceStrategy.LEAST_LOADED,
    max_concurrent_tasks=100,
    default_timeout_seconds=60.0,
)
scheduler = TaskScheduler(config)
await scheduler.start()

# Register agents
scheduler.register_agent(AgentCapacity(
    agent_id="agent-1",
    capabilities={"thermal_calculation", "energy_balance"},
    max_concurrent_tasks=10,
))

# Register executor
async def thermal_executor(task: Task) -> Any:
    # Execute task
    return {"result": "calculated"}

scheduler.register_executor("thermal_calculation", thermal_executor)

# Schedule task
task = Task(
    task_type="thermal_calculation",
    payload={"temperature": 450},
    priority=TaskPriority.HIGH,
    timeout_seconds=30.0,
)
task_id = await scheduler.schedule(task)

# Wait for completion
result = await scheduler.wait_for_completion(task_id, timeout=60.0)

await scheduler.stop()
```

### Task States

- `PENDING` - Waiting to be scheduled
- `SCHEDULED` - Assigned to queue
- `RUNNING` - Being executed
- `COMPLETED` - Successfully completed
- `FAILED` - Execution failed
- `TIMEOUT` - Exceeded timeout
- `CANCELLED` - Cancelled by user
- `RETRYING` - Being retried

## CoordinationLayer

The `CoordinationLayer` provides patterns for coordinating multiple agents.

### Coordination Patterns

1. **MASTER_SLAVE** - Hierarchical coordination
2. **PEER_TO_PEER** - Equal peer coordination
3. **CONSENSUS** - Voting-based decisions
4. **CHOREOGRAPHY** - Event-driven coordination
5. **ORCHESTRATION** - Central control (default)

### Distributed Locking

```python
from greenlang.core import CoordinationLayer, CoordinationConfig

coordinator = CoordinationLayer(CoordinationConfig())

# Acquire lock
async with coordinator.acquire_lock("resource-1", "my-agent"):
    # Critical section
    await perform_exclusive_operation()
```

### Saga Transactions

The saga pattern enables long-running transactions with compensation:

```python
from greenlang.core import Saga, SagaStep

saga = Saga(
    name="thermal_optimization",
    steps=[
        SagaStep(
            name="shutdown_unit",
            agent_id="thermal-agent",
            action={"command": "shutdown", "unit_id": "unit-1"},
            compensation={"command": "startup", "unit_id": "unit-1"},
        ),
        SagaStep(
            name="optimize_settings",
            agent_id="optimizer-agent",
            action={"command": "optimize", "params": {...}},
            compensation={"command": "restore", "params": {...}},
        ),
    ]
)

result = await coordinator.run_saga(saga)
```

If any step fails, compensation actions run in reverse order.

### Consensus Voting

```python
# Create proposal
proposal = await coordinator.propose_consensus(
    topic="increase_temperature",
    proposed_by="orchestrator",
    data={"new_temp": 500},
    required_approval=0.5,  # 50% approval required
)

# Agents vote
await coordinator.vote_on_proposal(proposal.proposal_id, "agent-1", "approve")
await coordinator.vote_on_proposal(proposal.proposal_id, "agent-2", "approve")
await coordinator.vote_on_proposal(proposal.proposal_id, "agent-3", "reject")

# Resolve
result = await coordinator.resolve_consensus(proposal.proposal_id)
# result: ConsensusResult.ACHIEVED
```

## SafetyMonitor

The `SafetyMonitor` provides safety oversight with constraints, rate limiting, and circuit breakers.

### Safety Constraints

```python
from greenlang.core import (
    SafetyMonitor,
    SafetyConfig,
    SafetyConstraint,
    ConstraintType,
    SafetyLevel,
    OperationContext,
)

monitor = SafetyMonitor(SafetyConfig(halt_on_critical=True))

# Add threshold constraint
monitor.add_constraint(SafetyConstraint(
    name="max_temperature",
    constraint_type=ConstraintType.THRESHOLD,
    max_value=600.0,
    level=SafetyLevel.CRITICAL,
    metadata={"parameter": "temperature"},
))

# Validate operation
context = OperationContext(
    operation_type="set_temperature",
    agent_id="thermal-agent",
    parameters={"temperature": 550},
)
result = await monitor.validate_operation(context)

if result.is_safe:
    await execute_operation()
else:
    print(f"Violations: {result.violations}")
```

### Circuit Breakers

Circuit breakers protect against cascading failures:

```python
# Add circuit breaker
monitor.add_circuit_breaker(
    name="scada_integration",
    failure_threshold=5,  # Open after 5 failures
    timeout_seconds=60.0,  # Try again after 60s
)

# Record results
monitor.record_success("agent-id", "scada_integration")
monitor.record_failure("agent-id", "scada_integration")

# Check status
status = monitor.get_circuit_breaker_status()
```

Circuit breaker states:
- `CLOSED` - Normal operation
- `OPEN` - Blocking calls after failures
- `HALF_OPEN` - Testing if service recovered

### Rate Limiting

```python
# Add rate limiter
monitor.add_rate_limiter(
    key="api_calls",
    max_requests=100,
    window_seconds=60.0,
)
```

## Best Practices

### 1. Use Type Safety

Always specify input/output types for your orchestrator:

```python
class MyOrchestrator(BaseOrchestrator[ProcessInput, ProcessOutput]):
    ...
```

### 2. Enable Safety Monitoring

Always enable safety monitoring in production:

```python
config = OrchestratorConfig(
    enable_safety_monitoring=True,
)
```

### 3. Define Domain-Specific Constraints

Add constraints relevant to your domain:

```python
def _add_constraints(self):
    self.safety_monitor.add_constraint(SafetyConstraint(
        name="max_pressure",
        constraint_type=ConstraintType.THRESHOLD,
        max_value=50.0,
        level=SafetyLevel.CRITICAL,
    ))
```

### 4. Use Sagas for Multi-Step Operations

When operations span multiple agents, use sagas for transactional integrity:

```python
saga = Saga(
    name="multi_step_operation",
    steps=[...],
)
result = await coordinator.run_saga(saga)
```

### 5. Handle Errors Gracefully

Override `_handle_error_recovery` for custom recovery logic:

```python
async def _handle_error_recovery(self, error, input_data, execution_id):
    # Custom recovery logic
    if isinstance(error, TimeoutError):
        return await self._retry_with_backoff(input_data)
    return None
```

### 6. Track Provenance

All results include provenance hashes for audit trails:

```python
result = await orchestrator.execute(input_data)
audit_log.record(result.provenance_hash)
```

## Example: Process Heat Orchestrator

See `GL-001/process_heat_orchestrator.py` for a complete example of a production orchestrator implementing these patterns.

Key features demonstrated:
- Custom safety constraints for industrial operations
- SCADA/ERP integration
- Thermal efficiency calculations
- Heat distribution optimization
- Emissions compliance checking
- KPI dashboard generation

## API Reference

For detailed API documentation, see the module docstrings in:
- `greenlang/core/base_orchestrator.py`
- `greenlang/core/message_bus.py`
- `greenlang/core/task_scheduler.py`
- `greenlang/core/coordination_layer.py`
- `greenlang/core/safety_monitor.py`
