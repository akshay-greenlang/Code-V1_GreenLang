# GreenLang Messaging System - Quick Reference Card

## Installation

```bash
pip install redis[hiredis] pydantic pyyaml msgpack
redis-server  # Start Redis
```

## 1. Basic Publish/Subscribe

```python
from messaging import RedisStreamsBroker

# Connect
broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
await broker.connect()

# Publish
msg_id = await broker.publish("agent.tasks", {"task": "analyze_esg"})

# Consume
async for message in broker.consume("agent.tasks", "workers"):
    print(message.payload)
    await broker.acknowledge(message)
```

## 2. Batch Publishing (80% Faster)

```python
# Publish 100 messages at once
payloads = [{"task": f"task_{i}"} for i in range(100)]
msg_ids = await broker.publish_batch("agent.tasks", payloads)
```

## 3. Request-Reply Pattern

```python
from messaging import RequestReplyPattern

pattern = RequestReplyPattern(broker)

# Send request, wait for response
response = await pattern.send_request(
    "agent.llm",
    {"prompt": "Analyze ESG report"},
    timeout=30.0
)
print(response.payload)
```

## 4. Work Queue (Multiple Workers)

```python
from messaging import WorkQueuePattern

pattern = WorkQueuePattern(broker)

# Submit tasks
await pattern.submit_batch("agent.tasks", tasks)

# Process with 10 workers
def handler(payload):
    return process(payload)

await pattern.process_tasks("agent.tasks", handler, num_workers=10)
```

## 5. Pub-Sub (Broadcast)

```python
from messaging import PubSubPattern

pattern = PubSubPattern(broker)

# Subscribe to events
async def event_handler(message):
    print(f"Event: {message.payload}")

await pattern.subscribe("agent.events.*", event_handler)

# Publish event
await pattern.publish("agent.events.complete", {"status": "done"})
```

## 6. Saga Pattern (Distributed Transactions)

```python
from messaging import SagaPattern

saga = SagaPattern(broker)

# Add steps with compensation
saga.add_step("validate", validate_fn, compensate_validate)
saga.add_step("calculate", calc_fn, compensate_calc)
saga.add_step("save", save_fn, compensate_save)

# Execute (auto-compensation on failure)
result = await saga.execute({"data": ...})
```

## 7. Circuit Breaker (Fault Tolerance)

```python
from messaging import CircuitBreakerPattern

breaker = CircuitBreakerPattern(failure_threshold=5, timeout_seconds=60)

# Call external service with protection
try:
    result = await breaker.call(external_service_call)
except CircuitBreakerError:
    result = fallback_logic()
```

## 8. Dead Letter Queue

```python
# Failed messages go to DLQ after max retries
async for message in broker.consume("agent.tasks", "workers"):
    try:
        process(message)
        await broker.acknowledge(message)
    except Exception as e:
        await broker.nack(message, str(e), requeue=True)

# Check DLQ
dlq_msgs = await broker.get_dead_letter_messages("agent.tasks")

# Reprocess
await broker.reprocess_dead_letter_message(dlq_msgs[0])
```

## 9. Message Priorities

```python
from messaging import MessagePriority

# High priority (processed first)
await broker.publish("agent.tasks", {...}, priority=MessagePriority.HIGH)

# Normal priority
await broker.publish("agent.tasks", {...}, priority=MessagePriority.NORMAL)

# Low priority
await broker.publish("agent.tasks", {...}, priority=MessagePriority.LOW)
```

## 10. Configuration

```python
from messaging import MessagingConfig

# From YAML
config = MessagingConfig.from_yaml("config/messaging.yaml")

# From environment
config = MessagingConfig.from_env()

# Create broker
broker = RedisStreamsBroker(**config.redis.to_dict())
```

## 11. Monitoring

```python
# Get metrics
metrics = broker.get_metrics()
print(f"Published: {metrics['messages_published']}")
print(f"Throughput: {metrics['throughput_per_second']}")

# Health check
health = await broker.health_check()
print(f"Status: {health['status']}, Latency: {health['latency_ms']}ms")

# Consumer lag
lag = await broker.get_consumer_lag("agent.tasks", "workers")
print(f"Pending: {lag} messages")
```

## 12. Error Handling

```python
async for message in broker.consume("agent.tasks", "workers"):
    try:
        result = await process(message.payload)
        await broker.acknowledge(message)

    except TemporaryError as e:
        # Retry (up to max_retries)
        await broker.nack(message, str(e), requeue=True)

    except PermanentError as e:
        # Move to DLQ (no retry)
        await broker.nack(message, str(e), requeue=False)
```

## 13. Message TTL (Expiration)

```python
# Message expires after 1 hour
await broker.publish(
    "agent.tasks",
    {"task": "time_sensitive"},
    ttl_seconds=3600
)
```

## 14. Consumer Groups

```python
# Create consumer group
await broker.create_consumer_group("agent.tasks", "group1")

# Multiple consumers in same group (load balancing)
async for msg in broker.consume("agent.tasks", "group1", consumer_id="c1"):
    await process(msg)

async for msg in broker.consume("agent.tasks", "group1", consumer_id="c2"):
    await process(msg)
```

## 15. Event Sourcing

```python
from messaging import EventSourcingPattern

pattern = EventSourcingPattern(broker)

# Log events for audit trail
await pattern.log_event(
    "calculation",
    {
        "input": {"activity_data": 1000},
        "output": {"emissions": 2500},
        "formula": "activity_data * emission_factor"
    },
    agent_id="calc_agent_001"
)
```

## Environment Variables

```bash
export GREENLANG_BROKER_TYPE=redis
export GREENLANG_REDIS_HOST=localhost
export GREENLANG_REDIS_PORT=6379
export GREENLANG_REDIS_PASSWORD=your_password
export GREENLANG_LOG_LEVEL=INFO
```

## Performance Tips

1. **Use Batch Publishing**: 80% faster for bulk operations
2. **Tune Worker Count**: Start with 10, scale based on CPU
3. **Monitor Consumer Lag**: Add workers if lag > 1000
4. **Connection Pooling**: max_connections=50 for high load
5. **Batch Size**: 10-100 messages per batch
6. **Priority Queues**: Use HIGH priority for critical messages

## Common Patterns

### Pattern 1: Task Distribution
```python
# Submit 1000 tasks
await pattern.submit_batch("tasks", tasks)
# Process with 20 workers
await pattern.process_tasks("tasks", handler, num_workers=20)
```

### Pattern 2: Request-Response
```python
# Agent A requests → Agent B processes → Agent A receives
response = await pattern.send_request("agent.llm", {...}, timeout=30)
```

### Pattern 3: Event Notification
```python
# One agent publishes → Multiple agents subscribe
await pattern.subscribe("agent.events.*", handler)
await pattern.publish("agent.events.complete", {...})
```

## Testing

```bash
# Run all tests
pytest messaging/tests/ -v

# Run specific test
pytest messaging/tests/test_messaging_integration.py::test_saga_pattern -v

# Run examples
python messaging/examples/messaging_examples.py --example all
```

## Troubleshooting

**Connection Refused**
```bash
redis-cli ping  # Check Redis is running
redis-server    # Start Redis
```

**High Latency**
```python
# Increase connection pool
broker = RedisStreamsBroker(max_connections=100)
```

**Consumer Lag**
```python
# Scale up workers
await pattern.process_tasks(..., num_workers=50)
```

## Production Checklist

- [ ] Redis persistence enabled (AOF + RDB)
- [ ] Connection pooling configured (50+ connections)
- [ ] Dead letter queue monitoring
- [ ] Health checks enabled
- [ ] Metrics exported to Prometheus
- [ ] Consumer lag alerts (<1000)
- [ ] Circuit breaker enabled
- [ ] Logging configured (INFO level)
- [ ] Retry limits set (max_retries=3)
- [ ] TTL configured for time-sensitive messages

## Performance Targets

| Metric | Target |
|--------|--------|
| Throughput | 10,000 msg/s |
| Latency P95 | < 10ms |
| Max Consumers | 100 |
| Batch Size | 100 messages |
| Retention | 7 days |

---

**Documentation**: See README.md for complete guide
**Examples**: See examples/messaging_examples.py
**Tests**: See tests/test_messaging_integration.py
