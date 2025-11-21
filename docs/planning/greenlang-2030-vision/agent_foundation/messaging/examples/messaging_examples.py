# -*- coding: utf-8 -*-
"""
Messaging System Usage Examples

Complete examples demonstrating all messaging patterns and features.

Run examples:
    python messaging_examples.py --example basic
    python messaging_examples.py --example batch
    python messaging_examples.py --example patterns
    python messaging_examples.py --example all
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# Import messaging components
import sys
from pathlib import Path
from greenlang.determinism import DeterministicClock
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from messaging.redis_streams_broker import RedisStreamsBroker
from messaging.config import MessagingConfig
from messaging.message import Message, MessagePriority
from messaging.patterns import (
    RequestReplyPattern,
    PubSubPattern,
    WorkQueuePattern,
    EventSourcingPattern,
    SagaPattern,
    CircuitBreakerPattern,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Basic Publishing and Consuming
async def example_basic_pubsub():
    """Example: Basic message publishing and consuming."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Publish/Subscribe")
    logger.info("=" * 60)

    # Create broker
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish messages
        logger.info("Publishing messages...")
        msg_id = await broker.publish(
            "agent.tasks",
            {"task": "analyze_esg", "company": "ACME Corp"},
            priority=MessagePriority.HIGH,
        )
        logger.info(f"Published message: {msg_id}")

        # Consume messages
        logger.info("Consuming messages...")
        consumer_group = "demo_workers"
        count = 0

        async for message in broker.consume("agent.tasks", consumer_group):
            logger.info(f"Received: {message.payload}")
            await broker.acknowledge(message)

            count += 1
            if count >= 1:  # Process one message for demo
                break

        logger.info("Basic example completed!\n")

    finally:
        await broker.disconnect()


# Example 2: Batch Publishing
async def example_batch_publishing():
    """Example: Efficient batch publishing (80% overhead reduction)."""
    logger.info("=" * 60)
    logger.info("Example 2: Batch Publishing")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Prepare batch of 100 messages
        payloads = [
            {"task": f"calculate_emissions", "record_id": i}
            for i in range(100)
        ]

        logger.info("Publishing batch of 100 messages...")
        start_time = DeterministicClock.utcnow()

        message_ids = await broker.publish_batch(
            "agent.calculations",
            payloads,
            priority=MessagePriority.NORMAL,
        )

        duration_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Published {len(message_ids)} messages in {duration_ms:.2f}ms")
        logger.info(f"Average: {duration_ms/len(message_ids):.2f}ms per message")
        logger.info("Batch example completed!\n")

    finally:
        await broker.disconnect()


# Example 3: Request-Reply Pattern
async def example_request_reply():
    """Example: Request-reply for synchronous communication."""
    logger.info("=" * 60)
    logger.info("Example 3: Request-Reply Pattern")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = RequestReplyPattern(broker)

        # Define request handler
        def llm_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Simulate LLM processing."""
            prompt = payload.get("prompt", "")
            return {
                "result": f"Processed: {prompt}",
                "tokens": 150,
                "model": "claude-sonnet-4",
            }

        # Start request handler in background
        handler_task = asyncio.create_task(
            pattern.handle_request(
                "agent.llm",
                llm_handler,
                consumer_group="llm_handlers"
            )
        )

        # Give handler time to start
        await asyncio.sleep(1)

        # Send request
        logger.info("Sending request...")
        response = await pattern.send_request(
            "agent.llm",
            {"prompt": "Analyze this ESG report"},
            timeout=10.0,
        )

        if response:
            logger.info(f"Received response: {response.payload}")
        else:
            logger.warning("Request timed out")

        # Cleanup
        handler_task.cancel()
        logger.info("Request-reply example completed!\n")

    finally:
        await broker.disconnect()


# Example 4: Work Queue Pattern
async def example_work_queue():
    """Example: Work queue with multiple workers."""
    logger.info("=" * 60)
    logger.info("Example 4: Work Queue Pattern")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = WorkQueuePattern(broker)

        # Submit tasks
        logger.info("Submitting tasks to work queue...")
        tasks = [
            {"calculation": "scope1_emissions", "site_id": i}
            for i in range(20)
        ]

        await pattern.submit_batch("agent.work_queue", tasks)
        logger.info(f"Submitted {len(tasks)} tasks")

        # Define worker
        def process_task(payload: Dict[str, Any]) -> Any:
            """Simulate task processing."""
            calculation = payload.get("calculation")
            site_id = payload.get("site_id")
            logger.info(f"Processing {calculation} for site {site_id}")
            # Simulate work
            import time
            time.sleep(0.1)
            return {"status": "completed", "result": 123.45}

        # Process with 5 workers (runs until cancelled)
        logger.info("Starting 5 workers...")
        worker_task = asyncio.create_task(
            pattern.process_tasks(
                "agent.work_queue",
                process_task,
                consumer_group="calculation_workers",
                num_workers=5,
            )
        )

        # Let workers run for a bit
        await asyncio.sleep(3)

        # Cleanup
        worker_task.cancel()
        logger.info("Work queue example completed!\n")

    finally:
        await broker.disconnect()


# Example 5: Pub-Sub Pattern
async def example_pubsub_pattern():
    """Example: Publish-subscribe for broadcasting."""
    logger.info("=" * 60)
    logger.info("Example 5: Pub-Sub Pattern")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = PubSubPattern(broker)

        # Define event handler
        async def event_handler(message: Message):
            """Handle broadcast events."""
            logger.info(f"Event received: {message.payload}")

        # Subscribe to events
        logger.info("Subscribing to agent.events.*...")
        await pattern.subscribe("agent.events.*", event_handler)

        # Give subscription time to start
        await asyncio.sleep(1)

        # Publish events
        logger.info("Broadcasting events...")
        await pattern.publish(
            "agent.events.calculation_complete",
            {
                "event": "calculation_complete",
                "agent_id": "calc_001",
                "timestamp": DeterministicClock.utcnow().isoformat(),
            }
        )

        await asyncio.sleep(1)
        logger.info("Pub-sub example completed!\n")

    finally:
        await broker.disconnect()


# Example 6: Event Sourcing Pattern
async def example_event_sourcing():
    """Example: Event sourcing for audit trails."""
    logger.info("=" * 60)
    logger.info("Example 6: Event Sourcing Pattern")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = EventSourcingPattern(broker)

        # Log events
        logger.info("Logging agent events...")
        await pattern.log_event(
            "calculation",
            {
                "input": {"activity_data": 1000, "emission_factor": 2.5},
                "output": {"emissions": 2500},
                "formula": "activity_data * emission_factor",
            },
            agent_id="calc_agent_001",
        )

        await pattern.log_event(
            "validation",
            {
                "rules_checked": 50,
                "errors": 0,
                "warnings": 2,
            },
            agent_id="validation_agent_001",
        )

        logger.info("Events logged for audit trail")
        logger.info("Event sourcing example completed!\n")

    finally:
        await broker.disconnect()


# Example 7: Saga Pattern
async def example_saga_pattern():
    """Example: Saga for distributed transactions."""
    logger.info("=" * 60)
    logger.info("Example 7: Saga Pattern")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        saga = SagaPattern(broker)

        # Define saga steps
        def validate_data(data: Dict) -> Dict:
            logger.info("Step 1: Validating data...")
            return {"validated": True}

        def compensate_validate(data: Dict):
            logger.info("Compensating: Validation rollback")

        def calculate_emissions(data: Dict) -> Dict:
            logger.info("Step 2: Calculating emissions...")
            # Simulate calculation failure
            # raise Exception("Calculation failed!")
            return {"emissions": 2500}

        def compensate_calculate(data: Dict):
            logger.info("Compensating: Calculation rollback")

        def save_results(data: Dict) -> Dict:
            logger.info("Step 3: Saving results...")
            return {"saved": True}

        def compensate_save(data: Dict):
            logger.info("Compensating: Delete saved results")

        # Add steps
        saga.add_step("validate", validate_data, compensate_validate)
        saga.add_step("calculate", calculate_emissions, compensate_calculate)
        saga.add_step("save", save_results, compensate_save)

        # Execute saga
        logger.info("Executing saga...")
        result = await saga.execute({"company": "ACME Corp"})
        logger.info(f"Saga completed: {result}")
        logger.info("Saga example completed!\n")

    finally:
        await broker.disconnect()


# Example 8: Circuit Breaker Pattern
async def example_circuit_breaker():
    """Example: Circuit breaker for fault tolerance."""
    logger.info("=" * 60)
    logger.info("Example 8: Circuit Breaker Pattern")
    logger.info("=" * 60)

    breaker = CircuitBreakerPattern(
        failure_threshold=3,
        timeout_seconds=5,
    )

    # Define unreliable service
    call_count = 0

    def unreliable_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            logger.info(f"Call {call_count}: Service failing...")
            raise Exception("Service unavailable")
        else:
            logger.info(f"Call {call_count}: Service recovered")
            return "Success!"

    # Test circuit breaker
    for i in range(6):
        try:
            result = await breaker.call(unreliable_service)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.warning(f"Error: {e}")

        await asyncio.sleep(1)

    logger.info(f"Circuit breaker state: {breaker.state}")
    logger.info("Circuit breaker example completed!\n")


# Example 9: Dead Letter Queue
async def example_dead_letter_queue():
    """Example: Dead letter queue for failed messages."""
    logger.info("=" * 60)
    logger.info("Example 9: Dead Letter Queue")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish a message that will fail
        logger.info("Publishing message that will fail...")
        await broker.publish(
            "agent.failing_tasks",
            {"task": "impossible_task"},
        )

        # Consume and fail
        async for message in broker.consume("agent.failing_tasks", "dlq_demo"):
            logger.info(f"Processing message: {message.id}")

            # Simulate failure
            logger.warning("Processing failed!")
            await broker.nack(
                message,
                error_message="Task impossible to complete",
                requeue=True,  # Will retry up to max_retries
            )
            break

        # Check DLQ (after max retries)
        await asyncio.sleep(1)
        dlq_messages = await broker.get_dead_letter_messages("agent.failing_tasks")
        logger.info(f"DLQ contains {len(dlq_messages)} messages")

        # Reprocess from DLQ
        if dlq_messages:
            logger.info("Reprocessing message from DLQ...")
            await broker.reprocess_dead_letter_message(dlq_messages[0])

        logger.info("DLQ example completed!\n")

    finally:
        await broker.disconnect()


# Example 10: Monitoring and Metrics
async def example_monitoring():
    """Example: Monitoring and metrics collection."""
    logger.info("=" * 60)
    logger.info("Example 10: Monitoring and Metrics")
    logger.info("=" * 60)

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish some messages
        logger.info("Generating activity...")
        for i in range(50):
            await broker.publish("agent.metrics_test", {"count": i})

        # Get metrics
        metrics = broker.get_metrics()
        logger.info("Broker Metrics:")
        logger.info(f"  Messages Published: {metrics['messages_published']}")
        logger.info(f"  Messages Consumed: {metrics['messages_consumed']}")
        logger.info(f"  Messages Failed: {metrics['messages_failed']}")
        logger.info(f"  Throughput: {metrics['throughput_per_second']}")
        logger.info(f"  Latency: {metrics['average_latency_ms']}")

        # Health check
        health = await broker.health_check()
        logger.info(f"\nHealth Status: {health['status']}")
        logger.info(f"  Connected: {health['connected']}")
        logger.info(f"  Latency: {health['latency_ms']:.2f}ms")
        logger.info(f"  Uptime: {health['uptime_seconds']}s")

        # Consumer lag
        lag = await broker.get_consumer_lag("agent.metrics_test", "demo_workers")
        logger.info(f"\nConsumer Lag: {lag} pending messages")

        logger.info("\nMonitoring example completed!\n")

    finally:
        await broker.disconnect()


async def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Messaging System Examples")
    parser.add_argument(
        "--example",
        choices=[
            "basic", "batch", "request_reply", "work_queue", "pubsub",
            "event_sourcing", "saga", "circuit_breaker", "dlq", "monitoring", "all"
        ],
        default="all",
        help="Example to run"
    )
    args = parser.parse_args()

    examples = {
        "basic": example_basic_pubsub,
        "batch": example_batch_publishing,
        "request_reply": example_request_reply,
        "work_queue": example_work_queue,
        "pubsub": example_pubsub_pattern,
        "event_sourcing": example_event_sourcing,
        "saga": example_saga_pattern,
        "circuit_breaker": example_circuit_breaker,
        "dlq": example_dead_letter_queue,
        "monitoring": example_monitoring,
    }

    try:
        if args.example == "all":
            for name, func in examples.items():
                logger.info(f"\nRunning example: {name}")
                await func()
        else:
            await examples[args.example]()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
