"""
Basic Messaging Usage Examples

Simple examples demonstrating core messaging functionality.
"""

import asyncio
from datetime import datetime

from ..redis_streams_broker import RedisStreamsBroker
from ..message import MessagePriority


async def example_publish_consume():
    """Basic publish and consume example."""
    print("\n=== Basic Publish/Consume ===\n")

    # Create broker
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish message
        print("Publishing message...")
        message_id = await broker.publish(
            topic="agent.tasks",
            payload={
                "task_type": "esg_calculation",
                "company_id": "COMP123",
                "data": {"emissions": 1500, "scope": "scope1"},
            },
            priority=MessagePriority.HIGH,
        )
        print(f"Published message: {message_id}")

        # Consume message
        print("\nConsuming messages...")
        async for message in broker.consume(
            topic="agent.tasks",
            consumer_group="workers",
            batch_size=10,
            timeout_ms=5000,
        ):
            print(f"Received: {message.id}")
            print(f"Payload: {message.payload}")
            print(f"Priority: {message.priority}")

            # Acknowledge message
            await broker.acknowledge(message)
            print("Message acknowledged")
            break

    finally:
        await broker.disconnect()


async def example_batch_publish():
    """Batch publishing for high throughput."""
    print("\n=== Batch Publishing ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Prepare batch
        payloads = [
            {
                "company_id": f"COMP{i:03d}",
                "calculation_type": "carbon_footprint",
                "value": i * 100,
            }
            for i in range(100)
        ]

        # Publish batch
        print(f"Publishing batch of {len(payloads)} messages...")
        start = datetime.utcnow()

        message_ids = await broker.publish_batch(
            topic="agent.calculations",
            payloads=payloads,
            priority=MessagePriority.NORMAL,
        )

        duration = (datetime.utcnow() - start).total_seconds()
        throughput = len(message_ids) / duration

        print(f"Published {len(message_ids)} messages in {duration:.2f}s")
        print(f"Throughput: {throughput:.2f} msg/s")

    finally:
        await broker.disconnect()


async def example_message_priority():
    """Message priority handling."""
    print("\n=== Message Priority ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish messages with different priorities
        print("Publishing messages with different priorities...")

        await broker.publish(
            "agent.alerts",
            {"alert": "Critical system failure!"},
            priority=MessagePriority.CRITICAL,
        )
        print("✓ Critical priority message published")

        await broker.publish(
            "agent.alerts",
            {"alert": "High CPU usage detected"},
            priority=MessagePriority.HIGH,
        )
        print("✓ High priority message published")

        await broker.publish(
            "agent.alerts",
            {"alert": "Regular status update"},
            priority=MessagePriority.NORMAL,
        )
        print("✓ Normal priority message published")

        await broker.publish(
            "agent.alerts",
            {"alert": "Background maintenance scheduled"},
            priority=MessagePriority.LOW,
        )
        print("✓ Low priority message published")

        print("\nHigh/Critical priority messages are processed first!")

    finally:
        await broker.disconnect()


async def example_error_handling():
    """Error handling and retries."""
    print("\n=== Error Handling ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish test message
        await broker.publish(
            "agent.processing",
            {"data": "test_data"},
        )

        # Consume and simulate error
        async for message in broker.consume("agent.processing", "error_handlers"):
            print(f"Processing message: {message.id}")

            try:
                # Simulate processing error
                raise ValueError("Simulated processing error")

            except Exception as e:
                print(f"Error: {e}")

                # Negative acknowledge - retry
                await broker.nack(
                    message,
                    error_message=str(e),
                    requeue=True,
                )
                print(f"Message requeued (retry {message.retry_count}/{message.max_retries})")
                break

    finally:
        await broker.disconnect()


async def example_dead_letter_queue():
    """Dead letter queue for failed messages."""
    print("\n=== Dead Letter Queue ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish message that will fail
        await broker.publish(
            "agent.failing",
            {"data": "will_fail"},
        )

        # Consume and fail multiple times
        for attempt in range(4):  # max_retries = 3
            async for message in broker.consume("agent.failing", "failing_group"):
                print(f"Attempt {attempt + 1}: Processing {message.id}")

                await broker.nack(
                    message,
                    error_message="Processing always fails",
                    requeue=True,
                )

                if not message.can_retry():
                    print("Max retries exceeded - moved to DLQ")
                break

        # Check DLQ
        print("\nChecking dead letter queue...")
        dlq_messages = await broker.get_dead_letter_messages("agent.failing")

        if dlq_messages:
            print(f"Found {len(dlq_messages)} message(s) in DLQ")
            for dlq_msg in dlq_messages:
                print(f"  - Message: {dlq_msg.original_message.id}")
                print(f"    Failure: {dlq_msg.failure_reason}")

    finally:
        await broker.disconnect()


async def example_health_monitoring():
    """Health monitoring and metrics."""
    print("\n=== Health Monitoring ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Perform some operations
        await broker.publish("test.health", {"data": "test"})

        async for message in broker.consume("test.health", "health_group"):
            await broker.acknowledge(message)
            break

        # Check health
        print("Checking broker health...")
        health = await broker.health_check()

        print(f"Status: {health['status']}")
        print(f"Connected: {health['connected']}")
        print(f"Latency: {health['latency_ms']:.2f}ms")
        print(f"Uptime: {health['uptime_seconds']:.0f}s")
        print(f"Memory: {health['used_memory_mb']:.2f}MB")

        # Get metrics
        print("\nBroker metrics:")
        metrics = broker.get_metrics()

        print(f"Messages published: {metrics['messages_published']}")
        print(f"Messages consumed: {metrics['messages_consumed']}")
        print(f"Messages failed: {metrics['messages_failed']}")

        throughput = metrics['throughput_per_second']
        print(f"Publish throughput: {throughput['publish']:.2f} msg/s")
        print(f"Consume throughput: {throughput['consume']:.2f} msg/s")

    finally:
        await broker.disconnect()


async def example_consumer_groups():
    """Consumer groups for parallel processing."""
    print("\n=== Consumer Groups ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Create consumer group
        print("Creating consumer group...")
        await broker.create_consumer_group("agent.parallel", "workers")

        # Publish tasks
        print("Publishing 10 tasks...")
        for i in range(10):
            await broker.publish(
                "agent.parallel",
                {"task_id": i, "data": f"task_{i}"},
            )

        # Simulate multiple consumers processing in parallel
        print("\nProcessing with 3 parallel consumers...")

        async def consumer_worker(consumer_id):
            """Worker function."""
            count = 0
            async for message in broker.consume(
                "agent.parallel",
                "workers",
                consumer_id=consumer_id,
            ):
                print(f"Consumer {consumer_id} processing: {message.payload['task_id']}")
                await asyncio.sleep(0.1)  # Simulate work
                await broker.acknowledge(message)
                count += 1

                if count >= 4:  # Each processes ~3-4 messages
                    break

        # Run 3 consumers in parallel
        await asyncio.gather(
            consumer_worker("worker-1"),
            consumer_worker("worker-2"),
            consumer_worker("worker-3"),
        )

        print("\nAll tasks processed by parallel consumers!")

    finally:
        await broker.disconnect()


async def example_message_ttl():
    """Message time-to-live."""
    print("\n=== Message TTL ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Publish message with TTL
        print("Publishing message with 2 second TTL...")
        await broker.publish(
            "agent.ttl",
            {"data": "expires_soon"},
            ttl_seconds=2,
        )

        # Wait for expiration
        print("Waiting 3 seconds for expiration...")
        await asyncio.sleep(3)

        # Try to consume
        print("Attempting to consume expired message...")

        consumed = False
        async for message in broker.consume("agent.ttl", "ttl_group"):
            if message.is_expired():
                print("Message expired and skipped!")
            else:
                print("Message still valid")
                consumed = True
                await broker.acknowledge(message)
            break

        if not consumed:
            print("No valid messages found (expected)")

    finally:
        await broker.disconnect()


async def main():
    """Run all examples."""
    examples = [
        ("Basic Publish/Consume", example_publish_consume),
        ("Batch Publishing", example_batch_publish),
        ("Message Priority", example_message_priority),
        ("Error Handling", example_error_handling),
        ("Dead Letter Queue", example_dead_letter_queue),
        ("Health Monitoring", example_health_monitoring),
        ("Consumer Groups", example_consumer_groups),
        ("Message TTL", example_message_ttl),
    ]

    print("=" * 60)
    print("GreenLang Messaging System - Basic Examples")
    print("=" * 60)

    for name, example_func in examples:
        try:
            await example_func()
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")

        await asyncio.sleep(1)  # Brief pause between examples

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
