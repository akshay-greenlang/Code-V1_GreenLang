"""
Advanced Messaging Patterns Examples

Demonstrates sophisticated coordination patterns for agent systems.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from ..redis_streams_broker import RedisStreamsBroker
from ..patterns import (
    RequestReplyPattern,
    PubSubPattern,
    WorkQueuePattern,
    SagaPattern,
    CircuitBreakerPattern,
)
from ..consumer_group import ConsumerGroupManager


async def example_request_reply():
    """Request-Reply pattern for RPC-style communication."""
    print("\n=== Request-Reply Pattern ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = RequestReplyPattern(broker)

        # Define calculator service
        async def calculator_service(payload):
            """Simulates LLM or calculation service."""
            operation = payload["operation"]
            a, b = payload["a"], payload["b"]

            await asyncio.sleep(0.1)  # Simulate processing

            if operation == "add":
                result = a + b
            elif operation == "multiply":
                result = a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")

            return {
                "result": result,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Start service handler
        print("Starting calculator service...")
        handler_task = asyncio.create_task(
            pattern.handle_request(
                "services.calculator",
                calculator_service,
                "calculator_handlers",
            )
        )

        await asyncio.sleep(0.5)  # Wait for handler to start

        # Send requests
        print("\nSending calculation requests...")

        response1 = await pattern.send_request(
            "services.calculator",
            {"operation": "add", "a": 10, "b": 32},
            timeout=5.0,
        )
        print(f"10 + 32 = {response1.payload['result']}")

        response2 = await pattern.send_request(
            "services.calculator",
            {"operation": "multiply", "a": 6, "b": 7},
            timeout=5.0,
        )
        print(f"6 × 7 = {response2.payload['result']}")

        handler_task.cancel()
        print("\n✓ Request-Reply pattern complete")

    finally:
        await broker.disconnect()


async def example_pubsub():
    """Pub-Sub pattern for event broadcasting."""
    print("\n=== Pub-Sub Pattern ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = PubSubPattern(broker)

        # Event handlers
        events_received = []

        async def monitoring_handler(message):
            """Monitor all agent events."""
            events_received.append(message.payload)
            print(f"[Monitor] Event: {message.payload['event_type']}")

        async def alert_handler(message):
            """Handle critical alerts."""
            if message.payload.get("severity") == "critical":
                print(f"[Alert] CRITICAL: {message.payload['message']}")

        # Subscribe handlers
        print("Subscribing event handlers...")
        await pattern.subscribe("agent.events.*", monitoring_handler)
        await pattern.subscribe("agent.events.alert", alert_handler)

        await asyncio.sleep(0.5)

        # Publish events
        print("\nPublishing events...")

        await pattern.publish(
            "agent.events.calculation",
            {
                "event_type": "calculation_complete",
                "agent_id": "calc_001",
                "result": 42,
            },
        )

        await pattern.publish(
            "agent.events.alert",
            {
                "event_type": "alert",
                "severity": "critical",
                "message": "High memory usage detected",
            },
        )

        await pattern.publish(
            "agent.events.status",
            {
                "event_type": "status_update",
                "agent_id": "worker_005",
                "status": "healthy",
            },
        )

        await asyncio.sleep(1.0)  # Wait for event processing

        print(f"\nTotal events received: {len(events_received)}")
        print("✓ Pub-Sub pattern complete")

    finally:
        await broker.disconnect()


async def example_work_queue():
    """Work Queue pattern for distributed task processing."""
    print("\n=== Work Queue Pattern ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        pattern = WorkQueuePattern(broker)

        # Task handler
        processed_tasks = []

        async def process_esg_report(payload):
            """Simulates ESG report generation."""
            company_id = payload["company_id"]
            report_type = payload["report_type"]

            print(f"Processing {report_type} for {company_id}...")
            await asyncio.sleep(0.2)  # Simulate processing

            processed_tasks.append(company_id)

            return {
                "company_id": company_id,
                "report_type": report_type,
                "status": "completed",
                "generated_at": datetime.utcnow().isoformat(),
            }

        # Submit batch of tasks
        print("Submitting 15 report generation tasks...")
        tasks = [
            {
                "company_id": f"COMP{i:03d}",
                "report_type": "carbon_footprint",
                "year": 2024,
            }
            for i in range(15)
        ]

        await pattern.submit_batch("tasks.esg_reports", tasks)

        # Process with multiple workers
        print("Starting 5 worker agents...")

        async def run_workers():
            await pattern.process_tasks(
                "tasks.esg_reports",
                process_esg_report,
                consumer_group="report_workers",
                num_workers=5,
            )

        worker_task = asyncio.create_task(run_workers())
        await asyncio.sleep(3.0)  # Let workers process
        worker_task.cancel()

        print(f"\nProcessed {len(processed_tasks)} tasks with 5 parallel workers")
        print("✓ Work Queue pattern complete")

    finally:
        await broker.disconnect()


async def example_saga():
    """Saga pattern for distributed transactions."""
    print("\n=== Saga Pattern ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        saga = SagaPattern(broker)

        # Simulated state
        state = {
            "data_validated": False,
            "calculations_done": False,
            "report_generated": False,
        }

        # Saga steps with compensation
        def validate_data(data):
            """Step 1: Validate input data."""
            print("1. Validating data...")
            state["data_validated"] = True
            return {"validation_status": "passed"}

        def compensate_validate(data):
            """Rollback validation."""
            print("   ↩ Rolling back validation")
            state["data_validated"] = False

        def calculate_emissions(data):
            """Step 2: Calculate emissions."""
            print("2. Calculating emissions...")
            state["calculations_done"] = True
            return {"emissions_total": 15000}

        def compensate_calculate(data):
            """Rollback calculations."""
            print("   ↩ Rolling back calculations")
            state["calculations_done"] = False

        def generate_report(data):
            """Step 3: Generate report (will fail)."""
            print("3. Generating report...")
            raise ValueError("Report generation failed!")

        def compensate_report(data):
            """Rollback report."""
            print("   ↩ Rolling back report")
            state["report_generated"] = False

        # Add saga steps
        saga.add_step("validate", validate_data, compensate_validate)
        saga.add_step("calculate", calculate_emissions, compensate_calculate)
        saga.add_step("report", generate_report, compensate_report)

        # Execute saga
        print("Executing saga transaction...\n")

        try:
            await saga.execute({"company_id": "COMP001"})
        except Exception as e:
            print(f"\nSaga failed: {e}")
            print("All completed steps have been compensated (rolled back)")

        print(f"\nFinal state: {state}")
        print("✓ Saga pattern complete")

    finally:
        await broker.disconnect()


async def example_circuit_breaker():
    """Circuit Breaker pattern for fault tolerance."""
    print("\n=== Circuit Breaker Pattern ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        breaker = CircuitBreakerPattern(
            failure_threshold=3,
            timeout_seconds=2,
        )

        # Simulated external service
        call_count = [0]

        async def unreliable_service():
            """Simulates flaky external API."""
            call_count[0] += 1

            if call_count[0] <= 3:
                # First 3 calls fail
                raise ConnectionError("Service unavailable")
            else:
                # Later calls succeed
                return {"status": "success"}

        print("Calling unreliable external service...\n")

        # Try calls through circuit breaker
        for i in range(6):
            try:
                result = await breaker.call(unreliable_service)
                print(f"Call {i+1}: ✓ Success - {result}")

            except ConnectionError as e:
                print(f"Call {i+1}: ✗ Failed - {e}")

            except Exception as e:
                print(f"Call {i+1}: ✗ Circuit OPEN - {e}")

            print(f"         Circuit state: {breaker.state}")
            await asyncio.sleep(0.5)

        print("\n✓ Circuit Breaker pattern complete")

    finally:
        await broker.disconnect()


async def example_consumer_group_management():
    """Advanced consumer group management."""
    print("\n=== Consumer Group Management ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        manager = ConsumerGroupManager(broker)

        # Create consumer group
        print("Creating consumer group...")
        await manager.create_group("tasks.processing", "dynamic_workers")

        # Define handler
        processed = []

        async def task_handler(message):
            """Process task."""
            await asyncio.sleep(0.1)
            processed.append(message.id)
            print(f"  Processed task: {message.payload.get('task_id')}")

        # Publish tasks
        print("\nPublishing 20 tasks...")
        for i in range(20):
            await broker.publish(
                "tasks.processing",
                {"task_id": i, "data": f"task_{i}"},
            )

        # Start with 2 consumers
        print("\nScaling to 2 consumers...")
        await manager.scale_consumers(
            "tasks.processing",
            "dynamic_workers",
            count=2,
            handler=task_handler,
        )

        await asyncio.sleep(1.0)

        # Get stats
        stats = await manager.get_group_stats("tasks.processing", "dynamic_workers")
        print(f"  Active consumers: {stats.consumer_count}")
        print(f"  Messages processed: {stats.total_messages_processed}")

        # Scale up to 5 consumers
        print("\nScaling to 5 consumers for faster processing...")
        await manager.scale_consumers(
            "tasks.processing",
            "dynamic_workers",
            count=5,
            handler=task_handler,
        )

        await asyncio.sleep(2.0)

        # Get updated stats
        stats = await manager.get_group_stats("tasks.processing", "dynamic_workers")
        print(f"  Active consumers: {stats.consumer_count}")
        print(f"  Messages processed: {stats.total_messages_processed}")
        print(f"  Average throughput: {stats.average_throughput:.2f} msg/s")

        # Scale down
        print("\nScaling down to 1 consumer...")
        await manager.scale_consumers(
            "tasks.processing",
            "dynamic_workers",
            count=1,
            handler=task_handler,
        )

        await asyncio.sleep(1.0)

        # Cleanup
        await manager.shutdown()

        print(f"\nTotal tasks processed: {len(processed)}")
        print("✓ Consumer Group Management complete")

    finally:
        await broker.disconnect()


async def example_combined_patterns():
    """Combining multiple patterns for complex workflows."""
    print("\n=== Combined Patterns ===\n")

    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()

    try:
        # Initialize patterns
        request_reply = RequestReplyPattern(broker)
        pubsub = PubSubPattern(broker)
        work_queue = WorkQueuePattern(broker)
        saga = SagaPattern(broker)

        # Workflow: ESG Report Generation Pipeline
        print("ESG Report Generation Pipeline\n")

        # 1. Request validation service
        print("1. Requesting data validation...")

        async def validator(payload):
            return {"valid": True, "company_id": payload["company_id"]}

        validator_task = asyncio.create_task(
            request_reply.handle_request(
                "services.validator",
                validator,
                "validators",
            )
        )
        await asyncio.sleep(0.5)

        validation_result = await request_reply.send_request(
            "services.validator",
            {"company_id": "COMP001", "data": {...}},
        )
        print(f"   ✓ Validation: {validation_result.payload}")

        # 2. Publish event
        print("\n2. Publishing validation event...")
        await pubsub.publish(
            "agent.events.validation",
            {"company_id": "COMP001", "status": "validated"},
        )

        # 3. Submit calculation tasks to work queue
        print("\n3. Submitting calculation tasks...")
        await work_queue.submit_batch(
            "tasks.calculations",
            [
                {"scope": "scope1", "value": 1000},
                {"scope": "scope2", "value": 2000},
                {"scope": "scope3", "value": 3000},
            ],
        )

        # 4. Execute saga for report generation
        print("\n4. Executing report generation saga...")

        def collect_results(data):
            print("   - Collecting calculation results")
            return {"results": [1000, 2000, 3000]}

        def generate_sections(data):
            print("   - Generating report sections")
            return {"sections": ["intro", "data", "conclusion"]}

        def finalize_report(data):
            print("   - Finalizing report")
            return {"report_id": "RPT001", "status": "complete"}

        saga.add_step("collect", collect_results, lambda d: None)
        saga.add_step("generate", generate_sections, lambda d: None)
        saga.add_step("finalize", finalize_report, lambda d: None)

        result = await saga.execute({"company_id": "COMP001"})
        print(f"\n   ✓ Report generated: {result['report_id']}")

        # 5. Publish completion event
        print("\n5. Publishing completion event...")
        await pubsub.publish(
            "agent.events.report_complete",
            {"report_id": result["report_id"], "company_id": "COMP001"},
        )

        validator_task.cancel()

        print("\n✓ Combined patterns workflow complete")

    finally:
        await broker.disconnect()


async def main():
    """Run all advanced pattern examples."""
    examples = [
        ("Request-Reply", example_request_reply),
        ("Pub-Sub", example_pubsub),
        ("Work Queue", example_work_queue),
        ("Saga", example_saga),
        ("Circuit Breaker", example_circuit_breaker),
        ("Consumer Group Management", example_consumer_group_management),
        ("Combined Patterns", example_combined_patterns),
    ]

    print("=" * 60)
    print("GreenLang Messaging System - Advanced Patterns")
    print("=" * 60)

    for name, example_func in examples:
        try:
            await example_func()
            print(f"\n{'='*60}\n")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}\n")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(1)

    print("=" * 60)
    print("All advanced examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
