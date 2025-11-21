# -*- coding: utf-8 -*-
"""
Example 7: Complete Telemetry Setup
====================================

Demonstrates comprehensive monitoring with logging, metrics, and tracing.
"""

import asyncio
from greenlang.telemetry import (
    get_logger,
    get_metrics_collector,
    TelemetryManager,
    get_tracer
)


async def main():
    """Run monitoring example."""
    # Initialize telemetry
    telemetry = TelemetryManager()

    # Get logger
    logger = get_logger(__name__)

    # Get metrics collector
    metrics = get_metrics_collector()

    # Get tracer
    tracer = get_tracer(__name__)

    # Example operation with full telemetry
    logger.info("Starting monitored operation")

    with tracer.start_span("example_operation") as span:
        # Log different levels
        logger.debug("Debug message: detailed information")
        logger.info("Info message: operation progress")
        logger.warning("Warning message: something to note")

        # Record metrics
        metrics.increment("operations.started")
        metrics.record("operation.size", 1000)

        # Simulate work
        await asyncio.sleep(0.1)

        # More metrics
        metrics.increment("operations.completed")
        metrics.record("operation.duration", 0.1)

        # Add span attributes
        span.set_attribute("operation.type", "example")
        span.set_attribute("records.processed", 1000)

        logger.info("Operation completed successfully")

    # Get metrics summary
    print("\nMetrics Summary:")
    print(f"  Operations started: 1")
    print(f"  Operations completed: 1")
    print(f"  Operation duration: 0.1s")

    # Shutdown telemetry
    telemetry.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
