"""
Integration Example: Pipeline + TelemetryManager
=================================================

Demonstrates how to integrate pipelines with complete telemetry.
"""

import asyncio
from datetime import datetime
from greenlang.core import Pipeline, PipelineStage
from greenlang.telemetry import TelemetryManager, get_logger, get_metrics_collector, get_tracer


async def main():
    """Run Pipeline + TelemetryManager integration."""
    # Initialize telemetry
    telemetry = TelemetryManager()
    logger = get_logger(__name__)
    metrics = get_metrics_collector()
    tracer = get_tracer(__name__)

    print("\nPipeline + TelemetryManager Integration")
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline(name="monitored_pipeline")

    # Stage 1: Data loading with telemetry
    async def load_data(context: dict) -> dict:
        with tracer.start_span("load_data") as span:
            logger.info("Stage 1: Loading data")
            metrics.increment("pipeline.stage.load.started")

            # Simulate data loading
            await asyncio.sleep(0.1)

            data = {"records": 1000, "source": "database"}

            span.set_attribute("records.count", data["records"])
            metrics.record("pipeline.records.loaded", data["records"])
            metrics.increment("pipeline.stage.load.completed")

            logger.info(f"Loaded {data['records']} records")

            return {"data": data}

    pipeline.add_stage(PipelineStage(
        name="load",
        handler=load_data,
        timeout=30
    ))

    # Stage 2: Data processing with telemetry
    async def process_data(context: dict) -> dict:
        with tracer.start_span("process_data") as span:
            logger.info("Stage 2: Processing data")
            metrics.increment("pipeline.stage.process.started")

            data = context["data"]

            # Simulate processing
            await asyncio.sleep(0.15)

            processed = data["records"] * 2

            span.set_attribute("records.processed", processed)
            metrics.record("pipeline.records.processed", processed)
            metrics.increment("pipeline.stage.process.completed")

            logger.info(f"Processed {processed} records")

            return {"processed": processed}

    pipeline.add_stage(PipelineStage(
        name="process",
        handler=process_data,
        timeout=30
    ))

    # Stage 3: Data output with telemetry
    async def output_data(context: dict) -> dict:
        with tracer.start_span("output_data") as span:
            logger.info("Stage 3: Outputting data")
            metrics.increment("pipeline.stage.output.started")

            processed = context["processed"]

            # Simulate output
            await asyncio.sleep(0.05)

            span.set_attribute("records.output", processed)
            metrics.record("pipeline.records.output", processed)
            metrics.increment("pipeline.stage.output.completed")

            logger.info(f"Output {processed} records")

            return {"success": True, "records": processed}

    pipeline.add_stage(PipelineStage(
        name="output",
        handler=output_data,
        timeout=30
    ))

    # Execute pipeline with complete telemetry
    logger.info("Starting monitored pipeline execution")
    metrics.increment("pipeline.executions.started")

    start_time = datetime.now()

    with tracer.start_span("pipeline_execution") as span:
        try:
            # Execute all stages
            context = {}
            for stage in pipeline.stages:
                logger.info(f"Executing stage: {stage.name}")
                context = await stage.handler(context)

            duration = (datetime.now() - start_time).total_seconds()

            span.set_attribute("pipeline.duration", duration)
            span.set_attribute("pipeline.success", True)

            metrics.record("pipeline.duration", duration)
            metrics.increment("pipeline.executions.completed")

            logger.info(f"Pipeline completed in {duration:.3f}s")

            print(f"\nPipeline Execution Summary:")
            print(f"  Success: {context['success']}")
            print(f"  Records processed: {context['records']}")
            print(f"  Duration: {duration:.3f}s")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            metrics.increment("pipeline.executions.failed")
            span.set_attribute("pipeline.error", str(e))
            raise

    print("\n" + "=" * 60)
    print("Telemetry Captured:")
    print("  - Structured logging at each stage")
    print("  - Metrics for counts and durations")
    print("  - Distributed tracing spans")
    print("=" * 60)

    # Shutdown telemetry
    telemetry.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
