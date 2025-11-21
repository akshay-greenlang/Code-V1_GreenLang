"""
Examples of using idempotency guarantees in GreenLang pipelines.

These examples demonstrate various patterns for ensuring idempotent
pipeline execution following Stripe/Twilio best practices.
"""

import time
from datetime import datetime
from typing import Dict, Any

from greenlang.pipeline.idempotency import (
    IdempotencyKey,
    IdempotencyManager,
    IdempotentPipeline,
    IdempotentPipelineBase,
    FileStorageBackend,
    RedisStorageBackend
)


# =============================================================================
# Example 1: Simple Function Idempotency
# =============================================================================

@IdempotentPipeline(ttl_seconds=3600)  # 1 hour cache
def calculate_carbon_emissions(activity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate carbon emissions with automatic idempotency.

    Multiple calls with same input will return cached result.
    """
    print(f"[{datetime.now()}] Calculating emissions for: {activity_data}")

    # Simulate expensive calculation
    time.sleep(1)

    emissions = activity_data["fuel_consumed"] * 2.5  # kg CO2 per liter

    return {
        "emissions_kg_co2": emissions,
        "activity_data": activity_data,
        "calculated_at": datetime.now().isoformat(),
        "methodology": "IPCC 2006"
    }


def example_simple_idempotency():
    """Demonstrate simple function idempotency."""
    print("\n=== Example 1: Simple Function Idempotency ===\n")

    activity = {
        "fuel_consumed": 100,
        "fuel_type": "diesel",
        "vehicle_id": "TRUCK-001"
    }

    # First call - executes calculation
    print("First call:")
    result1 = calculate_carbon_emissions(activity)
    print(f"Result: {result1['emissions_kg_co2']} kg CO2")

    # Second call - returns cached result
    print("\nSecond call (cached):")
    result2 = calculate_carbon_emissions(activity)
    print(f"Result: {result2['emissions_kg_co2']} kg CO2")

    # Verify same result
    assert result1 == result2
    print("\n✅ Same result returned from cache!")


# =============================================================================
# Example 2: Custom Idempotency Keys
# =============================================================================

@IdempotentPipeline(ttl_seconds=1800)
def process_supplier_data(supplier_id: str, data: Dict) -> Dict:
    """Process supplier data with custom idempotency key support."""
    print(f"Processing supplier {supplier_id}")
    time.sleep(0.5)

    return {
        "supplier_id": supplier_id,
        "processed_data": data,
        "status": "completed"
    }


def example_custom_keys():
    """Demonstrate custom idempotency keys."""
    print("\n=== Example 2: Custom Idempotency Keys ===\n")

    # Using auto-generated key
    print("Auto-generated keys:")
    result1 = process_supplier_data("SUP-001", {"emissions": 1000})
    result2 = process_supplier_data("SUP-001", {"emissions": 1000})
    print(f"Same inputs -> Same result: {result1 == result2}")

    # Using custom key to group different inputs
    print("\nCustom keys:")
    result3 = process_supplier_data(
        "SUP-002",
        {"emissions": 2000},
        idempotency_key="daily_batch_2024_01_15"
    )

    # Different input but same key returns cached result
    result4 = process_supplier_data(
        "SUP-003",  # Different supplier!
        {"emissions": 3000},  # Different data!
        idempotency_key="daily_batch_2024_01_15"  # Same key
    )

    print(f"Different inputs, same key -> Same result: {result3 == result4}")


# =============================================================================
# Example 3: Pipeline Class with Idempotency
# =============================================================================

class EmissionsPipeline(IdempotentPipelineBase):
    """
    Complete emissions calculation pipeline with built-in idempotency.
    """

    def __init__(self):
        # Initialize with 2-hour TTL
        super().__init__(idempotency_ttl=7200)
        self.calculation_count = 0

    def _execute_pipeline(self, input_data: Dict, **kwargs) -> Dict:
        """Execute the actual pipeline logic."""
        self.calculation_count += 1

        print(f"[Execution #{self.calculation_count}] Processing pipeline...")

        # Step 1: Validate
        if "activity_data" not in input_data:
            raise ValueError("Missing activity_data")

        # Step 2: Calculate
        activity = input_data["activity_data"]
        emissions = activity.get("fuel", 0) * 2.5 + activity.get("electricity", 0) * 0.5

        # Step 3: Create result with provenance
        return {
            "emissions_total": emissions,
            "breakdown": {
                "fuel_emissions": activity.get("fuel", 0) * 2.5,
                "electricity_emissions": activity.get("electricity", 0) * 0.5
            },
            "provenance": {
                "pipeline": "EmissionsPipeline",
                "version": "1.0.0",
                "execution_count": self.calculation_count
            }
        }


def example_pipeline_idempotency():
    """Demonstrate pipeline class idempotency."""
    print("\n=== Example 3: Pipeline Class Idempotency ===\n")

    pipeline = EmissionsPipeline()

    input_data = {
        "activity_data": {
            "fuel": 100,
            "electricity": 200
        },
        "period": "2024-Q1"
    }

    # First execution
    print("First execution:")
    result1 = pipeline.execute(input_data)
    print(f"Emissions: {result1['emissions_total']}")
    print(f"Execution count: {pipeline.calculation_count}")

    # Second execution - cached
    print("\nSecond execution (cached):")
    result2 = pipeline.execute(input_data)
    print(f"Emissions: {result2['emissions_total']}")
    print(f"Execution count: {pipeline.calculation_count}")  # Still 1
    print(f"Cached: {result2['provenance']['idempotency']['cached']}")

    # Force execution with skip_cache
    print("\nThird execution (forced):")
    result3 = pipeline.execute(input_data, skip_cache=True)
    print(f"Execution count: {pipeline.calculation_count}")  # Now 2


# =============================================================================
# Example 4: Handling Failures and Retries
# =============================================================================

failure_count = 0


@IdempotentPipeline(ttl_seconds=300)
def unreliable_calculation(data: Dict) -> Dict:
    """Simulate unreliable calculation that may fail."""
    global failure_count
    failure_count += 1

    print(f"Attempt #{failure_count}")

    # Fail first 2 attempts
    if failure_count <= 2:
        raise ConnectionError("External service unavailable")

    return {"result": data["value"] * 10, "attempts": failure_count}


def example_failure_handling():
    """Demonstrate failure handling and retries."""
    print("\n=== Example 4: Failure Handling ===\n")

    global failure_count
    failure_count = 0

    input_data = {"value": 5}

    # First attempt - fails
    print("First attempt:")
    try:
        result = unreliable_calculation(input_data)
    except ConnectionError as e:
        print(f"Failed: {e}")

    # Second attempt - also fails (result cached as failed)
    print("\nSecond attempt:")
    try:
        result = unreliable_calculation(input_data)
    except RuntimeError as e:
        print(f"Cached failure: {e}")

    # Clear cache and retry
    print("\nClearing cache and retrying:")
    key = unreliable_calculation.get_idempotency_key(input_data)
    unreliable_calculation.clear_cache(key)

    # This time it should work (3rd attempt)
    result = unreliable_calculation(input_data)
    print(f"Success! Result: {result}")


# =============================================================================
# Example 5: Redis Backend for Production
# =============================================================================

def example_redis_backend():
    """Demonstrate Redis backend configuration."""
    print("\n=== Example 5: Redis Backend (Production) ===\n")

    # Note: Requires Redis connection
    # Uncomment and configure for production use

    """
    import redis

    # Create Redis client
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False  # Need binary for pickle
    )

    # Create storage backend
    storage = RedisStorageBackend(redis_client, prefix="greenlang:idempotency:")

    # Create manager with Redis storage
    manager = IdempotencyManager(
        storage=storage,
        default_ttl=3600,
        enable_locking=True  # Distributed locking
    )

    @IdempotentPipeline(manager=manager)
    def distributed_calculation(data):
        return {"processed": data, "node": "worker-1"}

    result = distributed_calculation({"value": 42})
    print(f"Result: {result}")
    """

    print("Redis backend example (see code comments for implementation)")


# =============================================================================
# Example 6: Batch Processing with Idempotency
# =============================================================================

class BatchProcessingPipeline(IdempotentPipelineBase):
    """Pipeline for batch processing with idempotency per batch."""

    def _execute_pipeline(self, input_data: Dict, **kwargs) -> Dict:
        """Process batch of items."""
        batch_id = input_data["batch_id"]
        items = input_data["items"]

        print(f"Processing batch {batch_id} with {len(items)} items")

        results = []
        for item in items:
            # Process each item
            result = {
                "id": item["id"],
                "emissions": item["activity"] * 2.5,
                "status": "processed"
            }
            results.append(result)

        return {
            "batch_id": batch_id,
            "processed_count": len(results),
            "results": results,
            "provenance": {
                "timestamp": datetime.now().isoformat()
            }
        }


def example_batch_processing():
    """Demonstrate batch processing with idempotency."""
    print("\n=== Example 6: Batch Processing ===\n")

    pipeline = BatchProcessingPipeline()

    batch = {
        "batch_id": "BATCH-2024-01-15-001",
        "items": [
            {"id": "ITEM-001", "activity": 100},
            {"id": "ITEM-002", "activity": 200},
            {"id": "ITEM-003", "activity": 150}
        ]
    }

    # Process batch
    print("First processing:")
    result1 = pipeline.execute(batch)
    print(f"Processed {result1['processed_count']} items")

    # Try to process same batch again
    print("\nAttempting reprocessing (will use cache):")
    result2 = pipeline.execute(batch)
    print(f"Cached: {result2['provenance']['idempotency']['cached']}")

    # Different batch processes normally
    batch2 = batch.copy()
    batch2["batch_id"] = "BATCH-2024-01-15-002"

    print("\nProcessing different batch:")
    result3 = pipeline.execute(batch2)
    print(f"Processed batch {result3['batch_id']}")


# =============================================================================
# Example 7: Monitoring and Observability
# =============================================================================

def example_monitoring():
    """Demonstrate monitoring idempotency metrics."""
    print("\n=== Example 7: Monitoring & Observability ===\n")

    # Create manager with file storage
    manager = IdempotencyManager(
        storage=FileStorageBackend(".idempotency_monitoring"),
        default_ttl=600
    )

    @IdempotentPipeline(manager=manager)
    def monitored_operation(data):
        time.sleep(0.1)
        return {"processed": data}

    # Execute operations
    print("Executing operations...")
    for i in range(5):
        data = {"id": i % 2}  # Only 2 unique inputs
        result = monitored_operation(data)

    # Check manager statistics
    print("\nIdempotency Statistics:")
    print(f"Active operations: {len(manager._active_operations)}")

    # Get specific operation status
    key = monitored_operation.get_idempotency_key({"id": 0})
    status = manager.storage.get(key)

    if status:
        print(f"\nOperation status for key {key[:20]}...:")
        print(f"  Status: {status.status}")
        print(f"  Created: {status.created_at}")
        print(f"  TTL remaining: {status.time_to_live} seconds")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GreenLang Pipeline Idempotency Examples")
    print("=" * 60)

    # Run all examples
    example_simple_idempotency()
    example_custom_keys()
    example_pipeline_idempotency()
    example_failure_handling()
    example_redis_backend()
    example_batch_processing()
    example_monitoring()

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)