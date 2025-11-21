# -*- coding: utf-8 -*-
"""
Batch Optimization Example - 100K Suppliers/Hour Throughput
GL-VCCI Scope 3 Platform

Demonstrates optimized batch processing with:
- Chunked parallel processing
- Streaming/generator patterns
- Bulk database operations
- Memory management
- Real-time metrics

Performance Target: 100,000 suppliers per hour
Chunk Size: 1,000 suppliers per batch

Author: Team 5 - Performance Optimization Lead
Date: 2025-11-09
"""

import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime

# Import calculator agent
from services.agents.calculator.agent import Scope3CalculatorAgent
from services.agents.calculator.models import Category1Input


async def example_basic_batch_processing():
    """
    Example 1: Basic optimized batch processing.

    Processes 10,000 suppliers in chunks of 1,000.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Optimized Batch Processing")
    print("=" * 80)

    # Initialize agent (assume factor_broker is available)
    # agent = Scope3CalculatorAgent(factor_broker=factor_broker)

    # Generate sample supplier data
    suppliers = []
    for i in range(10000):
        supplier = {
            "supplier_id": f"SUP-{i:05d}",
            "product_name": f"Product {i}",
            "quantity": 100.0,
            "unit": "kg",
            "spend_usd": 1000.0,
            "supplier_country": "USA"
        }
        suppliers.append(supplier)

    print(f"\nProcessing {len(suppliers)} suppliers...")
    print(f"Chunk size: 1,000")
    print(f"Expected chunks: {len(suppliers) // 1000}")
    print()

    # Process with optimized batch processing
    # async for chunk_idx, results, metrics in agent.process_suppliers_optimized(
    #     suppliers=suppliers,
    #     category=1,
    #     chunk_size=1000
    # ):
    #     print(f"Chunk {chunk_idx}:")
    #     print(f"  - Processed: {metrics['successful']} suppliers")
    #     print(f"  - Failed: {metrics['failed']} suppliers")
    #     print(f"  - Emissions: {metrics['emissions_tco2e']:.3f} tCO2e")
    #     print(f"  - Time: {metrics['time_ms']:.2f}ms")
    #     print(f"  - Throughput: {metrics['throughput_per_hour']:.0f} suppliers/hour")
    #     print()

    print("Basic batch processing completed!")


async def example_streaming_with_database():
    """
    Example 2: Streaming processing with database bulk inserts.

    Demonstrates real-time processing with immediate database persistence.
    """
    print("=" * 80)
    print("EXAMPLE 2: Streaming with Database Bulk Inserts")
    print("=" * 80)

    # Mock database connection
    class MockDBConnection:
        def __init__(self):
            self.inserted_count = 0

        async def bulk_insert(self, table: str, data: List[Dict[str, Any]]):
            """Simulate bulk insert."""
            await asyncio.sleep(0.01)  # Simulate I/O
            self.inserted_count += len(data)
            print(f"  [DB] Bulk inserted {len(data)} records to {table}")

    db = MockDBConnection()

    # Generate sample data
    suppliers = [
        {"supplier_id": f"SUP-{i:05d}", "spend_usd": 1000.0}
        for i in range(5000)
    ]

    print(f"\nProcessing {len(suppliers)} suppliers with database persistence...")
    print()

    # Process with database connection
    # async for chunk_idx, results, metrics in agent.process_suppliers_optimized(
    #     suppliers=suppliers,
    #     category=1,
    #     chunk_size=1000,
    #     db_connection=db
    # ):
    #     print(f"Chunk {chunk_idx}: {metrics['successful']} processed, "
    #           f"{db.inserted_count} total in database")

    print(f"\nTotal records in database: {db.inserted_count}")


async def example_performance_monitoring():
    """
    Example 3: Real-time performance monitoring.

    Tracks throughput, memory usage, and processing time.
    """
    print("=" * 80)
    print("EXAMPLE 3: Real-Time Performance Monitoring")
    print("=" * 80)

    suppliers = [{"supplier_id": f"SUP-{i:05d}"} for i in range(20000)]

    start_time = time.time()
    total_processed = 0
    throughputs = []

    print(f"\nProcessing {len(suppliers)} suppliers...")
    print(f"Target: 100,000 suppliers/hour")
    print()

    # async for chunk_idx, results, metrics in agent.process_suppliers_optimized(
    #     suppliers=suppliers,
    #     chunk_size=1000
    # ):
    #     total_processed += metrics['successful']
    #     elapsed = time.time() - start_time
    #     current_throughput = (total_processed / elapsed) * 3600
    #     throughputs.append(current_throughput)
    #
    #     print(f"Chunk {chunk_idx:02d} | "
    #           f"Processed: {total_processed:,} | "
    #           f"Current throughput: {current_throughput:,.0f}/hour | "
    #           f"Target: {'✓' if current_throughput >= 100000 else '✗'}")

    # Calculate statistics
    # avg_throughput = sum(throughputs) / len(throughputs)
    # print()
    # print(f"Average throughput: {avg_throughput:,.0f} suppliers/hour")
    # print(f"Target achieved: {'YES' if avg_throughput >= 100000 else 'NO'}")


async def example_error_handling():
    """
    Example 4: Robust error handling in batch processing.

    Demonstrates graceful handling of failed calculations.
    """
    print("=" * 80)
    print("EXAMPLE 4: Error Handling and Recovery")
    print("=" * 80)

    # Mix of valid and invalid supplier data
    suppliers = []
    for i in range(5000):
        if i % 100 == 0:
            # Invalid data (will fail)
            suppliers.append({"invalid": "data"})
        else:
            # Valid data
            suppliers.append({
                "supplier_id": f"SUP-{i:05d}",
                "spend_usd": 1000.0
            })

    print(f"\nProcessing {len(suppliers)} suppliers (with some invalid data)...")
    print()

    total_successful = 0
    total_failed = 0

    # async for chunk_idx, results, metrics in agent.process_suppliers_optimized(
    #     suppliers=suppliers,
    #     chunk_size=1000
    # ):
    #     total_successful += metrics['successful']
    #     total_failed += metrics['failed']
    #
    #     print(f"Chunk {chunk_idx}: "
    #           f"Success: {metrics['successful']}, "
    #           f"Failed: {metrics['failed']}")
    #
    #     if metrics['errors']:
    #         print(f"  Sample error: {metrics['errors'][0]['error']}")

    print()
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {total_successful / len(suppliers) * 100:.2f}%")


async def example_memory_optimization():
    """
    Example 5: Memory-efficient processing of very large datasets.

    Demonstrates garbage collection and memory management.
    """
    print("=" * 80)
    print("EXAMPLE 5: Memory-Efficient Large-Scale Processing")
    print("=" * 80)

    # Simulate 100K suppliers
    print("\nSimulating 100,000 suppliers processing...")
    print("Memory-efficient streaming with periodic garbage collection")
    print()

    def generate_supplier(index: int) -> Dict[str, Any]:
        """Generate supplier data on-the-fly to save memory."""
        return {
            "supplier_id": f"SUP-{index:06d}",
            "spend_usd": 1000.0,
            "product_name": f"Product {index % 100}"
        }

    # Generate suppliers lazily
    suppliers = [generate_supplier(i) for i in range(100000)]

    print(f"Generated {len(suppliers)} suppliers")
    print(f"Chunk size: 1,000")
    print(f"GC frequency: Every 10 chunks")
    print()

    chunks_processed = 0

    # async for chunk_idx, results, metrics in agent.process_suppliers_optimized(
    #     suppliers=suppliers,
    #     chunk_size=1000
    # ):
    #     chunks_processed += 1
    #
    #     if chunk_idx % 10 == 0:
    #         print(f"Chunk {chunk_idx:03d} | "
    #               f"Progress: {chunk_idx}% | "
    #               f"Throughput: {metrics['throughput_per_hour']:,.0f}/hour | "
    #               f"[GC triggered]")
    #     else:
    #         print(f"Chunk {chunk_idx:03d} | "
    #               f"Progress: {chunk_idx}% | "
    #               f"Throughput: {metrics['throughput_per_hour']:,.0f}/hour")

    print()
    print(f"Completed {chunks_processed} chunks")


async def example_custom_chunk_size():
    """
    Example 6: Tuning chunk size for optimal performance.

    Demonstrates different chunk sizes and their impact on throughput.
    """
    print("=" * 80)
    print("EXAMPLE 6: Chunk Size Optimization")
    print("=" * 80)

    suppliers = [{"supplier_id": f"SUP-{i:05d}"} for i in range(10000)]

    chunk_sizes = [100, 500, 1000, 2000, 5000]

    print(f"\nTesting different chunk sizes on {len(suppliers)} suppliers:")
    print()

    for chunk_size in chunk_sizes:
        print(f"Testing chunk_size={chunk_size}...")
        start = time.time()

        # async for chunk_idx, results, metrics in agent.process_suppliers_optimized(
        #     suppliers=suppliers,
        #     chunk_size=chunk_size
        # ):
        #     pass  # Just measure throughput

        elapsed = time.time() - start
        throughput = (len(suppliers) / elapsed) * 3600

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:,.0f} suppliers/hour")
        print()


def print_optimization_summary():
    """Print summary of optimization features."""
    print("=" * 80)
    print("BATCH OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()
    print("FEATURES IMPLEMENTED:")
    print("  1. Chunked Processing - Process 1,000 suppliers at a time")
    print("  2. Parallel Execution - asyncio.gather() for concurrent calculations")
    print("  3. Streaming Results - Generator pattern with yield")
    print("  4. Bulk Database Ops - Batch inserts instead of individual")
    print("  5. Memory Management - Periodic garbage collection")
    print("  6. Error Resilience - Continue processing on individual failures")
    print("  7. Real-time Metrics - Throughput tracking per chunk")
    print("  8. Configurable Chunks - Tunable chunk size for optimization")
    print()
    print("PERFORMANCE METRICS:")
    print("  - Target: 100,000 suppliers/hour")
    print("  - Chunk size: 1,000 suppliers")
    print("  - Expected time per chunk: 36 seconds")
    print("  - Parallel tasks per chunk: 1,000")
    print("  - Database batch size: 1,000 records")
    print()
    print("USAGE:")
    print("  async for chunk_idx, results, metrics in agent.process_suppliers_optimized(")
    print("      suppliers=suppliers,")
    print("      category=1,")
    print("      chunk_size=1000,")
    print("      db_connection=db")
    print("  ):")
    print("      # Process each chunk as it completes")
    print("      print(f'Throughput: {metrics[\"throughput_per_hour\"]:.0f}/hour')")
    print()


async def main():
    """Run all examples."""
    print_optimization_summary()
    print()

    # Note: Examples are demonstration code
    # Uncomment and run with actual agent instance
    print("NOTE: These examples require an initialized Scope3CalculatorAgent")
    print("      with a valid factor_broker instance.")
    print()
    print("To run examples:")
    print("  1. Initialize agent with factor_broker")
    print("  2. Uncomment the agent.process_suppliers_optimized() calls")
    print("  3. Run: python examples/batch_optimization_example.py")
    print()

    # await example_basic_batch_processing()
    # await example_streaming_with_database()
    # await example_performance_monitoring()
    # await example_error_handling()
    # await example_memory_optimization()
    # await example_custom_chunk_size()


if __name__ == "__main__":
    asyncio.run(main())
