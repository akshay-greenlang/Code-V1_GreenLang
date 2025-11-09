"""
Example 4: Multi-Tier Caching
==============================

Demonstrates L1/L2/L3 caching for performance optimization.
"""

import asyncio
from datetime import datetime
from greenlang.cache import initialize_cache_manager, get_cache_manager


async def main():
    """Run caching example."""
    # Initialize cache manager with L1 only
    initialize_cache_manager(
        enable_l1=True,
        enable_l2=False,
        enable_l3=False
    )

    cache = get_cache_manager()

    # Expensive computation simulation
    async def expensive_computation():
        print("  Computing (expensive)...")
        await asyncio.sleep(0.5)
        return {"result": 42, "computed_at": datetime.now().isoformat()}

    # First call - cache miss
    print("\nFirst call (cache miss):")
    start = datetime.now()
    result = await cache.get_or_compute(
        key="expensive_result",
        compute_fn=expensive_computation,
        ttl=3600
    )
    duration = (datetime.now() - start).total_seconds()
    print(f"  Duration: {duration:.3f}s")
    print(f"  Result: {result}")

    # Second call - cache hit
    print("\nSecond call (cache hit):")
    start = datetime.now()
    result = await cache.get_or_compute(
        key="expensive_result",
        compute_fn=expensive_computation,
        ttl=3600
    )
    duration = (datetime.now() - start).total_seconds()
    print(f"  Duration: {duration:.3f}s (should be much faster!)")

    # Get analytics
    analytics = cache.get_analytics()
    print(f"\nCache Analytics:")
    print(f"  Total requests: {analytics.total_requests}")
    print(f"  Hit rate: {analytics.hit_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
