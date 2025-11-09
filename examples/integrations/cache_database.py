"""
Integration Example: CacheManager + DatabaseManager
====================================================

Demonstrates how to integrate caching with database operations.
"""

import asyncio
from datetime import datetime
import pandas as pd
from greenlang.cache import initialize_cache_manager, get_cache_manager
from greenlang.db import DatabaseManager


async def main():
    """Run CacheManager + DatabaseManager integration."""
    print("\nCacheManager + DatabaseManager Integration")
    print("=" * 60)

    # Initialize cache
    initialize_cache_manager(enable_l1=True)
    cache = get_cache_manager()

    # Initialize database
    db = DatabaseManager(connection_string="sqlite:///cache_db_example.db")

    # Expensive database query function
    async def fetch_emissions_from_db(facility_id: str) -> dict:
        """Fetch emissions data from database (expensive operation)."""
        print(f"  Querying database for facility: {facility_id}...")

        # Simulate expensive query
        await asyncio.sleep(0.5)

        query = f"""
            SELECT facility_id, emissions, year
            FROM emissions
            WHERE facility_id = '{facility_id}'
        """

        # In real scenario, execute query
        # For demo, return mock data
        return {
            "facility_id": facility_id,
            "emissions": 1500.5,
            "year": 2024,
            "retrieved_at": datetime.now().isoformat()
        }

    # Function with cache-aside pattern
    async def get_facility_emissions(facility_id: str) -> dict:
        """Get facility emissions with caching."""
        cache_key = f"emissions:{facility_id}"

        # Try cache first
        cached_data = await cache.get(cache_key)
        if cached_data:
            print(f"  ✓ Cache HIT for {facility_id}")
            return cached_data

        # Cache miss - fetch from database
        print(f"  ✗ Cache MISS for {facility_id}")
        data = await fetch_emissions_from_db(facility_id)

        # Store in cache with TTL
        await cache.set(cache_key, data, ttl=3600)

        return data

    # Test 1: First call (cache miss)
    print("\n[Test 1] First call - Cache MISS expected:")
    start = datetime.now()
    data1 = await get_facility_emissions("FAC-001")
    duration1 = (datetime.now() - start).total_seconds()

    print(f"  Result: {data1}")
    print(f"  Duration: {duration1:.3f}s")

    # Test 2: Second call (cache hit)
    print("\n[Test 2] Second call - Cache HIT expected:")
    start = datetime.now()
    data2 = await get_facility_emissions("FAC-001")
    duration2 = (datetime.now() - start).total_seconds()

    print(f"  Result: {data2}")
    print(f"  Duration: {duration2:.3f}s")
    print(f"  Speedup: {duration1 / duration2:.1f}x")

    # Test 3: Different facility (cache miss)
    print("\n[Test 3] Different facility - Cache MISS expected:")
    start = datetime.now()
    data3 = await get_facility_emissions("FAC-002")
    duration3 = (datetime.now() - start).total_seconds()

    print(f"  Result: {data3}")
    print(f"  Duration: {duration3:.3f}s")

    # Cache statistics
    analytics = cache.get_analytics()
    print(f"\nCache Analytics:")
    print(f"  Total requests: {analytics.total_requests}")
    print(f"  Cache hits: {analytics.cache_hits}")
    print(f"  Cache misses: {analytics.cache_misses}")
    print(f"  Hit rate: {analytics.hit_rate:.1f}%")

    print("\n" + "=" * 60)
    print("Integration Pattern: Cache-Aside")
    print("  1. Check cache first")
    print("  2. On miss, query database")
    print("  3. Store result in cache")
    print("  4. Return data")
    print("=" * 60)

    # Cleanup
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
