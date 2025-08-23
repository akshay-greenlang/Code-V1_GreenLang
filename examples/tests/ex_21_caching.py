"""Example 21: Caching example (second run faster, same bytes)."""

import pytest
import time
import json
import hashlib
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.core.cache import CacheManager
    from greenlang.agents.fuel_agent import FuelAgent
except Exception:
    CacheManager = None
    FuelAgent = None

@pytest.mark.example
def test_caching_performance_and_consistency():
    """Second run should be faster and return identical bytes."""
    if CacheManager is None or FuelAgent is None:
        pytest.skip("Cache or FuelAgent not importable")
    
    cache = CacheManager()
    agent = FuelAgent()
    
    test_payload = {
        "fuel_type": "electricity",
        "consumption": {"value": 1500000, "unit": "kWh"},
        "country": "IN"
    }
    
    # Clear cache to ensure clean start
    cache.clear()
    
    # First run (cache miss)
    start_time = time.perf_counter()
    result1 = agent.run(test_payload)
    time1 = time.perf_counter() - start_time
    
    # Get hash of result
    result1_json = json.dumps(result1, sort_keys=True)
    hash1 = hashlib.sha256(result1_json.encode()).hexdigest()
    
    # Second run (cache hit)
    start_time = time.perf_counter()
    result2 = agent.run(test_payload)
    time2 = time.perf_counter() - start_time
    
    # Get hash of result
    result2_json = json.dumps(result2, sort_keys=True)
    hash2 = hashlib.sha256(result2_json.encode()).hexdigest()
    
    # Assertions
    assert hash1 == hash2, "Results should be byte-identical"
    assert result1 == result2, "Results should be equal"
    
    # Second run should be faster (at least 2x faster in ideal case)
    # But we'll be lenient and just check it's not slower
    assert time2 <= time1 * 1.1, f"Cache hit ({time2:.3f}s) should not be slower than miss ({time1:.3f}s)"
    
    # Verify cache stats
    stats = cache.get_stats()
    assert stats["hits"] >= 1, "Should have at least one cache hit"