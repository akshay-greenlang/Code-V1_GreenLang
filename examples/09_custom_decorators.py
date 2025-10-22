#!/usr/bin/env python3
"""
Example 9: Custom Decorators for Agent Methods
===============================================

This example demonstrates using custom decorators:
- @deterministic: Ensure reproducible results
- @cached: Cache function results
- @traced: Log execution details
- @timed: Measure execution time
- @validated: Automatic input validation

Run: python examples/09_custom_decorators.py
"""

import time
import hashlib
import json
import random
from functools import wraps, lru_cache
from typing import Dict, Any, Callable
from pathlib import Path
from greenlang.sdk.base import Agent, Result, Metadata


# Custom Decorators
def deterministic(seed: int = 42):
    """Decorator to ensure deterministic execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set random seed
            random.seed(seed)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cached(func):
    """Decorator for LRU caching"""
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        cache_key = str(args) + str(sorted(kwargs.items()))
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

        if cache_hash in cache:
            print(f"    [CACHE HIT] {func.__name__}")
            return cache[cache_hash]

        print(f"    [CACHE MISS] {func.__name__}")
        result = func(*args, **kwargs)
        cache[cache_hash] = result
        return result

    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    wrapper.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}
    return wrapper


def traced(func):
    """Decorator to trace execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"    [TRACE] Entering {func.__name__}")
        print(f"    [TRACE] Args: {args[1:] if len(args) > 1 else 'none'}")  # Skip self
        print(f"    [TRACE] Kwargs: {kwargs if kwargs else 'none'}")

        result = func(*args, **kwargs)

        print(f"    [TRACE] Exiting {func.__name__}")
        print(f"    [TRACE] Result type: {type(result).__name__}")
        return result

    return wrapper


def timed(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start_time) * 1000  # ms

        print(f"    [TIMING] {func.__name__} took {elapsed:.2f} ms")
        return result

    return wrapper


def validated(schema: Dict[str, type]):
    """Decorator for input validation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the input data (assuming second argument after self)
            if len(args) > 1:
                input_data = args[1]

                # Validate schema
                for field, expected_type in schema.items():
                    if field not in input_data:
                        raise ValueError(f"Missing required field: {field}")
                    if not isinstance(input_data[field], expected_type):
                        raise TypeError(
                            f"Field {field} must be {expected_type.__name__}, "
                            f"got {type(input_data[field]).__name__}"
                        )

                print(f"    [VALIDATION] Input validated successfully")

            return func(*args, **kwargs)

        return wrapper
    return decorator


class DecoratedCalculatorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Calculator agent demonstrating custom decorators.
    """

    def __init__(self):
        metadata = Metadata(
            id="decorated_calculator",
            name="Decorated Calculator Agent",
            version="1.0.0",
            description="Agent demonstrating custom decorators",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Basic validation"""
        return "energy_data" in input_data

    @timed
    @traced
    @deterministic(seed=42)
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with multiple decorators applied"""
        return self._calculate_emissions(input_data)

    @validated(schema={"energy_data": list, "country": str})
    @cached
    def _calculate_emissions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emissions with validation and caching"""
        # Simulate expensive calculation
        time.sleep(0.1)

        energy_data = input_data["energy_data"]
        country = input_data.get("country", "US")

        total_emissions = 0.0
        for entry in energy_data:
            fuel_type = entry["fuel_type"]
            consumption = entry["consumption"]
            factor = self.factors[fuel_type][country]["value"]
            emissions_tons = (consumption * factor) / 1000
            total_emissions += emissions_tons

        # Add some randomness (deterministic due to decorator)
        uncertainty = random.uniform(0.95, 1.05)
        total_emissions *= uncertainty

        return {
            "total_emissions_tons": round(total_emissions, 4),
            "country": country,
            "uncertainty_applied": round(uncertainty, 4)
        }

    @lru_cache(maxsize=32)
    def get_emission_factor(self, fuel_type: str, country: str) -> float:
        """Get emission factor with built-in LRU cache"""
        return self.factors[fuel_type][country]["value"]


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 9: Custom Decorators for Agent Methods")
    print("="*70 + "\n")

    agent = DecoratedCalculatorAgent()

    test_input = {
        "energy_data": [
            {"fuel_type": "electricity", "consumption": 50000},
            {"fuel_type": "natural_gas", "consumption": 1000}
        ],
        "country": "US"
    }

    # Test 1: First run (shows all decorators in action)
    print("Test 1: First Run (All Decorators Active)")
    print("-" * 70)

    result1 = agent.run(test_input)
    if result1.success:
        print(f"  Result: {result1.data['total_emissions_tons']:.4f} tCO2e")

    # Test 2: Second run (should hit cache)
    print("\n\nTest 2: Second Run (Cache Hit Expected)")
    print("-" * 70)

    result2 = agent.run(test_input)
    if result2.success:
        print(f"  Result: {result2.data['total_emissions_tons']:.4f} tCO2e")
        print(f"  Same as first: {result1.data['total_emissions_tons'] == result2.data['total_emissions_tons']}")

    # Test 3: Deterministic behavior
    print("\n\nTest 3: Deterministic Behavior (Same Seed)")
    print("-" * 70)

    # Clear cache to force recalculation
    agent._calculate_emissions.clear_cache()

    result3 = agent.run(test_input)
    if result3.success:
        print(f"  Run 1: {result1.data['total_emissions_tons']:.4f} tCO2e")
        print(f"  Run 3: {result3.data['total_emissions_tons']:.4f} tCO2e")
        print(f"  Deterministic: {result1.data['total_emissions_tons'] == result3.data['total_emissions_tons']}")

    # Test 4: Cache statistics
    print("\n\nTest 4: Cache Statistics")
    print("-" * 70)

    cache_info = agent._calculate_emissions.cache_info()
    print(f"  Cache size: {cache_info['size']}")
    print(f"  Cache keys: {len(cache_info['keys'])}")

    # Test 5: Validation error
    print("\n\nTest 5: Validation Error Handling")
    print("-" * 70)

    bad_input = {
        "energy_data": "not a list",  # Should be list
        "country": "US"
    }

    try:
        agent.run(bad_input)
    except (ValueError, TypeError) as e:
        print(f"  Caught expected error: {e}")

    # Test 6: LRU cache for emission factors
    print("\n\nTest 6: Built-in LRU Cache for Factors")
    print("-" * 70)

    print("  Getting factors (first time):")
    factor1 = agent.get_emission_factor("electricity", "US")
    print(f"    Electricity/US: {factor1}")

    print("  Getting same factors (cached):")
    factor2 = agent.get_emission_factor("electricity", "US")
    print(f"    Electricity/US: {factor2}")

    # Show LRU cache info
    lru_info = agent.get_emission_factor.cache_info()
    print(f"\n  LRU Cache Stats:")
    print(f"    Hits: {lru_info.hits}")
    print(f"    Misses: {lru_info.misses}")
    print(f"    Size: {lru_info.currsize}/{lru_info.maxsize}")

    # Test 7: Performance comparison
    print("\n\nTest 7: Performance - Cached vs Uncached")
    print("-" * 70)

    # Clear cache
    agent._calculate_emissions.clear_cache()

    # Uncached run
    start = time.time()
    agent.run(test_input)
    uncached_time = (time.time() - start) * 1000

    # Cached run
    start = time.time()
    agent.run(test_input)
    cached_time = (time.time() - start) * 1000

    print(f"  Uncached: {uncached_time:.2f} ms")
    print(f"  Cached: {cached_time:.2f} ms")
    print(f"  Speedup: {uncached_time/cached_time:.2f}x")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
