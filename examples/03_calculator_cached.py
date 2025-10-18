#!/usr/bin/env python3
"""
Example 3: Calculator with Caching and Determinism
===================================================

This example demonstrates:
- Function caching for expensive calculations
- Deterministic execution with seeded randomness
- Performance comparison between cached and uncached runs

Run: python examples/03_calculator_cached.py
"""

import json
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, Any
from functools import lru_cache
from greenlang.sdk.base import Agent, Result, Metadata


class CachedEmissionsCalculator(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Emissions calculator with caching for performance optimization.

    Uses LRU cache for emission factor lookups and calculation results.
    Supports deterministic mode for reproducible results.
    """

    def __init__(self, deterministic: bool = True, seed: int = 42):
        metadata = Metadata(
            id="cached_emissions_calculator",
            name="Cached Emissions Calculator",
            version="1.0.0",
            description="High-performance calculator with caching and determinism",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        self.deterministic = deterministic
        self.seed = seed

        # Set random seed for deterministic behavior
        if self.deterministic:
            random.seed(self.seed)

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors_data = json.load(f)

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    @lru_cache(maxsize=128)
    def _get_emission_factor(self, country: str, fuel_type: str) -> float:
        """
        Get emission factor with caching.

        Uses LRU cache to avoid repeated dictionary lookups.
        """
        try:
            return self.factors_data["factors"][fuel_type][country]["value"]
        except KeyError:
            # Fallback to US factors
            return self.factors_data["factors"][fuel_type]["US"]["value"]

    def _calculate_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Calculate stable hash of input for caching key"""
        # Sort keys for deterministic hash
        input_str = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(input_str.encode()).hexdigest()[:16]

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required = ["energy_data"]
        if not all(field in input_data for field in required):
            return False

        energy_data = input_data["energy_data"]
        if not isinstance(energy_data, list):
            return False

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions with caching and optional uncertainty.

        Supports deterministic mode for reproducible results.
        """
        energy_data = input_data["energy_data"]
        country = input_data.get("country", "US")
        include_uncertainty = input_data.get("include_uncertainty", False)

        # Calculate hash for this calculation
        calc_hash = self._calculate_input_hash(input_data)

        results = []
        total_emissions = 0.0

        for entry in energy_data:
            fuel_type = entry["fuel_type"]
            consumption = entry["consumption"]

            # Get emission factor (cached)
            factor = self._get_emission_factor(country, fuel_type)

            # Calculate base emissions
            emissions_kg = consumption * factor

            # Add uncertainty if requested (deterministic if seed set)
            if include_uncertainty:
                # Apply +/- 5% uncertainty
                uncertainty_factor = 1.0 + random.uniform(-0.05, 0.05)
                emissions_kg *= uncertainty_factor

            emissions_tons = emissions_kg / 1000
            total_emissions += emissions_tons

            results.append({
                "fuel_type": fuel_type,
                "consumption": consumption,
                "factor": factor,
                "emissions_tons": round(emissions_tons, 4),
                "unit": entry.get("unit", "")
            })

        return {
            "calculation_hash": calc_hash,
            "total_emissions_tons": round(total_emissions, 4),
            "breakdown": results,
            "country": country,
            "deterministic": self.deterministic,
            "seed": self.seed if self.deterministic else None,
            "cache_info": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "size": len(self._get_emission_factor.cache_info().currsize)
                        if hasattr(self._get_emission_factor, 'cache_info') else 0
            }
        }


def benchmark_caching(calculator, input_data, iterations=1000):
    """Benchmark caching performance"""
    print(f"\nBenchmarking with {iterations} iterations...")
    print("-" * 70)

    # Warm-up
    calculator.run(input_data)

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        calculator.run(input_data)
    elapsed = time.time() - start_time

    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Average per calculation: {(elapsed/iterations)*1000:.3f} ms")
    print(f"  Calculations per second: {iterations/elapsed:.0f}")


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 3: Calculator with Caching and Determinism")
    print("="*70 + "\n")

    # Example data
    input_data = {
        "energy_data": [
            {"fuel_type": "electricity", "consumption": 50000, "unit": "kWh"},
            {"fuel_type": "natural_gas", "consumption": 1000, "unit": "therms"},
            {"fuel_type": "electricity", "consumption": 25000, "unit": "kWh"},
            {"fuel_type": "natural_gas", "consumption": 500, "unit": "therms"}
        ],
        "country": "US"
    }

    # Test 1: Basic calculation
    print("Test 1: Basic Cached Calculation")
    print("-" * 70)

    calculator = CachedEmissionsCalculator(deterministic=True, seed=42)
    result = calculator.run(input_data)

    if result.success:
        print(f"Calculation Hash: {result.data['calculation_hash']}")
        print(f"Total Emissions: {result.data['total_emissions_tons']:.4f} tCO2e")
        print(f"\nBreakdown:")
        for item in result.data['breakdown']:
            print(f"  {item['fuel_type']}: {item['emissions_tons']:.4f} tons")

    # Test 2: Deterministic reproducibility
    print("\n\nTest 2: Deterministic Reproducibility")
    print("-" * 70)

    # Add uncertainty but with same seed
    input_data_uncertain = input_data.copy()
    input_data_uncertain["include_uncertainty"] = True

    calc1 = CachedEmissionsCalculator(deterministic=True, seed=42)
    calc2 = CachedEmissionsCalculator(deterministic=True, seed=42)

    result1 = calc1.run(input_data_uncertain)
    result2 = calc2.run(input_data_uncertain)

    print(f"Run 1: {result1.data['total_emissions_tons']:.4f} tCO2e")
    print(f"Run 2: {result2.data['total_emissions_tons']:.4f} tCO2e")
    print(f"Identical: {result1.data['total_emissions_tons'] == result2.data['total_emissions_tons']}")

    # Test 3: Different seeds produce different results
    print("\n\nTest 3: Different Seeds, Different Results")
    print("-" * 70)

    calc3 = CachedEmissionsCalculator(deterministic=True, seed=123)
    result3 = calc3.run(input_data_uncertain)

    print(f"Seed 42: {result1.data['total_emissions_tons']:.4f} tCO2e")
    print(f"Seed 123: {result3.data['total_emissions_tons']:.4f} tCO2e")
    print(f"Different: {result1.data['total_emissions_tons'] != result3.data['total_emissions_tons']}")

    # Test 4: Performance benchmark
    print("\n\nTest 4: Performance Benchmark")
    print("-" * 70)

    calculator_perf = CachedEmissionsCalculator(deterministic=True, seed=42)
    benchmark_caching(calculator_perf, input_data, iterations=1000)

    # Show cache statistics
    cache_info = calculator_perf._get_emission_factor.cache_info()
    print(f"\nCache Statistics:")
    print(f"  Cache hits: {cache_info.hits}")
    print(f"  Cache misses: {cache_info.misses}")
    print(f"  Hit rate: {cache_info.hits/(cache_info.hits + cache_info.misses)*100:.1f}%")
    print(f"  Current cache size: {cache_info.currsize}")
    print(f"  Max cache size: {cache_info.maxsize}")

    # Test 5: Multi-country caching
    print("\n\nTest 5: Multi-Country Caching")
    print("-" * 70)

    countries = ["US", "UK", "CA"]
    for country in countries:
        country_input = input_data.copy()
        country_input["country"] = country
        result = calculator_perf.run(country_input)
        print(f"{country}: {result.data['total_emissions_tons']:.4f} tCO2e")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
