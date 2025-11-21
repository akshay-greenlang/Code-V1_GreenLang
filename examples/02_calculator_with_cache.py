#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2: Calculator with Caching

Demonstrates a calculator agent with caching for repeated calculations
and use of Decimal for precision.

Key Concepts:
- Caching expensive calculations
- Decimal precision for financial calculations
- Unit conversion
- Performance optimization

Usage:
    python 02_calculator_with_cache.py
"""

from greenlang.sdk import Agent, Result
from pydantic import BaseModel, Field
from decimal import Decimal, getcontext
from typing import Dict, Tuple
import hashlib
import time
from greenlang.determinism import FinancialDecimal

# Set decimal precision
getcontext().prec = 10


class CalculationInput(BaseModel):
    """Input for emissions calculation"""
    fuel_type: str = Field(..., description="Type of fuel")
    consumption: Decimal = Field(..., gt=0, description="Consumption amount")
    unit: str = Field(default="kWh", description="Unit of measurement")


class CachedCalculatorAgent(Agent[CalculationInput, Dict]):
    """
    Calculator agent with built-in caching for performance.

    Caches results based on input hash to avoid redundant calculations.
    Uses Decimal for precise financial calculations.
    """

    def __init__(self):
        super().__init__(
            metadata={
                "id": "cached_calculator",
                "name": "Cached Emissions Calculator",
                "version": "1.0.0"
            }
        )
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Emission factors (as Decimal for precision)
        self.factors = {
            "electricity": Decimal("0.417"),  # kgCO2e/kWh
            "natural_gas": Decimal("5.3"),     # kgCO2e/therm
            "diesel": Decimal("2.68"),         # kgCO2e/liter
        }

        # Unit conversions
        self.unit_conversions = {
            ("MWh", "kWh"): Decimal("1000"),
            ("therms", "therms"): Decimal("1"),
            ("gallons", "liters"): Decimal("3.785"),
        }

    def validate(self, input_data: CalculationInput) -> bool:
        """Validate input"""
        return input_data.fuel_type in self.factors

    def _get_cache_key(self, input_data: CalculationInput) -> str:
        """Generate cache key from input"""
        key_string = f"{input_data.fuel_type}:{input_data.consumption}:{input_data.unit}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _convert_units(self, consumption: Decimal, from_unit: str, to_unit: str = "kWh") -> Decimal:
        """Convert units to standard unit"""
        conversion_key = (from_unit, to_unit)
        if conversion_key in self.unit_conversions:
            return consumption * self.unit_conversions[conversion_key]
        return consumption

    def process(self, input_data: CalculationInput) -> Dict:
        """
        Calculate emissions with caching.

        Args:
            input_data: Calculation input

        Returns:
            Dictionary with emissions and calculation metadata
        """
        # Check cache
        cache_key = self._get_cache_key(input_data)

        if cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            return {
                **cached_result,
                "from_cache": True,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses
            }

        # Cache miss - perform calculation
        self.cache_misses += 1
        start_time = time.time()

        # Convert units if needed
        consumption_standard = self._convert_units(
            input_data.consumption,
            input_data.unit
        )

        # Get emission factor
        factor = self.factors[input_data.fuel_type]

        # Calculate emissions (using Decimal for precision)
        emissions_kg = consumption_standard * factor
        emissions_tons = emissions_kg / Decimal("1000")

        # Simulate expensive calculation
        time.sleep(0.01)  # 10ms delay

        calculation_time = time.time() - start_time

        result = {
            "fuel_type": input_data.fuel_type,
            "consumption": FinancialDecimal.from_string(input_data.consumption),
            "unit": input_data.unit,
            "emissions_kg": FinancialDecimal.from_string(emissions_kg),
            "emissions_tons": FinancialDecimal.from_string(emissions_tons),
            "emission_factor": FinancialDecimal.from_string(factor),
            "calculation_time_ms": calculation_time * 1000,
            "from_cache": False,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }

        # Store in cache
        self.cache[cache_key] = result

        return result


def main():
    """Demonstrate cached calculator"""
    print("=" * 60)
    print("Example 2: Calculator with Caching")
    print("=" * 60)

    # Initialize calculator
    calculator = CachedCalculatorAgent()
    print(f"\nInitialized {calculator.metadata['name']}")
    print("Features: Caching, Decimal precision, Unit conversion")

    # Test calculations
    print("\n--- First Run (Cache Misses) ---\n")

    test_cases = [
        ("electricity", Decimal("1000"), "kWh"),
        ("natural_gas", Decimal("500"), "therms"),
        ("diesel", Decimal("100"), "liters"),
        ("electricity", Decimal("1000"), "kWh"),  # Duplicate
        ("natural_gas", Decimal("500"), "therms"),  # Duplicate
    ]

    for fuel_type, consumption, unit in test_cases:
        input_data = CalculationInput(
            fuel_type=fuel_type,
            consumption=consumption,
            unit=unit
        )

        result = calculator.run(input_data)

        if result.success:
            data = result.data
            cache_status = "CACHE HIT" if data["from_cache"] else "CACHE MISS"
            print(f"{cache_status} - {fuel_type.title()}: {consumption} {unit}")
            print(f"  Emissions: {data['emissions_kg']:.2f} kg CO2e "
                  f"({data['emissions_tons']:.3f} tons)")
            print(f"  Calculation time: {data['calculation_time_ms']:.2f} ms")
            print(f"  Cache stats: {data['cache_hits']} hits, {data['cache_misses']} misses")
            print()

    # Performance comparison
    print("--- Performance Analysis ---\n")

    # Test without cache
    calculator_no_cache = CachedCalculatorAgent()
    start = time.time()
    for _ in range(100):
        input_data = CalculationInput(fuel_type="electricity", consumption=Decimal("1000"), unit="kWh")
        calculator_no_cache.process(input_data)
    no_cache_time = time.time() - start

    # Test with cache
    calculator_with_cache = CachedCalculatorAgent()
    input_data = CalculationInput(fuel_type="electricity", consumption=Decimal("1000"), unit="kWh")

    # Prime cache
    calculator_with_cache.run(input_data)

    # Test cached performance
    start = time.time()
    for _ in range(100):
        calculator_with_cache.process(input_data)
    cached_time = time.time() - start

    speedup = no_cache_time / cached_time if cached_time > 0 else 0

    print(f"100 calculations without cache: {no_cache_time*1000:.2f} ms")
    print(f"100 calculations with cache: {cached_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.1f}x faster with caching")

    # Cache efficiency
    hit_rate = (calculator.cache_hits / (calculator.cache_hits + calculator.cache_misses) * 100
                if (calculator.cache_hits + calculator.cache_misses) > 0 else 0)

    print(f"\nCache Hit Rate: {hit_rate:.1f}%")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("  - Caching provides significant performance improvements")
    print("  - Decimal precision for financial accuracy")
    print("  - Unit conversion built-in")
    print("  - Cache hit tracking for monitoring")


if __name__ == "__main__":
    main()
