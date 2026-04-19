"""
Performance Benchmarks for GL-003 UNIFIEDSTEAM

Tests throughput, latency, and scalability of key components:
- IAPWS-IF97 property calculations
- Enthalpy balance computations
- Recommendation generation
- Kafka message processing

Author: GL-003 Test Engineering Team
"""

import asyncio
import time
import statistics
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Callable
import pytest


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, samples: List[float]):
        self.name = name
        self.samples = samples
        self.count = len(samples)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples) * 1000

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.samples) * 1000 if len(self.samples) > 1 else 0

    @property
    def min_ms(self) -> float:
        return min(self.samples) * 1000

    @property
    def max_ms(self) -> float:
        return max(self.samples) * 1000

    @property
    def p50_ms(self) -> float:
        sorted_samples = sorted(self.samples)
        return sorted_samples[int(len(sorted_samples) * 0.5)] * 1000

    @property
    def p95_ms(self) -> float:
        sorted_samples = sorted(self.samples)
        return sorted_samples[int(len(sorted_samples) * 0.95)] * 1000

    @property
    def p99_ms(self) -> float:
        sorted_samples = sorted(self.samples)
        return sorted_samples[int(len(sorted_samples) * 0.99)] * 1000

    @property
    def throughput_per_sec(self) -> float:
        total_time = sum(self.samples)
        return self.count / total_time if total_time > 0 else 0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Count: {self.count}\n"
            f"  Mean: {self.mean_ms:.3f} ms\n"
            f"  Std: {self.std_ms:.3f} ms\n"
            f"  Min: {self.min_ms:.3f} ms\n"
            f"  Max: {self.max_ms:.3f} ms\n"
            f"  P50: {self.p50_ms:.3f} ms\n"
            f"  P95: {self.p95_ms:.3f} ms\n"
            f"  P99: {self.p99_ms:.3f} ms\n"
            f"  Throughput: {self.throughput_per_sec:.1f}/sec"
        )


def benchmark(func: Callable, iterations: int = 1000, warmup: int = 100) -> BenchmarkResult:
    """
    Run a benchmark on a function.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        samples.append(end - start)

    return BenchmarkResult(func.__name__, samples)


class TestIF97Performance:
    """Performance tests for IAPWS-IF97 calculations."""

    def test_property_calculation_throughput(self):
        """Test throughput of steam property calculations."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        def calc_properties():
            # Typical superheated steam conditions
            return compute_properties_pt(
                pressure_kpa=Decimal("2000"),
                temperature_c=Decimal("300"),
            )

        result = benchmark(calc_properties, iterations=10000)
        print(f"\n{result}")

        # Assertions
        assert result.mean_ms < 1.0, f"Mean latency {result.mean_ms}ms exceeds 1ms"
        assert result.p95_ms < 2.0, f"P95 latency {result.p95_ms}ms exceeds 2ms"
        assert result.throughput_per_sec > 1000, f"Throughput {result.throughput_per_sec}/s below 1000/s"

    def test_saturation_calculation_throughput(self):
        """Test throughput of saturation property calculations."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                get_saturation_properties,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        def calc_saturation():
            return get_saturation_properties(pressure_kpa=Decimal("1500"))

        result = benchmark(calc_saturation, iterations=10000)
        print(f"\n{result}")

        assert result.mean_ms < 0.5, f"Mean latency {result.mean_ms}ms exceeds 0.5ms"
        assert result.throughput_per_sec > 2000, f"Throughput below 2000/s"

    def test_region_determination_throughput(self):
        """Test throughput of IF97 region determination."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                determine_region,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        test_cases = [
            (Decimal("500"), Decimal("150")),   # Region 1
            (Decimal("1000"), Decimal("250")),  # Region 2
            (Decimal("2000"), Decimal("350")),  # Region 2
        ]

        def determine_regions():
            for p, t in test_cases:
                determine_region(pressure_kpa=p, temperature_c=t)

        result = benchmark(determine_regions, iterations=5000)
        print(f"\n{result}")

        assert result.mean_ms < 0.3, f"Mean latency {result.mean_ms}ms exceeds 0.3ms"


class TestEnthalpyBalancePerformance:
    """Performance tests for enthalpy balance calculations."""

    def test_balance_computation_throughput(self):
        """Test throughput of enthalpy balance computations."""
        try:
            from GL_Agents.GL003_UnifiedSteam.calculators.heat_balance_calculator import (
                HeatBalanceCalculator,
            )
        except ImportError:
            pytest.skip("Heat balance calculator not available")

        calculator = HeatBalanceCalculator()

        # Sample balance node data
        node_data = {
            "node_id": "HEADER_300PSIG",
            "inflows": [
                {"source": "BOILER_1", "mass_kg_s": Decimal("10.0"), "enthalpy_kj_kg": Decimal("2800")},
                {"source": "BOILER_2", "mass_kg_s": Decimal("8.0"), "enthalpy_kj_kg": Decimal("2790")},
            ],
            "outflows": [
                {"dest": "USER_1", "mass_kg_s": Decimal("5.0"), "enthalpy_kj_kg": Decimal("2780")},
                {"dest": "USER_2", "mass_kg_s": Decimal("6.0"), "enthalpy_kj_kg": Decimal("2775")},
                {"dest": "USER_3", "mass_kg_s": Decimal("7.0"), "enthalpy_kj_kg": Decimal("2770")},
            ],
        }

        def compute_balance():
            return calculator.compute_balance(node_data)

        result = benchmark(compute_balance, iterations=5000)
        print(f"\n{result}")

        assert result.mean_ms < 2.0, f"Mean latency {result.mean_ms}ms exceeds 2ms"
        assert result.throughput_per_sec > 500, f"Throughput below 500/s"


class TestRecommendationPerformance:
    """Performance tests for recommendation generation."""

    def test_recommendation_generation_throughput(self):
        """Test throughput of recommendation generation."""
        try:
            from GL_Agents.GL003_UnifiedSteam.optimization.recommendation_engine import (
                RecommendationEngine,
            )
        except ImportError:
            pytest.skip("Recommendation engine not available")

        engine = RecommendationEngine()

        # Sample optimization result
        optimization_result = {
            "desuperheater_id": "DSH_001",
            "current_outlet_temp_c": Decimal("280"),
            "target_outlet_temp_c": Decimal("275"),
            "spray_flow_change_pct": Decimal("5.0"),
            "expected_savings_kw": Decimal("150"),
            "confidence": Decimal("0.85"),
        }

        def generate_recommendation():
            return engine.generate(optimization_result)

        result = benchmark(generate_recommendation, iterations=2000)
        print(f"\n{result}")

        assert result.mean_ms < 5.0, f"Mean latency {result.mean_ms}ms exceeds 5ms"
        assert result.throughput_per_sec > 200, f"Throughput below 200/s"


class TestClimateCalculationsPerformance:
    """Performance tests for climate/emissions calculations."""

    def test_emissions_calculation_throughput(self):
        """Test throughput of CO2e calculations."""
        try:
            from GL_Agents.GL003_UnifiedSteam.climate.co2e_calculator import CO2eCalculator
            from GL_Agents.GL003_UnifiedSteam.climate.emission_factors import FuelType
        except ImportError:
            pytest.skip("Climate module not available")

        calculator = CO2eCalculator()

        def calc_emissions():
            return calculator.calculate_fuel_emissions(
                fuel_type=FuelType.NATURAL_GAS,
                fuel_consumption_gj=Decimal("1000"),
            )

        result = benchmark(calc_emissions, iterations=5000)
        print(f"\n{result}")

        assert result.mean_ms < 1.0, f"Mean latency {result.mean_ms}ms exceeds 1ms"
        assert result.throughput_per_sec > 1000, f"Throughput below 1000/s"

    def test_climate_impact_calculation_throughput(self):
        """Test throughput of complete climate impact calculations."""
        try:
            from GL_Agents.GL003_UnifiedSteam.climate.co2e_calculator import CO2eCalculator
            from GL_Agents.GL003_UnifiedSteam.climate.emission_factors import FuelType
        except ImportError:
            pytest.skip("Climate module not available")

        calculator = CO2eCalculator()

        def calc_complete_impact():
            return calculator.calculate_complete_impact(
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
                steam_production_kg=Decimal("1000000"),
                fuel_consumptions={
                    FuelType.NATURAL_GAS: Decimal("5000"),
                },
                electricity_kwh=Decimal("10000"),
            )

        result = benchmark(calc_complete_impact, iterations=1000)
        print(f"\n{result}")

        assert result.mean_ms < 10.0, f"Mean latency {result.mean_ms}ms exceeds 10ms"


class TestSchemaPerformance:
    """Performance tests for Kafka schema operations."""

    def test_schema_serialization_throughput(self):
        """Test throughput of Kafka message serialization."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                RawSignalSchema,
                SensorQuality,
            )
        except ImportError:
            pytest.skip("Schemas module not available")

        signal = RawSignalSchema(
            ts=datetime.now(timezone.utc),
            site="SITE_A",
            area="UTILITIES",
            asset="HEADER_300PSIG",
            tag="STEAM_PRESSURE",
            value=2068.0,
            unit="kPa_g",
            quality=SensorQuality.GOOD,
        )

        def serialize():
            return signal.to_kafka_dict()

        result = benchmark(serialize, iterations=10000)
        print(f"\n{result}")

        assert result.mean_ms < 0.1, f"Mean latency {result.mean_ms}ms exceeds 0.1ms"
        assert result.throughput_per_sec > 10000, f"Throughput below 10000/s"


class TestConcurrentPerformance:
    """Performance tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_property_calculations(self):
        """Test concurrent IF97 calculations."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        async def async_calc(p: Decimal, t: Decimal):
            return compute_properties_pt(pressure_kpa=p, temperature_c=t)

        # Generate test cases
        test_cases = [
            (Decimal(str(500 + i * 100)), Decimal(str(150 + i * 10)))
            for i in range(100)
        ]

        start = time.perf_counter()

        # Run concurrently
        tasks = [async_calc(p, t) for p, t in test_cases]
        results = await asyncio.gather(*tasks)

        end = time.perf_counter()
        total_time = end - start

        throughput = len(test_cases) / total_time
        print(f"\nConcurrent IF97 calculations:")
        print(f"  Total: {len(test_cases)} calculations")
        print(f"  Time: {total_time*1000:.1f} ms")
        print(f"  Throughput: {throughput:.1f}/sec")

        assert len(results) == len(test_cases)
        assert throughput > 500, f"Concurrent throughput {throughput}/s below 500/s"


class TestMemoryPerformance:
    """Performance tests for memory usage."""

    def test_memory_stability_under_load(self):
        """Test that memory usage is stable under sustained load."""
        import sys

        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Get initial memory
        initial_objects = len([o for o in gc.get_objects()])

        # Run many calculations
        for _ in range(10000):
            result = compute_properties_pt(
                pressure_kpa=Decimal("2000"),
                temperature_c=Decimal("300"),
            )
            del result

        # Force garbage collection
        import gc
        gc.collect()

        # Check memory growth
        final_objects = len([o for o in gc.get_objects()])
        growth = final_objects - initial_objects

        print(f"\nMemory stability test:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Growth: {growth}")

        # Allow some growth but not excessive
        assert growth < 1000, f"Object growth {growth} exceeds threshold"


# Performance targets from implementation guide
PERFORMANCE_TARGETS = {
    "if97_property_calculation_ms": 1.0,
    "if97_saturation_calculation_ms": 0.5,
    "enthalpy_balance_computation_ms": 2.0,
    "recommendation_generation_ms": 5.0,
    "emissions_calculation_ms": 1.0,
    "schema_serialization_ms": 0.1,
    "concurrent_throughput_per_sec": 500,
}


def print_performance_report():
    """Print performance test summary against targets."""
    print("\n" + "=" * 60)
    print("GL-003 UNIFIEDSTEAM Performance Targets")
    print("=" * 60)
    for metric, target in PERFORMANCE_TARGETS.items():
        unit = "ms" if metric.endswith("_ms") else "/sec"
        print(f"  {metric}: {target} {unit}")
    print("=" * 60)


if __name__ == "__main__":
    print_performance_report()
    pytest.main([__file__, "-v", "-s"])
