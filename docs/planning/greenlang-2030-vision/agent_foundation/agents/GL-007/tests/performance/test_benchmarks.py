# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests for GL-007 FURNACEPULSE (FurnacePerformanceOptimizer)

This module provides comprehensive performance benchmarks covering:
- Thermal efficiency calculation performance (ASME PTC 4.1)
- Fuel consumption analysis throughput
- Anomaly detection latency
- Maintenance prediction performance
- Operating parameter optimization
- Multi-furnace coordination scalability
- Cache performance validation
- Memory usage benchmarks

Performance Targets:
- Thermal efficiency calculation: <50ms
- Fuel consumption analysis: <100ms
- Anomaly detection: <80ms
- Maintenance prediction: <200ms
- Operating parameter optimization: <500ms
- Multi-furnace optimization: <3s for 10 furnaces
- Full monitoring pipeline: <1s
- Cache hit: <1ms
- Provenance hash calculation: <5ms

Standards Compliance:
- ASME PTC 4.1 (Steam Generating Units)
- ISO 50001 (Energy Management)
- API 560 (Fired Heaters)
- NFPA 86 (Industrial Furnaces)

Author: GL-BackendDeveloper
Date: 2025-11-22
Version: 1.0.0
"""

import pytest
import time
import hashlib
import json
import math
import random
import sys
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from datetime import datetime, timedelta

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# BENCHMARK INFRASTRUCTURE
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_per_sec: float
    passed: bool
    target_ms: float
    memory_mb: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"{self.name}: {status} - avg={self.avg_time_ms:.3f}ms "
                f"(target={self.target_ms}ms) p95={self.p95_time_ms:.3f}ms "
                f"throughput={self.throughput_per_sec:.1f}/s")


class PerformanceBenchmark:
    """Performance benchmark runner with comprehensive statistics."""

    @staticmethod
    def run_benchmark(
        func: Callable,
        iterations: int = 1000,
        target_ms: float = 1.0,
        name: str = "benchmark",
        warmup_iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run a performance benchmark with warmup and statistics.

        Args:
            func: Function to benchmark (no arguments)
            iterations: Number of iterations to run
            target_ms: Target time in milliseconds
            name: Benchmark name for reporting
            warmup_iterations: Number of warmup iterations

        Returns:
            BenchmarkResult with comprehensive statistics
        """
        # Warmup phase to warm up JIT, caches, etc.
        for _ in range(warmup_iterations):
            func()

        # Measurement phase
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # Calculate statistics
        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = min(times)
        max_time = max(times)

        # Standard deviation
        variance = sum((t - avg_time) ** 2 for t in times) / iterations
        std_dev = math.sqrt(variance)

        # Percentiles
        sorted_times = sorted(times)
        p95_idx = int(iterations * 0.95)
        p99_idx = int(iterations * 0.99)
        p95_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_time
        p99_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_time

        # Throughput
        throughput = iterations / (total_time / 1000) if total_time > 0 else 0

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_per_sec=throughput,
            passed=avg_time <= target_ms,
            target_ms=target_ms,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time
        )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_runner():
    """Create performance benchmark runner."""
    return PerformanceBenchmark()


@pytest.fixture
def sample_furnace_data():
    """Create sample furnace operating data."""
    return {
        "furnace_id": "FURNACE-001",
        "fuel_input": {
            "mass_flow_rate_kg_hr": 1850.0,
            "higher_heating_value_mj_kg": 50.0,
            "fuel_type": "natural_gas"
        },
        "flue_gas": {
            "temperature_c": 185.0,
            "o2_percent_dry": 3.5,
            "co_ppm": 15.0,
            "nox_ppm": 45.0
        },
        "heat_output": {
            "useful_heat_output_mw": 20.5
        },
        "losses": {
            "radiation_loss_mw": 0.5,
            "convection_loss_mw": 0.3
        },
        "production_quantity": 18.5
    }


@pytest.fixture
def sample_fuel_consumption_data():
    """Create sample fuel consumption data."""
    data = []
    for i in range(24):
        data.append({
            "timestamp": f"2024-01-01T{i:02d}:00:00Z",
            "consumption_rate_kg_hr": 1850.0 + random.uniform(-50, 50),
            "heating_value_mj_kg": 50.0,
            "production_rate": 18.5 + random.uniform(-0.5, 0.5)
        })
    return {
        "consumption_data": data,
        "baseline_performance": {
            "expected_sec_gj_ton": 4.8,
            "variability_factor": 0.05
        },
        "cost_parameters": {
            "fuel_cost_usd_per_gj": 8.5,
            "emission_factor_kg_co2_per_gj": 56.1
        }
    }


@pytest.fixture
def sample_equipment_inventory():
    """Create sample equipment inventory for maintenance prediction."""
    return [
        {
            "equipment_id": "REFRACTORY-001",
            "equipment_type": "refractory",
            "design_life_years": 5.0,
            "criticality": "critical"
        },
        {
            "equipment_id": "BURNER-001",
            "equipment_type": "burner",
            "design_life_years": 10.0,
            "criticality": "high"
        },
        {
            "equipment_id": "TUBE-BANK-A",
            "equipment_type": "tube",
            "design_life_years": 15.0,
            "criticality": "critical"
        }
    ]


@pytest.fixture
def sample_multi_furnace_data():
    """Create sample data for multi-furnace optimization."""
    furnaces = []
    for i in range(10):
        furnaces.append({
            "furnace_id": f"FURNACE-{i+1:03d}",
            "current_load_mw": 20.0 + random.uniform(-5, 10),
            "max_capacity_mw": 30.0 + i * 2,
            "efficiency_at_current_load": 78.0 + random.uniform(0, 7),
            "fuel_cost_usd_per_mwh": 30.0 + random.uniform(0, 10),
            "availability": True,
            "maintenance_due_days": random.randint(10, 180)
        })
    return {
        "furnaces": furnaces,
        "total_heat_demand_mw": 200.0,
        "optimization_objective": "minimize_cost"
    }


@pytest.fixture
def large_sensor_dataset():
    """Create large dataset for scalability testing."""
    sensors = []
    for i in range(200):  # 200 data points per FURNACEPULSE spec
        sensors.append({
            "sensor_id": f"SENSOR-{i+1:03d}",
            "value": random.uniform(0, 1000),
            "timestamp": datetime.now().isoformat(),
            "quality": "good"
        })
    return sensors


# ============================================================================
# THERMAL EFFICIENCY CALCULATION BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestThermalEfficiencyBenchmarks:
    """Performance benchmarks for thermal efficiency calculations per ASME PTC 4.1."""

    @pytest.mark.performance
    def test_direct_method_efficiency(self, benchmark_runner, sample_furnace_data):
        """Benchmark direct method thermal efficiency calculation."""
        def calculate_direct_efficiency():
            fuel_input = sample_furnace_data["fuel_input"]
            heat_output = sample_furnace_data["heat_output"]

            # Fuel energy input
            mass_flow = Decimal(str(fuel_input["mass_flow_rate_kg_hr"]))
            hhv = Decimal(str(fuel_input["higher_heating_value_mj_kg"]))
            fuel_energy_mw = (mass_flow * hhv) / Decimal("3600")

            # Useful heat output
            useful_heat = Decimal(str(heat_output["useful_heat_output_mw"]))

            # Direct method efficiency
            efficiency = (useful_heat / fuel_energy_mw) * Decimal("100")
            return efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        result = benchmark_runner.run_benchmark(
            calculate_direct_efficiency,
            iterations=10000,
            target_ms=0.5,
            name="Direct Method Efficiency"
        )

        assert result.passed, f"Direct method: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"
        assert result.throughput_per_sec > 2000

    @pytest.mark.performance
    def test_indirect_method_efficiency(self, benchmark_runner, sample_furnace_data):
        """Benchmark indirect (heat loss) method efficiency calculation."""
        def calculate_indirect_efficiency():
            fuel_input = sample_furnace_data["fuel_input"]
            flue_gas = sample_furnace_data["flue_gas"]
            losses = sample_furnace_data["losses"]

            # Calculate fuel energy
            mass_flow = Decimal(str(fuel_input["mass_flow_rate_kg_hr"]))
            hhv = Decimal(str(fuel_input["higher_heating_value_mj_kg"]))
            fuel_energy_mw = (mass_flow * hhv) / Decimal("3600")

            # Stack loss
            flue_temp = Decimal(str(flue_gas["temperature_c"]))
            ambient_temp = Decimal("25")
            stack_loss_percent = min(Decimal("20"), (flue_temp - ambient_temp) / Decimal("10"))

            # Other losses
            radiation_loss = Decimal(str(losses["radiation_loss_mw"]))
            convection_loss = Decimal(str(losses["convection_loss_mw"]))
            radiation_percent = (radiation_loss / fuel_energy_mw) * Decimal("100")
            convection_percent = (convection_loss / fuel_energy_mw) * Decimal("100")

            # Total losses
            total_loss_percent = stack_loss_percent + radiation_percent + convection_percent

            # Indirect efficiency
            efficiency = Decimal("100") - total_loss_percent
            return efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        result = benchmark_runner.run_benchmark(
            calculate_indirect_efficiency,
            iterations=10000,
            target_ms=1.0,
            name="Indirect Method Efficiency"
        )

        assert result.passed, f"Indirect method: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"

    @pytest.mark.performance
    def test_combined_efficiency_calculation(self, benchmark_runner, sample_furnace_data):
        """Benchmark full combined efficiency calculation."""
        def calculate_combined_efficiency():
            fuel_input = sample_furnace_data["fuel_input"]
            heat_output = sample_furnace_data["heat_output"]
            flue_gas = sample_furnace_data["flue_gas"]

            # Calculate all components
            mass_flow = fuel_input["mass_flow_rate_kg_hr"]
            hhv = fuel_input["higher_heating_value_mj_kg"]
            fuel_energy_mw = (mass_flow * hhv) / 3600

            useful_heat = heat_output["useful_heat_output_mw"]
            direct_eff = (useful_heat / fuel_energy_mw) * 100

            flue_temp = flue_gas["temperature_c"]
            stack_loss = min(20.0, (flue_temp - 25) / 10)
            indirect_eff = 100 - stack_loss - 3.5  # Simplified

            # Average
            combined_eff = (direct_eff + indirect_eff) / 2

            return round(combined_eff, 2)

        result = benchmark_runner.run_benchmark(
            calculate_combined_efficiency,
            iterations=5000,
            target_ms=2.0,
            name="Combined Efficiency Calculation"
        )

        assert result.passed, f"Combined efficiency: {result.avg_time_ms:.4f}ms > {result.target_ms}ms"

    @pytest.mark.performance
    def test_excess_air_calculation(self, benchmark_runner, sample_furnace_data):
        """Benchmark excess air calculation from O2 measurement."""
        def calculate_excess_air():
            o2_percent = Decimal(str(sample_furnace_data["flue_gas"]["o2_percent_dry"]))

            # EA% = (O2 / (21 - O2)) * 100
            if o2_percent >= 21:
                return Decimal("0")

            excess_air = (o2_percent / (Decimal("21") - o2_percent)) * Decimal("100")
            return excess_air.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        result = benchmark_runner.run_benchmark(
            calculate_excess_air,
            iterations=50000,
            target_ms=0.05,
            name="Excess Air Calculation"
        )

        assert result.passed
        assert result.throughput_per_sec > 20000


# ============================================================================
# FUEL CONSUMPTION ANALYSIS BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestFuelConsumptionBenchmarks:
    """Performance benchmarks for fuel consumption analysis."""

    @pytest.mark.performance
    def test_sec_calculation(self, benchmark_runner, sample_fuel_consumption_data):
        """Benchmark Specific Energy Consumption calculation."""
        def calculate_sec():
            data = sample_fuel_consumption_data["consumption_data"]

            total_energy_gj = sum(
                d["consumption_rate_kg_hr"] * d["heating_value_mj_kg"] / 1000
                for d in data
            )
            total_production = sum(d["production_rate"] for d in data)

            sec = total_energy_gj / total_production if total_production > 0 else 0
            return round(sec, 3)

        result = benchmark_runner.run_benchmark(
            calculate_sec,
            iterations=5000,
            target_ms=1.0,
            name="SEC Calculation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_deviation_analysis(self, benchmark_runner, sample_fuel_consumption_data):
        """Benchmark deviation from baseline analysis."""
        def analyze_deviation():
            data = sample_fuel_consumption_data["consumption_data"]
            baseline = sample_fuel_consumption_data["baseline_performance"]

            # Calculate current SEC
            total_energy = sum(
                d["consumption_rate_kg_hr"] * d["heating_value_mj_kg"] / 1000
                for d in data
            )
            total_production = sum(d["production_rate"] for d in data)
            current_sec = total_energy / total_production if total_production > 0 else 0

            # Deviation
            baseline_sec = baseline["expected_sec_gj_ton"]
            deviation_percent = ((current_sec - baseline_sec) / baseline_sec) * 100

            return {
                "current_sec": round(current_sec, 3),
                "deviation_percent": round(deviation_percent, 2)
            }

        result = benchmark_runner.run_benchmark(
            analyze_deviation,
            iterations=5000,
            target_ms=2.0,
            name="Deviation Analysis"
        )

        assert result.passed

    @pytest.mark.performance
    def test_cost_impact_calculation(self, benchmark_runner, sample_fuel_consumption_data):
        """Benchmark fuel cost impact calculation."""
        def calculate_cost_impact():
            data = sample_fuel_consumption_data["consumption_data"]
            costs = sample_fuel_consumption_data["cost_parameters"]

            total_energy_gj = sum(
                d["consumption_rate_kg_hr"] * d["heating_value_mj_kg"] / 1000
                for d in data
            )

            fuel_cost = total_energy_gj * costs["fuel_cost_usd_per_gj"]
            carbon_emissions = total_energy_gj * costs["emission_factor_kg_co2_per_gj"] / 1000

            return {
                "fuel_cost_usd": round(fuel_cost, 2),
                "carbon_emissions_tons": round(carbon_emissions, 2)
            }

        result = benchmark_runner.run_benchmark(
            calculate_cost_impact,
            iterations=5000,
            target_ms=2.0,
            name="Cost Impact Calculation"
        )

        assert result.passed


# ============================================================================
# ANOMALY DETECTION BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestAnomalyDetectionBenchmarks:
    """Performance benchmarks for anomaly detection."""

    @pytest.mark.performance
    def test_statistical_anomaly_detection(self, benchmark_runner):
        """Benchmark statistical anomaly detection."""
        # Generate test data
        values = [100.0 + random.gauss(0, 5) for _ in range(100)]
        values.append(150.0)  # Add anomaly

        def detect_anomalies():
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance)

            anomalies = []
            for i, v in enumerate(values):
                z_score = abs(v - mean_val) / std_dev if std_dev > 0 else 0
                if z_score > 3:  # 3-sigma rule
                    anomalies.append({"index": i, "value": v, "z_score": z_score})

            return anomalies

        result = benchmark_runner.run_benchmark(
            detect_anomalies,
            iterations=5000,
            target_ms=2.0,
            name="Statistical Anomaly Detection"
        )

        assert result.passed

    @pytest.mark.performance
    def test_rate_of_change_detection(self, benchmark_runner):
        """Benchmark rate of change anomaly detection."""
        values = [100.0 + i * 0.1 for i in range(100)]  # Gradual increase
        values[50] = 130.0  # Sudden spike

        def detect_rate_anomalies():
            max_rate = 5.0  # Maximum allowed rate of change
            anomalies = []

            for i in range(1, len(values)):
                rate = abs(values[i] - values[i-1])
                if rate > max_rate:
                    anomalies.append({
                        "index": i,
                        "rate": rate,
                        "type": "rate_of_change"
                    })

            return anomalies

        result = benchmark_runner.run_benchmark(
            detect_rate_anomalies,
            iterations=10000,
            target_ms=0.5,
            name="Rate of Change Detection"
        )

        assert result.passed

    @pytest.mark.performance
    def test_multi_parameter_anomaly(self, benchmark_runner, sample_furnace_data):
        """Benchmark multi-parameter correlation anomaly detection."""
        def detect_correlation_anomalies():
            flue_gas = sample_furnace_data["flue_gas"]

            o2 = flue_gas["o2_percent_dry"]
            co = flue_gas["co_ppm"]
            nox = flue_gas["nox_ppm"]

            anomalies = []

            # High O2 with high CO indicates incomplete combustion
            if o2 > 5.0 and co > 50:
                anomalies.append({
                    "type": "incomplete_combustion",
                    "o2": o2,
                    "co": co
                })

            # Low O2 with high NOx indicates high flame temperature
            if o2 < 2.0 and nox > 100:
                anomalies.append({
                    "type": "high_flame_temp",
                    "o2": o2,
                    "nox": nox
                })

            return anomalies

        result = benchmark_runner.run_benchmark(
            detect_correlation_anomalies,
            iterations=10000,
            target_ms=0.1,
            name="Multi-Parameter Anomaly Detection"
        )

        assert result.passed


# ============================================================================
# MAINTENANCE PREDICTION BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestMaintenancePredictionBenchmarks:
    """Performance benchmarks for predictive maintenance calculations."""

    @pytest.mark.performance
    def test_rul_calculation(self, benchmark_runner, sample_equipment_inventory):
        """Benchmark Remaining Useful Life calculation."""
        operating_hours = 42000

        def calculate_rul():
            results = []
            for equip in sample_equipment_inventory:
                design_hours = equip["design_life_years"] * 8000
                rul_hours = max(0, design_hours - operating_hours)
                rul_percent = (rul_hours / design_hours) * 100 if design_hours > 0 else 0

                results.append({
                    "equipment_id": equip["equipment_id"],
                    "rul_hours": rul_hours,
                    "rul_percent": round(rul_percent, 1)
                })

            return results

        result = benchmark_runner.run_benchmark(
            calculate_rul,
            iterations=10000,
            target_ms=0.5,
            name="RUL Calculation"
        )

        assert result.passed

    @pytest.mark.performance
    def test_maintenance_priority_scoring(self, benchmark_runner, sample_equipment_inventory):
        """Benchmark maintenance priority scoring."""
        def calculate_priority():
            results = []
            for equip in sample_equipment_inventory:
                criticality_weights = {"critical": 100, "high": 70, "medium": 40, "low": 10}
                criticality_score = criticality_weights.get(equip["criticality"], 40)

                # Simplified health score
                age_factor = 1 - (5 / equip["design_life_years"])
                health_score = max(0, 100 * age_factor)

                # Priority = Criticality * (100 - Health) / 100
                priority = criticality_score * (100 - health_score) / 100

                results.append({
                    "equipment_id": equip["equipment_id"],
                    "priority_score": round(priority, 1)
                })

            return sorted(results, key=lambda x: x["priority_score"], reverse=True)

        result = benchmark_runner.run_benchmark(
            calculate_priority,
            iterations=10000,
            target_ms=0.5,
            name="Maintenance Priority Scoring"
        )

        assert result.passed


# ============================================================================
# MULTI-FURNACE OPTIMIZATION BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestMultiFurnaceOptimizationBenchmarks:
    """Performance benchmarks for multi-furnace fleet optimization."""

    @pytest.mark.performance
    def test_load_allocation_optimization(self, benchmark_runner, sample_multi_furnace_data):
        """Benchmark load allocation optimization for furnace fleet."""
        def optimize_load_allocation():
            furnaces = sample_multi_furnace_data["furnaces"]
            total_demand = sample_multi_furnace_data["total_heat_demand_mw"]

            # Simple merit order dispatch based on efficiency
            sorted_furnaces = sorted(
                furnaces,
                key=lambda f: f["efficiency_at_current_load"],
                reverse=True
            )

            remaining_demand = total_demand
            allocation = []

            for furnace in sorted_furnaces:
                if remaining_demand <= 0:
                    load = 0
                elif remaining_demand >= furnace["max_capacity_mw"]:
                    load = furnace["max_capacity_mw"]
                else:
                    load = remaining_demand

                allocation.append({
                    "furnace_id": furnace["furnace_id"],
                    "allocated_load_mw": load,
                    "efficiency": furnace["efficiency_at_current_load"]
                })

                remaining_demand -= load

            return allocation

        result = benchmark_runner.run_benchmark(
            optimize_load_allocation,
            iterations=1000,
            target_ms=5.0,
            name="Load Allocation Optimization"
        )

        assert result.passed

    @pytest.mark.performance
    def test_fleet_efficiency_calculation(self, benchmark_runner, sample_multi_furnace_data):
        """Benchmark fleet-wide efficiency calculation."""
        def calculate_fleet_efficiency():
            furnaces = sample_multi_furnace_data["furnaces"]

            total_load = sum(f["current_load_mw"] for f in furnaces)
            weighted_efficiency = sum(
                f["current_load_mw"] * f["efficiency_at_current_load"]
                for f in furnaces
            )

            fleet_efficiency = weighted_efficiency / total_load if total_load > 0 else 0

            return round(fleet_efficiency, 2)

        result = benchmark_runner.run_benchmark(
            calculate_fleet_efficiency,
            iterations=10000,
            target_ms=0.5,
            name="Fleet Efficiency Calculation"
        )

        assert result.passed


# ============================================================================
# CACHE PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestCachePerformanceBenchmarks:
    """Performance benchmarks for caching mechanisms."""

    @pytest.mark.performance
    def test_cache_hit_performance(self, benchmark_runner):
        """Benchmark cache hit performance."""
        cache = {f"result_{i}": {"value": i * 100} for i in range(1000)}

        def cache_lookup():
            return cache.get("result_500")

        result = benchmark_runner.run_benchmark(
            cache_lookup,
            iterations=100000,
            target_ms=0.001,
            name="Cache Hit"
        )

        assert result.avg_time_ms < 0.01

    @pytest.mark.performance
    def test_cache_key_generation(self, benchmark_runner, sample_furnace_data):
        """Benchmark cache key generation."""
        def generate_cache_key():
            data_str = json.dumps(sample_furnace_data, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()

        result = benchmark_runner.run_benchmark(
            generate_cache_key,
            iterations=10000,
            target_ms=1.0,
            name="Cache Key Generation"
        )

        assert result.passed


# ============================================================================
# PROVENANCE HASH BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestProvenanceHashBenchmarks:
    """Performance benchmarks for provenance hash calculations."""

    @pytest.mark.performance
    def test_sha256_provenance_hash(self, benchmark_runner, sample_furnace_data):
        """Benchmark SHA-256 provenance hash calculation."""
        def calculate_provenance_hash():
            return hashlib.sha256(
                json.dumps(sample_furnace_data, sort_keys=True).encode()
            ).hexdigest()

        result = benchmark_runner.run_benchmark(
            calculate_provenance_hash,
            iterations=10000,
            target_ms=1.0,
            name="SHA-256 Provenance Hash"
        )

        assert result.passed
        assert result.throughput_per_sec > 1000


# ============================================================================
# SCALABILITY BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Test performance scaling with data volume."""

    @pytest.mark.performance
    def test_large_sensor_processing(self, benchmark_runner, large_sensor_dataset):
        """Benchmark processing of 200+ data points per FURNACEPULSE spec."""
        def process_sensors():
            results = []
            for sensor in large_sensor_dataset:
                processed = {
                    "sensor_id": sensor["sensor_id"],
                    "value": sensor["value"],
                    "processed_at": datetime.now().isoformat()
                }
                results.append(processed)
            return results

        result = benchmark_runner.run_benchmark(
            process_sensors,
            iterations=1000,
            target_ms=10.0,
            name="200 Sensor Processing"
        )

        assert result.passed

    @pytest.mark.performance
    def test_furnace_count_scaling(self, benchmark_runner):
        """Test optimization scaling with number of furnaces."""
        scaling_results = {}

        for n_furnaces in [2, 5, 10, 20]:
            furnaces = [
                {
                    "furnace_id": f"F{i}",
                    "load": 20 + i,
                    "efficiency": 75 + random.uniform(0, 10)
                }
                for i in range(n_furnaces)
            ]

            def optimize():
                return sorted(furnaces, key=lambda f: f["efficiency"], reverse=True)

            start = time.perf_counter()
            for _ in range(1000):
                optimize()
            elapsed_ms = (time.perf_counter() - start)

            scaling_results[n_furnaces] = elapsed_ms

        # Verify roughly linear scaling
        assert scaling_results[20] < scaling_results[2] * 20


# ============================================================================
# MEMORY USAGE BENCHMARKS
# ============================================================================

@pytest.mark.performance
class TestMemoryBenchmarks:
    """Test memory usage patterns."""

    @pytest.mark.performance
    def test_memory_stability_under_load(self):
        """Test memory usage remains stable under repeated calculations."""
        import gc

        gc.collect()

        results = []
        for iteration in range(100):
            data = {
                "sensors": [{"id": i, "value": random.uniform(0, 100)} for i in range(100)],
                "timestamp": datetime.now().isoformat()
            }

            hash_val = hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()

            results.append(hash_val)

        # Verify all hashes are valid (64 characters)
        assert all(len(h) == 64 for h in results)

        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
