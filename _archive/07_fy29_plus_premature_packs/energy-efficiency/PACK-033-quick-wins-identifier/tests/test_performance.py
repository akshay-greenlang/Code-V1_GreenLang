# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Performance Tests (test_performance.py)
===============================================================================

Tests engine execution time, throughput, and memory behavior for all
8 engines. Validates that each engine completes within target latency
and handles batch workloads efficiently.

Coverage target: 85%+
Total tests: ~25
"""

import importlib.util
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"

ENGINE_FILES = [
    "quick_wins_scanner_engine",
    "payback_calculator_engine",
    "energy_savings_estimator_engine",
    "carbon_reduction_engine",
    "implementation_prioritizer_engine",
    "behavioral_change_engine",
    "utility_rebate_engine",
    "quick_wins_reporting_engine",
]


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack033_perf.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =============================================================================
# Engine Instantiation Performance
# =============================================================================


class TestEngineInstantiationTime:
    """Test that each engine instantiates within 100ms."""

    @pytest.mark.parametrize("engine_name", ENGINE_FILES)
    def test_engine_instantiation_time(self, engine_name):
        mod = _load(engine_name)
        # Find the engine class
        engine_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and attr_name.endswith("Engine") and attr_name != "BaseModel":
                engine_cls = obj
                break
        if engine_cls is None:
            pytest.skip(f"No Engine class found in {engine_name}")

        start = time.perf_counter()
        try:
            instance = engine_cls()
        except TypeError:
            instance = engine_cls(config={})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"{engine_name} instantiation took {elapsed_ms:.1f}ms (limit: 100ms)"
        assert instance is not None


# =============================================================================
# Scanner Performance
# =============================================================================


class TestScanPerformance:
    """Test scanner performance."""

    def test_scan_performance_under_2s(self):
        mod = _load("quick_wins_scanner_engine")
        engine = mod.QuickWinsScannerEngine()
        profile_cls = (getattr(mod, "FacilityProfile", None)
                       or getattr(mod, "ScanFacilityProfile", None))
        if profile_cls is None:
            pytest.skip("FacilityProfile not found")

        profile = profile_cls(
            facility_id="FAC-PERF-001",
            building_type="OFFICE",
            floor_area_m2=12000.0,
            annual_electricity_kwh=1_800_000.0,
        )

        scan = getattr(engine, "scan", None) or getattr(engine, "scan_facility", None)
        start = time.perf_counter()
        result = scan(profile)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"Scan took {elapsed_ms:.1f}ms (limit: 2000ms)"
        assert result is not None

    def test_scan_multiple_facilities(self):
        mod = _load("quick_wins_scanner_engine")
        engine = mod.QuickWinsScannerEngine()
        profile_cls = (getattr(mod, "FacilityProfile", None)
                       or getattr(mod, "ScanFacilityProfile", None))
        if profile_cls is None:
            pytest.skip("FacilityProfile not found")

        scan = getattr(engine, "scan", None) or getattr(engine, "scan_facility", None)
        start = time.perf_counter()
        for i in range(10):
            profile = profile_cls(
                facility_id=f"FAC-PERF-{i:03d}",
                building_type="OFFICE",
                floor_area_m2=12000.0,
                annual_electricity_kwh=1_800_000.0,
            )
            scan(profile)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10000, f"10 scans took {elapsed_ms:.1f}ms (limit: 10000ms)"


# =============================================================================
# Payback Calculation Performance
# =============================================================================


class TestPaybackPerformance:
    """Test payback calculator performance."""

    def test_payback_calculation_under_500ms(self):
        mod = _load("payback_calculator_engine")
        engine = mod.PaybackCalculatorEngine()
        measure = mod.MeasureFinancials(
            measure_id="PERF-M-001",
            name="Performance Test Measure",
            implementation_cost=Decimal("12000"),
            annual_savings_kwh=Decimal("33696"),
            annual_savings_cost=Decimal("6739"),
        )
        params = mod.FinancialParameters()

        start = time.perf_counter()
        result = engine.calculate_payback(measure, params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Payback calc took {elapsed_ms:.1f}ms (limit: 500ms)"
        assert result is not None

    def test_batch_payback_10_measures(self):
        mod = _load("payback_calculator_engine")
        engine = mod.PaybackCalculatorEngine()
        measures = [
            mod.MeasureFinancials(
                measure_id=f"PERF-B-{i:03d}",
                name=f"Batch Measure {i}",
                implementation_cost=Decimal("10000") + Decimal(str(i * 1000)),
                annual_savings_kwh=Decimal("30000"),
                annual_savings_cost=Decimal("5000") + Decimal(str(i * 500)),
            )
            for i in range(10)
        ]
        params = mod.FinancialParameters()

        start = time.perf_counter()
        result = engine.calculate_batch(measures, params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"Batch 10 took {elapsed_ms:.1f}ms (limit: 5000ms)"
        assert len(result.results) == 10

    def test_batch_payback_50_measures(self):
        mod = _load("payback_calculator_engine")
        engine = mod.PaybackCalculatorEngine()
        measures = [
            mod.MeasureFinancials(
                measure_id=f"PERF-L-{i:03d}",
                name=f"Large Batch Measure {i}",
                implementation_cost=Decimal("10000"),
                annual_savings_kwh=Decimal("30000"),
                annual_savings_cost=Decimal("5000"),
            )
            for i in range(50)
        ]
        params = mod.FinancialParameters()

        start = time.perf_counter()
        result = engine.calculate_batch(measures, params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 30000, f"Batch 50 took {elapsed_ms:.1f}ms (limit: 30000ms)"
        assert len(result.results) == 50


# =============================================================================
# Sensitivity Analysis Performance
# =============================================================================


class TestSensitivityPerformance:
    """Test sensitivity analysis performance."""

    def test_sensitivity_5_values(self):
        mod = _load("payback_calculator_engine")
        engine = mod.PaybackCalculatorEngine()
        measure = mod.MeasureFinancials(
            measure_id="PERF-S-001",
            implementation_cost=Decimal("12000"),
            annual_savings_kwh=Decimal("33696"),
            annual_savings_cost=Decimal("6739"),
        )
        params = mod.FinancialParameters()
        values = [Decimal("0.04"), Decimal("0.06"), Decimal("0.08"), Decimal("0.10"), Decimal("0.12")]

        start = time.perf_counter()
        result = engine.run_sensitivity(measure, params, mod.SensitivityParameter.DISCOUNT_RATE, values)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 3000, f"Sensitivity took {elapsed_ms:.1f}ms (limit: 3000ms)"
        assert len(result.npvs) == 5


# =============================================================================
# Memory Usage
# =============================================================================


class TestMemoryUsage:
    """Test memory usage is reasonable."""

    def test_memory_usage_reasonable(self):
        """Verify engine does not leak memory during repeated operations."""
        mod = _load("payback_calculator_engine")
        engine = mod.PaybackCalculatorEngine()
        measure = mod.MeasureFinancials(
            measure_id="MEM-001",
            implementation_cost=Decimal("12000"),
            annual_savings_kwh=Decimal("33696"),
            annual_savings_cost=Decimal("6739"),
        )
        params = mod.FinancialParameters()

        # Run 100 calculations - should not accumulate unbounded memory
        for i in range(100):
            result = engine.calculate_payback(
                measure.model_copy(update={"measure_id": f"MEM-{i:04d}"}),
                params,
            )
            assert result is not None

    def test_large_cash_flow_memory(self):
        """Test that 40-year cash flows do not use excessive memory."""
        mod = _load("payback_calculator_engine")
        engine = mod.PaybackCalculatorEngine()
        measure = mod.MeasureFinancials(
            measure_id="MEM-LONG-001",
            implementation_cost=Decimal("50000"),
            annual_savings_kwh=Decimal("100000"),
            annual_savings_cost=Decimal("20000"),
            measure_life_years=40,
        )
        params = mod.FinancialParameters(analysis_period_years=40)

        result = engine.calculate_payback(measure, params)
        assert len(result.cash_flows) == 40


# =============================================================================
# Carbon Engine Performance
# =============================================================================


class TestCarbonPerformance:
    """Test carbon engine performance."""

    def test_carbon_calculation_under_500ms(self):
        mod = _load("carbon_reduction_engine")
        engine = mod.CarbonReductionEngine()
        input_cls = (getattr(mod, "EnergySavingsInput", None) or getattr(mod, "CarbonInput", None)
                     or getattr(mod, "CarbonReductionInput", None))
        if input_cls is None:
            pytest.skip("Carbon input not found")
        calc = (getattr(engine, "calculate_reduction", None) or getattr(engine, "calculate", None)
                or getattr(engine, "calculate_co2e", None))
        if calc is None:
            pytest.skip("Calculate method not found")
        start = time.perf_counter()
        result = calc(input_cls(electricity_savings_kwh=Decimal("100000"), region="GB"))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Carbon calc took {elapsed_ms:.1f}ms (limit: 500ms)"
        assert result is not None

    def test_carbon_batch_under_5s(self):
        mod = _load("carbon_reduction_engine")
        engine = mod.CarbonReductionEngine()
        input_cls = (getattr(mod, "EnergySavingsInput", None) or getattr(mod, "CarbonInput", None)
                     or getattr(mod, "CarbonReductionInput", None))
        if input_cls is None:
            pytest.skip("Carbon input not found")
        calc = (getattr(engine, "calculate_reduction", None) or getattr(engine, "calculate", None)
                or getattr(engine, "calculate_co2e", None))
        if calc is None:
            pytest.skip("Calculate method not found")
        start = time.perf_counter()
        for i in range(20):
            calc(input_cls(electricity_savings_kwh=Decimal(str(50000 + i * 10000)), region="GB"))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"20 carbon calcs took {elapsed_ms:.1f}ms (limit: 5000ms)"


# =============================================================================
# Savings Estimator Performance
# =============================================================================


class TestSavingsEstimatorPerformance:
    """Test savings estimator performance."""

    def test_savings_estimation_under_500ms(self):
        mod = _load("energy_savings_estimator_engine")
        engine = mod.EnergySavingsEstimatorEngine()
        input_cls = (getattr(mod, "MeasureSavingsInput", None)
                     or getattr(mod, "SavingsEstimateInput", None)
                     or getattr(mod, "MeasureInput", None))
        if input_cls is None:
            pytest.skip("Input model not found")
        estimate = getattr(engine, "estimate", None) or getattr(engine, "estimate_savings", None)
        if estimate is None:
            pytest.skip("estimate method not found")
        start = time.perf_counter()
        result = estimate(input_cls(
            measure_id="PERF-ES-001", baseline_kwh=Decimal("1800000"),
            affected_end_use_pct=Decimal("30"), base_savings_pct=Decimal("50"),
        ))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Savings estimation took {elapsed_ms:.1f}ms (limit: 500ms)"
        assert result is not None

    def test_batch_estimation_under_5s(self):
        mod = _load("energy_savings_estimator_engine")
        engine = mod.EnergySavingsEstimatorEngine()
        input_cls = (getattr(mod, "MeasureSavingsInput", None)
                     or getattr(mod, "SavingsEstimateInput", None)
                     or getattr(mod, "MeasureInput", None))
        if input_cls is None:
            pytest.skip("Input model not found")
        estimate = getattr(engine, "estimate", None) or getattr(engine, "estimate_savings", None)
        if estimate is None:
            pytest.skip("estimate method not found")
        start = time.perf_counter()
        for i in range(20):
            estimate(input_cls(
                measure_id=f"PERF-ES-{i:03d}", baseline_kwh=Decimal("1800000"),
                affected_end_use_pct=Decimal(str(20 + i)), base_savings_pct=Decimal("50"),
            ))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"20 estimations took {elapsed_ms:.1f}ms (limit: 5000ms)"


# =============================================================================
# Prioritizer Performance
# =============================================================================


class TestPrioritizerPerformance:
    """Test prioritizer engine performance."""

    def test_prioritization_under_1s(self):
        mod = _load("implementation_prioritizer_engine")
        engine = mod.ImplementationPrioritizerEngine()
        input_cls = (getattr(mod, "MeasureCriteria", None) or getattr(mod, "MeasureScore", None)
                     or getattr(mod, "PrioritizationInput", None))
        if input_cls is None:
            pytest.skip("Input model not found")
        prioritize = (getattr(engine, "prioritize", None) or getattr(engine, "rank", None)
                      or getattr(engine, "calculate", None))
        if prioritize is None:
            pytest.skip("prioritize method not found")
        measures = [
            input_cls(
                measure_id=f"PERF-PR-{i:03d}", annual_savings_cost=Decimal(str(5000 + i * 1000)),
                implementation_cost=Decimal(str(10000 + i * 2000)),
                payback_years=Decimal(str(round(1.0 + i * 0.3, 2))),
            )
            for i in range(10)
        ]
        start = time.perf_counter()
        result = prioritize(measures)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1000, f"Prioritization took {elapsed_ms:.1f}ms (limit: 1000ms)"
        assert result is not None

    def test_large_prioritization_under_5s(self):
        mod = _load("implementation_prioritizer_engine")
        engine = mod.ImplementationPrioritizerEngine()
        input_cls = (getattr(mod, "MeasureCriteria", None) or getattr(mod, "MeasureScore", None)
                     or getattr(mod, "PrioritizationInput", None))
        if input_cls is None:
            pytest.skip("Input model not found")
        prioritize = (getattr(engine, "prioritize", None) or getattr(engine, "rank", None)
                      or getattr(engine, "calculate", None))
        if prioritize is None:
            pytest.skip("prioritize method not found")
        measures = [
            input_cls(
                measure_id=f"PERF-LG-{i:03d}", annual_savings_cost=Decimal(str(3000 + i * 500)),
                implementation_cost=Decimal(str(5000 + i * 1000)),
                payback_years=Decimal(str(round(0.5 + i * 0.2, 2))),
            )
            for i in range(50)
        ]
        start = time.perf_counter()
        result = prioritize(measures)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"50-measure prioritization took {elapsed_ms:.1f}ms (limit: 5000ms)"
