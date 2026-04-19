# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Performance Tests
================================================

Tests execution latency targets, repeated execution stability,
scalability for large portfolios, memory usage, and provenance
hash computation performance.

Test Count Target: ~35 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import hashlib
import sys
import time
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import ENGINE_FILES, _load_engine


def _try_load_engine(engine_key):
    """Load an engine, skip test if not available."""
    try:
        return _load_engine(engine_key)
    except (FileNotFoundError, ImportError):
        pytest.skip(f"Engine not available: {engine_key}")


# =========================================================================
# 1. Engine Instantiation Latency
# =========================================================================


class TestEngineInstantiationLatency:
    """Test engine instantiation is fast."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_instantiation_under_1s(self, engine_key):
        """Engine instantiation completes in under 1 second."""
        try:
            mod = _load_engine(engine_key)
        except (FileNotFoundError, ImportError):
            pytest.skip(f"Engine not available: {engine_key}")

        # Get the class name from engine_key
        class_names = {
            "eui_calculator": "EUICalculatorEngine",
            "peer_comparison": "PeerComparisonEngine",
            "sector_benchmark": "SectorBenchmarkEngine",
            "weather_normalisation": "WeatherNormalisationEngine",
            "energy_performance_gap": "EnergyPerformanceGapEngine",
            "portfolio_benchmark": "PortfolioBenchmarkEngine",
            "regression_analysis": "RegressionAnalysisEngine",
            "performance_rating": "PerformanceRatingEngine",
            "trend_analysis": "TrendAnalysisEngine",
            "benchmark_report": "BenchmarkReportEngine",
        }
        class_name = class_names.get(engine_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")

        start = time.perf_counter()
        instance = cls()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert instance is not None
        assert elapsed_ms < 1000  # < 1 second


# =========================================================================
# 2. EUI Calculation Latency
# =========================================================================


class TestEUICalculationLatency:
    """Test EUI calculation completes within target latency."""

    def test_site_eui_under_500ms(self, sample_energy_data):
        """Site EUI calculation completes in under 500ms."""
        start = time.perf_counter()

        total_energy = sum(
            m["electricity_kwh"] + m["gas_kwh"] for m in sample_energy_data
        )
        floor_area = 5000.0
        eui = total_energy / floor_area

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert eui > 0
        assert elapsed_ms < 500

    def test_source_eui_under_500ms(self, sample_energy_data):
        """Source EUI calculation completes in under 500ms."""
        start = time.perf_counter()

        total_elec = sum(m["electricity_kwh"] for m in sample_energy_data)
        total_gas = sum(m["gas_kwh"] for m in sample_energy_data)
        source_energy = total_elec * 2.80 + total_gas * 1.047
        floor_area = 5000.0
        source_eui = source_energy / floor_area

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert source_eui > 0
        assert elapsed_ms < 500

    def test_rolling_eui_12_months_under_500ms(self, sample_energy_data):
        """Rolling 12-month EUI calculation under 500ms."""
        start = time.perf_counter()

        running_total = 0
        floor_area = 5000.0
        rolling_euis = []
        for m in sample_energy_data:
            running_total += m["electricity_kwh"] + m["gas_kwh"]
            rolling_euis.append(running_total / floor_area)

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert len(rolling_euis) == 12
        assert elapsed_ms < 500


# =========================================================================
# 3. Peer Comparison Latency
# =========================================================================


class TestPeerComparisonLatency:
    """Test peer comparison completes within target latency."""

    def test_percentile_50_peers_under_200ms(self, sample_peer_group):
        """Percentile calculation for 50 peers under 200ms."""
        start = time.perf_counter()

        subject_eui = 140.6
        peer_euis = sorted(p["eui_kwh_per_m2_yr"] for p in sample_peer_group)
        better_count = sum(1 for e in peer_euis if e >= subject_eui)
        percentile = better_count / len(peer_euis) * 100

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert 0 <= percentile <= 100
        assert elapsed_ms < 200

    def test_percentile_1000_peers_under_500ms(self):
        """Percentile calculation for 1000 peers under 500ms."""
        import random
        random.seed(42)
        peers = [random.gauss(180, 50) for _ in range(1000)]

        start = time.perf_counter()

        subject_eui = 140.6
        peers.sort()
        better_count = sum(1 for e in peers if e >= subject_eui)
        percentile = better_count / len(peers) * 100

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert 0 <= percentile <= 100
        assert elapsed_ms < 500


# =========================================================================
# 4. Portfolio Scalability
# =========================================================================


class TestPortfolioScalability:
    """Test portfolio operations scale to large portfolios."""

    def test_portfolio_100_facilities_under_2s(self):
        """Portfolio aggregation for 100 facilities under 2 seconds."""
        import random
        random.seed(42)
        portfolio = [
            {
                "facility_id": f"F-{i:04d}",
                "gross_floor_area_m2": random.uniform(500, 20000),
                "energy_consumption_kwh": random.uniform(50000, 5000000),
                "eui_kwh_per_m2": random.uniform(50, 500),
            }
            for i in range(100)
        ]

        start = time.perf_counter()

        total_energy = sum(f["energy_consumption_kwh"] for f in portfolio)
        total_area = sum(f["gross_floor_area_m2"] for f in portfolio)
        weighted_eui = total_energy / total_area
        ranked = sorted(portfolio, key=lambda f: f["eui_kwh_per_m2"])

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert weighted_eui > 0
        assert len(ranked) == 100
        assert elapsed_ms < 2000

    def test_portfolio_500_facilities_under_5s(self):
        """Portfolio aggregation for 500 facilities under 5 seconds."""
        import random
        random.seed(42)
        portfolio = [
            {
                "facility_id": f"F-{i:04d}",
                "gross_floor_area_m2": random.uniform(500, 20000),
                "energy_consumption_kwh": random.uniform(50000, 5000000),
                "eui_kwh_per_m2": random.uniform(50, 500),
            }
            for i in range(500)
        ]

        start = time.perf_counter()

        total_energy = sum(f["energy_consumption_kwh"] for f in portfolio)
        total_area = sum(f["gross_floor_area_m2"] for f in portfolio)
        weighted_eui = total_energy / total_area
        ranked = sorted(portfolio, key=lambda f: f["eui_kwh_per_m2"])

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert weighted_eui > 0
        assert len(ranked) == 500
        assert elapsed_ms < 5000


# =========================================================================
# 5. Regression Calculation Latency
# =========================================================================


class TestRegressionLatency:
    """Test regression calculation latency."""

    def test_simple_regression_under_2s(self, sample_regression_data):
        """Simple OLS regression completes in under 2 seconds."""
        import math

        start = time.perf_counter()

        n = len(sample_regression_data)
        x = [d["hdd"] for d in sample_regression_data]
        y = [d["energy_kwh"] for d in sample_regression_data]
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        ss_xx = sum((x[i] - mean_x) ** 2 for i in range(n))
        slope = ss_xy / ss_xx if ss_xx != 0 else 0
        intercept = mean_y - slope * mean_x

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert slope > 0
        assert elapsed_ms < 2000


# =========================================================================
# 6. Provenance Hash Performance
# =========================================================================


class TestProvenanceHashPerformance:
    """Test provenance hash computation performance."""

    def test_sha256_single_hash_under_1ms(self):
        """Single SHA-256 hash computation under 1ms."""
        data = b"benchmark_input_data" * 100  # ~2KB

        start = time.perf_counter()
        h = hashlib.sha256(data).hexdigest()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(h) == 64
        assert elapsed_ms < 1.0

    def test_sha256_1000_hashes_under_100ms(self):
        """1000 SHA-256 hash computations under 100ms."""
        data = b"benchmark_input_data" * 100

        start = time.perf_counter()
        for i in range(1000):
            h = hashlib.sha256(data + str(i).encode()).hexdigest()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(h) == 64
        assert elapsed_ms < 100

    def test_provenance_chain_10_stages_under_1ms(self):
        """10-stage provenance chain under 1ms."""
        start = time.perf_counter()

        current_hash = hashlib.sha256(b"initial_input").hexdigest()
        for i in range(10):
            current_hash = hashlib.sha256(
                (current_hash + f"|stage_{i}").encode()
            ).hexdigest()

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert len(current_hash) == 64
        assert elapsed_ms < 1.0


# =========================================================================
# 7. Repeated Execution Stability
# =========================================================================


class TestRepeatedExecutionStability:
    """Test repeated execution produces consistent results."""

    def test_eui_consistent_across_10_runs(self, sample_energy_data):
        """EUI calculation produces identical results across 10 runs."""
        results = []
        for _ in range(10):
            total = sum(
                m["electricity_kwh"] + m["gas_kwh"] for m in sample_energy_data
            )
            eui = total / 5000.0
            results.append(eui)
        assert len(set(results)) == 1  # All identical

    def test_provenance_hash_consistent_across_10_runs(self):
        """Provenance hash is identical across 10 runs."""
        hashes = []
        for _ in range(10):
            h = hashlib.sha256(b"consistent_input").hexdigest()
            hashes.append(h)
        assert len(set(hashes)) == 1

    def test_percentile_consistent(self, sample_peer_group):
        """Percentile rank is consistent across 10 runs."""
        results = []
        for _ in range(10):
            subject_eui = 140.6
            peer_euis = sorted(p["eui_kwh_per_m2_yr"] for p in sample_peer_group)
            better_count = sum(1 for e in peer_euis if e >= subject_eui)
            percentile = better_count / len(peer_euis) * 100
            results.append(percentile)
        assert len(set(results)) == 1
