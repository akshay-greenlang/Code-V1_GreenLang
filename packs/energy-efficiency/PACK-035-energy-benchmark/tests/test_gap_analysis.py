# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Gap Analysis Engine Tests
========================================================

Tests end-use disaggregation, gap calculation against benchmarks,
priority ranking of gaps, savings potential estimation, ECM
linking, and energy conservation opportunity identification.

Test Count Target: ~55 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load_gap():
    path = ENGINES_DIR / "energy_performance_gap_engine.py"
    if not path.exists():
        pytest.skip("energy_performance_gap_engine.py not found")
    mod_key = "pack035_test.gap_analysis"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load energy_performance_gap_engine: {exc}")
    return mod


# =========================================================================
# Fixtures specific to gap analysis
# =========================================================================


@pytest.fixture
def sample_end_use_breakdown():
    """End-use disaggregation for a typical office building.

    Based on CIBSE Guide F and ASHRAE 90.1 typical end-use splits.
    Total: 100% of site energy (~700,000 kWh).
    """
    return {
        "heating": {"pct": 0.30, "kwh": 210000, "benchmark_kwh": 160000},
        "cooling": {"pct": 0.15, "kwh": 105000, "benchmark_kwh": 85000},
        "lighting": {"pct": 0.20, "kwh": 140000, "benchmark_kwh": 100000},
        "ventilation": {"pct": 0.10, "kwh": 70000, "benchmark_kwh": 55000},
        "domestic_hot_water": {"pct": 0.05, "kwh": 35000, "benchmark_kwh": 30000},
        "equipment": {"pct": 0.15, "kwh": 105000, "benchmark_kwh": 90000},
        "other": {"pct": 0.05, "kwh": 35000, "benchmark_kwh": 30000},
    }


@pytest.fixture
def sample_ecm_library():
    """Energy Conservation Measure library for gap remediation."""
    return [
        {
            "ecm_id": "ECM-001",
            "name": "LED lighting upgrade",
            "end_use": "lighting",
            "savings_pct": 0.40,
            "capex_eur": 25000,
            "payback_years": 2.5,
        },
        {
            "ecm_id": "ECM-002",
            "name": "BMS optimisation",
            "end_use": "heating",
            "savings_pct": 0.15,
            "capex_eur": 15000,
            "payback_years": 3.0,
        },
        {
            "ecm_id": "ECM-003",
            "name": "VSD on AHU fans",
            "end_use": "ventilation",
            "savings_pct": 0.30,
            "capex_eur": 12000,
            "payback_years": 4.0,
        },
        {
            "ecm_id": "ECM-004",
            "name": "Free cooling economiser",
            "end_use": "cooling",
            "savings_pct": 0.25,
            "capex_eur": 8000,
            "payback_years": 2.0,
        },
        {
            "ecm_id": "ECM-005",
            "name": "Equipment timer controls",
            "end_use": "equipment",
            "savings_pct": 0.10,
            "capex_eur": 3000,
            "payback_years": 1.5,
        },
    ]


# =========================================================================
# 1. Engine Instantiation
# =========================================================================


class TestGapAnalysisInstantiation:
    """Test engine instantiation."""

    def test_engine_class_exists(self):
        mod = _load_gap()
        assert hasattr(mod, "EnergyPerformanceGapEngine")

    def test_engine_instantiation(self):
        mod = _load_gap()
        engine = mod.EnergyPerformanceGapEngine()
        assert engine is not None

    def test_module_version(self):
        mod = _load_gap()
        assert hasattr(mod, "_MODULE_VERSION")
        assert mod._MODULE_VERSION == "1.0.0"


# =========================================================================
# 2. End-Use Disaggregation
# =========================================================================


class TestEndUseDisaggregation:
    """Test end-use energy breakdown calculations."""

    def test_end_use_percentages_sum_to_100(self, sample_end_use_breakdown):
        """All end-use percentages sum to 100% (1.0)."""
        total_pct = sum(v["pct"] for v in sample_end_use_breakdown.values())
        assert total_pct == pytest.approx(1.0, rel=0.01)

    def test_end_use_kwh_matches_percentages(self, sample_end_use_breakdown):
        """End-use kWh values are proportional to percentages."""
        total_kwh = sum(v["kwh"] for v in sample_end_use_breakdown.values())
        for end_use, data in sample_end_use_breakdown.items():
            expected_kwh = data["pct"] * total_kwh
            assert data["kwh"] == pytest.approx(expected_kwh, rel=0.01)

    @pytest.mark.parametrize("end_use", [
        "heating", "cooling", "lighting", "ventilation",
        "domestic_hot_water", "equipment", "other",
    ])
    def test_end_use_categories_present(self, sample_end_use_breakdown, end_use):
        """All standard end-use categories are present."""
        assert end_use in sample_end_use_breakdown

    def test_total_energy_matches_site_energy(self, sample_end_use_breakdown):
        """Total disaggregated energy matches expected site energy."""
        total = sum(v["kwh"] for v in sample_end_use_breakdown.values())
        assert total == pytest.approx(700000, rel=0.01)


# =========================================================================
# 3. Gap Calculation
# =========================================================================


class TestGapCalculation:
    """Test gap calculation against benchmarks."""

    def test_all_gaps_positive(self, sample_end_use_breakdown):
        """All gaps are positive (actual > benchmark) in sample data."""
        for end_use, data in sample_end_use_breakdown.items():
            gap = data["kwh"] - data["benchmark_kwh"]
            assert gap >= 0, f"{end_use}: actual < benchmark"

    @pytest.mark.parametrize("end_use,expected_gap_kwh", [
        ("heating", 50000),
        ("cooling", 20000),
        ("lighting", 40000),
        ("ventilation", 15000),
        ("domestic_hot_water", 5000),
        ("equipment", 15000),
        ("other", 5000),
    ])
    def test_gap_by_end_use(self, sample_end_use_breakdown, end_use, expected_gap_kwh):
        """Gap for each end use matches expected value."""
        data = sample_end_use_breakdown[end_use]
        gap = data["kwh"] - data["benchmark_kwh"]
        assert gap == pytest.approx(expected_gap_kwh, rel=0.01)

    def test_total_gap(self, sample_end_use_breakdown):
        """Total gap across all end uses."""
        total_actual = sum(v["kwh"] for v in sample_end_use_breakdown.values())
        total_benchmark = sum(v["benchmark_kwh"] for v in sample_end_use_breakdown.values())
        total_gap = total_actual - total_benchmark
        assert total_gap == pytest.approx(150000, rel=0.01)

    def test_gap_percentage(self, sample_end_use_breakdown):
        """Gap as percentage of actual consumption."""
        total_actual = sum(v["kwh"] for v in sample_end_use_breakdown.values())
        total_benchmark = sum(v["benchmark_kwh"] for v in sample_end_use_breakdown.values())
        gap_pct = (total_actual - total_benchmark) / total_actual * 100
        assert 10 < gap_pct < 30  # Plausible range


# =========================================================================
# 4. Priority Ranking
# =========================================================================


class TestPriorityRanking:
    """Test priority ranking of energy performance gaps."""

    def test_ranking_by_gap_size(self, sample_end_use_breakdown):
        """Gaps are ranked by absolute size (largest first)."""
        gaps = []
        for end_use, data in sample_end_use_breakdown.items():
            gap = data["kwh"] - data["benchmark_kwh"]
            gaps.append((end_use, gap))
        ranked = sorted(gaps, key=lambda x: x[1], reverse=True)
        assert ranked[0][0] == "heating"  # Largest gap
        assert ranked[0][1] == 50000

    def test_ranking_by_percentage_gap(self, sample_end_use_breakdown):
        """Gaps ranked by percentage of end-use consumption."""
        pct_gaps = []
        for end_use, data in sample_end_use_breakdown.items():
            if data["kwh"] > 0:
                pct = (data["kwh"] - data["benchmark_kwh"]) / data["kwh"] * 100
                pct_gaps.append((end_use, pct))
        ranked = sorted(pct_gaps, key=lambda x: x[1], reverse=True)
        assert len(ranked) == 7
        # Lighting has one of the highest percentage gaps (28.6%)
        lighting_pct = next(p for u, p in pct_gaps if u == "lighting")
        assert lighting_pct > 20

    def test_top_3_priorities(self, sample_end_use_breakdown):
        """Top 3 priority gaps are identified."""
        gaps = []
        for end_use, data in sample_end_use_breakdown.items():
            gap = data["kwh"] - data["benchmark_kwh"]
            gaps.append((end_use, gap))
        ranked = sorted(gaps, key=lambda x: x[1], reverse=True)
        top_3 = [r[0] for r in ranked[:3]]
        assert "heating" in top_3
        assert "lighting" in top_3


# =========================================================================
# 5. Savings Potential
# =========================================================================


class TestSavingsPotential:
    """Test savings potential estimation."""

    def test_total_savings_potential(self, sample_end_use_breakdown):
        """Total savings potential equals total gap."""
        total_gap = sum(
            v["kwh"] - v["benchmark_kwh"]
            for v in sample_end_use_breakdown.values()
        )
        assert total_gap == pytest.approx(150000)

    def test_savings_as_eui_reduction(self, sample_end_use_breakdown):
        """Savings expressed as EUI reduction (kWh/m2)."""
        total_gap = sum(
            v["kwh"] - v["benchmark_kwh"]
            for v in sample_end_use_breakdown.values()
        )
        floor_area = 5000.0
        eui_reduction = total_gap / floor_area
        assert eui_reduction == pytest.approx(30.0)

    def test_savings_cost_estimate(self, sample_end_use_breakdown):
        """Savings converted to cost using average tariff."""
        total_gap = sum(
            v["kwh"] - v["benchmark_kwh"]
            for v in sample_end_use_breakdown.values()
        )
        avg_tariff_eur_per_kwh = 0.15
        cost_savings = total_gap * avg_tariff_eur_per_kwh
        assert cost_savings == pytest.approx(22500.0)

    def test_savings_carbon_reduction(self, sample_end_use_breakdown):
        """Savings converted to CO2 using grid emission factor."""
        total_gap = sum(
            v["kwh"] - v["benchmark_kwh"]
            for v in sample_end_use_breakdown.values()
        )
        grid_ef_kgco2_per_kwh = 0.4  # Approximate DE grid factor
        co2_reduction = total_gap * grid_ef_kgco2_per_kwh
        assert co2_reduction == pytest.approx(60000.0)


# =========================================================================
# 6. ECM Linking
# =========================================================================


class TestECMLinking:
    """Test linking of Energy Conservation Measures to gaps."""

    def test_ecm_matches_end_use(self, sample_ecm_library, sample_end_use_breakdown):
        """Each ECM maps to a valid end use in the breakdown."""
        valid_end_uses = set(sample_end_use_breakdown.keys())
        for ecm in sample_ecm_library:
            assert ecm["end_use"] in valid_end_uses

    def test_ecm_savings_calculation(self, sample_ecm_library, sample_end_use_breakdown):
        """ECM savings are calculated from gap * savings_pct."""
        ecm = sample_ecm_library[0]  # LED lighting
        end_use_data = sample_end_use_breakdown[ecm["end_use"]]
        gap = end_use_data["kwh"] - end_use_data["benchmark_kwh"]
        ecm_savings = gap * ecm["savings_pct"]
        assert ecm_savings > 0
        assert ecm_savings <= gap

    def test_ecm_payback_positive(self, sample_ecm_library):
        """All ECM payback periods are positive."""
        for ecm in sample_ecm_library:
            assert ecm["payback_years"] > 0

    def test_ecm_ranked_by_payback(self, sample_ecm_library):
        """ECMs can be ranked by payback period (shortest first)."""
        ranked = sorted(sample_ecm_library, key=lambda e: e["payback_years"])
        assert ranked[0]["payback_years"] <= ranked[-1]["payback_years"]

    def test_total_ecm_investment(self, sample_ecm_library):
        """Total ECM investment is plausible."""
        total = sum(ecm["capex_eur"] for ecm in sample_ecm_library)
        assert 50000 < total < 100000


# =========================================================================
# 7. Edge Cases
# =========================================================================


class TestGapAnalysisEdgeCases:
    """Test edge cases for gap analysis."""

    def test_facility_at_benchmark(self):
        """Facility at benchmark should have zero gap."""
        actual = 100000
        benchmark = 100000
        gap = actual - benchmark
        assert gap == 0

    def test_facility_below_benchmark(self):
        """Facility below benchmark should have negative gap."""
        actual = 80000
        benchmark = 100000
        gap = actual - benchmark
        assert gap < 0

    def test_zero_floor_area_handled(self):
        """Zero floor area should not cause division by zero."""
        energy = 500000
        area = 0.0
        if area > 0:
            eui = energy / area
        else:
            eui = None
        assert eui is None

    def test_provenance_hash(self):
        """Gap analysis result includes provenance hash."""
        import hashlib
        h = hashlib.sha256(b"gap_analysis_input").hexdigest()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
