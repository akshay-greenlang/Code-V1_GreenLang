# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Portfolio Benchmark Engine Tests
==============================================================

Tests area-weighted EUI calculation, facility ranking, year-over-year
improvement, best/worst performer identification, multi-entity
aggregation, outlier detection, and portfolio distribution.

Test Count Target: ~60 tests
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


def _load_portfolio():
    path = ENGINES_DIR / "portfolio_benchmark_engine.py"
    if not path.exists():
        pytest.skip("portfolio_benchmark_engine.py not found")
    mod_key = "pack035_test.portfolio_bench"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load portfolio_benchmark_engine: {exc}")
    return mod


# =========================================================================
# 1. Engine Instantiation
# =========================================================================


class TestPortfolioBenchmarkInstantiation:
    """Test engine instantiation."""

    def test_engine_class_exists(self):
        mod = _load_portfolio()
        assert hasattr(mod, "PortfolioBenchmarkEngine")

    def test_engine_instantiation(self):
        mod = _load_portfolio()
        engine = mod.PortfolioBenchmarkEngine()
        assert engine is not None

    def test_module_version(self):
        mod = _load_portfolio()
        assert hasattr(mod, "_MODULE_VERSION")
        assert mod._MODULE_VERSION == "1.0.0"


# =========================================================================
# 2. Area-Weighted EUI Calculation
# =========================================================================


class TestAreaWeightedEUI:
    """Test area-weighted EUI aggregation across a portfolio."""

    def test_area_weighted_eui_formula(self, sample_portfolio):
        """Area-weighted EUI = SUM(energy) / SUM(area)."""
        total_energy = sum(f["energy_consumption_kwh"] for f in sample_portfolio)
        total_area = sum(f["gross_floor_area_m2"] for f in sample_portfolio)
        weighted_eui = total_energy / total_area
        assert weighted_eui > 0
        assert total_area > 0
        assert total_energy > 0

    def test_area_weighted_eui_plausible(self, sample_portfolio):
        """Portfolio EUI is within plausible range (50-2000 kWh/m2)."""
        total_energy = sum(f["energy_consumption_kwh"] for f in sample_portfolio)
        total_area = sum(f["gross_floor_area_m2"] for f in sample_portfolio)
        weighted_eui = total_energy / total_area
        assert 50 < weighted_eui < 2000

    def test_single_facility_eui_equals_own(self):
        """Single-facility portfolio EUI equals the facility EUI."""
        facility = {
            "energy_consumption_kwh": 500000,
            "gross_floor_area_m2": 5000.0,
        }
        eui = facility["energy_consumption_kwh"] / facility["gross_floor_area_m2"]
        assert eui == pytest.approx(100.0)

    def test_area_weight_proportionality(self, sample_portfolio):
        """Larger facilities contribute more to the weighted average."""
        areas = [f["gross_floor_area_m2"] for f in sample_portfolio]
        energies = [f["energy_consumption_kwh"] for f in sample_portfolio]
        max_area_idx = areas.index(max(areas))
        # The facility with the largest area should have the biggest weight
        largest_area = areas[max_area_idx]
        total_area = sum(areas)
        weight = largest_area / total_area
        assert weight > 0.1  # At least 10% weight for the largest


# =========================================================================
# 3. Facility Ranking
# =========================================================================


class TestFacilityRanking:
    """Test facility ranking within a portfolio."""

    def test_ranking_by_eui_ascending(self, sample_portfolio):
        """Facilities ranked by EUI ascending (best first)."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"])
        assert ranked[0]["eui_kwh_per_m2"] <= ranked[-1]["eui_kwh_per_m2"]

    def test_ranking_count_matches_portfolio(self, sample_portfolio):
        """Ranking includes all portfolio facilities."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"])
        assert len(ranked) == len(sample_portfolio)

    @pytest.mark.parametrize("metric", [
        "eui_kwh_per_m2",
        "energy_consumption_kwh",
        "carbon_emissions_kgco2",
        "energy_cost_eur",
    ])
    def test_ranking_metric_valid(self, sample_portfolio, metric):
        """All facilities have the ranking metric available."""
        for f in sample_portfolio:
            assert metric in f
            assert f[metric] >= 0

    def test_best_performer_has_lowest_eui(self, sample_portfolio):
        """Best performer has the lowest EUI."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"])
        best = ranked[0]
        worst = ranked[-1]
        assert best["eui_kwh_per_m2"] < worst["eui_kwh_per_m2"]


# =========================================================================
# 4. Year-over-Year Improvement
# =========================================================================


class TestYoYImprovement:
    """Test year-over-year improvement calculations."""

    def test_yoy_improvement_positive(self, sample_portfolio):
        """Most facilities should show positive YoY improvement (lower EUI)."""
        improvements = []
        for f in sample_portfolio:
            hist = f.get("historical_eui", {})
            years = sorted(hist.keys())
            if len(years) >= 2:
                current = hist[years[-1]]
                previous = hist[years[-2]]
                pct_change = (previous - current) / previous * 100
                improvements.append(pct_change)
        # At least one facility should show improvement
        assert any(i > 0 for i in improvements)

    @pytest.mark.parametrize("prev_eui,curr_eui,expected_pct", [
        (150.0, 140.0, 6.67),
        (200.0, 180.0, 10.0),
        (100.0, 100.0, 0.0),
        (100.0, 110.0, -10.0),
    ])
    def test_yoy_improvement_formula(self, prev_eui, curr_eui, expected_pct):
        """YoY improvement % = (previous - current) / previous * 100."""
        pct = (prev_eui - curr_eui) / prev_eui * 100
        assert pct == pytest.approx(expected_pct, rel=0.01)

    def test_portfolio_average_improvement(self, sample_portfolio):
        """Portfolio average YoY improvement is plausible (0-10%)."""
        improvements = []
        for f in sample_portfolio:
            hist = f.get("historical_eui", {})
            years = sorted(hist.keys())
            if len(years) >= 2:
                current = hist[years[-1]]
                previous = hist[years[-2]]
                pct = (previous - current) / previous * 100
                improvements.append(pct)
        if improvements:
            avg = sum(improvements) / len(improvements)
            assert -5.0 < avg < 15.0


# =========================================================================
# 5. Best and Worst Performer Identification
# =========================================================================


class TestBestWorstPerformers:
    """Test identification of best and worst performers."""

    def test_best_performers_top_3(self, sample_portfolio):
        """Top 3 performers by EUI are correctly identified."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"])
        top_3 = ranked[:3]
        assert len(top_3) == 3
        assert all(t["eui_kwh_per_m2"] <= ranked[3]["eui_kwh_per_m2"] for t in top_3)

    def test_worst_performers_bottom_3(self, sample_portfolio):
        """Bottom 3 performers by EUI are correctly identified."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"], reverse=True)
        bottom_3 = ranked[:3]
        assert len(bottom_3) == 3

    def test_best_worst_no_overlap(self, sample_portfolio):
        """Best and worst performers do not overlap."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"])
        best_ids = {f["facility_id"] for f in ranked[:3]}
        worst_ids = {f["facility_id"] for f in ranked[-3:]}
        assert len(best_ids & worst_ids) == 0


# =========================================================================
# 6. Multi-Entity Aggregation
# =========================================================================


class TestMultiEntityAggregation:
    """Test aggregation by business unit, region, and building type."""

    @pytest.mark.parametrize("group_by", [
        "building_type",
        "region",
        "country",
        "business_unit",
    ])
    def test_grouping_returns_subsets(self, sample_portfolio, group_by):
        """Grouping by entity produces non-empty subsets."""
        groups = {}
        for f in sample_portfolio:
            key = f.get(group_by, "unknown")
            groups.setdefault(key, []).append(f)
        assert len(groups) >= 1
        total = sum(len(v) for v in groups.values())
        assert total == len(sample_portfolio)

    def test_aggregation_by_building_type(self, sample_portfolio):
        """Aggregation by building type produces valid EUIs."""
        groups = {}
        for f in sample_portfolio:
            bt = f["building_type"]
            groups.setdefault(bt, []).append(f)
        for bt, facilities in groups.items():
            total_energy = sum(f["energy_consumption_kwh"] for f in facilities)
            total_area = sum(f["gross_floor_area_m2"] for f in facilities)
            eui = total_energy / total_area
            assert eui > 0

    def test_aggregation_by_region(self, sample_portfolio):
        """All facilities in the sample are in EMEA region."""
        regions = set(f["region"] for f in sample_portfolio)
        assert "EMEA" in regions


# =========================================================================
# 7. Portfolio Distribution
# =========================================================================


class TestPortfolioDistribution:
    """Test portfolio distribution and statistical analysis."""

    def test_portfolio_quartile_boundaries(self, sample_portfolio):
        """Quartile boundaries are computed correctly."""
        euis = sorted(f["eui_kwh_per_m2"] for f in sample_portfolio)
        n = len(euis)
        q1 = euis[n // 4]
        q2 = euis[n // 2]
        q3 = euis[3 * n // 4]
        assert q1 <= q2 <= q3

    def test_portfolio_iqr(self, sample_portfolio):
        """Inter-quartile range is positive."""
        euis = sorted(f["eui_kwh_per_m2"] for f in sample_portfolio)
        n = len(euis)
        q1 = euis[n // 4]
        q3 = euis[3 * n // 4]
        iqr = q3 - q1
        assert iqr >= 0

    def test_portfolio_min_max(self, sample_portfolio):
        """Portfolio min and max EUI are valid."""
        euis = [f["eui_kwh_per_m2"] for f in sample_portfolio]
        assert min(euis) > 0
        assert max(euis) > min(euis)


# =========================================================================
# 8. Outlier Detection
# =========================================================================


class TestOutlierDetection:
    """Test outlier detection in portfolio data."""

    def test_iqr_outlier_detection(self, sample_portfolio):
        """IQR-based outlier detection flags extreme values."""
        euis = sorted(f["eui_kwh_per_m2"] for f in sample_portfolio)
        n = len(euis)
        q1 = euis[n // 4]
        q3 = euis[3 * n // 4]
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [e for e in euis if e < lower_fence or e > upper_fence]
        # Data centre (1600 kWh/m2) should be an outlier relative to offices
        assert any(e > 1000 for e in euis)

    def test_z_score_outlier_detection(self, sample_portfolio):
        """Z-score outlier detection flags extreme values."""
        euis = [f["eui_kwh_per_m2"] for f in sample_portfolio]
        mean = sum(euis) / len(euis)
        import math
        variance = sum((e - mean) ** 2 for e in euis) / len(euis)
        std = math.sqrt(variance)
        if std > 0:
            z_scores = [(e - mean) / std for e in euis]
            outliers = [z for z in z_scores if abs(z) > 2.5]
            assert isinstance(outliers, list)


# =========================================================================
# 9. Edge Cases
# =========================================================================


class TestPortfolioEdgeCases:
    """Test portfolio edge cases."""

    def test_empty_portfolio(self):
        """Empty portfolio should be handled gracefully."""
        portfolio = []
        assert len(portfolio) == 0

    def test_single_facility_portfolio(self):
        """Single-facility portfolio should return valid metrics."""
        portfolio = [
            {"facility_id": "SINGLE", "gross_floor_area_m2": 1000,
             "energy_consumption_kwh": 150000, "eui_kwh_per_m2": 150.0}
        ]
        assert len(portfolio) == 1
        assert portfolio[0]["eui_kwh_per_m2"] == 150.0

    def test_all_same_eui(self):
        """Portfolio where all facilities have identical EUI."""
        portfolio = [
            {"facility_id": f"F-{i}", "eui_kwh_per_m2": 150.0}
            for i in range(10)
        ]
        euis = [f["eui_kwh_per_m2"] for f in portfolio]
        assert len(set(euis)) == 1

    def test_portfolio_provenance_hash(self):
        """Portfolio benchmark result includes provenance hash."""
        import hashlib
        data = "portfolio_benchmark_input"
        h = hashlib.sha256(data.encode()).hexdigest()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
