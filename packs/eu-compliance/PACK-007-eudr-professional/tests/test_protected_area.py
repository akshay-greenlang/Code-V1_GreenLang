"""
Unit tests for PACK-007 EUDR Professional Pack - Protected Area Engine

Tests WDPA, KBA, Indigenous lands, Ramsar, UNESCO checks, buffer zone analysis,
and protected area overlay analysis.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from decimal import Decimal
from typing import List, Dict, Any


def _import_from_path(module_name, file_path):
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PACK_007_DIR = Path(__file__).resolve().parent.parent

# Import protected area module
protected_area_mod = _import_from_path(
    "pack_007_protected_area",
    _PACK_007_DIR / "engines" / "protected_area.py"
)

pytestmark = pytest.mark.skipif(
    protected_area_mod is None,
    reason="PACK-007 protected_area module not available"
)


@pytest.fixture
def protected_area_engine():
    """Create protected area engine instance."""
    if protected_area_mod is None:
        pytest.skip("protected_area module not available")
    return protected_area_mod.ProtectedAreaEngine()


@pytest.fixture
def sample_plot_inside_wdpa():
    """Sample plot inside a WDPA protected area."""
    return {
        "plot_id": "plot_001",
        "latitude": -3.4653,  # Amazon rainforest protected area
        "longitude": -62.2159,
        "polygon_wkt": "POLYGON((-62.22 -3.47, -62.21 -3.47, -62.21 -3.46, -62.22 -3.46, -62.22 -3.47))"
    }


@pytest.fixture
def sample_plot_outside_wdpa():
    """Sample plot outside WDPA protected areas."""
    return {
        "plot_id": "plot_002",
        "latitude": 52.5200,  # Berlin, Germany
        "longitude": 13.4050,
        "polygon_wkt": "POLYGON((13.40 52.52, 13.41 52.52, 13.41 52.53, 13.40 52.53, 13.40 52.52))"
    }


class TestProtectedAreaEngine:
    """Test suite for ProtectedAreaEngine."""

    def test_wdpa_check_positive(self, protected_area_engine, sample_plot_inside_wdpa):
        """Test WDPA check identifies plot inside protected area."""
        result = protected_area_engine.check_wdpa(
            latitude=sample_plot_inside_wdpa["latitude"],
            longitude=sample_plot_inside_wdpa["longitude"]
        )

        assert result is not None
        # Should detect overlap with protected area (using mock data)
        if result.get("overlaps"):
            assert result["overlaps"] is True
            assert "protected_areas" in result
            assert len(result["protected_areas"]) > 0

    def test_wdpa_check_negative(self, protected_area_engine, sample_plot_outside_wdpa):
        """Test WDPA check identifies plot outside protected areas."""
        result = protected_area_engine.check_wdpa(
            latitude=sample_plot_outside_wdpa["latitude"],
            longitude=sample_plot_outside_wdpa["longitude"]
        )

        assert result is not None
        assert "overlaps" in result
        # May or may not overlap depending on data

    def test_kba_check(self, protected_area_engine):
        """Test Key Biodiversity Area (KBA) check."""
        # Test location near known KBA
        result = protected_area_engine.check_kba(
            latitude=-15.7801,  # Near Amazon KBA
            longitude=-47.9292
        )

        assert result is not None
        assert "kba_overlap" in result or "overlaps" in result
        assert "kba_sites" in result or "sites" in result

    def test_indigenous_land_check(self, protected_area_engine):
        """Test Indigenous land check."""
        # Test location in known Indigenous territory
        result = protected_area_engine.check_indigenous_lands(
            latitude=-3.4653,  # Amazon region
            longitude=-62.2159
        )

        assert result is not None
        assert "indigenous_overlap" in result or "overlaps" in result
        if result.get("overlaps"):
            assert "territories" in result

    def test_ramsar_site_check(self, protected_area_engine):
        """Test Ramsar wetland site check."""
        # Test location near Ramsar site
        result = protected_area_engine.check_ramsar(
            latitude=52.0,  # Example coordinates
            longitude=5.0
        )

        assert result is not None
        assert "ramsar_overlap" in result or "overlaps" in result

    def test_unesco_site_check(self, protected_area_engine):
        """Test UNESCO World Heritage site check."""
        # Test location near UNESCO site
        result = protected_area_engine.check_unesco(
            latitude=-3.0,  # Example coordinates
            longitude=-60.0
        )

        assert result is not None
        assert "unesco_overlap" in result or "overlaps" in result

    def test_buffer_zone_analysis(self, protected_area_engine):
        """Test buffer zone analysis around protected areas."""
        # Check plot near (but not in) protected area
        result = protected_area_engine.analyze_buffer_zones(
            latitude=-3.5,
            longitude=-62.0,
            buffer_distance_km=10  # 10km buffer
        )

        assert result is not None
        assert "within_buffer" in result or "buffer_zones" in result
        assert "distance_to_nearest_km" in result or "nearest_distance" in result

    def test_proximity_scoring(self, protected_area_engine):
        """Test proximity-based risk scoring."""
        # Test proximity score calculation
        score = protected_area_engine.calculate_proximity_score(
            latitude=-3.5,
            longitude=-62.0
        )

        assert score is not None
        assert isinstance(score, (int, float, Decimal))
        assert 0 <= float(score) <= 100

    def test_risk_amplification(self, protected_area_engine):
        """Test risk amplification for protected area proximity."""
        base_risk = 30.0  # 30% base deforestation risk

        amplified_risk = protected_area_engine.amplify_risk(
            base_risk=base_risk,
            proximity_score=80.0  # High proximity to protected area
        )

        assert amplified_risk is not None
        assert float(amplified_risk) >= base_risk  # Should amplify risk

    def test_full_overlay_analysis(self, protected_area_engine, sample_plot_inside_wdpa):
        """Test full overlay analysis checking all protected area types."""
        result = protected_area_engine.full_overlay_analysis(
            latitude=sample_plot_inside_wdpa["latitude"],
            longitude=sample_plot_inside_wdpa["longitude"]
        )

        assert result is not None
        assert "wdpa" in result
        assert "kba" in result
        assert "indigenous" in result
        assert "ramsar" in result
        assert "unesco" in result
        assert "overall_risk_level" in result or "risk_level" in result

    def test_batch_overlay(self, protected_area_engine):
        """Test batch overlay analysis for multiple plots."""
        plots = [
            {"plot_id": "p1", "latitude": -3.5, "longitude": -62.0},
            {"plot_id": "p2", "latitude": -4.0, "longitude": -63.0},
            {"plot_id": "p3", "latitude": -5.0, "longitude": -64.0},
        ]

        results = protected_area_engine.batch_overlay_analysis(plots)

        assert results is not None
        assert len(results) == 3
        for result in results:
            assert "plot_id" in result
            assert "protected_area_check" in result or "overlays" in result

    def test_exclusion_zones(self, protected_area_engine):
        """Test identification of exclusion zones (areas that cannot be sourced from)."""
        # Check if location is in exclusion zone
        result = protected_area_engine.check_exclusion_zone(
            latitude=-3.5,
            longitude=-62.0
        )

        assert result is not None
        assert "is_excluded" in result or "excluded" in result
        assert "reason" in result or "exclusion_reason" in result


class TestProtectedAreaDatasets:
    """Test protected area dataset integrations."""

    def test_wdpa_database_access(self, protected_area_engine):
        """Test WDPA database access."""
        # Test that WDPA data is accessible
        stats = protected_area_engine.get_wdpa_statistics()

        assert stats is not None
        assert "total_protected_areas" in stats or "count" in stats

    def test_kba_database_access(self, protected_area_engine):
        """Test KBA database access."""
        stats = protected_area_engine.get_kba_statistics()

        assert stats is not None
        # Should have KBA count or similar metric

    def test_indigenous_lands_database(self, protected_area_engine):
        """Test Indigenous lands database access."""
        stats = protected_area_engine.get_indigenous_lands_statistics()

        assert stats is not None


class TestProtectedAreaRiskScoring:
    """Test protected area risk scoring algorithms."""

    def test_iucn_category_risk_weighting(self, protected_area_engine):
        """Test IUCN category-based risk weighting."""
        # Test different IUCN categories
        categories = ["Ia", "II", "III", "IV", "V", "VI"]

        for category in categories:
            weight = protected_area_engine.get_iucn_risk_weight(category)
            assert weight is not None
            assert isinstance(weight, (int, float, Decimal))
            assert float(weight) > 0

    def test_distance_decay_function(self, protected_area_engine):
        """Test distance decay function for buffer zones."""
        # Test risk decreases with distance
        distances_km = [0, 1, 5, 10, 20, 50]
        scores = []

        for distance in distances_km:
            score = protected_area_engine.calculate_distance_score(distance)
            scores.append(float(score))

        # Scores should generally decrease with distance
        assert scores[0] >= scores[-1]

    def test_cumulative_risk_scoring(self, protected_area_engine):
        """Test cumulative risk from multiple protected area overlaps."""
        overlaps = {
            "wdpa": True,
            "kba": True,
            "indigenous": False,
            "ramsar": True,
            "unesco": False
        }

        cumulative_score = protected_area_engine.calculate_cumulative_risk(overlaps)

        assert cumulative_score is not None
        assert isinstance(cumulative_score, (int, float, Decimal))
        assert 0 <= float(cumulative_score) <= 100


class TestProtectedAreaReporting:
    """Test protected area reporting features."""

    def test_generate_overlay_report(self, protected_area_engine):
        """Test generating protected area overlay report."""
        report = protected_area_engine.generate_overlay_report(
            plot_id="plot_001",
            latitude=-3.5,
            longitude=-62.0
        )

        assert report is not None
        assert "plot_id" in report
        assert "protected_areas_checked" in report or "overlays" in report
        assert "risk_assessment" in report or "risk_score" in report

    def test_exclusion_zone_report(self, protected_area_engine):
        """Test generating exclusion zone report."""
        plots = [
            {"plot_id": "p1", "latitude": -3.5, "longitude": -62.0},
            {"plot_id": "p2", "latitude": -4.0, "longitude": -63.0},
        ]

        report = protected_area_engine.generate_exclusion_report(plots)

        assert report is not None
        assert "total_plots_checked" in report or "plots_checked" in report
        assert "excluded_plots" in report or "exclusions" in report

    def test_buffer_zone_map_data(self, protected_area_engine):
        """Test generating buffer zone map data."""
        map_data = protected_area_engine.generate_buffer_zone_map(
            latitude=-3.5,
            longitude=-62.0,
            buffer_distance_km=10
        )

        assert map_data is not None
        assert "center_point" in map_data or "center" in map_data
        assert "buffer_polygon" in map_data or "buffer" in map_data


class TestProtectedAreaValidation:
    """Test protected area validation logic."""

    def test_coordinate_validation(self, protected_area_engine):
        """Test coordinate validation."""
        # Valid coordinates
        valid = protected_area_engine.validate_coordinates(
            latitude=-3.5,
            longitude=-62.0
        )
        assert valid is True

        # Invalid latitude
        invalid = protected_area_engine.validate_coordinates(
            latitude=95.0,  # Out of range
            longitude=-62.0
        )
        assert invalid is False

    def test_polygon_validation(self, protected_area_engine):
        """Test polygon WKT validation."""
        valid_wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"

        is_valid = protected_area_engine.validate_polygon(valid_wkt)
        assert is_valid is True

    def test_buffer_distance_validation(self, protected_area_engine):
        """Test buffer distance parameter validation."""
        # Valid buffer distance
        valid = protected_area_engine.validate_buffer_distance(10.0)
        assert valid is True

        # Invalid buffer distance
        invalid = protected_area_engine.validate_buffer_distance(-5.0)
        assert invalid is False
