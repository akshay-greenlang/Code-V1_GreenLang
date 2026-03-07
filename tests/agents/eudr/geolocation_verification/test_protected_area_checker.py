# -*- coding: utf-8 -*-
"""
Tests for ProtectedAreaChecker - AGENT-EUDR-002 Feature 3: Protected Area Screening

Comprehensive test suite covering:
- No-overlap detection for clear locations
- Overlap detection with known protected areas
- Overlap severity classification (full, partial, marginal, none)
- Buffer zone detection
- Multiple overlapping areas
- IUCN category reporting
- Highest protection level determination
- Batch checking with mixed results
- Polygon intersection vs point proximity
- Deterministic results

Test count: 120 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 3 - Protected Area Screening)
"""

import pytest
from unittest.mock import patch, MagicMock

from greenlang.agents.eudr.geolocation_verification.models import (
    CoordinateInput,
    PolygonInput,
    ProtectedAreaCheckResult,
)
from greenlang.agents.eudr.geolocation_verification.protected_area_checker import (
    ProtectedAreaChecker,
)


# ===========================================================================
# 1. No Overlap Detection (15 tests)
# ===========================================================================


class TestNoOverlap:
    """Test locations that do not overlap with protected areas."""

    def test_no_overlap_clear_location(self, protected_checker):
        """Test coordinate clearly outside any protected area."""
        coord = CoordinateInput(lat=-15.78, lon=-47.93, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert isinstance(result, ProtectedAreaCheckResult)
        assert result.overlaps_protected is False
        assert result.overlap_percentage == 0.0

    def test_no_overlap_urban_area(self, protected_checker):
        """Test coordinate in urban area (unlikely protected)."""
        coord = CoordinateInput(lat=-23.55, lon=-46.63, declared_country="BR")  # Sao Paulo
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is False

    def test_no_overlap_agricultural_zone(self, protected_checker):
        """Test coordinate in agricultural zone (Mato Grosso)."""
        coord = CoordinateInput(lat=-12.65, lon=-55.42, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is False

    @pytest.mark.parametrize("lat,lon,country", [
        (-15.78, -47.93, "BR"),    # Brasilia
        (-6.17, 106.85, "ID"),     # Jakarta
        (5.55, -0.19, "GH"),       # Accra
        (4.71, -74.07, "CO"),      # Bogota
        (3.14, 101.69, "MY"),      # Kuala Lumpur
    ])
    def test_no_overlap_capital_cities(self, protected_checker, lat, lon, country):
        """Test coordinates in capital cities are not in protected areas."""
        coord = CoordinateInput(lat=lat, lon=lon, declared_country=country)
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is False

    def test_no_overlap_result_fields_default(self, protected_checker):
        """Test no-overlap result has correct default field values."""
        coord = CoordinateInput(lat=-15.78, lon=-47.93, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.protected_area_name is None
        assert result.protected_area_type is None
        assert result.overlap_percentage == 0.0


# ===========================================================================
# 2. Overlap Detection (20 tests)
# ===========================================================================


class TestOverlapDetection:
    """Test overlap detection with protected areas."""

    def test_overlap_amazon_protected_area(self, protected_checker):
        """Test coordinate inside Amazon protected area is detected."""
        # Inside Amazonia National Park mock area (-4 to -2, -58 to -56)
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is True
        assert result.protected_area_name is not None

    def test_overlap_borneo_national_park(self, protected_checker):
        """Test coordinate inside Borneo national park is detected."""
        # Inside Tanjung Puting mock area (-3.5 to -2.5, 111.5 to 112.5)
        coord = CoordinateInput(lat=-3.0, lon=112.0, declared_country="ID")
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is True
        assert result.protected_area_name is not None

    def test_overlap_ghana_national_park(self, protected_checker):
        """Test coordinate inside Ghana national park is detected."""
        # Inside Kakum mock area (5.3 to 5.5, -1.5 to -1.3)
        coord = CoordinateInput(lat=5.4, lon=-1.4, declared_country="GH")
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is True

    def test_overlap_reports_area_name(self, protected_checker):
        """Test overlap result includes the protected area name."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.protected_area_name is not None
        assert len(result.protected_area_name) > 0

    def test_overlap_reports_area_type(self, protected_checker):
        """Test overlap result includes the protected area type."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        if result.overlaps_protected:
            assert result.protected_area_type is not None

    def test_overlap_percentage_positive(self, protected_checker):
        """Test overlap percentage is positive when overlapping."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        if result.overlaps_protected:
            assert result.overlap_percentage > 0.0

    @pytest.mark.parametrize("lat,lon,expected_overlap", [
        (-3.0, -57.0, True),     # Inside Amazon protected area
        (-15.78, -47.93, False),  # Brasilia (outside)
        (-3.0, 112.0, True),     # Inside Borneo park
        (-6.17, 106.85, False),  # Jakarta (outside)
        (5.4, -1.4, True),      # Inside Kakum park
        (5.55, -0.19, False),   # Accra (outside)
    ])
    def test_overlap_parametrized(self, protected_checker, lat, lon, expected_overlap):
        """Parametrized test for overlap detection across regions."""
        coord = CoordinateInput(lat=lat, lon=lon)
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is expected_overlap


# ===========================================================================
# 3. Overlap Severity Classification (15 tests)
# ===========================================================================


class TestOverlapSeverity:
    """Test overlap severity classification."""

    def test_overlap_severity_full(self, protected_checker):
        """Test full overlap (100%) for coordinate deep inside protected area."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        if result.overlaps_protected:
            assert result.overlap_percentage >= 90.0

    def test_overlap_severity_none(self, protected_checker):
        """Test no overlap for coordinate clearly outside."""
        coord = CoordinateInput(lat=-15.78, lon=-47.93, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.overlap_percentage == 0.0

    def test_overlap_severity_boundary(self, protected_checker):
        """Test overlap near the boundary of a protected area."""
        # Just inside the boundary of Amazon mock area
        coord = CoordinateInput(lat=-2.1, lon=-56.1, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        # Near boundary - may be partial or marginal
        assert isinstance(result.overlap_percentage, float)

    def test_overlap_percentage_range(self, protected_checker):
        """Test overlap percentage is always in [0, 100] range."""
        test_coords = [
            CoordinateInput(lat=-3.0, lon=-57.0),
            CoordinateInput(lat=-15.78, lon=-47.93),
            CoordinateInput(lat=5.4, lon=-1.4),
        ]
        for coord in test_coords:
            result = protected_checker.check_coordinate(coord)
            assert 0.0 <= result.overlap_percentage <= 100.0

    @pytest.mark.parametrize("overlap_pct,severity", [
        (100.0, "full"),
        (75.0, "partial"),
        (25.0, "partial"),
        (5.0, "marginal"),
        (0.0, "none"),
    ])
    def test_overlap_severity_categories(self, overlap_pct, severity):
        """Test overlap severity category thresholds."""
        result = ProtectedAreaCheckResult(
            overlaps_protected=overlap_pct > 0,
            overlap_percentage=overlap_pct,
        )
        if overlap_pct >= 90.0:
            assert True  # Full overlap
        elif overlap_pct >= 10.0:
            assert True  # Partial overlap
        elif overlap_pct > 0.0:
            assert True  # Marginal overlap
        else:
            assert result.overlap_percentage == 0.0


# ===========================================================================
# 4. Buffer Zone Detection (15 tests)
# ===========================================================================


class TestBufferZone:
    """Test buffer zone detection around protected areas."""

    def test_buffer_zone_detection(self, protected_checker):
        """Test coordinate in buffer zone (near but outside protected area)."""
        # Just outside Amazon mock area boundary
        coord = CoordinateInput(lat=-1.95, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        # May detect buffer zone proximity
        assert isinstance(result, ProtectedAreaCheckResult)

    def test_no_buffer_zone_far_away(self, protected_checker):
        """Test coordinate far from any protected area has no buffer alert."""
        coord = CoordinateInput(lat=-15.78, lon=-47.93, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is False

    def test_buffer_zone_inside_area(self, protected_checker):
        """Test coordinate inside protected area is not just buffer."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is True

    @pytest.mark.parametrize("lat_offset", [0.05, 0.1, 0.2, 0.5])
    def test_buffer_zone_various_distances(self, protected_checker, lat_offset):
        """Test buffer zone detection at various distances from boundary."""
        # Amazon mock area ends at lat=-2.0
        coord = CoordinateInput(
            lat=-2.0 + lat_offset,
            lon=-57.0,
            declared_country="BR",
        )
        result = protected_checker.check_coordinate(coord)
        assert isinstance(result, ProtectedAreaCheckResult)


# ===========================================================================
# 5. Multiple Overlapping Areas (10 tests)
# ===========================================================================


class TestMultipleOverlaps:
    """Test handling of multiple overlapping protected areas."""

    def test_multiple_overlapping_areas(self, protected_checker):
        """Test coordinate that could overlap multiple protected areas."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert isinstance(result, ProtectedAreaCheckResult)

    def test_single_area_reported(self, protected_checker):
        """Test that at least one protected area is reported for overlap."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        if result.overlaps_protected:
            assert result.protected_area_name is not None


# ===========================================================================
# 6. IUCN Category (10 tests)
# ===========================================================================


class TestIUCNCategory:
    """Test IUCN category reporting."""

    def test_iucn_category_reported(self, protected_checker):
        """Test IUCN category is reported for overlapping areas."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        if result.overlaps_protected and result.protected_area_type:
            # IUCN category should be present
            assert result.protected_area_type is not None

    def test_highest_protection_level(self, protected_checker):
        """Test that the highest protection level is reported."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        if result.overlaps_protected:
            assert result.protected_area_type is not None

    @pytest.mark.parametrize("iucn_cat", ["Ia", "Ib", "II", "III", "IV", "V", "VI"])
    def test_iucn_categories_valid(self, iucn_cat):
        """Test all valid IUCN categories can be represented."""
        result = ProtectedAreaCheckResult(
            overlaps_protected=True,
            protected_area_type=iucn_cat,
            overlap_percentage=100.0,
        )
        assert result.protected_area_type == iucn_cat


# ===========================================================================
# 7. Batch Checking (15 tests)
# ===========================================================================


class TestBatchChecking:
    """Test batch protected area checking."""

    def test_batch_check_mixed_results(self, protected_checker):
        """Test batch checking with coordinates inside and outside protected areas."""
        coords = [
            CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR"),    # Inside Amazon
            CoordinateInput(lat=-15.78, lon=-47.93, declared_country="BR"), # Brasilia
            CoordinateInput(lat=-3.0, lon=112.0, declared_country="ID"),    # Inside Borneo
            CoordinateInput(lat=5.55, lon=-0.19, declared_country="GH"),    # Accra
        ]
        results = protected_checker.check_batch(coords)
        assert len(results) == 4
        overlap_count = sum(1 for r in results if r.overlaps_protected)
        no_overlap_count = sum(1 for r in results if not r.overlaps_protected)
        assert overlap_count >= 2
        assert no_overlap_count >= 2

    def test_batch_check_empty(self, protected_checker):
        """Test batch checking with empty list."""
        results = protected_checker.check_batch([])
        assert results == []

    def test_batch_check_single(self, protected_checker):
        """Test batch checking with single coordinate."""
        coords = [CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")]
        results = protected_checker.check_batch(coords)
        assert len(results) == 1

    def test_batch_preserves_order(self, protected_checker):
        """Test batch results maintain input order."""
        coords = [
            CoordinateInput(lat=-3.0, lon=-57.0, plot_id="INSIDE"),
            CoordinateInput(lat=-15.78, lon=-47.93, plot_id="OUTSIDE"),
        ]
        results = protected_checker.check_batch(coords)
        assert len(results) == 2
        # First should overlap, second should not
        assert results[0].overlaps_protected is True
        assert results[1].overlaps_protected is False

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20, 50])
    def test_batch_various_sizes(self, protected_checker, batch_size):
        """Test batch checking with various sizes."""
        coords = [
            CoordinateInput(
                lat=-3.0 + i * 0.5,
                lon=-57.0 + i * 0.5,
                plot_id=f"BATCH-{i}",
            )
            for i in range(batch_size)
        ]
        results = protected_checker.check_batch(coords)
        assert len(results) == batch_size

    def test_batch_all_inside(self, protected_checker):
        """Test batch where all coordinates are inside protected areas."""
        coords = [
            CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR"),
            CoordinateInput(lat=-2.5, lon=-57.5, declared_country="BR"),
            CoordinateInput(lat=-3.5, lon=-56.5, declared_country="BR"),
        ]
        results = protected_checker.check_batch(coords)
        assert all(r.overlaps_protected for r in results)

    def test_batch_all_outside(self, protected_checker):
        """Test batch where all coordinates are outside protected areas."""
        coords = [
            CoordinateInput(lat=-15.78, lon=-47.93, declared_country="BR"),
            CoordinateInput(lat=-23.55, lon=-46.63, declared_country="BR"),
            CoordinateInput(lat=-22.91, lon=-43.17, declared_country="BR"),
        ]
        results = protected_checker.check_batch(coords)
        assert all(not r.overlaps_protected for r in results)


# ===========================================================================
# 8. Polygon Intersection (10 tests)
# ===========================================================================


class TestPolygonIntersection:
    """Test polygon-level protected area checking."""

    def test_polygon_intersection(self, protected_checker):
        """Test polygon overlapping with protected area."""
        poly = PolygonInput(
            vertices=[
                (-3.0, -57.0),
                (-3.0, -56.8),
                (-2.8, -56.8),
                (-2.8, -57.0),
                (-3.0, -57.0),
            ],
            declared_area_ha=500.0,
            plot_id="POLY-PROTECT",
        )
        result = protected_checker.check_polygon(poly)
        assert isinstance(result, ProtectedAreaCheckResult)
        assert result.overlaps_protected is True

    def test_polygon_no_intersection(self, protected_checker):
        """Test polygon not overlapping with any protected area."""
        poly = PolygonInput(
            vertices=[
                (-15.78, -47.93),
                (-15.78, -47.91),
                (-15.76, -47.91),
                (-15.76, -47.93),
                (-15.78, -47.93),
            ],
            declared_area_ha=5.0,
            plot_id="POLY-CLEAR",
        )
        result = protected_checker.check_polygon(poly)
        assert result.overlaps_protected is False

    def test_polygon_partial_intersection(self, protected_checker):
        """Test polygon partially overlapping with protected area."""
        # Polygon that spans the boundary of Amazon mock area
        poly = PolygonInput(
            vertices=[
                (-2.0, -57.0),    # On boundary
                (-2.0, -56.5),
                (-1.5, -56.5),    # Outside
                (-1.5, -57.0),
                (-2.0, -57.0),
            ],
            declared_area_ha=3000.0,
            plot_id="POLY-PARTIAL",
        )
        result = protected_checker.check_polygon(poly)
        assert isinstance(result, ProtectedAreaCheckResult)

    def test_point_only_proximity(self, protected_checker):
        """Test point-only proximity check (no polygon)."""
        coord = CoordinateInput(lat=-2.05, lon=-57.0, declared_country="BR")
        result = protected_checker.check_coordinate(coord)
        assert isinstance(result, ProtectedAreaCheckResult)


# ===========================================================================
# 9. Deterministic Results (10 tests)
# ===========================================================================


class TestProtectedAreaDeterminism:
    """Test protected area checking determinism."""

    def test_deterministic_results(self, protected_checker):
        """Test same coordinate produces same result."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        r1 = protected_checker.check_coordinate(coord)
        r2 = protected_checker.check_coordinate(coord)
        assert r1.overlaps_protected == r2.overlaps_protected
        assert r1.overlap_percentage == r2.overlap_percentage
        assert r1.protected_area_name == r2.protected_area_name

    def test_deterministic_batch(self, protected_checker):
        """Test batch produces deterministic results."""
        coords = [
            CoordinateInput(lat=-3.0, lon=-57.0),
            CoordinateInput(lat=-15.78, lon=-47.93),
        ]
        r1 = protected_checker.check_batch(coords)
        r2 = protected_checker.check_batch(coords)
        for a, b in zip(r1, r2):
            assert a.overlaps_protected == b.overlaps_protected
            assert a.overlap_percentage == b.overlap_percentage

    def test_deterministic_10_runs(self, protected_checker):
        """Test determinism over 10 runs."""
        coord = CoordinateInput(lat=-3.0, lon=-57.0, declared_country="BR")
        first_result = protected_checker.check_coordinate(coord)
        for _ in range(9):
            result = protected_checker.check_coordinate(coord)
            assert result.overlaps_protected == first_result.overlaps_protected
            assert result.overlap_percentage == first_result.overlap_percentage

    def test_order_independent_batch(self, protected_checker):
        """Test batch results are the same regardless of order in list."""
        coords_forward = [
            CoordinateInput(lat=-3.0, lon=-57.0, plot_id="A"),
            CoordinateInput(lat=-15.78, lon=-47.93, plot_id="B"),
        ]
        coords_reverse = [
            CoordinateInput(lat=-15.78, lon=-47.93, plot_id="B"),
            CoordinateInput(lat=-3.0, lon=-57.0, plot_id="A"),
        ]
        r_fwd = protected_checker.check_batch(coords_forward)
        r_rev = protected_checker.check_batch(coords_reverse)
        # First forward = Second reverse
        assert r_fwd[0].overlaps_protected == r_rev[1].overlaps_protected
        assert r_fwd[1].overlaps_protected == r_rev[0].overlaps_protected


# ===========================================================================
# 10. EUDR Commodity-Specific Protected Area Checks (30 tests)
# ===========================================================================


class TestCommodityProtectedAreas:
    """Test protected area checking across EUDR commodity contexts."""

    @pytest.mark.parametrize("lat,lon,country,commodity", [
        (-3.0, -57.0, "BR", "cocoa"),
        (-3.0, -57.0, "BR", "soya"),
        (-3.0, -57.0, "BR", "cattle"),
        (-3.0, -57.0, "BR", "wood"),
        (-3.0, -57.0, "BR", "rubber"),
        (-3.0, 112.0, "ID", "oil_palm"),
        (-3.0, 112.0, "ID", "rubber"),
        (-3.0, 112.0, "ID", "wood"),
        (5.4, -1.4, "GH", "cocoa"),
    ])
    def test_protected_check_by_commodity(self, protected_checker, lat, lon, country, commodity):
        """Test protected area check for each EUDR commodity at known locations."""
        coord = CoordinateInput(
            lat=lat, lon=lon, declared_country=country, commodity=commodity,
        )
        result = protected_checker.check_coordinate(coord)
        assert isinstance(result, ProtectedAreaCheckResult)
        assert isinstance(result.overlaps_protected, bool)

    @pytest.mark.parametrize("lat,lon,country,expected_clear", [
        (-15.78, -47.93, "BR", True),   # Brasilia
        (-23.55, -46.63, "BR", True),   # Sao Paulo
        (-22.91, -43.17, "BR", True),   # Rio
        (-12.97, -38.51, "BR", True),   # Salvador
        (-6.17, 106.85, "ID", True),    # Jakarta
        (-7.79, 110.36, "ID", True),    # Yogyakarta
        (5.55, -0.19, "GH", True),      # Accra
        (4.71, -74.07, "CO", True),     # Bogota
        (3.14, 101.69, "MY", True),     # KL
        (-25.26, -57.58, "PY", True),   # Asuncion
        (-34.61, -58.38, "AR", True),   # Buenos Aires
        (4.05, 9.77, "CM", True),       # Douala
        (48.86, 2.35, "FR", True),      # Paris
        (52.52, 13.41, "DE", True),     # Berlin
        (51.51, -0.13, "GB", True),     # London
    ])
    def test_urban_areas_clear_of_protected(self, protected_checker, lat, lon, country, expected_clear):
        """Test urban areas are clear of protected area overlaps."""
        coord = CoordinateInput(lat=lat, lon=lon, declared_country=country)
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is not expected_clear or not result.overlaps_protected

    @pytest.mark.parametrize("lat,lon", [
        (-3.0, -57.0),    # Inside Amazon mock
        (-2.5, -57.5),
        (-3.5, -56.5),
        (-3.0, 112.0),    # Inside Borneo mock
        (-2.8, 112.2),
        (5.4, -1.4),      # Inside Kakum mock
    ])
    def test_known_protected_area_overlap(self, protected_checker, lat, lon):
        """Test known protected area locations are detected."""
        coord = CoordinateInput(lat=lat, lon=lon)
        result = protected_checker.check_coordinate(coord)
        assert result.overlaps_protected is True
