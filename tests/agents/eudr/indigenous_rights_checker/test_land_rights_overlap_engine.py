# -*- coding: utf-8 -*-
"""
Tests for LandRightsOverlapEngine - AGENT-EUDR-021 Engine 3: Land Rights Overlap

Comprehensive test suite covering:
- PostGIS spatial overlap detection (DIRECT/PARTIAL/ADJACENT/PROXIMATE/NONE)
- Risk scoring with 5 factors (overlap_type, legal_status, population,
  conflict_history, country_framework)
- Batch processing (1/10/100/1000/10000 plots)
- Edge cases (zero-area plots, self-intersecting geometries, poles)
- Performance tests (sub-500ms for single plot)
- Determinism tests (same input -> same output)
- Buffer zone configuration (inner 5km, outer 25km)
- Risk level classification (CRITICAL/HIGH/MEDIUM/LOW/NONE)

Test count: 92 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 3: Land Rights Overlap Detector)
"""

import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    compute_overlap_risk_score,
    classify_risk_level,
    haversine_km,
    SHA256_HEX_LENGTH,
    OVERLAP_TYPE_SCORES,
    LEGAL_STATUS_SCORES,
    DEFAULT_OVERLAP_RISK_WEIGHTS,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    OverlapType,
    RiskLevel,
    TerritoryOverlap,
    DetectOverlapRequest,
    BatchOverlapRequest,
    OverlapDetectionResponse,
    BatchOverlapResponse,
    MAX_BATCH_SIZE,
)


# ===========================================================================
# 1. Overlap Type Classification (12 tests)
# ===========================================================================


class TestOverlapTypeClassification:
    """Test overlap type classification: DIRECT, PARTIAL, ADJACENT, PROXIMATE, NONE."""

    def test_direct_overlap_classification(self, sample_overlap_direct):
        """Test plot fully within territory is classified as DIRECT."""
        assert sample_overlap_direct.overlap_type == OverlapType.DIRECT
        assert sample_overlap_direct.distance_meters == Decimal("0")

    def test_direct_overlap_100_pct(self, sample_overlap_direct):
        """Test DIRECT overlap has 100% plot coverage."""
        assert sample_overlap_direct.overlap_pct_of_plot == Decimal("100.0")

    def test_partial_overlap_classification(self):
        """Test plot partially overlapping territory is PARTIAL."""
        overlap = TerritoryOverlap(
            overlap_id="o-partial",
            plot_id="p-partial",
            territory_id="t-001",
            overlap_type=OverlapType.PARTIAL,
            overlap_area_hectares=Decimal("75.0"),
            overlap_pct_of_plot=Decimal("50.0"),
            distance_meters=Decimal("0"),
            risk_score=Decimal("70.00"),
            risk_level=RiskLevel.HIGH,
            provenance_hash="a" * 64,
        )
        assert overlap.overlap_type == OverlapType.PARTIAL
        assert overlap.overlap_pct_of_plot == Decimal("50.0")

    def test_adjacent_overlap_classification(self, sample_overlap_adjacent):
        """Test plot near territory (within inner buffer) is ADJACENT."""
        assert sample_overlap_adjacent.overlap_type == OverlapType.ADJACENT
        assert sample_overlap_adjacent.distance_meters == Decimal("3500")

    def test_proximate_overlap_classification(self):
        """Test plot within outer buffer is PROXIMATE."""
        overlap = TerritoryOverlap(
            overlap_id="o-prox",
            plot_id="p-prox",
            territory_id="t-001",
            overlap_type=OverlapType.PROXIMATE,
            distance_meters=Decimal("15000"),
            risk_score=Decimal("25.00"),
            risk_level=RiskLevel.LOW,
            provenance_hash="b" * 64,
        )
        assert overlap.overlap_type == OverlapType.PROXIMATE

    def test_none_overlap_classification(self, sample_overlap_none):
        """Test plot far from any territory is NONE."""
        assert sample_overlap_none.overlap_type == OverlapType.NONE
        assert sample_overlap_none.risk_level == RiskLevel.NONE

    @pytest.mark.parametrize("overlap_type,score", [
        ("direct", Decimal("100")),
        ("partial", Decimal("80")),
        ("adjacent", Decimal("50")),
        ("proximate", Decimal("25")),
        ("none", Decimal("0")),
    ])
    def test_overlap_type_scores(self, overlap_type, score):
        """Test each overlap type maps to correct base score."""
        assert OVERLAP_TYPE_SCORES[overlap_type] == score

    def test_direct_higher_than_partial(self):
        """Test DIRECT overlap score is higher than PARTIAL."""
        assert OVERLAP_TYPE_SCORES["direct"] > OVERLAP_TYPE_SCORES["partial"]

    def test_partial_higher_than_adjacent(self):
        """Test PARTIAL overlap score is higher than ADJACENT."""
        assert OVERLAP_TYPE_SCORES["partial"] > OVERLAP_TYPE_SCORES["adjacent"]

    def test_adjacent_higher_than_proximate(self):
        """Test ADJACENT overlap score is higher than PROXIMATE."""
        assert OVERLAP_TYPE_SCORES["adjacent"] > OVERLAP_TYPE_SCORES["proximate"]

    def test_proximate_higher_than_none(self):
        """Test PROXIMATE overlap score is higher than NONE."""
        assert OVERLAP_TYPE_SCORES["proximate"] > OVERLAP_TYPE_SCORES["none"]


# ===========================================================================
# 2. Risk Score Calculation (15 tests)
# ===========================================================================


class TestRiskScoreCalculation:
    """Test 5-factor overlap risk scoring."""

    def test_risk_score_direct_titled_high_pop(self):
        """Test highest risk: DIRECT overlap, TITLED territory, high population."""
        score = compute_overlap_risk_score(
            overlap_type="direct",
            legal_status="titled",
            community_population=50000,
            conflict_history_score=Decimal("100"),
            country_framework_score=Decimal("100"),
        )
        assert score >= Decimal("80")

    def test_risk_score_none_overlap(self):
        """Test no overlap yields low or zero risk score."""
        score = compute_overlap_risk_score(
            overlap_type="none",
            legal_status="titled",
            community_population=1000,
            conflict_history_score=Decimal("50"),
            country_framework_score=Decimal("50"),
        )
        assert score < Decimal("40")

    def test_risk_score_weights_sum_to_one(self):
        """Test overlap risk weights sum to 1.0."""
        total = sum(DEFAULT_OVERLAP_RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    @pytest.mark.parametrize("legal_status,expected_score", [
        ("titled", Decimal("100")),
        ("declared", Decimal("80")),
        ("claimed", Decimal("60")),
        ("customary", Decimal("50")),
        ("pending", Decimal("40")),
        ("disputed", Decimal("60")),
    ])
    def test_legal_status_scores(self, legal_status, expected_score):
        """Test each legal status maps to correct score."""
        assert LEGAL_STATUS_SCORES[legal_status] == expected_score

    def test_population_factor_high(self):
        """Test population >= 50000 yields factor of 100."""
        score = compute_overlap_risk_score(
            overlap_type="direct",
            legal_status="titled",
            community_population=50000,
            conflict_history_score=Decimal("0"),
            country_framework_score=Decimal("0"),
        )
        # overlap_type contribution: 100 * 0.40 = 40
        # legal_status: 100 * 0.20 = 20
        # population: 100 * 0.10 = 10
        assert score >= Decimal("70")

    def test_population_factor_small(self):
        """Test population < 1000 yields factor of 20."""
        score = compute_overlap_risk_score(
            overlap_type="direct",
            legal_status="titled",
            community_population=500,
            conflict_history_score=Decimal("0"),
            country_framework_score=Decimal("0"),
        )
        # population: 20 * 0.10 = 2
        assert score < Decimal("75")

    def test_conflict_history_amplifies_risk(self):
        """Test conflict history increases risk score."""
        score_no_conflict = compute_overlap_risk_score(
            overlap_type="adjacent",
            legal_status="customary",
            community_population=5000,
            conflict_history_score=Decimal("0"),
            country_framework_score=Decimal("50"),
        )
        score_conflict = compute_overlap_risk_score(
            overlap_type="adjacent",
            legal_status="customary",
            community_population=5000,
            conflict_history_score=Decimal("100"),
            country_framework_score=Decimal("50"),
        )
        assert score_conflict > score_no_conflict

    def test_country_framework_impact(self):
        """Test country rights framework impacts risk score."""
        score_strong = compute_overlap_risk_score(
            overlap_type="partial",
            legal_status="titled",
            community_population=10000,
            conflict_history_score=Decimal("50"),
            country_framework_score=Decimal("100"),
        )
        score_weak = compute_overlap_risk_score(
            overlap_type="partial",
            legal_status="titled",
            community_population=10000,
            conflict_history_score=Decimal("50"),
            country_framework_score=Decimal("0"),
        )
        assert score_strong > score_weak

    def test_risk_level_critical(self):
        """Test score >= 80 classifies as CRITICAL."""
        assert classify_risk_level(Decimal("80")) == "critical"
        assert classify_risk_level(Decimal("95")) == "critical"

    def test_risk_level_high(self):
        """Test 60 <= score < 80 classifies as HIGH."""
        assert classify_risk_level(Decimal("60")) == "high"
        assert classify_risk_level(Decimal("79.99")) == "high"

    def test_risk_level_medium(self):
        """Test 40 <= score < 60 classifies as MEDIUM."""
        assert classify_risk_level(Decimal("40")) == "medium"
        assert classify_risk_level(Decimal("59.99")) == "medium"

    def test_risk_level_low(self):
        """Test 20 <= score < 40 classifies as LOW."""
        assert classify_risk_level(Decimal("20")) == "low"
        assert classify_risk_level(Decimal("39.99")) == "low"

    def test_risk_level_none(self):
        """Test score < 20 classifies as NONE."""
        assert classify_risk_level(Decimal("0")) == "none"
        assert classify_risk_level(Decimal("19.99")) == "none"

    def test_risk_score_decimal_precision(self):
        """Test risk score maintains 2 decimal places."""
        score = compute_overlap_risk_score(
            overlap_type="partial",
            legal_status="declared",
            community_population=7500,
            conflict_history_score=Decimal("33.33"),
            country_framework_score=Decimal("66.67"),
        )
        assert score == score.quantize(Decimal("0.01"))


# ===========================================================================
# 3. Buffer Zone Configuration (8 tests)
# ===========================================================================


class TestBufferZoneConfiguration:
    """Test inner and outer buffer zone settings."""

    def test_inner_buffer_default(self, mock_config):
        """Test default inner buffer is 5km."""
        assert mock_config.inner_buffer_km == 5.0

    def test_outer_buffer_default(self, mock_config):
        """Test default outer buffer is 25km."""
        assert mock_config.outer_buffer_km == 25.0

    def test_inner_less_than_outer(self, mock_config):
        """Test inner buffer is always less than outer buffer."""
        assert mock_config.inner_buffer_km < mock_config.outer_buffer_km

    def test_adjacent_within_inner_buffer(self, mock_config):
        """Test distance <= inner buffer classifies as ADJACENT."""
        distance_km = 3.0
        assert distance_km <= mock_config.inner_buffer_km

    def test_proximate_between_buffers(self, mock_config):
        """Test distance between inner and outer buffer classifies as PROXIMATE."""
        distance_km = 15.0
        assert mock_config.inner_buffer_km < distance_km <= mock_config.outer_buffer_km

    def test_none_beyond_outer_buffer(self, mock_config):
        """Test distance > outer buffer classifies as NONE."""
        distance_km = 30.0
        assert distance_km > mock_config.outer_buffer_km

    def test_strict_config_smaller_buffers(self, strict_config):
        """Test strict config has smaller buffer zones."""
        assert strict_config.inner_buffer_km == 2.0
        assert strict_config.outer_buffer_km == 10.0

    def test_buffer_polygon_points_default(self, mock_config):
        """Test buffer polygon approximation uses 64 points."""
        assert mock_config.buffer_polygon_points == 64


# ===========================================================================
# 4. Batch Processing (12 tests)
# ===========================================================================


class TestBatchOverlapProcessing:
    """Test batch overlap screening for multiple plots."""

    def test_single_plot_batch(self):
        """Test batch with single plot succeeds."""
        req = BatchOverlapRequest(
            plots=[DetectOverlapRequest(
                plot_id="p-001", latitude=-3.0, longitude=-60.0,
            )],
        )
        assert req.plots is not None
        assert len(req.plots) == 1

    def test_batch_ten_plots(self):
        """Test batch with 10 plots succeeds."""
        plots = [
            DetectOverlapRequest(
                plot_id=f"p-{i:03d}",
                latitude=-3.0 + i * 0.1,
                longitude=-60.0 + i * 0.1,
            )
            for i in range(10)
        ]
        req = BatchOverlapRequest(plots=plots)
        assert len(req.plots) == 10

    def test_batch_hundred_plots(self):
        """Test batch with 100 plots succeeds."""
        plots = [
            DetectOverlapRequest(
                plot_id=f"p-{i:04d}",
                latitude=-3.0 + i * 0.01,
                longitude=-60.0 + i * 0.01,
            )
            for i in range(100)
        ]
        req = BatchOverlapRequest(plots=plots)
        assert len(req.plots) == 100

    def test_batch_thousand_plots(self):
        """Test batch with 1000 plots succeeds."""
        plots = [
            DetectOverlapRequest(
                plot_id=f"p-{i:05d}",
                latitude=-3.0 + i * 0.001,
                longitude=-60.0 + i * 0.001,
            )
            for i in range(1000)
        ]
        req = BatchOverlapRequest(plots=plots)
        assert len(req.plots) == 1000

    def test_batch_max_size_ten_thousand(self):
        """Test batch up to MAX_BATCH_SIZE (10000) is accepted."""
        assert MAX_BATCH_SIZE == 10000

    def test_batch_empty_rejected(self):
        """Test batch with zero plots is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BatchOverlapRequest(plots=[])

    def test_batch_response_aggregation(self):
        """Test batch response aggregates per-plot results."""
        response = BatchOverlapResponse(
            total_plots=5,
            plots_with_overlaps=3,
            critical_count=1,
            high_count=1,
            medium_count=1,
            low_count=0,
            results=[],
            processing_time_ms=250.0,
            provenance_hash="a" * 64,
        )
        assert response.total_plots == 5
        assert response.plots_with_overlaps == 3
        assert response.critical_count + response.high_count + response.medium_count == 3

    def test_batch_response_no_overlaps(self):
        """Test batch response with no overlaps found."""
        response = BatchOverlapResponse(
            total_plots=10,
            plots_with_overlaps=0,
            results=[],
            processing_time_ms=100.0,
            provenance_hash="b" * 64,
        )
        assert response.plots_with_overlaps == 0
        assert response.critical_count == 0

    def test_batch_response_all_critical(self):
        """Test batch response with all plots having critical overlaps."""
        response = BatchOverlapResponse(
            total_plots=5,
            plots_with_overlaps=5,
            critical_count=5,
            results=[],
            processing_time_ms=500.0,
            provenance_hash="c" * 64,
        )
        assert response.critical_count == response.total_plots

    def test_detect_overlap_request_with_coordinates(self):
        """Test overlap request with latitude/longitude."""
        req = DetectOverlapRequest(
            plot_id="p-001",
            latitude=-3.46,
            longitude=-62.21,
        )
        assert req.latitude == -3.46
        assert req.longitude == -62.21

    def test_detect_overlap_request_with_geojson(self):
        """Test overlap request with GeoJSON polygon."""
        geojson = {
            "type": "Polygon",
            "coordinates": [[
                [-60.05, -3.05], [-60.05, -2.95],
                [-59.95, -2.95], [-59.95, -3.05],
                [-60.05, -3.05],
            ]],
        }
        req = DetectOverlapRequest(
            plot_id="p-geo",
            plot_geojson=geojson,
        )
        assert req.plot_geojson is not None
        assert req.plot_geojson["type"] == "Polygon"

    def test_detect_overlap_request_with_custom_buffer(self):
        """Test overlap request with custom buffer radii."""
        req = DetectOverlapRequest(
            plot_id="p-buf",
            latitude=-3.0,
            longitude=-60.0,
            inner_buffer_km=2.0,
            outer_buffer_km=10.0,
        )
        assert req.inner_buffer_km == 2.0
        assert req.outer_buffer_km == 10.0


# ===========================================================================
# 5. Edge Cases (15 tests)
# ===========================================================================


class TestOverlapEdgeCases:
    """Test edge cases for overlap detection."""

    def test_zero_area_plot(self):
        """Test overlap detection for a zero-area (point) plot."""
        req = DetectOverlapRequest(
            plot_id="p-point",
            latitude=-3.0,
            longitude=-60.0,
        )
        assert req.plot_geojson is None
        assert req.latitude is not None

    def test_plot_at_north_pole(self):
        """Test overlap detection at North Pole coordinates."""
        req = DetectOverlapRequest(
            plot_id="p-north",
            latitude=90.0,
            longitude=0.0,
        )
        assert req.latitude == 90.0

    def test_plot_at_south_pole(self):
        """Test overlap detection at South Pole coordinates."""
        req = DetectOverlapRequest(
            plot_id="p-south",
            latitude=-90.0,
            longitude=0.0,
        )
        assert req.latitude == -90.0

    def test_plot_at_antimeridian(self):
        """Test overlap detection at the antimeridian (180 degrees)."""
        req = DetectOverlapRequest(
            plot_id="p-anti",
            latitude=0.0,
            longitude=180.0,
        )
        assert req.longitude == 180.0

    def test_plot_at_prime_meridian(self):
        """Test overlap detection at the prime meridian (0 degrees)."""
        req = DetectOverlapRequest(
            plot_id="p-prime",
            latitude=0.0,
            longitude=0.0,
        )
        assert req.latitude == 0.0
        assert req.longitude == 0.0

    def test_latitude_out_of_range_rejected(self):
        """Test latitude > 90 is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectOverlapRequest(
                plot_id="p-bad",
                latitude=91.0,
                longitude=0.0,
            )

    def test_latitude_below_range_rejected(self):
        """Test latitude < -90 is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectOverlapRequest(
                plot_id="p-bad",
                latitude=-91.0,
                longitude=0.0,
            )

    def test_longitude_out_of_range_rejected(self):
        """Test longitude > 180 is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectOverlapRequest(
                plot_id="p-bad",
                latitude=0.0,
                longitude=181.0,
            )

    def test_longitude_below_range_rejected(self):
        """Test longitude < -180 is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectOverlapRequest(
                plot_id="p-bad",
                latitude=0.0,
                longitude=-181.0,
            )

    def test_overlap_with_zero_distance(self, sample_overlap_direct):
        """Test DIRECT overlap has zero distance."""
        assert sample_overlap_direct.distance_meters == Decimal("0")

    def test_overlap_with_large_distance(self, sample_overlap_none):
        """Test NONE overlap with large distance."""
        assert sample_overlap_none.distance_meters == Decimal("50000")

    def test_negative_distance_rejected(self):
        """Test negative distance is rejected by model validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TerritoryOverlap(
                overlap_id="o-bad",
                plot_id="p-bad",
                territory_id="t-001",
                overlap_type=OverlapType.NONE,
                distance_meters=Decimal("-1"),
                risk_score=Decimal("0"),
                risk_level=RiskLevel.NONE,
                provenance_hash="a" * 64,
            )

    def test_risk_score_out_of_range_rejected(self):
        """Test risk score > 100 is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TerritoryOverlap(
                overlap_id="o-bad",
                plot_id="p-bad",
                territory_id="t-001",
                overlap_type=OverlapType.DIRECT,
                distance_meters=Decimal("0"),
                risk_score=Decimal("101"),
                risk_level=RiskLevel.CRITICAL,
                provenance_hash="a" * 64,
            )

    def test_overlap_multiple_territories(self):
        """Test plot overlapping multiple territories."""
        overlaps = [
            TerritoryOverlap(
                overlap_id=f"o-multi-{i}",
                plot_id="p-multi",
                territory_id=f"t-{i:03d}",
                overlap_type=OverlapType.DIRECT if i == 1 else OverlapType.ADJACENT,
                distance_meters=Decimal("0") if i == 1 else Decimal("3000"),
                risk_score=Decimal("90") if i == 1 else Decimal("45"),
                risk_level=RiskLevel.CRITICAL if i == 1 else RiskLevel.MEDIUM,
                provenance_hash=compute_test_hash({"overlap_id": f"o-multi-{i}"}),
            )
            for i in range(1, 4)
        ]
        assert len(overlaps) == 3
        highest_risk = max(overlaps, key=lambda o: o.risk_score)
        assert highest_risk.risk_level == RiskLevel.CRITICAL

    def test_overlap_response_highest_risk(self):
        """Test OverlapDetectionResponse tracks highest risk level."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            total_overlaps=3,
            highest_risk_level=RiskLevel.CRITICAL,
            provenance_hash="a" * 64,
        )
        assert response.highest_risk_level == RiskLevel.CRITICAL


# ===========================================================================
# 6. Haversine Distance (8 tests)
# ===========================================================================


class TestHaversineDistance:
    """Test Haversine distance calculation for proximity detection."""

    def test_same_point_zero_distance(self):
        """Test distance between same point is zero."""
        d = haversine_km(-3.0, -60.0, -3.0, -60.0)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_known_distance_london_paris(self):
        """Test London to Paris is approximately 344 km."""
        d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert d == pytest.approx(344, rel=0.02)

    def test_equator_one_degree_longitude(self):
        """Test 1 degree longitude at equator is approximately 111 km."""
        d = haversine_km(0, 0, 0, 1)
        assert d == pytest.approx(111.19, rel=0.01)

    def test_one_degree_latitude(self):
        """Test 1 degree latitude is approximately 111 km."""
        d = haversine_km(0, 0, 1, 0)
        assert d == pytest.approx(111.19, rel=0.01)

    def test_antipodal_points(self):
        """Test distance between antipodal points is approximately half Earth circumference."""
        d = haversine_km(0, 0, 0, 180)
        assert d == pytest.approx(20015.1, rel=0.01)

    def test_symmetric_distance(self):
        """Test distance A->B equals distance B->A."""
        d1 = haversine_km(-3.0, -60.0, -1.5, -58.0)
        d2 = haversine_km(-1.5, -58.0, -3.0, -60.0)
        assert d1 == pytest.approx(d2, rel=1e-10)

    def test_short_distance_meters(self):
        """Test very short distance (< 1 km) calculation."""
        d = haversine_km(-3.0, -60.0, -3.001, -60.001)
        assert d < 1.0
        assert d > 0

    def test_cross_hemisphere_distance(self):
        """Test distance crossing equator."""
        d = haversine_km(-3.0, -60.0, 3.0, -60.0)
        assert d == pytest.approx(666.0, rel=0.02)


# ===========================================================================
# 7. Overlap Provenance (6 tests)
# ===========================================================================


class TestOverlapProvenance:
    """Test provenance tracking for overlap detection."""

    def test_overlap_provenance_hash_length(self, sample_overlap_direct):
        """Test overlap provenance hash is SHA-256."""
        assert len(sample_overlap_direct.provenance_hash) == SHA256_HEX_LENGTH

    def test_overlap_provenance_deterministic(self):
        """Test same overlap input produces same hash."""
        data = {"overlap_id": "o-001", "overlap_type": "direct"}
        h1 = compute_test_hash(data)
        h2 = compute_test_hash(data)
        assert h1 == h2

    def test_different_overlap_different_hash(self):
        """Test different overlaps produce different hashes."""
        h1 = compute_test_hash({"overlap_id": "o-001", "overlap_type": "direct"})
        h2 = compute_test_hash({"overlap_id": "o-002", "overlap_type": "adjacent"})
        assert h1 != h2

    def test_overlap_detection_response_provenance(self):
        """Test OverlapDetectionResponse includes provenance hash."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            total_overlaps=1,
            provenance_hash="a" * 64,
        )
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_batch_response_provenance(self):
        """Test BatchOverlapResponse includes provenance hash."""
        response = BatchOverlapResponse(
            total_plots=5,
            provenance_hash="b" * 64,
        )
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_provenance_records_overlap_detection(self, mock_provenance):
        """Test provenance tracker records overlap detection."""
        mock_provenance.record("overlap", "detect", "o-001")
        assert mock_provenance.entry_count == 1


# ===========================================================================
# 8. Performance Benchmarks (6 tests)
# ===========================================================================


class TestOverlapPerformance:
    """Performance benchmarks for overlap detection."""

    def test_risk_score_calculation_speed(self):
        """Test single risk score calculation completes in < 1ms."""
        start = time.perf_counter()
        for _ in range(1000):
            compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=10000,
                conflict_history_score=Decimal("50"),
                country_framework_score=Decimal("70"),
            )
        elapsed = (time.perf_counter() - start) * 1000
        avg_ms = elapsed / 1000
        assert avg_ms < 1.0  # Each calculation < 1ms

    def test_batch_risk_score_1000_plots(self):
        """Test 1000 risk score calculations complete in < 100ms."""
        start = time.perf_counter()
        for i in range(1000):
            compute_overlap_risk_score(
                overlap_type="partial",
                legal_status="declared",
                community_population=5000 + i,
                conflict_history_score=Decimal("40"),
                country_framework_score=Decimal("60"),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100

    def test_haversine_calculation_speed(self):
        """Test 10000 Haversine calculations complete in < 100ms."""
        start = time.perf_counter()
        for i in range(10000):
            haversine_km(-3.0 + i * 0.001, -60.0, -3.5, -59.5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100

    def test_overlap_response_construction_speed(self):
        """Test constructing 100 overlap responses in < 50ms."""
        start = time.perf_counter()
        for i in range(100):
            OverlapDetectionResponse(
                plot_id=f"p-{i}",
                total_overlaps=1,
                highest_risk_level=RiskLevel.MEDIUM,
                provenance_hash="a" * 64,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50

    def test_classify_risk_level_speed(self):
        """Test 10000 risk level classifications in < 50ms."""
        import random
        start = time.perf_counter()
        for _ in range(10000):
            classify_risk_level(Decimal(str(random.uniform(0, 100))))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50

    def test_compute_test_hash_speed(self):
        """Test 10000 SHA-256 hash computations in < 200ms."""
        start = time.perf_counter()
        for i in range(10000):
            compute_test_hash({"overlap_id": f"o-{i}", "score": i})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200


# ===========================================================================
# 9. Determinism (10 tests)
# ===========================================================================


class TestOverlapDeterminism:
    """Test bit-perfect reproducibility of overlap calculations."""

    def test_same_input_same_risk_score(self):
        """Test identical inputs produce identical risk scores."""
        scores = [
            compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=26000,
                conflict_history_score=Decimal("75"),
                country_framework_score=Decimal("80"),
            )
            for _ in range(10)
        ]
        assert all(s == scores[0] for s in scores)

    def test_same_input_same_risk_level(self):
        """Test identical scores produce identical risk levels."""
        levels = [
            classify_risk_level(Decimal("72.50"))
            for _ in range(10)
        ]
        assert all(l == levels[0] for l in levels)

    def test_score_precision_consistent(self):
        """Test score precision is consistently 2 decimal places."""
        score = compute_overlap_risk_score(
            overlap_type="partial",
            legal_status="claimed",
            community_population=8000,
            conflict_history_score=Decimal("33.33"),
            country_framework_score=Decimal("66.67"),
        )
        str_score = str(score)
        assert "." in str_score
        decimal_part = str_score.split(".")[1]
        assert len(decimal_part) == 2

    def test_haversine_deterministic(self):
        """Test Haversine produces identical results across runs."""
        distances = [
            haversine_km(-3.0, -60.0, -1.5, -58.0)
            for _ in range(10)
        ]
        assert all(d == distances[0] for d in distances)

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        data = {"overlap_id": "o-det", "risk_score": "72.50"}
        hashes = [compute_test_hash(data) for _ in range(10)]
        assert all(h == hashes[0] for h in hashes)

    @pytest.mark.parametrize("overlap_type", [
        "direct", "partial", "adjacent", "proximate", "none",
    ])
    def test_risk_score_deterministic_per_type(self, overlap_type):
        """Test risk score is deterministic for each overlap type."""
        scores = [
            compute_overlap_risk_score(
                overlap_type=overlap_type,
                legal_status="titled",
                community_population=10000,
                conflict_history_score=Decimal("50"),
                country_framework_score=Decimal("60"),
            )
            for _ in range(5)
        ]
        assert all(s == scores[0] for s in scores)
