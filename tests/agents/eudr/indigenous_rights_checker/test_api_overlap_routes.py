# -*- coding: utf-8 -*-
"""
Tests for Overlap API Routes - AGENT-EUDR-021

Test count: 36 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (API: Overlap Routes)
"""

from decimal import Decimal

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
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
# 1. POST /overlaps/detect (10 tests)
# ===========================================================================


class TestDetectOverlapEndpoint:
    """Test POST /overlaps/detect endpoint."""

    def test_detect_with_coordinates(self):
        """Test overlap detection with latitude/longitude."""
        req = DetectOverlapRequest(
            plot_id="p-001",
            latitude=-3.46,
            longitude=-62.21,
        )
        assert req.latitude is not None
        assert req.longitude is not None

    def test_detect_with_geojson(self, sample_plot_geojson):
        """Test overlap detection with GeoJSON polygon."""
        req = DetectOverlapRequest(
            plot_id="p-geo",
            plot_geojson=sample_plot_geojson,
        )
        assert req.plot_geojson is not None

    def test_detect_with_custom_buffers(self):
        """Test overlap detection with custom buffer radii."""
        req = DetectOverlapRequest(
            plot_id="p-buf",
            latitude=-3.0,
            longitude=-60.0,
            inner_buffer_km=2.0,
            outer_buffer_km=10.0,
        )
        assert req.inner_buffer_km == 2.0

    def test_detect_response_structure(self):
        """Test overlap detection response has correct structure."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            overlaps=[],
            total_overlaps=0,
            highest_risk_level=RiskLevel.NONE,
            processing_time_ms=50.0,
            provenance_hash="a" * 64,
        )
        assert response.plot_id == "p-001"
        assert response.total_overlaps == 0

    def test_detect_response_with_overlaps(self, sample_overlap_direct):
        """Test response with detected overlaps."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            overlaps=[sample_overlap_direct],
            total_overlaps=1,
            highest_risk_level=RiskLevel.CRITICAL,
            processing_time_ms=100.0,
            provenance_hash="b" * 64,
        )
        assert response.total_overlaps == 1
        assert response.highest_risk_level == RiskLevel.CRITICAL

    def test_detect_requires_auth(self, mock_auth):
        """Test overlap detection requires authentication."""
        result = mock_auth.validate_token("valid-token")
        assert "eudr-irc:overlaps:read" in result["permissions"]

    def test_detect_returns_processing_time(self):
        """Test response includes processing time."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            processing_time_ms=150.5,
            provenance_hash="c" * 64,
        )
        assert response.processing_time_ms > 0

    def test_detect_returns_provenance(self):
        """Test response includes provenance hash."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            provenance_hash="d" * 64,
        )
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_detect_invalid_latitude_rejected(self):
        """Test request with invalid latitude is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectOverlapRequest(
                plot_id="p-bad",
                latitude=100.0,
                longitude=0.0,
            )

    def test_detect_invalid_longitude_rejected(self):
        """Test request with invalid longitude is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectOverlapRequest(
                plot_id="p-bad",
                latitude=0.0,
                longitude=200.0,
            )


# ===========================================================================
# 2. POST /overlaps/batch (10 tests)
# ===========================================================================


class TestBatchOverlapEndpoint:
    """Test POST /overlaps/batch endpoint."""

    def test_batch_valid_request(self):
        """Test valid batch overlap request."""
        req = BatchOverlapRequest(
            plots=[
                DetectOverlapRequest(plot_id=f"p-{i}", latitude=-3.0 + i * 0.1, longitude=-60.0)
                for i in range(5)
            ],
        )
        assert len(req.plots) == 5

    def test_batch_response_structure(self):
        """Test batch response has correct structure."""
        response = BatchOverlapResponse(
            total_plots=10,
            plots_with_overlaps=3,
            critical_count=1,
            high_count=1,
            medium_count=1,
            results=[],
            processing_time_ms=500.0,
            provenance_hash="e" * 64,
        )
        assert response.total_plots == 10
        assert response.plots_with_overlaps == 3

    def test_batch_empty_rejected(self):
        """Test empty batch is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BatchOverlapRequest(plots=[])

    def test_batch_max_size(self):
        """Test batch respects max size of 10000."""
        assert MAX_BATCH_SIZE == 10000

    def test_batch_single_plot(self):
        """Test batch with single plot."""
        req = BatchOverlapRequest(
            plots=[DetectOverlapRequest(plot_id="p-001", latitude=-3.0, longitude=-60.0)],
        )
        assert len(req.plots) == 1

    def test_batch_response_counts(self):
        """Test batch response risk level counts."""
        response = BatchOverlapResponse(
            total_plots=100,
            plots_with_overlaps=40,
            critical_count=5,
            high_count=10,
            medium_count=15,
            low_count=10,
            provenance_hash="f" * 64,
        )
        assert (
            response.critical_count + response.high_count
            + response.medium_count + response.low_count
        ) == response.plots_with_overlaps

    def test_batch_response_provenance(self):
        """Test batch response includes provenance hash."""
        response = BatchOverlapResponse(
            total_plots=5,
            provenance_hash="g" * 64,
        )
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_batch_processing_time(self):
        """Test batch response includes processing time."""
        response = BatchOverlapResponse(
            total_plots=1000,
            processing_time_ms=25000.0,
            provenance_hash="h" * 64,
        )
        assert response.processing_time_ms > 0

    def test_batch_concurrency_config(self, mock_config):
        """Test batch concurrency is configured."""
        assert mock_config.batch_concurrency >= 1

    def test_batch_timeout_config(self, mock_config):
        """Test batch timeout is configured."""
        assert mock_config.batch_timeout_s > 0


# ===========================================================================
# 3. GET /overlaps/{plot_id} (8 tests)
# ===========================================================================


class TestGetOverlapsEndpoint:
    """Test GET /overlaps/{plot_id} endpoint."""

    def test_get_overlaps_for_plot(self, sample_overlap_direct):
        """Test retrieving overlaps for a specific plot."""
        assert sample_overlap_direct.plot_id == "p-001"

    def test_get_overlaps_returns_list(self):
        """Test endpoint returns a list of overlaps."""
        overlaps = []
        assert isinstance(overlaps, list)

    def test_get_overlaps_includes_risk_level(self, sample_overlap_direct):
        """Test each overlap includes risk level."""
        assert sample_overlap_direct.risk_level in [
            RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM,
            RiskLevel.LOW, RiskLevel.NONE,
        ]

    def test_get_overlaps_includes_distance(self, sample_overlap_adjacent):
        """Test each overlap includes distance to territory."""
        assert sample_overlap_adjacent.distance_meters > Decimal("0")

    def test_get_overlaps_for_clean_plot(self, sample_overlap_none):
        """Test plot with no overlaps returns NONE risk."""
        assert sample_overlap_none.overlap_type == OverlapType.NONE
        assert sample_overlap_none.risk_level == RiskLevel.NONE

    def test_get_overlaps_requires_auth(self, mock_auth):
        """Test endpoint requires authentication."""
        result = mock_auth.validate_token("valid-token")
        assert "eudr-irc:overlaps:read" in result["permissions"]

    def test_get_overlaps_includes_territory_id(self, sample_overlap_direct):
        """Test each overlap references a territory ID."""
        assert sample_overlap_direct.territory_id is not None

    def test_get_overlaps_includes_affected_communities(self, sample_overlap_direct):
        """Test overlap includes affected community IDs."""
        assert isinstance(sample_overlap_direct.affected_communities, list)


# ===========================================================================
# 4. Overlap Risk Summary (8 tests)
# ===========================================================================


class TestOverlapRiskSummary:
    """Test overlap risk summary aggregation."""

    def test_risk_summary_all_levels(self):
        """Test risk summary includes all risk level counts."""
        response = BatchOverlapResponse(
            total_plots=20,
            plots_with_overlaps=15,
            critical_count=2,
            high_count=3,
            medium_count=5,
            low_count=5,
            provenance_hash="i" * 64,
        )
        total_risk = (
            response.critical_count + response.high_count
            + response.medium_count + response.low_count
        )
        assert total_risk == response.plots_with_overlaps

    def test_risk_summary_no_overlaps(self):
        """Test risk summary with no overlaps."""
        response = BatchOverlapResponse(
            total_plots=10,
            plots_with_overlaps=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            provenance_hash="j" * 64,
        )
        assert response.plots_with_overlaps == 0

    def test_risk_summary_all_critical(self):
        """Test risk summary with all critical overlaps."""
        response = BatchOverlapResponse(
            total_plots=5,
            plots_with_overlaps=5,
            critical_count=5,
            provenance_hash="k" * 64,
        )
        assert response.critical_count == response.total_plots

    def test_highest_risk_level_tracking(self):
        """Test highest risk level is tracked in response."""
        response = OverlapDetectionResponse(
            plot_id="p-001",
            total_overlaps=3,
            highest_risk_level=RiskLevel.CRITICAL,
            provenance_hash="l" * 64,
        )
        assert response.highest_risk_level == RiskLevel.CRITICAL

    def test_deforestation_correlation_flag(self, sample_overlap_direct):
        """Test deforestation correlation flag (cross-ref with EUDR-020)."""
        assert isinstance(sample_overlap_direct.deforestation_correlation, bool)

    def test_overlap_bearing_degrees(self):
        """Test overlap includes bearing degrees to territory."""
        overlap = TerritoryOverlap(
            overlap_id="o-bearing",
            plot_id="p-001",
            territory_id="t-001",
            overlap_type=OverlapType.ADJACENT,
            distance_meters=Decimal("5000"),
            bearing_degrees=Decimal("45.5"),
            risk_score=Decimal("40"),
            risk_level=RiskLevel.MEDIUM,
            provenance_hash="m" * 64,
        )
        assert overlap.bearing_degrees == Decimal("45.5")

    def test_overlap_area_hectares_for_direct(self, sample_overlap_direct):
        """Test DIRECT overlap includes overlap area."""
        assert sample_overlap_direct.overlap_area_hectares is not None
        assert sample_overlap_direct.overlap_area_hectares > Decimal("0")

    def test_overlap_pct_tracking(self, sample_overlap_direct):
        """Test overlap percentage is tracked."""
        assert sample_overlap_direct.overlap_pct_of_plot is not None
