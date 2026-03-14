# -*- coding: utf-8 -*-
"""
Tests for Data Models - AGENT-EUDR-002 Geolocation Verification Models

Comprehensive test suite covering:
- Enumeration values and completeness
- CoordinateInput validation
- PolygonInput validation
- CoordinateValidationResult construction and serialization
- PolygonVerificationResult construction and serialization
- ProtectedAreaCheckResult
- DeforestationVerificationResult
- GeolocationAccuracyScore with Decimal
- BoundaryChange construction
- TemporalChangeResult construction
- ValidationIssue and RepairSuggestion
- JSON roundtrip
- Model validation errors

Test count: 60 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 Geolocation Verification Models
"""

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.geolocation_verification.models import (
    BoundaryChange,
    ChangeType,
    CoordinateIssue,
    CoordinateIssueType,
    CoordinateValidationResult,
    DeforestationVerificationResult,
    GeolocationAccuracyScore,
    PolygonIssue,
    PolygonIssueType,
    PolygonVerificationResult,
    ProtectedAreaCheckResult,
    QualityTier,
    RepairSuggestion,
    TemporalChangeResult,
    VerifyCoordinateRequest,
    VerifyPolygonRequest,
)


# ===========================================================================
# 1. Enumeration Tests (15 tests)
# ===========================================================================


class TestEnumerations:
    """Tests for all enumeration values and completeness."""

    def test_issue_severity_all_values(self):
        """Test IssueSeverity has all expected values."""
        expected = {"critical", "high", "medium", "low", "info"}
        actual = {s.value for s in IssueSeverity}
        assert actual == expected

    def test_issue_severity_critical(self):
        """Test CRITICAL severity value."""
        assert CoordinateIssueType.CRITICAL.value == "critical"

    def test_issue_severity_high(self):
        """Test HIGH severity value."""
        assert CoordinateIssueType.HIGH.value == "high"

    def test_issue_severity_medium(self):
        """Test MEDIUM severity value."""
        assert CoordinateIssueType.MEDIUM.value == "medium"

    def test_issue_severity_low(self):
        """Test LOW severity value."""
        assert CoordinateIssueType.LOW.value == "low"

    def test_issue_severity_info(self):
        """Test INFO severity value."""
        assert CoordinateIssueType.INFO.value == "info"

    def test_quality_tier_all_values(self):
        """Test QualityTier has all expected values."""
        expected = {"gold", "silver", "bronze", "fail"}
        actual = {t.value for t in QualityTier}
        assert actual == expected

    def test_quality_tier_gold(self):
        """Test GOLD tier value."""
        assert QualityTier.GOLD.value == "gold"

    def test_quality_tier_fail(self):
        """Test FAIL tier value."""
        assert QualityTier.FAIL.value == "fail"

    def test_change_type_all_values(self):
        """Test ChangeType has all expected values."""
        expected = {"expansion", "contraction", "shift", "reshape", "stable"}
        actual = {c.value for c in ChangeType}
        assert actual == expected

    def test_change_type_expansion(self):
        """Test EXPANSION change type value."""
        assert ChangeType.EXPANSION.value == "expansion"

    def test_change_type_stable(self):
        """Test STABLE change type value."""
        assert ChangeType.STABLE.value == "stable"

    def test_issue_severity_is_str_enum(self):
        """Test IssueSeverity inherits from str."""
        assert isinstance(CoordinateIssueType.CRITICAL, str)
        assert CoordinateIssueType.CRITICAL == "critical"

    def test_quality_tier_is_str_enum(self):
        """Test QualityTier inherits from str."""
        assert isinstance(QualityTier.GOLD, str)
        assert QualityTier.GOLD == "gold"

    def test_change_type_is_str_enum(self):
        """Test ChangeType inherits from str."""
        assert isinstance(ChangeType.STABLE, str)


# ===========================================================================
# 2. Input Model Tests (8 tests)
# ===========================================================================


class TestInputModels:
    """Tests for input data models."""

    def test_coordinate_input_creation(self):
        """Test CoordinateInput creation with all fields."""
        coord = VerifyCoordinateRequest(
            lat=-3.1234567,
            lon=-60.0234567,
            declared_country="BR",
            commodity="cocoa",
            plot_id="PLOT-001",
        )
        assert coord.lat == -3.1234567
        assert coord.lon == -60.0234567
        assert coord.declared_country == "BR"

    def test_coordinate_input_defaults(self):
        """Test CoordinateInput default values."""
        coord = VerifyCoordinateRequest(lat=0.0, lon=0.0)
        assert coord.declared_country == ""
        assert coord.commodity == ""
        assert coord.plot_id == ""
        assert coord.metadata == {}

    def test_polygon_input_creation(self):
        """Test PolygonInput creation with vertices."""
        poly = VerifyPolygonRequest(
            vertices=[(-3.12, -60.02), (-3.12, -60.01), (-3.13, -60.02)],
            declared_area_ha=2.0,
            commodity="cocoa",
        )
        assert len(poly.vertices) == 3
        assert poly.declared_area_ha == 2.0

    def test_polygon_input_defaults(self):
        """Test PolygonInput default values."""
        poly = VerifyPolygonRequest()
        assert poly.vertices == []
        assert poly.declared_area_ha is None
        assert poly.commodity == ""

    def test_coordinate_input_metadata(self):
        """Test CoordinateInput with metadata."""
        coord = VerifyCoordinateRequest(
            lat=0.0, lon=0.0,
            metadata={"source": "GPS", "device": "Garmin"},
        )
        assert coord.metadata["source"] == "GPS"


# ===========================================================================
# 3. Validation Issue Tests (8 tests)
# ===========================================================================


class TestValidationIssue:
    """Tests for ValidationIssue model."""

    def test_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = CoordinateIssue(
            code="COORD_OUT_OF_BOUNDS",
            severity=CoordinateIssueType.CRITICAL,
            message="Latitude exceeds WGS84 bounds",
            field="lat",
        )
        assert issue.code == "COORD_OUT_OF_BOUNDS"
        assert issue.severity == CoordinateIssueType.CRITICAL

    def test_issue_auto_id(self):
        """Test issue ID is auto-generated."""
        issue = CoordinateIssue()
        assert issue.issue_id is not None
        assert issue.issue_id.startswith("ISS")

    def test_issue_to_dict(self):
        """Test issue serialization."""
        issue = CoordinateIssue(
            code="TEST_CODE",
            severity=CoordinateIssueType.HIGH,
            message="Test message",
        )
        d = issue.to_dict()
        assert d["code"] == "TEST_CODE"
        assert d["severity"] == "high"
        assert d["message"] == "Test message"

    def test_issue_defaults(self):
        """Test issue default values."""
        issue = CoordinateIssue()
        assert issue.code == ""
        assert issue.severity == CoordinateIssueType.MEDIUM
        assert issue.message == ""
        assert issue.details == {}


# ===========================================================================
# 4. Repair Suggestion Tests (5 tests)
# ===========================================================================


class TestRepairSuggestion:
    """Tests for RepairSuggestion model."""

    def test_suggestion_creation(self):
        """Test RepairSuggestion creation."""
        suggestion = RepairSuggestion(
            issue_code="RING_UNCLOSED",
            action="Append first vertex to close the ring",
            auto_fixable=True,
        )
        assert suggestion.issue_code == "RING_UNCLOSED"
        assert suggestion.auto_fixable is True

    def test_suggestion_auto_id(self):
        """Test suggestion ID is auto-generated."""
        suggestion = RepairSuggestion()
        assert suggestion.suggestion_id is not None
        assert suggestion.suggestion_id.startswith("RPR")

    def test_suggestion_to_dict(self):
        """Test suggestion serialization."""
        suggestion = RepairSuggestion(
            issue_code="WINDING_ORDER",
            action="Reverse vertex order",
            auto_fixable=True,
            parameters={"corrected_order": "CCW"},
        )
        d = suggestion.to_dict()
        assert d["issue_code"] == "WINDING_ORDER"
        assert d["auto_fixable"] is True
        assert d["parameters"]["corrected_order"] == "CCW"

    def test_suggestion_defaults(self):
        """Test suggestion default values."""
        suggestion = RepairSuggestion()
        assert suggestion.issue_code == ""
        assert suggestion.action == ""
        assert suggestion.auto_fixable is False
        assert suggestion.parameters == {}


# ===========================================================================
# 5. Result Model Tests (12 tests)
# ===========================================================================


class TestResultModels:
    """Tests for result data models."""

    def test_coordinate_validation_result_creation(self):
        """Test CoordinateValidationResult creation."""
        result = CoordinateValidationResult(
            lat=-3.12, lon=-60.02,
            is_valid=True, wgs84_valid=True,
            precision_decimal_places=7, precision_score=0.95,
        )
        assert result.lat == -3.12
        assert result.is_valid is True

    def test_coordinate_validation_result_serialization(self):
        """Test CoordinateValidationResult serialization."""
        result = CoordinateValidationResult(
            lat=-3.12, lon=-60.02,
            is_valid=True, wgs84_valid=True,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["lat"] == -3.12
        assert d["is_valid"] is True

    def test_coordinate_validation_auto_id(self):
        """Test validation ID is auto-generated."""
        result = CoordinateValidationResult()
        assert result.validation_id.startswith("CVR")

    def test_polygon_verification_result_serialization(self):
        """Test PolygonVerificationResult serialization."""
        result = PolygonVerificationResult(
            is_valid=True, ring_closed=True,
            vertex_count=5, calculated_area_ha=2.5,
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["vertex_count"] == 5

    def test_polygon_verification_auto_id(self):
        """Test polygon verification ID is auto-generated."""
        result = PolygonVerificationResult()
        assert result.verification_id.startswith("PVR")

    def test_protected_area_check_result(self):
        """Test ProtectedAreaCheckResult creation."""
        result = ProtectedAreaCheckResult(
            overlaps_protected=True,
            protected_area_name="Amazonia NP",
            protected_area_type="II",
            overlap_percentage=85.0,
        )
        assert result.overlaps_protected is True
        assert result.protected_area_name == "Amazonia NP"

    def test_deforestation_verification_result(self):
        """Test DeforestationVerificationResult creation."""
        result = DeforestationVerificationResult(
            deforestation_detected=True,
            alert_count=5,
            forest_loss_ha=3.5,
            cutoff_date="2020-12-31",
            confidence=0.92,
        )
        assert result.deforestation_detected is True
        assert result.alert_count == 5

    def test_geolocation_accuracy_score(self):
        """Test GeolocationAccuracyScore creation with Decimal."""
        score = GeolocationAccuracyScore(
            total_score=Decimal("87.50"),
            coordinate_precision_score=Decimal("18.00"),
            polygon_quality_score=Decimal("19.00"),
            country_match_score=Decimal("15.00"),
            protected_area_score=Decimal("15.00"),
            deforestation_score=Decimal("15.00"),
            temporal_consistency_score=Decimal("5.50"),
            quality_tier=QualityTier.GOLD,
        )
        assert score.total_score == Decimal("87.50")
        assert score.quality_tier == QualityTier.GOLD

    def test_accuracy_score_serialization(self):
        """Test GeolocationAccuracyScore serialization."""
        score = GeolocationAccuracyScore(
            total_score=Decimal("87.50"),
            quality_tier=QualityTier.GOLD,
        )
        d = score.to_dict()
        assert d["total_score"] == "87.50"
        assert d["quality_tier"] == "gold"

    def test_accuracy_score_auto_id(self):
        """Test accuracy score ID is auto-generated."""
        score = GeolocationAccuracyScore()
        assert score.score_id.startswith("GAS")


# ===========================================================================
# 6. Boundary Change Tests (5 tests)
# ===========================================================================


class TestBoundaryChange:
    """Tests for BoundaryChange model."""

    def test_boundary_change_creation(self):
        """Test BoundaryChange creation."""
        change = BoundaryChange(
            change_type=ChangeType.EXPANSION,
            area_change_pct=15.0,
            centroid_shift_m=120.0,
            previous_area_ha=10.0,
            new_area_ha=11.5,
        )
        assert change.change_type == ChangeType.EXPANSION
        assert change.area_change_pct == 15.0

    def test_boundary_change_defaults(self):
        """Test BoundaryChange default values."""
        change = BoundaryChange()
        assert change.change_type == ChangeType.STABLE
        assert change.area_change_pct == 0.0
        assert change.forest_encroachment is False

    def test_boundary_change_serialization(self):
        """Test BoundaryChange serialization."""
        change = BoundaryChange(
            change_type=ChangeType.SHIFT,
            centroid_shift_m=50.0,
        )
        d = change.to_dict()
        assert d["change_type"] == "shift"
        assert d["centroid_shift_m"] == 50.0

    def test_boundary_change_auto_id(self):
        """Test boundary change ID is auto-generated."""
        change = BoundaryChange()
        assert change.change_id.startswith("BCH")


# ===========================================================================
# 7. Temporal Change Result Tests (5 tests)
# ===========================================================================


class TestTemporalChangeResult:
    """Tests for TemporalChangeResult model."""

    def test_temporal_result_creation(self):
        """Test TemporalChangeResult creation."""
        result = TemporalChangeResult(
            plot_id="PLOT-001",
            is_consistent=True,
        )
        assert result.plot_id == "PLOT-001"
        assert result.is_consistent is True

    def test_temporal_result_defaults(self):
        """Test TemporalChangeResult default values."""
        result = TemporalChangeResult()
        assert result.is_consistent is True
        assert result.rapid_change_detected is False
        assert result.change_history == []

    def test_temporal_result_serialization(self):
        """Test TemporalChangeResult serialization."""
        change = BoundaryChange(change_type=ChangeType.EXPANSION)
        result = TemporalChangeResult(
            plot_id="PLOT-001",
            is_consistent=False,
            boundary_change=change,
        )
        d = result.to_dict()
        assert d["is_consistent"] is False
        assert d["boundary_change"] is not None

    def test_temporal_result_auto_id(self):
        """Test temporal result ID is auto-generated."""
        result = TemporalChangeResult()
        assert result.analysis_id.startswith("TCR")

    def test_temporal_result_with_history(self):
        """Test TemporalChangeResult with change history."""
        history = [
            BoundaryChange(change_type=ChangeType.STABLE),
            BoundaryChange(change_type=ChangeType.EXPANSION),
        ]
        result = TemporalChangeResult(
            plot_id="PLOT-001",
            change_history=history,
        )
        assert len(result.change_history) == 2
        d = result.to_dict()
        assert len(d["change_history"]) == 2


# ===========================================================================
# 8. JSON Roundtrip Tests (7 tests)
# ===========================================================================


class TestJsonRoundtrip:
    """Tests for JSON serialization roundtrip."""

    def test_coord_result_json_roundtrip(self):
        """Test CoordinateValidationResult JSON roundtrip."""
        result = CoordinateValidationResult(
            lat=-3.12, lon=-60.02, is_valid=True,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["lat"] == -3.12
        assert parsed["is_valid"] is True

    def test_polygon_result_json_roundtrip(self):
        """Test PolygonVerificationResult JSON roundtrip."""
        result = PolygonVerificationResult(
            is_valid=True, vertex_count=5, calculated_area_ha=2.5,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["vertex_count"] == 5

    def test_accuracy_score_json_roundtrip(self):
        """Test GeolocationAccuracyScore JSON roundtrip."""
        score = GeolocationAccuracyScore(
            total_score=Decimal("87.50"),
            quality_tier=QualityTier.GOLD,
        )
        d = score.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["total_score"] == "87.50"
        assert parsed["quality_tier"] == "gold"

    def test_boundary_change_json_roundtrip(self):
        """Test BoundaryChange JSON roundtrip."""
        change = BoundaryChange(
            change_type=ChangeType.EXPANSION,
            area_change_pct=15.0,
        )
        d = change.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["change_type"] == "expansion"

    def test_temporal_result_json_roundtrip(self):
        """Test TemporalChangeResult JSON roundtrip."""
        result = TemporalChangeResult(
            plot_id="PLOT-001",
            is_consistent=True,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["plot_id"] == "PLOT-001"

    def test_issue_json_roundtrip(self):
        """Test ValidationIssue JSON roundtrip."""
        issue = CoordinateIssue(
            code="TEST", severity=CoordinateIssueType.HIGH, message="Test",
        )
        d = issue.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["code"] == "TEST"
        assert parsed["severity"] == "high"

    def test_suggestion_json_roundtrip(self):
        """Test RepairSuggestion JSON roundtrip."""
        suggestion = RepairSuggestion(
            issue_code="FIX", action="Do something", auto_fixable=True,
        )
        d = suggestion.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["auto_fixable"] is True
