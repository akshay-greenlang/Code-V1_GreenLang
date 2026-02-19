# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 Land Use Change Tracker Engine.

Tests record_transition() for all 36 possible transitions, transition
matrix generation, deforestation/afforestation/wetland change detection,
peatland conversion tracking, area consistency validation, transition
history retrieval, 20-year transition period tracking, transition age
calculation, portfolio summary, reversal detection, annual rates,
cumulative area tracking, and edge cases.

Target: 120 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.land_use_emissions.land_use_change_tracker import (
    LAND_CATEGORIES,
    DEFAULT_TRANSITION_PERIOD,
    TRANSITION_REMAINING,
    TRANSITION_CONVERSION,
    TransitionRecord,
    LandUseChangeTrackerEngine,
    _D,
    _ZERO,
    _ONE,
    _PRECISION,
    _compute_hash,
    _utcnow,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tracker() -> LandUseChangeTrackerEngine:
    """Create a fresh LandUseChangeTrackerEngine instance."""
    return LandUseChangeTrackerEngine()


@pytest.fixture
def base_request() -> dict:
    """Create a minimal valid transition request."""
    return {
        "parcel_id": "P001",
        "from_category": "FOREST_LAND",
        "to_category": "CROPLAND",
        "area_ha": 50.0,
        "transition_date": "2023-01-15",
    }


@pytest.fixture
def tracker_with_transitions(tracker) -> LandUseChangeTrackerEngine:
    """Return a tracker pre-loaded with several transitions for testing queries."""
    transitions = [
        # Deforestation events
        {"parcel_id": "P001", "from_category": "FOREST_LAND", "to_category": "CROPLAND",
         "area_ha": 50.0, "transition_date": "2022-03-10"},
        {"parcel_id": "P002", "from_category": "FOREST_LAND", "to_category": "GRASSLAND",
         "area_ha": 30.0, "transition_date": "2022-06-15"},
        {"parcel_id": "P003", "from_category": "FOREST_LAND", "to_category": "SETTLEMENTS",
         "area_ha": 10.0, "transition_date": "2023-02-01"},
        # Afforestation events
        {"parcel_id": "P004", "from_category": "CROPLAND", "to_category": "FOREST_LAND",
         "area_ha": 25.0, "transition_date": "2022-09-20"},
        {"parcel_id": "P005", "from_category": "GRASSLAND", "to_category": "FOREST_LAND",
         "area_ha": 15.0, "transition_date": "2023-05-01"},
        # Wetland drainage
        {"parcel_id": "P006", "from_category": "WETLANDS", "to_category": "CROPLAND",
         "area_ha": 20.0, "transition_date": "2022-04-01"},
        # Rewetting
        {"parcel_id": "P007", "from_category": "CROPLAND", "to_category": "WETLANDS",
         "area_ha": 8.0, "transition_date": "2023-07-01"},
        # Remaining
        {"parcel_id": "P008", "from_category": "FOREST_LAND", "to_category": "FOREST_LAND",
         "area_ha": 200.0, "transition_date": "2022-01-01"},
        # Other land conversions
        {"parcel_id": "P009", "from_category": "GRASSLAND", "to_category": "SETTLEMENTS",
         "area_ha": 5.0, "transition_date": "2023-11-01"},
        {"parcel_id": "P010", "from_category": "OTHER_LAND", "to_category": "CROPLAND",
         "area_ha": 12.0, "transition_date": "2023-01-15"},
    ]
    for t in transitions:
        tracker.record_transition(t)
    return tracker


# ===========================================================================
# Module Constants Tests
# ===========================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_land_categories_has_six_entries(self):
        """LAND_CATEGORIES has exactly six IPCC categories."""
        assert len(LAND_CATEGORIES) == 6

    def test_land_categories_content(self):
        """LAND_CATEGORIES contains the six IPCC categories."""
        expected = [
            "FOREST_LAND", "CROPLAND", "GRASSLAND",
            "WETLANDS", "SETTLEMENTS", "OTHER_LAND",
        ]
        assert LAND_CATEGORIES == expected

    def test_default_transition_period(self):
        """DEFAULT_TRANSITION_PERIOD is 20 years per IPCC Tier 1."""
        assert DEFAULT_TRANSITION_PERIOD == 20

    def test_transition_remaining_constant(self):
        """TRANSITION_REMAINING is 'REMAINING'."""
        assert TRANSITION_REMAINING == "REMAINING"

    def test_transition_conversion_constant(self):
        """TRANSITION_CONVERSION is 'CONVERSION'."""
        assert TRANSITION_CONVERSION == "CONVERSION"


# ===========================================================================
# Decimal Helper Tests
# ===========================================================================


class TestDecimalHelpers:
    """Tests for _D, _ZERO, _ONE, _PRECISION helpers."""

    def test_d_from_int(self):
        """_D converts integer to Decimal."""
        assert _D(42) == Decimal("42")

    def test_d_from_float(self):
        """_D converts float to Decimal via str to avoid float imprecision."""
        result = _D(0.1)
        assert result == Decimal("0.1")

    def test_d_from_string(self):
        """_D converts string to Decimal."""
        assert _D("123.456") == Decimal("123.456")

    def test_d_from_decimal_passthrough(self):
        """_D returns Decimal unchanged if already a Decimal."""
        d = Decimal("99.99")
        assert _D(d) is d

    def test_zero_constant(self):
        """_ZERO equals Decimal('0')."""
        assert _ZERO == Decimal("0")

    def test_one_constant(self):
        """_ONE equals Decimal('1')."""
        assert _ONE == Decimal("1")

    def test_precision_constant(self):
        """_PRECISION is 8 decimal places."""
        assert _PRECISION == Decimal("0.00000001")


# ===========================================================================
# _compute_hash Tests
# ===========================================================================


class TestComputeHash:
    """Tests for the _compute_hash helper."""

    def test_returns_64_char_hex(self):
        """_compute_hash returns a 64-character hex SHA-256 digest."""
        h = _compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        """Same input produces the same hash."""
        data = {"a": 1, "b": "two"}
        assert _compute_hash(data) == _compute_hash(data)

    def test_different_input_different_hash(self):
        """Different inputs produce different hashes."""
        h1 = _compute_hash({"x": 1})
        h2 = _compute_hash({"x": 2})
        assert h1 != h2

    def test_key_order_independent(self):
        """Dict key ordering does not affect hash (sort_keys=True)."""
        h1 = _compute_hash({"b": 2, "a": 1})
        h2 = _compute_hash({"a": 1, "b": 2})
        assert h1 == h2


# ===========================================================================
# TransitionRecord Tests
# ===========================================================================


class TestTransitionRecord:
    """Tests for the TransitionRecord dataclass."""

    def test_creation(self):
        """TransitionRecord can be created with all required fields."""
        rec = TransitionRecord(
            transition_id="T001",
            parcel_id="P001",
            from_category="FOREST_LAND",
            to_category="CROPLAND",
            area_ha=Decimal("50"),
            transition_date="2023-01-15",
            transition_type="CONVERSION",
            transition_period_years=20,
            completion_date="2043-01-15",
            is_deforestation=True,
            is_afforestation=False,
            is_wetland_drainage=False,
            is_peatland_conversion=False,
            notes="",
            provenance_hash="a" * 64,
            recorded_at="2023-01-15T00:00:00",
        )
        assert rec.transition_id == "T001"
        assert rec.is_deforestation is True
        assert rec.area_ha == Decimal("50")

    def test_to_dict_keys(self):
        """to_dict includes all expected keys."""
        rec = TransitionRecord(
            transition_id="T001", parcel_id="P001",
            from_category="FOREST_LAND", to_category="CROPLAND",
            area_ha=Decimal("50"), transition_date="2023-01-15",
            transition_type="CONVERSION", transition_period_years=20,
            completion_date="2043-01-15", is_deforestation=True,
            is_afforestation=False, is_wetland_drainage=False,
            is_peatland_conversion=False, notes="test",
            provenance_hash="b" * 64, recorded_at="2023-01-15T00:00:00",
        )
        d = rec.to_dict()
        assert "transition_id" in d
        assert "parcel_id" in d
        assert "from_category" in d
        assert "to_category" in d
        assert "area_ha" in d
        assert "is_deforestation" in d
        assert "provenance_hash" in d
        assert "notes" in d

    def test_to_dict_area_is_string(self):
        """to_dict converts area_ha Decimal to string."""
        rec = TransitionRecord(
            transition_id="T001", parcel_id="P001",
            from_category="FOREST_LAND", to_category="CROPLAND",
            area_ha=Decimal("123.456"), transition_date="2023-01-15",
            transition_type="CONVERSION", transition_period_years=20,
            completion_date="2043-01-15", is_deforestation=True,
            is_afforestation=False, is_wetland_drainage=False,
            is_peatland_conversion=False, notes="",
            provenance_hash="c" * 64, recorded_at="2023-01-15T00:00:00",
        )
        d = rec.to_dict()
        assert d["area_ha"] == "123.456"
        assert isinstance(d["area_ha"], str)


# ===========================================================================
# Engine Initialization Tests
# ===========================================================================


class TestEngineInit:
    """Tests for LandUseChangeTrackerEngine initialization."""

    def test_init_creates_6x6_matrix(self, tracker):
        """Initialization creates a 6x6 zero matrix."""
        matrix_result = tracker.get_transition_matrix()
        m = matrix_result["matrix"]
        assert len(m) == 6
        for from_cat in LAND_CATEGORIES:
            assert len(m[from_cat]) == 6
            for to_cat in LAND_CATEGORIES:
                assert m[from_cat][to_cat] == "0"

    def test_init_empty_history(self, tracker):
        """Initialization starts with empty history."""
        hist = tracker.get_transition_history()
        assert hist["total_count"] == 0
        assert hist["transitions"] == []

    def test_init_zero_transitions(self, tracker):
        """Statistics show zero transitions at init."""
        stats = tracker.get_statistics()
        assert stats["total_transitions"] == 0
        assert stats["total_parcels"] == 0

    def test_init_engine_name(self, tracker):
        """Statistics report correct engine name."""
        stats = tracker.get_statistics()
        assert stats["engine"] == "LandUseChangeTrackerEngine"
        assert stats["version"] == "1.0.0"

    def test_init_categories_in_stats(self, tracker):
        """Statistics include all six categories."""
        stats = tracker.get_statistics()
        assert stats["categories"] == LAND_CATEGORIES


# ===========================================================================
# Record Transition - Basic Tests
# ===========================================================================


class TestRecordTransitionBasic:
    """Tests for record_transition() basic functionality."""

    def test_success_status(self, tracker, base_request):
        """Successful transition returns status SUCCESS."""
        result = tracker.record_transition(base_request)
        assert result["status"] == "SUCCESS"

    def test_returns_transition_id(self, tracker, base_request):
        """Result includes a UUID transition_id."""
        result = tracker.record_transition(base_request)
        assert "transition_id" in result
        assert len(result["transition_id"]) == 36  # UUID format

    def test_returns_provenance_hash(self, tracker, base_request):
        """Result includes a 64-char SHA-256 provenance hash."""
        result = tracker.record_transition(base_request)
        assert len(result["provenance_hash"]) == 64

    def test_returns_processing_time(self, tracker, base_request):
        """Result includes processing_time_ms."""
        result = tracker.record_transition(base_request)
        assert result["processing_time_ms"] >= 0

    def test_increments_transition_count(self, tracker, base_request):
        """Recording a transition increments the total count."""
        tracker.record_transition(base_request)
        stats = tracker.get_statistics()
        assert stats["total_transitions"] == 1

    def test_updates_parcel_category(self, tracker, base_request):
        """Recording a transition updates parcel current category."""
        tracker.record_transition(base_request)
        stats = tracker.get_statistics()
        assert stats["total_parcels"] == 1

    def test_area_normalised_to_decimal(self, tracker, base_request):
        """area_ha is stored as a Decimal string."""
        result = tracker.record_transition(base_request)
        # area_ha is returned as a string in to_dict
        assert result["area_ha"] == "50.0"

    def test_notes_stored(self, tracker, base_request):
        """Optional notes field is stored."""
        base_request["notes"] = "Cleared for agriculture"
        result = tracker.record_transition(base_request)
        assert result["notes"] == "Cleared for agriculture"

    def test_custom_transition_period(self, tracker, base_request):
        """Custom transition_period_years overrides the 20-year default."""
        base_request["transition_period_years"] = 10
        result = tracker.record_transition(base_request)
        assert result["transition_period_years"] == 10


# ===========================================================================
# Record Transition - Validation Error Tests
# ===========================================================================


class TestRecordTransitionValidation:
    """Tests for record_transition() validation logic."""

    def test_missing_parcel_id(self, tracker):
        """Empty parcel_id returns VALIDATION_ERROR."""
        result = tracker.record_transition({
            "parcel_id": "",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert any("parcel_id" in e for e in result["errors"])

    def test_invalid_from_category(self, tracker):
        """Invalid from_category returns VALIDATION_ERROR."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "OCEAN",
            "to_category": "CROPLAND",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert any("Unknown land category" in e for e in result["errors"])

    def test_invalid_to_category(self, tracker):
        """Invalid to_category returns VALIDATION_ERROR."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "DESERT",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "VALIDATION_ERROR"

    def test_zero_area(self, tracker):
        """Zero area returns VALIDATION_ERROR."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 0,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert any("area_ha" in e for e in result["errors"])

    def test_negative_area(self, tracker):
        """Negative area returns VALIDATION_ERROR."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": -5,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "VALIDATION_ERROR"

    def test_invalid_date_format(self, tracker):
        """Unparseable date returns VALIDATION_ERROR."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 10,
            "transition_date": "not-a-date",
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert any("Cannot parse date" in e for e in result["errors"])

    def test_multiple_validation_errors(self, tracker):
        """Multiple invalid fields accumulate errors."""
        result = tracker.record_transition({
            "parcel_id": "",
            "from_category": "INVALID",
            "to_category": "ALSO_INVALID",
            "area_ha": -1,
            "transition_date": "bad",
        })
        assert result["status"] == "VALIDATION_ERROR"
        assert len(result["errors"]) >= 3

    def test_slash_date_format_accepted(self, tracker):
        """YYYY/MM/DD date format is also accepted."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 10,
            "transition_date": "2023/01/15",
        })
        assert result["status"] == "SUCCESS"
        assert result["transition_date"] == "2023-01-15"


# ===========================================================================
# Transition Classification Tests
# ===========================================================================


class TestTransitionClassification:
    """Tests for REMAINING vs CONVERSION classification."""

    def test_same_category_is_remaining(self, tracker):
        """Same from/to category is classified as REMAINING."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "FOREST_LAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        assert result["transition_type"] == TRANSITION_REMAINING

    def test_different_category_is_conversion(self, tracker, base_request):
        """Different from/to category is classified as CONVERSION."""
        result = tracker.record_transition(base_request)
        assert result["transition_type"] == TRANSITION_CONVERSION

    @pytest.mark.parametrize("cat", LAND_CATEGORIES)
    def test_all_remaining_transitions(self, tracker, cat):
        """Every diagonal (same-category) transition is REMAINING."""
        result = tracker.record_transition({
            "parcel_id": f"P_{cat}",
            "from_category": cat,
            "to_category": cat,
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["transition_type"] == TRANSITION_REMAINING
        assert result["is_deforestation"] is False
        assert result["is_afforestation"] is False

    def test_category_normalisation_case_insensitive(self, tracker):
        """Categories are normalised to uppercase."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "forest_land",
            "to_category": "cropland",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "SUCCESS"
        assert result["from_category"] == "FOREST_LAND"
        assert result["to_category"] == "CROPLAND"


# ===========================================================================
# Deforestation Detection Tests
# ===========================================================================


class TestDeforestationDetection:
    """Tests for deforestation classification and detection."""

    @pytest.mark.parametrize("to_cat", [
        "CROPLAND", "GRASSLAND", "WETLANDS", "SETTLEMENTS", "OTHER_LAND",
    ])
    def test_forest_to_other_is_deforestation(self, tracker, to_cat):
        """Any conversion FROM FOREST_LAND to another category is deforestation."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": to_cat,
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["is_deforestation"] is True

    def test_forest_to_forest_not_deforestation(self, tracker):
        """FOREST_LAND remaining is not deforestation."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "FOREST_LAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        assert result["is_deforestation"] is False

    def test_cropland_to_grassland_not_deforestation(self, tracker):
        """Non-forest conversion is not deforestation."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "CROPLAND",
            "to_category": "GRASSLAND",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["is_deforestation"] is False

    def test_detect_deforestation_method(self, tracker_with_transitions):
        """detect_deforestation returns all deforestation events."""
        result = tracker_with_transitions.detect_deforestation()
        assert result["event_count"] == 3  # FL->CL, FL->GL, FL->SL
        assert Decimal(result["total_deforestation_ha"]) == Decimal("90")

    def test_detect_deforestation_by_target(self, tracker_with_transitions):
        """detect_deforestation groups by target category."""
        result = tracker_with_transitions.detect_deforestation()
        by_target = result["by_target_category"]
        assert Decimal(by_target["CROPLAND"]) == Decimal("50")
        assert Decimal(by_target["GRASSLAND"]) == Decimal("30")
        assert Decimal(by_target["SETTLEMENTS"]) == Decimal("10")

    def test_detect_deforestation_date_filter(self, tracker_with_transitions):
        """detect_deforestation respects date filters."""
        result = tracker_with_transitions.detect_deforestation(
            start_date="2023-01-01",
        )
        # Only FL->SL on 2023-02-01
        assert result["event_count"] == 1
        assert Decimal(result["total_deforestation_ha"]) == Decimal("10")

    def test_detect_deforestation_parcel_filter(self, tracker_with_transitions):
        """detect_deforestation respects parcel_id filter."""
        result = tracker_with_transitions.detect_deforestation(parcel_id="P001")
        assert result["event_count"] == 1
        assert Decimal(result["total_deforestation_ha"]) == Decimal("50")

    def test_detect_deforestation_provenance_hash(self, tracker_with_transitions):
        """detect_deforestation includes a provenance hash."""
        result = tracker_with_transitions.detect_deforestation()
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Afforestation Detection Tests
# ===========================================================================


class TestAfforestationDetection:
    """Tests for afforestation/reforestation classification and detection."""

    @pytest.mark.parametrize("from_cat", [
        "CROPLAND", "GRASSLAND", "WETLANDS", "SETTLEMENTS", "OTHER_LAND",
    ])
    def test_other_to_forest_is_afforestation(self, tracker, from_cat):
        """Any conversion TO FOREST_LAND is afforestation."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": from_cat,
            "to_category": "FOREST_LAND",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["is_afforestation"] is True

    def test_detect_afforestation_method(self, tracker_with_transitions):
        """detect_afforestation returns all afforestation events."""
        result = tracker_with_transitions.detect_afforestation()
        assert result["event_count"] == 2  # CL->FL, GL->FL
        assert Decimal(result["total_afforestation_ha"]) == Decimal("40")

    def test_detect_afforestation_by_source(self, tracker_with_transitions):
        """detect_afforestation groups by source category."""
        result = tracker_with_transitions.detect_afforestation()
        by_source = result["by_source_category"]
        assert Decimal(by_source["CROPLAND"]) == Decimal("25")
        assert Decimal(by_source["GRASSLAND"]) == Decimal("15")

    def test_detect_afforestation_date_filter(self, tracker_with_transitions):
        """detect_afforestation respects date filters."""
        result = tracker_with_transitions.detect_afforestation(
            start_date="2023-01-01", end_date="2023-12-31",
        )
        assert result["event_count"] == 1  # GL->FL on 2023-05-01
        assert Decimal(result["total_afforestation_ha"]) == Decimal("15")


# ===========================================================================
# Wetland Change Detection Tests
# ===========================================================================


class TestWetlandChangeDetection:
    """Tests for wetland drainage, rewetting, and peatland conversion."""

    def test_wetland_to_cropland_is_drainage(self, tracker):
        """WETLANDS -> CROPLAND is classified as wetland drainage."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "WETLANDS",
            "to_category": "CROPLAND",
            "area_ha": 20,
            "transition_date": "2023-01-01",
        })
        assert result["is_wetland_drainage"] is True

    @pytest.mark.parametrize("to_cat", [
        "CROPLAND", "GRASSLAND", "FOREST_LAND", "SETTLEMENTS", "OTHER_LAND",
    ])
    def test_wetland_to_any_non_wetland_is_drainage(self, tracker, to_cat):
        """Any conversion FROM WETLANDS is wetland drainage."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "WETLANDS",
            "to_category": to_cat,
            "area_ha": 5,
            "transition_date": "2023-01-01",
        })
        assert result["is_wetland_drainage"] is True

    def test_wetland_remaining_not_drainage(self, tracker):
        """WETLANDS remaining is not drainage."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "WETLANDS",
            "to_category": "WETLANDS",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        assert result["is_wetland_drainage"] is False

    @pytest.mark.parametrize("to_cat", [
        "CROPLAND", "GRASSLAND", "SETTLEMENTS",
    ])
    def test_peatland_conversion(self, tracker, to_cat):
        """WETLANDS to CROPLAND/GRASSLAND/SETTLEMENTS is peatland conversion."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "WETLANDS",
            "to_category": to_cat,
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["is_peatland_conversion"] is True

    def test_wetland_to_forest_not_peatland_conversion(self, tracker):
        """WETLANDS to FOREST_LAND is drainage but not peatland conversion."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "WETLANDS",
            "to_category": "FOREST_LAND",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["is_wetland_drainage"] is True
        assert result["is_peatland_conversion"] is False

    def test_wetland_to_other_land_not_peatland_conversion(self, tracker):
        """WETLANDS to OTHER_LAND is drainage but not peatland conversion."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "WETLANDS",
            "to_category": "OTHER_LAND",
            "area_ha": 10,
            "transition_date": "2023-01-01",
        })
        assert result["is_wetland_drainage"] is True
        assert result["is_peatland_conversion"] is False

    def test_detect_wetland_changes(self, tracker_with_transitions):
        """detect_wetland_changes returns drainage, rewetting, and peatland stats."""
        result = tracker_with_transitions.detect_wetland_changes()
        assert result["drainage_count"] == 1
        assert Decimal(result["total_drainage_ha"]) == Decimal("20")
        assert result["rewetting_count"] == 1
        assert Decimal(result["total_rewetting_ha"]) == Decimal("8")
        assert result["peatland_conversion_count"] == 1
        assert Decimal(result["total_peatland_conversion_ha"]) == Decimal("20")

    def test_net_wetland_change(self, tracker_with_transitions):
        """Net wetland change = rewetting - drainage."""
        result = tracker_with_transitions.detect_wetland_changes()
        net = Decimal(result["net_wetland_change_ha"])
        assert net == Decimal("8") - Decimal("20")  # -12

    def test_detect_wetland_changes_provenance(self, tracker_with_transitions):
        """detect_wetland_changes includes a provenance hash."""
        result = tracker_with_transitions.detect_wetland_changes()
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# All 36 Transition Combinations Test
# ===========================================================================


class TestAll36Transitions:
    """Tests that all 6x6 = 36 possible transitions are accepted."""

    @pytest.mark.parametrize("from_cat", LAND_CATEGORIES)
    @pytest.mark.parametrize("to_cat", LAND_CATEGORIES)
    def test_transition_accepted(self, tracker, from_cat, to_cat):
        """Every combination of from/to category is accepted."""
        result = tracker.record_transition({
            "parcel_id": f"P_{from_cat}_{to_cat}",
            "from_category": from_cat,
            "to_category": to_cat,
            "area_ha": 1.0,
            "transition_date": "2023-06-15",
        })
        assert result["status"] == "SUCCESS"
        assert result["from_category"] == from_cat
        assert result["to_category"] == to_cat
        if from_cat == to_cat:
            assert result["transition_type"] == TRANSITION_REMAINING
        else:
            assert result["transition_type"] == TRANSITION_CONVERSION


# ===========================================================================
# Transition Matrix Tests
# ===========================================================================


class TestTransitionMatrix:
    """Tests for get_transition_matrix()."""

    def test_empty_matrix(self, tracker):
        """Empty tracker returns all-zero matrix."""
        result = tracker.get_transition_matrix()
        assert result["total_transitions"] == 0
        assert result["total_remaining_ha"] == "0"
        assert result["total_conversion_ha"] == "0"

    def test_single_conversion_in_matrix(self, tracker, base_request):
        """Single conversion appears in the correct matrix cell."""
        tracker.record_transition(base_request)
        result = tracker.get_transition_matrix()
        m = result["matrix"]
        assert m["FOREST_LAND"]["CROPLAND"] == "50.0"

    def test_matrix_accumulates(self, tracker):
        """Multiple transitions to the same cell accumulate area."""
        for _ in range(3):
            tracker.record_transition({
                "parcel_id": "P001",
                "from_category": "FOREST_LAND",
                "to_category": "CROPLAND",
                "area_ha": 10,
                "transition_date": "2023-01-01",
            })
        result = tracker.get_transition_matrix()
        assert Decimal(result["matrix"]["FOREST_LAND"]["CROPLAND"]) == Decimal("30.0")

    def test_matrix_remaining_vs_conversion_totals(self, tracker_with_transitions):
        """Remaining and conversion totals sum correctly."""
        result = tracker_with_transitions.get_transition_matrix()
        remaining = Decimal(result["total_remaining_ha"])
        conversion = Decimal(result["total_conversion_ha"])
        total = Decimal(result["total_area_ha"])
        assert total == remaining + conversion

    def test_matrix_has_provenance(self, tracker_with_transitions):
        """get_transition_matrix includes a provenance hash."""
        result = tracker_with_transitions.get_transition_matrix()
        assert len(result["provenance_hash"]) == 64

    def test_matrix_categories_list(self, tracker):
        """Matrix result includes the categories list."""
        result = tracker.get_transition_matrix()
        assert result["categories"] == LAND_CATEGORIES


# ===========================================================================
# Transition History Tests
# ===========================================================================


class TestTransitionHistory:
    """Tests for get_transition_history() with filters and pagination."""

    def test_unfiltered_history(self, tracker_with_transitions):
        """Unfiltered history returns all transitions."""
        result = tracker_with_transitions.get_transition_history()
        assert result["total_count"] == 10

    def test_filter_by_parcel(self, tracker_with_transitions):
        """Filter by parcel_id returns only that parcel's transitions."""
        result = tracker_with_transitions.get_transition_history(parcel_id="P001")
        assert result["total_count"] == 1
        assert result["transitions"][0]["parcel_id"] == "P001"

    def test_filter_by_from_category(self, tracker_with_transitions):
        """Filter by from_category returns matching records."""
        result = tracker_with_transitions.get_transition_history(
            from_category="FOREST_LAND",
        )
        # 3 deforestation + 1 remaining = 4
        assert result["total_count"] == 4

    def test_filter_by_to_category(self, tracker_with_transitions):
        """Filter by to_category returns matching records."""
        result = tracker_with_transitions.get_transition_history(
            to_category="FOREST_LAND",
        )
        # 2 afforestation + 1 remaining = 3
        assert result["total_count"] == 3

    def test_filter_by_date_range(self, tracker_with_transitions):
        """Date range filters work correctly."""
        result = tracker_with_transitions.get_transition_history(
            start_date="2023-01-01", end_date="2023-12-31",
        )
        # 2023 events: FL->SL, GL->FL, CL->WL, GL->SL, OL->CL = 5
        assert result["total_count"] == 5

    def test_pagination_limit(self, tracker_with_transitions):
        """limit parameter restricts results."""
        result = tracker_with_transitions.get_transition_history(limit=3)
        assert len(result["transitions"]) == 3
        assert result["total_count"] == 10
        assert result["has_more"] is True

    def test_pagination_offset(self, tracker_with_transitions):
        """offset parameter skips records."""
        result = tracker_with_transitions.get_transition_history(limit=5, offset=8)
        assert len(result["transitions"]) == 2
        assert result["has_more"] is False


# ===========================================================================
# Transition Period and Completion Date Tests
# ===========================================================================


class TestTransitionPeriod:
    """Tests for 20-year transition period and completion date calculation."""

    def test_default_20_year_period(self, tracker, base_request):
        """Default transition period is 20 years."""
        result = tracker.record_transition(base_request)
        assert result["transition_period_years"] == 20

    def test_completion_date_20_years_later(self, tracker, base_request):
        """Completion date is 20 years after transition date."""
        result = tracker.record_transition(base_request)
        assert result["completion_date"] == "2043-01-15"

    def test_custom_period_completion(self, tracker):
        """Custom transition period adjusts the completion date."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "CROPLAND",
            "to_category": "GRASSLAND",
            "area_ha": 10,
            "transition_date": "2020-06-01",
            "transition_period_years": 10,
        })
        assert result["completion_date"] == "2030-06-01"

    def test_leap_day_completion(self, tracker):
        """Feb 29 transition date gets Feb 28 completion if target year is not leap."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 10,
            "transition_date": "2024-02-29",
        })
        # 2024+20=2044 is a leap year, so Feb 29 is valid
        assert result["completion_date"] == "2044-02-29"

    def test_remaining_also_gets_completion_date(self, tracker):
        """Even REMAINING transitions get a completion date (informational)."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "FOREST_LAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        assert result["completion_date"] == "2043-01-01"


# ===========================================================================
# Transition Age Tests
# ===========================================================================


class TestTransitionAge:
    """Tests for get_transition_age()."""

    def test_age_calculation(self, tracker, base_request):
        """Transition age is correctly calculated."""
        result = tracker.record_transition(base_request)
        tid = result["transition_id"]
        age_result = tracker.get_transition_age(tid, as_of_date="2025-01-15")
        # 2 years from 2023-01-15 to 2025-01-15
        age = Decimal(age_result["age_years"])
        assert age > Decimal("1.9")
        assert age < Decimal("2.1")

    def test_soc_transition_not_complete(self, tracker, base_request):
        """Within the 20-year period, SOC transition is not complete."""
        result = tracker.record_transition(base_request)
        tid = result["transition_id"]
        age_result = tracker.get_transition_age(tid, as_of_date="2030-01-15")
        assert age_result["is_soc_transition_complete"] is False

    def test_soc_transition_complete_after_20_years(self, tracker, base_request):
        """After the 20-year period, SOC transition is complete."""
        result = tracker.record_transition(base_request)
        tid = result["transition_id"]
        age_result = tracker.get_transition_age(tid, as_of_date="2043-01-16")
        assert age_result["is_soc_transition_complete"] is True

    def test_age_not_found(self, tracker):
        """Non-existent transition_id returns NOT_FOUND."""
        result = tracker.get_transition_age("nonexistent-id")
        assert result["status"] == "NOT_FOUND"


# ===========================================================================
# Active Transitions Tests
# ===========================================================================


class TestActiveTransitions:
    """Tests for get_active_transitions()."""

    def test_conversion_is_active_within_period(self, tracker, base_request):
        """A conversion is active if as_of_date is within transition period."""
        tracker.record_transition(base_request)
        result = tracker.get_active_transitions(as_of_date="2030-01-01")
        assert result["active_count"] == 1

    def test_remaining_excluded_from_active(self, tracker):
        """REMAINING transitions are excluded from active transitions."""
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "FOREST_LAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        result = tracker.get_active_transitions(as_of_date="2030-01-01")
        assert result["active_count"] == 0

    def test_completed_transition_not_active(self, tracker, base_request):
        """A transition past its completion date is not active."""
        tracker.record_transition(base_request)
        result = tracker.get_active_transitions(as_of_date="2050-01-01")
        assert result["active_count"] == 0

    def test_active_includes_progress(self, tracker, base_request):
        """Active transitions include elapsed_years, remaining_years, progress_pct."""
        tracker.record_transition(base_request)
        result = tracker.get_active_transitions(as_of_date="2033-01-15")
        entry = result["active_transitions"][0]
        assert "elapsed_years" in entry
        assert "remaining_years" in entry
        assert "progress_pct" in entry
        # 10 years elapsed -> ~50%
        progress = Decimal(entry["progress_pct"])
        assert progress > Decimal("49")
        assert progress < Decimal("51")


# ===========================================================================
# Portfolio Summary Tests
# ===========================================================================


class TestPortfolioSummary:
    """Tests for get_portfolio_summary()."""

    def test_summary_totals(self, tracker_with_transitions):
        """Portfolio summary reports correct totals."""
        result = tracker_with_transitions.get_portfolio_summary()
        assert result["total_parcels"] == 10
        assert result["total_transitions"] == 10

    def test_summary_deforestation(self, tracker_with_transitions):
        """Portfolio summary counts deforestation correctly."""
        result = tracker_with_transitions.get_portfolio_summary()
        assert result["deforestation"]["count"] == 3
        assert Decimal(result["deforestation"]["area_ha"]) == Decimal("90")

    def test_summary_afforestation(self, tracker_with_transitions):
        """Portfolio summary counts afforestation correctly."""
        result = tracker_with_transitions.get_portfolio_summary()
        assert result["afforestation"]["count"] == 2
        assert Decimal(result["afforestation"]["area_ha"]) == Decimal("40")

    def test_summary_net_forest_change(self, tracker_with_transitions):
        """Net forest change = afforestation - deforestation."""
        result = tracker_with_transitions.get_portfolio_summary()
        net = Decimal(result["net_forest_change_ha"])
        assert net == Decimal("40") - Decimal("90")  # -50

    def test_summary_remaining_vs_conversion(self, tracker_with_transitions):
        """Remaining and conversion counts are correct."""
        result = tracker_with_transitions.get_portfolio_summary()
        assert result["remaining"]["count"] == 1  # FL->FL
        assert result["conversions"]["count"] == 9

    def test_summary_provenance_hash(self, tracker_with_transitions):
        """Portfolio summary includes a provenance hash."""
        result = tracker_with_transitions.get_portfolio_summary()
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Transition Rate Tests
# ===========================================================================


class TestTransitionRate:
    """Tests for get_transition_rate()."""

    def test_annual_rate_calculation(self, tracker_with_transitions):
        """Annual rate = total area / years."""
        result = tracker_with_transitions.get_transition_rate(
            from_category="FOREST_LAND",
            to_category="CROPLAND",
            start_year=2022,
            end_year=2024,
        )
        # 50 ha over 2 years = 25 ha/yr
        assert Decimal(result["annual_rate_ha_yr"]) == Decimal("25.00000000")
        assert result["transition_count"] == 1

    def test_rate_invalid_year_range(self, tracker):
        """end_year <= start_year raises ValueError."""
        with pytest.raises(ValueError, match="end_year must be > start_year"):
            tracker.get_transition_rate("FOREST_LAND", "CROPLAND", 2023, 2023)

    def test_rate_invalid_category(self, tracker):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown land category"):
            tracker.get_transition_rate("OCEAN", "CROPLAND", 2020, 2025)


# ===========================================================================
# Area Consistency Validation Tests
# ===========================================================================


class TestAreaConsistency:
    """Tests for validate_area_consistency()."""

    def test_balanced_transitions_consistent(self, tracker):
        """From and to totals balance when transitions are symmetric."""
        # Each record has from = to in terms of total area tracking
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 50,
            "transition_date": "2023-01-01",
        })
        result = tracker.validate_area_consistency()
        assert result["is_consistent"] is True
        # From = 50, To = 50 (same record)
        assert result["total_from_ha"] == "50.0"
        assert result["total_to_ha"] == "50.0"
        assert result["discrepancy_ha"] == "0.0"

    def test_expected_total_check(self, tracker):
        """Validation against an expected total area."""
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        result = tracker.validate_area_consistency(
            expected_total_ha=Decimal("100"),
        )
        assert result["is_consistent"] is True

    def test_expected_total_mismatch(self, tracker):
        """Mismatch with expected total area triggers finding."""
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        result = tracker.validate_area_consistency(
            expected_total_ha=Decimal("200"),
        )
        assert result["is_consistent"] is False
        assert len(result["findings"]) >= 1

    def test_empty_tracker_consistent(self, tracker):
        """Empty tracker is consistent (no discrepancy)."""
        result = tracker.validate_area_consistency()
        assert result["is_consistent"] is True

    def test_consistency_provenance_hash(self, tracker_with_transitions):
        """Consistency validation includes a provenance hash."""
        result = tracker_with_transitions.validate_area_consistency()
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Transition Reversal Detection Tests
# ===========================================================================


class TestReversalDetection:
    """Tests for detect_reversals()."""

    def test_reversal_detected(self, tracker):
        """A->B followed by B->A on the same parcel is a reversal."""
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 50,
            "transition_date": "2020-01-01",
        })
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "CROPLAND",
            "to_category": "FOREST_LAND",
            "area_ha": 50,
            "transition_date": "2023-01-01",
        })
        result = tracker.detect_reversals(parcel_id="P001")
        assert result["reversal_count"] == 1
        rev = result["reversals"][0]
        assert rev["original_from"] == "FOREST_LAND"
        assert rev["original_to"] == "CROPLAND"
        assert rev["reversal_from"] == "CROPLAND"
        assert rev["reversal_to"] == "FOREST_LAND"

    def test_no_reversal_different_parcels(self, tracker):
        """A->B on P001 and B->A on P002 is not a reversal (different parcels)."""
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 50,
            "transition_date": "2020-01-01",
        })
        tracker.record_transition({
            "parcel_id": "P002",
            "from_category": "CROPLAND",
            "to_category": "FOREST_LAND",
            "area_ha": 50,
            "transition_date": "2023-01-01",
        })
        result = tracker.detect_reversals(parcel_id="P001")
        assert result["reversal_count"] == 0

    def test_remaining_transitions_excluded(self, tracker):
        """REMAINING transitions are not counted as reversals."""
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "FOREST_LAND",
            "area_ha": 100,
            "transition_date": "2020-01-01",
        })
        tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "FOREST_LAND",
            "area_ha": 100,
            "transition_date": "2023-01-01",
        })
        result = tracker.detect_reversals(parcel_id="P001")
        assert result["reversal_count"] == 0

    def test_reversal_provenance_hash(self, tracker):
        """detect_reversals includes a provenance hash."""
        result = tracker.detect_reversals()
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Cumulative Transition Area Tests
# ===========================================================================


class TestCumulativeTransitions:
    """Tests for get_cumulative_transitions()."""

    def test_annual_area_by_year(self, tracker_with_transitions):
        """Annual area is grouped by year correctly."""
        result = tracker_with_transitions.get_cumulative_transitions()
        annual = result["annual_area"]
        assert "2022" in annual
        assert "2023" in annual

    def test_cumulative_accumulates(self, tracker_with_transitions):
        """Cumulative area accumulates year over year."""
        result = tracker_with_transitions.get_cumulative_transitions()
        cumulative = result["cumulative_area"]
        years = sorted(cumulative.keys())
        assert len(years) >= 2
        assert Decimal(cumulative[years[-1]]) >= Decimal(cumulative[years[0]])

    def test_filter_by_from_category(self, tracker_with_transitions):
        """from_category filter restricts cumulative results."""
        result = tracker_with_transitions.get_cumulative_transitions(
            from_category="FOREST_LAND",
        )
        total = Decimal(result["total_area_ha"])
        # FL->CL(50) + FL->GL(30) + FL->SL(10) + FL->FL(200) = 290
        assert total == Decimal("290.0")

    def test_filter_by_to_category(self, tracker_with_transitions):
        """to_category filter restricts cumulative results."""
        result = tracker_with_transitions.get_cumulative_transitions(
            to_category="CROPLAND",
        )
        total = Decimal(result["total_area_ha"])
        # FL->CL(50) + WL->CL(20) + OL->CL(12) = 82
        assert total == Decimal("82.0")

    def test_years_covered(self, tracker_with_transitions):
        """years_covered lists all years with transitions."""
        result = tracker_with_transitions.get_cumulative_transitions()
        assert 2022 in result["years_covered"]
        assert 2023 in result["years_covered"]


# ===========================================================================
# Reset and Statistics Tests
# ===========================================================================


class TestResetAndStatistics:
    """Tests for reset() and get_statistics()."""

    def test_reset_clears_all(self, tracker_with_transitions):
        """reset() clears all state."""
        tracker_with_transitions.reset()
        stats = tracker_with_transitions.get_statistics()
        assert stats["total_transitions"] == 0
        assert stats["total_parcels"] == 0

    def test_reset_clears_matrix(self, tracker_with_transitions):
        """reset() zeroes the transition matrix."""
        tracker_with_transitions.reset()
        result = tracker_with_transitions.get_transition_matrix()
        assert result["total_area_ha"] == "0"

    def test_reset_clears_history(self, tracker_with_transitions):
        """reset() clears transition history."""
        tracker_with_transitions.reset()
        hist = tracker_with_transitions.get_transition_history()
        assert hist["total_count"] == 0

    def test_statistics_after_transitions(self, tracker_with_transitions):
        """Statistics reflect recorded transitions."""
        stats = tracker_with_transitions.get_statistics()
        assert stats["total_transitions"] == 10
        assert stats["total_parcels"] == 10
        assert stats["engine"] == "LandUseChangeTrackerEngine"


# ===========================================================================
# Thread Safety Tests
# ===========================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_record_transitions(self, tracker):
        """Concurrent record_transition calls do not corrupt state."""
        errors = []

        def record_worker(idx):
            try:
                tracker.record_transition({
                    "parcel_id": f"P{idx:04d}",
                    "from_category": LAND_CATEGORIES[idx % 6],
                    "to_category": LAND_CATEGORIES[(idx + 1) % 6],
                    "area_ha": 1.0,
                    "transition_date": "2023-01-01",
                })
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=record_worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = tracker.get_statistics()
        assert stats["total_transitions"] == 50


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_area(self, tracker):
        """Very small area (0.001 ha) is accepted."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 0.001,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "SUCCESS"
        assert Decimal(result["area_ha"]) == Decimal("0.001")

    def test_very_large_area(self, tracker):
        """Very large area (1,000,000 ha) is accepted."""
        result = tracker.record_transition({
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 1000000,
            "transition_date": "2023-01-01",
        })
        assert result["status"] == "SUCCESS"

    def test_same_parcel_multiple_transitions(self, tracker):
        """Same parcel can have multiple transitions recorded."""
        for to_cat in ["CROPLAND", "GRASSLAND", "SETTLEMENTS"]:
            tracker.record_transition({
                "parcel_id": "P001",
                "from_category": "FOREST_LAND",
                "to_category": to_cat,
                "area_ha": 10,
                "transition_date": "2023-01-01",
            })
        hist = tracker.get_transition_history(parcel_id="P001")
        assert hist["total_count"] == 3

    def test_deterministic_provenance_for_same_inputs(self, tracker):
        """Two trackers recording the same data produce the same provenance hash in the matrix."""
        t1 = LandUseChangeTrackerEngine()
        t2 = LandUseChangeTrackerEngine()
        req = {
            "parcel_id": "P001",
            "from_category": "FOREST_LAND",
            "to_category": "CROPLAND",
            "area_ha": 50,
            "transition_date": "2023-01-15",
        }
        t1.record_transition(req)
        t2.record_transition(req)
        m1 = t1.get_transition_matrix()
        m2 = t2.get_transition_matrix()
        assert m1["provenance_hash"] == m2["provenance_hash"]
