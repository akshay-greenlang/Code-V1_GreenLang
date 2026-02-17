# -*- coding: utf-8 -*-
"""
Unit tests for ExposureAssessorEngine (Engine 4 of 7).

AGENT-DATA-020: Climate Hazard Connector
Tests asset registration, exposure assessment, portfolio analysis,
supply-chain exposure, hotspot identification, exposure mapping,
statistics, and engine lifecycle management.

Target: 85%+ code coverage with 90+ test functions.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.climate_hazard.exposure_assessor import (
    AssetRecord,
    AssetType,
    EARTH_RADIUS_KM,
    ExposureAssessment,
    ExposureAssessorEngine,
    ExposureLevel,
    EXPOSURE_THRESHOLDS,
    HAZARD_MAX_RADIUS_KM,
    HazardType,
    VALID_ASSET_TYPES,
    VALID_EXPOSURE_LEVELS,
    VALID_HAZARD_TYPES,
    WEIGHT_ELEVATION,
    WEIGHT_FREQUENCY,
    WEIGHT_INTENSITY,
    WEIGHT_POPULATION,
    WEIGHT_PROXIMITY,
    _build_hash,
    _clamp,
    _haversine_km,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ExposureAssessorEngine:
    """Return a fresh ExposureAssessorEngine for each test."""
    return ExposureAssessorEngine()


@pytest.fixture
def sample_location() -> Dict[str, Any]:
    """Standard asset location."""
    return {"latitude": 51.5074, "longitude": -0.1278}


@pytest.fixture
def sample_location_with_elevation() -> Dict[str, Any]:
    """Asset location with elevation data."""
    return {"latitude": 51.5074, "longitude": -0.1278, "elevation_m": 15.0}


@pytest.fixture
def registered_asset(engine, sample_location) -> Dict[str, Any]:
    """Register and return a standard asset."""
    return engine.register_asset(
        asset_id="A-001",
        name="Factory Alpha",
        asset_type="FACILITY",
        location=sample_location,
        sector="manufacturing",
        value_usd=1_000_000.0,
    )


@pytest.fixture
def multiple_assets(engine) -> List[Dict[str, Any]]:
    """Register multiple assets for portfolio testing."""
    assets = []
    locations = [
        {"latitude": 51.5, "longitude": -0.1},
        {"latitude": 48.8, "longitude": 2.3},
        {"latitude": 40.7, "longitude": -74.0},
    ]
    for i, loc in enumerate(locations, 1):
        asset = engine.register_asset(
            asset_id=f"A-{i:03d}",
            name=f"Asset {i}",
            asset_type="FACILITY",
            location=loc,
            sector="manufacturing",
        )
        assets.append(asset)
    return assets


# ===================================================================
# Utility function tests
# ===================================================================


class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_clamp_within_range(self):
        """Value within range is unchanged."""
        assert _clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below_lower(self):
        """Value below lower bound is clamped."""
        assert _clamp(-1.0, 0.0, 10.0) == 0.0

    def test_clamp_above_upper(self):
        """Value above upper bound is clamped."""
        assert _clamp(15.0, 0.0, 10.0) == 10.0

    def test_haversine_same_point(self):
        """Haversine distance of same point is zero."""
        dist = _haversine_km(51.5, -0.1, 51.5, -0.1)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        """Haversine London-Paris is approximately 340 km."""
        dist = _haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert 330.0 < dist < 360.0

    def test_haversine_antipodal(self):
        """Haversine of antipodal points is approximately half Earth circumference."""
        dist = _haversine_km(0.0, 0.0, 0.0, 180.0)
        assert dist == pytest.approx(math.pi * EARTH_RADIUS_KM, rel=0.01)

    def test_build_hash_deterministic(self):
        """build_hash produces deterministic results."""
        data = {"key": "value", "num": 42}
        h1 = _build_hash(data)
        h2 = _build_hash(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_build_hash_different_data(self):
        """Different data produces different hashes."""
        h1 = _build_hash({"a": 1})
        h2 = _build_hash({"a": 2})
        assert h1 != h2


# ===================================================================
# Enum and constant tests
# ===================================================================


class TestEnumsAndConstants:
    """Tests for enumerations and module-level constants."""

    def test_asset_type_count(self):
        """There are 8 asset types."""
        assert len(AssetType) == 8
        assert len(VALID_ASSET_TYPES) == 8

    def test_hazard_type_count(self):
        """There are 12 hazard types."""
        assert len(HazardType) == 12
        assert len(VALID_HAZARD_TYPES) == 12

    def test_exposure_level_count(self):
        """There are 5 exposure levels."""
        assert len(ExposureLevel) == 5
        assert len(VALID_EXPOSURE_LEVELS) == 5

    def test_hazard_max_radius_all_present(self):
        """All 12 hazard types have a max radius."""
        for ht in HazardType:
            assert ht.value in HAZARD_MAX_RADIUS_KM

    def test_weight_sum(self):
        """Composite weights sum to 1.0."""
        total = (WEIGHT_PROXIMITY + WEIGHT_INTENSITY + WEIGHT_FREQUENCY
                 + WEIGHT_ELEVATION + WEIGHT_POPULATION)
        assert total == pytest.approx(1.0)

    def test_exposure_thresholds_cover_full_range(self):
        """Thresholds cover 0-100 without gaps."""
        assert EXPOSURE_THRESHOLDS[0][0] == 0.0
        # Last threshold upper bound covers 100
        assert EXPOSURE_THRESHOLDS[-1][1] >= 100.0


# ===================================================================
# Engine initialization tests
# ===================================================================


class TestEngineInit:
    """Tests for engine construction."""

    def test_default_init(self, engine):
        """Default engine initialises with empty state."""
        assert engine.get_asset_count() == 0
        assert engine.get_assessment_count() == 0

    def test_init_with_risk_engine(self):
        """Engine accepts an optional risk_engine reference."""
        mock = MagicMock()
        eng = ExposureAssessorEngine(risk_engine=mock)
        assert eng._risk_engine is mock

    def test_init_standalone(self, engine):
        """Engine operates standalone by default."""
        assert engine._risk_engine is None


# ===================================================================
# register_asset tests
# ===================================================================


class TestRegisterAsset:
    """Tests for register_asset."""

    def test_basic_registration(self, registered_asset):
        """Asset is registered with correct fields."""
        assert registered_asset["asset_id"] == "A-001"
        assert registered_asset["name"] == "Factory Alpha"
        assert registered_asset["asset_type"] == "FACILITY"
        assert registered_asset["sector"] == "manufacturing"
        assert registered_asset["value_usd"] == 1_000_000.0
        assert len(registered_asset["provenance_hash"]) == 64

    def test_all_8_asset_types(self, engine, sample_location):
        """All 8 asset types are accepted."""
        for i, at in enumerate(AssetType):
            asset = engine.register_asset(
                asset_id=f"AT-{i}",
                name=f"Asset {at.value}",
                asset_type=at.value,
                location=sample_location,
            )
            assert asset["asset_type"] == at.value

    def test_case_insensitive_type(self, engine, sample_location):
        """Asset type is case insensitive."""
        asset = engine.register_asset(
            asset_id="CI-001",
            name="Test",
            asset_type="facility",
            location=sample_location,
        )
        assert asset["asset_type"] == "FACILITY"

    def test_empty_asset_id_raises(self, engine, sample_location):
        """Empty asset_id raises ValueError."""
        with pytest.raises(ValueError):
            engine.register_asset(
                asset_id="",
                name="Test",
                asset_type="FACILITY",
                location=sample_location,
            )

    def test_empty_name_raises(self, engine, sample_location):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError):
            engine.register_asset(
                asset_id="E-001",
                name="",
                asset_type="FACILITY",
                location=sample_location,
            )

    def test_invalid_asset_type_raises(self, engine, sample_location):
        """Invalid asset type raises ValueError."""
        with pytest.raises(ValueError):
            engine.register_asset(
                asset_id="E-001",
                name="Test",
                asset_type="INVALID_TYPE",
                location=sample_location,
            )

    def test_negative_value_raises(self, engine, sample_location):
        """Negative value_usd raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            engine.register_asset(
                asset_id="E-001",
                name="Test",
                asset_type="FACILITY",
                location=sample_location,
                value_usd=-100.0,
            )

    def test_metadata_stored(self, engine, sample_location):
        """Custom metadata is stored."""
        asset = engine.register_asset(
            asset_id="M-001",
            name="Test",
            asset_type="FACILITY",
            location=sample_location,
            metadata={"custom_key": "custom_value"},
        )
        assert asset["metadata"]["custom_key"] == "custom_value"

    def test_deep_copy_returned(self, engine, sample_location):
        """Returned asset is a deep copy."""
        asset = engine.register_asset(
            asset_id="DC-001",
            name="Test",
            asset_type="FACILITY",
            location=sample_location,
        )
        asset["name"] = "MODIFIED"
        stored = engine.get_asset("DC-001")
        assert stored["name"] == "Test"


# ===================================================================
# get_asset tests
# ===================================================================


class TestGetAsset:
    """Tests for get_asset."""

    def test_existing_asset(self, engine, registered_asset):
        """Existing asset is retrieved correctly."""
        result = engine.get_asset("A-001")
        assert result is not None
        assert result["asset_id"] == "A-001"

    def test_non_existent_asset(self, engine):
        """Non-existent asset returns None."""
        assert engine.get_asset("DOES_NOT_EXIST") is None

    def test_empty_id(self, engine):
        """Empty asset_id returns None."""
        assert engine.get_asset("") is None


# ===================================================================
# list_assets tests
# ===================================================================


class TestListAssets:
    """Tests for list_assets."""

    def test_list_empty(self, engine):
        """Empty engine returns empty list."""
        assert engine.list_assets() == []

    def test_list_all(self, engine, multiple_assets):
        """list_assets returns all registered assets."""
        result = engine.list_assets()
        assert len(result) == 3

    def test_filter_by_type(self, engine, sample_location):
        """Filter by asset_type works."""
        engine.register_asset("F-001", "Fac", "FACILITY", sample_location)
        engine.register_asset("R-001", "Real", "REAL_ESTATE", sample_location)
        result = engine.list_assets(asset_type="FACILITY")
        assert all(a["asset_type"] == "FACILITY" for a in result)

    def test_filter_by_sector(self, engine, sample_location):
        """Filter by sector works."""
        engine.register_asset("S1", "A1", "FACILITY", sample_location, sector="energy")
        engine.register_asset("S2", "A2", "FACILITY", sample_location, sector="manufacturing")
        result = engine.list_assets(sector="energy")
        assert all(a["sector"] == "energy" for a in result)

    def test_limit(self, engine, multiple_assets):
        """Limit parameter is respected."""
        result = engine.list_assets(limit=2)
        assert len(result) == 2


# ===================================================================
# update_asset tests
# ===================================================================


class TestUpdateAsset:
    """Tests for update_asset."""

    def test_update_name(self, engine, registered_asset):
        """Name can be updated."""
        result = engine.update_asset("A-001", name="Factory Beta")
        assert result["name"] == "Factory Beta"

    def test_update_sector(self, engine, registered_asset):
        """Sector can be updated."""
        result = engine.update_asset("A-001", sector="energy")
        assert result["sector"] == "energy"

    def test_update_value_usd(self, engine, registered_asset):
        """value_usd can be updated."""
        result = engine.update_asset("A-001", value_usd=2_000_000.0)
        assert result["value_usd"] == 2_000_000.0

    def test_update_non_existent_raises(self, engine):
        """Updating non-existent asset raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.update_asset("DOES_NOT_EXIST", name="New")

    def test_update_empty_id_raises(self, engine):
        """Empty asset_id raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.update_asset("", name="New")

    def test_update_negative_value_raises(self, engine, registered_asset):
        """Negative value_usd raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            engine.update_asset("A-001", value_usd=-100.0)

    def test_update_provenance_hash_changes(self, engine, registered_asset):
        """Provenance hash changes after update."""
        original_hash = registered_asset["provenance_hash"]
        updated = engine.update_asset("A-001", name="Updated")
        assert updated["provenance_hash"] != original_hash


# ===================================================================
# delete_asset tests
# ===================================================================


class TestDeleteAsset:
    """Tests for delete_asset."""

    def test_delete_existing(self, engine, registered_asset):
        """Deleting existing asset returns True."""
        assert engine.delete_asset("A-001") is True
        assert engine.get_asset("A-001") is None

    def test_delete_non_existent(self, engine):
        """Deleting non-existent asset returns False."""
        assert engine.delete_asset("DOES_NOT_EXIST") is False

    def test_delete_empty_id(self, engine):
        """Empty asset_id returns False."""
        assert engine.delete_asset("") is False

    def test_delete_removes_assessments(self, engine, registered_asset):
        """Deleting asset also removes its assessments."""
        engine.assess_exposure(
            asset_id="A-001",
            hazard_type="RIVERINE_FLOOD",
            hazard_intensity=0.7,
            hazard_probability=0.6,
            hazard_frequency=0.5,
        )
        engine.delete_asset("A-001")
        # Asset assessments should be removed
        assessments = engine.list_assessments(asset_id="A-001")
        assert len(assessments) == 0


# ===================================================================
# assess_exposure tests
# ===================================================================


class TestAssessExposure:
    """Tests for assess_exposure."""

    def test_basic_assessment(self, engine, registered_asset):
        """Basic assessment returns expected structure."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="RIVERINE_FLOOD",
            hazard_intensity=0.7,
            hazard_probability=0.6,
            hazard_frequency=0.5,
        )
        assert "assessment_id" in result
        assert result["asset_id"] == "A-001"
        assert result["hazard_type"] == "RIVERINE_FLOOD"
        assert result["exposure_level"] in VALID_EXPOSURE_LEVELS
        assert 0.0 <= result["composite_score"] <= 100.0
        assert len(result["provenance_hash"]) == 64

    def test_all_12_hazard_types(self, engine, registered_asset):
        """All 12 hazard types produce valid assessments."""
        for ht in HazardType:
            result = engine.assess_exposure(
                asset_id="A-001",
                hazard_type=ht.value,
                hazard_intensity=0.5,
                hazard_probability=0.5,
                hazard_frequency=0.5,
            )
            assert result["hazard_type"] == ht.value

    def test_zero_intensity_low_score(self, engine, registered_asset):
        """Zero intensity and frequency produces low exposure."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.0,
            hazard_probability=1.0,
            hazard_frequency=0.0,
        )
        # With zero intensity and zero frequency, the score is still
        # non-trivial because proximity and probability contribute.
        assert result["composite_score"] < 50.0

    def test_max_all_factors_high_score(self, engine, registered_asset):
        """Maximum all factors produces high/critical exposure."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="EXTREME_HEAT",
            hazard_intensity=1.0,
            hazard_probability=1.0,
            hazard_frequency=1.0,
            elevation_factor=1.0,
            population_factor=1.0,
        )
        assert result["composite_score"] > 50.0

    def test_distance_reduces_proximity(self, engine, registered_asset):
        """Large distance reduces proximity score."""
        close = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="RIVERINE_FLOOD",
            hazard_intensity=0.7,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            distance_km=0.0,
        )
        far = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="RIVERINE_FLOOD",
            hazard_intensity=0.7,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            distance_km=49.0,
        )
        assert close["proximity_score"] > far["proximity_score"]
        assert close["composite_score"] > far["composite_score"]

    def test_distance_beyond_radius_zero_proximity(self, engine, registered_asset):
        """Distance beyond max radius gives zero proximity."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="RIVERINE_FLOOD",  # max radius 50 km
            hazard_intensity=0.7,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            distance_km=100.0,
        )
        assert result["proximity_score"] == pytest.approx(0.0)

    def test_unregistered_asset_raises(self, engine):
        """Unregistered asset raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.assess_exposure(
                asset_id="DOES_NOT_EXIST",
                hazard_type="DROUGHT",
                hazard_intensity=0.5,
                hazard_probability=0.5,
                hazard_frequency=0.5,
            )

    def test_invalid_hazard_type_raises(self, engine, registered_asset):
        """Invalid hazard type raises ValueError."""
        with pytest.raises(ValueError):
            engine.assess_exposure(
                asset_id="A-001",
                hazard_type="EARTHQUAKE",
                hazard_intensity=0.5,
                hazard_probability=0.5,
                hazard_frequency=0.5,
            )

    def test_intensity_clamped(self, engine, registered_asset):
        """Intensity values are clamped to [0, 1]."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=2.0,
            hazard_probability=0.5,
            hazard_frequency=0.5,
        )
        assert result["intensity_at_location"] == pytest.approx(1.0)

    def test_optional_scenario_passed(self, engine, registered_asset):
        """Optional scenario label is stored."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.5,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            scenario="SSP2-4.5",
        )
        assert result["scenario"] == "SSP2-4.5"

    def test_optional_time_horizon_passed(self, engine, registered_asset):
        """Optional time horizon label is stored."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.5,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            time_horizon="MID_TERM",
        )
        assert result["time_horizon"] == "MID_TERM"

    def test_assessment_stored(self, engine, registered_asset):
        """Assessment is retrievable after creation."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.5,
            hazard_probability=0.5,
            hazard_frequency=0.5,
        )
        stored = engine.get_assessment(result["assessment_id"])
        assert stored is not None
        assert stored["assessment_id"] == result["assessment_id"]

    def test_custom_elevation_factor(self, engine, registered_asset):
        """Custom elevation_factor is used."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.5,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            elevation_factor=0.8,
        )
        assert result["elevation_factor"] == pytest.approx(0.8)

    def test_custom_population_factor(self, engine, registered_asset):
        """Custom population_factor is used."""
        result = engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.5,
            hazard_probability=0.5,
            hazard_frequency=0.5,
            population_factor=0.9,
        )
        assert result["population_factor"] == pytest.approx(0.9)


# ===================================================================
# Exposure level classification tests
# ===================================================================


class TestExposureClassification:
    """Tests for exposure level classification boundaries."""

    def test_classify_none(self, engine):
        """Score < 10 is classified as NONE."""
        level = engine.classify_exposure(5.0)
        assert level == "NONE"

    def test_classify_low(self, engine):
        """Score 10-30 is classified as LOW."""
        level = engine.classify_exposure(20.0)
        assert level == "LOW"

    def test_classify_moderate(self, engine):
        """Score 30-55 is classified as MODERATE."""
        level = engine.classify_exposure(40.0)
        assert level == "MODERATE"

    def test_classify_high(self, engine):
        """Score 55-80 is classified as HIGH."""
        level = engine.classify_exposure(65.0)
        assert level == "HIGH"

    def test_classify_critical(self, engine):
        """Score 80-100 is classified as CRITICAL."""
        level = engine.classify_exposure(90.0)
        assert level == "CRITICAL"

    def test_classify_boundary_10(self, engine):
        """Score exactly 10 is LOW (not NONE)."""
        level = engine.classify_exposure(10.0)
        assert level == "LOW"


# ===================================================================
# assess_portfolio_exposure tests
# ===================================================================


class TestAssessPortfolioExposure:
    """Tests for assess_portfolio_exposure."""

    def test_basic_portfolio(self, engine, multiple_assets):
        """Portfolio assessment returns expected structure."""
        hazard_data = {
            "DROUGHT": {"intensity": 0.6, "probability": 0.5, "frequency": 0.4},
        }
        result = engine.assess_portfolio_exposure(
            asset_ids=["A-001", "A-002", "A-003"],
            hazard_types=["DROUGHT"],
            hazard_data=hazard_data,
        )
        assert "portfolio_summary" in result
        assert "per_asset_results" in result
        assert len(result["per_asset_results"]) == 3
        assert result["portfolio_summary"]["total_assessments"] == 3

    def test_portfolio_multiple_hazards(self, engine, multiple_assets):
        """Portfolio with multiple hazards produces cross-product."""
        hazard_data = {
            "DROUGHT": {"intensity": 0.6, "probability": 0.5, "frequency": 0.4},
            "WILDFIRE": {"intensity": 0.5, "probability": 0.3, "frequency": 0.2},
        }
        result = engine.assess_portfolio_exposure(
            asset_ids=["A-001", "A-002"],
            hazard_types=["DROUGHT", "WILDFIRE"],
            hazard_data=hazard_data,
        )
        assert result["portfolio_summary"]["total_assessments"] == 4  # 2 assets x 2 hazards

    def test_empty_asset_ids_raises(self, engine):
        """Empty asset_ids raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.assess_portfolio_exposure(
                asset_ids=[],
                hazard_types=["DROUGHT"],
                hazard_data={"DROUGHT": {"intensity": 0.5, "probability": 0.5, "frequency": 0.5}},
            )

    def test_empty_hazard_types_raises(self, engine, multiple_assets):
        """Empty hazard_types raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.assess_portfolio_exposure(
                asset_ids=["A-001"],
                hazard_types=[],
                hazard_data={},
            )

    def test_portfolio_summary_stats(self, engine, multiple_assets):
        """Portfolio summary includes avg, max, min scores."""
        hazard_data = {
            "DROUGHT": {"intensity": 0.5, "probability": 0.5, "frequency": 0.5},
        }
        result = engine.assess_portfolio_exposure(
            asset_ids=["A-001", "A-002", "A-003"],
            hazard_types=["DROUGHT"],
            hazard_data=hazard_data,
        )
        summary = result["portfolio_summary"]
        assert "avg_score" in summary
        assert "max_score" in summary
        assert "min_score" in summary
        assert summary["max_score"] >= summary["avg_score"]
        assert summary["avg_score"] >= summary["min_score"]

    def test_portfolio_provenance(self, engine, multiple_assets):
        """Portfolio result includes provenance hash."""
        hazard_data = {
            "DROUGHT": {"intensity": 0.5, "probability": 0.5, "frequency": 0.5},
        }
        result = engine.assess_portfolio_exposure(
            asset_ids=["A-001"],
            hazard_types=["DROUGHT"],
            hazard_data=hazard_data,
        )
        assert len(result["provenance_hash"]) == 64


# ===================================================================
# assess_supply_chain_exposure tests
# ===================================================================


class TestAssessSupplyChainExposure:
    """Tests for assess_supply_chain_exposure."""

    def test_basic_supply_chain(self, engine):
        """Supply chain assessment with tiered nodes."""
        nodes = [
            {
                "asset_id": "SC-001",
                "name": "Supplier 1",
                "asset_type": "SUPPLY_CHAIN_NODE",
                "location": {"latitude": 51.5, "longitude": -0.1},
                "tier": 1,
            },
            {
                "asset_id": "SC-002",
                "name": "Supplier 2",
                "asset_type": "SUPPLY_CHAIN_NODE",
                "location": {"latitude": 48.8, "longitude": 2.3},
                "tier": 2,
            },
        ]
        hazard_data = {
            "DROUGHT": {"intensity": 0.5, "probability": 0.5, "frequency": 0.5},
        }
        result = engine.assess_supply_chain_exposure(
            supply_chain_nodes=nodes,
            hazard_types=["DROUGHT"],
            hazard_data=hazard_data,
        )
        assert "tier_summary" in result
        assert "per_node_results" in result
        assert len(result["per_node_results"]) == 2
        assert "critical_path_exposure" in result

    def test_empty_nodes_raises(self, engine):
        """Empty supply_chain_nodes raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.assess_supply_chain_exposure(
                supply_chain_nodes=[],
                hazard_types=["DROUGHT"],
                hazard_data={},
            )

    def test_empty_hazard_types_raises(self, engine):
        """Empty hazard_types raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.assess_supply_chain_exposure(
                supply_chain_nodes=[{
                    "asset_id": "SC-001", "name": "S1",
                    "asset_type": "SUPPLY_CHAIN_NODE",
                    "location": {"latitude": 0, "longitude": 0},
                    "tier": 1,
                }],
                hazard_types=[],
                hazard_data={},
            )

    def test_auto_registration(self, engine):
        """Unregistered nodes are auto-registered."""
        nodes = [{
            "asset_id": "AUTO-001",
            "name": "Auto Node",
            "asset_type": "SUPPLY_CHAIN_NODE",
            "location": {"latitude": 0.0, "longitude": 0.0},
            "tier": 1,
        }]
        hazard_data = {
            "DROUGHT": {"intensity": 0.5, "probability": 0.5, "frequency": 0.5},
        }
        engine.assess_supply_chain_exposure(
            supply_chain_nodes=nodes,
            hazard_types=["DROUGHT"],
            hazard_data=hazard_data,
        )
        asset = engine.get_asset("AUTO-001")
        assert asset is not None

    def test_tier_summary_structure(self, engine):
        """Tier summary has expected structure for tiers 1, 2, 3."""
        nodes = [
            {
                "asset_id": "T1", "name": "T1", "asset_type": "SUPPLY_CHAIN_NODE",
                "location": {"latitude": 0, "longitude": 0}, "tier": 1,
            },
            {
                "asset_id": "T2", "name": "T2", "asset_type": "SUPPLY_CHAIN_NODE",
                "location": {"latitude": 0, "longitude": 0}, "tier": 2,
            },
        ]
        hazard_data = {"DROUGHT": {"intensity": 0.5, "probability": 0.5, "frequency": 0.5}}
        result = engine.assess_supply_chain_exposure(
            supply_chain_nodes=nodes,
            hazard_types=["DROUGHT"],
            hazard_data=hazard_data,
        )
        for tier_key in ["1", "2", "3"]:
            assert tier_key in result["tier_summary"]
            ts = result["tier_summary"][tier_key]
            assert "avg_score" in ts
            assert "max_score" in ts
            assert "node_count" in ts
            assert "critical_count" in ts


# ===================================================================
# identify_hotspots tests
# ===================================================================


class TestIdentifyHotspots:
    """Tests for identify_hotspots."""

    def test_no_hotspots(self, engine, registered_asset):
        """No hotspots when all scores are below threshold."""
        engine.assess_exposure(
            asset_id="A-001",
            hazard_type="DROUGHT",
            hazard_intensity=0.1,
            hazard_probability=0.1,
            hazard_frequency=0.1,
        )
        hotspots = engine.identify_hotspots(threshold=55.0)
        assert len(hotspots) == 0

    def test_hotspot_found(self, engine, registered_asset):
        """High-scoring assessment is identified as hotspot."""
        engine.assess_exposure(
            asset_id="A-001",
            hazard_type="EXTREME_HEAT",
            hazard_intensity=1.0,
            hazard_probability=1.0,
            hazard_frequency=1.0,
            elevation_factor=1.0,
            population_factor=1.0,
        )
        hotspots = engine.identify_hotspots(threshold=50.0)
        assert len(hotspots) >= 1
        assert hotspots[0]["asset_id"] == "A-001"

    def test_hotspots_sorted_descending(self, engine, sample_location):
        """Hotspots are sorted by composite_score descending."""
        engine.register_asset("H-001", "High", "FACILITY", sample_location)
        engine.register_asset("H-002", "Low", "FACILITY", sample_location)
        engine.assess_exposure("H-001", "DROUGHT", 1.0, 1.0, 1.0)
        engine.assess_exposure("H-002", "DROUGHT", 0.8, 0.8, 0.8)
        hotspots = engine.identify_hotspots(threshold=0.0)
        if len(hotspots) > 1:
            assert hotspots[0]["composite_score"] >= hotspots[1]["composite_score"]

    def test_filter_by_asset_ids(self, engine, sample_location):
        """Hotspot search can be filtered by asset IDs."""
        engine.register_asset("F-001", "F1", "FACILITY", sample_location)
        engine.register_asset("F-002", "F2", "FACILITY", sample_location)
        engine.assess_exposure("F-001", "DROUGHT", 1.0, 1.0, 1.0)
        engine.assess_exposure("F-002", "DROUGHT", 1.0, 1.0, 1.0)
        hotspots = engine.identify_hotspots(asset_ids=["F-001"], threshold=0.0)
        assert all(h["asset_id"] == "F-001" for h in hotspots)

    def test_filter_by_hazard_types(self, engine, registered_asset):
        """Hotspot search can be filtered by hazard types."""
        engine.assess_exposure("A-001", "DROUGHT", 1.0, 1.0, 1.0)
        engine.assess_exposure("A-001", "WILDFIRE", 1.0, 1.0, 1.0)
        hotspots = engine.identify_hotspots(hazard_types=["DROUGHT"], threshold=0.0)
        assert all(h["hazard_type"] == "DROUGHT" for h in hotspots)

    def test_custom_threshold(self, engine, registered_asset):
        """Custom threshold is respected."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        low = engine.identify_hotspots(threshold=0.0)
        high = engine.identify_hotspots(threshold=99.0)
        assert len(low) >= len(high)


# ===================================================================
# get_exposure_map tests
# ===================================================================


class TestGetExposureMap:
    """Tests for get_exposure_map."""

    def test_basic_map(self, engine):
        """Basic exposure map is generated."""
        result = engine.get_exposure_map(
            hazard_type="DROUGHT",
            bounding_box={
                "min_lat": 51.0, "max_lat": 52.0,
                "min_lon": -0.5, "max_lon": 0.5,
            },
            resolution_km=50.0,
        )
        assert result["hazard_type"] == "DROUGHT"
        assert result["cell_count"] > 0
        assert len(result["grid_cells"]) > 0
        assert len(result["provenance_hash"]) == 64

    def test_grid_cells_structure(self, engine):
        """Each grid cell has expected keys."""
        result = engine.get_exposure_map(
            hazard_type="WILDFIRE",
            bounding_box={
                "min_lat": 40.0, "max_lat": 41.0,
                "min_lon": -74.5, "max_lon": -73.5,
            },
            resolution_km=50.0,
        )
        cell = result["grid_cells"][0]
        assert "latitude" in cell
        assert "longitude" in cell
        assert "exposure_score" in cell
        assert "exposure_level" in cell
        assert "distance_km" in cell

    def test_missing_bounding_box_keys_raises(self, engine):
        """Missing bounding box keys raises ValueError."""
        with pytest.raises(ValueError, match="missing keys"):
            engine.get_exposure_map(
                hazard_type="DROUGHT",
                bounding_box={"min_lat": 51.0, "max_lat": 52.0},
            )

    def test_invalid_bounding_box_raises(self, engine):
        """min_lat >= max_lat raises ValueError."""
        with pytest.raises(ValueError, match="must be <"):
            engine.get_exposure_map(
                hazard_type="DROUGHT",
                bounding_box={
                    "min_lat": 52.0, "max_lat": 51.0,
                    "min_lon": -0.5, "max_lon": 0.5,
                },
            )

    def test_out_of_range_lat_raises(self, engine):
        """Latitude outside [-90, 90] raises ValueError."""
        with pytest.raises(ValueError, match="Latitude"):
            engine.get_exposure_map(
                hazard_type="DROUGHT",
                bounding_box={
                    "min_lat": -100.0, "max_lat": 52.0,
                    "min_lon": -0.5, "max_lon": 0.5,
                },
            )


# ===================================================================
# get_statistics tests
# ===================================================================


class TestGetStatistics:
    """Tests for get_statistics."""

    def test_initial_stats(self, engine):
        """Initial stats are all zeros."""
        stats = engine.get_statistics()
        assert stats["total_assets"] == 0
        assert stats["total_assessments"] == 0
        assert stats["total_portfolios"] == 0
        assert stats["total_supply_chains"] == 0
        assert stats["total_errors"] == 0

    def test_stats_after_operations(self, engine, registered_asset):
        """Stats reflect accumulated operations."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        stats = engine.get_statistics()
        assert stats["total_assets"] == 1
        assert stats["total_assessments"] == 1

    def test_exposure_level_distribution(self, engine, registered_asset):
        """Stats include exposure level distribution."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        stats = engine.get_statistics()
        assert "exposure_level_distribution" in stats
        total = sum(stats["exposure_level_distribution"].values())
        assert total == 1


# ===================================================================
# clear tests
# ===================================================================


class TestClear:
    """Tests for clear."""

    def test_clear_resets_all(self, engine, registered_asset):
        """clear removes all assets, assessments, and resets counters."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        engine.clear()
        assert engine.get_asset_count() == 0
        assert engine.get_assessment_count() == 0
        stats = engine.get_statistics()
        assert stats["total_assets"] == 0
        assert stats["total_assessments"] == 0

    def test_clear_idempotent(self, engine):
        """Clearing empty engine is safe."""
        engine.clear()
        assert engine.get_asset_count() == 0


# ===================================================================
# Utility method tests
# ===================================================================


class TestUtilityMethods:
    """Tests for public utility methods."""

    def test_compute_proximity_score(self, engine):
        """Proximity score decreases with distance."""
        score_close = engine.compute_proximity_score(0.0, "RIVERINE_FLOOD")
        score_mid = engine.compute_proximity_score(25.0, "RIVERINE_FLOOD")
        score_far = engine.compute_proximity_score(50.0, "RIVERINE_FLOOD")
        assert score_close == pytest.approx(1.0)
        assert score_mid == pytest.approx(0.5)
        assert score_far == pytest.approx(0.0)

    def test_compute_composite_score(self, engine):
        """Composite score calculation is deterministic."""
        score = engine.compute_composite_score(
            proximity_score=1.0,
            intensity_norm=1.0,
            frequency_norm=1.0,
            elevation_factor=1.0,
            population_factor=1.0,
            probability_norm=1.0,
        )
        assert score == pytest.approx(100.0)

    def test_composite_score_zero(self, engine):
        """All zeros yields zero composite score."""
        score = engine.compute_composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_get_hazard_max_radius(self, engine):
        """get_hazard_max_radius returns correct values."""
        assert engine.get_hazard_max_radius("RIVERINE_FLOOD") == 50.0
        assert engine.get_hazard_max_radius("DROUGHT") == 500.0
        assert engine.get_hazard_max_radius("UNKNOWN") == 100.0

    def test_get_supported_asset_types(self, engine):
        """Returns sorted list of 8 asset types."""
        types = engine.get_supported_asset_types()
        assert len(types) == 8
        assert types == sorted(types)

    def test_get_supported_hazard_types(self, engine):
        """Returns sorted list of 12 hazard types."""
        types = engine.get_supported_hazard_types()
        assert len(types) == 12
        assert types == sorted(types)

    def test_get_supported_exposure_levels(self, engine):
        """Returns ordered list of 5 exposure levels."""
        levels = engine.get_supported_exposure_levels()
        assert len(levels) == 5
        assert levels[0] == "NONE"
        assert levels[-1] == "CRITICAL"

    def test_haversine_distance_method(self, engine):
        """Public haversine_distance wrapper works."""
        dist = engine.haversine_distance(51.5, -0.1, 48.8, 2.3)
        assert 300.0 < dist < 400.0

    def test_get_asset_count(self, engine, registered_asset):
        """get_asset_count returns correct count."""
        assert engine.get_asset_count() == 1

    def test_get_assessment_count(self, engine, registered_asset):
        """get_assessment_count returns correct count."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        assert engine.get_assessment_count() == 1

    def test_batch_register_assets(self, engine):
        """batch_register_assets registers multiple assets."""
        assets = [
            {"asset_id": "B-001", "name": "B1", "asset_type": "FACILITY",
             "location": {"latitude": 0, "longitude": 0}},
            {"asset_id": "B-002", "name": "B2", "asset_type": "REAL_ESTATE",
             "location": {"latitude": 1, "longitude": 1}},
        ]
        results = engine.batch_register_assets(assets)
        assert len(results) == 2
        assert engine.get_asset_count() == 2


# ===================================================================
# Thread safety tests
# ===================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registrations(self, engine):
        """Concurrent asset registrations are safe."""
        errors = []

        def register(i):
            try:
                engine.register_asset(
                    asset_id=f"T-{i:04d}",
                    name=f"Thread Asset {i}",
                    asset_type="FACILITY",
                    location={"latitude": i * 0.01, "longitude": i * 0.01},
                )
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        assert engine.get_asset_count() == 20

    def test_concurrent_assessments(self, engine, sample_location):
        """Concurrent assessments are safe."""
        engine.register_asset("C-001", "Concurrent", "FACILITY", sample_location)
        errors = []

        def assess(i):
            try:
                engine.assess_exposure(
                    asset_id="C-001",
                    hazard_type="DROUGHT",
                    hazard_intensity=0.5,
                    hazard_probability=0.5,
                    hazard_frequency=0.5,
                )
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=assess, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        assert engine.get_assessment_count() == 20


# ===================================================================
# Edge case tests
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_assessment_id_unique(self, engine, registered_asset):
        """Each assessment gets a unique ID."""
        ids = set()
        for _ in range(10):
            result = engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
            ids.add(result["assessment_id"])
        assert len(ids) == 10

    def test_get_assessments_for_asset(self, engine, registered_asset):
        """get_assessments_for_asset returns all assessments."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        engine.assess_exposure("A-001", "WILDFIRE", 0.5, 0.5, 0.5)
        assessments = engine.get_assessments_for_asset("A-001")
        assert len(assessments) == 2

    def test_get_assessments_for_hazard(self, engine, registered_asset):
        """get_assessments_for_hazard returns filtered assessments."""
        engine.assess_exposure("A-001", "DROUGHT", 0.5, 0.5, 0.5)
        engine.assess_exposure("A-001", "WILDFIRE", 0.5, 0.5, 0.5)
        assessments = engine.get_assessments_for_hazard("DROUGHT")
        assert all(a["hazard_type"] == "DROUGHT" for a in assessments)

    def test_get_worst_exposure_for_asset(self, engine, registered_asset):
        """get_worst_exposure returns highest-scoring assessment."""
        engine.assess_exposure("A-001", "DROUGHT", 0.1, 0.1, 0.1)
        engine.assess_exposure("A-001", "WILDFIRE", 0.9, 0.9, 0.9)
        worst = engine.get_worst_exposure_for_asset("A-001")
        assert worst is not None
        assert worst["hazard_type"] == "WILDFIRE"

    def test_get_worst_exposure_no_assessments(self, engine):
        """get_worst_exposure returns None when no assessments."""
        assert engine.get_worst_exposure_for_asset("DOES_NOT_EXIST") is None

    def test_list_assessments_filter_exposure_level(self, engine, registered_asset):
        """list_assessments can filter by exposure level."""
        engine.assess_exposure("A-001", "DROUGHT", 0.1, 0.1, 0.1)
        assessments = engine.list_assessments(exposure_level="NONE")
        # Scores that low are classified NONE
        for a in assessments:
            assert a["exposure_level"] == "NONE"

    def test_get_assessment_empty_id(self, engine):
        """get_assessment returns None for empty ID."""
        assert engine.get_assessment("") is None

    def test_get_assessment_non_existent(self, engine):
        """get_assessment returns None for non-existent ID."""
        assert engine.get_assessment("NON_EXISTENT") is None
