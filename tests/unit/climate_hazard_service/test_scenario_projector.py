# -*- coding: utf-8 -*-
"""
Unit tests for ScenarioProjectorEngine (Engine 3 of 7).

AGENT-DATA-020: Climate Hazard Connector
Tests scenario projection, warming delta calculations, scaling factors,
multi-scenario comparison, time-series projections, import/export, and
engine lifecycle management.

Target: 85%+ code coverage with 90+ test functions.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.climate_hazard.scenario_projector import (
    ScenarioProjectorEngine,
    _HAZARD_DURATION_FACTORS,
    _HAZARD_PROBABILITY_FACTORS,
    _HAZARD_SCALING_FACTORS,
    _SCENARIO_REGISTRY,
    _TIME_HORIZON_REGISTRY,
    _VALID_HAZARD_TYPES,
    _VALID_SCENARIO_IDS,
    _VALID_TIME_HORIZONS,
    _validate_baseline_risk,
    _validate_hazard_type,
    _validate_location,
    _validate_scenario,
    _validate_time_horizon,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ScenarioProjectorEngine:
    """Return a fresh ScenarioProjectorEngine for each test."""
    return ScenarioProjectorEngine()


@pytest.fixture
def baseline_risk() -> Dict[str, float]:
    """Standard baseline risk for testing."""
    return {
        "probability": 0.15,
        "intensity": 0.6,
        "frequency": 2.0,
        "duration_days": 5.0,
    }


@pytest.fixture
def location() -> Dict[str, Any]:
    """Standard location for testing."""
    return {"lat": 40.7128, "lon": -74.0060, "name": "New York"}


@pytest.fixture
def zero_baseline() -> Dict[str, float]:
    """Baseline risk with all zeros."""
    return {
        "probability": 0.0,
        "intensity": 0.0,
        "frequency": 0.0,
        "duration_days": 0.0,
    }


@pytest.fixture
def high_baseline() -> Dict[str, float]:
    """Baseline risk with high values."""
    return {
        "probability": 0.95,
        "intensity": 0.9,
        "frequency": 10.0,
        "duration_days": 30.0,
    }


# ===================================================================
# Module-level validation function tests
# ===================================================================


class TestValidateScenario:
    """Tests for _validate_scenario."""

    def test_canonical_ssp_identifiers(self):
        """All canonical SSP identifiers are accepted."""
        for sid in ["ssp1_1.9", "ssp1_2.6", "ssp2_4.5", "ssp3_7.0", "ssp5_8.5"]:
            assert _validate_scenario(sid) == sid

    def test_canonical_rcp_identifiers(self):
        """All canonical RCP identifiers are accepted."""
        for sid in ["rcp2.6", "rcp4.5", "rcp8.5"]:
            assert _validate_scenario(sid) == sid

    def test_case_insensitive(self):
        """Scenario validation is case insensitive."""
        assert _validate_scenario("SSP2_4.5") == "ssp2_4.5"
        assert _validate_scenario("RCP8.5") == "rcp8.5"

    def test_hyphen_normalisation(self):
        """Hyphens are normalised to underscores."""
        assert _validate_scenario("ssp2-4.5") == "ssp2_4.5"

    def test_display_name_accepted(self):
        """Display names like 'SSP2-4.5' are accepted."""
        assert _validate_scenario("SSP2-4.5") == "ssp2_4.5"

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_scenario("")

    def test_none_raises(self):
        """None raises ValueError."""
        with pytest.raises(ValueError):
            _validate_scenario(None)

    def test_unknown_raises(self):
        """Unknown scenario raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            _validate_scenario("ssp9_9.9")

    def test_whitespace_stripped(self):
        """Leading and trailing whitespace is stripped."""
        assert _validate_scenario("  ssp2_4.5  ") == "ssp2_4.5"


class TestValidateTimeHorizon:
    """Tests for _validate_time_horizon."""

    def test_all_horizons_valid(self):
        """All five time horizons are accepted."""
        for h in ["BASELINE", "NEAR_TERM", "MID_TERM", "LONG_TERM", "END_CENTURY"]:
            assert _validate_time_horizon(h) == h

    def test_case_insensitive(self):
        """Time horizon validation is case insensitive."""
        assert _validate_time_horizon("mid_term") == "MID_TERM"
        assert _validate_time_horizon("baseline") == "BASELINE"

    def test_empty_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_time_horizon("")

    def test_unknown_raises(self):
        """Unknown horizon raises ValueError."""
        with pytest.raises(ValueError, match="Unknown time_horizon"):
            _validate_time_horizon("DISTANT_FUTURE")


class TestValidateHazardType:
    """Tests for _validate_hazard_type."""

    def test_all_12_hazard_types_valid(self):
        """All 12 hazard types are accepted."""
        for ht in _VALID_HAZARD_TYPES:
            assert _validate_hazard_type(ht) == ht

    def test_case_insensitive(self):
        """Hazard type validation is case insensitive."""
        assert _validate_hazard_type("extreme_heat") == "EXTREME_HEAT"
        assert _validate_hazard_type("Drought") == "DROUGHT"

    def test_hyphen_normalised(self):
        """Hyphens are normalised to underscores."""
        assert _validate_hazard_type("EXTREME-HEAT") == "EXTREME_HEAT"

    def test_empty_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            _validate_hazard_type("")

    def test_unknown_raises(self):
        """Unknown hazard type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hazard_type"):
            _validate_hazard_type("EARTHQUAKE")


class TestValidateBaselineRisk:
    """Tests for _validate_baseline_risk."""

    def test_valid_baseline(self, baseline_risk):
        """Valid baseline passes without error."""
        result = _validate_baseline_risk(baseline_risk)
        assert result["probability"] == pytest.approx(0.15)
        assert result["intensity"] == pytest.approx(0.6)

    def test_probability_clamped_to_01(self):
        """Probability is clamped to [0.0, 1.0]."""
        result = _validate_baseline_risk({
            "probability": 1.5, "intensity": 0.5,
            "frequency": 1.0, "duration_days": 1.0,
        })
        assert result["probability"] == pytest.approx(1.0)

    def test_missing_keys_filled_from_defaults(self):
        """Missing keys are filled with defaults."""
        result = _validate_baseline_risk({"probability": 0.5})
        assert "intensity" in result
        assert result["intensity"] == 0.0

    def test_empty_dict_raises(self):
        """Empty dict raises ValueError."""
        with pytest.raises(ValueError):
            _validate_baseline_risk({})

    def test_none_raises(self):
        """None raises ValueError."""
        with pytest.raises(ValueError):
            _validate_baseline_risk(None)

    def test_non_numeric_value_raises(self):
        """Non-numeric value raises ValueError."""
        with pytest.raises(ValueError, match="numeric"):
            _validate_baseline_risk({"probability": "abc", "intensity": 0.5,
                                     "frequency": 1.0, "duration_days": 1.0})

    def test_negative_intensity_raises(self):
        """Negative intensity raises ValueError."""
        with pytest.raises(ValueError, match=">= 0.0"):
            _validate_baseline_risk({"probability": 0.5, "intensity": -1.0,
                                     "frequency": 1.0, "duration_days": 1.0})


class TestValidateLocation:
    """Tests for _validate_location."""

    def test_dict_location_passthrough(self, location):
        """Dict location is returned as-is."""
        result = _validate_location(location)
        assert result["lat"] == 40.7128

    def test_string_location(self):
        """String location is wrapped in a name dict."""
        result = _validate_location("New York")
        assert result == {"name": "New York"}

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _validate_location("")

    def test_none_raises(self):
        """None raises ValueError."""
        with pytest.raises(ValueError, match="not be None"):
            _validate_location(None)

    def test_numeric_location_converted(self):
        """Numeric location is converted to a dict with identifier key."""
        result = _validate_location(42)
        assert result == {"identifier": "42"}


# ===================================================================
# ScenarioProjectorEngine initialisation tests
# ===================================================================


class TestEngineInit:
    """Tests for engine construction."""

    def test_default_init(self, engine):
        """Default engine initialises with zero projections."""
        stats = engine.get_statistics()
        assert stats["total_projections"] == 0
        assert stats["stored_projections"] == 0

    def test_init_with_risk_engine(self):
        """Engine accepts an optional risk_engine reference."""
        mock_risk = MagicMock()
        eng = ScenarioProjectorEngine(risk_engine=mock_risk)
        stats = eng.get_statistics()
        assert stats["risk_engine_attached"] is True

    def test_init_without_risk_engine(self, engine):
        """Engine operates standalone when risk_engine is None."""
        stats = engine.get_statistics()
        assert stats["risk_engine_attached"] is False

    def test_init_provenance_disabled(self):
        """Provenance can be explicitly disabled with False."""
        eng = ScenarioProjectorEngine(provenance=False)
        stats = eng.get_statistics()
        assert stats["provenance_enabled"] is False

    def test_init_custom_genesis_hash(self):
        """Custom genesis hash is accepted."""
        eng = ScenarioProjectorEngine(genesis_hash="my-custom-genesis")
        assert eng is not None

    def test_supported_counts(self, engine):
        """Engine reports correct supported counts."""
        stats = engine.get_statistics()
        assert stats["supported_scenarios"] == 8
        assert stats["supported_horizons"] == 5
        assert stats["supported_hazard_types"] == 12


# ===================================================================
# project_hazard tests
# ===================================================================


class TestProjectHazard:
    """Tests for project_hazard."""

    def test_basic_projection(self, engine, baseline_risk, location):
        """Basic projection returns expected keys."""
        result = engine.project_hazard(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        assert "projection_id" in result
        assert result["hazard_type"] == "EXTREME_HEAT"
        assert result["scenario"] == "ssp2_4.5"
        assert result["time_horizon"] == "MID_TERM"
        assert result["warming_delta_c"] > 0.0
        assert "projected_risk" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_projected_probability_increases_for_heat(self, engine, baseline_risk, location):
        """Probability increases for EXTREME_HEAT under warming."""
        result = engine.project_hazard(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp3_7.0",
            time_horizon="END_CENTURY",
        )
        assert result["projected_risk"]["probability"] > baseline_risk["probability"]

    def test_projected_intensity_decreases_for_cold(self, engine, location):
        """Intensity decreases for EXTREME_COLD under warming."""
        baseline = {"probability": 0.5, "intensity": 0.8,
                    "frequency": 4.0, "duration_days": 10.0}
        result = engine.project_hazard(
            hazard_type="EXTREME_COLD",
            location=location,
            baseline_risk=baseline,
            scenario="ssp5_8.5",
            time_horizon="END_CENTURY",
        )
        # EXTREME_COLD intensity_factor is 0.7 (< 1.0), so intensity decreases
        assert result["projected_risk"]["intensity"] < baseline["intensity"]

    def test_baseline_horizon_returns_zero_delta(self, engine, baseline_risk, location):
        """BASELINE horizon produces zero warming delta."""
        result = engine.project_hazard(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="BASELINE",
        )
        assert result["warming_delta_c"] == pytest.approx(0.0)
        # Projected risk should equal baseline with zero warming
        assert result["projected_risk"]["probability"] == pytest.approx(
            baseline_risk["probability"], rel=1e-4
        )

    def test_invalid_hazard_raises(self, engine, baseline_risk, location):
        """Invalid hazard type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hazard_type"):
            engine.project_hazard(
                hazard_type="EARTHQUAKE",
                location=location,
                baseline_risk=baseline_risk,
                scenario="ssp2_4.5",
                time_horizon="MID_TERM",
            )

    def test_invalid_scenario_raises(self, engine, baseline_risk, location):
        """Invalid scenario raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            engine.project_hazard(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenario="INVALID",
                time_horizon="MID_TERM",
            )

    def test_invalid_horizon_raises(self, engine, baseline_risk, location):
        """Invalid time horizon raises ValueError."""
        with pytest.raises(ValueError, match="Unknown time_horizon"):
            engine.project_hazard(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenario="ssp2_4.5",
                time_horizon="FAR_FUTURE",
            )

    def test_projection_stored(self, engine, baseline_risk, location):
        """Projection is stored and retrievable."""
        result = engine.project_hazard(
            hazard_type="WILDFIRE",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="NEAR_TERM",
        )
        stored = engine.get_projection(result["projection_id"])
        assert stored is not None
        assert stored["projection_id"] == result["projection_id"]

    def test_projection_deep_copied(self, engine, baseline_risk, location):
        """Returned projection is a deep copy."""
        result = engine.project_hazard(
            hazard_type="WILDFIRE",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="NEAR_TERM",
        )
        result["hazard_type"] = "MODIFIED"
        stored = engine.get_projection(result["projection_id"])
        assert stored["hazard_type"] == "WILDFIRE"  # Unchanged

    def test_stats_increment(self, engine, baseline_risk, location):
        """Statistics counter increments on projection."""
        engine.project_hazard(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp1_2.6",
            time_horizon="MID_TERM",
        )
        stats = engine.get_statistics()
        assert stats["total_projections"] == 1
        assert stats["projections_by_scenario"]["ssp1_2.6"] == 1
        assert stats["projections_by_hazard"]["DROUGHT"] == 1
        assert stats["projections_by_horizon"]["MID_TERM"] == 1

    def test_risk_change_pct_included(self, engine, baseline_risk, location):
        """Result includes risk_change_pct with expected keys."""
        result = engine.project_hazard(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        pct = result["risk_change_pct"]
        assert "probability_change_pct" in pct
        assert "intensity_change_pct" in pct
        assert "frequency_change_pct" in pct
        assert "duration_days_change_pct" in pct

    def test_scenario_info_included(self, engine, baseline_risk, location):
        """Result includes scenario_info metadata."""
        result = engine.project_hazard(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        assert result["scenario_info"]["name"] == "SSP2-4.5"
        assert result["scenario_info"]["warming_by_2100"] == 2.7

    def test_horizon_info_included(self, engine, baseline_risk, location):
        """Result includes horizon_info metadata."""
        result = engine.project_hazard(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        assert result["horizon_info"]["warming_fraction"] == 0.55

    def test_string_location_accepted(self, engine, baseline_risk):
        """String location is accepted."""
        result = engine.project_hazard(
            hazard_type="DROUGHT",
            location="London",
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        assert result["location"]["name"] == "London"

    def test_all_12_hazard_types(self, engine, baseline_risk, location):
        """All 12 hazard types produce valid projections."""
        for ht in _VALID_HAZARD_TYPES:
            result = engine.project_hazard(
                hazard_type=ht,
                location=location,
                baseline_risk=baseline_risk,
                scenario="ssp2_4.5",
                time_horizon="MID_TERM",
            )
            assert result["hazard_type"] == ht

    def test_all_8_scenarios(self, engine, baseline_risk, location):
        """All 8 scenarios produce valid projections."""
        for sid in _VALID_SCENARIO_IDS:
            result = engine.project_hazard(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenario=sid,
                time_horizon="MID_TERM",
            )
            assert result["scenario"] == sid

    def test_all_5_time_horizons(self, engine, baseline_risk, location):
        """All 5 time horizons produce valid projections."""
        for h in _VALID_TIME_HORIZONS:
            result = engine.project_hazard(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenario="ssp2_4.5",
                time_horizon=h,
            )
            assert result["time_horizon"] == h

    def test_zero_baseline_no_amplification(self, engine, zero_baseline, location):
        """Zero baseline produces zero projected risk."""
        result = engine.project_hazard(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=zero_baseline,
            scenario="ssp5_8.5",
            time_horizon="END_CENTURY",
        )
        # With zero baseline, scaling should yield near-zero or zero
        assert result["projected_risk"]["probability"] == pytest.approx(0.0, abs=1e-6)
        assert result["projected_risk"]["frequency"] == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# project_multi_scenario tests
# ===================================================================


class TestProjectMultiScenario:
    """Tests for project_multi_scenario."""

    def test_two_scenarios(self, engine, baseline_risk, location):
        """Multi-scenario with two scenarios returns comparison."""
        result = engine.project_multi_scenario(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=["ssp1_2.6", "ssp5_8.5"],
            time_horizon="END_CENTURY",
        )
        assert result["scenarios_count"] == 2
        assert len(result["per_scenario_projections"]) == 2
        assert "warming_range" in result
        assert result["warming_range"]["max_warming_c"] > result["warming_range"]["min_warming_c"]

    def test_all_8_scenarios_comparison(self, engine, baseline_risk, location):
        """All 8 scenarios can be compared in one call."""
        result = engine.project_multi_scenario(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=sorted(_VALID_SCENARIO_IDS),
            time_horizon="MID_TERM",
        )
        assert result["scenarios_count"] == 8

    def test_empty_scenarios_raises(self, engine, baseline_risk, location):
        """Empty scenarios list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            engine.project_multi_scenario(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenarios=[],
                time_horizon="MID_TERM",
            )

    def test_all_invalid_scenarios_raises(self, engine, baseline_risk, location):
        """All invalid scenarios raises ValueError."""
        with pytest.raises(ValueError, match="No valid scenarios"):
            engine.project_multi_scenario(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenarios=["INVALID1", "INVALID2"],
                time_horizon="MID_TERM",
            )

    def test_partial_invalid_scenarios_skipped(self, engine, baseline_risk, location):
        """Invalid scenarios are skipped, valid ones projected."""
        result = engine.project_multi_scenario(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=["ssp2_4.5", "INVALID"],
            time_horizon="MID_TERM",
        )
        assert result["scenarios_count"] == 1

    def test_scenario_comparison_sorted(self, engine, baseline_risk, location):
        """Scenario comparison is sorted by risk change."""
        result = engine.project_multi_scenario(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=["ssp1_1.9", "ssp5_8.5"],
            time_horizon="END_CENTURY",
        )
        assert "scenario_comparison" in result
        assert len(result["scenario_comparison"]) == 2

    def test_stats_multi_incremented(self, engine, baseline_risk, location):
        """Multi-scenario counter is incremented."""
        engine.project_multi_scenario(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=["ssp2_4.5"],
            time_horizon="MID_TERM",
        )
        stats = engine.get_statistics()
        assert stats["total_multi_scenario"] == 1

    def test_provenance_hash_present(self, engine, baseline_risk, location):
        """Multi-scenario result includes provenance hash."""
        result = engine.project_multi_scenario(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=["ssp2_4.5"],
            time_horizon="MID_TERM",
        )
        assert len(result["provenance_hash"]) == 64


# ===================================================================
# project_time_series tests
# ===================================================================


class TestProjectTimeSeries:
    """Tests for project_time_series."""

    def test_default_all_horizons(self, engine, baseline_risk, location):
        """Default time series projects across all 5 horizons."""
        result = engine.project_time_series(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
        )
        assert result["horizons_count"] == 5
        assert len(result["time_series"]) == 5

    def test_custom_horizons(self, engine, baseline_risk, location):
        """Custom horizons list is respected."""
        result = engine.project_time_series(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizons=["NEAR_TERM", "MID_TERM"],
        )
        assert result["horizons_count"] == 2

    def test_warming_trajectory_included(self, engine, baseline_risk, location):
        """Time series includes warming_trajectory summary."""
        result = engine.project_time_series(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp3_7.0",
        )
        trajectory = result["warming_trajectory"]
        assert len(trajectory) >= 1
        assert "warming_delta_c" in trajectory[0]

    def test_trend_direction(self, engine, baseline_risk, location):
        """Trend direction is one of the expected values."""
        result = engine.project_time_series(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp5_8.5",
        )
        assert result["trend_direction"] in (
            "increasing", "decreasing", "stable", "non_monotonic"
        )

    def test_empty_horizons_list_raises(self, engine, baseline_risk, location):
        """Empty horizons list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            engine.project_time_series(
                hazard_type="EXTREME_HEAT",
                location=location,
                baseline_risk=baseline_risk,
                scenario="ssp2_4.5",
                time_horizons=[],
            )

    def test_all_invalid_horizons_raises(self, engine, baseline_risk, location):
        """All invalid horizons raises ValueError."""
        with pytest.raises(ValueError, match="No valid time horizons"):
            engine.project_time_series(
                hazard_type="DROUGHT",
                location=location,
                baseline_risk=baseline_risk,
                scenario="ssp2_4.5",
                time_horizons=["INVALID1", "INVALID2"],
            )

    def test_stats_time_series_incremented(self, engine, baseline_risk, location):
        """Time series counter is incremented."""
        engine.project_time_series(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
        )
        stats = engine.get_statistics()
        assert stats["total_time_series"] == 1

    def test_sorted_by_warming(self, engine, baseline_risk, location):
        """Time series projections are sorted by warming delta."""
        result = engine.project_time_series(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
        )
        deltas = [p["warming_delta_c"] for p in result["time_series"]]
        assert deltas == sorted(deltas)


# ===================================================================
# calculate_warming_delta tests
# ===================================================================


class TestCalculateWarmingDelta:
    """Tests for calculate_warming_delta."""

    def test_ssp2_mid_term(self, engine):
        """SSP2-4.5 MID_TERM: 2.7 * 0.55 = 1.485."""
        delta = engine.calculate_warming_delta("ssp2_4.5", "MID_TERM")
        assert delta == pytest.approx(1.485, rel=1e-3)

    def test_baseline_zero(self, engine):
        """BASELINE always produces zero warming."""
        for sid in _VALID_SCENARIO_IDS:
            delta = engine.calculate_warming_delta(sid, "BASELINE")
            assert delta == pytest.approx(0.0)

    def test_end_century_equals_warming_2100(self, engine):
        """END_CENTURY warming fraction is 1.0 so delta == warming_by_2100."""
        for sid in _VALID_SCENARIO_IDS:
            delta = engine.calculate_warming_delta(sid, "END_CENTURY")
            expected = _SCENARIO_REGISTRY[sid]["warming_by_2100"]
            assert delta == pytest.approx(expected, rel=1e-4)

    def test_monotonic_warming_across_horizons(self, engine):
        """Warming delta increases across horizons for any scenario."""
        for sid in _VALID_SCENARIO_IDS:
            deltas = []
            for h in ["BASELINE", "NEAR_TERM", "MID_TERM", "LONG_TERM", "END_CENTURY"]:
                deltas.append(engine.calculate_warming_delta(sid, h))
            # Should be non-decreasing
            for i in range(1, len(deltas)):
                assert deltas[i] >= deltas[i - 1]

    def test_higher_emission_more_warming(self, engine):
        """Higher emission scenarios produce more warming at same horizon."""
        low = engine.calculate_warming_delta("ssp1_1.9", "END_CENTURY")
        high = engine.calculate_warming_delta("ssp5_8.5", "END_CENTURY")
        assert high > low

    def test_stats_counter_incremented(self, engine):
        """Warming calculations counter is incremented."""
        engine.calculate_warming_delta("ssp2_4.5", "MID_TERM")
        stats = engine.get_statistics()
        assert stats["total_warming_calculations"] >= 1

    def test_rcp_warming_values(self, engine):
        """RCP scenarios produce expected warming values."""
        delta = engine.calculate_warming_delta("rcp8.5", "END_CENTURY")
        assert delta == pytest.approx(4.3, rel=1e-3)


# ===================================================================
# apply_scaling_factors tests
# ===================================================================


class TestApplyScalingFactors:
    """Tests for apply_scaling_factors."""

    def test_extreme_heat_scaling(self, engine, baseline_risk):
        """EXTREME_HEAT scaling amplifies all components."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_HEAT", 1.0)
        assert projected["probability"] > baseline_risk["probability"]
        assert projected["intensity"] > baseline_risk["intensity"]
        assert projected["frequency"] > baseline_risk["frequency"]
        assert projected["duration_days"] > baseline_risk["duration_days"]

    def test_zero_warming_no_change(self, engine, baseline_risk):
        """Zero warming produces no change from baseline."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_HEAT", 0.0)
        assert projected["probability"] == pytest.approx(baseline_risk["probability"], rel=1e-4)
        assert projected["intensity"] == pytest.approx(baseline_risk["intensity"], rel=1e-4)
        assert projected["frequency"] == pytest.approx(baseline_risk["frequency"], rel=1e-4)
        assert projected["duration_days"] == pytest.approx(baseline_risk["duration_days"], rel=1e-4)

    def test_probability_clamped(self, engine, high_baseline):
        """Projected probability is clamped to [0.0, 1.0]."""
        projected = engine.apply_scaling_factors(high_baseline, "EXTREME_HEAT", 5.0)
        assert 0.0 <= projected["probability"] <= 1.0

    def test_frequency_floored_at_zero(self, engine, baseline_risk):
        """Frequency cannot go below zero."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_COLD", 10.0)
        assert projected["frequency"] >= 0.0

    def test_duration_floored_at_zero(self, engine, baseline_risk):
        """Duration cannot go below zero."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_COLD", 10.0)
        assert projected["duration_days"] >= 0.0

    def test_sea_level_rise_additive(self, engine):
        """SEA_LEVEL_RISE uses additive intensity scaling."""
        baseline = {"probability": 0.5, "intensity": 0.3,
                    "frequency": 1.0, "duration_days": 0.0}
        projected = engine.apply_scaling_factors(baseline, "SEA_LEVEL_RISE", 2.0)
        # intensity_factor for SLR is 0.3, additive: 0.3 + 0.3*2 = 0.9
        assert projected["intensity"] > baseline["intensity"]

    def test_extreme_cold_decreasing(self, engine, baseline_risk):
        """EXTREME_COLD intensity and frequency decrease with warming."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_COLD", 2.0)
        assert projected["intensity"] < baseline_risk["intensity"]
        assert projected["frequency"] < baseline_risk["frequency"]

    def test_stats_scaling_incremented(self, engine, baseline_risk):
        """Scaling applications counter is incremented."""
        engine.apply_scaling_factors(baseline_risk, "DROUGHT", 1.0)
        stats = engine.get_statistics()
        assert stats["total_scaling_applications"] >= 1

    def test_result_keys_present(self, engine, baseline_risk):
        """Result contains all four risk components."""
        projected = engine.apply_scaling_factors(baseline_risk, "DROUGHT", 1.0)
        assert set(projected.keys()) == {
            "probability", "intensity", "frequency", "duration_days"
        }


# ===================================================================
# get_scaling_factors tests
# ===================================================================


class TestGetScalingFactors:
    """Tests for get_scaling_factors."""

    def test_extreme_heat_factors(self, engine):
        """EXTREME_HEAT has known scaling factors."""
        factors = engine.get_scaling_factors("EXTREME_HEAT")
        assert factors["hazard_type"] == "EXTREME_HEAT"
        assert factors["intensity_factor"] == 2.0
        assert factors["frequency_factor"] == 1.8

    def test_all_hazard_types_have_factors(self, engine):
        """All 12 hazard types return scaling factors."""
        for ht in _VALID_HAZARD_TYPES:
            factors = engine.get_scaling_factors(ht)
            assert "intensity_factor" in factors
            assert "frequency_factor" in factors
            assert "probability_factor" in factors
            assert "duration_factor" in factors

    def test_direction_indicators(self, engine):
        """Direction indicators are computed correctly."""
        factors = engine.get_scaling_factors("EXTREME_HEAT")
        assert factors["intensity_direction"] == "increasing"
        assert factors["frequency_direction"] == "increasing"

        cold_factors = engine.get_scaling_factors("EXTREME_COLD")
        assert cold_factors["intensity_direction"] == "decreasing"
        assert cold_factors["frequency_direction"] == "decreasing"

    def test_invalid_hazard_raises(self, engine):
        """Invalid hazard type raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_scaling_factors("EARTHQUAKE")


# ===================================================================
# list_scenarios, get_scenario_info tests
# ===================================================================


class TestScenarioListing:
    """Tests for list_scenarios and get_scenario_info."""

    def test_list_scenarios_count(self, engine):
        """list_scenarios returns all 8 scenarios."""
        scenarios = engine.list_scenarios()
        assert len(scenarios) == 8

    def test_list_scenarios_sorted_by_warming(self, engine):
        """Scenarios are sorted by warming_by_2100 ascending."""
        scenarios = engine.list_scenarios()
        warmings = [s["warming_by_2100"] for s in scenarios]
        assert warmings == sorted(warmings)

    def test_get_scenario_info(self, engine):
        """get_scenario_info returns correct metadata."""
        info = engine.get_scenario_info("ssp2_4.5")
        assert info["name"] == "SSP2-4.5"
        assert info["warming_by_2100"] == 2.7
        assert info["pathway"] == "SSP"

    def test_get_scenario_info_invalid(self, engine):
        """Invalid scenario raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_scenario_info("INVALID")


# ===================================================================
# list_projections, get_projection tests
# ===================================================================


class TestProjectionListing:
    """Tests for list_projections and get_projection."""

    def test_list_empty(self, engine):
        """Empty engine returns empty projection list."""
        result = engine.list_projections()
        assert result == []

    def test_list_after_projection(self, engine, baseline_risk, location):
        """list_projections returns stored projections."""
        engine.project_hazard(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        result = engine.list_projections()
        assert len(result) == 1
        assert result[0]["hazard_type"] == "DROUGHT"

    def test_filter_by_hazard_type(self, engine, baseline_risk, location):
        """Filtering by hazard_type works."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        engine.project_hazard("WILDFIRE", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        result = engine.list_projections(hazard_type="DROUGHT")
        assert all(p["hazard_type"] == "DROUGHT" for p in result)

    def test_filter_by_scenario(self, engine, baseline_risk, location):
        """Filtering by scenario works."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp1_2.6", "MID_TERM")
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp5_8.5", "MID_TERM")
        result = engine.list_projections(scenario="ssp1_2.6")
        assert all(p["scenario"] == "ssp1_2.6" for p in result)

    def test_filter_by_horizon(self, engine, baseline_risk, location):
        """Filtering by time_horizon works."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "NEAR_TERM")
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "END_CENTURY")
        result = engine.list_projections(time_horizon="NEAR_TERM")
        assert all(p["time_horizon"] == "NEAR_TERM" for p in result)

    def test_limit_respected(self, engine, baseline_risk, location):
        """Limit parameter is respected."""
        for _ in range(5):
            engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        result = engine.list_projections(limit=3)
        assert len(result) == 3

    def test_get_projection_returns_none_for_missing(self, engine):
        """get_projection returns None for non-existent ID."""
        assert engine.get_projection("NON_EXISTENT") is None

    def test_get_projection_empty_id(self, engine):
        """get_projection returns None for empty ID."""
        assert engine.get_projection("") is None


# ===================================================================
# get_statistics tests
# ===================================================================


class TestGetStatistics:
    """Tests for get_statistics."""

    def test_initial_stats(self, engine):
        """Initial stats are all zeros."""
        stats = engine.get_statistics()
        assert stats["total_projections"] == 0
        assert stats["total_multi_scenario"] == 0
        assert stats["total_time_series"] == 0
        assert stats["total_warming_calculations"] == 0
        assert stats["total_errors"] == 0

    def test_stats_after_operations(self, engine, baseline_risk, location):
        """Stats reflect accumulated operations."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        engine.project_hazard("WILDFIRE", location, baseline_risk, "ssp1_2.6", "NEAR_TERM")
        stats = engine.get_statistics()
        assert stats["total_projections"] == 2
        assert stats["stored_projections"] == 2

    def test_stats_have_meta_counts(self, engine):
        """Stats include supported scenario/horizon/hazard counts."""
        stats = engine.get_statistics()
        assert stats["supported_scenarios"] == 8
        assert stats["supported_horizons"] == 5
        assert stats["supported_hazard_types"] == 12


# ===================================================================
# clear tests
# ===================================================================


class TestClear:
    """Tests for clear."""

    def test_clear_resets_projections(self, engine, baseline_risk, location):
        """clear removes all stored projections."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        summary = engine.clear()
        assert summary["projections_cleared"] >= 1
        stats = engine.get_statistics()
        assert stats["total_projections"] == 0
        assert stats["stored_projections"] == 0

    def test_clear_returns_previous_stats(self, engine, baseline_risk, location):
        """clear returns a summary with previous stats."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        summary = engine.clear()
        assert "previous_stats" in summary
        assert summary["previous_stats"]["total_projections"] == 1

    def test_clear_idempotent(self, engine):
        """Clearing an empty engine is safe."""
        summary = engine.clear()
        assert summary["projections_cleared"] == 0


# ===================================================================
# Time horizon and hazard listing tests
# ===================================================================


class TestListingMethods:
    """Tests for list_time_horizons, get_time_horizon_info, list_hazard_types."""

    def test_list_time_horizons_count(self, engine):
        """list_time_horizons returns all 5 horizons."""
        horizons = engine.list_time_horizons()
        assert len(horizons) == 5

    def test_list_time_horizons_sorted(self, engine):
        """Horizons sorted by period_start ascending."""
        horizons = engine.list_time_horizons()
        starts = [h["period_start"] for h in horizons]
        assert starts == sorted(starts)

    def test_get_time_horizon_info(self, engine):
        """get_time_horizon_info returns correct metadata."""
        info = engine.get_time_horizon_info("MID_TERM")
        assert info["warming_fraction"] == 0.55
        assert info["period_start"] == 2041
        assert info["period_end"] == 2060

    def test_get_time_horizon_info_invalid(self, engine):
        """Invalid horizon raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_time_horizon_info("INVALID")

    def test_list_hazard_types_count(self, engine):
        """list_hazard_types returns all 12 types."""
        hazards = engine.list_hazard_types()
        assert len(hazards) == 12

    def test_list_hazard_types_alphabetical(self, engine):
        """Hazard types are sorted alphabetically."""
        hazards = engine.list_hazard_types()
        names = [h["hazard_type"] for h in hazards]
        assert names == sorted(names)


# ===================================================================
# get_warming_matrix tests
# ===================================================================


class TestGetWarmingMatrix:
    """Tests for get_warming_matrix."""

    def test_matrix_shape(self, engine):
        """Matrix has 8 scenarios x 5 horizons."""
        matrix = engine.get_warming_matrix()
        assert len(matrix) == 8
        for sid in matrix:
            assert len(matrix[sid]) == 5

    def test_matrix_values(self, engine):
        """Matrix values match individual calculations."""
        matrix = engine.get_warming_matrix()
        expected = engine.calculate_warming_delta("ssp2_4.5", "END_CENTURY")
        assert matrix["ssp2_4.5"]["END_CENTURY"] == pytest.approx(expected)

    def test_matrix_deep_copied(self, engine):
        """Matrix is a deep copy."""
        matrix = engine.get_warming_matrix()
        matrix["ssp2_4.5"]["END_CENTURY"] = 999.9
        matrix2 = engine.get_warming_matrix()
        assert matrix2["ssp2_4.5"]["END_CENTURY"] != 999.9


# ===================================================================
# project_all_hazards tests
# ===================================================================


class TestProjectAllHazards:
    """Tests for project_all_hazards."""

    def test_multiple_hazards(self, engine, baseline_risk, location):
        """Projects multiple hazard types."""
        baselines = {
            "EXTREME_HEAT": baseline_risk,
            "DROUGHT": baseline_risk,
        }
        result = engine.project_all_hazards(
            location=location,
            baseline_risks=baselines,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        assert result["hazard_count"] == 2
        assert "EXTREME_HEAT" in result["projections"]
        assert "DROUGHT" in result["projections"]

    def test_invalid_hazard_captured_as_error(self, engine, baseline_risk, location):
        """Invalid hazard types are captured in errors dict."""
        baselines = {
            "EXTREME_HEAT": baseline_risk,
            "EARTHQUAKE": baseline_risk,
        }
        result = engine.project_all_hazards(
            location=location,
            baseline_risks=baselines,
            scenario="ssp2_4.5",
            time_horizon="MID_TERM",
        )
        assert result["hazard_count"] == 1
        assert "EARTHQUAKE" in result["errors"]

    def test_empty_baselines_raises(self, engine, location):
        """Empty baseline_risks dict raises ValueError."""
        with pytest.raises(ValueError, match="non-empty dictionary"):
            engine.project_all_hazards(
                location=location,
                baseline_risks={},
                scenario="ssp2_4.5",
                time_horizon="MID_TERM",
            )


# ===================================================================
# compare_scenarios_over_time tests
# ===================================================================


class TestCompareScenariosOverTime:
    """Tests for compare_scenarios_over_time."""

    def test_full_matrix(self, engine, baseline_risk, location):
        """Full comparison matrix produced."""
        result = engine.compare_scenarios_over_time(
            hazard_type="DROUGHT",
            location=location,
            baseline_risk=baseline_risk,
            scenarios=["ssp1_1.9", "ssp5_8.5"],
        )
        assert "matrix" in result
        assert "summary" in result
        assert result["summary"]["scenarios_count"] == 2
        assert result["summary"]["horizons_count"] == 5

    def test_default_all_scenarios(self, engine, baseline_risk, location):
        """None scenarios defaults to all 8."""
        result = engine.compare_scenarios_over_time(
            hazard_type="EXTREME_HEAT",
            location=location,
            baseline_risk=baseline_risk,
        )
        assert result["summary"]["scenarios_count"] == 8


# ===================================================================
# export_projections / import_projections tests
# ===================================================================


class TestExportImport:
    """Tests for export_projections and import_projections."""

    def test_export_json(self, engine, baseline_risk, location):
        """Export as JSON returns valid JSON string."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        exported = engine.export_projections(format="json")
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_dict(self, engine, baseline_risk, location):
        """Export as dict returns list of dicts."""
        engine.project_hazard("DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM")
        exported = engine.export_projections(format="dict")
        assert isinstance(exported, list)
        assert len(exported) == 1

    def test_export_invalid_format_raises(self, engine):
        """Invalid export format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            engine.export_projections(format="xml")

    def test_export_empty(self, engine):
        """Exporting empty engine returns empty list."""
        exported = engine.export_projections(format="dict")
        assert exported == []


# ===================================================================
# Thread safety tests
# ===================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_projections(self, engine, baseline_risk, location):
        """Multiple threads can project concurrently without error."""
        errors = []

        def worker(scenario):
            try:
                engine.project_hazard(
                    hazard_type="DROUGHT",
                    location=location,
                    baseline_risk=baseline_risk,
                    scenario=scenario,
                    time_horizon="MID_TERM",
                )
            except Exception as exc:
                errors.append(str(exc))

        threads = []
        scenarios = sorted(_VALID_SCENARIO_IDS)
        for s in scenarios:
            t = threading.Thread(target=worker, args=(s,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["total_projections"] == 8

    def test_concurrent_warming_calculations(self, engine):
        """Concurrent warming calculations are thread safe."""
        results = {}

        def calc(sid, horizon):
            results[(sid, horizon)] = engine.calculate_warming_delta(sid, horizon)

        threads = []
        for sid in sorted(_VALID_SCENARIO_IDS):
            for h in _VALID_TIME_HORIZONS:
                t = threading.Thread(target=calc, args=(sid, h))
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=10.0)

        assert len(results) == 8 * 5


# ===================================================================
# Edge case tests
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_very_high_warming_delta(self, engine, baseline_risk):
        """Very high warming delta does not cause math errors."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_HEAT", 100.0)
        assert 0.0 <= projected["probability"] <= 1.0
        assert projected["intensity"] >= 0.0
        assert projected["frequency"] >= 0.0
        assert projected["duration_days"] >= 0.0

    def test_negative_warming_delta(self, engine, baseline_risk):
        """Negative warming delta (cooling) is handled."""
        projected = engine.apply_scaling_factors(baseline_risk, "EXTREME_HEAT", -1.0)
        # With negative delta, heat intensity should decrease
        assert projected["intensity"] < baseline_risk["intensity"]

    def test_registry_data_completeness(self):
        """All scenario registry entries have required keys."""
        required_keys = {"name", "pathway", "forcing", "warming_by_2100", "description",
                         "ipcc_report", "narrative", "emission_trajectory"}
        for sid, meta in _SCENARIO_REGISTRY.items():
            for key in required_keys:
                assert key in meta, f"Missing '{key}' in scenario '{sid}'"

    def test_time_horizon_registry_completeness(self):
        """All time horizon entries have required keys."""
        required_keys = {"name", "period_start", "period_end", "warming_fraction", "description"}
        for hid, meta in _TIME_HORIZON_REGISTRY.items():
            for key in required_keys:
                assert key in meta, f"Missing '{key}' in horizon '{hid}'"

    def test_hazard_scaling_factors_completeness(self):
        """All hazard types have intensity and frequency factors."""
        for ht, factors in _HAZARD_SCALING_FACTORS.items():
            assert "intensity_factor" in factors, f"Missing intensity_factor for {ht}"
            assert "frequency_factor" in factors, f"Missing frequency_factor for {ht}"
            assert ht in _HAZARD_PROBABILITY_FACTORS, f"Missing probability factor for {ht}"
            assert ht in _HAZARD_DURATION_FACTORS, f"Missing duration factor for {ht}"

    def test_projection_id_unique(self, engine, baseline_risk, location):
        """Each projection gets a unique ID."""
        ids = set()
        for _ in range(10):
            result = engine.project_hazard(
                "DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM"
            )
            ids.add(result["projection_id"])
        assert len(ids) == 10

    def test_many_projections(self, engine, baseline_risk, location):
        """Engine handles many projections without error."""
        for _ in range(50):
            engine.project_hazard(
                "DROUGHT", location, baseline_risk, "ssp2_4.5", "MID_TERM"
            )
        stats = engine.get_statistics()
        assert stats["total_projections"] == 50
        assert stats["stored_projections"] == 50
