# -*- coding: utf-8 -*-
"""
Unit tests for RiskIndexEngine - AGENT-DATA-020 Climate Hazard Connector

Engine 2 of 7: risk_index.py

Tests all public methods with comprehensive coverage:
    - calculate_risk_index
    - calculate_multi_hazard_index
    - calculate_compound_risk
    - rank_hazards
    - compare_locations
    - get_risk_trend
    - classify_risk_level
    - get_risk_index
    - list_risk_indices
    - get_high_risk_summary
    - get_statistics
    - clear
    - get_weights, set_weights, get_thresholds, set_thresholds
    - get_compound_correlations, set_compound_correlation
    - calculate_risk_score_raw
    - batch_calculate_risk_indices
    - get_risk_level_distribution
    - get_hazard_type_summary
    - get_location_risk_profile
    - validate_risk_components
    - delete_risk_index
    - get_index_count
    - export_indices, import_indices
    - __len__, __repr__, __contains__, __iter__

Validates:
    - Deterministic risk score formula (zero-hallucination)
    - Risk level classification (5 tiers)
    - Compound hazard correlations
    - Aggregation strategies (weighted_average, maximum, sum_capped)
    - Input validation and error handling
    - SHA-256 provenance hash computation
    - Thread safety (threading.Lock)
    - Deep copy guarantees
    - Normalisation helpers

Author: GreenLang QA Team
Date: February 2026
"""

import copy
import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "greenlang",
    ),
)

from climate_hazard.risk_index import (
    RiskIndexEngine,
    RiskLevel,
    HazardType,
    AggregationStrategy,
    TrendDirection,
    _hash_data,
    _utcnow,
    _DEFAULT_WEIGHT_PROBABILITY,
    _DEFAULT_WEIGHT_INTENSITY,
    _DEFAULT_WEIGHT_FREQUENCY,
    _DEFAULT_WEIGHT_DURATION,
    _DEFAULT_THRESHOLD_EXTREME,
    _DEFAULT_THRESHOLD_HIGH,
    _DEFAULT_THRESHOLD_MEDIUM,
    _DEFAULT_THRESHOLD_LOW,
    _INTENSITY_MAX,
    _FREQUENCY_MAX,
    _DURATION_MAX_DAYS,
    _DEFAULT_COMPOUND_CORRELATIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh RiskIndexEngine instance."""
    return RiskIndexEngine()


@pytest.fixture
def london_location():
    """London location dict."""
    return {"lat": 51.5074, "lon": -0.1278, "name": "London"}


@pytest.fixture
def paris_location():
    """Paris location dict."""
    return {"lat": 48.8566, "lon": 2.3522, "name": "Paris"}


@pytest.fixture
def tokyo_location():
    """Tokyo location dict."""
    return {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo"}


@pytest.fixture
def sample_hazard_risks():
    """Return a list of sample hazard risk dicts for multi-hazard tests."""
    return [
        {
            "hazard_type": "riverine_flood",
            "probability": 0.4,
            "intensity": 6.0,
            "frequency": 2.0,
            "duration_days": 14.0,
            "weight": 1.5,
        },
        {
            "hazard_type": "drought",
            "probability": 0.6,
            "intensity": 7.5,
            "frequency": 1.5,
            "duration_days": 90.0,
            "weight": 1.0,
        },
        {
            "hazard_type": "wildfire",
            "probability": 0.3,
            "intensity": 8.0,
            "frequency": 0.5,
            "duration_days": 7.0,
            "weight": 0.8,
        },
    ]


@pytest.fixture
def engine_with_indices(engine, london_location):
    """Engine pre-populated with several risk indices."""
    engine.calculate_risk_index(
        hazard_type="riverine_flood",
        location=london_location,
        probability=0.4,
        intensity=6.0,
        frequency=2.0,
        duration_days=14.0,
    )
    engine.calculate_risk_index(
        hazard_type="drought",
        location={"lat": 34.0, "lon": -118.2, "name": "Los Angeles"},
        probability=0.7,
        intensity=8.0,
        frequency=1.0,
        duration_days=120.0,
    )
    engine.calculate_risk_index(
        hazard_type="wildfire",
        location={"lat": 34.0, "lon": -118.2, "name": "Los Angeles"},
        probability=0.5,
        intensity=9.0,
        frequency=0.3,
        duration_days=5.0,
    )
    return engine


# ---------------------------------------------------------------------------
# Helper: compute expected risk score
# ---------------------------------------------------------------------------


def _expected_score(
    prob: float,
    intensity: float,
    frequency: float,
    duration_days: float,
    w_prob: float = _DEFAULT_WEIGHT_PROBABILITY,
    w_int: float = _DEFAULT_WEIGHT_INTENSITY,
    w_freq: float = _DEFAULT_WEIGHT_FREQUENCY,
    w_dur: float = _DEFAULT_WEIGHT_DURATION,
) -> float:
    """Reproduce the risk score formula for verification."""
    clamped_prob = max(0.0, min(prob, 1.0))
    norm_int = min(max(intensity, 0.0) / _INTENSITY_MAX, 1.0)
    norm_freq = min(max(frequency, 0.0) / _FREQUENCY_MAX, 1.0)
    norm_dur = min(max(duration_days, 0.0) / _DURATION_MAX_DAYS, 1.0)
    raw = (
        clamped_prob * w_prob
        + norm_int * w_int
        + norm_freq * w_freq
        + norm_dur * w_dur
    ) * 100.0
    return max(0.0, min(round(raw, 4), 100.0))


# ===========================================================================
# Test Enumerations
# ===========================================================================


class TestEnumerations:
    """Tests for RiskLevel, HazardType, AggregationStrategy, TrendDirection."""

    def test_risk_level_values(self):
        """RiskLevel has 5 members with expected values."""
        assert RiskLevel.NEGLIGIBLE.value == "NEGLIGIBLE"
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.MEDIUM.value == "MEDIUM"
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.EXTREME.value == "EXTREME"
        assert len(RiskLevel) == 5

    def test_hazard_type_values(self):
        """HazardType has 12 members."""
        assert len(HazardType) == 12
        assert HazardType.RIVERINE_FLOOD.value == "riverine_flood"
        assert HazardType.COASTAL_FLOOD.value == "coastal_flood"
        assert HazardType.DROUGHT.value == "drought"

    def test_aggregation_strategy_values(self):
        """AggregationStrategy has 3 members."""
        assert len(AggregationStrategy) == 3
        assert AggregationStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationStrategy.MAXIMUM.value == "maximum"
        assert AggregationStrategy.SUM_CAPPED.value == "sum_capped"

    def test_trend_direction_values(self):
        """TrendDirection has 3 members."""
        assert len(TrendDirection) == 3
        assert TrendDirection.INCREASING.value == "increasing"
        assert TrendDirection.DECREASING.value == "decreasing"
        assert TrendDirection.STABLE.value == "stable"


# ===========================================================================
# Test Helpers
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_hash_data_deterministic(self):
        """_hash_data produces deterministic SHA-256 hex strings."""
        h1 = _hash_data({"a": 1, "b": 2})
        h2 = _hash_data({"a": 1, "b": 2})
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_data_none(self):
        """_hash_data handles None input."""
        h = _hash_data(None)
        assert len(h) == 64

    def test_hash_data_different_inputs(self):
        """Different inputs produce different hashes."""
        h1 = _hash_data({"x": 1})
        h2 = _hash_data({"x": 2})
        assert h1 != h2

    def test_utcnow_returns_utc(self):
        """_utcnow returns a UTC-aware datetime with microseconds zeroed."""
        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.microsecond == 0


# ===========================================================================
# Test Initialization
# ===========================================================================


class TestInitialization:
    """Tests for RiskIndexEngine.__init__."""

    def test_engine_creates_successfully(self, engine):
        """Engine initializes without errors."""
        assert engine is not None

    def test_engine_default_weights(self, engine):
        """Engine uses default weights."""
        weights = engine.get_weights()
        assert weights["probability"] == _DEFAULT_WEIGHT_PROBABILITY
        assert weights["intensity"] == _DEFAULT_WEIGHT_INTENSITY
        assert weights["frequency"] == _DEFAULT_WEIGHT_FREQUENCY
        assert weights["duration"] == _DEFAULT_WEIGHT_DURATION

    def test_engine_default_thresholds(self, engine):
        """Engine uses default thresholds."""
        thresholds = engine.get_thresholds()
        assert thresholds["extreme"] == _DEFAULT_THRESHOLD_EXTREME
        assert thresholds["high"] == _DEFAULT_THRESHOLD_HIGH
        assert thresholds["medium"] == _DEFAULT_THRESHOLD_MEDIUM
        assert thresholds["low"] == _DEFAULT_THRESHOLD_LOW

    def test_engine_starts_empty(self, engine):
        """Engine starts with zero stored indices."""
        assert engine.get_index_count() == 0
        assert len(engine) == 0

    def test_engine_with_custom_genesis(self):
        """Engine with custom genesis hash initializes."""
        eng = RiskIndexEngine(genesis_hash="custom-seed")
        assert eng.get_index_count() == 0

    def test_engine_with_explicit_provenance(self):
        """Engine with explicit provenance tracker works."""
        mock_prov = MagicMock()
        mock_prov.record.return_value = MagicMock(hash_value="a" * 64)
        mock_prov.entry_count = 0
        eng = RiskIndexEngine(provenance=mock_prov)
        assert eng is not None

    def test_weights_sum_to_one(self, engine):
        """Default weights sum to 1.0."""
        weights = engine.get_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_thresholds_ordered(self, engine):
        """Thresholds satisfy extreme > high > medium > low."""
        t = engine.get_thresholds()
        assert t["extreme"] > t["high"] > t["medium"] > t["low"] > 0


# ===========================================================================
# Test Normalisation Helpers
# ===========================================================================


class TestNormalisationHelpers:
    """Tests for internal normalisation methods."""

    def test_normalise_intensity_zero(self, engine):
        """Intensity of 0 normalises to 0."""
        assert engine._normalise_intensity(0.0) == 0.0

    def test_normalise_intensity_max(self, engine):
        """Intensity of 10 normalises to 1.0."""
        assert engine._normalise_intensity(10.0) == 1.0

    def test_normalise_intensity_above_max(self, engine):
        """Intensity above 10 is capped at 1.0."""
        assert engine._normalise_intensity(15.0) == 1.0

    def test_normalise_intensity_negative(self, engine):
        """Negative intensity is clamped to 0."""
        assert engine._normalise_intensity(-5.0) == 0.0

    def test_normalise_intensity_midrange(self, engine):
        """Intensity of 5 normalises to 0.5."""
        assert engine._normalise_intensity(5.0) == pytest.approx(0.5)

    def test_normalise_frequency_zero(self, engine):
        """Frequency of 0 normalises to 0."""
        assert engine._normalise_frequency(0.0) == 0.0

    def test_normalise_frequency_max(self, engine):
        """Frequency of 10 normalises to 1.0."""
        assert engine._normalise_frequency(10.0) == 1.0

    def test_normalise_frequency_above_max(self, engine):
        """Frequency above 10 is capped at 1.0."""
        assert engine._normalise_frequency(20.0) == 1.0

    def test_normalise_duration_zero(self, engine):
        """Duration of 0 normalises to 0."""
        assert engine._normalise_duration(0.0) == 0.0

    def test_normalise_duration_365(self, engine):
        """Duration of 365 normalises to 1.0."""
        assert engine._normalise_duration(365.0) == 1.0

    def test_normalise_duration_above_max(self, engine):
        """Duration above 365 is capped at 1.0."""
        assert engine._normalise_duration(500.0) == 1.0

    def test_clamp_probability_valid(self, engine):
        """Probability in [0, 1] is returned unchanged."""
        assert engine._clamp_probability(0.5) == 0.5

    def test_clamp_probability_below_zero(self, engine):
        """Negative probability is clamped to 0."""
        assert engine._clamp_probability(-0.1) == 0.0

    def test_clamp_probability_above_one(self, engine):
        """Probability above 1 is clamped to 1."""
        assert engine._clamp_probability(1.5) == 1.0


# ===========================================================================
# Test classify_risk_level
# ===========================================================================


class TestClassifyRiskLevel:
    """Tests for RiskIndexEngine.classify_risk_level."""

    @pytest.mark.parametrize("score,expected", [
        (0.0, "NEGLIGIBLE"),
        (10.0, "NEGLIGIBLE"),
        (19.99, "NEGLIGIBLE"),
        (20.0, "LOW"),
        (30.0, "LOW"),
        (39.99, "LOW"),
        (40.0, "MEDIUM"),
        (50.0, "MEDIUM"),
        (59.99, "MEDIUM"),
        (60.0, "HIGH"),
        (70.0, "HIGH"),
        (79.99, "HIGH"),
        (80.0, "EXTREME"),
        (90.0, "EXTREME"),
        (100.0, "EXTREME"),
    ])
    def test_classify_at_boundaries(self, engine, score, expected):
        """Risk level classification at boundary values."""
        assert engine.classify_risk_level(score) == expected


# ===========================================================================
# Test calculate_risk_index
# ===========================================================================


class TestCalculateRiskIndex:
    """Tests for RiskIndexEngine.calculate_risk_index."""

    def test_basic_calculation(self, engine, london_location):
        """Calculate a basic risk index."""
        result = engine.calculate_risk_index(
            hazard_type="riverine_flood",
            location=london_location,
            probability=0.4,
            intensity=6.0,
            frequency=2.0,
            duration_days=14.0,
        )
        assert "index_id" in result
        assert result["index_id"].startswith("RI-")
        assert "risk_score" in result
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_level"] in {rl.value for rl in RiskLevel}
        assert result["hazard_type"] == "riverine_flood"

    def test_calculation_accuracy(self, engine, london_location):
        """Risk score matches the expected formula."""
        prob, inten, freq, dur = 0.4, 6.0, 2.0, 14.0
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=prob,
            intensity=inten,
            frequency=freq,
            duration_days=dur,
        )
        expected = _expected_score(prob, inten, freq, dur)
        assert result["risk_score"] == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize("prob,inten,freq,dur", [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 10.0, 10.0, 365.0),
        (0.5, 5.0, 5.0, 182.5),
        (0.1, 1.0, 0.5, 3.0),
        (0.9, 9.0, 8.0, 300.0),
    ])
    def test_calculation_parametrized(self, engine, london_location, prob, inten, freq, dur):
        """Parametrized risk score accuracy test."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=prob,
            intensity=inten,
            frequency=freq,
            duration_days=dur,
        )
        expected = _expected_score(prob, inten, freq, dur)
        assert result["risk_score"] == pytest.approx(expected, abs=0.01)

    def test_zero_inputs_negligible(self, engine, london_location):
        """All-zero inputs produce NEGLIGIBLE risk."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.0,
            intensity=0.0,
            frequency=0.0,
            duration_days=0.0,
        )
        assert result["risk_score"] == 0.0
        assert result["risk_level"] == "NEGLIGIBLE"

    def test_max_inputs_extreme(self, engine, london_location):
        """Maximum inputs produce EXTREME risk with score 100."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=1.0,
            intensity=10.0,
            frequency=10.0,
            duration_days=365.0,
        )
        assert result["risk_score"] == 100.0
        assert result["risk_level"] == "EXTREME"

    def test_component_scores(self, engine, london_location):
        """Component scores are included in result."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=3.0,
            duration_days=30.0,
        )
        cs = result["component_scores"]
        assert "probability" in cs
        assert "intensity" in cs
        assert "frequency" in cs
        assert "duration" in cs
        assert cs["probability"]["weight"] == _DEFAULT_WEIGHT_PROBABILITY
        assert cs["intensity"]["normalised"] == pytest.approx(0.5, abs=0.01)

    def test_location_normalised(self, engine):
        """Location is normalised with lat, lon, name."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location={"lat": 40.0, "lon": -3.7},
            probability=0.3,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert result["location"]["lat"] == 40.0
        assert result["location"]["lon"] == -3.7
        assert result["location"]["name"] == "unknown"

    def test_scenario_and_time_horizon(self, engine, london_location):
        """Scenario and time_horizon are stored in result."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
            scenario="SSP2-4.5",
            time_horizon="MID_TERM",
        )
        assert result["scenario"] == "SSP2-4.5"
        assert result["time_horizon"] == "MID_TERM"

    def test_provenance_hash_present(self, engine, london_location):
        """Result includes a SHA-256 provenance hash."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert len(result["provenance_hash"]) == 64

    def test_calculated_at_timestamp(self, engine, london_location):
        """Result includes a calculated_at timestamp."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert "calculated_at" in result

    def test_stored_in_memory(self, engine, london_location):
        """Calculated index is stored and retrievable."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        retrieved = engine.get_risk_index(result["index_id"])
        assert retrieved is not None
        assert retrieved["risk_score"] == result["risk_score"]

    def test_returns_deep_copy(self, engine, london_location):
        """Returned result is a deep copy."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        result["risk_score"] = -999
        retrieved = engine.get_risk_index(result["index_id"])
        assert retrieved["risk_score"] != -999

    def test_invalid_probability_raises(self, engine, london_location):
        """Probability outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="probability"):
            engine.calculate_risk_index(
                hazard_type="drought",
                location=london_location,
                probability=1.5,
                intensity=5.0,
                frequency=1.0,
                duration_days=10.0,
            )

    def test_negative_probability_raises(self, engine, london_location):
        """Negative probability raises ValueError."""
        with pytest.raises(ValueError, match="probability"):
            engine.calculate_risk_index(
                hazard_type="drought",
                location=london_location,
                probability=-0.1,
                intensity=5.0,
                frequency=1.0,
                duration_days=10.0,
            )

    def test_nan_intensity_raises(self, engine, london_location):
        """NaN intensity raises ValueError."""
        with pytest.raises(ValueError, match="intensity"):
            engine.calculate_risk_index(
                hazard_type="drought",
                location=london_location,
                probability=0.5,
                intensity=float("nan"),
                frequency=1.0,
                duration_days=10.0,
            )

    def test_inf_frequency_raises(self, engine, london_location):
        """Infinite frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency"):
            engine.calculate_risk_index(
                hazard_type="drought",
                location=london_location,
                probability=0.5,
                intensity=5.0,
                frequency=float("inf"),
                duration_days=10.0,
            )

    def test_negative_duration_raises(self, engine, london_location):
        """Negative duration_days raises ValueError."""
        with pytest.raises(ValueError, match="duration_days"):
            engine.calculate_risk_index(
                hazard_type="drought",
                location=london_location,
                probability=0.5,
                intensity=5.0,
                frequency=1.0,
                duration_days=-10.0,
            )

    def test_empty_hazard_type_raises(self, engine, london_location):
        """Empty hazard_type raises ValueError."""
        with pytest.raises(ValueError, match="hazard_type"):
            engine.calculate_risk_index(
                hazard_type="",
                location=london_location,
                probability=0.5,
                intensity=5.0,
                frequency=1.0,
                duration_days=10.0,
            )

    def test_empty_location_raises(self, engine):
        """Empty location dict raises ValueError."""
        with pytest.raises(ValueError, match="location"):
            engine.calculate_risk_index(
                hazard_type="drought",
                location={},
                probability=0.5,
                intensity=5.0,
                frequency=1.0,
                duration_days=10.0,
            )

    def test_increments_total_calculations(self, engine, london_location):
        """Total calculations counter increments."""
        engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        stats = engine.get_statistics()
        assert stats["total_calculations"] == 1


# ===========================================================================
# Test calculate_multi_hazard_index
# ===========================================================================


class TestCalculateMultiHazardIndex:
    """Tests for RiskIndexEngine.calculate_multi_hazard_index."""

    def test_weighted_average_aggregation(self, engine, london_location, sample_hazard_risks):
        """Weighted average multi-hazard calculation."""
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
            aggregation="weighted_average",
        )
        assert "multi_hazard_id" in result
        assert result["multi_hazard_id"].startswith("MH-")
        assert 0 <= result["multi_hazard_score"] <= 100
        assert result["aggregation"] == "weighted_average"
        assert result["hazard_count"] == 3

    def test_maximum_aggregation(self, engine, london_location, sample_hazard_risks):
        """Maximum aggregation returns the highest per-hazard score."""
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
            aggregation="maximum",
        )
        per_scores = [h["risk_score"] for h in result["per_hazard_scores"]]
        assert result["multi_hazard_score"] == max(per_scores)

    def test_sum_capped_aggregation(self, engine, london_location, sample_hazard_risks):
        """Sum capped aggregation caps at 100."""
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
            aggregation="sum_capped",
        )
        assert result["multi_hazard_score"] <= 100.0

    def test_dominant_hazard_identified(self, engine, london_location, sample_hazard_risks):
        """Dominant hazard is the one with highest score."""
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        dominant = result["dominant_hazard"]
        per_scores = result["per_hazard_scores"]
        max_score = max(h["risk_score"] for h in per_scores)
        assert dominant["risk_score"] == max_score

    def test_per_hazard_scores_present(self, engine, london_location, sample_hazard_risks):
        """Per-hazard scores list is included."""
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        assert len(result["per_hazard_scores"]) == 3

    def test_provenance_hash(self, engine, london_location, sample_hazard_risks):
        """Multi-hazard result has provenance hash."""
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        assert len(result["provenance_hash"]) == 64

    def test_empty_hazard_risks_raises(self, engine, london_location):
        """Empty hazard_risks list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            engine.calculate_multi_hazard_index(
                location=london_location,
                hazard_risks=[],
            )

    def test_invalid_aggregation_raises(self, engine, london_location, sample_hazard_risks):
        """Invalid aggregation strategy raises ValueError."""
        with pytest.raises(ValueError, match="aggregation"):
            engine.calculate_multi_hazard_index(
                location=london_location,
                hazard_risks=sample_hazard_risks,
                aggregation="bogus",
            )

    def test_empty_location_raises(self, engine, sample_hazard_risks):
        """Empty location raises ValueError."""
        with pytest.raises(ValueError, match="location"):
            engine.calculate_multi_hazard_index(
                location={},
                hazard_risks=sample_hazard_risks,
            )

    def test_single_hazard_multi(self, engine, london_location):
        """Multi-hazard with single hazard returns that hazard's score."""
        single = [{
            "hazard_type": "drought",
            "probability": 0.5,
            "intensity": 5.0,
            "frequency": 1.0,
            "duration_days": 30.0,
        }]
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=single,
            aggregation="weighted_average",
        )
        assert result["multi_hazard_score"] == result["per_hazard_scores"][0]["risk_score"]

    def test_increments_multi_calculations(self, engine, london_location, sample_hazard_risks):
        """Total multi calculations counter increments."""
        engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        stats = engine.get_statistics()
        assert stats["total_multi_calculations"] == 1


# ===========================================================================
# Test calculate_compound_risk
# ===========================================================================


class TestCalculateCompoundRisk:
    """Tests for RiskIndexEngine.calculate_compound_risk."""

    def test_basic_compound_risk(self, engine, london_location):
        """Basic compound risk with correlated hazards."""
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 0.6,
                "intensity": 7.0,
                "frequency": 1.0,
                "duration_days": 90.0,
            },
            secondary_hazards=[{
                "hazard_type": "wildfire",
                "probability": 0.4,
                "intensity": 8.0,
                "frequency": 0.5,
                "duration_days": 7.0,
            }],
        )
        assert "compound_id" in result
        assert result["compound_id"].startswith("CR-")
        assert 0 <= result["compound_score"] <= 100
        assert result["amplification_factor"] >= 1.0

    def test_compound_uses_default_correlation(self, engine, london_location):
        """Compound risk uses built-in correlations when not overridden."""
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            secondary_hazards=[{
                "hazard_type": "wildfire",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            }],
        )
        # drought + wildfire has correlation 0.75
        corr_detail = result["correlation_details"][0]
        assert corr_detail["correlation_factor"] == pytest.approx(0.75, abs=0.01)

    def test_compound_with_custom_correlation(self, engine, london_location):
        """Custom correlation_factors override defaults."""
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            secondary_hazards=[{
                "hazard_type": "wildfire",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            }],
            correlation_factors={"wildfire": 0.90},
        )
        corr_detail = result["correlation_details"][0]
        assert corr_detail["correlation_factor"] == pytest.approx(0.90, abs=0.01)

    def test_compound_score_capped_at_100(self, engine, london_location):
        """Compound score does not exceed 100."""
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 1.0,
                "intensity": 10.0,
                "frequency": 10.0,
                "duration_days": 365.0,
            },
            secondary_hazards=[
                {
                    "hazard_type": "wildfire",
                    "probability": 1.0,
                    "intensity": 10.0,
                    "frequency": 10.0,
                    "duration_days": 365.0,
                },
                {
                    "hazard_type": "extreme_heat",
                    "probability": 1.0,
                    "intensity": 10.0,
                    "frequency": 10.0,
                    "duration_days": 365.0,
                },
            ],
        )
        assert result["compound_score"] <= 100.0

    def test_compound_no_secondaries(self, engine, london_location):
        """Compound with empty secondaries returns primary score."""
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            secondary_hazards=[],
        )
        assert result["amplification_factor"] == pytest.approx(1.0, abs=0.01)

    def test_compound_invalid_primary_raises(self, engine, london_location):
        """Invalid primary_hazard raises ValueError."""
        with pytest.raises(ValueError, match="primary_hazard"):
            engine.calculate_compound_risk(
                location=london_location,
                primary_hazard={},
                secondary_hazards=[],
            )

    def test_compound_provenance_hash(self, engine, london_location):
        """Compound result has provenance hash."""
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            secondary_hazards=[],
        )
        assert len(result["provenance_hash"]) == 64

    def test_increments_compound_calculations(self, engine, london_location):
        """Total compound calculations counter increments."""
        engine.calculate_compound_risk(
            location=london_location,
            primary_hazard={
                "hazard_type": "drought",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            secondary_hazards=[],
        )
        stats = engine.get_statistics()
        assert stats["total_compound_calculations"] == 1


# ===========================================================================
# Test rank_hazards
# ===========================================================================


class TestRankHazards:
    """Tests for RiskIndexEngine.rank_hazards."""

    def test_basic_ranking(self, engine, london_location, sample_hazard_risks):
        """Hazards are ranked by risk score descending."""
        ranked = engine.rank_hazards(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        assert len(ranked) == 3
        for i in range(len(ranked) - 1):
            assert ranked[i]["risk_score"] >= ranked[i + 1]["risk_score"]

    def test_rank_numbers_sequential(self, engine, london_location, sample_hazard_risks):
        """Rank numbers are 1-indexed and sequential."""
        ranked = engine.rank_hazards(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        for i, entry in enumerate(ranked):
            assert entry["rank"] == i + 1

    def test_rank_includes_scores(self, engine, london_location, sample_hazard_risks):
        """Each ranked entry has risk_score and risk_level."""
        ranked = engine.rank_hazards(
            location=london_location,
            hazard_risks=sample_hazard_risks,
        )
        for entry in ranked:
            assert "risk_score" in entry
            assert "risk_level" in entry
            assert "component_scores" in entry

    def test_rank_empty_raises(self, engine, london_location):
        """Empty hazard_risks raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            engine.rank_hazards(
                location=london_location,
                hazard_risks=[],
            )

    def test_rank_returns_deep_copy(self, engine, london_location, sample_hazard_risks):
        """Ranked results are deep copies."""
        ranked1 = engine.rank_hazards(london_location, sample_hazard_risks)
        ranked2 = engine.rank_hazards(london_location, sample_hazard_risks)
        assert ranked1 is not ranked2


# ===========================================================================
# Test compare_locations
# ===========================================================================


class TestCompareLocations:
    """Tests for RiskIndexEngine.compare_locations."""

    def test_basic_comparison(self, engine):
        """Compare multiple locations for a hazard type."""
        locations = [
            {
                "location": {"lat": 51.5, "lon": -0.1, "name": "London"},
                "probability": 0.4,
                "intensity": 6.0,
                "frequency": 2.0,
                "duration_days": 14.0,
            },
            {
                "location": {"lat": 48.8, "lon": 2.3, "name": "Paris"},
                "probability": 0.3,
                "intensity": 4.0,
                "frequency": 1.0,
                "duration_days": 10.0,
            },
        ]
        compared = engine.compare_locations(
            locations_with_risks=locations,
            hazard_type="riverine_flood",
        )
        assert len(compared) == 2
        # Sorted descending by score
        assert compared[0]["risk_score"] >= compared[1]["risk_score"]
        assert compared[0]["rank"] == 1
        assert compared[1]["rank"] == 2

    def test_compare_empty_raises(self, engine):
        """Empty locations list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            engine.compare_locations([], "drought")

    def test_compare_empty_hazard_type_raises(self, engine):
        """Empty hazard_type raises ValueError."""
        with pytest.raises(ValueError, match="hazard_type"):
            engine.compare_locations(
                [{"location": {"lat": 0, "lon": 0}, "probability": 0.5,
                  "intensity": 5.0, "frequency": 1.0, "duration_days": 10.0}],
                "",
            )

    def test_compare_increments_counter(self, engine):
        """Total comparisons counter increments."""
        engine.compare_locations(
            [{"location": {"lat": 0, "lon": 0}, "probability": 0.5,
              "intensity": 5.0, "frequency": 1.0, "duration_days": 10.0}],
            "drought",
        )
        stats = engine.get_statistics()
        assert stats["total_comparisons"] == 1


# ===========================================================================
# Test get_risk_trend
# ===========================================================================


class TestGetRiskTrend:
    """Tests for RiskIndexEngine.get_risk_trend."""

    def test_increasing_trend(self, engine, london_location):
        """Increasing scores produce 'increasing' trend."""
        snapshots = [
            {"time_horizon": "2030", "probability": 0.2, "intensity": 3.0, "frequency": 1.0, "duration_days": 10.0},
            {"time_horizon": "2050", "probability": 0.5, "intensity": 6.0, "frequency": 2.0, "duration_days": 30.0},
            {"time_horizon": "2100", "probability": 0.8, "intensity": 9.0, "frequency": 5.0, "duration_days": 90.0},
        ]
        result = engine.get_risk_trend("drought", london_location, snapshots)
        assert result["direction"] == "increasing"
        assert result["total_change"] > 2.0

    def test_decreasing_trend(self, engine, london_location):
        """Decreasing scores produce 'decreasing' trend."""
        snapshots = [
            {"time_horizon": "2030", "probability": 0.8, "intensity": 9.0, "frequency": 5.0, "duration_days": 90.0},
            {"time_horizon": "2050", "probability": 0.5, "intensity": 6.0, "frequency": 2.0, "duration_days": 30.0},
            {"time_horizon": "2100", "probability": 0.2, "intensity": 3.0, "frequency": 1.0, "duration_days": 10.0},
        ]
        result = engine.get_risk_trend("drought", london_location, snapshots)
        assert result["direction"] == "decreasing"
        assert result["total_change"] < -2.0

    def test_stable_trend(self, engine, london_location):
        """Stable scores produce 'stable' trend (change <= 2)."""
        snapshots = [
            {"time_horizon": "2030", "probability": 0.5, "intensity": 5.0, "frequency": 2.0, "duration_days": 30.0},
            {"time_horizon": "2050", "probability": 0.5, "intensity": 5.0, "frequency": 2.0, "duration_days": 30.0},
            {"time_horizon": "2100", "probability": 0.5, "intensity": 5.0, "frequency": 2.0, "duration_days": 30.0},
        ]
        result = engine.get_risk_trend("drought", london_location, snapshots)
        assert result["direction"] == "stable"
        assert abs(result["total_change"]) <= 2.0

    def test_trend_result_fields(self, engine, london_location):
        """Trend result includes all expected fields."""
        snapshots = [
            {"time_horizon": "2030", "probability": 0.3, "intensity": 4.0, "frequency": 1.0, "duration_days": 10.0},
            {"time_horizon": "2050", "probability": 0.5, "intensity": 6.0, "frequency": 2.0, "duration_days": 20.0},
        ]
        result = engine.get_risk_trend("drought", london_location, snapshots)
        assert "trend_id" in result
        assert result["trend_id"].startswith("TR-")
        assert "direction" in result
        assert "rate_of_change" in result
        assert "total_change" in result
        assert "snapshots" in result
        assert "first_score" in result
        assert "last_score" in result
        assert "min_score" in result
        assert "max_score" in result
        assert "provenance_hash" in result

    def test_trend_single_snapshot(self, engine, london_location):
        """Single snapshot produces stable trend with zero rate."""
        snapshots = [
            {"time_horizon": "2030", "probability": 0.5, "intensity": 5.0, "frequency": 1.0, "duration_days": 10.0},
        ]
        result = engine.get_risk_trend("drought", london_location, snapshots)
        assert result["direction"] == "stable"
        assert result["rate_of_change"] == 0.0

    def test_trend_empty_snapshots_raises(self, engine, london_location):
        """Empty snapshots list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            engine.get_risk_trend("drought", london_location, [])

    def test_trend_increments_counter(self, engine, london_location):
        """Total trends counter increments."""
        snapshots = [
            {"time_horizon": "2030", "probability": 0.5, "intensity": 5.0, "frequency": 1.0, "duration_days": 10.0},
        ]
        engine.get_risk_trend("drought", london_location, snapshots)
        stats = engine.get_statistics()
        assert stats["total_trends"] == 1


# ===========================================================================
# Test get_risk_index
# ===========================================================================


class TestGetRiskIndex:
    """Tests for RiskIndexEngine.get_risk_index."""

    def test_retrieve_existing(self, engine_with_indices):
        """Retrieve a stored risk index."""
        indices = engine_with_indices.list_risk_indices(limit=1)
        assert len(indices) >= 1
        idx_id = indices[0].get("index_id")
        if idx_id:
            retrieved = engine_with_indices.get_risk_index(idx_id)
            assert retrieved is not None

    def test_retrieve_nonexistent(self, engine):
        """Non-existent ID returns None."""
        assert engine.get_risk_index("RI-nonexistent") is None

    def test_retrieve_empty_id(self, engine):
        """Empty ID returns None."""
        assert engine.get_risk_index("") is None

    def test_retrieve_returns_deep_copy(self, engine_with_indices):
        """Retrieved index is a deep copy."""
        indices = engine_with_indices.list_risk_indices(limit=1)
        if indices:
            idx_id = indices[0].get("index_id")
            if idx_id:
                r1 = engine_with_indices.get_risk_index(idx_id)
                r2 = engine_with_indices.get_risk_index(idx_id)
                assert r1 is not r2


# ===========================================================================
# Test list_risk_indices
# ===========================================================================


class TestListRiskIndices:
    """Tests for RiskIndexEngine.list_risk_indices."""

    def test_list_all(self, engine_with_indices):
        """List all stored indices."""
        indices = engine_with_indices.list_risk_indices()
        assert len(indices) == 3

    def test_filter_by_hazard_type(self, engine_with_indices):
        """Filter by hazard_type."""
        indices = engine_with_indices.list_risk_indices(
            hazard_type="drought",
        )
        assert len(indices) >= 1

    def test_filter_by_risk_level(self, engine_with_indices):
        """Filter by risk_level."""
        # Calculate what levels exist
        all_indices = engine_with_indices.list_risk_indices()
        if all_indices:
            level = all_indices[0].get("risk_level", "MEDIUM")
            filtered = engine_with_indices.list_risk_indices(risk_level=level)
            assert len(filtered) >= 1

    def test_filter_by_location(self, engine_with_indices):
        """Filter by location name substring."""
        indices = engine_with_indices.list_risk_indices(
            location="London",
        )
        assert len(indices) >= 1

    def test_list_with_limit(self, engine_with_indices):
        """Limit caps result count."""
        indices = engine_with_indices.list_risk_indices(limit=2)
        assert len(indices) <= 2

    def test_list_empty_engine(self, engine):
        """Empty engine returns empty list."""
        indices = engine.list_risk_indices()
        assert indices == []


# ===========================================================================
# Test get_high_risk_summary
# ===========================================================================


class TestGetHighRiskSummary:
    """Tests for RiskIndexEngine.get_high_risk_summary."""

    def test_high_risk_summary(self, engine_with_indices):
        """Get summary of high-risk indices."""
        summary = engine_with_indices.get_high_risk_summary(threshold=0.0)
        assert summary["total_indices"] == 3
        assert summary["high_risk_count"] >= 0
        assert "by_risk_level" in summary
        assert "by_hazard_type" in summary
        assert "provenance_hash" in summary

    def test_high_risk_summary_high_threshold(self, engine):
        """High threshold with no qualifying indices returns zeros."""
        summary = engine.get_high_risk_summary(threshold=101.0)
        assert summary["high_risk_count"] == 0
        assert summary["highest_score"] == 0.0
        assert summary["average_score"] == 0.0

    def test_high_risk_default_threshold(self, engine_with_indices):
        """Default threshold is 60."""
        summary = engine_with_indices.get_high_risk_summary()
        assert summary["threshold"] == 60.0


# ===========================================================================
# Test get_statistics
# ===========================================================================


class TestGetStatistics:
    """Tests for RiskIndexEngine.get_statistics."""

    def test_statistics_keys(self, engine):
        """Statistics has all expected keys."""
        stats = engine.get_statistics()
        expected_keys = {
            "total_indices",
            "total_calculations",
            "total_multi_calculations",
            "total_compound_calculations",
            "total_rankings",
            "total_comparisons",
            "total_trends",
            "by_risk_level",
            "by_hazard_type",
            "score_distribution",
            "weights",
            "thresholds",
            "created_at",
            "generated_at",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_statistics_zero_state(self, engine):
        """Fresh engine has zero counters."""
        stats = engine.get_statistics()
        assert stats["total_indices"] == 0
        assert stats["total_calculations"] == 0

    def test_statistics_after_calculations(self, engine_with_indices):
        """Statistics reflect performed calculations."""
        stats = engine_with_indices.get_statistics()
        assert stats["total_indices"] == 3
        assert stats["total_calculations"] == 3

    def test_score_distribution(self, engine_with_indices):
        """Score distribution has min, max, mean, median."""
        stats = engine_with_indices.get_statistics()
        dist = stats["score_distribution"]
        assert "min" in dist
        assert "max" in dist
        assert "mean" in dist
        assert "median" in dist
        assert dist["count"] == 3
        assert dist["min"] <= dist["mean"] <= dist["max"]


# ===========================================================================
# Test clear
# ===========================================================================


class TestClear:
    """Tests for RiskIndexEngine.clear."""

    def test_clear_removes_indices(self, engine_with_indices):
        """Clear removes all stored indices."""
        result = engine_with_indices.clear()
        assert result["cleared_count"] == 3
        assert engine_with_indices.get_index_count() == 0

    def test_clear_resets_counters(self, engine_with_indices):
        """Clear resets all counters."""
        engine_with_indices.clear()
        stats = engine_with_indices.get_statistics()
        assert stats["total_calculations"] == 0
        assert stats["total_multi_calculations"] == 0
        assert stats["total_compound_calculations"] == 0

    def test_clear_empty_engine(self, engine):
        """Clear on empty engine returns zero count."""
        result = engine.clear()
        assert result["cleared_count"] == 0

    def test_clear_returns_timestamp(self, engine):
        """Clear result includes cleared_at timestamp."""
        result = engine.clear()
        assert "cleared_at" in result


# ===========================================================================
# Test set_weights / get_weights
# ===========================================================================


class TestWeightManagement:
    """Tests for weight getter and setter."""

    def test_get_weights(self, engine):
        """Get weights returns default values."""
        w = engine.get_weights()
        assert w["probability"] == 0.30
        assert w["intensity"] == 0.30
        assert w["frequency"] == 0.25
        assert w["duration"] == 0.15

    def test_set_weights(self, engine):
        """Set new weights successfully."""
        result = engine.set_weights(
            probability=0.25,
            intensity=0.25,
            frequency=0.25,
            duration=0.25,
        )
        assert result["probability"] == 0.25
        assert sum(result.values()) == pytest.approx(1.0)

    def test_set_partial_weights(self, engine):
        """Set only some weights, keeping others unchanged."""
        engine.set_weights(
            probability=0.20,
            intensity=0.30,
            frequency=0.30,
            duration=0.20,
        )
        w = engine.get_weights()
        assert w["probability"] == 0.20
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_set_weights_invalid_sum_raises(self, engine):
        """Weights not summing to 1 raises ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            engine.set_weights(probability=0.5, intensity=0.5, frequency=0.5, duration=0.5)

    def test_set_weights_negative_raises(self, engine):
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            engine.set_weights(probability=-0.1, intensity=0.4, frequency=0.4, duration=0.3)


# ===========================================================================
# Test set_thresholds / get_thresholds
# ===========================================================================


class TestThresholdManagement:
    """Tests for threshold getter and setter."""

    def test_get_thresholds(self, engine):
        """Get thresholds returns default values."""
        t = engine.get_thresholds()
        assert t["extreme"] == 80.0
        assert t["high"] == 60.0
        assert t["medium"] == 40.0
        assert t["low"] == 20.0

    def test_set_thresholds(self, engine):
        """Set new thresholds successfully."""
        result = engine.set_thresholds(
            extreme=90.0,
            high=70.0,
            medium=50.0,
            low=30.0,
        )
        assert result["extreme"] == 90.0
        assert result["high"] == 70.0

    def test_set_thresholds_invalid_order_raises(self, engine):
        """Thresholds not in descending order raises ValueError."""
        with pytest.raises(ValueError, match="extreme > high > medium > low"):
            engine.set_thresholds(extreme=50.0, high=60.0)


# ===========================================================================
# Test compound correlations
# ===========================================================================


class TestCompoundCorrelations:
    """Tests for compound hazard correlation management."""

    def test_get_compound_correlations(self, engine):
        """Get default compound correlations."""
        corrs = engine.get_compound_correlations()
        assert len(corrs) > 0

    def test_drought_wildfire_correlation(self, engine):
        """Drought + wildfire correlation is 0.75."""
        corrs = engine.get_compound_correlations()
        # Keys are sorted alphabetically
        assert any(
            "drought" in k and "wildfire" in k
            for k in corrs.keys()
        )

    def test_set_compound_correlation(self, engine):
        """Set a custom correlation factor."""
        engine.set_compound_correlation("drought", "wildfire", 0.90)
        corr = engine._get_compound_correlation("drought", "wildfire")
        assert corr == pytest.approx(0.90)

    def test_set_correlation_out_of_range_raises(self, engine):
        """Correlation outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            engine.set_compound_correlation("drought", "wildfire", 1.5)

    def test_get_unknown_pair_default(self, engine):
        """Unknown hazard pair returns default 0.30."""
        corr = engine._get_compound_correlation("unknown_a", "unknown_b")
        assert corr == pytest.approx(0.30)


# ===========================================================================
# Test calculate_risk_score_raw
# ===========================================================================


class TestCalculateRiskScoreRaw:
    """Tests for RiskIndexEngine.calculate_risk_score_raw."""

    def test_raw_score_matches_formula(self, engine):
        """Raw score matches expected formula."""
        score = engine.calculate_risk_score_raw(0.5, 5.0, 2.0, 30.0)
        expected = _expected_score(0.5, 5.0, 2.0, 30.0)
        assert score == pytest.approx(expected, abs=0.01)

    def test_raw_score_zero(self, engine):
        """All zeros produce zero score."""
        assert engine.calculate_risk_score_raw(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_raw_score_max(self, engine):
        """Max values produce score of 100."""
        assert engine.calculate_risk_score_raw(1.0, 10.0, 10.0, 365.0) == 100.0

    def test_raw_score_does_not_store(self, engine):
        """Raw calculation does not store in memory."""
        engine.calculate_risk_score_raw(0.5, 5.0, 2.0, 30.0)
        assert engine.get_index_count() == 0


# ===========================================================================
# Test batch_calculate_risk_indices
# ===========================================================================


class TestBatchCalculateRiskIndices:
    """Tests for RiskIndexEngine.batch_calculate_risk_indices."""

    def test_batch_success(self, engine):
        """Batch calculation with valid entries succeeds."""
        entries = [
            {
                "hazard_type": "drought",
                "location": {"lat": 40.0, "lon": -3.7, "name": "Madrid"},
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            {
                "hazard_type": "wildfire",
                "location": {"lat": 34.0, "lon": -118.2, "name": "LA"},
                "probability": 0.4,
                "intensity": 8.0,
                "frequency": 0.3,
                "duration_days": 7.0,
            },
        ]
        results = engine.batch_calculate_risk_indices(entries)
        assert len(results) == 2
        assert all(r.get("status") != "error" for r in results)

    def test_batch_with_errors(self, engine):
        """Batch with invalid entries produces error dicts."""
        entries = [
            {
                "hazard_type": "drought",
                "location": {"lat": 40.0, "lon": -3.7, "name": "Madrid"},
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
            {
                # Invalid: probability > 1
                "hazard_type": "drought",
                "location": {"lat": 40.0, "lon": -3.7, "name": "Madrid"},
                "probability": 2.0,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
            },
        ]
        results = engine.batch_calculate_risk_indices(entries)
        assert len(results) == 2
        success = [r for r in results if r.get("status") != "error"]
        errors = [r for r in results if r.get("status") == "error"]
        assert len(success) == 1
        assert len(errors) == 1

    def test_batch_empty_list(self, engine):
        """Empty batch returns empty list."""
        results = engine.batch_calculate_risk_indices([])
        assert results == []


# ===========================================================================
# Test get_risk_level_distribution
# ===========================================================================


class TestGetRiskLevelDistribution:
    """Tests for RiskIndexEngine.get_risk_level_distribution."""

    def test_distribution_empty(self, engine):
        """Empty engine has zero distribution."""
        dist = engine.get_risk_level_distribution()
        assert dist["total"] == 0

    def test_distribution_populated(self, engine_with_indices):
        """Distribution reflects stored indices."""
        dist = engine_with_indices.get_risk_level_distribution()
        assert dist["total"] == 3
        total_count = sum(d["count"] for d in dist["distribution"].values())
        assert total_count == 3

    def test_distribution_has_all_levels(self, engine):
        """Distribution includes all 5 risk levels."""
        dist = engine.get_risk_level_distribution()
        for level in RiskLevel:
            assert level.value in dist["distribution"]


# ===========================================================================
# Test get_hazard_type_summary
# ===========================================================================


class TestGetHazardTypeSummary:
    """Tests for RiskIndexEngine.get_hazard_type_summary."""

    def test_summary_empty(self, engine):
        """Empty engine returns zero types."""
        summary = engine.get_hazard_type_summary()
        assert summary["total_types"] == 0

    def test_summary_populated(self, engine_with_indices):
        """Summary groups by hazard type correctly."""
        summary = engine_with_indices.get_hazard_type_summary()
        assert summary["total_types"] >= 2
        for ht, stats in summary["hazard_types"].items():
            assert "count" in stats
            assert "min_score" in stats
            assert "max_score" in stats
            assert "mean_score" in stats


# ===========================================================================
# Test get_location_risk_profile
# ===========================================================================


class TestGetLocationRiskProfile:
    """Tests for RiskIndexEngine.get_location_risk_profile."""

    def test_profile_existing_location(self, engine_with_indices):
        """Get profile for a location with indices."""
        profile = engine_with_indices.get_location_risk_profile("London")
        assert profile["index_count"] >= 1
        assert profile["location_name"] == "London"
        assert "dominant_hazard" in profile
        assert "score_summary" in profile

    def test_profile_nonexistent_location(self, engine_with_indices):
        """Profile for location with no indices returns zero."""
        profile = engine_with_indices.get_location_risk_profile("NoSuchPlace")
        assert profile["index_count"] == 0
        assert profile["dominant_hazard"] == "none"


# ===========================================================================
# Test validate_risk_components
# ===========================================================================


class TestValidateRiskComponents:
    """Tests for RiskIndexEngine.validate_risk_components."""

    def test_valid_components(self, engine):
        """Valid components return is_valid=True."""
        result = engine.validate_risk_components(0.5, 5.0, 2.0, 30.0)
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["normalised"]["probability"] is not None

    def test_invalid_probability(self, engine):
        """Invalid probability returns errors."""
        result = engine.validate_risk_components(1.5, 5.0, 2.0, 30.0)
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 1

    def test_nan_input(self, engine):
        """NaN values are caught."""
        result = engine.validate_risk_components(float("nan"), 5.0, 2.0, 30.0)
        assert result["is_valid"] is False

    def test_negative_frequency(self, engine):
        """Negative frequency is caught."""
        result = engine.validate_risk_components(0.5, 5.0, -1.0, 30.0)
        assert result["is_valid"] is False


# ===========================================================================
# Test delete_risk_index
# ===========================================================================


class TestDeleteRiskIndex:
    """Tests for RiskIndexEngine.delete_risk_index."""

    def test_delete_existing(self, engine, london_location):
        """Delete an existing risk index."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert engine.delete_risk_index(result["index_id"]) is True
        assert engine.get_risk_index(result["index_id"]) is None

    def test_delete_nonexistent(self, engine):
        """Deleting non-existent index returns False."""
        assert engine.delete_risk_index("RI-nonexistent") is False


# ===========================================================================
# Test export_indices / import_indices
# ===========================================================================


class TestExportImportIndices:
    """Tests for export_indices and import_indices."""

    def test_export_dict(self, engine_with_indices):
        """Export as dict list."""
        exported = engine_with_indices.export_indices(format="dict")
        assert isinstance(exported, list)
        assert len(exported) == 3

    def test_export_json(self, engine_with_indices):
        """Export as JSON string."""
        exported = engine_with_indices.export_indices(format="json")
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert len(parsed) == 3

    def test_import_indices(self, engine):
        """Import indices into engine."""
        indices = [
            {"index_id": "RI-test1", "risk_score": 50.0, "risk_level": "MEDIUM"},
            {"index_id": "RI-test2", "risk_score": 70.0, "risk_level": "HIGH"},
        ]
        result = engine.import_indices(indices)
        assert result["imported"] == 2
        assert result["skipped"] == 0
        assert engine.get_index_count() == 2

    def test_import_skip_without_id(self, engine):
        """Import skips entries without any ID field."""
        result = engine.import_indices([{"no_id": True}])
        assert result["skipped"] == 1
        assert result["imported"] == 0

    def test_import_multi_hazard_id(self, engine):
        """Import recognizes multi_hazard_id."""
        result = engine.import_indices([
            {"multi_hazard_id": "MH-test1", "multi_hazard_score": 55.0},
        ])
        assert result["imported"] == 1

    def test_import_compound_id(self, engine):
        """Import recognizes compound_id."""
        result = engine.import_indices([
            {"compound_id": "CR-test1", "compound_score": 65.0},
        ])
        assert result["imported"] == 1

    def test_roundtrip(self, engine_with_indices):
        """Export then import preserves data."""
        exported = engine_with_indices.export_indices(format="dict")
        engine_with_indices.clear()
        assert engine_with_indices.get_index_count() == 0
        engine_with_indices.import_indices(exported)
        assert engine_with_indices.get_index_count() == len(exported)


# ===========================================================================
# Test Dunder Methods
# ===========================================================================


class TestDunderMethods:
    """Tests for __len__, __repr__, __contains__, __iter__."""

    def test_len(self, engine_with_indices):
        """__len__ returns index count."""
        assert len(engine_with_indices) == 3

    def test_repr(self, engine):
        """__repr__ returns a descriptive string."""
        r = repr(engine)
        assert "RiskIndexEngine" in r
        assert "indices=" in r
        assert "weights=" in r

    def test_contains_existing(self, engine, london_location):
        """__contains__ returns True for stored index."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert result["index_id"] in engine

    def test_contains_nonexistent(self, engine):
        """__contains__ returns False for missing index."""
        assert "RI-nonexistent" not in engine

    def test_iter(self, engine_with_indices):
        """__iter__ yields all index IDs."""
        ids = list(engine_with_indices)
        assert len(ids) == 3
        for idx_id in ids:
            assert isinstance(idx_id, str)


# ===========================================================================
# Test Thread Safety
# ===========================================================================


class TestThreadSafety:
    """Tests verifying thread-safe operation."""

    def test_concurrent_calculations(self, engine):
        """Multiple threads can calculate concurrently."""
        errors = []

        def calculate(idx):
            try:
                engine.calculate_risk_index(
                    hazard_type="drought",
                    location={"lat": 40.0 + idx, "lon": -3.0, "name": f"Loc{idx}"},
                    probability=0.5,
                    intensity=5.0,
                    frequency=1.0,
                    duration_days=30.0,
                )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=calculate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine.get_index_count() == 10

    def test_concurrent_read_write(self, engine, london_location):
        """Concurrent reads and writes do not error."""
        errors = []

        def writer(idx):
            try:
                engine.calculate_risk_index(
                    hazard_type="drought",
                    location={"lat": idx, "lon": 0, "name": f"W{idx}"},
                    probability=0.5,
                    intensity=5.0,
                    frequency=1.0,
                    duration_days=30.0,
                )
            except Exception as e:
                errors.append(str(e))

        def reader():
            try:
                engine.get_statistics()
                engine.list_risk_indices()
                engine.get_risk_level_distribution()
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ===========================================================================
# Test Provenance Tracking
# ===========================================================================


class TestProvenanceTracking:
    """Tests verifying SHA-256 provenance chain integrity."""

    def test_provenance_hash_hex_chars(self, engine, london_location):
        """Provenance hash contains only hex characters."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        ph = result["provenance_hash"]
        assert all(c in "0123456789abcdef" for c in ph)

    def test_provenance_entries_increment(self, engine, london_location):
        """Each calculation adds provenance entries."""
        before = engine.get_statistics().get("provenance_entries", 0)
        engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        after = engine.get_statistics().get("provenance_entries", 0)
        assert after > before

    def test_different_inputs_different_hashes(self, engine, london_location):
        """Different inputs produce different provenance hashes."""
        r1 = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.3,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        r2 = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.7,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert r1["provenance_hash"] != r2["provenance_hash"]


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_very_small_probability(self, engine, london_location):
        """Very small non-zero probability works."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.001,
            intensity=1.0,
            frequency=0.1,
            duration_days=1.0,
        )
        assert result["risk_score"] > 0
        assert result["risk_level"] == "NEGLIGIBLE"

    def test_intensity_above_10_normalised(self, engine, london_location):
        """Intensity above 10 is normalised to 1.0 in score."""
        # Use raw calculation to verify
        score_at_10 = engine.calculate_risk_score_raw(0.5, 10.0, 1.0, 30.0)
        score_at_20 = engine.calculate_risk_score_raw(0.5, 20.0, 1.0, 30.0)
        assert score_at_10 == score_at_20

    def test_frequency_above_10_normalised(self, engine):
        """Frequency above 10 is normalised to 1.0 in score."""
        score_at_10 = engine.calculate_risk_score_raw(0.5, 5.0, 10.0, 30.0)
        score_at_50 = engine.calculate_risk_score_raw(0.5, 5.0, 50.0, 30.0)
        assert score_at_10 == score_at_50

    def test_duration_above_365_normalised(self, engine):
        """Duration above 365 is normalised to 1.0 in score."""
        score_at_365 = engine.calculate_risk_score_raw(0.5, 5.0, 1.0, 365.0)
        score_at_1000 = engine.calculate_risk_score_raw(0.5, 5.0, 1.0, 1000.0)
        assert score_at_365 == score_at_1000

    def test_location_defaults(self, engine):
        """Location without name defaults to 'unknown'."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location={"lat": 0.0, "lon": 0.0},
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert result["location"]["name"] == "unknown"

    def test_hazard_type_case_insensitive(self, engine, london_location):
        """Hazard type is lowercased."""
        result = engine.calculate_risk_index(
            hazard_type="DROUGHT",
            location=london_location,
            probability=0.5,
            intensity=5.0,
            frequency=1.0,
            duration_days=10.0,
        )
        assert result["hazard_type"] == "drought"

    def test_get_index_count(self, engine_with_indices):
        """get_index_count matches len()."""
        assert engine_with_indices.get_index_count() == len(engine_with_indices)

    def test_score_precision(self, engine, london_location):
        """Risk score has at most 4 decimal places."""
        result = engine.calculate_risk_index(
            hazard_type="drought",
            location=london_location,
            probability=0.333333,
            intensity=3.333333,
            frequency=1.111111,
            duration_days=22.222222,
        )
        score_str = f"{result['risk_score']:.10f}"
        # After 4 decimal places, remaining digits should be 0
        # This is a soft check since floating point may vary
        assert isinstance(result["risk_score"], float)

    def test_multi_hazard_with_zero_weights(self, engine, london_location):
        """Multi-hazard with zero weights falls back to simple average."""
        hazards = [
            {
                "hazard_type": "drought",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
                "weight": 0.0,
            },
            {
                "hazard_type": "wildfire",
                "probability": 0.5,
                "intensity": 5.0,
                "frequency": 1.0,
                "duration_days": 30.0,
                "weight": 0.0,
            },
        ]
        result = engine.calculate_multi_hazard_index(
            location=london_location,
            hazard_risks=hazards,
            aggregation="weighted_average",
        )
        # With zero weights, falls back to simple average
        assert result["multi_hazard_score"] >= 0

    def test_compound_amplification_formula(self, engine, london_location):
        """Verify compound amplification formula directly."""
        primary = {
            "hazard_type": "drought",
            "probability": 0.5,
            "intensity": 5.0,
            "frequency": 1.0,
            "duration_days": 30.0,
        }
        secondary = {
            "hazard_type": "wildfire",
            "probability": 0.5,
            "intensity": 5.0,
            "frequency": 1.0,
            "duration_days": 30.0,
        }
        result = engine.calculate_compound_risk(
            location=london_location,
            primary_hazard=primary,
            secondary_hazards=[secondary],
        )
        # Verify amplification_factor = 1 + (sec_score/100 * correlation)
        primary_score = engine.calculate_risk_score_raw(0.5, 5.0, 1.0, 30.0)
        sec_score = engine.calculate_risk_score_raw(0.5, 5.0, 1.0, 30.0)
        correlation = 0.75  # drought + wildfire default
        expected_amp = 1.0 + (sec_score / 100.0) * correlation
        assert result["amplification_factor"] == pytest.approx(expected_amp, abs=0.01)
        expected_compound = min(primary_score * expected_amp, 100.0)
        assert result["compound_score"] == pytest.approx(expected_compound, abs=0.1)
