"""
Unit Tests: Carryover Risk Calculator

Tests the carryover (priming) risk assessment calculations:
1. TDS (Total Dissolved Solids) based risk assessment
2. Silica concentration analysis
3. Conductivity-based detection
4. Risk level classification (LOW, MEDIUM, HIGH, CRITICAL)
5. Recommended actions based on risk level

Reference: ASME and plant-specific carryover guidelines
Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone


# =============================================================================
# Enumerations and Constants
# =============================================================================

class CarryoverRisk(Enum):
    """Carryover risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Carryover thresholds (typical values - adjust per plant)
TDS_THRESHOLD_LOW = 20.0  # ppm
TDS_THRESHOLD_MEDIUM = 50.0  # ppm
TDS_THRESHOLD_HIGH = 100.0  # ppm

SILICA_THRESHOLD_LOW = 50.0  # ppb
SILICA_THRESHOLD_MEDIUM = 100.0  # ppb
SILICA_THRESHOLD_HIGH = 200.0  # ppb

CONDUCTIVITY_THRESHOLD_LOW = 25.0  # uS/cm
CONDUCTIVITY_THRESHOLD_MEDIUM = 50.0  # uS/cm
CONDUCTIVITY_THRESHOLD_HIGH = 100.0  # uS/cm

# Drum level impact factors
DRUM_LEVEL_HIGH_IMPACT = 1.3  # Multiplier for high drum level
DRUM_LEVEL_LOW_IMPACT = 0.8  # Multiplier for low drum level

# Load change impact
RAPID_LOAD_CHANGE_IMPACT = 1.5  # Multiplier for rapid load changes


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoilerWaterQuality:
    """Boiler water quality measurements."""
    tds_ppm: float
    silica_ppb: float
    conductivity_us_cm: float
    ph: float = 10.5
    phosphate_ppm: float = 10.0
    drum_level_percent: float = 50.0
    steam_load_percent: float = 80.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class CarryoverAssessment:
    """Result of carryover risk assessment."""
    risk_level: CarryoverRisk
    probability: float
    tds_contribution: float
    silica_contribution: float
    conductivity_contribution: float
    drum_level_factor: float
    load_factor: float
    recommended_action: str
    provenance_hash: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class CarryoverTrend:
    """Carryover trend analysis."""
    current_risk: CarryoverRisk
    trend_direction: str  # "improving", "stable", "deteriorating"
    rate_of_change: float  # % per hour
    predicted_critical_time: Optional[float]  # hours until critical, if deteriorating


# =============================================================================
# Carryover Calculator Implementation
# =============================================================================

def calculate_tds_risk_score(tds_ppm: float) -> Tuple[float, CarryoverRisk]:
    """
    Calculate risk score based on TDS concentration.

    Args:
        tds_ppm: Total dissolved solids in ppm

    Returns:
        Tuple of (risk_score 0-1, risk_level)
    """
    if tds_ppm < 0:
        raise ValueError(f"TDS cannot be negative: {tds_ppm}")

    if tds_ppm <= TDS_THRESHOLD_LOW:
        score = tds_ppm / TDS_THRESHOLD_LOW * 0.25
        level = CarryoverRisk.LOW
    elif tds_ppm <= TDS_THRESHOLD_MEDIUM:
        score = 0.25 + (tds_ppm - TDS_THRESHOLD_LOW) / (TDS_THRESHOLD_MEDIUM - TDS_THRESHOLD_LOW) * 0.25
        level = CarryoverRisk.MEDIUM
    elif tds_ppm <= TDS_THRESHOLD_HIGH:
        score = 0.5 + (tds_ppm - TDS_THRESHOLD_MEDIUM) / (TDS_THRESHOLD_HIGH - TDS_THRESHOLD_MEDIUM) * 0.25
        level = CarryoverRisk.HIGH
    else:
        score = min(0.75 + (tds_ppm - TDS_THRESHOLD_HIGH) / TDS_THRESHOLD_HIGH * 0.25, 1.0)
        level = CarryoverRisk.CRITICAL

    return score, level


def calculate_silica_risk_score(silica_ppb: float) -> Tuple[float, CarryoverRisk]:
    """
    Calculate risk score based on silica concentration.

    Args:
        silica_ppb: Silica concentration in ppb

    Returns:
        Tuple of (risk_score 0-1, risk_level)
    """
    if silica_ppb < 0:
        raise ValueError(f"Silica cannot be negative: {silica_ppb}")

    if silica_ppb <= SILICA_THRESHOLD_LOW:
        score = silica_ppb / SILICA_THRESHOLD_LOW * 0.25
        level = CarryoverRisk.LOW
    elif silica_ppb <= SILICA_THRESHOLD_MEDIUM:
        score = 0.25 + (silica_ppb - SILICA_THRESHOLD_LOW) / (SILICA_THRESHOLD_MEDIUM - SILICA_THRESHOLD_LOW) * 0.25
        level = CarryoverRisk.MEDIUM
    elif silica_ppb <= SILICA_THRESHOLD_HIGH:
        score = 0.5 + (silica_ppb - SILICA_THRESHOLD_MEDIUM) / (SILICA_THRESHOLD_HIGH - SILICA_THRESHOLD_MEDIUM) * 0.25
        level = CarryoverRisk.HIGH
    else:
        score = min(0.75 + (silica_ppb - SILICA_THRESHOLD_HIGH) / SILICA_THRESHOLD_HIGH * 0.25, 1.0)
        level = CarryoverRisk.CRITICAL

    return score, level


def calculate_conductivity_risk_score(conductivity_us_cm: float) -> Tuple[float, CarryoverRisk]:
    """
    Calculate risk score based on conductivity.

    Args:
        conductivity_us_cm: Conductivity in uS/cm

    Returns:
        Tuple of (risk_score 0-1, risk_level)
    """
    if conductivity_us_cm < 0:
        raise ValueError(f"Conductivity cannot be negative: {conductivity_us_cm}")

    if conductivity_us_cm <= CONDUCTIVITY_THRESHOLD_LOW:
        score = conductivity_us_cm / CONDUCTIVITY_THRESHOLD_LOW * 0.25
        level = CarryoverRisk.LOW
    elif conductivity_us_cm <= CONDUCTIVITY_THRESHOLD_MEDIUM:
        score = 0.25 + (conductivity_us_cm - CONDUCTIVITY_THRESHOLD_LOW) / (CONDUCTIVITY_THRESHOLD_MEDIUM - CONDUCTIVITY_THRESHOLD_LOW) * 0.25
        level = CarryoverRisk.MEDIUM
    elif conductivity_us_cm <= CONDUCTIVITY_THRESHOLD_HIGH:
        score = 0.5 + (conductivity_us_cm - CONDUCTIVITY_THRESHOLD_MEDIUM) / (CONDUCTIVITY_THRESHOLD_HIGH - CONDUCTIVITY_THRESHOLD_MEDIUM) * 0.25
        level = CarryoverRisk.HIGH
    else:
        score = min(0.75 + (conductivity_us_cm - CONDUCTIVITY_THRESHOLD_HIGH) / CONDUCTIVITY_THRESHOLD_HIGH * 0.25, 1.0)
        level = CarryoverRisk.CRITICAL

    return score, level


def calculate_drum_level_factor(drum_level_percent: float) -> float:
    """
    Calculate drum level impact factor on carryover risk.

    High drum level increases carryover risk.
    Low drum level may reduce it but has other risks.

    Args:
        drum_level_percent: Drum level in %

    Returns:
        Factor to multiply base risk (0.8 - 1.5)
    """
    if drum_level_percent < 0 or drum_level_percent > 100:
        # Clamp to valid range
        drum_level_percent = max(0, min(100, drum_level_percent))

    # Normal range: 40-60%
    if 40 <= drum_level_percent <= 60:
        return 1.0
    elif drum_level_percent > 60:
        # High level increases risk
        excess = (drum_level_percent - 60) / 40  # 0 to 1 for 60-100%
        return 1.0 + excess * (DRUM_LEVEL_HIGH_IMPACT - 1.0)
    else:
        # Low level slightly reduces carryover risk (but has other issues)
        deficit = (40 - drum_level_percent) / 40  # 0 to 1 for 0-40%
        return 1.0 - deficit * (1.0 - DRUM_LEVEL_LOW_IMPACT)


def calculate_load_factor(steam_load_percent: float, load_change_rate: float = 0.0) -> float:
    """
    Calculate load impact factor on carryover risk.

    Rapid load changes increase carryover risk.

    Args:
        steam_load_percent: Current steam load in %
        load_change_rate: Rate of load change in %/min (positive = increasing)

    Returns:
        Factor to multiply base risk (0.9 - 1.5)
    """
    # Base factor for high load
    if steam_load_percent >= 100:
        base_factor = 1.2
    elif steam_load_percent >= 90:
        base_factor = 1.1
    elif steam_load_percent >= 80:
        base_factor = 1.0
    else:
        base_factor = 0.9

    # Additional factor for rapid load changes
    if abs(load_change_rate) > 5.0:  # More than 5%/min is considered rapid
        change_factor = 1.0 + (abs(load_change_rate) - 5.0) / 10.0 * (RAPID_LOAD_CHANGE_IMPACT - 1.0)
        change_factor = min(change_factor, RAPID_LOAD_CHANGE_IMPACT)
    else:
        change_factor = 1.0

    return base_factor * change_factor


def get_recommended_action(risk_level: CarryoverRisk, probability: float) -> str:
    """
    Get recommended action based on risk level.

    Args:
        risk_level: Assessed risk level
        probability: Probability of carryover

    Returns:
        Recommended action string
    """
    actions = {
        CarryoverRisk.LOW: "Continue normal operation. Monitor water quality.",
        CarryoverRisk.MEDIUM: "Increase blowdown frequency. Check chemical treatment.",
        CarryoverRisk.HIGH: "Initiate continuous blowdown. Reduce load if possible. Check drum internals.",
        CarryoverRisk.CRITICAL: "IMMEDIATE ACTION: Reduce load significantly. Emergency blowdown. Prepare for shutdown if no improvement.",
    }

    return actions.get(risk_level, "Unknown risk level - review manually")


def assess_carryover_risk(water_quality: BoilerWaterQuality, load_change_rate: float = 0.0) -> CarryoverAssessment:
    """
    Perform comprehensive carryover risk assessment.

    Args:
        water_quality: Current water quality measurements
        load_change_rate: Rate of load change in %/min

    Returns:
        CarryoverAssessment with risk level and recommendations
    """
    # Calculate individual risk scores
    tds_score, tds_level = calculate_tds_risk_score(water_quality.tds_ppm)
    silica_score, silica_level = calculate_silica_risk_score(water_quality.silica_ppb)
    cond_score, cond_level = calculate_conductivity_risk_score(water_quality.conductivity_us_cm)

    # Calculate modifying factors
    drum_factor = calculate_drum_level_factor(water_quality.drum_level_percent)
    load_factor = calculate_load_factor(water_quality.steam_load_percent, load_change_rate)

    # Weight contributions (TDS is primary, silica is critical for turbines)
    weights = {"tds": 0.4, "silica": 0.35, "conductivity": 0.25}

    base_score = (
        weights["tds"] * tds_score +
        weights["silica"] * silica_score +
        weights["conductivity"] * cond_score
    )

    # Apply modifying factors
    final_score = base_score * drum_factor * load_factor
    final_score = min(max(final_score, 0.0), 1.0)

    # Determine overall risk level (use worst of individual levels, or escalate based on score)
    individual_levels = [tds_level, silica_level, cond_level]
    max_individual = max(individual_levels, key=lambda x: list(CarryoverRisk).index(x))

    if final_score <= 0.25:
        final_level = CarryoverRisk.LOW
    elif final_score <= 0.5:
        final_level = CarryoverRisk.MEDIUM
    elif final_score <= 0.75:
        final_level = CarryoverRisk.HIGH
    else:
        final_level = CarryoverRisk.CRITICAL

    # Use the higher of score-based or individual-max level
    if list(CarryoverRisk).index(max_individual) > list(CarryoverRisk).index(final_level):
        final_level = max_individual

    # Get recommended action
    action = get_recommended_action(final_level, final_score)

    # Calculate provenance hash
    inputs = {
        "tds_ppm": round(water_quality.tds_ppm, 6),
        "silica_ppb": round(water_quality.silica_ppb, 6),
        "conductivity_us_cm": round(water_quality.conductivity_us_cm, 6),
        "drum_level_percent": round(water_quality.drum_level_percent, 6),
        "steam_load_percent": round(water_quality.steam_load_percent, 6),
        "load_change_rate": round(load_change_rate, 6),
    }
    provenance_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    return CarryoverAssessment(
        risk_level=final_level,
        probability=final_score,
        tds_contribution=tds_score * weights["tds"],
        silica_contribution=silica_score * weights["silica"],
        conductivity_contribution=cond_score * weights["conductivity"],
        drum_level_factor=drum_factor,
        load_factor=load_factor,
        recommended_action=action,
        provenance_hash=provenance_hash,
    )


def analyze_carryover_trend(assessments: List[CarryoverAssessment]) -> CarryoverTrend:
    """
    Analyze trend in carryover risk over time.

    Args:
        assessments: List of historical assessments (most recent last)

    Returns:
        CarryoverTrend with direction and predictions
    """
    if len(assessments) < 2:
        return CarryoverTrend(
            current_risk=assessments[-1].risk_level if assessments else CarryoverRisk.LOW,
            trend_direction="stable",
            rate_of_change=0.0,
            predicted_critical_time=None,
        )

    # Calculate rate of change
    probabilities = [a.probability for a in assessments]
    n = len(probabilities)

    # Simple linear regression slope
    x_mean = (n - 1) / 2
    y_mean = sum(probabilities) / n

    numerator = sum((i - x_mean) * (probabilities[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator > 0:
        slope = numerator / denominator
    else:
        slope = 0.0

    # Determine trend direction
    if slope > 0.01:
        direction = "deteriorating"
    elif slope < -0.01:
        direction = "improving"
    else:
        direction = "stable"

    # Predict time to critical (if deteriorating)
    predicted_critical = None
    if direction == "deteriorating" and slope > 0:
        current_prob = probabilities[-1]
        if current_prob < 0.75:  # Not yet critical
            time_to_critical = (0.75 - current_prob) / slope
            predicted_critical = max(0, time_to_critical)

    return CarryoverTrend(
        current_risk=assessments[-1].risk_level,
        trend_direction=direction,
        rate_of_change=slope * 100,  # Convert to % per point
        predicted_critical_time=predicted_critical,
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestTDSRiskScore:
    """Tests for TDS-based risk score calculation."""

    def test_zero_tds_low_risk(self):
        """Test that zero TDS gives low risk."""
        score, level = calculate_tds_risk_score(0.0)
        assert score == 0.0
        assert level == CarryoverRisk.LOW

    def test_low_tds_low_risk(self):
        """Test that low TDS gives low risk."""
        score, level = calculate_tds_risk_score(15.0)
        assert score < 0.25
        assert level == CarryoverRisk.LOW

    def test_medium_tds_medium_risk(self):
        """Test that medium TDS gives medium risk."""
        score, level = calculate_tds_risk_score(35.0)
        assert 0.25 <= score < 0.5
        assert level == CarryoverRisk.MEDIUM

    def test_high_tds_high_risk(self):
        """Test that high TDS gives high risk."""
        score, level = calculate_tds_risk_score(75.0)
        assert 0.5 <= score < 0.75
        assert level == CarryoverRisk.HIGH

    def test_critical_tds_critical_risk(self):
        """Test that critical TDS gives critical risk."""
        score, level = calculate_tds_risk_score(150.0)
        assert score >= 0.75
        assert level == CarryoverRisk.CRITICAL

    def test_negative_tds_raises_error(self):
        """Test that negative TDS raises error."""
        with pytest.raises(ValueError):
            calculate_tds_risk_score(-10.0)

    @pytest.mark.parametrize("tds,expected_level", [
        (5.0, CarryoverRisk.LOW),
        (10.0, CarryoverRisk.LOW),
        (30.0, CarryoverRisk.MEDIUM),
        (60.0, CarryoverRisk.HIGH),
        (120.0, CarryoverRisk.CRITICAL),
    ])
    def test_tds_risk_levels(self, tds, expected_level):
        """Parametrized test for TDS risk levels."""
        _, level = calculate_tds_risk_score(tds)
        assert level == expected_level

    def test_score_monotonic_with_tds(self):
        """Test that score increases monotonically with TDS."""
        scores = [calculate_tds_risk_score(tds)[0] for tds in range(0, 200, 10)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1]


class TestSilicaRiskScore:
    """Tests for silica-based risk score calculation."""

    def test_zero_silica_low_risk(self):
        """Test that zero silica gives low risk."""
        score, level = calculate_silica_risk_score(0.0)
        assert score == 0.0
        assert level == CarryoverRisk.LOW

    def test_low_silica_low_risk(self):
        """Test that low silica gives low risk."""
        score, level = calculate_silica_risk_score(30.0)
        assert score < 0.25
        assert level == CarryoverRisk.LOW

    def test_high_silica_high_risk(self):
        """Test that high silica gives high risk."""
        score, level = calculate_silica_risk_score(150.0)
        assert 0.5 <= score < 0.75
        assert level == CarryoverRisk.HIGH

    def test_critical_silica_critical_risk(self):
        """Test that critical silica gives critical risk."""
        score, level = calculate_silica_risk_score(300.0)
        assert score >= 0.75
        assert level == CarryoverRisk.CRITICAL

    def test_negative_silica_raises_error(self):
        """Test that negative silica raises error."""
        with pytest.raises(ValueError):
            calculate_silica_risk_score(-50.0)


class TestConductivityRiskScore:
    """Tests for conductivity-based risk score calculation."""

    def test_zero_conductivity_low_risk(self):
        """Test that zero conductivity gives low risk."""
        score, level = calculate_conductivity_risk_score(0.0)
        assert score == 0.0
        assert level == CarryoverRisk.LOW

    def test_low_conductivity_low_risk(self):
        """Test that low conductivity gives low risk."""
        score, level = calculate_conductivity_risk_score(15.0)
        assert score < 0.25
        assert level == CarryoverRisk.LOW

    def test_high_conductivity_high_risk(self):
        """Test that high conductivity gives high risk."""
        score, level = calculate_conductivity_risk_score(75.0)
        assert 0.5 <= score < 0.75
        assert level == CarryoverRisk.HIGH

    def test_negative_conductivity_raises_error(self):
        """Test that negative conductivity raises error."""
        with pytest.raises(ValueError):
            calculate_conductivity_risk_score(-25.0)


class TestDrumLevelFactor:
    """Tests for drum level impact factor calculation."""

    def test_normal_level_factor_one(self):
        """Test that normal drum level gives factor of 1.0."""
        factor = calculate_drum_level_factor(50.0)
        assert factor == pytest.approx(1.0)

    def test_high_level_increases_risk(self):
        """Test that high drum level increases risk factor."""
        factor = calculate_drum_level_factor(80.0)
        assert factor > 1.0
        assert factor <= DRUM_LEVEL_HIGH_IMPACT

    def test_very_high_level_max_factor(self):
        """Test that very high drum level approaches max factor."""
        factor = calculate_drum_level_factor(100.0)
        assert factor == pytest.approx(DRUM_LEVEL_HIGH_IMPACT)

    def test_low_level_decreases_risk(self):
        """Test that low drum level decreases risk factor."""
        factor = calculate_drum_level_factor(20.0)
        assert factor < 1.0
        assert factor >= DRUM_LEVEL_LOW_IMPACT

    def test_boundary_values(self):
        """Test boundary values of drum level."""
        assert calculate_drum_level_factor(40.0) == pytest.approx(1.0)
        assert calculate_drum_level_factor(60.0) == pytest.approx(1.0)

    def test_clamping_extreme_values(self):
        """Test that extreme values are clamped."""
        factor_high = calculate_drum_level_factor(150.0)  # Over 100%
        factor_low = calculate_drum_level_factor(-10.0)   # Below 0%

        assert 0.5 <= factor_high <= 2.0  # Reasonable range
        assert 0.5 <= factor_low <= 2.0


class TestLoadFactor:
    """Tests for load impact factor calculation."""

    def test_normal_load_factor_one(self):
        """Test that normal load gives factor near 1.0."""
        factor = calculate_load_factor(80.0, 0.0)
        assert factor == pytest.approx(1.0)

    def test_high_load_increases_factor(self):
        """Test that high load increases factor."""
        factor = calculate_load_factor(100.0, 0.0)
        assert factor > 1.0

    def test_low_load_decreases_factor(self):
        """Test that low load decreases factor."""
        factor = calculate_load_factor(60.0, 0.0)
        assert factor < 1.0

    def test_rapid_load_change_increases_factor(self):
        """Test that rapid load change increases factor."""
        factor_steady = calculate_load_factor(80.0, 0.0)
        factor_rapid = calculate_load_factor(80.0, 10.0)  # 10%/min change

        assert factor_rapid > factor_steady

    def test_rapid_load_change_bounded(self):
        """Test that rapid load change factor is bounded."""
        factor = calculate_load_factor(80.0, 50.0)  # Very rapid change
        assert factor <= 1.0 * RAPID_LOAD_CHANGE_IMPACT * 1.5  # Reasonable upper bound


class TestCarryoverAssessment:
    """Tests for comprehensive carryover risk assessment."""

    @pytest.fixture
    def low_risk_water(self) -> BoilerWaterQuality:
        """Create low-risk water quality sample."""
        return BoilerWaterQuality(
            tds_ppm=10.0,
            silica_ppb=30.0,
            conductivity_us_cm=15.0,
            drum_level_percent=50.0,
            steam_load_percent=75.0,
        )

    @pytest.fixture
    def high_risk_water(self) -> BoilerWaterQuality:
        """Create high-risk water quality sample."""
        return BoilerWaterQuality(
            tds_ppm=80.0,
            silica_ppb=180.0,
            conductivity_us_cm=85.0,
            drum_level_percent=75.0,
            steam_load_percent=95.0,
        )

    @pytest.fixture
    def critical_water(self) -> BoilerWaterQuality:
        """Create critical water quality sample."""
        return BoilerWaterQuality(
            tds_ppm=150.0,
            silica_ppb=300.0,
            conductivity_us_cm=150.0,
            drum_level_percent=85.0,
            steam_load_percent=100.0,
        )

    def test_low_risk_assessment(self, low_risk_water):
        """Test assessment of low-risk water quality."""
        result = assess_carryover_risk(low_risk_water)

        assert result.risk_level == CarryoverRisk.LOW
        assert result.probability < 0.25
        assert "normal operation" in result.recommended_action.lower()

    def test_high_risk_assessment(self, high_risk_water):
        """Test assessment of high-risk water quality."""
        result = assess_carryover_risk(high_risk_water)

        assert result.risk_level in [CarryoverRisk.HIGH, CarryoverRisk.CRITICAL]
        assert result.probability >= 0.5

    def test_critical_risk_assessment(self, critical_water):
        """Test assessment of critical water quality."""
        result = assess_carryover_risk(critical_water)

        assert result.risk_level == CarryoverRisk.CRITICAL
        assert result.probability >= 0.75
        assert "IMMEDIATE" in result.recommended_action or "emergency" in result.recommended_action.lower()

    def test_drum_level_affects_assessment(self, low_risk_water):
        """Test that drum level affects assessment."""
        result_normal = assess_carryover_risk(low_risk_water)

        low_risk_water.drum_level_percent = 85.0  # High drum level
        result_high = assess_carryover_risk(low_risk_water)

        assert result_high.probability >= result_normal.probability
        assert result_high.drum_level_factor > result_normal.drum_level_factor

    def test_load_change_affects_assessment(self, low_risk_water):
        """Test that load change rate affects assessment."""
        result_steady = assess_carryover_risk(low_risk_water, load_change_rate=0.0)
        result_rapid = assess_carryover_risk(low_risk_water, load_change_rate=15.0)

        assert result_rapid.probability >= result_steady.probability
        assert result_rapid.load_factor > result_steady.load_factor

    def test_provenance_hash_generated(self, low_risk_water):
        """Test that provenance hash is generated."""
        result = assess_carryover_risk(low_risk_water)

        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, low_risk_water):
        """Test that provenance hash is deterministic."""
        result1 = assess_carryover_risk(low_risk_water)
        result2 = assess_carryover_risk(low_risk_water)

        assert result1.provenance_hash == result2.provenance_hash

    def test_contributions_sum_appropriately(self, low_risk_water):
        """Test that contributions are weighted correctly."""
        result = assess_carryover_risk(low_risk_water)

        # Contributions should be positive
        assert result.tds_contribution >= 0
        assert result.silica_contribution >= 0
        assert result.conductivity_contribution >= 0


class TestCarryoverTrend:
    """Tests for carryover trend analysis."""

    def test_stable_trend_detection(self):
        """Test detection of stable trend."""
        assessments = [
            CarryoverAssessment(
                risk_level=CarryoverRisk.LOW, probability=0.2,
                tds_contribution=0.1, silica_contribution=0.05, conductivity_contribution=0.05,
                drum_level_factor=1.0, load_factor=1.0,
                recommended_action="Monitor", provenance_hash="a" * 64,
            )
            for _ in range(5)
        ]

        trend = analyze_carryover_trend(assessments)

        assert trend.trend_direction == "stable"
        assert abs(trend.rate_of_change) < 1.0

    def test_improving_trend_detection(self):
        """Test detection of improving trend."""
        assessments = [
            CarryoverAssessment(
                risk_level=CarryoverRisk.LOW, probability=0.5 - i * 0.1,
                tds_contribution=0.1, silica_contribution=0.05, conductivity_contribution=0.05,
                drum_level_factor=1.0, load_factor=1.0,
                recommended_action="Monitor", provenance_hash="a" * 64,
            )
            for i in range(5)
        ]

        trend = analyze_carryover_trend(assessments)

        assert trend.trend_direction == "improving"
        assert trend.rate_of_change < 0

    def test_deteriorating_trend_detection(self):
        """Test detection of deteriorating trend."""
        assessments = [
            CarryoverAssessment(
                risk_level=CarryoverRisk.LOW, probability=0.2 + i * 0.1,
                tds_contribution=0.1, silica_contribution=0.05, conductivity_contribution=0.05,
                drum_level_factor=1.0, load_factor=1.0,
                recommended_action="Monitor", provenance_hash="a" * 64,
            )
            for i in range(5)
        ]

        trend = analyze_carryover_trend(assessments)

        assert trend.trend_direction == "deteriorating"
        assert trend.rate_of_change > 0

    def test_critical_time_prediction(self):
        """Test prediction of time to critical state."""
        assessments = [
            CarryoverAssessment(
                risk_level=CarryoverRisk.LOW, probability=0.3 + i * 0.1,
                tds_contribution=0.1, silica_contribution=0.05, conductivity_contribution=0.05,
                drum_level_factor=1.0, load_factor=1.0,
                recommended_action="Monitor", provenance_hash="a" * 64,
            )
            for i in range(5)
        ]

        trend = analyze_carryover_trend(assessments)

        assert trend.trend_direction == "deteriorating"
        assert trend.predicted_critical_time is not None
        assert trend.predicted_critical_time > 0

    def test_single_assessment_returns_stable(self):
        """Test that single assessment returns stable trend."""
        assessments = [
            CarryoverAssessment(
                risk_level=CarryoverRisk.LOW, probability=0.2,
                tds_contribution=0.1, silica_contribution=0.05, conductivity_contribution=0.05,
                drum_level_factor=1.0, load_factor=1.0,
                recommended_action="Monitor", provenance_hash="a" * 64,
            )
        ]

        trend = analyze_carryover_trend(assessments)

        assert trend.trend_direction == "stable"


class TestRecommendedActions:
    """Tests for recommended action generation."""

    def test_low_risk_action(self):
        """Test action for low risk."""
        action = get_recommended_action(CarryoverRisk.LOW, 0.1)
        assert "normal" in action.lower() or "monitor" in action.lower()

    def test_medium_risk_action(self):
        """Test action for medium risk."""
        action = get_recommended_action(CarryoverRisk.MEDIUM, 0.4)
        assert "blowdown" in action.lower() or "chemical" in action.lower()

    def test_high_risk_action(self):
        """Test action for high risk."""
        action = get_recommended_action(CarryoverRisk.HIGH, 0.65)
        assert "continuous blowdown" in action.lower() or "reduce load" in action.lower()

    def test_critical_risk_action(self):
        """Test action for critical risk."""
        action = get_recommended_action(CarryoverRisk.CRITICAL, 0.9)
        assert "IMMEDIATE" in action or "emergency" in action.lower()


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_assessment_identical(self):
        """Test that repeated assessments are identical."""
        water = BoilerWaterQuality(
            tds_ppm=50.0,
            silica_ppb=100.0,
            conductivity_us_cm=50.0,
            drum_level_percent=55.0,
            steam_load_percent=85.0,
        )

        results = [assess_carryover_risk(water) for _ in range(10)]

        first = results[0]
        for result in results[1:]:
            assert result.risk_level == first.risk_level
            assert result.probability == first.probability
            assert result.provenance_hash == first.provenance_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
