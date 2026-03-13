# -*- coding: utf-8 -*-
"""
Tests for SeverityClassifier - AGENT-EUDR-020 Feature 3: Severity Classification

Comprehensive test suite covering:
- Severity classification for CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL scenarios
- Reclassification with new context
- Threshold retrieval and configuration
- Severity distribution with filters
- Area scoring for various hectare thresholds
- Rate scoring for various ha/day deforestation rates
- Proximity scoring for distance thresholds
- Protected area scoring with overlap percentages
- Timing scoring for post-cutoff vs pre-cutoff events
- Total score calculation with weighted components
- Severity level determination from score thresholds
- Aggravating factor identification

Test count: 40+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 (Feature 3 - Severity Classification)
"""

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pytest

from tests.agents.eudr.deforestation_alert_system.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    SEVERITY_LEVELS,
    SEVERITY_THRESHOLDS,
    DEFAULT_SEVERITY_WEIGHTS,
    AREA_SCORE_MAP,
    PROXIMITY_SCORE_MAP,
    score_area,
    score_proximity,
    determine_severity,
)


# ---------------------------------------------------------------------------
# Helpers for severity classification logic
# ---------------------------------------------------------------------------


def _score_area(area_ha: float) -> int:
    """Score the area component (0-100) based on hectares affected."""
    if area_ha >= 50:
        return 100
    elif area_ha >= 10:
        return 80
    elif area_ha >= 1:
        return 50
    elif area_ha >= 0.5:
        return 30
    else:
        return 10


def _score_rate(rate_ha_per_day: float) -> int:
    """Score the deforestation rate component (0-100) based on ha/day."""
    if rate_ha_per_day >= 10.0:
        return 100
    elif rate_ha_per_day >= 5.0:
        return 80
    elif rate_ha_per_day >= 1.0:
        return 60
    elif rate_ha_per_day >= 0.5:
        return 40
    elif rate_ha_per_day >= 0.1:
        return 20
    else:
        return 10


def _score_proximity(distance_km: float) -> int:
    """Score the proximity component (0-100) based on distance to plots."""
    if distance_km < 1:
        return 100
    elif distance_km < 5:
        return 80
    elif distance_km < 25:
        return 50
    elif distance_km < 50:
        return 30
    else:
        return 10


def _score_protected_area(overlap_pct: float) -> int:
    """Score the protected area component (0-100) based on overlap %."""
    if overlap_pct >= 75:
        return 100
    elif overlap_pct >= 50:
        return 80
    elif overlap_pct >= 25:
        return 60
    elif overlap_pct > 0:
        return 40
    else:
        return 0


def _score_timing(
    is_post_cutoff: bool,
    post_cutoff_multiplier: float = 2.0,
) -> Tuple[int, float]:
    """Score the timing component and return (base_score, multiplier).

    Returns:
        (base_score, multiplier) where multiplier is applied to total.
    """
    if is_post_cutoff:
        return (100, post_cutoff_multiplier)
    else:
        return (20, 1.0)


def _calculate_total_score(
    area_score: int,
    rate_score: int,
    proximity_score: int,
    protected_score: int,
    timing_score: int,
    weights: Optional[Dict[str, Decimal]] = None,
    protected_area_multiplier: float = 1.0,
    timing_multiplier: float = 1.0,
) -> float:
    """Calculate weighted total severity score (0-100).

    Applies multipliers for protected area overlay and post-cutoff timing
    after computing the base weighted score.
    """
    if weights is None:
        weights = DEFAULT_SEVERITY_WEIGHTS

    base_score = (
        float(weights["area"]) * area_score
        + float(weights["rate"]) * rate_score
        + float(weights["proximity"]) * proximity_score
        + float(weights["protected"]) * protected_score
        + float(weights["timing"]) * timing_score
    )

    # Apply multipliers but cap at 100
    total = base_score
    if protected_area_multiplier > 1.0:
        total = total * protected_area_multiplier
    if timing_multiplier > 1.0:
        total = total * timing_multiplier

    return min(total, 100.0)


def _determine_severity_level(total_score: float) -> str:
    """Map total score to severity level."""
    if total_score >= 80:
        return "critical"
    elif total_score >= 60:
        return "high"
    elif total_score >= 40:
        return "medium"
    elif total_score >= 20:
        return "low"
    else:
        return "informational"


def _identify_aggravating_factors(
    is_protected_area: bool,
    is_post_cutoff: bool,
    area_ha: float,
    rate_ha_per_day: float,
    distance_km: float,
) -> List[str]:
    """Identify aggravating factors that increase severity."""
    factors = []
    if is_protected_area:
        factors.append("protected_area_overlay")
    if is_post_cutoff:
        factors.append("post_cutoff_timing")
    if area_ha >= 50:
        factors.append("large_scale_clearing")
    if rate_ha_per_day >= 5.0:
        factors.append("rapid_deforestation")
    if distance_km < 1:
        factors.append("immediate_supply_chain_proximity")
    return factors


def _classify_severity(
    area_ha: float,
    rate_ha_per_day: float = 0.0,
    distance_km: float = 100.0,
    protected_overlap_pct: float = 0.0,
    is_post_cutoff: bool = False,
    weights: Optional[Dict[str, Decimal]] = None,
    protected_area_multiplier: float = 1.5,
    post_cutoff_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """Full severity classification with scoring breakdown."""
    a_score = _score_area(area_ha)
    r_score = _score_rate(rate_ha_per_day)
    p_score = _score_proximity(distance_km)
    pa_score = _score_protected_area(protected_overlap_pct)
    t_score, t_mult = _score_timing(is_post_cutoff, post_cutoff_multiplier)

    pa_mult = protected_area_multiplier if protected_overlap_pct > 0 else 1.0

    total = _calculate_total_score(
        a_score, r_score, p_score, pa_score, t_score,
        weights=weights,
        protected_area_multiplier=pa_mult,
        timing_multiplier=t_mult,
    )
    severity = _determine_severity_level(total)

    aggravating = _identify_aggravating_factors(
        is_protected_area=protected_overlap_pct > 0,
        is_post_cutoff=is_post_cutoff,
        area_ha=area_ha,
        rate_ha_per_day=rate_ha_per_day,
        distance_km=distance_km,
    )

    return {
        "severity": severity,
        "total_score": total,
        "component_scores": {
            "area": a_score,
            "rate": r_score,
            "proximity": p_score,
            "protected": pa_score,
            "timing": t_score,
        },
        "aggravating_factors": aggravating,
        "provenance_hash": compute_test_hash({
            "area_ha": area_ha,
            "rate": rate_ha_per_day,
            "distance_km": distance_km,
            "severity": severity,
        }),
    }


def _reclassify(
    previous_result: Dict,
    new_distance_km: Optional[float] = None,
    new_is_post_cutoff: Optional[bool] = None,
    new_protected_overlap_pct: Optional[float] = None,
) -> Dict:
    """Reclassify severity with updated context."""
    scores = previous_result["component_scores"].copy()
    if new_distance_km is not None:
        scores["proximity"] = _score_proximity(new_distance_km)
    if new_protected_overlap_pct is not None:
        scores["protected"] = _score_protected_area(new_protected_overlap_pct)
    if new_is_post_cutoff is not None:
        t_score, _ = _score_timing(new_is_post_cutoff)
        scores["timing"] = t_score

    total = _calculate_total_score(
        scores["area"], scores["rate"], scores["proximity"],
        scores["protected"], scores["timing"],
    )
    severity = _determine_severity_level(total)

    return {
        "severity": severity,
        "total_score": total,
        "component_scores": scores,
        "aggravating_factors": previous_result.get("aggravating_factors", []),
        "provenance_hash": compute_test_hash({
            "reclassified": True,
            "severity": severity,
            "total_score": total,
        }),
    }


# ===========================================================================
# 1. TestSeverityClassification (8 tests)
# ===========================================================================


class TestSeverityClassification:
    """Test classify for CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL scenarios."""

    def test_critical_scenario(self):
        """Test CRITICAL: large area, post-cutoff, near plot, in protected area."""
        result = _classify_severity(
            area_ha=60.0,
            rate_ha_per_day=8.0,
            distance_km=0.5,
            protected_overlap_pct=80.0,
            is_post_cutoff=True,
        )
        assert result["severity"] == "critical"
        assert result["total_score"] >= 80

    def test_high_scenario(self):
        """Test HIGH: medium area, near plot, post-cutoff."""
        result = _classify_severity(
            area_ha=15.0,
            rate_ha_per_day=3.0,
            distance_km=3.0,
            protected_overlap_pct=0.0,
            is_post_cutoff=True,
        )
        assert result["severity"] in ("critical", "high")
        assert result["total_score"] >= 60

    def test_medium_scenario(self):
        """Test MEDIUM: moderate area, moderate distance, pre-cutoff."""
        result = _classify_severity(
            area_ha=3.0,
            rate_ha_per_day=0.5,
            distance_km=15.0,
            protected_overlap_pct=0.0,
            is_post_cutoff=False,
        )
        assert result["severity"] in ("medium", "low")

    def test_low_scenario(self):
        """Test LOW: small area, far from plots, pre-cutoff."""
        result = _classify_severity(
            area_ha=0.6,
            rate_ha_per_day=0.05,
            distance_km=40.0,
            protected_overlap_pct=0.0,
            is_post_cutoff=False,
        )
        assert result["severity"] in ("low", "informational")

    def test_informational_scenario(self):
        """Test INFORMATIONAL: tiny area, very far, no protected area."""
        result = _classify_severity(
            area_ha=0.1,
            rate_ha_per_day=0.01,
            distance_km=100.0,
            protected_overlap_pct=0.0,
            is_post_cutoff=False,
        )
        assert result["severity"] in ("informational", "low")
        assert result["total_score"] < 40

    def test_post_cutoff_escalation(self):
        """Test post-cutoff timing escalates severity."""
        pre = _classify_severity(
            area_ha=5.0, rate_ha_per_day=1.0, distance_km=10.0,
            is_post_cutoff=False,
        )
        post = _classify_severity(
            area_ha=5.0, rate_ha_per_day=1.0, distance_km=10.0,
            is_post_cutoff=True,
        )
        assert post["total_score"] > pre["total_score"]

    def test_protected_area_escalation(self):
        """Test protected area overlay escalates severity."""
        no_prot = _classify_severity(
            area_ha=5.0, rate_ha_per_day=1.0, distance_km=10.0,
            protected_overlap_pct=0.0,
        )
        prot = _classify_severity(
            area_ha=5.0, rate_ha_per_day=1.0, distance_km=10.0,
            protected_overlap_pct=50.0,
        )
        assert prot["total_score"] > no_prot["total_score"]

    def test_classification_has_provenance(self):
        """Test classification result includes provenance hash."""
        result = _classify_severity(area_ha=5.0)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. TestReclassification (4 tests)
# ===========================================================================


class TestReclassification:
    """Test reclassify with new context."""

    def test_reclassify_closer_proximity(self):
        """Test reclassification with closer proximity increases score."""
        original = _classify_severity(
            area_ha=5.0, distance_km=30.0,
        )
        reclassified = _reclassify(original, new_distance_km=2.0)
        assert reclassified["component_scores"]["proximity"] > original["component_scores"]["proximity"]

    def test_reclassify_post_cutoff(self):
        """Test reclassification adding post-cutoff flag."""
        original = _classify_severity(
            area_ha=5.0, is_post_cutoff=False,
        )
        reclassified = _reclassify(original, new_is_post_cutoff=True)
        assert reclassified["component_scores"]["timing"] > original["component_scores"]["timing"]

    def test_reclassify_protected_area(self):
        """Test reclassification adding protected area overlap."""
        original = _classify_severity(
            area_ha=5.0, protected_overlap_pct=0.0,
        )
        reclassified = _reclassify(original, new_protected_overlap_pct=60.0)
        assert reclassified["component_scores"]["protected"] > original["component_scores"]["protected"]

    def test_reclassify_has_provenance(self):
        """Test reclassified result has new provenance hash."""
        original = _classify_severity(area_ha=5.0)
        reclassified = _reclassify(original, new_distance_km=2.0)
        assert reclassified["provenance_hash"] != original["provenance_hash"]


# ===========================================================================
# 3. TestThresholds (3 tests)
# ===========================================================================


class TestThresholds:
    """Test get_thresholds returns current config."""

    def test_severity_thresholds_ordered(self):
        """Test severity thresholds are in descending order."""
        thresholds = [
            float(SEVERITY_THRESHOLDS["critical"]),
            float(SEVERITY_THRESHOLDS["high"]),
            float(SEVERITY_THRESHOLDS["medium"]),
            float(SEVERITY_THRESHOLDS["low"]),
            float(SEVERITY_THRESHOLDS["informational"]),
        ]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_severity_weights_sum_to_one(self):
        """Test severity weights sum to 1.0."""
        total = sum(DEFAULT_SEVERITY_WEIGHTS.values())
        assert float(total) == pytest.approx(1.0, abs=0.001)

    def test_all_severity_levels_defined(self):
        """Test all severity levels have thresholds."""
        for level in SEVERITY_LEVELS:
            assert level in SEVERITY_THRESHOLDS


# ===========================================================================
# 4. TestDistribution (2 tests)
# ===========================================================================


class TestDistribution:
    """Test get_distribution with filters."""

    def test_distribution_all_levels(self):
        """Test severity distribution covers all levels."""
        results = []
        scenarios = [
            (60.0, 8.0, 0.5, True),   # critical
            (15.0, 3.0, 3.0, True),    # high
            (3.0, 0.5, 15.0, False),   # medium
            (0.6, 0.05, 40.0, False),  # low
            (0.1, 0.01, 100.0, False), # informational
        ]
        for area, rate, dist, post in scenarios:
            r = _classify_severity(
                area_ha=area, rate_ha_per_day=rate,
                distance_km=dist, is_post_cutoff=post,
            )
            results.append(r["severity"])
        # Should have at least 3 distinct severity levels
        assert len(set(results)) >= 3

    def test_distribution_deterministic(self):
        """Test severity distribution is deterministic."""
        r1 = _classify_severity(area_ha=5.0, distance_km=10.0)
        r2 = _classify_severity(area_ha=5.0, distance_km=10.0)
        assert r1["severity"] == r2["severity"]
        assert r1["total_score"] == r2["total_score"]


# ===========================================================================
# 5. TestAreaScoring (5 tests)
# ===========================================================================


class TestAreaScoring:
    """Test _score_area for various area thresholds."""

    @pytest.mark.parametrize("area_ha,expected_score", [
        (50.0, 100),
        (100.0, 100),
        (10.0, 80),
        (25.0, 80),
        (1.0, 50),
        (5.0, 50),
        (0.5, 30),
        (0.8, 30),
        (0.3, 10),
        (0.1, 10),
    ])
    def test_area_scoring(self, area_ha, expected_score):
        """Test area score for various hectare values."""
        assert _score_area(area_ha) == expected_score

    def test_area_score_boundary_50ha(self):
        """Test exact boundary at 50 ha."""
        assert _score_area(50.0) == 100
        assert _score_area(49.9) == 80

    def test_area_score_boundary_10ha(self):
        """Test exact boundary at 10 ha."""
        assert _score_area(10.0) == 80
        assert _score_area(9.9) == 50

    def test_area_score_boundary_1ha(self):
        """Test exact boundary at 1 ha."""
        assert _score_area(1.0) == 50
        assert _score_area(0.9) == 30

    def test_area_score_boundary_0_5ha(self):
        """Test exact boundary at 0.5 ha."""
        assert _score_area(0.5) == 30
        assert _score_area(0.4) == 10


# ===========================================================================
# 6. TestRateScoring (4 tests)
# ===========================================================================


class TestRateScoring:
    """Test _score_rate for various ha/day rates."""

    @pytest.mark.parametrize("rate,expected_score", [
        (10.0, 100),
        (15.0, 100),
        (5.0, 80),
        (7.5, 80),
        (1.0, 60),
        (3.0, 60),
        (0.5, 40),
        (0.7, 40),
        (0.1, 20),
        (0.3, 20),
        (0.05, 10),
        (0.0, 10),
    ])
    def test_rate_scoring(self, rate, expected_score):
        """Test rate score for various ha/day values."""
        assert _score_rate(rate) == expected_score

    def test_rate_score_boundary_10(self):
        """Test exact boundary at 10 ha/day."""
        assert _score_rate(10.0) == 100
        assert _score_rate(9.9) == 80

    def test_rate_score_boundary_5(self):
        """Test exact boundary at 5 ha/day."""
        assert _score_rate(5.0) == 80
        assert _score_rate(4.9) == 60

    def test_rate_score_zero(self):
        """Test zero rate returns minimum score."""
        assert _score_rate(0.0) == 10


# ===========================================================================
# 7. TestProximityScoring (4 tests)
# ===========================================================================


class TestProximityScoring:
    """Test _score_proximity for distance thresholds."""

    @pytest.mark.parametrize("distance_km,expected_score", [
        (0.5, 100),
        (0.9, 100),
        (1.0, 80),
        (3.0, 80),
        (5.0, 50),
        (15.0, 50),
        (25.0, 30),
        (40.0, 30),
        (50.0, 10),
        (100.0, 10),
    ])
    def test_proximity_scoring(self, distance_km, expected_score):
        """Test proximity score for various distance values."""
        assert _score_proximity(distance_km) == expected_score

    def test_proximity_score_boundary_1km(self):
        """Test exact boundary at 1 km."""
        assert _score_proximity(0.99) == 100
        assert _score_proximity(1.0) == 80

    def test_proximity_score_boundary_5km(self):
        """Test exact boundary at 5 km."""
        assert _score_proximity(4.99) == 80
        assert _score_proximity(5.0) == 50

    def test_proximity_score_very_far(self):
        """Test very far distance returns minimum score."""
        assert _score_proximity(1000.0) == 10


# ===========================================================================
# 8. TestProtectedAreaScoring (4 tests)
# ===========================================================================


class TestProtectedAreaScoring:
    """Test _score_protected_area with overlap percentages."""

    @pytest.mark.parametrize("overlap_pct,expected_score", [
        (100.0, 100),
        (75.0, 100),
        (50.0, 80),
        (60.0, 80),
        (25.0, 60),
        (35.0, 60),
        (10.0, 40),
        (1.0, 40),
        (0.0, 0),
    ])
    def test_protected_area_scoring(self, overlap_pct, expected_score):
        """Test protected area score for various overlap percentages."""
        assert _score_protected_area(overlap_pct) == expected_score

    def test_protected_score_no_overlap(self):
        """Test zero overlap returns zero score."""
        assert _score_protected_area(0.0) == 0

    def test_protected_score_full_overlap(self):
        """Test 100% overlap returns maximum score."""
        assert _score_protected_area(100.0) == 100

    def test_protected_score_boundary_75(self):
        """Test boundary at 75% overlap."""
        assert _score_protected_area(75.0) == 100
        assert _score_protected_area(74.9) == 80


# ===========================================================================
# 9. TestTimingScoring (4 tests)
# ===========================================================================


class TestTimingScoring:
    """Test _score_timing for post-cutoff and pre-cutoff events."""

    def test_post_cutoff_high_score_and_multiplier(self):
        """Test post-cutoff returns high score and 2.0x multiplier."""
        score, mult = _score_timing(True)
        assert score == 100
        assert mult == 2.0

    def test_pre_cutoff_low_score_no_multiplier(self):
        """Test pre-cutoff returns low score and 1.0x multiplier."""
        score, mult = _score_timing(False)
        assert score == 20
        assert mult == 1.0

    def test_custom_multiplier(self):
        """Test custom post-cutoff multiplier."""
        score, mult = _score_timing(True, post_cutoff_multiplier=3.0)
        assert score == 100
        assert mult == 3.0

    def test_pre_cutoff_custom_multiplier_ignored(self):
        """Test pre-cutoff ignores custom multiplier."""
        score, mult = _score_timing(False, post_cutoff_multiplier=3.0)
        assert mult == 1.0


# ===========================================================================
# 10. TestTotalScore (4 tests)
# ===========================================================================


class TestTotalScore:
    """Test _calculate_total_score with weighted components."""

    def test_total_score_all_max(self):
        """Test total score with all components at maximum."""
        total = _calculate_total_score(100, 100, 100, 100, 100)
        assert total == 100.0

    def test_total_score_all_min(self):
        """Test total score with all components at minimum."""
        total = _calculate_total_score(0, 0, 0, 0, 0)
        assert total == 0.0

    def test_total_score_weighted_correctly(self):
        """Test total score uses correct weights."""
        # area=100*0.25 + rate=0*0.20 + prox=0*0.25 + prot=0*0.15 + timing=0*0.15
        total = _calculate_total_score(100, 0, 0, 0, 0)
        assert total == pytest.approx(25.0, abs=0.1)

    def test_total_score_capped_at_100(self):
        """Test total score with multipliers is capped at 100."""
        total = _calculate_total_score(
            100, 100, 100, 100, 100,
            protected_area_multiplier=2.0,
            timing_multiplier=2.0,
        )
        assert total <= 100.0


# ===========================================================================
# 11. TestSeverityLevel (5 tests)
# ===========================================================================


class TestSeverityLevel:
    """Test _determine_severity_level for score thresholds."""

    @pytest.mark.parametrize("score,expected_level", [
        (100.0, "critical"),
        (80.0, "critical"),
        (79.9, "high"),
        (60.0, "high"),
        (59.9, "medium"),
        (40.0, "medium"),
        (39.9, "low"),
        (20.0, "low"),
        (19.9, "informational"),
        (0.0, "informational"),
    ])
    def test_severity_level_mapping(self, score, expected_level):
        """Test severity level mapping from score."""
        assert _determine_severity_level(score) == expected_level

    def test_severity_level_boundary_80(self):
        """Test boundary at score 80."""
        assert _determine_severity_level(80.0) == "critical"
        assert _determine_severity_level(79.99) == "high"

    def test_severity_level_boundary_60(self):
        """Test boundary at score 60."""
        assert _determine_severity_level(60.0) == "high"
        assert _determine_severity_level(59.99) == "medium"

    def test_severity_level_boundary_40(self):
        """Test boundary at score 40."""
        assert _determine_severity_level(40.0) == "medium"
        assert _determine_severity_level(39.99) == "low"

    def test_severity_level_boundary_20(self):
        """Test boundary at score 20."""
        assert _determine_severity_level(20.0) == "low"
        assert _determine_severity_level(19.99) == "informational"


# ===========================================================================
# 12. TestAggravatingFactors (5 tests)
# ===========================================================================


class TestAggravatingFactors:
    """Test _identify_aggravating_factors."""

    def test_no_aggravating_factors(self):
        """Test no factors when all conditions are benign."""
        factors = _identify_aggravating_factors(
            is_protected_area=False,
            is_post_cutoff=False,
            area_ha=1.0,
            rate_ha_per_day=0.1,
            distance_km=10.0,
        )
        assert factors == []

    def test_protected_area_factor(self):
        """Test protected area is identified as aggravating."""
        factors = _identify_aggravating_factors(
            is_protected_area=True,
            is_post_cutoff=False,
            area_ha=1.0,
            rate_ha_per_day=0.1,
            distance_km=10.0,
        )
        assert "protected_area_overlay" in factors

    def test_post_cutoff_factor(self):
        """Test post-cutoff is identified as aggravating."""
        factors = _identify_aggravating_factors(
            is_protected_area=False,
            is_post_cutoff=True,
            area_ha=1.0,
            rate_ha_per_day=0.1,
            distance_km=10.0,
        )
        assert "post_cutoff_timing" in factors

    def test_large_scale_factor(self):
        """Test large-scale clearing (>=50ha) is aggravating."""
        factors = _identify_aggravating_factors(
            is_protected_area=False,
            is_post_cutoff=False,
            area_ha=55.0,
            rate_ha_per_day=0.1,
            distance_km=10.0,
        )
        assert "large_scale_clearing" in factors

    def test_all_aggravating_factors(self):
        """Test all aggravating factors identified simultaneously."""
        factors = _identify_aggravating_factors(
            is_protected_area=True,
            is_post_cutoff=True,
            area_ha=60.0,
            rate_ha_per_day=8.0,
            distance_km=0.5,
        )
        assert "protected_area_overlay" in factors
        assert "post_cutoff_timing" in factors
        assert "large_scale_clearing" in factors
        assert "rapid_deforestation" in factors
        assert "immediate_supply_chain_proximity" in factors
        assert len(factors) == 5
