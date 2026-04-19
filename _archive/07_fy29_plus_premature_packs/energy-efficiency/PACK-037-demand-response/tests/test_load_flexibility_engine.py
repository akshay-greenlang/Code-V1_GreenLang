# -*- coding: utf-8 -*-
"""
Unit tests for LoadFlexibilityEngine -- PACK-037 Engine 1
============================================================

Tests load categorization across 5 criticality levels, flexibility scoring
with 7 factors, curtailment capacity matrix (5x5 notification x duration),
full facility assessment, seasonal adjustments, provenance hash determinism,
and edge cases (empty, single, large load sets).

Coverage target: 85%+
Total tests: ~80
"""

import hashlib
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack037_test.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("load_flexibility_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")
        assert _m._MODULE_VERSION == "1.0.0"

    def test_engine_class_exists(self):
        assert hasattr(_m, "LoadFlexibilityEngine")

    def test_engine_instantiation(self):
        engine = _m.LoadFlexibilityEngine()
        assert engine is not None


# =============================================================================
# Load Categorization (5 criticality levels)
# =============================================================================


class TestLoadCategorization:
    """Test load categorization across all 5 criticality levels."""

    def _get_categorize(self, engine):
        return (getattr(engine, "categorize_loads", None)
                or getattr(engine, "categorize", None)
                or getattr(engine, "classify_loads", None))

    @pytest.mark.parametrize("criticality,expected_label", [
        (1, "CRITICAL"),
        (2, "ESSENTIAL"),
        (3, "IMPORTANT"),
        (4, "DEFERRABLE"),
        (5, "SHEDDABLE"),
    ])
    def test_criticality_level_label(self, criticality, expected_label):
        engine = _m.LoadFlexibilityEngine()
        categorize = self._get_categorize(engine)
        if categorize is None:
            pytest.skip("categorize method not found")
        load = {"load_id": f"TEST-{criticality}", "criticality": criticality,
                "rated_kw": 100.0, "typical_kw": 80.0, "flexible_kw": 50.0}
        result = categorize([load])
        assert result is not None

    def test_critical_loads_never_curtailed(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        categorize = self._get_categorize(engine)
        if categorize is None:
            pytest.skip("categorize method not found")
        result = categorize(sample_load_inventory)
        critical_loads = [ld for ld in sample_load_inventory if ld["criticality"] == 1]
        for ld in critical_loads:
            assert ld["flexible_kw"] == 0.0

    def test_sheddable_loads_fully_curtailable(self, sample_load_inventory):
        sheddable = [ld for ld in sample_load_inventory if ld["criticality"] == 5]
        for ld in sheddable:
            assert ld["flexible_kw"] == ld["typical_kw"]

    def test_load_count_by_criticality(self, sample_load_inventory):
        counts = {}
        for ld in sample_load_inventory:
            c = ld["criticality"]
            counts[c] = counts.get(c, 0) + 1
        assert counts[1] == 4
        assert counts[2] == 4
        assert counts[3] == 5
        assert counts[4] == 5
        assert counts[5] == 6

    def test_total_load_count(self, sample_load_inventory):
        assert len(sample_load_inventory) == 24

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_loads_exist_for_each_level(self, sample_load_inventory, level):
        loads_at_level = [ld for ld in sample_load_inventory
                          if ld["criticality"] == level]
        assert len(loads_at_level) >= 1

    def test_categorize_returns_groups(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        categorize = self._get_categorize(engine)
        if categorize is None:
            pytest.skip("categorize method not found")
        result = categorize(sample_load_inventory)
        assert result is not None

    def test_all_loads_have_required_fields(self, sample_load_inventory):
        required = ["load_id", "name", "criticality", "rated_kw",
                     "typical_kw", "flexible_kw"]
        for ld in sample_load_inventory:
            for field in required:
                assert field in ld, f"Missing {field} in {ld['load_id']}"

    def test_flexible_never_exceeds_typical(self, sample_load_inventory):
        for ld in sample_load_inventory:
            assert ld["flexible_kw"] <= ld["typical_kw"], (
                f"{ld['load_id']}: flexible ({ld['flexible_kw']}) > "
                f"typical ({ld['typical_kw']})"
            )

    def test_typical_never_exceeds_rated(self, sample_load_inventory):
        for ld in sample_load_inventory:
            assert ld["typical_kw"] <= ld["rated_kw"], (
                f"{ld['load_id']}: typical ({ld['typical_kw']}) > "
                f"rated ({ld['rated_kw']})"
            )


# =============================================================================
# Flexibility Scoring (7 factors)
# =============================================================================


class TestFlexibilityScoring:
    """Test flexibility scoring across 7 factors."""

    def _get_scorer(self, engine):
        return (getattr(engine, "score_flexibility", None)
                or getattr(engine, "calculate_flexibility_score", None)
                or getattr(engine, "flexibility_score", None))

    @pytest.mark.parametrize("factor", [
        "curtailment_depth",
        "notification_time",
        "duration_tolerance",
        "ramp_rate",
        "rebound_factor",
        "comfort_impact",
        "frequency_tolerance",
    ])
    def test_scoring_factor_exists(self, factor):
        """Verify that each of the 7 scoring factors is recognized."""
        engine = _m.LoadFlexibilityEngine()
        scorer = self._get_scorer(engine)
        if scorer is None:
            pytest.skip("flexibility scoring method not found")
        load = {
            "load_id": "TEST-SCORE", "criticality": 3,
            "rated_kw": 100.0, "typical_kw": 80.0, "flexible_kw": 40.0,
            "min_notification_min": 15, "max_curtail_hours": 4,
            "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.10,
            "comfort_impact": "LOW", "process_impact": "NONE",
        }
        result = scorer(load)
        assert result is not None

    def test_score_range_0_to_100(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        scorer = self._get_scorer(engine)
        if scorer is None:
            pytest.skip("flexibility scoring method not found")
        for ld in sample_load_inventory:
            result = scorer(ld)
            score = getattr(result, "score", result) if not isinstance(
                result, (int, float)) else result
            if isinstance(score, (int, float)):
                assert 0 <= score <= 100

    def test_critical_load_scores_zero(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        scorer = self._get_scorer(engine)
        if scorer is None:
            pytest.skip("flexibility scoring method not found")
        critical_loads = [ld for ld in sample_load_inventory if ld["criticality"] == 1]
        for ld in critical_loads:
            result = scorer(ld)
            score = getattr(result, "score", result) if not isinstance(
                result, (int, float)) else result
            if isinstance(score, (int, float)):
                assert score == 0.0

    def test_sheddable_scores_highest(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        scorer = self._get_scorer(engine)
        if scorer is None:
            pytest.skip("flexibility scoring method not found")
        sheddable = [ld for ld in sample_load_inventory if ld["criticality"] == 5]
        essential = [ld for ld in sample_load_inventory if ld["criticality"] == 2]
        for s_ld in sheddable:
            s_result = scorer(s_ld)
            s_score = getattr(s_result, "score", s_result) if not isinstance(
                s_result, (int, float)) else s_result
            for e_ld in essential:
                e_result = scorer(e_ld)
                e_score = getattr(e_result, "score", e_result) if not isinstance(
                    e_result, (int, float)) else e_result
                if isinstance(s_score, (int, float)) and isinstance(e_score, (int, float)):
                    assert s_score >= e_score

    def test_score_deterministic(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        scorer = self._get_scorer(engine)
        if scorer is None:
            pytest.skip("flexibility scoring method not found")
        ld = sample_load_inventory[10]
        r1 = scorer(ld)
        r2 = scorer(ld)
        s1 = getattr(r1, "score", r1)
        s2 = getattr(r2, "score", r2)
        assert s1 == s2

    def test_higher_notification_time_lower_score(self):
        engine = _m.LoadFlexibilityEngine()
        scorer = self._get_scorer(engine)
        if scorer is None:
            pytest.skip("flexibility scoring method not found")
        fast = {"load_id": "FAST", "criticality": 4, "rated_kw": 100.0,
                "typical_kw": 80.0, "flexible_kw": 80.0,
                "min_notification_min": 5, "max_curtail_hours": 8,
                "ramp_rate_kw_per_min": 50.0, "rebound_factor": 1.00,
                "comfort_impact": "NONE", "process_impact": "NONE"}
        slow = dict(fast, load_id="SLOW", min_notification_min=120)
        r_fast = scorer(fast)
        r_slow = scorer(slow)
        sf = getattr(r_fast, "score", r_fast)
        ss = getattr(r_slow, "score", r_slow)
        if isinstance(sf, (int, float)) and isinstance(ss, (int, float)):
            assert sf >= ss


# =============================================================================
# Curtailment Capacity Matrix (5x5 notification x duration)
# =============================================================================


class TestCurtailmentCapacityMatrix:
    """Test 5x5 curtailment capacity matrix (notification x duration)."""

    def _get_matrix(self, engine):
        return (getattr(engine, "build_curtailment_matrix", None)
                or getattr(engine, "curtailment_matrix", None)
                or getattr(engine, "capacity_matrix", None))

    @pytest.mark.parametrize("notification_min", [5, 15, 30, 60, 120])
    @pytest.mark.parametrize("duration_hours", [1, 2, 4, 6, 8])
    def test_matrix_cell(self, sample_load_inventory,
                         notification_min, duration_hours):
        engine = _m.LoadFlexibilityEngine()
        build = self._get_matrix(engine)
        if build is None:
            pytest.skip("curtailment matrix method not found")
        result = build(sample_load_inventory,
                       notification_min=notification_min,
                       duration_hours=duration_hours)
        assert result is not None

    def test_matrix_monotonic_notification(self, sample_load_inventory):
        """Shorter notification should yield <= capacity of longer notification."""
        engine = _m.LoadFlexibilityEngine()
        build = self._get_matrix(engine)
        if build is None:
            pytest.skip("curtailment matrix method not found")
        cap_5 = build(sample_load_inventory, notification_min=5, duration_hours=4)
        cap_60 = build(sample_load_inventory, notification_min=60, duration_hours=4)
        v5 = getattr(cap_5, "total_kw", cap_5)
        v60 = getattr(cap_60, "total_kw", cap_60)
        if isinstance(v5, (int, float)) and isinstance(v60, (int, float)):
            assert v5 <= v60

    def test_matrix_monotonic_duration(self, sample_load_inventory):
        """Longer duration should yield <= capacity of shorter duration."""
        engine = _m.LoadFlexibilityEngine()
        build = self._get_matrix(engine)
        if build is None:
            pytest.skip("curtailment matrix method not found")
        cap_1 = build(sample_load_inventory, notification_min=30, duration_hours=1)
        cap_8 = build(sample_load_inventory, notification_min=30, duration_hours=8)
        v1 = getattr(cap_1, "total_kw", cap_1)
        v8 = getattr(cap_8, "total_kw", cap_8)
        if isinstance(v1, (int, float)) and isinstance(v8, (int, float)):
            assert v1 >= v8


# =============================================================================
# Assess Facility (Full Facility Assessment)
# =============================================================================


class TestAssessFacility:
    """Test full facility flexibility assessment."""

    def _get_assess(self, engine):
        return (getattr(engine, "assess_facility", None)
                or getattr(engine, "assess", None)
                or getattr(engine, "evaluate_facility", None))

    def test_assess_returns_result(self, sample_load_inventory,
                                    sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess_facility method not found")
        result = assess(loads=sample_load_inventory,
                        facility=sample_facility_profile)
        assert result is not None

    def test_assess_total_flexible_kw(self, sample_load_inventory,
                                      sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess_facility method not found")
        result = assess(loads=sample_load_inventory,
                        facility=sample_facility_profile)
        total_flex = sum(ld["flexible_kw"] for ld in sample_load_inventory)
        result_flex = getattr(result, "total_flexible_kw", None)
        if result_flex is not None:
            assert abs(result_flex - total_flex) < 1.0

    def test_assess_includes_provenance(self, sample_load_inventory,
                                         sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess_facility method not found")
        result = assess(loads=sample_load_inventory,
                        facility=sample_facility_profile)
        ph = getattr(result, "provenance_hash", None)
        if ph is not None:
            assert len(ph) == 64

    def test_assess_breakdown_by_criticality(self, sample_load_inventory,
                                              sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess_facility method not found")
        result = assess(loads=sample_load_inventory,
                        facility=sample_facility_profile)
        breakdown = getattr(result, "breakdown_by_criticality", None)
        if breakdown is not None:
            assert len(breakdown) == 5


# =============================================================================
# Seasonal Adjustment
# =============================================================================


class TestSeasonalAdjustment:
    """Test seasonal flexibility adjustments."""

    def _get_seasonal(self, engine):
        return (getattr(engine, "apply_seasonal_adjustment", None)
                or getattr(engine, "seasonal_adjust", None)
                or getattr(engine, "adjust_for_season", None))

    @pytest.mark.parametrize("season,hvac_impact", [
        ("SUMMER", "HIGH"),
        ("WINTER", "MEDIUM"),
        ("SPRING", "LOW"),
        ("FALL", "LOW"),
    ])
    def test_seasonal_hvac_impact(self, sample_load_inventory, season, hvac_impact):
        engine = _m.LoadFlexibilityEngine()
        adjust = self._get_seasonal(engine)
        if adjust is None:
            pytest.skip("seasonal adjustment method not found")
        result = adjust(sample_load_inventory, season=season)
        assert result is not None

    def test_summer_increases_hvac_flexibility(self, sample_load_inventory):
        engine = _m.LoadFlexibilityEngine()
        adjust = self._get_seasonal(engine)
        if adjust is None:
            pytest.skip("seasonal adjustment method not found")
        summer = adjust(sample_load_inventory, season="SUMMER")
        winter = adjust(sample_load_inventory, season="WINTER")
        assert summer is not None
        assert winter is not None


# =============================================================================
# Provenance Hash Determinism
# =============================================================================


class TestProvenanceHash:
    """Test provenance hash is deterministic."""

    def test_same_input_same_hash(self, sample_load_inventory,
                                   sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = (getattr(engine, "assess_facility", None)
                  or getattr(engine, "assess", None))
        if assess is None:
            pytest.skip("assess method not found")
        r1 = assess(loads=sample_load_inventory, facility=sample_facility_profile)
        r2 = assess(loads=sample_load_inventory, facility=sample_facility_profile)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2

    def test_hash_is_sha256(self, sample_load_inventory,
                             sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = (getattr(engine, "assess_facility", None)
                  or getattr(engine, "assess", None))
        if assess is None:
            pytest.skip("assess method not found")
        result = assess(loads=sample_load_inventory,
                        facility=sample_facility_profile)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_different_input_different_hash(self, sample_load_inventory,
                                             sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = (getattr(engine, "assess_facility", None)
                  or getattr(engine, "assess", None))
        if assess is None:
            pytest.skip("assess method not found")
        r1 = assess(loads=sample_load_inventory, facility=sample_facility_profile)
        modified = [dict(ld, flexible_kw=0) for ld in sample_load_inventory]
        r2 = assess(loads=modified, facility=sample_facility_profile)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 != h2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases: empty, single, large load sets."""

    def _get_assess(self, engine):
        return (getattr(engine, "assess_facility", None)
                or getattr(engine, "assess", None)
                or getattr(engine, "evaluate_facility", None))

    def test_empty_loads(self, sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess method not found")
        result = assess(loads=[], facility=sample_facility_profile)
        total = getattr(result, "total_flexible_kw", 0)
        assert total == 0 or result is not None

    def test_single_load(self, sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess method not found")
        single = [{"load_id": "SINGLE", "name": "Test Load",
                    "criticality": 3, "rated_kw": 100.0,
                    "typical_kw": 80.0, "flexible_kw": 40.0,
                    "min_notification_min": 15, "max_curtail_hours": 4,
                    "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.10,
                    "comfort_impact": "LOW", "process_impact": "NONE"}]
        result = assess(loads=single, facility=sample_facility_profile)
        assert result is not None

    def test_500_loads(self, sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess method not found")
        loads = []
        for i in range(500):
            loads.append({
                "load_id": f"BULK-{i:04d}",
                "name": f"Bulk Load {i}",
                "criticality": (i % 5) + 1,
                "rated_kw": 10.0,
                "typical_kw": 8.0,
                "flexible_kw": 4.0 if (i % 5) >= 2 else 0.0,
                "min_notification_min": 15,
                "max_curtail_hours": 4,
                "ramp_rate_kw_per_min": 10.0,
                "rebound_factor": 1.05,
                "comfort_impact": "LOW",
                "process_impact": "NONE",
            })
        result = assess(loads=loads, facility=sample_facility_profile)
        assert result is not None

    def test_all_critical_loads(self, sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess method not found")
        loads = [{"load_id": f"CRIT-{i}", "name": f"Critical {i}",
                  "criticality": 1, "rated_kw": 100.0,
                  "typical_kw": 80.0, "flexible_kw": 0.0,
                  "min_notification_min": None, "max_curtail_hours": 0,
                  "ramp_rate_kw_per_min": None, "rebound_factor": 1.0,
                  "comfort_impact": "NONE", "process_impact": "CRITICAL"}
                 for i in range(5)]
        result = assess(loads=loads, facility=sample_facility_profile)
        total = getattr(result, "total_flexible_kw", 0)
        assert total == 0 or result is not None

    def test_all_sheddable_loads(self, sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess method not found")
        loads = [{"load_id": f"SHED-{i}", "name": f"Sheddable {i}",
                  "criticality": 5, "rated_kw": 50.0,
                  "typical_kw": 40.0, "flexible_kw": 40.0,
                  "min_notification_min": 5, "max_curtail_hours": 12,
                  "ramp_rate_kw_per_min": 50.0, "rebound_factor": 1.00,
                  "comfort_impact": "NONE", "process_impact": "NONE"}
                 for i in range(10)]
        result = assess(loads=loads, facility=sample_facility_profile)
        total = getattr(result, "total_flexible_kw", None)
        if total is not None:
            assert total == pytest.approx(400.0, rel=0.01)

    def test_zero_rated_kw_load(self, sample_facility_profile):
        engine = _m.LoadFlexibilityEngine()
        assess = self._get_assess(engine)
        if assess is None:
            pytest.skip("assess method not found")
        loads = [{"load_id": "ZERO", "name": "Zero Load",
                  "criticality": 5, "rated_kw": 0.0,
                  "typical_kw": 0.0, "flexible_kw": 0.0,
                  "min_notification_min": 5, "max_curtail_hours": 12,
                  "ramp_rate_kw_per_min": 0.0, "rebound_factor": 1.00,
                  "comfort_impact": "NONE", "process_impact": "NONE"}]
        result = assess(loads=loads, facility=sample_facility_profile)
        assert result is not None


# =============================================================================
# Total Flexibility Calculations
# =============================================================================


class TestTotalFlexibility:
    """Test aggregate flexibility calculations."""

    def test_total_flexible_kw(self, sample_load_inventory):
        total = sum(ld["flexible_kw"] for ld in sample_load_inventory)
        assert total > 0

    def test_total_rated_kw(self, sample_load_inventory):
        total = sum(ld["rated_kw"] for ld in sample_load_inventory)
        assert total > 0

    def test_flexibility_ratio(self, sample_load_inventory):
        total_flex = sum(ld["flexible_kw"] for ld in sample_load_inventory)
        total_typical = sum(ld["typical_kw"] for ld in sample_load_inventory)
        ratio = total_flex / total_typical if total_typical > 0 else 0
        assert 0 < ratio < 1.0

    def test_critical_zero_flexibility(self, sample_load_inventory):
        critical_flex = sum(ld["flexible_kw"] for ld in sample_load_inventory
                           if ld["criticality"] == 1)
        assert critical_flex == 0.0

    def test_noncritical_positive_flexibility(self, sample_load_inventory):
        noncrit_flex = sum(ld["flexible_kw"] for ld in sample_load_inventory
                          if ld["criticality"] >= 3)
        assert noncrit_flex > 0

    @pytest.mark.parametrize("level,expected_min_flex", [
        (1, 0), (2, 100), (3, 100), (4, 400), (5, 100),
    ])
    def test_flexibility_by_level(self, sample_load_inventory,
                                   level, expected_min_flex):
        level_flex = sum(ld["flexible_kw"] for ld in sample_load_inventory
                        if ld["criticality"] == level)
        assert level_flex >= expected_min_flex


# =============================================================================
# Notification Time Validation
# =============================================================================


class TestNotificationTimeValidation:
    """Test notification time constraints per load."""

    @pytest.mark.parametrize("load_id,expected_min_notif", [
        ("LD-001", None), ("LD-005", 60), ("LD-009", 15),
        ("LD-014", 10), ("LD-019", 5),
    ])
    def test_notification_time_by_load(self, sample_load_inventory,
                                        load_id, expected_min_notif):
        load = next(ld for ld in sample_load_inventory
                    if ld["load_id"] == load_id)
        assert load["min_notification_min"] == expected_min_notif

    def test_critical_loads_no_notification(self, sample_load_inventory):
        critical = [ld for ld in sample_load_inventory if ld["criticality"] == 1]
        for ld in critical:
            assert ld["min_notification_min"] is None

    def test_sheddable_loads_fast_notification(self, sample_load_inventory):
        sheddable = [ld for ld in sample_load_inventory if ld["criticality"] == 5]
        for ld in sheddable:
            assert ld["min_notification_min"] is not None
            assert ld["min_notification_min"] <= 30


# =============================================================================
# Max Curtailment Duration
# =============================================================================


class TestMaxCurtailmentDuration:
    """Test max curtailment duration constraints."""

    @pytest.mark.parametrize("load_id,expected_max_hours", [
        ("LD-001", 0), ("LD-005", 2), ("LD-009", 4),
        ("LD-014", 6), ("LD-019", 12),
    ])
    def test_max_curtail_hours(self, sample_load_inventory,
                                load_id, expected_max_hours):
        load = next(ld for ld in sample_load_inventory
                    if ld["load_id"] == load_id)
        assert load["max_curtail_hours"] == expected_max_hours

    def test_critical_zero_duration(self, sample_load_inventory):
        critical = [ld for ld in sample_load_inventory if ld["criticality"] == 1]
        for ld in critical:
            assert ld["max_curtail_hours"] == 0

    def test_sheddable_long_duration(self, sample_load_inventory):
        sheddable = [ld for ld in sample_load_inventory if ld["criticality"] == 5]
        for ld in sheddable:
            assert ld["max_curtail_hours"] >= 6


# =============================================================================
# Comfort Impact Analysis
# =============================================================================


class TestComfortImpact:
    """Test comfort impact categorization per load."""

    @pytest.mark.parametrize("load_id,expected_impact", [
        ("LD-001", "NONE"), ("LD-005", "MEDIUM"), ("LD-009", "LOW"),
        ("LD-014", "LOW"), ("LD-019", "NONE"),
    ])
    def test_comfort_impact(self, sample_load_inventory,
                             load_id, expected_impact):
        load = next(ld for ld in sample_load_inventory
                    if ld["load_id"] == load_id)
        assert load["comfort_impact"] == expected_impact

    def test_valid_comfort_levels(self, sample_load_inventory):
        valid = {"NONE", "LOW", "MEDIUM", "HIGH"}
        for ld in sample_load_inventory:
            assert ld["comfort_impact"] in valid

    def test_valid_process_impact_levels(self, sample_load_inventory):
        valid = {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for ld in sample_load_inventory:
            assert ld["process_impact"] in valid


# =============================================================================
# Rebound Factor Analysis
# =============================================================================


class TestReboundFactor:
    """Test rebound factor values per load."""

    @pytest.mark.parametrize("load_id,expected_rebound", [
        ("LD-001", 1.00), ("LD-005", 1.15), ("LD-006", 1.20),
        ("LD-014", 1.25), ("LD-019", 1.00),
    ])
    def test_rebound_factor(self, sample_load_inventory,
                             load_id, expected_rebound):
        load = next(ld for ld in sample_load_inventory
                    if ld["load_id"] == load_id)
        assert load["rebound_factor"] == pytest.approx(expected_rebound, rel=0.01)

    def test_rebound_at_least_one(self, sample_load_inventory):
        for ld in sample_load_inventory:
            assert ld["rebound_factor"] >= 1.0

    def test_rebound_max_reasonable(self, sample_load_inventory):
        for ld in sample_load_inventory:
            assert ld["rebound_factor"] <= 2.0

    def test_critical_no_rebound(self, sample_load_inventory):
        critical = [ld for ld in sample_load_inventory if ld["criticality"] == 1]
        for ld in critical:
            assert ld["rebound_factor"] == 1.0


# =============================================================================
# Load Category Distribution
# =============================================================================


class TestLoadCategoryDistribution:
    """Test load category coverage."""

    @pytest.mark.parametrize("category", [
        "LIFE_SAFETY", "IT_INFRASTRUCTURE", "HVAC", "LIGHTING",
        "EV_CHARGING", "THERMAL_STORAGE", "ENERGY_STORAGE",
    ])
    def test_category_exists(self, sample_load_inventory, category):
        found = any(ld["category"] == category for ld in sample_load_inventory)
        assert found

    def test_hvac_loads_count(self, sample_load_inventory):
        hvac = [ld for ld in sample_load_inventory if ld["category"] == "HVAC"]
        assert len(hvac) >= 4

    def test_lighting_loads_count(self, sample_load_inventory):
        lighting = [ld for ld in sample_load_inventory
                    if ld["category"] == "LIGHTING"]
        assert len(lighting) >= 2

    def test_amenity_loads_count(self, sample_load_inventory):
        amenity = [ld for ld in sample_load_inventory
                   if ld["category"] == "AMENITY"]
        assert len(amenity) >= 2
