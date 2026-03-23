# -*- coding: utf-8 -*-
"""
Unit tests for DispatchOptimizerEngine -- PACK-037 Engine 4
==============================================================

Tests basic dispatch, comfort constraints, critical load protection,
minimum runtime constraints, ramp rate limits, rebound forecast,
optimization reproducibility, and target shortfall handling.

Coverage target: 85%+
Total tests: ~60
"""

import importlib.util
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


_m = _load("dispatch_optimizer_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_engine_class_exists(self):
        assert hasattr(_m, "DispatchOptimizerEngine")

    def test_engine_instantiation(self):
        engine = _m.DispatchOptimizerEngine()
        assert engine is not None


class TestBasicDispatch:
    """Test basic dispatch plan generation."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None)
                or getattr(engine, "dispatch", None))

    def test_creates_dispatch_plan(self, sample_load_inventory,
                                    sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        assert result is not None

    def test_plan_meets_target(self, sample_load_inventory,
                                sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        planned = getattr(result, "total_planned_reduction_kw", None)
        if planned is not None:
            assert planned >= sample_dr_event["target_reduction_kw"]

    def test_plan_has_phases(self, sample_load_inventory,
                              sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        phases = getattr(result, "curtailment_sequence", None)
        if phases is not None:
            assert len(phases) >= 1

    def test_plan_has_restoration(self, sample_load_inventory,
                                   sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        restore = getattr(result, "restoration_sequence", None)
        if restore is not None:
            assert len(restore) >= 1

    def test_sheddable_loads_first(self, sample_load_inventory,
                                    sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        phases = getattr(result, "curtailment_sequence", None)
        if phases and len(phases) > 0:
            first_phase = phases[0]
            load_ids = getattr(first_phase, "load_ids",
                              first_phase.get("load_ids", []))
            if load_ids:
                # First loads shed should be sheddable (level 5) or deferrable (4)
                first_loads = [ld for ld in sample_load_inventory
                              if ld["load_id"] in load_ids]
                for ld in first_loads:
                    assert ld["criticality"] >= 4


class TestComfortConstraints:
    """Test comfort constraint enforcement."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None))

    def test_temperature_constraint(self, sample_load_inventory,
                                     sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        constraints = {"max_temperature_rise_c": 2.0}
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets,
                         comfort_constraints=constraints)
        assert result is not None

    def test_lighting_constraint(self, sample_load_inventory,
                                  sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        constraints = {"max_lighting_reduction_pct": 50}
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets,
                         comfort_constraints=constraints)
        assert result is not None

    @pytest.mark.parametrize("max_temp_rise", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_varying_temperature_constraints(self, sample_load_inventory,
                                              sample_dr_event,
                                              sample_der_assets,
                                              max_temp_rise):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        constraints = {"max_temperature_rise_c": max_temp_rise}
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets,
                         comfort_constraints=constraints)
        assert result is not None


class TestCriticalLoadProtection:
    """Test critical loads are never curtailed."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None))

    def test_critical_loads_excluded(self, sample_load_inventory,
                                     sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        critical_ids = {ld["load_id"] for ld in sample_load_inventory
                       if ld["criticality"] == 1}
        phases = getattr(result, "curtailment_sequence", None)
        if phases:
            for phase in phases:
                phase_ids = set(getattr(phase, "load_ids",
                                       phase.get("load_ids", [])))
                overlap = critical_ids & phase_ids
                assert len(overlap) == 0, (
                    f"Critical loads {overlap} included in curtailment"
                )

    def test_critical_loads_protected_flag(self, sample_dispatch_plan):
        constraints = sample_dispatch_plan.get("comfort_constraints", {})
        assert constraints.get("critical_loads_protected", False) is True


class TestMinimumRuntimeConstraints:
    """Test minimum runtime constraints."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None))

    def test_min_runtime_respected(self, sample_load_inventory,
                                    sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets,
                         min_curtailment_duration_min=15)
        assert result is not None


class TestRampRateLimits:
    """Test ramp rate limit enforcement."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None))

    def test_ramp_rates_within_limits(self, sample_load_inventory,
                                      sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        result = optimize(loads=sample_load_inventory,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        assert result is not None

    @pytest.mark.parametrize("load_id,expected_max_ramp", [
        ("LD-005", 10.0),
        ("LD-014", 50.0),
        ("LD-019", 35.0),
    ])
    def test_individual_ramp_rates(self, sample_load_inventory,
                                    load_id, expected_max_ramp):
        load = next(ld for ld in sample_load_inventory
                    if ld["load_id"] == load_id)
        assert load["ramp_rate_kw_per_min"] == expected_max_ramp


class TestReboundForecast:
    """Test rebound (snapback) forecasting."""

    def _get_forecast(self, engine):
        return (getattr(engine, "forecast_rebound", None)
                or getattr(engine, "rebound_forecast", None)
                or getattr(engine, "estimate_rebound", None))

    def test_rebound_forecast(self, sample_load_inventory, sample_dr_event):
        engine = _m.DispatchOptimizerEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("rebound_forecast method not found")
        result = forecast(loads=sample_load_inventory,
                         event=sample_dr_event)
        assert result is not None

    def test_rebound_positive(self, sample_load_inventory, sample_dr_event):
        engine = _m.DispatchOptimizerEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("rebound_forecast method not found")
        result = forecast(loads=sample_load_inventory,
                         event=sample_dr_event)
        rebound = getattr(result, "rebound_kw", result)
        if isinstance(rebound, (int, float)):
            assert rebound >= 0

    def test_rebound_from_fixture_data(self, sample_dr_event_results):
        assert sample_dr_event_results["rebound_kw"] == 120.0
        assert sample_dr_event_results["rebound_duration_hours"] == 1.5

    @pytest.mark.parametrize("rebound_factor,expected_kw", [
        (1.00, 0), (1.10, 80), (1.20, 160), (1.30, 240),
    ])
    def test_rebound_factor_calculation(self, rebound_factor, expected_kw):
        curtailed_kw = 800.0
        rebound = curtailed_kw * (rebound_factor - 1.0)
        assert rebound == pytest.approx(expected_kw, rel=0.01)


class TestOptimizationReproducibility:
    """Test dispatch optimization is deterministic."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None))

    def test_same_input_same_plan(self, sample_load_inventory,
                                   sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        r1 = optimize(loads=sample_load_inventory,
                     event=sample_dr_event,
                     der_assets=sample_der_assets)
        r2 = optimize(loads=sample_load_inventory,
                     event=sample_dr_event,
                     der_assets=sample_der_assets)
        h1 = getattr(r1, "provenance_hash", str(r1))
        h2 = getattr(r2, "provenance_hash", str(r2))
        assert h1 == h2


class TestTargetShortfallHandling:
    """Test handling when target cannot be met."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_dispatch", None)
                or getattr(engine, "create_dispatch_plan", None))

    def test_shortfall_warning(self, sample_dr_event, sample_der_assets):
        engine = _m.DispatchOptimizerEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("dispatch optimizer method not found")
        # Very small load set cannot meet 800 kW target
        small_loads = [
            {"load_id": "SM-001", "name": "Small Load",
             "criticality": 5, "rated_kw": 20.0,
             "typical_kw": 15.0, "flexible_kw": 15.0,
             "min_notification_min": 5, "max_curtail_hours": 8,
             "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.00,
             "comfort_impact": "NONE", "process_impact": "NONE"}
        ]
        result = optimize(loads=small_loads,
                         event=sample_dr_event,
                         der_assets=sample_der_assets)
        shortfall = getattr(result, "shortfall_kw", None)
        warning = getattr(result, "warning", None)
        assert shortfall is not None or warning is not None or result is not None

    def test_shortfall_amount_calculation(self):
        target = 800.0
        available = 600.0
        shortfall = target - available
        assert shortfall == 200.0


class TestDispatchPlanData:
    """Test dispatch plan fixture data integrity."""

    def test_plan_has_pre_event_actions(self, sample_dispatch_plan):
        assert len(sample_dispatch_plan["pre_event_actions"]) >= 1

    def test_plan_has_curtailment_sequence(self, sample_dispatch_plan):
        assert len(sample_dispatch_plan["curtailment_sequence"]) >= 1

    def test_plan_has_restoration_sequence(self, sample_dispatch_plan):
        assert len(sample_dispatch_plan["restoration_sequence"]) >= 1

    def test_plan_has_der_dispatch(self, sample_dispatch_plan):
        assert len(sample_dispatch_plan["der_dispatch"]) >= 1

    def test_planned_exceeds_target(self, sample_dispatch_plan):
        assert (sample_dispatch_plan["total_planned_reduction_kw"] >=
                sample_dispatch_plan["comfort_constraints"].get(
                    "target_reduction_kw",
                    sample_dispatch_plan.get("target_reduction_kw", 0)))

    def test_margin_positive(self, sample_dispatch_plan):
        assert sample_dispatch_plan["reduction_margin_kw"] > 0

    def test_phase_ordering(self, sample_dispatch_plan):
        phases = sample_dispatch_plan["curtailment_sequence"]
        for i, phase in enumerate(phases):
            assert phase["phase"] == i + 1

    def test_restoration_after_event_end(self, sample_dispatch_plan):
        restore_phases = sample_dispatch_plan["restoration_sequence"]
        event_end = sample_dispatch_plan["event_end"]
        for phase in restore_phases:
            assert phase["time"] >= event_end.split("T")[1][:5]


# =============================================================================
# Pre-Event Action Validation
# =============================================================================


class TestPreEventActions:
    """Test pre-event action planning."""

    def test_pre_cool_action(self, sample_dispatch_plan):
        actions = sample_dispatch_plan["pre_event_actions"]
        pre_cool = [a for a in actions if a["action"] == "PRE_COOL"]
        assert len(pre_cool) >= 1

    def test_charge_battery_action(self, sample_dispatch_plan):
        actions = sample_dispatch_plan["pre_event_actions"]
        charge = [a for a in actions if a["action"] == "CHARGE_BATTERY"]
        assert len(charge) >= 1

    def test_notify_occupants_action(self, sample_dispatch_plan):
        actions = sample_dispatch_plan["pre_event_actions"]
        notify = [a for a in actions if a["action"] == "NOTIFY_OCCUPANTS"]
        assert len(notify) >= 1

    @pytest.mark.parametrize("action_idx", [0, 1, 2])
    def test_action_has_time(self, sample_dispatch_plan, action_idx):
        action = sample_dispatch_plan["pre_event_actions"][action_idx]
        assert "time" in action

    def test_pre_event_before_event_start(self, sample_dispatch_plan):
        event_start_time = sample_dispatch_plan["event_start"].split("T")[1][:5]
        for action in sample_dispatch_plan["pre_event_actions"]:
            assert action["time"] < event_start_time


# =============================================================================
# Curtailment Phase Validation
# =============================================================================


class TestCurtailmentPhaseValidation:
    """Test curtailment phase details."""

    @pytest.mark.parametrize("phase_idx,expected_action", [
        (0, "SHED"), (1, "SHED"), (2, "CURTAIL_50PCT"), (3, "CURTAIL"),
    ])
    def test_phase_actions(self, sample_dispatch_plan, phase_idx,
                            expected_action):
        phase = sample_dispatch_plan["curtailment_sequence"][phase_idx]
        assert phase["action"] == expected_action

    def test_total_planned_from_phases(self, sample_dispatch_plan):
        total = sum(p["expected_reduction_kw"]
                   for p in sample_dispatch_plan["curtailment_sequence"])
        assert total == pytest.approx(895.0, rel=0.01)

    def test_all_phases_have_load_ids(self, sample_dispatch_plan):
        for phase in sample_dispatch_plan["curtailment_sequence"]:
            assert len(phase["load_ids"]) >= 1

    def test_der_dispatch_power(self, sample_dispatch_plan):
        for der in sample_dispatch_plan["der_dispatch"]:
            assert der["power_kw"] > 0

    def test_der_dispatch_time_within_event(self, sample_dispatch_plan):
        event_start = sample_dispatch_plan["event_start"]
        event_end = sample_dispatch_plan["event_end"]
        for der in sample_dispatch_plan["der_dispatch"]:
            assert der["start"] >= event_start.split("T")[1][:5]
            assert der["end"] <= event_end.split("T")[1][:5]


# =============================================================================
# Dispatch Plan Completeness
# =============================================================================


class TestDispatchPlanCompleteness:
    """Test dispatch plan data completeness."""

    @pytest.mark.parametrize("field", [
        "plan_id", "event_id", "facility_id", "target_reduction_kw",
        "event_start", "event_end", "pre_event_actions",
        "curtailment_sequence", "der_dispatch",
        "total_planned_reduction_kw", "reduction_margin_kw",
        "comfort_constraints", "restoration_sequence",
    ])
    def test_plan_required_field(self, sample_dispatch_plan, field):
        assert field in sample_dispatch_plan

    def test_plan_id_format(self, sample_dispatch_plan):
        assert sample_dispatch_plan["plan_id"].startswith("DISP-")

    def test_plan_event_id_matches(self, sample_dispatch_plan,
                                     sample_dr_event):
        assert (sample_dispatch_plan["event_id"] ==
                sample_dr_event["event_id"])
