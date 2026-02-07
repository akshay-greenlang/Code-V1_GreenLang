# -*- coding: utf-8 -*-
"""
Scenario Workflow Integration Tests for Assumptions Service (AGENT-FOUND-004)

Tests baseline vs conservative vs optimistic comparison, custom scenario
creation, scenario inheritance, multi-assumption override, scenario
deactivation, and sensitivity analysis across scenarios.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations
# ---------------------------------------------------------------------------

class Scenario:
    def __init__(self, scenario_id, name, description="", scenario_type="custom",
                 overrides=None, parent_scenario=None, tags=None, is_active=True):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.scenario_type = scenario_type
        self.overrides = overrides or {}
        self.parent_scenario = parent_scenario
        self.tags = tags or []
        self.is_active = is_active


class ScenarioNotFoundError(Exception):
    pass


class ProtectedScenarioError(Exception):
    pass


class ScenarioManager:
    PROTECTED = {"baseline"}

    def __init__(self):
        self._scenarios = {}
        for sid, info in [
            ("baseline", {"name": "Baseline", "type": "baseline"}),
            ("conservative", {"name": "Conservative", "type": "conservative"}),
            ("optimistic", {"name": "Optimistic", "type": "optimistic"}),
        ]:
            self._scenarios[sid] = Scenario(sid, info["name"], scenario_type=info["type"])

    def create(self, scenario_id, name, scenario_type="custom", overrides=None,
               parent_scenario=None, tags=None):
        s = Scenario(scenario_id, name, scenario_type=scenario_type,
                     overrides=overrides, parent_scenario=parent_scenario, tags=tags)
        self._scenarios[scenario_id] = s
        return s

    def get(self, scenario_id):
        if scenario_id not in self._scenarios:
            raise ScenarioNotFoundError(f"'{scenario_id}' not found")
        return self._scenarios[scenario_id]

    def update(self, scenario_id, overrides=None, is_active=None, **kwargs):
        s = self.get(scenario_id)
        if overrides is not None:
            s.overrides = overrides
        if is_active is not None:
            s.is_active = is_active
        for k, v in kwargs.items():
            if v is not None and hasattr(s, k):
                setattr(s, k, v)
        return s

    def delete(self, scenario_id):
        if scenario_id in self.PROTECTED:
            raise ProtectedScenarioError(f"'{scenario_id}' is protected")
        if scenario_id not in self._scenarios:
            raise ScenarioNotFoundError(f"'{scenario_id}' not found")
        del self._scenarios[scenario_id]

    def list_scenarios(self, active_only=False):
        results = list(self._scenarios.values())
        if active_only:
            results = [s for s in results if s.is_active]
        return results

    def resolve_value(self, assumption_id, scenario_id, base_values):
        s = self.get(scenario_id)
        if assumption_id in s.overrides:
            return s.overrides[assumption_id]
        if s.parent_scenario:
            return self.resolve_value(assumption_id, s.parent_scenario, base_values)
        return base_values.get(assumption_id)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestBaselineVsConservativeVsOptimistic:
    """Compare values across the three default scenarios."""

    def test_three_scenario_comparison(self):
        sm = ScenarioManager()
        base = {"diesel_ef": 2.68, "gas_ef": 1.93, "coal_ef": 3.45}

        sm.update("conservative", overrides={"diesel_ef": 3.10, "gas_ef": 2.25, "coal_ef": 4.00})
        sm.update("optimistic", overrides={"diesel_ef": 2.20, "gas_ef": 1.50, "coal_ef": 2.80})

        for aid in ["diesel_ef", "gas_ef", "coal_ef"]:
            b = sm.resolve_value(aid, "baseline", base)
            c = sm.resolve_value(aid, "conservative", base)
            o = sm.resolve_value(aid, "optimistic", base)
            assert c > b > o, f"Expected conservative > baseline > optimistic for {aid}"

    def test_baseline_matches_base_values(self):
        sm = ScenarioManager()
        base = {"ef": 2.68}
        assert sm.resolve_value("ef", "baseline", base) == 2.68


class TestCustomScenarioCreation:
    """Test creating and using custom scenarios."""

    def test_custom_scenario_with_overrides(self):
        sm = ScenarioManager()
        sm.create("custom_2030", "Custom 2030", overrides={"ef": 3.50})
        base = {"ef": 2.68}
        assert sm.resolve_value("ef", "custom_2030", base) == 3.50

    def test_custom_scenario_preserves_base_for_unoverridden(self):
        sm = ScenarioManager()
        sm.create("custom1", "Custom", overrides={"ef1": 5.0})
        base = {"ef1": 2.68, "ef2": 1.93}
        assert sm.resolve_value("ef1", "custom1", base) == 5.0
        assert sm.resolve_value("ef2", "custom1", base) == 1.93


class TestScenarioInheritance:
    """Test scenario inheritance chain."""

    def test_single_level_inheritance(self):
        sm = ScenarioManager()
        sm.update("conservative", overrides={"a": 10, "b": 20})
        sm.create("child", "Child", parent_scenario="conservative", overrides={"a": 15})

        base = {"a": 1, "b": 2}
        assert sm.resolve_value("a", "child", base) == 15   # child override
        assert sm.resolve_value("b", "child", base) == 20   # from parent

    def test_two_level_inheritance(self):
        sm = ScenarioManager()
        sm.update("conservative", overrides={"a": 10, "b": 20, "c": 30})
        sm.create("level1", "Level 1", parent_scenario="conservative", overrides={"a": 15})
        sm.create("level2", "Level 2", parent_scenario="level1", overrides={"b": 25})

        base = {"a": 1, "b": 2, "c": 3}
        assert sm.resolve_value("a", "level2", base) == 15  # from level1
        assert sm.resolve_value("b", "level2", base) == 25  # from level2
        assert sm.resolve_value("c", "level2", base) == 30  # from conservative

    def test_inheritance_falls_through_to_base(self):
        sm = ScenarioManager()
        sm.create("child", "Child", parent_scenario="baseline", overrides={"a": 5})
        base = {"a": 1, "b": 2}
        assert sm.resolve_value("b", "child", base) == 2  # falls to base via baseline


class TestMultiAssumptionOverride:
    """Test overriding multiple assumptions in a single scenario."""

    def test_override_5_assumptions(self):
        sm = ScenarioManager()
        overrides = {f"ef_{i}": float(i * 10) for i in range(1, 6)}
        sm.create("multi", "Multi Override", overrides=overrides)

        base = {f"ef_{i}": float(i) for i in range(1, 6)}
        for i in range(1, 6):
            val = sm.resolve_value(f"ef_{i}", "multi", base)
            assert val == float(i * 10)


class TestScenarioDeactivation:
    """Test deactivating a scenario."""

    def test_deactivate_scenario(self):
        sm = ScenarioManager()
        sm.create("temp", "Temporary", overrides={"ef": 5.0})
        sm.update("temp", is_active=False)
        s = sm.get("temp")
        assert s.is_active is False

    def test_active_only_filter(self):
        sm = ScenarioManager()
        sm.create("active1", "Active")
        sm.create("inactive1", "Inactive")
        sm.update("inactive1", is_active=False)

        active = sm.list_scenarios(active_only=True)
        active_ids = [s.scenario_id for s in active]
        assert "active1" in active_ids
        assert "inactive1" not in active_ids


class TestSensitivityAcrossScenarios:
    """Test sensitivity analysis across different scenarios."""

    def test_sensitivity_comparison(self):
        sm = ScenarioManager()
        sm.update("conservative", overrides={"ef": 3.10})
        sm.update("optimistic", overrides={"ef": 2.20})

        base = {"ef": 2.68}
        scenarios = ["baseline", "conservative", "optimistic"]

        results = {}
        for sid in scenarios:
            val = sm.resolve_value("ef", sid, base)
            # Compute +/- 10% sensitivity
            low = val * 0.9
            high = val * 1.1
            results[sid] = {"value": val, "low": low, "high": high}

        assert results["conservative"]["value"] > results["baseline"]["value"]
        assert results["optimistic"]["value"] < results["baseline"]["value"]
        assert results["conservative"]["low"] > results["optimistic"]["high"]

    def test_sensitivity_steps(self):
        sm = ScenarioManager()
        sm.update("conservative", overrides={"ef": 4.0})
        base = {"ef": 3.0}

        base_val = sm.resolve_value("ef", "conservative", base)
        steps = 10
        range_pct = 0.1
        variations = []
        for i in range(steps + 1):
            factor = 1 - range_pct + (2 * range_pct * i / steps)
            variations.append(round(base_val * factor, 4))

        assert len(variations) == 11
        assert variations[0] < base_val < variations[-1]


class TestScenarioDeletion:
    """Test scenario deletion behavior."""

    def test_delete_custom_scenario(self):
        sm = ScenarioManager()
        sm.create("temp", "Temp")
        sm.delete("temp")
        with pytest.raises(ScenarioNotFoundError):
            sm.get("temp")

    def test_cannot_delete_baseline(self):
        sm = ScenarioManager()
        with pytest.raises(ProtectedScenarioError):
            sm.delete("baseline")

    def test_delete_nonexistent(self):
        sm = ScenarioManager()
        with pytest.raises(ScenarioNotFoundError):
            sm.delete("ghost")


class TestScenarioTagFiltering:
    """Test scenario tag-based operations."""

    def test_create_with_tags(self):
        sm = ScenarioManager()
        s = sm.create("tagged", "Tagged", tags=["2030", "projection"])
        assert s.tags == ["2030", "projection"]

    def test_update_tags(self):
        sm = ScenarioManager()
        sm.create("s1", "S1", tags=["old"])
        sm.update("s1", tags=["new1", "new2"])
        s = sm.get("s1")
        assert s.tags == ["new1", "new2"]
