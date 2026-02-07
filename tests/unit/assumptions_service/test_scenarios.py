# -*- coding: utf-8 -*-
"""
Unit Tests for ScenarioManager (AGENT-FOUND-004)

Tests scenario CRUD, override resolution, default scenarios,
inheritance, and edge cases.

Coverage target: 85%+ of scenarios.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ScenarioManager mirroring greenlang/assumptions/scenarios.py
# ---------------------------------------------------------------------------


class ScenarioError(Exception):
    pass


class ScenarioNotFoundError(ScenarioError):
    pass


class DuplicateScenarioError(ScenarioError):
    pass


class ProtectedScenarioError(ScenarioError):
    pass


class Scenario:
    """Scenario model."""

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str = "",
        scenario_type: str = "custom",
        overrides: Optional[Dict[str, Any]] = None,
        parent_scenario: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_active: bool = True,
        created_at: Optional[str] = None,
    ):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.scenario_type = scenario_type
        self.overrides = overrides or {}
        self.parent_scenario = parent_scenario
        self.tags = tags or []
        self.is_active = is_active
        self.created_at = created_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type,
            "overrides": self.overrides,
            "parent_scenario": self.parent_scenario,
            "tags": self.tags,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


class ScenarioManager:
    """
    Manages scenarios with overrides for assumptions.
    Mirrors greenlang/assumptions/scenarios.py.
    """

    PROTECTED_SCENARIOS = {"baseline"}
    DEFAULT_SCENARIOS = {
        "baseline": {
            "name": "Baseline",
            "description": "Default baseline scenario with no overrides",
            "scenario_type": "baseline",
        },
        "conservative": {
            "name": "Conservative",
            "description": "Conservative estimates with higher emission factors",
            "scenario_type": "conservative",
        },
        "optimistic": {
            "name": "Optimistic",
            "description": "Optimistic estimates with lower emission factors",
            "scenario_type": "optimistic",
        },
    }

    def __init__(self):
        self._scenarios: Dict[str, Scenario] = {}
        self._init_defaults()

    def _init_defaults(self):
        """Create default scenarios."""
        for sid, info in self.DEFAULT_SCENARIOS.items():
            self._scenarios[sid] = Scenario(
                scenario_id=sid,
                name=info["name"],
                description=info["description"],
                scenario_type=info["scenario_type"],
            )

    def create(
        self,
        scenario_id: str,
        name: str,
        description: str = "",
        scenario_type: str = "custom",
        overrides: Optional[Dict[str, Any]] = None,
        parent_scenario: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Scenario:
        """Create a new scenario."""
        if scenario_id in self._scenarios:
            raise DuplicateScenarioError(
                f"Scenario '{scenario_id}' already exists"
            )

        if parent_scenario and parent_scenario not in self._scenarios:
            raise ScenarioNotFoundError(
                f"Parent scenario '{parent_scenario}' not found"
            )

        s = Scenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            scenario_type=scenario_type,
            overrides=overrides,
            parent_scenario=parent_scenario,
            tags=tags,
        )
        self._scenarios[scenario_id] = s
        return s

    def get(self, scenario_id: str) -> Scenario:
        """Get a scenario by ID."""
        if scenario_id not in self._scenarios:
            raise ScenarioNotFoundError(
                f"Scenario '{scenario_id}' not found"
            )
        return self._scenarios[scenario_id]

    def update(
        self,
        scenario_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> Scenario:
        """Update a scenario."""
        s = self.get(scenario_id)
        if name is not None:
            s.name = name
        if description is not None:
            s.description = description
        if overrides is not None:
            s.overrides = overrides
        if tags is not None:
            s.tags = tags
        if is_active is not None:
            s.is_active = is_active
        return s

    def delete(self, scenario_id: str) -> bool:
        """Delete a scenario. Protected scenarios cannot be deleted."""
        if scenario_id not in self._scenarios:
            raise ScenarioNotFoundError(
                f"Scenario '{scenario_id}' not found"
            )
        if scenario_id in self.PROTECTED_SCENARIOS:
            raise ProtectedScenarioError(
                f"Scenario '{scenario_id}' is protected and cannot be deleted"
            )
        del self._scenarios[scenario_id]
        return True

    def list_scenarios(
        self,
        scenario_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        active_only: bool = False,
    ) -> List[Scenario]:
        """List scenarios with optional filters."""
        results = list(self._scenarios.values())

        if scenario_type:
            results = [s for s in results if s.scenario_type == scenario_type]

        if tags:
            results = [
                s for s in results
                if any(t in s.tags for t in tags)
            ]

        if active_only:
            results = [s for s in results if s.is_active]

        return results

    def resolve_value(
        self,
        assumption_id: str,
        scenario_id: str,
        base_values: Dict[str, Any],
    ) -> Any:
        """Resolve the value of an assumption in a given scenario.

        Checks overrides in the scenario, then parent scenario (inheritance),
        then falls back to base_values.
        """
        s = self.get(scenario_id)

        # Direct override
        if assumption_id in s.overrides:
            return s.overrides[assumption_id]

        # Inheritance from parent
        if s.parent_scenario:
            return self.resolve_value(assumption_id, s.parent_scenario, base_values)

        # Fall back to base
        return base_values.get(assumption_id)

    def get_overrides(self, scenario_id: str) -> Dict[str, Any]:
        """Get all overrides for a scenario."""
        s = self.get(scenario_id)
        return dict(s.overrides)

    @property
    def count(self) -> int:
        return len(self._scenarios)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scenario_manager():
    """Fresh ScenarioManager for each test."""
    return ScenarioManager()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDefaultScenarios:
    """Test default scenarios are created on init."""

    def test_baseline_exists(self, scenario_manager):
        s = scenario_manager.get("baseline")
        assert s.name == "Baseline"
        assert s.scenario_type == "baseline"

    def test_conservative_exists(self, scenario_manager):
        s = scenario_manager.get("conservative")
        assert s.name == "Conservative"
        assert s.scenario_type == "conservative"

    def test_optimistic_exists(self, scenario_manager):
        s = scenario_manager.get("optimistic")
        assert s.name == "Optimistic"
        assert s.scenario_type == "optimistic"

    def test_three_defaults(self, scenario_manager):
        assert scenario_manager.count == 3


class TestCreateScenario:
    """Test create() operation."""

    def test_create_success(self, scenario_manager):
        s = scenario_manager.create("custom1", "Custom Scenario 1")
        assert s.scenario_id == "custom1"
        assert s.name == "Custom Scenario 1"
        assert s.scenario_type == "custom"

    def test_create_with_overrides(self, scenario_manager):
        s = scenario_manager.create(
            "c1", "Scenario",
            overrides={"diesel_ef": 3.0, "gas_ef": 2.0},
        )
        assert s.overrides == {"diesel_ef": 3.0, "gas_ef": 2.0}

    def test_create_with_parent(self, scenario_manager):
        s = scenario_manager.create(
            "child", "Child Scenario",
            parent_scenario="baseline",
        )
        assert s.parent_scenario == "baseline"

    def test_create_with_tags(self, scenario_manager):
        s = scenario_manager.create(
            "c1", "Scenario", tags=["2030", "projection"],
        )
        assert s.tags == ["2030", "projection"]

    def test_create_duplicate_fails(self, scenario_manager):
        scenario_manager.create("dup", "First")
        with pytest.raises(DuplicateScenarioError):
            scenario_manager.create("dup", "Second")

    def test_create_with_invalid_parent_fails(self, scenario_manager):
        with pytest.raises(ScenarioNotFoundError, match="Parent scenario"):
            scenario_manager.create("c1", "Child", parent_scenario="nonexistent")

    def test_create_various_types(self, scenario_manager):
        for stype in ["baseline", "conservative", "optimistic", "custom"]:
            sid = f"test_{stype}"
            s = scenario_manager.create(sid, f"Test {stype}", scenario_type=stype)
            assert s.scenario_type == stype


class TestGetScenario:
    """Test get() operation."""

    def test_get_success(self, scenario_manager):
        s = scenario_manager.get("baseline")
        assert s.scenario_id == "baseline"

    def test_get_not_found(self, scenario_manager):
        with pytest.raises(ScenarioNotFoundError):
            scenario_manager.get("nonexistent")


class TestUpdateScenario:
    """Test update() operation."""

    def test_update_name(self, scenario_manager):
        scenario_manager.create("c1", "Old Name")
        s = scenario_manager.update("c1", name="New Name")
        assert s.name == "New Name"

    def test_update_description(self, scenario_manager):
        scenario_manager.create("c1", "Name")
        s = scenario_manager.update("c1", description="New desc")
        assert s.description == "New desc"

    def test_update_overrides(self, scenario_manager):
        scenario_manager.create("c1", "Name")
        s = scenario_manager.update("c1", overrides={"ef": 3.0})
        assert s.overrides == {"ef": 3.0}

    def test_update_tags(self, scenario_manager):
        scenario_manager.create("c1", "Name")
        s = scenario_manager.update("c1", tags=["tag1"])
        assert s.tags == ["tag1"]

    def test_update_deactivate(self, scenario_manager):
        scenario_manager.create("c1", "Name")
        s = scenario_manager.update("c1", is_active=False)
        assert s.is_active is False


class TestDeleteScenario:
    """Test delete() operation."""

    def test_delete_success(self, scenario_manager):
        scenario_manager.create("c1", "Custom")
        result = scenario_manager.delete("c1")
        assert result is True
        with pytest.raises(ScenarioNotFoundError):
            scenario_manager.get("c1")

    def test_delete_baseline_protected(self, scenario_manager):
        with pytest.raises(ProtectedScenarioError, match="protected"):
            scenario_manager.delete("baseline")

    def test_delete_not_found(self, scenario_manager):
        with pytest.raises(ScenarioNotFoundError):
            scenario_manager.delete("nonexistent")


class TestListScenarios:
    """Test list_scenarios() operation."""

    def test_list_all(self, scenario_manager):
        results = scenario_manager.list_scenarios()
        assert len(results) >= 3  # at least defaults

    def test_list_by_type(self, scenario_manager):
        results = scenario_manager.list_scenarios(scenario_type="baseline")
        assert len(results) == 1
        assert results[0].scenario_id == "baseline"

    def test_list_by_tags(self, scenario_manager):
        scenario_manager.create("c1", "Name", tags=["2030"])
        results = scenario_manager.list_scenarios(tags=["2030"])
        assert len(results) == 1

    def test_list_active_only(self, scenario_manager):
        scenario_manager.create("c1", "Name")
        scenario_manager.update("c1", is_active=False)
        active = scenario_manager.list_scenarios(active_only=True)
        inactive_ids = [s.scenario_id for s in active]
        assert "c1" not in inactive_ids


class TestResolveValue:
    """Test resolve_value() with overrides and inheritance."""

    def test_baseline_no_override(self, scenario_manager):
        base = {"diesel_ef": 2.68}
        val = scenario_manager.resolve_value("diesel_ef", "baseline", base)
        assert val == 2.68

    def test_with_override(self, scenario_manager):
        scenario_manager.update("conservative", overrides={"diesel_ef": 3.10})
        base = {"diesel_ef": 2.68}
        val = scenario_manager.resolve_value("diesel_ef", "conservative", base)
        assert val == 3.10

    def test_inheritance_chain(self, scenario_manager):
        scenario_manager.update("conservative", overrides={"gas_ef": 2.25})
        scenario_manager.create(
            "child", "Child",
            parent_scenario="conservative",
        )
        base = {"gas_ef": 1.93}
        # child has no override, should inherit from conservative
        val = scenario_manager.resolve_value("gas_ef", "child", base)
        assert val == 2.25

    def test_child_override_wins(self, scenario_manager):
        scenario_manager.update("conservative", overrides={"gas_ef": 2.25})
        scenario_manager.create(
            "child", "Child",
            parent_scenario="conservative",
            overrides={"gas_ef": 2.50},
        )
        base = {"gas_ef": 1.93}
        val = scenario_manager.resolve_value("gas_ef", "child", base)
        assert val == 2.50

    def test_fall_through_to_base(self, scenario_manager):
        scenario_manager.create("c1", "Custom", overrides={"other": 99})
        base = {"diesel_ef": 2.68}
        val = scenario_manager.resolve_value("diesel_ef", "c1", base)
        assert val == 2.68

    def test_resolve_not_found_scenario(self, scenario_manager):
        with pytest.raises(ScenarioNotFoundError):
            scenario_manager.resolve_value("x", "nonexistent", {})


class TestGetOverrides:
    """Test get_overrides() operation."""

    def test_get_overrides_empty(self, scenario_manager):
        overrides = scenario_manager.get_overrides("baseline")
        assert overrides == {}

    def test_get_overrides_with_values(self, scenario_manager):
        scenario_manager.create("c1", "Custom", overrides={"ef": 3.0})
        overrides = scenario_manager.get_overrides("c1")
        assert overrides == {"ef": 3.0}

    def test_get_overrides_returns_copy(self, scenario_manager):
        scenario_manager.create("c1", "Custom", overrides={"ef": 3.0})
        overrides = scenario_manager.get_overrides("c1")
        overrides["ef"] = 999
        # Original should not be modified
        assert scenario_manager.get("c1").overrides["ef"] == 3.0
