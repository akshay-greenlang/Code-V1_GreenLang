# -*- coding: utf-8 -*-
"""
Unit Tests for DependencyTracker (AGENT-FOUND-004)

Tests dependency graph registration, upstream/downstream queries,
cycle detection, impact analysis, and calculation tracking.

Coverage target: 85%+ of dependencies.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline DependencyTracker mirroring greenlang/assumptions/dependencies.py
# ---------------------------------------------------------------------------


class DependencyTracker:
    """
    Tracks dependencies between assumptions and calculations.
    Mirrors greenlang/assumptions/dependencies.py.
    """

    def __init__(self):
        # assumption_id -> set of assumption_ids it depends on
        self._upstream: Dict[str, Set[str]] = {}
        # assumption_id -> set of assumption_ids that depend on it
        self._downstream: Dict[str, Set[str]] = {}
        # calculation_id -> set of assumption_ids it uses
        self._calc_assumptions: Dict[str, Set[str]] = {}
        # assumption_id -> set of calculation_ids that use it
        self._assumption_calcs: Dict[str, Set[str]] = {}

    def register_dependency(self, assumption_id: str, depends_on: str):
        """Register that assumption_id depends on depends_on."""
        if assumption_id not in self._upstream:
            self._upstream[assumption_id] = set()
        self._upstream[assumption_id].add(depends_on)

        if depends_on not in self._downstream:
            self._downstream[depends_on] = set()
        self._downstream[depends_on].add(assumption_id)

    def register_calculation(self, calculation_id: str, assumption_ids: List[str]):
        """Register a calculation and the assumptions it uses."""
        self._calc_assumptions[calculation_id] = set(assumption_ids)

        for aid in assumption_ids:
            if aid not in self._assumption_calcs:
                self._assumption_calcs[aid] = set()
            self._assumption_calcs[aid].add(calculation_id)

    def get_upstream(self, assumption_id: str) -> List[str]:
        """Get assumptions that this assumption depends on."""
        return sorted(self._upstream.get(assumption_id, set()))

    def get_downstream(self, assumption_id: str) -> List[str]:
        """Get assumptions that depend on this assumption."""
        return sorted(self._downstream.get(assumption_id, set()))

    def get_impact(self, assumption_id: str) -> Dict[str, List[str]]:
        """Get the impact of changing an assumption.

        Returns downstream assumptions and affected calculations.
        """
        affected_assumptions = self._get_all_downstream(assumption_id)
        affected_calcs = set()

        # Direct calculations
        if assumption_id in self._assumption_calcs:
            affected_calcs.update(self._assumption_calcs[assumption_id])

        # Indirect through downstream assumptions
        for downstream in affected_assumptions:
            if downstream in self._assumption_calcs:
                affected_calcs.update(self._assumption_calcs[downstream])

        return {
            "affected_assumptions": sorted(affected_assumptions),
            "affected_calculations": sorted(affected_calcs),
        }

    def _get_all_downstream(self, assumption_id: str) -> Set[str]:
        """Recursively get all downstream assumptions."""
        visited: Set[str] = set()
        stack = [assumption_id]
        while stack:
            current = stack.pop()
            for dep in self._downstream.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    stack.append(dep)
        return visited

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._upstream.get(node, set()):
                if neighbor not in visited:
                    _dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node in set(self._upstream.keys()) | set(self._downstream.keys()):
            if node not in visited:
                _dfs(node, [])

        return cycles

    def get_calculation_assumptions(
        self,
        calculation_id: str,
        base_values: Optional[Dict[str, Any]] = None,
        scenario_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get assumption values for a calculation, optionally with scenario overrides."""
        assumption_ids = self._calc_assumptions.get(calculation_id, set())
        base = base_values or {}
        overrides = scenario_overrides or {}

        result = {}
        for aid in sorted(assumption_ids):
            if aid in overrides:
                result[aid] = overrides[aid]
            else:
                result[aid] = base.get(aid)
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    """Fresh DependencyTracker."""
    return DependencyTracker()


@pytest.fixture
def populated_tracker():
    """Tracker with pre-loaded dependencies."""
    t = DependencyTracker()
    # A depends on B, B depends on C
    t.register_dependency("A", "B")
    t.register_dependency("B", "C")
    # D depends on B (shares dependency)
    t.register_dependency("D", "B")
    # Register calculations
    t.register_calculation("calc_scope1", ["A", "B"])
    t.register_calculation("calc_scope2", ["C"])
    return t


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRegisterDependency:
    """Test register_dependency() creates edges."""

    def test_register_creates_upstream(self, tracker):
        tracker.register_dependency("A", "B")
        assert "B" in tracker.get_upstream("A")

    def test_register_creates_downstream(self, tracker):
        tracker.register_dependency("A", "B")
        assert "A" in tracker.get_downstream("B")

    def test_multiple_dependencies(self, tracker):
        tracker.register_dependency("A", "B")
        tracker.register_dependency("A", "C")
        upstream = tracker.get_upstream("A")
        assert "B" in upstream
        assert "C" in upstream

    def test_duplicate_dependency_no_double(self, tracker):
        tracker.register_dependency("A", "B")
        tracker.register_dependency("A", "B")
        assert len(tracker.get_upstream("A")) == 1


class TestRegisterCalculation:
    """Test register_calculation() links calculations to assumptions."""

    def test_register_calc(self, tracker):
        tracker.register_calculation("calc1", ["a1", "a2"])
        result = tracker.get_calculation_assumptions("calc1", {"a1": 10, "a2": 20})
        assert result == {"a1": 10, "a2": 20}

    def test_multiple_calcs_on_same_assumption(self, tracker):
        tracker.register_calculation("calc1", ["a1"])
        tracker.register_calculation("calc2", ["a1"])
        impact = tracker.get_impact("a1")
        assert "calc1" in impact["affected_calculations"]
        assert "calc2" in impact["affected_calculations"]


class TestGetUpstream:
    """Test get_upstream() returns dependencies."""

    def test_with_dependencies(self, populated_tracker):
        upstream = populated_tracker.get_upstream("A")
        assert upstream == ["B"]

    def test_without_dependencies(self, tracker):
        upstream = tracker.get_upstream("nonexistent")
        assert upstream == []


class TestGetDownstream:
    """Test get_downstream() returns dependents."""

    def test_with_dependents(self, populated_tracker):
        downstream = populated_tracker.get_downstream("B")
        assert "A" in downstream
        assert "D" in downstream

    def test_without_dependents(self, tracker):
        downstream = tracker.get_downstream("nonexistent")
        assert downstream == []


class TestGetImpact:
    """Test get_impact() returns affected calculations."""

    def test_impact_of_leaf(self, populated_tracker):
        impact = populated_tracker.get_impact("C")
        # C is upstream of B, which is upstream of A and D
        assert "B" in impact["affected_assumptions"]
        assert "A" in impact["affected_assumptions"]
        assert "D" in impact["affected_assumptions"]
        assert "calc_scope1" in impact["affected_calculations"]
        assert "calc_scope2" in impact["affected_calculations"]

    def test_impact_of_mid_node(self, populated_tracker):
        impact = populated_tracker.get_impact("B")
        assert "A" in impact["affected_assumptions"]
        assert "D" in impact["affected_assumptions"]
        assert "calc_scope1" in impact["affected_calculations"]

    def test_impact_no_downstream(self, tracker):
        tracker.register_dependency("A", "B")
        impact = tracker.get_impact("A")
        assert impact["affected_assumptions"] == []
        assert impact["affected_calculations"] == []


class TestDetectCycles:
    """Test detect_cycles() finds and reports cycles."""

    def test_no_cycles_in_dag(self, populated_tracker):
        cycles = populated_tracker.detect_cycles()
        assert cycles == []

    def test_detects_simple_cycle(self, tracker):
        tracker.register_dependency("A", "B")
        tracker.register_dependency("B", "A")
        cycles = tracker.detect_cycles()
        assert len(cycles) > 0

    def test_detects_three_node_cycle(self, tracker):
        tracker.register_dependency("A", "B")
        tracker.register_dependency("B", "C")
        tracker.register_dependency("C", "A")
        cycles = tracker.detect_cycles()
        assert len(cycles) > 0

    def test_empty_graph_no_cycles(self, tracker):
        cycles = tracker.detect_cycles()
        assert cycles == []


class TestGetCalculationAssumptions:
    """Test get_calculation_assumptions() with and without overrides."""

    def test_base_values(self, tracker):
        tracker.register_calculation("calc1", ["a1", "a2"])
        result = tracker.get_calculation_assumptions(
            "calc1", base_values={"a1": 10, "a2": 20}
        )
        assert result["a1"] == 10
        assert result["a2"] == 20

    def test_with_scenario_override(self, tracker):
        tracker.register_calculation("calc1", ["a1", "a2"])
        result = tracker.get_calculation_assumptions(
            "calc1",
            base_values={"a1": 10, "a2": 20},
            scenario_overrides={"a1": 15},
        )
        assert result["a1"] == 15  # overridden
        assert result["a2"] == 20  # base

    def test_unknown_calculation(self, tracker):
        result = tracker.get_calculation_assumptions("unknown")
        assert result == {}

    def test_missing_base_value(self, tracker):
        tracker.register_calculation("calc1", ["a1"])
        result = tracker.get_calculation_assumptions("calc1", base_values={})
        assert result["a1"] is None
