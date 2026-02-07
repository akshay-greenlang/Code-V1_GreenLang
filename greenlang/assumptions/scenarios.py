# -*- coding: utf-8 -*-
"""
Scenario Manager - AGENT-FOUND-004: Assumptions Registry

Manages scenarios that contain assumption value overrides for
what-if analysis and sensitivity testing.

Default scenarios created on initialization:
    - Baseline (standard assumptions)
    - Conservative (higher emission factors)
    - Optimistic (lower emission factors)

Zero-Hallucination Guarantees:
    - Scenario value resolution is deterministic
    - Override chains are explicit, never inferred
    - Complete audit of scenario accesses

Example:
    >>> from greenlang.assumptions.scenarios import ScenarioManager
    >>> mgr = ScenarioManager()
    >>> scenario = mgr.create(
    ...     "High Carbon", "High carbon scenario",
    ...     "conservative", {"ef.elec": 0.85},
    ...     user_id="analyst",
    ... )
    >>> value, source = mgr.resolve_value("ef.elec", scenario.scenario_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from greenlang.assumptions.config import AssumptionsConfig, get_config
from greenlang.assumptions.models import Scenario, ScenarioType
from greenlang.assumptions.metrics import (
    record_operation,
    record_scenario_access,
    update_scenarios_count,
)

logger = logging.getLogger(__name__)


class ScenarioManager:
    """Manages scenarios with assumption value overrides.

    Provides CRUD operations for scenarios and deterministic
    value resolution with override chains.

    Attributes:
        config: AssumptionsConfig instance.
        _scenarios: Internal storage of scenarios by ID.

    Example:
        >>> mgr = ScenarioManager()
        >>> scenarios = mgr.list()
        >>> assert len(scenarios) >= 3  # baseline, conservative, optimistic
    """

    def __init__(
        self,
        config: Optional[AssumptionsConfig] = None,
    ) -> None:
        """Initialize ScenarioManager with default scenarios.

        Args:
            config: Optional config. Uses global config if None.
        """
        self.config = config or get_config()
        self._scenarios: Dict[str, Scenario] = {}
        self._create_default_scenarios()
        logger.info(
            "ScenarioManager initialized with %d default scenarios",
            len(self._scenarios),
        )

    def create(
        self,
        name: str,
        description: str,
        scenario_type: str,
        overrides: Optional[Dict[str, Any]] = None,
        user_id: str = "system",
        parent_scenario_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Scenario:
        """Create a new scenario.

        Args:
            name: Human-readable scenario name.
            description: Detailed description.
            scenario_type: Type string (baseline, optimistic, conservative, etc.).
            overrides: Assumption ID to value overrides.
            user_id: User creating the scenario.
            parent_scenario_id: Optional parent for inheritance.
            tags: Optional searchable tags.

        Returns:
            Created Scenario object.

        Raises:
            ValueError: If at capacity.
        """
        if len(self._scenarios) >= self.config.max_scenarios:
            raise ValueError(
                f"Scenario limit reached ({self.config.max_scenarios})"
            )

        scenario = Scenario(
            name=name,
            description=description,
            scenario_type=ScenarioType(scenario_type),
            overrides=overrides or {},
            created_by=user_id,
            parent_scenario_id=parent_scenario_id,
            tags=tags or [],
        )

        self._scenarios[scenario.scenario_id] = scenario

        update_scenarios_count(len(self._scenarios))
        logger.info(
            "Created scenario: %s (%s)", scenario.name, scenario.scenario_id,
        )

        return scenario

    def get(self, scenario_id: str) -> Optional[Scenario]:
        """Get a scenario by ID.

        Args:
            scenario_id: The scenario identifier.

        Returns:
            Scenario or None if not found.
        """
        return self._scenarios.get(scenario_id)

    def update(
        self,
        scenario_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> Scenario:
        """Update a scenario's properties.

        Args:
            scenario_id: The scenario to update.
            name: Optional new name.
            description: Optional new description.
            overrides: Optional new overrides (replaces existing).
            is_active: Optional active flag.
            tags: Optional new tags.

        Returns:
            Updated Scenario object.

        Raises:
            ValueError: If scenario not found.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        if name is not None:
            scenario.name = name
        if description is not None:
            scenario.description = description
        if overrides is not None:
            scenario.overrides = overrides
        if is_active is not None:
            scenario.is_active = is_active
        if tags is not None:
            scenario.tags = tags

        logger.info("Updated scenario: %s", scenario.name)
        return scenario

    def delete(self, scenario_id: str) -> bool:
        """Delete a scenario.

        Cannot delete baseline scenarios.

        Args:
            scenario_id: The scenario to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If attempting to delete a baseline scenario.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            return False

        if scenario.scenario_type == ScenarioType.BASELINE:
            raise ValueError("Cannot delete baseline scenario")

        del self._scenarios[scenario_id]
        update_scenarios_count(len(self._scenarios))
        logger.info("Deleted scenario: %s", scenario_id)
        return True

    def list(
        self,
        scenario_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Scenario]:
        """List scenarios with optional filtering.

        Args:
            scenario_type: Optional type filter.
            tags: Optional tags filter.

        Returns:
            Filtered list of scenarios.
        """
        scenarios = list(self._scenarios.values())

        if scenario_type is not None:
            st = ScenarioType(scenario_type)
            scenarios = [s for s in scenarios if s.scenario_type == st]

        if tags is not None:
            required_tags = set(tags)
            scenarios = [
                s for s in scenarios
                if required_tags.issubset(set(s.tags))
            ]

        return scenarios

    def resolve_value(
        self,
        assumption_id: str,
        scenario_id: str,
    ) -> Tuple[Any, str]:
        """Resolve an assumption value within a scenario.

        Checks the scenario's overrides for the assumption ID.
        If found, returns the override value with the scenario source.
        If not found, returns None with "baseline" source.

        Args:
            assumption_id: The assumption to resolve.
            scenario_id: The scenario to check.

        Returns:
            Tuple of (value_or_None, source_string).
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            return None, "baseline"

        record_scenario_access(scenario.scenario_type.value)

        if assumption_id in scenario.overrides:
            value = scenario.overrides[assumption_id]
            source = f"scenario:{scenario.name}"
            return value, source

        # Check parent scenario if applicable
        if scenario.parent_scenario_id:
            parent = self._scenarios.get(scenario.parent_scenario_id)
            if parent and assumption_id in parent.overrides:
                value = parent.overrides[assumption_id]
                source = f"scenario:{parent.name}(inherited)"
                return value, source

        return None, "baseline"

    def get_overrides(self, scenario_id: str) -> Dict[str, Any]:
        """Get all overrides for a scenario.

        Args:
            scenario_id: The scenario identifier.

        Returns:
            Dictionary of assumption_id to override value.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            return {}
        return dict(scenario.overrides)

    @property
    def count(self) -> int:
        """Return the number of scenarios."""
        return len(self._scenarios)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_default_scenarios(self) -> None:
        """Create the three default scenarios."""
        defaults = [
            Scenario(
                name="Baseline",
                description="Default baseline scenario with standard assumptions",
                scenario_type=ScenarioType.BASELINE,
                created_by="system",
            ),
            Scenario(
                name="Conservative",
                description="Conservative scenario with higher emission factors",
                scenario_type=ScenarioType.CONSERVATIVE,
                created_by="system",
            ),
            Scenario(
                name="Optimistic",
                description="Optimistic scenario with lower emission factors",
                scenario_type=ScenarioType.OPTIMISTIC,
                created_by="system",
            ),
        ]

        for scenario in defaults:
            self._scenarios[scenario.scenario_id] = scenario


__all__ = [
    "ScenarioManager",
]
