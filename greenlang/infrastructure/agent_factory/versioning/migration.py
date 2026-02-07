# -*- coding: utf-8 -*-
"""
Version Migration Framework - Register and execute agent version migrations.

Provides a framework for defining migration scripts that transform agent
data, configuration, or state when upgrading (up) or downgrading (down)
between versions. Supports ordered execution, dry-run mode, and migration
history tracking.

Example:
    >>> framework = VersionMigrationFramework()
    >>> framework.register("my-agent", MigrationScript(
    ...     from_version="1.0.0", to_version="1.1.0",
    ...     up_fn=migrate_1_0_to_1_1, down_fn=rollback_1_1_to_1_0,
    ...     description="Add carbon_intensity field",
    ... ))
    >>> result = await framework.migrate("my-agent", "1.0.0", "1.1.0")
    >>> assert result.success

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from greenlang.infrastructure.agent_factory.versioning.semver import SemanticVersion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MigrationFn = Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]
"""A migration function takes a context dict and returns the updated context."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MigrationScript:
    """A single version migration step.

    Attributes:
        from_version: Source version.
        to_version: Target version.
        up_fn: Function to execute when upgrading.
        down_fn: Function to execute when downgrading.
        description: Human-readable description of what this migration does.
    """

    from_version: str
    to_version: str
    up_fn: MigrationFn
    down_fn: Optional[MigrationFn] = None
    description: str = ""

    @property
    def from_semver(self) -> SemanticVersion:
        """Parse from_version as SemanticVersion."""
        return SemanticVersion.parse(self.from_version)

    @property
    def to_semver(self) -> SemanticVersion:
        """Parse to_version as SemanticVersion."""
        return SemanticVersion.parse(self.to_version)


@dataclass(frozen=True)
class MigrationStep:
    """Record of a single migration step execution.

    Attributes:
        from_version: Source version.
        to_version: Target version.
        direction: 'up' or 'down'.
        duration_ms: Execution time in milliseconds.
        success: Whether the step succeeded.
        error: Error message if the step failed.
    """

    from_version: str
    to_version: str
    direction: str
    duration_ms: float
    success: bool
    error: str = ""


@dataclass
class MigrationResult:
    """Outcome of a migration operation.

    Attributes:
        success: Whether all steps completed successfully.
        from_version: Starting version.
        to_version: Target version.
        steps_executed: List of executed migration steps.
        errors: Aggregated error messages.
        dry_run: Whether this was a dry-run (no actual changes).
    """

    success: bool
    from_version: str
    to_version: str
    steps_executed: List[MigrationStep] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass(frozen=True)
class MigrationHistoryEntry:
    """Record of a completed migration for audit trail.

    Attributes:
        agent_key: Agent that was migrated.
        from_version: Source version.
        to_version: Target version.
        executed_at: UTC timestamp of execution.
        direction: 'up' or 'down'.
        success: Whether the migration succeeded.
    """

    agent_key: str
    from_version: str
    to_version: str
    executed_at: str
    direction: str
    success: bool


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------


class VersionMigrationFramework:
    """Register and execute version migration scripts for GreenLang agents.

    Migration scripts are registered per agent and ordered by version.
    The framework finds the path from the current version to the target
    version and executes each step in order.

    Attributes:
        migrations: Registry of migration scripts per agent key.
        history: Ordered list of completed migration records.
    """

    def __init__(self) -> None:
        """Initialize the migration framework."""
        self._migrations: Dict[str, List[MigrationScript]] = {}
        self._history: List[MigrationHistoryEntry] = []

    def register(self, agent_key: str, script: MigrationScript) -> None:
        """Register a migration script for an agent.

        Args:
            agent_key: Agent identifier.
            script: Migration script defining up/down functions.
        """
        self._migrations.setdefault(agent_key, []).append(script)
        # Keep sorted by from_version
        self._migrations[agent_key].sort(
            key=lambda s: SemanticVersion.parse(s.from_version)
        )
        logger.info(
            "Registered migration for %s: %s -> %s (%s)",
            agent_key,
            script.from_version,
            script.to_version,
            script.description,
        )

    async def migrate(
        self,
        agent_key: str,
        from_version: str,
        to_version: str,
        context: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Execute migrations from one version to another.

        Determines the direction (up or down) and executes each
        migration step in order.

        Args:
            agent_key: Agent identifier.
            from_version: Current version.
            to_version: Target version.
            context: Optional context dict passed to migration functions.
            dry_run: If True, validate the migration path without executing.

        Returns:
            MigrationResult with execution details.
        """
        ctx = context or {}
        from_sv = SemanticVersion.parse(from_version)
        to_sv = SemanticVersion.parse(to_version)
        direction = "up" if to_sv > from_sv else "down"

        # Find migration path
        scripts = self._find_path(agent_key, from_version, to_version, direction)
        if scripts is None:
            return MigrationResult(
                success=False,
                from_version=from_version,
                to_version=to_version,
                errors=[
                    f"No migration path found from {from_version} to {to_version} "
                    f"for agent '{agent_key}'."
                ],
                dry_run=dry_run,
            )

        if dry_run:
            steps = [
                MigrationStep(
                    from_version=s.from_version,
                    to_version=s.to_version,
                    direction=direction,
                    duration_ms=0,
                    success=True,
                )
                for s in scripts
            ]
            logger.info(
                "Dry-run migration for %s: %s -> %s (%d steps)",
                agent_key, from_version, to_version, len(steps),
            )
            return MigrationResult(
                success=True,
                from_version=from_version,
                to_version=to_version,
                steps_executed=steps,
                dry_run=True,
            )

        # Execute migration steps
        executed: List[MigrationStep] = []
        errors: List[str] = []

        for script in scripts:
            step_start = time.monotonic()
            fn = script.up_fn if direction == "up" else script.down_fn

            if fn is None:
                err = (
                    f"No {direction} function for migration "
                    f"{script.from_version} -> {script.to_version}"
                )
                errors.append(err)
                executed.append(MigrationStep(
                    from_version=script.from_version,
                    to_version=script.to_version,
                    direction=direction,
                    duration_ms=0,
                    success=False,
                    error=err,
                ))
                break

            try:
                result = fn(ctx)
                # Support async migration functions
                if hasattr(result, "__await__"):
                    ctx = await result
                else:
                    ctx = result if isinstance(result, dict) else ctx

                step_ms = (time.monotonic() - step_start) * 1000
                executed.append(MigrationStep(
                    from_version=script.from_version,
                    to_version=script.to_version,
                    direction=direction,
                    duration_ms=step_ms,
                    success=True,
                ))
            except Exception as exc:
                step_ms = (time.monotonic() - step_start) * 1000
                err = f"Migration step failed ({script.from_version} -> {script.to_version}): {exc}"
                errors.append(err)
                logger.error(err, exc_info=True)
                executed.append(MigrationStep(
                    from_version=script.from_version,
                    to_version=script.to_version,
                    direction=direction,
                    duration_ms=step_ms,
                    success=False,
                    error=str(exc),
                ))
                break

        success = len(errors) == 0
        # Record history
        self._history.append(MigrationHistoryEntry(
            agent_key=agent_key,
            from_version=from_version,
            to_version=to_version,
            executed_at=datetime.now(timezone.utc).isoformat(),
            direction=direction,
            success=success,
        ))

        return MigrationResult(
            success=success,
            from_version=from_version,
            to_version=to_version,
            steps_executed=executed,
            errors=errors,
        )

    def get_history(self, agent_key: Optional[str] = None) -> List[MigrationHistoryEntry]:
        """Return migration history, optionally filtered by agent.

        Args:
            agent_key: If provided, filter to this agent only.

        Returns:
            List of migration history entries.
        """
        if agent_key:
            return [h for h in self._history if h.agent_key == agent_key]
        return list(self._history)

    def list_migrations(self, agent_key: str) -> List[MigrationScript]:
        """List all registered migration scripts for an agent.

        Args:
            agent_key: Agent identifier.

        Returns:
            Sorted list of migration scripts.
        """
        return list(self._migrations.get(agent_key, []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_path(
        self,
        agent_key: str,
        from_version: str,
        to_version: str,
        direction: str,
    ) -> Optional[List[MigrationScript]]:
        """Find an ordered list of migration scripts connecting two versions.

        For 'up' direction, finds scripts chained from_version -> ... -> to_version.
        For 'down' direction, finds scripts in reverse order.

        Returns:
            Ordered list of scripts, or None if no path exists.
        """
        all_scripts = self._migrations.get(agent_key, [])
        if not all_scripts:
            return None

        if direction == "up":
            return self._find_forward_path(all_scripts, from_version, to_version)
        return self._find_reverse_path(all_scripts, from_version, to_version)

    def _find_forward_path(
        self,
        scripts: List[MigrationScript],
        from_ver: str,
        to_ver: str,
    ) -> Optional[List[MigrationScript]]:
        """Find forward (upgrade) migration path using BFS."""
        # Build adjacency: from_version -> list of scripts
        adjacency: Dict[str, List[MigrationScript]] = {}
        for s in scripts:
            adjacency.setdefault(s.from_version, []).append(s)

        visited: set[str] = set()
        queue: List[tuple[str, List[MigrationScript]]] = [(from_ver, [])]

        while queue:
            current, path = queue.pop(0)
            if current == to_ver:
                return path
            if current in visited:
                continue
            visited.add(current)
            for script in adjacency.get(current, []):
                queue.append((script.to_version, path + [script]))

        return None

    def _find_reverse_path(
        self,
        scripts: List[MigrationScript],
        from_ver: str,
        to_ver: str,
    ) -> Optional[List[MigrationScript]]:
        """Find reverse (downgrade) migration path.

        For downgrade, we reverse the script direction: to_version -> from_version.
        """
        # Build reverse adjacency: to_version -> list of scripts
        adjacency: Dict[str, List[MigrationScript]] = {}
        for s in scripts:
            adjacency.setdefault(s.to_version, []).append(s)

        visited: set[str] = set()
        queue: List[tuple[str, List[MigrationScript]]] = [(from_ver, [])]

        while queue:
            current, path = queue.pop(0)
            if current == to_ver:
                return path
            if current in visited:
                continue
            visited.add(current)
            for script in adjacency.get(current, []):
                queue.append((script.from_version, path + [script]))

        return None
