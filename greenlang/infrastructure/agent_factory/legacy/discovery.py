"""
Legacy Agent Discovery - Agent Factory (INFRA-010)

Scans ``greenlang/agents/`` for classes extending AgentSpecV2Base,
DeterministicAgent, ReasoningAgent, or InsightAgent.  Creates synthetic
registrations for legacy agents that do not yet have ``agent.pack.yaml``
files, enabling the Agent Factory to manage all 119 existing agents
without requiring an immediate rewrite.

Classes:
    - DiscoveredAgent: Value object representing a discovered legacy agent.
    - LegacyAgentDiscovery: Codebase scanner that locates legacy agents.

Example:
    >>> discovery = LegacyAgentDiscovery()
    >>> agents = discovery.discover_all()
    >>> for agent in agents:
    ...     print(agent.agent_key, agent.base_class)
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised base classes whose subclasses count as "agents"
_BASE_CLASS_NAMES: Set[str] = frozenset({
    "AgentSpecV2Base",
    "DeterministicAgent",
    "ReasoningAgent",
    "InsightAgent",
})

# Module-level names that should never be treated as agent classes
_EXCLUDED_CLASS_NAMES: Set[str] = frozenset({
    "AgentSpecV2Base",
    "DeterministicAgent",
    "ReasoningAgent",
    "InsightAgent",
    "ABC",
})

# Submodule paths that should be skipped during scanning
_SKIP_MODULE_PATTERNS: Set[str] = frozenset({
    "greenlang.agents.base",
    "greenlang.agents.base_agents",
    "greenlang.agents.agentspec_v2_base",
    "greenlang.agents.categories",
    "greenlang.agents.decorators",
    "greenlang.agents.formulas",
})

_AGENT_KEY_PREFIX = "gl-"

# ---------------------------------------------------------------------------
# DiscoveredAgent Value Object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredAgent:
    """Represents a single legacy agent found during codebase scanning.

    Attributes:
        module_path: Fully qualified Python module path
            (e.g. ``greenlang.agents.intake.carbon_intake``).
        class_name: The agent class name (e.g. ``CarbonIntakeAgent``).
        base_class: Name of the recognised base class this agent extends
            (``DeterministicAgent``, ``ReasoningAgent``, ``InsightAgent``,
            or ``AgentSpecV2Base``).
        agent_key: Auto-derived stable key (e.g. ``gl-carbon-intake``).
        file_path: Absolute filesystem path to the source module.
        has_pack_yaml: Whether an ``agent.pack.yaml`` already exists in
            the agent's directory.
        has_tests: Whether a corresponding test file was detected.
        input_type_hint: String representation of the input type hint
            extracted from the agent class, if available.
        output_type_hint: String representation of the output type hint
            extracted from the agent class, if available.
    """

    module_path: str
    class_name: str
    base_class: str
    agent_key: str
    file_path: Path
    has_pack_yaml: bool = False
    has_tests: bool = False
    input_type_hint: str = ""
    output_type_hint: str = ""


# ---------------------------------------------------------------------------
# LegacyAgentDiscovery
# ---------------------------------------------------------------------------


class LegacyAgentDiscovery:
    """Scans the codebase for legacy agents not yet managed by Agent Factory.

    The scanner recursively walks the configured package paths, imports
    each module, and inspects its classes to find subclasses of the
    recognised base classes.  Results are cached after the first full
    scan.

    Attributes:
        BASE_CLASSES: Set of recognised base class names.

    Example:
        >>> discovery = LegacyAgentDiscovery(scan_paths=["greenlang.agents"])
        >>> agents = discovery.discover_all()
        >>> print(f"Found {len(agents)} legacy agents")
    """

    BASE_CLASSES: Set[str] = _BASE_CLASS_NAMES

    def __init__(
        self,
        scan_paths: Optional[List[str]] = None,
        *,
        project_root: Optional[Path] = None,
    ) -> None:
        """Initialize the legacy agent discovery scanner.

        Args:
            scan_paths: List of top-level dotted module paths to scan.
                Defaults to ``["greenlang.agents"]``.
            project_root: Filesystem root for resolving source files and
                checking for pack YAML / test files.  Defaults to
                ``Path.cwd()``.
        """
        self._scan_paths: List[str] = scan_paths or ["greenlang.agents"]
        self._project_root: Path = project_root or Path.cwd()
        self._discovered: Dict[str, DiscoveredAgent] = {}
        self._scan_errors: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_all(self) -> List[DiscoveredAgent]:
        """Scan all configured paths and return discovered agents.

        Results are cached; subsequent calls return the same list unless
        ``reset()`` is called first.

        Returns:
            List of DiscoveredAgent sorted by agent_key.
        """
        if self._discovered:
            return sorted(self._discovered.values(), key=lambda a: a.agent_key)

        start = time.perf_counter()
        logger.info(
            "LegacyAgentDiscovery: starting scan of %s",
            self._scan_paths,
        )

        for root_module in self._scan_paths:
            found = self._scan_module(root_module)
            for agent in found:
                self._discovered[agent.agent_key] = agent

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "LegacyAgentDiscovery: found %d agents in %.1fms (%d scan errors)",
            len(self._discovered),
            elapsed_ms,
            len(self._scan_errors),
        )

        return sorted(self._discovered.values(), key=lambda a: a.agent_key)

    def get_agent(self, agent_key: str) -> Optional[DiscoveredAgent]:
        """Return a previously discovered agent by key.

        Args:
            agent_key: The agent key to look up.

        Returns:
            DiscoveredAgent or None if not found.
        """
        return self._discovered.get(agent_key)

    @property
    def scan_errors(self) -> List[str]:
        """Return a list of import errors encountered during the last scan."""
        return list(self._scan_errors)

    def reset(self) -> None:
        """Clear cached results so the next ``discover_all`` rescans."""
        self._discovered.clear()
        self._scan_errors.clear()

    # ------------------------------------------------------------------
    # Internal: module scanning
    # ------------------------------------------------------------------

    def _scan_module(self, module_path: str) -> List[DiscoveredAgent]:
        """Recursively scan a module for agent classes.

        Args:
            module_path: Dotted module path (e.g. ``greenlang.agents``).

        Returns:
            List of DiscoveredAgent found in this module tree.
        """
        results: List[DiscoveredAgent] = []

        if self._should_skip_module(module_path):
            return results

        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            msg = f"Failed to import {module_path}: {exc}"
            logger.debug(msg)
            self._scan_errors.append(msg)
            return results

        # Inspect classes defined in this module
        results.extend(self._inspect_module_classes(mod, module_path))

        # Recurse into sub-packages
        if hasattr(mod, "__path__"):
            try:
                for importer, subname, is_pkg in pkgutil.iter_modules(
                    mod.__path__, prefix=f"{module_path}."
                ):
                    results.extend(self._scan_module(subname))
            except Exception as exc:
                msg = f"Failed to iterate sub-modules of {module_path}: {exc}"
                logger.debug(msg)
                self._scan_errors.append(msg)

        return results

    def _inspect_module_classes(
        self,
        mod: Any,
        module_path: str,
    ) -> List[DiscoveredAgent]:
        """Inspect all classes in a module for agent base class inheritance.

        Args:
            mod: The imported module object.
            module_path: Dotted path used for logging.

        Returns:
            List of DiscoveredAgent for classes that qualify.
        """
        results: List[DiscoveredAgent] = []

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            # Only consider classes *defined* in this module
            if getattr(obj, "__module__", None) != module_path:
                continue

            # Skip abstract bases and excluded names
            if name in _EXCLUDED_CLASS_NAMES:
                continue

            if inspect.isabstract(obj):
                continue

            base_class = self._detect_base_class(obj)
            if base_class is None:
                continue

            file_path = self._resolve_file_path(mod)
            if file_path is None:
                continue

            agent_key = self._derive_agent_key(module_path, name)
            has_pack = self._check_pack_yaml(file_path)
            has_tests = self._check_tests(module_path, name)
            input_hint, output_hint = self._extract_type_hints(obj)

            agent = DiscoveredAgent(
                module_path=module_path,
                class_name=name,
                base_class=base_class,
                agent_key=agent_key,
                file_path=file_path,
                has_pack_yaml=has_pack,
                has_tests=has_tests,
                input_type_hint=input_hint,
                output_type_hint=output_hint,
            )

            logger.debug(
                "Discovered legacy agent: %s (%s) -> %s",
                agent_key,
                base_class,
                module_path,
            )
            results.append(agent)

        return results

    # ------------------------------------------------------------------
    # Internal: detection helpers
    # ------------------------------------------------------------------

    def _detect_base_class(self, cls: Type) -> Optional[str]:
        """Determine which recognised base class the agent extends.

        Walks the MRO and returns the first match against
        ``BASE_CLASSES``.

        Args:
            cls: The class to inspect.

        Returns:
            The base class name string or None if no match.
        """
        for parent in inspect.getmro(cls):
            parent_name = parent.__name__
            if parent_name in self.BASE_CLASSES and parent is not cls:
                return parent_name
        return None

    @staticmethod
    def _derive_agent_key(module_path: str, class_name: str) -> str:
        """Derive a stable agent key from the module path.

        Convention:
            ``greenlang.agents.intake.carbon_intake`` becomes
            ``gl-carbon-intake``.  The ``greenlang.agents.`` prefix and
            any repeated path segments are stripped.

        Args:
            module_path: Dotted module path.
            class_name: Class name (used as fallback).

        Returns:
            An agent key like ``gl-carbon-intake``.
        """
        # Strip the common prefix
        key = module_path
        for prefix in ("greenlang.agents.", "greenlang."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break

        # Replace dots with hyphens, underscores with hyphens
        key = key.replace(".", "-").replace("_", "-")

        # Collapse repeated segments (e.g. "industry-industry" -> "industry")
        parts = key.split("-")
        deduped: List[str] = []
        for part in parts:
            if not deduped or part != deduped[-1]:
                deduped.append(part)
        key = "-".join(deduped)

        # Remove trailing "-agent" or "-agent-ai" for cleaner keys
        key = re.sub(r"-agent(-ai)?(-v\d+)?$", "", key)

        if not key:
            # Fallback: derive from class name
            key = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name).lower()
            key = re.sub(r"-agent(-ai)?$", "", key)

        return f"{_AGENT_KEY_PREFIX}{key}"

    def _check_pack_yaml(self, file_path: Path) -> bool:
        """Check if agent.pack.yaml exists in the agent's directory.

        Args:
            file_path: Path to the agent Python source file.

        Returns:
            True if ``agent.pack.yaml`` is found.
        """
        parent_dir = file_path.parent
        return (parent_dir / "agent.pack.yaml").exists()

    def _check_tests(self, module_path: str, class_name: str) -> bool:
        """Heuristic check for the existence of test files.

        Looks for ``tests/test_<module_leaf>.py`` or
        ``tests/<subpath>/test_<module_leaf>.py`` under the project root.

        Args:
            module_path: Dotted module path.
            class_name: Agent class name.

        Returns:
            True if a likely test file was found.
        """
        leaf = module_path.rsplit(".", maxsplit=1)[-1]
        tests_dir = self._project_root / "tests"

        if not tests_dir.exists():
            return False

        # Check direct test file
        if (tests_dir / f"test_{leaf}.py").exists():
            return True

        # Check nested test directories
        for candidate in tests_dir.rglob(f"test_{leaf}.py"):
            return True

        return False

    @staticmethod
    def _resolve_file_path(mod: Any) -> Optional[Path]:
        """Resolve the filesystem path of a module.

        Args:
            mod: An imported Python module.

        Returns:
            Absolute Path to the module source file, or None.
        """
        source_file = getattr(mod, "__file__", None)
        if source_file is None:
            return None
        return Path(source_file).resolve()

    @staticmethod
    def _extract_type_hints(cls: Type) -> tuple[str, str]:
        """Extract input/output type hint strings from the agent class.

        Inspects the ``execute``, ``execute_impl``, ``run``, ``reason``,
        or ``calculate`` methods for type annotations.

        Args:
            cls: The agent class.

        Returns:
            Tuple of (input_type_hint, output_type_hint) as strings.
        """
        input_hint = ""
        output_hint = ""

        # Candidate entry-point methods in priority order
        method_names = [
            "execute_impl",
            "execute",
            "run",
            "reason",
            "calculate",
        ]

        for method_name in method_names:
            method = getattr(cls, method_name, None)
            if method is None:
                continue

            try:
                hints = inspect.get_annotations(method, eval_str=False)
            except Exception:
                continue

            # Extract the first non-self parameter type
            params = list(inspect.signature(method).parameters.values())
            for param in params:
                if param.name in ("self", "cls"):
                    continue
                hint = hints.get(param.name)
                if hint is not None:
                    input_hint = str(hint)
                    break

            # Extract return type
            return_hint = hints.get("return")
            if return_hint is not None:
                output_hint = str(return_hint)

            break  # Use first matching method

        return input_hint, output_hint

    def _should_skip_module(self, module_path: str) -> bool:
        """Determine if a module should be skipped during scanning.

        Args:
            module_path: Dotted module path.

        Returns:
            True if the module should be skipped.
        """
        for pattern in _SKIP_MODULE_PATTERNS:
            if module_path == pattern or module_path.startswith(f"{pattern}."):
                return True
        return False


__all__ = [
    "DiscoveredAgent",
    "LegacyAgentDiscovery",
]
