"""
Legacy Pack Generator - Agent Factory (INFRA-010)

Generates ``agent.pack.yaml`` files from existing legacy agent code,
enabling them to be managed by the Agent Factory packaging system
without requiring manual YAML authoring.

The generator inspects each DiscoveredAgent's class to extract:
  - Input/output type hints for schema stubs.
  - Base class to determine ``agent_type``.
  - Module-level imports to suggest Python dependencies.
  - Existing resource hints from class-level attributes.

Classes:
    - LegacyPackGenerator: Generates and writes pack YAML for legacy agents.

Example:
    >>> generator = LegacyPackGenerator()
    >>> yaml_content = generator.generate(agent)
    >>> path = generator.write_pack_yaml(agent)
    >>> print(f"Wrote pack file to {path}")
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import yaml

from greenlang.infrastructure.agent_factory.legacy.discovery import (
    DiscoveredAgent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPEC_VERSION = "1.0"
_DEFAULT_VERSION = "0.1.0"
_DEFAULT_CPU = "500m"
_DEFAULT_MEMORY = "512Mi"
_DEFAULT_TIMEOUT_S = 300

# Map recognised base class names to pack ``agent_type`` values
_BASE_CLASS_TO_TYPE: Dict[str, str] = {
    "DeterministicAgent": "deterministic",
    "ReasoningAgent": "reasoning",
    "InsightAgent": "insight",
    "AgentSpecV2Base": "deterministic",
}

# Well-known third-party packages that should be listed as Python deps
# if they appear in the module's imports.
_KNOWN_THIRD_PARTY: Set[str] = frozenset({
    "numpy",
    "pandas",
    "scipy",
    "pydantic",
    "httpx",
    "aiohttp",
    "requests",
    "sqlalchemy",
    "psycopg",
    "asyncpg",
    "redis",
    "yaml",
    "pyyaml",
    "openai",
    "anthropic",
    "sklearn",
    "scikit-learn",
    "torch",
    "transformers",
    "statsmodels",
    "xgboost",
    "lightgbm",
    "openpyxl",
    "xlsxwriter",
    "jinja2",
    "celery",
})

# Packages that are part of the standard library and should be skipped
_STDLIB_PREFIXES: Set[str] = frozenset({
    "os",
    "sys",
    "re",
    "json",
    "math",
    "time",
    "datetime",
    "pathlib",
    "hashlib",
    "logging",
    "typing",
    "abc",
    "enum",
    "dataclasses",
    "collections",
    "functools",
    "itertools",
    "io",
    "tempfile",
    "shutil",
    "copy",
    "uuid",
    "asyncio",
    "inspect",
    "importlib",
    "contextvars",
    "warnings",
    "traceback",
    "unittest",
    "csv",
    "configparser",
    "struct",
    "ctypes",
    "threading",
    "multiprocessing",
    "concurrent",
    "statistics",
    "decimal",
    "fractions",
})


# ---------------------------------------------------------------------------
# LegacyPackGenerator
# ---------------------------------------------------------------------------


class LegacyPackGenerator:
    """Generates ``agent.pack.yaml`` from existing agent module code.

    Inspects the agent class and its module to synthesise a valid pack
    manifest without requiring the developer to write YAML by hand.

    Example:
        >>> gen = LegacyPackGenerator()
        >>> content = gen.generate(discovered_agent)
        >>> path = gen.write_pack_yaml(discovered_agent, output_dir=Path("/tmp"))
    """

    def __init__(
        self,
        *,
        default_author: str = "GreenLang Platform Team",
        default_license: str = "Proprietary",
    ) -> None:
        """Initialize the pack generator.

        Args:
            default_author: Author string to embed in generated packs.
            default_license: License identifier for the metadata block.
        """
        self._default_author = default_author
        self._default_license = default_license

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, agent: DiscoveredAgent) -> str:
        """Generate pack.yaml content for a legacy agent.

        Args:
            agent: A DiscoveredAgent from the legacy discovery scanner.

        Returns:
            YAML string ready to be written to ``agent.pack.yaml``.
        """
        start = time.perf_counter()

        agent_type = _BASE_CLASS_TO_TYPE.get(agent.base_class, "deterministic")
        entry_point = f"{agent.module_path}.{agent.class_name}"

        # Try to load the module to extract richer information
        cls = self._load_class(agent)
        python_deps = self._detect_python_deps(agent)
        input_schema = self._build_input_schema(agent, cls)
        output_schema = self._build_output_schema(agent, cls)
        resources = self._detect_resource_hints(cls)
        tags = self._derive_tags(agent, agent_type)

        pack_dict: Dict[str, Any] = {
            "name": agent.agent_key,
            "version": _DEFAULT_VERSION,
            "spec_version": _SPEC_VERSION,
            "description": self._derive_description(agent, cls),
            "agent_type": agent_type,
            "entry_point": entry_point,
            "base_class": f"greenlang.agents.base_agents.{agent.base_class}",
            "dependencies": {
                "agents": [],
                "python": python_deps,
            },
            "inputs": input_schema,
            "outputs": output_schema,
            "resources": resources,
            "metadata": {
                "author": self._default_author,
                "license": self._default_license,
                "tags": tags,
                "regulatory": self._detect_regulatory_tags(agent, cls),
                "legacy": True,
                "auto_generated": True,
            },
        }

        yaml_content = yaml.dump(
            pack_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "LegacyPackGenerator: generated pack for %s in %.1fms",
            agent.agent_key,
            elapsed_ms,
        )

        return yaml_content

    def write_pack_yaml(
        self,
        agent: DiscoveredAgent,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Write agent.pack.yaml to the agent's directory.

        Args:
            agent: A DiscoveredAgent from the legacy discovery scanner.
            output_dir: Override directory to write the file into.
                Defaults to the parent directory of the agent's source
                file.

        Returns:
            Path to the written ``agent.pack.yaml`` file.
        """
        content = self.generate(agent)
        target_dir = output_dir or agent.file_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "agent.pack.yaml"

        target_path.write_text(content, encoding="utf-8")

        logger.info(
            "LegacyPackGenerator: wrote %s (%d bytes)",
            target_path,
            len(content),
        )

        return target_path

    def generate_batch(
        self,
        agents: List[DiscoveredAgent],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Generate pack YAML files for a batch of agents.

        Args:
            agents: List of DiscoveredAgent instances.
            output_dir: Optional base directory.  Each agent gets a
                subdirectory named after its agent_key.

        Returns:
            Mapping of agent_key to the path of the written file.
        """
        results: Dict[str, Path] = {}

        for agent in agents:
            try:
                if output_dir is not None:
                    agent_dir = output_dir / agent.agent_key
                else:
                    agent_dir = None
                path = self.write_pack_yaml(agent, output_dir=agent_dir)
                results[agent.agent_key] = path
            except Exception as exc:
                logger.error(
                    "LegacyPackGenerator: failed for %s: %s",
                    agent.agent_key,
                    exc,
                )

        logger.info(
            "LegacyPackGenerator: batch complete (%d/%d succeeded)",
            len(results),
            len(agents),
        )
        return results

    # ------------------------------------------------------------------
    # Internal: class loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_class(agent: DiscoveredAgent) -> Optional[Type]:
        """Attempt to import and return the agent class.

        Args:
            agent: The discovered agent descriptor.

        Returns:
            The class object, or None if import fails.
        """
        try:
            mod = importlib.import_module(agent.module_path)
            return getattr(mod, agent.class_name, None)
        except Exception as exc:
            logger.debug(
                "LegacyPackGenerator: could not load %s.%s: %s",
                agent.module_path,
                agent.class_name,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Internal: dependency detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_python_deps(agent: DiscoveredAgent) -> List[Dict[str, str]]:
        """Detect third-party Python dependencies by scanning imports.

        Args:
            agent: The discovered agent.

        Returns:
            List of dicts with ``package`` and ``version_constraint``.
        """
        deps: List[Dict[str, str]] = []
        seen: Set[str] = set()

        try:
            source = agent.file_path.read_text(encoding="utf-8")
        except Exception:
            return deps

        # Match ``import foo`` and ``from foo import ...``
        import_pattern = re.compile(
            r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            re.MULTILINE,
        )

        for match in import_pattern.finditer(source):
            top_level = match.group(1).lower()
            if top_level in _STDLIB_PREFIXES:
                continue
            if top_level.startswith("greenlang"):
                continue
            if top_level in seen:
                continue

            # Check if this is a known third-party package
            canonical = top_level
            if canonical == "yaml":
                canonical = "pyyaml"
            if canonical == "sklearn":
                canonical = "scikit-learn"

            if canonical in _KNOWN_THIRD_PARTY or top_level in _KNOWN_THIRD_PARTY:
                seen.add(top_level)
                deps.append({
                    "package": canonical,
                    "version_constraint": "*",
                })

        return deps

    # ------------------------------------------------------------------
    # Internal: schema extraction
    # ------------------------------------------------------------------

    def _build_input_schema(
        self,
        agent: DiscoveredAgent,
        cls: Optional[Type],
    ) -> Dict[str, Any]:
        """Build an input schema stub from type hints.

        Args:
            agent: The discovered agent.
            cls: The loaded class, or None.

        Returns:
            Schema dict with ``schema_type`` and ``fields``.
        """
        fields: Dict[str, Any] = {}

        if agent.input_type_hint:
            fields["_raw_hint"] = {
                "type": "object",
                "description": f"Auto-detected type: {agent.input_type_hint}",
            }

        if cls is not None:
            fields.update(self._extract_fields_from_method(cls))

        return {
            "schema_type": "json_schema",
            "fields": fields or {"input_data": {"type": "object", "description": "TODO: define input schema"}},
        }

    def _build_output_schema(
        self,
        agent: DiscoveredAgent,
        cls: Optional[Type],
    ) -> Dict[str, Any]:
        """Build an output schema stub from type hints.

        Args:
            agent: The discovered agent.
            cls: The loaded class, or None.

        Returns:
            Schema dict with ``schema_type`` and ``fields``.
        """
        fields: Dict[str, Any] = {}

        if agent.output_type_hint:
            fields["_raw_hint"] = {
                "type": "object",
                "description": f"Auto-detected type: {agent.output_type_hint}",
            }

        return {
            "schema_type": "json_schema",
            "fields": fields or {"result": {"type": "object", "description": "TODO: define output schema"}},
        }

    @staticmethod
    def _extract_fields_from_method(cls: Type) -> Dict[str, Any]:
        """Extract parameter names from the primary entry method.

        Args:
            cls: The agent class.

        Returns:
            Dict mapping parameter names to stub schema entries.
        """
        fields: Dict[str, Any] = {}
        for method_name in ("execute_impl", "execute", "run", "reason", "calculate"):
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            try:
                sig = inspect.signature(method)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "cls", "context", "session",
                                 "rag_engine", "tools", "temperature"):
                        continue
                    annotation = param.annotation
                    type_str = "object"
                    if annotation is not inspect.Parameter.empty:
                        type_str = str(annotation)
                    fields[pname] = {
                        "type": type_str,
                        "description": f"Parameter: {pname}",
                    }
                break
            except (ValueError, TypeError):
                continue
        return fields

    # ------------------------------------------------------------------
    # Internal: resource hints
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_resource_hints(cls: Optional[Type]) -> Dict[str, Any]:
        """Extract resource hints from class attributes if available.

        Args:
            cls: The agent class, or None.

        Returns:
            Resource spec dict with cpu_limit, memory_limit, timeout.
        """
        resources: Dict[str, Any] = {
            "cpu_limit": _DEFAULT_CPU,
            "memory_limit": _DEFAULT_MEMORY,
            "timeout_seconds": _DEFAULT_TIMEOUT_S,
        }

        if cls is None:
            return resources

        # Check for class-level resource attributes
        for attr in ("resource_cpu", "cpu_limit"):
            val = getattr(cls, attr, None)
            if val is not None:
                resources["cpu_limit"] = str(val)

        for attr in ("resource_memory", "memory_limit"):
            val = getattr(cls, attr, None)
            if val is not None:
                resources["memory_limit"] = str(val)

        for attr in ("timeout_seconds", "max_execution_time"):
            val = getattr(cls, attr, None)
            if val is not None:
                resources["timeout_seconds"] = int(val)

        return resources

    # ------------------------------------------------------------------
    # Internal: metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_description(
        agent: DiscoveredAgent,
        cls: Optional[Type],
    ) -> str:
        """Derive a human-readable description.

        Uses the class docstring if available, otherwise synthesises
        one from the agent key and base class.

        Args:
            agent: The discovered agent.
            cls: The loaded class, or None.

        Returns:
            Description string (max 2048 chars).
        """
        if cls is not None:
            docstring = inspect.getdoc(cls)
            if docstring:
                # Take the first paragraph only
                first_para = docstring.split("\n\n")[0].strip()
                if first_para:
                    return first_para[:2048]

        return (
            f"Legacy {agent.base_class} agent: {agent.class_name} "
            f"(auto-discovered from {agent.module_path})"
        )

    @staticmethod
    def _derive_tags(agent: DiscoveredAgent, agent_type: str) -> List[str]:
        """Derive searchable tags from the agent key and type.

        Args:
            agent: The discovered agent.
            agent_type: The agent classification.

        Returns:
            List of tag strings.
        """
        tags = ["legacy", agent_type]

        # Extract meaningful words from the agent key
        key_parts = agent.agent_key.replace("gl-", "").split("-")
        for part in key_parts:
            if len(part) > 2:
                tags.append(part)

        return list(dict.fromkeys(tags))  # Deduplicate while preserving order

    @staticmethod
    def _detect_regulatory_tags(
        agent: DiscoveredAgent,
        cls: Optional[Type],
    ) -> List[str]:
        """Detect regulatory framework references in the agent.

        Scans the module path and class docstring for known framework
        names.

        Args:
            agent: The discovered agent.
            cls: The loaded class, or None.

        Returns:
            List of detected regulatory framework names.
        """
        frameworks: List[str] = []
        search_text = agent.module_path.lower()

        if cls is not None:
            docstring = inspect.getdoc(cls) or ""
            search_text += " " + docstring.lower()

        known_frameworks = {
            "cbam": "CBAM",
            "csrd": "CSRD",
            "esrs": "ESRS",
            "ghg": "GHG Protocol",
            "sb253": "SB 253",
            "tcfd": "TCFD",
            "tnfd": "TNFD",
            "sfdr": "SFDR",
            "eu-taxonomy": "EU Taxonomy",
            "iso14064": "ISO 14064",
            "sec-climate": "SEC Climate",
        }

        for keyword, label in known_frameworks.items():
            if keyword in search_text:
                frameworks.append(label)

        return frameworks


__all__ = [
    "LegacyPackGenerator",
]
