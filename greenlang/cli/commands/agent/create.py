# -*- coding: utf-8 -*-
"""
gl agent create - Scaffold a new GreenLang agent from a template.

Generates a complete agent directory structure with agent.py, __init__.py,
agent.pack.yaml, input/output JSON schemas, and a starter test suite.
Supports three templates: deterministic (default), reasoning, and insight.

Example:
    gl agent create --name carbon-calc --template deterministic
    gl agent create --name eudr-checker --template reasoning --description "EUDR compliance"
    gl agent create --name climate-insight --spec spec.yaml --output-dir ./custom/

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]{1,62}[a-z0-9]$")


def _validate_agent_name(name: str) -> str:
    """Validate that the agent name is lowercase alphanumeric with hyphens."""
    if not _AGENT_NAME_RE.match(name):
        raise typer.BadParameter(
            "Agent name must be 3-64 chars, lowercase alphanumeric and "
            "hyphens only, start with a letter, end with letter/digit."
        )
    return name


# ---------------------------------------------------------------------------
# Template content generators
# ---------------------------------------------------------------------------

def _render_agent_py(name: str, template: str, description: str) -> str:
    """Render the main agent.py source file."""
    class_name = name.replace("-", "_").title().replace("_", "") + "Agent"
    module_name = name.replace("-", "_")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base_class = {
        "deterministic": "BaseAgent",
        "reasoning": "ReasoningAgent",
        "insight": "InsightAgent",
    }.get(template, "BaseAgent")

    process_body = {
        "deterministic": (
            '        # Zero-hallucination: all calculations are deterministic\n'
            '        validated = self._validate_input(input_data)\n'
            '        result = self._compute(validated)\n'
            '        provenance_hash = self._provenance_hash(input_data, result)\n'
            '        logger.info("Processing complete", extra={{"agent": self.agent_key}})\n'
            '        return {{"result": result, "provenance_hash": provenance_hash}}'
        ),
        "reasoning": (
            '        # LLM-assisted reasoning with deterministic validation\n'
            '        context = self._build_context(input_data)\n'
            '        reasoning_output = await self._reason(context)\n'
            '        validated = self._validate_reasoning(reasoning_output)\n'
            '        logger.info("Reasoning complete", extra={{"agent": self.agent_key}})\n'
            '        return validated'
        ),
        "insight": (
            '        # Insight generation with provenance tracking\n'
            '        data = self._prepare_data(input_data)\n'
            '        insights = self._generate_insights(data)\n'
            '        ranked = self._rank_insights(insights)\n'
            '        logger.info("Insights generated", extra={{"agent": self.agent_key}})\n'
            '        return ranked'
        ),
    }.get(template, "        pass")

    return f'''# -*- coding: utf-8 -*-
"""
{class_name} - {description}

Template: {template}
Created: {now_iso}
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {description}

    This agent follows the GreenLang {template} pattern.
    """

    agent_key: str = "{module_name}"
    version: str = "0.1.0"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize {class_name}.

        Args:
            config: Optional agent configuration dictionary.
        """
        self.config = config or {{}}
        logger.info("Initialized %s v%s", self.agent_key, self.version)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing entry point.

        Args:
            input_data: Validated input payload.

        Returns:
            Processing result dictionary.
        """
{process_body}

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data meets requirements."""
        if not data:
            raise ValueError("Input data must not be empty")
        return data

    def _compute(self, data: Dict[str, Any]) -> Any:
        """Core deterministic computation (zero-hallucination)."""
        return data

    def _provenance_hash(self, input_data: Any, output_data: Any) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = f"{{input_data}}|{{output_data}}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
'''


def _render_init_py(name: str, description: str) -> str:
    """Render the package __init__.py."""
    class_name = name.replace("-", "_").title().replace("_", "") + "Agent"
    module_name = name.replace("-", "_")
    return f'''# -*- coding: utf-8 -*-
"""
{description}
"""

from __future__ import annotations

__version__ = "0.1.0"

from {module_name}.agent import {class_name}

__all__ = ["{class_name}"]
'''


def _render_pack_yaml(name: str, template: str, description: str) -> str:
    """Render the agent.pack.yaml manifest."""
    module_name = name.replace("-", "_")
    return f'''# Agent Pack Manifest - GreenLang Agent Factory
# Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

id: "{module_name}"
name: "{name}"
version: "0.1.0"
description: "{description}"
type: "{template}"
license: "Apache-2.0"

runtime:
  python: ">=3.11"
  memory_mb: 256
  timeout_seconds: 30

entry_point: "agent.py"
class_name: "{name.replace("-", "_").title().replace("_", "")}Agent"

inputs:
  - name: "input_data"
    type: "object"
    required: true

outputs:
  - name: "result"
    type: "object"

dependencies:
  python: []
  agents: []

tests:
  golden: "tests/test_agent.py"
  coverage_threshold: 85
'''


def _render_input_schema() -> str:
    """Render the default input JSON schema."""
    return '''{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AgentInput",
  "type": "object",
  "properties": {
    "data": {
      "type": "object",
      "description": "Input data payload"
    },
    "options": {
      "type": "object",
      "description": "Processing options",
      "default": {}
    }
  },
  "required": ["data"]
}
'''


def _render_output_schema() -> str:
    """Render the default output JSON schema."""
    return '''{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AgentOutput",
  "type": "object",
  "properties": {
    "result": {
      "type": "object",
      "description": "Processing result"
    },
    "provenance_hash": {
      "type": "string",
      "description": "SHA-256 provenance hash"
    },
    "processing_time_ms": {
      "type": "number",
      "description": "Elapsed time in milliseconds"
    }
  },
  "required": ["result", "provenance_hash"]
}
'''


def _render_test_py(name: str, description: str) -> str:
    """Render the starter test module."""
    class_name = name.replace("-", "_").title().replace("_", "") + "Agent"
    module_name = name.replace("-", "_")
    return f'''# -*- coding: utf-8 -*-
"""Tests for {class_name}."""

from __future__ import annotations

import asyncio
import pytest

from {module_name}.agent import {class_name}


@pytest.fixture
def agent() -> {class_name}:
    """Create a default agent instance."""
    return {class_name}()


class TestInit:
    """Initialization tests."""

    def test_default_config(self, agent: {class_name}) -> None:
        assert agent.config == {{}}
        assert agent.agent_key == "{module_name}"
        assert agent.version == "0.1.0"

    def test_custom_config(self) -> None:
        cfg = {{"timeout": 60}}
        a = {class_name}(config=cfg)
        assert a.config["timeout"] == 60


class TestProcess:
    """Processing tests."""

    def test_process_returns_result(self, agent: {class_name}) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            agent.process({{"data": {{"value": 42}}}})
        )
        assert result is not None

    def test_process_empty_input_raises(self, agent: {class_name}) -> None:
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(
                agent.process({{}})
            )


class TestProvenance:
    """Provenance tracking tests."""

    def test_hash_deterministic(self, agent: {class_name}) -> None:
        h1 = agent._provenance_hash("a", "b")
        h2 = agent._provenance_hash("a", "b")
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_changes_with_input(self, agent: {class_name}) -> None:
        h1 = agent._provenance_hash("a", "b")
        h2 = agent._provenance_hash("c", "d")
        assert h1 != h2
'''


# ---------------------------------------------------------------------------
# Command implementation
# ---------------------------------------------------------------------------

def create(
    name: str = typer.Option(
        ...,
        "--name", "-n",
        help="Agent name (lowercase, alphanumeric + hyphens).",
        callback=_validate_agent_name,
    ),
    template: str = typer.Option(
        "deterministic",
        "--template", "-t",
        help="Agent template: deterministic, reasoning, or insight.",
    ),
    spec: Optional[Path] = typer.Option(
        None,
        "--spec", "-s",
        help="Path to AgentSpec YAML for additional metadata.",
    ),
    output_dir: Path = typer.Option(
        Path("greenlang/agents/"),
        "--output-dir", "-o",
        help="Parent directory for the new agent.",
    ),
    description: str = typer.Option(
        "",
        "--description", "-d",
        help="Short description for the agent.",
    ),
) -> None:
    """
    Scaffold a new GreenLang agent from a template.

    Creates the full directory layout, ready for implementation.

    Example:
        gl agent create --name carbon-calc --template deterministic
    """
    # Validate template choice
    valid_templates = ("deterministic", "reasoning", "insight")
    if template not in valid_templates:
        console.print(
            f"[red]Invalid template '{template}'. "
            f"Choose from: {', '.join(valid_templates)}[/red]"
        )
        raise typer.Exit(1)

    if not description:
        description = f"GreenLang {template} agent: {name}"

    # Load additional metadata from spec if provided
    spec_metadata: dict = {}
    if spec and spec.exists():
        try:
            import yaml  # type: ignore[import-untyped]

            with open(spec, "r", encoding="utf-8") as fh:
                spec_metadata = yaml.safe_load(fh) or {}
            description = spec_metadata.get("description", description)
            console.print(f"[cyan]Loaded spec from {spec}[/cyan]")
        except Exception as exc:
            console.print(f"[yellow]Warning: could not parse spec file: {exc}[/yellow]")

    agent_dir = output_dir / name
    module_name = name.replace("-", "_")

    console.print(Panel(
        f"[bold cyan]GreenLang Agent Scaffolder[/bold cyan]\n"
        f"Creating agent: [bold]{name}[/bold]  template: [bold]{template}[/bold]",
        border_style="cyan",
    ))

    # Check for existing directory
    if agent_dir.exists():
        overwrite = typer.confirm(f"Directory {agent_dir} already exists. Overwrite?")
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    # Build file map
    files: dict[str, str] = {
        "agent.py": _render_agent_py(name, template, description),
        "__init__.py": _render_init_py(name, description),
        "agent.pack.yaml": _render_pack_yaml(name, template, description),
        os.path.join("schemas", "input.json"): _render_input_schema(),
        os.path.join("schemas", "output.json"): _render_output_schema(),
        os.path.join("tests", "test_agent.py"): _render_test_py(name, description),
    }

    # Write files
    created_paths: list[str] = []
    for rel_path, content in files.items():
        full_path = agent_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        created_paths.append(str(full_path))
        logger.debug("Wrote %s", full_path)

    # Display results table
    table = Table(title="Created Files")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right", style="green")

    for p in created_paths:
        size = Path(p).stat().st_size
        display = str(Path(p).relative_to(Path.cwd())) if Path(p).is_relative_to(Path.cwd()) else p
        table.add_row(display, f"{size:,} B")

    console.print(table)
    console.print(
        f"\n[bold green]Agent '{name}' scaffolded at:[/bold green] {agent_dir}\n"
    )
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  1. Implement logic in {agent_dir / 'agent.py'}")
    console.print(f"  2. gl agent test --agent-dir {agent_dir}")
    console.print(f"  3. gl agent pack --agent-dir {agent_dir}")
    console.print(f"  4. gl agent deploy --agent-key {module_name} --env dev")
