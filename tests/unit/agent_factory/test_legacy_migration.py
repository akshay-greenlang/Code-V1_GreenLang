# -*- coding: utf-8 -*-
"""
Unit tests for Legacy Agent Discovery, Registration, and Pack Generation
(INFRA-010 iteration).

Tests the VersionMigrationFramework for migration path finding, script
execution (up/down), dry-run mode, and history tracking. Also validates
pack format loading, template generation, and agent discovery patterns
used during legacy migration.

Coverage target: 85%+ of:
  - greenlang.infrastructure.agent_factory.versioning.migration
  - greenlang.infrastructure.agent_factory.packaging.pack_format
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from greenlang.infrastructure.agent_factory.versioning.migration import (
    MigrationHistoryEntry,
    MigrationResult,
    MigrationScript,
    MigrationStep,
    VersionMigrationFramework,
)
from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentDependency,
    AgentMetadata,
    AgentPack,
    AgentType,
    InputOutputSchema,
    PackFormat,
    PythonDependency,
    ResourceSpec,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def migration_framework() -> VersionMigrationFramework:
    """Create a fresh migration framework."""
    return VersionMigrationFramework()


@pytest.fixture
def simple_up_fn():
    """A simple synchronous upgrade migration function."""
    def _up(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["migrated"] = True
        ctx.setdefault("version_trail", []).append("up")
        return ctx
    return _up


@pytest.fixture
def simple_down_fn():
    """A simple synchronous downgrade migration function."""
    def _down(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["migrated"] = False
        ctx.setdefault("version_trail", []).append("down")
        return ctx
    return _down


@pytest.fixture
def async_up_fn():
    """An async upgrade migration function."""
    async def _up(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["async_migrated"] = True
        return ctx
    return _up


@pytest.fixture
def chain_scripts(simple_up_fn, simple_down_fn) -> List[MigrationScript]:
    """Create a chain of migration scripts: 1.0.0 -> 1.1.0 -> 1.2.0 -> 2.0.0."""
    return [
        MigrationScript(
            from_version="1.0.0",
            to_version="1.1.0",
            up_fn=simple_up_fn,
            down_fn=simple_down_fn,
            description="Add carbon_intensity field",
        ),
        MigrationScript(
            from_version="1.1.0",
            to_version="1.2.0",
            up_fn=simple_up_fn,
            down_fn=simple_down_fn,
            description="Rename scope_1 to scope_one",
        ),
        MigrationScript(
            from_version="1.2.0",
            to_version="2.0.0",
            up_fn=simple_up_fn,
            down_fn=simple_down_fn,
            description="Major schema overhaul",
        ),
    ]


@pytest.fixture
def sample_pack_yaml(tmp_path: Path) -> Path:
    """Create a sample agent.pack.yaml file on disk."""
    pack_data = {
        "name": "emissions-calc",
        "version": "1.2.0",
        "description": "Calculates Scope 1-3 emissions",
        "agent_type": "deterministic",
        "entry_point": "greenlang.agents.emissions_calc.agent",
        "base_class": "greenlang.agents.base.BaseAgent",
        "dependencies": {
            "agents": [
                {"name": "factor-lookup", "version_constraint": "^1.0.0"},
            ],
            "python": [
                {"package": "pandas", "version_constraint": ">=1.5.0"},
            ],
        },
        "resources": {
            "cpu_limit": "1",
            "memory_limit": "1Gi",
            "timeout_seconds": 600,
        },
        "metadata": {
            "author": "GreenLang Team",
            "license": "Proprietary",
            "tags": ["emissions", "scope1", "scope2"],
            "regulatory": ["GHG Protocol", "ISO 14064"],
        },
    }
    filepath = tmp_path / "agent.pack.yaml"
    with open(filepath, "w") as f:
        yaml.dump(pack_data, f, default_flow_style=False)
    return filepath


@pytest.fixture
def sample_reasoning_pack_yaml(tmp_path: Path) -> Path:
    """Create a reasoning agent pack.yaml file on disk."""
    pack_data = {
        "name": "entity-resolver",
        "version": "0.5.0",
        "description": "LLM-based entity resolution for supplier matching",
        "agent_type": "reasoning",
        "entry_point": "greenlang.agents.entity_resolver.agent",
        "base_class": "greenlang.agents.base.ReasoningAgent",
        "dependencies": {"agents": [], "python": []},
        "metadata": {
            "author": "GreenLang Team",
            "tags": ["reasoning", "entity-resolution"],
        },
    }
    filepath = tmp_path / "reasoning.pack.yaml"
    with open(filepath, "w") as f:
        yaml.dump(pack_data, f, default_flow_style=False)
    return filepath


@pytest.fixture
def sample_insight_pack_yaml(tmp_path: Path) -> Path:
    """Create an insight agent pack.yaml file on disk."""
    pack_data = {
        "name": "narrative-gen",
        "version": "0.3.0",
        "description": "Generates executive narrative summaries",
        "agent_type": "insight",
        "entry_point": "greenlang.agents.narrative_gen.agent",
        "dependencies": {"agents": [], "python": []},
        "metadata": {"tags": ["insight", "narrative"]},
    }
    filepath = tmp_path / "insight.pack.yaml"
    with open(filepath, "w") as f:
        yaml.dump(pack_data, f, default_flow_style=False)
    return filepath


# ============================================================================
# TestLegacyAgentDiscovery
# ============================================================================


class TestLegacyAgentDiscovery:
    """Tests for legacy agent discovery via pack.yaml loading."""

    def test_discover_deterministic_agent(self, sample_pack_yaml: Path) -> None:
        """Deterministic agent is discovered and loaded from pack.yaml."""
        pack = PackFormat.load(sample_pack_yaml)
        assert pack.name == "emissions-calc"
        assert pack.agent_type == AgentType.DETERMINISTIC
        assert pack.version == "1.2.0"

    def test_discover_reasoning_agent(self, sample_reasoning_pack_yaml: Path) -> None:
        """Reasoning agent is discovered with correct type classification."""
        pack = PackFormat.load(sample_reasoning_pack_yaml)
        assert pack.name == "entity-resolver"
        assert pack.agent_type == AgentType.REASONING
        assert "ReasoningAgent" in pack.base_class

    def test_discover_insight_agent(self, sample_insight_pack_yaml: Path) -> None:
        """Insight agent is discovered with correct type classification."""
        pack = PackFormat.load(sample_insight_pack_yaml)
        assert pack.name == "narrative-gen"
        assert pack.agent_type == AgentType.INSIGHT

    def test_derive_agent_key_from_module(self) -> None:
        """Agent key is derived from the pack name field."""
        template = PackFormat.generate_template("cbam-intake-agent")
        assert template.name == "cbam-intake-agent"
        assert "cbam_intake_agent" in template.entry_point

    def test_skip_agent_with_missing_pack_yaml(self, tmp_path: Path) -> None:
        """Loading a non-existent pack.yaml raises FileNotFoundError."""
        missing = tmp_path / "nonexistent" / "agent.pack.yaml"
        with pytest.raises(FileNotFoundError):
            PackFormat.load(missing)

    def test_detect_base_class(self, sample_pack_yaml: Path) -> None:
        """Pack file correctly captures the base class."""
        pack = PackFormat.load(sample_pack_yaml)
        assert pack.base_class == "greenlang.agents.base.BaseAgent"

    def test_handle_import_error_gracefully(self, tmp_path: Path) -> None:
        """Invalid YAML content is handled gracefully."""
        bad_file = tmp_path / "bad.pack.yaml"
        with open(bad_file, "w") as f:
            f.write("- this is a list not a mapping")

        errors = PackFormat.validate(bad_file)
        assert len(errors) > 0
        assert "must contain a YAML mapping" in errors[0].lower() or "validation failed" in errors[0].lower()

    def test_discover_all_returns_list_via_search(self, tmp_path: Path) -> None:
        """Multiple pack.yaml files can be loaded as a list."""
        packs = []
        for name in ["agent-a", "agent-b", "agent-c"]:
            pack_data = {
                "name": name,
                "version": "0.1.0",
                "entry_point": f"greenlang.agents.{name.replace('-', '_')}.agent",
                "dependencies": {"agents": [], "python": []},
            }
            filepath = tmp_path / name / "agent.pack.yaml"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                yaml.dump(pack_data, f)
            packs.append(PackFormat.load(filepath))

        assert len(packs) == 3
        names = [p.name for p in packs]
        assert "agent-a" in names
        assert "agent-b" in names
        assert "agent-c" in names


# ============================================================================
# TestLegacyRegistrar (via VersionMigrationFramework)
# ============================================================================


class TestLegacyRegistrar:
    """Tests for legacy agent registration via the migration framework."""

    def test_register_single_agent(
        self, migration_framework: VersionMigrationFramework, simple_up_fn
    ) -> None:
        """Registering a single migration script for an agent succeeds."""
        script = MigrationScript(
            from_version="0.1.0",
            to_version="1.0.0",
            up_fn=simple_up_fn,
            description="Initial legacy registration",
        )
        migration_framework.register("emissions-calc", script)
        scripts = migration_framework.list_migrations("emissions-calc")
        assert len(scripts) == 1
        assert scripts[0].from_version == "0.1.0"

    def test_register_idempotent(
        self, migration_framework: VersionMigrationFramework, simple_up_fn
    ) -> None:
        """Registering the same script twice adds it twice (append semantics)."""
        script = MigrationScript(
            from_version="0.1.0",
            to_version="1.0.0",
            up_fn=simple_up_fn,
        )
        migration_framework.register("agent-a", script)
        migration_framework.register("agent-a", script)
        scripts = migration_framework.list_migrations("agent-a")
        assert len(scripts) == 2

    @pytest.mark.asyncio
    async def test_register_all_generates_report(
        self, migration_framework: VersionMigrationFramework, chain_scripts
    ) -> None:
        """Running a full migration chain produces a complete report."""
        for script in chain_scripts:
            migration_framework.register("test-agent", script)

        result = await migration_framework.migrate(
            "test-agent", "1.0.0", "2.0.0"
        )
        assert result.success is True
        assert len(result.steps_executed) == 3
        assert result.from_version == "1.0.0"
        assert result.to_version == "2.0.0"

    def test_skip_already_registered_agent(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """list_migrations for unregistered agent returns empty list."""
        scripts = migration_framework.list_migrations("nonexistent-agent")
        assert scripts == []

    def test_synthetic_version_is_0_1_0(self) -> None:
        """Template generation uses 0.1.0 as default synthetic version."""
        template = PackFormat.generate_template("legacy-agent")
        assert template.version == "0.1.0"

    def test_legacy_metadata_flag(self) -> None:
        """Generated template includes appropriate metadata."""
        template = PackFormat.generate_template(
            "legacy-agent", agent_type=AgentType.DETERMINISTIC
        )
        assert template.metadata.author == "GreenLang Platform Team"
        assert "legacy-agent" in template.metadata.tags
        assert "deterministic" in template.metadata.tags

    @pytest.mark.asyncio
    async def test_migration_history_recorded(
        self, migration_framework: VersionMigrationFramework, simple_up_fn
    ) -> None:
        """Each migration run is recorded in the history."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0", to_version="1.1.0", up_fn=simple_up_fn,
        ))

        await migration_framework.migrate("agent-a", "1.0.0", "1.1.0")
        history = migration_framework.get_history("agent-a")
        assert len(history) == 1
        assert history[0].agent_key == "agent-a"
        assert history[0].success is True
        assert history[0].direction == "up"

    @pytest.mark.asyncio
    async def test_migration_history_filter_by_agent(
        self, migration_framework: VersionMigrationFramework, simple_up_fn
    ) -> None:
        """History can be filtered by agent key."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0", to_version="1.1.0", up_fn=simple_up_fn,
        ))
        migration_framework.register("agent-b", MigrationScript(
            from_version="1.0.0", to_version="1.1.0", up_fn=simple_up_fn,
        ))

        await migration_framework.migrate("agent-a", "1.0.0", "1.1.0")
        await migration_framework.migrate("agent-b", "1.0.0", "1.1.0")

        history_a = migration_framework.get_history("agent-a")
        assert len(history_a) == 1
        all_history = migration_framework.get_history()
        assert len(all_history) == 2


# ============================================================================
# TestLegacyPackGenerator
# ============================================================================


class TestLegacyPackGenerator:
    """Tests for pack.yaml generation and validation."""

    def test_generate_pack_yaml_deterministic(self) -> None:
        """Generating a deterministic pack template produces valid AgentPack."""
        pack = PackFormat.generate_template(
            "emissions-calc",
            agent_type=AgentType.DETERMINISTIC,
            version="1.0.0",
        )
        assert pack.name == "emissions-calc"
        assert pack.version == "1.0.0"
        assert pack.agent_type == AgentType.DETERMINISTIC
        assert pack.entry_point == "greenlang.agents.emissions_calc.agent"
        assert pack.spec_version == "1.0"

    def test_generate_pack_yaml_reasoning(self) -> None:
        """Generating a reasoning pack template sets correct type."""
        pack = PackFormat.generate_template(
            "entity-resolver",
            agent_type=AgentType.REASONING,
        )
        assert pack.agent_type == AgentType.REASONING
        assert "entity_resolver" in pack.entry_point

    def test_generate_pack_yaml_insight(self) -> None:
        """Generating an insight pack template sets correct type."""
        pack = PackFormat.generate_template(
            "narrative-gen",
            agent_type=AgentType.INSIGHT,
        )
        assert pack.agent_type == AgentType.INSIGHT

    def test_extract_input_output_types(self, sample_pack_yaml: Path) -> None:
        """Pack file captures input/output schema definitions."""
        pack = PackFormat.load(sample_pack_yaml)
        assert isinstance(pack.inputs, InputOutputSchema)
        assert isinstance(pack.outputs, InputOutputSchema)
        # Defaults when not specified
        assert pack.inputs.schema_type == "json_schema"

    def test_write_pack_yaml_to_disk(self, tmp_path: Path) -> None:
        """PackFormat.save writes valid YAML to the filesystem."""
        pack = PackFormat.generate_template("test-agent", version="1.0.0")
        output_path = tmp_path / "test-agent" / "agent.pack.yaml"

        PackFormat.save(pack, output_path)

        assert output_path.exists()
        # Reload and verify
        reloaded = PackFormat.load(output_path)
        assert reloaded.name == "test-agent"
        assert reloaded.version == "1.0.0"

    def test_pack_yaml_valid_format(self, sample_pack_yaml: Path) -> None:
        """Loaded pack passes validation with no errors."""
        errors = PackFormat.validate(sample_pack_yaml)
        assert errors == []

    def test_pack_yaml_invalid_name(self, tmp_path: Path) -> None:
        """Pack with invalid agent name fails validation."""
        pack_data = {
            "name": "INVALID-NAME",  # Uppercase not allowed
            "version": "1.0.0",
            "entry_point": "module.agent",
            "dependencies": {"agents": [], "python": []},
        }
        filepath = tmp_path / "invalid.pack.yaml"
        with open(filepath, "w") as f:
            yaml.dump(pack_data, f)

        errors = PackFormat.validate(filepath)
        assert len(errors) > 0

    def test_pack_yaml_invalid_version(self, tmp_path: Path) -> None:
        """Pack with invalid semver fails validation."""
        pack_data = {
            "name": "valid-agent",
            "version": "not-a-version",
            "entry_point": "module.agent",
            "dependencies": {"agents": [], "python": []},
        }
        filepath = tmp_path / "badver.pack.yaml"
        with open(filepath, "w") as f:
            yaml.dump(pack_data, f)

        errors = PackFormat.validate(filepath)
        assert len(errors) > 0

    def test_pack_yaml_dependencies_parsed(self, sample_pack_yaml: Path) -> None:
        """Agent and Python dependencies are parsed into typed objects."""
        pack = PackFormat.load(sample_pack_yaml)
        agent_deps = pack.agent_dependencies
        python_deps = pack.python_dependencies

        assert len(agent_deps) == 1
        assert isinstance(agent_deps[0], AgentDependency)
        assert agent_deps[0].name == "factor-lookup"
        assert agent_deps[0].version_constraint == "^1.0.0"

        assert len(python_deps) == 1
        assert isinstance(python_deps[0], PythonDependency)
        assert python_deps[0].package == "pandas"

    def test_pack_yaml_resource_spec(self, sample_pack_yaml: Path) -> None:
        """Resource limits are correctly parsed."""
        pack = PackFormat.load(sample_pack_yaml)
        assert isinstance(pack.resources, ResourceSpec)
        assert pack.resources.cpu_limit == "1"
        assert pack.resources.memory_limit == "1Gi"
        assert pack.resources.timeout_seconds == 600

    def test_pack_yaml_to_yaml_dict(self, sample_pack_yaml: Path) -> None:
        """to_yaml_dict produces a serializable dict."""
        pack = PackFormat.load(sample_pack_yaml)
        data = pack.to_yaml_dict()

        assert data["name"] == "emissions-calc"
        assert isinstance(data["dependencies"]["agents"], list)
        assert isinstance(data["dependencies"]["python"], list)
        # Dependencies should be plain dicts
        if data["dependencies"]["agents"]:
            assert isinstance(data["dependencies"]["agents"][0], dict)

    def test_pack_yaml_metadata(self, sample_pack_yaml: Path) -> None:
        """Metadata fields (author, tags, regulatory) are preserved."""
        pack = PackFormat.load(sample_pack_yaml)
        assert pack.metadata.author == "GreenLang Team"
        assert "emissions" in pack.metadata.tags
        assert "GHG Protocol" in pack.metadata.regulatory


# ============================================================================
# TestMigrationExecution
# ============================================================================


class TestMigrationExecution:
    """Tests for the VersionMigrationFramework migration execution engine."""

    @pytest.mark.asyncio
    async def test_upgrade_single_step(
        self, migration_framework: VersionMigrationFramework, simple_up_fn, simple_down_fn
    ) -> None:
        """Single-step upgrade executes the up function."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0",
            to_version="1.1.0",
            up_fn=simple_up_fn,
            down_fn=simple_down_fn,
            description="Add field",
        ))

        result = await migration_framework.migrate("agent-a", "1.0.0", "1.1.0")
        assert result.success is True
        assert len(result.steps_executed) == 1
        assert result.steps_executed[0].direction == "up"
        assert result.steps_executed[0].success is True
        assert result.steps_executed[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_downgrade_single_step(
        self, migration_framework: VersionMigrationFramework, simple_up_fn, simple_down_fn
    ) -> None:
        """Single-step downgrade executes the down function."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0",
            to_version="1.1.0",
            up_fn=simple_up_fn,
            down_fn=simple_down_fn,
        ))

        result = await migration_framework.migrate("agent-a", "1.1.0", "1.0.0")
        assert result.success is True
        assert len(result.steps_executed) == 1
        assert result.steps_executed[0].direction == "down"

    @pytest.mark.asyncio
    async def test_multi_step_upgrade(
        self, migration_framework: VersionMigrationFramework, chain_scripts
    ) -> None:
        """Multi-step upgrade traverses the full migration chain."""
        for script in chain_scripts:
            migration_framework.register("test-agent", script)

        result = await migration_framework.migrate("test-agent", "1.0.0", "2.0.0")
        assert result.success is True
        assert len(result.steps_executed) == 3
        versions = [(s.from_version, s.to_version) for s in result.steps_executed]
        assert versions == [("1.0.0", "1.1.0"), ("1.1.0", "1.2.0"), ("1.2.0", "2.0.0")]

    @pytest.mark.asyncio
    async def test_dry_run_does_not_execute(
        self, migration_framework: VersionMigrationFramework, chain_scripts
    ) -> None:
        """Dry run validates the path without executing migration functions."""
        for script in chain_scripts:
            migration_framework.register("test-agent", script)

        result = await migration_framework.migrate(
            "test-agent", "1.0.0", "2.0.0", dry_run=True
        )
        assert result.success is True
        assert result.dry_run is True
        assert len(result.steps_executed) == 3
        # All steps should have 0 duration since they were not executed
        for step in result.steps_executed:
            assert step.duration_ms == 0

    @pytest.mark.asyncio
    async def test_no_migration_path_returns_failure(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """Missing migration path returns a failure result with error message."""
        result = await migration_framework.migrate(
            "nonexistent-agent", "1.0.0", "2.0.0"
        )
        assert result.success is False
        assert len(result.errors) > 0
        assert "no migration path" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_migration_failure_stops_execution(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """A failing migration step stops the chain and reports the error."""
        def failing_fn(ctx):
            raise RuntimeError("Database connection lost")

        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0", to_version="1.1.0",
            up_fn=lambda ctx: ctx,
        ))
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.1.0", to_version="1.2.0",
            up_fn=failing_fn,
        ))
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.2.0", to_version="2.0.0",
            up_fn=lambda ctx: ctx,
        ))

        result = await migration_framework.migrate("agent-a", "1.0.0", "2.0.0")
        assert result.success is False
        assert len(result.steps_executed) == 2  # First succeeded, second failed
        assert result.steps_executed[0].success is True
        assert result.steps_executed[1].success is False
        assert "Database connection lost" in result.steps_executed[1].error

    @pytest.mark.asyncio
    async def test_missing_down_fn_fails_downgrade(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """Downgrade fails when the migration script has no down function."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0",
            to_version="1.1.0",
            up_fn=lambda ctx: ctx,
            down_fn=None,  # No downgrade function
        ))

        result = await migration_framework.migrate("agent-a", "1.1.0", "1.0.0")
        assert result.success is False
        assert any("no down function" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_async_migration_function(
        self, migration_framework: VersionMigrationFramework, async_up_fn
    ) -> None:
        """Async migration functions are properly awaited."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0",
            to_version="1.1.0",
            up_fn=async_up_fn,
        ))

        result = await migration_framework.migrate(
            "agent-a", "1.0.0", "1.1.0", context={}
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_context_passed_through_chain(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """Context dict is passed through each migration step in sequence."""
        def step1(ctx):
            ctx["step1"] = True
            return ctx

        def step2(ctx):
            ctx["step2"] = True
            assert ctx.get("step1") is True  # Step 1 already ran
            return ctx

        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0", to_version="1.1.0", up_fn=step1,
        ))
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.1.0", to_version="1.2.0", up_fn=step2,
        ))

        result = await migration_framework.migrate(
            "agent-a", "1.0.0", "1.2.0", context={}
        )
        assert result.success is True
        assert len(result.steps_executed) == 2

    @pytest.mark.asyncio
    async def test_migration_script_sorted_by_version(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """Scripts are sorted by from_version regardless of registration order."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.2.0", to_version="2.0.0", up_fn=lambda ctx: ctx,
        ))
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0", to_version="1.1.0", up_fn=lambda ctx: ctx,
        ))
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.1.0", to_version="1.2.0", up_fn=lambda ctx: ctx,
        ))

        scripts = migration_framework.list_migrations("agent-a")
        versions = [s.from_version for s in scripts]
        assert versions == ["1.0.0", "1.1.0", "1.2.0"]

    @pytest.mark.asyncio
    async def test_failed_migration_recorded_in_history(
        self, migration_framework: VersionMigrationFramework
    ) -> None:
        """Failed migrations are still recorded in history."""
        migration_framework.register("agent-a", MigrationScript(
            from_version="1.0.0",
            to_version="1.1.0",
            up_fn=lambda ctx: (_ for _ in ()).throw(ValueError("bad data")),
        ))

        result = await migration_framework.migrate("agent-a", "1.0.0", "1.1.0")
        assert result.success is False

        history = migration_framework.get_history("agent-a")
        assert len(history) == 1
        assert history[0].success is False
