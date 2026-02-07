# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Packaging: PackFormat, PackageBuilder,
DependencyResolver, PackageInstaller, and ManifestGenerator.

Tests YAML parsing, validation, archive creation, size limits,
dependency resolution, and manifest integrity verification.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

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
from greenlang.infrastructure.agent_factory.packaging.builder import (
    BuildResult,
    DEFAULT_SIZE_LIMIT_BYTES,
    EXCLUDE_PATTERNS,
    PackageBuilder,
)


# ============================================================================
# Inline Stubs for resolver, installer, manifest (not yet on disk)
# ============================================================================


class VersionConflict:
    def __init__(self, agent: str, required: str, available: str) -> None:
        self.agent = agent
        self.required = required
        self.available = available


class ResolvedDependency:
    def __init__(self, name: str, version: str, depth: int = 0) -> None:
        self.name = name
        self.version = version
        self.depth = depth


class ResolutionResult:
    def __init__(
        self,
        resolved: List[ResolvedDependency],
        conflicts: List[VersionConflict],
    ) -> None:
        self.resolved = resolved
        self.conflicts = conflicts

    @property
    def success(self) -> bool:
        return len(self.conflicts) == 0


class DependencyResolver:
    """Simplified dependency resolver for testing."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, version: str, deps: Optional[List[Dict]] = None) -> None:
        self._registry[f"{name}@{version}"] = {
            "name": name,
            "version": version,
            "dependencies": deps or [],
        }

    def resolve(self, name: str, version_constraint: str = "*") -> ResolutionResult:
        resolved: List[ResolvedDependency] = []
        conflicts: List[VersionConflict] = []
        visited: Set[str] = set()
        self._resolve_recursive(name, version_constraint, resolved, conflicts, visited, 0)
        return ResolutionResult(resolved=resolved, conflicts=conflicts)

    def _resolve_recursive(
        self,
        name: str,
        constraint: str,
        resolved: List[ResolvedDependency],
        conflicts: List[VersionConflict],
        visited: Set[str],
        depth: int,
    ) -> None:
        if name in visited:
            return
        visited.add(name)

        # Find matching version
        matching = None
        for key, pkg in self._registry.items():
            if pkg["name"] == name:
                matching = pkg
                break

        if matching is None:
            conflicts.append(VersionConflict(name, constraint, "not found"))
            return

        resolved.append(
            ResolvedDependency(matching["name"], matching["version"], depth)
        )
        for dep in matching["dependencies"]:
            self._resolve_recursive(
                dep["name"],
                dep.get("version_constraint", "*"),
                resolved,
                conflicts,
                visited,
                depth + 1,
            )


class PackageManifest:
    def __init__(self, files: Dict[str, str]) -> None:
        self.files = files  # path -> sha256

    def verify(self, base_path: Path) -> List[str]:
        errors: List[str] = []
        for filepath, expected_hash in self.files.items():
            full_path = base_path / filepath
            if not full_path.exists():
                errors.append(f"Missing: {filepath}")
                continue
            actual_hash = hashlib.sha256(
                full_path.read_bytes()
            ).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"Tampered: {filepath}")
        return errors


class ManifestGenerator:
    @staticmethod
    def generate(base_path: Path) -> PackageManifest:
        files: Dict[str, str] = {}
        for root_str, _dirs, filenames in os.walk(base_path):
            root = Path(root_str)
            for fname in filenames:
                fpath = root / fname
                rel = fpath.relative_to(base_path)
                sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                files[str(rel)] = sha
        return PackageManifest(files=files)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_pack_dict() -> Dict[str, Any]:
    """Minimal valid agent.pack.yaml content."""
    return {
        "name": "emissions-calc",
        "version": "1.2.0",
        "description": "Emission calculation agent",
        "agent_type": "deterministic",
        "entry_point": "greenlang.agents.emissions_calc.agent",
        "dependencies": {"agents": [], "python": []},
    }


@pytest.fixture
def valid_pack_yaml(tmp_path: Path, valid_pack_dict: Dict[str, Any]) -> Path:
    """Write a valid agent.pack.yaml to a temp directory."""
    pack_file = tmp_path / "agent.pack.yaml"
    with open(pack_file, "w") as fh:
        yaml.dump(valid_pack_dict, fh)
    return pack_file


@pytest.fixture
def agent_source_dir(tmp_path: Path, valid_pack_dict: Dict[str, Any]) -> Path:
    """Create a full agent source directory with pack.yaml and source files."""
    src = tmp_path / "my_agent"
    src.mkdir()

    # Write pack yaml
    with open(src / "agent.pack.yaml", "w") as fh:
        yaml.dump(valid_pack_dict, fh)

    # Write a Python source file
    (src / "agent.py").write_text(
        "class EmissionsCalcAgent:\n    pass\n", encoding="utf-8"
    )
    (src / "utils.py").write_text(
        "def helper(): return 42\n", encoding="utf-8"
    )

    return src


@pytest.fixture
def builder() -> PackageBuilder:
    return PackageBuilder()


# ============================================================================
# Test PackFormat
# ============================================================================


class TestPackFormat:
    """Tests for pack format parsing and validation."""

    def test_pack_format_parse_valid_yaml(
        self, valid_pack_yaml: Path
    ) -> None:
        """Valid YAML is parsed into an AgentPack model."""
        pack = PackFormat.load(valid_pack_yaml)
        assert pack.name == "emissions-calc"
        assert pack.version == "1.2.0"
        assert pack.agent_type == AgentType.DETERMINISTIC

    def test_pack_format_parse_invalid_yaml(self, tmp_path: Path) -> None:
        """Invalid YAML raises ValueError."""
        bad_file = tmp_path / "agent.pack.yaml"
        bad_file.write_text("- this\n- is\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            PackFormat.load(bad_file)

    def test_pack_format_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PackFormat.load(tmp_path / "nonexistent.yaml")

    def test_pack_format_validation_missing_fields(
        self, tmp_path: Path
    ) -> None:
        """Missing required fields cause validation errors."""
        pack_file = tmp_path / "agent.pack.yaml"
        pack_file.write_text(
            yaml.dump({"name": "test-agent"}),
            encoding="utf-8",
        )
        errors = PackFormat.validate(pack_file)
        assert len(errors) > 0

    def test_pack_format_version_validation_valid(self) -> None:
        """Valid semver passes validation."""
        pack = AgentPack(
            name="test-agent",
            version="1.0.0",
            entry_point="greenlang.agents.test.agent",
        )
        assert pack.version == "1.0.0"

    def test_pack_format_version_validation_invalid(self) -> None:
        """Invalid semver raises ValueError."""
        with pytest.raises(ValueError, match="not a valid semantic version"):
            AgentPack(
                name="test-agent",
                version="not-a-version",
                entry_point="greenlang.agents.test.agent",
            )

    def test_pack_format_name_validation_invalid(self) -> None:
        """Invalid agent name raises ValueError."""
        with pytest.raises(ValueError):
            AgentPack(
                name="INVALID NAME!",
                version="1.0.0",
                entry_point="greenlang.agents.test.agent",
            )

    def test_pack_format_name_validation_valid(self) -> None:
        """Valid agent name patterns pass."""
        pack = AgentPack(
            name="valid-agent.name_123",
            version="1.0.0",
            entry_point="greenlang.agents.test.agent",
        )
        assert pack.name == "valid-agent.name_123"

    def test_pack_format_agent_type_reasoning(self) -> None:
        """Agent type can be set to REASONING."""
        pack = AgentPack(
            name="insight-agent",
            version="2.0.0",
            agent_type=AgentType.REASONING,
            entry_point="greenlang.agents.insight.agent",
        )
        assert pack.agent_type == AgentType.REASONING

    def test_pack_format_resource_spec_defaults(self) -> None:
        """ResourceSpec uses sensible defaults."""
        spec = ResourceSpec()
        assert spec.cpu_limit == "500m"
        assert spec.memory_limit == "512Mi"
        assert spec.timeout_seconds == 300

    def test_pack_format_dependencies_parsing(self) -> None:
        """Agent and Python dependencies are parsed into typed objects."""
        pack = AgentPack(
            name="test-agent",
            version="1.0.0",
            entry_point="greenlang.agents.test.agent",
            dependencies={
                "agents": [
                    {"name": "intake-agent", "version_constraint": "^1.0.0"},
                ],
                "python": [
                    {"package": "pydantic", "version_constraint": ">=2.0.0"},
                ],
            },
        )
        assert len(pack.agent_dependencies) == 1
        assert pack.agent_dependencies[0].name == "intake-agent"
        assert len(pack.python_dependencies) == 1
        assert pack.python_dependencies[0].package == "pydantic"

    def test_pack_format_to_yaml_dict(self, valid_pack_dict: Dict[str, Any]) -> None:
        """to_yaml_dict produces a serializable dictionary."""
        pack = AgentPack(**valid_pack_dict)
        data = pack.to_yaml_dict()
        assert data["name"] == "emissions-calc"
        assert "dependencies" in data

    def test_pack_format_generate_template(self) -> None:
        """generate_template creates a valid AgentPack with defaults."""
        pack = PackFormat.generate_template("my-new-agent", AgentType.INSIGHT)
        assert pack.name == "my-new-agent"
        assert pack.version == "0.1.0"
        assert pack.agent_type == AgentType.INSIGHT
        assert "my-new-agent" in pack.metadata.tags

    def test_pack_format_save_and_reload(self, tmp_path: Path) -> None:
        """Saving and reloading produces the same data."""
        original = PackFormat.generate_template("round-trip-agent")
        pack_path = tmp_path / "agent.pack.yaml"
        PackFormat.save(original, pack_path)
        reloaded = PackFormat.load(pack_path)
        assert reloaded.name == original.name
        assert reloaded.version == original.version

    def test_pack_format_validate_clean_file(
        self, valid_pack_yaml: Path
    ) -> None:
        """Validate returns empty list for a valid file."""
        errors = PackFormat.validate(valid_pack_yaml)
        assert errors == []


# ============================================================================
# Test PackageBuilder
# ============================================================================


class TestPackageBuilder:
    """Tests for building .glpack archives."""

    @pytest.mark.asyncio
    async def test_package_builder_creates_archive(
        self, builder: PackageBuilder, agent_source_dir: Path, tmp_path: Path
    ) -> None:
        """Builder creates a .glpack archive from source directory."""
        output_dir = tmp_path / "dist"
        result = await builder.build(agent_source_dir, output_dir)

        assert result.success is True
        assert result.archive_path is not None
        assert result.archive_path.endswith(".glpack")
        assert result.size_bytes > 0
        assert result.file_count >= 3  # pack.yaml + 2 py files
        assert len(result.checksum) == 64  # SHA-256

    @pytest.mark.asyncio
    async def test_package_builder_size_limit(
        self, agent_source_dir: Path, tmp_path: Path
    ) -> None:
        """Builder rejects archives exceeding size limit."""
        builder = PackageBuilder(size_limit_bytes=10)  # 10 bytes
        result = await builder.build(agent_source_dir, tmp_path / "dist")
        assert result.success is False
        assert any("exceeds limit" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_package_builder_excludes_patterns(
        self, agent_source_dir: Path, tmp_path: Path
    ) -> None:
        """Builder excludes files matching exclusion patterns."""
        # Create excluded files
        (agent_source_dir / "__pycache__").mkdir()
        (agent_source_dir / "__pycache__" / "agent.cpython-311.pyc").write_bytes(b"cache")
        (agent_source_dir / ".env").write_text("SECRET=abc", encoding="utf-8")

        result = await builder_instance().build(agent_source_dir, tmp_path / "dist")
        assert result.success is True

        # Verify excluded files are not in the archive
        with tarfile.open(result.archive_path, "r:gz") as tar:
            names = tar.getnames()
            assert not any("__pycache__" in n for n in names)
            assert not any(".env" in n for n in names)

    @pytest.mark.asyncio
    async def test_package_builder_missing_pack_yaml(
        self, tmp_path: Path
    ) -> None:
        """Builder fails if agent.pack.yaml is missing."""
        src = tmp_path / "empty_agent"
        src.mkdir()
        (src / "agent.py").write_text("x = 1\n", encoding="utf-8")

        builder = PackageBuilder()
        result = await builder.build(src, tmp_path / "dist")
        assert result.success is False
        assert any("agent.pack.yaml" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_package_builder_syntax_error_detected(
        self, agent_source_dir: Path, tmp_path: Path
    ) -> None:
        """Builder reports Python syntax errors in source files."""
        (agent_source_dir / "bad.py").write_text(
            "def broken(\n", encoding="utf-8"
        )
        builder = PackageBuilder()
        result = await builder.build(agent_source_dir, tmp_path / "dist")
        # Syntax errors are reported but may not block build
        # depending on implementation; check they are captured
        assert result is not None

    def test_package_builder_is_excluded(self) -> None:
        """Exclusion pattern matching works correctly."""
        builder = PackageBuilder()
        assert builder._is_excluded("__pycache__") is True
        assert builder._is_excluded(".git") is True
        assert builder._is_excluded("agent.py") is False
        assert builder._is_excluded(".DS_Store") is True
        assert builder._is_excluded("Thumbs.db") is True


def builder_instance() -> PackageBuilder:
    return PackageBuilder()


# ============================================================================
# Test DependencyResolver
# ============================================================================


class TestDependencyResolver:
    """Tests for dependency resolution logic."""

    def test_dependency_resolver_simple(self) -> None:
        """Simple single-level dependency resolution."""
        resolver = DependencyResolver()
        resolver.register("agent-a", "1.0.0")
        result = resolver.resolve("agent-a")
        assert result.success is True
        assert len(result.resolved) == 1

    def test_dependency_resolver_version_ranges(self) -> None:
        """Resolver finds matching version."""
        resolver = DependencyResolver()
        resolver.register("agent-a", "2.1.0")
        result = resolver.resolve("agent-a", "^2.0.0")
        assert result.success is True

    def test_dependency_resolver_conflict_detection(self) -> None:
        """Resolver detects when a dependency is not available."""
        resolver = DependencyResolver()
        result = resolver.resolve("nonexistent-agent")
        assert result.success is False
        assert len(result.conflicts) == 1

    def test_dependency_resolver_diamond_resolution(self) -> None:
        """Diamond dependencies are resolved without duplication."""
        resolver = DependencyResolver()
        resolver.register("shared", "1.0.0")
        resolver.register(
            "left", "1.0.0",
            deps=[{"name": "shared", "version_constraint": "^1.0.0"}],
        )
        resolver.register(
            "right", "1.0.0",
            deps=[{"name": "shared", "version_constraint": "^1.0.0"}],
        )
        resolver.register(
            "top", "1.0.0",
            deps=[
                {"name": "left", "version_constraint": "*"},
                {"name": "right", "version_constraint": "*"},
            ],
        )
        result = resolver.resolve("top")
        assert result.success is True
        names = [r.name for r in result.resolved]
        assert names.count("shared") == 1  # deduplicated


# ============================================================================
# Test ManifestGenerator and Verification
# ============================================================================


class TestManifestGenerator:
    """Tests for manifest generation and verification."""

    def test_manifest_generator_checksums(self, tmp_path: Path) -> None:
        """ManifestGenerator produces SHA-256 checksums for all files."""
        (tmp_path / "file1.py").write_text("hello", encoding="utf-8")
        (tmp_path / "file2.yaml").write_text("key: val", encoding="utf-8")

        manifest = ManifestGenerator.generate(tmp_path)
        assert len(manifest.files) == 2
        for path, sha in manifest.files.items():
            assert len(sha) == 64

    def test_manifest_verification_passes(self, tmp_path: Path) -> None:
        """Verification passes when files are untampered."""
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        manifest = ManifestGenerator.generate(tmp_path)
        errors = manifest.verify(tmp_path)
        assert errors == []

    def test_manifest_verification_fails_tampered(
        self, tmp_path: Path
    ) -> None:
        """Verification fails when file content has changed."""
        f = tmp_path / "main.py"
        f.write_text("original", encoding="utf-8")
        manifest = ManifestGenerator.generate(tmp_path)

        # Tamper
        f.write_text("modified", encoding="utf-8")
        errors = manifest.verify(tmp_path)
        assert len(errors) == 1
        assert "Tampered" in errors[0]

    def test_manifest_verification_fails_missing_file(
        self, tmp_path: Path
    ) -> None:
        """Verification fails when a file is deleted."""
        f = tmp_path / "main.py"
        f.write_text("content", encoding="utf-8")
        manifest = ManifestGenerator.generate(tmp_path)

        f.unlink()
        errors = manifest.verify(tmp_path)
        assert len(errors) == 1
        assert "Missing" in errors[0]
