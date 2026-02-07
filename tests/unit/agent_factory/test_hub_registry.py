# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Hub Registry: publishing, searching,
downloading, listing versions, unpublishing, local index management,
and package validation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


@dataclass
class PackageInfo:
    name: str
    version: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0
    published_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HubValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


class HubValidator:
    REQUIRED_FIELDS = {"name", "version", "entry_point"}

    def validate(self, package: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in package or not package[field_name]:
                errors.append(f"Missing required field: {field_name}")
        name = package.get("name", "")
        if name and not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            errors.append(f"Invalid package name: {name}")
        version = package.get("version", "")
        if version and not all(p.isdigit() for p in version.split(".")[:3]):
            errors.append(f"Invalid version format: {version}")
        return errors


class HubRegistry:
    """In-memory agent hub for testing."""

    def __init__(self) -> None:
        self._packages: Dict[str, Dict[str, PackageInfo]] = {}  # name -> version -> info
        self._validator = HubValidator()

    def publish(self, package: PackageInfo) -> bool:
        errors = self._validator.validate({
            "name": package.name,
            "version": package.version,
            "entry_point": package.metadata.get("entry_point", ""),
        })
        if errors:
            raise HubValidationError(errors)

        if package.name not in self._packages:
            self._packages[package.name] = {}
        if package.version in self._packages[package.name]:
            return False  # already exists
        self._packages[package.name][package.version] = package
        return True

    def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[PackageInfo]:
        results: List[PackageInfo] = []
        for name, versions in self._packages.items():
            for version, pkg in versions.items():
                if query and query.lower() not in name.lower() and query.lower() not in pkg.description.lower():
                    continue
                if tags and not any(t in pkg.tags for t in tags):
                    continue
                results.append(pkg)
        return results

    def download(self, name: str, version: Optional[str] = None) -> Optional[PackageInfo]:
        versions = self._packages.get(name)
        if versions is None:
            return None
        if version:
            return versions.get(version)
        # Return latest
        if not versions:
            return None
        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_versions(self, name: str) -> List[str]:
        versions = self._packages.get(name, {})
        return sorted(versions.keys())

    def unpublish(self, name: str, version: str) -> bool:
        versions = self._packages.get(name)
        if versions is None or version not in versions:
            return False
        del versions[version]
        if not versions:
            del self._packages[name]
        return True


class LocalIndex:
    """Local package index for installed agents."""

    def __init__(self) -> None:
        self._installed: Dict[str, PackageInfo] = {}  # name -> info
        self._persisted: List[Dict[str, Any]] = []

    def add(self, package: PackageInfo) -> None:
        self._installed[package.name] = package
        self._persist()

    def remove(self, name: str) -> bool:
        if name not in self._installed:
            return False
        del self._installed[name]
        self._persist()
        return True

    def get(self, name: str) -> Optional[PackageInfo]:
        return self._installed.get(name)

    def list_all(self) -> List[PackageInfo]:
        return list(self._installed.values())

    def _persist(self) -> None:
        self._persisted = [
            {"name": p.name, "version": p.version}
            for p in self._installed.values()
        ]

    @property
    def persisted_state(self) -> List[Dict[str, Any]]:
        return list(self._persisted)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub() -> HubRegistry:
    return HubRegistry()


@pytest.fixture
def local_index() -> LocalIndex:
    return LocalIndex()


@pytest.fixture
def sample_package() -> PackageInfo:
    return PackageInfo(
        name="emissions-calc",
        version="1.0.0",
        description="Carbon emissions calculator agent",
        author="GreenLang Team",
        tags=["emissions", "scope1"],
        checksum="abc123",
        size_bytes=1024,
        metadata={"entry_point": "greenlang.agents.emissions_calc.agent"},
    )


# ============================================================================
# Tests
# ============================================================================


class TestHubRegistry:
    """Tests for the agent hub registry."""

    def test_hub_registry_publish(
        self, hub: HubRegistry, sample_package: PackageInfo
    ) -> None:
        """Publishing a new package succeeds."""
        result = hub.publish(sample_package)
        assert result is True

    def test_hub_registry_publish_duplicate(
        self, hub: HubRegistry, sample_package: PackageInfo
    ) -> None:
        """Publishing the same version twice returns False."""
        hub.publish(sample_package)
        result = hub.publish(sample_package)
        assert result is False

    def test_hub_registry_search(
        self, hub: HubRegistry, sample_package: PackageInfo
    ) -> None:
        """Searching by name finds the published package."""
        hub.publish(sample_package)
        results = hub.search("emissions")
        assert len(results) == 1
        assert results[0].name == "emissions-calc"

    def test_hub_registry_search_by_tags(
        self, hub: HubRegistry, sample_package: PackageInfo
    ) -> None:
        """Searching by tags finds matching packages."""
        hub.publish(sample_package)
        results = hub.search(tags=["scope1"])
        assert len(results) == 1

    def test_hub_registry_search_no_results(
        self, hub: HubRegistry
    ) -> None:
        """Searching with no matching query returns empty."""
        results = hub.search("nonexistent-agent")
        assert results == []

    def test_hub_registry_download(
        self, hub: HubRegistry, sample_package: PackageInfo
    ) -> None:
        """Downloading a published package returns the info."""
        hub.publish(sample_package)
        pkg = hub.download("emissions-calc", "1.0.0")
        assert pkg is not None
        assert pkg.version == "1.0.0"

    def test_hub_registry_download_latest(
        self, hub: HubRegistry
    ) -> None:
        """Downloading without version returns the latest."""
        hub.publish(PackageInfo(
            name="agent-x", version="1.0.0",
            metadata={"entry_point": "x.agent"},
        ))
        hub.publish(PackageInfo(
            name="agent-x", version="2.0.0",
            metadata={"entry_point": "x.agent"},
        ))
        pkg = hub.download("agent-x")
        assert pkg is not None
        assert pkg.version == "2.0.0"

    def test_hub_registry_download_nonexistent(
        self, hub: HubRegistry
    ) -> None:
        """Downloading a nonexistent package returns None."""
        assert hub.download("ghost-agent") is None

    def test_hub_registry_list_versions(
        self, hub: HubRegistry
    ) -> None:
        """list_versions returns all published versions."""
        hub.publish(PackageInfo(
            name="agent-x", version="1.0.0",
            metadata={"entry_point": "x.agent"},
        ))
        hub.publish(PackageInfo(
            name="agent-x", version="1.1.0",
            metadata={"entry_point": "x.agent"},
        ))
        versions = hub.list_versions("agent-x")
        assert versions == ["1.0.0", "1.1.0"]

    def test_hub_registry_unpublish(
        self, hub: HubRegistry, sample_package: PackageInfo
    ) -> None:
        """Unpublishing removes the version."""
        hub.publish(sample_package)
        result = hub.unpublish("emissions-calc", "1.0.0")
        assert result is True
        assert hub.download("emissions-calc") is None

    def test_hub_registry_unpublish_nonexistent(
        self, hub: HubRegistry
    ) -> None:
        """Unpublishing a nonexistent version returns False."""
        assert hub.unpublish("ghost", "1.0.0") is False


class TestLocalIndex:
    """Tests for the local installed package index."""

    def test_local_index_add(
        self, local_index: LocalIndex, sample_package: PackageInfo
    ) -> None:
        """Adding a package makes it retrievable."""
        local_index.add(sample_package)
        pkg = local_index.get("emissions-calc")
        assert pkg is not None
        assert pkg.version == "1.0.0"

    def test_local_index_remove(
        self, local_index: LocalIndex, sample_package: PackageInfo
    ) -> None:
        """Removing a package makes it unretrievable."""
        local_index.add(sample_package)
        result = local_index.remove("emissions-calc")
        assert result is True
        assert local_index.get("emissions-calc") is None

    def test_local_index_remove_nonexistent(
        self, local_index: LocalIndex
    ) -> None:
        """Removing a nonexistent package returns False."""
        assert local_index.remove("ghost") is False

    def test_local_index_persistence(
        self, local_index: LocalIndex, sample_package: PackageInfo
    ) -> None:
        """Persistence state is updated on add/remove."""
        local_index.add(sample_package)
        assert len(local_index.persisted_state) == 1
        assert local_index.persisted_state[0]["name"] == "emissions-calc"

        local_index.remove("emissions-calc")
        assert len(local_index.persisted_state) == 0

    def test_local_index_list_all(
        self, local_index: LocalIndex
    ) -> None:
        """list_all returns all installed packages."""
        local_index.add(PackageInfo(
            name="a", version="1.0.0",
            metadata={"entry_point": "a.agent"},
        ))
        local_index.add(PackageInfo(
            name="b", version="2.0.0",
            metadata={"entry_point": "b.agent"},
        ))
        assert len(local_index.list_all()) == 2


class TestHubValidator:
    """Tests for package validation."""

    def test_hub_validator_valid_package(self) -> None:
        """Valid package passes validation."""
        validator = HubValidator()
        errors = validator.validate({
            "name": "valid-agent",
            "version": "1.0.0",
            "entry_point": "greenlang.agents.valid.agent",
        })
        assert errors == []

    def test_hub_validator_invalid_package(self) -> None:
        """Missing fields are reported."""
        validator = HubValidator()
        errors = validator.validate({})
        assert len(errors) == 3  # name, version, entry_point all missing

    def test_hub_validator_invalid_name(self) -> None:
        """Invalid name characters are reported."""
        validator = HubValidator()
        errors = validator.validate({
            "name": "INVALID NAME!",
            "version": "1.0.0",
            "entry_point": "x.agent",
        })
        assert any("Invalid package name" in e for e in errors)
