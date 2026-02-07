# -*- coding: utf-8 -*-
"""
Agent Hub Registry - Central package registry for published agent packages.

Provides package publishing, versioned storage, search, download, and
unpublish capabilities. Uses the filesystem (or S3) for package storage
and an in-memory index (or PostgreSQL) for metadata.

Example:
    >>> registry = AgentHubRegistry(storage_path="./hub_storage")
    >>> record = await registry.publish("dist/emissions-calc-1.0.0.glpack")
    >>> results = await registry.search("emissions")
    >>> path = await registry.download("emissions-calc", "1.0.0")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentPack,
    PackFormat,
)
from greenlang.infrastructure.agent_factory.packaging.installer import (
    PackageInstaller,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PackageRecord:
    """Metadata record for a published agent package.

    Attributes:
        agent_key: Agent identifier.
        version: Package version string.
        description: Agent description.
        author: Package author.
        checksum: SHA-256 hex digest of the .glpack file.
        size: Package size in bytes.
        published_at: UTC ISO-8601 timestamp.
        download_count: Number of times this version has been downloaded.
        tags: Searchable tags.
        agent_type: Agent classification (deterministic/reasoning/insight).
        storage_path: Path to the stored .glpack file.
    """

    agent_key: str
    version: str
    description: str = ""
    author: str = ""
    checksum: str = ""
    size: int = 0
    published_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    download_count: int = 0
    tags: List[str] = field(default_factory=list)
    agent_type: str = "deterministic"
    storage_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_key": self.agent_key,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "checksum": self.checksum,
            "size": self.size,
            "published_at": self.published_at,
            "download_count": self.download_count,
            "tags": self.tags,
            "agent_type": self.agent_type,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AgentHubRegistry:
    """Central registry for published GreenLang agent packages.

    Stores package files in a structured directory hierarchy and maintains
    an in-memory metadata index. Designed to be backed by S3 for packages
    and PostgreSQL for metadata in production.

    Attributes:
        storage_path: Root directory for package storage.
    """

    def __init__(self, storage_path: str | Path) -> None:
        """Initialize the registry.

        Args:
            storage_path: Root directory for storing published packages.
        """
        self.storage_path = Path(storage_path).resolve()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, PackageRecord]] = {}
        logger.info("Initialized AgentHubRegistry at %s", self.storage_path)

    async def publish(self, package_path: str | Path) -> PackageRecord:
        """Publish an agent package to the registry.

        Steps:
          1. Validate the .glpack archive
          2. Extract and parse agent.pack.yaml
          3. Compute checksum
          4. Store the package file
          5. Create index record

        Args:
            package_path: Path to the .glpack archive.

        Returns:
            PackageRecord for the published package.

        Raises:
            FileNotFoundError: If the archive does not exist.
            ValueError: If the archive is invalid.
        """
        archive = Path(package_path).resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Package not found: {archive}")

        # Extract and validate pack.yaml
        pack = await asyncio.to_thread(self._extract_and_validate, archive)

        # Check for duplicate version
        existing = self._index.get(pack.name, {}).get(pack.version)
        if existing:
            raise ValueError(
                f"Version {pack.version} of '{pack.name}' is already published. "
                f"Unpublish it first or bump the version."
            )

        # Compute checksum
        checksum = self._compute_sha256(archive)
        size = archive.stat().st_size

        # Store the package
        dest_dir = self.storage_path / pack.name / pack.version
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / archive.name
        await asyncio.to_thread(shutil.copy2, str(archive), str(dest_path))

        # Create record
        record = PackageRecord(
            agent_key=pack.name,
            version=pack.version,
            description=pack.description,
            author=pack.metadata.author,
            checksum=checksum,
            size=size,
            tags=pack.metadata.tags,
            agent_type=pack.agent_type.value,
            storage_path=str(dest_path),
        )

        self._index.setdefault(pack.name, {})[pack.version] = record

        logger.info(
            "Published %s v%s (checksum=%s, size=%d bytes)",
            pack.name,
            pack.version,
            checksum[:12],
            size,
        )
        return record

    async def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
    ) -> List[PackageRecord]:
        """Search for packages in the registry.

        Args:
            query: Text search against agent_key and description.
            tags: Filter by tags (AND logic).
            agent_type: Filter by agent type.

        Returns:
            List of matching PackageRecord (latest version per agent).
        """
        results: List[PackageRecord] = []
        query_lower = query.lower()

        for agent_key, versions in self._index.items():
            # Get latest version
            latest = self._get_latest(versions)
            if latest is None:
                continue

            # Text search
            if query_lower:
                searchable = f"{latest.agent_key} {latest.description}".lower()
                if query_lower not in searchable:
                    continue

            # Tag filter
            if tags:
                record_tags = set(latest.tags)
                if not all(t in record_tags for t in tags):
                    continue

            # Type filter
            if agent_type and latest.agent_type != agent_type:
                continue

            results.append(latest)

        return results

    async def download(
        self,
        agent_key: str,
        version: Optional[str] = None,
    ) -> str:
        """Get the path to a published package for download.

        Args:
            agent_key: Agent identifier.
            version: Specific version. If None, returns the latest.

        Returns:
            Filesystem path to the .glpack file.

        Raises:
            KeyError: If the agent or version is not found.
        """
        versions = self._index.get(agent_key)
        if not versions:
            raise KeyError(f"Agent '{agent_key}' not found in registry")

        if version:
            record = versions.get(version)
            if not record:
                raise KeyError(f"Version '{version}' of '{agent_key}' not found")
        else:
            record = self._get_latest(versions)
            if record is None:
                raise KeyError(f"No versions found for '{agent_key}'")

        record.download_count += 1
        logger.info(
            "Download requested: %s v%s (total downloads: %d)",
            agent_key,
            record.version,
            record.download_count,
        )
        return record.storage_path

    async def list_versions(self, agent_key: str) -> List[str]:
        """List all published versions of an agent.

        Args:
            agent_key: Agent identifier.

        Returns:
            List of version strings, sorted ascending.
        """
        versions = self._index.get(agent_key, {})
        return sorted(versions.keys())

    async def unpublish(self, agent_key: str, version: str) -> bool:
        """Remove a specific version from the registry.

        Args:
            agent_key: Agent identifier.
            version: Version to remove.

        Returns:
            True if the version was found and removed.
        """
        versions = self._index.get(agent_key, {})
        record = versions.pop(version, None)
        if record is None:
            return False

        # Remove stored file
        storage = Path(record.storage_path)
        if storage.exists():
            storage.unlink()
            # Clean up empty directories
            parent = storage.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

        # Remove agent key if no versions remain
        if not versions:
            self._index.pop(agent_key, None)

        logger.info("Unpublished %s v%s", agent_key, version)
        return True

    def get_record(self, agent_key: str, version: str) -> Optional[PackageRecord]:
        """Get a specific package record.

        Args:
            agent_key: Agent identifier.
            version: Version string.

        Returns:
            PackageRecord or None.
        """
        return self._index.get(agent_key, {}).get(version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_validate(self, archive: Path) -> AgentPack:
        """Extract agent.pack.yaml from archive and validate it."""
        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory(prefix="glhub_") as tmpdir:
            with tarfile.open(archive, "r:gz") as tar:
                # Find and extract agent.pack.yaml
                members = tar.getmembers()
                pack_member = None
                for m in members:
                    if m.name == "agent.pack.yaml" or m.name.endswith("/agent.pack.yaml"):
                        pack_member = m
                        break
                if pack_member is None:
                    raise ValueError("Archive does not contain agent.pack.yaml")

                tar.extract(pack_member, tmpdir)
                pack_path = Path(tmpdir) / pack_member.name
                return PackFormat.load(pack_path)

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _get_latest(
        versions: Dict[str, PackageRecord],
    ) -> Optional[PackageRecord]:
        """Get the latest version record from a versions dict."""
        if not versions:
            return None
        # Sort by semver-like comparison
        def version_key(v: str) -> tuple:
            parts = v.split("-")[0].split("+")[0].split(".")
            return tuple(int(p) for p in parts if p.isdigit())

        latest_version = max(versions.keys(), key=version_key)
        return versions[latest_version]
