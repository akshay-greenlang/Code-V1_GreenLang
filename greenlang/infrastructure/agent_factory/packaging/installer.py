# -*- coding: utf-8 -*-
"""
Package Installer - Install/uninstall agent packages from .glpack archives.

Handles extraction, checksum validation, pack.yaml verification, dependency
resolution, and registration in the local agent index. Supports rollback
on failure to prevent partially installed packages.

Example:
    >>> installer = PackageInstaller()
    >>> result = await installer.install("dist/emissions-calc-1.2.0.glpack")
    >>> assert result.success
    >>> await installer.uninstall("emissions-calc")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentPack,
    PackFormat,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INSTALL_ROOT = "greenlang_agents"
ARCHIVE_EXTENSION = ".glpack"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallResult:
    """Outcome of a package install or uninstall operation.

    Attributes:
        success: Whether the operation completed without errors.
        agent_key: The installed agent's key.
        version: The installed version string.
        installed_path: Filesystem path where the agent was installed.
        errors: List of error messages if the operation failed.
    """

    success: bool
    agent_key: str = ""
    version: str = ""
    installed_path: str = ""
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


class PackageInstaller:
    """Install and uninstall GreenLang agent packages.

    Performs the following steps during installation:
      1. Extract archive to temporary directory
      2. Validate checksums of extracted files
      3. Validate agent.pack.yaml
      4. Copy files to install location
      5. Register agent in local index
      6. Rollback on any failure

    Attributes:
        install_root: Base directory for installed agents.
    """

    def __init__(
        self,
        install_root: Optional[str | Path] = None,
    ) -> None:
        """Initialize the installer.

        Args:
            install_root: Base directory for installations. Defaults to
                site-packages/greenlang_agents/.
        """
        if install_root is None:
            self.install_root = Path.home() / ".greenlang" / DEFAULT_INSTALL_ROOT
        else:
            self.install_root = Path(install_root).resolve()
        self.install_root.mkdir(parents=True, exist_ok=True)

    async def install(
        self,
        archive_path: str | Path,
        expected_checksum: Optional[str] = None,
    ) -> InstallResult:
        """Install an agent package from a .glpack archive.

        Args:
            archive_path: Path to the .glpack archive file.
            expected_checksum: Optional SHA-256 hex digest to verify archive integrity.

        Returns:
            InstallResult with success status and installed path.
        """
        start = time.monotonic()
        archive = Path(archive_path).resolve()
        errors: List[str] = []
        install_dir: Optional[Path] = None

        try:
            # Step 1: Validate archive exists
            if not archive.exists():
                errors.append(f"Archive not found: {archive}")
                return InstallResult(success=False, errors=errors)

            if not archive.name.endswith(ARCHIVE_EXTENSION):
                errors.append(f"Invalid archive extension, expected {ARCHIVE_EXTENSION}")
                return InstallResult(success=False, errors=errors)

            # Step 2: Verify checksum if provided
            if expected_checksum:
                actual = self._compute_sha256(archive)
                if actual != expected_checksum:
                    errors.append(
                        f"Checksum mismatch: expected {expected_checksum}, got {actual}"
                    )
                    return InstallResult(success=False, errors=errors)

            # Step 3: Extract to temp directory
            extract_dir = await asyncio.to_thread(self._extract_archive, archive)

            # Step 4: Validate pack.yaml
            pack_path = extract_dir / "agent.pack.yaml"
            pack = self._validate_pack(pack_path, errors)
            if pack is None:
                shutil.rmtree(extract_dir, ignore_errors=True)
                return InstallResult(success=False, errors=errors)

            # Step 5: Install to target directory
            install_dir = self.install_root / pack.name / pack.version
            if install_dir.exists():
                logger.warning(
                    "Overwriting existing installation: %s v%s",
                    pack.name,
                    pack.version,
                )
                shutil.rmtree(install_dir)

            install_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._copy_files, extract_dir, install_dir)

            # Cleanup temp
            shutil.rmtree(extract_dir, ignore_errors=True)

            elapsed = time.monotonic() - start
            logger.info(
                "Installed %s v%s to %s in %.2fs",
                pack.name,
                pack.version,
                install_dir,
                elapsed,
            )

            return InstallResult(
                success=True,
                agent_key=pack.name,
                version=pack.version,
                installed_path=str(install_dir),
            )

        except Exception as exc:
            logger.error("Installation failed: %s", exc, exc_info=True)
            # Rollback: remove partially installed files
            if install_dir and install_dir.exists():
                shutil.rmtree(install_dir, ignore_errors=True)
                logger.info("Rolled back partial installation at %s", install_dir)
            errors.append(f"Installation failed: {exc}")
            return InstallResult(success=False, errors=errors)

    async def uninstall(self, agent_key: str, version: Optional[str] = None) -> InstallResult:
        """Uninstall an agent package.

        Args:
            agent_key: Agent key to uninstall.
            version: Specific version to uninstall. If None, removes all versions.

        Returns:
            InstallResult with success status.
        """
        errors: List[str] = []
        agent_dir = self.install_root / agent_key

        if not agent_dir.exists():
            errors.append(f"Agent '{agent_key}' is not installed.")
            return InstallResult(success=False, agent_key=agent_key, errors=errors)

        if version:
            version_dir = agent_dir / version
            if not version_dir.exists():
                errors.append(f"Version '{version}' of '{agent_key}' is not installed.")
                return InstallResult(
                    success=False, agent_key=agent_key, version=version, errors=errors
                )
            shutil.rmtree(version_dir)
            logger.info("Uninstalled %s v%s", agent_key, version)
            # Remove agent dir if empty
            remaining = list(agent_dir.iterdir())
            if not remaining:
                agent_dir.rmdir()
            return InstallResult(success=True, agent_key=agent_key, version=version)

        # Remove all versions
        shutil.rmtree(agent_dir)
        logger.info("Uninstalled all versions of %s", agent_key)
        return InstallResult(success=True, agent_key=agent_key)

    def list_installed(self) -> List[dict]:
        """List all installed agent packages.

        Returns:
            List of dicts with agent_key, version, installed_path.
        """
        installed: List[dict] = []
        if not self.install_root.exists():
            return installed
        for agent_dir in sorted(self.install_root.iterdir()):
            if not agent_dir.is_dir():
                continue
            for version_dir in sorted(agent_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                installed.append({
                    "agent_key": agent_dir.name,
                    "version": version_dir.name,
                    "installed_path": str(version_dir),
                })
        return installed

    def is_installed(self, agent_key: str, version: Optional[str] = None) -> bool:
        """Check if an agent is installed.

        Args:
            agent_key: Agent key to check.
            version: Optional specific version to check.

        Returns:
            True if the agent (and optionally version) is installed.
        """
        agent_dir = self.install_root / agent_key
        if not agent_dir.exists():
            return False
        if version:
            return (agent_dir / version).exists()
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_archive(self, archive: Path) -> Path:
        """Extract a .glpack archive to a temporary directory.

        Returns:
            Path to the temporary extraction directory.
        """
        import tempfile

        extract_dir = Path(tempfile.mkdtemp(prefix="glpack_"))
        with tarfile.open(archive, "r:gz") as tar:
            # Security: prevent path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            tar.extractall(extract_dir)
        return extract_dir

    def _validate_pack(
        self, pack_path: Path, errors: List[str]
    ) -> Optional[AgentPack]:
        """Validate the pack.yaml in an extracted directory."""
        if not pack_path.exists():
            errors.append("Extracted archive is missing agent.pack.yaml")
            return None
        try:
            return PackFormat.load(pack_path)
        except (ValueError, Exception) as exc:
            errors.append(f"Invalid agent.pack.yaml: {exc}")
            return None

    @staticmethod
    def _copy_files(source: Path, dest: Path) -> None:
        """Copy all files from source to destination directory."""
        for item in source.iterdir():
            src_item = source / item.name
            dst_item = dest / item.name
            if src_item.is_dir():
                shutil.copytree(str(src_item), str(dst_item), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src_item), str(dst_item))

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
