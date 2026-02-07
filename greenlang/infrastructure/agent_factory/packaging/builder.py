# -*- coding: utf-8 -*-
"""
Package Builder - Build .glpack archives from agent source directories.

Collects source files, test files, schemas, validates imports, and creates
a compressed tar archive with the agent.pack.yaml at the archive root.
Integrates with PushGateway for batch job metrics (OBS-001 Phase 3).

Example:
    >>> builder = PackageBuilder()
    >>> result = await builder.build("agents/emissions_calc/", "dist/")
    >>> assert result.success
    >>> print(result.archive_path, result.checksum)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-001 Phase 3 (PushGateway Integration)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentPack,
    PackFormat,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PushGateway Integration (OBS-001 Phase 3)
# ---------------------------------------------------------------------------

try:
    from greenlang.monitoring.pushgateway import (
        BatchJobMetrics,
        get_pushgateway_client,
    )
    _PUSHGATEWAY_AVAILABLE = True
except ImportError:
    _PUSHGATEWAY_AVAILABLE = False
    BatchJobMetrics = None  # type: ignore[assignment, misc]
    get_pushgateway_client = None  # type: ignore[assignment]
    logger.debug("PushGateway SDK not available; build metrics disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SIZE_LIMIT_BYTES = 50 * 1024 * 1024  # 50 MB
ARCHIVE_EXTENSION = ".glpack"

EXCLUDE_PATTERNS: Set[str] = {
    "__pycache__",
    ".pyc",
    ".pyo",
    ".git",
    ".env",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a package build operation.

    Attributes:
        success: Whether the build completed without errors.
        archive_path: Filesystem path to the generated .glpack file.
        size_bytes: Total archive size in bytes.
        file_count: Number of files included in the archive.
        checksum: SHA-256 hex digest of the archive file.
        errors: List of error messages if the build failed.
    """

    success: bool
    archive_path: Optional[str] = None
    size_bytes: int = 0
    file_count: int = 0
    checksum: str = ""
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class PackageBuilder:
    """Build .glpack archives from agent source directories.

    The builder performs the following steps:
      1. Validate agent.pack.yaml
      2. Collect source files (*.py)
      3. Collect test files (tests/)
      4. Collect schema files (*.json, *.yaml)
      5. Validate that no excluded patterns are included
      6. Create compressed tar archive

    Attributes:
        size_limit_bytes: Maximum allowed archive size.
        exclude_patterns: Set of filename/directory patterns to exclude.
    """

    def __init__(
        self,
        size_limit_bytes: int = DEFAULT_SIZE_LIMIT_BYTES,
        exclude_patterns: Optional[Set[str]] = None,
        enable_pushgateway: bool = True,
    ) -> None:
        """Initialize PackageBuilder.

        Args:
            size_limit_bytes: Max archive size (default 50MB).
            exclude_patterns: Additional patterns to exclude.
            enable_pushgateway: Enable PushGateway batch job metrics (OBS-001).
        """
        self.size_limit_bytes = size_limit_bytes
        self.exclude_patterns = EXCLUDE_PATTERNS.copy()
        if exclude_patterns:
            self.exclude_patterns.update(exclude_patterns)

        # PushGateway batch job metrics (OBS-001 Phase 3)
        self._enable_pushgateway = enable_pushgateway and _PUSHGATEWAY_AVAILABLE
        self._pushgateway_metrics: Optional[BatchJobMetrics] = None
        if self._enable_pushgateway:
            self._pushgateway_metrics = get_pushgateway_client("pack-build-job")
            logger.info("PushGateway batch metrics enabled for PackageBuilder")

    async def build(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
    ) -> BuildResult:
        """Build a .glpack archive from a source directory.

        Args:
            source_dir: Path to the agent source directory containing agent.pack.yaml.
            output_dir: Directory where the .glpack file will be written.

        Returns:
            BuildResult with success status, archive path, and metadata.
        """
        start = time.monotonic()
        source = Path(source_dir).resolve()
        output = Path(output_dir).resolve()
        errors: List[str] = []

        # Start PushGateway tracking (OBS-001 Phase 3)
        if self._pushgateway_metrics:
            self._pushgateway_metrics.set_status("running")

        # Step 1: Validate pack.yaml
        pack_path = source / "agent.pack.yaml"
        pack = self._validate_pack(pack_path, errors)
        if pack is None:
            self._record_build_failure(start, "validation_failed", errors)
            return BuildResult(success=False, errors=errors)

        # Step 2-4: Collect files
        collected = self._collect_files(source, errors)
        if errors:
            self._record_build_failure(start, "collection_failed", errors)
            return BuildResult(success=False, errors=errors)

        # Step 5: Validate imports
        self._validate_imports(source, collected, errors)

        # Step 6: Create archive
        output.mkdir(parents=True, exist_ok=True)
        archive_name = f"{pack.name}-{pack.version}{ARCHIVE_EXTENSION}"
        archive_path = output / archive_name

        try:
            file_count = await asyncio.to_thread(
                self._create_archive, source, collected, archive_path
            )
        except Exception as exc:
            errors.append(f"Archive creation failed: {exc}")
            self._record_build_failure(start, "archive_failed", errors)
            return BuildResult(success=False, errors=errors)

        # Check size limit
        archive_size = archive_path.stat().st_size
        if archive_size > self.size_limit_bytes:
            errors.append(
                f"Archive size {archive_size} bytes exceeds limit "
                f"{self.size_limit_bytes} bytes."
            )
            archive_path.unlink(missing_ok=True)
            self._record_build_failure(start, "size_limit_exceeded", errors)
            return BuildResult(success=False, errors=errors)

        # Compute checksum
        checksum = self._compute_sha256(archive_path)

        elapsed = time.monotonic() - start
        logger.info(
            "Built package %s v%s: %d files, %d bytes, %.2fs",
            pack.name,
            pack.version,
            file_count,
            archive_size,
            elapsed,
        )

        # Record success in PushGateway (OBS-001 Phase 3)
        if self._pushgateway_metrics:
            self._pushgateway_metrics._record_duration(elapsed)
            self._pushgateway_metrics.record_records(file_count, "files")
            self._pushgateway_metrics.record_success()
            self._pushgateway_metrics.push()

        return BuildResult(
            success=True,
            archive_path=str(archive_path),
            size_bytes=archive_size,
            file_count=file_count,
            checksum=checksum,
            errors=errors,
        )

    def _record_build_failure(
        self,
        start_time: float,
        error_type: str,
        errors: List[str],
    ) -> None:
        """Record build failure in PushGateway (OBS-001 Phase 3).

        Args:
            start_time: Build start time from time.monotonic().
            error_type: Category of the failure.
            errors: List of error messages.
        """
        if not self._pushgateway_metrics:
            return

        elapsed = time.monotonic() - start_time
        self._pushgateway_metrics._record_duration(elapsed)
        self._pushgateway_metrics.record_failure(error_type)
        self._pushgateway_metrics.push()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_pack(
        self, pack_path: Path, errors: List[str]
    ) -> Optional[AgentPack]:
        """Validate agent.pack.yaml and return parsed model."""
        if not pack_path.exists():
            errors.append(f"agent.pack.yaml not found at {pack_path}")
            return None
        try:
            return PackFormat.load(pack_path)
        except (ValueError, Exception) as exc:
            errors.append(f"Invalid agent.pack.yaml: {exc}")
            return None

    def _collect_files(
        self, source: Path, errors: List[str]
    ) -> List[Path]:
        """Collect all includable files from the source directory."""
        collected: List[Path] = []
        for root_str, dirs, files in os.walk(source):
            root = Path(root_str)
            # Prune excluded directories in-place
            dirs[:] = [
                d for d in dirs if not self._is_excluded(d)
            ]
            for fname in files:
                if self._is_excluded(fname):
                    continue
                fpath = root / fname
                collected.append(fpath)
        if not collected:
            errors.append("No files found in source directory.")
        return collected

    def _is_excluded(self, name: str) -> bool:
        """Check if a file or directory name matches exclusion patterns."""
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern or name.endswith(pattern):
                return True
        return False

    def _validate_imports(
        self,
        source: Path,
        files: List[Path],
        errors: List[str],
    ) -> None:
        """Basic validation that Python files have valid syntax."""
        for fpath in files:
            if fpath.suffix != ".py":
                continue
            try:
                content = fpath.read_text(encoding="utf-8")
                compile(content, str(fpath), "exec")
            except SyntaxError as exc:
                errors.append(f"Syntax error in {fpath.name}: {exc}")

    def _create_archive(
        self,
        source: Path,
        files: List[Path],
        archive_path: Path,
    ) -> int:
        """Create a .glpack (tar.gz) archive.

        Returns:
            Number of files added to the archive.
        """
        file_count = 0
        with tarfile.open(archive_path, "w:gz") as tar:
            for fpath in files:
                arcname = fpath.relative_to(source)
                tar.add(str(fpath), arcname=str(arcname))
                file_count += 1
        return file_count

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
