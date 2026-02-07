# -*- coding: utf-8 -*-
"""
Manifest Generator - SHA-256 integrity manifests for agent packages.

Generates a file manifest with SHA-256 checksums for every file in an agent
package. Supports verification against installed files to detect tampering,
and optional GPG signature support for cryptographic provenance.

The manifest is stored as agent.manifest.json alongside the package.

Example:
    >>> generator = ManifestGenerator()
    >>> manifest = generator.generate("agents/emissions_calc/", "1.2.0")
    >>> generator.save(manifest, "agents/emissions_calc/agent.manifest.json")
    >>> valid = generator.verify(manifest, "agents/emissions_calc/")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "agent.manifest.json"
HASH_ALGORITHM = "sha256"
CHUNK_SIZE = 8192


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PackageManifest:
    """Complete file manifest for an agent package.

    Attributes:
        agent_key: Agent identifier.
        version: Package version string.
        files: Mapping of relative file path to SHA-256 hex digest.
        total_size: Sum of all file sizes in bytes.
        created_at: UTC ISO-8601 timestamp of manifest generation.
        signature: Optional GPG signature of the manifest content.
        hash_algorithm: Hash algorithm used (default: sha256).
    """

    agent_key: str
    version: str
    files: Dict[str, str] = field(default_factory=dict)
    total_size: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: Optional[str] = None
    hash_algorithm: str = HASH_ALGORITHM

    def to_dict(self) -> Dict:
        """Serialize manifest to a plain dictionary."""
        return {
            "agent_key": self.agent_key,
            "version": self.version,
            "hash_algorithm": self.hash_algorithm,
            "files": self.files,
            "total_size": self.total_size,
            "created_at": self.created_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> PackageManifest:
        """Deserialize manifest from a dictionary.

        Args:
            data: Dictionary with manifest fields.

        Returns:
            PackageManifest instance.
        """
        return cls(
            agent_key=data["agent_key"],
            version=data["version"],
            files=data.get("files", {}),
            total_size=data.get("total_size", 0),
            created_at=data.get("created_at", ""),
            signature=data.get("signature"),
            hash_algorithm=data.get("hash_algorithm", HASH_ALGORITHM),
        )

    @property
    def file_count(self) -> int:
        """Return number of files in the manifest."""
        return len(self.files)


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of manifest verification.

    Attributes:
        valid: True if all files match their recorded checksums.
        mismatched: Files whose checksums do not match.
        missing: Files present in the manifest but missing from disk.
        extra: Files present on disk but not in the manifest.
    """

    valid: bool
    mismatched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ManifestGenerator:
    """Generate and verify SHA-256 file manifests for agent packages.

    Attributes:
        exclude_patterns: File patterns to exclude from the manifest.
    """

    def __init__(
        self,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the manifest generator.

        Args:
            exclude_patterns: Filename patterns to exclude (e.g. __pycache__).
        """
        self.exclude_patterns = exclude_patterns or [
            "__pycache__",
            ".pyc",
            ".pyo",
            ".git",
            MANIFEST_FILENAME,
        ]

    def generate(
        self,
        source_dir: str | Path,
        agent_key: str,
        version: str,
    ) -> PackageManifest:
        """Generate a manifest for all files in a source directory.

        Args:
            source_dir: Root directory to scan.
            agent_key: Agent identifier.
            version: Package version string.

        Returns:
            PackageManifest with SHA-256 checksums for all included files.
        """
        start = time.monotonic()
        source = Path(source_dir).resolve()
        files: Dict[str, str] = {}
        total_size = 0

        for root_str, dirs, filenames in os.walk(source):
            root = Path(root_str)
            dirs[:] = [d for d in dirs if not self._is_excluded(d)]
            for fname in filenames:
                if self._is_excluded(fname):
                    continue
                fpath = root / fname
                rel_path = str(fpath.relative_to(source)).replace("\\", "/")
                checksum = self._hash_file(fpath)
                files[rel_path] = checksum
                total_size += fpath.stat().st_size

        elapsed = time.monotonic() - start
        logger.info(
            "Generated manifest for %s v%s: %d files, %d bytes, %.3fs",
            agent_key,
            version,
            len(files),
            total_size,
            elapsed,
        )

        return PackageManifest(
            agent_key=agent_key,
            version=version,
            files=files,
            total_size=total_size,
        )

    def verify(
        self,
        manifest: PackageManifest,
        installed_dir: str | Path,
    ) -> VerificationResult:
        """Verify installed files against a manifest.

        Detects:
          - Mismatched checksums (tampering or corruption)
          - Missing files (incomplete installation)
          - Extra files (unexpected additions)

        Args:
            manifest: The expected file manifest.
            installed_dir: Directory containing installed files.

        Returns:
            VerificationResult with validity status and discrepancies.
        """
        installed = Path(installed_dir).resolve()
        mismatched: List[str] = []
        missing: List[str] = []

        # Check all manifest entries
        for rel_path, expected_hash in manifest.files.items():
            fpath = installed / rel_path
            if not fpath.exists():
                missing.append(rel_path)
                continue
            actual_hash = self._hash_file(fpath)
            if actual_hash != expected_hash:
                mismatched.append(rel_path)

        # Check for extra files
        actual_files: set[str] = set()
        for root_str, dirs, filenames in os.walk(installed):
            root = Path(root_str)
            dirs[:] = [d for d in dirs if not self._is_excluded(d)]
            for fname in filenames:
                if self._is_excluded(fname):
                    continue
                rel = str((root / fname).relative_to(installed)).replace("\\", "/")
                actual_files.add(rel)

        manifest_files = set(manifest.files.keys())
        extra = sorted(actual_files - manifest_files)

        valid = len(mismatched) == 0 and len(missing) == 0
        if not valid:
            logger.warning(
                "Manifest verification FAILED for %s v%s: "
                "%d mismatched, %d missing, %d extra",
                manifest.agent_key,
                manifest.version,
                len(mismatched),
                len(missing),
                len(extra),
            )
        else:
            logger.info(
                "Manifest verification PASSED for %s v%s",
                manifest.agent_key,
                manifest.version,
            )

        return VerificationResult(
            valid=valid,
            mismatched=mismatched,
            missing=missing,
            extra=extra,
        )

    def save(self, manifest: PackageManifest, path: str | Path) -> None:
        """Save manifest to a JSON file.

        Args:
            manifest: Manifest to save.
            path: Destination file path.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(manifest.to_dict(), fh, indent=2, sort_keys=False)
        logger.info("Saved manifest: %s", filepath)

    def load(self, path: str | Path) -> PackageManifest:
        """Load manifest from a JSON file.

        Args:
            path: Path to the manifest JSON file.

        Returns:
            Deserialized PackageManifest.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Manifest not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return PackageManifest.from_dict(data)

    def sign(
        self,
        manifest: PackageManifest,
        gpg_key_id: str,
    ) -> PackageManifest:
        """Sign a manifest using GPG.

        Args:
            manifest: Manifest to sign.
            gpg_key_id: GPG key ID to use for signing.

        Returns:
            New manifest instance with the signature field populated.

        Raises:
            RuntimeError: If GPG signing fails.
        """
        content = json.dumps(manifest.to_dict(), sort_keys=True)
        try:
            result = subprocess.run(
                ["gpg", "--detach-sign", "--armor", "-u", gpg_key_id, "-"],
                input=content.encode("utf-8"),
                capture_output=True,
                timeout=30,
                check=True,
            )
            signature = result.stdout.decode("utf-8")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(f"GPG signing failed: {exc}") from exc

        return PackageManifest(
            agent_key=manifest.agent_key,
            version=manifest.version,
            files=manifest.files,
            total_size=manifest.total_size,
            created_at=manifest.created_at,
            signature=signature,
            hash_algorithm=manifest.hash_algorithm,
        )

    def verify_signature(
        self,
        manifest: PackageManifest,
    ) -> bool:
        """Verify the GPG signature on a manifest.

        Args:
            manifest: Manifest with a signature to verify.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if not manifest.signature:
            logger.warning("No signature present on manifest for %s", manifest.agent_key)
            return False

        content = json.dumps(
            PackageManifest(
                agent_key=manifest.agent_key,
                version=manifest.version,
                files=manifest.files,
                total_size=manifest.total_size,
                created_at=manifest.created_at,
                signature=None,
                hash_algorithm=manifest.hash_algorithm,
            ).to_dict(),
            sort_keys=True,
        )

        try:
            result = subprocess.run(
                ["gpg", "--verify", "-"],
                input=(manifest.signature + "\n" + content).encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, name: str) -> bool:
        """Check if a filename matches exclusion patterns."""
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern or name.endswith(pattern):
                return True
        return False

    @staticmethod
    def _hash_file(filepath: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as fh:
            for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
