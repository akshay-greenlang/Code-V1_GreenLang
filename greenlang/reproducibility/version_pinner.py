# -*- coding: utf-8 -*-
"""
Version Pinning and Manifest Management - AGENT-FOUND-008: Reproducibility Agent

Manages version pins for agents, models, emission factors, and data sources,
and produces version manifests for reproducible execution snapshots.

Zero-Hallucination Guarantees:
    - Version pins are exact string matches
    - Manifest hashes are deterministic SHA-256
    - No version inference or prediction
    - Complete provenance for all manifests

Example:
    >>> from greenlang.reproducibility.version_pinner import VersionPinner
    >>> from greenlang.reproducibility.config import ReproducibilityConfig
    >>> pinner = VersionPinner(ReproducibilityConfig())
    >>> manifest = pinner.create_manifest(
    ...     agent_versions={"intake": pinner.create_version_pin("agent", "intake", "1.0.0")}
    ... )
    >>> print(manifest.manifest_hash)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.reproducibility.config import ReproducibilityConfig
from greenlang.reproducibility.models import (
    VersionPin,
    VersionManifest,
    VerificationStatus,
    VerificationCheck,
)
from greenlang.reproducibility.metrics import record_non_determinism

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class VersionPinner:
    """Version pinning and manifest management engine.

    Creates version pins for individual components, assembles them into
    complete manifests, and verifies manifest consistency across runs.

    Attributes:
        _config: Reproducibility configuration.
        _manifests: In-memory store of version manifests.

    Example:
        >>> pinner = VersionPinner(config)
        >>> pin = pinner.create_version_pin("agent", "intake", "1.2.0")
        >>> manifest = pinner.create_manifest(agent_versions={"intake": pin})
    """

    def __init__(self, config: ReproducibilityConfig) -> None:
        """Initialize VersionPinner.

        Args:
            config: Reproducibility configuration instance.
        """
        self._config = config
        self._manifests: Dict[str, VersionManifest] = {}
        logger.info("VersionPinner initialized")

    def create_version_pin(
        self,
        component_type: str,
        component_id: str,
        version: str,
        version_hash: str = "",
    ) -> VersionPin:
        """Create a version pin for a component.

        Args:
            component_type: Type of component (agent, model, factor, data).
            component_id: Unique component identifier.
            version: Version string to pin.
            version_hash: Optional hash of the version content.

        Returns:
            VersionPin instance.
        """
        pin = VersionPin(
            component_type=component_type,
            component_id=component_id,
            version=version,
            version_hash=version_hash,
        )
        logger.debug(
            "Created version pin: %s/%s@%s",
            component_type, component_id, version,
        )
        return pin

    def create_manifest(
        self,
        agent_versions: Optional[Dict[str, VersionPin]] = None,
        model_versions: Optional[Dict[str, VersionPin]] = None,
        factor_versions: Optional[Dict[str, VersionPin]] = None,
        data_versions: Optional[Dict[str, VersionPin]] = None,
    ) -> VersionManifest:
        """Create a version manifest from component version pins.

        Args:
            agent_versions: Agent version pins (agent_id -> VersionPin).
            model_versions: Model version pins.
            factor_versions: Emission factor version pins.
            data_versions: Data source version pins.

        Returns:
            VersionManifest with computed manifest hash.
        """
        manifest = VersionManifest(
            agent_versions=agent_versions or {},
            model_versions=model_versions or {},
            factor_versions=factor_versions or {},
            data_versions=data_versions or {},
        )

        # Compute manifest hash
        manifest_hash = self._compute_manifest_hash(manifest)
        manifest.manifest_hash = manifest_hash
        manifest.manifest_id = manifest_hash[:16]

        # Store
        self._manifests[manifest.manifest_id] = manifest

        logger.info(
            "Created version manifest: id=%s, agents=%d, models=%d, factors=%d, data=%d",
            manifest.manifest_id,
            len(manifest.agent_versions),
            len(manifest.model_versions),
            len(manifest.factor_versions),
            len(manifest.data_versions),
        )
        return manifest

    def _compute_manifest_hash(self, manifest: VersionManifest) -> str:
        """Compute deterministic SHA-256 hash of a version manifest.

        Args:
            manifest: VersionManifest to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        hash_data: Dict[str, Any] = {
            "agents": {
                k: {"version": v.version, "hash": v.version_hash}
                for k, v in sorted(manifest.agent_versions.items())
            },
            "models": {
                k: {"version": v.version, "hash": v.version_hash}
                for k, v in sorted(manifest.model_versions.items())
            },
            "factors": {
                k: {"version": v.version, "hash": v.version_hash}
                for k, v in sorted(manifest.factor_versions.items())
            },
            "data": {
                k: {"version": v.version, "hash": v.version_hash}
                for k, v in sorted(manifest.data_versions.items())
            },
        }
        serialized = json.dumps(hash_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def verify_manifest(
        self,
        current: VersionManifest,
        expected: VersionManifest,
    ) -> Dict[str, VerificationCheck]:
        """Verify current version manifest matches expected.

        Compares each version group (agents, models, factors, data)
        and returns individual checks.

        Args:
            current: Current version manifest.
            expected: Expected version manifest.

        Returns:
            Dictionary of group_name -> VerificationCheck.
        """
        checks: Dict[str, VerificationCheck] = {}

        # Verify each version group
        agent_checks = self._verify_version_group(
            {k: v.version for k, v in current.agent_versions.items()},
            {k: v.version for k, v in expected.agent_versions.items()},
            "agent",
        )
        for check in agent_checks:
            checks[check.check_name] = check

        model_checks = self._verify_version_group(
            {k: v.version for k, v in current.model_versions.items()},
            {k: v.version for k, v in expected.model_versions.items()},
            "model",
        )
        for check in model_checks:
            checks[check.check_name] = check

        factor_checks = self._verify_version_group(
            {k: v.version for k, v in current.factor_versions.items()},
            {k: v.version for k, v in expected.factor_versions.items()},
            "factor",
        )
        for check in factor_checks:
            checks[check.check_name] = check

        data_checks = self._verify_version_group(
            {k: v.version for k, v in current.data_versions.items()},
            {k: v.version for k, v in expected.data_versions.items()},
            "data",
        )
        for check in data_checks:
            checks[check.check_name] = check

        return checks

    def _verify_version_group(
        self,
        current: Dict[str, str],
        expected: Dict[str, str],
        group_name: str,
    ) -> List[VerificationCheck]:
        """Verify a group of version pins against expected.

        Args:
            current: Current version mapping (id -> version).
            expected: Expected version mapping.
            group_name: Group name for check naming.

        Returns:
            List of VerificationCheck results.
        """
        checks: List[VerificationCheck] = []
        all_keys = set(current.keys()) | set(expected.keys())

        for key in sorted(all_keys):
            check_name = f"version_{group_name}_{key}"
            cur_ver = current.get(key)
            exp_ver = expected.get(key)

            if exp_ver is None:
                # Extra component in current -- informational
                checks.append(VerificationCheck(
                    check_name=check_name,
                    status=VerificationStatus.WARNING,
                    actual_value=cur_ver,
                    message=f"Extra {group_name} {key} (version {cur_ver}) not in expected manifest",
                ))
                continue

            if cur_ver is None:
                # Missing component in current
                record_non_determinism("dependency_version")
                checks.append(VerificationCheck(
                    check_name=check_name,
                    status=VerificationStatus.FAIL,
                    expected_value=exp_ver,
                    message=f"Missing {group_name} {key} (expected version {exp_ver})",
                ))
                continue

            if cur_ver == exp_ver:
                checks.append(VerificationCheck(
                    check_name=check_name,
                    status=VerificationStatus.PASS,
                    expected_value=exp_ver,
                    actual_value=cur_ver,
                    message=f"{group_name.capitalize()} {key} version verified: {cur_ver}",
                ))
            else:
                record_non_determinism("dependency_version")
                checks.append(VerificationCheck(
                    check_name=check_name,
                    status=VerificationStatus.FAIL,
                    expected_value=exp_ver,
                    actual_value=cur_ver,
                    message=f"{group_name.capitalize()} {key} version mismatch: {cur_ver} vs {exp_ver}",
                ))

        return checks

    def store_manifest(self, manifest: VersionManifest) -> str:
        """Store a version manifest.

        Args:
            manifest: VersionManifest to store.

        Returns:
            Manifest ID.
        """
        manifest_id = manifest.manifest_id or manifest.manifest_hash[:16]
        self._manifests[manifest_id] = manifest
        logger.debug("Stored version manifest: %s", manifest_id)
        return manifest_id

    def get_manifest(self, manifest_id: str) -> Optional[VersionManifest]:
        """Get a stored version manifest.

        Args:
            manifest_id: Unique manifest identifier.

        Returns:
            VersionManifest or None if not found.
        """
        return self._manifests.get(manifest_id)

    def list_manifests(self, limit: int = 50) -> List[VersionManifest]:
        """List stored version manifests.

        Args:
            limit: Maximum number of results.

        Returns:
            List of VersionManifest records, newest first.
        """
        manifests = list(self._manifests.values())
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests[:limit]

    def pin_current_versions(self) -> VersionManifest:
        """Capture and pin the current system state as a version manifest.

        Automatically detects the current Python version and installed
        key packages.

        Returns:
            VersionManifest representing the current system state.
        """
        # Create pins for system components
        agent_versions: Dict[str, VersionPin] = {}
        model_versions: Dict[str, VersionPin] = {}
        factor_versions: Dict[str, VersionPin] = {}
        data_versions: Dict[str, VersionPin] = {}

        # Pin Python version as a data source
        data_versions["python"] = self.create_version_pin(
            "data", "python", sys.version.split()[0],
        )

        # Pin key package versions
        key_packages = [
            "pydantic", "fastapi", "numpy", "pandas",
            "httpx", "uvicorn", "sqlalchemy",
        ]
        for pkg_name in key_packages:
            try:
                mod = __import__(pkg_name)
                version = getattr(mod, "__version__", "unknown")
                data_versions[pkg_name] = self.create_version_pin(
                    "data", pkg_name, str(version),
                )
            except ImportError:
                pass

        return self.create_manifest(
            agent_versions=agent_versions,
            model_versions=model_versions,
            factor_versions=factor_versions,
            data_versions=data_versions,
        )


__all__ = [
    "VersionPinner",
]
