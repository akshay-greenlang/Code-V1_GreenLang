# -*- coding: utf-8 -*-
"""
Environment Fingerprinting Engine - AGENT-FOUND-008: Reproducibility Agent

Captures execution environment details (Python version, platform, dependencies,
environment variables) into a deterministic fingerprint for reproducibility
verification.

Zero-Hallucination Guarantees:
    - All environment data is captured from runtime APIs
    - Environment hash is deterministic SHA-256
    - Only non-sensitive environment variables are captured (GL_ prefix)
    - Dependency versions are captured from installed packages

Example:
    >>> from greenlang.reproducibility.environment_capture import EnvironmentCapture
    >>> from greenlang.reproducibility.config import ReproducibilityConfig
    >>> capture = EnvironmentCapture(ReproducibilityConfig())
    >>> fp = capture.capture()
    >>> print(fp.python_version, fp.platform_system)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.reproducibility.config import ReproducibilityConfig
from greenlang.reproducibility.models import (
    EnvironmentFingerprint,
    VerificationStatus,
    VerificationCheck,
)
from greenlang.reproducibility.metrics import record_environment_mismatch

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class EnvironmentCapture:
    """Environment fingerprinting engine.

    Captures the current execution environment into a deterministic
    fingerprint including Python version, platform info, dependency
    versions, and relevant environment variables.

    Attributes:
        _config: Reproducibility configuration.
        _fingerprints: In-memory store of captured fingerprints.

    Example:
        >>> capture = EnvironmentCapture(config)
        >>> fp = capture.capture()
        >>> print(fp.environment_hash)
    """

    # Key packages to capture versions for
    _KEY_PACKAGES = [
        "pydantic",
        "fastapi",
        "numpy",
        "pandas",
        "httpx",
        "uvicorn",
        "sqlalchemy",
    ]

    # Safe environment variable prefixes to capture
    _SAFE_VAR_PREFIXES = ["GL_", "GREENLANG_"]

    # Specific safe environment variables
    _SAFE_VARS = [
        "PYTHONPATH",
        "TZ",
        "LANG",
        "LC_ALL",
        "PYTHONHASHSEED",
    ]

    def __init__(self, config: ReproducibilityConfig) -> None:
        """Initialize EnvironmentCapture.

        Args:
            config: Reproducibility configuration instance.
        """
        self._config = config
        self._fingerprints: Dict[str, EnvironmentFingerprint] = {}
        logger.info("EnvironmentCapture initialized")

    def capture(self) -> EnvironmentFingerprint:
        """Capture the current execution environment.

        Returns:
            EnvironmentFingerprint with current environment state.
        """
        python_version = self._get_python_version()
        plat_system, plat_release, plat_machine = self._get_platform_info()
        hostname = self._get_hostname()
        dep_versions = self._get_dependency_versions()
        env_vars = self._get_environment_variables()

        # Compute environment hash
        hash_data = {
            "python_version": python_version,
            "platform_system": plat_system,
            "platform_release": plat_release,
            "platform_machine": plat_machine,
            "dependencies": dep_versions,
            "env_vars": env_vars,
        }
        env_hash = self._compute_environment_hash(hash_data)

        fingerprint = EnvironmentFingerprint(
            python_version=python_version,
            platform_system=plat_system,
            platform_release=plat_release,
            platform_machine=plat_machine,
            hostname=hostname,
            captured_at=_utcnow(),
            environment_hash=env_hash,
            greenlang_version="1.0.0",
            dependency_versions=dep_versions,
            environment_variables=env_vars,
        )

        logger.debug(
            "Environment captured: python=%s, platform=%s/%s, hash=%s",
            python_version, plat_system, plat_machine, env_hash[:16],
        )

        return fingerprint

    def _get_python_version(self) -> str:
        """Get the Python interpreter version string.

        Returns:
            Python version (e.g. '3.11.5').
        """
        return sys.version.split()[0]

    def _get_platform_info(self) -> Tuple[str, str, str]:
        """Get platform system, release, and machine architecture.

        Returns:
            Tuple of (system, release, machine).
        """
        return (
            platform.system(),
            platform.release(),
            platform.machine(),
        )

    def _get_hostname(self) -> str:
        """Get the hostname of the current machine.

        Returns:
            Hostname string, or empty string if unavailable.
        """
        try:
            return platform.node()
        except Exception:
            return ""

    def _get_dependency_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies.

        Tries to import each key package and capture its version.

        Returns:
            Dictionary mapping package name to version string.
        """
        versions: Dict[str, str] = {}

        for package_name in self._KEY_PACKAGES:
            try:
                mod = __import__(package_name)
                version = getattr(mod, "__version__", "unknown")
                versions[package_name] = str(version)
            except ImportError:
                pass
            except Exception:
                pass

        return versions

    def _get_environment_variables(self) -> Dict[str, str]:
        """Get non-sensitive environment variables.

        Captures variables with GL_ or GREENLANG_ prefixes, plus
        specific safe variables like PYTHONPATH and TZ.

        Returns:
            Dictionary of safe environment variable name-value pairs.
        """
        safe_vars: Dict[str, str] = {}

        # Capture by prefix
        for key, value in os.environ.items():
            for prefix in self._SAFE_VAR_PREFIXES:
                if key.startswith(prefix):
                    # Skip anything that looks sensitive
                    lower_key = key.lower()
                    if any(s in lower_key for s in ("secret", "password", "token", "key")):
                        continue
                    safe_vars[key] = value
                    break

        # Capture specific safe variables
        for var_name in self._SAFE_VARS:
            value = os.environ.get(var_name)
            if value is not None:
                safe_vars[var_name] = value

        return safe_vars

    def _compute_environment_hash(self, fingerprint_data: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 hash of environment data.

        Args:
            fingerprint_data: Dictionary of environment attributes.

        Returns:
            First 16 hex characters of the SHA-256 hash.
        """
        serialized = json.dumps(
            fingerprint_data, sort_keys=True, ensure_ascii=True, default=str,
        )
        full_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return full_hash[:16]

    def compare(
        self,
        fp1: EnvironmentFingerprint,
        fp2: EnvironmentFingerprint,
        strict: bool = False,
    ) -> VerificationCheck:
        """Compare two environment fingerprints.

        Args:
            fp1: First (current) environment fingerprint.
            fp2: Second (expected) environment fingerprint.
            strict: If True, fail on any mismatch. If False, warn.

        Returns:
            VerificationCheck result.
        """
        mismatches: List[str] = []

        # Check critical fields
        if fp1.python_version != fp2.python_version:
            mismatches.append(
                f"Python version: {fp1.python_version} vs {fp2.python_version}"
            )

        if fp1.platform_system != fp2.platform_system:
            mismatches.append(
                f"Platform: {fp1.platform_system} vs {fp2.platform_system}"
            )

        if fp1.platform_machine != fp2.platform_machine:
            mismatches.append(
                f"Architecture: {fp1.platform_machine} vs {fp2.platform_machine}"
            )

        # Check dependency versions
        for pkg, version in fp2.dependency_versions.items():
            current_version = fp1.dependency_versions.get(pkg, "unknown")
            if current_version != version:
                mismatches.append(f"{pkg}: {current_version} vs {version}")

        if not mismatches:
            return VerificationCheck(
                check_name="environment_verification",
                status=VerificationStatus.PASS,
                expected_value=fp2.environment_hash,
                actual_value=fp1.environment_hash,
                message="Environment matches expected configuration",
            )

        record_environment_mismatch()

        status = VerificationStatus.FAIL if strict else VerificationStatus.WARNING
        return VerificationCheck(
            check_name="environment_verification",
            status=status,
            expected_value=fp2.environment_hash,
            actual_value=fp1.environment_hash,
            message=f"Environment mismatches: {'; '.join(mismatches)}",
        )

    def store_fingerprint(self, fingerprint: EnvironmentFingerprint) -> str:
        """Store an environment fingerprint.

        Args:
            fingerprint: EnvironmentFingerprint to store.

        Returns:
            Fingerprint ID (environment_hash).
        """
        fp_id = fingerprint.environment_hash
        self._fingerprints[fp_id] = fingerprint
        logger.debug("Stored environment fingerprint: %s", fp_id)
        return fp_id

    def get_fingerprint(self, fingerprint_id: str) -> Optional[EnvironmentFingerprint]:
        """Get a stored environment fingerprint.

        Args:
            fingerprint_id: Environment hash ID.

        Returns:
            EnvironmentFingerprint or None if not found.
        """
        return self._fingerprints.get(fingerprint_id)


__all__ = [
    "EnvironmentCapture",
]
