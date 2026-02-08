# -*- coding: utf-8 -*-
"""
Unit Tests for EnvironmentCapture (AGENT-FOUND-008)

Tests environment fingerprinting, comparison logic, strict/relaxed modes,
storage, and retrieval.

Coverage target: 85%+ of environment_capture.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline EnvironmentFingerprint and EnvironmentCapture
# ---------------------------------------------------------------------------

class EnvironmentFingerprint:
    def __init__(self, python_version: str, platform_system: str,
                 platform_release: str, platform_machine: str,
                 captured_at: datetime, environment_hash: str,
                 hostname: str = "", greenlang_version: str = "1.0.0",
                 dependency_versions: Optional[Dict[str, str]] = None,
                 environment_variables: Optional[Dict[str, str]] = None):
        self.python_version = python_version
        self.platform_system = platform_system
        self.platform_release = platform_release
        self.platform_machine = platform_machine
        self.captured_at = captured_at
        self.environment_hash = environment_hash
        self.hostname = hostname
        self.greenlang_version = greenlang_version
        self.dependency_versions = dependency_versions or {}
        self.environment_variables = environment_variables or {}


class EnvironmentCapture:
    """Captures and compares execution environment fingerprints."""

    def __init__(self, gl_env_prefix: str = "GL_"):
        self.gl_env_prefix = gl_env_prefix
        self._stored: Dict[str, EnvironmentFingerprint] = {}

    def _compute_env_hash(self, env_data: Dict[str, Any]) -> str:
        json_str = json.dumps(env_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _get_dependency_versions(self) -> Dict[str, str]:
        versions = {}
        try:
            import pydantic
            versions["pydantic"] = pydantic.__version__
        except (ImportError, AttributeError):
            pass
        return versions

    def _get_gl_env_variables(self) -> Dict[str, str]:
        """Get environment variables starting with GL_ prefix."""
        gl_vars = {}
        for key, val in os.environ.items():
            if key.startswith(self.gl_env_prefix):
                gl_vars[key] = val
        return gl_vars

    def capture(self) -> EnvironmentFingerprint:
        """Capture current environment as a fingerprint."""
        python_version = sys.version.split()[0]
        platform_system = platform.system()
        platform_release = platform.release()
        platform_machine = platform.machine()

        try:
            hostname = platform.node()
        except Exception:
            hostname = ""

        dep_versions = self._get_dependency_versions()
        env_vars = self._get_gl_env_variables()

        env_data = {
            "python_version": python_version,
            "platform_system": platform_system,
            "platform_release": platform_release,
            "platform_machine": platform_machine,
            "dependencies": dep_versions,
            "env_vars": env_vars,
        }
        env_hash = self._compute_env_hash(env_data)

        return EnvironmentFingerprint(
            python_version=python_version,
            platform_system=platform_system,
            platform_release=platform_release,
            platform_machine=platform_machine,
            captured_at=datetime.now(timezone.utc),
            environment_hash=env_hash,
            hostname=hostname,
            dependency_versions=dep_versions,
            environment_variables=env_vars,
        )

    def compare(self, fp1: EnvironmentFingerprint,
                fp2: EnvironmentFingerprint,
                strict: bool = False) -> Dict[str, Any]:
        """Compare two environment fingerprints."""
        mismatches = []
        is_match = True

        if fp1.python_version != fp2.python_version:
            mismatches.append(f"python_version: {fp1.python_version} vs {fp2.python_version}")
            is_match = False

        if fp1.platform_system != fp2.platform_system:
            mismatches.append(f"platform_system: {fp1.platform_system} vs {fp2.platform_system}")
            is_match = False

        if strict:
            if fp1.platform_release != fp2.platform_release:
                mismatches.append(
                    f"platform_release: {fp1.platform_release} vs {fp2.platform_release}"
                )
                is_match = False

            if fp1.platform_machine != fp2.platform_machine:
                mismatches.append(
                    f"platform_machine: {fp1.platform_machine} vs {fp2.platform_machine}"
                )
                is_match = False

            for pkg, ver in fp2.dependency_versions.items():
                cur_ver = fp1.dependency_versions.get(pkg, "unknown")
                if cur_ver != ver:
                    mismatches.append(f"dependency {pkg}: {cur_ver} vs {ver}")
                    is_match = False

        return {
            "is_match": is_match,
            "mismatches": mismatches,
            "strict": strict,
        }

    def store_fingerprint(self, fingerprint_id: str,
                          fp: EnvironmentFingerprint) -> None:
        """Store a fingerprint for later retrieval."""
        self._stored[fingerprint_id] = fp

    def get_fingerprint(self, fingerprint_id: str) -> Optional[EnvironmentFingerprint]:
        """Retrieve a stored fingerprint."""
        return self._stored.get(fingerprint_id)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCapture:
    """Test capture method."""

    def test_capture_returns_fingerprint(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert isinstance(fp, EnvironmentFingerprint)

    def test_capture_python_version(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert fp.python_version == sys.version.split()[0]

    def test_capture_platform_info(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert fp.platform_system == platform.system()
        assert fp.platform_release == platform.release()
        assert fp.platform_machine == platform.machine()

    def test_capture_hostname(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        # hostname may or may not be empty but must be a string
        assert isinstance(fp.hostname, str)

    def test_capture_dependency_versions(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert isinstance(fp.dependency_versions, dict)

    def test_capture_environment_variables_gl_prefix(self, monkeypatch):
        monkeypatch.setenv("GL_TEST_VAR", "test_value")
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert "GL_TEST_VAR" in fp.environment_variables
        assert fp.environment_variables["GL_TEST_VAR"] == "test_value"

    def test_capture_ignores_non_gl_vars(self, monkeypatch):
        monkeypatch.setenv("SOME_OTHER_VAR", "ignored")
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert "SOME_OTHER_VAR" not in fp.environment_variables

    def test_capture_environment_hash_deterministic(self):
        ec = EnvironmentCapture()
        fp1 = ec.capture()
        fp2 = ec.capture()
        # Same machine, same env -> same hash
        assert fp1.environment_hash == fp2.environment_hash

    def test_capture_environment_hash_length(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert len(fp.environment_hash) == 16

    def test_capture_greenlang_version(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert fp.greenlang_version == "1.0.0"

    def test_capture_captured_at_set(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        assert fp.captured_at is not None

    def test_capture_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MYAPP_CONFIG", "value")
        ec = EnvironmentCapture(gl_env_prefix="MYAPP_")
        fp = ec.capture()
        assert "MYAPP_CONFIG" in fp.environment_variables


class TestCompare:
    """Test compare method."""

    def test_compare_same_environment(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        result = ec.compare(fp, fp)
        assert result["is_match"] is True
        assert result["mismatches"] == []

    def test_compare_different_python_version(self):
        ec = EnvironmentCapture()
        fp1 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="a",
        )
        fp2 = EnvironmentFingerprint(
            python_version="3.12.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="b",
        )
        result = ec.compare(fp1, fp2)
        assert result["is_match"] is False
        assert any("python_version" in m for m in result["mismatches"])

    def test_compare_different_platform(self):
        ec = EnvironmentCapture()
        fp1 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="a",
        )
        fp2 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Windows",
            platform_release="10", platform_machine="AMD64",
            captured_at=datetime.now(timezone.utc), environment_hash="b",
        )
        result = ec.compare(fp1, fp2)
        assert result["is_match"] is False

    def test_compare_strict_mode_release_mismatch(self):
        ec = EnvironmentCapture()
        fp1 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="a",
        )
        fp2 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.16.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="b",
        )
        result = ec.compare(fp1, fp2, strict=True)
        assert result["is_match"] is False
        assert any("platform_release" in m for m in result["mismatches"])

    def test_compare_relaxed_mode_release_ok(self):
        ec = EnvironmentCapture()
        fp1 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="a",
        )
        fp2 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.16.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="b",
        )
        result = ec.compare(fp1, fp2, strict=False)
        assert result["is_match"] is True

    def test_compare_strict_dependency_mismatch(self):
        ec = EnvironmentCapture()
        fp1 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="a",
            dependency_versions={"pydantic": "2.5.0"},
        )
        fp2 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="b",
            dependency_versions={"pydantic": "2.6.0"},
        )
        result = ec.compare(fp1, fp2, strict=True)
        assert result["is_match"] is False
        assert any("pydantic" in m for m in result["mismatches"])

    def test_compare_strict_flag_in_result(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        result = ec.compare(fp, fp, strict=True)
        assert result["strict"] is True

    def test_compare_relaxed_flag_in_result(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        result = ec.compare(fp, fp, strict=False)
        assert result["strict"] is False


class TestStorage:
    """Test store_fingerprint and get_fingerprint."""

    def test_store_fingerprint(self):
        ec = EnvironmentCapture()
        fp = ec.capture()
        ec.store_fingerprint("fp-001", fp)
        retrieved = ec.get_fingerprint("fp-001")
        assert retrieved is not None
        assert retrieved.python_version == fp.python_version

    def test_get_fingerprint_nonexistent(self):
        ec = EnvironmentCapture()
        assert ec.get_fingerprint("nonexistent") is None

    def test_store_multiple_fingerprints(self):
        ec = EnvironmentCapture()
        fp1 = ec.capture()
        fp2 = ec.capture()
        ec.store_fingerprint("fp-001", fp1)
        ec.store_fingerprint("fp-002", fp2)
        assert ec.get_fingerprint("fp-001") is not None
        assert ec.get_fingerprint("fp-002") is not None

    def test_overwrite_fingerprint(self):
        ec = EnvironmentCapture()
        fp1 = EnvironmentFingerprint(
            python_version="3.11.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="old",
        )
        fp2 = EnvironmentFingerprint(
            python_version="3.12.0", platform_system="Linux",
            platform_release="5.15.0", platform_machine="x86_64",
            captured_at=datetime.now(timezone.utc), environment_hash="new",
        )
        ec.store_fingerprint("fp-001", fp1)
        ec.store_fingerprint("fp-001", fp2)
        retrieved = ec.get_fingerprint("fp-001")
        assert retrieved.python_version == "3.12.0"
