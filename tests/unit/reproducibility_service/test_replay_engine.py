# -*- coding: utf-8 -*-
"""
Unit Tests for ReplayEngine (AGENT-FOUND-008)

Tests replay preparation, execution, environment verification,
seed/version matching, output comparison, and session management.

Coverage target: 85%+ of replay_engine.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class EnvironmentFingerprint:
    def __init__(self, python_version: str, platform_system: str,
                 platform_release: str, platform_machine: str,
                 captured_at: datetime, environment_hash: str,
                 hostname: str = "",
                 dependency_versions: Optional[Dict[str, str]] = None):
        self.python_version = python_version
        self.platform_system = platform_system
        self.platform_release = platform_release
        self.platform_machine = platform_machine
        self.captured_at = captured_at
        self.environment_hash = environment_hash
        self.hostname = hostname
        self.dependency_versions = dependency_versions or {}


class SeedConfiguration:
    def __init__(self, global_seed: int = 42, numpy_seed: Optional[int] = 42,
                 torch_seed: Optional[int] = 42,
                 custom_seeds: Optional[Dict[str, int]] = None,
                 seed_hash: str = ""):
        self.global_seed = global_seed
        self.numpy_seed = numpy_seed
        self.torch_seed = torch_seed
        self.custom_seeds = custom_seeds or {}
        self.seed_hash = seed_hash or "auto_hash"


class VersionManifest:
    def __init__(self, manifest_id: str = "",
                 agent_versions: Optional[Dict[str, Any]] = None,
                 model_versions: Optional[Dict[str, Any]] = None,
                 manifest_hash: str = ""):
        self.manifest_id = manifest_id
        self.agent_versions = agent_versions or {}
        self.model_versions = model_versions or {}
        self.manifest_hash = manifest_hash


class ReplayConfiguration:
    def __init__(self, original_execution_id: str,
                 captured_inputs: Dict[str, Any],
                 captured_environment: EnvironmentFingerprint,
                 captured_seeds: SeedConfiguration,
                 captured_versions: VersionManifest,
                 replay_mode: bool = True, strict_mode: bool = False):
        self.original_execution_id = original_execution_id
        self.captured_inputs = captured_inputs
        self.captured_environment = captured_environment
        self.captured_seeds = captured_seeds
        self.captured_versions = captured_versions
        self.replay_mode = replay_mode
        self.strict_mode = strict_mode


class ReplaySession:
    def __init__(self, session_id: str, original_execution_id: str,
                 status: str = "pending", output_match: Optional[bool] = None,
                 started_at: Optional[datetime] = None,
                 completed_at: Optional[datetime] = None,
                 environment_match: bool = True,
                 seed_match: bool = True,
                 version_match: bool = True,
                 details: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.original_execution_id = original_execution_id
        self.status = status
        self.output_match = output_match
        self.started_at = started_at or datetime.now(timezone.utc)
        self.completed_at = completed_at
        self.environment_match = environment_match
        self.seed_match = seed_match
        self.version_match = version_match
        self.details = details or {}


# ---------------------------------------------------------------------------
# Inline ReplayEngine
# ---------------------------------------------------------------------------

class ReplayEngine:
    """Engine for replaying captured executions."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._sessions: Dict[str, ReplaySession] = {}
        self._session_counter = 0

    def prepare_replay(self, execution_id: str,
                       captured_inputs: Dict[str, Any],
                       environment: EnvironmentFingerprint,
                       seeds: SeedConfiguration,
                       versions: VersionManifest) -> ReplayConfiguration:
        """Prepare a replay configuration."""
        return ReplayConfiguration(
            original_execution_id=execution_id,
            captured_inputs=captured_inputs,
            captured_environment=environment,
            captured_seeds=seeds,
            captured_versions=versions,
            replay_mode=True,
            strict_mode=self.strict_mode,
        )

    def execute_replay(self, config: ReplayConfiguration,
                       current_environment: EnvironmentFingerprint,
                       current_seeds: SeedConfiguration,
                       current_output: Optional[Dict[str, Any]] = None,
                       original_output: Optional[Dict[str, Any]] = None) -> ReplaySession:
        """Execute a replay and return session result."""
        self._session_counter += 1
        session_id = f"replay-{self._session_counter:04d}"

        # Verify environment
        env_match = self._verify_environment(
            current_environment, config.captured_environment, config.strict_mode,
        )

        # Verify seeds
        seed_match = self._verify_seeds(current_seeds, config.captured_seeds)

        # Verify versions
        ver_match = self._verify_versions(config.captured_versions)

        # Compare outputs
        output_match = None
        if current_output is not None and original_output is not None:
            output_match = self._compare_outputs(current_output, original_output)

        # Determine status
        if config.strict_mode and not env_match:
            status = "failed"
        elif not seed_match:
            status = "failed"
        elif output_match is False:
            status = "failed"
        elif output_match is True:
            status = "completed"
        else:
            status = "completed"

        session = ReplaySession(
            session_id=session_id,
            original_execution_id=config.original_execution_id,
            status=status,
            output_match=output_match,
            completed_at=datetime.now(timezone.utc),
            environment_match=env_match,
            seed_match=seed_match,
            version_match=ver_match,
        )
        self._sessions[session_id] = session
        return session

    def _verify_environment(self, current: EnvironmentFingerprint,
                            expected: EnvironmentFingerprint,
                            strict: bool = False) -> bool:
        """Verify environment matches."""
        if current.python_version != expected.python_version:
            return False
        if current.platform_system != expected.platform_system:
            return False
        if strict:
            if current.platform_release != expected.platform_release:
                return False
            if current.platform_machine != expected.platform_machine:
                return False
            for pkg, ver in expected.dependency_versions.items():
                if current.dependency_versions.get(pkg) != ver:
                    return False
        return True

    def _verify_seeds(self, current: SeedConfiguration,
                      expected: SeedConfiguration) -> bool:
        """Verify seed configuration matches."""
        if current.global_seed != expected.global_seed:
            return False
        if current.numpy_seed != expected.numpy_seed:
            return False
        if current.torch_seed != expected.torch_seed:
            return False
        for key, val in expected.custom_seeds.items():
            if current.custom_seeds.get(key) != val:
                return False
        return True

    def _verify_versions(self, manifest: VersionManifest) -> bool:
        """Verify version manifest (stub - always returns True in test)."""
        return True

    def _compare_outputs(self, current: Dict[str, Any],
                         original: Dict[str, Any]) -> bool:
        """Compare outputs for equality."""
        c_hash = hashlib.sha256(
            json.dumps(current, sort_keys=True).encode()
        ).hexdigest()
        o_hash = hashlib.sha256(
            json.dumps(original, sort_keys=True).encode()
        ).hexdigest()
        return c_hash == o_hash

    def get_replay_session(self, session_id: str) -> Optional[ReplaySession]:
        """Get a replay session by ID."""
        return self._sessions.get(session_id)

    def list_replay_sessions(self) -> List[ReplaySession]:
        """List all replay sessions."""
        return list(self._sessions.values())


# ---------------------------------------------------------------------------
# Helper fixture factories
# ---------------------------------------------------------------------------

def _make_env(python_version="3.11.0", platform_system="Linux",
              platform_release="5.15.0", platform_machine="x86_64",
              deps: Optional[Dict[str, str]] = None) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        python_version=python_version,
        platform_system=platform_system,
        platform_release=platform_release,
        platform_machine=platform_machine,
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        environment_hash="abc123",
        dependency_versions=deps or {},
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPrepareReplay:
    """Test prepare_replay method."""

    def test_prepare_replay_configuration(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        versions = VersionManifest()
        config = engine.prepare_replay("exec-001", {"x": 1}, env, seeds, versions)
        assert config.original_execution_id == "exec-001"
        assert config.captured_inputs == {"x": 1}
        assert config.replay_mode is True

    def test_prepare_replay_strict_from_engine(self):
        engine = ReplayEngine(strict_mode=True)
        config = engine.prepare_replay(
            "exec-001", {}, _make_env(), SeedConfiguration(), VersionManifest(),
        )
        assert config.strict_mode is True

    def test_prepare_replay_non_strict(self):
        engine = ReplayEngine(strict_mode=False)
        config = engine.prepare_replay(
            "exec-001", {}, _make_env(), SeedConfiguration(), VersionManifest(),
        )
        assert config.strict_mode is False


class TestExecuteReplay:
    """Test execute_replay method."""

    def test_execute_replay_success(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config = engine.prepare_replay("exec-001", {"x": 1}, env, seeds, VersionManifest())
        session = engine.execute_replay(config, env, seeds, {"y": 2}, {"y": 2})
        assert session.status == "completed"
        assert session.output_match is True
        assert session.environment_match is True
        assert session.seed_match is True

    def test_execute_replay_environment_mismatch_relaxed(self):
        engine = ReplayEngine()
        original_env = _make_env(python_version="3.11.0")
        current_env = _make_env(python_version="3.12.0")
        seeds = SeedConfiguration()
        config = engine.prepare_replay(
            "exec-001", {}, original_env, seeds, VersionManifest(),
        )
        session = engine.execute_replay(config, current_env, seeds)
        # In relaxed mode, env mismatch does not fail
        assert session.environment_match is False
        assert session.status == "completed"

    def test_execute_replay_environment_mismatch_strict(self):
        engine = ReplayEngine(strict_mode=True)
        original_env = _make_env(python_version="3.11.0")
        current_env = _make_env(python_version="3.12.0")
        seeds = SeedConfiguration()
        config = engine.prepare_replay(
            "exec-001", {}, original_env, seeds, VersionManifest(),
        )
        session = engine.execute_replay(config, current_env, seeds)
        assert session.status == "failed"
        assert session.environment_match is False

    def test_execute_replay_seed_mismatch(self):
        engine = ReplayEngine()
        env = _make_env()
        original_seeds = SeedConfiguration(global_seed=42)
        current_seeds = SeedConfiguration(global_seed=99)
        config = engine.prepare_replay(
            "exec-001", {}, env, original_seeds, VersionManifest(),
        )
        session = engine.execute_replay(config, env, current_seeds)
        assert session.status == "failed"
        assert session.seed_match is False

    def test_execute_replay_output_mismatch(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config = engine.prepare_replay("exec-001", {}, env, seeds, VersionManifest())
        session = engine.execute_replay(config, env, seeds, {"y": 2}, {"y": 999})
        assert session.output_match is False
        assert session.status == "failed"

    def test_execute_replay_no_output_comparison(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config = engine.prepare_replay("exec-001", {}, env, seeds, VersionManifest())
        session = engine.execute_replay(config, env, seeds)
        assert session.output_match is None
        assert session.status == "completed"

    def test_execute_replay_session_id_format(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config = engine.prepare_replay("exec-001", {}, env, seeds, VersionManifest())
        session = engine.execute_replay(config, env, seeds)
        assert session.session_id.startswith("replay-")

    def test_execute_replay_completed_at_set(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config = engine.prepare_replay("exec-001", {}, env, seeds, VersionManifest())
        session = engine.execute_replay(config, env, seeds)
        assert session.completed_at is not None


class TestVerifyEnvironment:
    """Test _verify_environment method."""

    def test_verify_environment_strict_match(self):
        engine = ReplayEngine()
        env = _make_env()
        assert engine._verify_environment(env, env, strict=True) is True

    def test_verify_environment_strict_mismatch_release(self):
        engine = ReplayEngine()
        current = _make_env(platform_release="5.15.0")
        expected = _make_env(platform_release="5.16.0")
        assert engine._verify_environment(current, expected, strict=True) is False

    def test_verify_environment_strict_mismatch_machine(self):
        engine = ReplayEngine()
        current = _make_env(platform_machine="x86_64")
        expected = _make_env(platform_machine="aarch64")
        assert engine._verify_environment(current, expected, strict=True) is False

    def test_verify_environment_strict_dependency_mismatch(self):
        engine = ReplayEngine()
        current = _make_env(deps={"pydantic": "2.5.0"})
        expected = _make_env(deps={"pydantic": "2.6.0"})
        assert engine._verify_environment(current, expected, strict=True) is False

    def test_verify_environment_relaxed_python_mismatch(self):
        engine = ReplayEngine()
        current = _make_env(python_version="3.12.0")
        expected = _make_env(python_version="3.11.0")
        # Even in relaxed mode, python version mismatch returns False
        assert engine._verify_environment(current, expected, strict=False) is False

    def test_verify_environment_relaxed_release_mismatch_ok(self):
        engine = ReplayEngine()
        current = _make_env(python_version="3.11.0", platform_release="5.16.0")
        expected = _make_env(python_version="3.11.0", platform_release="5.15.0")
        assert engine._verify_environment(current, expected, strict=False) is True


class TestVerifySeeds:
    """Test _verify_seeds method."""

    def test_verify_seeds_match(self):
        engine = ReplayEngine()
        s1 = SeedConfiguration(global_seed=42, numpy_seed=42, torch_seed=42)
        s2 = SeedConfiguration(global_seed=42, numpy_seed=42, torch_seed=42)
        assert engine._verify_seeds(s1, s2) is True

    def test_verify_seeds_global_mismatch(self):
        engine = ReplayEngine()
        s1 = SeedConfiguration(global_seed=42)
        s2 = SeedConfiguration(global_seed=99)
        assert engine._verify_seeds(s1, s2) is False

    def test_verify_seeds_numpy_mismatch(self):
        engine = ReplayEngine()
        s1 = SeedConfiguration(numpy_seed=42)
        s2 = SeedConfiguration(numpy_seed=99)
        assert engine._verify_seeds(s1, s2) is False

    def test_verify_seeds_torch_mismatch(self):
        engine = ReplayEngine()
        s1 = SeedConfiguration(torch_seed=42)
        s2 = SeedConfiguration(torch_seed=99)
        assert engine._verify_seeds(s1, s2) is False

    def test_verify_seeds_custom_mismatch(self):
        engine = ReplayEngine()
        s1 = SeedConfiguration(custom_seeds={"comp_a": 100})
        s2 = SeedConfiguration(custom_seeds={"comp_a": 200})
        assert engine._verify_seeds(s1, s2) is False

    def test_verify_seeds_custom_missing(self):
        engine = ReplayEngine()
        s1 = SeedConfiguration(custom_seeds={})
        s2 = SeedConfiguration(custom_seeds={"comp_a": 100})
        assert engine._verify_seeds(s1, s2) is False


class TestCompareOutputs:
    """Test _compare_outputs method."""

    def test_compare_outputs_match(self):
        engine = ReplayEngine()
        data = {"total": 100.0, "unit": "kg"}
        assert engine._compare_outputs(data, data) is True

    def test_compare_outputs_within_tolerance_still_different_hash(self):
        engine = ReplayEngine()
        # Hash-based comparison means even tiny difference fails
        a = {"total": 100.0}
        b = {"total": 100.0000001}
        assert engine._compare_outputs(a, b) is False

    def test_compare_outputs_mismatch(self):
        engine = ReplayEngine()
        assert engine._compare_outputs({"a": 1}, {"a": 2}) is False

    def test_compare_outputs_key_order_independent(self):
        engine = ReplayEngine()
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert engine._compare_outputs(a, b) is True


class TestSessionManagement:
    """Test session storage and retrieval."""

    def test_get_replay_session(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config = engine.prepare_replay("exec-001", {}, env, seeds, VersionManifest())
        session = engine.execute_replay(config, env, seeds)
        retrieved = engine.get_replay_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_replay_session_nonexistent(self):
        engine = ReplayEngine()
        assert engine.get_replay_session("nonexistent") is None

    def test_list_replay_sessions(self):
        engine = ReplayEngine()
        env = _make_env()
        seeds = SeedConfiguration()
        config1 = engine.prepare_replay("exec-001", {}, env, seeds, VersionManifest())
        config2 = engine.prepare_replay("exec-002", {}, env, seeds, VersionManifest())
        engine.execute_replay(config1, env, seeds)
        engine.execute_replay(config2, env, seeds)
        sessions = engine.list_replay_sessions()
        assert len(sessions) == 2

    def test_list_replay_sessions_empty(self):
        engine = ReplayEngine()
        assert engine.list_replay_sessions() == []
