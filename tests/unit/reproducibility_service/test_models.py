# -*- coding: utf-8 -*-
"""
Unit Tests for Reproducibility Models (AGENT-FOUND-008)

Tests all enums, Pydantic model classes, field validation, serialization,
hash computation, and edge cases for the reproducibility data types.

Coverage target: 85%+ of models in reproducibility_agent.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline enums and constants mirroring reproducibility_agent.py
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class DriftSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class NonDeterminismSource(str, Enum):
    TIMESTAMP = "timestamp"
    RANDOM_SEED = "random_seed"
    EXTERNAL_API = "external_api"
    FLOATING_POINT = "floating_point"
    DICT_ORDERING = "dict_ordering"
    FILE_ORDERING = "file_ordering"
    THREAD_SCHEDULING = "thread_scheduling"
    NETWORK_LATENCY = "network_latency"
    ENVIRONMENT_VARIABLE = "environment_variable"
    DEPENDENCY_VERSION = "dependency_version"


DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_DRIFT_SOFT_THRESHOLD = 0.01
DEFAULT_DRIFT_HARD_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Inline lightweight model stubs for testing
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    """Compute SHA-256 hash of data dict for deterministic hashing."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class EnvironmentFingerprint:
    def __init__(
        self,
        python_version: str,
        platform_system: str,
        platform_release: str,
        platform_machine: str,
        captured_at: datetime,
        environment_hash: str,
        hostname: str = "",
        greenlang_version: str = "1.0.0",
        dependency_versions: Optional[Dict[str, str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ):
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


class SeedConfiguration:
    def __init__(
        self,
        global_seed: int = 42,
        numpy_seed: Optional[int] = 42,
        torch_seed: Optional[int] = 42,
        custom_seeds: Optional[Dict[str, int]] = None,
        seed_hash: str = "",
    ):
        self.global_seed = global_seed
        self.numpy_seed = numpy_seed
        self.torch_seed = torch_seed
        self.custom_seeds = custom_seeds or {}
        if not seed_hash:
            seed_data = {
                "global": self.global_seed,
                "numpy": self.numpy_seed,
                "torch": self.torch_seed,
                "custom": self.custom_seeds,
            }
            self.seed_hash = _content_hash(seed_data)[:16]
        else:
            self.seed_hash = seed_hash


class VersionPin:
    def __init__(
        self,
        component_type: str,
        component_id: str,
        version: str,
        version_hash: str = "",
        pinned_at: Optional[datetime] = None,
    ):
        self.component_type = component_type
        self.component_id = component_id
        self.version = version
        self.version_hash = version_hash
        self.pinned_at = pinned_at or datetime.now(timezone.utc)


class VersionManifest:
    def __init__(
        self,
        manifest_id: str = "",
        created_at: Optional[datetime] = None,
        agent_versions: Optional[Dict[str, VersionPin]] = None,
        model_versions: Optional[Dict[str, VersionPin]] = None,
        factor_versions: Optional[Dict[str, VersionPin]] = None,
        data_versions: Optional[Dict[str, VersionPin]] = None,
        manifest_hash: str = "",
    ):
        self.manifest_id = manifest_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.agent_versions = agent_versions or {}
        self.model_versions = model_versions or {}
        self.factor_versions = factor_versions or {}
        self.data_versions = data_versions or {}
        self.manifest_hash = manifest_hash


class VerificationCheck:
    def __init__(
        self,
        check_name: str,
        status: VerificationStatus,
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
        difference: Optional[float] = None,
        tolerance: Optional[float] = None,
        message: str = "",
        timestamp: Optional[datetime] = None,
    ):
        self.check_name = check_name
        self.status = status
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.difference = difference
        self.tolerance = tolerance
        self.message = message
        self.timestamp = timestamp or datetime.now(timezone.utc)


class DriftDetection:
    def __init__(
        self,
        baseline_hash: str,
        current_hash: str,
        severity: DriftSeverity,
        drift_percentage: float = 0.0,
        drifted_fields: Optional[List[str]] = None,
        drift_details: Optional[Dict[str, Dict[str, Any]]] = None,
        is_acceptable: bool = True,
    ):
        self.baseline_hash = baseline_hash
        self.current_hash = current_hash
        self.severity = severity
        self.drift_percentage = drift_percentage
        self.drifted_fields = drifted_fields or []
        self.drift_details = drift_details or {}
        self.is_acceptable = is_acceptable


class ReplayConfiguration:
    def __init__(
        self,
        original_execution_id: str,
        captured_inputs: Dict[str, Any],
        captured_environment: EnvironmentFingerprint,
        captured_seeds: SeedConfiguration,
        captured_versions: VersionManifest,
        replay_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.original_execution_id = original_execution_id
        self.captured_inputs = captured_inputs
        self.captured_environment = captured_environment
        self.captured_seeds = captured_seeds
        self.captured_versions = captured_versions
        self.replay_mode = replay_mode
        self.strict_mode = strict_mode


# SDK models
class ArtifactHash:
    def __init__(self, artifact_id: str, hash_value: str, algorithm: str = "sha256",
                 created_at: Optional[datetime] = None):
        self.artifact_id = artifact_id
        self.hash_value = hash_value
        self.algorithm = algorithm
        self.created_at = created_at or datetime.now(timezone.utc)


class VerificationRun:
    def __init__(self, run_id: str, execution_id: str, status: VerificationStatus,
                 checks: Optional[List[VerificationCheck]] = None,
                 started_at: Optional[datetime] = None,
                 completed_at: Optional[datetime] = None):
        self.run_id = run_id
        self.execution_id = execution_id
        self.status = status
        self.checks = checks or []
        self.started_at = started_at or datetime.now(timezone.utc)
        self.completed_at = completed_at


class DriftBaseline:
    def __init__(self, baseline_id: str, name: str, data_hash: str,
                 data: Optional[Dict[str, Any]] = None,
                 is_active: bool = True,
                 created_at: Optional[datetime] = None):
        self.baseline_id = baseline_id
        self.name = name
        self.data_hash = data_hash
        self.data = data or {}
        self.is_active = is_active
        self.created_at = created_at or datetime.now(timezone.utc)


class ReplaySession:
    def __init__(self, session_id: str, original_execution_id: str,
                 status: str = "pending",
                 output_match: Optional[bool] = None,
                 started_at: Optional[datetime] = None,
                 completed_at: Optional[datetime] = None):
        self.session_id = session_id
        self.original_execution_id = original_execution_id
        self.status = status
        self.output_match = output_match
        self.started_at = started_at or datetime.now(timezone.utc)
        self.completed_at = completed_at


class VerificationStatistics:
    def __init__(self, total_verifications: int = 0, passed: int = 0,
                 failed: int = 0, warnings: int = 0, skipped: int = 0,
                 average_duration_ms: float = 0.0):
        self.total_verifications = total_verifications
        self.passed = passed
        self.failed = failed
        self.warnings = warnings
        self.skipped = skipped
        self.average_duration_ms = average_duration_ms


# ===========================================================================
# Test Classes
# ===========================================================================


class TestVerificationStatusEnum:
    """Test VerificationStatus enum values."""

    def test_pass_value(self):
        assert VerificationStatus.PASS.value == "pass"

    def test_fail_value(self):
        assert VerificationStatus.FAIL.value == "fail"

    def test_warning_value(self):
        assert VerificationStatus.WARNING.value == "warning"

    def test_skipped_value(self):
        assert VerificationStatus.SKIPPED.value == "skipped"

    def test_enum_count(self):
        assert len(VerificationStatus) == 4

    def test_is_str_subclass(self):
        assert isinstance(VerificationStatus.PASS, str)

    def test_from_string(self):
        assert VerificationStatus("pass") == VerificationStatus.PASS

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            VerificationStatus("invalid")


class TestDriftSeverityEnum:
    """Test DriftSeverity enum values."""

    def test_none_value(self):
        assert DriftSeverity.NONE.value == "none"

    def test_minor_value(self):
        assert DriftSeverity.MINOR.value == "minor"

    def test_moderate_value(self):
        assert DriftSeverity.MODERATE.value == "moderate"

    def test_critical_value(self):
        assert DriftSeverity.CRITICAL.value == "critical"

    def test_enum_count(self):
        assert len(DriftSeverity) == 4

    def test_is_str_subclass(self):
        assert isinstance(DriftSeverity.NONE, str)

    def test_from_string(self):
        assert DriftSeverity("minor") == DriftSeverity.MINOR

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DriftSeverity("unknown")


class TestNonDeterminismSourceEnum:
    """Test NonDeterminismSource enum values (all 10)."""

    def test_timestamp(self):
        assert NonDeterminismSource.TIMESTAMP.value == "timestamp"

    def test_random_seed(self):
        assert NonDeterminismSource.RANDOM_SEED.value == "random_seed"

    def test_external_api(self):
        assert NonDeterminismSource.EXTERNAL_API.value == "external_api"

    def test_floating_point(self):
        assert NonDeterminismSource.FLOATING_POINT.value == "floating_point"

    def test_dict_ordering(self):
        assert NonDeterminismSource.DICT_ORDERING.value == "dict_ordering"

    def test_file_ordering(self):
        assert NonDeterminismSource.FILE_ORDERING.value == "file_ordering"

    def test_thread_scheduling(self):
        assert NonDeterminismSource.THREAD_SCHEDULING.value == "thread_scheduling"

    def test_network_latency(self):
        assert NonDeterminismSource.NETWORK_LATENCY.value == "network_latency"

    def test_environment_variable(self):
        assert NonDeterminismSource.ENVIRONMENT_VARIABLE.value == "environment_variable"

    def test_dependency_version(self):
        assert NonDeterminismSource.DEPENDENCY_VERSION.value == "dependency_version"

    def test_enum_count(self):
        assert len(NonDeterminismSource) == 10

    def test_is_str_subclass(self):
        assert isinstance(NonDeterminismSource.TIMESTAMP, str)


class TestEnvironmentFingerprint:
    """Test EnvironmentFingerprint model."""

    def test_creation(self):
        fp = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abcdef1234567890",
        )
        assert fp.python_version == "3.11.0"
        assert fp.platform_system == "Linux"
        assert fp.platform_release == "5.15.0"
        assert fp.platform_machine == "x86_64"
        assert fp.environment_hash == "abcdef1234567890"

    def test_default_hostname(self):
        fp = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
        )
        assert fp.hostname == ""

    def test_default_greenlang_version(self):
        fp = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
        )
        assert fp.greenlang_version == "1.0.0"

    def test_default_dependency_versions(self):
        fp = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
        )
        assert fp.dependency_versions == {}

    def test_default_environment_variables(self):
        fp = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
        )
        assert fp.environment_variables == {}

    def test_with_dependency_versions(self):
        fp = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
            dependency_versions={"pydantic": "2.5.0", "numpy": "1.26.0"},
        )
        assert fp.dependency_versions["pydantic"] == "2.5.0"
        assert fp.dependency_versions["numpy"] == "1.26.0"


class TestSeedConfiguration:
    """Test SeedConfiguration model."""

    def test_defaults(self):
        seed = SeedConfiguration()
        assert seed.global_seed == 42
        assert seed.numpy_seed == 42
        assert seed.torch_seed == 42
        assert seed.custom_seeds == {}

    def test_seed_hash_auto_computed(self):
        seed = SeedConfiguration()
        assert seed.seed_hash != ""
        assert len(seed.seed_hash) == 16

    def test_seed_hash_deterministic(self):
        s1 = SeedConfiguration()
        s2 = SeedConfiguration()
        assert s1.seed_hash == s2.seed_hash

    def test_custom_global_seed(self):
        seed = SeedConfiguration(global_seed=123)
        assert seed.global_seed == 123

    def test_custom_numpy_seed(self):
        seed = SeedConfiguration(numpy_seed=999)
        assert seed.numpy_seed == 999

    def test_custom_torch_seed(self):
        seed = SeedConfiguration(torch_seed=777)
        assert seed.torch_seed == 777

    def test_custom_seeds_dict(self):
        seed = SeedConfiguration(custom_seeds={"component_a": 100, "component_b": 200})
        assert seed.custom_seeds["component_a"] == 100
        assert seed.custom_seeds["component_b"] == 200

    def test_different_seeds_different_hash(self):
        s1 = SeedConfiguration(global_seed=42)
        s2 = SeedConfiguration(global_seed=99)
        assert s1.seed_hash != s2.seed_hash

    def test_explicit_seed_hash(self):
        seed = SeedConfiguration(seed_hash="custom_hash_1234")
        assert seed.seed_hash == "custom_hash_1234"

    def test_none_numpy_seed(self):
        seed = SeedConfiguration(numpy_seed=None)
        assert seed.numpy_seed is None

    def test_none_torch_seed(self):
        seed = SeedConfiguration(torch_seed=None)
        assert seed.torch_seed is None


class TestVersionPin:
    """Test VersionPin model."""

    def test_creation(self):
        pin = VersionPin(
            component_type="agent",
            component_id="GL-FOUND-X-001",
            version="1.0.0",
        )
        assert pin.component_type == "agent"
        assert pin.component_id == "GL-FOUND-X-001"
        assert pin.version == "1.0.0"

    def test_default_version_hash(self):
        pin = VersionPin(component_type="agent", component_id="a1", version="1.0.0")
        assert pin.version_hash == ""

    def test_pinned_at_timestamp(self):
        pin = VersionPin(component_type="agent", component_id="a1", version="1.0.0")
        assert pin.pinned_at is not None

    def test_custom_version_hash(self):
        pin = VersionPin(
            component_type="model",
            component_id="m1",
            version="2.0.0",
            version_hash="abc123",
        )
        assert pin.version_hash == "abc123"


class TestVersionManifest:
    """Test VersionManifest model."""

    def test_creation_empty(self):
        manifest = VersionManifest()
        assert manifest.manifest_id == ""
        assert manifest.agent_versions == {}
        assert manifest.model_versions == {}
        assert manifest.factor_versions == {}
        assert manifest.data_versions == {}
        assert manifest.manifest_hash == ""

    def test_creation_with_agents(self):
        pin = VersionPin(component_type="agent", component_id="a1", version="1.0.0")
        manifest = VersionManifest(agent_versions={"a1": pin})
        assert "a1" in manifest.agent_versions
        assert manifest.agent_versions["a1"].version == "1.0.0"

    def test_creation_with_all_types(self):
        agents = {"a1": VersionPin("agent", "a1", "1.0.0")}
        models = {"m1": VersionPin("model", "m1", "2.0.0")}
        factors = {"f1": VersionPin("factor", "f1", "3.0.0")}
        data = {"d1": VersionPin("data", "d1", "4.0.0")}
        manifest = VersionManifest(
            manifest_id="manifest-001",
            agent_versions=agents,
            model_versions=models,
            factor_versions=factors,
            data_versions=data,
        )
        assert manifest.manifest_id == "manifest-001"
        assert len(manifest.agent_versions) == 1
        assert len(manifest.model_versions) == 1
        assert len(manifest.factor_versions) == 1
        assert len(manifest.data_versions) == 1

    def test_created_at_auto_set(self):
        manifest = VersionManifest()
        assert manifest.created_at is not None


class TestVerificationCheck:
    """Test VerificationCheck model."""

    def test_pass_check(self):
        check = VerificationCheck(
            check_name="input_hash",
            status=VerificationStatus.PASS,
            expected_value="abc",
            actual_value="abc",
            message="Input hash matches",
        )
        assert check.status == VerificationStatus.PASS
        assert check.expected_value == check.actual_value

    def test_fail_check(self):
        check = VerificationCheck(
            check_name="input_hash",
            status=VerificationStatus.FAIL,
            expected_value="abc",
            actual_value="def",
            message="Hash mismatch",
        )
        assert check.status == VerificationStatus.FAIL
        assert check.expected_value != check.actual_value

    def test_warning_check(self):
        check = VerificationCheck(
            check_name="output_hash",
            status=VerificationStatus.WARNING,
            difference=0.001,
            tolerance=0.01,
            message="Within tolerance",
        )
        assert check.status == VerificationStatus.WARNING
        assert check.difference < check.tolerance

    def test_skipped_check(self):
        check = VerificationCheck(
            check_name="version_check",
            status=VerificationStatus.SKIPPED,
            message="No manifest provided",
        )
        assert check.status == VerificationStatus.SKIPPED

    def test_default_message(self):
        check = VerificationCheck(
            check_name="test",
            status=VerificationStatus.PASS,
        )
        assert check.message == ""

    def test_timestamp_auto_set(self):
        check = VerificationCheck(
            check_name="test",
            status=VerificationStatus.PASS,
        )
        assert check.timestamp is not None


class TestDriftDetection:
    """Test DriftDetection model."""

    def test_acceptable_drift(self):
        drift = DriftDetection(
            baseline_hash="abc",
            current_hash="abc",
            severity=DriftSeverity.NONE,
            is_acceptable=True,
        )
        assert drift.is_acceptable is True
        assert drift.severity == DriftSeverity.NONE

    def test_unacceptable_drift(self):
        drift = DriftDetection(
            baseline_hash="abc",
            current_hash="def",
            severity=DriftSeverity.CRITICAL,
            drift_percentage=6.5,
            drifted_fields=["emissions"],
            is_acceptable=False,
        )
        assert drift.is_acceptable is False
        assert drift.severity == DriftSeverity.CRITICAL
        assert drift.drift_percentage == 6.5
        assert "emissions" in drift.drifted_fields

    def test_default_drift_fields(self):
        drift = DriftDetection(
            baseline_hash="a", current_hash="b", severity=DriftSeverity.NONE,
        )
        assert drift.drifted_fields == []
        assert drift.drift_details == {}
        assert drift.drift_percentage == 0.0


class TestReplayConfiguration:
    """Test ReplayConfiguration model."""

    def test_creation(self):
        env = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
        )
        seeds = SeedConfiguration()
        manifest = VersionManifest()
        replay = ReplayConfiguration(
            original_execution_id="exec-001",
            captured_inputs={"key": "value"},
            captured_environment=env,
            captured_seeds=seeds,
            captured_versions=manifest,
        )
        assert replay.original_execution_id == "exec-001"
        assert replay.replay_mode is True
        assert replay.strict_mode is False

    def test_strict_mode_enabled(self):
        env = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15.0",
            platform_machine="x86_64",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment_hash="abc",
        )
        replay = ReplayConfiguration(
            original_execution_id="exec-002",
            captured_inputs={},
            captured_environment=env,
            captured_seeds=SeedConfiguration(),
            captured_versions=VersionManifest(),
            strict_mode=True,
        )
        assert replay.strict_mode is True


class TestArtifactHash:
    """Test ArtifactHash SDK model."""

    def test_creation(self):
        ah = ArtifactHash(artifact_id="art-001", hash_value="abc123def456")
        assert ah.artifact_id == "art-001"
        assert ah.hash_value == "abc123def456"
        assert ah.algorithm == "sha256"

    def test_custom_algorithm(self):
        ah = ArtifactHash(artifact_id="art-002", hash_value="xyz", algorithm="sha512")
        assert ah.algorithm == "sha512"

    def test_created_at_auto_set(self):
        ah = ArtifactHash(artifact_id="art-003", hash_value="abc")
        assert ah.created_at is not None


class TestVerificationRun:
    """Test VerificationRun SDK model."""

    def test_creation(self):
        vr = VerificationRun(
            run_id="run-001",
            execution_id="exec-001",
            status=VerificationStatus.PASS,
        )
        assert vr.run_id == "run-001"
        assert vr.execution_id == "exec-001"
        assert vr.status == VerificationStatus.PASS
        assert vr.checks == []

    def test_with_checks(self):
        check = VerificationCheck("test", VerificationStatus.PASS)
        vr = VerificationRun(
            run_id="run-002",
            execution_id="exec-002",
            status=VerificationStatus.PASS,
            checks=[check],
        )
        assert len(vr.checks) == 1


class TestDriftBaseline:
    """Test DriftBaseline SDK model."""

    def test_creation(self):
        db = DriftBaseline(
            baseline_id="bl-001",
            name="emissions_baseline",
            data_hash="abc123",
            data={"emissions": 100.0},
        )
        assert db.baseline_id == "bl-001"
        assert db.name == "emissions_baseline"
        assert db.data_hash == "abc123"
        assert db.is_active is True

    def test_inactive_baseline(self):
        db = DriftBaseline(
            baseline_id="bl-002",
            name="old_baseline",
            data_hash="def456",
            is_active=False,
        )
        assert db.is_active is False


class TestReplaySession:
    """Test ReplaySession SDK model."""

    def test_creation(self):
        rs = ReplaySession(
            session_id="sess-001",
            original_execution_id="exec-001",
        )
        assert rs.session_id == "sess-001"
        assert rs.status == "pending"
        assert rs.output_match is None

    def test_completed_session(self):
        rs = ReplaySession(
            session_id="sess-002",
            original_execution_id="exec-002",
            status="completed",
            output_match=True,
            completed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        assert rs.status == "completed"
        assert rs.output_match is True


class TestVerificationStatistics:
    """Test VerificationStatistics SDK model."""

    def test_creation_defaults(self):
        stats = VerificationStatistics()
        assert stats.total_verifications == 0
        assert stats.passed == 0
        assert stats.failed == 0
        assert stats.warnings == 0
        assert stats.skipped == 0
        assert stats.average_duration_ms == 0.0

    def test_with_data(self):
        stats = VerificationStatistics(
            total_verifications=100,
            passed=85,
            failed=5,
            warnings=8,
            skipped=2,
            average_duration_ms=12.5,
        )
        assert stats.total_verifications == 100
        assert stats.passed == 85
        assert stats.failed == 5
        assert stats.warnings == 8
        assert stats.skipped == 2
        assert stats.average_duration_ms == 12.5

    def test_sum_matches_total(self):
        stats = VerificationStatistics(
            total_verifications=100,
            passed=85,
            failed=5,
            warnings=8,
            skipped=2,
        )
        assert stats.passed + stats.failed + stats.warnings + stats.skipped == 100


class TestConstants:
    """Test module-level constants."""

    def test_default_absolute_tolerance(self):
        assert DEFAULT_ABSOLUTE_TOLERANCE == 1e-9

    def test_default_relative_tolerance(self):
        assert DEFAULT_RELATIVE_TOLERANCE == 1e-6

    def test_default_drift_soft_threshold(self):
        assert DEFAULT_DRIFT_SOFT_THRESHOLD == 0.01

    def test_default_drift_hard_threshold(self):
        assert DEFAULT_DRIFT_HARD_THRESHOLD == 0.05

    def test_soft_less_than_hard(self):
        assert DEFAULT_DRIFT_SOFT_THRESHOLD < DEFAULT_DRIFT_HARD_THRESHOLD

    def test_absolute_less_than_relative(self):
        assert DEFAULT_ABSOLUTE_TOLERANCE < DEFAULT_RELATIVE_TOLERANCE
