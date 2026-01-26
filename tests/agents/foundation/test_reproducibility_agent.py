# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-008: Run Reproducibility Agent

Tests cover:
    - Determinism verification (same inputs = same outputs)
    - Hash comparison across runs
    - Environment capture and fingerprinting
    - Seed management for reproducibility
    - Version pinning verification
    - Drift detection from baseline
    - Replay mode execution
    - Non-determinism source tracking
    - Floating-point tolerance handling
    - Report generation

Author: GreenLang Team
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.foundation.reproducibility_agent import (
    ReproducibilityAgent,
    ReproducibilityInput,
    ReproducibilityOutput,
    ReproducibilityReport,
    VerificationStatus,
    DriftSeverity,
    NonDeterminismSource,
    EnvironmentFingerprint,
    SeedConfiguration,
    VersionManifest,
    VersionPin,
    ReplayConfiguration,
    VerificationCheck,
    DriftDetection,
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
)
from greenlang.utilities.determinism.clock import DeterministicClock


class TestEnvironmentFingerprint:
    """Tests for EnvironmentFingerprint model."""

    def test_fingerprint_creation(self):
        """Test creating an environment fingerprint."""
        fingerprint = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Windows",
            platform_release="10",
            platform_machine="AMD64",
            captured_at=DeterministicClock.now(),
            environment_hash="abc123def456"
        )

        assert fingerprint.python_version == "3.11.0"
        assert fingerprint.platform_system == "Windows"
        assert fingerprint.environment_hash == "abc123def456"

    def test_fingerprint_with_dependencies(self):
        """Test fingerprint with dependency versions."""
        fingerprint = EnvironmentFingerprint(
            python_version="3.11.0",
            platform_system="Linux",
            platform_release="5.15",
            platform_machine="x86_64",
            captured_at=DeterministicClock.now(),
            environment_hash="hash123",
            dependency_versions={
                "pydantic": "2.0.0",
                "numpy": "1.24.0"
            }
        )

        assert "pydantic" in fingerprint.dependency_versions
        assert fingerprint.dependency_versions["numpy"] == "1.24.0"


class TestSeedConfiguration:
    """Tests for SeedConfiguration model."""

    def test_default_seed_configuration(self):
        """Test default seed configuration."""
        seeds = SeedConfiguration()

        assert seeds.global_seed == 42
        assert seeds.numpy_seed == 42
        assert seeds.torch_seed == 42
        assert seeds.seed_hash != ""

    def test_custom_seed_configuration(self):
        """Test custom seed configuration."""
        seeds = SeedConfiguration(
            global_seed=12345,
            numpy_seed=67890,
            custom_seeds={"model_a": 111, "model_b": 222}
        )

        assert seeds.global_seed == 12345
        assert seeds.numpy_seed == 67890
        assert seeds.custom_seeds["model_a"] == 111

    def test_seed_hash_deterministic(self):
        """Test that same seeds produce same hash."""
        seeds1 = SeedConfiguration(global_seed=42, numpy_seed=42)
        seeds2 = SeedConfiguration(global_seed=42, numpy_seed=42)

        assert seeds1.seed_hash == seeds2.seed_hash

    def test_different_seeds_different_hash(self):
        """Test that different seeds produce different hash."""
        seeds1 = SeedConfiguration(global_seed=42)
        seeds2 = SeedConfiguration(global_seed=43)

        assert seeds1.seed_hash != seeds2.seed_hash


class TestVersionManifest:
    """Tests for VersionManifest model."""

    def test_empty_manifest(self):
        """Test creating empty version manifest."""
        manifest = VersionManifest()

        assert manifest.agent_versions == {}
        assert manifest.model_versions == {}
        assert manifest.factor_versions == {}

    def test_manifest_with_pins(self):
        """Test manifest with version pins."""
        manifest = VersionManifest(
            manifest_id="manifest_001",
            agent_versions={
                "GL-MRV-X-001": VersionPin(
                    component_type="agent",
                    component_id="GL-MRV-X-001",
                    version="1.0.0"
                )
            },
            factor_versions={
                "GHG_ELECTRICITY_US": VersionPin(
                    component_type="factor",
                    component_id="GHG_ELECTRICITY_US",
                    version="2024.1"
                )
            }
        )

        assert "GL-MRV-X-001" in manifest.agent_versions
        assert manifest.factor_versions["GHG_ELECTRICITY_US"].version == "2024.1"


class TestVerificationCheck:
    """Tests for VerificationCheck model."""

    def test_passing_check(self):
        """Test creating a passing verification check."""
        check = VerificationCheck(
            check_name="input_hash",
            status=VerificationStatus.PASS,
            expected_value="abc123",
            actual_value="abc123",
            message="Input hash matches"
        )

        assert check.status == VerificationStatus.PASS
        assert check.expected_value == check.actual_value

    def test_failing_check(self):
        """Test creating a failing verification check."""
        check = VerificationCheck(
            check_name="output_hash",
            status=VerificationStatus.FAIL,
            expected_value="abc123",
            actual_value="def456",
            message="Output hash mismatch"
        )

        assert check.status == VerificationStatus.FAIL
        assert check.expected_value != check.actual_value

    def test_check_with_tolerance(self):
        """Test check with numeric tolerance."""
        check = VerificationCheck(
            check_name="float_comparison",
            status=VerificationStatus.WARNING,
            expected_value="0.100000000",
            actual_value="0.100000001",
            difference=1e-9,
            tolerance=1e-8,
            message="Within tolerance"
        )

        assert check.tolerance == 1e-8
        assert check.difference is not None


class TestDriftDetection:
    """Tests for DriftDetection model."""

    def test_no_drift(self):
        """Test drift detection with no drift."""
        drift = DriftDetection(
            baseline_hash="hash123",
            current_hash="hash123",
            severity=DriftSeverity.NONE,
            drift_percentage=0.0,
            is_acceptable=True
        )

        assert drift.severity == DriftSeverity.NONE
        assert drift.is_acceptable

    def test_minor_drift(self):
        """Test minor drift detection."""
        drift = DriftDetection(
            baseline_hash="hash123",
            current_hash="hash456",
            severity=DriftSeverity.MINOR,
            drift_percentage=0.5,
            drifted_fields=["emissions_total"],
            is_acceptable=True
        )

        assert drift.severity == DriftSeverity.MINOR
        assert len(drift.drifted_fields) == 1
        assert drift.is_acceptable

    def test_critical_drift(self):
        """Test critical drift detection."""
        drift = DriftDetection(
            baseline_hash="hash123",
            current_hash="hash789",
            severity=DriftSeverity.CRITICAL,
            drift_percentage=15.0,
            drifted_fields=["scope1", "scope2", "total"],
            is_acceptable=False
        )

        assert drift.severity == DriftSeverity.CRITICAL
        assert not drift.is_acceptable


class TestReproducibilityAgent:
    """Tests for the ReproducibilityAgent."""

    @pytest.fixture
    def agent(self):
        """Create a ReproducibilityAgent instance."""
        return ReproducibilityAgent()

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for testing."""
        return {
            "company_id": "CORP001",
            "reporting_period": "2024-Q1",
            "emissions": {
                "scope1": 1000.5,
                "scope2": 2500.0,
                "scope3": 5000.0
            }
        }

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-FOUND-X-008"
        assert agent.AGENT_NAME == "Run Reproducibility Agent"
        # Environment is captured lazily, but seeds are initialized
        assert agent._seeds is not None

    def test_environment_capture(self, agent):
        """Test environment is captured correctly."""
        env = agent.get_current_environment()

        assert env.python_version != ""
        assert env.platform_system != ""
        assert env.environment_hash != ""
        assert env.captured_at is not None

    def test_seed_configuration(self, agent):
        """Test seed configuration is available."""
        seeds = agent.get_current_seeds()

        assert seeds.global_seed == 42
        assert seeds.seed_hash != ""

    def test_basic_verification_pass(self, agent, sample_input_data):
        """Test basic verification with matching hashes."""
        # First, compute the expected hash
        first_result = agent.run({
            "execution_id": "exec_001",
            "input_data": sample_input_data
        })

        assert first_result.success
        expected_hash = first_result.data["input_hash"]

        # Now verify with the expected hash
        result = agent.run({
            "execution_id": "exec_002",
            "input_data": sample_input_data,
            "expected_input_hash": expected_hash
        })

        assert result.success
        assert result.data["verification_status"] == "pass"
        assert result.data["is_reproducible"]
        assert result.data["input_hash"] == expected_hash

    def test_input_hash_mismatch(self, agent, sample_input_data):
        """Test verification fails on input hash mismatch."""
        result = agent.run({
            "execution_id": "exec_003",
            "input_data": sample_input_data,
            "expected_input_hash": "wrong_hash_value_abc123"
        })

        assert result.success  # Agent ran successfully
        assert result.data["verification_status"] == "fail"
        assert not result.data["is_reproducible"]
        assert result.data["checks_failed"] >= 1

    def test_deterministic_hashing(self, agent, sample_input_data):
        """Test that hashing is deterministic."""
        result1 = agent.run({
            "execution_id": "exec_004a",
            "input_data": sample_input_data
        })

        result2 = agent.run({
            "execution_id": "exec_004b",
            "input_data": sample_input_data
        })

        assert result1.data["input_hash"] == result2.data["input_hash"]

    def test_different_inputs_different_hash(self, agent):
        """Test that different inputs produce different hashes."""
        result1 = agent.run({
            "execution_id": "exec_005a",
            "input_data": {"value": 100}
        })

        result2 = agent.run({
            "execution_id": "exec_005b",
            "input_data": {"value": 200}
        })

        assert result1.data["input_hash"] != result2.data["input_hash"]

    def test_output_hash_verification(self, agent, sample_input_data):
        """Test output hash verification."""
        output_data = {
            "total_emissions": 8500.5,
            "intensity": 0.85
        }

        result = agent.run({
            "execution_id": "exec_006",
            "input_data": sample_input_data,
            "output_data": output_data
        })

        assert result.success
        assert result.data["output_hash"] != ""

    def test_floating_point_tolerance(self, agent):
        """Test floating-point comparison with tolerance."""
        input_data = {"value": 0.1 + 0.2}  # Known float precision issue

        result = agent.run({
            "execution_id": "exec_007",
            "input_data": input_data,
            "absolute_tolerance": 1e-9,
            "relative_tolerance": 1e-6
        })

        assert result.success
        assert result.data["input_hash"] != ""

    def test_drift_detection_no_drift(self, agent):
        """Test drift detection with identical results."""
        baseline = {"emissions": 1000.0, "intensity": 0.5}
        current = {"emissions": 1000.0, "intensity": 0.5}

        result = agent.run({
            "execution_id": "exec_008",
            "input_data": {"test": "data"},
            "output_data": current,
            "baseline_result": baseline
        })

        assert result.success
        drift = result.data.get("drift_detection")
        assert drift is not None
        assert drift["severity"] == "none"
        assert drift["is_acceptable"]

    def test_drift_detection_minor_drift(self, agent):
        """Test drift detection with minor drift."""
        baseline = {"emissions": 1000.0}
        current = {"emissions": 1005.0}  # 0.5% drift

        result = agent.run({
            "execution_id": "exec_009",
            "input_data": {"test": "data"},
            "output_data": current,
            "baseline_result": baseline,
            "drift_soft_threshold": 0.01,  # 1%
            "drift_hard_threshold": 0.05   # 5%
        })

        assert result.success
        drift = result.data.get("drift_detection")
        assert drift is not None
        assert drift["severity"] in ("minor", "none")

    def test_drift_detection_critical_drift(self, agent):
        """Test drift detection with critical drift."""
        baseline = {"emissions": 1000.0}
        current = {"emissions": 1200.0}  # 20% drift

        result = agent.run({
            "execution_id": "exec_010",
            "input_data": {"test": "data"},
            "output_data": current,
            "baseline_result": baseline,
            "drift_soft_threshold": 0.01,
            "drift_hard_threshold": 0.05
        })

        assert result.success
        drift = result.data.get("drift_detection")
        assert drift is not None
        assert drift["severity"] == "critical"
        assert not drift["is_acceptable"]

    def test_non_determinism_tracking_timestamps(self, agent):
        """Test tracking of timestamp non-determinism."""
        input_data = {
            "value": 100,
            "timestamp": "2024-01-01T00:00:00",
            "created_at": "2024-01-01T12:00:00"
        }

        result = agent.run({
            "execution_id": "exec_011",
            "input_data": input_data,
            "track_non_determinism": True
        })

        assert result.success
        sources = result.data.get("non_determinism_sources", [])
        assert "timestamp" in sources

    def test_non_determinism_tracking_random(self, agent):
        """Test tracking of random value non-determinism."""
        input_data = {
            "value": 100,
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "session_id": "abc123def456ghi789jkl012mno345pqr678"
        }

        result = agent.run({
            "execution_id": "exec_012",
            "input_data": input_data,
            "track_non_determinism": True
        })

        assert result.success
        sources = result.data.get("non_determinism_sources", [])
        assert "random_seed" in sources

    def test_version_manifest_verification(self, agent, sample_input_data):
        """Test version manifest verification."""
        manifest = VersionManifest(
            manifest_id="test_manifest",
            agent_versions={
                "GL-MRV-X-001": VersionPin(
                    component_type="agent",
                    component_id="GL-MRV-X-001",
                    version="1.0.0"
                )
            }
        )

        result = agent.run({
            "execution_id": "exec_013",
            "input_data": sample_input_data,
            "version_manifest": manifest.model_dump()
        })

        assert result.success
        assert result.data.get("version_verification") is not None

    def test_replay_configuration_capture(self, agent, sample_input_data):
        """Test capturing execution state for replay."""
        replay_config = agent.capture_execution_state(
            execution_id="exec_014",
            input_data=sample_input_data
        )

        assert replay_config.original_execution_id == "exec_014"
        assert replay_config.captured_inputs == sample_input_data
        assert replay_config.captured_environment is not None
        assert replay_config.captured_seeds is not None
        assert replay_config.replay_mode

    def test_replay_mode_verification(self, agent, sample_input_data):
        """Test replay mode verification."""
        # Capture state
        replay_config = agent.capture_execution_state(
            execution_id="exec_015_original",
            input_data=sample_input_data
        )

        # Run in replay mode
        result = agent.run({
            "execution_id": "exec_015_replay",
            "input_data": sample_input_data,
            "replay_config": replay_config.model_dump()
        })

        assert result.success
        # Should have environment and seed verification checks
        check_names = [c["check_name"] for c in result.data["checks"]]
        assert "environment_verification" in check_names
        assert "seed_verification" in check_names

    def test_report_generation(self, agent, sample_input_data):
        """Test reproducibility report generation."""
        # First run verification
        result = agent.run({
            "execution_id": "exec_016",
            "input_data": sample_input_data
        })

        assert result.success

        # Generate report
        output = ReproducibilityOutput(**result.data)
        report = agent.generate_report(output)

        assert report.report_id != ""
        assert report.execution_id == "exec_016"
        assert report.overall_status == VerificationStatus.PASS
        assert report.confidence_score > 0
        assert len(report.recommendations) > 0
        assert report.report_hash != ""

    def test_report_with_failures(self, agent, sample_input_data):
        """Test report generation with failures."""
        result = agent.run({
            "execution_id": "exec_017",
            "input_data": sample_input_data,
            "expected_input_hash": "wrong_hash"
        })

        assert result.success

        output = ReproducibilityOutput(**result.data)
        report = agent.generate_report(output)

        assert report.overall_status == VerificationStatus.FAIL
        assert not report.is_reproducible

    def test_hash_comparison_utility(self, agent):
        """Test hash comparison utility method."""
        match, explanation = agent.compare_hashes("abc123", "abc123")
        assert match
        assert "match" in explanation.lower()

        match, explanation = agent.compare_hashes("abc123", "def456")
        assert not match
        assert "differ" in explanation.lower()

    def test_set_version_manifest(self, agent):
        """Test setting version manifest."""
        manifest = VersionManifest(manifest_id="test_001")
        agent.set_version_manifest(manifest)

        assert agent._version_manifest is not None
        assert agent._version_manifest.manifest_id == "test_001"

    def test_nested_data_hashing(self, agent):
        """Test hashing of deeply nested data structures."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "values": [1, 2, 3],
                        "nested_list": [[1, 2], [3, 4]]
                    }
                }
            }
        }

        result1 = agent.run({
            "execution_id": "exec_018a",
            "input_data": nested_data
        })

        result2 = agent.run({
            "execution_id": "exec_018b",
            "input_data": nested_data
        })

        assert result1.data["input_hash"] == result2.data["input_hash"]

    def test_dict_ordering_determinism(self, agent):
        """Test that dict ordering doesn't affect hash."""
        data1 = {"z": 1, "a": 2, "m": 3}
        data2 = {"a": 2, "m": 3, "z": 1}

        result1 = agent.run({
            "execution_id": "exec_019a",
            "input_data": data1
        })

        result2 = agent.run({
            "execution_id": "exec_019b",
            "input_data": data2
        })

        assert result1.data["input_hash"] == result2.data["input_hash"]

    def test_checks_summary(self, agent, sample_input_data):
        """Test that checks summary is accurate."""
        result = agent.run({
            "execution_id": "exec_020",
            "input_data": sample_input_data,
            "expected_input_hash": "wrong_hash"
        })

        assert result.success
        total_checks = len(result.data["checks"])
        passed = result.data["checks_passed"]
        failed = result.data["checks_failed"]
        warned = result.data["checks_warned"]

        # Verify counts add up (skipped checks exist too)
        assert passed + failed + warned <= total_checks

    def test_provenance_hash_unique(self, agent):
        """Test that provenance hash is unique per execution."""
        result1 = agent.run({
            "execution_id": "exec_021a",
            "input_data": {"value": 1}
        })

        result2 = agent.run({
            "execution_id": "exec_021b",
            "input_data": {"value": 1}
        })

        # Provenance includes execution_id, so should differ
        assert result1.data["provenance_hash"] != result2.data["provenance_hash"]

    def test_processing_time_tracked(self, agent, sample_input_data):
        """Test that processing time is tracked."""
        result = agent.run({
            "execution_id": "exec_022",
            "input_data": sample_input_data
        })

        assert result.success
        # Processing time should be non-negative (can be 0 on fast systems)
        assert result.data["processing_time_ms"] >= 0


class TestReproducibilityInput:
    """Tests for ReproducibilityInput model."""

    def test_minimal_input(self):
        """Test minimal valid input."""
        input_model = ReproducibilityInput(
            execution_id="exec_001",
            input_data={"key": "value"}
        )

        assert input_model.execution_id == "exec_001"
        assert input_model.absolute_tolerance == DEFAULT_ABSOLUTE_TOLERANCE

    def test_full_input(self):
        """Test fully specified input."""
        input_model = ReproducibilityInput(
            execution_id="exec_002",
            input_data={"key": "value"},
            expected_input_hash="hash123",
            expected_output_hash="hash456",
            output_data={"result": 100},
            baseline_result={"result": 99},
            absolute_tolerance=1e-8,
            relative_tolerance=1e-5,
            drift_soft_threshold=0.02,
            drift_hard_threshold=0.10
        )

        assert input_model.expected_input_hash == "hash123"
        assert input_model.drift_soft_threshold == 0.02

    def test_negative_tolerance_rejected(self):
        """Test that negative tolerances are rejected."""
        with pytest.raises(ValueError):
            ReproducibilityInput(
                execution_id="exec_003",
                input_data={"key": "value"},
                absolute_tolerance=-1e-9
            )


class TestReproducibilityOutput:
    """Tests for ReproducibilityOutput model."""

    def test_output_creation(self):
        """Test creating output model."""
        output = ReproducibilityOutput(
            execution_id="exec_001",
            verification_status=VerificationStatus.PASS,
            is_reproducible=True,
            input_hash="abc123",
            provenance_hash="def456",
            environment=EnvironmentFingerprint(
                python_version="3.11",
                platform_system="Linux",
                platform_release="5.15",
                platform_machine="x86_64",
                captured_at=DeterministicClock.now(),
                environment_hash="env123"
            ),
            seeds=SeedConfiguration(),
            processing_time_ms=10.5
        )

        assert output.is_reproducible
        assert output.checks_passed == 0  # No checks added


class TestDriftDetectionLogic:
    """Tests for drift detection logic."""

    @pytest.fixture
    def agent(self):
        """Create agent for drift testing."""
        return ReproducibilityAgent()

    def test_nested_drift_detection(self, agent):
        """Test drift detection in nested structures."""
        baseline = {
            "company": {
                "emissions": {
                    "scope1": 1000.0,
                    "scope2": 2000.0
                }
            }
        }
        current = {
            "company": {
                "emissions": {
                    "scope1": 1000.0,
                    "scope2": 2100.0  # 5% drift
                }
            }
        }

        result = agent.run({
            "execution_id": "drift_test_001",
            "input_data": {"test": True},
            "output_data": current,
            "baseline_result": baseline,
            "drift_soft_threshold": 0.01,
            "drift_hard_threshold": 0.10
        })

        assert result.success
        drift = result.data["drift_detection"]
        assert len(drift["drifted_fields"]) > 0
        assert any("scope2" in f for f in drift["drifted_fields"])

    def test_array_drift_detection(self, agent):
        """Test drift detection in arrays."""
        baseline = {"values": [1.0, 2.0, 3.0]}
        current = {"values": [1.0, 2.1, 3.0]}  # 5% drift on second element

        result = agent.run({
            "execution_id": "drift_test_002",
            "input_data": {"test": True},
            "output_data": current,
            "baseline_result": baseline
        })

        assert result.success
        drift = result.data["drift_detection"]
        assert len(drift["drifted_fields"]) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def agent(self):
        """Create agent for edge case testing."""
        return ReproducibilityAgent()

    def test_empty_input_data(self, agent):
        """Test handling of empty input data."""
        result = agent.run({
            "execution_id": "edge_001",
            "input_data": {}
        })

        assert result.success
        assert result.data["input_hash"] != ""

    def test_none_values_in_data(self, agent):
        """Test handling of None values."""
        result = agent.run({
            "execution_id": "edge_002",
            "input_data": {"key": None, "other": "value"}
        })

        assert result.success

    def test_special_float_values(self, agent):
        """Test handling of special float values."""
        result = agent.run({
            "execution_id": "edge_003",
            "input_data": {
                "zero": 0.0,
                "negative": -100.5,
                "small": 1e-10
            }
        })

        assert result.success

    def test_large_numbers(self, agent):
        """Test handling of large numbers."""
        result = agent.run({
            "execution_id": "edge_004",
            "input_data": {
                "large_int": 10**18,
                "large_float": 1e15
            }
        })

        assert result.success

    def test_unicode_data(self, agent):
        """Test handling of unicode data."""
        result = agent.run({
            "execution_id": "edge_005",
            "input_data": {
                "company": "Test GmbH",
                "location": "Beijing",
                "notes": "Test data"
            }
        })

        assert result.success

    def test_missing_optional_fields(self, agent):
        """Test with all optional fields omitted."""
        result = agent.run({
            "execution_id": "edge_006",
            "input_data": {"minimal": True}
        })

        assert result.success
        assert result.data["output_hash"] == ""  # No output provided
        assert result.data["drift_detection"] is None


class TestIntegration:
    """Integration tests for ReproducibilityAgent."""

    def test_full_verification_workflow(self):
        """Test complete verification workflow."""
        agent = ReproducibilityAgent()

        # Step 1: Initial execution
        input_data = {
            "company_id": "CORP001",
            "period": "2024-Q1",
            "emissions": {"scope1": 1000, "scope2": 2000}
        }

        initial_result = agent.run({
            "execution_id": "workflow_001",
            "input_data": input_data
        })

        assert initial_result.success
        expected_hash = initial_result.data["input_hash"]

        # Step 2: Capture state for replay
        replay_config = agent.capture_execution_state(
            "workflow_001",
            input_data
        )

        # Step 3: Re-run with verification
        verification_result = agent.run({
            "execution_id": "workflow_002",
            "input_data": input_data,
            "expected_input_hash": expected_hash,
            "replay_config": replay_config.model_dump()
        })

        assert verification_result.success
        assert verification_result.data["is_reproducible"]

        # Step 4: Generate report
        output = ReproducibilityOutput(**verification_result.data)
        report = agent.generate_report(output)

        assert report.is_reproducible
        assert report.confidence_score > 0.5

    def test_drift_monitoring_workflow(self):
        """Test drift monitoring over multiple runs."""
        agent = ReproducibilityAgent()

        baseline = {"emissions": 1000.0}

        # Run 1: Establish baseline
        run1 = agent.run({
            "execution_id": "drift_monitor_001",
            "input_data": {"run": 1},
            "output_data": baseline
        })
        assert run1.success

        # Run 2: No drift
        run2 = agent.run({
            "execution_id": "drift_monitor_002",
            "input_data": {"run": 2},
            "output_data": {"emissions": 1000.0},
            "baseline_result": baseline
        })
        assert run2.success
        assert run2.data["drift_detection"]["severity"] == "none"

        # Run 3: Minor drift
        run3 = agent.run({
            "execution_id": "drift_monitor_003",
            "input_data": {"run": 3},
            "output_data": {"emissions": 1002.0},  # 0.2% drift
            "baseline_result": baseline,
            "drift_soft_threshold": 0.01
        })
        assert run3.success
        assert run3.data["drift_detection"]["severity"] in ("none", "minor")

        # Run 4: Critical drift
        run4 = agent.run({
            "execution_id": "drift_monitor_004",
            "input_data": {"run": 4},
            "output_data": {"emissions": 1150.0},  # 15% drift
            "baseline_result": baseline,
            "drift_hard_threshold": 0.05
        })
        assert run4.success
        assert run4.data["drift_detection"]["severity"] == "critical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
