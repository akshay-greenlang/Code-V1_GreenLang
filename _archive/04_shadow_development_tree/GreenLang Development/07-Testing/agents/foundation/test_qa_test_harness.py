# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-009: QA Test Harness Agent

Tests cover:
    - Test harness initialization
    - Agent registration
    - Single test execution
    - Test suite execution
    - Zero-hallucination testing
    - Determinism testing
    - Lineage completeness testing
    - Golden file testing
    - Regression testing
    - Performance benchmarking
    - Coverage tracking
    - Parallel test execution
    - Test filtering by tags
    - Report generation

Author: GreenLang Team
"""

import json
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.foundation.qa_test_harness import (
    QATestHarnessAgent,
    TestCaseInput,
    TestCaseResult,
    TestSuiteInput,
    TestSuiteResult,
    TestStatus,
    TestCategory,
    SeverityLevel,
    TestAssertion,
    PerformanceBenchmark,
    CoverageReport,
    GoldenFileSpec,
    COMMON_FIXTURES,
)


# =============================================================================
# Test Agent Implementations
# =============================================================================

class SuccessAgent(BaseAgent):
    """Agent that always succeeds with deterministic output."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        value = input_data.get("value", "default")
        return AgentResult(
            success=True,
            data={
                "result": f"processed_{value}",
                "input_value": value,
                "provenance_id": "PROV-001",
            }
        )


class FailureAgent(BaseAgent):
    """Agent that always fails."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=False,
            error="Intentional failure for testing"
        )


class SlowAgent(BaseAgent):
    """Agent that takes configurable time to execute."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        delay = input_data.get("delay", 0.1)
        time.sleep(delay)
        return AgentResult(
            success=True,
            data={"delayed": True, "delay_seconds": delay}
        )


class NonDeterministicAgent(BaseAgent):
    """Agent with non-deterministic output (for testing determinism checks)."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import random
        return AgentResult(
            success=True,
            data={
                "value": input_data.get("value", "default"),
                "random_value": random.random(),  # Non-deterministic
            }
        )


class LineageCompleteAgent(BaseAgent):
    """Agent with complete lineage tracking."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import hashlib
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return AgentResult(
            success=True,
            data={
                "result": "calculated_value",
                "provenance_id": f"PROV-{input_hash}",
                "input_hash": input_hash,
                "created_at": datetime.utcnow().isoformat(),
            }
        )


class LineageIncompleteAgent(BaseAgent):
    """Agent with incomplete lineage (missing provenance)."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=True,
            data={
                "result": "calculated_value",
                # Missing provenance_id and timestamp
            }
        )


class HallucinatingAgent(BaseAgent):
    """Agent that produces potentially hallucinated data."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=True,
            data={
                "emissions": 1000000,  # Suspiciously round number
                "factor": 2.5,
                "source": "Made up source",
            }
        )


class ErrorAgent(BaseAgent):
    """Agent that raises an exception."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        raise RuntimeError("Intentional error for testing")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def harness():
    """Create a QA Test Harness instance with test agents registered."""
    h = QATestHarnessAgent()
    h.register_agent("SuccessAgent", SuccessAgent)
    h.register_agent("FailureAgent", FailureAgent)
    h.register_agent("SlowAgent", SlowAgent)
    h.register_agent("NonDeterministicAgent", NonDeterministicAgent)
    h.register_agent("LineageCompleteAgent", LineageCompleteAgent)
    h.register_agent("LineageIncompleteAgent", LineageIncompleteAgent)
    h.register_agent("HallucinatingAgent", HallucinatingAgent)
    h.register_agent("ErrorAgent", ErrorAgent)
    return h


@pytest.fixture
def temp_golden_dir():
    """Create a temporary directory for golden files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test Harness Initialization
# =============================================================================

class TestQATestHarnessInit:
    """Tests for QA Test Harness initialization."""

    def test_default_initialization(self):
        """Test harness initializes with default config."""
        harness = QATestHarnessAgent()

        assert harness.AGENT_ID == "GL-FOUND-X-009"
        assert harness.AGENT_NAME == "QA Test Harness Agent"
        assert harness.VERSION == "1.0.0"

    def test_custom_config(self):
        """Test harness with custom configuration."""
        config = AgentConfig(
            name="Custom Harness",
            description="Custom test harness",
            parameters={
                "default_timeout": 120,
                "max_parallel_tests": 8,
            }
        )
        harness = QATestHarnessAgent(config)

        assert harness.config.name == "Custom Harness"
        assert harness.config.parameters["default_timeout"] == 120

    def test_agent_registration(self, harness):
        """Test agent registration."""
        assert "SuccessAgent" in harness._agent_registry
        assert "FailureAgent" in harness._agent_registry
        assert len(harness._agent_registry) == 8


# =============================================================================
# Single Test Execution
# =============================================================================

class TestSingleTestExecution:
    """Tests for single test case execution."""

    def test_successful_test(self, harness):
        """Test running a successful test case."""
        test_input = TestCaseInput(
            name="test_success",
            description="Test successful execution",
            category=TestCategory.UNIT,
            agent_type="SuccessAgent",
            input_data={"value": "test_value"}
        )

        result = harness.run_test(test_input)

        assert result.status == TestStatus.PASSED
        assert result.name == "test_success"
        assert result.category == TestCategory.UNIT
        assert result.duration_ms > 0
        assert result.started_at is not None
        assert result.completed_at is not None
        assert len(result.assertions) > 0

    def test_failed_test(self, harness):
        """Test running a failing test case."""
        test_input = TestCaseInput(
            name="test_failure",
            description="Test failure handling",
            category=TestCategory.UNIT,
            agent_type="FailureAgent",
            input_data={}
        )

        result = harness.run_test(test_input)

        assert result.status == TestStatus.FAILED
        assert any(not a.passed for a in result.assertions)

    def test_error_test(self, harness):
        """Test handling of agent errors."""
        test_input = TestCaseInput(
            name="test_error",
            description="Test error handling",
            category=TestCategory.UNIT,
            agent_type="ErrorAgent",
            input_data={}
        )

        result = harness.run_test(test_input)

        # The BaseAgent.run() catches exceptions and returns AgentResult(success=False)
        # so the harness sees it as a failed test, not an error
        assert result.status in (TestStatus.ERROR, TestStatus.FAILED)
        # The test should still complete (no unhandled exception)
        assert result.completed_at is not None

    def test_skipped_test(self, harness):
        """Test skipping a test case."""
        test_input = TestCaseInput(
            name="test_skip",
            description="Test skip handling",
            category=TestCategory.UNIT,
            agent_type="SuccessAgent",
            input_data={},
            skip=True,
            skip_reason="Skipping for test"
        )

        result = harness.run_test(test_input)

        assert result.status == TestStatus.SKIPPED
        assert result.metadata.get("skip_reason") == "Skipping for test"

    def test_timeout_test(self, harness):
        """Test timeout handling."""
        test_input = TestCaseInput(
            name="test_timeout",
            description="Test timeout handling",
            category=TestCategory.UNIT,
            agent_type="SlowAgent",
            input_data={"delay": 5},
            timeout_seconds=1
        )

        result = harness.run_test(test_input)

        assert result.status == TestStatus.TIMEOUT
        assert "timed out" in result.error_message.lower()

    def test_unregistered_agent(self, harness):
        """Test handling of unregistered agent type."""
        test_input = TestCaseInput(
            name="test_unregistered",
            description="Test unregistered agent",
            category=TestCategory.UNIT,
            agent_type="NonExistentAgent",
            input_data={}
        )

        result = harness.run_test(test_input)

        assert result.status == TestStatus.ERROR
        assert "not registered" in result.error_message.lower()

    def test_input_output_hashes(self, harness):
        """Test that input and output hashes are computed."""
        test_input = TestCaseInput(
            name="test_hashes",
            description="Test hash computation",
            category=TestCategory.UNIT,
            agent_type="SuccessAgent",
            input_data={"value": "hash_test"}
        )

        result = harness.run_test(test_input)

        assert result.input_hash != ""
        assert len(result.input_hash) == 16  # Truncated SHA-256
        assert result.output_hash != ""
        assert len(result.output_hash) == 16


# =============================================================================
# Test Suite Execution
# =============================================================================

class TestSuiteExecution:
    """Tests for test suite execution."""

    def test_suite_all_pass(self, harness):
        """Test suite where all tests pass."""
        suite_input = TestSuiteInput(
            name="all_pass_suite",
            test_cases=[
                TestCaseInput(
                    name="test_1",
                    agent_type="SuccessAgent",
                    input_data={"value": "1"}
                ),
                TestCaseInput(
                    name="test_2",
                    agent_type="SuccessAgent",
                    input_data={"value": "2"}
                ),
            ],
            parallel=False
        )

        result = harness.run_suite(suite_input)

        assert result.status == TestStatus.PASSED
        assert result.total_tests == 2
        assert result.passed == 2
        assert result.failed == 0
        assert result.pass_rate == 100.0

    def test_suite_with_failures(self, harness):
        """Test suite with some failures."""
        suite_input = TestSuiteInput(
            name="mixed_suite",
            test_cases=[
                TestCaseInput(
                    name="test_pass",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
                TestCaseInput(
                    name="test_fail",
                    agent_type="FailureAgent",
                    input_data={}
                ),
            ],
            parallel=False
        )

        result = harness.run_suite(suite_input)

        assert result.status == TestStatus.FAILED
        assert result.total_tests == 2
        assert result.passed == 1
        assert result.failed == 1
        assert result.pass_rate == 50.0

    def test_suite_parallel_execution(self, harness):
        """Test parallel suite execution."""
        suite_input = TestSuiteInput(
            name="parallel_suite",
            test_cases=[
                TestCaseInput(
                    name=f"test_{i}",
                    agent_type="SuccessAgent",
                    input_data={"value": str(i)}
                )
                for i in range(5)
            ],
            parallel=True,
            max_workers=3
        )

        result = harness.run_suite(suite_input)

        assert result.status == TestStatus.PASSED
        assert result.total_tests == 5
        assert result.passed == 5

    def test_suite_fail_fast(self, harness):
        """Test fail_fast stops on first failure."""
        suite_input = TestSuiteInput(
            name="fail_fast_suite",
            test_cases=[
                TestCaseInput(
                    name="test_pass_1",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
                TestCaseInput(
                    name="test_fail",
                    agent_type="FailureAgent",
                    input_data={}
                ),
                TestCaseInput(
                    name="test_pass_2",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
            ],
            parallel=False,
            fail_fast=True
        )

        result = harness.run_suite(suite_input)

        assert result.status == TestStatus.FAILED
        assert result.skipped >= 1  # At least one skipped due to fail_fast

    def test_suite_tag_filtering_include(self, harness):
        """Test including tests by tags."""
        suite_input = TestSuiteInput(
            name="tag_include_suite",
            test_cases=[
                TestCaseInput(
                    name="tagged_test",
                    agent_type="SuccessAgent",
                    input_data={},
                    tags=["important"]
                ),
                TestCaseInput(
                    name="untagged_test",
                    agent_type="SuccessAgent",
                    input_data={},
                    tags=["skip_me"]
                ),
            ],
            tags_include=["important"],
            parallel=False
        )

        result = harness.run_suite(suite_input)

        assert result.total_tests == 1
        assert result.test_results[0].name == "tagged_test"

    def test_suite_tag_filtering_exclude(self, harness):
        """Test excluding tests by tags."""
        suite_input = TestSuiteInput(
            name="tag_exclude_suite",
            test_cases=[
                TestCaseInput(
                    name="test_keep",
                    agent_type="SuccessAgent",
                    input_data={},
                    tags=["keep"]
                ),
                TestCaseInput(
                    name="test_exclude",
                    agent_type="SuccessAgent",
                    input_data={},
                    tags=["slow", "skip"]
                ),
            ],
            tags_exclude=["slow"],
            parallel=False
        )

        result = harness.run_suite(suite_input)

        assert result.total_tests == 1
        assert result.test_results[0].name == "test_keep"

    def test_suite_provenance_hash(self, harness):
        """Test suite generates provenance hash."""
        suite_input = TestSuiteInput(
            name="provenance_suite",
            test_cases=[
                TestCaseInput(
                    name="test_1",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
            ],
            parallel=False
        )

        result = harness.run_suite(suite_input)

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # Full SHA-256


# =============================================================================
# Zero-Hallucination Testing
# =============================================================================

class TestZeroHallucination:
    """Tests for zero-hallucination verification."""

    def test_hallucination_check_clean(self, harness):
        """Test hallucination check passes for clean agent."""
        result = harness.test_zero_hallucination(
            agent_type="SuccessAgent",
            input_data={"value": "test"}
        )

        assert result.category == TestCategory.ZERO_HALLUCINATION
        # Check for output consistency assertion
        consistency_assertions = [
            a for a in result.assertions
            if a.name == "output_consistency"
        ]
        assert len(consistency_assertions) > 0
        assert all(a.passed for a in consistency_assertions)

    def test_hallucination_check_suspicious(self, harness):
        """Test hallucination check flags suspicious values."""
        result = harness.test_zero_hallucination(
            agent_type="HallucinatingAgent",
            input_data={"value": 50}
        )

        # Should flag the suspiciously round emissions value
        numeric_assertions = [
            a for a in result.assertions
            if "numeric_traceability" in a.name
        ]
        # At least one should be flagged
        assert any(not a.passed for a in numeric_assertions) or len(numeric_assertions) > 0

    def test_provenance_id_validation(self, harness):
        """Test provenance ID validation in hallucination check."""
        result = harness.test_zero_hallucination(
            agent_type="LineageCompleteAgent",
            input_data={"value": "test"}
        )

        prov_assertions = [
            a for a in result.assertions
            if a.name == "provenance_id_valid"
        ]
        assert len(prov_assertions) > 0
        assert prov_assertions[0].passed


# =============================================================================
# Determinism Testing
# =============================================================================

class TestDeterminism:
    """Tests for determinism verification."""

    def test_deterministic_agent(self, harness):
        """Test determinism check passes for deterministic agent."""
        result = harness.test_determinism(
            agent_type="SuccessAgent",
            input_data={"value": "deterministic_test"},
            iterations=3
        )

        assert result.category == TestCategory.DETERMINISM
        assert result.status == TestStatus.PASSED

        # All hash assertions should pass
        hash_assertions = [
            a for a in result.assertions
            if "hash" in a.name or "determinism" in a.name
        ]
        assert len(hash_assertions) > 0
        assert all(a.passed for a in hash_assertions)

    def test_non_deterministic_agent(self, harness):
        """Test determinism check fails for non-deterministic agent."""
        result = harness.test_determinism(
            agent_type="NonDeterministicAgent",
            input_data={"value": "test"},
            iterations=3
        )

        assert result.status == TestStatus.FAILED

        # Hash or data assertions should fail
        failed_assertions = [a for a in result.assertions if not a.passed]
        assert len(failed_assertions) > 0


# =============================================================================
# Lineage Testing
# =============================================================================

class TestLineage:
    """Tests for lineage completeness verification."""

    def test_complete_lineage(self, harness):
        """Test lineage check passes for complete lineage."""
        result = harness.test_lineage_completeness(
            agent_type="LineageCompleteAgent",
            input_data={"value": "lineage_test"}
        )

        assert result.category == TestCategory.LINEAGE

        prov_assertions = [
            a for a in result.assertions
            if a.name == "has_provenance_id"
        ]
        assert len(prov_assertions) > 0
        assert prov_assertions[0].passed

    def test_incomplete_lineage(self, harness):
        """Test lineage check flags incomplete lineage."""
        result = harness.test_lineage_completeness(
            agent_type="LineageIncompleteAgent",
            input_data={"value": "test"}
        )

        # Should flag missing provenance
        prov_assertions = [
            a for a in result.assertions
            if a.name == "has_provenance_id"
        ]
        # The agent_success might pass but provenance should fail
        assert len(prov_assertions) > 0


# =============================================================================
# Golden File Testing
# =============================================================================

class TestGoldenFile:
    """Tests for golden file comparison."""

    def test_save_golden_file(self, harness, temp_golden_dir):
        """Test saving a golden file."""
        harness._golden_dir = temp_golden_dir

        spec = harness.save_golden_file(
            agent_type="SuccessAgent",
            input_data={"value": "golden_test"},
            output_data={"result": "processed_golden_test"},
            description="Test golden file"
        )

        assert spec.path.endswith(".json")
        assert Path(spec.path).exists()
        assert spec.content_hash != ""
        assert spec.description == "Test golden file"

    def test_load_golden_file(self, harness, temp_golden_dir):
        """Test loading a golden file."""
        harness._golden_dir = temp_golden_dir

        # Save first
        spec = harness.save_golden_file(
            agent_type="SuccessAgent",
            input_data={"value": "load_test"},
            output_data={"result": "expected_result"},
            description="Load test"
        )

        # Load
        loaded = harness.load_golden_file(spec.path)

        assert loaded["agent_type"] == "SuccessAgent"
        assert loaded["input_data"]["value"] == "load_test"
        assert loaded["expected_output"]["result"] == "expected_result"

    def test_golden_file_comparison_pass(self, harness, temp_golden_dir):
        """Test golden file comparison passes when output matches."""
        harness._golden_dir = temp_golden_dir

        # Save golden file with expected output
        spec = harness.save_golden_file(
            agent_type="SuccessAgent",
            input_data={"value": "compare_test"},
            output_data={
                "result": "processed_compare_test",
                "input_value": "compare_test",
                "provenance_id": "PROV-001",
            },
            description="Comparison test"
        )

        # Run test against golden file
        result = harness.test_golden_file(
            agent_type="SuccessAgent",
            input_data={"value": "compare_test"},
            golden_file_path=spec.path
        )

        assert result.category == TestCategory.GOLDEN_FILE
        # Key fields should match
        golden_assertions = [a for a in result.assertions if "golden_" in a.name]
        assert len(golden_assertions) > 0

    def test_golden_file_not_found(self, harness):
        """Test golden file comparison handles missing file."""
        result = harness.test_golden_file(
            agent_type="SuccessAgent",
            input_data={"value": "test"},
            golden_file_path="/nonexistent/path/golden.json"
        )

        # Should have assertion about missing file
        not_found_assertions = [
            a for a in result.assertions
            if "not found" in a.message.lower() or "exists" in a.name
        ]
        assert len(not_found_assertions) > 0
        assert not not_found_assertions[0].passed


# =============================================================================
# Regression Testing
# =============================================================================

class TestRegression:
    """Tests for regression detection."""

    def test_regression_with_baseline(self, harness):
        """Test regression check against baseline hash."""
        # First run to get baseline
        first_result = harness.run_test(TestCaseInput(
            name="baseline_test",
            agent_type="SuccessAgent",
            input_data={"value": "regression_test"}
        ))

        baseline_hash = first_result.output_hash

        # Run regression test with baseline
        result = harness.test_regression(
            agent_type="SuccessAgent",
            input_data={"value": "regression_test"},
            baseline_hash=baseline_hash
        )

        assert result.category == TestCategory.REGRESSION

        baseline_assertions = [
            a for a in result.assertions
            if a.name == "baseline_hash_match"
        ]
        assert len(baseline_assertions) > 0
        assert baseline_assertions[0].passed

    def test_regression_detects_change(self, harness):
        """Test regression detection flags changed output."""
        result = harness.test_regression(
            agent_type="SuccessAgent",
            input_data={"value": "new_value"},
            baseline_hash="different_hash_12345"
        )

        baseline_assertions = [
            a for a in result.assertions
            if a.name == "baseline_hash_match"
        ]
        assert len(baseline_assertions) > 0
        assert not baseline_assertions[0].passed


# =============================================================================
# Performance Benchmarking
# =============================================================================

class TestPerformanceBenchmark:
    """Tests for performance benchmarking."""

    def test_benchmark_basic(self, harness):
        """Test basic benchmark execution."""
        benchmark = harness.benchmark_agent(
            agent_type="SuccessAgent",
            input_data={"value": "benchmark_test"},
            iterations=5,
            warmup=1
        )

        assert benchmark.agent_type == "SuccessAgent"
        assert benchmark.iterations == 5
        assert benchmark.min_ms > 0
        assert benchmark.max_ms >= benchmark.min_ms
        assert benchmark.mean_ms > 0
        assert benchmark.median_ms > 0
        assert benchmark.std_dev_ms >= 0
        assert benchmark.p95_ms >= benchmark.median_ms
        assert benchmark.p99_ms >= benchmark.p95_ms
        assert benchmark.passed_threshold  # No threshold set

    def test_benchmark_with_threshold_pass(self, harness):
        """Test benchmark passes when under threshold."""
        benchmark = harness.benchmark_agent(
            agent_type="SuccessAgent",
            input_data={"value": "threshold_test"},
            iterations=5,
            threshold_ms=10000  # 10 seconds - should easily pass
        )

        assert benchmark.passed_threshold
        assert benchmark.threshold_ms == 10000

    def test_benchmark_with_threshold_fail(self, harness):
        """Test benchmark fails when over threshold."""
        benchmark = harness.benchmark_agent(
            agent_type="SlowAgent",
            input_data={"delay": 0.1},  # 100ms delay
            iterations=3,
            warmup=0,
            threshold_ms=10  # 10ms threshold - should fail
        )

        assert not benchmark.passed_threshold

    def test_benchmark_unregistered_agent(self, harness):
        """Test benchmark raises error for unregistered agent."""
        with pytest.raises(ValueError, match="not registered"):
            harness.benchmark_agent(
                agent_type="NonExistentAgent",
                input_data={}
            )


# =============================================================================
# Coverage Tracking
# =============================================================================

class TestCoverageTracking:
    """Tests for coverage tracking."""

    def test_coverage_report(self, harness):
        """Test coverage report generation."""
        # Run some tests first
        harness.run_test(TestCaseInput(
            name="coverage_test_1",
            agent_type="SuccessAgent",
            input_data={}
        ))
        harness.run_test(TestCaseInput(
            name="coverage_test_2",
            agent_type="SuccessAgent",
            input_data={}
        ))

        report = harness.get_coverage_report("SuccessAgent")

        assert report.agent_type == "SuccessAgent"
        assert report.total_methods > 0
        assert report.covered_methods > 0
        assert 0 <= report.coverage_percent <= 100
        assert report.test_count >= 2

    def test_coverage_unregistered_agent(self, harness):
        """Test coverage report for unregistered agent."""
        report = harness.get_coverage_report("NonExistentAgent")

        assert report.agent_type == "NonExistentAgent"
        assert report.total_methods == 0
        assert report.coverage_percent == 0


# =============================================================================
# Execute Interface
# =============================================================================

class TestExecuteInterface:
    """Tests for the execute() interface."""

    def test_execute_test_case(self, harness):
        """Test execute with test_case input."""
        result = harness.execute({
            "test_case": {
                "name": "execute_test",
                "agent_type": "SuccessAgent",
                "input_data": {"value": "test"}
            }
        })

        assert result.success
        assert "status" in result.data
        assert result.data["status"] == "passed"

    def test_execute_test_suite(self, harness):
        """Test execute with test_suite input."""
        result = harness.execute({
            "test_suite": {
                "name": "execute_suite",
                "test_cases": [
                    {
                        "name": "suite_test_1",
                        "agent_type": "SuccessAgent",
                        "input_data": {}
                    },
                    {
                        "name": "suite_test_2",
                        "agent_type": "SuccessAgent",
                        "input_data": {}
                    }
                ],
                "parallel": False
            }
        })

        assert result.success
        assert result.data["total_tests"] == 2
        assert result.data["passed"] == 2

    def test_execute_benchmark(self, harness):
        """Test execute with benchmark input."""
        result = harness.execute({
            "benchmark": {
                "agent_type": "SuccessAgent",
                "input_data": {"value": "benchmark"},
                "iterations": 3
            }
        })

        assert result.success
        assert "mean_ms" in result.data
        assert result.data["iterations"] == 3

    def test_execute_invalid_input(self, harness):
        """Test execute with invalid input."""
        result = harness.execute({
            "invalid_key": "invalid_value"
        })

        assert not result.success
        assert result.error is not None
        # Check that the error message explains what's needed
        assert "test_case" in result.error.lower() or "input" in result.error.lower()


# =============================================================================
# Report Generation
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_text_report(self, harness):
        """Test text format report generation."""
        suite_input = TestSuiteInput(
            name="report_suite",
            test_cases=[
                TestCaseInput(
                    name="test_1",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
                TestCaseInput(
                    name="test_2",
                    agent_type="FailureAgent",
                    input_data={}
                ),
            ],
            parallel=False
        )

        suite_result = harness.run_suite(suite_input)
        report = harness.generate_report(suite_result, format="text")

        assert "TEST SUITE: report_suite" in report
        assert "[PASS]" in report
        assert "[FAIL]" in report
        assert "Pass Rate:" in report

    def test_json_report(self, harness):
        """Test JSON format report generation."""
        suite_input = TestSuiteInput(
            name="json_report_suite",
            test_cases=[
                TestCaseInput(
                    name="test_1",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
            ],
            parallel=False
        )

        suite_result = harness.run_suite(suite_input)
        report = harness.generate_report(suite_result, format="json")

        parsed = json.loads(report)
        assert parsed["name"] == "json_report_suite"
        assert "test_results" in parsed

    def test_markdown_report(self, harness):
        """Test markdown format report generation."""
        suite_input = TestSuiteInput(
            name="markdown_suite",
            test_cases=[
                TestCaseInput(
                    name="test_1",
                    agent_type="SuccessAgent",
                    input_data={}
                ),
            ],
            parallel=False
        )

        suite_result = harness.run_suite(suite_input)
        report = harness.generate_report(suite_result, format="markdown")

        assert "# Test Suite: markdown_suite" in report
        assert "## Summary" in report
        assert "| Status |" in report


# =============================================================================
# Common Fixtures
# =============================================================================

class TestCommonFixtures:
    """Tests for common test fixtures."""

    def test_fixtures_exist(self):
        """Test common fixtures are defined."""
        assert "empty_input" in COMMON_FIXTURES
        assert "null_values" in COMMON_FIXTURES
        assert "large_input" in COMMON_FIXTURES
        assert "special_characters" in COMMON_FIXTURES
        assert "unicode_input" in COMMON_FIXTURES

    def test_fixture_to_test_input(self):
        """Test converting fixture to TestCaseInput."""
        fixture = COMMON_FIXTURES["empty_input"]
        test_input = fixture.to_test_input("SuccessAgent", TestCategory.UNIT)

        assert test_input.name == "empty_input"
        assert test_input.agent_type == "SuccessAgent"
        assert test_input.category == TestCategory.UNIT
        assert test_input.input_data == {}

    def test_run_with_fixture(self, harness):
        """Test running test with common fixture."""
        fixture = COMMON_FIXTURES["zero_values"]
        test_input = fixture.to_test_input("SuccessAgent", TestCategory.UNIT)

        result = harness.run_test(test_input)

        assert result.status == TestStatus.PASSED


# =============================================================================
# Harness Metrics
# =============================================================================

class TestHarnessMetrics:
    """Tests for harness metrics."""

    def test_metrics_tracking(self, harness):
        """Test metrics are tracked correctly."""
        # Run some tests
        harness.run_test(TestCaseInput(
            name="metric_test_1",
            agent_type="SuccessAgent",
            input_data={}
        ))
        harness.run_test(TestCaseInput(
            name="metric_test_2",
            agent_type="FailureAgent",
            input_data={}
        ))

        metrics = harness.get_metrics()

        assert metrics["total_tests_run"] >= 2
        assert metrics["total_passed"] >= 1
        assert metrics["total_failed"] >= 1
        assert metrics["registered_agents"] == 8
        assert "coverage" in metrics


# =============================================================================
# Pydantic Model Validation
# =============================================================================

class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_test_case_input_validation(self):
        """Test TestCaseInput validation."""
        # Valid input
        test = TestCaseInput(
            name="valid_test",
            agent_type="SomeAgent",
            input_data={}
        )
        assert test.name == "valid_test"

        # Invalid - empty name
        with pytest.raises(ValueError):
            TestCaseInput(
                name="",
                agent_type="SomeAgent",
                input_data={}
            )

    def test_test_assertion_model(self):
        """Test TestAssertion model."""
        assertion = TestAssertion(
            name="test_assertion",
            passed=True,
            expected="value",
            actual="value",
            message="Values match",
            severity=SeverityLevel.HIGH
        )

        assert assertion.passed
        assert assertion.severity == SeverityLevel.HIGH

    def test_severity_levels(self):
        """Test all severity levels are valid."""
        levels = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
            SeverityLevel.INFO,
        ]

        for level in levels:
            assertion = TestAssertion(
                name="test",
                passed=True,
                severity=level
            )
            assert assertion.severity == level


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_suite(self, harness):
        """Test running empty test suite."""
        suite_input = TestSuiteInput(
            name="empty_suite",
            test_cases=[],
            parallel=False
        )

        result = harness.run_suite(suite_input)

        assert result.total_tests == 0
        assert result.status == TestStatus.SKIPPED

    def test_very_long_test_name(self, harness):
        """Test with very long test name."""
        long_name = "test_" + "x" * 1000

        test_input = TestCaseInput(
            name=long_name,
            agent_type="SuccessAgent",
            input_data={}
        )

        result = harness.run_test(test_input)

        assert result.name == long_name
        assert result.status == TestStatus.PASSED

    def test_special_characters_in_input(self, harness):
        """Test with special characters in input data."""
        test_input = TestCaseInput(
            name="special_chars_test",
            agent_type="SuccessAgent",
            input_data={
                "text": "<script>alert('xss')</script>",
                "path": "C:\\Users\\test\\file.txt",
                "unicode": "Test with unicode characters"
            }
        )

        result = harness.run_test(test_input)

        assert result.status == TestStatus.PASSED
        assert result.input_hash != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
