# -*- coding: utf-8 -*-
"""
QA Test Harness Service Data Models - AGENT-FOUND-009: QA Test Harness

Pydantic v2 data models for the QA Test Harness SDK. Re-exports the Layer 1
enumerations, models, fixtures, and constants from the foundation agent, and
defines additional SDK models for test runs, golden file entries, performance
baselines, coverage snapshots, regression baselines, statistics, and
request/response wrappers.

Models:
    - Re-exported enums: TestStatus, TestCategory, SeverityLevel
    - Re-exported Layer 1: TestAssertion, TestCaseInput, TestCaseResult,
        TestSuiteInput, TestSuiteResult, GoldenFileSpec, PerformanceBenchmark,
        CoverageReport
    - Re-exported fixtures: TestFixture, COMMON_FIXTURES
    - SDK models: TestRun, GoldenFileEntry, PerformanceBaseline,
        CoverageSnapshot, RegressionBaseline, QAStatistics
    - Request/Response: RunTestRequest, RunSuiteRequest, BenchmarkRequest,
        ReportRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.qa_test_harness import (
    TestStatus,
    TestCategory,
    SeverityLevel,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.qa_test_harness import (
    TestAssertion,
    TestCaseInput,
    TestCaseResult,
    TestSuiteInput,
    TestSuiteResult,
    GoldenFileSpec,
    PerformanceBenchmark,
    CoverageReport,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 fixtures
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.qa_test_harness import (
    TestFixture,
    COMMON_FIXTURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# SDK Data Models
# =============================================================================


class TestRun(BaseModel):
    """Record of a single test case execution for persistent storage.

    Captures the full state and results of a test run for audit
    and analysis purposes.

    Attributes:
        run_id: Unique identifier for this test run.
        suite_id: ID of the parent suite (if applicable).
        test_case_id: ID of the test case executed.
        agent_type: Type of agent being tested.
        category: Test category classification.
        status: Final test status.
        assertions: List of assertion results.
        input_hash: SHA-256 hash of the test input data.
        output_hash: SHA-256 hash of the agent output data.
        duration_ms: Test execution duration in milliseconds.
        error_message: Error description if test failed or errored.
        created_at: Timestamp when the run was created.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this test run",
    )
    suite_id: str = Field(
        default="", description="ID of the parent suite"
    )
    test_case_id: str = Field(
        ..., description="ID of the test case executed"
    )
    agent_type: str = Field(
        ..., description="Type of agent being tested"
    )
    category: TestCategory = Field(
        ..., description="Test category classification"
    )
    status: TestStatus = Field(
        ..., description="Final test status"
    )
    assertions: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of assertion results"
    )
    input_hash: str = Field(
        default="", description="SHA-256 hash of the test input data"
    )
    output_hash: str = Field(
        default="", description="SHA-256 hash of the agent output data"
    )
    duration_ms: float = Field(
        default=0.0, description="Test execution duration in milliseconds"
    )
    error_message: Optional[str] = Field(
        None, description="Error description if test failed or errored"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the run was created"
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation"
    )

    model_config = {"extra": "forbid"}

    @field_validator("test_case_id")
    @classmethod
    def validate_test_case_id(cls, v: str) -> str:
        """Validate test_case_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("test_case_id must be non-empty")
        return v

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("agent_type must be non-empty")
        return v


class GoldenFileEntry(BaseModel):
    """Record of a golden file for snapshot testing.

    Stores metadata about a golden file including its content hash,
    versioning, and lifecycle state.

    Attributes:
        file_id: Unique identifier for this golden file entry.
        agent_type: Type of agent this golden file applies to.
        name: Human-readable name for the golden file.
        version: Version string for the golden file content.
        input_hash: SHA-256 hash of the input data used to generate output.
        content_hash: SHA-256 hash of the golden file content.
        file_path: Filesystem path where the golden file is stored.
        created_at: Timestamp when the golden file was created.
        updated_at: Timestamp of the last update.
        created_by: User or system that created the golden file.
        is_active: Whether this golden file is currently active.
    """

    file_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this golden file entry",
    )
    agent_type: str = Field(
        ..., description="Type of agent this golden file applies to"
    )
    name: str = Field(
        ..., description="Human-readable name for the golden file"
    )
    version: str = Field(
        default="1.0.0", description="Version string for the golden file content"
    )
    input_hash: str = Field(
        default="", description="SHA-256 hash of the input data"
    )
    content_hash: str = Field(
        default="", description="SHA-256 hash of the golden file content"
    )
    file_path: str = Field(
        default="", description="Filesystem path where the golden file is stored"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the golden file was created"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp of the last update"
    )
    created_by: str = Field(
        default="system", description="User or system that created the golden file"
    )
    is_active: bool = Field(
        default=True, description="Whether this golden file is currently active"
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("agent_type must be non-empty")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class PerformanceBaseline(BaseModel):
    """Stored performance baseline for benchmark comparison.

    A performance snapshot used as a reference point for detecting
    performance regressions in subsequent benchmark runs.

    Attributes:
        baseline_id: Unique identifier for this baseline.
        agent_type: Type of agent this baseline applies to.
        operation: Operation being benchmarked.
        p95_ms: 95th percentile latency in milliseconds.
        p99_ms: 99th percentile latency in milliseconds.
        mean_ms: Mean latency in milliseconds.
        threshold_ms: Performance threshold in milliseconds.
        created_at: Timestamp when the baseline was created.
        is_active: Whether this baseline is currently active.
    """

    baseline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this baseline",
    )
    agent_type: str = Field(
        ..., description="Type of agent this baseline applies to"
    )
    operation: str = Field(
        default="execute", description="Operation being benchmarked"
    )
    p95_ms: float = Field(
        default=0.0, description="95th percentile latency in milliseconds"
    )
    p99_ms: float = Field(
        default=0.0, description="99th percentile latency in milliseconds"
    )
    mean_ms: float = Field(
        default=0.0, description="Mean latency in milliseconds"
    )
    threshold_ms: Optional[float] = Field(
        None, description="Performance threshold in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the baseline was created"
    )
    is_active: bool = Field(
        default=True, description="Whether this baseline is currently active"
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("agent_type must be non-empty")
        return v


class CoverageSnapshot(BaseModel):
    """Point-in-time snapshot of test coverage for an agent.

    Captures coverage metrics at a specific point in time for
    trend analysis and compliance reporting.

    Attributes:
        snapshot_id: Unique identifier for this snapshot.
        agent_type: Type of agent this snapshot applies to.
        total_methods: Total number of public methods on the agent.
        covered_methods: Number of methods covered by tests.
        coverage_percent: Coverage percentage (0-100).
        uncovered_methods: List of uncovered method names.
        created_at: Timestamp when the snapshot was taken.
    """

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this snapshot",
    )
    agent_type: str = Field(
        ..., description="Type of agent this snapshot applies to"
    )
    total_methods: int = Field(
        default=0, ge=0, description="Total number of public methods"
    )
    covered_methods: int = Field(
        default=0, ge=0, description="Number of methods covered by tests"
    )
    coverage_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Coverage percentage"
    )
    uncovered_methods: List[str] = Field(
        default_factory=list, description="List of uncovered method names"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the snapshot was taken"
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("agent_type must be non-empty")
        return v


class RegressionBaseline(BaseModel):
    """Stored regression baseline for output comparison.

    A snapshot of expected output hash for a given input hash, used to
    detect regressions when agent behavior changes.

    Attributes:
        baseline_id: Unique identifier for this baseline.
        agent_type: Type of agent this baseline applies to.
        input_hash: SHA-256 hash of the input data.
        output_hash: SHA-256 hash of the expected output.
        created_at: Timestamp when the baseline was created.
        is_active: Whether this baseline is currently active.
    """

    baseline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this baseline",
    )
    agent_type: str = Field(
        ..., description="Type of agent this baseline applies to"
    )
    input_hash: str = Field(
        ..., description="SHA-256 hash of the input data"
    )
    output_hash: str = Field(
        ..., description="SHA-256 hash of the expected output"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the baseline was created"
    )
    is_active: bool = Field(
        default=True, description="Whether this baseline is currently active"
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("agent_type must be non-empty")
        return v

    @field_validator("input_hash", "output_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate hash is non-empty."""
        if not v or not v.strip():
            raise ValueError("hash must be non-empty")
        return v


class QAStatistics(BaseModel):
    """Aggregated statistics for the QA test harness.

    Attributes:
        total_runs: Total number of test runs executed.
        passed: Number of test runs that passed.
        failed: Number of test runs that failed.
        skipped: Number of test runs that were skipped.
        errors: Number of test runs that errored.
        pass_rate: Pass rate percentage (0-100).
        avg_duration_ms: Average test duration in milliseconds.
        regressions_detected: Total number of regressions detected.
        golden_file_mismatches: Total number of golden file mismatches.
        coverage_percent: Overall coverage percentage.
    """

    total_runs: int = Field(
        default=0, description="Total number of test runs executed"
    )
    passed: int = Field(
        default=0, description="Number of test runs that passed"
    )
    failed: int = Field(
        default=0, description="Number of test runs that failed"
    )
    skipped: int = Field(
        default=0, description="Number of test runs that were skipped"
    )
    errors: int = Field(
        default=0, description="Number of test runs that errored"
    )
    pass_rate: float = Field(
        default=0.0, description="Pass rate percentage"
    )
    avg_duration_ms: float = Field(
        default=0.0, description="Average test duration in milliseconds"
    )
    regressions_detected: int = Field(
        default=0, description="Total number of regressions detected"
    )
    golden_file_mismatches: int = Field(
        default=0, description="Total number of golden file mismatches"
    )
    coverage_percent: float = Field(
        default=0.0, description="Overall coverage percentage"
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request / Response Models
# =============================================================================


class RunTestRequest(BaseModel):
    """Request body for running a single test case.

    Attributes:
        test_case: Test case specification to execute.
    """

    test_case: Dict[str, Any] = Field(
        ..., description="Test case specification to execute"
    )

    model_config = {"extra": "forbid"}


class RunSuiteRequest(BaseModel):
    """Request body for running a test suite.

    Attributes:
        suite: Test suite specification to execute.
    """

    suite: Dict[str, Any] = Field(
        ..., description="Test suite specification to execute"
    )

    model_config = {"extra": "forbid"}


class BenchmarkRequest(BaseModel):
    """Request body for running a performance benchmark.

    Attributes:
        agent_type: Type of agent to benchmark.
        input_data: Input data for the benchmark.
        iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        threshold_ms: Performance threshold in milliseconds.
    """

    agent_type: str = Field(
        ..., description="Type of agent to benchmark"
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the benchmark"
    )
    iterations: int = Field(
        default=10, ge=1, description="Number of benchmark iterations"
    )
    warmup: int = Field(
        default=2, ge=0, description="Number of warmup iterations"
    )
    threshold_ms: Optional[float] = Field(
        None, description="Performance threshold in milliseconds"
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("agent_type must be non-empty")
        return v


class ReportRequest(BaseModel):
    """Request body for generating a test report.

    Attributes:
        suite_result: Suite result data to generate report from.
        format: Report format (text, json, markdown, html).
    """

    suite_result: Dict[str, Any] = Field(
        ..., description="Suite result data to generate report from"
    )
    format: str = Field(
        default="markdown", description="Report format"
    )

    model_config = {"extra": "forbid"}

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate format is a supported type."""
        allowed = {"text", "json", "markdown", "html"}
        if v.lower() not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v.lower()


__all__ = [
    # Re-exported enums
    "TestStatus",
    "TestCategory",
    "SeverityLevel",
    # Re-exported Layer 1 models
    "TestAssertion",
    "TestCaseInput",
    "TestCaseResult",
    "TestSuiteInput",
    "TestSuiteResult",
    "GoldenFileSpec",
    "PerformanceBenchmark",
    "CoverageReport",
    # Re-exported fixtures
    "TestFixture",
    "COMMON_FIXTURES",
    # SDK models
    "TestRun",
    "GoldenFileEntry",
    "PerformanceBaseline",
    "CoverageSnapshot",
    "RegressionBaseline",
    "QAStatistics",
    # Request / Response
    "RunTestRequest",
    "RunSuiteRequest",
    "BenchmarkRequest",
    "ReportRequest",
]
