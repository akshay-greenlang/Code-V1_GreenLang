# -*- coding: utf-8 -*-
"""
GL-FOUND-X-009: GreenLang QA Test Harness Service SDK
=====================================================

This package provides the QA test harness, test execution, assertion engine,
golden file management, regression detection, performance benchmarking,
coverage tracking, report generation, and provenance tracking SDK for the
GreenLang framework. It supports:

- Single test case and test suite execution with parallel/sequential modes
- Category-specific assertions (zero-hallucination, determinism, lineage, etc.)
- Golden file / snapshot testing with content hash verification
- Regression detection with baseline management and historical consistency
- Performance benchmarking with warmup, p95/p99, threshold checking
- Method-level test coverage tracking with snapshots
- Report generation in text, JSON, markdown, and HTML formats
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_QA_TEST_HARNESS_ env prefix

Key Components:
    - config: QATestHarnessConfig with GL_QA_TEST_HARNESS_ env prefix
    - models: Pydantic v2 models for all data structures
    - test_runner: Test execution engine with parallel/sequential modes
    - assertion_engine: Category-specific assertion engine
    - golden_file_manager: Golden file / snapshot testing manager
    - regression_detector: Regression detection engine
    - performance_benchmarker: Performance benchmarking engine
    - coverage_tracker: Test coverage tracking engine
    - report_generator: Multi-format report generation
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: QATestHarnessService facade

Example:
    >>> from greenlang.qa_test_harness import QATestHarnessService
    >>> service = QATestHarnessService()
    >>> service.register_agent("MyAgent", MyAgentClass)
    >>> result = service.run_test(test_input)
    >>> print(result.status)
    passed

Agent ID: GL-FOUND-X-009
Agent Name: Quality Gate & Test Harness Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-009"
__agent_name__ = "Quality Gate & Test Harness Agent"

# SDK availability flag
QA_TEST_HARNESS_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.qa_test_harness.config import (
    QATestHarnessConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, Layer 1, SDK)
# ---------------------------------------------------------------------------
from greenlang.qa_test_harness.models import (
    # Enumerations
    TestStatus,
    TestCategory,
    SeverityLevel,
    # Layer 1 models
    TestAssertion,
    TestCaseInput,
    TestCaseResult,
    TestSuiteInput,
    TestSuiteResult,
    GoldenFileSpec,
    PerformanceBenchmark,
    CoverageReport,
    # Layer 1 fixtures
    TestFixture,
    COMMON_FIXTURES,
    # SDK models
    TestRun,
    GoldenFileEntry,
    PerformanceBaseline,
    CoverageSnapshot,
    RegressionBaseline,
    QAStatistics,
    # Request / Response
    RunTestRequest,
    RunSuiteRequest,
    BenchmarkRequest,
    ReportRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.qa_test_harness.test_runner import TestRunner
from greenlang.qa_test_harness.assertion_engine import AssertionEngine
from greenlang.qa_test_harness.golden_file_manager import GoldenFileManager
from greenlang.qa_test_harness.regression_detector import RegressionDetector
from greenlang.qa_test_harness.performance_benchmarker import PerformanceBenchmarker
from greenlang.qa_test_harness.coverage_tracker import CoverageTracker
from greenlang.qa_test_harness.report_generator import ReportGenerator
from greenlang.qa_test_harness.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.qa_test_harness.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    qa_test_runs_total,
    qa_test_duration_seconds,
    qa_test_assertions_total,
    qa_test_pass_rate,
    qa_test_failures_total,
    qa_test_regressions_total,
    qa_golden_file_mismatches_total,
    qa_performance_threshold_breaches_total,
    qa_coverage_percent,
    qa_suites_total,
    qa_cache_hits_total,
    qa_cache_misses_total,
    # Helper functions
    record_test_run,
    record_assertion,
    record_failure,
    record_regression,
    record_golden_file_mismatch,
    record_performance_breach,
    update_coverage,
    update_pass_rate,
    record_suite,
    record_cache_hit,
    record_cache_miss,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.qa_test_harness.setup import (
    QATestHarnessService,
    configure_qa_test_harness,
    get_qa_test_harness,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "QA_TEST_HARNESS_SDK_AVAILABLE",
    # Configuration
    "QATestHarnessConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "TestStatus",
    "TestCategory",
    "SeverityLevel",
    # Layer 1 models
    "TestAssertion",
    "TestCaseInput",
    "TestCaseResult",
    "TestSuiteInput",
    "TestSuiteResult",
    "GoldenFileSpec",
    "PerformanceBenchmark",
    "CoverageReport",
    # Layer 1 fixtures
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
    # Core engines
    "TestRunner",
    "AssertionEngine",
    "GoldenFileManager",
    "RegressionDetector",
    "PerformanceBenchmarker",
    "CoverageTracker",
    "ReportGenerator",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "qa_test_runs_total",
    "qa_test_duration_seconds",
    "qa_test_assertions_total",
    "qa_test_pass_rate",
    "qa_test_failures_total",
    "qa_test_regressions_total",
    "qa_golden_file_mismatches_total",
    "qa_performance_threshold_breaches_total",
    "qa_coverage_percent",
    "qa_suites_total",
    "qa_cache_hits_total",
    "qa_cache_misses_total",
    # Metric helper functions
    "record_test_run",
    "record_assertion",
    "record_failure",
    "record_regression",
    "record_golden_file_mismatch",
    "record_performance_breach",
    "update_coverage",
    "update_pass_rate",
    "record_suite",
    "record_cache_hit",
    "record_cache_miss",
    # Service setup facade
    "QATestHarnessService",
    "configure_qa_test_harness",
    "get_qa_test_harness",
    "get_router",
]
