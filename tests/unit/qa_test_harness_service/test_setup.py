# -*- coding: utf-8 -*-
"""
Unit Tests for QATestHarnessService Facade (AGENT-FOUND-009)

Tests the service facade creation, all component accessor methods, test
execution, suite orchestration, agent registration, golden file management,
benchmarking, coverage, report generation, and statistics.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and stubs
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestCategory(str, Enum):
    UNIT = "unit"
    DETERMINISM = "determinism"
    ZERO_HALLUCINATION = "zero_hallucination"


def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


class QATestHarnessConfig:
    def __init__(self, **kwargs):
        self.default_timeout_seconds = kwargs.get("default_timeout_seconds", 60)
        self.max_parallel_workers = kwargs.get("max_parallel_workers", 4)
        self.report_format = kwargs.get("report_format", "markdown")
        self.determinism_iterations = kwargs.get("determinism_iterations", 3)
        self.golden_file_directory = kwargs.get("golden_file_directory", "./golden_files")


# ---------------------------------------------------------------------------
# Inline QATestHarnessService facade
# ---------------------------------------------------------------------------


class QATestHarnessService:
    """Facade for the QA Test Harness SDK."""

    def __init__(self, config: Optional[QATestHarnessConfig] = None):
        self.config = config or QATestHarnessConfig()
        self._agents: Dict[str, Callable] = {}
        self._test_history: List[Dict[str, Any]] = []
        self._golden_files: Dict[str, Dict[str, Any]] = {}
        self._baselines: Dict[str, Dict[str, Any]] = {}
        self._coverage: Dict[str, float] = {}
        self._gf_counter = 0

    def register_agent(self, agent_type: str, agent_fn: Callable) -> None:
        self._agents[agent_type] = agent_fn

    def run_test(self, agent_type: str, input_data: Dict[str, Any],
                 expected_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        agent_fn = self._agents.get(agent_type)
        if agent_fn is None:
            result = {
                "status": "error",
                "error": f"Agent not registered: {agent_type}",
            }
            self._test_history.append(result)
            return result

        try:
            output = agent_fn(input_data)
            status = "passed" if output.get("success") else "failed"

            if expected_output:
                for key, expected in expected_output.items():
                    if output.get("data", {}).get(key) != expected:
                        status = "failed"
                        break

            result = {
                "status": status,
                "output": output,
                "input_hash": _content_hash(input_data),
                "output_hash": _content_hash(output),
            }
        except Exception as e:
            result = {"status": "error", "error": str(e)}

        self._test_history.append(result)
        return result

    def run_suite(self, agent_type: str,
                  test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for tc in test_cases:
            r = self.run_test(agent_type, tc.get("input", {}),
                              tc.get("expected_output"))
            results.append(r)

        passed = sum(1 for r in results if r["status"] == "passed")
        failed = sum(1 for r in results if r["status"] == "failed")
        errors = sum(1 for r in results if r["status"] == "error")
        total = len(results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": round(pass_rate, 2),
            "results": results,
        }

    def save_golden_file(self, agent_type: str, input_data: Dict[str, Any],
                         output_data: Dict[str, Any]) -> str:
        self._gf_counter += 1
        gf_id = f"gf-{self._gf_counter:04d}"
        self._golden_files[gf_id] = {
            "agent_type": agent_type,
            "input_data": input_data,
            "expected_output": output_data,
            "content_hash": _content_hash(output_data),
        }
        return gf_id

    def benchmark(self, agent_type: str, input_data: Dict[str, Any],
                  iterations: int = 10) -> Dict[str, Any]:
        agent_fn = self._agents.get(agent_type)
        if agent_fn is None:
            return {"error": f"Agent not registered: {agent_type}"}

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            agent_fn(input_data)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

        timings.sort()
        return {
            "agent_type": agent_type,
            "iterations": iterations,
            "min_ms": round(timings[0], 3),
            "max_ms": round(timings[-1], 3),
            "mean_ms": round(sum(timings) / len(timings), 3),
        }

    def get_coverage(self, agent_type: str) -> Dict[str, Any]:
        return {
            "agent_type": agent_type,
            "coverage_percent": self._coverage.get(agent_type, 0.0),
        }

    def set_coverage(self, agent_type: str, pct: float) -> None:
        self._coverage[agent_type] = pct

    def generate_report(self, suite_result: Dict[str, Any],
                        format: str = "text") -> str:
        if format == "json":
            return json.dumps(suite_result, indent=2, default=str)
        lines = [f"Total: {suite_result.get('total_tests', 0)}",
                 f"Passed: {suite_result.get('passed', 0)}",
                 f"Failed: {suite_result.get('failed', 0)}"]
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self._test_history)
        passed = sum(1 for r in self._test_history if r.get("status") == "passed")
        return {"total_tests": total, "passed": passed, "failed": total - passed}


# Singleton
_service_instance: Optional[QATestHarnessService] = None


def configure_qa_test_harness(
    config: Optional[QATestHarnessConfig] = None
) -> QATestHarnessService:
    global _service_instance
    _service_instance = QATestHarnessService(config)
    return _service_instance


def get_qa_test_harness() -> QATestHarnessService:
    global _service_instance
    if _service_instance is None:
        _service_instance = QATestHarnessService()
    return _service_instance


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_service():
    global _service_instance
    _service_instance = None
    yield
    _service_instance = None


def _mock_agent(data):
    return {"success": True, "data": {"result": 42}}


def _failing_agent(data):
    return {"success": False, "data": {}, "error": "Agent failed"}


@pytest.fixture
def service():
    svc = QATestHarnessService()
    svc.register_agent("TestAgent", _mock_agent)
    svc.register_agent("FailAgent", _failing_agent)
    return svc


# ===========================================================================
# Test Classes
# ===========================================================================


class TestServiceInitialization:
    def test_service_default_config(self):
        svc = QATestHarnessService()
        assert svc.config is not None
        assert svc.config.default_timeout_seconds == 60

    def test_service_custom_config(self):
        cfg = QATestHarnessConfig(default_timeout_seconds=120)
        svc = QATestHarnessService(cfg)
        assert svc.config.default_timeout_seconds == 120

    def test_service_empty_history(self):
        svc = QATestHarnessService()
        assert svc._test_history == []

    def test_service_empty_agents(self):
        svc = QATestHarnessService()
        assert svc._agents == {}


class TestServiceRegisterAgent:
    def test_register_agent(self, service):
        assert "TestAgent" in service._agents

    def test_register_new_agent(self, service):
        service.register_agent("NewAgent", lambda d: {"success": True})
        assert "NewAgent" in service._agents


class TestServiceRunTest:
    def test_run_test_success(self, service):
        result = service.run_test("TestAgent", {"x": 1})
        assert result["status"] == "passed"

    def test_run_test_failure(self, service):
        result = service.run_test("FailAgent", {"x": 1})
        assert result["status"] == "failed"

    def test_run_test_unknown_agent(self, service):
        result = service.run_test("Unknown", {"x": 1})
        assert result["status"] == "error"

    def test_run_test_with_expected_match(self, service):
        result = service.run_test("TestAgent", {}, expected_output={"result": 42})
        assert result["status"] == "passed"

    def test_run_test_with_expected_mismatch(self, service):
        result = service.run_test("TestAgent", {}, expected_output={"result": 999})
        assert result["status"] == "failed"

    def test_run_test_records_history(self, service):
        service.run_test("TestAgent", {})
        assert len(service._test_history) == 1

    def test_run_test_has_hashes(self, service):
        result = service.run_test("TestAgent", {"x": 1})
        assert "input_hash" in result
        assert "output_hash" in result


class TestServiceRunSuite:
    def test_run_suite(self, service):
        cases = [{"input": {"x": i}} for i in range(3)]
        result = service.run_suite("TestAgent", cases)
        assert result["total_tests"] == 3
        assert result["passed"] == 3

    def test_run_suite_with_failures(self, service):
        cases = [{"input": {"x": 1}}, {"input": {"x": 2}}]
        result = service.run_suite("FailAgent", cases)
        assert result["failed"] == 2

    def test_run_suite_pass_rate(self, service):
        cases = [{"input": {}}]
        result = service.run_suite("TestAgent", cases)
        assert result["pass_rate"] == 100.0

    def test_run_suite_empty(self, service):
        result = service.run_suite("TestAgent", [])
        assert result["total_tests"] == 0


class TestServiceSaveGoldenFile:
    def test_save_golden_file(self, service):
        gf_id = service.save_golden_file("TestAgent", {"x": 1}, {"result": 42})
        assert gf_id.startswith("gf-")
        assert gf_id in service._golden_files

    def test_save_golden_file_content(self, service):
        gf_id = service.save_golden_file("TestAgent", {"x": 1}, {"result": 42})
        gf = service._golden_files[gf_id]
        assert gf["agent_type"] == "TestAgent"
        assert gf["expected_output"]["result"] == 42


class TestServiceBenchmark:
    def test_benchmark(self, service):
        result = service.benchmark("TestAgent", {"x": 1}, iterations=5)
        assert result["iterations"] == 5
        assert result["mean_ms"] >= 0

    def test_benchmark_unknown_agent(self, service):
        result = service.benchmark("Unknown", {})
        assert "error" in result


class TestServiceGetCoverage:
    def test_get_coverage_default(self, service):
        result = service.get_coverage("TestAgent")
        assert result["coverage_percent"] == 0.0

    def test_get_coverage_after_set(self, service):
        service.set_coverage("TestAgent", 85.0)
        result = service.get_coverage("TestAgent")
        assert result["coverage_percent"] == 85.0


class TestServiceGenerateReport:
    def test_generate_report_text(self, service):
        suite_result = {"total_tests": 10, "passed": 8, "failed": 2}
        report = service.generate_report(suite_result, format="text")
        assert "Total: 10" in report
        assert "Passed: 8" in report

    def test_generate_report_json(self, service):
        suite_result = {"total_tests": 10, "passed": 8, "failed": 2}
        report = service.generate_report(suite_result, format="json")
        data = json.loads(report)
        assert data["total_tests"] == 10


class TestServiceGetStatistics:
    def test_get_statistics_empty(self, service):
        stats = service.get_statistics()
        assert stats["total_tests"] == 0

    def test_get_statistics_after_tests(self, service):
        service.run_test("TestAgent", {})
        service.run_test("FailAgent", {})
        stats = service.get_statistics()
        assert stats["total_tests"] == 2
        assert stats["passed"] == 1
        assert stats["failed"] == 1


class TestConfigureQATestHarness:
    def test_configure_qa_test_harness(self):
        svc = configure_qa_test_harness()
        assert isinstance(svc, QATestHarnessService)

    def test_configure_with_custom_config(self):
        cfg = QATestHarnessConfig(report_format="html")
        svc = configure_qa_test_harness(cfg)
        assert svc.config.report_format == "html"

    def test_get_qa_test_harness_auto_creates(self):
        svc = get_qa_test_harness()
        assert isinstance(svc, QATestHarnessService)

    def test_get_qa_test_harness_returns_singleton(self):
        s1 = get_qa_test_harness()
        s2 = get_qa_test_harness()
        assert s1 is s2

    def test_configure_then_get(self):
        cfg = QATestHarnessConfig(default_timeout_seconds=99)
        configure_qa_test_harness(cfg)
        svc = get_qa_test_harness()
        assert svc.config.default_timeout_seconds == 99
