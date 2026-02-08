# -*- coding: utf-8 -*-
"""
QA Test Harness Service REST API Router - AGENT-FOUND-009: QA Test Harness

FastAPI router providing 20 endpoints for test execution, golden file
management, performance benchmarking, coverage tracking, regression
detection, report generation, and health monitoring.

All endpoints are mounted under ``/api/v1/qa-test-harness``.

Endpoints:
    1.  POST   /v1/tests/run                    - Run single test case
    2.  POST   /v1/suites/run                   - Run test suite
    3.  GET    /v1/runs                          - List test runs
    4.  GET    /v1/runs/{run_id}                 - Get run details
    5.  GET    /v1/runs/{run_id}/assertions       - Get assertions for run
    6.  POST   /v1/tests/determinism             - Run determinism test
    7.  POST   /v1/tests/zero-hallucination      - Run zero-hallucination test
    8.  POST   /v1/tests/lineage                 - Run lineage test
    9.  POST   /v1/tests/regression              - Run regression test
    10. POST   /v1/golden-files                  - Save golden file
    11. GET    /v1/golden-files                  - List golden files
    12. GET    /v1/golden-files/{file_id}        - Get golden file
    13. POST   /v1/golden-files/{file_id}/compare - Compare with golden file
    14. POST   /v1/benchmarks/run                - Run benchmark
    15. GET    /v1/benchmarks/{agent_type}        - Get benchmark baseline
    16. GET    /v1/coverage/{agent_type}          - Get coverage report
    17. GET    /v1/coverage                       - Get all coverage
    18. POST   /v1/report                         - Generate report
    19. GET    /v1/statistics                     - Get QA statistics
    20. GET    /health                            - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; QA test harness router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RunTestBody(BaseModel):
        """Request body for running a single test case."""
        name: str = Field(..., description="Test case name")
        agent_type: str = Field(..., description="Agent type to test")
        category: str = Field(default="unit", description="Test category")
        input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
        expected_output: Optional[Dict[str, Any]] = Field(None, description="Expected output")
        golden_file_path: Optional[str] = Field(None, description="Golden file path")
        timeout_seconds: int = Field(default=60, description="Timeout seconds")
        tags: List[str] = Field(default_factory=list, description="Test tags")

    class RunSuiteBody(BaseModel):
        """Request body for running a test suite."""
        name: str = Field(..., description="Suite name")
        test_cases: List[Dict[str, Any]] = Field(..., description="Test cases")
        parallel: bool = Field(default=True, description="Parallel execution")
        max_workers: int = Field(default=4, description="Max workers")
        fail_fast: bool = Field(default=False, description="Stop on first failure")
        tags_include: List[str] = Field(default_factory=list, description="Include tags")
        tags_exclude: List[str] = Field(default_factory=list, description="Exclude tags")

    class DeterminismTestBody(BaseModel):
        """Request body for determinism test."""
        agent_type: str = Field(..., description="Agent type to test")
        input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
        iterations: int = Field(default=3, ge=2, description="Iterations")

    class ZeroHallucinationTestBody(BaseModel):
        """Request body for zero-hallucination test."""
        agent_type: str = Field(..., description="Agent type to test")
        input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
        checks: List[str] = Field(default_factory=list, description="Specific checks")

    class LineageTestBody(BaseModel):
        """Request body for lineage test."""
        agent_type: str = Field(..., description="Agent type to test")
        input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")

    class RegressionTestBody(BaseModel):
        """Request body for regression test."""
        agent_type: str = Field(..., description="Agent type to test")
        input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
        baseline_hash: Optional[str] = Field(None, description="Expected baseline hash")

    class SaveGoldenFileBody(BaseModel):
        """Request body for saving a golden file."""
        agent_type: str = Field(..., description="Agent type")
        name: str = Field(..., description="Golden file name")
        input_data: Dict[str, Any] = Field(..., description="Input data")
        output_data: Dict[str, Any] = Field(..., description="Output data")
        description: str = Field(default="", description="Description")

    class CompareGoldenBody(BaseModel):
        """Request body for golden file comparison."""
        agent_result: Dict[str, Any] = Field(..., description="Agent result to compare")

    class RunBenchmarkBody(BaseModel):
        """Request body for running a benchmark."""
        agent_type: str = Field(..., description="Agent type to benchmark")
        input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
        iterations: int = Field(default=10, ge=1, description="Iterations")
        warmup: int = Field(default=2, ge=0, description="Warmup iterations")
        threshold_ms: Optional[float] = Field(None, description="Threshold ms")

    class GenerateReportBody(BaseModel):
        """Request body for report generation."""
        suite_result: Dict[str, Any] = Field(..., description="Suite result data")
        format: str = Field(default="markdown", description="Report format")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/qa-test-harness",
        tags=["qa-test-harness"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract QATestHarnessService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        QATestHarnessService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "qa_test_harness_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="QA test harness service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # 1. Run single test
    @router.post("/v1/tests/run")
    async def run_test(
        body: RunTestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a single test case."""
        service = _get_service(request)
        try:
            from greenlang.qa_test_harness.models import TestCaseInput, TestCategory
            test_input = TestCaseInput(
                name=body.name,
                agent_type=body.agent_type,
                category=TestCategory(body.category),
                input_data=body.input_data,
                expected_output=body.expected_output,
                golden_file_path=body.golden_file_path,
                timeout_seconds=body.timeout_seconds,
                tags=body.tags,
            )
            result = service.run_test(test_input)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 2. Run test suite
    @router.post("/v1/suites/run")
    async def run_suite(
        body: RunSuiteBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a test suite."""
        service = _get_service(request)
        try:
            from greenlang.qa_test_harness.models import TestSuiteInput, TestCaseInput
            test_cases = [TestCaseInput(**tc) for tc in body.test_cases]
            suite_input = TestSuiteInput(
                name=body.name,
                test_cases=test_cases,
                parallel=body.parallel,
                max_workers=body.max_workers,
                fail_fast=body.fail_fast,
                tags_include=body.tags_include,
                tags_exclude=body.tags_exclude,
            )
            result = service.run_suite(suite_input)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 3. List test runs
    @router.get("/v1/runs")
    async def list_runs(
        agent_type: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List test runs."""
        service = _get_service(request)
        runs = service.runner.get_run_history(agent_type=agent_type)
        runs = runs[-limit:]
        return {
            "runs": [r.model_dump(mode="json") for r in runs],
            "count": len(runs),
        }

    # 4. Get run details
    @router.get("/v1/runs/{run_id}")
    async def get_run(
        run_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get test run details."""
        service = _get_service(request)
        run = service.runner.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Test run {run_id} not found",
            )
        return run.model_dump(mode="json")

    # 5. Get assertions for run
    @router.get("/v1/runs/{run_id}/assertions")
    async def get_run_assertions(
        run_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get assertions for a test run."""
        service = _get_service(request)
        run = service.runner.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Test run {run_id} not found",
            )
        return {
            "run_id": run_id,
            "assertions": run.assertions,
            "count": len(run.assertions),
        }

    # 6. Determinism test
    @router.post("/v1/tests/determinism")
    async def run_determinism_test(
        body: DeterminismTestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a determinism test."""
        service = _get_service(request)
        try:
            result = service.test_determinism(
                agent_type=body.agent_type,
                input_data=body.input_data,
                iterations=body.iterations,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 7. Zero-hallucination test
    @router.post("/v1/tests/zero-hallucination")
    async def run_zero_hallucination_test(
        body: ZeroHallucinationTestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a zero-hallucination test."""
        service = _get_service(request)
        try:
            result = service.test_zero_hallucination(
                agent_type=body.agent_type,
                input_data=body.input_data,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 8. Lineage test
    @router.post("/v1/tests/lineage")
    async def run_lineage_test(
        body: LineageTestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a lineage completeness test."""
        service = _get_service(request)
        try:
            result = service.test_lineage(
                agent_type=body.agent_type,
                input_data=body.input_data,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 9. Regression test
    @router.post("/v1/tests/regression")
    async def run_regression_test(
        body: RegressionTestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a regression test."""
        service = _get_service(request)
        try:
            result = service.test_regression(
                agent_type=body.agent_type,
                input_data=body.input_data,
                baseline_hash=body.baseline_hash,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 10. Save golden file
    @router.post("/v1/golden-files")
    async def save_golden_file(
        body: SaveGoldenFileBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Save a golden file."""
        service = _get_service(request)
        try:
            entry = service.save_golden_file(
                agent_type=body.agent_type,
                name=body.name,
                input_data=body.input_data,
                output_data=body.output_data,
            )
            return entry.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 11. List golden files
    @router.get("/v1/golden-files")
    async def list_golden_files(
        agent_type: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List golden files."""
        service = _get_service(request)
        files = service.golden_file_manager.list_golden_files(agent_type=agent_type)
        return {
            "golden_files": [f.model_dump(mode="json") for f in files],
            "count": len(files),
        }

    # 12. Get golden file
    @router.get("/v1/golden-files/{file_id}")
    async def get_golden_file(
        file_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a golden file entry."""
        service = _get_service(request)
        entry = service.golden_file_manager.get_golden_file(file_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"Golden file {file_id} not found",
            )
        return entry.model_dump(mode="json")

    # 13. Compare with golden file
    @router.post("/v1/golden-files/{file_id}/compare")
    async def compare_golden_file(
        file_id: str,
        body: CompareGoldenBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Compare an agent result with a golden file."""
        service = _get_service(request)
        try:
            entry = service.golden_file_manager.get_golden_file(file_id)
            if entry is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Golden file {file_id} not found",
                )

            # Create a simple result wrapper
            class _ResultWrapper:
                def __init__(self, data: Dict[str, Any]) -> None:
                    self.data = data

            agent_result = _ResultWrapper(body.agent_result)
            assertions = service.golden_file_manager.compare_with_golden(
                agent_result, entry,
            )
            return {
                "assertions": [a.model_dump(mode="json") for a in assertions],
                "count": len(assertions),
                "all_passed": all(a.passed for a in assertions),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 14. Run benchmark
    @router.post("/v1/benchmarks/run")
    async def run_benchmark(
        body: RunBenchmarkBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run a performance benchmark."""
        service = _get_service(request)
        try:
            result = service.benchmark(
                agent_type=body.agent_type,
                input_data=body.input_data,
                iterations=body.iterations,
                threshold_ms=body.threshold_ms,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 15. Get benchmark baseline
    @router.get("/v1/benchmarks/{agent_type}")
    async def get_benchmark_baseline(
        agent_type: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get performance benchmark baseline for an agent type."""
        service = _get_service(request)
        baseline = service.benchmarker.get_baseline(agent_type)
        if baseline is None:
            raise HTTPException(
                status_code=404,
                detail=f"No benchmark baseline for {agent_type}",
            )
        return baseline.model_dump(mode="json")

    # 16. Get coverage report
    @router.get("/v1/coverage/{agent_type}")
    async def get_coverage_report(
        agent_type: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get coverage report for an agent type."""
        service = _get_service(request)
        report = service.get_coverage(agent_type)
        return report.model_dump(mode="json")

    # 17. Get all coverage
    @router.get("/v1/coverage")
    async def get_all_coverage(
        request: Request,
    ) -> Dict[str, Any]:
        """Get coverage reports for all tracked agents."""
        service = _get_service(request)
        reports = service.coverage_tracker.get_all_reports(
            agent_classes=service._agent_registry,
        )
        return {
            "coverage": {
                k: v.model_dump(mode="json")
                for k, v in reports.items()
            },
            "count": len(reports),
        }

    # 18. Generate report
    @router.post("/v1/report")
    async def generate_report(
        body: GenerateReportBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Generate a test report."""
        service = _get_service(request)
        try:
            from greenlang.qa_test_harness.models import TestSuiteResult
            suite_result = TestSuiteResult(**body.suite_result)
            report_content = service.generate_report(
                suite_result, format=body.format,
            )
            return {
                "report": report_content,
                "format": body.format,
            }
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 19. Get statistics
    @router.get("/v1/statistics")
    async def get_statistics(
        request: Request,
    ) -> Dict[str, Any]:
        """Get QA test harness statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")

    # 20. Health check
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """QA test harness service health check endpoint."""
        return {"status": "healthy", "service": "qa-test-harness"}


__all__ = [
    "router",
]
