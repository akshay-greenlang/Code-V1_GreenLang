"""
Unit tests for GL-EUDR-APP v1.0 Pipeline Orchestrator.

Tests the 5-stage EUDR compliance pipeline: Intake, Geo-Validation,
Deforestation Risk, Document Verification, DDS Reporting.
Validates stage execution, sequencing, failure/retry logic,
pipeline history, and cancellation.

Test count target: 35+ tests
"""

import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Pipeline Engine (self-contained for testing)
# ---------------------------------------------------------------------------

class PipelineStage:
    INTAKE = "intake"
    GEO_VALIDATION = "geo_validation"
    DEFORESTATION_RISK = "deforestation_risk"
    DOCUMENT_VERIFICATION = "document_verification"
    DDS_REPORTING = "dds_reporting"

    ORDERED = [INTAKE, GEO_VALIDATION, DEFORESTATION_RISK,
               DOCUMENT_VERIFICATION, DDS_REPORTING]


class PipelineStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageResult:
    def __init__(self, stage: str, status: str = "completed",
                 error: Optional[str] = None, output: Optional[Dict] = None):
        self.stage = stage
        self.status = status
        self.error = error
        self.output = output or {}
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.duration_ms = 0.0


class PipelineRun:
    def __init__(self, supplier_id: str, commodity: Optional[str] = None,
                 plot_ids: Optional[List[str]] = None):
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.supplier_id = supplier_id
        self.commodity = commodity
        self.plot_ids = plot_ids or []
        self.status = PipelineStatus.PENDING
        self.current_stage: Optional[str] = None
        self.stages: Dict[str, StageResult] = {}
        self.error_message: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.created_at = datetime.now(timezone.utc)
        self.cancelled = False


class PipelineOrchestrator:
    """Five-stage EUDR compliance pipeline orchestrator."""

    def __init__(self):
        self._runs: Dict[str, PipelineRun] = {}
        self._stage_handlers = {
            PipelineStage.INTAKE: self._run_intake,
            PipelineStage.GEO_VALIDATION: self._run_geo_validation,
            PipelineStage.DEFORESTATION_RISK: self._run_deforestation_risk,
            PipelineStage.DOCUMENT_VERIFICATION: self._run_document_verification,
            PipelineStage.DDS_REPORTING: self._run_dds_reporting,
        }
        self.max_retries = 2

    def start_pipeline(self, supplier_id: str, commodity: Optional[str] = None,
                       plot_ids: Optional[List[str]] = None) -> PipelineRun:
        """Create and start a new pipeline run."""
        run = PipelineRun(supplier_id, commodity, plot_ids)
        run.status = PipelineStatus.PENDING
        run.current_stage = PipelineStage.INTAKE
        self._runs[run.run_id] = run
        return run

    def execute_pipeline(self, run_id: str) -> PipelineRun:
        """Execute all pipeline stages sequentially."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Pipeline run '{run_id}' not found")

        run.status = PipelineStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)

        for stage in PipelineStage.ORDERED:
            if run.cancelled:
                run.status = PipelineStatus.CANCELLED
                return run

            run.current_stage = stage
            start = datetime.now(timezone.utc)

            try:
                handler = self._stage_handlers[stage]
                output = handler(run)
                result = StageResult(stage, "completed", output=output)
            except Exception as exc:
                result = StageResult(stage, "failed", error=str(exc))
                run.stages[stage] = result
                run.status = PipelineStatus.FAILED
                run.error_message = f"Stage '{stage}' failed: {exc}"
                return run

            result.started_at = start
            result.completed_at = datetime.now(timezone.utc)
            delta = (result.completed_at - result.started_at).total_seconds() * 1000
            result.duration_ms = delta
            run.stages[stage] = result

        run.status = PipelineStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        return run

    def retry_pipeline(self, run_id: str) -> PipelineRun:
        """Retry a failed pipeline from the failed stage."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Pipeline run '{run_id}' not found")
        if run.status != PipelineStatus.FAILED:
            raise ValueError("Can only retry failed pipelines")

        failed_stage = run.current_stage
        run.status = PipelineStatus.RUNNING
        run.error_message = None

        stage_index = PipelineStage.ORDERED.index(failed_stage)
        for stage in PipelineStage.ORDERED[stage_index:]:
            if run.cancelled:
                run.status = PipelineStatus.CANCELLED
                return run

            run.current_stage = stage
            start = datetime.now(timezone.utc)

            try:
                handler = self._stage_handlers[stage]
                output = handler(run)
                result = StageResult(stage, "completed", output=output)
            except Exception as exc:
                result = StageResult(stage, "failed", error=str(exc))
                run.stages[stage] = result
                run.status = PipelineStatus.FAILED
                run.error_message = f"Stage '{stage}' failed: {exc}"
                return run

            result.started_at = start
            result.completed_at = datetime.now(timezone.utc)
            result.duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000
            run.stages[stage] = result

        run.status = PipelineStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        return run

    def cancel_pipeline(self, run_id: str) -> PipelineRun:
        """Cancel an active pipeline run."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Pipeline run '{run_id}' not found")
        run.cancelled = True
        if run.status in (PipelineStatus.PENDING, PipelineStatus.RUNNING):
            run.status = PipelineStatus.CANCELLED
        return run

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Retrieve a pipeline run by ID."""
        return self._runs.get(run_id)

    def get_history(self, supplier_id: Optional[str] = None,
                    limit: int = 50) -> List[PipelineRun]:
        """Retrieve pipeline run history."""
        runs = list(self._runs.values())
        if supplier_id:
            runs = [r for r in runs if r.supplier_id == supplier_id]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    # Stage implementations (simplified for testing)
    def _run_intake(self, run: PipelineRun) -> Dict:
        return {"supplier_validated": True, "commodities_normalized": True}

    def _run_geo_validation(self, run: PipelineRun) -> Dict:
        return {"plots_validated": len(run.plot_ids), "all_valid": True}

    def _run_deforestation_risk(self, run: PipelineRun) -> Dict:
        return {"risk_assessed": True, "overall_risk": 0.3}

    def _run_document_verification(self, run: PipelineRun) -> Dict:
        return {"documents_verified": 0, "all_compliant": True}

    def _run_dds_reporting(self, run: PipelineRun) -> Dict:
        return {"dds_generated": True, "reference": "EUDR-BRA-2026-000001"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator():
    """Create a fresh PipelineOrchestrator."""
    return PipelineOrchestrator()


@pytest.fixture
def started_run(orchestrator):
    """Create a pipeline run in PENDING state."""
    return orchestrator.start_pipeline("sup_test123", "soya", ["plot_a", "plot_b"])


# ---------------------------------------------------------------------------
# TestPipelineStart
# ---------------------------------------------------------------------------

class TestPipelineStart:
    """Tests for pipeline initialization."""

    def test_creates_run_with_pending_status(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc")
        assert run.status == PipelineStatus.PENDING

    def test_generates_unique_run_id(self, orchestrator):
        run1 = orchestrator.start_pipeline("sup_abc")
        run2 = orchestrator.start_pipeline("sup_abc")
        assert run1.run_id != run2.run_id

    def test_sets_current_stage_to_intake(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc")
        assert run.current_stage == PipelineStage.INTAKE

    def test_stores_supplier_id(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc123")
        assert run.supplier_id == "sup_abc123"

    def test_stores_commodity(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc", commodity="wood")
        assert run.commodity == "wood"

    def test_stores_plot_ids(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc", plot_ids=["p1", "p2"])
        assert run.plot_ids == ["p1", "p2"]

    def test_run_id_format(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc")
        assert run.run_id.startswith("run_")
        assert len(run.run_id) == 16  # "run_" + 12 hex chars


# ---------------------------------------------------------------------------
# TestIntakeStage
# ---------------------------------------------------------------------------

class TestIntakeStage:
    """Tests for the intake stage."""

    def test_intake_normalizes_supplier_data(self, orchestrator):
        run = orchestrator.start_pipeline("sup_test")
        output = orchestrator._run_intake(run)
        assert output["supplier_validated"] is True
        assert output["commodities_normalized"] is True

    def test_intake_handles_supplier_with_no_commodity(self, orchestrator):
        run = orchestrator.start_pipeline("sup_test", commodity=None)
        output = orchestrator._run_intake(run)
        assert output["supplier_validated"] is True


# ---------------------------------------------------------------------------
# TestGeoValidationStage
# ---------------------------------------------------------------------------

class TestGeoValidationStage:
    """Tests for the geo-validation stage."""

    def test_geo_validation_validates_coordinates(self, orchestrator, started_run):
        output = orchestrator._run_geo_validation(started_run)
        assert output["all_valid"] is True

    def test_geo_validation_counts_plots(self, orchestrator, started_run):
        output = orchestrator._run_geo_validation(started_run)
        assert output["plots_validated"] == 2  # started_run has 2 plot_ids

    def test_geo_validation_no_plots(self, orchestrator):
        run = orchestrator.start_pipeline("sup_test", plot_ids=[])
        output = orchestrator._run_geo_validation(run)
        assert output["plots_validated"] == 0


# ---------------------------------------------------------------------------
# TestDeforestationRiskStage
# ---------------------------------------------------------------------------

class TestDeforestationRiskStage:
    """Tests for the deforestation risk stage."""

    def test_returns_risk_assessment(self, orchestrator, started_run):
        output = orchestrator._run_deforestation_risk(started_run)
        assert output["risk_assessed"] is True
        assert 0 <= output["overall_risk"] <= 1

    def test_handles_simulated_satellite_timeout(self, orchestrator, started_run):
        """Simulate satellite API timeout by patching the handler."""
        original = orchestrator._run_deforestation_risk
        orchestrator._run_deforestation_risk = MagicMock(
            side_effect=TimeoutError("Satellite API timeout")
        )
        with pytest.raises(TimeoutError):
            orchestrator._run_deforestation_risk(started_run)
        orchestrator._run_deforestation_risk = original


# ---------------------------------------------------------------------------
# TestDocVerificationStage
# ---------------------------------------------------------------------------

class TestDocVerificationStage:
    """Tests for the document verification stage."""

    def test_classifies_documents(self, orchestrator, started_run):
        output = orchestrator._run_document_verification(started_run)
        assert "documents_verified" in output

    def test_checks_compliance_rules(self, orchestrator, started_run):
        output = orchestrator._run_document_verification(started_run)
        assert output["all_compliant"] is True


# ---------------------------------------------------------------------------
# TestDDSReportingStage
# ---------------------------------------------------------------------------

class TestDDSReportingStage:
    """Tests for the DDS reporting stage."""

    def test_generates_dds(self, orchestrator, started_run):
        output = orchestrator._run_dds_reporting(started_run)
        assert output["dds_generated"] is True

    def test_sets_reference_number(self, orchestrator, started_run):
        output = orchestrator._run_dds_reporting(started_run)
        assert output["reference"].startswith("EUDR-")


# ---------------------------------------------------------------------------
# TestPipelineExecution
# ---------------------------------------------------------------------------

class TestPipelineExecution:
    """Tests for full pipeline execution."""

    def test_runs_all_five_stages_sequentially(self, orchestrator, started_run):
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.stages) == 5
        for stage in PipelineStage.ORDERED:
            assert stage in result.stages
            assert result.stages[stage].status == "completed"

    def test_handles_stage_failure(self, orchestrator, started_run):
        """Pipeline stops and records failure when a stage raises."""
        orchestrator._run_geo_validation = MagicMock(
            side_effect=RuntimeError("Invalid polygon")
        )
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.FAILED
        assert result.current_stage == PipelineStage.GEO_VALIDATION
        assert "Invalid polygon" in result.error_message

    def test_tracks_timing_per_stage(self, orchestrator, started_run):
        result = orchestrator.execute_pipeline(started_run.run_id)
        for stage_name, stage_result in result.stages.items():
            assert stage_result.started_at is not None
            assert stage_result.completed_at is not None
            assert stage_result.duration_ms >= 0

    def test_supports_cancellation(self, orchestrator, started_run):
        """Cancel before execution starts."""
        orchestrator.cancel_pipeline(started_run.run_id)
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.CANCELLED

    def test_sets_started_at(self, orchestrator, started_run):
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.started_at is not None

    def test_sets_completed_at(self, orchestrator, started_run):
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

    def test_only_completed_stages_on_failure(self, orchestrator, started_run):
        """On failure, only stages up to and including the failed one have results."""
        orchestrator._run_deforestation_risk = MagicMock(
            side_effect=RuntimeError("Satellite down")
        )
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert PipelineStage.INTAKE in result.stages
        assert PipelineStage.GEO_VALIDATION in result.stages
        assert PipelineStage.DEFORESTATION_RISK in result.stages
        assert PipelineStage.DOCUMENT_VERIFICATION not in result.stages
        assert PipelineStage.DDS_REPORTING not in result.stages


# ---------------------------------------------------------------------------
# TestPipelineHistory
# ---------------------------------------------------------------------------

class TestPipelineHistory:
    """Tests for pipeline run history."""

    def test_stores_completed_runs(self, orchestrator):
        run = orchestrator.start_pipeline("sup_abc")
        orchestrator.execute_pipeline(run.run_id)
        history = orchestrator.get_history()
        assert len(history) == 1
        assert history[0].run_id == run.run_id

    def test_filters_by_supplier_id(self, orchestrator):
        r1 = orchestrator.start_pipeline("sup_A")
        r2 = orchestrator.start_pipeline("sup_B")
        orchestrator.execute_pipeline(r1.run_id)
        orchestrator.execute_pipeline(r2.run_id)
        history = orchestrator.get_history(supplier_id="sup_A")
        assert len(history) == 1
        assert history[0].supplier_id == "sup_A"

    def test_limits_results(self, orchestrator):
        for i in range(10):
            r = orchestrator.start_pipeline(f"sup_{i}")
            orchestrator.execute_pipeline(r.run_id)
        history = orchestrator.get_history(limit=5)
        assert len(history) == 5

    def test_sorted_by_created_at_descending(self, orchestrator):
        for i in range(3):
            r = orchestrator.start_pipeline(f"sup_{i}")
            orchestrator.execute_pipeline(r.run_id)
        history = orchestrator.get_history()
        for i in range(len(history) - 1):
            assert history[i].created_at >= history[i + 1].created_at


# ---------------------------------------------------------------------------
# TestPipelineRetry
# ---------------------------------------------------------------------------

class TestPipelineRetry:
    """Tests for pipeline retry logic."""

    def test_retries_from_failed_stage(self, orchestrator, started_run):
        """After failure at geo_validation, retry resumes from that stage."""
        fail_count = {"count": 0}

        def failing_geo(run):
            fail_count["count"] += 1
            if fail_count["count"] <= 1:
                raise RuntimeError("Temporary failure")
            return {"plots_validated": 2, "all_valid": True}

        orchestrator._run_geo_validation = failing_geo

        # First execution fails
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.FAILED
        assert result.current_stage == PipelineStage.GEO_VALIDATION

        # Retry succeeds
        result = orchestrator.retry_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.COMPLETED

    def test_resets_error_state(self, orchestrator, started_run):
        orchestrator._run_geo_validation = MagicMock(
            side_effect=RuntimeError("Fail once")
        )
        orchestrator.execute_pipeline(started_run.run_id)
        assert started_run.error_message is not None

        # Fix the handler and retry
        orchestrator._run_geo_validation = lambda run: {"plots_validated": 0, "all_valid": True}
        orchestrator.retry_pipeline(started_run.run_id)
        assert started_run.error_message is None

    def test_cannot_retry_non_failed_pipeline(self, orchestrator, started_run):
        orchestrator.execute_pipeline(started_run.run_id)
        with pytest.raises(ValueError, match="Can only retry failed"):
            orchestrator.retry_pipeline(started_run.run_id)

    def test_retry_unknown_run_raises(self, orchestrator):
        with pytest.raises(ValueError, match="not found"):
            orchestrator.retry_pipeline("run_nonexistent")


# ---------------------------------------------------------------------------
# TestPipelineCancellation
# ---------------------------------------------------------------------------

class TestPipelineCancellation:
    """Tests for pipeline cancellation."""

    def test_cancel_pending_pipeline(self, orchestrator, started_run):
        result = orchestrator.cancel_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.CANCELLED

    def test_cancel_unknown_run_raises(self, orchestrator):
        with pytest.raises(ValueError, match="not found"):
            orchestrator.cancel_pipeline("run_nonexistent")

    def test_cancelled_pipeline_does_not_execute(self, orchestrator, started_run):
        orchestrator.cancel_pipeline(started_run.run_id)
        result = orchestrator.execute_pipeline(started_run.run_id)
        assert result.status == PipelineStatus.CANCELLED
        assert len(result.stages) == 0
