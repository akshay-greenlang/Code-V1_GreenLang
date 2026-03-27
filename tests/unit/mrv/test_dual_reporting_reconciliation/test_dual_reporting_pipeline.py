# -*- coding: utf-8 -*-
"""
Unit tests for DualReportingPipelineEngine (Engine 7 of 7).

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 70+ tests covering 10-stage pipeline orchestration.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

import pytest

from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import (
    DualReportingPipelineEngine,
    PipelineExecutionError,
    StageExecutionError,
)
from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    PipelineStage,
    ReconciliationStatus,
    ReconciliationWorkspace,
    ReconciliationReport,
    ReconciliationRequest,
    BatchReconciliationRequest,
    BatchReconciliationResult,
    BatchStatus,
    ExportFormat,
    ReportingFramework,
    UpstreamResult,
    Scope2Method,
    EnergyType,
    DiscrepancyReport,
    DiscrepancyItem,
    DiscrepancyType,
    DiscrepancyDirection,
    MaterialityLevel,
    QualityAssessment,
    QualityScore,
    ReportingTableSet,
    TrendReport,
    TrendDataPoint,
    ComplianceCheckResult,
    ComplianceIssue,
    AggregationResult,
    ZERO,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a DualReportingPipelineEngine instance."""
    return DualReportingPipelineEngine()


@pytest.fixture
def sample_workspace() -> ReconciliationWorkspace:
    """Return a sample ReconciliationWorkspace."""
    loc = UpstreamResult(
        agent="mrv_009",
        method=Scope2Method.LOCATION_BASED,
        energy_type=EnergyType.ELECTRICITY,
        emissions_tco2e=Decimal("1250.50"),
        energy_quantity_mwh=Decimal("5000.0"),
        ef_used=Decimal("0.2501"),
        ef_source="eGRID 2023 CAMX",
        ef_hierarchy="grid_average",
        facility_id="FAC-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
    )
    mkt = UpstreamResult(
        agent="mrv_010",
        method=Scope2Method.MARKET_BASED,
        energy_type=EnergyType.ELECTRICITY,
        emissions_tco2e=Decimal("625.25"),
        energy_quantity_mwh=Decimal("5000.0"),
        ef_used=Decimal("0.12505"),
        ef_source="Supplier Disclosure 2024",
        ef_hierarchy="supplier_no_cert",
        facility_id="FAC-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
    )
    return ReconciliationWorkspace(
        reconciliation_id="RECON-TEST-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        location_results=[loc],
        market_results=[mkt],
        total_location_tco2e=Decimal("1250.50"),
        total_market_tco2e=Decimal("625.25"),
    )


@pytest.fixture
def sample_discrepancy_report() -> DiscrepancyReport:
    """Return a sample DiscrepancyReport."""
    item = DiscrepancyItem(
        energy_type=EnergyType.ELECTRICITY,
        discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
        direction=DiscrepancyDirection.MARKET_LOWER,
        materiality=MaterialityLevel.HIGH,
        absolute_tco2e=Decimal("625.25"),
        percentage=Decimal("100.00"),
        description="RECs purchased reducing market-based emissions by 50%",
        recommendation="Continue purchasing high-quality RECs to maintain reduction",
    )
    return DiscrepancyReport(
        total_delta_tco2e=Decimal("625.25"),
        total_delta_pct=Decimal("50.00"),
        items=[item],
    )


@pytest.fixture
def sample_quality_assessment() -> QualityAssessment:
    """Return a sample QualityAssessment."""
    return QualityAssessment(
        overall_score=Decimal("85.0"),
        location_based_score=Decimal("90.0"),
        market_based_score=Decimal("80.0"),
        quality_grade="B",
        dimensions={
            "completeness": QualityScore(
                dimension="completeness",
                score=Decimal("90.0"),
                weight=Decimal("0.30"),
            ),
            "consistency": QualityScore(
                dimension="consistency",
                score=Decimal("85.0"),
                weight=Decimal("0.25"),
            ),
            "accuracy": QualityScore(
                dimension="accuracy",
                score=Decimal("80.0"),
                weight=Decimal("0.30"),
            ),
            "transparency": QualityScore(
                dimension="transparency",
                score=Decimal("85.0"),
                weight=Decimal("0.15"),
            ),
        },
    )


@pytest.fixture
def sample_request(sample_upstream_results) -> ReconciliationRequest:
    """Return a sample ReconciliationRequest."""
    # Convert dict upstream results to UpstreamResult models
    location_results = []
    market_results = []

    for r in sample_upstream_results:
        result = UpstreamResult(
            agent=r["agent"],
            method=Scope2Method(r["method"]),
            energy_type=EnergyType(r["energy_type"]),
            emissions_tco2e=r["emissions_tco2e"],
            energy_quantity_mwh=r["energy_quantity_mwh"],
            ef_used=r["ef_used"],
            ef_source=r["ef_source"],
            ef_hierarchy=r["ef_hierarchy"],
            facility_id=r["facility_id"],
            tenant_id=r["tenant_id"],
            period_start=date.fromisoformat(r["period_start"]),
            period_end=date.fromisoformat(r["period_end"]),
        )
        if result.method == Scope2Method.LOCATION_BASED:
            location_results.append(result)
        else:
            market_results.append(result)

    return ReconciliationRequest(
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        location_results=location_results,
        market_results=market_results,
        frameworks=[ReportingFramework.GHG_PROTOCOL],
        include_trends=False,
        include_quality=True,
    )


# ---------------------------------------------------------------------------
# Test Class: Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """Test singleton pattern implementation."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return the same instance."""
        engine1 = DualReportingPipelineEngine()
        engine2 = DualReportingPipelineEngine()

        assert engine1 is engine2

    def test_singleton_initialized_once(self):
        """Test that initialization happens only once."""
        engine = DualReportingPipelineEngine()

        assert engine._initialized is True
        assert engine.config is not None
        assert engine.metrics is not None
        assert engine.provenance is not None

    def test_lazy_engine_loading(self, engine):
        """Test that engines are lazy-loaded."""
        # Engines should be None initially
        assert engine._collector is None
        assert engine._discrepancy_analyzer is None
        assert engine._quality_scorer is None
        assert engine._table_generator is None
        assert engine._trend_analyzer is None
        assert engine._compliance_checker is None

    def test_reset_clears_engines(self, engine):
        """Test that reset clears all engines."""
        # Load an engine
        collector = engine._get_collector()
        assert collector is not None
        assert engine._collector is not None

        # Reset
        engine.reset()

        # Verify engines cleared
        assert engine._collector is None
        assert engine._current_pipeline_id is None
        assert engine._current_stage is None

    def test_reset_state(self, engine):
        """Test that reset clears pipeline state."""
        engine._current_pipeline_id = "test-123"
        engine._current_stage = PipelineStage.COLLECT_RESULTS
        engine._pipeline_start_time = datetime.utcnow()

        engine.reset()

        assert engine._current_pipeline_id is None
        assert engine._current_stage is None
        assert engine._pipeline_start_time is None

    def test_repr(self, engine):
        """Test string representation."""
        repr_str = repr(engine)

        assert "DualReportingPipelineEngine" in repr_str
        assert "initialized=True" in repr_str


# ---------------------------------------------------------------------------
# Test Class: Run Pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Test run_pipeline method."""

    @patch.object(DualReportingPipelineEngine, '_execute_stage')
    def test_run_pipeline_success(self, mock_execute_stage, engine, sample_request):
        """Test successful pipeline execution."""
        # Mock all 10 stages
        mock_context = MagicMock()
        mock_context.completed_stages = []
        mock_context.stage_timings_ms = {}

        # Create final report
        final_report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )
        mock_context.final_report = final_report

        mock_execute_stage.return_value = mock_context

        result = engine.run_pipeline(sample_request)

        assert isinstance(result, ReconciliationReport)
        assert result.status == ReconciliationStatus.COMPLETED
        assert mock_execute_stage.call_count == 10

    def test_run_pipeline_invalid_no_upstream_results(self, engine):
        """Test pipeline fails with no upstream results."""
        request = ReconciliationRequest(
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            upstream_results=[],
            frameworks=[ReportingFramework.GHG_PROTOCOL],
        )

        with pytest.raises(ValueError, match="At least one upstream result required"):
            engine.run_pipeline(request)

    def test_run_pipeline_missing_location_method(self, engine, sample_upstream_results):
        """Test pipeline fails without location-based result."""
        # Remove location-based result
        market_only = [
            UpstreamResult(
                agent="mrv_010",
                method=Scope2Method.MARKET_BASED,
                energy_type=EnergyType.ELECTRICITY,
                emissions_tco2e=Decimal("625.25"),
                energy_quantity_mwh=Decimal("5000.0"),
                ef_used=Decimal("0.12505"),
                ef_source="Supplier",
                ef_hierarchy="supplier_no_cert",
                facility_id="FAC-001",
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
            )
        ]

        request = ReconciliationRequest(
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            upstream_results=market_only,
            frameworks=[ReportingFramework.GHG_PROTOCOL],
        )

        with pytest.raises(ValueError, match="Location-based result required"):
            engine.run_pipeline(request)

    def test_run_pipeline_missing_market_method(self, engine):
        """Test pipeline fails without market-based result."""
        location_only = [
            UpstreamResult(
                agent="mrv_009",
                method=Scope2Method.LOCATION_BASED,
                energy_type=EnergyType.ELECTRICITY,
                emissions_tco2e=Decimal("1250.50"),
                energy_quantity_mwh=Decimal("5000.0"),
                ef_used=Decimal("0.2501"),
                ef_source="eGRID",
                ef_hierarchy="grid_average",
                facility_id="FAC-001",
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
            )
        ]

        request = ReconciliationRequest(
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            upstream_results=location_only,
            frameworks=[ReportingFramework.GHG_PROTOCOL],
        )

        with pytest.raises(ValueError, match="Market-based result required"):
            engine.run_pipeline(request)

    def test_run_pipeline_invalid_period(self, engine, sample_upstream_results):
        """Test pipeline fails with invalid period."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        request = ReconciliationRequest(
            tenant_id="tenant-001",
            period_start=date(2024, 12, 31),
            period_end=date(2024, 1, 1),  # End before start
            upstream_results=upstream,
            frameworks=[ReportingFramework.GHG_PROTOCOL],
        )

        with pytest.raises(ValueError, match="Invalid period"):
            engine.run_pipeline(request)

    def test_run_pipeline_no_frameworks(self, engine, sample_upstream_results):
        """Test pipeline fails without frameworks."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        request = ReconciliationRequest(
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            upstream_results=upstream,
            frameworks=[],  # Empty
        )

        with pytest.raises(ValueError, match="At least one reporting framework required"):
            engine.run_pipeline(request)

    @patch.object(DualReportingPipelineEngine, '_execute_stage')
    def test_run_pipeline_stage_failure(self, mock_execute_stage, engine, sample_request):
        """Test pipeline handles stage failure."""
        mock_execute_stage.side_effect = StageExecutionError(
            stage=PipelineStage.COLLECT_RESULTS,
            message="Stage failed",
        )

        with pytest.raises(PipelineExecutionError):
            engine.run_pipeline(sample_request)

    @patch.object(DualReportingPipelineEngine, '_execute_stage')
    def test_run_pipeline_records_metrics(self, mock_execute_stage, engine, sample_request):
        """Test pipeline records metrics."""
        mock_context = MagicMock()
        mock_context.completed_stages = []
        mock_context.stage_timings_ms = {}
        mock_context.final_report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )
        mock_execute_stage.return_value = mock_context

        with patch.object(engine.metrics, 'record_pipeline_execution') as mock_record:
            result = engine.run_pipeline(sample_request)

            assert mock_record.called
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs['tenant_id'] == "tenant-001"
            assert call_kwargs['success'] is True

    def test_run_pipeline_sets_duration(self, engine, sample_request):
        """Test pipeline sets duration in report."""
        with patch.object(engine, '_execute_stage') as mock_execute:
            mock_context = MagicMock()
            mock_context.completed_stages = []
            mock_context.stage_timings_ms = {}
            mock_context.final_report = ReconciliationReport(
                reconciliation_id="test-123",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="test-hash",
                created_at=datetime.utcnow(),
            )
            mock_execute.return_value = mock_context

            result = engine.run_pipeline(sample_request)

            assert result.pipeline_duration_ms >= ZERO

    def test_run_pipeline_clears_state_on_completion(self, engine, sample_request):
        """Test pipeline clears state after completion."""
        with patch.object(engine, '_execute_stage') as mock_execute:
            mock_context = MagicMock()
            mock_context.completed_stages = []
            mock_context.stage_timings_ms = {}
            mock_context.final_report = ReconciliationReport(
                reconciliation_id="test-123",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="test-hash",
                created_at=datetime.utcnow(),
            )
            mock_execute.return_value = mock_context

            engine.run_pipeline(sample_request)

            assert engine._current_pipeline_id is None
            assert engine._current_stage is None
            assert engine._pipeline_start_time is None

    def test_run_pipeline_clears_state_on_failure(self, engine, sample_request):
        """Test pipeline clears state after failure."""
        with patch.object(engine, '_execute_stage') as mock_execute:
            mock_execute.side_effect = Exception("Test error")

            with pytest.raises(PipelineExecutionError):
                engine.run_pipeline(sample_request)

            assert engine._current_pipeline_id is None
            assert engine._current_stage is None
            assert engine._pipeline_start_time is None


# ---------------------------------------------------------------------------
# Test Class: Run Batch
# ---------------------------------------------------------------------------


class TestRunBatch:
    """Test run_batch method."""

    def test_run_batch_success(self, engine, sample_upstream_results):
        """Test successful batch processing."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        periods = [
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2023, 1, 1),
                period_end=date(2023, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
        ]

        batch_request = BatchReconciliationRequest(
            periods=periods,
            fail_fast=False,
            include_aggregation=True,
        )

        with patch.object(engine, 'run_pipeline') as mock_run:
            mock_report = ReconciliationReport(
                reconciliation_id="test-123",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(
                    total_location_tco2e=Decimal("1000"),
                    total_market_tco2e=Decimal("800"),
                    pif=Decimal("0.20"),
                ),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(overall_score=Decimal("85")),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="test-hash",
                created_at=datetime.utcnow(),
            )
            mock_run.return_value = mock_report

            result = engine.run_batch(batch_request)

            assert isinstance(result, BatchReconciliationResult)
            assert result.status == BatchStatus.COMPLETED
            assert result.total_periods == 2
            assert result.completed_periods == 2
            assert result.failed_periods == 0
            assert len(result.reports) == 2
            assert result.aggregation is not None

    def test_run_batch_partial_failure(self, engine, sample_upstream_results):
        """Test batch with partial failures."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        periods = [
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2023, 1, 1),
                period_end=date(2023, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
        ]

        batch_request = BatchReconciliationRequest(
            periods=periods,
            fail_fast=False,
            include_aggregation=False,
        )

        mock_report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        with patch.object(engine, 'run_pipeline') as mock_run:
            # First succeeds, second fails
            mock_run.side_effect = [mock_report, Exception("Test error")]

            result = engine.run_batch(batch_request)

            assert result.status == BatchStatus.PARTIAL
            assert result.completed_periods == 1
            assert result.failed_periods == 1

    def test_run_batch_fail_fast(self, engine, sample_upstream_results):
        """Test batch with fail_fast enabled."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        periods = [
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2023, 1, 1),
                period_end=date(2023, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
        ]

        batch_request = BatchReconciliationRequest(
            periods=periods,
            fail_fast=True,
            include_aggregation=False,
        )

        with patch.object(engine, 'run_pipeline') as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = engine.run_batch(batch_request)

            assert result.failed_periods == 1
            # Should stop after first failure
            assert mock_run.call_count == 1

    def test_run_batch_empty_periods(self, engine):
        """Test batch with empty periods list."""
        batch_request = BatchReconciliationRequest(
            periods=[],
            fail_fast=False,
            include_aggregation=False,
        )

        result = engine.run_batch(batch_request)

        assert result.status == BatchStatus.FAILED
        assert result.total_periods == 0
        assert result.completed_periods == 0
        assert result.failed_periods == 0

    def test_run_batch_too_many_periods(self, engine):
        """Test batch validation with too many periods."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.models import MAX_BATCH_PERIODS

        periods = [MagicMock() for _ in range(MAX_BATCH_PERIODS + 1)]

        batch_request = BatchReconciliationRequest(
            periods=periods,
            fail_fast=False,
            include_aggregation=False,
        )

        with pytest.raises(ValueError, match="Too many periods in batch"):
            engine.run_batch(batch_request)

    def test_run_batch_calculates_provenance_hash(self, engine, sample_upstream_results):
        """Test batch calculates provenance hash."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        periods = [
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
        ]

        batch_request = BatchReconciliationRequest(
            periods=periods,
            fail_fast=False,
            include_aggregation=False,
        )

        with patch.object(engine, 'run_pipeline') as mock_run:
            mock_run.return_value = MagicMock(reconciliation_id="test-123")

            result = engine.run_batch(batch_request)

            assert result.provenance_hash is not None
            assert len(result.provenance_hash) == 64  # SHA-256

    def test_run_batch_with_aggregation(self, engine, sample_upstream_results):
        """Test batch with aggregation enabled."""
        upstream = [
            UpstreamResult(
                agent=r["agent"],
                method=Scope2Method(r["method"]),
                energy_type=EnergyType(r["energy_type"]),
                emissions_tco2e=r["emissions_tco2e"],
                energy_quantity_mwh=r["energy_quantity_mwh"],
                ef_used=r["ef_used"],
                ef_source=r["ef_source"],
                ef_hierarchy=r["ef_hierarchy"],
                facility_id=r["facility_id"],
                tenant_id=r["tenant_id"],
                period_start=date.fromisoformat(r["period_start"]),
                period_end=date.fromisoformat(r["period_end"]),
            )
            for r in sample_upstream_results
        ]

        periods = [
            ReconciliationRequest(
                tenant_id="tenant-001",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                upstream_results=upstream,
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            ),
        ]

        batch_request = BatchReconciliationRequest(
            periods=periods,
            fail_fast=False,
            include_aggregation=True,
        )

        with patch.object(engine, 'run_pipeline') as mock_run:
            mock_report = ReconciliationReport(
                reconciliation_id="test-123",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(
                    total_location_tco2e=Decimal("1000"),
                    total_market_tco2e=Decimal("800"),
                    pif=Decimal("0.20"),
                ),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(overall_score=Decimal("85")),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="test-hash",
                created_at=datetime.utcnow(),
            )
            mock_run.return_value = mock_report

            result = engine.run_batch(batch_request)

            assert result.aggregation is not None
            assert isinstance(result.aggregation, AggregationResult)


# ---------------------------------------------------------------------------
# Test Class: Execute Stage
# ---------------------------------------------------------------------------


class TestExecuteStage:
    """Test _execute_stage method."""

    def test_execute_stage_collect_results(self, engine, sample_request):
        """Test stage 1: COLLECT_RESULTS."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(request=sample_request)

        with patch.object(engine, '_stage_collect_results') as mock_stage:
            mock_stage.return_value = MagicMock()

            result = engine._execute_stage(PipelineStage.COLLECT_RESULTS, context)

            assert mock_stage.called
            assert result.workspace is not None

    def test_execute_stage_align_boundaries(self, engine, sample_request, sample_workspace):
        """Test stage 2: ALIGN_BOUNDARIES."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(request=sample_request, workspace=sample_workspace)

        with patch.object(engine, '_stage_align_boundaries') as mock_stage:
            mock_stage.return_value = sample_workspace

            result = engine._execute_stage(PipelineStage.ALIGN_BOUNDARIES, context)

            assert mock_stage.called

    def test_execute_stage_map_energy_types(self, engine, sample_request, sample_workspace):
        """Test stage 3: MAP_ENERGY_TYPES."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(request=sample_request, workspace=sample_workspace)

        with patch.object(engine, '_stage_map_energy_types') as mock_stage:
            mock_stage.return_value = sample_workspace

            result = engine._execute_stage(PipelineStage.MAP_ENERGY_TYPES, context)

            assert mock_stage.called

    def test_execute_stage_analyze_discrepancies(self, engine, sample_request, sample_workspace):
        """Test stage 4: ANALYZE_DISCREPANCIES."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(request=sample_request, workspace=sample_workspace)

        with patch.object(engine, '_stage_analyze_discrepancies') as mock_stage:
            mock_stage.return_value = MagicMock()

            result = engine._execute_stage(PipelineStage.ANALYZE_DISCREPANCIES, context)

            assert mock_stage.called
            assert result.discrepancy_report is not None

    def test_execute_stage_score_quality(self, engine, sample_request, sample_workspace):
        """Test stage 5: SCORE_QUALITY."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(request=sample_request, workspace=sample_workspace)

        with patch.object(engine, '_stage_score_quality') as mock_stage:
            mock_stage.return_value = MagicMock()

            result = engine._execute_stage(PipelineStage.SCORE_QUALITY, context)

            assert mock_stage.called
            assert result.quality_assessment is not None

    def test_execute_stage_generate_tables(self, engine, sample_request, sample_workspace):
        """Test stage 6: GENERATE_TABLES."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(
            request=sample_request,
            workspace=sample_workspace,
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
        )

        with patch.object(engine, '_stage_generate_tables') as mock_stage:
            mock_stage.return_value = MagicMock()

            result = engine._execute_stage(PipelineStage.GENERATE_TABLES, context)

            assert mock_stage.called
            assert result.reporting_tables is not None

    def test_execute_stage_analyze_trends(self, engine, sample_request, sample_workspace):
        """Test stage 7: ANALYZE_TRENDS."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(request=sample_request, workspace=sample_workspace)

        with patch.object(engine, '_stage_analyze_trends') as mock_stage:
            mock_stage.return_value = MagicMock()

            result = engine._execute_stage(PipelineStage.ANALYZE_TRENDS, context)

            assert mock_stage.called

    def test_execute_stage_check_compliance(self, engine, sample_request, sample_workspace):
        """Test stage 8: CHECK_COMPLIANCE."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(
            request=sample_request,
            workspace=sample_workspace,
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
        )

        with patch.object(engine, '_stage_check_compliance') as mock_stage:
            mock_stage.return_value = {}

            result = engine._execute_stage(PipelineStage.CHECK_COMPLIANCE, context)

            assert mock_stage.called
            assert result.compliance_results is not None

    def test_execute_stage_assemble_report(self, engine, sample_request, sample_workspace):
        """Test stage 9: ASSEMBLE_REPORT."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        context = PipelineContext(
            request=sample_request,
            workspace=sample_workspace,
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
        )

        with patch.object(engine, '_stage_assemble_report') as mock_stage:
            mock_stage.return_value = MagicMock()

            result = engine._execute_stage(PipelineStage.ASSEMBLE_REPORT, context)

            assert mock_stage.called
            assert hasattr(result, 'final_report')

    def test_execute_stage_seal_provenance(self, engine, sample_request):
        """Test stage 10: SEAL_PROVENANCE."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline import PipelineContext

        final_report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="",
            created_at=datetime.utcnow(),
        )

        context = PipelineContext(request=sample_request)
        context.final_report = final_report

        with patch.object(engine, '_stage_seal_provenance') as mock_stage:
            mock_stage.return_value = final_report

            result = engine._execute_stage(PipelineStage.SEAL_PROVENANCE, context)

            assert mock_stage.called


# ---------------------------------------------------------------------------
# Test Class: Stage - Collect Results
# ---------------------------------------------------------------------------


class TestStageCollectResults:
    """Test _stage_collect_results method."""

    def test_stage_collect_results(self, engine, sample_request):
        """Test collecting upstream results."""
        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_workspace = MagicMock()
            mock_workspace.location_results = []
            mock_workspace.market_results = []
            mock_collector.collect_results.return_value = mock_workspace
            mock_get_collector.return_value = mock_collector

            result = engine._stage_collect_results(sample_request)

            assert mock_collector.collect_results.called
            assert result == mock_workspace

    def test_stage_collect_results_logs_count(self, engine, sample_request):
        """Test stage logs result counts."""
        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_workspace = MagicMock()
            mock_workspace.location_results = [MagicMock(), MagicMock()]
            mock_workspace.market_results = [MagicMock()]
            mock_collector.collect_results.return_value = mock_workspace
            mock_get_collector.return_value = mock_collector

            result = engine._stage_collect_results(sample_request)

            assert len(result.location_results) == 2
            assert len(result.market_results) == 1

    def test_stage_collect_results_empty(self, engine, sample_request):
        """Test collecting with no results."""
        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_workspace = MagicMock()
            mock_workspace.location_results = []
            mock_workspace.market_results = []
            mock_collector.collect_results.return_value = mock_workspace
            mock_get_collector.return_value = mock_collector

            result = engine._stage_collect_results(sample_request)

            assert len(result.location_results) == 0
            assert len(result.market_results) == 0

    def test_stage_collect_results_lazy_loads_collector(self, engine, sample_request):
        """Test stage lazy-loads collector engine."""
        assert engine._collector is None

        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.collect_results.return_value = MagicMock(
                location_results=[],
                market_results=[],
            )
            mock_get_collector.return_value = mock_collector

            engine._stage_collect_results(sample_request)

            assert mock_get_collector.called


# ---------------------------------------------------------------------------
# Test Class: Stage - Align Boundaries
# ---------------------------------------------------------------------------


class TestStageAlignBoundaries:
    """Test _stage_align_boundaries method."""

    def test_stage_align_boundaries_success(self, engine, sample_workspace):
        """Test boundary alignment with no issues."""
        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.verify_boundary_alignment.return_value = []
            mock_get_collector.return_value = mock_collector

            result = engine._stage_align_boundaries(sample_workspace)

            assert mock_collector.verify_boundary_alignment.called
            assert result == sample_workspace

    def test_stage_align_boundaries_with_issues(self, engine, sample_workspace):
        """Test boundary alignment with issues."""
        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            issues = [
                {
                    "location_boundary": "FAC-001",
                    "market_boundary": "FAC-002",
                    "mismatch_type": "facility_mismatch",
                }
            ]
            mock_collector.verify_boundary_alignment.return_value = issues
            mock_get_collector.return_value = mock_collector

            result = engine._stage_align_boundaries(sample_workspace)

            assert "boundary_alignment_issues" in result.metadata

    def test_stage_align_boundaries_stores_issues_in_metadata(self, engine, sample_workspace):
        """Test boundary issues stored in workspace metadata."""
        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            issues = [
                {
                    "location_boundary": "FAC-001",
                    "market_boundary": "FAC-001",
                    "mismatch_type": "period_mismatch",
                }
            ]
            mock_collector.verify_boundary_alignment.return_value = issues
            mock_get_collector.return_value = mock_collector

            result = engine._stage_align_boundaries(sample_workspace)

            assert len(result.metadata["boundary_alignment_issues"]) == 1

    def test_stage_align_boundaries_empty_workspace(self, engine):
        """Test boundary alignment with empty workspace."""
        empty_workspace = ReconciliationWorkspace(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            location_results=[],
            market_results=[],
            total_location_tco2e=ZERO,
            total_market_tco2e=ZERO,
        )

        with patch.object(engine, '_get_collector') as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.verify_boundary_alignment.return_value = []
            mock_get_collector.return_value = mock_collector

            result = engine._stage_align_boundaries(empty_workspace)

            assert result == empty_workspace


# ---------------------------------------------------------------------------
# Test Class: Stage - Analyze Discrepancies
# ---------------------------------------------------------------------------


class TestStageAnalyzeDiscrepancies:
    """Test _stage_analyze_discrepancies method."""

    def test_stage_analyze_discrepancies(self, engine, sample_workspace):
        """Test discrepancy analysis."""
        with patch.object(engine, '_get_discrepancy_analyzer') as mock_get_analyzer:
            mock_analyzer = MagicMock()
            mock_report = MagicMock()
            mock_report.items = []
            mock_report.total_delta_tco2e = ZERO
            mock_analyzer.analyze_discrepancies.return_value = mock_report
            mock_get_analyzer.return_value = mock_analyzer

            result = engine._stage_analyze_discrepancies(sample_workspace)

            assert mock_analyzer.analyze_discrepancies.called
            assert result == mock_report

    def test_stage_analyze_discrepancies_with_items(self, engine, sample_workspace):
        """Test discrepancy analysis with items."""
        with patch.object(engine, '_get_discrepancy_analyzer') as mock_get_analyzer:
            mock_analyzer = MagicMock()
            mock_report = MagicMock()
            mock_report.items = [MagicMock(), MagicMock()]
            mock_report.total_delta_tco2e = Decimal("625.25")
            mock_analyzer.analyze_discrepancies.return_value = mock_report
            mock_get_analyzer.return_value = mock_analyzer

            result = engine._stage_analyze_discrepancies(sample_workspace)

            assert len(result.items) == 2
            assert result.total_delta_tco2e == Decimal("625.25")

    def test_stage_analyze_discrepancies_logs_summary(self, engine, sample_workspace):
        """Test stage logs discrepancy summary."""
        with patch.object(engine, '_get_discrepancy_analyzer') as mock_get_analyzer:
            mock_analyzer = MagicMock()
            mock_report = MagicMock()
            mock_report.items = [MagicMock()]
            mock_report.total_delta_tco2e = Decimal("100")
            mock_analyzer.analyze_discrepancies.return_value = mock_report
            mock_get_analyzer.return_value = mock_analyzer

            result = engine._stage_analyze_discrepancies(sample_workspace)

            assert len(result.items) == 1

    def test_stage_analyze_discrepancies_lazy_loads_analyzer(self, engine, sample_workspace):
        """Test stage lazy-loads discrepancy analyzer."""
        assert engine._discrepancy_analyzer is None

        with patch.object(engine, '_get_discrepancy_analyzer') as mock_get_analyzer:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_discrepancies.return_value = MagicMock(
                items=[],
                total_delta_tco2e=ZERO,
            )
            mock_get_analyzer.return_value = mock_analyzer

            engine._stage_analyze_discrepancies(sample_workspace)

            assert mock_get_analyzer.called


# ---------------------------------------------------------------------------
# Test Class: Stage - Score Quality
# ---------------------------------------------------------------------------


class TestStageScoreQuality:
    """Test _stage_score_quality method."""

    def test_stage_score_quality(self, engine, sample_workspace):
        """Test quality scoring."""
        with patch.object(engine, '_get_quality_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_assessment = MagicMock()
            mock_assessment.overall_score = Decimal("85")
            mock_assessment.location_based_score = Decimal("90")
            mock_assessment.market_based_score = Decimal("80")
            mock_scorer.score_quality.return_value = mock_assessment
            mock_get_scorer.return_value = mock_scorer

            result = engine._stage_score_quality(sample_workspace)

            assert mock_scorer.score_quality.called
            assert result == mock_assessment

    def test_stage_score_quality_logs_scores(self, engine, sample_workspace):
        """Test stage logs quality scores."""
        with patch.object(engine, '_get_quality_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_assessment = MagicMock()
            mock_assessment.overall_score = Decimal("75")
            mock_assessment.location_based_score = Decimal("80")
            mock_assessment.market_based_score = Decimal("70")
            mock_scorer.score_quality.return_value = mock_assessment
            mock_get_scorer.return_value = mock_scorer

            result = engine._stage_score_quality(sample_workspace)

            assert result.overall_score == Decimal("75")

    def test_stage_score_quality_lazy_loads_scorer(self, engine, sample_workspace):
        """Test stage lazy-loads quality scorer."""
        assert engine._quality_scorer is None

        with patch.object(engine, '_get_quality_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.score_quality.return_value = MagicMock(
                overall_score=Decimal("85"),
                location_based_score=Decimal("90"),
                market_based_score=Decimal("80"),
            )
            mock_get_scorer.return_value = mock_scorer

            engine._stage_score_quality(sample_workspace)

            assert mock_get_scorer.called

    def test_stage_score_quality_with_low_score(self, engine, sample_workspace):
        """Test quality scoring with low score."""
        with patch.object(engine, '_get_quality_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_assessment = MagicMock()
            mock_assessment.overall_score = Decimal("45")  # Low score
            mock_assessment.location_based_score = Decimal("50")
            mock_assessment.market_based_score = Decimal("40")
            mock_scorer.score_quality.return_value = mock_assessment
            mock_get_scorer.return_value = mock_scorer

            result = engine._stage_score_quality(sample_workspace)

            assert result.overall_score < Decimal("50")


# ---------------------------------------------------------------------------
# Test Class: Stage - Check Compliance
# ---------------------------------------------------------------------------


class TestStageCheckCompliance:
    """Test _stage_check_compliance method."""

    def test_stage_check_compliance_single_framework(self, engine, sample_workspace):
        """Test compliance checking for single framework."""
        with patch.object(engine, '_get_compliance_checker') as mock_get_checker:
            mock_checker = MagicMock()
            mock_result = MagicMock()
            mock_result.issues = []
            mock_checker.check_compliance.return_value = mock_result
            mock_get_checker.return_value = mock_checker

            result = engine._stage_check_compliance(
                workspace=sample_workspace,
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(),
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            )

            assert mock_checker.check_compliance.called
            assert ReportingFramework.GHG_PROTOCOL.value in result

    def test_stage_check_compliance_multiple_frameworks(self, engine, sample_workspace):
        """Test compliance checking for multiple frameworks."""
        with patch.object(engine, '_get_compliance_checker') as mock_get_checker:
            mock_checker = MagicMock()
            mock_result = MagicMock()
            mock_result.issues = []
            mock_checker.check_compliance.return_value = mock_result
            mock_get_checker.return_value = mock_checker

            frameworks = [
                ReportingFramework.GHG_PROTOCOL,
                ReportingFramework.CSRD_ESRS,
                ReportingFramework.CDP,
            ]

            result = engine._stage_check_compliance(
                workspace=sample_workspace,
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(),
                frameworks=frameworks,
            )

            assert len(result) == 3
            assert mock_checker.check_compliance.call_count == 3

    def test_stage_check_compliance_with_issues(self, engine, sample_workspace):
        """Test compliance checking with issues."""
        with patch.object(engine, '_get_compliance_checker') as mock_get_checker:
            mock_checker = MagicMock()
            mock_result = MagicMock()
            mock_result.issues = [MagicMock(), MagicMock()]
            mock_checker.check_compliance.return_value = mock_result
            mock_get_checker.return_value = mock_checker

            result = engine._stage_check_compliance(
                workspace=sample_workspace,
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(),
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            )

            assert len(result[ReportingFramework.GHG_PROTOCOL.value].issues) == 2

    def test_stage_check_compliance_lazy_loads_checker(self, engine, sample_workspace):
        """Test stage lazy-loads compliance checker."""
        assert engine._compliance_checker is None

        with patch.object(engine, '_get_compliance_checker') as mock_get_checker:
            mock_checker = MagicMock()
            mock_checker.check_compliance.return_value = MagicMock(issues=[])
            mock_get_checker.return_value = mock_checker

            engine._stage_check_compliance(
                workspace=sample_workspace,
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(),
                frameworks=[ReportingFramework.GHG_PROTOCOL],
            )

            assert mock_get_checker.called


# ---------------------------------------------------------------------------
# Test Class: Aggregate Results
# ---------------------------------------------------------------------------


class TestAggregateResults:
    """Test aggregate_results method."""

    def test_aggregate_results_single_report(self, engine):
        """Test aggregating single report."""
        report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(
                total_location_tco2e=Decimal("1000"),
                total_market_tco2e=Decimal("800"),
                pif=Decimal("0.20"),
            ),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(overall_score=Decimal("85")),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        result = engine.aggregate_results([report])

        assert isinstance(result, AggregationResult)
        assert result.total_location_tco2e == Decimal("1000")
        assert result.total_market_tco2e == Decimal("800")
        assert result.period_count == 1

    def test_aggregate_results_multiple_reports(self, engine):
        """Test aggregating multiple reports."""
        reports = [
            ReconciliationReport(
                reconciliation_id=f"test-{i}",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(
                    total_location_tco2e=Decimal("1000"),
                    total_market_tco2e=Decimal("800"),
                    pif=Decimal("0.20"),
                ),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(overall_score=Decimal("85")),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash=f"hash-{i}",
                created_at=datetime.utcnow(),
            )
            for i in range(3)
        ]

        result = engine.aggregate_results(reports)

        assert result.total_location_tco2e == Decimal("3000")
        assert result.total_market_tco2e == Decimal("2400")
        assert result.period_count == 3

    def test_aggregate_results_calculates_averages(self, engine):
        """Test aggregate calculates averages correctly."""
        reports = [
            ReconciliationReport(
                reconciliation_id="test-1",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(
                    total_location_tco2e=Decimal("1000"),
                    total_market_tco2e=Decimal("800"),
                    pif=Decimal("0.20"),
                ),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(overall_score=Decimal("80")),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="hash-1",
                created_at=datetime.utcnow(),
            ),
            ReconciliationReport(
                reconciliation_id="test-2",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(
                    total_location_tco2e=Decimal("2000"),
                    total_market_tco2e=Decimal("1600"),
                    pif=Decimal("0.40"),
                ),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(overall_score=Decimal("90")),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="hash-2",
                created_at=datetime.utcnow(),
            ),
        ]

        result = engine.aggregate_results(reports)

        assert result.average_pif == Decimal("0.30")  # (0.20 + 0.40) / 2
        assert result.average_quality_score == Decimal("85")  # (80 + 90) / 2

    def test_aggregate_results_empty_list_raises_error(self, engine):
        """Test aggregating empty list raises error."""
        with pytest.raises(ValueError, match="Cannot aggregate empty report list"):
            engine.aggregate_results([])

    def test_aggregate_results_handles_none_pif(self, engine):
        """Test aggregate handles reports with None PIF."""
        reports = [
            ReconciliationReport(
                reconciliation_id="test-1",
                tenant_id="tenant-001",
                status=ReconciliationStatus.COMPLETED,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                workspace=MagicMock(
                    total_location_tco2e=Decimal("1000"),
                    total_market_tco2e=Decimal("800"),
                    pif=None,
                ),
                discrepancy_report=MagicMock(),
                quality_assessment=MagicMock(overall_score=Decimal("85")),
                reporting_tables=MagicMock(),
                compliance_results={},
                pipeline_stages_completed=[],
                pipeline_duration_ms=ZERO,
                provenance_hash="hash-1",
                created_at=datetime.utcnow(),
            ),
        ]

        result = engine.aggregate_results(reports)

        assert result.average_pif == ZERO

    def test_aggregate_results_calculates_provenance_hash(self, engine):
        """Test aggregate calculates provenance hash."""
        report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(
                total_location_tco2e=Decimal("1000"),
                total_market_tco2e=Decimal("800"),
                pif=Decimal("0.20"),
            ),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(overall_score=Decimal("85")),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        result = engine.aggregate_results([report])

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256


# ---------------------------------------------------------------------------
# Test Class: Export Report
# ---------------------------------------------------------------------------


class TestExportReport:
    """Test export_report method."""

    def test_export_report_json(self, engine):
        """Test exporting report as JSON."""
        report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        result = engine.export_report(report, ExportFormat.JSON)

        assert isinstance(result, str)
        assert "test-123" in result

    def test_export_report_csv(self, engine, sample_workspace, sample_discrepancy_report, sample_quality_assessment):
        """Test exporting report as CSV."""
        report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=sample_workspace,
            discrepancy_report=sample_discrepancy_report,
            quality_assessment=sample_quality_assessment,
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        result = engine.export_report(report, ExportFormat.CSV)

        assert isinstance(result, str)
        assert "RECONCILIATION SUMMARY" in result
        assert "test-123" in result

    def test_export_report_unsupported_format(self, engine):
        """Test exporting with unsupported format raises error."""
        report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=[],
            pipeline_duration_ms=ZERO,
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        # Mock an invalid format
        invalid_format = MagicMock()
        invalid_format.value = "INVALID"

        with pytest.raises(ValueError, match="Unsupported export format"):
            engine.export_report(report, invalid_format)

    def test_export_json_includes_all_fields(self, engine):
        """Test JSON export includes all report fields."""
        report = ReconciliationReport(
            reconciliation_id="test-123",
            tenant_id="tenant-001",
            status=ReconciliationStatus.COMPLETED,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            workspace=MagicMock(),
            discrepancy_report=MagicMock(),
            quality_assessment=MagicMock(),
            reporting_tables=MagicMock(),
            compliance_results={},
            pipeline_stages_completed=["COLLECT_RESULTS"],
            pipeline_duration_ms=Decimal("123.45"),
            provenance_hash="test-hash",
            created_at=datetime.utcnow(),
        )

        result = engine.export_report(report, ExportFormat.JSON)

        assert "reconciliation_id" in result
        assert "tenant_id" in result
        assert "status" in result


# ---------------------------------------------------------------------------
# Test Class: Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_status(self, engine):
        """Test health check returns status."""
        result = engine.health_check()

        assert result["status"] in ["healthy", "degraded"]
        assert result["initialized"] is True

    def test_health_check_includes_agent_info(self, engine):
        """Test health check includes agent information."""
        from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
            AGENT_ID,
            AGENT_COMPONENT,
            VERSION,
        )

        result = engine.health_check()

        assert result["agent_id"] == AGENT_ID
        assert result["component"] == AGENT_COMPONENT
        assert result["version"] == VERSION

    def test_health_check_includes_engine_status(self, engine):
        """Test health check includes engine status."""
        result = engine.health_check()

        assert "engines" in result
        assert isinstance(result["engines"], dict)

    def test_health_check_shows_not_loaded_engines(self, engine):
        """Test health check shows not_loaded for uninitialized engines."""
        result = engine.health_check()

        # All engines should be not_loaded initially
        assert result["engines"]["collector"] == "not_loaded"
        assert result["engines"]["discrepancy_analyzer"] == "not_loaded"
        assert result["engines"]["quality_scorer"] == "not_loaded"
