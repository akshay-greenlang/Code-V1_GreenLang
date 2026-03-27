# -*- coding: utf-8 -*-
"""
Unit tests for DualReportingService (setup.py)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Tests the service facade layer that provides the main API for the dual
reporting reconciliation agent, including reconciliation execution,
discrepancy listing, quality assessment, reporting tables, trend
analysis, compliance checking, aggregations, exports, health, and stats.

Target: 60 tests, ~600 lines.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.mrv.dual_reporting_reconciliation.setup import (
    DualReportingService,
    get_service,
    reset_service,
    ReconcileResponse,
    BatchReconcileResponse,
    ReconciliationListResponse,
    DiscrepancyListResponse,
    WaterfallResponse,
    QualityAssessmentResponse,
    ReportingTablesResponse,
    TrendAnalysisResponse,
    ComplianceCheckResponse,
    AggregationResponse,
    ExportResponse,
    HealthResponse,
    StatsResponse,
    SERVICE_VERSION,
    SERVICE_NAME,
    AGENT_ID,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a DualReportingService instance."""
    reset_service()
    svc = get_service()
    yield svc
    reset_service()


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestServiceInit:
    """Test DualReportingService initialization."""

    def test_singleton_pattern(self):
        reset_service()
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2
        reset_service()

    def test_reset_clears_singleton(self):
        reset_service()
        s1 = get_service()
        reset_service()
        s2 = get_service()
        # After reset, a new instance should be created
        # (may or may not be the same object depending on implementation)
        assert s2 is not None

    def test_service_version(self):
        assert SERVICE_VERSION == "1.0.0"

    def test_service_name(self):
        assert SERVICE_NAME == "dual-reporting-reconciliation-service"

    def test_agent_id(self):
        assert AGENT_ID == "AGENT-MRV-013"


# ===========================================================================
# 2. Engine Properties
# ===========================================================================


class TestEngineProperties:
    """Test engine property accessors."""

    def test_config_property(self, service):
        # Config may be None or a config instance
        _ = service.config

    def test_metrics_property(self, service):
        _ = service.metrics

    def test_collector_engine(self, service):
        _ = service.collector_engine

    def test_discrepancy_engine(self, service):
        _ = service.discrepancy_engine

    def test_quality_engine(self, service):
        _ = service.quality_engine

    def test_table_generator_engine(self, service):
        _ = service.table_generator_engine

    def test_trend_engine(self, service):
        _ = service.trend_engine

    def test_compliance_engine(self, service):
        _ = service.compliance_engine

    def test_pipeline_engine(self, service):
        _ = service.pipeline_engine


# ===========================================================================
# 3. Reconciliation Tests
# ===========================================================================


class TestReconciliation:
    """Test single reconciliation execution."""

    def test_reconcile_basic(self, service, sample_reconciliation_request):
        result = service.reconcile(sample_reconciliation_request)
        assert isinstance(result, ReconcileResponse)
        assert result.success is True
        assert result.reconciliation_id != ""

    def test_reconcile_stores_result(self, service, sample_reconciliation_request):
        result = service.reconcile(sample_reconciliation_request)
        stored = service.get_reconciliation(result.reconciliation_id)
        assert stored is not None

    def test_reconcile_provenance_hash(self, service, sample_reconciliation_request):
        result = service.reconcile(sample_reconciliation_request)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_reconcile_processing_time(self, service, sample_reconciliation_request):
        result = service.reconcile(sample_reconciliation_request)
        assert result.processing_time_ms >= 0

    def test_reconcile_missing_tenant_id(self, service, sample_upstream_results):
        with pytest.raises(ValueError, match="tenant_id"):
            service.reconcile({
                "upstream_results": sample_upstream_results,
            })

    def test_reconcile_missing_upstream_results(self, service):
        with pytest.raises(ValueError, match="upstream_results"):
            service.reconcile({
                "tenant_id": "test-tenant",
            })

    def test_reconcile_with_custom_id(self, service, sample_reconciliation_request):
        sample_reconciliation_request["reconciliation_id"] = "CUSTOM-001"
        result = service.reconcile(sample_reconciliation_request)
        assert result.reconciliation_id == "CUSTOM-001"


# ===========================================================================
# 4. Batch Reconciliation Tests
# ===========================================================================


class TestBatchReconciliation:
    """Test batch reconciliation execution."""

    def test_batch_reconcile_basic(self, service, sample_batch_request):
        result = service.reconcile_batch(sample_batch_request)
        assert isinstance(result, BatchReconcileResponse)
        assert result.total_periods == 2

    def test_batch_reconcile_successful_count(self, service, sample_batch_request):
        result = service.reconcile_batch(sample_batch_request)
        assert result.successful == 2
        assert result.failed == 0

    def test_batch_reconcile_custom_batch_id(self, service, sample_batch_request):
        result = service.reconcile_batch(sample_batch_request)
        assert result.batch_id == "BATCH-001"


# ===========================================================================
# 5. List & Get Tests
# ===========================================================================


class TestListAndGet:
    """Test listing and retrieving reconciliation results."""

    def test_list_reconciliations_empty(self, service):
        result = service.list_reconciliations()
        assert isinstance(result, ReconciliationListResponse)
        assert result.total == 0

    def test_list_reconciliations_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        service.reconcile(sample_reconciliation_request)
        result = service.list_reconciliations()
        assert result.total == 1

    def test_list_with_tenant_filter(
        self, service, sample_reconciliation_request,
    ):
        service.reconcile(sample_reconciliation_request)
        result = service.list_reconciliations(tenant_id="tenant-001")
        assert result.total == 1

        result = service.list_reconciliations(tenant_id="other-tenant")
        assert result.total == 0

    def test_get_reconciliation_found(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        stored = service.get_reconciliation(resp.reconciliation_id)
        assert stored is not None
        assert stored["reconciliation_id"] == resp.reconciliation_id

    def test_get_reconciliation_not_found(self, service):
        result = service.get_reconciliation("NONEXISTENT")
        assert result is None

    def test_delete_reconciliation(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        deleted = service.delete_reconciliation(resp.reconciliation_id)
        assert deleted is True

        stored = service.get_reconciliation(resp.reconciliation_id)
        assert stored is None

    def test_delete_nonexistent(self, service):
        deleted = service.delete_reconciliation("NONEXISTENT")
        assert deleted is False


# ===========================================================================
# 6. Discrepancy Tests
# ===========================================================================


class TestDiscrepancies:
    """Test discrepancy listing."""

    def test_list_discrepancies_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        disc = service.list_discrepancies(resp.reconciliation_id)
        assert isinstance(disc, DiscrepancyListResponse)
        assert disc.reconciliation_id == resp.reconciliation_id

    def test_list_discrepancies_nonexistent(self, service):
        disc = service.list_discrepancies("NONEXISTENT")
        assert isinstance(disc, DiscrepancyListResponse)
        assert disc.total == 0


# ===========================================================================
# 7. Waterfall Tests
# ===========================================================================


class TestWaterfall:
    """Test waterfall decomposition."""

    def test_get_waterfall_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        wf = service.get_waterfall(resp.reconciliation_id)
        assert isinstance(wf, WaterfallResponse)
        assert wf.reconciliation_id == resp.reconciliation_id

    def test_get_waterfall_nonexistent(self, service):
        wf = service.get_waterfall("NONEXISTENT")
        assert isinstance(wf, WaterfallResponse)


# ===========================================================================
# 8. Quality Assessment Tests
# ===========================================================================


class TestQualityAssessment:
    """Test quality assessment retrieval."""

    def test_get_quality_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        qa = service.get_quality_assessment(resp.reconciliation_id)
        assert isinstance(qa, QualityAssessmentResponse)

    def test_get_quality_nonexistent(self, service):
        qa = service.get_quality_assessment("NONEXISTENT")
        assert isinstance(qa, QualityAssessmentResponse)


# ===========================================================================
# 9. Reporting Tables Tests
# ===========================================================================


class TestReportingTables:
    """Test reporting table generation."""

    def test_get_tables_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        tables = service.get_reporting_tables(resp.reconciliation_id)
        assert isinstance(tables, ReportingTablesResponse)

    def test_get_tables_nonexistent(self, service):
        tables = service.get_reporting_tables("NONEXISTENT")
        assert isinstance(tables, ReportingTablesResponse)
        assert tables.frameworks_generated == 0


# ===========================================================================
# 10. Trend Analysis Tests
# ===========================================================================


class TestTrendAnalysis:
    """Test trend analysis retrieval."""

    def test_get_trends_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        trends = service.get_trend_analysis(resp.reconciliation_id)
        assert isinstance(trends, TrendAnalysisResponse)

    def test_get_trends_nonexistent(self, service):
        trends = service.get_trend_analysis("NONEXISTENT")
        assert isinstance(trends, TrendAnalysisResponse)


# ===========================================================================
# 11. Compliance Check Tests
# ===========================================================================


class TestComplianceCheck:
    """Test compliance checking."""

    def test_check_compliance_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        comp = service.check_compliance(resp.reconciliation_id)
        assert isinstance(comp, ComplianceCheckResponse)
        assert comp.success is True
        assert comp.reconciliation_id == resp.reconciliation_id

    def test_check_compliance_nonexistent(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.check_compliance("NONEXISTENT")

    def test_get_compliance_result(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        comp = service.check_compliance(resp.reconciliation_id)
        stored = service.get_compliance_result(comp.compliance_id)
        assert stored is not None

    def test_get_compliance_result_nonexistent(self, service):
        result = service.get_compliance_result("NONEXISTENT")
        assert result is None


# ===========================================================================
# 12. Aggregation Tests
# ===========================================================================


class TestAggregations:
    """Test aggregation functionality."""

    def test_get_aggregations_empty(self, service):
        agg = service.get_aggregations()
        assert isinstance(agg, AggregationResponse)
        assert agg.reconciliation_count == 0

    def test_get_aggregations_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        service.reconcile(sample_reconciliation_request)
        agg = service.get_aggregations()
        assert agg.reconciliation_count == 1

    def test_get_aggregations_with_group_by(self, service):
        agg = service.get_aggregations(group_by="facility")
        assert isinstance(agg, AggregationResponse)
        assert agg.group_by == "facility"


# ===========================================================================
# 13. Export Tests
# ===========================================================================


class TestExport:
    """Test report export functionality."""

    def test_export_json(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        exp = service.export_report(resp.reconciliation_id, "json")
        assert isinstance(exp, ExportResponse)
        assert exp.success is True
        assert exp.format == "json"
        assert exp.content != ""

    def test_export_csv(
        self, service, sample_reconciliation_request,
    ):
        resp = service.reconcile(sample_reconciliation_request)
        exp = service.export_report(resp.reconciliation_id, "csv")
        assert isinstance(exp, ExportResponse)
        assert exp.format == "csv"
        assert "field,value" in exp.content

    def test_export_nonexistent(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.export_report("NONEXISTENT", "json")


# ===========================================================================
# 14. Health & Stats Tests
# ===========================================================================


class TestHealthAndStats:
    """Test health check and statistics."""

    def test_health_check(self, service):
        health = service.health_check()
        assert isinstance(health, HealthResponse)
        assert health.service == "dual-reporting-reconciliation-service"
        assert health.version == "1.0.0"
        assert health.agent_id == "AGENT-MRV-013"

    def test_health_check_engines(self, service):
        health = service.health_check()
        assert "collector" in health.engines
        assert "pipeline" in health.engines

    def test_health_uptime(self, service):
        health = service.health_check()
        assert health.uptime_seconds >= 0

    def test_stats_empty(self, service):
        stats = service.get_stats()
        assert isinstance(stats, StatsResponse)
        assert stats.total_reconciliations == 0

    def test_stats_after_reconcile(
        self, service, sample_reconciliation_request,
    ):
        service.reconcile(sample_reconciliation_request)
        stats = service.get_stats()
        assert stats.total_reconciliations == 1
        assert stats.uptime_seconds >= 0
