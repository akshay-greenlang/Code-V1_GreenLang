# -*- coding: utf-8 -*-
"""
Performance benchmark tests for AGENT-EUDR-024 Third-Party Audit Manager.

Tests latency targets, throughput benchmarks, memory usage, and batch
processing performance for all eight processing engines. Performance
targets from PRD:
  - Audit scheduling: < 500ms for 100 suppliers
  - NC classification: < 100ms per NC
  - CAR issuance: < 100ms per CAR
  - CAR transition: < 50ms per transition
  - Report generation: < 2s per report
  - Certificate sync: < 500ms
  - Coverage gap analysis: < 200ms
  - Analytics calculation: < 1s for 1-year window

Target: ~20 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import time
from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.audit_planning_scheduling_engine import (
    AuditPlanningSchedulingEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.auditor_registry_qualification_engine import (
    AuditorRegistryQualificationEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_execution_engine import (
    AuditExecutionEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.non_conformance_detection_engine import (
    NonConformanceDetectionEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.car_management_engine import (
    CARManagementEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.certification_integration_engine import (
    CertificationIntegrationEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_reporting_engine import (
    AuditReportingEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_analytics_engine import (
    AuditAnalyticsEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    Audit,
    AuditStatus,
    AuditScope,
    NCSeverity,
    CARStatus,
    CertificationScheme,
    ScheduleAuditRequest,
    MatchAuditorRequest,
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    CalculateAnalyticsRequest,
    CorrectiveActionRequest,
    CertificateRecord,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    EUDR_COMMODITIES,
)


def _elapsed_ms(start: float) -> float:
    """Return elapsed time in milliseconds since start."""
    return (time.perf_counter() - start) * 1000.0


# -----------------------------------------------------------------------
# Scheduling Performance
# -----------------------------------------------------------------------


class TestSchedulingPerformance:
    """Performance tests for audit scheduling engine."""

    @pytest.mark.performance
    def test_schedule_100_suppliers_under_500ms(self, planning_engine):
        """Audit scheduling for 100 suppliers must complete in < 500ms."""
        req = ScheduleAuditRequest(
            operator_id="OP-PERF-001",
            supplier_ids=[f"SUP-PERF-{i:04d}" for i in range(100)],
            planning_year=2026,
        )
        start = time.perf_counter()
        resp = planning_engine.schedule_audits(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 500.0, f"Scheduling took {elapsed:.1f}ms (target: <500ms)"

    @pytest.mark.performance
    def test_schedule_500_suppliers_under_2500ms(self, planning_engine):
        """Audit scheduling for 500 suppliers must complete in < 2500ms."""
        req = ScheduleAuditRequest(
            operator_id="OP-PERF-002",
            supplier_ids=[f"SUP-PERF-{i:04d}" for i in range(500)],
            planning_year=2026,
        )
        start = time.perf_counter()
        resp = planning_engine.schedule_audits(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 2500.0, f"Scheduling took {elapsed:.1f}ms (target: <2500ms)"

    @pytest.mark.performance
    def test_priority_score_calculation_under_5ms(self, planning_engine):
        """Single priority score calculation must complete in < 5ms."""
        start = time.perf_counter()
        score = planning_engine.calculate_priority_score(
            country_risk=Decimal("80"),
            supplier_risk=Decimal("60"),
            nc_history_score=Decimal("50"),
            certification_gap_score=Decimal("40"),
            deforestation_alert_score=Decimal("30"),
        )
        elapsed = _elapsed_ms(start)
        assert score is not None
        assert elapsed < 5.0, f"Priority calc took {elapsed:.1f}ms (target: <5ms)"


# -----------------------------------------------------------------------
# NC Classification Performance
# -----------------------------------------------------------------------


class TestNCClassificationPerformance:
    """Performance tests for NC classification engine."""

    @pytest.mark.performance
    def test_single_nc_classification_under_100ms(self, nc_engine):
        """Single NC classification must complete in < 100ms."""
        req = ClassifyNCRequest(
            audit_id="AUD-PERF-NC-001",
            finding_statement="Evidence of active deforestation post-cutoff",
            objective_evidence="Satellite imagery confirms forest loss 2021-2023",
            indicators={"active_deforestation_post_cutoff": True},
        )
        start = time.perf_counter()
        resp = nc_engine.classify_nc(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 100.0, f"NC classification took {elapsed:.1f}ms (target: <100ms)"

    @pytest.mark.performance
    def test_batch_50_nc_classifications_under_2s(self, nc_engine):
        """Batch classification of 50 NCs must complete in < 2000ms."""
        requests = [
            ClassifyNCRequest(
                audit_id=f"AUD-PERF-NC-{i:03d}",
                finding_statement=f"Test finding {i}",
                objective_evidence=f"Test evidence {i}",
                indicators={"incomplete_risk_assessment": True} if i % 2 == 0 else {},
            )
            for i in range(50)
        ]
        start = time.perf_counter()
        for req in requests:
            nc_engine.classify_nc(req)
        elapsed = _elapsed_ms(start)
        assert elapsed < 2000.0, f"Batch NC classification took {elapsed:.1f}ms (target: <2000ms)"


# -----------------------------------------------------------------------
# CAR Management Performance
# -----------------------------------------------------------------------


class TestCARPerformance:
    """Performance tests for CAR management engine."""

    @pytest.mark.performance
    def test_car_issuance_under_100ms(self, car_engine):
        """Single CAR issuance must complete in < 100ms."""
        req = IssueCARRequest(
            nc_ids=["NC-PERF-001"],
            audit_id="AUD-PERF-CAR-001",
            supplier_id="SUP-PERF-001",
            issued_by="AUR-FSC-001",
        )
        start = time.perf_counter()
        resp = car_engine.issue_car(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 100.0, f"CAR issuance took {elapsed:.1f}ms (target: <100ms)"

    @pytest.mark.performance
    def test_car_transition_under_50ms(self, car_engine):
        """Single CAR status transition must complete in < 50ms."""
        car = CorrectiveActionRequest(
            nc_ids=["NC-PERF-002"],
            audit_id="AUD-PERF-CAR-002",
            supplier_id="SUP-PERF-002",
            severity=NCSeverity.MAJOR,
            sla_deadline=FROZEN_NOW + timedelta(days=90),
            status=CARStatus.ISSUED,
            issued_by="AUR-FSC-001",
            issued_at=FROZEN_NOW,
        )
        start = time.perf_counter()
        result = car_engine.transition_status(car=car, new_status=CARStatus.ACKNOWLEDGED)
        elapsed = _elapsed_ms(start)
        assert result is not None
        assert elapsed < 50.0, f"CAR transition took {elapsed:.1f}ms (target: <50ms)"

    @pytest.mark.performance
    def test_batch_sla_check_100_cars_under_500ms(self, car_engine):
        """Batch SLA check for 100 CARs must complete in < 500ms."""
        cars = [
            CorrectiveActionRequest(
                nc_ids=[f"NC-PERF-SLA-{i:03d}"],
                audit_id="AUD-PERF-SLA-001",
                supplier_id="SUP-PERF-SLA-001",
                severity=NCSeverity.MAJOR,
                sla_deadline=FROZEN_NOW + timedelta(days=90 - i),
                status=CARStatus.IN_PROGRESS,
                issued_by="AUR-FSC-001",
                issued_at=FROZEN_NOW - timedelta(days=i),
            )
            for i in range(100)
        ]
        start = time.perf_counter()
        results = car_engine.batch_check_sla(cars)
        elapsed = _elapsed_ms(start)
        assert results is not None
        assert elapsed < 500.0, f"Batch SLA check took {elapsed:.1f}ms (target: <500ms)"


# -----------------------------------------------------------------------
# Report Generation Performance
# -----------------------------------------------------------------------


class TestReportPerformance:
    """Performance tests for report generation engine."""

    @pytest.mark.performance
    def test_json_report_under_2s(self, reporting_engine):
        """JSON report generation must complete in < 2000ms."""
        req = GenerateReportRequest(
            audit_id="AUD-PERF-RPT-001",
            report_format="json",
            language="en",
        )
        start = time.perf_counter()
        resp = reporting_engine.generate_report(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 2000.0, f"JSON report took {elapsed:.1f}ms (target: <2000ms)"

    @pytest.mark.performance
    def test_pdf_report_under_2s(self, reporting_engine):
        """PDF report generation must complete in < 2000ms."""
        req = GenerateReportRequest(
            audit_id="AUD-PERF-RPT-002",
            report_format="pdf",
            language="en",
        )
        start = time.perf_counter()
        resp = reporting_engine.generate_report(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 2000.0, f"PDF report took {elapsed:.1f}ms (target: <2000ms)"

    @pytest.mark.performance
    def test_multi_language_reports_under_10s(self, reporting_engine):
        """5 language reports must complete in < 10000ms total."""
        languages = ["en", "fr", "de", "es", "pt"]
        start = time.perf_counter()
        for lang in languages:
            req = GenerateReportRequest(
                audit_id="AUD-PERF-RPT-003",
                report_format="json",
                language=lang,
            )
            reporting_engine.generate_report(req)
        elapsed = _elapsed_ms(start)
        assert elapsed < 10000.0, f"Multi-lang reports took {elapsed:.1f}ms (target: <10000ms)"


# -----------------------------------------------------------------------
# Analytics Performance
# -----------------------------------------------------------------------


class TestAnalyticsPerformance:
    """Performance tests for analytics engine."""

    @pytest.mark.performance
    def test_analytics_1_year_window_under_1s(self, analytics_engine):
        """Analytics for a 1-year window must complete in < 1000ms."""
        req = CalculateAnalyticsRequest(
            operator_id="OP-PERF-ANA-001",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        )
        start = time.perf_counter()
        resp = analytics_engine.calculate_analytics(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 1000.0, f"Analytics took {elapsed:.1f}ms (target: <1000ms)"

    @pytest.mark.performance
    def test_analytics_90_day_window_under_500ms(self, analytics_engine):
        """Analytics for a 90-day window must complete in < 500ms."""
        req = CalculateAnalyticsRequest(
            operator_id="OP-PERF-ANA-002",
            time_period_start=FROZEN_DATE - timedelta(days=90),
            time_period_end=FROZEN_DATE,
        )
        start = time.perf_counter()
        resp = analytics_engine.calculate_analytics(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 500.0, f"Analytics took {elapsed:.1f}ms (target: <500ms)"


# -----------------------------------------------------------------------
# Certification Engine Performance
# -----------------------------------------------------------------------


class TestCertificationPerformance:
    """Performance tests for certification integration engine."""

    @pytest.mark.performance
    def test_certificate_verification_under_500ms(self, certification_engine):
        """Certificate verification must complete in < 500ms."""
        cert = CertificateRecord(
            scheme=CertificationScheme.FSC,
            certificate_number="FSC-PERF-001",
            holder_name="Perf Test Org",
            holder_id="SUP-PERF-CERT-001",
            status="active",
            issue_date=date(2023, 1, 1),
            expiry_date=date(2028, 12, 31),
        )
        certification_engine.register_certificate(cert)
        start = time.perf_counter()
        result = certification_engine.verify_certificate(cert)
        elapsed = _elapsed_ms(start)
        assert result is not None
        assert elapsed < 500.0, f"Cert verification took {elapsed:.1f}ms (target: <500ms)"

    @pytest.mark.performance
    def test_coverage_gap_analysis_under_200ms(self, certification_engine):
        """Coverage gap analysis must complete in < 200ms."""
        cert = CertificateRecord(
            scheme=CertificationScheme.FSC,
            certificate_number="FSC-PERF-002",
            holder_name="Gap Test Org",
            holder_id="SUP-PERF-GAP-001",
            status="active",
            issue_date=date(2023, 1, 1),
            expiry_date=date(2028, 12, 31),
        )
        certification_engine.register_certificate(cert)
        start = time.perf_counter()
        result = certification_engine.analyze_coverage_gap(
            supplier_id="SUP-PERF-GAP-001",
            commodity="wood",
        )
        elapsed = _elapsed_ms(start)
        assert result is not None
        assert elapsed < 200.0, f"Gap analysis took {elapsed:.1f}ms (target: <200ms)"


# -----------------------------------------------------------------------
# Auditor Matching Performance
# -----------------------------------------------------------------------


class TestAuditorMatchingPerformance:
    """Performance tests for auditor matching engine."""

    @pytest.mark.performance
    def test_match_auditors_under_200ms(self, auditor_registry_engine, sample_auditor_fsc):
        """Auditor matching must complete in < 200ms."""
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        req = MatchAuditorRequest(
            audit_id="AUD-PERF-MATCH-001",
            commodity="wood",
            country_code="BR",
        )
        start = time.perf_counter()
        resp = auditor_registry_engine.match_auditors(req)
        elapsed = _elapsed_ms(start)
        assert resp is not None
        assert elapsed < 200.0, f"Auditor matching took {elapsed:.1f}ms (target: <200ms)"
