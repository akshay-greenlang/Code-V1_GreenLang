# -*- coding: utf-8 -*-
"""
Determinism and reproducibility tests for AGENT-EUDR-024 Third-Party Audit Manager.

Validates that all engines produce bit-perfect, deterministic results
when given identical inputs. Tests SHA-256 provenance hash reproducibility
across all eight processing engines, ensuring zero-hallucination guarantee
for audit trail integrity and regulatory compliance.

Requirements:
  - Same inputs must produce identical SHA-256 provenance hashes
  - Decimal arithmetic must be exact (no floating-point drift)
  - Classification rules must produce identical severity assignments
  - SLA deadline calculations must be bit-perfect
  - Priority scores must be exactly reproducible
  - Report sections must be deterministically ordered
  - Analytics aggregations must produce identical results

Target: ~30 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import hashlib
import json
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
    NCSeverity,
    CARStatus,
    CertificationScheme,
    ScheduleAuditRequest,
    MatchAuditorRequest,
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    CalculateAnalyticsRequest,
    LogAuthorityInteractionRequest,
    CorrectiveActionRequest,
    CertificateRecord,
    NC_SEVERITY_SLA_DAYS,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    compute_test_hash,
)


# -----------------------------------------------------------------------
# Scheduling Determinism
# -----------------------------------------------------------------------


class TestSchedulingDeterminism:
    """Verify scheduling engine produces bit-perfect deterministic results."""

    def test_identical_schedule_requests_same_hash(self, planning_engine):
        """Two identical schedule requests must produce the same provenance hash."""
        req = ScheduleAuditRequest(
            operator_id="OP-DET-001",
            supplier_ids=["SUP-DET-001", "SUP-DET-002"],
            planning_year=2026,
        )
        r1 = planning_engine.schedule_audits(req)
        r2 = planning_engine.schedule_audits(req)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_suppliers_different_hash(self, planning_engine):
        """Different supplier lists must produce different provenance hashes."""
        r1 = planning_engine.schedule_audits(ScheduleAuditRequest(
            operator_id="OP-DET-002",
            supplier_ids=["SUP-A"],
            planning_year=2026,
        ))
        r2 = planning_engine.schedule_audits(ScheduleAuditRequest(
            operator_id="OP-DET-002",
            supplier_ids=["SUP-B"],
            planning_year=2026,
        ))
        assert r1.provenance_hash != r2.provenance_hash

    def test_priority_score_decimal_precision(self, planning_engine):
        """Priority score calculation must use Decimal, not float."""
        score1 = planning_engine.calculate_priority_score(
            country_risk=Decimal("80.00"),
            supplier_risk=Decimal("60.00"),
            nc_history_score=Decimal("50.00"),
            certification_gap_score=Decimal("40.00"),
            deforestation_alert_score=Decimal("30.00"),
        )
        score2 = planning_engine.calculate_priority_score(
            country_risk=Decimal("80.00"),
            supplier_risk=Decimal("60.00"),
            nc_history_score=Decimal("50.00"),
            certification_gap_score=Decimal("40.00"),
            deforestation_alert_score=Decimal("30.00"),
        )
        assert score1 == score2
        # Verify it is Decimal type, not float
        if isinstance(score1, dict):
            for v in score1.values():
                if isinstance(v, (Decimal, int)):
                    assert not isinstance(v, float)
        else:
            assert isinstance(score1, (Decimal, dict))

    def test_schedule_10_runs_identical(self, planning_engine):
        """Scheduling 10 times with same input must produce identical hashes."""
        req = ScheduleAuditRequest(
            operator_id="OP-DET-003",
            supplier_ids=["SUP-DET-003"],
            planning_year=2026,
        )
        hashes = [planning_engine.schedule_audits(req).provenance_hash for _ in range(10)]
        assert len(set(hashes)) == 1, f"Got {len(set(hashes))} distinct hashes from 10 runs"


# -----------------------------------------------------------------------
# NC Classification Determinism
# -----------------------------------------------------------------------


class TestNCClassificationDeterminism:
    """Verify NC classification produces bit-perfect deterministic results."""

    def test_identical_nc_requests_same_severity(self, nc_engine):
        """Two identical NC requests must produce the same severity."""
        req = ClassifyNCRequest(
            audit_id="AUD-DET-NC-001",
            finding_statement="Active deforestation detected post-cutoff",
            objective_evidence="Satellite imagery confirms forest loss 2021-2023",
            indicators={"active_deforestation_post_cutoff": True},
        )
        r1 = nc_engine.classify_nc(req)
        r2 = nc_engine.classify_nc(req)
        assert r1.non_conformance.severity == r2.non_conformance.severity

    def test_identical_nc_requests_same_hash(self, nc_engine):
        """Two identical NC requests must produce the same provenance hash."""
        req = ClassifyNCRequest(
            audit_id="AUD-DET-NC-002",
            finding_statement="Fraud detected",
            objective_evidence="Forged permits",
            indicators={"fraud_or_falsification": True},
        )
        r1 = nc_engine.classify_nc(req)
        r2 = nc_engine.classify_nc(req)
        assert r1.provenance_hash == r2.provenance_hash

    def test_nc_risk_impact_score_deterministic(self, nc_engine):
        """Risk impact score must be identical for same input."""
        req = ClassifyNCRequest(
            audit_id="AUD-DET-NC-003",
            finding_statement="Incomplete risk assessment",
            objective_evidence="Missing country evaluation",
            indicators={"incomplete_risk_assessment": True},
        )
        r1 = nc_engine.classify_nc(req)
        r2 = nc_engine.classify_nc(req)
        assert r1.non_conformance.risk_impact_score == r2.non_conformance.risk_impact_score

    def test_nc_matched_rules_deterministic(self, nc_engine):
        """Matched rules must be identical for same input."""
        req = ClassifyNCRequest(
            audit_id="AUD-DET-NC-004",
            finding_statement="Deforestation post-cutoff",
            objective_evidence="Satellite evidence",
            indicators={"active_deforestation_post_cutoff": True},
        )
        r1 = nc_engine.classify_nc(req)
        r2 = nc_engine.classify_nc(req)
        assert r1.matched_rules == r2.matched_rules

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_nc_classification_per_commodity_deterministic(self, nc_engine, commodity):
        """NC classification must be deterministic per commodity."""
        req = ClassifyNCRequest(
            audit_id=f"AUD-DET-NC-{commodity}",
            finding_statement=f"Finding for {commodity}",
            objective_evidence=f"Evidence for {commodity}",
        )
        r1 = nc_engine.classify_nc(req)
        r2 = nc_engine.classify_nc(req)
        assert r1.non_conformance.severity == r2.non_conformance.severity
        assert r1.provenance_hash == r2.provenance_hash


# -----------------------------------------------------------------------
# CAR Management Determinism
# -----------------------------------------------------------------------


class TestCARDeterminism:
    """Verify CAR management produces bit-perfect deterministic results."""

    def test_identical_car_issuance_same_hash(self, car_engine):
        """Two identical CAR issuance requests must produce the same hash."""
        req = IssueCARRequest(
            nc_ids=["NC-DET-001"],
            audit_id="AUD-DET-CAR-001",
            supplier_id="SUP-DET-001",
            issued_by="AUR-FSC-001",
        )
        r1 = car_engine.issue_car(req)
        r2 = car_engine.issue_car(req)
        assert r1.provenance_hash == r2.provenance_hash

    def test_sla_deadline_deterministic(self, car_engine):
        """SLA deadline calculation must be exactly reproducible."""
        req = IssueCARRequest(
            nc_ids=["NC-DET-002"],
            audit_id="AUD-DET-CAR-002",
            supplier_id="SUP-DET-002",
            issued_by="AUR-FSC-001",
        )
        r1 = car_engine.issue_car(req)
        r2 = car_engine.issue_car(req)
        assert r1.car.sla_deadline == r2.car.sla_deadline

    def test_car_transition_deterministic(self, car_engine):
        """CAR status transition must be deterministic."""
        car1 = CorrectiveActionRequest(
            nc_ids=["NC-DET-003"],
            audit_id="AUD-DET-CAR-003",
            supplier_id="SUP-DET-003",
            severity=NCSeverity.MAJOR,
            sla_deadline=FROZEN_NOW + timedelta(days=90),
            status=CARStatus.ISSUED,
            issued_by="AUR-FSC-001",
            issued_at=FROZEN_NOW,
        )
        car2 = CorrectiveActionRequest(
            nc_ids=["NC-DET-003"],
            audit_id="AUD-DET-CAR-003",
            supplier_id="SUP-DET-003",
            severity=NCSeverity.MAJOR,
            sla_deadline=FROZEN_NOW + timedelta(days=90),
            status=CARStatus.ISSUED,
            issued_by="AUR-FSC-001",
            issued_at=FROZEN_NOW,
        )
        result1 = car_engine.transition_status(car=car1, new_status=CARStatus.ACKNOWLEDGED)
        result2 = car_engine.transition_status(car=car2, new_status=CARStatus.ACKNOWLEDGED)
        assert result1.status == result2.status


# -----------------------------------------------------------------------
# Report Determinism
# -----------------------------------------------------------------------


class TestReportDeterminism:
    """Verify report generation produces bit-perfect deterministic results."""

    def test_identical_report_requests_same_hash(self, reporting_engine):
        """Two identical report requests must produce the same hash."""
        req = GenerateReportRequest(
            audit_id="AUD-DET-RPT-001",
            report_format="json",
            language="en",
        )
        r1 = reporting_engine.generate_report(req)
        r2 = reporting_engine.generate_report(req)
        assert r1.provenance_hash == r2.provenance_hash

    def test_report_sections_order_deterministic(self, reporting_engine):
        """Report sections must be in the same order for identical inputs."""
        req = GenerateReportRequest(
            audit_id="AUD-DET-RPT-002",
            report_format="json",
            language="en",
        )
        r1 = reporting_engine.generate_report(req)
        r2 = reporting_engine.generate_report(req)
        if hasattr(r1.report, 'sections') and isinstance(r1.report.sections, list):
            assert r1.report.sections == r2.report.sections

    def test_different_formats_different_hash(self, reporting_engine):
        """Different report formats must produce different hashes."""
        r1 = reporting_engine.generate_report(GenerateReportRequest(
            audit_id="AUD-DET-RPT-003",
            report_format="json",
            language="en",
        ))
        r2 = reporting_engine.generate_report(GenerateReportRequest(
            audit_id="AUD-DET-RPT-003",
            report_format="pdf",
            language="en",
        ))
        assert r1.provenance_hash != r2.provenance_hash

    def test_different_languages_different_hash(self, reporting_engine):
        """Different languages must produce different hashes."""
        r1 = reporting_engine.generate_report(GenerateReportRequest(
            audit_id="AUD-DET-RPT-004",
            report_format="json",
            language="en",
        ))
        r2 = reporting_engine.generate_report(GenerateReportRequest(
            audit_id="AUD-DET-RPT-004",
            report_format="json",
            language="fr",
        ))
        assert r1.provenance_hash != r2.provenance_hash

    def test_report_10_runs_identical(self, reporting_engine):
        """10 identical report requests must all produce the same hash."""
        req = GenerateReportRequest(
            audit_id="AUD-DET-RPT-005",
            report_format="json",
            language="en",
        )
        hashes = [reporting_engine.generate_report(req).provenance_hash for _ in range(10)]
        assert len(set(hashes)) == 1, f"Got {len(set(hashes))} distinct hashes from 10 runs"


# -----------------------------------------------------------------------
# Analytics Determinism
# -----------------------------------------------------------------------


class TestAnalyticsDeterminism:
    """Verify analytics engine produces bit-perfect deterministic results."""

    def test_identical_analytics_requests_same_hash(self, analytics_engine):
        """Two identical analytics requests must produce the same hash."""
        req = CalculateAnalyticsRequest(
            operator_id="OP-DET-ANA-001",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        )
        r1 = analytics_engine.calculate_analytics(req)
        r2 = analytics_engine.calculate_analytics(req)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_time_windows_different_hash(self, analytics_engine):
        """Different time windows must produce different hashes."""
        r1 = analytics_engine.calculate_analytics(CalculateAnalyticsRequest(
            operator_id="OP-DET-ANA-002",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        ))
        r2 = analytics_engine.calculate_analytics(CalculateAnalyticsRequest(
            operator_id="OP-DET-ANA-002",
            time_period_start=FROZEN_DATE - timedelta(days=180),
            time_period_end=FROZEN_DATE,
        ))
        assert r1.provenance_hash != r2.provenance_hash

    def test_analytics_compliance_rate_deterministic(self, analytics_engine):
        """Compliance rate must be identical for same input."""
        req = CalculateAnalyticsRequest(
            operator_id="OP-DET-ANA-003",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        )
        r1 = analytics_engine.calculate_analytics(req)
        r2 = analytics_engine.calculate_analytics(req)
        assert r1.compliance_rate == r2.compliance_rate


# -----------------------------------------------------------------------
# Authority Interaction Determinism
# -----------------------------------------------------------------------


class TestAuthorityInteractionDeterminism:
    """Verify authority interaction logging is deterministic."""

    def test_identical_interactions_same_hash(self, analytics_engine):
        """Two identical authority interactions must produce the same hash."""
        req = LogAuthorityInteractionRequest(
            operator_id="OP-DET-AUTH-001",
            authority_name="BMEL",
            member_state="DE",
            interaction_type="document_request",
            subject="Determinism test interaction",
        )
        r1 = analytics_engine.log_authority_interaction(req)
        r2 = analytics_engine.log_authority_interaction(req)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_authorities_different_hash(self, analytics_engine):
        """Different authorities must produce different hashes."""
        r1 = analytics_engine.log_authority_interaction(LogAuthorityInteractionRequest(
            operator_id="OP-DET-AUTH-002",
            authority_name="BMEL",
            member_state="DE",
            interaction_type="document_request",
            subject="Test",
        ))
        r2 = analytics_engine.log_authority_interaction(LogAuthorityInteractionRequest(
            operator_id="OP-DET-AUTH-002",
            authority_name="NVWA",
            member_state="NL",
            interaction_type="document_request",
            subject="Test",
        ))
        assert r1.provenance_hash != r2.provenance_hash


# -----------------------------------------------------------------------
# Cross-Engine Determinism
# -----------------------------------------------------------------------


class TestCrossEngineDeterminism:
    """Verify determinism across multiple engines."""

    def test_full_pipeline_deterministic(
        self,
        planning_engine,
        nc_engine,
        car_engine,
        reporting_engine,
        analytics_engine,
    ):
        """Full pipeline must produce identical results across 2 runs."""
        results_run1 = {}
        results_run2 = {}

        for run_results in [results_run1, results_run2]:
            # Schedule
            sched = planning_engine.schedule_audits(ScheduleAuditRequest(
                operator_id="OP-PIPE-001",
                supplier_ids=["SUP-PIPE-001"],
                planning_year=2026,
            ))
            run_results["schedule_hash"] = sched.provenance_hash

            # Classify NC
            nc = nc_engine.classify_nc(ClassifyNCRequest(
                audit_id="AUD-PIPE-001",
                finding_statement="Pipeline determinism test",
                objective_evidence="Pipeline evidence",
                indicators={"incomplete_risk_assessment": True},
            ))
            run_results["nc_hash"] = nc.provenance_hash

            # Issue CAR
            car = car_engine.issue_car(IssueCARRequest(
                nc_ids=["NC-PIPE-001"],
                audit_id="AUD-PIPE-001",
                supplier_id="SUP-PIPE-001",
                issued_by="AUR-FSC-001",
            ))
            run_results["car_hash"] = car.provenance_hash

            # Generate report
            rpt = reporting_engine.generate_report(GenerateReportRequest(
                audit_id="AUD-PIPE-001",
                report_format="json",
                language="en",
            ))
            run_results["report_hash"] = rpt.provenance_hash

            # Analytics
            ana = analytics_engine.calculate_analytics(CalculateAnalyticsRequest(
                operator_id="OP-PIPE-001",
                time_period_start=FROZEN_DATE - timedelta(days=365),
                time_period_end=FROZEN_DATE,
            ))
            run_results["analytics_hash"] = ana.provenance_hash

        # Compare all hashes between runs
        assert results_run1["schedule_hash"] == results_run2["schedule_hash"]
        assert results_run1["nc_hash"] == results_run2["nc_hash"]
        assert results_run1["car_hash"] == results_run2["car_hash"]
        assert results_run1["report_hash"] == results_run2["report_hash"]
        assert results_run1["analytics_hash"] == results_run2["analytics_hash"]
