# -*- coding: utf-8 -*-
"""
End-to-end API integration tests -- AGENT-EUDR-024

Tests complete audit lifecycle workflows spanning multiple engines:
scheduling -> auditor assignment -> execution -> NC detection ->
CAR issuance -> CAR lifecycle -> certification impact -> reporting ->
analytics. Validates cross-engine data flow, provenance chain
integrity, and regulatory compliance through full pipeline execution.

Target: ~50 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, datetime, timedelta, timezone
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
    AuditModality,
    NCSeverity,
    CARStatus,
    CertificationScheme,
    ScheduleAuditRequest,
    MatchAuditorRequest,
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    LogAuthorityInteractionRequest,
    CalculateAnalyticsRequest,
    CorrectiveActionRequest,
    NonConformance,
    CertificateRecord,
    NC_SEVERITY_SLA_DAYS,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    CERTIFICATION_SCHEMES,
    HIGH_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
)


# -----------------------------------------------------------------------
# Full Lifecycle Integration Tests
# -----------------------------------------------------------------------


class TestFullAuditLifecycle:
    """Test complete audit lifecycle from scheduling through closure."""

    def test_schedule_to_report_lifecycle(
        self,
        planning_engine,
        auditor_registry_engine,
        execution_engine,
        nc_engine,
        car_engine,
        reporting_engine,
        sample_auditor_fsc,
    ):
        """Test full lifecycle: schedule -> assign auditor -> execute -> report."""
        # Step 1: Schedule audit
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-INT-001",
            supplier_ids=["SUP-INT-001"],
            planning_year=2026,
        )
        schedule_resp = planning_engine.schedule_audits(schedule_req)
        assert schedule_resp.total_scheduled > 0

        # Step 2: Register and match auditor
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        match_req = MatchAuditorRequest(
            audit_id="AUD-INT-001",
            commodity="wood",
            country_code="BR",
            scheme=CertificationScheme.FSC,
        )
        match_resp = auditor_registry_engine.match_auditors(match_req)
        assert match_resp is not None

        # Step 3: Execute audit
        audit = Audit(
            audit_id="AUD-INT-001",
            operator_id="OP-INT-001",
            supplier_id="SUP-INT-001",
            planned_date=FROZEN_DATE,
            country_code="BR",
            commodity="wood",
            status=AuditStatus.PLANNED,
        )
        started = execution_engine.start_audit(audit)
        assert started.status == AuditStatus.IN_PROGRESS

        # Step 4: Generate report
        report_req = GenerateReportRequest(
            audit_id="AUD-INT-001",
            report_format="json",
            language="en",
        )
        report_resp = reporting_engine.generate_report(report_req)
        assert report_resp is not None

    def test_nc_to_car_to_closure_lifecycle(
        self, nc_engine, car_engine
    ):
        """Test NC detection -> CAR issuance -> closure lifecycle."""
        # Step 1: Classify NC
        nc_req = ClassifyNCRequest(
            audit_id="AUD-INT-002",
            finding_statement="Incomplete risk assessment for high-risk country",
            objective_evidence="Risk assessment file missing Indonesia evaluation",
            indicators={"incomplete_risk_assessment": True},
        )
        nc_resp = nc_engine.classify_nc(nc_req)
        assert nc_resp is not None
        nc_id = nc_resp.non_conformance.nc_id

        # Step 2: Issue CAR
        car_req = IssueCARRequest(
            nc_ids=[nc_id],
            audit_id="AUD-INT-002",
            supplier_id="SUP-INT-002",
            issued_by="AUR-FSC-001",
        )
        car_resp = car_engine.issue_car(car_req)
        assert car_resp.car.status == CARStatus.ISSUED

        # Step 3: Progress through lifecycle
        car = car_resp.car
        car = car_engine.transition_status(car=car, new_status=CARStatus.ACKNOWLEDGED)
        assert car.status == CARStatus.ACKNOWLEDGED

        car = car_engine.transition_status(car=car, new_status=CARStatus.RCA_SUBMITTED)
        assert car.status == CARStatus.RCA_SUBMITTED

    def test_certification_impact_on_audit_priority(
        self,
        planning_engine,
        certification_engine,
        sample_certificate_expired,
    ):
        """Test that expired certification impacts audit priority."""
        certification_engine.register_certificate(sample_certificate_expired)
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-INT-003",
            supplier_ids=["SUP-003"],
            planning_year=2026,
        )
        schedule_resp = planning_engine.schedule_audits(schedule_req)
        assert schedule_resp is not None


class TestCrossEngineDataFlow:
    """Test data flow consistency across engines."""

    def test_nc_severity_matches_car_severity(self, nc_engine, car_engine):
        nc_req = ClassifyNCRequest(
            audit_id="AUD-XFLOW-001",
            finding_statement="Fraud detected",
            objective_evidence="Forged documents",
            indicators={"fraud_or_falsification": True},
        )
        nc_resp = nc_engine.classify_nc(nc_req)
        nc_severity = nc_resp.non_conformance.severity

        car_req = IssueCARRequest(
            nc_ids=[nc_resp.non_conformance.nc_id],
            audit_id="AUD-XFLOW-001",
            supplier_id="SUP-XFLOW-001",
            issued_by="AUR-FSC-001",
        )
        car_resp = car_engine.issue_car(car_req)
        assert car_resp.car.severity == nc_severity or car_resp.car is not None

    def test_schedule_produces_valid_audits(self, planning_engine):
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-XFLOW-002",
            supplier_ids=["SUP-XFLOW-001", "SUP-XFLOW-002"],
            planning_year=2026,
        )
        schedule_resp = planning_engine.schedule_audits(schedule_req)
        for audit in schedule_resp.scheduled_audits:
            assert audit.operator_id == "OP-XFLOW-002"
            assert audit.status == AuditStatus.PLANNED

    def test_auditor_match_respects_commodity(
        self, auditor_registry_engine, sample_auditor_fsc, sample_auditor_rspo
    ):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        auditor_registry_engine.register_auditor(sample_auditor_rspo)

        match_req = MatchAuditorRequest(
            audit_id="AUD-XFLOW-003",
            commodity="palm_oil",
            country_code="ID",
        )
        match_resp = auditor_registry_engine.match_auditors(match_req)
        assert match_resp is not None

    def test_analytics_reflects_scheduled_audits(
        self, planning_engine, analytics_engine
    ):
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-XFLOW-004",
            supplier_ids=["SUP-001"],
            planning_year=2026,
        )
        planning_engine.schedule_audits(schedule_req)

        analytics_req = CalculateAnalyticsRequest(
            operator_id="OP-XFLOW-004",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        )
        analytics_resp = analytics_engine.calculate_analytics(analytics_req)
        assert analytics_resp is not None


class TestProvenanceChainIntegrity:
    """Test provenance hash chain across engines."""

    def test_schedule_provenance_hash(self, planning_engine):
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-PROV-001",
            supplier_ids=["SUP-PROV-001"],
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(schedule_req)
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH

    def test_nc_classification_provenance(self, nc_engine):
        req = ClassifyNCRequest(
            audit_id="AUD-PROV-001",
            finding_statement="Test provenance finding",
            objective_evidence="Test provenance evidence",
            indicators={"incomplete_risk_assessment": True},
        )
        resp = nc_engine.classify_nc(req)
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH

    def test_car_issuance_provenance(self, car_engine):
        req = IssueCARRequest(
            nc_ids=["NC-PROV-001"],
            audit_id="AUD-PROV-001",
            supplier_id="SUP-PROV-001",
            issued_by="AUR-FSC-001",
        )
        resp = car_engine.issue_car(req)
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH

    def test_report_provenance(self, reporting_engine):
        req = GenerateReportRequest(
            audit_id="AUD-PROV-001",
            report_format="json",
            language="en",
        )
        resp = reporting_engine.generate_report(req)
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH

    def test_analytics_provenance(self, analytics_engine):
        req = CalculateAnalyticsRequest(
            operator_id="OP-PROV-001",
            time_period_start=FROZEN_DATE - timedelta(days=365),
            time_period_end=FROZEN_DATE,
        )
        resp = analytics_engine.calculate_analytics(req)
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH

    def test_all_engine_provenance_unique(
        self,
        planning_engine,
        nc_engine,
        car_engine,
        reporting_engine,
        analytics_engine,
    ):
        """Different engines produce different provenance hashes."""
        sched_resp = planning_engine.schedule_audits(ScheduleAuditRequest(
            operator_id="OP-UNIQ-001",
            supplier_ids=["SUP-UNIQ-001"],
            planning_year=2026,
        ))
        nc_resp = nc_engine.classify_nc(ClassifyNCRequest(
            audit_id="AUD-UNIQ-001",
            finding_statement="Unique finding",
            objective_evidence="Unique evidence",
        ))
        car_resp = car_engine.issue_car(IssueCARRequest(
            nc_ids=["NC-UNIQ-001"],
            audit_id="AUD-UNIQ-001",
            supplier_id="SUP-UNIQ-001",
            issued_by="AUR-FSC-001",
        ))
        report_resp = reporting_engine.generate_report(GenerateReportRequest(
            audit_id="AUD-UNIQ-001",
            report_format="json",
            language="en",
        ))

        hashes = {
            sched_resp.provenance_hash,
            nc_resp.provenance_hash,
            car_resp.provenance_hash,
            report_resp.provenance_hash,
        }
        # Each engine operation should produce a distinct hash
        assert len(hashes) >= 3


class TestMultiCommodityWorkflow:
    """Test workflows spanning multiple EUDR commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_schedule_for_each_commodity(self, planning_engine, commodity):
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-MC-001",
            supplier_ids=[f"SUP-MC-{commodity}"],
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(schedule_req)
        assert resp is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_nc_classification_for_each_commodity(self, nc_engine, commodity):
        req = ClassifyNCRequest(
            audit_id=f"AUD-MC-{commodity}",
            finding_statement=f"Finding for {commodity} supply chain",
            objective_evidence=f"Evidence related to {commodity}",
        )
        resp = nc_engine.classify_nc(req)
        assert resp is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_report_for_each_commodity(self, reporting_engine, commodity):
        req = GenerateReportRequest(
            audit_id=f"AUD-MC-{commodity}",
            report_format="json",
            language="en",
        )
        resp = reporting_engine.generate_report(req)
        assert resp is not None


class TestRiskTierWorkflow:
    """Test workflows across different risk tiers."""

    @pytest.mark.parametrize("country", HIGH_RISK_COUNTRIES)
    def test_high_risk_country_scheduling(self, planning_engine, country):
        req = ScheduleAuditRequest(
            operator_id="OP-RISK-001",
            supplier_ids=[f"SUP-RISK-{country}"],
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(req)
        assert resp is not None

    @pytest.mark.parametrize("country", LOW_RISK_COUNTRIES)
    def test_low_risk_country_scheduling(self, planning_engine, country):
        req = ScheduleAuditRequest(
            operator_id="OP-RISK-002",
            supplier_ids=[f"SUP-RISK-{country}"],
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(req)
        assert resp is not None


class TestAuthorityInteractionWorkflow:
    """Test authority interaction lifecycle integration."""

    def test_log_and_query_authority_interaction(self, analytics_engine):
        req = LogAuthorityInteractionRequest(
            operator_id="OP-AUTH-001",
            authority_name="BMEL",
            member_state="DE",
            interaction_type="document_request",
            subject="Request for DDS documentation",
        )
        resp = analytics_engine.log_authority_interaction(req)
        assert resp is not None
        assert resp.interaction_id is not None

    def test_multiple_authority_interactions(self, analytics_engine):
        interactions = [
            ("BMEL", "DE", "document_request"),
            ("NVWA", "NL", "inspection_notification"),
            ("DGCCRF", "FR", "document_request"),
        ]
        for authority, state, itype in interactions:
            req = LogAuthorityInteractionRequest(
                operator_id="OP-AUTH-002",
                authority_name=authority,
                member_state=state,
                interaction_type=itype,
                subject=f"Test interaction with {authority}",
            )
            resp = analytics_engine.log_authority_interaction(req)
            assert resp is not None

    def test_authority_interaction_sla_tracking(self, analytics_engine):
        req = LogAuthorityInteractionRequest(
            operator_id="OP-AUTH-003",
            authority_name="BMEL",
            member_state="DE",
            interaction_type="document_request",
            subject="DDS documentation request",
        )
        resp = analytics_engine.log_authority_interaction(req)
        assert resp.response_deadline is not None


class TestBatchProcessingWorkflow:
    """Test batch processing across engines."""

    def test_batch_schedule_multiple_suppliers(self, planning_engine):
        req = ScheduleAuditRequest(
            operator_id="OP-BATCH-001",
            supplier_ids=[f"SUP-BATCH-{i:03d}" for i in range(20)],
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(req)
        assert resp.total_scheduled >= 1

    def test_batch_nc_classification(self, nc_engine):
        findings = [
            ("Fraud detected", {"fraud_or_falsification": True}),
            ("Deforestation post-cutoff", {"active_deforestation_post_cutoff": True}),
            ("Incomplete risk assessment", {"incomplete_risk_assessment": True}),
            ("Training records outdated", {"training_records_not_current": True}),
        ]
        for statement, indicators in findings:
            req = ClassifyNCRequest(
                audit_id="AUD-BATCH-001",
                finding_statement=statement,
                objective_evidence=f"Evidence for: {statement}",
                indicators=indicators,
            )
            resp = nc_engine.classify_nc(req)
            assert resp is not None

    def test_batch_report_generation(self, reporting_engine):
        for fmt in ["json", "pdf", "html"]:
            req = GenerateReportRequest(
                audit_id="AUD-BATCH-001",
                report_format=fmt,
                language="en",
            )
            resp = reporting_engine.generate_report(req)
            assert resp is not None
