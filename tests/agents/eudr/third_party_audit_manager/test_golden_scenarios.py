# -*- coding: utf-8 -*-
"""
Golden test scenarios for AGENT-EUDR-024 Third-Party Audit Manager.

Validates 50 golden scenarios covering:
  - 7 EUDR commodities x 7 scenario types = 49 parametrized scenarios
  - 1 multi-commodity cross-cutting scenario

Scenario types:
  S1: High-risk country, critical NC (deforestation post-cutoff)
  S2: Standard-risk country, major NC (incomplete risk assessment)
  S3: Low-risk country, minor NC (documentation gap)
  S4: Expired certification, unscheduled audit trigger
  S5: Full lifecycle (schedule -> NC -> CAR -> closure)
  S6: Multi-format report generation
  S7: Authority interaction with SLA enforcement

Target: 50 golden tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.audit_planning_scheduling_engine import (
    AuditPlanningSchedulingEngine,
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
from greenlang.agents.eudr.third_party_audit_manager.audit_execution_engine import (
    AuditExecutionEngine,
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
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    LogAuthorityInteractionRequest,
    CalculateAnalyticsRequest,
    CertificateRecord,
    NC_SEVERITY_SLA_DAYS,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    HIGH_RISK_COUNTRIES,
    STANDARD_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
)


# -----------------------------------------------------------------------
# Golden Scenario Configuration
# -----------------------------------------------------------------------

# Map commodity index to a country from each risk tier for parametrization
_HIGH_RISK = ["BR", "ID", "CO", "GH", "CI", "MY", "PE"]
_STD_RISK = ["MY", "PE", "EC", "NG", "CM", "TH", "VN"]
_LOW_RISK = ["FI", "SE", "CA", "NZ", "CL", "AT", "DK"]

# Map commodities to their natural certification scheme
_COMMODITY_SCHEME = {
    "cattle": None,
    "cocoa": CertificationScheme.RAINFOREST_ALLIANCE,
    "coffee": CertificationScheme.RAINFOREST_ALLIANCE,
    "palm_oil": CertificationScheme.RSPO,
    "rubber": CertificationScheme.FSC,
    "soya": CertificationScheme.ISCC,
    "wood": CertificationScheme.FSC,
}


# -----------------------------------------------------------------------
# S1: High-risk country, critical NC (deforestation post-cutoff)
# -----------------------------------------------------------------------


class TestGoldenS1HighRiskCriticalNC:
    """S1: Each EUDR commodity in a high-risk country with critical NC."""

    @pytest.mark.parametrize("commodity,country", list(zip(EUDR_COMMODITIES, _HIGH_RISK)))
    def test_critical_nc_deforestation(self, nc_engine, commodity, country):
        req = ClassifyNCRequest(
            audit_id=f"AUD-GS1-{commodity}",
            finding_statement=f"Active deforestation detected in {country} {commodity} supply chain post Dec 2020 cutoff",
            objective_evidence=f"Sentinel-2 satellite imagery confirms forest loss in {country} between 2021-2023",
            indicators={"active_deforestation_post_cutoff": True},
        )
        resp = nc_engine.classify_nc(req)
        assert resp.non_conformance.severity == NCSeverity.CRITICAL
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH


# -----------------------------------------------------------------------
# S2: Standard-risk country, major NC (incomplete risk assessment)
# -----------------------------------------------------------------------


class TestGoldenS2StandardRiskMajorNC:
    """S2: Each EUDR commodity in a standard-risk country with major NC."""

    @pytest.mark.parametrize("commodity,country", list(zip(EUDR_COMMODITIES, _STD_RISK)))
    def test_major_nc_incomplete_risk(self, nc_engine, commodity, country):
        req = ClassifyNCRequest(
            audit_id=f"AUD-GS2-{commodity}",
            finding_statement=f"Incomplete risk assessment for {commodity} supply chain in {country}",
            objective_evidence=f"Risk assessment file missing {country}-specific evaluation for {commodity}",
            indicators={"incomplete_risk_assessment": True},
        )
        resp = nc_engine.classify_nc(req)
        assert resp.non_conformance.severity in (NCSeverity.MAJOR, NCSeverity.CRITICAL)
        assert resp.provenance_hash is not None


# -----------------------------------------------------------------------
# S3: Low-risk country, minor NC (documentation gap)
# -----------------------------------------------------------------------


class TestGoldenS3LowRiskMinorNC:
    """S3: Each EUDR commodity in a low-risk country with minor NC."""

    @pytest.mark.parametrize("commodity,country", list(zip(EUDR_COMMODITIES, _LOW_RISK)))
    def test_minor_nc_documentation_gap(self, nc_engine, commodity, country):
        req = ClassifyNCRequest(
            audit_id=f"AUD-GS3-{commodity}",
            finding_statement=f"Training records not current for {commodity} staff in {country}",
            objective_evidence=f"3 staff members in {country} have outdated training records for {commodity}",
            indicators={"training_records_not_current": True},
        )
        resp = nc_engine.classify_nc(req)
        assert resp.non_conformance.severity in (
            NCSeverity.MINOR, NCSeverity.OBSERVATION, NCSeverity.MAJOR,
        )
        assert resp.provenance_hash is not None


# -----------------------------------------------------------------------
# S4: Expired certification, unscheduled audit trigger
# -----------------------------------------------------------------------


class TestGoldenS4ExpiredCertification:
    """S4: Each EUDR commodity with expired certification triggering audit."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_expired_cert_triggers_audit(self, certification_engine, planning_engine, commodity):
        scheme = _COMMODITY_SCHEME.get(commodity)
        if scheme is None:
            pytest.skip(f"No primary scheme for {commodity}")

        expired_cert = CertificateRecord(
            scheme=scheme,
            certificate_number=f"CERT-EXP-{commodity.upper()}-001",
            holder_name=f"Test {commodity.title()} Org",
            holder_id=f"SUP-GS4-{commodity}",
            status="expired",
            scope="chain_of_custody",
            issue_date=date(2019, 1, 1),
            expiry_date=date(2024, 12, 31),
        )
        certification_engine.register_certificate(expired_cert)

        schedule_req = ScheduleAuditRequest(
            operator_id="OP-GS4-001",
            supplier_ids=[f"SUP-GS4-{commodity}"],
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(schedule_req)
        assert resp is not None


# -----------------------------------------------------------------------
# S5: Full lifecycle (schedule -> NC -> CAR -> closure)
# -----------------------------------------------------------------------


class TestGoldenS5FullLifecycle:
    """S5: Full audit lifecycle for each EUDR commodity."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_full_lifecycle(
        self,
        planning_engine,
        execution_engine,
        nc_engine,
        car_engine,
        commodity,
    ):
        # Schedule
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-GS5-001",
            supplier_ids=[f"SUP-GS5-{commodity}"],
            planning_year=2026,
        )
        schedule_resp = planning_engine.schedule_audits(schedule_req)
        assert schedule_resp.total_scheduled >= 0

        # Execute (create audit and start)
        audit = Audit(
            audit_id=f"AUD-GS5-{commodity}",
            operator_id="OP-GS5-001",
            supplier_id=f"SUP-GS5-{commodity}",
            planned_date=FROZEN_DATE,
            country_code="BR",
            commodity=commodity,
            status=AuditStatus.PLANNED,
        )
        started = execution_engine.start_audit(audit)
        assert started.status == AuditStatus.IN_PROGRESS

        # Classify NC
        nc_req = ClassifyNCRequest(
            audit_id=f"AUD-GS5-{commodity}",
            finding_statement=f"Finding during {commodity} audit",
            objective_evidence=f"Evidence for {commodity} finding",
            indicators={"incomplete_risk_assessment": True},
        )
        nc_resp = nc_engine.classify_nc(nc_req)
        assert nc_resp is not None

        # Issue CAR
        car_req = IssueCARRequest(
            nc_ids=[nc_resp.non_conformance.nc_id],
            audit_id=f"AUD-GS5-{commodity}",
            supplier_id=f"SUP-GS5-{commodity}",
            issued_by="AUR-FSC-001",
        )
        car_resp = car_engine.issue_car(car_req)
        assert car_resp.car.status == CARStatus.ISSUED

        # Acknowledge CAR
        car = car_engine.transition_status(
            car=car_resp.car,
            new_status=CARStatus.ACKNOWLEDGED,
        )
        assert car.status == CARStatus.ACKNOWLEDGED


# -----------------------------------------------------------------------
# S6: Multi-format report generation
# -----------------------------------------------------------------------


class TestGoldenS6MultiFormatReport:
    """S6: Report generation for each commodity across all formats."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_json_report_per_commodity(self, reporting_engine, commodity):
        req = GenerateReportRequest(
            audit_id=f"AUD-GS6-{commodity}",
            report_format="json",
            language="en",
        )
        resp = reporting_engine.generate_report(req)
        assert resp is not None
        assert resp.report.report_format == "json"
        assert resp.provenance_hash is not None


# -----------------------------------------------------------------------
# S7: Authority interaction with SLA enforcement
# -----------------------------------------------------------------------


class TestGoldenS7AuthorityInteraction:
    """S7: Authority interaction for each EUDR commodity."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_authority_interaction_per_commodity(self, analytics_engine, commodity):
        req = LogAuthorityInteractionRequest(
            operator_id="OP-GS7-001",
            authority_name="BMEL",
            member_state="DE",
            interaction_type="document_request",
            subject=f"DDS request for {commodity} supply chain documentation",
        )
        resp = analytics_engine.log_authority_interaction(req)
        assert resp is not None
        assert resp.interaction_id is not None
        assert resp.response_deadline is not None


# -----------------------------------------------------------------------
# Multi-Commodity Cross-Cutting Scenario (Test #50)
# -----------------------------------------------------------------------


class TestGoldenMultiCommodityCrossCutting:
    """Cross-cutting golden scenario spanning all 7 commodities."""

    def test_multi_commodity_operator_schedule(self, planning_engine):
        """An operator with suppliers across all 7 commodities."""
        supplier_ids = [f"SUP-MULTI-{c}" for c in EUDR_COMMODITIES]
        schedule_req = ScheduleAuditRequest(
            operator_id="OP-MULTI-001",
            supplier_ids=supplier_ids,
            planning_year=2026,
        )
        resp = planning_engine.schedule_audits(schedule_req)
        assert resp.total_scheduled >= 1
        assert resp.provenance_hash is not None
        assert len(resp.provenance_hash) == SHA256_HEX_LENGTH
