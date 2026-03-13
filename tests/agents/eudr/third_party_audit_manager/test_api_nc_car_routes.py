# -*- coding: utf-8 -*-
"""
API tests for NC & CAR Routes -- AGENT-EUDR-024

Tests non-conformance classification, CAR issuance, CAR lifecycle
transitions, SLA enforcement, escalation workflows, dispute management,
and status monitoring API endpoints including CRUD operations, request
validation, response format, error handling, and provenance tracking.

Target: ~40 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.non_conformance_detection_engine import (
    NonConformanceDetectionEngine,
    CRITICAL_RULES,
    MAJOR_RULES,
)
from greenlang.agents.eudr.third_party_audit_manager.car_management_engine import (
    CARManagementEngine,
    VALID_CAR_TRANSITIONS,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    ClassifyNCRequest,
    ClassifyNCResponse,
    IssueCARRequest,
    IssueCARResponse,
    NonConformance,
    CorrectiveActionRequest,
    NCSeverity,
    CARStatus,
    NC_SEVERITY_SLA_DAYS,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    NC_SEVERITIES,
    EUDR_COMMODITIES,
)


# -----------------------------------------------------------------------
# NC Classification API Tests
# -----------------------------------------------------------------------


class TestNCClassifyEndpoint:
    """Test POST /v1/eudr-tam/ncs/classify endpoint logic."""

    def test_classify_fraud_returns_critical(
        self, nc_engine, classify_nc_fraud_request
    ):
        response = nc_engine.classify_nc(classify_nc_fraud_request)
        assert response is not None
        assert response.non_conformance.severity == NCSeverity.CRITICAL

    def test_classify_deforestation_returns_critical(
        self, nc_engine, classify_nc_deforestation_request
    ):
        response = nc_engine.classify_nc(classify_nc_deforestation_request)
        assert response is not None
        assert response.non_conformance.severity == NCSeverity.CRITICAL

    def test_classify_incomplete_risk_returns_major(
        self, nc_engine, classify_nc_incomplete_risk_request
    ):
        response = nc_engine.classify_nc(classify_nc_incomplete_risk_request)
        assert response is not None
        assert response.non_conformance.severity in (NCSeverity.MAJOR, NCSeverity.CRITICAL)

    def test_classify_minor_request_returns_minor_or_above(
        self, nc_engine, classify_nc_minor_request
    ):
        response = nc_engine.classify_nc(classify_nc_minor_request)
        assert response is not None
        assert response.non_conformance.severity in (
            NCSeverity.MINOR,
            NCSeverity.MAJOR,
            NCSeverity.OBSERVATION,
        )

    def test_classify_returns_classification_rationale(
        self, nc_engine, classify_nc_fraud_request
    ):
        response = nc_engine.classify_nc(classify_nc_fraud_request)
        assert response.classification_rationale is not None
        assert len(response.classification_rationale) > 0

    def test_classify_returns_matched_rules(
        self, nc_engine, classify_nc_fraud_request
    ):
        response = nc_engine.classify_nc(classify_nc_fraud_request)
        assert isinstance(response.matched_rules, list)
        assert len(response.matched_rules) >= 1

    def test_classify_returns_provenance_hash(
        self, nc_engine, classify_nc_fraud_request
    ):
        response = nc_engine.classify_nc(classify_nc_fraud_request)
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_classify_deterministic(
        self, nc_engine, classify_nc_fraud_request
    ):
        r1 = nc_engine.classify_nc(classify_nc_fraud_request)
        r2 = nc_engine.classify_nc(classify_nc_fraud_request)
        assert r1.non_conformance.severity == r2.non_conformance.severity
        assert r1.provenance_hash == r2.provenance_hash

    def test_classify_empty_finding_raises(self, nc_engine):
        with pytest.raises((ValueError, Exception)):
            nc_engine.classify_nc(ClassifyNCRequest(
                audit_id="AUD-TEST-001",
                finding_statement="",
                objective_evidence="Some evidence",
            ))

    def test_classify_empty_evidence_raises(self, nc_engine):
        with pytest.raises((ValueError, Exception)):
            nc_engine.classify_nc(ClassifyNCRequest(
                audit_id="AUD-TEST-001",
                finding_statement="Some finding",
                objective_evidence="",
            ))

    def test_classify_returns_processing_time(
        self, nc_engine, classify_nc_fraud_request
    ):
        response = nc_engine.classify_nc(classify_nc_fraud_request)
        assert response.processing_time_ms >= Decimal("0")

    def test_classify_nc_has_risk_impact_score(
        self, nc_engine, classify_nc_fraud_request
    ):
        response = nc_engine.classify_nc(classify_nc_fraud_request)
        score = response.non_conformance.risk_impact_score
        assert Decimal("0") <= score <= Decimal("100")


class TestNCCRUD:
    """Test NC CRUD operations."""

    def test_nc_requires_audit_id(self):
        with pytest.raises((ValueError, Exception)):
            NonConformance(
                audit_id="",
                finding_statement="Finding",
                objective_evidence="Evidence",
                severity=NCSeverity.MINOR,
            )

    def test_nc_requires_finding_statement(self):
        with pytest.raises((ValueError, Exception)):
            NonConformance(
                audit_id="AUD-TEST-001",
                finding_statement="",
                objective_evidence="Evidence",
                severity=NCSeverity.MINOR,
            )

    def test_nc_requires_objective_evidence(self):
        with pytest.raises((ValueError, Exception)):
            NonConformance(
                audit_id="AUD-TEST-001",
                finding_statement="Finding",
                objective_evidence="",
                severity=NCSeverity.MINOR,
            )

    @pytest.mark.parametrize("severity", [
        NCSeverity.CRITICAL, NCSeverity.MAJOR, NCSeverity.MINOR, NCSeverity.OBSERVATION,
    ])
    def test_all_nc_severities(self, severity):
        nc = NonConformance(
            audit_id="AUD-TEST-001",
            finding_statement="Test finding",
            objective_evidence="Test evidence",
            severity=severity,
        )
        assert nc.severity == severity

    def test_nc_default_status_open(self):
        nc = NonConformance(
            audit_id="AUD-TEST-001",
            finding_statement="Test finding",
            objective_evidence="Test evidence",
            severity=NCSeverity.MINOR,
        )
        assert nc.status == "open"


# -----------------------------------------------------------------------
# CAR Issuance API Tests
# -----------------------------------------------------------------------


class TestCARIssueEndpoint:
    """Test POST /v1/eudr-tam/cars/issue endpoint logic."""

    def test_issue_car_critical(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response is not None
        assert response.car is not None
        assert response.car.status == CARStatus.ISSUED

    def test_issue_car_major(self, car_engine, issue_car_request_major):
        response = car_engine.issue_car(issue_car_request_major)
        assert response is not None
        assert response.car.status == CARStatus.ISSUED

    def test_issue_car_critical_sla_30_days(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        issued_at = response.car.issued_at
        sla_deadline = response.car.sla_deadline
        delta = sla_deadline - issued_at
        assert delta.days <= 30

    def test_issue_car_major_sla_90_days(self, car_engine, issue_car_request_major):
        response = car_engine.issue_car(issue_car_request_major)
        issued_at = response.car.issued_at
        sla_deadline = response.car.sla_deadline
        delta = sla_deadline - issued_at
        assert delta.days <= 90

    def test_issue_car_returns_sla_details(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.sla_details is not None

    def test_issue_car_returns_provenance_hash(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == SHA256_HEX_LENGTH

    def test_issue_car_empty_nc_ids_raises(self, car_engine):
        with pytest.raises((ValueError, Exception)):
            car_engine.issue_car(IssueCARRequest(
                nc_ids=[],
                audit_id="AUD-TEST-001",
                supplier_id="SUP-001",
                issued_by="AUR-FSC-001",
            ))

    def test_issue_car_processing_time(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.processing_time_ms >= Decimal("0")


class TestCARLifecycleEndpoints:
    """Test CAR lifecycle status transition endpoint logic."""

    def test_car_acknowledge(self, car_engine, sample_car_critical):
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.ACKNOWLEDGED,
        )
        assert result.status == CARStatus.ACKNOWLEDGED

    def test_car_submit_rca(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.ACKNOWLEDGED
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.RCA_SUBMITTED,
        )
        assert result.status == CARStatus.RCA_SUBMITTED

    def test_invalid_transition_raises(self, car_engine, sample_car_critical):
        with pytest.raises((ValueError, Exception)):
            car_engine.transition_status(
                car=sample_car_critical,
                new_status=CARStatus.CLOSED,
            )

    @pytest.mark.parametrize("from_status,to_status", [
        (CARStatus.ISSUED, CARStatus.ACKNOWLEDGED),
        (CARStatus.ACKNOWLEDGED, CARStatus.RCA_SUBMITTED),
        (CARStatus.RCA_SUBMITTED, CARStatus.CAP_SUBMITTED),
        (CARStatus.CAP_SUBMITTED, CARStatus.CAP_APPROVED),
        (CARStatus.CAP_APPROVED, CARStatus.IN_PROGRESS),
        (CARStatus.IN_PROGRESS, CARStatus.EVIDENCE_SUBMITTED),
        (CARStatus.EVIDENCE_SUBMITTED, CARStatus.VERIFICATION_PENDING),
        (CARStatus.VERIFICATION_PENDING, CARStatus.CLOSED),
    ])
    def test_valid_car_transitions(self, car_engine, from_status, to_status):
        car = CorrectiveActionRequest(
            nc_ids=["NC-001"],
            audit_id="AUD-TEST-001",
            supplier_id="SUP-001",
            severity=NCSeverity.MAJOR,
            sla_deadline=FROZEN_NOW + timedelta(days=90),
            status=from_status,
            issued_by="AUR-FSC-001",
            issued_at=FROZEN_NOW,
        )
        result = car_engine.transition_status(car=car, new_status=to_status)
        assert result.status == to_status


class TestCARSLAMonitoring:
    """Test CAR SLA monitoring endpoints."""

    def test_overdue_car_detected(self, car_engine, sample_car_overdue):
        status = car_engine.check_sla_status(sample_car_overdue)
        assert status in ("overdue", "critical")

    def test_on_track_car_detected(self, car_engine, sample_car_critical):
        status = car_engine.check_sla_status(sample_car_critical)
        assert status in ("on_track", "warning")

    def test_batch_sla_check(self, car_engine, batch_cars):
        results = car_engine.batch_check_sla(batch_cars)
        assert isinstance(results, list)
        assert len(results) == len(batch_cars)
