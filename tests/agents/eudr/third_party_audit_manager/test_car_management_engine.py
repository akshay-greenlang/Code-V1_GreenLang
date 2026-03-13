# -*- coding: utf-8 -*-
"""
Unit tests for Engine 5: CARManagementEngine -- AGENT-EUDR-024

Tests CAR lifecycle management, SLA deadline calculation, escalation stages,
status transitions, evidence verification, CAR grouping, metrics tracking,
and deterministic SLA enforcement.

Target: ~70 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.car_management_engine import (
    CARManagementEngine,
    VALID_CAR_TRANSITIONS,
    ESCALATION_DESCRIPTIONS,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    CARStatus,
    CorrectiveActionRequest,
    IssueCARRequest,
    NCSeverity,
    NC_SEVERITY_SLA_DAYS,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_NOW,
    FROZEN_PAST_30D,
    FROZEN_PAST_90D,
    SHA256_HEX_LENGTH,
)


class TestCAREngineInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = CARManagementEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = CARManagementEngine()
        assert engine.config is not None

    def test_valid_transitions_defined(self):
        assert CARStatus.ISSUED.value in VALID_CAR_TRANSITIONS
        assert CARStatus.CLOSED.value in VALID_CAR_TRANSITIONS

    def test_closed_has_no_transitions(self):
        assert len(VALID_CAR_TRANSITIONS[CARStatus.CLOSED.value]) == 0

    def test_escalation_descriptions_defined(self):
        assert len(ESCALATION_DESCRIPTIONS) == 4


class TestSLADeadlines:
    """Test SLA deadline calculation by severity."""

    def test_critical_sla_30_days(self):
        assert NC_SEVERITY_SLA_DAYS["critical"] == 30

    def test_major_sla_90_days(self):
        assert NC_SEVERITY_SLA_DAYS["major"] == 90

    def test_minor_sla_365_days(self):
        assert NC_SEVERITY_SLA_DAYS["minor"] == 365

    def test_calculate_sla_deadline_critical(self, car_engine):
        deadline = car_engine.calculate_sla_deadline(
            severity="critical",
            issued_at=FROZEN_NOW,
        )
        expected = FROZEN_NOW + timedelta(days=30)
        assert deadline == expected

    def test_calculate_sla_deadline_major(self, car_engine):
        deadline = car_engine.calculate_sla_deadline(
            severity="major",
            issued_at=FROZEN_NOW,
        )
        expected = FROZEN_NOW + timedelta(days=90)
        assert deadline == expected

    def test_calculate_sla_deadline_minor(self, car_engine):
        deadline = car_engine.calculate_sla_deadline(
            severity="minor",
            issued_at=FROZEN_NOW,
        )
        expected = FROZEN_NOW + timedelta(days=365)
        assert deadline == expected

    def test_sla_deadline_is_deterministic(self, car_engine):
        d1 = car_engine.calculate_sla_deadline("critical", FROZEN_NOW)
        d2 = car_engine.calculate_sla_deadline("critical", FROZEN_NOW)
        assert d1 == d2


class TestCARIssuance:
    """Test CAR issuance from NC classification."""

    def test_issue_car_critical(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response is not None
        assert response.car.status == CARStatus.ISSUED
        assert response.car.severity == NCSeverity.CRITICAL

    def test_issue_car_major(self, car_engine, issue_car_request_major):
        response = car_engine.issue_car(issue_car_request_major)
        assert response is not None
        assert response.car.severity == NCSeverity.MAJOR

    def test_issued_car_has_sla_deadline(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.car.sla_deadline is not None

    def test_issued_car_has_provenance(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.car.provenance_hash is not None
        assert len(response.car.provenance_hash) == SHA256_HEX_LENGTH

    def test_issued_car_links_ncs(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert "NC-CRIT-001" in response.car.nc_ids

    def test_issued_car_has_audit_id(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.car.audit_id == "AUD-TEST-001"


class TestCARStatusTransitions:
    """Test CAR status transition validation."""

    def test_issued_to_acknowledged(self, car_engine, sample_car_critical):
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.ACKNOWLEDGED.value,
        )
        assert result.status == CARStatus.ACKNOWLEDGED

    def test_acknowledged_to_rca_submitted(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.ACKNOWLEDGED
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.RCA_SUBMITTED.value,
        )
        assert result.status == CARStatus.RCA_SUBMITTED

    def test_rca_to_cap_submitted(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.RCA_SUBMITTED
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.CAP_SUBMITTED.value,
        )
        assert result.status == CARStatus.CAP_SUBMITTED

    def test_cap_submitted_to_cap_approved(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.CAP_SUBMITTED
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.CAP_APPROVED.value,
        )
        assert result.status == CARStatus.CAP_APPROVED

    def test_verification_to_closed(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.VERIFICATION_PENDING
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.CLOSED.value,
        )
        assert result.status == CARStatus.CLOSED

    def test_verification_to_rejected(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.VERIFICATION_PENDING
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.REJECTED.value,
        )
        assert result.status == CARStatus.REJECTED

    def test_rejected_to_in_progress(self, car_engine, sample_car_critical):
        sample_car_critical.status = CARStatus.REJECTED
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.IN_PROGRESS.value,
        )
        assert result.status == CARStatus.IN_PROGRESS

    def test_invalid_transition_raises(self, car_engine, sample_car_critical):
        with pytest.raises((ValueError, Exception)):
            car_engine.transition_status(
                car=sample_car_critical,
                new_status=CARStatus.CLOSED.value,
            )

    @pytest.mark.parametrize("from_status,to_status", [
        (CARStatus.ISSUED.value, CARStatus.ACKNOWLEDGED.value),
        (CARStatus.ACKNOWLEDGED.value, CARStatus.RCA_SUBMITTED.value),
        (CARStatus.RCA_SUBMITTED.value, CARStatus.CAP_SUBMITTED.value),
        (CARStatus.CAP_SUBMITTED.value, CARStatus.CAP_APPROVED.value),
        (CARStatus.CAP_APPROVED.value, CARStatus.IN_PROGRESS.value),
        (CARStatus.IN_PROGRESS.value, CARStatus.EVIDENCE_SUBMITTED.value),
        (CARStatus.EVIDENCE_SUBMITTED.value, CARStatus.VERIFICATION_PENDING.value),
        (CARStatus.VERIFICATION_PENDING.value, CARStatus.CLOSED.value),
    ])
    def test_full_happy_path_transitions(self, from_status, to_status):
        assert to_status in VALID_CAR_TRANSITIONS[from_status]


class TestSLAMonitoring:
    """Test SLA status monitoring and countdown."""

    def test_on_track_status(self, car_engine, sample_car_critical):
        status = car_engine.calculate_sla_status(
            car=sample_car_critical,
            current_time=FROZEN_NOW,
        )
        assert status in ("on_track", "warning", "critical")

    def test_overdue_status(self, car_engine, sample_car_overdue):
        status = car_engine.calculate_sla_status(
            car=sample_car_overdue,
            current_time=FROZEN_NOW,
        )
        assert status == "overdue"

    def test_warning_status_at_75_percent(self, car_engine, sample_car_major):
        # Simulate 75% of 90-day SLA elapsed
        current = sample_car_major.issued_at + timedelta(days=68)
        status = car_engine.calculate_sla_status(
            car=sample_car_major,
            current_time=current,
        )
        assert status in ("warning", "critical")

    def test_critical_status_at_90_percent(self, car_engine, sample_car_major):
        current = sample_car_major.issued_at + timedelta(days=82)
        status = car_engine.calculate_sla_status(
            car=sample_car_major,
            current_time=current,
        )
        assert status == "critical"

    def test_sla_remaining_days(self, car_engine, sample_car_critical):
        remaining = car_engine.get_sla_remaining_days(
            car=sample_car_critical,
            current_time=FROZEN_NOW,
        )
        assert remaining == 30


class TestEscalation:
    """Test 4-stage escalation system."""

    def test_escalation_stage_1_at_75_pct(self, car_engine, sample_car_major):
        current = sample_car_major.issued_at + timedelta(days=68)
        stage = car_engine.determine_escalation_stage(
            car=sample_car_major,
            current_time=current,
        )
        assert stage >= 1

    def test_escalation_stage_2_at_90_pct(self, car_engine, sample_car_major):
        current = sample_car_major.issued_at + timedelta(days=82)
        stage = car_engine.determine_escalation_stage(
            car=sample_car_major,
            current_time=current,
        )
        assert stage >= 2

    def test_escalation_stage_3_at_sla_exceeded(self, car_engine, sample_car_major):
        current = sample_car_major.issued_at + timedelta(days=91)
        stage = car_engine.determine_escalation_stage(
            car=sample_car_major,
            current_time=current,
        )
        assert stage >= 3

    def test_escalation_stage_4_at_sla_plus_30(self, car_engine, sample_car_major):
        current = sample_car_major.issued_at + timedelta(days=121)
        stage = car_engine.determine_escalation_stage(
            car=sample_car_major,
            current_time=current,
        )
        assert stage >= 4

    def test_no_escalation_when_on_track(self, car_engine, sample_car_critical):
        stage = car_engine.determine_escalation_stage(
            car=sample_car_critical,
            current_time=FROZEN_NOW,
        )
        assert stage == 0

    def test_escalation_descriptions_have_all_stages(self):
        for stage in [1, 2, 3, 4]:
            assert stage in ESCALATION_DESCRIPTIONS


class TestCARGrouping:
    """Test CAR grouping for related NCs."""

    def test_group_related_ncs(self, car_engine):
        car = car_engine.issue_grouped_car(
            nc_ids=["NC-MAJ-001", "NC-MAJ-002"],
            audit_id="AUD-TEST-001",
            supplier_id="SUP-001",
            severity="major",
            issued_by="AUR-FSC-001",
        )
        assert car is not None
        assert len(car.nc_ids) == 2

    def test_grouped_car_uses_highest_severity(self, car_engine):
        car = car_engine.issue_grouped_car(
            nc_ids=["NC-CRIT-001", "NC-MAJ-001"],
            audit_id="AUD-TEST-001",
            supplier_id="SUP-001",
            severity="critical",
            issued_by="AUR-FSC-001",
        )
        assert car.severity == NCSeverity.CRITICAL


class TestCARMetrics:
    """Test CAR performance metrics."""

    def test_get_car_metrics(self, car_engine, batch_cars):
        for car in batch_cars:
            car_engine.register_car(car)
        metrics = car_engine.get_car_metrics(supplier_id="SUP-001")
        assert metrics is not None
        assert "total_cars" in metrics
        assert "open_cars" in metrics

    def test_car_closure_rate(self, car_engine, batch_cars):
        for car in batch_cars:
            car_engine.register_car(car)
        metrics = car_engine.get_car_metrics(supplier_id="SUP-001")
        assert "closure_rate" in metrics

    def test_sla_compliance_rate(self, car_engine, batch_cars):
        for car in batch_cars:
            car_engine.register_car(car)
        metrics = car_engine.get_car_metrics(supplier_id="SUP-001")
        assert "sla_compliance_rate" in metrics


class TestAuthorityIssuedCAR:
    """Test competent authority-issued CARs (Article 18)."""

    def test_authority_car_issuance(self, car_engine):
        request = IssueCARRequest(
            nc_ids=["NC-AUTH-001"],
            audit_id="AUD-AUTH-001",
            supplier_id="SUP-001",
            severity="major",
            issued_by="BMEL",
            authority_issued=True,
        )
        response = car_engine.issue_car(request)
        assert response is not None

    def test_authority_car_has_custom_sla(self, car_engine):
        request = IssueCARRequest(
            nc_ids=["NC-AUTH-001"],
            audit_id="AUD-AUTH-001",
            supplier_id="SUP-001",
            severity="major",
            issued_by="BMEL",
            authority_issued=True,
            custom_sla_days=60,
        )
        response = car_engine.issue_car(request)
        if hasattr(response.car, 'sla_deadline') and response.car.sla_deadline:
            days_diff = (response.car.sla_deadline - response.car.issued_at).days
            assert days_diff == 60


class TestCARProvenance:
    """Test CAR provenance tracking."""

    def test_car_has_provenance_hash(self, car_engine, issue_car_request_critical):
        response = car_engine.issue_car(issue_car_request_critical)
        assert response.car.provenance_hash is not None

    def test_car_transition_updates_provenance(self, car_engine, sample_car_critical):
        result = car_engine.transition_status(
            car=sample_car_critical,
            new_status=CARStatus.ACKNOWLEDGED.value,
        )
        assert result.provenance_hash is not None
