# -*- coding: utf-8 -*-
"""
Unit tests for Engine 3: AuditExecutionEngine -- AGENT-EUDR-024

Tests audit execution workflows, checklist management, evidence collection,
sampling plan generation, status transitions, progress tracking, multi-site
coordination, and evidence integrity verification.

Target: ~70 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.audit_execution_engine import (
    AuditExecutionEngine,
    VALID_STATUS_TRANSITIONS,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    Audit,
    AuditChecklist,
    AuditEvidence,
    AuditModality,
    AuditScope,
    AuditStatus,
    CertificationScheme,
    SUPPORTED_COMMODITIES,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
)


class TestExecutionEngineInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = AuditExecutionEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = AuditExecutionEngine()
        assert engine.config is not None


class TestStatusTransitions:
    """Test valid audit status transitions."""

    def test_planned_can_transition_to_auditor_assigned(self):
        assert AuditStatus.AUDITOR_ASSIGNED.value in VALID_STATUS_TRANSITIONS[AuditStatus.PLANNED.value]

    def test_planned_can_be_cancelled(self):
        assert AuditStatus.CANCELLED.value in VALID_STATUS_TRANSITIONS[AuditStatus.PLANNED.value]

    def test_auditor_assigned_can_transition_to_preparation(self):
        assert AuditStatus.IN_PREPARATION.value in VALID_STATUS_TRANSITIONS[AuditStatus.AUDITOR_ASSIGNED.value]

    def test_closed_has_no_transitions(self):
        assert len(VALID_STATUS_TRANSITIONS.get(AuditStatus.CLOSED.value, [])) == 0

    def test_advance_status_valid(self, execution_engine, sample_audit):
        result = execution_engine.advance_status(
            sample_audit, AuditStatus.AUDITOR_ASSIGNED.value
        )
        assert result.status == AuditStatus.AUDITOR_ASSIGNED

    def test_advance_status_invalid_raises(self, execution_engine, sample_audit):
        with pytest.raises((ValueError, Exception)):
            execution_engine.advance_status(
                sample_audit, AuditStatus.CLOSED.value
            )

    @pytest.mark.parametrize("from_status,to_status", [
        (AuditStatus.PLANNED.value, AuditStatus.AUDITOR_ASSIGNED.value),
        (AuditStatus.AUDITOR_ASSIGNED.value, AuditStatus.IN_PREPARATION.value),
    ])
    def test_valid_transition_pairs(self, from_status, to_status):
        assert to_status in VALID_STATUS_TRANSITIONS[from_status]

    def test_all_statuses_have_transition_rules(self):
        for status in AuditStatus:
            if status != AuditStatus.CLOSED and status != AuditStatus.CANCELLED:
                assert status.value in VALID_STATUS_TRANSITIONS


class TestChecklistManagement:
    """Test audit checklist creation and management."""

    def test_create_eudr_checklist(self, execution_engine, sample_audit):
        checklist = execution_engine.create_checklist(
            audit_id=sample_audit.audit_id,
            checklist_type="eudr",
        )
        assert checklist is not None
        assert checklist.checklist_type == "eudr"
        assert checklist.total_criteria > 0

    @pytest.mark.parametrize("scheme", ["fsc", "pefc", "rspo", "rainforest_alliance", "iscc"])
    def test_create_scheme_checklist(self, execution_engine, sample_audit, scheme):
        checklist = execution_engine.create_checklist(
            audit_id=sample_audit.audit_id,
            checklist_type=scheme,
        )
        assert checklist is not None
        assert checklist.checklist_type == scheme
        assert checklist.total_criteria > 0

    def test_checklist_has_version(self, execution_engine, sample_audit):
        checklist = execution_engine.create_checklist(
            audit_id=sample_audit.audit_id,
            checklist_type="eudr",
        )
        assert checklist.checklist_version is not None

    def test_checklist_completion_starts_at_zero(self, execution_engine, sample_audit):
        checklist = execution_engine.create_checklist(
            audit_id=sample_audit.audit_id,
            checklist_type="eudr",
        )
        assert checklist.completion_percentage == Decimal("0")

    def test_update_criterion_pass(self, execution_engine, sample_checklist_eudr):
        updated = execution_engine.update_criterion(
            checklist=sample_checklist_eudr,
            criterion_id="EUDR-ART3-001",
            result="pass",
            auditor_notes="Verified through satellite imagery",
        )
        assert updated is not None

    def test_update_criterion_fail(self, execution_engine, sample_checklist_eudr):
        updated = execution_engine.update_criterion(
            checklist=sample_checklist_eudr,
            criterion_id="EUDR-ART9-GEO-001",
            result="fail",
            auditor_notes="Geolocation data missing for 3 plots",
        )
        assert updated is not None

    def test_update_criterion_na(self, execution_engine, sample_checklist_eudr):
        updated = execution_engine.update_criterion(
            checklist=sample_checklist_eudr,
            criterion_id="EUDR-ART10-RA-001",
            result="na",
            auditor_notes="Not applicable for this commodity",
        )
        assert updated is not None

    def test_calculate_completion_percentage(self, execution_engine, sample_checklist_eudr):
        pct = execution_engine.calculate_completion(sample_checklist_eudr)
        assert Decimal("0") <= pct <= Decimal("100")

    def test_checklist_provenance_hash(self, execution_engine, sample_audit):
        checklist = execution_engine.create_checklist(
            audit_id=sample_audit.audit_id,
            checklist_type="eudr",
        )
        assert checklist.provenance_hash is not None
        assert len(checklist.provenance_hash) == SHA256_HEX_LENGTH


class TestEvidenceCollection:
    """Test evidence collection and integrity verification."""

    def test_register_evidence(self, execution_engine, sample_evidence_document):
        result = execution_engine.register_evidence(sample_evidence_document)
        assert result is not None
        assert result.evidence_id == "EV-DOC-001"

    def test_evidence_has_sha256_hash(self, sample_evidence_document):
        assert sample_evidence_document.sha256_hash is not None
        assert len(sample_evidence_document.sha256_hash) == SHA256_HEX_LENGTH

    def test_evidence_type_classification(self, sample_evidence_document, sample_evidence_photo):
        assert sample_evidence_document.evidence_type == "permit"
        assert sample_evidence_photo.evidence_type == "photo"

    def test_evidence_metadata_tags(self, sample_evidence_document):
        assert "country" in sample_evidence_document.tags
        assert sample_evidence_document.tags["country"] == "BR"

    def test_evidence_file_size_tracked(self, sample_evidence_document):
        assert sample_evidence_document.file_size_bytes > 0

    def test_evidence_mime_type_tracked(self, sample_evidence_document):
        assert sample_evidence_document.mime_type == "application/pdf"

    def test_evidence_linked_to_audit(self, sample_evidence_document):
        assert sample_evidence_document.audit_id == "AUD-TEST-001"

    @pytest.mark.parametrize("evidence_type", [
        "permit", "certificate", "photo", "gps_record",
        "interview_transcript", "lab_result", "document",
    ])
    def test_supported_evidence_types(self, execution_engine, evidence_type):
        evidence = AuditEvidence(
            audit_id="AUD-TEST-001",
            evidence_type=evidence_type,
            file_name=f"test_{evidence_type}.pdf",
            file_size_bytes=1000,
        )
        result = execution_engine.register_evidence(evidence)
        assert result is not None

    def test_evidence_file_size_validation(self, execution_engine):
        max_size = execution_engine.config.max_evidence_file_size_bytes
        oversized = AuditEvidence(
            audit_id="AUD-TEST-001",
            evidence_type="document",
            file_name="oversized.pdf",
            file_size_bytes=max_size + 1,
        )
        with pytest.raises((ValueError, Exception)):
            execution_engine.register_evidence(oversized)

    def test_verify_evidence_integrity(self, execution_engine, sample_evidence_document):
        execution_engine.register_evidence(sample_evidence_document)
        is_valid = execution_engine.verify_evidence_integrity(
            evidence_id="EV-DOC-001",
            expected_hash=sample_evidence_document.sha256_hash,
        )
        assert is_valid is True


class TestSamplingPlan:
    """Test ISO 19011 Annex A sampling plan generation."""

    def test_calculate_sample_size_basic(self, execution_engine):
        size = execution_engine.calculate_sample_size(
            population_size=100,
            risk_level="standard",
        )
        assert size > 0
        assert size <= 100

    @pytest.mark.parametrize("risk_level", ["high", "standard", "low"])
    def test_sample_size_for_risk_levels(self, execution_engine, risk_level):
        size = execution_engine.calculate_sample_size(
            population_size=100,
            risk_level=risk_level,
        )
        assert size > 0

    def test_high_risk_has_larger_sample(self, execution_engine):
        high = execution_engine.calculate_sample_size(100, "high")
        low = execution_engine.calculate_sample_size(100, "low")
        assert high >= low

    def test_larger_population_larger_sample(self, execution_engine):
        small = execution_engine.calculate_sample_size(10, "standard")
        large = execution_engine.calculate_sample_size(1000, "standard")
        assert large >= small

    def test_sample_size_at_least_one(self, execution_engine):
        size = execution_engine.calculate_sample_size(1, "low")
        assert size >= 1

    def test_sample_size_deterministic(self, execution_engine):
        s1 = execution_engine.calculate_sample_size(200, "high")
        s2 = execution_engine.calculate_sample_size(200, "high")
        assert s1 == s2

    def test_generate_sampling_plan(self, execution_engine, sample_audit):
        plan = execution_engine.generate_sampling_plan(
            audit_id=sample_audit.audit_id,
            population_size=500,
            risk_level="high",
        )
        assert plan is not None
        assert plan["sample_size"] > 0
        assert plan["methodology"] in ["statistical", "judgmental"]


class TestAuditProgress:
    """Test audit progress tracking."""

    def test_calculate_progress_empty_checklist(self, execution_engine):
        checklist = AuditChecklist(
            audit_id="AUD-TEST-001",
            total_criteria=0,
        )
        pct = execution_engine.calculate_completion(checklist)
        assert pct == Decimal("0")

    def test_calculate_progress_partial(self, execution_engine, sample_checklist_eudr):
        pct = execution_engine.calculate_completion(sample_checklist_eudr)
        assert Decimal("0") < pct < Decimal("100")

    def test_calculate_progress_complete(self, execution_engine, sample_checklist_fsc):
        pct = execution_engine.calculate_completion(sample_checklist_fsc)
        assert pct == Decimal("100")

    def test_get_audit_progress_summary(self, execution_engine, sample_audit_in_progress):
        summary = execution_engine.get_progress_summary(sample_audit_in_progress)
        assert summary is not None
        assert "completion_percentage" in summary
        assert "status" in summary


class TestAuditModality:
    """Test audit modality support."""

    @pytest.mark.parametrize("modality", [
        AuditModality.ON_SITE,
        AuditModality.REMOTE,
        AuditModality.HYBRID,
        AuditModality.UNANNOUNCED,
    ])
    def test_all_modalities_supported(self, modality):
        audit = Audit(
            operator_id="OP-001",
            supplier_id="SUP-001",
            modality=modality,
            planned_date=FROZEN_DATE,
            country_code="BR",
            commodity="wood",
        )
        assert audit.modality == modality

    def test_hold_audit(self, execution_engine, sample_audit_in_progress):
        result = execution_engine.hold_audit(
            audit=sample_audit_in_progress,
            reason="Critical issue discovered requiring immediate escalation",
        )
        assert result is not None

    def test_suspend_audit(self, execution_engine, sample_audit_in_progress):
        result = execution_engine.suspend_audit(
            audit=sample_audit_in_progress,
            reason="Unsafe field conditions",
        )
        assert result is not None
