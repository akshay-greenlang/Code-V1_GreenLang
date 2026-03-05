# -*- coding: utf-8 -*-
"""
Unit tests for QualityManager -- ISO 14064-1:2018 Clause 7.

Tests quality plan CRUD, procedures, data quality assessment,
corrective actions, calibration records, document version control,
and quality summary with 20+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import DataQualityTier
from services.quality_management import QualityManager


class TestPlanCRUD:
    """Test quality management plan lifecycle."""

    def test_create_plan(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        assert plan.org_id == "org-1"
        assert len(plan.id) == 36

    def test_get_plan(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        retrieved = quality_manager.get_plan(plan.id)
        assert retrieved is not None
        assert retrieved.id == plan.id

    def test_get_nonexistent_plan(self, quality_manager):
        assert quality_manager.get_plan("nonexistent") is None

    def test_get_plans_for_org(self, quality_manager):
        quality_manager.create_plan("org-1")
        quality_manager.create_plan("org-1")
        quality_manager.create_plan("org-2")
        plans = quality_manager.get_plans_for_org("org-1")
        assert len(plans) == 2

    def test_delete_plan(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        result = quality_manager.delete_plan(plan.id)
        assert result is True
        assert quality_manager.get_plan(plan.id) is None

    def test_delete_nonexistent_plan(self, quality_manager):
        result = quality_manager.delete_plan("bad-id")
        assert result is False


class TestProcedures:
    """Test procedure management within plans."""

    def test_add_procedure(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        proc = quality_manager.add_procedure(
            plan.id, "Data Collection Protocol",
            procedure_type="data_collection",
            description="Protocol for collecting activity data",
            responsible="GHG Manager",
            frequency="monthly",
        )
        assert proc.title == "Data Collection Protocol"
        assert proc.procedure_type == "data_collection"

    def test_get_procedures(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.add_procedure(plan.id, "Proc A", procedure_type="data_collection")
        quality_manager.add_procedure(plan.id, "Proc B", procedure_type="review")
        all_procs = quality_manager.get_procedures(plan.id)
        assert len(all_procs) == 2

    def test_get_procedures_filtered_by_type(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.add_procedure(plan.id, "A", procedure_type="data_collection")
        quality_manager.add_procedure(plan.id, "B", procedure_type="review")
        filtered = quality_manager.get_procedures(plan.id, procedure_type="review")
        assert len(filtered) == 1
        assert filtered[0].procedure_type == "review"

    def test_add_procedure_invalid_plan_raises(self, quality_manager):
        with pytest.raises(ValueError, match="not found"):
            quality_manager.add_procedure("bad-id", "X")


class TestDataQualityAssessment:
    """Test per-source data quality assessment."""

    def test_high_quality_assessment(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        result = quality_manager.assess_data_quality(
            plan.id, "Boiler NG",
            activity_data_quality=95,
            emission_factor_quality=90,
        )
        # Composite: 0.60*95 + 0.40*90 = 57 + 36 = 93
        assert Decimal(result["composite_quality"]) == Decimal("93.0")
        assert result["tier"] == DataQualityTier.TIER_4.value

    def test_medium_quality_assessment(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        result = quality_manager.assess_data_quality(
            plan.id, "Fleet Diesel",
            activity_data_quality=80,
            emission_factor_quality=70,
        )
        # Composite: 0.60*80 + 0.40*70 = 48 + 28 = 76
        assert Decimal(result["composite_quality"]) == Decimal("76.0")
        assert result["tier"] == DataQualityTier.TIER_3.value

    def test_low_quality_tier_1(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        result = quality_manager.assess_data_quality(
            plan.id, "Estimated Source",
            activity_data_quality=30,
            emission_factor_quality=20,
        )
        # Composite: 0.60*30 + 0.40*20 = 18 + 8 = 26
        assert result["tier"] == DataQualityTier.TIER_1.value

    def test_overall_quality_score(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.assess_data_quality(plan.id, "Src A", 90, 90)
        quality_manager.assess_data_quality(plan.id, "Src B", 80, 80)
        overall = quality_manager.get_overall_quality_score(plan.id)
        # (90.0 + 80.0) / 2 = 85.0
        assert overall == Decimal("85.0")

    def test_overall_score_empty(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        assert quality_manager.get_overall_quality_score(plan.id) == Decimal("0")


class TestCorrectiveActions:
    """Test corrective action lifecycle."""

    def test_add_corrective_action(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        ca = quality_manager.add_corrective_action(
            plan.id,
            finding="Missing emission factor documentation",
            action="Update EF registry with source references",
            responsible="EF Manager",
        )
        assert ca["status"] == "open"
        assert ca["finding"] == "Missing emission factor documentation"

    def test_resolve_corrective_action(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        ca = quality_manager.add_corrective_action(
            plan.id, "Finding", "Action",
        )
        resolved = quality_manager.resolve_corrective_action(
            plan.id, ca["id"], "EF registry updated",
        )
        assert resolved["status"] == "resolved"
        assert resolved["resolution"] == "EF registry updated"

    def test_resolve_nonexistent_action_raises(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        with pytest.raises(ValueError, match="not found"):
            quality_manager.resolve_corrective_action(
                plan.id, "bad-id", "Resolution",
            )

    def test_filter_open_actions(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.add_corrective_action(plan.id, "F1", "A1")
        ca2 = quality_manager.add_corrective_action(plan.id, "F2", "A2")
        quality_manager.resolve_corrective_action(plan.id, ca2["id"], "Done")
        open_actions = quality_manager.get_corrective_actions(plan.id, status="open")
        assert len(open_actions) == 1


class TestCalibrationRecords:
    """Test calibration record management."""

    def test_add_calibration_record(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        record = quality_manager.add_calibration_record(
            plan.id,
            equipment_name="CEMS Analyzer",
            calibration_date="2025-01-15",
            next_calibration_date="2026-01-15",
            certificate_ref="CAL-2025-001",
        )
        assert record["equipment_name"] == "CEMS Analyzer"
        assert record["status"] == "valid"

    def test_get_calibration_records(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.add_calibration_record(plan.id, "Meter A", "2025-01-01")
        quality_manager.add_calibration_record(plan.id, "Meter B", "2025-06-01")
        records = quality_manager.get_calibration_records(plan.id)
        assert len(records) == 2


class TestDocumentVersionControl:
    """Test document version management."""

    def test_add_document_version(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        doc = quality_manager.add_document_version(
            plan.id, "EF Registry", "emission_factors",
            version="1.0", effective_date="2025-01-01",
        )
        assert doc["document_name"] == "EF Registry"
        assert doc["version"] == "1.0"
        assert doc["superseded_date"] is None

    def test_new_version_supersedes_previous(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.add_document_version(
            plan.id, "EF Registry", "emission_factors",
            version="1.0", effective_date="2025-01-01",
        )
        quality_manager.add_document_version(
            plan.id, "EF Registry", "emission_factors",
            version="2.0", effective_date="2025-07-01",
        )
        all_docs = quality_manager.get_document_versions(plan.id)
        current = quality_manager.get_document_versions(plan.id, current_only=True)
        assert len(all_docs) == 2
        assert len(current) == 1
        assert current[0]["version"] == "2.0"


class TestQualitySummary:
    """Test comprehensive quality summary."""

    def test_generate_summary(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        quality_manager.add_procedure(plan.id, "P1")
        quality_manager.assess_data_quality(plan.id, "S1", 80, 80)
        quality_manager.add_corrective_action(plan.id, "F1", "A1")
        quality_manager.add_calibration_record(plan.id, "M1", "2025-01-01")
        quality_manager.add_document_version(
            plan.id, "Doc1", "procedure", "1.0", "2025-01-01",
        )

        summary = quality_manager.generate_quality_summary(plan.id)
        assert summary["procedures_count"] == 1
        assert summary["quality_assessments"] == 1
        assert summary["corrective_actions"]["total"] == 1
        assert summary["corrective_actions"]["open"] == 1
        assert summary["calibration_records"] == 1
        assert summary["document_versions"]["total"] == 1

    def test_summary_for_missing_plan(self, quality_manager):
        summary = quality_manager.generate_quality_summary("bad-id")
        assert summary["message"] == "No quality plan found"


class TestAuditSchedule:
    """Test audit schedule management."""

    def test_set_audit_schedule(self, quality_manager):
        plan = quality_manager.create_plan("org-1")
        updated = quality_manager.set_audit_schedule(
            plan.id, "Quarterly internal audits",
        )
        assert updated.internal_audit_schedule == "Quarterly internal audits"
