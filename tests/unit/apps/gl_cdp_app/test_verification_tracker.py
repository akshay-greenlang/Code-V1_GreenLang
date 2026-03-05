# -*- coding: utf-8 -*-
"""
Unit tests for VerificationTracker -- third-party verification management.

Tests verification record creation, coverage percentage calculation,
A-level verification requirements, assurance level tracking, and
verification schedule with 22+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from services.config import AssuranceLevel, VerificationScope
from services.models import (
    CDPVerificationRecord,
    _new_id,
)
from services.verification_tracker import VerificationTracker


# ---------------------------------------------------------------------------
# Verification record creation
# ---------------------------------------------------------------------------

class TestVerificationRecordCreation:
    """Test creating and managing verification records."""

    def test_create_verification_record(self, verification_tracker, sample_organization, sample_questionnaire):
        record = verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_1",
            coverage_pct=Decimal("100.0"),
            verifier_name="Big4 Audit LLP",
            verifier_accreditation="ISO 14065",
            assurance_level=AssuranceLevel.REASONABLE,
            verification_standard="ISO 14064-3:2019",
            statement_date=date(2025, 6, 15),
            valid_until=date(2026, 6, 14),
        )
        assert isinstance(record, CDPVerificationRecord)
        assert record.scope == "scope_1"
        assert record.coverage_pct == Decimal("100.0")

    def test_create_scope2_record(self, verification_tracker, sample_organization, sample_questionnaire):
        record = verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_2",
            coverage_pct=Decimal("100.0"),
            verifier_name="Audit Corp",
            assurance_level=AssuranceLevel.REASONABLE,
        )
        assert record.scope == "scope_2"

    def test_create_scope3_record(self, verification_tracker, sample_organization, sample_questionnaire):
        record = verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_3_cat1",
            coverage_pct=Decimal("75.0"),
            verifier_name="Verify Inc",
            assurance_level=AssuranceLevel.LIMITED,
        )
        assert record.scope == "scope_3_cat1"
        assert record.coverage_pct == Decimal("75.0")

    def test_get_records_by_org(self, verification_tracker, sample_organization, sample_questionnaire):
        for scope in ["scope_1", "scope_2", "scope_3_cat1"]:
            verification_tracker.create_record(
                org_id=sample_organization.id,
                questionnaire_id=sample_questionnaire.id,
                scope=scope,
                coverage_pct=Decimal("100.0"),
                verifier_name="Verifier",
                assurance_level=AssuranceLevel.LIMITED,
            )
        records = verification_tracker.get_records(org_id=sample_organization.id)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Coverage percentage calculation
# ---------------------------------------------------------------------------

class TestCoverageCalculation:
    """Test verification coverage percentage."""

    def test_scope1_2_coverage_100(self, verification_tracker, sample_organization, sample_questionnaire):
        for scope in ["scope_1", "scope_2"]:
            verification_tracker.create_record(
                org_id=sample_organization.id,
                questionnaire_id=sample_questionnaire.id,
                scope=scope,
                coverage_pct=Decimal("100.0"),
                verifier_name="Verifier",
                assurance_level=AssuranceLevel.REASONABLE,
            )
        coverage = verification_tracker.get_scope1_2_coverage(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
        )
        assert coverage == Decimal("100.0")

    def test_partial_scope1_coverage(self, verification_tracker, sample_organization, sample_questionnaire):
        verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_1",
            coverage_pct=Decimal("80.0"),
            verifier_name="Verifier",
            assurance_level=AssuranceLevel.LIMITED,
        )
        coverage = verification_tracker.get_scope_coverage(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_1",
        )
        assert coverage == Decimal("80.0")

    def test_no_verification_zero_coverage(self, verification_tracker, sample_organization, sample_questionnaire):
        coverage = verification_tracker.get_scope1_2_coverage(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
        )
        assert coverage == Decimal("0")

    def test_scope3_coverage_by_category(self, verification_tracker, sample_organization, sample_questionnaire):
        verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_3_cat1",
            coverage_pct=Decimal("75.0"),
            verifier_name="Verifier",
            assurance_level=AssuranceLevel.LIMITED,
        )
        coverage = verification_tracker.get_scope3_category_coverage(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            category="cat1",
        )
        assert coverage == Decimal("75.0")


# ---------------------------------------------------------------------------
# A-level verification requirements
# ---------------------------------------------------------------------------

class TestALevelRequirements:
    """Test A-level verification requirements checker."""

    def test_a_level_fully_met(self, verification_tracker, sample_organization, sample_questionnaire):
        # 100% Scope 1 + 2 verified
        for scope in ["scope_1", "scope_2"]:
            verification_tracker.create_record(
                org_id=sample_organization.id,
                questionnaire_id=sample_questionnaire.id,
                scope=scope,
                coverage_pct=Decimal("100.0"),
                verifier_name="Big4",
                assurance_level=AssuranceLevel.REASONABLE,
            )
        # >= 70% Scope 3 category
        verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_3_cat1",
            coverage_pct=Decimal("80.0"),
            verifier_name="Big4",
            assurance_level=AssuranceLevel.LIMITED,
        )
        result = verification_tracker.check_a_level_requirements(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
        )
        assert result["scope1_2_met"] is True
        assert result["scope3_met"] is True

    def test_a_level_missing_scope3(self, verification_tracker, sample_organization, sample_questionnaire):
        for scope in ["scope_1", "scope_2"]:
            verification_tracker.create_record(
                org_id=sample_organization.id,
                questionnaire_id=sample_questionnaire.id,
                scope=scope,
                coverage_pct=Decimal("100.0"),
                verifier_name="Big4",
                assurance_level=AssuranceLevel.REASONABLE,
            )
        result = verification_tracker.check_a_level_requirements(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
        )
        assert result["scope1_2_met"] is True
        assert result["scope3_met"] is False

    def test_a_level_insufficient_scope1(self, verification_tracker, sample_organization, sample_questionnaire):
        verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_1",
            coverage_pct=Decimal("80.0"),
            verifier_name="Big4",
            assurance_level=AssuranceLevel.REASONABLE,
        )
        result = verification_tracker.check_a_level_requirements(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
        )
        assert result["scope1_2_met"] is False


# ---------------------------------------------------------------------------
# Assurance level tracking
# ---------------------------------------------------------------------------

class TestAssuranceLevel:
    """Test assurance level tracking."""

    def test_reasonable_assurance(self, verification_tracker, sample_verification_record):
        verification_tracker._store[sample_verification_record.id] = sample_verification_record
        assert sample_verification_record.assurance_level == AssuranceLevel.REASONABLE

    def test_limited_assurance(self, verification_tracker, sample_organization, sample_questionnaire):
        record = verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_3_cat6",
            coverage_pct=Decimal("70.0"),
            verifier_name="Auditor",
            assurance_level=AssuranceLevel.LIMITED,
        )
        assert record.assurance_level == AssuranceLevel.LIMITED


# ---------------------------------------------------------------------------
# Verification schedule
# ---------------------------------------------------------------------------

class TestVerificationSchedule:
    """Test verification schedule management."""

    def test_check_expiry(self, verification_tracker, sample_verification_record):
        verification_tracker._store[sample_verification_record.id] = sample_verification_record
        is_valid = verification_tracker.is_verification_valid(
            record_id=sample_verification_record.id,
            as_of=date(2025, 12, 1),
        )
        assert is_valid is True

    def test_expired_verification(self, verification_tracker, sample_verification_record):
        verification_tracker._store[sample_verification_record.id] = sample_verification_record
        is_valid = verification_tracker.is_verification_valid(
            record_id=sample_verification_record.id,
            as_of=date(2026, 7, 1),  # After valid_until
        )
        assert is_valid is False

    def test_upcoming_expirations(self, verification_tracker, sample_organization, sample_questionnaire):
        record = verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_1",
            coverage_pct=Decimal("100.0"),
            verifier_name="Verifier",
            assurance_level=AssuranceLevel.REASONABLE,
            valid_until=date.today() + timedelta(days=30),
        )
        expiring = verification_tracker.get_upcoming_expirations(
            org_id=sample_organization.id,
            within_days=60,
        )
        assert len(expiring) >= 1

    def test_no_upcoming_expirations(self, verification_tracker, sample_organization, sample_questionnaire):
        record = verification_tracker.create_record(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
            scope="scope_1",
            coverage_pct=Decimal("100.0"),
            verifier_name="Verifier",
            assurance_level=AssuranceLevel.REASONABLE,
            valid_until=date.today() + timedelta(days=365),
        )
        expiring = verification_tracker.get_upcoming_expirations(
            org_id=sample_organization.id,
            within_days=30,
        )
        assert len(expiring) == 0


# ---------------------------------------------------------------------------
# Verification summary
# ---------------------------------------------------------------------------

class TestVerificationSummary:
    """Test verification summary and completeness reporting."""

    def test_verification_summary(self, verification_tracker, sample_organization, sample_questionnaire):
        for scope in ["scope_1", "scope_2"]:
            verification_tracker.create_record(
                org_id=sample_organization.id,
                questionnaire_id=sample_questionnaire.id,
                scope=scope,
                coverage_pct=Decimal("100.0"),
                verifier_name="Verifier Corp",
                assurance_level=AssuranceLevel.REASONABLE,
            )
        summary = verification_tracker.get_verification_summary(
            org_id=sample_organization.id,
            questionnaire_id=sample_questionnaire.id,
        )
        assert summary["total_records"] == 2
        assert summary["scopes_covered"] >= 2

    def test_update_record(self, verification_tracker, sample_verification_record):
        verification_tracker._store[sample_verification_record.id] = sample_verification_record
        updated = verification_tracker.update_record(
            record_id=sample_verification_record.id,
            coverage_pct=Decimal("95.0"),
        )
        assert updated.coverage_pct == Decimal("95.0")

    def test_delete_record(self, verification_tracker, sample_verification_record):
        verification_tracker._store[sample_verification_record.id] = sample_verification_record
        verification_tracker.delete_record(sample_verification_record.id)
        assert verification_tracker.get_record(sample_verification_record.id) is None

    def test_get_record_by_id(self, verification_tracker, sample_verification_record):
        verification_tracker._store[sample_verification_record.id] = sample_verification_record
        record = verification_tracker.get_record(sample_verification_record.id)
        assert record is not None
        assert record.id == sample_verification_record.id

    def test_nonexistent_record_returns_none(self, verification_tracker):
        result = verification_tracker.get_record("nonexistent-id")
        assert result is None
