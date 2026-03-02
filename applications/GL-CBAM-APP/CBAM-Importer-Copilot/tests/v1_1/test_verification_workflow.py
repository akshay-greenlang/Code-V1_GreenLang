# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Verification Workflow

Tests VerifierRegistryEngine:
- Initialization
- Register verifier (basic, with NAB credentials, expiry)
- Verifier expertise sectors
- Conflict of interest checks (clean, flagged)
- Search verifiers by sector / country
- Accreditation validity and expiry

Tests VerificationSchedulerEngine:
- Initialization
- Schedule annual/biennial visits
- Remote audit scheduling
- Assign verifier
- Visit outcome (pass, fail, conditional)
- Upcoming visits

Tests MaterialityAssessorEngine:
- Initialization
- 5% materiality threshold (below, at, above)
- Materiality by CN code
- Recommended verification scope
- Historical materiality trend
- Direct plus indirect materiality
- Provenance hash

Target: 80+ tests
"""

import pytest
import hashlib
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from verification_workflow.verifier_registry import (
    VerifierRegistryEngine,
    Verifier,
    AccreditationRecord,
    VerifierPerformance,
    ConflictOfInterestCheck,
    VerifierStatus,
    COIResult,
)
from verification_workflow.verification_scheduler import (
    VerificationSchedulerEngine,
    SiteVisit,
    VisitType,
    VisitStatus,
    VisitOutcome,
    VisitFinding,
    FindingSeverity,
)
from verification_workflow.materiality_assessor import (
    MaterialityAssessorEngine,
    MaterialityResult,
    CnCodeMateriality,
    MaterialityTrend,
    VerificationScope,
    MATERIALITY_THRESHOLD_PCT,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def registry():
    """Create a fresh VerifierRegistryEngine."""
    return VerifierRegistryEngine()


@pytest.fixture
def sample_verifier():
    """Create a sample Verifier model."""
    return Verifier(
        company_name="EcoVerify GmbH",
        contact_email="info@ecoverify.de",
        nab_country="DE",
        accreditation_number="DAkkS-D-PL-20145-01",
        accredited_until=date.today() + timedelta(days=365),
        sector_expertise=["iron_steel", "aluminium"],
    )


@pytest.fixture
def registered_verifier(registry, sample_verifier):
    """Register and return a verifier."""
    return registry.register_verifier(sample_verifier)


@pytest.fixture
def second_verifier(registry):
    """Register a second verifier for multi-verifier tests."""
    v = Verifier(
        company_name="GreenAudit NV",
        contact_email="audit@greenaudit.be",
        nab_country="BE",
        accreditation_number="BELAC-2026-001",
        accredited_until=date.today() + timedelta(days=400),
        sector_expertise=["cement", "fertilisers"],
    )
    return registry.register_verifier(v)


@pytest.fixture
def scheduler(registry):
    """Create a VerificationSchedulerEngine backed by the registry."""
    return VerificationSchedulerEngine(registry)


@pytest.fixture
def assessor():
    """Create a fresh MaterialityAssessorEngine."""
    return MaterialityAssessorEngine()


# ===========================================================================
# TEST CLASS -- VerifierRegistryEngine initialization
# ===========================================================================

class TestVerifierRegistryInit:
    """Tests for VerifierRegistryEngine initialization."""

    def test_init(self, registry):
        assert registry is not None
        assert registry.get_all_verifiers() == []


# ===========================================================================
# TEST CLASS -- Register verifier
# ===========================================================================

class TestRegisterVerifier:
    """Tests for register_verifier."""

    def test_register_basic(self, registry, sample_verifier):
        v = registry.register_verifier(sample_verifier)
        assert v.company_name == "EcoVerify GmbH"
        assert v.status == VerifierStatus.ACTIVE
        assert len(v.provenance_hash) == 64

    def test_register_with_nab_credentials(self, registered_verifier):
        assert registered_verifier.nab_country == "DE"
        assert registered_verifier.accreditation_number == "DAkkS-D-PL-20145-01"
        assert len(registered_verifier.accreditation_records) >= 1

    def test_register_creates_accreditation_record(self, registered_verifier):
        records = registered_verifier.accreditation_records
        assert len(records) >= 1
        assert records[0].nab_country == "DE"
        assert records[0].accreditation_number == "DAkkS-D-PL-20145-01"

    def test_register_expired_raises(self, registry):
        v = Verifier(
            company_name="Expired Corp",
            nab_country="FR",
            accreditation_number="COFRAC-EXP-001",
            accredited_until=date(2020, 1, 1),
            sector_expertise=["cement"],
        )
        with pytest.raises(ValueError, match="expired"):
            registry.register_verifier(v)

    def test_register_sets_active_status(self, registered_verifier):
        assert registered_verifier.status == VerifierStatus.ACTIVE

    def test_get_verifier_by_id(self, registry, registered_verifier):
        v = registry.get_verifier(registered_verifier.verifier_id)
        assert v.company_name == "EcoVerify GmbH"

    def test_get_verifier_not_found(self, registry):
        with pytest.raises(KeyError):
            registry.get_verifier("NONEXISTENT-ID")


# ===========================================================================
# TEST CLASS -- Accreditation expiry
# ===========================================================================

class TestAccreditationExpiry:
    """Tests for accreditation validity and expiry."""

    def test_valid_accreditation(self, registry, registered_verifier):
        assert registry.check_accreditation_validity(
            registered_verifier.verifier_id
        ) is True

    def test_expired_accreditation(self, registry):
        v = Verifier(
            company_name="Soon-to-expire",
            nab_country="NL",
            accreditation_number="RvA-EXP-001",
            accredited_until=date.today() + timedelta(days=1),
            sector_expertise=["iron_steel"],
        )
        registered = registry.register_verifier(v)
        # Manually set to past for test
        registered.accredited_until = date.today() - timedelta(days=1)
        valid = registry.check_accreditation_validity(registered.verifier_id)
        assert valid is False
        assert registered.status == VerifierStatus.EXPIRED

    def test_get_expiring_accreditations(self, registry, registered_verifier):
        expiring = registry.get_expiring_accreditations(days_ahead=400)
        assert len(expiring) >= 1

    def test_update_accreditation(self, registry, registered_verifier):
        new_expiry = date.today() + timedelta(days=730)
        updated = registry.update_accreditation(
            verifier_id=registered_verifier.verifier_id,
            nab_country="DE",
            accreditation_number="DAkkS-D-PL-20145-02",
            expiry_date=new_expiry,
        )
        assert updated.accredited_until == new_expiry
        assert updated.accreditation_number == "DAkkS-D-PL-20145-02"
        assert len(updated.accreditation_records) >= 2


# ===========================================================================
# TEST CLASS -- Verifier expertise sectors
# ===========================================================================

class TestVerifierSectors:
    """Tests for verifier sector expertise."""

    def test_sector_expertise_set(self, registered_verifier):
        assert "iron_steel" in registered_verifier.sector_expertise
        assert "aluminium" in registered_verifier.sector_expertise

    def test_multiple_sectors(self, second_verifier):
        assert "cement" in second_verifier.sector_expertise
        assert "fertilisers" in second_verifier.sector_expertise


# ===========================================================================
# TEST CLASS -- Conflict of interest
# ===========================================================================

class TestConflictOfInterest:
    """Tests for conflict of interest checks."""

    def test_coi_clean(self, registry, registered_verifier):
        result = registry.check_conflict_of_interest(
            registered_verifier.verifier_id, "INST-001"
        )
        assert result.result == COIResult.CLEAR
        assert len(result.reasons) == 0

    def test_coi_flagged_consulting(self, registry):
        v = Verifier(
            company_name="ConflictCo",
            nab_country="NL",
            accreditation_number="RvA-COI-001",
            accredited_until=date.today() + timedelta(days=365),
            sector_expertise=["iron_steel"],
            consulting_clients=["INST-CONFLICT"],
        )
        registered = registry.register_verifier(v)
        result = registry.check_conflict_of_interest(
            registered.verifier_id, "INST-CONFLICT"
        )
        assert result.result == COIResult.CONFLICT_DETECTED
        assert len(result.reasons) >= 1
        assert result.cooling_off_until is not None

    def test_coi_consecutive_reviews(self, registry, registered_verifier):
        vid = registered_verifier.verifier_id
        # Record 3 visit outcomes for same installation
        for _ in range(3):
            registry.record_visit_outcome(
                verifier_id=vid,
                installation_id="INST-REPEAT",
                outcome="pass",
                finding_count=0,
                days_to_complete=10,
            )
        result = registry.check_conflict_of_interest(vid, "INST-REPEAT")
        assert result.result == COIResult.REVIEW_REQUIRED

    def test_coi_has_provenance(self, registry, registered_verifier):
        result = registry.check_conflict_of_interest(
            registered_verifier.verifier_id, "INST-001"
        )
        assert len(result.provenance_hash) == 64


# ===========================================================================
# TEST CLASS -- Search verifiers
# ===========================================================================

class TestSearchVerifiers:
    """Tests for search_verifiers."""

    def test_search_by_sector(self, registry, registered_verifier, second_verifier):
        results = registry.search_verifiers(sector_expertise="iron_steel")
        assert len(results) == 1
        assert results[0].company_name == "EcoVerify GmbH"

    def test_search_by_country(self, registry, registered_verifier, second_verifier):
        results = registry.search_verifiers(country="DE")
        assert len(results) == 1
        assert results[0].nab_country == "DE"

    def test_search_no_filters(self, registry, registered_verifier, second_verifier):
        results = registry.search_verifiers()
        assert len(results) == 2

    def test_search_no_results(self, registry, registered_verifier):
        results = registry.search_verifiers(sector_expertise="electricity")
        assert len(results) == 0

    def test_search_by_status(self, registry, registered_verifier):
        results = registry.search_verifiers(
            accreditation_status=VerifierStatus.ACTIVE
        )
        assert len(results) >= 1

    def test_search_by_country_and_sector(
        self, registry, registered_verifier, second_verifier
    ):
        results = registry.search_verifiers(
            country="BE", sector_expertise="cement"
        )
        assert len(results) == 1
        assert results[0].company_name == "GreenAudit NV"


# ===========================================================================
# TEST CLASS -- VerificationSchedulerEngine
# ===========================================================================

class TestVerificationSchedulerInit:
    """Tests for VerificationSchedulerEngine initialization."""

    def test_init(self, scheduler):
        assert scheduler is not None

    def test_init_with_custom_registry(self):
        registry = VerifierRegistryEngine()
        scheduler = VerificationSchedulerEngine(registry)
        assert scheduler is not None


# ===========================================================================
# TEST CLASS -- Schedule visits
# ===========================================================================

class TestScheduleVisits:
    """Tests for schedule_site_visit."""

    def test_schedule_annual_visit(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-001",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date(2026, 6, 15),
            visit_type=VisitType.ON_SITE,
            year=2026,
        )
        assert visit.status == VisitStatus.SCHEDULED
        assert visit.installation_id == "INST-001"
        assert visit.visit_type == VisitType.ON_SITE

    def test_schedule_remote_audit(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-002",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date(2027, 3, 10),
            visit_type=VisitType.REMOTE,
            year=2027,
        )
        assert visit.visit_type == VisitType.REMOTE

    def test_schedule_hybrid_visit(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-003",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date(2026, 9, 1),
            visit_type=VisitType.HYBRID,
        )
        assert visit.visit_type == VisitType.HYBRID

    def test_schedule_has_provenance(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-001",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date(2026, 6, 15),
        )
        assert len(visit.provenance_hash) == 64

    def test_schedule_with_coi_raises(self, scheduler, registry):
        v = Verifier(
            company_name="ConflictVerifier",
            nab_country="NL",
            accreditation_number="RvA-COI-SCH",
            accredited_until=date.today() + timedelta(days=365),
            sector_expertise=["iron_steel"],
            consulting_clients=["INST-CONFLICT"],
        )
        registered = registry.register_verifier(v)
        with pytest.raises(ValueError, match="Conflict of interest"):
            scheduler.schedule_site_visit(
                installation_id="INST-CONFLICT",
                verifier_id=registered.verifier_id,
                visit_date=date(2026, 6, 15),
            )


# ===========================================================================
# TEST CLASS -- Visit outcomes
# ===========================================================================

class TestVisitOutcome:
    """Tests for recording visit outcomes."""

    def test_outcome_pass(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-001",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date.today(),
        )
        updated = scheduler.record_visit_outcome(
            visit_id=visit.visit_id,
            outcome=VisitOutcome.PASS,
            report_reference="RPT-2026-001",
        )
        assert updated.status == VisitStatus.COMPLETED
        assert updated.outcome == VisitOutcome.PASS

    def test_outcome_fail(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-002",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date.today(),
        )
        findings = [
            VisitFinding(
                category="emission_calculation",
                severity=FindingSeverity.CRITICAL,
                description="Major error in emission factor",
                corrective_action="Recalculate using correct EF",
            )
        ]
        updated = scheduler.record_visit_outcome(
            visit_id=visit.visit_id,
            outcome=VisitOutcome.FAIL,
            findings=findings,
        )
        assert updated.outcome == VisitOutcome.FAIL
        assert len(updated.findings) == 1

    def test_outcome_conditional(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-003",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date.today(),
        )
        updated = scheduler.record_visit_outcome(
            visit_id=visit.visit_id,
            outcome=VisitOutcome.CONDITIONAL,
            findings=[VisitFinding(severity=FindingSeverity.MINOR)],
        )
        assert updated.outcome == VisitOutcome.CONDITIONAL

    def test_outcome_already_completed_raises(self, scheduler, registered_verifier):
        visit = scheduler.schedule_site_visit(
            installation_id="INST-004",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date.today(),
        )
        scheduler.record_visit_outcome(
            visit_id=visit.visit_id,
            outcome=VisitOutcome.PASS,
        )
        with pytest.raises(ValueError, match="Cannot record outcome"):
            scheduler.record_visit_outcome(
                visit_id=visit.visit_id,
                outcome=VisitOutcome.FAIL,
            )


# ===========================================================================
# TEST CLASS -- Upcoming visits
# ===========================================================================

class TestUpcomingVisits:
    """Tests for get_upcoming_visits."""

    def test_upcoming_visits(self, scheduler, registered_verifier):
        future_date = date.today() + timedelta(days=10)
        scheduler.schedule_site_visit(
            installation_id="INST-001",
            verifier_id=registered_verifier.verifier_id,
            visit_date=future_date,
        )
        upcoming = scheduler.get_upcoming_visits(days_ahead=30)
        assert len(upcoming) == 1

    def test_upcoming_visits_empty(self, scheduler):
        upcoming = scheduler.get_upcoming_visits(days_ahead=30)
        assert len(upcoming) == 0

    def test_visit_schedule_for_installation(self, scheduler, registered_verifier):
        scheduler.schedule_site_visit(
            installation_id="INST-001",
            verifier_id=registered_verifier.verifier_id,
            visit_date=date.today() + timedelta(days=5),
        )
        schedule = scheduler.get_visit_schedule("INST-001")
        assert len(schedule) == 1

    def test_biennial_visit_2027(self, scheduler, registered_verifier):
        # In 2027+, biennial visits apply
        required = scheduler.is_visit_required("INST-NEW", 2027)
        assert required is True


# ===========================================================================
# TEST CLASS -- MaterialityAssessorEngine
# ===========================================================================

class TestMaterialityAssessorInit:
    """Tests for MaterialityAssessorEngine initialization."""

    def test_init(self, assessor):
        assert assessor is not None


# ===========================================================================
# TEST CLASS -- 5% materiality threshold
# ===========================================================================

class TestMaterialityThreshold:
    """Tests for 5% materiality threshold."""

    def test_threshold_below(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("102.0"),
                },
            },
        )
        assert result.above_threshold is False
        assert result.cn_codes_above_threshold == 0

    def test_threshold_at_5pct(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("105.0"),  # 4.76% -> below
                },
            },
        )
        # 100-105 = 5/105 = 4.76% which is below 5%
        assert result.cn_codes_above_threshold == 0

    def test_threshold_above(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("120.0"),  # 16.67%
                },
            },
        )
        assert result.above_threshold is True
        assert result.cn_codes_above_threshold == 1

    def test_calculate_materiality_percentage(self, assessor):
        pct = assessor.calculate_materiality_percentage(
            Decimal("100"), Decimal("110")
        )
        # |100-110|/110*100 = 9.09%
        assert pct == Decimal("9.09")

    def test_calculate_materiality_zero_verified(self, assessor):
        pct = assessor.calculate_materiality_percentage(
            Decimal("100"), Decimal("0")
        )
        assert pct == Decimal("100")

    def test_calculate_materiality_both_zero(self, assessor):
        pct = assessor.calculate_materiality_percentage(
            Decimal("0"), Decimal("0")
        )
        assert pct == Decimal("0")


# ===========================================================================
# TEST CLASS -- Materiality by CN code
# ===========================================================================

class TestMaterialityByCnCode:
    """Tests for per-CN-code materiality assessment."""

    def test_multiple_cn_codes(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("102.0"),
                },
                "25231000": {
                    "declared": Decimal("200.0"),
                    "verified": Decimal("250.0"),  # 20% under
                },
            },
        )
        assert len(result.cn_code_results) == 2
        above = [cn for cn in result.cn_code_results if cn.above_threshold]
        assert len(above) == 1
        assert above[0].cn_code == "25231000"

    def test_cn_code_direction_over(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("120.0"),
                    "verified": Decimal("100.0"),
                },
            },
        )
        cn = result.cn_code_results[0]
        assert cn.direction == "over"

    def test_cn_code_direction_under(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("80.0"),
                    "verified": Decimal("100.0"),
                },
            },
        )
        cn = result.cn_code_results[0]
        assert cn.direction == "under"

    def test_cn_code_direction_none(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("100.0"),
                },
            },
        )
        cn = result.cn_code_results[0]
        assert cn.direction == "none"

    def test_corrective_action_required(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("80.0"),
                    "verified": Decimal("100.0"),  # 20%
                },
            },
        )
        cn = result.cn_code_results[0]
        assert cn.corrective_action_required is True


# ===========================================================================
# TEST CLASS -- Recommended verification scope
# ===========================================================================

class TestRecommendedScope:
    """Tests for get_recommended_scope."""

    def test_first_verification_full_scope(self, assessor):
        scope = assessor.get_recommended_scope(
            installation_id="INST-NEW",
            year=2026,
        )
        assert scope.recommended_sample_size_pct == Decimal("100")
        assert "First verification" in scope.scope_rationale

    def test_scope_with_history(self, assessor):
        # Create history first
        assessor.assess_materiality(
            installation_id="INST-HIST",
            reporting_year=2025,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("120.0"),  # Above 5%
                },
                "25231000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("101.0"),  # Below 2%
                },
            },
        )
        scope = assessor.get_recommended_scope(
            installation_id="INST-HIST",
            year=2026,
            cn_codes=["72011000", "25231000"],
        )
        assert "72011000" in scope.high_risk_cn_codes
        assert "25231000" in scope.low_risk_cn_codes

    def test_scope_estimated_hours(self, assessor):
        scope = assessor.get_recommended_scope(
            installation_id="INST-NEW",
            year=2026,
        )
        assert scope.estimated_verification_hours >= Decimal("0")


# ===========================================================================
# TEST CLASS -- Historical materiality trend
# ===========================================================================

class TestMaterialityTrend:
    """Tests for get_materiality_trend."""

    def test_trend_insufficient_data(self, assessor):
        assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("105.0"),
                },
            },
        )
        trend = assessor.get_materiality_trend("INST-001")
        assert len(trend.years) == 1
        assert "Insufficient" in trend.trend_description

    def test_trend_improving(self, assessor):
        assessor.assess_materiality(
            installation_id="INST-TREND",
            reporting_year=2025,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("120.0"),
                },
            },
        )
        assessor.assess_materiality(
            installation_id="INST-TREND",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("105.0"),
                },
            },
        )
        trend = assessor.get_materiality_trend("INST-TREND")
        assert trend.improving is True
        assert "improving" in trend.trend_description.lower()

    def test_trend_worsening(self, assessor):
        assessor.assess_materiality(
            installation_id="INST-WORSE",
            reporting_year=2025,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("101.0"),
                },
            },
        )
        assessor.assess_materiality(
            installation_id="INST-WORSE",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("130.0"),
                },
            },
        )
        trend = assessor.get_materiality_trend("INST-WORSE")
        assert trend.improving is False
        assert "worsening" in trend.trend_description.lower()

    def test_trend_has_provenance(self, assessor):
        assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("105.0"),
                },
            },
        )
        trend = assessor.get_materiality_trend("INST-001")
        assert len(trend.provenance_hash) == 64


# ===========================================================================
# TEST CLASS -- Direct plus indirect materiality
# ===========================================================================

class TestDirectPlusIndirect:
    """Tests for aggregate materiality (direct + indirect)."""

    def test_aggregate_materiality(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-AGG",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("500.0"),
                    "verified": Decimal("520.0"),
                },
                "72031000": {
                    "declared": Decimal("300.0"),
                    "verified": Decimal("310.0"),
                },
            },
        )
        assert result.total_declared_tco2e == Decimal("800.0")
        assert result.total_verified_tco2e == Decimal("830.0")
        assert result.overall_discrepancy_tco2e == Decimal("30.0")

    def test_aggregate_within_threshold(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-AGG2",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("1000.0"),
                    "verified": Decimal("1020.0"),
                },
            },
        )
        # 20/1020 = 1.96%
        assert result.overall_materiality_pct < MATERIALITY_THRESHOLD_PCT

    def test_no_emissions_data(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-EMPTY",
            reporting_year=2026,
            emissions_data=None,
        )
        assert result.total_declared_tco2e == Decimal("0")
        assert result.cn_codes_above_threshold == 0


# ===========================================================================
# TEST CLASS -- Materiality provenance
# ===========================================================================

class TestMaterialityProvenance:
    """Tests for provenance hash on materiality result."""

    def test_result_has_provenance(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-001",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("102.0"),
                },
            },
        )
        assert len(result.provenance_hash) == 64

    def test_recommended_actions_below_threshold(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-OK",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("101.0"),
                },
            },
        )
        assert any("No corrective" in a for a in result.recommended_actions)

    def test_recommended_actions_above_threshold(self, assessor):
        result = assessor.assess_materiality(
            installation_id="INST-BAD",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("80.0"),
                    "verified": Decimal("100.0"),
                },
            },
        )
        assert any("corrective" in a.lower() for a in result.recommended_actions)

    def test_flag_threshold_breaches(self, assessor):
        assessor.assess_materiality(
            installation_id="INST-BREACH",
            reporting_year=2026,
            emissions_data={
                "72011000": {
                    "declared": Decimal("80.0"),
                    "verified": Decimal("100.0"),
                },
                "25231000": {
                    "declared": Decimal("100.0"),
                    "verified": Decimal("101.0"),
                },
            },
        )
        breaches = assessor.flag_threshold_breaches("INST-BREACH", 2026)
        assert len(breaches) == 1
        assert breaches[0]["cn_code"] == "72011000"


# ===========================================================================
# TEST CLASS -- Verifier performance
# ===========================================================================

class TestVerifierPerformance:
    """Tests for verifier performance tracking."""

    def test_performance_no_history(self, registry, registered_verifier):
        perf = registry.get_verifier_performance(registered_verifier.verifier_id)
        assert perf.total_verifications == 0
        assert perf.performance_score >= Decimal("0")

    def test_performance_with_outcomes(self, registry, registered_verifier):
        vid = registered_verifier.verifier_id
        registry.record_visit_outcome(vid, "INST-001", "pass", 1, 20, "iron_steel", "TR")
        registry.record_visit_outcome(vid, "INST-002", "pass", 0, 15, "aluminium", "CN")
        registry.record_visit_outcome(vid, "INST-003", "fail", 3, 30, "iron_steel", "TR")

        perf = registry.get_verifier_performance(vid)
        assert perf.total_verifications == 3
        assert perf.passed == 2
        assert perf.failed == 1
        assert len(perf.provenance_hash) == 64
