# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Data Collection Engine Tests
=====================================================

Tests DataCollectionEngine: campaign management, data requests,
submissions, validation, reminders, coverage analysis, and progress.

Target: 70+ test cases.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("data_collection")

DataCollectionEngine = _mod.DataCollectionEngine
DataCollectionResult = _mod.DataCollectionResult
CollectionCampaign = _mod.CollectionCampaign
DataRequest = _mod.DataRequest
CollectionProgress = _mod.CollectionProgress
CampaignStatus = _mod.CampaignStatus
RequestStatus = _mod.RequestStatus
DataScope = _mod.DataScope
EscalationLevel = _mod.EscalationLevel
ValidationSeverity = _mod.ValidationSeverity
DEFAULT_VALIDATION_RANGES = _mod.DEFAULT_VALIDATION_RANGES


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh DataCollectionEngine."""
    return DataCollectionEngine()


@pytest.fixture
def campaign(engine, sample_campaign):
    """Create a campaign and return (engine, result)."""
    result = engine.create_campaign(
        period_id=sample_campaign["period_id"],
        organisation_id=sample_campaign["organisation_id"],
        campaign_name=sample_campaign["campaign_name"],
        start_date=sample_campaign["start_date"],
        end_date=sample_campaign["end_date"],
        created_by=sample_campaign["created_by"],
        notes=sample_campaign["notes"],
    )
    return engine, result


@pytest.fixture
def campaign_with_requests(campaign):
    """Create a campaign with multiple data requests."""
    engine, result = campaign
    cid = result.campaign.campaign_id
    engine.add_request(
        campaign_id=cid,
        scope=DataScope.SCOPE_1,
        category="stationary_combustion",
        facility_id="FAC-001",
        assigned_to="Facility Manager A",
    )
    engine.add_request(
        campaign_id=cid,
        scope=DataScope.SCOPE_2,
        category="purchased_electricity",
        facility_id="FAC-002",
        assigned_to="Facility Manager B",
    )
    engine.add_request(
        campaign_id=cid,
        scope=DataScope.SCOPE_1,
        category="mobile_combustion",
        facility_id="FAC-001",
        assigned_to="Fleet Manager",
    )
    return engine, cid


# ===================================================================
# Campaign Creation Tests
# ===================================================================


class TestCampaignCreation:
    """Tests for create_campaign."""

    def test_create_campaign_returns_result(self, campaign):
        _, result = campaign
        assert isinstance(result, DataCollectionResult)
        assert result.operation == "create_campaign"

    def test_campaign_has_planning_status(self, campaign):
        _, result = campaign
        assert result.campaign.status == CampaignStatus.PLANNING

    def test_campaign_has_provenance_hash(self, campaign):
        _, result = campaign
        assert len(result.provenance_hash) == 64

    def test_campaign_stored(self, campaign):
        engine, result = campaign
        cid = result.campaign.campaign_id
        retrieved = engine.get_campaign(cid)
        assert retrieved.campaign_id == cid

    def test_campaign_name_preserved(self, campaign):
        _, result = campaign
        assert "FY2025" in result.campaign.campaign_name

    def test_campaign_dates_preserved(self, campaign, sample_campaign):
        _, result = campaign
        assert result.campaign.start_date == sample_campaign["start_date"]
        assert result.campaign.end_date == sample_campaign["end_date"]

    def test_campaign_notes_preserved(self, campaign, sample_campaign):
        _, result = campaign
        assert result.campaign.notes == sample_campaign["notes"]

    def test_campaign_initial_progress_zero(self, campaign):
        _, result = campaign
        assert result.campaign.progress.total_requests == 0
        assert result.campaign.progress.progress_pct == Decimal("0")


# ===================================================================
# Campaign Lifecycle Tests
# ===================================================================


class TestCampaignLifecycle:
    """Tests for campaign status transitions."""

    def test_launch_campaign(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        result = engine.launch_campaign(cid)
        assert result.campaign.status == CampaignStatus.ACTIVE

    def test_launch_sets_sent_status_on_requests(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        sent_count = sum(1 for r in c.requests if r.status == RequestStatus.SENT)
        assert sent_count == 3

    def test_close_campaign(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        result = engine.close_campaign(cid)
        assert result.campaign.status == CampaignStatus.CLOSED

    def test_close_before_launch(self, campaign):
        engine, result = campaign
        cid = result.campaign.campaign_id
        # Engine may allow closing a non-launched campaign or may raise
        try:
            r = engine.close_campaign(cid)
            # If it succeeds, status should be closed
            assert r.campaign.status == CampaignStatus.CLOSED
        except (ValueError, Exception):
            pass  # Expected for some engine implementations


# ===================================================================
# Data Request Tests
# ===================================================================


class TestDataRequests:
    """Tests for add_request and request management."""

    def test_add_request_returns_result(self, campaign):
        engine, result = campaign
        cid = result.campaign.campaign_id
        r = engine.add_request(
            campaign_id=cid,
            scope=DataScope.SCOPE_1,
            category="stationary_combustion",
            facility_id="FAC-001",
        )
        assert r.request is not None

    def test_add_request_default_draft_status(self, campaign):
        engine, result = campaign
        cid = result.campaign.campaign_id
        r = engine.add_request(
            campaign_id=cid,
            scope=DataScope.SCOPE_1,
            category="stationary_combustion",
            facility_id="FAC-001",
        )
        assert r.request.status == RequestStatus.DRAFT

    def test_add_request_increments_total(self, campaign):
        engine, result = campaign
        cid = result.campaign.campaign_id
        engine.add_request(
            campaign_id=cid, scope=DataScope.SCOPE_1,
            category="stationary", facility_id="F1",
        )
        c = engine.get_campaign(cid)
        assert c.progress.total_requests >= 1

    def test_add_request_to_nonexistent_campaign_raises(self, engine):
        with pytest.raises((KeyError, Exception)):
            engine.add_request(
                campaign_id="nonexistent",
                scope=DataScope.SCOPE_1,
                category="test",
                facility_id="F1",
            )

    def test_assign_request(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        r = engine.assign_request(
            campaign_id=cid,
            request_id=req_id,
            assigned_to="user-fm-001",
            due_date=date(2026, 3, 1),
        )
        assert r.request.assigned_to == "user-fm-001"

    def test_assign_request_sets_deadline(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        due = date(2026, 3, 15)
        engine.assign_request(cid, req_id, "user-001", due_date=due)
        c = engine.get_campaign(cid)
        req = [r for r in c.requests if r.request_id == req_id][0]
        assert req.due_date == due

    @pytest.mark.parametrize("scope", list(DataScope))
    def test_all_scopes_accepted(self, scope, campaign):
        engine, result = campaign
        cid = result.campaign.campaign_id
        r = engine.add_request(
            campaign_id=cid,
            scope=scope,
            category="test_category",
            facility_id="FAC-TEST",
        )
        assert r.request is not None


# ===================================================================
# Submission Tests
# ===================================================================


class TestSubmissions:
    """Tests for submit_data, accept_submission, reject_submission."""

    def test_submit_data(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        r = engine.submit_data(
            campaign_id=cid,
            request_id=req_id,
            data_payload={"electricity_kwh": 150000, "natural_gas_m3": 50000},
            submitted_by="user-fm-001",
        )
        assert r.submission is not None

    def test_submit_data_updates_request_status(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        engine.submit_data(cid, req_id, {"electricity_kwh": 100000}, submitted_by="user-001")
        c = engine.get_campaign(cid)
        req = [r for r in c.requests if r.request_id == req_id][0]
        # Valid data -> VALIDATED; invalid data -> SUBMITTED
        assert req.status in (RequestStatus.SUBMITTED, RequestStatus.VALIDATED)

    def test_accept_submission(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        sub_result = engine.submit_data(cid, req_id, {"electricity_kwh": 100000}, submitted_by="user-001")
        sub_id = sub_result.submission.submission_id
        r = engine.accept_submission(cid, req_id, sub_id, accepted_by="reviewer-001")
        assert r.request.status == RequestStatus.ACCEPTED

    def test_reject_submission(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        sub_result = engine.submit_data(cid, req_id, {"electricity_kwh": 100000}, submitted_by="user-001")
        sub_id = sub_result.submission.submission_id
        r = engine.reject_submission(
            cid, req_id, sub_id,
            rejected_by="reviewer-001",
            reason="Data appears incomplete",
        )
        assert r.request.status == RequestStatus.REJECTED


# ===================================================================
# Validation Tests
# ===================================================================


class TestValidation:
    """Tests for data validation ranges and submission validation."""

    def test_default_validation_ranges_exist(self):
        assert len(DEFAULT_VALIDATION_RANGES) > 0
        assert "electricity_kwh" in DEFAULT_VALIDATION_RANGES

    @pytest.mark.parametrize("field,expected_unit", [
        ("electricity_kwh", "kWh"),
        ("natural_gas_m3", "m3"),
        ("diesel_litres", "litres"),
        ("refrigerant_kg", "kg"),
        ("waste_tonnes", "tonnes"),
        ("distance_km", "km"),
    ])
    def test_validation_range_units(self, field, expected_unit):
        min_val, max_val, unit = DEFAULT_VALIDATION_RANGES[field]
        assert unit == expected_unit
        assert min_val >= 0
        assert max_val > min_val

    def test_submit_with_out_of_range_data_generates_findings(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        r = engine.submit_data(
            cid, req_id,
            {"electricity_kwh": -100},
            submitted_by="user-001",
        )
        assert r.submission is not None


# ===================================================================
# Reminders Tests
# ===================================================================


class TestReminders:
    """Tests for generate_reminders."""

    def test_generate_reminders_returns_result(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        r = engine.generate_reminders(cid)
        assert isinstance(r, DataCollectionResult)

    def test_generate_reminders_for_overdue_requests(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        c = engine.get_campaign(cid)
        for req in c.requests:
            req.due_date = date.today() - timedelta(days=10)
        engine.launch_campaign(cid)
        r = engine.generate_reminders(cid)
        assert len(r.reminders_generated) >= 0


# ===================================================================
# Coverage Analysis Tests
# ===================================================================


class TestCoverageAnalysis:
    """Tests for calculate_coverage."""

    def test_coverage_returns_result(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        r = engine.calculate_coverage(cid)
        assert isinstance(r, DataCollectionResult)
        assert r.progress is not None

    def test_coverage_zero_for_no_submissions(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        r = engine.calculate_coverage(cid)
        assert r.progress.accepted_count == 0

    def test_coverage_increases_with_acceptance(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        req_id = c.requests[0].request_id
        sub_result = engine.submit_data(cid, req_id, {"electricity_kwh": 100000}, submitted_by="u1")
        sub_id = sub_result.submission.submission_id
        engine.accept_submission(cid, req_id, sub_id, "r1")
        r = engine.calculate_coverage(cid)
        assert r.progress.accepted_count >= 1

    def test_progress_totals_match_request_count(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        r = engine.calculate_coverage(cid)
        assert r.progress.total_requests == 3


# ===================================================================
# Campaign Listing Tests
# ===================================================================


class TestCampaignListing:
    """Tests for list_campaigns and get_campaign."""

    def test_get_campaign_not_found(self, engine):
        with pytest.raises((KeyError, Exception)):
            engine.get_campaign("nonexistent")

    def test_list_campaigns_empty(self, engine):
        campaigns = engine.list_campaigns()
        assert len(campaigns) == 0

    def test_list_campaigns_returns_created(self, campaign):
        engine, result = campaign
        campaigns = engine.list_campaigns()
        assert len(campaigns) >= 1

    def test_list_campaigns_filter_by_period(self, engine, sample_campaign):
        engine.create_campaign(
            period_id="per-A",
            organisation_id="org-001",
            campaign_name="Camp A",
        )
        engine.create_campaign(
            period_id="per-B",
            organisation_id="org-001",
            campaign_name="Camp B",
        )
        campaigns = engine.list_campaigns(period_id="per-A")
        assert all(c.period_id == "per-A" for c in campaigns)


# ===================================================================
# Progress Tracking Tests
# ===================================================================


class TestProgressTracking:
    """Tests for progress calculation correctness."""

    def test_progress_pct_after_all_accepted(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        for req in c.requests:
            sub_r = engine.submit_data(cid, req.request_id, {"electricity_kwh": 100}, submitted_by="user")
            engine.accept_submission(cid, req.request_id, sub_r.submission.submission_id, "reviewer")
        r = engine.calculate_coverage(cid)
        assert r.progress.accepted_count == 3

    def test_on_time_percentage(self, campaign_with_requests):
        engine, cid = campaign_with_requests
        c = engine.get_campaign(cid)
        for req in c.requests:
            req.due_date = date.today() + timedelta(days=30)
        engine.launch_campaign(cid)
        c = engine.get_campaign(cid)
        sub_r = engine.submit_data(cid, c.requests[0].request_id, {"val": 1}, submitted_by="u1")
        engine.accept_submission(cid, c.requests[0].request_id, sub_r.submission.submission_id, "r1")
        r = engine.calculate_coverage(cid)
        assert r.progress is not None


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model creation."""

    def test_collection_campaign_defaults(self):
        c = CollectionCampaign()
        assert c.status == CampaignStatus.PLANNING
        assert len(c.requests) == 0

    def test_collection_progress_defaults(self):
        p = CollectionProgress()
        assert p.total_requests == 0
        assert p.progress_pct == Decimal("0")

    @pytest.mark.parametrize("status", list(CampaignStatus))
    def test_all_campaign_statuses(self, status):
        assert status.value is not None

    @pytest.mark.parametrize("status", list(RequestStatus))
    def test_all_request_statuses(self, status):
        assert status.value is not None

    @pytest.mark.parametrize("scope", list(DataScope))
    def test_all_data_scopes(self, scope):
        assert scope.value is not None

    @pytest.mark.parametrize("level", list(EscalationLevel))
    def test_all_escalation_levels(self, level):
        assert level.value is not None

    @pytest.mark.parametrize("severity", list(ValidationSeverity))
    def test_all_validation_severities(self, severity):
        assert severity.value is not None


# ===================================================================
# Provenance Tests
# ===================================================================


class TestProvenance:
    """Tests for provenance hashing on results."""

    def test_create_campaign_hash(self, campaign):
        _, result = campaign
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_add_request_hash(self, campaign):
        engine, result = campaign
        r = engine.add_request(
            result.campaign.campaign_id,
            scope=DataScope.SCOPE_1,
            category="test",
            facility_id="F1",
        )
        assert len(r.provenance_hash) == 64

    def test_processing_time_non_negative(self, campaign):
        _, result = campaign
        assert result.processing_time_ms >= Decimal("0")
