# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Change Management Engine Tests
======================================================

Tests ChangeManagementEngine: change request processing, impact
assessment, approval routing, base year trigger detection, status
transitions, batch assessment, and summaries.

Target: 60+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("change_management")

ChangeManagementEngine = _mod.ChangeManagementEngine
ChangeRequest = _mod.ChangeRequest
AffectedSource = _mod.AffectedSource
ChangeImpact = _mod.ChangeImpact
ChangeApproval = _mod.ChangeApproval
BaseYearTriggerDetection = _mod.BaseYearTriggerDetection
ChangeManagementResult = _mod.ChangeManagementResult
ChangeCategory = _mod.ChangeCategory
ChangeStatus = _mod.ChangeStatus
ImpactSeverity = _mod.ImpactSeverity
ApprovalLevel = _mod.ApprovalLevel
BaseYearTriggerType = _mod.BaseYearTriggerType
VALID_TRANSITIONS = _mod.VALID_TRANSITIONS
IMPACT_THRESHOLDS = _mod.IMPACT_THRESHOLDS


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh ChangeManagementEngine."""
    return ChangeManagementEngine()


@pytest.fixture
def low_impact_request():
    """Change request with impact < 1%."""
    return ChangeRequest(
        title="Minor EF update for office heating",
        description="Updated emission factor for natural gas boiler",
        category=ChangeCategory.METHODOLOGICAL,
        requester_id="user-001",
        requester_name="Analyst",
        reporting_year=2025,
        total_inventory_tco2e=Decimal("55000"),
        affected_sources=[
            AffectedSource(
                source_id="SRC-001",
                source_name="Office heating gas",
                scope="scope1",
                category="stationary_combustion",
                old_value_tco2e=Decimal("500"),
                new_value_tco2e=Decimal("495"),
                delta_tco2e=Decimal("-5"),
            ),
        ],
        affected_scopes=["scope1"],
    )


@pytest.fixture
def medium_impact_request():
    """Change request with impact 1-5%."""
    return ChangeRequest(
        title="Updated grid emission factors",
        description="IEA 2025 grid factors for Germany",
        category=ChangeCategory.METHODOLOGICAL,
        requester_id="user-002",
        requester_name="Sustainability Manager",
        reporting_year=2025,
        total_inventory_tco2e=Decimal("55000"),
        affected_sources=[
            AffectedSource(
                source_id="SRC-ELEC-001",
                source_name="Grid electricity",
                scope="scope2",
                category="purchased_electricity",
                old_value_tco2e=Decimal("5000"),
                new_value_tco2e=Decimal("4000"),
                delta_tco2e=Decimal("-1000"),
            ),
        ],
        affected_scopes=["scope2"],
    )


@pytest.fixture
def high_impact_request():
    """Change request with impact 5-15%."""
    return ChangeRequest(
        title="Acquisition of new manufacturing plant",
        description="Integration of acquired facility emissions",
        category=ChangeCategory.STRUCTURAL,
        requester_id="user-003",
        requester_name="CFO",
        reporting_year=2025,
        total_inventory_tco2e=Decimal("55000"),
        affected_sources=[
            AffectedSource(
                source_id="SRC-NEW-001",
                source_name="Acquired plant Scope 1",
                scope="scope1",
                old_value_tco2e=Decimal("0"),
                new_value_tco2e=Decimal("5000"),
                delta_tco2e=Decimal("5000"),
            ),
            AffectedSource(
                source_id="SRC-NEW-002",
                source_name="Acquired plant Scope 2",
                scope="scope2",
                old_value_tco2e=Decimal("0"),
                new_value_tco2e=Decimal("2000"),
                delta_tco2e=Decimal("2000"),
            ),
        ],
        affected_scopes=["scope1", "scope2"],
    )


@pytest.fixture
def critical_impact_request():
    """Change request with impact > 15%."""
    return ChangeRequest(
        title="Divestiture of major subsidiary",
        description="Selling off 40% of business",
        category=ChangeCategory.STRUCTURAL,
        requester_id="user-ceo",
        requester_name="CEO",
        reporting_year=2025,
        total_inventory_tco2e=Decimal("55000"),
        affected_sources=[
            AffectedSource(
                source_id="SRC-DIV-001",
                source_name="Divested subsidiary",
                scope="scope1",
                old_value_tco2e=Decimal("12000"),
                new_value_tco2e=Decimal("0"),
                delta_tco2e=Decimal("-12000"),
            ),
        ],
        affected_scopes=["scope1", "scope2"],
    )


# ===================================================================
# Impact Assessment Tests
# ===================================================================


class TestImpactAssessment:
    """Tests for assess_impact."""

    def test_low_impact_severity(self, engine, low_impact_request):
        impact = engine.assess_impact(low_impact_request)
        assert impact.severity == ImpactSeverity.LOW

    def test_medium_impact_severity(self, engine, medium_impact_request):
        impact = engine.assess_impact(medium_impact_request)
        assert impact.severity == ImpactSeverity.MEDIUM

    def test_high_impact_severity(self, engine, high_impact_request):
        impact = engine.assess_impact(high_impact_request)
        assert impact.severity == ImpactSeverity.HIGH

    def test_critical_impact_severity(self, engine, critical_impact_request):
        impact = engine.assess_impact(critical_impact_request)
        assert impact.severity == ImpactSeverity.CRITICAL

    def test_impact_pct_calculation(self, engine, medium_impact_request):
        impact = engine.assess_impact(medium_impact_request)
        expected_pct = (Decimal("1000") / Decimal("55000")) * 100
        assert abs(impact.impact_pct - expected_pct) < Decimal("0.1")

    def test_net_impact_signed(self, engine, medium_impact_request):
        impact = engine.assess_impact(medium_impact_request)
        assert impact.net_impact_tco2e < Decimal("0")

    def test_total_impact_absolute(self, engine, medium_impact_request):
        impact = engine.assess_impact(medium_impact_request)
        assert impact.total_impact_tco2e > Decimal("0")

    def test_affected_source_count(self, engine, high_impact_request):
        impact = engine.assess_impact(high_impact_request)
        assert impact.affected_source_count == 2

    def test_scope_impacts_populated(self, engine, high_impact_request):
        impact = engine.assess_impact(high_impact_request)
        assert len(impact.scope_impacts) >= 1

    def test_provenance_hash(self, engine, low_impact_request):
        # assess_impact returns ChangeImpact directly (no provenance_hash)
        # Use process_change for provenance hash test
        result = engine.process_change(low_impact_request)
        assert len(result.provenance_hash) == 64


# ===================================================================
# Approval Routing Tests
# ===================================================================


class TestApprovalRouting:
    """Tests for determine_approval_routing."""

    def test_low_impact_auto_approved(self, engine, low_impact_request):
        impact = engine.assess_impact(low_impact_request)
        approval = engine.determine_approval_routing(low_impact_request, impact)
        assert approval.auto_approved is True
        assert approval.required_level == ApprovalLevel.AUTO

    def test_medium_impact_single_approver(self, engine, medium_impact_request):
        impact = engine.assess_impact(medium_impact_request)
        approval = engine.determine_approval_routing(medium_impact_request, impact)
        assert approval.required_level == ApprovalLevel.SINGLE_APPROVER

    def test_high_impact_dual_approval(self, engine, high_impact_request):
        impact = engine.assess_impact(high_impact_request)
        approval = engine.determine_approval_routing(high_impact_request, impact)
        assert approval.required_level == ApprovalLevel.DUAL_APPROVAL

    def test_critical_impact_board_level(self, engine, critical_impact_request):
        impact = engine.assess_impact(critical_impact_request)
        approval = engine.determine_approval_routing(critical_impact_request, impact)
        assert approval.required_level == ApprovalLevel.BOARD_LEVEL
        assert approval.requires_external_verifier is True

    def test_structural_always_requires_approval(self, engine):
        req = ChangeRequest(
            title="Small structural change",
            category=ChangeCategory.STRUCTURAL,
            total_inventory_tco2e=Decimal("100000"),
            affected_sources=[
                AffectedSource(
                    source_id="S1", scope="scope1",
                    old_value_tco2e=Decimal("100"),
                    new_value_tco2e=Decimal("100"),
                    delta_tco2e=Decimal("0"),
                ),
            ],
        )
        impact = engine.assess_impact(req)
        approval = engine.determine_approval_routing(req, impact)
        assert approval.auto_approved is False


# ===================================================================
# Base Year Trigger Tests
# ===================================================================


class TestBaseYearTrigger:
    """Tests for detect_base_year_trigger."""

    def test_structural_change_triggers_recalculation(self, high_impact_request):
        engine = ChangeManagementEngine(
            base_year=2019,
            base_year_total_tco2e=Decimal("60000"),
        )
        impact = engine.assess_impact(high_impact_request)
        trigger = engine.detect_base_year_trigger(high_impact_request, impact)
        assert trigger.triggers_recalculation is True

    def test_low_impact_no_trigger(self, low_impact_request):
        engine = ChangeManagementEngine(
            base_year=2019,
            base_year_total_tco2e=Decimal("60000"),
        )
        impact = engine.assess_impact(low_impact_request)
        trigger = engine.detect_base_year_trigger(low_impact_request, impact)
        assert trigger.triggers_recalculation is False

    def test_trigger_type_set(self, high_impact_request):
        engine = ChangeManagementEngine(
            base_year=2019,
            base_year_total_tco2e=Decimal("60000"),
        )
        impact = engine.assess_impact(high_impact_request)
        trigger = engine.detect_base_year_trigger(high_impact_request, impact)
        assert trigger.trigger_type != BaseYearTriggerType.NO_TRIGGER

    def test_no_trigger_type_for_low(self, low_impact_request):
        engine = ChangeManagementEngine(
            base_year=2019,
            base_year_total_tco2e=Decimal("60000"),
        )
        impact = engine.assess_impact(low_impact_request)
        trigger = engine.detect_base_year_trigger(low_impact_request, impact)
        assert trigger.trigger_type == BaseYearTriggerType.NO_TRIGGER


# ===================================================================
# Process Change Tests (full workflow)
# ===================================================================


class TestProcessChange:
    """Tests for process_change (combined workflow)."""

    def test_process_low_impact(self, engine, low_impact_request):
        result = engine.process_change(low_impact_request)
        assert result.impact is not None
        assert result.approval_routing is not None
        assert result.request.status != ChangeStatus.DRAFT

    def test_process_high_impact(self, engine, high_impact_request):
        result = engine.process_change(high_impact_request)
        assert result.impact.severity == ImpactSeverity.HIGH

    def test_process_populates_change_log(self, engine, medium_impact_request):
        result = engine.process_change(medium_impact_request)
        assert len(result.change_log) >= 1


# ===================================================================
# Status Transition Tests
# ===================================================================


class TestStatusTransitions:
    """Tests for transition_status."""

    def test_transition_from_draft(self, engine):
        req = ChangeRequest(
            title="Transition test",
            category=ChangeCategory.ERROR_CORRECTION,
            total_inventory_tco2e=Decimal("50000"),
            affected_sources=[
                AffectedSource(
                    source_id="S1", scope="scope1",
                    old_value_tco2e=Decimal("100"),
                    new_value_tco2e=Decimal("90"),
                ),
            ],
        )
        updated = engine.transition_status(req, ChangeStatus.SUBMITTED)
        assert updated.status == ChangeStatus.SUBMITTED

    def test_invalid_transition_raises(self, engine):
        req = ChangeRequest(
            title="Invalid transition test",
            status=ChangeStatus.DRAFT,
        )
        with pytest.raises((ValueError, Exception)):
            engine.transition_status(req, ChangeStatus.IMPLEMENTED)

    def test_valid_transitions_lookup(self, engine):
        valid = engine.get_valid_transitions(ChangeStatus.DRAFT)
        assert ChangeStatus.SUBMITTED in valid

    @pytest.mark.parametrize("from_status,to_statuses", list(VALID_TRANSITIONS.items()))
    def test_transition_matrix_entries(self, from_status, to_statuses):
        assert isinstance(to_statuses, (set, frozenset, list))


# ===================================================================
# Batch Assessment Tests
# ===================================================================


class TestBatchAssessment:
    """Tests for batch_assess."""

    def test_batch_assess_multiple(self, engine, low_impact_request, medium_impact_request):
        results = engine.batch_assess([low_impact_request, medium_impact_request])
        assert len(results) == 2

    def test_batch_assess_empty(self, engine):
        results = engine.batch_assess([])
        assert len(results) == 0


# ===================================================================
# Summarise Changes Tests
# ===================================================================


class TestSummariseChanges:
    """Tests for summarise_changes."""

    def test_summarise_after_processing(self, engine, low_impact_request, medium_impact_request):
        r1 = engine.process_change(low_impact_request)
        r2 = engine.process_change(medium_impact_request)
        summary = engine.summarise_changes([r1, r2])
        assert summary is not None


# ===================================================================
# Impact Threshold Tests
# ===================================================================


class TestImpactThresholds:
    """Tests for threshold configuration constants."""

    def test_low_max_threshold(self):
        assert IMPACT_THRESHOLDS["low_max"] == Decimal("1")

    def test_medium_max_threshold(self):
        assert IMPACT_THRESHOLDS["medium_max"] == Decimal("5")

    def test_high_max_threshold(self):
        assert IMPACT_THRESHOLDS["high_max"] == Decimal("15")


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults and enum values."""

    @pytest.mark.parametrize("cat", list(ChangeCategory))
    def test_change_categories(self, cat):
        assert cat.value is not None

    @pytest.mark.parametrize("status", list(ChangeStatus))
    def test_change_statuses(self, status):
        assert status.value is not None

    @pytest.mark.parametrize("sev", list(ImpactSeverity))
    def test_impact_severities(self, sev):
        assert sev.value is not None

    @pytest.mark.parametrize("level", list(ApprovalLevel))
    def test_approval_levels(self, level):
        assert level.value is not None

    @pytest.mark.parametrize("trigger", list(BaseYearTriggerType))
    def test_base_year_trigger_types(self, trigger):
        assert trigger.value is not None

    def test_change_request_defaults(self):
        cr = ChangeRequest()
        assert cr.status == ChangeStatus.DRAFT
        assert cr.category == ChangeCategory.METHODOLOGICAL

    def test_affected_source_defaults(self):
        src = AffectedSource()
        assert src.scope == "scope1"
        assert src.old_value_tco2e == Decimal("0")
