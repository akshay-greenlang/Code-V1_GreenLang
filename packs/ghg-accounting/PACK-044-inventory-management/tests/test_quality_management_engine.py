# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Quality Management Engine Tests
=======================================================

Tests QualityManagementEngine: QA/QC checks, quality scoring,
issue tracking, action management, verification readiness,
and improvement planning.

Target: 70+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("quality_management")

QualityManagementEngine = _mod.QualityManagementEngine
QAQCCheck = _mod.QAQCCheck
QAQCResult = _mod.QAQCResult
DimensionScore = _mod.DimensionScore
QualityScore = _mod.QualityScore
QualityDimension = _mod.QualityDimension
CheckSeverity = _mod.CheckSeverity
CheckResult = _mod.CheckResult
IssueStatus = _mod.IssueStatus
ActionPriority = _mod.ActionPriority
ActionStatus = _mod.ActionStatus
DEFAULT_DIMENSION_WEIGHTS = _mod.DEFAULT_DIMENSION_WEIGHTS
DEFAULT_CHECKS = _mod.DEFAULT_CHECKS


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh QualityManagementEngine."""
    return QualityManagementEngine()


@pytest.fixture
def all_pass_inputs():
    """Inputs that cause all default checks to pass."""
    return {
        "COMP-001": True, "COMP-002": True, "COMP-003": True,
        "COMP-004": True, "COMP-005": True,
        "CONS-001": True, "CONS-002": True, "CONS-003": True, "CONS-004": True,
        "ACCU-001": True, "ACCU-002": True, "ACCU-003": True,
        "ACCU-004": True, "ACCU-005": True,
        "TRAN-001": True, "TRAN-002": True, "TRAN-003": True,
        "TRAN-004": True, "TRAN-005": True,
    }


@pytest.fixture
def mixed_inputs():
    """Inputs with some failures."""
    return {
        "COMP-001": True, "COMP-002": True, "COMP-003": True,
        "COMP-004": False, "COMP-005": True,
        "CONS-001": True, "CONS-002": False, "CONS-003": True, "CONS-004": True,
        "ACCU-001": True, "ACCU-002": False, "ACCU-003": True,
        "ACCU-004": True, "ACCU-005": True,
        "TRAN-001": True, "TRAN-002": True, "TRAN-003": False,
        "TRAN-004": True, "TRAN-005": True,
    }


@pytest.fixture
def all_fail_inputs():
    """Inputs that cause all checks to fail."""
    return {k: False for k in [
        "COMP-001", "COMP-002", "COMP-003", "COMP-004", "COMP-005",
        "CONS-001", "CONS-002", "CONS-003", "CONS-004",
        "ACCU-001", "ACCU-002", "ACCU-003", "ACCU-004", "ACCU-005",
        "TRAN-001", "TRAN-002", "TRAN-003", "TRAN-004", "TRAN-005",
    ]}


# ===================================================================
# Default Checks Tests
# ===================================================================


class TestDefaultChecks:
    """Tests for the default check definitions."""

    def test_default_checks_count(self):
        assert len(DEFAULT_CHECKS) == 19

    def test_completeness_checks(self):
        comp = [c for c in DEFAULT_CHECKS if c["check_id"].startswith("COMP")]
        assert len(comp) == 5

    def test_consistency_checks(self):
        cons = [c for c in DEFAULT_CHECKS if c["check_id"].startswith("CONS")]
        assert len(cons) == 4

    def test_accuracy_checks(self):
        accu = [c for c in DEFAULT_CHECKS if c["check_id"].startswith("ACCU")]
        assert len(accu) == 5

    def test_transparency_checks(self):
        tran = [c for c in DEFAULT_CHECKS if c["check_id"].startswith("TRAN")]
        assert len(tran) == 5

    @pytest.mark.parametrize("check_id", [
        "COMP-001", "COMP-002", "COMP-003", "COMP-004", "COMP-005",
        "CONS-001", "CONS-002", "CONS-003", "CONS-004",
        "ACCU-001", "ACCU-002", "ACCU-003", "ACCU-004", "ACCU-005",
        "TRAN-001", "TRAN-002", "TRAN-003", "TRAN-004", "TRAN-005",
    ])
    def test_check_has_required_fields(self, check_id):
        check = [c for c in DEFAULT_CHECKS if c["check_id"] == check_id]
        assert len(check) == 1
        c = check[0]
        assert "name" in c
        assert "dimension" in c
        assert "severity" in c


# ===================================================================
# Dimension Weights Tests
# ===================================================================


class TestDimensionWeights:
    """Tests for dimension weight configuration."""

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert abs(float(total) - 1.0) < 0.001

    def test_completeness_weight(self):
        assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.COMPLETENESS] == Decimal("0.30")

    def test_consistency_weight(self):
        assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.CONSISTENCY] == Decimal("0.25")

    def test_accuracy_weight(self):
        assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.ACCURACY] == Decimal("0.25")

    def test_transparency_weight(self):
        assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.TRANSPARENCY] == Decimal("0.20")


# ===================================================================
# Run Checks Tests
# ===================================================================


class TestRunChecks:
    """Tests for run_checks execution."""

    def test_run_checks_all_pass(self, engine, all_pass_inputs):
        result = engine.run_checks(
            period_id="per-001",
            organisation_id="org-001",
            check_inputs=all_pass_inputs,
        )
        assert result is not None
        assert result.quality_score is not None
        assert result.quality_score.composite_score >= Decimal("90")

    def test_run_checks_returns_all_check_results(self, engine, all_pass_inputs):
        result = engine.run_checks(
            period_id="per-001",
            organisation_id="org-001",
            check_inputs=all_pass_inputs,
        )
        assert result.quality_score.total_checks_run == 19

    def test_run_checks_mixed_results(self, engine, mixed_inputs):
        result = engine.run_checks(
            period_id="per-001",
            organisation_id="org-001",
            check_inputs=mixed_inputs,
        )
        assert result.quality_score.total_failed > 0
        assert result.quality_score.composite_score < Decimal("100")

    def test_run_checks_all_fail(self, engine, all_fail_inputs):
        result = engine.run_checks(
            period_id="per-001",
            organisation_id="org-001",
            check_inputs=all_fail_inputs,
        )
        assert result.quality_score.composite_score < Decimal("50")

    def test_grade_A_all_pass(self, engine, all_pass_inputs):
        result = engine.run_checks("p1", "o1", all_pass_inputs)
        assert result.quality_score.grade == "A"

    def test_grade_below_A_with_failures(self, engine, mixed_inputs):
        result = engine.run_checks("p1", "o1", mixed_inputs)
        assert result.quality_score.grade in ("B", "C", "D")

    def test_grade_F_all_fail(self, engine, all_fail_inputs):
        result = engine.run_checks("p1", "o1", all_fail_inputs)
        assert result.quality_score.grade in ("D", "F")

    def test_dimension_scores_present(self, engine, all_pass_inputs):
        result = engine.run_checks("p1", "o1", all_pass_inputs)
        dims = {ds.dimension for ds in result.quality_score.dimension_scores}
        assert QualityDimension.COMPLETENESS in dims
        assert QualityDimension.ACCURACY in dims

    def test_provenance_hash_on_result(self, engine, all_pass_inputs):
        result = engine.run_checks("p1", "o1", all_pass_inputs)
        assert len(result.provenance_hash) == 64


# ===================================================================
# Single Check Tests
# ===================================================================


class TestSingleCheck:
    """Tests for run_single_check."""

    def test_run_single_check_pass(self, engine):
        result = engine.run_single_check("COMP-001", True)
        assert result.result == CheckResult.PASS

    def test_run_single_check_fail(self, engine):
        result = engine.run_single_check("COMP-001", False)
        assert result.result == CheckResult.FAIL

    def test_run_single_check_unknown_id(self, engine):
        with pytest.raises((KeyError, ValueError, Exception)):
            engine.run_single_check("UNKNOWN-999", True)

    @pytest.mark.parametrize("check_id", [
        "COMP-001", "CONS-001", "ACCU-001", "TRAN-001",
    ])
    def test_single_check_across_dimensions(self, engine, check_id):
        result = engine.run_single_check(check_id, True)
        assert result.result == CheckResult.PASS


# ===================================================================
# Issue Tracking Tests
# ===================================================================


class TestIssueTracking:
    """Tests for create_issue, update_issue_status, list_issues."""

    def test_create_issue(self, engine):
        result = engine.create_issue(
            period_id="per-001",
            title="Missing fugitive emissions data",
            description="Scope 1 fugitive emissions not quantified for 3 facilities",
            dimension=QualityDimension.COMPLETENESS,
            severity=CheckSeverity.MAJOR,
        )
        assert result is not None

    def test_issue_default_open_status(self, engine):
        result = engine.create_issue(
            period_id="per-001",
            title="Test issue",
            description="Desc",
            dimension=QualityDimension.ACCURACY,
            severity=CheckSeverity.MINOR,
        )
        issues = engine.list_issues(period_id="per-001")
        assert len(issues) >= 1
        assert issues[0].status == IssueStatus.OPEN

    def test_update_issue_to_investigating(self, engine):
        engine.create_issue(
            period_id="per-001",
            title="Investigate",
            description="Desc",
            dimension=QualityDimension.CONSISTENCY,
            severity=CheckSeverity.MINOR,
        )
        issues = engine.list_issues(period_id="per-001")
        issue_id = issues[0].issue_id
        engine.update_issue_status(issue_id, IssueStatus.INVESTIGATING)
        updated = engine.list_issues(period_id="per-001")
        found = [i for i in updated if i.issue_id == issue_id]
        assert found[0].status == IssueStatus.INVESTIGATING

    def test_update_issue_to_resolved(self, engine):
        engine.create_issue(
            period_id="per-001",
            title="Resolve me",
            description="Desc",
            dimension=QualityDimension.TRANSPARENCY,
            severity=CheckSeverity.OBSERVATION,
        )
        issues = engine.list_issues(period_id="per-001")
        engine.update_issue_status(issues[0].issue_id, IssueStatus.RESOLVED)
        updated = engine.list_issues(period_id="per-001")
        found = [i for i in updated if i.issue_id == issues[0].issue_id]
        assert found[0].status == IssueStatus.RESOLVED

    def test_list_issues_empty(self, engine):
        issues = engine.list_issues(period_id="per-999")
        assert len(issues) == 0


# ===================================================================
# Action Management Tests
# ===================================================================


class TestActionManagement:
    """Tests for create_action, update_action_status, list_actions."""

    def test_create_action(self, engine):
        result = engine.create_action(
            period_id="per-001",
            title="Install sub-meters on building A",
            description="Replace estimated electricity with metered data",
            priority=ActionPriority.HIGH,
        )
        assert result is not None

    def test_action_default_status(self, engine):
        engine.create_action(
            period_id="per-001",
            title="Action 1",
            description="Desc",
            priority=ActionPriority.MEDIUM,
        )
        actions = engine.list_actions(period_id="per-001")
        assert len(actions) >= 1
        assert actions[0].status == ActionStatus.PLANNED

    def test_update_action_in_progress(self, engine):
        engine.create_action(
            period_id="per-001",
            title="Progress",
            description="Desc",
            priority=ActionPriority.LOW,
        )
        actions = engine.list_actions(period_id="per-001")
        engine.update_action_status(actions[0].action_id, ActionStatus.IN_PROGRESS)
        updated = engine.list_actions(period_id="per-001")
        found = [a for a in updated if a.action_id == actions[0].action_id]
        assert found[0].status == ActionStatus.IN_PROGRESS

    def test_update_action_completed(self, engine):
        engine.create_action(
            period_id="per-001",
            title="Complete",
            description="Desc",
            priority=ActionPriority.CRITICAL,
        )
        actions = engine.list_actions(period_id="per-001")
        engine.update_action_status(actions[0].action_id, ActionStatus.COMPLETED)
        updated = engine.list_actions(period_id="per-001")
        found = [a for a in updated if a.action_id == actions[0].action_id]
        assert found[0].status == ActionStatus.COMPLETED


# ===================================================================
# Verification Readiness Tests
# ===================================================================


class TestVerificationReadiness:
    """Tests for assess_verification_readiness.

    assess_verification_readiness requires (period_id, organisation_id, check_inputs).
    """

    def test_readiness_all_pass(self, engine, all_pass_inputs):
        result = engine.assess_verification_readiness("per-001", "org-001", all_pass_inputs)
        assert result.quality_score.verification_ready is True

    def test_readiness_with_critical_issues(self, engine, all_pass_inputs):
        # First create a critical issue, then assess readiness
        engine.create_issue(
            period_id="per-001",
            title="Critical gap",
            description="Missing entire scope 1",
            dimension=QualityDimension.COMPLETENESS,
            severity=CheckSeverity.CRITICAL,
        )
        result = engine.assess_verification_readiness("per-001", "org-001", all_pass_inputs)
        assert result.quality_score.verification_ready is False

    def test_readiness_with_all_failures(self, engine, all_fail_inputs):
        result = engine.assess_verification_readiness("per-001", "org-001", all_fail_inputs)
        assert result.quality_score.verification_ready is False


# ===================================================================
# Registered Checks Tests
# ===================================================================


class TestRegisteredChecks:
    """Tests for get_registered_checks and register_check."""

    def test_get_registered_checks_default(self, engine):
        checks = engine.get_registered_checks()
        assert len(checks) == 19

    def test_register_custom_check(self, engine):
        custom_check = QAQCCheck(
            check_id="CUSTOM-001",
            name="Custom completeness check",
            dimension=QualityDimension.COMPLETENESS,
            description="Verify custom data source is present",
            severity=CheckSeverity.MINOR,
        )
        engine.register_check(custom_check)
        checks = engine.get_registered_checks()
        assert len(checks) == 20
        custom = [c for c in checks if c.check_id == "CUSTOM-001"]
        assert len(custom) == 1


# ===================================================================
# Improvement Plan Tests
# ===================================================================


class TestImprovementPlan:
    """Tests for generate_improvement_plan.

    generate_improvement_plan requires (period_id, organisation_id).
    """

    def test_generate_plan_with_issues(self, engine, mixed_inputs):
        engine.run_checks("per-001", "org-001", mixed_inputs)
        engine.create_issue(
            period_id="per-001",
            title="Gap 1",
            description="Desc",
            dimension=QualityDimension.ACCURACY,
            severity=CheckSeverity.MAJOR,
        )
        plan = engine.generate_improvement_plan("per-001", "org-001")
        assert plan is not None

    def test_generate_plan_empty_period(self, engine):
        plan = engine.generate_improvement_plan("per-empty", "org-empty")
        assert plan is not None


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults and enum values."""

    @pytest.mark.parametrize("dim", list(QualityDimension))
    def test_quality_dimensions(self, dim):
        assert dim.value is not None

    @pytest.mark.parametrize("sev", list(CheckSeverity))
    def test_check_severities(self, sev):
        assert sev.value is not None

    @pytest.mark.parametrize("result", list(CheckResult))
    def test_check_results(self, result):
        assert result.value is not None

    @pytest.mark.parametrize("status", list(IssueStatus))
    def test_issue_statuses(self, status):
        assert status.value is not None

    @pytest.mark.parametrize("priority", list(ActionPriority))
    def test_action_priorities(self, priority):
        assert priority.value is not None

    @pytest.mark.parametrize("status", list(ActionStatus))
    def test_action_statuses(self, status):
        assert status.value is not None

    def test_qaqc_check_defaults(self):
        c = QAQCCheck()
        assert c.severity == CheckSeverity.MINOR

    def test_quality_score_defaults(self):
        qs = QualityScore()
        assert qs.composite_score == Decimal("0")
        assert qs.grade == "F"

    def test_dimension_score_defaults(self):
        ds = DimensionScore(dimension=QualityDimension.COMPLETENESS)
        assert ds.total_checks == 0
        assert ds.score == Decimal("0")
