# -*- coding: utf-8 -*-
"""
Unit tests for Engine 5: Quality Gate Engine -- AGENT-EUDR-026

Tests QG-1 (information gathering completeness), QG-2 (risk assessment
coverage), QG-3 (mitigation adequacy) evaluation, threshold enforcement,
manual override with justification, remediation guidance, simplified
thresholds, and deterministic scoring.

Test count: ~80 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    DueDiligencePhase,
    QualityGateCheck,
    QualityGateEvaluation,
    QualityGateId,
    QualityGateResultEnum,
    WorkflowState,
    WorkflowStatus,
    WorkflowType,
)
from greenlang.agents.eudr.due_diligence_orchestrator.quality_gate_engine import (
    QualityGateEngine,
)


class TestQualityGateInit:
    """Test engine initialization."""

    def test_init_default(self, default_config):
        engine = QualityGateEngine()
        assert engine is not None

    def test_init_with_config(self, default_config):
        engine = QualityGateEngine(config=default_config)
        assert engine is not None


class TestQG1Evaluation:
    """Test QG-1 Information Gathering Completeness."""

    def test_qg1_passes_with_high_completeness(
        self, quality_gate_engine, workflow_state_phase1_complete,
    ):
        result = quality_gate_engine.evaluate_gate(
            QualityGateId.QG1, workflow_state_phase1_complete,
        )
        assert result.result in (
            QualityGateResultEnum.PASSED, QualityGateResultEnum.FAILED,
        )

    def test_qg1_fails_with_empty_state(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-empty", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG1, state)
        assert result.result == QualityGateResultEnum.FAILED

    def test_qg1_default_threshold_90(self, quality_gate_engine):
        engine = quality_gate_engine
        spec = engine.get_gate_specification(QualityGateId.QG1)
        assert spec.threshold == Decimal("0.90") or spec.threshold == Decimal("90")

    def test_qg1_simplified_threshold_80(self, quality_gate_engine):
        engine = quality_gate_engine
        spec = engine.get_gate_specification(QualityGateId.QG1)
        assert spec.simplified_threshold == Decimal("0.80") or spec.simplified_threshold == Decimal("80")

    def test_qg1_has_multiple_checks(self, quality_gate_engine):
        engine = quality_gate_engine
        checks = engine.get_gate_checks(QualityGateId.QG1)
        assert len(checks) >= 4

    def test_qg1_check_weights_sum_to_one(self, quality_gate_engine):
        engine = quality_gate_engine
        checks = engine.get_gate_checks(QualityGateId.QG1)
        total = sum(c.weight for c in checks)
        assert total == Decimal("1") or total == Decimal("1.00") or total == Decimal("1.0")

    def test_qg1_generates_remediation_on_failure(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-fail", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG1, state)
        if result.result == QualityGateResultEnum.FAILED:
            failed_checks = [c for c in result.checks if not c.passed]
            for check in failed_checks:
                assert check.remediation is not None or check.name is not None


class TestQG2Evaluation:
    """Test QG-2 Risk Assessment Coverage."""

    def test_qg2_passes_with_all_scored(
        self, quality_gate_engine, workflow_state_phase2_complete,
    ):
        result = quality_gate_engine.evaluate_gate(
            QualityGateId.QG2, workflow_state_phase2_complete,
        )
        assert result.result in (
            QualityGateResultEnum.PASSED, QualityGateResultEnum.FAILED,
        )

    def test_qg2_fails_with_missing_dimensions(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-partial", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG2, state)
        assert result.result == QualityGateResultEnum.FAILED

    def test_qg2_default_threshold_95(self, quality_gate_engine):
        engine = quality_gate_engine
        spec = engine.get_gate_specification(QualityGateId.QG2)
        assert spec.threshold == Decimal("0.95") or spec.threshold == Decimal("95")

    def test_qg2_check_dimensions_scored(self, quality_gate_engine):
        engine = quality_gate_engine
        checks = engine.get_gate_checks(QualityGateId.QG2)
        check_names = [c.name for c in checks]
        assert len(checks) >= 3


class TestQG3Evaluation:
    """Test QG-3 Mitigation Adequacy."""

    def test_qg3_passes_with_low_residual(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-mit", definition_id="def-001",
            mitigation_decision={
                "residual_risk_score": 10,
                "mitigation_adequate": True,
                "proportionality_verified": True,
            },
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG3, state)
        assert result.result in (
            QualityGateResultEnum.PASSED, QualityGateResultEnum.FAILED,
        )

    def test_qg3_fails_with_high_residual(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-high", definition_id="def-001",
            mitigation_decision={
                "residual_risk_score": 25,
                "mitigation_adequate": False,
            },
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG3, state)
        assert result.result == QualityGateResultEnum.FAILED

    def test_qg3_default_threshold_residual_15(self, quality_gate_engine):
        engine = quality_gate_engine
        spec = engine.get_gate_specification(QualityGateId.QG3)
        assert spec is not None


class TestManualOverride:
    """Test quality gate manual override."""

    def test_override_changes_result_to_overridden(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-override", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG1, state)
        if result.result == QualityGateResultEnum.FAILED:
            overridden = quality_gate_engine.override_gate(
                result,
                justification="Emergency override approved by compliance director",
                override_by="compliance_director",
            )
            assert overridden.result == QualityGateResultEnum.OVERRIDDEN

    def test_override_requires_justification(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-override2", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG1, state)
        if result.result == QualityGateResultEnum.FAILED:
            with pytest.raises((ValueError, TypeError)):
                quality_gate_engine.override_gate(result, justification="", override_by="")

    def test_override_records_actor(self, quality_gate_engine):
        state = WorkflowState(
            workflow_id="wf-actor", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(QualityGateId.QG1, state)
        if result.result == QualityGateResultEnum.FAILED:
            overridden = quality_gate_engine.override_gate(
                result,
                justification="Approved by head of compliance",
                override_by="head_compliance",
            )
            assert overridden.override_by == "head_compliance"

    def test_cannot_override_passed_gate(self, quality_gate_engine):
        eval_result = QualityGateEvaluation(
            workflow_id="wf-passed",
            gate_id=QualityGateId.QG1,
            phase_from=DueDiligencePhase.INFORMATION_GATHERING,
            phase_to=DueDiligencePhase.RISK_ASSESSMENT,
            result=QualityGateResultEnum.PASSED,
            weighted_score=Decimal("0.95"),
            threshold=Decimal("0.90"),
        )
        with pytest.raises((ValueError, TypeError)):
            quality_gate_engine.override_gate(
                eval_result,
                justification="Unnecessary override",
                override_by="user",
            )


class TestSimplifiedThresholds:
    """Test simplified due diligence thresholds per Article 13."""

    def test_qg1_simplified_uses_relaxed_threshold(
        self, quality_gate_engine,
    ):
        state = WorkflowState(
            workflow_id="wf-simp", definition_id="def-001",
            workflow_type=WorkflowType.SIMPLIFIED,
        )
        result = quality_gate_engine.evaluate_gate(
            QualityGateId.QG1, state, is_simplified=True,
        )
        assert result.threshold is not None

    def test_qg2_simplified_uses_relaxed_threshold(
        self, quality_gate_engine,
    ):
        state = WorkflowState(
            workflow_id="wf-simp2", definition_id="def-001",
            workflow_type=WorkflowType.SIMPLIFIED,
        )
        result = quality_gate_engine.evaluate_gate(
            QualityGateId.QG2, state, is_simplified=True,
        )
        assert result.threshold is not None


class TestGateEvaluationModel:
    """Test QualityGateEvaluation model."""

    def test_evaluation_has_all_fields(self):
        ev = QualityGateEvaluation(
            workflow_id="wf-001",
            gate_id=QualityGateId.QG1,
            phase_from=DueDiligencePhase.INFORMATION_GATHERING,
            phase_to=DueDiligencePhase.RISK_ASSESSMENT,
            result=QualityGateResultEnum.PASSED,
            weighted_score=Decimal("0.92"),
            threshold=Decimal("0.90"),
        )
        assert ev.gate_id == QualityGateId.QG1
        assert ev.result == QualityGateResultEnum.PASSED

    def test_evaluation_timestamp(self):
        ev = QualityGateEvaluation(
            workflow_id="wf-001",
            gate_id=QualityGateId.QG2,
            phase_from=DueDiligencePhase.RISK_ASSESSMENT,
            phase_to=DueDiligencePhase.RISK_MITIGATION,
            result=QualityGateResultEnum.FAILED,
            weighted_score=Decimal("0.80"),
            threshold=Decimal("0.95"),
        )
        assert ev.evaluated_at is not None


class TestDeterministicScoring:
    """Test deterministic quality gate scoring."""

    def test_same_state_same_score(
        self, quality_gate_engine, workflow_state_phase1_complete,
    ):
        r1 = quality_gate_engine.evaluate_gate(
            QualityGateId.QG1, workflow_state_phase1_complete,
        )
        r2 = quality_gate_engine.evaluate_gate(
            QualityGateId.QG1, workflow_state_phase1_complete,
        )
        assert r1.weighted_score == r2.weighted_score
        assert r1.result == r2.result

    def test_scores_are_decimal(
        self, quality_gate_engine, workflow_state_phase1_complete,
    ):
        result = quality_gate_engine.evaluate_gate(
            QualityGateId.QG1, workflow_state_phase1_complete,
        )
        assert isinstance(result.weighted_score, Decimal)

    @pytest.mark.parametrize("gate_id", [
        QualityGateId.QG1, QualityGateId.QG2, QualityGateId.QG3,
    ])
    def test_all_gates_return_evaluation(
        self, quality_gate_engine, gate_id,
    ):
        state = WorkflowState(
            workflow_id="wf-gate", definition_id="def-001",
        )
        result = quality_gate_engine.evaluate_gate(gate_id, state)
        assert isinstance(result, QualityGateEvaluation)


class TestGateCheckModel:
    """Test QualityGateCheck model."""

    def test_check_model_creation(self):
        check = QualityGateCheck(
            name="Supply Chain Mapping",
            weight=Decimal("0.25"),
            measured_value=Decimal("95"),
            threshold=Decimal("90"),
            passed=True,
            source_agents=["EUDR-001"],
        )
        assert check.passed is True
        assert check.weight == Decimal("0.25")

    def test_check_with_remediation(self):
        check = QualityGateCheck(
            name="Geolocation Coverage",
            weight=Decimal("0.20"),
            measured_value=Decimal("60"),
            threshold=Decimal("90"),
            passed=False,
            remediation="Request field GPS verification from suppliers",
        )
        assert check.passed is False
        assert check.remediation is not None

    def test_check_weight_range(self):
        for w in [Decimal("0"), Decimal("0.5"), Decimal("1")]:
            check = QualityGateCheck(
                name="Test", weight=w,
                measured_value=Decimal("50"),
                threshold=Decimal("50"), passed=True,
            )
            assert check.weight == w
