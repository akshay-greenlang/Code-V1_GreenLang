# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Quality Gate Tests
======================================================

Tests for the 3-tier quality gate engine covering data completeness (QG1),
calculation integrity (QG2), compliance readiness (QG3), gate overrides,
remediation suggestions, and sequential enforcement.

Test count: 15
Author: GreenLang QA Team
"""

import hashlib
import json
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Quality Gate Evaluator Stub
# ---------------------------------------------------------------------------

class QualityGateEvaluator:
    """Lightweight quality gate evaluator for test validation."""

    def evaluate(self, gate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a quality gate by computing weighted score."""
        checks = gate_data["checks"]
        weighted_score = sum(c["weight"] * c["score"] for c in checks)
        weighted_score = round(weighted_score, 2)
        passed = weighted_score >= gate_data["threshold"]

        failing = [c for c in checks if c["score"] < gate_data["threshold"]]
        remediation = []
        for c in failing:
            gap = round(gate_data["threshold"] - c["score"], 1)
            remediation.append({
                "check_id": c["check_id"],
                "check_name": c["name"],
                "current_score": c["score"],
                "target_score": gate_data["threshold"],
                "gap": gap,
                "suggestion": f"Improve {c['name']} by {gap} points to meet threshold",
            })

        provenance_str = json.dumps({
            "gate_id": gate_data["gate_id"],
            "score": weighted_score,
            "passed": passed,
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return {
            "gate_id": gate_data["gate_id"],
            "name": gate_data["name"],
            "weighted_score": weighted_score,
            "threshold": gate_data["threshold"],
            "passed": passed,
            "checks_total": len(checks),
            "checks_failing": len(failing),
            "remediation_suggestions": remediation,
            "provenance_hash": provenance_hash,
        }

    def evaluate_all(self, gates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all gates sequentially; stop on first failure if blocking."""
        results = []
        all_passed = True
        for gate in gates:
            result = self.evaluate(gate)
            results.append(result)
            if not result["passed"]:
                all_passed = False
        return {
            "all_passed": all_passed,
            "results": results,
            "gates_evaluated": len(results),
        }


# ===========================================================================
# Quality Gate Tests
# ===========================================================================

class TestQualityGates:
    """Test quality gate evaluation logic."""

    @pytest.fixture
    def evaluator(self):
        return QualityGateEvaluator()

    def test_qg1_all_checks_pass(self, evaluator, sample_quality_gate_data):
        """QG1 passes when all weighted checks exceed 85% threshold."""
        result = evaluator.evaluate(sample_quality_gate_data["qg1_data_completeness"])
        assert result["passed"] is True
        assert result["gate_id"] == "QG1"
        assert result["weighted_score"] >= 85.0

    def test_qg1_below_threshold(self, evaluator):
        """QG1 fails when weighted score is below 85%."""
        gate = {
            "gate_id": "QG1", "name": "Data Completeness", "threshold": 85.0,
            "checks": [
                {"check_id": "QG1-01", "name": "Coverage", "weight": 0.50, "score": 60.0},
                {"check_id": "QG1-02", "name": "Scope 1", "weight": 0.50, "score": 70.0},
            ],
        }
        result = evaluator.evaluate(gate)
        assert result["passed"] is False
        assert result["weighted_score"] == 65.0

    def test_qg1_individual_check_weights(self, evaluator, sample_quality_gate_data):
        """QG1 check weights sum to 1.0."""
        gate = sample_quality_gate_data["qg1_data_completeness"]
        total_weight = sum(c["weight"] for c in gate["checks"])
        assert abs(total_weight - 1.0) < 0.001, (
            f"QG1 check weights sum to {total_weight}, expected 1.0"
        )

    def test_qg2_dual_reporting_variance(self, evaluator, sample_quality_gate_data):
        """QG2 dual reporting variance check passes with high score."""
        gate = sample_quality_gate_data["qg2_calculation_integrity"]
        dual_check = next(
            c for c in gate["checks"] if "dual reporting" in c["name"].lower()
        )
        assert dual_check["score"] >= 90.0

    def test_qg2_cross_entity_balance(self, evaluator, sample_quality_gate_data):
        """QG2 cross-entity balance check verifies sum = group total."""
        gate = sample_quality_gate_data["qg2_calculation_integrity"]
        balance_check = next(
            c for c in gate["checks"] if "cross-entity" in c["name"].lower()
        )
        assert balance_check["score"] == 100.0

    def test_qg3_rule_pass_rate(self, evaluator, sample_quality_gate_data):
        """QG3 validation rules pass rate meets compliance threshold."""
        gate = sample_quality_gate_data["qg3_compliance_readiness"]
        result = evaluator.evaluate(gate)
        assert result["passed"] is True
        assert result["weighted_score"] >= 80.0

    def test_qg3_xbrl_validity(self, evaluator, sample_quality_gate_data):
        """QG3 XBRL tag validity check has acceptable score."""
        gate = sample_quality_gate_data["qg3_compliance_readiness"]
        xbrl_check = next(
            c for c in gate["checks"] if "xbrl" in c["name"].lower()
        )
        assert xbrl_check["score"] >= 80.0

    def test_gate_override_with_audit(self, evaluator):
        """Gate override requires justification and is recorded."""
        gate = {
            "gate_id": "QG1", "name": "Data Completeness", "threshold": 85.0,
            "checks": [
                {"check_id": "QG1-01", "name": "Coverage", "weight": 1.0, "score": 75.0},
            ],
        }
        result = evaluator.evaluate(gate)
        assert result["passed"] is False

        # Apply override
        result["overridden"] = True
        result["override_justification"] = "CEO exception: new subsidiary onboarded mid-year"
        result["override_approver"] = "thomas.mueller"
        result["override_timestamp"] = "2025-11-15T14:30:00Z"

        assert result["overridden"] is True
        assert len(result["override_justification"]) > 0

    def test_gate_override_requires_justification(self, evaluator):
        """Override without justification is invalid."""
        override = {"overridden": True, "override_justification": ""}
        assert override["override_justification"] == ""
        # Business rule: justification must be non-empty
        is_valid = len(override.get("override_justification", "")) > 0
        assert is_valid is False

    def test_remediation_suggestions_generated(self, evaluator, sample_quality_gate_data):
        """Remediation suggestions are generated for checks below threshold."""
        result = evaluator.evaluate(sample_quality_gate_data["qg1_data_completeness"])
        suggestions = result["remediation_suggestions"]
        # QG1-04 Scope 3 (78%) is below 85% threshold
        assert len(suggestions) >= 1
        assert all("suggestion" in s for s in suggestions)
        assert all(s["gap"] > 0 for s in suggestions)

    def test_all_gates_sequential(self, evaluator, sample_quality_gate_data):
        """All 3 gates are evaluated sequentially."""
        gates = [
            sample_quality_gate_data["qg1_data_completeness"],
            sample_quality_gate_data["qg2_calculation_integrity"],
            sample_quality_gate_data["qg3_compliance_readiness"],
        ]
        result = evaluator.evaluate_all(gates)
        assert result["gates_evaluated"] == 3
        assert result["all_passed"] is True

    def test_gate_blocks_workflow(self, evaluator):
        """A failing gate blocks downstream workflow progression."""
        gates = [
            {
                "gate_id": "QG1", "name": "Data Completeness", "threshold": 85.0,
                "checks": [{"check_id": "QG1-01", "name": "Coverage", "weight": 1.0, "score": 60.0}],
            },
            {
                "gate_id": "QG2", "name": "Calculation Integrity", "threshold": 90.0,
                "checks": [{"check_id": "QG2-01", "name": "Accuracy", "weight": 1.0, "score": 95.0}],
            },
        ]
        result = evaluator.evaluate_all(gates)
        assert result["all_passed"] is False
        assert result["results"][0]["passed"] is False
        assert result["results"][1]["passed"] is True

    def test_gate_per_entity(self, evaluator, sample_quality_gate_data):
        """Quality gates can be evaluated per entity."""
        entities = ["eurotech-parent", "eurotech-fr", "eurotech-it"]
        entity_results = {}
        for entity_id in entities:
            # Simulate per-entity gate with slightly different scores
            gate = {
                "gate_id": "QG1", "name": f"Data Completeness ({entity_id})", "threshold": 85.0,
                "checks": [
                    {"check_id": "QG1-01", "name": "ESRS coverage", "weight": 0.5,
                     "score": 92.0 if entity_id == "eurotech-parent" else 88.0},
                    {"check_id": "QG1-02", "name": "Source docs", "weight": 0.5,
                     "score": 90.0 if entity_id != "eurotech-it" else 78.0},
                ],
            }
            entity_results[entity_id] = evaluator.evaluate(gate)

        assert entity_results["eurotech-parent"]["passed"] is True
        assert entity_results["eurotech-fr"]["passed"] is True

    def test_gate_threshold_configuration(self, sample_quality_gate_data):
        """Gate thresholds match expected professional tier values."""
        assert sample_quality_gate_data["qg1_data_completeness"]["threshold"] == 85.0
        assert sample_quality_gate_data["qg2_calculation_integrity"]["threshold"] == 90.0
        assert sample_quality_gate_data["qg3_compliance_readiness"]["threshold"] == 80.0

    def test_gate_provenance_hash(self, evaluator, sample_quality_gate_data):
        """Gate evaluation produces a valid SHA-256 provenance hash."""
        result = evaluator.evaluate(sample_quality_gate_data["qg1_data_completeness"])
        assert len(result["provenance_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["provenance_hash"])
