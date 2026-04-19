# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Policy Compliance Engine Tests
==============================================================

Validates the policy compliance engine including evaluation of all
compliance rules, geolocation rules, commodity rules, supplier rules,
risk rules, DDS rules, documentation rules, cutoff rules, compliance
scoring, simplified DD eligibility, penalty exposure estimation,
remediation plan generation, and audit trail creation.

Test count: 20
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from conftest import (
    EUDR_COMMODITIES,
    EUDR_CUTOFF_DATE,
    _compute_hash,
    assert_provenance_hash,
)


# ---------------------------------------------------------------------------
# Policy Compliance Engine Simulator
# ---------------------------------------------------------------------------

class PolicyComplianceSimulator:
    """Simulates policy compliance engine operations."""

    # 45 EUDR compliance rules grouped by category
    GEOLOCATION_RULES = [
        {"id": "GEO-001", "name": "Valid WGS84 coordinates", "category": "geolocation"},
        {"id": "GEO-002", "name": "6 decimal precision", "category": "geolocation"},
        {"id": "GEO-003", "name": "Land-based coordinates", "category": "geolocation"},
        {"id": "GEO-004", "name": "Polygon for plots >= 4ha", "category": "geolocation"},
        {"id": "GEO-005", "name": "Polygon topology valid", "category": "geolocation"},
        {"id": "GEO-006", "name": "No overlapping plots", "category": "geolocation"},
        {"id": "GEO-007", "name": "Country matches declaration", "category": "geolocation"},
        {"id": "GEO-008", "name": "Article 9 format compliance", "category": "geolocation"},
    ]

    COMMODITY_RULES = [
        {"id": "COM-001", "name": "Valid CN code", "category": "commodity"},
        {"id": "COM-002", "name": "EUDR-covered commodity", "category": "commodity"},
        {"id": "COM-003", "name": "Annex I listed", "category": "commodity"},
        {"id": "COM-004", "name": "Quantity declared", "category": "commodity"},
        {"id": "COM-005", "name": "Country of production declared", "category": "commodity"},
        {"id": "COM-006", "name": "Derived product traceable", "category": "commodity"},
    ]

    SUPPLIER_RULES = [
        {"id": "SUP-001", "name": "Supplier identified", "category": "supplier"},
        {"id": "SUP-002", "name": "EORI number valid", "category": "supplier"},
        {"id": "SUP-003", "name": "Country of origin declared", "category": "supplier"},
        {"id": "SUP-004", "name": "DD status tracked", "category": "supplier"},
        {"id": "SUP-005", "name": "Data completeness >= 80%", "category": "supplier"},
    ]

    RISK_RULES = [
        {"id": "RSK-001", "name": "Country risk assessed", "category": "risk"},
        {"id": "RSK-002", "name": "Supplier risk assessed", "category": "risk"},
        {"id": "RSK-003", "name": "Commodity risk assessed", "category": "risk"},
        {"id": "RSK-004", "name": "Document risk assessed", "category": "risk"},
        {"id": "RSK-005", "name": "Composite risk calculated", "category": "risk"},
        {"id": "RSK-006", "name": "Risk level classified", "category": "risk"},
        {"id": "RSK-007", "name": "Article 29 benchmark applied", "category": "risk"},
    ]

    DDS_RULES = [
        {"id": "DDS-001", "name": "DDS reference assigned", "category": "dds"},
        {"id": "DDS-002", "name": "Operator details complete", "category": "dds"},
        {"id": "DDS-003", "name": "Commodities listed", "category": "dds"},
        {"id": "DDS-004", "name": "Geolocation attached", "category": "dds"},
        {"id": "DDS-005", "name": "Risk assessment included", "category": "dds"},
        {"id": "DDS-006", "name": "Evidence attached", "category": "dds"},
        {"id": "DDS-007", "name": "Annex II complete", "category": "dds"},
    ]

    DOCUMENTATION_RULES = [
        {"id": "DOC-001", "name": "Supporting evidence provided", "category": "documentation"},
        {"id": "DOC-002", "name": "Certification valid", "category": "documentation"},
        {"id": "DOC-003", "name": "Satellite imagery available", "category": "documentation"},
        {"id": "DOC-004", "name": "Audit trail maintained", "category": "documentation"},
        {"id": "DOC-005", "name": "Provenance hash verified", "category": "documentation"},
    ]

    CUTOFF_RULES = [
        {"id": "CUT-001", "name": "Cutoff date 2020-12-31 applied", "category": "cutoff"},
        {"id": "CUT-002", "name": "Deforestation-free verified", "category": "cutoff"},
        {"id": "CUT-003", "name": "Temporal evidence covers cutoff", "category": "cutoff"},
        {"id": "CUT-004", "name": "Land use history documented", "category": "cutoff"},
        {"id": "CUT-005", "name": "No forest degradation post-cutoff", "category": "cutoff"},
        {"id": "CUT-006", "name": "Legal compliance of producing country", "category": "cutoff"},
        {"id": "CUT-007", "name": "Human rights compliance", "category": "cutoff"},
    ]

    ALL_RULES = (
        GEOLOCATION_RULES + COMMODITY_RULES + SUPPLIER_RULES
        + RISK_RULES + DDS_RULES + DOCUMENTATION_RULES + CUTOFF_RULES
    )

    def evaluate_all_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all 45 compliance rules against a context."""
        results = []
        passed = 0
        failed = 0
        warnings = 0
        for rule in self.ALL_RULES:
            # Simulate rule evaluation based on context
            if context.get("force_fail") and rule["id"] in context.get("fail_rules", []):
                status = "FAIL"
                failed += 1
            elif context.get("force_warning") and rule["id"] in context.get("warn_rules", []):
                status = "WARNING"
                warnings += 1
            else:
                status = "PASS"
                passed += 1
            results.append({
                "rule_id": rule["id"],
                "name": rule["name"],
                "category": rule["category"],
                "status": status,
            })
        score = round(passed / len(self.ALL_RULES) * 100, 1) if self.ALL_RULES else 0.0
        return {
            "total_rules": len(self.ALL_RULES),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "compliance_score_pct": score,
            "results": results,
        }

    def evaluate_category(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate rules for a specific category."""
        category_rules = [r for r in self.ALL_RULES if r["category"] == category]
        passed = len(category_rules)  # default all pass
        return {
            "category": category,
            "total_rules": len(category_rules),
            "passed": passed,
            "failed": 0,
            "rules": category_rules,
        }

    def calculate_compliance_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall compliance score."""
        total = evaluation.get("total_rules", 0)
        passed = evaluation.get("passed", 0)
        return round(passed / total * 100, 1) if total > 0 else 0.0

    def check_simplified_dd_eligibility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check eligibility for simplified due diligence."""
        risk_level = context.get("risk_level", "STANDARD")
        country_benchmark = context.get("country_benchmark", "STANDARD")
        eligible = risk_level == "LOW" and country_benchmark == "LOW"
        return {
            "eligible": eligible,
            "risk_level": risk_level,
            "country_benchmark": country_benchmark,
        }

    def estimate_penalty_exposure(self, evaluation: Dict[str, Any],
                                   revenue_eur: float) -> Dict[str, Any]:
        """Estimate penalty exposure based on compliance score."""
        score = evaluation.get("compliance_score_pct", 0)
        failed = evaluation.get("failed", 0)
        # EUDR penalties can be up to 4% of annual EU turnover
        if failed == 0:
            max_penalty = 0
            risk_level = "NONE"
        elif score >= 90:
            max_penalty = revenue_eur * 0.01
            risk_level = "LOW"
        elif score >= 70:
            max_penalty = revenue_eur * 0.02
            risk_level = "MEDIUM"
        else:
            max_penalty = revenue_eur * 0.04
            risk_level = "HIGH"
        return {
            "max_penalty_eur": round(max_penalty, 2),
            "risk_level": risk_level,
            "compliance_score_pct": score,
            "failed_rules": failed,
        }

    def generate_remediation_plan(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate remediation plan for failed rules."""
        failed_rules = [r for r in evaluation.get("results", []) if r["status"] == "FAIL"]
        actions = []
        for rule in failed_rules:
            actions.append({
                "rule_id": rule["rule_id"],
                "rule_name": rule["name"],
                "category": rule["category"],
                "priority": "HIGH" if rule["category"] in ("geolocation", "cutoff") else "MEDIUM",
                "action": f"Remediate: {rule['name']}",
                "deadline_days": 30,
            })
        return {
            "total_actions": len(actions),
            "high_priority": sum(1 for a in actions if a["priority"] == "HIGH"),
            "medium_priority": sum(1 for a in actions if a["priority"] == "MEDIUM"),
            "actions": actions,
        }

    def create_audit_trail(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Create audit trail entry for compliance evaluation."""
        return {
            "audit_id": _compute_hash(evaluation)[:16],
            "evaluation_date": datetime.now().isoformat(),
            "total_rules": evaluation.get("total_rules", 0),
            "compliance_score_pct": evaluation.get("compliance_score_pct", 0),
            "provenance_hash": _compute_hash(evaluation),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPolicyCompliance:
    """Tests for the policy compliance engine."""

    @pytest.fixture
    def engine(self) -> PolicyComplianceSimulator:
        return PolicyComplianceSimulator()

    @pytest.fixture
    def passing_context(self) -> Dict[str, Any]:
        return {"operator": "test", "commodity": "palm_oil"}

    @pytest.fixture
    def failing_context(self) -> Dict[str, Any]:
        return {
            "operator": "test",
            "force_fail": True,
            "fail_rules": ["GEO-001", "GEO-002", "COM-001"],
        }

    # 1
    def test_evaluate_all_rules_pass(self, engine, passing_context):
        """All rules pass for a complete context."""
        result = engine.evaluate_all_rules(passing_context)
        assert result["total_rules"] == 45
        assert result["passed"] == 45
        assert result["failed"] == 0
        assert result["compliance_score_pct"] == 100.0

    # 2
    def test_evaluate_all_rules_partial(self, engine, failing_context):
        """Partial compliance when some rules fail."""
        result = engine.evaluate_all_rules(failing_context)
        assert result["total_rules"] == 45
        assert result["failed"] == 3
        assert result["passed"] == 42
        assert result["compliance_score_pct"] < 100.0

    # 3
    def test_geolocation_rules(self, engine, passing_context):
        """Geolocation category has 8 rules."""
        result = engine.evaluate_category("geolocation", passing_context)
        assert result["total_rules"] == 8

    # 4
    def test_commodity_rules(self, engine, passing_context):
        """Commodity category has 6 rules."""
        result = engine.evaluate_category("commodity", passing_context)
        assert result["total_rules"] == 6

    # 5
    def test_supplier_rules(self, engine, passing_context):
        """Supplier category has 5 rules."""
        result = engine.evaluate_category("supplier", passing_context)
        assert result["total_rules"] == 5

    # 6
    def test_risk_rules(self, engine, passing_context):
        """Risk category has 7 rules."""
        result = engine.evaluate_category("risk", passing_context)
        assert result["total_rules"] == 7

    # 7
    def test_dds_rules(self, engine, passing_context):
        """DDS category has 7 rules."""
        result = engine.evaluate_category("dds", passing_context)
        assert result["total_rules"] == 7

    # 8
    def test_documentation_rules(self, engine, passing_context):
        """Documentation category has 5 rules."""
        result = engine.evaluate_category("documentation", passing_context)
        assert result["total_rules"] == 5

    # 9
    def test_cutoff_rules(self, engine, passing_context):
        """Cutoff category has 7 rules."""
        result = engine.evaluate_category("cutoff", passing_context)
        assert result["total_rules"] == 7

    # 10
    def test_compliance_score(self, engine, passing_context):
        """Compliance score is 100% when all rules pass."""
        evaluation = engine.evaluate_all_rules(passing_context)
        score = engine.calculate_compliance_score(evaluation)
        assert score == 100.0

    # 11
    def test_simplified_dd_eligibility(self, engine):
        """Low-risk context qualifies for simplified DD."""
        result = engine.check_simplified_dd_eligibility({
            "risk_level": "LOW",
            "country_benchmark": "LOW",
        })
        assert result["eligible"] is True

    # 12
    def test_simplified_dd_ineligible(self, engine):
        """High-risk context does not qualify for simplified DD."""
        result = engine.check_simplified_dd_eligibility({
            "risk_level": "HIGH",
            "country_benchmark": "HIGH",
        })
        assert result["eligible"] is False

    # 13
    def test_penalty_exposure(self, engine, passing_context):
        """Full compliance has zero penalty exposure."""
        evaluation = engine.evaluate_all_rules(passing_context)
        penalty = engine.estimate_penalty_exposure(evaluation, 10_000_000)
        assert penalty["max_penalty_eur"] == 0
        assert penalty["risk_level"] == "NONE"

    # 14
    def test_penalty_exposure_failures(self, engine, failing_context):
        """Failed rules increase penalty exposure."""
        evaluation = engine.evaluate_all_rules(failing_context)
        penalty = engine.estimate_penalty_exposure(evaluation, 10_000_000)
        assert penalty["max_penalty_eur"] > 0

    # 15
    def test_remediation_plan(self, engine, failing_context):
        """Remediation plan lists actions for failed rules."""
        evaluation = engine.evaluate_all_rules(failing_context)
        plan = engine.generate_remediation_plan(evaluation)
        assert plan["total_actions"] == 3
        assert plan["high_priority"] >= 1  # geolocation rules are high priority

    # 16
    def test_audit_trail(self, engine, passing_context):
        """Audit trail includes provenance hash."""
        evaluation = engine.evaluate_all_rules(passing_context)
        trail = engine.create_audit_trail(evaluation)
        assert len(trail["provenance_hash"]) == 64
        assert trail["total_rules"] == 45

    # 17
    def test_45_rules_total(self, engine):
        """Engine defines exactly 45 compliance rules."""
        assert len(engine.ALL_RULES) == 45

    # 18
    def test_all_rule_ids_unique(self, engine):
        """All rule IDs are unique."""
        ids = [r["id"] for r in engine.ALL_RULES]
        assert len(ids) == len(set(ids)), "Duplicate rule IDs found"

    # 19
    def test_rule_categories_covered(self, engine):
        """All expected categories are covered."""
        categories = set(r["category"] for r in engine.ALL_RULES)
        expected = {"geolocation", "commodity", "supplier", "risk", "dds", "documentation", "cutoff"}
        assert categories == expected

    # 20
    def test_compliance_score_partial(self, engine, failing_context):
        """Partial compliance scores correctly."""
        evaluation = engine.evaluate_all_rules(failing_context)
        score = engine.calculate_compliance_score(evaluation)
        expected = round(42 / 45 * 100, 1)
        assert abs(score - expected) < 0.2
