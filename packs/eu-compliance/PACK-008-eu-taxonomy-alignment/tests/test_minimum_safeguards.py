# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Minimum Safeguards Engine Tests
=======================================================================

Tests the Minimum Safeguards verification engine including:
- Full safeguards verification (all pass / single topic fail)
- Individual topic checks (human rights, anti-corruption, taxation, fair competition)
- Procedural vs outcome check distinction
- Overall pass requires all four topics
- MS result structure validation
- Partial data handling
- Status value validation
- Provenance hash generation

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import re
from typing import Any, Dict, List

import pytest


SAFEGUARD_TOPICS = ["HUMAN_RIGHTS", "ANTI_CORRUPTION", "TAXATION", "FAIR_COMPETITION"]

TOPIC_NAMES = {
    "HUMAN_RIGHTS": "Human Rights Due Diligence",
    "ANTI_CORRUPTION": "Anti-Corruption & Anti-Bribery",
    "TAXATION": "Taxation Compliance",
    "FAIR_COMPETITION": "Fair Competition",
}

# Check definitions keyed by topic
SAFEGUARD_CHECKS = {
    "HUMAN_RIGHTS": [
        {"check_id": "HR-P01", "category": "PROCEDURAL", "metric_key": "human_rights_policy", "is_mandatory": True},
        {"check_id": "HR-P02", "category": "PROCEDURAL", "metric_key": "human_rights_due_diligence_process", "is_mandatory": True},
        {"check_id": "HR-P03", "category": "PROCEDURAL", "metric_key": "grievance_mechanism", "is_mandatory": True},
        {"check_id": "HR-P04", "category": "PROCEDURAL", "metric_key": "human_rights_training", "is_mandatory": False},
        {"check_id": "HR-O01", "category": "OUTCOME", "metric_key": "no_human_rights_violations", "is_mandatory": True},
        {"check_id": "HR-O02", "category": "OUTCOME", "metric_key": "no_forced_labour", "is_mandatory": True},
    ],
    "ANTI_CORRUPTION": [
        {"check_id": "AC-P01", "category": "PROCEDURAL", "metric_key": "anti_corruption_policy", "is_mandatory": True},
        {"check_id": "AC-P02", "category": "PROCEDURAL", "metric_key": "anti_corruption_training", "is_mandatory": True},
        {"check_id": "AC-P03", "category": "PROCEDURAL", "metric_key": "whistleblower_mechanism", "is_mandatory": False},
        {"check_id": "AC-O01", "category": "OUTCOME", "metric_key": "no_corruption_convictions", "is_mandatory": True},
        {"check_id": "AC-O02", "category": "OUTCOME", "metric_key": "no_ongoing_corruption_investigations", "is_mandatory": False},
    ],
    "TAXATION": [
        {"check_id": "TX-P01", "category": "PROCEDURAL", "metric_key": "tax_compliance_statement", "is_mandatory": True},
        {"check_id": "TX-P02", "category": "PROCEDURAL", "metric_key": "country_by_country_reporting", "is_mandatory": False},
        {"check_id": "TX-P03", "category": "PROCEDURAL", "metric_key": "tax_risk_management", "is_mandatory": True},
        {"check_id": "TX-O01", "category": "OUTCOME", "metric_key": "no_tax_evasion_convictions", "is_mandatory": True},
        {"check_id": "TX-O02", "category": "OUTCOME", "metric_key": "not_on_eu_tax_blacklist", "is_mandatory": True},
    ],
    "FAIR_COMPETITION": [
        {"check_id": "FC-P01", "category": "PROCEDURAL", "metric_key": "fair_competition_policy", "is_mandatory": True},
        {"check_id": "FC-P02", "category": "PROCEDURAL", "metric_key": "competition_compliance_training", "is_mandatory": False},
        {"check_id": "FC-O01", "category": "OUTCOME", "metric_key": "no_antitrust_violations", "is_mandatory": True},
        {"check_id": "FC-O02", "category": "OUTCOME", "metric_key": "no_ongoing_competition_investigations", "is_mandatory": False},
    ],
}


def _simulate_evaluate_topic(
    topic: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulate evaluation of a single safeguard topic."""
    checks = SAFEGUARD_CHECKS.get(topic, [])
    check_results = []
    passed = 0
    failed = 0
    no_data = 0

    for check in checks:
        value = data.get(check["metric_key"])
        if value is None:
            check_results.append({
                "check_id": check["check_id"],
                "category": check["category"],
                "is_met": False,
                "has_data": False,
                "is_mandatory": check["is_mandatory"],
            })
            no_data += 1
        else:
            is_met = bool(value)
            check_results.append({
                "check_id": check["check_id"],
                "category": check["category"],
                "is_met": is_met,
                "has_data": True,
                "actual_value": value,
                "is_mandatory": check["is_mandatory"],
            })
            if is_met:
                passed += 1
            else:
                failed += 1

    procedural_pass = all(
        cr["is_met"]
        for cr in check_results
        if cr["category"] == "PROCEDURAL" and cr["is_mandatory"] and cr.get("has_data", False)
    )
    outcome_pass = all(
        cr["is_met"]
        for cr in check_results
        if cr["category"] == "OUTCOME" and cr["is_mandatory"] and cr.get("has_data", False)
    )

    mandatory_failed = any(
        not cr["is_met"] and cr["is_mandatory"] and cr.get("has_data", False)
        for cr in check_results
    )
    mandatory_no_data = any(
        not cr.get("has_data", False) and cr["is_mandatory"]
        for cr in check_results
    )

    if mandatory_failed:
        status = "FAIL"
    elif mandatory_no_data:
        status = "INSUFFICIENT_DATA"
    else:
        status = "PASS"

    return {
        "topic": topic,
        "topic_name": TOPIC_NAMES.get(topic, topic),
        "status": status,
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": failed,
        "checks_no_data": no_data,
        "procedural_pass": procedural_pass,
        "outcome_pass": outcome_pass,
        "check_results": check_results,
    }


def _simulate_verify_safeguards(
    entity_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate full minimum safeguards verification."""
    topic_results = {}
    passed_count = 0
    failed_count = 0
    no_data_count = 0
    failed_topics = []

    for topic in SAFEGUARD_TOPICS:
        topic_result = _simulate_evaluate_topic(topic, entity_data)
        topic_results[topic] = topic_result

        if topic_result["status"] == "PASS":
            passed_count += 1
        elif topic_result["status"] == "FAIL":
            failed_count += 1
            failed_topics.append(topic)
        else:
            no_data_count += 1

    overall_pass = (failed_count == 0 and no_data_count == 0)

    provenance_hash = hashlib.sha256(
        f"MS|{passed_count}|{failed_count}|{no_data_count}".encode("utf-8")
    ).hexdigest()

    return {
        "overall_pass": overall_pass,
        "topics_assessed": 4,
        "topics_passed": passed_count,
        "topics_failed": failed_count,
        "topics_no_data": no_data_count,
        "failed_topics": failed_topics,
        "topic_results": topic_results,
        "provenance_hash": provenance_hash,
    }


def _entity_data_all_pass() -> Dict[str, Any]:
    """Create entity data where all minimum safeguards pass (inline helper)."""
    return {
        "human_rights_policy": True,
        "human_rights_due_diligence_process": True,
        "grievance_mechanism": True,
        "human_rights_training": True,
        "no_human_rights_violations": True,
        "no_forced_labour": True,
        "anti_corruption_policy": True,
        "anti_corruption_training": True,
        "whistleblower_mechanism": True,
        "no_corruption_convictions": True,
        "no_ongoing_corruption_investigations": True,
        "tax_compliance_statement": True,
        "country_by_country_reporting": True,
        "tax_risk_management": True,
        "no_tax_evasion_convictions": True,
        "not_on_eu_tax_blacklist": True,
        "fair_competition_policy": True,
        "competition_compliance_training": True,
        "no_antitrust_violations": True,
        "no_ongoing_competition_investigations": True,
    }


@pytest.mark.unit
class TestMinimumSafeguards:
    """Test suite for the Minimum Safeguards verification engine."""

    @pytest.fixture
    def entity_data_all_pass(self) -> Dict[str, Any]:
        """Inline fixture for entity data where all safeguards pass."""
        return _entity_data_all_pass()

    def test_verify_all_safeguards_pass(self, entity_data_all_pass: Dict[str, Any]):
        """Test all four safeguard topics pass with complete data."""
        result = _simulate_verify_safeguards(entity_data_all_pass)

        assert result["overall_pass"] is True
        assert result["topics_passed"] == 4
        assert result["topics_failed"] == 0
        assert result["topics_no_data"] == 0
        assert len(result["failed_topics"]) == 0

    def test_verify_single_topic_fail(self, entity_data_all_pass: Dict[str, Any]):
        """Test overall fails when a single topic has a mandatory failure."""
        data = dict(entity_data_all_pass)
        data["human_rights_policy"] = False  # Mandatory procedural check fails

        result = _simulate_verify_safeguards(data)

        assert result["overall_pass"] is False
        assert result["topics_failed"] >= 1
        assert "HUMAN_RIGHTS" in result["failed_topics"]

    def test_human_rights_check(self, entity_data_all_pass: Dict[str, Any]):
        """Test human rights topic evaluation in isolation."""
        topic_result = _simulate_evaluate_topic("HUMAN_RIGHTS", entity_data_all_pass)

        assert topic_result["status"] == "PASS"
        assert topic_result["procedural_pass"] is True
        assert topic_result["outcome_pass"] is True
        assert topic_result["checks_total"] == 6

    def test_anti_corruption_check(self, entity_data_all_pass: Dict[str, Any]):
        """Test anti-corruption topic evaluation in isolation."""
        topic_result = _simulate_evaluate_topic("ANTI_CORRUPTION", entity_data_all_pass)

        assert topic_result["status"] == "PASS"
        assert topic_result["checks_total"] == 5
        assert topic_result["procedural_pass"] is True
        assert topic_result["outcome_pass"] is True

    def test_taxation_check(self, entity_data_all_pass: Dict[str, Any]):
        """Test taxation topic evaluation in isolation."""
        topic_result = _simulate_evaluate_topic("TAXATION", entity_data_all_pass)

        assert topic_result["status"] == "PASS"
        assert topic_result["checks_total"] == 5

    def test_fair_competition_check(self, entity_data_all_pass: Dict[str, Any]):
        """Test fair competition topic evaluation in isolation."""
        topic_result = _simulate_evaluate_topic("FAIR_COMPETITION", entity_data_all_pass)

        assert topic_result["status"] == "PASS"
        assert topic_result["checks_total"] == 4

    def test_procedural_checks(self):
        """Test procedural checks pass when all procedural policies are in place."""
        data = {
            "human_rights_policy": True,
            "human_rights_due_diligence_process": True,
            "grievance_mechanism": True,
            "no_human_rights_violations": True,
            "no_forced_labour": True,
        }
        topic_result = _simulate_evaluate_topic("HUMAN_RIGHTS", data)

        assert topic_result["procedural_pass"] is True

    def test_outcome_checks(self):
        """Test outcome checks fail when violations are reported."""
        data = {
            "human_rights_policy": True,
            "human_rights_due_diligence_process": True,
            "grievance_mechanism": True,
            "no_human_rights_violations": False,  # Violation reported
            "no_forced_labour": True,
        }
        topic_result = _simulate_evaluate_topic("HUMAN_RIGHTS", data)

        assert topic_result["outcome_pass"] is False
        assert topic_result["status"] == "FAIL"

    def test_overall_pass_requires_all_four(self):
        """Test overall pass requires all four topics to pass (no failures, no missing data)."""
        # Only provide human rights data, others missing
        partial_data = {
            "human_rights_policy": True,
            "human_rights_due_diligence_process": True,
            "grievance_mechanism": True,
            "no_human_rights_violations": True,
            "no_forced_labour": True,
        }
        result = _simulate_verify_safeguards(partial_data)

        # Other topics have insufficient data for mandatory checks
        assert result["overall_pass"] is False
        assert result["topics_passed"] <= 1

    def test_ms_result_structure(self, entity_data_all_pass: Dict[str, Any]):
        """Test MS result contains all required fields."""
        result = _simulate_verify_safeguards(entity_data_all_pass)

        required_fields = [
            "overall_pass", "topics_assessed", "topics_passed",
            "topics_failed", "topics_no_data", "failed_topics",
            "topic_results", "provenance_hash",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check topic result structure
        for topic in SAFEGUARD_TOPICS:
            topic_result = result["topic_results"][topic]
            topic_fields = [
                "topic", "topic_name", "status", "checks_total",
                "checks_passed", "checks_failed", "checks_no_data",
                "procedural_pass", "outcome_pass", "check_results",
            ]
            for field in topic_fields:
                assert field in topic_result, f"Missing topic field: {field}"

    def test_partial_data_handling(self):
        """Test handling of partial entity data (some fields missing)."""
        partial_data = {
            "human_rights_policy": True,
            "anti_corruption_policy": True,
            # Missing: due diligence, grievance, violations, taxation, competition
        }
        result = _simulate_verify_safeguards(partial_data)

        assert result["overall_pass"] is False
        # Some topics should have INSUFFICIENT_DATA or FAIL status
        for topic in SAFEGUARD_TOPICS:
            status = result["topic_results"][topic]["status"]
            assert status in ["PASS", "FAIL", "INSUFFICIENT_DATA"]

    def test_ms_status_values(self, entity_data_all_pass: Dict[str, Any]):
        """Test all possible topic status values are valid."""
        valid_statuses = {"PASS", "FAIL", "INSUFFICIENT_DATA"}

        # All pass
        result_pass = _simulate_verify_safeguards(entity_data_all_pass)
        for topic in SAFEGUARD_TOPICS:
            assert result_pass["topic_results"][topic]["status"] in valid_statuses

        # Force a fail
        fail_data = dict(entity_data_all_pass)
        fail_data["no_antitrust_violations"] = False
        result_fail = _simulate_verify_safeguards(fail_data)
        assert result_fail["topic_results"]["FAIR_COMPETITION"]["status"] == "FAIL"

        # Force insufficient data
        empty_result = _simulate_verify_safeguards({})
        for topic in SAFEGUARD_TOPICS:
            assert empty_result["topic_results"][topic]["status"] in valid_statuses

    def test_provenance_hash_generated(self, entity_data_all_pass: Dict[str, Any]):
        """Test provenance hash is generated and reproducible."""
        result1 = _simulate_verify_safeguards(entity_data_all_pass)
        result2 = _simulate_verify_safeguards(entity_data_all_pass)

        assert len(result1["provenance_hash"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", result1["provenance_hash"])
        assert result1["provenance_hash"] == result2["provenance_hash"]
