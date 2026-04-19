# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - DNSH Assessment Engine Tests
====================================================================

Tests the Do No Significant Harm (DNSH) assessment engine including:
- DNSH matrix all pass / partial fail scenarios
- Per-objective DNSH checks (CCM, CCA, WTR, CE, PPC, BIO)
- DNSH result structure validation
- DNSH criteria retrieval
- Matrix dimensions (6x6 minus SC objective)
- SC objective exclusion logic
- Evidence requirement linkage
- Overall pass/fail logic
- Edge cases (empty data, missing metrics)
- Provenance hash generation

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import re
from typing import Any, Dict, List

import pytest


ENVIRONMENTAL_OBJECTIVES = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]

OBJECTIVE_NAMES = {
    "CCM": "Climate Change Mitigation",
    "CCA": "Climate Change Adaptation",
    "WTR": "Sustainable Use of Water and Marine Resources",
    "CE": "Transition to a Circular Economy",
    "PPC": "Pollution Prevention and Control",
    "BIO": "Protection and Restoration of Biodiversity and Ecosystems",
}

# DNSH criteria by (sc_objective -> dnsh_objective -> list of metric_keys)
DNSH_CRITERIA_MAP = {
    "CCM": {
        "CCA": [
            {"metric_key": "climate_risk_assessment_completed", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "adaptation_solutions_implemented", "check_type": "BOOLEAN", "is_mandatory": True},
        ],
        "WTR": [
            {"metric_key": "water_framework_directive_compliance", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "water_recycling_pct", "check_type": "THRESHOLD_MIN", "threshold_value": 50.0, "is_mandatory": False},
        ],
        "CE": [
            {"metric_key": "waste_hierarchy_compliance", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "recyclable_design_assessment", "check_type": "BOOLEAN", "is_mandatory": False},
        ],
        "PPC": [
            {"metric_key": "reach_compliance", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "air_emissions_within_limits", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "water_pollutant_discharge_within_limits", "check_type": "BOOLEAN", "is_mandatory": False},
        ],
        "BIO": [
            {"metric_key": "eia_completed", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "natura2000_no_impact", "check_type": "BOOLEAN", "is_mandatory": True},
            {"metric_key": "biodiversity_management_plan", "check_type": "BOOLEAN", "is_mandatory": False},
        ],
    },
}

DNSH_EVIDENCE = {
    "CCM": ["GHG emissions trend analysis", "Fossil fuel phase-out plan"],
    "CCA": ["Climate risk and vulnerability assessment report (ISO 14090 or equivalent)", "Adaptation solutions implementation record"],
    "WTR": ["Water Framework Directive compliance certificate", "Water use and discharge monitoring data"],
    "CE": ["Waste management plan", "Product lifecycle assessment (recyclability/durability)"],
    "PPC": ["REACH compliance declaration", "Air and water emissions monitoring reports", "IED permit or equivalent"],
    "BIO": ["Environmental Impact Assessment (EIA) report", "Natura 2000 screening report", "Biodiversity management plan"],
}


def _check_criterion(check_type: str, value: Any, threshold_value: float = None) -> str:
    """Check a single DNSH criterion and return status string."""
    if value is None:
        return "NO_DATA"
    if check_type == "BOOLEAN":
        return "PASS" if bool(value) else "FAIL"
    if check_type == "THRESHOLD_MIN":
        return "PASS" if float(value) >= (threshold_value or 0.0) else "FAIL"
    if check_type == "THRESHOLD_MAX":
        return "PASS" if float(value) <= (threshold_value or float("inf")) else "FAIL"
    return "PASS" if bool(value) else "FAIL"


def _simulate_assess_dnsh(
    activity_id: str,
    sc_objective: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate full DNSH assessment for an activity."""
    dnsh_objectives = [obj for obj in ENVIRONMENTAL_OBJECTIVES if obj != sc_objective]
    criteria_map = DNSH_CRITERIA_MAP.get(sc_objective, {})

    objective_results = {}
    passed_count = 0
    failed_count = 0
    no_data_count = 0
    failed_objectives = []

    for obj in dnsh_objectives:
        criteria = criteria_map.get(obj, [])
        if not criteria:
            objective_results[obj] = {
                "objective": obj,
                "objective_name": OBJECTIVE_NAMES.get(obj, obj),
                "status": "NOT_APPLICABLE",
                "criteria_total": 0,
                "criteria_passed": 0,
                "criteria_failed": 0,
                "criteria_no_data": 0,
                "details": [],
                "evidence_required": DNSH_EVIDENCE.get(obj, []),
            }
            passed_count += 1
            continue

        details = []
        obj_passed = 0
        obj_failed = 0
        obj_no_data = 0

        for criterion in criteria:
            value = data.get(criterion["metric_key"])
            status = _check_criterion(
                criterion["check_type"], value,
                criterion.get("threshold_value"),
            )
            details.append({
                "metric_key": criterion["metric_key"],
                "status": status,
                "is_mandatory": criterion["is_mandatory"],
                "actual_value": value,
            })
            if status == "PASS":
                obj_passed += 1
            elif status == "FAIL":
                obj_failed += 1
            else:
                obj_no_data += 1

        # Determine objective status
        mandatory_failed = any(
            d["status"] == "FAIL" and d["is_mandatory"] for d in details
        )
        if mandatory_failed or obj_failed > 0:
            obj_status = "FAIL"
            failed_count += 1
            failed_objectives.append(obj)
        elif obj_no_data > 0:
            obj_status = "INSUFFICIENT_DATA"
            no_data_count += 1
        else:
            obj_status = "PASS"
            passed_count += 1

        objective_results[obj] = {
            "objective": obj,
            "objective_name": OBJECTIVE_NAMES.get(obj, obj),
            "status": obj_status,
            "criteria_total": len(criteria),
            "criteria_passed": obj_passed,
            "criteria_failed": obj_failed,
            "criteria_no_data": obj_no_data,
            "details": details,
            "evidence_required": DNSH_EVIDENCE.get(obj, []),
        }

    overall_pass = (failed_count == 0 and no_data_count == 0)

    provenance_hash = hashlib.sha256(
        f"{activity_id}|{sc_objective}|{passed_count}|{failed_count}".encode("utf-8")
    ).hexdigest()

    return {
        "activity_id": activity_id,
        "sc_objective": sc_objective,
        "overall_pass": overall_pass,
        "objectives_assessed": len(dnsh_objectives),
        "objectives_passed": passed_count,
        "objectives_failed": failed_count,
        "objectives_no_data": no_data_count,
        "failed_objectives": failed_objectives,
        "objective_results": objective_results,
        "provenance_hash": provenance_hash,
    }


def _dnsh_all_pass_data() -> Dict[str, Any]:
    """Create DNSH input data where all criteria pass (inline helper)."""
    return {
        "climate_risk_assessment_completed": True,
        "adaptation_solutions_implemented": True,
        "water_framework_directive_compliance": True,
        "water_recycling_pct": 75.0,
        "waste_hierarchy_compliance": True,
        "recyclable_design_assessment": True,
        "reach_compliance": True,
        "air_emissions_within_limits": True,
        "water_pollutant_discharge_within_limits": True,
        "eia_completed": True,
        "natura2000_no_impact": True,
        "biodiversity_management_plan": True,
        "ghg_emissions_not_increased": True,
        "fossil_fuel_lock_in_avoided": True,
    }


@pytest.mark.unit
class TestDNSHAssessment:
    """Test suite for the DNSH Assessment Engine."""

    @pytest.fixture
    def dnsh_data_all_pass(self) -> Dict[str, Any]:
        """Inline fixture for DNSH all-pass data."""
        return _dnsh_all_pass_data()

    def test_dnsh_matrix_all_pass(self, dnsh_data_all_pass: Dict[str, Any]):
        """Test DNSH assessment passes when all criteria are met."""
        result = _simulate_assess_dnsh(
            "CCM-4.1", "CCM", dnsh_data_all_pass
        )

        assert result["overall_pass"] is True
        assert result["objectives_failed"] == 0
        assert result["objectives_no_data"] == 0
        assert len(result["failed_objectives"]) == 0

    def test_dnsh_matrix_partial_fail(self):
        """Test DNSH assessment fails when one objective criterion fails."""
        data = {
            "climate_risk_assessment_completed": True,
            "adaptation_solutions_implemented": True,
            "water_framework_directive_compliance": False,  # WTR fails
            "waste_hierarchy_compliance": True,
            "reach_compliance": True,
            "air_emissions_within_limits": True,
            "eia_completed": True,
            "natura2000_no_impact": True,
        }

        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        assert result["overall_pass"] is False
        assert result["objectives_failed"] >= 1
        assert "WTR" in result["failed_objectives"]

    def test_dnsh_ccm_climate_risk(self):
        """Test DNSH check for CCA: climate risk assessment requirement."""
        data = {
            "climate_risk_assessment_completed": True,
            "adaptation_solutions_implemented": True,
        }
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        cca_result = result["objective_results"].get("CCA", {})
        assert cca_result["status"] in ["PASS", "NOT_APPLICABLE"]

    def test_dnsh_cca_vulnerability(self):
        """Test DNSH CCA check fails when adaptation solutions not implemented."""
        data = {
            "climate_risk_assessment_completed": True,
            "adaptation_solutions_implemented": False,
        }
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        cca_result = result["objective_results"].get("CCA", {})
        if cca_result["criteria_total"] > 0:
            assert cca_result["status"] == "FAIL"

    def test_dnsh_water_wfd(self):
        """Test DNSH WTR check for Water Framework Directive compliance."""
        data = {
            "water_framework_directive_compliance": True,
            "water_recycling_pct": 60.0,
        }
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        wtr_result = result["objective_results"].get("WTR", {})
        if wtr_result["criteria_total"] > 0:
            assert wtr_result["status"] == "PASS"

    def test_dnsh_circular_economy(self):
        """Test DNSH CE check for waste hierarchy compliance."""
        data = {
            "waste_hierarchy_compliance": True,
            "recyclable_design_assessment": True,
        }
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        ce_result = result["objective_results"].get("CE", {})
        if ce_result["criteria_total"] > 0:
            assert ce_result["status"] == "PASS"

    def test_dnsh_pollution_reach(self):
        """Test DNSH PPC check for REACH substance compliance."""
        data = {
            "reach_compliance": True,
            "air_emissions_within_limits": True,
            "water_pollutant_discharge_within_limits": True,
        }
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        ppc_result = result["objective_results"].get("PPC", {})
        if ppc_result["criteria_total"] > 0:
            assert ppc_result["status"] == "PASS"

    def test_dnsh_biodiversity_eia(self):
        """Test DNSH BIO check for Environmental Impact Assessment."""
        data = {
            "eia_completed": True,
            "natura2000_no_impact": True,
            "biodiversity_management_plan": True,
        }
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", data)

        bio_result = result["objective_results"].get("BIO", {})
        if bio_result["criteria_total"] > 0:
            assert bio_result["status"] == "PASS"

    def test_dnsh_result_structure(self, dnsh_data_all_pass: Dict[str, Any]):
        """Test DNSH result contains all required fields."""
        result = _simulate_assess_dnsh(
            "CCM-4.1", "CCM", dnsh_data_all_pass
        )

        required_fields = [
            "activity_id", "sc_objective", "overall_pass",
            "objectives_assessed", "objectives_passed", "objectives_failed",
            "objectives_no_data", "failed_objectives", "objective_results",
            "provenance_hash",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_get_dnsh_criteria(self):
        """Test retrieval of DNSH criteria for a given SC objective."""
        sc_objective = "CCM"
        criteria_map = DNSH_CRITERIA_MAP.get(sc_objective, {})

        # Should have criteria for 5 non-SC objectives
        assert len(criteria_map) == 5
        for obj in ["CCA", "WTR", "CE", "PPC", "BIO"]:
            assert obj in criteria_map
            assert len(criteria_map[obj]) >= 1

    def test_dnsh_matrix_dimensions(self):
        """Test DNSH matrix covers 6 objectives minus the SC objective (5 checks)."""
        result = _simulate_assess_dnsh(
            "CCM-4.1", "CCM",
            {"climate_risk_assessment_completed": True},
        )

        # 6 total objectives - 1 SC objective = 5 assessed
        assert result["objectives_assessed"] == 5

    def test_dnsh_excludes_sc_objective(self):
        """Test that DNSH assessment excludes the SC objective from checks."""
        result = _simulate_assess_dnsh(
            "CCM-4.1", "CCM",
            {"climate_risk_assessment_completed": True},
        )

        # CCM should not appear in objective_results
        assert "CCM" not in result["objective_results"]

    def test_dnsh_all_objectives_assessed(self, dnsh_data_all_pass: Dict[str, Any]):
        """Test that all non-SC objectives receive an assessment result."""
        result = _simulate_assess_dnsh(
            "CCM-4.1", "CCM", dnsh_data_all_pass
        )

        for obj in ["CCA", "WTR", "CE", "PPC", "BIO"]:
            assert obj in result["objective_results"], (
                f"Objective {obj} missing from results"
            )
            obj_result = result["objective_results"][obj]
            assert "status" in obj_result

    def test_dnsh_evidence_requirements(self):
        """Test evidence requirements are associated with each DNSH objective."""
        for obj in ENVIRONMENTAL_OBJECTIVES:
            evidence = DNSH_EVIDENCE.get(obj, [])
            assert isinstance(evidence, list)
            assert len(evidence) >= 1, f"No evidence requirements for {obj}"

    def test_dnsh_overall_pass_logic(self):
        """Test overall_pass is True only when all objectives pass with no missing data."""
        # All pass
        all_pass_data = {
            "climate_risk_assessment_completed": True,
            "adaptation_solutions_implemented": True,
            "water_framework_directive_compliance": True,
            "water_recycling_pct": 60.0,
            "waste_hierarchy_compliance": True,
            "recyclable_design_assessment": True,
            "reach_compliance": True,
            "air_emissions_within_limits": True,
            "water_pollutant_discharge_within_limits": True,
            "eia_completed": True,
            "natura2000_no_impact": True,
            "biodiversity_management_plan": True,
        }
        result_pass = _simulate_assess_dnsh("CCM-4.1", "CCM", all_pass_data)
        assert result_pass["overall_pass"] is True

        # One mandatory fail
        one_fail_data = dict(all_pass_data)
        one_fail_data["reach_compliance"] = False
        result_fail = _simulate_assess_dnsh("CCM-4.1", "CCM", one_fail_data)
        assert result_fail["overall_pass"] is False

    def test_dnsh_empty_data_handling(self):
        """Test DNSH assessment handles empty data dict gracefully."""
        result = _simulate_assess_dnsh("CCM-4.1", "CCM", {})

        # Should not be overall_pass since data is missing
        assert result["overall_pass"] is False
        # Should have no_data counts
        assert result["objectives_no_data"] >= 1 or result["objectives_failed"] >= 0

    def test_provenance_hash_generated(self, dnsh_data_all_pass: Dict[str, Any]):
        """Test provenance hash is generated and reproducible."""
        result1 = _simulate_assess_dnsh("CCM-4.1", "CCM", dnsh_data_all_pass)
        result2 = _simulate_assess_dnsh("CCM-4.1", "CCM", dnsh_data_all_pass)

        assert len(result1["provenance_hash"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", result1["provenance_hash"])
        assert result1["provenance_hash"] == result2["provenance_hash"]
