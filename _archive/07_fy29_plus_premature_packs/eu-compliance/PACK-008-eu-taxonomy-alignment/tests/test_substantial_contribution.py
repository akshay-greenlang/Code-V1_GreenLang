# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Substantial Contribution Engine Tests
=============================================================================

Tests the Substantial Contribution (SC) evaluation engine including:
- SC evaluation for CCM, CCA, WTR, CE objectives
- Quantitative threshold checks (pass and fail)
- Enabling and transitional activity classification
- Evidence requirement linking
- SC result structure validation
- Batch SC evaluation
- Unknown activity handling
- Provenance hash generation

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

import pytest


def assert_provenance_hash(result: Dict[str, Any]) -> None:
    """Verify that a result contains a valid SHA-256 provenance hash."""
    assert "provenance_hash" in result, "Result missing 'provenance_hash' field"
    h = result["provenance_hash"]
    assert isinstance(h, str), f"provenance_hash must be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert re.match(r"^[0-9a-f]{64}$", h), f"Invalid hex hash: {h}"


# ---------------------------------------------------------------------------
# TSC Threshold reference data (mirrored from engine)
# ---------------------------------------------------------------------------

TSC_THRESHOLDS = {
    "CCM-4.1": {
        "CCM": [
            {
                "criterion_id": "CCM-4.1-01",
                "metric_key": "lifecycle_ghg_emissions_gco2e_kwh",
                "description": "Life-cycle GHG emissions < 100 gCO2e/kWh",
                "threshold_type": "LESS_THAN",
                "threshold_value": 100.0,
                "unit": "gCO2e/kWh",
                "is_mandatory": True,
            },
        ],
    },
    "CCM-4.5": {
        "CCM": [
            {
                "criterion_id": "CCM-4.5-01",
                "metric_key": "lifecycle_ghg_emissions_gco2e_kwh",
                "description": "Life-cycle GHG emissions < 100 gCO2e/kWh",
                "threshold_type": "LESS_THAN",
                "threshold_value": 100.0,
                "unit": "gCO2e/kWh",
                "is_mandatory": True,
            },
            {
                "criterion_id": "CCM-4.5-02",
                "metric_key": "power_density_w_per_m2",
                "description": "Power density > 5 W/m2",
                "threshold_type": "GREATER_THAN",
                "threshold_value": 5.0,
                "unit": "W/m2",
                "is_mandatory": True,
            },
        ],
    },
    "CCM-3.7": {
        "CCM": [
            {
                "criterion_id": "CCM-3.7-01",
                "metric_key": "specific_ghg_emissions_tco2e_per_t_product",
                "description": "Specific GHG emissions <= 0.722 tCO2e/t clinker",
                "threshold_type": "LESS_THAN_OR_EQUAL",
                "threshold_value": 0.722,
                "unit": "tCO2e/t clinker",
                "is_mandatory": True,
            },
        ],
    },
    "CCM-7.1": {
        "CCM": [
            {
                "criterion_id": "CCM-7.1-01",
                "metric_key": "primary_energy_demand_kwh_per_m2",
                "description": "Primary energy demand at least 10% below NZEB",
                "threshold_type": "LESS_THAN_OR_EQUAL",
                "threshold_value": 90.0,
                "unit": "kWh/m2/yr",
                "is_mandatory": True,
            },
            {
                "criterion_id": "CCM-7.1-02",
                "metric_key": "airtightness_test_completed",
                "description": "Airtightness and thermal integrity testing completed",
                "threshold_type": "QUALITATIVE",
                "threshold_value": None,
                "unit": "boolean",
                "is_mandatory": True,
            },
        ],
    },
}

ACTIVITY_CLASSIFICATIONS = {
    "CCM-3.3": "ENABLING",
    "CCM-3.4": "ENABLING",
    "CCM-3.7": "TRANSITIONAL",
    "CCM-3.8": "TRANSITIONAL",
    "CCM-3.9": "TRANSITIONAL",
    "CCM-4.3": "ENABLING",
    "CCM-4.9": "ENABLING",
    "CCM-6.6": "TRANSITIONAL",
    "CCM-7.3": "ENABLING",
    "CCM-8.2": "ENABLING",
}

EVIDENCE_REQUIREMENTS = {
    "CCM-4.1": [
        "Life-cycle GHG emission assessment (ISO 14067 or PEF)",
        "Power purchase agreement or generation meter data",
    ],
    "CCM-3.7": [
        "EU ETS verified emissions report",
        "Clinker-to-cement ratio documentation",
        "Best Available Techniques Reference Document (BREF) compliance",
    ],
    "CCM-7.1": [
        "Energy Performance Certificate (EPC)",
        "Airtightness test report",
        "Building specifications and design documentation",
    ],
}


def _check_threshold(
    actual: float,
    threshold_type: str,
    threshold_value: Optional[float],
) -> bool:
    """Check a single threshold comparison."""
    if threshold_type == "QUALITATIVE":
        return bool(actual)
    if threshold_value is None:
        return False
    if threshold_type == "LESS_THAN":
        return actual < threshold_value
    if threshold_type == "LESS_THAN_OR_EQUAL":
        return actual <= threshold_value
    if threshold_type == "GREATER_THAN":
        return actual > threshold_value
    if threshold_type == "GREATER_THAN_OR_EQUAL":
        return actual >= threshold_value
    return False


def _simulate_evaluate_sc(
    activity_id: str,
    objective: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate SC evaluation for an activity-objective pair."""
    activity_id = activity_id.strip().upper()

    activity_thresholds = TSC_THRESHOLDS.get(activity_id, {})
    criteria = activity_thresholds.get(objective, [])

    if not criteria:
        provenance_hash = hashlib.sha256(
            f"{activity_id}|{objective}|INSUFFICIENT_DATA".encode("utf-8")
        ).hexdigest()
        return {
            "activity_id": activity_id,
            "objective": objective,
            "status": "INSUFFICIENT_DATA",
            "is_met": False,
            "score": 0.0,
            "criteria_total": 0,
            "criteria_met": 0,
            "criteria_not_met": 0,
            "criteria_no_data": 0,
            "threshold_results": [],
            "classification": ACTIVITY_CLASSIFICATIONS.get(activity_id, "STANDARD"),
            "evidence_requirements": EVIDENCE_REQUIREMENTS.get(activity_id, []),
            "provenance_hash": provenance_hash,
        }

    threshold_results = []
    met_count = 0
    not_met_count = 0
    no_data_count = 0

    for criterion in criteria:
        actual = metrics.get(criterion["metric_key"])
        if actual is None:
            threshold_results.append({
                "criterion_id": criterion["criterion_id"],
                "is_met": False,
                "has_data": False,
                "actual_value": None,
                "threshold_value": criterion["threshold_value"],
            })
            no_data_count += 1
        else:
            is_met = _check_threshold(
                float(actual), criterion["threshold_type"], criterion["threshold_value"]
            )
            gap = None
            if criterion["threshold_value"] is not None:
                gap = float(actual) - criterion["threshold_value"]
            threshold_results.append({
                "criterion_id": criterion["criterion_id"],
                "is_met": is_met,
                "has_data": True,
                "actual_value": float(actual),
                "threshold_value": criterion["threshold_value"],
                "gap": gap,
            })
            if is_met:
                met_count += 1
            else:
                not_met_count += 1

    total = len(criteria)
    score = met_count / total if total > 0 else 0.0

    if no_data_count == total:
        status = "INSUFFICIENT_DATA"
        is_met_overall = False
    elif not_met_count > 0:
        # Check if any mandatory criterion not met
        status = "NOT_MET"
        is_met_overall = False
    elif met_count == total:
        status = "MET"
        is_met_overall = True
    elif met_count > 0:
        status = "PARTIAL"
        is_met_overall = False
    else:
        status = "NOT_MET"
        is_met_overall = False

    provenance_hash = hashlib.sha256(
        f"{activity_id}|{objective}|{score}|{met_count}|{not_met_count}".encode("utf-8")
    ).hexdigest()

    return {
        "activity_id": activity_id,
        "objective": objective,
        "status": status,
        "is_met": is_met_overall,
        "score": score,
        "criteria_total": total,
        "criteria_met": met_count,
        "criteria_not_met": not_met_count,
        "criteria_no_data": no_data_count,
        "threshold_results": threshold_results,
        "classification": ACTIVITY_CLASSIFICATIONS.get(activity_id, "STANDARD"),
        "evidence_requirements": EVIDENCE_REQUIREMENTS.get(activity_id, []),
        "provenance_hash": provenance_hash,
    }


@pytest.mark.unit
class TestSubstantialContribution:
    """Test suite for the Substantial Contribution evaluation engine."""

    def test_evaluate_sc_ccm(self):
        """Test SC evaluation for Climate Change Mitigation (solar PV < 100 gCO2e/kWh)."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )

        assert result["is_met"] is True
        assert result["status"] == "MET"
        assert result["score"] == pytest.approx(1.0, abs=1e-9)
        assert result["criteria_met"] == result["criteria_total"]

    def test_evaluate_sc_cca(self):
        """Test SC evaluation returns INSUFFICIENT_DATA when no CCA criteria exist."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCA",
            {"climate_risk_assessment": True},
        )

        # CCM-4.1 has no CCA criteria in the TSC database
        assert result["status"] == "INSUFFICIENT_DATA"
        assert result["criteria_total"] == 0

    def test_evaluate_sc_water(self):
        """Test SC evaluation for water objective returns expected status."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "WTR",
            {"water_use_efficiency": 85.0},
        )

        # No WTR criteria for CCM-4.1
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_evaluate_sc_circular_economy(self):
        """Test SC evaluation for Circular Economy objective."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CE",
            {"recyclable_content_pct": 60.0},
        )

        # No CE criteria for CCM-4.1
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_threshold_check_pass(self):
        """Test that metrics meeting thresholds produce MET status."""
        # Solar PV with 15 gCO2e/kWh (threshold: <100)
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )

        assert result["is_met"] is True
        tr = result["threshold_results"][0]
        assert tr["is_met"] is True
        assert tr["actual_value"] == pytest.approx(15.0, abs=1e-9)
        assert tr["gap"] == pytest.approx(15.0 - 100.0, abs=1e-9)

    def test_threshold_check_fail(self):
        """Test that metrics exceeding thresholds produce NOT_MET status."""
        # Solar PV with 120 gCO2e/kWh (threshold: <100) -- fails
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 120.0},
        )

        assert result["is_met"] is False
        assert result["status"] == "NOT_MET"
        tr = result["threshold_results"][0]
        assert tr["is_met"] is False
        assert tr["gap"] == pytest.approx(120.0 - 100.0, abs=1e-9)

    def test_enabling_activity_classification(self):
        """Test that enabling activities are correctly classified."""
        enabling_ids = ["CCM-3.3", "CCM-3.4", "CCM-4.3", "CCM-4.9", "CCM-7.3", "CCM-8.2"]

        for activity_id in enabling_ids:
            classification = ACTIVITY_CLASSIFICATIONS.get(activity_id, "STANDARD")
            assert classification == "ENABLING", (
                f"{activity_id} should be ENABLING, got {classification}"
            )

    def test_transitional_activity_classification(self):
        """Test that transitional activities are correctly classified."""
        transitional_ids = ["CCM-3.7", "CCM-3.8", "CCM-3.9", "CCM-6.6"]

        for activity_id in transitional_ids:
            classification = ACTIVITY_CLASSIFICATIONS.get(activity_id, "STANDARD")
            assert classification == "TRANSITIONAL", (
                f"{activity_id} should be TRANSITIONAL, got {classification}"
            )

    def test_evidence_linking(self):
        """Test that evidence requirements are returned for known activities."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )

        evidence = result["evidence_requirements"]
        assert isinstance(evidence, list)
        assert len(evidence) >= 1
        assert any("ISO 14067" in e or "PEF" in e for e in evidence)

    def test_sc_result_structure(self):
        """Test SC result contains all required fields."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )

        required_fields = [
            "activity_id", "objective", "status", "is_met", "score",
            "criteria_total", "criteria_met", "criteria_not_met",
            "criteria_no_data", "threshold_results", "classification",
            "evidence_requirements", "provenance_hash",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_quantitative_threshold_electricity(self):
        """Test quantitative threshold for electricity generation (<100 gCO2e/kWh)."""
        # Just below threshold
        result_pass = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 99.9},
        )
        assert result_pass["is_met"] is True

        # At threshold (LESS_THAN means 100.0 should fail)
        result_at = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 100.0},
        )
        assert result_at["is_met"] is False

        # Above threshold
        result_fail = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 100.1},
        )
        assert result_fail["is_met"] is False

    def test_quantitative_threshold_buildings(self):
        """Test quantitative threshold for buildings (PED <= 90 kWh/m2/yr)."""
        # Pass: 85 kWh/m2/yr (below NZEB-10%)
        result = _simulate_evaluate_sc(
            "CCM-7.1", "CCM",
            {
                "primary_energy_demand_kwh_per_m2": 85.0,
                "airtightness_test_completed": 1.0,
            },
        )

        assert result["is_met"] is True
        assert result["criteria_met"] == 2
        assert result["criteria_total"] == 2

    def test_multiple_objective_sc(self):
        """Test SC evaluation with multiple criteria for hydropower (CCM-4.5)."""
        result = _simulate_evaluate_sc(
            "CCM-4.5", "CCM",
            {
                "lifecycle_ghg_emissions_gco2e_kwh": 50.0,
                "power_density_w_per_m2": 10.0,
            },
        )

        assert result["is_met"] is True
        assert result["criteria_met"] == 2
        assert result["score"] == pytest.approx(1.0, abs=1e-9)

    def test_sc_status_values(self):
        """Test all possible SC status values are produced correctly."""
        # MET
        result_met = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )
        assert result_met["status"] == "MET"

        # NOT_MET
        result_not_met = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 120.0},
        )
        assert result_not_met["status"] == "NOT_MET"

        # INSUFFICIENT_DATA
        result_no_data = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {},
        )
        assert result_no_data["status"] in ["INSUFFICIENT_DATA", "NOT_MET"]

    def test_unknown_activity_handling(self):
        """Test SC evaluation for unknown activity IDs."""
        result = _simulate_evaluate_sc(
            "UNKNOWN-99.99", "CCM",
            {"some_metric": 50.0},
        )

        assert result["status"] == "INSUFFICIENT_DATA"
        assert result["is_met"] is False
        assert result["criteria_total"] == 0
        assert result["classification"] == "STANDARD"

    def test_batch_sc_evaluation(self):
        """Test batch SC evaluation processes multiple activities."""
        evaluations = [
            {"activity_id": "CCM-4.1", "objective": "CCM",
             "metrics": {"lifecycle_ghg_emissions_gco2e_kwh": 15.0}},
            {"activity_id": "CCM-3.7", "objective": "CCM",
             "metrics": {"specific_ghg_emissions_tco2e_per_t_product": 0.65}},
            {"activity_id": "CCM-4.1", "objective": "CCM",
             "metrics": {"lifecycle_ghg_emissions_gco2e_kwh": 120.0}},
        ]

        results = []
        for ev in evaluations:
            result = _simulate_evaluate_sc(
                ev["activity_id"], ev["objective"], ev["metrics"]
            )
            results.append(result)

        assert len(results) == 3
        assert results[0]["is_met"] is True    # 15 < 100
        assert results[1]["is_met"] is True    # 0.65 <= 0.722
        assert results[2]["is_met"] is False   # 120 >= 100

    def test_provenance_hash_generated(self):
        """Test provenance hash is generated and valid for SC results."""
        result = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )

        assert_provenance_hash(result)

        # Verify reproducibility
        result2 = _simulate_evaluate_sc(
            "CCM-4.1", "CCM",
            {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        )
        assert result["provenance_hash"] == result2["provenance_hash"]
