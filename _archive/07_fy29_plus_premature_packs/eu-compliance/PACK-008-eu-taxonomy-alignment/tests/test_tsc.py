# -*- coding: utf-8 -*-
"""
Unit tests for PACK-008 EU Taxonomy Alignment Pack - Technical Screening Criteria Engine

Tests TSC lookup by activity and objective, quantitative threshold evaluation,
qualitative criteria checks, Delegated Act versioning, criteria change tracking,
gap identification, and database coverage.
"""

import pytest
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Simulated TSC Enumerations and Models
# ---------------------------------------------------------------------------

ENVIRONMENTAL_OBJECTIVES = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]

DELEGATED_ACTS = {
    "EU_2021_2139": {
        "da_id": "EU_2021_2139",
        "version": "2021-06-04",
        "publication_date": "2021-12-09",
        "effective_date": "2022-01-01",
        "description": "Climate Delegated Act (Mitigation & Adaptation)",
    },
    "EU_2023_2486": {
        "da_id": "EU_2023_2486",
        "version": "2023-06-27",
        "publication_date": "2023-11-21",
        "effective_date": "2024-01-01",
        "description": "Environmental Delegated Act (WTR, CE, PPC, BIO)",
    },
    "EU_2022_1214": {
        "da_id": "EU_2022_1214",
        "version": "2022-03-09",
        "publication_date": "2022-07-15",
        "effective_date": "2023-01-01",
        "description": "Complementary DA (nuclear/gas)",
    },
}

# Representative TSC database used by the simulated engine
TSC_DATABASE = {
    "4.1": [
        {
            "criterion_id": "TSC-4.1-CCM-01",
            "activity_id": "4.1",
            "activity_name": "Electricity generation (low-carbon)",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh",
            "metric": "gco2e_kwh",
            "unit": "gCO2e/kWh",
            "operator": "LT",
            "threshold": 100.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 4.1",
        },
    ],
    "7.1": [
        {
            "criterion_id": "TSC-7.1-CCM-01",
            "activity_id": "7.1",
            "activity_name": "Construction of new buildings",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Primary energy demand 10% below NZEB threshold",
            "metric": "ped_kwh_m2",
            "unit": "kWh/m2/year",
            "operator": "LT",
            "threshold": 90.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 7.1",
        },
        {
            "criterion_id": "TSC-7.1-CCM-02",
            "activity_id": "7.1",
            "activity_name": "Construction of new buildings",
            "objective": "CCM",
            "criterion_type": "QUALITATIVE",
            "description": "Building must have air-tightness and thermal integrity tested",
            "metric": None,
            "unit": None,
            "operator": None,
            "threshold": None,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 7.1",
        },
    ],
    "7.2": [
        {
            "criterion_id": "TSC-7.2-CCM-01",
            "activity_id": "7.2",
            "activity_name": "Renovation of existing buildings",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "At least 30% reduction in primary energy demand",
            "metric": "ped_reduction_pct",
            "unit": "%",
            "operator": "GE",
            "threshold": 30.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 7.2",
        },
    ],
    "3.9": [
        {
            "criterion_id": "TSC-3.9-CCM-01",
            "activity_id": "3.9",
            "activity_name": "Iron and steel manufacturing",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Steel production emissions below 1.331 tCO2e/t",
            "metric": "tco2e_per_t",
            "unit": "tCO2e/t",
            "operator": "LT",
            "threshold": 1.331,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 3.9",
        },
    ],
}


# ---------------------------------------------------------------------------
# Simulated Engine
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _compare(operator: str, actual: float, threshold: float,
             threshold_upper: Optional[float] = None) -> bool:
    """Evaluate a comparison operator."""
    if operator == "LT":
        return actual < threshold
    elif operator == "LE":
        return actual <= threshold
    elif operator == "GT":
        return actual > threshold
    elif operator == "GE":
        return actual >= threshold
    elif operator == "EQ":
        return actual == threshold
    elif operator == "BETWEEN":
        return threshold <= actual <= (threshold_upper or threshold)
    return False


class SimulatedTSCEngine:
    """Simulated Technical Screening Criteria Engine for testing."""

    def __init__(self):
        self.criteria_db = TSC_DATABASE
        self.da_versions = DELEGATED_ACTS

    def get_criteria(self, activity_id: str, objective: str) -> List[Dict[str, Any]]:
        """Retrieve TSC for an activity and objective."""
        raw = self.criteria_db.get(activity_id, [])
        return [c for c in raw if c["objective"] == objective]

    def evaluate_criteria(self, activity_id: str, objective: str,
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all TSC for a given activity + objective against data."""
        start = datetime.utcnow()
        criteria = self.get_criteria(activity_id, objective)
        if not criteria:
            raise ValueError(
                f"No TSC found for activity {activity_id} / {objective}"
            )

        results = []
        gaps = []
        for criterion in criteria:
            evaluation = self._evaluate_single(criterion, data)
            results.append(evaluation)
            if not evaluation["passed"]:
                gaps.append(evaluation)

        passed_count = sum(1 for r in results if r["passed"])
        overall_pass = passed_count == len(results)

        da_key = criteria[0]["da_id"]
        da_version = self.da_versions.get(da_key)

        provenance = _compute_hash({
            "activity_id": activity_id,
            "objective": objective,
            "data_keys": sorted(data.keys()),
            "ts": start.isoformat(),
        })

        return {
            "activity_id": activity_id,
            "objective": objective,
            "overall_pass": overall_pass,
            "criteria_results": results,
            "total_criteria": len(results),
            "passed_criteria": passed_count,
            "gaps": gaps,
            "da_version": da_version,
            "evaluation_date": start.isoformat(),
            "provenance_hash": provenance,
        }

    def get_da_version(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Determine the applicable DA version for an activity."""
        criteria_list = self.criteria_db.get(activity_id, [])
        if not criteria_list:
            return None
        da_key = criteria_list[0]["da_id"]
        return self.da_versions.get(da_key)

    def get_all_activities(self) -> List[str]:
        """Return all activity IDs present in the database."""
        return sorted(self.criteria_db.keys())

    def get_criteria_changes(self, activity_id: str, from_da: str,
                             to_da: str) -> Dict[str, Any]:
        """Track criteria changes between two DA versions."""
        all_criteria = self.criteria_db.get(activity_id, [])
        from_criteria = [c for c in all_criteria if c["da_id"] == from_da]
        to_criteria = [c for c in all_criteria if c["da_id"] == to_da]

        from_ids = {c["criterion_id"] for c in from_criteria}
        to_ids = {c["criterion_id"] for c in to_criteria}

        return {
            "activity_id": activity_id,
            "from_da": from_da,
            "to_da": to_da,
            "added": sorted(to_ids - from_ids),
            "removed": sorted(from_ids - to_ids),
            "modified": [],
        }

    def _evaluate_single(self, criterion: Dict[str, Any],
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single criterion."""
        ctype = criterion["criterion_type"]

        if ctype == "QUANTITATIVE":
            metric = criterion.get("metric")
            if metric is None or metric not in data:
                return {
                    "criterion_id": criterion["criterion_id"],
                    "passed": False,
                    "actual_value": None,
                    "threshold_value": criterion.get("threshold"),
                    "unit": criterion.get("unit"),
                    "gap": None,
                    "message": f"Missing metric '{metric}' in input data",
                }
            actual = float(data[metric])
            threshold = criterion["threshold"]
            operator = criterion["operator"]
            passed = _compare(operator, actual, threshold)
            gap = actual - threshold if not passed else None
            return {
                "criterion_id": criterion["criterion_id"],
                "passed": passed,
                "actual_value": actual,
                "threshold_value": threshold,
                "unit": criterion.get("unit"),
                "gap": gap,
                "message": f"{'PASS' if passed else 'FAIL'}: {actual} {operator} {threshold}",
            }
        else:
            # Qualitative: check for a boolean flag in data
            key = criterion.get("metric") or criterion["criterion_id"]
            passed = bool(data.get(key, False))
            return {
                "criterion_id": criterion["criterion_id"],
                "passed": passed,
                "actual_value": None,
                "threshold_value": None,
                "unit": None,
                "gap": None,
                "message": f"Qualitative check {'PASS' if passed else 'FAIL'}",
            }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tsc_engine():
    """Create a simulated TSC engine."""
    return SimulatedTSCEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTechnicalScreeningCriteria:
    """Test suite for TechnicalScreeningCriteriaEngine."""

    def test_get_criteria_electricity_ccm(self, tsc_engine):
        """Test retrieving TSC for electricity generation (4.1) under CCM."""
        criteria = tsc_engine.get_criteria("4.1", "CCM")

        assert len(criteria) >= 1
        first = criteria[0]
        assert first["criterion_id"] == "TSC-4.1-CCM-01"
        assert first["objective"] == "CCM"
        assert first["criterion_type"] == "QUANTITATIVE"
        assert first["metric"] == "gco2e_kwh"
        assert first["threshold"] == pytest.approx(100.0)
        assert first["operator"] == "LT"
        assert first["da_id"] == "EU_2021_2139"

    def test_get_criteria_building_renovation(self, tsc_engine):
        """Test retrieving TSC for building renovation (7.2) under CCM."""
        criteria = tsc_engine.get_criteria("7.2", "CCM")

        assert len(criteria) >= 1
        first = criteria[0]
        assert first["criterion_id"] == "TSC-7.2-CCM-01"
        assert first["objective"] == "CCM"
        assert first["metric"] == "ped_reduction_pct"
        assert first["operator"] == "GE"
        assert first["threshold"] == pytest.approx(30.0)
        assert first["unit"] == "%"

    def test_evaluate_criteria_pass(self, tsc_engine):
        """Test evaluation where all criteria pass (4.1 electricity below 100 gCO2e/kWh)."""
        result = tsc_engine.evaluate_criteria(
            activity_id="4.1",
            objective="CCM",
            data={"gco2e_kwh": 80.0}
        )

        assert result["overall_pass"] is True
        assert result["passed_criteria"] == result["total_criteria"]
        assert result["total_criteria"] >= 1
        assert len(result["gaps"]) == 0
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_evaluate_criteria_fail(self, tsc_engine):
        """Test evaluation where criteria fail (4.1 electricity above 100 gCO2e/kWh)."""
        result = tsc_engine.evaluate_criteria(
            activity_id="4.1",
            objective="CCM",
            data={"gco2e_kwh": 150.0}
        )

        assert result["overall_pass"] is False
        assert result["passed_criteria"] < result["total_criteria"]
        assert len(result["gaps"]) >= 1

        gap = result["gaps"][0]
        assert gap["passed"] is False
        assert gap["actual_value"] == pytest.approx(150.0)
        assert gap["threshold_value"] == pytest.approx(100.0)
        assert gap["gap"] is not None
        assert gap["gap"] == pytest.approx(50.0)  # 150 - 100

    def test_get_da_version(self, tsc_engine):
        """Test retrieving Delegated Act version for a known activity."""
        da_version = tsc_engine.get_da_version("4.1")

        assert da_version is not None
        assert da_version["da_id"] == "EU_2021_2139"
        assert da_version["version"] == "2021-06-04"
        assert da_version["effective_date"] == "2022-01-01"
        assert "Climate" in da_version["description"]

    def test_quantitative_threshold_check(self, tsc_engine):
        """Test quantitative threshold evaluation for building renovation (GE 30% PED reduction)."""
        # Pass case: 35% reduction >= 30% threshold
        result_pass = tsc_engine.evaluate_criteria(
            activity_id="7.2",
            objective="CCM",
            data={"ped_reduction_pct": 35.0}
        )
        assert result_pass["overall_pass"] is True

        # Fail case: 20% reduction < 30% threshold
        result_fail = tsc_engine.evaluate_criteria(
            activity_id="7.2",
            objective="CCM",
            data={"ped_reduction_pct": 20.0}
        )
        assert result_fail["overall_pass"] is False

    def test_qualitative_criteria_check(self, tsc_engine):
        """Test qualitative criteria evaluation for building construction (7.1)."""
        # Activity 7.1 has both quantitative and qualitative criteria
        # Pass when quantitative is met and qualitative flag is true
        result = tsc_engine.evaluate_criteria(
            activity_id="7.1",
            objective="CCM",
            data={
                "ped_kwh_m2": 80.0,  # Below 90 threshold
                "TSC-7.1-CCM-02": True,  # Qualitative criterion flag
            }
        )

        assert result["overall_pass"] is True
        assert result["total_criteria"] == 2
        assert result["passed_criteria"] == 2

    def test_criteria_change_tracking(self, tsc_engine):
        """Test tracking criteria changes between Delegated Act versions."""
        changes = tsc_engine.get_criteria_changes(
            activity_id="4.1",
            from_da="EU_2021_2139",
            to_da="EU_2023_2486"
        )

        assert changes is not None
        assert changes["activity_id"] == "4.1"
        assert changes["from_da"] == "EU_2021_2139"
        assert changes["to_da"] == "EU_2023_2486"
        assert "added" in changes
        assert "removed" in changes
        assert "modified" in changes
        assert isinstance(changes["added"], list)
        assert isinstance(changes["removed"], list)

    def test_gap_identification(self, tsc_engine):
        """Test gap identification for partially compliant activities."""
        result = tsc_engine.evaluate_criteria(
            activity_id="7.1",
            objective="CCM",
            data={
                "ped_kwh_m2": 80.0,   # Quantitative: PASS (< 90)
                # TSC-7.1-CCM-02 not set: Qualitative: FAIL
            }
        )

        assert result["overall_pass"] is False
        assert len(result["gaps"]) >= 1
        gap_ids = [g["criterion_id"] for g in result["gaps"]]
        assert "TSC-7.1-CCM-02" in gap_ids

    def test_tsc_database_coverage(self, tsc_engine):
        """Test the TSC database covers multiple activities and sectors."""
        activities = tsc_engine.get_all_activities()

        assert len(activities) >= 4  # At least 4 in our simulated DB
        assert "4.1" in activities  # Electricity generation
        assert "7.1" in activities  # Building construction
        assert "7.2" in activities  # Building renovation
        assert "3.9" in activities  # Iron and steel

    def test_multiple_da_versions(self, tsc_engine):
        """Test that multiple Delegated Act versions are available."""
        assert "EU_2021_2139" in tsc_engine.da_versions
        assert "EU_2023_2486" in tsc_engine.da_versions
        assert "EU_2022_1214" in tsc_engine.da_versions

        climate_da = tsc_engine.da_versions["EU_2021_2139"]
        env_da = tsc_engine.da_versions["EU_2023_2486"]

        assert climate_da["effective_date"] < env_da["effective_date"]

    def test_unknown_activity_criteria(self, tsc_engine):
        """Test that unknown activity raises ValueError on evaluation."""
        # get_criteria returns empty list
        criteria = tsc_engine.get_criteria("99.99", "CCM")
        assert criteria == []

        # evaluate_criteria raises ValueError
        with pytest.raises(ValueError, match="No TSC found"):
            tsc_engine.evaluate_criteria(
                activity_id="99.99",
                objective="CCM",
                data={"some_metric": 42.0}
            )
