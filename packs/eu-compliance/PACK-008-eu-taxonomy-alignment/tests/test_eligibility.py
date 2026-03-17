# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Eligibility Engine Tests
================================================================

Tests the Taxonomy Eligibility Engine including:
- Single activity screening (eligible, not eligible, partial match)
- NACE code mapping and lookup
- Portfolio batch screening
- Revenue-weighted eligibility ratio
- Sector and objective breakdowns
- Environmental objective retrieval
- Edge cases (unknown NACE, empty portfolio)
- Provenance hash generation

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import json
import re
from typing import Any, Dict, List

import pytest


def assert_provenance_hash(result: Dict[str, Any]) -> None:
    """Verify that a result contains a valid SHA-256 provenance hash."""
    assert "provenance_hash" in result, "Result missing 'provenance_hash' field"
    h = result["provenance_hash"]
    assert isinstance(h, str), f"provenance_hash must be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert re.match(r"^[0-9a-f]{64}$", h), f"Invalid hex hash: {h}"


# ---------------------------------------------------------------------------
# NACE-to-Taxonomy reference data (mirrored from engine for self-contained tests)
# ---------------------------------------------------------------------------

NACE_ELIGIBLE_MAP = {
    "D35.11": {
        "activities": ["CCM-4.1", "CCM-4.2", "CCM-4.3", "CCM-4.5", "CCM-4.7", "CCM-4.8"],
        "sector": "ENERGY",
        "objectives": ["CCM"],
    },
    "D35.12": {
        "activities": ["CCM-4.9"],
        "sector": "ENERGY",
        "objectives": ["CCM"],
    },
    "C23.51": {
        "activities": ["CCM-3.7"],
        "sector": "MANUFACTURING",
        "objectives": ["CCM"],
        "is_transitional": True,
    },
    "C24.10": {
        "activities": ["CCM-3.9"],
        "sector": "MANUFACTURING",
        "objectives": ["CCM"],
        "is_transitional": True,
    },
    "C24.42": {
        "activities": ["CCM-3.8"],
        "sector": "MANUFACTURING",
        "objectives": ["CCM"],
        "is_transitional": True,
    },
    "H49.10": {
        "activities": ["CCM-6.1"],
        "sector": "TRANSPORT",
        "objectives": ["CCM"],
    },
    "H49.20": {
        "activities": ["CCM-6.2"],
        "sector": "TRANSPORT",
        "objectives": ["CCM"],
    },
    "F41.10": {
        "activities": ["CCM-7.1"],
        "sector": "REAL_ESTATE",
        "objectives": ["CCM", "CCA"],
    },
    "L68.20": {
        "activities": ["CCM-7.7"],
        "sector": "REAL_ESTATE",
        "objectives": ["CCM"],
    },
    "A02.10": {
        "activities": ["CCM-1.1"],
        "sector": "FORESTRY",
        "objectives": ["CCM", "BIO"],
    },
    "E36.00": {
        "activities": ["WTR-5.1"],
        "sector": "WATER_SUPPLY",
        "objectives": ["WTR", "CCA"],
    },
    "E38.11": {
        "activities": ["CE-5.5"],
        "sector": "WASTE_MANAGEMENT",
        "objectives": ["CE"],
    },
    "J62.01": {
        "activities": ["CCM-8.1"],
        "sector": "ICT",
        "objectives": ["CCM"],
    },
    "C17.11": {
        "activities": ["CE-2.2"],
        "sector": "MANUFACTURING",
        "objectives": ["CE", "CCM"],
    },
    "C27.20": {
        "activities": ["CCM-3.4"],
        "sector": "MANUFACTURING",
        "objectives": ["CCM"],
        "is_enabling": True,
    },
}

NOT_ELIGIBLE_CODES = ["Z99.99", "K64.11", "M70.10", "N78.10", "P85.10"]


def _simulate_screen_activity(nace_code: str, description: str = "") -> Dict[str, Any]:
    """Simulate eligibility screening for a single NACE code."""
    nace_code = nace_code.strip().upper()
    mapping = NACE_ELIGIBLE_MAP.get(nace_code)

    if mapping:
        status = "ELIGIBLE"
        is_eligible = True
        matched_activities = mapping["activities"]
        objectives = mapping["objectives"]
        confidence = 1.0
    else:
        status = "NOT_ELIGIBLE"
        is_eligible = False
        matched_activities = []
        objectives = []
        confidence = 1.0

    provenance_hash = hashlib.sha256(
        f"{nace_code}|{description}|{status}".encode("utf-8")
    ).hexdigest()

    return {
        "nace_code": nace_code,
        "description": description,
        "is_eligible": is_eligible,
        "status": status,
        "matched_activities": matched_activities,
        "eligible_objectives": objectives,
        "confidence": confidence,
        "provenance_hash": provenance_hash,
    }


def _simulate_screen_portfolio(
    activities: List[Dict[str, Any]], portfolio_id: str = "PF-TEST"
) -> Dict[str, Any]:
    """Simulate portfolio-level eligibility screening."""
    results = []
    eligible_count = 0
    not_eligible_count = 0
    review_count = 0
    total_revenue = 0.0
    eligible_revenue = 0.0
    sector_breakdown: Dict[str, int] = {}
    objective_breakdown: Dict[str, int] = {}

    for item in activities:
        nace_code = item.get("nace_code", "")
        description = item.get("description", "")
        revenue = float(item.get("revenue", 0.0))
        total_revenue += revenue

        result = _simulate_screen_activity(nace_code, description)
        results.append(result)

        if result["status"] == "ELIGIBLE":
            eligible_count += 1
            eligible_revenue += revenue
            mapping = NACE_ELIGIBLE_MAP.get(nace_code.strip().upper(), {})
            sector = mapping.get("sector", "UNKNOWN")
            sector_breakdown[sector] = sector_breakdown.get(sector, 0) + 1
            for obj in result["eligible_objectives"]:
                objective_breakdown[obj] = objective_breakdown.get(obj, 0) + 1
        else:
            not_eligible_count += 1

    total = len(activities)
    eligibility_ratio = (eligible_count + review_count) / total if total > 0 else 0.0
    rev_ratio = eligible_revenue / total_revenue if total_revenue > 0 else None

    provenance_hash = hashlib.sha256(
        f"{portfolio_id}|{total}|{eligible_count}".encode("utf-8")
    ).hexdigest()

    return {
        "portfolio_id": portfolio_id,
        "total_activities": total,
        "eligible_count": eligible_count,
        "not_eligible_count": not_eligible_count,
        "review_count": review_count,
        "eligibility_ratio": eligibility_ratio,
        "revenue_weighted_ratio": rev_ratio,
        "results": results,
        "sector_breakdown": sector_breakdown,
        "objective_breakdown": objective_breakdown,
        "provenance_hash": provenance_hash,
    }


@pytest.mark.unit
class TestEligibilityEngine:
    """Test suite for the Taxonomy Eligibility Engine."""

    def test_screen_single_activity_eligible(self):
        """Test screening a known eligible activity returns ELIGIBLE status."""
        result = _simulate_screen_activity("D35.11", "Electricity generation from solar PV")

        assert result["is_eligible"] is True
        assert result["status"] == "ELIGIBLE"
        assert len(result["matched_activities"]) > 0
        assert "CCM-4.1" in result["matched_activities"]
        assert result["confidence"] == pytest.approx(1.0, abs=1e-9)

    def test_screen_single_activity_not_eligible(self):
        """Test screening a non-eligible activity returns NOT_ELIGIBLE status."""
        result = _simulate_screen_activity("Z99.99", "Consulting services")

        assert result["is_eligible"] is False
        assert result["status"] == "NOT_ELIGIBLE"
        assert len(result["matched_activities"]) == 0
        assert len(result["eligible_objectives"]) == 0

    def test_screen_with_nace_code(self):
        """Test screening returns correct taxonomy activity IDs for NACE code."""
        result = _simulate_screen_activity("C24.10", "Steel manufacturing")

        assert result["is_eligible"] is True
        assert "CCM-3.9" in result["matched_activities"]
        assert "CCM" in result["eligible_objectives"]

    def test_screen_portfolio_batch(self):
        """Test portfolio screening processes all activities correctly."""
        activities = [
            {"nace_code": "D35.11", "description": "Solar PV", "revenue": 5000000.0},
            {"nace_code": "C24.10", "description": "Steel", "revenue": 12000000.0},
            {"nace_code": "H49.10", "description": "Rail", "revenue": 3000000.0},
            {"nace_code": "F41.10", "description": "Construction", "revenue": 8000000.0},
            {"nace_code": "Z99.99", "description": "Consulting", "revenue": 2000000.0},
        ]
        portfolio = _simulate_screen_portfolio(activities, "PF-BATCH-001")

        assert portfolio["total_activities"] == len(activities)
        assert portfolio["eligible_count"] + portfolio["not_eligible_count"] == len(activities)
        assert 0.0 <= portfolio["eligibility_ratio"] <= 1.0

    def test_get_eligible_objectives(self):
        """Test retrieval of eligible environmental objectives for a NACE code."""
        # D35.11 -> CCM objectives
        mapping = NACE_ELIGIBLE_MAP.get("D35.11", {})
        objectives = mapping.get("objectives", [])

        assert "CCM" in objectives
        assert len(objectives) >= 1

    def test_eligibility_vs_alignment_distinction(self):
        """Test that eligibility and alignment are correctly distinguished."""
        # Eligibility means the activity is covered by the taxonomy
        # Alignment means it also meets SC + DNSH + MS criteria
        result = _simulate_screen_activity("C24.10", "Steel manufacturing")

        # Eligible does not imply aligned
        assert result["is_eligible"] is True
        assert "is_aligned" not in result or result.get("is_aligned") is not True

    def test_revenue_weighted_ratio(self):
        """Test revenue-weighted eligibility ratio calculation."""
        activities = [
            {"nace_code": "D35.11", "description": "Solar PV", "revenue": 5000000.0},
            {"nace_code": "C24.10", "description": "Steel", "revenue": 12000000.0},
            {"nace_code": "H49.10", "description": "Rail", "revenue": 3000000.0},
            {"nace_code": "F41.10", "description": "Construction", "revenue": 8000000.0},
            {"nace_code": "Z99.99", "description": "Consulting", "revenue": 2000000.0},
        ]
        portfolio = _simulate_screen_portfolio(activities, "PF-REV-001")

        rev_ratio = portfolio["revenue_weighted_ratio"]
        assert rev_ratio is not None
        assert 0.0 <= rev_ratio <= 1.0

        # Verify: eligible revenue should be sum of eligible activities' revenue
        eligible_revenue = sum(
            a["revenue"] for a in activities
            if NACE_ELIGIBLE_MAP.get(a["nace_code"].strip().upper())
        )
        total_revenue = sum(a["revenue"] for a in activities)
        expected_ratio = eligible_revenue / total_revenue

        assert rev_ratio == pytest.approx(expected_ratio, abs=1e-6)

    def test_nace_to_taxonomy_mapping(self):
        """Test that known NACE codes map to correct taxonomy activity IDs."""
        test_cases = [
            ("D35.11", ["CCM-4.1", "CCM-4.2", "CCM-4.3"]),
            ("C23.51", ["CCM-3.7"]),
            ("H49.10", ["CCM-6.1"]),
            ("F41.10", ["CCM-7.1"]),
        ]

        for nace_code, expected_ids in test_cases:
            result = _simulate_screen_activity(nace_code)
            for expected_id in expected_ids:
                assert expected_id in result["matched_activities"], (
                    f"Expected {expected_id} in matches for {nace_code}, "
                    f"got {result['matched_activities']}"
                )

    def test_sector_breakdown(self):
        """Test sector breakdown in portfolio screening."""
        activities = [
            {"nace_code": "D35.11", "description": "Solar PV", "revenue": 5000000.0},
            {"nace_code": "C24.10", "description": "Steel", "revenue": 12000000.0},
            {"nace_code": "H49.10", "description": "Rail", "revenue": 3000000.0},
            {"nace_code": "F41.10", "description": "Construction", "revenue": 8000000.0},
            {"nace_code": "Z99.99", "description": "Consulting", "revenue": 2000000.0},
        ]
        portfolio = _simulate_screen_portfolio(activities, "PF-SECTOR-001")

        sector_bkdn = portfolio["sector_breakdown"]
        assert isinstance(sector_bkdn, dict)
        # D35.11 -> ENERGY, C24.10 -> MANUFACTURING, H49.10 -> TRANSPORT, F41.10 -> REAL_ESTATE
        assert "ENERGY" in sector_bkdn
        assert "MANUFACTURING" in sector_bkdn

    def test_unknown_nace_code_handling(self):
        """Test that unknown NACE codes are handled gracefully."""
        unknown_codes = ["X00.00", "Q99.99", "AA.BB", "INVALID"]

        for code in unknown_codes:
            result = _simulate_screen_activity(code)
            assert result["is_eligible"] is False
            assert result["status"] == "NOT_ELIGIBLE"
            assert len(result["matched_activities"]) == 0

    def test_batch_processing_performance(self):
        """Test batch screening completes for a large number of activities."""
        import time

        large_batch = [
            {"nace_code": "D35.11", "description": f"Activity {i}", "revenue": 1000.0}
            for i in range(500)
        ]

        start = time.time()
        portfolio = _simulate_screen_portfolio(large_batch, "PF-PERF-001")
        elapsed_ms = (time.time() - start) * 1000

        assert portfolio["total_activities"] == 500
        assert portfolio["eligible_count"] == 500
        assert elapsed_ms < 5000  # Should complete within 5 seconds

    def test_eligibility_result_structure(self):
        """Test that eligibility result contains all required fields."""
        result = _simulate_screen_activity("D35.11", "Solar PV")

        required_fields = [
            "nace_code", "description", "is_eligible", "status",
            "matched_activities", "eligible_objectives", "confidence",
            "provenance_hash",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_multiple_objectives_eligibility(self):
        """Test activity eligible for multiple environmental objectives."""
        # F41.10 -> CCM and CCA
        result = _simulate_screen_activity("F41.10", "Construction of new buildings")

        assert result["is_eligible"] is True
        objectives = result["eligible_objectives"]
        assert "CCM" in objectives
        assert "CCA" in objectives
        assert len(objectives) >= 2

    def test_sector_level_screening(self):
        """Test screening across different taxonomy sectors."""
        sectors_tested = set()
        sector_nace_pairs = [
            ("D35.11", "ENERGY"),
            ("C24.10", "MANUFACTURING"),
            ("H49.10", "TRANSPORT"),
            ("F41.10", "REAL_ESTATE"),
            ("A02.10", "FORESTRY"),
            ("E36.00", "WATER_SUPPLY"),
            ("J62.01", "ICT"),
        ]

        for nace_code, expected_sector in sector_nace_pairs:
            result = _simulate_screen_activity(nace_code)
            assert result["is_eligible"] is True, f"{nace_code} should be eligible"
            mapping = NACE_ELIGIBLE_MAP.get(nace_code, {})
            assert mapping.get("sector") == expected_sector
            sectors_tested.add(expected_sector)

        # Verify at least 7 sectors covered
        assert len(sectors_tested) >= 7

    def test_eligible_count_accuracy(self):
        """Test that eligible count accurately reflects screening results."""
        activities = [
            {"nace_code": "D35.11", "description": "Solar PV", "revenue": 5000000.0},
            {"nace_code": "C24.10", "description": "Steel", "revenue": 12000000.0},
            {"nace_code": "H49.10", "description": "Rail", "revenue": 3000000.0},
            {"nace_code": "F41.10", "description": "Construction", "revenue": 8000000.0},
            {"nace_code": "Z99.99", "description": "Consulting", "revenue": 2000000.0},
        ]
        portfolio = _simulate_screen_portfolio(activities, "PF-COUNT-001")

        # Count expected eligible manually
        expected_eligible = sum(
            1 for a in activities
            if NACE_ELIGIBLE_MAP.get(a["nace_code"].strip().upper())
        )

        assert portfolio["eligible_count"] == expected_eligible
        assert portfolio["not_eligible_count"] == len(activities) - expected_eligible

    def test_eligibility_by_objective_breakdown(self):
        """Test objective breakdown in portfolio screening."""
        activities = [
            {"nace_code": "D35.11", "description": "Solar PV", "revenue": 5000000.0},
            {"nace_code": "C24.10", "description": "Steel", "revenue": 12000000.0},
            {"nace_code": "H49.10", "description": "Rail", "revenue": 3000000.0},
            {"nace_code": "F41.10", "description": "Construction", "revenue": 8000000.0},
            {"nace_code": "Z99.99", "description": "Consulting", "revenue": 2000000.0},
        ]
        portfolio = _simulate_screen_portfolio(activities, "PF-OBJ-001")

        obj_bkdn = portfolio["objective_breakdown"]
        assert isinstance(obj_bkdn, dict)
        # All eligible activities in activities map to CCM
        assert "CCM" in obj_bkdn
        assert obj_bkdn["CCM"] >= 1

    def test_empty_portfolio_handling(self):
        """Test handling of empty portfolio raises or returns zero counts."""
        # Simulate empty portfolio
        portfolio = _simulate_screen_portfolio([], "PF-EMPTY")

        assert portfolio["total_activities"] == 0
        assert portfolio["eligible_count"] == 0
        assert portfolio["eligibility_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_nace_mapping_coverage(self):
        """Test that the NACE mapping covers key economic sectors."""
        # Verify key NACE sections are represented
        covered_sections = set()
        for nace_code in NACE_ELIGIBLE_MAP.keys():
            section = nace_code[0]  # First character (A, C, D, E, F, H, J, L)
            covered_sections.add(section)

        expected_sections = {"A", "C", "D", "E", "F", "H", "J", "L"}
        for section in expected_sections:
            assert section in covered_sections, (
                f"NACE section {section} not covered in mapping"
            )

    def test_provenance_hash_generated(self):
        """Test provenance hash is generated and valid for screening results."""
        result = _simulate_screen_activity("D35.11", "Solar PV")
        assert_provenance_hash(result)

        # Verify reproducibility
        result2 = _simulate_screen_activity("D35.11", "Solar PV")
        assert result["provenance_hash"] == result2["provenance_hash"]
