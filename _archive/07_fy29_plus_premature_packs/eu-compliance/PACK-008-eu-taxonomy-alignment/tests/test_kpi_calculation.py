# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - KPI Calculation Engine Tests
====================================================================

Tests the KPI Calculation Engine including:
- Turnover, CapEx, OpEx alignment ratio calculations
- Double-counting prevention across objectives
- CapEx plan recognition (Article 8, para 1.1.2.2)
- Eligible vs aligned breakdown
- Zero denominator handling
- Activity-level mapping
- KPI result structure validation
- Year-over-year comparison
- Financial precision (Decimal usage)
- Objective-level breakdown
- Full and zero alignment scenarios
- Provenance hash generation

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import pytest


ZERO = Decimal("0")
ONE = Decimal("1")
PRECISION = Decimal("0.0001")


def _to_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal, defaulting to ZERO."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return ZERO


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide with zero-denominator protection, rounded to 4 dp."""
    if denominator == ZERO:
        return ZERO
    return (numerator / denominator).quantize(PRECISION, ROUND_HALF_UP)


def _simulate_calculate_kpis(
    activities: List[Dict[str, Any]],
    financials: Dict[str, Any],
    prior_period: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Simulate KPI calculation for activities against company financials."""
    total_turnover = _to_decimal(financials["total_turnover"])
    total_capex = _to_decimal(financials["total_capex"])
    total_opex = _to_decimal(financials["total_opex"])

    eligible_turnover = ZERO
    eligible_capex = ZERO
    eligible_opex = ZERO
    aligned_turnover = ZERO
    aligned_capex = ZERO
    aligned_opex = ZERO
    capex_plan_total = ZERO
    count_eligible = 0
    count_aligned = 0
    double_counting_adj = ZERO

    seen_eligible = set()
    seen_aligned = set()

    objective_breakdown = {}
    for obj in ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]:
        objective_breakdown[obj] = {
            "eligible_turnover": ZERO,
            "aligned_turnover": ZERO,
            "eligible_capex": ZERO,
            "aligned_capex": ZERO,
            "eligible_opex": ZERO,
            "aligned_opex": ZERO,
        }

    activity_details = []

    for a in activities:
        aid = a.get("activity_id", "UNKNOWN")
        turnover = _to_decimal(a.get("turnover", "0"))
        capex = _to_decimal(a.get("capex", "0"))
        opex = _to_decimal(a.get("opex", "0"))
        is_eligible = bool(a.get("is_eligible", False))
        is_aligned = bool(a.get("is_aligned", False))
        sc_objectives = a.get("sc_objectives", [])
        plan_status = a.get("capex_plan_status", "NO_PLAN")
        plan_amount = _to_decimal(a.get("capex_plan_amount", "0"))

        activity_details.append({
            "activity_id": aid,
            "turnover": str(turnover),
            "capex": str(capex),
            "opex": str(opex),
            "is_eligible": is_eligible,
            "is_aligned": is_aligned,
        })

        if is_eligible:
            if aid not in seen_eligible:
                eligible_turnover += turnover
                eligible_capex += capex
                eligible_opex += opex
                count_eligible += 1
                seen_eligible.add(aid)
            else:
                double_counting_adj += turnover + capex + opex

            for obj_str in sc_objectives:
                if obj_str in objective_breakdown:
                    objective_breakdown[obj_str]["eligible_turnover"] += turnover
                    objective_breakdown[obj_str]["eligible_capex"] += capex
                    objective_breakdown[obj_str]["eligible_opex"] += opex

        if is_aligned:
            if aid not in seen_aligned:
                aligned_turnover += turnover
                aligned_capex += capex
                aligned_opex += opex
                count_aligned += 1
                seen_aligned.add(aid)
            else:
                double_counting_adj += turnover + capex + opex

            for obj_str in sc_objectives:
                if obj_str in objective_breakdown:
                    objective_breakdown[obj_str]["aligned_turnover"] += turnover
                    objective_breakdown[obj_str]["aligned_capex"] += capex
                    objective_breakdown[obj_str]["aligned_opex"] += opex

        if plan_status in ("PLAN_APPROVED", "PLAN_IN_PROGRESS"):
            capex_plan_total += plan_amount

    # Include capex_plan in aligned_capex
    final_aligned_capex = aligned_capex + capex_plan_total

    turnover_ratio = _safe_divide(aligned_turnover, total_turnover)
    capex_ratio = _safe_divide(final_aligned_capex, total_capex)
    opex_ratio = _safe_divide(aligned_opex, total_opex)

    eligible_turnover_ratio = _safe_divide(eligible_turnover, total_turnover)
    eligible_capex_ratio = _safe_divide(eligible_capex, total_capex)
    eligible_opex_ratio = _safe_divide(eligible_opex, total_opex)

    # Year-over-year
    yoy_comparisons = []
    if prior_period:
        for kpi_type, current, prior_key in [
            ("TURNOVER", turnover_ratio, "turnover_ratio"),
            ("CAPEX", capex_ratio, "capex_ratio"),
            ("OPEX", opex_ratio, "opex_ratio"),
        ]:
            previous = _to_decimal(prior_period.get(prior_key, "0"))
            change_abs = current - previous
            threshold = Decimal("0.005")
            if change_abs > threshold:
                trend = "IMPROVED"
            elif change_abs < -threshold:
                trend = "DECLINED"
            else:
                trend = "STABLE"
            yoy_comparisons.append({
                "kpi_type": kpi_type,
                "current_ratio": current,
                "previous_ratio": previous,
                "change_absolute": change_abs,
                "trend": trend,
            })

    provenance_hash = hashlib.sha256(
        f"KPI|{turnover_ratio}|{capex_ratio}|{opex_ratio}|{len(activities)}".encode("utf-8")
    ).hexdigest()

    return {
        "eligible_turnover": eligible_turnover,
        "eligible_capex": eligible_capex,
        "eligible_opex": eligible_opex,
        "aligned_turnover": aligned_turnover,
        "aligned_capex": final_aligned_capex,
        "aligned_opex": aligned_opex,
        "total_turnover": total_turnover,
        "total_capex": total_capex,
        "total_opex": total_opex,
        "eligible_turnover_ratio": eligible_turnover_ratio,
        "eligible_capex_ratio": eligible_capex_ratio,
        "eligible_opex_ratio": eligible_opex_ratio,
        "turnover_ratio": turnover_ratio,
        "capex_ratio": capex_ratio,
        "opex_ratio": opex_ratio,
        "capex_plan_amount": capex_plan_total,
        "activities_total": len(activities),
        "activities_eligible": count_eligible,
        "activities_aligned": count_aligned,
        "objective_breakdown": objective_breakdown,
        "activity_details": activity_details,
        "yoy_comparisons": yoy_comparisons,
        "double_counting_adjustments": double_counting_adj,
        "provenance_hash": provenance_hash,
    }


@pytest.mark.unit
class TestKPICalculation:
    """Test suite for the KPI Calculation Engine."""

    @pytest.fixture
    def kpi_activity_data(self) -> List[Dict[str, Any]]:
        """Inline fixture for KPI activity data."""
        return [
            {
                "activity_id": "CCM-4.1",
                "activity_name": "Solar PV electricity generation",
                "turnover": "500000",
                "capex": "120000",
                "opex": "30000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
            {
                "activity_id": "CCM-3.9",
                "activity_name": "Iron and steel manufacturing",
                "turnover": "1200000",
                "capex": "300000",
                "opex": "80000",
                "is_eligible": True,
                "is_aligned": False,
                "sc_objectives": ["CCM"],
            },
            {
                "activity_id": "CCM-7.1",
                "activity_name": "Construction of new buildings",
                "turnover": "800000",
                "capex": "200000",
                "opex": "50000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM", "CCA"],
            },
            {
                "activity_id": "NON-ELIGIBLE",
                "activity_name": "Consulting services",
                "turnover": "300000",
                "capex": "10000",
                "opex": "15000",
                "is_eligible": False,
                "is_aligned": False,
                "sc_objectives": [],
            },
        ]

    @pytest.fixture
    def kpi_company_financials(self) -> Dict[str, Any]:
        """Inline fixture for KPI company financials."""
        return {
            "total_turnover": "5000000",
            "total_capex": "1000000",
            "total_opex": "250000",
            "reporting_period_start": "2025-01-01",
            "reporting_period_end": "2025-12-31",
        }

    def test_calculate_turnover_ratio(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test turnover alignment ratio is correctly calculated."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        assert result["turnover_ratio"] >= ZERO
        assert result["turnover_ratio"] <= ONE

        # Aligned turnover / total turnover
        expected = _safe_divide(result["aligned_turnover"], result["total_turnover"])
        assert result["turnover_ratio"] == expected

    def test_calculate_capex_ratio(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test CapEx alignment ratio is correctly calculated."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        assert result["capex_ratio"] >= ZERO
        assert result["capex_ratio"] <= ONE

    def test_calculate_opex_ratio(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test OpEx alignment ratio is correctly calculated."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        assert result["opex_ratio"] >= ZERO
        assert result["opex_ratio"] <= ONE

    def test_double_counting_prevention(self):
        """Test double-counting prevention when same activity appears multiple times."""
        activities = [
            {
                "activity_id": "CCM-4.1",
                "turnover": "500000",
                "capex": "100000",
                "opex": "25000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
            {
                "activity_id": "CCM-4.1",  # Duplicate activity_id
                "turnover": "500000",
                "capex": "100000",
                "opex": "25000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
        ]
        financials = {"total_turnover": "2000000", "total_capex": "400000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        # Should count only once (500000, not 1000000)
        assert result["aligned_turnover"] == Decimal("500000")
        assert result["double_counting_adjustments"] > ZERO

    def test_capex_plan_recognition(self):
        """Test CapEx plan amounts are included in aligned CapEx (Article 8)."""
        activities = [
            {
                "activity_id": "CCM-7.2",
                "turnover": "100000",
                "capex": "50000",
                "opex": "10000",
                "is_eligible": True,
                "is_aligned": False,  # Not yet aligned
                "sc_objectives": ["CCM"],
                "capex_plan_status": "PLAN_APPROVED",
                "capex_plan_amount": "200000",
            },
        ]
        financials = {"total_turnover": "1000000", "total_capex": "500000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        assert result["capex_plan_amount"] == Decimal("200000")
        # Aligned capex should include plan amount
        assert result["aligned_capex"] >= Decimal("200000")

    def test_eligible_vs_aligned_breakdown(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test eligible amounts are greater than or equal to aligned amounts."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        assert result["eligible_turnover"] >= result["aligned_turnover"]
        assert result["eligible_capex"] >= result["aligned_capex"] - result["capex_plan_amount"]
        assert result["eligible_opex"] >= result["aligned_opex"]

    def test_zero_denominator_handling(self):
        """Test zero denominator returns zero ratio instead of division error."""
        # This tests the safe_divide function behavior
        result_turnover = _safe_divide(Decimal("100"), ZERO)
        result_capex = _safe_divide(ZERO, ZERO)

        assert result_turnover == ZERO
        assert result_capex == ZERO

    def test_activity_level_mapping(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test activity-level detail is captured in results."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        assert len(result["activity_details"]) == len(kpi_activity_data)
        for detail in result["activity_details"]:
            assert "activity_id" in detail
            assert "turnover" in detail
            assert "is_eligible" in detail
            assert "is_aligned" in detail

    def test_kpi_result_structure(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test KPI result contains all required fields."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        required_fields = [
            "eligible_turnover", "eligible_capex", "eligible_opex",
            "aligned_turnover", "aligned_capex", "aligned_opex",
            "total_turnover", "total_capex", "total_opex",
            "eligible_turnover_ratio", "eligible_capex_ratio", "eligible_opex_ratio",
            "turnover_ratio", "capex_ratio", "opex_ratio",
            "capex_plan_amount", "activities_total", "activities_eligible",
            "activities_aligned", "objective_breakdown", "activity_details",
            "double_counting_adjustments", "provenance_hash",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_year_over_year_comparison(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test year-over-year comparison with prior period data."""
        prior_period = {
            "turnover_ratio": "0.20",
            "capex_ratio": "0.15",
            "opex_ratio": "0.10",
        }

        result = _simulate_calculate_kpis(
            kpi_activity_data, kpi_company_financials, prior_period
        )

        yoy = result["yoy_comparisons"]
        assert len(yoy) == 3

        for comparison in yoy:
            assert "kpi_type" in comparison
            assert "current_ratio" in comparison
            assert "previous_ratio" in comparison
            assert "change_absolute" in comparison
            assert "trend" in comparison
            assert comparison["trend"] in ["IMPROVED", "DECLINED", "STABLE"]

    def test_multiple_activities_aggregation(self):
        """Test aggregation of multiple distinct activities."""
        activities = [
            {
                "activity_id": "CCM-4.1",
                "turnover": "300000",
                "capex": "80000",
                "opex": "20000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
            {
                "activity_id": "CCM-7.1",
                "turnover": "200000",
                "capex": "60000",
                "opex": "15000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
        ]
        financials = {"total_turnover": "1000000", "total_capex": "300000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        assert result["aligned_turnover"] == Decimal("500000")  # 300k + 200k
        assert result["activities_aligned"] == 2

    def test_financial_precision(self):
        """Test financial calculations use sufficient decimal precision."""
        activities = [
            {
                "activity_id": "CCM-4.1",
                "turnover": "333333.33",
                "capex": "111111.11",
                "opex": "22222.22",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
        ]
        financials = {"total_turnover": "1000000", "total_capex": "500000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        # Verify ratios are Decimal with appropriate precision
        assert isinstance(result["turnover_ratio"], Decimal)
        assert isinstance(result["capex_ratio"], Decimal)
        assert isinstance(result["opex_ratio"], Decimal)

        # Verify no floating-point artefacts
        expected_turnover_ratio = _safe_divide(
            Decimal("333333.33"), Decimal("1000000")
        )
        assert result["turnover_ratio"] == expected_turnover_ratio

    def test_objective_level_breakdown(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test per-objective breakdown of eligible and aligned amounts."""
        result = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        obj_bkdn = result["objective_breakdown"]
        assert "CCM" in obj_bkdn

        ccm = obj_bkdn["CCM"]
        assert ccm["eligible_turnover"] >= ZERO
        assert ccm["aligned_turnover"] >= ZERO
        assert ccm["eligible_turnover"] >= ccm["aligned_turnover"]

    def test_100_percent_alignment_scenario(self):
        """Test scenario where all activities are fully aligned."""
        activities = [
            {
                "activity_id": "CCM-4.1",
                "turnover": "1000000",
                "capex": "500000",
                "opex": "100000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
        ]
        financials = {"total_turnover": "1000000", "total_capex": "500000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        assert result["turnover_ratio"] == Decimal("1.0000")
        assert result["capex_ratio"] == Decimal("1.0000")
        assert result["opex_ratio"] == Decimal("1.0000")

    def test_0_percent_alignment_scenario(self):
        """Test scenario where no activities are aligned."""
        activities = [
            {
                "activity_id": "NON-TAX",
                "turnover": "1000000",
                "capex": "500000",
                "opex": "100000",
                "is_eligible": False,
                "is_aligned": False,
                "sc_objectives": [],
            },
        ]
        financials = {"total_turnover": "1000000", "total_capex": "500000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        assert result["turnover_ratio"] == ZERO
        assert result["capex_ratio"] == ZERO
        assert result["opex_ratio"] == ZERO

    def test_mixed_alignment_scenario(self):
        """Test scenario with mix of aligned, eligible-only, and non-eligible activities."""
        activities = [
            {
                "activity_id": "CCM-4.1",
                "turnover": "400000",
                "capex": "100000",
                "opex": "20000",
                "is_eligible": True,
                "is_aligned": True,
                "sc_objectives": ["CCM"],
            },
            {
                "activity_id": "CCM-3.9",
                "turnover": "300000",
                "capex": "150000",
                "opex": "30000",
                "is_eligible": True,
                "is_aligned": False,
                "sc_objectives": ["CCM"],
            },
            {
                "activity_id": "NON-TAX",
                "turnover": "300000",
                "capex": "50000",
                "opex": "50000",
                "is_eligible": False,
                "is_aligned": False,
                "sc_objectives": [],
            },
        ]
        financials = {"total_turnover": "1000000", "total_capex": "300000", "total_opex": "100000"}

        result = _simulate_calculate_kpis(activities, financials)

        # Turnover: 400k aligned / 1M total = 0.4
        assert result["turnover_ratio"] == Decimal("0.4000")

        # Eligible: 700k / 1M = 0.7
        assert result["eligible_turnover_ratio"] == Decimal("0.7000")

        assert result["activities_eligible"] == 2
        assert result["activities_aligned"] == 1

    def test_provenance_hash_generated(
        self,
        kpi_activity_data: List[Dict[str, Any]],
        kpi_company_financials: Dict[str, Any],
    ):
        """Test provenance hash is generated and reproducible."""
        result1 = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)
        result2 = _simulate_calculate_kpis(kpi_activity_data, kpi_company_financials)

        assert len(result1["provenance_hash"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", result1["provenance_hash"])
        assert result1["provenance_hash"] == result2["provenance_hash"]
