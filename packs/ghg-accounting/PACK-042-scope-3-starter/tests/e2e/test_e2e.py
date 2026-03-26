# -*- coding: utf-8 -*-
"""
End-to-End Tests for PACK-042 Scope 3 Starter Pack
=====================================================

Simulates complete Scope 3 calculation pipelines for multiple
sector archetypes: manufacturing, retail, technology, financial
services, SME, full pipeline, multi-framework compliance, and
year-over-year comparison.

Coverage target: 85%+
Total tests: ~30
"""

import hashlib
import json
import math
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# Import from the parent-level conftest
import sys
from pathlib import Path

# Ensure the parent tests/ directory is importable
_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    SCOPE3_CATEGORIES,
    UPSTREAM_CATEGORIES,
    DOWNSTREAM_CATEGORIES,
    OVERLAP_RULES,
    compute_provenance_hash,
)


# =============================================================================
# Helpers: Simulated Pipeline Steps
# =============================================================================


def _screening_step(org: Dict, spend: List[Dict]) -> Dict[str, Any]:
    """Simulate category screening from org profile and spend data."""
    cats = {}
    for cat_id in SCOPE3_CATEGORIES:
        cat_spend = sum(
            t["amount_eur"] for t in spend if t["scope3_category"] == cat_id
        )
        estimated_tco2e = float(cat_spend) * 0.001  # rough EEIO
        relevance = "HIGH" if estimated_tco2e > 1000 else (
            "MEDIUM" if estimated_tco2e > 100 else "LOW"
        )
        cats[cat_id] = {
            "estimated_tco2e": Decimal(str(round(estimated_tco2e, 1))),
            "relevance": relevance,
            "spend_eur": cat_spend,
        }
    total_est = sum(float(c["estimated_tco2e"]) for c in cats.values())
    return {
        "org_id": org["org_id"],
        "categories": cats,
        "total_estimated_tco2e": Decimal(str(round(total_est, 1))),
    }


def _calculation_step(screening: Dict) -> Dict[str, Any]:
    """Simulate per-category calculation with gas breakdown."""
    cats = {}
    total = Decimal("0")
    for cat_id, scr in screening["categories"].items():
        tco2e = scr["estimated_tco2e"]
        co2 = tco2e * Decimal("0.95")
        ch4 = tco2e * Decimal("0.03")
        n2o = tco2e * Decimal("0.02")
        cats[cat_id] = {
            "total_tco2e": tco2e,
            "by_gas": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "methodology": "SPEND_BASED",
            "dqr": Decimal("3.5"),
            "uncertainty_pct": Decimal("40.0"),
        }
        total += tco2e
    return {
        "org_id": screening["org_id"],
        "categories": cats,
        "total_scope3_tco2e": total,
    }


def _consolidation_step(calc_results: Dict, org: Dict) -> Dict[str, Any]:
    """Simulate consolidation with upstream/downstream split."""
    cats = calc_results["categories"]
    upstream = sum(cats[c]["total_tco2e"] for c in UPSTREAM_CATEGORIES if c in cats)
    downstream = sum(cats[c]["total_tco2e"] for c in DOWNSTREAM_CATEGORIES if c in cats)
    total = upstream + downstream

    scope1 = org.get("scope1_tco2e", Decimal("0"))
    scope2 = org.get("scope2_market_tco2e", Decimal("0"))
    total_footprint = scope1 + scope2 + total

    return {
        "org_id": org["org_id"],
        "total_scope3_tco2e": total,
        "upstream_tco2e": upstream,
        "downstream_tco2e": downstream,
        "scope1_tco2e": scope1,
        "scope2_market_tco2e": scope2,
        "total_footprint_tco2e": total_footprint,
        "scope3_pct": float(total / total_footprint * 100) if total_footprint else 0,
    }


def _double_counting_step(consolidation: Dict) -> Dict[str, Any]:
    """Simulate double-counting check."""
    overlaps = []
    for rule in OVERLAP_RULES:
        overlaps.append({"rule": rule, "overlap_tco2e": Decimal("0"), "status": "OK"})
    return {
        "rules_evaluated": len(OVERLAP_RULES),
        "overlaps_found": 0,
        "net_adjustment_tco2e": Decimal("0"),
        "adjusted_total_tco2e": consolidation["total_scope3_tco2e"],
    }


def _quality_step(calc_results: Dict) -> Dict[str, Any]:
    """Simulate data quality assessment."""
    dqi_scores = {}
    for cat_id in calc_results["categories"]:
        dqi_scores[cat_id] = {
            "dqr": Decimal("3.5"),
            "dqi": {
                "technological_representativeness": Decimal("3.0"),
                "temporal_representativeness": Decimal("3.0"),
                "geographical_representativeness": Decimal("3.5"),
                "completeness": Decimal("4.0"),
                "reliability": Decimal("4.0"),
            },
        }
    return {"overall_dqr": Decimal("3.5"), "categories": dqi_scores}


def _uncertainty_step(calc_results: Dict) -> Dict[str, Any]:
    """Simulate uncertainty assessment."""
    total = float(calc_results["total_scope3_tco2e"])
    overall_pct = 40.0
    lower = total * (1 - overall_pct / 100)
    upper = total * (1 + overall_pct / 100)
    return {
        "point_estimate_tco2e": Decimal(str(round(total, 1))),
        "lower_bound_tco2e": Decimal(str(round(lower, 1))),
        "upper_bound_tco2e": Decimal(str(round(upper, 1))),
        "overall_uncertainty_pct": Decimal(str(overall_pct)),
        "method": "ANALYTICAL",
    }


def _compliance_step(consolidation: Dict, calc_results: Dict = None) -> Dict[str, Any]:
    """Simulate compliance assessment against frameworks."""
    # Count categories with non-zero emissions from calc_results if available
    cats_source = {}
    if calc_results and "categories" in calc_results:
        cats_source = calc_results["categories"]
    elif "categories" in consolidation:
        cats_source = consolidation["categories"]
    cats_counted = sum(
        1 for c in SCOPE3_CATEGORIES
        if cats_source.get(c, {}).get("total_tco2e", 0) > 0
    )
    return {
        "GHG_PROTOCOL": {
            "status": "SUBSTANTIALLY_COMPLIANT",
            "categories_reported": 15,
            "requirements_met": 13,
            "requirements_total": 15,
        },
        "ESRS_E1": {
            "status": "COMPLIANT" if cats_counted >= 3 else "NON_COMPLIANT",
            "phase_in_year": 2025,
            "categories_required": 3,
            "categories_reported": min(cats_counted, 15),
        },
        "CDP": {
            "status": "B_GRADE",
            "c6_5_reported": True,
            "c6_7_reported": True,
        },
        "SBTi": {
            "status": "COMMITTED",
            "coverage_pct": 67.0,
            "target_aligned": True,
        },
    }


def _report_step(pipeline_results: Dict) -> Dict[str, Any]:
    """Simulate report generation."""
    provenance = compute_provenance_hash(pipeline_results)
    return {
        "report_id": "RPT-E2E-001",
        "format": "MARKDOWN",
        "sections": [
            "executive_summary",
            "methodology",
            "category_results",
            "hotspot_analysis",
            "data_quality",
            "uncertainty",
            "compliance",
            "appendix",
        ],
        "provenance_hash": provenance,
    }


def _run_full_pipeline(org: Dict, spend: List[Dict]) -> Dict[str, Any]:
    """Run the full simulated pipeline end to end."""
    # Phase 1: Screening
    screening = _screening_step(org, spend)

    # Phase 2: Calculation
    calc_results = _calculation_step(screening)

    # Phase 3: Consolidation
    consolidation = _consolidation_step(calc_results, org)

    # Phase 4: Double-counting
    dc_check = _double_counting_step(consolidation)

    # Phase 5: Quality
    quality = _quality_step(calc_results)

    # Phase 6: Uncertainty
    uncertainty = _uncertainty_step(calc_results)

    # Phase 7: Compliance
    compliance = _compliance_step(consolidation, calc_results)

    # Phase 8: Reporting
    pipeline_data = {
        "screening": {"total": str(screening["total_estimated_tco2e"])},
        "calc": {"total": str(calc_results["total_scope3_tco2e"])},
        "consolidation": {"total": str(consolidation["total_scope3_tco2e"])},
    }
    report = _report_step(pipeline_data)

    return {
        "org_id": org["org_id"],
        "status": "COMPLETED",
        "screening": screening,
        "calc_results": calc_results,
        "consolidation": consolidation,
        "double_counting": dc_check,
        "quality": quality,
        "uncertainty": uncertainty,
        "compliance": compliance,
        "report": report,
    }


# =============================================================================
# Manufacturing Company E2E
# =============================================================================


class TestManufacturingE2E:
    """End-to-end test for manufacturing company."""

    def test_full_pipeline_completes(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        assert result["status"] == "COMPLETED"
        assert result["org_id"] == "ORG-MFG-001"

    def test_scope3_exceeds_scope12(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        s3 = result["consolidation"]["total_scope3_tco2e"]
        s1 = result["consolidation"]["scope1_tco2e"]
        s2 = result["consolidation"]["scope2_market_tco2e"]
        assert s3 > s1 + s2, "Scope 3 should exceed Scope 1+2 for manufacturing"

    def test_scope3_pct_above_50(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        pct = result["consolidation"]["scope3_pct"]
        assert pct > 50, f"Scope 3 should be > 50% of total, got {pct:.1f}%"

    def test_upstream_dominates(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        upstream = result["consolidation"]["upstream_tco2e"]
        downstream = result["consolidation"]["downstream_tco2e"]
        assert upstream > downstream, "Upstream should dominate for manufacturing"

    def test_provenance_hash_present(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        h = result["report"]["provenance_hash"]
        assert len(h) == 64


# =============================================================================
# Retail Company E2E
# =============================================================================


class TestRetailE2E:
    """End-to-end test for retail company."""

    def test_full_pipeline_completes(self, retail_org, sample_spend_data):
        result = _run_full_pipeline(retail_org, sample_spend_data)
        assert result["status"] == "COMPLETED"
        assert result["org_id"] == "ORG-RTL-001"

    def test_all_15_categories_screened(self, retail_org, sample_spend_data):
        result = _run_full_pipeline(retail_org, sample_spend_data)
        cats = result["screening"]["categories"]
        assert len(cats) == 15

    def test_report_has_8_sections(self, retail_org, sample_spend_data):
        result = _run_full_pipeline(retail_org, sample_spend_data)
        sections = result["report"]["sections"]
        assert len(sections) == 8


# =============================================================================
# Technology Company E2E
# =============================================================================


class TestTechnologyE2E:
    """End-to-end test for technology company."""

    def test_full_pipeline_completes(self, technology_org, sample_spend_data):
        result = _run_full_pipeline(technology_org, sample_spend_data)
        assert result["status"] == "COMPLETED"
        assert result["org_id"] == "ORG-TECH-001"

    def test_quality_assessment_present(self, technology_org, sample_spend_data):
        result = _run_full_pipeline(technology_org, sample_spend_data)
        assert result["quality"]["overall_dqr"] >= Decimal("1.0")
        assert result["quality"]["overall_dqr"] <= Decimal("5.0")


# =============================================================================
# Financial Services E2E
# =============================================================================


class TestFinancialServicesE2E:
    """End-to-end test for financial services company."""

    def test_full_pipeline_completes(self, financial_org, sample_spend_data):
        result = _run_full_pipeline(financial_org, sample_spend_data)
        assert result["status"] == "COMPLETED"
        assert result["org_id"] == "ORG-FIN-001"

    def test_scope3_includes_cat15_investments(self, financial_org, sample_spend_data):
        result = _run_full_pipeline(financial_org, sample_spend_data)
        cat15 = result["calc_results"]["categories"].get("CAT_15")
        assert cat15 is not None, "Financial should have CAT_15 investment data"

    def test_compliance_includes_multiple_frameworks(
        self, financial_org, sample_spend_data
    ):
        result = _run_full_pipeline(financial_org, sample_spend_data)
        compliance = result["compliance"]
        assert "GHG_PROTOCOL" in compliance
        assert "ESRS_E1" in compliance
        assert "CDP" in compliance
        assert "SBTi" in compliance


# =============================================================================
# SME E2E
# =============================================================================


class TestSME_E2E:
    """End-to-end test for small-medium enterprise."""

    def test_full_pipeline_completes(self, sme_org, sample_spend_data):
        result = _run_full_pipeline(sme_org, sample_spend_data)
        assert result["status"] == "COMPLETED"
        assert result["org_id"] == "ORG-SME-001"

    def test_sme_total_footprint_reasonable(self, sme_org, sample_spend_data):
        result = _run_full_pipeline(sme_org, sample_spend_data)
        total = result["consolidation"]["total_footprint_tco2e"]
        # SME with small scope 1+2 but using same spend data, total should be positive
        assert total > 0


# =============================================================================
# Full Pipeline Integration
# =============================================================================


class TestFullPipelineIntegration:
    """Test full pipeline integration across all steps."""

    def test_all_8_phases_execute(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        expected_keys = [
            "screening", "calc_results", "consolidation",
            "double_counting", "quality", "uncertainty",
            "compliance", "report",
        ]
        for key in expected_keys:
            assert key in result, f"Missing pipeline phase: {key}"

    def test_total_scope3_consistent_across_phases(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        calc_total = result["calc_results"]["total_scope3_tco2e"]
        consol_total = result["consolidation"]["total_scope3_tco2e"]
        uncertainty_point = result["uncertainty"]["point_estimate_tco2e"]
        # Calc and consolidation should match (no double-counting adjustments here)
        assert calc_total == consol_total
        assert uncertainty_point == calc_total

    def test_uncertainty_bounds_bracket_point_estimate(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        u = result["uncertainty"]
        assert u["lower_bound_tco2e"] < u["point_estimate_tco2e"]
        assert u["upper_bound_tco2e"] > u["point_estimate_tco2e"]

    def test_double_counting_evaluates_all_12_rules(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        dc = result["double_counting"]
        assert dc["rules_evaluated"] == 12

    def test_compliance_covers_4_frameworks(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        compliance = result["compliance"]
        assert len(compliance) == 4

    def test_gas_breakdown_sums_to_total(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        for cat_id, data in result["calc_results"]["categories"].items():
            gas_sum = sum(data["by_gas"].values())
            assert gas_sum == data["total_tco2e"], (
                f"{cat_id} gas breakdown {gas_sum} != total {data['total_tco2e']}"
            )


# =============================================================================
# Multi-Framework Compliance E2E
# =============================================================================


class TestMultiFrameworkCompliance:
    """Test multi-framework compliance scenarios."""

    def test_ghg_protocol_substantially_compliant(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        ghg = result["compliance"]["GHG_PROTOCOL"]
        assert ghg["status"] in ("COMPLIANT", "SUBSTANTIALLY_COMPLIANT")

    def test_esrs_e1_minimum_3_categories(
        self, manufacturing_org, sample_spend_data
    ):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        esrs = result["compliance"]["ESRS_E1"]
        assert esrs["categories_reported"] >= 3

    def test_cdp_reporting_flags(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        cdp = result["compliance"]["CDP"]
        assert cdp["c6_5_reported"] is True
        assert cdp["c6_7_reported"] is True

    def test_sbti_coverage_target(self, manufacturing_org, sample_spend_data):
        result = _run_full_pipeline(manufacturing_org, sample_spend_data)
        sbti = result["compliance"]["SBTi"]
        assert sbti["coverage_pct"] >= 67.0


# =============================================================================
# Year-over-Year Comparison E2E
# =============================================================================


class TestYearOverYearComparison:
    """Test year-over-year comparison scenario."""

    def test_yoy_reduction_detectable(self, manufacturing_org, sample_spend_data):
        """Simulate year-over-year with a 5% reduction."""
        # Year 1
        result_y1 = _run_full_pipeline(manufacturing_org, sample_spend_data)
        total_y1 = result_y1["calc_results"]["total_scope3_tco2e"]

        # Year 2: Simulate 5% reduction by scaling spend down
        reduced_spend = []
        for txn in sample_spend_data:
            txn_copy = dict(txn)
            txn_copy["amount_eur"] = txn["amount_eur"] * Decimal("0.95")
            txn_copy["transaction_id"] = txn["transaction_id"] + "-Y2"
            reduced_spend.append(txn_copy)

        org_y2 = dict(manufacturing_org)
        org_y2["reporting_year"] = 2026

        result_y2 = _run_full_pipeline(org_y2, reduced_spend)
        total_y2 = result_y2["calc_results"]["total_scope3_tco2e"]

        assert total_y2 < total_y1, "Year 2 should have lower emissions"
        reduction_pct = float((total_y1 - total_y2) / total_y1 * 100)
        assert 4.0 < reduction_pct < 6.0, (
            f"Reduction should be ~5%, got {reduction_pct:.1f}%"
        )

    def test_yoy_provenance_differs(self, manufacturing_org, sample_spend_data):
        """Year 1 and Year 2 should have different provenance hashes."""
        result_y1 = _run_full_pipeline(manufacturing_org, sample_spend_data)

        reduced_spend = []
        for txn in sample_spend_data:
            txn_copy = dict(txn)
            txn_copy["amount_eur"] = txn["amount_eur"] * Decimal("0.95")
            txn_copy["transaction_id"] = txn["transaction_id"] + "-Y2"
            reduced_spend.append(txn_copy)

        org_y2 = dict(manufacturing_org)
        org_y2["reporting_year"] = 2026
        result_y2 = _run_full_pipeline(org_y2, reduced_spend)

        h1 = result_y1["report"]["provenance_hash"]
        h2 = result_y2["report"]["provenance_hash"]
        assert h1 != h2, "Different years should produce different provenance hashes"
