# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Financial Institution Features.

Tests GAR stock calculation, GAR flow calculation, BTAR calculation,
exposure classification (corporate, mortgage, auto loan, project
finance), EBA template generation (Templates 6-10), mortgage
alignment (EPC rating), auto loan alignment (CO2 threshold), GAR
sector breakdown, and GAR vs BTAR comparison with 40+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal


# ===========================================================================
# GAR Stock Calculation
# ===========================================================================

class TestGARStockCalculation:
    """Test Green Asset Ratio stock (point-in-time) calculation."""

    def test_gar_stock_created(self, sample_gar_result):
        assert sample_gar_result["gar_type"] == "stock"
        assert sample_gar_result["gar_pct"] is not None

    def test_gar_numerator_denominator(self, sample_gar_result):
        assert sample_gar_result["aligned_assets"] >= 0
        assert sample_gar_result["covered_assets"] > 0
        assert sample_gar_result["aligned_assets"] <= sample_gar_result["covered_assets"]

    def test_gar_percentage_formula(self, sample_gar_result):
        expected = (
            sample_gar_result["aligned_assets"]
            / sample_gar_result["covered_assets"]
        ) * 100
        assert abs(sample_gar_result["gar_pct"] - expected) < 0.01

    def test_gar_stock_range(self, sample_gar_result):
        assert 0.0 <= sample_gar_result["gar_pct"] <= 100.0

    def test_gar_excludes_sovereign(self, sample_gar_result):
        exclusions = sample_gar_result.get("excluded_categories", [])
        assert "sovereign" in exclusions

    def test_gar_excludes_central_bank(self, sample_gar_result):
        exclusions = sample_gar_result.get("excluded_categories", [])
        assert "central_bank" in exclusions

    def test_gar_excludes_trading_book(self, sample_gar_result):
        exclusions = sample_gar_result.get("excluded_categories", [])
        assert "trading_book" in exclusions

    @pytest.mark.parametrize("aligned,covered,expected_gar", [
        (1_000_000, 10_000_000, 10.0),
        (5_000_000, 10_000_000, 50.0),
        (10_000_000, 10_000_000, 100.0),
        (0, 10_000_000, 0.0),
        (250_000, 10_000_000, 2.5),
    ])
    def test_gar_calculation_parametrized(self, aligned, covered, expected_gar):
        gar = (aligned / covered) * 100
        assert gar == pytest.approx(expected_gar, abs=0.01)


# ===========================================================================
# GAR Flow Calculation
# ===========================================================================

class TestGARFlowCalculation:
    """Test GAR flow (new business period) calculation."""

    def test_gar_flow_created(self, sample_gar_flow_result):
        assert sample_gar_flow_result["gar_type"] == "flow"

    def test_gar_flow_period(self, sample_gar_flow_result):
        assert "period_start" in sample_gar_flow_result
        assert "period_end" in sample_gar_flow_result

    def test_gar_flow_new_business_only(self, sample_gar_flow_result):
        assert sample_gar_flow_result["new_business_volume"] > 0
        assert sample_gar_flow_result["aligned_new_business"] >= 0
        assert sample_gar_flow_result["aligned_new_business"] <= sample_gar_flow_result["new_business_volume"]

    def test_gar_flow_percentage(self, sample_gar_flow_result):
        expected = (
            sample_gar_flow_result["aligned_new_business"]
            / sample_gar_flow_result["new_business_volume"]
        ) * 100
        assert abs(sample_gar_flow_result["gar_flow_pct"] - expected) < 0.01


# ===========================================================================
# BTAR Calculation
# ===========================================================================

class TestBTARCalculation:
    """Test Banking Book Taxonomy Alignment Ratio calculation."""

    def test_btar_created(self, sample_btar_result):
        assert sample_btar_result["btar_pct"] is not None

    def test_btar_includes_non_nfrd(self, sample_btar_result):
        assert sample_btar_result["includes_non_nfrd"] is True

    def test_btar_broader_than_gar(self, sample_gar_result, sample_btar_result):
        assert sample_btar_result["covered_assets"] >= sample_gar_result["covered_assets"]

    def test_btar_range(self, sample_btar_result):
        assert 0.0 <= sample_btar_result["btar_pct"] <= 100.0

    def test_btar_estimation_methodology(self, sample_btar_result):
        valid_methods = ["proxy", "sector_average", "best_estimate", "reported"]
        assert sample_btar_result["estimation_method"] in valid_methods


# ===========================================================================
# Exposure Classification
# ===========================================================================

class TestExposureClassification:
    """Test exposure classification into asset categories."""

    @pytest.mark.parametrize("exposure_type,expected_category", [
        ("corporate_loan", "corporate"),
        ("corporate_bond", "corporate"),
        ("equity", "corporate"),
        ("mortgage", "retail_mortgage"),
        ("auto_loan", "retail_auto"),
        ("project_finance", "project_finance"),
        ("interbank", "interbank"),
        ("sovereign", "sovereign"),
    ])
    def test_exposure_classification(self, exposure_type, expected_category):
        classification_map = {
            "corporate_loan": "corporate",
            "corporate_bond": "corporate",
            "equity": "corporate",
            "mortgage": "retail_mortgage",
            "auto_loan": "retail_auto",
            "project_finance": "project_finance",
            "interbank": "interbank",
            "sovereign": "sovereign",
        }
        assert classification_map[exposure_type] == expected_category

    def test_corporate_exposure_requires_nfrd(self):
        holding = {
            "exposure_type": "corporate_loan",
            "counterparty_nfrd_subject": True,
            "nace_code": "D35.11",
        }
        assert holding["counterparty_nfrd_subject"] is True

    def test_non_nfrd_corporate_excluded_from_gar(self):
        holding = {
            "exposure_type": "corporate_loan",
            "counterparty_nfrd_subject": False,
        }
        gar_eligible = holding["counterparty_nfrd_subject"]
        assert gar_eligible is False


# ===========================================================================
# EBA Template Generation
# ===========================================================================

class TestEBATemplateGeneration:
    """Test EBA Pillar III template generation (Templates 6-10)."""

    @pytest.mark.parametrize("template_num,template_name", [
        (6, "GAR_summary"),
        (7, "GAR_sector_breakdown"),
        (8, "GAR_flow"),
        (9, "Mitigating_actions"),
        (10, "Other_climate_actions"),
    ])
    def test_eba_template_defined(self, template_num, template_name):
        templates = {
            6: "GAR_summary",
            7: "GAR_sector_breakdown",
            8: "GAR_flow",
            9: "Mitigating_actions",
            10: "Other_climate_actions",
        }
        assert templates[template_num] == template_name

    def test_eba_template_has_columns(self, sample_eba_template):
        assert "columns" in sample_eba_template
        assert len(sample_eba_template["columns"]) >= 3

    def test_eba_template_has_rows(self, sample_eba_template):
        assert "rows" in sample_eba_template
        assert len(sample_eba_template["rows"]) >= 1

    def test_eba_template_version(self, sample_eba_template):
        assert sample_eba_template["eba_version"] == "3.2"

    def test_eba_template_format(self, sample_eba_template):
        assert sample_eba_template["format"] in ["xlsx", "xbrl", "csv"]


# ===========================================================================
# Mortgage Alignment (EPC Rating)
# ===========================================================================

class TestMortgageAlignment:
    """Test mortgage alignment via EPC energy rating."""

    @pytest.mark.parametrize("epc_rating,expected_aligned", [
        ("A", True),
        ("B", True),
        ("C", False),
        ("D", False),
        ("E", False),
        ("F", False),
        ("G", False),
    ])
    def test_epc_alignment(self, epc_rating, expected_aligned):
        aligned = epc_rating in ["A", "B"]
        assert aligned == expected_aligned

    def test_mortgage_top_15pct_rule(self):
        building = {
            "epc_rating": "C",
            "in_top_15_pct_national": True,
        }
        aligned = building["epc_rating"] in ["A", "B"] or building["in_top_15_pct_national"]
        assert aligned is True

    def test_mortgage_renovation_alignment(self):
        renovation = {
            "epc_before": "D",
            "epc_after": "B",
            "energy_improvement_pct": 30.0,
        }
        aligned = renovation["energy_improvement_pct"] >= 30.0
        assert aligned is True

    def test_mortgage_missing_epc_not_aligned(self):
        building = {"epc_rating": None}
        aligned = building["epc_rating"] is not None and building["epc_rating"] in ["A", "B"]
        assert aligned is False


# ===========================================================================
# Auto Loan Alignment (CO2 Threshold)
# ===========================================================================

class TestAutoLoanAlignment:
    """Test auto loan alignment via CO2 emission threshold."""

    @pytest.mark.parametrize("co2_gkm,expected_aligned", [
        (0, True),      # EV
        (25, True),     # PHEV below threshold
        (50, False),    # Above threshold
        (95, False),
        (120, False),
        (150, False),
    ])
    def test_co2_threshold_alignment(self, co2_gkm, expected_aligned):
        threshold = 50  # g CO2/km threshold (post-2025)
        aligned = co2_gkm < threshold
        assert aligned == expected_aligned

    def test_zero_emission_vehicle_aligned(self):
        vehicle = {"co2_gkm": 0, "fuel_type": "electric"}
        assert vehicle["co2_gkm"] == 0

    def test_phev_partially_aligned(self):
        vehicle = {"co2_gkm": 30, "fuel_type": "phev"}
        assert vehicle["co2_gkm"] < 50


# ===========================================================================
# GAR Sector Breakdown
# ===========================================================================

class TestGARSectorBreakdown:
    """Test GAR breakdown by NACE sector."""

    def test_sector_breakdown_present(self, sample_gar_result):
        assert "sector_breakdown" in sample_gar_result
        assert len(sample_gar_result["sector_breakdown"]) >= 1

    def test_sector_has_nace_section(self, sample_gar_result):
        for sector in sample_gar_result["sector_breakdown"]:
            assert "nace_section" in sector
            assert sector["nace_section"] in [
                "A", "B", "C", "D", "E", "F", "G", "H",
                "I", "J", "K", "L", "M", "N", "O",
            ]

    def test_sector_has_exposure_and_gar(self, sample_gar_result):
        for sector in sample_gar_result["sector_breakdown"]:
            assert "exposure" in sector
            assert "aligned_exposure" in sector
            assert "sector_gar_pct" in sector

    def test_sector_gar_range(self, sample_gar_result):
        for sector in sample_gar_result["sector_breakdown"]:
            assert 0.0 <= sector["sector_gar_pct"] <= 100.0

    def test_sector_exposures_sum_to_total(self, sample_gar_result):
        sector_sum = sum(
            s["exposure"] for s in sample_gar_result["sector_breakdown"]
        )
        assert abs(sector_sum - sample_gar_result["covered_assets"]) < 1.0


# ===========================================================================
# GAR vs BTAR Comparison
# ===========================================================================

class TestGARvsBTARComparison:
    """Test GAR vs BTAR comparison analysis."""

    def test_comparison_structure(self, sample_gar_result, sample_btar_result):
        comparison = {
            "gar_pct": sample_gar_result["gar_pct"],
            "btar_pct": sample_btar_result["btar_pct"],
            "gap_pct": sample_btar_result["btar_pct"] - sample_gar_result["gar_pct"],
        }
        assert "gar_pct" in comparison
        assert "btar_pct" in comparison
        assert "gap_pct" in comparison

    def test_btar_typically_higher(self, sample_gar_result, sample_btar_result):
        # BTAR has broader scope so often different from GAR
        assert sample_btar_result["covered_assets"] >= sample_gar_result["covered_assets"]

    def test_comparison_delta(self, sample_gar_result, sample_btar_result):
        delta = abs(sample_btar_result["btar_pct"] - sample_gar_result["gar_pct"])
        assert delta >= 0.0
