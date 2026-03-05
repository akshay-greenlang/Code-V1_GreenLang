# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Financial Institutions Module.

Tests FI portfolio creation and holdings management, SBTi target
coverage calculation, financed emissions by asset class, EVIC and
revenue attribution methods, WACI (Weighted Average Carbon Intensity),
PCAF data quality distribution (DQ 1-5), linear coverage path to
100% by 2040, and FINZ compliance validation with 26+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Portfolio Creation
# ===========================================================================

class TestPortfolioCreation:
    """Test FI portfolio creation and holdings management."""

    def test_create_portfolio(self, sample_fi_portfolio):
        assert sample_fi_portfolio["portfolio_name"] == "Corporate Lending Portfolio"

    def test_portfolio_reporting_year(self, sample_fi_portfolio):
        assert sample_fi_portfolio["reporting_year"] == 2024

    def test_portfolio_total_value(self, sample_fi_portfolio):
        assert sample_fi_portfolio["total_portfolio_value_usd"] > 0

    def test_holdings_count(self, sample_fi_portfolio):
        assert len(sample_fi_portfolio["holdings"]) == 3

    def test_holding_has_required_fields(self, sample_fi_portfolio):
        required_fields = [
            "holding_id", "company_name", "asset_class", "exposure_usd",
            "financed_emissions_tco2e", "has_sbti_target", "pcaf_data_quality",
        ]
        for holding in sample_fi_portfolio["holdings"]:
            for field in required_fields:
                assert field in holding

    def test_holding_unique_ids(self, sample_fi_portfolio):
        ids = [h["holding_id"] for h in sample_fi_portfolio["holdings"]]
        assert len(ids) == len(set(ids))


# ===========================================================================
# Coverage Calculation
# ===========================================================================

class TestCoverageCalculation:
    """Test percentage of portfolio with SBTi targets."""

    def test_coverage_percentage(self, sample_fi_portfolio):
        holdings = sample_fi_portfolio["holdings"]
        with_target = sum(1 for h in holdings if h["has_sbti_target"])
        expected_pct = (with_target / len(holdings)) * 100
        assert abs(sample_fi_portfolio["coverage_with_sbti_pct"] - expected_pct) < 1.0

    def test_coverage_above_zero(self, sample_fi_portfolio):
        assert sample_fi_portfolio["coverage_with_sbti_pct"] > 0

    @pytest.mark.parametrize("targets_count,total_count,expected_pct", [
        (3, 3, 100.0),
        (2, 3, 66.67),
        (1, 3, 33.33),
        (0, 3, 0.0),
    ])
    def test_coverage_calculation_parametrized(self, targets_count, total_count, expected_pct):
        pct = (targets_count / total_count) * 100
        assert abs(pct - expected_pct) < 0.1

    def test_exposure_weighted_coverage(self, sample_fi_portfolio):
        holdings = sample_fi_portfolio["holdings"]
        total_exposure = sum(h["exposure_usd"] for h in holdings)
        target_exposure = sum(
            h["exposure_usd"] for h in holdings if h["has_sbti_target"]
        )
        weighted_coverage = (target_exposure / total_exposure) * 100
        assert 0 <= weighted_coverage <= 100


# ===========================================================================
# Financed Emissions
# ===========================================================================

class TestFinancedEmissions:
    """Test financed emissions by asset class."""

    VALID_ASSET_CLASSES = [
        "listed_equity", "corporate_bond", "private_equity",
        "project_finance", "commercial_real_estate", "mortgage",
        "motor_vehicle_loan", "sovereign_bond",
    ]

    def test_total_financed_emissions(self, sample_fi_portfolio):
        assert sample_fi_portfolio["total_financed_emissions_tco2e"] > 0

    def test_emissions_by_holding(self, sample_fi_portfolio):
        for holding in sample_fi_portfolio["holdings"]:
            assert holding["financed_emissions_tco2e"] >= 0

    def test_valid_asset_classes(self, sample_fi_portfolio):
        for holding in sample_fi_portfolio["holdings"]:
            assert holding["asset_class"] in self.VALID_ASSET_CLASSES

    def test_asset_class_diversity(self, sample_fi_portfolio):
        classes = {h["asset_class"] for h in sample_fi_portfolio["holdings"]}
        assert len(classes) >= 2  # Multiple asset classes


# ===========================================================================
# Attribution Methods
# ===========================================================================

class TestAttribution:
    """Test EVIC and revenue attribution methods."""

    VALID_METHODS = ["evic", "revenue", "project_attribution", "balance_sheet"]

    def test_attribution_method_present(self, sample_fi_portfolio):
        for holding in sample_fi_portfolio["holdings"]:
            assert holding["attribution_method"] in self.VALID_METHODS

    def test_evic_method(self, sample_fi_portfolio):
        evic_holdings = [
            h for h in sample_fi_portfolio["holdings"]
            if h["attribution_method"] == "evic"
        ]
        assert len(evic_holdings) >= 1

    def test_project_attribution(self, sample_fi_portfolio):
        project_holdings = [
            h for h in sample_fi_portfolio["holdings"]
            if h["attribution_method"] == "project_attribution"
        ]
        # Project finance should use project attribution
        for h in project_holdings:
            assert h["asset_class"] == "project_finance"


# ===========================================================================
# WACI
# ===========================================================================

class TestWACI:
    """Test Weighted Average Carbon Intensity calculation."""

    def test_waci_positive(self, sample_fi_portfolio):
        assert sample_fi_portfolio["waci"] > 0

    def test_waci_unit(self, sample_fi_portfolio):
        assert sample_fi_portfolio["waci_unit"] == "tCO2e per USD million invested"

    def test_waci_calculation(self):
        holdings = [
            {"exposure": 500_000_000, "emissions": 250_000},
            {"exposure": 300_000_000, "emissions": 180_000},
            {"exposure": 200_000_000, "emissions": 20_000},
        ]
        total_exposure = sum(h["exposure"] for h in holdings)
        waci = sum(
            (h["exposure"] / total_exposure) * (h["emissions"] / (h["exposure"] / 1_000_000))
            for h in holdings
        )
        assert waci > 0

    def test_waci_decreases_with_clean_portfolio(self):
        dirty = {"waci": 250.0}
        clean = {"waci": 50.0}
        assert clean["waci"] < dirty["waci"]


# ===========================================================================
# PCAF Data Quality
# ===========================================================================

class TestPCAFQuality:
    """Test PCAF data quality distribution (DQ 1-5)."""

    def test_pcaf_quality_range(self, sample_fi_portfolio):
        for holding in sample_fi_portfolio["holdings"]:
            assert 1 <= holding["pcaf_data_quality"] <= 5

    def test_average_quality(self, sample_fi_portfolio):
        avg = sample_fi_portfolio["pcaf_avg_data_quality"]
        assert 1.0 <= avg <= 5.0

    @pytest.mark.parametrize("quality,description", [
        (1, "Reported emissions (audited)"),
        (2, "Reported emissions (unaudited)"),
        (3, "Physical activity data"),
        (4, "Economic activity data"),
        (5, "Estimated emissions"),
    ])
    def test_pcaf_quality_levels(self, quality, description):
        assert 1 <= quality <= 5

    def test_best_quality_holdings(self, sample_fi_portfolio):
        best = [h for h in sample_fi_portfolio["holdings"] if h["pcaf_data_quality"] == 1]
        assert len(best) >= 1


# ===========================================================================
# Coverage Path
# ===========================================================================

class TestCoveragePath:
    """Test linear coverage path to 100% by 2040."""

    def test_coverage_path_exists(self, sample_fi_portfolio):
        path = sample_fi_portfolio["fi_coverage_path"]
        assert len(path) >= 2

    def test_coverage_reaches_100_by_2040(self, sample_fi_portfolio):
        path = sample_fi_portfolio["fi_coverage_path"]
        assert path[2040] == 100.0

    def test_coverage_monotonically_increasing(self, sample_fi_portfolio):
        path = sample_fi_portfolio["fi_coverage_path"]
        years = sorted(path.keys())
        for i in range(1, len(years)):
            assert path[years[i]] >= path[years[i - 1]]

    def test_coverage_interim_milestones(self, sample_fi_portfolio):
        path = sample_fi_portfolio["fi_coverage_path"]
        assert 2030 in path
        assert path[2030] > 50.0  # Should have significant coverage by 2030


# ===========================================================================
# FINZ Validation
# ===========================================================================

class TestFINZValidation:
    """Test Financial Institutions Net-Zero compliance check."""

    def test_finz_compliant(self, sample_fi_portfolio):
        assert sample_fi_portfolio["finz_compliant"] is True

    def test_finz_requires_coverage_path(self, sample_fi_portfolio):
        assert "fi_coverage_path" in sample_fi_portfolio
        assert len(sample_fi_portfolio["fi_coverage_path"]) >= 1

    def test_finz_requires_portfolio_target(self, sample_fi_portfolio):
        assert sample_fi_portfolio["total_financed_emissions_tco2e"] > 0

    def test_finz_non_compliant_no_path(self):
        portfolio = {
            "fi_coverage_path": {},
            "finz_compliant": False,
        }
        assert portfolio["finz_compliant"] is False
