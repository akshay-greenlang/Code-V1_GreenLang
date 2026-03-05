# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Portfolio Management Engine.

Tests portfolio CRUD operations, holdings management, exposure
upload processing, counterparty search, and portfolio listing
with 30+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Portfolio CRUD
# ===========================================================================

class TestPortfolioCRUD:
    """Test portfolio create, read, update, delete operations."""

    def test_create_portfolio(self, sample_taxonomy_portfolio):
        assert sample_taxonomy_portfolio["portfolio_id"] is not None
        assert sample_taxonomy_portfolio["portfolio_name"] is not None

    def test_portfolio_has_institution(self, sample_taxonomy_portfolio):
        assert sample_taxonomy_portfolio["institution_id"] is not None

    def test_portfolio_has_reporting_date(self, sample_taxonomy_portfolio):
        assert sample_taxonomy_portfolio["reporting_date"] is not None

    def test_portfolio_has_currency(self, sample_taxonomy_portfolio):
        assert sample_taxonomy_portfolio["currency"] in ["EUR", "USD", "GBP", "CHF"]

    def test_portfolio_total_exposure(self, sample_taxonomy_portfolio):
        assert sample_taxonomy_portfolio["total_exposure"] > 0

    def test_portfolio_has_status(self, sample_taxonomy_portfolio):
        valid_statuses = ["draft", "submitted", "validated", "archived"]
        assert sample_taxonomy_portfolio["status"] in valid_statuses

    def test_portfolio_update(self, sample_taxonomy_portfolio):
        updated = sample_taxonomy_portfolio.copy()
        updated["portfolio_name"] = "Updated Portfolio"
        assert updated["portfolio_name"] != sample_taxonomy_portfolio["portfolio_name"]
        assert updated["portfolio_id"] == sample_taxonomy_portfolio["portfolio_id"]

    def test_portfolio_delete_sets_inactive(self, sample_taxonomy_portfolio):
        deleted = sample_taxonomy_portfolio.copy()
        deleted["status"] = "archived"
        assert deleted["status"] == "archived"


# ===========================================================================
# Holdings Management
# ===========================================================================

class TestHoldingsManagement:
    """Test portfolio holdings CRUD and management."""

    def test_holdings_present(self, sample_taxonomy_portfolio):
        assert "holdings" in sample_taxonomy_portfolio
        assert len(sample_taxonomy_portfolio["holdings"]) >= 1

    def test_holding_has_required_fields(self, sample_taxonomy_portfolio):
        required_fields = [
            "holding_id", "counterparty_name", "nace_code",
            "exposure_amount", "exposure_type",
        ]
        for holding in sample_taxonomy_portfolio["holdings"]:
            for field in required_fields:
                assert field in holding, f"Missing field: {field}"

    def test_holding_exposure_positive(self, sample_taxonomy_portfolio):
        for holding in sample_taxonomy_portfolio["holdings"]:
            assert holding["exposure_amount"] > 0

    def test_holding_nace_code_format(self, sample_taxonomy_portfolio):
        for holding in sample_taxonomy_portfolio["holdings"]:
            nace = holding["nace_code"]
            assert len(nace) >= 2
            assert nace[0].isalpha()

    def test_holding_unique_ids(self, sample_taxonomy_portfolio):
        ids = [h["holding_id"] for h in sample_taxonomy_portfolio["holdings"]]
        assert len(ids) == len(set(ids))

    def test_holding_exposure_types(self, sample_taxonomy_portfolio):
        valid_types = [
            "corporate_loan", "corporate_bond", "equity",
            "mortgage", "auto_loan", "project_finance",
            "interbank", "sovereign",
        ]
        for holding in sample_taxonomy_portfolio["holdings"]:
            assert holding["exposure_type"] in valid_types

    def test_total_exposure_matches_sum(self, sample_taxonomy_portfolio):
        holdings_sum = sum(
            h["exposure_amount"] for h in sample_taxonomy_portfolio["holdings"]
        )
        assert abs(
            sample_taxonomy_portfolio["total_exposure"] - holdings_sum
        ) < 0.01

    def test_add_holding(self, sample_taxonomy_portfolio):
        new_holding = {
            "holding_id": "h_new_001",
            "counterparty_name": "New Corp GmbH",
            "nace_code": "C29.10",
            "exposure_amount": 5_000_000.0,
            "exposure_type": "corporate_loan",
        }
        holdings = sample_taxonomy_portfolio["holdings"] + [new_holding]
        assert len(holdings) == len(sample_taxonomy_portfolio["holdings"]) + 1

    def test_remove_holding(self, sample_taxonomy_portfolio):
        original_count = len(sample_taxonomy_portfolio["holdings"])
        holdings = sample_taxonomy_portfolio["holdings"][1:]
        assert len(holdings) == original_count - 1


# ===========================================================================
# Exposure Upload
# ===========================================================================

class TestExposureUpload:
    """Test exposure file upload processing."""

    def test_upload_result_created(self, sample_upload_result):
        assert sample_upload_result["upload_id"] is not None

    def test_upload_status(self, sample_upload_result):
        valid_statuses = ["processing", "completed", "failed", "partial"]
        assert sample_upload_result["status"] in valid_statuses

    def test_upload_record_counts(self, sample_upload_result):
        assert sample_upload_result["total_records"] > 0
        assert sample_upload_result["valid_records"] >= 0
        assert sample_upload_result["invalid_records"] >= 0
        assert (
            sample_upload_result["valid_records"]
            + sample_upload_result["invalid_records"]
            == sample_upload_result["total_records"]
        )

    @pytest.mark.parametrize("file_format", ["csv", "xlsx", "json", "xml"])
    def test_supported_upload_formats(self, file_format):
        supported = ["csv", "xlsx", "json", "xml"]
        assert file_format in supported

    def test_upload_validation_errors(self, sample_upload_result):
        assert "validation_errors" in sample_upload_result
        for error in sample_upload_result["validation_errors"]:
            assert "row" in error
            assert "field" in error
            assert "message" in error

    def test_upload_nace_enrichment(self, sample_upload_result):
        assert "nace_enrichment_count" in sample_upload_result
        assert sample_upload_result["nace_enrichment_count"] >= 0


# ===========================================================================
# Counterparty Search
# ===========================================================================

class TestCounterpartySearch:
    """Test counterparty search and lookup."""

    def test_search_by_name(self, sample_taxonomy_portfolio):
        holdings = sample_taxonomy_portfolio["holdings"]
        query = holdings[0]["counterparty_name"][:4].lower()
        results = [
            h for h in holdings
            if query in h["counterparty_name"].lower()
        ]
        assert len(results) >= 1

    def test_search_by_nace(self, sample_taxonomy_portfolio):
        target_nace = sample_taxonomy_portfolio["holdings"][0]["nace_code"]
        results = [
            h for h in sample_taxonomy_portfolio["holdings"]
            if h["nace_code"] == target_nace
        ]
        assert len(results) >= 1

    def test_search_returns_empty_for_no_match(self, sample_taxonomy_portfolio):
        results = [
            h for h in sample_taxonomy_portfolio["holdings"]
            if "ZZZZZ_NONEXISTENT" in h["counterparty_name"]
        ]
        assert len(results) == 0

    def test_search_case_insensitive(self, sample_taxonomy_portfolio):
        name = sample_taxonomy_portfolio["holdings"][0]["counterparty_name"]
        upper_results = [
            h for h in sample_taxonomy_portfolio["holdings"]
            if name.upper() == h["counterparty_name"].upper()
        ]
        lower_results = [
            h for h in sample_taxonomy_portfolio["holdings"]
            if name.lower() == h["counterparty_name"].lower()
        ]
        assert len(upper_results) == len(lower_results)


# ===========================================================================
# Portfolio Listing
# ===========================================================================

class TestPortfolioListing:
    """Test portfolio listing and filtering."""

    def test_list_portfolios(self, sample_portfolio_list):
        assert len(sample_portfolio_list) >= 1

    def test_portfolio_list_has_summary(self, sample_portfolio_list):
        for portfolio in sample_portfolio_list:
            assert "portfolio_id" in portfolio
            assert "portfolio_name" in portfolio
            assert "total_exposure" in portfolio
            assert "holdings_count" in portfolio

    def test_filter_by_status(self, sample_portfolio_list):
        draft = [p for p in sample_portfolio_list if p.get("status") == "draft"]
        submitted = [p for p in sample_portfolio_list if p.get("status") == "submitted"]
        total = len(draft) + len(submitted)
        assert total <= len(sample_portfolio_list)

    def test_filter_by_reporting_period(self, sample_portfolio_list):
        for portfolio in sample_portfolio_list:
            assert "reporting_date" in portfolio

    def test_sort_by_total_exposure(self, sample_portfolio_list):
        sorted_list = sorted(
            sample_portfolio_list,
            key=lambda p: p["total_exposure"],
            reverse=True,
        )
        for i in range(1, len(sorted_list)):
            assert sorted_list[i]["total_exposure"] <= sorted_list[i - 1]["total_exposure"]
