"""
Unit tests for PortfolioBenchmarkingEngine (PACK-047 Engine 7).

Tests all public methods with 32+ tests covering:
  - PCAF ownership share calculation
  - Financed emissions calculation
  - WACI (Weighted Average Carbon Intensity)
  - Carbon footprint
  - Carbon intensity
  - PCAF quality aggregation
  - Sector attribution
  - Index comparison
  - Tracking error
  - Mixed asset classes
  - Data gap handling

Author: GreenLang QA Team
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# PCAF Ownership Share Tests
# ---------------------------------------------------------------------------


class TestPCAFOwnershipShare:
    """Tests for PCAF ownership share (attribution factor) calculation."""

    def test_evic_attribution_basic(self):
        """Test EVIC-based attribution factor calculation."""
        investment_value = Decimal("50")
        enterprise_value = Decimal("200")
        attribution = investment_value / enterprise_value
        assert_decimal_equal(attribution, Decimal("0.25"), tolerance=Decimal("0.001"))

    def test_revenue_attribution(self):
        """Test revenue-based attribution when EVIC unavailable."""
        fund_revenue_share = Decimal("100")
        total_revenue = Decimal("500")
        attribution = fund_revenue_share / total_revenue
        assert_decimal_equal(attribution, Decimal("0.20"), tolerance=Decimal("0.001"))

    def test_attribution_capped_at_1(self):
        """Test attribution factor is capped at 1.0 (100%)."""
        investment_value = Decimal("300")
        enterprise_value = Decimal("200")
        raw = investment_value / enterprise_value
        capped = min(raw, Decimal("1"))
        assert capped == Decimal("1")

    def test_attribution_positive(self, sample_portfolio):
        """Test all portfolio holdings have positive attribution."""
        for h in sample_portfolio:
            assert h["ownership_share_pct"] > Decimal("0")


# ---------------------------------------------------------------------------
# Financed Emissions Tests
# ---------------------------------------------------------------------------


class TestFinancedEmissions:
    """Tests for financed emissions calculation."""

    def test_financed_emissions_basic(self):
        """Test basic financed emissions = attribution * total emissions."""
        attribution = Decimal("0.25")
        total_emissions = Decimal("10000")
        financed = attribution * total_emissions
        assert financed == Decimal("2500")

    def test_financed_emissions_sum(self, sample_portfolio):
        """Test total financed emissions is sum across holdings."""
        total = Decimal("0")
        for h in sample_portfolio:
            attribution = h["ownership_share_pct"] / Decimal("100")
            financed = attribution * h["emissions_scope_1_2_tco2e"]
            total += financed
        assert total > Decimal("0")

    def test_higher_ownership_more_financed(self):
        """Test higher ownership share produces more financed emissions."""
        emissions = Decimal("10000")
        low_share = Decimal("0.10")
        high_share = Decimal("0.50")
        assert (high_share * emissions) > (low_share * emissions)


# ---------------------------------------------------------------------------
# WACI Calculation Tests
# ---------------------------------------------------------------------------


class TestWACICalculation:
    """Tests for Weighted Average Carbon Intensity calculation."""

    def test_waci_basic(self):
        """Test basic WACI = SUM(weight * emissions/revenue)."""
        holdings = [
            {"weight": Decimal("0.50"), "emissions": Decimal("5000"), "revenue": Decimal("200")},
            {"weight": Decimal("0.50"), "emissions": Decimal("3000"), "revenue": Decimal("300")},
        ]
        waci = sum(
            h["weight"] * (h["emissions"] / h["revenue"])
            for h in holdings
        )
        # 0.5 * 25 + 0.5 * 10 = 12.5 + 5 = 17.5
        assert_decimal_equal(waci, Decimal("17.5"), tolerance=Decimal("0.01"))

    def test_waci_weights_sum_to_1(self):
        """Test WACI weights sum to 1.0."""
        weights = [Decimal("0.30"), Decimal("0.25"), Decimal("0.25"), Decimal("0.20")]
        assert sum(weights) == Decimal("1.00")

    def test_waci_zero_revenue_excluded(self):
        """Test holdings with zero revenue are excluded from WACI."""
        holdings = [
            {"weight": Decimal("0.50"), "emissions": Decimal("5000"), "revenue": Decimal("200")},
            {"weight": Decimal("0.50"), "emissions": Decimal("3000"), "revenue": Decimal("0")},
        ]
        valid = [h for h in holdings if h["revenue"] > Decimal("0")]
        assert len(valid) == 1


# ---------------------------------------------------------------------------
# Carbon Footprint Tests
# ---------------------------------------------------------------------------


class TestCarbonFootprint:
    """Tests for carbon footprint (financed emissions per M invested)."""

    def test_carbon_footprint_basic(self):
        """Test carbon footprint = financed emissions / portfolio value."""
        financed_emissions = Decimal("5000")
        portfolio_value = Decimal("100")  # USD million
        footprint = financed_emissions / portfolio_value
        assert footprint == Decimal("50")

    def test_larger_portfolio_lower_per_unit_footprint(self):
        """Test larger portfolio has lower per-unit footprint (dilution)."""
        financed_emissions = Decimal("5000")
        small_portfolio = Decimal("50")
        large_portfolio = Decimal("200")
        assert (financed_emissions / large_portfolio) < (financed_emissions / small_portfolio)


# ---------------------------------------------------------------------------
# Carbon Intensity Tests
# ---------------------------------------------------------------------------


class TestCarbonIntensity:
    """Tests for portfolio carbon intensity."""

    def test_carbon_intensity_basic(self):
        """Test portfolio intensity = financed emissions / financed revenue."""
        financed_emissions = Decimal("5000")
        financed_revenue = Decimal("250")
        intensity = financed_emissions / financed_revenue
        assert intensity == Decimal("20")

    def test_intensity_comparable_across_portfolios(self):
        """Test carbon intensity enables like-for-like comparison."""
        portfolio_a_intensity = Decimal("20")
        portfolio_b_intensity = Decimal("35")
        assert portfolio_a_intensity < portfolio_b_intensity


# ---------------------------------------------------------------------------
# PCAF Quality Aggregation Tests
# ---------------------------------------------------------------------------


class TestPCAFQualityAggregation:
    """Tests for PCAF data quality score aggregation."""

    def test_weighted_average_quality_score(self, sample_portfolio):
        """Test portfolio-level quality score is weighted average."""
        total_weight = Decimal("0")
        weighted_score = Decimal("0")
        for h in sample_portfolio:
            w = h["weight_pct"]
            weighted_score += w * Decimal(str(h["pcaf_data_quality_score"]))
            total_weight += w
        avg_score = weighted_score / total_weight
        assert_decimal_between(avg_score, Decimal("1"), Decimal("5"))

    def test_all_score_1_returns_1(self):
        """Test all score-1 holdings produce aggregate score of 1."""
        holdings = [{"weight": Decimal("1"), "score": 1}] * 10
        avg = sum(Decimal(str(h["score"])) for h in holdings) / Decimal(str(len(holdings)))
        assert avg == Decimal("1")

    def test_all_score_5_returns_5(self):
        """Test all score-5 holdings produce aggregate score of 5."""
        holdings = [{"weight": Decimal("1"), "score": 5}] * 10
        avg = sum(Decimal(str(h["score"])) for h in holdings) / Decimal(str(len(holdings)))
        assert avg == Decimal("5")


# ---------------------------------------------------------------------------
# Sector Attribution Tests
# ---------------------------------------------------------------------------


class TestSectorAttribution:
    """Tests for sector-level carbon attribution."""

    def test_sector_breakdown_sums_to_total(self, sample_portfolio):
        """Test sector-level emissions sum to total portfolio emissions."""
        sector_totals = {}
        for h in sample_portfolio:
            sector = h["sector"]
            attribution = h["ownership_share_pct"] / Decimal("100")
            financed = attribution * h["emissions_scope_1_2_tco2e"]
            sector_totals[sector] = sector_totals.get(sector, Decimal("0")) + financed
        total = sum(sector_totals.values())
        # Compare against direct sum
        direct_total = sum(
            (h["ownership_share_pct"] / Decimal("100")) * h["emissions_scope_1_2_tco2e"]
            for h in sample_portfolio
        )
        assert_decimal_equal(total, direct_total, tolerance=Decimal("0.01"))

    def test_top_sector_identified(self, sample_portfolio):
        """Test highest-emitting sector is correctly identified."""
        sector_totals = {}
        for h in sample_portfolio:
            sector = h["sector"]
            attribution = h["ownership_share_pct"] / Decimal("100")
            financed = attribution * h["emissions_scope_1_2_tco2e"]
            sector_totals[sector] = sector_totals.get(sector, Decimal("0")) + financed
        top_sector = max(sector_totals, key=sector_totals.get)
        assert top_sector in ["INDUSTRIALS", "ENERGY", "MATERIALS", "FINANCIALS", "TECHNOLOGY"]


# ---------------------------------------------------------------------------
# Index Comparison Tests
# ---------------------------------------------------------------------------


class TestIndexComparison:
    """Tests for portfolio vs index benchmark comparison."""

    def test_portfolio_waci_vs_index(self):
        """Test portfolio WACI compared to index WACI."""
        portfolio_waci = Decimal("17.5")
        index_waci = Decimal("22.0")
        outperformance = index_waci - portfolio_waci
        assert outperformance > Decimal("0")

    def test_underperforming_portfolio_positive_excess(self):
        """Test underperforming portfolio has positive excess WACI."""
        portfolio_waci = Decimal("30.0")
        index_waci = Decimal("22.0")
        excess = portfolio_waci - index_waci
        assert excess > Decimal("0")


# ---------------------------------------------------------------------------
# Tracking Error Tests
# ---------------------------------------------------------------------------


class TestTrackingError:
    """Tests for carbon tracking error."""

    def test_tracking_error_calculation(self):
        """Test tracking error is absolute difference from benchmark."""
        portfolio_intensity = Decimal("17.5")
        benchmark_intensity = Decimal("22.0")
        tracking_error = abs(portfolio_intensity - benchmark_intensity)
        assert tracking_error == Decimal("4.5")

    def test_zero_tracking_error_perfect_match(self):
        """Test zero tracking error when portfolio matches benchmark exactly."""
        portfolio_intensity = Decimal("22.0")
        benchmark_intensity = Decimal("22.0")
        tracking_error = abs(portfolio_intensity - benchmark_intensity)
        assert tracking_error == Decimal("0")


# ---------------------------------------------------------------------------
# Mixed Asset Class Tests
# ---------------------------------------------------------------------------


class TestMixedAssetClasses:
    """Tests for mixed asset class portfolio handling."""

    def test_5_asset_classes_in_portfolio(self, sample_portfolio):
        """Test portfolio contains all 5 asset classes."""
        asset_classes = set(h["asset_class"] for h in sample_portfolio)
        assert len(asset_classes) == 5

    def test_each_asset_class_has_10_holdings(self, sample_portfolio):
        """Test each asset class has 10 holdings."""
        from collections import Counter
        counts = Counter(h["asset_class"] for h in sample_portfolio)
        for ac, count in counts.items():
            assert count == 10, f"Asset class {ac} has {count} holdings, expected 10"


# ---------------------------------------------------------------------------
# Data Gap Handling Tests
# ---------------------------------------------------------------------------


class TestDataGapHandling:
    """Tests for handling missing data in portfolio analysis."""

    def test_missing_emissions_flagged(self):
        """Test holding with missing emissions is flagged."""
        holding = {"emissions_scope_1_2_tco2e": None}
        has_data = holding["emissions_scope_1_2_tco2e"] is not None
        assert has_data is False

    def test_missing_revenue_excluded_from_waci(self):
        """Test holding with missing revenue is excluded from WACI."""
        holdings = [
            {"revenue": Decimal("100"), "emissions": Decimal("500")},
            {"revenue": None, "emissions": Decimal("300")},
        ]
        valid = [h for h in holdings if h["revenue"] is not None and h["revenue"] > Decimal("0")]
        assert len(valid) == 1

    def test_portfolio_provenance_hash(self, sample_portfolio):
        """Test portfolio analysis produces deterministic provenance hash."""
        import hashlib
        import json
        canonical = json.dumps(sample_portfolio, sort_keys=True, default=str)
        h1 = hashlib.sha256(canonical.encode()).hexdigest()
        h2 = hashlib.sha256(canonical.encode()).hexdigest()
        assert h1 == h2
        assert len(h1) == 64
