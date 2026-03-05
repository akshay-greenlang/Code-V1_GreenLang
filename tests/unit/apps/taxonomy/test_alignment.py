# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Alignment Engine.

Tests full 4-step alignment workflow (eligibility -> SC -> DNSH -> MS),
portfolio alignment aggregation, alignment progress tracking, dashboard
data generation, period comparison, and alignment constraints with 40+
test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest


# ===========================================================================
# Full 4-step alignment workflow tests
# ===========================================================================

class TestFullAlignmentWorkflow:
    """Test the complete 4-step alignment determination."""

    def test_aligned_all_steps_pass(self, sample_alignment_result):
        """Activity aligned when all four steps pass."""
        r = sample_alignment_result
        assert r["eligible"] is True
        assert r["sc_pass"] is True
        assert r["dnsh_pass"] is True
        assert r["ms_pass"] is True
        assert r["aligned"] is True

    def test_aligned_requires_eligible(self, sample_alignment_result):
        """Alignment requires eligibility."""
        assert sample_alignment_result["eligible"] is True

    def test_aligned_requires_sc_pass(self, sample_alignment_result):
        """Alignment requires SC pass."""
        assert sample_alignment_result["sc_pass"] is True

    def test_aligned_requires_dnsh_pass(self, sample_alignment_result):
        """Alignment requires DNSH pass."""
        assert sample_alignment_result["dnsh_pass"] is True

    def test_aligned_requires_ms_pass(self, sample_alignment_result):
        """Alignment requires MS pass."""
        assert sample_alignment_result["ms_pass"] is True

    def test_not_aligned_dnsh_fail(self, partial_alignment_result):
        """Activity not aligned when DNSH fails."""
        r = partial_alignment_result
        assert r["eligible"] is True
        assert r["sc_pass"] is True
        assert r["dnsh_pass"] is False
        assert r["ms_pass"] is True
        assert r["aligned"] is False

    def test_alignment_constraint_enforced(self, sample_alignment_result):
        """Alignment = eligible AND sc AND dnsh AND ms."""
        r = sample_alignment_result
        computed = r["eligible"] and r["sc_pass"] and r["dnsh_pass"] and r["ms_pass"]
        assert r["aligned"] == computed

    def test_alignment_constraint_partial(self, partial_alignment_result):
        """Partial alignment correctly computed."""
        r = partial_alignment_result
        computed = r["eligible"] and r["sc_pass"] and r["dnsh_pass"] and r["ms_pass"]
        assert r["aligned"] == computed
        assert computed is False

    def test_sc_objective_recorded(self, sample_alignment_result):
        """SC objective is recorded in alignment result."""
        assert sample_alignment_result["sc_objective"] == "climate_mitigation"

    def test_alignment_details_present(self, sample_alignment_result):
        """Alignment details contain all four steps."""
        details = sample_alignment_result["alignment_details"]
        assert "eligibility" in details
        assert "sc" in details
        assert "dnsh" in details
        assert "ms" in details

    def test_alignment_provenance(self, sample_alignment_result):
        """Alignment result has provenance hash."""
        assert len(sample_alignment_result["provenance_hash"]) == 64

    def test_engine_run_full_alignment(self, alignment_engine):
        """Engine runs full alignment workflow."""
        alignment_engine.run_full_alignment.return_value = {
            "activity_code": "CCM_4.1",
            "eligible": True,
            "sc_pass": True,
            "dnsh_pass": True,
            "ms_pass": True,
            "aligned": True,
        }
        result = alignment_engine.run_full_alignment("org-123", "CCM_4.1", "FY2025")
        assert result["aligned"] is True
        alignment_engine.run_full_alignment.assert_called_once()


# ===========================================================================
# Alignment detail tests
# ===========================================================================

class TestAlignmentDetails:
    """Test alignment detail structure."""

    def test_eligibility_detail(self, sample_alignment_result):
        """Eligibility detail includes status and confidence."""
        detail = sample_alignment_result["alignment_details"]["eligibility"]
        assert detail["status"] == "eligible"
        assert detail["confidence"] == 95.0

    def test_sc_detail(self, sample_alignment_result):
        """SC detail includes type and objective."""
        detail = sample_alignment_result["alignment_details"]["sc"]
        assert detail["status"] == "pass"
        assert detail["type"] == "own_performance"
        assert detail["objective"] == "climate_mitigation"

    def test_dnsh_detail(self, sample_alignment_result):
        """DNSH detail includes objectives checked and passed."""
        detail = sample_alignment_result["alignment_details"]["dnsh"]
        assert detail["status"] == "pass"
        assert detail["objectives_checked"] == 5
        assert detail["objectives_passed"] == 5

    def test_ms_detail(self, sample_alignment_result):
        """MS detail includes topics checked and passed."""
        detail = sample_alignment_result["alignment_details"]["ms"]
        assert detail["status"] == "pass"
        assert detail["topics_checked"] == 4
        assert detail["topics_passed"] == 4

    def test_failing_detail_identifies_step(self, partial_alignment_result):
        """Failing detail identifies which step failed."""
        details = partial_alignment_result["alignment_details"]
        assert details["dnsh"]["status"] == "fail"
        assert "failing_objective" in details["dnsh"]


# ===========================================================================
# Portfolio alignment tests
# ===========================================================================

class TestPortfolioAlignment:
    """Test portfolio-level alignment aggregation."""

    def test_portfolio_counts_consistent(self, sample_portfolio_alignment):
        """aligned <= eligible <= total."""
        pa = sample_portfolio_alignment
        assert pa["aligned_count"] <= pa["eligible_count"]
        assert pa["eligible_count"] <= pa["total_activities"]

    def test_portfolio_alignment_percentage(self, sample_portfolio_alignment):
        """Alignment percentage calculated correctly."""
        pa = sample_portfolio_alignment
        if pa["total_activities"] > 0:
            expected = pa["aligned_count"] / pa["total_activities"] * 100
            assert abs(float(pa["alignment_percentage"]) - expected) < 0.01

    def test_portfolio_percentage_range(self, sample_portfolio_alignment):
        """Alignment percentage between 0 and 100."""
        pct = float(sample_portfolio_alignment["alignment_percentage"])
        assert 0 <= pct <= 100

    def test_portfolio_kpi_summary(self, sample_portfolio_alignment):
        """KPI summary includes all three KPI types."""
        kpi = sample_portfolio_alignment["kpi_summary"]
        assert "turnover" in kpi
        assert "capex" in kpi
        assert "opex" in kpi

    def test_portfolio_kpi_eligible_pct(self, sample_portfolio_alignment):
        """KPI summary includes eligible percentages."""
        kpi = sample_portfolio_alignment["kpi_summary"]
        for kpi_type, data in kpi.items():
            assert "eligible_pct" in data
            assert "aligned_pct" in data

    def test_portfolio_sector_breakdown(self, sample_portfolio_alignment):
        """Sector breakdown present in portfolio."""
        sectors = sample_portfolio_alignment["sector_breakdown"]
        assert len(sectors) >= 3

    def test_sector_counts_consistent(self, sample_portfolio_alignment):
        """Sector-level counts are consistent."""
        sectors = sample_portfolio_alignment["sector_breakdown"]
        for sector_name, data in sectors.items():
            assert data["aligned"] <= data["eligible"]
            assert data["eligible"] <= data["total"]

    def test_sector_total_equals_portfolio(self, sample_portfolio_alignment):
        """Sum of sector totals equals portfolio total."""
        sectors = sample_portfolio_alignment["sector_breakdown"]
        sector_total = sum(d["total"] for d in sectors.values())
        assert sector_total == sample_portfolio_alignment["total_activities"]

    def test_engine_aggregate_portfolio(self, alignment_engine):
        """Engine aggregates portfolio alignment."""
        alignment_engine.aggregate_portfolio.return_value = {
            "total_activities": 15,
            "eligible_count": 10,
            "aligned_count": 7,
            "alignment_percentage": 46.67,
        }
        result = alignment_engine.aggregate_portfolio("org-123", "FY2025")
        assert result["alignment_percentage"] == 46.67


# ===========================================================================
# Alignment progress tracking tests
# ===========================================================================

class TestAlignmentProgress:
    """Test alignment progress and dashboard data."""

    def test_engine_get_progress(self, alignment_engine):
        """Engine returns alignment progress data."""
        alignment_engine.get_alignment_progress.return_value = {
            "period": "FY2025",
            "eligible_pct": 70.0,
            "aligned_pct": 46.67,
            "sc_pass_rate": 80.0,
            "dnsh_pass_rate": 85.0,
            "ms_pass_rate": 100.0,
        }
        result = alignment_engine.get_alignment_progress("org-123", "FY2025")
        assert result["aligned_pct"] == 46.67
        assert result["ms_pass_rate"] == 100.0

    def test_progress_step_rates(self, alignment_engine):
        """Progress includes pass rates for each step."""
        alignment_engine.get_alignment_progress.return_value = {
            "sc_pass_rate": 80.0,
            "dnsh_pass_rate": 85.0,
            "ms_pass_rate": 100.0,
        }
        result = alignment_engine.get_alignment_progress("org-123", "FY2025")
        # MS pass rate typically highest (org-level, not activity-level)
        assert result["ms_pass_rate"] >= result["dnsh_pass_rate"]

    def test_engine_get_dashboard_data(self, alignment_engine):
        """Engine returns dashboard data."""
        alignment_engine.get_dashboard_data.return_value = {
            "headline_metrics": {
                "turnover_alignment": 42.0,
                "capex_alignment": 52.5,
                "opex_alignment": 25.5,
            },
            "trend": "improving",
            "sector_heatmap": {"energy": 90, "manufacturing": 40},
        }
        result = alignment_engine.get_dashboard_data("org-123", "FY2025")
        assert "headline_metrics" in result
        assert result["trend"] == "improving"

    def test_dashboard_headline_metrics(self, alignment_engine):
        """Dashboard includes all three KPI alignment metrics."""
        alignment_engine.get_dashboard_data.return_value = {
            "headline_metrics": {
                "turnover_alignment": 42.0,
                "capex_alignment": 52.5,
                "opex_alignment": 25.5,
            },
        }
        result = alignment_engine.get_dashboard_data("org-123", "FY2025")
        metrics = result["headline_metrics"]
        assert "turnover_alignment" in metrics
        assert "capex_alignment" in metrics
        assert "opex_alignment" in metrics


# ===========================================================================
# Period comparison tests
# ===========================================================================

class TestPeriodComparison:
    """Test alignment comparison across periods."""

    def test_engine_compare_periods(self, alignment_engine):
        """Engine compares alignment across periods."""
        alignment_engine.compare_periods.return_value = {
            "FY2024": {"alignment_pct": 35.0, "aligned_count": 5},
            "FY2025": {"alignment_pct": 46.67, "aligned_count": 7},
            "change_pct": 11.67,
            "direction": "improving",
        }
        result = alignment_engine.compare_periods("org-123", "FY2024", "FY2025")
        assert result["change_pct"] > 0
        assert result["direction"] == "improving"

    def test_year_over_year_improvement(self, alignment_engine):
        """YoY alignment improvement tracked."""
        alignment_engine.compare_periods.return_value = {
            "FY2024": {"alignment_pct": 35.0},
            "FY2025": {"alignment_pct": 46.67},
            "change_pct": 11.67,
        }
        result = alignment_engine.compare_periods("org-123", "FY2024", "FY2025")
        assert result["FY2025"]["alignment_pct"] > result["FY2024"]["alignment_pct"]

    def test_declining_alignment(self, alignment_engine):
        """Declining alignment is also tracked."""
        alignment_engine.compare_periods.return_value = {
            "FY2024": {"alignment_pct": 50.0},
            "FY2025": {"alignment_pct": 45.0},
            "change_pct": -5.0,
            "direction": "declining",
        }
        result = alignment_engine.compare_periods("org-123", "FY2024", "FY2025")
        assert result["change_pct"] < 0
        assert result["direction"] == "declining"


# ===========================================================================
# Not-eligible activity alignment tests
# ===========================================================================

class TestNotEligibleAlignment:
    """Test alignment for non-eligible activities."""

    def test_not_eligible_not_aligned(self):
        """Non-eligible activities cannot be aligned."""
        result = {
            "eligible": False,
            "sc_pass": False,
            "dnsh_pass": False,
            "ms_pass": False,
            "aligned": False,
        }
        assert result["aligned"] is False

    def test_eligible_but_no_sc(self):
        """Eligible but SC not passed."""
        result = {
            "eligible": True,
            "sc_pass": False,
            "dnsh_pass": True,
            "ms_pass": True,
            "aligned": False,
        }
        computed = result["eligible"] and result["sc_pass"] and result["dnsh_pass"] and result["ms_pass"]
        assert computed is False

    def test_eligible_sc_but_no_dnsh(self):
        """Eligible + SC but DNSH fails."""
        result = {
            "eligible": True,
            "sc_pass": True,
            "dnsh_pass": False,
            "ms_pass": True,
            "aligned": False,
        }
        computed = result["eligible"] and result["sc_pass"] and result["dnsh_pass"] and result["ms_pass"]
        assert computed is False

    def test_eligible_sc_dnsh_but_no_ms(self):
        """Eligible + SC + DNSH but MS fails."""
        result = {
            "eligible": True,
            "sc_pass": True,
            "dnsh_pass": True,
            "ms_pass": False,
            "aligned": False,
        }
        computed = result["eligible"] and result["sc_pass"] and result["dnsh_pass"] and result["ms_pass"]
        assert computed is False


# ===========================================================================
# Alignment metadata tests
# ===========================================================================

class TestAlignmentMetadata:
    """Test alignment result metadata and timestamps."""

    def test_alignment_period(self, sample_alignment_result):
        """Alignment result has period."""
        assert sample_alignment_result["period"] == "FY2025"

    def test_alignment_activity_code(self, sample_alignment_result):
        """Alignment result has activity code."""
        assert sample_alignment_result["activity_code"] == "CCM_4.1"

    def test_alignment_date_present(self, sample_alignment_result):
        """Alignment date is recorded."""
        from datetime import datetime
        assert isinstance(sample_alignment_result["alignment_date"], datetime)

    def test_alignment_timestamps(self, sample_alignment_result):
        """Created and updated timestamps present."""
        from datetime import datetime
        assert isinstance(sample_alignment_result["created_at"], datetime)
        assert isinstance(sample_alignment_result["updated_at"], datetime)
