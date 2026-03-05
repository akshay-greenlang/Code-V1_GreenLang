# -*- coding: utf-8 -*-
"""
Unit tests for BaseYearManager -- ISO 14064-1:2018 Clause 5.3 / 7.3.

Tests base year CRUD, locking, recalculation triggers, YoY change,
trend data, and summary with 20+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.base_year_manager import BaseYearManager
from services.config import ISO14064AppConfig


class TestBaseYearCRUD:
    """Test set, get, lock, unlock operations."""

    def test_set_base_year(self, base_year_manager):
        record = base_year_manager.set_base_year(
            org_id="org-1",
            year=2020,
            emissions_tco2e=Decimal("10000"),
            justification="Representative year",
        )
        assert record.base_year == 2020
        assert record.original_emissions_tco2e == Decimal("10000")
        assert record.provenance_hash is not None
        assert len(record.provenance_hash) == 64

    def test_get_base_year(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        record = base_year_manager.get_base_year("org-1")
        assert record is not None
        assert record.base_year == 2020

    def test_get_nonexistent_base_year(self, base_year_manager):
        assert base_year_manager.get_base_year("nonexistent") is None

    def test_lock_base_year(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.lock_base_year("org-1")
        assert base_year_manager.is_locked("org-1") is True

    def test_set_locked_base_year_raises(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.lock_base_year("org-1")
        with pytest.raises(ValueError, match="locked"):
            base_year_manager.set_base_year("org-1", 2021, Decimal("11000"), "New")

    def test_unlock_base_year(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.lock_base_year("org-1")
        base_year_manager.unlock_base_year("org-1")
        assert base_year_manager.is_locked("org-1") is False

    def test_lock_nonexistent_raises(self, base_year_manager):
        with pytest.raises(ValueError, match="No base year"):
            base_year_manager.lock_base_year("nonexistent")


class TestRecalculation:
    """Test recalculation trigger logic."""

    def test_check_recalculation_needed_above_threshold(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        needed = base_year_manager.check_recalculation_needed(
            "org-1", Decimal("10600"), "structural_change",
        )
        # 6% change >= 5% threshold
        assert needed is True

    def test_check_recalculation_not_needed(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        needed = base_year_manager.check_recalculation_needed(
            "org-1", Decimal("10200"), "structural_change",
        )
        # 2% change < 5% threshold
        assert needed is False

    def test_unrecognized_trigger_returns_false(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        needed = base_year_manager.check_recalculation_needed(
            "org-1", Decimal("20000"), "fake_trigger",
        )
        assert needed is False

    def test_check_with_no_base_year(self, base_year_manager):
        needed = base_year_manager.check_recalculation_needed(
            "nonexistent", Decimal("10000"), "structural_change",
        )
        assert needed is False

    def test_recalculate_base_year(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        trigger = base_year_manager.recalculate_base_year(
            org_id="org-1",
            trigger_type="structural_change",
            new_emissions=Decimal("12000"),
            description="Acquired subsidiary",
        )
        assert trigger.trigger_type == "structural_change"
        assert trigger.impact_tco2e == Decimal("2000")
        assert trigger.requires_recalculation is True

        record = base_year_manager.get_base_year("org-1")
        assert record.recalculated_emissions_tco2e == Decimal("12000")

    def test_invalid_trigger_type_raises(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        with pytest.raises(ValueError, match="Invalid trigger"):
            base_year_manager.recalculate_base_year(
                "org-1", "invalid_trigger", Decimal("12000"),
            )

    def test_recalculation_history(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.recalculate_base_year(
            "org-1", "structural_change", Decimal("12000"), "Acquisition",
        )
        base_year_manager.recalculate_base_year(
            "org-1", "methodology_change", Decimal("11500"), "Updated EFs",
        )
        history = base_year_manager.get_recalculation_history("org-1")
        assert len(history) == 2

    def test_recognized_triggers_list(self):
        assert "structural_change" in BaseYearManager.RECOGNIZED_TRIGGERS
        assert "acquisition" in BaseYearManager.RECOGNIZED_TRIGGERS
        assert len(BaseYearManager.RECOGNIZED_TRIGGERS) == 8


class TestYoYChange:
    """Test year-over-year change calculation."""

    def test_yoy_increase(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        result = base_year_manager.calculate_yoy_change("org-1", Decimal("12000"))
        assert result["direction"] == "increase"
        assert result["change_pct"] == "20.00"

    def test_yoy_decrease(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        result = base_year_manager.calculate_yoy_change("org-1", Decimal("8000"))
        assert result["direction"] == "decrease"
        assert result["change_pct"] == "-20.00"

    def test_yoy_no_base_year(self, base_year_manager):
        result = base_year_manager.calculate_yoy_change("nonexistent", Decimal("5000"))
        assert "error" in result

    def test_yoy_uses_recalculated_emissions(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.recalculate_base_year(
            "org-1", "structural_change", Decimal("12000"), "Acquisition",
        )
        result = base_year_manager.calculate_yoy_change("org-1", Decimal("12000"))
        # Change from recalculated (12000) -> 12000 = 0%
        assert result["change_pct"] == "0.00"


class TestTrendData:
    """Test multi-year trend calculation."""

    def test_trend_data(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        yearly = {
            2021: Decimal("10500"),
            2022: Decimal("11000"),
            2023: Decimal("9500"),
        }
        trends = base_year_manager.get_trend_data("org-1", yearly)
        assert len(trends) == 3
        # 2023: (9500-10000)/10000*100 = -5%
        assert trends[2]["change_from_base_pct"] == "-5.00"

    def test_trend_without_base_year(self, base_year_manager):
        yearly = {2021: Decimal("5000")}
        trends = base_year_manager.get_trend_data("nonexistent", yearly)
        assert len(trends) == 1
        # No base year => no change_from_base_pct
        assert "change_from_base_pct" not in trends[0]


class TestBaseYearSummary:
    """Test summary generation."""

    def test_summary_with_base_year(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.lock_base_year("org-1")
        summary = base_year_manager.get_base_year_summary("org-1")
        assert summary["base_year"] == 2020
        assert summary["locked"] is True
        assert summary["original_emissions_tco2e"] == "10000"
        assert summary["provenance_hash"] is not None

    def test_summary_without_base_year(self, base_year_manager):
        summary = base_year_manager.get_base_year_summary("nonexistent")
        assert summary["message"] == "No base year set"

    def test_summary_with_recalculation(self, base_year_manager):
        base_year_manager.set_base_year("org-1", 2020, Decimal("10000"), "Reason")
        base_year_manager.recalculate_base_year(
            "org-1", "structural_change", Decimal("12000"), "Acquisition",
        )
        summary = base_year_manager.get_base_year_summary("org-1")
        assert summary["recalculated_emissions_tco2e"] == "12000"
        assert summary["current_emissions_tco2e"] == "12000"
        assert summary["recalculation_count"] == 1
