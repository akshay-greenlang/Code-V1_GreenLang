# -*- coding: utf-8 -*-
"""
Unit tests for DataConnector -- MRV agent data auto-population.

Tests MRV agent mapping to CDP format, Scope 1/2/3 auto-population,
unit conversion, data freshness validation, manual override tracking,
and reconciliation with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from services.models import CDPResponse, _new_id
from services.data_connector import DataConnector


# ---------------------------------------------------------------------------
# MRV agent mapping
# ---------------------------------------------------------------------------

class TestMRVMapping:
    """Test MRV agent to CDP format mapping."""

    def test_scope1_agent_mapping(self, data_connector):
        mapping = data_connector.get_mrv_mapping("scope_1")
        assert "MRV-001" in mapping  # Stationary Combustion
        assert "MRV-002" in mapping  # Refrigerants
        assert "MRV-003" in mapping  # Mobile Combustion
        assert "MRV-004" in mapping  # Process Emissions
        assert "MRV-005" in mapping  # Fugitive Emissions

    def test_scope1_land_use_mapping(self, data_connector):
        mapping = data_connector.get_mrv_mapping("scope_1_land")
        assert "MRV-006" in mapping  # Land Use
        assert "MRV-007" in mapping  # Waste Treatment
        assert "MRV-008" in mapping  # Agricultural

    def test_scope2_agent_mapping(self, data_connector):
        mapping = data_connector.get_mrv_mapping("scope_2")
        assert "MRV-009" in mapping  # Location-Based
        assert "MRV-010" in mapping  # Market-Based
        assert "MRV-011" in mapping  # Steam/Heat
        assert "MRV-012" in mapping  # Cooling
        assert "MRV-013" in mapping  # Dual Reporting

    def test_scope3_upstream_mapping(self, data_connector):
        mapping = data_connector.get_mrv_mapping("scope_3_upstream")
        for agent_num in range(14, 22):  # MRV-014 through MRV-021
            assert f"MRV-{agent_num:03d}" in mapping

    def test_scope3_downstream_mapping(self, data_connector):
        mapping = data_connector.get_mrv_mapping("scope_3_downstream")
        for agent_num in range(22, 29):  # MRV-022 through MRV-028
            assert f"MRV-{agent_num:03d}" in mapping

    def test_cross_cutting_mapping(self, data_connector):
        mapping = data_connector.get_mrv_mapping("cross_cutting")
        assert "MRV-029" in mapping  # Category Mapper
        assert "MRV-030" in mapping  # Audit Trail


# ---------------------------------------------------------------------------
# Scope 1 auto-population
# ---------------------------------------------------------------------------

class TestScope1AutoPopulation:
    """Test Scope 1 data auto-population from MRV agents."""

    def test_populate_stationary_combustion(self, data_connector):
        mock_data = {
            "total_tco2e": Decimal("5000.5"),
            "fuel_types": ["natural_gas", "diesel"],
            "method": "tier_2",
        }
        with patch.object(data_connector, "_fetch_mrv_data", return_value=mock_data):
            result = data_connector.auto_populate_scope1_stationary(
                org_id=_new_id(), reporting_year=2025,
            )
            assert result["total_tco2e"] == Decimal("5000.5")
            assert result["source"] == "MRV-001"
            assert result["auto_populated"] is True

    def test_populate_mobile_combustion(self, data_connector):
        mock_data = {
            "total_tco2e": Decimal("2500.0"),
            "vehicle_types": ["truck", "car"],
        }
        with patch.object(data_connector, "_fetch_mrv_data", return_value=mock_data):
            result = data_connector.auto_populate_scope1_mobile(
                org_id=_new_id(), reporting_year=2025,
            )
            assert result["total_tco2e"] == Decimal("2500.0")
            assert result["source"] == "MRV-003"

    def test_populate_refrigerants(self, data_connector):
        mock_data = {"total_tco2e": Decimal("800.0"), "gas_types": ["R-134a"]}
        with patch.object(data_connector, "_fetch_mrv_data", return_value=mock_data):
            result = data_connector.auto_populate_scope1_refrigerants(
                org_id=_new_id(), reporting_year=2025,
            )
            assert result["total_tco2e"] == Decimal("800.0")
            assert result["source"] == "MRV-002"

    def test_aggregate_scope1_total(self, data_connector):
        components = {
            "stationary": Decimal("5000"),
            "mobile": Decimal("2500"),
            "refrigerants": Decimal("800"),
            "process": Decimal("1200"),
            "fugitive": Decimal("300"),
        }
        total = data_connector.aggregate_scope1_total(components)
        assert total == Decimal("9800")


# ---------------------------------------------------------------------------
# Scope 2 auto-population
# ---------------------------------------------------------------------------

class TestScope2AutoPopulation:
    """Test Scope 2 data auto-population from MRV agents."""

    def test_populate_location_based(self, data_connector):
        mock_data = {"total_tco2e": Decimal("3500.0"), "grid_regions": ["US-RFCW"]}
        with patch.object(data_connector, "_fetch_mrv_data", return_value=mock_data):
            result = data_connector.auto_populate_scope2_location(
                org_id=_new_id(), reporting_year=2025,
            )
            assert result["total_tco2e"] == Decimal("3500.0")
            assert result["source"] == "MRV-009"

    def test_populate_market_based(self, data_connector):
        mock_data = {"total_tco2e": Decimal("2800.0"), "instruments": ["REC"]}
        with patch.object(data_connector, "_fetch_mrv_data", return_value=mock_data):
            result = data_connector.auto_populate_scope2_market(
                org_id=_new_id(), reporting_year=2025,
            )
            assert result["total_tco2e"] == Decimal("2800.0")
            assert result["source"] == "MRV-010"

    def test_dual_reporting_reconciliation(self, data_connector):
        location = Decimal("3500.0")
        market = Decimal("2800.0")
        recon = data_connector.reconcile_scope2_dual(location, market)
        assert recon["location_based"] == Decimal("3500.0")
        assert recon["market_based"] == Decimal("2800.0")
        assert recon["difference"] == Decimal("700.0")


# ---------------------------------------------------------------------------
# Scope 3 auto-population
# ---------------------------------------------------------------------------

class TestScope3AutoPopulation:
    """Test Scope 3 data auto-population from MRV agents."""

    @pytest.mark.parametrize("category,agent_id", [
        (1, "MRV-014"),   # Purchased Goods
        (2, "MRV-015"),   # Capital Goods
        (3, "MRV-016"),   # Fuel & Energy
        (4, "MRV-017"),   # Upstream Transport
        (5, "MRV-018"),   # Waste Generated
        (6, "MRV-019"),   # Business Travel
        (7, "MRV-020"),   # Employee Commuting
        (8, "MRV-021"),   # Upstream Leased
        (9, "MRV-022"),   # Downstream Transport
        (10, "MRV-023"),  # Processing Sold Products
        (11, "MRV-024"),  # Use of Sold Products
        (12, "MRV-025"),  # End-of-Life Treatment
        (13, "MRV-026"),  # Downstream Leased
        (14, "MRV-027"),  # Franchises
        (15, "MRV-028"),  # Investments
    ])
    def test_scope3_category_mapping(self, data_connector, category, agent_id):
        mapping = data_connector.get_scope3_category_agent(category)
        assert mapping == agent_id

    def test_aggregate_scope3_total(self, data_connector):
        category_data = {
            f"cat_{i}": Decimal(str(i * 100)) for i in range(1, 16)
        }
        total = data_connector.aggregate_scope3_total(category_data)
        expected = sum(Decimal(str(i * 100)) for i in range(1, 16))
        assert total == expected


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

class TestUnitConversion:
    """Test unit conversion for CDP format."""

    def test_tonnes_to_metric_tonnes(self, data_connector):
        result = data_connector.convert_units(
            value=Decimal("1000"), from_unit="kg_co2e", to_unit="tco2e",
        )
        assert result == Decimal("1.0")

    def test_metric_tonnes_identity(self, data_connector):
        result = data_connector.convert_units(
            value=Decimal("500"), from_unit="tco2e", to_unit="tco2e",
        )
        assert result == Decimal("500")

    def test_mwh_to_gj(self, data_connector):
        result = data_connector.convert_units(
            value=Decimal("100"), from_unit="MWh", to_unit="GJ",
        )
        assert result == Decimal("360.0")


# ---------------------------------------------------------------------------
# Data freshness validation
# ---------------------------------------------------------------------------

class TestDataFreshness:
    """Test data freshness validation."""

    def test_fresh_data_passes(self, data_connector):
        is_fresh = data_connector.validate_data_freshness(
            data_timestamp=datetime.now() - timedelta(days=30),
            reporting_year=2025,
        )
        assert is_fresh is True

    def test_stale_data_fails(self, data_connector):
        is_fresh = data_connector.validate_data_freshness(
            data_timestamp=datetime.now() - timedelta(days=730),
            reporting_year=2025,
        )
        assert is_fresh is False

    def test_freshness_alert_generated(self, data_connector):
        alert = data_connector.generate_freshness_alert(
            data_timestamp=datetime.now() - timedelta(days=400),
            source="MRV-001",
            reporting_year=2025,
        )
        assert alert is not None
        assert "stale" in alert["message"].lower() or "outdated" in alert["message"].lower()


# ---------------------------------------------------------------------------
# Manual override tracking
# ---------------------------------------------------------------------------

class TestManualOverride:
    """Test manual override with justification tracking."""

    def test_track_manual_override(self, data_connector):
        override = data_connector.track_manual_override(
            question_id=_new_id(),
            auto_value=Decimal("5000"),
            manual_value=Decimal("5200"),
            justification="Corrected for late-arriving facility data",
            overridden_by=_new_id(),
        )
        assert override["auto_value"] == Decimal("5000")
        assert override["manual_value"] == Decimal("5200")
        assert override["justification"] is not None

    def test_override_records_delta(self, data_connector):
        override = data_connector.track_manual_override(
            question_id=_new_id(),
            auto_value=Decimal("5000"),
            manual_value=Decimal("5200"),
            justification="Correction",
            overridden_by=_new_id(),
        )
        assert override["delta"] == Decimal("200")
        assert override["delta_pct"] == pytest.approx(4.0, rel=0.1)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

class TestReconciliation:
    """Test auto vs. manual data reconciliation."""

    def test_reconciliation_no_differences(self, data_connector):
        result = data_connector.reconcile_auto_manual(
            auto_data={"scope1": Decimal("5000")},
            manual_data={"scope1": Decimal("5000")},
        )
        assert result["has_differences"] is False

    def test_reconciliation_with_differences(self, data_connector):
        result = data_connector.reconcile_auto_manual(
            auto_data={"scope1": Decimal("5000")},
            manual_data={"scope1": Decimal("5500")},
        )
        assert result["has_differences"] is True
        assert result["differences"]["scope1"]["delta"] == Decimal("500")

    def test_reconciliation_report(self, data_connector):
        report = data_connector.generate_reconciliation_report(
            auto_data={"scope1": Decimal("5000"), "scope2": Decimal("3000")},
            manual_data={"scope1": Decimal("5200"), "scope2": Decimal("3000")},
        )
        assert "scope1" in report["discrepancies"]
        assert "scope2" not in report["discrepancies"]
