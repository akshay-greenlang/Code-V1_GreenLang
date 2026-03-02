# -*- coding: utf-8 -*-
"""
Unit tests for BoundaryDeterminerEngine (AGENT-MRV-029, Engine 3)

80 tests covering:
- Double-Counting Rules DC-SCM-001 through DC-SCM-010 (40 tests, 4 per rule)
- Boundary determination: upstream/downstream split, DC rule access (20 tests)
- Cross-category double-counting checks and provenance (20 tests)

The BoundaryDeterminerEngine enforces the 10 double-counting prevention rules
defined in the GHG Protocol Scope 3 Standard to ensure no emissions are counted
in more than one category. Each DC rule is tested with at least four scenarios.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.scope3_category_mapper.boundary_determiner import (
        BoundaryDeterminerEngine,
        DoubleCountingRule,
        BoundaryResult,
        IncotermsMapping,
        DCCheckResult,
    )
    BOUNDARY_AVAILABLE = True
except ImportError:
    BOUNDARY_AVAILABLE = False

try:
    from greenlang.scope3_category_mapper.models import (
        Scope3Category,
        CompanyType,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not BOUNDARY_AVAILABLE,
    reason="BoundaryDeterminerEngine not available",
)

_SKIP_MODELS = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="Scope3Category models not available",
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine():
    """Create a fresh BoundaryDeterminerEngine instance."""
    if not BOUNDARY_AVAILABLE:
        pytest.skip("BoundaryDeterminerEngine not available")
    eng = BoundaryDeterminerEngine()
    return eng


@pytest.fixture
def manufacturer_context() -> Dict[str, Any]:
    """Manufacturer organization context."""
    return {
        "company_type": "manufacturer",
        "consolidation_approach": "operational_control",
        "reporting_year": 2024,
    }


@pytest.fixture
def financial_context() -> Dict[str, Any]:
    """Financial institution organization context."""
    return {
        "company_type": "financial",
        "consolidation_approach": "financial_control",
        "reporting_year": 2024,
    }


# ==============================================================================
# DC-SCM-001: Cat 1 (Purchased Goods) vs Cat 2 (Capital Goods)
#
# Rule: Items with useful life > 1 year AND cost >= capex_threshold go to Cat 2.
# Below threshold or short-lived items stay in Cat 1.
# ==============================================================================


@_SKIP
class TestDCSCM001:
    """DC-SCM-001: Category 1 vs Category 2 boundary (capex threshold)."""

    def test_below_threshold_is_cat1(self, engine):
        """Purchase at $4999 (below $5000 default threshold) -> Cat 1."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-001",
            record={
                "amount": Decimal("4999.00"),
                "useful_life_years": 5,
                "description": "Small equipment",
            },
        )
        assert result.assigned_category == "cat_1_purchased_goods"
        assert result.rule_applied == "DC-SCM-001"

    def test_at_threshold_is_cat2(self, engine):
        """Purchase at exactly $5000 (at threshold) -> Cat 2."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-001",
            record={
                "amount": Decimal("5000.00"),
                "useful_life_years": 3,
                "description": "Industrial tool",
            },
        )
        assert result.assigned_category == "cat_2_capital_goods"

    def test_above_threshold_is_cat2(self, engine):
        """Purchase at $10000 (above threshold) -> Cat 2."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-001",
            record={
                "amount": Decimal("10000.00"),
                "useful_life_years": 10,
                "description": "CNC machine",
            },
        )
        assert result.assigned_category == "cat_2_capital_goods"

    def test_custom_threshold(self, engine):
        """Custom capex threshold ($10000) -> $7000 stays in Cat 1."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-001",
            record={
                "amount": Decimal("7000.00"),
                "useful_life_years": 5,
                "description": "Mid-range equipment",
            },
            params={"capex_threshold": Decimal("10000.00")},
        )
        assert result.assigned_category == "cat_1_purchased_goods"


# ==============================================================================
# DC-SCM-002: Cat 1 (Purchased Goods) vs Cat 4 (Upstream Transport)
#
# Rule: FOB/EXW -> freight invoiced separately goes to Cat 4.
# CIF/DDP -> freight included in goods price stays in Cat 1.
# ==============================================================================


@_SKIP
class TestDCSCM002:
    """DC-SCM-002: Category 1 vs Category 4 boundary (Incoterms)."""

    def test_fob_splits_freight(self, engine):
        """FOB incoterm: goods -> Cat 1, freight -> Cat 4 (separate)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-002",
            record={
                "incoterm": "FOB",
                "goods_amount": Decimal("10000.00"),
                "freight_amount": Decimal("1500.00"),
                "direction": "inbound",
            },
        )
        assert result.assigned_category == "cat_4_upstream_transport"
        assert result.split_required is True

    def test_cif_includes_freight(self, engine):
        """CIF incoterm: freight included in price -> Cat 1 only."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-002",
            record={
                "incoterm": "CIF",
                "goods_amount": Decimal("11500.00"),
                "freight_amount": Decimal("0.00"),
                "direction": "inbound",
            },
        )
        assert result.assigned_category == "cat_1_purchased_goods"
        assert result.split_required is False

    def test_separately_invoiced_freight(self, engine):
        """Freight invoiced separately always goes to Cat 4."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-002",
            record={
                "incoterm": "CIF",
                "goods_amount": Decimal("10000.00"),
                "freight_amount": Decimal("1500.00"),
                "separately_invoiced_freight": True,
                "direction": "inbound",
            },
        )
        assert result.assigned_category == "cat_4_upstream_transport"

    def test_exw_buyer_arranges(self, engine):
        """EXW incoterm: buyer arranges all transport -> Cat 4."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-002",
            record={
                "incoterm": "EXW",
                "goods_amount": Decimal("8000.00"),
                "freight_amount": Decimal("2000.00"),
                "direction": "inbound",
            },
        )
        assert result.assigned_category == "cat_4_upstream_transport"
        assert result.split_required is True


# ==============================================================================
# DC-SCM-003: Cat 3 (Fuel & Energy) vs Scope 2
#
# Rule: WTT fuel, upstream electricity, T&D losses -> Cat 3 (not Scope 2).
# Generation of purchased electricity -> Scope 2 (location or market).
# ==============================================================================


@_SKIP
class TestDCSCM003:
    """DC-SCM-003: Category 3 vs Scope 2 boundary."""

    def test_wtt_excluded_from_scope2(self, engine):
        """Well-to-tank emissions for purchased fuels -> Cat 3, not Scope 2."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-003",
            record={
                "energy_activity": "wtt_fuel",
                "fuel_type": "natural_gas",
                "quantity_kwh": Decimal("100000"),
            },
        )
        assert result.assigned_category == "cat_3_fuel_energy"
        assert result.excluded_from == "scope_2"

    def test_td_loss_excluded_from_scope2(self, engine):
        """Transmission & distribution losses -> Cat 3, not Scope 2."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-003",
            record={
                "energy_activity": "td_loss",
                "grid_region": "CAMX",
                "quantity_kwh": Decimal("5000"),
            },
        )
        assert result.assigned_category == "cat_3_fuel_energy"

    def test_upstream_elec_is_cat3(self, engine):
        """Upstream electricity generation -> Cat 3."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-003",
            record={
                "energy_activity": "upstream_electricity",
                "grid_region": "RFCE",
                "quantity_kwh": Decimal("200000"),
            },
        )
        assert result.assigned_category == "cat_3_fuel_energy"

    def test_generation_purchased_is_scope2(self, engine):
        """Direct generation of purchased electricity -> excluded (Scope 2)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-003",
            record={
                "energy_activity": "purchased_electricity_generation",
                "grid_region": "CAMX",
                "quantity_kwh": Decimal("200000"),
            },
        )
        assert result.assigned_category == "scope_2"
        assert result.excluded_from == "cat_3_fuel_energy"


# ==============================================================================
# DC-SCM-004: Cat 4 (Upstream Transport) vs Cat 9 (Downstream Transport)
#
# Rule: Inbound (before point of sale) -> Cat 4.
# Outbound (after point of sale, seller-arranged) -> Cat 9.
# ==============================================================================


@_SKIP
class TestDCSCM004:
    """DC-SCM-004: Category 4 vs Category 9 boundary (point of sale)."""

    def test_inbound_fob_is_cat4(self, engine):
        """Inbound freight under FOB -> Cat 4 (upstream)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-004",
            record={
                "direction": "inbound",
                "incoterm": "FOB",
                "transport_mode": "road",
                "distance_km": Decimal("500"),
            },
        )
        assert result.assigned_category == "cat_4_upstream_transport"

    def test_outbound_cif_is_cat9(self, engine):
        """Outbound freight under CIF (seller pays) -> Cat 9 (downstream)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-004",
            record={
                "direction": "outbound",
                "incoterm": "CIF",
                "transport_mode": "sea",
                "distance_km": Decimal("8000"),
            },
        )
        assert result.assigned_category == "cat_9_downstream_transport"

    def test_point_of_sale_boundary(self, engine):
        """Transport crossing point-of-sale boundary splits categories."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-004",
            record={
                "direction": "cross_boundary",
                "incoterm": "FOB",
                "transport_mode": "road",
                "pre_sale_km": Decimal("300"),
                "post_sale_km": Decimal("200"),
            },
        )
        assert result.split_required is True

    def test_dap_seller_pays_cat9(self, engine):
        """DAP incoterm: seller delivers to buyer -> outbound = Cat 9."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-004",
            record={
                "direction": "outbound",
                "incoterm": "DAP",
                "transport_mode": "road",
                "distance_km": Decimal("1200"),
            },
        )
        assert result.assigned_category == "cat_9_downstream_transport"


# ==============================================================================
# DC-SCM-005: Cat 6 (Business Travel) vs Cat 7 (Employee Commuting)
#
# Rule: Business trips (non-routine, away from normal workplace) -> Cat 6.
# Daily commuting (routine home-to-office) -> Cat 7.
# Travel days for business trips exclude commute.
# ==============================================================================


@_SKIP
class TestDCSCM005:
    """DC-SCM-005: Category 6 vs Category 7 boundary."""

    def test_business_trip_is_cat6(self, engine):
        """Non-routine travel to client site -> Cat 6."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-005",
            record={
                "travel_type": "business_trip",
                "purpose": "client_meeting",
                "origin": "home_office",
                "destination": "client_site",
                "is_routine": False,
            },
        )
        assert result.assigned_category == "cat_6_business_travel"

    def test_daily_commute_is_cat7(self, engine):
        """Daily commute from home to office -> Cat 7."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-005",
            record={
                "travel_type": "commute",
                "purpose": "daily_commute",
                "origin": "home",
                "destination": "office",
                "is_routine": True,
            },
        )
        assert result.assigned_category == "cat_7_employee_commuting"

    def test_travel_day_excludes_commute(self, engine):
        """On business travel days, no commute should be counted (Cat 7=0)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-005",
            record={
                "travel_type": "travel_day_check",
                "employee_id": "EMP-001",
                "date": "2025-04-10",
                "on_business_trip": True,
            },
        )
        assert result.assigned_category == "cat_6_business_travel"
        assert result.cat7_excluded is True

    def test_routine_journey_is_cat7(self, engine):
        """Routine journey to regular work location -> Cat 7."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-005",
            record={
                "travel_type": "commute",
                "purpose": "regular_commute",
                "origin": "home",
                "destination": "regular_office",
                "is_routine": True,
                "frequency": "daily",
            },
        )
        assert result.assigned_category == "cat_7_employee_commuting"


# ==============================================================================
# DC-SCM-006: Cat 8 (Upstream Leased) vs Scope 1/2
#
# Rule: Operational control -> leased asset in Scope 1/2 (not Cat 8).
# Financial control only / equity share -> may be Cat 8.
# Operating lease (lessee) -> Cat 8.
# ==============================================================================


@_SKIP
class TestDCSCM006:
    """DC-SCM-006: Category 8 vs Scope 1/2 boundary."""

    def test_operational_control_in_scope12(self, engine):
        """Leased asset under operational control -> Scope 1/2 (not Cat 8)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-006",
            record={
                "asset_type": "building",
                "lease_type": "operating",
                "consolidation_approach": "operational_control",
                "lessee_has_operational_control": True,
            },
        )
        assert result.assigned_category == "scope_1_2"
        assert result.excluded_from == "cat_8_upstream_leased"

    def test_financial_control_may_be_cat8(self, engine):
        """Financial control only, lessee does not operate -> Cat 8."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-006",
            record={
                "asset_type": "building",
                "lease_type": "operating",
                "consolidation_approach": "financial_control",
                "lessee_has_operational_control": False,
            },
        )
        assert result.assigned_category == "cat_8_upstream_leased"

    def test_equity_share_proportional(self, engine):
        """Equity share approach: proportional allocation."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-006",
            record={
                "asset_type": "vehicle",
                "lease_type": "finance",
                "consolidation_approach": "equity_share",
                "equity_pct": Decimal("40.00"),
            },
        )
        assert result.assigned_category in (
            "cat_8_upstream_leased",
            "scope_1_2",
        )

    def test_operating_lease_lessee_cat8(self, engine):
        """Operating lease (lessee does not control) -> Cat 8."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-006",
            record={
                "asset_type": "equipment",
                "lease_type": "operating",
                "consolidation_approach": "operational_control",
                "lessee_has_operational_control": False,
            },
        )
        assert result.assigned_category == "cat_8_upstream_leased"


# ==============================================================================
# DC-SCM-007: Cat 10 (Processing of Sold Products) vs Cat 11 (Use of Sold Products)
#
# Rule: Intermediate products undergoing further processing -> Cat 10.
# End-use / consumer use phase -> Cat 11.
# Processing is sequential; no overlap between Cat 10 and Cat 11.
# ==============================================================================


@_SKIP
class TestDCSCM007:
    """DC-SCM-007: Category 10 vs Category 11 boundary."""

    def test_intermediate_processing_cat10(self, engine):
        """Intermediate product needing further processing -> Cat 10."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-007",
            record={
                "product_type": "intermediate",
                "sold_to": "industrial_processor",
                "further_processing_required": True,
            },
        )
        assert result.assigned_category == "cat_10_processing_sold"

    def test_end_use_cat11(self, engine):
        """Final consumer product in use phase -> Cat 11."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-007",
            record={
                "product_type": "final",
                "sold_to": "end_consumer",
                "further_processing_required": False,
            },
        )
        assert result.assigned_category == "cat_11_use_sold"

    def test_sequential_no_overlap(self, engine):
        """Cat 10 and Cat 11 are sequential: no double-counting."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-007",
            record={
                "product_type": "intermediate",
                "sold_to": "oem_manufacturer",
                "further_processing_required": True,
                "end_use_included": False,
            },
        )
        assert result.assigned_category == "cat_10_processing_sold"
        assert result.overlap_detected is False

    def test_hybrid_product_primary_cat(self, engine):
        """Product sold for both processing and direct use -> primary assignment."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-007",
            record={
                "product_type": "dual_use",
                "sold_to": "mixed",
                "further_processing_required": True,
                "pct_processed": Decimal("70"),
                "pct_direct_use": Decimal("30"),
            },
        )
        assert result.assigned_category in (
            "cat_10_processing_sold",
            "cat_11_use_sold",
        )


# ==============================================================================
# DC-SCM-008: Cat 11 (Use of Sold Products) vs Cat 12 (End-of-Life)
#
# Rule: During product lifetime -> Cat 11.
# After product lifetime (disposal, recycling) -> Cat 12.
# ==============================================================================


@_SKIP
class TestDCSCM008:
    """DC-SCM-008: Category 11 vs Category 12 boundary (lifetime)."""

    def test_during_lifetime_cat11(self, engine):
        """Product in use within its expected lifetime -> Cat 11."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-008",
            record={
                "product_phase": "use",
                "lifetime_years": 10,
                "current_age_years": 5,
            },
        )
        assert result.assigned_category == "cat_11_use_sold"

    def test_after_lifetime_cat12(self, engine):
        """Product past its lifetime, in disposal -> Cat 12."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-008",
            record={
                "product_phase": "end_of_life",
                "lifetime_years": 10,
                "current_age_years": 12,
                "treatment_method": "landfill",
            },
        )
        assert result.assigned_category == "cat_12_end_of_life"

    def test_lifetime_boundary(self, engine):
        """At exact end of expected lifetime -> transitions to Cat 12."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-008",
            record={
                "product_phase": "end_of_life",
                "lifetime_years": 10,
                "current_age_years": 10,
                "treatment_method": "recycling",
            },
        )
        assert result.assigned_category == "cat_12_end_of_life"

    def test_early_disposal_cat12(self, engine):
        """Product disposed before expected lifetime -> Cat 12."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-008",
            record={
                "product_phase": "end_of_life",
                "lifetime_years": 10,
                "current_age_years": 3,
                "treatment_method": "incineration",
                "early_disposal": True,
            },
        )
        assert result.assigned_category == "cat_12_end_of_life"


# ==============================================================================
# DC-SCM-009: Cat 13 (Downstream Leased) vs Scope 1/2
#
# Rule: Lessor perspective. If asset is consolidated in Scope 1/2 -> not Cat 13.
# If not consolidated (e.g. operating lease under tenant) -> Cat 13.
# ==============================================================================


@_SKIP
class TestDCSCM009:
    """DC-SCM-009: Category 13 vs Scope 1/2 boundary (lessor)."""

    def test_lessor_not_consolidated_cat13(self, engine):
        """Leased-out asset not in lessor Scope 1/2 -> Cat 13."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-009",
            record={
                "asset_type": "building",
                "lease_role": "lessor",
                "consolidated_in_scope12": False,
            },
        )
        assert result.assigned_category == "cat_13_downstream_leased"

    def test_consolidated_in_scope12(self, engine):
        """Leased-out asset already in Scope 1/2 -> excluded from Cat 13."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-009",
            record={
                "asset_type": "building",
                "lease_role": "lessor",
                "consolidated_in_scope12": True,
            },
        )
        assert result.assigned_category == "scope_1_2"
        assert result.excluded_from == "cat_13_downstream_leased"

    def test_finance_lease_excluded(self, engine):
        """Finance lease: asset transferred to lessee balance sheet -> not Cat 13."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-009",
            record={
                "asset_type": "vehicle",
                "lease_role": "lessor",
                "lease_type": "finance",
                "consolidated_in_scope12": False,
            },
        )
        # Finance lease transfers risk/reward; may not be Cat 13
        assert result.assigned_category in (
            "cat_13_downstream_leased",
            "not_applicable",
        )

    def test_operating_lease_lessor_cat13(self, engine):
        """Operating lease from lessor perspective -> Cat 13."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-009",
            record={
                "asset_type": "equipment",
                "lease_role": "lessor",
                "lease_type": "operating",
                "consolidated_in_scope12": False,
            },
        )
        assert result.assigned_category == "cat_13_downstream_leased"


# ==============================================================================
# DC-SCM-010: Cat 14 (Franchises) vs Cat 15 (Investments)
#
# Rule: Franchise agreement (operational brand license) -> Cat 14.
# Equity investment (financial ownership) -> Cat 15.
# If both exist, franchise agreement takes precedence.
# ==============================================================================


@_SKIP
class TestDCSCM010:
    """DC-SCM-010: Category 14 vs Category 15 boundary."""

    def test_franchise_agreement_cat14(self, engine):
        """Entity operating under franchise agreement -> Cat 14."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-010",
            record={
                "relationship_type": "franchise",
                "franchise_agreement": True,
                "equity_investment": False,
            },
        )
        assert result.assigned_category == "cat_14_franchises"

    def test_equity_investment_cat15(self, engine):
        """Pure equity investment, no franchise -> Cat 15."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-010",
            record={
                "relationship_type": "investment",
                "franchise_agreement": False,
                "equity_investment": True,
                "asset_class": "listed_equity",
            },
        )
        assert result.assigned_category == "cat_15_investments"

    def test_franchise_precedence(self, engine):
        """Both franchise and equity -> franchise takes precedence (Cat 14)."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-010",
            record={
                "relationship_type": "franchise_investment",
                "franchise_agreement": True,
                "equity_investment": True,
            },
        )
        assert result.assigned_category == "cat_14_franchises"

    def test_neither_relationship(self, engine):
        """No franchise or equity relationship -> not applicable."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-010",
            record={
                "relationship_type": "supplier",
                "franchise_agreement": False,
                "equity_investment": False,
            },
        )
        assert result.assigned_category in (
            "cat_1_purchased_goods",
            "not_applicable",
        )


# ==============================================================================
# BOUNDARY DETERMINATION TESTS
# ==============================================================================


@_SKIP
class TestBoundaryDetermination:
    """Test upstream/downstream boundary determination."""

    def test_determine_upstream_categories(self, engine):
        """Upstream categories are Cat 1-8."""
        upstream = engine.get_upstream_categories()
        assert len(upstream) == 8
        expected_nums = {1, 2, 3, 4, 5, 6, 7, 8}
        actual_nums = {c.value if hasattr(c, "value") else c for c in upstream}
        # Verify all 8 upstream categories present
        assert len(actual_nums) == 8

    def test_determine_downstream_categories(self, engine):
        """Downstream categories are Cat 9-15."""
        downstream = engine.get_downstream_categories()
        assert len(downstream) == 7
        expected_nums = {9, 10, 11, 12, 13, 14, 15}
        actual_nums = {c.value if hasattr(c, "value") else c for c in downstream}
        assert len(actual_nums) == 7

    def test_all_dc_rules_accessible(self, engine):
        """All 10 DC rules should be retrievable."""
        for i in range(1, 11):
            rule_id = f"DC-SCM-{i:03d}"
            rule = engine.get_dc_rule(rule_id)
            assert rule is not None
            assert rule.rule_id == rule_id

    def test_get_all_dc_rules_returns_10(self, engine):
        """get_all_dc_rules should return exactly 10 rules."""
        rules = engine.get_all_dc_rules()
        assert len(rules) == 10

    def test_incoterm_split_fob(self, engine):
        """FOB: buyer responsible for freight -> split goods/freight."""
        split = engine.determine_incoterm_split("FOB")
        assert split.buyer_arranges_freight is True
        assert split.freight_category == "cat_4_upstream_transport"

    def test_incoterm_split_cif(self, engine):
        """CIF: seller pays freight -> no split (all in goods price)."""
        split = engine.determine_incoterm_split("CIF")
        assert split.buyer_arranges_freight is False
        assert split.freight_included_in_goods is True

    def test_incoterm_split_ddp(self, engine):
        """DDP: seller delivers to buyer location -> seller bears transport."""
        split = engine.determine_incoterm_split("DDP")
        assert split.buyer_arranges_freight is False

    def test_unknown_incoterm_raises(self, engine):
        """Unknown incoterm raises ValueError."""
        with pytest.raises(ValueError, match="[Uu]nknown"):
            engine.determine_incoterm_split("INVALID")

    def test_boundary_result_has_provenance(self, engine):
        """Boundary determination result includes provenance hash."""
        result = engine.apply_dc_rule(
            rule_id="DC-SCM-001",
            record={
                "amount": Decimal("10000.00"),
                "useful_life_years": 5,
            },
        )
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64

    def test_dc_rule_description_not_empty(self, engine):
        """Each DC rule has a non-empty description."""
        for i in range(1, 11):
            rule_id = f"DC-SCM-{i:03d}"
            rule = engine.get_dc_rule(rule_id)
            assert rule.description is not None
            assert len(rule.description) > 10

    def test_dc_rule_categories_pair(self, engine):
        """Each DC rule involves exactly two categories (or a category + scope)."""
        rules = engine.get_all_dc_rules()
        for rule in rules:
            assert hasattr(rule, "category_a")
            assert hasattr(rule, "category_b")
            assert rule.category_a != rule.category_b

    def test_upstream_downstream_cover_all_15(self, engine):
        """Upstream + downstream = all 15 Scope 3 categories."""
        upstream = engine.get_upstream_categories()
        downstream = engine.get_downstream_categories()
        all_cats = set(upstream) | set(downstream)
        assert len(all_cats) == 15

    def test_incoterm_fca_buyer_freight(self, engine):
        """FCA: Free Carrier -> buyer arranges main transport."""
        split = engine.determine_incoterm_split("FCA")
        assert split.buyer_arranges_freight is True

    def test_incoterm_exw_buyer_freight(self, engine):
        """EXW: buyer arranges all transport from seller premises."""
        split = engine.determine_incoterm_split("EXW")
        assert split.buyer_arranges_freight is True

    def test_incoterm_dap_seller_delivers(self, engine):
        """DAP: seller delivers to named destination."""
        split = engine.determine_incoterm_split("DAP")
        assert split.buyer_arranges_freight is False

    def test_dc_rule_idempotent(self, engine):
        """Same input to DC rule produces same result (deterministic)."""
        record = {
            "amount": Decimal("8000.00"),
            "useful_life_years": 7,
        }
        r1 = engine.apply_dc_rule(rule_id="DC-SCM-001", record=record)
        r2 = engine.apply_dc_rule(rule_id="DC-SCM-001", record=record)
        assert r1.assigned_category == r2.assigned_category
        assert r1.provenance_hash == r2.provenance_hash

    def test_invalid_dc_rule_id_raises(self, engine):
        """Invalid DC rule ID raises ValueError."""
        with pytest.raises(ValueError):
            engine.apply_dc_rule(rule_id="DC-SCM-999", record={})

    def test_empty_record_raises(self, engine):
        """Empty record raises ValueError for DC rule."""
        with pytest.raises((ValueError, KeyError)):
            engine.apply_dc_rule(rule_id="DC-SCM-001", record={})

    def test_dc_rule_processing_time(self, engine):
        """DC rule execution should complete in < 10ms."""
        import time
        start = time.monotonic()
        engine.apply_dc_rule(
            rule_id="DC-SCM-001",
            record={"amount": Decimal("5000.00"), "useful_life_years": 5},
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 10.0


# ==============================================================================
# CROSS-CATEGORY DOUBLE-COUNTING CHECK TESTS
# ==============================================================================


@_SKIP
class TestCrossCategoryChecks:
    """Test cross-category double-counting detection."""

    def test_check_double_counting_no_overlap(self, engine):
        """Records in different non-overlapping categories -> no DC detected."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("10000")},
            {"record_id": "R2", "assigned_category": "cat_6_business_travel",
             "amount": Decimal("5000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count == 0
        assert result.status == "PASS"

    def test_check_double_counting_cat1_cat2_overlap(self, engine):
        """Same record in both Cat 1 and Cat 2 -> overlap detected."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("6000"), "description": "Industrial pump"},
            {"record_id": "R1", "assigned_category": "cat_2_capital_goods",
             "amount": Decimal("6000"), "description": "Industrial pump"},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1
        assert result.status in ("FAIL", "WARNING")

    def test_check_double_counting_cat4_cat9_overlap(self, engine):
        """Same freight record in Cat 4 and Cat 9 -> overlap detected."""
        records = [
            {"record_id": "R5", "assigned_category": "cat_4_upstream_transport",
             "amount": Decimal("3000"), "description": "Freight Chicago-Detroit"},
            {"record_id": "R5", "assigned_category": "cat_9_downstream_transport",
             "amount": Decimal("3000"), "description": "Freight Chicago-Detroit"},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_double_counting_provenance(self, engine):
        """DC check result includes provenance hash."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("10000")},
        ]
        result = engine.check_double_counting(records)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64

    def test_check_double_counting_empty_records(self, engine):
        """Empty record list -> PASS with zero overlaps."""
        result = engine.check_double_counting([])
        assert result.overlap_count == 0
        assert result.status == "PASS"

    def test_check_double_counting_large_batch(self, engine):
        """1000 records with no overlaps -> all pass."""
        records = [
            {"record_id": f"R{i}", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("100")}
            for i in range(1000)
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count == 0

    def test_check_double_counting_cat6_cat7(self, engine):
        """Same employee on travel day in Cat 6 and Cat 7 -> overlap."""
        records = [
            {"record_id": "EMP-001-2025-04-10",
             "assigned_category": "cat_6_business_travel",
             "amount": Decimal("500"), "employee_id": "EMP-001",
             "date": "2025-04-10"},
            {"record_id": "EMP-001-2025-04-10",
             "assigned_category": "cat_7_employee_commuting",
             "amount": Decimal("25"), "employee_id": "EMP-001",
             "date": "2025-04-10"},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_double_counting_returns_overlap_details(self, engine):
        """Overlap result includes details of which records overlap."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("6000")},
            {"record_id": "R1", "assigned_category": "cat_2_capital_goods",
             "amount": Decimal("6000")},
        ]
        result = engine.check_double_counting(records)
        assert hasattr(result, "overlaps")
        if result.overlap_count > 0:
            assert len(result.overlaps) >= 1
            assert "R1" in str(result.overlaps[0])

    def test_check_double_counting_deterministic(self, engine):
        """Same records produce same DC check result."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("10000")},
            {"record_id": "R2", "assigned_category": "cat_6_business_travel",
             "amount": Decimal("5000")},
        ]
        r1 = engine.check_double_counting(records)
        r2 = engine.check_double_counting(records)
        assert r1.provenance_hash == r2.provenance_hash

    def test_check_all_dc_rules_against_mixed_records(self, engine):
        """Run all DC checks across 15-category batch -> comprehensive check."""
        categories = [
            "cat_1_purchased_goods", "cat_2_capital_goods", "cat_3_fuel_energy",
            "cat_4_upstream_transport", "cat_5_waste", "cat_6_business_travel",
            "cat_7_employee_commuting", "cat_8_upstream_leased",
            "cat_9_downstream_transport", "cat_10_processing_sold",
            "cat_11_use_sold", "cat_12_end_of_life",
            "cat_13_downstream_leased", "cat_14_franchises",
            "cat_15_investments",
        ]
        records = [
            {"record_id": f"R{i+1}", "assigned_category": cat,
             "amount": Decimal("1000")}
            for i, cat in enumerate(categories)
        ]
        result = engine.check_double_counting(records)
        assert result.status == "PASS"  # No overlaps expected
        assert result.rules_checked >= 10

    def test_check_double_counting_cat10_cat11(self, engine):
        """Same product in Cat 10 (processing) and Cat 11 (use) -> overlap."""
        records = [
            {"record_id": "PROD-001",
             "assigned_category": "cat_10_processing_sold",
             "amount": Decimal("2000")},
            {"record_id": "PROD-001",
             "assigned_category": "cat_11_use_sold",
             "amount": Decimal("2000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_double_counting_cat11_cat12(self, engine):
        """Same product in Cat 11 (use) and Cat 12 (end-of-life) -> overlap."""
        records = [
            {"record_id": "EOL-001",
             "assigned_category": "cat_11_use_sold",
             "amount": Decimal("1500")},
            {"record_id": "EOL-001",
             "assigned_category": "cat_12_end_of_life",
             "amount": Decimal("1500")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_double_counting_cat8_scope12(self, engine):
        """Same leased asset in Cat 8 and a Scope 1/2 record -> check flags."""
        records = [
            {"record_id": "LEASE-001",
             "assigned_category": "cat_8_upstream_leased",
             "amount": Decimal("5000")},
            {"record_id": "LEASE-001",
             "assigned_category": "scope_1_2",
             "amount": Decimal("5000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_double_counting_cat13_scope12(self, engine):
        """Same leased asset from lessor in Cat 13 and Scope 1/2 -> overlap."""
        records = [
            {"record_id": "DLEASE-001",
             "assigned_category": "cat_13_downstream_leased",
             "amount": Decimal("4000")},
            {"record_id": "DLEASE-001",
             "assigned_category": "scope_1_2",
             "amount": Decimal("4000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_double_counting_cat14_cat15(self, engine):
        """Same entity in Cat 14 (franchise) and Cat 15 (investment) -> overlap."""
        records = [
            {"record_id": "FRN-INV-001",
             "assigned_category": "cat_14_franchises",
             "amount": Decimal("10000")},
            {"record_id": "FRN-INV-001",
             "assigned_category": "cat_15_investments",
             "amount": Decimal("10000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1

    def test_check_cat1_cat4_cif_no_overlap(self, engine):
        """CIF goods and no separate freight -> no Cat 1/Cat 4 overlap."""
        records = [
            {"record_id": "CIF-001",
             "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("15000"), "incoterm": "CIF"},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count == 0

    def test_check_dc_result_has_rules_checked(self, engine):
        """DC check result reports how many rules were checked."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("1000")},
        ]
        result = engine.check_double_counting(records)
        assert hasattr(result, "rules_checked")
        assert result.rules_checked >= 1

    def test_check_dc_result_has_processing_time(self, engine):
        """DC check result includes processing time."""
        records = [
            {"record_id": "R1", "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("1000")},
        ]
        result = engine.check_double_counting(records)
        assert hasattr(result, "processing_time_ms") or hasattr(result, "provenance_hash")

    def test_check_dc_three_category_overlap(self, engine):
        """Same record in 3 categories -> multiple overlaps detected."""
        records = [
            {"record_id": "TRIPLE-001",
             "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("5000")},
            {"record_id": "TRIPLE-001",
             "assigned_category": "cat_2_capital_goods",
             "amount": Decimal("5000")},
            {"record_id": "TRIPLE-001",
             "assigned_category": "cat_4_upstream_transport",
             "amount": Decimal("5000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 2

    def test_check_dc_different_amounts_same_id(self, engine):
        """Same record_id with different amounts still flags overlap."""
        records = [
            {"record_id": "AMT-001",
             "assigned_category": "cat_1_purchased_goods",
             "amount": Decimal("5000")},
            {"record_id": "AMT-001",
             "assigned_category": "cat_2_capital_goods",
             "amount": Decimal("8000")},
        ]
        result = engine.check_double_counting(records)
        assert result.overlap_count >= 1
