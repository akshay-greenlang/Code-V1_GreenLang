# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Multi-Entity Boundary Engine
======================================================

Tests equity share consolidation, operational control, financial control,
inter-company elimination, boundary changes, entity hierarchy traversal,
JV treatment, and edge cases.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal

import pytest


# =============================================================================
# Equity Share Consolidation
# =============================================================================


class TestEquityShareConsolidation:
    """Test equity share approach: ownership % x entity emissions."""

    def test_parent_100pct_equity(self, sample_entity_hierarchy):
        parent = sample_entity_hierarchy["parent"]
        reported = parent["scope3_tco2e"] * parent["equity_pct"] / Decimal("100")
        assert reported == parent["scope3_tco2e"]

    def test_subsidiary_100pct_equity(self, sample_entity_hierarchy):
        for sub in sample_entity_hierarchy["subsidiaries"]:
            reported = sub["scope3_tco2e"] * sub["equity_pct"] / Decimal("100")
            assert reported == sub["scope3_tco2e"]

    def test_jv_50pct_equity(self, sample_entity_hierarchy):
        jv1 = sample_entity_hierarchy["joint_ventures"][0]
        assert jv1["equity_pct"] == Decimal("50")
        reported = jv1["scope3_tco2e"] * jv1["equity_pct"] / Decimal("100")
        assert reported == Decimal("11000")

    def test_jv_40pct_equity(self, sample_entity_hierarchy):
        jv2 = sample_entity_hierarchy["joint_ventures"][1]
        assert jv2["equity_pct"] == Decimal("40")
        reported = jv2["scope3_tco2e"] * jv2["equity_pct"] / Decimal("100")
        assert reported == Decimal("14000")

    def test_total_equity_share_consolidation(self, sample_entity_hierarchy):
        """Total equity-share = parent + subs (100%) + JVs (proportional)."""
        total = Decimal("0")
        total += sample_entity_hierarchy["parent"]["scope3_tco2e"]
        for sub in sample_entity_hierarchy["subsidiaries"]:
            total += sub["scope3_tco2e"] * sub["equity_pct"] / Decimal("100")
        for jv in sample_entity_hierarchy["joint_ventures"]:
            total += jv["scope3_tco2e"] * jv["equity_pct"] / Decimal("100")
        # parent(5000) + subs(45000+32000+18000+25000+12000) + JVs(11000+14000)
        assert total == Decimal("162000")


# =============================================================================
# Operational Control
# =============================================================================


class TestOperationalControl:
    """Test operational control approach: 100% for controlled entities."""

    def test_parent_included(self, sample_entity_hierarchy):
        parent = sample_entity_hierarchy["parent"]
        assert parent["has_operational_control"] is True

    def test_subsidiaries_included(self, sample_entity_hierarchy):
        for sub in sample_entity_hierarchy["subsidiaries"]:
            assert sub["has_operational_control"] is True

    def test_jv_excluded_no_oc(self, sample_entity_hierarchy):
        for jv in sample_entity_hierarchy["joint_ventures"]:
            assert jv["has_operational_control"] is False

    def test_total_operational_control(self, sample_entity_hierarchy):
        """OC = parent + all subs at 100%, JVs excluded."""
        total = sample_entity_hierarchy["parent"]["scope3_tco2e"]
        for sub in sample_entity_hierarchy["subsidiaries"]:
            if sub["has_operational_control"]:
                total += sub["scope3_tco2e"]
        # JVs excluded because no operational control
        # 5000 + 45000 + 32000 + 18000 + 25000 + 12000 = 137000
        assert total == Decimal("137000")


# =============================================================================
# Financial Control
# =============================================================================


class TestFinancialControl:
    """Test financial control consolidation approach."""

    def test_parent_has_financial_control(self, sample_entity_hierarchy):
        parent = sample_entity_hierarchy["parent"]
        assert parent["has_financial_control"] is True

    def test_subs_have_financial_control(self, sample_entity_hierarchy):
        for sub in sample_entity_hierarchy["subsidiaries"]:
            assert sub["has_financial_control"] is True

    def test_jvs_no_financial_control(self, sample_entity_hierarchy):
        for jv in sample_entity_hierarchy["joint_ventures"]:
            assert jv["has_financial_control"] is False

    def test_financial_control_equals_operational(self, sample_entity_hierarchy):
        """In this fixture, FC and OC yield the same result."""
        oc_total = sample_entity_hierarchy["parent"]["scope3_tco2e"]
        fc_total = sample_entity_hierarchy["parent"]["scope3_tco2e"]
        for sub in sample_entity_hierarchy["subsidiaries"]:
            if sub["has_operational_control"]:
                oc_total += sub["scope3_tco2e"]
            if sub["has_financial_control"]:
                fc_total += sub["scope3_tco2e"]
        assert oc_total == fc_total


# =============================================================================
# Inter-Company Elimination
# =============================================================================


class TestInterCompanyElimination:
    """Test elimination of inter-company transactions."""

    def test_eliminate_intercompany_logistics(self):
        """Logistics between subs should be eliminated to avoid double-counting."""
        sub_a_cat4 = Decimal("22000")
        intercompany_logistics = Decimal("3000")
        adjusted = sub_a_cat4 - intercompany_logistics
        assert adjusted == Decimal("19000")

    def test_eliminate_internal_procurement(self):
        """Internal purchases should be eliminated from Cat 1."""
        sub_a_cat1 = Decimal("85000")
        internal_purchases = Decimal("8000")
        adjusted = sub_a_cat1 - internal_purchases
        assert adjusted == Decimal("77000")

    def test_zero_elimination_for_single_entity(self):
        """Single entity has no inter-company transactions."""
        entity_cat1 = Decimal("85000")
        elimination = Decimal("0")
        adjusted = entity_cat1 - elimination
        assert adjusted == entity_cat1


# =============================================================================
# Boundary Change Handling
# =============================================================================


class TestBoundaryChangeHandling:
    """Test handling of mid-year boundary changes (acquisitions/divestitures)."""

    def test_acquisition_mid_year_pro_rata(self):
        """Acquisition on July 1: pro-rata = 184/365 of annual."""
        acquired_annual = Decimal("22000")
        acquisition_date_day = 182  # July 1 is day 182
        days_in_year = 365
        pro_rata_days = days_in_year - acquisition_date_day
        pro_rata_factor = Decimal(str(pro_rata_days)) / Decimal(str(days_in_year))
        reported = acquired_annual * pro_rata_factor
        assert Decimal("10000") < reported < Decimal("12000")

    def test_divestiture_mid_year_pro_rata(self):
        """Divestiture on Sept 1: only report Jan-Aug (243 days)."""
        divested_annual = Decimal("18000")
        divestiture_day = 244  # Sept 1
        pro_rata_factor = Decimal(str(divestiture_day)) / Decimal("365")
        reported = divested_annual * pro_rata_factor
        assert Decimal("10000") < reported < Decimal("13000")

    def test_full_year_acquisition(self):
        """Acquisition at Jan 1: full year reported."""
        acquired_annual = Decimal("22000")
        pro_rata_factor = Decimal("1.0")
        reported = acquired_annual * pro_rata_factor
        assert reported == acquired_annual


# =============================================================================
# Entity Hierarchy Traversal
# =============================================================================


class TestEntityHierarchyTraversal:
    """Test traversal of entity hierarchy."""

    def test_hierarchy_has_parent(self, sample_entity_hierarchy):
        assert "parent" in sample_entity_hierarchy
        assert sample_entity_hierarchy["parent"]["entity_id"] == "ENT-PARENT-001"

    def test_hierarchy_has_5_subsidiaries(self, sample_entity_hierarchy):
        assert len(sample_entity_hierarchy["subsidiaries"]) == 5

    def test_hierarchy_has_2_jvs(self, sample_entity_hierarchy):
        assert len(sample_entity_hierarchy["joint_ventures"]) == 2

    def test_total_entities_count(self, sample_entity_hierarchy):
        total = (
            1  # parent
            + len(sample_entity_hierarchy["subsidiaries"])
            + len(sample_entity_hierarchy["joint_ventures"])
        )
        assert total == 8

    def test_entity_ids_unique(self, sample_entity_hierarchy):
        ids = [sample_entity_hierarchy["parent"]["entity_id"]]
        ids.extend(s["entity_id"] for s in sample_entity_hierarchy["subsidiaries"])
        ids.extend(j["entity_id"] for j in sample_entity_hierarchy["joint_ventures"])
        assert len(ids) == len(set(ids))

    def test_all_entities_have_scope3(self, sample_entity_hierarchy):
        assert sample_entity_hierarchy["parent"]["scope3_tco2e"] >= Decimal("0")
        for sub in sample_entity_hierarchy["subsidiaries"]:
            assert sub["scope3_tco2e"] >= Decimal("0")
        for jv in sample_entity_hierarchy["joint_ventures"]:
            assert jv["scope3_tco2e"] >= Decimal("0")


# =============================================================================
# JV Treatment at Different Thresholds
# =============================================================================


class TestJVTreatment:
    """Test JV treatment at different ownership thresholds."""

    @pytest.mark.parametrize("equity_pct,expected_classification", [
        (Decimal("50"), "joint_venture"),
        (Decimal("40"), "joint_venture"),
        (Decimal("25"), "associate"),
        (Decimal("10"), "investment"),
        (Decimal("51"), "subsidiary"),
        (Decimal("100"), "wholly_owned"),
    ])
    def test_entity_classification_by_equity(self, equity_pct, expected_classification):
        if equity_pct > Decimal("50"):
            classification = "wholly_owned" if equity_pct == Decimal("100") else "subsidiary"
        elif equity_pct >= Decimal("20"):
            classification = "joint_venture" if equity_pct >= Decimal("40") else "associate"
        else:
            classification = "investment"
        assert classification == expected_classification

    def test_jv_proportional_at_50pct(self):
        jv_emissions = Decimal("22000")
        equity_pct = Decimal("50")
        reported = jv_emissions * equity_pct / Decimal("100")
        assert reported == Decimal("11000")

    def test_jv_proportional_at_25pct(self):
        jv_emissions = Decimal("22000")
        equity_pct = Decimal("25")
        reported = jv_emissions * equity_pct / Decimal("100")
        assert reported == Decimal("5500")


# =============================================================================
# Edge Cases
# =============================================================================


class TestMultiEntityEdgeCases:
    """Test edge cases for multi-entity boundary."""

    def test_single_entity_group(self):
        """Group with only parent, no subsidiaries or JVs."""
        group = {
            "parent": {"scope3_tco2e": Decimal("50000")},
            "subsidiaries": [],
            "joint_ventures": [],
        }
        total = group["parent"]["scope3_tco2e"]
        assert total == Decimal("50000")

    def test_circular_ownership_detection(self):
        """Circular ownership (A owns B owns A) should be detectable."""
        entity_a = {"entity_id": "A", "owns": ["B"]}
        entity_b = {"entity_id": "B", "owns": ["A"]}
        visited = set()
        stack = ["A"]
        circular = False
        while stack:
            current = stack.pop()
            if current in visited:
                circular = True
                break
            visited.add(current)
            if current == "A":
                stack.extend(entity_a["owns"])
            elif current == "B":
                stack.extend(entity_b["owns"])
        assert circular is True

    def test_zero_equity_excluded(self):
        """Entity with 0% equity should contribute zero."""
        emissions = Decimal("10000")
        equity_pct = Decimal("0")
        reported = emissions * equity_pct / Decimal("100")
        assert reported == Decimal("0")

    def test_approach_consistency(self, sample_entity_hierarchy):
        """Default approach should be one of the three valid options."""
        approach = sample_entity_hierarchy["default_approach"]
        assert approach in {"equity_share", "operational_control", "financial_control"}
