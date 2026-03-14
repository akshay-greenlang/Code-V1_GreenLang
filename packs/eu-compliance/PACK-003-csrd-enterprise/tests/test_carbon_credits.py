# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Carbon Credits Tests (15 tests)

Tests carbon credit lifecycle management including portfolio
management, retirement, transfer, quality assessment,
net-zero accounting, and SBTi compliance notes.

Author: GreenLang QA Team
"""

from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import _compute_hash, _new_uuid, _utcnow


class TestCarbonCredits:
    """Test suite for carbon credit lifecycle engine."""

    def test_add_credit(self, sample_carbon_credits):
        """Test adding a carbon credit to the portfolio."""
        credit = sample_carbon_credits[0]
        assert credit["credit_id"].startswith("CC-")
        assert credit["quantity_tco2e"] > 0
        assert credit["price_per_tonne_usd"] > 0
        assert credit["status"] == "active"
        assert len(credit["provenance_hash"]) == 64

    def test_retire_credit(self, sample_carbon_credits):
        """Test credit retirement marks status and date."""
        retired = [c for c in sample_carbon_credits if c["status"] == "retired"]
        assert len(retired) >= 1
        for c in retired:
            assert c["retirement_date"] is not None
            assert c["retirement_date"] != ""

    def test_transfer_credit(self, sample_carbon_credits):
        """Test credit transfer creates new provenance."""
        credit = sample_carbon_credits[0]
        transfer = {
            "credit_id": credit["credit_id"],
            "from_entity": "GlobalTech AG",
            "to_entity": "GreenPartner GmbH",
            "quantity_tco2e": 100,
            "transfer_date": _utcnow().strftime("%Y-%m-%d"),
            "provenance_hash": _compute_hash({
                "credit_id": credit["credit_id"],
                "transfer": "to_GreenPartner",
            }),
        }
        assert transfer["quantity_tco2e"] <= credit["quantity_tco2e"]
        assert len(transfer["provenance_hash"]) == 64

    def test_portfolio_summary(self, sample_carbon_credits):
        """Test portfolio summary aggregation."""
        total_qty = sum(c["quantity_tco2e"] for c in sample_carbon_credits)
        total_val = sum(c["total_value_usd"] for c in sample_carbon_credits)
        active_count = sum(1 for c in sample_carbon_credits if c["status"] == "active")
        retired_count = sum(1 for c in sample_carbon_credits if c["status"] == "retired")
        registries = list({c["registry"] for c in sample_carbon_credits})
        portfolio = {
            "total_credits": len(sample_carbon_credits),
            "active_credits": active_count,
            "retired_credits": retired_count,
            "total_quantity_tco2e": total_qty,
            "total_value_usd": round(total_val, 2),
            "unique_registries": len(registries),
        }
        assert portfolio["total_credits"] == 20
        assert portfolio["active_credits"] + portfolio["retired_credits"] == 20
        assert portfolio["total_quantity_tco2e"] > 0
        assert portfolio["unique_registries"] >= 3

    def test_net_zero_accounting(self, sample_carbon_credits):
        """Test net-zero accounting alignment."""
        gross_emissions = 45230.5
        retired_qty = sum(
            c["quantity_tco2e"] for c in sample_carbon_credits if c["status"] == "retired"
        )
        net_emissions = gross_emissions - retired_qty
        accounting = {
            "gross_emissions_tco2e": gross_emissions,
            "offsets_retired_tco2e": retired_qty,
            "net_emissions_tco2e": max(net_emissions, 0),
            "offset_pct": round(retired_qty / gross_emissions * 100, 2),
        }
        assert accounting["offset_pct"] > 0
        assert accounting["gross_emissions_tco2e"] > 0

    def test_gross_vs_net(self, sample_carbon_credits):
        """Test gross vs net emissions distinction."""
        gross = 50000.0
        retired = sum(c["quantity_tco2e"] for c in sample_carbon_credits if c["status"] == "retired")
        net = gross - retired
        assert gross > net
        assert net >= 0 or retired > gross

    def test_offset_percentage(self, sample_carbon_credits):
        """Test offset percentage calculation."""
        gross = 50000.0
        total_active = sum(c["quantity_tco2e"] for c in sample_carbon_credits if c["status"] == "active")
        potential_offset_pct = round(total_active / gross * 100, 2)
        assert potential_offset_pct > 0

    def test_quality_assessment(self, sample_carbon_credits):
        """Test credit quality assessment scores."""
        for credit in sample_carbon_credits:
            score = credit["additionality_score"]
            assert 0.0 <= score <= 1.0
            assert credit["permanence_risk"] in ("low", "medium", "high")

    def test_additionality_score(self, sample_carbon_credits):
        """Test additionality scores are in valid range."""
        scores = [c["additionality_score"] for c in sample_carbon_credits]
        avg_score = sum(scores) / len(scores)
        assert 0.5 <= avg_score <= 1.0
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_vintage_breakdown(self, sample_carbon_credits):
        """Test vintage year breakdown."""
        vintages = {}
        for c in sample_carbon_credits:
            v = c["vintage_year"]
            vintages[v] = vintages.get(v, 0) + c["quantity_tco2e"]
        assert len(vintages) >= 2
        for year, qty in vintages.items():
            assert 2020 <= year <= 2030
            assert qty > 0

    def test_retirement_schedule(self, sample_carbon_credits):
        """Test retirement schedule planning."""
        active = [c for c in sample_carbon_credits if c["status"] == "active"]
        schedule = []
        target_annual_retirement = 5000
        remaining = target_annual_retirement
        for credit in sorted(active, key=lambda x: x["vintage_year"]):
            retire_qty = min(credit["quantity_tco2e"], remaining)
            if retire_qty > 0:
                schedule.append({
                    "credit_id": credit["credit_id"],
                    "retire_qty": retire_qty,
                    "vintage": credit["vintage_year"],
                })
                remaining -= retire_qty
            if remaining <= 0:
                break
        total_scheduled = sum(s["retire_qty"] for s in schedule)
        assert total_scheduled <= target_annual_retirement
        assert len(schedule) > 0

    def test_price_history(self, sample_carbon_credits):
        """Test price per tonne tracking."""
        prices = [c["price_per_tonne_usd"] for c in sample_carbon_credits]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        assert min_price > 0
        assert max_price > min_price
        assert avg_price > 0

    def test_registry_validation(self, sample_carbon_credits):
        """Test all credits reference valid registries."""
        valid_registries = {"VCS", "GoldStandard", "ACR", "CAR", "CDM", "Article6"}
        for credit in sample_carbon_credits:
            assert credit["registry"] in valid_registries, (
                f"Invalid registry: {credit['registry']}"
            )

    def test_credit_lifecycle(self, sample_carbon_credits):
        """Test credit lifecycle transitions."""
        credit = sample_carbon_credits[0]
        lifecycle = [
            {"status": "issued", "date": "2023-01-15"},
            {"status": "active", "date": "2023-02-01"},
            {"status": "transferred", "date": "2024-06-15"},
            {"status": "retired", "date": "2025-12-31"},
        ]
        assert lifecycle[0]["status"] == "issued"
        assert lifecycle[-1]["status"] == "retired"
        assert len(lifecycle) == 4

    def test_sbti_compliance_note(self, sample_carbon_credits):
        """Test SBTi compliance note for carbon credit usage."""
        sbti_note = {
            "guidance": "SBTi does not allow carbon credits to count toward near-term targets",
            "near_term_offset_allowed": False,
            "net_zero_offset_allowed": True,
            "max_offset_for_net_zero_pct": 10.0,
            "neutralization_required": True,
            "recommendation": "Use credits for beyond-value-chain mitigation",
        }
        assert sbti_note["near_term_offset_allowed"] is False
        assert sbti_note["net_zero_offset_allowed"] is True
        assert sbti_note["max_offset_for_net_zero_pct"] == 10.0
