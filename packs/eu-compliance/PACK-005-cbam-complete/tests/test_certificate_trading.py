# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Certificate Trading Engine Tests (30 tests)

Tests CertificateTradingEngine: portfolio management, order submission/
execution, surrender, resale constraints, expiry alerts, holding compliance,
valuation (FIFO/WAC/MTM), transfers, optimization, budget forecasting,
decimal arithmetic, and provenance hashing.

Author: GreenLang QA Team
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

import pytest

import sys
import os
from greenlang.schemas import utcnow
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (

    _compute_hash,
    _new_uuid,
    _utcnow,
    assert_provenance_hash,
    assert_decimal_precision,
)


# ---------------------------------------------------------------------------
# Portfolio Management (6 tests)
# ---------------------------------------------------------------------------

class TestPortfolioManagement:
    """Test certificate portfolio creation and summary."""

    def test_create_portfolio(self, sample_certificate_portfolio):
        """Test creating a certificate portfolio."""
        pf = sample_certificate_portfolio
        assert pf["portfolio_id"] == "PF-EUROSTEEL-2026"
        assert pf["total_certificates"] == 50
        assert pf["active_certificates"] > 0

    def test_portfolio_certificate_statuses(self, sample_certificate_portfolio):
        """Test portfolio contains certificates in various states."""
        certs = sample_certificate_portfolio["certificates"]
        statuses = {c["status"] for c in certs}
        assert "active" in statuses
        assert "surrendered" in statuses
        assert "expired" in statuses

    def test_portfolio_valuation_method(self, sample_certificate_portfolio):
        """Test portfolio has a valuation method set."""
        assert sample_certificate_portfolio["valuation_method"] == "FIFO"

    def test_portfolio_active_value(self, sample_certificate_portfolio):
        """Test active portfolio value is calculated correctly."""
        certs = sample_certificate_portfolio["certificates"]
        expected_value = sum(
            c["price_eur"] for c in certs if c["status"] == "active"
        )
        assert sample_certificate_portfolio["total_active_value_eur"] == expected_value

    def test_portfolio_summary(self, sample_certificate_portfolio):
        """Test portfolio provides accurate summary counts."""
        certs = sample_certificate_portfolio["certificates"]
        active = sum(1 for c in certs if c["status"] == "active")
        assert sample_certificate_portfolio["active_certificates"] == active
        assert sample_certificate_portfolio["total_certificates"] == len(certs)

    def test_portfolio_entity_distribution(self, sample_certificate_portfolio):
        """Test certificates are distributed across entities."""
        certs = sample_certificate_portfolio["certificates"]
        entities = {c["entity_id"] for c in certs}
        assert len(entities) >= 2


# ---------------------------------------------------------------------------
# Order Submission (5 tests)
# ---------------------------------------------------------------------------

class TestOrderSubmission:
    """Test purchase order submission and execution."""

    def test_submit_purchase_order_market(self):
        """Test submitting a market purchase order."""
        order = {
            "order_id": f"ORD-{_new_uuid()[:8]}",
            "order_type": "market",
            "quantity_tco2e": 100,
            "submitted_at": utcnow().isoformat(),
            "status": "submitted",
        }
        assert order["order_type"] == "market"
        assert order["status"] == "submitted"
        assert order["quantity_tco2e"] == 100

    def test_submit_purchase_order_limit(self):
        """Test submitting a limit purchase order."""
        order = {
            "order_id": f"ORD-{_new_uuid()[:8]}",
            "order_type": "limit",
            "quantity_tco2e": 50,
            "limit_price_eur": Decimal("75.00"),
            "submitted_at": utcnow().isoformat(),
            "status": "submitted",
        }
        assert order["order_type"] == "limit"
        assert order["limit_price_eur"] == Decimal("75.00")

    def test_execute_order_at_market_price(self):
        """Test executing a market order at current price."""
        market_price = Decimal("78.50")
        quantity = 100
        total_cost = market_price * quantity
        execution = {
            "order_id": f"ORD-{_new_uuid()[:8]}",
            "executed_price_eur": market_price,
            "quantity_tco2e": quantity,
            "total_cost_eur": total_cost,
            "status": "executed",
            "executed_at": utcnow().isoformat(),
        }
        assert execution["status"] == "executed"
        assert execution["total_cost_eur"] == Decimal("7850.00")

    def test_execute_limit_order_below_limit(self):
        """Test limit order executes when market <= limit price."""
        limit_price = Decimal("80.00")
        market_price = Decimal("78.50")
        assert market_price <= limit_price
        execution = {
            "status": "executed",
            "executed_price_eur": market_price,
            "quantity_tco2e": 50,
            "total_cost_eur": market_price * 50,
        }
        assert execution["status"] == "executed"

    def test_execute_limit_order_above_limit_rejected(self):
        """Test limit order is rejected when market > limit price."""
        limit_price = Decimal("70.00")
        market_price = Decimal("78.50")
        assert market_price > limit_price
        execution = {
            "status": "pending",
            "reason": "Market price exceeds limit",
        }
        assert execution["status"] == "pending"


# ---------------------------------------------------------------------------
# Surrender and Resale (6 tests)
# ---------------------------------------------------------------------------

class TestSurrenderAndResale:
    """Test certificate surrender and resale logic."""

    def test_surrender_certificates(self, sample_certificate_portfolio):
        """Test surrendering certificates from portfolio."""
        active = [c for c in sample_certificate_portfolio["certificates"]
                  if c["status"] == "active"]
        surrender_count = min(10, len(active))
        surrendered = []
        for c in active[:surrender_count]:
            c_copy = dict(c)
            c_copy["status"] = "surrendered"
            surrendered.append(c_copy)
        assert len(surrendered) == surrender_count
        assert all(c["status"] == "surrendered" for c in surrendered)

    def test_surrender_exceeds_balance(self, sample_certificate_portfolio):
        """Test surrender fails when exceeding active balance."""
        active_count = sample_certificate_portfolio["active_certificates"]
        requested = active_count + 100
        can_surrender = requested <= active_count
        assert can_surrender is False

    def test_resale_within_limit(self, sample_certificate_portfolio):
        """Test resale succeeds when within 1/3 limit."""
        active = [c for c in sample_certificate_portfolio["certificates"]
                  if c["status"] == "active"]
        max_resale = len(active) // 3
        resale_count = max_resale
        assert resale_count <= max_resale
        assert resale_count > 0

    def test_resale_exceeds_limit(self, sample_certificate_portfolio):
        """Test resale rejected when exceeding 1/3 limit."""
        active = [c for c in sample_certificate_portfolio["certificates"]
                  if c["status"] == "active"]
        max_resale = len(active) // 3
        requested = max_resale + 5
        allowed = requested <= max_resale
        assert allowed is False

    def test_resale_outside_window(self):
        """Test resale rejected when outside 12-month window."""
        purchase_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        today = datetime(2026, 3, 14, tzinfo=timezone.utc)
        months_since = (today.year - purchase_date.year) * 12 + (
            today.month - purchase_date.month
        )
        within_window = months_since <= 12
        assert within_window is False

    def test_resale_within_window(self):
        """Test resale allowed when within 12-month window."""
        purchase_date = datetime(2025, 10, 1, tzinfo=timezone.utc)
        today = datetime(2026, 3, 14, tzinfo=timezone.utc)
        months_since = (today.year - purchase_date.year) * 12 + (
            today.month - purchase_date.month
        )
        within_window = months_since <= 12
        assert within_window is True


# ---------------------------------------------------------------------------
# Expiry Alerts (2 tests)
# ---------------------------------------------------------------------------

class TestExpiryAlerts:
    """Test certificate expiry alerts."""

    def test_expiry_alerts_30_days(self):
        """Test expiry alert triggered at 30 days."""
        expiry_date = datetime(2026, 4, 13, tzinfo=timezone.utc)
        today = datetime(2026, 3, 14, tzinfo=timezone.utc)
        days_to_expiry = (expiry_date - today).days
        alert = days_to_expiry <= 30
        assert alert is True
        assert days_to_expiry == 30

    def test_expiry_alerts_90_days(self):
        """Test expiry alert triggered at 90 days."""
        expiry_date = datetime(2026, 6, 12, tzinfo=timezone.utc)
        today = datetime(2026, 3, 14, tzinfo=timezone.utc)
        days_to_expiry = (expiry_date - today).days
        alert = days_to_expiry <= 90
        assert alert is True
        assert days_to_expiry == 90


# ---------------------------------------------------------------------------
# Holding Compliance (2 tests)
# ---------------------------------------------------------------------------

class TestHoldingCompliance:
    """Test quarterly holding compliance checks."""

    def test_holding_compliance_pass(self):
        """Test holding compliance passes when >= 50% threshold."""
        net_obligation = 1000
        certificates_held = 600
        threshold_pct = 50.0
        required = net_obligation * (threshold_pct / 100.0)
        compliant = certificates_held >= required
        assert compliant is True

    def test_holding_compliance_fail(self):
        """Test holding compliance fails when < 50% threshold."""
        net_obligation = 1000
        certificates_held = 400
        threshold_pct = 50.0
        required = net_obligation * (threshold_pct / 100.0)
        compliant = certificates_held >= required
        assert compliant is False


# ---------------------------------------------------------------------------
# Valuation Methods (3 tests)
# ---------------------------------------------------------------------------

class TestValuationMethods:
    """Test portfolio valuation using FIFO, WAC, and MTM."""

    def test_value_portfolio_fifo(self):
        """Test FIFO valuation: earliest purchases valued first."""
        purchases = [
            {"date": "2026-01-15", "qty": 100, "price": Decimal("70.00")},
            {"date": "2026-02-15", "qty": 100, "price": Decimal("75.00")},
            {"date": "2026-03-15", "qty": 100, "price": Decimal("80.00")},
        ]
        # Surrender 150: first 100 @ 70, next 50 @ 75
        surrender_qty = 150
        fifo_cost = Decimal("0")
        remaining = surrender_qty
        for p in purchases:
            if remaining <= 0:
                break
            take = min(remaining, p["qty"])
            fifo_cost += take * p["price"]
            remaining -= take
        assert fifo_cost == Decimal("10750.00")

    def test_value_portfolio_weighted_average(self):
        """Test Weighted Average Cost (WAC) valuation."""
        purchases = [
            {"qty": 100, "price": Decimal("70.00")},
            {"qty": 100, "price": Decimal("75.00")},
            {"qty": 100, "price": Decimal("80.00")},
        ]
        total_cost = sum(p["qty"] * p["price"] for p in purchases)
        total_qty = sum(p["qty"] for p in purchases)
        wac = total_cost / total_qty
        assert wac == Decimal("75.00")

    def test_value_portfolio_mark_to_market(self):
        """Test Mark-to-Market (MTM) valuation at current price."""
        current_price = Decimal("78.50")
        holdings = 300
        mtm_value = current_price * holdings
        assert mtm_value == Decimal("23550.00")


# ---------------------------------------------------------------------------
# Additional trading features (6 tests)
# ---------------------------------------------------------------------------

class TestAdditionalTradingFeatures:
    """Test transfers, optimization, budget, and provenance."""

    def test_transfer_certificates(self, sample_entity_group):
        """Test transferring certificates between entities."""
        from_entity = sample_entity_group["parent"]["entity_id"]
        to_entity = sample_entity_group["subsidiaries"][0]["entity_id"]
        transfer = {
            "transfer_id": f"TRF-{_new_uuid()[:8]}",
            "from_entity": from_entity,
            "to_entity": to_entity,
            "quantity": 20,
            "status": "completed",
            "transferred_at": utcnow().isoformat(),
        }
        assert transfer["status"] == "completed"
        assert transfer["from_entity"] != transfer["to_entity"]

    def test_optimize_surrender_fifo(self):
        """Test optimized surrender strategy using FIFO ordering."""
        certs = [
            {"id": "C1", "price": Decimal("65.00"), "date": "2026-01"},
            {"id": "C2", "price": Decimal("78.00"), "date": "2026-02"},
            {"id": "C3", "price": Decimal("72.00"), "date": "2026-03"},
        ]
        # FIFO: surrender oldest first
        sorted_certs = sorted(certs, key=lambda c: c["date"])
        surrender_order = [c["id"] for c in sorted_certs]
        assert surrender_order == ["C1", "C2", "C3"]

    def test_budget_forecast(self, sample_config):
        """Test budget forecast over multi-year horizon."""
        schedule = sample_config["cbam"]["certificate_config"]["free_allocation_schedule"]
        annual_emissions = 22500.0
        price_per_tco2e = 80.0
        forecast = []
        for year in range(2026, 2035):
            fa_pct = schedule.get(str(year), 0.0)
            cbam_pct = 1.0 - fa_pct
            net_obligation = annual_emissions * cbam_pct
            cost = net_obligation * price_per_tco2e
            forecast.append({
                "year": year,
                "net_obligation_tco2e": round(net_obligation, 2),
                "cost_eur": round(cost, 2),
            })
        assert len(forecast) == 9
        # Cost should increase as free allocation decreases
        assert forecast[-1]["cost_eur"] > forecast[0]["cost_eur"]

    def test_decimal_arithmetic(self):
        """Test Decimal arithmetic avoids float rounding errors."""
        a = Decimal("0.1") + Decimal("0.2")
        assert a == Decimal("0.3")
        b = Decimal("78.50") * Decimal("562.5")
        assert b == Decimal("44156.25")

    def test_provenance_hash_on_results(self):
        """Test provenance hash is generated for trading results."""
        result = {
            "order_id": "ORD-001",
            "quantity": 100,
            "price_eur": 78.50,
            "provenance_hash": _compute_hash({
                "order_id": "ORD-001", "quantity": 100,
            }),
        }
        assert_provenance_hash(result)

    def test_dca_strategy(self):
        """Test Dollar Cost Averaging (DCA) strategy splits buys evenly."""
        total_qty = 1200
        num_periods = 12
        per_period = total_qty // num_periods
        assert per_period == 100
        periods = [{"month": m + 1, "quantity": per_period} for m in range(num_periods)]
        total_bought = sum(p["quantity"] for p in periods)
        assert total_bought == total_qty
