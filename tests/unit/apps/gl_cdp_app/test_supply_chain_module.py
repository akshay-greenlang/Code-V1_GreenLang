# -*- coding: utf-8 -*-
"""
Unit tests for SupplyChainModule -- CDP supplier engagement.

Tests supplier invitation, response tracking, emissions aggregation,
engagement scoring, cascade requests, and hotspot identification
with 27+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from services.config import SupplierStatus
from services.models import (
    CDPSupplyChainRequest,
    CDPSupplierResponse,
    _new_id,
)
from services.supply_chain_module import SupplyChainModule


# ---------------------------------------------------------------------------
# Supplier invitation
# ---------------------------------------------------------------------------

class TestSupplierInvitation:
    """Test supplier invitation management."""

    def test_invite_supplier(self, supply_chain_module):
        request = supply_chain_module.invite_supplier(
            org_id=_new_id(),
            supplier_name="Steel Supplier Inc",
            supplier_email="contact@steelsupplier.com",
            supplier_sector="15104010",
        )
        assert isinstance(request, CDPSupplyChainRequest)
        assert request.status == SupplierStatus.INVITED
        assert request.invitation_sent_at is not None

    def test_invite_multiple_suppliers(self, supply_chain_module):
        org_id = _new_id()
        suppliers = [
            ("Supplier A", "a@test.com", "20101010"),
            ("Supplier B", "b@test.com", "15104010"),
            ("Supplier C", "c@test.com", "25501010"),
        ]
        requests = supply_chain_module.invite_bulk(
            org_id=org_id,
            suppliers=suppliers,
        )
        assert len(requests) == 3
        assert all(r.status == SupplierStatus.INVITED for r in requests)

    def test_duplicate_invite_raises(self, supply_chain_module):
        org_id = _new_id()
        supply_chain_module.invite_supplier(
            org_id=org_id,
            supplier_name="Dup Supplier",
            supplier_email="dup@test.com",
            supplier_sector="20101010",
        )
        with pytest.raises(ValueError, match="[Aa]lready invited"):
            supply_chain_module.invite_supplier(
                org_id=org_id,
                supplier_name="Dup Supplier",
                supplier_email="dup@test.com",
                supplier_sector="20101010",
            )

    def test_send_reminder(self, supply_chain_module, sample_supplier):
        supply_chain_module._store[sample_supplier.id] = sample_supplier
        updated = supply_chain_module.send_reminder(sample_supplier.id)
        assert updated.reminder_count >= 1


# ---------------------------------------------------------------------------
# Response tracking
# ---------------------------------------------------------------------------

class TestResponseTracking:
    """Test supplier response tracking."""

    def test_record_supplier_response(self, supply_chain_module, sample_supplier):
        supply_chain_module._store[sample_supplier.id] = sample_supplier
        response = supply_chain_module.record_response(
            request_id=sample_supplier.id,
            scope1_emissions=Decimal("1200"),
            scope2_emissions=Decimal("800"),
            scope3_emissions=Decimal("5000"),
            has_targets=True,
            has_verification=False,
            response_data={"additional": "data"},
        )
        assert isinstance(response, CDPSupplierResponse)
        assert response.scope1_emissions == Decimal("1200")

    def test_response_updates_request_status(self, supply_chain_module, sample_supplier):
        supply_chain_module._store[sample_supplier.id] = sample_supplier
        supply_chain_module.record_response(
            request_id=sample_supplier.id,
            scope1_emissions=Decimal("1000"),
            scope2_emissions=Decimal("500"),
        )
        updated_request = supply_chain_module.get_request(sample_supplier.id)
        assert updated_request.status == SupplierStatus.RESPONDED

    def test_mark_declined(self, supply_chain_module, sample_supplier):
        supply_chain_module._store[sample_supplier.id] = sample_supplier
        updated = supply_chain_module.mark_declined(sample_supplier.id)
        assert updated.status == SupplierStatus.DECLINED

    def test_get_response_rate(self, supply_chain_module):
        org_id = _new_id()
        for i in range(10):
            req = supply_chain_module.invite_supplier(
                org_id=org_id,
                supplier_name=f"Supplier {i}",
                supplier_email=f"s{i}@test.com",
                supplier_sector="20101010",
            )
            if i < 6:  # 6 out of 10 respond
                supply_chain_module.record_response(
                    request_id=req.id,
                    scope1_emissions=Decimal("1000"),
                )
        rate = supply_chain_module.get_response_rate(org_id)
        assert rate == Decimal("60.0")


# ---------------------------------------------------------------------------
# Emissions aggregation
# ---------------------------------------------------------------------------

class TestEmissionsAggregation:
    """Test supplier emissions aggregation."""

    def test_aggregate_supplier_emissions(self, supply_chain_module):
        org_id = _new_id()
        for i in range(3):
            req = supply_chain_module.invite_supplier(
                org_id=org_id,
                supplier_name=f"Agg Supplier {i}",
                supplier_email=f"agg{i}@test.com",
                supplier_sector="20101010",
            )
            supply_chain_module.record_response(
                request_id=req.id,
                scope1_emissions=Decimal("1000"),
                scope2_emissions=Decimal("500"),
                scope3_emissions=Decimal("2000"),
            )
        total = supply_chain_module.aggregate_emissions(org_id)
        assert total["total_scope1"] == Decimal("3000")
        assert total["total_scope2"] == Decimal("1500")
        assert total["total_scope3"] == Decimal("6000")

    def test_aggregate_empty_returns_zero(self, supply_chain_module):
        total = supply_chain_module.aggregate_emissions(_new_id())
        assert total["total_scope1"] == Decimal("0")


# ---------------------------------------------------------------------------
# Engagement scoring
# ---------------------------------------------------------------------------

class TestEngagementScoring:
    """Test supplier engagement scoring."""

    def test_calculate_engagement_score(self, supply_chain_module):
        score = supply_chain_module.calculate_engagement_score(
            has_responded=True,
            has_targets=True,
            has_verification=True,
            scope1_emissions=Decimal("1000"),
            scope2_emissions=Decimal("500"),
        )
        assert Decimal("0") <= score <= Decimal("100")

    def test_high_engagement_score(self, supply_chain_module):
        score = supply_chain_module.calculate_engagement_score(
            has_responded=True,
            has_targets=True,
            has_verification=True,
            scope1_emissions=Decimal("1000"),
            scope2_emissions=Decimal("500"),
        )
        assert score >= Decimal("70")

    def test_low_engagement_no_response(self, supply_chain_module):
        score = supply_chain_module.calculate_engagement_score(
            has_responded=False,
            has_targets=False,
            has_verification=False,
        )
        assert score <= Decimal("20")

    def test_average_engagement_score(self, supply_chain_module):
        org_id = _new_id()
        for i in range(5):
            req = supply_chain_module.invite_supplier(
                org_id=org_id,
                supplier_name=f"Eng Supplier {i}",
                supplier_email=f"eng{i}@test.com",
                supplier_sector="20101010",
            )
            supply_chain_module.record_response(
                request_id=req.id,
                scope1_emissions=Decimal("1000"),
                has_targets=(i < 3),
                has_verification=(i < 2),
            )
        avg = supply_chain_module.get_average_engagement_score(org_id)
        assert Decimal("0") <= avg <= Decimal("100")


# ---------------------------------------------------------------------------
# Cascade requests
# ---------------------------------------------------------------------------

class TestCascadeRequests:
    """Test cascade request management."""

    def test_create_cascade_request(self, supply_chain_module):
        cascade = supply_chain_module.create_cascade_request(
            org_id=_new_id(),
            tier=2,
            scope="scope_1_2",
            target_suppliers=["supplier-a", "supplier-b"],
        )
        assert cascade["tier"] == 2
        assert len(cascade["target_suppliers"]) == 2

    def test_cascade_tracks_responses(self, supply_chain_module):
        org_id = _new_id()
        cascade = supply_chain_module.create_cascade_request(
            org_id=org_id,
            tier=2,
            scope="all",
            target_suppliers=["s1", "s2", "s3"],
        )
        supply_chain_module.record_cascade_response(cascade["id"], "s1")
        status = supply_chain_module.get_cascade_status(cascade["id"])
        assert status["responded"] == 1
        assert status["pending"] == 2


# ---------------------------------------------------------------------------
# Hotspot identification
# ---------------------------------------------------------------------------

class TestHotspotIdentification:
    """Test supply chain emissions hotspot identification."""

    def test_identify_hotspots(self, supply_chain_module):
        org_id = _new_id()
        # High-emission supplier
        req1 = supply_chain_module.invite_supplier(
            org_id=org_id, supplier_name="Big Emitter",
            supplier_email="big@test.com", supplier_sector="15104010",
        )
        supply_chain_module.record_response(
            request_id=req1.id, scope1_emissions=Decimal("50000"),
            scope2_emissions=Decimal("20000"),
        )
        # Low-emission supplier
        req2 = supply_chain_module.invite_supplier(
            org_id=org_id, supplier_name="Small Emitter",
            supplier_email="small@test.com", supplier_sector="25501010",
        )
        supply_chain_module.record_response(
            request_id=req2.id, scope1_emissions=Decimal("100"),
            scope2_emissions=Decimal("50"),
        )
        hotspots = supply_chain_module.identify_hotspots(org_id, top_n=1)
        assert len(hotspots) == 1
        assert hotspots[0]["supplier_name"] == "Big Emitter"

    def test_hotspot_by_sector(self, supply_chain_module):
        org_id = _new_id()
        for i in range(5):
            req = supply_chain_module.invite_supplier(
                org_id=org_id, supplier_name=f"Sector Supplier {i}",
                supplier_email=f"sec{i}@test.com", supplier_sector="15104010",
            )
            supply_chain_module.record_response(
                request_id=req.id,
                scope1_emissions=Decimal(str(1000 * (i + 1))),
            )
        sector_hotspots = supply_chain_module.identify_hotspots_by_sector(org_id)
        assert "15104010" in sector_hotspots


# ---------------------------------------------------------------------------
# Supplier data quality
# ---------------------------------------------------------------------------

class TestSupplierDataQuality:
    """Test supplier data quality assessment."""

    def test_data_quality_high(self, supply_chain_module):
        quality = supply_chain_module.assess_data_quality(
            has_scope1=True,
            has_scope2=True,
            has_scope3=True,
            has_targets=True,
            has_verification=True,
        )
        assert quality["level"] == "high"

    def test_data_quality_medium(self, supply_chain_module):
        quality = supply_chain_module.assess_data_quality(
            has_scope1=True,
            has_scope2=True,
            has_scope3=False,
            has_targets=True,
            has_verification=False,
        )
        assert quality["level"] == "medium"

    def test_data_quality_low(self, supply_chain_module):
        quality = supply_chain_module.assess_data_quality(
            has_scope1=True,
            has_scope2=False,
            has_scope3=False,
            has_targets=False,
            has_verification=False,
        )
        assert quality["level"] == "low"

    def test_data_quality_score_range(self, supply_chain_module):
        quality = supply_chain_module.assess_data_quality(
            has_scope1=True, has_scope2=True, has_scope3=True,
            has_targets=True, has_verification=True,
        )
        assert Decimal("0") <= quality["score"] <= Decimal("100")


# ---------------------------------------------------------------------------
# Supplier improvement tracking
# ---------------------------------------------------------------------------

class TestSupplierImprovement:
    """Test supplier improvement tracking over time."""

    def test_track_improvement(self, supply_chain_module):
        org_id = _new_id()
        req = supply_chain_module.invite_supplier(
            org_id=org_id, supplier_name="Improving Co",
            supplier_email="imp@test.com", supplier_sector="20101010",
        )
        supply_chain_module.record_response(
            request_id=req.id,
            scope1_emissions=Decimal("5000"),
            scope2_emissions=Decimal("3000"),
        )
        improvement = supply_chain_module.track_supplier_improvement(
            request_id=req.id,
            previous_scope1=Decimal("6000"),
            previous_scope2=Decimal("3500"),
        )
        assert improvement["scope1_delta"] == Decimal("-1000")
        assert improvement["scope2_delta"] == Decimal("-500")
        assert improvement["improved"] is True

    def test_track_no_improvement(self, supply_chain_module):
        org_id = _new_id()
        req = supply_chain_module.invite_supplier(
            org_id=org_id, supplier_name="Stagnant Co",
            supplier_email="stag@test.com", supplier_sector="20101010",
        )
        supply_chain_module.record_response(
            request_id=req.id,
            scope1_emissions=Decimal("6000"),
        )
        improvement = supply_chain_module.track_supplier_improvement(
            request_id=req.id,
            previous_scope1=Decimal("5000"),
        )
        assert improvement["improved"] is False

    def test_supplier_summary_statistics(self, supply_chain_module):
        org_id = _new_id()
        for i in range(5):
            req = supply_chain_module.invite_supplier(
                org_id=org_id, supplier_name=f"Stats Supplier {i}",
                supplier_email=f"stats{i}@test.com", supplier_sector="20101010",
            )
            if i < 3:
                supply_chain_module.record_response(
                    request_id=req.id,
                    scope1_emissions=Decimal(str(1000 * (i + 1))),
                )
        stats = supply_chain_module.get_summary_statistics(org_id)
        assert stats["total_invited"] == 5
        assert stats["total_responded"] == 3
        assert stats["response_rate"] == Decimal("60.0")
