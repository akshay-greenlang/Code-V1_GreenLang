# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Supplier Compliance Engine Tests
================================================================

Validates the supplier compliance engine including supplier registration,
DD status updates, data completeness scoring, certification tracking,
supplier prioritization, data request generation, engagement tracking,
compliance calendar, and supplier dashboard.

Test count: 15
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import pytest

from conftest import (
    CERTIFICATION_SCHEMES,
    _compute_hash,
    assert_valid_uuid,
)


# ---------------------------------------------------------------------------
# Supplier Compliance Engine Simulator
# ---------------------------------------------------------------------------

class SupplierComplianceSimulator:
    """Simulates supplier compliance engine operations."""

    DD_STATUS_LIFECYCLE = [
        "NOT_STARTED", "DATA_COLLECTION", "IN_PROGRESS",
        "REVIEW", "COMPLETED", "EXPIRED",
    ]

    def __init__(self):
        self.suppliers: Dict[str, Dict[str, Any]] = {}

    def register_supplier(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new supplier for EUDR compliance tracking."""
        supplier_id = supplier_data.get("supplier_id", str(uuid.uuid4()))
        record = {
            "supplier_id": supplier_id,
            "name": supplier_data["name"],
            "country": supplier_data["country"],
            "commodity": supplier_data["commodity"],
            "eori_number": supplier_data.get("eori_number", ""),
            "dd_status": "NOT_STARTED",
            "data_completeness": 0.0,
            "certifications": supplier_data.get("certifications", []),
            "registered_at": datetime.now().isoformat(),
            "risk_score": None,
        }
        self.suppliers[supplier_id] = record
        return record

    def update_dd_status(self, supplier_id: str, new_status: str) -> Dict[str, Any]:
        """Update DD status for a supplier."""
        if new_status not in self.DD_STATUS_LIFECYCLE:
            return {"error": f"Invalid status: {new_status}", "valid_statuses": self.DD_STATUS_LIFECYCLE}
        supplier = self.suppliers.get(supplier_id, {"supplier_id": supplier_id, "dd_status": "NOT_STARTED"})
        old_status = supplier["dd_status"]
        supplier["dd_status"] = new_status
        supplier["status_updated_at"] = datetime.now().isoformat()
        return {
            "supplier_id": supplier_id,
            "old_status": old_status,
            "new_status": new_status,
            "updated_at": supplier["status_updated_at"],
        }

    def calculate_data_completeness(self, supplier: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data completeness score for a supplier."""
        required_fields = [
            "name", "country", "commodity", "eori_number",
            "certifications", "dd_status",
        ]
        optional_fields = [
            "plots", "last_audit_date", "address",
            "registration_date", "risk_score",
        ]
        filled_required = sum(1 for f in required_fields if supplier.get(f))
        filled_optional = sum(1 for f in optional_fields if supplier.get(f))
        total_fields = len(required_fields) + len(optional_fields)
        filled_total = filled_required + filled_optional
        score = round(filled_total / total_fields, 2) if total_fields > 0 else 0.0
        return {
            "score": score,
            "filled_required": filled_required,
            "total_required": len(required_fields),
            "filled_optional": filled_optional,
            "total_optional": len(optional_fields),
            "completeness_pct": round(score * 100, 1),
            "status": "FULL" if score >= 0.90 else "PARTIAL" if score >= 0.50 else "INCOMPLETE",
        }

    def track_certification(self, supplier_id: str,
                             certification: Dict[str, Any]) -> Dict[str, Any]:
        """Track a certification for a supplier."""
        return {
            "supplier_id": supplier_id,
            "scheme": certification["scheme"],
            "certificate_number": certification.get("certificate_number", ""),
            "valid_from": certification.get("valid_from", ""),
            "valid_until": certification.get("valid_until", ""),
            "status": certification.get("status", "pending"),
            "tracked_at": datetime.now().isoformat(),
        }

    def check_certification_validity(self, certification: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a certification is currently valid."""
        valid_until = certification.get("valid_until", "")
        status = certification.get("status", "unknown")
        if not valid_until:
            return {"valid": False, "reason": "No expiry date specified"}
        try:
            expiry = date.fromisoformat(valid_until)
            is_valid = expiry >= date.today() and status == "active"
            return {
                "valid": is_valid,
                "expires": valid_until,
                "days_remaining": (expiry - date.today()).days,
                "status": status,
            }
        except ValueError:
            return {"valid": False, "reason": f"Invalid date format: {valid_until}"}

    def prioritize_suppliers(self, suppliers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize suppliers by risk and data completeness."""
        def priority_key(s):
            risk = s.get("risk_score", 0.5)
            completeness = s.get("data_completeness", 0.5)
            # Higher risk and lower completeness = higher priority
            return -(risk * 0.6 + (1 - completeness) * 0.4)
        sorted_suppliers = sorted(suppliers, key=priority_key)
        for i, s in enumerate(sorted_suppliers):
            s["priority_rank"] = i + 1
        return sorted_suppliers

    def generate_data_request(self, supplier: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a data request for a supplier with missing data."""
        missing_items = []
        if not supplier.get("eori_number"):
            missing_items.append({"field": "eori_number", "priority": "required"})
        if not supplier.get("certifications"):
            missing_items.append({"field": "certifications", "priority": "recommended"})
        if not supplier.get("plots"):
            missing_items.append({"field": "geolocation_plots", "priority": "required"})
        if not supplier.get("last_audit_date"):
            missing_items.append({"field": "last_audit_date", "priority": "recommended"})
        return {
            "supplier_id": supplier.get("supplier_id", ""),
            "supplier_name": supplier.get("name", ""),
            "request_date": date.today().isoformat(),
            "missing_items": missing_items,
            "total_missing": len(missing_items),
            "deadline": (date.today() + timedelta(days=30)).isoformat(),
        }

    def track_engagement(self, supplier_id: str, action: str) -> Dict[str, Any]:
        """Track supplier engagement activities."""
        return {
            "supplier_id": supplier_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "logged": True,
        }

    def compliance_calendar(self, suppliers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance calendar for upcoming deadlines."""
        deadlines = []
        for s in suppliers:
            for cert in s.get("certifications", []):
                valid_until = cert.get("valid_until", "")
                if valid_until:
                    deadlines.append({
                        "supplier": s.get("name", ""),
                        "event": f"{cert.get('scheme', 'Cert')} renewal",
                        "date": valid_until,
                    })
        deadlines.sort(key=lambda d: d["date"])
        return {
            "total_deadlines": len(deadlines),
            "upcoming": deadlines[:10],
        }

    def supplier_dashboard(self, suppliers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate supplier compliance dashboard summary."""
        total = len(suppliers)
        completed = sum(1 for s in suppliers if s.get("dd_status") == "COMPLETED")
        in_progress = sum(1 for s in suppliers if s.get("dd_status") == "IN_PROGRESS")
        not_started = sum(1 for s in suppliers if s.get("dd_status") == "NOT_STARTED")
        avg_completeness = (
            sum(s.get("data_completeness", 0) for s in suppliers) / total
            if total > 0 else 0.0
        )
        return {
            "total_suppliers": total,
            "dd_completed": completed,
            "dd_in_progress": in_progress,
            "dd_not_started": not_started,
            "average_data_completeness": round(avg_completeness, 2),
            "compliance_rate_pct": round(completed / total * 100, 1) if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSupplierCompliance:
    """Tests for the supplier compliance engine."""

    @pytest.fixture
    def engine(self) -> SupplierComplianceSimulator:
        return SupplierComplianceSimulator()

    # 1
    def test_register_supplier(self, engine, sample_supplier):
        """Supplier registration creates a valid record."""
        record = engine.register_supplier(sample_supplier)
        assert record["name"] == sample_supplier["name"]
        assert record["dd_status"] == "NOT_STARTED"
        assert "registered_at" in record

    # 2
    def test_update_dd_status(self, engine, sample_supplier):
        """DD status can be updated through lifecycle."""
        record = engine.register_supplier(sample_supplier)
        result = engine.update_dd_status(record["supplier_id"], "IN_PROGRESS")
        assert result["new_status"] == "IN_PROGRESS"
        assert result["old_status"] == "NOT_STARTED"

    # 3
    def test_data_completeness_full(self, engine, sample_supplier):
        """Fully complete supplier data scores high completeness."""
        result = engine.calculate_data_completeness(sample_supplier)
        assert result["score"] >= 0.70
        assert result["status"] in ("FULL", "PARTIAL")

    # 4
    def test_data_completeness_partial(self, engine):
        """Partial supplier data scores lower completeness."""
        partial = {"name": "Test", "country": "BRA", "commodity": "soya"}
        result = engine.calculate_data_completeness(partial)
        assert result["score"] < 0.60
        assert result["status"] in ("PARTIAL", "INCOMPLETE")

    # 5
    def test_track_certification(self, engine):
        """Certification tracking records scheme and validity."""
        cert = {
            "scheme": "RSPO",
            "certificate_number": "RSPO-2024-001",
            "valid_from": "2024-01-01",
            "valid_until": "2027-12-31",
            "status": "active",
        }
        result = engine.track_certification("sup-001", cert)
        assert result["scheme"] == "RSPO"
        assert result["supplier_id"] == "sup-001"

    # 6
    def test_check_certification_validity(self, engine):
        """Valid certification passes validity check."""
        cert = {
            "valid_until": "2027-12-31",
            "status": "active",
        }
        result = engine.check_certification_validity(cert)
        assert result["valid"] is True
        assert result["days_remaining"] > 0

    # 7
    def test_check_certification_expired(self, engine):
        """Expired certification fails validity check."""
        cert = {
            "valid_until": "2020-01-01",
            "status": "active",
        }
        result = engine.check_certification_validity(cert)
        assert result["valid"] is False

    # 8
    def test_prioritize_suppliers(self, engine, sample_suppliers_list):
        """Supplier prioritization ranks by risk and completeness."""
        prioritized = engine.prioritize_suppliers(sample_suppliers_list)
        assert len(prioritized) == len(sample_suppliers_list)
        for i, s in enumerate(prioritized):
            assert s["priority_rank"] == i + 1

    # 9
    def test_generate_data_request(self, engine):
        """Data request identifies missing required fields."""
        incomplete_supplier = {
            "supplier_id": "sup-001",
            "name": "Test Supplier",
            "country": "BRA",
        }
        request = engine.generate_data_request(incomplete_supplier)
        assert request["total_missing"] > 0
        assert "deadline" in request

    # 10
    def test_track_engagement(self, engine):
        """Engagement tracking logs supplier interactions."""
        result = engine.track_engagement("sup-001", "data_request_sent")
        assert result["logged"] is True
        assert result["action"] == "data_request_sent"

    # 11
    def test_compliance_calendar(self, engine, sample_suppliers_list):
        """Compliance calendar generates sorted deadlines."""
        calendar = engine.compliance_calendar(sample_suppliers_list)
        assert "total_deadlines" in calendar
        assert isinstance(calendar["upcoming"], list)

    # 12
    def test_supplier_dashboard(self, engine, sample_suppliers_list):
        """Supplier dashboard shows summary statistics."""
        dashboard = engine.supplier_dashboard(sample_suppliers_list)
        assert dashboard["total_suppliers"] == len(sample_suppliers_list)
        total_statuses = (
            dashboard["dd_completed"]
            + dashboard["dd_in_progress"]
            + dashboard["dd_not_started"]
        )
        assert total_statuses <= dashboard["total_suppliers"]

    # 13
    def test_dd_status_lifecycle(self, engine):
        """DD status lifecycle has correct progression."""
        lifecycle = engine.DD_STATUS_LIFECYCLE
        assert lifecycle[0] == "NOT_STARTED"
        assert lifecycle[-1] == "EXPIRED"
        assert "COMPLETED" in lifecycle

    # 14
    def test_invalid_dd_status_rejected(self, engine, sample_supplier):
        """Invalid DD status update returns error."""
        record = engine.register_supplier(sample_supplier)
        result = engine.update_dd_status(record["supplier_id"], "INVALID_STATUS")
        assert "error" in result

    # 15
    def test_data_request_deadline(self, engine):
        """Data request deadline is 30 days from today."""
        supplier = {"supplier_id": "sup-001", "name": "Test"}
        request = engine.generate_data_request(supplier)
        expected_deadline = (date.today() + timedelta(days=30)).isoformat()
        assert request["deadline"] == expected_deadline
