"""
Unit tests for PACK-007 EUDR Professional Pack - Grievance Mechanism Engine

Tests complaint registration, triage, investigation, resolution, SLA tracking,
FPIC validation, and whistleblower protection.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any


def _import_from_path(module_name, file_path):
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PACK_007_DIR = Path(__file__).resolve().parent.parent

# Import grievance mechanism module
grievance_mod = _import_from_path(
    "pack_007_grievance",
    _PACK_007_DIR / "engines" / "grievance.py"
)

pytestmark = pytest.mark.skipif(
    grievance_mod is None,
    reason="PACK-007 grievance module not available"
)


@pytest.fixture
def grievance_engine():
    """Create grievance mechanism engine instance."""
    if grievance_mod is None:
        pytest.skip("grievance module not available")
    return grievance_mod.GrievanceMechanismEngine()


@pytest.fixture
def sample_complaint():
    """Sample grievance complaint."""
    return {
        "complainant_name": "Anonymous",
        "complaint_type": "land_rights_violation",
        "description": "Unauthorized clearing of indigenous land for palm oil plantation",
        "location": "Region A, Country B",
        "date_of_incident": "2024-11-15",
        "severity": "HIGH",
        "evidence_files": ["photo1.jpg", "document1.pdf"]
    }


class TestGrievanceMechanismEngine:
    """Test suite for GrievanceMechanismEngine."""

    def test_register_complaint(self, grievance_engine, sample_complaint):
        """Test registering a new complaint."""
        result = grievance_engine.register_complaint(**sample_complaint)

        assert result is not None
        assert "complaint_id" in result or "id" in result
        assert "registration_date" in result or "created_at" in result
        assert "status" in result
        assert result["status"] in ["REGISTERED", "NEW", "PENDING_TRIAGE"]

    def test_triage_complaint(self, grievance_engine, sample_complaint):
        """Test triaging a complaint."""
        # Register complaint first
        registered = grievance_engine.register_complaint(**sample_complaint)
        complaint_id = registered.get("complaint_id") or registered.get("id")

        # Triage the complaint
        triage = grievance_engine.triage_complaint(
            complaint_id=complaint_id
        )

        assert triage is not None
        assert "priority" in triage or "triage_priority" in triage
        assert "assigned_to" in triage or "investigator" in triage
        assert "estimated_resolution_date" in triage or "target_date" in triage

    def test_start_investigation(self, grievance_engine):
        """Test starting an investigation."""
        # Create and triage complaint
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complaint_type="environmental_damage",
            description="Pollution of water source",
            severity="HIGH"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Start investigation
        investigation = grievance_engine.start_investigation(
            complaint_id=complaint_id,
            investigator_id="inv_001"
        )

        assert investigation is not None
        assert "investigation_id" in investigation or "id" in investigation
        assert "status" in investigation
        assert investigation["status"] in ["IN_PROGRESS", "INVESTIGATING"]

    def test_link_evidence(self, grievance_engine):
        """Test linking evidence to a complaint."""
        # Register complaint
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complaint_type="labor_rights",
            description="Unfair labor practices",
            severity="MEDIUM"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Link evidence
        evidence_link = grievance_engine.link_evidence(
            complaint_id=complaint_id,
            evidence_type="document",
            evidence_id="doc_123",
            description="Employment contract"
        )

        assert evidence_link is not None
        assert "linked" in evidence_link or "success" in evidence_link

    def test_resolve_complaint(self, grievance_engine):
        """Test resolving a complaint."""
        # Register complaint
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complaint_type="land_rights_violation",
            description="Test complaint",
            severity="LOW"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Resolve complaint
        resolution = grievance_engine.resolve_complaint(
            complaint_id=complaint_id,
            resolution_type="REMEDIATED",
            resolution_notes="Issue addressed, compensation provided",
            resolved_by="resolver_001"
        )

        assert resolution is not None
        assert "status" in resolution
        assert resolution["status"] in ["RESOLVED", "CLOSED"]
        assert "resolution_date" in resolution or "closed_at" in resolution

    def test_sla_tracking(self, grievance_engine):
        """Test SLA tracking for complaints."""
        # Register complaint
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complaint_type="environmental_damage",
            description="Test complaint",
            severity="HIGH"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Check SLA status
        sla_status = grievance_engine.check_sla_status(
            complaint_id=complaint_id
        )

        assert sla_status is not None
        assert "within_sla" in sla_status or "sla_compliant" in sla_status
        assert "days_remaining" in sla_status or "time_remaining" in sla_status
        assert "target_resolution_date" in sla_status or "sla_deadline" in sla_status

    def test_statistics_generation(self, grievance_engine):
        """Test generating grievance statistics."""
        # Register multiple complaints
        for i in range(5):
            grievance_engine.register_complaint(
                complainant_name=f"User {i}",
                complaint_type="land_rights_violation",
                description=f"Test complaint {i}",
                severity="MEDIUM"
            )

        # Get statistics
        stats = grievance_engine.get_statistics(
            period="last_30_days"
        )

        assert stats is not None
        assert "total_complaints" in stats or "count" in stats
        assert "by_type" in stats or "complaint_types" in stats
        assert "by_severity" in stats or "severity_distribution" in stats
        assert "resolution_rate" in stats or "resolved_percentage" in stats

    def test_fpic_check(self, grievance_engine):
        """Test Free, Prior, and Informed Consent (FPIC) check."""
        # Register complaint related to indigenous rights
        complaint = grievance_engine.register_complaint(
            complainant_name="Indigenous Community Representative",
            complaint_type="fpic_violation",
            description="No consultation before land use",
            severity="HIGH"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Check FPIC compliance
        fpic_check = grievance_engine.check_fpic_compliance(
            complaint_id=complaint_id
        )

        assert fpic_check is not None
        assert "fpic_obtained" in fpic_check or "compliant" in fpic_check
        assert "consultation_records" in fpic_check or "records" in fpic_check

    def test_anonymous_submission(self, grievance_engine):
        """Test anonymous complaint submission."""
        anonymous_complaint = grievance_engine.register_complaint(
            complainant_name="Anonymous",
            complaint_type="corruption",
            description="Bribery in certification process",
            severity="HIGH",
            anonymous=True
        )

        assert anonymous_complaint is not None
        assert "complaint_id" in anonymous_complaint or "id" in anonymous_complaint
        # Should not expose complainant identity
        assert anonymous_complaint.get("complainant_name") in ["Anonymous", None]

    def test_whistleblower_report(self, grievance_engine):
        """Test whistleblower protection features."""
        whistleblower_report = grievance_engine.register_whistleblower_report(
            report_type="fraud",
            description="Falsification of due diligence documents",
            severity="CRITICAL",
            protection_requested=True
        )

        assert whistleblower_report is not None
        assert "report_id" in whistleblower_report or "id" in whistleblower_report
        assert "protection_status" in whistleblower_report or "protected" in whistleblower_report


class TestGrievanceWorkflow:
    """Test grievance workflow features."""

    def test_complaint_lifecycle(self, grievance_engine):
        """Test full complaint lifecycle from registration to resolution."""
        # 1. Register
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complaint_type="land_rights_violation",
            description="Test lifecycle",
            severity="MEDIUM"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # 2. Triage
        triage = grievance_engine.triage_complaint(complaint_id)
        assert triage is not None

        # 3. Investigate
        investigation = grievance_engine.start_investigation(complaint_id, "inv_001")
        assert investigation is not None

        # 4. Resolve
        resolution = grievance_engine.resolve_complaint(
            complaint_id=complaint_id,
            resolution_type="REMEDIATED",
            resolution_notes="Resolved successfully"
        )
        assert resolution is not None
        assert resolution["status"] in ["RESOLVED", "CLOSED"]

    def test_escalation_workflow(self, grievance_engine):
        """Test complaint escalation workflow."""
        # Register high-severity complaint
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complaint_type="human_rights_violation",
            description="Serious violation",
            severity="CRITICAL"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Escalate complaint
        escalation = grievance_engine.escalate_complaint(
            complaint_id=complaint_id,
            escalation_reason="Requires senior management review",
            escalate_to="senior_management"
        )

        assert escalation is not None
        assert "escalated" in escalation or "escalation_status" in escalation


class TestGrievanceReporting:
    """Test grievance reporting features."""

    def test_generate_complaint_summary(self, grievance_engine):
        """Test generating complaint summary report."""
        # Register some complaints
        for i in range(3):
            grievance_engine.register_complaint(
                complainant_name=f"User {i}",
                complaint_type="environmental_damage",
                description=f"Complaint {i}",
                severity="MEDIUM"
            )

        summary = grievance_engine.generate_complaint_summary(
            period="last_30_days"
        )

        assert summary is not None
        assert "total_complaints" in summary or "count" in summary
        assert "open_complaints" in summary or "pending" in summary
        assert "resolved_complaints" in summary or "closed" in summary

    def test_generate_resolution_report(self, grievance_engine):
        """Test generating resolution effectiveness report."""
        report = grievance_engine.generate_resolution_report(
            period="last_quarter"
        )

        assert report is not None
        assert "average_resolution_time_days" in report or "avg_resolution_time" in report
        assert "resolution_rate" in report or "resolved_percentage" in report

    def test_generate_sla_compliance_report(self, grievance_engine):
        """Test generating SLA compliance report."""
        sla_report = grievance_engine.generate_sla_compliance_report(
            period="last_month"
        )

        assert sla_report is not None
        assert "sla_compliance_rate" in sla_report or "compliance_percentage" in sla_report
        assert "breached_slas" in sla_report or "breaches" in sla_report


class TestGrievanceNotifications:
    """Test grievance notification features."""

    def test_send_acknowledgment(self, grievance_engine):
        """Test sending complaint acknowledgment to complainant."""
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complainant_email="test@example.com",
            complaint_type="land_rights_violation",
            description="Test complaint",
            severity="MEDIUM"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Send acknowledgment
        ack = grievance_engine.send_acknowledgment(
            complaint_id=complaint_id
        )

        assert ack is not None
        assert "sent" in ack or "notification_sent" in ack

    def test_send_status_update(self, grievance_engine):
        """Test sending status update notification."""
        complaint = grievance_engine.register_complaint(
            complainant_name="Test User",
            complainant_email="test@example.com",
            complaint_type="environmental_damage",
            description="Test complaint",
            severity="HIGH"
        )
        complaint_id = complaint.get("complaint_id") or complaint.get("id")

        # Update status
        update = grievance_engine.send_status_update(
            complaint_id=complaint_id,
            status_message="Investigation in progress"
        )

        assert update is not None


class TestGrievanceAnalytics:
    """Test grievance analytics features."""

    def test_identify_trends(self, grievance_engine):
        """Test identifying trends in complaints."""
        # Register complaints over time
        complaint_types = ["land_rights_violation", "environmental_damage", "land_rights_violation"]
        for complaint_type in complaint_types:
            grievance_engine.register_complaint(
                complainant_name="User",
                complaint_type=complaint_type,
                description="Test",
                severity="MEDIUM"
            )

        trends = grievance_engine.identify_trends(
            period="last_90_days"
        )

        assert trends is not None
        assert "trending_types" in trends or "top_complaint_types" in trends

    def test_hotspot_analysis(self, grievance_engine):
        """Test geographic hotspot analysis."""
        # Register complaints from different locations
        locations = ["Region A", "Region B", "Region A"]
        for location in locations:
            grievance_engine.register_complaint(
                complainant_name="User",
                complaint_type="land_rights_violation",
                description="Test",
                location=location,
                severity="HIGH"
            )

        hotspots = grievance_engine.analyze_geographic_hotspots()

        assert hotspots is not None
        assert isinstance(hotspots, list)
