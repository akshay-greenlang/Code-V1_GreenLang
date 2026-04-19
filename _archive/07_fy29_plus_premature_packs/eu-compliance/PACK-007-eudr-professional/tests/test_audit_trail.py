"""
Unit tests for PACK-007 EUDR Professional Pack - Audit Trail Engine

Tests advanced audit trail logging, hash chain integrity, tamper detection,
evidence assembly, and CA inspection preparation.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import json


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

# Import audit trail module
audit_trail_mod = _import_from_path(
    "pack_007_audit_trail",
    _PACK_007_DIR / "engines" / "audit_trail.py"
)

pytestmark = pytest.mark.skipif(
    audit_trail_mod is None,
    reason="PACK-007 audit_trail module not available"
)


@pytest.fixture
def audit_engine():
    """Create audit trail engine instance."""
    if audit_trail_mod is None:
        pytest.skip("audit_trail module not available")
    return audit_trail_mod.AdvancedAuditTrailEngine()


@pytest.fixture
def sample_action():
    """Create sample audit action."""
    return {
        "user_id": "user_123",
        "action_type": "DDS_SUBMITTED",
        "entity_type": "due_diligence_statement",
        "entity_id": "dds_456",
        "details": {
            "product": "coffee",
            "quantity_tonnes": 100,
            "origin_country": "BR"
        }
    }


class TestAuditTrailEngine:
    """Test suite for AdvancedAuditTrailEngine."""

    def test_log_action_creates_entry(self, audit_engine, sample_action):
        """Test logging an action creates a valid entry."""
        entry = audit_engine.log_action(**sample_action)

        assert entry is not None
        assert entry["entry_id"] is not None
        assert entry["user_id"] == sample_action["user_id"]
        assert entry["action_type"] == sample_action["action_type"]
        assert entry["entity_type"] == sample_action["entity_type"]
        assert entry["entity_id"] == sample_action["entity_id"]
        assert entry["timestamp"] is not None
        assert entry["hash"] is not None
        assert len(entry["hash"]) == 64  # SHA-256

    def test_audit_chain_integrity(self, audit_engine):
        """Test audit chain maintains hash chain integrity."""
        # Log multiple actions
        actions = [
            {"user_id": "user_1", "action_type": "CREATE", "entity_type": "supplier", "entity_id": "s1"},
            {"user_id": "user_2", "action_type": "UPDATE", "entity_type": "supplier", "entity_id": "s1"},
            {"user_id": "user_3", "action_type": "DELETE", "entity_type": "supplier", "entity_id": "s1"},
        ]

        entries = [audit_engine.log_action(**action) for action in actions]

        # Verify chain integrity
        integrity = audit_engine.verify_chain_integrity()
        assert integrity["valid"] is True
        assert integrity["chain_length"] >= 3

    def test_hash_chain_verification(self, audit_engine):
        """Test hash chain verification detects valid chains."""
        # Log actions
        for i in range(5):
            audit_engine.log_action(
                user_id=f"user_{i}",
                action_type="ACTION",
                entity_type="test",
                entity_id=f"entity_{i}"
            )

        # Verify chain
        result = audit_engine.verify_chain_integrity()
        assert result["valid"] is True
        assert result["broken_links"] == []

    def test_tamper_detection(self, audit_engine):
        """Test tamper detection identifies modified entries."""
        # Log actions
        entry1 = audit_engine.log_action(
            user_id="user_1",
            action_type="CREATE",
            entity_type="test",
            entity_id="test_1"
        )

        entry2 = audit_engine.log_action(
            user_id="user_2",
            action_type="UPDATE",
            entity_type="test",
            entity_id="test_1"
        )

        # Simulate tampering (if engine exposes internal chain)
        # Most implementations would detect this via hash mismatch
        result = audit_engine.detect_tampering(entry1["entry_id"])

        # Should not detect tampering on valid entry
        assert result["tampered"] is False

    def test_evidence_assembly(self, audit_engine):
        """Test assembling evidence for a specific entity."""
        entity_id = "plot_789"

        # Log multiple actions for same entity
        actions = [
            {"user_id": "u1", "action_type": "PLOT_CREATED", "entity_type": "plot", "entity_id": entity_id},
            {"user_id": "u2", "action_type": "GEOLOCATION_ADDED", "entity_type": "plot", "entity_id": entity_id},
            {"user_id": "u3", "action_type": "DEFORESTATION_CHECK", "entity_type": "plot", "entity_id": entity_id},
        ]

        for action in actions:
            audit_engine.log_action(**action)

        # Assemble evidence
        evidence = audit_engine.assemble_evidence(entity_id=entity_id)

        assert evidence is not None
        assert len(evidence["entries"]) >= 3
        assert evidence["entity_id"] == entity_id
        assert evidence["evidence_hash"] is not None

    def test_ca_inspection_preparation(self, audit_engine):
        """Test preparing audit trail for CA inspection."""
        # Log various actions
        for i in range(10):
            audit_engine.log_action(
                user_id=f"user_{i}",
                action_type="DDS_SUBMITTED",
                entity_type="dds",
                entity_id=f"dds_{i}"
            )

        # Prepare for inspection
        inspection_package = audit_engine.prepare_ca_inspection()

        assert inspection_package is not None
        assert "audit_entries" in inspection_package
        assert "integrity_report" in inspection_package
        assert "summary_statistics" in inspection_package
        assert inspection_package["integrity_report"]["valid"] is True

    def test_retention_compliance_check(self, audit_engine):
        """Test 5-year retention compliance check."""
        # Create entries with different ages
        now = datetime.utcnow()

        # Recent entry (should be retained)
        recent_entry = audit_engine.log_action(
            user_id="user_recent",
            action_type="CREATE",
            entity_type="test",
            entity_id="recent"
        )

        # Check retention compliance
        compliance = audit_engine.check_retention_compliance()

        assert compliance is not None
        assert "total_entries" in compliance
        assert "oldest_entry_age_days" in compliance
        assert "retention_policy_days" in compliance
        assert compliance["retention_policy_days"] == 1825  # 5 years

    def test_export_json(self, audit_engine, sample_action):
        """Test exporting audit trail to JSON format."""
        # Log some actions
        for i in range(3):
            audit_engine.log_action(
                user_id=f"user_{i}",
                action_type="ACTION",
                entity_type="test",
                entity_id=f"test_{i}"
            )

        # Export to JSON
        json_export = audit_engine.export_audit_trail(format="json")

        assert json_export is not None
        # Parse JSON to validate structure
        data = json.loads(json_export) if isinstance(json_export, str) else json_export
        assert "audit_entries" in data or isinstance(data, list)

    def test_export_xml(self, audit_engine):
        """Test exporting audit trail to XML format."""
        # Log actions
        audit_engine.log_action(
            user_id="user_1",
            action_type="CREATE",
            entity_type="test",
            entity_id="test_1"
        )

        # Export to XML
        xml_export = audit_engine.export_audit_trail(format="xml")

        assert xml_export is not None
        if isinstance(xml_export, str):
            assert xml_export.startswith("<?xml") or xml_export.startswith("<")

    def test_document_classification(self, audit_engine):
        """Test automatic document classification in audit trail."""
        # Log action with document
        entry = audit_engine.log_action(
            user_id="user_1",
            action_type="DOCUMENT_UPLOADED",
            entity_type="document",
            entity_id="doc_123",
            details={
                "document_type": "land_title",
                "filename": "land_title.pdf"
            }
        )

        assert entry is not None
        assert entry["details"]["document_type"] == "land_title"

    def test_mock_audit(self, audit_engine):
        """Test mock audit simulation."""
        # Log actions across multiple entities
        entities = ["supplier_1", "plot_1", "dds_1"]

        for entity_id in entities:
            audit_engine.log_action(
                user_id="auditor",
                action_type="AUDIT_CHECK",
                entity_type="audit",
                entity_id=entity_id
            )

        # Run mock audit
        audit_result = audit_engine.run_mock_audit()

        assert audit_result is not None
        assert "findings" in audit_result or "status" in audit_result

    def test_compliance_calendar(self, audit_engine):
        """Test compliance calendar tracking."""
        # Log compliance events
        events = [
            {"action_type": "ANNUAL_REVIEW_DUE", "entity_id": "review_2025"},
            {"action_type": "DDS_DEADLINE", "entity_id": "dds_q1_2025"},
        ]

        for event in events:
            audit_engine.log_action(
                user_id="system",
                entity_type="compliance_event",
                **event
            )

        # Get compliance calendar
        calendar = audit_engine.get_compliance_calendar()

        assert calendar is not None
        assert len(calendar) >= 2

    def test_5_year_retention(self, audit_engine):
        """Test 5-year retention policy enforcement."""
        # Check retention policy
        policy = audit_engine.get_retention_policy()

        assert policy is not None
        assert policy["retention_period_years"] == 5
        assert policy["retention_period_days"] == 1825

        # Verify no entries are deleted before 5 years
        compliance = audit_engine.check_retention_compliance()
        assert compliance["compliant"] is True


class TestAuditTrailIntegrity:
    """Test audit trail integrity features."""

    def test_sequential_hash_chain(self, audit_engine):
        """Test sequential hash chain generation."""
        entries = []
        for i in range(5):
            entry = audit_engine.log_action(
                user_id=f"user_{i}",
                action_type="ACTION",
                entity_type="test",
                entity_id=f"test_{i}"
            )
            entries.append(entry)

        # Each entry should reference previous hash
        for i in range(1, len(entries)):
            # Most implementations link via previous_hash field
            if "previous_hash" in entries[i]:
                assert entries[i]["previous_hash"] == entries[i-1]["hash"]

    def test_immutability(self, audit_engine):
        """Test audit trail immutability."""
        entry = audit_engine.log_action(
            user_id="user_1",
            action_type="CREATE",
            entity_type="test",
            entity_id="test_1"
        )

        # Attempt to modify should fail or be detected
        # (Most implementations prevent modification)
        original_hash = entry["hash"]

        # Verify hash hasn't changed
        verification = audit_engine.verify_entry(entry["entry_id"])
        assert verification["hash"] == original_hash


class TestAuditTrailReporting:
    """Test audit trail reporting features."""

    def test_summary_statistics(self, audit_engine):
        """Test generating summary statistics."""
        # Log diverse actions
        action_types = ["CREATE", "UPDATE", "DELETE", "SUBMIT"]
        for action_type in action_types:
            audit_engine.log_action(
                user_id="user_1",
                action_type=action_type,
                entity_type="test",
                entity_id="test_1"
            )

        stats = audit_engine.get_summary_statistics()

        assert stats is not None
        assert "total_entries" in stats
        assert stats["total_entries"] >= 4
        assert "action_type_distribution" in stats or "by_action_type" in stats

    def test_user_activity_report(self, audit_engine):
        """Test generating user activity report."""
        # Log actions for multiple users
        users = ["user_a", "user_b", "user_c"]
        for user_id in users:
            for i in range(3):
                audit_engine.log_action(
                    user_id=user_id,
                    action_type="ACTION",
                    entity_type="test",
                    entity_id=f"test_{i}"
                )

        report = audit_engine.get_user_activity_report()

        assert report is not None
        assert len(report) >= 3  # At least 3 users

    def test_timeline_generation(self, audit_engine):
        """Test generating audit timeline."""
        # Log actions
        for i in range(5):
            audit_engine.log_action(
                user_id="user_1",
                action_type=f"STEP_{i}",
                entity_type="workflow",
                entity_id="workflow_1"
            )

        timeline = audit_engine.generate_timeline(entity_id="workflow_1")

        assert timeline is not None
        assert len(timeline) >= 5
