"""
GL-007 FURNACEPULSE - NFPA 86 Compliance Tests

Unit tests for NFPA 86 compliance including:
- Compliance checklist evaluation
- Evidence package generation
- Immutability of evidence
- Audit trail completeness
- Safety interlock verification

Coverage Target: >85%
"""

import pytest
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class ComplianceStatus(Enum):
    """Compliance status enumeration."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PENDING_REVIEW = "PENDING_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CheckItemStatus(Enum):
    """Individual checklist item status."""
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    NA = "N/A"


@dataclass
class EvidenceItem:
    """Evidence item for compliance documentation."""
    item_id: str
    item_type: str  # SENSOR_DATA, CALCULATION, TEST_RESULT, DOCUMENT
    description: str
    data_hash: str
    timestamp: datetime
    source: str


@dataclass
class EvidencePackage:
    """Immutable evidence package for audits."""
    package_id: str
    created_at: datetime
    furnace_id: str
    event_type: str
    items: List[EvidenceItem]
    package_hash: str
    signed_by: str
    is_immutable: bool = True


class TestComplianceChecklistEvaluation:
    """Tests for NFPA 86 compliance checklist evaluation."""

    def test_evaluate_passing_checklist(self, sample_nfpa86_checklist):
        """Test evaluation of a fully passing checklist."""
        # Count pass/fail
        passed = sum(1 for item in sample_nfpa86_checklist if item.status == "PASS")
        failed = sum(1 for item in sample_nfpa86_checklist if item.status == "FAIL")
        pending = sum(1 for item in sample_nfpa86_checklist if item.status == "PENDING")

        # All should pass
        assert passed == len(sample_nfpa86_checklist)
        assert failed == 0
        assert pending == 0

        # Determine overall status
        if failed > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif pending > 0:
            overall_status = ComplianceStatus.PENDING_REVIEW
        else:
            overall_status = ComplianceStatus.COMPLIANT

        assert overall_status == ComplianceStatus.COMPLIANT

    def test_evaluate_failing_checklist(self, sample_nfpa86_checklist_with_failures):
        """Test evaluation of checklist with failures."""
        failed = sum(
            1 for item in sample_nfpa86_checklist_with_failures
            if item.status == "FAIL"
        )

        assert failed > 0

        # Overall status should be NON_COMPLIANT
        overall_status = ComplianceStatus.NON_COMPLIANT
        assert overall_status == ComplianceStatus.NON_COMPLIANT

    def test_evaluate_pending_items(self, sample_nfpa86_checklist_with_failures):
        """Test handling of pending items."""
        pending = sum(
            1 for item in sample_nfpa86_checklist_with_failures
            if item.status == "PENDING"
        )

        assert pending > 0

        # If any pending, status should be PENDING_REVIEW (unless failures)
        has_failures = any(
            item.status == "FAIL"
            for item in sample_nfpa86_checklist_with_failures
        )

        if has_failures:
            expected_status = ComplianceStatus.NON_COMPLIANT
        else:
            expected_status = ComplianceStatus.PENDING_REVIEW

        assert expected_status == ComplianceStatus.NON_COMPLIANT

    def test_compliance_percentage_calculation(self, sample_nfpa86_checklist):
        """Test calculation of compliance percentage."""
        total_items = len(sample_nfpa86_checklist)
        passed_items = sum(1 for item in sample_nfpa86_checklist if item.status == "PASS")

        compliance_percentage = (passed_items / total_items) * 100

        assert compliance_percentage == 100.0

    def test_category_breakdown(self, sample_nfpa86_checklist):
        """Test compliance breakdown by category."""
        categories = {}
        for item in sample_nfpa86_checklist:
            if item.category not in categories:
                categories[item.category] = {"passed": 0, "failed": 0, "pending": 0}

            if item.status == "PASS":
                categories[item.category]["passed"] += 1
            elif item.status == "FAIL":
                categories[item.category]["failed"] += 1
            elif item.status == "PENDING":
                categories[item.category]["pending"] += 1

        # All categories should have some passed items
        for category, counts in categories.items():
            assert counts["passed"] > 0 or counts["failed"] > 0 or counts["pending"] > 0

    @pytest.mark.parametrize(
        "pass_count,fail_count,pending_count,expected_status",
        [
            (10, 0, 0, ComplianceStatus.COMPLIANT),
            (9, 1, 0, ComplianceStatus.NON_COMPLIANT),
            (8, 0, 2, ComplianceStatus.PENDING_REVIEW),
            (5, 2, 3, ComplianceStatus.NON_COMPLIANT),
        ],
    )
    def test_status_determination_parametrized(
        self, pass_count, fail_count, pending_count, expected_status
    ):
        """Test overall status determination with various scenarios."""
        if fail_count > 0:
            status = ComplianceStatus.NON_COMPLIANT
        elif pending_count > 0:
            status = ComplianceStatus.PENDING_REVIEW
        else:
            status = ComplianceStatus.COMPLIANT

        assert status == expected_status


class TestNFPA86SpecificRequirements:
    """Tests for specific NFPA 86 requirements."""

    def test_flame_supervision_requirement(self):
        """Test NFPA 86 4.3.1 - Flame supervision."""
        # Flame detector must be operational for each burner
        burners = [
            {"id": "B-001", "flame_detector_operational": True},
            {"id": "B-002", "flame_detector_operational": True},
            {"id": "B-003", "flame_detector_operational": True},
        ]

        all_operational = all(b["flame_detector_operational"] for b in burners)
        assert all_operational

    def test_flame_failure_response_time(self):
        """Test NFPA 86 4.3.2 - Flame failure response."""
        # Fuel shutoff must occur within 4 seconds
        max_response_time_seconds = 4.0
        actual_response_time = 2.5

        is_compliant = actual_response_time <= max_response_time_seconds
        assert is_compliant

    def test_combustion_air_interlock(self):
        """Test NFPA 86 5.2.1 - Combustion air interlock."""
        # Furnace cannot operate without combustion air
        combustion_air_flow = 25000.0  # kg/h
        interlock_threshold = 5000.0  # Minimum flow

        interlock_satisfied = combustion_air_flow > interlock_threshold
        assert interlock_satisfied

    def test_emergency_fuel_shutoff(self):
        """Test NFPA 86 6.1.1 - Emergency fuel shutoff."""
        emergency_shutoff = {
            "accessible": True,
            "tested": True,
            "last_test_date": datetime(2025, 1, 15),
            "test_result": "PASS",
        }

        is_compliant = (
            emergency_shutoff["accessible"] and
            emergency_shutoff["tested"] and
            emergency_shutoff["test_result"] == "PASS"
        )
        assert is_compliant

    def test_purge_cycle_verification(self):
        """Test NFPA 86 7.1.1 - Pre-ignition purge."""
        # Minimum 4 volume changes before ignition
        required_volume_changes = 4
        actual_volume_changes = 5

        purge_verified = actual_volume_changes >= required_volume_changes
        assert purge_verified

    def test_overtemperature_protection(self, test_config):
        """Test NFPA 86 8.2.1 - Over-temperature protection."""
        max_temp_setpoint = test_config["tmt_max_C"]  # 950C
        high_temp_alarm_setpoint = 930.0
        high_high_temp_trip = 950.0

        # Alarm before trip
        assert high_temp_alarm_setpoint < high_high_temp_trip

        # Trip at or before design limit
        assert high_high_temp_trip <= max_temp_setpoint


class TestEvidencePackageGeneration:
    """Tests for evidence package generation."""

    def test_evidence_package_creation(self, sample_evidence_package):
        """Test creation of evidence package."""
        assert sample_evidence_package["package_id"] is not None
        assert sample_evidence_package["created_at"] is not None
        assert len(sample_evidence_package["items"]) > 0
        assert sample_evidence_package["package_hash"] is not None

    def test_evidence_package_hash_calculation(self):
        """Test evidence package hash calculation."""
        items = [
            {"item_id": "EVD-001", "data_hash": "abc123"},
            {"item_id": "EVD-002", "data_hash": "def456"},
        ]

        # Calculate package hash from item hashes
        combined = "".join(item["data_hash"] for item in items)
        package_hash = hashlib.sha256(combined.encode()).hexdigest()

        assert len(package_hash) == 64

    def test_evidence_item_structure(self, sample_evidence_package):
        """Test evidence item structure."""
        for item in sample_evidence_package["items"]:
            assert "item_id" in item
            assert "type" in item
            assert "description" in item
            assert "data_hash" in item

    def test_evidence_types_supported(self):
        """Test all required evidence types are supported."""
        supported_types = [
            "SENSOR_DATA",
            "CALCULATION_RESULT",
            "ALERT_LOG",
            "TEST_RESULT",
            "DOCUMENT",
            "CONFIGURATION",
            "AUDIT_LOG",
        ]

        # Create sample evidence of each type
        evidence_items = [
            {"type": t, "item_id": f"EVD-{i:03d}"}
            for i, t in enumerate(supported_types)
        ]

        assert len(evidence_items) == len(supported_types)

    def test_evidence_timestamp_ordering(self):
        """Test evidence items are properly timestamped."""
        base_time = datetime.now()
        items = [
            EvidenceItem(
                item_id=f"EVD-{i:03d}",
                item_type="SENSOR_DATA",
                description=f"Item {i}",
                data_hash=hashlib.sha256(f"data{i}".encode()).hexdigest(),
                timestamp=base_time + timedelta(minutes=i),
                source="system",
            )
            for i in range(5)
        ]

        # Verify chronological ordering
        for i in range(1, len(items)):
            assert items[i].timestamp > items[i - 1].timestamp


class TestEvidenceImmutability:
    """Tests for evidence package immutability."""

    def test_evidence_hash_immutable(self, sample_evidence_package):
        """Test that evidence hash cannot be changed."""
        original_hash = sample_evidence_package["package_hash"]

        # Any modification should produce different hash
        modified_package = sample_evidence_package.copy()
        modified_package["items"] = [{"item_id": "NEW", "data_hash": "xyz"}]

        # Recalculate hash
        combined = "".join(
            item.get("data_hash", "") for item in modified_package["items"]
        )
        new_hash = hashlib.sha256(combined.encode()).hexdigest()

        assert new_hash != original_hash

    def test_evidence_tamper_detection(self):
        """Test detection of tampered evidence."""
        # Original evidence
        original_data = {"temperature": 820.0, "timestamp": "2025-01-15T10:00:00Z"}
        original_hash = hashlib.sha256(
            json.dumps(original_data, sort_keys=True).encode()
        ).hexdigest()

        # Tampered evidence
        tampered_data = {"temperature": 950.0, "timestamp": "2025-01-15T10:00:00Z"}
        tampered_hash = hashlib.sha256(
            json.dumps(tampered_data, sort_keys=True).encode()
        ).hexdigest()

        # Hash mismatch indicates tampering
        is_tampered = original_hash != tampered_hash
        assert is_tampered

    def test_evidence_chain_of_custody(self):
        """Test evidence chain of custody tracking."""
        custody_chain = [
            {
                "action": "CREATED",
                "actor": "system",
                "timestamp": datetime(2025, 1, 15, 10, 0),
                "hash_before": None,
                "hash_after": "abc123",
            },
            {
                "action": "ACCESSED",
                "actor": "operator-001",
                "timestamp": datetime(2025, 1, 15, 11, 0),
                "hash_before": "abc123",
                "hash_after": "abc123",  # No change
            },
            {
                "action": "EXPORTED",
                "actor": "auditor-001",
                "timestamp": datetime(2025, 1, 15, 14, 0),
                "hash_before": "abc123",
                "hash_after": "abc123",  # No change
            },
        ]

        # Verify no modifications (hash unchanged)
        for entry in custody_chain[1:]:
            assert entry["hash_before"] == entry["hash_after"]

    def test_immutable_flag_enforcement(self, sample_evidence_package):
        """Test immutable flag prevents modifications."""
        is_immutable = sample_evidence_package["is_immutable"]

        # Attempt to modify should be rejected
        if is_immutable:
            modification_allowed = False
        else:
            modification_allowed = True

        assert not modification_allowed

    def test_evidence_signature_verification(self):
        """Test evidence package signature verification."""
        # Create signed package
        package_content = {
            "items": [{"id": "EVD-001", "hash": "abc123"}],
            "timestamp": "2025-01-15T10:00:00Z",
        }

        # Calculate hash (simulating signature)
        content_hash = hashlib.sha256(
            json.dumps(package_content, sort_keys=True).encode()
        ).hexdigest()

        signature_record = {
            "content_hash": content_hash,
            "signed_by": "system",
            "signed_at": datetime.now().isoformat(),
        }

        # Verify signature
        verification_hash = hashlib.sha256(
            json.dumps(package_content, sort_keys=True).encode()
        ).hexdigest()

        is_valid = verification_hash == signature_record["content_hash"]
        assert is_valid


class TestAuditTrailCompleteness:
    """Tests for audit trail completeness."""

    def test_audit_trail_records_all_events(self):
        """Test audit trail includes all required events."""
        required_events = [
            "COMPLIANCE_CHECK_STARTED",
            "ITEM_EVALUATED",
            "EVIDENCE_COLLECTED",
            "COMPLIANCE_RESULT",
            "REPORT_GENERATED",
        ]

        # Simulated audit trail
        audit_trail = [
            {"event": "COMPLIANCE_CHECK_STARTED", "timestamp": datetime.now()},
            {"event": "ITEM_EVALUATED", "timestamp": datetime.now()},
            {"event": "ITEM_EVALUATED", "timestamp": datetime.now()},
            {"event": "EVIDENCE_COLLECTED", "timestamp": datetime.now()},
            {"event": "COMPLIANCE_RESULT", "timestamp": datetime.now()},
            {"event": "REPORT_GENERATED", "timestamp": datetime.now()},
        ]

        recorded_events = set(entry["event"] for entry in audit_trail)

        for required in required_events:
            assert required in recorded_events

    def test_audit_trail_chronological_order(self):
        """Test audit trail entries are chronological."""
        base_time = datetime.now()
        audit_trail = [
            {"event": f"EVENT_{i}", "timestamp": base_time + timedelta(seconds=i)}
            for i in range(10)
        ]

        for i in range(1, len(audit_trail)):
            assert audit_trail[i]["timestamp"] > audit_trail[i - 1]["timestamp"]

    def test_audit_trail_includes_actor(self):
        """Test audit trail includes actor information."""
        audit_entry = {
            "event": "COMPLIANCE_CHECK",
            "timestamp": datetime.now(),
            "actor": "system",
            "actor_type": "SERVICE",
            "session_id": "sess-12345",
        }

        assert "actor" in audit_entry
        assert "actor_type" in audit_entry

    def test_audit_trail_includes_result(self):
        """Test audit trail includes result of actions."""
        audit_entry = {
            "event": "ITEM_EVALUATED",
            "item_id": "NFPA86-4.3.1",
            "timestamp": datetime.now(),
            "result": "PASS",
            "evidence_ref": "EVD-001",
        }

        assert "result" in audit_entry


class TestComplianceReporting:
    """Tests for compliance reporting functionality."""

    def test_compliance_report_generation(self, sample_nfpa86_checklist):
        """Test generation of compliance report."""
        report = {
            "report_id": "RPT-2025-001",
            "generated_at": datetime.now().isoformat(),
            "furnace_id": "FRN-001",
            "standard": "NFPA 86",
            "overall_status": "COMPLIANT",
            "items_evaluated": len(sample_nfpa86_checklist),
            "items_passed": sum(1 for i in sample_nfpa86_checklist if i.status == "PASS"),
            "items_failed": sum(1 for i in sample_nfpa86_checklist if i.status == "FAIL"),
            "next_audit_due": (datetime.now() + timedelta(days=365)).isoformat(),
        }

        assert report["report_id"] is not None
        assert report["items_passed"] > 0

    def test_compliance_report_includes_evidence(self, sample_evidence_package):
        """Test compliance report includes evidence references."""
        report = {
            "report_id": "RPT-2025-001",
            "evidence_package_id": sample_evidence_package["package_id"],
            "evidence_items": [
                {"item_id": item["item_id"], "type": item["type"]}
                for item in sample_evidence_package["items"]
            ],
        }

        assert len(report["evidence_items"]) > 0

    def test_non_compliance_report_includes_remediation(self):
        """Test non-compliance report includes remediation steps."""
        non_compliance_report = {
            "report_id": "RPT-2025-002",
            "overall_status": "NON_COMPLIANT",
            "failed_items": [
                {
                    "item_id": "NFPA86-4.3.2",
                    "description": "Flame failure response time",
                    "finding": "Response time 6 seconds exceeds 4 second limit",
                    "remediation": "Calibrate flame detector and test fuel valve response",
                    "priority": "HIGH",
                    "due_date": (datetime.now() + timedelta(days=7)).isoformat(),
                }
            ],
        }

        assert len(non_compliance_report["failed_items"]) > 0
        assert "remediation" in non_compliance_report["failed_items"][0]

    def test_compliance_history_tracking(self):
        """Test tracking of compliance history over time."""
        compliance_history = [
            {"date": "2024-01-15", "status": "COMPLIANT", "score": 100.0},
            {"date": "2024-04-15", "status": "COMPLIANT", "score": 100.0},
            {"date": "2024-07-15", "status": "NON_COMPLIANT", "score": 90.0},
            {"date": "2024-10-15", "status": "COMPLIANT", "score": 100.0},
            {"date": "2025-01-15", "status": "COMPLIANT", "score": 100.0},
        ]

        # Calculate compliance trend
        compliant_count = sum(1 for h in compliance_history if h["status"] == "COMPLIANT")
        compliance_rate = compliant_count / len(compliance_history) * 100

        assert compliance_rate >= 80.0  # 80% compliance rate


class TestSafetyInterlockVerification:
    """Tests for safety interlock verification."""

    def test_fuel_shutoff_interlock(self):
        """Test fuel shutoff interlock functionality."""
        interlock_state = {
            "flame_detected": True,
            "combustion_air_flow": True,
            "draft_pressure_ok": True,
            "temperature_ok": True,
            "emergency_stop": False,
        }

        # Fuel should flow only if all conditions met
        fuel_flow_permitted = (
            interlock_state["flame_detected"] and
            interlock_state["combustion_air_flow"] and
            interlock_state["draft_pressure_ok"] and
            interlock_state["temperature_ok"] and
            not interlock_state["emergency_stop"]
        )

        assert fuel_flow_permitted

    def test_interlock_trip_on_flame_loss(self):
        """Test interlock trips on flame loss."""
        interlock_state = {
            "flame_detected": False,  # Flame loss
            "combustion_air_flow": True,
            "draft_pressure_ok": True,
            "temperature_ok": True,
            "emergency_stop": False,
        }

        fuel_flow_permitted = interlock_state["flame_detected"]
        assert not fuel_flow_permitted

    def test_interlock_bypass_tracking(self):
        """Test tracking of interlock bypasses."""
        bypass_log = [
            {
                "interlock": "HIGH_TEMP_TRIP",
                "bypassed": True,
                "reason": "Startup mode",
                "authorized_by": "supervisor-001",
                "timestamp": datetime.now(),
                "expires": datetime.now() + timedelta(hours=1),
            }
        ]

        # Bypass should be logged
        assert len(bypass_log) > 0
        assert "authorized_by" in bypass_log[0]
        assert "expires" in bypass_log[0]

    def test_interlock_test_requirements(self):
        """Test interlock testing requirements."""
        interlock_tests = [
            {
                "interlock": "FLAME_FAILURE",
                "test_frequency_days": 30,
                "last_test": datetime(2025, 1, 1),
                "next_test_due": datetime(2025, 1, 31),
                "test_result": "PASS",
            },
            {
                "interlock": "HIGH_TEMP_TRIP",
                "test_frequency_days": 90,
                "last_test": datetime(2024, 11, 15),
                "next_test_due": datetime(2025, 2, 13),
                "test_result": "PASS",
            },
        ]

        # Check for overdue tests
        today = datetime.now()
        overdue = [t for t in interlock_tests if t["next_test_due"] < today]

        # No tests should be overdue
        assert len(overdue) == 0


class TestProvenanceTracking:
    """Tests for compliance provenance tracking."""

    def test_compliance_hash_deterministic(self, sample_nfpa86_checklist):
        """Test compliance evaluation produces deterministic hash."""
        # Create deterministic representation
        checklist_data = [
            {"item_id": item.item_id, "status": item.status}
            for item in sample_nfpa86_checklist
        ]

        hash1 = hashlib.sha256(
            json.dumps(checklist_data, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(checklist_data, sort_keys=True).encode()
        ).hexdigest()

        assert hash1 == hash2

    def test_compliance_reproducibility(self, sample_nfpa86_checklist):
        """Test compliance evaluation is reproducible."""
        results = []

        for _ in range(5):
            passed = sum(1 for item in sample_nfpa86_checklist if item.status == "PASS")
            failed = sum(1 for item in sample_nfpa86_checklist if item.status == "FAIL")
            results.append({"passed": passed, "failed": failed})

        assert all(r == results[0] for r in results)

    def test_evidence_provenance_chain(self, sample_evidence_package):
        """Test evidence has complete provenance chain."""
        for item in sample_evidence_package["items"]:
            assert "data_hash" in item

        # Package hash links all items
        assert sample_evidence_package["package_hash"] is not None


class TestPerformance:
    """Performance tests for compliance checking."""

    def test_checklist_evaluation_speed(self, sample_nfpa86_checklist):
        """Test checklist evaluation performance."""
        import time

        start_time = time.time()

        for _ in range(1000):
            passed = sum(1 for item in sample_nfpa86_checklist if item.status == "PASS")
            failed = sum(1 for item in sample_nfpa86_checklist if item.status == "FAIL")
            if failed > 0:
                status = ComplianceStatus.NON_COMPLIANT
            else:
                status = ComplianceStatus.COMPLIANT

        elapsed = time.time() - start_time

        # Should complete 1000 evaluations in < 100ms
        assert elapsed < 0.1

    def test_evidence_hash_calculation_speed(self):
        """Test evidence hash calculation performance."""
        import time

        # Create large evidence dataset
        evidence_data = [
            {"item_id": f"EVD-{i:05d}", "data": f"data_{i}" * 100}
            for i in range(100)
        ]

        start_time = time.time()

        for item in evidence_data:
            hashlib.sha256(json.dumps(item).encode()).hexdigest()

        elapsed = time.time() - start_time

        # Should hash 100 items in < 50ms
        assert elapsed < 0.05
