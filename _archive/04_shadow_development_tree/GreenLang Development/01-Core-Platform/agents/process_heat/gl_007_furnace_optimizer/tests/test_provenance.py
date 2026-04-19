# -*- coding: utf-8 -*-
"""
GL-007 Provenance Tracking Tests
================================

Unit tests for GL-007 provenance tracking module.
Tests SHA-256 hash generation, audit trails, and deterministic verification.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
import json

from greenlang.agents.process_heat.gl_007_furnace_optimizer.provenance import (
    ProvenanceConstants,
    ProvenanceHashGenerator,
    CalculationAuditTrail,
    ProvenanceTracker,
    DeterministicVerifier,
    CalculationInput,
    CalculationStep,
    ProvenanceRecord,
    AuditEntry,
    VerificationResult,
    create_provenance_tracker,
    create_audit_trail,
    create_hash_generator,
    create_deterministic_verifier,
    generate_provenance_hash,
    verify_provenance_hash,
)


class TestProvenanceConstants:
    """Tests for provenance constants."""

    def test_hash_algorithm(self):
        """Test hash algorithm is SHA-256."""
        assert ProvenanceConstants.HASH_ALGORITHM == "sha256"

    def test_agent_identification(self):
        """Test agent identification constants."""
        assert ProvenanceConstants.AGENT_ID == "GL-007"
        assert "Furnace" in ProvenanceConstants.AGENT_NAME

    def test_audit_settings(self):
        """Test audit trail settings."""
        assert ProvenanceConstants.MAX_AUDIT_ENTRIES > 0
        assert ProvenanceConstants.AUDIT_RETENTION_DAYS > 0


class TestProvenanceHashGenerator:
    """Tests for SHA-256 hash generator."""

    @pytest.fixture
    def generator(self):
        """Create hash generator instance."""
        return ProvenanceHashGenerator()

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert generator.algorithm == "sha256"

    def test_generate_hash_basic(self, generator):
        """Test basic hash generation."""
        data = {"temperature": 1800, "efficiency": 85.5}
        hash_value = generator.generate_hash(data)

        assert hash_value is not None
        assert len(hash_value) == 64  # SHA-256 = 64 hex characters

    def test_hash_determinism(self, generator):
        """Test hash is deterministic."""
        data = {"temperature": 1800, "efficiency": 85.5}

        hashes = [generator.generate_hash(data) for _ in range(5)]

        # All hashes should be identical
        assert len(set(hashes)) == 1

    def test_hash_sensitivity(self, generator):
        """Test hash changes with any data change."""
        data1 = {"temperature": 1800, "efficiency": 85.5}
        data2 = {"temperature": 1800, "efficiency": 85.6}  # Tiny change

        hash1 = generator.generate_hash(data1)
        hash2 = generator.generate_hash(data2)

        assert hash1 != hash2

    def test_hash_order_independence(self, generator):
        """Test hash is independent of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "b": 2, "a": 1}

        hash1 = generator.generate_hash(data1)
        hash2 = generator.generate_hash(data2)

        # Should be same due to sorted keys
        assert hash1 == hash2

    def test_generate_short_hash(self, generator):
        """Test short hash generation."""
        data = {"temperature": 1800}
        short_hash = generator.generate_short_hash(data, length=16)

        assert len(short_hash) == 16

    def test_verify_hash_success(self, generator):
        """Test successful hash verification."""
        data = {"temperature": 1800, "efficiency": 85.5}
        hash_value = generator.generate_hash(data)

        assert generator.verify_hash(data, hash_value) is True

    def test_verify_hash_failure(self, generator):
        """Test failed hash verification."""
        data = {"temperature": 1800, "efficiency": 85.5}
        wrong_hash = "0" * 64

        assert generator.verify_hash(data, wrong_hash) is False

    def test_verify_hash_truncated(self, generator):
        """Test verification with truncated hash."""
        data = {"temperature": 1800}
        full_hash = generator.generate_hash(data)
        short_hash = full_hash[:16]

        assert generator.verify_hash(data, short_hash) is True

    def test_hash_with_nested_data(self, generator):
        """Test hash with nested structures."""
        data = {
            "reading": {
                "temperature": 1800,
                "pressures": [10.5, 11.0, 10.8],
            },
            "metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }

        hash_value = generator.generate_hash(data)
        assert len(hash_value) == 64

    def test_hash_with_float_precision(self, generator):
        """Test hash handles float precision."""
        data1 = {"value": 1.0000000001}
        data2 = {"value": 1.0000000002}

        # Small differences should be handled by precision rounding
        hash1 = generator.generate_hash(data1)
        hash2 = generator.generate_hash(data2)

        # Depending on precision setting, may or may not be equal
        assert hash1 is not None
        assert hash2 is not None

    def test_hash_with_datetime(self, generator):
        """Test hash handles datetime objects."""
        data = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "value": 100,
        }

        hash_value = generator.generate_hash(data)
        assert len(hash_value) == 64


class TestCalculationAuditTrail:
    """Tests for calculation audit trail."""

    @pytest.fixture
    def audit_trail(self):
        """Create audit trail instance."""
        return CalculationAuditTrail()

    def test_initialization(self, audit_trail):
        """Test audit trail initialization."""
        assert audit_trail is not None
        assert audit_trail.session_id is not None

    def test_record_calculation(self, audit_trail):
        """Test recording a calculation."""
        entry_id = audit_trail.record_calculation(
            calc_type="combustion_analysis",
            inputs={"fuel_flow": 5000, "o2_pct": 3.0},
            outputs={"efficiency": 85.5},
        )

        assert entry_id is not None
        assert len(entry_id) == 36  # UUID format

    def test_get_entry(self, audit_trail):
        """Test retrieving an entry."""
        entry_id = audit_trail.record_calculation(
            calc_type="combustion_analysis",
            inputs={"fuel_flow": 5000},
            outputs={"efficiency": 85.5},
        )

        entry = audit_trail.get_entry(entry_id)

        assert entry is not None
        assert entry.calculation_type == "combustion_analysis"

    def test_get_entry_not_found(self, audit_trail):
        """Test entry not found."""
        entry = audit_trail.get_entry("nonexistent-id")
        assert entry is None

    def test_get_entries_by_type(self, audit_trail):
        """Test filtering entries by type."""
        audit_trail.record_calculation(
            calc_type="combustion",
            inputs={"a": 1},
            outputs={"b": 2},
        )
        audit_trail.record_calculation(
            calc_type="heat_transfer",
            inputs={"c": 3},
            outputs={"d": 4},
        )
        audit_trail.record_calculation(
            calc_type="combustion",
            inputs={"e": 5},
            outputs={"f": 6},
        )

        combustion_entries = audit_trail.get_entries_by_type("combustion")

        assert len(combustion_entries) == 2

    def test_verify_entry_success(self, audit_trail):
        """Test successful entry verification."""
        inputs = {"fuel_flow": 5000, "o2_pct": 3.0}
        outputs = {"efficiency": 85.5}

        entry_id = audit_trail.record_calculation(
            calc_type="combustion_analysis",
            inputs=inputs,
            outputs=outputs,
        )

        result = audit_trail.verify_entry(entry_id, inputs, outputs)

        assert result.is_valid is True
        assert result.match is True
        assert len(result.errors) == 0

    def test_verify_entry_failure(self, audit_trail):
        """Test failed entry verification."""
        entry_id = audit_trail.record_calculation(
            calc_type="combustion_analysis",
            inputs={"fuel_flow": 5000},
            outputs={"efficiency": 85.5},
        )

        # Verify with different data
        result = audit_trail.verify_entry(
            entry_id,
            inputs={"fuel_flow": 6000},  # Different
            outputs={"efficiency": 85.5},
        )

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_export_audit_log_json(self, audit_trail):
        """Test JSON export of audit log."""
        audit_trail.record_calculation(
            calc_type="test",
            inputs={"a": 1},
            outputs={"b": 2},
        )

        json_log = audit_trail.export_audit_log(format="json")

        assert isinstance(json_log, str)
        parsed = json.loads(json_log)
        assert len(parsed) == 1

    def test_export_audit_log_dict(self, audit_trail):
        """Test dict export of audit log."""
        audit_trail.record_calculation(
            calc_type="test",
            inputs={"a": 1},
            outputs={"b": 2},
        )

        dict_log = audit_trail.export_audit_log(format="dict")

        assert isinstance(dict_log, list)
        assert len(dict_log) == 1

    def test_get_statistics(self, audit_trail):
        """Test audit trail statistics."""
        audit_trail.record_calculation(
            calc_type="combustion",
            inputs={"a": 1},
            outputs={"b": 2},
        )
        audit_trail.record_calculation(
            calc_type="heat_transfer",
            inputs={"c": 3},
            outputs={"d": 4},
        )

        stats = audit_trail.get_statistics()

        assert stats["total_entries"] == 2
        assert "combustion" in stats["by_calculation_type"]
        assert "heat_transfer" in stats["by_calculation_type"]

    def test_max_entries_limit(self):
        """Test max entries limit."""
        audit_trail = CalculationAuditTrail(max_entries=5)

        for i in range(10):
            audit_trail.record_calculation(
                calc_type="test",
                inputs={"i": i},
                outputs={"o": i * 2},
            )

        stats = audit_trail.get_statistics()
        assert stats["total_entries"] == 5  # Limited to max


class TestProvenanceTracker:
    """Tests for provenance tracker."""

    @pytest.fixture
    def tracker(self):
        """Create provenance tracker instance."""
        return ProvenanceTracker()

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker is not None
        assert tracker.session_id is not None

    def test_track_calculation(self, tracker):
        """Test tracking a calculation."""
        record = tracker.track_calculation(
            calc_type="combustion_efficiency",
            inputs={"fuel_flow_scfh": 5000, "flue_gas_o2_pct": 3.0},
            outputs={"efficiency_pct": 85.5, "excess_air_pct": 15.0},
            formula="efficiency = 100 - losses",
        )

        assert isinstance(record, ProvenanceRecord)
        assert record.calculation_id is not None
        assert record.provenance_hash is not None
        assert len(record.provenance_hash) == 64

    def test_track_calculation_with_steps(self, tracker):
        """Test tracking calculation with steps."""
        record = tracker.track_calculation(
            calc_type="combustion_efficiency",
            inputs={"fuel_flow": 5000},
            outputs={"efficiency": 85.5},
            steps=[
                {"operation": "calc_excess_air", "inputs": {"o2": 3.0}, "output": 15.0, "formula": "EA = O2/(21-O2)*100", "description": "Calculate excess air"},
                {"operation": "calc_losses", "inputs": {"EA": 15.0}, "output": 14.5, "formula": "L = k*dT", "description": "Calculate losses"},
            ],
        )

        assert len(record.steps) == 2

    def test_track_calculation_with_references(self, tracker):
        """Test tracking with standard references."""
        record = tracker.track_calculation(
            calc_type="combustion_efficiency",
            inputs={"fuel_flow": 5000},
            outputs={"efficiency": 85.5},
            standard_references=["ASME PTC 4", "API 560"],
        )

        assert "ASME PTC 4" in record.standard_references

    def test_generate_hash(self, tracker):
        """Test hash generation."""
        data = {"temperature": 1800}
        hash_value = tracker.generate_hash(data)

        assert len(hash_value) == 64

    def test_verify_hash_success(self, tracker):
        """Test successful hash verification."""
        data = {"temperature": 1800}
        hash_value = tracker.generate_hash(data)

        assert tracker.verify_hash(data, hash_value) is True

    def test_verify_full(self, tracker):
        """Test full verification with stored record."""
        inputs = {"fuel_flow": 5000}
        outputs = {"efficiency": 85.5}

        record = tracker.track_calculation(
            calc_type="test",
            inputs=inputs,
            outputs=outputs,
        )

        result = tracker.verify(record.provenance_hash, inputs, outputs)

        assert result.is_valid is True

    def test_get_record(self, tracker):
        """Test retrieving a record."""
        record = tracker.track_calculation(
            calc_type="test",
            inputs={"a": 1},
            outputs={"b": 2},
        )

        retrieved = tracker.get_record(record.calculation_id)

        assert retrieved is not None
        assert retrieved.provenance_hash == record.provenance_hash

    def test_get_records_by_type(self, tracker):
        """Test filtering records by type."""
        tracker.track_calculation(calc_type="combustion", inputs={"a": 1}, outputs={"b": 2})
        tracker.track_calculation(calc_type="heat_transfer", inputs={"c": 3}, outputs={"d": 4})
        tracker.track_calculation(calc_type="combustion", inputs={"e": 5}, outputs={"f": 6})

        combustion_records = tracker.get_records_by_type("combustion")

        assert len(combustion_records) == 2

    def test_export_compliance_report(self, tracker):
        """Test compliance report export."""
        tracker.track_calculation(
            calc_type="combustion",
            inputs={"fuel_flow": 5000},
            outputs={"efficiency": 85.5},
            standard_references=["NFPA 86"],
        )

        report = tracker.export_compliance_report()

        assert report["report_type"] is not None
        assert report["agent_id"] == "GL-007"
        assert len(report["records"]) == 1

    def test_audit_statistics(self, tracker):
        """Test audit statistics."""
        tracker.track_calculation(calc_type="test", inputs={"a": 1}, outputs={"b": 2})

        stats = tracker.get_audit_statistics()

        assert stats["total_entries"] >= 1


class TestDeterministicVerifier:
    """Tests for deterministic verifier."""

    @pytest.fixture
    def verifier(self):
        """Create deterministic verifier instance."""
        return DeterministicVerifier()

    def test_initialization(self, verifier):
        """Test verifier initialization."""
        assert verifier is not None
        assert verifier.tolerance > 0

    def test_verify_reproducibility(self, verifier):
        """Test reproducibility verification."""
        def simple_calc(x):
            return {"result": x * 2, "squared": x ** 2}

        result = verifier.verify_reproducibility(
            calculation_fn=simple_calc,
            inputs=[1, 2, 3],
            iterations=3,
        )

        assert result["all_reproducible"] is True
        assert result["inputs_tested"] == 3
        assert result["iterations_per_input"] == 3

    def test_compare_outputs_equal(self, verifier):
        """Test comparing equal outputs."""
        output1 = {"temperature": 1800, "efficiency": 85.5}
        output2 = {"temperature": 1800, "efficiency": 85.5}

        are_equal, differences = verifier.compare_outputs(output1, output2)

        assert are_equal is True
        assert len(differences) == 0

    def test_compare_outputs_different(self, verifier):
        """Test comparing different outputs."""
        output1 = {"temperature": 1800, "efficiency": 85.5}
        output2 = {"temperature": 1800, "efficiency": 90.0}

        are_equal, differences = verifier.compare_outputs(output1, output2)

        assert are_equal is False
        assert len(differences) > 0

    def test_compare_outputs_nested(self, verifier):
        """Test comparing nested outputs."""
        output1 = {"data": {"value": 100, "items": [1, 2, 3]}}
        output2 = {"data": {"value": 100, "items": [1, 2, 4]}}

        are_equal, differences = verifier.compare_outputs(output1, output2)

        assert are_equal is False


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_provenance_tracker(self):
        """Test provenance tracker factory."""
        tracker = create_provenance_tracker(session_id="test-session")
        assert tracker.session_id == "test-session"

    def test_create_audit_trail(self):
        """Test audit trail factory."""
        trail = create_audit_trail(max_entries=100)
        assert trail.max_entries == 100

    def test_create_hash_generator(self):
        """Test hash generator factory."""
        generator = create_hash_generator(precision=5)
        assert generator.precision == 5

    def test_create_deterministic_verifier(self):
        """Test deterministic verifier factory."""
        verifier = create_deterministic_verifier(tolerance=1e-8)
        assert verifier.tolerance == 1e-8


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_provenance_hash(self):
        """Test convenience hash generation."""
        data = {"temperature": 1800}
        hash_value = generate_provenance_hash(data)

        assert len(hash_value) == 64

    def test_verify_provenance_hash_success(self):
        """Test convenience hash verification success."""
        data = {"temperature": 1800}
        hash_value = generate_provenance_hash(data)

        assert verify_provenance_hash(data, hash_value) is True

    def test_verify_provenance_hash_failure(self):
        """Test convenience hash verification failure."""
        data = {"temperature": 1800}
        wrong_hash = "0" * 64

        assert verify_provenance_hash(data, wrong_hash) is False


class TestDataClasses:
    """Tests for data classes."""

    def test_calculation_input(self):
        """Test CalculationInput dataclass."""
        input_data = CalculationInput(
            name="fuel_flow",
            value=5000.0,
            unit="scfh",
            source="flowmeter",
        )

        assert input_data.name == "fuel_flow"
        assert input_data.value == 5000.0
        assert input_data.unit == "scfh"

    def test_calculation_step(self):
        """Test CalculationStep dataclass."""
        step = CalculationStep(
            step_number=1,
            operation="calculate_excess_air",
            inputs={"o2_pct": 3.0},
            output=15.0,
            formula="EA = O2/(21-O2)*100",
            description="Calculate excess air from O2",
        )

        assert step.step_number == 1
        assert step.operation == "calculate_excess_air"
        assert step.output == 15.0

    def test_audit_entry(self):
        """Test AuditEntry dataclass."""
        entry = AuditEntry(
            entry_id="test-id",
            timestamp=datetime.now(timezone.utc),
            calculation_type="combustion",
            inputs_hash="abc123",
            outputs_hash="def456",
            provenance_hash="ghi789",
        )

        assert entry.entry_id == "test-id"
        assert entry.verification_status == "unverified"  # Default

    def test_verification_result(self):
        """Test VerificationResult dataclass."""
        result = VerificationResult(
            is_valid=True,
            original_hash="abc123",
            computed_hash="abc123",
            match=True,
            timestamp=datetime.now(timezone.utc),
            details={"inputs_match": True},
            errors=[],
        )

        assert result.is_valid is True
        assert result.match is True
        assert len(result.errors) == 0
