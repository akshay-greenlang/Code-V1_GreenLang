"""
GL-001 ThermalCommand: Data Quality Tests

Comprehensive tests for data quality validation, lineage tracking,
and truth label handling.

Test Coverage:
- Time synchronization validation
- Completeness checking
- Validity validation
- Data lineage tracking
- Truth label management
- Quality scoring
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

import sys
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/tests/', 1)[0])

from data_contracts.data_quality import (
    # Enums
    ValidationSeverity,
    TimeSource,
    DataLineageType,
    TruthLabelStatus,
    # Validators
    TimeSyncValidator,
    CompletenessValidator,
    ValidityValidator,
    # Lineage
    LineageNode,
    LineageTracker,
    # Truth labels
    TruthLabel,
    TruthLabelHandler,
    # Scoring
    DataQualityScorer,
    # Manager
    DataQualityManager,
    # Functions
    get_quality_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_time() -> datetime:
    """Provide base time for tests."""
    return datetime.now(timezone.utc)


@pytest.fixture
def time_validator(base_time) -> TimeSyncValidator:
    """Create time sync validator."""
    return TimeSyncValidator(reference_time=base_time)


@pytest.fixture
def completeness_validator() -> CompletenessValidator:
    """Create completeness validator with test fields."""
    required_fields = {
        "TestSchema": ["facility_id", "timestamp", "value"],
        "ProcessSensorData": ["facility_id", "timestamp"],
    }
    return CompletenessValidator(required_fields)


@pytest.fixture
def validity_validator() -> ValidityValidator:
    """Create validity validator."""
    return ValidityValidator()


@pytest.fixture
def lineage_tracker() -> LineageTracker:
    """Create lineage tracker."""
    return LineageTracker()


@pytest.fixture
def truth_label_handler() -> TruthLabelHandler:
    """Create truth label handler."""
    return TruthLabelHandler()


# =============================================================================
# TimeSyncValidator Tests
# =============================================================================

class TestTimeSyncValidator:
    """Tests for TimeSyncValidator class."""

    def test_validate_timestamp_valid(self, time_validator, base_time):
        """Test validation of valid timestamp."""
        timestamp = base_time - timedelta(seconds=5)
        is_valid, issues = time_validator.validate_timestamp(timestamp)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_timestamp_future(self, time_validator, base_time):
        """Test validation of future timestamp."""
        timestamp = base_time + timedelta(hours=1)
        is_valid, issues = time_validator.validate_timestamp(timestamp)
        assert is_valid is False
        assert any("future" in issue.lower() for issue in issues)

    def test_validate_timestamp_stale(self, time_validator, base_time):
        """Test validation of stale timestamp."""
        timestamp = base_time - timedelta(minutes=5)
        is_valid, issues = time_validator.validate_timestamp(
            timestamp,
            data_category="standard"  # 30 second threshold
        )
        assert is_valid is False
        assert any("stale" in issue.lower() for issue in issues)

    def test_validate_timestamp_stale_slow_category(self, time_validator, base_time):
        """Test stale validation for slow-changing data."""
        timestamp = base_time - timedelta(minutes=4)
        is_valid, issues = time_validator.validate_timestamp(
            timestamp,
            data_category="slow"  # 300 second threshold
        )
        assert is_valid is True

    def test_validate_timestamp_very_old(self, time_validator, base_time):
        """Test validation of very old data."""
        timestamp = base_time - timedelta(days=2)
        is_valid, issues = time_validator.validate_timestamp(timestamp)
        assert is_valid is False
        assert any("very old" in issue.lower() for issue in issues)

    def test_check_time_drift_no_drift(self, time_validator):
        """Test time drift check with no drift."""
        base = datetime.now(timezone.utc)
        timestamps = [base + timedelta(seconds=i) for i in range(10)]
        max_drift, warnings = time_validator.check_time_drift(
            timestamps,
            expected_interval_ms=1000
        )
        assert max_drift < 100  # Very small drift

    def test_check_time_drift_with_drift(self, time_validator):
        """Test time drift check with significant drift."""
        base = datetime.now(timezone.utc)
        timestamps = [
            base,
            base + timedelta(seconds=1),
            base + timedelta(seconds=3),  # Drift here
            base + timedelta(seconds=4),
        ]
        max_drift, warnings = time_validator.check_time_drift(
            timestamps,
            expected_interval_ms=1000
        )
        assert max_drift > 500  # Significant drift
        assert len(warnings) > 0

    def test_different_time_sources(self, base_time):
        """Test different time source configurations."""
        ntp_validator = TimeSyncValidator(
            reference_time=base_time,
            time_source=TimeSource.NTP
        )
        assert ntp_validator.max_drift == timedelta(milliseconds=100)

        ptp_validator = TimeSyncValidator(
            reference_time=base_time,
            time_source=TimeSource.PTP
        )
        assert ptp_validator.max_drift == timedelta(milliseconds=1)


# =============================================================================
# CompletenessValidator Tests
# =============================================================================

class TestCompletenessValidator:
    """Tests for CompletenessValidator class."""

    def test_check_record_completeness_complete(self, completeness_validator):
        """Test completeness check on complete record."""
        record = {
            "facility_id": "PLANT-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": 42.5,
        }
        score, missing = completeness_validator.check_record_completeness(
            record, "TestSchema"
        )
        assert score == 1.0
        assert len(missing) == 0

    def test_check_record_completeness_incomplete(self, completeness_validator):
        """Test completeness check on incomplete record."""
        record = {
            "facility_id": "PLANT-001",
            # Missing timestamp and value
        }
        score, missing = completeness_validator.check_record_completeness(
            record, "TestSchema"
        )
        assert score < 1.0
        assert "timestamp" in missing
        assert "value" in missing

    def test_check_record_completeness_empty_string(self, completeness_validator):
        """Test that empty strings are treated as missing."""
        record = {
            "facility_id": "",  # Empty string
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": 42.5,
        }
        score, missing = completeness_validator.check_record_completeness(
            record, "TestSchema"
        )
        assert "facility_id" in missing

    def test_check_batch_completeness(self, completeness_validator):
        """Test batch completeness check."""
        records = [
            {"facility_id": "P1", "timestamp": "2024-01-01T00:00:00Z", "value": 1.0},
            {"facility_id": "P1", "timestamp": "2024-01-01T00:01:00Z"},  # Missing value
            {"facility_id": "P1", "value": 3.0},  # Missing timestamp
            {"facility_id": "P1", "timestamp": "2024-01-01T00:03:00Z", "value": 4.0},
        ]

        result = completeness_validator.check_batch_completeness(
            records, "TestSchema"
        )
        assert result["total_records"] == 4
        assert result["complete_records"] == 2
        assert result["completeness_rate"] == 0.5

    def test_check_batch_completeness_empty(self, completeness_validator):
        """Test batch completeness with empty batch."""
        result = completeness_validator.check_batch_completeness([], "TestSchema")
        assert result["total_records"] == 0
        assert result["completeness_rate"] == 0.0


# =============================================================================
# ValidityValidator Tests
# =============================================================================

class TestValidityValidator:
    """Tests for ValidityValidator class."""

    def test_validate_range_valid(self, validity_validator):
        """Test range validation - valid."""
        is_valid, error = validity_validator.validate_range(
            50.0, min_val=0.0, max_val=100.0
        )
        assert is_valid is True
        assert error is None

    def test_validate_range_below_min(self, validity_validator):
        """Test range validation - below minimum."""
        is_valid, error = validity_validator.validate_range(
            -5.0, min_val=0.0, max_val=100.0
        )
        assert is_valid is False
        assert "below" in error.lower()

    def test_validate_range_above_max(self, validity_validator):
        """Test range validation - above maximum."""
        is_valid, error = validity_validator.validate_range(
            150.0, min_val=0.0, max_val=100.0
        )
        assert is_valid is False
        assert "above" in error.lower()

    def test_validate_rate_of_change_valid(self, validity_validator):
        """Test rate of change validation - valid."""
        is_valid, error = validity_validator.validate_rate_of_change(
            current=100.0,
            previous=95.0,
            max_change_per_second=10.0,
            interval_seconds=1.0,
        )
        assert is_valid is True

    def test_validate_rate_of_change_exceeded(self, validity_validator):
        """Test rate of change validation - exceeded."""
        is_valid, error = validity_validator.validate_rate_of_change(
            current=100.0,
            previous=50.0,
            max_change_per_second=10.0,
            interval_seconds=1.0,
        )
        assert is_valid is False
        assert "exceeds" in error.lower()

    def test_validate_physical_consistency(self, validity_validator):
        """Test physical consistency validation."""
        data = {
            "inlet_temp": 100.0,
            "outlet_temp": 80.0,  # Should be less than inlet for cooling
        }
        rules = [
            {
                "type": "greater_than",
                "field1": "inlet_temp",
                "field2": "outlet_temp",
            }
        ]
        errors = validity_validator.validate_physical_consistency(data, rules)
        assert len(errors) == 0

        # Invalid case
        data["outlet_temp"] = 120.0  # Higher than inlet
        errors = validity_validator.validate_physical_consistency(data, rules)
        assert len(errors) > 0

    def test_register_custom_rule(self, validity_validator):
        """Test registering custom validation rule."""
        def check_positive(value):
            if value > 0:
                return True, None
            return False, "Value must be positive"

        validity_validator.register_rule("positive_check", check_positive)
        assert "positive_check" in validity_validator.custom_rules


# =============================================================================
# LineageTracker Tests
# =============================================================================

class TestLineageTracker:
    """Tests for LineageTracker class."""

    def test_compute_hash_deterministic(self, lineage_tracker):
        """Test hash computation is deterministic."""
        data = {"value": 42.5, "tag": "test"}
        hash1 = lineage_tracker.compute_hash(data)
        hash2 = lineage_tracker.compute_hash(data)
        assert hash1 == hash2

    def test_compute_hash_different_data(self, lineage_tracker):
        """Test different data produces different hash."""
        hash1 = lineage_tracker.compute_hash({"value": 42.5})
        hash2 = lineage_tracker.compute_hash({"value": 42.6})
        assert hash1 != hash2

    def test_create_source_record(self, lineage_tracker):
        """Test creating source record."""
        data = {"facility_id": "P1", "value": 42.5}
        record_id = lineage_tracker.create_source_record(
            data,
            source_system="SCADA",
        )
        assert record_id is not None
        assert len(record_id) == 36  # UUID format

    def test_create_derived_record(self, lineage_tracker):
        """Test creating derived record."""
        # Create source
        source_data = {"value": 42.5}
        source_id = lineage_tracker.create_source_record(
            source_data,
            source_system="SCADA",
        )

        # Create derived
        derived_data = {"value": 84.0}  # e.g., doubled
        derived_id = lineage_tracker.create_derived_record(
            derived_data,
            parent_ids=[source_id],
            transformation="multiply_by_2",
        )

        assert derived_id != source_id

    def test_verify_integrity_valid(self, lineage_tracker):
        """Test integrity verification - valid."""
        data = {"value": 42.5}
        record_id = lineage_tracker.create_source_record(
            data,
            source_system="SCADA",
        )
        assert lineage_tracker.verify_integrity(record_id, data) is True

    def test_verify_integrity_tampered(self, lineage_tracker):
        """Test integrity verification - tampered data."""
        data = {"value": 42.5}
        record_id = lineage_tracker.create_source_record(
            data,
            source_system="SCADA",
        )

        tampered_data = {"value": 99.9}
        assert lineage_tracker.verify_integrity(record_id, tampered_data) is False

    def test_get_lineage_chain(self, lineage_tracker):
        """Test getting lineage chain."""
        # Create chain: source -> transform1 -> transform2
        source_id = lineage_tracker.create_source_record(
            {"value": 1.0},
            source_system="SCADA",
        )
        t1_id = lineage_tracker.create_derived_record(
            {"value": 2.0},
            parent_ids=[source_id],
            transformation="double",
        )
        t2_id = lineage_tracker.create_derived_record(
            {"value": 4.0},
            parent_ids=[t1_id],
            transformation="double",
        )

        chain = lineage_tracker.get_lineage_chain(t2_id)
        assert len(chain) == 3

    def test_export_lineage(self, lineage_tracker):
        """Test exporting lineage information."""
        source_id = lineage_tracker.create_source_record(
            {"value": 42.5},
            source_system="SCADA",
        )
        export = lineage_tracker.export_lineage(source_id)
        assert "record_id" in export
        assert "lineage" in export
        assert len(export["lineage"]) == 1


# =============================================================================
# TruthLabelHandler Tests
# =============================================================================

class TestTruthLabelHandler:
    """Tests for TruthLabelHandler class."""

    def test_add_label(self, truth_label_handler):
        """Test adding a truth label."""
        label = truth_label_handler.add_label(
            record_id="record-001",
            label_type="anomaly",
            label_value=True,
            labeled_by="analyst-1",
            confidence=0.9,
        )
        assert label.record_id == "record-001"
        assert label.status == TruthLabelStatus.LABELED

    def test_verify_label_correct(self, truth_label_handler):
        """Test verifying label as correct."""
        truth_label_handler.add_label(
            record_id="record-001",
            label_type="anomaly",
            label_value=True,
            labeled_by="analyst-1",
        )

        label = truth_label_handler.verify_label(
            record_id="record-001",
            label_type="anomaly",
            verified_by="senior-analyst",
            is_correct=True,
        )
        assert label.status == TruthLabelStatus.VERIFIED

    def test_verify_label_disputed(self, truth_label_handler):
        """Test verifying label as incorrect (disputed)."""
        truth_label_handler.add_label(
            record_id="record-001",
            label_type="anomaly",
            label_value=True,
            labeled_by="analyst-1",
        )

        label = truth_label_handler.verify_label(
            record_id="record-001",
            label_type="anomaly",
            verified_by="senior-analyst",
            is_correct=False,
        )
        assert label.status == TruthLabelStatus.DISPUTED

    def test_get_consensus_label(self, truth_label_handler):
        """Test getting consensus label from multiple annotators."""
        # Add multiple labels
        for i in range(5):
            truth_label_handler.add_label(
                record_id="record-001",
                label_type="equipment_state",
                label_value="normal" if i < 4 else "faulty",
                labeled_by=f"analyst-{i}",
            )

        result = truth_label_handler.get_consensus_label(
            record_id="record-001",
            label_type="equipment_state",
            min_annotators=3,
            agreement_threshold=0.6,
        )
        assert result is not None
        value, agreement = result
        assert value == "normal"
        assert agreement >= 0.8

    def test_get_consensus_no_agreement(self, truth_label_handler):
        """Test consensus when there's no agreement."""
        # Add split labels
        truth_label_handler.add_label(
            record_id="record-001",
            label_type="state",
            label_value="A",
            labeled_by="analyst-1",
        )
        truth_label_handler.add_label(
            record_id="record-001",
            label_type="state",
            label_value="B",
            labeled_by="analyst-2",
        )

        result = truth_label_handler.get_consensus_label(
            record_id="record-001",
            label_type="state",
            min_annotators=2,
            agreement_threshold=0.8,
        )
        assert result is None  # No consensus

    def test_get_labels_for_training(self, truth_label_handler):
        """Test getting labels for ML training."""
        # Add verified label
        truth_label_handler.add_label(
            record_id="record-001",
            label_type="anomaly",
            label_value=True,
            labeled_by="analyst-1",
        )
        truth_label_handler.verify_label(
            record_id="record-001",
            label_type="anomaly",
            verified_by="senior",
            is_correct=True,
        )

        # Add unverified label
        truth_label_handler.add_label(
            record_id="record-002",
            label_type="anomaly",
            label_value=False,
            labeled_by="analyst-2",
        )

        # Get only verified
        labels = truth_label_handler.get_labels_for_training(
            label_type="anomaly",
            min_status=TruthLabelStatus.VERIFIED,
        )
        assert len(labels) == 1
        assert labels[0]["record_id"] == "record-001"


# =============================================================================
# DataQualityScorer Tests
# =============================================================================

class TestDataQualityScorer:
    """Tests for DataQualityScorer class."""

    @pytest.fixture
    def scorer(self, completeness_validator, validity_validator, time_validator):
        """Create quality scorer."""
        return DataQualityScorer(
            completeness_validator,
            validity_validator,
            time_validator,
        )

    def test_score_record_high_quality(self, scorer, base_time):
        """Test scoring a high quality record."""
        record = {
            "facility_id": "PLANT-001",
            "timestamp": base_time.isoformat(),
        }
        result = scorer.score_record(
            record,
            "ProcessSensorData",
            base_time,
        )
        assert result["total_score"] > 80
        assert result["quality_level"] in ["excellent", "good"]

    def test_score_record_incomplete(self, scorer, base_time):
        """Test scoring an incomplete record."""
        record = {
            # Missing facility_id and timestamp
        }
        result = scorer.score_record(
            record,
            "ProcessSensorData",
            base_time,
        )
        assert result["total_score"] < 50
        assert result["components"]["completeness"]["points"] < 30

    def test_score_record_stale(self, scorer, base_time):
        """Test scoring a stale record."""
        old_time = base_time - timedelta(hours=1)
        record = {
            "facility_id": "PLANT-001",
            "timestamp": old_time.isoformat(),
        }
        result = scorer.score_record(
            record,
            "ProcessSensorData",
            old_time,
            data_category="standard",
        )
        assert result["components"]["timeliness"]["points"] < 20

    def test_score_batch(self, scorer, base_time):
        """Test scoring a batch of records."""
        records = [
            {"facility_id": "P1", "timestamp": base_time.isoformat()},
            {"facility_id": "P1", "timestamp": base_time.isoformat()},
            {},  # Bad record
        ]
        result = scorer.score_batch(
            records,
            "ProcessSensorData",
            timestamp_field="timestamp",
        )
        assert result["total_records"] == 3
        assert "average_score" in result
        assert "quality_distribution" in result


# =============================================================================
# DataQualityManager Tests
# =============================================================================

class TestDataQualityManager:
    """Tests for DataQualityManager class."""

    @pytest.fixture
    def manager(self) -> DataQualityManager:
        """Create quality manager."""
        return DataQualityManager()

    def test_validate_record(self, manager, base_time):
        """Test record validation."""
        record = {
            "facility_id": "PLANT-001",
            "timestamp": base_time.isoformat(),
        }
        result = manager.validate_record(
            record,
            "ProcessSensorData",
            timestamp=base_time,
        )
        assert "validation_status" in result
        assert "quality_report" in result
        assert "lineage_id" in result

    def test_create_quality_metrics(self, manager):
        """Test creating quality metrics."""
        metrics = manager.create_quality_metrics(
            completeness=0.95,
            validity=0.90,
            timeliness=0.85,
            consistency=0.92,
        )
        assert metrics["overall_score"] > 80
        assert metrics["quality_level"] in ["good", "fair"]


# =============================================================================
# Singleton Tests
# =============================================================================

class TestGetQualityManager:
    """Tests for get_quality_manager singleton."""

    def test_singleton_returns_same_instance(self):
        """Test singleton returns same instance."""
        manager1 = get_quality_manager()
        manager2 = get_quality_manager()
        assert manager1 is manager2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
