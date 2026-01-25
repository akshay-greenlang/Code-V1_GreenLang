"""
GL-001 ThermalCommand: Data Quality Rules and Validation

This module implements comprehensive data quality rules for the ThermalCommand
ProcessHeatOrchestrator system, including:

1. Time synchronization validation (NTP/PTP)
2. Unit governance and conversion verification
3. Completeness checks per tag
4. Truth label handling for ML model training
5. Lineage tracking with data hashes and IDs

Data Quality Scoring (0-100):
- Completeness: 30 points (% of required fields populated)
- Validity: 30 points (% passing schema/range validation)
- Timeliness: 20 points (data freshness score)
- Consistency: 20 points (cross-field validation score)

Standards Compliance:
- ISO 8000 Data Quality
- DAMA-DMBOK Data Management
- ISO 50001 Energy Data Quality

Author: GreenLang Data Integration Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enumerations
# =============================================================================

class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    ERROR = "error"         # Data rejected
    WARNING = "warning"     # Data accepted with flag
    INFO = "info"           # Informational note


class TimeSource(str, Enum):
    """Time synchronization source types."""
    NTP = "ntp"
    PTP = "ptp"
    GPS = "gps"
    LOCAL = "local"
    UNKNOWN = "unknown"


class DataLineageType(str, Enum):
    """Data lineage relationship types."""
    SOURCE = "source"           # Original data source
    DERIVED = "derived"         # Calculated from other data
    AGGREGATED = "aggregated"   # Aggregated from multiple records
    TRANSFORMED = "transformed" # Transformed/cleaned
    ENRICHED = "enriched"       # Enriched with external data


class TruthLabelStatus(str, Enum):
    """Truth label status for ML training data."""
    UNLABELED = "unlabeled"
    PENDING_REVIEW = "pending_review"
    LABELED = "labeled"
    VERIFIED = "verified"
    DISPUTED = "disputed"


# =============================================================================
# Time Synchronization Validation
# =============================================================================

class TimeSyncValidator:
    """
    Validates time synchronization quality for industrial data.

    Requirements:
    - NTP: Accuracy within 100ms for general SCADA data
    - PTP: Accuracy within 1ms for safety-critical and CEMS data
    - GPS: Accuracy within 50ns for metering applications
    """

    # Maximum allowed time drift in milliseconds by source
    MAX_DRIFT_MS = {
        TimeSource.NTP: 100,
        TimeSource.PTP: 1,
        TimeSource.GPS: 0.05,  # 50 nanoseconds
        TimeSource.LOCAL: 1000,
        TimeSource.UNKNOWN: 5000,
    }

    # Stale data thresholds in seconds
    STALE_THRESHOLD_SECONDS = {
        "critical": 5,      # Safety-critical tags
        "standard": 30,     # Standard process tags
        "slow": 300,        # Slow-changing values (prices, weather)
        "batch": 3600,      # Batch/periodic data
    }

    def __init__(
        self,
        reference_time: Optional[datetime] = None,
        time_source: TimeSource = TimeSource.NTP
    ):
        """
        Initialize time sync validator.

        Args:
            reference_time: Reference time for validation (default: current UTC)
            time_source: Time synchronization source
        """
        self.reference_time = reference_time or datetime.now(timezone.utc)
        self.time_source = time_source
        self.max_drift = timedelta(milliseconds=self.MAX_DRIFT_MS[time_source])

    def validate_timestamp(
        self,
        timestamp: datetime,
        data_category: str = "standard",
        tag_name: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a timestamp for quality issues.

        Args:
            timestamp: Timestamp to validate
            data_category: Data category for stale threshold
            tag_name: Optional tag name for context

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Ensure timezone aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Check for future timestamp
        if timestamp > self.reference_time + self.max_drift:
            issues.append(
                f"Timestamp {timestamp.isoformat()} is in the future "
                f"(reference: {self.reference_time.isoformat()})"
            )

        # Check for stale data
        stale_threshold = self.STALE_THRESHOLD_SECONDS.get(data_category, 30)
        age_seconds = (self.reference_time - timestamp).total_seconds()

        if age_seconds > stale_threshold:
            issues.append(
                f"Data is stale: age {age_seconds:.1f}s exceeds "
                f"threshold {stale_threshold}s for category '{data_category}'"
            )

        # Check for unreasonably old data (>24 hours)
        if age_seconds > 86400:
            issues.append(
                f"Data is very old: {age_seconds/3600:.1f} hours"
            )

        is_valid = len(issues) == 0
        return is_valid, issues

    def check_time_drift(
        self,
        timestamps: List[datetime],
        expected_interval_ms: int
    ) -> Tuple[float, List[str]]:
        """
        Check for time drift in a sequence of timestamps.

        Args:
            timestamps: List of timestamps in order
            expected_interval_ms: Expected interval between samples

        Returns:
            Tuple of (max_drift_ms, list of drift warnings)
        """
        if len(timestamps) < 2:
            return 0.0, []

        warnings = []
        max_drift = 0.0
        expected_interval = timedelta(milliseconds=expected_interval_ms)

        for i in range(1, len(timestamps)):
            actual_interval = timestamps[i] - timestamps[i-1]
            drift = abs(actual_interval - expected_interval)
            drift_ms = drift.total_seconds() * 1000

            if drift_ms > max_drift:
                max_drift = drift_ms

            # Check for significant drift
            if drift_ms > expected_interval_ms * 0.1:  # >10% drift
                warnings.append(
                    f"Significant time drift at index {i}: "
                    f"expected {expected_interval_ms}ms, "
                    f"actual {actual_interval.total_seconds()*1000:.1f}ms"
                )

        return max_drift, warnings


# =============================================================================
# Data Completeness Validation
# =============================================================================

class CompletenessValidator:
    """
    Validates data completeness for records and batches.

    Completeness Requirements by Category:
    - Safety-critical: 100% required
    - Process control: 95% required
    - Reporting: 90% required
    - Analytics: 80% required
    """

    COMPLETENESS_REQUIREMENTS = {
        "safety_critical": 1.0,
        "process_control": 0.95,
        "reporting": 0.90,
        "analytics": 0.80,
    }

    def __init__(self, required_fields: Dict[str, List[str]]):
        """
        Initialize completeness validator.

        Args:
            required_fields: Dict mapping schema name to required field list
        """
        self.required_fields = required_fields

    def check_record_completeness(
        self,
        record: Dict[str, Any],
        schema_name: str,
        category: str = "process_control"
    ) -> Tuple[float, List[str]]:
        """
        Check completeness of a single record.

        Args:
            record: Data record as dictionary
            schema_name: Schema name for required fields lookup
            category: Data category for threshold

        Returns:
            Tuple of (completeness_score, missing_fields)
        """
        required = self.required_fields.get(schema_name, [])
        if not required:
            return 1.0, []

        missing = []
        for field_name in required:
            value = self._get_nested_value(record, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)

        completeness = 1.0 - (len(missing) / len(required))
        return completeness, missing

    def check_batch_completeness(
        self,
        records: List[Dict[str, Any]],
        schema_name: str,
        category: str = "process_control"
    ) -> Dict[str, Any]:
        """
        Check completeness across a batch of records.

        Args:
            records: List of data records
            schema_name: Schema name
            category: Data category

        Returns:
            Completeness report dictionary
        """
        if not records:
            return {
                "total_records": 0,
                "complete_records": 0,
                "completeness_rate": 0.0,
                "field_completeness": {},
                "meets_requirement": False,
            }

        required = self.required_fields.get(schema_name, [])
        field_counts = {f: 0 for f in required}
        complete_count = 0

        for record in records:
            score, missing = self.check_record_completeness(record, schema_name)
            if score >= self.COMPLETENESS_REQUIREMENTS.get(category, 0.95):
                complete_count += 1

            for field_name in required:
                if field_name not in missing:
                    field_counts[field_name] += 1

        total = len(records)
        field_completeness = {
            f: count / total for f, count in field_counts.items()
        }

        overall_rate = complete_count / total
        requirement = self.COMPLETENESS_REQUIREMENTS.get(category, 0.95)

        return {
            "total_records": total,
            "complete_records": complete_count,
            "completeness_rate": overall_rate,
            "field_completeness": field_completeness,
            "requirement": requirement,
            "meets_requirement": overall_rate >= requirement,
            "lowest_completeness_fields": sorted(
                field_completeness.items(),
                key=lambda x: x[1]
            )[:5],
        }

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


# =============================================================================
# Data Validity Validation
# =============================================================================

class ValidityValidator:
    """
    Validates data values against schema rules and physical constraints.
    """

    def __init__(self):
        """Initialize validity validator with standard rules."""
        self.custom_rules: Dict[str, Callable] = {}

    def register_rule(
        self,
        rule_name: str,
        rule_func: Callable[[Any], Tuple[bool, Optional[str]]]
    ):
        """
        Register a custom validation rule.

        Args:
            rule_name: Unique rule identifier
            rule_func: Function taking value, returning (is_valid, error_msg)
        """
        self.custom_rules[rule_name] = rule_func

    def validate_range(
        self,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        field_name: str = "value"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate numeric value is within range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Field name for error message

        Returns:
            Tuple of (is_valid, error_message)
        """
        if min_val is not None and value < min_val:
            return False, f"{field_name} ({value}) below minimum ({min_val})"
        if max_val is not None and value > max_val:
            return False, f"{field_name} ({value}) above maximum ({max_val})"
        return True, None

    def validate_rate_of_change(
        self,
        current: float,
        previous: float,
        max_change_per_second: float,
        interval_seconds: float,
        field_name: str = "value"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate rate of change is within physical limits.

        Args:
            current: Current value
            previous: Previous value
            max_change_per_second: Maximum allowed change rate
            interval_seconds: Time interval between values
            field_name: Field name for error message

        Returns:
            Tuple of (is_valid, error_message)
        """
        if interval_seconds <= 0:
            return True, None

        actual_rate = abs(current - previous) / interval_seconds
        max_allowed = max_change_per_second

        if actual_rate > max_allowed:
            return False, (
                f"{field_name} rate of change ({actual_rate:.2f}/s) "
                f"exceeds limit ({max_allowed}/s)"
            )
        return True, None

    def validate_physical_consistency(
        self,
        data: Dict[str, float],
        rules: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Validate physical consistency between related values.

        Args:
            data: Dictionary of field values
            rules: List of consistency rules

        Returns:
            List of validation errors
        """
        errors = []

        for rule in rules:
            rule_type = rule.get("type")

            if rule_type == "greater_than":
                field1 = rule.get("field1")
                field2 = rule.get("field2")
                if field1 in data and field2 in data:
                    if data[field1] <= data[field2]:
                        errors.append(
                            f"{field1} ({data[field1]}) should be greater than "
                            f"{field2} ({data[field2]})"
                        )

            elif rule_type == "sum_equals":
                fields = rule.get("fields", [])
                total_field = rule.get("total_field")
                tolerance = rule.get("tolerance", 0.01)
                if all(f in data for f in fields) and total_field in data:
                    calculated_sum = sum(data[f] for f in fields)
                    expected = data[total_field]
                    if abs(calculated_sum - expected) > tolerance * expected:
                        errors.append(
                            f"Sum of {fields} ({calculated_sum:.2f}) "
                            f"doesn't match {total_field} ({expected:.2f})"
                        )

            elif rule_type == "ratio_range":
                numerator = rule.get("numerator")
                denominator = rule.get("denominator")
                min_ratio = rule.get("min", 0)
                max_ratio = rule.get("max", float("inf"))
                if numerator in data and denominator in data:
                    if data[denominator] != 0:
                        ratio = data[numerator] / data[denominator]
                        if ratio < min_ratio or ratio > max_ratio:
                            errors.append(
                                f"Ratio {numerator}/{denominator} ({ratio:.2f}) "
                                f"outside range [{min_ratio}, {max_ratio}]"
                            )

        return errors


# =============================================================================
# Data Lineage Tracking
# =============================================================================

@dataclass
class LineageNode:
    """Represents a node in the data lineage graph."""

    record_id: str
    data_hash: str
    source_system: str
    timestamp: datetime
    lineage_type: DataLineageType
    parent_ids: List[str] = field(default_factory=list)
    transformation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LineageTracker:
    """
    Tracks data lineage for audit and debugging.

    Provides:
    - Hash-based integrity verification
    - Parent-child relationship tracking
    - Transformation chain recording
    - Lineage graph visualization
    """

    def __init__(self):
        """Initialize lineage tracker."""
        self._nodes: Dict[str, LineageNode] = {}

    def compute_hash(self, data: Any) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Data to hash (must be JSON serializable)

        Returns:
            Hex-encoded SHA-256 hash
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def create_source_record(
        self,
        data: Dict[str, Any],
        source_system: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a source lineage record.

        Args:
            data: Source data
            source_system: Source system identifier
            timestamp: Data timestamp (default: now)
            metadata: Additional metadata

        Returns:
            Generated record ID
        """
        record_id = str(uuid.uuid4())
        data_hash = self.compute_hash(data)

        node = LineageNode(
            record_id=record_id,
            data_hash=data_hash,
            source_system=source_system,
            timestamp=timestamp or datetime.now(timezone.utc),
            lineage_type=DataLineageType.SOURCE,
            metadata=metadata or {},
        )

        self._nodes[record_id] = node
        return record_id

    def create_derived_record(
        self,
        data: Dict[str, Any],
        parent_ids: List[str],
        transformation: str,
        source_system: str = "greenlang",
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a derived lineage record.

        Args:
            data: Derived data
            parent_ids: List of parent record IDs
            transformation: Description of transformation applied
            source_system: Processing system
            timestamp: Processing timestamp
            metadata: Additional metadata

        Returns:
            Generated record ID
        """
        record_id = str(uuid.uuid4())
        data_hash = self.compute_hash(data)

        node = LineageNode(
            record_id=record_id,
            data_hash=data_hash,
            source_system=source_system,
            timestamp=timestamp or datetime.now(timezone.utc),
            lineage_type=DataLineageType.DERIVED,
            parent_ids=parent_ids,
            transformation=transformation,
            metadata=metadata or {},
        )

        self._nodes[record_id] = node
        return record_id

    def verify_integrity(self, record_id: str, data: Dict[str, Any]) -> bool:
        """
        Verify data integrity against stored hash.

        Args:
            record_id: Record ID to verify
            data: Current data to check

        Returns:
            True if hash matches, False otherwise
        """
        node = self._nodes.get(record_id)
        if not node:
            return False

        current_hash = self.compute_hash(data)
        return current_hash == node.data_hash

    def get_lineage_chain(self, record_id: str) -> List[LineageNode]:
        """
        Get complete lineage chain for a record.

        Args:
            record_id: Record ID to trace

        Returns:
            List of lineage nodes from source to current
        """
        chain = []
        visited = set()

        def traverse(rid: str):
            if rid in visited or rid not in self._nodes:
                return
            visited.add(rid)

            node = self._nodes[rid]
            for parent_id in node.parent_ids:
                traverse(parent_id)
            chain.append(node)

        traverse(record_id)
        return chain

    def export_lineage(self, record_id: str) -> Dict[str, Any]:
        """
        Export lineage information for a record.

        Args:
            record_id: Record ID

        Returns:
            Lineage information dictionary
        """
        chain = self.get_lineage_chain(record_id)

        return {
            "record_id": record_id,
            "chain_length": len(chain),
            "lineage": [
                {
                    "record_id": node.record_id,
                    "data_hash": node.data_hash,
                    "source_system": node.source_system,
                    "timestamp": node.timestamp.isoformat(),
                    "lineage_type": node.lineage_type.value,
                    "parent_ids": node.parent_ids,
                    "transformation": node.transformation,
                }
                for node in chain
            ]
        }


# =============================================================================
# Truth Label Handler
# =============================================================================

@dataclass
class TruthLabel:
    """Truth label for ML training data."""

    record_id: str
    label_type: str
    label_value: Any
    confidence: float
    status: TruthLabelStatus
    labeled_by: str
    labeled_at: datetime
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    notes: Optional[str] = None


class TruthLabelHandler:
    """
    Manages truth labels for ML model training data.

    Supports:
    - Label creation and verification workflows
    - Multi-annotator agreement tracking
    - Label quality scoring
    - Export for ML pipelines
    """

    def __init__(self):
        """Initialize truth label handler."""
        self._labels: Dict[str, List[TruthLabel]] = {}

    def add_label(
        self,
        record_id: str,
        label_type: str,
        label_value: Any,
        labeled_by: str,
        confidence: float = 1.0,
        notes: Optional[str] = None
    ) -> TruthLabel:
        """
        Add a truth label for a record.

        Args:
            record_id: Data record ID
            label_type: Type of label (e.g., 'anomaly', 'equipment_state')
            label_value: Label value
            labeled_by: Annotator identifier
            confidence: Label confidence (0-1)
            notes: Optional notes

        Returns:
            Created TruthLabel
        """
        label = TruthLabel(
            record_id=record_id,
            label_type=label_type,
            label_value=label_value,
            confidence=confidence,
            status=TruthLabelStatus.LABELED,
            labeled_by=labeled_by,
            labeled_at=datetime.now(timezone.utc),
            notes=notes,
        )

        if record_id not in self._labels:
            self._labels[record_id] = []
        self._labels[record_id].append(label)

        return label

    def verify_label(
        self,
        record_id: str,
        label_type: str,
        verified_by: str,
        is_correct: bool
    ) -> Optional[TruthLabel]:
        """
        Verify or dispute a truth label.

        Args:
            record_id: Data record ID
            label_type: Label type to verify
            verified_by: Verifier identifier
            is_correct: Whether label is correct

        Returns:
            Updated label or None
        """
        labels = self._labels.get(record_id, [])
        for label in labels:
            if label.label_type == label_type:
                label.verified_by = verified_by
                label.verified_at = datetime.now(timezone.utc)
                label.status = (
                    TruthLabelStatus.VERIFIED if is_correct
                    else TruthLabelStatus.DISPUTED
                )
                return label
        return None

    def get_consensus_label(
        self,
        record_id: str,
        label_type: str,
        min_annotators: int = 2,
        agreement_threshold: float = 0.8
    ) -> Optional[Tuple[Any, float]]:
        """
        Get consensus label from multiple annotators.

        Args:
            record_id: Data record ID
            label_type: Label type
            min_annotators: Minimum number of annotators required
            agreement_threshold: Minimum agreement ratio

        Returns:
            Tuple of (consensus_value, agreement_ratio) or None
        """
        labels = [
            l for l in self._labels.get(record_id, [])
            if l.label_type == label_type
        ]

        if len(labels) < min_annotators:
            return None

        # Count label values
        value_counts: Dict[Any, int] = {}
        for label in labels:
            key = str(label.label_value)  # Convert to string for dict key
            value_counts[key] = value_counts.get(key, 0) + 1

        # Find consensus
        total = len(labels)
        for value, count in value_counts.items():
            agreement = count / total
            if agreement >= agreement_threshold:
                return (value, agreement)

        return None

    def get_labels_for_training(
        self,
        label_type: str,
        min_status: TruthLabelStatus = TruthLabelStatus.LABELED
    ) -> List[Dict[str, Any]]:
        """
        Get labels ready for ML training.

        Args:
            label_type: Label type to export
            min_status: Minimum label status

        Returns:
            List of label dictionaries
        """
        status_order = [
            TruthLabelStatus.UNLABELED,
            TruthLabelStatus.PENDING_REVIEW,
            TruthLabelStatus.LABELED,
            TruthLabelStatus.VERIFIED,
        ]

        min_idx = status_order.index(min_status)

        results = []
        for record_id, labels in self._labels.items():
            for label in labels:
                if label.label_type != label_type:
                    continue
                if label.status == TruthLabelStatus.DISPUTED:
                    continue
                if status_order.index(label.status) >= min_idx:
                    results.append({
                        "record_id": record_id,
                        "label_value": label.label_value,
                        "confidence": label.confidence,
                        "status": label.status.value,
                        "labeled_by": label.labeled_by,
                        "labeled_at": label.labeled_at.isoformat(),
                    })

        return results


# =============================================================================
# Data Quality Scorer
# =============================================================================

class DataQualityScorer:
    """
    Calculates overall data quality score.

    Score Components (0-100 total):
    - Completeness: 30 points
    - Validity: 30 points
    - Timeliness: 20 points
    - Consistency: 20 points
    """

    def __init__(
        self,
        completeness_validator: CompletenessValidator,
        validity_validator: ValidityValidator,
        time_validator: TimeSyncValidator
    ):
        """
        Initialize data quality scorer.

        Args:
            completeness_validator: Completeness validator instance
            validity_validator: Validity validator instance
            time_validator: Time sync validator instance
        """
        self.completeness = completeness_validator
        self.validity = validity_validator
        self.time_sync = time_validator

    def score_record(
        self,
        record: Dict[str, Any],
        schema_name: str,
        timestamp: datetime,
        validation_rules: Optional[List[Dict]] = None,
        data_category: str = "standard"
    ) -> Dict[str, Any]:
        """
        Calculate quality score for a single record.

        Args:
            record: Data record
            schema_name: Schema name for validation
            timestamp: Record timestamp
            validation_rules: Optional consistency rules
            data_category: Data category for thresholds

        Returns:
            Quality score report
        """
        # Completeness score (30 points)
        completeness_score, missing_fields = self.completeness.check_record_completeness(
            record, schema_name
        )
        completeness_points = completeness_score * 30

        # Timeliness score (20 points)
        time_valid, time_issues = self.time_sync.validate_timestamp(
            timestamp, data_category
        )
        # Calculate age penalty
        age_seconds = (
            self.time_sync.reference_time - timestamp
        ).total_seconds()
        stale_threshold = TimeSyncValidator.STALE_THRESHOLD_SECONDS.get(
            data_category, 30
        )
        if age_seconds <= 0:
            timeliness_score = 1.0
        elif age_seconds <= stale_threshold:
            timeliness_score = 1.0 - (age_seconds / stale_threshold) * 0.5
        else:
            timeliness_score = max(0, 0.5 - (age_seconds - stale_threshold) / stale_threshold * 0.5)
        timeliness_points = timeliness_score * 20

        # Validity score (30 points)
        validity_errors = []
        if validation_rules:
            # Extract numeric fields for consistency check
            numeric_data = {
                k: v for k, v in record.items()
                if isinstance(v, (int, float))
            }
            validity_errors = self.validity.validate_physical_consistency(
                numeric_data, validation_rules
            )

        validity_score = 1.0 if not validity_errors else max(0, 1.0 - len(validity_errors) * 0.2)
        validity_points = validity_score * 30

        # Consistency score (20 points) - based on missing fields and validity
        consistency_score = 1.0
        if missing_fields:
            consistency_score -= len(missing_fields) * 0.1
        if validity_errors:
            consistency_score -= len(validity_errors) * 0.1
        consistency_score = max(0, consistency_score)
        consistency_points = consistency_score * 20

        # Calculate total
        total_score = (
            completeness_points +
            validity_points +
            timeliness_points +
            consistency_points
        )

        # Determine quality level
        if total_score >= 95:
            quality_level = "excellent"
        elif total_score >= 85:
            quality_level = "good"
        elif total_score >= 70:
            quality_level = "fair"
        elif total_score >= 50:
            quality_level = "poor"
        else:
            quality_level = "critical"

        return {
            "total_score": round(total_score, 2),
            "quality_level": quality_level,
            "components": {
                "completeness": {
                    "score": round(completeness_score, 4),
                    "points": round(completeness_points, 2),
                    "max_points": 30,
                    "missing_fields": missing_fields,
                },
                "validity": {
                    "score": round(validity_score, 4),
                    "points": round(validity_points, 2),
                    "max_points": 30,
                    "errors": validity_errors,
                },
                "timeliness": {
                    "score": round(timeliness_score, 4),
                    "points": round(timeliness_points, 2),
                    "max_points": 20,
                    "issues": time_issues,
                    "data_age_seconds": round(age_seconds, 2),
                },
                "consistency": {
                    "score": round(consistency_score, 4),
                    "points": round(consistency_points, 2),
                    "max_points": 20,
                },
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def score_batch(
        self,
        records: List[Dict[str, Any]],
        schema_name: str,
        timestamp_field: str = "timestamp",
        validation_rules: Optional[List[Dict]] = None,
        data_category: str = "standard"
    ) -> Dict[str, Any]:
        """
        Calculate aggregate quality scores for a batch.

        Args:
            records: List of data records
            schema_name: Schema name
            timestamp_field: Field containing timestamp
            validation_rules: Optional consistency rules
            data_category: Data category

        Returns:
            Batch quality report
        """
        if not records:
            return {
                "total_records": 0,
                "average_score": 0.0,
                "quality_distribution": {},
            }

        scores = []
        quality_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "critical": 0}

        for record in records:
            timestamp = record.get(timestamp_field, datetime.now(timezone.utc))
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            score_report = self.score_record(
                record, schema_name, timestamp, validation_rules, data_category
            )
            scores.append(score_report["total_score"])
            quality_counts[score_report["quality_level"]] += 1

        return {
            "total_records": len(records),
            "average_score": round(statistics.mean(scores), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "std_dev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
            "quality_distribution": quality_counts,
            "records_meeting_threshold": {
                "excellent_95": sum(1 for s in scores if s >= 95),
                "good_85": sum(1 for s in scores if s >= 85),
                "acceptable_70": sum(1 for s in scores if s >= 70),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# Data Quality Manager
# =============================================================================

class DataQualityManager:
    """
    Central manager for all data quality operations.

    Provides unified interface for:
    - Time synchronization validation
    - Completeness checking
    - Validity validation
    - Lineage tracking
    - Truth label management
    - Quality scoring
    """

    def __init__(
        self,
        required_fields: Optional[Dict[str, List[str]]] = None,
        time_source: TimeSource = TimeSource.NTP
    ):
        """
        Initialize data quality manager.

        Args:
            required_fields: Dict of schema -> required fields
            time_source: Time synchronization source
        """
        # Initialize validators
        self.time_validator = TimeSyncValidator(time_source=time_source)
        self.completeness_validator = CompletenessValidator(
            required_fields or self._default_required_fields()
        )
        self.validity_validator = ValidityValidator()

        # Initialize trackers
        self.lineage_tracker = LineageTracker()
        self.truth_label_handler = TruthLabelHandler()

        # Initialize scorer
        self.scorer = DataQualityScorer(
            self.completeness_validator,
            self.validity_validator,
            self.time_validator
        )

        # Register standard validation rules
        self._register_standard_rules()

    def _default_required_fields(self) -> Dict[str, List[str]]:
        """Get default required fields for standard schemas."""
        return {
            "ProcessSensorData": [
                "facility_id",
                "timestamp",
            ],
            "EnergyConsumptionData": [
                "facility_id",
                "timestamp",
                "period_start",
                "period_end",
            ],
            "SafetySystemStatus": [
                "facility_id",
                "timestamp",
                "dispatch_enabled",
                "sil_status",
            ],
            "ProductionSchedule": [
                "facility_id",
                "timestamp",
                "schedule_horizon_start",
                "schedule_horizon_end",
            ],
            "WeatherForecast": [
                "facility_id",
                "timestamp",
                "current_temperature_c",
            ],
            "EnergyPrices": [
                "facility_id",
                "timestamp",
                "current_rt_price_usd_mwh",
            ],
            "EquipmentHealth": [
                "facility_id",
                "timestamp",
                "equipment_id",
                "health_score",
            ],
            "AlarmState": [
                "facility_id",
                "timestamp",
            ],
        }

    def _register_standard_rules(self):
        """Register standard validation rules."""
        # Temperature must be reasonable
        self.validity_validator.register_rule(
            "temperature_reasonable",
            lambda v: (
                (True, None) if -273.15 <= v <= 2000
                else (False, f"Temperature {v}C is physically impossible")
            )
        )

        # Pressure must be non-negative (absolute)
        self.validity_validator.register_rule(
            "pressure_absolute_positive",
            lambda v: (
                (True, None) if v >= 0
                else (False, f"Absolute pressure {v} cannot be negative")
            )
        )

        # Efficiency must be 0-100%
        self.validity_validator.register_rule(
            "efficiency_percentage",
            lambda v: (
                (True, None) if 0 <= v <= 100
                else (False, f"Efficiency {v}% must be 0-100")
            )
        )

    def validate_record(
        self,
        record: Dict[str, Any],
        schema_name: str,
        timestamp: Optional[datetime] = None,
        consistency_rules: Optional[List[Dict]] = None,
        data_category: str = "standard"
    ) -> Dict[str, Any]:
        """
        Perform full validation on a record.

        Args:
            record: Data record to validate
            schema_name: Schema name for validation rules
            timestamp: Record timestamp (extracted from record if not provided)
            consistency_rules: Optional consistency rules
            data_category: Data category for thresholds

        Returns:
            Complete validation report
        """
        # Extract timestamp
        if timestamp is None:
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

        # Get quality score
        quality_report = self.scorer.score_record(
            record, schema_name, timestamp, consistency_rules, data_category
        )

        # Create lineage record
        lineage_id = self.lineage_tracker.create_source_record(
            record,
            source_system=record.get("provenance", {}).get("source_system", "unknown"),
            timestamp=timestamp,
            metadata={"schema": schema_name, "quality_score": quality_report["total_score"]}
        )

        return {
            "validation_status": "passed" if quality_report["total_score"] >= 70 else "failed",
            "quality_report": quality_report,
            "lineage_id": lineage_id,
            "schema_name": schema_name,
            "data_category": data_category,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

    def create_quality_metrics(
        self,
        completeness: float,
        validity: float,
        timeliness: float,
        consistency: float,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a DataQualityMetrics-compatible dictionary.

        Args:
            completeness: Completeness score (0-1)
            validity: Validity score (0-1)
            timeliness: Timeliness score (0-1)
            consistency: Consistency score (0-1)
            errors: List of validation errors
            warnings: List of validation warnings

        Returns:
            Quality metrics dictionary
        """
        overall = (
            completeness * 30 +
            validity * 30 +
            timeliness * 20 +
            consistency * 20
        )

        if overall >= 95:
            level = "good"
        elif overall >= 80:
            level = "fair"
        else:
            level = "poor"

        return {
            "quality_level": level,
            "completeness_score": completeness,
            "validity_score": validity,
            "timeliness_score": timeliness,
            "consistency_score": consistency,
            "overall_score": overall,
            "validation_errors": errors or [],
            "validation_warnings": warnings or [],
        }


# =============================================================================
# Factory Functions
# =============================================================================

_quality_manager: Optional[DataQualityManager] = None


def get_quality_manager() -> DataQualityManager:
    """
    Get the global data quality manager instance.

    Returns:
        DataQualityManager singleton instance
    """
    global _quality_manager
    if _quality_manager is None:
        _quality_manager = DataQualityManager()
    return _quality_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ValidationSeverity",
    "TimeSource",
    "DataLineageType",
    "TruthLabelStatus",
    # Validators
    "TimeSyncValidator",
    "CompletenessValidator",
    "ValidityValidator",
    # Lineage
    "LineageNode",
    "LineageTracker",
    # Truth labels
    "TruthLabel",
    "TruthLabelHandler",
    # Scoring
    "DataQualityScorer",
    # Manager
    "DataQualityManager",
    # Functions
    "get_quality_manager",
]
