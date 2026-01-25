"""
GL-004 BURNMASTER - Integration Data Validator

Data quality and validation for integration layer.

Features:
    - Tag value validation against limits
    - Data freshness checking
    - Stale data detection
    - Quality code validation
    - Multi-source value reconciliation

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityCode(int, Enum):
    """OPC-style quality codes."""
    GOOD = 192
    GOOD_LOCAL_OVERRIDE = 216
    UNCERTAIN = 64
    UNCERTAIN_LAST_USABLE = 68
    BAD = 0
    BAD_NOT_CONNECTED = 8
    BAD_DEVICE_FAILURE = 12
    BAD_SENSOR_FAILURE = 16
    BAD_COMM_FAILURE = 24
    BAD_OUT_OF_SERVICE = 28


class ValidationStatus(str, Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    OUT_OF_RANGE = "out_of_range"
    STALE = "stale"
    BAD_QUALITY = "bad_quality"


@dataclass
class TagLimits:
    """Tag value limits for validation."""
    tag: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    engineering_units: str = ""
    rate_of_change_limit: Optional[float] = None

    def is_in_range(self, value: float) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class TagValue:
    """Tag value for validation."""
    tag: str
    value: Any
    quality: int
    timestamp: datetime
    source: str = "unknown"


@dataclass
class ValidationResult:
    """Result of tag value validation."""
    is_valid: bool
    status: ValidationStatus
    tag: str
    value: Any
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "tag": self.tag,
            "value": self.value,
            "message": self.message,
        }


@dataclass
class FreshnessCheck:
    """Result of data freshness check."""
    is_fresh: bool
    tag: str
    timestamp: datetime
    age_seconds: float
    max_age_seconds: float
    message: str = ""


@dataclass
class QualityValidation:
    """Result of quality code validation."""
    is_good: bool
    quality_code: int
    quality_name: str
    is_usable: bool
    message: str = ""


@dataclass
class ReconciledValue:
    """Result of multi-source value reconciliation."""
    tag: str
    reconciled_value: float
    confidence: float
    sources_used: List[str]
    sources_rejected: List[str]
    method: str = "weighted_average"
    message: str = ""


class IntegrationDataValidator:
    """
    Data validator for integration layer.

    Validates tag values, checks freshness, detects stale data,
    and reconciles values from multiple sources.
    """

    def __init__(self):
        self._limits_registry: Dict[str, TagLimits] = {}
        self._last_values: Dict[str, TagValue] = {}
        self._stats = {
            "validations": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "stale_count": 0,
        }
        logger.info("IntegrationDataValidator initialized")

    def register_limits(self, limits: TagLimits) -> None:
        """Register limits for a tag."""
        self._limits_registry[limits.tag] = limits
        logger.debug(f"Registered limits for {limits.tag}")

    def validate_tag_value(
        self,
        value: TagValue,
        limits: Optional[TagLimits] = None,
    ) -> ValidationResult:
        """
        Validate a tag value against limits.

        Args:
            value: Tag value to validate
            limits: Optional limits (uses registered limits if not provided)

        Returns:
            ValidationResult with validation status
        """
        self._stats["validations"] += 1

        tag_limits = limits or self._limits_registry.get(value.tag)

        quality_check = self.validate_quality_code(value.quality)
        if not quality_check.is_usable:
            self._stats["invalid_count"] += 1
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.BAD_QUALITY,
                tag=value.tag,
                value=value.value,
                message=quality_check.message,
            )

        if tag_limits:
            if tag_limits.min_value is not None and value.value < tag_limits.min_value:
                self._stats["invalid_count"] += 1
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.OUT_OF_RANGE,
                    tag=value.tag,
                    value=value.value,
                    message=f"Value {value.value} below minimum {tag_limits.min_value}",
                )

            if tag_limits.max_value is not None and value.value > tag_limits.max_value:
                self._stats["invalid_count"] += 1
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.OUT_OF_RANGE,
                    tag=value.tag,
                    value=value.value,
                    message=f"Value {value.value} above maximum {tag_limits.max_value}",
                )

            if tag_limits.warning_low is not None and value.value < tag_limits.warning_low:
                self._stats["valid_count"] += 1
                return ValidationResult(
                    is_valid=True,
                    status=ValidationStatus.WARNING,
                    tag=value.tag,
                    value=value.value,
                    message=f"Value {value.value} below warning threshold {tag_limits.warning_low}",
                )

            if tag_limits.warning_high is not None and value.value > tag_limits.warning_high:
                self._stats["valid_count"] += 1
                return ValidationResult(
                    is_valid=True,
                    status=ValidationStatus.WARNING,
                    tag=value.tag,
                    value=value.value,
                    message=f"Value {value.value} above warning threshold {tag_limits.warning_high}",
                )

        self._last_values[value.tag] = value
        self._stats["valid_count"] += 1

        return ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
            tag=value.tag,
            value=value.value,
            message="Value within acceptable range",
        )

    def check_data_freshness(
        self,
        timestamp: datetime,
        max_age: float,
        tag: str = "",
    ) -> FreshnessCheck:
        """
        Check if data is fresh (not stale).

        Args:
            timestamp: Data timestamp
            max_age: Maximum acceptable age in seconds
            tag: Optional tag name for reporting

        Returns:
            FreshnessCheck with freshness status
        """
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_seconds = (now - timestamp).total_seconds()
        is_fresh = age_seconds <= max_age

        if not is_fresh:
            self._stats["stale_count"] += 1

        return FreshnessCheck(
            is_fresh=is_fresh,
            tag=tag,
            timestamp=timestamp,
            age_seconds=age_seconds,
            max_age_seconds=max_age,
            message="Data is fresh" if is_fresh else f"Data is stale ({age_seconds:.1f}s old, max {max_age}s)",
        )

    def detect_stale_data(
        self,
        values: Dict[str, TagValue],
        max_age_seconds: float = 60.0,
    ) -> List[str]:
        """
        Detect stale tags from a collection of values.

        Args:
            values: Dict of tag name to TagValue
            max_age_seconds: Maximum acceptable age

        Returns:
            List of stale tag names
        """
        stale_tags = []
        now = datetime.now(timezone.utc)

        for tag, value in values.items():
            timestamp = value.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            age = (now - timestamp).total_seconds()
            if age > max_age_seconds:
                stale_tags.append(tag)

        return stale_tags

    def validate_quality_code(self, quality: int) -> QualityValidation:
        """
        Validate OPC-style quality code.

        Args:
            quality: Quality code integer

        Returns:
            QualityValidation with quality status
        """
        try:
            qc = QualityCode(quality)
            quality_name = qc.name
        except ValueError:
            quality_name = f"UNKNOWN_{quality}"

        is_good = quality >= 192
        is_usable = quality >= 64

        if is_good:
            message = "Quality is good"
        elif is_usable:
            message = "Quality is uncertain but usable"
        else:
            message = "Quality is bad, data not usable"

        return QualityValidation(
            is_good=is_good,
            quality_code=quality,
            quality_name=quality_name,
            is_usable=is_usable,
            message=message,
        )

    def reconcile_values(
        self,
        sources: Dict[str, TagValue],
    ) -> ReconciledValue:
        """
        Reconcile values from multiple sources.

        Uses weighted average based on quality codes.

        Args:
            sources: Dict of source name to TagValue

        Returns:
            ReconciledValue with reconciled result
        """
        if not sources:
            raise ValueError("No sources provided for reconciliation")

        tag = list(sources.values())[0].tag
        good_values = []
        sources_used = []
        sources_rejected = []

        for source, value in sources.items():
            quality_check = self.validate_quality_code(value.quality)
            if quality_check.is_usable:
                weight = 1.0 if quality_check.is_good else 0.5
                good_values.append((float(value.value), weight))
                sources_used.append(source)
            else:
                sources_rejected.append(source)

        if not good_values:
            return ReconciledValue(
                tag=tag,
                reconciled_value=0.0,
                confidence=0.0,
                sources_used=[],
                sources_rejected=list(sources.keys()),
                method="none",
                message="No usable values available for reconciliation",
            )

        total_weight = sum(w for _, w in good_values)
        reconciled = sum(v * w for v, w in good_values) / total_weight
        confidence = min(1.0, total_weight / len(sources))

        return ReconciledValue(
            tag=tag,
            reconciled_value=round(reconciled, 4),
            confidence=round(confidence, 2),
            sources_used=sources_used,
            sources_rejected=sources_rejected,
            method="weighted_average",
            message=f"Reconciled from {len(sources_used)} sources",
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            **self._stats,
            "registered_limits": len(self._limits_registry),
            "tracked_tags": len(self._last_values),
        }
