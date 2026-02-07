# -*- coding: utf-8 -*-
"""
Evidence Validator - SEC-009 Phase 3

Comprehensive validation for SOC 2 audit evidence including:
    - Integrity: Hash verification and content completeness
    - Completeness: Coverage of required evidence for criteria
    - Freshness: Evidence is within the audit period
    - Format: Evidence structure and content validation

The validator supports both synchronous and asynchronous validation
with configurable validation rules and severity levels.

Example:
    >>> validator = EvidenceValidator(config)
    >>> result = validator.validate_integrity(evidence)
    >>> completeness = validator.validate_completeness("CC6.1", evidence_list)
    >>> all_results = await validator.validate_all(evidence_list)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field

from greenlang.infrastructure.soc2_preparation.evidence.models import (
    DateRange,
    Evidence,
    EvidenceSource,
    EvidenceStatus,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class EvidenceValidatorConfig(BaseModel):
    """Configuration for the evidence validator."""

    # Validation settings
    strict_mode: bool = Field(
        default=True,
        description="Fail on warnings in strict mode",
    )
    max_content_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum content size in MB",
    )
    min_description_length: int = Field(
        default=10,
        ge=0,
        description="Minimum description length",
    )

    # Freshness settings
    max_age_days: int = Field(
        default=365,
        ge=1,
        description="Maximum age of evidence in days",
    )
    freshness_warning_days: int = Field(
        default=30,
        ge=1,
        description="Warn if evidence is older than this",
    )

    # Completeness requirements
    required_fields: List[str] = Field(
        default_factory=lambda: [
            "evidence_id",
            "criterion_id",
            "evidence_type",
            "source",
            "title",
        ],
    )
    recommended_fields: List[str] = Field(
        default_factory=lambda: [
            "description",
            "collected_at",
            "period_start",
            "period_end",
        ],
    )


# ---------------------------------------------------------------------------
# Criterion Evidence Requirements
# ---------------------------------------------------------------------------

# Minimum evidence requirements per criterion
CRITERION_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # CC6: Logical and Physical Access Controls
    "CC6.1": {
        "min_evidence_count": 3,
        "required_types": [EvidenceType.POLICY, EvidenceType.LOG_EXPORT],
        "description": "Logical access security policies and access logs",
    },
    "CC6.2": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.CONFIGURATION, EvidenceType.LOG_EXPORT],
        "description": "Access provisioning and de-provisioning evidence",
    },
    "CC6.3": {
        "min_evidence_count": 3,
        "required_types": [EvidenceType.ACCESS_REVIEW, EvidenceType.LOG_EXPORT],
        "description": "Access review and authorization evidence",
    },
    "CC6.4": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.CONFIGURATION, EvidenceType.LOG_EXPORT],
        "description": "Physical access restrictions evidence",
    },
    "CC6.5": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.CONFIGURATION],
        "description": "Data transmission security evidence",
    },
    "CC6.6": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.CONFIGURATION, EvidenceType.LOG_EXPORT],
        "description": "Encryption and key management evidence",
    },
    "CC6.7": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.LOG_EXPORT],
        "description": "Modification and deletion restriction evidence",
    },
    "CC6.8": {
        "min_evidence_count": 1,
        "required_types": [EvidenceType.POLICY],
        "description": "Malicious software protection evidence",
    },

    # CC7: System Operations
    "CC7.1": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.METRIC_EXPORT, EvidenceType.LOG_EXPORT],
        "description": "System monitoring evidence",
    },
    "CC7.2": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.LOG_EXPORT],
        "description": "Anomaly detection evidence",
    },
    "CC7.3": {
        "min_evidence_count": 3,
        "required_types": [EvidenceType.TICKET, EvidenceType.CODE_CHANGE],
        "description": "Change management evidence",
    },
    "CC7.4": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.INCIDENT_REPORT, EvidenceType.TICKET],
        "description": "Security incident management evidence",
    },
    "CC7.5": {
        "min_evidence_count": 1,
        "required_types": [EvidenceType.RECOVERY_TEST],
        "description": "System recovery evidence",
    },

    # CC8: Change Management
    "CC8.1": {
        "min_evidence_count": 5,
        "required_types": [EvidenceType.TICKET, EvidenceType.CODE_CHANGE],
        "description": "Change management process evidence",
    },

    # A1: Availability
    "A1.1": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.METRIC_EXPORT],
        "description": "Capacity planning and monitoring evidence",
    },
    "A1.2": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.BACKUP_VERIFICATION],
        "description": "Backup and recovery evidence",
    },
    "A1.3": {
        "min_evidence_count": 1,
        "required_types": [EvidenceType.RECOVERY_TEST],
        "description": "Disaster recovery evidence",
    },

    # C1: Confidentiality
    "C1.1": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.POLICY, EvidenceType.LOG_EXPORT],
        "description": "Confidential data identification evidence",
    },
    "C1.2": {
        "min_evidence_count": 2,
        "required_types": [EvidenceType.LOG_EXPORT],
        "description": "Confidential data disposal evidence",
    },
}


# ---------------------------------------------------------------------------
# Evidence Validator
# ---------------------------------------------------------------------------


class EvidenceValidator:
    """Comprehensive validator for SOC 2 audit evidence.

    Performs integrity, completeness, freshness, and format validation
    on evidence items to ensure they meet audit requirements.

    Example:
        >>> config = EvidenceValidatorConfig()
        >>> validator = EvidenceValidator(config)
        >>> integrity = validator.validate_integrity(evidence)
        >>> completeness = validator.validate_completeness("CC6.1", evidence_list)
    """

    def __init__(self, config: Optional[EvidenceValidatorConfig] = None) -> None:
        """Initialize the evidence validator.

        Args:
            config: Validator configuration (uses defaults if not provided).
        """
        self.config = config or EvidenceValidatorConfig()

    def validate_integrity(self, evidence: Evidence) -> ValidationResult:
        """Validate evidence integrity via hash verification.

        Args:
            evidence: Evidence to validate.

        Returns:
            ValidationResult with integrity status.
        """
        details: Dict[str, Any] = {}
        messages: List[str] = []

        # Check if provenance hash exists
        if evidence.provenance_hash is None:
            # Compute hash for comparison
            computed_hash = evidence.compute_provenance_hash()
            details["computed_hash"] = computed_hash
            messages.append("Provenance hash not set, computed for reference")
            status = ValidationStatus.WARNING
        else:
            # Verify hash matches
            computed_hash = evidence.compute_provenance_hash()
            if computed_hash == evidence.provenance_hash:
                details["hash_verified"] = True
                details["provenance_hash"] = evidence.provenance_hash
                status = ValidationStatus.PASS
                messages.append("Provenance hash verified successfully")
            else:
                details["hash_verified"] = False
                details["stored_hash"] = evidence.provenance_hash
                details["computed_hash"] = computed_hash
                status = ValidationStatus.FAIL
                messages.append(
                    "Provenance hash mismatch - evidence may have been modified"
                )

        # Check content existence
        has_content = bool(
            evidence.content or evidence.file_path or evidence.s3_key
        )
        if not has_content:
            status = ValidationStatus.FAIL
            messages.append("Evidence has no content, file path, or S3 key")
            details["has_content"] = False
        else:
            details["has_content"] = True

        # Check content size
        if evidence.content:
            content_size_mb = len(evidence.content) / (1024 * 1024)
            details["content_size_mb"] = round(content_size_mb, 2)
            if content_size_mb > self.config.max_content_size_mb:
                if status != ValidationStatus.FAIL:
                    status = ValidationStatus.WARNING
                messages.append(
                    f"Content size ({content_size_mb:.2f} MB) exceeds "
                    f"recommended maximum ({self.config.max_content_size_mb} MB)"
                )

        return ValidationResult(
            evidence_id=evidence.evidence_id,
            validation_type="integrity",
            status=status,
            message=" | ".join(messages),
            details=details,
        )

    def validate_completeness(
        self,
        criterion_id: str,
        evidence: List[Evidence],
    ) -> ValidationResult:
        """Validate evidence completeness for a criterion.

        Args:
            criterion_id: SOC 2 criterion ID.
            evidence: List of evidence for this criterion.

        Returns:
            ValidationResult with completeness status.
        """
        details: Dict[str, Any] = {
            "criterion_id": criterion_id,
            "evidence_count": len(evidence),
        }
        messages: List[str] = []
        status = ValidationStatus.PASS

        # Get requirements for this criterion
        requirements = CRITERION_REQUIREMENTS.get(
            criterion_id.upper(),
            {"min_evidence_count": 1, "required_types": []},
        )

        min_count = requirements.get("min_evidence_count", 1)
        required_types = requirements.get("required_types", [])

        details["min_required"] = min_count
        details["required_types"] = [t.value for t in required_types]

        # Check minimum count
        if len(evidence) < min_count:
            status = ValidationStatus.FAIL
            messages.append(
                f"Insufficient evidence: {len(evidence)}/{min_count} required"
            )

        # Check required types
        evidence_types = {e.evidence_type for e in evidence}
        missing_types = set(required_types) - evidence_types

        if missing_types:
            if status != ValidationStatus.FAIL:
                status = ValidationStatus.WARNING
            details["missing_types"] = [t.value for t in missing_types]
            messages.append(
                f"Missing evidence types: {[t.value for t in missing_types]}"
            )
        else:
            details["all_required_types_present"] = True

        # Check evidence status
        rejected_count = sum(
            1 for e in evidence if e.status == EvidenceStatus.REJECTED
        )
        if rejected_count > 0:
            status = ValidationStatus.FAIL
            details["rejected_count"] = rejected_count
            messages.append(f"{rejected_count} evidence items are rejected")

        # Check for expired evidence
        expired_count = sum(
            1 for e in evidence if e.status == EvidenceStatus.EXPIRED
        )
        if expired_count > 0:
            if status != ValidationStatus.FAIL:
                status = ValidationStatus.WARNING
            details["expired_count"] = expired_count
            messages.append(f"{expired_count} evidence items are expired")

        if not messages:
            messages.append(
                f"Criterion {criterion_id} has complete evidence coverage"
            )

        return ValidationResult(
            evidence_id=evidence[0].evidence_id if evidence else UUID(int=0),
            validation_type="completeness",
            status=status,
            message=" | ".join(messages),
            details=details,
        )

    def validate_freshness(
        self,
        evidence: Evidence,
        audit_period: DateRange,
    ) -> ValidationResult:
        """Validate evidence freshness within audit period.

        Args:
            evidence: Evidence to validate.
            audit_period: Audit date range.

        Returns:
            ValidationResult with freshness status.
        """
        details: Dict[str, Any] = {
            "collected_at": evidence.collected_at.isoformat(),
            "audit_start": audit_period.start.isoformat(),
            "audit_end": audit_period.end.isoformat(),
        }
        messages: List[str] = []
        status = ValidationStatus.PASS

        now = datetime.now(timezone.utc)

        # Check if evidence is within audit period
        if evidence.period_start and evidence.period_end:
            # Evidence has explicit period
            if evidence.period_end < audit_period.start:
                status = ValidationStatus.FAIL
                messages.append(
                    "Evidence period ends before audit period starts"
                )
            elif evidence.period_start > audit_period.end:
                status = ValidationStatus.FAIL
                messages.append(
                    "Evidence period starts after audit period ends"
                )
            else:
                # Check overlap
                overlap_start = max(evidence.period_start, audit_period.start)
                overlap_end = min(evidence.period_end, audit_period.end)
                overlap_days = (overlap_end - overlap_start).days
                details["overlap_days"] = overlap_days

                if overlap_days < 1:
                    status = ValidationStatus.WARNING
                    messages.append("Minimal overlap with audit period")

        # Check evidence age
        age_days = (now - evidence.collected_at).days
        details["age_days"] = age_days

        if age_days > self.config.max_age_days:
            status = ValidationStatus.FAIL
            messages.append(
                f"Evidence is {age_days} days old, exceeds maximum "
                f"of {self.config.max_age_days} days"
            )
        elif age_days > self.config.freshness_warning_days:
            if status != ValidationStatus.FAIL:
                status = ValidationStatus.WARNING
            messages.append(
                f"Evidence is {age_days} days old, consider refreshing"
            )

        # Check collected_at vs audit period
        if evidence.collected_at > audit_period.end:
            # Evidence collected after audit period ends - this is fine
            # for point-in-time evidence collected for the audit
            details["post_period_collection"] = True
        elif evidence.collected_at < audit_period.start:
            # Historical evidence - might need validation
            days_before = (audit_period.start - evidence.collected_at).days
            details["days_before_period"] = days_before
            if days_before > 30:
                if status != ValidationStatus.FAIL:
                    status = ValidationStatus.WARNING
                messages.append(
                    f"Evidence collected {days_before} days before audit period"
                )

        if not messages:
            messages.append("Evidence freshness validated successfully")

        return ValidationResult(
            evidence_id=evidence.evidence_id,
            validation_type="freshness",
            status=status,
            message=" | ".join(messages),
            details=details,
        )

    def validate_format(self, evidence: Evidence) -> ValidationResult:
        """Validate evidence format and structure.

        Args:
            evidence: Evidence to validate.

        Returns:
            ValidationResult with format status.
        """
        details: Dict[str, Any] = {}
        messages: List[str] = []
        status = ValidationStatus.PASS

        # Check required fields
        missing_required: List[str] = []
        for field in self.config.required_fields:
            value = getattr(evidence, field, None)
            if value is None or value == "":
                missing_required.append(field)

        if missing_required:
            status = ValidationStatus.FAIL
            details["missing_required_fields"] = missing_required
            messages.append(f"Missing required fields: {missing_required}")

        # Check recommended fields
        missing_recommended: List[str] = []
        for field in self.config.recommended_fields:
            value = getattr(evidence, field, None)
            if value is None or value == "":
                missing_recommended.append(field)

        if missing_recommended:
            if status != ValidationStatus.FAIL:
                status = ValidationStatus.WARNING
            details["missing_recommended_fields"] = missing_recommended
            messages.append(f"Missing recommended fields: {missing_recommended}")

        # Validate title
        if evidence.title:
            if len(evidence.title) < 3:
                if status != ValidationStatus.FAIL:
                    status = ValidationStatus.WARNING
                messages.append("Title is too short (< 3 characters)")
            if len(evidence.title) > 256:
                if status != ValidationStatus.FAIL:
                    status = ValidationStatus.WARNING
                messages.append("Title is too long (> 256 characters)")

        # Validate description
        if evidence.description:
            if len(evidence.description) < self.config.min_description_length:
                if status != ValidationStatus.FAIL:
                    status = ValidationStatus.WARNING
                messages.append(
                    f"Description is too short "
                    f"(< {self.config.min_description_length} characters)"
                )

        # Validate criterion ID format
        criterion_pattern = re.compile(r"^[A-Z]+\d+\.\d+$")
        if not criterion_pattern.match(evidence.criterion_id.upper()):
            if status != ValidationStatus.FAIL:
                status = ValidationStatus.WARNING
            details["invalid_criterion_format"] = evidence.criterion_id
            messages.append(
                f"Invalid criterion ID format: {evidence.criterion_id}"
            )

        # Validate content format for JSON content
        if evidence.content:
            try:
                # Check if content is valid JSON (if it looks like JSON)
                if evidence.content.strip().startswith(("{", "[")):
                    json.loads(evidence.content)
                    details["content_format"] = "valid_json"
            except json.JSONDecodeError:
                # Not valid JSON - might be plain text, which is OK
                details["content_format"] = "text"

        # Check metadata structure
        if evidence.metadata:
            if not isinstance(evidence.metadata, dict):
                status = ValidationStatus.FAIL
                messages.append("Metadata is not a valid dictionary")
            else:
                details["metadata_keys"] = list(evidence.metadata.keys())

        if not messages:
            messages.append("Evidence format validated successfully")

        return ValidationResult(
            evidence_id=evidence.evidence_id,
            validation_type="format",
            status=status,
            message=" | ".join(messages),
            details=details,
        )

    async def validate_all(
        self,
        evidence: List[Evidence],
        audit_period: Optional[DateRange] = None,
    ) -> List[ValidationResult]:
        """Validate all evidence items with all validation types.

        Args:
            evidence: List of evidence to validate.
            audit_period: Optional audit period for freshness validation.

        Returns:
            List of ValidationResults for all evidence and validation types.
        """
        results: List[ValidationResult] = []

        for item in evidence:
            # Integrity validation
            results.append(self.validate_integrity(item))

            # Format validation
            results.append(self.validate_format(item))

            # Freshness validation (if audit period provided)
            if audit_period:
                results.append(self.validate_freshness(item, audit_period))

        # Group by criterion for completeness validation
        by_criterion: Dict[str, List[Evidence]] = {}
        for item in evidence:
            criterion = item.criterion_id.upper()
            if criterion not in by_criterion:
                by_criterion[criterion] = []
            by_criterion[criterion].append(item)

        for criterion_id, criterion_evidence in by_criterion.items():
            results.append(
                self.validate_completeness(criterion_id, criterion_evidence)
            )

        return results

    def get_validation_summary(
        self,
        results: List[ValidationResult],
    ) -> Dict[str, Any]:
        """Generate summary of validation results.

        Args:
            results: List of validation results.

        Returns:
            Summary dictionary with counts and status.
        """
        total = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASS)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAIL)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)

        by_type: Dict[str, Dict[str, int]] = {}
        for r in results:
            if r.validation_type not in by_type:
                by_type[r.validation_type] = {
                    "pass": 0,
                    "fail": 0,
                    "warning": 0,
                    "skipped": 0,
                }
            by_type[r.validation_type][r.status.value] += 1

        overall_status = "pass"
        if failed > 0:
            overall_status = "fail"
        elif warnings > 0 and self.config.strict_mode:
            overall_status = "fail"
        elif warnings > 0:
            overall_status = "warning"

        return {
            "overall_status": overall_status,
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "skipped": skipped,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "by_validation_type": by_type,
            "strict_mode": self.config.strict_mode,
        }

    def get_failed_validations(
        self,
        results: List[ValidationResult],
    ) -> List[ValidationResult]:
        """Get only failed validation results.

        Args:
            results: List of validation results.

        Returns:
            List of failed validations.
        """
        return [r for r in results if r.status == ValidationStatus.FAIL]

    def get_warnings(
        self,
        results: List[ValidationResult],
    ) -> List[ValidationResult]:
        """Get only warning validation results.

        Args:
            results: List of validation results.

        Returns:
            List of warning validations.
        """
        return [r for r in results if r.status == ValidationStatus.WARNING]
