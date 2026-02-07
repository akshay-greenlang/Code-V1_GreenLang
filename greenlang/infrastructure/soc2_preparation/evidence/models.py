# -*- coding: utf-8 -*-
"""
Evidence Management Data Models - SEC-009 Phase 3

Pydantic models for SOC 2 Type II evidence management including:
    - Evidence records with provenance tracking
    - Evidence packages for auditor delivery
    - Validation results and version history
    - Date ranges and collection metadata

All models follow GreenLang patterns with:
    - Immutable audit records (frozen=True where appropriate)
    - SHA-256 provenance hashing for integrity
    - UTC datetime normalization
    - Comprehensive validation

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EvidenceType(str, Enum):
    """Type of evidence collected for SOC 2 controls."""

    POLICY = "policy"
    PROCEDURE = "procedure"
    SCREENSHOT = "screenshot"
    LOG_EXPORT = "log_export"
    CONFIGURATION = "configuration"
    ATTESTATION = "attestation"
    TICKET = "ticket"
    CODE_CHANGE = "code_change"
    ACCESS_REVIEW = "access_review"
    SECURITY_SCAN = "security_scan"
    INCIDENT_REPORT = "incident_report"
    TRAINING_RECORD = "training_record"
    VENDOR_ASSESSMENT = "vendor_assessment"
    PENETRATION_TEST = "penetration_test"
    AUDIT_REPORT = "audit_report"
    METRIC_EXPORT = "metric_export"
    BACKUP_VERIFICATION = "backup_verification"
    RECOVERY_TEST = "recovery_test"


class EvidenceSource(str, Enum):
    """Source system where evidence was collected from."""

    CLOUDTRAIL = "cloudtrail"
    GITHUB = "github"
    POSTGRESQL = "postgresql"
    LOKI = "loki"
    AUTH_SERVICE = "auth_service"
    JIRA = "jira"
    OKTA = "okta"
    S3 = "s3"
    MANUAL = "manual"
    PROMETHEUS = "prometheus"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    CONFLUENCE = "confluence"


class EvidenceStatus(str, Enum):
    """Status of evidence in the collection lifecycle."""

    PENDING = "pending"
    COLLECTED = "collected"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class ValidationStatus(str, Enum):
    """Result of evidence validation."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class ControlFrequency(str, Enum):
    """Frequency at which a control operates."""

    CONTINUOUS = "continuous"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    """Date range for evidence collection periods.

    Attributes:
        start: Start of the period (inclusive).
        end: End of the period (exclusive).
    """

    model_config = ConfigDict(frozen=True)

    start: datetime = Field(..., description="Period start (inclusive)")
    end: datetime = Field(..., description="Period end (exclusive)")

    @field_validator("start", "end", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        """Ensure start is before end."""
        if self.start >= self.end:
            raise ValueError(
                f"start ({self.start.isoformat()}) must be before "
                f"end ({self.end.isoformat()})"
            )
        return self

    @property
    def days(self) -> int:
        """Number of days in the range."""
        return (self.end - self.start).days


class Evidence(BaseModel):
    """Core evidence record for SOC 2 controls.

    Represents a single piece of evidence collected for audit purposes.
    Each evidence record has a unique ID, provenance hash for integrity,
    and complete metadata for auditor review.

    Attributes:
        evidence_id: Unique identifier for this evidence.
        criterion_id: SOC 2 control criterion (e.g., "CC6.1").
        evidence_type: Type of evidence (policy, log, screenshot, etc.).
        source: Source system where evidence was collected.
        title: Human-readable title.
        description: Detailed description of what this evidence demonstrates.
        content: Evidence content (for inline evidence).
        file_path: Path to evidence file (for file-based evidence).
        s3_key: S3 object key if stored in S3.
        collected_at: When the evidence was collected.
        period_start: Start of the period this evidence covers.
        period_end: End of the period this evidence covers.
        status: Current status of the evidence.
        collector_id: ID of the user/system that collected this evidence.
        metadata: Additional structured metadata.
        provenance_hash: SHA-256 hash for integrity verification.
        version: Version number for this evidence.
        supersedes_id: ID of evidence this record supersedes.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    evidence_id: UUID = Field(
        default_factory=uuid4,
        description="Unique evidence identifier",
    )
    criterion_id: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description="SOC 2 control criterion ID (e.g., 'CC6.1')",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence",
    )
    source: EvidenceSource = Field(
        ...,
        description="Source system where evidence was collected",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable evidence title",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of evidence",
    )
    content: Optional[str] = Field(
        default=None,
        max_length=1_000_000,
        description="Inline evidence content (for small text-based evidence)",
    )
    file_path: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="Local file path for file-based evidence",
    )
    s3_key: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="S3 object key if stored in S3",
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the evidence was collected",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the period this evidence covers",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the period this evidence covers",
    )
    status: EvidenceStatus = Field(
        default=EvidenceStatus.COLLECTED,
        description="Current status of the evidence",
    )
    collector_id: str = Field(
        default="system",
        max_length=256,
        description="ID of the collector (user or system)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        max_length=64,
        description="SHA-256 hash for integrity verification",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Evidence version number",
    )
    supersedes_id: Optional[UUID] = Field(
        default=None,
        description="ID of evidence this record supersedes",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )

    @field_validator("collected_at", "period_start", "period_end", mode="before")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_format(cls, v: str) -> str:
        """Validate criterion ID format (e.g., CC6.1, A1.2)."""
        v = v.strip().upper()
        if not v:
            raise ValueError("Criterion ID cannot be empty")
        return v

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 provenance hash for this evidence.

        Returns:
            SHA-256 hash string.
        """
        data = {
            "evidence_id": str(self.evidence_id),
            "criterion_id": self.criterion_id,
            "evidence_type": self.evidence_type.value,
            "source": self.source.value,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "collected_at": self.collected_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class EvidenceVersion(BaseModel):
    """Version record for evidence history tracking.

    Attributes:
        version_id: Unique version identifier.
        evidence_id: ID of the evidence this version belongs to.
        version_number: Sequential version number.
        created_at: When this version was created.
        created_by: Who created this version.
        change_summary: Summary of changes from previous version.
        provenance_hash: Hash of evidence at this version.
        s3_key: S3 key for this specific version.
        tags: Version tags (e.g., "auditor-approved").
    """

    model_config = ConfigDict(frozen=True)

    version_id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID = Field(...)
    version_number: int = Field(..., ge=1)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_by: str = Field(default="system", max_length=256)
    change_summary: str = Field(default="", max_length=1024)
    provenance_hash: str = Field(..., max_length=64)
    s3_key: Optional[str] = Field(default=None, max_length=1024)
    tags: List[str] = Field(default_factory=list)


class VersionDiff(BaseModel):
    """Difference between two evidence versions.

    Attributes:
        evidence_id: ID of the evidence being compared.
        from_version: Source version number.
        to_version: Target version number.
        changes: Dictionary of field changes.
        hash_changed: Whether provenance hash changed.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: UUID = Field(...)
    from_version: int = Field(..., ge=1)
    to_version: int = Field(..., ge=1)
    changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    hash_changed: bool = Field(default=False)


class ValidationResult(BaseModel):
    """Result of validating a piece of evidence.

    Attributes:
        evidence_id: ID of the evidence validated.
        validation_type: Type of validation performed.
        status: Pass/fail/warning status.
        message: Human-readable validation message.
        details: Additional validation details.
        validated_at: When validation was performed.
        validator_id: ID of the validator.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: UUID = Field(...)
    validation_type: str = Field(..., max_length=64)
    status: ValidationStatus = Field(...)
    message: str = Field(default="", max_length=2048)
    details: Dict[str, Any] = Field(default_factory=dict)
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    validator_id: str = Field(default="system", max_length=256)


class EvidencePackageManifest(BaseModel):
    """Manifest for an evidence package delivered to auditors.

    Attributes:
        package_id: Unique package identifier.
        package_name: Human-readable package name.
        audit_period: Date range for the audit period.
        criteria: List of criteria included in package.
        evidence_count: Total number of evidence items.
        total_size_bytes: Total size of all evidence files.
        created_at: When package was created.
        created_by: Who created the package.
        package_hash: SHA-256 hash of entire package.
        evidence_hashes: Map of evidence_id to hash.
        metadata: Additional package metadata.
    """

    model_config = ConfigDict(frozen=True)

    package_id: UUID = Field(default_factory=uuid4)
    package_name: str = Field(..., max_length=256)
    audit_period: DateRange = Field(...)
    criteria: List[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_by: str = Field(default="system", max_length=256)
    package_hash: Optional[str] = Field(default=None, max_length=64)
    evidence_hashes: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidencePackage(BaseModel):
    """Complete evidence package for auditor delivery.

    Attributes:
        manifest: Package manifest with metadata.
        evidence: List of evidence items in package.
        populations: Population files for sampling.
        s3_location: S3 location of uploaded package.
        download_url: Presigned download URL.
    """

    manifest: EvidencePackageManifest = Field(...)
    evidence: List[Evidence] = Field(default_factory=list)
    populations: Dict[str, str] = Field(default_factory=dict)
    s3_location: Optional[str] = Field(default=None)
    download_url: Optional[str] = Field(default=None)


class CollectionMetadata(BaseModel):
    """Metadata about an evidence collection run.

    Attributes:
        collection_id: Unique collection run identifier.
        criterion_id: Criterion being collected for.
        source: Source system.
        started_at: When collection started.
        completed_at: When collection completed.
        evidence_count: Number of evidence items collected.
        error_count: Number of errors during collection.
        errors: List of error messages.
        status: Collection status.
    """

    collection_id: UUID = Field(default_factory=uuid4)
    criterion_id: str = Field(...)
    source: EvidenceSource = Field(...)
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = Field(default=None)
    evidence_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    status: str = Field(default="running")


class SamplingResult(BaseModel):
    """Result of population sampling for audit.

    Attributes:
        population_id: Unique identifier for this population.
        population_name: Human-readable name.
        control_frequency: Frequency of the control.
        population_size: Total population size.
        sample_size: Number of items sampled.
        sample_items: List of sampled item identifiers.
        methodology: Sampling methodology used.
        documentation: Sampling documentation for auditor.
        created_at: When sampling was performed.
    """

    model_config = ConfigDict(frozen=True)

    population_id: UUID = Field(default_factory=uuid4)
    population_name: str = Field(..., max_length=256)
    control_frequency: ControlFrequency = Field(...)
    population_size: int = Field(..., ge=0)
    sample_size: int = Field(..., ge=0)
    sample_items: List[Any] = Field(default_factory=list)
    methodology: str = Field(default="random", max_length=64)
    documentation: str = Field(default="", max_length=4096)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
