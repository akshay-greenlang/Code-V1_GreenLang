"""
Data Pipeline Models

Core data models for the enterprise data pipeline system.
Handles versioning, change tracking, validation results, and job management.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import hashlib
import json


class ImportStatus(str, Enum):
    """Import job status."""
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


class ReviewStatus(str, Enum):
    """Change request review status."""
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ChangeType(str, Enum):
    """Type of change to emission factor."""
    ADDED = "added"
    UPDATED = "updated"
    DEPRECATED = "deprecated"
    DELETED = "deleted"
    VALUE_CHANGED = "value_changed"
    METADATA_CHANGED = "metadata_changed"


class DataQualityTier(str, Enum):
    """Data quality tiers."""
    TIER_1 = "Tier 1 - National Average"
    TIER_2 = "Tier 2 - Technology Specific"
    TIER_3 = "Tier 3 - Facility Specific"
    TIER_4 = "Tier 4 - Continuous Monitoring"


class FactorVersion(BaseModel):
    """
    Version history for emission factors.

    Tracks all changes to emission factors over time for audit compliance.
    """
    version_id: str = Field(..., description="Unique version identifier")
    factor_id: str = Field(..., description="Emission factor ID")
    version_number: int = Field(..., description="Sequential version number")

    # Factor data snapshot
    factor_data: Dict[str, Any] = Field(..., description="Complete factor data at this version")
    previous_data: Optional[Dict[str, Any]] = Field(None, description="Previous version data")

    # Change metadata
    change_type: ChangeType
    change_summary: str = Field(..., description="Summary of what changed")
    changed_fields: List[str] = Field(default_factory=list, description="List of changed fields")

    # Provenance
    changed_by: str = Field(..., description="User or system that made the change")
    change_reason: Optional[str] = Field(None, description="Reason for change")
    source_file: Optional[str] = Field(None, description="Source YAML file")

    # Timestamps
    version_timestamp: datetime = Field(default_factory=datetime.now)
    effective_from: datetime = Field(default_factory=datetime.now)
    effective_until: Optional[datetime] = None

    # Validation
    validation_passed: bool = True
    validation_warnings: List[str] = Field(default_factory=list)

    # Audit
    data_hash: str = Field(..., description="SHA-256 hash of factor data")

    class Config:
        use_enum_values = True

    @validator('data_hash', pre=True, always=True)
    def generate_hash(cls, v, values):
        """Generate SHA-256 hash of factor data."""
        if v:
            return v

        factor_data = values.get('factor_data', {})
        data_str = json.dumps(factor_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class ChangeLog(BaseModel):
    """
    Change log entry for tracking all modifications.

    Provides audit trail for compliance and debugging.
    """
    log_id: str = Field(..., description="Unique log entry ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # What changed
    change_type: ChangeType
    factor_id: Optional[str] = None
    affected_factors: List[str] = Field(default_factory=list)

    # Change details
    summary: str = Field(..., description="Human-readable summary")
    details: Dict[str, Any] = Field(default_factory=dict)

    # Before/After
    before_value: Optional[Any] = None
    after_value: Optional[Any] = None

    # Provenance
    changed_by: str
    change_reason: Optional[str] = None
    import_job_id: Optional[str] = None

    # Approval workflow
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    review_notes: Optional[str] = None

    class Config:
        use_enum_values = True


class ValidationResult(BaseModel):
    """
    Result of data validation.

    Comprehensive validation results with detailed error reporting.
    """
    validation_id: str = Field(..., description="Unique validation ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Overall status
    is_valid: bool = Field(..., description="Overall validation result")
    quality_score: float = Field(..., ge=0, le=100, description="Data quality score 0-100")

    # Validation details
    total_records: int
    valid_records: int
    invalid_records: int
    warning_records: int

    # Rule results
    rules_passed: List[str] = Field(default_factory=list)
    rules_failed: List[str] = Field(default_factory=list)
    rules_warnings: List[str] = Field(default_factory=list)

    # Detailed errors
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

    # Metrics breakdown
    uri_validation: Dict[str, Any] = Field(default_factory=dict)
    range_validation: Dict[str, Any] = Field(default_factory=dict)
    freshness_validation: Dict[str, Any] = Field(default_factory=dict)
    unit_validation: Dict[str, Any] = Field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Processing time
    validation_duration_ms: float = 0.0


class DataQualityMetrics(BaseModel):
    """
    Comprehensive data quality metrics.

    Tracks data quality across multiple dimensions for monitoring.
    """
    metric_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Overall metrics
    total_factors: int
    overall_quality_score: float = Field(..., ge=0, le=100)

    # Completeness (0-100)
    completeness_score: float = Field(..., ge=0, le=100)
    missing_required_fields: int = 0
    missing_optional_fields: int = 0

    # Accuracy (0-100)
    accuracy_score: float = Field(..., ge=0, le=100)
    invalid_ranges: int = 0
    invalid_uris: int = 0
    invalid_dates: int = 0

    # Freshness (0-100)
    freshness_score: float = Field(..., ge=0, le=100)
    stale_factors_count: int = 0
    avg_age_days: float = 0.0
    oldest_factor_days: int = 0

    # Consistency (0-100)
    consistency_score: float = Field(..., ge=0, le=100)
    inconsistent_units: int = 0
    duplicate_factors: int = 0

    # Coverage metrics
    coverage_metrics: Dict[str, Any] = Field(default_factory=dict)

    # Source diversity
    unique_sources: int
    source_distribution: Dict[str, int] = Field(default_factory=dict)

    # Geographic coverage
    geographic_coverage: Dict[str, int] = Field(default_factory=dict)

    # Category coverage
    category_coverage: Dict[str, int] = Field(default_factory=dict)

    # Quality tier distribution
    tier_distribution: Dict[str, int] = Field(default_factory=dict)


class ImportJob(BaseModel):
    """
    Import job tracking and management.

    Tracks the entire lifecycle of an import job.
    """
    job_id: str = Field(..., description="Unique job identifier")
    job_name: str = Field(..., description="Human-readable job name")

    # Status
    status: ImportStatus = ImportStatus.PENDING
    progress_percent: float = Field(0.0, ge=0, le=100)

    # Configuration
    source_files: List[str] = Field(..., description="List of source YAML files")
    target_database: str = Field(..., description="Target database path")
    overwrite_existing: bool = False
    validate_before_import: bool = True

    # Execution metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Results
    total_factors_processed: int = 0
    successful_imports: int = 0
    failed_imports: int = 0
    skipped_imports: int = 0
    duplicate_factors: int = 0

    # Validation
    validation_result: Optional[ValidationResult] = None
    pre_import_validation_passed: bool = False

    # Errors and warnings
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

    # Rollback support
    backup_path: Optional[str] = None
    can_rollback: bool = False
    rolled_back: bool = False
    rollback_timestamp: Optional[datetime] = None

    # Provenance
    triggered_by: str = Field(..., description="User or system that triggered job")
    trigger_type: Literal["manual", "scheduled", "webhook", "api"] = "manual"

    # Logs
    log_file_path: Optional[str] = None

    class Config:
        use_enum_values = True

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_factors_processed == 0:
            return 0.0
        return (self.successful_imports / self.total_factors_processed) * 100

    @property
    def is_complete(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [
            ImportStatus.COMPLETED,
            ImportStatus.FAILED,
            ImportStatus.ROLLED_BACK
        ]


class ChangeRequest(BaseModel):
    """
    Change request for emission factor updates.

    Implements approval workflow for factor changes.
    """
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Change details
    change_type: ChangeType
    factor_id: str
    proposed_changes: Dict[str, Any]
    current_values: Dict[str, Any]

    # Justification
    change_reason: str
    supporting_documentation: List[str] = Field(default_factory=list)
    source_references: List[str] = Field(default_factory=list)

    # Requester
    requested_by: str
    requester_organization: Optional[str] = None

    # Review workflow
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    assigned_reviewer: Optional[str] = None
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    review_notes: Optional[str] = None

    # Impact assessment
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)
    affected_calculations: int = 0

    # Approval
    approved: bool = False
    approval_conditions: List[str] = Field(default_factory=list)

    # Implementation
    implemented: bool = False
    implementation_timestamp: Optional[datetime] = None
    implementation_job_id: Optional[str] = None

    class Config:
        use_enum_values = True


class URIAccessibilityCheck(BaseModel):
    """URI accessibility validation result."""
    uri: str
    accessible: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    checked_at: datetime = Field(default_factory=datetime.now)


class FactorRangeCheck(BaseModel):
    """Emission factor range validation result."""
    factor_id: str
    value: float
    unit: str
    category: str
    in_range: bool
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    deviation_percent: Optional[float] = None
    warning_threshold_exceeded: bool = False
