"""
ValueChain Intake Agent Data Models

Pydantic v2 models for multi-format data ingestion, entity resolution,
review queue management, and data quality assessment.

Key Models:
- IngestionRecord: Raw ingestion record from any source
- ResolvedEntity: Entity resolution result with confidence scoring
- ReviewQueueItem: Record awaiting human review
- DataQualityAssessment: DQI and completeness assessment
- IngestionResult: Summary of ingestion batch processing

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from pathlib import Path


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class IngestionFormat(str, Enum):
    """Supported ingestion formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    XML = "xml"
    PDF = "pdf"
    API = "api"


class SourceSystem(str, Enum):
    """Source systems for data ingestion."""
    SAP_S4HANA = "SAP_S4HANA"
    ORACLE_FUSION = "Oracle_Fusion"
    WORKDAY = "Workday"
    MANUAL_UPLOAD = "Manual_Upload"
    API = "API"
    OTHER = "Other"


class EntityType(str, Enum):
    """Entity types for resolution."""
    SUPPLIER = "supplier"
    PRODUCT = "product"
    CATEGORY = "category"
    FACILITY = "facility"


class ResolutionMethod(str, Enum):
    """Entity resolution methods."""
    EXACT_MATCH = "Exact_Match"
    FUZZY_MATCH = "Fuzzy_Match"
    MDM_LOOKUP = "MDM_Lookup"
    MANUAL_OVERRIDE = "Manual_Override"
    FAILED = "Failed"


class ReviewStatus(str, Enum):
    """Review queue item status."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"
    SPLIT = "split"


class ValidationStatus(str, Enum):
    """Validation status for records."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"


# ============================================================================
# INGESTION MODELS
# ============================================================================

class IngestionMetadata(BaseModel):
    """Metadata for ingestion tracking."""

    source_file: Optional[str] = Field(None, description="Source file path")
    source_system: SourceSystem = Field(..., description="Source system")
    ingestion_format: IngestionFormat = Field(..., description="Data format")
    ingested_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")
    ingested_by: Optional[str] = Field(None, description="User or system that ingested data")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    row_number: Optional[int] = Field(None, description="Row number in source file")
    original_data: Optional[Dict[str, Any]] = Field(None, description="Original raw data")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "source_file": "suppliers_2025.csv",
                "source_system": "Manual_Upload",
                "ingestion_format": "csv",
                "batch_id": "BATCH-20250130-001",
                "row_number": 42
            }]
        }
    }


class IngestionRecord(BaseModel):
    """
    Raw ingestion record from any source.

    This is the universal container for all ingested data before entity resolution.
    """

    record_id: str = Field(..., description="Unique record identifier")
    entity_type: EntityType = Field(..., description="Type of entity (supplier, product, etc)")
    tenant_id: str = Field(..., description="Multi-tenant identifier")

    # Core fields (flexible schema)
    entity_name: str = Field(..., min_length=1, max_length=500, description="Entity name")
    entity_identifier: Optional[str] = Field(None, description="External identifier (ERP ID, etc)")

    # Additional data (schema-less)
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional entity data")

    # Metadata
    metadata: IngestionMetadata = Field(..., description="Ingestion metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "record_id": "ING-20250130-00001",
                "entity_type": "supplier",
                "tenant_id": "tenant-acme-corp",
                "entity_name": "Global Steel Manufacturing Ltd",
                "entity_identifier": "SAP-V-12345",
                "data": {
                    "country": "US",
                    "industry": "Metals_Mining",
                    "annual_spend_usd": 5000000
                },
                "metadata": {
                    "source_system": "SAP_S4HANA",
                    "ingestion_format": "api",
                    "batch_id": "BATCH-20250130-001"
                }
            }]
        }
    }


# ============================================================================
# ENTITY RESOLUTION MODELS
# ============================================================================

class EntityMatchCandidate(BaseModel):
    """Candidate match from entity resolution."""

    canonical_id: str = Field(..., description="Canonical entity ID")
    canonical_name: str = Field(..., description="Canonical entity name")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Match confidence (0-100)")
    resolution_method: ResolutionMethod = Field(..., description="Resolution method used")
    match_details: Dict[str, Any] = Field(default_factory=dict, description="Match details")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "canonical_id": "ENT-GLOBSTEEL001",
                "canonical_name": "Global Steel Manufacturing Limited (US)",
                "confidence_score": 98.5,
                "resolution_method": "Fuzzy_Match",
                "match_details": {
                    "fuzzy_ratio": 98,
                    "token_sort_ratio": 100,
                    "lei_match": True
                }
            }]
        }
    }


class ResolvedEntity(BaseModel):
    """
    Entity resolution result with confidence scoring.

    Represents the outcome of entity resolution process.
    """

    record_id: str = Field(..., description="Original ingestion record ID")
    entity_type: EntityType = Field(..., description="Entity type")

    # Resolution result
    resolved: bool = Field(..., description="Whether entity was successfully resolved")
    canonical_id: Optional[str] = Field(None, description="Resolved canonical entity ID")
    canonical_name: Optional[str] = Field(None, description="Resolved canonical name")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Overall confidence (0-100)")

    # Resolution details
    resolution_method: ResolutionMethod = Field(..., description="Primary resolution method")
    candidates: List[EntityMatchCandidate] = Field(default_factory=list, description="All match candidates")

    # Review flag
    requires_review: bool = Field(..., description="Whether manual review is required")
    review_reason: Optional[str] = Field(None, description="Reason for manual review")

    # Metadata
    resolved_at: datetime = Field(default_factory=datetime.utcnow, description="Resolution timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "record_id": "ING-20250130-00001",
                "entity_type": "supplier",
                "resolved": True,
                "canonical_id": "ENT-GLOBSTEEL001",
                "canonical_name": "Global Steel Manufacturing Limited (US)",
                "confidence_score": 98.5,
                "resolution_method": "Fuzzy_Match",
                "requires_review": False,
                "candidates": []
            }]
        }
    }


# ============================================================================
# REVIEW QUEUE MODELS
# ============================================================================

class ReviewAction(str, Enum):
    """Available review actions."""
    APPROVE = "approve"
    REJECT = "reject"
    MERGE = "merge"
    SPLIT = "split"
    REQUEST_INFO = "request_info"


class ReviewQueueItem(BaseModel):
    """
    Record in human review queue.

    Low-confidence matches (<85%) are sent to review queue for human decision.
    """

    queue_item_id: str = Field(..., description="Unique queue item ID")
    record_id: str = Field(..., description="Original ingestion record ID")
    entity_type: EntityType = Field(..., description="Entity type")

    # Original data
    original_name: str = Field(..., description="Original entity name from source")
    original_data: Dict[str, Any] = Field(default_factory=dict, description="Original record data")

    # Resolution candidates
    candidates: List[EntityMatchCandidate] = Field(..., description="Candidate matches")
    top_candidate_score: float = Field(..., ge=0.0, le=100.0, description="Top candidate confidence")

    # Review status
    status: ReviewStatus = Field(default=ReviewStatus.PENDING, description="Review status")
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="Review priority")

    # Review reason
    review_reason: str = Field(..., description="Reason for manual review")
    additional_context: Optional[str] = Field(None, description="Additional context for reviewer")

    # Review actions
    assigned_to: Optional[str] = Field(None, description="Assigned reviewer")
    reviewed_by: Optional[str] = Field(None, description="Reviewer who actioned")
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    action_taken: Optional[ReviewAction] = Field(None, description="Action taken")
    action_details: Optional[Dict[str, Any]] = Field(None, description="Action details")
    reviewer_notes: Optional[str] = Field(None, description="Reviewer notes")

    # Resolution result (after review)
    resolved_canonical_id: Optional[str] = Field(None, description="Resolved canonical ID")
    resolved_canonical_name: Optional[str] = Field(None, description="Resolved canonical name")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Queue entry timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "queue_item_id": "QUEUE-20250130-00042",
                "record_id": "ING-20250130-00001",
                "entity_type": "supplier",
                "original_name": "Global Steel Mfg Ltd",
                "candidates": [],
                "top_candidate_score": 75.5,
                "status": "pending",
                "priority": "high",
                "review_reason": "Confidence below threshold (75.5% < 85%)"
            }]
        }
    }


# ============================================================================
# DATA QUALITY MODELS
# ============================================================================

class CompletenessAssessment(BaseModel):
    """Data completeness assessment."""

    total_fields: int = Field(..., ge=0, description="Total number of fields")
    populated_fields: int = Field(..., ge=0, description="Number of populated fields")
    completeness_score: float = Field(..., ge=0.0, le=100.0, description="Completeness % (0-100)")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing fields")
    critical_missing: List[str] = Field(default_factory=list, description="Critical missing fields")

    @model_validator(mode='after')
    def validate_completeness(self) -> 'CompletenessAssessment':
        """Validate completeness score calculation."""
        if self.total_fields > 0:
            expected = (self.populated_fields / self.total_fields) * 100
            if abs(self.completeness_score - expected) > 0.1:
                raise ValueError("Completeness score doesn't match populated/total ratio")
        return self


class DataQualityAssessment(BaseModel):
    """
    Comprehensive data quality assessment for ingested record.

    Integrates with DQI Calculator from methodologies module.
    """

    record_id: str = Field(..., description="Record identifier")

    # DQI Score (from methodologies/dqi_calculator.py)
    dqi_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Data Quality Index (0-100)")
    dqi_quality_label: Optional[str] = Field(None, description="DQI quality label")

    # Completeness
    completeness: CompletenessAssessment = Field(..., description="Completeness assessment")

    # Validation results
    validation_status: ValidationStatus = Field(..., description="Overall validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")

    # Schema validation
    schema_valid: bool = Field(..., description="Whether data passes schema validation")
    schema_errors: List[str] = Field(default_factory=list, description="Schema validation errors")

    # Data tier
    data_tier: Optional[int] = Field(None, ge=1, le=3, description="Data tier (1=primary, 2=secondary, 3=estimated)")

    # Metadata
    assessed_at: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "record_id": "ING-20250130-00001",
                "dqi_score": 85.5,
                "dqi_quality_label": "Good",
                "completeness": {
                    "total_fields": 20,
                    "populated_fields": 18,
                    "completeness_score": 90.0,
                    "missing_fields": ["lei", "vat_number"],
                    "critical_missing": []
                },
                "validation_status": "valid",
                "validation_errors": [],
                "validation_warnings": [],
                "schema_valid": True,
                "schema_errors": [],
                "data_tier": 1
            }]
        }
    }


# ============================================================================
# BATCH PROCESSING MODELS
# ============================================================================

class IngestionStatistics(BaseModel):
    """Statistics for ingestion batch."""

    total_records: int = Field(..., ge=0, description="Total records processed")
    successful: int = Field(..., ge=0, description="Successfully ingested")
    failed: int = Field(..., ge=0, description="Failed ingestion")

    # Resolution stats
    resolved_auto: int = Field(default=0, ge=0, description="Auto-resolved entities")
    sent_to_review: int = Field(default=0, ge=0, description="Sent to review queue")
    resolution_failures: int = Field(default=0, ge=0, description="Resolution failures")

    # Quality stats
    avg_dqi_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Average DQI score")
    avg_confidence: Optional[float] = Field(None, ge=0.0, le=100.0, description="Average confidence")
    avg_completeness: Optional[float] = Field(None, ge=0.0, le=100.0, description="Average completeness")

    # Performance
    processing_time_seconds: Optional[float] = Field(None, ge=0.0, description="Processing time")
    records_per_second: Optional[float] = Field(None, ge=0.0, description="Throughput (records/sec)")

    @model_validator(mode='after')
    def validate_totals(self) -> 'IngestionStatistics':
        """Validate total counts."""
        if self.successful + self.failed != self.total_records:
            raise ValueError("successful + failed must equal total_records")
        return self


class GapAnalysisReport(BaseModel):
    """Gap analysis report for missing data."""

    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation time")

    # Missing suppliers
    missing_suppliers_by_category: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Missing suppliers grouped by category"
    )

    # Missing products
    missing_products_by_supplier: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Missing products grouped by supplier"
    )

    # Quality summary
    quality_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data quality summary statistics"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for gap closure"
    )


class IngestionResult(BaseModel):
    """
    Summary result of batch ingestion processing.

    Returned by ValueChainIntakeAgent after processing a batch.
    """

    batch_id: str = Field(..., description="Batch identifier")
    tenant_id: str = Field(..., description="Tenant identifier")

    # Statistics
    statistics: IngestionStatistics = Field(..., description="Ingestion statistics")

    # Detailed results
    ingested_records: List[str] = Field(default_factory=list, description="Ingested record IDs")
    resolved_entities: List[str] = Field(default_factory=list, description="Resolved entity IDs")
    review_queue_items: List[str] = Field(default_factory=list, description="Review queue item IDs")
    failed_records: List[Dict[str, Any]] = Field(default_factory=list, description="Failed records with errors")

    # Quality assessment
    quality_summary: Dict[str, Any] = Field(default_factory=dict, description="Quality summary")

    # Gap analysis
    gap_analysis: Optional[GapAnalysisReport] = Field(None, description="Gap analysis report")

    # Metadata
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing end time")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "batch_id": "BATCH-20250130-001",
                "tenant_id": "tenant-acme-corp",
                "statistics": {
                    "total_records": 1000,
                    "successful": 950,
                    "failed": 50,
                    "resolved_auto": 900,
                    "sent_to_review": 50,
                    "avg_dqi_score": 85.5,
                    "processing_time_seconds": 120.5,
                    "records_per_second": 8.3
                },
                "started_at": "2025-01-30T10:00:00Z"
            }]
        }
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "IngestionFormat",
    "SourceSystem",
    "EntityType",
    "ResolutionMethod",
    "ReviewStatus",
    "ValidationStatus",
    "ReviewAction",

    # Ingestion Models
    "IngestionMetadata",
    "IngestionRecord",

    # Entity Resolution Models
    "EntityMatchCandidate",
    "ResolvedEntity",

    # Review Queue Models
    "ReviewQueueItem",

    # Data Quality Models
    "CompletenessAssessment",
    "DataQualityAssessment",

    # Batch Processing Models
    "IngestionStatistics",
    "GapAnalysisReport",
    "IngestionResult",
]
