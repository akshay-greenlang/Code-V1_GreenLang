# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Configuration - EU Deforestation Regulation Compliance Platform

Application configuration using Pydantic BaseSettings with EUDR_APP_ env prefix.
Defines all enumerations, thresholds, and operational parameters for the EUDR
compliance pipeline including DDS management, satellite risk assessment,
document verification, and EU Information System connectivity.

Environment variables are loaded with the EUDR_APP_ prefix:
    EUDR_APP_DATABASE_URL=postgresql://...
    EUDR_APP_REDIS_URL=redis://...
    EUDR_APP_PIPELINE_MAX_CONCURRENT=20

Example:
    >>> from services.config import EUDRAppConfig
    >>> config = EUDRAppConfig()
    >>> print(config.database_url)

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EUDRCommodity(str, Enum):
    """Seven commodities covered by EU Regulation 2023/1115 (EUDR).

    These are the commodity groups whose production may not be linked
    to deforestation or forest degradation after 31 December 2020.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOY = "soy"
    WOOD = "wood"


class RiskLevel(str, Enum):
    """Risk classification levels for EUDR compliance assessment.

    LOW:       Minimal deforestation risk; simplified due diligence.
    STANDARD:  Normal risk; standard due diligence required.
    HIGH:      Elevated risk; enhanced due diligence and monitoring.
    CRITICAL:  Severe risk; immediate action required, may block trade.
    """

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class DDSStatus(str, Enum):
    """Due Diligence Statement lifecycle statuses per EUDR Articles 4 and 9.

    DRAFT:     Initial creation, not yet reviewed.
    REVIEW:    Under internal review before validation.
    VALIDATED: Passed all internal checks, ready for submission.
    SUBMITTED: Submitted to EU Information System.
    ACCEPTED:  Accepted by the competent authority.
    REJECTED:  Rejected by the competent authority; amendment required.
    AMENDED:   Amended after rejection and resubmitted.
    """

    DRAFT = "draft"
    REVIEW = "review"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDED = "amended"


class PipelineStage(str, Enum):
    """Five-stage EUDR compliance pipeline stages.

    INTAKE:                 Supplier data normalization and validation.
    GEO_VALIDATION:         Plot coordinate validation via AGENT-DATA-005.
    DEFORESTATION_RISK:     Satellite risk assessment via AGENT-DATA-007.
    DOCUMENT_VERIFICATION:  Compliance document verification.
    DDS_REPORTING:          DDS generation and EU submission.
    """

    INTAKE = "intake"
    GEO_VALIDATION = "geo_validation"
    DEFORESTATION_RISK = "deforestation_risk"
    DOCUMENT_VERIFICATION = "document_verification"
    DDS_REPORTING = "dds_reporting"


class PipelineStatus(str, Enum):
    """Overall pipeline run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentType(str, Enum):
    """Types of compliance documents for EUDR verification."""

    CERTIFICATE = "CERTIFICATE"
    PERMIT = "PERMIT"
    LAND_TITLE = "LAND_TITLE"
    INVOICE = "INVOICE"
    TRANSPORT = "TRANSPORT"
    OTHER = "OTHER"


class VerificationStatus(str, Enum):
    """Document verification outcome."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    EXPIRED = "expired"


class ComplianceStatus(str, Enum):
    """Supplier compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"


class ProcurementStatus(str, Enum):
    """Procurement record status."""

    DRAFT = "draft"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class SatelliteAssessmentStatus(str, Enum):
    """Status of satellite deforestation assessment for a plot."""

    NOT_ASSESSED = "not_assessed"
    IN_PROGRESS = "in_progress"
    CLEAR = "clear"
    RISK_DETECTED = "risk_detected"
    DEFORESTATION_CONFIRMED = "deforestation_confirmed"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------


class EUDRAppConfig(BaseSettings):
    """Central configuration for the GL-EUDR-APP platform.

    All settings can be overridden via environment variables with the
    EUDR_APP_ prefix. For example, EUDR_APP_DATABASE_URL overrides
    the database_url field.

    Attributes:
        database_url: PostgreSQL connection string.
        redis_url: Redis connection string for caching.
        pipeline_max_concurrent: Maximum concurrent pipeline runs.
        pipeline_retry_max: Maximum stage retry attempts.
        pipeline_timeout_seconds: Per-stage timeout in seconds.
        dds_reference_prefix: Prefix for DDS reference numbers.
        dds_auto_submit: Whether to auto-submit validated DDS to EU system.
        eu_system_endpoint: EU Information System API endpoint.
        eu_system_api_key: API key for EU Information System.
        ndvi_change_threshold: NDVI change below which deforestation flagged.
        deforestation_cutoff_date: EUDR cutoff date (2020-12-31).
        satellite_cache_days: Days to cache satellite assessment results.
        high_risk_countries: ISO-3 codes of high deforestation risk countries.
        risk_threshold_high: Score threshold for HIGH risk classification.
        risk_threshold_critical: Score threshold for CRITICAL risk classification.
        max_upload_size_mb: Maximum document upload size in megabytes.
        allowed_extensions: Permitted file extensions for document uploads.
    """

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://localhost:5432/eudr_app",
        description="PostgreSQL database connection URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching and queues",
    )

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------
    pipeline_max_concurrent: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent pipeline runs",
    )
    pipeline_retry_max: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts per pipeline stage",
    )
    pipeline_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Per-stage timeout in seconds",
    )

    # -----------------------------------------------------------------------
    # DDS Configuration
    # -----------------------------------------------------------------------
    dds_reference_prefix: str = Field(
        default="EUDR",
        description="Prefix for DDS reference numbers",
    )
    dds_auto_submit: bool = Field(
        default=False,
        description="Auto-submit validated DDS to EU Information System",
    )
    eu_system_endpoint: str = Field(
        default="https://eudr.ec.europa.eu/api/v1",
        description="EU Information System API endpoint",
    )
    eu_system_api_key: str = Field(
        default="",
        description="API key for EU Information System authentication",
    )

    # -----------------------------------------------------------------------
    # Satellite / Deforestation
    # -----------------------------------------------------------------------
    ndvi_change_threshold: float = Field(
        default=-0.15,
        ge=-1.0,
        le=0.0,
        description="NDVI change threshold for deforestation detection",
    )
    deforestation_cutoff_date: str = Field(
        default="2020-12-31",
        description="EUDR deforestation cutoff date (ISO format)",
    )
    satellite_cache_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to cache satellite assessment results",
    )

    # -----------------------------------------------------------------------
    # Risk Assessment
    # -----------------------------------------------------------------------
    high_risk_countries: List[str] = Field(
        default_factory=lambda: [
            "BRA", "IDN", "COD", "COG", "CMR",
            "MYS", "PNG", "BOL", "PER", "COL",
        ],
        description="ISO-3166 alpha-3 codes of high deforestation risk countries",
    )
    risk_threshold_high: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Score threshold for HIGH risk classification",
    )
    risk_threshold_critical: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Score threshold for CRITICAL risk classification",
    )
    risk_weight_satellite: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight of satellite risk in overall score",
    )
    risk_weight_country: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight of country risk in overall score",
    )
    risk_weight_supplier: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight of supplier risk in overall score",
    )
    risk_weight_document: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight of document risk in overall score",
    )

    # -----------------------------------------------------------------------
    # Document Verification
    # -----------------------------------------------------------------------
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum document upload size in megabytes",
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".jpg", ".png", ".xlsx", ".csv", ".xml",
        ],
        description="Permitted file extensions for document uploads",
    )
    document_retention_days: int = Field(
        default=3650,
        ge=365,
        description="Document retention period in days (EUDR requires 5 years)",
    )

    # -----------------------------------------------------------------------
    # Logging / Observability
    # -----------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Application log level",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics collection",
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry distributed tracing",
    )

    model_config = {
        "env_prefix": "EUDR_APP_",
        "case_sensitive": False,
        "extra": "ignore",
    }
