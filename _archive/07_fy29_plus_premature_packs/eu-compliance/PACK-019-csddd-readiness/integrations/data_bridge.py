# -*- coding: utf-8 -*-
"""
DataBridge - AGENT-DATA Integration Bridge for PACK-019 CSDDD Readiness
==========================================================================

This module routes data intake agents for supplier questionnaires, spend data,
and other data sources needed for CSDDD due diligence. It aggregates data from
multiple AGENT-DATA agents, validates data quality, and tracks data freshness
to ensure CSDDD compliance assessments are based on reliable evidence.

Legal References:
    - Directive (EU) 2024/1760 (CSDDD), Article 6 - Identifying adverse impacts
    - Art 13 - Monitoring effectiveness of due diligence measures
    - Art 14 - Reporting and public communication
    - Data must be current, accurate, and verifiable for due diligence

DATA Agent Routing:
    Intake: DATA-001 (PDF), DATA-002 (Excel/CSV), DATA-003 (ERP),
            DATA-004 (API), DATA-008 (Supplier Questionnaire),
            DATA-009 (Spend Categorizer)
    Quality: DATA-010 (Data Quality Profiler), DATA-016 (Freshness Monitor),
             DATA-019 (Validation Rule Engine)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataSourceType(str, Enum):
    """Supported data source types for CSDDD due diligence."""

    SUPPLIER_QUESTIONNAIRE = "supplier_questionnaire"
    SPEND_DATA = "spend_data"
    ERP_EXPORT = "erp_export"
    PDF_REPORT = "pdf_report"
    EXCEL_CSV = "excel_csv"
    API_FEED = "api_feed"
    AUDIT_REPORT = "audit_report"
    GRIEVANCE_LOG = "grievance_log"

class QualityLevel(str, Enum):
    """Data quality assessment level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"
    FAILED = "failed"

class FreshnessStatus(str, Enum):
    """Data freshness status."""

    CURRENT = "current"
    RECENT = "recent"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

class DataAgentId(str, Enum):
    """AGENT-DATA identifiers relevant to CSDDD."""

    PDF_EXTRACTOR = "DATA-001"
    EXCEL_NORMALIZER = "DATA-002"
    ERP_CONNECTOR = "DATA-003"
    API_GATEWAY = "DATA-004"
    QUESTIONNAIRE_PROCESSOR = "DATA-008"
    SPEND_CATEGORIZER = "DATA-009"
    QUALITY_PROFILER = "DATA-010"
    FRESHNESS_MONITOR = "DATA-016"
    VALIDATION_ENGINE = "DATA-019"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DataBridgeConfig(BaseModel):
    """Configuration for the Data Bridge."""

    pack_id: str = Field(default="PACK-019")
    enable_provenance: bool = Field(default=True)
    freshness_threshold_days: int = Field(
        default=90, ge=1,
        description="Max days before data is considered stale",
    )
    expiry_threshold_days: int = Field(
        default=365, ge=30,
        description="Max days before data is considered expired",
    )
    min_quality_score: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum data quality score for acceptance",
    )

class DataSourceRecord(BaseModel):
    """A data source record ingested by a DATA agent."""

    source_id: str = Field(default_factory=_new_uuid)
    source_type: DataSourceType = Field(default=DataSourceType.SUPPLIER_QUESTIONNAIRE)
    agent_id: DataAgentId = Field(default=DataAgentId.QUESTIONNAIRE_PROCESSOR)
    company_id: str = Field(default="")
    file_name: Optional[str] = Field(None)
    record_count: int = Field(default=0, ge=0)
    ingested_at: datetime = Field(default_factory=utcnow)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_level: QualityLevel = Field(default=QualityLevel.UNVERIFIED)
    freshness: FreshnessStatus = Field(default=FreshnessStatus.UNKNOWN)
    validation_errors: List[str] = Field(default_factory=list)

class QuestionnaireData(BaseModel):
    """Processed supplier questionnaire data."""

    questionnaire_id: str = Field(default_factory=_new_uuid)
    company_id: str = Field(default="")
    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    submitted_at: Optional[datetime] = Field(None)
    response_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    human_rights_responses: Dict[str, Any] = Field(default_factory=dict)
    environmental_responses: Dict[str, Any] = Field(default_factory=dict)
    governance_responses: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)

class SpendDataSummary(BaseModel):
    """Aggregated spend data for CSDDD value chain analysis."""

    company_id: str = Field(default="")
    total_spend_eur: float = Field(default=0.0, ge=0.0)
    supplier_count: int = Field(default=0, ge=0)
    category_breakdown: Dict[str, float] = Field(default_factory=dict)
    country_breakdown: Dict[str, float] = Field(default_factory=dict)
    high_risk_spend_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    uncategorized_spend_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class DataAggregation(BaseModel):
    """Aggregated data from multiple sources."""

    aggregation_id: str = Field(default_factory=_new_uuid)
    company_id: str = Field(default="")
    sources_aggregated: int = Field(default=0)
    total_records: int = Field(default=0)
    quality_summary: Dict[str, int] = Field(default_factory=dict)
    freshness_summary: Dict[str, int] = Field(default_factory=dict)
    overall_quality: QualityLevel = Field(default=QualityLevel.UNVERIFIED)
    overall_freshness: FreshnessStatus = Field(default=FreshnessStatus.UNKNOWN)
    provenance_hash: str = Field(default="")

class DataQualityReport(BaseModel):
    """Data quality validation report."""

    report_id: str = Field(default_factory=_new_uuid)
    source_id: str = Field(default="")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_level: QualityLevel = Field(default=QualityLevel.UNVERIFIED)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    validation_errors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class BridgeResult(BaseModel):
    """Result of a data bridge operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# DataBridge
# ---------------------------------------------------------------------------

class DataBridge:
    """AGENT-DATA integration bridge for PACK-019 CSDDD Readiness.

    Routes data intake agents for supplier questionnaires, spend data,
    and other due diligence data sources. Validates quality, tracks
    freshness, and aggregates across sources.

    Attributes:
        config: Bridge configuration.
        _sources: Cached data source records.

    Example:
        >>> bridge = DataBridge(DataBridgeConfig())
        >>> result = bridge.get_questionnaire_data("company_123")
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[DataBridgeConfig] = None) -> None:
        """Initialize DataBridge."""
        self.config = config or DataBridgeConfig()
        self._sources: Dict[str, List[DataSourceRecord]] = {}
        logger.info("DataBridge initialized (pack=%s)", self.config.pack_id)

    def get_questionnaire_data(
        self,
        company_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BridgeResult:
        """Get supplier questionnaire data via DATA-008.

        Args:
            company_id: Company identifier.
            context: Optional context with pre-loaded questionnaire data.

        Returns:
            BridgeResult with status and records processed.
        """
        result = BridgeResult(started_at=utcnow())
        ctx = context or {}

        try:
            questionnaires = ctx.get("questionnaires", [])
            records: List[DataSourceRecord] = []

            for q in questionnaires:
                record = DataSourceRecord(
                    source_type=DataSourceType.SUPPLIER_QUESTIONNAIRE,
                    agent_id=DataAgentId.QUESTIONNAIRE_PROCESSOR,
                    company_id=company_id,
                    record_count=q.get("response_count", 1),
                    quality_score=q.get("quality_score", 0.5),
                )
                record.quality_level = self._score_to_quality(record.quality_score)
                records.append(record)

            self._sources.setdefault(company_id, []).extend(records)
            result.records_processed = len(records)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "company_id": company_id,
                    "questionnaire_count": len(records),
                })

            logger.info(
                "Questionnaire data loaded for %s: %d records",
                company_id,
                len(records),
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Questionnaire data retrieval failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_spend_data(
        self,
        company_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SpendDataSummary:
        """Get spend data analysis via DATA-009 Spend Categorizer.

        Args:
            company_id: Company identifier.
            context: Optional context with pre-loaded spend data.

        Returns:
            SpendDataSummary with category and country breakdowns.
        """
        ctx = context or {}
        spend_data = ctx.get("spend_data", {})

        total_spend = spend_data.get("total_spend_eur", 0.0)
        categories = spend_data.get("categories", {})
        countries = spend_data.get("countries", {})
        supplier_count = spend_data.get("supplier_count", 0)

        # Calculate high-risk spend percentage
        high_risk_countries = {"CN", "BD", "MM", "PK", "ET", "CD"}
        high_risk_spend = sum(
            v for k, v in countries.items() if k in high_risk_countries
        )
        high_risk_pct = (
            round(high_risk_spend / total_spend * 100, 1) if total_spend > 0 else 0.0
        )

        # Calculate uncategorized percentage
        categorized = sum(categories.values())
        uncategorized_pct = (
            round((1.0 - categorized / total_spend) * 100, 1)
            if total_spend > 0 and categorized <= total_spend
            else 0.0
        )

        summary = SpendDataSummary(
            company_id=company_id,
            total_spend_eur=total_spend,
            supplier_count=supplier_count,
            category_breakdown=categories,
            country_breakdown=countries,
            high_risk_spend_pct=high_risk_pct,
            uncategorized_spend_pct=max(uncategorized_pct, 0.0),
        )

        logger.info(
            "Spend data for %s: EUR %.2f, %d suppliers, %.1f%% high-risk",
            company_id,
            total_spend,
            supplier_count,
            high_risk_pct,
        )
        return summary

    def aggregate_data_sources(
        self,
        sources: List[Dict[str, Any]],
    ) -> DataAggregation:
        """Aggregate data from multiple CSDDD data sources.

        Args:
            sources: List of source dicts with keys:
                source_type, record_count, quality_score, days_since_update.

        Returns:
            DataAggregation with quality and freshness summaries.
        """
        quality_counts: Dict[str, int] = {
            QualityLevel.HIGH.value: 0,
            QualityLevel.MEDIUM.value: 0,
            QualityLevel.LOW.value: 0,
            QualityLevel.UNVERIFIED.value: 0,
        }
        freshness_counts: Dict[str, int] = {
            FreshnessStatus.CURRENT.value: 0,
            FreshnessStatus.RECENT.value: 0,
            FreshnessStatus.STALE.value: 0,
            FreshnessStatus.EXPIRED.value: 0,
        }

        total_records = 0
        quality_scores: List[float] = []

        for src in sources:
            record_count = src.get("record_count", 0)
            total_records += record_count

            q_score = src.get("quality_score", 0.0)
            quality_scores.append(q_score)
            q_level = self._score_to_quality(q_score)
            quality_counts[q_level.value] = quality_counts.get(q_level.value, 0) + 1

            days = src.get("days_since_update", 999)
            f_status = self._days_to_freshness(days)
            freshness_counts[f_status.value] = (
                freshness_counts.get(f_status.value, 0) + 1
            )

        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )
        overall_quality = self._score_to_quality(avg_quality)

        # Overall freshness is the worst freshness
        if freshness_counts.get(FreshnessStatus.EXPIRED.value, 0) > 0:
            overall_freshness = FreshnessStatus.EXPIRED
        elif freshness_counts.get(FreshnessStatus.STALE.value, 0) > 0:
            overall_freshness = FreshnessStatus.STALE
        elif freshness_counts.get(FreshnessStatus.RECENT.value, 0) > 0:
            overall_freshness = FreshnessStatus.RECENT
        elif freshness_counts.get(FreshnessStatus.CURRENT.value, 0) > 0:
            overall_freshness = FreshnessStatus.CURRENT
        else:
            overall_freshness = FreshnessStatus.UNKNOWN

        aggregation = DataAggregation(
            sources_aggregated=len(sources),
            total_records=total_records,
            quality_summary=quality_counts,
            freshness_summary=freshness_counts,
            overall_quality=overall_quality,
            overall_freshness=overall_freshness,
        )
        aggregation.provenance_hash = _compute_hash(aggregation)

        logger.info(
            "Aggregated %d sources, %d records (quality=%s, freshness=%s)",
            len(sources),
            total_records,
            overall_quality.value,
            overall_freshness.value,
        )
        return aggregation

    def validate_data_quality(
        self,
        data: Dict[str, Any],
    ) -> DataQualityReport:
        """Validate data quality for CSDDD due diligence requirements.

        Uses deterministic scoring (zero-hallucination).

        Args:
            data: Data dict with keys: records, completeness_fields,
                  expected_format, source_type.

        Returns:
            DataQualityReport with dimensional scores and recommendations.
        """
        records = data.get("records", [])
        required_fields = data.get("required_fields", [])
        total_fields = len(required_fields) * len(records) if records else 0

        # Completeness: percentage of required fields present
        filled_fields = 0
        for record in records:
            for field in required_fields:
                if record.get(field) is not None and record.get(field) != "":
                    filled_fields += 1

        completeness = (
            round(filled_fields / total_fields * 100, 1) if total_fields > 0 else 0.0
        )

        # Accuracy: percentage of records with no validation errors
        error_records = data.get("error_record_count", 0)
        accuracy = (
            round((1.0 - error_records / len(records)) * 100, 1)
            if records else 0.0
        )

        # Consistency: percentage of records matching expected format
        consistent_records = data.get("consistent_record_count", len(records))
        consistency = (
            round(consistent_records / len(records) * 100, 1) if records else 0.0
        )

        # Timeliness: based on data age
        days_old = data.get("days_since_update", 0)
        if days_old <= 30:
            timeliness = 100.0
        elif days_old <= 90:
            timeliness = 80.0
        elif days_old <= 180:
            timeliness = 50.0
        elif days_old <= 365:
            timeliness = 25.0
        else:
            timeliness = 0.0

        # Overall quality score (equal weights)
        quality_score = round(
            (completeness + accuracy + consistency + timeliness) / 400.0, 3
        )
        quality_level = self._score_to_quality(quality_score)

        # Build validation errors
        validation_errors: List[str] = []
        if completeness < 80.0:
            validation_errors.append(
                f"Completeness below threshold: {completeness}% (min 80%)"
            )
        if accuracy < 90.0:
            validation_errors.append(
                f"Accuracy below threshold: {accuracy}% (min 90%)"
            )
        if timeliness < 50.0:
            validation_errors.append(
                f"Data is {days_old} days old; consider refreshing"
            )

        # Build recommendations
        recommendations: List[str] = []
        if completeness < 80.0:
            recommendations.append("Fill missing required fields in supplier records")
        if accuracy < 90.0:
            recommendations.append("Review and correct data validation errors")
        if timeliness < 50.0:
            recommendations.append("Request updated data from suppliers")

        report = DataQualityReport(
            quality_score=quality_score,
            quality_level=quality_level,
            completeness_pct=completeness,
            accuracy_pct=accuracy,
            consistency_pct=consistency,
            timeliness_pct=timeliness,
            validation_errors=validation_errors,
            recommendations=recommendations,
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Data quality: score=%.3f, level=%s (comp=%.1f%%, acc=%.1f%%)",
            quality_score,
            quality_level.value,
            completeness,
            accuracy,
        )
        return report

    def get_data_freshness(
        self,
        source: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get data freshness status for a specific source.

        Args:
            source: Source dict with keys: source_type, last_updated,
                    days_since_update.

        Returns:
            Dict with freshness status, age, and recommendations.
        """
        days = source.get("days_since_update", 999)
        freshness = self._days_to_freshness(days)

        recommendations: List[str] = []
        if freshness == FreshnessStatus.STALE:
            recommendations.append(
                f"Data is {days} days old; update within "
                f"{self.config.freshness_threshold_days} days for compliance"
            )
        elif freshness == FreshnessStatus.EXPIRED:
            recommendations.append(
                f"Data expired ({days} days old); immediate refresh required "
                f"for CSDDD due diligence validity"
            )

        return {
            "source_type": source.get("source_type", "unknown"),
            "days_since_update": days,
            "freshness_status": freshness.value,
            "is_acceptable": freshness in (
                FreshnessStatus.CURRENT,
                FreshnessStatus.RECENT,
            ),
            "recommendations": recommendations,
            "provenance_hash": _compute_hash(source),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_to_quality(self, score: float) -> QualityLevel:
        """Map quality score to quality level."""
        if score >= 0.8:
            return QualityLevel.HIGH
        if score >= 0.6:
            return QualityLevel.MEDIUM
        if score >= 0.3:
            return QualityLevel.LOW
        return QualityLevel.UNVERIFIED

    def _days_to_freshness(self, days: int) -> FreshnessStatus:
        """Map days since update to freshness status."""
        if days <= 30:
            return FreshnessStatus.CURRENT
        if days <= self.config.freshness_threshold_days:
            return FreshnessStatus.RECENT
        if days <= self.config.expiry_threshold_days:
            return FreshnessStatus.STALE
        return FreshnessStatus.EXPIRED
