# -*- coding: utf-8 -*-
"""
Continuous Monitoring Agent Models - AGENT-EUDR-033

Pydantic v2 models for supply chain monitoring, deforestation alert
correlation, compliance auditing, change detection, risk score tracking,
data freshness validation, and regulatory update tracking.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 Continuous Monitoring Agent (GL-EUDR-CM-033)
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 10, 11, 12, 14, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums (14)
# ---------------------------------------------------------------------------


class MonitoringScope(str, enum.Enum):
    """Scope of continuous monitoring operations."""
    SUPPLY_CHAIN = "supply_chain"
    DEFORESTATION = "deforestation"
    COMPLIANCE = "compliance"
    RISK = "risk"
    DATA_FRESHNESS = "data_freshness"
    REGULATORY = "regulatory"
    CHANGE_DETECTION = "change_detection"


class ScanStatus(str, enum.Enum):
    """Status of a monitoring scan cycle."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AlertSeverity(str, enum.Enum):
    """Severity level for monitoring alerts."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    """Lifecycle status of an alert."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class CertificationStatus(str, enum.Enum):
    """Certification validity status."""
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_RENEWAL = "pending_renewal"


class ChangeType(str, enum.Enum):
    """Type of detected change."""
    SUPPLIER_STATUS = "supplier_status"
    CERTIFICATION = "certification"
    GEOLOCATION = "geolocation"
    RISK_SCORE = "risk_score"
    COMPLIANCE_STATUS = "compliance_status"
    REGULATORY = "regulatory"
    DEFORESTATION = "deforestation"
    OWNERSHIP = "ownership"


class ChangeImpact(str, enum.Enum):
    """Impact level of a detected change."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, enum.Enum):
    """EUDR compliance verification status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    PENDING_REVIEW = "pending_review"
    EXPIRED = "expired"


class TrendDirection(str, enum.Enum):
    """Trend direction for monitored metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"


class FreshnessStatus(str, enum.Enum):
    """Data freshness classification."""
    FRESH = "fresh"
    STALE_WARNING = "stale_warning"
    STALE_CRITICAL = "stale_critical"
    UNKNOWN = "unknown"


class RegulatoryImpact(str, enum.Enum):
    """Impact level of a regulatory change."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    BREAKING = "breaking"


class RiskLevel(str, enum.Enum):
    """Risk level classification."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InvestigationStatus(str, enum.Enum):
    """Status of a triggered investigation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    CLOSED = "closed"


class AuditAction(str, enum.Enum):
    """Audit trail action types for continuous monitoring events."""
    SCAN = "scan"
    DETECT = "detect"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    VERIFY = "verify"
    ASSESS = "assess"
    TRACK = "track"
    NOTIFY = "notify"
    REFRESH = "refresh"
    CORRELATE = "correlate"
    AUDIT = "audit"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-CM-033"
AGENT_VERSION = "1.0.0"

EUDR_ARTICLES_MONITORED: List[str] = [
    "Article 4", "Article 8", "Article 10", "Article 11",
    "Article 12", "Article 14", "Article 29", "Article 31",
]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SupplierChange(BaseModel):
    """A detected change in supplier data."""
    supplier_id: str = Field(..., description="Supplier identifier")
    field_changed: str = Field(..., description="Field that changed")
    old_value: str = Field(default="", description="Previous value")
    new_value: str = Field(default="", description="New value")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False, "extra": "ignore"}


class CertificationCheck(BaseModel):
    """Certification expiry check result."""
    certification_id: str = Field(..., description="Certification identifier")
    supplier_id: str = Field(..., description="Supplier identifier")
    certification_type: str = Field(default="", description="Type of certification")
    expiry_date: Optional[datetime] = None
    days_until_expiry: int = Field(default=0, description="Days until expiry")
    status: CertificationStatus = CertificationStatus.VALID

    model_config = {"frozen": False, "extra": "ignore"}


class GeolocationShift(BaseModel):
    """Detected geolocation coordinate shift."""
    entity_id: str = Field(..., description="Entity identifier (plot or supplier)")
    original_lat: Decimal = Field(default=Decimal("0"), description="Original latitude")
    original_lon: Decimal = Field(default=Decimal("0"), description="Original longitude")
    current_lat: Decimal = Field(default=Decimal("0"), description="Current latitude")
    current_lon: Decimal = Field(default=Decimal("0"), description="Current longitude")
    drift_km: Decimal = Field(default=Decimal("0"), ge=0, description="Drift distance in km")
    is_stable: bool = Field(default=True, description="Whether within threshold")

    model_config = {"frozen": False, "extra": "ignore"}


class DeforestationCorrelation(BaseModel):
    """Correlation between deforestation alert and supply chain entity."""
    alert_id: str = Field(..., description="Deforestation alert ID")
    entity_id: str = Field(..., description="Correlated entity ID (plot/supplier)")
    entity_type: str = Field(default="plot", description="Entity type")
    distance_km: Decimal = Field(default=Decimal("0"), ge=0, description="Distance in km")
    area_hectares: Decimal = Field(default=Decimal("0"), ge=0, description="Affected area")
    confidence: Decimal = Field(default=Decimal("0"), ge=0, le=100, description="Correlation confidence")

    model_config = {"frozen": False, "extra": "ignore"}


class ComplianceCheckItem(BaseModel):
    """A single compliance check within an audit."""
    check_id: str = Field(..., description="Check identifier")
    article_reference: str = Field(..., description="EUDR article reference")
    description: str = Field(default="", description="Check description")
    status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    details: Dict[str, Any] = Field(default_factory=dict, description="Check details")

    model_config = {"frozen": False, "extra": "ignore"}


class RiskScoreSnapshot(BaseModel):
    """Point-in-time risk score snapshot for trend analysis."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entity_id: str = Field(..., description="Entity identifier")
    score: Decimal = Field(default=Decimal("0"), ge=0, le=100, description="Risk score")
    risk_level: RiskLevel = RiskLevel.LOW

    model_config = {"frozen": False, "extra": "ignore"}


class StaleEntity(BaseModel):
    """An entity with stale data."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(default="supplier", description="Entity type")
    last_updated: Optional[datetime] = None
    age_hours: Decimal = Field(default=Decimal("0"), ge=0, description="Data age in hours")
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    recommended_action: str = Field(default="refresh", description="Recommended action")

    model_config = {"frozen": False, "extra": "ignore"}


class RegulatoryUpdate(BaseModel):
    """A detected regulatory update or change."""
    update_id: str = Field(..., description="Update identifier")
    source: str = Field(default="", description="Source (eur-lex, etc.)")
    title: str = Field(default="", description="Update title")
    summary: str = Field(default="", description="Brief summary")
    published_date: Optional[datetime] = None
    impact_level: RegulatoryImpact = RegulatoryImpact.NONE
    affected_articles: List[str] = Field(default_factory=list, description="Affected EUDR articles")

    model_config = {"frozen": False, "extra": "ignore"}


class ActionRecommendation(BaseModel):
    """Recommended action from monitoring analysis."""
    action: str = Field(..., description="Action description")
    priority: str = Field(default="medium", description="Priority (low/medium/high/critical)")
    deadline_days: int = Field(default=30, ge=1, description="Suggested deadline in days")
    category: str = Field(default="general", description="Action category")

    model_config = {"frozen": False, "extra": "ignore"}


# ---------------------------------------------------------------------------
# Core Models (15+)
# ---------------------------------------------------------------------------


class SupplyChainScanRecord(BaseModel):
    """Supply chain monitoring scan record.

    Represents a complete scan cycle across suppliers, certifications,
    and geolocations within an operator's supply chain.
    """
    scan_id: str = Field(..., description="Unique scan identifier")
    operator_id: str = Field(..., description="Operator identifier")
    scan_status: ScanStatus = ScanStatus.PENDING
    scan_scope: MonitoringScope = MonitoringScope.SUPPLY_CHAIN
    suppliers_scanned: int = Field(default=0, ge=0, description="Suppliers scanned")
    changes_detected: int = Field(default=0, ge=0, description="Changes detected")
    certifications_expiring: int = Field(default=0, ge=0, description="Expiring certifications")
    geolocation_drifts: int = Field(default=0, ge=0, description="Geolocation drifts")
    supplier_changes: List[SupplierChange] = Field(default_factory=list)
    certification_checks: List[CertificationCheck] = Field(default_factory=list)
    geolocation_shifts: List[GeolocationShift] = Field(default_factory=list)
    alerts_generated: int = Field(default=0, ge=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: Decimal = Field(default=Decimal("0"), ge=0)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DeforestationMonitorRecord(BaseModel):
    """Deforestation monitoring record.

    Integrates with EUDR-020 (Deforestation Alert System) for alert
    correlation, supply chain impact assessment, and investigation triggering.
    """
    monitor_id: str = Field(..., description="Unique monitor record identifier")
    operator_id: str = Field(..., description="Operator identifier")
    alerts_checked: int = Field(default=0, ge=0, description="Alerts checked")
    correlations_found: int = Field(default=0, ge=0, description="Correlations found")
    investigations_triggered: int = Field(default=0, ge=0, description="Investigations triggered")
    total_area_affected_hectares: Decimal = Field(default=Decimal("0"), ge=0)
    correlations: List[DeforestationCorrelation] = Field(default_factory=list)
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)
    investigation_ids: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ComplianceAuditRecord(BaseModel):
    """Automated compliance audit record.

    Verifies EUDR compliance across Article 8, risk assessments,
    due diligence statements, and data freshness requirements.
    """
    audit_id: str = Field(..., description="Unique audit identifier")
    operator_id: str = Field(..., description="Operator identifier")
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    overall_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    checks_total: int = Field(default=0, ge=0)
    check_items: List[ComplianceCheckItem] = Field(default_factory=list)
    article_8_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    risk_assessment_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    due_diligence_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    recommendations: List[ActionRecommendation] = Field(default_factory=list)
    audited_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    next_audit_date: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ChangeDetectionRecord(BaseModel):
    """Change detection record across supply chain entities.

    Detects and classifies changes in supplier status, certifications,
    geolocations, risk scores, and compliance posture.
    """
    detection_id: str = Field(..., description="Unique detection identifier")
    operator_id: str = Field(..., description="Operator identifier")
    entity_id: str = Field(..., description="Affected entity identifier")
    entity_type: str = Field(default="supplier", description="Entity type")
    change_type: ChangeType = ChangeType.SUPPLIER_STATUS
    change_impact: ChangeImpact = ChangeImpact.LOW
    impact_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    description: str = Field(default="", description="Change description")
    old_state: Dict[str, Any] = Field(default_factory=dict)
    new_state: Dict[str, Any] = Field(default_factory=dict)
    recommended_actions: List[ActionRecommendation] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RiskScoreMonitorRecord(BaseModel):
    """Risk score monitoring and trend analysis record.

    Tracks risk score trends over time, detects degradation,
    and correlates risk changes with incidents.
    """
    monitor_id: str = Field(..., description="Unique monitor identifier")
    operator_id: str = Field(..., description="Operator identifier")
    entity_id: str = Field(..., description="Monitored entity identifier")
    entity_type: str = Field(default="supplier", description="Entity type")
    current_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    previous_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    score_delta: Decimal = Field(default=Decimal("0"), description="Score change")
    risk_level: RiskLevel = RiskLevel.LOW
    trend_direction: TrendDirection = TrendDirection.STABLE
    degradation_detected: bool = Field(default=False)
    trend_snapshots: List[RiskScoreSnapshot] = Field(default_factory=list)
    correlated_incidents: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[ActionRecommendation] = Field(default_factory=list)
    monitored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DataFreshnessRecord(BaseModel):
    """Data freshness validation record.

    Validates data currency across the supply chain, identifies
    stale entities, and schedules data refresh operations.
    """
    freshness_id: str = Field(..., description="Unique freshness record identifier")
    operator_id: str = Field(..., description="Operator identifier")
    entities_checked: int = Field(default=0, ge=0)
    fresh_count: int = Field(default=0, ge=0)
    stale_warning_count: int = Field(default=0, ge=0)
    stale_critical_count: int = Field(default=0, ge=0)
    freshness_percentage: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    meets_target: bool = Field(default=False)
    stale_entities: List[StaleEntity] = Field(default_factory=list)
    refresh_scheduled: int = Field(default=0, ge=0)
    refresh_schedule: List[Dict[str, Any]] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RegulatoryTrackingRecord(BaseModel):
    """Regulatory change tracking record.

    Monitors regulatory updates from EU and national sources,
    assesses impact on EUDR compliance, and notifies stakeholders.
    """
    tracking_id: str = Field(..., description="Unique tracking record identifier")
    operator_id: str = Field(..., description="Operator identifier")
    updates_found: int = Field(default=0, ge=0)
    high_impact_count: int = Field(default=0, ge=0)
    sources_checked: List[str] = Field(default_factory=list)
    regulatory_updates: List[RegulatoryUpdate] = Field(default_factory=list)
    entity_mappings: List[Dict[str, Any]] = Field(default_factory=list)
    notifications_sent: int = Field(default=0, ge=0)
    notification_channels: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class MonitoringAlert(BaseModel):
    """A monitoring alert generated by any engine."""
    alert_id: str = Field(..., description="Unique alert identifier")
    operator_id: str = Field(..., description="Operator identifier")
    source_engine: MonitoringScope = MonitoringScope.SUPPLY_CHAIN
    severity: AlertSeverity = AlertSeverity.INFO
    alert_status: AlertStatus = AlertStatus.OPEN
    title: str = Field(default="", description="Alert title")
    description: str = Field(default="", description="Alert description")
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recommended_actions: List[ActionRecommendation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class InvestigationRecord(BaseModel):
    """Investigation triggered by monitoring alerts."""
    investigation_id: str = Field(..., description="Unique investigation identifier")
    operator_id: str = Field(..., description="Operator identifier")
    trigger_alert_id: str = Field(..., description="Alert that triggered investigation")
    investigation_status: InvestigationStatus = InvestigationStatus.PENDING
    investigation_type: str = Field(default="general", description="Investigation type")
    assigned_to: Optional[str] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[ActionRecommendation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class MonitoringSummary(BaseModel):
    """Periodic monitoring summary across all engines."""
    summary_id: str = Field(..., description="Unique summary identifier")
    operator_id: str = Field(..., description="Operator identifier")
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    supply_chain_scans: int = Field(default=0, ge=0)
    deforestation_checks: int = Field(default=0, ge=0)
    compliance_audits: int = Field(default=0, ge=0)
    changes_detected: int = Field(default=0, ge=0)
    risk_monitors: int = Field(default=0, ge=0)
    freshness_checks: int = Field(default=0, ge=0)
    regulatory_checks: int = Field(default=0, ge=0)
    total_alerts: int = Field(default=0, ge=0)
    critical_alerts: int = Field(default=0, ge=0)
    investigations_opened: int = Field(default=0, ge=0)
    overall_compliance_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    overall_risk_level: RiskLevel = RiskLevel.LOW
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class AuditEntry(BaseModel):
    """An audit trail entry for continuous monitoring events."""
    entry_id: str = Field(..., description="Unique audit entry identifier")
    entity_type: str = Field(..., description="Entity type being audited")
    entity_id: str = Field(..., description="Entity identifier")
    action: AuditAction = AuditAction.SCAN
    actor: str = Field(..., description="Actor performing the action")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(BaseModel):
    """Health check response for the Continuous Monitoring Agent."""
    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}
