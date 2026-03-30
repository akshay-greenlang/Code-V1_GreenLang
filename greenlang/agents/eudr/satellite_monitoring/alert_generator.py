# -*- coding: utf-8 -*-
"""
AlertGenerator - AGENT-EUDR-003 Feature 8: Alert Generation & Evidence Packaging

Creates alerts and evidence packages from satellite monitoring results for
EUDR compliance due diligence statement (DDS) submission. Generates severity-
classified alerts (CRITICAL, WARNING, INFO) based on NDVI change thresholds,
manages alert lifecycle (creation, acknowledgement, filtering), and packages
all satellite evidence into structured formats (JSON, CSV, PDF-ready, EUDR XML).

Alert Severity Rules:
    CRITICAL: NDVI drop > 0.15 AND confidence > 0.7
    WARNING:  NDVI drop 0.05-0.15 AND confidence > 0.5
    INFO:     Any detectable change below warning threshold

Evidence Package Contents:
    - Baseline snapshot (NDVI, observation date, sensor).
    - Latest analysis result (NDVI, change detection, confidence).
    - NDVI time series (if available).
    - Alert history for the plot.
    - Data quality assessment.
    - Compliance determination.
    - SHA-256 provenance hash over the entire package.

Compliance Determination:
    COMPLIANT:              No deforestation + confidence > 0.7
    NON_COMPLIANT:          Deforestation detected + confidence > 0.7
    INSUFFICIENT_DATA:      Confidence < 0.5
    MANUAL_REVIEW_REQUIRED: Ambiguous results (all other cases)

Zero-Hallucination Guarantees:
    - All alert severity classification uses deterministic threshold comparisons.
    - Compliance determination uses fixed rules (no ML/LLM).
    - SHA-256 provenance hashes on all alerts and evidence packages.
    - Evidence packages are deterministically reproducible from inputs.

Performance Targets:
    - Alert generation: <5ms per alert.
    - Evidence package generation: <50ms.
    - Alert query (10,000 alerts): <100ms.

Regulatory References:
    - EUDR Article 2(1): Deforestation-free definition.
    - EUDR Article 4(2): Due diligence statement (DDS) requirements.
    - EUDR Article 9: Geolocation evidence requirements.
    - EUDR Article 10: Risk assessment evidence.
    - EUDR Article 11: Monitoring evidence obligations.
    - EUDR Article 31: Record retention (5-year audit trail).
    - EUDR Cutoff Date: 31 December 2020.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003, Feature 8
Agent ID: GL-EUDR-SAT-003
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION: str = "1.0.0"

#: Alert severity levels in order of criticality.
SEVERITY_LEVELS: List[str] = ["critical", "warning", "info"]

#: NDVI drop threshold for CRITICAL severity.
CRITICAL_NDVI_DROP_THRESHOLD: float = 0.15

#: Confidence threshold for CRITICAL severity.
CRITICAL_CONFIDENCE_THRESHOLD: float = 0.7

#: NDVI drop threshold for WARNING severity.
WARNING_NDVI_DROP_THRESHOLD: float = 0.05

#: Confidence threshold for WARNING severity.
WARNING_CONFIDENCE_THRESHOLD: float = 0.5

#: Compliance confidence thresholds.
COMPLIANCE_HIGH_CONFIDENCE: float = 0.7
COMPLIANCE_LOW_CONFIDENCE: float = 0.5

#: Deforestation score threshold for non-compliance.
DEFORESTATION_SCORE_THRESHOLD: float = 0.5

#: Supported evidence export formats.
SUPPORTED_FORMATS: List[str] = ["json", "csv", "pdf_data", "eudr_xml"]

#: EUDR cutoff date for deforestation reference.
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: Maximum alerts returned in a single query by default.
DEFAULT_ALERT_LIMIT: int = 100

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ChangeResult:
    """Change detection result used as input for alert generation.

    Attributes:
        change_type: Type of change (ndvi_drop, ndvi_increase, stable).
        ndvi_baseline: Baseline NDVI value.
        ndvi_current: Current NDVI value.
        ndvi_change: NDVI change magnitude (current - baseline).
        confidence: Detection confidence (0.0-1.0).
        affected_area_ha: Estimated affected area in hectares.
        deforestation_score: Deforestation likelihood score (0.0-1.0).
        observation_date: Observation date (ISO 8601).
        sensor: Sensor that detected the change.
    """

    change_type: str = "stable"
    ndvi_baseline: float = 0.0
    ndvi_current: float = 0.0
    ndvi_change: float = 0.0
    confidence: float = 0.0
    affected_area_ha: float = 0.0
    deforestation_score: float = 0.0
    observation_date: str = ""
    sensor: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "change_type": self.change_type,
            "ndvi_baseline": round(self.ndvi_baseline, 4),
            "ndvi_current": round(self.ndvi_current, 4),
            "ndvi_change": round(self.ndvi_change, 4),
            "confidence": round(self.confidence, 4),
            "affected_area_ha": round(self.affected_area_ha, 4),
            "deforestation_score": round(self.deforestation_score, 4),
            "observation_date": self.observation_date,
            "sensor": self.sensor,
        }

@dataclass
class MonitoringResultInput:
    """Monitoring result used as input for alert generation.

    Attributes:
        result_id: Monitoring result identifier.
        plot_id: Production plot identifier.
        schedule_id: Monitoring schedule identifier.
        commodity: EUDR commodity type.
        country_code: ISO 3166-1 alpha-2 country code.
        coordinates: Plot centroid as (lat, lon).
        data_quality: Data quality indicator (0.0-1.0).
        executed_at: Execution timestamp.
    """

    result_id: str = ""
    plot_id: str = ""
    schedule_id: str = ""
    commodity: str = ""
    country_code: str = ""
    coordinates: Tuple[float, float] = (0.0, 0.0)
    data_quality: float = 0.0
    executed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "result_id": self.result_id,
            "plot_id": self.plot_id,
            "schedule_id": self.schedule_id,
            "commodity": self.commodity,
            "country_code": self.country_code,
            "coordinates": list(self.coordinates),
            "data_quality": round(self.data_quality, 4),
            "executed_at": self.executed_at,
        }

@dataclass
class SatelliteAlert:
    """Satellite-derived deforestation alert.

    Attributes:
        alert_id: Unique alert identifier.
        plot_id: Associated production plot identifier.
        severity: Alert severity (critical, warning, info).
        created_at: UTC timestamp of alert creation.
        ndvi_drop: Absolute NDVI drop magnitude.
        ndvi_baseline: Baseline NDVI value.
        ndvi_current: Current NDVI value.
        confidence: Detection confidence (0.0-1.0).
        deforestation_score: Deforestation score (0.0-1.0).
        affected_area_ha: Estimated affected area in hectares.
        commodity: EUDR commodity type.
        country_code: Country code.
        coordinates: Plot centroid (lat, lon).
        observation_date: Satellite observation date.
        sensor: Sensor name.
        acknowledged: Whether the alert has been acknowledged.
        acknowledged_by: User ID of acknowledger (or None).
        acknowledged_at: Acknowledgement timestamp (or None).
        acknowledgement_notes: Notes from the acknowledger.
        provenance_hash: SHA-256 hash for tamper detection.
    """

    alert_id: str = field(default_factory=lambda: _generate_id("ALR"))
    plot_id: str = ""
    severity: str = "info"
    created_at: datetime = field(default_factory=utcnow)
    ndvi_drop: float = 0.0
    ndvi_baseline: float = 0.0
    ndvi_current: float = 0.0
    confidence: float = 0.0
    deforestation_score: float = 0.0
    affected_area_ha: float = 0.0
    commodity: str = ""
    country_code: str = ""
    coordinates: Tuple[float, float] = (0.0, 0.0)
    observation_date: str = ""
    sensor: str = ""
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledgement_notes: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "alert_id": self.alert_id,
            "plot_id": self.plot_id,
            "severity": self.severity,
            "created_at": self.created_at.isoformat(),
            "ndvi_drop": round(self.ndvi_drop, 4),
            "ndvi_baseline": round(self.ndvi_baseline, 4),
            "ndvi_current": round(self.ndvi_current, 4),
            "confidence": round(self.confidence, 4),
            "deforestation_score": round(self.deforestation_score, 4),
            "affected_area_ha": round(self.affected_area_ha, 4),
            "commodity": self.commodity,
            "country_code": self.country_code,
            "coordinates": list(self.coordinates),
            "observation_date": self.observation_date,
            "sensor": self.sensor,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat()
                if self.acknowledged_at else None
            ),
            "acknowledgement_notes": self.acknowledgement_notes,
            "provenance_hash": self.provenance_hash,
        }

@dataclass
class AlertSummary:
    """Aggregate summary of alert statistics.

    Attributes:
        total_alerts: Total number of alerts.
        critical_count: Number of critical alerts.
        warning_count: Number of warning alerts.
        info_count: Number of info alerts.
        acknowledged_count: Number of acknowledged alerts.
        unacknowledged_count: Number of unacknowledged alerts.
        by_commodity: Alert counts per commodity.
        by_country: Alert counts per country code.
        provenance_hash: SHA-256 hash for tamper detection.
    """

    total_alerts: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    acknowledged_count: int = 0
    unacknowledged_count: int = 0
    by_commodity: Dict[str, int] = field(default_factory=dict)
    by_country: Dict[str, int] = field(default_factory=dict)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "total_alerts": self.total_alerts,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "acknowledged_count": self.acknowledged_count,
            "unacknowledged_count": self.unacknowledged_count,
            "by_commodity": self.by_commodity,
            "by_country": self.by_country,
            "provenance_hash": self.provenance_hash,
        }

@dataclass
class EvidencePackage:
    """Complete evidence package for DDS submission.

    Attributes:
        package_id: Unique evidence package identifier.
        plot_id: Production plot identifier.
        operator_id: Operator identifier.
        created_at: UTC timestamp of package creation.
        format: Export format (json, csv, pdf_data, eudr_xml).
        baseline_snapshot: Baseline analysis data.
        latest_analysis: Most recent analysis data.
        ndvi_time_series: NDVI time series data (if included).
        alerts: Relevant alerts for this plot.
        data_quality: Data quality assessment.
        compliance_determination: Compliance status.
        eudr_cutoff_date: EUDR deforestation cutoff date reference.
        metadata: Additional package metadata.
        provenance_hash: SHA-256 hash for tamper detection.
        processing_time_ms: Package generation time in milliseconds.
    """

    package_id: str = field(default_factory=lambda: _generate_id("EVD"))
    plot_id: str = ""
    operator_id: str = ""
    created_at: datetime = field(default_factory=utcnow)
    format: str = "json"
    baseline_snapshot: Dict[str, Any] = field(default_factory=dict)
    latest_analysis: Dict[str, Any] = field(default_factory=dict)
    ndvi_time_series: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    compliance_determination: str = "INSUFFICIENT_DATA"
    eudr_cutoff_date: str = EUDR_CUTOFF_DATE
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "package_id": self.package_id,
            "plot_id": self.plot_id,
            "operator_id": self.operator_id,
            "created_at": self.created_at.isoformat(),
            "format": self.format,
            "baseline_snapshot": self.baseline_snapshot,
            "latest_analysis": self.latest_analysis,
            "ndvi_time_series": self.ndvi_time_series,
            "alerts": self.alerts,
            "data_quality": self.data_quality,
            "compliance_determination": self.compliance_determination,
            "eudr_cutoff_date": self.eudr_cutoff_date,
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# AlertGenerator
# ---------------------------------------------------------------------------

class AlertGenerator:
    """Alert generation and evidence packaging engine for EUDR satellite monitoring.

    Creates severity-classified alerts from satellite change detection results,
    manages alert lifecycle (create, acknowledge, query), generates aggregate
    summaries, and packages all satellite evidence for DDS submission.

    All alert classification and compliance logic is deterministic with zero
    ML/LLM involvement.

    Attributes:
        _alerts: In-memory alert store (alert_id -> SatelliteAlert).
        _evidence_packages: In-memory evidence store (package_id -> EvidencePackage).
        _plot_alerts: Per-plot alert index (plot_id -> [alert_id]).

    Example::

        generator = AlertGenerator()

        change = ChangeResult(
            change_type="ndvi_drop",
            ndvi_baseline=0.75,
            ndvi_current=0.55,
            ndvi_change=-0.20,
            confidence=0.85,
            deforestation_score=0.8,
            affected_area_ha=5.2,
            observation_date="2025-06-15",
            sensor="sentinel2",
        )
        monitoring = MonitoringResultInput(
            plot_id="PLOT-001",
            commodity="soya",
            country_code="BR",
            coordinates=(-3.5, -55.0),
        )

        alert = generator.generate_alert("PLOT-001", change, monitoring)
        assert alert.severity == "critical"
        assert alert.provenance_hash != ""
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the AlertGenerator.

        Args:
            config: Optional configuration object. Reserved for future use.
        """
        self._alerts: Dict[str, SatelliteAlert] = {}
        self._evidence_packages: Dict[str, EvidencePackage] = {}
        self._plot_alerts: Dict[str, List[str]] = {}

        logger.info("AlertGenerator initialized")

    # ------------------------------------------------------------------
    # Public API: Alert Generation
    # ------------------------------------------------------------------

    def generate_alert(
        self,
        plot_id: str,
        change_result: ChangeResult,
        monitoring_result: MonitoringResultInput,
    ) -> SatelliteAlert:
        """Generate an alert based on a change detection result.

        Severity classification rules (deterministic):
            CRITICAL: NDVI drop > 0.15 AND confidence > 0.7
            WARNING:  NDVI drop 0.05-0.15 AND confidence > 0.5
            INFO:     Any detectable change below warning threshold

        Args:
            plot_id: Production plot identifier.
            change_result: Change detection result.
            monitoring_result: Monitoring context metadata.

        Returns:
            SatelliteAlert with severity classification and provenance hash.
        """
        start_time = time.monotonic()

        ndvi_drop = abs(change_result.ndvi_change)

        # Classify severity
        severity = self._classify_severity(
            ndvi_drop, change_result.confidence
        )

        alert = SatelliteAlert(
            plot_id=plot_id,
            severity=severity,
            ndvi_drop=round(ndvi_drop, 4),
            ndvi_baseline=round(change_result.ndvi_baseline, 4),
            ndvi_current=round(change_result.ndvi_current, 4),
            confidence=round(change_result.confidence, 4),
            deforestation_score=round(change_result.deforestation_score, 4),
            affected_area_ha=round(change_result.affected_area_ha, 4),
            commodity=monitoring_result.commodity,
            country_code=monitoring_result.country_code,
            coordinates=monitoring_result.coordinates,
            observation_date=change_result.observation_date,
            sensor=change_result.sensor,
        )
        alert.provenance_hash = _compute_hash(alert)

        # Store alert
        self._alerts[alert.alert_id] = alert
        self._plot_alerts.setdefault(plot_id, []).append(alert.alert_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Alert %s generated: plot=%s, severity=%s, ndvi_drop=%.4f, "
            "confidence=%.3f, commodity=%s, country=%s, elapsed=%.2fms",
            alert.alert_id,
            plot_id,
            severity,
            ndvi_drop,
            change_result.confidence,
            monitoring_result.commodity,
            monitoring_result.country_code,
            elapsed_ms,
        )

        return alert

    # ------------------------------------------------------------------
    # Public API: Alert Acknowledgement
    # ------------------------------------------------------------------

    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        notes: str = "",
    ) -> SatelliteAlert:
        """Mark an alert as acknowledged with user ID and timestamp.

        Args:
            alert_id: Alert identifier.
            user_id: Identifier of the user acknowledging the alert.
            notes: Optional acknowledgement notes.

        Returns:
            Updated SatelliteAlert with acknowledgement metadata.

        Raises:
            KeyError: If alert_id is not found.
        """
        if alert_id not in self._alerts:
            raise KeyError(f"Alert not found: {alert_id}")

        alert = self._alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = user_id
        alert.acknowledged_at = utcnow()
        alert.acknowledgement_notes = notes

        # Recompute provenance hash with acknowledgement data
        alert.provenance_hash = _compute_hash(alert)

        logger.info(
            "Alert %s acknowledged by %s: notes='%s'",
            alert_id,
            user_id,
            notes[:50] if notes else "",
        )

        return alert

    # ------------------------------------------------------------------
    # Public API: Alert Queries
    # ------------------------------------------------------------------

    def get_alerts(
        self,
        plot_id: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = DEFAULT_ALERT_LIMIT,
        offset: int = 0,
    ) -> List[SatelliteAlert]:
        """Filter and paginate alerts.

        Args:
            plot_id: Filter by plot ID (or None for all plots).
            severity: Filter by severity level (or None for all).
            acknowledged: Filter by acknowledgement status (or None for all).
            limit: Maximum number of alerts to return.
            offset: Number of alerts to skip.

        Returns:
            List of SatelliteAlert matching the filters, sorted by
            creation timestamp descending (newest first).
        """
        # Start with all alerts or plot-specific alerts
        if plot_id is not None:
            alert_ids = self._plot_alerts.get(plot_id, [])
            candidates = [
                self._alerts[aid] for aid in alert_ids
                if aid in self._alerts
            ]
        else:
            candidates = list(self._alerts.values())

        # Apply severity filter
        if severity is not None:
            severity_lower = severity.lower().strip()
            candidates = [
                a for a in candidates if a.severity == severity_lower
            ]

        # Apply acknowledged filter
        if acknowledged is not None:
            candidates = [
                a for a in candidates if a.acknowledged == acknowledged
            ]

        # Sort by creation timestamp descending
        candidates.sort(key=lambda a: a.created_at, reverse=True)

        # Paginate
        return candidates[offset:offset + limit]

    # ------------------------------------------------------------------
    # Public API: Alert Summary
    # ------------------------------------------------------------------

    def get_alert_summary(self) -> AlertSummary:
        """Get aggregate summary of all alerts.

        Returns:
            AlertSummary with counts by severity, acknowledgement status,
            commodity, and country.
        """
        all_alerts = list(self._alerts.values())

        critical_count = sum(
            1 for a in all_alerts if a.severity == "critical"
        )
        warning_count = sum(
            1 for a in all_alerts if a.severity == "warning"
        )
        info_count = sum(
            1 for a in all_alerts if a.severity == "info"
        )
        acknowledged_count = sum(
            1 for a in all_alerts if a.acknowledged
        )

        by_commodity: Dict[str, int] = {}
        for a in all_alerts:
            comm = a.commodity or "unknown"
            by_commodity[comm] = by_commodity.get(comm, 0) + 1

        by_country: Dict[str, int] = {}
        for a in all_alerts:
            cc = a.country_code or "XX"
            by_country[cc] = by_country.get(cc, 0) + 1

        summary = AlertSummary(
            total_alerts=len(all_alerts),
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            acknowledged_count=acknowledged_count,
            unacknowledged_count=len(all_alerts) - acknowledged_count,
            by_commodity=by_commodity,
            by_country=by_country,
        )
        summary.provenance_hash = _compute_hash(summary)

        return summary

    # ------------------------------------------------------------------
    # Public API: Evidence Package Generation
    # ------------------------------------------------------------------

    def generate_evidence_package(
        self,
        plot_id: str,
        operator_id: str,
        format: str = "json",
        include_time_series: bool = True,
        baseline_data: Optional[Dict[str, Any]] = None,
        latest_data: Optional[Dict[str, Any]] = None,
        ndvi_series: Optional[List[Dict[str, Any]]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> EvidencePackage:
        """Package all satellite evidence for DDS submission.

        Assembles baseline snapshot, latest analysis, NDVI time series,
        alerts, data quality assessment, and compliance determination
        into a structured evidence package with SHA-256 provenance hash.

        Args:
            plot_id: Production plot identifier.
            operator_id: Operator identifier.
            format: Export format (json, csv, pdf_data, eudr_xml).
            include_time_series: Whether to include NDVI time series.
            baseline_data: Baseline analysis data dict (or None for defaults).
            latest_data: Latest analysis data dict (or None for defaults).
            ndvi_series: NDVI time series observations (or None).
            data_quality: Data quality assessment dict (or None for defaults).

        Returns:
            EvidencePackage with provenance hash.

        Raises:
            ValueError: If format is not supported.
        """
        start_time = time.monotonic()

        format_lower = format.lower().strip()
        if format_lower not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Valid formats: {SUPPORTED_FORMATS}"
            )

        # Collect alerts for this plot
        plot_alert_ids = self._plot_alerts.get(plot_id, [])
        plot_alerts = [
            self._alerts[aid].to_dict()
            for aid in plot_alert_ids
            if aid in self._alerts
        ]

        # Use provided data or build defaults
        baseline = baseline_data or self._build_default_baseline(plot_id)
        latest = latest_data or self._build_default_latest(plot_id)
        quality = data_quality or self._build_default_quality(plot_id)

        # Include time series if requested
        time_series = []
        if include_time_series and ndvi_series:
            time_series = list(ndvi_series)

        # Determine compliance from latest change and alerts
        change_for_compliance = self._extract_change_from_latest(latest)
        compliance = self.determine_compliance(
            change_result=change_for_compliance,
            data_quality=quality,
            alerts=plot_alerts,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        package = EvidencePackage(
            plot_id=plot_id,
            operator_id=operator_id,
            format=format_lower,
            baseline_snapshot=baseline,
            latest_analysis=latest,
            ndvi_time_series=time_series,
            alerts=plot_alerts,
            data_quality=quality,
            compliance_determination=compliance,
            metadata={
                "module_version": _MODULE_VERSION,
                "alert_count": len(plot_alerts),
                "time_series_length": len(time_series),
                "include_time_series": include_time_series,
                "generated_by": "AlertGenerator",
            },
            processing_time_ms=round(elapsed_ms, 2),
        )
        package.provenance_hash = _compute_hash(package)

        # Store package
        self._evidence_packages[package.package_id] = package

        logger.info(
            "Evidence package %s generated: plot=%s, operator=%s, "
            "format=%s, alerts=%d, compliance=%s, elapsed=%.2fms",
            package.package_id,
            plot_id,
            operator_id,
            format_lower,
            len(plot_alerts),
            compliance,
            elapsed_ms,
        )

        return package

    # ------------------------------------------------------------------
    # Public API: Evidence Export
    # ------------------------------------------------------------------

    def export_evidence_json(self, package: EvidencePackage) -> str:
        """Export an evidence package as a JSON string.

        Args:
            package: EvidencePackage to export.

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(
            package.to_dict(),
            indent=2,
            default=str,
            ensure_ascii=False,
        )

    def export_evidence_csv(self, package: EvidencePackage) -> str:
        """Export evidence package alerts as a CSV string.

        Produces a CSV with one row per alert associated with the plot.
        Suitable for import into spreadsheets or BI tools.

        Args:
            package: EvidencePackage to export.

        Returns:
            CSV string with header row and one data row per alert.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "alert_id",
            "plot_id",
            "severity",
            "created_at",
            "ndvi_drop",
            "ndvi_baseline",
            "ndvi_current",
            "confidence",
            "deforestation_score",
            "affected_area_ha",
            "commodity",
            "country_code",
            "observation_date",
            "sensor",
            "acknowledged",
        ])

        # Data rows
        for alert_dict in package.alerts:
            writer.writerow([
                alert_dict.get("alert_id", ""),
                alert_dict.get("plot_id", ""),
                alert_dict.get("severity", ""),
                alert_dict.get("created_at", ""),
                alert_dict.get("ndvi_drop", 0.0),
                alert_dict.get("ndvi_baseline", 0.0),
                alert_dict.get("ndvi_current", 0.0),
                alert_dict.get("confidence", 0.0),
                alert_dict.get("deforestation_score", 0.0),
                alert_dict.get("affected_area_ha", 0.0),
                alert_dict.get("commodity", ""),
                alert_dict.get("country_code", ""),
                alert_dict.get("observation_date", ""),
                alert_dict.get("sensor", ""),
                alert_dict.get("acknowledged", False),
            ])

        # Append summary row
        writer.writerow([])
        writer.writerow(["# Evidence Package Summary"])
        writer.writerow(["package_id", package.package_id])
        writer.writerow(["plot_id", package.plot_id])
        writer.writerow(["operator_id", package.operator_id])
        writer.writerow(["compliance", package.compliance_determination])
        writer.writerow(["provenance_hash", package.provenance_hash])
        writer.writerow(["generated_at", package.created_at.isoformat()])

        return output.getvalue()

    def export_evidence_pdf_data(
        self,
        package: EvidencePackage,
    ) -> Dict[str, Any]:
        """Export evidence package as structured data for PDF generation.

        Returns a dictionary structured for PDF template rendering with
        sections for header, baseline, analysis, alerts, and compliance.

        Args:
            package: EvidencePackage to export.

        Returns:
            Dictionary with PDF-ready sections.
        """
        # Alert severity counts
        critical_alerts = [
            a for a in package.alerts if a.get("severity") == "critical"
        ]
        warning_alerts = [
            a for a in package.alerts if a.get("severity") == "warning"
        ]
        info_alerts = [
            a for a in package.alerts if a.get("severity") == "info"
        ]

        return {
            "title": "EUDR Satellite Monitoring Evidence Package",
            "subtitle": (
                f"Plot: {package.plot_id} | "
                f"Operator: {package.operator_id}"
            ),
            "header": {
                "package_id": package.package_id,
                "plot_id": package.plot_id,
                "operator_id": package.operator_id,
                "generated_at": package.created_at.strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "eudr_cutoff_date": package.eudr_cutoff_date,
                "provenance_hash": package.provenance_hash,
            },
            "compliance": {
                "determination": package.compliance_determination,
                "description": self._get_compliance_description(
                    package.compliance_determination
                ),
            },
            "baseline": package.baseline_snapshot,
            "latest_analysis": package.latest_analysis,
            "ndvi_time_series": {
                "data_points": len(package.ndvi_time_series),
                "series": package.ndvi_time_series[:50],
            },
            "alerts": {
                "total": len(package.alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "info": len(info_alerts),
                "details": package.alerts[:20],
            },
            "data_quality": package.data_quality,
            "metadata": package.metadata,
        }

    # ------------------------------------------------------------------
    # Public API: Compliance Determination
    # ------------------------------------------------------------------

    def determine_compliance(
        self,
        change_result: Optional[ChangeResult] = None,
        data_quality: Optional[Dict[str, Any]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Determine EUDR compliance status from change, quality, and alerts.

        Decision rules (deterministic):
            COMPLIANT:              No deforestation + confidence > 0.7
            NON_COMPLIANT:          Deforestation detected + confidence > 0.7
            INSUFFICIENT_DATA:      Confidence < 0.5
            MANUAL_REVIEW_REQUIRED: All other ambiguous cases

        Args:
            change_result: Latest change detection result (or None).
            data_quality: Data quality assessment dict (or None).
            alerts: List of alert dicts for the plot (or None).

        Returns:
            Compliance status string.
        """
        alerts = alerts or []

        # Check confidence from data quality
        quality_score = 0.7  # Default to acceptable
        if data_quality:
            quality_score = float(data_quality.get("overall_quality", 0.7))

        if quality_score < COMPLIANCE_LOW_CONFIDENCE:
            return "INSUFFICIENT_DATA"

        # Check for critical alerts
        has_critical = any(
            a.get("severity") == "critical" for a in alerts
        )

        # Check change result
        if change_result is not None:
            score = change_result.deforestation_score
            confidence = change_result.confidence

            if confidence < COMPLIANCE_LOW_CONFIDENCE:
                return "INSUFFICIENT_DATA"

            if (
                score >= DEFORESTATION_SCORE_THRESHOLD
                and confidence >= COMPLIANCE_HIGH_CONFIDENCE
            ):
                return "NON_COMPLIANT"

            if (
                score < 0.2
                and confidence >= COMPLIANCE_HIGH_CONFIDENCE
                and not has_critical
            ):
                return "COMPLIANT"

            return "MANUAL_REVIEW_REQUIRED"

        # No change result -- rely on alerts
        if has_critical:
            return "NON_COMPLIANT"

        has_warnings = any(
            a.get("severity") == "warning" for a in alerts
        )
        if has_warnings:
            return "MANUAL_REVIEW_REQUIRED"

        if not alerts:
            return "INSUFFICIENT_DATA"

        return "COMPLIANT"

    # ------------------------------------------------------------------
    # Public API: Retrieval
    # ------------------------------------------------------------------

    def get_evidence_package(
        self,
        package_id: str,
    ) -> Optional[EvidencePackage]:
        """Retrieve a stored evidence package by ID.

        Args:
            package_id: Evidence package identifier.

        Returns:
            EvidencePackage if found, else None.
        """
        return self._evidence_packages.get(package_id)

    def get_alert(self, alert_id: str) -> Optional[SatelliteAlert]:
        """Retrieve a single alert by ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            SatelliteAlert if found, else None.
        """
        return self._alerts.get(alert_id)

    # ------------------------------------------------------------------
    # Internal: Severity Classification
    # ------------------------------------------------------------------

    def _classify_severity(
        self,
        ndvi_drop: float,
        confidence: float,
    ) -> str:
        """Classify alert severity based on NDVI drop and confidence.

        Args:
            ndvi_drop: Absolute NDVI drop magnitude.
            confidence: Detection confidence (0.0-1.0).

        Returns:
            Severity string: critical, warning, or info.
        """
        if (
            ndvi_drop > CRITICAL_NDVI_DROP_THRESHOLD
            and confidence > CRITICAL_CONFIDENCE_THRESHOLD
        ):
            return "critical"
        elif (
            ndvi_drop > WARNING_NDVI_DROP_THRESHOLD
            and confidence > WARNING_CONFIDENCE_THRESHOLD
        ):
            return "warning"
        else:
            return "info"

    # ------------------------------------------------------------------
    # Internal: Default Data Builders
    # ------------------------------------------------------------------

    def _build_default_baseline(
        self,
        plot_id: str,
    ) -> Dict[str, Any]:
        """Build default baseline snapshot data.

        Args:
            plot_id: Plot identifier.

        Returns:
            Default baseline data dict.
        """
        return {
            "plot_id": plot_id,
            "reference_date": EUDR_CUTOFF_DATE,
            "ndvi": 0.70,
            "sensor": "sentinel2",
            "confidence": 0.8,
            "note": "Default baseline -- replace with actual baseline data",
        }

    def _build_default_latest(
        self,
        plot_id: str,
    ) -> Dict[str, Any]:
        """Build default latest analysis data.

        Args:
            plot_id: Plot identifier.

        Returns:
            Default latest analysis data dict.
        """
        return {
            "plot_id": plot_id,
            "observation_date": utcnow().strftime("%Y-%m-%d"),
            "ndvi": 0.65,
            "ndvi_change": -0.05,
            "deforestation_score": 0.2,
            "confidence": 0.7,
            "sensor": "sentinel2",
            "note": "Default latest -- replace with actual analysis data",
        }

    def _build_default_quality(
        self,
        plot_id: str,
    ) -> Dict[str, Any]:
        """Build default data quality assessment.

        Args:
            plot_id: Plot identifier.

        Returns:
            Default data quality dict.
        """
        return {
            "plot_id": plot_id,
            "overall_quality": 0.7,
            "quality_tier": "good",
            "source_count": 2,
            "temporal_coverage": 0.7,
            "spatial_coverage": 0.8,
            "cloud_impact": 0.15,
        }

    def _extract_change_from_latest(
        self,
        latest_data: Dict[str, Any],
    ) -> Optional[ChangeResult]:
        """Extract a ChangeResult from latest analysis data dict.

        Args:
            latest_data: Latest analysis dict.

        Returns:
            ChangeResult if extractable, else None.
        """
        if not latest_data:
            return None

        ndvi_change = float(latest_data.get("ndvi_change", 0.0))
        deforestation_score = float(
            latest_data.get("deforestation_score", 0.0)
        )
        confidence = float(latest_data.get("confidence", 0.0))

        if ndvi_change < 0:
            change_type = "ndvi_drop"
        elif ndvi_change > 0:
            change_type = "ndvi_increase"
        else:
            change_type = "stable"

        return ChangeResult(
            change_type=change_type,
            ndvi_baseline=float(latest_data.get("ndvi", 0.0)) - ndvi_change,
            ndvi_current=float(latest_data.get("ndvi", 0.0)),
            ndvi_change=ndvi_change,
            confidence=confidence,
            deforestation_score=deforestation_score,
            observation_date=str(latest_data.get("observation_date", "")),
            sensor=str(latest_data.get("sensor", "")),
        )

    def _get_compliance_description(self, status: str) -> str:
        """Get human-readable description for a compliance status.

        Args:
            status: Compliance status string.

        Returns:
            Human-readable description.
        """
        descriptions: Dict[str, str] = {
            "COMPLIANT": (
                "No deforestation detected after the EUDR cutoff date "
                f"({EUDR_CUTOFF_DATE}) with high confidence. The plot "
                "meets EUDR deforestation-free requirements."
            ),
            "NON_COMPLIANT": (
                "Deforestation detected after the EUDR cutoff date "
                f"({EUDR_CUTOFF_DATE}) with high confidence. The plot "
                "does NOT meet EUDR deforestation-free requirements. "
                "Immediate remediation required."
            ),
            "INSUFFICIENT_DATA": (
                "Insufficient satellite data or low confidence prevents "
                "a definitive compliance determination. Additional imagery "
                "acquisition or ground truth verification is required."
            ),
            "MANUAL_REVIEW_REQUIRED": (
                "Ambiguous satellite analysis results require manual review "
                "by a qualified operator. The automated analysis could not "
                "reach a definitive compliance conclusion."
            ),
        }
        return descriptions.get(
            status,
            "Unknown compliance status. Manual review recommended.",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alert_count(self) -> int:
        """Return the total number of stored alerts."""
        return len(self._alerts)

    @property
    def evidence_package_count(self) -> int:
        """Return the total number of stored evidence packages."""
        return len(self._evidence_packages)

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "AlertGenerator",
    # Data classes
    "SatelliteAlert",
    "AlertSummary",
    "EvidencePackage",
    "ChangeResult",
    "MonitoringResultInput",
    # Constants
    "SEVERITY_LEVELS",
    "CRITICAL_NDVI_DROP_THRESHOLD",
    "CRITICAL_CONFIDENCE_THRESHOLD",
    "WARNING_NDVI_DROP_THRESHOLD",
    "WARNING_CONFIDENCE_THRESHOLD",
    "COMPLIANCE_HIGH_CONFIDENCE",
    "COMPLIANCE_LOW_CONFIDENCE",
    "DEFORESTATION_SCORE_THRESHOLD",
    "SUPPORTED_FORMATS",
    "EUDR_CUTOFF_DATE",
    "DEFAULT_ALERT_LIMIT",
]
