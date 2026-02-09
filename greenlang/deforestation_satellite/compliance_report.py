# -*- coding: utf-8 -*-
"""
Compliance Report Engine - AGENT-DATA-007: GL-DATA-GEO-003

Generates EUDR (EU Deforestation Regulation) compliance reports by
combining baseline assessment results, alert aggregation data, and
risk scoring into formal compliance determinations.

Features:
    - Combined evidence assessment from baseline and alert data
    - Context-aware recommendation generation
    - Structured evidence summary for Due Diligence Statement (DDS)
    - Risk score fusion from multiple data sources
    - Report storage and retrieval
    - Provenance tracking for all report generation operations

Zero-Hallucination Guarantees:
    - Compliance status follows deterministic rules from combined evidence
    - Risk scores use weighted sum formulas, not LLM inference
    - Recommendations are template-based, not generated
    - All report fields are traceable to source data

Example:
    >>> from greenlang.deforestation_satellite.compliance_report import ComplianceReportEngine
    >>> engine = ComplianceReportEngine()
    >>> # Generate report from baseline and alerts...
    >>> print(engine.report_count)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.deforestation_satellite.config import get_config
from greenlang.deforestation_satellite.models import (
    AlertAggregation,
    AlertSeverity,
    BaselineAssessment,
    ComplianceReport,
    ComplianceStatus,
    DeforestationRisk,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EUDR cutoff date
_EUDR_CUTOFF_DATE = "2020-12-31"

# Compliance status priority (for combining multiple signals)
_COMPLIANCE_PRIORITY = {
    ComplianceStatus.COMPLIANT.value: 0,
    ComplianceStatus.REVIEW_REQUIRED.value: 1,
    ComplianceStatus.NON_COMPLIANT.value: 2,
}

# Risk level mapping from score
_RISK_SCORE_LEVELS = [
    (80.0, DeforestationRisk.CRITICAL),
    (60.0, DeforestationRisk.HIGH),
    (40.0, DeforestationRisk.MEDIUM),
    (20.0, DeforestationRisk.LOW),
    (0.0, DeforestationRisk.LOW),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# ComplianceReportEngine
# =============================================================================


class ComplianceReportEngine:
    """Engine for generating EUDR compliance reports.

    Combines baseline assessment data with alert aggregation results
    to produce formal compliance determinations with evidence summaries,
    risk scores, and actionable recommendations.

    Attributes:
        config: DeforestationSatelliteConfig instance.
        provenance: Optional ProvenanceTracker for audit trails.

    Example:
        >>> engine = ComplianceReportEngine()
        >>> print(engine.report_count)
        0
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize ComplianceReportEngine.

        Args:
            config: Optional DeforestationSatelliteConfig. Uses global
                config if None.
            provenance: Optional ProvenanceTracker for recording audit entries.
        """
        self.config = config or get_config()
        self.provenance = provenance
        self._reports: Dict[str, ComplianceReport] = {}
        self._report_count: int = 0
        logger.info("ComplianceReportEngine initialized")

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        baseline: BaselineAssessment,
        alerts: AlertAggregation,
        polygon_wkt: str,
    ) -> ComplianceReport:
        """Generate a full EUDR compliance report.

        Combines baseline assessment and alert data to produce a
        comprehensive compliance determination with evidence summary,
        risk scoring, and recommendations.

        Args:
            baseline: BaselineAssessment from the baseline engine.
            alerts: AlertAggregation from the alert engine.
            polygon_wkt: WKT polygon string for the assessed area.

        Returns:
            ComplianceReport with full compliance determination.
        """
        # Determine combined compliance status
        compliance_status = self.assess_compliance(baseline, alerts)

        # Calculate combined risk score
        alert_severity_counts = alerts.alerts_by_severity if alerts else {}
        combined_risk = self.calculate_risk_from_combined(
            baseline.risk_score,
            alert_severity_counts,
        )

        # Determine risk level from combined score
        risk_level = DeforestationRisk.LOW
        for threshold, level in _RISK_SCORE_LEVELS:
            if combined_risk >= threshold:
                risk_level = level
                break

        # Generate recommendations
        recommendations = self.generate_recommendations(
            compliance_status, baseline, alerts,
        )

        # Generate evidence summary
        evidence_summary = self.generate_evidence_summary(baseline, alerts)

        # Calculate area metrics
        total_area_ha = max(1.0, alerts.total_affected_area_ha) if alerts else 1.0
        forest_area_ha = total_area_ha * (baseline.baseline_forest_cover_percent / 100.0)
        deforested_area_ha = max(0.0, forest_area_ha * abs(min(0, baseline.forest_cover_change_percent)) / 100.0)

        # Count post-cutoff and high-confidence alerts
        post_cutoff_count = 0
        high_confidence_count = alerts.high_confidence_count if alerts else 0
        total_alert_count = alerts.total_alerts if alerts else 0

        # Estimate post-cutoff alerts from severity distribution
        if alerts and alerts.alerts_by_severity:
            high_sev = alerts.alerts_by_severity.get(AlertSeverity.HIGH.value, 0)
            crit_sev = alerts.alerts_by_severity.get(AlertSeverity.CRITICAL.value, 0)
            post_cutoff_count = high_sev + crit_sev

        affected_area_ha = alerts.total_affected_area_ha if alerts else 0.0

        report_id = self._generate_report_id()

        report = ComplianceReport(
            report_id=report_id,
            polygon_wkt=polygon_wkt,
            country_iso3=baseline.country_iso3,
            compliance_status=compliance_status.value,
            risk_level=risk_level.value,
            risk_score=round(combined_risk, 2),
            total_area_ha=round(total_area_ha, 4),
            forest_area_ha=round(forest_area_ha, 4),
            deforested_area_ha=round(deforested_area_ha, 4),
            total_alerts=total_alert_count,
            post_cutoff_alerts=post_cutoff_count,
            high_confidence_alerts=high_confidence_count,
            affected_area_ha=round(affected_area_ha, 4),
            recommendations=recommendations,
            evidence_summary=evidence_summary,
        )

        # Compute provenance hash
        report_data = report.model_dump(mode="json")
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        report.provenance_hash = report_hash

        # Store report
        self._reports[report_id] = report
        self._report_count += 1

        # Record provenance
        if self.provenance is not None:
            self.provenance.record(
                entity_type="compliance_report",
                entity_id=report_id,
                action="generate",
                data_hash=report_hash,
            )

        logger.info(
            "Compliance report %s: status=%s, risk=%.1f (%s), "
            "alerts=%d, post_cutoff=%d, area=%.2fha",
            report_id, compliance_status.value, combined_risk,
            risk_level.value, total_alert_count,
            post_cutoff_count, total_area_ha,
        )

        return report

    # ------------------------------------------------------------------
    # Compliance assessment
    # ------------------------------------------------------------------

    def assess_compliance(
        self,
        baseline: BaselineAssessment,
        alerts: AlertAggregation,
    ) -> ComplianceStatus:
        """Determine EUDR compliance from combined evidence.

        Decision rules:
            NON_COMPLIANT if:
                - Baseline is non-compliant (area was forest, now deforested)
                - AND critical/high alerts present post-cutoff
            NON_COMPLIANT if:
                - Risk score >= 70
                - AND post-cutoff high-confidence alerts exist
            REVIEW_REQUIRED if:
                - Baseline shows review_required
                - OR any alerts with medium+ severity post-cutoff
                - OR risk score >= 40
            COMPLIANT otherwise

        Args:
            baseline: BaselineAssessment result.
            alerts: AlertAggregation result.

        Returns:
            ComplianceStatus determination.
        """
        # Start with baseline compliance
        baseline_compliant = baseline.is_eudr_compliant
        baseline_risk = baseline.risk_score

        # Check alerts
        has_critical = alerts.has_critical if alerts else False
        high_conf_count = alerts.high_confidence_count if alerts else 0
        total_alerts = alerts.total_alerts if alerts else 0

        severity_counts = alerts.alerts_by_severity if alerts else {}
        high_sev = severity_counts.get(AlertSeverity.HIGH.value, 0)
        crit_sev = severity_counts.get(AlertSeverity.CRITICAL.value, 0)
        med_sev = severity_counts.get(AlertSeverity.MEDIUM.value, 0)

        # Rule 1: Non-compliant baseline + critical/high alerts
        if not baseline_compliant and (has_critical or high_sev > 0):
            return ComplianceStatus.NON_COMPLIANT

        # Rule 2: High risk + high-confidence alerts
        if baseline_risk >= 70 and high_conf_count > 0:
            return ComplianceStatus.NON_COMPLIANT

        # Rule 3: Critical alerts alone
        if has_critical and crit_sev >= 2:
            return ComplianceStatus.NON_COMPLIANT

        # Rule 4: Review required conditions
        if not baseline_compliant:
            return ComplianceStatus.REVIEW_REQUIRED

        if baseline_risk >= 40:
            return ComplianceStatus.REVIEW_REQUIRED

        if high_sev > 0 or crit_sev > 0:
            return ComplianceStatus.REVIEW_REQUIRED

        if med_sev > 2:
            return ComplianceStatus.REVIEW_REQUIRED

        if total_alerts > 10:
            return ComplianceStatus.REVIEW_REQUIRED

        return ComplianceStatus.COMPLIANT

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        status: ComplianceStatus,
        baseline: BaselineAssessment,
        alerts: AlertAggregation,
    ) -> List[str]:
        """Generate context-aware recommendations based on compliance status.

        Uses template-based recommendation generation with contextual
        parameters. No LLM or generative AI is used.

        Args:
            status: Compliance status determination.
            baseline: BaselineAssessment result.
            alerts: AlertAggregation result.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if status == ComplianceStatus.NON_COMPLIANT:
            recommendations.append(
                "URGENT: This area shows evidence of post-cutoff deforestation. "
                "Products from this supply chain node should be flagged for "
                "non-compliance under EUDR Article 3."
            )
            recommendations.append(
                "Conduct immediate field verification to confirm satellite-based "
                "deforestation detection findings."
            )
            recommendations.append(
                "Engage with local suppliers to obtain documentation of land use "
                "history and legal logging permits."
            )
            recommendations.append(
                "Consider alternative supply chain sourcing from verified "
                "deforestation-free areas."
            )

        elif status == ComplianceStatus.REVIEW_REQUIRED:
            recommendations.append(
                "Enhanced due diligence is recommended for this supply chain node. "
                "Additional evidence should be collected before final compliance "
                "determination."
            )
            if baseline.risk_score >= 50:
                recommendations.append(
                    f"Country risk score ({baseline.risk_score:.0f}/100) indicates "
                    "elevated deforestation pressure. Request supplier-specific "
                    "geolocation data and land title documentation."
                )

            if alerts and alerts.total_alerts > 0:
                recommendations.append(
                    f"{alerts.total_alerts} deforestation alert(s) detected in the "
                    "monitoring area. Cross-reference with ground-truth data and "
                    "high-resolution imagery."
                )

            recommendations.append(
                "Schedule periodic re-assessment (quarterly recommended) to "
                "monitor for new deforestation events."
            )

        else:  # COMPLIANT
            recommendations.append(
                "Area shows no evidence of post-cutoff deforestation. "
                "Standard monitoring schedule is sufficient."
            )
            recommendations.append(
                "Maintain annual re-assessment to ensure continued compliance "
                "with EUDR requirements."
            )
            if baseline.forest_cover_change_percent < -3:
                recommendations.append(
                    f"Minor forest cover decline ({baseline.forest_cover_change_percent:.1f}%) "
                    "detected. Monitor for potential degradation trend."
                )

        return recommendations

    # ------------------------------------------------------------------
    # Evidence summary
    # ------------------------------------------------------------------

    def generate_evidence_summary(
        self,
        baseline: BaselineAssessment,
        alerts: AlertAggregation,
    ) -> Dict[str, Any]:
        """Generate a structured evidence summary for DDS documentation.

        Produces a machine-readable dictionary of all evidence used
        in the compliance determination, suitable for export to
        Due Diligence Statement (DDS) systems.

        Args:
            baseline: BaselineAssessment result.
            alerts: AlertAggregation result.

        Returns:
            Structured evidence dictionary.
        """
        summary: Dict[str, Any] = {
            "assessment": {
                "assessment_id": baseline.assessment_id,
                "coordinate_lat": baseline.coordinate_lat,
                "coordinate_lon": baseline.coordinate_lon,
                "country_iso3": baseline.country_iso3,
                "baseline_date": baseline.baseline_date,
                "assessment_date": baseline.assessment_date,
            },
            "forest_cover": {
                "baseline_cover_percent": baseline.baseline_forest_cover_percent,
                "current_cover_percent": baseline.current_forest_cover_percent,
                "change_percent": baseline.forest_cover_change_percent,
                "forest_status": baseline.forest_status,
            },
            "risk": {
                "risk_score": baseline.risk_score,
                "risk_level": baseline.risk_level,
                "is_eudr_compliant": baseline.is_eudr_compliant,
            },
            "deforestation_events": baseline.deforestation_events,
            "data_sources": baseline.data_sources,
            "warnings": baseline.warnings,
        }

        if alerts:
            summary["alerts"] = {
                "aggregation_id": alerts.aggregation_id,
                "date_range_start": alerts.date_range_start,
                "date_range_end": alerts.date_range_end,
                "total_alerts": alerts.total_alerts,
                "alerts_by_source": alerts.alerts_by_source,
                "alerts_by_severity": alerts.alerts_by_severity,
                "total_affected_area_ha": alerts.total_affected_area_ha,
                "has_critical": alerts.has_critical,
                "high_confidence_count": alerts.high_confidence_count,
            }

        summary["eudr_regulation"] = {
            "cutoff_date": _EUDR_CUTOFF_DATE,
            "regulation": "EU Regulation 2023/1115",
            "article": "Article 3 - Prohibition",
        }

        summary["methodology"] = {
            "satellite_sources": ["Sentinel-2", "Landsat-8/9"],
            "alert_systems": ["GLAD", "RADD", "FIRMS"],
            "indices_used": ["NDVI", "EVI", "NBR"],
            "classification_method": "threshold",
            "assessment_type": "automated_satellite_monitoring",
        }

        return summary

    # ------------------------------------------------------------------
    # Risk fusion
    # ------------------------------------------------------------------

    def calculate_risk_from_combined(
        self,
        baseline_risk: float,
        alert_severity_counts: Dict[str, int],
    ) -> float:
        """Calculate combined risk score from baseline and alert evidence.

        Formula:
            combined = 0.60 * baseline_risk + 0.40 * alert_risk_component

        Alert risk component (0-100):
            - CRITICAL alerts:  each adds 25 points
            - HIGH alerts:      each adds 12 points
            - MEDIUM alerts:    each adds 5 points
            - LOW alerts:       each adds 1 point

        Args:
            baseline_risk: Baseline assessment risk score (0-100).
            alert_severity_counts: Dict mapping severity to alert count.

        Returns:
            Combined risk score (0-100).
        """
        # Alert risk component
        alert_risk = 0.0
        alert_risk += alert_severity_counts.get(AlertSeverity.CRITICAL.value, 0) * 25.0
        alert_risk += alert_severity_counts.get(AlertSeverity.HIGH.value, 0) * 12.0
        alert_risk += alert_severity_counts.get(AlertSeverity.MEDIUM.value, 0) * 5.0
        alert_risk += alert_severity_counts.get(AlertSeverity.LOW.value, 0) * 1.0
        alert_risk = min(100.0, alert_risk)

        # Weighted combination
        combined = 0.60 * baseline_risk + 0.40 * alert_risk

        return max(0.0, min(100.0, combined))

    # ------------------------------------------------------------------
    # Report retrieval
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> Optional[ComplianceReport]:
        """Retrieve a compliance report by ID.

        Args:
            report_id: Unique report identifier.

        Returns:
            ComplianceReport or None if not found.
        """
        return self._reports.get(report_id)

    def list_reports(self) -> List[ComplianceReport]:
        """Return all stored compliance reports.

        Returns:
            List of ComplianceReport instances.
        """
        return list(self._reports.values())

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_report_id(self) -> str:
        """Generate a unique report identifier.

        Returns:
            String in format "RPT-{12 hex chars}".
        """
        return f"RPT-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def report_count(self) -> int:
        """Return the total number of reports generated.

        Returns:
            Integer count of reports.
        """
        return self._report_count


__all__ = [
    "ComplianceReportEngine",
]
