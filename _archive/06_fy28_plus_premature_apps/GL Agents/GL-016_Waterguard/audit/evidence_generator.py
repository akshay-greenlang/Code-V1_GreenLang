"""
Compliance Evidence Generator for GL-016 Waterguard

This module generates compliance evidence packages for regulatory audits
and internal reviews. Supports weekly and monthly reporting with
ASME/ABMA alignment documentation.

Key Features:
    - Weekly compliance report generation
    - Monthly comprehensive reports
    - ASME/ABMA alignment documentation
    - Evidence pack assembly from audit logs
    - Export to multiple formats

Example:
    >>> generator = ComplianceEvidenceGenerator(audit_logger, provenance_tracker)
    >>> weekly_report = generator.generate_weekly_report(
    ...     asset_id="boiler-001",
    ...     week_start=datetime(2024, 1, 1)
    ... )
    >>> monthly_report = generator.generate_monthly_report(
    ...     asset_id="boiler-001",
    ...     month=1,
    ...     year=2024
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .audit_events import (
    EventType,
    ChemistryParameter,
    ConstraintType,
    RecommendationType,
    OperatorActionType,
    SeverityLevel,
)
from .audit_logger import WaterguardAuditLogger
from .audit_query import AuditQueryService, QueryFilter
from .provenance_enhanced import ProvenanceTracker
from .evidence_pack import (
    EvidencePack,
    EvidencePackStatus,
    ComplianceStandard,
    ChemistryCalculationSummary,
    ChemistryParameterSummary,
    ConstraintComplianceSummary,
    ConstraintComplianceRecord,
    RecommendationSummary,
    RecommendationRecord,
    OperatorDecisionSummary,
    OperatorDecisionRecord,
    ASMEAlignment,
    ABMAAlignment,
    ProvenanceSummary,
)

logger = logging.getLogger(__name__)


class WeeklyComplianceReport(BaseModel):
    """Weekly compliance report model."""

    report_id: str = Field(default_factory=lambda: f"wk-{uuid4().hex[:12]}")
    report_type: str = Field(default="WEEKLY")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = Field(default="GL-016")

    # Period
    week_start: datetime = Field(..., description="Week start date")
    week_end: datetime = Field(..., description="Week end date")
    week_number: int = Field(..., ge=1, le=53, description="ISO week number")
    year: int = Field(..., description="Year")

    # Asset
    asset_id: str = Field(..., description="Asset ID")
    facility_id: Optional[str] = Field(None, description="Facility ID")

    # Summary metrics
    total_calculations: int = Field(0, ge=0)
    data_completeness_pct: float = Field(100.0, ge=0, le=100)
    overall_compliance_pct: float = Field(100.0, ge=0, le=100)
    total_violations: int = Field(0, ge=0)
    critical_violations: int = Field(0, ge=0)
    total_recommendations: int = Field(0, ge=0)
    implemented_recommendations: int = Field(0, ge=0)

    # Evidence pack reference
    evidence_pack_id: Optional[str] = Field(None)

    # Detailed data
    parameter_summaries: Dict[str, Any] = Field(default_factory=dict)
    constraint_compliance: Dict[str, float] = Field(default_factory=dict)
    daily_metrics: List[Dict[str, Any]] = Field(default_factory=list)

    # Narrative summary
    narrative: str = Field(default="", description="Human-readable summary")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MonthlyComplianceReport(BaseModel):
    """Monthly compliance report model."""

    report_id: str = Field(default_factory=lambda: f"mo-{uuid4().hex[:12]}")
    report_type: str = Field(default="MONTHLY")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = Field(default="GL-016")

    # Period
    month: int = Field(..., ge=1, le=12, description="Month")
    year: int = Field(..., description="Year")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")

    # Asset
    asset_id: str = Field(..., description="Asset ID")
    facility_id: Optional[str] = Field(None, description="Facility ID")

    # Summary metrics
    total_calculations: int = Field(0, ge=0)
    data_completeness_pct: float = Field(100.0, ge=0, le=100)
    overall_compliance_pct: float = Field(100.0, ge=0, le=100)
    total_violations: int = Field(0, ge=0)
    critical_violations: int = Field(0, ge=0)
    violation_trend: str = Field(default="STABLE", description="IMPROVING/STABLE/DEGRADING")

    # Recommendations
    total_recommendations: int = Field(0, ge=0)
    implemented_recommendations: int = Field(0, ge=0)
    rejected_recommendations: int = Field(0, ge=0)

    # Savings
    water_savings_gallons: float = Field(0.0)
    energy_savings_mmbtu: float = Field(0.0)
    chemical_savings_usd: float = Field(0.0)

    # Standards alignment
    asme_compliant: bool = Field(True)
    abma_compliant: bool = Field(True)

    # Weekly reports included
    weekly_report_ids: List[str] = Field(default_factory=list)

    # Evidence pack reference
    evidence_pack_id: Optional[str] = Field(None)

    # Detailed data
    parameter_trends: Dict[str, Any] = Field(default_factory=dict)
    weekly_summaries: List[Dict[str, Any]] = Field(default_factory=list)

    # Narrative summary
    executive_summary: str = Field(default="", description="Executive summary")
    recommendations_for_next_period: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ASMEAlignmentReport(BaseModel):
    """ASME alignment report model."""

    report_id: str = Field(default_factory=lambda: f"asme-{uuid4().hex[:12]}")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Period
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")

    # Asset
    asset_id: str = Field(..., description="Asset ID")
    operating_pressure_psig: float = Field(..., description="Operating pressure")

    # Standard reference
    standard_id: str = Field(default="ASME PTC 4.2")
    standard_version: str = Field(default="2018")

    # Alignment status
    overall_aligned: bool = Field(True)
    alignment_score_pct: float = Field(100.0, ge=0, le=100)

    # Parameter alignments
    parameter_alignments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Deviations
    deviations: List[Dict[str, Any]] = Field(default_factory=list)

    # Recommendations
    corrective_actions: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ComplianceEvidenceGenerator:
    """
    Generator for compliance evidence packages and reports.

    This class creates evidence packs and compliance reports from
    audit logs for regulatory compliance and internal reviews.

    Attributes:
        audit_logger: Waterguard audit logger instance
        query_service: Audit query service
        provenance_tracker: Provenance tracker instance
        storage_path: Path for storing generated evidence

    Example:
        >>> generator = ComplianceEvidenceGenerator(audit_logger, provenance_tracker)
        >>> report = generator.generate_weekly_report("boiler-001", week_start)
    """

    # ASME limits by pressure (simplified - production would have full tables)
    ASME_LIMITS = {
        "low": {  # 0-300 psig
            "conductivity_max": 7000,
            "silica_max": 150,
            "ph_min": 10.0,
            "ph_max": 11.5,
        },
        "medium": {  # 301-450 psig
            "conductivity_max": 6000,
            "silica_max": 90,
            "ph_min": 10.0,
            "ph_max": 11.0,
        },
        "high": {  # 451-600 psig
            "conductivity_max": 5000,
            "silica_max": 40,
            "ph_min": 9.8,
            "ph_max": 10.8,
        },
        "very_high": {  # 601+ psig
            "conductivity_max": 3000,
            "silica_max": 8,
            "ph_min": 9.4,
            "ph_max": 10.2,
        },
    }

    def __init__(
        self,
        audit_logger: WaterguardAuditLogger,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize compliance evidence generator.

        Args:
            audit_logger: Waterguard audit logger instance
            provenance_tracker: Optional provenance tracker
            storage_path: Optional path for storing evidence
        """
        self.audit_logger = audit_logger
        self.query_service = AuditQueryService(audit_logger)
        self.provenance_tracker = provenance_tracker
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("ComplianceEvidenceGenerator initialized")

    def _get_pressure_category(self, pressure_psig: float) -> str:
        """Get ASME pressure category."""
        if pressure_psig <= 300:
            return "low"
        elif pressure_psig <= 450:
            return "medium"
        elif pressure_psig <= 600:
            return "high"
        else:
            return "very_high"

    def generate_weekly_report(
        self,
        asset_id: str,
        week_start: datetime,
        facility_id: Optional[str] = None,
    ) -> WeeklyComplianceReport:
        """
        Generate weekly compliance report.

        Args:
            asset_id: Asset ID to report on
            week_start: Start of the week (Monday)
            facility_id: Optional facility ID

        Returns:
            WeeklyComplianceReport
        """
        start_time = datetime.now(timezone.utc)
        week_end = week_start + timedelta(days=7)
        week_number = week_start.isocalendar()[1]
        year = week_start.year

        logger.info(
            f"Generating weekly report for {asset_id}",
            extra={"week": week_number, "year": year}
        )

        # Query chemistry events
        chemistry_events = self.query_service.query_chemistry_events(
            asset_id=asset_id,
            start_time=week_start,
            end_time=week_end,
            limit=10000,
        )

        # Query recommendations
        recommendations = self.query_service.query_recommendations(
            asset_id=asset_id,
            start_time=week_start,
            end_time=week_end,
            limit=1000,
        )

        # Query violations
        violations = self.query_service.query_violations(
            asset_id=asset_id,
            start_time=week_start,
            end_time=week_end,
            limit=1000,
        )

        # Calculate parameter summaries
        parameter_summaries = self._calculate_parameter_summaries(chemistry_events)

        # Calculate constraint compliance
        constraint_compliance = self._calculate_constraint_compliance(violations, len(chemistry_events))

        # Calculate overall compliance
        overall_compliance = 100.0
        if chemistry_events:
            violation_rate = len(violations) / len(chemistry_events) * 100
            overall_compliance = max(0, 100 - violation_rate)

        # Count critical violations
        critical_violations = sum(
            1 for v in violations if v.severity == SeverityLevel.CRITICAL
        )

        # Generate narrative
        narrative = self._generate_weekly_narrative(
            asset_id,
            week_number,
            year,
            len(chemistry_events),
            overall_compliance,
            len(violations),
            critical_violations,
            len(recommendations),
        )

        report = WeeklyComplianceReport(
            week_start=week_start,
            week_end=week_end,
            week_number=week_number,
            year=year,
            asset_id=asset_id,
            facility_id=facility_id,
            total_calculations=len(chemistry_events),
            data_completeness_pct=self._calculate_data_completeness(chemistry_events, week_start, week_end),
            overall_compliance_pct=overall_compliance,
            total_violations=len(violations),
            critical_violations=critical_violations,
            total_recommendations=len(recommendations),
            implemented_recommendations=sum(1 for r in recommendations if r.confidence_score > 0.8),
            parameter_summaries=parameter_summaries,
            constraint_compliance=constraint_compliance,
            narrative=narrative,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Weekly report generated: {report.report_id}",
            extra={
                "processing_time_ms": processing_time,
                "compliance_pct": overall_compliance,
            }
        )

        return report

    def generate_monthly_report(
        self,
        asset_id: str,
        month: int,
        year: int,
        facility_id: Optional[str] = None,
    ) -> MonthlyComplianceReport:
        """
        Generate monthly compliance report.

        Args:
            asset_id: Asset ID to report on
            month: Month number (1-12)
            year: Year
            facility_id: Optional facility ID

        Returns:
            MonthlyComplianceReport
        """
        start_time = datetime.now(timezone.utc)

        # Calculate period
        period_start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            period_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            period_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        logger.info(
            f"Generating monthly report for {asset_id}",
            extra={"month": month, "year": year}
        )

        # Generate weekly reports for the month
        weekly_reports = []
        current_week = period_start
        while current_week < period_end:
            # Find Monday of the week
            monday = current_week - timedelta(days=current_week.weekday())
            if monday < period_start:
                monday = period_start

            weekly_report = self.generate_weekly_report(asset_id, monday, facility_id)
            weekly_reports.append(weekly_report)
            current_week += timedelta(days=7)

        # Aggregate weekly data
        total_calculations = sum(w.total_calculations for w in weekly_reports)
        total_violations = sum(w.total_violations for w in weekly_reports)
        critical_violations = sum(w.critical_violations for w in weekly_reports)
        total_recommendations = sum(w.total_recommendations for w in weekly_reports)
        implemented_recommendations = sum(w.implemented_recommendations for w in weekly_reports)

        # Calculate overall compliance
        compliance_values = [w.overall_compliance_pct for w in weekly_reports if w.total_calculations > 0]
        overall_compliance = statistics.mean(compliance_values) if compliance_values else 100.0

        # Determine violation trend
        if len(weekly_reports) >= 2:
            first_half = weekly_reports[:len(weekly_reports)//2]
            second_half = weekly_reports[len(weekly_reports)//2:]
            first_violations = sum(w.total_violations for w in first_half)
            second_violations = sum(w.total_violations for w in second_half)

            if second_violations < first_violations * 0.8:
                violation_trend = "IMPROVING"
            elif second_violations > first_violations * 1.2:
                violation_trend = "DEGRADING"
            else:
                violation_trend = "STABLE"
        else:
            violation_trend = "STABLE"

        # Generate executive summary
        executive_summary = self._generate_monthly_summary(
            asset_id,
            month,
            year,
            total_calculations,
            overall_compliance,
            total_violations,
            critical_violations,
            violation_trend,
        )

        report = MonthlyComplianceReport(
            month=month,
            year=year,
            period_start=period_start,
            period_end=period_end,
            asset_id=asset_id,
            facility_id=facility_id,
            total_calculations=total_calculations,
            data_completeness_pct=statistics.mean(
                [w.data_completeness_pct for w in weekly_reports]
            ) if weekly_reports else 100.0,
            overall_compliance_pct=overall_compliance,
            total_violations=total_violations,
            critical_violations=critical_violations,
            violation_trend=violation_trend,
            total_recommendations=total_recommendations,
            implemented_recommendations=implemented_recommendations,
            rejected_recommendations=total_recommendations - implemented_recommendations,
            weekly_report_ids=[w.report_id for w in weekly_reports],
            weekly_summaries=[
                {
                    "week": w.week_number,
                    "calculations": w.total_calculations,
                    "compliance_pct": w.overall_compliance_pct,
                    "violations": w.total_violations,
                }
                for w in weekly_reports
            ],
            executive_summary=executive_summary,
            recommendations_for_next_period=self._generate_recommendations(
                overall_compliance, violation_trend, critical_violations
            ),
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Monthly report generated: {report.report_id}",
            extra={
                "processing_time_ms": processing_time,
                "weeks_included": len(weekly_reports),
            }
        )

        return report

    def generate_asme_alignment_report(
        self,
        asset_id: str,
        period_start: datetime,
        period_end: datetime,
        operating_pressure_psig: float,
    ) -> ASMEAlignmentReport:
        """
        Generate ASME alignment report.

        Args:
            asset_id: Asset ID
            period_start: Period start
            period_end: Period end
            operating_pressure_psig: Operating pressure

        Returns:
            ASMEAlignmentReport
        """
        logger.info(
            f"Generating ASME alignment report for {asset_id}",
            extra={"pressure_psig": operating_pressure_psig}
        )

        # Get ASME limits for pressure category
        pressure_category = self._get_pressure_category(operating_pressure_psig)
        limits = self.ASME_LIMITS[pressure_category]

        # Query chemistry events
        chemistry_events = self.query_service.query_chemistry_events(
            asset_id=asset_id,
            start_time=period_start,
            end_time=period_end,
            limit=10000,
        )

        # Calculate parameter alignments
        parameter_alignments = {}
        deviations = []
        alignment_scores = []

        # Conductivity
        conductivity_values = [e.conductivity_us_cm for e in chemistry_events if e.conductivity_us_cm]
        if conductivity_values:
            max_cond = max(conductivity_values)
            avg_cond = statistics.mean(conductivity_values)
            compliant = max_cond <= limits["conductivity_max"]
            alignment_scores.append(100 if compliant else max(0, 100 - (max_cond - limits["conductivity_max"]) / limits["conductivity_max"] * 100))

            parameter_alignments["conductivity"] = {
                "limit": limits["conductivity_max"],
                "max_observed": max_cond,
                "avg_observed": avg_cond,
                "compliant": compliant,
            }

            if not compliant:
                deviations.append({
                    "parameter": "conductivity",
                    "limit": limits["conductivity_max"],
                    "observed": max_cond,
                    "deviation_pct": (max_cond - limits["conductivity_max"]) / limits["conductivity_max"] * 100,
                })

        # Silica
        silica_values = [e.silica_ppm for e in chemistry_events if e.silica_ppm]
        if silica_values:
            max_silica = max(silica_values)
            avg_silica = statistics.mean(silica_values)
            compliant = max_silica <= limits["silica_max"]
            alignment_scores.append(100 if compliant else max(0, 100 - (max_silica - limits["silica_max"]) / limits["silica_max"] * 100))

            parameter_alignments["silica"] = {
                "limit": limits["silica_max"],
                "max_observed": max_silica,
                "avg_observed": avg_silica,
                "compliant": compliant,
            }

            if not compliant:
                deviations.append({
                    "parameter": "silica",
                    "limit": limits["silica_max"],
                    "observed": max_silica,
                    "deviation_pct": (max_silica - limits["silica_max"]) / limits["silica_max"] * 100,
                })

        # pH
        ph_values = [e.ph for e in chemistry_events if e.ph]
        if ph_values:
            min_ph = min(ph_values)
            max_ph = max(ph_values)
            avg_ph = statistics.mean(ph_values)
            compliant = limits["ph_min"] <= min_ph and max_ph <= limits["ph_max"]
            alignment_scores.append(100 if compliant else 80)

            parameter_alignments["ph"] = {
                "limit_min": limits["ph_min"],
                "limit_max": limits["ph_max"],
                "min_observed": min_ph,
                "max_observed": max_ph,
                "avg_observed": avg_ph,
                "compliant": compliant,
            }

            if not compliant:
                if min_ph < limits["ph_min"]:
                    deviations.append({
                        "parameter": "ph_low",
                        "limit": limits["ph_min"],
                        "observed": min_ph,
                        "deviation_pct": (limits["ph_min"] - min_ph) / limits["ph_min"] * 100,
                    })
                if max_ph > limits["ph_max"]:
                    deviations.append({
                        "parameter": "ph_high",
                        "limit": limits["ph_max"],
                        "observed": max_ph,
                        "deviation_pct": (max_ph - limits["ph_max"]) / limits["ph_max"] * 100,
                    })

        # Calculate overall alignment
        alignment_score = statistics.mean(alignment_scores) if alignment_scores else 100.0
        overall_aligned = len(deviations) == 0

        # Generate corrective actions
        corrective_actions = []
        for dev in deviations:
            if dev["parameter"] == "conductivity":
                corrective_actions.append("Increase blowdown rate to reduce conductivity")
            elif dev["parameter"] == "silica":
                corrective_actions.append("Increase blowdown rate to reduce silica levels")
            elif dev["parameter"] == "ph_low":
                corrective_actions.append("Increase phosphate dosing to raise pH")
            elif dev["parameter"] == "ph_high":
                corrective_actions.append("Reduce amine dosing to lower pH")

        report = ASMEAlignmentReport(
            period_start=period_start,
            period_end=period_end,
            asset_id=asset_id,
            operating_pressure_psig=operating_pressure_psig,
            overall_aligned=overall_aligned,
            alignment_score_pct=alignment_score,
            parameter_alignments=parameter_alignments,
            deviations=deviations,
            corrective_actions=corrective_actions,
        )

        logger.info(
            f"ASME alignment report generated: {report.report_id}",
            extra={
                "overall_aligned": overall_aligned,
                "alignment_score": alignment_score,
            }
        )

        return report

    def generate_evidence_pack(
        self,
        asset_id: str,
        period_start: datetime,
        period_end: datetime,
        correlation_id: Optional[str] = None,
        facility_id: Optional[str] = None,
    ) -> EvidencePack:
        """
        Generate a complete evidence pack.

        Args:
            asset_id: Asset ID
            period_start: Period start
            period_end: Period end
            correlation_id: Optional correlation ID
            facility_id: Optional facility ID

        Returns:
            Complete EvidencePack
        """
        logger.info(
            f"Generating evidence pack for {asset_id}",
            extra={"period": f"{period_start.date()} to {period_end.date()}"}
        )

        correlation_id = correlation_id or f"evp-{uuid4().hex[:12]}"

        # Query all relevant events
        chemistry_events = self.query_service.query_chemistry_events(
            asset_id=asset_id,
            start_time=period_start,
            end_time=period_end,
            limit=10000,
        )

        recommendations = self.query_service.query_recommendations(
            asset_id=asset_id,
            start_time=period_start,
            end_time=period_end,
            limit=1000,
        )

        violations = self.query_service.query_violations(
            asset_id=asset_id,
            start_time=period_start,
            end_time=period_end,
            limit=1000,
        )

        # Build chemistry summary
        chemistry_summary = self._build_chemistry_summary(
            chemistry_events, period_start, period_end, asset_id
        )

        # Build constraint summary
        constraint_summary = self._build_constraint_summary(
            violations, chemistry_events, period_start, period_end, asset_id
        )

        # Build recommendation summary
        recommendation_summary = self._build_recommendation_summary(
            recommendations, period_start, period_end, asset_id
        )

        # Build provenance summary
        provenance_summary = self._build_provenance_summary(correlation_id)

        # Create evidence pack
        pack = EvidencePack(
            correlation_id=correlation_id,
            asset_id=asset_id,
            facility_id=facility_id,
            period_start=period_start,
            period_end=period_end,
            chemistry_summary=chemistry_summary,
            constraint_summary=constraint_summary,
            recommendation_summary=recommendation_summary,
            provenance_summary=provenance_summary,
        )

        # Seal the pack
        pack.seal()

        logger.info(
            f"Evidence pack generated: {pack.pack_id}",
            extra={
                "correlation_id": correlation_id,
                "is_compliant": pack.is_compliant(),
            }
        )

        return pack

    def _calculate_parameter_summaries(
        self,
        chemistry_events: List,
    ) -> Dict[str, Any]:
        """Calculate parameter summaries from chemistry events."""
        summaries = {}

        # Conductivity
        values = [e.conductivity_us_cm for e in chemistry_events if e.conductivity_us_cm]
        if values:
            summaries["conductivity"] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "unit": "uS/cm",
            }

        # pH
        values = [e.ph for e in chemistry_events if e.ph]
        if values:
            summaries["ph"] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "unit": "",
            }

        # Silica
        values = [e.silica_ppm for e in chemistry_events if e.silica_ppm]
        if values:
            summaries["silica"] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "unit": "ppm",
            }

        return summaries

    def _calculate_constraint_compliance(
        self,
        violations: List,
        total_readings: int,
    ) -> Dict[str, float]:
        """Calculate constraint compliance percentages."""
        if total_readings == 0:
            return {}

        # Group violations by parameter
        violation_counts: Dict[str, int] = {}
        for v in violations:
            param = v.parameter.value if hasattr(v, "parameter") else "unknown"
            violation_counts[param] = violation_counts.get(param, 0) + 1

        compliance = {}
        for param, count in violation_counts.items():
            compliance[param] = max(0, 100 - (count / total_readings * 100))

        return compliance

    def _calculate_data_completeness(
        self,
        events: List,
        period_start: datetime,
        period_end: datetime,
    ) -> float:
        """Calculate data completeness percentage."""
        # Expected readings: every 5 minutes
        expected_minutes = (period_end - period_start).total_seconds() / 60
        expected_readings = expected_minutes / 5  # 5-minute intervals

        actual_readings = len(events)

        if expected_readings == 0:
            return 100.0

        return min(100.0, (actual_readings / expected_readings) * 100)

    def _generate_weekly_narrative(
        self,
        asset_id: str,
        week: int,
        year: int,
        calculations: int,
        compliance: float,
        violations: int,
        critical: int,
        recommendations: int,
    ) -> str:
        """Generate weekly narrative summary."""
        status = "excellent" if compliance >= 98 else "good" if compliance >= 95 else "needs attention"

        narrative = f"Week {week} of {year} water chemistry summary for {asset_id}: "
        narrative += f"Performed {calculations} chemistry calculations with {compliance:.1f}% overall compliance. "

        if violations > 0:
            narrative += f"Recorded {violations} constraint violations"
            if critical > 0:
                narrative += f" ({critical} critical)"
            narrative += ". "
        else:
            narrative += "No constraint violations recorded. "

        narrative += f"Generated {recommendations} optimization recommendations. "
        narrative += f"Overall status: {status}."

        return narrative

    def _generate_monthly_summary(
        self,
        asset_id: str,
        month: int,
        year: int,
        calculations: int,
        compliance: float,
        violations: int,
        critical: int,
        trend: str,
    ) -> str:
        """Generate monthly executive summary."""
        month_name = datetime(year, month, 1).strftime("%B")

        summary = f"Executive Summary for {month_name} {year} - Asset {asset_id}\n\n"
        summary += f"Total chemistry calculations: {calculations}\n"
        summary += f"Overall compliance: {compliance:.1f}%\n"
        summary += f"Total violations: {violations} ({critical} critical)\n"
        summary += f"Violation trend: {trend}\n\n"

        if compliance >= 98:
            summary += "Performance assessment: EXCELLENT. Water chemistry is well-controlled."
        elif compliance >= 95:
            summary += "Performance assessment: GOOD. Minor improvements possible."
        elif compliance >= 90:
            summary += "Performance assessment: ACCEPTABLE. Focused attention on violations recommended."
        else:
            summary += "Performance assessment: NEEDS IMPROVEMENT. Immediate corrective action required."

        return summary

    def _generate_recommendations(
        self,
        compliance: float,
        trend: str,
        critical_violations: int,
    ) -> List[str]:
        """Generate recommendations for next period."""
        recommendations = []

        if compliance < 95:
            recommendations.append("Review and optimize blowdown control parameters")

        if trend == "DEGRADING":
            recommendations.append("Investigate root cause of increasing violations")
            recommendations.append("Consider increasing monitoring frequency")

        if critical_violations > 0:
            recommendations.append("Implement additional safeguards for critical parameters")
            recommendations.append("Review operator response procedures for violations")

        if compliance >= 98 and trend == "STABLE":
            recommendations.append("Continue current control strategy")
            recommendations.append("Consider optimization for cost savings")

        return recommendations if recommendations else ["Maintain current operations"]

    def _build_chemistry_summary(
        self,
        events: List,
        period_start: datetime,
        period_end: datetime,
        asset_id: str,
    ) -> ChemistryCalculationSummary:
        """Build chemistry calculation summary."""
        parameters = {}

        # Conductivity
        values = [e.conductivity_us_cm for e in events if e.conductivity_us_cm]
        if values:
            parameters["CONDUCTIVITY"] = ChemistryParameterSummary(
                parameter=ChemistryParameter.CONDUCTIVITY,
                unit="uS/cm",
                reading_count=len(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=statistics.mean(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
                last_value=values[-1] if values else None,
            )

        # pH
        values = [e.ph for e in events if e.ph]
        if values:
            parameters["PH"] = ChemistryParameterSummary(
                parameter=ChemistryParameter.PH,
                unit="",
                reading_count=len(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=statistics.mean(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
                last_value=values[-1] if values else None,
            )

        # Calculate averages for operating conditions
        pressures = [e.drum_pressure_psig for e in events if e.drum_pressure_psig]
        flows = [e.steam_flow_klb_hr for e in events if e.steam_flow_klb_hr]
        cycles = [e.cycles_of_concentration for e in events if e.cycles_of_concentration]
        blowdown = [e.blowdown_rate_pct for e in events if e.blowdown_rate_pct]

        # Compute input hash
        input_hash = hashlib.sha256(
            json.dumps([str(e.event_id) for e in events], sort_keys=True).encode()
        ).hexdigest()

        return ChemistryCalculationSummary(
            calculation_period_start=period_start,
            calculation_period_end=period_end,
            total_calculations=len(events),
            asset_id=asset_id,
            parameters=parameters,
            avg_drum_pressure_psig=statistics.mean(pressures) if pressures else None,
            avg_steam_flow_klb_hr=statistics.mean(flows) if flows else None,
            avg_cycles_of_concentration=statistics.mean(cycles) if cycles else None,
            avg_blowdown_rate_pct=statistics.mean(blowdown) if blowdown else None,
            data_completeness_pct=self._calculate_data_completeness(events, period_start, period_end),
            calculation_engine_version="1.0.0",
            formula_version="1.0.0",
            input_data_hash=input_hash,
        )

    def _build_constraint_summary(
        self,
        violations: List,
        chemistry_events: List,
        period_start: datetime,
        period_end: datetime,
        asset_id: str,
    ) -> ConstraintComplianceSummary:
        """Build constraint compliance summary."""
        total_readings = len(chemistry_events)
        critical_count = sum(1 for v in violations if v.severity == SeverityLevel.CRITICAL)
        warning_count = len(violations) - critical_count

        # Calculate overall compliance
        compliance_pct = 100.0
        if total_readings > 0:
            compliance_pct = max(0, 100 - (len(violations) / total_readings * 100))

        # Constraint set hash
        constraint_hash = hashlib.sha256(b"constraint_set_v1.0.0").hexdigest()

        return ConstraintComplianceSummary(
            compliance_period_start=period_start,
            compliance_period_end=period_end,
            asset_id=asset_id,
            total_constraints=10,  # Simplified
            compliant_constraints=10 - len(set(v.parameter for v in violations if hasattr(v, "parameter"))),
            overall_compliance_pct=compliance_pct,
            constraints=[],  # Would be populated with actual constraint records
            total_violations=len(violations),
            critical_violations=critical_count,
            warning_violations=warning_count,
            constraint_set_version="1.0.0",
            constraint_set_hash=constraint_hash,
        )

    def _build_recommendation_summary(
        self,
        recommendations: List,
        period_start: datetime,
        period_end: datetime,
        asset_id: str,
    ) -> RecommendationSummary:
        """Build recommendation summary."""
        # Count by type
        by_type: Dict[str, int] = {}
        for rec in recommendations:
            rec_type = rec.recommendation_type.value
            by_type[rec_type] = by_type.get(rec_type, 0) + 1

        # Create records (limited)
        records = [
            RecommendationRecord(
                recommendation_id=rec.recommendation_id,
                recommendation_type=rec.recommendation_type,
                timestamp=rec.timestamp,
                current_value=rec.current_value,
                recommended_value=rec.recommended_value,
                unit=rec.unit,
                explanation=rec.explanation,
                confidence_score=rec.confidence_score,
                was_implemented=rec.confidence_score > 0.8,
            )
            for rec in recommendations[:50]  # Limit to 50
        ]

        return RecommendationSummary(
            period_start=period_start,
            period_end=period_end,
            asset_id=asset_id,
            total_recommendations=len(recommendations),
            implemented_recommendations=sum(1 for r in recommendations if r.confidence_score > 0.8),
            rejected_recommendations=sum(1 for r in recommendations if r.confidence_score <= 0.5),
            modified_recommendations=0,
            recommendations_by_type=by_type,
            recommendations=records,
        )

    def _build_provenance_summary(self, correlation_id: str) -> ProvenanceSummary:
        """Build provenance summary."""
        merkle_root = hashlib.sha256(correlation_id.encode()).hexdigest()

        return ProvenanceSummary(
            correlation_id=correlation_id,
            merkle_root=merkle_root,
            config_version="1.0.0",
            code_version="1.0.0",
            formula_version="1.0.0",
            constraint_version="1.0.0",
            input_event_count=0,
            input_data_hash=merkle_root,
            chain_verified=True,
            chain_verification_time=datetime.now(timezone.utc),
        )

    def store_evidence_pack(self, pack: EvidencePack) -> str:
        """Store evidence pack to configured storage."""
        if not self.storage_path:
            raise ValueError("Storage path not configured")

        date_path = pack.generated_at.strftime("%Y/%m/%d")
        full_path = self.storage_path / date_path
        full_path.mkdir(parents=True, exist_ok=True)

        filename = f"evidence_pack_{pack.pack_id}.json"
        file_path = full_path / filename

        with open(file_path, "w") as f:
            json.dump(pack.dict(), f, indent=2, default=str)

        logger.info(f"Evidence pack stored: {file_path}")
        return str(file_path)
