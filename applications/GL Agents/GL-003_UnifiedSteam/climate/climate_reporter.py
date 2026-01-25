"""
Climate Reporter for GL-003 UNIFIEDSTEAM

Provides climate impact reporting, sustainability dashboards, and
compliance documentation for steam system operations.

Author: GL-003 Climate Intelligence Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

from .co2e_calculator import (
    CO2eCalculator,
    ClimateImpactResult,
    EmissionsBreakdown,
    SteamCarbonIntensity,
)
from .emission_factors import EmissionScope, FuelType, GridRegion
from .m_and_v import MVReport, SavingsType

logger = logging.getLogger(__name__)


class ReportFrequency(Enum):
    """Reporting frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class ComplianceStandard(Enum):
    """Compliance reporting standards."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CDP = "cdp"
    TCFD = "tcfd"
    EU_ETS = "eu_ets"
    SEC_CLIMATE = "sec_climate"


@dataclass
class KPIMetric:
    """Key performance indicator metric."""
    name: str
    value: Decimal
    unit: str
    target: Optional[Decimal] = None
    trend: str = "neutral"  # up, down, neutral
    period: str = ""
    notes: str = ""


@dataclass
class ClimateImpactSummary:
    """
    Summary of climate impact for a reporting period.

    Provides executive-level metrics and trends.
    """
    period_start: datetime
    period_end: datetime
    total_co2e_tonnes: Decimal
    scope_1_tonnes: Decimal
    scope_2_tonnes: Decimal
    scope_3_tonnes: Decimal
    intensity_kg_per_gj: Decimal
    intensity_kg_per_tonne_steam: Decimal
    vs_baseline_pct: Decimal
    vs_target_pct: Optional[Decimal] = None
    kpis: List[KPIMetric] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "emissions": {
                "total_co2e_tonnes": str(self.total_co2e_tonnes),
                "scope_1_tonnes": str(self.scope_1_tonnes),
                "scope_2_tonnes": str(self.scope_2_tonnes),
                "scope_3_tonnes": str(self.scope_3_tonnes),
            },
            "intensity": {
                "kg_per_gj": str(self.intensity_kg_per_gj),
                "kg_per_tonne_steam": str(self.intensity_kg_per_tonne_steam),
            },
            "performance": {
                "vs_baseline_pct": str(self.vs_baseline_pct),
                "vs_target_pct": str(self.vs_target_pct) if self.vs_target_pct else None,
            },
            "kpis": [
                {
                    "name": k.name,
                    "value": str(k.value),
                    "unit": k.unit,
                    "target": str(k.target) if k.target else None,
                    "trend": k.trend,
                }
                for k in self.kpis
            ],
            "highlights": self.highlights,
            "risks": self.risks,
        }


@dataclass
class SustainabilityDashboard:
    """
    Sustainability dashboard data for visualization.

    Provides time series, breakdowns, and comparisons.
    """
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Time series data
    emissions_time_series: List[Dict[str, Any]]
    intensity_time_series: List[Dict[str, Any]]

    # Breakdowns
    scope_breakdown: Dict[str, Decimal]
    fuel_breakdown: Dict[str, Decimal]
    asset_breakdown: Dict[str, Decimal]

    # Comparisons
    period_comparison: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]

    # KPIs
    summary_metrics: ClimateImpactSummary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "time_series": {
                "emissions": self.emissions_time_series,
                "intensity": self.intensity_time_series,
            },
            "breakdowns": {
                "by_scope": {k: str(v) for k, v in self.scope_breakdown.items()},
                "by_fuel": {k: str(v) for k, v in self.fuel_breakdown.items()},
                "by_asset": {k: str(v) for k, v in self.asset_breakdown.items()},
            },
            "comparisons": {
                "period": self.period_comparison,
                "benchmark": self.benchmark_comparison,
            },
            "summary": self.summary_metrics.to_dict(),
        }


@dataclass
class ComplianceReport:
    """
    Compliance report for regulatory requirements.

    Structured according to reporting standards.
    """
    report_id: str
    standard: ComplianceStandard
    reporting_year: int
    organization: str
    facility: str

    # Emissions data
    scope_1_total: Decimal
    scope_2_location: Decimal
    scope_2_market: Optional[Decimal] = None
    scope_3_total: Optional[Decimal] = None

    # Methodology
    calculation_methodology: str
    emission_factor_sources: List[str] = field(default_factory=list)
    data_quality_assessment: str = ""

    # Verification
    verification_status: str = "unverified"
    verifier: Optional[str] = None
    verification_date: Optional[datetime] = None

    # Attestations
    prepared_by: str = ""
    reviewed_by: str = ""
    approved_by: str = ""

    # Supporting data
    supporting_documents: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "standard": self.standard.value,
            "reporting_year": self.reporting_year,
            "organization": self.organization,
            "facility": self.facility,
            "emissions": {
                "scope_1_total_tonnes": str(self.scope_1_total),
                "scope_2_location_tonnes": str(self.scope_2_location),
                "scope_2_market_tonnes": str(self.scope_2_market) if self.scope_2_market else None,
                "scope_3_total_tonnes": str(self.scope_3_total) if self.scope_3_total else None,
            },
            "methodology": {
                "calculation": self.calculation_methodology,
                "emission_factor_sources": self.emission_factor_sources,
                "data_quality": self.data_quality_assessment,
            },
            "verification": {
                "status": self.verification_status,
                "verifier": self.verifier,
                "date": self.verification_date.isoformat() if self.verification_date else None,
            },
            "attestations": {
                "prepared_by": self.prepared_by,
                "reviewed_by": self.reviewed_by,
                "approved_by": self.approved_by,
            },
            "supporting_documents": self.supporting_documents,
            "audit_trail": self.audit_trail,
        }


class ClimateReporter:
    """
    Climate impact reporter for GL-003 UNIFIEDSTEAM.

    Generates summaries, dashboards, and compliance reports
    for steam system climate impact.
    """

    def __init__(
        self,
        co2e_calculator: Optional[CO2eCalculator] = None,
        organization: str = "GreenLang",
        facility: str = "Steam Facility",
    ):
        """
        Initialize climate reporter.

        Args:
            co2e_calculator: CO2e calculator instance
            organization: Organization name for reports
            facility: Facility name for reports
        """
        self.calculator = co2e_calculator or CO2eCalculator()
        self.organization = organization
        self.facility = facility
        self._impact_history: List[ClimateImpactResult] = []
        self._reports: Dict[str, ComplianceReport] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def record_impact(self, impact: ClimateImpactResult):
        """
        Record a climate impact calculation.

        Args:
            impact: ClimateImpactResult to record
        """
        self._impact_history.append(impact)
        self._log_action("record_impact", {
            "calculation_id": impact.calculation_id,
            "total_co2e_kg": str(impact.total_co2e_kg),
        })

    def generate_summary(
        self,
        period_start: datetime,
        period_end: datetime,
        baseline_co2e_kg: Optional[Decimal] = None,
        target_co2e_kg: Optional[Decimal] = None,
    ) -> ClimateImpactSummary:
        """
        Generate climate impact summary for a period.

        Args:
            period_start: Start of reporting period
            period_end: End of reporting period
            baseline_co2e_kg: Baseline emissions for comparison
            target_co2e_kg: Target emissions for comparison

        Returns:
            ClimateImpactSummary
        """
        # Filter impacts for period
        period_impacts = [
            i for i in self._impact_history
            if period_start <= i.period_start and i.period_end <= period_end
        ]

        if not period_impacts:
            # Return empty summary if no data
            return ClimateImpactSummary(
                period_start=period_start,
                period_end=period_end,
                total_co2e_tonnes=Decimal("0"),
                scope_1_tonnes=Decimal("0"),
                scope_2_tonnes=Decimal("0"),
                scope_3_tonnes=Decimal("0"),
                intensity_kg_per_gj=Decimal("0"),
                intensity_kg_per_tonne_steam=Decimal("0"),
                vs_baseline_pct=Decimal("0"),
            )

        # Aggregate emissions
        total_co2e = sum(i.total_co2e_kg for i in period_impacts)
        scope_1 = sum(i.scope_1_total_kg for i in period_impacts)
        scope_2 = sum(i.scope_2_total_kg for i in period_impacts)
        scope_3 = sum(i.scope_3_total_kg for i in period_impacts)

        # Calculate average intensity
        total_steam = sum(i.total_steam_gj for i in period_impacts)
        avg_intensity_gj = total_co2e / total_steam if total_steam > 0 else Decimal("0")

        # Get average intensity per tonne from individual impacts
        intensities = [i.steam_carbon_intensity.intensity_kg_co2e_per_tonne
                      for i in period_impacts]
        avg_intensity_tonne = (
            sum(intensities) / len(intensities) if intensities else Decimal("0")
        )

        # Calculate vs baseline
        vs_baseline = Decimal("0")
        if baseline_co2e_kg and baseline_co2e_kg > 0:
            vs_baseline = ((total_co2e - baseline_co2e_kg) / baseline_co2e_kg) * Decimal("100")

        # Calculate vs target
        vs_target = None
        if target_co2e_kg and target_co2e_kg > 0:
            vs_target = ((total_co2e - target_co2e_kg) / target_co2e_kg) * Decimal("100")

        # Build KPIs
        kpis = [
            KPIMetric(
                name="Total Emissions",
                value=(total_co2e / Decimal("1000")).quantize(Decimal("0.1")),
                unit="tonnes CO2e",
                trend="down" if vs_baseline < 0 else "up" if vs_baseline > 0 else "neutral",
            ),
            KPIMetric(
                name="Carbon Intensity",
                value=avg_intensity_gj.quantize(Decimal("0.01")),
                unit="kg CO2e/GJ",
            ),
            KPIMetric(
                name="Total Steam Energy",
                value=total_steam.quantize(Decimal("0.1")),
                unit="GJ",
            ),
        ]

        # Generate highlights
        highlights = []
        if vs_baseline < Decimal("-5"):
            highlights.append(
                f"Emissions reduced by {abs(vs_baseline):.1f}% vs baseline"
            )
        if len(period_impacts) > 0:
            highlights.append(
                f"Processed {len(period_impacts)} impact calculations"
            )

        # Generate risks
        risks = []
        if vs_baseline > Decimal("5"):
            risks.append(
                f"Emissions increased by {vs_baseline:.1f}% vs baseline"
            )

        summary = ClimateImpactSummary(
            period_start=period_start,
            period_end=period_end,
            total_co2e_tonnes=(total_co2e / Decimal("1000")).quantize(Decimal("0.01")),
            scope_1_tonnes=(scope_1 / Decimal("1000")).quantize(Decimal("0.01")),
            scope_2_tonnes=(scope_2 / Decimal("1000")).quantize(Decimal("0.01")),
            scope_3_tonnes=(scope_3 / Decimal("1000")).quantize(Decimal("0.01")),
            intensity_kg_per_gj=avg_intensity_gj.quantize(Decimal("0.01")),
            intensity_kg_per_tonne_steam=avg_intensity_tonne.quantize(Decimal("0.01")),
            vs_baseline_pct=vs_baseline.quantize(Decimal("0.1")),
            vs_target_pct=vs_target.quantize(Decimal("0.1")) if vs_target else None,
            kpis=kpis,
            highlights=highlights,
            risks=risks,
        )

        self._log_action("generate_summary", {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_co2e_tonnes": str(summary.total_co2e_tonnes),
        })

        return summary

    def generate_dashboard(
        self,
        period_start: datetime,
        period_end: datetime,
        granularity: ReportFrequency = ReportFrequency.DAILY,
        baseline_co2e_kg: Optional[Decimal] = None,
    ) -> SustainabilityDashboard:
        """
        Generate sustainability dashboard data.

        Args:
            period_start: Start of period
            period_end: End of period
            granularity: Time series granularity
            baseline_co2e_kg: Baseline for comparison

        Returns:
            SustainabilityDashboard with all visualization data
        """
        # Generate summary
        summary = self.generate_summary(
            period_start, period_end, baseline_co2e_kg
        )

        # Build time series (simplified - would aggregate by granularity)
        emissions_ts = []
        intensity_ts = []

        for impact in self._impact_history:
            if period_start <= impact.period_start <= period_end:
                emissions_ts.append({
                    "date": impact.period_start.isoformat(),
                    "total_kg": str(impact.total_co2e_kg),
                    "scope_1_kg": str(impact.scope_1_total_kg),
                    "scope_2_kg": str(impact.scope_2_total_kg),
                })
                intensity_ts.append({
                    "date": impact.period_start.isoformat(),
                    "kg_per_gj": str(impact.steam_carbon_intensity.intensity_kg_co2e_per_gj),
                })

        # Calculate breakdowns
        scope_breakdown = {
            "scope_1": summary.scope_1_tonnes,
            "scope_2": summary.scope_2_tonnes,
            "scope_3": summary.scope_3_tonnes,
        }

        # Fuel breakdown (simplified)
        fuel_breakdown: Dict[str, Decimal] = {}
        for impact in self._impact_history:
            if period_start <= impact.period_start <= period_end:
                for fuel, share in impact.steam_carbon_intensity.fuel_mix.items():
                    fuel_name = fuel.value
                    fuel_breakdown[fuel_name] = fuel_breakdown.get(
                        fuel_name, Decimal("0")
                    ) + share

        # Asset breakdown (placeholder)
        asset_breakdown = {"steam_system": summary.total_co2e_tonnes}

        # Period comparison (vs previous period)
        period_comparison = {
            "current_tonnes": str(summary.total_co2e_tonnes),
            "previous_tonnes": str(baseline_co2e_kg / Decimal("1000")) if baseline_co2e_kg else "N/A",
            "change_pct": str(summary.vs_baseline_pct),
        }

        # Benchmark comparison (placeholder)
        benchmark_comparison = {
            "facility_intensity": str(summary.intensity_kg_per_gj),
            "industry_benchmark": "65.0",  # Example benchmark
            "performance": "below_average" if summary.intensity_kg_per_gj > 65 else "above_average",
        }

        dashboard = SustainabilityDashboard(
            generated_at=datetime.now(timezone.utc),
            period_start=period_start,
            period_end=period_end,
            emissions_time_series=emissions_ts,
            intensity_time_series=intensity_ts,
            scope_breakdown=scope_breakdown,
            fuel_breakdown=fuel_breakdown,
            asset_breakdown=asset_breakdown,
            period_comparison=period_comparison,
            benchmark_comparison=benchmark_comparison,
            summary_metrics=summary,
        )

        return dashboard

    def generate_compliance_report(
        self,
        reporting_year: int,
        standard: ComplianceStandard = ComplianceStandard.GHG_PROTOCOL,
        prepared_by: str = "",
    ) -> ComplianceReport:
        """
        Generate compliance report for regulatory requirements.

        Args:
            reporting_year: Year to report
            standard: Compliance standard to follow
            prepared_by: Name of preparer

        Returns:
            ComplianceReport
        """
        import uuid

        # Filter impacts for the year
        year_start = datetime(reporting_year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(reporting_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        year_impacts = [
            i for i in self._impact_history
            if year_start <= i.period_start <= year_end
        ]

        # Aggregate emissions
        scope_1_total = sum(i.scope_1_total_kg for i in year_impacts)
        scope_2_total = sum(i.scope_2_total_kg for i in year_impacts)
        scope_3_total = sum(i.scope_3_total_kg for i in year_impacts)

        # Convert to tonnes
        scope_1_tonnes = scope_1_total / Decimal("1000")
        scope_2_tonnes = scope_2_total / Decimal("1000")
        scope_3_tonnes = scope_3_total / Decimal("1000")

        report_id = f"COMPLIANCE-{reporting_year}-{uuid.uuid4().hex[:6].upper()}"

        report = ComplianceReport(
            report_id=report_id,
            standard=standard,
            reporting_year=reporting_year,
            organization=self.organization,
            facility=self.facility,
            scope_1_total=scope_1_tonnes.quantize(Decimal("0.01")),
            scope_2_location=scope_2_tonnes.quantize(Decimal("0.01")),
            scope_3_total=scope_3_tonnes.quantize(Decimal("0.01")),
            calculation_methodology=(
                "GHG Protocol Corporate Standard using operational control approach"
            ),
            emission_factor_sources=[
                "EPA Emission Factors Hub 2024",
                "EPA eGRID 2022",
                "IEA Emission Factors 2023",
            ],
            data_quality_assessment=(
                "Primary data from continuous monitoring systems; "
                "emission factors from authoritative sources"
            ),
            prepared_by=prepared_by,
            audit_trail=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "report_generated",
                "report_id": report_id,
            }],
        )

        self._reports[report_id] = report

        self._log_action("generate_compliance_report", {
            "report_id": report_id,
            "standard": standard.value,
            "year": reporting_year,
        })

        return report

    def export_report(
        self,
        report_id: str,
        format: str = "json",
    ) -> str:
        """
        Export report in specified format.

        Args:
            report_id: Report ID to export
            format: Export format (json, csv)

        Returns:
            Formatted report string
        """
        if report_id not in self._reports:
            raise KeyError(f"Report not found: {report_id}")

        report = self._reports[report_id]

        if format == "json":
            return json.dumps(report.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log."""
        return self._audit_log.copy()
