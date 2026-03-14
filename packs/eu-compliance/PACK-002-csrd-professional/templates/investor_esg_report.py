# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Investor ESG Report Template
================================================

Investor-focused ESG report template with ESG score dashboards,
rating agency predictions, peer benchmarking, SBTi targets,
EU Taxonomy KPIs, climate risk summaries, and target progress tracking.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class RatingAgency(str, Enum):
    """ESG rating agency."""
    MSCI = "MSCI"
    SUSTAINALYTICS = "SUSTAINALYTICS"
    CDP = "CDP"
    ISS = "ISS"
    MOODYS = "MOODYS"


class RiskLevel(str, Enum):
    """Risk level classification."""
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class TargetTrackingStatus(str, Enum):
    """Target progress tracking status."""
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    OFF_TRACK = "OFF_TRACK"
    ACHIEVED = "ACHIEVED"
    NOT_STARTED = "NOT_STARTED"


class SBTiValidationStatus(str, Enum):
    """SBTi target validation status."""
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    NOT_SET = "NOT_SET"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ESGScores(BaseModel):
    """ESG pillar scores."""
    environment_score: float = Field(
        ..., ge=0.0, le=100.0, description="Environment pillar score"
    )
    social_score: float = Field(
        ..., ge=0.0, le=100.0, description="Social pillar score"
    )
    governance_score: float = Field(
        ..., ge=0.0, le=100.0, description="Governance pillar score"
    )
    overall: float = Field(
        ..., ge=0.0, le=100.0, description="Overall ESG score"
    )


class RatingPrediction(BaseModel):
    """Rating agency score prediction."""
    agency: RatingAgency = Field(..., description="Rating agency")
    predicted_score: str = Field(..., description="Predicted score/rating")
    confidence: float = Field(
        0.0, ge=0.0, le=100.0, description="Prediction confidence %"
    )
    previous_score: Optional[str] = Field(None, description="Previous period score")
    trend: Optional[str] = Field(None, description="Trend direction")


class PeerBenchmark(BaseModel):
    """Single metric peer benchmark comparison."""
    metric: str = Field(..., description="Metric name")
    company_value: float = Field(..., description="Company value")
    peer_median: float = Field(..., description="Peer median value")
    percentile: float = Field(
        ..., ge=0.0, le=100.0, description="Percentile rank"
    )
    unit: str = Field("", description="Metric unit")
    better_direction: Optional[str] = Field(
        None, description="Direction of improvement (higher/lower)"
    )


class SBTiStatus(BaseModel):
    """SBTi target status."""
    targets_set: bool = Field(False, description="Whether targets are set")
    validation_status: SBTiValidationStatus = Field(
        SBTiValidationStatus.NOT_SET, description="Validation status"
    )
    implied_temperature: Optional[float] = Field(
        None, description="Implied temperature rise (C)"
    )
    progress_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Progress toward target"
    )
    near_term_target: Optional[str] = Field(None, description="Near-term target")
    long_term_target: Optional[str] = Field(None, description="Long-term target")


class TaxonomyKPIs(BaseModel):
    """EU Taxonomy key performance indicators."""
    gar: float = Field(0.0, ge=0.0, le=100.0, description="Green Asset Ratio %")
    btar: float = Field(0.0, ge=0.0, le=100.0, description="BTAR %")
    eligible_revenue_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Revenue eligible %"
    )
    aligned_revenue_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Revenue aligned %"
    )
    capex_aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="CapEx aligned %"
    )
    opex_aligned_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="OpEx aligned %"
    )


class ClimateRiskSummary(BaseModel):
    """High-level climate risk summary for investors."""
    physical_risk_level: RiskLevel = Field(..., description="Physical risk level")
    transition_risk_level: RiskLevel = Field(..., description="Transition risk level")
    financial_impact_low_eur: Optional[float] = Field(
        None, description="Financial impact range - low"
    )
    financial_impact_high_eur: Optional[float] = Field(
        None, description="Financial impact range - high"
    )
    key_physical_risks: List[str] = Field(
        default_factory=list, description="Key physical risks"
    )
    key_transition_risks: List[str] = Field(
        default_factory=list, description="Key transition risks"
    )


class TargetProgress(BaseModel):
    """Progress toward a sustainability target."""
    target_name: str = Field(..., description="Target name")
    base_year: int = Field(..., description="Baseline year")
    target_year: int = Field(..., description="Target year")
    base_value: float = Field(..., description="Baseline value")
    target_value: float = Field(..., description="Target value")
    current_value: float = Field(..., description="Current value")
    unit: str = Field("tCO2e", description="Value unit")
    on_track: TargetTrackingStatus = Field(
        TargetTrackingStatus.NOT_STARTED, description="Tracking status"
    )

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        total_reduction = self.base_value - self.target_value
        if total_reduction == 0:
            return 100.0
        achieved = self.base_value - self.current_value
        return min(100.0, max(0.0, (achieved / total_reduction) * 100.0))


class InvestorESGReportInput(BaseModel):
    """Complete input for the investor ESG report."""
    organization_name: str = Field(..., description="Organization name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    ticker_symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    esg_scores: ESGScores = Field(..., description="ESG pillar scores")
    rating_predictions: List[RatingPrediction] = Field(
        default_factory=list, description="Rating agency predictions"
    )
    peer_benchmarks: List[PeerBenchmark] = Field(
        default_factory=list, description="Peer comparison benchmarks"
    )
    sbti_status: Optional[SBTiStatus] = Field(
        None, description="SBTi target status"
    )
    taxonomy_kpis: Optional[TaxonomyKPIs] = Field(
        None, description="EU Taxonomy KPIs"
    )
    climate_risks: Optional[ClimateRiskSummary] = Field(
        None, description="Climate risk summary"
    )
    targets_progress: List[TargetProgress] = Field(
        default_factory=list, description="Target progress tracking"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format numeric value."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _tracking_badge(status: TargetTrackingStatus) -> str:
    """Badge for tracking status."""
    return f"[{status.value.replace('_', ' ')}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class InvestorESGReportTemplate:
    """Generate investor-focused ESG report.

    Sections:
        1. ESG Score Dashboard
        2. Rating Agency Predictions
        3. Peer Comparison
        4. SBTi Targets & Progress
        5. EU Taxonomy KPIs
        6. Climate Risk Summary
        7. Target Progress Tracker
        8. Shareholder Value Proposition

    Example:
        >>> template = InvestorESGReportTemplate()
        >>> data = InvestorESGReportInput(
        ...     organization_name="Acme", reporting_year=2025,
        ...     esg_scores=ESGScores(environment_score=75, social_score=70,
        ...                          governance_score=80, overall=75)
        ... )
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "investor_esg_report"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the investor ESG report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: InvestorESGReportInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_esg_scores(data),
            self._md_rating_predictions(data),
            self._md_peer_comparison(data),
            self._md_sbti_targets(data),
            self._md_taxonomy_kpis(data),
            self._md_climate_risks(data),
            self._md_target_progress(data),
            self._md_value_proposition(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: InvestorESGReportInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_esg_scores(data),
            self._html_rating_predictions(data),
            self._html_peer_comparison(data),
            self._html_sbti_targets(data),
            self._html_taxonomy_kpis(data),
            self._html_climate_risks(data),
            self._html_target_progress(data),
            self._html_value_proposition(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: InvestorESGReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict."""
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        result: Dict[str, Any] = {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_year": data.reporting_year,
            "ticker_symbol": data.ticker_symbol,
            "esg_scores": data.esg_scores.model_dump(mode="json"),
            "rating_predictions": [
                r.model_dump(mode="json") for r in data.rating_predictions
            ],
            "peer_benchmarks": [
                b.model_dump(mode="json") for b in data.peer_benchmarks
            ],
            "targets_progress": [
                {**t.model_dump(mode="json"), "progress_pct": t.progress_pct}
                for t in data.targets_progress
            ],
        }
        if data.sbti_status:
            result["sbti_status"] = data.sbti_status.model_dump(mode="json")
        if data.taxonomy_kpis:
            result["taxonomy_kpis"] = data.taxonomy_kpis.model_dump(mode="json")
        if data.climate_risks:
            result["climate_risks"] = data.climate_risks.model_dump(mode="json")
        return result

    def _compute_provenance(self, data: InvestorESGReportInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: InvestorESGReportInput) -> str:
        ticker = f" ({data.ticker_symbol})" if data.ticker_symbol else ""
        return (
            f"# Investor ESG Report - {data.organization_name}{ticker}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Overall ESG Score:** {data.esg_scores.overall:.0f}/100\n\n---"
        )

    def _md_esg_scores(self, data: InvestorESGReportInput) -> str:
        s = data.esg_scores
        return (
            "## 1. ESG Score Dashboard\n\n"
            "| Pillar | Score |\n"
            "|--------|-------|\n"
            f"| Environment | {s.environment_score:.0f}/100 |\n"
            f"| Social | {s.social_score:.0f}/100 |\n"
            f"| Governance | {s.governance_score:.0f}/100 |\n"
            f"| **Overall** | **{s.overall:.0f}/100** |"
        )

    def _md_rating_predictions(self, data: InvestorESGReportInput) -> str:
        if not data.rating_predictions:
            return "## 2. Rating Agency Predictions\n\nNo predictions available."
        lines = [
            "## 2. Rating Agency Predictions",
            "",
            "| Agency | Predicted Score | Confidence | Previous | Trend |",
            "|--------|----------------|------------|----------|-------|",
        ]
        for r in data.rating_predictions:
            prev = r.previous_score or "-"
            trend = r.trend or "-"
            lines.append(
                f"| {r.agency.value} | {r.predicted_score} "
                f"| {_fmt_pct(r.confidence)} | {prev} | {trend} |"
            )
        return "\n".join(lines)

    def _md_peer_comparison(self, data: InvestorESGReportInput) -> str:
        if not data.peer_benchmarks:
            return "## 3. Peer Comparison\n\nNo peer benchmark data available."
        lines = [
            "## 3. Peer Comparison",
            "",
            "| Metric | Company | Peer Median | Percentile | Unit |",
            "|--------|---------|-------------|------------|------|",
        ]
        for b in sorted(data.peer_benchmarks, key=lambda x: x.percentile, reverse=True):
            unit = b.unit or "-"
            lines.append(
                f"| {b.metric} | {_fmt_number(b.company_value, 1)} "
                f"| {_fmt_number(b.peer_median, 1)} "
                f"| {b.percentile:.0f}th | {unit} |"
            )
        return "\n".join(lines)

    def _md_sbti_targets(self, data: InvestorESGReportInput) -> str:
        if not data.sbti_status:
            return "## 4. SBTi Targets & Progress\n\nNo SBTi data available."
        s = data.sbti_status
        temp = f"{s.implied_temperature:.1f} C" if s.implied_temperature else "N/A"
        lines = [
            "## 4. SBTi Targets & Progress",
            "",
            f"- **Targets Set:** {'Yes' if s.targets_set else 'No'}",
            f"- **Validation Status:** {s.validation_status.value}",
            f"- **Implied Temperature:** {temp}",
            f"- **Progress:** {_fmt_pct(s.progress_pct)}",
        ]
        if s.near_term_target:
            lines.append(f"- **Near-term Target:** {s.near_term_target}")
        if s.long_term_target:
            lines.append(f"- **Long-term Target:** {s.long_term_target}")
        return "\n".join(lines)

    def _md_taxonomy_kpis(self, data: InvestorESGReportInput) -> str:
        if not data.taxonomy_kpis:
            return "## 5. EU Taxonomy KPIs\n\nNo Taxonomy data available."
        t = data.taxonomy_kpis
        return (
            "## 5. EU Taxonomy KPIs\n\n"
            "| KPI | Value |\n"
            "|-----|-------|\n"
            f"| Green Asset Ratio | {_fmt_pct(t.gar)} |\n"
            f"| BTAR | {_fmt_pct(t.btar)} |\n"
            f"| Revenue Eligible | {_fmt_pct(t.eligible_revenue_pct)} |\n"
            f"| Revenue Aligned | {_fmt_pct(t.aligned_revenue_pct)} |\n"
            f"| CapEx Aligned | {_fmt_pct(t.capex_aligned_pct)} |\n"
            f"| OpEx Aligned | {_fmt_pct(t.opex_aligned_pct)} |"
        )

    def _md_climate_risks(self, data: InvestorESGReportInput) -> str:
        if not data.climate_risks:
            return "## 6. Climate Risk Summary\n\nNo climate risk data available."
        cr = data.climate_risks
        impact_range = "N/A"
        if cr.financial_impact_low_eur is not None and cr.financial_impact_high_eur is not None:
            impact_range = (
                f"{_fmt_number(cr.financial_impact_low_eur, 1, ' EUR')} - "
                f"{_fmt_number(cr.financial_impact_high_eur, 1, ' EUR')}"
            )
        lines = [
            "## 6. Climate Risk Summary",
            "",
            f"- **Physical Risk Level:** {cr.physical_risk_level.value}",
            f"- **Transition Risk Level:** {cr.transition_risk_level.value}",
            f"- **Financial Impact Range:** {impact_range}",
        ]
        if cr.key_physical_risks:
            lines.append("\n**Key Physical Risks:**")
            for r in cr.key_physical_risks:
                lines.append(f"- {r}")
        if cr.key_transition_risks:
            lines.append("\n**Key Transition Risks:**")
            for r in cr.key_transition_risks:
                lines.append(f"- {r}")
        return "\n".join(lines)

    def _md_target_progress(self, data: InvestorESGReportInput) -> str:
        if not data.targets_progress:
            return "## 7. Target Progress Tracker\n\nNo targets tracked."
        lines = [
            "## 7. Target Progress Tracker",
            "",
            "| Target | Base Year | Target Year | Base | Current | Target | Progress | Status |",
            "|--------|-----------|-------------|------|---------|--------|----------|--------|",
        ]
        for t in data.targets_progress:
            lines.append(
                f"| {t.target_name} | {t.base_year} | {t.target_year} "
                f"| {_fmt_number(t.base_value, 1, f' {t.unit}')} "
                f"| {_fmt_number(t.current_value, 1, f' {t.unit}')} "
                f"| {_fmt_number(t.target_value, 1, f' {t.unit}')} "
                f"| {t.progress_pct:.0f}% | {_tracking_badge(t.on_track)} |"
            )
        return "\n".join(lines)

    def _md_value_proposition(self, data: InvestorESGReportInput) -> str:
        lines = ["## 8. Shareholder Value Proposition", ""]
        props = []
        if data.esg_scores.overall >= 75:
            props.append(
                "Strong overall ESG performance places the organization in "
                "the upper quartile of sustainability leaders."
            )
        if data.taxonomy_kpis and data.taxonomy_kpis.aligned_revenue_pct >= 30:
            props.append(
                f"{_fmt_pct(data.taxonomy_kpis.aligned_revenue_pct)} of revenue "
                f"is EU Taxonomy-aligned, demonstrating green business model alignment."
            )
        if data.sbti_status and data.sbti_status.validation_status == SBTiValidationStatus.VALIDATED:
            props.append(
                "SBTi-validated targets demonstrate credible commitment to "
                "climate action and transition readiness."
            )
        top_quartile = [b for b in data.peer_benchmarks if b.percentile >= 75]
        if top_quartile:
            props.append(
                f"Top-quartile performance on {len(top_quartile)} metric(s) "
                f"relative to sector peers."
            )
        if not props:
            props.append(
                "The organization demonstrates commitment to sustainability "
                "improvement and transparent ESG disclosure."
            )
        for i, prop in enumerate(props, 1):
            lines.append(f"{i}. {prop}")
        return "\n".join(lines)

    def _md_footer(self, data: InvestorESGReportInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, year: int, body: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Investor ESG Report - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".score-card{display:inline-block;text-align:center;padding:1.5rem 2rem;"
            "border:2px solid #0f3460;border-radius:12px;margin:0.5rem;background:#f8f9fa;}\n"
            ".score-value{font-size:2rem;font-weight:bold;color:#0f3460;}\n"
            ".score-label{font-size:0.85rem;color:#666;margin-top:0.25rem;}\n"
            ".on-track{color:#1a7f37;font-weight:bold;}\n"
            ".at-risk{color:#b08800;font-weight:bold;}\n"
            ".off-track{color:#cf222e;font-weight:bold;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: InvestorESGReportInput) -> str:
        ticker = f" ({data.ticker_symbol})" if data.ticker_symbol else ""
        return (
            '<div class="section">\n'
            f"<h1>Investor ESG Report &mdash; {data.organization_name}{ticker}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()}</p>\n"
            "<hr>\n</div>"
        )

    def _html_esg_scores(self, data: InvestorESGReportInput) -> str:
        s = data.esg_scores
        cards = [
            (f"{s.environment_score:.0f}", "Environment"),
            (f"{s.social_score:.0f}", "Social"),
            (f"{s.governance_score:.0f}", "Governance"),
            (f"{s.overall:.0f}", "Overall"),
        ]
        card_html = "\n".join(
            f'<div class="score-card"><div class="score-value">{v}</div>'
            f'<div class="score-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>1. ESG Score Dashboard</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_rating_predictions(self, data: InvestorESGReportInput) -> str:
        if not data.rating_predictions:
            return (
                '<div class="section"><h2>2. Rating Predictions</h2>'
                "<p>No predictions available.</p></div>"
            )
        rows = []
        for r in data.rating_predictions:
            prev = r.previous_score or "-"
            trend = r.trend or "-"
            rows.append(
                f"<tr><td>{r.agency.value}</td><td><strong>{r.predicted_score}</strong></td>"
                f"<td>{_fmt_pct(r.confidence)}</td><td>{prev}</td><td>{trend}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>2. Rating Agency Predictions</h2>\n'
            "<table><thead><tr><th>Agency</th><th>Predicted</th>"
            "<th>Confidence</th><th>Previous</th><th>Trend</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_peer_comparison(self, data: InvestorESGReportInput) -> str:
        if not data.peer_benchmarks:
            return (
                '<div class="section"><h2>3. Peer Comparison</h2>'
                "<p>No data available.</p></div>"
            )
        rows = []
        for b in sorted(data.peer_benchmarks, key=lambda x: x.percentile, reverse=True):
            unit = b.unit or "-"
            css = "on-track" if b.percentile >= 75 else ("at-risk" if b.percentile >= 50 else "off-track")
            rows.append(
                f"<tr><td>{b.metric}</td><td>{_fmt_number(b.company_value, 1)}</td>"
                f"<td>{_fmt_number(b.peer_median, 1)}</td>"
                f'<td class="{css}">{b.percentile:.0f}th</td><td>{unit}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>3. Peer Comparison</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Company</th>"
            "<th>Peer Median</th><th>Percentile</th><th>Unit</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_sbti_targets(self, data: InvestorESGReportInput) -> str:
        if not data.sbti_status:
            return (
                '<div class="section"><h2>4. SBTi Targets</h2>'
                "<p>No data available.</p></div>"
            )
        s = data.sbti_status
        temp = f"{s.implied_temperature:.1f} C" if s.implied_temperature else "N/A"
        cards = [
            (s.validation_status.value, "Status"),
            (temp, "Implied Temp"),
            (_fmt_pct(s.progress_pct), "Progress"),
        ]
        card_html = "\n".join(
            f'<div class="score-card"><div class="score-value">{v}</div>'
            f'<div class="score-label">{l}</div></div>'
            for v, l in cards
        )
        details = "<ul>\n"
        if s.near_term_target:
            details += f"<li><strong>Near-term:</strong> {s.near_term_target}</li>\n"
        if s.long_term_target:
            details += f"<li><strong>Long-term:</strong> {s.long_term_target}</li>\n"
        details += "</ul>"
        return (
            '<div class="section">\n<h2>4. SBTi Targets &amp; Progress</h2>\n'
            f"<div>{card_html}</div>\n{details}\n</div>"
        )

    def _html_taxonomy_kpis(self, data: InvestorESGReportInput) -> str:
        if not data.taxonomy_kpis:
            return (
                '<div class="section"><h2>5. EU Taxonomy KPIs</h2>'
                "<p>No data available.</p></div>"
            )
        t = data.taxonomy_kpis
        rows = [
            f"<tr><td>Green Asset Ratio</td><td>{_fmt_pct(t.gar)}</td></tr>",
            f"<tr><td>BTAR</td><td>{_fmt_pct(t.btar)}</td></tr>",
            f"<tr><td>Revenue Eligible</td><td>{_fmt_pct(t.eligible_revenue_pct)}</td></tr>",
            f"<tr><td>Revenue Aligned</td><td>{_fmt_pct(t.aligned_revenue_pct)}</td></tr>",
            f"<tr><td>CapEx Aligned</td><td>{_fmt_pct(t.capex_aligned_pct)}</td></tr>",
            f"<tr><td>OpEx Aligned</td><td>{_fmt_pct(t.opex_aligned_pct)}</td></tr>",
        ]
        return (
            '<div class="section">\n<h2>5. EU Taxonomy KPIs</h2>\n'
            "<table><thead><tr><th>KPI</th><th>Value</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_climate_risks(self, data: InvestorESGReportInput) -> str:
        if not data.climate_risks:
            return (
                '<div class="section"><h2>6. Climate Risk Summary</h2>'
                "<p>No data available.</p></div>"
            )
        cr = data.climate_risks
        impact_range = "N/A"
        if cr.financial_impact_low_eur is not None and cr.financial_impact_high_eur is not None:
            impact_range = (
                f"{_fmt_number(cr.financial_impact_low_eur, 1, ' EUR')} - "
                f"{_fmt_number(cr.financial_impact_high_eur, 1, ' EUR')}"
            )
        parts = [
            '<div class="section">\n<h2>6. Climate Risk Summary</h2>\n',
            f'<div class="score-card"><div class="score-value">'
            f'{cr.physical_risk_level.value}</div>'
            f'<div class="score-label">Physical Risk</div></div>\n',
            f'<div class="score-card"><div class="score-value">'
            f'{cr.transition_risk_level.value}</div>'
            f'<div class="score-label">Transition Risk</div></div>\n',
            f"<p><strong>Financial Impact Range:</strong> {impact_range}</p>\n",
        ]
        if cr.key_physical_risks:
            items = "".join(f"<li>{r}</li>" for r in cr.key_physical_risks)
            parts.append(f"<p><strong>Key Physical Risks:</strong></p><ul>{items}</ul>\n")
        if cr.key_transition_risks:
            items = "".join(f"<li>{r}</li>" for r in cr.key_transition_risks)
            parts.append(f"<p><strong>Key Transition Risks:</strong></p><ul>{items}</ul>\n")
        parts.append("</div>")
        return "".join(parts)

    def _html_target_progress(self, data: InvestorESGReportInput) -> str:
        if not data.targets_progress:
            return (
                '<div class="section"><h2>7. Target Progress</h2>'
                "<p>No targets tracked.</p></div>"
            )
        rows = []
        for t in data.targets_progress:
            css_map = {
                TargetTrackingStatus.ON_TRACK: "on-track",
                TargetTrackingStatus.ACHIEVED: "on-track",
                TargetTrackingStatus.AT_RISK: "at-risk",
                TargetTrackingStatus.OFF_TRACK: "off-track",
                TargetTrackingStatus.NOT_STARTED: "",
            }
            css = css_map.get(t.on_track, "")
            rows.append(
                f"<tr><td>{t.target_name}</td><td>{t.base_year}-{t.target_year}</td>"
                f"<td>{_fmt_number(t.current_value, 1, f' {t.unit}')}</td>"
                f"<td>{t.progress_pct:.0f}%</td>"
                f'<td class="{css}">{t.on_track.value.replace("_", " ")}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>7. Target Progress Tracker</h2>\n'
            "<table><thead><tr><th>Target</th><th>Period</th>"
            "<th>Current</th><th>Progress</th><th>Status</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_value_proposition(self, data: InvestorESGReportInput) -> str:
        props = []
        if data.esg_scores.overall >= 75:
            props.append("Strong ESG performance in the upper quartile.")
        if data.taxonomy_kpis and data.taxonomy_kpis.aligned_revenue_pct >= 30:
            props.append(
                f"{_fmt_pct(data.taxonomy_kpis.aligned_revenue_pct)} Taxonomy-aligned revenue."
            )
        if data.sbti_status and data.sbti_status.validation_status == SBTiValidationStatus.VALIDATED:
            props.append("SBTi-validated climate targets.")
        if not props:
            props.append("Committed to transparent ESG disclosure and improvement.")
        items = "".join(f"<li>{p}</li>" for p in props)
        return (
            '<div class="section">\n<h2>8. Shareholder Value Proposition</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: InvestorESGReportInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
