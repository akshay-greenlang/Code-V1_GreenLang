# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Scenario Analysis Report Template
=====================================================

Climate scenario analysis results template covering physical risks,
transition risks, financial impacts, resilience assessment, and
marginal abatement cost curve (MACC) data. Aligned with TCFD/ESRS E1
scenario analysis requirements.

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

class ScenarioType(str, Enum):
    """Climate scenario type."""
    PHYSICAL = "PHYSICAL"
    TRANSITION = "TRANSITION"
    COMBINED = "COMBINED"


class TimeHorizon(str, Enum):
    """Time horizon classification."""
    SHORT = "SHORT_TERM"
    MEDIUM = "MEDIUM_TERM"
    LONG = "LONG_TERM"


class RiskLikelihood(str, Enum):
    """Likelihood rating."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class RiskMagnitude(str, Enum):
    """Impact magnitude rating."""
    SEVERE = "SEVERE"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"


class PhysicalRiskType(str, Enum):
    """Physical risk type classification."""
    ACUTE_FLOODING = "ACUTE_FLOODING"
    ACUTE_WILDFIRE = "ACUTE_WILDFIRE"
    ACUTE_STORM = "ACUTE_STORM"
    ACUTE_HEATWAVE = "ACUTE_HEATWAVE"
    CHRONIC_SEA_LEVEL = "CHRONIC_SEA_LEVEL"
    CHRONIC_TEMPERATURE = "CHRONIC_TEMPERATURE"
    CHRONIC_WATER_STRESS = "CHRONIC_WATER_STRESS"
    CHRONIC_PRECIPITATION = "CHRONIC_PRECIPITATION"


class TransitionRiskDriver(str, Enum):
    """Transition risk driver category."""
    POLICY = "POLICY"
    LEGAL = "LEGAL"
    TECHNOLOGY = "TECHNOLOGY"
    MARKET = "MARKET"
    REPUTATION = "REPUTATION"


class ResilienceLevel(str, Enum):
    """Overall resilience assessment level."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ScenarioSummary(BaseModel):
    """Summary of a single climate scenario analyzed."""
    name: str = Field(..., description="Scenario name (e.g. SSP1-2.6)")
    scenario_type: ScenarioType = Field(..., description="Physical/Transition/Combined")
    temperature_target: float = Field(
        ..., description="Temperature target in degrees C"
    )
    time_horizon: str = Field(..., description="Time horizon (e.g. 2030, 2050)")
    source: str = Field(..., description="Scenario source (IPCC, IEA, NGFS)")
    description: Optional[str] = Field(None, description="Brief scenario description")


class PhysicalRiskEntry(BaseModel):
    """Physical risk assessment for an asset or location."""
    asset: str = Field(..., description="Asset or location name")
    risk_type: PhysicalRiskType = Field(..., description="Physical risk type")
    exposure_score: float = Field(
        ..., ge=0.0, le=10.0, description="Exposure score 0-10"
    )
    vulnerability_score: float = Field(
        0.0, ge=0.0, le=10.0, description="Vulnerability score 0-10"
    )
    overall_score: float = Field(
        ..., ge=0.0, le=10.0, description="Overall risk score 0-10"
    )
    adaptation_measures: List[str] = Field(
        default_factory=list, description="Identified adaptation measures"
    )
    financial_impact_eur: Optional[float] = Field(
        None, description="Estimated financial impact in EUR"
    )


class TransitionRiskEntry(BaseModel):
    """Transition risk assessment entry."""
    risk_type: str = Field(..., description="Specific risk description")
    driver: TransitionRiskDriver = Field(..., description="Risk driver category")
    likelihood: RiskLikelihood = Field(..., description="Likelihood rating")
    magnitude: RiskMagnitude = Field(..., description="Impact magnitude")
    time_horizon: TimeHorizon = Field(..., description="Time horizon")
    mitigation: str = Field("", description="Mitigation strategy")
    financial_impact_eur: Optional[float] = Field(
        None, description="Estimated financial impact in EUR"
    )


class FinancialImpactEntry(BaseModel):
    """Financial impact under a specific scenario."""
    scenario: str = Field(..., description="Scenario name")
    metric: str = Field(..., description="Financial metric")
    baseline_value: float = Field(..., description="Baseline value")
    impact_value: float = Field(..., description="Impact value under scenario")
    impact_pct: float = Field(..., description="Impact as percentage of baseline")
    unit: str = Field("EUR", description="Value unit")


class ResilienceAssessment(BaseModel):
    """Overall resilience assessment."""
    overall_level: ResilienceLevel = Field(..., description="Overall resilience level")
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall resilience score 0-100"
    )
    strengths: List[str] = Field(
        default_factory=list, description="Identified strengths"
    )
    vulnerabilities: List[str] = Field(
        default_factory=list, description="Identified vulnerabilities"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended resilience actions"
    )


class MACCEntry(BaseModel):
    """Marginal abatement cost curve entry."""
    abatement_option: str = Field(..., description="Abatement option name")
    cost_per_tco2: float = Field(..., description="Cost per tCO2e avoided (EUR)")
    potential_tco2: float = Field(
        ..., ge=0.0, description="Abatement potential in tCO2e"
    )
    implementation_year: Optional[int] = Field(
        None, description="Target implementation year"
    )
    category: Optional[str] = Field(
        None, description="Abatement category (energy efficiency, renewables, etc.)"
    )


class ScenarioAnalysisReportInput(BaseModel):
    """Complete input for the scenario analysis report."""
    organization_name: str = Field(..., description="Organization name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    scenarios_analyzed: List[ScenarioSummary] = Field(
        default_factory=list, description="Scenarios analyzed"
    )
    physical_risks: List[PhysicalRiskEntry] = Field(
        default_factory=list, description="Physical risk entries"
    )
    transition_risks: List[TransitionRiskEntry] = Field(
        default_factory=list, description="Transition risk entries"
    )
    financial_impacts: List[FinancialImpactEntry] = Field(
        default_factory=list, description="Financial impact entries"
    )
    resilience_assessment: Optional[ResilienceAssessment] = Field(
        None, description="Overall resilience assessment"
    )
    macc_curves: Optional[List[MACCEntry]] = Field(
        None, description="MACC curve data"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format numeric value with thousands separator."""
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
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_eur(value: Optional[float]) -> str:
    """Format EUR value."""
    return _fmt_number(value, decimals=1, suffix=" EUR")


def _likelihood_sort(likelihood: RiskLikelihood) -> int:
    """Sort key for likelihood."""
    return {
        RiskLikelihood.VERY_HIGH: 0,
        RiskLikelihood.HIGH: 1,
        RiskLikelihood.MEDIUM: 2,
        RiskLikelihood.LOW: 3,
        RiskLikelihood.VERY_LOW: 4,
    }.get(likelihood, 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ScenarioAnalysisReportTemplate:
    """Generate climate scenario analysis results report.

    Sections:
        1. Scenario Overview Matrix
        2. Physical Risk Heatmap
        3. Transition Risk Matrix
        4. Financial Impact Comparison
        5. MACC Curve Data
        6. Resilience Assessment
        7. Recommendations

    Example:
        >>> template = ScenarioAnalysisReportTemplate()
        >>> data = ScenarioAnalysisReportInput(
        ...     organization_name="Acme", reporting_year=2025
        ... )
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "scenario_analysis_report"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the scenario analysis report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: ScenarioAnalysisReportInput) -> str:
        """Render as Markdown.

        Args:
            data: Validated scenario analysis input.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_scenario_overview(data),
            self._md_physical_risks(data),
            self._md_transition_risks(data),
            self._md_financial_impacts(data),
            self._md_macc_curves(data),
            self._md_resilience(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: ScenarioAnalysisReportInput) -> str:
        """Render as HTML document.

        Args:
            data: Validated scenario analysis input.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_scenario_overview(data),
            self._html_physical_risks(data),
            self._html_transition_risks(data),
            self._html_financial_impacts(data),
            self._html_macc_curves(data),
            self._html_resilience(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: ScenarioAnalysisReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated scenario analysis input.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)

        result: Dict[str, Any] = {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_year": data.reporting_year,
            "scenarios_analyzed": [
                s.model_dump(mode="json") for s in data.scenarios_analyzed
            ],
            "physical_risks": [
                r.model_dump(mode="json") for r in data.physical_risks
            ],
            "transition_risks": [
                r.model_dump(mode="json") for r in data.transition_risks
            ],
            "financial_impacts": [
                f.model_dump(mode="json") for f in data.financial_impacts
            ],
        }
        if data.resilience_assessment:
            result["resilience_assessment"] = data.resilience_assessment.model_dump(
                mode="json"
            )
        if data.macc_curves:
            result["macc_curves"] = [
                m.model_dump(mode="json") for m in data.macc_curves
            ]
        return result

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: ScenarioAnalysisReportInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ScenarioAnalysisReportInput) -> str:
        """Markdown header."""
        return (
            f"# Climate Scenario Analysis Report - {data.organization_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Scenarios:** {len(data.scenarios_analyzed)}\n\n---"
        )

    def _md_scenario_overview(self, data: ScenarioAnalysisReportInput) -> str:
        """Scenario overview matrix."""
        lines = [
            "## 1. Scenario Overview",
            "",
            "| Scenario | Type | Temp Target | Horizon | Source |",
            "|----------|------|-------------|---------|--------|",
        ]
        for s in data.scenarios_analyzed:
            lines.append(
                f"| {s.name} | {s.scenario_type.value} "
                f"| {s.temperature_target:.1f} C "
                f"| {s.time_horizon} | {s.source} |"
            )
        if not data.scenarios_analyzed:
            lines.append("| - | No scenarios analyzed | - | - | - |")
        return "\n".join(lines)

    def _md_physical_risks(self, data: ScenarioAnalysisReportInput) -> str:
        """Physical risk heatmap table."""
        if not data.physical_risks:
            return "## 2. Physical Risk Assessment\n\nNo physical risk data available."
        lines = [
            "## 2. Physical Risk Assessment",
            "",
            "| Asset | Risk Type | Exposure | Vulnerability | Overall | Impact (EUR) |",
            "|-------|-----------|----------|---------------|---------|-------------|",
        ]
        for r in sorted(data.physical_risks, key=lambda x: x.overall_score, reverse=True):
            impact = _fmt_eur(r.financial_impact_eur)
            risk_type = r.risk_type.value.replace("_", " ").title()
            lines.append(
                f"| {r.asset} | {risk_type} "
                f"| {r.exposure_score:.1f}/10 | {r.vulnerability_score:.1f}/10 "
                f"| {r.overall_score:.1f}/10 | {impact} |"
            )
        # Adaptation measures summary
        lines.extend(["", "### Adaptation Measures", ""])
        for r in data.physical_risks:
            if r.adaptation_measures:
                lines.append(f"**{r.asset} ({r.risk_type.value}):**")
                for measure in r.adaptation_measures:
                    lines.append(f"- {measure}")
                lines.append("")
        return "\n".join(lines)

    def _md_transition_risks(self, data: ScenarioAnalysisReportInput) -> str:
        """Transition risk matrix."""
        if not data.transition_risks:
            return "## 3. Transition Risk Assessment\n\nNo transition risk data available."
        lines = [
            "## 3. Transition Risk Assessment",
            "",
            "| Risk | Driver | Likelihood | Magnitude | Horizon | Mitigation | Impact (EUR) |",
            "|------|--------|-----------|-----------|---------|------------|-------------|",
        ]
        sorted_risks = sorted(
            data.transition_risks,
            key=lambda x: _likelihood_sort(x.likelihood),
        )
        for r in sorted_risks:
            impact = _fmt_eur(r.financial_impact_eur)
            mitigation = r.mitigation or "-"
            lines.append(
                f"| {r.risk_type} | {r.driver.value} "
                f"| {r.likelihood.value} | {r.magnitude.value} "
                f"| {r.time_horizon.value} | {mitigation} | {impact} |"
            )
        return "\n".join(lines)

    def _md_financial_impacts(self, data: ScenarioAnalysisReportInput) -> str:
        """Financial impact comparison table."""
        if not data.financial_impacts:
            return "## 4. Financial Impact Comparison\n\nNo financial impact data."
        lines = [
            "## 4. Financial Impact Comparison",
            "",
            "| Scenario | Metric | Baseline | Impact | Change |",
            "|----------|--------|----------|--------|--------|",
        ]
        for f in data.financial_impacts:
            lines.append(
                f"| {f.scenario} | {f.metric} "
                f"| {_fmt_number(f.baseline_value, 1, f' {f.unit}')} "
                f"| {_fmt_number(f.impact_value, 1, f' {f.unit}')} "
                f"| {_fmt_pct(f.impact_pct)} |"
            )
        return "\n".join(lines)

    def _md_macc_curves(self, data: ScenarioAnalysisReportInput) -> str:
        """MACC curve data table."""
        if not data.macc_curves:
            return "## 5. Marginal Abatement Cost Curve\n\nNo MACC data available."
        lines = [
            "## 5. Marginal Abatement Cost Curve",
            "",
            "| Abatement Option | Cost/tCO2e (EUR) | Potential (tCO2e) | Year | Category |",
            "|-----------------|------------------|-------------------|------|----------|",
        ]
        for m in sorted(data.macc_curves, key=lambda x: x.cost_per_tco2):
            year = str(m.implementation_year) if m.implementation_year else "TBD"
            cat = m.category or "-"
            lines.append(
                f"| {m.abatement_option} | {_fmt_number(m.cost_per_tco2, 0, ' EUR')} "
                f"| {_fmt_number(m.potential_tco2, 1)} | {year} | {cat} |"
            )
        total_potential = sum(m.potential_tco2 for m in data.macc_curves)
        lines.append(
            f"| **Total Abatement Potential** | - | **{_fmt_number(total_potential, 1)}** | - | - |"
        )
        return "\n".join(lines)

    def _md_resilience(self, data: ScenarioAnalysisReportInput) -> str:
        """Resilience assessment narrative."""
        if not data.resilience_assessment:
            return "## 6. Resilience Assessment\n\nNo resilience data available."
        ra = data.resilience_assessment
        lines = [
            "## 6. Resilience Assessment",
            "",
            f"**Overall Level:** {ra.overall_level.value} | "
            f"**Score:** {ra.overall_score:.0f}/100",
            "",
        ]
        if ra.strengths:
            lines.append("### Strengths")
            for s in ra.strengths:
                lines.append(f"- {s}")
            lines.append("")
        if ra.vulnerabilities:
            lines.append("### Vulnerabilities")
            for v in ra.vulnerabilities:
                lines.append(f"- {v}")
            lines.append("")
        if ra.recommended_actions:
            lines.append("### Recommended Actions")
            for a in ra.recommended_actions:
                lines.append(f"- {a}")
        return "\n".join(lines)

    def _md_recommendations(self, data: ScenarioAnalysisReportInput) -> str:
        """Auto-generated recommendations."""
        lines = ["## 7. Recommendations", ""]
        recs = []
        high_physical = [
            r for r in data.physical_risks if r.overall_score >= 7.0
        ]
        if high_physical:
            recs.append(
                f"Prioritize adaptation for {len(high_physical)} high-risk "
                f"asset(s) with physical risk scores above 7.0."
            )
        critical_transition = [
            r for r in data.transition_risks
            if r.likelihood in (RiskLikelihood.VERY_HIGH, RiskLikelihood.HIGH)
            and r.magnitude in (RiskMagnitude.SEVERE, RiskMagnitude.HIGH)
        ]
        if critical_transition:
            recs.append(
                f"Develop mitigation plans for {len(critical_transition)} "
                f"high-likelihood, high-impact transition risk(s)."
            )
        if data.macc_curves:
            negative_cost = [m for m in data.macc_curves if m.cost_per_tco2 < 0]
            if negative_cost:
                recs.append(
                    f"Implement {len(negative_cost)} negative-cost abatement "
                    f"option(s) for immediate financial and emissions benefits."
                )
        if data.resilience_assessment:
            if data.resilience_assessment.overall_level in (
                ResilienceLevel.LOW, ResilienceLevel.CRITICAL
            ):
                recs.append(
                    "Strengthen organizational resilience urgently - "
                    "current level is below acceptable threshold."
                )
        if not recs:
            recs.append(
                "Continue regular scenario analysis updates and monitor "
                "emerging climate risks."
            )
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: ScenarioAnalysisReportInput) -> str:
        """Markdown footer."""
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
        """HTML wrapper."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scenario Analysis - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "h3{color:#533483;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".risk-high{background:#fecaca;}\n"
            ".risk-moderate{background:#fed7aa;}\n"
            ".risk-low{background:#d1fae5;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".strength{color:#1a7f37;}\n"
            ".vulnerability{color:#cf222e;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML header."""
        return (
            '<div class="section">\n'
            f"<h1>Climate Scenario Analysis &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()} | "
            f"<strong>Scenarios:</strong> {len(data.scenarios_analyzed)}</p>\n"
            "<hr>\n</div>"
        )

    def _html_scenario_overview(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML scenario overview."""
        rows = []
        for s in data.scenarios_analyzed:
            desc = s.description or "-"
            rows.append(
                f"<tr><td>{s.name}</td><td>{s.scenario_type.value}</td>"
                f"<td>{s.temperature_target:.1f} C</td>"
                f"<td>{s.time_horizon}</td><td>{s.source}</td>"
                f"<td>{desc}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No scenarios analyzed</td></tr>')
        return (
            '<div class="section">\n<h2>1. Scenario Overview</h2>\n'
            "<table><thead><tr><th>Scenario</th><th>Type</th><th>Temp</th>"
            "<th>Horizon</th><th>Source</th><th>Description</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_physical_risks(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML physical risk heatmap."""
        if not data.physical_risks:
            return (
                '<div class="section"><h2>2. Physical Risks</h2>'
                "<p>No data available.</p></div>"
            )
        rows = []
        for r in sorted(data.physical_risks, key=lambda x: x.overall_score, reverse=True):
            css = ""
            if r.overall_score >= 7.0:
                css = ' class="risk-high"'
            elif r.overall_score >= 4.0:
                css = ' class="risk-moderate"'
            else:
                css = ' class="risk-low"'
            risk_type = r.risk_type.value.replace("_", " ").title()
            impact = _fmt_eur(r.financial_impact_eur)
            rows.append(
                f"<tr{css}><td>{r.asset}</td><td>{risk_type}</td>"
                f"<td>{r.exposure_score:.1f}/10</td>"
                f"<td>{r.vulnerability_score:.1f}/10</td>"
                f"<td>{r.overall_score:.1f}/10</td>"
                f"<td>{impact}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>2. Physical Risk Assessment</h2>\n'
            "<table><thead><tr><th>Asset</th><th>Risk Type</th>"
            "<th>Exposure</th><th>Vulnerability</th><th>Overall</th>"
            f"<th>Impact</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_transition_risks(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML transition risk matrix."""
        if not data.transition_risks:
            return (
                '<div class="section"><h2>3. Transition Risks</h2>'
                "<p>No data available.</p></div>"
            )
        rows = []
        for r in sorted(data.transition_risks, key=lambda x: _likelihood_sort(x.likelihood)):
            impact = _fmt_eur(r.financial_impact_eur)
            mitigation = r.mitigation or "-"
            rows.append(
                f"<tr><td>{r.risk_type}</td><td>{r.driver.value}</td>"
                f"<td>{r.likelihood.value}</td><td>{r.magnitude.value}</td>"
                f"<td>{r.time_horizon.value}</td><td>{mitigation}</td>"
                f"<td>{impact}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>3. Transition Risk Assessment</h2>\n'
            "<table><thead><tr><th>Risk</th><th>Driver</th>"
            "<th>Likelihood</th><th>Magnitude</th><th>Horizon</th>"
            "<th>Mitigation</th><th>Impact</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_financial_impacts(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML financial impact comparison."""
        if not data.financial_impacts:
            return (
                '<div class="section"><h2>4. Financial Impacts</h2>'
                "<p>No data available.</p></div>"
            )
        rows = []
        for f in data.financial_impacts:
            rows.append(
                f"<tr><td>{f.scenario}</td><td>{f.metric}</td>"
                f"<td>{_fmt_number(f.baseline_value, 1, f' {f.unit}')}</td>"
                f"<td>{_fmt_number(f.impact_value, 1, f' {f.unit}')}</td>"
                f"<td>{_fmt_pct(f.impact_pct)}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>4. Financial Impact Comparison</h2>\n'
            "<table><thead><tr><th>Scenario</th><th>Metric</th>"
            "<th>Baseline</th><th>Impact</th><th>Change</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_macc_curves(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML MACC curve data."""
        if not data.macc_curves:
            return (
                '<div class="section"><h2>5. MACC Curve</h2>'
                "<p>No data available.</p></div>"
            )
        rows = []
        for m in sorted(data.macc_curves, key=lambda x: x.cost_per_tco2):
            year = str(m.implementation_year) if m.implementation_year else "TBD"
            cat = m.category or "-"
            css = ' style="background:#d1fae5"' if m.cost_per_tco2 < 0 else ""
            rows.append(
                f"<tr{css}><td>{m.abatement_option}</td>"
                f"<td>{_fmt_number(m.cost_per_tco2, 0, ' EUR')}</td>"
                f"<td>{_fmt_number(m.potential_tco2, 1)}</td>"
                f"<td>{year}</td><td>{cat}</td></tr>"
            )
        total = sum(m.potential_tco2 for m in data.macc_curves)
        rows.append(
            f"<tr style='font-weight:bold'><td>Total Potential</td><td>-</td>"
            f"<td>{_fmt_number(total, 1)}</td><td>-</td><td>-</td></tr>"
        )
        return (
            '<div class="section">\n<h2>5. Marginal Abatement Cost Curve</h2>\n'
            "<table><thead><tr><th>Option</th><th>Cost/tCO2e</th>"
            "<th>Potential</th><th>Year</th><th>Category</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_resilience(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML resilience assessment."""
        if not data.resilience_assessment:
            return (
                '<div class="section"><h2>6. Resilience</h2>'
                "<p>No data available.</p></div>"
            )
        ra = data.resilience_assessment
        parts = [
            '<div class="section">\n<h2>6. Resilience Assessment</h2>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f"{ra.overall_level.value}</div>"
            f'<div class="metric-label">Resilience Level</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f"{ra.overall_score:.0f}/100</div>"
            f'<div class="metric-label">Score</div></div>\n',
        ]
        if ra.strengths:
            items = "".join(
                f'<li class="strength">{s}</li>' for s in ra.strengths
            )
            parts.append(f"<h3>Strengths</h3><ul>{items}</ul>\n")
        if ra.vulnerabilities:
            items = "".join(
                f'<li class="vulnerability">{v}</li>' for v in ra.vulnerabilities
            )
            parts.append(f"<h3>Vulnerabilities</h3><ul>{items}</ul>\n")
        if ra.recommended_actions:
            items = "".join(f"<li>{a}</li>" for a in ra.recommended_actions)
            parts.append(f"<h3>Recommended Actions</h3><ol>{items}</ol>\n")
        parts.append("</div>")
        return "".join(parts)

    def _html_recommendations(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML recommendations."""
        recs = []
        high_physical = [r for r in data.physical_risks if r.overall_score >= 7.0]
        if high_physical:
            recs.append(
                f"Prioritize adaptation for {len(high_physical)} high-risk asset(s)."
            )
        critical_transition = [
            r for r in data.transition_risks
            if r.likelihood in (RiskLikelihood.VERY_HIGH, RiskLikelihood.HIGH)
            and r.magnitude in (RiskMagnitude.SEVERE, RiskMagnitude.HIGH)
        ]
        if critical_transition:
            recs.append(
                f"Develop mitigation for {len(critical_transition)} critical transition risk(s)."
            )
        if data.macc_curves:
            neg = [m for m in data.macc_curves if m.cost_per_tco2 < 0]
            if neg:
                recs.append(f"Implement {len(neg)} negative-cost abatement option(s).")
        if not recs:
            recs.append("Continue monitoring and updating scenario analysis.")
        items = "".join(f"<li>{r}</li>" for r in recs)
        return (
            '<div class="section">\n<h2>7. Recommendations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: ScenarioAnalysisReportInput) -> str:
        """HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
