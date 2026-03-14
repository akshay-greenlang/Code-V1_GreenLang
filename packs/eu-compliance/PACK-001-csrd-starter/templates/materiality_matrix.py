# -*- coding: utf-8 -*-
"""
PACK-001 Phase 3: Materiality Matrix Template
==============================================

Generates a materiality assessment report aligned with ESRS 1 double
materiality requirements. Produces methodology description, impact and
financial materiality results, 2D scatter-plot data, material topic
list, stakeholder engagement summary, and year-over-year comparison.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MaterialityOutcome(str, Enum):
    """Double materiality classification outcome."""
    MATERIAL = "MATERIAL"
    NOT_MATERIAL = "NOT_MATERIAL"
    UNDER_REVIEW = "UNDER_REVIEW"


class StakeholderGroup(str, Enum):
    """Stakeholder groups for engagement tracking."""
    EMPLOYEES = "EMPLOYEES"
    INVESTORS = "INVESTORS"
    CUSTOMERS = "CUSTOMERS"
    SUPPLIERS = "SUPPLIERS"
    COMMUNITIES = "COMMUNITIES"
    REGULATORS = "REGULATORS"
    NGOS = "NGOS"
    OTHER = "OTHER"


class EngagementMethod(str, Enum):
    """Methods of stakeholder engagement."""
    SURVEY = "SURVEY"
    INTERVIEW = "INTERVIEW"
    WORKSHOP = "WORKSHOP"
    FOCUS_GROUP = "FOCUS_GROUP"
    PUBLIC_CONSULTATION = "PUBLIC_CONSULTATION"
    ADVISORY_PANEL = "ADVISORY_PANEL"
    OTHER = "OTHER"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ImpactMaterialityScores(BaseModel):
    """Impact materiality dimension scores per ESRS 1 Chapter 3."""
    severity: float = Field(..., ge=0.0, le=10.0, description="Severity of impact (0-10)")
    scope: float = Field(..., ge=0.0, le=10.0, description="Scope/scale of impact (0-10)")
    irremediability: float = Field(..., ge=0.0, le=10.0, description="Irremediability (0-10)")

    @property
    def composite_score(self) -> float:
        """Weighted composite: severity 50%, scope 30%, irremediability 20%."""
        return self.severity * 0.5 + self.scope * 0.3 + self.irremediability * 0.2


class FinancialMaterialityScores(BaseModel):
    """Financial materiality dimension scores per ESRS 1 Chapter 3."""
    magnitude: float = Field(..., ge=0.0, le=10.0, description="Financial magnitude (0-10)")
    likelihood: float = Field(..., ge=0.0, le=10.0, description="Likelihood of occurrence (0-10)")

    @property
    def composite_score(self) -> float:
        """Composite: magnitude 60%, likelihood 40%."""
        return self.magnitude * 0.6 + self.likelihood * 0.4


class MaterialTopic(BaseModel):
    """A single material topic with full scoring."""
    topic_id: str = Field(..., description="Unique topic identifier")
    topic_name: str = Field(..., description="Topic name")
    esrs_standard: Optional[str] = Field(None, description="Related ESRS standard (e.g. E1, S1)")
    description: str = Field("", description="Topic description")
    impact_scores: ImpactMaterialityScores = Field(..., description="Impact materiality scores")
    financial_scores: FinancialMaterialityScores = Field(..., description="Financial materiality scores")
    outcome: MaterialityOutcome = Field(MaterialityOutcome.UNDER_REVIEW, description="Materiality outcome")
    threshold_impact: float = Field(5.0, ge=0.0, le=10.0, description="Impact materiality threshold")
    threshold_financial: float = Field(5.0, ge=0.0, le=10.0, description="Financial materiality threshold")
    prior_year_impact: Optional[float] = Field(None, ge=0.0, le=10.0, description="Prior year impact score")
    prior_year_financial: Optional[float] = Field(None, ge=0.0, le=10.0, description="Prior year financial score")
    rationale: Optional[str] = Field(None, description="Rationale for materiality determination")

    @property
    def is_impact_material(self) -> bool:
        """Whether topic exceeds the impact materiality threshold."""
        return self.impact_scores.composite_score >= self.threshold_impact

    @property
    def is_financial_material(self) -> bool:
        """Whether topic exceeds the financial materiality threshold."""
        return self.financial_scores.composite_score >= self.threshold_financial

    @property
    def is_double_material(self) -> bool:
        """Material under either or both dimensions (ESRS 1 double materiality)."""
        return self.is_impact_material or self.is_financial_material


class MatrixDataPoint(BaseModel):
    """A single point in the 2D materiality scatter plot."""
    topic_id: str = Field(..., description="Topic identifier")
    topic_name: str = Field(..., description="Label for the data point")
    x: float = Field(..., ge=0.0, le=10.0, description="X-axis value (financial materiality)")
    y: float = Field(..., ge=0.0, le=10.0, description="Y-axis value (impact materiality)")
    is_material: bool = Field(False, description="Whether the point is material")
    quadrant: str = Field("", description="Quadrant label")

    def model_post_init(self, __context: Any) -> None:
        """Auto-derive quadrant from coordinates."""
        if not self.quadrant:
            if self.x >= 5.0 and self.y >= 5.0:
                self.quadrant = "Double Material"
            elif self.x >= 5.0:
                self.quadrant = "Financial Only"
            elif self.y >= 5.0:
                self.quadrant = "Impact Only"
            else:
                self.quadrant = "Not Material"


class StakeholderEngagement(BaseModel):
    """Stakeholder engagement record for the materiality process."""
    stakeholder_group: StakeholderGroup = Field(..., description="Stakeholder group")
    method: EngagementMethod = Field(..., description="Engagement method")
    participant_count: int = Field(0, ge=0, description="Number of participants")
    engagement_date: Optional[date] = Field(None, description="Date of engagement")
    key_topics_raised: List[str] = Field(default_factory=list, description="Topics raised")
    summary: Optional[str] = Field(None, description="Engagement summary")


class MaterialityMatrixInput(BaseModel):
    """Full input for the materiality matrix report."""
    company_name: str = Field(..., description="Reporting entity")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Fiscal year")
    report_date: date = Field(default_factory=date.today, description="Generation date")
    methodology_description: str = Field(
        "Double materiality assessment performed in accordance with ESRS 1 Chapter 3, "
        "applying both impact materiality (severity, scope, irremediability) and financial "
        "materiality (magnitude, likelihood) criteria.",
        description="Methodology narrative",
    )
    impact_threshold: float = Field(5.0, ge=0.0, le=10.0, description="Impact threshold")
    financial_threshold: float = Field(5.0, ge=0.0, le=10.0, description="Financial threshold")
    topics: List[MaterialTopic] = Field(default_factory=list, description="Assessed topics")
    stakeholder_engagements: List[StakeholderEngagement] = Field(
        default_factory=list, description="Stakeholder engagement records"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _score_bar(value: float, max_val: float = 10.0, width: int = 10) -> str:
    """Render a simple text bar chart segment."""
    filled = int(round(value / max_val * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.1f}/{max_val:.0f}"


def _outcome_badge(outcome: MaterialityOutcome) -> str:
    """Text badge for materiality outcome."""
    return {
        MaterialityOutcome.MATERIAL: "[MATERIAL]",
        MaterialityOutcome.NOT_MATERIAL: "[NOT MATERIAL]",
        MaterialityOutcome.UNDER_REVIEW: "[UNDER REVIEW]",
    }.get(outcome, "[?]")


def _outcome_css(outcome: MaterialityOutcome) -> str:
    """CSS class for materiality outcome."""
    return f"outcome-{outcome.value.lower().replace('_', '-')}"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class MaterialityMatrixTemplate:
    """Generate materiality assessment report with visual matrix.

    Components:
        1. Methodology description (ESRS 1 double materiality)
        2. Impact materiality results (severity x scope x irremediability)
        3. Financial materiality results (magnitude x likelihood)
        4. 2D scatter plot data for materiality matrix
        5. Material topic list with scores and thresholds
        6. Stakeholder engagement summary
        7. Year-over-year comparison (if available)

    Example:
        >>> template = MaterialityMatrixTemplate()
        >>> data = MaterialityMatrixInput(company_name="Acme", reporting_year=2025, ...)
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> chart_data = template.get_matrix_data(data)
    """

    def __init__(self) -> None:
        """Initialize the materiality matrix template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def get_matrix_data(self, data: MaterialityMatrixInput) -> List[MatrixDataPoint]:
        """Generate scatter plot data points from assessed topics.

        Args:
            data: Materiality matrix input.

        Returns:
            List of MatrixDataPoint instances ready for visualization.
        """
        points = []
        for t in data.topics:
            points.append(MatrixDataPoint(
                topic_id=t.topic_id,
                topic_name=t.topic_name,
                x=t.financial_scores.composite_score,
                y=t.impact_scores.composite_score,
                is_material=t.is_double_material,
            ))
        return points

    def render_markdown(self, data: MaterialityMatrixInput) -> str:
        """Render the full materiality report as Markdown.

        Args:
            data: Validated input.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_topic_results(data),
            self._md_matrix_text(data),
            self._md_stakeholder_engagement(data),
            self._md_yoy_comparison(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: MaterialityMatrixInput) -> str:
        """Render the report as HTML.

        Args:
            data: Validated input.

        Returns:
            HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_methodology(data),
            self._html_topic_results(data),
            self._html_matrix_text(data),
            self._html_stakeholder_engagement(data),
            self._html_yoy_comparison(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.company_name, data.reporting_year, body)

    def render_json(self, data: MaterialityMatrixInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated input.

        Returns:
            Dictionary for serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        matrix_points = self.get_matrix_data(data)
        material_topics = [t for t in data.topics if t.is_double_material]

        return {
            "template": "materiality_matrix",
            "version": "1.0.0",
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "company_name": data.company_name,
            "reporting_year": data.reporting_year,
            "methodology": data.methodology_description,
            "thresholds": {
                "impact": data.impact_threshold,
                "financial": data.financial_threshold,
            },
            "topics_assessed": len(data.topics),
            "topics_material": len(material_topics),
            "topics": [t.model_dump(mode="json") for t in data.topics],
            "matrix_data": [p.model_dump(mode="json") for p in matrix_points],
            "stakeholder_engagements": [
                e.model_dump(mode="json") for e in data.stakeholder_engagements
            ],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: MaterialityMatrixInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: MaterialityMatrixInput) -> str:
        """Markdown header."""
        material_count = sum(1 for t in data.topics if t.is_double_material)
        return (
            f"# Materiality Assessment Report - {data.company_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Topics Assessed:** {len(data.topics)} | "
            f"**Material:** {material_count}\n\n---"
        )

    def _md_methodology(self, data: MaterialityMatrixInput) -> str:
        """Methodology section."""
        return (
            "## 1. Methodology\n\n"
            f"{data.methodology_description}\n\n"
            f"**Impact Materiality Threshold:** {data.impact_threshold:.1f}/10 "
            f"(composite of severity 50%, scope 30%, irremediability 20%)\n"
            f"**Financial Materiality Threshold:** {data.financial_threshold:.1f}/10 "
            f"(composite of magnitude 60%, likelihood 40%)\n\n"
            "A topic is considered material if it exceeds the threshold on **either** "
            "the impact or financial dimension (ESRS 1 double materiality principle)."
        )

    def _md_topic_results(self, data: MaterialityMatrixInput) -> str:
        """Topic results table."""
        lines = [
            "## 2. Assessment Results",
            "",
            "### Impact Materiality",
            "",
            "| Topic | Severity | Scope | Irremediability | Composite | Threshold | Result |",
            "|-------|----------|-------|----------------|-----------|-----------|--------|",
        ]
        for t in sorted(data.topics, key=lambda x: x.impact_scores.composite_score, reverse=True):
            result = "MATERIAL" if t.is_impact_material else "Below"
            lines.append(
                f"| {t.topic_name} | {t.impact_scores.severity:.1f} | "
                f"{t.impact_scores.scope:.1f} | {t.impact_scores.irremediability:.1f} | "
                f"**{t.impact_scores.composite_score:.1f}** | {t.threshold_impact:.1f} | {result} |"
            )
        lines.extend([
            "",
            "### Financial Materiality",
            "",
            "| Topic | Magnitude | Likelihood | Composite | Threshold | Result |",
            "|-------|-----------|-----------|-----------|-----------|--------|",
        ])
        for t in sorted(data.topics, key=lambda x: x.financial_scores.composite_score, reverse=True):
            result = "MATERIAL" if t.is_financial_material else "Below"
            lines.append(
                f"| {t.topic_name} | {t.financial_scores.magnitude:.1f} | "
                f"{t.financial_scores.likelihood:.1f} | "
                f"**{t.financial_scores.composite_score:.1f}** | "
                f"{t.threshold_financial:.1f} | {result} |"
            )
        lines.extend([
            "",
            "### Combined Outcome",
            "",
            "| Topic | ESRS | Impact | Financial | Outcome | Rationale |",
            "|-------|------|--------|-----------|---------|-----------|",
        ])
        for t in data.topics:
            esrs = t.esrs_standard or "-"
            rationale = t.rationale or "-"
            lines.append(
                f"| {t.topic_name} | {esrs} | {t.impact_scores.composite_score:.1f} | "
                f"{t.financial_scores.composite_score:.1f} | "
                f"{_outcome_badge(t.outcome)} | {rationale} |"
            )
        return "\n".join(lines)

    def _md_matrix_text(self, data: MaterialityMatrixInput) -> str:
        """Text representation of the 2D matrix for Markdown."""
        points = self.get_matrix_data(data)
        if not points:
            return "## 3. Materiality Matrix\n\nNo data points available."
        lines = [
            "## 3. Materiality Matrix",
            "",
            "| Topic | Financial (X) | Impact (Y) | Quadrant | Material |",
            "|-------|--------------|-----------|----------|----------|",
        ]
        for p in sorted(points, key=lambda pt: pt.x * pt.y, reverse=True):
            mat = "Yes" if p.is_material else "No"
            lines.append(
                f"| {p.topic_name} | {p.x:.1f} | {p.y:.1f} | {p.quadrant} | {mat} |"
            )
        return "\n".join(lines)

    def _md_stakeholder_engagement(self, data: MaterialityMatrixInput) -> str:
        """Stakeholder engagement summary."""
        if not data.stakeholder_engagements:
            return "## 4. Stakeholder Engagement\n\nNo stakeholder engagement data recorded."
        total_participants = sum(e.participant_count for e in data.stakeholder_engagements)
        lines = [
            "## 4. Stakeholder Engagement",
            "",
            f"**Total Engagements:** {len(data.stakeholder_engagements)} | "
            f"**Total Participants:** {total_participants}",
            "",
            "| Group | Method | Participants | Date | Key Topics |",
            "|-------|--------|-------------|------|------------|",
        ]
        for e in data.stakeholder_engagements:
            eng_date = e.engagement_date.isoformat() if e.engagement_date else "N/A"
            topics = ", ".join(e.key_topics_raised) if e.key_topics_raised else "-"
            lines.append(
                f"| {e.stakeholder_group.value} | {e.method.value} | "
                f"{e.participant_count} | {eng_date} | {topics} |"
            )
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: MaterialityMatrixInput) -> str:
        """Year-over-year comparison if prior year data is available."""
        has_prior = any(
            t.prior_year_impact is not None or t.prior_year_financial is not None
            for t in data.topics
        )
        if not has_prior:
            return ""
        lines = [
            "## 5. Year-over-Year Comparison",
            "",
            "| Topic | Impact (Current) | Impact (Prior) | Change | "
            "Financial (Current) | Financial (Prior) | Change |",
            "|-------|-----------------|---------------|--------|"
            "--------------------|------------------|--------|",
        ]
        for t in data.topics:
            impact_curr = t.impact_scores.composite_score
            impact_prior = t.prior_year_impact
            fin_curr = t.financial_scores.composite_score
            fin_prior = t.prior_year_financial
            if impact_prior is not None:
                impact_change = f"{impact_curr - impact_prior:+.1f}"
            else:
                impact_change = "N/A"
                impact_prior = "N/A"
            if fin_prior is not None:
                fin_change = f"{fin_curr - fin_prior:+.1f}"
            else:
                fin_change = "N/A"
                fin_prior = "N/A"
            lines.append(
                f"| {t.topic_name} | {impact_curr:.1f} | {impact_prior} | {impact_change} | "
                f"{fin_curr:.1f} | {fin_prior} | {fin_change} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: MaterialityMatrixInput) -> str:
        """Markdown footer with provenance."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, company: str, year: int, body: str) -> str:
        """Wrap body in HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Materiality Assessment - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#222;max-width:1000px;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f5f5f5;}\n"
            ".outcome-material{color:#1a7f37;font-weight:bold;}\n"
            ".outcome-not-material{color:#888;}\n"
            ".outcome-under-review{color:#b08800;font-weight:bold;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: MaterialityMatrixInput) -> str:
        """HTML header."""
        material_count = sum(1 for t in data.topics if t.is_double_material)
        return (
            '<div class="section">\n'
            f"<h1>Materiality Assessment Report &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()} | "
            f"<strong>Topics Assessed:</strong> {len(data.topics)} | "
            f"<strong>Material:</strong> {material_count}</p>\n<hr>\n</div>"
        )

    def _html_methodology(self, data: MaterialityMatrixInput) -> str:
        """HTML methodology section."""
        return (
            '<div class="section">\n<h2>1. Methodology</h2>\n'
            f"<p>{data.methodology_description}</p>\n"
            f"<p><strong>Impact Threshold:</strong> {data.impact_threshold:.1f}/10 | "
            f"<strong>Financial Threshold:</strong> {data.financial_threshold:.1f}/10</p>\n"
            "<p>A topic is considered material if it exceeds the threshold on "
            "<strong>either</strong> dimension (ESRS 1 double materiality).</p>\n</div>"
        )

    def _html_topic_results(self, data: MaterialityMatrixInput) -> str:
        """HTML combined results table."""
        rows = []
        for t in data.topics:
            esrs = t.esrs_standard or "-"
            css = _outcome_css(t.outcome)
            rationale = t.rationale or "-"
            rows.append(
                f"<tr><td>{t.topic_name}</td><td>{esrs}</td>"
                f"<td>{t.impact_scores.composite_score:.1f}</td>"
                f"<td>{t.financial_scores.composite_score:.1f}</td>"
                f'<td class="{css}">{t.outcome.value}</td>'
                f"<td>{rationale}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>2. Assessment Results</h2>\n'
            "<table><thead><tr><th>Topic</th><th>ESRS</th><th>Impact</th>"
            "<th>Financial</th><th>Outcome</th><th>Rationale</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_matrix_text(self, data: MaterialityMatrixInput) -> str:
        """HTML matrix table."""
        points = self.get_matrix_data(data)
        if not points:
            return '<div class="section"><h2>3. Materiality Matrix</h2><p>No data.</p></div>'
        rows = []
        for p in sorted(points, key=lambda pt: pt.x * pt.y, reverse=True):
            mat = "Yes" if p.is_material else "No"
            rows.append(
                f"<tr><td>{p.topic_name}</td><td>{p.x:.1f}</td><td>{p.y:.1f}</td>"
                f"<td>{p.quadrant}</td><td>{mat}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>3. Materiality Matrix</h2>\n'
            "<table><thead><tr><th>Topic</th><th>Financial (X)</th>"
            "<th>Impact (Y)</th><th>Quadrant</th><th>Material</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_stakeholder_engagement(self, data: MaterialityMatrixInput) -> str:
        """HTML stakeholder engagement table."""
        if not data.stakeholder_engagements:
            return (
                '<div class="section"><h2>4. Stakeholder Engagement</h2>'
                "<p>No engagement data recorded.</p></div>"
            )
        total = sum(e.participant_count for e in data.stakeholder_engagements)
        rows = []
        for e in data.stakeholder_engagements:
            eng_date = e.engagement_date.isoformat() if e.engagement_date else "N/A"
            topics = ", ".join(e.key_topics_raised) if e.key_topics_raised else "-"
            rows.append(
                f"<tr><td>{e.stakeholder_group.value}</td><td>{e.method.value}</td>"
                f"<td>{e.participant_count}</td><td>{eng_date}</td><td>{topics}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>4. Stakeholder Engagement</h2>\n'
            f"<p><strong>Total Engagements:</strong> {len(data.stakeholder_engagements)} | "
            f"<strong>Total Participants:</strong> {total}</p>\n"
            "<table><thead><tr><th>Group</th><th>Method</th><th>Participants</th>"
            "<th>Date</th><th>Key Topics</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_yoy_comparison(self, data: MaterialityMatrixInput) -> str:
        """HTML year-over-year comparison."""
        has_prior = any(
            t.prior_year_impact is not None or t.prior_year_financial is not None
            for t in data.topics
        )
        if not has_prior:
            return ""
        rows = []
        for t in data.topics:
            ic = t.impact_scores.composite_score
            ip = t.prior_year_impact
            fc = t.financial_scores.composite_score
            fp = t.prior_year_financial
            ic_str = f"{ic:.1f}"
            fc_str = f"{fc:.1f}"
            ip_str = f"{ip:.1f}" if ip is not None else "N/A"
            fp_str = f"{fp:.1f}" if fp is not None else "N/A"
            ic_chg = f"{ic - ip:+.1f}" if ip is not None else "N/A"
            fc_chg = f"{fc - fp:+.1f}" if fp is not None else "N/A"
            rows.append(
                f"<tr><td>{t.topic_name}</td>"
                f"<td>{ic_str}</td><td>{ip_str}</td><td>{ic_chg}</td>"
                f"<td>{fc_str}</td><td>{fp_str}</td><td>{fc_chg}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>5. Year-over-Year Comparison</h2>\n'
            "<table><thead><tr><th>Topic</th><th>Impact (Curr)</th>"
            "<th>Impact (Prior)</th><th>Change</th>"
            "<th>Financial (Curr)</th><th>Financial (Prior)</th>"
            "<th>Change</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: MaterialityMatrixInput) -> str:
        """HTML footer with provenance."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
