"""
TaxonomyAlignmentReportTemplate - EU Taxonomy alignment report.

This module implements the taxonomy alignment report template for PACK-010
SFDR Article 8 products. It provides detailed analysis of portfolio
alignment with the EU Taxonomy Regulation, including objective breakdowns,
eligible vs. aligned analysis, commitment adherence tracking, and
top aligned holdings.

Example:
    >>> template = TaxonomyAlignmentReportTemplate()
    >>> data = AlignmentReportData(alignment_ratio=45.0, ...)
    >>> markdown = template.render_markdown(data.model_dump())
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class AlignmentSummary(BaseModel):
    """Overall alignment summary."""

    total_eligible_pct: float = Field(0.0, ge=0.0, le=100.0, description="Total eligible %")
    total_aligned_pct: float = Field(0.0, ge=0.0, le=100.0, description="Total aligned %")
    conversion_rate: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Eligible-to-aligned conversion rate %"
    )
    previous_aligned_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Previous period aligned %"
    )
    as_of_date: str = Field("", description="Data as-of date")


class ObjectiveBreakdown(BaseModel):
    """Breakdown by EU Taxonomy environmental objective."""

    objective_name: str = Field("", description="Environmental objective name")
    objective_code: str = Field("", description="Objective code (CCM, CCA, WFR, CE, PP, BIO)")
    eligible_pct: float = Field(0.0, ge=0.0, le=100.0, description="Eligible %")
    aligned_pct: float = Field(0.0, ge=0.0, le=100.0, description="Aligned %")
    substantial_contribution: bool = Field(
        False, description="Substantial contribution criteria met"
    )
    dnsh_compliant: bool = Field(False, description="DNSH criteria met")
    minimum_safeguards: bool = Field(False, description="Minimum safeguards met")
    holdings_count: int = Field(0, ge=0, description="Number of holdings contributing")


class EligibleVsAligned(BaseModel):
    """Eligible vs aligned analysis entry."""

    sector: str = Field("", description="Sector/NACE code")
    eligible_revenue_pct: float = Field(0.0, ge=0.0, le=100.0)
    aligned_revenue_pct: float = Field(0.0, ge=0.0, le=100.0)
    eligible_capex_pct: float = Field(0.0, ge=0.0, le=100.0)
    aligned_capex_pct: float = Field(0.0, ge=0.0, le=100.0)
    eligible_opex_pct: float = Field(0.0, ge=0.0, le=100.0)
    aligned_opex_pct: float = Field(0.0, ge=0.0, le=100.0)
    gap_analysis: str = Field("", description="Gap analysis notes")


class CommitmentAdherence(BaseModel):
    """Commitment adherence tracking."""

    commitment_name: str = Field("", description="Commitment name")
    target_pct: float = Field(0.0, ge=0.0, le=100.0, description="Target %")
    actual_pct: float = Field(0.0, ge=0.0, le=100.0, description="Actual %")
    deviation_pct: Optional[float] = Field(None, description="Deviation from target")
    status: str = Field("on_track", description="on_track, at_risk, breached")
    corrective_action: str = Field("", description="Corrective action if needed")


class TopAlignedHolding(BaseModel):
    """Top taxonomy-aligned holding."""

    rank: int = Field(0, ge=1, description="Rank")
    name: str = Field("", description="Holding name")
    sector: str = Field("", description="Sector")
    weight_pct: float = Field(0.0, ge=0.0, le=100.0, description="Portfolio weight %")
    aligned_revenue_pct: float = Field(0.0, ge=0.0, le=100.0, description="Aligned revenue %")
    primary_objective: str = Field("", description="Primary objective contributing to")
    transitional: bool = Field(False, description="Transitional activity")
    enabling: bool = Field(False, description="Enabling activity")


class FossilGasNuclearDisclosure(BaseModel):
    """Mandatory fossil gas and nuclear disclosure."""

    fossil_gas_eligible_pct: float = Field(0.0, ge=0.0, le=100.0)
    fossil_gas_aligned_pct: float = Field(0.0, ge=0.0, le=100.0)
    nuclear_eligible_pct: float = Field(0.0, ge=0.0, le=100.0)
    nuclear_aligned_pct: float = Field(0.0, ge=0.0, le=100.0)
    disclosure_statement: str = Field("", description="Mandatory disclosure statement")


class AlignmentReportData(BaseModel):
    """Complete input data for taxonomy alignment report."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN")
    alignment_ratio: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall alignment ratio"
    )
    summary: AlignmentSummary = Field(default_factory=AlignmentSummary)
    by_objective: List[ObjectiveBreakdown] = Field(default_factory=list)
    eligible_vs_aligned: List[EligibleVsAligned] = Field(default_factory=list)
    commitment: List[CommitmentAdherence] = Field(default_factory=list)
    top_holdings: List[TopAlignedHolding] = Field(default_factory=list)
    fossil_gas_nuclear: FossilGasNuclearDisclosure = Field(
        default_factory=FossilGasNuclearDisclosure
    )
    transitional_pct: float = Field(0.0, ge=0.0, le=100.0)
    enabling_pct: float = Field(0.0, ge=0.0, le=100.0)
    methodology_notes: str = Field("", description="Methodology notes")


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class TaxonomyAlignmentReportTemplate:
    """
    EU Taxonomy alignment report template for SFDR Article 8 products.

    Generates detailed taxonomy alignment analysis with objective breakdowns,
    eligible vs. aligned comparisons, commitment adherence, and top holdings.

    Example:
        >>> template = TaxonomyAlignmentReportTemplate()
        >>> md = template.render_markdown(data)
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "taxonomy_alignment_report"
    VERSION = "1.0"

    OBJECTIVE_LABELS = {
        "CCM": "Climate Change Mitigation",
        "CCA": "Climate Change Adaptation",
        "WFR": "Water and Marine Resources",
        "CE": "Circular Economy",
        "PP": "Pollution Prevention",
        "BIO": "Biodiversity and Ecosystems",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TaxonomyAlignmentReportTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render taxonomy alignment report in the specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'.")

    # ------------------------------------------------------------------ #
    #  Markdown rendering
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render taxonomy alignment report as Markdown."""
        sections: List[str] = [
            self._md_header(data),
            self._md_alignment_summary(data),
            self._md_objective_breakdown(data),
            self._md_fossil_gas_nuclear(data),
            self._md_eligible_vs_aligned(data),
            self._md_commitment_adherence(data),
            self._md_top_holdings(data),
            self._md_methodology(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render taxonomy alignment report as HTML."""
        sections: List[str] = [
            self._html_alignment_summary(data),
            self._html_objective_breakdown(data),
            self._html_fossil_gas_nuclear(data),
            self._html_eligible_vs_aligned(data),
            self._html_commitment_adherence(data),
            self._html_top_holdings(data),
            self._html_methodology(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html("EU Taxonomy Alignment Report", body, provenance_hash)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render taxonomy alignment report as JSON."""
        report: Dict[str, Any] = {
            "report_type": "sfdr_taxonomy_alignment",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_name": data.get("fund_name", ""),
            "isin": data.get("isin", ""),
            "alignment_ratio": data.get("alignment_ratio", 0.0),
            "summary": data.get("summary", {}),
            "by_objective": data.get("by_objective", []),
            "eligible_vs_aligned": data.get("eligible_vs_aligned", []),
            "commitment": data.get("commitment", []),
            "top_holdings": data.get("top_holdings", []),
            "fossil_gas_nuclear": data.get("fossil_gas_nuclear", {}),
            "transitional_pct": data.get("transitional_pct", 0.0),
            "enabling_pct": data.get("enabling_pct", 0.0),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build header."""
        return (
            f"# EU Taxonomy Alignment Report\n\n"
            f"**Fund:** {data.get('fund_name', 'Unknown')}\n\n"
            f"**ISIN:** {data.get('isin', '') or 'N/A'}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_alignment_summary(self, data: Dict[str, Any]) -> str:
        """Build alignment summary."""
        s = data.get("summary", {})
        ratio = data.get("alignment_ratio", 0.0)
        eligible = s.get("total_eligible_pct", 0.0)
        aligned = s.get("total_aligned_pct", 0.0)
        prev = s.get("previous_aligned_pct")
        conversion = s.get("conversion_rate")

        lines = [
            "## Alignment Summary\n",
            "```",
            f"  Overall Alignment: {self._ascii_gauge(ratio)} {ratio:.1f}%",
            f"  Eligible:          {self._ascii_gauge(eligible)} {eligible:.1f}%",
            f"  Aligned:           {self._ascii_gauge(aligned)} {aligned:.1f}%",
        ]

        if conversion is not None:
            lines.append(f"  Conversion Rate:   {conversion:.1f}%")

        if prev is not None:
            diff = aligned - prev
            lines.append(f"  Previous Period:   {prev:.1f}%  (change: {diff:+.1f}pp)")

        lines.append(f"  Transitional:      {data.get('transitional_pct', 0.0):.1f}%")
        lines.append(f"  Enabling:          {data.get('enabling_pct', 0.0):.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_objective_breakdown(self, data: Dict[str, Any]) -> str:
        """Build objective breakdown table."""
        objectives = data.get("by_objective", [])

        lines = [
            "## Breakdown by Environmental Objective\n",
            "| Objective | Code | Eligible | Aligned | SC | DNSH | MS | Holdings |",
            "|-----------|------|----------|---------|----|----- |----|----------|",
        ]

        for obj in objectives:
            sc = "Yes" if obj.get("substantial_contribution") else "No"
            dnsh = "Yes" if obj.get("dnsh_compliant") else "No"
            ms = "Yes" if obj.get("minimum_safeguards") else "No"
            lines.append(
                f"| {obj.get('objective_name', '')} | "
                f"{obj.get('objective_code', '')} | "
                f"{obj.get('eligible_pct', 0.0):.1f}% | "
                f"{obj.get('aligned_pct', 0.0):.1f}% | "
                f"{sc} | {dnsh} | {ms} | "
                f"{obj.get('holdings_count', 0)} |"
            )

        if not objectives:
            lines.append("| *No objective data* | | | | | | | |")

        lines.append("\n*SC = Substantial Contribution, DNSH = Do No Significant Harm, "
                      "MS = Minimum Safeguards*")

        return "\n".join(lines)

    def _md_fossil_gas_nuclear(self, data: Dict[str, Any]) -> str:
        """Build fossil gas and nuclear disclosure."""
        fgn = data.get("fossil_gas_nuclear", {})

        lines = [
            "## Fossil Gas and Nuclear Disclosure\n",
            "| Activity | Eligible | Aligned |",
            "|----------|----------|---------|",
            f"| Fossil Gas | {fgn.get('fossil_gas_eligible_pct', 0.0):.1f}% | "
            f"{fgn.get('fossil_gas_aligned_pct', 0.0):.1f}% |",
            f"| Nuclear | {fgn.get('nuclear_eligible_pct', 0.0):.1f}% | "
            f"{fgn.get('nuclear_aligned_pct', 0.0):.1f}% |",
        ]

        statement = fgn.get("disclosure_statement", "")
        if statement:
            lines.append(f"\n{statement}")

        return "\n".join(lines)

    def _md_eligible_vs_aligned(self, data: Dict[str, Any]) -> str:
        """Build eligible vs aligned analysis."""
        eva = data.get("eligible_vs_aligned", [])
        if not eva:
            return ""

        lines = [
            "## Eligible vs. Aligned by Sector\n",
            "| Sector | Rev. Elig. | Rev. Aligned | CapEx Elig. | CapEx Aligned | "
            "OpEx Elig. | OpEx Aligned |",
            "|--------|-----------|-------------|------------|--------------|"
            "----------|-------------|",
        ]

        for e in eva:
            lines.append(
                f"| {e.get('sector', '')} | "
                f"{e.get('eligible_revenue_pct', 0.0):.1f}% | "
                f"{e.get('aligned_revenue_pct', 0.0):.1f}% | "
                f"{e.get('eligible_capex_pct', 0.0):.1f}% | "
                f"{e.get('aligned_capex_pct', 0.0):.1f}% | "
                f"{e.get('eligible_opex_pct', 0.0):.1f}% | "
                f"{e.get('aligned_opex_pct', 0.0):.1f}% |"
            )

        return "\n".join(lines)

    def _md_commitment_adherence(self, data: Dict[str, Any]) -> str:
        """Build commitment adherence section."""
        commitments = data.get("commitment", [])
        if not commitments:
            return ""

        lines = [
            "## Commitment Adherence\n",
            "| Commitment | Target | Actual | Deviation | Status |",
            "|------------|--------|--------|-----------|--------|",
        ]

        for c in commitments:
            status = c.get("status", "on_track")
            status_label = {"on_track": "ON TRACK", "at_risk": "AT RISK", "breached": "BREACHED"}.get(
                status, status.upper()
            )
            dev = c.get("deviation_pct")
            dev_str = f"{dev:+.1f}pp" if dev is not None else "N/A"
            lines.append(
                f"| {c.get('commitment_name', '')} | "
                f"{c.get('target_pct', 0.0):.1f}% | "
                f"{c.get('actual_pct', 0.0):.1f}% | "
                f"{dev_str} | {status_label} |"
            )

            corrective = c.get("corrective_action", "")
            if corrective and status != "on_track":
                lines.append(f"\n> *Corrective Action:* {corrective}\n")

        return "\n".join(lines)

    def _md_top_holdings(self, data: Dict[str, Any]) -> str:
        """Build top aligned holdings table."""
        holdings = data.get("top_holdings", [])
        if not holdings:
            return ""

        lines = [
            "## Top Taxonomy-Aligned Holdings\n",
            "| # | Name | Sector | Weight | Aligned Rev. | Objective | Type |",
            "|---|------|--------|--------|-------------|-----------|------|",
        ]

        for h in holdings:
            activity_type = (
                "Transitional" if h.get("transitional")
                else "Enabling" if h.get("enabling")
                else "Standard"
            )
            lines.append(
                f"| {h.get('rank', 0)} | {h.get('name', '')} | "
                f"{h.get('sector', '')} | {h.get('weight_pct', 0.0):.2f}% | "
                f"{h.get('aligned_revenue_pct', 0.0):.1f}% | "
                f"{h.get('primary_objective', '')} | {activity_type} |"
            )

        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Build methodology notes section."""
        notes = data.get("methodology_notes", "")
        if not notes:
            return ""
        return f"## Methodology Notes\n\n{notes}"

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_alignment_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML alignment summary."""
        s = data.get("summary", {})
        eligible = s.get("total_eligible_pct", 0.0)
        aligned = s.get("total_aligned_pct", 0.0)
        ratio = data.get("alignment_ratio", 0.0)

        parts = ['<div class="section"><h2>Alignment Summary</h2>']
        parts.append('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px;">')
        for label, pct, color in [
            ("Eligible", eligible, "#3498db"),
            ("Aligned", aligned, "#2ecc71"),
            ("Overall Ratio", ratio, "#27ae60"),
        ]:
            parts.append(
                f'<div style="text-align:center;padding:15px;background:#f8f9fa;border-radius:6px;">'
                f'<div style="font-size:0.85em;color:#7f8c8d;">{_esc(label)}</div>'
                f'<div style="font-size:2em;font-weight:bold;color:{color};">{pct:.1f}%</div>'
                f'<div style="background:#ecf0f1;height:8px;border-radius:4px;margin-top:8px;">'
                f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;"></div>'
                f"</div></div>"
            )
        parts.append("</div></div>")
        return "".join(parts)

    def _html_objective_breakdown(self, data: Dict[str, Any]) -> str:
        """Build HTML objective breakdown."""
        objectives = data.get("by_objective", [])
        parts = ['<div class="section"><h2>By Environmental Objective</h2>']

        if objectives:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Objective</th><th>Code</th><th>Eligible</th>"
                "<th>Aligned</th><th>SC</th><th>DNSH</th><th>MS</th></tr>"
            )
            for obj in objectives:
                sc_color = "#2ecc71" if obj.get("substantial_contribution") else "#e74c3c"
                dnsh_color = "#2ecc71" if obj.get("dnsh_compliant") else "#e74c3c"
                ms_color = "#2ecc71" if obj.get("minimum_safeguards") else "#e74c3c"
                parts.append(
                    f"<tr><td>{_esc(obj.get('objective_name', ''))}</td>"
                    f"<td>{_esc(obj.get('objective_code', ''))}</td>"
                    f"<td>{obj.get('eligible_pct', 0.0):.1f}%</td>"
                    f"<td>{obj.get('aligned_pct', 0.0):.1f}%</td>"
                    f'<td style="color:{sc_color};font-weight:bold;">{"Yes" if obj.get("substantial_contribution") else "No"}</td>'
                    f'<td style="color:{dnsh_color};font-weight:bold;">{"Yes" if obj.get("dnsh_compliant") else "No"}</td>'
                    f'<td style="color:{ms_color};font-weight:bold;">{"Yes" if obj.get("minimum_safeguards") else "No"}</td></tr>'
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_fossil_gas_nuclear(self, data: Dict[str, Any]) -> str:
        """Build HTML fossil gas/nuclear disclosure."""
        fgn = data.get("fossil_gas_nuclear", {})
        parts = ['<div class="section"><h2>Fossil Gas and Nuclear</h2>']
        parts.append('<table class="data-table">')
        parts.append("<tr><th>Activity</th><th>Eligible</th><th>Aligned</th></tr>")
        parts.append(
            f"<tr><td>Fossil Gas</td>"
            f"<td>{fgn.get('fossil_gas_eligible_pct', 0.0):.1f}%</td>"
            f"<td>{fgn.get('fossil_gas_aligned_pct', 0.0):.1f}%</td></tr>"
        )
        parts.append(
            f"<tr><td>Nuclear</td>"
            f"<td>{fgn.get('nuclear_eligible_pct', 0.0):.1f}%</td>"
            f"<td>{fgn.get('nuclear_aligned_pct', 0.0):.1f}%</td></tr>"
        )
        parts.append("</table>")
        statement = fgn.get("disclosure_statement", "")
        if statement:
            parts.append(f"<p>{_esc(statement)}</p>")
        parts.append("</div>")
        return "".join(parts)

    def _html_eligible_vs_aligned(self, data: Dict[str, Any]) -> str:
        """Build HTML eligible vs aligned table."""
        eva = data.get("eligible_vs_aligned", [])
        if not eva:
            return ""

        parts = ['<div class="section"><h2>Eligible vs. Aligned by Sector</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>Sector</th><th>Rev Elig.</th><th>Rev Aligned</th>"
            "<th>CapEx Elig.</th><th>CapEx Aligned</th></tr>"
        )
        for e in eva:
            parts.append(
                f"<tr><td>{_esc(e.get('sector', ''))}</td>"
                f"<td>{e.get('eligible_revenue_pct', 0.0):.1f}%</td>"
                f"<td>{e.get('aligned_revenue_pct', 0.0):.1f}%</td>"
                f"<td>{e.get('eligible_capex_pct', 0.0):.1f}%</td>"
                f"<td>{e.get('aligned_capex_pct', 0.0):.1f}%</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_commitment_adherence(self, data: Dict[str, Any]) -> str:
        """Build HTML commitment adherence."""
        commitments = data.get("commitment", [])
        if not commitments:
            return ""

        parts = ['<div class="section"><h2>Commitment Adherence</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>Commitment</th><th>Target</th><th>Actual</th>"
            "<th>Deviation</th><th>Status</th></tr>"
        )
        for c in commitments:
            status = c.get("status", "on_track")
            color = {"on_track": "#2ecc71", "at_risk": "#f39c12", "breached": "#e74c3c"}.get(
                status, "#2c3e50"
            )
            label = {"on_track": "ON TRACK", "at_risk": "AT RISK", "breached": "BREACHED"}.get(
                status, status.upper()
            )
            dev = c.get("deviation_pct")
            dev_str = f"{dev:+.1f}pp" if dev is not None else "N/A"
            parts.append(
                f"<tr><td>{_esc(c.get('commitment_name', ''))}</td>"
                f"<td>{c.get('target_pct', 0.0):.1f}%</td>"
                f"<td>{c.get('actual_pct', 0.0):.1f}%</td>"
                f"<td>{dev_str}</td>"
                f'<td style="color:{color};font-weight:bold;">{label}</td></tr>'
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_top_holdings(self, data: Dict[str, Any]) -> str:
        """Build HTML top holdings table."""
        holdings = data.get("top_holdings", [])
        if not holdings:
            return ""

        parts = ['<div class="section"><h2>Top Aligned Holdings</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>#</th><th>Name</th><th>Sector</th><th>Weight</th>"
            "<th>Aligned Rev.</th><th>Objective</th><th>Type</th></tr>"
        )
        for h in holdings:
            activity = (
                "Transitional" if h.get("transitional")
                else "Enabling" if h.get("enabling")
                else "Standard"
            )
            parts.append(
                f"<tr><td>{h.get('rank', 0)}</td>"
                f"<td>{_esc(h.get('name', ''))}</td>"
                f"<td>{_esc(h.get('sector', ''))}</td>"
                f"<td>{h.get('weight_pct', 0.0):.2f}%</td>"
                f"<td>{h.get('aligned_revenue_pct', 0.0):.1f}%</td>"
                f"<td>{_esc(h.get('primary_objective', ''))}</td>"
                f"<td>{_esc(activity)}</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Build HTML methodology notes."""
        notes = data.get("methodology_notes", "")
        if not notes:
            return ""
        return (
            f'<div class="section"><h2>Methodology Notes</h2>'
            f"<p>{_esc(notes)}</p></div>"
        )

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ascii_gauge(value: float, width: int = 20) -> str:
        """Create an ASCII gauge bar."""
        filled = int((value / 100) * width)
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown footer."""
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap in HTML document."""
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px auto; "
            "color: #2c3e50; line-height: 1.6; max-width: 1100px; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; }\n"
            ".section { margin-bottom: 25px; padding: 15px; background: #fafafa; "
            "border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #2c3e50; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f"<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
