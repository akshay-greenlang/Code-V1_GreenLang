"""
AuditTrailReportTemplate - Data lineage and provenance audit trail.

This module implements the audit trail report template for PACK-010
SFDR Article 8 products. It provides comprehensive audit trail
documentation including data lineage diagrams, calculation provenance
with SHA-256 hashes, methodology references, data source inventory,
estimation logs, and assumptions registry.

Example:
    >>> template = AuditTrailReportTemplate()
    >>> data = AuditTrailData(data_lineage=[...], ...)
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

class LineageNode(BaseModel):
    """Data lineage node."""

    node_id: str = Field("", description="Unique node identifier")
    node_type: str = Field(
        "data",
        description="data, transformation, aggregation, calculation, validation, output",
    )
    name: str = Field("", description="Node display name")
    description: str = Field("", description="Node description")
    source: str = Field("", description="Data source or upstream node")
    timestamp: str = Field("", description="Processing timestamp")
    record_count: Optional[int] = Field(None, ge=0, description="Record count")
    upstream_nodes: List[str] = Field(
        default_factory=list, description="Upstream node IDs"
    )
    downstream_nodes: List[str] = Field(
        default_factory=list, description="Downstream node IDs"
    )


class CalculationProvenance(BaseModel):
    """Provenance record for a calculation step."""

    calculation_id: str = Field("", description="Calculation identifier")
    calculation_name: str = Field("", description="Human-readable name")
    formula: str = Field("", description="Formula or method used")
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Input values used"
    )
    output_value: Any = Field(None, description="Calculated output value")
    output_unit: str = Field("", description="Output unit")
    timestamp: str = Field("", description="Calculation timestamp")
    input_hash: str = Field("", description="SHA-256 hash of inputs")
    output_hash: str = Field("", description="SHA-256 hash of output")
    agent_name: str = Field("", description="Agent that performed the calculation")
    deterministic: bool = Field(
        True, description="Whether calculation is deterministic (no LLM)"
    )


class MethodologyReference(BaseModel):
    """Reference to a methodology or standard used."""

    reference_id: str = Field("", description="Reference identifier")
    standard_name: str = Field("", description="Standard or methodology name")
    version: str = Field("", description="Version/edition")
    section: str = Field("", description="Specific section referenced")
    url: str = Field("", description="URL to standard")
    used_in: List[str] = Field(
        default_factory=list, description="Where this methodology is applied"
    )
    compliance_status: str = Field(
        "compliant", description="compliant, partially_compliant, non_compliant"
    )


class DataSourceInventory(BaseModel):
    """Inventory entry for a data source."""

    source_id: str = Field("", description="Source identifier")
    source_name: str = Field("", description="Source name")
    provider: str = Field("", description="Data provider")
    data_type: str = Field("", description="reported, estimated, modeled, proxy")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage %")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Quality score"
    )
    last_updated: str = Field("", description="Last update date")
    update_frequency: str = Field("", description="Update frequency")
    records_ingested: Optional[int] = Field(None, ge=0, description="Records ingested")
    validation_status: str = Field("validated", description="validated, pending, failed")


class EstimationEntry(BaseModel):
    """Log entry for an estimation/proxy used."""

    estimation_id: str = Field("", description="Estimation identifier")
    data_point: str = Field("", description="Data point estimated")
    estimation_method: str = Field("", description="Method used")
    proxy_source: str = Field("", description="Proxy source if applicable")
    confidence_level: str = Field("medium", description="low, medium, high")
    coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Coverage of estimated data"
    )
    impact_on_result: str = Field("", description="Impact on final result")
    justification: str = Field("", description="Justification for estimation")


class AssumptionEntry(BaseModel):
    """Registry entry for an assumption made."""

    assumption_id: str = Field("", description="Assumption identifier")
    assumption: str = Field("", description="Assumption statement")
    category: str = Field(
        "", description="Category: data, methodology, scope, temporal, other"
    )
    rationale: str = Field("", description="Rationale for the assumption")
    sensitivity: str = Field("low", description="low, medium, high")
    impact_if_wrong: str = Field("", description="Impact if assumption is incorrect")
    review_date: str = Field("", description="Next review date")
    approved_by: str = Field("", description="Who approved this assumption")


class AuditTrailData(BaseModel):
    """Complete input data for audit trail report."""

    fund_name: str = Field("", description="Fund name")
    reporting_period: str = Field("", description="Reporting period")
    data_lineage: List[LineageNode] = Field(default_factory=list)
    calculation_provenance: List[CalculationProvenance] = Field(default_factory=list)
    methodology_refs: List[MethodologyReference] = Field(default_factory=list)
    data_sources: List[DataSourceInventory] = Field(default_factory=list)
    estimations: List[EstimationEntry] = Field(default_factory=list)
    assumptions: List[AssumptionEntry] = Field(default_factory=list)
    overall_data_quality_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Overall data quality score"
    )
    auditor_notes: str = Field("", description="Additional auditor notes")


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AuditTrailReportTemplate:
    """
    Audit trail report template for SFDR Article 8 products.

    Generates comprehensive audit trail documentation with data lineage,
    calculation provenance, methodology references, data source inventory,
    estimation logs, and assumptions registry.

    Example:
        >>> template = AuditTrailReportTemplate()
        >>> md = template.render_markdown(data)
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "audit_trail_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AuditTrailReportTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render audit trail report in the specified format."""
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
        """Render audit trail report as Markdown."""
        sections: List[str] = [
            self._md_header(data),
            self._md_data_quality_overview(data),
            self._md_data_lineage(data),
            self._md_calculation_provenance(data),
            self._md_methodology_refs(data),
            self._md_data_source_inventory(data),
            self._md_estimation_log(data),
            self._md_assumptions_registry(data),
            self._md_auditor_notes(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render audit trail report as HTML."""
        sections: List[str] = [
            self._html_data_quality_overview(data),
            self._html_data_lineage(data),
            self._html_calculation_provenance(data),
            self._html_methodology_refs(data),
            self._html_data_source_inventory(data),
            self._html_estimation_log(data),
            self._html_assumptions_registry(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html("SFDR Audit Trail Report", body, provenance_hash)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render audit trail report as JSON."""
        report: Dict[str, Any] = {
            "report_type": "sfdr_audit_trail",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_name": data.get("fund_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "overall_data_quality_score": data.get("overall_data_quality_score"),
            "data_lineage": data.get("data_lineage", []),
            "calculation_provenance": data.get("calculation_provenance", []),
            "methodology_refs": data.get("methodology_refs", []),
            "data_sources": data.get("data_sources", []),
            "estimations": data.get("estimations", []),
            "assumptions": data.get("assumptions", []),
            "auditor_notes": data.get("auditor_notes", ""),
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
            f"# Audit Trail Report\n\n"
            f"**Fund:** {data.get('fund_name', 'Unknown')}\n\n"
            f"**Period:** {data.get('reporting_period', 'N/A')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_data_quality_overview(self, data: Dict[str, Any]) -> str:
        """Build data quality overview section."""
        score = data.get("overall_data_quality_score")
        sources = data.get("data_sources", [])
        estimations = data.get("estimations", [])
        assumptions = data.get("assumptions", [])

        total_sources = len(sources)
        validated = sum(1 for s in sources if s.get("validation_status") == "validated")
        estimated_count = len(estimations)
        assumption_count = len(assumptions)

        lines = ["## Data Quality Overview\n"]

        if score is not None:
            gauge = self._ascii_gauge(score)
            lines.append(f"**Overall Quality Score:** {gauge} {score:.1f}/100\n")

        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Data Sources | {total_sources} |")
        lines.append(f"| Validated Sources | {validated}/{total_sources} |")
        lines.append(f"| Estimations Used | {estimated_count} |")
        lines.append(f"| Assumptions Made | {assumption_count} |")

        high_sensitivity = sum(
            1 for a in assumptions if a.get("sensitivity") == "high"
        )
        if high_sensitivity > 0:
            lines.append(f"| High-Sensitivity Assumptions | {high_sensitivity} |")

        return "\n".join(lines)

    def _md_data_lineage(self, data: Dict[str, Any]) -> str:
        """Build data lineage diagram section."""
        nodes = data.get("data_lineage", [])
        if not nodes:
            return ""

        lines = ["## Data Lineage\n"]

        # ASCII flow diagram
        lines.append("### Data Flow\n")
        lines.append("```")
        for node in nodes:
            node_type = node.get("node_type", "data")
            icon = {
                "data": "[D]",
                "transformation": "[T]",
                "aggregation": "[A]",
                "calculation": "[C]",
                "validation": "[V]",
                "output": "[O]",
            }.get(node_type, "[?]")

            upstream = node.get("upstream_nodes", [])
            arrow = " <-- " + ", ".join(upstream) if upstream else ""
            records = node.get("record_count")
            rec_str = f" ({records:,} records)" if records is not None else ""

            lines.append(f"  {icon} {node.get('node_id', '')}: {node.get('name', '')}{rec_str}{arrow}")

        lines.append("```\n")

        # Detail table
        lines.append("### Node Details\n")
        lines.append("| ID | Type | Name | Source | Records | Timestamp |")
        lines.append("|----|------|------|--------|---------|-----------|")

        for node in nodes:
            records = node.get("record_count")
            rec_str = f"{records:,}" if records is not None else "N/A"
            lines.append(
                f"| {node.get('node_id', '')} | {node.get('node_type', '')} | "
                f"{node.get('name', '')} | {node.get('source', '')} | "
                f"{rec_str} | {node.get('timestamp', '')} |"
            )

        return "\n".join(lines)

    def _md_calculation_provenance(self, data: Dict[str, Any]) -> str:
        """Build calculation provenance section."""
        calcs = data.get("calculation_provenance", [])
        if not calcs:
            return ""

        lines = ["## Calculation Provenance\n"]

        for calc in calcs:
            deterministic = calc.get("deterministic", True)
            det_label = "DETERMINISTIC" if deterministic else "NON-DETERMINISTIC"

            lines.append(f"### {calc.get('calculation_name', 'Unknown')}\n")
            lines.append(f"**ID:** {calc.get('calculation_id', '')}\n")
            lines.append(f"**Formula:** `{calc.get('formula', '')}`\n")
            lines.append(f"**Type:** {det_label}\n")
            lines.append(f"**Agent:** {calc.get('agent_name', 'N/A')}\n")

            # Inputs
            inputs = calc.get("inputs", {})
            if inputs:
                lines.append("**Inputs:**\n")
                lines.append("| Parameter | Value |")
                lines.append("|-----------|-------|")
                for k, v in inputs.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")

            output = calc.get("output_value")
            unit = calc.get("output_unit", "")
            if output is not None:
                lines.append(f"**Output:** {output} {unit}\n")

            input_hash = calc.get("input_hash", "")
            output_hash = calc.get("output_hash", "")
            if input_hash:
                lines.append(f"**Input Hash:** `{input_hash}`\n")
            if output_hash:
                lines.append(f"**Output Hash:** `{output_hash}`\n")

            lines.append(f"**Timestamp:** {calc.get('timestamp', 'N/A')}\n")

        return "\n".join(lines)

    def _md_methodology_refs(self, data: Dict[str, Any]) -> str:
        """Build methodology references section."""
        refs = data.get("methodology_refs", [])
        if not refs:
            return ""

        lines = [
            "## Methodology References\n",
            "| ID | Standard | Version | Section | Status | Used In |",
            "|----|----------|---------|---------|--------|---------|",
        ]

        for r in refs:
            used_in = ", ".join(r.get("used_in", []))
            status = r.get("compliance_status", "compliant").upper()
            lines.append(
                f"| {r.get('reference_id', '')} | {r.get('standard_name', '')} | "
                f"{r.get('version', '')} | {r.get('section', '')} | "
                f"{status} | {used_in} |"
            )

        return "\n".join(lines)

    def _md_data_source_inventory(self, data: Dict[str, Any]) -> str:
        """Build data source inventory section."""
        sources = data.get("data_sources", [])
        if not sources:
            return ""

        lines = [
            "## Data Source Inventory\n",
            "| ID | Source | Provider | Type | Coverage | Quality | Status | Updated |",
            "|----|--------|----------|------|----------|---------|--------|---------|",
        ]

        for s in sources:
            quality = f"{s.get('quality_score', 0):.0f}" if s.get("quality_score") is not None else "N/A"
            records = s.get("records_ingested")
            lines.append(
                f"| {s.get('source_id', '')} | {s.get('source_name', '')} | "
                f"{s.get('provider', '')} | {s.get('data_type', '')} | "
                f"{s.get('coverage_pct', 0.0):.0f}% | {quality} | "
                f"{s.get('validation_status', '')} | {s.get('last_updated', '')} |"
            )

        return "\n".join(lines)

    def _md_estimation_log(self, data: Dict[str, Any]) -> str:
        """Build estimation log section."""
        estimations = data.get("estimations", [])
        if not estimations:
            return ""

        lines = [
            "## Estimation Log\n",
            "| ID | Data Point | Method | Confidence | Coverage | Impact |",
            "|----|------------|--------|------------|----------|--------|",
        ]

        for e in estimations:
            lines.append(
                f"| {e.get('estimation_id', '')} | {e.get('data_point', '')} | "
                f"{e.get('estimation_method', '')} | {e.get('confidence_level', '').upper()} | "
                f"{e.get('coverage_pct', 0.0):.0f}% | {e.get('impact_on_result', '')} |"
            )

        # Detailed justifications for low-confidence estimations
        low_conf = [e for e in estimations if e.get("confidence_level") == "low"]
        if low_conf:
            lines.append("\n### Low-Confidence Estimations\n")
            for e in low_conf:
                lines.append(f"**{e.get('estimation_id', '')} - {e.get('data_point', '')}**\n")
                lines.append(f"Justification: {e.get('justification', 'N/A')}\n")

        return "\n".join(lines)

    def _md_assumptions_registry(self, data: Dict[str, Any]) -> str:
        """Build assumptions registry section."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return ""

        lines = [
            "## Assumptions Registry\n",
            "| ID | Assumption | Category | Sensitivity | Review Date | Approved By |",
            "|----|------------|----------|-------------|-------------|-------------|",
        ]

        for a in assumptions:
            lines.append(
                f"| {a.get('assumption_id', '')} | {a.get('assumption', '')} | "
                f"{a.get('category', '')} | {a.get('sensitivity', '').upper()} | "
                f"{a.get('review_date', '')} | {a.get('approved_by', '')} |"
            )

        # Detail for high-sensitivity assumptions
        high = [a for a in assumptions if a.get("sensitivity") == "high"]
        if high:
            lines.append("\n### High-Sensitivity Assumptions\n")
            for a in high:
                lines.append(f"**{a.get('assumption_id', '')}:** {a.get('assumption', '')}\n")
                lines.append(f"- **Rationale:** {a.get('rationale', 'N/A')}")
                lines.append(f"- **Impact if Wrong:** {a.get('impact_if_wrong', 'N/A')}\n")

        return "\n".join(lines)

    def _md_auditor_notes(self, data: Dict[str, Any]) -> str:
        """Build auditor notes section."""
        notes = data.get("auditor_notes", "")
        if not notes:
            return ""
        return f"## Auditor Notes\n\n{notes}"

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_data_quality_overview(self, data: Dict[str, Any]) -> str:
        """Build HTML data quality overview."""
        score = data.get("overall_data_quality_score")
        sources = data.get("data_sources", [])
        estimations = data.get("estimations", [])
        assumptions = data.get("assumptions", [])

        validated = sum(1 for s in sources if s.get("validation_status") == "validated")

        parts = ['<div class="section"><h2>Data Quality Overview</h2>']

        if score is not None:
            color = self._quality_color(score)
            parts.append(
                f'<div style="text-align:center;margin:10px 0;">'
                f'<div style="font-size:0.9em;color:#7f8c8d;">Overall Quality Score</div>'
                f'<div style="font-size:2.5em;font-weight:bold;color:{color};">{score:.1f}</div>'
                f'<div style="background:#ecf0f1;height:10px;border-radius:5px;max-width:300px;margin:10px auto;">'
                f'<div style="background:{color};width:{score}%;height:100%;border-radius:5px;"></div>'
                f"</div></div>"
            )

        parts.append('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:15px 0;">')
        tiles = [
            ("Data Sources", str(len(sources))),
            ("Validated", f"{validated}/{len(sources)}"),
            ("Estimations", str(len(estimations))),
            ("Assumptions", str(len(assumptions))),
        ]
        for label, val in tiles:
            parts.append(
                f'<div style="text-align:center;padding:10px;background:#f8f9fa;border-radius:6px;">'
                f'<div style="font-size:0.85em;color:#7f8c8d;">{_esc(label)}</div>'
                f'<div style="font-size:1.3em;font-weight:bold;">{val}</div></div>'
            )
        parts.append("</div></div>")
        return "".join(parts)

    def _html_data_lineage(self, data: Dict[str, Any]) -> str:
        """Build HTML data lineage section."""
        nodes = data.get("data_lineage", [])
        if not nodes:
            return ""

        type_colors = {
            "data": "#3498db",
            "transformation": "#9b59b6",
            "aggregation": "#e67e22",
            "calculation": "#2ecc71",
            "validation": "#f39c12",
            "output": "#1abc9c",
        }

        parts = ['<div class="section"><h2>Data Lineage</h2>']

        # Visual flow
        for node in nodes:
            node_type = node.get("node_type", "data")
            color = type_colors.get(node_type, "#95a5a6")
            records = node.get("record_count")
            rec_str = f" - {records:,} records" if records is not None else ""
            upstream = node.get("upstream_nodes", [])

            parts.append(
                f'<div style="display:flex;align-items:center;margin:8px 0;">'
                f'<div style="background:{color};color:white;padding:8px 14px;'
                f'border-radius:6px;min-width:200px;">'
                f'<strong>{_esc(node.get("node_id", ""))}</strong><br>'
                f'<small>{_esc(node.get("name", ""))}{rec_str}</small></div>'
            )
            if upstream:
                parts.append(
                    f'<div style="margin-left:15px;color:#7f8c8d;">'
                    f'from: {", ".join(_esc(u) for u in upstream)}</div>'
                )
            parts.append("</div>")

        parts.append("</div>")
        return "".join(parts)

    def _html_calculation_provenance(self, data: Dict[str, Any]) -> str:
        """Build HTML calculation provenance."""
        calcs = data.get("calculation_provenance", [])
        if not calcs:
            return ""

        parts = ['<div class="section"><h2>Calculation Provenance</h2>']

        for calc in calcs:
            deterministic = calc.get("deterministic", True)
            det_color = "#2ecc71" if deterministic else "#f39c12"
            det_label = "DETERMINISTIC" if deterministic else "NON-DETERMINISTIC"

            parts.append(
                f'<div style="border:1px solid #ddd;padding:12px;margin:10px 0;'
                f'border-radius:6px;background:#fdfdfd;">'
                f"<h3>{_esc(calc.get('calculation_name', ''))}</h3>"
                f'<span style="background:{det_color};color:white;padding:2px 8px;'
                f'border-radius:3px;font-size:0.8em;">{det_label}</span>'
                f"<p><strong>Formula:</strong> <code>{_esc(calc.get('formula', ''))}</code></p>"
            )

            output = calc.get("output_value")
            if output is not None:
                parts.append(
                    f"<p><strong>Output:</strong> {output} {_esc(calc.get('output_unit', ''))}</p>"
                )

            input_hash = calc.get("input_hash", "")
            output_hash = calc.get("output_hash", "")
            if input_hash or output_hash:
                parts.append(
                    f'<div style="font-family:monospace;font-size:0.8em;'
                    f'background:#eaf2f8;padding:6px;border-radius:4px;margin-top:8px;">'
                )
                if input_hash:
                    parts.append(f"Input: {_esc(input_hash)}<br>")
                if output_hash:
                    parts.append(f"Output: {_esc(output_hash)}")
                parts.append("</div>")

            parts.append("</div>")

        parts.append("</div>")
        return "".join(parts)

    def _html_methodology_refs(self, data: Dict[str, Any]) -> str:
        """Build HTML methodology references."""
        refs = data.get("methodology_refs", [])
        if not refs:
            return ""

        parts = ['<div class="section"><h2>Methodology References</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>ID</th><th>Standard</th><th>Version</th>"
            "<th>Section</th><th>Status</th></tr>"
        )
        for r in refs:
            status = r.get("compliance_status", "compliant")
            color = {
                "compliant": "#2ecc71",
                "partially_compliant": "#f39c12",
                "non_compliant": "#e74c3c",
            }.get(status, "#2c3e50")
            parts.append(
                f"<tr><td>{_esc(r.get('reference_id', ''))}</td>"
                f"<td>{_esc(r.get('standard_name', ''))}</td>"
                f"<td>{_esc(r.get('version', ''))}</td>"
                f"<td>{_esc(r.get('section', ''))}</td>"
                f'<td style="color:{color};font-weight:bold;">{status.upper()}</td></tr>'
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_data_source_inventory(self, data: Dict[str, Any]) -> str:
        """Build HTML data source inventory."""
        sources = data.get("data_sources", [])
        if not sources:
            return ""

        parts = ['<div class="section"><h2>Data Source Inventory</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>ID</th><th>Source</th><th>Provider</th><th>Type</th>"
            "<th>Coverage</th><th>Quality</th><th>Status</th></tr>"
        )
        for s in sources:
            quality = f"{s.get('quality_score', 0):.0f}" if s.get("quality_score") is not None else "N/A"
            status = s.get("validation_status", "pending")
            status_color = {"validated": "#2ecc71", "pending": "#f39c12", "failed": "#e74c3c"}.get(
                status, "#2c3e50"
            )
            parts.append(
                f"<tr><td>{_esc(s.get('source_id', ''))}</td>"
                f"<td>{_esc(s.get('source_name', ''))}</td>"
                f"<td>{_esc(s.get('provider', ''))}</td>"
                f"<td>{_esc(s.get('data_type', ''))}</td>"
                f"<td>{s.get('coverage_pct', 0.0):.0f}%</td>"
                f"<td>{quality}</td>"
                f'<td style="color:{status_color};">{status.upper()}</td></tr>'
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_estimation_log(self, data: Dict[str, Any]) -> str:
        """Build HTML estimation log."""
        estimations = data.get("estimations", [])
        if not estimations:
            return ""

        parts = ['<div class="section"><h2>Estimation Log</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>ID</th><th>Data Point</th><th>Method</th>"
            "<th>Confidence</th><th>Coverage</th></tr>"
        )
        for e in estimations:
            conf = e.get("confidence_level", "medium")
            conf_color = {"low": "#e74c3c", "medium": "#f39c12", "high": "#2ecc71"}.get(
                conf, "#2c3e50"
            )
            parts.append(
                f"<tr><td>{_esc(e.get('estimation_id', ''))}</td>"
                f"<td>{_esc(e.get('data_point', ''))}</td>"
                f"<td>{_esc(e.get('estimation_method', ''))}</td>"
                f'<td style="color:{conf_color};font-weight:bold;">{conf.upper()}</td>'
                f"<td>{e.get('coverage_pct', 0.0):.0f}%</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_assumptions_registry(self, data: Dict[str, Any]) -> str:
        """Build HTML assumptions registry."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return ""

        parts = ['<div class="section"><h2>Assumptions Registry</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>ID</th><th>Assumption</th><th>Category</th>"
            "<th>Sensitivity</th><th>Review Date</th></tr>"
        )
        for a in assumptions:
            sens = a.get("sensitivity", "low")
            sens_color = {"low": "#3498db", "medium": "#f39c12", "high": "#e74c3c"}.get(
                sens, "#2c3e50"
            )
            parts.append(
                f"<tr><td>{_esc(a.get('assumption_id', ''))}</td>"
                f"<td>{_esc(a.get('assumption', ''))}</td>"
                f"<td>{_esc(a.get('category', ''))}</td>"
                f'<td style="color:{sens_color};font-weight:bold;">{sens.upper()}</td>'
                f"<td>{_esc(a.get('review_date', ''))}</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ascii_gauge(value: float, width: int = 20) -> str:
        """Create an ASCII gauge bar."""
        filled = int((value / 100) * width)
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    @staticmethod
    def _quality_color(score: float) -> str:
        """Return color based on quality score."""
        if score >= 80:
            return "#2ecc71"
        if score >= 60:
            return "#f39c12"
        return "#e74c3c"

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
            "h3 { color: #2c3e50; }\n"
            ".section { margin-bottom: 25px; padding: 15px; background: #fafafa; "
            "border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #2c3e50; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            "code { background: #f1f2f6; padding: 2px 6px; border-radius: 3px; }\n"
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
