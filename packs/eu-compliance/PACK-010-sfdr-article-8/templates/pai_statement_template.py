"""
PAIStatementTemplate - SFDR Principal Adverse Impact statement template.

This module implements the PAI indicator statement template for PACK-010
SFDR Article 8 products. It generates the mandatory PAI statement covering
all 18 mandatory indicators from Table 1 of Annex I of the SFDR RTS,
with year-over-year comparisons, explanations, and actions taken.

Example:
    >>> template = PAIStatementTemplate()
    >>> data = PAIStatementData(entity_name="Asset Manager AG", ...)
    >>> markdown = template.render_markdown(data.model_dump())
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  PAI Indicator Reference Data (18 Mandatory Indicators)
# ---------------------------------------------------------------------------

PAI_INDICATOR_DEFINITIONS: Dict[int, Dict[str, str]] = {
    1: {
        "name": "GHG Emissions",
        "category": "Climate and other environment-related indicators",
        "metric": "Scope 1, 2, 3 GHG emissions",
        "unit": "tCO2e",
    },
    2: {
        "name": "Carbon Footprint",
        "category": "Climate and other environment-related indicators",
        "metric": "Carbon footprint per EUR million invested",
        "unit": "tCO2e/EUR M invested",
    },
    3: {
        "name": "GHG Intensity of Investee Companies",
        "category": "Climate and other environment-related indicators",
        "metric": "GHG intensity per EUR million revenue",
        "unit": "tCO2e/EUR M revenue",
    },
    4: {
        "name": "Exposure to Fossil Fuel Sector",
        "category": "Climate and other environment-related indicators",
        "metric": "Share of investments in fossil fuel sector",
        "unit": "%",
    },
    5: {
        "name": "Non-renewable Energy Share",
        "category": "Climate and other environment-related indicators",
        "metric": "Share of non-renewable energy consumption/production",
        "unit": "%",
    },
    6: {
        "name": "Energy Consumption Intensity",
        "category": "Climate and other environment-related indicators",
        "metric": "Energy consumption intensity per high impact climate sector",
        "unit": "GWh/EUR M revenue",
    },
    7: {
        "name": "Biodiversity Impact",
        "category": "Climate and other environment-related indicators",
        "metric": "Activities negatively affecting biodiversity-sensitive areas",
        "unit": "Share %",
    },
    8: {
        "name": "Water Emissions",
        "category": "Climate and other environment-related indicators",
        "metric": "Emissions to water",
        "unit": "tonnes",
    },
    9: {
        "name": "Hazardous Waste Ratio",
        "category": "Climate and other environment-related indicators",
        "metric": "Hazardous waste and radioactive waste ratio",
        "unit": "tonnes",
    },
    10: {
        "name": "UNGC/OECD Violations",
        "category": "Social and employee, respect for human rights",
        "metric": "Violations of UNGC principles and OECD Guidelines",
        "unit": "Share %",
    },
    11: {
        "name": "UNGC/OECD Monitoring Gaps",
        "category": "Social and employee, respect for human rights",
        "metric": "Lack of processes and compliance mechanisms for UNGC/OECD",
        "unit": "Share %",
    },
    12: {
        "name": "Gender Pay Gap",
        "category": "Social and employee, respect for human rights",
        "metric": "Unadjusted gender pay gap",
        "unit": "%",
    },
    13: {
        "name": "Board Gender Diversity",
        "category": "Social and employee, respect for human rights",
        "metric": "Board gender diversity",
        "unit": "% female",
    },
    14: {
        "name": "Controversial Weapons Exposure",
        "category": "Social and employee, respect for human rights",
        "metric": "Exposure to controversial weapons",
        "unit": "Share %",
    },
    15: {
        "name": "GHG Intensity (Sovereigns)",
        "category": "Climate (sovereigns and supranationals)",
        "metric": "GHG intensity of investee countries",
        "unit": "tCO2e/EUR M GDP",
    },
    16: {
        "name": "Social Violations (Sovereigns)",
        "category": "Social (sovereigns and supranationals)",
        "metric": "Investee countries subject to social violations",
        "unit": "Count",
    },
    17: {
        "name": "Real Estate Fossil Fuels",
        "category": "Additional: Fossil fuels (real estate)",
        "metric": "Exposure to fossil fuels through real estate assets",
        "unit": "Share %",
    },
    18: {
        "name": "Real Estate Energy Inefficiency",
        "category": "Additional: Energy efficiency (real estate)",
        "metric": "Exposure to energy-inefficient real estate assets",
        "unit": "Share %",
    },
}


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class PAIIndicatorData(BaseModel):
    """Data for a single PAI indicator."""

    indicator_id: int = Field(..., ge=1, le=18, description="PAI indicator number (1-18)")
    name: str = Field("", description="Indicator name (auto-filled if empty)")
    category: str = Field("", description="Indicator category (auto-filled if empty)")
    metric: str = Field("", description="Metric description (auto-filled if empty)")
    value: float = Field(0.0, description="Current period value")
    unit: str = Field("", description="Unit (auto-filled if empty)")
    coverage_pct: float = Field(
        100.0, ge=0.0, le=100.0, description="Data coverage percentage"
    )
    previous_value: Optional[float] = Field(None, description="Previous period value")
    explanation: str = Field("", description="Explanation of the metric value")
    actions_taken: str = Field("", description="Actions taken to address this indicator")
    engagement_outcomes: str = Field("", description="Engagement outcomes related to this indicator")
    data_source: str = Field("", description="Primary data source")

    def model_post_init(self, __context: Any) -> None:
        """Auto-fill reference data from PAI definitions if not provided."""
        ref = PAI_INDICATOR_DEFINITIONS.get(self.indicator_id, {})
        if not self.name:
            self.name = ref.get("name", "")
        if not self.category:
            self.category = ref.get("category", "")
        if not self.metric:
            self.metric = ref.get("metric", "")
        if not self.unit:
            self.unit = ref.get("unit", "")


class PAINarrative(BaseModel):
    """Narrative sections for the PAI statement."""

    entity_description: str = Field(
        "", description="Description of the financial market participant"
    )
    policies_description: str = Field(
        "", description="Description of policies to identify and prioritize PAIs"
    )
    engagement_policies: str = Field(
        "", description="Description of engagement policies"
    )
    international_standards: str = Field(
        "", description="Adherence to international standards"
    )
    historical_comparison: str = Field(
        "", description="Historical comparison narrative"
    )


class PAIStatementData(BaseModel):
    """Complete input data for PAI statement template."""

    entity_name: str = Field(..., min_length=1, description="Entity name")
    entity_lei: str = Field("", description="Entity LEI")
    reporting_period_start: str = Field(..., description="Period start (YYYY-MM-DD)")
    reporting_period_end: str = Field(..., description="Period end (YYYY-MM-DD)")
    indicators: List[PAIIndicatorData] = Field(
        default_factory=list, description="PAI indicator data"
    )
    narrative: PAINarrative = Field(default_factory=PAINarrative)
    additional_indicators: List[Dict[str, Any]] = Field(
        default_factory=list, description="Optional additional PAI indicators"
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class PAIStatementTemplate:
    """
    SFDR PAI statement template.

    Generates the mandatory Principal Adverse Impact statement covering
    all 18 indicators from SFDR RTS Annex I Table 1, with year-over-year
    comparisons, explanations, and actions taken.

    Example:
        >>> template = PAIStatementTemplate()
        >>> md = template.render_markdown(data)
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "pai_statement"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PAIStatementTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render PAI statement in the specified format."""
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
        """Render PAI statement as Markdown."""
        sections: List[str] = [
            self._md_header(data),
            self._md_entity_info(data),
            self._md_indicator_summary_table(data),
            self._md_climate_indicators(data),
            self._md_social_indicators(data),
            self._md_sovereign_indicators(data),
            self._md_real_estate_indicators(data),
            self._md_additional_indicators(data),
            self._md_narrative_sections(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render PAI statement as self-contained HTML."""
        sections: List[str] = [
            self._html_entity_info(data),
            self._html_indicator_summary(data),
            self._html_indicator_details(data),
            self._html_narrative(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html("SFDR PAI Statement", body, provenance_hash)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render PAI statement as structured JSON."""
        report: Dict[str, Any] = {
            "report_type": "sfdr_pai_statement",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "entity_name": data.get("entity_name", ""),
            "entity_lei": data.get("entity_lei", ""),
            "reporting_period": {
                "start": data.get("reporting_period_start", ""),
                "end": data.get("reporting_period_end", ""),
            },
            "indicators": self._json_indicators(data),
            "additional_indicators": data.get("additional_indicators", []),
            "narrative": data.get("narrative", {}),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown header."""
        return (
            f"# Principal Adverse Impact Statement\n\n"
            f"**SFDR RTS Annex I - Table 1**\n\n"
            f"**Entity:** {data.get('entity_name', 'Unknown')}\n\n"
            f"**Period:** {data.get('reporting_period_start', '')} to "
            f"{data.get('reporting_period_end', '')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_entity_info(self, data: Dict[str, Any]) -> str:
        """Build entity information section."""
        narrative = data.get("narrative", {})
        lines = ["## Entity Information\n"]
        lines.append(f"**Entity:** {data.get('entity_name', 'N/A')}\n")
        lei = data.get("entity_lei", "")
        if lei:
            lines.append(f"**LEI:** {lei}\n")
        desc = narrative.get("entity_description", "")
        if desc:
            lines.append(f"{desc}")
        return "\n".join(lines)

    def _md_indicator_summary_table(self, data: Dict[str, Any]) -> str:
        """Build summary table of all 18 PAI indicators."""
        indicators = data.get("indicators", [])
        indicator_map = {i.get("indicator_id"): i for i in indicators}

        lines = [
            "## PAI Indicator Summary\n",
            "| # | Indicator | Value | Unit | Previous | YoY Change | Coverage |",
            "|---|-----------|-------|------|----------|------------|----------|",
        ]

        for ind_id in range(1, 19):
            ref = PAI_INDICATOR_DEFINITIONS.get(ind_id, {})
            ind = indicator_map.get(ind_id, {})

            value = ind.get("value", 0.0) if ind else 0.0
            unit = ind.get("unit", ref.get("unit", ""))
            prev = ind.get("previous_value") if ind else None
            coverage = ind.get("coverage_pct", 0.0) if ind else 0.0

            prev_str = f"{prev:.4f}" if prev is not None else "N/A"
            yoy = self._calc_yoy(value, prev)
            yoy_str = f"{yoy:+.2f}%" if yoy is not None else "N/A"

            lines.append(
                f"| {ind_id} | {ref.get('name', '')} | "
                f"{value:.4f} | {unit} | {prev_str} | {yoy_str} | "
                f"{coverage:.0f}% |"
            )

        return "\n".join(lines)

    def _md_climate_indicators(self, data: Dict[str, Any]) -> str:
        """Build detailed climate indicator sections (PAI 1-9)."""
        return self._md_indicator_group(
            data, range(1, 10), "Climate and Environment Indicators (PAI 1-9)"
        )

    def _md_social_indicators(self, data: Dict[str, Any]) -> str:
        """Build detailed social indicator sections (PAI 10-14)."""
        return self._md_indicator_group(
            data, range(10, 15), "Social and Governance Indicators (PAI 10-14)"
        )

    def _md_sovereign_indicators(self, data: Dict[str, Any]) -> str:
        """Build sovereign indicator sections (PAI 15-16)."""
        return self._md_indicator_group(
            data, range(15, 17), "Sovereigns and Supranationals (PAI 15-16)"
        )

    def _md_real_estate_indicators(self, data: Dict[str, Any]) -> str:
        """Build real estate indicator sections (PAI 17-18)."""
        return self._md_indicator_group(
            data, range(17, 19), "Real Estate Indicators (PAI 17-18)"
        )

    def _md_indicator_group(
        self, data: Dict[str, Any], id_range: range, title: str
    ) -> str:
        """Build a group of indicator detail sections."""
        indicators = data.get("indicators", [])
        indicator_map = {i.get("indicator_id"): i for i in indicators}

        lines = [f"## {title}\n"]

        for ind_id in id_range:
            ref = PAI_INDICATOR_DEFINITIONS.get(ind_id, {})
            ind = indicator_map.get(ind_id, {})

            value = ind.get("value", 0.0) if ind else 0.0
            unit = ind.get("unit", ref.get("unit", ""))
            prev = ind.get("previous_value") if ind else None
            yoy = self._calc_yoy(value, prev)

            lines.append(f"### PAI #{ind_id}: {ref.get('name', 'Unknown')}\n")
            lines.append(f"**Category:** {ref.get('category', '')}\n")
            lines.append(f"**Metric:** {ref.get('metric', '')}\n")
            lines.append(f"**Value:** {value:.4f} {unit}\n")

            if prev is not None:
                lines.append(f"**Previous Period:** {prev:.4f} {unit}\n")
                yoy_str = f"{yoy:+.2f}%" if yoy is not None else "N/A"
                lines.append(f"**Year-over-Year Change:** {yoy_str}\n")

            coverage = ind.get("coverage_pct", 0.0) if ind else 0.0
            lines.append(f"**Data Coverage:** {coverage:.0f}%\n")

            explanation = ind.get("explanation", "") if ind else ""
            if explanation:
                lines.append(f"**Explanation:** {explanation}\n")

            actions = ind.get("actions_taken", "") if ind else ""
            if actions:
                lines.append(f"**Actions Taken:** {actions}\n")

            engagement = ind.get("engagement_outcomes", "") if ind else ""
            if engagement:
                lines.append(f"**Engagement Outcomes:** {engagement}\n")

        return "\n".join(lines)

    def _md_additional_indicators(self, data: Dict[str, Any]) -> str:
        """Build additional (optional) indicators section."""
        additional = data.get("additional_indicators", [])
        if not additional:
            return ""

        lines = ["## Additional PAI Indicators\n"]
        lines.append("| Indicator | Value | Unit | Explanation |")
        lines.append("|-----------|-------|------|-------------|")
        for a in additional:
            lines.append(
                f"| {a.get('name', '')} | {a.get('value', 0):.4f} | "
                f"{a.get('unit', '')} | {a.get('explanation', '')} |"
            )
        return "\n".join(lines)

    def _md_narrative_sections(self, data: Dict[str, Any]) -> str:
        """Build narrative sections."""
        narrative = data.get("narrative", {})
        lines = []

        policies = narrative.get("policies_description", "")
        if policies:
            lines.append(f"## Policies to Identify and Prioritize PAIs\n\n{policies}\n")

        engagement = narrative.get("engagement_policies", "")
        if engagement:
            lines.append(f"## Engagement Policies\n\n{engagement}\n")

        standards = narrative.get("international_standards", "")
        if standards:
            lines.append(
                f"## Adherence to International Standards\n\n{standards}\n"
            )

        historical = narrative.get("historical_comparison", "")
        if historical:
            lines.append(f"## Historical Comparison\n\n{historical}")

        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_entity_info(self, data: Dict[str, Any]) -> str:
        """Build HTML entity information."""
        parts = ['<div class="section"><h2>Entity Information</h2>']
        parts.append(f"<p><strong>Entity:</strong> {_esc(data.get('entity_name', ''))}</p>")
        lei = data.get("entity_lei", "")
        if lei:
            parts.append(f"<p><strong>LEI:</strong> {_esc(lei)}</p>")
        parts.append(
            f"<p><strong>Period:</strong> {_esc(data.get('reporting_period_start', ''))} to "
            f"{_esc(data.get('reporting_period_end', ''))}</p>"
        )
        parts.append("</div>")
        return "".join(parts)

    def _html_indicator_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML indicator summary table."""
        indicators = data.get("indicators", [])
        indicator_map = {i.get("indicator_id"): i for i in indicators}

        parts = ['<div class="section"><h2>PAI Indicator Summary</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>#</th><th>Indicator</th><th>Value</th>"
            "<th>Unit</th><th>Previous</th><th>YoY</th><th>Coverage</th></tr>"
        )

        for ind_id in range(1, 19):
            ref = PAI_INDICATOR_DEFINITIONS.get(ind_id, {})
            ind = indicator_map.get(ind_id, {})
            value = ind.get("value", 0.0) if ind else 0.0
            unit = ind.get("unit", ref.get("unit", ""))
            prev = ind.get("previous_value") if ind else None
            coverage = ind.get("coverage_pct", 0.0) if ind else 0.0
            yoy = self._calc_yoy(value, prev)

            prev_str = f"{prev:.4f}" if prev is not None else "N/A"
            yoy_str = f"{yoy:+.2f}%" if yoy is not None else "N/A"
            yoy_color = self._yoy_color(yoy, ind_id)

            parts.append(
                f"<tr><td>{ind_id}</td>"
                f"<td>{_esc(ref.get('name', ''))}</td>"
                f"<td>{value:.4f}</td>"
                f"<td>{_esc(unit)}</td>"
                f"<td>{prev_str}</td>"
                f'<td style="color:{yoy_color};font-weight:bold;">{yoy_str}</td>'
                f"<td>{coverage:.0f}%</td></tr>"
            )

        parts.append("</table></div>")
        return "".join(parts)

    def _html_indicator_details(self, data: Dict[str, Any]) -> str:
        """Build HTML indicator detail cards."""
        indicators = data.get("indicators", [])
        indicator_map = {i.get("indicator_id"): i for i in indicators}

        parts = ['<div class="section"><h2>Indicator Details</h2>']

        for ind_id in range(1, 19):
            ref = PAI_INDICATOR_DEFINITIONS.get(ind_id, {})
            ind = indicator_map.get(ind_id, {})

            value = ind.get("value", 0.0) if ind else 0.0
            unit = ind.get("unit", ref.get("unit", ""))

            parts.append(
                f'<div style="border:1px solid #ddd;padding:12px;margin:10px 0;'
                f'border-radius:6px;background:#fdfdfd;">'
                f"<h3>PAI #{ind_id}: {_esc(ref.get('name', ''))}</h3>"
                f"<p><strong>Category:</strong> {_esc(ref.get('category', ''))}</p>"
                f"<p><strong>Value:</strong> {value:.4f} {_esc(unit)}</p>"
            )

            explanation = ind.get("explanation", "") if ind else ""
            if explanation:
                parts.append(f"<p><strong>Explanation:</strong> {_esc(explanation)}</p>")

            actions = ind.get("actions_taken", "") if ind else ""
            if actions:
                parts.append(f"<p><strong>Actions:</strong> {_esc(actions)}</p>")

            parts.append("</div>")

        parts.append("</div>")
        return "".join(parts)

    def _html_narrative(self, data: Dict[str, Any]) -> str:
        """Build HTML narrative sections."""
        narrative = data.get("narrative", {})
        parts = ['<div class="section"><h2>Narrative Disclosures</h2>']

        policies = narrative.get("policies_description", "")
        if policies:
            parts.append(f"<h3>Policies</h3><p>{_esc(policies)}</p>")

        engagement = narrative.get("engagement_policies", "")
        if engagement:
            parts.append(f"<h3>Engagement</h3><p>{_esc(engagement)}</p>")

        standards = narrative.get("international_standards", "")
        if standards:
            parts.append(f"<h3>International Standards</h3><p>{_esc(standards)}</p>")

        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  JSON helpers
    # ------------------------------------------------------------------ #

    def _json_indicators(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON indicator list with reference data enrichment."""
        indicators = data.get("indicators", [])
        indicator_map = {i.get("indicator_id"): i for i in indicators}
        result: List[Dict[str, Any]] = []

        for ind_id in range(1, 19):
            ref = PAI_INDICATOR_DEFINITIONS.get(ind_id, {})
            ind = indicator_map.get(ind_id, {})

            value = ind.get("value", 0.0) if ind else 0.0
            prev = ind.get("previous_value") if ind else None
            yoy = self._calc_yoy(value, prev)

            result.append({
                "indicator_id": ind_id,
                "name": ref.get("name", ""),
                "category": ref.get("category", ""),
                "metric": ref.get("metric", ""),
                "value": value,
                "unit": ind.get("unit", ref.get("unit", "")),
                "coverage_pct": ind.get("coverage_pct", 0.0) if ind else 0.0,
                "previous_value": prev,
                "yoy_change": yoy,
                "explanation": ind.get("explanation", "") if ind else "",
                "actions_taken": ind.get("actions_taken", "") if ind else "",
                "data_source": ind.get("data_source", "") if ind else "",
            })

        return result

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_yoy(current: float, previous: Optional[float]) -> Optional[float]:
        """Calculate year-over-year percentage change."""
        if previous is None or previous == 0:
            return None
        return ((current - previous) / abs(previous)) * 100

    @staticmethod
    def _yoy_color(yoy: Optional[float], indicator_id: int) -> str:
        """Return color for YoY change (red = worse, green = better)."""
        if yoy is None:
            return "#2c3e50"
        # For most PAI indicators, lower is better (except board diversity #13)
        if indicator_id == 13:
            return "#2ecc71" if yoy > 0 else "#e74c3c" if yoy < 0 else "#2c3e50"
        return "#e74c3c" if yoy > 0 else "#2ecc71" if yoy < 0 else "#2c3e50"

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
        """Wrap HTML body in a complete document."""
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
            "h3 { color: #2c3e50; margin-top: 15px; }\n"
            ".section { margin-bottom: 25px; padding: 15px; background: #fafafa; "
            "border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 10px; border: 1px solid #ddd; "
            "font-size: 0.9em; }\n"
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
