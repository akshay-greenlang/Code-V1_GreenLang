"""
EBA Pillar 3 GAR Report Template - PACK-008 EU Taxonomy Alignment Pack

This module generates EBA Pillar 3 ESG disclosure Templates 6-10 for credit institutions:
- Template 6: GAR summary (stock)
- Template 7: GAR by sector/counterparty
- Template 8: BTAR (Banking Book Taxonomy Alignment Ratio)
- Template 9: GAR flow
- Template 10: Other mitigating actions

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from eba_pillar3_gar_report import EBAPillar3GARReportTemplate, ReportData
    >>> data = ReportData(
    ...     institution_name="Green Bank AG",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = EBAPillar3GARReportTemplate()
    >>> report = template.generate_full_eba_report(data, format="markdown")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """Configuration for EBA Pillar 3 GAR Report generation."""

    include_template_6: bool = Field(default=True, description="Include GAR summary (stock)")
    include_template_7: bool = Field(default=True, description="Include GAR by sector")
    include_template_8: bool = Field(default=True, description="Include BTAR")
    include_template_9: bool = Field(default=True, description="Include GAR flow")
    include_template_10: bool = Field(default=True, description="Include other mitigating actions")
    currency: str = Field(default="EUR", description="Reporting currency")


class GARExposureRow(BaseModel):
    """A single exposure row in the GAR calculation."""

    exposure_type: str = Field(
        ...,
        description="Exposure type (loans, debt securities, equity, mortgages, etc.)"
    )
    total_gross_carrying_amount_eur: float = Field(
        ..., ge=0,
        description="Total gross carrying amount in EUR"
    )
    taxonomy_aligned_eur: float = Field(
        default=0.0, ge=0,
        description="Taxonomy-aligned amount in EUR"
    )
    taxonomy_eligible_eur: float = Field(
        default=0.0, ge=0,
        description="Taxonomy-eligible (not aligned) amount in EUR"
    )
    non_eligible_eur: float = Field(
        default=0.0, ge=0,
        description="Non-eligible amount in EUR"
    )
    ccm_aligned_eur: float = Field(default=0.0, ge=0, description="CCM-aligned amount")
    cca_aligned_eur: float = Field(default=0.0, ge=0, description="CCA-aligned amount")
    wtr_aligned_eur: float = Field(default=0.0, ge=0, description="WTR-aligned amount")
    ce_aligned_eur: float = Field(default=0.0, ge=0, description="CE-aligned amount")
    ppc_aligned_eur: float = Field(default=0.0, ge=0, description="PPC-aligned amount")
    bio_aligned_eur: float = Field(default=0.0, ge=0, description="BIO-aligned amount")


class GARSectorRow(BaseModel):
    """GAR breakdown by sector/counterparty type."""

    sector: str = Field(..., description="NACE sector or counterparty type")
    total_exposures_eur: float = Field(..., ge=0, description="Total exposures")
    aligned_eur: float = Field(default=0.0, ge=0, description="Aligned amount")
    aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Aligned percentage")
    eligible_eur: float = Field(default=0.0, ge=0, description="Eligible amount")
    eligible_pct: float = Field(default=0.0, ge=0, le=100, description="Eligible percentage")
    counterparty_count: int = Field(default=0, ge=0, description="Number of counterparties")


class BTARRow(BaseModel):
    """Banking Book Taxonomy Alignment Ratio row."""

    portfolio_segment: str = Field(..., description="Portfolio segment")
    total_assets_eur: float = Field(..., ge=0, description="Total assets")
    aligned_assets_eur: float = Field(default=0.0, ge=0, description="Aligned assets")
    btar_pct: float = Field(default=0.0, ge=0, le=100, description="BTAR percentage")
    eligible_assets_eur: float = Field(default=0.0, ge=0, description="Eligible assets")
    non_eligible_eur: float = Field(default=0.0, ge=0, description="Non-eligible assets")


class GARFlowRow(BaseModel):
    """GAR flow data (new originations in reporting period)."""

    exposure_type: str = Field(..., description="Exposure type")
    new_originations_eur: float = Field(..., ge=0, description="New originations in EUR")
    aligned_eur: float = Field(default=0.0, ge=0, description="Aligned amount")
    aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Aligned percentage")
    eligible_eur: float = Field(default=0.0, ge=0, description="Eligible amount")


class MitigatingAction(BaseModel):
    """Other mitigating action for Template 10."""

    action_type: str = Field(..., description="Type of mitigating action")
    description: str = Field(..., description="Description of the action")
    exposure_covered_eur: float = Field(default=0.0, ge=0, description="Exposure covered")
    impact_assessment: str = Field(default="", description="Impact assessment summary")
    status: str = Field(default="In Progress", description="Action status")


class ReportData(BaseModel):
    """Data model for EBA Pillar 3 GAR Report."""

    institution_name: str = Field(..., description="Credit institution name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    lei_code: str = Field(default="", description="Legal Entity Identifier")
    gar_stock_rows: List[GARExposureRow] = Field(
        default_factory=list,
        description="Template 6: GAR stock exposure rows"
    )
    gar_sector_rows: List[GARSectorRow] = Field(
        default_factory=list,
        description="Template 7: GAR by sector rows"
    )
    btar_rows: List[BTARRow] = Field(
        default_factory=list,
        description="Template 8: BTAR rows"
    )
    gar_flow_rows: List[GARFlowRow] = Field(
        default_factory=list,
        description="Template 9: GAR flow rows"
    )
    mitigating_actions: List[MitigatingAction] = Field(
        default_factory=list,
        description="Template 10: Other mitigating actions"
    )
    total_on_balance_eur: float = Field(
        default=0.0, ge=0,
        description="Total on-balance-sheet assets"
    )
    total_gar_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Overall GAR percentage"
    )
    total_btar_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Overall BTAR percentage"
    )
    notes: List[str] = Field(default_factory=list, description="Report notes")

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class EBAPillar3GARReportTemplate:
    """
    EBA Pillar 3 GAR Report Template for EU Taxonomy Alignment Pack.

    Generates EBA Pillar 3 ESG disclosure Templates 6-10 for credit institutions
    covering GAR stock, GAR by sector, BTAR, GAR flow, and mitigating actions.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = EBAPillar3GARReportTemplate()
        >>> report = template.generate_full_eba_report(data, format="markdown")
        >>> assert "Template 6" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize EBA Pillar 3 GAR Report Template."""
        self.config = config or ReportConfig()

    def generate_full_eba_report(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Generate the full EBA Pillar 3 ESG disclosure report.

        Args:
            data: Report data
            format: Output format

        Returns:
            Full EBA Pillar 3 report

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Generating full EBA Pillar 3 GAR report for {data.institution_name} "
            f"in {format} format"
        )

        if format == "markdown":
            content = self._render_full_markdown(data)
        elif format == "html":
            content = self._render_full_html(data)
        elif format == "json":
            content = self._render_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        content_hash = self._calculate_hash(content)
        logger.info(f"Report generated with hash: {content_hash}")

        return content

    def generate_template_6(self, data: ReportData) -> str:
        """Generate Template 6: GAR summary (stock)."""
        return self._render_template_6(data)

    def generate_template_7(self, data: ReportData) -> str:
        """Generate Template 7: GAR by sector/counterparty."""
        return self._render_template_7(data)

    def generate_template_8(self, data: ReportData) -> str:
        """Generate Template 8: BTAR."""
        return self._render_template_8(data)

    def generate_template_9(self, data: ReportData) -> str:
        """Generate Template 9: GAR flow."""
        return self._render_template_9(data)

    def generate_template_10(self, data: ReportData) -> str:
        """Generate Template 10: Other mitigating actions."""
        return self._render_template_10(data)

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """Main render entry point (alias for generate_full_eba_report)."""
        return self.generate_full_eba_report(data, format)

    def _render_template_6(self, data: ReportData) -> str:
        """Render Template 6: GAR summary (stock)."""
        sections = []
        sections.append(f"### Template 6: Green Asset Ratio (GAR) - Summary (Stock)")
        sections.append(f"")
        sections.append(
            f"**Total On-Balance-Sheet Assets:** EUR {data.total_on_balance_eur:,.0f}"
        )
        sections.append(f"**Overall GAR:** {data.total_gar_pct:.2f}%")
        sections.append(f"")

        sections.append(
            f"| Exposure Type | Gross Carrying (EUR) | Aligned (EUR) | "
            f"Eligible (EUR) | Non-Eligible (EUR) | CCM | CCA | WTR | CE | PPC | BIO |"
        )
        sections.append(
            f"|---------------|--------------------:|-------------:|"
            f"--------------:|------------------:|----:|----:|----:|---:|----:|----:|"
        )

        for row in data.gar_stock_rows:
            aligned_pct = (row.taxonomy_aligned_eur / row.total_gross_carrying_amount_eur * 100) if row.total_gross_carrying_amount_eur > 0 else 0
            sections.append(
                f"| {row.exposure_type} | {row.total_gross_carrying_amount_eur:,.0f} | "
                f"{row.taxonomy_aligned_eur:,.0f} | "
                f"{row.taxonomy_eligible_eur:,.0f} | "
                f"{row.non_eligible_eur:,.0f} | "
                f"{row.ccm_aligned_eur:,.0f} | {row.cca_aligned_eur:,.0f} | "
                f"{row.wtr_aligned_eur:,.0f} | {row.ce_aligned_eur:,.0f} | "
                f"{row.ppc_aligned_eur:,.0f} | {row.bio_aligned_eur:,.0f} |"
            )

        total_aligned = sum(r.taxonomy_aligned_eur for r in data.gar_stock_rows)
        total_gca = sum(r.total_gross_carrying_amount_eur for r in data.gar_stock_rows)
        sections.append(
            f"| **Total** | **{total_gca:,.0f}** | **{total_aligned:,.0f}** | | | | | | | | |"
        )
        sections.append(f"")

        return "\n".join(sections)

    def _render_template_7(self, data: ReportData) -> str:
        """Render Template 7: GAR by sector/counterparty."""
        sections = []
        sections.append(f"### Template 7: GAR by Sector / Counterparty Type")
        sections.append(f"")

        sections.append(
            f"| Sector | Total (EUR) | Aligned (EUR) | Aligned % | "
            f"Eligible (EUR) | Eligible % | Counterparties |"
        )
        sections.append(
            f"|--------|----------:|-------------:|----------:|"
            f"--------------:|----------:|---------------:|"
        )

        for row in data.gar_sector_rows:
            sections.append(
                f"| {row.sector} | {row.total_exposures_eur:,.0f} | "
                f"{row.aligned_eur:,.0f} | {row.aligned_pct:.1f}% | "
                f"{row.eligible_eur:,.0f} | {row.eligible_pct:.1f}% | "
                f"{row.counterparty_count} |"
            )
        sections.append(f"")

        return "\n".join(sections)

    def _render_template_8(self, data: ReportData) -> str:
        """Render Template 8: BTAR."""
        sections = []
        sections.append(f"### Template 8: Banking Book Taxonomy Alignment Ratio (BTAR)")
        sections.append(f"")
        sections.append(f"**Overall BTAR:** {data.total_btar_pct:.2f}%")
        sections.append(f"")

        sections.append(
            f"| Portfolio Segment | Total Assets (EUR) | Aligned (EUR) | "
            f"BTAR % | Eligible (EUR) | Non-Eligible (EUR) |"
        )
        sections.append(
            f"|-------------------|-----------------:|-------------:|"
            f"------:|--------------:|------------------:|"
        )

        for row in data.btar_rows:
            sections.append(
                f"| {row.portfolio_segment} | {row.total_assets_eur:,.0f} | "
                f"{row.aligned_assets_eur:,.0f} | {row.btar_pct:.2f}% | "
                f"{row.eligible_assets_eur:,.0f} | {row.non_eligible_eur:,.0f} |"
            )
        sections.append(f"")

        return "\n".join(sections)

    def _render_template_9(self, data: ReportData) -> str:
        """Render Template 9: GAR flow."""
        sections = []
        sections.append(f"### Template 9: GAR Flow (New Originations)")
        sections.append(f"")

        sections.append(
            f"| Exposure Type | New Originations (EUR) | Aligned (EUR) | "
            f"Aligned % | Eligible (EUR) |"
        )
        sections.append(
            f"|---------------|---------------------:|-------------:|"
            f"----------:|--------------:|"
        )

        for row in data.gar_flow_rows:
            sections.append(
                f"| {row.exposure_type} | {row.new_originations_eur:,.0f} | "
                f"{row.aligned_eur:,.0f} | {row.aligned_pct:.1f}% | "
                f"{row.eligible_eur:,.0f} |"
            )

        total_new = sum(r.new_originations_eur for r in data.gar_flow_rows)
        total_aligned_flow = sum(r.aligned_eur for r in data.gar_flow_rows)
        flow_pct = (total_aligned_flow / total_new * 100) if total_new > 0 else 0
        sections.append(
            f"| **Total** | **{total_new:,.0f}** | **{total_aligned_flow:,.0f}** | "
            f"**{flow_pct:.1f}%** | |"
        )
        sections.append(f"")

        return "\n".join(sections)

    def _render_template_10(self, data: ReportData) -> str:
        """Render Template 10: Other mitigating actions."""
        sections = []
        sections.append(f"### Template 10: Other Mitigating Actions")
        sections.append(f"")

        if not data.mitigating_actions:
            sections.append(f"No other mitigating actions reported.")
            sections.append(f"")
            return "\n".join(sections)

        sections.append(
            f"| Action Type | Description | Exposure Covered (EUR) | "
            f"Impact Assessment | Status |"
        )
        sections.append(
            f"|-------------|-------------|---------------------:|"
            f"-------------------|--------|"
        )

        for action in data.mitigating_actions:
            sections.append(
                f"| {action.action_type} | {action.description[:50]} | "
                f"{action.exposure_covered_eur:,.0f} | "
                f"{action.impact_assessment[:40]} | {action.status} |"
            )
        sections.append(f"")

        return "\n".join(sections)

    def _render_full_markdown(self, data: ReportData) -> str:
        """Render the full EBA Pillar 3 report in Markdown."""
        sections = []

        # Header
        sections.append(f"# EBA Pillar 3 ESG Disclosure - GAR Report")
        sections.append(f"")
        sections.append(f"**Institution:** {data.institution_name}")
        if data.lei_code:
            sections.append(f"**LEI:** {data.lei_code}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        sections.append(
            f"This report presents the EBA Pillar 3 ESG disclosures (Templates 6-10) "
            f"for **{data.institution_name}**. The overall Green Asset Ratio (GAR) is "
            f"**{data.total_gar_pct:.2f}%** and the Banking Book Taxonomy Alignment "
            f"Ratio (BTAR) is **{data.total_btar_pct:.2f}%** for the reporting period "
            f"{data.reporting_period}."
        )
        sections.append(f"")

        # Templates
        if self.config.include_template_6 and data.gar_stock_rows:
            sections.append(self._render_template_6(data))

        if self.config.include_template_7 and data.gar_sector_rows:
            sections.append(self._render_template_7(data))

        if self.config.include_template_8 and data.btar_rows:
            sections.append(self._render_template_8(data))

        if self.config.include_template_9 and data.gar_flow_rows:
            sections.append(self._render_template_9(data))

        if self.config.include_template_10:
            sections.append(self._render_template_10(data))

        # Notes
        if data.notes:
            sections.append(f"## Notes")
            sections.append(f"")
            for idx, note in enumerate(data.notes, 1):
                sections.append(f"{idx}. {note}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack*"
        )

        return "\n".join(sections)

    def _render_full_html(self, data: ReportData) -> str:
        """Render the full EBA report in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EBA Pillar 3 GAR Report - {data.institution_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #2980b9; color: white; text-align: center; }}
        td:first-child {{ text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #2980b9; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>EBA Pillar 3 ESG Disclosure - GAR Report</h1>
    <div class="summary">
        <p><strong>Institution:</strong> {data.institution_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>GAR:</strong> <span class="metric">{data.total_gar_pct:.2f}%</span></p>
        <p><strong>BTAR:</strong> <span class="metric">{data.total_btar_pct:.2f}%</span></p>
    </div>

    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "eba_pillar3_gar",
            "institution_name": data.institution_name,
            "lei_code": data.lei_code,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "summary": {
                "total_on_balance_eur": data.total_on_balance_eur,
                "total_gar_pct": data.total_gar_pct,
                "total_btar_pct": data.total_btar_pct,
            },
            "template_6_gar_stock": [r.dict() for r in data.gar_stock_rows],
            "template_7_gar_sector": [r.dict() for r in data.gar_sector_rows],
            "template_8_btar": [r.dict() for r in data.btar_rows],
            "template_9_gar_flow": [r.dict() for r in data.gar_flow_rows],
            "template_10_mitigating": [a.dict() for a in data.mitigating_actions],
            "notes": data.notes,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "EBAPillar3GARReportTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
