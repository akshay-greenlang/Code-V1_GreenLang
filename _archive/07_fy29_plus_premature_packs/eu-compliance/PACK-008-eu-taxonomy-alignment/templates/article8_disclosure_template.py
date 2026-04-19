"""
Article 8 Disclosure Template - PACK-008 EU Taxonomy Alignment Pack

This module generates mandatory disclosure tables per Delegated Regulation (EU) 2021/2178
for Article 8 of the EU Taxonomy Regulation. It produces Turnover, CapEx, and OpEx
disclosure tables with the required column structure, plus nuclear/gas supplementary
disclosure templates per Complementary DA (EU) 2022/1214.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from article8_disclosure_template import Article8DisclosureTemplate, ReportData
    >>> data = ReportData(
    ...     organization_name="Acme Manufacturing GmbH",
    ...     report_date="2026-03-15",
    ...     reporting_period="FY 2025"
    ... )
    >>> template = Article8DisclosureTemplate()
    >>> report = template.generate_full_disclosure(data, format="markdown")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """Configuration for Article 8 Disclosure generation."""

    include_nuclear_gas: bool = Field(
        default=False,
        description="Include nuclear/gas supplementary disclosure (DA 2022/1214)"
    )
    include_prior_year: bool = Field(
        default=False,
        description="Include prior year comparatives"
    )
    currency: str = Field(
        default="EUR",
        description="Reporting currency"
    )


class DisclosureRow(BaseModel):
    """Single row in an Article 8 disclosure table."""

    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field(..., description="NACE code(s)")
    amount_eur: float = Field(..., ge=0, description="Absolute amount in EUR")
    proportion_pct: float = Field(..., ge=0, le=100, description="Proportion of total")
    ccm_sc: bool = Field(default=False, description="SC to Climate Change Mitigation")
    cca_sc: bool = Field(default=False, description="SC to Climate Change Adaptation")
    wtr_sc: bool = Field(default=False, description="SC to Water")
    ce_sc: bool = Field(default=False, description="SC to Circular Economy")
    ppc_sc: bool = Field(default=False, description="SC to Pollution Prevention")
    bio_sc: bool = Field(default=False, description="SC to Biodiversity")
    ccm_dnsh: bool = Field(default=True, description="DNSH Climate Mitigation")
    cca_dnsh: bool = Field(default=True, description="DNSH Climate Adaptation")
    wtr_dnsh: bool = Field(default=True, description="DNSH Water")
    ce_dnsh: bool = Field(default=True, description="DNSH Circular Economy")
    ppc_dnsh: bool = Field(default=True, description="DNSH Pollution")
    bio_dnsh: bool = Field(default=True, description="DNSH Biodiversity")
    minimum_safeguards: bool = Field(default=False, description="Minimum Safeguards compliance")
    aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Taxonomy-aligned proportion")
    eligible_pct: float = Field(default=0.0, ge=0, le=100, description="Taxonomy-eligible proportion")
    category: str = Field(
        default="T",
        description="Category: T (Transitional), E (Enabling), or - (Standard)"
    )
    is_aligned: bool = Field(default=False, description="Activity is taxonomy-aligned")
    prior_year_pct: Optional[float] = Field(
        None, ge=0, le=100,
        description="Prior year aligned proportion"
    )


class NuclearGasRow(BaseModel):
    """Row for nuclear/gas supplementary disclosure."""

    activity_name: str = Field(..., description="Nuclear or gas activity name")
    nace_code: str = Field(default="D.35.11", description="NACE code")
    amount_eur: float = Field(..., ge=0, description="Absolute amount in EUR")
    proportion_pct: float = Field(..., ge=0, le=100, description="Proportion of total")
    ccm_sc: bool = Field(default=False, description="SC to CCM")
    cca_sc: bool = Field(default=False, description="SC to CCA")
    minimum_safeguards: bool = Field(default=False, description="MS compliance")
    aligned_pct: float = Field(default=0.0, ge=0, le=100, description="Aligned proportion")
    is_nuclear: bool = Field(default=False, description="Nuclear activity flag")
    is_gas: bool = Field(default=False, description="Gas activity flag")


class DisclosureTable(BaseModel):
    """A single Article 8 disclosure table (Turnover, CapEx, or OpEx)."""

    table_name: str = Field(..., description="Table name (Turnover, CapEx, OpEx)")
    total_amount_eur: float = Field(..., ge=0, description="Total denominator amount")
    aligned_rows: List[DisclosureRow] = Field(
        default_factory=list,
        description="Taxonomy-aligned activities"
    )
    eligible_not_aligned_rows: List[DisclosureRow] = Field(
        default_factory=list,
        description="Eligible but not aligned activities"
    )
    non_eligible_amount_eur: float = Field(
        default=0.0, ge=0,
        description="Non-eligible amount"
    )
    non_eligible_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Non-eligible proportion"
    )


class ReportData(BaseModel):
    """Data model for Article 8 Disclosure."""

    organization_name: str = Field(..., description="Organization name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    reporting_period: str = Field(..., description="Reporting period (e.g., FY 2025)")
    turnover_table: Optional[DisclosureTable] = Field(
        None, description="Table 1: Turnover disclosure"
    )
    capex_table: Optional[DisclosureTable] = Field(
        None, description="Table 2: CapEx disclosure"
    )
    opex_table: Optional[DisclosureTable] = Field(
        None, description="Table 3: OpEx disclosure"
    )
    nuclear_gas_rows: List[NuclearGasRow] = Field(
        default_factory=list,
        description="Nuclear/gas supplementary rows"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Disclosure notes"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class Article8DisclosureTemplate:
    """
    Article 8 Disclosure Template for EU Taxonomy Alignment Pack.

    Generates mandatory disclosure tables per Delegated Regulation (EU) 2021/2178:
    - Table 1: Turnover taxonomy-aligned proportions
    - Table 2: CapEx taxonomy-aligned proportions
    - Table 3: OpEx taxonomy-aligned proportions
    Plus nuclear/gas supplementary disclosures per Complementary DA (EU) 2022/1214.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = Article8DisclosureTemplate()
        >>> report = template.generate_full_disclosure(data, format="markdown")
        >>> assert "Turnover" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Article 8 Disclosure Template."""
        self.config = config or ReportConfig()

    def generate_full_disclosure(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Generate the full Article 8 disclosure report.

        Args:
            data: Report data containing all three tables
            format: Output format (markdown, html, json)

        Returns:
            Complete disclosure report

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Generating full Article 8 disclosure for {data.organization_name} "
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
        logger.info(f"Disclosure generated with hash: {content_hash}")

        return content

    def generate_turnover_table(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """Generate Table 1: Turnover disclosure only."""
        logger.info(f"Generating turnover table for {data.organization_name}")
        if data.turnover_table is None:
            return "No turnover data available."
        return self._render_single_table_markdown(data.turnover_table, data)

    def generate_capex_table(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """Generate Table 2: CapEx disclosure only."""
        logger.info(f"Generating CapEx table for {data.organization_name}")
        if data.capex_table is None:
            return "No CapEx data available."
        return self._render_single_table_markdown(data.capex_table, data)

    def generate_opex_table(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """Generate Table 3: OpEx disclosure only."""
        logger.info(f"Generating OpEx table for {data.organization_name}")
        if data.opex_table is None:
            return "No OpEx data available."
        return self._render_single_table_markdown(data.opex_table, data)

    def generate_nuclear_gas_supplement(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """Generate nuclear/gas supplementary disclosure per DA 2022/1214."""
        logger.info(f"Generating nuclear/gas supplement for {data.organization_name}")
        if not data.nuclear_gas_rows:
            return "No nuclear or gas activities to disclose."
        return self._render_nuclear_gas_markdown(data)

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Main render entry point (alias for generate_full_disclosure).

        Args:
            data: Report data
            format: Output format

        Returns:
            Full disclosure report
        """
        return self.generate_full_disclosure(data, format)

    def _render_single_table_markdown(
        self, table: DisclosureTable, data: ReportData
    ) -> str:
        """Render a single disclosure table in Markdown."""
        sections = []
        sections.append(f"### {table.table_name} Disclosure")
        sections.append(f"")
        sections.append(f"**Total {table.table_name}:** EUR {table.total_amount_eur:,.0f}")
        sections.append(f"")

        header = (
            "| Activity | NACE | Amount (EUR) | % | "
            "CCM | CCA | WTR | CE | PPC | BIO | MS | Aligned % | Eligible % | Cat |"
        )
        separator = (
            "|----------|------|------------:|---:|"
            "-----|-----|-----|----|-----|-----|----|----------:|----------:|-----|"
        )
        sections.append(header)
        sections.append(separator)

        # A. Taxonomy-aligned activities
        if table.aligned_rows:
            for row in table.aligned_rows:
                sections.append(self._format_disclosure_row(row))

        aligned_total = sum(r.amount_eur for r in table.aligned_rows)
        aligned_pct = (aligned_total / table.total_amount_eur * 100) if table.total_amount_eur > 0 else 0
        sections.append(
            f"| **A. Aligned Total** | | **{aligned_total:,.0f}** | "
            f"**{aligned_pct:.1f}** | | | | | | | | **{aligned_pct:.1f}** | | |"
        )

        # B. Eligible but not aligned
        if table.eligible_not_aligned_rows:
            for row in table.eligible_not_aligned_rows:
                sections.append(self._format_disclosure_row(row))

        elig_na_total = sum(r.amount_eur for r in table.eligible_not_aligned_rows)
        elig_na_pct = (elig_na_total / table.total_amount_eur * 100) if table.total_amount_eur > 0 else 0
        sections.append(
            f"| **B. Eligible Not Aligned** | | **{elig_na_total:,.0f}** | "
            f"**{elig_na_pct:.1f}** | | | | | | | | | **{elig_na_pct:.1f}** | |"
        )

        # C. Non-eligible
        sections.append(
            f"| **C. Non-Eligible** | | **{table.non_eligible_amount_eur:,.0f}** | "
            f"**{table.non_eligible_pct:.1f}** | | | | | | | | | | |"
        )

        # Total
        sections.append(
            f"| **Total (A+B+C)** | | **{table.total_amount_eur:,.0f}** | "
            f"**100.0** | | | | | | | | | | |"
        )
        sections.append(f"")

        return "\n".join(sections)

    def _format_disclosure_row(self, row: DisclosureRow) -> str:
        """Format a single disclosure row for Markdown output."""
        return (
            f"| {row.activity_name[:30]} | {row.nace_code} | "
            f"{row.amount_eur:,.0f} | {row.proportion_pct:.1f} | "
            f"{'Y' if row.ccm_sc else 'N'} | "
            f"{'Y' if row.cca_sc else 'N'} | "
            f"{'Y' if row.wtr_sc else 'N'} | "
            f"{'Y' if row.ce_sc else 'N'} | "
            f"{'Y' if row.ppc_sc else 'N'} | "
            f"{'Y' if row.bio_sc else 'N'} | "
            f"{'Y' if row.minimum_safeguards else 'N'} | "
            f"{row.aligned_pct:.1f} | {row.eligible_pct:.1f} | {row.category} |"
        )

    def _render_full_markdown(self, data: ReportData) -> str:
        """Render the full Article 8 disclosure in Markdown."""
        sections = []

        # Header
        sections.append(f"# Article 8 Taxonomy Disclosure")
        sections.append(f"")
        sections.append(f"**Organization:** {data.organization_name}")
        sections.append(f"**Reporting Period:** {data.reporting_period}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(
            f"**Regulation:** Delegated Regulation (EU) 2021/2178, Article 8"
        )
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Table 1: Turnover
        if data.turnover_table:
            sections.append(f"## Table 1: Proportion of Turnover from Taxonomy-Aligned Activities")
            sections.append(f"")
            sections.append(self._render_single_table_markdown(data.turnover_table, data))

        # Table 2: CapEx
        if data.capex_table:
            sections.append(f"## Table 2: Proportion of CapEx from Taxonomy-Aligned Activities")
            sections.append(f"")
            sections.append(self._render_single_table_markdown(data.capex_table, data))

        # Table 3: OpEx
        if data.opex_table:
            sections.append(f"## Table 3: Proportion of OpEx from Taxonomy-Aligned Activities")
            sections.append(f"")
            sections.append(self._render_single_table_markdown(data.opex_table, data))

        # Nuclear/Gas Supplement
        if self.config.include_nuclear_gas and data.nuclear_gas_rows:
            sections.append(f"## Nuclear and Gas Supplementary Disclosure")
            sections.append(f"")
            sections.append(
                f"*Per Complementary Delegated Act (EU) 2022/1214*"
            )
            sections.append(f"")
            sections.append(self._render_nuclear_gas_markdown(data))

        # Notes
        if data.notes:
            sections.append(f"## Disclosure Notes")
            sections.append(f"")
            for idx, note in enumerate(data.notes, 1):
                sections.append(f"{idx}. {note}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Disclosure generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack*"
        )

        return "\n".join(sections)

    def _render_nuclear_gas_markdown(self, data: ReportData) -> str:
        """Render nuclear/gas supplementary disclosure in Markdown."""
        sections = []

        nuclear_rows = [r for r in data.nuclear_gas_rows if r.is_nuclear]
        gas_rows = [r for r in data.nuclear_gas_rows if r.is_gas]

        if nuclear_rows:
            sections.append(f"### Nuclear Energy Activities")
            sections.append(f"")
            sections.append(
                f"| Activity | NACE | Amount (EUR) | % | CCM | CCA | MS | Aligned % |"
            )
            sections.append(
                f"|----------|------|------------:|---:|-----|-----|----|----------:|"
            )
            for row in nuclear_rows:
                sections.append(
                    f"| {row.activity_name} | {row.nace_code} | "
                    f"{row.amount_eur:,.0f} | {row.proportion_pct:.1f} | "
                    f"{'Y' if row.ccm_sc else 'N'} | "
                    f"{'Y' if row.cca_sc else 'N'} | "
                    f"{'Y' if row.minimum_safeguards else 'N'} | "
                    f"{row.aligned_pct:.1f} |"
                )
            sections.append(f"")

        if gas_rows:
            sections.append(f"### Fossil Gas Activities")
            sections.append(f"")
            sections.append(
                f"| Activity | NACE | Amount (EUR) | % | CCM | CCA | MS | Aligned % |"
            )
            sections.append(
                f"|----------|------|------------:|---:|-----|-----|----|----------:|"
            )
            for row in gas_rows:
                sections.append(
                    f"| {row.activity_name} | {row.nace_code} | "
                    f"{row.amount_eur:,.0f} | {row.proportion_pct:.1f} | "
                    f"{'Y' if row.ccm_sc else 'N'} | "
                    f"{'Y' if row.cca_sc else 'N'} | "
                    f"{'Y' if row.minimum_safeguards else 'N'} | "
                    f"{row.aligned_pct:.1f} |"
                )
            sections.append(f"")

        return "\n".join(sections)

    def _render_full_html(self, data: ReportData) -> str:
        """Render the full Article 8 disclosure in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Article 8 Taxonomy Disclosure - {data.organization_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #27ae60; color: white; font-size: 0.85em; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr.subtotal {{ background-color: #d5f5e3; font-weight: bold; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Article 8 Taxonomy Disclosure</h1>
    <div class="summary">
        <p><strong>Organization:</strong> {data.organization_name}</p>
        <p><strong>Reporting Period:</strong> {data.reporting_period}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Regulation:</strong> Delegated Regulation (EU) 2021/2178, Article 8</p>
    </div>
"""

        for table in [data.turnover_table, data.capex_table, data.opex_table]:
            if table:
                html += self._render_table_html(table)

        html += f"""
    <div class="footer">
        <p><em>Disclosure generated on {data.report_date} using GreenLang EU Taxonomy Alignment Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_table_html(self, table: DisclosureTable) -> str:
        """Render a single disclosure table in HTML format."""
        html = f"""
    <h2>{table.table_name} Disclosure</h2>
    <p><strong>Total {table.table_name}:</strong> EUR {table.total_amount_eur:,.0f}</p>
    <table>
        <tr>
            <th>Activity</th><th>NACE</th><th>Amount (EUR)</th><th>%</th>
            <th>CCM</th><th>CCA</th><th>WTR</th><th>CE</th><th>PPC</th><th>BIO</th>
            <th>MS</th><th>Aligned %</th><th>Eligible %</th><th>Cat</th>
        </tr>
"""
        for row in table.aligned_rows + table.eligible_not_aligned_rows:
            html += f"""        <tr>
            <td>{row.activity_name[:30]}</td><td>{row.nace_code}</td>
            <td>{row.amount_eur:,.0f}</td><td>{row.proportion_pct:.1f}</td>
            <td>{'Y' if row.ccm_sc else 'N'}</td><td>{'Y' if row.cca_sc else 'N'}</td>
            <td>{'Y' if row.wtr_sc else 'N'}</td><td>{'Y' if row.ce_sc else 'N'}</td>
            <td>{'Y' if row.ppc_sc else 'N'}</td><td>{'Y' if row.bio_sc else 'N'}</td>
            <td>{'Y' if row.minimum_safeguards else 'N'}</td>
            <td>{row.aligned_pct:.1f}</td><td>{row.eligible_pct:.1f}</td>
            <td>{row.category}</td>
        </tr>
"""
        html += """    </table>
"""
        return html

    def _render_json(self, data: ReportData) -> str:
        """Render disclosure in JSON format."""
        report_dict = {
            "report_type": "article8_disclosure",
            "regulation": "Delegated Regulation (EU) 2021/2178",
            "organization_name": data.organization_name,
            "report_date": data.report_date,
            "reporting_period": data.reporting_period,
            "turnover_table": data.turnover_table.dict() if data.turnover_table else None,
            "capex_table": data.capex_table.dict() if data.capex_table else None,
            "opex_table": data.opex_table.dict() if data.opex_table else None,
            "nuclear_gas_supplement": [r.dict() for r in data.nuclear_gas_rows],
            "notes": data.notes,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "Article8DisclosureTemplate",
                "version": "1.0.0",
                "pack": "PACK-008-eu-taxonomy-alignment",
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
