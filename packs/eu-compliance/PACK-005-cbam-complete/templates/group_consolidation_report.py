"""
GroupConsolidationReport - CBAM group-level consolidation template.

This module implements the group consolidation report for PACK-005 CBAM Complete.
It generates reports covering multi-entity group structures, per-entity breakdowns,
consolidated obligations, cost allocation across subsidiaries, member state summaries,
de minimis analysis, and financial guarantee requirements.

Example:
    >>> template = GroupConsolidationReport()
    >>> data = GroupConsolidationData(
    ...     group_overview=GroupOverview(group_name="Acme Holdings", ...),
    ...     entities=[EntityDetail(entity_name="Acme DE", ...)],
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class GroupOverview(BaseModel):
    """Group-level overview information."""

    group_name: str = Field("", description="Group name")
    parent_entity: str = Field("", description="Parent entity name")
    parent_eori: str = Field("", description="Parent entity EORI number")
    number_of_subsidiaries: int = Field(0, ge=0, description="Total subsidiary count")
    total_import_volume_tonnes: float = Field(0.0, ge=0.0, description="Total imports in tonnes")
    total_import_value_eur: float = Field(0.0, ge=0.0, description="Total import value")
    reporting_period: str = Field("", description="Reporting period label")
    consolidation_method: str = Field("financial_control", description="Consolidation method used")


class EntityDetail(BaseModel):
    """Per-entity detail record."""

    entity_name: str = Field("", description="Entity name")
    eori_number: str = Field("", description="EORI number")
    member_state: str = Field("", description="EU member state code")
    parent_entity: str = Field("", description="Parent entity name")
    tier: int = Field(1, ge=1, description="Hierarchy tier (1 = parent)")
    import_volume_tonnes: float = Field(0.0, ge=0.0, description="Import volume in tonnes")
    embedded_emissions_tco2e: float = Field(0.0, ge=0.0, description="Embedded emissions tCO2e")
    certificate_obligation: int = Field(0, ge=0, description="Certificate obligation count")
    estimated_cost_eur: float = Field(0.0, ge=0.0, description="Estimated CBAM cost")


class ConsolidatedObligation(BaseModel):
    """Group-level consolidated obligation."""

    gross_obligation: int = Field(0, ge=0, description="Gross certificate obligation")
    netting_adjustment: int = Field(0, description="Netting adjustment")
    de_minimis_adjustment: int = Field(0, description="De minimis threshold adjustment")
    net_obligation: int = Field(0, ge=0, description="Net certificate obligation")
    total_embedded_emissions_tco2e: float = Field(0.0, ge=0.0, description="Total embedded emissions")
    free_allocation_deduction: int = Field(0, ge=0, description="Free allocation deduction")
    carbon_price_deduction: int = Field(0, ge=0, description="Carbon price paid deduction")


class CostAllocation(BaseModel):
    """Cost allocation to entity."""

    entity_name: str = Field("", description="Entity name")
    allocation_method: str = Field("proportional", description="Allocation method")
    allocation_basis: str = Field("emissions", description="Allocation basis")
    allocated_cost_eur: float = Field(0.0, ge=0.0, description="Allocated CBAM cost")
    share_pct: float = Field(0.0, ge=0.0, le=100.0, description="Share percentage")


class MemberStateSummary(BaseModel):
    """Per-member-state summary."""

    member_state: str = Field("", description="EU member state code")
    member_state_name: str = Field("", description="Full state name")
    entity_count: int = Field(0, ge=0, description="Entities in this state")
    import_volume_tonnes: float = Field(0.0, ge=0.0, description="Import volume in tonnes")
    certificate_obligation: int = Field(0, ge=0, description="Certificate obligation")
    nca_name: str = Field("", description="National Competent Authority name")
    nca_coordination_notes: str = Field("", description="NCA coordination notes")


class DeMinimisEntity(BaseModel):
    """De minimis assessment per entity."""

    entity_name: str = Field("", description="Entity name")
    import_volume_tonnes: float = Field(0.0, ge=0.0, description="Import volume in tonnes")
    threshold_tonnes: float = Field(50.0, ge=0.0, description="De minimis threshold")
    below_threshold: bool = Field(False, description="Whether below threshold")
    exempt: bool = Field(False, description="Whether exempt from CBAM")


class FinancialGuarantee(BaseModel):
    """Financial guarantee requirements."""

    required_amount_eur: float = Field(0.0, ge=0.0, description="Required guarantee amount")
    current_guarantee_eur: float = Field(0.0, ge=0.0, description="Current guarantee amount")
    guarantee_provider: str = Field("", description="Guarantee provider")
    guarantee_type: str = Field("", description="Type of guarantee")
    valid_until: str = Field("", description="Validity date")
    status: str = Field("ADEQUATE", description="ADEQUATE or INSUFFICIENT")
    shortfall_eur: float = Field(0.0, ge=0.0, description="Shortfall if insufficient")


class GroupConsolidationData(BaseModel):
    """Complete input data for group consolidation report."""

    group_overview: GroupOverview = Field(default_factory=GroupOverview)
    entities: List[EntityDetail] = Field(default_factory=list)
    consolidated_obligation: ConsolidatedObligation = Field(
        default_factory=ConsolidatedObligation
    )
    cost_allocations: List[CostAllocation] = Field(default_factory=list)
    member_state_summaries: List[MemberStateSummary] = Field(default_factory=list)
    de_minimis_entities: List[DeMinimisEntity] = Field(default_factory=list)
    financial_guarantee: FinancialGuarantee = Field(default_factory=FinancialGuarantee)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class GroupConsolidationReport:
    """
    CBAM group consolidation report template.

    Generates multi-entity group consolidation reports with entity hierarchies,
    per-entity breakdowns, consolidated obligations, cost allocation,
    member state summaries, de minimis analysis, and financial guarantees.

    Attributes:
        config: Optional configuration dictionary.
        pack_id: Pack identifier (PACK-005).
        template_name: Template name for metadata.
        version: Template version.

    Example:
        >>> template = GroupConsolidationReport()
        >>> md = template.render_markdown(data)
        >>> assert "Group Overview" in md
    """

    PACK_ID = "PACK-005"
    TEMPLATE_NAME = "group_consolidation_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize GroupConsolidationReport.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - currency (str): Currency code (default: EUR).
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the group consolidation report as Markdown.

        Args:
            data: Report data dictionary matching GroupConsolidationData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header())
        sections.append(self._md_group_overview(data))
        sections.append(self._md_entity_hierarchy(data))
        sections.append(self._md_per_entity_breakdown(data))
        sections.append(self._md_consolidated_obligation(data))
        sections.append(self._md_cost_allocation(data))
        sections.append(self._md_member_state_summary(data))
        sections.append(self._md_de_minimis_analysis(data))
        sections.append(self._md_financial_guarantee(data))

        content = "\n\n".join(sections)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the group consolidation report as self-contained HTML.

        Args:
            data: Report data dictionary matching GroupConsolidationData schema.

        Returns:
            Complete HTML document string with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._html_group_overview(data))
        sections.append(self._html_entity_hierarchy(data))
        sections.append(self._html_per_entity_breakdown(data))
        sections.append(self._html_consolidated_obligation(data))
        sections.append(self._html_cost_allocation(data))
        sections.append(self._html_member_state_summary(data))
        sections.append(self._html_de_minimis_analysis(data))
        sections.append(self._html_financial_guarantee(data))

        body = "\n".join(sections)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="CBAM Group Consolidation Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the group consolidation report as structured JSON.

        Args:
            data: Report data dictionary matching GroupConsolidationData schema.

        Returns:
            Dictionary with all report sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_group_consolidation",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "group_overview": self._json_group_overview(data),
            "entities": self._json_entities(data),
            "consolidated_obligation": self._json_consolidated_obligation(data),
            "cost_allocations": self._json_cost_allocation(data),
            "member_state_summaries": self._json_member_state_summary(data),
            "de_minimis_analysis": self._json_de_minimis(data),
            "financial_guarantee": self._json_financial_guarantee(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown report header."""
        return (
            "# CBAM Group Consolidation Report\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_group_overview(self, data: Dict[str, Any]) -> str:
        """Build Markdown group overview section."""
        go = data.get("group_overview", {})
        cur = self._currency()
        return (
            "## 1. Group Overview\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Group Name | {go.get('group_name', 'N/A')} |\n"
            f"| Parent Entity | {go.get('parent_entity', 'N/A')} |\n"
            f"| Parent EORI | {go.get('parent_eori', 'N/A')} |\n"
            f"| Number of Subsidiaries | {go.get('number_of_subsidiaries', 0)} |\n"
            f"| Total Import Volume | {self._fmt_num(go.get('total_import_volume_tonnes', 0.0))} tonnes |\n"
            f"| Total Import Value | {self._fmt_cur(go.get('total_import_value_eur', 0.0), cur)} |\n"
            f"| Reporting Period | {go.get('reporting_period', 'N/A')} |\n"
            f"| Consolidation Method | {go.get('consolidation_method', 'financial_control')} |"
        )

    def _md_entity_hierarchy(self, data: Dict[str, Any]) -> str:
        """Build Markdown entity hierarchy section as a tree visualization."""
        entities = data.get("entities", [])
        go = data.get("group_overview", {})

        lines: List[str] = [
            "## 2. Entity Hierarchy\n\n"
            "```"
        ]

        # Build tree by tier
        parent_name = go.get("group_name", "Group")
        parent_eori = go.get("parent_eori", "")
        lines.append(f"{parent_name} [{parent_eori}]")

        sorted_entities = sorted(entities, key=lambda e: (e.get("tier", 1), e.get("entity_name", "")))

        for entity in sorted_entities:
            tier = entity.get("tier", 1)
            name = entity.get("entity_name", "")
            eori = entity.get("eori_number", "")
            ms = entity.get("member_state", "")
            indent = "  " * tier
            connector = "|-- " if tier <= 2 else "|   " * (tier - 2) + "|-- "
            lines.append(f"{indent}{connector}{name} [{eori}] ({ms})")

        lines.append("```")
        return "\n".join(lines)

    def _md_per_entity_breakdown(self, data: Dict[str, Any]) -> str:
        """Build Markdown per-entity breakdown section."""
        entities = data.get("entities", [])
        cur = self._currency()

        header = (
            "## 3. Per-Entity Breakdown\n\n"
            "| Entity | EORI | Member State | Import (t) | Emissions (tCO2e) | Certificates | Cost |\n"
            "|--------|------|-------------|------------|-------------------|--------------|------|\n"
        )

        rows: List[str] = []
        for e in entities:
            rows.append(
                f"| {e.get('entity_name', '')} | "
                f"{e.get('eori_number', '')} | "
                f"{e.get('member_state', '')} | "
                f"{self._fmt_num(e.get('import_volume_tonnes', 0.0))} | "
                f"{self._fmt_num(e.get('embedded_emissions_tco2e', 0.0))} | "
                f"{self._fmt_int(e.get('certificate_obligation', 0))} | "
                f"{self._fmt_cur(e.get('estimated_cost_eur', 0.0), cur)} |"
            )

        if not rows:
            return header + "| *No entities* | | | | | | |"

        return header + "\n".join(rows)

    def _md_consolidated_obligation(self, data: Dict[str, Any]) -> str:
        """Build Markdown consolidated obligation section."""
        co = data.get("consolidated_obligation", {})

        return (
            "## 4. Consolidated Obligation\n\n"
            "```\n"
            f"Gross obligation:              {self._fmt_int(co.get('gross_obligation', 0)):>12}\n"
            f"(-) Free allocation:           {'-' + self._fmt_int(co.get('free_allocation_deduction', 0)):>12}\n"
            f"(-) Carbon price deduction:    {'-' + self._fmt_int(co.get('carbon_price_deduction', 0)):>12}\n"
            f"(+/-) Netting adjustment:      {self._fmt_int(co.get('netting_adjustment', 0)):>12}\n"
            f"(-) De minimis adjustment:     {'-' + self._fmt_int(co.get('de_minimis_adjustment', 0)):>12}\n"
            f"                               {'=' * 12}\n"
            f"Net obligation:                {self._fmt_int(co.get('net_obligation', 0)):>12}\n"
            f"\n"
            f"Total embedded emissions:      {self._fmt_num(co.get('total_embedded_emissions_tco2e', 0.0)):>12} tCO2e\n"
            "```"
        )

    def _md_cost_allocation(self, data: Dict[str, Any]) -> str:
        """Build Markdown cost allocation section."""
        allocations = data.get("cost_allocations", [])
        cur = self._currency()

        header = (
            "## 5. Cost Allocation\n\n"
            "| Entity | Method | Basis | Allocated Cost | Share |\n"
            "|--------|--------|-------|----------------|-------|\n"
        )

        rows: List[str] = []
        for a in allocations:
            rows.append(
                f"| {a.get('entity_name', '')} | "
                f"{a.get('allocation_method', '')} | "
                f"{a.get('allocation_basis', '')} | "
                f"{self._fmt_cur(a.get('allocated_cost_eur', 0.0), cur)} | "
                f"{a.get('share_pct', 0.0):.1f}% |"
            )

        if not rows:
            return header + "| *No allocations* | | | | |"

        return header + "\n".join(rows)

    def _md_member_state_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown member state summary section."""
        states = data.get("member_state_summaries", [])

        header = (
            "## 6. Member State Summary\n\n"
            "| State | Name | Entities | Import (t) | Certificates | NCA | Notes |\n"
            "|-------|------|----------|------------|--------------|-----|-------|\n"
        )

        rows: List[str] = []
        for s in states:
            notes = s.get("nca_coordination_notes", "") or "-"
            rows.append(
                f"| {s.get('member_state', '')} | "
                f"{s.get('member_state_name', '')} | "
                f"{s.get('entity_count', 0)} | "
                f"{self._fmt_num(s.get('import_volume_tonnes', 0.0))} | "
                f"{self._fmt_int(s.get('certificate_obligation', 0))} | "
                f"{s.get('nca_name', '')} | "
                f"{notes} |"
            )

        if not rows:
            return header + "| *No member states* | | | | | | |"

        return header + "\n".join(rows)

    def _md_de_minimis_analysis(self, data: Dict[str, Any]) -> str:
        """Build Markdown de minimis analysis section."""
        entities = data.get("de_minimis_entities", [])

        header = (
            "## 7. De Minimis Analysis\n\n"
            "Per CBAM Regulation: entities importing <50 tonnes per CN code "
            "may be exempt from reporting obligations.\n\n"
            "| Entity | Import (t) | Threshold (t) | Below Threshold | Exempt |\n"
            "|--------|------------|---------------|-----------------|--------|\n"
        )

        rows: List[str] = []
        for e in entities:
            below = "YES" if e.get("below_threshold", False) else "NO"
            exempt = "EXEMPT" if e.get("exempt", False) else "SUBJECT"
            rows.append(
                f"| {e.get('entity_name', '')} | "
                f"{self._fmt_num(e.get('import_volume_tonnes', 0.0))} | "
                f"{self._fmt_num(e.get('threshold_tonnes', 50.0))} | "
                f"{below} | "
                f"{exempt} |"
            )

        if not rows:
            return header + "| *No entities assessed* | | | | |"

        # Group-level summary
        total_import = sum(e.get("import_volume_tonnes", 0.0) for e in entities)
        exempt_count = sum(1 for e in entities if e.get("exempt", False))
        subject_count = len(entities) - exempt_count

        summary = (
            f"\n\n**Group-Level Summary:** "
            f"Total imports: {self._fmt_num(total_import)} tonnes | "
            f"Exempt entities: {exempt_count} | "
            f"Subject entities: {subject_count}"
        )

        return header + "\n".join(rows) + summary

    def _md_financial_guarantee(self, data: Dict[str, Any]) -> str:
        """Build Markdown financial guarantee section."""
        fg = data.get("financial_guarantee", {})
        cur = self._currency()
        status = fg.get("status", "ADEQUATE")
        shortfall = fg.get("shortfall_eur", 0.0)

        section = (
            "## 8. Financial Guarantee\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Required Amount | {self._fmt_cur(fg.get('required_amount_eur', 0.0), cur)} |\n"
            f"| Current Guarantee | {self._fmt_cur(fg.get('current_guarantee_eur', 0.0), cur)} |\n"
            f"| Guarantee Provider | {fg.get('guarantee_provider', 'N/A')} |\n"
            f"| Guarantee Type | {fg.get('guarantee_type', 'N/A')} |\n"
            f"| Valid Until | {fg.get('valid_until', 'N/A')} |\n"
            f"| **Status** | **{status}** |"
        )

        if status == "INSUFFICIENT" and shortfall > 0:
            section += (
                f"\n| Shortfall | **{self._fmt_cur(shortfall, cur)}** |\n\n"
                f"> **ACTION REQUIRED:** Increase financial guarantee by "
                f"{self._fmt_cur(shortfall, cur)} to meet CBAM requirements."
            )

        return section

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Pack: {self.PACK_ID}*\n\n"
            f"*Provenance Hash: `{provenance_hash}`*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML report header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Group Consolidation Report</h1>'
            f'<div class="meta-item">Pack: {self.PACK_ID} | '
            f'Template: {self.TEMPLATE_NAME} | Version: {self.VERSION}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_group_overview(self, data: Dict[str, Any]) -> str:
        """Build HTML group overview section."""
        go = data.get("group_overview", {})
        cur = self._currency()

        fields = [
            ("Group Name", go.get("group_name", "N/A")),
            ("Parent Entity", go.get("parent_entity", "N/A")),
            ("Parent EORI", go.get("parent_eori", "N/A")),
            ("Subsidiaries", str(go.get("number_of_subsidiaries", 0))),
            ("Total Import Volume", f"{self._fmt_num(go.get('total_import_volume_tonnes', 0.0))} tonnes"),
            ("Total Import Value", self._fmt_cur(go.get("total_import_value_eur", 0.0), cur)),
            ("Reporting Period", go.get("reporting_period", "N/A")),
            ("Consolidation Method", go.get("consolidation_method", "financial_control")),
        ]

        rows = "".join(
            f'<tr><td><strong>{label}</strong></td><td>{val}</td></tr>'
            for label, val in fields
        )

        return (
            '<div class="section"><h2>1. Group Overview</h2>'
            f'<table><tbody>{rows}</tbody></table></div>'
        )

    def _html_entity_hierarchy(self, data: Dict[str, Any]) -> str:
        """Build HTML entity hierarchy section."""
        entities = data.get("entities", [])
        go = data.get("group_overview", {})

        parent_name = go.get("group_name", "Group")
        parent_eori = go.get("parent_eori", "")

        tree_html = f'<div class="tree-node tier-0">{parent_name} [{parent_eori}]</div>'

        sorted_entities = sorted(
            entities,
            key=lambda e: (e.get("tier", 1), e.get("entity_name", ""))
        )

        for entity in sorted_entities:
            tier = entity.get("tier", 1)
            name = entity.get("entity_name", "")
            eori = entity.get("eori_number", "")
            ms = entity.get("member_state", "")
            margin = tier * 24
            tree_html += (
                f'<div class="tree-node" style="margin-left:{margin}px">'
                f'<span style="color:#7f8c8d">|--</span> '
                f'{name} <span style="color:#95a5a6">[{eori}]</span> '
                f'<span style="color:#3498db">({ms})</span></div>'
            )

        return (
            '<div class="section"><h2>2. Entity Hierarchy</h2>'
            f'<div style="font-family:monospace;font-size:14px;'
            f'background:#f8f9fa;padding:16px;border-radius:8px">'
            f'{tree_html}</div></div>'
        )

    def _html_per_entity_breakdown(self, data: Dict[str, Any]) -> str:
        """Build HTML per-entity breakdown section."""
        entities = data.get("entities", [])
        cur = self._currency()

        rows_html = ""
        for e in entities:
            rows_html += (
                f'<tr><td>{e.get("entity_name", "")}</td>'
                f'<td>{e.get("eori_number", "")}</td>'
                f'<td>{e.get("member_state", "")}</td>'
                f'<td class="num">{self._fmt_num(e.get("import_volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_num(e.get("embedded_emissions_tco2e", 0.0))}</td>'
                f'<td class="num">{self._fmt_int(e.get("certificate_obligation", 0))}</td>'
                f'<td class="num">{self._fmt_cur(e.get("estimated_cost_eur", 0.0), cur)}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No entities</em></td></tr>'

        return (
            '<div class="section"><h2>3. Per-Entity Breakdown</h2>'
            '<table><thead><tr>'
            '<th>Entity</th><th>EORI</th><th>State</th>'
            '<th>Import (t)</th><th>Emissions (tCO2e)</th>'
            '<th>Certificates</th><th>Cost</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_consolidated_obligation(self, data: Dict[str, Any]) -> str:
        """Build HTML consolidated obligation section."""
        co = data.get("consolidated_obligation", {})
        gross = co.get("gross_obligation", 0)
        free_alloc = co.get("free_allocation_deduction", 0)
        carbon_ded = co.get("carbon_price_deduction", 0)
        netting = co.get("netting_adjustment", 0)
        de_min = co.get("de_minimis_adjustment", 0)
        net = co.get("net_obligation", 0)
        max_val = max(gross, 1)

        items = [
            ("Gross obligation", gross, "positive"),
            ("(-) Free allocation", free_alloc, "negative"),
            ("(-) Carbon price deduction", carbon_ded, "negative"),
            ("(+/-) Netting adjustment", abs(netting), "neutral"),
            ("(-) De minimis adjustment", de_min, "negative"),
        ]

        bars = ""
        for label, val, bar_class in items:
            pct = val / max_val * 100 if max_val > 0 else 0
            bars += (
                f'<div style="display:flex;align-items:center;margin-bottom:8px">'
                f'<div style="width:260px;font-size:14px">{label}</div>'
                f'<div class="waterfall-bar {bar_class}" '
                f'style="width:{max(pct, 5):.0f}%">{self._fmt_int(val)}</div></div>'
            )

        net_pct = net / max_val * 100 if max_val > 0 else 0
        bars += (
            f'<div style="display:flex;align-items:center;margin-top:8px;'
            f'border-top:2px solid #1a5276;padding-top:8px">'
            f'<div style="width:260px;font-size:14px"><strong>Net obligation</strong></div>'
            f'<div class="waterfall-bar net" '
            f'style="width:{max(net_pct, 5):.0f}%"><strong>{self._fmt_int(net)}</strong></div></div>'
        )

        emissions = co.get("total_embedded_emissions_tco2e", 0.0)

        return (
            '<div class="section"><h2>4. Consolidated Obligation</h2>'
            f'<div class="waterfall">{bars}</div>'
            f'<p style="margin-top:12px;color:#7f8c8d">Total embedded emissions: '
            f'<strong>{self._fmt_num(emissions)} tCO2e</strong></p></div>'
        )

    def _html_cost_allocation(self, data: Dict[str, Any]) -> str:
        """Build HTML cost allocation section."""
        allocations = data.get("cost_allocations", [])
        cur = self._currency()

        rows_html = ""
        for a in allocations:
            share = a.get("share_pct", 0.0)
            rows_html += (
                f'<tr><td>{a.get("entity_name", "")}</td>'
                f'<td>{a.get("allocation_method", "")}</td>'
                f'<td>{a.get("allocation_basis", "")}</td>'
                f'<td class="num">{self._fmt_cur(a.get("allocated_cost_eur", 0.0), cur)}</td>'
                f'<td class="num">{share:.1f}%</td>'
                f'<td>'
                f'<div class="progress-bar">'
                f'<div class="progress-fill" style="width:{min(share, 100):.0f}%;'
                f'background:#3498db"></div></div></td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No allocations</em></td></tr>'

        return (
            '<div class="section"><h2>5. Cost Allocation</h2>'
            '<table><thead><tr>'
            '<th>Entity</th><th>Method</th><th>Basis</th>'
            '<th>Allocated Cost</th><th>Share</th><th>Distribution</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_member_state_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML member state summary section."""
        states = data.get("member_state_summaries", [])

        rows_html = ""
        for s in states:
            notes = s.get("nca_coordination_notes", "") or "-"
            rows_html += (
                f'<tr><td><strong>{s.get("member_state", "")}</strong></td>'
                f'<td>{s.get("member_state_name", "")}</td>'
                f'<td class="num">{s.get("entity_count", 0)}</td>'
                f'<td class="num">{self._fmt_num(s.get("import_volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_int(s.get("certificate_obligation", 0))}</td>'
                f'<td>{s.get("nca_name", "")}</td>'
                f'<td>{notes}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No member states</em></td></tr>'

        return (
            '<div class="section"><h2>6. Member State Summary</h2>'
            '<table><thead><tr>'
            '<th>State</th><th>Name</th><th>Entities</th>'
            '<th>Import (t)</th><th>Certificates</th>'
            '<th>NCA</th><th>Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_de_minimis_analysis(self, data: Dict[str, Any]) -> str:
        """Build HTML de minimis analysis section."""
        entities = data.get("de_minimis_entities", [])

        rows_html = ""
        for e in entities:
            below = e.get("below_threshold", False)
            exempt = e.get("exempt", False)
            color = "#2ecc71" if exempt else "#e74c3c"
            rows_html += (
                f'<tr><td>{e.get("entity_name", "")}</td>'
                f'<td class="num">{self._fmt_num(e.get("import_volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_num(e.get("threshold_tonnes", 50.0))}</td>'
                f'<td style="color:{color};font-weight:bold">'
                f'{"YES" if below else "NO"}</td>'
                f'<td style="color:{color};font-weight:bold">'
                f'{"EXEMPT" if exempt else "SUBJECT"}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="5"><em>No entities assessed</em></td></tr>'

        exempt_count = sum(1 for e in entities if e.get("exempt", False))
        subject_count = len(entities) - exempt_count

        return (
            '<div class="section"><h2>7. De Minimis Analysis</h2>'
            '<p>Per CBAM Regulation: entities importing &lt;50 tonnes per CN code '
            'may be exempt from reporting obligations.</p>'
            '<table><thead><tr>'
            '<th>Entity</th><th>Import (t)</th><th>Threshold (t)</th>'
            '<th>Below Threshold</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
            f'<div style="margin-top:12px;padding:12px;background:#f8f9fa;border-radius:4px">'
            f'<strong>Group Summary:</strong> '
            f'Exempt: {exempt_count} | Subject: {subject_count} | '
            f'Total: {len(entities)}</div></div>'
        )

    def _html_financial_guarantee(self, data: Dict[str, Any]) -> str:
        """Build HTML financial guarantee section."""
        fg = data.get("financial_guarantee", {})
        cur = self._currency()
        status = fg.get("status", "ADEQUATE")
        shortfall = fg.get("shortfall_eur", 0.0)
        color = "#2ecc71" if status == "ADEQUATE" else "#e74c3c"

        fields = [
            ("Required Amount", self._fmt_cur(fg.get("required_amount_eur", 0.0), cur)),
            ("Current Guarantee", self._fmt_cur(fg.get("current_guarantee_eur", 0.0), cur)),
            ("Guarantee Provider", fg.get("guarantee_provider", "N/A")),
            ("Guarantee Type", fg.get("guarantee_type", "N/A")),
            ("Valid Until", fg.get("valid_until", "N/A")),
        ]

        rows = "".join(
            f'<tr><td><strong>{label}</strong></td><td>{val}</td></tr>'
            for label, val in fields
        )
        rows += (
            f'<tr><td><strong>Status</strong></td>'
            f'<td style="color:{color};font-weight:bold;font-size:16px">{status}</td></tr>'
        )

        alert = ""
        if status == "INSUFFICIENT" and shortfall > 0:
            alert = (
                f'<div style="margin-top:12px;padding:12px;background:#fdf2f2;'
                f'border-left:4px solid #e74c3c;border-radius:4px">'
                f'<strong>ACTION REQUIRED:</strong> Increase financial guarantee by '
                f'{self._fmt_cur(shortfall, cur)} to meet CBAM requirements.</div>'
            )

        return (
            '<div class="section"><h2>8. Financial Guarantee</h2>'
            f'<table><tbody>{rows}</tbody></table>'
            f'{alert}</div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_group_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON group overview."""
        go = data.get("group_overview", {})
        return {
            "group_name": go.get("group_name", ""),
            "parent_entity": go.get("parent_entity", ""),
            "parent_eori": go.get("parent_eori", ""),
            "number_of_subsidiaries": go.get("number_of_subsidiaries", 0),
            "total_import_volume_tonnes": round(go.get("total_import_volume_tonnes", 0.0), 2),
            "total_import_value_eur": round(go.get("total_import_value_eur", 0.0), 2),
            "reporting_period": go.get("reporting_period", ""),
            "consolidation_method": go.get("consolidation_method", "financial_control"),
        }

    def _json_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON entity list."""
        return [
            {
                "entity_name": e.get("entity_name", ""),
                "eori_number": e.get("eori_number", ""),
                "member_state": e.get("member_state", ""),
                "parent_entity": e.get("parent_entity", ""),
                "tier": e.get("tier", 1),
                "import_volume_tonnes": round(e.get("import_volume_tonnes", 0.0), 2),
                "embedded_emissions_tco2e": round(e.get("embedded_emissions_tco2e", 0.0), 2),
                "certificate_obligation": e.get("certificate_obligation", 0),
                "estimated_cost_eur": round(e.get("estimated_cost_eur", 0.0), 2),
            }
            for e in data.get("entities", [])
        ]

    def _json_consolidated_obligation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON consolidated obligation."""
        co = data.get("consolidated_obligation", {})
        return {
            "gross_obligation": co.get("gross_obligation", 0),
            "free_allocation_deduction": co.get("free_allocation_deduction", 0),
            "carbon_price_deduction": co.get("carbon_price_deduction", 0),
            "netting_adjustment": co.get("netting_adjustment", 0),
            "de_minimis_adjustment": co.get("de_minimis_adjustment", 0),
            "net_obligation": co.get("net_obligation", 0),
            "total_embedded_emissions_tco2e": round(
                co.get("total_embedded_emissions_tco2e", 0.0), 2
            ),
        }

    def _json_cost_allocation(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON cost allocation."""
        return [
            {
                "entity_name": a.get("entity_name", ""),
                "allocation_method": a.get("allocation_method", ""),
                "allocation_basis": a.get("allocation_basis", ""),
                "allocated_cost_eur": round(a.get("allocated_cost_eur", 0.0), 2),
                "share_pct": round(a.get("share_pct", 0.0), 2),
            }
            for a in data.get("cost_allocations", [])
        ]

    def _json_member_state_summary(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON member state summary."""
        return [
            {
                "member_state": s.get("member_state", ""),
                "member_state_name": s.get("member_state_name", ""),
                "entity_count": s.get("entity_count", 0),
                "import_volume_tonnes": round(s.get("import_volume_tonnes", 0.0), 2),
                "certificate_obligation": s.get("certificate_obligation", 0),
                "nca_name": s.get("nca_name", ""),
                "nca_coordination_notes": s.get("nca_coordination_notes", ""),
            }
            for s in data.get("member_state_summaries", [])
        ]

    def _json_de_minimis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON de minimis analysis."""
        entities = data.get("de_minimis_entities", [])
        exempt_count = sum(1 for e in entities if e.get("exempt", False))
        return {
            "entities": [
                {
                    "entity_name": e.get("entity_name", ""),
                    "import_volume_tonnes": round(e.get("import_volume_tonnes", 0.0), 2),
                    "threshold_tonnes": round(e.get("threshold_tonnes", 50.0), 2),
                    "below_threshold": e.get("below_threshold", False),
                    "exempt": e.get("exempt", False),
                }
                for e in entities
            ],
            "summary": {
                "total_entities": len(entities),
                "exempt_count": exempt_count,
                "subject_count": len(entities) - exempt_count,
            },
        }

    def _json_financial_guarantee(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON financial guarantee."""
        fg = data.get("financial_guarantee", {})
        return {
            "required_amount_eur": round(fg.get("required_amount_eur", 0.0), 2),
            "current_guarantee_eur": round(fg.get("current_guarantee_eur", 0.0), 2),
            "guarantee_provider": fg.get("guarantee_provider", ""),
            "guarantee_type": fg.get("guarantee_type", ""),
            "valid_until": fg.get("valid_until", ""),
            "status": fg.get("status", "ADEQUATE"),
            "shortfall_eur": round(fg.get("shortfall_eur", 0.0), 2),
        }

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _compute_provenance_hash(self, content: str) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _currency(self) -> str:
        """Get configured currency code."""
        return self.config.get("currency", "EUR")

    def _fmt_int(self, value: Union[int, float, None]) -> str:
        """Format integer with thousand separators."""
        if value is None:
            return "0"
        return f"{int(value):,}"

    def _fmt_num(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format number with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _fmt_cur(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format currency value."""
        return f"{currency} {value:,.2f}"

    def _fmt_date(self, dt: Union[datetime, str]) -> str:
        """Format datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if dt else ""
        return dt.strftime("%Y-%m-%d")

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = self._get_css()
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f'Pack: {self.PACK_ID} | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )

    def _get_css(self) -> str:
        """Return inline CSS for HTML reports."""
        return (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#fff;padding:20px;border-radius:8px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:28px;font-weight:700;color:#1a5276}"
            ".kpi-unit{font-size:12px;color:#95a5a6;margin-top:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            ".section h3{margin:16px 0 8px 0;font-size:15px;color:#2c3e50}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".tree-node{padding:4px 0;font-size:14px}"
            ".waterfall-bar{padding:6px 12px;border-radius:4px;color:#fff;"
            "font-size:13px;min-width:60px;text-align:right}"
            ".waterfall-bar.positive{background:#2ecc71}"
            ".waterfall-bar.negative{background:#e74c3c}"
            ".waterfall-bar.neutral{background:#f39c12}"
            ".waterfall-bar.net{background:#1a5276}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
