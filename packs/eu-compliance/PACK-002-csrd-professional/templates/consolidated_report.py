# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Consolidated Report Template
================================================

Multi-entity consolidated ESRS report template. Generates group-level
reports with entity registry, consolidation methodology, intercompany
eliminations, minority interest disclosures, and entity-to-group
reconciliation summaries.

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

class ConsolidationApproach(str, Enum):
    """Consolidation methodology approach."""
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    EQUITY_SHARE = "EQUITY_SHARE"


class CoverageStatus(str, Enum):
    """Coverage status for standard disclosures."""
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    GAP = "GAP"
    NOT_APPLICABLE = "N/A"


class DataQualityLevel(str, Enum):
    """Quality level for entity data."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    ESTIMATED = "ESTIMATED"


class ReconciliationStatus(str, Enum):
    """Reconciliation outcome status."""
    RECONCILED = "RECONCILED"
    VARIANCE = "VARIANCE"
    UNRECONCILED = "UNRECONCILED"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class EntitySummary(BaseModel):
    """Summary data for a single consolidated entity."""
    name: str = Field(..., description="Legal entity name")
    country: str = Field(..., description="Country of incorporation")
    ownership_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Group ownership percentage"
    )
    scope1_total: float = Field(0.0, ge=0.0, description="Scope 1 tCO2e")
    scope2_total: float = Field(0.0, ge=0.0, description="Scope 2 tCO2e")
    scope3_total: float = Field(0.0, ge=0.0, description="Scope 3 tCO2e")
    quality_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Data quality score 0-100"
    )
    consolidation_method: Optional[str] = Field(
        None, description="Specific consolidation method for this entity"
    )
    employee_count: Optional[int] = Field(None, ge=0, description="FTE count")
    revenue_eur: Optional[float] = Field(None, ge=0.0, description="Revenue in EUR")

    @property
    def total_emissions(self) -> float:
        """Total emissions across all scopes."""
        return self.scope1_total + self.scope2_total + self.scope3_total


class ConsolidatedEmissions(BaseModel):
    """Consolidated emissions data across entities."""
    scope1_total: float = Field(0.0, ge=0.0, description="Consolidated Scope 1 tCO2e")
    scope2_total: float = Field(0.0, ge=0.0, description="Consolidated Scope 2 tCO2e")
    scope3_total: float = Field(0.0, ge=0.0, description="Consolidated Scope 3 tCO2e")
    by_entity: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Emissions by entity name -> {scope1, scope2, scope3}",
    )
    eliminations_tco2e: float = Field(
        0.0, description="Intercompany eliminations in tCO2e"
    )
    adjustment_tco2e: float = Field(
        0.0, description="Consolidation adjustments in tCO2e"
    )

    @property
    def grand_total(self) -> float:
        """Grand total after eliminations and adjustments."""
        return (
            self.scope1_total
            + self.scope2_total
            + self.scope3_total
            - self.eliminations_tco2e
            + self.adjustment_tco2e
        )


class StandardDisclosure(BaseModel):
    """Disclosure status for a single ESRS standard."""
    standard_id: str = Field(..., description="ESRS standard ID")
    standard_name: str = Field(..., description="Standard name")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage %")
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Key metric values"
    )
    gaps: List[str] = Field(
        default_factory=list, description="Identified disclosure gaps"
    )
    status: CoverageStatus = Field(
        CoverageStatus.GAP, description="Overall coverage status"
    )


class EliminationEntry(BaseModel):
    """Intercompany elimination entry."""
    entity_from: str = Field(..., description="Selling/providing entity")
    entity_to: str = Field(..., description="Buying/receiving entity")
    elimination_type: str = Field(
        ..., description="Type: emissions, revenue, services"
    )
    amount_tco2e: float = Field(0.0, description="Eliminated amount in tCO2e")
    description: str = Field("", description="Description of elimination")
    methodology: Optional[str] = Field(
        None, description="Elimination methodology applied"
    )


class ReconciliationEntry(BaseModel):
    """Entity-to-group reconciliation entry."""
    metric_name: str = Field(..., description="Metric being reconciled")
    entity_sum: float = Field(0.0, description="Sum of entity values")
    consolidated_value: float = Field(0.0, description="Group consolidated value")
    difference: float = Field(0.0, description="Difference (entity - consolidated)")
    difference_pct: float = Field(0.0, description="Difference as percentage")
    status: ReconciliationStatus = Field(
        ReconciliationStatus.UNRECONCILED, description="Reconciliation status"
    )
    explanation: str = Field("", description="Explanation for variance")


class ConsolidatedReportInput(BaseModel):
    """Complete input for the consolidated ESRS report."""
    group_name: str = Field(..., description="Group / parent company name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.FINANCIAL_CONTROL,
        description="Consolidation approach used",
    )
    entities: List[EntitySummary] = Field(
        default_factory=list, description="List of consolidated entities"
    )
    consolidated_emissions: ConsolidatedEmissions = Field(
        default_factory=ConsolidatedEmissions,
        description="Consolidated emissions data",
    )
    esrs_disclosures: Dict[str, StandardDisclosure] = Field(
        default_factory=dict,
        description="Per-standard ESRS disclosure summaries",
    )
    intercompany_eliminations: List[EliminationEntry] = Field(
        default_factory=list, description="Intercompany elimination entries"
    )
    minority_interest_disclosures: List[Dict[str, Any]] = Field(
        default_factory=list, description="Minority interest disclosure items"
    )
    reconciliation_summary: List[ReconciliationEntry] = Field(
        default_factory=list, description="Entity-to-group reconciliation"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format a numeric value with thousands separator, or return 'N/A'."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format a percentage value, or return 'N/A'."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with appropriate scale."""
    return _fmt_number(value, decimals=1, suffix=" tCO2e")


def _status_badge(status: CoverageStatus) -> str:
    """Return text badge for coverage status."""
    return f"[{status.value}]"


def _recon_badge(status: ReconciliationStatus) -> str:
    """Return text badge for reconciliation status."""
    return f"[{status.value}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ConsolidatedReportTemplate:
    """Generate multi-entity consolidated ESRS report.

    Sections:
        1. Group Overview
        2. Entity Registry
        3. Consolidation Methodology
        4. Consolidated GHG Emissions (with entity waterfall)
        5. Per-Standard ESRS Disclosure Summary
        6. Intercompany Eliminations
        7. Minority Interest Disclosures
        8. Entity-to-Group Reconciliation
        9. Appendix (per-entity detail)

    Example:
        >>> template = ConsolidatedReportTemplate()
        >>> data = ConsolidatedReportInput(group_name="Acme Group", reporting_year=2025)
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> payload = template.render_json(data)
    """

    TEMPLATE_NAME = "consolidated_report"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the consolidated report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: ConsolidatedReportInput) -> str:
        """Render as Markdown.

        Args:
            data: Validated consolidated report input.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_group_overview(data),
            self._md_entity_registry(data),
            self._md_consolidation_methodology(data),
            self._md_consolidated_emissions(data),
            self._md_esrs_disclosures(data),
            self._md_intercompany_eliminations(data),
            self._md_minority_interest(data),
            self._md_reconciliation(data),
            self._md_appendix(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: ConsolidatedReportInput) -> str:
        """Render as HTML document.

        Args:
            data: Validated consolidated report input.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_group_overview(data),
            self._html_entity_registry(data),
            self._html_consolidation_methodology(data),
            self._html_consolidated_emissions(data),
            self._html_esrs_disclosures(data),
            self._html_intercompany_eliminations(data),
            self._html_minority_interest(data),
            self._html_reconciliation(data),
            self._html_appendix(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.group_name, data.reporting_year, body)

    def render_json(self, data: ConsolidatedReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated consolidated report input.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)

        return {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "group_name": data.group_name,
            "reporting_year": data.reporting_year,
            "consolidation_approach": data.consolidation_approach.value,
            "entity_count": len(data.entities),
            "entities": [e.model_dump(mode="json") for e in data.entities],
            "consolidated_emissions": data.consolidated_emissions.model_dump(mode="json"),
            "esrs_disclosures": {
                k: v.model_dump(mode="json")
                for k, v in data.esrs_disclosures.items()
            },
            "intercompany_eliminations": [
                e.model_dump(mode="json") for e in data.intercompany_eliminations
            ],
            "minority_interest_disclosures": data.minority_interest_disclosures,
            "reconciliation_summary": [
                r.model_dump(mode="json") for r in data.reconciliation_summary
            ],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: ConsolidatedReportInput) -> str:
        """Compute SHA-256 provenance hash for the input data."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ConsolidatedReportInput) -> str:
        """Markdown header."""
        total = data.consolidated_emissions.grand_total
        return (
            f"# Consolidated ESRS Report - {data.group_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Entities:** {len(data.entities)} | "
            f"**Total Emissions:** {_fmt_tco2e(total)}\n\n---"
        )

    def _md_group_overview(self, data: ConsolidatedReportInput) -> str:
        """Group overview section."""
        ce = data.consolidated_emissions
        total_revenue = sum(
            e.revenue_eur for e in data.entities if e.revenue_eur is not None
        )
        total_employees = sum(
            e.employee_count for e in data.entities if e.employee_count is not None
        )
        countries = sorted(set(e.country for e in data.entities)) if data.entities else []
        lines = [
            "## 1. Group Overview",
            "",
            f"- **Group Name:** {data.group_name}",
            f"- **Consolidation Approach:** {data.consolidation_approach.value.replace('_', ' ').title()}",
            f"- **Number of Entities:** {len(data.entities)}",
            f"- **Countries of Operation:** {', '.join(countries) if countries else 'N/A'}",
            f"- **Total Group Revenue:** {_fmt_number(total_revenue if total_revenue else None, 1, ' EUR')}",
            f"- **Total Employees:** {_fmt_number(float(total_employees) if total_employees else None, 0)}",
            "",
            "### Consolidated Emissions Summary",
            "",
            "| Scope | Value |",
            "|-------|-------|",
            f"| Scope 1 | {_fmt_tco2e(ce.scope1_total)} |",
            f"| Scope 2 | {_fmt_tco2e(ce.scope2_total)} |",
            f"| Scope 3 | {_fmt_tco2e(ce.scope3_total)} |",
            f"| Eliminations | -{_fmt_tco2e(ce.eliminations_tco2e)} |",
            f"| Adjustments | {_fmt_tco2e(ce.adjustment_tco2e)} |",
            f"| **Grand Total** | **{_fmt_tco2e(ce.grand_total)}** |",
        ]
        return "\n".join(lines)

    def _md_entity_registry(self, data: ConsolidatedReportInput) -> str:
        """Entity registry table."""
        lines = [
            "## 2. Entity Registry",
            "",
            "| Entity | Country | Ownership | Scope 1 | Scope 2 | Scope 3 | Quality |",
            "|--------|---------|-----------|---------|---------|---------|---------|",
        ]
        for e in sorted(data.entities, key=lambda x: x.ownership_pct, reverse=True):
            lines.append(
                f"| {e.name} | {e.country} | {e.ownership_pct:.1f}% "
                f"| {_fmt_tco2e(e.scope1_total)} | {_fmt_tco2e(e.scope2_total)} "
                f"| {_fmt_tco2e(e.scope3_total)} | {e.quality_score:.0f}% |"
            )
        if not data.entities:
            lines.append("| - | No entities registered | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_consolidation_methodology(self, data: ConsolidatedReportInput) -> str:
        """Consolidation methodology section."""
        approach = data.consolidation_approach.value.replace("_", " ").title()
        lines = [
            "## 3. Consolidation Methodology",
            "",
            f"**Approach:** {approach}",
            "",
        ]
        descriptions = {
            ConsolidationApproach.FINANCIAL_CONTROL: (
                "Under the financial control approach, the group accounts for 100% of "
                "emissions from entities over which it has financial control, regardless "
                "of equity share. Entities under financial control are fully consolidated."
            ),
            ConsolidationApproach.OPERATIONAL_CONTROL: (
                "Under the operational control approach, the group accounts for 100% of "
                "emissions from entities over which it has operational control (i.e., "
                "has the authority to introduce and implement operating policies)."
            ),
            ConsolidationApproach.EQUITY_SHARE: (
                "Under the equity share approach, the group accounts for its share of "
                "emissions from entities proportional to its equity ownership percentage. "
                "This approach aligns with financial accounting consolidation."
            ),
        }
        lines.append(descriptions.get(data.consolidation_approach, ""))
        return "\n".join(lines)

    def _md_consolidated_emissions(self, data: ConsolidatedReportInput) -> str:
        """Consolidated emissions waterfall."""
        ce = data.consolidated_emissions
        lines = [
            "## 4. Consolidated GHG Emissions",
            "",
            "### Entity Waterfall",
            "",
            "| Entity | Scope 1 | Scope 2 | Scope 3 | Total |",
            "|--------|---------|---------|---------|-------|",
        ]
        for entity_name, scopes in ce.by_entity.items():
            s1 = scopes.get("scope1", 0.0)
            s2 = scopes.get("scope2", 0.0)
            s3 = scopes.get("scope3", 0.0)
            total = s1 + s2 + s3
            lines.append(
                f"| {entity_name} | {_fmt_tco2e(s1)} | {_fmt_tco2e(s2)} "
                f"| {_fmt_tco2e(s3)} | {_fmt_tco2e(total)} |"
            )
        if not ce.by_entity:
            lines.append("| - | No entity breakdown available | - | - | - |")

        lines.extend([
            f"| **Subtotal** | **{_fmt_tco2e(ce.scope1_total)}** "
            f"| **{_fmt_tco2e(ce.scope2_total)}** "
            f"| **{_fmt_tco2e(ce.scope3_total)}** "
            f"| **{_fmt_tco2e(ce.scope1_total + ce.scope2_total + ce.scope3_total)}** |",
            f"| Eliminations | - | - | - | -{_fmt_tco2e(ce.eliminations_tco2e)} |",
            f"| Adjustments | - | - | - | {_fmt_tco2e(ce.adjustment_tco2e)} |",
            f"| **Grand Total** | - | - | - | **{_fmt_tco2e(ce.grand_total)}** |",
        ])
        return "\n".join(lines)

    def _md_esrs_disclosures(self, data: ConsolidatedReportInput) -> str:
        """Per-standard ESRS disclosure summary."""
        lines = [
            "## 5. ESRS Disclosure Summary",
            "",
            "| Standard | Name | Coverage | Status | Gaps |",
            "|----------|------|----------|--------|------|",
        ]
        for std_id in sorted(data.esrs_disclosures.keys()):
            disc = data.esrs_disclosures[std_id]
            gap_count = len(disc.gaps)
            gap_text = f"{gap_count} gap(s)" if gap_count > 0 else "None"
            lines.append(
                f"| {disc.standard_id} | {disc.standard_name} "
                f"| {_fmt_pct(disc.coverage_pct)} "
                f"| {_status_badge(disc.status)} | {gap_text} |"
            )
        if not data.esrs_disclosures:
            lines.append("| - | No ESRS disclosures available | - | - | - |")

        # Gap details
        all_gaps = []
        for std_id, disc in data.esrs_disclosures.items():
            for gap in disc.gaps:
                all_gaps.append((std_id, gap))
        if all_gaps:
            lines.extend([
                "",
                "### Disclosure Gaps",
                "",
                "| Standard | Gap Description |",
                "|----------|----------------|",
            ])
            for std_id, gap in all_gaps:
                lines.append(f"| {std_id} | {gap} |")
        return "\n".join(lines)

    def _md_intercompany_eliminations(self, data: ConsolidatedReportInput) -> str:
        """Intercompany eliminations table."""
        lines = [
            "## 6. Intercompany Eliminations",
            "",
            "| From | To | Type | Amount | Description |",
            "|------|-----|------|--------|-------------|",
        ]
        total_elim = 0.0
        for e in data.intercompany_eliminations:
            total_elim += e.amount_tco2e
            desc = e.description or "-"
            lines.append(
                f"| {e.entity_from} | {e.entity_to} | {e.elimination_type} "
                f"| {_fmt_tco2e(e.amount_tco2e)} | {desc} |"
            )
        if not data.intercompany_eliminations:
            lines.append("| - | - | No eliminations | - | - |")
        else:
            lines.append(
                f"| **Total** | - | - | **{_fmt_tco2e(total_elim)}** | - |"
            )
        return "\n".join(lines)

    def _md_minority_interest(self, data: ConsolidatedReportInput) -> str:
        """Minority interest disclosures."""
        lines = [
            "## 7. Minority Interest Disclosures",
            "",
        ]
        if not data.minority_interest_disclosures:
            lines.append("No minority interest disclosures applicable.")
            return "\n".join(lines)
        for i, item in enumerate(data.minority_interest_disclosures, 1):
            entity = item.get("entity", "N/A")
            minority_pct = item.get("minority_pct", "N/A")
            treatment = item.get("treatment", "N/A")
            notes = item.get("notes", "-")
            lines.extend([
                f"### {i}. {entity}",
                f"- **Minority Stake:** {minority_pct}%"
                if isinstance(minority_pct, (int, float))
                else f"- **Minority Stake:** {minority_pct}",
                f"- **Treatment:** {treatment}",
                f"- **Notes:** {notes}",
                "",
            ])
        return "\n".join(lines)

    def _md_reconciliation(self, data: ConsolidatedReportInput) -> str:
        """Entity-to-group reconciliation summary."""
        lines = [
            "## 8. Entity-to-Group Reconciliation",
            "",
            "| Metric | Entity Sum | Consolidated | Difference | Diff % | Status | Explanation |",
            "|--------|-----------|-------------|------------|--------|--------|-------------|",
        ]
        for r in data.reconciliation_summary:
            lines.append(
                f"| {r.metric_name} | {_fmt_number(r.entity_sum)} "
                f"| {_fmt_number(r.consolidated_value)} "
                f"| {_fmt_number(r.difference)} | {_fmt_pct(r.difference_pct)} "
                f"| {_recon_badge(r.status)} | {r.explanation or '-'} |"
            )
        if not data.reconciliation_summary:
            lines.append(
                "| - | - | No reconciliation data | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_appendix(self, data: ConsolidatedReportInput) -> str:
        """Appendix with per-entity detail."""
        if not data.entities:
            return "## 9. Appendix: Per-Entity Detail\n\nNo entity data available."
        lines = ["## 9. Appendix: Per-Entity Detail", ""]
        for i, entity in enumerate(data.entities, 1):
            total = entity.total_emissions
            method = entity.consolidation_method or data.consolidation_approach.value
            lines.extend([
                f"### {i}. {entity.name}",
                "",
                f"- **Country:** {entity.country}",
                f"- **Ownership:** {entity.ownership_pct:.1f}%",
                f"- **Consolidation Method:** {method}",
                f"- **Employees:** {_fmt_number(float(entity.employee_count) if entity.employee_count else None, 0)}",
                f"- **Revenue:** {_fmt_number(entity.revenue_eur, 1, ' EUR')}",
                f"- **Data Quality Score:** {entity.quality_score:.0f}%",
                "",
                "| Scope | Emissions |",
                "|-------|-----------|",
                f"| Scope 1 | {_fmt_tco2e(entity.scope1_total)} |",
                f"| Scope 2 | {_fmt_tco2e(entity.scope2_total)} |",
                f"| Scope 3 | {_fmt_tco2e(entity.scope3_total)} |",
                f"| **Total** | **{_fmt_tco2e(total)}** |",
                "",
            ])
        return "\n".join(lines)

    def _md_footer(self, data: ConsolidatedReportInput) -> str:
        """Markdown footer with provenance."""
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

    def _wrap_html(self, group: str, year: int, body: str) -> str:
        """Wrap body in full HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Consolidated ESRS Report - {group} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem;color:#1a1a2e;"
            "max-width:1200px;margin:2rem auto;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "h3{color:#533483;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".status-full{color:#1a7f37;font-weight:bold;}\n"
            ".status-partial{color:#b08800;font-weight:bold;}\n"
            ".status-gap{color:#cf222e;font-weight:bold;}\n"
            ".status-na{color:#888;}\n"
            ".recon-reconciled{color:#1a7f37;}\n"
            ".recon-variance{color:#b08800;}\n"
            ".recon-unreconciled{color:#cf222e;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".entity-detail{background:#f8f9fa;border:1px solid #e1e5e9;"
            "border-radius:6px;padding:1rem;margin:1rem 0;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: ConsolidatedReportInput) -> str:
        """HTML header."""
        total = data.consolidated_emissions.grand_total
        return (
            '<div class="section">\n'
            f"<h1>Consolidated ESRS Report &mdash; {data.group_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()} | "
            f"<strong>Entities:</strong> {len(data.entities)} | "
            f"<strong>Total Emissions:</strong> {_fmt_tco2e(total)}</p>\n"
            "<hr>\n</div>"
        )

    def _html_group_overview(self, data: ConsolidatedReportInput) -> str:
        """HTML group overview."""
        ce = data.consolidated_emissions
        total_revenue = sum(
            e.revenue_eur for e in data.entities if e.revenue_eur is not None
        )
        total_employees = sum(
            e.employee_count for e in data.entities if e.employee_count is not None
        )
        countries = sorted(set(e.country for e in data.entities)) if data.entities else []
        approach = data.consolidation_approach.value.replace("_", " ").title()

        cards = [
            (str(len(data.entities)), "Entities"),
            (f"{len(countries)}", "Countries"),
            (_fmt_number(total_revenue if total_revenue else None, 1, " EUR"), "Revenue"),
            (_fmt_number(float(total_employees) if total_employees else None, 0), "Employees"),
        ]
        card_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>'
            for v, l in cards
        )
        scope_rows = (
            f"<tr><td>Scope 1</td><td>{_fmt_tco2e(ce.scope1_total)}</td></tr>\n"
            f"<tr><td>Scope 2</td><td>{_fmt_tco2e(ce.scope2_total)}</td></tr>\n"
            f"<tr><td>Scope 3</td><td>{_fmt_tco2e(ce.scope3_total)}</td></tr>\n"
            f"<tr><td>Eliminations</td><td>-{_fmt_tco2e(ce.eliminations_tco2e)}</td></tr>\n"
            f"<tr><td>Adjustments</td><td>{_fmt_tco2e(ce.adjustment_tco2e)}</td></tr>\n"
            f"<tr><td><strong>Grand Total</strong></td>"
            f"<td><strong>{_fmt_tco2e(ce.grand_total)}</strong></td></tr>"
        )
        return (
            '<div class="section">\n'
            "<h2>1. Group Overview</h2>\n"
            f"<p><strong>Consolidation Approach:</strong> {approach}</p>\n"
            f"<div>{card_html}</div>\n"
            "<h3>Consolidated Emissions Summary</h3>\n"
            "<table><thead><tr><th>Scope</th><th>Value</th></tr></thead>\n"
            f"<tbody>{scope_rows}</tbody></table>\n</div>"
        )

    def _html_entity_registry(self, data: ConsolidatedReportInput) -> str:
        """HTML entity registry table."""
        rows = []
        for e in sorted(data.entities, key=lambda x: x.ownership_pct, reverse=True):
            rows.append(
                f"<tr><td>{e.name}</td><td>{e.country}</td>"
                f"<td>{e.ownership_pct:.1f}%</td>"
                f"<td>{_fmt_tco2e(e.scope1_total)}</td>"
                f"<td>{_fmt_tco2e(e.scope2_total)}</td>"
                f"<td>{_fmt_tco2e(e.scope3_total)}</td>"
                f"<td>{e.quality_score:.0f}%</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="7">No entities registered</td></tr>')
        return (
            '<div class="section">\n<h2>2. Entity Registry</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Country</th><th>Ownership</th>"
            "<th>Scope 1</th><th>Scope 2</th><th>Scope 3</th>"
            f"<th>Quality</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_consolidation_methodology(self, data: ConsolidatedReportInput) -> str:
        """HTML consolidation methodology."""
        approach = data.consolidation_approach.value.replace("_", " ").title()
        descriptions = {
            ConsolidationApproach.FINANCIAL_CONTROL: (
                "Under the financial control approach, the group accounts for 100% of "
                "emissions from entities over which it has financial control."
            ),
            ConsolidationApproach.OPERATIONAL_CONTROL: (
                "Under the operational control approach, the group accounts for 100% of "
                "emissions from entities over which it has operational control."
            ),
            ConsolidationApproach.EQUITY_SHARE: (
                "Under the equity share approach, emissions are accounted proportional "
                "to equity ownership percentage."
            ),
        }
        desc = descriptions.get(data.consolidation_approach, "")
        return (
            '<div class="section">\n<h2>3. Consolidation Methodology</h2>\n'
            f"<p><strong>Approach:</strong> {approach}</p>\n"
            f"<p>{desc}</p>\n</div>"
        )

    def _html_consolidated_emissions(self, data: ConsolidatedReportInput) -> str:
        """HTML consolidated emissions with entity waterfall."""
        ce = data.consolidated_emissions
        rows = []
        for name, scopes in ce.by_entity.items():
            s1 = scopes.get("scope1", 0.0)
            s2 = scopes.get("scope2", 0.0)
            s3 = scopes.get("scope3", 0.0)
            total = s1 + s2 + s3
            rows.append(
                f"<tr><td>{name}</td><td>{_fmt_tco2e(s1)}</td>"
                f"<td>{_fmt_tco2e(s2)}</td><td>{_fmt_tco2e(s3)}</td>"
                f"<td>{_fmt_tco2e(total)}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="5">No entity breakdown available</td></tr>')
        subtotal = ce.scope1_total + ce.scope2_total + ce.scope3_total
        rows.extend([
            f"<tr style='font-weight:bold'><td>Subtotal</td>"
            f"<td>{_fmt_tco2e(ce.scope1_total)}</td>"
            f"<td>{_fmt_tco2e(ce.scope2_total)}</td>"
            f"<td>{_fmt_tco2e(ce.scope3_total)}</td>"
            f"<td>{_fmt_tco2e(subtotal)}</td></tr>",
            f"<tr><td>Eliminations</td><td>-</td><td>-</td><td>-</td>"
            f"<td>-{_fmt_tco2e(ce.eliminations_tco2e)}</td></tr>",
            f"<tr><td>Adjustments</td><td>-</td><td>-</td><td>-</td>"
            f"<td>{_fmt_tco2e(ce.adjustment_tco2e)}</td></tr>",
            f"<tr style='font-weight:bold;background:#e8f4e8'><td>Grand Total</td>"
            f"<td>-</td><td>-</td><td>-</td>"
            f"<td>{_fmt_tco2e(ce.grand_total)}</td></tr>",
        ])
        return (
            '<div class="section">\n<h2>4. Consolidated GHG Emissions</h2>\n'
            "<h3>Entity Waterfall</h3>\n"
            "<table><thead><tr><th>Entity</th><th>Scope 1</th><th>Scope 2</th>"
            f"<th>Scope 3</th><th>Total</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_esrs_disclosures(self, data: ConsolidatedReportInput) -> str:
        """HTML ESRS disclosure summary."""
        rows = []
        for std_id in sorted(data.esrs_disclosures.keys()):
            disc = data.esrs_disclosures[std_id]
            css = f"status-{disc.status.value.lower().replace('/', '')}"
            gap_count = len(disc.gaps)
            gap_text = f"{gap_count} gap(s)" if gap_count > 0 else "None"
            rows.append(
                f"<tr><td>{disc.standard_id}</td><td>{disc.standard_name}</td>"
                f"<td>{_fmt_pct(disc.coverage_pct)}</td>"
                f'<td class="{css}">{disc.status.value}</td>'
                f"<td>{gap_text}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="5">No ESRS disclosures available</td></tr>')
        return (
            '<div class="section">\n<h2>5. ESRS Disclosure Summary</h2>\n'
            "<table><thead><tr><th>Standard</th><th>Name</th><th>Coverage</th>"
            f"<th>Status</th><th>Gaps</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_intercompany_eliminations(self, data: ConsolidatedReportInput) -> str:
        """HTML intercompany eliminations."""
        rows = []
        total_elim = 0.0
        for e in data.intercompany_eliminations:
            total_elim += e.amount_tco2e
            desc = e.description or "-"
            rows.append(
                f"<tr><td>{e.entity_from}</td><td>{e.entity_to}</td>"
                f"<td>{e.elimination_type}</td>"
                f"<td>{_fmt_tco2e(e.amount_tco2e)}</td>"
                f"<td>{desc}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="5">No eliminations</td></tr>')
        else:
            rows.append(
                f"<tr style='font-weight:bold'><td>Total</td><td>-</td><td>-</td>"
                f"<td>{_fmt_tco2e(total_elim)}</td><td>-</td></tr>"
            )
        return (
            '<div class="section">\n<h2>6. Intercompany Eliminations</h2>\n'
            "<table><thead><tr><th>From</th><th>To</th><th>Type</th>"
            f"<th>Amount</th><th>Description</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_minority_interest(self, data: ConsolidatedReportInput) -> str:
        """HTML minority interest disclosures."""
        if not data.minority_interest_disclosures:
            return (
                '<div class="section">\n<h2>7. Minority Interest</h2>\n'
                "<p>No minority interest disclosures applicable.</p>\n</div>"
            )
        items_html = []
        for i, item in enumerate(data.minority_interest_disclosures, 1):
            entity = item.get("entity", "N/A")
            minority_pct = item.get("minority_pct", "N/A")
            treatment = item.get("treatment", "N/A")
            notes = item.get("notes", "-")
            items_html.append(
                f'<div class="entity-detail">\n'
                f"<h3>{i}. {entity}</h3>\n"
                f"<p><strong>Minority Stake:</strong> {minority_pct}"
                f"{'%' if isinstance(minority_pct, (int, float)) else ''}</p>\n"
                f"<p><strong>Treatment:</strong> {treatment}</p>\n"
                f"<p><strong>Notes:</strong> {notes}</p>\n</div>"
            )
        return (
            '<div class="section">\n<h2>7. Minority Interest</h2>\n'
            f"{''.join(items_html)}\n</div>"
        )

    def _html_reconciliation(self, data: ConsolidatedReportInput) -> str:
        """HTML reconciliation summary."""
        rows = []
        for r in data.reconciliation_summary:
            css = f"recon-{r.status.value.lower()}"
            lines_row = (
                f"<tr><td>{r.metric_name}</td>"
                f"<td>{_fmt_number(r.entity_sum)}</td>"
                f"<td>{_fmt_number(r.consolidated_value)}</td>"
                f"<td>{_fmt_number(r.difference)}</td>"
                f"<td>{_fmt_pct(r.difference_pct)}</td>"
                f'<td class="{css}">{r.status.value}</td>'
                f"<td>{r.explanation or '-'}</td></tr>"
            )
            rows.append(lines_row)
        if not rows:
            rows.append(
                '<tr><td colspan="7">No reconciliation data</td></tr>'
            )
        return (
            '<div class="section">\n<h2>8. Entity-to-Group Reconciliation</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Entity Sum</th>"
            "<th>Consolidated</th><th>Difference</th><th>Diff %</th>"
            f"<th>Status</th><th>Explanation</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_appendix(self, data: ConsolidatedReportInput) -> str:
        """HTML appendix per-entity detail."""
        if not data.entities:
            return (
                '<div class="section">\n<h2>9. Appendix</h2>\n'
                "<p>No entity data available.</p>\n</div>"
            )
        details = []
        for i, entity in enumerate(data.entities, 1):
            total = entity.total_emissions
            method = entity.consolidation_method or data.consolidation_approach.value
            details.append(
                f'<div class="entity-detail">\n'
                f"<h3>{i}. {entity.name}</h3>\n"
                f"<p><strong>Country:</strong> {entity.country} | "
                f"<strong>Ownership:</strong> {entity.ownership_pct:.1f}% | "
                f"<strong>Method:</strong> {method} | "
                f"<strong>Quality:</strong> {entity.quality_score:.0f}%</p>\n"
                "<table><thead><tr><th>Scope</th><th>Emissions</th></tr></thead>\n"
                f"<tbody>"
                f"<tr><td>Scope 1</td><td>{_fmt_tco2e(entity.scope1_total)}</td></tr>"
                f"<tr><td>Scope 2</td><td>{_fmt_tco2e(entity.scope2_total)}</td></tr>"
                f"<tr><td>Scope 3</td><td>{_fmt_tco2e(entity.scope3_total)}</td></tr>"
                f"<tr style='font-weight:bold'><td>Total</td>"
                f"<td>{_fmt_tco2e(total)}</td></tr>"
                f"</tbody></table>\n</div>"
            )
        return (
            '<div class="section">\n<h2>9. Appendix: Per-Entity Detail</h2>\n'
            f"{''.join(details)}\n</div>"
        )

    def _html_footer(self, data: ConsolidatedReportInput) -> str:
        """HTML footer with provenance."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
