"""
CustomsIntegrationReport - CBAM customs integration and reconciliation template.

This module implements the customs integration report for PACK-005 CBAM Complete.
It generates reports covering import summaries by CN code, CBAM applicability
assessments, CN code TARIC validation, anti-circumvention flags, customs procedure
breakdowns, combined duty + CBAM cost analysis, SAD reconciliation, and EORI
number validation.

Example:
    >>> template = CustomsIntegrationReport()
    >>> data = CustomsIntegrationData(
    ...     import_summary=ImportSummary(total_imports=250, ...),
    ...     cbam_applicability=[ApplicabilityRecord(...)],
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

class ImportByCode(BaseModel):
    """Import summary by CN code."""

    cn_code: str = Field("", description="CN/HS code")
    description: str = Field("", description="Product description")
    country_of_origin: str = Field("", description="Country of origin")
    customs_office: str = Field("", description="Customs office")
    volume_tonnes: float = Field(0.0, ge=0.0, description="Import volume tonnes")
    value_eur: float = Field(0.0, ge=0.0, description="Import value EUR")
    num_shipments: int = Field(0, ge=0, description="Number of shipments")


class ImportSummary(BaseModel):
    """Overall import summary."""

    total_imports: int = Field(0, ge=0, description="Total import shipments")
    total_volume_tonnes: float = Field(0.0, ge=0.0, description="Total volume imported")
    total_value_eur: float = Field(0.0, ge=0.0, description="Total import value")
    unique_cn_codes: int = Field(0, ge=0, description="Unique CN codes")
    unique_countries: int = Field(0, ge=0, description="Unique countries of origin")
    unique_customs_offices: int = Field(0, ge=0, description="Unique customs offices")
    reporting_period: str = Field("", description="Reporting period")
    imports_by_code: List[ImportByCode] = Field(default_factory=list)


class ApplicabilityRecord(BaseModel):
    """CBAM applicability assessment per import."""

    cn_code: str = Field("", description="CN code")
    description: str = Field("", description="Product description")
    cbam_covered: bool = Field(False, description="Whether CBAM applies")
    cbam_sector: str = Field("", description="CBAM sector if applicable")
    exemption_reason: str = Field("", description="Reason if exempt")
    volume_tonnes: float = Field(0.0, ge=0.0, description="Volume tonnes")


class CNCodeValidation(BaseModel):
    """CN code TARIC validation result."""

    cn_code: str = Field("", description="CN code")
    taric_valid: bool = Field(True, description="TARIC validation result")
    taric_description: str = Field("", description="TARIC description")
    cbam_annex_match: bool = Field(True, description="Matches CBAM Annex I")
    mismatch_details: str = Field("", description="Mismatch details if any")
    warning: str = Field("", description="Warning message")


class CircumventionFlag(BaseModel):
    """Anti-circumvention detection flag."""

    flag_id: str = Field("", description="Flag identifier")
    pattern_type: str = Field("", description="Pattern type detected")
    description: str = Field("", description="Description of suspicious pattern")
    severity: str = Field("low", description="low, medium, high, critical")
    cn_codes_involved: List[str] = Field(default_factory=list, description="CN codes involved")
    countries_involved: List[str] = Field(default_factory=list, description="Countries involved")
    evidence: str = Field("", description="Supporting evidence")
    recommended_action: str = Field("", description="Recommended follow-up action")


class CustomsProcedure(BaseModel):
    """Customs procedure breakdown."""

    procedure_code: str = Field("", description="Customs procedure code")
    procedure_name: str = Field("", description="Procedure name")
    volume_tonnes: float = Field(0.0, ge=0.0, description="Volume under this procedure")
    value_eur: float = Field(0.0, ge=0.0, description="Value under this procedure")
    shipment_count: int = Field(0, ge=0, description="Number of shipments")
    cbam_applicable: bool = Field(True, description="Whether CBAM applies")
    notes: str = Field("", description="Additional notes")


class DutyCBAMCost(BaseModel):
    """Combined duty and CBAM cost per category."""

    category: str = Field("", description="Shipment category or CN code group")
    customs_duty_eur: float = Field(0.0, ge=0.0, description="Customs duty amount")
    customs_duty_rate_pct: float = Field(0.0, ge=0.0, description="Duty rate percentage")
    cbam_cost_eur: float = Field(0.0, ge=0.0, description="CBAM certificate cost")
    combined_cost_eur: float = Field(0.0, ge=0.0, description="Total combined cost")
    cbam_share_pct: float = Field(0.0, ge=0.0, le=100.0, description="CBAM as share of total")
    volume_tonnes: float = Field(0.0, ge=0.0, description="Volume tonnes")


class SADReconciliation(BaseModel):
    """Single Administrative Document reconciliation record."""

    sad_reference: str = Field("", description="SAD reference number")
    cbam_import_id: str = Field("", description="CBAM import record ID")
    match_status: str = Field("matched", description="matched, partial, unmatched")
    volume_sad_tonnes: float = Field(0.0, ge=0.0, description="Volume per SAD")
    volume_cbam_tonnes: float = Field(0.0, ge=0.0, description="Volume per CBAM record")
    variance_tonnes: float = Field(0.0, description="Volume variance")
    variance_pct: float = Field(0.0, description="Variance percentage")
    notes: str = Field("", description="Reconciliation notes")


class EORIValidation(BaseModel):
    """EORI number validation record."""

    eori_number: str = Field("", description="EORI number")
    entity_name: str = Field("", description="Entity name")
    valid: bool = Field(True, description="Whether EORI is valid")
    member_state: str = Field("", description="Registration member state")
    validation_date: str = Field("", description="Last validation date")
    issues: str = Field("", description="Issues found")


class CustomsIntegrationData(BaseModel):
    """Complete input data for customs integration report."""

    import_summary: ImportSummary = Field(default_factory=ImportSummary)
    cbam_applicability: List[ApplicabilityRecord] = Field(default_factory=list)
    cn_code_validations: List[CNCodeValidation] = Field(default_factory=list)
    circumvention_flags: List[CircumventionFlag] = Field(default_factory=list)
    customs_procedures: List[CustomsProcedure] = Field(default_factory=list)
    duty_cbam_costs: List[DutyCBAMCost] = Field(default_factory=list)
    sad_reconciliations: List[SADReconciliation] = Field(default_factory=list)
    eori_validations: List[EORIValidation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class CustomsIntegrationReport:
    """
    CBAM customs integration report template.

    Generates reports covering customs-CBAM integration, including import
    summaries, applicability assessments, CN code validation, anti-circumvention
    detection, customs procedure breakdown, duty+CBAM cost, SAD reconciliation,
    and EORI validation.

    Attributes:
        config: Optional configuration dictionary.
        pack_id: Pack identifier (PACK-005).
        template_name: Template name for metadata.
        version: Template version.

    Example:
        >>> template = CustomsIntegrationReport()
        >>> md = template.render_markdown(data)
        >>> assert "Import Summary" in md
    """

    PACK_ID = "PACK-005"
    TEMPLATE_NAME = "customs_integration_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CustomsIntegrationReport.

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
        Render the customs integration report as Markdown.

        Args:
            data: Report data dictionary matching CustomsIntegrationData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header())
        sections.append(self._md_import_summary(data))
        sections.append(self._md_cbam_applicability(data))
        sections.append(self._md_cn_code_validation(data))
        sections.append(self._md_circumvention_flags(data))
        sections.append(self._md_customs_procedures(data))
        sections.append(self._md_duty_cbam_cost(data))
        sections.append(self._md_sad_reconciliation(data))
        sections.append(self._md_eori_validation(data))

        content = "\n\n".join(sections)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the customs integration report as self-contained HTML.

        Args:
            data: Report data dictionary matching CustomsIntegrationData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._html_import_summary(data))
        sections.append(self._html_cbam_applicability(data))
        sections.append(self._html_cn_code_validation(data))
        sections.append(self._html_circumvention_flags(data))
        sections.append(self._html_customs_procedures(data))
        sections.append(self._html_duty_cbam_cost(data))
        sections.append(self._html_sad_reconciliation(data))
        sections.append(self._html_eori_validation(data))

        body = "\n".join(sections)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="CBAM Customs Integration Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the customs integration report as structured JSON.

        Args:
            data: Report data dictionary matching CustomsIntegrationData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_customs_integration",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "import_summary": self._json_import_summary(data),
            "cbam_applicability": self._json_applicability(data),
            "cn_code_validations": self._json_cn_validations(data),
            "circumvention_flags": self._json_circumvention(data),
            "customs_procedures": self._json_procedures(data),
            "duty_cbam_costs": self._json_duty_cbam(data),
            "sad_reconciliations": self._json_sad(data),
            "eori_validations": self._json_eori(data),
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
            "# CBAM Customs Integration Report\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_import_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown import summary section."""
        ims = data.get("import_summary", {})
        cur = self._currency()
        imports = ims.get("imports_by_code", [])

        summary = (
            "## 1. Import Summary\n\n"
            f"**Reporting Period:** {ims.get('reporting_period', 'N/A')}\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Total Shipments | {self._fmt_int(ims.get('total_imports', 0))} |\n"
            f"| Total Volume | {self._fmt_num(ims.get('total_volume_tonnes', 0.0))} tonnes |\n"
            f"| Total Value | {self._fmt_cur(ims.get('total_value_eur', 0.0), cur)} |\n"
            f"| Unique CN Codes | {ims.get('unique_cn_codes', 0)} |\n"
            f"| Unique Countries | {ims.get('unique_countries', 0)} |\n"
            f"| Customs Offices | {ims.get('unique_customs_offices', 0)} |"
        )

        if not imports:
            return summary

        detail = (
            "\n\n### By CN Code\n\n"
            "| CN Code | Description | Country | Office | Volume (t) | Value | Shipments |\n"
            "|---------|-------------|---------|--------|------------|-------|----------|\n"
        )

        rows: List[str] = []
        for i in imports:
            rows.append(
                f"| {self._fmt_cn(i.get('cn_code', ''))} | "
                f"{i.get('description', '')} | "
                f"{i.get('country_of_origin', '')} | "
                f"{i.get('customs_office', '')} | "
                f"{self._fmt_num(i.get('volume_tonnes', 0.0))} | "
                f"{self._fmt_cur(i.get('value_eur', 0.0), cur)} | "
                f"{i.get('num_shipments', 0)} |"
            )

        return summary + detail + "\n".join(rows)

    def _md_cbam_applicability(self, data: Dict[str, Any]) -> str:
        """Build Markdown CBAM applicability section."""
        records = data.get("cbam_applicability", [])

        header = (
            "## 2. CBAM Applicability\n\n"
            "| CN Code | Description | CBAM Covered | Sector | Exemption | Volume (t) |\n"
            "|---------|-------------|-------------|--------|-----------|------------|\n"
        )

        rows: List[str] = []
        covered_count = 0
        exempt_count = 0

        for r in records:
            covered = r.get("cbam_covered", False)
            if covered:
                covered_count += 1
            else:
                exempt_count += 1

            status = "YES" if covered else "NO"
            exemption = r.get("exemption_reason", "") or "-"
            sector = r.get("cbam_sector", "") or "-"
            rows.append(
                f"| {self._fmt_cn(r.get('cn_code', ''))} | "
                f"{r.get('description', '')} | "
                f"**{status}** | "
                f"{sector} | "
                f"{exemption} | "
                f"{self._fmt_num(r.get('volume_tonnes', 0.0))} |"
            )

        if not rows:
            return header + "| *No records* | | | | | |"

        summary = (
            f"\n\n**Summary:** {covered_count} CBAM-covered | "
            f"{exempt_count} exempt"
        )

        return header + "\n".join(rows) + summary

    def _md_cn_code_validation(self, data: Dict[str, Any]) -> str:
        """Build Markdown CN code validation section."""
        validations = data.get("cn_code_validations", [])

        header = (
            "## 3. CN Code Validation (TARIC)\n\n"
            "| CN Code | TARIC Valid | TARIC Description | Annex Match | Mismatch | Warning |\n"
            "|---------|------------|-------------------|-------------|----------|--------|\n"
        )

        rows: List[str] = []
        for v in validations:
            taric_valid = "PASS" if v.get("taric_valid", True) else "FAIL"
            annex_match = "YES" if v.get("cbam_annex_match", True) else "NO"
            mismatch = v.get("mismatch_details", "") or "-"
            warning = v.get("warning", "") or "-"
            rows.append(
                f"| {self._fmt_cn(v.get('cn_code', ''))} | "
                f"{taric_valid} | "
                f"{v.get('taric_description', '')} | "
                f"{annex_match} | "
                f"{mismatch} | "
                f"{warning} |"
            )

        if not rows:
            return header + "| *No validations* | | | | | |"

        return header + "\n".join(rows)

    def _md_circumvention_flags(self, data: Dict[str, Any]) -> str:
        """Build Markdown anti-circumvention flags section."""
        flags = data.get("circumvention_flags", [])

        if not flags:
            return (
                "## 4. Anti-Circumvention Flags\n\n"
                "No suspicious patterns detected."
            )

        header = "## 4. Anti-Circumvention Flags\n\n"
        items: List[str] = []

        for f in flags:
            severity = f.get("severity", "low").upper()
            cn_codes = ", ".join(f.get("cn_codes_involved", []))
            countries = ", ".join(f.get("countries_involved", []))

            items.append(
                f"### Flag {f.get('flag_id', '')} - {severity}\n\n"
                f"- **Pattern:** {f.get('pattern_type', '')}\n"
                f"- **Description:** {f.get('description', '')}\n"
                f"- **CN Codes:** {cn_codes or 'N/A'}\n"
                f"- **Countries:** {countries or 'N/A'}\n"
                f"- **Evidence:** {f.get('evidence', 'N/A')}\n"
                f"- **Action:** {f.get('recommended_action', 'N/A')}"
            )

        critical = sum(1 for f in flags if f.get("severity") == "critical")
        high = sum(1 for f in flags if f.get("severity") == "high")

        summary = f"\n\n**Total Flags:** {len(flags)}"
        if critical > 0:
            summary += f" | **CRITICAL: {critical}**"
        if high > 0:
            summary += f" | **HIGH: {high}**"

        return header + "\n\n".join(items) + summary

    def _md_customs_procedures(self, data: Dict[str, Any]) -> str:
        """Build Markdown customs procedure breakdown section."""
        procedures = data.get("customs_procedures", [])
        cur = self._currency()

        header = (
            "## 5. Customs Procedure Breakdown\n\n"
            "| Code | Procedure | Volume (t) | Value | Shipments | CBAM | Notes |\n"
            "|------|-----------|-----------|-------|-----------|------|-------|\n"
        )

        rows: List[str] = []
        for p in procedures:
            cbam = "YES" if p.get("cbam_applicable", True) else "NO"
            notes = p.get("notes", "") or "-"
            rows.append(
                f"| {p.get('procedure_code', '')} | "
                f"{p.get('procedure_name', '')} | "
                f"{self._fmt_num(p.get('volume_tonnes', 0.0))} | "
                f"{self._fmt_cur(p.get('value_eur', 0.0), cur)} | "
                f"{p.get('shipment_count', 0)} | "
                f"{cbam} | "
                f"{notes} |"
            )

        if not rows:
            return header + "| *No procedures* | | | | | | |"

        return header + "\n".join(rows)

    def _md_duty_cbam_cost(self, data: Dict[str, Any]) -> str:
        """Build Markdown combined duty + CBAM cost section."""
        costs = data.get("duty_cbam_costs", [])
        cur = self._currency()

        header = (
            "## 6. Duty + CBAM Cost\n\n"
            "| Category | Duty | Rate | CBAM Cost | Combined | CBAM Share | Volume (t) |\n"
            "|----------|------|------|-----------|----------|------------|------------|\n"
        )

        rows: List[str] = []
        total_duty = 0.0
        total_cbam = 0.0
        total_combined = 0.0

        for c in costs:
            duty = c.get("customs_duty_eur", 0.0)
            cbam = c.get("cbam_cost_eur", 0.0)
            combined = c.get("combined_cost_eur", 0.0)
            total_duty += duty
            total_cbam += cbam
            total_combined += combined

            rows.append(
                f"| {c.get('category', '')} | "
                f"{self._fmt_cur(duty, cur)} | "
                f"{c.get('customs_duty_rate_pct', 0.0):.1f}% | "
                f"{self._fmt_cur(cbam, cur)} | "
                f"{self._fmt_cur(combined, cur)} | "
                f"{c.get('cbam_share_pct', 0.0):.1f}% | "
                f"{self._fmt_num(c.get('volume_tonnes', 0.0))} |"
            )

        if not rows:
            return header + "| *No cost data* | | | | | | |"

        cbam_total_share = (total_cbam / max(total_combined, 1)) * 100
        rows.append(
            f"| **TOTAL** | **{self._fmt_cur(total_duty, cur)}** | | "
            f"**{self._fmt_cur(total_cbam, cur)}** | "
            f"**{self._fmt_cur(total_combined, cur)}** | "
            f"**{cbam_total_share:.1f}%** | |"
        )

        return header + "\n".join(rows)

    def _md_sad_reconciliation(self, data: Dict[str, Any]) -> str:
        """Build Markdown SAD reconciliation section."""
        records = data.get("sad_reconciliations", [])

        header = (
            "## 7. SAD Reconciliation\n\n"
            "| SAD Ref | CBAM Import ID | Status | SAD Vol (t) | CBAM Vol (t) | Variance | Notes |\n"
            "|---------|----------------|--------|-------------|-------------|----------|-------|\n"
        )

        rows: List[str] = []
        for r in records:
            status = r.get("match_status", "matched")
            variance = r.get("variance_tonnes", 0.0)
            var_pct = r.get("variance_pct", 0.0)
            notes = r.get("notes", "") or "-"

            var_str = f"{variance:+.2f}t ({var_pct:+.1f}%)" if variance != 0 else "-"

            rows.append(
                f"| {r.get('sad_reference', '')} | "
                f"{r.get('cbam_import_id', '')} | "
                f"{status.upper()} | "
                f"{self._fmt_num(r.get('volume_sad_tonnes', 0.0))} | "
                f"{self._fmt_num(r.get('volume_cbam_tonnes', 0.0))} | "
                f"{var_str} | "
                f"{notes} |"
            )

        if not rows:
            return header + "| *No records* | | | | | | |"

        matched = sum(1 for r in records if r.get("match_status") == "matched")
        partial = sum(1 for r in records if r.get("match_status") == "partial")
        unmatched = sum(1 for r in records if r.get("match_status") == "unmatched")

        summary = (
            f"\n\n**Reconciliation Summary:** "
            f"{matched} matched | {partial} partial | {unmatched} unmatched"
        )

        return header + "\n".join(rows) + summary

    def _md_eori_validation(self, data: Dict[str, Any]) -> str:
        """Build Markdown EORI validation section."""
        records = data.get("eori_validations", [])

        header = (
            "## 8. EORI Validation\n\n"
            "| EORI Number | Entity | Valid | Member State | Last Validated | Issues |\n"
            "|-------------|--------|-------|-------------|---------------|--------|\n"
        )

        rows: List[str] = []
        for r in records:
            valid = "VALID" if r.get("valid", True) else "INVALID"
            issues = r.get("issues", "") or "-"
            rows.append(
                f"| {r.get('eori_number', '')} | "
                f"{r.get('entity_name', '')} | "
                f"**{valid}** | "
                f"{r.get('member_state', '')} | "
                f"{self._fmt_date(r.get('validation_date', ''))} | "
                f"{issues} |"
            )

        if not rows:
            return header + "| *No EORI records* | | | | | |"

        invalid_count = sum(1 for r in records if not r.get("valid", True))
        if invalid_count > 0:
            rows.append(
                f"\n\n> **WARNING:** {invalid_count} EORI number(s) failed validation. "
                f"Immediate action required."
            )

        return header + "\n".join(rows)

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
            '<h1>CBAM Customs Integration Report</h1>'
            f'<div class="meta-item">Pack: {self.PACK_ID} | '
            f'Template: {self.TEMPLATE_NAME} | Version: {self.VERSION}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_import_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML import summary section."""
        ims = data.get("import_summary", {})
        cur = self._currency()
        imports = ims.get("imports_by_code", [])

        kpis = (
            f'<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Total Shipments</div>'
            f'<div class="kpi-value">{self._fmt_int(ims.get("total_imports", 0))}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total Volume</div>'
            f'<div class="kpi-value">{self._fmt_num(ims.get("total_volume_tonnes", 0.0))}</div>'
            f'<div class="kpi-unit">tonnes</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total Value</div>'
            f'<div class="kpi-value">{self._fmt_cur(ims.get("total_value_eur", 0.0), cur)}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">CN Codes</div>'
            f'<div class="kpi-value">{ims.get("unique_cn_codes", 0)}</div></div>'
            f'</div>'
        )

        rows_html = ""
        for i in imports:
            rows_html += (
                f'<tr><td>{self._fmt_cn(i.get("cn_code", ""))}</td>'
                f'<td>{i.get("description", "")}</td>'
                f'<td>{i.get("country_of_origin", "")}</td>'
                f'<td>{i.get("customs_office", "")}</td>'
                f'<td class="num">{self._fmt_num(i.get("volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_cur(i.get("value_eur", 0.0), cur)}</td>'
                f'<td class="num">{i.get("num_shipments", 0)}</td></tr>'
            )

        table = ""
        if rows_html:
            table = (
                '<h3>By CN Code</h3>'
                '<table><thead><tr>'
                '<th>CN Code</th><th>Description</th><th>Country</th>'
                '<th>Office</th><th>Volume (t)</th><th>Value</th><th>Shipments</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>'
            )

        return (
            f'<div class="section"><h2>1. Import Summary</h2>'
            f'<p>Reporting Period: <strong>{ims.get("reporting_period", "N/A")}</strong></p>'
            f'{kpis}{table}</div>'
        )

    def _html_cbam_applicability(self, data: Dict[str, Any]) -> str:
        """Build HTML CBAM applicability section."""
        records = data.get("cbam_applicability", [])

        rows_html = ""
        for r in records:
            covered = r.get("cbam_covered", False)
            color = "#2ecc71" if covered else "#95a5a6"
            status = "YES" if covered else "NO"
            rows_html += (
                f'<tr><td>{self._fmt_cn(r.get("cn_code", ""))}</td>'
                f'<td>{r.get("description", "")}</td>'
                f'<td style="color:{color};font-weight:bold">{status}</td>'
                f'<td>{r.get("cbam_sector", "") or "-"}</td>'
                f'<td>{r.get("exemption_reason", "") or "-"}</td>'
                f'<td class="num">{self._fmt_num(r.get("volume_tonnes", 0.0))}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No records</em></td></tr>'

        return (
            '<div class="section"><h2>2. CBAM Applicability</h2>'
            '<table><thead><tr>'
            '<th>CN Code</th><th>Description</th><th>CBAM Covered</th>'
            '<th>Sector</th><th>Exemption</th><th>Volume (t)</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_cn_code_validation(self, data: Dict[str, Any]) -> str:
        """Build HTML CN code validation section."""
        validations = data.get("cn_code_validations", [])

        rows_html = ""
        for v in validations:
            taric_valid = v.get("taric_valid", True)
            annex_match = v.get("cbam_annex_match", True)
            t_color = "#2ecc71" if taric_valid else "#e74c3c"
            a_color = "#2ecc71" if annex_match else "#e74c3c"

            rows_html += (
                f'<tr><td>{self._fmt_cn(v.get("cn_code", ""))}</td>'
                f'<td style="color:{t_color};font-weight:bold">'
                f'{"PASS" if taric_valid else "FAIL"}</td>'
                f'<td>{v.get("taric_description", "")}</td>'
                f'<td style="color:{a_color};font-weight:bold">'
                f'{"YES" if annex_match else "NO"}</td>'
                f'<td>{v.get("mismatch_details", "") or "-"}</td>'
                f'<td>{v.get("warning", "") or "-"}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No validations</em></td></tr>'

        return (
            '<div class="section"><h2>3. CN Code Validation (TARIC)</h2>'
            '<table><thead><tr>'
            '<th>CN Code</th><th>TARIC Valid</th><th>TARIC Description</th>'
            '<th>Annex Match</th><th>Mismatch</th><th>Warning</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_circumvention_flags(self, data: Dict[str, Any]) -> str:
        """Build HTML anti-circumvention flags section."""
        flags = data.get("circumvention_flags", [])

        severity_colors = {
            "low": "#2ecc71", "medium": "#f39c12",
            "high": "#e74c3c", "critical": "#8e44ad",
        }

        cards = ""
        for f in flags:
            severity = f.get("severity", "low")
            color = severity_colors.get(severity, "#95a5a6")
            cn_codes = ", ".join(f.get("cn_codes_involved", [])) or "N/A"
            countries = ", ".join(f.get("countries_involved", [])) or "N/A"

            cards += (
                f'<div style="background:#f8f9fa;padding:16px;border-radius:8px;'
                f'border-left:4px solid {color};margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h3 style="margin:0">Flag {f.get("flag_id", "")}: '
                f'{f.get("pattern_type", "")}</h3>'
                f'<span style="background:{color};color:#fff;padding:4px 12px;'
                f'border-radius:12px;font-size:12px;font-weight:bold">'
                f'{severity.upper()}</span></div>'
                f'<p style="margin:8px 0">{f.get("description", "")}</p>'
                f'<div style="font-size:13px;color:#7f8c8d">'
                f'<div>CN Codes: {cn_codes}</div>'
                f'<div>Countries: {countries}</div>'
                f'<div>Action: {f.get("recommended_action", "N/A")}</div>'
                f'</div></div>'
            )

        if not cards:
            cards = '<p style="color:#2ecc71;font-weight:bold">No suspicious patterns detected.</p>'

        return f'<div class="section"><h2>4. Anti-Circumvention Flags</h2>{cards}</div>'

    def _html_customs_procedures(self, data: Dict[str, Any]) -> str:
        """Build HTML customs procedure breakdown section."""
        procedures = data.get("customs_procedures", [])
        cur = self._currency()

        rows_html = ""
        for p in procedures:
            cbam = p.get("cbam_applicable", True)
            cbam_color = "#2ecc71" if cbam else "#95a5a6"
            rows_html += (
                f'<tr><td>{p.get("procedure_code", "")}</td>'
                f'<td>{p.get("procedure_name", "")}</td>'
                f'<td class="num">{self._fmt_num(p.get("volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_cur(p.get("value_eur", 0.0), cur)}</td>'
                f'<td class="num">{p.get("shipment_count", 0)}</td>'
                f'<td style="color:{cbam_color};font-weight:bold">'
                f'{"YES" if cbam else "NO"}</td>'
                f'<td>{p.get("notes", "") or "-"}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No procedures</em></td></tr>'

        return (
            '<div class="section"><h2>5. Customs Procedure Breakdown</h2>'
            '<table><thead><tr>'
            '<th>Code</th><th>Procedure</th><th>Volume (t)</th>'
            '<th>Value</th><th>Shipments</th><th>CBAM</th><th>Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_duty_cbam_cost(self, data: Dict[str, Any]) -> str:
        """Build HTML combined duty + CBAM cost section."""
        costs = data.get("duty_cbam_costs", [])
        cur = self._currency()

        rows_html = ""
        total_duty = 0.0
        total_cbam = 0.0
        total_combined = 0.0

        for c in costs:
            duty = c.get("customs_duty_eur", 0.0)
            cbam = c.get("cbam_cost_eur", 0.0)
            combined = c.get("combined_cost_eur", 0.0)
            total_duty += duty
            total_cbam += cbam
            total_combined += combined

            cbam_share = c.get("cbam_share_pct", 0.0)
            rows_html += (
                f'<tr><td>{c.get("category", "")}</td>'
                f'<td class="num">{self._fmt_cur(duty, cur)}</td>'
                f'<td class="num">{c.get("customs_duty_rate_pct", 0.0):.1f}%</td>'
                f'<td class="num">{self._fmt_cur(cbam, cur)}</td>'
                f'<td class="num"><strong>{self._fmt_cur(combined, cur)}</strong></td>'
                f'<td class="num">{cbam_share:.1f}%</td></tr>'
            )

        if rows_html:
            cbam_total_share = (total_cbam / max(total_combined, 1)) * 100
            rows_html += (
                f'<tr style="font-weight:bold;background:#eef2f7">'
                f'<td>TOTAL</td>'
                f'<td class="num">{self._fmt_cur(total_duty, cur)}</td><td></td>'
                f'<td class="num">{self._fmt_cur(total_cbam, cur)}</td>'
                f'<td class="num">{self._fmt_cur(total_combined, cur)}</td>'
                f'<td class="num">{cbam_total_share:.1f}%</td></tr>'
            )
        else:
            rows_html = '<tr><td colspan="6"><em>No cost data</em></td></tr>'

        return (
            '<div class="section"><h2>6. Duty + CBAM Cost</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Duty</th><th>Rate</th>'
            '<th>CBAM Cost</th><th>Combined</th><th>CBAM Share</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_sad_reconciliation(self, data: Dict[str, Any]) -> str:
        """Build HTML SAD reconciliation section."""
        records = data.get("sad_reconciliations", [])

        status_colors = {"matched": "#2ecc71", "partial": "#f39c12", "unmatched": "#e74c3c"}

        rows_html = ""
        for r in records:
            status = r.get("match_status", "matched")
            color = status_colors.get(status, "#95a5a6")
            variance = r.get("variance_tonnes", 0.0)
            var_pct = r.get("variance_pct", 0.0)
            var_str = f"{variance:+.2f}t ({var_pct:+.1f}%)" if variance != 0 else "-"

            rows_html += (
                f'<tr><td>{r.get("sad_reference", "")}</td>'
                f'<td>{r.get("cbam_import_id", "")}</td>'
                f'<td style="color:{color};font-weight:bold">{status.upper()}</td>'
                f'<td class="num">{self._fmt_num(r.get("volume_sad_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_num(r.get("volume_cbam_tonnes", 0.0))}</td>'
                f'<td class="num">{var_str}</td>'
                f'<td>{r.get("notes", "") or "-"}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No records</em></td></tr>'

        matched = sum(1 for r in records if r.get("match_status") == "matched")
        partial = sum(1 for r in records if r.get("match_status") == "partial")
        unmatched = sum(1 for r in records if r.get("match_status") == "unmatched")

        summary = (
            f'<div style="margin-top:12px;display:flex;gap:24px">'
            f'<div style="color:#2ecc71"><strong>Matched:</strong> {matched}</div>'
            f'<div style="color:#f39c12"><strong>Partial:</strong> {partial}</div>'
            f'<div style="color:#e74c3c"><strong>Unmatched:</strong> {unmatched}</div>'
            f'</div>'
        )

        return (
            '<div class="section"><h2>7. SAD Reconciliation</h2>'
            '<table><thead><tr>'
            '<th>SAD Ref</th><th>CBAM ID</th><th>Status</th>'
            '<th>SAD Vol (t)</th><th>CBAM Vol (t)</th><th>Variance</th><th>Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>{summary}</div>'
        )

    def _html_eori_validation(self, data: Dict[str, Any]) -> str:
        """Build HTML EORI validation section."""
        records = data.get("eori_validations", [])

        rows_html = ""
        for r in records:
            valid = r.get("valid", True)
            color = "#2ecc71" if valid else "#e74c3c"
            issues = r.get("issues", "") or "-"
            rows_html += (
                f'<tr><td><code>{r.get("eori_number", "")}</code></td>'
                f'<td>{r.get("entity_name", "")}</td>'
                f'<td style="color:{color};font-weight:bold">'
                f'{"VALID" if valid else "INVALID"}</td>'
                f'<td>{r.get("member_state", "")}</td>'
                f'<td>{self._fmt_date(r.get("validation_date", ""))}</td>'
                f'<td>{issues}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No EORI records</em></td></tr>'

        invalid_count = sum(1 for r in records if not r.get("valid", True))
        alert = ""
        if invalid_count > 0:
            alert = (
                f'<div style="margin-top:12px;padding:12px;background:#fdf2f2;'
                f'border-left:4px solid #e74c3c;border-radius:4px">'
                f'<strong>WARNING:</strong> {invalid_count} EORI number(s) failed '
                f'validation. Immediate action required.</div>'
            )

        return (
            '<div class="section"><h2>8. EORI Validation</h2>'
            '<table><thead><tr>'
            '<th>EORI Number</th><th>Entity</th><th>Valid</th>'
            '<th>Member State</th><th>Last Validated</th><th>Issues</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>{alert}</div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_import_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON import summary."""
        ims = data.get("import_summary", {})
        return {
            "reporting_period": ims.get("reporting_period", ""),
            "total_imports": ims.get("total_imports", 0),
            "total_volume_tonnes": round(ims.get("total_volume_tonnes", 0.0), 2),
            "total_value_eur": round(ims.get("total_value_eur", 0.0), 2),
            "unique_cn_codes": ims.get("unique_cn_codes", 0),
            "unique_countries": ims.get("unique_countries", 0),
            "unique_customs_offices": ims.get("unique_customs_offices", 0),
            "imports_by_code": [
                {
                    "cn_code": i.get("cn_code", ""),
                    "description": i.get("description", ""),
                    "country_of_origin": i.get("country_of_origin", ""),
                    "customs_office": i.get("customs_office", ""),
                    "volume_tonnes": round(i.get("volume_tonnes", 0.0), 2),
                    "value_eur": round(i.get("value_eur", 0.0), 2),
                    "num_shipments": i.get("num_shipments", 0),
                }
                for i in ims.get("imports_by_code", [])
            ],
        }

    def _json_applicability(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON CBAM applicability."""
        return [
            {
                "cn_code": r.get("cn_code", ""),
                "description": r.get("description", ""),
                "cbam_covered": r.get("cbam_covered", False),
                "cbam_sector": r.get("cbam_sector", ""),
                "exemption_reason": r.get("exemption_reason", ""),
                "volume_tonnes": round(r.get("volume_tonnes", 0.0), 2),
            }
            for r in data.get("cbam_applicability", [])
        ]

    def _json_cn_validations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON CN code validations."""
        return [
            {
                "cn_code": v.get("cn_code", ""),
                "taric_valid": v.get("taric_valid", True),
                "taric_description": v.get("taric_description", ""),
                "cbam_annex_match": v.get("cbam_annex_match", True),
                "mismatch_details": v.get("mismatch_details", ""),
                "warning": v.get("warning", ""),
            }
            for v in data.get("cn_code_validations", [])
        ]

    def _json_circumvention(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON circumvention flags."""
        return [
            {
                "flag_id": f.get("flag_id", ""),
                "pattern_type": f.get("pattern_type", ""),
                "description": f.get("description", ""),
                "severity": f.get("severity", "low"),
                "cn_codes_involved": f.get("cn_codes_involved", []),
                "countries_involved": f.get("countries_involved", []),
                "evidence": f.get("evidence", ""),
                "recommended_action": f.get("recommended_action", ""),
            }
            for f in data.get("circumvention_flags", [])
        ]

    def _json_procedures(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON customs procedures."""
        return [
            {
                "procedure_code": p.get("procedure_code", ""),
                "procedure_name": p.get("procedure_name", ""),
                "volume_tonnes": round(p.get("volume_tonnes", 0.0), 2),
                "value_eur": round(p.get("value_eur", 0.0), 2),
                "shipment_count": p.get("shipment_count", 0),
                "cbam_applicable": p.get("cbam_applicable", True),
                "notes": p.get("notes", ""),
            }
            for p in data.get("customs_procedures", [])
        ]

    def _json_duty_cbam(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON duty + CBAM costs."""
        return [
            {
                "category": c.get("category", ""),
                "customs_duty_eur": round(c.get("customs_duty_eur", 0.0), 2),
                "customs_duty_rate_pct": round(c.get("customs_duty_rate_pct", 0.0), 2),
                "cbam_cost_eur": round(c.get("cbam_cost_eur", 0.0), 2),
                "combined_cost_eur": round(c.get("combined_cost_eur", 0.0), 2),
                "cbam_share_pct": round(c.get("cbam_share_pct", 0.0), 2),
                "volume_tonnes": round(c.get("volume_tonnes", 0.0), 2),
            }
            for c in data.get("duty_cbam_costs", [])
        ]

    def _json_sad(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON SAD reconciliations."""
        return [
            {
                "sad_reference": r.get("sad_reference", ""),
                "cbam_import_id": r.get("cbam_import_id", ""),
                "match_status": r.get("match_status", "matched"),
                "volume_sad_tonnes": round(r.get("volume_sad_tonnes", 0.0), 2),
                "volume_cbam_tonnes": round(r.get("volume_cbam_tonnes", 0.0), 2),
                "variance_tonnes": round(r.get("variance_tonnes", 0.0), 2),
                "variance_pct": round(r.get("variance_pct", 0.0), 2),
                "notes": r.get("notes", ""),
            }
            for r in data.get("sad_reconciliations", [])
        ]

    def _json_eori(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON EORI validations."""
        return [
            {
                "eori_number": r.get("eori_number", ""),
                "entity_name": r.get("entity_name", ""),
                "valid": r.get("valid", True),
                "member_state": r.get("member_state", ""),
                "validation_date": r.get("validation_date", ""),
                "issues": r.get("issues", ""),
            }
            for r in data.get("eori_validations", [])
        ]

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

    def _fmt_cn(self, code: str) -> str:
        """Format CN code to standard format."""
        clean = code.replace(".", "").replace(" ", "")
        if len(clean) >= 6:
            return f"{clean[:4]}.{clean[4:6]}"
        elif len(clean) == 4:
            return f"{clean}.00"
        return code

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
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
