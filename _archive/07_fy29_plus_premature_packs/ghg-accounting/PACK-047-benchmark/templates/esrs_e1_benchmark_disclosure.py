# -*- coding: utf-8 -*-
"""
ESRSE1BenchmarkDisclosure - ESRS E1-4 Benchmark Disclosure for PACK-047.

Generates an ESRS E1-4 compliant benchmark context disclosure with sector
comparison metrics per ESRS topical standards, EU Taxonomy alignment
benchmark data, XBRL tag mapping for ESRS benchmark fields, and
cross-reference to PACK-046 intensity disclosures.

ESRS References:
    - ESRS E1: Climate Change (topical standard)
    - ESRS E1-4: Targets related to climate change mitigation and adaptation
    - ESRS 1 paragraph 43: Benchmark disclosure context
    - EU Taxonomy Regulation 2020/852: Alignment benchmarks
    - EFRAG XBRL taxonomy for ESRS digital reporting

Sections:
    1. ESRS E1-4 Benchmark Context
    2. Sector Comparison Metrics
    3. EU Taxonomy Alignment Benchmark
    4. XBRL Tag Mapping
    5. Cross-Reference to Intensity Disclosures
    6. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured disclosure data)
    - XBRL (tagged disclosure elements)

Author: GreenLang Team
Version: 47.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    XBRL = "xbrl"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class BenchmarkContextParagraph(BaseModel):
    """Generated benchmark context paragraph for ESRS E1-4."""
    paragraph_id: str = Field("", description="Paragraph identifier")
    text: str = Field("", description="Paragraph text")
    data_references: List[str] = Field(
        default_factory=list, description="Referenced data points"
    )

class SectorComparisonMetric(BaseModel):
    """Sector comparison metric for ESRS topical standards."""
    metric_name: str = Field(..., description="Metric name")
    org_value: float = Field(0.0, description="Organisation value")
    sector_average: Optional[float] = Field(None, description="Sector average")
    sector_median: Optional[float] = Field(None, description="Sector median")
    sector_best: Optional[float] = Field(None, description="Sector best-in-class")
    unit: str = Field("", description="Metric unit")
    esrs_reference: str = Field("", description="ESRS data point reference")
    nace_code: str = Field("", description="NACE sector code")
    gap_to_average_pct: Optional[float] = Field(None, description="Gap to average (%)")
    reporting_year: int = Field(0, description="Reporting year")

class TaxonomyAlignmentBenchmark(BaseModel):
    """EU Taxonomy alignment benchmark entry."""
    activity_name: str = Field(..., description="Economic activity name")
    nace_code: str = Field("", description="NACE code")
    org_alignment_pct: float = Field(0.0, description="Organisation alignment (%)")
    sector_avg_alignment_pct: Optional[float] = Field(
        None, description="Sector average alignment (%)"
    )
    sector_best_alignment_pct: Optional[float] = Field(
        None, description="Sector best alignment (%)"
    )
    taxonomy_objective: str = Field(
        "climate_mitigation", description="Taxonomy environmental objective"
    )
    eligible_pct: float = Field(0.0, description="Eligibility (%)")
    substantial_contribution: bool = Field(
        False, description="Substantial contribution criteria met"
    )
    dnsh_compliant: bool = Field(False, description="DNSH criteria met")

class XBRLTagMapping(BaseModel):
    """XBRL tag mapping for ESRS benchmark fields."""
    field_name: str = Field(..., description="Field name")
    xbrl_element: str = Field("", description="XBRL element name")
    xbrl_namespace: str = Field(
        "esrs", description="XBRL namespace prefix"
    )
    data_type: str = Field("", description="XBRL data type")
    value: Optional[str] = Field(None, description="Tagged value")
    period_type: str = Field("duration", description="Period type (instant/duration)")

class IntensityCrossReference(BaseModel):
    """Cross-reference to PACK-046 intensity disclosures."""
    disclosure_reference: str = Field("", description="PACK-046 disclosure reference")
    metric_name: str = Field("", description="Intensity metric name")
    value: Optional[float] = Field(None, description="Intensity value")
    unit: str = Field("", description="Unit")
    benchmark_percentile: Optional[float] = Field(
        None, description="Benchmark percentile for this metric"
    )
    link_description: str = Field("", description="Link description")

class ESRSE1BenchmarkInput(BaseModel):
    """Complete input model for ESRSE1BenchmarkDisclosure."""
    company_name: str = Field("Organization", description="Company name")
    reporting_year: int = Field(0, description="Reporting year")
    reporting_period_start: str = Field("", description="Period start (ISO)")
    reporting_period_end: str = Field("", description="Period end (ISO)")
    esrs_reference: str = Field(
        "ESRS E1-4", description="Primary ESRS reference"
    )
    benchmark_context_paragraphs: List[BenchmarkContextParagraph] = Field(
        default_factory=list, description="Generated context paragraphs"
    )
    sector_comparison_metrics: List[SectorComparisonMetric] = Field(
        default_factory=list, description="Sector comparison metrics"
    )
    taxonomy_alignment: List[TaxonomyAlignmentBenchmark] = Field(
        default_factory=list, description="EU Taxonomy alignment"
    )
    xbrl_tag_mappings: List[XBRLTagMapping] = Field(
        default_factory=list, description="XBRL tag mappings"
    )
    intensity_cross_references: List[IntensityCrossReference] = Field(
        default_factory=list, description="Cross-references to PACK-046"
    )
    nace_sector: str = Field("", description="NACE sector classification")
    consolidation_approach: str = Field(
        "operational control", description="Consolidation approach"
    )

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ESRSE1BenchmarkDisclosure:
    """
    ESRS E1-4 benchmark disclosure template.

    Renders ESRS-compliant benchmark context disclosures with sector
    comparison metrics, EU Taxonomy alignment benchmarks, XBRL tag
    mappings, and cross-references to PACK-046 intensity disclosures.
    All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = ESRSE1BenchmarkDisclosure()
        >>> md = template.render_markdown(data)
        >>> xbrl = template.render_xbrl(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSE1BenchmarkDisclosure."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Any:
        """Render in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        elif fmt == "xbrl":
            return self.render_xbrl(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESRS benchmark disclosure as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS benchmark disclosure as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS benchmark disclosure as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_xbrl(self, data: Dict[str, Any]) -> str:
        """Render ESRS benchmark disclosure as XBRL XML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_xbrl(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(data)

    def to_html(self, data: Dict[str, Any]) -> str:
        """Alias for render_html."""
        return self.render_html(data)

    def to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for render_json."""
        return self.render_json(data)

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_benchmark_context(data),
            self._md_sector_comparison(data),
            self._md_taxonomy_alignment(data),
            self._md_xbrl_mapping(data),
            self._md_cross_references(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        ref = self._get_val(data, "esrs_reference", "ESRS E1-4")
        return (
            f"# {ref} Benchmark Disclosure - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_benchmark_context(self, data: Dict[str, Any]) -> str:
        """Render Markdown benchmark context paragraphs."""
        paragraphs = data.get("benchmark_context_paragraphs", [])
        if not paragraphs:
            return ""
        lines = ["## 1. Benchmark Context (ESRS E1-4)", ""]
        for p in paragraphs:
            pid = p.get("paragraph_id", "")
            text = p.get("text", "")
            if pid:
                lines.append(f"**{pid}:** {text}")
            else:
                lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def _md_sector_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector comparison metrics."""
        metrics = data.get("sector_comparison_metrics", [])
        if not metrics:
            return ""
        lines = [
            "## 2. Sector Comparison Metrics",
            "",
            "| Metric | Org Value | Sector Avg | Sector Median | Best | Gap to Avg | ESRS Ref |",
            "|--------|-----------|-----------|---------------|------|------------|----------|",
        ]
        for m in metrics:
            avg = m.get("sector_average")
            med = m.get("sector_median")
            best = m.get("sector_best")
            gap = m.get("gap_to_average_pct")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            med_str = f"{med:,.2f}" if med is not None else "-"
            best_str = f"{best:,.2f}" if best is not None else "-"
            gap_str = f"{gap:+.1f}%" if gap is not None else "-"
            lines.append(
                f"| {m.get('metric_name', '')} | {m.get('org_value', 0):,.2f} | "
                f"{avg_str} | {med_str} | {best_str} | {gap_str} | "
                f"{m.get('esrs_reference', '')} |"
            )
        return "\n".join(lines)

    def _md_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Render Markdown EU Taxonomy alignment benchmark."""
        entries = data.get("taxonomy_alignment", [])
        if not entries:
            return ""
        lines = [
            "## 3. EU Taxonomy Alignment Benchmark",
            "",
            "| Activity | NACE | Org Alignment | Sector Avg | Sector Best | Objective | SC | DNSH |",
            "|----------|------|--------------|-----------|-------------|-----------|----|----- |",
        ]
        for e in entries:
            s_avg = e.get("sector_avg_alignment_pct")
            s_best = e.get("sector_best_alignment_pct")
            s_avg_str = f"{s_avg:.1f}%" if s_avg is not None else "-"
            s_best_str = f"{s_best:.1f}%" if s_best is not None else "-"
            sc = "Yes" if e.get("substantial_contribution", False) else "No"
            dnsh = "Yes" if e.get("dnsh_compliant", False) else "No"
            lines.append(
                f"| {e.get('activity_name', '')} | {e.get('nace_code', '')} | "
                f"{e.get('org_alignment_pct', 0):.1f}% | {s_avg_str} | {s_best_str} | "
                f"{e.get('taxonomy_objective', '')} | {sc} | {dnsh} |"
            )
        return "\n".join(lines)

    def _md_xbrl_mapping(self, data: Dict[str, Any]) -> str:
        """Render Markdown XBRL tag mapping table."""
        tags = data.get("xbrl_tag_mappings", [])
        if not tags:
            return ""
        lines = [
            "## 4. XBRL Tag Mapping",
            "",
            "| Field | XBRL Element | Namespace | Type | Period |",
            "|-------|-------------|-----------|------|--------|",
        ]
        for t in tags:
            lines.append(
                f"| {t.get('field_name', '')} | {t.get('xbrl_element', '')} | "
                f"{t.get('xbrl_namespace', '')} | {t.get('data_type', '')} | "
                f"{t.get('period_type', '')} |"
            )
        return "\n".join(lines)

    def _md_cross_references(self, data: Dict[str, Any]) -> str:
        """Render Markdown cross-references to PACK-046."""
        refs = data.get("intensity_cross_references", [])
        if not refs:
            return ""
        lines = [
            "## 5. Cross-Reference to Intensity Disclosures (PACK-046)",
            "",
            "| Reference | Metric | Value | Unit | Benchmark Percentile |",
            "|-----------|--------|-------|------|---------------------|",
        ]
        for r in refs:
            val = r.get("value")
            val_str = f"{val:,.4f}" if val is not None else "-"
            pctile = r.get("benchmark_percentile")
            pctile_str = f"P{pctile:.0f}" if pctile is not None else "-"
            lines.append(
                f"| {r.get('disclosure_reference', '')} | {r.get('metric_name', '')} | "
                f"{val_str} | {r.get('unit', '')} | {pctile_str} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_benchmark_context(data),
            self._html_sector_comparison(data),
            self._html_taxonomy_alignment(data),
            self._html_cross_references(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>ESRS E1-4 Benchmark - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #003399;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#e8eef6;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".esrs-ref{background:#f0f4f8;border-left:4px solid #003399;"
            "padding:1rem 1.5rem;margin:1rem 0;font-size:0.9rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        ref = self._get_val(data, "esrs_reference", "ESRS E1-4")
        return (
            '<div class="section">\n'
            f"<h1>{ref}: Benchmark Disclosure &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n<hr>\n</div>"
        )

    def _html_benchmark_context(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark context paragraphs."""
        paragraphs = data.get("benchmark_context_paragraphs", [])
        if not paragraphs:
            return ""
        content = ""
        for p in paragraphs:
            pid = p.get("paragraph_id", "")
            text = p.get("text", "")
            label = f"<strong>{pid}:</strong> " if pid else ""
            content += f'<div class="esrs-ref">{label}{text}</div>\n'
        return (
            '<div class="section">\n<h2>1. Benchmark Context</h2>\n'
            f"{content}</div>"
        )

    def _html_sector_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML sector comparison table."""
        metrics = data.get("sector_comparison_metrics", [])
        if not metrics:
            return ""
        rows = ""
        for m in metrics:
            avg = m.get("sector_average")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            gap = m.get("gap_to_average_pct")
            gap_str = f"{gap:+.1f}%" if gap is not None else "-"
            rows += (
                f"<tr><td>{m.get('metric_name', '')}</td>"
                f"<td>{m.get('org_value', 0):,.2f}</td>"
                f"<td>{avg_str}</td>"
                f"<td>{gap_str}</td>"
                f"<td>{m.get('esrs_reference', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Sector Comparison</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Org Value</th>"
            "<th>Sector Avg</th><th>Gap</th><th>ESRS Ref</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Render HTML Taxonomy alignment table."""
        entries = data.get("taxonomy_alignment", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            s_avg = e.get("sector_avg_alignment_pct")
            s_avg_str = f"{s_avg:.1f}%" if s_avg is not None else "-"
            sc = "Yes" if e.get("substantial_contribution", False) else "No"
            dnsh = "Yes" if e.get("dnsh_compliant", False) else "No"
            rows += (
                f"<tr><td>{e.get('activity_name', '')}</td>"
                f"<td>{e.get('nace_code', '')}</td>"
                f"<td>{e.get('org_alignment_pct', 0):.1f}%</td>"
                f"<td>{s_avg_str}</td>"
                f"<td>{sc}</td><td>{dnsh}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. EU Taxonomy Alignment</h2>\n'
            "<table><thead><tr><th>Activity</th><th>NACE</th>"
            "<th>Org Alignment</th><th>Sector Avg</th>"
            "<th>SC</th><th>DNSH</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_cross_references(self, data: Dict[str, Any]) -> str:
        """Render HTML cross-references to PACK-046."""
        refs = data.get("intensity_cross_references", [])
        if not refs:
            return ""
        rows = ""
        for r in refs:
            val = r.get("value")
            val_str = f"{val:,.4f}" if val is not None else "-"
            pctile = r.get("benchmark_percentile")
            pctile_str = f"P{pctile:.0f}" if pctile is not None else "-"
            rows += (
                f"<tr><td>{r.get('disclosure_reference', '')}</td>"
                f"<td>{r.get('metric_name', '')}</td>"
                f"<td>{val_str}</td>"
                f"<td>{pctile_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Cross-Reference (PACK-046)</h2>\n'
            "<table><thead><tr><th>Reference</th><th>Metric</th>"
            "<th>Value</th><th>Benchmark</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # XBRL RENDERING
    # ==================================================================

    def _render_xbrl(self, data: Dict[str, Any]) -> str:
        """Render ESRS benchmark disclosure as XBRL-tagged XML."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        period_start = self._get_val(data, "reporting_period_start", f"{year}-01-01")
        period_end = self._get_val(data, "reporting_period_end", f"{year}-12-31")
        provenance = self._compute_provenance(data)

        # Sector comparison facts
        sector_facts = ""
        for m in data.get("sector_comparison_metrics", []):
            element = m.get("esrs_reference", "").replace(" ", "").replace("-", "")
            if not element:
                element = m.get("metric_name", "").replace(" ", "")
            org_val = m.get("org_value", 0)
            sector_facts += (
                f'    <esrs:BenchmarkOrgValue_{element} '
                f'contextRef="c_{year}" decimals="2">{org_val}'
                f'</esrs:BenchmarkOrgValue_{element}>\n'
            )
            s_avg = m.get("sector_average")
            if s_avg is not None:
                sector_facts += (
                    f'    <esrs:BenchmarkSectorAvg_{element} '
                    f'contextRef="c_{year}" decimals="2">{s_avg}'
                    f'</esrs:BenchmarkSectorAvg_{element}>\n'
                )

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"\n'
            '    xmlns:esrs="http://xbrl.efrag.org/taxonomy/esrs/2024"\n'
            '    xmlns:link="http://www.xbrl.org/2003/linkbase">\n\n'
            '  <!-- Context -->\n'
            f'  <xbrli:context id="c_{year}">\n'
            f'    <xbrli:entity>\n'
            f'      <xbrli:identifier scheme="http://greenlang.io">{company}</xbrli:identifier>\n'
            f'    </xbrli:entity>\n'
            f'    <xbrli:period>\n'
            f'      <xbrli:startDate>{period_start}</xbrli:startDate>\n'
            f'      <xbrli:endDate>{period_end}</xbrli:endDate>\n'
            f'    </xbrli:period>\n'
            f'  </xbrli:context>\n\n'
            '  <!-- ESRS E1-4 Benchmark Facts -->\n'
            f'{sector_facts}\n'
            f'  <!-- Provenance: {provenance} -->\n\n'
            '</xbrli:xbrl>\n'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS benchmark disclosure as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "esrs_e1_benchmark_disclosure",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "esrs_reference": self._get_val(data, "esrs_reference", "ESRS E1-4"),
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year", ""),
            "benchmark_context_paragraphs": data.get("benchmark_context_paragraphs", []),
            "sector_comparison_metrics": data.get("sector_comparison_metrics", []),
            "taxonomy_alignment": data.get("taxonomy_alignment", []),
            "xbrl_tag_mappings": data.get("xbrl_tag_mappings", []),
            "intensity_cross_references": data.get("intensity_cross_references", []),
        }
