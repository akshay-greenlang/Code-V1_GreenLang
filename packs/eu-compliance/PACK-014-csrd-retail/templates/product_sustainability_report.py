# -*- coding: utf-8 -*-
"""
ProductSustainabilityReportTemplate - Product sustainability report for PACK-014.

Renders product environmental footprint (PEF) results, Digital Product
Passport (DPP) readiness, ECGT green claims audit, product scoring,
and lifecycle impact summaries.

Example:
    >>> template = ProductSustainabilityReportTemplate()
    >>> data = {"products": [...], "claims": [...]}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProductSustainabilityReportTemplate:
    """
    Product sustainability report template for retail.

    Renders PEF lifecycle results, DPP readiness status, green claims
    audit with ECGT compliance, product sustainability scoring, and
    category-level benchmarking.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    LIFECYCLE_STAGES: List[str] = [
        "Raw Materials", "Manufacturing", "Transport",
        "Retail", "Use Phase", "End of Life",
    ]

    CLAIM_STATUS_COLORS: Dict[str, str] = {
        "verified": "#059669",
        "pending": "#d97706",
        "rejected": "#dc2626",
        "expired": "#6b7280",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProductSustainabilityReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render product sustainability report as Markdown.

        Args:
            data: Report data with products, claims, pef_results,
                  dpp_status, category_benchmarks.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_executive_summary(data))
        sections.append(self._md_product_scores(data))
        sections.append(self._md_pef_results(data))
        sections.append(self._md_dpp_readiness(data))
        sections.append(self._md_green_claims_audit(data))
        sections.append(self._md_lifecycle_impact(data))
        sections.append(self._md_category_benchmark(data))
        sections.append(self._md_provenance(data))

        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render product sustainability report as HTML.

        Args:
            data: Report data dict.

        Returns:
            Complete HTML string.
        """
        self.generated_at = datetime.utcnow()
        md = self.render_markdown(data)

        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            f"<title>Product Sustainability Report - {company} - {period}</title>",
            '<meta charset="utf-8">',
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; }",
            "h1 { color: #065f46; border-bottom: 3px solid #065f46; padding-bottom: 8px; }",
            "h2 { color: #047857; margin-top: 32px; }",
            "h3 { color: #059669; }",
            "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
            "th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }",
            "th { background: #f0fdf4; font-weight: 600; }",
            "tr:nth-child(even) { background: #f9fafb; }",
            ".verified { color: #059669; font-weight: 600; }",
            ".pending { color: #d97706; }",
            ".rejected { color: #dc2626; font-weight: 600; }",
            ".provenance { background: #f3f4f6; padding: 12px; border-radius: 6px; font-size: 11px; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        for line in md.split("\n"):
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_parts.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("| "):
                html_parts.append(self._md_table_row_to_html(line))
            elif line.startswith("- "):
                html_parts.append(f"<li>{line[2:]}</li>")
            elif line.strip():
                html_parts.append(f"<p>{line}</p>")

        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render product sustainability report as JSON.

        Args:
            data: Report data dict.

        Returns:
            Pretty-printed JSON string.
        """
        self.generated_at = datetime.utcnow()
        provenance_hash = self._compute_hash(data)

        output = {
            "template": "product_sustainability_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance_hash,
            "data": data,
        }
        return json.dumps(output, indent=2, default=str)

    # ------------------------------------------------------------------
    # Private markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render report header."""
        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")
        return (
            f"# Product Sustainability Report\n\n"
            f"**Company:** {company}  \n"
            f"**Period:** {period}  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 Product Sustainability v14.0.0"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        products = data.get("products", [])
        claims = data.get("claims", [])
        verified = sum(1 for c in claims if c.get("status") == "verified")

        return (
            f"## Executive Summary\n\n"
            f"- **Products Assessed:** {len(products)}\n"
            f"- **Average Sustainability Score:** {data.get('avg_score', 0):.1f}/100\n"
            f"- **Green Claims Filed:** {len(claims)}\n"
            f"- **Claims Verified:** {verified} ({(verified / len(claims) * 100) if claims else 0:.0f}%)\n"
            f"- **DPP Ready Products:** {data.get('dpp_ready_count', 0)}"
        )

    def _md_product_scores(self, data: Dict[str, Any]) -> str:
        """Render product sustainability scores."""
        products = data.get("products", [])
        if not products:
            return "## Product Scores\n\nNo product data available."

        lines = [
            "## Product Sustainability Scores\n",
            "| Product | Category | Score | PEF Class | DPP Ready | Claims |",
            "|---------|----------|-------|-----------|-----------|--------|",
        ]
        for p in products:
            lines.append(
                f"| {p.get('name', 'N/A')} | {p.get('category', 'N/A')} "
                f"| {p.get('score', 0):.0f} | {p.get('pef_class', 'N/A')} "
                f"| {'Yes' if p.get('dpp_ready') else 'No'} "
                f"| {p.get('claim_count', 0)} |"
            )
        return "\n".join(lines)

    def _md_pef_results(self, data: Dict[str, Any]) -> str:
        """Render PEF lifecycle assessment results."""
        pef = data.get("pef_results", [])
        if not pef:
            return "## PEF Results\n\nNo PEF data available."

        lines = [
            "## Product Environmental Footprint (PEF)\n",
            "| Product | Climate (kgCO2e) | Water (m3) | Land Use (m2a) | Resource Use (kgSb) | PEF Class |",
            "|---------|------------------|------------|----------------|---------------------|-----------|",
        ]
        for p in pef:
            lines.append(
                f"| {p.get('product', 'N/A')} | {p.get('climate_kgco2e', 0):.2f} "
                f"| {p.get('water_m3', 0):.3f} | {p.get('land_use_m2a', 0):.2f} "
                f"| {p.get('resource_kgsb', 0):.4f} | {p.get('pef_class', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_dpp_readiness(self, data: Dict[str, Any]) -> str:
        """Render Digital Product Passport readiness status."""
        dpp = data.get("dpp_status", [])
        if not dpp:
            return "## DPP Readiness\n\nNo DPP data available."

        lines = [
            "## Digital Product Passport (ESPR) Readiness\n",
            "| Product | Data Completeness | QR Code | Lifecycle Data | Recyclability | Status |",
            "|---------|-------------------|---------|----------------|---------------|--------|",
        ]
        for d in dpp:
            lines.append(
                f"| {d.get('product', 'N/A')} | {d.get('completeness_pct', 0):.0f}% "
                f"| {'Ready' if d.get('qr_code') else 'Missing'} "
                f"| {'Yes' if d.get('lifecycle_data') else 'No'} "
                f"| {'Yes' if d.get('recyclability_info') else 'No'} "
                f"| {d.get('status', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_green_claims_audit(self, data: Dict[str, Any]) -> str:
        """Render ECGT green claims audit results."""
        claims = data.get("claims", [])
        if not claims:
            return "## Green Claims Audit\n\nNo claims filed."

        lines = [
            "## Green Claims Audit (ECGT)\n",
            "| Claim | Product | Status | Evidence | Expiry | Risk |",
            "|-------|---------|--------|----------|--------|------|",
        ]
        for c in claims:
            lines.append(
                f"| {c.get('claim', 'N/A')} | {c.get('product', 'N/A')} "
                f"| {c.get('status', 'N/A')} | {c.get('evidence_count', 0)} docs "
                f"| {c.get('expiry', 'N/A')} | {c.get('risk', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_lifecycle_impact(self, data: Dict[str, Any]) -> str:
        """Render lifecycle impact summary."""
        lifecycle = data.get("lifecycle_summary", {})
        if not lifecycle:
            return "## Lifecycle Impact\n\nNo lifecycle data available."

        lines = ["## Lifecycle Impact Summary\n"]
        for stage in self.LIFECYCLE_STAGES:
            key = stage.lower().replace(" ", "_")
            pct = lifecycle.get(key, {}).get("pct_of_total", 0)
            tco2e = lifecycle.get(key, {}).get("tco2e", 0)
            bar_len = min(int(pct / 2), 40)
            bar = "#" * bar_len
            lines.append(f"- **{stage}:** {tco2e:,.1f} tCO2e ({pct:.1f}%) {bar}")
        return "\n".join(lines)

    def _md_category_benchmark(self, data: Dict[str, Any]) -> str:
        """Render category-level benchmarks."""
        benchmarks = data.get("category_benchmarks", [])
        if not benchmarks:
            return "## Category Benchmarks\n\nNo benchmark data available."

        lines = [
            "## Category Benchmarks\n",
            "| Category | Your Score | Sector Avg | Percentile | Best in Class |",
            "|----------|------------|------------|------------|---------------|",
        ]
        for b in benchmarks:
            lines.append(
                f"| {b.get('category', 'N/A')} | {b.get('score', 0):.0f} "
                f"| {b.get('sector_avg', 0):.0f} | P{b.get('percentile', 0):.0f} "
                f"| {b.get('best_in_class', 0):.0f} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance footer."""
        h = self._compute_hash(data)
        return (
            f"---\n\n"
            f"**Provenance:** SHA-256 `{h}`  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 ProductSustainabilityReportTemplate v14.0.0"
        )

    @staticmethod
    def _compute_hash(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data for provenance."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _md_table_row_to_html(line: str) -> str:
        """Convert a Markdown table row to HTML."""
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(c.startswith("-") for c in cells):
            return ""
        row = "".join(f"<td>{c}</td>" for c in cells)
        return f"<tr>{row}</tr>"
